#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import torch
from tqdm import tqdm


@dataclass(frozen=True)
class FrameInput:
    frame_idx: int
    rgb_path: Path
    image: np.ndarray
    annotation: dict
    rtmlib_hands: list[dict]


def patch_legacy_imports() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    raw_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return raw_load(*args, **kwargs)

    torch.load = torch_load_compat  # type: ignore[assignment]


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def project_points(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = np.clip(points_camera_m[:, 2], 1e-6, None)
    return np.c_[fx * points_camera_m[:, 0] / z + cx, fy * points_camera_m[:, 1] / z + cy]


def solve_source_camera_translation(local_points_m: np.ndarray, points2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    qx = (points2d[:, 0] - cx) / fx
    qy = (points2d[:, 1] - cy) / fy
    rows = []
    rhs = []
    for (x, y, z), u, v in zip(local_points_m, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float), rcond=None)
    return trans.astype(float)


def hand_bone_scale_m(joints: np.ndarray) -> float:
    if joints.shape != (21, 3):
        return float("nan")
    chains = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    lengths = []
    for chain in chains:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[np.asarray(points, dtype=float), np.ones(len(points), dtype=float)]
    return (np.asarray(transform, dtype=float) @ homog.T).T[:, :3]


def transform_for(frame: dict) -> np.ndarray:
    transform = np.asarray(frame.get("camera", {}).get("T_world_camera_metric", []), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} missing valid T_world_camera_metric")
    return transform


def transform_hand_to_world(hand: dict, T_world_camera: np.ndarray) -> None:
    joints = np.asarray(hand["joints3d_source_camera_m"], dtype=np.float64)
    hand["joints3d_world_m"] = transform_points(joints, T_world_camera).astype(float).tolist()
    vertices = np.asarray(hand["vertices_source_camera_m"], dtype=np.float64)
    hand["vertices_world_m"] = transform_points(vertices, T_world_camera).astype(float).tolist()
    hand["world_coordinate_status"] = "v3_vggt_native_metric_depth_scaled_local_world"


def localize_path(raw: str, local_root: Path | None, remote_root: Path | None) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    if local_root is not None and remote_root is not None:
        try:
            rel = Path(raw).relative_to(local_root)
        except ValueError:
            rel = None
        if rel is not None:
            candidate = remote_root / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"cannot resolve path {raw}")


def frame_manifest(manifest_path: Path, local_root: Path | None, remote_root: Path | None) -> dict[int, Path]:
    payload = load_json(manifest_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{manifest_path} has no frames")
    out: dict[int, Path] = {}
    for row in frames:
        frame_idx = int(row["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate frame_idx {frame_idx} in {manifest_path}")
        out[frame_idx] = localize_path(str(row["rgb"]), local_root, remote_root)
    return out


def load_annotations(path: Path, frame_start: int, frame_end: int) -> dict[int, dict]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} has no frames")
    out = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_start <= frame_idx <= frame_end:
            if frame_idx in out:
                raise RuntimeError(f"duplicate annotation frame {frame_idx}")
            out[frame_idx] = copy.deepcopy(frame)
    missing = [idx for idx in range(frame_start, frame_end + 1) if idx not in out]
    if missing:
        raise RuntimeError(f"target annotations missing frames {missing[:20]}")
    return out


def load_rtmlib(path: Path, frame_start: int, frame_end: int) -> dict[int, list[dict]]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} has no frames")
    out: dict[int, list[dict]] = {}
    for frame in frames:
        frame_idx = int(frame.get("local_frame_idx", frame["frame_idx"]))
        if frame_start <= frame_idx <= frame_end:
            hands = frame.get("hands", [])
            if not isinstance(hands, list):
                raise RuntimeError(f"RTMLib frame {frame_idx} hands is not a list")
            out[frame_idx] = hands
    missing = [idx for idx in range(frame_start, frame_end + 1) if idx not in out]
    if missing:
        raise RuntimeError(f"RTMLib hand evidence missing frames {missing[:20]}")
    return out


def valid_intrinsics(raw: object) -> np.ndarray | None:
    intr = np.asarray(raw, dtype=float)
    if intr.shape == (4,) and np.isfinite(intr).all():
        return intr
    return None


def intrinsics_for_with_source(frame: dict, explicit_intrinsics: object | None = None) -> tuple[np.ndarray, str]:
    candidates = []
    explicit = valid_intrinsics(explicit_intrinsics)
    if explicit is not None:
        candidates.append(("--source-intrinsics-fx-fy-cx-cy", explicit))
    camera = frame.get("camera", {})
    obj = frame.get("object", {})
    for source, raw in (
        ("camera.vggt_source_intrinsics_fx_fy_cx_cy", camera.get("vggt_source_intrinsics_fx_fy_cx_cy")),
        ("object.mesh_qc.source_intrinsics", obj.get("mesh_qc", {}).get("source_intrinsics")),
        ("object.source_intrinsics", obj.get("source_intrinsics")),
    ):
        intr = valid_intrinsics(raw)
        if intr is not None:
            candidates.append((source, intr))
    if not candidates:
        raise RuntimeError(f"frame {frame.get('frame_idx')} missing valid source intrinsics")
    source, intr = candidates[0]
    for other_source, other_intr in candidates[1:]:
        if not np.allclose(intr, other_intr, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"frame {frame.get('frame_idx')} has conflicting source intrinsics: "
                f"{source}={intr.tolist()} vs {other_source}={other_intr.tolist()}"
            )
    return intr, source


def intrinsics_for(frame: dict) -> np.ndarray:
    return intrinsics_for_with_source(frame)[0]


def validate_image_intrinsics_contract(frame: dict, image: np.ndarray, explicit_intrinsics: object | None = None) -> None:
    intr, source = intrinsics_for_with_source(frame, explicit_intrinsics)
    width, height = image.shape[1], image.shape[0]
    cx, cy = float(intr[2]), float(intr[3])
    if abs(cx - width / 2.0) > max(4.0, 0.1 * width) or abs(cy - height / 2.0) > max(4.0, 0.1 * height):
        raise RuntimeError(
            f"frame {frame.get('frame_idx')} RGB image size {(width, height)} is inconsistent with "
            f"{source} principal point {(cx, cy)}"
        )


def load_frame_inputs(args: argparse.Namespace) -> list[FrameInput]:
    rgb_by_frame = frame_manifest(args.frame_manifest, args.local_root, args.remote_root)
    annotations = load_annotations(args.target_annotations, args.frame_start, args.frame_end)
    rtmlib = load_rtmlib(args.rtmlib_json, args.frame_start, args.frame_end)
    inputs: list[FrameInput] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        rgb_path = rgb_by_frame.get(frame_idx)
        if rgb_path is None:
            raise RuntimeError(f"RGB manifest missing source frame {frame_idx}")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image {rgb_path}")
        validate_image_intrinsics_contract(annotations[frame_idx], image, args.source_intrinsics_fx_fy_cx_cy)
        obj_size = np.asarray(annotations[frame_idx].get("object", {}).get("source_image_size", []), dtype=float)
        if obj_size.shape == (2,) and np.any(np.asarray([image.shape[1], image.shape[0]], dtype=float) != obj_size):
            raise RuntimeError(f"frame {frame_idx} RGB size {image.shape[1::-1]} differs from annotation source size {obj_size}")
        inputs.append(
            FrameInput(
                frame_idx=frame_idx,
                rgb_path=rgb_path,
                image=image,
                annotation=annotations[frame_idx],
                rtmlib_hands=rtmlib[frame_idx],
            )
        )
    return inputs


def ensure_hamer_assets(args: argparse.Namespace) -> None:
    required = [
        args.hamer_root / "hamer" / "models" / "hamer.py",
        args.hamer_root / "hamer" / "datasets" / "vitdet_dataset.py",
        args.checkpoint,
        args.hamer_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
        args.hamer_root / "_DATA" / "data" / "mano" / "MANO_LEFT.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing HaMeR assets: " + ", ".join(missing))


def ensure_hamer_path(hamer_root: Path) -> None:
    root = str(hamer_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_hamer_backend(args: argparse.Namespace):
    patch_legacy_imports()
    ensure_hamer_assets(args)
    ensure_hamer_path(args.hamer_root)
    cwd = Path.cwd()
    os.chdir(args.hamer_root)
    try:
        from hamer.models import load_hamer

        model, cfg = load_hamer(str(args.checkpoint))
    finally:
        os.chdir(cwd)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)
    model.eval()
    return model, cfg, device


def valid_rtmlib_candidate(hand: dict, args: argparse.Namespace, source_size: tuple[int, int]) -> tuple[bool, dict]:
    keypoints = np.asarray(hand.get("keypoints", []), dtype=float)
    scores = np.asarray(hand.get("scores", []), dtype=float)
    box = np.asarray(hand.get("bbox_xyxy", []), dtype=float)
    if keypoints.shape != (21, 2) or scores.shape != (21,) or box.shape != (4,):
        return False, {"status": "invalid_shape"}
    finite = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores)
    valid = finite & (scores >= args.min_keypoint_score)
    if int(np.count_nonzero(valid)) < args.min_valid_keypoints:
        return False, {"status": "too_few_scored_keypoints", "valid_keypoints": int(np.count_nonzero(valid))}
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    area_frac = width * height / float(source_size[0] * source_size[1])
    span = np.ptp(keypoints[valid], axis=0)
    diag = float(math.hypot(width, height))
    geom = {
        "status": "candidate_qc",
        "valid_keypoints": int(np.count_nonzero(valid)),
        "mean_score": float(np.mean(scores[valid])),
        "median_score": float(np.median(scores[valid])),
        "bbox_area_fraction": float(area_frac),
        "bbox_width_px": width,
        "bbox_height_px": height,
        "bbox_diag_px": diag,
        "keypoint_span_x_px": float(span[0]),
        "keypoint_span_y_px": float(span[1]),
    }
    checks = [
        geom["mean_score"] >= args.min_mean_score,
        args.min_box_area_fraction <= area_frac <= args.max_box_area_fraction,
        args.min_box_side_px <= width <= args.max_box_side_px,
        args.min_box_side_px <= height <= args.max_box_side_px,
        float(span[0]) <= args.max_keypoint_span_x_px,
        float(span[1]) <= args.max_keypoint_span_y_px,
    ]
    return bool(all(checks)), geom


def proposal_rows(frame: FrameInput, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    boxes: list[np.ndarray] = []
    rights: list[float] = []
    meta: list[dict] = []
    source_size = (frame.image.shape[1], frame.image.shape[0])
    for hand in frame.rtmlib_hands:
        ok, qc = valid_rtmlib_candidate(hand, args, source_size)
        base = {
            "frame_idx": int(frame.frame_idx),
            "rtmlib_hand_idx": int(hand.get("hand_idx", len(meta))),
            "rtmlib_qc": qc,
            "accepted_for_hamer": bool(ok),
        }
        if not ok:
            meta.append(base)
            continue
        keypoints = np.asarray(hand["keypoints"], dtype=float)
        scores = np.asarray(hand["scores"], dtype=float)
        box = np.asarray(hand["bbox_xyxy"], dtype=float)
        valid = np.isfinite(scores) & (scores >= args.min_keypoint_score)
        for side, right_value in (("left", 0.0), ("right", 1.0)):
            boxes.append(box.astype(np.float32))
            rights.append(right_value)
            meta.append(
                {
                    **base,
                    "hypothesis_side": side,
                    "bbox_xyxy": box.astype(float).tolist(),
                    "keypoints": keypoints.astype(float).tolist(),
                    "scores": scores.astype(float).tolist(),
                    "detector_score": float(np.mean(scores[valid])),
                }
            )
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta
    return np.stack(boxes).astype(np.float32), np.asarray(rights, dtype=np.float32), meta


def solve_metric_hand(
    local_joints_m: np.ndarray,
    local_vertices_m: np.ndarray,
    raw2d: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    translation = solve_source_camera_translation(local_joints_m, raw2d, intrinsics)
    source_joints = local_joints_m + translation[None, :]
    source_vertices = local_vertices_m + translation[None, :]
    projected = project_points(source_joints, intrinsics)
    err = np.linalg.norm(projected - raw2d, axis=1)
    depth = source_joints[:, 2]
    metrics = {
        "median_reprojection_error_px": float(np.median(err)),
        "p95_reprojection_error_px": float(np.percentile(err, 95.0)),
        "median_depth_m": float(np.median(depth)),
        "min_depth_m": float(np.min(depth)),
        "max_depth_m": float(np.max(depth)),
        "hand_bone_scale_m": float(hand_bone_scale_m(local_joints_m)),
    }
    return translation, source_joints, source_vertices, metrics


def project_full_image(points: np.ndarray, cam_t: np.ndarray, focal: float, img_size: np.ndarray) -> np.ndarray:
    source_points = points + cam_t[None, :]
    z = np.clip(source_points[:, 2], 1e-6, None)
    cx = float(img_size[0]) * 0.5
    cy = float(img_size[1]) * 0.5
    return np.c_[float(focal) * source_points[:, 0] / z + cx, float(focal) * source_points[:, 1] / z + cy]


def sample_vertices(vertices: np.ndarray, max_vertices: int) -> tuple[np.ndarray, str]:
    if max_vertices <= 0 or len(vertices) <= max_vertices:
        return vertices, "full_mano_vertices"
    stride = int(math.ceil(len(vertices) / max_vertices))
    return vertices[::stride], f"sampled_every_{stride}_vertices"


def mano_params_for_sample(out: dict, n: int) -> dict:
    params = {}
    for key, value in out["pred_mano_params"].items():
        params[key] = value[n].detach().cpu().numpy().astype(float).tolist()
    return params


def run_hamer_on_frame(model, cfg, device, frame: FrameInput, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full

    boxes, rights, meta = proposal_rows(frame, args)
    if boxes.shape[0] == 0:
        return [], meta
    dataset = ViTDetDataset(cfg, frame.image, boxes, rights, rescale_factor=args.rescale_factor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    hands: list[dict] = []
    frame_meta = [m for m in meta if bool(m.get("accepted_for_hamer", False)) and "hypothesis_side" in m]
    pred_i = 0
    intr, intr_source = intrinsics_for_with_source(frame.annotation, args.source_intrinsics_fx_fy_cx_cy)
    T_world_camera = transform_for(frame.annotation)
    for batch in loader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        scaled_focal_np = scaled_focal_length.detach().cpu().numpy()
        img_size_np = img_size.detach().cpu().numpy()
        batch_count = int(batch["img"].shape[0])
        for n in range(batch_count):
            item = frame_meta[pred_i]
            side = str(item["hypothesis_side"])
            side_sign = 1.0 if side == "right" else -1.0
            vertices = out["pred_vertices"][n].detach().cpu().numpy().astype(float)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
            vertices[:, 0] = side_sign * vertices[:, 0]
            joints[:, 0] = side_sign * joints[:, 0]
            if args.measurement_source == "rtmlib-keypoints":
                raw2d = np.asarray(item["keypoints"], dtype=float)
            elif args.measurement_source == "hamer-full-projection":
                raw2d = project_full_image(joints, cam_t_full[n], float(scaled_focal_np), img_size_np[n])
            else:
                raise RuntimeError(f"unsupported measurement source {args.measurement_source}")
            translation, source_joints, source_vertices, metrics = solve_metric_hand(joints, vertices, raw2d, intr)
            sampled_vertices, surface_status = sample_vertices(vertices, args.max_vertices_per_hand)
            sampled_source_vertices = sampled_vertices + translation[None, :]
            projected = project_points(source_joints, intr)
            measurement_available = (
                args.min_depth_m <= metrics["median_depth_m"] <= args.max_depth_m
                and metrics["median_reprojection_error_px"] <= args.max_initial_reprojection_px
                and args.min_hand_bone_m <= metrics["hand_bone_scale_m"] <= args.max_hand_bone_m
            )
            hand = {
                "backend": "HaMeR",
                "side": side,
                "detector_score": float(item["detector_score"]),
                "bbox_xyxy": item["bbox_xyxy"],
                "cam_t": translation.astype(float).tolist(),
                "source_intrinsics": intr.astype(float).tolist(),
                "joints3d_camera": joints.astype(float).tolist(),
                "vertices_camera": sampled_vertices.astype(float).tolist(),
                "joints3d_source_camera_m": source_joints.astype(float).tolist(),
                "vertices_source_camera_m": sampled_source_vertices.astype(float).tolist(),
                "joints2d_raw": raw2d.astype(float).tolist(),
                "joints2d": projected.astype(float).tolist(),
                "mano_params": mano_params_for_sample(out, n),
                "mano_surface_status": surface_status,
                "mano_vertex_count": int(len(vertices)),
                "measurement_available": bool(measurement_available),
                "track_id": f"hamer_rtmlib_{int(item['rtmlib_hand_idx'])}_{side}",
                "track_source": "rtmlib_box_both_side_hamer_metric_translation",
                "filter_status": "measured_source_camera_solve" if measurement_available else "rejected_initial_metric_qc",
                "source_camera_solve": {
                    "status": "least_squares_translation_from_hamer_local_geometry_and_rtmlib_2d_keypoints",
                    "measurement_source": args.measurement_source,
                    "source_intrinsics_field": intr_source,
                    "hamer_virtual_focal_length": float(scaled_focal_np),
                    "hamer_virtual_cam_t_full": cam_t_full[n].astype(float).tolist(),
                    "median_reprojection_error_px": metrics["median_reprojection_error_px"],
                    "p95_reprojection_error_px": metrics["p95_reprojection_error_px"],
                    "median_depth_m": metrics["median_depth_m"],
                    "min_depth_m": metrics["min_depth_m"],
                    "max_depth_m": metrics["max_depth_m"],
                    "hand_bone_scale_m": metrics["hand_bone_scale_m"],
                },
                "rtmlib_measurement": {
                    "hand_idx": int(item["rtmlib_hand_idx"]),
                    "measurement_source": args.measurement_source,
                    "scores": item["scores"],
                    "candidate_qc": item["rtmlib_qc"],
                },
            }
            transform_hand_to_world(hand, T_world_camera)
            hands.append(hand)
            pred_i += 1
    return hands, meta


def summarize_hands(frames: list[dict]) -> dict:
    hands = [hand for frame in frames for hand in frame.get("hands", [])]
    measured = [hand for hand in hands if bool(hand.get("measurement_available", False))]
    reproj = [
        float(hand.get("source_camera_solve", {}).get("median_reprojection_error_px", np.nan))
        for hand in measured
    ]
    depth = [
        float(hand.get("source_camera_solve", {}).get("median_depth_m", np.nan))
        for hand in measured
    ]
    bone = [
        float(hand.get("source_camera_solve", {}).get("hand_bone_scale_m", np.nan))
        for hand in measured
    ]
    return {
        "hand_rows": int(len(hands)),
        "measured_hand_rows": int(len(measured)),
        "measured_reprojection_median_px": summarize(reproj),
        "measured_depth_m": summarize(depth),
        "measured_hand_bone_m": summarize(bone),
    }


def run(args: argparse.Namespace) -> dict:
    inputs = load_frame_inputs(args)
    model, cfg, device = load_hamer_backend(args)
    output_frames: list[dict] = []
    proposal_qc: list[dict] = []
    for frame in tqdm(inputs, desc="hamer_rtmlib"):
        out_frame = copy.deepcopy(frame.annotation)
        hands, proposal_meta = run_hamer_on_frame(model, cfg, device, frame, args)
        out_frame["hands"] = hands
        output_frames.append(out_frame)
        proposal_qc.extend(proposal_meta)
    if not output_frames:
        raise RuntimeError("HaMeR received no frames")
    summary = summarize_hands(output_frames)
    result = {"frames": output_frames}
    save_json(args.output_annotations, result)
    enough_measured = summary["measured_hand_rows"] >= args.min_measured_hands
    report = {
        "status": "ok" if enough_measured else "insufficient_measured_hands",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "run_hamer_rtmlib_hand_stream_v3",
        "hamer_root": str(args.hamer_root),
        "checkpoint": str(args.checkpoint),
        "target_annotations": str(args.target_annotations),
        "frame_manifest": str(args.frame_manifest),
        "rtmlib_json": str(args.rtmlib_json),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(output_frames)),
        **summary,
        "proposal_rows": int(len(proposal_qc)),
        "accepted_proposal_rows": int(sum(1 for row in proposal_qc if bool(row.get("accepted_for_hamer", False)))),
        "proposal_preview": proposal_qc[:80],
        "interpretation": (
            "This runs HaMeR on RTMLib hand boxes with both hand-side hypotheses, then solves MANO translation "
            "against annotation VGGT intrinsics and the configured 2D hand observation source. The output is a "
            "hand observation stream; metric-depth refit and mesh-surface contact diagnostics decide whether it is usable for V3."
        ),
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k != "proposal_preview"}, indent=2))
    if not enough_measured and not args.allow_insufficient_measured_hands:
        raise RuntimeError(f"only {summary['measured_hand_rows']} measured HaMeR hands, min_measured_hands={args.min_measured_hands}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--rtmlib-json", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--hamer-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--source-intrinsics-fx-fy-cx-cy", type=float, nargs=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--measurement-source", choices=["rtmlib-keypoints", "hamer-full-projection"], default="rtmlib-keypoints")
    parser.add_argument("--min-keypoint-score", type=float, default=0.2)
    parser.add_argument("--min-valid-keypoints", type=int, default=12)
    parser.add_argument("--min-mean-score", type=float, default=0.30)
    parser.add_argument("--min-box-area-fraction", type=float, default=0.006)
    parser.add_argument("--max-box-area-fraction", type=float, default=0.090)
    parser.add_argument("--min-box-side-px", type=float, default=45.0)
    parser.add_argument("--max-box-side-px", type=float, default=620.0)
    parser.add_argument("--max-keypoint-span-x-px", type=float, default=650.0)
    parser.add_argument("--max-keypoint-span-y-px", type=float, default=650.0)
    parser.add_argument("--min-depth-m", type=float, default=0.12)
    parser.add_argument("--max-depth-m", type=float, default=2.2)
    parser.add_argument("--max-initial-reprojection-px", type=float, default=55.0)
    parser.add_argument("--min-hand-bone-m", type=float, default=0.12)
    parser.add_argument("--max-hand-bone-m", type=float, default=0.24)
    parser.add_argument("--max-vertices-per-hand", type=int, default=1600)
    parser.add_argument("--min-measured-hands", type=int, default=1)
    parser.add_argument("--allow-insufficient-measured-hands", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
