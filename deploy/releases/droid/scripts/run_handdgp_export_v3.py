#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(float)
    z = np.clip(points[:, 2], 1e-6, None)
    return np.c_[fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy]


def source_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def bbox_square(bbox_xyxy: list[float], width: int, height: int, scale: float) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid bbox {bbox_xyxy}")
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0, 1.0) * float(scale)
    return np.asarray([cx - 0.5 * side, cy - 0.5 * side, side, side], dtype=np.float32)


def crop_and_intrinsics(
    image: np.ndarray,
    bbox_xyxy: list[float],
    intrinsics: np.ndarray,
    input_size: int,
    crop_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    patch = bbox_square(bbox_xyxy, width, height, crop_scale)
    x, y, side, _ = patch.astype(float)
    src = np.asarray(
        [[x, y], [x + side, y], [x, y + side]],
        dtype=np.float32,
    )
    dst = np.asarray(
        [[0.0, 0.0], [float(input_size), 0.0], [0.0, float(input_size)]],
        dtype=np.float32,
    )
    affine = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        image,
        affine,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    fx, fy, cx, cy = intrinsics.astype(float)
    crop_intr = np.eye(4, dtype=np.float32)
    scale = float(input_size) / max(side, 1e-6)
    crop_intr[0, 0] = fx * scale
    crop_intr[1, 1] = fy * scale
    crop_intr[0, 2] = (cx - x) * scale
    crop_intr[1, 2] = (cy - y) * scale
    return crop, crop_intr, affine.astype(np.float32)


def observed_hands(frame: dict, min_score: float) -> list[dict]:
    hands = []
    for hand in frame.get("hands", []):
        if not bool(hand.get("measurement_available", False)):
            continue
        score = float(hand.get("detector_score", 0.0))
        if score < min_score:
            continue
        side = str(hand.get("side", "")).lower()
        if side not in {"left", "right"}:
            continue
        raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
        bbox = hand.get("bbox_xyxy")
        if raw2d.shape != (21, 2) or intr.shape != (4,) or not isinstance(bbox, list):
            continue
        hands.append(hand)
    return hands


def load_handdgp(handdgp_root: Path, checkpoint: Path, batch_size: int, input_size: int, device: torch.device):
    handdgp_root = handdgp_root.resolve()
    checkpoint = checkpoint.resolve()
    transform_cache = handdgp_root / "third_party" / "HandMesh" / "template" / "transform.pkl"
    if not transform_cache.is_file():
        raise RuntimeError(f"HandDGP transform cache is missing: {transform_cache}")
    sys.path.insert(0, str(handdgp_root))
    prev_cwd = Path.cwd()
    os.chdir(handdgp_root)
    try:
        from src.models.handdgp import HandDGP

        model = HandDGP(
            batch_size=batch_size,
            latent_size=256,
            spiral_len=(9, 9, 9, 9),
            spiral_dilation=(1, 1, 1, 1),
            spiral_out_channels=(32, 64, 128, 256),
            variant="resnet50",
            imagenet_pretrain=False,
            input_size=input_size,
        )
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        cleaned = {}
        for key, value in state.items():
            clean_key = str(key)
            for prefix in ("model.", "module."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            cleaned[clean_key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        model_keys = set(model.state_dict())
        loaded_keys = model_keys.intersection(cleaned)
        if len(loaded_keys) < int(0.8 * len(model_keys)):
            raise RuntimeError(
                f"HandDGP checkpoint loaded too few model keys: {len(loaded_keys)}/{len(model_keys)} "
                f"missing={missing[:20]} unexpected={unexpected[:20]}"
            )
        model.to(device).eval()
    finally:
        os.chdir(prev_cwd)
    return model


def tensor_from_crop(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return tensor.to(device)


def mirror_image_and_intrinsics(crop: np.ndarray, intr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirrored = np.ascontiguousarray(crop[:, ::-1])
    intr_m = intr.copy()
    width = crop.shape[1]
    intr_m[0, 2] = float(width - 1) - float(intr_m[0, 2])
    return mirrored, intr_m


def unmirror_points(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 0] *= -1.0
    return out


def mpii_to_annotation_order(points: np.ndarray) -> np.ndarray:
    if points.shape[0] != 21:
        raise RuntimeError(f"expected 21 hand joints, got {points.shape}")
    labels_mpii = [
        "W",
        "T0",
        "T1",
        "T2",
        "T3",
        "I0",
        "I1",
        "I2",
        "I3",
        "M0",
        "M1",
        "M2",
        "M3",
        "R0",
        "R1",
        "R2",
        "R3",
        "L0",
        "L1",
        "L2",
        "L3",
    ]
    labels_annotation = [
        "W",
        "T0",
        "T1",
        "T2",
        "T3",
        "I0",
        "I1",
        "I2",
        "I3",
        "M0",
        "M1",
        "M2",
        "M3",
        "R0",
        "R1",
        "R2",
        "R3",
        "L0",
        "L1",
        "L2",
        "L3",
    ]
    return np.stack([points[labels_mpii.index(label)] for label in labels_annotation], axis=0)


def hand_record(
    frame: dict,
    source_hand: dict,
    joints_source: np.ndarray,
    vertices_source: np.ndarray,
    keypoints2d_crop: np.ndarray,
    keypoints2d_from_3d_crop: np.ndarray,
    weights: np.ndarray,
    translation: np.ndarray,
) -> tuple[dict, dict]:
    intr = np.asarray(source_hand["source_intrinsics"], dtype=float)
    raw2d = np.asarray(source_hand["joints2d_raw"], dtype=float)
    projected = project(joints_source, intr)
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    joints_world = source_to_world(joints_source, T_world_camera)
    vertices_world = source_to_world(vertices_source, T_world_camera)
    root = joints_source[0]
    local_joints = joints_source - root[None, :]
    local_vertices = vertices_source - root[None, :]
    hand = {
        "backend": "HandDGP",
        "side": str(source_hand["side"]).lower(),
        "measurement_available": True,
        "detector_score": float(source_hand.get("detector_score", 0.0)),
        "filter_status": "handdgp_camera_space",
        "track_id": str(source_hand.get("track_id", "")),
        "track_source": "handdgp_from_" + str(source_hand.get("backend", "unknown")).lower(),
        "bbox_xyxy": [float(v) for v in source_hand["bbox_xyxy"]],
        "source_intrinsics": intr.astype(float).tolist(),
        "cam_t": root.astype(float).tolist(),
        "joints3d_camera": local_joints.astype(float).tolist(),
        "vertices_camera": local_vertices.astype(float).tolist(),
        "joints3d_source_camera_m": joints_source.astype(float).tolist(),
        "vertices_source_camera_m": vertices_source.astype(float).tolist(),
        "joints3d_world_m": joints_world.astype(float).tolist(),
        "vertices_world_m": vertices_world.astype(float).tolist(),
        "joints2d": projected.astype(float).tolist(),
        "joints2d_raw": raw2d.astype(float).tolist(),
        "projection_residual_to_measurement_px": {
            "median": float(np.median(reproj)),
            "p95": float(np.percentile(reproj, 95.0)),
        },
        "handdgp": {
            "translation_m": translation.astype(float).tolist(),
            "keypoints2d_crop": keypoints2d_crop.astype(float).tolist(),
            "keypoints2d_from_3d_crop": keypoints2d_from_3d_crop.astype(float).tolist(),
            "dgp_weights": weights.astype(float).tolist(),
            "source_backend": source_hand.get("backend"),
            "source_filter_status": source_hand.get("filter_status"),
            "source_track_id": source_hand.get("track_id"),
            "source_candidate_stream": source_hand.get("v7_candidate_stream"),
        },
        "world_coordinate_status": "handdgp_camera_space_transformed_by_existing_annotation_camera_pose",
        "mano_surface_status": "full_vertices_without_mano_params",
        "mano_vertex_count": int(len(vertices_source)),
    }
    row = {
        "frame_idx": int(frame["frame_idx"]),
        "side": hand["side"],
        "detector_score": hand["detector_score"],
        "median_reprojection_px": float(np.median(reproj)),
        "p95_reprojection_px": float(np.percentile(reproj, 95.0)),
        "median_depth_m": float(np.median(joints_source[:, 2])),
        "translation_m": translation.astype(float).tolist(),
        "mean_dgp_weight": float(np.mean(weights)),
    }
    return hand, row


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frame_map = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {args.video}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_handdgp(args.handdgp_root, args.checkpoint, args.batch_size, args.input_size, device)
    output = copy.deepcopy(annotations)
    output_frame_map = {int(frame["frame_idx"]): frame for frame in output["frames"]}

    pending: list[dict] = []
    rows: list[dict] = []
    skipped: list[dict] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        images = torch.stack([item["tensor"] for item in pending], dim=0)
        intrinsics = torch.stack([item["intr"] for item in pending], dim=0)
        with torch.no_grad():
            out = model(images, intrinsics=intrinsics, hand_scale=args.hand_scale)
        joints = out["keypoints3D_cs"].detach().cpu().numpy()
        vertices = out["mesh_vertices3D_cs"].detach().cpu().numpy()
        key2d = (out["keypoints2D"].detach().cpu().numpy() * float(args.input_size))
        key2d_from_3d = (out["keypoints2D_from_3D"].detach().cpu().numpy() * float(args.input_size))
        weights = out["weights"].detach().cpu().numpy()
        translation = out["translation"].detach().cpu().numpy()
        for i, item in enumerate(pending):
            pred_joints = mpii_to_annotation_order(joints[i])
            pred_vertices = vertices[i]
            pred_translation = translation[i]
            if item["mirrored"]:
                pred_joints = unmirror_points(pred_joints)
                pred_vertices = unmirror_points(pred_vertices)
                pred_translation = unmirror_points(pred_translation.reshape(1, 3))[0]
            hand, row = hand_record(
                item["frame"],
                item["source_hand"],
                pred_joints,
                pred_vertices,
                key2d[i],
                key2d_from_3d[i],
                weights[i],
                pred_translation,
            )
            output_frame = output_frame_map[int(item["frame"]["frame_idx"])]
            output_frame.setdefault("hands", [])
            output_frame["hands"].append(hand)
            rows.append(row)
        pending = []

    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frame_map.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation"})
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = cap.read()
        if not ok:
            skipped.append({"frame_idx": frame_idx, "reason": "video_read_failed"})
            continue
        output_frame_map[frame_idx]["hands"] = []
        for source_hand in observed_hands(frame, args.min_score):
            try:
                intr = np.asarray(source_hand["source_intrinsics"], dtype=float)
                crop, crop_intr, affine = crop_and_intrinsics(
                    image,
                    [float(v) for v in source_hand["bbox_xyxy"]],
                    intr,
                    args.input_size,
                    args.crop_scale,
                )
                mirrored = str(source_hand["side"]).lower() == "left"
                if mirrored:
                    crop, crop_intr = mirror_image_and_intrinsics(crop, crop_intr)
                pending.append(
                    {
                        "frame": frame,
                        "source_hand": source_hand,
                        "tensor": tensor_from_crop(crop, device),
                        "intr": torch.from_numpy(crop_intr).float().to(device),
                        "mirrored": mirrored,
                        "affine": affine,
                    }
                )
                if len(pending) >= args.batch_size:
                    flush()
            except Exception as exc:
                skipped.append(
                    {
                        "frame_idx": frame_idx,
                        "side": source_hand.get("side"),
                        "reason": str(exc),
                    }
                )
    flush()
    cap.release()
    save_json(args.output_annotations, output)
    raw_rows = {
        "frame_idx": np.asarray([row["frame_idx"] for row in rows], dtype=np.int32),
        "side": np.asarray([row["side"] for row in rows]),
        "median_reprojection_px": np.asarray([row["median_reprojection_px"] for row in rows], dtype=np.float32),
        "median_depth_m": np.asarray([row["median_depth_m"] for row in rows], dtype=np.float32),
    }
    np.savez_compressed(args.output_raw_npz, **raw_rows)
    reproj = np.asarray([row["median_reprojection_px"] for row in rows], dtype=float)
    depth = np.asarray([row["median_depth_m"] for row in rows], dtype=float)
    report = {
        "status": "diagnostic_handdgp_camera_space_export",
        "annotation_ready": False,
        "diagnostic_only": True,
        "video": str(args.video),
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "output_raw_npz": str(args.output_raw_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "hands": int(len(rows)),
        "skipped": int(len(skipped)),
        "median_reprojection_px": None if len(reproj) == 0 else float(np.median(reproj)),
        "p95_reprojection_px": None if len(reproj) == 0 else float(np.percentile(reproj, 95.0)),
        "median_depth_m": None if len(depth) == 0 else float(np.median(depth)),
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:120],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--handdgp-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-raw-npz", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--hand-scale", type=float, default=0.2)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
