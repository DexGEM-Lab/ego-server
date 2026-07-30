#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
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

from run_hamer_rtmlib_hand_stream_v3 import (
    frame_manifest,
    hand_bone_scale_m,
    intrinsics_for,
    load_annotations,
    load_json,
    project_points,
    sample_vertices,
    save_json,
    solve_metric_hand,
    transform_for,
    transform_hand_to_world,
)
from run_wilor_full_frame import ensure_wilor_assets, mano_params_for_sample, patch_legacy_imports, project_full_image


@dataclass(frozen=True)
class FrameInput:
    frame_idx: int
    rgb_path: Path
    image: np.ndarray
    annotation: dict
    maskbox_hands: list[dict]


def metric_scale_from_raw(raw_frames: list[dict], target_bone_m: float) -> dict:
    bones = []
    for frame in raw_frames:
        for hand in frame["raw_hands"]:
            joints = np.asarray(hand["joints3d_camera"], dtype=float)
            bone = hand_bone_scale_m(joints)
            if 0.08 < bone < 0.24:
                bones.append(bone)
    if not bones:
        raise RuntimeError("no plausible WiLoR local hand-bone scales")
    arr = np.asarray(bones, dtype=float)
    median_bone = float(np.median(arr))
    scale = float(target_bone_m) / median_bone
    residual = arr * scale - float(target_bone_m)
    return {
        "status": "wilor_local_hand_geometry_scaled_by_median_finger_chain_bone",
        "target_hand_bone_m": float(target_bone_m),
        "median_wilor_hand_bone": median_bone,
        "wilor_local_to_meters": float(scale),
        "sample_count": int(len(arr)),
        "residual_iqr_m": [float(np.percentile(residual, 25)), float(np.percentile(residual, 75))],
    }


def load_maskbox(path: Path, frame_start: int, frame_end: int, track_id: str) -> dict[int, list[dict]]:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} has no frames")
    out: dict[int, list[dict]] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < frame_start or frame_idx > frame_end:
            continue
        hands = []
        for hand in frame.get("hands", []):
            if str(hand.get("track_id", "")) == track_id:
                hands.append(hand)
        out[frame_idx] = hands
    missing = [idx for idx in range(frame_start, frame_end + 1) if idx not in out]
    if missing:
        raise RuntimeError(f"mask-box hand evidence missing frames {missing[:20]}")
    empty = [idx for idx, hands in sorted(out.items()) if not hands]
    if empty:
        raise RuntimeError(f"mask-box hand evidence has no hands for track {track_id}: {empty[:20]}")
    return out


def load_frame_inputs(args: argparse.Namespace) -> list[FrameInput]:
    rgb_by_frame = frame_manifest(args.frame_manifest, args.local_root, args.remote_root)
    annotations = load_annotations(args.target_annotations, args.frame_start, args.frame_end)
    maskbox = load_maskbox(args.maskbox_json, args.frame_start, args.frame_end, args.track_id)
    inputs: list[FrameInput] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        rgb_path = rgb_by_frame.get(frame_idx)
        if rgb_path is None:
            raise RuntimeError(f"RGB manifest missing source frame {frame_idx}")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image {rgb_path}")
        inputs.append(
            FrameInput(
                frame_idx=frame_idx,
                rgb_path=rgb_path,
                image=image,
                annotation=annotations[frame_idx],
                maskbox_hands=maskbox[frame_idx],
            )
        )
    return inputs


def load_wilor_model(args: argparse.Namespace):
    ensure_wilor_assets(args.wilor_root, args.mano_right)
    patch_legacy_imports()
    sys.path.insert(0, str(args.wilor_root.resolve()))
    from wilor.models import load_wilor

    cwd = Path.cwd()
    os.chdir(args.wilor_root)
    try:
        model, cfg = load_wilor("./pretrained_models/wilor_final.ckpt", "./pretrained_models/model_config.yaml")
    finally:
        os.chdir(cwd)
    device = torch.device(args.device)
    model = model.to(device).eval()
    return model, cfg, device


def side_hypotheses(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    if args.side == "left":
        return (("left", 0.0),)
    if args.side == "right":
        return (("right", 1.0),)
    return (("left", 0.0), ("right", 1.0))


def run_wilor_on_maskboxes(model, cfg, device, frame: FrameInput, args: argparse.Namespace) -> list[dict]:
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    boxes: list[np.ndarray] = []
    rights: list[float] = []
    meta: list[dict] = []
    for hand_i, evidence in enumerate(frame.maskbox_hands):
        box = np.asarray(evidence["bbox_xyxy"], dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise RuntimeError(f"frame {frame.frame_idx} invalid mask-box bbox")
        for side, right_value in side_hypotheses(args):
            boxes.append(box)
            rights.append(right_value)
            meta.append({"maskbox_index": int(hand_i), "side": side, "evidence": evidence})
    if not boxes:
        return []
    dataset = ViTDetDataset(
        cfg,
        frame.image,
        np.stack(boxes).astype(np.float32),
        np.asarray(rights, dtype=np.float32),
        rescale_factor=float(args.rescale_factor),
        fp16=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    predictions: list[dict] = []
    pred_i = 0
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
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        focal = float(scaled_focal_length.detach().cpu().numpy())
        img_size_np = img_size.detach().cpu().numpy()
        for n in range(int(batch["img"].shape[0])):
            item = meta[pred_i]
            side = str(item["side"])
            sign = 1.0 if side == "right" else -1.0
            vertices = out["pred_vertices"][n].detach().cpu().numpy().astype(float)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
            vertices[:, 0] = sign * vertices[:, 0]
            joints[:, 0] = sign * joints[:, 0]
            joints2d = project_full_image(joints, cam_t[n], focal, img_size_np[n])
            evidence = item["evidence"]
            predictions.append(
                {
                    "backend": "WiLoR",
                    "side": side,
                    "detector_score": float(evidence.get("mean_score", 1.0)),
                    "bbox_xyxy": np.asarray(evidence["bbox_xyxy"], dtype=float).tolist(),
                    "cam_t": cam_t[n].astype(float).tolist(),
                    "focal_length": focal,
                    "joints3d_camera": joints.astype(float).tolist(),
                    "vertices_camera": vertices.astype(float).tolist(),
                    "joints2d_raw": joints2d.astype(float).tolist(),
                    "joints2d": joints2d.astype(float).tolist(),
                    "mano_params": mano_params_for_sample(out, n),
                    "filter_status": "wilor_maskbox_raw",
                    "maskbox_measurement": {
                        "track_id": str(evidence.get("track_id", "")),
                        "maskbox_index": int(item["maskbox_index"]),
                        "mask_area_px": int(evidence.get("mask_area_px", 0)),
                        "mask_center_xy": evidence.get("mask_center_xy"),
                    },
                }
            )
            pred_i += 1
    return predictions


def normalize_raw_frame(frame: dict, annotation: dict, scale: float, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    intr = intrinsics_for(annotation)
    T_world_camera = transform_for(annotation)
    hands = []
    rejected = []
    for hand_i, raw in enumerate(frame.get("raw_hands", [])):
        try:
            local_joints = np.asarray(raw["joints3d_camera"], dtype=float) * float(scale)
            local_vertices = np.asarray(raw["vertices_camera"], dtype=float) * float(scale)
            raw2d = np.asarray(raw["joints2d_raw"], dtype=float)
            if local_joints.shape != (21, 3) or local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                raise RuntimeError("invalid WiLoR geometry")
            if raw2d.shape != (21, 2):
                raise RuntimeError("invalid WiLoR projected keypoints")
            translation, source_joints, source_vertices, metrics = solve_metric_hand(local_joints, local_vertices, raw2d, intr)
            sampled_vertices, surface_status = sample_vertices(local_vertices, int(args.max_vertices_per_hand))
            sampled_source_vertices = sampled_vertices + translation[None, :]
            measurement_available = (
                float(args.min_depth_m) <= metrics["median_depth_m"] <= float(args.max_depth_m)
                and metrics["median_reprojection_error_px"] <= float(args.max_initial_reprojection_px)
                and float(args.min_hand_bone_m) <= metrics["hand_bone_scale_m"] <= float(args.max_hand_bone_m)
            )
            hand = copy.deepcopy(raw)
            hand["track_id"] = str(args.track_id)
            hand["track_source"] = "vlm_sam_maskbox_wilor_both_side_metric_translation"
            hand["cam_t"] = translation.astype(float).tolist()
            hand["source_intrinsics"] = intr.astype(float).tolist()
            hand["joints3d_camera"] = local_joints.astype(float).tolist()
            hand["vertices_camera"] = local_vertices.astype(float).tolist()
            hand["joints3d_source_camera_m"] = source_joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = source_vertices.astype(float).tolist()
            hand["vertices_source_camera_m_sample"] = sampled_source_vertices.astype(float).tolist()
            hand["joints2d"] = project_points(source_joints, intr).astype(float).tolist()
            hand["mano_surface_status"] = surface_status
            hand["mano_vertex_count"] = int(len(local_vertices))
            hand["measurement_available"] = bool(measurement_available)
            hand["filter_status"] = "measured_source_camera_solve" if measurement_available else "rejected_initial_metric_qc"
            hand["source_camera_solve"] = {
                "status": "least_squares_translation_from_wilor_local_geometry_and_wilor_full_projection",
                "measurement_source": "wilor-full-projection",
                "median_reprojection_error_px": metrics["median_reprojection_error_px"],
                "p95_reprojection_error_px": metrics["p95_reprojection_error_px"],
                "median_depth_m": metrics["median_depth_m"],
                "min_depth_m": metrics["min_depth_m"],
                "max_depth_m": metrics["max_depth_m"],
                "hand_bone_scale_m": metrics["hand_bone_scale_m"],
                "wilor_local_to_meters": float(scale),
            }
            transform_hand_to_world(hand, T_world_camera)
            hands.append(hand)
        except Exception as exc:
            rejected.append({"hand_index": int(hand_i), "reason": str(exc)})
    return hands, rejected


def summarize_hands(frames: list[dict]) -> dict:
    rows = [hand for frame in frames for hand in frame.get("hands", [])]
    measured = [hand for hand in rows if bool(hand.get("measurement_available"))]
    return {
        "hand_rows": int(len(rows)),
        "measured_hand_rows": int(len(measured)),
        "frames_with_hands": int(sum(1 for frame in frames if frame.get("hands"))),
        "frames_with_measured_hands": int(
            sum(1 for frame in frames if any(bool(hand.get("measurement_available")) for hand in frame.get("hands", [])))
        ),
    }


def run(args: argparse.Namespace) -> dict:
    inputs = load_frame_inputs(args)
    model, cfg, device = load_wilor_model(args)
    raw_frames = []
    try:
        for frame in tqdm(inputs, desc="wilor_maskbox"):
            raw_frames.append(
                {
                    "frame_idx": int(frame.frame_idx),
                    "time_s": frame.annotation.get("time_s"),
                    "caption": frame.annotation.get("caption", ""),
                    "raw_hands": run_wilor_on_maskboxes(model, cfg, device, frame, args),
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scale = metric_scale_from_raw(raw_frames, float(args.hand_bone_target_m))
    output_frames = []
    rejected = []
    for frame, raw in zip(inputs, raw_frames):
        out_frame = copy.deepcopy(frame.annotation)
        hands, rejected_rows = normalize_raw_frame(raw, out_frame, float(scale["wilor_local_to_meters"]), args)
        out_frame["hands"] = hands
        output_frames.append(out_frame)
        for row in rejected_rows:
            rejected.append({"frame_idx": int(frame.frame_idx), **row})
    result = {"frames": output_frames}
    save_json(args.output_annotations, result)
    raw_path = args.output_annotations.with_name(args.output_annotations.stem + "_raw.json")
    save_json(raw_path, {"frames": raw_frames})
    summary = summarize_hands(output_frames)
    enough = int(summary["measured_hand_rows"]) >= int(args.min_measured_hands)
    report = {
        "status": "ok" if enough else "insufficient_measured_hands",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "run_wilor_maskbox_hand_stream_v7",
        "target_annotations": str(args.target_annotations),
        "frame_manifest": str(args.frame_manifest),
        "maskbox_json": str(args.maskbox_json),
        "output_annotations": str(args.output_annotations),
        "raw_output": str(raw_path),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "track_id": str(args.track_id),
        "side": str(args.side),
        "scale": scale,
        **summary,
        "rejected_rows": int(len(rejected)),
        "rejected_preview": rejected[:80],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rejected_preview"}, indent=2))
    if not enough and not args.allow_insufficient_measured_hands:
        raise RuntimeError(f"only {summary['measured_hand_rows']} measured WiLoR hands, min_measured_hands={args.min_measured_hands}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--maskbox-json", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--mano-right", type=Path, default=Path("third_party/WiLoR/mano_data/MANO_RIGHT.pkl"))
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--side", choices=["left", "right", "any"], default="any")
    parser.add_argument("--hand-bone-target-m", type=float, default=0.165)
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
