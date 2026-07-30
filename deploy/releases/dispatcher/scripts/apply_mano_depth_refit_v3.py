#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from refit_mano_metric_depth_v3 import hand_vertex_key


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def source_to_world(points_camera_m: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera_m, np.ones(len(points_camera_m), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def hand_depth_shift(hand: dict, depth: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> tuple[float | None, dict]:
    if not hand.get("measurement_available", False):
        return None, {"reason": "not_measured"}
    score = float(hand.get("detector_score", np.nan))
    if not np.isfinite(score) or score < args.min_detector_score:
        return None, {"reason": "low_detector_score", "detector_score": score}
    joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
    raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
    intr = np.asarray(hand["source_intrinsics"], dtype=float)
    projected = project_points(joints, intr)
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    if float(np.median(reproj)) > args.max_initial_reprojection_px:
        return None, {"reason": "high_initial_reprojection", "median_reprojection_px": float(np.median(reproj))}
    samples = sample_depth(depth, raw2d, source_size)
    valid = np.isfinite(samples) & (samples > 0.0)
    valid &= reproj <= args.good_joint_reprojection_px
    if np.count_nonzero(valid) < args.min_depth_joints:
        return None, {"reason": "too_few_depth_joints", "depth_joints": int(np.count_nonzero(valid))}
    raw_shift = float(np.median(samples[valid] - joints[valid, 2]))
    shift = float(np.clip(raw_shift, -args.max_abs_ray_shift_m, args.max_abs_ray_shift_m))
    if abs(shift) < args.min_abs_shift_m:
        return None, {"reason": "small_shift", "raw_shift_m": raw_shift}
    return shift, {
        "reason": "applied",
        "raw_shift_m": raw_shift,
        "shift_m": shift,
        "shift_clipped": bool(abs(raw_shift - shift) > 1e-9),
        "median_initial_reprojection_px": float(np.median(reproj)),
        "depth_joints": int(np.count_nonzero(valid)),
    }


def apply_shift_to_hand(hand: dict, shift_m: float, T_world_camera: np.ndarray) -> dict:
    out = copy.deepcopy(hand)
    joints = np.asarray(out["joints3d_source_camera_m"], dtype=float)
    center = np.median(joints, axis=0)
    if center[2] <= 0.0:
        raise RuntimeError("hand center has non-positive depth")
    ray = center / center[2]
    joints_shifted = joints + shift_m * ray[None, :]
    local_joints = np.asarray(out["joints3d_camera"], dtype=float)
    out["cam_t"] = (np.asarray(out["cam_t"], dtype=float) + shift_m * ray).astype(float).tolist()
    out["joints3d_source_camera_m"] = joints_shifted.astype(float).tolist()
    out["joints2d"] = project_points(joints_shifted, np.asarray(out["source_intrinsics"], dtype=float)).astype(float).tolist()
    raw2d = np.asarray(out.get("joints2d_raw", []), dtype=float)
    if raw2d.shape == (21, 2):
        err = np.linalg.norm(np.asarray(out["joints2d"], dtype=float) - raw2d, axis=1)
        out["projection_residual_to_measurement_px"] = {
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95.0)),
        }
    local_key = hand_vertex_key(out)
    source_key = "vertices_source_camera_m" if local_key == "vertices_camera" else "vertices_source_camera_m_sample"
    vertices_local = np.asarray(out[local_key], dtype=float)
    vertices_shifted = vertices_local + np.asarray(out["cam_t"], dtype=float)[None, :]
    out[source_key] = vertices_shifted.astype(float).tolist()
    out["joints3d_world_m"] = source_to_world(joints_shifted, T_world_camera).astype(float).tolist()
    world_key = "vertices_world_m" if local_key == "vertices_camera" else "vertices_world_m_sample"
    out[world_key] = source_to_world(vertices_shifted, T_world_camera).astype(float).tolist()
    out["world_coordinate_status"] = "v3_metric_depth_refit_source_camera_mano_transformed_by_existing_camera_pose"
    out["filter_status"] = str(out.get("filter_status", "")) + "_v3_depth_refit"
    _ = local_joints
    return out


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    frames = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in frames)) != len(frames):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(frames)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    out_frames = []
    applied = []
    skipped = []
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            out_frames.append(frame)
            continue
        new_frame = copy.deepcopy(frame)
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(new_frame["object"]["source_image_size"], dtype=float)
            T_world_camera = np.asarray(new_frame["camera"]["T_world_camera_metric"], dtype=float)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            out_frames.append(new_frame)
            continue
        new_hands = []
        for hand_i, hand in enumerate(new_frame.get("hands", [])):
            shift, info = hand_depth_shift(hand, depth, source_size, args)
            if shift is None:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), **info})
                new_hands.append(hand)
                continue
            refit = apply_shift_to_hand(hand, shift, T_world_camera)
            refit["v3_metric_depth_refit"] = info
            applied.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), **info})
            new_hands.append(refit)
        new_frame["hands"] = new_hands
        out_frames.append(new_frame)
    output = {"frames": out_frames}
    save_json(args.output_annotations, output)
    shifts = np.asarray([row["shift_m"] for row in applied], dtype=float)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "applied_count": int(len(applied)),
        "skipped_count": int(len(skipped)),
        "shift_m": {
            "count": int(len(shifts)),
            "median": None if len(shifts) == 0 else float(np.median(shifts)),
            "p05": None if len(shifts) == 0 else float(np.percentile(shifts, 5.0)),
            "p95": None if len(shifts) == 0 else float(np.percentile(shifts, 95.0)),
            "max_abs": None if len(shifts) == 0 else float(np.max(np.abs(shifts))),
        },
        "applied_preview": applied[:80],
        "skipped_preview": skipped[:120],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"applied_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--max-initial-reprojection-px", type=float, default=20.0)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--min-depth-joints", type=int, default=12)
    parser.add_argument("--max-abs-ray-shift-m", type=float, default=0.16)
    parser.add_argument("--min-abs-shift-m", type=float, default=0.020)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
