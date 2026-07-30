#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame
from optimize_contact_depth_scale_v3 import summarize
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def depth_patch_stats(depth: np.ndarray, xy_depth: np.ndarray, radius: int) -> tuple[float, float, int]:
    x = int(np.clip(round(float(xy_depth[0])), 0, depth.shape[1] - 1))
    y = int(np.clip(round(float(xy_depth[1])), 0, depth.shape[0] - 1))
    patch = depth[max(0, y - radius) : min(depth.shape[0], y + radius + 1), max(0, x - radius) : min(depth.shape[1], x + radius + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0.0)]
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    return float(np.median(vals)), float(np.percentile(vals, 75.0) - np.percentile(vals, 25.0)), int(len(vals))


def summarize_key(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=float)
    return summarize(values)


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_frame_idx)) != len(depth_frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        obj = frame.get("object", {})
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            depth_mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            dist = mask_distance_map(depth_mask)
            depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            score = float(hand.get("detector_score", np.nan))
            if not hand.get("measurement_available", False) or not np.isfinite(score) or score < args.min_detector_score:
                continue
            joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
            raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
            intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
            if joints.shape != (21, 3) or raw2d.shape != (21, 2) or intr.shape != (4,):
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "reason": "invalid_hand_fields"})
                continue
            projected = project_points(joints, intr)
            reproj = np.linalg.norm(projected - raw2d, axis=1)
            for joint_i, xy_source in enumerate(raw2d):
                xy_depth = xy_source * depth_scale
                x = int(np.clip(round(float(xy_depth[0])), 0, depth.shape[1] - 1))
                y = int(np.clip(round(float(xy_depth[1])), 0, depth.shape[0] - 1))
                z = float(depth[y, x])
                if not np.isfinite(z) or z <= 0.0:
                    continue
                patch_med, patch_iqr, patch_count = depth_patch_stats(depth, xy_depth, args.patch_radius)
                depth_iqr_ratio = patch_iqr / max(1e-6, patch_med) if np.isfinite(patch_iqr) and np.isfinite(patch_med) else float("nan")
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_idx": hand_i,
                        "side": hand.get("side"),
                        "joint_i": int(joint_i),
                        "detector_score": score,
                        "joint_reprojection_px": float(reproj[joint_i]),
                        "mano_minus_metric_depth_m": float(joints[joint_i, 2] - z),
                        "metric_depth_m": z,
                        "mano_depth_m": float(joints[joint_i, 2]),
                        "depth_patch_iqr_m": patch_iqr,
                        "depth_patch_iqr_ratio": depth_iqr_ratio,
                        "depth_patch_count": patch_count,
                        "object_mask_distance_px": float(dist[y, x]),
                        "inside_object_mask": bool(depth_mask[y, x]),
                        "good_keypoint": bool(reproj[joint_i] <= args.good_joint_reprojection_px),
                        "stable_depth": bool(depth_iqr_ratio <= args.max_depth_iqr_ratio),
                    }
                )
    if not rows:
        raise RuntimeError("no depth reliability rows")
    good = [row for row in rows if row["good_keypoint"]]
    stable = [row for row in rows if row["good_keypoint"] and row["stable_depth"]]
    near_mask = [row for row in stable if row["object_mask_distance_px"] <= args.near_object_mask_px]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "rows": len(rows),
        "skipped_rows": len(skipped),
        "good_keypoint_rows": len(good),
        "stable_good_keypoint_rows": len(stable),
        "stable_good_near_object_rows": len(near_mask),
        "all": {
            "mano_minus_metric_depth_m": summarize_key(rows, "mano_minus_metric_depth_m"),
            "depth_patch_iqr_ratio": summarize_key(rows, "depth_patch_iqr_ratio"),
            "object_mask_distance_px": summarize_key(rows, "object_mask_distance_px"),
        },
        "good_keypoint": {
            "mano_minus_metric_depth_m": summarize_key(good, "mano_minus_metric_depth_m"),
            "depth_patch_iqr_ratio": summarize_key(good, "depth_patch_iqr_ratio"),
            "object_mask_distance_px": summarize_key(good, "object_mask_distance_px"),
        },
        "stable_good_keypoint": {
            "mano_minus_metric_depth_m": summarize_key(stable, "mano_minus_metric_depth_m"),
            "depth_patch_iqr_ratio": summarize_key(stable, "depth_patch_iqr_ratio"),
            "object_mask_distance_px": summarize_key(stable, "object_mask_distance_px"),
        },
        "stable_good_near_object": {
            "mano_minus_metric_depth_m": summarize_key(near_mask, "mano_minus_metric_depth_m"),
            "depth_patch_iqr_ratio": summarize_key(near_mask, "depth_patch_iqr_ratio"),
            "object_mask_distance_px": summarize_key(near_mask, "object_mask_distance_px"),
        },
        "interpretation": (
            "Metric-depth samples at hand joints should be weighted by local depth stability and keypoint reprojection. "
            "Samples near object masks are also suspect because the hand/object depth ordering can be ambiguous at contact or occlusion."
        ),
        "rows_preview": rows[:160],
        "skipped_preview": skipped[:80],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--near-object-mask-px", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
