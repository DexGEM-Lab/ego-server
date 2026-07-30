#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_contact_depth_conflict_v3 import mesh_frame_vertices, summarize
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_object_factor_graph_v3 import localize_path, project_world, resize_bool_mask


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def depth_frame(depths: np.ndarray, frame_to_depth_i: dict[int, int], frame_idx: int) -> np.ndarray:
    matches = [frame_to_depth_i[int(frame_idx)]] if int(frame_idx) in frame_to_depth_i else []
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(matches)} times in depth archive")
    return np.asarray(depths[int(matches[0])], dtype=float)


def sample_depth(depth: np.ndarray, uv_source: np.ndarray, source_size: np.ndarray) -> np.ndarray:
    scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size.astype(float)
    xy = uv_source * scale[None, :]
    valid = np.isfinite(xy).all(axis=1)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    out = np.full(len(uv_source), np.nan, dtype=float)
    out[valid] = depth[y[valid], x[valid]]
    return out


def masked_depth_values(depth: np.ndarray, mask: np.ndarray, source_size: np.ndarray) -> np.ndarray:
    if mask.shape != depth.shape:
        scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size.astype(float)
        # resize_bool_mask has already normalized to mask_image_size. If source and depth differ,
        # nearest-neighbor resampling preserves the selected object region for depth sampling.
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        _ = scale
    vals = depth[mask]
    return vals[np.isfinite(vals) & (vals > 0.0)]


def summarize_key(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=float)
    return summarize(values)


def group_summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "mano_minus_metric_depth_m": summarize_key(rows, "mano_minus_metric_depth_m"),
        "mano_over_metric_depth": summarize_key(rows, "mano_over_metric_depth"),
        "object_mesh_minus_metric_depth_m": summarize_key(rows, "object_mesh_minus_metric_depth_m"),
        "object_mesh_over_metric_depth": summarize_key(rows, "object_mesh_over_metric_depth"),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_frame_idx)) != len(depth_frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=float)
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        obj = frame.get("object", {})
        if not obj.get("mask_path"):
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            object_depth_vals = masked_depth_values(depth, mask, source_size)
            if len(object_depth_vals) < args.min_object_depth_pixels:
                raise RuntimeError(f"too few object depth pixels: {len(object_depth_vals)}")
            metric_object_depth = float(np.median(object_depth_vals))
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            _, object_mesh_z = project_world(object_vertices, T_world_camera, intrinsics)
            object_mesh_z = object_mesh_z[np.isfinite(object_mesh_z) & (object_mesh_z > 0.0)]
            if len(object_mesh_z) == 0:
                raise RuntimeError("object mesh has no positive camera depth")
            object_mesh_depth = float(np.median(object_mesh_z))
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue

        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
                if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) == 0:
                    raise RuntimeError("invalid joints3d_source_camera_m")
                uv = project_points(joints, np.asarray(hand["source_intrinsics"], dtype=float))
                sampled = sample_depth(depth, uv, source_size)
                valid = sampled[np.isfinite(sampled) & (sampled > 0.0)]
                if len(valid) < args.min_hand_depth_joints:
                    raise RuntimeError(f"too few hand depth samples: {len(valid)}")
                metric_hand_depth = float(np.median(valid))
                mano_depth = float(np.median(joints[:, 2]))
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_idx": hand_i,
                        "side": hand.get("side"),
                        "measurement_available": bool(hand.get("measurement_available", False)),
                        "detector_score": float(hand.get("detector_score", np.nan)),
                        "high_confidence_measured": bool(
                            hand.get("measurement_available", False)
                            and np.isfinite(float(hand.get("detector_score", np.nan)))
                            and float(hand.get("detector_score", np.nan)) >= args.report_min_score
                        ),
                        "mano_depth_m": mano_depth,
                        "metric_hand_depth_m": metric_hand_depth,
                        "mano_minus_metric_depth_m": mano_depth - metric_hand_depth,
                        "mano_over_metric_depth": mano_depth / metric_hand_depth,
                        "object_mesh_depth_m": object_mesh_depth,
                        "metric_object_depth_m": metric_object_depth,
                        "object_mesh_minus_metric_depth_m": object_mesh_depth - metric_object_depth,
                        "object_mesh_over_metric_depth": object_mesh_depth / metric_object_depth,
                    }
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
    if not rows:
        raise RuntimeError("no metric-depth alignment rows")
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "rows": len(rows),
        "skipped_rows": len(skipped),
        "all_rows_summary": group_summary(rows),
        "measured_rows_summary": group_summary([row for row in rows if row.get("measurement_available")]),
        "high_confidence_measured_rows_summary": group_summary([row for row in rows if row.get("high_confidence_measured")]),
        "interpretation": (
            "This compares current MANO and object mesh camera depths to the independent metric-depth map "
            "used for observed-surface meshing. Agreement here does not prove metric accuracy, but a large "
            "ratio mismatch localizes the scale conflict before contact optimization."
        ),
        "rows_preview": rows[:120],
        "skipped_preview": skipped[:120],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-object-depth-pixels", type=int, default=200)
    parser.add_argument("--min-hand-depth-joints", type=int, default=12)
    parser.add_argument("--report-min-score", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
