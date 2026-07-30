#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from run_vggt_object_geometry_v3 import (
    apply_sim3,
    camera_centers_from_annotations,
    camera_centers_from_vggt,
    frame_map,
    import_vggt,
    load_views,
    localize_manifest_paths,
    read_manifest,
    run_vggt,
    select_object_points,
    umeyama_similarity,
)


def export_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(colors, dtype=np.uint8)
    points = np.asarray(points, dtype=float)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.views_dir / "vggt_object_views_manifest.json", int(args.frame_start), int(args.frame_end))
    rows = localize_manifest_paths(rows, args.remote_output_root, args.local_output_root)
    frame_indices = [int(row["frame_idx"]) for row in rows]
    frame_by_idx = frame_map(args.annotations)
    images, masks, rgbs = load_views(rows, int(args.target_size))
    extrinsic, intrinsic, depth, depth_conf, points_vggt = run_vggt(args, images)
    vggt_centers = camera_centers_from_vggt(extrinsic)
    target_centers = camera_centers_from_annotations(frame_by_idx, frame_indices)
    scale, R, t = umeyama_similarity(vggt_centers, target_centers)
    center_error = np.linalg.norm(apply_sim3(vggt_centers, scale, R, t) - target_centers, axis=1)
    object_points_vggt, object_colors, point_reports = select_object_points(
        frame_indices, points_vggt, depth, depth_conf, masks, rgbs, args
    )
    object_points_metric = apply_sim3(object_points_vggt, scale, R, t)
    point_path = args.output_dir / "vggt_object_points_metric_raw.ply"
    export_point_ply(point_path, object_points_metric, object_colors)
    pred_path = args.output_dir / "vggt_predictions_object_geometry_raw.npz"
    np.savez_compressed(
        pred_path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        object_points_metric=object_points_metric.astype(np.float32),
        object_colors=object_colors.astype(np.uint8),
        masks=masks.astype(np.uint8),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
        camera_centers_vggt=vggt_centers.astype(np.float32),
        camera_centers_metric=target_centers.astype(np.float32),
        sim3_scale=np.asarray([scale], dtype=np.float32),
        sim3_rotation=R.astype(np.float32),
        sim3_translation=t.astype(np.float32),
    )
    report = {
        "status": "ok",
        "method": "vggt_masked_multiview_object_predictions",
        "frames": frame_indices,
        "model_id": str(args.model_id),
        "camera_alignment_median_m": float(np.median(center_error)),
        "camera_alignment_p95_m": float(np.percentile(center_error, 95)),
        "camera_alignment_max_m": float(np.max(center_error)),
        "sim3_scale": float(scale),
        "object_points": int(len(object_points_metric)),
        "object_point_extent_m": (object_points_metric.max(axis=0) - object_points_metric.min(axis=0)).astype(float).tolist(),
        "point_reports": point_reports,
        "outputs": {
            "points": str(point_path),
            "predictions": str(pred_path),
        },
    }
    (args.output_dir / "qc_vggt_object_predictions_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "point_reports"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--model-file", default="model.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-depth-conf", type=float, default=0.0)
    parser.add_argument("--conf-quantile", type=float, default=0.30)
    parser.add_argument("--min-points-per-frame", type=int, default=900)
    parser.add_argument("--max-points-per-frame", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
