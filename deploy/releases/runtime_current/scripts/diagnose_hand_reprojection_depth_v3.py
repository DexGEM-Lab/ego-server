#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from optimize_contact_depth_scale_v3 import summarize


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_corners(box: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    return np.asarray([[x0, y0], [x1, y1]], dtype=float)


def project_points(points_camera_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = np.clip(points_camera_m[:, 2], 1e-6, None)
    return np.c_[fx * points_camera_m[:, 0] / z + cx, fy * points_camera_m[:, 1] / z + cy]


def hand_rows(annotations: dict, frame_start: int, frame_end: int) -> list[dict]:
    rows = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        if idx < frame_start or idx > frame_end:
            continue
        for hand in frame.get("hands", []):
            joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
            vertices = np.asarray(hand.get("vertices_camera", []), dtype=float)
            cam_t = np.asarray(hand.get("cam_t", [0.0, 0.0, 0.0]), dtype=float)
            intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
            box = hand.get("bbox_xyxy")
            if joints.ndim != 2 or joints.shape[1] != 3 or vertices.ndim != 2 or vertices.shape[1] != 3:
                continue
            if intr.shape != (4,) or box is None:
                continue
            points = np.vstack([joints + cam_t[None, :], vertices + cam_t[None, :]])
            if not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 0):
                continue
            projected = project_points(points, intr)
            proj_min = projected.min(axis=0)
            proj_max = projected.max(axis=0)
            target = bbox_corners(box)
            bbox_residual = np.r_[proj_min - target[0], proj_max - target[1]]
            rows.append(
                {
                    "frame_idx": idx,
                    "side": hand.get("side"),
                    "detector_score": float(hand.get("detector_score", np.nan)),
                    "point_count": int(points.shape[0]),
                    "depth_median_m": float(np.median(points[:, 2])),
                    "depth_p05_m": float(np.percentile(points[:, 2], 5)),
                    "depth_p95_m": float(np.percentile(points[:, 2], 95)),
                    "projected_bbox_xyxy": [float(proj_min[0]), float(proj_min[1]), float(proj_max[0]), float(proj_max[1])],
                    "detector_bbox_xyxy": [float(v) for v in box],
                    "bbox_residual_px": [float(v) for v in bbox_residual],
                    "bbox_residual_l2_px": float(np.linalg.norm(bbox_residual)),
                    "bbox_residual_max_abs_px": float(np.max(np.abs(bbox_residual))),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    rows = hand_rows(annotations, args.frame_start, args.frame_end)
    if not rows:
        raise RuntimeError("no hand rows available for reprojection diagnostic")
    depths = np.asarray([row["depth_median_m"] for row in rows], dtype=float)
    residual_l2 = np.asarray([row["bbox_residual_l2_px"] for row in rows], dtype=float)
    residual_max = np.asarray([row["bbox_residual_max_abs_px"] for row in rows], dtype=float)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "hand_rows": len(rows),
        "depth_median_m": summarize(depths),
        "bbox_residual_l2_px": summarize(residual_l2),
        "bbox_residual_max_abs_px": summarize(residual_max),
        "interpretation": (
            "This checks whether the source-camera MANO translation remains consistent with its detector box. "
            "Small 2D bbox residuals with large hand-object depth conflict imply that contact cannot be fixed "
            "by arbitrary MANO depth shifts unless reprojection constraints are included."
        ),
        "rows_preview": rows[:80],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
