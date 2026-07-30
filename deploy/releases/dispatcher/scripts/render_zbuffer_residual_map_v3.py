#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json, load_mesh_archive
from render_mesh_zbuffer_qc_v3 import summarize, triangle_zbuffer


def component_summary(mask: np.ndarray, residual: np.ndarray, max_components: int) -> list[dict]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        take = labels == label
        values = residual[take]
        rows.append(
            {
                "area_px": area,
                "bbox_xywh": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "centroid_xy": [float(centroids[label][0]), float(centroids[label][1])],
                "signed_m": summarize(values),
                "abs_m": summarize(np.abs(values)),
            }
        )
    rows.sort(key=lambda row: row["area_px"], reverse=True)
    return rows[: int(max_components)]


def color_residual(residual: np.ndarray, valid: np.ndarray, max_abs_m: float) -> np.ndarray:
    scaled = np.zeros_like(residual, dtype=np.float32)
    scaled[valid] = np.clip(residual[valid] / float(max_abs_m), -1.0, 1.0)
    positive = np.clip(scaled, 0.0, 1.0)
    negative = np.clip(-scaled, 0.0, 1.0)
    color = np.zeros((*residual.shape, 3), dtype=np.uint8)
    color[..., 2] = (positive * 255.0).astype(np.uint8)
    color[..., 0] = (negative * 255.0).astype(np.uint8)
    color[..., 1] = ((1.0 - np.maximum(positive, negative)) * 80.0).astype(np.uint8)
    return color


def draw_label(image: np.ndarray, text: str, y: int) -> None:
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    entries = manifest.get("frames")
    frames = annotations.get("frames")
    if not isinstance(entries, list) or not isinstance(frames, list):
        raise RuntimeError("manifest and annotations must contain frames lists")
    annotation_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    meshes = load_mesh_archive(args.mesh_archive)
    depth_archive = load_depth_archive(args.metric_depth_npz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        if frame_idx not in annotation_by_idx:
            raise RuntimeError(f"annotations lack frame {frame_idx}")
        if frame_idx not in meshes:
            raise RuntimeError(f"mesh archive lacks frame {frame_idx}")
        if frame_idx not in depth_archive:
            raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            raise RuntimeError(f"failed to read RGB/mask for frame {frame_idx}")
        object_mask = mask > 0
        depth_m = np.asarray(depth_archive[frame_idx], dtype=np.float64)
        if depth_m.shape != object_mask.shape:
            raise RuntimeError(f"depth shape {depth_m.shape} does not match mask shape {object_mask.shape}")
        vertices_world, faces = meshes[frame_idx]
        annotation = annotation_by_idx[frame_idx]
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        K = intrinsics_for_frame(args, entry, annotation)
        vertices_camera = camera_points(vertices_world, T_world_camera)
        z = vertices_camera[:, 2]
        uv = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
        positive = z > 0.0
        uv[positive, 0] = K[0, 0] * vertices_camera[positive, 0] / z[positive] + K[0, 2]
        uv[positive, 1] = K[1, 1] * vertices_camera[positive, 1] / z[positive] + K[1, 2]
        zbuf = triangle_zbuffer(object_mask.shape, uv, z, faces, args.max_faces)
        valid = np.isfinite(zbuf) & object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
        residual = np.full_like(depth_m, np.nan, dtype=np.float64)
        residual[valid] = zbuf[valid].astype(np.float64) - depth_m[valid]
        high_far = valid & (residual > float(args.high_error_m))
        high_near = valid & (residual < -float(args.high_error_m))
        color = color_residual(residual, valid, float(args.color_max_abs_m))
        canvas = rgb.copy()
        canvas[valid] = cv2.addWeighted(canvas, 0.45, color, 0.55, 0)[valid]
        contour_mask = object_mask.astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        for high_mask, bgr in ((high_far, (0, 0, 255)), (high_near, (255, 0, 0))):
            high_contours, _ = cv2.findContours(high_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, high_contours, -1, bgr, 2, cv2.LINE_AA)
        values = residual[valid]
        draw_label(
            canvas,
            f"frame {frame_idx} residual zbuf-depth: red far, blue near, abs med {np.median(np.abs(values)):.3f}m p95 {np.percentile(np.abs(values),95):.3f}m",
            32,
        )
        draw_label(
            canvas,
            f">{args.high_error_m:.3f}m far {np.mean(high_far[valid]):.3f}, near {np.mean(high_near[valid]):.3f}",
            62,
        )
        out_path = args.output_dir / f"frame_{frame_idx:06d}_residual.png"
        cv2.imwrite(str(out_path), canvas)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "image": str(out_path),
                "samples": int(np.count_nonzero(valid)),
                "signed_m": summarize(values),
                "abs_m": summarize(np.abs(values)),
                "far_fraction_over_threshold": float(np.mean(high_far[valid])),
                "near_fraction_over_threshold": float(np.mean(high_near[valid])),
                "far_components": component_summary(high_far, residual, int(args.max_components)),
                "near_components": component_summary(high_near, residual, int(args.max_components)),
            }
        )
    if not rows:
        raise RuntimeError("no frames rendered")
    report = {
        "status": "ok",
        "method": "zbuffer_residual_map_v3",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "rows": rows,
        "summary": {
            "abs_median_m": summarize([row["abs_m"].get("median", np.nan) for row in rows]),
            "abs_p95_m": summarize([row["abs_m"].get("p95", np.nan) for row in rows]),
            "far_fraction_over_threshold": summarize([row["far_fraction_over_threshold"] for row in rows]),
            "near_fraction_over_threshold": summarize([row["near_fraction_over_threshold"] for row in rows]),
        },
        "parameters": {
            "high_error_m": float(args.high_error_m),
            "color_max_abs_m": float(args.color_max_abs_m),
            "max_faces": None if args.max_faces is None else int(args.max_faces),
        },
    }
    (args.output_dir / "qc_zbuffer_residual_map_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--high-error-m", type=float, default=0.040)
    parser.add_argument("--color-max-abs-m", type=float, default=0.080)
    parser.add_argument("--max-components", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_faces is not None and int(args.max_faces) <= 0:
        args.max_faces = None
    run(args)


if __name__ == "__main__":
    main()
