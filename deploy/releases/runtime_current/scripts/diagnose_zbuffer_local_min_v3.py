#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json, load_mesh_archive
from render_mesh_zbuffer_qc_v3 import summarize, triangle_zbuffer


def min_filter_zbuffer(zbuf: np.ndarray, radius_px: int) -> np.ndarray:
    if int(radius_px) <= 0:
        return zbuf.copy()
    kernel = np.ones((2 * int(radius_px) + 1, 2 * int(radius_px) + 1), dtype=np.uint8)
    filled = np.where(np.isfinite(zbuf), zbuf, np.inf).astype(np.float32)
    filtered = cv2.erode(filled, kernel)
    filtered[~np.isfinite(filtered)] = np.inf
    return filtered


def residual_summary(zbuf: np.ndarray, object_mask: np.ndarray, depth_m: np.ndarray) -> dict:
    silhouette = np.isfinite(zbuf)
    valid = silhouette & object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
    err = zbuf[valid].astype(np.float64) - depth_m[valid]
    return {
        "samples": int(len(err)),
        "signed_m": summarize(err),
        "abs_m": summarize(np.abs(err)),
        "closer_than_depth_fraction_5mm": float(np.mean(err < -0.005)) if len(err) else None,
        "farther_than_depth_fraction_5mm": float(np.mean(err > 0.005)) if len(err) else None,
    }


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
    radii = sorted({int(radius) for radius in args.radius_px})
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
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask for frame {frame_idx}")
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
        row = {"frame_idx": int(frame_idx), "radii": {}}
        for radius in radii:
            row["radii"][str(radius)] = residual_summary(min_filter_zbuffer(zbuf, radius), object_mask, depth_m)
        rows.append(row)
    if not rows:
        raise RuntimeError("no rows generated")
    radius_summary = {}
    for radius in radii:
        key = str(radius)
        radius_summary[key] = {
            "abs_median_m": summarize([row["radii"][key]["abs_m"].get("median", np.nan) for row in rows]),
            "abs_p95_m": summarize([row["radii"][key]["abs_m"].get("p95", np.nan) for row in rows]),
            "farther_than_depth_fraction_5mm": summarize(
                [row["radii"][key]["farther_than_depth_fraction_5mm"] for row in rows]
            ),
            "closer_than_depth_fraction_5mm": summarize(
                [row["radii"][key]["closer_than_depth_fraction_5mm"] for row in rows]
            ),
        }
    report = {
        "status": "ok",
        "method": "zbuffer_local_min_diagnostic_v3",
        "claim_tested": "high positive z-buffer residuals are caused by local surface/rasterization gaps if a small local z-minimum collapses them",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "rows": rows,
        "summary_by_radius_px": radius_summary,
        "parameters": {
            "radius_px": radii,
            "max_faces": None if args.max_faces is None else int(args.max_faces),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--radius-px", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_faces is not None and int(args.max_faces) <= 0:
        args.max_faces = None
    run(args)


if __name__ == "__main__":
    main()
