#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, load_depth_archive, manifest_by_frame
from optimize_mesh_prior_pose_graph_v3 import sample_rows
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {path}")
    if mask.shape != expected_shape:
        mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def sample_mask_points(mask: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray, count: int, seed: int) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError("mask has no valid depth samples")
    coords = np.c_[xs, ys]
    if len(coords) > int(count):
        rng = np.random.default_rng(int(seed))
        coords = coords[rng.choice(len(coords), size=int(count), replace=False)]
    z = depth_m[coords[:, 1], coords[:, 0]].astype(np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    x = (coords[:, 0].astype(np.float64) - cx) * z / fx
    y = (coords[:, 1].astype(np.float64) - cy) * z / fy
    points = np.c_[x, y, z]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise RuntimeError("sampled mask-depth points are invalid")
    return points.astype(np.float64)


def intrinsics_for(annotation: dict, depth_intrinsics: np.ndarray, source: str) -> np.ndarray:
    if source == "annotation-vggt":
        vals = annotation.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", [])
        intr = np.asarray(vals, dtype=np.float64)
    elif source == "metric-depth":
        intr = np.asarray(depth_intrinsics, dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported intrinsics source {source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid {source} intrinsics for frame {annotation.get('frame_idx')}")
    return intr


def mesh_vertices_for(meshes: dict[int, tuple[np.ndarray, np.ndarray]], frame_idx: int) -> np.ndarray:
    if frame_idx not in meshes:
        raise RuntimeError(f"mesh archive lacks frame {frame_idx}")
    vertices, _faces = meshes[frame_idx]
    return np.asarray(vertices, dtype=np.float64)


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.mesh_archive)
    rows = []
    all_abs_depth_errors = []
    all_mask_hits = []
    all_graph_surface_distance = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in manifest or frame_idx not in depths:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intr = intrinsics_for(annotation, depth_intrinsics, args.intrinsics_source)
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        graph_vertices_world = mesh_vertices_for(graph_meshes, frame_idx)
        graph_vertices_camera = camera_points(graph_vertices_world, T_world_camera)
        surface_camera = sample_rows(graph_vertices_camera, int(args.samples), int(args.seed) + frame_idx)
        observed_points = sample_mask_points(mask, depth_m, intr, int(args.samples), int(args.seed) + 5000 + frame_idx)
        d_observed_to_graph, _ = cKDTree(surface_camera).query(observed_points, k=1)
        positive = surface_camera[:, 2] > float(args.min_depth_m)
        uv = np.full((len(surface_camera), 2), np.nan, dtype=np.float64)
        if np.any(positive):
            uv[positive] = project_points(surface_camera[positive], intr)
        rounded = np.rint(uv).astype(np.int64)
        in_bounds = (
            positive
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < mask.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < mask.shape[0])
        )
        mask_hit = np.zeros(len(surface_camera), dtype=bool)
        depth_error = np.full(len(surface_camera), np.nan, dtype=np.float64)
        if np.any(in_bounds):
            y = rounded[in_bounds, 1]
            x = rounded[in_bounds, 0]
            mask_hit[in_bounds] = mask[y, x]
            metric_depth = depth_m[y, x]
            valid_depth = np.isfinite(metric_depth) & (metric_depth > 0.0)
            valid_indices = np.flatnonzero(in_bounds)[valid_depth]
            depth_error[valid_indices] = surface_camera[valid_indices, 2] - metric_depth[valid_depth]
        inside_depth = depth_error[mask_hit & np.isfinite(depth_error)]
        outside_depth = depth_error[(~mask_hit) & np.isfinite(depth_error)]
        row = {
            "frame_idx": int(frame_idx),
            "observed_to_graph_surface_distance_m": summarize(d_observed_to_graph),
            "projected_in_bounds_fraction": float(np.mean(in_bounds)),
            "projected_inside_mask_fraction": float(np.mean(mask_hit[in_bounds])) if np.any(in_bounds) else 0.0,
            "inside_mask_depth_error_m": summarize(inside_depth),
            "inside_mask_abs_depth_error_m": summarize(np.abs(inside_depth)),
            "outside_mask_abs_depth_error_m": summarize(np.abs(outside_depth)),
            "mask_area_px": int(np.count_nonzero(mask)),
        }
        rows.append(row)
        all_abs_depth_errors.extend(np.abs(inside_depth).astype(float).tolist())
        all_graph_surface_distance.extend(d_observed_to_graph.astype(float).tolist())
        if np.any(in_bounds):
            all_mask_hits.extend(mask_hit[in_bounds].astype(float).tolist())
    if not rows:
        raise RuntimeError("no frames diagnosed")
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_complete_mesh_surface_conflict_v3",
        "claim_tested": "whether the solved complete mesh pose explains the visible metric-depth surface",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "frames": [row["frame_idx"] for row in rows],
        "summary": {
            "observed_to_graph_surface_distance_m": summarize(all_graph_surface_distance),
            "projected_inside_mask_fraction": summarize(all_mask_hits),
            "inside_mask_abs_depth_error_m": summarize(all_abs_depth_errors),
        },
        "rows": rows,
    }
    save_json(args.output_json, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
