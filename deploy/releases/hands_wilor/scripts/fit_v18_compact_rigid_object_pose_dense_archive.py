#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Refit compact-rigid object pose using full visible-surface archive vertices.

`fit_v18_compact_rigid_object_pose.py` consumes the inline annotation preview of
visible vertices, which can be only 64 points. This variant resolves
`visible_geometry_candidate.archive_npz` and `archive_row_index` so the pose fit
uses the actual model-produced depth-backed visible surface for each frame.

The output is still a pose hypothesis, not accepted object geometry. It is meant
to test whether stale/sparse visible-surface association caused downstream MANO
constraint failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"not mesh: {path}")
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    rng = np.random.default_rng(1801)
    pts, _ = trimesh.sample.sample_surface(mesh, min(count, max(1, len(mesh.faces) * 2)), seed=rng)
    return np.asarray(pts, dtype=float)


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= int(max_points):
        return points.astype(float)
    idx = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return points[idx].astype(float)


def rigid_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(src) != len(dst) or len(src) < 3:
        raise RuntimeError("rigid fit requires matched arrays with >=3 points")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    xd = dst - mu_dst
    cov = xs.T @ xd / len(src)
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = mu_dst - (r @ mu_src)
    return r.astype(float), t.astype(float)


def apply_pose(points: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ r.T + t


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    if len(query) == 0 or len(target) == 0:
        return {"count": int(len(query)), "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
    d, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "p95_m": float(np.percentile(d, 95)),
        "mean_m": float(np.mean(d)),
        "max_m": float(np.max(d)),
    }


def fit_frame_pose(canonical_samples: np.ndarray, observed_world: np.ndarray, init_r: np.ndarray, init_t: np.ndarray, iterations: int) -> dict[str, Any]:
    r = init_r.copy()
    t = init_t.copy()
    trace = []
    for _ in range(iterations):
        model_world = apply_pose(canonical_samples, r, t)
        _, idx = cKDTree(model_world).query(observed_world, k=1, workers=-1)
        src = canonical_samples[idx]
        r, t = rigid_umeyama(src, observed_world)
        trace.append(nearest_summary(observed_world, apply_pose(canonical_samples, r, t)))
    model_world = apply_pose(canonical_samples, r, t)
    init_model_world = apply_pose(canonical_samples, init_r, init_t)
    return {
        "rotation_world_from_completed_canonical_matrix": r.astype(float).tolist(),
        "translation_world_m": t.astype(float).tolist(),
        "observed_to_mesh_initial": nearest_summary(observed_world, init_model_world),
        "observed_to_mesh_final": nearest_summary(observed_world, model_world),
        "mesh_to_observed_final": nearest_summary(model_world, observed_world),
        "icp_trace": trace,
    }


def load_archive_vertices(cache: dict[Path, Any], archive_path: Path, row_index: int) -> np.ndarray:
    if archive_path not in cache:
        z = np.load(archive_path, allow_pickle=True)
        cache[archive_path] = {
            "vertex_offsets": np.asarray(z["vertex_offsets"], dtype=np.int64),
            "vertices": np.asarray(z["vertices"], dtype=np.float32),
        }
    payload = cache[archive_path]
    offsets = payload["vertex_offsets"]
    vertices = payload["vertices"]
    start = int(offsets[int(row_index)])
    end = int(offsets[int(row_index) + 1])
    return np.asarray(vertices[start:end], dtype=float)


def observed_world_vertices(geom: dict[str, Any], archive_cache: dict[Path, Any], max_observed_points: int) -> tuple[np.ndarray, dict[str, Any]]:
    archive = geom.get("archive_npz")
    row_index = geom.get("archive_row_index")
    if isinstance(archive, str) and row_index is not None and Path(archive).exists():
        full = load_archive_vertices(archive_cache, Path(archive), int(row_index))
        sampled = downsample_points(full, int(max_observed_points))
        return sampled, {
            "visible_surface_source": "archive_npz_full_vertices_downsampled",
            "archive_npz": str(archive),
            "archive_row_index": int(row_index),
            "archive_vertex_count": int(len(full)),
            "visible_sample_count_used": int(len(sampled)),
        }
    inline = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
    if inline.ndim == 2 and inline.shape[1] == 3:
        return inline, {"visible_surface_source": "inline_world_vertices_sample_m", "visible_sample_count_used": int(len(inline))}
    return np.zeros((0, 3), dtype=float), {"visible_surface_source": "missing_visible_surface"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--max-observed-points", type=int, default=2500)
    parser.add_argument("--iterations", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_json(args.annotations)
    mesh = load_mesh(args.completed_mesh)
    canonical_samples = deterministic_sample_mesh(mesh, int(args.sample_count))
    archive_cache: dict[Path, Any] = {}
    rows = []
    missing_pose = 0
    missing_observed = 0
    archive_used = 0
    for frame in annotations.get("frames", []) if isinstance(annotations.get("frames"), list) else []:
        frame_idx = int(frame.get("frame_idx"))
        obj = None
        for candidate in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if candidate.get("object_id") == args.object_id:
                obj = candidate
                break
        if obj is None:
            continue
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        observed, obs_meta = observed_world_vertices(geom, archive_cache, int(args.max_observed_points))
        pose = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
        r = np.asarray(pose.get("rotation_world_from_canonical_matrix") or [], dtype=float)
        t = np.asarray(pose.get("translation_world_m") or [], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,):
            missing_pose += 1
            rows.append({"frame_idx": frame_idx, "status": "missing_initial_graph_pose", **obs_meta})
            continue
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 3:
            missing_observed += 1
            rows.append({
                "frame_idx": frame_idx,
                "status": "no_current_visible_depth_samples_pose_carried_from_graph",
                "rotation_world_from_completed_canonical_matrix": r.astype(float).tolist(),
                "translation_world_m": t.astype(float).tolist(),
                **obs_meta,
            })
            continue
        if obs_meta.get("visible_surface_source") == "archive_npz_full_vertices_downsampled":
            archive_used += 1
        fit = fit_frame_pose(canonical_samples, observed, r, t, int(args.iterations))
        fit.update({
            "frame_idx": frame_idx,
            "status": "fit_to_visible_depth_archive_vertices",
            "object_id": args.object_id,
            "initial_pose_source": pose.get("pose_source"),
            "initial_pose_observation_residual_norm": pose.get("pose_observation_residual_norm"),
            **obs_meta,
        })
        rows.append(fit)
    residuals = [row["observed_to_mesh_final"]["median_m"] for row in rows if row.get("status") == "fit_to_visible_depth_archive_vertices" and row["observed_to_mesh_final"]["median_m"] is not None]
    report = {
        "method": "fit_v18_compact_rigid_object_pose_dense_archive",
        "status": "ok",
        "claim_scope": "Per-frame compact-rigid pose refit using full visible-surface archive vertices rather than the inline preview. This is a pose hypothesis, not accepted complete geometry.",
        "object_id": args.object_id,
        "inputs": {"annotations": str(args.annotations), "completed_mesh": str(args.completed_mesh)},
        "sample_count": int(len(canonical_samples)),
        "max_observed_points": int(args.max_observed_points),
        "iterations": int(args.iterations),
        "frame_count": int(len(rows)),
        "fit_frame_count": sum(1 for row in rows if row.get("status") == "fit_to_visible_depth_archive_vertices"),
        "archive_visible_surface_used_count": int(archive_used),
        "missing_pose_count": int(missing_pose),
        "missing_visible_depth_sample_count": int(missing_observed),
        "final_observed_to_mesh_median_summary_m": {
            "median": float(np.median(residuals)) if residuals else None,
            "p90": float(np.percentile(residuals, 90)) if residuals else None,
            "max": float(np.max(residuals)) if residuals else None,
        },
        "pose_rows": rows,
    }
    out_path = args.output_dir / "v18_compact_rigid_object_pose_fit_dense_archive_report.json"
    write_json(out_path, report)
    print(json.dumps({k: report[k] for k in ["status", "object_id", "frame_count", "fit_frame_count", "archive_visible_surface_used_count", "missing_pose_count", "missing_visible_depth_sample_count", "final_observed_to_mesh_median_summary_m"]}, indent=2))


if __name__ == "__main__":
    main()
