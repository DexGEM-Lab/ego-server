#!/usr/bin/env python3
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


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
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
        tree = cKDTree(model_world)
        _, idx = tree.query(observed_world, k=1, workers=-1)
        src = canonical_samples[idx]
        r_new, t_new = rigid_umeyama(src, observed_world)
        r, t = r_new, t_new
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6000)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    completion = load_json(args.completion_report)
    mesh_path = Path(completion["outputs"]["completed_mesh_labeled"])
    mesh = load_mesh(mesh_path)
    canonical_samples = deterministic_sample_mesh(mesh, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_pose = 0
    missing_observed = 0
    for frame in annotations.get("frames", []):
        frame_idx = int(frame.get("frame_idx"))
        obj = None
        for candidate in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if candidate.get("object_id") == args.object_id:
                obj = candidate
                break
        if obj is None:
            continue
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
        pose = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else {}
        r = np.asarray(pose.get("rotation_world_from_canonical_matrix") or [], dtype=float)
        t = np.asarray(pose.get("translation_world_m") or [], dtype=float)
        if r.shape != (3, 3) or t.shape != (3,):
            missing_pose += 1
            rows.append({"frame_idx": frame_idx, "status": "missing_initial_graph_pose"})
            continue
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < 3:
            missing_observed += 1
            rows.append({
                "frame_idx": frame_idx,
                "status": "no_current_visible_depth_samples_pose_carried_from_graph",
                "rotation_world_from_completed_canonical_matrix": r.astype(float).tolist(),
                "translation_world_m": t.astype(float).tolist(),
            })
            continue
        fit = fit_frame_pose(canonical_samples, observed, r, t, args.iterations)
        fit.update({
            "frame_idx": frame_idx,
            "status": "fit_to_visible_depth_samples",
            "object_id": args.object_id,
            "visible_sample_count": int(len(observed)),
            "initial_pose_source": pose.get("pose_source"),
            "initial_pose_observation_residual_norm": pose.get("pose_observation_residual_norm"),
        })
        rows.append(fit)

    residuals = [row["observed_to_mesh_final"]["median_m"] for row in rows if row.get("status") == "fit_to_visible_depth_samples" and row["observed_to_mesh_final"]["median_m"] is not None]
    report = {
        "method": "fit_v18_compact_rigid_object_pose",
        "status": "ok",
        "claim_scope": "Per-frame completed-mesh pose is initialized from V18 graph SE3 and refit only against current visible depth samples; hidden TRELLIS faces do not create pose observations by themselves.",
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "completion_report": str(args.completion_report),
            "completed_mesh": str(mesh_path),
        },
        "sample_count": int(len(canonical_samples)),
        "iterations": int(args.iterations),
        "frame_count": len(rows),
        "fit_frame_count": sum(1 for r in rows if r.get("status") == "fit_to_visible_depth_samples"),
        "missing_pose_count": missing_pose,
        "missing_visible_depth_sample_count": missing_observed,
        "final_observed_to_mesh_median_summary_m": {
            "median": float(np.median(residuals)) if residuals else None,
            "p90": float(np.percentile(residuals, 90)) if residuals else None,
            "max": float(np.max(residuals)) if residuals else None,
        },
        "pose_rows": rows,
    }
    out_path = args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "object_id", "frame_count", "fit_frame_count", "missing_pose_count", "missing_visible_depth_sample_count", "final_observed_to_mesh_median_summary_m"]}, indent=2))


if __name__ == "__main__":
    main()
