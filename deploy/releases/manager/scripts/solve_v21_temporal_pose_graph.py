#!/usr/bin/env python3
"""V21 temporal rigid pose factor graph with surface-fit data term.

Design amendment relative to solve_v19_rigid_object_pose_graph.py:
The V19 graph optimizes correction deltas (nonpenetration correction stage)
and has no surface-fit data term in its residuals. It is a no-op without
nonpenetration constraint targets.

This graph optimizes physical poses (R, t) directly as variables, with:
1. Surface-fit data term: measures how well the mesh at the estimated pose
   aligns with observed surface points (per-frame nearest-neighbor residual).
2. Temporal smoothness: penalizes pose jumps between consecutive frames,
   trading off against data fit.
3. Temporal acceleration: penalizes acceleration for smoother trajectories.

This is the trajectory-smoothing factor graph that V21 Section 10 requires:
"temporal/object factor graph correction."

Variables: per-frame rotation (axis-angle, 3 params) + translation (3 params).
Total: 6 * N_frame parameters.

Output:
  measurements/object_geometry_mesh_pose/<object_id>/v21_pose_graph/
    v21_temporal_pose_graph_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import trimesh
except ImportError:
    raise RuntimeError("trimesh required")


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def sample_mesh(mesh, count, seed=42):
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(pts, dtype=float)


def unpack_poses(x, n):
    """Unpack flat parameter vector into rotations (rotvec) and translations."""
    arr = x.reshape(n, 6)
    return arr[:, :3], arr[:, 3:6]


def apply_pose(points, rotvec, trans):
    """Apply SE(3) transform: R(rotvec) @ points + trans."""
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return points @ R.T + trans


def nearest_neighbor_residual(source_pts, target_tree):
    """Compute per-point nearest-neighbor distances from source to target."""
    dists, _ = target_tree.query(source_pts, k=1)
    return dists


def residual_vector(x, observations, mesh_samples, args):
    """Compute the full residual vector for least_squares.

    Terms:
    1. Surface fit: for each frame, transform mesh samples to estimated pose,
       compute NN distance to observed points. Robust Huber-like scaling.
    2. Temporal smoothness: pose difference between consecutive frames.
    3. Temporal acceleration: second difference of pose.
    """
    n = len(observations)
    rotvecs, transs = unpack_poses(x, n)
    residuals = []

    # 1. Surface-fit data term (use median NN distance per frame, not per-point,
    #    to avoid the data term dominating by sheer count)
    for i, obs in enumerate(observations):
        mesh_world = apply_pose(mesh_samples, rotvecs[i], transs[i])
        nn_dists = nearest_neighbor_residual(mesh_world, obs["target_tree"])
        median_d = float(np.median(nn_dists))
        # Single residual per frame proportional to median surface distance
        sigma_data = max(float(args.min_data_sigma_m), 0.01)
        residuals.append(np.array([median_d / sigma_data]))
        # Also add p90 as a second residual for robustness
        p90_d = float(np.percentile(nn_dists, 90))
        residuals.append(np.array([p90_d / (3.0 * sigma_data)]))

    # 2. Temporal smoothness (translation + rotation)
    for i in range(1, n):
        gap = max(1, observations[i]["frame_idx"] - observations[i-1]["frame_idx"])
        scale = math.sqrt(gap)
        # Translation difference
        dt = (transs[i] - transs[i-1]) / (float(args.sigma_translation_step_m) * scale)
        residuals.append(dt)
        # Rotation difference (rotvec space approximation)
        dr = (rotvecs[i] - rotvecs[i-1]) / (float(args.sigma_rotation_step_rad) * scale)
        residuals.append(dr)

    # 3. Temporal acceleration (second difference)
    for i in range(1, n - 1):
        gap0 = max(1, observations[i]["frame_idx"] - observations[i-1]["frame_idx"])
        gap1 = max(1, observations[i+1]["frame_idx"] - observations[i]["frame_idx"])
        scale = math.sqrt(max(gap0, gap1))
        v0_t = (transs[i] - transs[i-1]) / gap0
        v1_t = (transs[i+1] - transs[i]) / gap1
        residuals.append((v1_t - v0_t) / (float(args.sigma_translation_accel_m) * scale))
        v0_r = (rotvecs[i] - rotvecs[i-1]) / gap0
        v1_r = (rotvecs[i+1] - rotvecs[i]) / gap1
        residuals.append((v1_r - v0_r) / (float(args.sigma_rotation_accel_rad) * scale))

    return np.concatenate([r.reshape(-1) for r in residuals]).astype(float)


def build_observations(annotations, icp_report, args):
    """Build per-frame observations from annotations + ICP pose report.

    Each observation has:
    - frame_idx
    - target_tree: cKDTree of observed surface points
    - init_rotvec, init_trans: initial pose from ICP
    - observed_point_count
    """
    frames_by_idx = {int(f["frame_idx"]): f for f in annotations.get("frames", [])}
    pose_rows = {int(r["frame_idx"]): r for r in icp_report.get("pose_rows", [])}

    observations = []
    skipped = []

    for frame_idx in sorted(pose_rows.keys()):
        if args.frame_start is not None and frame_idx < args.frame_start:
            continue
        if args.frame_end is not None and frame_idx > args.frame_end:
            continue

        row = pose_rows[frame_idx]
        if str(row.get("status", "")) not in ("fit_to_visible_depth_samples", "fit_to_visible_depth_archive_vertices"):
            continue

        frame = frames_by_idx.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "frame not in annotations"})
            continue

        # Find object
        obj = None
        for o in frame.get("objects", []):
            if o.get("object_id") == args.object_id:
                obj = o
                break
        if obj is None:
            skipped.append({"frame_idx": frame_idx, "reason": "object not in frame"})
            continue

        geom = obj.get("visible_geometry_candidate", {})
        observed = np.asarray(geom.get("world_vertices_sample_m") or geom.get("camera_vertices_sample_m") or [], dtype=float)
        if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < int(args.min_visible_points):
            skipped.append({"frame_idx": frame_idx, "reason": f"insufficient observed points: {observed.shape if observed.ndim == 2 else 'scalar'}"})
            continue

        # Initial pose from ICP
        R_init = np.asarray(row.get("rotation_world_from_completed_canonical_matrix") or row.get("rotation_world_from_canonical_matrix"), dtype=float)
        t_init = np.asarray(row.get("translation_world_m"), dtype=float)
        if R_init.shape != (3, 3) or t_init.shape != (3,):
            skipped.append({"frame_idx": frame_idx, "reason": "invalid initial pose"})
            continue

        rotvec_init = Rotation.from_matrix(R_init).as_rotvec()

        observations.append({
            "frame_idx": frame_idx,
            "target_tree": cKDTree(observed),
            "observed_points": observed,
            "init_rotvec": rotvec_init,
            "init_trans": t_init,
            "observed_point_count": len(observed),
            "icp_residual_median": (row.get("observed_to_mesh_final") or {}).get("median_m"),
        })

    if len(observations) < int(args.min_graph_frames):
        raise RuntimeError(f"only {len(observations)} usable observations; skipped={skipped[:8]}")

    return observations, skipped


def compute_surface_metrics(observations, mesh_samples, rotvecs, transs):
    """Compute per-frame surface fit metrics."""
    obs_to_mesh = []
    mesh_to_obs = []
    for i, obs in enumerate(observations):
        mesh_world = apply_pose(mesh_samples, rotvecs[i], transs[i])
        # obs -> mesh
        d1, _ = cKDTree(mesh_world).query(obs["observed_points"], k=1)
        # mesh -> obs
        d2, _ = obs["target_tree"].query(mesh_world, k=1)
        obs_to_mesh.append(float(np.median(d1)))
        mesh_to_obs.append(float(np.median(d2)))
    return {
        "observed_to_mesh_median_m": obs_to_mesh,
        "mesh_to_observed_median_m": mesh_to_obs,
    }


def run(args):
    started = time.time()
    run_root = Path(args.run_root)

    # Load inputs
    annotations = load_json(Path(args.annotations))
    icp_report = load_json(Path(args.icp_report))
    mesh = load_mesh(Path(args.mesh))

    mesh_samples = sample_mesh(mesh, int(args.mesh_sample_count), seed=42)
    object_radius = float(np.linalg.norm(mesh.extents) / 2.0)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh_samples)} samples, radius={object_radius:.4f}m", flush=True)

    observations, skipped = build_observations(annotations, icp_report, args)
    print(f"Observations: {len(observations)} frames, skipped: {len(skipped)}", flush=True)

    # Initial guess: ICP poses
    n = len(observations)
    x0 = np.zeros(n * 6, dtype=float)
    for i, obs in enumerate(observations):
        x0[i*6:i*6+3] = obs["init_rotvec"]
        x0[i*6+3:i*6+6] = obs["init_trans"]

    # Compute initial surface metrics
    rv0, t0 = unpack_poses(x0, n)
    metrics_before = compute_surface_metrics(observations, mesh_samples, rv0, t0)
    med_before = float(np.median(metrics_before["observed_to_mesh_median_m"]))

    print(f"Initial surface fit: median={med_before:.6f}m", flush=True)

    # Optimize
    before_residuals = residual_vector(x0, observations, mesh_samples, args)
    print(f"Initial residual RMS: {np.sqrt(np.mean(before_residuals**2)):.6f}", flush=True)

    result = least_squares(
        lambda x: residual_vector(x, observations, mesh_samples, args),
        x0,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )

    after_residuals = residual_vector(result.x, observations, mesh_samples, args)
    rv1, t1 = unpack_poses(result.x, n)
    metrics_after = compute_surface_metrics(observations, mesh_samples, rv1, t1)
    med_after = float(np.median(metrics_after["observed_to_mesh_median_m"]))

    print(f"Final surface fit: median={med_after:.6f}m", flush=True)
    print(f"Final residual RMS: {np.sqrt(np.mean(after_residuals**2)):.6f}", flush=True)

    # Build pose rows for downstream consumption
    pose_rows = []
    for i, obs in enumerate(observations):
        R = Rotation.from_rotvec(rv1[i]).as_matrix()
        pose_rows.append({
            "frame_idx": obs["frame_idx"],
            "status": "temporal_pose_graph_corrected",
            "rotation_world_from_completed_canonical_matrix": R.tolist(),
            "translation_world_m": t1[i].tolist(),
            "rotation_rotvec": rv1[i].tolist(),
            "observed_point_count": obs["observed_point_count"],
            "surface_fit_median_m": metrics_after["observed_to_mesh_median_m"][i],
            "icp_initial_pose": {
                "rotation": obs["init_rotvec"].tolist(),
                "translation": obs["init_trans"].tolist(),
            },
        })

    # Temporal smoothness metrics
    trans_steps = np.linalg.norm(np.diff(t1, axis=0), axis=1)
    rot_steps = np.linalg.norm(np.diff(rv1, axis=0), axis=1)

    report = {
        "method": "solve_v21_temporal_pose_graph",
        "design_amendment": "V19's solve_v19_rigid_object_pose_graph optimizes correction deltas (nonpenetration correction stage) with no surface-fit data term. This graph optimizes physical poses directly with a surface-fit data term and temporal smoothness, as required by V21 Section 10.",
        "status": "ok" if result.success else "optimizer_incomplete",
        "object_id": args.object_id,
        "mesh": str(args.mesh),
        "mesh_sample_count": len(mesh_samples),
        "graph_frame_count": n,
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "residual_rms_before": float(np.sqrt(np.mean(before_residuals**2))),
            "residual_rms_after": float(np.sqrt(np.mean(after_residuals**2))),
        },
        "surface_fit_before": {
            "observed_to_mesh_median_m": med_before,
            "values": metrics_before["observed_to_mesh_median_m"],
        },
        "surface_fit_after": {
            "observed_to_mesh_median_m": med_after,
            "values": metrics_after["observed_to_mesh_median_m"],
        },
        "temporal_smoothness": {
            "translation_step_median_m": float(np.median(trans_steps)) if len(trans_steps) > 0 else None,
            "translation_step_p90_m": float(np.percentile(trans_steps, 90)) if len(trans_steps) > 0 else None,
            "rotation_step_median_rad": float(np.median(rot_steps)) if len(rot_steps) > 0 else None,
            "rotation_step_p90_rad": float(np.percentile(rot_steps, 90)) if len(rot_steps) > 0 else None,
        },
        "parameters": {
            "sigma_translation_step_m": float(args.sigma_translation_step_m),
            "sigma_rotation_step_rad": float(args.sigma_rotation_step_rad),
            "sigma_translation_accel_m": float(args.sigma_translation_accel_m),
            "sigma_rotation_accel_rad": float(args.sigma_rotation_accel_rad),
            "min_data_sigma_m": float(args.min_data_sigma_m),
            "mesh_sample_count": int(args.mesh_sample_count),
            "max_nfev": int(args.max_nfev),
        },
        "pose_rows": pose_rows,
        "elapsed_s": float(time.time() - started),
    }

    output_dir = run_root / "measurements" / "object_geometry_mesh_pose" / args.object_id / "v21_pose_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "v21_temporal_pose_graph_report.json", report)
    print(json.dumps({k: report[k] for k in ["status", "graph_frame_count", "optimizer", "surface_fit_before", "surface_fit_after", "temporal_smoothness"]}, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--icp-report", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--min-graph-frames", type=int, default=8)
    ap.add_argument("--min-visible-points", type=int, default=10)
    ap.add_argument("--sigma-translation-step-m", type=float, default=0.05)
    ap.add_argument("--sigma-rotation-step-rad", type=float, default=0.30)
    ap.add_argument("--sigma-translation-accel-m", type=float, default=0.03)
    ap.add_argument("--sigma-rotation-accel-rad", type=float, default=0.15)
    ap.add_argument("--min-data-sigma-m", type=float, default=0.005)
    ap.add_argument("--mesh-sample-count", type=int, default=800)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
