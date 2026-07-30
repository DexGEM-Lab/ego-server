#!/usr/bin/env python3
"""V21 rigid object pose fitting via ICP against observed surfaces.

For each frame with an accepted mask:
1. Backproject mask+depth to 3D camera-space observed points
2. ICP-fit the mesh candidate to the observed points
3. Output per-frame SE(3) pose (rotation + translation in camera space)

The mesh candidate is treated as rigid. The pose is estimated per-frame.
Temporal continuity is used as initialization (previous frame's pose).

Output:
  measurements/object_geometry_mesh_pose/<object_id>/v21_pose_fit.json
  measurements/object_geometry_mesh_pose/<object_id>/v21_pose_fit_qc.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from v21_mask_sources import resolve_current_object_mask_dir

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh required", file=sys.stderr)
    sys.exit(1)


def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def sample_mesh_surface(mesh, count=2000):
    pts, _ = trimesh.sample.sample_surface(mesh, count)
    return np.asarray(pts, dtype=np.float64)


def load_depth_and_intrinsics(npz_path):
    data = np.load(str(npz_path))
    return data["depth"], data["intrinsics_fx_fy_cx_cy"]


def backproject_frame(mask, depth_frame, intrinsics, outlier_removal=True):
    fx, fy, cx, cy = intrinsics
    ys, xs = np.where(mask > 127)
    if len(ys) < 10:
        return np.zeros((0, 3))
    zs = np.asarray(depth_frame[ys, xs], dtype=np.float64)
    valid = zs > 0.1
    xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

    if outlier_removal and len(zs_v) > 50:
        z_median = np.median(zs_v)
        z_iqr = np.percentile(zs_v, 75) - np.percentile(zs_v, 25)
        z_lo = z_median - 3 * max(z_iqr, 0.01)
        z_hi = z_median + 3 * max(z_iqr, 0.01)
        inlier = (zs_v >= z_lo) & (zs_v <= z_hi)
        xs_v, ys_v, zs_v = xs_v[inlier], ys_v[inlier], zs_v[inlier]

    xc = (xs_v.astype(np.float64) - cx) * zs_v / fx
    yc = (ys_v.astype(np.float64) - cy) * zs_v / fy
    return np.column_stack([xc, yc, zs_v])


def icp_fit(source_pts, target_pts, init_R=None, init_t=None, max_iterations=30, tolerance=1e-5):
    """ICP: find R, t that aligns source_pts to target_pts."""
    src = source_pts.copy()
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()

    prev_error = float('inf')
    tree = cKDTree(target_pts)

    for iteration in range(max_iterations):
        # Transform source
        transformed = src @ R.T + t

        # Find nearest neighbors
        dists, indices = tree.query(transformed, k=1)
        correspondences = target_pts[indices]

        # Filter by distance (reject > 2x median)
        median_dist = np.median(dists)
        threshold = max(5 * median_dist, 0.02)  # at least 2cm
        valid = dists < threshold
        if valid.sum() < 3:
            break

        src_valid = transformed[valid]
        corr_valid = correspondences[valid]

        # Compute optimal rotation and translation (Umeyama)
        mu_src = src_valid.mean(axis=0)
        mu_dst = corr_valid.mean(axis=0)
        src_centered = src_valid - mu_src
        dst_centered = corr_valid - mu_dst

        H = src_centered.T @ dst_centered / len(src_valid)
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        D[2, 2] = np.linalg.det(Vt.T @ U.T)
        R_new = Vt.T @ D @ U.T
        t_new = mu_dst - R_new @ mu_src

        # Update (incremental, not replace — helps with large initial misalignment)
        alpha = 0.5
        R = R_new @ R
        t = R_new @ t * alpha + t_new * (1 - alpha) + R_new @ (t - R_new.T @ t_new) * alpha  # approximate

        # Actually, just use R_new, t_new directly
        R = R_new @ R_init if iteration == 0 else R_new @ R
        # Simpler: accumulate
        # Let me redo this properly

        mean_error = np.median(dists[valid])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R, t, prev_error


def icp_fit_v2(source_pts, target_pts, init_R=None, init_t=None, max_iterations=30, tolerance=1e-5):
    """ICP: find R, t that aligns source_pts to target_pts.
    
    source_pts and target_pts are both in the same coordinate frame.
    Returns R, t such that R @ source_pts.T + t ≈ target_pts.
    """
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()

    prev_error = float('inf')
    tree = cKDTree(target_pts)

    for iteration in range(max_iterations):
        # Transform source
        transformed = (R @ source_pts.T).T + t

        # Find nearest neighbors
        dists, indices = tree.query(transformed, k=1)

        # Reject outliers: keep within reasonable distance
        median_dist = np.median(dists)
        threshold = max(3 * median_dist, 0.015)  # 1.5cm minimum
        valid = dists < threshold
        if valid.sum() < 10:
            break

        src_v = source_pts[valid]
        corr_v = target_pts[indices[valid]]

        # Umeyama alignment: solve for R_step, t_step such that
        # R_step @ src_v.T + t_step ≈ corr_v
        mu_s = src_v.mean(axis=0)
        mu_c = corr_v.mean(axis=0)
        s_c = src_v - mu_s
        c_c = corr_v - mu_c

        H = s_c.T @ c_c / len(src_v)
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        D[2, 2] = np.linalg.det(Vt.T @ U.T)
        R_step = Vt.T @ D @ U.T
        t_step = mu_c - R_step @ mu_s

        # Apply step: compose with current transform
        # new_transform(p) = R_step @ (R @ p + t) + t_step
        #                  = (R_step @ R) @ p + (R_step @ t + t_step)
        R = R_step @ R
        t = R_step @ t + t_step

        mean_error = float(np.median(dists[valid]))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Final residual
    transformed = (R @ source_pts.T).T + t
    dists, _ = tree.query(transformed, k=1)
    final_residual = float(np.median(dists))

    return R, t, final_residual


def run(args):
    run_root = Path(args.run_root)
    object_id = args.object_id
    repo_root = Path(args.repo_root)

    # Load mesh candidate
    mesh_path = run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_candidate.obj"
    mesh = load_mesh(mesh_path)
    canonical_pts = sample_mesh_surface(mesh, args.sample_count)
    print(f"Loaded mesh: {len(mesh.vertices)} verts, sampled {len(canonical_pts)} surface points", flush=True)

    # Center mesh at origin for canonical pose
    mesh_center = canonical_pts.mean(axis=0)
    canonical_pts_centered = canonical_pts - mesh_center

    # Load depth
    depth_npz = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
    depth, per_frame_intrinsics = load_depth_and_intrinsics(depth_npz)

    # Load manifest
    manifest = json.loads((run_root / "input" / "raw_frame_manifest" / "manifest.json").read_text())
    total_frames = len(manifest["frames"])

    # Find current V19/V21 mask files. The removed V20 local_grabcut branch is intentionally not used.
    mask_dir = resolve_current_object_mask_dir(run_root, object_id)
    mask_files = {}
    for mf in sorted(glob.glob(str(mask_dir / "*.png"))):
        fidx = int(Path(mf).stem)
        mask_files[fidx] = mf

    print(f"Masks found: {len(mask_files)} frames", flush=True)

    # Fit pose per-frame
    pose_rows = []
    prev_R = None
    prev_t = None

    for fi, fm in enumerate(manifest["frames"]):
        frame_idx = fm["frame_idx"]

        if frame_idx not in mask_files:
            if prev_R is not None:
                # Carry forward previous pose
                pose_rows.append({
                    "frame_idx": frame_idx,
                    "status": "no_mask_pose_carried",
                    "rotation_matrix": prev_R.tolist(),
                    "translation_m": prev_t.tolist(),
                })
            else:
                pose_rows.append({"frame_idx": frame_idx, "status": "no_mask_no_pose"})
            continue

        mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        intr = per_frame_intrinsics[frame_idx]
        observed = backproject_frame(mask, depth[frame_idx], intr)

        if len(observed) < 10:
            pose_rows.append({"frame_idx": frame_idx, "status": "insufficient_depth_points"})
            continue

        # Initialize from previous frame or centroid alignment
        # Both canonical and observed are centered at their centroids for ICP
        obs_center = observed.mean(axis=0)
        observed_centered = observed - obs_center

        if prev_R is not None:
            init_R = prev_R
            init_t = prev_t
        else:
            init_R = np.eye(3)
            init_t = np.zeros(3)  # both centered at origin

        # ICP on centered point clouds
        R, t_centered, residual = icp_fit_v2(canonical_pts_centered, observed_centered,
                                              init_R, init_t,
                                              max_iterations=args.icp_iterations)
        # Recover camera-frame translation: t = t_centered + obs_center
        t = t_centered + obs_center

        # Temporal smoothing: blend with previous pose if jump is large
        if prev_R is not None and prev_t is not None:
            t_jump = np.linalg.norm(t - prev_t)
            if t_jump > 0.05:  # 5cm jump
                alpha = 0.05 / max(t_jump, 1e-6)
                t = prev_t + alpha * (t - prev_t)

        pose_rows.append({
            "frame_idx": frame_idx,
            "status": "fit",
            "rotation_matrix": R.tolist(),
            "translation_m": t.tolist(),
            "icp_residual_m": residual,
            "observed_points": len(observed),
        })

        prev_R, prev_t = R, t

        if fi % 50 == 0:
            print(f"  Frame {frame_idx}: residual={residual:.4f}m, pts={len(observed)}", flush=True)

    # Summary
    fit_residuals = [r["icp_residual_m"] for r in pose_rows if r.get("status") == "fit"]
    fit_count = sum(1 for r in pose_rows if r.get("status") == "fit")

    output_dir = run_root / "measurements" / "object_geometry_mesh_pose" / object_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_data = {
        "schema": "v21_rigid_pose_fit.v0",
        "object_id": object_id,
        "run_root": str(run_root),
        "mesh_candidate": str(mesh_path),
        "mesh_center_offset_m": mesh_center.tolist(),
        "canonical_sample_count": len(canonical_pts),
        "frame_count": len(pose_rows),
        "pose_rows": pose_rows,
    }
    (output_dir / "v21_pose_fit.json").write_text(json.dumps(pose_data, indent=2))

    qc = {
        "status": "ok",
        "method": "v21_rigid_icp_pose_fit",
        "object_id": object_id,
        "total_frames": total_frames,
        "fit_frames": fit_count,
        "fit_rate": fit_count / max(1, total_frames),
        "residual_summary": {
            "median_m": float(np.median(fit_residuals)) if fit_residuals else None,
            "p90_m": float(np.percentile(fit_residuals, 90)) if fit_residuals else None,
            "max_m": float(np.max(fit_residuals)) if fit_residuals else None,
        } if fit_residuals else {},
        "candidate_state": "icp_pose_fitted_mesh_candidate",
        "next_required_step": "render_pose_fitted_mesh_overlay",
        "output": str(output_dir / "v21_pose_fit.json"),
    }
    (output_dir / "v21_pose_fit_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--sample-count", type=int, default=2000)
    ap.add_argument("--icp-iterations", type=int, default=20)
    ap.add_argument("--repo-root", default="/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
