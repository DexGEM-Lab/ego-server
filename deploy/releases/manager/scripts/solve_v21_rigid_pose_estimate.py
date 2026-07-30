#!/usr/bin/env python3
"""V21 rigid object pose estimation from mask centroid + depth.

For each frame with an accepted mask:
1. Find the mask centroid in 2D image space
2. Sample median depth at the mask region
3. Backproject the 2D centroid + depth to 3D camera-space position
4. Estimate orientation from the mask's principal axes (PCA of mask pixels)
5. Place the mesh candidate at the estimated position and orientation

This is a robust approximation when depth is noisy. It produces a
per-frame SE(3) pose that is sane enough for rendering. Full ICP
refinement can be applied later with cleaner depth.

Output:
  measurements/object_geometry_mesh_pose/<object_id>/v21_pose_estimate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
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


def load_depth_and_intrinsics(npz_path):
    data = np.load(str(npz_path))
    return data["depth"], data["intrinsics_fx_fy_cx_cy"]


def estimate_frame_pose(mask, depth_frame, intrinsics, mesh_extent):
    """Estimate object SE(3) pose from mask + depth.

    Returns rotation matrix (3x3) and translation (3,) in camera space.
    """
    fx, fy, cx, cy = intrinsics

    # Find mask pixels
    ys, xs = np.where(mask > 127)
    if len(ys) < 10:
        return None, None

    # Sample depth at mask region
    zs = np.asarray(depth_frame[ys, xs], dtype=np.float64)
    valid = zs > 0.1
    if valid.sum() < 10:
        return None, None

    xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

    # Outlier removal in depth
    z_med = np.median(zs_v)
    z_iqr = np.percentile(zs_v, 75) - np.percentile(zs_v, 25)
    z_lo = z_med - 3 * max(z_iqr, 0.01)
    z_hi = z_med + 3 * max(z_iqr, 0.01)
    inlier = (zs_v >= z_lo) & (zs_v <= z_hi)
    xs_v, ys_v, zs_v = xs_v[inlier], ys_v[inlier], zs_v[inlier]

    if len(xs_v) < 10:
        return None, None

    # Median depth at the object
    obj_depth = float(np.median(zs_v))

    # Mask centroid in 2D
    cx_m = float(xs_v.mean())
    cy_m = float(ys_v.mean())

    # Backproject centroid to 3D
    tx = (cx_m - cx) * obj_depth / fx
    ty = (cy_m - cy) * obj_depth / fy
    tz = obj_depth
    translation = np.array([tx, ty, tz])

    # Estimate orientation from mask shape
    # Use PCA of mask pixel positions to get the two in-plane axes
    mask_pts = np.column_stack([xs_v - cx_m, ys_v - cy_m]).astype(np.float64)
    cov_2d = np.cov(mask_pts.T)
    evals_2d, evecs_2d = np.linalg.eigh(cov_2d)
    order = np.argsort(evals_2d)[::-1]
    principal_2d = evecs_2d[:, order[0]]  # first principal axis in 2D

    # Convert 2D principal direction to 3D camera direction
    # The 2D direction maps to a ray direction at the object depth
    dx = principal_2d[0] / fx  # angular component
    dy = principal_2d[1] / fy
    # The 3D direction in the image plane
    principal_3d = np.array([dx, dy, 0.0])
    principal_3d = principal_3d / max(np.linalg.norm(principal_3d), 1e-6)

    # Camera Z axis (viewing direction)
    cam_z = np.array([0.0, 0.0, 1.0])

    # Build rotation: align mesh's longest axis with the principal direction
    # Assume the mesh's longest axis is along its local Y axis (height of can)
    # So we rotate local Y to the principal_3d direction
    # And local Z faces the camera (negative viewing direction)
    
    # Simple approach: align mesh Y axis to the mask's principal axis
    # Mesh Y axis in camera frame should be the principal direction projected
    mesh_y_cam = principal_3d.copy()
    
    # Ensure mesh_y_cam is perpendicular to viewing direction (approximately)
    mesh_y_cam[2] = 0  # keep it in the image plane
    mesh_y_cam = mesh_y_cam / max(np.linalg.norm(mesh_y_cam), 1e-6)
    
    # Mesh Z axis points towards camera (negative viewing direction)
    mesh_z_cam = np.array([0.0, 0.0, -1.0])
    
    # Mesh X axis = Y cross Z
    mesh_x_cam = np.cross(mesh_y_cam, mesh_z_cam)
    nx = np.linalg.norm(mesh_x_cam)
    if nx < 0.1:
        mesh_x_cam = np.array([1.0, 0.0, 0.0])
    else:
        mesh_x_cam = mesh_x_cam / nx
    
    # Re-orthogonalize Y
    mesh_y_cam = np.cross(mesh_z_cam, mesh_x_cam)
    
    # Rotation matrix: columns are mesh axes in camera frame
    R = np.column_stack([mesh_x_cam, mesh_y_cam, mesh_z_cam])

    return R, translation


def run(args):
    run_root = Path(args.run_root)
    object_id = args.object_id

    # Load mesh candidate for extent
    mesh_path = run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_candidate.obj"
    mesh = load_mesh(mesh_path)
    mesh_extent = mesh.extents
    print(f"Loaded mesh: extent={mesh_extent}", flush=True)

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

    # Estimate pose per-frame
    pose_rows = []
    prev_t = None

    for fi, fm in enumerate(manifest["frames"]):
        frame_idx = fm["frame_idx"]

        if frame_idx not in mask_files:
            if prev_t is not None:
                pose_rows.append({
                    "frame_idx": frame_idx,
                    "status": "no_mask_pose_carried",
                    "rotation_matrix": pose_rows[-1]["rotation_matrix"] if pose_rows else np.eye(3).tolist(),
                    "translation_m": prev_t.tolist(),
                })
            else:
                pose_rows.append({"frame_idx": frame_idx, "status": "no_mask_no_pose"})
            continue

        mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
        intr = per_frame_intrinsics[frame_idx]
        R, t = estimate_frame_pose(mask, depth[frame_idx], intr, mesh_extent)

        if R is None:
            pose_rows.append({"frame_idx": frame_idx, "status": "insufficient_mask_depth"})
            continue

        # Temporal smoothing on translation
        if prev_t is not None:
            t_jump = np.linalg.norm(t - prev_t)
            if t_jump > 0.03:  # max 3cm per frame
                alpha = 0.03 / max(t_jump, 1e-6)
                t = prev_t + alpha * (t - prev_t)

        # Check rotation continuity with previous
        if len(pose_rows) > 0 and pose_rows[-1].get("rotation_matrix"):
            prev_R = np.array(pose_rows[-1]["rotation_matrix"])
            R_jump = np.linalg.norm(R - prev_R)
            if R_jump > 0.3:  # large rotation change
                R = prev_R + 0.1 * (R - prev_R)  # gentle blend
                # Re-orthogonalize
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt

        pose_rows.append({
            "frame_idx": frame_idx,
            "status": "estimated",
            "rotation_matrix": R.tolist(),
            "translation_m": t.tolist(),
            "object_depth_m": float(t[2]),
        })
        prev_t = t

        if fi % 50 == 0:
            print(f"  Frame {frame_idx}: t={t}, depth={t[2]:.3f}m", flush=True)

    # Summary
    estimated = [r for r in pose_rows if r.get("status") == "estimated"]
    depths = [r["object_depth_m"] for r in estimated if "object_depth_m" in r]

    output_dir = run_root / "measurements" / "object_geometry_mesh_pose" / object_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_data = {
        "schema": "v21_rigid_pose_estimate.v0",
        "method": "mask_centroid_depth_backprojection_with_pca_orientation",
        "object_id": object_id,
        "run_root": str(run_root),
        "mesh_candidate": str(mesh_path),
        "mesh_extent_m": mesh_extent.tolist(),
        "frame_count": len(pose_rows),
        "pose_rows": pose_rows,
    }
    (output_dir / "v21_pose_estimate.json").write_text(json.dumps(pose_data, indent=2))

    qc = {
        "status": "ok",
        "method": "v21_rigid_pose_from_mask_centroid_depth",
        "object_id": object_id,
        "total_frames": total_frames,
        "estimated_frames": len(estimated),
        "depth_summary": {
            "median_m": float(np.median(depths)) if depths else None,
            "min_m": float(np.min(depths)) if depths else None,
            "max_m": float(np.max(depths)) if depths else None,
        } if depths else {},
        "candidate_state": "pose_estimated_mesh_candidate",
        "next_required_step": "render_pose_estimated_mesh_overlay",
        "output": str(output_dir / "v21_pose_estimate.json"),
    }
    (output_dir / "v21_pose_estimate_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
