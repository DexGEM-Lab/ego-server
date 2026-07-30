#!/usr/bin/env python3
"""V21 robust rigid pose: hand-guided position + mask silhouette orientation.

Uses hand MANO wrist position as a depth prior for the object (object is
near the hand), and mask silhouette PCA for orientation. This avoids the
DepthPro depth noise that breaks ICP.

Output: v21_robust_pose.json (consumed by the temporal pose graph)
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
import cv2, numpy as np
from scipy.spatial import cKDTree

from v21_mask_sources import resolve_current_object_mask_dir


def load_json(p): return json.loads(Path(p).read_text())

def run(args):
    run_root = Path(args.run_root)
    obj = args.object_id

    # Load manifest
    manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    depth_data = np.load(str(run_root / "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz"))
    depth = depth_data["depth"]
    intr = depth_data["intrinsics_fx_fy_cx_cy"]

    # Load hands
    hands = load_json(run_root / "measurements/hand_candidates/wilor_v21_metric/wilor_metric_hands.json")
    hands_by_frame = {f["frame_idx"]: f["hands"] for f in hands["frames"]}

    # Load current V19/V21 masks. The removed V20 local_grabcut branch is intentionally not used.
    mask_dir = resolve_current_object_mask_dir(run_root, obj)
    mask_files = {int(Path(mf).stem): mf for mf in sorted(glob.glob(str(mask_dir / "*.png")))}

    # Load mesh summary for extent
    mesh_sum = load_json(run_root / f"measurements/object_geometry/v21_mesh_candidate/{obj}/mesh_candidate_summary.json")
    mesh_extent = np.array(mesh_sum["object_extent_m"])
    mesh_center_world = np.array(mesh_sum.get("canonical_center_m", mesh_sum.get("object_center_m", [0,0,0])))

    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]
    img_w = manifest["frames"][0]["manifest_width"]
    img_h = manifest["frames"][0]["manifest_height"]

    pose_rows = []
    prev_t = None
    prev_R = np.eye(3)

    for fm in manifest["frames"]:
        fidx = fm["frame_idx"]
        
        # Get mask
        if fidx not in mask_files:
            pose_rows.append({"frame_idx": fidx, "status": "no_mask", "rotation_matrix": prev_R.tolist(), "translation_m": (prev_t or [0,0,1.5]).tolist()})
            continue
        
        mask = cv2.imread(mask_files[fidx], cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 127)
        if len(ys) < 10:
            pose_rows.append({"frame_idx": fidx, "status": "small_mask", "rotation_matrix": prev_R.tolist(), "translation_m": (prev_t or [0,0,1.5]).tolist()})
            continue

        # Mask centroid in source resolution
        cx_m = xs.mean() * src_w / mask.shape[1]
        cy_m = ys.mean() * src_h / mask.shape[0]
        
        # Get intrinsics at source res
        fx, fy, cx, cy = intr[fidx]
        
        # Depth: use hand wrist depth if available, else depth median
        hand_depth = None
        hands_frame = hands_by_frame.get(fidx, [])
        for h in hands_frame:
            cam_t = h.get("cam_t_metric_smoothed") or h.get("cam_t_metric", [0,0,1.5])
            hand_depth = cam_t[2]
            break
        
        if hand_depth is None or hand_depth < 0.3 or hand_depth > 5.0:
            # Fallback: depth median at mask region (with outlier removal)
            mask_in_src = cv2.resize(mask, (src_w, src_h))
            mask_ys, mask_xs = np.where(mask_in_src > 127)
            mask_depths = depth[fidx][mask_ys, mask_xs]
            valid = mask_depths[(mask_depths > 0.3) & (mask_depths < 5.0)]
            hand_depth = float(np.median(valid)) if len(valid) > 10 else 1.5

        # Object is slightly in front of hand (hand wraps around object)
        obj_depth = hand_depth - 0.05  # 5cm in front of hand center
        
        # Backproject mask centroid to 3D
        tx = (cx_m - cx) * obj_depth / fx
        ty = (cy_m - cy) * obj_depth / fy
        tz = obj_depth
        t = np.array([tx, ty, tz])

        # Orientation from mask PCA (in image plane)
        mask_pts = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
        if len(mask_pts) > 20:
            cov = np.cov(mask_pts.T)
            evals, evecs = np.linalg.eigh(cov)
            principal = evecs[:, np.argmax(evals)]
            # Convert 2D principal to 3D (image plane)
            dy, dx = principal[1] / fy, principal[0] / fx
            principal_3d = np.array([dx, dy, 0.0])
            n = np.linalg.norm(principal_3d)
            if n > 1e-6:
                principal_3d /= n
            else:
                principal_3d = np.array([0, 1, 0])
            # Build rotation: Y axis = principal, Z = camera -Z
            mesh_z = np.array([0, 0, -1.0])
            mesh_y = principal_3d.copy()
            mesh_y[2] = 0
            mesh_y /= max(np.linalg.norm(mesh_y), 1e-6)
            mesh_x = np.cross(mesh_y, mesh_z)
            nx = np.linalg.norm(mesh_x)
            if nx > 0.1:
                mesh_x /= nx
            else:
                mesh_x = np.array([1, 0, 0])
            mesh_y = np.cross(mesh_z, mesh_x)
            R = np.column_stack([mesh_x, mesh_y, mesh_z])
        else:
            R = np.eye(3)

        # Temporal smoothing
        if prev_t is not None:
            # Smooth translation with limited jump
            jump = np.linalg.norm(t - prev_t)
            max_jump = 0.04  # 4cm per frame
            if jump > max_jump:
                alpha = max_jump / max(jump, 1e-6)
                t = prev_t + alpha * (t - prev_t)
            # Smooth rotation via SLERP-like blend
            R_diff = R @ prev_R.T
            from scipy.spatial.transform import Rotation, Slerp
            try:
                keyrots = Rotation.from_matrix([np.eye(3), R_diff])
                slerp = Slerp([0, 1], keyrots)
                alpha_r = min(0.3, 0.15 / max(np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec()), 1e-6))
                blended = slerp(alpha_r).as_matrix()
                R = blended @ prev_R
            except:
                R = prev_R @ 0.85 + R * 0.15  # fallback
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt

        # Also produce observed surface points for the pose graph
        mask_in_src = cv2.resize(mask, (src_w, src_h))
        m_ys, m_xs = np.where(mask_in_src > 127)
        m_zs = depth[fidx][m_ys, m_xs].astype(float)
        valid = (m_zs > 0.3) & (m_zs < 5.0)
        # Outlier removal
        if valid.sum() > 20:
            z_med = np.median(m_zs[valid])
            z_iqr = np.percentile(m_zs[valid], 75) - np.percentile(m_zs[valid], 25)
            inlier = valid & (m_zs >= z_med - 3*max(z_iqr,0.01)) & (m_zs <= z_med + 3*max(z_iqr,0.01))
        else:
            inlier = valid
        
        m_xs_v, m_ys_v, m_zs_v = m_xs[inlier], m_ys[inlier], m_zs[inlier]
        if len(m_xs_v) > 300:
            idx = np.random.default_rng(42).choice(len(m_xs_v), 300, replace=False)
            m_xs_v, m_ys_v, m_zs_v = m_xs_v[idx], m_ys_v[idx], m_zs_v[idx]
        xc = (m_xs_v.astype(float) - cx) * m_zs_v / fx
        yc = (m_ys_v.astype(float) - cy) * m_zs_v / fy
        obs_pts = np.column_stack([xc, yc, m_zs_v])

        pose_rows.append({
            "frame_idx": fidx,
            "status": "estimated",
            "rotation_matrix": R.tolist(),
            "translation_m": t.tolist(),
            "object_depth_m": float(tz),
            "hand_depth_m": float(hand_depth),
            "observed_points": obs_pts.tolist(),
            "observed_point_count": len(obs_pts),
        })
        prev_t, prev_R = t, R

    # Stats
    depths = [r["object_depth_m"] for r in pose_rows if "object_depth_m" in r]
    output_dir = run_root / f"measurements/object_geometry_mesh_pose/{obj}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v21_robust_pose.v0",
        "method": "hand_guided_position_mask_silhouette_orientation",
        "object_id": obj,
        "pose_rows": pose_rows,
    }
    (output_dir / "v21_robust_pose.json").write_text(json.dumps(payload, indent=2))
    
    qc = {
        "status": "ok",
        "estimated_frames": sum(1 for r in pose_rows if r["status"] == "estimated"),
        "depth_median_m": float(np.median(depths)) if depths else None,
        "depth_min_m": float(np.min(depths)) if depths else None,
        "depth_max_m": float(np.max(depths)) if depths else None,
    }
    (output_dir / "v21_robust_pose_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.run = ap.parse_args()
    run(ap.run)
