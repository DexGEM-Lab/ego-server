#!/usr/bin/env python3
"""V21 MANO metric refit from WiLoR candidates.

Takes raw WiLoR hand candidates (non-metric local scale) and produces
metric MANO state by:
1. Computing metric scale from median finger-chain bone length (target ~0.17m)
2. Applying scale to vertices/joints
3. Solving per-frame cam_t from 2D keypoints using per-frame DepthPro intrinsics
4. Refining cam_t z-translation against DepthPro depth at hand region
5. Temporal smoothing of cam_t

Output:
  measurements/hand_candidates/wilor_v21_metric/wilor_metric_hands.json
  measurements/hand_candidates/wilor_v21_metric/wilor_metric_qc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def hand_bone_scale_m(joints):
    if joints.shape != (21, 3):
        return float("nan")
    chains = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]]
    lengths = []
    for chain in chains:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def compute_metric_scale(raw_frames, target_bone_m=0.171):
    bones = []
    for frame in raw_frames:
        for hand in frame["raw_hands"]:
            joints = np.asarray(hand["joints3d_camera"], dtype=float)
            bone = hand_bone_scale_m(joints)
            if 0.01 < bone < 1.0:
                bones.append(bone)
    if not bones:
        return None, None
    arr = np.asarray(bones, dtype=float)
    median_bone = float(np.median(arr))
    scale = target_bone_m / median_bone
    residual = arr * scale - target_bone_m
    return scale, {
        "target_hand_bone_m": target_bone_m,
        "median_wilor_hand_bone": median_bone,
        "wilor_local_to_meters": scale,
        "sample_count": len(arr),
        "residual_iqr_m": [float(np.percentile(residual, 25)), float(np.percentile(residual, 75))],
    }


def solve_translation_z_from_depth(local_joints_m, points2d, intrinsics, depth_at_joints):
    """Refine cam_t z-translation by matching projected joints to depth.

    Uses the 2D keypoint positions to sample depth, then adjusts the z-component
    of cam_t so that the median projected joint depth matches the median observed depth.
    """
    fx, fy, cx, cy = intrinsics
    # Project local joints with unknown translation (z is the main unknown)
    # x_proj = fx * (x + tx) / (z + tz) + cx
    # y_proj = fy * (y + ty) / (z + tz) + cy
    # We already solved tx, ty, tz from 2D keypoints. Now refine tz using depth.

    # Get median observed depth at joint positions
    valid_mask = depth_at_joints > 0.1
    if valid_mask.sum() < 5:
        return None  # can't refine

    observed_depths = depth_at_joints[valid_mask]
    median_observed_z = float(np.median(observed_depths))

    # The local joints have their own z values (in metric after scaling)
    # cam_t_z should position the hand at the observed depth
    # median(z_local + cam_t_z) ≈ median_observed_z
    local_zs = local_joints_m[valid_mask, 2]
    cam_t_z_refined = median_observed_z - float(np.median(local_zs))

    return cam_t_z_refined


def run(args):
    run_root = Path(args.run_root)

    raw_path = run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"
    raw = load_json(raw_path)
    raw_frames = raw["frames"]

    manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    img_w = manifest["frames"][0]["manifest_width"]
    img_h = manifest["frames"][0]["manifest_height"]
    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]

    # Load DepthPro depth archive with per-frame intrinsics
    depth_npz_path = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
    depth_data = np.load(str(depth_npz_path))
    depth = depth_data["depth"]  # (N, src_h, src_w) float16, in meters
    per_frame_intrinsics = depth_data["intrinsics_fx_fy_cx_cy"]  # (N, 4)
    depth_h, depth_w = depth.shape[1], depth.shape[2]

    print(f"Manifest: {img_w}x{img_h}, Source: {src_w}x{src_h}, Depth: {depth_w}x{depth_h}", flush=True)

    # Step 1: Compute metric scale
    scale, scale_info = compute_metric_scale(raw_frames)
    if scale is None:
        print("ERROR: no plausible bone lengths", file=sys.stderr)
        sys.exit(1)
    print(f"Metric scale: {scale:.4f}", flush=True)

    # Precompute scale factors from manifest coords to depth coords
    scale_x = depth_w / img_w  # depth works at source res, joints2d at manifest res
    scale_y = depth_h / img_h

    # Step 2-4: Process each frame
    metric_frames = []
    depth_residuals_before = []
    depth_residuals_after = []

    for fi, frame in enumerate(raw_frames):
        frame_idx = frame["frame_idx"]
        intr = per_frame_intrinsics[frame_idx]
        fx, fy, cx_d, cy_d = intr

        # Adjust intrinsics principal point: depth is at source res, joints2d at manifest res
        # DepthPro intrinsics cx, cy are in source pixel coords
        # joints2d are in manifest pixel coords
        # Convert: joint_in_source = joint_in_manifest * (src_w / img_w)
        # So we need focal in manifest coords: focal_manifest = fx * (img_w / src_w)
        focal_manifest = fx * img_w / src_w
        cx_manifest = cx_d * img_w / src_w  # = img_w / 2 typically
        cy_manifest = cy_d * img_h / src_h
        intrinsics_manifest = [float(focal_manifest), float(focal_manifest), float(cx_manifest), float(cy_manifest)]

        metric_hands = []
        for hand in frame["raw_hands"]:
            joints3d = np.asarray(hand["joints3d_camera"], dtype=float) * scale
            verts = np.asarray(hand["vertices_camera"], dtype=float) * scale
            joints2d = np.asarray(hand["joints2d"], dtype=float)  # in manifest coords

            # Solve cam_t from 2D keypoints (gives rough translation)
            fx_m, fy_m, cx_m, cy_m = intrinsics_manifest
            qx = (joints2d[:, 0] - cx_m) / fx_m
            qy = (joints2d[:, 1] - cy_m) / fy_m
            rows, rhs = [], []
            for (x, y, z), u, v in zip(joints3d, qx, qy):
                rows.append([1.0, 0.0, -float(u)])
                rhs.append(float(u * z - x))
                rows.append([0.0, 1.0, -float(v)])
                rhs.append(float(v * z - y))
            cam_t, *_ = np.linalg.lstsq(np.array(rows, float), np.array(rhs, float), rcond=None)

            # Sample depth at joint positions (convert manifest to depth/source coords)
            px_d = np.clip((joints2d[:, 0] * scale_x).astype(int), 0, depth_w - 1)
            py_d = np.clip((joints2d[:, 1] * scale_y).astype(int), 0, depth_h - 1)
            hand_depths = np.asarray(depth[frame_idx, py_d, px_d], dtype=float)

            cam_t_z_before = float(cam_t[2])

            # Refine cam_t_z using depth
            valid = hand_depths > 0.1
            if valid.sum() >= 5:
                median_observed_z = float(np.median(hand_depths[valid]))
                median_local_z = float(np.median(joints3d[valid, 2]))
                cam_t_z_refined = median_observed_z - median_local_z
                # Blend: 70% depth-refined, 30% keypoint-solved (depth is more reliable for z)
                cam_t[2] = 0.7 * cam_t_z_refined + 0.3 * cam_t[2]
                depth_residuals_before.append(abs(cam_t_z_before - median_observed_z))
                depth_residuals_after.append(abs(cam_t[2] - median_observed_z))

            metric_hands.append({
                "backend": "WiLoR",
                "side": hand["side"],
                "detector_score": hand["detector_score"],
                "bbox_xyxy": hand["bbox_xyxy"],
                "cam_t_metric": [float(x) for x in cam_t],
                "focal_length": float(focal_manifest),
                "intrinsics_manifest": [float(x) for x in intrinsics_manifest],
                "joints3d_camera_metric": joints3d.tolist(),
                "joints2d": joints2d.tolist(),
                "mano_params": hand.get("mano_params", {}),
                "vertices_camera_metric": verts.tolist(),
                "vertices_camera_metric_sample": verts[::10].tolist(),
                "scale_applied": scale,
                "filter_status": "metric_scaled_and_depth_refined",
            })
        metric_frames.append({
            "frame_idx": frame_idx,
            "time_s": frame.get("time_s", frame_idx / 25.0),
            "hands": metric_hands,
        })

    # Step 5: Temporal smoothing
    prev_t = None
    for frame in metric_frames:
        for hand in frame["hands"]:
            cam_t = np.array(hand["cam_t_metric"], dtype=float)
            if prev_t is not None:
                jump = np.linalg.norm(cam_t - prev_t)
                if jump > 0.08:  # max 8cm frame-to-frame
                    alpha = min(0.6, 0.08 / max(jump, 1e-6))
                    cam_t = prev_t + alpha * (cam_t - prev_t)
            hand["cam_t_metric_smoothed"] = [float(x) for x in cam_t]
            prev_t = cam_t
        if not frame["hands"]:
            prev_t = None

    # Validation summary
    depth_validation = {}
    if depth_residuals_after:
        arr_b = np.array(depth_residuals_before)
        arr_a = np.array(depth_residuals_after)
        depth_validation = {
            "samples": len(arr_a),
            "median_residual_before_refit_m": float(np.median(arr_b)),
            "median_residual_after_refit_m": float(np.median(arr_a)),
            "p95_residual_after_refit_m": float(np.percentile(arr_a, 95)),
            "mean_residual_after_refit_m": float(np.mean(arr_a)),
        }

    # Save
    output_dir = run_root / "measurements" / "hand_candidates" / "wilor_v21_metric"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_payload = {
        "schema": "v21_wilor_metric_hands.v0",
        "backend": "WiLoR",
        "scale_source": "median_finger_chain_bone_length",
        "scale_info": scale_info,
        "frame_count": len(metric_frames),
        "frames": metric_frames,
    }
    save_json(output_dir / "wilor_metric_hands.json", metric_payload)

    qc = {
        "status": "ok",
        "method": "run_v21_mano_metric_refit_v2",
        "run_root": str(run_root),
        "scale_applied": scale,
        "scale_info": scale_info,
        "depth_validation": depth_validation,
        "candidate_state": "metric_scaled_depth_refined_candidate",
        "next_required_step": "active_shape_pose_scale_optimization_with_depth_and_silhouette",
        "output": str(output_dir / "wilor_metric_hands.json"),
    }
    save_json(output_dir / "wilor_metric_qc.json", qc)
    print(json.dumps(qc, indent=2))
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
