#!/usr/bin/env python3
"""V21 → V18 annotation adapter.

Assembles all V21 measurements into the V18 annotation JSON format that
the V18/V19 factor graph optimizers and renderer consume.

This is the critical bridge that connects V21's measurement scripts
(depth, segmentation, WiLoR hand candidates, mesh candidate, pose estimate)
to the V19 physical-state spine (factor graphs, contact/occlusion/
nonpenetration, V18 full pipeline renderer).

Output:
  state/annotations_v18_compatible.json

This file drives:
  - fit_v18_compact_rigid_object_pose.py (ICP)
  - solve_v19_rigid_object_pose_graph.py (temporal pose graph)
  - optimize_contact_aware_mano_graph_v8.py (MANO optimization)
  - run_v18_full_pipeline.py / render_v18_full_pipeline_from_annotations.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from v21_mask_sources import resolve_current_object_mask_dir


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_depth_and_intrinsics(npz_path):
    data = np.load(str(npz_path))
    return data["depth"], data["intrinsics_fx_fy_cx_cy"]


def backproject_frame(mask, depth_frame, intrinsics, outlier_removal=True, max_points=300):
    fx, fy, cx, cy = intrinsics
    ys, xs = np.where(mask > 127)
    if len(ys) < 5:
        return np.zeros((0, 3)), 0.0
    zs = np.asarray(depth_frame[ys, xs], dtype=np.float64)
    valid = zs > 0.1
    xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]
    if outlier_removal and len(zs_v) > 50:
        z_med = np.median(zs_v)
        z_iqr = np.percentile(zs_v, 75) - np.percentile(zs_v, 25)
        z_lo = z_med - 3 * max(z_iqr, 0.01)
        z_hi = z_med + 3 * max(z_iqr, 0.01)
        inlier = (zs_v >= z_lo) & (zs_v <= z_hi)
        xs_v, ys_v, zs_v = xs_v[inlier], ys_v[inlier], zs_v[inlier]
    if len(xs_v) > max_points:
        idx = np.random.default_rng(42).choice(len(xs_v), max_points, replace=False)
        xs_v, ys_v, zs_v = xs_v[idx], ys_v[idx], zs_v[idx]
    xc = (xs_v.astype(float) - cx) * zs_v / fx
    yc = (ys_v.astype(float) - cy) * zs_v / fy
    points = np.column_stack([xc, yc, zs_v])
    depth_median = float(np.median(zs_v)) if len(zs_v) > 0 else 0.0
    return points, depth_median


def run(args):
    run_root = Path(args.run_root)
    object_id = args.object_id
    repo_root = Path(args.repo_root)

    # Load manifest
    manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    total_frames = len(manifest["frames"])
    fps = manifest.get("fps", 25.0)
    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]

    # Load depth
    depth_npz = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
    depth, per_frame_intrinsics = load_depth_and_intrinsics(depth_npz)

    # Load metric hands
    hand_path = run_root / "measurements" / "hand_candidates" / "wilor_v21_metric" / "wilor_metric_hands.json"
    hand_data = load_json(hand_path) or {"frames": []}
    hands_by_frame = {}
    for f in hand_data["frames"]:
        hands_by_frame[f["frame_idx"]] = f.get("hands", [])

    # Load object pose (prefer robust pose over simple estimate)
    robust_pose_path = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v21_robust_pose.json"
    simple_pose_path = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v21_pose_estimate.json"
    pose_path = robust_pose_path if robust_pose_path.exists() else simple_pose_path
    pose_data = load_json(pose_path) or {"pose_rows": []}
    poses_by_frame = {}
    for r in pose_data["pose_rows"]:
        poses_by_frame[r["frame_idx"]] = r

    # Load current V19/V21 mask files. The removed V20 local_grabcut branch is intentionally not used.
    import glob
    mask_dir = resolve_current_object_mask_dir(run_root, object_id)
    mask_files = {}
    for mf in sorted(glob.glob(str(mask_dir / "*.png"))):
        fidx = int(Path(mf).stem)
        mask_files[fidx] = mf

    # Load existing visible geometry annotations for camera info
    vg_path = run_root / "measurements" / "object_visible_surfaces" / "depthpro_local_grabcut" / object_id / "annotations_v19_visible_geometry.json"
    vg_ann = load_json(vg_path) or {"frames": []}
    vg_by_frame = {}
    for f in vg_ann["frames"]:
        vg_by_frame[f["frame_idx"]] = f

    visible_objects_by_frame = {}
    for f in vg_ann["frames"]:
        frame_idx = f.get("frame_idx")
        if frame_idx is None:
            continue
        for obj in f.get("objects", []):
            if not isinstance(obj, dict):
                continue
            if obj.get("track_id") == object_id or obj.get("object_id") in {object_id, f"object:{object_id}"}:
                visible_objects_by_frame[int(frame_idx)] = obj
                break

    # Load mesh candidate for canonical center
    mesh_summary = load_json(run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_candidate_summary.json")
    mesh_center = np.array(mesh_summary.get("object_center_m", [0, 0, 0])) if mesh_summary else np.zeros(3)

    print(f"Assembling V18 annotations for {total_frames} frames...", flush=True)

    # Build annotation frames
    annotation_frames = []
    for fi, fm in enumerate(manifest["frames"]):
        frame_idx = fm["frame_idx"]

        # Camera info from visible geometry or identity
        vg_frame = vg_by_frame.get(frame_idx, {})
        camera = vg_frame.get("camera", {
            "T_world_camera_metric": np.eye(4).tolist(),
            "position_world_m": [0.0, 0.0, 0.0],
            "v19_camera_pose_source": "identity_camera_frame",
        })

        # Objects
        objects = []
        visible_object = visible_objects_by_frame.get(frame_idx)
        if visible_object is not None:
            obj = copy.deepcopy(visible_object)
            obj["object_id"] = f"object:{object_id}"
            obj["track_id"] = object_id
            obj.setdefault("label", object_id)
            obj.setdefault("v19_physical_model", "rigid_visible_surface_uncertain")
            objects.append(obj)
        elif frame_idx in mask_files:
            mask = cv2.imread(mask_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read object mask: {mask_files[frame_idx]}")
            if mask.shape != depth[frame_idx].shape:
                mask = cv2.resize(mask, (depth[frame_idx].shape[1], depth[frame_idx].shape[0]), interpolation=cv2.INTER_NEAREST)
            intr = per_frame_intrinsics[frame_idx]
            observed_pts, depth_median = backproject_frame(mask, depth[frame_idx], intr, max_points=300)

            # Get mask bbox
            ys, xs = np.where(mask > 127)
            if len(ys) > 0:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                area_px = int(len(ys))
            else:
                bbox = [0, 0, 0, 0]
                area_px = 0

            # World vertices = camera vertices (identity camera)
            world_verts = observed_pts.tolist() if len(observed_pts) > 0 else []

            # Pose from estimate
            pose_row = poses_by_frame.get(frame_idx, {})
            R = pose_row.get("rotation_matrix", np.eye(3).tolist())
            t = pose_row.get("translation_m", [0, 0, depth_median])

            obj = {
                "object_id": f"object:{object_id}",
                "track_id": object_id,
                "label": object_id,
                "visible": len(observed_pts) > 5,
                "mask_path": mask_files[frame_idx],
                "bbox_xyxy": bbox,
                "area_px": area_px,
                "depth_m": depth_median,
                "visible_geometry_candidate": {
                    "status": "visible_metric_surface_measurement" if len(observed_pts) > 5 else "insufficient_visible_surface",
                    "source": "depthpro_mask_backprojection",
                    "frame_idx": frame_idx,
                    "object_id": f"object:{object_id}",
                    "track_id": object_id,
                    "mask_path": mask_files[frame_idx],
                    "depth_npz": str(depth_npz),
                    "depth_frame_index": frame_idx,
                    "camera_pose_source": camera.get("v19_camera_pose_source", "identity"),
                    "intrinsics_fx_fy_cx_cy": intr.tolist(),
                    "vertex_count": len(observed_pts),
                    "world_vertices_sample_m": world_verts,
                    "camera_vertices_sample_m": world_verts,
                    "centroid_world_m": (observed_pts.mean(axis=0).tolist() if len(observed_pts) > 0 else [0, 0, 0]),
                    "world_extent_m": ((observed_pts.max(axis=0) - observed_pts.min(axis=0)).tolist() if len(observed_pts) > 0 else [0, 0, 0]),
                    "depth_median_m": depth_median,
                },
                "reconstructed_geometry_pose": {
                    "rotation_world_from_canonical_matrix": R,
                    "translation_world_m": t,
                    "pose_source": pose_row.get("status", "mask_centroid_depth_estimate"),
                    "pose_uncertainty": "high_mask_centroid_plus_noisy_depth",
                    "anchor_frame_idx": mesh_summary.get("anchor_frame", 0) if mesh_summary else 0,
                    "anchor_centroid_world_m": mesh_center.tolist(),
                },
                "v19_physical_model": "rigid_visible_surface_uncertain",
            }
            objects.append(obj)

        # Hands
        hands = []
        for hand in hands_by_frame.get(frame_idx, []):
            cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric", [0, 0, 1])
            mano_params = hand.get("mano_params", {})
            joints3d = hand.get("joints3d_camera_metric", hand.get("joints3d_camera", []))

            h = {
                "hand_side": hand.get("side", "right"),
                "bbox_xyxy": hand.get("bbox_xyxy", [0, 0, 0, 0]),
                "visibility_state": "visible",
                "mano_candidate": {
                    "source": "WiLoR_v21_metric_refit",
                    "bbox_xyxy": hand.get("bbox_xyxy", [0, 0, 0, 0]),
                    "joints3d_camera": joints3d,
                    "cam_t": cam_t,
                    "detector_score": hand.get("detector_score", 0),
                    "mano_params": mano_params,
                    "source_intrinsics": hand.get("intrinsics_manifest", per_frame_intrinsics[frame_idx].tolist()),
                    "uncertainty": "metric_scaled_candidate_not_active_optimized",
                },
                "metric_mano_state": {
                    "source": "WiLoR_v21_metric_refit",
                    "case_frame_idx": frame_idx,
                    "hand_side": hand.get("side", "right"),
                    "support_state": "wilor_metric_candidate",
                    "cam_t_metric": cam_t,
                    "joints3d_camera_metric": joints3d,
                    "mano_params": mano_params,
                    "vertices_camera_metric_sample": hand.get("vertices_camera_metric_sample", []),
                    "physical_factor_weight": 0.5,
                    "physical_factor_role": "metric_hand_candidate_evidence",
                    "depth_residual_m": hand.get("depth_residual", None),
                },
            }
            hands.append(h)

        annotation_frames.append({
            "frame_idx": frame_idx,
            "raw_frame_path": fm.get("rgb", ""),
            "source_width": src_w,
            "source_height": src_h,
            "camera": camera,
            "hands": hands,
            "objects": objects,
        })

        if fi % 100 == 0:
            print(f"  Frame {frame_idx}: {len(objects)} objects, {len(hands)} hands", flush=True)

    annotation = {
        "schema": "v18_full_annotation.v0",
        "case_id": run_root.name,
        "run_root": str(run_root),
        "frame_count": total_frames,
        "fps": fps,
        "source_width": src_w,
        "source_height": src_h,
        "frames": annotation_frames,
        "v21_measurement_sources": {
            "depth": str(depth_npz),
            "hands": str(hand_path),
            "object_pose": str(pose_path),
            "object_mesh": str(run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_candidate.obj"),
        },
    }

    output_path = run_root / "state" / "annotations_v18_compatible.json"
    output_path.write_text(json.dumps(annotation, indent=2))
    print(f"\nWrote {output_path} ({total_frames} frames)", flush=True)

    summary = {
        "status": "ok",
        "method": "assemble_v21_to_v18_annotations",
        "output": str(output_path),
        "frame_count": total_frames,
        "frames_with_objects": sum(1 for f in annotation_frames if f["objects"]),
        "frames_with_hands": sum(1 for f in annotation_frames if f["hands"]),
    }
    print(json.dumps(summary, indent=2))
    return annotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--repo-root", default="/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
