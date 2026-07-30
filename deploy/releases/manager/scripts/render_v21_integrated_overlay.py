#!/usr/bin/env python3
"""V21 integrated overlay renderer: hand MANO + object mesh pose on raw video.

Combines:
- Metric MANO hand candidates (WiLoR + depth refit)
- Object mesh pose estimate (mask centroid + depth)
into a single overlay video on the raw frames.

This produces the primary V21 overlay render showing both hand and
object physical state annotations.

Output:
  renders/v21_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh required", file=sys.stderr)
    sys.exit(1)

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

SIDE_COLORS = {"right": (0, 255, 100), "left": (100, 100, 255)}
OBJECT_COLOR = (0, 165, 255)  # orange
OBJECT_EDGE_COLOR = (0, 100, 200)


def project_points(points_3d_cam, cam_t, intrinsics):
    fx, fy, cx, cy = intrinsics
    pts = points_3d_cam + np.array(cam_t)
    z = np.clip(pts[:, 2], 0.01, None)
    x2d = fx * pts[:, 0] / z + cx
    y2d = fy * pts[:, 1] / z + cy
    return np.column_stack([x2d, y2d])


def project_3d_points(points_3d_cam, intrinsics):
    """Project 3D camera-space points directly (no additional translation)."""
    fx, fy, cx, cy = intrinsics
    z = np.clip(points_3d_cam[:, 2], 0.01, None)
    x2d = fx * points_3d_cam[:, 0] / z + cx
    y2d = fy * points_3d_cam[:, 1] / z + cy
    return np.column_stack([x2d, y2d])


def transform_mesh_vertices(verts, R, t):
    """Apply SE(3) transform to vertices."""
    return (R @ verts.T).T + t


def render_hand_overlay(img, hands_data):
    for hand in hands_data:
        side = hand.get("side", "right")
        color = SIDE_COLORS.get(side, (255, 255, 0))
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric")
        intrinsics = hand.get("intrinsics_manifest")
        if cam_t is None or intrinsics is None:
            continue

        joints3d = np.array(hand["joints3d_camera_metric"], dtype=float)
        joints2d_proj = project_points(joints3d, cam_t, intrinsics)

        for a, b in HAND_EDGES:
            cv2.line(img, tuple(joints2d_proj[a].astype(int)),
                     tuple(joints2d_proj[b].astype(int)), color, 3, cv2.LINE_AA)

        for j, pt in enumerate(joints2d_proj):
            radius = 5 if j in [0, 4, 8, 12, 16, 20] else 3
            cv2.circle(img, tuple(pt.astype(int)), radius, color, -1, cv2.LINE_AA)

        bbox = hand.get("bbox_xyxy", [])
        if len(bbox) == 4:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            score = hand.get("detector_score", 0)
            label = f"{side[0].upper()} {score:.2f}"
            cv2.putText(img, label, (int(bbox[0]), max(int(bbox[1]) - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_object_overlay(img, mesh_verts, R, t, intrinsics, source_resolution, manifest_resolution):
    """Render object mesh wireframe at the estimated pose."""
    # Transform mesh vertices to camera space
    verts_cam = transform_mesh_vertices(mesh_verts, R, t)

    # Project to 2D
    # Scale intrinsics if source != manifest resolution
    scale = manifest_resolution[0] / source_resolution[0]
    intr_manifest = [intrinsics[0] * scale, intrinsics[1] * scale,
                     intrinsics[2] * scale, intrinsics[3] * scale]
    verts2d = project_3d_points(verts_cam, intr_manifest)

    # Draw mesh vertices as small dots
    for pt in verts2d:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            cv2.circle(img, (px, py), 1, OBJECT_COLOR, -1)

    # Draw bounding box of the mesh at this pose
    min_2d = verts2d.min(axis=0)
    max_2d = verts2d.max(axis=0)
    cv2.rectangle(img,
                  (int(min_2d[0]), int(min_2d[1])),
                  (int(max_2d[0]), int(max_2d[1])),
                  OBJECT_EDGE_COLOR, 2)

    # Draw 3D extent label
    center_2d = (min_2d + max_2d) / 2
    cv2.putText(img, f"OBJ z={t[2]:.2f}m",
                (int(min_2d[0]), max(int(min_2d[1]) - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJECT_EDGE_COLOR, 1, cv2.LINE_AA)


def run(args):
    run_root = Path(args.run_root)
    repo_root = Path(args.repo_root)

    # Load manifest
    manifest = json.loads((run_root / "input" / "raw_frame_manifest" / "manifest.json").read_text())
    total_frames = len(manifest["frames"])
    img_w = manifest["frames"][0]["manifest_width"]
    img_h = manifest["frames"][0]["manifest_height"]
    src_w = manifest["frames"][0]["source_width"]
    src_h = manifest["frames"][0]["source_height"]
    fps = manifest.get("fps", 25.0)

    # Load depth for per-frame intrinsics
    depth_npz = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
    depth_data = np.load(str(depth_npz))
    per_frame_intrinsics = depth_data["intrinsics_fx_fy_cx_cy"]

    # Load hand candidates
    hand_path = run_root / "measurements" / "hand_candidates" / "wilor_v21_metric" / "wilor_metric_hands.json"
    hands_data = {}
    if hand_path.exists():
        hand_payload = json.loads(hand_path.read_text())
        hands_data = {f["frame_idx"]: f.get("hands", []) for f in hand_payload["frames"]}
        print(f"Loaded hand data for {len(hands_data)} frames", flush=True)
    else:
        print("No hand data found", flush=True)

    # Load object mesh
    object_id = args.object_id
    mesh_path = run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_candidate.obj"
    mesh = None
    mesh_verts = None
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(meshes)
        # Sample surface for rendering (reduce vertex count)
        mesh_verts, _ = trimesh.sample.sample_surface(mesh, 1500)
        mesh_verts = np.asarray(mesh_verts, dtype=np.float64)
        print(f"Loaded object mesh: {len(mesh.vertices)} verts, sampled {len(mesh_verts)} for rendering", flush=True)
    else:
        print(f"No mesh at {mesh_path}", flush=True)

    # Load object pose
    pose_path = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v21_pose_estimate.json"
    poses_data = {}
    if pose_path.exists():
        pose_payload = json.loads(pose_path.read_text())
        poses_data = {r["frame_idx"]: r for r in pose_payload["pose_rows"]}
        print(f"Loaded object poses for {len(poses_data)} frames", flush=True)
    else:
        print(f"No pose at {pose_path}", flush=True)

    # Render
    output_path = run_root / "renders" / "v21_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (img_w, img_h))

    for fm in tqdm(manifest["frames"], desc="V21 overlay render"):
        frame_idx = fm["frame_idx"]
        rgb_path = str(repo_root / fm["rgb"])
        img = cv2.imread(rgb_path)
        if img is None:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Render hand
        hands = hands_data.get(frame_idx, [])
        if hands:
            render_hand_overlay(img, hands)

        # Render object
        if mesh_verts is not None and frame_idx in poses_data:
            pose = poses_data[frame_idx]
            if pose.get("status") in ("estimated", "no_mask_pose_carried"):
                R = np.array(pose["rotation_matrix"])
                t = np.array(pose["translation_m"])
                intr = per_frame_intrinsics[frame_idx]
                render_object_overlay(img, mesh_verts, R, t, intr,
                                      (src_w, src_h), (img_w, img_h))

        writer.write(img)

    writer.release()

    summary = {
        "status": "ok",
        "method": "render_v21_integrated_overlay",
        "run_root": str(run_root),
        "output_video": str(output_path),
        "frame_count": total_frames,
        "has_hand_data": len(hands_data) > 0,
        "has_object_mesh": mesh_verts is not None,
        "has_object_pose": len(poses_data) > 0,
        "object_id": object_id,
    }
    (run_root / "renders" / "v21_overlay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", default="red_scandic_tin")
    ap.add_argument("--repo-root", default="/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
