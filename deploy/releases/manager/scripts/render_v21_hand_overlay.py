#!/usr/bin/env python3
"""V21 hand overlay renderer.

Renders metric MANO hand candidates on top of the raw video to verify
hand detection, side mapping, and 3D structure visually.

This is a diagnostic render, not the final integrated V21 render.
It overlays projected MANO vertices/joints on each frame to show
where the hand candidate state places the hand.

Output:
  renders/v21_hand_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# MANO face indices for fingertip spheres (simplified - just draw vertices)
SIDE_COLORS = {
    "right": (0, 255, 100),  # green
    "left": (100, 100, 255),  # red
}


def project_points(points_3d_cam, cam_t, intrinsics):
    """Project 3D camera-space points to 2D using intrinsics."""
    fx, fy, cx, cy = intrinsics
    pts = points_3d_cam + np.array(cam_t)
    z = np.clip(pts[:, 2], 0.01, None)
    x2d = fx * pts[:, 0] / z + cx
    y2d = fy * pts[:, 1] / z + cy
    return np.column_stack([x2d, y2d])


def render_frame(img, hands_data):
    """Render hand skeleton and vertices on image."""
    overlay = img.copy()

    for hand in hands_data:
        side = hand.get("side", "right")
        color = SIDE_COLORS.get(side, (255, 255, 0))
        cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric")
        intrinsics = hand.get("intrinsics_manifest")
        if cam_t is None or intrinsics is None:
            continue

        joints3d = np.array(hand["joints3d_camera_metric"], dtype=float)
        joints2d_proj = project_points(joints3d, cam_t, intrinsics)

        # Draw skeleton edges
        for a, b in HAND_EDGES:
            pt_a = joints2d_proj[a]
            pt_b = joints2d_proj[b]
            cv2.line(overlay, tuple(pt_a.astype(int)), tuple(pt_b.astype(int)), color, 3, cv2.LINE_AA)

        # Draw joints
        for j, pt in enumerate(joints2d_proj):
            radius = 5 if j in [0, 4, 8, 12, 16, 20] else 3
            cv2.circle(overlay, tuple(pt.astype(int)), radius, color, -1, cv2.LINE_AA)

        # Draw MANO mesh vertices (subsampled) as semi-transparent overlay
        verts = hand.get("vertices_camera_metric_sample", [])
        if verts and len(verts) > 10:
            verts3d = np.array(verts, dtype=float)
            verts2d = project_points(verts3d, cam_t, intrinsics)
            for pt in verts2d:
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < overlay.shape[1] and 0 <= py < overlay.shape[0]:
                    cv2.circle(overlay, (px, py), 1, color, -1)

        # Draw detection bbox
        bbox = hand.get("bbox_xyxy", [])
        if len(bbox) == 4:
            cv2.rectangle(overlay,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

        # Label
        score = hand.get("detector_score", 0)
        cam_z = cam_t[2] if isinstance(cam_t, list) else cam_t
        label = f"{side[0].upper()} s={score:.2f} z={float(cam_z):.2f}m"
        cv2.putText(overlay, label, (int(bbox[0]), max(int(bbox[1]) - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return overlay


def run(args):
    run_root = Path(args.run_root)
    repo_root = Path(args.repo_root)

    # Load metric hands
    metric_path = run_root / "measurements" / "hand_candidates" / "wilor_v21_metric" / "wilor_metric_hands.json"
    metric = json.loads(metric_path.read_text())
    frames_data = metric["frames"]

    # Load manifest
    manifest = json.loads((run_root / "input" / "raw_frame_manifest" / "manifest.json").read_text())
    total_frames = len(manifest["frames"])

    # Output path
    output_path = run_root / "renders" / "v21_hand_overlay.mp4"

    # Determine frame dimensions
    first_frame = manifest["frames"][0]
    img_w = first_frame["manifest_width"]
    img_h = first_frame["manifest_height"]
    fps = manifest.get("fps", 25.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (img_w, img_h))

    # Build lookup
    hand_by_frame = {f["frame_idx"]: f for f in frames_data}

    written = 0
    for fm in tqdm(manifest["frames"], desc="Hand overlay render"):
        frame_idx = fm["frame_idx"]
        rgb_path = str(repo_root / fm["rgb"])
        img = cv2.imread(rgb_path)
        if img is None:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        hd = hand_by_frame.get(frame_idx, {"hands": []})
        overlay = render_frame(img, hd.get("hands", []))
        writer.write(overlay)
        written += 1

    writer.release()

    summary = {
        "status": "ok",
        "method": "render_v21_hand_overlay",
        "run_root": str(run_root),
        "output_video": str(output_path),
        "frame_count": written,
        "expected_frames": total_frames,
        "frame_count_match": written == total_frames,
    }
    summary_path = run_root / "renders" / "v21_hand_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--repo-root", default="/mnt/user-home/zjh/ego-pipeline/ego_annotation-master")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
