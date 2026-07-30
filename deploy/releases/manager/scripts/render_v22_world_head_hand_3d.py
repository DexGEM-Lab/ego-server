#!/usr/bin/env python3
"""Render a perspective 3D world-coordinate head/camera + MANO hand video for V22.

The render is driven by metric world state, not image-space detections. It uses
HaWoR/hybrid `R_c2w/t_c2w` and `vertices_world_m/joints_world_m`, then places a
virtual camera at a side-rear position looking forward along the estimated
camera/head trajectory. The output is a 3D perspective QC view, not a flat X/Z
or X/Y projection.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS = {"left": (76, 220, 92), "right": (42, 205, 255)}
HEAD_COLOR = (60, 80, 255)
OUTLINE_COLOR = (3, 3, 3)
HAND_BONE_THICKNESS = 2
HAND_BONE_OUTLINE_EXTRA = 2
HEAD_FRUSTUM_THICKNESS = 1
HEAD_FRUSTUM_OUTLINE_EXTRA = 2
WORLD_UP = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def ffprobe_count(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-8:
        out = np.asarray(fallback, dtype=np.float64)
        return out / max(float(np.linalg.norm(out)), 1e-8)
    return vec / norm


def gauge_arrays(blob: Any) -> tuple[np.ndarray, np.ndarray]:
    if "t_c2w" in blob.files:
        t = np.asarray(blob["t_c2w"], dtype=np.float64)
    elif "T_world_camera" in blob.files:
        mats = np.asarray(blob["T_world_camera"], dtype=np.float64)
        t = mats[:, :3, 3]
    else:
        raise RuntimeError("hand/world NPZ lacks t_c2w or T_world_camera")
    if t.ndim != 2 or t.shape[1] != 3 or not np.isfinite(t).all():
        raise RuntimeError(f"invalid camera/head trajectory shape: {t.shape}")
    return np.asarray(blob["frame_idx"], dtype=int), t


def collect_extent(blob: Any, frame_positions: list[int], camera_centers: np.ndarray, vertex_stride: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    pts = [camera_centers]
    for side in ("left", "right"):
        j_key = f"{side}_joints_world_m"
        v_key = f"{side}_vertices_world_m"
        valid_key = f"{side}_valid"
        if j_key in blob.files:
            hand_points = np.asarray(blob[j_key], dtype=np.float64)
            stride = 1
        elif v_key in blob.files:
            hand_points = np.asarray(blob[v_key], dtype=np.float64)
            stride = max(1, vertex_stride)
        else:
            continue
        valid = np.asarray(blob[valid_key], dtype=np.uint8) if valid_key in blob.files else np.ones(hand_points.shape[0], dtype=np.uint8)
        sampled = []
        for pos in frame_positions:
            if 0 <= pos < hand_points.shape[0] and int(valid[pos]):
                sampled.append(hand_points[pos, ::stride, :])
        if sampled:
            pts.append(np.concatenate(sampled, axis=0))
    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts if p.size], axis=0)
    finite = all_pts[np.isfinite(all_pts).all(axis=1)]
    if finite.size == 0:
        center = np.zeros(3, dtype=np.float64)
        lo = center - 0.5
        hi = center + 0.5
        return center, 1.0, lo, hi
    lo = np.percentile(finite, 2, axis=0)
    hi = np.percentile(finite, 98, axis=0)
    center = (lo + hi) / 2.0
    span = max(float((hi - lo).max()), 0.5)
    return center, span * 1.25, lo, hi


def estimate_forward(camera_centers: np.ndarray) -> np.ndarray:
    if len(camera_centers) >= 2:
        motion = camera_centers[-1] - camera_centers[0]
        motion = np.asarray([motion[0], 0.0, motion[2]], dtype=np.float64)
        if np.linalg.norm(motion) > 0.1:
            return normalize(motion, np.asarray([0.0, 0.0, 1.0]))
    centered = camera_centers - np.mean(camera_centers, axis=0, keepdims=True)
    centered[:, 1] = 0.0
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.asarray([0.0, 0.0, 1.0])
    if len(camera_centers) >= 2 and np.dot(axis, camera_centers[-1] - camera_centers[0]) < 0:
        axis = -axis
    axis[1] = 0.0
    return normalize(axis, np.asarray([0.0, 0.0, 1.0]))


def build_view(camera_centers: np.ndarray, center3: np.ndarray, span: float) -> dict[str, np.ndarray | float]:
    forward = estimate_forward(camera_centers)
    side = normalize(np.cross(WORLD_UP, forward), np.asarray([1.0, 0.0, 0.0]))
    target = center3 + forward * (0.18 * span)
    eye = center3 - forward * (1.25 * span) + side * (0.78 * span) + WORLD_UP * (0.42 * span)
    view_forward = normalize(target - eye, forward)
    right = normalize(np.cross(WORLD_UP, view_forward), side)
    true_up = normalize(np.cross(view_forward, right), WORLD_UP)
    return {"eye": eye, "target": target, "forward": view_forward, "right": right, "up": true_up, "trajectory_forward": forward, "side": side, "focal_scale": 4.2}


def project_perspective(points: np.ndarray, view: dict[str, np.ndarray | float], dims: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = dims
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    eye = np.asarray(view["eye"], dtype=np.float64)
    right = np.asarray(view["right"], dtype=np.float64)
    up = np.asarray(view["up"], dtype=np.float64)
    forward = np.asarray(view["forward"], dtype=np.float64)
    rel = pts - eye.reshape(1, 3)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    focal = float(view.get("focal_scale", 1.15)) * min(width, height)
    pix = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(z) & (z > 1e-4)
    pix[valid, 0] = width * 0.50 + focal * x[valid] / z[valid]
    pix[valid, 1] = height * 0.56 - focal * y[valid] / z[valid]
    return pix, z, valid


def draw_line3d(panel: np.ndarray, points: np.ndarray, view: dict[str, np.ndarray | float], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(points) < 2:
        return
    pix, _, valid = project_perspective(points, view, (panel.shape[1], panel.shape[0]))
    h, w = panel.shape[:2]
    for i in range(len(points) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        a, b = pix[i], pix[i + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        xa, ya = int(round(float(a[0]))), int(round(float(a[1])))
        xb, yb = int(round(float(b[0]))), int(round(float(b[1])))
        if -200 <= xa <= w + 200 and -200 <= xb <= w + 200 and -200 <= ya <= h + 200 and -200 <= yb <= h + 200:
            cv2.line(panel, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def draw_points3d(panel: np.ndarray, points: np.ndarray, view: dict[str, np.ndarray | float], color: tuple[int, int, int], radius: int, alpha: float = 1.0) -> int:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return 0
    pix, depth, valid = project_perspective(pts, view, (panel.shape[1], panel.shape[0]))
    order = np.argsort(depth)[::-1]
    overlay = panel.copy()
    h, w = panel.shape[:2]
    drawn = 0
    for i in order:
        if not valid[i] or not np.isfinite(pix[i]).all():
            continue
        x, y = int(round(float(pix[i, 0]))), int(round(float(pix[i, 1])))
        if -30 <= x <= w + 30 and -30 <= y <= h + 30:
            cv2.circle(overlay, (x, y), radius, color, -1, cv2.LINE_AA)
            drawn += 1
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0.0, dst=panel)
    else:
        panel[:] = overlay
    return drawn


def draw_skeleton_line3d(panel: np.ndarray, points: np.ndarray, view: dict[str, np.ndarray | float], color: tuple[int, int, int], thickness: int = 3, outline_extra: int = 2) -> None:
    draw_line3d(panel, points, view, OUTLINE_COLOR, thickness + outline_extra)
    draw_line3d(panel, points, view, color, thickness)


def draw_hand_joints3d(panel: np.ndarray, joints: np.ndarray, view: dict[str, np.ndarray | float], color: tuple[int, int, int]) -> int:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape[0] < 21 or joints.shape[1] != 3:
        return 0
    pix, _, valid = project_perspective(joints, view, (panel.shape[1], panel.shape[0]))
    h, w = panel.shape[:2]
    visible_edges = 0
    for a, b in HAND_EDGES:
        if a >= len(joints) or b >= len(joints) or not (valid[a] and valid[b]):
            continue
        pa, pb = pix[a], pix[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
            xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
            if -80 <= xa <= w + 80 and -80 <= xb <= w + 80 and -80 <= ya <= h + 80 and -80 <= yb <= h + 80:
                draw_skeleton_line3d(panel, np.asarray([joints[a], joints[b]]), view, color, HAND_BONE_THICKNESS, HAND_BONE_OUTLINE_EXTRA)
                visible_edges += 1
    return visible_edges


def draw_inset_line(panel: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    pa = (int(round(float(a[0]))), int(round(float(a[1]))))
    pb = (int(round(float(b[0]))), int(round(float(b[1]))))
    cv2.line(panel, pa, pb, OUTLINE_COLOR, thickness + 1, cv2.LINE_AA)
    cv2.line(panel, pa, pb, color, thickness, cv2.LINE_AA)


def hand_inset_pixels(joints: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    joints = np.asarray(joints, dtype=np.float64)
    center = np.nanmean(joints, axis=0)
    centered = joints - center.reshape(1, 3)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_x = normalize(vh[0], np.asarray([1.0, 0.0, 0.0]))
        axis_y = normalize(vh[1], WORLD_UP)
    except np.linalg.LinAlgError:
        axis_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        axis_y = WORLD_UP.copy()
    if abs(float(np.dot(axis_x, axis_y))) > 0.92:
        axis_y = WORLD_UP.copy()
    coords = np.column_stack([centered @ axis_x, centered @ axis_y])
    lo = np.nanmin(coords, axis=0)
    hi = np.nanmax(coords, axis=0)
    span = max(float((hi - lo).max()), 1e-4)
    scale = 0.82 * min(x1 - x0, y1 - y0) / span
    origin = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.54], dtype=np.float64)
    return origin + (coords - (lo + hi).reshape(1, 2) * 0.5) * np.asarray([scale, -scale])


def draw_hand_zoom_inset(panel: np.ndarray, hand_joints: dict[str, np.ndarray]) -> bool:
    valid_hands = {side: np.asarray(joints, dtype=np.float64) for side, joints in hand_joints.items() if np.asarray(joints).shape == (21, 3)}
    if not valid_hands:
        return False
    h, w = panel.shape[:2]
    inset_w = min(520, max(420, int(w * 0.40)))
    inset_h = min(310, max(240, int(h * 0.40)))
    x0 = w - inset_w - 22
    y0 = 54
    x1 = x0 + inset_w
    y1 = y0 + inset_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.90, panel, 0.10, 0.0, dst=panel)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (92, 98, 105), 1, cv2.LINE_AA)
    cv2.putText(panel, "line-only hand skeletons", (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    content_top = y0 + 38
    pad = 18
    ordered = [side for side in ("left", "right") if side in valid_hands]
    if len(ordered) == 1:
        rects = {ordered[0]: (x0 + pad, content_top + pad, x1 - pad, y1 - pad)}
    else:
        mid = (x0 + x1) // 2
        rects = {
            "left": (x0 + pad, content_top + pad, mid - pad // 2, y1 - pad),
            "right": (mid + pad // 2, content_top + pad, x1 - pad, y1 - pad),
        }
        cv2.line(panel, (mid, content_top + 4), (mid, y1 - 8), (62, 68, 75), 1, cv2.LINE_AA)
    for side in ordered:
        rect = rects[side]
        pix = hand_inset_pixels(valid_hands[side], rect)
        color = COLORS[side]
        cv2.putText(panel, "L" if side == "left" else "R", (rect[0], content_top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        for a, b in HAND_EDGES:
            draw_inset_line(panel, pix[a], pix[b], color, 2)
    return True


def draw_floor_grid(panel: np.ndarray, view: dict[str, np.ndarray | float], lo: np.ndarray, hi: np.ndarray, center3: np.ndarray, span: float) -> None:
    floor_y = float(lo[1] - 0.08 * span)
    step = 0.25 if span <= 2.5 else 0.5 if span <= 6.0 else 1.0
    x0 = math.floor((lo[0] - 0.20 * span) / step) * step
    x1 = math.ceil((hi[0] + 0.20 * span) / step) * step
    z0 = math.floor((lo[2] - 0.20 * span) / step) * step
    z1 = math.ceil((hi[2] + 0.20 * span) / step) * step
    xs = np.arange(x0, x1 + step * 0.5, step)
    zs = np.arange(z0, z1 + step * 0.5, step)
    for x in xs:
        pts = np.asarray([[x, floor_y, z0], [x, floor_y, z1]], dtype=np.float64)
        draw_line3d(panel, pts, view, (58, 68, 78), 1)
    for z in zs:
        pts = np.asarray([[x0, floor_y, z], [x1, floor_y, z]], dtype=np.float64)
        draw_line3d(panel, pts, view, (58, 68, 78), 1)
    axis_len = min(max(span * 0.18, 0.25), 1.0)
    origin = np.asarray([center3[0], floor_y, center3[2]], dtype=np.float64)
    draw_line3d(panel, np.asarray([origin, origin + np.asarray([axis_len, 0.0, 0.0])]), view, (70, 90, 240), 3)
    draw_line3d(panel, np.asarray([origin, origin + np.asarray([0.0, axis_len, 0.0])]), view, (70, 220, 90), 3)
    draw_line3d(panel, np.asarray([origin, origin + np.asarray([0.0, 0.0, axis_len])]), view, (230, 190, 70), 3)


def camera_axes(rotation: np.ndarray | None, view: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape == (3, 3) and np.isfinite(rot).all():
            return normalize(rot[:, 0], np.asarray(view["right"], dtype=np.float64)), normalize(rot[:, 1], WORLD_UP), normalize(rot[:, 2], np.asarray(view["forward"], dtype=np.float64))
    return np.asarray(view["right"], dtype=np.float64), WORLD_UP, np.asarray(view["forward"], dtype=np.float64)


def draw_camera_marker(panel: np.ndarray, center: np.ndarray, rotation: np.ndarray | None, view: dict[str, np.ndarray | float], span: float) -> None:
    x_axis, y_axis, z_axis = camera_axes(rotation, view)
    scale = min(max(0.055 * float(span), 0.10), 0.28)
    front = center + z_axis * (scale * 1.15)
    half_w = scale * 0.55
    half_h = scale * 0.36
    corners = np.asarray(
        [
            front + x_axis * half_w + y_axis * half_h,
            front - x_axis * half_w + y_axis * half_h,
            front - x_axis * half_w - y_axis * half_h,
            front + x_axis * half_w - y_axis * half_h,
        ],
        dtype=np.float64,
    )
    for corner in corners:
        draw_skeleton_line3d(panel, np.asarray([center, corner]), view, HEAD_COLOR, HEAD_FRUSTUM_THICKNESS, HEAD_FRUSTUM_OUTLINE_EXTRA)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        draw_skeleton_line3d(panel, np.asarray([corners[a], corners[b]]), view, HEAD_COLOR, HEAD_FRUSTUM_THICKNESS, HEAD_FRUSTUM_OUTLINE_EXTRA)
    draw_skeleton_line3d(panel, np.asarray([center, center + x_axis * scale]), view, (70, 100, 255), HEAD_FRUSTUM_THICKNESS, HEAD_FRUSTUM_OUTLINE_EXTRA)
    draw_skeleton_line3d(panel, np.asarray([center, center + y_axis * scale]), view, (70, 230, 90), HEAD_FRUSTUM_THICKNESS, HEAD_FRUSTUM_OUTLINE_EXTRA)
    draw_skeleton_line3d(panel, np.asarray([center, center + z_axis * scale]), view, (255, 210, 80), HEAD_FRUSTUM_THICKNESS, HEAD_FRUSTUM_OUTLINE_EXTRA)


def render(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    raw_manifest = load_json(args.raw_frame_manifest or (run_root / "input" / "raw_frame_manifest" / "manifest.json"))
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    if not frames:
        raise RuntimeError("raw frame manifest contains no frames")
    hybrid_npz = args.hybrid_npz or (run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz")
    blob = np.load(hybrid_npz, allow_pickle=True)
    frame_idx, camera_centers = gauge_arrays(blob)
    rotations = np.asarray(blob["R_c2w"], dtype=np.float64) if "R_c2w" in blob.files else None
    frame_to_pos = {int(idx): pos for pos, idx in enumerate(frame_idx.tolist())}
    fps = finite_positive(raw_manifest.get("fps")) or finite_positive((raw_manifest.get("video") or {}).get("fps")) or 30.0
    width = int(args.width or 960)
    height = int(args.height or 540)
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    positions_for_extent = list(range(0, len(frame_idx), max(1, len(frame_idx) // 96)))
    center3, span, lo, hi = collect_extent(blob, positions_for_extent, camera_centers, int(args.vertex_stride))
    view = build_view(camera_centers, center3, span)
    output = args.output or (run_root / "renders" / "v22_world_head_hand_3d.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    counts = {
        "frames": 0,
        "frames_with_camera_pose": 0,
        "head_camera_marker_frames": 0,
        "left_skeleton_frames": 0,
        "right_skeleton_frames": 0,
        "left_skeleton_primitives": 0,
        "right_skeleton_primitives": 0,
        "hand_zoom_inset_frames": 0,
        "surface_points_rendered": bool(args.draw_surface_points),
        "surface_points_drawn_per_valid_hand": int(math.ceil(778 / max(1, int(args.vertex_stride)))) if args.draw_surface_points else 0,
    }
    try:
        for out_i, frame in enumerate(frames):
            idx = int(frame.get("frame_idx", frame.get("index", out_i)))
            canvas = np.full((height, width, 3), (28, 31, 34), dtype=np.uint8)
            draw_floor_grid(canvas, view, lo, hi, center3, span)
            pos = frame_to_pos.get(idx)
            if pos is not None:
                counts["frames_with_camera_pose"] += 1
                history = camera_centers[: pos + 1]
                draw_line3d(canvas, history, view, (90, 95, 190), 2)
                rotation = rotations[pos] if rotations is not None and pos < len(rotations) else None
                draw_camera_marker(canvas, camera_centers[pos], rotation, view, span)
                counts["head_camera_marker_frames"] += 1
                current_hand_joints: dict[str, np.ndarray] = {}
                for side_name in ("left", "right"):
                    valid_key = f"{side_name}_valid"
                    if valid_key in blob.files and int(np.asarray(blob[valid_key])[pos]) == 0:
                        continue
                    j_key = f"{side_name}_joints_world_m"
                    if j_key not in blob.files:
                        continue
                    joints = np.asarray(blob[j_key], dtype=np.float64)[pos]
                    current_hand_joints[side_name] = joints
                    color = COLORS[side_name]
                    if args.draw_surface_points:
                        v_key = f"{side_name}_vertices_world_m"
                        if v_key in blob.files:
                            vertices = np.asarray(blob[v_key], dtype=np.float64)[pos, :: max(1, int(args.vertex_stride)), :]
                            draw_points3d(canvas, vertices, view, color, radius=1, alpha=0.18)
                    primitive_count = draw_hand_joints3d(canvas, joints, view, color)
                    if primitive_count:
                        counts[f"{side_name}_skeleton_frames"] += 1
                        counts[f"{side_name}_skeleton_primitives"] += int(primitive_count)
                if draw_hand_zoom_inset(canvas, current_hand_joints):
                    counts["hand_zoom_inset_frames"] += 1
            cv2.putText(canvas, "3D world view: side-rear camera looking forward", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {idx:06d} | metric gauge: HaWoR/hybrid world | units: meters", (16, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, "red=head/camera frustum/path | green=left MANO skeleton | yellow=right MANO skeleton | floor grid is metric", (16, height - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 218, 228), 1, cv2.LINE_AA)
            writer.write(canvas)
            counts["frames"] += 1
    finally:
        writer.release()
    report = {
        "schema": "v22_world_head_hand_3d_render.v2",
        "status": "ok",
        "method": "render_v22_world_head_hand_3d",
        "claim_scope": "Metric world-coordinate perspective visualization driven by HaWoR/hybrid camera/head trajectory and MANO hand joints. Hands are rendered as 21-point MANO skeletons by default; surface points are an opt-in debug layer. It is a render of current physical state, not a metric accuracy certificate or contact proof.",
        "run_root": str(run_root),
        "hybrid_npz": str(hybrid_npz),
        "output_video": str(output),
        "frame_count": len(frames),
        "video_frame_count": ffprobe_count(output),
        "fps": float(fps),
        "render_size": [width, height],
        "view_mode": "perspective_side_rear_looking_forward_along_estimated_camera_trajectory",
        "world_gauge_source": "hybrid_or_hawor_npz_R_c2w_t_c2w_and_joints_world_m",
        "metric_extent_center_xyz_m": center3.astype(float).tolist(),
        "metric_extent_span_m": float(span),
        "view_eye_xyz_m": np.asarray(view["eye"], dtype=float).tolist(),
        "view_target_xyz_m": np.asarray(view["target"], dtype=float).tolist(),
        "view_trajectory_forward_xyz": np.asarray(view["trajectory_forward"], dtype=float).tolist(),
        "hand_render_style": "mano_21_bone_lines_only_no_keypoints_with_world_joint_zoom_inset",
        "head_render_style": "camera_frustum_axes_and_trajectory",
        "draw_counts": counts,
    }
    write_json(args.report_json or (output.parent / "v22_world_head_hand_3d_report.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hybrid-npz", type=Path, default=None)
    parser.add_argument("--raw-frame-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--vertex-stride", type=int, default=20)
    parser.add_argument("--draw-surface-points", action="store_true", help="Optional debug layer. Default output uses MANO joint skeletons only.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
