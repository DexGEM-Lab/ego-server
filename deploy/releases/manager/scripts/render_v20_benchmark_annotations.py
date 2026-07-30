#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, write_json

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLORS_BGR = [
    (60, 80, 255),
    (80, 230, 80),
    (255, 150, 60),
    (60, 220, 255),
    (255, 80, 220),
    (180, 120, 255),
    (120, 255, 220),
]


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def project(points: Any, intr: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    intr_arr = np.asarray(intr, dtype=float).reshape(-1)
    if intr_arr.size != 4:
        return np.zeros((0, 2), dtype=np.float32)
    fx, fy, cx, cy = intr_arr
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1.0e-6)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = (fx * pts[valid, 0] / z[valid] + cx).astype(np.float32)
    uv[valid, 1] = (fy * pts[valid, 1] / z[valid] + cy).astype(np.float32)
    return uv


def draw_text(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.45) -> None:
    x, y = xy
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_mask(img: np.ndarray, mask_path: str | None, color: tuple[int, int, int]) -> bool:
    if not mask_path:
        return False
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return False
    tint = np.zeros_like(img)
    tint[:, :] = np.asarray(color, dtype=np.uint8)
    blended = cv2.addWeighted(img, 0.58, tint, 0.42, 0.0)
    img[mask_bool] = blended[mask_bool]
    return True


def draw_object_overlay(img: np.ndarray, obj: dict[str, Any], color: tuple[int, int, int]) -> dict[str, int]:
    counts = {"masks": 0, "boxes": 0, "points": 0}
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    if draw_mask(img, geom.get("mask_path"), color):
        counts["masks"] += 1
    box = obj.get("bbox_xyxy") or geom.get("bbox_xyxy")
    if isinstance(box, list) and len(box) == 4 and all(math.isfinite(finite_float(v, float("nan"))) for v in box):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = str(obj.get("object_name") or obj.get("object_id"))[:32]
        draw_text(img, label, (max(4, x1), max(18, y1 - 4)), color)
        counts["boxes"] += 1
    intr = None
    points = geom.get("points_camera_sample_m")
    if isinstance(points, list):
        frame_camera = obj.get("_frame_camera_intrinsics")
        intr = frame_camera
    if intr is not None:
        uv = project(points, intr)
        valid = np.isfinite(uv).all(axis=1)
        for x, y in uv[valid][:: max(1, int(np.count_nonzero(valid) / 64))]:
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (int(round(x)), int(round(y))), 1, color, -1, cv2.LINE_AA)
                counts["points"] += 1
    return counts


def draw_hand_overlay(img: np.ndarray, hand: dict[str, Any], color: tuple[int, int, int]) -> int:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
    aligned_uv = metric.get("joints_2d_px_detector_aligned")
    if aligned_uv is not None:
        uv = np.asarray(aligned_uv, dtype=float)
    else:
        joints = metric.get("joints_current_v18_camera_m")
        uv = project(joints, intr)
    if uv.shape[0] != 21 or not np.isfinite(uv).any():
        return 0
    for a, b in HAND_EDGES:
        if np.isfinite(uv[[a, b]]).all():
            pa = tuple(np.round(uv[a]).astype(int))
            pb = tuple(np.round(uv[b]).astype(int))
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for i, pt in enumerate(uv):
        if not np.isfinite(pt).all():
            continue
        cv2.circle(img, tuple(np.round(pt).astype(int)), 3 if i else 5, color, -1, cv2.LINE_AA)
    side = str(hand.get("hand_side") or hand.get("side") or "hand")
    wrist = uv[0]
    if np.isfinite(wrist).all():
        draw_text(img, side, (int(wrist[0]) + 6, int(wrist[1]) - 6), color)
    return 1


def metric_bounds(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for frame in frames:
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            pts = np.asarray(geom.get("points_world_sample_m", []), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.size:
                chunks.append(pts[np.isfinite(pts).all(axis=1)])
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            pts = np.asarray(metric.get("joints_current_v18_world_m", []), dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.size:
                chunks.append(pts[np.isfinite(pts).all(axis=1)])
    if not chunks:
        return np.asarray([-0.5, -0.5, 0.0]), np.asarray([0.5, 0.5, 1.0])
    pts = np.concatenate([c for c in chunks if c.size], axis=0)
    if pts.size == 0:
        return np.asarray([-0.5, -0.5, 0.0]), np.asarray([0.5, 0.5, 1.0])
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    span = np.maximum(hi - lo, 0.1)
    pad = span * 0.12
    return lo - pad, hi + pad


def world_xy(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    x = int(round(70 + (point[0] - lo[0]) / max(1e-6, hi[0] - lo[0]) * (width - 160)))
    y = int(round(height - 70 - (point[1] - lo[1]) / max(1e-6, hi[1] - lo[1]) * (height - 150)))
    return x, y


def render_world_frame(frame: dict[str, Any], lo: np.ndarray, hi: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    img = np.full((height, width, 3), (22, 24, 29), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width, 44), (0, 0, 0), -1)
    draw_text(img, f"V20 benchmark world frame {frame.get('frame_idx')} — prediction-side geometry", (12, 28), (255, 255, 255), 0.55)
    cv2.rectangle(img, (70, 70), (width - 90, height - 70), (55, 55, 62), 1)
    for oi, obj in enumerate(frame.get("objects", []) if isinstance(frame.get("objects"), list) else []):
        color = COLORS_BGR[oi % len(COLORS_BGR)]
        geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
        pts = np.asarray(geom.get("points_world_sample_m", []), dtype=float)
        drawn = 0
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.size:
            valid = pts[np.isfinite(pts).all(axis=1)]
            step = max(1, valid.shape[0] // 120)
            for p in valid[::step]:
                xy = world_xy(p, lo, hi, width, height)
                if xy is None:
                    continue
                cv2.circle(img, xy, 1, color, -1, cv2.LINE_AA)
                drawn += 1
            center = valid.mean(axis=0) if valid.size else None
            if center is not None:
                xy = world_xy(center, lo, hi, width, height)
                if xy is not None:
                    cv2.circle(img, xy, 7, color, -1, cv2.LINE_AA)
                    draw_text(img, str(obj.get("object_name") or obj.get("object_id"))[:24], (xy[0] + 8, xy[1] - 6), color)
        if drawn == 0:
            y = 70 + 18 * oi
            draw_text(img, f"{obj.get('object_name')}: no visible surface", (width - 360, y), color)
    for hi_idx, hand in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
        metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
        joints = np.asarray(metric.get("joints_current_v18_world_m", []), dtype=float)
        color = (80, 255, 255) if str(hand.get("hand_side")) == "right" else (255, 180, 80)
        if joints.shape == (21, 3):
            pts2 = [world_xy(p, lo, hi, width, height) for p in joints]
            for a, b in HAND_EDGES:
                if pts2[a] is not None and pts2[b] is not None:
                    cv2.line(img, pts2[a], pts2[b], color, 2, cv2.LINE_AA)
            for pt in pts2:
                if pt is not None:
                    cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)
            if pts2[0] is not None:
                draw_text(img, str(hand.get("hand_side")), (pts2[0][0] + 8, pts2[0][1] - 8), color)
    return img


def encode_side_by_side(overlay: Path, world: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(overlay), "-i", str(world),
        "-filter_complex", "[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[left];[1:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
        return
    except FileNotFoundError:
        pass
    cap_l = cv2.VideoCapture(str(overlay))
    cap_r = cv2.VideoCapture(str(world))
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise ContractError("could_not_open_inputs_for_opencv_side_by_side")
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 10.0
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (1920, 540))
    if not writer.isOpened():
        raise ContractError(f"could_not_open_side_by_side_writer: {output}")
    try:
        while True:
            ok_l, left = cap_l.read()
            ok_r, right = cap_r.read()
            if not ok_l or not ok_r:
                break
            left = cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (960, 540), interpolation=cv2.INTER_AREA)
            writer.write(np.hstack([left, right]))
    finally:
        writer.release()
        cap_l.release()
        cap_r.release()


def frame_count(path: Path) -> int | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def render(args: argparse.Namespace) -> dict[str, Any]:
    ann = load_json(args.annotations)
    ensure_no_gt_in_prediction(ann, "render_annotations")
    frames = ann.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("annotations_have_no_frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / "v20_overlay.mp4"
    world_path = args.output_dir / "v20_world.mp4"
    side_path = args.output_dir / "v20_side_by_side.mp4"
    fps = finite_float(ann.get("fps"), args.fps)
    first = cv2.imread(str(frames[0].get("rgb_path")), cv2.IMREAD_COLOR)
    if first is None:
        raise ContractError(f"could_not_read_first_rgb: {frames[0].get('rgb_path')}")
    h, w = first.shape[:2]
    overlay_writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    world_size = (args.world_width, args.world_height)
    world_writer = cv2.VideoWriter(str(world_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, world_size)
    if not overlay_writer.isOpened() or not world_writer.isOpened():
        raise ContractError("could_not_open_v20_video_writers")
    lo, hi = metric_bounds(frames)
    counts: dict[str, int] = {"overlay_masks": 0, "overlay_boxes": 0, "overlay_object_points": 0, "overlay_hands": 0}
    try:
        for frame_i, frame in enumerate(frames):
            image = cv2.imread(str(frame.get("rgb_path")), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError(f"could_not_read_rgb: {frame.get('rgb_path')}")
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            intr = frame.get("camera", {}).get("intrinsics_fx_fy_cx_cy") if isinstance(frame.get("camera"), dict) else None
            for oi, obj in enumerate(frame.get("objects", []) if isinstance(frame.get("objects"), list) else []):
                if isinstance(obj, dict):
                    obj["_frame_camera_intrinsics"] = intr
                    c = draw_object_overlay(image, obj, COLORS_BGR[oi % len(COLORS_BGR)])
                    counts["overlay_masks"] += c["masks"]
                    counts["overlay_boxes"] += c["boxes"]
                    counts["overlay_object_points"] += c["points"]
                    obj.pop("_frame_camera_intrinsics", None)
            for hand_i, hand in enumerate(frame.get("hands", []) if isinstance(frame.get("hands"), list) else []):
                color = (80, 255, 255) if str(hand.get("hand_side")) == "right" else (255, 180, 80)
                counts["overlay_hands"] += draw_hand_overlay(image, hand, color)
            cv2.rectangle(image, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.putText(image, f"V20 benchmark prediction frame {frame.get('frame_idx')} ({frame_i+1}/{len(frames)})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            overlay_writer.write(image)
            world_writer.write(render_world_frame(frame, lo, hi, world_size))
    finally:
        overlay_writer.release()
        world_writer.release()
    encode_side_by_side(overlay_path, world_path, side_path)
    summary = {
        "status": "ok",
        "method": "render_v20_benchmark_annotations",
        "annotations": str(args.annotations),
        "overlay_video": str(overlay_path),
        "world_video": str(world_path),
        "side_by_side_video": str(side_path),
        "expected_frame_count": len(frames),
        "overlay_frame_count": frame_count(overlay_path),
        "world_frame_count": frame_count(world_path),
        "side_by_side_frame_count": frame_count(side_path),
        "frame_count_match": frame_count(overlay_path) == frame_count(world_path) == frame_count(side_path) == len(frames),
        "draw_counts": counts,
        "claim_scope": "Direct V20 benchmark renderer for prediction-side annotations; it does not consume evaluation eval refs.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render full-duration V20 benchmark prediction annotations directly.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--world-width", type=int, default=1280)
    parser.add_argument("--world-height", type=int, default=720)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
