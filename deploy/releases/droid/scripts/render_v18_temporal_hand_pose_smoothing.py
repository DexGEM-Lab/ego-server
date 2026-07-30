#!/usr/bin/env python3
"""Render temporal smoothing of graph-shifted MANO 2D joint tracks.

This targets one narrow V18 gap: visible hand pose jitter. It smooths projected
MANO joints in image space after applying the V18 graph-smoothed hand center. It
is not a 3D MANO optimization or accepted physical hand pose.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bb = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=bg)
    draw.text((x, y), text, fill=fill, font=fnt)


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_center(value: Any) -> tuple[float, float] | None:
    box = bbox_tuple(value)
    if box is None:
        return None
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def graph_hand_estimates(frame: dict[str, Any], source_w: float, source_h: float) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("hand_state", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        est = row.get("estimate")
        if vid.startswith("hand::") and isinstance(est, list) and len(est) >= 2:
            out[vid.split("::", 1)[1]] = (finite_float(est[0]) * source_w, finite_float(est[1]) * source_h)
    return out


def project_mano_joints(mano: dict[str, Any], source_w: float, source_h: float) -> np.ndarray | None:
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics") or [2304.0, 2304.0, source_w / 2.0, source_h / 2.0]
    if not (isinstance(joints, list) and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return None
    fx, fy, cx, cy = [finite_float(v) for v in intr]
    pts = []
    for raw in joints[:21]:
        if not (isinstance(raw, list) and len(raw) == 3):
            return None
        x = finite_float(raw[0]) + finite_float(cam_t[0])
        y = finite_float(raw[1]) + finite_float(cam_t[1])
        z = finite_float(raw[2]) + finite_float(cam_t[2])
        if z <= 1e-6:
            return None
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        pts.append([u, v])
    if len(pts) != 21:
        return None
    return np.asarray(pts, dtype=np.float64)


def collect_tracks(frames: list[Any], source_w: float, source_h: float) -> dict[str, dict[int, np.ndarray]]:
    tracks: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        graph = graph_hand_estimates(frame, source_w, source_h)
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side"))
            if side not in tracks or side not in graph:
                continue
            obs_center = bbox_center(hand.get("bbox_xyxy"))
            if obs_center is None:
                continue
            mano = hand.get("mano_candidate", {}) if isinstance(hand.get("mano_candidate"), dict) else {}
            pts = project_mano_joints(mano, source_w, source_h)
            if pts is None:
                continue
            dx = graph[side][0] - obs_center[0]
            dy = graph[side][1] - obs_center[1]
            tracks[side][frame_idx] = pts + np.asarray([dx, dy], dtype=np.float64)[None, :]
    return tracks


def smooth_tracks(tracks: dict[str, dict[int, np.ndarray]], radius: int) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    for side, rows in tracks.items():
        frame_ids = sorted(rows)
        for frame_idx in frame_ids:
            nearby = [rows[j] for j in frame_ids if abs(j - frame_idx) <= radius]
            if not nearby:
                continue
            out[side][frame_idx] = np.median(np.stack(nearby, axis=0), axis=0)
    return out


def out_of_bounds_count(pts: np.ndarray, source_w: float, source_h: float) -> int:
    out = (pts[:, 0] < 0.0) | (pts[:, 0] > source_w) | (pts[:, 1] < 0.0) | (pts[:, 1] > source_h)
    return int(np.sum(out))


def constrain_smoothed_tracks(
    tracks: dict[str, dict[int, np.ndarray]],
    candidates: dict[str, dict[int, np.ndarray]],
    source_w: float,
    source_h: float,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[tuple[int, str], dict[str, Any]]]:
    out: dict[str, dict[int, np.ndarray]] = {"left": {}, "right": {}}
    meta: dict[tuple[int, str], dict[str, Any]] = {}
    for side, rows in tracks.items():
        for frame_idx, raw_pts in rows.items():
            cand = candidates.get(side, {}).get(frame_idx)
            if cand is None:
                continue
            shifts = np.linalg.norm(cand - raw_pts, axis=1)
            centroid_shift = float(np.linalg.norm(np.mean(cand, axis=0) - np.mean(raw_pts, axis=0)))
            root_shift = float(np.linalg.norm(cand[0] - raw_pts[0]))
            max_shift = float(np.max(shifts))
            raw_oob = out_of_bounds_count(raw_pts, source_w, source_h)
            cand_oob = out_of_bounds_count(cand, source_w, source_h)
            reject_reasons: list[str] = []
            if max_shift > args.max_smoothing_joint_shift_px:
                reject_reasons.append("candidate_exceeds_joint_shift_gate")
            if centroid_shift > args.max_smoothing_centroid_shift_px:
                reject_reasons.append("candidate_exceeds_centroid_shift_gate")
            if root_shift > args.max_smoothing_root_shift_px:
                reject_reasons.append("candidate_exceeds_root_shift_gate")
            if cand_oob > 0:
                reject_reasons.append("candidate_has_out_of_source_frame_joints")
            if reject_reasons:
                status = "temporal_smoothing_rejected_raw_graph_shifted_mano2d_retained"
                pts = raw_pts
                applied = False
            else:
                status = "temporal_smoothed_graph_shifted_mano2d"
                pts = cand
                applied = True
            out[side][frame_idx] = pts
            meta[(frame_idx, side)] = {
                "status": status,
                "temporal_filter_applied": applied,
                "reject_reasons": reject_reasons,
                "max_joint_shift_from_graph_shifted_input_px": max_shift,
                "centroid_shift_from_graph_shifted_input_px": centroid_shift,
                "root_shift_from_graph_shifted_input_px": root_shift,
                "candidate_out_of_source_frame_joint_count": cand_oob,
                "raw_out_of_source_frame_joint_count": raw_oob,
                "output_out_of_source_frame_joint_count": out_of_bounds_count(pts, source_w, source_h),
            }
    return out, meta


def accel_stats(rows: dict[int, np.ndarray]) -> dict[str, Any]:
    vals_raw: list[float] = []
    ids = sorted(rows)
    by_id = rows
    for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
        if b != a + 1 or c != b + 1:
            continue
        acc = by_id[c] - 2.0 * by_id[b] + by_id[a]
        vals_raw.extend(np.linalg.norm(acc, axis=1).tolist())
    if not vals_raw:
        return {"sample_count": 0, "mean_joint_accel_px": None, "p95_joint_accel_px": None}
    vals = np.asarray(vals_raw, dtype=np.float64)
    return {"sample_count": int(len(vals)), "mean_joint_accel_px": float(np.mean(vals)), "p95_joint_accel_px": float(np.percentile(vals, 95))}


def draw_hand(draw: ImageDraw.ImageDraw, pts: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    xy = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAND_EDGES:
        draw.line((xy[a][0], xy[a][1], xy[b][0], xy[b][1]), fill=color, width=width)
    for x, y in xy:
        draw.ellipse((x - width, y - width, x + width, y + width), fill=color)


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}", "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path),
    ], check=True)


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    raw_video = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    source_w = finite_float(raw_video.get("width"), 1920.0)
    source_h = finite_float(raw_video.get("height"), 1080.0)
    fps = finite_float(ann.get("fps"), 30.0)
    tracks = collect_tracks(frames, source_w, source_h)
    candidates = smooth_tracks(tracks, args.smoothing_radius_frames)
    smoothed, smoothing_meta = constrain_smoothed_tracks(tracks, candidates, source_w, source_h, args)
    case_dir = args.output_root / case / "temporal_hand_pose_smoothing"
    frame_dir = case_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    counts: Counter[str] = Counter()
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        idx = int(frame.get("frame_idx", 0))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw_path).convert("RGB") if raw_path.exists() else Image.new("RGB", (int(source_w), int(source_h)), (12, 12, 12))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.size[0], 46), fill=(0, 0, 0))
        draw.text((12, 11), f"V18 temporal 2D MANO smoothing {case} frame {idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        sx = image.size[0] / source_w if source_w > 0 else 1.0
        sy = image.size[1] / source_h if source_h > 0 else 1.0
        scale = np.asarray([sx, sy], dtype=np.float64)[None, :]
        for side, color in [("left", (60, 255, 120)), ("right", (255, 210, 60))]:
            raw_pts = tracks[side].get(idx)
            smooth_pts = smoothed[side].get(idx)
            if raw_pts is not None:
                draw_hand(draw, raw_pts * scale, (150, 150, 150), 2)
                counts[f"raw_graph_shifted::{side}"] += 1
            if smooth_pts is not None:
                meta = smoothing_meta.get((idx, side), {})
                applied = meta.get("temporal_filter_applied") is True
                out_color = color if applied else (255, 110, 80)
                draw_hand(draw, smooth_pts * scale, out_color, 4)
                counts[f"temporal_filter::{meta.get('status', 'missing_status')}"] += 1
                draw_label(draw, (12, 54 if side == "left" else 74), f"{side}: colored=accepted temporal filter, red=input retained by anchor/bounds gate; gray=graph-shifted input", small, out_color, (0, 0, 0))
        image.save(frame_dir / f"{idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_temporal_hand_pose_smoothing.mp4"
    encode_video(frame_dir, video_path, fps)
    jitter: dict[str, Any] = {}
    for side in ["left", "right"]:
        raw_stats = accel_stats(tracks[side])
        smooth_stats = accel_stats(smoothed[side])
        reduction = None
        if isinstance(raw_stats.get("mean_joint_accel_px"), float) and isinstance(smooth_stats.get("mean_joint_accel_px"), float) and raw_stats["mean_joint_accel_px"] > 1e-9:
            reduction = 1.0 - smooth_stats["mean_joint_accel_px"] / raw_stats["mean_joint_accel_px"]
        jitter[side] = {"graph_shifted_input": raw_stats, "temporal_smoothed": smooth_stats, "mean_accel_reduction_fraction": reduction}
    smoothed_rows = []
    for side in ["left", "right"]:
        for frame_idx, pts in sorted(smoothed[side].items()):
            meta = smoothing_meta.get((frame_idx, side), {})
            smoothed_rows.append({
                "frame_idx": frame_idx,
                "hand_side": side,
                "joints2d_source_px": np.round(pts, 3).tolist(),
                "status": meta.get("status", "missing_temporal_filter_status"),
                "temporal_filter_applied": meta.get("temporal_filter_applied"),
                "reject_reasons": meta.get("reject_reasons", []),
                "max_joint_shift_from_graph_shifted_input_px": meta.get("max_joint_shift_from_graph_shifted_input_px"),
                "centroid_shift_from_graph_shifted_input_px": meta.get("centroid_shift_from_graph_shifted_input_px"),
                "root_shift_from_graph_shifted_input_px": meta.get("root_shift_from_graph_shifted_input_px"),
                "candidate_out_of_source_frame_joint_count": meta.get("candidate_out_of_source_frame_joint_count"),
                "raw_out_of_source_frame_joint_count": meta.get("raw_out_of_source_frame_joint_count"),
                "output_out_of_source_frame_joint_count": meta.get("output_out_of_source_frame_joint_count"),
                "accepted_3d_mano_pose": False,
                "state_role": "image_space_temporal_filter_with_anchor_and_bounds_gate_not_3d_mano_optimization",
            })
    report = {
        "method": "render_v18_temporal_hand_pose_smoothing",
        "case": case,
        "claim_scope": "image_space_temporal_smoothing_of_graph_shifted_mano2d_joints_not_3d_mano_optimization_or_physical_pose_acceptance",
        "frame_count": len(frames),
        "fps": fps,
        "parameters": {
            "smoothing_radius_frames": args.smoothing_radius_frames,
            "max_smoothing_joint_shift_px": args.max_smoothing_joint_shift_px,
            "max_smoothing_centroid_shift_px": args.max_smoothing_centroid_shift_px,
            "max_smoothing_root_shift_px": args.max_smoothing_root_shift_px,
        },
        "draw_counts": dict(sorted(counts.items())),
        "smoothed_rows": smoothed_rows,
        "jitter_probe": jitter,
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_temporal_hand_pose_smoothing_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_temporal_hand_pose_smoothing",
        "status": "temporal_hand_pose_smoothing_not_3d_pose_acceptance",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_temporal_hand_pose_smoothing_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--smoothing-radius-frames", type=int, default=3)
    parser.add_argument("--max-smoothing-joint-shift-px", type=float, default=120.0)
    parser.add_argument("--max-smoothing-centroid-shift-px", type=float, default=80.0)
    parser.add_argument("--max-smoothing-root-shift-px", type=float, default=120.0)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
