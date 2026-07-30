#!/usr/bin/env python3
"""Render full-timeline recovered MANO foundation overlays for V18."""

from __future__ import annotations

import argparse
import json
import math
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
HAND_COLORS = {"left": (70, 255, 120), "right": (255, 210, 60)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def arr(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if shape is not None and tuple(out.shape) != shape:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bb = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=bg)
    draw.text((x, y), text, fill=fill, font=fnt)


def project(points_local: np.ndarray, cam_t: np.ndarray, focal: Any, width: float, height: float) -> np.ndarray:
    if isinstance(focal, list) and len(focal) >= 2:
        fx, fy = finite_float(focal[0], 2304.0), finite_float(focal[1], 2304.0)
    else:
        fx = fy = finite_float(focal, 2304.0)
    cx, cy = width / 2.0, height / 2.0
    pts = points_local.astype(np.float64) + cam_t.astype(np.float64)[None, :]
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    ok = z > 1e-6
    out[ok, 0] = fx * pts[ok, 0] / z[ok] + cx
    out[ok, 1] = fy * pts[ok, 1] / z[ok] + cy
    return out


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


def measurement_manifest(case: str, root: Path) -> dict[str, Any]:
    return load_json(root / case / "v17_measurement_manifest.json")


def wilor_path(case: str, root: Path) -> Path:
    manifest = measurement_manifest(case, root)
    return Path(str(manifest["wilor_raw"]))


def best_wilor_rows(path: Path, frame_count: int) -> dict[tuple[int, str], dict[str, Any]]:
    payload = load_json(path)
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in payload.get("frames", []) if isinstance(payload, dict) else []:
        if not isinstance(frame, dict):
            continue
        frame_idx = frame.get("frame_idx")
        if not isinstance(frame_idx, int) or frame_idx < 0 or frame_idx >= frame_count:
            continue
        for hand in frame.get("raw_hands", []) if isinstance(frame.get("raw_hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = hand.get("side")
            if side not in {"left", "right"}:
                continue
            key = (frame_idx, str(side))
            score = finite_float(hand.get("detector_score"), 0.0)
            cur = best.get(key)
            if cur is None or score > finite_float(cur.get("detector_score"), 0.0):
                best[key] = hand
    return best


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.v18_full_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    raw = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    width = finite_float(raw.get("width"), 1920.0)
    height = finite_float(raw.get("height"), 1080.0)
    frame_count = int(ann.get("frame_count", len(frames)))
    fps = finite_float(ann.get("fps"), 30.0)
    foundation_report = load_json(args.output_root / "mano_foundation_audit" / case / "v18_mano_foundation_state_report.json")
    rows = best_wilor_rows(wilor_path(case, args.measurement_manifest_root), frame_count)
    case_dir = args.output_root / "mano_foundation_audit" / case
    frame_dir = case_dir / "overlay_frames"
    if frame_dir.exists():
        for p in frame_dir.glob("*.jpg"):
            p.unlink()
    else:
        frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = case_dir / "v18_mano_foundation_overlay.mp4"
    big = font(22)
    small = font(15)
    counts: Counter[str] = Counter()
    available_frame_sides: set[tuple[int, str]] = set()
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        idx = int(frame.get("frame_idx", len(available_frame_sides)))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw_path).convert("RGB") if raw_path.exists() else Image.new("RGB", (int(width), int(height)), (12, 12, 12))
        draw = ImageDraw.Draw(image)
        for side in ("left", "right"):
            hand = rows.get((idx, side))
            color = HAND_COLORS[side]
            if hand is None:
                counts[f"missing::{side}"] += 1
                continue
            joints = arr(hand.get("joints3d_camera"), (21, 3))
            vertices = arr(hand.get("vertices_camera"), (778, 3))
            cam_t = arr(hand.get("cam_t"), (3,))
            if joints is None or vertices is None or cam_t is None or not isinstance(hand.get("mano_params"), dict):
                counts[f"incomplete::{side}"] += 1
                continue
            available_frame_sides.add((idx, side))
            counts[f"rendered_surface::{side}"] += 1
            pts = project(vertices[:: args.vertex_stride], cam_t, hand.get("focal_length"), width, height)
            for x, y in pts:
                if math.isfinite(float(x)) and math.isfinite(float(y)) and -10 <= x <= width + 10 and -10 <= y <= height + 10:
                    draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=color)
            j2 = project(joints, cam_t, hand.get("focal_length"), width, height)
            for a, b in HAND_EDGES:
                xa, ya = j2[a]
                xb, yb = j2[b]
                if all(math.isfinite(float(v)) for v in (xa, ya, xb, yb)):
                    draw.line((int(xa), int(ya), int(xb), int(yb)), fill=color, width=2)
            for x, y in j2:
                if math.isfinite(float(x)) and math.isfinite(float(y)):
                    draw.ellipse((int(x) - 3, int(y) - 3, int(x) + 3, int(y) + 3), outline=color, width=2)
        draw_label(draw, (16, 16), f"V18 MANO foundation overlay: recovered WiLoR surfaces; accepted physical hand state = false", big, (255, 255, 255))
        draw_label(draw, (16, 46), f"frame {idx:06d} available sides: {sum((idx,s) in available_frame_sides for s in ('left','right'))}/2 | blockers: {', '.join(foundation_report.get('blocking_reasons', [])[:3])}", small, (255, 210, 120))
        image.save(frame_dir / f"{idx:06d}.jpg", quality=92)
    encode_video(frame_dir, video_path, fps)
    report = {
        "method": "render_v18_mano_foundation_overlay",
        "case": case,
        "claim_scope": "visualizes_recovered_wilor_mano_surface_candidates_not_accepted_physical_hand_state",
        "frame_count": frame_count,
        "fps": fps,
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(counts.items())),
        "available_frame_side_rows": len(available_frame_sides),
        "foundational_mano_state_valid": foundation_report.get("foundational_mano_state_valid"),
        "blocking_reasons": foundation_report.get("blocking_reasons"),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_mano_foundation_overlay_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18-full-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--measurement-manifest-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--vertex-stride", type=int, default=10)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_mano_foundation_overlay",
        "claim_scope": "full_timeline_visualization_of_recovered_mano_surface_candidates_not_physical_closure",
        "cases": reports,
        "all_video_frame_counts_match": all(r.get("frame_counts", {}).get("video") == r.get("frame_count") for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "mano_foundation_audit" / "v18_mano_foundation_overlay_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
