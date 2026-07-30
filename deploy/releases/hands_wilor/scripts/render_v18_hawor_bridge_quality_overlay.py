#!/usr/bin/env python3
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
QUALITY_COLORS = {
    "projection_supported_visible_hawor_bridge_candidate": (50, 240, 110),
    "projection_supported_nonvisible_hawor_bridge_candidate": (130, 240, 180),
    "moderate_residual_uncertain_hawor_bridge_candidate": (255, 210, 60),
    "unsupported_residual_uncertain_hawor_bridge_candidate": (255, 150, 60),
    "large_residual_in_frame_conflict_candidate": (255, 80, 80),
    "large_residual_uncertain_bridge_candidate": (220, 90, 255),
    "residual_tail_hawor_out_of_frame_or_visibility_conflict": (255, 60, 160),
    "no_current_projection_reference_candidate": (170, 170, 170),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def font(size: int) -> ImageFont.ImageFont:
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
    points = np.asarray(points_camera, dtype=np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64)
    if points.shape != (21, 3) or intr.shape != (4,) or np.any(points[:, 2] <= 1e-6):
        return None
    fx, fy, cx, cy = intr
    uv = np.empty((21, 2), dtype=np.float64)
    uv[:, 0] = fx * points[:, 0] / points[:, 2] + cx
    uv[:, 1] = fy * points[:, 1] / points[:, 2] + cy
    return uv if np.isfinite(uv).all() else None


def current_hands_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if not isinstance(hand, dict):
            continue
        side = str(hand.get("hand_side") or hand.get("side") or "").lower()
        if side in {"left", "right"}:
            out[side] = hand
    return out


def current_reference_projection(hand: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else None
    if mano is None:
        return None, None
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics")
    if not (isinstance(joints, list) and len(joints) == 21 and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return None, str(mano.get("source")) if mano.get("source") is not None else None
    points = np.asarray(joints, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)[None, :]
    return project(points, np.asarray(intr, dtype=np.float64)), str(mano.get("source")) if mano.get("source") is not None else None


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bb = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=bg)
    draw.text((x, y), text, font=fnt, fill=fill)


def scale_points_to_image(points: np.ndarray, image_size: tuple[int, int], source_size: tuple[float, float] = (1920.0, 1080.0)) -> np.ndarray:
    sx = image_size[0] / source_size[0]
    sy = image_size[1] / source_size[1]
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def draw_skeleton(draw: ImageDraw.ImageDraw, pts: np.ndarray, color: tuple[int, int, int], width: int = 4) -> None:
    if pts.shape != (21, 2):
        return
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAND_EDGES:
        draw.line((ipts[a][0], ipts[a][1], ipts[b][0], ipts[b][1]), fill=color, width=width)
    for x, y in ipts:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, color: tuple[int, int, int], image_size: tuple[int, int]) -> None:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return
    vals = [finite_float(v) for v in bbox]
    if all(math.isfinite(v) for v in vals):
        sx = image_size[0] / 1920.0
        sy = image_size[1] / 1080.0
        x0, y0, x1, y1 = vals
        left, right = sorted((x0 * sx, x1 * sx))
        top, bottom = sorted((y0 * sy, y1 * sy))
        if right > left and bottom > top:
            draw.rectangle((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))), outline=color, width=2)


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}", "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path),
    ], check=True)


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / args.case / "annotations_v18_full.json")
    frames = ann.get("frames") if isinstance(ann.get("frames"), list) else []
    fps = finite_float(ann.get("fps"), 30.0)
    quality_path = args.output_root / "hawor_bridge_state" / args.case / "v18_hawor_bridge_quality_state.json"
    quality = load_json(quality_path)
    bridge_npz = Path(str(quality.get("bridge_candidate_npz")))
    z = np.load(bridge_npz)
    quality_by_key = {(int(row["frame_idx"]), str(row["side"])): row for row in quality.get("quality_rows", []) if isinstance(row, dict) and row.get("side") in {"left", "right"}}
    row_by_key = {(int(frame), "left" if int(side) == 0 else "right"): idx for idx, (frame, side) in enumerate(zip(np.asarray(z["frame_idx"], dtype=int), np.asarray(z["side"], dtype=int)))}
    out_dir = args.output_root / "hawor_bridge_state" / args.case / "quality_overlay_frames"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    big = font(22)
    small = font(15)
    tiny = font(13)
    counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    intr_hawor = np.asarray([2304.0, 2304.0, 960.0, 540.0], dtype=np.float64)
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", 0))
        raw = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw).convert("RGB") if raw.exists() else Image.new("RGB", (1920, 1080), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.size[0], 70), fill=(0, 0, 0))
        draw.text((12, 8), f"V18 HaWoR bridge quality overlay {args.case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.text((12, 38), "cyan=HaWoR bridge projection; orange=current visible hand candidate reference only; label=color=quality state; candidate-only, not V18 foundation", fill=(235, 235, 235), font=small)
        current = current_hands_by_side(frame)
        y = 78
        drew_any = False
        for side in ["left", "right"]:
            key = (frame_idx, side)
            qrow = quality_by_key.get(key)
            if qrow is None:
                continue
            row_idx = row_by_key.get(key)
            if row_idx is None:
                continue
            hproj = project(np.asarray(z["joints_hawor_camera_m"], dtype=np.float64)[row_idx], intr_hawor)
            ref, ref_source = current_reference_projection(current.get(side, {})) if side in current else (None, None)
            quality_state = str(qrow.get("quality_state"))
            color = QUALITY_COLORS.get(quality_state, (200, 200, 200))
            if side in current:
                draw_bbox(draw, current[side].get("bbox_xyxy"), (255, 180, 40), image.size)
            if ref is not None:
                draw_skeleton(draw, scale_points_to_image(ref, image.size), (255, 170, 30), 3)
            if hproj is not None:
                draw_skeleton(draw, scale_points_to_image(hproj, image.size), (50, 220, 255), 4)
                drew_any = True
            med = qrow.get("projection_residual_px_median")
            med_text = "no-ref" if med is None else f"{float(med):.1f}px"
            draw_label(draw, (12, y), f"{side}: {quality_state} residual={med_text} vis={qrow.get('current_visibility_state')} ref={ref_source}", tiny, color)
            y += 24
            counts[quality_state] += 1
            frame_counts[f"frame_has::{quality_state}"] += 1
        if not drew_any:
            draw_label(draw, (12, y), "No HaWoR bridge projection drawn for this frame", small, (180, 180, 180))
        image.save(out_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = args.output_root / "hawor_bridge_state" / args.case / "v18_hawor_bridge_quality_overlay.mp4"
    encode_video(out_dir, video_path, fps)
    report = {
        "method": "render_v18_hawor_bridge_quality_overlay",
        "case": args.case,
        "claim_scope": "full_timeline_QC_overlay_for_HaWoR_bridge_candidates_not_foundation_acceptance",
        "quality_state": str(quality_path),
        "bridge_npz": str(bridge_npz),
        "outputs": {"video": str(video_path)},
        "frame_count": len(frames),
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "draw_counts_by_quality_state": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "hawor_bridge_state" / args.case / "v18_hawor_bridge_quality_overlay_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--case", default="trash_1050")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
