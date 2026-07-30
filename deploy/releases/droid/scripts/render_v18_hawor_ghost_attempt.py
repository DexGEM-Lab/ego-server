#!/usr/bin/env python3
"""Render HaWoR as an uncertain temporal hand prior.

This is a corrective V18 attempt, not an acceptance gate. It uses existing HaWoR
source annotations where present and renders them as ghosted uncertain hand
skeletons. For cases without HaWoR measurements, it writes the concrete execution
failure/provisioning evidence instead of pretending cached data should exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

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


def finite_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if abs(x) < 1e20 else default


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


def compose_side_by_side(left_path: Path, right_path: Path, output_path: Path, width_each: int = 960, height: int = 540) -> None:
    filt = (
        f"[0:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[left];"
        f"[1:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(left_path), "-i", str(right_path), "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path)], check=True)


def load_hawor_measurement_index(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], list[Path]]:
    if not path.exists():
        return {}, []
    rows = load_json(path)
    index: dict[tuple[int, str], dict[str, Any]] = {}
    sources: set[Path] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        entity = str(row.get("entity_id", ""))
        if not entity.startswith("hand:"):
            continue
        key = (int(row.get("frame_idx", -1)), entity.split(":", 1)[1])
        index[key] = row
        src = row.get("source_annotation")
        if isinstance(src, str) and Path(src).exists():
            sources.add(Path(src))
    return index, sorted(sources)


def load_hawor_source_hands(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        frames = payload.get("frames", [])
        for raw_frame in frames if isinstance(frames, list) else []:
            if not isinstance(raw_frame, dict):
                continue
            frame_idx = int(raw_frame.get("frame_idx", len(out)))
            for hand in raw_frame.get("hands", []):
                if isinstance(hand, dict) and hand.get("backend") == "HaWoR":
                    side = str(hand.get("hand_side") or hand.get("side"))
                    if side in {"left", "right"}:
                        out[(frame_idx, side)] = hand
    return out


def draw_hawor_hand(image: Image.Image, hand: dict[str, Any], role: str, side: str) -> bool:
    pts_raw = hand.get("joints2d")
    if not (isinstance(pts_raw, list) and len(pts_raw) >= 21):
        return False
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    color = (190, 80, 255, 150) if "infill" in role else (80, 210, 255, 130)
    pts: list[tuple[int, int]] = []
    for p in pts_raw[:21]:
        if not (isinstance(p, list) and len(p) >= 2):
            return False
        pts.append((int(round(finite_float(p[0]))), int(round(finite_float(p[1])))))
    for a, b in HAND_EDGES:
        odraw.line((pts[a][0], pts[a][1], pts[b][0], pts[b][1]), fill=color, width=5)
    for x, y in pts:
        odraw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    wrist = pts[0]
    draw_label(draw, (wrist[0] + 8, max(48, wrist[1] - 16)), f"HaWoR ghost {side}: {role}", font(14), color[:3], (0, 0, 0))
    return True


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", [])
    fps = finite_float(ann.get("fps"), 30.0)
    meas_path = args.measurement_root / case / "measurements_v17" / "hawor_measurements.json"
    measurement_index, source_paths = load_hawor_measurement_index(meas_path)
    source_hands = load_hawor_source_hands(source_paths)
    case_dir = args.output_root / case / "hawor_ghost_attempt"
    frame_dir = case_dir / "frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    counts: Counter[str] = Counter()
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw_path).convert("RGBA") if raw_path.exists() else Image.new("RGBA", (1920, 1080), (20, 20, 20, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.size[0], 46), fill=(0, 0, 0, 255))
        draw.text((12, 11), f"V18 HaWoR ghost-prior attempt {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        drew = 0
        for side in ["left", "right"]:
            key = (frame_idx, side)
            row = measurement_index.get(key)
            hand = source_hands.get(key)
            if row is not None and hand is not None:
                role = str(row.get("evidence_role") or ("observed_visible_hawor_measurement" if row.get("measurement_available") else "hawor_motion_infill_candidate"))
                if draw_hawor_hand(image, hand, role, side):
                    counts[f"hawor::{role}"] += 1
                    drew += 1
        if not measurement_index:
            msg = "HaWoR execution/provisioning failed locally: missing HaWoR repo/root and MANO_LEFT; see hawor_execution_attempt logs"
            draw_label(draw, (12, 54), msg[:150], small, (255, 120, 120), (0, 0, 0))
        elif drew == 0:
            counts["frames_without_hawor_prior"] += 1
        image.convert("RGB").save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_hawor_ghost_prior_attempt.mp4"
    encode_video(frame_dir, video_path, fps)
    side_path = None
    corrective_overlay = args.corrective_root / case / "v18_corrective_overlay_graph_driven.mp4"
    if corrective_overlay.exists():
        side_path = case_dir / "v18_hawor_ghost_side_by_side_attempt.mp4"
        compose_side_by_side(corrective_overlay, video_path, side_path)
    report = {
        "method": "render_v18_hawor_ghost_attempt",
        "case": case,
        "claim_scope": "HaWoR_temporal_prior_render_attempt_with_uncertainty; not_occlusion_pose_solution_or_full_HaWoR_execution_success",
        "measurement_file": str(meas_path),
        "measurement_rows": len(measurement_index),
        "source_annotation_paths": [str(p) for p in source_paths],
        "source_hawor_hand_rows": len(source_hands),
        "outputs": {"video": str(video_path), "side_by_side_video": str(side_path) if side_path else None},
        "frame_count": len(frames),
        "frame_counts": {"video": ffprobe_frame_count(video_path), "side_by_side": ffprobe_frame_count(side_path) if side_path else None},
        "draw_counts": dict(sorted(counts.items())),
        "execution_failure_logs": {
            "task5_export_attempt": str(args.output_root / "hawor_execution_attempt" / "task5_tomato_960" / "export_hawor_world_attempt.log"),
            "setup_preflight_attempt": str(args.output_root / "hawor_execution_attempt" / "setup_preflight" / "remote_setup_hawor_local_attempt.log"),
        },
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_hawor_ghost_attempt_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    out = {
        "method": "render_v18_hawor_ghost_attempt",
        "status": "hawor_prior_render_and_execution_failure_evidence_not_full_v18_closure",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_hawor_ghost_attempt_summary.json", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--measurement-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
