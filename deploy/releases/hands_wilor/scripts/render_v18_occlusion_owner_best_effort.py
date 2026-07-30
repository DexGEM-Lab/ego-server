#!/usr/bin/env python3
"""Render no-gating occlusion-owner best estimates with blockers preserved.

The strict V18 owner gate accepted zero owners. This script does not relax that
claim. It renders temporal-graph-selected owner rows as tentative estimates with
energy margins and blockers, so the artifact carries best-current owner evidence
instead of hiding all rows behind acceptance failure.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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
    return x if math.isfinite(x) else default


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
    vals = [finite_float(x, float("nan")) for x in value]
    if not all(math.isfinite(x) for x in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[int, int, int, int]:
    return int(round(box[0] * sx)), int(round(box[1] * sy)), int(round(box[2] * sx)), int(round(box[3] * sy))


def scale_point(pt: tuple[float, float], sx: float, sy: float) -> tuple[int, int]:
    return int(round(pt[0] * sx)), int(round(pt[1] * sy))


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


def compose_side_by_side(left_path: Path, right_path: Path, output_path: Path, width_each: int = 960, height: int = 540) -> None:
    filt = (
        f"[0:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[left];"
        f"[1:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(left_path), "-i", str(right_path), "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path)], check=True)


def selected_owner_index(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    report = load_json(path)
    index: dict[tuple[int, str], dict[str, Any]] = {}
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        for row in hand_graph.get("assignments", []) if isinstance(hand_graph, dict) else []:
            if not isinstance(row, dict):
                continue
            owner = row.get("chosen_owner_object_id")
            if isinstance(owner, str) and owner.startswith("object:"):
                index[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return index, report


def object_by_id(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id")): obj for obj in frame.get("objects", []) if isinstance(obj, dict)}


def hand_by_side(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(hand.get("hand_side") or hand.get("side")): hand for hand in frame.get("hands", []) if isinstance(hand, dict)}


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    raw_video = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    source_w = finite_float(raw_video.get("width"), 1920.0)
    source_h = finite_float(raw_video.get("height"), 1080.0)
    fps = finite_float(ann.get("fps"), 30.0)
    owner_index, graph_report = selected_owner_index(args.owner_graph_root / case / "v18_occlusion_owner_graph_report.json")
    case_dir = args.output_root / case / "occlusion_owner_best_effort"
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
        image = Image.open(raw_path).convert("RGB") if raw_path.exists() else Image.new("RGB", (int(source_w), int(source_h)), (12, 12, 12))
        sx = image.size[0] / source_w if source_w > 0 else 1.0
        sy = image.size[1] / source_h if source_h > 0 else 1.0
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.size[0], 46), fill=(0, 0, 0))
        draw.text((12, 11), f"V18 tentative occlusion owner best-effort {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        hands = hand_by_side(frame)
        objects = object_by_id(frame)
        frame_rows = [row for (idx, _side), row in owner_index.items() if idx == frame_idx]
        if not owner_index:
            draw_label(draw, (12, 54), "no temporal owner rows selected; strict acceptance remains zero", small, (255, 140, 100), (0, 0, 0))
        elif not frame_rows:
            counts["frames_without_selected_tentative_owner"] += 1
        for row in frame_rows:
            side = str(row.get("hand_side"))
            oid = str(row.get("chosen_owner_object_id"))
            hand = hands.get(side, {})
            obj = objects.get(oid, {})
            hbox = bbox_tuple(hand.get("bbox_xyxy"))
            obox = bbox_tuple(obj.get("bbox_xyxy"))
            source_row = row.get("source_row", {}) if isinstance(row.get("source_row"), dict) else {}
            object_name = str(source_row.get("object_name") or obj.get("name") or oid.replace("object:", ""))
            margin = finite_float(row.get("unary_energy_margin"), float("nan"))
            blockers = row.get("acceptance_blockers") if isinstance(row.get("acceptance_blockers"), list) else []
            color = (255, 90, 210) if side == "left" else (255, 170, 40)
            if hbox is not None:
                ih = scale_box(hbox, sx, sy)
                draw.rectangle(ih, outline=color, width=4)
                hp = scale_point(center(hbox), sx, sy)
            else:
                hp = (80, 80)
            if obox is not None:
                io = scale_box(obox, sx, sy)
                draw.rectangle(io, outline=(80, 210, 255), width=4)
                op = scale_point(center(obox), sx, sy)
            else:
                op = (hp[0] + 160, hp[1])
            draw.line((hp[0], hp[1], op[0], op[1]), fill=color, width=5)
            draw.ellipse((hp[0] - 6, hp[1] - 6, hp[0] + 6, hp[1] + 6), fill=color)
            draw.ellipse((op[0] - 6, op[1] - 6, op[0] + 6, op[1] + 6), fill=(80, 210, 255))
            label = f"tentative {side} owner: {object_name} margin={margin:.2f} NOT accepted"
            draw_label(draw, (min(hp[0], op[0]) + 8, max(54, min(hp[1], op[1]) - 22)), label[:110], small, color, (0, 0, 0))
            if blockers:
                draw_label(draw, (12, image.size[1] - 28 - 18 * min(3, len(blockers))), "blockers: " + ", ".join(str(b) for b in blockers[:3])[:140], small, (255, 210, 120), (0, 0, 0))
            counts[f"tentative_owner::{oid}"] += 1
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_occlusion_owner_best_effort.mp4"
    encode_video(frame_dir, video_path, fps)
    side_path = None
    corrective_overlay = args.corrective_root / case / "v18_corrective_overlay_graph_driven.mp4"
    if corrective_overlay.exists():
        side_path = case_dir / "v18_occlusion_owner_best_effort_side_by_side.mp4"
        compose_side_by_side(corrective_overlay, video_path, side_path)
    selected_rows = list(owner_index.values())
    report = {
        "method": "render_v18_occlusion_owner_best_effort",
        "case": case,
        "claim_scope": "temporal_graph_selected_owner_best_estimate_with_acceptance_blockers; strict_accepted_ownership_still_zero",
        "frame_count": len(frames),
        "fps": fps,
        "selected_tentative_owner_rows": len(selected_rows),
        "strict_accepted_owner_rows": sum(1 for row in selected_rows if row.get("accepted_occlusion_owner") is True),
        "owner_object_counts": dict(sorted(Counter(str(row.get("chosen_owner_object_id")) for row in selected_rows).items())),
        "acceptance_blocker_counts": graph_report.get("acceptance_blocker_counts"),
        "outputs": {"video": str(video_path), "side_by_side_video": str(side_path) if side_path else None},
        "frame_counts": {"video": ffprobe_frame_count(video_path), "side_by_side": ffprobe_frame_count(side_path) if side_path else None},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_occlusion_owner_best_effort_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_occlusion_owner_best_effort",
        "status": "occlusion_owner_best_effort_render_not_strict_acceptance",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_occlusion_owner_best_effort_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
