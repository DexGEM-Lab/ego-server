#!/usr/bin/env python3
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


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_full_duration_status_overlay"
CLAIM = (
    "This render is a full-duration V18 status overlay from raw frames and renderable annotation state. "
    "It visualizes observed hands, visible object masks/surfaces, unresolved state, and rejected contact; "
    "it is not a complete object-pose deliverable."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{label} must be a number")
    return float(value)


def color_from_bgr(value: Any, fallback_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    if not (isinstance(value, list) and len(value) == 3):
        return fallback_rgb
    b = int(value[0])
    g = int(value[1])
    r = int(value[2])
    return (r, g, b)


def bbox_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def text_font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int]) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    bg = (0, 0, 0)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


def draw_banner(draw: ImageDraw.ImageDraw, width: int, frame: dict[str, Any], frame_idx: int, frame_count: int, font: Any) -> None:
    summary = require_dict(frame.get("frame_summary"), "frame_summary")
    text = (
        f"V18 STATUS (not pose-complete)  frame {frame_idx + 1}/{frame_count}  "
        f"hands unresolved={summary.get('unresolved_hands')}  objects unresolved={summary.get('unresolved_objects')}  "
        f"depth-rejected contacts={summary.get('image_contact_rejected_by_metric_depth')}  "
        f"overlap-only={summary.get('image_overlap_only')}  contact-ready=0"
    )
    draw.rectangle((0, 0, width, 38), fill=(0, 0, 0))
    draw.text((12, 9), text, font=font, fill=(255, 255, 255))


def mask_overlay(base: Image.Image, mask_path: str, rgb: tuple[int, int, int], alpha_float: float) -> Image.Image:
    path = Path(mask_path)
    if not path.exists():
        return base
    mask = Image.open(path).convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    alpha_value = max(0, min(255, int(alpha_float * 255)))
    alpha = mask.point([alpha_value if p > 0 else 0 for p in range(256)])
    overlay = Image.new("RGB", base.size, rgb)
    return Image.composite(overlay, base, alpha)


def draw_objects(image: Image.Image, frame: dict[str, Any], font: Any, small_font: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    # Draw masks first, then boxes/labels so text remains readable.
    for raw_obj in require_list(frame.get("objects"), "frame objects"):
        obj = require_dict(raw_obj, "object row")
        if obj.get("visibility_state") != "visible":
            continue
        style = require_dict(obj.get("render_style"), "object render_style")
        rgb = color_from_bgr(style.get("color_bgr"), (220, 120, 255))
        mask_path = obj.get("mask_path")
        if isinstance(mask_path, str) and obj.get("renderable_mask") is True:
            image.paste(mask_overlay(image, mask_path, rgb, float(style.get("alpha", 0.18))))
            counts["object_masks_drawn"] += 1
    draw = ImageDraw.Draw(image)
    label_y = 48
    for raw_obj in require_list(frame.get("objects"), "frame objects"):
        obj = require_dict(raw_obj, "object row")
        visibility = str(obj.get("visibility_state"))
        if visibility == "out_of_frame":
            continue
        style = require_dict(obj.get("render_style"), "object render_style")
        rgb = color_from_bgr(style.get("color_bgr"), (220, 120, 255))
        bbox = bbox_tuple(obj.get("bbox_xyxy"))
        name = str(obj.get("name"))
        motion = str(obj.get("fast_motion_state"))
        geom = str(obj.get("geometry_scope"))
        if bbox is not None:
            draw.rectangle(bbox, outline=rgb, width=3)
            label = f"{name}: {visibility} | {geom} | {motion}"
            draw_label(draw, (bbox[0], max(42, bbox[1] - 22)), label[:120], small_font, rgb)
            counts["object_boxes_drawn"] += 1
        elif visibility == "unresolved":
            draw_label(draw, (12, label_y), f"UNRESOLVED OBJ {name}: no visible geometry", small_font, rgb)
            label_y += 24
            counts["object_unresolved_labels_drawn"] += 1
    draw_label(
        draw,
        (12, image.size[1] - 32),
        "Object geometry policy: visible surface/mask only; hidden geometry unresolved; object_pose_requirement_met=false",
        font,
        (255, 255, 255),
    )
    return counts


def draw_hands(image: Image.Image, frame: dict[str, Any], font: Any, small_font: Any) -> Counter[str]:
    draw = ImageDraw.Draw(image)
    counts: Counter[str] = Counter()
    unresolved_labels: list[str] = []
    for raw_hand in require_list(frame.get("hands"), "frame hands"):
        hand = require_dict(raw_hand, "hand row")
        style = require_dict(hand.get("render_style"), "hand render_style")
        rgb = color_from_bgr(style.get("color_bgr"), (80, 220, 80))
        bbox = bbox_tuple(hand.get("bbox_xyxy"))
        side = str(hand.get("hand_side"))
        visibility = str(hand.get("visibility_state"))
        depth_state = str(hand.get("metric_depth_state"))
        if bbox is not None:
            draw.rectangle(bbox, outline=rgb, width=4)
            label = f"{side} hand: {visibility} | {depth_state}"
            draw_label(draw, (bbox[0], max(42, bbox[1] - 22)), label[:100], small_font, rgb)
            counts["hand_boxes_drawn"] += 1
        elif visibility == "unresolved":
            unresolved_labels.append(f"{side} hand unresolved")
            counts["hand_unresolved_labels_drawn"] += 1
    y = 78
    for label in unresolved_labels[:3]:
        draw_label(draw, (12, y), label, small_font, (180, 180, 180))
        y += 24
    draw_label(draw, (12, image.size[1] - 62), "Contact policy: image overlap is not physical contact; current contact-ready rows=0", font, (255, 255, 255))
    return counts


def ffprobe_frame_count(path: Path) -> int | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip().splitlines()
    if not text:
        return None
    try:
        return int(text[-1])
    except ValueError:
        return None


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frame_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.annotation_root / case / "v18_annotation_state.json"
    state = require_dict(load_json(state_path), f"{case} annotation state")
    frame_count = require_int(state.get("frame_count"), "frame_count")
    raw_video = require_dict(state.get("raw_video"), "raw_video")
    fps = require_float(raw_video.get("fps"), "fps")
    output_dir = args.output_root / case
    frame_dir = output_dir / "status_overlay_frames"
    video_path = output_dir / "v18_status_overlay.mp4"
    if frame_dir.exists() and not args.keep_frames:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    font = text_font(18)
    small_font = text_font(15)
    start = time.perf_counter()
    draw_counts: Counter[str] = Counter()
    frames = require_list(state.get("frames"), "frames")
    if len(frames) != frame_count:
        raise RuntimeError(f"{case}: state frame list length {len(frames)} != frame_count {frame_count}")
    for raw_frame in frames:
        frame = require_dict(raw_frame, "frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        raw_frame_path = frame.get("raw_frame_path")
        if not isinstance(raw_frame_path, str) or not Path(raw_frame_path).exists():
            raise RuntimeError(f"{case}: raw frame path missing for frame {frame_idx}")
        image = Image.open(raw_frame_path).convert("RGB")
        draw_banner(ImageDraw.Draw(image), image.size[0], frame, frame_idx, frame_count, font)
        draw_counts.update(draw_objects(image, frame, font, small_font))
        draw_counts.update(draw_hands(image, frame, font, small_font))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=88)
    encode_video(frame_dir, video_path, fps)
    elapsed = time.perf_counter() - start
    rendered_count = len(list(frame_dir.glob("*.jpg")))
    video_frame_count = ffprobe_frame_count(video_path)
    qc = {
        "method": "render_v18_status_overlay",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"annotation_state": str(state_path)},
        "output_video": str(video_path),
        "frame_dir": str(frame_dir) if args.keep_frames else None,
        "fps": fps,
        "raw_frame_count": state.get("raw_frame_count"),
        "state_frame_count": frame_count,
        "rendered_frame_count": rendered_count,
        "video_frame_count": video_frame_count,
        "frame_count_match": rendered_count == frame_count and video_frame_count == frame_count,
        "elapsed_s": elapsed,
        "draw_counts": dict(sorted(draw_counts.items())),
        "render_contract": {
            "full_duration": True,
            "same_frame_count_as_raw": rendered_count == frame_count and video_frame_count == frame_count,
            "uses_raw_frames": True,
            "uses_observed_hand_boxes_only": True,
            "uses_verified_object_masks_only": True,
            "pose_filled_through_occlusion": False,
            "contact_factor_ready_rows": 0,
            "not_complete_object_pose_deliverable": True,
        },
        **FALSE_READY,
    }
    write_json(output_dir / "v18_status_overlay_qc.json", qc)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    return qc


def render(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    qcs = [render_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "render_v18_status_overlay",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(qcs),
        "elapsed_s": elapsed,
        "all_frame_counts_match": all(bool(qc.get("frame_count_match")) for qc in qcs),
        "outputs": [
            {
                "case": qc["case"],
                "output_video": qc["output_video"],
                "qc_path": str(args.output_root / str(qc["case"]) / "v18_status_overlay_qc.json"),
                "state_frame_count": qc["state_frame_count"],
                "video_frame_count": qc["video_frame_count"],
                "frame_count_match": qc["frame_count_match"],
                **FALSE_READY,
            }
            for qc in qcs
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_status_overlay_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_renders"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2))


if __name__ == "__main__":
    main()
