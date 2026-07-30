#!/usr/bin/env python3
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


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_full_duration_world_status_render"
CLAIM = (
    "This render is a full-duration V18 world/status visualization of bounded state variables, including "
    "candidate-only occlusion depth triage. It is image-normalized abstract status geometry, not a metric 3D "
    "reconstruction or complete object pose."
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


def text_font(size: int) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Any, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


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
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[-1])
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


def point_from_center(center: Any, source_w: float, source_h: float, canvas_w: int, canvas_h: int) -> tuple[int, int] | None:
    if not (isinstance(center, list) and len(center) == 2):
        return None
    try:
        cx = float(center[0])
        cy = float(center[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (cx, cy)):
        return None
    left, right = 70, canvas_w - 310
    top, bottom = 96, canvas_h - 90
    x = int(round(left + max(0.0, min(1.0, cx / source_w)) * (right - left)))
    y = int(round(top + max(0.0, min(1.0, cy / source_h)) * (bottom - top)))
    return x, y


def object_color(state: str) -> tuple[int, int, int]:
    if state == "visible_surface_only_hidden_geometry_unresolved":
        return (70, 180, 255)
    if state == "visible_mask_only_surface_rejected_hidden_geometry_unresolved":
        return (230, 100, 230)
    if state == "active_object_visibility_unresolved_unfilled":
        return (150, 150, 150)
    return (80, 80, 90)


def hand_color(state: str) -> tuple[int, int, int]:
    if state == "observed_hand_depth_consistent":
        return (70, 230, 100)
    if state == "observed_hand_depth_unchecked":
        return (80, 210, 255)
    if state == "partial_hand_observation_uncertain":
        return (255, 205, 60)
    if state == "short_gap_possible_occlusion_unfilled":
        return (255, 150, 60)
    if state == "short_gap_no_visible_occluder_evidence_unfilled":
        return (180, 150, 80)
    return (150, 150, 150)


def contact_color(state: str) -> tuple[int, int, int] | None:
    if state == "rejected_contact_current_metric_depth":
        return (255, 70, 70)
    if state == "near_image_overlap_only":
        return (180, 180, 180)
    if state == "unobserved_pair_unresolved":
        return (120, 120, 120)
    return None


def depth_triage_label_and_color(state: str, pair_count: int) -> tuple[str, tuple[int, int, int], str]:
    if state == "row_scene_depth_supports_at_least_one_foreground_candidate_owner_unaccepted":
        return f"depth triage: fg candidate support ({pair_count}) not owner", (255, 170, 80), "support"
    if state == "row_scene_depth_contradicts_foreground_candidate_no_support":
        return f"depth triage: contradicts fg occlusion ({pair_count})", (255, 90, 90), "contradiction"
    if state == "row_metric_compatible_no_foreground_occluder_signal":
        return f"depth triage: no fg signal ({pair_count})", (180, 210, 255), "metric_compatible"
    if state in {"row_insufficient_object_surface_depth", "row_insufficient_or_untrusted_hand_depth_state"}:
        return f"depth triage: insufficient ({pair_count})", (190, 190, 190), "insufficient"
    return f"depth triage: {state or 'none'}", (170, 170, 170), "other"


def draw_base(canvas_w: int, canvas_h: int, frame_idx: int, frame_count: int, font: Any, small_font: Any) -> Image.Image:
    image = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 25))
    draw = ImageDraw.Draw(image)
    # Image-normalized pseudo-world grid. This is intentionally not metric 3D.
    left, right = 70, canvas_w - 310
    top, bottom = 96, canvas_h - 90
    for i in range(0, 11):
        x = int(round(left + i * (right - left) / 10.0))
        y = int(round(top + i * (bottom - top) / 10.0))
        draw.line((x, top, x, bottom), fill=(42, 46, 55), width=1)
        draw.line((left, y, right, y), fill=(42, 46, 55), width=1)
    draw.rectangle((left, top, right, bottom), outline=(90, 95, 110), width=2)
    camera_x = (left + right) // 2
    camera_y = bottom + 48
    draw.polygon([(camera_x, camera_y), (camera_x - 120, bottom), (camera_x + 120, bottom)], outline=(120, 160, 255), fill=None)
    draw.ellipse((camera_x - 16, camera_y - 16, camera_x + 16, camera_y + 16), fill=(120, 160, 255))
    draw_label(draw, (camera_x - 75, camera_y - 44), "ego camera / image-normalized layout", small_font, (180, 205, 255), (18, 20, 25))
    draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
    draw.text((14, 13), f"V18 WORLD STATUS (abstract; not metric 3D) frame {frame_idx + 1}/{frame_count}", font=font, fill=(255, 255, 255))
    draw_label(draw, (canvas_w - 290, 68), "Legend", font, (255, 255, 255), (18, 20, 25))
    legend = [
        ((70, 180, 255), "object visible surface-only"),
        ((230, 100, 230), "object mask-only / surface rejected"),
        ((70, 230, 100), "hand observed depth-consistent"),
        ((80, 210, 255), "hand observed depth-unchecked"),
        ((255, 150, 60), "unfilled possible occlusion gap"),
        ((255, 170, 80), "depth triage supports candidate, not owner"),
        ((255, 90, 90), "depth triage contradicts fg occlusion"),
        ((255, 70, 70), "contact rejected by metric depth"),
        ((180, 180, 180), "image overlap only"),
    ]
    y = 106
    for color, text in legend:
        draw.rectangle((canvas_w - 290, y, canvas_w - 272, y + 18), fill=color)
        draw.text((canvas_w - 264, y), text, font=small_font, fill=(230, 230, 230))
        y += 28
    return image


def draw_world_frame(
    solution_frame: dict[str, Any],
    source_w: float,
    source_h: float,
    canvas_w: int,
    canvas_h: int,
    frame_count: int,
    font: Any,
    small_font: Any,
) -> tuple[Image.Image, Counter[str]]:
    frame_idx = require_int(solution_frame.get("frame_idx"), "frame_idx")
    image = draw_base(canvas_w, canvas_h, frame_idx, frame_count, font, small_font)
    draw = ImageDraw.Draw(image)
    counts: Counter[str] = Counter()
    hand_points: dict[str, tuple[int, int]] = {}
    object_points: dict[str, tuple[int, int]] = {}
    # Objects first.
    for raw_obj in require_list(solution_frame.get("objects"), "objects"):
        obj = require_dict(raw_obj, "object")
        state = str(obj.get("solution_state"))
        if state == "inactive_or_out_of_frame":
            continue
        color = object_color(state)
        point = point_from_center(obj.get("bbox_center_xy"), source_w, source_h, canvas_w, canvas_h)
        object_id = str(obj.get("object_id"))
        name = str(obj.get("name"))
        area = obj.get("bbox_area_px")
        radius = 18
        if isinstance(area, (int, float)) and area > 0:
            radius = max(12, min(72, int(round(math.sqrt(float(area)) / 7.0))))
        if point is not None:
            object_points[object_id] = point
            x, y = point
            if state == "active_object_visibility_unresolved_unfilled":
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
                draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=2)
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
            draw_label(draw, (x + radius + 4, y - 12), f"{name}: {state}"[:90], small_font, color, (18, 20, 25))
            counts["objects_drawn"] += 1
        elif state == "active_object_visibility_unresolved_unfilled":
            counts["unresolved_objects_without_position"] += 1
    # Hands.
    side_offset = {"left": -10, "right": 10}
    unresolved_y = 390
    for raw_hand in require_list(solution_frame.get("hands"), "hands"):
        hand = require_dict(raw_hand, "hand")
        state = str(hand.get("solution_state"))
        side = str(hand.get("hand_side"))
        color = hand_color(state)
        point = point_from_center(hand.get("bbox_center_xy"), source_w, source_h, canvas_w, canvas_h)
        if point is not None:
            x, y = point
            x += side_offset.get(side, 0)
            hand_points[side] = (x, y)
            if "unfilled" in state:
                draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline=color, width=2)
            else:
                draw.rectangle((x - 16, y - 16, x + 16, y + 16), outline=color, width=4)
            draw_label(draw, (x + 20, y - 10), f"{side} hand: {state}"[:80], small_font, color, (18, 20, 25))
            counts["hands_drawn"] += 1
        elif state == "short_gap_possible_occlusion_unfilled":
            occ = require_dict(hand.get("occlusion_solution"), "occlusion_solution")
            candidates = [require_dict(raw, "candidate") for raw in require_list(occ.get("occluder_owner_candidates", []), "candidates")]
            candidate_points = [point_from_center(c.get("bbox_center_xy"), source_w, source_h, canvas_w, canvas_h) for c in candidates]
            valid_points = [p for p in candidate_points if p is not None]
            if valid_points:
                x = int(round(sum(p[0] for p in valid_points) / len(valid_points)))
                y = int(round(sum(p[1] for p in valid_points) / len(valid_points)))
                hand_points[side] = (x, y)
                depth_state = str(occ.get("depth_order_evidence_state") or "")
                pair_count = require_int(occ.get("depth_order_candidate_pair_count", 0), "depth_order_candidate_pair_count")
                depth_label, depth_color, depth_key = depth_triage_label_and_color(depth_state, pair_count)
                draw.ellipse((x - 20, y - 20, x + 20, y + 20), outline=color, width=3)
                draw.line((x - 18, y, x + 18, y), fill=color, width=2)
                draw_label(draw, (x + 22, y - 12), f"{side} hand unfilled possible occlusion", small_font, color, (18, 20, 25))
                if pair_count > 0:
                    draw_label(draw, (x + 22, y + 12), depth_label[:90], small_font, depth_color, (18, 20, 25))
                    counts[f"occlusion_depth_triage_{depth_key}_labels_drawn"] += 1
                counts["hand_occlusion_candidates_drawn"] += 1
            else:
                draw_label(draw, (canvas_w - 290, unresolved_y), f"{side} hand unresolved gap", small_font, color, (18, 20, 25))
                unresolved_y += 24
                counts["unresolved_hands_without_position"] += 1
        elif "unresolved" in state or "unfilled" in state:
            draw_label(draw, (canvas_w - 290, unresolved_y), f"{side} hand {state}"[:38], small_font, color, (18, 20, 25))
            unresolved_y += 24
            counts["unresolved_hands_without_position"] += 1
    # Contacts after points exist.
    contact_drawn_per_state: Counter[str] = Counter()
    for raw_contact in require_list(solution_frame.get("contacts"), "contacts"):
        contact = require_dict(raw_contact, "contact")
        state = str(contact.get("solution_state"))
        color = contact_color(state)
        if color is None:
            continue
        if contact_drawn_per_state[state] >= 8:
            continue
        hand_side = str(contact.get("hand_side"))
        object_id = str(contact.get("object_id"))
        hp = hand_points.get(hand_side)
        op = object_points.get(object_id)
        if hp is None or op is None:
            continue
        width = 3 if state == "rejected_contact_current_metric_depth" else 2
        draw.line((hp[0], hp[1], op[0], op[1]), fill=color, width=width)
        mid = ((hp[0] + op[0]) // 2, (hp[1] + op[1]) // 2)
        label = "depth rejects contact" if state == "rejected_contact_current_metric_depth" else "overlap only"
        draw_label(draw, (mid[0] + 4, mid[1] + 4), label, small_font, color, (18, 20, 25))
        contact_drawn_per_state[state] += 1
        counts[f"contacts_drawn_{state}"] += 1
    summary = require_dict(solution_frame.get("frame_solution_summary"), "frame_solution_summary")
    draw_label(
        draw,
        (70, canvas_h - 54),
        "no pose filled through occlusion | occlusion depth triage is candidate-only | contact-ready=0 | object hidden geometry unresolved",
        font,
        (255, 255, 255),
        (0, 0, 0),
    )
    draw_label(
        draw,
        (70, 56),
        f"frame states: hands={summary.get('hand_solution_state_counts')} objects={summary.get('object_solution_state_counts')} contacts={summary.get('contact_solution_state_counts')}",
        small_font,
        (235, 235, 235),
        (18, 20, 25),
    )
    return image, counts


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    solution_path = args.solution_root / case / "v18_bounded_state_solution.json"
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    solution = require_dict(load_json(solution_path), f"{case} solution")
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    raw_video = require_dict(annotation.get("raw_video"), "raw_video")
    source_w = require_float(raw_video.get("width"), "source width")
    source_h = require_float(raw_video.get("height"), "source height")
    fps = require_float(raw_video.get("fps"), "fps")
    frame_count = require_int(solution.get("frame_count"), "frame_count")
    frames = [require_dict(raw, "solution frame") for raw in require_list(solution.get("frames"), "solution frames")]
    if len(frames) != frame_count:
        raise RuntimeError(f"{case}: solution frames {len(frames)} != frame_count {frame_count}")
    output_dir = args.output_root / case
    frame_dir = output_dir / "world_status_frames"
    video_path = output_dir / "v18_world_status.mp4"
    if frame_dir.exists() and not args.keep_frames:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    font = text_font(20)
    small_font = text_font(14)
    start = time.perf_counter()
    draw_counts: Counter[str] = Counter()
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        image, counts = draw_world_frame(frame, source_w, source_h, args.width, args.height, frame_count, font, small_font)
        draw_counts.update(counts)
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    encode_video(frame_dir, video_path, fps)
    elapsed = time.perf_counter() - start
    rendered_count = len(list(frame_dir.glob("*.jpg")))
    video_frame_count = ffprobe_frame_count(video_path)
    qc = {
        "method": "render_v18_world_status",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"bounded_state_solution": str(solution_path), "annotation_state": str(annotation_path)},
        "output_video": str(video_path),
        "frame_dir": str(frame_dir) if args.keep_frames else None,
        "fps": fps,
        "state_frame_count": frame_count,
        "rendered_frame_count": rendered_count,
        "video_frame_count": video_frame_count,
        "frame_count_match": rendered_count == frame_count and video_frame_count == frame_count,
        "elapsed_s": elapsed,
        "canvas_size": [args.width, args.height],
        "draw_counts": dict(sorted(draw_counts.items())),
        "render_contract": {
            "full_duration": True,
            "same_frame_count_as_raw": rendered_count == frame_count and video_frame_count == frame_count,
            "abstract_world_status_not_metric_3d": True,
            "uses_image_normalized_bbox_centers": True,
            "occlusion_depth_triage_candidate_only": True,
            "pose_filled_through_occlusion": False,
            "contact_factor_ready_rows": 0,
            "not_complete_object_pose_deliverable": True,
        },
        **FALSE_READY,
    }
    write_json(output_dir / "v18_world_status_qc.json", qc)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    return qc


def render(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    qcs = [render_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "render_v18_world_status",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(qcs),
        "elapsed_s": elapsed,
        "all_frame_counts_match": all(bool(qc.get("frame_count_match")) for qc in qcs),
        "outputs": [
            {
                "case": qc["case"],
                "output_video": qc["output_video"],
                "qc_path": str(args.output_root / str(qc["case"]) / "v18_world_status_qc.json"),
                "state_frame_count": qc["state_frame_count"],
                "video_frame_count": qc["video_frame_count"],
                "frame_count_match": qc["frame_count_match"],
                **FALSE_READY,
            }
            for qc in qcs
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_world_status_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_renders"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(render(parse_args()), indent=2))


if __name__ == "__main__":
    main()
