#!/usr/bin/env python3
"""Render a V18-state-driven corrective artifact.

This is intentionally not another readiness/audit stage. It renders from the
current V18 annotation JSON and uses the factor-graph hand/object estimates as
state drivers, on raw frames / a blank world canvas, rather than using V16 videos
as the visual base. It also reports whether the graph estimates actually reduce
observable hand-center acceleration relative to the raw observed boxes.
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
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        path = Path(p)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


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
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy)))


def shift_box_to_center(box: tuple[float, float, float, float], center: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx, cy = center
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: tuple[int, int, int], bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=bg)
    draw.text((x, y), text, fill=fill, font=fnt)


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], width: int = 2, dash: int = 10) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        draw.line((x, y0, min(x + dash, x1), y0), fill=fill, width=width)
        draw.line((x, y1, min(x + dash, x1), y1), fill=fill, width=width)
    for y in range(y0, y1, dash * 2):
        draw.line((x0, y, x0, min(y + dash, y1)), fill=fill, width=width)
        draw.line((x1, y, x1, min(y + dash, y1)), fill=fill, width=width)


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


def project_mano_joints(mano: dict[str, Any], source_w: float, source_h: float, image_w: float, image_h: float) -> list[tuple[float, float]]:
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics") or [2304.0, 2304.0, source_w / 2.0, source_h / 2.0]
    if not (isinstance(joints, list) and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return []
    fx, fy, cx, cy = [finite_float(v) for v in intr]
    sx = image_w / source_w if source_w > 0 else 1.0
    sy = image_h / source_h if source_h > 0 else 1.0
    pts: list[tuple[float, float]] = []
    for raw in joints:
        if not (isinstance(raw, list) and len(raw) == 3):
            return []
        x = finite_float(raw[0]) + finite_float(cam_t[0])
        y = finite_float(raw[1]) + finite_float(cam_t[1])
        z = finite_float(raw[2]) + finite_float(cam_t[2])
        if z <= 1e-6:
            return []
        u = (fx * x / z + cx) * sx
        v = (fy * y / z + cy) * sy
        if not (math.isfinite(u) and math.isfinite(v)):
            return []
        pts.append((u, v))
    return pts


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}", "-i", str(frame_dir / "%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path),
    ]
    subprocess.run(cmd, check=True)


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def compose_side_by_side(left_path: Path, right_path: Path, output_path: Path, width_each: int = 960, height: int = 540) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filt = (
        f"[0:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[left];"
        f"[1:v]scale={width_each}:{height}:force_original_aspect_ratio=decrease,pad={width_each}:{height}:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(left_path), "-i", str(right_path), "-filter_complex", filt, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(output_path)], check=True)


def graph_hand_estimates(frame: dict[str, Any], width: float, height: float) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    vars_raw = frame.get("factor_graph_solution", {}).get("variables", {}).get("hand_state", [])
    if not isinstance(vars_raw, list):
        return out
    for row in vars_raw:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("hand::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 2:
            side = vid.split("::", 1)[1]
            out[side] = (finite_float(est[0]) * width, finite_float(est[1]) * height)
    return out


def graph_object_estimates(frame: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    vars_raw = frame.get("factor_graph_solution", {}).get("variables", {}).get("object_se3", [])
    if not isinstance(vars_raw, list):
        return out
    for row in vars_raw:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("object_se3::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 3:
            out[vid.split("::", 1)[1]] = (finite_float(est[0]), finite_float(est[1]), finite_float(est[2]))
    return out


def acceleration_stats(series: list[tuple[int, float, float]]) -> dict[str, float | int | None]:
    if len(series) < 3:
        return {"count": len(series), "mean_accel_px": None, "p95_accel_px": None}
    vals: list[float] = []
    pts = [(x, y) for _, x, y in sorted(series)]
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
        ax = c[0] - 2.0 * b[0] + a[0]
        ay = c[1] - 2.0 * b[1] + a[1]
        vals.append(math.hypot(ax, ay))
    vals.sort()
    if not vals:
        return {"count": len(series), "mean_accel_px": None, "p95_accel_px": None}
    p95 = vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)]
    return {"count": len(series), "mean_accel_px": sum(vals) / len(vals), "p95_accel_px": p95}


def render_case(case: str, ann: dict[str, Any], output_root: Path, max_frames: int | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    case_dir = output_root / case
    overlay_dir = case_dir / "corrective_overlay_frames"
    world_dir = case_dir / "corrective_world_frames"
    for d in [overlay_dir, world_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    frames = ann.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]
    raw_video = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    source_w = finite_float(raw_video.get("width"), 1920.0)
    source_h = finite_float(raw_video.get("height"), 1080.0)
    fps = finite_float(ann.get("fps"), 30.0)
    big = font(22)
    small = font(15)
    counts: Counter[str] = Counter()
    obs_series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    graph_series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    object_world_series: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)

    # Pass 1: collect graph world ranges for world render scaling.
    for raw_frame in frames:
        if not isinstance(raw_frame, dict):
            continue
        frame_idx = int(raw_frame.get("frame_idx", 0))
        for oid, xyz in graph_object_estimates(raw_frame).items():
            object_world_series[oid].append((frame_idx, xyz[0], xyz[1], xyz[2]))
    xs = [x for rows in object_world_series.values() for _, x, _, _ in rows]
    zs = [z for rows in object_world_series.values() for _, _, _, z in rows]
    if not xs:
        xs = [-1.0, 1.0]
    if not zs:
        zs = [0.0, 5.0]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0; z_max += 1.0

    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        raw_path = Path(str(frame.get("raw_frame_path")))
        image = Image.open(raw_path).convert("RGB") if raw_path.exists() else Image.new("RGB", (int(source_w), int(source_h)), (12, 12, 12))
        draw = ImageDraw.Draw(image)
        sx = image.size[0] / source_w if source_w > 0 else 1.0
        sy = image.size[1] / source_h if source_h > 0 else 1.0

        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            oid = str(obj.get("object_id"))
            if obj.get("renderable_mask") is True and isinstance(obj.get("mask_path"), str):
                image = mask_overlay(image, str(obj.get("mask_path")), (80, 180, 255), 0.16)
                draw = ImageDraw.Draw(image)
                counts["object_masks"] += 1
            box = bbox_tuple(obj.get("bbox_xyxy"))
            if box:
                ibox = scale_box(box, sx, sy)
                draw.rectangle(ibox, outline=(80, 180, 255), width=2)
                draw_label(draw, (ibox[0], max(48, ibox[1] - 19)), f"{obj.get('name')} {obj.get('physical_state_candidate')}", small, (80, 180, 255))
                counts["object_boxes"] += 1
            if oid in graph_object_estimates(frame):
                counts["object_graph_estimates"] += 1

        hand_est = graph_hand_estimates(frame, source_w, source_h)
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            box = bbox_tuple(hand.get("bbox_xyxy"))
            if box is None:
                continue
            center = bbox_center(hand.get("bbox_xyxy"))
            if center:
                obs_series[side].append((frame_idx, center[0] * sx, center[1] * sy))
            observed_box = scale_box(box, sx, sy)
            draw_dashed_rect(draw, observed_box, (180, 180, 180), width=2)
            est_center = hand_est.get(side)
            if est_center:
                shifted = shift_box_to_center(box, est_center)
                graph_box = scale_box(shifted, sx, sy)
                graph_series[side].append((frame_idx, est_center[0] * sx, est_center[1] * sy))
                color = (60, 255, 120) if side == "left" else (255, 210, 60)
                draw.rectangle(graph_box, outline=color, width=4)
                if center:
                    dx = (est_center[0] - center[0]) * sx
                    dy = (est_center[1] - center[1]) * sy
                    draw.line((int(center[0] * sx), int(center[1] * sy), int(est_center[0] * sx), int(est_center[1] * sy)), fill=(255, 80, 255), width=2)
                    pts = project_mano_joints(hand.get("mano_candidate", {}) if isinstance(hand.get("mano_candidate"), dict) else {}, source_w, source_h, image.size[0], image.size[1])
                    if len(pts) >= 21:
                        shifted_pts = [(int(round(px + dx)), int(round(py + dy))) for px, py in pts]
                        for a, b in HAND_EDGES:
                            draw.line((shifted_pts[a][0], shifted_pts[a][1], shifted_pts[b][0], shifted_pts[b][1]), fill=color, width=3)
                        for px, py in shifted_pts:
                            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
                        counts["graph_shifted_mano_skeletons"] += 1
                draw_label(draw, (graph_box[0], max(48, graph_box[1] - 21)), f"V18 graph-smoothed {side} hand", small, color)
                counts["graph_driven_hand_boxes"] += 1
            else:
                counts["hands_without_graph_estimate"] += 1

        draw.rectangle((0, 0, image.size[0], 46), fill=(0, 0, 0))
        draw.text((12, 11), f"V18 corrective raw-frame render {case} frame {frame_idx+1}/{len(frames)} — graph state drives hand boxes", fill=(255, 255, 255), font=big)
        image.save(overlay_dir / f"{frame_idx:06d}.jpg", quality=90)

        # Blank world view driven by V18 graph estimates, not V16 world video.
        canvas_w, canvas_h = 1280, 720
        world = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 24))
        wdraw = ImageDraw.Draw(world)
        wdraw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        wdraw.text((12, 12), f"V18 corrective graph world {case} frame {frame_idx+1}/{len(frames)} — no V16 video base", fill=(255, 255, 255), font=font(20))
        plot_left, plot_top, plot_right, plot_bottom = 80, 90, canvas_w - 330, canvas_h - 80
        wdraw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 90, 100), width=1)
        # Hands in image-normalized coordinates on the same canvas for temporal comparison.
        for side, est in hand_est.items():
            x = int(plot_left + max(0.0, min(1.0, est[0] / source_w)) * (plot_right - plot_left))
            y = int(plot_top + max(0.0, min(1.0, est[1] / source_h)) * (plot_bottom - plot_top))
            color = (60, 255, 120) if side == "left" else (255, 210, 60)
            wdraw.rectangle((x - 8, y - 8, x + 8, y + 8), fill=color)
            draw_label(wdraw, (x + 10, y - 10), f"{side} graph hand", small, color, (16, 18, 24))
        # Objects in x-z graph world coordinates.
        for oid, xyz in graph_object_estimates(frame).items():
            xw, _yw, zw = xyz
            px = int(plot_left + (xw - x_min) / (x_max - x_min) * (plot_right - plot_left))
            py = int(plot_bottom - (zw - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            wdraw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(80, 180, 255))
            draw_label(wdraw, (px + 9, py - 10), oid.replace("object:", "")[:32], small, (80, 180, 255), (16, 18, 24))
        sol = frame.get("factor_graph_solution", {}).get("solution", {}) if isinstance(frame.get("factor_graph_solution"), dict) else {}
        draw_label(wdraw, (12, 52), f"active contacts={sol.get('active_contact_hypotheses')} unresolved/contradicted={sol.get('unresolved_or_contradicted_contact_hypotheses')}", small, (255, 255, 255))
        world.save(world_dir / f"{frame_idx:06d}.jpg", quality=90)

    overlay_path = case_dir / "v18_corrective_overlay_graph_driven.mp4"
    world_path = case_dir / "v18_corrective_world_graph_driven.mp4"
    side_path = case_dir / "v18_corrective_side_by_side_graph_driven.mp4"
    encode_video(overlay_dir, overlay_path, fps)
    encode_video(world_dir, world_path, fps)
    compose_side_by_side(overlay_path, world_path, side_path)

    jitter: dict[str, Any] = {}
    for side in sorted(set(obs_series) | set(graph_series)):
        obs = acceleration_stats(obs_series.get(side, []))
        graph = acceleration_stats(graph_series.get(side, []))
        reduction = None
        if isinstance(obs.get("mean_accel_px"), float) and isinstance(graph.get("mean_accel_px"), float) and obs["mean_accel_px"] > 1e-9:
            reduction = 1.0 - graph["mean_accel_px"] / obs["mean_accel_px"]
        jitter[side] = {"observed_bbox_center": obs, "graph_estimate_center": graph, "mean_accel_reduction_fraction": reduction}

    report = {
        "method": "render_v18_corrective_state",
        "case": case,
        "claim_scope": "changed_render_attempt_from_v18_factor_graph_state_not_v16_base; does_not_claim_full_physical_v18_closure",
        "frame_count": len(frames),
        "fps": fps,
        "outputs": {
            "overlay_video": str(overlay_path),
            "world_video": str(world_path),
            "side_by_side_video": str(side_path),
        },
        "frame_counts": {
            "overlay": ffprobe_frame_count(overlay_path),
            "world": ffprobe_frame_count(world_path),
            "side_by_side": ffprobe_frame_count(side_path),
        },
        "draw_counts": dict(sorted(counts.items())),
        "jitter_probe": jitter,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_corrective_state_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = []
    for case in args.cases:
        ann_path = args.source_root / case / "annotations_v18_full.json"
        ann = load_json(ann_path)
        reports.append(render_case(case, ann, args.output_root, args.max_frames))
    out = {
        "method": "render_v18_corrective_state",
        "status": "corrective_attempt_not_full_v18_closure",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "cases": reports,
        "all_frame_counts_match": all(r["frame_counts"].get(k) == r["frame_count"] for r in reports for k in ["overlay", "world", "side_by_side"]),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_corrective_state_report.json", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-frames", type=int, default=None, help="Debug only: render a prefix of each case.")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
