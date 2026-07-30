#!/usr/bin/env python3
"""Render per-frame visible surface state for generic rigid candidates.

This is a corrective mechanism study and artifact: when fused canonical geometry is
visibly contaminated by unstable pose alignment, the frame-local RGBD surface is
a better honest visible-geometry estimate. This script renders those surfaces for
objects selected by generic rigid/local-rigid metadata. It does not claim hidden
geometry completion or canonical object pose.
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


def color_for_index(i: int) -> tuple[int, int, int]:
    palette = [(255, 90, 90), (80, 180, 255), (255, 210, 60), (80, 255, 140), (220, 120, 255), (255, 150, 70)]
    return palette[i % len(palette)]


def rigid_candidates(report_path: Path) -> dict[str, dict[str, Any]]:
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    raw = report.get("candidate_objects", {})
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}


def visible_surface_rows(report_path: Path, candidate_ids: set[str]) -> dict[tuple[int, str], dict[str, Any]]:
    report = load_json(report_path)
    rows = report.get("surface_archive_rows", [])
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id"))
        if oid in candidate_ids:
            out[(int(row.get("frame_idx", -1)), oid)] = row
    return out


def row_vertices(npz: Any, row: dict[str, Any]) -> np.ndarray:
    i = int(row.get("archive_row_index"))
    vo = npz["vertex_offsets"]
    verts = npz["vertices"]
    return np.asarray(verts[int(vo[i]): int(vo[i + 1])], dtype=np.float64)


def extent_stats(extents: list[np.ndarray]) -> dict[str, Any]:
    if not extents:
        return {"count": 0, "median_xyz_m": None, "p95_xyz_m": None}
    arr = np.stack(extents, axis=0)
    return {
        "count": int(arr.shape[0]),
        "median_xyz_m": np.round(np.median(arr, axis=0), 6).tolist(),
        "p95_xyz_m": np.round(np.percentile(arr, 95, axis=0), 6).tolist(),
    }


def fused_extent(row: dict[str, Any]) -> list[float] | None:
    mn = row.get("canonical_bbox_min_m")
    mx = row.get("canonical_bbox_max_m")
    if isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3:
        return [finite_float(mx[i]) - finite_float(mn[i]) for i in range(3)]
    return None


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", [])
    fps = finite_float(ann.get("fps"), 30.0)
    candidates = rigid_candidates(args.corrective_root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json")
    visible_report_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    visible_report = load_json(visible_report_path)
    npz_path = Path(str(visible_report.get("archive_npz")))
    npz = np.load(npz_path)
    row_index = visible_surface_rows(visible_report_path, set(candidates))

    all_samples: list[np.ndarray] = []
    object_extents: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in row_index.values():
        pts = row_vertices(npz, row)
        if pts.size:
            object_extents[str(row.get("object_id"))].append(np.ptp(pts, axis=0))
            step = max(1, len(pts) // 120)
            all_samples.append(pts[::step])
    if all_samples:
        sample = np.concatenate(all_samples, axis=0)
        x_min, z_min = np.percentile(sample[:, [0, 2]], 1, axis=0)
        x_max, z_max = np.percentile(sample[:, [0, 2]], 99, axis=0)
    else:
        x_min, x_max, z_min, z_max = -1.0, 1.0, 0.0, 5.0
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0; z_max += 1.0

    case_dir = args.output_root / case / "visible_surface_state"
    frame_dir = case_dir / "world_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    counts: Counter[str] = Counter()
    rng = np.random.default_rng(18)
    canvas_w, canvas_h = 1280, 720
    plot_left, plot_top, plot_right, plot_bottom = 70, 88, canvas_w - 330, canvas_h - 78

    for raw_frame in frames if isinstance(frames, list) else []:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        image = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        draw.text((12, 12), f"V18 frame-local visible surface state {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 90, 100), width=1)
        y = 54
        for i, (oid, cand) in enumerate(candidates.items()):
            row = row_index.get((frame_idx, oid))
            color = color_for_index(i)
            if row is None:
                draw_label(draw, (12, y), f"{oid.replace('object:', '')}: no visible surface this frame", small, (120, 120, 130), (0, 0, 0))
                y += 20
                continue
            pts = row_vertices(npz, row)
            if len(pts) > args.max_points_per_object:
                pts = pts[rng.choice(len(pts), size=args.max_points_per_object, replace=False)]
            extent = np.ptp(pts, axis=0) if len(pts) else np.zeros(3)
            for xw, _yw, zw in pts:
                px = int(plot_left + (float(xw) - x_min) / (x_max - x_min) * (plot_right - plot_left))
                py = int(plot_bottom - (float(zw) - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
                if plot_left <= px <= plot_right and plot_top <= py <= plot_bottom:
                    draw.point((px, py), fill=color)
            center = np.mean(pts, axis=0) if len(pts) else np.zeros(3)
            cx = int(plot_left + (center[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
            cy = int(plot_bottom - (center[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=(255, 255, 255), width=2)
            draw_label(draw, (cx + 8, cy - 10), f"{oid.replace('object:', '')[:28]} visible surface", small, color, (16, 18, 24))
            draw_label(draw, (12, y), f"{oid.replace('object:', '')}: verts={len(pts)} extent={extent[0]:.3f},{extent[1]:.3f},{extent[2]:.3f}m", small, color, (0, 0, 0))
            y += 20
            counts[f"visible_surface_rendered::{oid}"] += 1
        draw_label(draw, (12, canvas_h - 42), "Scope: frame-local RGBD visible surface, not canonical hidden geometry or accepted pose", small, (255, 255, 255), (0, 0, 0))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)

    video_path = case_dir / "v18_visible_surface_state_world.mp4"
    encode_video(frame_dir, video_path, fps)
    side_path = None
    corrective_overlay = args.corrective_root / case / "v18_corrective_overlay_graph_driven.mp4"
    if corrective_overlay.exists():
        side_path = case_dir / "v18_visible_surface_state_side_by_side.mp4"
        compose_side_by_side(corrective_overlay, video_path, side_path)

    object_reports: dict[str, Any] = {}
    for oid, cand in candidates.items():
        ext = extent_stats(object_extents.get(oid, []))
        fext = fused_extent(cand)
        ratio = None
        if fext is not None and ext.get("median_xyz_m") is not None:
            med = ext["median_xyz_m"]
            ratio = [round(fext[i] / med[i], 3) if med[i] and med[i] > 1e-9 else None for i in range(3)]
        object_reports[oid] = {
            "selection_reasons": cand.get("selection_reasons"),
            "model_physical_state_type": cand.get("model_physical_state_type"),
            "fast_motion_state": cand.get("fast_motion_state"),
            "visible_surface_rows": ext,
            "fused_canonical_bbox_extent_m": fext,
            "fused_extent_divided_by_median_visible_extent": ratio,
            "interpretation": "large_ratio_indicates_fused_canonical_geometry_spreads_beyond_frame_local_visible_surface" if ratio and any(v is not None and v > 3.0 for v in ratio) else "no_large_fused_vs_visible_extent_ratio_detected",
        }

    report = {
        "method": "render_v18_visible_surface_state",
        "case": case,
        "claim_scope": "frame_local_visible_RGBD_surface_render_for_generic_rigid_candidates; not_hidden_geometry_completion_or_accepted_object_pose",
        "frame_count": len(frames),
        "fps": fps,
        "candidate_count": len(candidates),
        "candidate_objects": object_reports,
        "outputs": {"world_video": str(video_path), "side_by_side_video": str(side_path) if side_path else None},
        "frame_counts": {"world": ffprobe_frame_count(video_path), "side_by_side": ffprobe_frame_count(side_path) if side_path else None},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_visible_surface_state_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_visible_surface_state",
        "status": "visible_surface_state_render_not_full_v18_closure",
        "cases": reports,
        "all_world_frame_counts_match": all(r["frame_counts"].get("world") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_visible_surface_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--max-points-per-object", type=int, default=1600)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
