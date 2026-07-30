#!/usr/bin/env python3
"""Residual-test generic rigid SE(3) fused geometry against visible surfaces.

A stable rigid prior can make a trajectory smooth while still being wrong. This
script tests the transformed fused canonical point cloud against same-frame
visible RGBD surface vertices. It renders and records residual-supported vs
residual-rejected states without accepting object pose or hidden geometry.
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
from scipy.spatial import cKDTree


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


def read_binary_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        vertex_count: int | None = None
        fmt: str | None = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{path}: missing vertex count")
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: expected binary_little_endian, got {fmt}")
        arr = np.fromfile(f, dtype="<f8", count=vertex_count * 3)
    if arr.size != vertex_count * 3:
        raise ValueError(f"{path}: expected {vertex_count * 3} doubles, got {arr.size}")
    return arr.reshape(vertex_count, 3).astype(np.float64)


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = rotvec / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * (kx @ kx)


def transform_points(points: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    r = rodrigues(pose6[3:6])
    return points @ r.T + pose6[:3][None, :]


def graph_object_poses(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("object_se3", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("object_se3::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 6:
            out[vid.split("::", 1)[1]] = np.array([finite_float(v) for v in est[:6]], dtype=np.float64)
    return out


def stable_pose_index(frames: list[Any], candidate_ids: set[str], radius: int) -> dict[tuple[int, str], np.ndarray]:
    raw: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        for oid in candidate_ids:
            pose = poses.get(oid)
            if pose is not None:
                raw[oid].append((frame_idx, pose.copy()))
    stable: dict[tuple[int, str], np.ndarray] = {}
    for oid, rows in raw.items():
        ordered = sorted(rows, key=lambda x: x[0])
        if not ordered:
            continue
        translations = [pose[:3] for _idx, pose in ordered]
        median_rot = np.median(np.stack([pose[3:6] for _idx, pose in ordered], axis=0), axis=0)
        for i, (frame_idx, pose6) in enumerate(ordered):
            lo = max(0, i - radius)
            hi = min(len(ordered), i + radius + 1)
            stable_pose = pose6.copy()
            stable_pose[:3] = np.mean(np.stack(translations[lo:hi], axis=0), axis=0)
            stable_pose[3:6] = median_rot
            stable[(frame_idx, oid)] = stable_pose
    return stable


def rigid_candidates(report_path: Path) -> dict[str, dict[str, Any]]:
    report = load_json(report_path)
    raw = report.get("candidate_objects", {})
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}


def visible_row_index(report_path: Path, candidate_ids: set[str]) -> tuple[dict[tuple[int, str], dict[str, Any]], Any]:
    report = load_json(report_path)
    npz = np.load(str(report.get("archive_npz")))
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("surface_archive_rows", []) if isinstance(report.get("surface_archive_rows"), list) else []:
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id"))
        if oid in candidate_ids:
            out[(int(row.get("frame_idx", -1)), oid)] = row
    return out, npz


def row_vertices(npz: Any, row: dict[str, Any]) -> np.ndarray:
    i = int(row.get("archive_row_index"))
    vo = npz["vertex_offsets"]
    verts = npz["vertices"]
    return np.asarray(verts[int(vo[i]): int(vo[i + 1])], dtype=np.float64)


def pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def residual_stats(fused_world: np.ndarray, visible_world: np.ndarray) -> dict[str, Any]:
    if len(fused_world) == 0 or len(visible_world) == 0:
        return {"status": "missing_points"}
    tree_fused = cKDTree(fused_world)
    tree_visible = cKDTree(visible_world)
    visible_to_fused = tree_fused.query(visible_world, k=1)[0]
    fused_to_visible = tree_visible.query(fused_world, k=1)[0]
    return {
        "visible_to_fused_median_m": float(np.median(visible_to_fused)),
        "visible_to_fused_p95_m": pct(visible_to_fused, 95),
        "fused_to_visible_median_m": float(np.median(fused_to_visible)),
        "fused_to_visible_p95_m": pct(fused_to_visible, 95),
        "visible_point_count": int(len(visible_world)),
        "fused_point_count": int(len(fused_world)),
    }


def classify(stats: dict[str, Any], median_threshold: float, p95_threshold: float) -> str:
    v_med = stats.get("visible_to_fused_median_m")
    v_p95 = stats.get("visible_to_fused_p95_m")
    f_p95 = stats.get("fused_to_visible_p95_m")
    vals = [v_med, v_p95, f_p95]
    if not all(isinstance(v, float) and math.isfinite(v) for v in vals):
        return "residual_missing"
    visible_explained = v_med <= median_threshold and v_p95 <= p95_threshold
    fused_not_overspread = f_p95 <= p95_threshold
    if visible_explained and fused_not_overspread:
        return "bidirectional_residual_supported_uncertain"
    if visible_explained and not fused_not_overspread:
        return "visible_supported_but_fused_overspread"
    return "visible_surface_not_explained_by_fused_pose"


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


def color_for_status(status: str) -> tuple[int, int, int]:
    if status == "bidirectional_residual_supported_uncertain":
        return (80, 255, 140)
    if status == "visible_supported_but_fused_overspread":
        return (255, 200, 80)
    if status == "visible_surface_not_explained_by_fused_pose":
        return (255, 80, 80)
    return (180, 180, 180)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    fps = finite_float(ann.get("fps"), 30.0)
    candidates = rigid_candidates(args.corrective_root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json")
    stable = stable_pose_index(frames, set(candidates), args.translation_smoothing_radius)
    visible_rows, npz = visible_row_index(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json", set(candidates))
    rng = np.random.default_rng(18)
    fused_samples: dict[str, np.ndarray] = {}
    for oid, row in candidates.items():
        pts = read_binary_xyz_ply(Path(str(row.get("fused_point_cloud_path"))))
        if len(pts) > args.max_fused_points:
            pts = pts[rng.choice(len(pts), size=args.max_fused_points, replace=False)]
        fused_samples[oid] = pts

    residual_by_frame: dict[tuple[int, str], dict[str, Any]] = {}
    all_plot_samples: list[np.ndarray] = []
    object_status_counts: dict[str, Counter[str]] = {oid: Counter() for oid in candidates}
    object_residuals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (frame_idx, oid), vis_row in visible_rows.items():
        pose = stable.get((frame_idx, oid))
        fused = fused_samples.get(oid)
        if pose is None or fused is None:
            continue
        visible = row_vertices(npz, vis_row)
        if len(visible) > args.max_visible_points_for_residual:
            visible_eval = visible[rng.choice(len(visible), size=args.max_visible_points_for_residual, replace=False)]
        else:
            visible_eval = visible
        fused_world = transform_points(fused, pose)
        stats = residual_stats(fused_world, visible_eval)
        status = classify(stats, args.median_threshold_m, args.p95_threshold_m)
        stats["status"] = status
        stats["frame_idx"] = frame_idx
        stats["object_id"] = oid
        stats["thresholds_m"] = {"median": args.median_threshold_m, "p95": args.p95_threshold_m}
        residual_by_frame[(frame_idx, oid)] = stats
        object_status_counts[oid][status] += 1
        object_residuals[oid].append(stats)
        all_plot_samples.append(visible_eval[:: max(1, len(visible_eval) // 100)])
        all_plot_samples.append(fused_world[:: max(1, len(fused_world) // 100)])

    if all_plot_samples:
        sample = np.concatenate(all_plot_samples, axis=0)
        x_min, z_min = np.percentile(sample[:, [0, 2]], 1, axis=0)
        x_max, z_max = np.percentile(sample[:, [0, 2]], 99, axis=0)
    else:
        x_min, x_max, z_min, z_max = -1.0, 1.0, 0.0, 5.0
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0; z_max += 1.0

    case_dir = args.output_root / case / "rigid_se3_residual_check"
    frame_dir = case_dir / "world_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    canvas_w, canvas_h = 1280, 720
    plot_left, plot_top, plot_right, plot_bottom = 70, 88, canvas_w - 330, canvas_h - 78
    draw_counts: Counter[str] = Counter()

    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        image = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        draw.text((12, 12), f"V18 rigid SE(3) residual check {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 90, 100), width=1)
        y = 54
        for oid, cand in candidates.items():
            stats = residual_by_frame.get((frame_idx, oid))
            pose = stable.get((frame_idx, oid))
            vis_row = visible_rows.get((frame_idx, oid))
            if stats is None or pose is None or vis_row is None:
                draw_label(draw, (12, y), f"{oid.replace('object:', '')}: no residual row", small, (120, 120, 130), (0, 0, 0))
                y += 20
                continue
            visible = row_vertices(npz, vis_row)
            if len(visible) > args.max_render_points:
                visible = visible[rng.choice(len(visible), size=args.max_render_points, replace=False)]
            fused_world = transform_points(fused_samples[oid], pose)
            if len(fused_world) > args.max_render_points:
                fused_world = fused_world[rng.choice(len(fused_world), size=args.max_render_points, replace=False)]
            status = str(stats["status"])
            color = color_for_status(status)
            for pts, rgb in [(fused_world, (255, 150, 60)), (visible, (80, 190, 255))]:
                for xw, _yw, zw in pts:
                    px = int(plot_left + (float(xw) - x_min) / (x_max - x_min) * (plot_right - plot_left))
                    py = int(plot_bottom - (float(zw) - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
                    if plot_left <= px <= plot_right and plot_top <= py <= plot_bottom:
                        draw.point((px, py), fill=rgb)
            center = np.mean(visible, axis=0) if len(visible) else np.zeros(3)
            cx = int(plot_left + (center[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
            cy = int(plot_bottom - (center[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=color, width=3)
            label = f"{oid.replace('object:', '')[:24]} {status} med={stats.get('visible_to_fused_median_m'):.3f} p95={stats.get('visible_to_fused_p95_m'):.3f}"
            draw_label(draw, (12, y), label[:110], small, color, (0, 0, 0))
            y += 20
            draw_counts[f"residual::{status}"] += 1
            draw_counts[f"residual_object::{oid}"] += 1
        draw_label(draw, (12, canvas_h - 42), "Orange=fused canonical under stable SE3; blue=frame-local visible surface. Diagnostic only, not pose acceptance.", small, (255, 255, 255), (0, 0, 0))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)

    video_path = case_dir / "v18_rigid_se3_residual_check.mp4"
    encode_video(frame_dir, video_path, fps)
    object_reports: dict[str, Any] = {}
    for oid, rows in object_residuals.items():
        medians = [r.get("visible_to_fused_median_m") for r in rows if isinstance(r.get("visible_to_fused_median_m"), float)]
        p95s = [r.get("visible_to_fused_p95_m") for r in rows if isinstance(r.get("visible_to_fused_p95_m"), float)]
        fused_p95s = [r.get("fused_to_visible_p95_m") for r in rows if isinstance(r.get("fused_to_visible_p95_m"), float)]
        object_reports[oid] = {
            "row_count": len(rows),
            "status_counts": dict(sorted(object_status_counts[oid].items())),
            "visible_to_fused_median_of_medians_m": float(np.median(medians)) if medians else None,
            "visible_to_fused_p95_of_p95s_m": float(np.percentile(p95s, 95)) if p95s else None,
            "fused_to_visible_p95_of_p95s_m": float(np.percentile(fused_p95s, 95)) if fused_p95s else None,
            "thresholds_m": {"median": args.median_threshold_m, "p95": args.p95_threshold_m},
            "claim_scope": "bidirectional_residual_test_of_stable_rigid_prior_against_frame_local_visible_surface_not_pose_acceptance",
        }
    report = {
        "method": "render_v18_rigid_se3_residual_check",
        "case": case,
        "claim_scope": "tests_transformed_fused_canonical_geometry_against_frame_local_visible_surface; refutes_or_supports_uncertain_rigid_prior_without_accepting_pose",
        "frame_count": len(frames),
        "fps": fps,
        "candidate_objects": object_reports,
        "residual_rows": [residual_by_frame[key] for key in sorted(residual_by_frame)],
        "outputs": {"world_video": str(video_path)},
        "frame_counts": {"world": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(draw_counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_rigid_se3_residual_check_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_rigid_se3_residual_check",
        "status": "rigid_se3_residual_check_not_pose_acceptance",
        "cases": reports,
        "all_world_frame_counts_match": all(r["frame_counts"].get("world") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_rigid_se3_residual_check_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--translation-smoothing-radius", type=int, default=3)
    parser.add_argument("--max-fused-points", type=int, default=1400)
    parser.add_argument("--max-visible-points-for-residual", type=int, default=1600)
    parser.add_argument("--max-render-points", type=int, default=1200)
    parser.add_argument("--median-threshold-m", type=float, default=0.04)
    parser.add_argument("--p95-threshold-m", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
