#!/usr/bin/env python3
"""Generic rigid-object SE(3) corrective attempt.

Selects objects from model/fast-motion metadata, loads their V18 depth-fused
canonical point clouds, and renders those point clouds transformed by the V18
factor-graph object_se3 estimates. This is an artifact-changing attempt: it
renders geometry under SE(3), not just boxes/status fields. It is not full V18
closure and does not claim hidden geometry is complete.
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


def encode_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    t = pose6[:3]
    return points @ r.T + t[None, :]


def graph_object_poses(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
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
            out[vid.split("::", 1)[1]] = {
                "pose6": np.array([finite_float(v) for v in est[:6]], dtype=np.float64),
                "residual_norm": finite_float(row.get("observation_residual_norm"), float("nan")),
                "source": row.get("source"),
            }
    return out


def acceleration(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if len(values) < 3:
        return {"count": len(values), "mean": None, "p95": None}
    mags: list[float] = []
    for a, b, c in zip(values[:-2], values[1:-1], values[2:]):
        mags.append(float(np.linalg.norm(c - 2.0 * b + a)))
    mags.sort()
    return {"count": len(values), "mean": float(sum(mags) / len(mags)), "p95": float(mags[min(len(mags) - 1, int(math.ceil(0.95 * len(mags))) - 1)])}


def candidate_rows(case: str, visible_root: Path, fused_root: Path) -> dict[str, dict[str, Any]]:
    visible = load_json(visible_root / case / "v18_visible_geometry_archive_report.json")
    fused = load_json(fused_root / case / "v18_depth_fused_reconstruction_report.json")
    visible_by_id = {str(r.get("object_id")): r for r in visible.get("object_rows", []) if isinstance(r, dict)}
    out: dict[str, dict[str, Any]] = {}
    for row in fused.get("object_rows", []):
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id"))
        vis = visible_by_id.get(oid, {})
        model_state = str(vis.get("model_physical_state_type", "")).lower()
        fast_state = str(vis.get("fast_motion_state", "")).lower()
        selection_reasons = []
        if model_state == "rigid":
            selection_reasons.append("model_physical_state_type_rigid")
        if "rigid" in fast_state:
            selection_reasons.append(f"fast_motion_state_{fast_state}")
        mesh = row.get("mesh_reconstruction", {}) if isinstance(row.get("mesh_reconstruction"), dict) else {}
        ply = mesh.get("fused_point_cloud_path")
        if selection_reasons and isinstance(ply, str) and Path(ply).exists():
            out[oid] = {
                "object_id": oid,
                "name": str(vis.get("name") or oid),
                "model_physical_state_type": vis.get("model_physical_state_type"),
                "fast_motion_state": vis.get("fast_motion_state"),
                "selection_reasons": selection_reasons,
                "source_frame_count": row.get("source_frame_count"),
                "source_point_count": row.get("source_point_count"),
                "fused_point_cloud_path": str(ply),
                "canonical_bbox_min_m": row.get("canonical_bbox_min_m"),
                "canonical_bbox_max_m": row.get("canonical_bbox_max_m"),
                "hidden_geometry_status": row.get("hidden_geometry_status"),
            }
    return out


def color_for_index(i: int) -> tuple[int, int, int]:
    palette = [(255, 90, 90), (80, 180, 255), (255, 210, 60), (80, 255, 140), (220, 120, 255), (255, 150, 70)]
    return palette[i % len(palette)]


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", [])
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    fps = finite_float(ann.get("fps"), 30.0)
    candidates = candidate_rows(case, args.visible_geometry_root, args.depth_fused_root)
    point_clouds: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(18)
    for oid, row in candidates.items():
        pts = read_binary_xyz_ply(Path(str(row["fused_point_cloud_path"])))
        if pts.shape[0] > args.max_points_per_object:
            idx = rng.choice(pts.shape[0], size=args.max_points_per_object, replace=False)
            pts = pts[idx]
        point_clouds[oid] = pts

    raw_pose_rows: dict[str, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    frame_pose_rows: dict[int, dict[str, dict[str, Any]]] = {}
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        frame_pose_rows[frame_idx] = poses
        for oid in point_clouds:
            pose = poses.get(oid)
            if pose is not None:
                raw_pose_rows[oid].append((frame_idx, pose["pose6"].copy(), float(pose.get("residual_norm", float("nan")))))

    stable_pose_by_frame: dict[tuple[int, str], np.ndarray] = {}
    stable_pose_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    for oid, rows in raw_pose_rows.items():
        ordered = sorted(rows, key=lambda x: x[0])
        if not ordered:
            continue
        translations = [p[:3] for _, p, _ in ordered]
        median_rot = np.median(np.stack([p[3:6] for _, p, _ in ordered], axis=0), axis=0)
        for i, (frame_idx, pose6, _residual) in enumerate(ordered):
            lo = max(0, i - args.translation_smoothing_radius)
            hi = min(len(ordered), i + args.translation_smoothing_radius + 1)
            stable = pose6.copy()
            stable[:3] = np.mean(np.stack(translations[lo:hi], axis=0), axis=0)
            stable[3:6] = median_rot
            stable_pose_by_frame[(frame_idx, oid)] = stable
            stable_pose_rows[oid].append(stable.copy())

    transformed_samples: list[np.ndarray] = []
    stride = max(1, len(frames) // 80)
    for frame in frames[::stride]:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", 0))
        for oid, pts in point_clouds.items():
            stable_pose = stable_pose_by_frame.get((frame_idx, oid))
            if stable_pose is not None:
                transformed_samples.append(transform_points(pts[: min(len(pts), 200)], stable_pose))
    if transformed_samples:
        sample_all = np.concatenate(transformed_samples, axis=0)
        x_min, z_min = np.percentile(sample_all[:, [0, 2]], 1, axis=0)
        x_max, z_max = np.percentile(sample_all[:, [0, 2]], 99, axis=0)
    else:
        x_min, x_max, z_min, z_max = -1.0, 1.0, 0.0, 5.0
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0; z_max += 1.0

    case_dir = args.output_root / case / "rigid_se3_attempt"
    frame_dir = case_dir / "world_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    big = font(20)
    small = font(14)
    counts: Counter[str] = Counter()
    canvas_w, canvas_h = 1280, 720
    plot_left, plot_top, plot_right, plot_bottom = 70, 86, canvas_w - 330, canvas_h - 78

    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        image = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        draw.text((12, 12), f"Generic rigid SE(3) attempt {case} frame {frame_idx+1}/{len(frames)}", font=big, fill=(255, 255, 255))
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 90, 100), width=1)
        for i, (oid, pts) in enumerate(point_clouds.items()):
            pose = poses.get(oid)
            stable_pose = stable_pose_by_frame.get((frame_idx, oid))
            if pose is None or stable_pose is None:
                continue
            xyz = transform_points(pts, stable_pose)
            color = color_for_index(i)
            for xw, _yw, zw in xyz:
                px = int(plot_left + (float(xw) - x_min) / (x_max - x_min) * (plot_right - plot_left))
                py = int(plot_bottom - (float(zw) - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
                if plot_left <= px <= plot_right and plot_top <= py <= plot_bottom:
                    draw.point((px, py), fill=color)
            center = stable_pose[:3]
            cx = int(plot_left + (center[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
            cy = int(plot_bottom - (center[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            raw_center = pose["pose6"][:3]
            rx = int(plot_left + (raw_center[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
            ry = int(plot_bottom - (raw_center[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            draw.line((rx, ry, cx, cy), fill=(180, 180, 180), width=1)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=(255, 255, 255), width=2)
            draw_label(draw, (cx + 8, cy - 10), f"{oid.replace('object:', '')[:30]} stable rigid", small, color, (16, 18, 24))
            counts[f"rendered::{oid}"] += 1
        y = 54
        for i, (oid, row) in enumerate(candidates.items()):
            color = color_for_index(i)
            text = f"{oid.replace('object:', '')}: {row['selection_reasons'][0]} | points={len(point_clouds.get(oid, []))}"
            draw_label(draw, (12, y), text[:95], small, color, (0, 0, 0))
            y += 20
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)

    video_path = case_dir / "v18_rigid_se3_world_attempt.mp4"
    encode_video(frame_dir, video_path, fps)
    side_path = None
    corrective_overlay = args.corrective_root / case / "v18_corrective_overlay_graph_driven.mp4"
    if corrective_overlay.exists():
        side_path = case_dir / "v18_rigid_se3_side_by_side_attempt.mp4"
        compose_side_by_side(corrective_overlay, video_path, side_path)

    object_reports: dict[str, Any] = {}
    for oid, row in candidates.items():
        raw_pose_list = [p for _frame_idx, p, _residual in raw_pose_rows.get(oid, [])]
        stable_pose_list = stable_pose_rows.get(oid, [])
        raw_trans = [p[:3] for p in raw_pose_list]
        raw_rot = [p[3:6] for p in raw_pose_list]
        stable_trans = [p[:3] for p in stable_pose_list]
        stable_rot = [p[3:6] for p in stable_pose_list]
        residuals = sorted([res for _frame_idx, _pose, res in raw_pose_rows.get(oid, []) if math.isfinite(res)])
        p95_res = residuals[min(len(residuals) - 1, int(math.ceil(0.95 * len(residuals))) - 1)] if residuals else None
        object_reports[oid] = {
            **row,
            "rendered_pose_rows": len(stable_pose_list),
            "point_cloud_sample_count": int(point_clouds.get(oid, np.empty((0, 3))).shape[0]),
            "factor_graph_observation_residual_mean": float(sum(residuals) / len(residuals)) if residuals else None,
            "factor_graph_observation_residual_p95": p95_res,
            "raw_graph_translation_second_difference_m": acceleration(raw_trans),
            "raw_graph_rotation_vector_second_difference_rad": acceleration(raw_rot),
            "stable_rigid_prior_translation_second_difference_m": acceleration(stable_trans),
            "stable_rigid_prior_rotation_vector_second_difference_rad": acceleration(stable_rot),
            "stable_rigid_prior": "componentwise_median_rotation_vector_plus_local_mean_translation_no_object_name_branch",
            "attempt_claim": "generic_rigid_se3_geometry_render_attempt_from_depth_fused_point_cloud_and_stabilized_rigid_prior_pose",
        }

    report = {
        "method": "render_v18_rigid_se3_attempt",
        "case": case,
        "claim_scope": "generic_metadata_selected_rigid_or_local_rigid_se3_attempt; not_complete_hidden_geometry_or_final_v18_closure",
        "frame_count": len(frames),
        "fps": fps,
        "candidate_count": len(candidates),
        "candidate_objects": object_reports,
        "outputs": {
            "world_video": str(video_path),
            "side_by_side_video": str(side_path) if side_path else None,
        },
        "frame_counts": {
            "world": ffprobe_frame_count(video_path),
            "side_by_side": ffprobe_frame_count(side_path) if side_path else None,
        },
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_rigid_se3_attempt_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    out = {
        "method": "render_v18_rigid_se3_attempt",
        "status": "generic_rigid_se3_attempt_not_full_v18_closure",
        "output_root": str(args.output_root),
        "cases": reports,
        "all_world_frame_counts_match": all(r["frame_counts"].get("world") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_rigid_se3_attempt_summary.json", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--depth-fused-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-points-per-object", type=int, default=1200)
    parser.add_argument("--translation-smoothing-radius", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
