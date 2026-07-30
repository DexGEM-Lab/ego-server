#!/usr/bin/env python3
"""Render visible-surface coverage/completeness audit for V18 rigid candidates.

This is diagnostic only. It uses existing visible RGBD surfaces and uncertain
stable rigid priors to test whether current observations support a complete
object-geometry claim. It never accepts a proxy/hull/primitive as object mesh.
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
from scipy.spatial.transform import Rotation as R


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


def rounded(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), ndigits)
    if isinstance(value, list):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), ndigits) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


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
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def scale_box(box: tuple[float, float, float, float], sx: float, sy: float) -> tuple[int, int, int, int]:
    return int(round(box[0] * sx)), int(round(box[1] * sy)), int(round(box[2] * sx)), int(round(box[3] * sy))


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


def graph_object_poses(frame: dict[str, Any]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
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
            out[vid.split("::", 1)[1]] = [finite_float(v) for v in est[:6]]
    return out


def stable_pose_index_from_source(frames: list[Any], candidate_ids: set[str], radius: int) -> dict[tuple[int, str], list[float]]:
    raw: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        for oid in candidate_ids:
            pose = poses.get(oid)
            if pose is not None:
                raw[oid].append((frame_idx, np.asarray(pose, dtype=np.float64)))
    stable: dict[tuple[int, str], list[float]] = {}
    for oid, rows in raw.items():
        ordered = sorted(rows, key=lambda x: x[0])
        if not ordered:
            continue
        translations = [pose[:3] for _frame_idx, pose in ordered]
        median_rot = np.median(np.stack([pose[3:6] for _frame_idx, pose in ordered], axis=0), axis=0)
        for i, (frame_idx, pose6) in enumerate(ordered):
            lo = max(0, i - radius)
            hi = min(len(ordered), i + radius + 1)
            stable_pose = pose6.copy()
            stable_pose[:3] = np.mean(np.stack(translations[lo:hi], axis=0), axis=0)
            stable_pose[3:6] = median_rot
            stable[(frame_idx, oid)] = stable_pose.tolist()
    return stable


def candidate_ids(report: dict[str, Any]) -> list[str]:
    c = report.get("candidate_objects")
    if isinstance(c, dict):
        return sorted(str(k) for k in c)
    return []


def object_by_id(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id")): obj for obj in frame.get("objects", []) if isinstance(obj, dict)}


def transform_to_object(vertices_world: np.ndarray, pose6: list[float]) -> np.ndarray:
    rot = R.from_rotvec(np.asarray(pose6[3:6], dtype=np.float64)).as_matrix()
    t = np.asarray(pose6[:3], dtype=np.float64)
    return (vertices_world.astype(np.float64) - t[None, :]) @ rot


def spherical_coverage(points: np.ndarray, az_bins: int, el_bins: int) -> dict[str, Any]:
    if len(points) == 0:
        return {"bin_count": 0, "coverage_fraction": 0.0, "octant_count": 0}
    center = np.median(points, axis=0)
    vec = points - center[None, :]
    norms = np.linalg.norm(vec, axis=1)
    keep = norms > 1e-8
    if not np.any(keep):
        return {"bin_count": 0, "coverage_fraction": 0.0, "octant_count": 0}
    unit = vec[keep] / norms[keep, None]
    az = (np.arctan2(unit[:, 1], unit[:, 0]) + math.pi) / (2.0 * math.pi)
    el = (np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)) + math.pi / 2.0) / math.pi
    ai = np.clip((az * az_bins).astype(int), 0, az_bins - 1)
    ei = np.clip((el * el_bins).astype(int), 0, el_bins - 1)
    bins = set(zip(ai.tolist(), ei.tolist()))
    octants = set(tuple(v) for v in (unit > 0.0).astype(int).tolist())
    return {
        "bin_count": len(bins),
        "total_bins": az_bins * el_bins,
        "coverage_fraction": len(bins) / float(az_bins * el_bins),
        "octant_count": len(octants),
        "median_radius_m": float(np.median(norms[keep])),
        "p95_radius_m": float(np.percentile(norms[keep], 95)),
    }


def classify(row_count: int, max_extent_ratio: float, coverage_fraction: float, args: argparse.Namespace) -> str:
    if row_count < args.min_rows_for_coverage_claim:
        return "insufficient_view_count_for_geometry_completion_claim"
    if max_extent_ratio >= args.max_extent_ratio_for_alignment_stable:
        return "coverage_confounded_by_pose_alignment_overspread"
    if coverage_fraction >= args.coverage_fraction_broad_threshold:
        return "broad_visible_coverage_but_hidden_geometry_still_unresolved"
    return "sparse_visible_coverage_hidden_geometry_unresolved"


def analyze_case(case: str, frames: list[Any], args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    archive_report = load_json(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json")
    visible_state = load_json(args.output_root / case / "visible_surface_state" / "v18_visible_surface_state_report.json")
    ids = candidate_ids(visible_state)
    poses = stable_pose_index_from_source(frames, set(ids), args.translation_smoothing_radius)
    npz = np.load(archive_report["archive_npz"])
    vertices = npz["vertices"]
    object_ids = npz["object_id"]
    frame_ids = npz["frame_idx"]
    offsets = npz["vertex_offsets"]
    summaries: dict[str, dict[str, Any]] = {}
    frame_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for oid in ids:
        all_points: list[np.ndarray] = []
        extents: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        row_count = 0
        missing_pose_rows = 0
        for i, raw_oid in enumerate(object_ids):
            if str(raw_oid) != oid:
                continue
            frame_idx = int(frame_ids[i])
            pose = poses.get((frame_idx, oid))
            if pose is None:
                missing_pose_rows += 1
                continue
            pts = vertices[int(offsets[i]): int(offsets[i + 1])]
            if len(pts) == 0:
                continue
            if len(pts) > args.max_points_per_surface:
                pts = pts[:: max(1, int(math.ceil(len(pts) / args.max_points_per_surface)))]
            obj_pts = transform_to_object(pts, pose)
            all_points.append(obj_pts)
            extents.append(obj_pts.max(axis=0) - obj_pts.min(axis=0))
            centers.append(np.median(obj_pts, axis=0))
            row_count += 1
            frame_rows[frame_idx].append({"object_id": oid})
        if all_points:
            points = np.concatenate(all_points, axis=0)
            ext = np.asarray(extents, dtype=np.float64)
            ctr = np.asarray(centers, dtype=np.float64)
            agg_extent = points.max(axis=0) - points.min(axis=0)
            median_extent = np.median(ext, axis=0)
            ratio = agg_extent / np.maximum(median_extent, 1e-9)
            center_spread = np.linalg.norm(ctr - np.median(ctr, axis=0)[None, :], axis=1)
            cov = spherical_coverage(points, args.azimuth_bins, args.elevation_bins)
            status = classify(row_count, float(np.max(ratio)), float(cov["coverage_fraction"]), args)
            summary = {
                "object_id": oid,
                "status": status,
                "row_count": row_count,
                "missing_pose_rows": missing_pose_rows,
                "sampled_point_count": int(len(points)),
                "aggregate_extent_object_frame_m": rounded(agg_extent, 6),
                "median_frame_extent_object_frame_m": rounded(median_extent, 6),
                "aggregate_to_median_extent_ratio": rounded(ratio, 4),
                "max_aggregate_to_median_extent_ratio": float(np.max(ratio)),
                "center_spread_p95_m": float(np.percentile(center_spread, 95)) if len(center_spread) else None,
                "spherical_coverage": rounded(cov, 6),
                "object_geometry_complete": False,
                "accepted_complete_geometry": False,
                "alignment_pose_scope": "uses_uncertain_stable_rigid_prior_recomputed_from_source_factor_graph_for_diagnostic_alignment_not_accepted_pose",
                "state_role": "visible_surface_coverage_audit_not_geometry_completion",
            }
        else:
            summary = {
                "object_id": oid,
                "status": "no_aligned_visible_surface_points",
                "row_count": 0,
                "missing_pose_rows": missing_pose_rows,
                "object_geometry_complete": False,
                "accepted_complete_geometry": False,
                "alignment_pose_scope": "uses_uncertain_stable_rigid_prior_recomputed_from_source_factor_graph_for_diagnostic_alignment_not_accepted_pose",
                "state_role": "visible_surface_coverage_audit_not_geometry_completion",
            }
        summaries[oid] = summary
    for rows in frame_rows.values():
        for row in rows:
            row.update(summaries.get(row["object_id"], {}))
    return summaries, frame_rows


def color_for_status(status: str) -> tuple[int, int, int]:
    if status == "coverage_confounded_by_pose_alignment_overspread":
        return (255, 90, 90)
    if status == "insufficient_view_count_for_geometry_completion_claim":
        return (255, 190, 70)
    if status == "broad_visible_coverage_but_hidden_geometry_still_unresolved":
        return (80, 210, 255)
    if status == "sparse_visible_coverage_hidden_geometry_unresolved":
        return (190, 120, 255)
    return (180, 180, 180)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    raw_video = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    source_w = finite_float(raw_video.get("width"), 1920.0)
    source_h = finite_float(raw_video.get("height"), 1080.0)
    fps = finite_float(ann.get("fps"), 30.0)
    summaries, frame_rows = analyze_case(case, frames, args)
    case_dir = args.output_root / case / "geometry_coverage_audit"
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
        draw.rectangle((0, 0, image.size[0], 58), fill=(0, 0, 0))
        draw.text((12, 8), f"V18 visible geometry coverage audit {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.text((12, 34), "red=coverage confounded by alignment/overspread; cyan=broad visible only; yellow=too few views; no complete mesh accepted", fill=(230, 230, 230), font=small)
        objects = object_by_id(frame)
        y = 64
        rows = frame_rows.get(frame_idx, [])
        if not rows:
            counts["frames_without_candidate_visible_surface"] += 1
        for row in rows[: args.max_rows_per_frame]:
            oid = str(row.get("object_id"))
            status = str(row.get("status"))
            color = color_for_status(status)
            obj = objects.get(oid, {})
            box = bbox_tuple(obj.get("bbox_xyxy"))
            if box is not None:
                draw.rectangle(scale_box(box, sx, sy), outline=color, width=4)
            ratio = row.get("max_aggregate_to_median_extent_ratio")
            coverage = (row.get("spherical_coverage") or {}).get("coverage_fraction") if isinstance(row.get("spherical_coverage"), dict) else None
            label = f"{oid.replace('object:', '')} {status} rows={row.get('row_count')} ratio={finite_float(ratio, 0):.2f} coverage={finite_float(coverage, 0):.2f}"
            draw_label(draw, (12, y), label[:155], small, color, (0, 0, 0))
            y += 20
            counts[f"draw::{status}"] += 1
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_geometry_coverage_audit.mp4"
    encode_video(frame_dir, video_path, fps)
    status_counts = Counter(str(v.get("status")) for v in summaries.values())
    report = {
        "method": "render_v18_geometry_coverage_audit",
        "case": case,
        "claim_scope": "diagnostic_visible_surface_coverage_and_alignment_spread_audit_not_geometry_completion_or_pose_acceptance",
        "frame_count": len(frames),
        "fps": fps,
        "object_count": len(summaries),
        "status_counts": dict(sorted(status_counts.items())),
        "object_summaries": summaries,
        "parameters": {
            "azimuth_bins": args.azimuth_bins,
            "elevation_bins": args.elevation_bins,
            "min_rows_for_coverage_claim": args.min_rows_for_coverage_claim,
            "max_extent_ratio_for_alignment_stable": args.max_extent_ratio_for_alignment_stable,
            "coverage_fraction_broad_threshold": args.coverage_fraction_broad_threshold,
            "translation_smoothing_radius": args.translation_smoothing_radius,
        },
        "stable_pose_source": "recomputed_from_source_annotations_factor_graph_object_se3_with_same_stable_prior_as_corrective_annotation_builder",
        "outputs": {"video": str(video_path)},
        "frame_counts": {"video": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_geometry_coverage_audit_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_geometry_coverage_audit",
        "status": "geometry_coverage_audit_not_completion",
        "cases": reports,
        "all_video_frame_counts_match": all(r["frame_counts"].get("video") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_geometry_coverage_audit_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--azimuth-bins", type=int, default=24)
    parser.add_argument("--elevation-bins", type=int, default=12)
    parser.add_argument("--min-rows-for-coverage-claim", type=int, default=20)
    parser.add_argument("--max-extent-ratio-for-alignment-stable", type=float, default=3.0)
    parser.add_argument("--coverage-fraction-broad-threshold", type=float, default=0.75)
    parser.add_argument("--translation-smoothing-radius", type=int, default=3)
    parser.add_argument("--max-points-per-surface", type=int, default=200)
    parser.add_argument("--max-rows-per-frame", type=int, default=4)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
