#!/usr/bin/env python3
"""Render local nonpenetration repair proposals for contact rows.

This does not modify accepted annotations. It computes a diagnostic translation
that would move sampled hand points out of local triangle-normal penetration for
rows already flagged by local triangle evidence. The purpose is to quantify
whether contact could be repaired by a small local displacement, or whether the
geometry/contact state is too inconsistent for a credible correction.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_v18_triangle_nonpenetration_evidence import (
    closest_points_on_triangles,
    face_geometry,
    frame_mesh,
    hand_points,
    load_v16,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


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


def triangle_row_index(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    report = load_json(path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")), str(row.get("object_id")))] = row
    return out


def accepted_contact_rows(path: Path) -> list[dict[str, Any]]:
    report = load_json(path)
    rows = []
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict) and row.get("accepted_contact_owner") is True:
            rows.append(row)
    return rows


def local_triangle_signed_distances(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str | None]:
    tri, centroids, normals, _face_ids = face_geometry(vertices, faces)
    if len(tri) == 0:
        return np.asarray([], dtype=np.float64), np.zeros((0, 3), dtype=np.float64), "invalid_triangle_normals"
    k = min(args.nearest_triangle_candidates, len(tri))
    tree = cKDTree(centroids)
    _, raw_indices = tree.query(points, k=k)
    indices = np.atleast_2d(raw_indices)
    if indices.shape[0] != len(points):
        indices = indices.T
    signed: list[float] = []
    chosen_normals: list[np.ndarray] = []
    for i, point in enumerate(points):
        cand = np.asarray(indices[i], dtype=np.int64)
        cand_tri = tri[cand]
        closest = closest_points_on_triangles(point, cand_tri)
        vec = point[None, :] - closest
        d2 = np.einsum("ij,ij->i", vec, vec)
        best_local = int(np.argmin(d2))
        best_idx = int(cand[best_local])
        normal = normals[best_idx]
        signed.append(float(np.dot(vec[best_local], normal)))
        chosen_normals.append(normal)
    return np.asarray(signed, dtype=np.float64), np.asarray(chosen_normals, dtype=np.float64), None


def post_translation_metrics(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, translation: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    signed_arr, _normals_arr, blocker = local_triangle_signed_distances(points + translation[None, :], vertices, faces, args)
    if blocker is not None or len(signed_arr) == 0:
        return {"post_translation_local_check_status": "blocked", "post_translation_blocker": blocker or "empty_signed_distance_result"}
    penetrated = signed_arr < -args.penetration_tolerance_m
    return {
        "post_translation_local_check_status": "local_triangle_tolerance_pass" if not np.any(penetrated) else "local_triangle_tolerance_failed",
        "post_translation_min_signed_m": float(np.min(signed_arr)),
        "post_translation_median_signed_m": float(np.median(signed_arr)),
        "post_translation_penetrated_point_count": int(np.sum(penetrated)),
        "post_translation_penetrated_point_fraction": float(np.mean(penetrated)),
        "post_translation_local_metric_passed": bool(not np.any(penetrated)),
    }


def repair_from_geometry(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if len(points) > args.max_query_hand_points:
        step = max(1, int(math.ceil(len(points) / args.max_query_hand_points)))
        points = points[::step]
    signed_arr, normals_arr, blocker = local_triangle_signed_distances(points, vertices, faces, args)
    if blocker is not None or len(signed_arr) == 0:
        return {"status": "blocked", "blocker": blocker or "empty_signed_distance_result"}
    penetrated = signed_arr < -args.penetration_tolerance_m
    if not np.any(penetrated):
        return {
            "status": "no_local_repair_needed_by_triangle_tolerance",
            "sampled_hand_points": int(len(points)),
            "penetrated_point_count": 0,
            "min_signed_m": float(np.min(signed_arr)),
            "median_signed_m": float(np.median(signed_arr)),
            "proposal_complete_nonpenetration": False,
        }
    depths = -signed_arr[penetrated]
    pen_normals = normals_arr[penetrated]
    weighted = np.sum(pen_normals * depths[:, None], axis=0)
    norm = float(np.linalg.norm(weighted))
    if norm < 1e-9:
        direction = np.mean(pen_normals, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return {"status": "blocked", "blocker": "incoherent_penetration_normals"}
        direction = direction / norm
    else:
        direction = weighted / norm
    alignment = float(np.linalg.norm(np.mean(pen_normals, axis=0)))
    required = float(np.max(depths) + args.penetration_tolerance_m)
    translation = direction * required
    post = post_translation_metrics(points, vertices, faces, translation, args)
    post_passed = post.get("post_translation_local_metric_passed") is True
    if required <= args.small_repair_threshold_m and alignment >= args.normal_alignment_threshold:
        status = "small_coherent_translation_candidate_local_postcheck_pass" if post_passed else "small_coherent_translation_candidate_local_postcheck_failed"
    elif alignment < args.normal_alignment_threshold:
        status = "translation_candidate_unreliable_incoherent_normals"
    else:
        status = "large_local_translation_required"
    return {
        "status": status,
        "sampled_hand_points": int(len(points)),
        "penetrated_point_count": int(np.sum(penetrated)),
        "penetrated_point_fraction": float(np.mean(penetrated)),
        "min_signed_m": float(np.min(signed_arr)),
        "median_signed_m": float(np.median(signed_arr)),
        "max_penetration_depth_m": float(np.max(depths)),
        "proposed_translation_world_m": translation.tolist(),
        "proposed_translation_norm_m": required,
        "penetration_normal_alignment": alignment,
        **post,
        "proposal_complete_nonpenetration": False,
        "candidate_applied_to_annotation": False,
        "semantics": "single_local_normal_translation_candidate_from_v16_visible_hand_points_and_v16_open_object_mesh_triangle_normals_not_applied_not_sdf_not_current_v18_graph_shifted_hand_state",
    }


def color_for_status(status: str) -> tuple[int, int, int]:
    if status == "small_coherent_translation_candidate_local_postcheck_pass":
        return (80, 255, 140)
    if status == "small_coherent_translation_candidate_local_postcheck_failed":
        return (255, 120, 80)
    if status == "large_local_translation_required":
        return (255, 190, 60)
    if status == "translation_candidate_unreliable_incoherent_normals":
        return (255, 80, 80)
    return (180, 180, 180)


def render_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann = load_json(args.source_root / case / "annotations_v18_full.json")
    frames = ann.get("frames", []) if isinstance(ann.get("frames"), list) else []
    fps = float(ann.get("fps") or 30.0)
    contact_rows = accepted_contact_rows(args.contact_root / case / "v18_contact_ownership_graph_report.json")
    tri_rows = triangle_row_index(args.triangle_root / case / "v18_triangle_nonpenetration_evidence_report.json")
    v16_frames, mesh_index = load_v16(case, args)
    mesh_cache: dict[int, tuple[np.ndarray | None, np.ndarray | None, str | None]] = {}
    hand_cache: dict[tuple[int, str], tuple[np.ndarray | None, str | None, str | None]] = {}
    proposal_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    centers: list[np.ndarray] = []
    for row in contact_rows:
        frame_idx = int(row.get("frame_idx", -1))
        side = str(row.get("hand_side"))
        oid = str(row.get("object_id"))
        tri = tri_rows.get((frame_idx, side, oid), {})
        if tri.get("local_triangle_penetration_detected") is not True:
            continue
        if frame_idx not in mesh_cache:
            mesh_cache[frame_idx] = frame_mesh(mesh_index, frame_idx)
        vertices, faces, mesh_blocker = mesh_cache[frame_idx]
        if (frame_idx, side) not in hand_cache:
            hand_cache[(frame_idx, side)] = hand_points(v16_frames.get(frame_idx, {}), side, args.max_hand_points)
        points, hand_blocker, hand_source = hand_cache[(frame_idx, side)]
        source_claim = str(row.get("contact_owner_claim") or "")
        if source_claim.startswith("accepted_contact_owner"):
            source_claim = source_claim.replace("accepted_contact_owner", "source_graph_contact_candidate", 1)
        proposal = {
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": oid,
            "source_contact_owner_claim": source_claim,
            "source_contact_owner_claim_context": "contact_graph_candidate_before_local_nonpenetration_veto_not_final_contact_acceptance",
            "source_graph_contact_candidate_before_nonpenetration_veto": bool(row.get("accepted_contact_owner")),
            "source_min_unsigned_distance_m": row.get("min_hand_surface_to_v16_object_mesh_m"),
            "source_triangle_min_signed_m": tri.get("min_local_triangle_signed_distance_m"),
            "source_triangle_negative_fraction": tri.get("negative_triangle_signed_distance_fraction"),
            "hand_geometry_source": hand_source,
            "diagnostic_geometry_basis": "v16_visible_hand_points_and_v16_object_mesh_not_current_v18_graph_shifted_or_temporal_smoothed_hand_state",
        }
        if mesh_blocker or hand_blocker or vertices is None or faces is None or points is None:
            proposal.update({"status": "blocked", "blocker": mesh_blocker or hand_blocker or "missing_geometry"})
        else:
            proposal.update(repair_from_geometry(points, vertices, faces, args))
            centers.append(np.mean(points, axis=0))
            tv = proposal.get("proposed_translation_world_m")
            if isinstance(tv, list) and len(tv) == 3:
                centers.append(np.mean(points, axis=0) + np.asarray(tv, dtype=np.float64))
        counts[f"proposal::{proposal.get('status')}"] += 1
        proposal_index[(frame_idx, side, oid)] = proposal
    if centers:
        sample = np.stack(centers, axis=0)
        x_min, z_min = np.percentile(sample[:, [0, 2]], 1, axis=0)
        x_max, z_max = np.percentile(sample[:, [0, 2]], 99, axis=0)
    else:
        x_min, x_max, z_min, z_max = -1.0, 1.0, 0.0, 5.0
    if abs(x_max - x_min) < 1e-3:
        x_min -= 1.0; x_max += 1.0
    if abs(z_max - z_min) < 1e-3:
        z_min -= 1.0; z_max += 1.0

    case_dir = args.output_root / case / "nonpenetration_repair_proposal"
    stale_world_dir = case_dir / "world_frames"
    if stale_world_dir.exists():
        shutil.rmtree(stale_world_dir)
    frame_dir = case_dir / "diagnostic_xz_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = 1280, 720
    plot_left, plot_top, plot_right, plot_bottom = 70, 88, canvas_w - 330, canvas_h - 78
    big = font(20)
    small = font(14)
    frame_to_props: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prop in proposal_index.values():
        frame_to_props[int(prop["frame_idx"])].append(prop)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        image = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, canvas_w, 48), fill=(0, 0, 0))
        draw.text((12, 12), f"V18 local nonpenetration translation candidates {case} frame {frame_idx+1}/{len(frames)}", fill=(255, 255, 255), font=big)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(90, 90, 100), width=1)
        y = 54
        for prop in frame_to_props.get(frame_idx, []):
            side = str(prop.get("hand_side"))
            key = (frame_idx, side)
            points = hand_cache.get(key, (None, None, None))[0]
            if points is None:
                continue
            center = np.mean(points, axis=0)
            tv = prop.get("proposed_translation_world_m")
            status = str(prop.get("status"))
            color = color_for_status(status)
            cx = int(plot_left + (center[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
            cy = int(plot_bottom - (center[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
            if isinstance(tv, list) and len(tv) == 3:
                end = center + np.asarray(tv, dtype=np.float64)
                ex = int(plot_left + (end[0] - x_min) / (x_max - x_min) * (plot_right - plot_left))
                ey = int(plot_bottom - (end[2] - z_min) / (z_max - z_min) * (plot_bottom - plot_top))
                draw.line((cx, cy, ex, ey), fill=color, width=5)
                draw.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), outline=color, width=2)
            draw_label(draw, (12, y), f"{side}->{str(prop.get('object_id')).replace('object:', '')} {status} move={prop.get('proposed_translation_norm_m', None)} post={prop.get('post_translation_local_check_status')}", small, color, (0, 0, 0))
            y += 20
        if not frame_to_props.get(frame_idx):
            counts["frames_without_repair_proposal"] += 1
        draw_label(draw, (12, canvas_h - 42), "Abstract X-Z diagnostic from V16 local geometry; not a metric 3D repair render; not applied; not complete SDF/nonpenetration.", small, (255, 255, 255), (0, 0, 0))
        image.save(frame_dir / f"{frame_idx:06d}.jpg", quality=90)
    video_path = case_dir / "v18_nonpenetration_repair_proposal.mp4"
    encode_video(frame_dir, video_path, fps)
    proposal_rows = list(proposal_index.values())
    report = {
        "method": "render_v18_nonpenetration_repair_proposal",
        "case": case,
        "claim_scope": "diagnostic_local_translation_candidates_for_v16_triangle_penetration_rows_with_post_translation_local_check_not_applied_not_complete_nonpenetration",
        "frame_count": len(frames),
        "fps": fps,
        "proposal_rows": len(proposal_rows),
        "proposal_status_counts": dict(sorted(Counter(str(r.get("status")) for r in proposal_rows).items())),
        "parameters": {
            "penetration_tolerance_m": args.penetration_tolerance_m,
            "small_repair_threshold_m": args.small_repair_threshold_m,
            "normal_alignment_threshold": args.normal_alignment_threshold,
        },
        "rows": proposal_rows,
        "outputs": {"diagnostic_xz_video": str(video_path)},
        "frame_counts": {"diagnostic_xz": ffprobe_frame_count(video_path)},
        "draw_counts": dict(sorted(counts.items())),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(case_dir / "v18_nonpenetration_repair_proposal_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [render_case(case, args) for case in args.cases]
    summary = {
        "method": "render_v18_nonpenetration_repair_proposal",
        "status": "nonpenetration_repair_proposal_not_applied_not_complete_sdf",
        "cases": reports,
        "all_diagnostic_xz_frame_counts_match": all(r["frame_counts"].get("diagnostic_xz") == r["frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_nonpenetration_repair_proposal_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--contact-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--triangle-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-query-hand-points", type=int, default=128)
    parser.add_argument("--nearest-triangle-candidates", type=int, default=32)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    parser.add_argument("--small-repair-threshold-m", type=float, default=0.02)
    parser.add_argument("--normal-alignment-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
