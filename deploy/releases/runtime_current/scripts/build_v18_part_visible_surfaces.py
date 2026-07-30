#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_visible_surface_evidence"
CLAIM = (
    "This artifact extracts bounded depth-backed visible surfaces for accepted part/segment masks. The surfaces "
    "are part-level visible geometry evidence only; they do not reconstruct hidden geometry, estimate part pose, "
    "or complete object pose."
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


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def normalize_path(path: str) -> Path:
    candidates = [
        Path(path),
        Path(path.replace("/mnt/user-home/yiwen/ego_annotation_remote/data", "/data2/ego_annotation_outputs")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_metric_depth(path: Path) -> dict[str, Any]:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing keys {missing}")
    frame_idx = blob["frame_idx"].astype(np.int32)
    depth = blob["depth"].astype(np.float32)
    intrinsics = blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if len(frame_idx) != depth.shape[0] or len(frame_idx) != intrinsics.shape[0]:
        raise RuntimeError(f"{path} inconsistent depth rows")
    return {"frame_idx": frame_idx, "depth": depth, "intrinsics": intrinsics, "frame_to_i": {int(v): int(i) for i, v in enumerate(frame_idx)}}


def load_mask(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = str(path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not path.exists():
        raise RuntimeError(f"mask path missing: {path}")
    arr = np.asarray(Image.open(path).convert("L")) > 0
    cache[key] = arr
    return arr


def resize_bool_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape_hw:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def object_mask_index(annotation: dict[str, Any]) -> dict[tuple[int, str], Path]:
    out: dict[tuple[int, str], Path] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_obj in require_list(frame.get("objects"), "objects"):
            obj = require_dict(raw_obj, "object")
            if obj.get("renderable_mask") is not True or not isinstance(obj.get("mask_path"), str):
                continue
            out[(frame_idx, require_str(obj.get("object_id"), "object_id"))] = normalize_path(str(obj["mask_path"]))
    return out


def accepted_part_tracks(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_obj in require_list(report.get("object_rows"), "object rows"):
        obj = require_dict(raw_obj, "object row")
        object_id = require_str(obj.get("object_id"), "object_id")
        for raw_ev in require_list(obj.get("candidate_part_track_evaluations", []), "track evaluations"):
            ev = require_dict(raw_ev, "track evaluation")
            if ev.get("accepted_as_part_evidence") is not True:
                continue
            track_path = require_str(ev.get("track_path"), "track_path")
            track_label = require_str(ev.get("track_label"), "track_label")
            key = (object_id, track_label, track_path)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"object_id": object_id, "track_label": track_label, "track_path": track_path})
    return rows


def build_faces(index_grid: np.ndarray, vertices: np.ndarray, max_edge_m: float) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    h, w = index_grid.shape
    for y in range(h - 1):
        for x in range(w - 1):
            a = int(index_grid[y, x])
            b = int(index_grid[y, x + 1])
            c = int(index_grid[y + 1, x])
            d = int(index_grid[y + 1, x + 1])
            for tri in ((a, b, c), (b, d, c)):
                if min(tri) < 0:
                    continue
                pts = vertices[list(tri)]
                edge = max(
                    float(np.linalg.norm(pts[0] - pts[1])),
                    float(np.linalg.norm(pts[1] - pts[2])),
                    float(np.linalg.norm(pts[2] - pts[0])),
                )
                if edge <= max_edge_m:
                    faces.append(tri)
    return np.asarray(faces, dtype=np.int32)


def remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return vertices[:0], faces[:0]
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def surface_from_part_mask(
    frame_idx: int,
    part_mask_path: Path,
    object_mask_path: Path,
    depth: dict[str, Any],
    mask_cache: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        raise RuntimeError("metric_depth_missing_for_frame")
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    part_mask = resize_bool_mask(load_mask(part_mask_path, mask_cache), depth_m.shape)
    object_mask = resize_bool_mask(load_mask(object_mask_path, mask_cache), depth_m.shape)
    part_area = int(part_mask.sum())
    object_area = int(object_mask.sum())
    intersection = int(np.logical_and(part_mask, object_mask).sum())
    if part_area <= 0 or object_area <= 0:
        raise RuntimeError("empty_part_or_object_mask")
    containment = intersection / part_area
    if containment < float(args.min_frame_part_containment):
        raise RuntimeError("part_mask_not_contained_in_object_mask")
    valid = part_mask & object_mask & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
    values = depth_m[valid]
    if values.size < int(args.min_depth_pixels):
        raise RuntimeError("too_few_valid_masked_depth_pixels")
    lo = float(np.quantile(values, float(args.depth_low_quantile)))
    hi = float(np.quantile(values, float(args.depth_high_quantile)))
    keep = valid & (depth_m >= lo) & (depth_m <= hi)
    keep_ys, keep_xs = np.nonzero(keep)
    if keep_xs.size == 0:
        raise RuntimeError("too_few_valid_masked_depth_pixels")
    y0, y1 = int(keep_ys.min()), int(keep_ys.max())
    x0, x1 = int(keep_xs.min()), int(keep_xs.max())
    base_stride = max(1, int(args.mask_stride))
    stride_candidates: list[int] = []
    stride = base_stride
    while stride > 1:
        stride_candidates.append(stride)
        stride = max(1, stride // 2)
    stride_candidates.append(1)
    if not bool(args.adaptive_small_mask_stride):
        stride_candidates = [base_stride]
    vertices = np.zeros((0, 3), dtype=np.float64)
    faces = np.zeros((0, 3), dtype=np.int32)
    sampled_vertex_count = 0
    stride_used = stride_candidates[-1]
    target_vertices = max(int(args.min_vertices), int(args.target_surface_vertices))
    target_faces = max(int(args.min_faces), int(args.target_surface_faces))
    target_met = False
    last_rejection = "too_few_sampled_vertices"
    fx, fy, cx, cy = depth["intrinsics"][int(depth_i)].astype(float).tolist()
    for stride in stride_candidates:
        ys = np.arange(y0, y1 + 1, stride, dtype=np.int32)
        xs = np.arange(x0, x1 + 1, stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sampled = keep[np.ix_(ys, xs)]
        flat_x = grid_x[sampled].astype(np.float64)
        flat_y = grid_y[sampled].astype(np.float64)
        flat_z = depth_m[np.ix_(ys, xs)][sampled].astype(np.float64)
        sampled_vertex_count = int(len(flat_z))
        stride_used = int(stride)
        if len(flat_z) < int(args.min_vertices):
            last_rejection = "too_few_sampled_vertices"
            continue
        candidate_vertices = np.column_stack(((flat_x - cx) * flat_z / fx, (flat_y - cy) * flat_z / fy, flat_z))
        index_grid = np.full(sampled.shape, -1, dtype=np.int32)
        index_grid[sampled] = np.arange(len(candidate_vertices), dtype=np.int32)
        candidate_faces = build_faces(index_grid, candidate_vertices, float(args.max_triangle_edge_m))
        candidate_vertices, candidate_faces = remove_unreferenced(candidate_vertices, candidate_faces)
        if len(candidate_vertices) >= int(args.min_vertices) and len(candidate_faces) >= int(args.min_faces):
            vertices = candidate_vertices
            faces = candidate_faces
            target_met = len(candidate_vertices) >= target_vertices and len(candidate_faces) >= target_faces
            if target_met:
                break
            last_rejection = "surface_sampling_target_not_met_before_stride_exhausted"
            continue
        if len(candidate_vertices) > len(vertices) or len(candidate_faces) > len(faces):
            vertices = candidate_vertices
            faces = candidate_faces
        last_rejection = "too_few_vertices_or_faces_after_surface_connectivity"
    if len(vertices) < int(args.min_vertices) or len(faces) < int(args.min_faces):
        raise RuntimeError(last_rejection)
    row = {
        "frame_idx": frame_idx,
        "status": "accepted_part_visible_surface",
        "coordinate_frame": "metric_depth_camera",
        "part_mask_path": str(part_mask_path),
        "object_mask_path": str(object_mask_path),
        "part_area_px_depth_grid": part_area,
        "object_area_px_depth_grid": object_area,
        "intersection_px_depth_grid": intersection,
        "part_containment_in_object": float(containment),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "depth_median_m": float(np.median(values)),
        "depth_intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "mask_stride_requested": base_stride,
        "mask_stride_used": stride_used,
        "sampled_vertex_count_before_connectivity": sampled_vertex_count,
        "mask_sampling_target_vertices": target_vertices,
        "mask_sampling_target_faces": target_faces,
        "mask_sampling_target_met": target_met,
        "bbox_camera_min_m": vertices.min(axis=0).astype(float).tolist(),
        "bbox_camera_max_m": vertices.max(axis=0).astype(float).tolist(),
        "extent_camera_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        "geometry_claim": "part_visible_surface_only",
        "hidden_geometry_reconstructed": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }
    return vertices.astype(np.float32), faces.astype(np.int32), row


def save_archive(path: Path, observations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    for obs in observations:
        vertices = obs.pop("_vertices")
        faces = obs.pop("_faces")
        offset = vertex_offsets[-1]
        vertices_parts.append(vertices.astype(np.float32))
        faces_parts.append((faces + offset).astype(np.int32))
        vertex_offsets.append(offset + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    if vertices_parts:
        vertices_all = np.vstack(vertices_parts).astype(np.float32)
        faces_all = np.vstack(faces_parts).astype(np.int32)
    else:
        vertices_all = np.zeros((0, 3), dtype=np.float32)
        faces_all = np.zeros((0, 3), dtype=np.int32)
    metadata = {
        "status": STATUS,
        "claim": CLAIM,
        "archive_format": "v18_part_visible_surface_rows",
        "face_indices": "global_vertex_indices",
        "geometry_claim": "part_visible_surface_only",
        "hidden_geometry_reconstructed": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }
    np.savez_compressed(
        path,
        frame_idx=np.asarray([obs["frame_idx"] for obs in observations], dtype=np.int32),
        object_id=np.asarray([obs["object_id"] for obs in observations]),
        part_track_label=np.asarray([obs["part_track_label"] for obs in observations]),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=vertices_all,
        faces=faces_all,
        v18_archive_metadata_json=json.dumps(metadata),
    )


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    evidence_path = args.part_split_root / case / "v18_part_split_evidence_report.json"
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    depth_path = args.v16_root / case / "unidepth_metric" / "unidepth_metric_depth_v3.npz"
    evidence = require_dict(load_json(evidence_path), f"{case} part split evidence")
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    depth = load_metric_depth(depth_path)
    object_masks = object_mask_index(annotation)
    tracks = accepted_part_tracks(evidence)
    mask_cache: dict[str, np.ndarray] = {}
    observations: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for track in tracks:
        track_payload = require_dict(load_json(Path(track["track_path"])), f"track {track['track_path']}")
        object_id = require_str(track.get("object_id"), "track object_id")
        label = require_str(track.get("track_label"), "track label")
        for key, raw in track_payload.items():
            try:
                frame_idx = int(key)
            except ValueError:
                continue
            row = require_dict(raw, f"track row {key}")
            if row.get("visible") is not True or not isinstance(row.get("mask_path"), str):
                continue
            object_mask_path = object_masks.get((frame_idx, object_id))
            if object_mask_path is None:
                rejected_rows.append({"frame_idx": frame_idx, "object_id": object_id, "part_track_label": label, "reason": "no_object_mask_for_frame"})
                continue
            try:
                vertices, faces, surface_row = surface_from_part_mask(
                    frame_idx,
                    normalize_path(str(row["mask_path"])),
                    object_mask_path,
                    depth,
                    mask_cache,
                    args,
                )
            except RuntimeError as exc:
                rejected_rows.append({"frame_idx": frame_idx, "object_id": object_id, "part_track_label": label, "reason": str(exc)})
                continue
            surface_row.update(
                {
                    "object_id": object_id,
                    "part_track_label": label,
                    "part_track_path": track["track_path"],
                    "part_geometry_complete": False,
                    "part_pose_ready": False,
                    "object_pose_requirement_met": False,
                    "_vertices": vertices,
                    "_faces": faces,
                }
            )
            observations.append(surface_row)
    output_dir = args.output_root / case
    archive_path = output_dir / "v18_part_visible_surfaces_camera.npz"
    save_archive(archive_path, observations)
    surface_rows = [dict(obs) for obs in observations]
    for row in surface_rows:
        row.pop("_vertices", None)
        row.pop("_faces", None)
    reason_counts = Counter(str(row.get("reason")) for row in rejected_rows)
    part_counts = Counter(str(row.get("part_track_label")) for row in surface_rows)
    object_counts = Counter(str(row.get("object_id")) for row in surface_rows)
    total_vertices = sum(require_int(row.get("vertices"), "vertices") for row in surface_rows)
    total_faces = sum(require_int(row.get("faces"), "faces") for row in surface_rows)
    report = {
        "method": "build_v18_part_visible_surfaces",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"part_split_evidence": str(evidence_path), "annotation_state": str(annotation_path), "metric_depth_npz": str(depth_path)},
        "archive_npz": str(archive_path),
        "accepted_part_track_count": len(tracks),
        "surface_frame_rows": len(surface_rows),
        "rejected_candidate_rows": len(rejected_rows),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "surface_rows_by_object": dict(sorted(object_counts.items())),
        "surface_rows_by_part_track": dict(sorted(part_counts.items())),
        "total_vertices": total_vertices,
        "total_faces": total_faces,
        "surface_rows": surface_rows,
        "rejected_rows": rejected_rows,
        "part_visible_surface_archive_ready": len(surface_rows) > 0,
        "part_geometry_completion_ready": False,
        "part_pose_ready": False,
        "hidden_geometry_reconstructed": False,
        "object_pose_requirement_met": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(output_dir / "v18_part_visible_surfaces_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_part_visible_surfaces",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "surface_frame_rows": sum(require_int(report.get("surface_frame_rows"), "surface rows") for report in reports),
        "rejected_candidate_rows": sum(require_int(report.get("rejected_candidate_rows"), "rejected rows") for report in reports),
        "total_vertices": sum(require_int(report.get("total_vertices"), "vertices") for report in reports),
        "total_faces": sum(require_int(report.get("total_faces"), "faces") for report in reports),
        "part_visible_surface_archive_ready": any(bool(report.get("part_visible_surface_archive_ready")) for report in reports),
        "part_geometry_completion_ready": False,
        "part_pose_ready": False,
        "hidden_geometry_reconstructed": False,
        "object_pose_requirement_met": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_visible_surfaces_report.json"),
                "archive_npz": report["archive_npz"],
                "surface_frame_rows": report["surface_frame_rows"],
                "rejected_candidate_rows": report["rejected_candidate_rows"],
                "total_vertices": report["total_vertices"],
                "total_faces": report["total_faces"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_visible_surfaces_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-split-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--mask-stride", type=int, default=8)
    parser.add_argument("--adaptive-small-mask-stride", action="store_true", default=True)
    parser.add_argument("--min-depth-pixels", type=int, default=50)
    parser.add_argument("--min-vertices", type=int, default=8)
    parser.add_argument("--min-faces", type=int, default=6)
    parser.add_argument("--target-surface-vertices", type=int, default=100)
    parser.add_argument("--target-surface-faces", type=int, default=100)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.08)
    parser.add_argument("--min-frame-part-containment", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
