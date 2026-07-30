#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from run_v16_full_pipeline import load_metric_depth


STATUS = "multi_object_visible_surface_evidence_not_object_pose"
CLAIM = (
    "This artifact extracts per-object visible RGBD surface evidence for the V17 multi-object mask timeline. "
    "It does not solve canonical object geometry, object pose, deformation, or physical contact."
)


@dataclass(frozen=True)
class SurfaceObs:
    object_id: str
    frame_idx: int
    vertices_world: np.ndarray
    faces: np.ndarray
    row: dict[str, Any]


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
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric") from exc
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{label} must have shape {shape} with finite values")
    return arr


def annotation_frames(path: Path) -> dict[int, dict[str, Any]]:
    payload = require_dict(load_json(path), f"{path}")
    frames = require_list(payload.get("frames"), f"{path}.frames")
    out: dict[int, dict[str, Any]] = {}
    for i, frame in enumerate(frames):
        row = require_dict(frame, f"{path}.frames[{i}]")
        idx = require_int(row.get("frame_idx"), f"{path}.frames[{i}].frame_idx")
        out[idx] = row
    return out


def multi_object_rows(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]], int]:
    payload = require_dict(load_json(path), f"{path}")
    frame_count = require_int(payload.get("frame_count"), f"{path}.frame_count")
    objects_payload = require_list(payload.get("objects"), f"{path}.objects")
    object_meta = {
        require_str(row.get("object_id"), f"{path}.objects[{i}].object_id"): row
        for i, raw in enumerate(objects_payload)
        for row in [require_dict(raw, f"{path}.objects[{i}]")]
    }
    out: dict[int, list[dict[str, Any]]] = {}
    frames = require_list(payload.get("frames"), f"{path}.frames")
    if len(frames) != frame_count:
        raise RuntimeError(f"{path} must contain {frame_count} timeline frames")
    for i, raw in enumerate(frames):
        frame = require_dict(raw, f"{path}.frames[{i}]")
        idx = require_int(frame.get("frame_idx"), f"{path}.frames[{i}].frame_idx")
        objects = require_list(frame.get("objects"), f"{path}.frames[{i}].objects")
        rows: list[dict[str, Any]] = []
        for obj_i, obj_raw in enumerate(objects):
            obj = require_dict(obj_raw, f"{path}.frames[{i}].objects[{obj_i}]")
            if obj.get("visible") is True:
                rows.append(obj)
        out[idx] = rows
    return out, object_meta, frame_count


def build_faces(index_grid: np.ndarray, vertices: np.ndarray, max_edge_m: float) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    h, w = index_grid.shape
    for y in range(h - 1):
        for x in range(w - 1):
            quad = [
                int(index_grid[y, x]),
                int(index_grid[y, x + 1]),
                int(index_grid[y + 1, x]),
                int(index_grid[y + 1, x + 1]),
            ]
            for tri in ((quad[0], quad[1], quad[2]), (quad[1], quad[3], quad[2])):
                if min(tri) < 0:
                    continue
                pts = vertices[list(tri)]
                edges = (
                    float(np.linalg.norm(pts[0] - pts[1])),
                    float(np.linalg.norm(pts[1] - pts[2])),
                    float(np.linalg.norm(pts[2] - pts[0])),
                )
                if max(edges) <= float(max_edge_m):
                    faces.append(tri)
    return np.asarray(faces, dtype=np.int32)


def remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[faces].astype(np.int32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (np.c_[points, np.ones(len(points), dtype=np.float64)] @ matrix.T)[:, :3]


def save_surface_archive(path: Path, observations: list[SurfaceObs]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not observations:
        np.savez_compressed(
            path,
            frame_idx=np.zeros((0,), dtype=np.int32),
            object_id=np.zeros((0,), dtype="<U1"),
            vertex_offsets=np.zeros((1,), dtype=np.int64),
            face_offsets=np.zeros((1,), dtype=np.int64),
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
            v17_archive_metadata_json=json.dumps(
                {
                    "status": STATUS,
                    "claim": CLAIM,
                    "surface_count": 0,
                    "archive_format": "multi_object_visible_surface_rows",
                    "face_indices": "global_vertex_indices",
                    "object_geometry_complete": False,
                    "object_pose_requirement_met": False,
                    "annotation_ready": False,
                    "deliverable_ready": False,
                    "v3_solver_complete": False,
                }
            ),
        )
        return
    vertex_offsets = [0]
    face_offsets = [0]
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for obs in observations:
        if len(obs.vertices_world) == 0 or len(obs.faces) == 0:
            raise RuntimeError("surface archive cannot store empty surface observations")
        offset = vertex_offsets[-1]
        vertices.append(obs.vertices_world.astype(np.float32))
        faces.append((obs.faces + offset).astype(np.int32))
        vertex_offsets.append(offset + int(len(obs.vertices_world)))
        face_offsets.append(face_offsets[-1] + int(len(obs.faces)))
    metadata = {
        "status": STATUS,
        "claim": CLAIM,
        "surface_count": len(observations),
        "archive_format": "multi_object_visible_surface_rows",
        "face_indices": "global_vertex_indices",
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    np.savez_compressed(
        path,
        frame_idx=np.asarray([obs.frame_idx for obs in observations], dtype=np.int32),
        object_id=np.asarray([obs.object_id for obs in observations]),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices).astype(np.float32),
        faces=np.vstack(faces).astype(np.int32),
        v17_archive_metadata_json=json.dumps(metadata),
    )


def reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = require_str(row.get("reason"), "rejected row reason")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def surface_from_mask_depth(
    frame: dict[str, Any],
    obj: dict[str, Any],
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = require_int(frame.get("frame_idx"), "frame_idx")
    depth_i = depth["frame_to_i"].get(idx)
    if depth_i is None:
        raise RuntimeError("metric_depth_missing_for_frame")
    mask_path = Path(require_str(obj.get("mask_path"), f"frame {idx} object mask_path"))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"mask_read_failed:{mask_path}")
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    original_mask_shape = tuple(mask.shape)
    if mask.shape != depth_m.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0
    if int(args.mask_erode_px) > 0:
        k = np.ones((2 * int(args.mask_erode_px) + 1, 2 * int(args.mask_erode_px) + 1), dtype=np.uint8)
        mask_bool = cv2.erode(mask_bool.astype(np.uint8), k, iterations=1) > 0
    valid = mask_bool & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
    depth_values = depth_m[valid]
    if depth_values.size < int(args.min_depth_pixels):
        raise RuntimeError("too_few_valid_masked_depth_pixels")
    lo = float(np.quantile(depth_values, float(args.depth_low_quantile)))
    hi = float(np.quantile(depth_values, float(args.depth_high_quantile)))
    keep = valid & (depth_m >= lo) & (depth_m <= hi)
    ys = np.arange(0, depth_m.shape[0], int(args.mask_stride), dtype=np.int32)
    xs = np.arange(0, depth_m.shape[1], int(args.mask_stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = keep[np.ix_(ys, xs)]
    z = depth_m[np.ix_(ys, xs)][sampled].astype(np.float64)
    if len(z) < int(args.min_vertices):
        raise RuntimeError("too_few_sampled_vertices")

    depth_intr = depth["intrinsics"][int(depth_i)].astype(np.float64)
    dfx, dfy, dcx, dcy = depth_intr.tolist()
    x_depth = grid_x[sampled].astype(np.float64)
    y_depth = grid_y[sampled].astype(np.float64)
    camera_vertices = np.column_stack(((x_depth - dcx) * z / dfx, (y_depth - dcy) * z / dfy, z))
    index_grid = np.full(sampled.shape, -1, dtype=np.int32)
    index_grid[sampled] = np.arange(len(camera_vertices), dtype=np.int32)
    faces = build_faces(index_grid, camera_vertices, float(args.max_triangle_edge_m))
    camera_vertices, faces = remove_unreferenced(camera_vertices, faces)
    if len(camera_vertices) < int(args.min_vertices) or len(faces) < int(args.min_faces):
        raise RuntimeError("too_few_vertices_or_faces_after_surface_connectivity")
    T_wc = finite_array(require_dict(frame.get("camera"), f"frame {idx}.camera").get("T_world_camera_metric"), (4, 4), f"frame {idx}.camera.T_world_camera_metric")
    vertices_world = transform_points(camera_vertices, T_wc)
    extent = vertices_world.max(axis=0) - vertices_world.min(axis=0)
    center_world = np.median(vertices_world, axis=0)
    bbox_min = vertices_world.min(axis=0)
    bbox_max = vertices_world.max(axis=0)
    row = {
        "frame_idx": idx,
        "object_id": require_str(obj.get("object_id"), f"frame {idx} object_id"),
        "status": STATUS,
        "measurement_type": "multi_object_visible_surface_evidence",
        "coordinate_frame": "world_metric",
        "mask_path": str(mask_path),
        "mask_area_px": obj.get("area_px"),
        "bbox_xyxy": obj.get("bbox_xyxy"),
        "center_xy": obj.get("center_xy"),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "depth_median_m": float(np.median(depth_values)),
        "depth_pixel_shape_hw": [int(depth_m.shape[0]), int(depth_m.shape[1])],
        "original_mask_shape_hw": [int(original_mask_shape[0]), int(original_mask_shape[1])],
        "mask_was_resized_to_depth": original_mask_shape != tuple(depth_m.shape),
        "depth_intrinsics_fx_fy_cx_cy": [float(dfx), float(dfy), float(dcx), float(dcy)],
        "vertices": int(len(vertices_world)),
        "faces": int(len(faces)),
        "world_extent_m": extent.astype(float).tolist(),
        "center_world_m": center_world.astype(float).tolist(),
        "bbox_world_min_m": bbox_min.astype(float).tolist(),
        "bbox_world_max_m": bbox_max.astype(float).tolist(),
        "geometry_state": "visible_surface_only_not_canonical_mesh",
        "pose_state": "no_object_pose_variable",
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
    }
    return vertices_world.astype(np.float32), faces.astype(np.int32), row


def summarize_object(rows: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    object_id = require_str(rows[0].get("object_id"), "surface object_id") if rows else require_str(rejected[0].get("object_id"), "rejected object_id")
    extents = np.asarray([row["world_extent_m"] for row in rows], dtype=np.float64) if rows else np.zeros((0, 3), dtype=np.float64)
    return {
        "object_id": object_id,
        "surface_frame_count": len(rows),
        "rejected_frame_count": len(rejected),
        "surface_vertices": int(sum(int(row["vertices"]) for row in rows)),
        "surface_faces": int(sum(int(row["faces"]) for row in rows)),
        "world_extent_m_p50": np.percentile(extents, 50.0, axis=0).astype(float).tolist() if len(extents) else None,
        "world_extent_m_p95": np.percentile(extents, 95.0, axis=0).astype(float).tolist() if len(extents) else None,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
    }


def build_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    case_root = args.output_root / case
    case_root.mkdir(parents=True, exist_ok=True)
    annotations_path = args.graph_root / case / "annotations_v17_full_timeline_graph.json"
    timeline_path = args.multi_object_timeline_root / case / "v17_multi_object_timeline.json"
    depth_npz = args.depth_root / case / "unidepth_metric" / "unidepth_metric_depth_v3.npz"
    frames = annotation_frames(annotations_path)
    timeline, _object_meta, frame_count = multi_object_rows(timeline_path)
    depth = load_metric_depth(depth_npz)
    if len(frames) != frame_count:
        raise RuntimeError(f"{case} annotations and multi-object timeline disagree on frame count")
    observations: list[SurfaceObs] = []
    rejected: list[dict[str, Any]] = []
    for idx in range(frame_count):
        frame = frames[idx]
        for obj in timeline.get(idx, []):
            object_id = require_str(obj.get("object_id"), f"{case} frame {idx} object_id")
            try:
                vertices, faces, row = surface_from_mask_depth(frame, obj, depth, args)
            except RuntimeError as exc:
                rejected.append(
                    {
                        "frame_idx": idx,
                        "object_id": object_id,
                        "status": "rejected_visible_surface_evidence",
                        "reason": str(exc),
                        "mask_path": obj.get("mask_path"),
                        "object_geometry_complete": False,
                        "object_pose_requirement_met": False,
                        "annotation_ready": False,
                    }
                )
                continue
            observations.append(SurfaceObs(object_id=object_id, frame_idx=idx, vertices_world=vertices, faces=faces, row=row))

    archive_path = case_root / "multi_object_visible_surfaces_world.npz"
    save_surface_archive(archive_path, observations)

    surface_rows = [obs.row for obs in observations]
    by_object: dict[str, list[dict[str, Any]]] = {}
    rejected_by_object: dict[str, list[dict[str, Any]]] = {}
    for row in surface_rows:
        by_object.setdefault(require_str(row.get("object_id"), "surface object_id"), []).append(row)
    for row in rejected:
        rejected_by_object.setdefault(require_str(row.get("object_id"), "rejected object_id"), []).append(row)
    object_ids = sorted(set(by_object) | set(rejected_by_object))
    object_summaries = [summarize_object(by_object.get(object_id, []), rejected_by_object.get(object_id, [])) for object_id in object_ids]
    report = {
        "method": "build_v17_multi_object_visible_surfaces",
        "case": case,
        "status": STATUS,
        "claim": CLAIM,
        "source_annotations": str(annotations_path),
        "source_multi_object_timeline": str(timeline_path),
        "metric_depth_npz": str(depth_npz),
        "mesh_archive": str(archive_path),
        "frame_count": frame_count,
        "visible_object_frame_rows": sum(len(rows) for rows in timeline.values()),
        "surface_frame_rows": len(surface_rows),
        "rejected_visible_object_frame_rows": len(rejected),
        "depth_frame_count": int(len(depth["frame_idx"])),
        "object_summaries": object_summaries,
        "surface_rows": surface_rows,
        "rejected_rows": rejected,
        "rejection_reason_counts": reason_counts(rejected),
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        "thresholds": {
            "mask_stride": int(args.mask_stride),
            "mask_erode_px": int(args.mask_erode_px),
            "min_depth_pixels": int(args.min_depth_pixels),
            "min_vertices": int(args.min_vertices),
            "min_faces": int(args.min_faces),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "depth_low_quantile": float(args.depth_low_quantile),
            "depth_high_quantile": float(args.depth_high_quantile),
            "max_triangle_edge_m": float(args.max_triangle_edge_m),
        },
    }
    write_json(case_root / "v17_multi_object_visible_surface_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [build_case(args, case) for case in args.case]
    summary = {
        "method": "build_v17_multi_object_visible_surfaces",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "cases": [
            {
                "case": case["case"],
                "report": str(args.output_root / case["case"] / "v17_multi_object_visible_surface_report.json"),
                "mesh_archive": case["mesh_archive"],
                "visible_object_frame_rows": case["visible_object_frame_rows"],
                "surface_frame_rows": case["surface_frame_rows"],
                "rejected_visible_object_frame_rows": case["rejected_visible_object_frame_rows"],
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
            for case in cases
        ],
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_multi_object_visible_surface_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"))
    parser.add_argument("--multi-object-timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--depth-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"))
    parser.add_argument("--case", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--mask-stride", type=int, default=8)
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.08)
    parser.add_argument("--min-depth-pixels", type=int, default=120)
    parser.add_argument("--min-vertices", type=int, default=24)
    parser.add_argument("--min-faces", type=int, default=20)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=4.00)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
