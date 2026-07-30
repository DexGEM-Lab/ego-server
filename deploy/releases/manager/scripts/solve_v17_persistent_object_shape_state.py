#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from build_v17_contact_depth_object_repair import frames_by_index, object_masks_by_frame
from run_v16_full_pipeline import load_metric_depth, mesh_from_mask_depth


@dataclass(frozen=True)
class SurfaceObs:
    frame_idx: int
    vertices_world: np.ndarray
    faces: np.ndarray
    center_world: np.ndarray
    row: dict[str, Any]
    mask_measurement: dict[str, Any]


@dataclass(frozen=True)
class ScaleFilter:
    status: str
    max_extent_threshold_m: float
    q1_extent_m: float
    q3_extent_m: float
    log_iqr: float
    kept_frames: int
    rejected_frames: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def required_float(value: Any, field: str, context: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} field {field} must be a finite number, got {value!r}") from exc
    if not np.isfinite(out):
        raise RuntimeError(f"{context} field {field} must be finite, got {value!r}")
    return out


def compact_intrinsics(frame: dict[str, Any]) -> np.ndarray:
    for hand in frame.get("hands", []):
        raw = hand.get("source_intrinsics")
        if raw is None:
            continue
        intr = np.asarray(raw, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    raw = frame.get("object", {}).get("mesh_qc", {}).get("source_intrinsics")
    if raw is not None:
        intr = np.asarray(raw, dtype=np.float64)
        if intr.shape == (4,) and np.isfinite(intr).all():
            return intr
    raise RuntimeError(f"frame {frame.get('frame_idx')} has no valid source intrinsics")


def mask_center_source(mask_measurement: dict[str, Any], intrinsics: np.ndarray) -> tuple[float, float]:
    centroid = mask_measurement.get("centroid_xy")
    if isinstance(centroid, list) and len(centroid) == 2:
        x = required_float(centroid[0], "centroid_xy[0]", "mask measurement")
        y = required_float(centroid[1], "centroid_xy[1]", "mask measurement")
        return x, y
    bbox = mask_measurement.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        vals = [required_float(v, f"bbox_xyxy[{i}]", "mask measurement") for i, v in enumerate(bbox)]
        return 0.5 * (vals[0] + vals[2]), 0.5 * (vals[1] + vals[3])
    value = mask_measurement.get("value")
    if isinstance(value, dict):
        bbox = value.get("bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            vals = [required_float(v, f"value.bbox_xyxy[{i}]", "mask measurement") for i, v in enumerate(bbox)]
            return 0.5 * (vals[0] + vals[2]), 0.5 * (vals[1] + vals[3])
    return float(intrinsics[2]), float(intrinsics[3])


def object_center_world(frame: dict[str, Any], mask_measurement: dict[str, Any]) -> np.ndarray:
    context = f"frame {frame.get('frame_idx')}"
    depth_m = required_float(frame.get("object", {}).get("depth_m"), "object.depth_m", context)
    intr = compact_intrinsics(frame)
    cx, cy = mask_center_source(mask_measurement, intr)
    fx, fy, px, py = intr.astype(float).tolist()
    center_camera = np.asarray([(cx - px) * depth_m / fx, (cy - py) * depth_m / fy, depth_m], dtype=np.float64)
    T_wc = np.asarray(frame.get("camera", {}).get("T_world_camera_metric"), dtype=np.float64)
    if T_wc.shape != (4, 4) or not np.isfinite(T_wc).all():
        raise RuntimeError(f"{context} has invalid T_world_camera_metric")
    return (T_wc @ np.r_[center_camera, 1.0])[:3]


def frame_with_mask(frame: dict[str, Any], mask_path: Path) -> dict[str, Any]:
    out = dict(frame)
    obj = dict(frame.get("object", {}))
    obj["mask_path"] = str(mask_path)
    out["object"] = obj
    return out


def summarize(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def robust_extent(vertices: np.ndarray, low_percentile: float, high_percentile: float) -> list[float]:
    lo = np.percentile(vertices, float(low_percentile), axis=0)
    hi = np.percentile(vertices, float(high_percentile), axis=0)
    extent = hi - lo
    if not np.isfinite(extent).all():
        raise RuntimeError("canonical robust extent is invalid")
    return extent.astype(float).tolist()


def surface_max_extent(obs: SurfaceObs) -> float:
    extent = obs.vertices_world.max(axis=0) - obs.vertices_world.min(axis=0)
    if not np.isfinite(extent).all() or np.any(extent <= 0.0):
        return float("nan")
    return float(np.max(extent))


def filter_scale_consistent_surfaces(
    observations: list[SurfaceObs],
    args: argparse.Namespace,
) -> tuple[list[SurfaceObs], list[dict[str, Any]], ScaleFilter]:
    if not observations:
        raise RuntimeError("no measured object surfaces to scale-filter")
    extents = np.asarray([surface_max_extent(obs) for obs in observations], dtype=np.float64)
    finite = extents[np.isfinite(extents) & (extents > 0.0)]
    if finite.size < int(args.min_fusion_frames):
        raise RuntimeError(f"only {finite.size} finite surface extents; need {args.min_fusion_frames}")
    log_extents = np.log(finite)
    q1_log, q3_log = np.percentile(log_extents, [25.0, 75.0])
    log_iqr = float(q3_log - q1_log)
    threshold = float(np.exp(q3_log + float(args.surface_extent_log_iqr_multiplier) * log_iqr))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError("surface extent threshold is invalid")
    kept: list[SurfaceObs] = []
    rejected: list[dict[str, Any]] = []
    for obs, max_extent in zip(observations, extents, strict=True):
        if np.isfinite(max_extent) and float(max_extent) <= threshold:
            kept.append(obs)
            continue
        row = dict(obs.row)
        row["status"] = "rejected_temporal_surface_scale_outlier"
        row["reason"] = "surface_extent_outside_object_timeline_distribution"
        row["object_id"] = str(args.object_id)
        row["object_center_world_m"] = obs.center_world.astype(float).tolist()
        row["surface_max_extent_m"] = None if not np.isfinite(max_extent) else float(max_extent)
        row["surface_max_extent_threshold_m"] = threshold
        rejected.append(row)
    return (
        kept,
        rejected,
        ScaleFilter(
            status="applied_log_iqr_surface_extent_filter",
            max_extent_threshold_m=threshold,
            q1_extent_m=float(np.exp(q1_log)),
            q3_extent_m=float(np.exp(q3_log)),
            log_iqr=log_iqr,
            kept_frames=len(kept),
            rejected_frames=len(rejected),
        ),
    )


def save_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def collect_surfaces(
    frames: dict[int, dict[str, Any]],
    masks: dict[int, dict[str, Any]],
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[SurfaceObs], list[dict[str, Any]]]:
    observations: list[SurfaceObs] = []
    rejected: list[dict[str, Any]] = []
    for idx in sorted(masks):
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        frame = frames.get(idx)
        if frame is None:
            rejected.append({"frame_idx": idx, "status": "rejected_missing_annotation_frame"})
            continue
        mask_row = masks[idx]
        mask_path = Path(str(mask_row["mask_path"]))
        try:
            center_world = object_center_world(frame, mask_row)
            vertices, faces, row = mesh_from_mask_depth(
                frame_with_mask(frame, mask_path),
                depth,
                mask_stride=int(args.mask_stride),
                mask_erode_px=int(args.mask_erode_px),
                max_triangle_edge_m=float(args.max_triangle_edge_m),
                min_vertices=int(args.min_vertices),
                min_faces=int(args.min_faces),
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
                depth_low_quantile=float(args.depth_low_quantile),
                depth_high_quantile=float(args.depth_high_quantile),
            )
        except RuntimeError as exc:
            rejected.append(
                {
                    "frame_idx": idx,
                    "status": "rejected_surface_extraction_error",
                    "reason": str(exc),
                    "object_id": str(args.object_id),
                    "mask_path": str(mask_path),
                }
            )
            continue
        row["object_center_world_m"] = center_world.astype(float).tolist()
        row["object_id"] = str(args.object_id)
        if row.get("status") != "measured_mesh_from_mask_metric_depth":
            rejected.append(row)
            continue
        observations.append(
            SurfaceObs(
                frame_idx=idx,
                vertices_world=vertices.astype(np.float64),
                faces=faces.astype(np.int32),
                center_world=center_world,
                row=row,
                mask_measurement=mask_row,
            )
        )
    return observations, rejected


def fuse_canonical_mesh(observations: list[SurfaceObs]) -> tuple[np.ndarray, np.ndarray, dict[int, slice]]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    frame_slices: dict[int, slice] = {}
    offset = 0
    for obs in observations:
        canonical = obs.vertices_world - obs.center_world[None, :]
        vertices.append(canonical)
        faces.append(obs.faces + offset)
        frame_slices[obs.frame_idx] = slice(offset, offset + len(canonical))
        offset += len(canonical)
    if not vertices or not faces:
        raise RuntimeError("no measured object surfaces to fuse")
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0), frame_slices


def frame_residuals(vertices: np.ndarray, frame_slice: slice, frame_count: int, min_other_vertices: int) -> dict[str, Any]:
    current = vertices[frame_slice]
    if len(current) == 0:
        return {"status": "rejected_empty_frame_surface"}
    keep = np.ones(frame_count, dtype=bool)
    keep[frame_slice] = False
    other = vertices[keep]
    if len(other) < int(min_other_vertices):
        return {"status": "rejected_not_enough_other_canonical_vertices", "other_vertices": int(len(other))}
    distances, _ = KDTree(other).query(current, k=1)
    return {"status": "accepted_residual_computed", "surface_to_canonical_m": summarize(distances.astype(np.float64))}


def anchor_rows(
    observations: dict[int, SurfaceObs],
    vertices: np.ndarray,
    frame_slices: dict[int, slice],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in args.anchor_frame:
        obs = observations.get(int(idx))
        if obs is None:
            rows.append(
                {
                    "frame_idx": int(idx),
                    "status": "rejected_missing_surface_measurement",
                    "annotation_ready": False,
                    "failure_reason": "anchor_has_no_measured_surface",
                }
            )
            continue
        residual = frame_residuals(vertices, frame_slices[obs.frame_idx], len(vertices), int(args.min_other_vertices))
        summary = residual.get("surface_to_canonical_m")
        p95 = None if not isinstance(summary, dict) else summary.get("p95")
        ready = isinstance(p95, (int, float)) and float(p95) <= float(args.max_anchor_surface_p95_m)
        rows.append(
            {
                "frame_idx": int(idx),
                "status": "accepted_persistent_visible_surface_support" if ready else "rejected_persistent_visible_surface_support",
                "annotation_ready": bool(ready),
                "object_id": str(args.object_id),
                "covered_entity_ids": [str(args.covered_entity_id)] if args.covered_entity_id else [],
                "surface_vertices": int(len(obs.vertices_world)),
                "surface_faces": int(len(obs.faces)),
                "object_center_world_m": obs.center_world.astype(float).tolist(),
                "pose_model": "translation_only_orientation_unobservable_from_near_round_visible_surface",
                "surface_to_canonical_m": summary,
                "failure_reason": None if ready else "anchor_surface_not_supported_by_other_canonical_observations",
                "source_mask_path": obs.row.get("mask_path"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    frames = frames_by_index(annotations)
    masks = object_masks_by_frame(args.sam2_mask_measurements, str(args.object_id))
    depth = load_metric_depth(args.depth_npz)
    observations, rejected = collect_surfaces(frames, masks, depth, args)
    observations, scale_rejected, scale_filter = filter_scale_consistent_surfaces(observations, args)
    rejected.extend(scale_rejected)
    if len(observations) < int(args.min_fusion_frames):
        raise RuntimeError(f"only {len(observations)} measured surfaces; need {args.min_fusion_frames}")
    vertices, faces, frame_slices = fuse_canonical_mesh(observations)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if not np.isfinite(extent).all() or np.any(extent <= 0.0):
        raise RuntimeError("canonical mesh extent is invalid")
    robust_extent_1_99 = robust_extent(vertices, 1.0, 99.0)
    robust_extent_5_95 = robust_extent(vertices, 5.0, 95.0)
    shape_scale_ready = max(robust_extent_1_99) <= float(scale_filter.max_extent_threshold_m)
    archive = args.output_dir / "canonical_visible_surface_mesh.npz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        frame_idx=np.asarray([obs.frame_idx for obs in observations], dtype=np.int32),
        frame_vertex_start=np.asarray([frame_slices[obs.frame_idx].start for obs in observations], dtype=np.int64),
        frame_vertex_end=np.asarray([frame_slices[obs.frame_idx].stop for obs in observations], dtype=np.int64),
        object_center_world_m=np.asarray([obs.center_world for obs in observations], dtype=np.float32),
    )
    ply = args.output_dir / "canonical_visible_surface_mesh.ply"
    save_ply(ply, vertices, faces)
    obs_by_frame = {obs.frame_idx: obs for obs in observations}
    anchors = anchor_rows(obs_by_frame, vertices, frame_slices, args)
    accepted_anchors = [row for row in anchors if row.get("annotation_ready") is True]
    frame_rows = [
        {
            "frame_idx": int(obs.frame_idx),
            "object_id": str(args.object_id),
            "object_center_world_m": obs.center_world.astype(float).tolist(),
            "surface_vertices": int(len(obs.vertices_world)),
            "surface_faces": int(len(obs.faces)),
            "source_mask_path": obs.row.get("mask_path"),
            "surface_depth_model": obs.row.get("surface_depth_model"),
        }
        for obs in observations
    ]
    report = {
        "status": "accepted"
        if len(accepted_anchors) == len(args.anchor_frame) and shape_scale_ready
        else "rejected_persistent_shape_state",
        "annotation_ready": bool(len(accepted_anchors) == len(args.anchor_frame) and shape_scale_ready),
        "method": "solve_v17_persistent_object_shape_state",
        "claim_tested": (
            "model-produced object masks and metric-depth visible surfaces support one persistent canonical "
            "visible-surface mesh with per-frame object-center pose measurements"
        ),
        "object_id": str(args.object_id),
        "covered_entity_ids": [str(args.covered_entity_id)] if args.covered_entity_id else [],
        "annotations": str(args.annotations),
        "depth_npz": str(args.depth_npz),
        "sam2_mask_measurements": str(args.sam2_mask_measurements),
        "canonical_mesh_npz": str(archive),
        "canonical_mesh_ply": str(ply),
        "visible_mask_frames": int(len(masks)),
        "measured_surface_frames": int(len(observations)),
        "rejected_surface_frames": int(len(rejected)),
        "surface_scale_filter": {
            "status": scale_filter.status,
            "max_extent_threshold_m": scale_filter.max_extent_threshold_m,
            "q1_extent_m": scale_filter.q1_extent_m,
            "q3_extent_m": scale_filter.q3_extent_m,
            "log_iqr": scale_filter.log_iqr,
            "kept_frames": scale_filter.kept_frames,
            "rejected_frames": scale_filter.rejected_frames,
        },
        "canonical_vertices": int(len(vertices)),
        "canonical_faces": int(len(faces)),
        "canonical_extent_m": extent.astype(float).tolist(),
        "canonical_robust_extent_1_99_m": robust_extent_1_99,
        "canonical_robust_extent_5_95_m": robust_extent_5_95,
        "canonical_shape_scale_ready": bool(shape_scale_ready),
        "anchor_rows": anchors,
        "frame_rows": frame_rows,
        "rejected_rows_preview": rejected[:100],
        "thresholds": {
            "min_fusion_frames": int(args.min_fusion_frames),
            "min_other_vertices": int(args.min_other_vertices),
            "max_anchor_surface_p95_m": float(args.max_anchor_surface_p95_m),
            "mask_stride": int(args.mask_stride),
            "mask_erode_px": int(args.mask_erode_px),
            "max_triangle_edge_m": float(args.max_triangle_edge_m),
            "surface_extent_log_iqr_multiplier": float(args.surface_extent_log_iqr_multiplier),
        },
    }
    write_json(args.output_dir / "persistent_object_shape_state.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--sam2-mask-measurements", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--covered-entity-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-frame", type=int, action="append", required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--mask-stride", type=int, default=4)
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.05)
    parser.add_argument("--min-vertices", type=int, default=40)
    parser.add_argument("--min-faces", type=int, default=40)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=4.00)
    parser.add_argument("--depth-low-quantile", type=float, default=0.02)
    parser.add_argument("--depth-high-quantile", type=float, default=0.98)
    parser.add_argument("--min-fusion-frames", type=int, default=30)
    parser.add_argument("--min-other-vertices", type=int, default=1000)
    parser.add_argument("--max-anchor-surface-p95-m", type=float, default=0.025)
    parser.add_argument("--surface-extent-log-iqr-multiplier", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
