#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np


STATUS = "v17_hand_metric_depth_state_qc"
CLAIM = (
    "This artifact tests whether the current V17 MANO hand state shares the UniDepth metric depth field. "
    "It compares the front-most projected MANO surface depth with UniDepth at the same pixels, separately "
    "near active object masks and far from active object masks. It is a state-ownership diagnostic, not a solver."
)
FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


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


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: list[float]) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def source_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if payload is not None:
        out["status"] = payload.get("status")
        out["method"] = payload.get("method")
    return out


def annotation_frames(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = require_list(payload.get("frames"), "annotation frames")
    out: dict[int, dict[str, Any]] = {}
    for i, raw in enumerate(frames):
        frame = require_dict(raw, f"annotation frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"annotation frames[{i}].frame_idx")
        if frame_idx in out:
            raise RuntimeError(f"duplicate annotation frame {frame_idx}")
        out[frame_idx] = frame
    return out


def depth_archive(path: Path) -> dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    required = {"frame_idx", "depth", "source_size", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(npz.files))
    if missing:
        raise RuntimeError(f"{path} missing depth archive arrays: {missing}")
    frame_idx = np.asarray(npz["frame_idx"], dtype=np.int32)
    depth = np.asarray(npz["depth"], dtype=np.float32)
    intrinsics = np.asarray(npz["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    source_size = np.asarray(npz["source_size"], dtype=np.int32)
    if depth.ndim != 3:
        raise RuntimeError(f"{path} depth must have shape [frame,height,width]")
    if len(frame_idx) != depth.shape[0] or intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f"{path} depth arrays disagree in frame dimension")
    if source_size.shape != (2,):
        raise RuntimeError(f"{path} source_size must be [width,height]")
    return {
        "path": path,
        "frame_to_i": {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())},
        "depth": depth,
        "intrinsics": intrinsics,
        "source_size": (int(source_size[0]), int(source_size[1])),
    }


def hand_points(hand: dict[str, Any]) -> tuple[str, np.ndarray] | None:
    for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
        if key not in hand:
            continue
        try:
            points = np.asarray(hand.get(key), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            return key, points
    return None


def hand_intrinsics(hand: dict[str, Any]) -> np.ndarray | None:
    try:
        intr = np.asarray(hand.get("source_intrinsics"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if intr.shape != (4,) or not np.all(np.isfinite(intr)) or intr[0] <= 0.0 or intr[1] <= 0.0:
        return None
    return intr


def hand_residual_summary(hand: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    residual = require_dict(hand.get("projection_residual_to_measurement_px"), "hand projection residual")
    median = optional_float(residual.get("median"), "hand residual median")
    p95 = optional_float(residual.get("p95"), "hand residual p95")
    ok = bool(
        median is not None
        and p95 is not None
        and median <= float(args.max_hand_median_px)
        and p95 <= float(args.max_hand_p95_px)
    )
    return {
        "median": median,
        "p95": p95,
        "residual_ok": ok,
        "max_median_px": float(args.max_hand_median_px),
        "max_p95_px": float(args.max_hand_p95_px),
    }


def camera_world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    camera = require_dict(frame.get("camera"), "frame camera")
    try:
        transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frame has invalid T_world_camera_metric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("frame has invalid T_world_camera_metric")
    return (points_world - transform[:3, 3][None, :]) @ transform[:3, :3]


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-6)
    uv[valid, 0] = intrinsics[0] * points_camera[valid, 0] / z[valid] + intrinsics[2]
    uv[valid, 1] = intrinsics[1] * points_camera[valid, 1] / z[valid] + intrinsics[3]
    return uv, valid


def source_size(intrinsics: np.ndarray) -> tuple[float, float]:
    return 2.0 * float(intrinsics[2]), 2.0 * float(intrinsics[3])


def front_surface_depth_samples(
    *,
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
    depth_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    depth_h, depth_w = depth_shape
    uv, valid_z = project(points_camera, intrinsics)
    hand_w, hand_h = source_size(intrinsics)
    scale = np.asarray([depth_w / hand_w, depth_h / hand_h], dtype=np.float64)
    xy = uv * scale[None, :]
    valid = (
        valid_z
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < depth_w)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < depth_h)
    )
    if not np.any(valid):
        return None
    valid_ids = np.flatnonzero(valid)
    x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, depth_w - 1)
    y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, depth_h - 1)
    hand_z = points_camera[valid_ids, 2].astype(np.float64)
    lin = y.astype(np.int64) * int(depth_w) + x.astype(np.int64)
    order = np.lexsort((hand_z, lin))
    sorted_lin = lin[order]
    keep = np.r_[True, sorted_lin[1:] != sorted_lin[:-1]]
    front = order[keep]
    return {
        "x": x[front],
        "y": y[front],
        "hand_z": hand_z[front],
        "source_to_depth_scale_xy": scale,
        "projected_vertex_count": np.asarray([len(valid_ids)], dtype=np.int32),
    }


def mask_cache_key(mask_path: str, depth_shape: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    return (mask_path, depth_shape)


def resized_mask(
    mask_path: str,
    depth_shape: tuple[int, int],
    cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    key = mask_cache_key(mask_path, depth_shape)
    if key in cache:
        return cache[key]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    mask_bool = mask > 0
    depth_h, depth_w = depth_shape
    if mask_bool.shape != depth_shape:
        scale_x = mask_bool.shape[1] / float(depth_w)
        scale_y = mask_bool.shape[0] / float(depth_h)
        source_px_per_depth_px = float((scale_x + scale_y) / 2.0)
        mask_bool = cv2.resize(
            mask_bool.astype(np.uint8),
            (depth_w, depth_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        source_px_per_depth_px = 1.0
    cache[key] = (mask_bool, source_px_per_depth_px)
    return cache[key]


def active_object_distance(
    frame: dict[str, Any],
    depth_shape: tuple[int, int],
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
) -> dict[str, Any]:
    masks: list[np.ndarray] = []
    scales: list[float] = []
    mask_paths: list[str] = []
    for i, raw in enumerate(require_list(frame.get("objects", []), "frame objects")):
        obj = require_dict(raw, f"frame objects[{i}]")
        if obj.get("active") is not True or obj.get("visible") is not True:
            continue
        mask_path = obj.get("mask_path")
        if not isinstance(mask_path, str) or not mask_path:
            continue
        mask, source_px_per_depth_px = resized_mask(mask_path, depth_shape, mask_cache)
        masks.append(mask)
        scales.append(source_px_per_depth_px)
        mask_paths.append(mask_path)
    if not masks:
        return {
            "visible_active_object_mask_count": 0,
            "mask_paths": [],
            "distance_source_px": np.full(depth_shape, np.inf, dtype=np.float32),
        }
    union = np.zeros(depth_shape, dtype=bool)
    for mask in masks:
        union |= mask
    inv = (~union).astype(np.uint8)
    dist_depth_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    source_px_per_depth_px = float(np.median(np.asarray(scales, dtype=np.float64)))
    return {
        "visible_active_object_mask_count": len(masks),
        "mask_paths": mask_paths,
        "distance_source_px": dist_depth_px * source_px_per_depth_px,
    }


def partition_state(
    *,
    label: str,
    hand_z: np.ndarray,
    metric_z: np.ndarray,
    object_distance_px: np.ndarray,
    selected: np.ndarray,
    residual_ok: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_count = int(np.count_nonzero(selected))
    if selected_count < int(args.min_depth_pixels):
        return {
            "sample_partition": label,
            "measured": False,
            "state": "unobserved_hand_metric_depth",
            "selected_pixels": selected_count,
            "valid_depth_pixels": 0,
            "metric_depth_signal_compatible": False,
            "metric_depth_compatible": False,
        }
    h = hand_z[selected]
    z = metric_z[selected]
    d = object_distance_px[selected]
    valid = np.isfinite(z) & (z >= float(args.min_depth_m)) & (z <= float(args.max_depth_m))
    if int(np.count_nonzero(valid)) < int(args.min_depth_pixels):
        return {
            "sample_partition": label,
            "measured": False,
            "state": "unobserved_hand_metric_depth",
            "selected_pixels": selected_count,
            "valid_depth_pixels": int(np.count_nonzero(valid)),
            "metric_depth_signal_compatible": False,
            "metric_depth_compatible": False,
        }
    h = h[valid]
    z = z[valid]
    d = d[valid]
    gap = h - z
    abs_gap = np.abs(gap)
    median_gap = float(np.median(gap))
    p95_abs_gap = float(np.percentile(abs_gap, 95.0))
    signal_compatible = bool(
        abs(median_gap) <= float(args.max_median_abs_depth_gap_m)
        and p95_abs_gap <= float(args.max_p95_abs_depth_gap_m)
    )
    compatible = bool(residual_ok and signal_compatible)
    if compatible:
        state = "metric_depth_compatible"
    elif signal_compatible:
        state = "depth_match_projection_residual_untrusted"
    elif median_gap > float(args.max_median_abs_depth_gap_m):
        state = "hand_behind_metric_depth"
    elif median_gap < -float(args.max_median_abs_depth_gap_m):
        state = "hand_in_front_of_metric_depth"
    else:
        state = "depth_tail_incompatible"
    return {
        "sample_partition": label,
        "measured": True,
        "state": state,
        "selected_pixels": selected_count,
        "valid_depth_pixels": int(len(gap)),
        "metric_depth_signal_compatible": signal_compatible,
        "metric_depth_compatible": compatible,
        "hand_source_depth_m": summarize(h.astype(float).tolist()),
        "unidepth_m": summarize(z.astype(float).tolist()),
        "hand_minus_unidepth_depth_m": summarize(gap.astype(float).tolist()),
        "abs_hand_minus_unidepth_depth_m": summarize(abs_gap.astype(float).tolist()),
        "distance_to_active_object_mask_px": summarize(d.astype(float).tolist()),
    }


def measure_hand_row(
    *,
    case: str,
    frame: dict[str, Any],
    hand: dict[str, Any],
    hand_index: int,
    depth: dict[str, Any],
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
    side = require_str(hand.get("side"), "hand side")
    row_id = f"hand_metric_depth:v17:{frame_idx:06d}:{side}:{hand_index}"
    missing: list[str] = []
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        missing.append("metric_depth_frame")
    points_record = hand_points(hand)
    if points_record is None:
        missing.append("hand_world_points")
    intr = hand_intrinsics(hand)
    if intr is None:
        missing.append("hand_source_intrinsics")
    residual = hand_residual_summary(hand, args)
    if missing:
        return {
            "case": case,
            "hand_metric_depth_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "hand_metric_depth_state": "unobserved_hand_metric_depth",
            "missing_depth_evidence": missing,
            "measurement_available": bool(hand.get("measurement_available") is True),
            "detector_score": hand.get("detector_score"),
            "projection_residual_to_measurement_px": residual,
            "sample_partitions": {},
            "metric_depth_compatible": False,
            **FALSE_READY,
        }
    if points_record is None or intr is None or depth_i is None:
        raise RuntimeError("missing hand-depth evidence branch failed to return")
    points_key, points_world = points_record
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    points_camera = camera_world_to_camera(points_world, frame)
    samples = front_surface_depth_samples(
        points_camera=points_camera,
        intrinsics=intr,
        depth_shape=(depth_m.shape[0], depth_m.shape[1]),
    )
    if samples is None:
        return {
            "case": case,
            "hand_metric_depth_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "hand_metric_depth_state": "unobserved_hand_metric_depth",
            "missing_depth_evidence": ["projected_hand_surface_inside_depth_image"],
            "measurement_available": bool(hand.get("measurement_available") is True),
            "detector_score": hand.get("detector_score"),
            "projection_residual_to_measurement_px": residual,
            "sample_partitions": {},
            "metric_depth_compatible": False,
            **FALSE_READY,
        }
    x = samples["x"].astype(np.int32)
    y = samples["y"].astype(np.int32)
    hand_z = samples["hand_z"].astype(np.float64)
    metric_z = depth_m[y, x].astype(np.float64)
    object_distance = active_object_distance(frame, (depth_m.shape[0], depth_m.shape[1]), mask_cache)
    distance_image = object_distance["distance_source_px"]
    object_distance_px = distance_image[y, x].astype(np.float64)
    finite_distance = np.isfinite(object_distance_px)
    near = finite_distance & (object_distance_px <= float(args.near_object_mask_px))
    far = (~finite_distance) | (object_distance_px >= float(args.far_object_mask_px))
    all_samples = np.ones(len(hand_z), dtype=bool)
    residual_ok = bool(residual.get("residual_ok") is True)
    partitions = {
        "all_projected_hand_pixels": partition_state(
            label="all_projected_hand_pixels",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=all_samples,
            residual_ok=residual_ok,
            args=args,
        ),
        "near_active_object_masks": partition_state(
            label="near_active_object_masks",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=near,
            residual_ok=residual_ok,
            args=args,
        ),
        "far_from_active_object_masks": partition_state(
            label="far_from_active_object_masks",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=far,
            residual_ok=residual_ok,
            args=args,
        ),
    }
    far_partition = partitions["far_from_active_object_masks"]
    all_partition = partitions["all_projected_hand_pixels"]
    owner_partition = far_partition if far_partition["measured"] is True else all_partition
    state = require_str(owner_partition.get("state"), "hand metric depth owner state")
    compatible = bool(owner_partition.get("metric_depth_compatible") is True)
    return {
        "case": case,
        "hand_metric_depth_variable_id": row_id,
        "frame_idx": frame_idx,
        "hand_side": side,
        "hand_index": hand_index,
        "hand_metric_depth_state": state,
        "state_owner_sample_partition": require_str(owner_partition.get("sample_partition"), "sample partition"),
        "missing_depth_evidence": [],
        "measurement_available": bool(hand.get("measurement_available") is True),
        "detector_score": hand.get("detector_score"),
        "projection_residual_to_measurement_px": residual,
        "source_point_field": points_key,
        "input_point_count": int(len(points_world)),
        "front_projected_hand_pixels": int(len(hand_z)),
        "visible_active_object_mask_count": int(object_distance["visible_active_object_mask_count"]),
        "sample_partitions": partitions,
        "metric_depth_compatible": compatible,
        "hand_projection_grid": {
            "hand_source_size_wh": [float(source_size(intr)[0]), float(source_size(intr)[1])],
            "depth_source_size_wh": [int(depth["source_size"][0]), int(depth["source_size"][1])],
            "source_to_depth_scale_xy": samples["source_to_depth_scale_xy"].astype(float).tolist(),
            "hand_source_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
            "depth_intrinsics_fx_fy_cx_cy": depth["intrinsics"][int(depth_i)].astype(float).tolist(),
        },
        **FALSE_READY,
    }


def partition_case_summary(rows: list[dict[str, Any]], partition: str) -> dict[str, Any]:
    states: Counter[str] = Counter()
    med_gap: list[float] = []
    med_abs_gap: list[float] = []
    valid_pixels: list[float] = []
    measured_rows = 0
    signal_compatible = 0
    compatible = 0
    for row in rows:
        partitions = require_dict(row.get("sample_partitions"), "sample_partitions")
        raw_part = partitions.get(partition)
        if raw_part is None:
            states["unobserved_hand_metric_depth"] += 1
            continue
        part = require_dict(raw_part, f"sample_partitions.{partition}")
        states[require_str(part.get("state"), "partition state")] += 1
        if part.get("measured") is True:
            measured_rows += 1
            valid_pixels.append(float(require_int(part.get("valid_depth_pixels"), "valid_depth_pixels")))
            gap_summary = require_dict(part.get("hand_minus_unidepth_depth_m"), "hand_minus_unidepth_depth_m")
            abs_summary = require_dict(part.get("abs_hand_minus_unidepth_depth_m"), "abs_hand_minus_unidepth_depth_m")
            med_gap.append(finite_float(gap_summary.get("median"), "gap median"))
            med_abs_gap.append(finite_float(abs_summary.get("median"), "abs gap median"))
        if part.get("metric_depth_signal_compatible") is True:
            signal_compatible += 1
        if part.get("metric_depth_compatible") is True:
            compatible += 1
    return {
        "partition": partition,
        "hand_rows": len(rows),
        "measured_hand_rows": measured_rows,
        "unobserved_hand_rows": len(rows) - measured_rows,
        "metric_depth_signal_compatible_hand_rows": signal_compatible,
        "metric_depth_compatible_hand_rows": compatible,
        "state_counts": dict(sorted(states.items())),
        "valid_depth_pixels_per_hand": summarize(valid_pixels),
        "hand_minus_unidepth_depth_m": summarize(med_gap),
        "abs_hand_minus_unidepth_depth_m": summarize(med_abs_gap),
    }


def rows_by_side(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        side = require_str(row.get("hand_side"), "hand_side")
        out.setdefault(side, []).append(row)
    return out


def state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get("hand_metric_depth_state"), "hand state") for row in rows).items()))


def partition_measured(row: dict[str, Any], partition: str) -> bool:
    partitions = require_dict(row.get("sample_partitions"), "sample_partitions")
    raw_part = partitions.get(partition)
    if raw_part is None:
        return False
    return require_dict(raw_part, f"sample_partitions.{partition}").get("measured") is True


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "pairwise_contact_depth_gap": existing_path(
            args.pairwise_contact_depth_gap_root / case / "v17_pairwise_contact_depth_gap.json",
            f"{case} pairwise contact depth-gap report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    visible_surface = payloads["visible_surface"]
    pairwise_depth = payloads["pairwise_contact_depth_gap"]
    if frame_count != require_int(visible_surface.get("frame_count"), f"{case} visible-surface frame_count"):
        raise RuntimeError(f"{case} frame count disagrees with visible-surface report")
    if frame_count != require_int(pairwise_depth.get("frame_count"), f"{case} pairwise depth-gap frame_count"):
        raise RuntimeError(f"{case} frame count disagrees with pairwise depth-gap report")
    depth_path = existing_path(Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")), "metric depth npz")
    depth = depth_archive(depth_path)
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    for frame_idx in sorted(frames):
        frame = frames[frame_idx]
        for hand_index, raw_hand in enumerate(require_list(frame.get("hands", []), f"{case} frame hands")):
            hand = require_dict(raw_hand, f"{case} frame {frame_idx} hand {hand_index}")
            rows.append(
                measure_hand_row(
                    case=case,
                    frame=frame,
                    hand=hand,
                    hand_index=hand_index,
                    depth=depth,
                    mask_cache=mask_cache,
                    args=args,
                )
            )
    if not rows:
        raise RuntimeError(f"{case} has no hand rows")
    measured_rows = [row for row in rows if partition_measured(row, "all_projected_hand_pixels")]
    residual_ok_rows = [
        row
        for row in rows
        if require_dict(row.get("projection_residual_to_measurement_px"), "residual").get("residual_ok") is True
    ]
    by_side = rows_by_side(rows)
    partitions = ["all_projected_hand_pixels", "near_active_object_masks", "far_from_active_object_masks"]
    partition_summaries = {
        partition: partition_case_summary(rows, partition)
        for partition in partitions
    }
    report = {
        "method": "build_v17_hand_metric_depth_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(paths[name], payloads[name]) for name in payloads},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "hand_metric_depth_variable_count": len(rows),
        "measured_hand_depth_rows": len(measured_rows),
        "unobserved_hand_depth_rows": len(rows) - len(measured_rows),
        "projection_residual_ok_hand_rows": len(residual_ok_rows),
        "hand_metric_depth_state_counts": state_counts(rows),
        "partition_summaries": partition_summaries,
        "side_summaries": {
            side: {
                "hand_rows": len(side_rows),
                "hand_metric_depth_state_counts": state_counts(side_rows),
                "partition_summaries": {
                    partition: partition_case_summary(side_rows, partition)
                    for partition in partitions
                },
            }
            for side, side_rows in sorted(by_side.items())
        },
        "pairwise_contact_depth_gap_comparison": {
            "evaluated_pair_depth_rows": require_int(
                pairwise_depth.get("evaluated_pair_depth_rows"),
                "evaluated_pair_depth_rows",
            ),
            "measured_pair_depth_rows": require_int(
                pairwise_depth.get("measured_pair_depth_rows"),
                "measured_pair_depth_rows",
            ),
            "metric_depth_compatible_candidate_rows": require_int(
                pairwise_depth.get("metric_depth_compatible_candidate_rows"),
                "metric_depth_compatible_candidate_rows",
            ),
            "depth_gap_state_counts": require_dict(
                pairwise_depth.get("depth_gap_state_counts"),
                "depth_gap_state_counts",
            ),
            "hand_minus_object_depth_m": require_dict(
                pairwise_depth.get("hand_minus_object_depth_m"),
                "hand_minus_object_depth_m",
            ),
        },
        "problem_semantics": {
            "variable": "hand_metric_depth_state[frame_idx, hand_side]",
            "depth_test": "front-most projected MANO surface depth minus UniDepth at the same depth pixel",
            "near_object_partition": "samples within near_object_mask_px of any active visible object mask",
            "far_object_partition": "samples at least far_object_mask_px from every active visible object mask, or frames with no active visible object mask",
            "positive_gap_meaning": "the current MANO surface is behind the visible metric depth at the sampled pixel",
            "solver_implication": "far-object depth incompatibility implicates the hand or camera-depth state before object-contact ownership; near-only incompatibility implicates object/contact occlusion depth ordering.",
        },
        "parameters": {
            "near_object_mask_px": float(args.near_object_mask_px),
            "far_object_mask_px": float(args.far_object_mask_px),
            "min_depth_pixels": int(args.min_depth_pixels),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
            "max_hand_median_px": float(args.max_hand_median_px),
            "max_hand_p95_px": float(args.max_hand_p95_px),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_metric_depth_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.pairwise_contact_depth_gap_root / "v17_pairwise_contact_depth_gap_summary.json",
        "pairwise contact depth-gap summary",
    )
    summary = require_dict(load_json(summary_path), "pairwise contact depth-gap summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "summary case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_metric_depth_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_pairwise_contact_depth_gap_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_metric_depth_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_metric_depth_variable_count": require_int(
                    report.get("hand_metric_depth_variable_count"),
                    "hand metric-depth variable count",
                ),
                "measured_hand_depth_rows": require_int(
                    report.get("measured_hand_depth_rows"),
                    "measured hand-depth rows",
                ),
                "projection_residual_ok_hand_rows": require_int(
                    report.get("projection_residual_ok_hand_rows"),
                    "projection residual ok hand rows",
                ),
                "hand_metric_depth_state_counts": require_dict(
                    report.get("hand_metric_depth_state_counts"),
                    "hand metric-depth state counts",
                ),
                "partition_summaries": require_dict(
                    report.get("partition_summaries"),
                    "partition summaries",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_metric_depth_variable_count": sum(
            require_int(report.get("hand_metric_depth_variable_count"), "hand variable count")
            for report in reports
        ),
        "measured_hand_depth_rows": sum(
            require_int(report.get("measured_hand_depth_rows"), "measured hand-depth rows")
            for report in reports
        ),
        "projection_residual_ok_hand_rows": sum(
            require_int(report.get("projection_residual_ok_hand_rows"), "projection residual ok hand rows")
            for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_metric_depth_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--pairwise-contact-depth-gap-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
