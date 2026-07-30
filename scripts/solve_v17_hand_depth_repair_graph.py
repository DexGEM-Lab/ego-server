#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from build_v17_hand_intrinsics_depth_counterfactual import (
    annotation_hand_index,
    front_surface_depth_samples,
    local_hand_geometry,
    project,
    scale_depth_intrinsics,
    solve_translation,
    source_intrinsics,
    source_size_from_intrinsics,
)
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    active_object_distance,
    annotation_frames,
    depth_archive,
    load_json,
    partition_state,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from build_v17_hand_scale_depth_counterfactual import (
    existing_path,
    optional_finite,
    partition_owner,
    selected_valid_mask,
    source_summary,
)


STATUS = "v17_hand_depth_repair_graph_qc"
CLAIM = (
    "This artifact promotes the V3 hand-depth repair mechanism into V17 full-timeline evidence: "
    "a bounded case-global hand scale and per-row camera-ray depth shifts are fitted against UniDepth "
    "front-surface samples, with temporal smoothness and exact post-solve reprojection/depth evaluation. "
    "It is a hand-depth repair graph diagnostic; local MANO articulation, hand-surface deformation, "
    "depth-observation switches, object geometry, and contact ownership remain fixed or unimplemented."
)


def projection_residual_for_points(
    points_camera: np.ndarray,
    keypoints2d: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    projected, valid = project(points_camera, intrinsics)
    if not np.all(valid):
        return {"median": None, "p95": None, "residual_ok": False}
    residual = np.linalg.norm(projected - keypoints2d, axis=1)
    median = float(np.median(residual))
    p95 = float(np.percentile(residual, 95.0))
    return {
        "median": median,
        "p95": p95,
        "residual_ok": bool(median <= float(args.max_hand_median_px) and p95 <= float(args.max_hand_p95_px)),
        "max_median_px": float(args.max_hand_median_px),
        "max_p95_px": float(args.max_hand_p95_px),
    }


def mode_state(prefix: str, measured: bool, residual_ok: bool, compatible: bool) -> str:
    if compatible:
        return f"metric_depth_compatible_under_{prefix}"
    if measured and residual_ok:
        return f"depth_repair_candidate_under_{prefix}"
    if measured:
        return f"metric_depth_measured_projection_untrusted_under_{prefix}"
    return f"unobserved_under_{prefix}"


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        value = optional_finite(value)
        if value is not None:
            values.append(value)
    return summarize(values)


def selected_sample_mask(base: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    hand_z = np.asarray(base["hand_z"], dtype=np.float64)
    metric_z = np.asarray(base["metric_z"], dtype=np.float64)
    owner = require_str(base.get("base_owner_sample_partition"), "base owner sample partition")
    if owner == "far_from_active_object_masks":
        selected = np.asarray(base["far"], dtype=bool)
    else:
        selected = np.ones(len(hand_z), dtype=bool)
    return selected_valid_mask(hand_z=hand_z, metric_z=metric_z, selected=selected, args=args)


def sampled_depth_pairs(base: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    valid = selected_sample_mask(base, args)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    max_n = int(args.max_depth_samples_per_row)
    if ids.size > max_n:
        pick = np.linspace(0, ids.size - 1, max_n).round().astype(np.int64)
        ids = ids[pick]
    return (
        np.asarray(base["hand_z"], dtype=np.float64)[ids],
        np.asarray(base["metric_z"], dtype=np.float64)[ids],
    )


def projection_shift_factors(base: dict[str, Any], args: argparse.Namespace) -> tuple[list[tuple[float, float]], int]:
    joints = np.asarray(base["source_joints"], dtype=np.float64)
    target = np.asarray(base["keypoints2d"], dtype=np.float64)
    intrinsics = np.asarray(base["intrinsics"], dtype=np.float64)
    ray = np.asarray(base["center_ray"], dtype=np.float64)
    projected, valid = project(joints, intrinsics)
    if projected.shape != target.shape:
        raise RuntimeError("projection target shape mismatch")
    fx, fy = float(intrinsics[0]), float(intrinsics[1])
    factors: list[tuple[float, float]] = []
    for point, uv0, uv_target, ok in zip(joints, projected, target, valid):
        if not bool(ok) or not np.isfinite(point).all() or not np.isfinite(uv_target).all():
            continue
        z = float(point[2])
        if z <= 1e-6:
            continue
        du = fx * (float(ray[0]) * z - float(point[0]) * float(ray[2])) / (z * z)
        dv = fy * (float(ray[1]) * z - float(point[1]) * float(ray[2])) / (z * z)
        if math.isfinite(du):
            factors.append((du / float(args.sigma_keypoint_px), (float(uv_target[0]) - float(uv0[0])) / float(args.sigma_keypoint_px)))
        if math.isfinite(dv):
            factors.append((dv / float(args.sigma_keypoint_px), (float(uv_target[1]) - float(uv0[1])) / float(args.sigma_keypoint_px)))
    return factors, len(factors)


def build_base_row(
    *,
    case: str,
    frame: dict[str, Any],
    metric_row: dict[str, Any],
    hand: dict[str, Any] | None,
    depth: dict[str, Any],
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
    side = require_str(metric_row.get("hand_side"), "metric row hand_side")
    hand_index = require_int(metric_row.get("hand_index"), "metric row hand_index")
    row_id = require_str(metric_row.get("hand_metric_depth_variable_id"), "hand_metric_depth_variable_id")
    depth_i = depth["frame_to_i"].get(frame_idx)
    missing: list[str] = []
    if hand is None:
        missing.append("annotation_hand")
    if depth_i is None:
        missing.append("metric_depth_frame")
    hand_intr = source_intrinsics(hand) if hand is not None else None
    if hand_intr is None:
        missing.append("hand_source_intrinsics")
    geometry = local_hand_geometry(hand) if hand is not None else None
    if geometry is None:
        missing.append("local_hand_geometry_or_2d_keypoints")
    if missing:
        return {
            "case": case,
            "hand_depth_repair_graph_variable_id": row_id.replace(
                "hand_metric_depth:",
                "hand_depth_repair_graph:",
                1,
            ),
            "source_hand_metric_depth_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "base_available": False,
            "missing_graph_inputs": missing,
            "row_scale_candidate": {"available": False, "scale": None, "valid_depth_pixels": 0},
            **FALSE_READY,
        }
    if hand is None or hand_intr is None or geometry is None or depth_i is None:
        raise RuntimeError("missing hand-depth graph inputs branch failed to return")
    local_joints, local_vertices, keypoints2d = geometry
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    depth_intr = depth["intrinsics"][int(depth_i)].astype(np.float64)
    projection_source_size = source_size_from_intrinsics(hand_intr)
    candidate_intrinsics = scale_depth_intrinsics(depth_intr, depth["source_size"], projection_source_size)
    translation = solve_translation(local_joints, keypoints2d, candidate_intrinsics)
    source_joints = local_joints + translation[None, :]
    source_vertices = local_vertices + translation[None, :]
    center = np.median(source_joints, axis=0)
    if not np.all(np.isfinite(center)) or float(center[2]) <= 1e-6:
        return {
            "case": case,
            "hand_depth_repair_graph_variable_id": row_id.replace(
                "hand_metric_depth:",
                "hand_depth_repair_graph:",
                1,
            ),
            "source_hand_metric_depth_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "base_available": False,
            "missing_graph_inputs": ["positive_source_hand_center_depth"],
            "row_scale_candidate": {"available": False, "scale": None, "valid_depth_pixels": 0},
            **FALSE_READY,
        }
    center_ray = center / float(center[2])
    projection_residual = projection_residual_for_points(source_joints, keypoints2d, candidate_intrinsics, args)
    samples = front_surface_depth_samples(
        source_vertices,
        candidate_intrinsics,
        projection_source_size,
        (depth_m.shape[0], depth_m.shape[1]),
    )
    if samples is None:
        return {
            "case": case,
            "hand_depth_repair_graph_variable_id": row_id.replace(
                "hand_metric_depth:",
                "hand_depth_repair_graph:",
                1,
            ),
            "source_hand_metric_depth_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "base_available": False,
            "missing_graph_inputs": ["projected_hand_surface_inside_depth_image"],
            "projection_residual_to_measurement_px": projection_residual,
            "row_scale_candidate": {"available": False, "scale": None, "valid_depth_pixels": 0},
            **FALSE_READY,
        }
    x = samples["x"].astype(np.int32)
    y = samples["y"].astype(np.int32)
    hand_z = samples["hand_z"].astype(np.float64)
    metric_z = depth_m[y, x].astype(np.float64)
    object_distance = active_object_distance(frame, (depth_m.shape[0], depth_m.shape[1]), mask_cache)
    object_distance_px = object_distance["distance_source_px"][y, x].astype(np.float64)
    finite_distance = np.isfinite(object_distance_px)
    near = finite_distance & (object_distance_px <= float(args.near_object_mask_px))
    far = (~finite_distance) | (object_distance_px >= float(args.far_object_mask_px))
    residual_ok = bool(projection_residual.get("residual_ok") is True)
    base_partitions = {
        "all_projected_hand_pixels": partition_state(
            label="all_projected_hand_pixels",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=np.ones(len(hand_z), dtype=bool),
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
    owner = partition_owner(base_partitions)
    owner_label = require_str(owner.get("sample_partition"), "base owner partition")
    selected = far if owner_label == "far_from_active_object_masks" else np.ones(len(hand_z), dtype=bool)
    valid = selected_valid_mask(hand_z=hand_z, metric_z=metric_z, selected=selected, args=args)
    ratio = metric_z[valid] / hand_z[valid]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if len(ratio) < int(args.min_depth_pixels):
        row_scale = {"available": False, "scale": None, "valid_depth_pixels": int(len(ratio))}
    else:
        row_scale = {
            "available": True,
            "scale": float(np.median(ratio)),
            "valid_depth_pixels": int(len(ratio)),
            "sample_ratio_summary": summarize(ratio.astype(float).tolist()),
        }
    return {
        "case": case,
        "hand_depth_repair_graph_variable_id": row_id.replace(
            "hand_metric_depth:",
            "hand_depth_repair_graph:",
            1,
        ),
        "source_hand_metric_depth_variable_id": row_id,
        "frame_idx": frame_idx,
        "hand_side": side,
        "hand_index": hand_index,
        "base_available": True,
        "projection_residual_to_measurement_px": projection_residual,
        "residual_ok": residual_ok,
        "source_joints": source_joints,
        "source_vertices": source_vertices,
        "keypoints2d": keypoints2d,
        "intrinsics": candidate_intrinsics,
        "projection_source_size": projection_source_size,
        "depth_m": depth_m,
        "frame": frame,
        "center_ray": center_ray.astype(np.float64),
        "hand_z": hand_z,
        "metric_z": metric_z,
        "object_distance_px": object_distance_px,
        "near": near,
        "far": far,
        "base_owner_sample_partition": owner_label,
        "base_owner_depth_state": require_str(owner.get("state"), "base owner state"),
        "base_metric_depth_compatible": bool(owner.get("metric_depth_compatible") is True),
        "wrist_to_middle_tip_m": float(np.linalg.norm(source_joints[12] - source_joints[0])),
        "row_scale_candidate": row_scale,
        **FALSE_READY,
    }


def add_row(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    rhs: list[float],
    row_i: int,
    terms: list[tuple[int, float]],
    target: float,
) -> int:
    for col, val in terms:
        rows.append(row_i)
        cols.append(col)
        vals.append(float(val))
    rhs.append(float(target))
    return row_i + 1


def scale_bounds(base_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    wrists = [
        float(row["wrist_to_middle_tip_m"])
        for row in base_rows
        if row.get("base_available") is True and optional_finite(row.get("wrist_to_middle_tip_m")) is not None
    ]
    if not wrists:
        raise RuntimeError("hand-depth repair graph has no finite wrist-to-middle-tip lengths")
    median_wrist = float(np.median(np.asarray(wrists, dtype=np.float64)))
    if median_wrist <= 0.0:
        raise RuntimeError("median wrist-to-middle-tip length must be positive")
    lower = max(float(args.min_hand_scale), float(args.min_scaled_wrist_to_middle_tip_m) / median_wrist)
    upper = min(float(args.max_hand_scale), float(args.max_scaled_wrist_to_middle_tip_m) / median_wrist)
    if not lower < upper:
        raise RuntimeError(f"inconsistent hand scale bounds: lower={lower} upper={upper}")
    return lower, upper, {
        "median_current_wrist_to_middle_tip_m": median_wrist,
        "min_hand_scale": float(args.min_hand_scale),
        "max_hand_scale": float(args.max_hand_scale),
        "min_scaled_wrist_to_middle_tip_m": float(args.min_scaled_wrist_to_middle_tip_m),
        "max_scaled_wrist_to_middle_tip_m": float(args.max_scaled_wrist_to_middle_tip_m),
        "lower": lower,
        "upper": upper,
    }


def build_linear_system(
    base_rows: list[dict[str, Any]],
    var_rows: list[int],
    var_by_row: dict[int, int],
    scale_lower: float,
    scale_upper: float,
    args: argparse.Namespace,
) -> tuple[sparse.csr_matrix, np.ndarray, tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row_i = 0
    depth_factor_rows = 0
    depth_sample_factors = 0
    shift_prior_rows = 0
    smoothness_rows = 0
    scale_prior_rows = 0
    projection_factor_rows = 0
    projection_scalar_factors = 0
    min_depth_bound_rows = 0
    sigma_depth = float(args.sigma_metric_depth_m)
    sigma_shift = float(args.sigma_hand_ray_shift_prior_m)
    sigma_step = float(args.sigma_hand_ray_shift_step_m)
    sigma_scale = float(args.sigma_hand_scale_prior)
    scale_col = 0
    lower = np.r_[scale_lower, np.full(len(var_rows), -float(args.max_abs_hand_ray_shift_m))]
    upper = np.r_[scale_upper, np.full(len(var_rows), float(args.max_abs_hand_ray_shift_m))]
    row_i = add_row(rows, cols, vals, rhs, row_i, [(scale_col, 1.0 / sigma_scale)], 1.0 / sigma_scale)
    scale_prior_rows += 1
    for source_i in var_rows:
        shift_col = 1 + var_by_row[source_i]
        row_i = add_row(rows, cols, vals, rhs, row_i, [(shift_col, 1.0 / sigma_shift)], 0.0)
        shift_prior_rows += 1
        base = base_rows[source_i]
        ray_z = float(np.asarray(base["center_ray"], dtype=np.float64)[2])
        source_depth_min = float(
            min(
                np.min(np.asarray(base["source_joints"], dtype=np.float64)[:, 2]),
                np.min(np.asarray(base["source_vertices"], dtype=np.float64)[:, 2]),
            )
        )
        if ray_z <= 1e-6 or not math.isfinite(source_depth_min):
            raise RuntimeError("invalid source depth for positivity bound")
        positivity_scale = scale_lower if source_depth_min >= 0.0 else scale_upper
        min_shift = (float(args.min_corrected_hand_depth_m) - positivity_scale * source_depth_min) / ray_z
        lower[1 + var_by_row[source_i]] = max(lower[1 + var_by_row[source_i]], min_shift)
        min_depth_bound_rows += 1
        if base.get("residual_ok") is not True:
            continue
        projection_factors, projection_count = projection_shift_factors(base, args)
        if projection_count > 0:
            projection_factor_rows += 1
        projection_scalar_factors += projection_count
        for coefficient, target in projection_factors:
            row_i = add_row(rows, cols, vals, rhs, row_i, [(shift_col, coefficient)], target)
        hand_z, metric_z = sampled_depth_pairs(base, args)
        if len(hand_z) < int(args.min_depth_pixels):
            continue
        depth_factor_rows += 1
        for h, z in zip(hand_z, metric_z):
            row_i = add_row(
                rows,
                cols,
                vals,
                rhs,
                row_i,
                [(scale_col, float(h) / sigma_depth), (shift_col, 1.0 / sigma_depth)],
                float(z) / sigma_depth,
            )
            depth_sample_factors += 1
    by_side: dict[str, list[int]] = {}
    for source_i in var_rows:
        side = require_str(base_rows[source_i].get("hand_side"), "hand_side")
        by_side.setdefault(side, []).append(source_i)
    max_gap = int(args.max_temporal_smooth_gap_frames)
    for side_rows in by_side.values():
        ordered = sorted(side_rows, key=lambda i: require_int(base_rows[i].get("frame_idx"), "frame_idx"))
        for a, b in zip(ordered[:-1], ordered[1:]):
            frame_a = require_int(base_rows[a].get("frame_idx"), "frame_idx")
            frame_b = require_int(base_rows[b].get("frame_idx"), "frame_idx")
            dt = max(1, frame_b - frame_a)
            if dt > max_gap:
                continue
            weight = 1.0 / (sigma_step * float(dt))
            row_i = add_row(
                rows,
                cols,
                vals,
                rhs,
                row_i,
                [(1 + var_by_row[b], weight), (1 + var_by_row[a], -weight)],
                0.0,
            )
            smoothness_rows += 1
    width = 1 + len(var_rows)
    if depth_sample_factors == 0:
        raise RuntimeError("hand-depth repair graph has no metric-depth sample factors")
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(row_i, width))
    if np.any(lower > upper):
        bad = np.flatnonzero(lower > upper)
        raise RuntimeError(f"inconsistent hand-depth repair graph bounds at columns {bad[:12].tolist()}")
    summary = {
        "variable_count": width,
        "hand_depth_repair_variable_count": len(var_rows),
        "scale_variable_count": 1,
        "depth_factor_rows": depth_factor_rows,
        "depth_sample_factors": depth_sample_factors,
        "projection_factor_rows": projection_factor_rows,
        "projection_scalar_factors": projection_scalar_factors,
        "min_depth_bound_rows": min_depth_bound_rows,
        "shift_prior_rows": shift_prior_rows,
        "smoothness_rows": smoothness_rows,
        "scale_prior_rows": scale_prior_rows,
        "matrix_rows": row_i,
        "matrix_cols": width,
    }
    return matrix, np.asarray(rhs, dtype=np.float64), (lower, upper), summary


def corrected_points(base: dict[str, Any], scale: float, shift: float, key: str) -> np.ndarray:
    points = np.asarray(base[key], dtype=np.float64)
    ray = np.asarray(base["center_ray"], dtype=np.float64)
    out = float(scale) * points + float(shift) * ray[None, :]
    if np.any(~np.isfinite(out)) or np.any(out[:, 2] <= 1e-6):
        raise RuntimeError("corrected hand points have nonpositive or nonfinite depth")
    return out


def evaluate_row(
    base: dict[str, Any],
    scale: float | None,
    shift: float | None,
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row_id = require_str(base.get("hand_depth_repair_graph_variable_id"), "hand_depth_repair_graph_variable_id")
    common = {
        "case": require_str(base.get("case"), "case"),
        "hand_depth_repair_graph_variable_id": row_id,
        "source_hand_metric_depth_variable_id": require_str(
            base.get("source_hand_metric_depth_variable_id"),
            "source hand metric depth variable id",
        ),
        "frame_idx": require_int(base.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(base.get("hand_side"), "hand_side"),
        "hand_index": require_int(base.get("hand_index"), "hand_index"),
        "base_available": bool(base.get("base_available") is True),
        "solved_scale": scale,
        "hand_ray_shift_m": shift,
        "row_scale_candidate": base.get("row_scale_candidate"),
        **FALSE_READY,
    }
    if base.get("base_available") is not True or scale is None or shift is None:
        return {
            **common,
            "solver_state": "unobserved_under_hand_depth_repair_graph",
            "owner_depth_state": "unobserved_hand_metric_depth",
            "owner_sample_partition": None,
            "metric_depth_compatible": False,
            "depth_repair_factor_candidate": False,
            "projection_residual_to_measurement_px": base.get("projection_residual_to_measurement_px"),
            "missing_graph_inputs": base.get("missing_graph_inputs", []),
        }
    joints = corrected_points(base, scale, shift, "source_joints")
    vertices = corrected_points(base, scale, shift, "source_vertices")
    intrinsics = np.asarray(base["intrinsics"], dtype=np.float64)
    keypoints2d = np.asarray(base["keypoints2d"], dtype=np.float64)
    projection = projection_residual_for_points(joints, keypoints2d, intrinsics, args)
    residual_ok = bool(projection.get("residual_ok") is True)
    depth_m = np.asarray(base["depth_m"], dtype=np.float64)
    samples = front_surface_depth_samples(
        vertices,
        intrinsics,
        tuple(base["projection_source_size"]),
        (depth_m.shape[0], depth_m.shape[1]),
    )
    if samples is None:
        return {
            **common,
            "solver_state": "unobserved_under_hand_depth_repair_graph",
            "owner_depth_state": "unobserved_hand_metric_depth",
            "owner_sample_partition": None,
            "metric_depth_compatible": False,
            "depth_repair_factor_candidate": False,
            "projection_residual_to_measurement_px": projection,
            "missing_graph_inputs": ["corrected_projected_hand_surface_inside_depth_image"],
            "scaled_wrist_to_middle_tip_m": float(scale) * float(base["wrist_to_middle_tip_m"]),
        }
    x = samples["x"].astype(np.int32)
    y = samples["y"].astype(np.int32)
    hand_z = samples["hand_z"].astype(np.float64)
    metric_z = depth_m[y, x].astype(np.float64)
    frame = require_dict(base.get("frame"), "annotation frame")
    object_distance = active_object_distance(frame, (depth_m.shape[0], depth_m.shape[1]), mask_cache)
    object_distance_px = object_distance["distance_source_px"][y, x].astype(np.float64)
    finite_distance = np.isfinite(object_distance_px)
    near = finite_distance & (object_distance_px <= float(args.near_object_mask_px))
    far = (~finite_distance) | (object_distance_px >= float(args.far_object_mask_px))
    partitions = {
        "all_projected_hand_pixels": partition_state(
            label="all_projected_hand_pixels",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=np.ones(len(hand_z), dtype=bool),
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
    owner = partition_owner(partitions)
    measured = bool(owner.get("measured") is True)
    compatible = bool(owner.get("metric_depth_compatible") is True)
    gap_summary = owner.get("hand_minus_unidepth_depth_m")
    owner_gap = None
    if isinstance(gap_summary, dict):
        owner_gap = optional_finite(gap_summary.get("median"))
    return {
        **common,
        "solver_state": mode_state("hand_depth_repair_graph", measured, residual_ok, compatible),
        "owner_depth_state": require_str(owner.get("state"), "owner depth state"),
        "owner_sample_partition": require_str(owner.get("sample_partition"), "owner sample partition"),
        "metric_depth_compatible": compatible,
        "depth_repair_factor_candidate": bool(measured and residual_ok and not compatible),
        "projection_residual_to_measurement_px": projection,
        "owner_median_gap_m": owner_gap,
        "scaled_wrist_to_middle_tip_m": float(scale) * float(base["wrist_to_middle_tip_m"]),
        "x": x.astype(int).tolist(),
        "y": y.astype(int).tolist(),
        "hand_z": hand_z.astype(float).tolist(),
        "metric_z": metric_z.astype(float).tolist(),
        "object_distance_px": [
            None if not math.isfinite(float(value)) else float(value) for value in object_distance_px
        ],
        "near": near.astype(bool).tolist(),
        "far": far.astype(bool).tolist(),
        "depth_shape": [int(depth_m.shape[0]), int(depth_m.shape[1])],
        "projection_source_size": [float(v) for v in tuple(base["projection_source_size"])],
        "corrected_hand_depth_m": summarize(hand_z.astype(float).tolist()),
        "corrected_unidepth_m": summarize(metric_z.astype(float).tolist()),
        "partitions": partitions,
    }


def solve_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
        "hand_scale_depth_counterfactual": existing_path(
            args.hand_scale_depth_counterfactual_root / case / "v17_hand_scale_depth_counterfactual.json",
            f"{case} hand scale-depth counterfactual report",
        ),
        "hand_tail_depth_observation_state": existing_path(
            args.hand_tail_depth_observation_state_root / case / "v17_hand_tail_depth_observation_state.json",
            f"{case} hand tail depth-observation state report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    scale_cf = payloads["hand_scale_depth_counterfactual"]
    tail_depth = payloads["hand_tail_depth_observation_state"]
    frame_count = len(frames)
    for name, payload in [
        ("visible-surface", visible),
        ("hand metric-depth", hand_metric),
        ("hand scale-depth", scale_cf),
        ("hand tail depth-observation", tail_depth),
    ]:
        if frame_count != require_int(payload.get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} graph annotations disagree with {name} report")
    if require_int(scale_cf.get("hand_scale_counterfactual_variable_count"), f"{case} scale variable count") != require_int(
        hand_metric.get("hand_metric_depth_variable_count"),
        f"{case} hand metric variable count",
    ):
        raise RuntimeError(f"{case} scale counterfactual disagrees with hand metric-depth variable count")
    depth = depth_archive(existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive"))
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    base_rows: list[dict[str, Any]] = []
    for raw in require_list(hand_metric.get("rows"), f"{case} hand metric rows"):
        metric_row = require_dict(raw, "hand metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
        side = require_str(metric_row.get("hand_side"), "metric row hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "metric row hand_index")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} missing annotation frame {frame_idx}")
        base_rows.append(
            build_base_row(
                case=case,
                frame=frame,
                metric_row=metric_row,
                hand=hand_index.get((frame_idx, side, hand_i)),
                depth=depth,
                mask_cache=mask_cache,
                args=args,
            )
        )
    if len(base_rows) != require_int(hand_metric.get("hand_metric_depth_variable_count"), f"{case} hand variable count"):
        raise RuntimeError(f"{case} hand-depth repair graph rows disagree with hand metric-depth report")
    var_rows = [i for i, row in enumerate(base_rows) if row.get("base_available") is True]
    if not var_rows:
        raise RuntimeError(f"{case} has no hand-depth repair graph variables")
    var_by_row = {source_i: var_i for var_i, source_i in enumerate(var_rows)}
    scale_lower, scale_upper, scale_bound_summary = scale_bounds(base_rows, args)
    matrix, rhs, bounds, system_summary = build_linear_system(base_rows, var_rows, var_by_row, scale_lower, scale_upper, args)
    result = lsq_linear(
        matrix,
        rhs,
        bounds=bounds,
        method="trf",
        tol=float(args.solver_tol),
        lsmr_tol="auto",
        max_iter=int(args.max_solver_iter),
        verbose=0,
    )
    if result.x is None or len(result.x) != system_summary["variable_count"]:
        raise RuntimeError(f"{case} hand-depth repair graph returned invalid solution")
    scale = float(result.x[0])
    shifts = np.asarray(result.x[1:], dtype=np.float64)
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    for source_i, base in enumerate(base_rows):
        var_i = var_by_row.get(source_i)
        row_shift = None if var_i is None else float(shifts[var_i])
        rows.append(evaluate_row(base, scale if var_i is not None else None, row_shift, eval_mask_cache, args))
    shift_abs = np.abs(shifts)
    lower_hits = int(np.count_nonzero(np.isclose(shifts, -float(args.max_abs_hand_ray_shift_m), atol=float(args.bound_tolerance_m))))
    upper_hits = int(np.count_nonzero(np.isclose(shifts, float(args.max_abs_hand_ray_shift_m), atol=float(args.bound_tolerance_m))))
    scale_hit = bool(
        math.isclose(scale, scale_lower, abs_tol=float(args.bound_tolerance_scale))
        or math.isclose(scale, scale_upper, abs_tol=float(args.bound_tolerance_scale))
    )
    data_rows = [row for row in base_rows if row.get("base_available") is True and row.get("residual_ok") is True]
    report = {
        "method": "solve_v17_hand_depth_repair_graph",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_depth_repair_graph_variable_count": len(base_rows),
        "base_available_rows": len(var_rows),
        "depth_data_candidate_rows": len(data_rows),
        "case_global_scale": scale,
        "case_global_scale_bounds": scale_bound_summary,
        "case_global_scale_bound_hit": scale_hit,
        "case_global_scaled_wrist_to_middle_tip_m": numeric_summary(rows, "scaled_wrist_to_middle_tip_m"),
        "hand_ray_shift_m": summarize(shifts.astype(float).tolist()),
        "hand_ray_shift_abs_m": summarize(shift_abs.astype(float).tolist()),
        "hand_ray_shift_lower_bound_hit_rows": lower_hits,
        "hand_ray_shift_upper_bound_hit_rows": upper_hits,
        "hand_ray_shift_bound_hit_rows": lower_hits + upper_hits,
        "system": system_summary,
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "active_mask_counts": dict(sorted(Counter(int(x) for x in result.active_mask).items())),
        },
        "solver_state_counts": state_counts(rows, "solver_state"),
        "owner_depth_state_counts": state_counts(rows, "owner_depth_state"),
        "metric_hand_state_accepted_rows": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows": bool_count(rows, "depth_repair_factor_candidate"),
        "projection_residual_to_measurement_px": {
            "median": numeric_summary(rows, "projection_residual_to_measurement_px.median"),
            "p95": numeric_summary(rows, "projection_residual_to_measurement_px.p95"),
        },
        "owner_median_gap_m": numeric_summary(rows, "owner_median_gap_m"),
        "source_scale_counterfactual_comparison": {
            "case_global_scale": scale_cf.get("case_global_scale"),
            "side_global_scales": scale_cf.get("side_global_scales"),
            "row_scale_candidate_summary": scale_cf.get("row_scale_candidate_summary"),
            "mode_summaries": scale_cf.get("mode_summaries"),
        },
        "source_tail_depth_observation_comparison": {
            "tail_factor_candidate_rows": tail_depth.get("tail_factor_candidate_rows"),
            "tail_depth_observation_state_counts": tail_depth.get("tail_depth_observation_state_counts"),
            "supported_tail_depth_observation_state_counts": tail_depth.get(
                "supported_tail_depth_observation_state_counts"
            ),
        },
        "problem_semantics": {
            "variables": (
                "One case-global hand scale and one camera-ray depth shift for every row with reconstructed "
                "hand geometry and metric depth projection support."
            ),
            "depth_factors": (
                "Fixed-association UniDepth factors use the same owner partition as the hand scale counterfactual; "
                "the solved state is then reprojected and resampled before acceptance counts are computed."
            ),
            "claim_limit": (
                "This graph can repair projective hand-depth scale and translation errors. It cannot repair local "
                "MANO articulation, surface topology, occlusion/depth-observation ownership, object geometry, or "
                "physical contact ownership."
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_depth_repair_graph.json", report)
    return report


def case_summary(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    case = require_str(report.get("case"), "case")
    return {
        "case": case,
        "report_path": str(args.output_root / case / "v17_hand_depth_repair_graph.json"),
        "frame_count": require_int(report.get("frame_count"), "frame_count"),
        "hand_depth_repair_graph_variable_count": require_int(
            report.get("hand_depth_repair_graph_variable_count"),
            "hand depth repair graph variable count",
        ),
        "base_available_rows": require_int(report.get("base_available_rows"), "base available rows"),
        "depth_data_candidate_rows": require_int(report.get("depth_data_candidate_rows"), "depth data rows"),
        "case_global_scale": report.get("case_global_scale"),
        "case_global_scale_bounds": require_dict(report.get("case_global_scale_bounds"), "scale bounds"),
        "case_global_scale_bound_hit": bool(report.get("case_global_scale_bound_hit") is True),
        "case_global_scaled_wrist_to_middle_tip_m": require_dict(
            report.get("case_global_scaled_wrist_to_middle_tip_m"),
            "scaled wrist summary",
        ),
        "hand_ray_shift_abs_m": require_dict(report.get("hand_ray_shift_abs_m"), "shift abs summary"),
        "hand_ray_shift_bound_hit_rows": require_int(
            report.get("hand_ray_shift_bound_hit_rows"),
            "shift bound hits",
        ),
        "system": require_dict(report.get("system"), "system"),
        "solver": require_dict(report.get("solver"), "solver"),
        "solver_state_counts": require_dict(report.get("solver_state_counts"), "solver state counts"),
        "owner_depth_state_counts": require_dict(report.get("owner_depth_state_counts"), "owner depth state counts"),
        "metric_hand_state_accepted_rows": require_int(
            report.get("metric_hand_state_accepted_rows"),
            "accepted rows",
        ),
        "depth_repair_factor_candidate_rows": require_int(
            report.get("depth_repair_factor_candidate_rows"),
            "repair rows",
        ),
        "projection_residual_to_measurement_px": require_dict(
            report.get("projection_residual_to_measurement_px"),
            "projection residual summary",
        ),
        "owner_median_gap_m": require_dict(report.get("owner_median_gap_m"), "owner median gap"),
        **FALSE_READY,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_scale_depth_counterfactual_root / "v17_hand_scale_depth_counterfactual_summary.json",
        "hand scale-depth counterfactual summary",
    )
    summary = require_dict(load_json(summary_path), "hand scale-depth counterfactual summary")
    reports = [
        solve_case(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "solve_v17_hand_depth_repair_graph",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_scale_depth_counterfactual_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [case_summary(report, args) for report in reports],
        "hand_depth_repair_graph_variable_count": sum(
            require_int(report.get("hand_depth_repair_graph_variable_count"), "variable count") for report in reports
        ),
        "base_available_rows": sum(require_int(report.get("base_available_rows"), "base rows") for report in reports),
        "depth_data_candidate_rows": sum(
            require_int(report.get("depth_data_candidate_rows"), "depth data rows") for report in reports
        ),
        "metric_hand_state_accepted_rows": sum(
            require_int(report.get("metric_hand_state_accepted_rows"), "accepted rows") for report in reports
        ),
        "depth_repair_factor_candidate_rows": sum(
            require_int(report.get("depth_repair_factor_candidate_rows"), "repair rows") for report in reports
        ),
        "hand_ray_shift_bound_hit_rows": sum(
            require_int(report.get("hand_ray_shift_bound_hit_rows"), "bound hit rows") for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_depth_repair_graph_summary.json", payload)
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
        "--hand-metric-depth-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument(
        "--hand-scale-depth-counterfactual-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual"),
    )
    parser.add_argument(
        "--hand-tail-depth-observation-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--max-depth-samples-per-row", type=int, default=48)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--min-hand-scale", type=float, default=0.45)
    parser.add_argument("--max-hand-scale", type=float, default=1.35)
    parser.add_argument("--min-scaled-wrist-to-middle-tip-m", type=float, default=0.10)
    parser.add_argument("--max-scaled-wrist-to-middle-tip-m", type=float, default=0.22)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--min-corrected-hand-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-keypoint-px", type=float, default=6.0)
    parser.add_argument("--sigma-hand-scale-prior", type=float, default=0.20)
    parser.add_argument("--sigma-hand-ray-shift-prior-m", type=float, default=0.15)
    parser.add_argument("--sigma-hand-ray-shift-step-m", type=float, default=0.03)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=45)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--max-solver-iter", type=int, default=500)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    parser.add_argument("--bound-tolerance-scale", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
