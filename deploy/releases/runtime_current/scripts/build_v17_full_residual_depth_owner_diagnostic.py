#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

from build_v17_hand_depth_repair_residual_owner_state import (
    owner_valid_mask,
    row_samples,
    selected_residual,
)
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    source_summary,
    summarize,
    write_json,
)


STATUS = "v17_full_residual_depth_owner_diagnostic_qc"
CLAIM = (
    "This artifact tests the depth owner of the persistent full-residual surface-tail rows after "
    "pose-enabled relinearization. It recomputes the final residual sample assignment from the "
    "post-solve MANO projection, then separates assigned local-surface pixels from unassigned "
    "tail pixels by depth-gap sign and active-object-mask proximity."
)

SURFACE_FACTOR_STATE = "relinearized_surface_factor_variable"
DEPTH_TAIL_STATE = "depth_tail_incompatible"
LOCAL_STATE = "relinearized_reprojected_local_surface_factor_candidate"
MIXED_STATE = "relinearized_reprojected_mixed_surface_depth_owner"
DEPTH_OBSERVATION_STATE = "relinearized_reprojected_depth_observation_owner"


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def optional_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, label)


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be bool")
    return value


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if require_bool(row.get(key), key))


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = optional_finite_number(row.get(key), key)
        if value is not None:
            values.append(value)
    return summarize(values)


def private_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if isinstance(raw, list):
            values.extend(finite_number(value, f"{key} value") for value in raw)
    return values


def private_values_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize(private_values(rows, key))


def nested_numeric_summary(rows: list[dict[str, Any]], key: str, nested_key: str) -> dict[str, Any]:
    values = []
    for row in rows:
        nested = row.get(key)
        if not isinstance(nested, dict):
            continue
        value = optional_finite_number(nested.get(nested_key), f"{key}.{nested_key}")
        if value is not None:
            values.append(value)
    return summarize(values)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def summarized(values: np.ndarray) -> dict[str, Any]:
    vals = values.astype(np.float64)
    vals = vals[np.isfinite(vals)]
    return summarize(vals.astype(float).tolist())


def applied_rows(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("rows"), f"{label} rows"):
        row = require_dict(raw, f"{label} row")
        if row.get("relinearized_delta_applied") is not True:
            continue
        graph_id = require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            f"{label} hand depth repair graph id",
        )
        if graph_id in out:
            raise RuntimeError(f"{label} duplicate applied row id: {graph_id}")
        out[graph_id] = row
    expected = require_int(report.get("relinearized_variable_rows"), f"{label} variable rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} applied rows {len(out)} do not match reported variables {expected}")
    return out


def factor_rows(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("factor_rows"), f"{label} factor rows"):
        row = require_dict(raw, f"{label} factor row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            f"{label} source hand depth repair graph id",
        )
        if graph_id in out:
            raise RuntimeError(f"{label} duplicate factor row id: {graph_id}")
        out[graph_id] = row
    expected = require_int(report.get("relinearized_variable_rows"), f"{label} variable rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} factor rows {len(out)} do not match reported variables {expected}")
    return out


def geometry_rows(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("geometry_rows"), f"{label} geometry rows"):
        row = require_dict(raw, f"{label} geometry row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            f"{label} source hand depth repair graph id",
        )
        if graph_id in out:
            raise RuntimeError(f"{label} duplicate geometry row id: {graph_id}")
        out[graph_id] = row
    return out


def transition_rows(report: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("rows"), f"{label} transition rows"):
        row = require_dict(raw, f"{label} transition row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            f"{label} source hand depth repair graph id",
        )
        if graph_id in out:
            raise RuntimeError(f"{label} duplicate transition row id: {graph_id}")
        out[graph_id] = row
    expected = require_int(report.get("transition_variable_rows"), f"{label} transition rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} transition rows {len(out)} do not match reported variables {expected}")
    return out


def mask_summary(mask: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"count": 0}
    return summarize(selected.astype(float).tolist())


def assignment_masks(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = row_samples(row)
    selected = selected_residual(row, samples, args)
    x = cast(np.ndarray, samples["x"]).astype(np.float64)
    y = cast(np.ndarray, samples["y"]).astype(np.float64)
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    near = cast(np.ndarray, samples["near"]).astype(bool)
    far = cast(np.ndarray, samples["far"]).astype(bool)
    object_distance_px = cast(np.ndarray, samples["object_distance_px"]).astype(np.float64)
    owner_valid = owner_valid_mask(row, samples, args)
    all_valid = (
        np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    gap = hand_z - metric_z
    compatible_seed = all_valid & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    owner_compatible_seed = owner_valid & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    selected_i = np.flatnonzero(selected)
    seed_i = np.flatnonzero(compatible_seed)
    nearest = np.full(len(x), np.nan, dtype=np.float64)
    assigned = np.zeros(len(x), dtype=bool)
    if selected_i.size > 0 and seed_i.size > 0:
        selected_xy = np.stack([x[selected_i], y[selected_i]], axis=1)
        seed_xy = np.stack([x[seed_i], y[seed_i]], axis=1)
        distances, _ = cKDTree(seed_xy).query(selected_xy, k=1)
        distances = np.asarray(distances, dtype=np.float64)
        nearest[selected_i] = distances
        assigned[selected_i] = np.isfinite(distances) & (
            distances <= float(args.local_projection_search_radius_px)
        )
    unassigned = selected & ~assigned
    hand_in_front = selected & (gap < -float(args.depth_tail_abs_m))
    hand_behind = selected & (gap > float(args.depth_tail_abs_m))
    return {
        "selected": selected,
        "assigned": assigned,
        "unassigned": unassigned,
        "compatible_seed": compatible_seed,
        "owner_compatible_seed": owner_compatible_seed,
        "nearest_compatible_px": nearest,
        "gap_m": gap,
        "near": near,
        "far": far,
        "object_distance_px": object_distance_px,
        "hand_in_front": hand_in_front,
        "hand_behind": hand_behind,
    }


def fraction(mask: np.ndarray, denom: np.ndarray) -> float | None:
    count = int(np.count_nonzero(denom))
    if count == 0:
        return None
    return float(np.count_nonzero(mask & denom) / count)


def row_diagnostic(
    *,
    graph_id: str,
    pose_row: dict[str, Any],
    factor_row: dict[str, Any],
    geometry_row: dict[str, Any] | None,
    transition_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    factor_state = require_str(
        factor_row.get("relinearized_input_factor_state"),
        "factor state",
    )
    surface_factor = bool(factor_state == SURFACE_FACTOR_STATE)
    pose_state = require_str(pose_row.get("relinearized_reprojection_state"), "pose state")
    owner_depth_state = require_str(pose_row.get("owner_depth_state"), "owner depth state")
    persistent_residual = require_bool(
        transition_row.get("residual_owner_persistent"),
        "residual owner persistent",
    )
    geometry_after = None if geometry_row is None else require_dict(
        geometry_row.get("after"),
        "geometry after",
    )
    geometry_depth_median = None if geometry_after is None else optional_finite_number(
        geometry_after.get("depth_abs_median_m"),
        "geometry depth median",
    )
    geometry_depth_p95 = None if geometry_after is None else optional_finite_number(
        geometry_after.get("depth_abs_p95_m"),
        "geometry depth p95",
    )
    surface_geometry_depth_pass = bool(
        surface_factor
        and geometry_depth_median is not None
        and geometry_depth_p95 is not None
        and geometry_depth_median <= float(args.max_geometry_depth_median_m)
        and geometry_depth_p95 <= float(args.max_geometry_depth_p95_m)
    )
    masks = assignment_masks(pose_row, args)
    selected = cast(np.ndarray, masks["selected"])
    assigned = cast(np.ndarray, masks["assigned"])
    unassigned = cast(np.ndarray, masks["unassigned"])
    gap = cast(np.ndarray, masks["gap_m"])
    nearest = cast(np.ndarray, masks["nearest_compatible_px"])
    near = cast(np.ndarray, masks["near"])
    far = cast(np.ndarray, masks["far"])
    object_distance_px = cast(np.ndarray, masks["object_distance_px"])
    hand_in_front = cast(np.ndarray, masks["hand_in_front"])
    hand_behind = cast(np.ndarray, masks["hand_behind"])
    assigned_gap = mask_summary(assigned, gap)
    assigned_target_like = mask_summary(assigned, np.abs(gap))
    unassigned_gap = mask_summary(unassigned, gap)
    unassigned_abs_gap = mask_summary(unassigned, np.abs(gap))
    assigned_source_rejected = bool(
        require_int(assigned_gap.get("count"), "assigned gap count") > 0
        and abs(finite_number(assigned_gap.get("median"), "assigned gap median"))
        >= float(args.min_rejected_source_residual_abs_gap_m)
    )
    owner_depth_tail = bool(owner_depth_state == DEPTH_TAIL_STATE)
    persistent_surface_tail = bool(
        persistent_residual
        and surface_factor
        and owner_depth_tail
        and pose_state in {LOCAL_STATE, MIXED_STATE, DEPTH_OBSERVATION_STATE}
    )
    unassigned_count = int(np.count_nonzero(unassigned))
    unassigned_near_fraction = fraction(near, unassigned)
    unassigned_far_fraction = fraction(far, unassigned)
    unassigned_front_fraction = fraction(hand_in_front, unassigned)
    unassigned_behind_fraction = fraction(hand_behind, unassigned)
    unassigned_far_front_majority = bool(
        unassigned_count > 0
        and unassigned_far_fraction is not None
        and unassigned_front_fraction is not None
        and unassigned_far_fraction >= float(args.majority_fraction)
        and unassigned_front_fraction >= float(args.majority_fraction)
    )
    unassigned_near_object_majority = bool(
        unassigned_count > 0
        and unassigned_near_fraction is not None
        and unassigned_near_fraction >= float(args.majority_fraction)
    )
    studied_depth_owner_row = bool(
        persistent_surface_tail and surface_geometry_depth_pass and assigned_source_rejected
    )
    return {
        "case": require_str(pose_row.get("case"), "case"),
        "source_hand_depth_repair_graph_variable_id": graph_id,
        "frame_idx": require_int(pose_row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(pose_row.get("hand_side"), "hand_side"),
        "hand_index": require_int(pose_row.get("hand_index"), "hand_index"),
        "pose_reprojection_state": pose_state,
        "pose_input_factor_state": factor_state,
        "pose_owner_depth_state": owner_depth_state,
        "surface_factor_row": surface_factor,
        "persistent_residual_owner_row": persistent_residual,
        "persistent_surface_depth_tail_row": persistent_surface_tail,
        "surface_geometry_depth_pass": surface_geometry_depth_pass,
        "assigned_source_depth_rejected_by_final_residual": assigned_source_rejected,
        "studied_depth_owner_row": studied_depth_owner_row,
        "selected_residual_sample_count": int(np.count_nonzero(selected)),
        "compatible_seed_sample_count": int(np.count_nonzero(cast(np.ndarray, masks["compatible_seed"]))),
        "owner_compatible_seed_sample_count": int(
            np.count_nonzero(cast(np.ndarray, masks["owner_compatible_seed"]))
        ),
        "assigned_residual_sample_count": int(np.count_nonzero(assigned)),
        "unassigned_residual_sample_count": unassigned_count,
        "unassigned_far_object_majority": bool(
            unassigned_far_fraction is not None
            and unassigned_far_fraction >= float(args.majority_fraction)
        ),
        "unassigned_near_object_majority": unassigned_near_object_majority,
        "unassigned_hand_in_front_majority": bool(
            unassigned_front_fraction is not None
            and unassigned_front_fraction >= float(args.majority_fraction)
        ),
        "unassigned_hand_behind_majority": bool(
            unassigned_behind_fraction is not None
            and unassigned_behind_fraction >= float(args.majority_fraction)
        ),
        "unassigned_far_field_hand_in_front_depth_owner_row": bool(
            studied_depth_owner_row and unassigned_far_front_majority
        ),
        "unassigned_near_object_depth_owner_row": bool(
            studied_depth_owner_row and unassigned_near_object_majority
        ),
        "unassigned_near_active_object_fraction": unassigned_near_fraction,
        "unassigned_far_from_active_object_fraction": unassigned_far_fraction,
        "unassigned_hand_in_front_fraction": unassigned_front_fraction,
        "unassigned_hand_behind_fraction": unassigned_behind_fraction,
        "geometry_depth_abs_median_m": geometry_depth_median,
        "geometry_depth_abs_p95_m": geometry_depth_p95,
        "assigned_residual_gap_m": summarized(gap[assigned]),
        "assigned_abs_residual_gap_m": assigned_target_like,
        "unassigned_residual_gap_m": summarized(gap[unassigned]),
        "unassigned_abs_residual_gap_m": unassigned_abs_gap,
        "assigned_nearest_compatible_pixel_shift_px": summarized(nearest[assigned]),
        "unassigned_nearest_compatible_pixel_shift_px": summarized(nearest[unassigned]),
        "assigned_object_distance_px": summarized(object_distance_px[assigned]),
        "unassigned_object_distance_px": summarized(object_distance_px[unassigned]),
        "_assigned_residual_gap_values": gap[assigned][np.isfinite(gap[assigned])].astype(float).tolist()
        if studied_depth_owner_row
        else [],
        "_unassigned_residual_gap_values": gap[unassigned][np.isfinite(gap[unassigned])].astype(float).tolist()
        if studied_depth_owner_row
        else [],
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    pose_graph_path = existing_path(
        args.pose_full_residual_graph_root
        / case
        / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
        f"{case} pose full-residual graph",
    )
    transition_path = existing_path(
        args.pose_transition_diagnostic_root
        / case
        / "v17_full_residual_pose_transition_diagnostic.json",
        f"{case} pose transition diagnostic",
    )
    pose_graph = require_dict(load_json(pose_graph_path), f"{case} pose full-residual graph")
    transition = require_dict(load_json(transition_path), f"{case} transition diagnostic")
    if require_int(pose_graph.get("frame_count"), f"{case} pose frame_count") != require_int(
        transition.get("frame_count"),
        f"{case} transition frame_count",
    ):
        raise RuntimeError(f"{case} frame_count mismatch between pose graph and transition diagnostic")
    pose_rows = applied_rows(pose_graph, f"{case} pose graph")
    factors = factor_rows(pose_graph, f"{case} pose graph")
    geometries = geometry_rows(pose_graph, f"{case} pose graph")
    transitions = transition_rows(transition, f"{case} transition")
    if set(pose_rows) != set(factors):
        raise RuntimeError(f"{case} pose rows and factor rows disagree")
    if set(pose_rows) != set(transitions):
        raise RuntimeError(f"{case} pose rows and transition rows disagree")
    rows_private = [
        row_diagnostic(
            graph_id=graph_id,
            pose_row=pose_rows[graph_id],
            factor_row=factors[graph_id],
            geometry_row=geometries.get(graph_id),
            transition_row=transitions[graph_id],
            args=args,
        )
        for graph_id in sorted(pose_rows)
    ]
    studied = [row for row in rows_private if row["studied_depth_owner_row"]]
    rows = [public_row(row) for row in rows_private]
    studied_unassigned_gap_values = private_values(studied, "_unassigned_residual_gap_values")
    studied_assigned_gap_values = private_values(studied, "_assigned_residual_gap_values")
    report = {
        "method": "build_v17_full_residual_depth_owner_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "pose_full_residual_graph": source_summary(pose_graph_path, pose_graph),
            "pose_transition_diagnostic": source_summary(transition_path, transition),
        },
        "frame_count": require_int(pose_graph.get("frame_count"), f"{case} frame_count"),
        "transition_variable_rows": len(rows),
        "surface_factor_rows": bool_count(rows_private, "surface_factor_row"),
        "persistent_surface_depth_tail_rows": bool_count(rows_private, "persistent_surface_depth_tail_row"),
        "surface_geometry_depth_pass_rows": bool_count(rows_private, "surface_geometry_depth_pass"),
        "studied_depth_owner_rows": len(studied),
        "studied_rows_with_unassigned_residual_pixels": sum(
            1 for row in studied if require_int(row.get("unassigned_residual_sample_count"), "unassigned") > 0
        ),
        "studied_unassigned_far_object_majority_rows": bool_count(
            studied,
            "unassigned_far_object_majority",
        ),
        "studied_unassigned_near_object_majority_rows": bool_count(
            studied,
            "unassigned_near_object_majority",
        ),
        "studied_unassigned_hand_in_front_majority_rows": bool_count(
            studied,
            "unassigned_hand_in_front_majority",
        ),
        "studied_unassigned_hand_behind_majority_rows": bool_count(
            studied,
            "unassigned_hand_behind_majority",
        ),
        "studied_far_field_hand_in_front_depth_owner_rows": bool_count(
            studied,
            "unassigned_far_field_hand_in_front_depth_owner_row",
        ),
        "studied_near_object_depth_owner_rows": bool_count(
            studied,
            "unassigned_near_object_depth_owner_row",
        ),
        "studied_unassigned_residual_sample_count": sum(
            require_int(row.get("unassigned_residual_sample_count"), "unassigned") for row in studied
        ),
        "studied_assigned_residual_sample_count": sum(
            require_int(row.get("assigned_residual_sample_count"), "assigned") for row in studied
        ),
        "studied_unassigned_far_from_active_object_fraction": numeric_summary(
            studied,
            "unassigned_far_from_active_object_fraction",
        ),
        "studied_unassigned_hand_in_front_fraction": numeric_summary(
            studied,
            "unassigned_hand_in_front_fraction",
        ),
        "studied_unassigned_residual_gap_m": summarize(studied_unassigned_gap_values),
        "studied_assigned_residual_gap_m": summarize(studied_assigned_gap_values),
        "studied_unassigned_nearest_compatible_pixel_shift_px_median": nested_numeric_summary(
            studied,
            "unassigned_nearest_compatible_pixel_shift_px",
            "median",
        ),
        "pose_reprojection_state_counts": state_counts(rows_private, "pose_reprojection_state"),
        "rows": rows,
        "_studied_unassigned_residual_gap_values": studied_unassigned_gap_values,
        "_studied_assigned_residual_gap_values": studied_assigned_gap_values,
        **FALSE_READY,
    }
    write_json(
        args.output_root / case / "v17_full_residual_depth_owner_diagnostic.json",
        {key: value for key, value in report.items() if not key.startswith("_")},
    )
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [row for case in cases for row in require_list(case.get("rows"), "case rows")]
    studied = [row for row in rows if row["studied_depth_owner_row"]]
    studied_unassigned_gap_values = [
        value
        for case in cases
        for value in cast(list[float], case.get("_studied_unassigned_residual_gap_values", []))
    ]
    studied_assigned_gap_values = [
        value
        for case in cases
        for value in cast(list[float], case.get("_studied_assigned_residual_gap_values", []))
    ]
    summary = {
        "method": "build_v17_full_residual_depth_owner_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame_count") for case in cases),
        "transition_variable_rows": len(rows),
        "surface_factor_rows": bool_count(rows, "surface_factor_row"),
        "persistent_surface_depth_tail_rows": bool_count(rows, "persistent_surface_depth_tail_row"),
        "surface_geometry_depth_pass_rows": bool_count(rows, "surface_geometry_depth_pass"),
        "studied_depth_owner_rows": len(studied),
        "studied_rows_with_unassigned_residual_pixels": sum(
            1 for row in studied if require_int(row.get("unassigned_residual_sample_count"), "unassigned") > 0
        ),
        "studied_unassigned_far_object_majority_rows": bool_count(
            studied,
            "unassigned_far_object_majority",
        ),
        "studied_unassigned_near_object_majority_rows": bool_count(
            studied,
            "unassigned_near_object_majority",
        ),
        "studied_unassigned_hand_in_front_majority_rows": bool_count(
            studied,
            "unassigned_hand_in_front_majority",
        ),
        "studied_unassigned_hand_behind_majority_rows": bool_count(
            studied,
            "unassigned_hand_behind_majority",
        ),
        "studied_far_field_hand_in_front_depth_owner_rows": bool_count(
            studied,
            "unassigned_far_field_hand_in_front_depth_owner_row",
        ),
        "studied_near_object_depth_owner_rows": bool_count(
            studied,
            "unassigned_near_object_depth_owner_row",
        ),
        "studied_unassigned_residual_sample_count": sum(
            require_int(row.get("unassigned_residual_sample_count"), "unassigned") for row in studied
        ),
        "studied_assigned_residual_sample_count": sum(
            require_int(row.get("assigned_residual_sample_count"), "assigned") for row in studied
        ),
        "studied_unassigned_far_from_active_object_fraction": numeric_summary(
            studied,
            "unassigned_far_from_active_object_fraction",
        ),
        "studied_unassigned_hand_in_front_fraction": numeric_summary(
            studied,
            "unassigned_hand_in_front_fraction",
        ),
        "studied_unassigned_residual_gap_m": summarize(studied_unassigned_gap_values),
        "studied_assigned_residual_gap_m": summarize(studied_assigned_gap_values),
        "pose_reprojection_state_counts": state_counts(rows, "pose_reprojection_state"),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame_count"),
                "studied_depth_owner_rows": require_int(
                    case.get("studied_depth_owner_rows"),
                    "case studied rows",
                ),
                "studied_far_field_hand_in_front_depth_owner_rows": require_int(
                    case.get("studied_far_field_hand_in_front_depth_owner_rows"),
                    "case far-front rows",
                ),
                "studied_near_object_depth_owner_rows": require_int(
                    case.get("studied_near_object_depth_owner_rows"),
                    "case near-object rows",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_full_residual_depth_owner_diagnostic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose-full-residual-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose"),
    )
    parser.add_argument(
        "--pose-transition-diagnostic-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_pose_transition_diagnostic"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--depth-tail-abs-m", type=float, default=0.08)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--max-geometry-depth-median-m", type=float, default=0.03)
    parser.add_argument("--max-geometry-depth-p95-m", type=float, default=0.08)
    parser.add_argument("--min-rejected-source-residual-abs-gap-m", type=float, default=0.08)
    parser.add_argument("--majority-fraction", type=float, default=0.5)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
