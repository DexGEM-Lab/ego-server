#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

from apply_v17_hand_far_field_temporal_refit import (
    finite_float,
    refit_delta_by_graph_id,
)
from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    annotation_frames,
    depth_archive,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary
from build_v17_hand_temporal_reprojection_residual_owner_state import (
    owner_valid_mask,
    temporal_owner_state,
)
from solve_v17_hand_depth_repair_graph import build_base_row, evaluate_row, numeric_summary


STATUS = "v17_hand_temporal_owner_weighted_refit_qc"
CLAIM = (
    "This artifact solves a second-pass temporal hand-depth graph after far-field temporal reprojection. "
    "It gives geometry depth factors only to residual pixels assigned to nearby same-hand compatible depth, "
    "keeps depth-observation and projection-untrusted rows as explicit prior/smoothness variables, then "
    "reprojects MANO and resamples UniDepth. It does not update annotations and does not complete the V3 joint solver."
)

LOCAL_STATE = "temporal_reprojection_local_surface_factor_candidate"
MIXED_STATE = "temporal_reprojection_mixed_surface_depth_owner"
DEPTH_STATE = "temporal_reprojection_depth_observation_owner"
UNTRUSTED_STATE = "temporal_reprojection_projection_untrusted"
COMPATIBLE_STATE = "temporal_reprojection_metric_depth_compatible"


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def add_row(
    matrix_rows: list[int],
    matrix_cols: list[int],
    matrix_vals: list[float],
    rhs: list[float],
    row_i: int,
    terms: list[tuple[int, float]],
    target: float,
) -> int:
    for col, value in terms:
        matrix_rows.append(row_i)
        matrix_cols.append(col)
        matrix_vals.append(value)
    rhs.append(target)
    return row_i + 1


def thin(values: np.ndarray, max_count: int) -> np.ndarray:
    if values.size <= max_count:
        return values
    return values[np.linspace(0, values.size - 1, max_count, dtype=np.int64)]


def assignment_pairs(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x = cast(np.ndarray, samples["x"]).astype(np.float64)
    y = cast(np.ndarray, samples["y"]).astype(np.float64)
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
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
    selected_gap = gap[selected_i]
    out: dict[str, Any] = {
        "residual_sample_count": int(selected_i.size),
        "compatible_seed_sample_count": int(seed_i.size),
        "owner_compatible_seed_sample_count": int(np.count_nonzero(owner_compatible_seed)),
        "nearby_compatible_assignment_fraction": None,
        "assigned_residual_sample_count": 0,
        "unassigned_residual_sample_count": int(selected_i.size),
        "selected_residual_gap_m": summarize(selected_gap.astype(float).tolist()),
        "assigned_pixel_shift_px": summarize([]),
        "assigned_source_residual_abs_gap_m": summarize([]),
        "assigned_target_seed_abs_gap_m": summarize([]),
        "assigned_hand_depth_delta_to_seed_m": summarize([]),
        "assigned_metric_depth_delta_to_seed_m": summarize([]),
        "assigned_source_to_seed_depth_residual_m": summarize([]),
        "_source_to_seed_delta_m": np.asarray([], dtype=np.float64),
    }
    if selected_i.size == 0:
        return out
    if seed_i.size == 0:
        out["nearby_compatible_assignment_fraction"] = 0.0
        return out
    selected_xy = np.stack([x[selected_i], y[selected_i]], axis=1)
    seed_xy = np.stack([x[seed_i], y[seed_i]], axis=1)
    dist, nearest = cKDTree(seed_xy).query(selected_xy, k=1)
    dist = np.asarray(dist, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)
    assigned = np.isfinite(dist) & (dist <= float(args.local_projection_search_radius_px))
    assigned_count = int(np.count_nonzero(assigned))
    out["nearby_compatible_assignment_fraction"] = float(assigned_count / int(selected_i.size))
    out["assigned_residual_sample_count"] = assigned_count
    out["unassigned_residual_sample_count"] = int(selected_i.size) - assigned_count
    out["nearest_compatible_pixel_shift_px"] = summarize(dist[np.isfinite(dist)].astype(float).tolist())
    if assigned_count == 0:
        return out
    selected_match_i = selected_i[assigned]
    seed_match_i = seed_i[nearest[assigned]]
    source_to_seed = hand_z[selected_match_i] - metric_z[seed_match_i]
    out["assigned_pixel_shift_px"] = summarize(dist[assigned].astype(float).tolist())
    out["assigned_source_residual_abs_gap_m"] = summarize(
        np.abs(gap[selected_match_i]).astype(float).tolist()
    )
    out["assigned_target_seed_abs_gap_m"] = summarize(np.abs(gap[seed_match_i]).astype(float).tolist())
    out["assigned_hand_depth_delta_to_seed_m"] = summarize(
        (hand_z[seed_match_i] - hand_z[selected_match_i]).astype(float).tolist()
    )
    out["assigned_metric_depth_delta_to_seed_m"] = summarize(
        (metric_z[seed_match_i] - metric_z[selected_match_i]).astype(float).tolist()
    )
    out["assigned_source_to_seed_depth_residual_m"] = summarize(source_to_seed.astype(float).tolist())
    out["_source_to_seed_delta_m"] = (metric_z[seed_match_i] - hand_z[selected_match_i]).astype(
        np.float64
    )
    return out


def compatible_anchor_gaps(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    valid = owner_valid_mask(row, samples, args)
    gap = hand_z - metric_z
    anchors = valid & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    return gap[anchors].astype(np.float64)


def public_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in assignment.items() if not k.startswith("_")}


def input_factor_state(row: dict[str, Any]) -> str:
    if row.get("owner_weighted_variable_candidate") is not True:
        return require_str(row.get("owner_weighted_candidate_rejection"), "owner-weighted rejection")
    if row.get("owner_weighted_geometry_factor_row") is True:
        return "owner_weighted_geometry_factor_variable"
    if row.get("owner_weighted_compatible_anchor_row") is True:
        return "owner_weighted_compatible_anchor_variable"
    owner = require_str(row.get("source_temporal_reprojection_residual_owner_state"), "owner state")
    if owner == DEPTH_STATE:
        return "owner_weighted_depth_observation_prior_smooth_variable"
    if owner == UNTRUSTED_STATE:
        return "owner_weighted_projection_untrusted_prior_smooth_variable"
    if owner == COMPATIBLE_STATE:
        return "owner_weighted_compatible_without_anchor_prior_smooth_variable"
    return "owner_weighted_sparse_owner_prior_smooth_variable"


def output_owner_state(row: dict[str, Any], assignment: dict[str, Any] | None, args: argparse.Namespace) -> str:
    if row.get("source_temporal_refit_state") is None:
        return "not_far_field_temporal_refit_row"
    if row.get("temporal_refit_delta_applied") is not True:
        return "owner_weighted_source_temporal_delta_not_applied"
    if row.get("owner_weighted_delta_applied") is not True:
        return "owner_weighted_delta_not_applied"
    if row.get("owner_sample_partition") is None or not isinstance(row.get("partitions"), dict):
        return "owner_weighted_reprojected_unobserved"
    if assignment is None:
        raise RuntimeError("applied owner-weighted row needs an assignment state")
    temporal_state = temporal_owner_state(row, assignment, args)
    mapping = {
        "temporal_reprojection_metric_depth_compatible": "owner_weighted_reprojected_metric_depth_compatible",
        "temporal_reprojection_projection_untrusted": "owner_weighted_reprojected_projection_untrusted",
        "temporal_reprojection_residual_unobserved": "owner_weighted_reprojected_residual_unobserved",
        "temporal_reprojection_local_surface_factor_candidate": "owner_weighted_reprojected_local_surface_factor_candidate",
        "temporal_reprojection_mixed_surface_depth_owner": "owner_weighted_reprojected_mixed_surface_depth_owner",
        "temporal_reprojection_depth_observation_owner": "owner_weighted_reprojected_depth_observation_owner",
        "temporal_delta_not_applied": "owner_weighted_source_temporal_delta_not_applied",
    }
    if temporal_state not in mapping:
        raise RuntimeError(f"unknown temporal owner state after owner-weighted refit: {temporal_state}")
    return mapping[temporal_state]


def solve_owner_weighted_rows(
    rows_in: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_indices = [
        i for i, row in enumerate(rows_in) if row.get("owner_weighted_variable_candidate") is True
    ]
    if not candidate_indices:
        rows_out = [
            {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "owner_weighted_delta_shift_m": None,
                "owner_weighted_total_hand_ray_shift_m": row.get("current_temporal_hand_ray_shift_m"),
                "owner_weighted_delta_bound_hit": False,
                "owner_weighted_fixed_factor_depth_improved": False,
                "owner_weighted_fixed_factor_depth_threshold_met": False,
                "owner_weighted_factor_state": input_factor_state(row),
            }
            for row in rows_in
        ]
        return rows_out, {
            "variable_count": 0,
            "geometry_depth_sample_factor_count": 0,
            "compatible_anchor_sample_factor_count": 0,
            "prior_factor_count": 0,
            "smoothness_factor_count": 0,
            "matrix_rows": 0,
            "matrix_cols": 0,
            "solver_skipped_reason": "no_owner_weighted_temporal_variables",
        }
    var_by_row = {source_i: var_i for var_i, source_i in enumerate(candidate_indices)}
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_vals: list[float] = []
    rhs: list[float] = []
    row_i = 0
    geometry_factors = 0
    anchor_factors = 0
    prior_factors = 0
    smooth_factors = 0
    sigma_geometry = float(args.sigma_owner_geometry_depth_m)
    sigma_anchor = float(args.sigma_owner_compatible_anchor_m)
    sigma_prior = float(args.sigma_owner_delta_prior_m)
    sigma_step = float(args.sigma_owner_delta_step_m)
    lower = np.full(len(candidate_indices), -float(args.max_abs_owner_weighted_delta_m), dtype=np.float64)
    upper = np.full(len(candidate_indices), float(args.max_abs_owner_weighted_delta_m), dtype=np.float64)
    fixed_before_by_row: dict[int, np.ndarray] = {}
    for source_i in candidate_indices:
        var_i = var_by_row[source_i]
        row = rows_in[source_i]
        lower[var_i] = max(lower[var_i], finite_float(row.get("owner_delta_lower_bound_m"), "lower bound"))
        upper[var_i] = min(upper[var_i], finite_float(row.get("owner_delta_upper_bound_m"), "upper bound"))
        geometry_targets = thin(
            np.asarray(row.get("_owner_geometry_target_delta_m"), dtype=np.float64),
            int(args.max_factor_samples_per_row),
        )
        anchor_gaps = thin(
            np.asarray(row.get("_compatible_anchor_gap_m"), dtype=np.float64),
            int(args.max_factor_samples_per_row),
        )
        fixed_before_by_row[source_i] = np.concatenate([(-geometry_targets), anchor_gaps])
        for target_delta in geometry_targets:
            row_i = add_row(
                matrix_rows,
                matrix_cols,
                matrix_vals,
                rhs,
                row_i,
                [(var_i, 1.0 / sigma_geometry)],
                float(target_delta) / sigma_geometry,
            )
            geometry_factors += 1
        for gap in anchor_gaps:
            row_i = add_row(
                matrix_rows,
                matrix_cols,
                matrix_vals,
                rhs,
                row_i,
                [(var_i, 1.0 / sigma_anchor)],
                -float(gap) / sigma_anchor,
            )
            anchor_factors += 1
        row_i = add_row(
            matrix_rows,
            matrix_cols,
            matrix_vals,
            rhs,
            row_i,
            [(var_i, 1.0 / sigma_prior)],
            0.0,
        )
        prior_factors += 1
    by_hand: dict[tuple[str, int], list[int]] = {}
    for source_i in candidate_indices:
        row = rows_in[source_i]
        by_hand.setdefault(
            (
                require_str(row.get("hand_side"), "hand_side"),
                require_int(row.get("hand_index"), "hand_index"),
            ),
            [],
        ).append(source_i)
    for source_rows in by_hand.values():
        ordered = sorted(source_rows, key=lambda i: require_int(rows_in[i].get("frame_idx"), "frame_idx"))
        for a, b in zip(ordered[:-1], ordered[1:]):
            frame_a = require_int(rows_in[a].get("frame_idx"), "frame_idx")
            frame_b = require_int(rows_in[b].get("frame_idx"), "frame_idx")
            dt = max(1, frame_b - frame_a)
            if dt > int(args.max_temporal_smooth_gap_frames):
                continue
            weight = 1.0 / (sigma_step * float(dt))
            row_i = add_row(
                matrix_rows,
                matrix_cols,
                matrix_vals,
                rhs,
                row_i,
                [(var_by_row[b], weight), (var_by_row[a], -weight)],
                0.0,
            )
            smooth_factors += 1
    if np.any(lower > upper):
        bad = np.flatnonzero(lower > upper)
        raise RuntimeError(f"inconsistent owner-weighted bounds at variables {bad[:12].tolist()}")
    matrix = sparse.csr_matrix(
        (matrix_vals, (matrix_rows, matrix_cols)),
        shape=(row_i, len(candidate_indices)),
    )
    result = lsq_linear(
        matrix,
        np.asarray(rhs, dtype=np.float64),
        bounds=(lower, upper),
        method="trf",
        tol=float(args.solver_tol),
        lsmr_tol="auto",
        max_iter=int(args.max_solver_iter),
        verbose=0,
    )
    if result.x is None or len(result.x) != len(candidate_indices):
        raise RuntimeError("owner-weighted temporal refit returned invalid solution")
    rows_out: list[dict[str, Any]] = []
    for source_i, row in enumerate(rows_in):
        base = {k: v for k, v in row.items() if not k.startswith("_")}
        if source_i not in var_by_row:
            out = {
                **base,
                "owner_weighted_delta_shift_m": None,
                "owner_weighted_total_hand_ray_shift_m": row.get("current_temporal_hand_ray_shift_m"),
                "owner_weighted_delta_bound_hit": False,
                "owner_weighted_fixed_factor_depth_improved": False,
                "owner_weighted_fixed_factor_depth_threshold_met": False,
            }
            rows_out.append({**out, "owner_weighted_factor_state": input_factor_state(out)})
            continue
        var_i = var_by_row[source_i]
        delta = float(result.x[var_i])
        current_shift = finite_float(row.get("current_temporal_hand_ray_shift_m"), "current shift")
        fixed_before = fixed_before_by_row[source_i]
        fixed_after = fixed_before + delta
        before_abs_median = float(np.median(np.abs(fixed_before))) if len(fixed_before) else math.inf
        after_abs_median = float(np.median(np.abs(fixed_after))) if len(fixed_after) else math.inf
        after_abs_p95 = float(np.percentile(np.abs(fixed_after), 95.0)) if len(fixed_after) else math.inf
        lower_i = finite_float(row.get("owner_delta_lower_bound_m"), "lower bound")
        upper_i = finite_float(row.get("owner_delta_upper_bound_m"), "upper bound")
        out = {
            **base,
            "owner_weighted_delta_shift_m": delta,
            "owner_weighted_total_hand_ray_shift_m": current_shift + delta,
            "owner_weighted_fixed_factor_residual": {
                "before": {
                    "signed_gap_m": summarize(fixed_before.astype(float).tolist()),
                    "abs_gap_m": summarize(np.abs(fixed_before).astype(float).tolist()),
                },
                "after": {
                    "signed_gap_m": summarize(fixed_after.astype(float).tolist()),
                    "abs_gap_m": summarize(np.abs(fixed_after).astype(float).tolist()),
                },
            },
            "owner_weighted_delta_bound_hit": bool(
                math.isclose(delta, lower_i, abs_tol=float(args.bound_tolerance_m))
                or math.isclose(delta, upper_i, abs_tol=float(args.bound_tolerance_m))
            ),
            "owner_weighted_fixed_factor_depth_improved": bool(
                before_abs_median - after_abs_median >= float(args.min_depth_improvement_m)
            ),
            "owner_weighted_fixed_factor_depth_threshold_met": bool(
                after_abs_median <= float(args.max_median_abs_depth_gap_m)
                and after_abs_p95 <= float(args.max_p95_abs_depth_gap_m)
            )
            if len(fixed_after)
            else False,
        }
        rows_out.append({**out, "owner_weighted_factor_state": input_factor_state(out)})
    return rows_out, {
        "variable_count": len(candidate_indices),
        "geometry_depth_sample_factor_count": geometry_factors,
        "compatible_anchor_sample_factor_count": anchor_factors,
        "prior_factor_count": prior_factors,
        "smoothness_factor_count": smooth_factors,
        "matrix_rows": row_i,
        "matrix_cols": len(candidate_indices),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_cost": float(result.cost),
        "solver_optimality": float(result.optimality),
        "active_mask_counts": dict(sorted(Counter(int(x) for x in result.active_mask).items())),
    }


def temporal_inputs(case: str, reprojection: dict[str, Any], owner: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    owner_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "owner source graph id"): row
        for row in [
            require_dict(raw, "temporal residual-owner row")
            for raw in require_list(owner.get("rows"), "temporal residual-owner rows")
        ]
    }
    rows_in: list[dict[str, Any]] = []
    for raw in require_list(reprojection.get("rows"), f"{case} temporal reprojection rows"):
        row = require_dict(raw, "temporal reprojection row")
        if row.get("source_temporal_refit_state") is None:
            continue
        graph_id = require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        owner_row = owner_by_id.get(graph_id)
        if owner_row is None:
            raise RuntimeError(f"{case} missing temporal residual-owner row for {graph_id}")
        owner_state = require_str(
            owner_row.get("temporal_reprojection_residual_owner_state"),
            "temporal residual-owner state",
        )
        assignment: dict[str, Any] | None = None
        geometry_targets = np.asarray([], dtype=np.float64)
        anchor_gaps = np.asarray([], dtype=np.float64)
        if row.get("temporal_refit_delta_applied") is True:
            samples = row_samples(row)
            selected = selected_residual(row, samples, args)
            assignment = assignment_pairs(row, samples, selected, args)
            if owner_state in {LOCAL_STATE, MIXED_STATE}:
                geometry_targets = cast(np.ndarray, assignment["_source_to_seed_delta_m"])
            elif owner_state == COMPATIBLE_STATE:
                anchor_gaps = compatible_anchor_gaps(row, samples, args)
        current_shift = None
        if row.get("hand_ray_shift_m") is not None:
            current_shift = finite_float(row.get("hand_ray_shift_m"), "temporal reprojection shift")
        candidate = bool(row.get("temporal_refit_delta_applied") is True and current_shift is not None)
        rejection = None
        if not candidate:
            rejection = "owner_weighted_no_applied_temporal_shift"
        lower = None
        upper = None
        if candidate:
            current = finite_float(current_shift, "current temporal shift")
            lower = max(
                -float(args.max_abs_hand_ray_shift_m) - current,
                -float(args.max_abs_owner_weighted_delta_m),
            )
            upper = min(
                float(args.max_abs_hand_ray_shift_m) - current,
                float(args.max_abs_owner_weighted_delta_m),
            )
            if lower > upper:
                candidate = False
                rejection = "owner_weighted_inconsistent_shift_bounds"
        geometry_factor_row = bool(
            candidate and geometry_targets.size >= int(args.min_owner_weighted_factor_samples)
        )
        anchor_row = bool(candidate and anchor_gaps.size >= int(args.min_owner_weighted_factor_samples))
        rows_in.append(
            {
                "case": case,
                "hand_temporal_owner_weighted_refit_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "hand_temporal_owner_weighted_refit:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_temporal_refit_variable_id": row.get("source_temporal_refit_variable_id"),
                "source_temporal_refit_state": row.get("source_temporal_refit_state"),
                "source_temporal_reprojection_state": row.get("temporal_reprojection_state"),
                "source_temporal_reprojection_residual_owner_state": owner_state,
                "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(row.get("hand_side"), "hand_side"),
                "hand_index": require_int(row.get("hand_index"), "hand_index"),
                "owner_sample_partition": row.get("owner_sample_partition"),
                "owner_depth_state": row.get("owner_depth_state"),
                "temporal_refit_delta_applied": bool(row.get("temporal_refit_delta_applied") is True),
                "source_temporal_refit_delta_shift_m": row.get("temporal_refit_delta_shift_m"),
                "current_temporal_hand_ray_shift_m": current_shift,
                "owner_weighted_variable_candidate": candidate,
                "owner_weighted_candidate_rejection": rejection,
                "owner_weighted_geometry_factor_row": geometry_factor_row,
                "owner_weighted_compatible_anchor_row": anchor_row,
                "owner_weighted_prior_smooth_only_row": bool(candidate and not geometry_factor_row and not anchor_row),
                "owner_weighted_depth_observation_prior_smooth_row": bool(candidate and owner_state == DEPTH_STATE),
                "owner_weighted_projection_untrusted_prior_smooth_row": bool(candidate and owner_state == UNTRUSTED_STATE),
                "owner_delta_lower_bound_m": lower,
                "owner_delta_upper_bound_m": upper,
                "owner_weighted_assignment": None if assignment is None else public_assignment(assignment),
                "owner_weighted_geometry_target_delta_m": summarize(geometry_targets.astype(float).tolist()),
                "owner_weighted_compatible_anchor_gap_m": summarize(anchor_gaps.astype(float).tolist()),
                "_owner_geometry_target_delta_m": geometry_targets,
                "_compatible_anchor_gap_m": anchor_gaps,
                **FALSE_READY,
            }
        )
    return rows_in


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
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
        "hand_far_field_temporal_refit": existing_path(
            args.hand_far_field_temporal_refit_root / case / "v17_hand_far_field_temporal_refit.json",
            f"{case} hand far-field temporal refit",
        ),
        "hand_far_field_temporal_reprojection": existing_path(
            args.hand_far_field_temporal_reprojection_root
            / case
            / "v17_hand_far_field_temporal_reprojection.json",
            f"{case} hand far-field temporal reprojection",
        ),
        "hand_temporal_reprojection_residual_owner_state": existing_path(
            args.hand_temporal_reprojection_residual_owner_state_root
            / case
            / "v17_hand_temporal_reprojection_residual_owner_state.json",
            f"{case} hand temporal reprojection residual-owner state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    for name in [
        "visible_surface",
        "hand_metric_depth_state",
        "hand_depth_repair_graph",
        "hand_far_field_temporal_refit",
        "hand_far_field_temporal_reprojection",
        "hand_temporal_reprojection_residual_owner_state",
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count disagrees with {name}")
    inputs = temporal_inputs(
        case,
        payloads["hand_far_field_temporal_reprojection"],
        payloads["hand_temporal_reprojection_residual_owner_state"],
        args,
    )
    solved_inputs, solver_report = solve_owner_weighted_rows(inputs, args)
    solved_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "source graph id"): row
        for row in solved_inputs
    }
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    repair = payloads["hand_depth_repair_graph"]
    refit = payloads["hand_far_field_temporal_refit"]
    reprojection = payloads["hand_far_field_temporal_reprojection"]
    owner = payloads["hand_temporal_reprojection_residual_owner_state"]
    depth = depth_archive(
        existing_path(
            Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")),
            "metric depth archive",
        )
    )
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    reprojection_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "temporal reprojection graph id"): row
        for row in [
            require_dict(raw, "temporal reprojection row")
            for raw in require_list(reprojection.get("rows"), "temporal reprojection rows")
        ]
    }
    refit_by_id = refit_delta_by_graph_id(refit)
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    hand_index = annotation_hand_index(frames)
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[Any, float]] = {}
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[Any, float]] = {}
    rows: list[dict[str, Any]] = []
    for raw in require_list(hand_metric.get("rows"), f"{case} hand metric rows"):
        metric_row = require_dict(raw, "hand metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
        side = require_str(metric_row.get("hand_side"), "metric row hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "metric row hand_index")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} missing annotation frame {frame_idx}")
        base = build_base_row(
            case=case,
            frame=frame,
            metric_row=metric_row,
            hand=hand_index.get((frame_idx, side, hand_i)),
            depth=depth,
            mask_cache=mask_cache,  # type: ignore[arg-type]
            args=args,
        )
        graph_id = require_str(base.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        repair_row = repair_by_id.get(graph_id)
        if repair_row is None:
            raise RuntimeError(f"{case} missing repair graph row {graph_id}")
        current_shift = repair_row.get("hand_ray_shift_m")
        refit_row = refit_by_id.get(graph_id)
        temporal_delta = None
        if refit_row is not None and refit_row.get("temporal_refit_delta_shift_m") is not None:
            temporal_delta = finite_float(refit_row.get("temporal_refit_delta_shift_m"), "temporal delta")
        solved_row = solved_by_id.get(graph_id)
        owner_delta = None
        if solved_row is not None and solved_row.get("owner_weighted_delta_shift_m") is not None:
            owner_delta = finite_float(solved_row.get("owner_weighted_delta_shift_m"), "owner delta")
        final_shift = None
        if current_shift is not None and base.get("base_available") is True:
            final_shift = finite_float(current_shift, "repair shift") + (0.0 if temporal_delta is None else temporal_delta)
            final_shift += 0.0 if owner_delta is None else owner_delta
        evaluated = (
            evaluate_row(base, None, None, eval_mask_cache, args)  # type: ignore[arg-type]
            if final_shift is None
            else evaluate_row(base, scale, final_shift, eval_mask_cache, args)  # type: ignore[arg-type]
        )
        source_temporal_row = reprojection_by_id.get(graph_id)
        source_abs_gap = None
        if source_temporal_row is not None and source_temporal_row.get("owner_median_gap_m") is not None:
            source_abs_gap = abs(finite_float(source_temporal_row.get("owner_median_gap_m"), "source gap"))
        new_abs_gap = None
        if evaluated.get("owner_median_gap_m") is not None:
            new_abs_gap = abs(finite_float(evaluated.get("owner_median_gap_m"), "new gap"))
        improved = bool(
            source_abs_gap is not None
            and new_abs_gap is not None
            and source_abs_gap - new_abs_gap >= float(args.min_depth_improvement_m)
        )
        assignment = None
        if source_temporal_row is not None and temporal_delta is not None and evaluated.get("owner_sample_partition") is not None:
            if isinstance(evaluated.get("partitions"), dict):
                samples = row_samples(evaluated)
                selected = selected_residual(evaluated, samples, args)
                assignment = assignment_pairs(evaluated, samples, selected, args)
        enriched = {
            **evaluated,
            "source_temporal_refit_variable_id": None
            if refit_row is None
            else refit_row.get("hand_far_field_temporal_refit_variable_id"),
            "source_temporal_refit_state": None if refit_row is None else refit_row.get("temporal_refit_state"),
            "source_temporal_reprojection_state": None
            if source_temporal_row is None
            else source_temporal_row.get("temporal_reprojection_state"),
            "source_temporal_reprojection_owner_median_gap_m": None
            if source_temporal_row is None
            else source_temporal_row.get("owner_median_gap_m"),
            "source_temporal_reprojection_residual_owner_state": None
            if solved_row is None
            else solved_row.get("source_temporal_reprojection_residual_owner_state"),
            "source_hand_depth_repair_graph_shift_m": current_shift,
            "temporal_refit_delta_shift_m": temporal_delta,
            "temporal_refit_delta_applied": bool(temporal_delta is not None),
            "owner_weighted_delta_shift_m": owner_delta,
            "owner_weighted_delta_applied": bool(owner_delta is not None),
            "owner_weighted_total_hand_ray_shift_m": final_shift,
            "owner_weighted_reprojected_depth_improved": improved,
            "owner_weighted_reprojection_assignment": None
            if assignment is None
            else public_assignment(assignment),
            **FALSE_READY,
        }
        rows.append(
            {
                **enriched,
                "owner_weighted_reprojection_state": output_owner_state(enriched, assignment, args),
            }
        )
    temporal_rows = [row for row in rows if row.get("source_temporal_refit_state") is not None]
    applied_rows = [row for row in temporal_rows if row.get("owner_weighted_delta_applied") is True]
    residual_rows = [
        row
        for row in applied_rows
        if row["owner_weighted_reprojection_state"]
        not in {
            "owner_weighted_reprojected_metric_depth_compatible",
            "owner_weighted_reprojected_projection_untrusted",
            "owner_weighted_reprojected_unobserved",
        }
    ]
    report = {
        "method": "solve_v17_hand_temporal_owner_weighted_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_temporal_owner_weighted_refit_variable_count": len(inputs),
        "owner_weighted_temporal_source_rows": len(temporal_rows),
        "owner_weighted_variable_rows": bool_count(solved_inputs, "owner_weighted_variable_candidate"),
        "owner_weighted_geometry_factor_rows": bool_count(
            solved_inputs,
            "owner_weighted_geometry_factor_row",
        ),
        "owner_weighted_compatible_anchor_rows": bool_count(
            solved_inputs,
            "owner_weighted_compatible_anchor_row",
        ),
        "owner_weighted_prior_smooth_only_rows": bool_count(
            solved_inputs,
            "owner_weighted_prior_smooth_only_row",
        ),
        "owner_weighted_depth_observation_prior_smooth_rows": bool_count(
            solved_inputs,
            "owner_weighted_depth_observation_prior_smooth_row",
        ),
        "owner_weighted_projection_untrusted_prior_smooth_rows": bool_count(
            solved_inputs,
            "owner_weighted_projection_untrusted_prior_smooth_row",
        ),
        "owner_weighted_geometry_depth_sample_factor_count": solver_report[
            "geometry_depth_sample_factor_count"
        ],
        "owner_weighted_compatible_anchor_sample_factor_count": solver_report[
            "compatible_anchor_sample_factor_count"
        ],
        "owner_weighted_delta_bound_hit_rows": bool_count(
            solved_inputs,
            "owner_weighted_delta_bound_hit",
        ),
        "owner_weighted_fixed_factor_depth_improved_rows": bool_count(
            solved_inputs,
            "owner_weighted_fixed_factor_depth_improved",
        ),
        "owner_weighted_fixed_factor_depth_threshold_met_rows": bool_count(
            solved_inputs,
            "owner_weighted_fixed_factor_depth_threshold_met",
        ),
        "owner_weighted_reprojected_metric_depth_compatible_rows": bool_count(
            applied_rows,
            "metric_depth_compatible",
        ),
        "owner_weighted_reprojected_depth_improved_rows": bool_count(
            applied_rows,
            "owner_weighted_reprojected_depth_improved",
        ),
        "metric_hand_state_accepted_rows_after_owner_weighted_refit": bool_count(
            rows,
            "metric_depth_compatible",
        ),
        "depth_repair_factor_candidate_rows_after_owner_weighted_refit": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "owner_weighted_reprojection_residual_owner_rows": len(residual_rows),
        "owner_weighted_reprojection_local_surface_factor_candidate_rows": sum(
            1
            for row in residual_rows
            if row["owner_weighted_reprojection_state"]
            == "owner_weighted_reprojected_local_surface_factor_candidate"
        ),
        "owner_weighted_reprojection_mixed_surface_depth_owner_rows": sum(
            1
            for row in residual_rows
            if row["owner_weighted_reprojection_state"]
            == "owner_weighted_reprojected_mixed_surface_depth_owner"
        ),
        "owner_weighted_reprojection_depth_observation_owner_rows": sum(
            1
            for row in residual_rows
            if row["owner_weighted_reprojection_state"]
            == "owner_weighted_reprojected_depth_observation_owner"
        ),
        "owner_weighted_reprojection_projection_untrusted_rows": sum(
            1
            for row in applied_rows
            if row["owner_weighted_reprojection_state"]
            == "owner_weighted_reprojected_projection_untrusted"
        ),
        "owner_weighted_input_factor_state_counts": state_counts(
            solved_inputs,
            "owner_weighted_factor_state",
        ),
        "owner_weighted_reprojection_state_counts": state_counts(
            rows,
            "owner_weighted_reprojection_state",
        ),
        "owner_weighted_temporal_reprojection_state_counts": state_counts(
            temporal_rows,
            "owner_weighted_reprojection_state",
        ),
        "owner_weighted_owner_depth_state_counts_after_reprojection": state_counts(
            rows,
            "owner_depth_state",
        ),
        "owner_weighted_owner_median_gap_m_after_reprojection": numeric_summary(
            rows,
            "owner_median_gap_m",
        ),
        "solver": solver_report,
        "source_temporal_reprojection_comparison": {
            "temporal_refit_source_rows": reprojection.get("temporal_refit_source_rows"),
            "temporal_refit_delta_applied_rows": reprojection.get("temporal_refit_delta_applied_rows"),
            "temporal_refit_reprojected_metric_depth_compatible_rows": reprojection.get(
                "temporal_refit_reprojected_metric_depth_compatible_rows"
            ),
            "temporal_refit_reprojected_depth_improved_rows": reprojection.get(
                "temporal_refit_reprojected_depth_improved_rows"
            ),
            "metric_hand_state_accepted_rows_after_temporal_reprojection": reprojection.get(
                "metric_hand_state_accepted_rows_after_temporal_reprojection"
            ),
            "depth_repair_factor_candidate_rows_after_temporal_reprojection": reprojection.get(
                "depth_repair_factor_candidate_rows_after_temporal_reprojection"
            ),
            "temporal_refit_reprojection_state_counts": reprojection.get(
                "temporal_refit_reprojection_state_counts"
            ),
        },
        "source_temporal_residual_owner_comparison": {
            "temporal_reprojection_residual_owner_rows": owner.get(
                "temporal_reprojection_residual_owner_rows"
            ),
            "temporal_reprojection_local_surface_factor_candidate_rows": owner.get(
                "temporal_reprojection_local_surface_factor_candidate_rows"
            ),
            "temporal_reprojection_mixed_surface_depth_owner_rows": owner.get(
                "temporal_reprojection_mixed_surface_depth_owner_rows"
            ),
            "temporal_reprojection_depth_observation_owner_rows": owner.get(
                "temporal_reprojection_depth_observation_owner_rows"
            ),
            "temporal_reprojection_projection_untrusted_rows": owner.get(
                "temporal_reprojection_projection_untrusted_rows"
            ),
            "applied_temporal_reprojection_residual_owner_state_counts": owner.get(
                "applied_temporal_reprojection_residual_owner_state_counts"
            ),
        },
        "problem_semantics": {
            "owner_weighted_geometry_factor_variable": (
                "a local or mixed residual-owner row supplies residual-to-compatible-depth sample pairs"
            ),
            "owner_weighted_depth_observation_prior_smooth_variable": (
                "the row remains an explicit temporal depth variable but its residual pixels are owned by the depth-observation state"
            ),
            "owner_weighted_reprojected_metric_depth_compatible": (
                "the owner-weighted update survives MANO reprojection and UniDepth resampling"
            ),
            "claim_limit": (
                "this is still a ray-depth relinearization test; MANO articulation, shape, object geometry, and contact are not optimized"
            ),
        },
        "solver_rows": solved_inputs,
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_temporal_owner_weighted_refit.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = ["trash_1050", "task5_tomato_960"]
    reports = [case_problem(case, args) for case in cases]
    summary = {
        "method": "solve_v17_hand_temporal_owner_weighted_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "cases": cases,
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "hand_temporal_owner_weighted_refit_variable_count": sum(
            require_int(
                report.get("hand_temporal_owner_weighted_refit_variable_count"),
                "owner-weighted variable count",
            )
            for report in reports
        ),
        "owner_weighted_temporal_source_rows": sum(
            require_int(report.get("owner_weighted_temporal_source_rows"), "temporal source rows")
            for report in reports
        ),
        "owner_weighted_variable_rows": sum(
            require_int(report.get("owner_weighted_variable_rows"), "owner-weighted variables")
            for report in reports
        ),
        "owner_weighted_geometry_factor_rows": sum(
            require_int(report.get("owner_weighted_geometry_factor_rows"), "geometry factor rows")
            for report in reports
        ),
        "owner_weighted_compatible_anchor_rows": sum(
            require_int(report.get("owner_weighted_compatible_anchor_rows"), "compatible anchor rows")
            for report in reports
        ),
        "owner_weighted_prior_smooth_only_rows": sum(
            require_int(report.get("owner_weighted_prior_smooth_only_rows"), "prior only rows")
            for report in reports
        ),
        "owner_weighted_depth_observation_prior_smooth_rows": sum(
            require_int(
                report.get("owner_weighted_depth_observation_prior_smooth_rows"),
                "depth prior rows",
            )
            for report in reports
        ),
        "owner_weighted_projection_untrusted_prior_smooth_rows": sum(
            require_int(
                report.get("owner_weighted_projection_untrusted_prior_smooth_rows"),
                "projection prior rows",
            )
            for report in reports
        ),
        "owner_weighted_geometry_depth_sample_factor_count": sum(
            require_int(
                report.get("owner_weighted_geometry_depth_sample_factor_count"),
                "geometry factor samples",
            )
            for report in reports
        ),
        "owner_weighted_compatible_anchor_sample_factor_count": sum(
            require_int(
                report.get("owner_weighted_compatible_anchor_sample_factor_count"),
                "anchor factor samples",
            )
            for report in reports
        ),
        "owner_weighted_delta_bound_hit_rows": sum(
            require_int(report.get("owner_weighted_delta_bound_hit_rows"), "bound hit rows")
            for report in reports
        ),
        "owner_weighted_fixed_factor_depth_improved_rows": sum(
            require_int(
                report.get("owner_weighted_fixed_factor_depth_improved_rows"),
                "fixed factor improved rows",
            )
            for report in reports
        ),
        "owner_weighted_fixed_factor_depth_threshold_met_rows": sum(
            require_int(
                report.get("owner_weighted_fixed_factor_depth_threshold_met_rows"),
                "fixed factor threshold rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojected_metric_depth_compatible_rows": sum(
            require_int(
                report.get("owner_weighted_reprojected_metric_depth_compatible_rows"),
                "reprojected compatible rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojected_depth_improved_rows": sum(
            require_int(
                report.get("owner_weighted_reprojected_depth_improved_rows"),
                "reprojected improved rows",
            )
            for report in reports
        ),
        "metric_hand_state_accepted_rows_after_owner_weighted_refit": sum(
            require_int(
                report.get("metric_hand_state_accepted_rows_after_owner_weighted_refit"),
                "accepted rows",
            )
            for report in reports
        ),
        "depth_repair_factor_candidate_rows_after_owner_weighted_refit": sum(
            require_int(
                report.get("depth_repair_factor_candidate_rows_after_owner_weighted_refit"),
                "residual rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojection_residual_owner_rows": sum(
            require_int(
                report.get("owner_weighted_reprojection_residual_owner_rows"),
                "owner-weighted residual rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(
                report.get("owner_weighted_reprojection_local_surface_factor_candidate_rows"),
                "owner-weighted local rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(
                report.get("owner_weighted_reprojection_mixed_surface_depth_owner_rows"),
                "owner-weighted mixed rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojection_depth_observation_owner_rows": sum(
            require_int(
                report.get("owner_weighted_reprojection_depth_observation_owner_rows"),
                "owner-weighted depth rows",
            )
            for report in reports
        ),
        "owner_weighted_reprojection_projection_untrusted_rows": sum(
            require_int(
                report.get("owner_weighted_reprojection_projection_untrusted_rows"),
                "owner-weighted projection rows",
            )
            for report in reports
        ),
        "owner_weighted_input_factor_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("owner_weighted_input_factor_state_counts"), "factor counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "owner_weighted_temporal_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("owner_weighted_temporal_reprojection_state_counts"),
                                "temporal state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "source_temporal_refit_delta_applied_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_reprojection_comparison"),
                    "source temporal comparison",
                ).get("temporal_refit_delta_applied_rows"),
                "source temporal applied rows",
            )
            for report in reports
        ),
        "source_temporal_reprojected_metric_depth_compatible_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_reprojection_comparison"),
                    "source temporal comparison",
                ).get("temporal_refit_reprojected_metric_depth_compatible_rows"),
                "source temporal compatible rows",
            )
            for report in reports
        ),
        "source_metric_hand_state_accepted_rows_after_temporal_reprojection": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_reprojection_comparison"),
                    "source temporal comparison",
                ).get("metric_hand_state_accepted_rows_after_temporal_reprojection"),
                "source temporal accepted rows",
            )
            for report in reports
        ),
        "source_depth_repair_factor_candidate_rows_after_temporal_reprojection": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_reprojection_comparison"),
                    "source temporal comparison",
                ).get("depth_repair_factor_candidate_rows_after_temporal_reprojection"),
                "source temporal residual rows",
            )
            for report in reports
        ),
        "source_temporal_reprojection_residual_owner_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_residual_owner_comparison"),
                    "source temporal owner comparison",
                ).get("temporal_reprojection_residual_owner_rows"),
                "source temporal residual-owner rows",
            )
            for report in reports
        ),
        "source_temporal_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_residual_owner_comparison"),
                    "source temporal owner comparison",
                ).get("temporal_reprojection_local_surface_factor_candidate_rows"),
                "source temporal local rows",
            )
            for report in reports
        ),
        "source_temporal_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_residual_owner_comparison"),
                    "source temporal owner comparison",
                ).get("temporal_reprojection_mixed_surface_depth_owner_rows"),
                "source temporal mixed rows",
            )
            for report in reports
        ),
        "source_temporal_reprojection_depth_observation_owner_rows": sum(
            require_int(
                require_dict(
                    report.get("source_temporal_residual_owner_comparison"),
                    "source temporal owner comparison",
                ).get("temporal_reprojection_depth_observation_owner_rows"),
                "source temporal depth rows",
            )
            for report in reports
        ),
        "case_reports": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "owner_weighted_variable_rows": require_int(
                    report.get("owner_weighted_variable_rows"),
                    "owner-weighted variables",
                ),
                "owner_weighted_geometry_factor_rows": require_int(
                    report.get("owner_weighted_geometry_factor_rows"),
                    "geometry rows",
                ),
                "owner_weighted_reprojected_metric_depth_compatible_rows": require_int(
                    report.get("owner_weighted_reprojected_metric_depth_compatible_rows"),
                    "compatible rows",
                ),
                "metric_hand_state_accepted_rows_after_owner_weighted_refit": require_int(
                    report.get("metric_hand_state_accepted_rows_after_owner_weighted_refit"),
                    "accepted rows",
                ),
                "depth_repair_factor_candidate_rows_after_owner_weighted_refit": require_int(
                    report.get("depth_repair_factor_candidate_rows_after_owner_weighted_refit"),
                    "residual rows",
                ),
                "owner_weighted_temporal_reprojection_state_counts": require_dict(
                    report.get("owner_weighted_temporal_reprojection_state_counts"),
                    "temporal state counts",
                ),
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_temporal_owner_weighted_refit_summary.json", summary)
    return summary


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
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-far-field-temporal-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit"),
    )
    parser.add_argument(
        "--hand-far-field-temporal-reprojection-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection"),
    )
    parser.add_argument(
        "--hand-temporal-reprojection-residual-owner-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_reprojection_residual_owner_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--max-depth-samples-per-row", type=int, default=48)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--min-local-projection-candidate-fraction", type=float, default=0.75)
    parser.add_argument("--min-mixed-projection-depth-fraction", type=float, default=0.25)
    parser.add_argument("--min-owner-weighted-factor-samples", type=int, default=3)
    parser.add_argument("--max-factor-samples-per-row", type=int, default=64)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-owner-weighted-delta-m", type=float, default=0.12)
    parser.add_argument("--min-corrected-hand-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-owner-geometry-depth-m", type=float, default=0.02)
    parser.add_argument("--sigma-owner-compatible-anchor-m", type=float, default=0.03)
    parser.add_argument("--sigma-owner-delta-prior-m", type=float, default=0.08)
    parser.add_argument("--sigma-owner-delta-step-m", type=float, default=0.03)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=45)
    parser.add_argument("--min-depth-improvement-m", type=float, default=0.005)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--max-solver-iter", type=int, default=500)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
