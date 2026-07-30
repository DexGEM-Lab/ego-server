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

from apply_v17_hand_far_field_temporal_refit import finite_float
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
from build_v17_hand_temporal_reprojection_residual_owner_state import temporal_owner_state
from solve_v17_hand_depth_repair_graph import build_base_row, evaluate_row, numeric_summary
from solve_v17_hand_temporal_owner_weighted_refit import (
    add_row,
    assignment_pairs,
    compatible_anchor_gaps,
    public_assignment,
    thin,
)


STATUS = "v17_post_temporal_depth_observation_weighted_refit_qc"
CLAIM = (
    "This artifact tests a support-weighted hand-depth observation graph after the owner-weighted "
    "temporal refit. Same-side keypoint-supported depth-observation rows contribute UniDepth residual "
    "factors; sparse, absent, unsupported, and projection-untrusted rows remain explicit prior/smoothness "
    "variables. The output is a causal diagnostic, not annotation closure."
)

LOCAL_STATE = "owner_weighted_reprojected_local_surface_factor_candidate"
MIXED_STATE = "owner_weighted_reprojected_mixed_surface_depth_owner"
DEPTH_STATE = "owner_weighted_reprojected_depth_observation_owner"
UNTRUSTED_STATE = "owner_weighted_reprojected_projection_untrusted"
COMPATIBLE_STATE = "owner_weighted_reprojected_metric_depth_compatible"

OBSERVATION_FACTOR_STATES = {
    "same_side_independent_keypoint_partial",
    "same_side_independent_keypoint_strong",
}
SAME_SIDE_BOX_SUPPORT_STATES = {
    "tail_pixels_inside_same_side_independent_model_box",
    "tail_pixels_near_same_side_independent_model_box",
}


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def keypoint_sigma(state: str, args: argparse.Namespace) -> float:
    if state == "same_side_independent_keypoint_strong":
        return float(args.sigma_depth_observation_strong_m)
    if state == "same_side_independent_keypoint_partial":
        return float(args.sigma_depth_observation_partial_m)
    raise RuntimeError(f"keypoint state has no depth-observation sigma: {state}")


def depth_observation_targets(row: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    samples = row_samples(row)
    selected = selected_residual(row, samples, args)
    hand_z = np.asarray(samples["hand_z"], dtype=np.float64)
    metric_z = np.asarray(samples["metric_z"], dtype=np.float64)
    target = metric_z[selected] - hand_z[selected]
    return thin(target.astype(np.float64), int(args.max_depth_observation_samples_per_row))


def geometry_targets(row: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any] | None]:
    samples = row_samples(row)
    selected = selected_residual(row, samples, args)
    assignment = assignment_pairs(row, samples, selected, args)
    target = np.asarray(assignment["_source_to_seed_delta_m"], dtype=np.float64)
    return thin(target, int(args.max_factor_samples_per_row)), public_assignment(assignment)


def input_factor_state(row: dict[str, Any]) -> str:
    if row.get("post_temporal_observation_variable_candidate") is not True:
        return require_str(row.get("post_temporal_observation_candidate_rejection"), "candidate rejection")
    if row.get("post_temporal_geometry_factor_row") is True:
        return "post_temporal_geometry_factor_variable"
    if row.get("post_temporal_depth_observation_factor_row") is True:
        support_state = require_str(row.get("independent_keypoint_support_state"), "keypoint support state")
        if support_state == "same_side_independent_keypoint_strong":
            return "post_temporal_depth_observation_strong_keypoint_factor_variable"
        if support_state == "same_side_independent_keypoint_partial":
            return "post_temporal_depth_observation_partial_keypoint_factor_variable"
        return "post_temporal_depth_observation_unexpected_keypoint_factor_variable"
    if row.get("post_temporal_compatible_anchor_row") is True:
        return "post_temporal_compatible_anchor_variable"
    owner = require_str(row.get("source_owner_weighted_reprojection_state"), "source owner state")
    if owner == DEPTH_STATE:
        support_state = require_str(row.get("independent_keypoint_support_state"), "keypoint support state")
        return f"post_temporal_depth_observation_{support_state}_prior_smooth_variable"
    if owner == UNTRUSTED_STATE:
        return "post_temporal_projection_untrusted_prior_smooth_variable"
    if owner == COMPATIBLE_STATE:
        return "post_temporal_compatible_without_anchor_prior_smooth_variable"
    return "post_temporal_sparse_owner_prior_smooth_variable"


def post_temporal_inputs(
    case: str,
    owner_weighted: dict[str, Any],
    support: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    support_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "support graph id"): row
        for row in [
            require_dict(raw, "post-temporal support row")
            for raw in require_list(support.get("rows"), "post-temporal support rows")
        ]
    }
    rows_in: list[dict[str, Any]] = []
    for raw in require_list(owner_weighted.get("rows"), f"{case} owner-weighted rows"):
        row = require_dict(raw, "owner-weighted row")
        if row.get("source_temporal_refit_state") is None:
            continue
        graph_id = require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        source_state = require_str(row.get("owner_weighted_reprojection_state"), "owner-weighted state")
        current_shift_raw = row.get("owner_weighted_total_hand_ray_shift_m")
        current_shift = None
        if current_shift_raw is not None:
            current_shift = finite_float(current_shift_raw, "owner-weighted total shift")
        candidate = bool(row.get("owner_weighted_delta_applied") is True and current_shift is not None)
        rejection = None
        if not candidate:
            rejection = "post_temporal_no_owner_weighted_shift"
        lower = None
        upper = None
        if candidate:
            current = finite_float(current_shift, "current owner-weighted shift")
            lower = max(
                -float(args.max_abs_hand_ray_shift_m) - current,
                -float(args.max_abs_post_temporal_delta_m),
            )
            upper = min(
                float(args.max_abs_hand_ray_shift_m) - current,
                float(args.max_abs_post_temporal_delta_m),
            )
            if lower > upper:
                candidate = False
                rejection = "post_temporal_inconsistent_shift_bounds"
        geometry_target = np.asarray([], dtype=np.float64)
        anchor_gap = np.asarray([], dtype=np.float64)
        observation_target = np.asarray([], dtype=np.float64)
        assignment = None
        support_row = support_by_id.get(graph_id)
        keypoint_state = "same_side_independent_keypoints_unmeasured"
        box_state = None
        selected_residual_count = 0
        if candidate and source_state in {LOCAL_STATE, MIXED_STATE}:
            geometry_target, assignment = geometry_targets(row, args)
        elif candidate and source_state == COMPATIBLE_STATE:
            samples = row_samples(row)
            anchor_gap = thin(
                compatible_anchor_gaps(row, samples, args),
                int(args.max_factor_samples_per_row),
            )
        elif candidate and source_state == DEPTH_STATE:
            if support_row is None:
                raise RuntimeError(f"{case} missing post-temporal support row for {graph_id}")
            keypoint_state = require_str(
                support_row.get("independent_keypoint_support_state"),
                "independent keypoint support state",
            )
            box_state = support_row.get("independent_support_state")
            selected_residual_count = require_int(
                support_row.get("selected_residual_sample_count"),
                "selected residual sample count",
            )
            if box_state in SAME_SIDE_BOX_SUPPORT_STATES and keypoint_state in OBSERVATION_FACTOR_STATES:
                observation_target = depth_observation_targets(row, args)
        geometry_factor_row = bool(
            candidate and geometry_target.size >= int(args.min_post_temporal_factor_samples)
        )
        anchor_row = bool(candidate and anchor_gap.size >= int(args.min_post_temporal_factor_samples))
        observation_factor_row = bool(
            candidate and observation_target.size >= int(args.min_post_temporal_factor_samples)
        )
        rows_in.append(
            {
                "case": case,
                "post_temporal_depth_observation_weighted_refit_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "post_temporal_depth_observation_weighted_refit:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_temporal_owner_weighted_refit_variable_id": row.get(
                    "hand_temporal_owner_weighted_refit_variable_id"
                ),
                "source_temporal_refit_state": row.get("source_temporal_refit_state"),
                "source_owner_weighted_reprojection_state": source_state,
                "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(row.get("hand_side"), "hand_side"),
                "hand_index": require_int(row.get("hand_index"), "hand_index"),
                "owner_sample_partition": row.get("owner_sample_partition"),
                "owner_depth_state": row.get("owner_depth_state"),
                "current_owner_weighted_hand_ray_shift_m": current_shift,
                "post_temporal_observation_variable_candidate": candidate,
                "post_temporal_observation_candidate_rejection": rejection,
                "post_temporal_geometry_factor_row": geometry_factor_row,
                "post_temporal_compatible_anchor_row": anchor_row,
                "post_temporal_depth_observation_factor_row": observation_factor_row,
                "post_temporal_prior_smooth_only_row": bool(
                    candidate and not geometry_factor_row and not anchor_row and not observation_factor_row
                ),
                "post_temporal_depth_observation_prior_smooth_row": bool(
                    candidate and source_state == DEPTH_STATE and not observation_factor_row
                ),
                "post_temporal_projection_untrusted_prior_smooth_row": bool(
                    candidate and source_state == UNTRUSTED_STATE
                ),
                "independent_support_state": box_state,
                "independent_keypoint_support_state": keypoint_state,
                "support_selected_residual_sample_count": selected_residual_count,
                "post_temporal_delta_lower_bound_m": lower,
                "post_temporal_delta_upper_bound_m": upper,
                "post_temporal_geometry_assignment": assignment,
                "post_temporal_geometry_target_delta_m": summarize(geometry_target.astype(float).tolist()),
                "post_temporal_compatible_anchor_gap_m": summarize(anchor_gap.astype(float).tolist()),
                "post_temporal_depth_observation_target_delta_m": summarize(
                    observation_target.astype(float).tolist()
                ),
                "_geometry_target_delta_m": geometry_target,
                "_compatible_anchor_gap_m": anchor_gap,
                "_depth_observation_target_delta_m": observation_target,
                **FALSE_READY,
            }
        )
    return rows_in


def solve_post_temporal_rows(
    rows_in: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_indices = [
        i for i, row in enumerate(rows_in) if row.get("post_temporal_observation_variable_candidate") is True
    ]
    if not candidate_indices:
        rows_out = [
            {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "post_temporal_observation_delta_shift_m": None,
                "post_temporal_observation_total_hand_ray_shift_m": row.get(
                    "current_owner_weighted_hand_ray_shift_m"
                ),
                "post_temporal_observation_delta_bound_hit": False,
                "post_temporal_observation_fixed_factor_depth_improved": False,
                "post_temporal_observation_fixed_factor_depth_threshold_met": False,
                "post_temporal_observation_factor_state": input_factor_state(row),
            }
            for row in rows_in
        ]
        return rows_out, {
            "variable_count": 0,
            "geometry_depth_sample_factor_count": 0,
            "compatible_anchor_sample_factor_count": 0,
            "depth_observation_sample_factor_count": 0,
            "prior_factor_count": 0,
            "smoothness_factor_count": 0,
            "matrix_rows": 0,
            "matrix_cols": 0,
            "solver_skipped_reason": "no_post_temporal_observation_variables",
        }
    var_by_row = {source_i: var_i for var_i, source_i in enumerate(candidate_indices)}
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_vals: list[float] = []
    rhs: list[float] = []
    row_i = 0
    geometry_factors = 0
    anchor_factors = 0
    observation_factors = 0
    prior_factors = 0
    smooth_factors = 0
    sigma_geometry = float(args.sigma_post_temporal_geometry_depth_m)
    sigma_anchor = float(args.sigma_post_temporal_compatible_anchor_m)
    sigma_prior = float(args.sigma_post_temporal_delta_prior_m)
    sigma_step = float(args.sigma_post_temporal_delta_step_m)
    lower = np.full(len(candidate_indices), -float(args.max_abs_post_temporal_delta_m), dtype=np.float64)
    upper = np.full(len(candidate_indices), float(args.max_abs_post_temporal_delta_m), dtype=np.float64)
    fixed_before_by_row: dict[int, np.ndarray] = {}
    for source_i in candidate_indices:
        var_i = var_by_row[source_i]
        row = rows_in[source_i]
        lower[var_i] = max(
            lower[var_i],
            finite_float(row.get("post_temporal_delta_lower_bound_m"), "lower bound"),
        )
        upper[var_i] = min(
            upper[var_i],
            finite_float(row.get("post_temporal_delta_upper_bound_m"), "upper bound"),
        )
        geometry_target = thin(
            np.asarray(row.get("_geometry_target_delta_m"), dtype=np.float64),
            int(args.max_factor_samples_per_row),
        )
        anchor_gap = thin(
            np.asarray(row.get("_compatible_anchor_gap_m"), dtype=np.float64),
            int(args.max_factor_samples_per_row),
        )
        observation_target = thin(
            np.asarray(row.get("_depth_observation_target_delta_m"), dtype=np.float64),
            int(args.max_depth_observation_samples_per_row),
        )
        fixed_before_by_row[source_i] = np.concatenate([(-geometry_target), anchor_gap, (-observation_target)])
        for target_delta in geometry_target:
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
        for gap in anchor_gap:
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
        if observation_target.size:
            support_state = require_str(row.get("independent_keypoint_support_state"), "support state")
            sigma_observation = keypoint_sigma(support_state, args)
            for target_delta in observation_target:
                row_i = add_row(
                    matrix_rows,
                    matrix_cols,
                    matrix_vals,
                    rhs,
                    row_i,
                    [(var_i, 1.0 / sigma_observation)],
                    float(target_delta) / sigma_observation,
                )
                observation_factors += 1
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
        raise RuntimeError(f"inconsistent post-temporal observation bounds at variables {bad[:12].tolist()}")
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
        raise RuntimeError("post-temporal observation refit returned invalid solution")
    rows_out: list[dict[str, Any]] = []
    for source_i, row in enumerate(rows_in):
        base = {k: v for k, v in row.items() if not k.startswith("_")}
        if source_i not in var_by_row:
            out = {
                **base,
                "post_temporal_observation_delta_shift_m": None,
                "post_temporal_observation_total_hand_ray_shift_m": row.get(
                    "current_owner_weighted_hand_ray_shift_m"
                ),
                "post_temporal_observation_delta_bound_hit": False,
                "post_temporal_observation_fixed_factor_depth_improved": False,
                "post_temporal_observation_fixed_factor_depth_threshold_met": False,
            }
            rows_out.append({**out, "post_temporal_observation_factor_state": input_factor_state(out)})
            continue
        var_i = var_by_row[source_i]
        delta = float(result.x[var_i])
        current_shift = finite_float(row.get("current_owner_weighted_hand_ray_shift_m"), "current shift")
        fixed_before = fixed_before_by_row[source_i]
        fixed_after = fixed_before + delta
        before_abs_median = float(np.median(np.abs(fixed_before))) if len(fixed_before) else math.inf
        after_abs_median = float(np.median(np.abs(fixed_after))) if len(fixed_after) else math.inf
        after_abs_p95 = float(np.percentile(np.abs(fixed_after), 95.0)) if len(fixed_after) else math.inf
        lower_i = finite_float(row.get("post_temporal_delta_lower_bound_m"), "lower bound")
        upper_i = finite_float(row.get("post_temporal_delta_upper_bound_m"), "upper bound")
        out = {
            **base,
            "post_temporal_observation_delta_shift_m": delta,
            "post_temporal_observation_total_hand_ray_shift_m": current_shift + delta,
            "post_temporal_observation_fixed_factor_residual": {
                "before": {
                    "signed_gap_m": summarize(fixed_before.astype(float).tolist()),
                    "abs_gap_m": summarize(np.abs(fixed_before).astype(float).tolist()),
                },
                "after": {
                    "signed_gap_m": summarize(fixed_after.astype(float).tolist()),
                    "abs_gap_m": summarize(np.abs(fixed_after).astype(float).tolist()),
                },
            },
            "post_temporal_observation_delta_bound_hit": bool(
                math.isclose(delta, lower_i, abs_tol=float(args.bound_tolerance_m))
                or math.isclose(delta, upper_i, abs_tol=float(args.bound_tolerance_m))
            ),
            "post_temporal_observation_fixed_factor_depth_improved": bool(
                before_abs_median - after_abs_median >= float(args.min_depth_improvement_m)
            ),
            "post_temporal_observation_fixed_factor_depth_threshold_met": bool(
                after_abs_median <= float(args.max_median_abs_depth_gap_m)
                and after_abs_p95 <= float(args.max_p95_abs_depth_gap_m)
            )
            if len(fixed_after)
            else False,
        }
        rows_out.append({**out, "post_temporal_observation_factor_state": input_factor_state(out)})
    return rows_out, {
        "variable_count": len(candidate_indices),
        "geometry_depth_sample_factor_count": geometry_factors,
        "compatible_anchor_sample_factor_count": anchor_factors,
        "depth_observation_sample_factor_count": observation_factors,
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


def output_state(row: dict[str, Any], assignment: dict[str, Any] | None, args: argparse.Namespace) -> str:
    if row.get("source_temporal_refit_state") is None:
        return "not_post_temporal_observation_row"
    if row.get("post_temporal_observation_delta_applied") is not True:
        return "post_temporal_observation_delta_not_applied"
    if row.get("owner_sample_partition") is None or not isinstance(row.get("partitions"), dict):
        return "post_temporal_observation_reprojected_unobserved"
    if assignment is None:
        raise RuntimeError("applied post-temporal observation row needs an assignment state")
    temporal_state = temporal_owner_state(row, assignment, args)
    mapping = {
        "temporal_reprojection_metric_depth_compatible": "post_temporal_observation_reprojected_metric_depth_compatible",
        "temporal_reprojection_projection_untrusted": "post_temporal_observation_reprojected_projection_untrusted",
        "temporal_reprojection_residual_unobserved": "post_temporal_observation_reprojected_residual_unobserved",
        "temporal_reprojection_local_surface_factor_candidate": "post_temporal_observation_reprojected_local_surface_factor_candidate",
        "temporal_reprojection_mixed_surface_depth_owner": "post_temporal_observation_reprojected_mixed_surface_depth_owner",
        "temporal_reprojection_depth_observation_owner": "post_temporal_observation_reprojected_depth_observation_owner",
        "temporal_delta_not_applied": "post_temporal_observation_delta_not_applied",
    }
    if temporal_state not in mapping:
        raise RuntimeError(f"unknown post-temporal observation owner state: {temporal_state}")
    return mapping[temporal_state]


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
        "hand_temporal_owner_weighted_refit": existing_path(
            args.hand_temporal_owner_weighted_refit_root
            / case
            / "v17_hand_temporal_owner_weighted_refit.json",
            f"{case} hand temporal owner-weighted refit",
        ),
        "post_temporal_depth_observation_support": existing_path(
            args.post_temporal_depth_observation_support_state_root
            / case
            / "v17_post_temporal_depth_observation_support_state.json",
            f"{case} post-temporal depth-observation support state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    for name in [
        "visible_surface",
        "hand_metric_depth_state",
        "hand_depth_repair_graph",
        "hand_temporal_owner_weighted_refit",
        "post_temporal_depth_observation_support",
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count disagrees with {name}")
    inputs = post_temporal_inputs(
        case,
        payloads["hand_temporal_owner_weighted_refit"],
        payloads["post_temporal_depth_observation_support"],
        args,
    )
    solved_inputs, solver_report = solve_post_temporal_rows(inputs, args)
    solved_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "source graph id"): row
        for row in solved_inputs
    }
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    repair = payloads["hand_depth_repair_graph"]
    owner_weighted = payloads["hand_temporal_owner_weighted_refit"]
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
    owner_weighted_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "owner-weighted graph id"): row
        for row in [
            require_dict(raw, "owner-weighted row")
            for raw in require_list(owner_weighted.get("rows"), "owner-weighted rows")
        ]
    }
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    hand_index = annotation_hand_index(frames)
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
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
            mask_cache=mask_cache,
            args=args,
        )
        graph_id = require_str(base.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        if graph_id not in repair_by_id:
            raise RuntimeError(f"{case} missing repair graph row {graph_id}")
        owner_row = owner_weighted_by_id.get(graph_id)
        current_shift = None
        if owner_row is not None and owner_row.get("owner_weighted_total_hand_ray_shift_m") is not None:
            current_shift = finite_float(owner_row.get("owner_weighted_total_hand_ray_shift_m"), "current shift")
        solved_row = solved_by_id.get(graph_id)
        delta = None
        if solved_row is not None and solved_row.get("post_temporal_observation_delta_shift_m") is not None:
            delta = finite_float(solved_row.get("post_temporal_observation_delta_shift_m"), "post delta")
        final_shift = None
        if current_shift is not None and base.get("base_available") is True:
            final_shift = current_shift + (0.0 if delta is None else delta)
        evaluated = (
            evaluate_row(base, None, None, eval_mask_cache, args)
            if final_shift is None
            else evaluate_row(base, scale, final_shift, eval_mask_cache, args)
        )
        source_abs_gap = None
        if owner_row is not None and owner_row.get("owner_median_gap_m") is not None:
            source_abs_gap = abs(finite_float(owner_row.get("owner_median_gap_m"), "source gap"))
        new_abs_gap = None
        if evaluated.get("owner_median_gap_m") is not None:
            new_abs_gap = abs(finite_float(evaluated.get("owner_median_gap_m"), "new gap"))
        improved = bool(
            source_abs_gap is not None
            and new_abs_gap is not None
            and source_abs_gap - new_abs_gap >= float(args.min_depth_improvement_m)
        )
        assignment = None
        if (
            owner_row is not None
            and solved_row is not None
            and solved_row.get("post_temporal_observation_delta_shift_m") is not None
            and evaluated.get("owner_sample_partition") is not None
            and isinstance(evaluated.get("partitions"), dict)
        ):
            samples = row_samples(evaluated)
            selected = selected_residual(evaluated, samples, args)
            assignment = assignment_pairs(evaluated, samples, selected, args)
        enriched = {
            **evaluated,
            "source_hand_temporal_owner_weighted_refit_variable_id": None
            if owner_row is None
            else owner_row.get("hand_temporal_owner_weighted_refit_variable_id"),
            "source_temporal_refit_state": None if owner_row is None else owner_row.get("source_temporal_refit_state"),
            "source_owner_weighted_reprojection_state": None
            if owner_row is None
            else owner_row.get("owner_weighted_reprojection_state"),
            "source_owner_weighted_owner_median_gap_m": None
            if owner_row is None
            else owner_row.get("owner_median_gap_m"),
            "source_owner_weighted_total_hand_ray_shift_m": current_shift,
            "post_temporal_observation_delta_shift_m": delta,
            "post_temporal_observation_delta_applied": bool(delta is not None),
            "temporal_refit_delta_applied": bool(delta is not None),
            "post_temporal_observation_total_hand_ray_shift_m": final_shift,
            "post_temporal_observation_reprojected_depth_improved": improved,
            "post_temporal_observation_reprojection_assignment": None
            if assignment is None
            else public_assignment(assignment),
            **FALSE_READY,
        }
        rows.append(
            {
                **enriched,
                "post_temporal_observation_reprojection_state": output_state(enriched, assignment, args),
            }
        )
    temporal_rows = [row for row in rows if row.get("source_temporal_refit_state") is not None]
    applied_rows = [row for row in temporal_rows if row.get("post_temporal_observation_delta_applied") is True]
    residual_rows = [
        row
        for row in applied_rows
        if row["post_temporal_observation_reprojection_state"]
        not in {
            "post_temporal_observation_reprojected_metric_depth_compatible",
            "post_temporal_observation_reprojected_projection_untrusted",
            "post_temporal_observation_reprojected_unobserved",
        }
    ]
    report = {
        "method": "solve_v17_post_temporal_depth_observation_weighted_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "post_temporal_observation_weighted_refit_input_rows": len(inputs),
        "post_temporal_observation_weighted_variable_rows": bool_count(
            solved_inputs,
            "post_temporal_observation_variable_candidate",
        ),
        "post_temporal_observation_geometry_factor_rows": bool_count(
            solved_inputs,
            "post_temporal_geometry_factor_row",
        ),
        "post_temporal_observation_compatible_anchor_rows": bool_count(
            solved_inputs,
            "post_temporal_compatible_anchor_row",
        ),
        "post_temporal_observation_depth_factor_rows": bool_count(
            solved_inputs,
            "post_temporal_depth_observation_factor_row",
        ),
        "post_temporal_observation_depth_factor_keypoint_state_counts": state_counts(
            [
                row
                for row in solved_inputs
                if row.get("post_temporal_depth_observation_factor_row") is True
            ],
            "independent_keypoint_support_state",
        )
        if any(row.get("post_temporal_depth_observation_factor_row") is True for row in solved_inputs)
        else {},
        "post_temporal_observation_prior_smooth_only_rows": bool_count(
            solved_inputs,
            "post_temporal_prior_smooth_only_row",
        ),
        "post_temporal_depth_observation_prior_smooth_rows": bool_count(
            solved_inputs,
            "post_temporal_depth_observation_prior_smooth_row",
        ),
        "post_temporal_projection_untrusted_prior_smooth_rows": bool_count(
            solved_inputs,
            "post_temporal_projection_untrusted_prior_smooth_row",
        ),
        "post_temporal_observation_geometry_depth_sample_factor_count": solver_report[
            "geometry_depth_sample_factor_count"
        ],
        "post_temporal_observation_compatible_anchor_sample_factor_count": solver_report[
            "compatible_anchor_sample_factor_count"
        ],
        "post_temporal_depth_observation_sample_factor_count": solver_report[
            "depth_observation_sample_factor_count"
        ],
        "post_temporal_observation_delta_bound_hit_rows": bool_count(
            solved_inputs,
            "post_temporal_observation_delta_bound_hit",
        ),
        "post_temporal_observation_fixed_factor_depth_improved_rows": bool_count(
            solved_inputs,
            "post_temporal_observation_fixed_factor_depth_improved",
        ),
        "post_temporal_observation_fixed_factor_depth_threshold_met_rows": bool_count(
            solved_inputs,
            "post_temporal_observation_fixed_factor_depth_threshold_met",
        ),
        "post_temporal_observation_reprojected_metric_depth_compatible_rows": bool_count(
            applied_rows,
            "metric_depth_compatible",
        ),
        "post_temporal_observation_reprojected_depth_improved_rows": bool_count(
            applied_rows,
            "post_temporal_observation_reprojected_depth_improved",
        ),
        "metric_hand_state_accepted_rows_after_post_temporal_observation_refit": bool_count(
            rows,
            "metric_depth_compatible",
        ),
        "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "post_temporal_observation_reprojection_residual_owner_rows": len(residual_rows),
        "post_temporal_observation_reprojection_local_surface_factor_candidate_rows": sum(
            1
            for row in residual_rows
            if row["post_temporal_observation_reprojection_state"]
            == "post_temporal_observation_reprojected_local_surface_factor_candidate"
        ),
        "post_temporal_observation_reprojection_mixed_surface_depth_owner_rows": sum(
            1
            for row in residual_rows
            if row["post_temporal_observation_reprojection_state"]
            == "post_temporal_observation_reprojected_mixed_surface_depth_owner"
        ),
        "post_temporal_observation_reprojection_depth_observation_owner_rows": sum(
            1
            for row in residual_rows
            if row["post_temporal_observation_reprojection_state"]
            == "post_temporal_observation_reprojected_depth_observation_owner"
        ),
        "post_temporal_observation_reprojection_projection_untrusted_rows": sum(
            1
            for row in applied_rows
            if row["post_temporal_observation_reprojection_state"]
            == "post_temporal_observation_reprojected_projection_untrusted"
        ),
        "post_temporal_observation_input_factor_state_counts": state_counts(
            solved_inputs,
            "post_temporal_observation_factor_state",
        ),
        "post_temporal_observation_reprojection_state_counts": state_counts(
            rows,
            "post_temporal_observation_reprojection_state",
        ),
        "post_temporal_observation_temporal_reprojection_state_counts": state_counts(
            temporal_rows,
            "post_temporal_observation_reprojection_state",
        ),
        "post_temporal_observation_owner_depth_state_counts_after_reprojection": state_counts(
            rows,
            "owner_depth_state",
        ),
        "post_temporal_observation_owner_median_gap_m_after_reprojection": numeric_summary(
            rows,
            "owner_median_gap_m",
        ),
        "solver": solver_report,
        "source_owner_weighted_comparison": {
            "owner_weighted_variable_rows": owner_weighted.get("owner_weighted_variable_rows"),
            "owner_weighted_depth_observation_prior_smooth_rows": owner_weighted.get(
                "owner_weighted_depth_observation_prior_smooth_rows"
            ),
            "owner_weighted_reprojected_metric_depth_compatible_rows": owner_weighted.get(
                "owner_weighted_reprojected_metric_depth_compatible_rows"
            ),
            "metric_hand_state_accepted_rows_after_owner_weighted_refit": owner_weighted.get(
                "metric_hand_state_accepted_rows_after_owner_weighted_refit"
            ),
            "depth_repair_factor_candidate_rows_after_owner_weighted_refit": owner_weighted.get(
                "depth_repair_factor_candidate_rows_after_owner_weighted_refit"
            ),
            "owner_weighted_reprojection_state_counts": owner_weighted.get(
                "owner_weighted_temporal_reprojection_state_counts"
            ),
        },
        "problem_semantics": {
            "post_temporal_depth_observation_strong_keypoint_factor_variable": (
                "same-side independent keypoints support at least half of selected residual samples"
            ),
            "post_temporal_depth_observation_partial_keypoint_factor_variable": (
                "same-side independent keypoints support at least one quarter of selected residual samples"
            ),
            "post_temporal_depth_observation_keypoint_sparse_or_absent_prior_smooth_variable": (
                "box-supported rows without enough keypoint support remain explicit observation variables without a depth equality factor"
            ),
            "claim_limit": (
                "this graph changes only a bounded camera-ray depth variable; MANO pose, MANO shape, object geometry, and contact are not optimized"
            ),
        },
        "solver_rows": solved_inputs,
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_post_temporal_depth_observation_weighted_refit.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = ["trash_1050", "task5_tomato_960"]
    reports = [case_problem(case, args) for case in cases]
    summary = {
        "method": "solve_v17_post_temporal_depth_observation_weighted_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "cases": cases,
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "post_temporal_observation_weighted_refit_input_rows": sum(
            require_int(report.get("post_temporal_observation_weighted_refit_input_rows"), "input rows")
            for report in reports
        ),
        "post_temporal_observation_weighted_variable_rows": sum(
            require_int(report.get("post_temporal_observation_weighted_variable_rows"), "variable rows")
            for report in reports
        ),
        "post_temporal_observation_geometry_factor_rows": sum(
            require_int(report.get("post_temporal_observation_geometry_factor_rows"), "geometry rows")
            for report in reports
        ),
        "post_temporal_observation_depth_factor_rows": sum(
            require_int(report.get("post_temporal_observation_depth_factor_rows"), "depth obs rows")
            for report in reports
        ),
        "post_temporal_observation_depth_factor_keypoint_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_observation_depth_factor_keypoint_state_counts"),
                                "keypoint factor counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_observation_prior_smooth_only_rows": sum(
            require_int(report.get("post_temporal_observation_prior_smooth_only_rows"), "prior rows")
            for report in reports
        ),
        "post_temporal_depth_observation_prior_smooth_rows": sum(
            require_int(report.get("post_temporal_depth_observation_prior_smooth_rows"), "depth prior rows")
            for report in reports
        ),
        "post_temporal_projection_untrusted_prior_smooth_rows": sum(
            require_int(
                report.get("post_temporal_projection_untrusted_prior_smooth_rows"),
                "projection prior rows",
            )
            for report in reports
        ),
        "post_temporal_observation_geometry_depth_sample_factor_count": sum(
            require_int(
                report.get("post_temporal_observation_geometry_depth_sample_factor_count"),
                "geometry samples",
            )
            for report in reports
        ),
        "post_temporal_depth_observation_sample_factor_count": sum(
            require_int(report.get("post_temporal_depth_observation_sample_factor_count"), "obs samples")
            for report in reports
        ),
        "post_temporal_observation_delta_bound_hit_rows": sum(
            require_int(report.get("post_temporal_observation_delta_bound_hit_rows"), "bound hits")
            for report in reports
        ),
        "post_temporal_observation_fixed_factor_depth_improved_rows": sum(
            require_int(report.get("post_temporal_observation_fixed_factor_depth_improved_rows"), "fixed improved")
            for report in reports
        ),
        "post_temporal_observation_fixed_factor_depth_threshold_met_rows": sum(
            require_int(report.get("post_temporal_observation_fixed_factor_depth_threshold_met_rows"), "fixed met")
            for report in reports
        ),
        "post_temporal_observation_reprojected_metric_depth_compatible_rows": sum(
            require_int(
                report.get("post_temporal_observation_reprojected_metric_depth_compatible_rows"),
                "reprojected compatible",
            )
            for report in reports
        ),
        "post_temporal_observation_reprojected_depth_improved_rows": sum(
            require_int(report.get("post_temporal_observation_reprojected_depth_improved_rows"), "improved")
            for report in reports
        ),
        "metric_hand_state_accepted_rows_after_post_temporal_observation_refit": sum(
            require_int(
                report.get("metric_hand_state_accepted_rows_after_post_temporal_observation_refit"),
                "accepted",
            )
            for report in reports
        ),
        "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit": sum(
            require_int(
                report.get("depth_repair_factor_candidate_rows_after_post_temporal_observation_refit"),
                "residual",
            )
            for report in reports
        ),
        "post_temporal_observation_reprojection_residual_owner_rows": sum(
            require_int(report.get("post_temporal_observation_reprojection_residual_owner_rows"), "residual owners")
            for report in reports
        ),
        "post_temporal_observation_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(
                report.get("post_temporal_observation_reprojection_local_surface_factor_candidate_rows"),
                "local rows",
            )
            for report in reports
        ),
        "post_temporal_observation_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(
                report.get("post_temporal_observation_reprojection_mixed_surface_depth_owner_rows"),
                "mixed rows",
            )
            for report in reports
        ),
        "post_temporal_observation_reprojection_depth_observation_owner_rows": sum(
            require_int(
                report.get("post_temporal_observation_reprojection_depth_observation_owner_rows"),
                "depth rows",
            )
            for report in reports
        ),
        "post_temporal_observation_reprojection_projection_untrusted_rows": sum(
            require_int(
                report.get("post_temporal_observation_reprojection_projection_untrusted_rows"),
                "projection rows",
            )
            for report in reports
        ),
        "post_temporal_observation_input_factor_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_observation_input_factor_state_counts"),
                                "input factor counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_observation_temporal_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_observation_temporal_reprojection_state_counts"),
                                "temporal state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "source_owner_weighted_variable_rows": sum(
            require_int(
                require_dict(report.get("source_owner_weighted_comparison"), "owner comparison").get(
                    "owner_weighted_variable_rows"
                ),
                "source variables",
            )
            for report in reports
        ),
        "source_owner_weighted_depth_observation_prior_smooth_rows": sum(
            require_int(
                require_dict(report.get("source_owner_weighted_comparison"), "owner comparison").get(
                    "owner_weighted_depth_observation_prior_smooth_rows"
                ),
                "source depth prior rows",
            )
            for report in reports
        ),
        "source_owner_weighted_reprojected_metric_depth_compatible_rows": sum(
            require_int(
                require_dict(report.get("source_owner_weighted_comparison"), "owner comparison").get(
                    "owner_weighted_reprojected_metric_depth_compatible_rows"
                ),
                "source compatible rows",
            )
            for report in reports
        ),
        "source_metric_hand_state_accepted_rows_after_owner_weighted_refit": sum(
            require_int(
                require_dict(report.get("source_owner_weighted_comparison"), "owner comparison").get(
                    "metric_hand_state_accepted_rows_after_owner_weighted_refit"
                ),
                "source accepted rows",
            )
            for report in reports
        ),
        "source_depth_repair_factor_candidate_rows_after_owner_weighted_refit": sum(
            require_int(
                require_dict(report.get("source_owner_weighted_comparison"), "owner comparison").get(
                    "depth_repair_factor_candidate_rows_after_owner_weighted_refit"
                ),
                "source residual rows",
            )
            for report in reports
        ),
        "case_reports": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "post_temporal_observation_weighted_variable_rows": require_int(
                    report.get("post_temporal_observation_weighted_variable_rows"),
                    "variable rows",
                ),
                "post_temporal_observation_depth_factor_rows": require_int(
                    report.get("post_temporal_observation_depth_factor_rows"),
                    "depth factor rows",
                ),
                "post_temporal_observation_reprojected_metric_depth_compatible_rows": require_int(
                    report.get("post_temporal_observation_reprojected_metric_depth_compatible_rows"),
                    "compatible rows",
                ),
                "metric_hand_state_accepted_rows_after_post_temporal_observation_refit": require_int(
                    report.get("metric_hand_state_accepted_rows_after_post_temporal_observation_refit"),
                    "accepted rows",
                ),
                "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit": require_int(
                    report.get("depth_repair_factor_candidate_rows_after_post_temporal_observation_refit"),
                    "residual rows",
                ),
                "post_temporal_observation_temporal_reprojection_state_counts": require_dict(
                    report.get("post_temporal_observation_temporal_reprojection_state_counts"),
                    "temporal state counts",
                ),
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_post_temporal_depth_observation_weighted_refit_summary.json", summary)
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
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--post-temporal-depth-observation-support-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit"),
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
    parser.add_argument("--min-post-temporal-factor-samples", type=int, default=3)
    parser.add_argument("--max-factor-samples-per-row", type=int, default=64)
    parser.add_argument("--max-depth-observation-samples-per-row", type=int, default=64)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-post-temporal-delta-m", type=float, default=0.12)
    parser.add_argument("--min-corrected-hand-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-post-temporal-geometry-depth-m", type=float, default=0.02)
    parser.add_argument("--sigma-post-temporal-compatible-anchor-m", type=float, default=0.03)
    parser.add_argument("--sigma-depth-observation-strong-m", type=float, default=0.03)
    parser.add_argument("--sigma-depth-observation-partial-m", type=float, default=0.05)
    parser.add_argument("--sigma-post-temporal-delta-prior-m", type=float, default=0.08)
    parser.add_argument("--sigma-post-temporal-delta-step-m", type=float, default=0.03)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=45)
    parser.add_argument("--min-depth-improvement-m", type=float, default=0.005)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--max-solver-iter", type=int, default=500)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
