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

from build_v17_hand_depth_repair_residual_owner_state import (
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
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary


STATUS = "v17_hand_far_field_temporal_refit_qc"
CLAIM = (
    "This artifact tests a relinearized full-timeline hand-depth factor on persistent far-field "
    "residual segments. It fits bounded incremental camera-ray depth shifts against the post-repair "
    "residual samples selected by the depth-observation switch problem, with temporal smoothness over "
    "same-hand rows. It does not update hand annotations, does not re-evaluate 2D projection geometry, "
    "and does not complete the V3 joint solver."
)


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def depth_id_to_graph_id(depth_id: str) -> str:
    if not depth_id.startswith("hand_depth_observation_switch:"):
        raise RuntimeError(f"unexpected depth-observation id: {depth_id}")
    return depth_id.replace("hand_depth_observation_switch:", "hand_depth_repair_graph:", 1)


def gap_summary(gaps: np.ndarray) -> dict[str, Any]:
    return {
        "signed_gap_m": summarize(gaps.astype(float).tolist()),
        "abs_gap_m": summarize(np.abs(gaps).astype(float).tolist()),
    }


def refit_depth_state(row: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("temporal_refit_variable_candidate") is not True:
        return require_str(row.get("temporal_refit_candidate_rejection"), "candidate rejection")
    if row.get("temporal_refit_bound_hit") is True:
        if row.get("temporal_refit_depth_threshold_met") is True:
            return "temporal_refit_depth_threshold_met_with_bound_hit"
        return "temporal_refit_bound_hit_residual_remaining"
    if row.get("temporal_refit_depth_threshold_met") is True:
        return "temporal_refit_depth_threshold_met"
    if row.get("temporal_refit_depth_improved") is True:
        return "temporal_refit_depth_improved_residual_remaining"
    return "temporal_refit_no_material_depth_gain"


def temporal_row_inputs(
    *,
    case: str,
    segment: dict[str, Any],
    graph_id: str,
    repair_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = row_samples(repair_row)
    selected = selected_residual(repair_row, samples, args)
    hand_z = np.asarray(samples["hand_z"], dtype=np.float64)
    metric_z = np.asarray(samples["metric_z"], dtype=np.float64)
    gaps = (hand_z - metric_z)[selected]
    current_shift = finite_float(repair_row.get("hand_ray_shift_m"), "hand_ray_shift_m")
    lower = max(
        -float(args.max_abs_hand_ray_shift_m) - current_shift,
        -float(args.max_abs_temporal_delta_m),
    )
    upper = min(
        float(args.max_abs_hand_ray_shift_m) - current_shift,
        float(args.max_abs_temporal_delta_m),
    )
    projection = require_dict(
        repair_row.get("projection_residual_to_measurement_px"),
        "projection residual",
    )
    projection_ok = bool(projection.get("residual_ok") is True)
    candidate = bool(
        len(gaps) >= int(args.min_depth_pixels)
        and projection_ok
        and lower <= upper
        and repair_row.get("base_available") is True
    )
    rejection = None
    if not candidate:
        if repair_row.get("base_available") is not True:
            rejection = "temporal_refit_missing_base_hand_depth_state"
        elif len(gaps) < int(args.min_depth_pixels):
            rejection = "temporal_refit_too_few_residual_depth_samples"
        elif not projection_ok:
            rejection = "temporal_refit_projection_untrusted_before_refit"
        elif lower > upper:
            rejection = "temporal_refit_no_remaining_ray_shift_bound"
        else:
            rejection = "temporal_refit_candidate_rejected"
    return {
        "case": case,
        "hand_far_field_temporal_refit_variable_id": graph_id.replace(
            "hand_depth_repair_graph:",
            "hand_far_field_temporal_refit:",
            1,
        ),
        "source_hand_depth_repair_graph_variable_id": graph_id,
        "source_hand_far_field_depth_temporal_segment_id": require_str(
            segment.get("hand_far_field_depth_temporal_segment_id"),
            "segment id",
        ),
        "frame_idx": require_int(repair_row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(repair_row.get("hand_side"), "hand_side"),
        "hand_index": require_int(repair_row.get("hand_index"), "hand_index"),
        "depth_sign_state": require_str(segment.get("depth_sign_state"), "depth sign state"),
        "current_hand_ray_shift_m": current_shift,
        "temporal_delta_lower_bound_m": lower,
        "temporal_delta_upper_bound_m": upper,
        "temporal_refit_variable_candidate": candidate,
        "temporal_refit_candidate_rejection": rejection,
        "selected_residual_sample_count": int(len(gaps)),
        "current_selected_residual": gap_summary(gaps),
        "projection_residual_to_measurement_px": projection,
        "_selected_gaps": gaps,
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
    for col, value in terms:
        rows.append(row_i)
        cols.append(col)
        vals.append(float(value))
    rhs.append(float(target))
    return row_i + 1


def solve_temporal_rows(rows_in: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_indices = [
        i for i, row in enumerate(rows_in) if row.get("temporal_refit_variable_candidate") is True
    ]
    if not candidate_indices:
        rows_out = [
            {k: v for k, v in row.items() if k != "_selected_gaps"}
            | {
                "temporal_refit_delta_shift_m": None,
                "temporal_refit_total_hand_ray_shift_m": row.get("current_hand_ray_shift_m"),
                "temporal_refit_depth_improved": False,
                "temporal_refit_depth_threshold_met": False,
                "temporal_refit_bound_hit": False,
                "temporal_refit_state": refit_depth_state(row, args),
            }
            for row in rows_in
        ]
        return rows_out, {
            "variable_count": 0,
            "depth_sample_factor_count": 0,
            "prior_factor_count": 0,
            "smoothness_factor_count": 0,
            "matrix_rows": 0,
            "matrix_cols": 0,
            "solver_skipped_reason": "no_temporal_refit_variables",
        }
    var_by_row = {source_i: var_i for var_i, source_i in enumerate(candidate_indices)}
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_vals: list[float] = []
    rhs: list[float] = []
    row_i = 0
    depth_factors = 0
    prior_factors = 0
    smooth_factors = 0
    sigma_depth = float(args.sigma_temporal_depth_m)
    sigma_prior = float(args.sigma_temporal_delta_prior_m)
    sigma_step = float(args.sigma_temporal_delta_step_m)
    lower = np.full(len(candidate_indices), -float(args.max_abs_temporal_delta_m), dtype=np.float64)
    upper = np.full(len(candidate_indices), float(args.max_abs_temporal_delta_m), dtype=np.float64)
    for source_i in candidate_indices:
        var_i = var_by_row[source_i]
        row = rows_in[source_i]
        lower[var_i] = max(lower[var_i], finite_float(row.get("temporal_delta_lower_bound_m"), "lower bound"))
        upper[var_i] = min(upper[var_i], finite_float(row.get("temporal_delta_upper_bound_m"), "upper bound"))
        gaps = np.asarray(row["_selected_gaps"], dtype=np.float64)
        for gap in gaps:
            row_i = add_row(
                matrix_rows,
                matrix_cols,
                matrix_vals,
                rhs,
                row_i,
                [(var_i, 1.0 / sigma_depth)],
                -float(gap) / sigma_depth,
            )
            depth_factors += 1
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
        raise RuntimeError(f"inconsistent temporal refit bounds at variables {bad[:12].tolist()}")
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
        raise RuntimeError("temporal refit returned invalid solution")
    rows_out: list[dict[str, Any]] = []
    for source_i, row in enumerate(rows_in):
        base = {k: v for k, v in row.items() if k != "_selected_gaps"}
        if source_i not in var_by_row:
            out = {
                **base,
                "temporal_refit_delta_shift_m": None,
                "temporal_refit_total_hand_ray_shift_m": row.get("current_hand_ray_shift_m"),
                "temporal_refit_depth_improved": False,
                "temporal_refit_depth_threshold_met": False,
                "temporal_refit_bound_hit": False,
            }
            rows_out.append({**out, "temporal_refit_state": refit_depth_state(out, args)})
            continue
        var_i = var_by_row[source_i]
        delta = float(result.x[var_i])
        current_shift = finite_float(row.get("current_hand_ray_shift_m"), "current shift")
        before = np.asarray(row["_selected_gaps"], dtype=np.float64)
        after = before + delta
        before_abs_median = float(np.median(np.abs(before))) if len(before) else math.inf
        after_abs_median = float(np.median(np.abs(after))) if len(after) else math.inf
        after_abs_p95 = float(np.percentile(np.abs(after), 95.0)) if len(after) else math.inf
        lower_i = finite_float(row.get("temporal_delta_lower_bound_m"), "lower bound")
        upper_i = finite_float(row.get("temporal_delta_upper_bound_m"), "upper bound")
        out = {
            **base,
            "temporal_refit_delta_shift_m": delta,
            "temporal_refit_total_hand_ray_shift_m": current_shift + delta,
            "temporal_refit_selected_residual": gap_summary(after),
            "temporal_refit_depth_improved": bool(
                before_abs_median - after_abs_median >= float(args.min_depth_improvement_m)
            ),
            "temporal_refit_depth_threshold_met": bool(
                after_abs_median <= float(args.max_median_abs_depth_gap_m)
                and after_abs_p95 <= float(args.max_p95_abs_depth_gap_m)
            ),
            "temporal_refit_bound_hit": bool(
                math.isclose(delta, lower_i, abs_tol=float(args.bound_tolerance_m))
                or math.isclose(delta, upper_i, abs_tol=float(args.bound_tolerance_m))
            ),
        }
        rows_out.append({**out, "temporal_refit_state": refit_depth_state(out, args)})
    return rows_out, {
        "variable_count": len(candidate_indices),
        "depth_sample_factor_count": depth_factors,
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


def segment_summary(segment: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    segment_id = require_str(segment.get("hand_far_field_depth_temporal_segment_id"), "segment id")
    selected = [row for row in rows if row.get("source_hand_far_field_depth_temporal_segment_id") == segment_id]
    return {
        "case": require_str(segment.get("case"), "case"),
        "source_hand_far_field_depth_temporal_segment_id": segment_id,
        "hand_side": require_str(segment.get("hand_side"), "hand_side"),
        "hand_index": require_int(segment.get("hand_index"), "hand_index"),
        "depth_sign_state": require_str(segment.get("depth_sign_state"), "depth sign state"),
        "start_frame_idx": require_int(segment.get("start_frame_idx"), "start frame"),
        "end_frame_idx": require_int(segment.get("end_frame_idx"), "end frame"),
        "frame_count": require_int(segment.get("frame_count"), "segment frame_count"),
        "temporal_refit_variable_candidate_rows": bool_count(selected, "temporal_refit_variable_candidate"),
        "temporal_refit_depth_improved_rows": bool_count(selected, "temporal_refit_depth_improved"),
        "temporal_refit_depth_threshold_met_rows": bool_count(selected, "temporal_refit_depth_threshold_met"),
        "temporal_refit_bound_hit_rows": bool_count(selected, "temporal_refit_bound_hit"),
        "temporal_refit_state_counts": state_counts(selected, "temporal_refit_state") if selected else {},
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_far_field_depth_temporal_problem": existing_path(
            args.hand_far_field_depth_temporal_problem_root
            / case
            / "v17_hand_far_field_depth_temporal_problem.json",
            f"{case} hand far-field depth temporal problem",
        ),
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    temporal = payloads["hand_far_field_depth_temporal_problem"]
    repair = payloads["hand_depth_repair_graph"]
    frame_count = require_int(temporal.get("frame_count"), f"{case} temporal frame_count")
    if frame_count != require_int(repair.get("frame_count"), f"{case} repair frame_count"):
        raise RuntimeError(f"{case} temporal and repair graph frame counts disagree")
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    segments = [
        require_dict(raw, "temporal segment")
        for raw in require_list(temporal.get("segments"), f"{case} segments")
        if require_dict(raw, "temporal segment").get("temporal_segment_state")
        == "far_field_temporal_factor_candidate"
    ]
    rows_in: list[dict[str, Any]] = []
    for segment in segments:
        segment_sample_count = 0
        for depth_id_raw in require_list(
            segment.get("source_hand_depth_observation_switch_variable_ids"),
            "source depth ids",
        ):
            graph_id = depth_id_to_graph_id(require_str(depth_id_raw, "source depth id"))
            repair_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
            row = temporal_row_inputs(
                case=case,
                segment=segment,
                graph_id=graph_id,
                repair_row=repair_row,
                args=args,
            )
            segment_sample_count += require_int(
                row.get("selected_residual_sample_count"),
                "row residual sample count",
            )
            rows_in.append(row)
        expected_samples = require_int(
            segment.get("selected_residual_sample_count"),
            "segment selected residual sample count",
        )
        if segment_sample_count != expected_samples:
            raise RuntimeError(
                f"{case} segment {segment.get('hand_far_field_depth_temporal_segment_id')} sample count "
                f"mismatch: expected {expected_samples}, got {segment_sample_count}"
            )
    rows, system = solve_temporal_rows(rows_in, args)
    segment_rows = [segment_summary(segment, rows) for segment in segments]
    report = {
        "method": "solve_v17_hand_far_field_temporal_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "far_field_temporal_refit_segment_count": len(segment_rows),
        "far_field_temporal_refit_row_count": len(rows),
        "temporal_refit_variable_candidate_rows": bool_count(rows, "temporal_refit_variable_candidate"),
        "temporal_refit_depth_improved_rows": bool_count(rows, "temporal_refit_depth_improved"),
        "temporal_refit_depth_threshold_met_rows": bool_count(rows, "temporal_refit_depth_threshold_met"),
        "temporal_refit_bound_hit_rows": bool_count(rows, "temporal_refit_bound_hit"),
        "temporal_refit_state_counts": state_counts(rows, "temporal_refit_state") if rows else {},
        "system": system,
        "problem_semantics": {
            "temporal_refit_depth_threshold_met": "the post-repair selected residual samples meet the depth thresholds after the incremental temporal ray-depth solve",
            "temporal_refit_depth_threshold_met_with_bound_hit": "the depth samples meet threshold only while a ray-shift bound is active",
            "temporal_refit_depth_improved_residual_remaining": "the incremental temporal solve reduces median absolute depth residual but does not meet threshold",
            "claim_limit": "projection geometry and MANO surface ownership are not re-evaluated here, so rows cannot become accepted hand states in this artifact",
        },
        "segments": segment_rows,
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_far_field_temporal_refit.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_far_field_depth_temporal_problem_root
        / "v17_hand_far_field_depth_temporal_problem_summary.json",
        "hand far-field depth temporal problem summary",
    )
    summary = require_dict(load_json(summary_path), "hand far-field temporal summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "solve_v17_hand_far_field_temporal_refit",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_far_field_depth_temporal_problem_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_far_field_temporal_refit.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "far_field_temporal_refit_segment_count": require_int(
                    report.get("far_field_temporal_refit_segment_count"),
                    "segment count",
                ),
                "far_field_temporal_refit_row_count": require_int(
                    report.get("far_field_temporal_refit_row_count"),
                    "row count",
                ),
                "temporal_refit_variable_candidate_rows": require_int(
                    report.get("temporal_refit_variable_candidate_rows"),
                    "candidate rows",
                ),
                "temporal_refit_depth_improved_rows": require_int(
                    report.get("temporal_refit_depth_improved_rows"),
                    "improved rows",
                ),
                "temporal_refit_depth_threshold_met_rows": require_int(
                    report.get("temporal_refit_depth_threshold_met_rows"),
                    "threshold rows",
                ),
                "temporal_refit_bound_hit_rows": require_int(
                    report.get("temporal_refit_bound_hit_rows"),
                    "bound hit rows",
                ),
                "temporal_refit_state_counts": require_dict(
                    report.get("temporal_refit_state_counts"),
                    "state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "far_field_temporal_refit_segment_count": sum(
            require_int(report.get("far_field_temporal_refit_segment_count"), "segment count")
            for report in reports
        ),
        "far_field_temporal_refit_row_count": sum(
            require_int(report.get("far_field_temporal_refit_row_count"), "row count")
            for report in reports
        ),
        "temporal_refit_variable_candidate_rows": sum(
            require_int(report.get("temporal_refit_variable_candidate_rows"), "candidate rows")
            for report in reports
        ),
        "temporal_refit_depth_improved_rows": sum(
            require_int(report.get("temporal_refit_depth_improved_rows"), "improved rows")
            for report in reports
        ),
        "temporal_refit_depth_threshold_met_rows": sum(
            require_int(report.get("temporal_refit_depth_threshold_met_rows"), "threshold rows")
            for report in reports
        ),
        "temporal_refit_bound_hit_rows": sum(
            require_int(report.get("temporal_refit_bound_hit_rows"), "bound hit rows")
            for report in reports
        ),
        "temporal_refit_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("temporal_refit_state_counts"), "state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_far_field_temporal_refit_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-far-field-depth-temporal-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_depth_temporal_problem"),
    )
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit"),
    )
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--min-depth-improvement-m", type=float, default=0.005)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-temporal-delta-m", type=float, default=0.20)
    parser.add_argument("--sigma-temporal-depth-m", type=float, default=0.025)
    parser.add_argument("--sigma-temporal-delta-prior-m", type=float, default=0.12)
    parser.add_argument("--sigma-temporal-delta-step-m", type=float, default=0.025)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=2)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--max-solver-iter", type=int, default=500)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
