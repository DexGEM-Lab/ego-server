#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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


STATUS = "v17_full_residual_pose_transition_diagnostic_qc"
CLAIM = (
    "This artifact compares the scalar-only and pose-enabled full-residual relinearized hand "
    "graphs row by row. It tests whether activating MANO pose variables creates stable metric "
    "closure, or merely moves residual ownership while saturating pose deltas."
)

COMPATIBLE_STATE = "relinearized_reprojected_metric_depth_compatible"
PROJECTION_UNTRUSTED_STATE = "relinearized_reprojected_projection_untrusted"
LOCAL_STATE = "relinearized_reprojected_local_surface_factor_candidate"
MIXED_STATE = "relinearized_reprojected_mixed_surface_depth_owner"
DEPTH_OBSERVATION_STATE = "relinearized_reprojected_depth_observation_owner"
RESIDUAL_OWNER_STATES = {LOCAL_STATE, MIXED_STATE, DEPTH_OBSERVATION_STATE}


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
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


def transition_counts(rows: list[dict[str, Any]], before_key: str, after_key: str) -> dict[str, int]:
    counts = Counter(
        f"{require_str(row.get(before_key), before_key)} -> {require_str(row.get(after_key), after_key)}"
        for row in rows
    )
    return dict(sorted(counts.items()))


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(require_str(row.get(key), key) for row in rows)
    return dict(sorted(counts.items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if require_bool(row.get(key), key))


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = optional_finite_number(row.get(key), key)
        if value is not None:
            values.append(value)
    return summarize(values)


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


def residual_state(state: str) -> bool:
    return state in RESIDUAL_OWNER_STATES


def row_delta(
    *,
    graph_id: str,
    scalar_row: dict[str, Any],
    pose_row: dict[str, Any],
    scalar_factor: dict[str, Any],
    pose_factor: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scalar_state = require_str(
        scalar_row.get("relinearized_reprojection_state"),
        "scalar reprojection state",
    )
    pose_state = require_str(
        pose_row.get("relinearized_reprojection_state"),
        "pose reprojection state",
    )
    scalar_gap = optional_finite_number(scalar_row.get("owner_median_gap_m"), "scalar owner gap")
    pose_gap = optional_finite_number(pose_row.get("owner_median_gap_m"), "pose owner gap")
    if scalar_gap is None or pose_gap is None:
        abs_gap_improvement = None
    else:
        abs_gap_improvement = abs(scalar_gap) - abs(pose_gap)
    pose_delta_abs_max_rad = optional_finite_number(
        pose_row.get("relinearized_pose_delta_abs_max_rad"),
        "pose delta max",
    )
    scalar_shift = optional_finite_number(
        scalar_row.get("relinearized_delta_shift_m"),
        "scalar delta shift",
    )
    pose_shift = optional_finite_number(
        pose_row.get("relinearized_delta_shift_m"),
        "pose delta shift",
    )
    if scalar_shift is None or pose_shift is None:
        scalar_shift_change_m = None
    else:
        scalar_shift_change_m = pose_shift - scalar_shift
    compatible_gain = scalar_state != COMPATIBLE_STATE and pose_state == COMPATIBLE_STATE
    compatible_loss = scalar_state == COMPATIBLE_STATE and pose_state != COMPATIBLE_STATE
    residual_owner_persistent = residual_state(scalar_state) and residual_state(pose_state)
    residual_owner_created = not residual_state(scalar_state) and residual_state(pose_state)
    residual_owner_resolved = residual_state(scalar_state) and not residual_state(pose_state)
    improved_5mm = bool(
        abs_gap_improvement is not None
        and abs_gap_improvement >= float(args.min_abs_gap_improvement_m)
    )
    regressed_5mm = bool(
        abs_gap_improvement is not None
        and abs_gap_improvement <= -float(args.min_abs_gap_improvement_m)
    )
    clamp_hit = bool(
        pose_delta_abs_max_rad is not None
        and pose_delta_abs_max_rad >= float(args.pose_clamp_rad) - float(args.pose_clamp_tolerance_rad)
    )
    return {
        "case": require_str(scalar_row.get("case"), "case"),
        "source_hand_depth_repair_graph_variable_id": graph_id,
        "frame_idx": require_int(scalar_row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(scalar_row.get("hand_side"), "hand_side"),
        "hand_index": require_int(scalar_row.get("hand_index"), "hand_index"),
        "scalar_reprojection_state": scalar_state,
        "pose_reprojection_state": pose_state,
        "scalar_input_factor_state": require_str(
            scalar_factor.get("relinearized_input_factor_state"),
            "scalar input factor state",
        ),
        "pose_input_factor_state": require_str(
            pose_factor.get("relinearized_input_factor_state"),
            "pose input factor state",
        ),
        "scalar_owner_depth_state": require_str(
            scalar_row.get("owner_depth_state"),
            "scalar owner depth state",
        ),
        "pose_owner_depth_state": require_str(
            pose_row.get("owner_depth_state"),
            "pose owner depth state",
        ),
        "scalar_owner_median_gap_m": scalar_gap,
        "pose_owner_median_gap_m": pose_gap,
        "abs_owner_median_gap_improvement_m": abs_gap_improvement,
        "scalar_relinearized_delta_shift_m": scalar_shift,
        "pose_relinearized_delta_shift_m": pose_shift,
        "pose_minus_scalar_delta_shift_m": scalar_shift_change_m,
        "pose_delta_abs_max_rad": pose_delta_abs_max_rad,
        "pose_delta_clamp_hit": clamp_hit,
        "compatible_gain": compatible_gain,
        "compatible_loss": compatible_loss,
        "residual_owner_persistent": residual_owner_persistent,
        "residual_owner_created": residual_owner_created,
        "residual_owner_resolved": residual_owner_resolved,
        "abs_gap_improved_at_least_5mm": improved_5mm,
        "abs_gap_regressed_at_least_5mm": regressed_5mm,
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    scalar_path = existing_path(
        args.scalar_full_residual_graph_root
        / case
        / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
        f"{case} scalar full-residual graph",
    )
    pose_path = existing_path(
        args.pose_full_residual_graph_root
        / case
        / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
        f"{case} pose full-residual graph",
    )
    scalar = require_dict(load_json(scalar_path), f"{case} scalar graph")
    pose = require_dict(load_json(pose_path), f"{case} pose graph")
    if require_int(scalar.get("frame_count"), f"{case} scalar frame_count") != require_int(
        pose.get("frame_count"),
        f"{case} pose frame_count",
    ):
        raise RuntimeError(f"{case} scalar and pose graphs have different frame counts")
    if require_str(scalar.get("relinearized_variable_scope"), f"{case} scalar variable scope") != "full_residual_coverage":
        raise RuntimeError(f"{case} scalar graph is not full_residual_coverage")
    if require_str(pose.get("relinearized_variable_scope"), f"{case} pose variable scope") != "full_residual_coverage":
        raise RuntimeError(f"{case} pose graph is not full_residual_coverage")
    if scalar.get("relinearized_geometry_pose_optimization_enabled") is True:
        raise RuntimeError(f"{case} scalar graph unexpectedly has pose optimization enabled")
    if pose.get("relinearized_geometry_pose_optimization_enabled") is not True:
        raise RuntimeError(f"{case} pose graph does not have pose optimization enabled")
    scalar_rows = applied_rows(scalar, f"{case} scalar")
    pose_rows = applied_rows(pose, f"{case} pose")
    scalar_factors = factor_rows(scalar, f"{case} scalar")
    pose_factors = factor_rows(pose, f"{case} pose")
    if set(scalar_rows) != set(pose_rows):
        raise RuntimeError(f"{case} scalar and pose applied row ids differ")
    if set(scalar_rows) != set(scalar_factors):
        raise RuntimeError(f"{case} scalar rows and factors differ")
    if set(pose_rows) != set(pose_factors):
        raise RuntimeError(f"{case} pose rows and factors differ")
    rows = [
        row_delta(
            graph_id=graph_id,
            scalar_row=scalar_rows[graph_id],
            pose_row=pose_rows[graph_id],
            scalar_factor=scalar_factors[graph_id],
            pose_factor=pose_factors[graph_id],
            args=args,
        )
        for graph_id in sorted(scalar_rows)
    ]
    net_compatible_gain = bool_count(rows, "compatible_gain") - bool_count(rows, "compatible_loss")
    out = {
        "method": "build_v17_full_residual_pose_transition_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "scalar_full_residual_graph": source_summary(scalar_path, scalar),
            "pose_full_residual_graph": source_summary(pose_path, pose),
        },
        "frame_count": require_int(scalar.get("frame_count"), f"{case} frame_count"),
        "transition_variable_rows": len(rows),
        "scalar_variable_rows": require_int(scalar.get("relinearized_variable_rows"), f"{case} scalar variables"),
        "pose_variable_rows": require_int(pose.get("relinearized_variable_rows"), f"{case} pose variables"),
        "scalar_accepted_rows_after_reprojection": require_int(
            scalar.get("metric_hand_state_accepted_rows_after_relinearized_graph"),
            f"{case} scalar accepted rows",
        ),
        "pose_accepted_rows_after_reprojection": require_int(
            pose.get("metric_hand_state_accepted_rows_after_relinearized_graph"),
            f"{case} pose accepted rows",
        ),
        "scalar_residual_rows_after_reprojection": require_int(
            scalar.get("depth_repair_factor_candidate_rows_after_relinearized_graph"),
            f"{case} scalar residual rows",
        ),
        "pose_residual_rows_after_reprojection": require_int(
            pose.get("depth_repair_factor_candidate_rows_after_relinearized_graph"),
            f"{case} pose residual rows",
        ),
        "compatible_gain_rows": bool_count(rows, "compatible_gain"),
        "compatible_loss_rows": bool_count(rows, "compatible_loss"),
        "net_compatible_gain_rows": net_compatible_gain,
        "residual_owner_persistent_rows": bool_count(rows, "residual_owner_persistent"),
        "residual_owner_created_rows": bool_count(rows, "residual_owner_created"),
        "residual_owner_resolved_rows": bool_count(rows, "residual_owner_resolved"),
        "pose_delta_clamp_hit_rows": bool_count(rows, "pose_delta_clamp_hit"),
        "abs_gap_improved_at_least_5mm_rows": bool_count(rows, "abs_gap_improved_at_least_5mm"),
        "abs_gap_regressed_at_least_5mm_rows": bool_count(rows, "abs_gap_regressed_at_least_5mm"),
        "abs_owner_median_gap_improvement_m": numeric_summary(
            rows,
            "abs_owner_median_gap_improvement_m",
        ),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        "pose_minus_scalar_delta_shift_m": numeric_summary(
            rows,
            "pose_minus_scalar_delta_shift_m",
        ),
        "reprojection_state_transition_counts": transition_counts(
            rows,
            "scalar_reprojection_state",
            "pose_reprojection_state",
        ),
        "input_factor_state_transition_counts": transition_counts(
            rows,
            "scalar_input_factor_state",
            "pose_input_factor_state",
        ),
        "owner_depth_state_transition_counts": transition_counts(
            rows,
            "scalar_owner_depth_state",
            "pose_owner_depth_state",
        ),
        "pose_state_counts": state_counts(rows, "pose_reprojection_state"),
        "scalar_state_counts": state_counts(rows, "scalar_reprojection_state"),
        "problem_semantics": {
            "compatible_gain_rows": "rows that were not metric-depth compatible in the scalar full-residual graph and became compatible after pose optimization",
            "compatible_loss_rows": "rows that were metric-depth compatible in the scalar full-residual graph and became non-compatible after pose optimization",
            "net_compatible_gain_rows": "compatible gains minus compatible losses over the same full-residual variable population",
            "pose_delta_clamp_hit_rows": "rows whose pose update reaches the configured per-axis pose delta clamp",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_full_residual_pose_transition_diagnostic.json", out)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [
        row
        for case in cases
        for row in require_list(case.get("rows"), "case transition rows")
    ]
    summary = {
        "method": "build_v17_full_residual_pose_transition_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame_count") for case in cases),
        "transition_variable_rows": len(rows),
        "scalar_variable_rows": sum(require_int(case.get("scalar_variable_rows"), "case scalar variables") for case in cases),
        "pose_variable_rows": sum(require_int(case.get("pose_variable_rows"), "case pose variables") for case in cases),
        "scalar_accepted_rows_after_reprojection": sum(
            require_int(case.get("scalar_accepted_rows_after_reprojection"), "case scalar accepted rows")
            for case in cases
        ),
        "pose_accepted_rows_after_reprojection": sum(
            require_int(case.get("pose_accepted_rows_after_reprojection"), "case pose accepted rows")
            for case in cases
        ),
        "scalar_residual_rows_after_reprojection": sum(
            require_int(case.get("scalar_residual_rows_after_reprojection"), "case scalar residual rows")
            for case in cases
        ),
        "pose_residual_rows_after_reprojection": sum(
            require_int(case.get("pose_residual_rows_after_reprojection"), "case pose residual rows")
            for case in cases
        ),
        "compatible_gain_rows": bool_count(rows, "compatible_gain"),
        "compatible_loss_rows": bool_count(rows, "compatible_loss"),
        "net_compatible_gain_rows": bool_count(rows, "compatible_gain") - bool_count(rows, "compatible_loss"),
        "residual_owner_persistent_rows": bool_count(rows, "residual_owner_persistent"),
        "residual_owner_created_rows": bool_count(rows, "residual_owner_created"),
        "residual_owner_resolved_rows": bool_count(rows, "residual_owner_resolved"),
        "pose_delta_clamp_hit_rows": bool_count(rows, "pose_delta_clamp_hit"),
        "abs_gap_improved_at_least_5mm_rows": bool_count(rows, "abs_gap_improved_at_least_5mm"),
        "abs_gap_regressed_at_least_5mm_rows": bool_count(rows, "abs_gap_regressed_at_least_5mm"),
        "abs_owner_median_gap_improvement_m": numeric_summary(
            rows,
            "abs_owner_median_gap_improvement_m",
        ),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        "pose_minus_scalar_delta_shift_m": numeric_summary(
            rows,
            "pose_minus_scalar_delta_shift_m",
        ),
        "reprojection_state_transition_counts": transition_counts(
            rows,
            "scalar_reprojection_state",
            "pose_reprojection_state",
        ),
        "input_factor_state_transition_counts": transition_counts(
            rows,
            "scalar_input_factor_state",
            "pose_input_factor_state",
        ),
        "owner_depth_state_transition_counts": transition_counts(
            rows,
            "scalar_owner_depth_state",
            "pose_owner_depth_state",
        ),
        "pose_state_counts": state_counts(rows, "pose_reprojection_state"),
        "scalar_state_counts": state_counts(rows, "scalar_reprojection_state"),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame_count"),
                "transition_variable_rows": require_int(
                    case.get("transition_variable_rows"),
                    "case transition variables",
                ),
                "compatible_gain_rows": require_int(case.get("compatible_gain_rows"), "case gains"),
                "compatible_loss_rows": require_int(case.get("compatible_loss_rows"), "case losses"),
                "net_compatible_gain_rows": require_int(case.get("net_compatible_gain_rows"), "case net gain"),
                "pose_delta_clamp_hit_rows": require_int(case.get("pose_delta_clamp_hit_rows"), "case clamp hits"),
                "residual_owner_persistent_rows": require_int(
                    case.get("residual_owner_persistent_rows"),
                    "case persistent residual owners",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_full_residual_pose_transition_diagnostic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scalar-full-residual-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument(
        "--pose-full-residual-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_pose_transition_diagnostic"),
    )
    parser.add_argument("--min-abs-gap-improvement-m", type=float, default=0.005)
    parser.add_argument("--pose-clamp-rad", type=float, default=0.35)
    parser.add_argument("--pose-clamp-tolerance-rad", type=float, default=1.0e-5)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
