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


STATUS = "v17_full_residual_surface_tail_diagnostic_qc"
CLAIM = (
    "This artifact tests the dominant residual class after the pose-enabled full-residual graph: "
    "surface-factor rows that still have depth-tail residuals. It separates local MANO surface "
    "factor failure from rows where the surface factor fits a nearby compatible hand surface "
    "while the depth-tail oracle still trusts incompatible UniDepth pixels."
)

SURFACE_FACTOR_STATE = "relinearized_surface_factor_variable"
LOCAL_STATE = "relinearized_reprojected_local_surface_factor_candidate"
MIXED_STATE = "relinearized_reprojected_mixed_surface_depth_owner"
DEPTH_OBSERVATION_STATE = "relinearized_reprojected_depth_observation_owner"
DEPTH_TAIL_STATE = "depth_tail_incompatible"


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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


def summary_stat(summary: Any, key: str, label: str) -> float | None:
    if summary is None:
        return None
    obj = require_dict(summary, label)
    value = obj.get(key)
    if value is None:
        return None
    return finite_number(value, f"{label}.{key}")


def compact_assignment(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    assignment = require_dict(value, "surface assignment")
    return {
        key: assignment.get(key)
        for key in [
            "residual_sample_count",
            "compatible_seed_sample_count",
            "local_projection_search_radius_px",
            "compatible_depth_abs_m",
            "assigned_residual_sample_count",
            "unassigned_residual_sample_count",
            "nearby_compatible_assignment_fraction",
            "assigned_pixel_shift_px",
            "assigned_source_residual_abs_gap_m",
            "assigned_target_seed_abs_gap_m",
            "assigned_hand_depth_delta_to_seed_m",
            "assigned_metric_depth_delta_to_seed_m",
        ]
    }


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
    expected = require_int(report.get("transition_variable_rows"), f"{label} transition variable rows")
    if len(out) != expected:
        raise RuntimeError(f"{label} transition rows {len(out)} do not match reported variables {expected}")
    return out


def row_summary(
    *,
    graph_id: str,
    pose_row: dict[str, Any],
    factor_row: dict[str, Any],
    geometry_row: dict[str, Any] | None,
    transition_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pose_state = require_str(
        pose_row.get("relinearized_reprojection_state"),
        "pose reprojection state",
    )
    factor_state = require_str(
        factor_row.get("relinearized_input_factor_state"),
        "pose input factor state",
    )
    owner_depth_state = require_str(pose_row.get("owner_depth_state"), "pose owner depth state")
    surface_factor = bool(factor_state == SURFACE_FACTOR_STATE)
    assignment = compact_assignment(factor_row.get("surface_assignment"))
    assigned_fraction = None if assignment is None else optional_finite_number(
        assignment.get("nearby_compatible_assignment_fraction"),
        "nearby compatible assignment fraction",
    )
    assigned_count = 0 if assignment is None else require_int(
        assignment.get("assigned_residual_sample_count"),
        "assigned residual samples",
    )
    unassigned_count = 0 if assignment is None else require_int(
        assignment.get("unassigned_residual_sample_count"),
        "unassigned residual samples",
    )
    selected_count = require_int(
        factor_row.get("selected_residual_sample_count"),
        "selected residual sample count",
    )
    assigned_source_gap_median = None if assignment is None else summary_stat(
        assignment.get("assigned_source_residual_abs_gap_m"),
        "median",
        "assigned source residual gap",
    )
    assigned_target_gap_median = None if assignment is None else summary_stat(
        assignment.get("assigned_target_seed_abs_gap_m"),
        "median",
        "assigned target seed gap",
    )
    assigned_hand_delta_median = None if assignment is None else summary_stat(
        assignment.get("assigned_hand_depth_delta_to_seed_m"),
        "median",
        "assigned hand depth delta to seed",
    )
    geometry_after = None if geometry_row is None else require_dict(
        geometry_row.get("after"),
        "geometry after metrics",
    )
    geometry_depth_median = None if geometry_after is None else optional_finite_number(
        geometry_after.get("depth_abs_median_m"),
        "geometry depth median",
    )
    geometry_depth_p95 = None if geometry_after is None else optional_finite_number(
        geometry_after.get("depth_abs_p95_m"),
        "geometry depth p95",
    )
    geometry_projection_median = None if geometry_after is None else optional_finite_number(
        geometry_after.get("projection_to_seed_median_px"),
        "geometry projection median",
    )
    geometry_joint_p95 = None if geometry_after is None else optional_finite_number(
        geometry_after.get("joint_reprojection_p95_px"),
        "geometry joint p95",
    )
    geometry_depth_pass = bool(
        surface_factor
        and geometry_depth_median is not None
        and geometry_depth_p95 is not None
        and geometry_depth_median <= float(args.max_geometry_depth_median_m)
        and geometry_depth_p95 <= float(args.max_geometry_depth_p95_m)
    )
    assignment_rejects_source_depth = bool(
        surface_factor
        and assigned_source_gap_median is not None
        and assigned_target_gap_median is not None
        and assigned_source_gap_median >= float(args.min_rejected_source_residual_abs_gap_m)
        and assigned_target_gap_median <= float(args.max_assigned_target_seed_abs_gap_m)
    )
    persistent_residual = require_bool(
        transition_row.get("residual_owner_persistent"),
        "residual owner persistent",
    )
    persistent_surface_depth_tail = bool(
        persistent_residual
        and surface_factor
        and owner_depth_state == DEPTH_TAIL_STATE
        and pose_state in {LOCAL_STATE, MIXED_STATE, DEPTH_OBSERVATION_STATE}
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
        "persistent_surface_depth_tail_row": persistent_surface_depth_tail,
        "surface_assignment_fraction": assigned_fraction,
        "selected_residual_sample_count": selected_count,
        "assigned_residual_sample_count": assigned_count,
        "unassigned_residual_sample_count": unassigned_count,
        "surface_assignment_incomplete": bool(surface_factor and unassigned_count > 0),
        "assigned_source_residual_abs_gap_median_m": assigned_source_gap_median,
        "assigned_target_seed_abs_gap_median_m": assigned_target_gap_median,
        "assigned_hand_depth_delta_to_seed_median_m": assigned_hand_delta_median,
        "surface_assignment_rejects_source_depth": assignment_rejects_source_depth,
        "geometry_depth_abs_median_m": geometry_depth_median,
        "geometry_depth_abs_p95_m": geometry_depth_p95,
        "geometry_projection_to_seed_median_px": geometry_projection_median,
        "geometry_joint_reprojection_p95_px": geometry_joint_p95,
        "surface_geometry_depth_pass": geometry_depth_pass,
        "surface_geometry_pass_but_depth_tail": bool(
            persistent_surface_depth_tail and geometry_depth_pass
        ),
        "surface_geometry_pass_and_rejects_source_depth_tail": bool(
            persistent_surface_depth_tail and geometry_depth_pass and assignment_rejects_source_depth
        ),
        "surface_assignment": assignment,
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
    transition = require_dict(load_json(transition_path), f"{case} pose transition diagnostic")
    if require_int(pose_graph.get("frame_count"), f"{case} pose graph frame_count") != require_int(
        transition.get("frame_count"),
        f"{case} transition frame_count",
    ):
        raise RuntimeError(f"{case} pose graph and transition diagnostic frame counts differ")
    pose_rows = applied_rows(pose_graph, f"{case} pose graph")
    factors = factor_rows(pose_graph, f"{case} pose graph")
    geometries = geometry_rows(pose_graph, f"{case} pose graph")
    transitions = transition_rows(transition, f"{case} transition")
    if set(pose_rows) != set(factors):
        raise RuntimeError(f"{case} pose rows and factor rows disagree")
    if set(pose_rows) != set(transitions):
        raise RuntimeError(f"{case} pose rows and transition rows disagree")
    rows = [
        row_summary(
            graph_id=graph_id,
            pose_row=pose_rows[graph_id],
            factor_row=factors[graph_id],
            geometry_row=geometries.get(graph_id),
            transition_row=transitions[graph_id],
            args=args,
        )
        for graph_id in sorted(pose_rows)
    ]
    surface_rows = [row for row in rows if row["surface_factor_row"]]
    persistent_surface_tail = [row for row in rows if row["persistent_surface_depth_tail_row"]]
    out = {
        "method": "build_v17_full_residual_surface_tail_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "pose_full_residual_graph": source_summary(pose_graph_path, pose_graph),
            "pose_transition_diagnostic": source_summary(transition_path, transition),
        },
        "frame_count": require_int(pose_graph.get("frame_count"), f"{case} frame_count"),
        "transition_variable_rows": len(rows),
        "pose_surface_factor_rows": len(surface_rows),
        "pose_surface_geometry_rows": len(geometries),
        "surface_geometry_depth_pass_rows": bool_count(rows, "surface_geometry_depth_pass"),
        "surface_assignment_rejects_source_depth_rows": bool_count(
            rows,
            "surface_assignment_rejects_source_depth",
        ),
        "persistent_surface_depth_tail_rows": len(persistent_surface_tail),
        "persistent_surface_depth_tail_geometry_pass_rows": bool_count(
            persistent_surface_tail,
            "surface_geometry_pass_but_depth_tail",
        ),
        "persistent_surface_depth_tail_rejects_source_depth_rows": bool_count(
            persistent_surface_tail,
            "surface_assignment_rejects_source_depth",
        ),
        "persistent_surface_depth_tail_geometry_pass_and_rejects_source_depth_rows": bool_count(
            persistent_surface_tail,
            "surface_geometry_pass_and_rejects_source_depth_tail",
        ),
        "persistent_surface_depth_tail_unassigned_residual_sample_count": sum(
            require_int(row.get("unassigned_residual_sample_count"), "unassigned samples")
            for row in persistent_surface_tail
        ),
        "surface_assignment_incomplete_rows": bool_count(rows, "surface_assignment_incomplete"),
        "persistent_surface_depth_tail_state_counts": state_counts(
            persistent_surface_tail,
            "pose_reprojection_state",
        ),
        "surface_factor_owner_depth_state_counts": state_counts(surface_rows, "pose_owner_depth_state"),
        "surface_assignment_fraction": numeric_summary(surface_rows, "surface_assignment_fraction"),
        "assigned_source_residual_abs_gap_median_m": numeric_summary(
            surface_rows,
            "assigned_source_residual_abs_gap_median_m",
        ),
        "assigned_target_seed_abs_gap_median_m": numeric_summary(
            surface_rows,
            "assigned_target_seed_abs_gap_median_m",
        ),
        "assigned_hand_depth_delta_to_seed_median_m": numeric_summary(
            surface_rows,
            "assigned_hand_depth_delta_to_seed_median_m",
        ),
        "geometry_depth_abs_median_m": numeric_summary(surface_rows, "geometry_depth_abs_median_m"),
        "geometry_depth_abs_p95_m": numeric_summary(surface_rows, "geometry_depth_abs_p95_m"),
        "problem_semantics": {
            "surface_geometry_pass_but_depth_tail": (
                "the local MANO surface factor is within the depth thresholds, but the row remains "
                "a post-reprojection depth-tail residual"
            ),
            "surface_assignment_rejects_source_depth": (
                "assigned residual pixels have large original hand-vs-UniDepth gaps while the "
                "nearby seed pixels used as surface targets are already depth-compatible"
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_full_residual_surface_tail_diagnostic.json", out)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [row for case in cases for row in require_list(case.get("rows"), "case rows")]
    surface_rows = [row for row in rows if row["surface_factor_row"]]
    persistent_surface_tail = [row for row in rows if row["persistent_surface_depth_tail_row"]]
    summary = {
        "method": "build_v17_full_residual_surface_tail_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame count") for case in cases),
        "transition_variable_rows": len(rows),
        "pose_surface_factor_rows": len(surface_rows),
        "pose_surface_geometry_rows": sum(
            require_int(case.get("pose_surface_geometry_rows"), "case geometry rows") for case in cases
        ),
        "surface_geometry_depth_pass_rows": bool_count(rows, "surface_geometry_depth_pass"),
        "surface_assignment_rejects_source_depth_rows": bool_count(
            rows,
            "surface_assignment_rejects_source_depth",
        ),
        "persistent_surface_depth_tail_rows": len(persistent_surface_tail),
        "persistent_surface_depth_tail_geometry_pass_rows": bool_count(
            persistent_surface_tail,
            "surface_geometry_pass_but_depth_tail",
        ),
        "persistent_surface_depth_tail_rejects_source_depth_rows": bool_count(
            persistent_surface_tail,
            "surface_assignment_rejects_source_depth",
        ),
        "persistent_surface_depth_tail_geometry_pass_and_rejects_source_depth_rows": bool_count(
            persistent_surface_tail,
            "surface_geometry_pass_and_rejects_source_depth_tail",
        ),
        "persistent_surface_depth_tail_unassigned_residual_sample_count": sum(
            require_int(row.get("unassigned_residual_sample_count"), "unassigned samples")
            for row in persistent_surface_tail
        ),
        "surface_assignment_incomplete_rows": bool_count(rows, "surface_assignment_incomplete"),
        "persistent_surface_depth_tail_state_counts": state_counts(
            persistent_surface_tail,
            "pose_reprojection_state",
        ),
        "surface_factor_owner_depth_state_counts": state_counts(surface_rows, "pose_owner_depth_state"),
        "surface_assignment_fraction": numeric_summary(surface_rows, "surface_assignment_fraction"),
        "assigned_source_residual_abs_gap_median_m": numeric_summary(
            surface_rows,
            "assigned_source_residual_abs_gap_median_m",
        ),
        "assigned_target_seed_abs_gap_median_m": numeric_summary(
            surface_rows,
            "assigned_target_seed_abs_gap_median_m",
        ),
        "assigned_hand_depth_delta_to_seed_median_m": numeric_summary(
            surface_rows,
            "assigned_hand_depth_delta_to_seed_median_m",
        ),
        "geometry_depth_abs_median_m": numeric_summary(surface_rows, "geometry_depth_abs_median_m"),
        "geometry_depth_abs_p95_m": numeric_summary(surface_rows, "geometry_depth_abs_p95_m"),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame count"),
                "pose_surface_factor_rows": require_int(
                    case.get("pose_surface_factor_rows"),
                    "case surface rows",
                ),
                "persistent_surface_depth_tail_rows": require_int(
                    case.get("persistent_surface_depth_tail_rows"),
                    "case persistent surface tail rows",
                ),
                "persistent_surface_depth_tail_geometry_pass_rows": require_int(
                    case.get("persistent_surface_depth_tail_geometry_pass_rows"),
                    "case geometry pass tail rows",
                ),
                "persistent_surface_depth_tail_geometry_pass_and_rejects_source_depth_rows": require_int(
                    case.get("persistent_surface_depth_tail_geometry_pass_and_rejects_source_depth_rows"),
                    "case geometry pass source-reject rows",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_full_residual_surface_tail_diagnostic_summary.json", summary)
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
        default=Path("/data2/ego_annotation_outputs/v17_full_residual_surface_tail_diagnostic"),
    )
    parser.add_argument("--max-geometry-depth-median-m", type=float, default=0.03)
    parser.add_argument("--max-geometry-depth-p95-m", type=float, default=0.08)
    parser.add_argument("--min-rejected-source-residual-abs-gap-m", type=float, default=0.08)
    parser.add_argument("--max-assigned-target-seed-abs-gap-m", type=float, default=0.03)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
