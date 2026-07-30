#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

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


STATUS = "v17_hand_residual_switch_problem_qc"
CLAIM = (
    "This artifact materializes the discrete owner switches for post-repair V17 hand-depth residuals. "
    "It attaches residual-owner evidence, local projection assignment evidence, and local MANO articulation "
    "solve evidence to one switch variable per remaining residual row. It does not update hand annotations; "
    "it exposes which residual rows still require local surface/articulation, mixed projection-depth, "
    "depth-observation or occlusion, or projection-support variables in the complete joint solver."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def local_articulation_ready(row: dict[str, Any], max_pose_delta_rad: float) -> bool:
    if row.get("local_articulation_solve_state") != "local_articulation_factor_solved_under_local_thresholds":
        return False
    pose_delta = finite_float(row.get("pose_delta_abs_max_rad"), "pose_delta_abs_max_rad")
    return pose_delta < max_pose_delta_rad - 1.0e-5


def local_switch_state(
    local_row: dict[str, Any],
    articulation_row: dict[str, Any] | None,
    max_pose_delta_rad: float,
) -> str:
    if local_row.get("local_projection_repair_factor_candidate") is not True:
        raise RuntimeError("local_switch_state requires a local projection candidate")
    if articulation_row is None:
        return "local_surface_candidate_missing_articulation_solve"
    if local_articulation_ready(articulation_row, max_pose_delta_rad):
        return "local_surface_articulation_factor_ready"
    pose_delta = finite_float(articulation_row.get("pose_delta_abs_max_rad"), "pose_delta_abs_max_rad")
    if pose_delta >= max_pose_delta_rad - 1.0e-5:
        return "local_surface_articulation_pose_bound_switch_required"
    if articulation_row.get("local_articulation_projection_trusted") is not True:
        return "local_surface_articulation_projection_switch_required"
    if articulation_row.get("local_articulation_depth_improved") is not True:
        return "local_surface_articulation_no_gain_switch_required"
    return "local_surface_articulation_depth_switch_required"


def residual_switch_state(
    local_row: dict[str, Any],
    articulation_row: dict[str, Any] | None,
    max_pose_delta_rad: float,
) -> str:
    if local_row.get("projection_support_unresolved") is True:
        return "projection_support_switch_required"
    if local_row.get("depth_observation_or_occlusion_owner") is True:
        return "depth_observation_or_occlusion_switch_required"
    if local_row.get("partial_projection_depth_mixed_owner") is True:
        return "mixed_projection_depth_switch_required"
    if local_row.get("local_projection_repair_factor_candidate") is True:
        return local_switch_state(local_row, articulation_row, max_pose_delta_rad)
    raise RuntimeError(
        f"unclassified residual switch owner: {local_row.get('source_hand_depth_repair_graph_variable_id')}"
    )


def assignment_evidence(local_row: dict[str, Any]) -> dict[str, Any]:
    assignment = require_dict(local_row.get("assignment"), "local projection assignment")
    return {
        "residual_sample_count": require_int(assignment.get("residual_sample_count"), "residual sample count"),
        "assigned_residual_sample_count": require_int(
            assignment.get("assigned_residual_sample_count"),
            "assigned residual sample count",
        ),
        "unassigned_residual_sample_count": require_int(
            assignment.get("unassigned_residual_sample_count"),
            "unassigned residual sample count",
        ),
        "compatible_seed_sample_count": require_int(
            assignment.get("compatible_seed_sample_count"),
            "compatible seed sample count",
        ),
        "owner_compatible_seed_sample_count": require_int(
            assignment.get("owner_compatible_seed_sample_count"),
            "owner compatible seed sample count",
        ),
        "nearby_compatible_assignment_fraction": optional_float(
            assignment.get("nearby_compatible_assignment_fraction")
        ),
        "selected_residual_gap_m": require_dict(
            assignment.get("selected_residual_gap_m"),
            "selected residual gap summary",
        ),
        "assigned_source_residual_abs_gap_m": require_dict(
            assignment.get("assigned_source_residual_abs_gap_m"),
            "assigned source residual gap summary",
        ),
        "assigned_target_seed_abs_gap_m": require_dict(
            assignment.get("assigned_target_seed_abs_gap_m"),
            "assigned target seed gap summary",
        ),
    }


def articulation_evidence(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "local_articulation_solve_state": require_str(
            row.get("local_articulation_solve_state"),
            "local articulation solve state",
        ),
        "local_articulation_depth_improved": bool(row.get("local_articulation_depth_improved") is True),
        "local_articulation_depth_threshold_met": bool(
            row.get("local_articulation_depth_threshold_met") is True
        ),
        "local_articulation_projection_trusted": bool(
            row.get("local_articulation_projection_trusted") is True
        ),
        "depth_abs_median_improvement_m": finite_float(
            row.get("depth_abs_median_improvement_m"),
            "depth_abs_median_improvement_m",
        ),
        "before_depth_abs_median_m": finite_float(
            require_dict(row.get("before"), "before").get("depth_abs_median_m"),
            "before depth_abs_median_m",
        ),
        "after_depth_abs_median_m": finite_float(
            require_dict(row.get("after"), "after").get("depth_abs_median_m"),
            "after depth_abs_median_m",
        ),
        "after_depth_abs_p95_m": finite_float(
            require_dict(row.get("after"), "after").get("depth_abs_p95_m"),
            "after depth_abs_p95_m",
        ),
        "after_joint_reprojection_median_px": finite_float(
            require_dict(row.get("after"), "after").get("joint_reprojection_median_px"),
            "after joint_reprojection_median_px",
        ),
        "after_joint_reprojection_p95_px": finite_float(
            require_dict(row.get("after"), "after").get("joint_reprojection_p95_px"),
            "after joint_reprojection_p95_px",
        ),
        "pose_delta_abs_max_rad": finite_float(row.get("pose_delta_abs_max_rad"), "pose_delta_abs_max_rad"),
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_local_projection_repair_problem": existing_path(
            args.hand_local_projection_repair_problem_root
            / case
            / "v17_hand_local_projection_repair_problem.json",
            f"{case} hand local projection repair problem",
        ),
        "mano_articulation_local_solve": existing_path(
            args.mano_articulation_local_solve_root
            / case
            / "v17_mano_articulation_local_solve.json",
            f"{case} MANO local articulation solve",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    local_report = payloads["hand_local_projection_repair_problem"]
    articulation_report = payloads["mano_articulation_local_solve"]
    frame_count = require_int(local_report.get("frame_count"), f"{case} local projection frame_count")
    if frame_count != require_int(articulation_report.get("frame_count"), f"{case} articulation frame_count"):
        raise RuntimeError(f"{case} local projection and articulation frame counts disagree")
    max_pose_delta_rad = finite_float(
        require_dict(articulation_report.get("parameters"), "articulation parameters").get(
            "max_pose_delta_rad"
        ),
        "max_pose_delta_rad",
    )
    articulation_rows = [
        require_dict(raw, "articulation row")
        for raw in require_list(articulation_report.get("rows"), f"{case} articulation rows")
    ]
    articulation_by_graph_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "articulation graph id"): row
        for row in articulation_rows
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(local_report.get("rows"), f"{case} local projection rows"):
        local_row = require_dict(raw, "local projection row")
        if local_row.get("repair_residual_factor_candidate") is not True:
            continue
        graph_id = require_str(
            local_row.get("source_hand_depth_repair_graph_variable_id"),
            "local projection graph id",
        )
        articulation_row = articulation_by_graph_id.get(graph_id)
        switch_state = residual_switch_state(local_row, articulation_row, max_pose_delta_rad)
        switch_ready = bool(switch_state == "local_surface_articulation_factor_ready")
        rows.append(
            {
                "case": case,
                "hand_residual_switch_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "hand_residual_switch:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_local_projection_repair_variable_id": require_str(
                    local_row.get("hand_local_projection_repair_variable_id"),
                    "local projection variable id",
                ),
                "source_mano_articulation_local_solve_variable_id": None
                if articulation_row is None
                else require_str(
                    articulation_row.get("mano_local_articulation_solve_variable_id"),
                    "articulation variable id",
                ),
                "frame_idx": require_int(local_row.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(local_row.get("hand_side"), "hand_side"),
                "hand_index": require_int(local_row.get("hand_index"), "hand_index"),
                "residual_owner_state": local_row.get("residual_owner_state"),
                "residual_depth_observation_state": local_row.get("residual_depth_observation_state"),
                "residual_independent_support_state": local_row.get("residual_independent_support_state"),
                "local_projection_repair_state": local_row.get("local_projection_repair_state"),
                "owner_sample_partition": local_row.get("owner_sample_partition"),
                "local_projection_repair_factor_candidate": bool(
                    local_row.get("local_projection_repair_factor_candidate") is True
                ),
                "partial_projection_depth_mixed_owner": bool(
                    local_row.get("partial_projection_depth_mixed_owner") is True
                ),
                "depth_observation_or_occlusion_owner": bool(
                    local_row.get("depth_observation_or_occlusion_owner") is True
                ),
                "projection_support_unresolved": bool(
                    local_row.get("projection_support_unresolved") is True
                ),
                "assignment": assignment_evidence(local_row),
                "articulation": articulation_evidence(articulation_row),
                "residual_switch_state": switch_state,
                "residual_switch_ready": switch_ready,
                **FALSE_READY,
            }
        )
    local_candidate_rows = [row for row in rows if row.get("local_projection_repair_factor_candidate") is True]
    articulation_attached_rows = [row for row in local_candidate_rows if row.get("articulation") is not None]
    if len(articulation_attached_rows) != len(local_candidate_rows):
        raise RuntimeError(f"{case} local projection candidates without articulation solve evidence")
    report = {
        "method": "build_v17_hand_residual_switch_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_residual_switch_variable_count": len(rows),
        "local_projection_candidate_rows": len(local_candidate_rows),
        "local_articulation_solve_attached_rows": len(articulation_attached_rows),
        "local_articulation_factor_ready_rows": bool_count(rows, "residual_switch_ready"),
        "mixed_projection_depth_switch_rows": bool_count(rows, "partial_projection_depth_mixed_owner"),
        "depth_observation_or_occlusion_switch_rows": bool_count(
            rows,
            "depth_observation_or_occlusion_owner",
        ),
        "projection_support_switch_rows": bool_count(rows, "projection_support_unresolved"),
        "residual_switch_state_counts": state_counts(rows, "residual_switch_state"),
        "articulation_pose_delta_abs_max_rad": summarize(
            [
                finite_float(
                    require_dict(row.get("articulation"), "articulation").get("pose_delta_abs_max_rad"),
                    "pose delta",
                )
                for row in articulation_attached_rows
            ]
        ),
        "articulation_depth_abs_median_improvement_m": summarize(
            [
                finite_float(
                    require_dict(row.get("articulation"), "articulation").get(
                        "depth_abs_median_improvement_m"
                    ),
                    "depth improvement",
                )
                for row in articulation_attached_rows
            ]
        ),
        "local_projection_assignment": {
            "residual_sample_count": sum(
                require_int(require_dict(row.get("assignment"), "assignment").get("residual_sample_count"), "samples")
                for row in rows
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                    "assigned samples",
                )
                for row in rows
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(row.get("assignment"), "assignment").get("compatible_seed_sample_count"),
                    "compatible seed samples",
                )
                for row in rows
            ),
        },
        "source_problem_comparison": {
            "repair_residual_factor_candidate_rows": local_report.get(
                "repair_residual_factor_candidate_rows"
            ),
            "local_projection_repair_factor_candidate_rows": local_report.get(
                "local_projection_repair_factor_candidate_rows"
            ),
            "partial_projection_depth_mixed_owner_rows": local_report.get(
                "partial_projection_depth_mixed_owner_rows"
            ),
            "depth_observation_or_occlusion_owner_rows": local_report.get(
                "depth_observation_or_occlusion_owner_rows"
            ),
            "projection_support_unresolved_rows": local_report.get("projection_support_unresolved_rows"),
            "mano_local_articulation_solve_candidate_rows": articulation_report.get(
                "mano_local_articulation_solve_candidate_rows"
            ),
            "local_articulation_pose_delta_clamp_hit_rows": articulation_report.get(
                "local_articulation_pose_delta_clamp_hit_rows"
            ),
        },
        "problem_semantics": {
            "local_surface_articulation_factor_ready": "local MANO articulation causally reduces the residual, passes depth and projection predicates, and stays inside the pose-delta bound",
            "local_surface_articulation_pose_bound_switch_required": "local MANO articulation was the active surface mechanism but hit the pose-delta bound",
            "local_surface_articulation_no_gain_switch_required": "local MANO articulation satisfies some local predicates but does not causally reduce the residual",
            "local_surface_articulation_depth_switch_required": "local MANO articulation stayed within the pose bound but did not meet local depth thresholds",
            "local_surface_articulation_projection_switch_required": "local MANO articulation improved depth but did not preserve projection trust",
            "mixed_projection_depth_switch_required": "nearby compatible depth explains only part of the residual row",
            "depth_observation_or_occlusion_switch_required": "supported hand evidence lacks nearby compatible depth samples",
            "projection_support_switch_required": "residual pixels lack independent hand support",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_residual_switch_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_local_projection_repair_problem_root
        / "v17_hand_local_projection_repair_problem_summary.json",
        "hand local projection repair summary",
    )
    local_summary = require_dict(load_json(summary_path), "hand local projection repair summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(local_summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_residual_switch_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_local_projection_repair_problem_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_residual_switch_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_residual_switch_variable_count": require_int(
                    report.get("hand_residual_switch_variable_count"),
                    "switch variable count",
                ),
                "local_projection_candidate_rows": require_int(
                    report.get("local_projection_candidate_rows"),
                    "local projection rows",
                ),
                "local_articulation_solve_attached_rows": require_int(
                    report.get("local_articulation_solve_attached_rows"),
                    "local articulation attached rows",
                ),
                "local_articulation_factor_ready_rows": require_int(
                    report.get("local_articulation_factor_ready_rows"),
                    "local articulation ready rows",
                ),
                "mixed_projection_depth_switch_rows": require_int(
                    report.get("mixed_projection_depth_switch_rows"),
                    "mixed switch rows",
                ),
                "depth_observation_or_occlusion_switch_rows": require_int(
                    report.get("depth_observation_or_occlusion_switch_rows"),
                    "depth switch rows",
                ),
                "projection_support_switch_rows": require_int(
                    report.get("projection_support_switch_rows"),
                    "projection switch rows",
                ),
                "residual_switch_state_counts": require_dict(
                    report.get("residual_switch_state_counts"),
                    "switch state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_residual_switch_variable_count": sum(
            require_int(report.get("hand_residual_switch_variable_count"), "switch variable count")
            for report in reports
        ),
        "local_projection_candidate_rows": sum(
            require_int(report.get("local_projection_candidate_rows"), "local projection rows")
            for report in reports
        ),
        "local_articulation_solve_attached_rows": sum(
            require_int(report.get("local_articulation_solve_attached_rows"), "attached rows")
            for report in reports
        ),
        "local_articulation_factor_ready_rows": sum(
            require_int(report.get("local_articulation_factor_ready_rows"), "ready rows")
            for report in reports
        ),
        "mixed_projection_depth_switch_rows": sum(
            require_int(report.get("mixed_projection_depth_switch_rows"), "mixed rows")
            for report in reports
        ),
        "depth_observation_or_occlusion_switch_rows": sum(
            require_int(report.get("depth_observation_or_occlusion_switch_rows"), "depth rows")
            for report in reports
        ),
        "projection_support_switch_rows": sum(
            require_int(report.get("projection_support_switch_rows"), "projection rows")
            for report in reports
        ),
        "residual_switch_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("residual_switch_state_counts"), "state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "articulation_pose_delta_abs_max_rad": {
            "case_summaries": [
                require_dict(report.get("articulation_pose_delta_abs_max_rad"), "pose summary")
                for report in reports
            ]
        },
        "articulation_depth_abs_median_improvement_m": {
            "case_summaries": [
                require_dict(
                    report.get("articulation_depth_abs_median_improvement_m"),
                    "improvement summary",
                )
                for report in reports
            ]
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_residual_switch_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-local-projection-repair-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem"),
    )
    parser.add_argument(
        "--mano-articulation-local-solve-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_articulation_local_solve"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_residual_switch_problem"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
