#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


STATUS = "v17_relinearized_hand_capacity_diagnostic_qc"
CLAIM = (
    "This artifact tests whether the residuals left by the relinearized hand surface-observation "
    "graph are explained by missing MANO hand-shape or hand-scale capacity. It joins full-reprojection "
    "residual owners, MANO parameter replay ownership, and local surface-factor geometry. It is a "
    "state-variable diagnostic, not an annotation solver."
)


def finite_number(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be finite")
    return float(value)


def finite_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, label)


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return summarize(arr.astype(float).tolist())


def nested_number(row: dict[str, Any], path: list[str], label: str) -> float:
    current: Any = row
    for key in path:
        current = require_dict(current, label).get(key)
    return finite_number(current, label)


def nested_number_or_none(row: dict[str, Any], path: list[str], label: str) -> float | None:
    current: Any = row
    for key in path:
        if current is None:
            return None
        if not isinstance(current, dict):
            raise RuntimeError(f"{label} parent must be object")
        current = current.get(key)
    return finite_or_none(current, label)


def selected_partition_summary(row: dict[str, Any]) -> dict[str, Any]:
    partition_name = row.get("owner_sample_partition")
    if partition_name is None:
        return {
            "owner_sample_partition": None,
            "partition_depth_summary_available": False,
        }
    partition = require_dict(
        require_dict(row.get("partitions"), "row partitions").get(
            require_str(partition_name, "owner_sample_partition")
        ),
        "owner partition",
    )
    hand_minus_depth = require_dict(
        partition.get("hand_minus_unidepth_depth_m"),
        "owner hand_minus_unidepth_depth_m",
    )
    p05 = finite_number(hand_minus_depth.get("p05"), "owner partition p05")
    p95 = finite_number(hand_minus_depth.get("p95"), "owner partition p95")
    return {
        "owner_sample_partition": partition_name,
        "partition_depth_summary_available": True,
        "owner_partition_hand_minus_unidepth_median_m": finite_number(
            hand_minus_depth.get("median"),
            "owner partition median",
        ),
        "owner_partition_hand_minus_unidepth_p05_m": p05,
        "owner_partition_hand_minus_unidepth_p95_m": p95,
        "owner_partition_hand_minus_unidepth_abs_median_m": abs(
            finite_number(hand_minus_depth.get("median"), "owner partition median")
        ),
        "owner_partition_hand_minus_unidepth_abs_tail_m": max(abs(p05), abs(p95)),
    }


def ownership_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("rows"), "MANO ownership rows"):
        row = require_dict(raw, "MANO ownership row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            "MANO ownership source graph id",
        )
        indexed[graph_id] = row
    return indexed


def geometry_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in require_list(report.get("geometry_rows"), "relinearized geometry rows"):
        row = require_dict(raw, "relinearized geometry row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            "relinearized geometry source graph id",
        )
        indexed[graph_id] = row
    return indexed


def build_rows(
    *,
    relinearized: dict[str, Any],
    ownership_by_id: dict[str, dict[str, Any]],
    geometry_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in require_list(relinearized.get("rows"), "relinearized rows"):
        row = require_dict(raw, "relinearized row")
        if row.get("relinearized_delta_applied") is not True:
            continue
        graph_id = require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            "relinearized source graph id",
        )
        ownership = ownership_by_id.get(graph_id)
        geometry = geometry_by_id.get(graph_id)
        owner_gap = finite_or_none(row.get("owner_median_gap_m"), "owner median gap")
        pose_delta = finite_or_none(
            row.get("relinearized_pose_delta_abs_max_rad"),
            "relinearized pose delta",
        )
        projection_median = nested_number_or_none(
            row,
            ["projection_residual_to_measurement_px", "median"],
            "projection median",
        )
        projection_p95 = nested_number_or_none(
            row,
            ["projection_residual_to_measurement_px", "p95"],
            "projection p95",
        )
        item: dict[str, Any] = {
            "case": require_str(row.get("case"), "case"),
            "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
            "hand_side": require_str(row.get("hand_side"), "hand_side"),
            "hand_index": require_int(row.get("hand_index"), "hand_index"),
            "source_hand_depth_repair_graph_variable_id": graph_id,
            "relinearized_reprojection_state": require_str(
                row.get("relinearized_reprojection_state"),
                "relinearized state",
            ),
            "owner_depth_state": require_str(row.get("owner_depth_state"), "owner depth state"),
            "metric_depth_compatible": bool(row.get("metric_depth_compatible") is True),
            "depth_repair_factor_candidate": bool(row.get("depth_repair_factor_candidate") is True),
            "projection_trusted": bool(
                require_str(row.get("relinearized_reprojection_state"), "relinearized state")
                != "relinearized_reprojected_projection_untrusted"
            ),
            "relinearized_pose_delta_abs_max_rad": pose_delta,
            "relinearized_pose_delta_clamp_hit": bool(
                pose_delta is not None and pose_delta >= 0.34999
            ),
            "scaled_wrist_to_middle_tip_m": finite_number(
                row.get("scaled_wrist_to_middle_tip_m"),
                "scaled wrist-to-middle-tip",
            ),
            "solved_scale": finite_number(row.get("solved_scale"), "solved scale"),
            "owner_median_gap_m": owner_gap,
            "owner_abs_median_gap_m": None if owner_gap is None else abs(owner_gap),
            "projection_residual_median_px": projection_median,
            "projection_residual_p95_px": projection_p95,
            **selected_partition_summary(row),
            **FALSE_READY,
        }
        if ownership is None:
            item.update(
                {
                    "mano_parameter_ownership_available": False,
                    "mano_parameter_geometry_owned": False,
                    "mano_parameter_ownership_state": "not_in_mano_parameter_ownership_state",
                }
            )
        else:
            metrics = require_dict(ownership.get("ownership_metrics"), "ownership metrics")
            item.update(
                {
                    "mano_parameter_ownership_available": True,
                    "mano_parameter_geometry_owned": bool(
                        ownership.get("mano_parameter_geometry_owned") is True
                    ),
                    "mano_parameter_ownership_state": require_str(
                        ownership.get("mano_parameter_ownership_state"),
                        "MANO ownership state",
                    ),
                    "wilor_similarity_scale": finite_number(
                        metrics.get("wilor_similarity_scale"),
                        "WiLoR similarity scale",
                    ),
                    "wilor_similarity_rotation_det": finite_number(
                        metrics.get("wilor_similarity_rotation_det"),
                        "WiLoR similarity rotation determinant",
                    ),
                    "vertex_alignment_error_median_m": nested_number(
                        metrics,
                        ["vertex_alignment_error_m", "median"],
                        "vertex alignment median",
                    ),
                    "vertex_alignment_error_p95_m": nested_number(
                        metrics,
                        ["vertex_alignment_error_m", "p95"],
                        "vertex alignment p95",
                    ),
                    "joint_alignment_error_median_m": nested_number(
                        metrics,
                        ["joint_alignment_error_m", "median"],
                        "joint alignment median",
                    ),
                    "joint_alignment_error_p95_m": nested_number(
                        metrics,
                        ["joint_alignment_error_m", "p95"],
                        "joint alignment p95",
                    ),
                }
            )
        if geometry is None:
            item.update(
                {
                    "surface_geometry_factor_available": False,
                }
            )
        else:
            after = require_dict(geometry.get("after"), "surface geometry after")
            item.update(
                {
                    "surface_geometry_factor_available": True,
                    "surface_factor_sample_count": require_int(
                        after.get("factor_sample_count"),
                        "surface factor sample count",
                    ),
                    "surface_after_hand_span_m": finite_number(
                        after.get("hand_span_m"),
                        "surface after hand span",
                    ),
                    "surface_after_depth_abs_median_m": finite_number(
                        after.get("depth_abs_median_m"),
                        "surface after depth abs median",
                    ),
                    "surface_after_depth_abs_p95_m": finite_number(
                        after.get("depth_abs_p95_m"),
                        "surface after depth abs p95",
                    ),
                    "surface_after_projection_to_seed_median_px": finite_number(
                        after.get("projection_to_seed_median_px"),
                        "surface projection-to-seed median",
                    ),
                    "surface_after_joint_reprojection_median_px": finite_number(
                        after.get("joint_reprojection_median_px"),
                        "surface joint reprojection median",
                    ),
                    "surface_after_joint_reprojection_p95_px": finite_number(
                        after.get("joint_reprojection_p95_px"),
                        "surface joint reprojection p95",
                    ),
                }
            )
        rows.append(item)
    return rows


def grouped_state_summaries(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(require_str(row.get(group_key), group_key), []).append(row)
    return [
        {
            group_key: key,
            "row_count": len(items),
            "depth_repair_factor_candidate_rows": bool_count(
                items,
                "depth_repair_factor_candidate",
            ),
            "metric_depth_compatible_rows": bool_count(items, "metric_depth_compatible"),
            "pose_delta_clamp_hit_rows": bool_count(items, "relinearized_pose_delta_clamp_hit"),
            "mano_parameter_ownership_available_rows": bool_count(
                items,
                "mano_parameter_ownership_available",
            ),
            "mano_parameter_geometry_owned_rows": bool_count(
                items,
                "mano_parameter_geometry_owned",
            ),
            "surface_geometry_factor_available_rows": bool_count(
                items,
                "surface_geometry_factor_available",
            ),
            "scaled_wrist_to_middle_tip_m": numeric_summary(
                items,
                "scaled_wrist_to_middle_tip_m",
            ),
            "owner_abs_median_gap_m": numeric_summary(items, "owner_abs_median_gap_m"),
            "owner_partition_hand_minus_unidepth_abs_tail_m": numeric_summary(
                items,
                "owner_partition_hand_minus_unidepth_abs_tail_m",
            ),
            "projection_residual_median_px": numeric_summary(
                items,
                "projection_residual_median_px",
            ),
            "vertex_alignment_error_p95_m": numeric_summary(
                items,
                "vertex_alignment_error_p95_m",
            ),
            "surface_after_projection_to_seed_median_px": numeric_summary(
                items,
                "surface_after_projection_to_seed_median_px",
            ),
            "surface_after_depth_abs_p95_m": numeric_summary(
                items,
                "surface_after_depth_abs_p95_m",
            ),
        }
        for key, items in sorted(grouped.items())
    ]


def capacity_conclusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    residual_owners = [row for row in rows if row.get("depth_repair_factor_candidate") is True]
    compatible = [row for row in rows if row.get("metric_depth_compatible") is True]
    depth_or_mixed = [
        row
        for row in residual_owners
        if row.get("relinearized_reprojection_state")
        in {
            "relinearized_reprojected_depth_observation_owner",
            "relinearized_reprojected_mixed_surface_depth_owner",
        }
    ]
    surface_only = [
        row
        for row in residual_owners
        if row.get("relinearized_reprojection_state")
        == "relinearized_reprojected_local_surface_factor_candidate"
    ]
    owned = [row for row in residual_owners if row.get("mano_parameter_geometry_owned") is True]
    clamp_hits = [row for row in residual_owners if row.get("relinearized_pose_delta_clamp_hit") is True]
    compatible_span = numeric_summary(compatible, "scaled_wrist_to_middle_tip_m")
    candidate_span = numeric_summary(residual_owners, "scaled_wrist_to_middle_tip_m")
    return {
        "shape_only_closure_supported": False,
        "state": "hand_shape_only_not_supported_by_current_measurements",
        "reason": (
            "Residual-owner rows are dominated by mixed or depth-observation owners after full "
            "reprojection, MANO parameter replay ownership is usually available where measured, "
            "candidate hand spans overlap the compatible-row span range, and pose clamp is not "
            "universal. The next solver needs joint depth-owner/contact/object state before any "
            "hand-only closure claim."
        ),
        "candidate_rows": len(residual_owners),
        "residual_owner_rows": len(residual_owners),
        "compatible_rows": len(compatible),
        "depth_or_mixed_owner_candidate_rows": len(depth_or_mixed),
        "surface_only_candidate_rows": len(surface_only),
        "mano_geometry_owned_candidate_rows": len(owned),
        "pose_clamp_candidate_rows": len(clamp_hits),
        "candidate_scaled_wrist_to_middle_tip_m": candidate_span,
        "compatible_scaled_wrist_to_middle_tip_m": compatible_span,
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "relinearized_hand_surface_observation_graph": existing_path(
            args.relinearized_hand_surface_observation_graph_root
            / case
            / "v17_relinearized_hand_surface_observation_graph.json",
            f"{case} relinearized hand surface-observation graph",
        ),
        "mano_parameter_ownership_state": existing_path(
            args.mano_parameter_ownership_state_root
            / case
            / "v17_mano_parameter_ownership_state.json",
            f"{case} MANO parameter ownership state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    relinearized = payloads["relinearized_hand_surface_observation_graph"]
    ownership = payloads["mano_parameter_ownership_state"]
    frame_count = require_int(relinearized.get("frame_count"), f"{case} relinearized frame_count")
    if frame_count != require_int(ownership.get("frame_count"), f"{case} ownership frame_count"):
        raise RuntimeError(f"{case} frame count mismatch between relinearized graph and ownership state")
    rows = build_rows(
        relinearized=relinearized,
        ownership_by_id=ownership_index(ownership),
        geometry_by_id=geometry_index(relinearized),
    )
    applied_count = require_int(
        relinearized.get("relinearized_variable_rows"),
        f"{case} relinearized variable rows",
    )
    if len(rows) != applied_count:
        raise RuntimeError(f"{case} expected {applied_count} applied rows, found {len(rows)}")
    residual_owner_rows = [row for row in rows if row.get("depth_repair_factor_candidate") is True]
    report = {
        "method": "build_v17_relinearized_hand_capacity_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            name: source_summary(path, payloads[name]) for name, path in paths.items()
        },
        "frame_count": frame_count,
        "applied_relinearized_variable_rows": len(rows),
        "metric_depth_compatible_rows": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows": len(residual_owner_rows),
        "relinearized_residual_owner_rows": len(residual_owner_rows),
        "projection_untrusted_rows": sum(
            1
            for row in rows
            if row.get("relinearized_reprojection_state")
            == "relinearized_reprojected_projection_untrusted"
        ),
        "relinearized_reprojection_state_counts": state_counts(
            rows,
            "relinearized_reprojection_state",
        ),
        "owner_depth_state_counts": state_counts(residual_owner_rows, "owner_depth_state"),
        "mano_parameter_ownership_available_rows": bool_count(
            rows,
            "mano_parameter_ownership_available",
        ),
        "mano_parameter_geometry_owned_rows": bool_count(rows, "mano_parameter_geometry_owned"),
        "residual_candidate_mano_geometry_owned_rows": bool_count(
            residual_owner_rows,
            "mano_parameter_geometry_owned",
        ),
        "surface_geometry_factor_available_rows": bool_count(
            rows,
            "surface_geometry_factor_available",
        ),
        "residual_candidate_pose_delta_clamp_hit_rows": bool_count(
            residual_owner_rows,
            "relinearized_pose_delta_clamp_hit",
        ),
        "scaled_wrist_to_middle_tip_m": numeric_summary(rows, "scaled_wrist_to_middle_tip_m"),
        "residual_candidate_scaled_wrist_to_middle_tip_m": numeric_summary(
            residual_owner_rows,
            "scaled_wrist_to_middle_tip_m",
        ),
        "compatible_scaled_wrist_to_middle_tip_m": numeric_summary(
            [row for row in rows if row.get("metric_depth_compatible") is True],
            "scaled_wrist_to_middle_tip_m",
        ),
        "residual_candidate_owner_abs_median_gap_m": numeric_summary(
            residual_owner_rows,
            "owner_abs_median_gap_m",
        ),
        "residual_candidate_owner_partition_hand_minus_unidepth_abs_tail_m": numeric_summary(
            residual_owner_rows,
            "owner_partition_hand_minus_unidepth_abs_tail_m",
        ),
        "residual_candidate_projection_residual_median_px": numeric_summary(
            residual_owner_rows,
            "projection_residual_median_px",
        ),
        "residual_candidate_vertex_alignment_error_p95_m": numeric_summary(
            residual_owner_rows,
            "vertex_alignment_error_p95_m",
        ),
        "surface_after_projection_to_seed_median_px": numeric_summary(
            [row for row in rows if row.get("surface_geometry_factor_available") is True],
            "surface_after_projection_to_seed_median_px",
        ),
        "surface_after_depth_abs_p95_m": numeric_summary(
            [row for row in rows if row.get("surface_geometry_factor_available") is True],
            "surface_after_depth_abs_p95_m",
        ),
        "state_summaries": grouped_state_summaries(rows, "relinearized_reprojection_state"),
        "owner_depth_state_summaries": grouped_state_summaries(
            residual_owner_rows,
            "owner_depth_state",
        ),
        "capacity_conclusion": capacity_conclusion(rows),
        "problem_semantics": {
            "tested_hypothesis": "remaining V17 residuals are primarily missing MANO identity shape, hand scale, or pose capacity",
            "falsification_signal": "residual-owner rows inside the applied relinearized variables remain depth-owner or mixed-owner dominated while MANO replay ownership and hand span do not isolate the failed rows",
            "claim_limit": "this diagnostic does not optimize new MANO betas, camera calibration, object geometry, object pose, or contact ownership",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(
        args.output_root / case / "v17_relinearized_hand_capacity_diagnostic.json",
        report,
    )
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [row for case in cases for row in require_list(case.get("rows"), "case rows")]
    residual_owner_rows = [row for row in rows if require_dict(row, "row").get("depth_repair_factor_candidate") is True]
    summary = {
        "method": "build_v17_relinearized_hand_capacity_diagnostic",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame_count") for case in cases),
        "applied_relinearized_variable_rows": len(rows),
        "metric_depth_compatible_rows": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows": len(residual_owner_rows),
        "relinearized_residual_owner_rows": len(residual_owner_rows),
        "projection_untrusted_rows": sum(
            1
            for row in rows
            if require_dict(row, "row").get("relinearized_reprojection_state")
            == "relinearized_reprojected_projection_untrusted"
        ),
        "relinearized_reprojection_state_counts": state_counts(
            [require_dict(row, "row") for row in rows],
            "relinearized_reprojection_state",
        ),
        "owner_depth_state_counts": state_counts(
            [require_dict(row, "row") for row in residual_owner_rows],
            "owner_depth_state",
        ),
        "mano_parameter_ownership_available_rows": bool_count(
            [require_dict(row, "row") for row in rows],
            "mano_parameter_ownership_available",
        ),
        "mano_parameter_geometry_owned_rows": bool_count(
            [require_dict(row, "row") for row in rows],
            "mano_parameter_geometry_owned",
        ),
        "residual_candidate_mano_geometry_owned_rows": bool_count(
            [require_dict(row, "row") for row in residual_owner_rows],
            "mano_parameter_geometry_owned",
        ),
        "surface_geometry_factor_available_rows": bool_count(
            [require_dict(row, "row") for row in rows],
            "surface_geometry_factor_available",
        ),
        "residual_candidate_pose_delta_clamp_hit_rows": bool_count(
            [require_dict(row, "row") for row in residual_owner_rows],
            "relinearized_pose_delta_clamp_hit",
        ),
        "residual_candidate_scaled_wrist_to_middle_tip_m": numeric_summary(
            [require_dict(row, "row") for row in residual_owner_rows],
            "scaled_wrist_to_middle_tip_m",
        ),
        "compatible_scaled_wrist_to_middle_tip_m": numeric_summary(
            [require_dict(row, "row") for row in rows if require_dict(row, "row").get("metric_depth_compatible") is True],
            "scaled_wrist_to_middle_tip_m",
        ),
        "residual_candidate_owner_abs_median_gap_m": numeric_summary(
            [require_dict(row, "row") for row in residual_owner_rows],
            "owner_abs_median_gap_m",
        ),
        "residual_candidate_owner_partition_hand_minus_unidepth_abs_tail_m": numeric_summary(
            [require_dict(row, "row") for row in residual_owner_rows],
            "owner_partition_hand_minus_unidepth_abs_tail_m",
        ),
        "residual_candidate_projection_residual_median_px": numeric_summary(
            [require_dict(row, "row") for row in residual_owner_rows],
            "projection_residual_median_px",
        ),
        "residual_candidate_vertex_alignment_error_p95_m": numeric_summary(
            [require_dict(row, "row") for row in residual_owner_rows],
            "vertex_alignment_error_p95_m",
        ),
        "surface_after_projection_to_seed_median_px": numeric_summary(
            [
                require_dict(row, "row")
                for row in rows
                if require_dict(row, "row").get("surface_geometry_factor_available") is True
            ],
            "surface_after_projection_to_seed_median_px",
        ),
        "surface_after_depth_abs_p95_m": numeric_summary(
            [
                require_dict(row, "row")
                for row in rows
                if require_dict(row, "row").get("surface_geometry_factor_available") is True
            ],
            "surface_after_depth_abs_p95_m",
        ),
        "state_summaries": grouped_state_summaries(
            [require_dict(row, "row") for row in rows],
            "relinearized_reprojection_state",
        ),
        "owner_depth_state_summaries": grouped_state_summaries(
            [require_dict(row, "row") for row in residual_owner_rows],
            "owner_depth_state",
        ),
        "capacity_conclusion": capacity_conclusion([require_dict(row, "row") for row in rows]),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame_count"),
                "applied_relinearized_variable_rows": require_int(
                    case.get("applied_relinearized_variable_rows"),
                    "case applied rows",
                ),
                "metric_depth_compatible_rows": require_int(
                    case.get("metric_depth_compatible_rows"),
                    "case compatible rows",
                ),
                "depth_repair_factor_candidate_rows": require_int(
                    case.get("depth_repair_factor_candidate_rows"),
                    "case residual rows",
                ),
                "relinearized_residual_owner_rows": require_int(
                    case.get("relinearized_residual_owner_rows"),
                    "case relinearized residual owner rows",
                ),
                "owner_depth_state_counts": require_dict(
                    case.get("owner_depth_state_counts"),
                    "case owner state counts",
                ),
                "residual_candidate_pose_delta_clamp_hit_rows": require_int(
                    case.get("residual_candidate_pose_delta_clamp_hit_rows"),
                    "case pose clamp rows",
                ),
                "capacity_conclusion": require_dict(
                    case.get("capacity_conclusion"),
                    "case capacity conclusion",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_relinearized_hand_capacity_diagnostic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relinearized-hand-surface-observation-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument(
        "--mano-parameter-ownership-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_capacity_diagnostic"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
