#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

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
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary


STATUS = "v17_hand_depth_observation_switch_problem_qc"
CLAIM = (
    "This artifact materializes the depth-observation side of the V17 hand residual switches. "
    "It reuses the hand-depth repair graph sample partitions to separate residual rows whose depth "
    "contradiction is near active object masks from rows whose contradiction is far from active object masks. "
    "It does not decide a corrected hand state; it exposes which rows require depth-observation, occlusion, "
    "or broader hand-depth state variables in the complete joint solver."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def value_count(rows: list[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def sample_partition_summary(
    repair_row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    gap = hand_z - metric_z
    valid = owner_valid_mask(repair_row, samples, args)
    near = cast(np.ndarray, samples["near"]).astype(bool)
    far = cast(np.ndarray, samples["far"]).astype(bool)
    object_distance = cast(np.ndarray, samples["object_distance_px"]).astype(np.float64)
    selected_valid = selected & valid
    near_selected = selected_valid & near
    far_selected = selected_valid & far
    unknown_selected = selected_valid & ~(near | far)
    positive = selected_valid & (gap > float(args.max_median_abs_depth_gap_m))
    negative = selected_valid & (gap < -float(args.max_median_abs_depth_gap_m))
    abs_tail = selected_valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))
    return {
        "selected_residual_sample_count": int(np.count_nonzero(selected_valid)),
        "near_active_object_residual_sample_count": int(np.count_nonzero(near_selected)),
        "far_from_active_object_residual_sample_count": int(np.count_nonzero(far_selected)),
        "unknown_object_distance_residual_sample_count": int(np.count_nonzero(unknown_selected)),
        "positive_hand_behind_depth_sample_count": int(np.count_nonzero(positive)),
        "negative_hand_in_front_depth_sample_count": int(np.count_nonzero(negative)),
        "abs_depth_tail_sample_count": int(np.count_nonzero(abs_tail)),
        "selected_residual_gap_m": summarize(gap[selected_valid].astype(float).tolist()),
        "near_active_object_residual_gap_m": summarize(gap[near_selected].astype(float).tolist()),
        "far_from_active_object_residual_gap_m": summarize(gap[far_selected].astype(float).tolist()),
        "near_active_object_distance_px": summarize(
            object_distance[near_selected & np.isfinite(object_distance)].astype(float).tolist()
        ),
        "selected_object_distance_px": summarize(
            object_distance[selected_valid & np.isfinite(object_distance)].astype(float).tolist()
        ),
    }


def depth_observation_switch_state(
    *,
    switch_row: dict[str, Any],
    partition_summary: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    switch_state = require_str(switch_row.get("residual_switch_state"), "residual switch state")
    if switch_state == "projection_support_switch_required":
        return "projection_support_owner_not_depth_observation"
    if switch_state.startswith("local_surface_articulation"):
        return "local_surface_owner_not_depth_observation"
    selected = require_int(
        partition_summary.get("selected_residual_sample_count"),
        "selected residual sample count",
    )
    if selected == 0:
        return "depth_observation_switch_unobserved"
    near = require_int(
        partition_summary.get("near_active_object_residual_sample_count"),
        "near object sample count",
    )
    far = require_int(
        partition_summary.get("far_from_active_object_residual_sample_count"),
        "far object sample count",
    )
    near_fraction = near / selected
    far_fraction = far / selected
    if near_fraction >= float(args.min_near_object_fraction):
        return "object_or_occluder_depth_observation_switch"
    if far_fraction >= float(args.min_far_object_fraction):
        return "far_field_hand_depth_observation_switch"
    return "mixed_object_and_far_field_depth_observation_switch"


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
        "hand_residual_switch_problem": existing_path(
            args.hand_residual_switch_problem_root / case / "v17_hand_residual_switch_problem.json",
            f"{case} hand residual switch problem",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    repair = payloads["hand_depth_repair_graph"]
    switch_report = payloads["hand_residual_switch_problem"]
    frame_count = require_int(repair.get("frame_count"), f"{case} repair graph frame_count")
    if frame_count != require_int(switch_report.get("frame_count"), f"{case} switch frame_count"):
        raise RuntimeError(f"{case} repair graph and residual switch frame counts disagree")
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(switch_report.get("rows"), f"{case} switch rows"):
        switch_row = require_dict(raw, "switch row")
        graph_id = require_str(
            switch_row.get("source_hand_depth_repair_graph_variable_id"),
            "source graph id",
        )
        repair_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
        samples = row_samples(repair_row)
        selected = selected_residual(repair_row, samples, args)
        partition_summary = sample_partition_summary(repair_row, samples, selected, args)
        state = depth_observation_switch_state(
            switch_row=switch_row,
            partition_summary=partition_summary,
            args=args,
        )
        rows.append(
            {
                "case": case,
                "hand_depth_observation_switch_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "hand_depth_observation_switch:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_residual_switch_variable_id": require_str(
                    switch_row.get("hand_residual_switch_variable_id"),
                    "switch variable id",
                ),
                "frame_idx": require_int(switch_row.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(switch_row.get("hand_side"), "hand_side"),
                "hand_index": require_int(switch_row.get("hand_index"), "hand_index"),
                "owner_sample_partition": repair_row.get("owner_sample_partition"),
                "owner_depth_state": repair_row.get("owner_depth_state"),
                "residual_switch_state": switch_row.get("residual_switch_state"),
                "residual_depth_observation_state": switch_row.get("residual_depth_observation_state"),
                "depth_observation_switch_state": state,
                "depth_observation_switch_candidate": bool(
                    state
                    in {
                        "object_or_occluder_depth_observation_switch",
                        "far_field_hand_depth_observation_switch",
                        "mixed_object_and_far_field_depth_observation_switch",
                    }
                ),
                "partition_summary": partition_summary,
                **FALSE_READY,
            }
        )
    depth_rows = [row for row in rows if row.get("depth_observation_switch_candidate") is True]
    report = {
        "method": "build_v17_hand_depth_observation_switch_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_depth_observation_switch_variable_count": len(rows),
        "depth_observation_switch_candidate_rows": len(depth_rows),
        "object_or_occluder_depth_observation_switch_rows": value_count(
            rows,
            "depth_observation_switch_state",
            "object_or_occluder_depth_observation_switch",
        ),
        "far_field_hand_depth_observation_switch_rows": value_count(
            rows,
            "depth_observation_switch_state",
            "far_field_hand_depth_observation_switch",
        ),
        "mixed_object_and_far_field_depth_observation_switch_rows": value_count(
            rows,
            "depth_observation_switch_state",
            "mixed_object_and_far_field_depth_observation_switch",
        ),
        "depth_observation_switch_state_counts": state_counts(rows, "depth_observation_switch_state"),
        "depth_observation_candidate_state_counts": state_counts(depth_rows, "depth_observation_switch_state"),
        "candidate_partition_sample_counts": {
            "selected_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition summary").get(
                        "selected_residual_sample_count"
                    ),
                    "selected residual sample count",
                )
                for row in depth_rows
            ),
            "near_active_object_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition summary").get(
                        "near_active_object_residual_sample_count"
                    ),
                    "near residual sample count",
                )
                for row in depth_rows
            ),
            "far_from_active_object_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition summary").get(
                        "far_from_active_object_residual_sample_count"
                    ),
                    "far residual sample count",
                )
                for row in depth_rows
            ),
        },
        "source_problem_comparison": {
            "hand_residual_switch_variable_count": switch_report.get("hand_residual_switch_variable_count"),
            "mixed_projection_depth_switch_rows": switch_report.get("mixed_projection_depth_switch_rows"),
            "depth_observation_or_occlusion_switch_rows": switch_report.get(
                "depth_observation_or_occlusion_switch_rows"
            ),
            "projection_support_switch_rows": switch_report.get("projection_support_switch_rows"),
        },
        "problem_semantics": {
            "object_or_occluder_depth_observation_switch": "most residual samples lie near active object masks, so object/occluder ownership must be decided jointly with hand depth",
            "far_field_hand_depth_observation_switch": "most residual samples are far from active object masks, so object occlusion alone cannot own the residual",
            "mixed_object_and_far_field_depth_observation_switch": "residual samples span object-near and far-field partitions",
            "projection_support_owner_not_depth_observation": "the residual lacks independent hand support and belongs to projection/support ownership first",
            "local_surface_owner_not_depth_observation": "the residual belongs to the local hand-surface/articulation owner before depth-observation switching",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_depth_observation_switch_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_residual_switch_problem_root / "v17_hand_residual_switch_problem_summary.json",
        "hand residual switch summary",
    )
    switch_summary = require_dict(load_json(summary_path), "hand residual switch summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(switch_summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_depth_observation_switch_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_residual_switch_problem_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_depth_observation_switch_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_depth_observation_switch_variable_count": require_int(
                    report.get("hand_depth_observation_switch_variable_count"),
                    "depth observation switch variable count",
                ),
                "depth_observation_switch_candidate_rows": require_int(
                    report.get("depth_observation_switch_candidate_rows"),
                    "depth observation candidate rows",
                ),
                "depth_observation_switch_state_counts": require_dict(
                    report.get("depth_observation_switch_state_counts"),
                    "switch state counts",
                ),
                "depth_observation_candidate_state_counts": require_dict(
                    report.get("depth_observation_candidate_state_counts"),
                    "candidate state counts",
                ),
                "candidate_partition_sample_counts": require_dict(
                    report.get("candidate_partition_sample_counts"),
                    "candidate partition counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_depth_observation_switch_variable_count": sum(
            require_int(
                report.get("hand_depth_observation_switch_variable_count"),
                "depth observation switch variable count",
            )
            for report in reports
        ),
        "depth_observation_switch_candidate_rows": sum(
            require_int(report.get("depth_observation_switch_candidate_rows"), "candidate rows")
            for report in reports
        ),
        "depth_observation_switch_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("depth_observation_switch_state_counts"), "states"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "depth_observation_candidate_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("depth_observation_candidate_state_counts"), "states"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "candidate_partition_sample_counts": {
            "selected_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_partition_sample_counts"), "partition counts").get(
                        "selected_residual_sample_count"
                    ),
                    "selected residual sample count",
                )
                for report in reports
            ),
            "near_active_object_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_partition_sample_counts"), "partition counts").get(
                        "near_active_object_residual_sample_count"
                    ),
                    "near residual sample count",
                )
                for report in reports
            ),
            "far_from_active_object_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_partition_sample_counts"), "partition counts").get(
                        "far_from_active_object_residual_sample_count"
                    ),
                    "far residual sample count",
                )
                for report in reports
            ),
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_depth_observation_switch_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-residual-switch-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_residual_switch_problem"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_observation_switch_problem"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--min-near-object-fraction", type=float, default=0.75)
    parser.add_argument("--min-far-object-fraction", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
