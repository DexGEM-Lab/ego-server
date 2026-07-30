#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
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
from build_v17_hand_temporal_reprojection_residual_owner_state import owner_valid_mask
from solve_v17_hand_temporal_owner_weighted_refit import assignment_pairs, public_assignment


STATUS = "v17_post_temporal_depth_observation_state_qc"
CLAIM = (
    "This artifact measures the depth-observation rows left after owner-weighted temporal hand-depth refit. "
    "It classifies the residual sample distribution and nearby compatible-depth search for rows that are "
    "projection-trusted but still lack enough local surface ownership. It is not an optimizer and does not "
    "update annotations."
)

DEPTH_STATE = "owner_weighted_reprojected_depth_observation_owner"


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


def owner_partition_summary(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    near = cast(np.ndarray, samples["near"]).astype(bool)
    far = cast(np.ndarray, samples["far"]).astype(bool)
    object_distance = cast(np.ndarray, samples["object_distance_px"]).astype(np.float64)
    gap = hand_z - metric_z
    valid = owner_valid_mask(row, samples, args)
    selected_valid = selected & valid
    selected_i = np.flatnonzero(selected_valid)
    owner_i = np.flatnonzero(valid)
    positive = selected_valid & (gap > float(args.max_median_abs_depth_gap_m))
    negative = selected_valid & (gap < -float(args.max_median_abs_depth_gap_m))
    abs_tail = selected_valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))
    direct_compatible = selected_valid & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    near_selected = selected_valid & near
    far_selected = selected_valid & far
    return {
        "owner_valid_sample_count": int(owner_i.size),
        "selected_residual_sample_count": int(selected_i.size),
        "near_active_object_residual_sample_count": int(np.count_nonzero(near_selected)),
        "far_from_active_object_residual_sample_count": int(np.count_nonzero(far_selected)),
        "direct_compatible_residual_sample_count": int(np.count_nonzero(direct_compatible)),
        "positive_hand_behind_depth_sample_count": int(np.count_nonzero(positive)),
        "negative_hand_in_front_depth_sample_count": int(np.count_nonzero(negative)),
        "abs_depth_tail_sample_count": int(np.count_nonzero(abs_tail)),
        "owner_gap_m": summarize(gap[owner_i].astype(float).tolist()),
        "selected_residual_gap_m": summarize(gap[selected_i].astype(float).tolist()),
        "selected_residual_abs_gap_m": summarize(np.abs(gap[selected_i]).astype(float).tolist()),
        "near_active_object_residual_gap_m": summarize(gap[near_selected].astype(float).tolist()),
        "far_from_active_object_residual_gap_m": summarize(gap[far_selected].astype(float).tolist()),
        "selected_object_distance_px": summarize(
            object_distance[selected_valid & np.isfinite(object_distance)].astype(float).tolist()
        ),
    }


def depth_observation_state(
    assignment: dict[str, Any],
    partition: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    sign = residual_sign_state(partition)
    local = local_assignment_state(assignment, args)
    if local == "no_compatible_seed":
        return f"post_temporal_depth_observation_no_compatible_seed_{sign}"
    if local == "zero_local_assignment":
        return f"post_temporal_depth_observation_zero_local_assignment_{sign}"
    if local == "sparse_local_assignment":
        return f"post_temporal_depth_observation_sparse_local_assignment_{sign}"
    if local == "weak_local_assignment":
        return f"post_temporal_depth_observation_weak_local_assignment_{sign}"
    return f"post_temporal_depth_observation_unexpected_local_assignment_{sign}"


def local_assignment_state(assignment: dict[str, Any], args: argparse.Namespace) -> str:
    compatible_seed_count = require_int(
        assignment.get("compatible_seed_sample_count"),
        "compatible seed sample count",
    )
    if compatible_seed_count == 0:
        return "no_compatible_seed"
    assigned = require_int(
        assignment.get("assigned_residual_sample_count"),
        "assigned residual sample count",
    )
    if assigned == 0:
        return "zero_local_assignment"
    fraction = optional_float(assignment.get("nearby_compatible_assignment_fraction"))
    if fraction is None:
        return "unobserved_local_assignment"
    if fraction < float(args.min_sparse_assignment_fraction):
        return "sparse_local_assignment"
    if fraction < float(args.min_mixed_projection_depth_fraction):
        return "weak_local_assignment"
    return "local_assignment_reaches_mixed_threshold"


def residual_sign_state(partition: dict[str, Any]) -> str:
    residual_count = require_int(partition.get("selected_residual_sample_count"), "selected residual count")
    if residual_count == 0:
        return "unobserved"
    positive = require_int(
        partition.get("positive_hand_behind_depth_sample_count"),
        "positive hand-behind samples",
    )
    negative = require_int(
        partition.get("negative_hand_in_front_depth_sample_count"),
        "negative hand-in-front samples",
    )
    positive_fraction = positive / residual_count
    negative_fraction = negative / residual_count
    if positive_fraction >= 0.75:
        return "hand_behind_tail"
    if negative_fraction >= 0.75:
        return "hand_in_front_tail"
    return "mixed_sign_tail"


def sample_owner_state(row: dict[str, Any], partition: dict[str, Any]) -> str:
    owner_depth_state = require_str(row.get("owner_depth_state"), "owner depth state")
    if owner_depth_state != "depth_tail_incompatible":
        return owner_depth_state
    abs_gap = require_dict(partition.get("selected_residual_abs_gap_m"), "selected abs gap")
    median = optional_float(abs_gap.get("median"))
    p95 = optional_float(abs_gap.get("p95"))
    if median is not None and p95 is not None and median <= 0.03 and p95 > 0.08:
        return "depth_tail_incompatible_median_compatible_p95_tail"
    return owner_depth_state


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_temporal_owner_weighted_refit": existing_path(
            args.hand_temporal_owner_weighted_refit_root
            / case
            / "v17_hand_temporal_owner_weighted_refit.json",
            f"{case} hand temporal owner-weighted refit",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    refit = payloads["hand_temporal_owner_weighted_refit"]
    rows: list[dict[str, Any]] = []
    for raw in require_list(refit.get("rows"), f"{case} owner-weighted rows"):
        source = require_dict(raw, "owner-weighted row")
        if source.get("owner_weighted_reprojection_state") != DEPTH_STATE:
            continue
        graph_id = require_str(source.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        samples = row_samples(source)
        selected = selected_residual(source, samples, args)
        assignment = assignment_pairs(source, samples, selected, args)
        partition = owner_partition_summary(source, samples, selected, args)
        state = depth_observation_state(assignment, partition, args)
        local_state = local_assignment_state(assignment, args)
        sign_state = residual_sign_state(partition)
        rows.append(
            {
                "case": case,
                "post_temporal_depth_observation_state_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "post_temporal_depth_observation:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_temporal_owner_weighted_refit_variable_id": source.get(
                    "hand_temporal_owner_weighted_refit_variable_id"
                ),
                "frame_idx": require_int(source.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(source.get("hand_side"), "hand_side"),
                "hand_index": require_int(source.get("hand_index"), "hand_index"),
                "source_owner_weighted_reprojection_state": source.get(
                    "owner_weighted_reprojection_state"
                ),
                "owner_sample_partition": source.get("owner_sample_partition"),
                "owner_depth_state": source.get("owner_depth_state"),
                "sample_owner_state": sample_owner_state(source, partition),
                "post_temporal_depth_observation_state": state,
                "local_assignment_state": local_state,
                "residual_sign_state": sign_state,
                "post_temporal_depth_observation_candidate": True,
                "owner_weighted_total_hand_ray_shift_m": source.get(
                    "owner_weighted_total_hand_ray_shift_m"
                ),
                "projection_residual_to_measurement_px": source.get(
                    "projection_residual_to_measurement_px"
                ),
                "partition_summary": partition,
                "local_compatible_depth_assignment": public_assignment(assignment),
                **FALSE_READY,
            }
        )
    report = {
        "method": "build_v17_post_temporal_depth_observation_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(refit.get("frame_count"), f"{case} refit frame_count"),
        "post_temporal_depth_observation_variable_count": len(rows),
        "post_temporal_depth_observation_candidate_rows": len(rows),
        "post_temporal_depth_observation_state_counts": state_counts(
            rows,
            "post_temporal_depth_observation_state",
        ),
        "post_temporal_depth_observation_owner_partition_counts": state_counts(
            rows,
            "owner_sample_partition",
        ),
        "post_temporal_depth_observation_owner_depth_state_counts": state_counts(
            rows,
            "owner_depth_state",
        ),
        "post_temporal_depth_observation_sample_owner_state_counts": state_counts(
            rows,
            "sample_owner_state",
        ),
        "post_temporal_depth_observation_local_assignment_state_counts": state_counts(
            rows,
            "local_assignment_state",
        ),
        "post_temporal_depth_observation_residual_sign_state_counts": state_counts(
            rows,
            "residual_sign_state",
        ),
        "source_owner_weighted_comparison": {
            "owner_weighted_reprojection_depth_observation_owner_rows": refit.get(
                "owner_weighted_reprojection_depth_observation_owner_rows"
            ),
            "owner_weighted_reprojection_state_counts": refit.get(
                "owner_weighted_temporal_reprojection_state_counts"
            ),
        },
        "candidate_sample_counts": {
            "selected_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition").get(
                        "selected_residual_sample_count"
                    ),
                    "selected residual count",
                )
                for row in rows
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(row.get("local_compatible_depth_assignment"), "assignment").get(
                        "compatible_seed_sample_count"
                    ),
                    "compatible seed count",
                )
                for row in rows
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("local_compatible_depth_assignment"), "assignment").get(
                        "assigned_residual_sample_count"
                    ),
                    "assigned residual count",
                )
                for row in rows
            ),
            "direct_compatible_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition").get(
                        "direct_compatible_residual_sample_count"
                    ),
                    "direct compatible count",
                )
                for row in rows
            ),
            "abs_depth_tail_sample_count": sum(
                require_int(
                    require_dict(row.get("partition_summary"), "partition").get(
                        "abs_depth_tail_sample_count"
                    ),
                    "tail count",
                )
                for row in rows
            ),
        },
        "assignment_fraction": summarize(
            [
                float(value)
                for row in rows
                for value in [
                    optional_float(
                        require_dict(
                            row.get("local_compatible_depth_assignment"),
                            "assignment",
                        ).get("nearby_compatible_assignment_fraction")
                    )
                ]
                if value is not None
            ]
        ),
        "selected_residual_abs_gap_m": summarize(
            [
                float(value)
                for row in rows
                for value in [
                    optional_float(
                        require_dict(
                            require_dict(row.get("partition_summary"), "partition").get(
                                "selected_residual_abs_gap_m"
                            ),
                            "selected abs gap",
                        ).get("median")
                    )
                ]
                if value is not None
            ]
        ),
        "problem_semantics": {
            "no_compatible_seed": (
                "the current reprojected hand mask has no depth-compatible same-hand seed samples"
            ),
            "zero_local_assignment": (
                "same-hand compatible depth exists in the mask, but no selected residual sample attaches within the local search radius"
            ),
            "sparse_local_assignment": (
                "same-hand compatible depth exists, but too few selected residual samples attach locally for mixed ownership"
            ),
            "hand_in_front_tail": (
                "selected residual samples place the current MANO surface in front of UniDepth on at least 75 percent of samples"
            ),
            "hand_behind_tail": (
                "selected residual samples place the current MANO surface behind UniDepth on at least 75 percent of samples"
            ),
            "claim_limit": (
                "this state classifies observation ownership after temporal and local-MANO tests; it does not solve the coupled hand-depth graph"
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_post_temporal_depth_observation_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_temporal_owner_weighted_refit_root / "v17_hand_temporal_owner_weighted_refit_summary.json",
        "hand temporal owner-weighted refit summary",
    )
    summary = require_dict(load_json(summary_path), "hand temporal owner-weighted refit summary")
    reports = [
        case_problem(case, args) for case in require_list(summary.get("cases"), "summary cases")
    ]
    payload = {
        "method": "build_v17_post_temporal_depth_observation_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_temporal_owner_weighted_refit_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_post_temporal_depth_observation_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "post_temporal_depth_observation_variable_count": require_int(
                    report.get("post_temporal_depth_observation_variable_count"),
                    "variable count",
                ),
                "post_temporal_depth_observation_candidate_rows": require_int(
                    report.get("post_temporal_depth_observation_candidate_rows"),
                    "candidate rows",
                ),
                "post_temporal_depth_observation_state_counts": require_dict(
                    report.get("post_temporal_depth_observation_state_counts"),
                    "state counts",
                ),
                "post_temporal_depth_observation_owner_partition_counts": require_dict(
                    report.get("post_temporal_depth_observation_owner_partition_counts"),
                    "owner partition counts",
                ),
                "post_temporal_depth_observation_sample_owner_state_counts": require_dict(
                    report.get("post_temporal_depth_observation_sample_owner_state_counts"),
                    "sample owner counts",
                ),
                "post_temporal_depth_observation_local_assignment_state_counts": require_dict(
                    report.get("post_temporal_depth_observation_local_assignment_state_counts"),
                    "local assignment counts",
                ),
                "post_temporal_depth_observation_residual_sign_state_counts": require_dict(
                    report.get("post_temporal_depth_observation_residual_sign_state_counts"),
                    "residual sign counts",
                ),
                "candidate_sample_counts": require_dict(
                    report.get("candidate_sample_counts"),
                    "candidate sample counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "post_temporal_depth_observation_variable_count": sum(
            require_int(report.get("post_temporal_depth_observation_variable_count"), "variable count")
            for report in reports
        ),
        "post_temporal_depth_observation_candidate_rows": sum(
            require_int(report.get("post_temporal_depth_observation_candidate_rows"), "candidate rows")
            for report in reports
        ),
        "post_temporal_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_depth_observation_state_counts"),
                                "state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_depth_observation_owner_partition_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_depth_observation_owner_partition_counts"),
                                "partition counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_depth_observation_sample_owner_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_depth_observation_sample_owner_state_counts"),
                                "sample owner counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_depth_observation_local_assignment_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_depth_observation_local_assignment_state_counts"),
                                "local assignment counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "post_temporal_depth_observation_residual_sign_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("post_temporal_depth_observation_residual_sign_state_counts"),
                                "residual sign counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "candidate_sample_counts": {
            "selected_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_sample_counts"), "sample counts").get(
                        "selected_residual_sample_count"
                    ),
                    "selected residual count",
                )
                for report in reports
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_sample_counts"), "sample counts").get(
                        "compatible_seed_sample_count"
                    ),
                    "compatible seed count",
                )
                for report in reports
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_sample_counts"), "sample counts").get(
                        "assigned_residual_sample_count"
                    ),
                    "assigned residual count",
                )
                for report in reports
            ),
            "direct_compatible_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_sample_counts"), "sample counts").get(
                        "direct_compatible_residual_sample_count"
                    ),
                    "direct compatible count",
                )
                for report in reports
            ),
            "abs_depth_tail_sample_count": sum(
                require_int(
                    require_dict(report.get("candidate_sample_counts"), "sample counts").get(
                        "abs_depth_tail_sample_count"
                    ),
                    "tail count",
                )
                for report in reports
            ),
        },
        "assignment_fraction": summarize(
            [
                float(value)
                for report in reports
                for value in [
                    optional_float(
                        require_dict(report.get("assignment_fraction"), "assignment fraction").get("median")
                    )
                ]
                if value is not None
            ]
        ),
        "source_owner_weighted_comparison": {
            "owner_weighted_reprojection_depth_observation_owner_rows": summary.get(
                "owner_weighted_reprojection_depth_observation_owner_rows"
            ),
            "owner_weighted_temporal_reprojection_state_counts": summary.get(
                "owner_weighted_temporal_reprojection_state_counts"
            ),
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_post_temporal_depth_observation_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_state"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--min-sparse-assignment-fraction", type=float, default=0.10)
    parser.add_argument("--min-mixed-projection-depth-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
