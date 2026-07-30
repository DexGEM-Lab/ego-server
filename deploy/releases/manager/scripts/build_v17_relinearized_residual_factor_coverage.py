#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    annotation_frames,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import case_support_sources, existing_path, source_summary
from build_v17_post_temporal_depth_observation_support_state import (
    SAME_SIDE_INDEPENDENT_SUPPORT_STATES,
)
from solve_v17_hand_temporal_owner_weighted_refit import assignment_pairs, public_assignment, thin
from solve_v17_relinearized_hand_surface_observation_graph import (
    OBSERVATION_FACTOR_STATES,
    support_for_selected_pixels,
)


STATUS = "v17_relinearized_residual_factor_coverage_qc"
CLAIM = (
    "This artifact tests factor coverage for every hand-depth residual row left after the "
    "relinearized hand surface-observation graph. It reuses the same owner partition, residual "
    "selection, compatible-depth assignment, and independent hand-support predicates as the "
    "current relinearized graph, but it does not optimize any variable."
)

SURFACE_STATES = {
    "full_residual_local_surface_factor_candidate",
    "full_residual_mixed_surface_depth_owner",
}
DEPTH_STATE = "full_residual_depth_observation_owner"
COMPATIBLE_STATE = "full_residual_metric_depth_compatible"
PROJECTION_UNTRUSTED_STATE = "full_residual_projection_untrusted"
UNOBSERVED_STATE = "full_residual_unobserved"


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def optional_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number or null")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number or null")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number or null")
    return out


def coverage_state(
    row: dict[str, Any],
    assignment: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if row.get("metric_depth_compatible") is True:
        return COMPATIBLE_STATE
    projection = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    if projection.get("residual_ok") is not True:
        return PROJECTION_UNTRUSTED_STATE
    fraction = assignment.get("nearby_compatible_assignment_fraction")
    if fraction is None:
        return UNOBSERVED_STATE
    value = float(fraction)
    if value >= float(args.min_local_projection_candidate_fraction):
        return "full_residual_local_surface_factor_candidate"
    if value >= float(args.min_mixed_projection_depth_fraction):
        return "full_residual_mixed_surface_depth_owner"
    return DEPTH_STATE


def factor_state(row: dict[str, Any]) -> str:
    if row.get("full_residual_surface_factor_row") is True:
        return "full_residual_surface_factor_variable"
    if row.get("full_residual_depth_observation_factor_row") is True:
        support = require_str(row.get("independent_keypoint_support_state"), "keypoint support")
        return f"full_residual_depth_observation_{support}_factor_variable"
    if row.get("full_residual_compatible_anchor_row") is True:
        return "full_residual_compatible_anchor_variable"
    state = require_str(row.get("full_residual_factor_coverage_state"), "coverage state")
    if state == DEPTH_STATE:
        support = require_str(row.get("independent_keypoint_support_state"), "keypoint support")
        return f"full_residual_depth_observation_{support}_prior_smooth_variable"
    if state == PROJECTION_UNTRUSTED_STATE:
        return "full_residual_projection_untrusted_prior_smooth_variable"
    if state == COMPATIBLE_STATE:
        return "full_residual_compatible_without_anchor_prior_smooth_variable"
    if state == UNOBSERVED_STATE:
        return "full_residual_unobserved_prior_smooth_variable"
    return "full_residual_sparse_owner_prior_smooth_variable"


def compatible_anchor_gap_count(
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> int:
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    gap = hand_z - metric_z
    valid = (
        selected
        & np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
        & (np.abs(gap) <= float(args.compatible_depth_abs_m))
    )
    return int(thin(gap[valid].astype(np.float64), int(args.max_factor_samples_per_row)).size)


def depth_observation_sample_count(
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> int:
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    target = (metric_z[selected] - hand_z[selected]).astype(np.float64)
    return int(thin(target, int(args.max_depth_observation_samples_per_row)).size)


def support_defaults() -> dict[str, Any]:
    return {
        "selected_support_state": "not_depth_observation_owner",
        "independent_support_state": "not_depth_observation_owner",
        "independent_keypoint_support_state": "same_side_independent_keypoints_unmeasured",
        "support_shape_counts": {},
        "support": None,
    }


def row_coverage(
    row: dict[str, Any],
    *,
    frame: dict[str, Any],
    support_sources: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    samples = row_samples(row)
    selected = selected_residual(row, samples, args)
    assignment = assignment_pairs(row, samples, selected, args)
    state = coverage_state(row, assignment, args)
    selected_count = int(np.count_nonzero(selected))
    support = support_defaults()
    depth_observation_samples = 0
    if state == DEPTH_STATE and selected_count > 0:
        support = support_for_selected_pixels(
            row=row,
            samples=samples,
            selected=selected,
            frame=frame,
            support_sources=support_sources,
            args=args,
        )
        box_state = require_str(support.get("independent_support_state"), "independent support")
        keypoint_state = require_str(
            support.get("independent_keypoint_support_state"),
            "independent keypoint support",
        )
        if box_state in SAME_SIDE_INDEPENDENT_SUPPORT_STATES and keypoint_state in OBSERVATION_FACTOR_STATES:
            depth_observation_samples = depth_observation_sample_count(samples, selected, args)
    assigned_count = require_int(
        assignment.get("assigned_residual_sample_count"),
        "assigned residual sample count",
    )
    compatible_seed_count = require_int(
        assignment.get("compatible_seed_sample_count"),
        "compatible seed sample count",
    )
    anchor_sample_count = compatible_anchor_gap_count(samples, selected, args) if state == COMPATIBLE_STATE else 0
    surface_factor = bool(state in SURFACE_STATES and assigned_count >= int(args.min_post_temporal_factor_samples))
    depth_observation_factor = bool(
        state == DEPTH_STATE and depth_observation_samples >= int(args.min_post_temporal_factor_samples)
    )
    anchor_factor = bool(
        state == COMPATIBLE_STATE and anchor_sample_count >= int(args.min_post_temporal_factor_samples)
    )
    direct_factor = surface_factor or depth_observation_factor or anchor_factor
    out = {
        "case": require_str(row.get("case"), "case"),
        "full_residual_factor_coverage_variable_id": require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            "hand graph id",
        ).replace("hand_depth_repair_graph:", "full_residual_factor_coverage:", 1),
        "source_hand_depth_repair_graph_variable_id": require_str(
            row.get("hand_depth_repair_graph_variable_id"),
            "hand graph id",
        ),
        "frame_idx": require_int(row.get("frame_idx"), "frame_idx"),
        "hand_side": require_str(row.get("hand_side"), "hand_side"),
        "hand_index": require_int(row.get("hand_index"), "hand_index"),
        "current_relinearized_delta_applied": bool(row.get("relinearized_delta_applied") is True),
        "current_relinearized_reprojection_state": require_str(
            row.get("relinearized_reprojection_state"),
            "current relinearized state",
        ),
        "source_temporal_refit_state": row.get("source_temporal_refit_state"),
        "source_post_temporal_observation_reprojection_state": row.get(
            "source_post_temporal_observation_reprojection_state"
        ),
        "scalar_variable_candidate": bool(row.get("relinearized_total_hand_ray_shift_m") is not None),
        "owner_sample_partition": require_str(row.get("owner_sample_partition"), "owner sample partition"),
        "owner_depth_state": require_str(row.get("owner_depth_state"), "owner depth state"),
        "owner_median_gap_m": optional_finite_number(row.get("owner_median_gap_m"), "owner median gap"),
        "selected_residual_sample_count": selected_count,
        "compatible_seed_sample_count": compatible_seed_count,
        "owner_compatible_seed_sample_count": require_int(
            assignment.get("owner_compatible_seed_sample_count"),
            "owner compatible seed count",
        ),
        "nearby_compatible_assignment_fraction": optional_finite_number(
            assignment.get("nearby_compatible_assignment_fraction"),
            "nearby compatible assignment fraction",
        ),
        "assigned_residual_sample_count": assigned_count,
        "unassigned_residual_sample_count": require_int(
            assignment.get("unassigned_residual_sample_count"),
            "unassigned residual sample count",
        ),
        "full_residual_factor_coverage_state": state,
        "full_residual_surface_factor_row": surface_factor,
        "full_residual_depth_observation_factor_row": depth_observation_factor,
        "full_residual_compatible_anchor_row": anchor_factor,
        "full_residual_direct_factor_row": direct_factor,
        "full_residual_prior_smooth_only_row": not direct_factor,
        "depth_observation_sample_count": depth_observation_samples,
        "compatible_anchor_sample_count": anchor_sample_count,
        "surface_assignment": public_assignment(assignment),
        **support,
        **FALSE_READY,
    }
    return {**out, "full_residual_factor_state": factor_state(out)}


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "relinearized_hand_surface_observation_graph": existing_path(
            args.relinearized_hand_surface_observation_graph_root
            / case
            / "v17_relinearized_hand_surface_observation_graph.json",
            f"{case} relinearized hand surface-observation graph",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    report = payloads["relinearized_hand_surface_observation_graph"]
    if frame_count != require_int(report.get("frame_count"), f"{case} relinearized graph frame_count"):
        raise RuntimeError(f"{case} frame count mismatch")
    support_sources = case_support_sources(case, args)
    rows = []
    for raw in require_list(report.get("rows"), f"{case} relinearized rows"):
        row = require_dict(raw, "relinearized row")
        if row.get("depth_repair_factor_candidate") is not True:
            continue
        frame_idx = require_int(row.get("frame_idx"), "frame_idx")
        rows.append(
            row_coverage(
                row,
                frame=require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}"),
                support_sources=support_sources,
                args=args,
            )
        )
    applied_rows = [row for row in rows if row["current_relinearized_delta_applied"] is True]
    nonapplied_rows = [row for row in rows if row["current_relinearized_delta_applied"] is not True]
    out = {
        "method": "build_v17_relinearized_residual_factor_coverage",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            **{name: source_summary(path, payloads.get(name)) for name, path in paths.items()},
            **{
                f"support_{name}": source_summary(path)
                for name, path in require_dict(support_sources["paths"], "support paths").items()
            },
        },
        "frame_count": frame_count,
        "relinearized_hand_residual_rows": len(rows),
        "current_relinearized_applied_rows": len(applied_rows),
        "current_relinearized_nonapplied_rows": len(nonapplied_rows),
        "full_residual_scalar_variable_candidate_rows": bool_count(rows, "scalar_variable_candidate"),
        "full_residual_direct_factor_rows": bool_count(rows, "full_residual_direct_factor_row"),
        "full_residual_surface_factor_rows": bool_count(rows, "full_residual_surface_factor_row"),
        "full_residual_depth_observation_factor_rows": bool_count(
            rows,
            "full_residual_depth_observation_factor_row",
        ),
        "full_residual_compatible_anchor_rows": bool_count(rows, "full_residual_compatible_anchor_row"),
        "full_residual_prior_smooth_only_rows": bool_count(rows, "full_residual_prior_smooth_only_row"),
        "nonapplied_full_residual_direct_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_direct_factor_row",
        ),
        "nonapplied_full_residual_surface_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_surface_factor_row",
        ),
        "nonapplied_full_residual_depth_observation_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_depth_observation_factor_row",
        ),
        "nonapplied_full_residual_prior_smooth_only_rows": bool_count(
            nonapplied_rows,
            "full_residual_prior_smooth_only_row",
        ),
        "full_residual_factor_coverage_state_counts": state_counts(
            rows,
            "full_residual_factor_coverage_state",
        ),
        "full_residual_factor_state_counts": state_counts(rows, "full_residual_factor_state"),
        "nonapplied_full_residual_factor_coverage_state_counts": state_counts(
            nonapplied_rows,
            "full_residual_factor_coverage_state",
        ),
        "nonapplied_full_residual_factor_state_counts": state_counts(
            nonapplied_rows,
            "full_residual_factor_state",
        ),
        "independent_keypoint_support_state_counts": state_counts(
            rows,
            "independent_keypoint_support_state",
        ),
        "selected_residual_sample_count": sum(
            require_int(row.get("selected_residual_sample_count"), "selected residual samples")
            for row in rows
        ),
        "assigned_residual_sample_count": sum(
            require_int(row.get("assigned_residual_sample_count"), "assigned residual samples")
            for row in rows
        ),
        "compatible_seed_sample_count": sum(
            require_int(row.get("compatible_seed_sample_count"), "compatible seed samples")
            for row in rows
        ),
        "problem_semantics": {
            "full_residual_direct_factor_row": (
                "the row has a surface, depth-observation, or compatible-anchor factor under the same "
                "coverage predicates used by the current relinearized hand graph"
            ),
            "current_relinearized_nonapplied_rows": (
                "residual rows that were evaluated after the current graph but were not variables in it"
            ),
            "prior_smooth_only": (
                "the row can be represented as a scalar hand-depth variable only through priors and temporal "
                "smoothness until another measurement source is added"
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_relinearized_residual_factor_coverage.json", out)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_problem(case, args) for case in args.cases]
    rows = [row for case in cases for row in require_list(case.get("rows"), "case rows")]
    nonapplied_rows = [row for row in rows if row["current_relinearized_delta_applied"] is not True]
    summary = {
        "method": "build_v17_relinearized_residual_factor_coverage",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "frame_count": sum(require_int(case.get("frame_count"), "case frame count") for case in cases),
        "relinearized_hand_residual_rows": len(rows),
        "current_relinearized_applied_rows": bool_count(rows, "current_relinearized_delta_applied"),
        "current_relinearized_nonapplied_rows": len(nonapplied_rows),
        "full_residual_scalar_variable_candidate_rows": bool_count(rows, "scalar_variable_candidate"),
        "full_residual_direct_factor_rows": bool_count(rows, "full_residual_direct_factor_row"),
        "full_residual_surface_factor_rows": bool_count(rows, "full_residual_surface_factor_row"),
        "full_residual_depth_observation_factor_rows": bool_count(
            rows,
            "full_residual_depth_observation_factor_row",
        ),
        "full_residual_compatible_anchor_rows": bool_count(rows, "full_residual_compatible_anchor_row"),
        "full_residual_prior_smooth_only_rows": bool_count(rows, "full_residual_prior_smooth_only_row"),
        "nonapplied_full_residual_direct_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_direct_factor_row",
        ),
        "nonapplied_full_residual_surface_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_surface_factor_row",
        ),
        "nonapplied_full_residual_depth_observation_factor_rows": bool_count(
            nonapplied_rows,
            "full_residual_depth_observation_factor_row",
        ),
        "nonapplied_full_residual_prior_smooth_only_rows": bool_count(
            nonapplied_rows,
            "full_residual_prior_smooth_only_row",
        ),
        "full_residual_factor_coverage_state_counts": state_counts(
            rows,
            "full_residual_factor_coverage_state",
        ),
        "full_residual_factor_state_counts": state_counts(rows, "full_residual_factor_state"),
        "nonapplied_full_residual_factor_coverage_state_counts": state_counts(
            nonapplied_rows,
            "full_residual_factor_coverage_state",
        ),
        "nonapplied_full_residual_factor_state_counts": state_counts(
            nonapplied_rows,
            "full_residual_factor_state",
        ),
        "independent_keypoint_support_state_counts": state_counts(
            rows,
            "independent_keypoint_support_state",
        ),
        "selected_residual_sample_count": sum(
            require_int(row.get("selected_residual_sample_count"), "selected residual samples")
            for row in rows
        ),
        "assigned_residual_sample_count": sum(
            require_int(row.get("assigned_residual_sample_count"), "assigned residual samples")
            for row in rows
        ),
        "compatible_seed_sample_count": sum(
            require_int(row.get("compatible_seed_sample_count"), "compatible seed samples")
            for row in rows
        ),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "frame_count": require_int(case.get("frame_count"), "case frame count"),
                "relinearized_hand_residual_rows": require_int(
                    case.get("relinearized_hand_residual_rows"),
                    "case residual rows",
                ),
                "current_relinearized_applied_rows": require_int(
                    case.get("current_relinearized_applied_rows"),
                    "case applied rows",
                ),
                "current_relinearized_nonapplied_rows": require_int(
                    case.get("current_relinearized_nonapplied_rows"),
                    "case nonapplied rows",
                ),
                "full_residual_direct_factor_rows": require_int(
                    case.get("full_residual_direct_factor_rows"),
                    "case direct factor rows",
                ),
                "full_residual_surface_factor_rows": require_int(
                    case.get("full_residual_surface_factor_rows"),
                    "case surface factor rows",
                ),
                "full_residual_depth_observation_factor_rows": require_int(
                    case.get("full_residual_depth_observation_factor_rows"),
                    "case depth observation factor rows",
                ),
                "full_residual_prior_smooth_only_rows": require_int(
                    case.get("full_residual_prior_smooth_only_rows"),
                    "case prior smooth rows",
                ),
                "full_residual_factor_coverage_state_counts": require_dict(
                    case.get("full_residual_factor_coverage_state_counts"),
                    "case coverage state counts",
                ),
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_relinearized_residual_factor_coverage_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--relinearized-hand-surface-observation-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_residual_factor_coverage"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
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
    parser.add_argument("--min-keypoint-supported-fraction", type=float, default=0.05)
    parser.add_argument("--strong-keypoint-supported-fraction", type=float, default=0.25)
    parser.add_argument("--max-assign-center-px", type=float, default=150.0)
    parser.add_argument("--near-support-bbox-px", type=float, default=20.0)
    parser.add_argument("--near-support-keypoint-px", type=float, default=20.0)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
