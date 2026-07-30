#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

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


STATUS = "v17_hand_temporal_reprojection_residual_owner_state_qc"
CLAIM = (
    "This artifact classifies the residual rows after far-field temporal reprojection. "
    "It tests whether remaining post-reprojection residual pixels have nearby same-hand UniDepth samples "
    "compatible with the reprojected MANO surface. It does not update MANO pose, depth, contact, or annotations."
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


def owner_valid_mask(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    owner = require_str(row.get("owner_sample_partition"), "owner sample partition")
    hand_z = np.asarray(samples["hand_z"], dtype=np.float64)
    metric_z = np.asarray(samples["metric_z"], dtype=np.float64)
    if owner == "all_projected_hand_pixels":
        owner_mask = np.ones(len(hand_z), dtype=bool)
    elif owner == "near_active_object_masks":
        owner_mask = np.asarray(samples["near"], dtype=bool)
    elif owner == "far_from_active_object_masks":
        owner_mask = np.asarray(samples["far"], dtype=bool)
    else:
        raise RuntimeError(f"unknown owner sample partition: {owner}")
    return (
        owner_mask
        & np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )


def nearest_compatible_assignment(
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
        "local_projection_search_radius_px": float(args.local_projection_search_radius_px),
        "compatible_depth_abs_m": float(args.compatible_depth_abs_m),
        "nearby_compatible_assignment_fraction": None,
        "assigned_residual_sample_count": 0,
        "unassigned_residual_sample_count": int(selected_i.size),
        "selected_residual_gap_m": summarize(selected_gap.astype(float).tolist()),
        "assigned_pixel_shift_px": summarize([]),
        "assigned_source_residual_abs_gap_m": summarize([]),
        "assigned_target_seed_abs_gap_m": summarize([]),
        "assigned_hand_depth_delta_to_seed_m": summarize([]),
        "assigned_metric_depth_delta_to_seed_m": summarize([]),
    }
    if selected_i.size == 0:
        return out
    if seed_i.size == 0:
        out["nearby_compatible_assignment_fraction"] = 0.0
        return out
    selected_xy = np.stack([x[selected_i], y[selected_i]], axis=1)
    seed_xy = np.stack([x[seed_i], y[seed_i]], axis=1)
    tree = cKDTree(seed_xy)
    dist, nearest = tree.query(selected_xy, k=1)
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
    return out


def temporal_owner_state(row: dict[str, Any], assignment: dict[str, Any], args: argparse.Namespace) -> str:
    if row.get("temporal_refit_delta_applied") is not True:
        return "temporal_delta_not_applied"
    if row.get("metric_depth_compatible") is True:
        return "temporal_reprojection_metric_depth_compatible"
    projection = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    if projection.get("residual_ok") is not True:
        return "temporal_reprojection_projection_untrusted"
    fraction = optional_float(assignment.get("nearby_compatible_assignment_fraction"))
    if fraction is None:
        return "temporal_reprojection_residual_unobserved"
    if fraction >= float(args.min_local_projection_candidate_fraction):
        return "temporal_reprojection_local_surface_factor_candidate"
    if fraction >= float(args.min_mixed_projection_depth_fraction):
        return "temporal_reprojection_mixed_surface_depth_owner"
    return "temporal_reprojection_depth_observation_owner"


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_far_field_temporal_reprojection": existing_path(
            args.hand_far_field_temporal_reprojection_root
            / case
            / "v17_hand_far_field_temporal_reprojection.json",
            f"{case} hand far-field temporal reprojection report",
        ),
        "hand_far_field_temporal_refit": existing_path(
            args.hand_far_field_temporal_refit_root
            / case
            / "v17_hand_far_field_temporal_refit.json",
            f"{case} hand far-field temporal refit report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    reprojection = payloads["hand_far_field_temporal_reprojection"]
    refit = payloads["hand_far_field_temporal_refit"]
    if require_int(reprojection.get("frame_count"), f"{case} reprojection frame_count") != require_int(
        refit.get("frame_count"),
        f"{case} temporal refit frame_count",
    ):
        raise RuntimeError(f"{case} temporal reprojection frame count disagrees with temporal refit")
    if require_int(reprojection.get("temporal_refit_source_rows"), f"{case} reprojection source rows") != require_int(
        refit.get("far_field_temporal_refit_row_count"),
        f"{case} temporal refit row count",
    ):
        raise RuntimeError(f"{case} temporal reprojection source rows disagree with temporal refit")
    rows: list[dict[str, Any]] = []
    for raw in require_list(reprojection.get("rows"), f"{case} temporal reprojection rows"):
        source = require_dict(raw, "temporal reprojection row")
        if source.get("source_temporal_refit_state") is None:
            continue
        samples = row_samples(source)
        selected = selected_residual(source, samples, args)
        assignment = nearest_compatible_assignment(source, samples, selected, args)
        state = temporal_owner_state(source, assignment, args)
        rows.append(
            {
                "case": case,
                "hand_temporal_reprojection_residual_owner_variable_id": require_str(
                    source.get("hand_depth_repair_graph_variable_id"),
                    "repair graph variable id",
                ).replace(
                    "hand_depth_repair_graph:",
                    "hand_temporal_reprojection_residual_owner:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": require_str(
                    source.get("hand_depth_repair_graph_variable_id"),
                    "repair graph variable id",
                ),
                "source_temporal_refit_variable_id": source.get("source_temporal_refit_variable_id"),
                "frame_idx": require_int(source.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(source.get("hand_side"), "hand_side"),
                "hand_index": require_int(source.get("hand_index"), "hand_index"),
                "source_temporal_refit_state": source.get("source_temporal_refit_state"),
                "temporal_reprojection_state": source.get("temporal_reprojection_state"),
                "owner_sample_partition": source.get("owner_sample_partition"),
                "owner_depth_state": source.get("owner_depth_state"),
                "metric_depth_compatible": bool(source.get("metric_depth_compatible") is True),
                "temporal_refit_delta_applied": bool(source.get("temporal_refit_delta_applied") is True),
                "temporal_refit_reprojected_depth_improved": bool(
                    source.get("temporal_refit_reprojected_depth_improved") is True
                ),
                "temporal_reprojection_residual_owner_state": state,
                "temporal_reprojection_local_surface_factor_candidate": bool(
                    state == "temporal_reprojection_local_surface_factor_candidate"
                ),
                "temporal_reprojection_mixed_surface_depth_owner": bool(
                    state == "temporal_reprojection_mixed_surface_depth_owner"
                ),
                "temporal_reprojection_depth_observation_owner": bool(
                    state == "temporal_reprojection_depth_observation_owner"
                ),
                "temporal_reprojection_projection_untrusted": bool(
                    state == "temporal_reprojection_projection_untrusted"
                ),
                "assignment": assignment,
                **FALSE_READY,
            }
        )
    applied_rows = [row for row in rows if row["temporal_refit_delta_applied"] is True]
    residual_rows = [
        row
        for row in applied_rows
        if row["temporal_reprojection_residual_owner_state"]
        not in {
            "temporal_reprojection_metric_depth_compatible",
            "temporal_reprojection_projection_untrusted",
        }
    ]
    report = {
        "method": "build_v17_hand_temporal_reprojection_residual_owner_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(reprojection.get("frame_count"), f"{case} frame_count"),
        "temporal_reprojection_source_rows": len(rows),
        "temporal_reprojection_delta_applied_rows": len(applied_rows),
        "temporal_reprojection_residual_owner_rows": len(residual_rows),
        "temporal_reprojection_local_surface_factor_candidate_rows": bool_count(
            residual_rows,
            "temporal_reprojection_local_surface_factor_candidate",
        ),
        "temporal_reprojection_mixed_surface_depth_owner_rows": bool_count(
            residual_rows,
            "temporal_reprojection_mixed_surface_depth_owner",
        ),
        "temporal_reprojection_depth_observation_owner_rows": bool_count(
            residual_rows,
            "temporal_reprojection_depth_observation_owner",
        ),
        "temporal_reprojection_projection_untrusted_rows": bool_count(
            applied_rows,
            "temporal_reprojection_projection_untrusted",
        ),
        "temporal_reprojection_residual_owner_state_counts": state_counts(
            rows,
            "temporal_reprojection_residual_owner_state",
        ),
        "applied_temporal_reprojection_residual_owner_state_counts": state_counts(
            applied_rows,
            "temporal_reprojection_residual_owner_state",
        ),
        "source_temporal_reprojection_state_counts": reprojection.get("temporal_refit_reprojection_state_counts"),
        "local_assignment": {
            "residual_sample_count": sum(
                require_int(require_dict(row["assignment"], "assignment").get("residual_sample_count"), "samples")
                for row in residual_rows
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(row["assignment"], "assignment").get("assigned_residual_sample_count"),
                    "assigned samples",
                )
                for row in residual_rows
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(row["assignment"], "assignment").get("compatible_seed_sample_count"),
                    "seed samples",
                )
                for row in residual_rows
            ),
            "assigned_pixel_shift_px": summarize(
                [
                    float(require_dict(row["assignment"], "assignment")["assigned_pixel_shift_px"]["median"])
                    for row in residual_rows
                    if require_int(
                        require_dict(row["assignment"], "assignment").get("assigned_residual_sample_count"),
                        "assigned samples",
                    )
                    > 0
                    and optional_float(
                        require_dict(row["assignment"], "assignment")["assigned_pixel_shift_px"].get("median")
                    )
                    is not None
                ]
            ),
        },
        "problem_semantics": {
            "temporal_reprojection_local_surface_factor_candidate": "post-temporal residual pixels have nearby same-hand compatible depth, so local MANO surface or projection factors can own the row",
            "temporal_reprojection_mixed_surface_depth_owner": "nearby compatible depth explains only part of the post-temporal residual row",
            "temporal_reprojection_depth_observation_owner": "post-temporal residual pixels lack nearby compatible depth after reprojecting MANO",
            "claim_limit": "this artifact classifies residual ownership; it is not an optimizer and it does not update annotation state",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_temporal_reprojection_residual_owner_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_far_field_temporal_reprojection_root / "v17_hand_far_field_temporal_reprojection_summary.json",
        "hand far-field temporal reprojection summary",
    )
    summary = require_dict(load_json(summary_path), "hand far-field temporal reprojection summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_temporal_reprojection_residual_owner_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_far_field_temporal_reprojection_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_temporal_reprojection_residual_owner_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "temporal_reprojection_source_rows": require_int(
                    report.get("temporal_reprojection_source_rows"),
                    "source rows",
                ),
                "temporal_reprojection_delta_applied_rows": require_int(
                    report.get("temporal_reprojection_delta_applied_rows"),
                    "applied rows",
                ),
                "temporal_reprojection_residual_owner_rows": require_int(
                    report.get("temporal_reprojection_residual_owner_rows"),
                    "residual owner rows",
                ),
                "temporal_reprojection_local_surface_factor_candidate_rows": require_int(
                    report.get("temporal_reprojection_local_surface_factor_candidate_rows"),
                    "local rows",
                ),
                "temporal_reprojection_mixed_surface_depth_owner_rows": require_int(
                    report.get("temporal_reprojection_mixed_surface_depth_owner_rows"),
                    "mixed rows",
                ),
                "temporal_reprojection_depth_observation_owner_rows": require_int(
                    report.get("temporal_reprojection_depth_observation_owner_rows"),
                    "depth rows",
                ),
                "temporal_reprojection_projection_untrusted_rows": require_int(
                    report.get("temporal_reprojection_projection_untrusted_rows"),
                    "projection rows",
                ),
                "applied_temporal_reprojection_residual_owner_state_counts": require_dict(
                    report.get("applied_temporal_reprojection_residual_owner_state_counts"),
                    "applied state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "temporal_reprojection_source_rows": sum(
            require_int(report.get("temporal_reprojection_source_rows"), "source rows")
            for report in reports
        ),
        "temporal_reprojection_delta_applied_rows": sum(
            require_int(report.get("temporal_reprojection_delta_applied_rows"), "applied rows")
            for report in reports
        ),
        "temporal_reprojection_residual_owner_rows": sum(
            require_int(report.get("temporal_reprojection_residual_owner_rows"), "residual rows")
            for report in reports
        ),
        "temporal_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(report.get("temporal_reprojection_local_surface_factor_candidate_rows"), "local rows")
            for report in reports
        ),
        "temporal_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(report.get("temporal_reprojection_mixed_surface_depth_owner_rows"), "mixed rows")
            for report in reports
        ),
        "temporal_reprojection_depth_observation_owner_rows": sum(
            require_int(report.get("temporal_reprojection_depth_observation_owner_rows"), "depth rows")
            for report in reports
        ),
        "temporal_reprojection_projection_untrusted_rows": sum(
            require_int(report.get("temporal_reprojection_projection_untrusted_rows"), "projection rows")
            for report in reports
        ),
        "applied_temporal_reprojection_residual_owner_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("applied_temporal_reprojection_residual_owner_state_counts"),
                                "state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_temporal_reprojection_residual_owner_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-far-field-temporal-reprojection-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection"),
    )
    parser.add_argument(
        "--hand-far-field-temporal-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_reprojection_residual_owner_state"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--min-local-projection-candidate-fraction", type=float, default=0.75)
    parser.add_argument("--min-mixed-projection-depth-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
