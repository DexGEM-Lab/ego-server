#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

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


STATUS = "v17_hand_local_projection_repair_problem_qc"
CLAIM = (
    "This artifact converts hand-depth repair residual ownership into local projection/surface repair factors. "
    "For each residual row it searches from residual owner pixels to nearby same-hand pixels whose UniDepth "
    "already matches repaired MANO depth. Rows with enough nearby compatible assignments become local "
    "projection or surface repair candidates; rows without such assignments remain depth-observation, "
    "occlusion, or projection-support owners. It is a factor problem layer; MANO articulation updates remain "
    "outside this artifact."
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


def nearest_compatible_assignment(
    repair_row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x = cast(np.ndarray, samples["x"]).astype(np.float64)
    y = cast(np.ndarray, samples["y"]).astype(np.float64)
    hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
    owner_valid = owner_valid_mask(repair_row, samples, args)
    gap = hand_z - metric_z
    all_valid = (
        np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
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


def local_projection_state(base: dict[str, Any], assignment: dict[str, Any], args: argparse.Namespace) -> str:
    if base.get("repair_residual_factor_candidate") is not True:
        return "not_repair_residual_factor_candidate"
    owner = require_str(base.get("residual_owner_state"), "residual owner state")
    if owner == "residual_unsupported_projection_owner":
        return "projection_support_unresolved"
    fraction = optional_float(assignment.get("nearby_compatible_assignment_fraction"))
    if fraction is None:
        return "local_projection_repair_unobserved"
    if fraction >= float(args.min_local_projection_candidate_fraction):
        return "local_projection_repair_factor_candidate"
    if fraction >= float(args.min_mixed_projection_depth_fraction):
        return "mixed_projection_depth_observation_owner"
    return "depth_observation_or_occlusion_owner"


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph report",
        ),
        "hand_depth_repair_residual_owner_state": existing_path(
            args.hand_depth_repair_residual_owner_state_root
            / case
            / "v17_hand_depth_repair_residual_owner_state.json",
            f"{case} hand depth repair residual-owner state report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    repair = payloads["hand_depth_repair_graph"]
    owner_report = payloads["hand_depth_repair_residual_owner_state"]
    if require_int(repair.get("frame_count"), f"{case} repair graph frame_count") != require_int(
        owner_report.get("frame_count"),
        f"{case} residual owner frame_count",
    ):
        raise RuntimeError(f"{case} repair graph frame count disagrees with residual-owner state")
    owner_rows = [
        require_dict(raw, "residual owner row")
        for raw in require_list(owner_report.get("rows"), f"{case} residual owner rows")
    ]
    owner_by_graph_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "source repair graph id"): row
        for row in owner_rows
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(repair.get("rows"), f"{case} repair graph rows"):
        repair_row = require_dict(raw, "repair graph row")
        graph_id = require_str(repair_row.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        owner_row = require_dict(owner_by_graph_id.get(graph_id), f"{case} residual owner row {graph_id}")
        base = {
            "case": case,
            "hand_local_projection_repair_variable_id": graph_id.replace(
                "hand_depth_repair_graph:",
                "hand_local_projection_repair:",
                1,
            ),
            "source_hand_depth_repair_graph_variable_id": graph_id,
            "source_hand_depth_repair_residual_owner_variable_id": require_str(
                owner_row.get("hand_depth_repair_residual_owner_variable_id"),
                "residual owner variable id",
            ),
            "frame_idx": require_int(repair_row.get("frame_idx"), "repair row frame_idx"),
            "hand_side": require_str(repair_row.get("hand_side"), "repair row hand_side"),
            "hand_index": require_int(repair_row.get("hand_index"), "repair row hand_index"),
            "repair_residual_factor_candidate": bool(
                owner_row.get("repair_residual_factor_candidate") is True
            ),
            "residual_owner_state": owner_row.get("residual_owner_state"),
            "residual_depth_observation_state": owner_row.get("depth_observation_state"),
            "residual_independent_support_state": owner_row.get("independent_support_state"),
            "owner_depth_state": repair_row.get("owner_depth_state"),
            "owner_sample_partition": repair_row.get("owner_sample_partition"),
            "metric_depth_compatible": bool(repair_row.get("metric_depth_compatible") is True),
            **FALSE_READY,
        }
        if base["repair_residual_factor_candidate"] is not True:
            rows.append(
                {
                    **base,
                    "local_projection_repair_state": "not_repair_residual_factor_candidate",
                    "local_projection_repair_factor_candidate": False,
                    "assignment": None,
                    "missing_local_projection_inputs": [],
                }
            )
            continue
        samples = row_samples(repair_row)
        selected = selected_residual(repair_row, samples, args)
        assignment = nearest_compatible_assignment(repair_row, samples, selected, args)
        state = local_projection_state(base, assignment, args)
        rows.append(
            {
                **base,
                "local_projection_repair_state": state,
                "local_projection_repair_factor_candidate": bool(
                    state == "local_projection_repair_factor_candidate"
                ),
                "partial_projection_depth_mixed_owner": bool(
                    state == "mixed_projection_depth_observation_owner"
                ),
                "depth_observation_or_occlusion_owner": bool(
                    state == "depth_observation_or_occlusion_owner"
                ),
                "projection_support_unresolved": bool(state == "projection_support_unresolved"),
                "assignment": assignment,
                "missing_local_projection_inputs": [],
            }
        )
    residual_rows = [row for row in rows if row.get("repair_residual_factor_candidate") is True]
    report = {
        "method": "build_v17_hand_local_projection_repair_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(repair.get("frame_count"), f"{case} repair graph frame_count"),
        "hand_local_projection_repair_variable_count": len(rows),
        "repair_residual_factor_candidate_rows": len(residual_rows),
        "local_projection_repair_factor_candidate_rows": bool_count(
            residual_rows,
            "local_projection_repair_factor_candidate",
        ),
        "partial_projection_depth_mixed_owner_rows": bool_count(
            residual_rows,
            "partial_projection_depth_mixed_owner",
        ),
        "depth_observation_or_occlusion_owner_rows": bool_count(
            residual_rows,
            "depth_observation_or_occlusion_owner",
        ),
        "projection_support_unresolved_rows": bool_count(residual_rows, "projection_support_unresolved"),
        "local_projection_repair_state_counts": state_counts(rows, "local_projection_repair_state"),
        "residual_local_projection_repair_state_counts": state_counts(
            residual_rows,
            "local_projection_repair_state",
        ),
        "source_residual_owner_comparison": {
            "repair_residual_factor_candidate_rows": owner_report.get("repair_residual_factor_candidate_rows"),
            "independent_supported_repair_residual_rows": owner_report.get(
                "independent_supported_repair_residual_rows"
            ),
            "independent_unsupported_repair_residual_rows": owner_report.get(
                "independent_unsupported_repair_residual_rows"
            ),
            "residual_owner_state_counts": owner_report.get("residual_owner_state_counts"),
        },
        "local_projection_assignment": {
            "residual_sample_count": sum(
                require_int(require_dict(row.get("assignment"), "assignment").get("residual_sample_count"), "samples")
                for row in residual_rows
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                    "assigned samples",
                )
                for row in residual_rows
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(row.get("assignment"), "assignment").get("compatible_seed_sample_count"),
                    "seed samples",
                )
                for row in residual_rows
            ),
            "assigned_pixel_shift_px": summarize(
                [
                    float(require_dict(row.get("assignment"), "assignment")["assigned_pixel_shift_px"]["median"])
                    for row in residual_rows
                    if require_int(
                        require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                        "assigned samples",
                    )
                    > 0
                    and optional_float(
                        require_dict(row.get("assignment"), "assignment")["assigned_pixel_shift_px"].get("median")
                    )
                    is not None
                ]
            ),
        },
        "problem_semantics": {
            "local_projection_repair_factor_candidate": "nearby same-hand compatible depth supports a local MANO surface or projection factor",
            "mixed_projection_depth_observation_owner": "nearby compatible depth supports only part of the residual row",
            "depth_observation_or_occlusion_owner": "supported hand evidence lacks nearby same-hand compatible depth",
            "projection_support_unresolved": "residual pixels lack independent hand-model support",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_local_projection_repair_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_depth_repair_residual_owner_state_root
        / "v17_hand_depth_repair_residual_owner_state_summary.json",
        "hand depth repair residual-owner summary",
    )
    summary = require_dict(load_json(summary_path), "hand depth repair residual-owner summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_local_projection_repair_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_depth_repair_residual_owner_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_local_projection_repair_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_local_projection_repair_variable_count": require_int(
                    report.get("hand_local_projection_repair_variable_count"),
                    "local projection variable count",
                ),
                "repair_residual_factor_candidate_rows": require_int(
                    report.get("repair_residual_factor_candidate_rows"),
                    "repair residual rows",
                ),
                "local_projection_repair_factor_candidate_rows": require_int(
                    report.get("local_projection_repair_factor_candidate_rows"),
                    "local projection candidate rows",
                ),
                "partial_projection_depth_mixed_owner_rows": require_int(
                    report.get("partial_projection_depth_mixed_owner_rows"),
                    "mixed projection-depth rows",
                ),
                "depth_observation_or_occlusion_owner_rows": require_int(
                    report.get("depth_observation_or_occlusion_owner_rows"),
                    "depth observation owner rows",
                ),
                "projection_support_unresolved_rows": require_int(
                    report.get("projection_support_unresolved_rows"),
                    "projection support unresolved rows",
                ),
                "residual_local_projection_repair_state_counts": require_dict(
                    report.get("residual_local_projection_repair_state_counts"),
                    "residual local projection state counts",
                ),
                "local_projection_assignment": require_dict(
                    report.get("local_projection_assignment"),
                    "local projection assignment",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_local_projection_repair_variable_count": sum(
            require_int(report.get("hand_local_projection_repair_variable_count"), "variable count")
            for report in reports
        ),
        "repair_residual_factor_candidate_rows": sum(
            require_int(report.get("repair_residual_factor_candidate_rows"), "residual rows")
            for report in reports
        ),
        "local_projection_repair_factor_candidate_rows": sum(
            require_int(report.get("local_projection_repair_factor_candidate_rows"), "projection candidate rows")
            for report in reports
        ),
        "partial_projection_depth_mixed_owner_rows": sum(
            require_int(report.get("partial_projection_depth_mixed_owner_rows"), "mixed rows")
            for report in reports
        ),
        "depth_observation_or_occlusion_owner_rows": sum(
            require_int(report.get("depth_observation_or_occlusion_owner_rows"), "depth owner rows")
            for report in reports
        ),
        "projection_support_unresolved_rows": sum(
            require_int(report.get("projection_support_unresolved_rows"), "projection unresolved rows")
            for report in reports
        ),
        "residual_local_projection_repair_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("residual_local_projection_repair_state_counts"),
                                "state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "local_projection_assignment": {
            "residual_sample_count": sum(
                require_int(
                    require_dict(report.get("local_projection_assignment"), "assignment").get(
                        "residual_sample_count"
                    ),
                    "residual samples",
                )
                for report in reports
            ),
            "assigned_residual_sample_count": sum(
                require_int(
                    require_dict(report.get("local_projection_assignment"), "assignment").get(
                        "assigned_residual_sample_count"
                    ),
                    "assigned samples",
                )
                for report in reports
            ),
            "compatible_seed_sample_count": sum(
                require_int(
                    require_dict(report.get("local_projection_assignment"), "assignment").get(
                        "compatible_seed_sample_count"
                    ),
                    "seed samples",
                )
                for report in reports
            ),
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_local_projection_repair_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-depth-repair-residual-owner-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--min-local-projection-candidate-fraction", type=float, default=0.75)
    parser.add_argument("--min-mixed-projection-depth-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
