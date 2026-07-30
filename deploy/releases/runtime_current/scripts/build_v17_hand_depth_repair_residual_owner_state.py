#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np

from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
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
from build_v17_hand_tail_depth_observation_state import (
    depth_observation_state,
    local_depth_compatibility,
)
from build_v17_hand_tail_support_state import (
    case_support_sources,
    existing_path,
    independent_support_state,
    selected_support_state,
    source_summary,
    subset_support,
    support_shapes_for_row,
)


STATUS = "v17_hand_depth_repair_residual_owner_state_qc"
CLAIM = (
    "This artifact localizes residual hand-depth failures after the bounded V17 hand-depth repair graph. "
    "It tests whether residual pixels are supported by independent hand detections and whether nearby UniDepth "
    "contains depth compatible with the repaired MANO surface. Supported residuals with nearby compatible depth "
    "point to local MANO surface or projection repair; supported residuals lacking nearby compatible depth point "
    "to depth-observation or occlusion variables."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def owner_valid_mask(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    partitions = require_dict(row.get("partitions"), "repair graph partitions")
    owner = require_str(row.get("owner_sample_partition"), "owner sample partition")
    partition = require_dict(partitions.get(owner), f"owner partition {owner}")
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
    valid = (
        owner_mask
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    expected = require_int(partition.get("valid_depth_pixels"), f"owner partition {owner} valid depth pixels")
    actual = int(np.count_nonzero(valid))
    if actual != expected:
        raise RuntimeError(f"owner partition {owner} valid pixel count mismatch: expected {expected}, got {actual}")
    return valid & np.isfinite(hand_z) & (hand_z > 1e-6)


def selected_residual(
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    partitions = require_dict(row.get("partitions"), "repair graph partitions")
    owner = require_str(row.get("owner_sample_partition"), "owner sample partition")
    partition = require_dict(partitions.get(owner), f"owner partition {owner}")
    gap_summary = require_dict(partition.get("hand_minus_unidepth_depth_m"), "owner gap summary")
    state = require_str(row.get("owner_depth_state"), "owner depth state")
    hand_z = np.asarray(samples["hand_z"], dtype=np.float64)
    metric_z = np.asarray(samples["metric_z"], dtype=np.float64)
    gap = hand_z - metric_z
    valid = owner_valid_mask(row, samples, args)
    if state == "hand_behind_metric_depth":
        return valid & (gap > float(args.max_median_abs_depth_gap_m))
    if state == "hand_in_front_of_metric_depth":
        return valid & (gap < -float(args.max_median_abs_depth_gap_m))
    if state == "depth_tail_incompatible":
        return valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))
    median = gap_summary.get("median")
    if isinstance(median, (int, float)) and not isinstance(median, bool):
        if float(median) > 0.0:
            return valid & (gap > float(args.max_median_abs_depth_gap_m))
        return valid & (gap < -float(args.max_median_abs_depth_gap_m))
    return valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))


def residual_owner_state(row: dict[str, Any], depth_state: str) -> str:
    if row.get("repair_residual_factor_candidate") is not True:
        return "not_repair_residual_factor_candidate"
    support_state = require_str(row.get("independent_support_state"), "independent support state")
    if support_state == "tail_pixels_unsupported_by_independent_model_boxes":
        return "residual_unsupported_projection_owner"
    if depth_state == "supported_tail_has_nearby_compatible_depth":
        return "residual_supported_with_nearby_compatible_depth"
    if depth_state == "supported_tail_has_partial_nearby_compatible_depth":
        return "residual_supported_with_partial_nearby_compatible_depth"
    if depth_state == "supported_tail_lacks_nearby_compatible_depth":
        return "residual_supported_lacks_nearby_compatible_depth"
    return "residual_depth_search_unobserved"


def row_samples(row: dict[str, Any]) -> dict[str, np.ndarray | tuple[int, int] | tuple[float, float]]:
    partitions = require_dict(row.get("partitions"), "repair graph partitions")
    owner = require_str(row.get("owner_sample_partition"), "owner sample partition")
    partition = require_dict(partitions.get(owner), f"owner partition {owner}")
    hand_depth = require_dict(partition.get("hand_source_depth_m"), "owner hand depth summary")
    depth_shape_raw = row.get("depth_shape")
    projection_size_raw = row.get("projection_source_size")
    if not isinstance(depth_shape_raw, list) or len(depth_shape_raw) != 2:
        raise RuntimeError("repair row depth_shape must be a two-item list")
    if not isinstance(projection_size_raw, list) or len(projection_size_raw) != 2:
        raise RuntimeError("repair row projection_source_size must be a two-item list")
    sample_count = require_int(hand_depth.get("count"), "owner hand depth count")
    x = np.asarray(row.get("x"), dtype=np.int32)
    y = np.asarray(row.get("y"), dtype=np.int32)
    hand_z = np.asarray(row.get("hand_z"), dtype=np.float64)
    metric_z = np.asarray(row.get("metric_z"), dtype=np.float64)
    near = np.asarray(row.get("near"), dtype=bool)
    far = np.asarray(row.get("far"), dtype=bool)
    object_distance_raw = row.get("object_distance_px")
    if not isinstance(object_distance_raw, list):
        raise RuntimeError("repair row object_distance_px must be a JSON array")
    object_distance_px = np.asarray(
        [np.nan if value is None else value for value in object_distance_raw],
        dtype=np.float64,
    )
    if not (len(x) == len(y) == len(hand_z) == len(metric_z) == len(near) == len(far) == len(object_distance_px)):
        raise RuntimeError("repair row sample arrays must have equal length")
    if sample_count > len(x):
        raise RuntimeError("owner sample count exceeds stored repair row samples")
    return {
        "x": x,
        "y": y,
        "hand_z": hand_z,
        "metric_z": metric_z,
        "object_distance_px": object_distance_px,
        "near": near,
        "far": far,
        "depth_shape": (int(depth_shape_raw[0]), int(depth_shape_raw[1])),
        "projection_source_size": (float(projection_size_raw[0]), float(projection_size_raw[1])),
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    repair = payloads["hand_depth_repair_graph"]
    if len(frames) != require_int(repair.get("frame_count"), f"{case} repair graph frame_count"):
        raise RuntimeError(f"{case} annotation frame count disagrees with hand-depth repair graph")
    support_sources = case_support_sources(case, args)
    rows: list[dict[str, Any]] = []
    for raw in require_list(repair.get("rows"), f"{case} repair graph rows"):
        repair_row = require_dict(raw, "repair graph row")
        frame_idx = require_int(repair_row.get("frame_idx"), "repair row frame_idx")
        hand_i = require_int(repair_row.get("hand_index"), "repair row hand_index")
        frame = frames.get(frame_idx)
        base = {
            "case": case,
            "hand_depth_repair_residual_owner_variable_id": require_str(
                repair_row.get("hand_depth_repair_graph_variable_id"),
                "repair graph variable id",
            ).replace("hand_depth_repair_graph:", "hand_depth_repair_residual_owner:", 1),
            "source_hand_depth_repair_graph_variable_id": require_str(
                repair_row.get("hand_depth_repair_graph_variable_id"),
                "repair graph variable id",
            ),
            "frame_idx": frame_idx,
            "hand_side": require_str(repair_row.get("hand_side"), "repair row hand_side"),
            "hand_index": hand_i,
            "repair_solver_state": repair_row.get("solver_state"),
            "owner_depth_state": repair_row.get("owner_depth_state"),
            "owner_sample_partition": repair_row.get("owner_sample_partition"),
            "metric_depth_compatible": bool(repair_row.get("metric_depth_compatible") is True),
            "repair_residual_factor_candidate": bool(repair_row.get("depth_repair_factor_candidate") is True),
            "tail_factor_candidate": bool(repair_row.get("depth_repair_factor_candidate") is True),
            **FALSE_READY,
        }
        if base["repair_residual_factor_candidate"] is not True:
            rows.append(
                {
                    **base,
                    "selected_support_state": "not_repair_residual_factor_candidate",
                    "independent_support_state": "not_repair_residual_factor_candidate",
                    "depth_observation_state": "not_tail_factor_candidate",
                    "residual_owner_state": "not_repair_residual_factor_candidate",
                    "missing_owner_inputs": [],
                }
            )
            continue
        if frame is None:
            rows.append(
                {
                    **base,
                    "selected_support_state": "missing_annotation_frame",
                    "independent_support_state": "missing_annotation_frame",
                    "depth_observation_state": "missing_annotation_hand",
                    "residual_owner_state": "residual_depth_search_unobserved",
                    "missing_owner_inputs": ["annotation_frame"],
                }
            )
            continue
        samples = row_samples(repair_row)
        x = np.asarray(samples["x"], dtype=np.int32)
        y = np.asarray(samples["y"], dtype=np.int32)
        hand_z = np.asarray(samples["hand_z"], dtype=np.float64)
        metric_z = np.asarray(samples["metric_z"], dtype=np.float64)
        selected = selected_residual(repair_row, samples, args)
        if selected.shape != hand_z.shape:
            raise RuntimeError(f"{case} residual selection shape mismatch at frame {frame_idx}")
        shapes = support_shapes_for_row(
            frame=frame,
            hand_i=hand_i,
            support_sources=support_sources,
            args=args,
        )
        support = {
            "residual_samples": subset_support(
                x=x,
                y=y,
                selected=selected,
                shapes=shapes,
                projection_source_size=samples["projection_source_size"],  # type: ignore[arg-type]
                depth_shape=samples["depth_shape"],  # type: ignore[arg-type]
                args=args,
            )
        }
        selected_state = selected_support_state(
            base,
            require_dict(support["residual_samples"], "residual support"),
        )
        independent_state = independent_support_state(
            base,
            require_dict(support["residual_samples"], "residual support"),
        )
        depth_summary = local_depth_compatibility(
            x=x,
            y=y,
            hand_z=hand_z,
            metric_z=metric_z,
            selected=selected,
            depth_shape=samples["depth_shape"],  # type: ignore[arg-type]
            args=args,
        )
        depth_state = depth_observation_state({**base, "independent_support_state": independent_state, "tail_factor_candidate": True}, depth_summary)
        rows.append(
            {
                **base,
                "selected_support_state": selected_state,
                "independent_support_state": independent_state,
                "depth_observation_state": depth_state,
                "residual_owner_state": residual_owner_state(
                    {**base, "independent_support_state": independent_state},
                    depth_state,
                ),
                "residual_sample_count": int(np.count_nonzero(selected)),
                "projection_residual_to_measurement_px": repair_row.get("projection_residual_to_measurement_px"),
                "support": support,
                "local_depth_compatibility": depth_summary,
                "missing_owner_inputs": [],
            }
        )
    residual_rows = [row for row in rows if row.get("repair_residual_factor_candidate") is True]
    supported_residual_rows = [
        row
        for row in residual_rows
        if require_str(row.get("independent_support_state"), "independent support state")
        != "tail_pixels_unsupported_by_independent_model_boxes"
    ]
    report = {
        "method": "build_v17_hand_depth_repair_residual_owner_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            **{name: source_summary(path, payloads[name]) for name, path in paths.items()},
            **{
                f"support_{name}": source_summary(path)
                for name, path in require_dict(support_sources["paths"], "support paths").items()
            },
        },
        "frame_count": require_int(repair.get("frame_count"), f"{case} repair graph frame_count"),
        "hand_depth_repair_residual_owner_variable_count": len(rows),
        "repair_residual_factor_candidate_rows": len(residual_rows),
        "independent_supported_repair_residual_rows": len(supported_residual_rows),
        "independent_unsupported_repair_residual_rows": len(residual_rows) - len(supported_residual_rows),
        "selected_support_state_counts": state_counts(rows, "selected_support_state"),
        "independent_support_state_counts": state_counts(rows, "independent_support_state"),
        "residual_selected_support_state_counts": state_counts(residual_rows, "selected_support_state"),
        "residual_independent_support_state_counts": state_counts(residual_rows, "independent_support_state"),
        "depth_observation_state_counts": state_counts(rows, "depth_observation_state"),
        "residual_depth_observation_state_counts": state_counts(residual_rows, "depth_observation_state"),
        "supported_residual_depth_observation_state_counts": state_counts(supported_residual_rows, "depth_observation_state"),
        "residual_owner_state_counts": state_counts(residual_rows, "residual_owner_state"),
        "residual_sample_count": sum(require_int(row.get("residual_sample_count", 0), "residual samples") for row in residual_rows),
        "source_hand_depth_repair_graph_comparison": {
            "metric_hand_state_accepted_rows": repair.get("metric_hand_state_accepted_rows"),
            "depth_repair_factor_candidate_rows": repair.get("depth_repair_factor_candidate_rows"),
            "owner_depth_state_counts": repair.get("owner_depth_state_counts"),
        },
        "problem_semantics": {
            "residual_supported_with_nearby_compatible_depth": "local MANO surface or projection repair owner",
            "residual_supported_with_partial_nearby_compatible_depth": "mixed local surface and depth-observation owner",
            "residual_supported_lacks_nearby_compatible_depth": "depth-observation or occlusion owner",
            "residual_unsupported_projection_owner": "hand projection or 2D support owner",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_depth_repair_residual_owner_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_depth_repair_graph_root / "v17_hand_depth_repair_graph_summary.json",
        "hand depth repair graph summary",
    )
    summary = require_dict(load_json(summary_path), "hand depth repair graph summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_depth_repair_residual_owner_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_depth_repair_graph_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_depth_repair_residual_owner_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_depth_repair_residual_owner_variable_count": require_int(
                    report.get("hand_depth_repair_residual_owner_variable_count"),
                    "residual owner variable count",
                ),
                "repair_residual_factor_candidate_rows": require_int(
                    report.get("repair_residual_factor_candidate_rows"),
                    "repair residual candidate rows",
                ),
                "independent_supported_repair_residual_rows": require_int(
                    report.get("independent_supported_repair_residual_rows"),
                    "independent supported residual rows",
                ),
                "independent_unsupported_repair_residual_rows": require_int(
                    report.get("independent_unsupported_repair_residual_rows"),
                    "independent unsupported residual rows",
                ),
                "residual_independent_support_state_counts": require_dict(
                    report.get("residual_independent_support_state_counts"),
                    "residual independent support counts",
                ),
                "residual_depth_observation_state_counts": require_dict(
                    report.get("residual_depth_observation_state_counts"),
                    "residual depth observation counts",
                ),
                "supported_residual_depth_observation_state_counts": require_dict(
                    report.get("supported_residual_depth_observation_state_counts"),
                    "supported residual depth observation counts",
                ),
                "residual_owner_state_counts": require_dict(
                    report.get("residual_owner_state_counts"),
                    "residual owner state counts",
                ),
                "residual_sample_count": require_int(report.get("residual_sample_count"), "residual sample count"),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_depth_repair_residual_owner_variable_count": sum(
            require_int(report.get("hand_depth_repair_residual_owner_variable_count"), "variable count")
            for report in reports
        ),
        "repair_residual_factor_candidate_rows": sum(
            require_int(report.get("repair_residual_factor_candidate_rows"), "residual rows")
            for report in reports
        ),
        "independent_supported_repair_residual_rows": sum(
            require_int(report.get("independent_supported_repair_residual_rows"), "supported residual rows")
            for report in reports
        ),
        "independent_unsupported_repair_residual_rows": sum(
            require_int(report.get("independent_unsupported_repair_residual_rows"), "unsupported residual rows")
            for report in reports
        ),
        "residual_independent_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("residual_independent_support_state_counts"), "support counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "residual_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("residual_depth_observation_state_counts"), "depth counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "supported_residual_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("supported_residual_depth_observation_state_counts"),
                                "supported depth counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "residual_owner_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("residual_owner_state_counts"), "owner counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "residual_sample_count": sum(
            require_int(report.get("residual_sample_count"), "residual sample count") for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_depth_repair_residual_owner_state_summary.json", payload)
    return payload


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
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state"),
    )
    parser.add_argument("--max-assign-center-px", type=float, default=260.0)
    parser.add_argument("--near-support-bbox-px", type=float, default=8.0)
    parser.add_argument("--near-support-keypoint-px", type=float, default=24.0)
    parser.add_argument("--local-depth-search-radius-px", type=int, default=8)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
