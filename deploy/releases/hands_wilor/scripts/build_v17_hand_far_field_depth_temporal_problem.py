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


STATUS = "v17_hand_far_field_depth_temporal_problem_qc"
CLAIM = (
    "This artifact groups far-field hand-depth residual switches into temporal runs. "
    "It tests whether the dominant depth-observation residual owner is frame-isolated noise or a "
    "persistent hand-depth state that needs full-timeline temporal variables. It does not update hand "
    "annotations and does not complete the V3 joint solver."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def depth_sign(row: dict[str, Any], args: argparse.Namespace) -> str:
    median = finite_float(
        require_dict(
            require_dict(row.get("partition_summary"), "partition summary").get(
                "selected_residual_gap_m"
            ),
            "selected residual gap",
        ).get("median"),
        "selected residual median gap",
    )
    if median > float(args.max_median_abs_depth_gap_m):
        return "hand_behind_metric_depth"
    if median < -float(args.max_median_abs_depth_gap_m):
        return "hand_in_front_of_metric_depth"
    return "signed_depth_tail_near_zero_median"


def segment_state(segment: dict[str, Any], args: argparse.Namespace) -> str:
    if require_int(segment.get("frame_count"), "segment frame_count") >= int(args.min_segment_frames):
        return "far_field_temporal_factor_candidate"
    return "short_far_field_depth_switch_run"


def make_segment(
    *,
    case: str,
    segment_index: int,
    key: tuple[str, int, str],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hand_side, hand_index, sign = key
    frame_indices = [require_int(row.get("frame_idx"), "frame_idx") for row in rows]
    gaps = [
        finite_float(
            require_dict(
                require_dict(row.get("partition_summary"), "partition summary").get(
                    "selected_residual_gap_m"
                ),
                "selected residual gap",
            ).get("median"),
            "selected residual median gap",
        )
        for row in rows
    ]
    selected_samples = [
        require_int(
            require_dict(row.get("partition_summary"), "partition summary").get(
                "selected_residual_sample_count"
            ),
            "selected residual sample count",
        )
        for row in rows
    ]
    near_samples = [
        require_int(
            require_dict(row.get("partition_summary"), "partition summary").get(
                "near_active_object_residual_sample_count"
            ),
            "near residual sample count",
        )
        for row in rows
    ]
    far_samples = [
        require_int(
            require_dict(row.get("partition_summary"), "partition summary").get(
                "far_from_active_object_residual_sample_count"
            ),
            "far residual sample count",
        )
        for row in rows
    ]
    segment = {
        "case": case,
        "hand_far_field_depth_temporal_segment_id": (
            f"hand_far_field_depth_temporal:v17:{case}:{hand_side}:{hand_index}:{sign}:{segment_index:04d}"
        ),
        "hand_side": hand_side,
        "hand_index": hand_index,
        "depth_sign_state": sign,
        "start_frame_idx": min(frame_indices),
        "end_frame_idx": max(frame_indices),
        "frame_count": len(frame_indices),
        "source_hand_depth_observation_switch_variable_ids": [
            require_str(
                row.get("hand_depth_observation_switch_variable_id"),
                "depth observation switch variable id",
            )
            for row in rows
        ],
        "source_hand_residual_switch_variable_ids": [
            require_str(row.get("source_hand_residual_switch_variable_id"), "residual switch variable id")
            for row in rows
        ],
        "selected_residual_gap_m": summarize(gaps),
        "selected_residual_sample_count": int(sum(selected_samples)),
        "near_active_object_residual_sample_count": int(sum(near_samples)),
        "far_from_active_object_residual_sample_count": int(sum(far_samples)),
        **FALSE_READY,
    }
    return {**segment, "temporal_segment_state": segment_state(segment, args)}


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    path = existing_path(
        args.hand_depth_observation_switch_problem_root
        / case
        / "v17_hand_depth_observation_switch_problem.json",
        f"{case} hand depth-observation switch problem",
    )
    report = require_dict(load_json(path), f"{case} depth-observation switch report")
    frame_count = require_int(report.get("frame_count"), f"{case} frame_count")
    far_rows = [
        require_dict(row, "depth observation row")
        for row in require_list(report.get("rows"), f"{case} rows")
        if require_dict(row, "depth observation row").get("depth_observation_switch_state")
        == "far_field_hand_depth_observation_switch"
    ]
    ordered = sorted(
        far_rows,
        key=lambda row: (
            require_str(row.get("hand_side"), "hand_side"),
            require_int(row.get("hand_index"), "hand_index"),
            depth_sign(row, args),
            require_int(row.get("frame_idx"), "frame_idx"),
        ),
    )
    segments: list[dict[str, Any]] = []
    current_key: tuple[str, int, str] | None = None
    current_rows: list[dict[str, Any]] = []
    last_frame: int | None = None
    segment_index = 0
    for row in ordered:
        key = (
            require_str(row.get("hand_side"), "hand_side"),
            require_int(row.get("hand_index"), "hand_index"),
            depth_sign(row, args),
        )
        frame_idx = require_int(row.get("frame_idx"), "frame_idx")
        if current_key is None:
            current_key = key
            current_rows = [row]
            last_frame = frame_idx
            continue
        if key == current_key and last_frame is not None and frame_idx == last_frame + 1:
            current_rows.append(row)
            last_frame = frame_idx
            continue
        segments.append(
            make_segment(case=case, segment_index=segment_index, key=current_key, rows=current_rows, args=args)
        )
        segment_index += 1
        current_key = key
        current_rows = [row]
        last_frame = frame_idx
    if current_key is not None:
        segments.append(
            make_segment(case=case, segment_index=segment_index, key=current_key, rows=current_rows, args=args)
        )
    candidate_segments = [
        segment
        for segment in segments
        if segment.get("temporal_segment_state") == "far_field_temporal_factor_candidate"
    ]
    report_out = {
        "method": "build_v17_hand_far_field_depth_temporal_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"hand_depth_observation_switch_problem": source_summary(path, report)},
        "frame_count": frame_count,
        "far_field_depth_switch_rows": len(far_rows),
        "far_field_depth_temporal_segment_count": len(segments),
        "far_field_temporal_factor_candidate_segments": len(candidate_segments),
        "far_field_temporal_factor_candidate_rows": sum(
            require_int(segment.get("frame_count"), "candidate segment frame_count")
            for segment in candidate_segments
        ),
        "longest_far_field_temporal_segment_frames": max(
            [require_int(segment.get("frame_count"), "segment frame_count") for segment in segments],
            default=0,
        ),
        "far_field_temporal_segment_state_counts": state_counts(segments, "temporal_segment_state"),
        "far_field_temporal_depth_sign_state_counts": state_counts(segments, "depth_sign_state"),
        "source_depth_observation_switch_comparison": {
            "far_field_hand_depth_observation_switch_rows": report.get(
                "far_field_hand_depth_observation_switch_rows"
            ),
            "depth_observation_switch_candidate_rows": report.get("depth_observation_switch_candidate_rows"),
        },
        "problem_semantics": {
            "far_field_temporal_factor_candidate": "consecutive far-field residual rows with the same hand and signed depth owner support a temporal hand-depth variable",
            "short_far_field_depth_switch_run": "far-field residual row run is too short for the temporal factor threshold",
        },
        "segments": segments,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_far_field_depth_temporal_problem.json", report_out)
    return report_out


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_depth_observation_switch_problem_root
        / "v17_hand_depth_observation_switch_problem_summary.json",
        "hand depth-observation switch summary",
    )
    summary = require_dict(load_json(summary_path), "hand depth-observation switch summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_far_field_depth_temporal_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_depth_observation_switch_problem_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_far_field_depth_temporal_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "far_field_depth_switch_rows": require_int(
                    report.get("far_field_depth_switch_rows"),
                    "far-field depth switch rows",
                ),
                "far_field_depth_temporal_segment_count": require_int(
                    report.get("far_field_depth_temporal_segment_count"),
                    "temporal segment count",
                ),
                "far_field_temporal_factor_candidate_segments": require_int(
                    report.get("far_field_temporal_factor_candidate_segments"),
                    "temporal candidate segments",
                ),
                "far_field_temporal_factor_candidate_rows": require_int(
                    report.get("far_field_temporal_factor_candidate_rows"),
                    "temporal candidate rows",
                ),
                "longest_far_field_temporal_segment_frames": require_int(
                    report.get("longest_far_field_temporal_segment_frames"),
                    "longest temporal segment",
                ),
                "far_field_temporal_segment_state_counts": require_dict(
                    report.get("far_field_temporal_segment_state_counts"),
                    "segment state counts",
                ),
                "far_field_temporal_depth_sign_state_counts": require_dict(
                    report.get("far_field_temporal_depth_sign_state_counts"),
                    "depth sign counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "far_field_depth_switch_rows": sum(
            require_int(report.get("far_field_depth_switch_rows"), "far-field rows") for report in reports
        ),
        "far_field_depth_temporal_segment_count": sum(
            require_int(report.get("far_field_depth_temporal_segment_count"), "segment count")
            for report in reports
        ),
        "far_field_temporal_factor_candidate_segments": sum(
            require_int(report.get("far_field_temporal_factor_candidate_segments"), "candidate segments")
            for report in reports
        ),
        "far_field_temporal_factor_candidate_rows": sum(
            require_int(report.get("far_field_temporal_factor_candidate_rows"), "candidate rows")
            for report in reports
        ),
        "longest_far_field_temporal_segment_frames": max(
            [
                require_int(report.get("longest_far_field_temporal_segment_frames"), "longest segment")
                for report in reports
            ],
            default=0,
        ),
        "far_field_temporal_segment_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("far_field_temporal_segment_state_counts"), "states"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "far_field_temporal_depth_sign_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("far_field_temporal_depth_sign_state_counts"), "states"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_far_field_depth_temporal_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-depth-observation-switch-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_observation_switch_problem"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_depth_temporal_problem"),
    )
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--min-segment-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
