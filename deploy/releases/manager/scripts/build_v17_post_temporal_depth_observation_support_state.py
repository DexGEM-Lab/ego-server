#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    annotation_frames,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    write_json,
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


STATUS = "v17_post_temporal_depth_observation_support_state_qc"
CLAIM = (
    "This artifact tests whether the post-temporal depth-observation residual pixels are supported by "
    "model-produced 2D hand evidence. It distinguishes real visible-hand depth-observation rows from "
    "projected-hand spillover before any coupled solver changes the hand-depth objective."
)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


SAME_SIDE_INDEPENDENT_SUPPORT_STATES = {
    "tail_pixels_inside_same_side_independent_model_box",
    "tail_pixels_near_same_side_independent_model_box",
}


def same_side_independent_supported(row: dict[str, Any]) -> bool:
    return row.get("independent_support_state") in SAME_SIDE_INDEPENDENT_SUPPORT_STATES


def independent_keypoint_fraction(support: dict[str, Any]) -> float | None:
    same_side = support.get("same_side_independent_models")
    if not isinstance(same_side, dict):
        return None
    value = same_side.get("near_keypoint_fraction")
    if not isinstance(value, int | float) or not np.isfinite(float(value)):
        return None
    return float(value)


def independent_keypoint_support_state(support: dict[str, Any], args: argparse.Namespace) -> str:
    fraction = independent_keypoint_fraction(support)
    if fraction is None:
        return "same_side_independent_keypoints_unmeasured"
    if fraction >= float(args.strong_keypoint_supported_fraction):
        return "same_side_independent_keypoint_strong"
    if fraction >= float(args.min_keypoint_supported_fraction):
        return "same_side_independent_keypoint_partial"
    if fraction > 0.0:
        return "same_side_independent_keypoint_sparse"
    return "same_side_independent_keypoint_absent"


def independent_keypoint_supported(row: dict[str, Any]) -> bool:
    return row.get("independent_keypoint_support_state") in {
        "same_side_independent_keypoint_partial",
        "same_side_independent_keypoint_strong",
    }


def independent_keypoint_strong(row: dict[str, Any]) -> bool:
    return row.get("independent_keypoint_support_state") == "same_side_independent_keypoint_strong"


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "post_temporal_depth_observation_state": existing_path(
            args.post_temporal_depth_observation_state_root
            / case
            / "v17_post_temporal_depth_observation_state.json",
            f"{case} post-temporal depth-observation state",
        ),
        "hand_temporal_owner_weighted_refit": existing_path(
            args.hand_temporal_owner_weighted_refit_root
            / case
            / "v17_hand_temporal_owner_weighted_refit.json",
            f"{case} hand temporal owner-weighted refit",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hands = annotation_hand_index(frames)
    support_sources = case_support_sources(case, args)
    source_report = payloads["post_temporal_depth_observation_state"]
    refit_report = payloads["hand_temporal_owner_weighted_refit"]
    if require_int(source_report.get("frame_count"), f"{case} source frame_count") != require_int(
        refit_report.get("frame_count"),
        f"{case} refit frame_count",
    ):
        raise RuntimeError(f"{case} post-temporal depth-observation frame count disagrees with refit")
    refit_by_graph_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "refit graph id"): row
        for row in [
            require_dict(raw, "owner-weighted refit row")
            for raw in require_list(refit_report.get("rows"), f"{case} owner-weighted refit rows")
        ]
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(source_report.get("rows"), f"{case} post-temporal depth-observation rows"):
        source = require_dict(raw, "post-temporal depth-observation row")
        graph_id = require_str(
            source.get("source_hand_depth_repair_graph_variable_id"),
            "source hand-depth repair graph id",
        )
        refit_row = require_dict(refit_by_graph_id.get(graph_id), f"{case} owner-weighted row {graph_id}")
        frame_idx = require_int(source.get("frame_idx"), "frame_idx")
        side = require_str(source.get("hand_side"), "hand_side")
        hand_i = require_int(source.get("hand_index"), "hand_index")
        frame = frames.get(frame_idx)
        hand = hands.get((frame_idx, side, hand_i))
        base = {
            "case": case,
            "post_temporal_depth_observation_support_state_id": require_str(
                source.get("post_temporal_depth_observation_state_id"),
                "post-temporal depth-observation id",
            ).replace(
                "post_temporal_depth_observation:",
                "post_temporal_depth_observation_support:",
                1,
            ),
            "source_post_temporal_depth_observation_state_id": require_str(
                source.get("post_temporal_depth_observation_state_id"),
                "post-temporal depth-observation id",
            ),
            "source_hand_depth_repair_graph_variable_id": graph_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_i,
            "source_post_temporal_depth_observation_state": source.get(
                "post_temporal_depth_observation_state"
            ),
            "local_assignment_state": source.get("local_assignment_state"),
            "residual_sign_state": source.get("residual_sign_state"),
            "owner_sample_partition": source.get("owner_sample_partition"),
            **FALSE_READY,
        }
        if frame is None or hand is None:
            rows.append(
                {
                    **base,
                    "selected_support_state": "missing_annotation_hand",
                    "independent_support_state": "missing_annotation_hand",
                    "independent_keypoint_support_state": "same_side_independent_keypoints_unmeasured",
                    "missing_support_inputs": ["annotation_hand"],
                }
            )
            continue
        samples = row_samples(refit_row)
        selected = selected_residual(refit_row, samples, args)
        if not np.any(selected):
            rows.append(
                {
                    **base,
                    "selected_support_state": "unobserved_depth_observation_pixels_for_support",
                    "independent_support_state": "unobserved_depth_observation_pixels_for_support",
                    "independent_keypoint_support_state": "same_side_independent_keypoints_unmeasured",
                    "missing_support_inputs": ["selected_residual_pixels"],
                }
            )
            continue
        shapes = support_shapes_for_row(
            frame=frame,
            hand_i=hand_i,
            support_sources=support_sources,
            args=args,
        )
        support = subset_support(
            x=cast(np.ndarray, samples["x"]).astype(np.int32),
            y=cast(np.ndarray, samples["y"]).astype(np.int32),
            selected=selected,
            shapes=shapes,
            projection_source_size=cast(tuple[float, float], samples["projection_source_size"]),
            depth_shape=cast(tuple[int, int], samples["depth_shape"]),
            args=args,
        )
        selected_state = selected_support_state(
            {**base, "tail_factor_candidate": True},
            support,
        )
        independent_state = independent_support_state(
            {**base, "tail_factor_candidate": True},
            support,
        )
        rows.append(
            {
                **base,
                "selected_support_state": selected_state,
                "independent_support_state": independent_state,
                "independent_keypoint_support_state": independent_keypoint_support_state(support, args),
                "selected_residual_sample_count": int(np.count_nonzero(selected)),
                "support_shape_counts": {name: len(value) for name, value in shapes.items()},
                "support": support,
                "missing_support_inputs": [],
            }
        )
    report = {
        "method": "build_v17_post_temporal_depth_observation_support_state",
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
        "frame_count": require_int(source_report.get("frame_count"), f"{case} source frame_count"),
        "post_temporal_depth_observation_support_variable_count": len(rows),
        "post_temporal_depth_observation_support_candidate_rows": len(rows),
        "selected_support_state_counts": state_counts(rows, "selected_support_state"),
        "independent_support_state_counts": state_counts(rows, "independent_support_state"),
        "independent_keypoint_support_state_counts": state_counts(rows, "independent_keypoint_support_state"),
        "independent_supported_depth_observation_rows": sum(
            1 for row in rows if same_side_independent_supported(row)
        ),
        "independent_unsupported_depth_observation_rows": sum(
            1 for row in rows if not same_side_independent_supported(row)
        ),
        "independent_keypoint_supported_depth_observation_rows": sum(
            1 for row in rows if independent_keypoint_supported(row)
        ),
        "independent_keypoint_strong_depth_observation_rows": sum(
            1 for row in rows if independent_keypoint_strong(row)
        ),
        "source_depth_observation_state_counts": state_counts(
            rows,
            "source_post_temporal_depth_observation_state",
        ),
        "local_assignment_state_counts": state_counts(rows, "local_assignment_state"),
        "residual_sign_state_counts": state_counts(rows, "residual_sign_state"),
        "selected_residual_sample_count": sum(
            require_int(row.get("selected_residual_sample_count", 0), "selected residual samples")
            for row in rows
        ),
        "problem_semantics": {
            "independent_support_state": (
                "same-side RTMLib, WiLoR, HaMeR, and VLM hand evidence is tested against selected residual pixels"
            ),
            "unsupported_depth_observation_row": (
                "selected residual pixels lack independent 2D hand support and should be treated as projection/support ownership before depth equality"
            ),
            "supported_depth_observation_row": (
                "selected residual pixels have independent hand support and need a hand-depth observation variable in the coupled graph"
            ),
            "independent_keypoint_support_state": (
                "same-side independent keypoints provide graded anatomical support inside the box-supported depth-observation rows"
            ),
            "claim_limit": (
                "this state measures 2D support for post-temporal depth-observation rows; hand geometry remains unchanged"
            ),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_post_temporal_depth_observation_support_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.post_temporal_depth_observation_state_root
        / "v17_post_temporal_depth_observation_state_summary.json",
        "post-temporal depth-observation summary",
    )
    summary = require_dict(load_json(summary_path), "post-temporal depth-observation summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_post_temporal_depth_observation_support_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_post_temporal_depth_observation_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_post_temporal_depth_observation_support_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "post_temporal_depth_observation_support_variable_count": require_int(
                    report.get("post_temporal_depth_observation_support_variable_count"),
                    "support variable count",
                ),
                "post_temporal_depth_observation_support_candidate_rows": require_int(
                    report.get("post_temporal_depth_observation_support_candidate_rows"),
                    "support candidate rows",
                ),
                "selected_support_state_counts": require_dict(
                    report.get("selected_support_state_counts"),
                    "selected support state counts",
                ),
                "independent_support_state_counts": require_dict(
                    report.get("independent_support_state_counts"),
                    "independent support state counts",
                ),
                "independent_keypoint_support_state_counts": require_dict(
                    report.get("independent_keypoint_support_state_counts"),
                    "independent keypoint support state counts",
                ),
                "independent_supported_depth_observation_rows": require_int(
                    report.get("independent_supported_depth_observation_rows"),
                    "independent supported rows",
                ),
                "independent_unsupported_depth_observation_rows": require_int(
                    report.get("independent_unsupported_depth_observation_rows"),
                    "independent unsupported rows",
                ),
                "independent_keypoint_supported_depth_observation_rows": require_int(
                    report.get("independent_keypoint_supported_depth_observation_rows"),
                    "independent keypoint supported rows",
                ),
                "independent_keypoint_strong_depth_observation_rows": require_int(
                    report.get("independent_keypoint_strong_depth_observation_rows"),
                    "independent keypoint strong rows",
                ),
                "source_depth_observation_state_counts": require_dict(
                    report.get("source_depth_observation_state_counts"),
                    "source depth-observation state counts",
                ),
                "local_assignment_state_counts": require_dict(
                    report.get("local_assignment_state_counts"),
                    "local assignment state counts",
                ),
                "residual_sign_state_counts": require_dict(
                    report.get("residual_sign_state_counts"),
                    "residual sign state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "post_temporal_depth_observation_support_variable_count": sum(
            require_int(
                report.get("post_temporal_depth_observation_support_variable_count"),
                "support variable count",
            )
            for report in reports
        ),
        "post_temporal_depth_observation_support_candidate_rows": sum(
            require_int(
                report.get("post_temporal_depth_observation_support_candidate_rows"),
                "support candidate rows",
            )
            for report in reports
        ),
        "selected_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("selected_support_state_counts"), "selected counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "independent_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(report.get("independent_support_state_counts"), "independent counts")
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "independent_keypoint_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("independent_keypoint_support_state_counts"),
                                "independent keypoint counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "independent_supported_depth_observation_rows": sum(
            require_int(report.get("independent_supported_depth_observation_rows"), "supported rows")
            for report in reports
        ),
        "independent_unsupported_depth_observation_rows": sum(
            require_int(report.get("independent_unsupported_depth_observation_rows"), "unsupported rows")
            for report in reports
        ),
        "independent_keypoint_supported_depth_observation_rows": sum(
            require_int(
                report.get("independent_keypoint_supported_depth_observation_rows"),
                "keypoint supported rows",
            )
            for report in reports
        ),
        "independent_keypoint_strong_depth_observation_rows": sum(
            require_int(
                report.get("independent_keypoint_strong_depth_observation_rows"),
                "keypoint strong rows",
            )
            for report in reports
        ),
        "source_depth_observation_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("source_depth_observation_state_counts"),
                                "source state counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "local_assignment_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("local_assignment_state_counts"),
                                "local assignment counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "residual_sign_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("residual_sign_state_counts"), "sign counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "selected_residual_sample_count": sum(
            require_int(report.get("selected_residual_sample_count"), "selected residual samples")
            for report in reports
        ),
        "source_depth_observation_comparison": {
            "post_temporal_depth_observation_candidate_rows": summary.get(
                "post_temporal_depth_observation_candidate_rows"
            ),
            "post_temporal_depth_observation_state_counts": summary.get(
                "post_temporal_depth_observation_state_counts"
            ),
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_post_temporal_depth_observation_support_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--post-temporal-depth-observation-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_state"),
    )
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state"),
    )
    parser.add_argument("--max-assignment-distance-px", type=float, default=200.0)
    parser.add_argument("--max-assign-center-px", type=float, default=160.0)
    parser.add_argument("--near-support-bbox-px", type=float, default=24.0)
    parser.add_argument("--near-support-keypoint-px", type=float, default=32.0)
    parser.add_argument("--min-keypoint-supported-fraction", type=float, default=0.25)
    parser.add_argument("--strong-keypoint-supported-fraction", type=float, default=0.5)
    parser.add_argument("--near-box-margin-px", type=float, default=24.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
