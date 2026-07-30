#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    annotation_frames,
    depth_archive,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary
from build_v17_mano_articulation_factor_input import (
    assignment_pairs,
    repaired_surface_vertex_ids,
)


STATUS = "v17_post_temporal_mano_factor_input_qc"
CLAIM = (
    "This artifact materializes MANO surface correspondence factors after the owner-weighted temporal "
    "hand-depth refit. It consumes the current reprojected rows, builds residual-to-compatible-depth "
    "MANO vertex pairs for local and mixed owner rows, and leaves depth-observation rows explicit. "
    "It does not optimize MANO pose and does not complete the V3 joint solver."
)

LOCAL_STATE = "owner_weighted_reprojected_local_surface_factor_candidate"
MIXED_STATE = "owner_weighted_reprojected_mixed_surface_depth_owner"
DEPTH_STATE = "owner_weighted_reprojected_depth_observation_owner"
UNTRUSTED_STATE = "owner_weighted_reprojected_projection_untrusted"
COMPATIBLE_STATE = "owner_weighted_reprojected_metric_depth_compatible"


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
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
    refit = payloads["hand_temporal_owner_weighted_refit"]
    frame_count = len(frames)
    if frame_count != require_int(refit.get("frame_count"), f"{case} refit frame_count"):
        raise RuntimeError(f"{case} frame count disagrees with owner-weighted refit")
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth npz",
    )
    depth = depth_archive(depth_path)
    rows: list[dict[str, Any]] = []
    for raw in require_list(refit.get("rows"), f"{case} owner-weighted rows"):
        source = require_dict(raw, "owner-weighted row")
        source_state = source.get("owner_weighted_reprojection_state")
        if source_state not in {LOCAL_STATE, MIXED_STATE}:
            continue
        graph_id = require_str(source.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        frame_idx = require_int(source.get("frame_idx"), "frame_idx")
        side = require_str(source.get("hand_side"), "hand_side")
        hand_index = require_int(source.get("hand_index"), "hand_index")
        hand = require_dict(hands.get((frame_idx, side, hand_index)), f"{case} annotation hand {graph_id}")
        samples = row_samples(source)
        selected = selected_residual(source, samples, args)
        vertex_id = repaired_surface_vertex_ids(graph_row=source, hand=hand, depth=depth, args=args)
        pairs = assignment_pairs(source, samples, selected, vertex_id, args)
        assigned = require_int(pairs.get("assigned_residual_sample_count"), "assigned residual samples")
        materialized = bool(assigned > 0)
        rows.append(
            {
                "case": case,
                "post_temporal_mano_factor_input_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "post_temporal_mano_factor_input:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_hand_temporal_owner_weighted_refit_variable_id": source.get(
                    "hand_temporal_owner_weighted_refit_variable_id"
                ),
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_index": hand_index,
                "source_owner_weighted_reprojection_state": source_state,
                "post_temporal_factor_input_state": "post_temporal_mano_factor_input_materialized"
                if materialized
                else "post_temporal_mano_factor_input_unassigned",
                "post_temporal_mano_factor_input_materialized": materialized,
                "post_temporal_mano_local_surface_factor_row": bool(source_state == LOCAL_STATE),
                "post_temporal_mano_mixed_surface_depth_factor_row": bool(source_state == MIXED_STATE),
                "owner_sample_partition": source.get("owner_sample_partition"),
                "owner_depth_state": source.get("owner_depth_state"),
                "owner_weighted_delta_shift_m": source.get("owner_weighted_delta_shift_m"),
                "owner_weighted_total_hand_ray_shift_m": source.get(
                    "owner_weighted_total_hand_ray_shift_m"
                ),
                "projection_residual_to_measurement_px": source.get("projection_residual_to_measurement_px"),
                "assignment": pairs,
                **FALSE_READY,
            }
        )
    materialized_rows = [row for row in rows if row.get("post_temporal_mano_factor_input_materialized") is True]
    report = {
        "method": "build_v17_post_temporal_mano_factor_input",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "post_temporal_mano_factor_input_candidate_rows": len(rows),
        "post_temporal_mano_factor_input_materialized_rows": len(materialized_rows),
        "post_temporal_mano_local_surface_factor_rows": bool_count(
            materialized_rows,
            "post_temporal_mano_local_surface_factor_row",
        ),
        "post_temporal_mano_mixed_surface_depth_factor_rows": bool_count(
            materialized_rows,
            "post_temporal_mano_mixed_surface_depth_factor_row",
        ),
        "post_temporal_factor_input_state_counts": state_counts(rows, "post_temporal_factor_input_state"),
        "source_owner_weighted_reprojection_state_counts": state_counts(
            rows,
            "source_owner_weighted_reprojection_state",
        ),
        "assigned_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"), "assigned")
            for row in rows
        ),
        "residual_factor_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("residual_sample_count"), "residual")
            for row in rows
        ),
        "compatible_seed_sample_count": sum(
            require_int(require_dict(row.get("assignment"), "assignment").get("compatible_seed_sample_count"), "seed")
            for row in rows
        ),
        "assigned_pixel_shift_px": summarize(
            [
                float(require_dict(row.get("assignment"), "assignment")["assigned_pixel_shift_px"]["median"])
                for row in rows
                if require_int(
                    require_dict(row.get("assignment"), "assignment").get("assigned_residual_sample_count"),
                    "assigned",
                )
                > 0
            ]
        ),
        "source_owner_weighted_refit_comparison": {
            "owner_weighted_reprojection_local_surface_factor_candidate_rows": refit.get(
                "owner_weighted_reprojection_local_surface_factor_candidate_rows"
            ),
            "owner_weighted_reprojection_mixed_surface_depth_owner_rows": refit.get(
                "owner_weighted_reprojection_mixed_surface_depth_owner_rows"
            ),
            "owner_weighted_reprojection_depth_observation_owner_rows": refit.get(
                "owner_weighted_reprojection_depth_observation_owner_rows"
            ),
            "owner_weighted_reprojection_state_counts": refit.get(
                "owner_weighted_temporal_reprojection_state_counts"
            ),
        },
        "problem_semantics": {
            "surface_correspondence": "front-most owner-weighted MANO vertex id at each residual or compatible seed depth pixel",
            "factor_pair": "a post-temporal residual hand-surface pixel paired with a nearby same-hand compatible-depth seed pixel",
            "solver_implication": "materialized rows can feed a MANO articulation/surface solve, while depth-observation rows remain explicit non-geometry owners",
        },
        "parameters": {
            "local_projection_search_radius_px": float(args.local_projection_search_radius_px),
            "compatible_depth_abs_m": float(args.compatible_depth_abs_m),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
            "max_surface_depth_reconstruction_error_m": float(args.max_surface_depth_reconstruction_error_m),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_post_temporal_mano_factor_input.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = ["trash_1050", "task5_tomato_960"]
    reports = [case_problem(case, args) for case in cases]
    summary = {
        "method": "build_v17_post_temporal_mano_factor_input",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "cases": cases,
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "post_temporal_mano_factor_input_candidate_rows": sum(
            require_int(report.get("post_temporal_mano_factor_input_candidate_rows"), "candidate rows")
            for report in reports
        ),
        "post_temporal_mano_factor_input_materialized_rows": sum(
            require_int(report.get("post_temporal_mano_factor_input_materialized_rows"), "materialized rows")
            for report in reports
        ),
        "post_temporal_mano_local_surface_factor_rows": sum(
            require_int(report.get("post_temporal_mano_local_surface_factor_rows"), "local rows")
            for report in reports
        ),
        "post_temporal_mano_mixed_surface_depth_factor_rows": sum(
            require_int(report.get("post_temporal_mano_mixed_surface_depth_factor_rows"), "mixed rows")
            for report in reports
        ),
        "assigned_factor_sample_count": sum(
            require_int(report.get("assigned_factor_sample_count"), "assigned samples") for report in reports
        ),
        "residual_factor_sample_count": sum(
            require_int(report.get("residual_factor_sample_count"), "residual samples") for report in reports
        ),
        "compatible_seed_sample_count": sum(
            require_int(report.get("compatible_seed_sample_count"), "compatible seed samples") for report in reports
        ),
        "post_temporal_factor_input_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("post_temporal_factor_input_state_counts"), "state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "source_owner_weighted_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("source_owner_weighted_reprojection_state_counts"), "source counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "case_reports": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "post_temporal_mano_factor_input_materialized_rows": require_int(
                    report.get("post_temporal_mano_factor_input_materialized_rows"),
                    "materialized rows",
                ),
                "post_temporal_mano_local_surface_factor_rows": require_int(
                    report.get("post_temporal_mano_local_surface_factor_rows"),
                    "local rows",
                ),
                "post_temporal_mano_mixed_surface_depth_factor_rows": require_int(
                    report.get("post_temporal_mano_mixed_surface_depth_factor_rows"),
                    "mixed rows",
                ),
                "assigned_factor_sample_count": require_int(
                    report.get("assigned_factor_sample_count"),
                    "assigned samples",
                ),
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_post_temporal_mano_factor_input_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--max-surface-depth-reconstruction-error-m", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
