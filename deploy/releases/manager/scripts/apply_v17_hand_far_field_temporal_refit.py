#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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
    write_json,
)
from build_v17_hand_tail_support_state import existing_path, source_summary
from solve_v17_hand_depth_repair_graph import (
    build_base_row,
    evaluate_row,
    numeric_summary,
    state_counts,
)


STATUS = "v17_hand_far_field_temporal_refit_reprojection_qc"
CLAIM = (
    "This artifact applies the far-field temporal refit deltas to the V17 hand-depth repair graph, "
    "then reprojects and resamples the MANO surface against UniDepth. It tests whether the fixed-sample "
    "temporal refit survives the actual post-update measurement path. It does not update annotations and "
    "does not complete the V3 joint solver."
)


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def temporal_reprojection_state(row: dict[str, Any]) -> str:
    refit_state = row.get("source_temporal_refit_state")
    if refit_state is None:
        return "not_far_field_temporal_refit_row"
    if row.get("temporal_refit_delta_applied") is not True:
        return f"temporal_refit_delta_not_applied:{refit_state}"
    if row.get("metric_depth_compatible") is True:
        return "temporal_refit_reprojected_metric_depth_compatible"
    projection = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    if projection.get("residual_ok") is not True:
        return "temporal_refit_reprojected_projection_untrusted"
    if row.get("temporal_refit_reprojected_depth_improved") is True:
        return "temporal_refit_reprojected_depth_improved_residual_remaining"
    return "temporal_refit_reprojected_residual_remaining"


def refit_delta_by_graph_id(refit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(refit.get("rows"), "temporal refit rows"):
        row = require_dict(raw, "temporal refit row")
        graph_id = require_str(
            row.get("source_hand_depth_repair_graph_variable_id"),
            "source graph id",
        )
        out[graph_id] = row
    return out


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
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
        "hand_far_field_temporal_refit": existing_path(
            args.hand_far_field_temporal_refit_root
            / case
            / "v17_hand_far_field_temporal_refit.json",
            f"{case} hand far-field temporal refit",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    repair = payloads["hand_depth_repair_graph"]
    refit = payloads["hand_far_field_temporal_refit"]
    frame_count = len(frames)
    for name, payload in [
        ("visible-surface", visible),
        ("hand metric-depth", hand_metric),
        ("hand depth repair graph", repair),
        ("hand far-field temporal refit", refit),
    ]:
        if frame_count != require_int(payload.get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} graph annotations disagree with {name} report")
    depth = depth_archive(
        existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive")
    )
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    refit_by_id = refit_delta_by_graph_id(refit)
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[Any, float]] = {}
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[Any, float]] = {}
    rows: list[dict[str, Any]] = []
    for raw in require_list(hand_metric.get("rows"), f"{case} hand metric rows"):
        metric_row = require_dict(raw, "hand metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
        side = require_str(metric_row.get("hand_side"), "metric row hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "metric row hand_index")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} missing annotation frame {frame_idx}")
        base = build_base_row(
            case=case,
            frame=frame,
            metric_row=metric_row,
            hand=hand_index.get((frame_idx, side, hand_i)),
            depth=depth,
            mask_cache=mask_cache,  # type: ignore[arg-type]
            args=args,
        )
        graph_id = require_str(
            base.get("hand_depth_repair_graph_variable_id"),
            "hand depth repair graph variable id",
        )
        repair_row = repair_by_id.get(graph_id)
        if repair_row is None:
            raise RuntimeError(f"{case} missing repair graph row {graph_id}")
        current_shift = repair_row.get("hand_ray_shift_m")
        refit_row = refit_by_id.get(graph_id)
        delta = None
        if refit_row is not None:
            raw_delta = refit_row.get("temporal_refit_delta_shift_m")
            if raw_delta is not None:
                delta = finite_float(raw_delta, f"{case} temporal refit delta")
        if current_shift is None or base.get("base_available") is not True:
            evaluated = evaluate_row(base, None, None, eval_mask_cache, args)  # type: ignore[arg-type]
        else:
            evaluated = evaluate_row(
                base,
                scale,
                finite_float(current_shift, f"{case} current shift") + (0.0 if delta is None else delta),
                eval_mask_cache,  # type: ignore[arg-type]
                args,
            )
        original_gap = repair_row.get("owner_median_gap_m")
        new_gap = evaluated.get("owner_median_gap_m")
        original_abs_gap = None if original_gap is None else abs(finite_float(original_gap, "original gap"))
        new_abs_gap = None if new_gap is None else abs(finite_float(new_gap, "new gap"))
        improved = bool(
            original_abs_gap is not None
            and new_abs_gap is not None
            and original_abs_gap - new_abs_gap >= float(args.min_depth_improvement_m)
        )
        enriched = {
            **evaluated,
            "source_hand_depth_repair_graph_state": repair_row.get("solver_state"),
            "source_hand_depth_repair_graph_metric_depth_compatible": bool(
                repair_row.get("metric_depth_compatible") is True
            ),
            "source_hand_depth_repair_graph_owner_depth_state": repair_row.get("owner_depth_state"),
            "source_hand_depth_repair_graph_owner_median_gap_m": original_gap,
            "source_hand_depth_repair_graph_shift_m": current_shift,
            "source_temporal_refit_variable_id": None
            if refit_row is None
            else refit_row.get("hand_far_field_temporal_refit_variable_id"),
            "source_temporal_refit_state": None if refit_row is None else refit_row.get("temporal_refit_state"),
            "temporal_refit_delta_shift_m": delta,
            "temporal_refit_delta_applied": bool(delta is not None),
            "temporal_refit_reprojected_depth_improved": improved,
            **FALSE_READY,
        }
        rows.append({**enriched, "temporal_reprojection_state": temporal_reprojection_state(enriched)})
    temporal_rows = [row for row in rows if row.get("source_temporal_refit_state") is not None]
    applied_rows = [row for row in temporal_rows if row.get("temporal_refit_delta_applied") is True]
    report = {
        "method": "apply_v17_hand_far_field_temporal_refit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_depth_temporal_reprojection_variable_count": len(rows),
        "temporal_refit_source_rows": len(temporal_rows),
        "temporal_refit_delta_applied_rows": len(applied_rows),
        "temporal_refit_reprojected_metric_depth_compatible_rows": bool_count(
            temporal_rows,
            "metric_depth_compatible",
        ),
        "temporal_refit_reprojected_depth_improved_rows": bool_count(
            temporal_rows,
            "temporal_refit_reprojected_depth_improved",
        ),
        "metric_hand_state_accepted_rows_after_temporal_reprojection": bool_count(
            rows,
            "metric_depth_compatible",
        ),
        "depth_repair_factor_candidate_rows_after_temporal_reprojection": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "temporal_reprojection_state_counts": state_counts(rows, "temporal_reprojection_state"),
        "temporal_refit_reprojection_state_counts": state_counts(
            temporal_rows,
            "temporal_reprojection_state",
        )
        if temporal_rows
        else {},
        "owner_depth_state_counts_after_temporal_reprojection": state_counts(rows, "owner_depth_state"),
        "owner_median_gap_m_after_temporal_reprojection": numeric_summary(rows, "owner_median_gap_m"),
        "source_hand_depth_repair_graph_comparison": {
            "metric_hand_state_accepted_rows": repair.get("metric_hand_state_accepted_rows"),
            "depth_repair_factor_candidate_rows": repair.get("depth_repair_factor_candidate_rows"),
            "owner_depth_state_counts": repair.get("owner_depth_state_counts"),
            "owner_median_gap_m": repair.get("owner_median_gap_m"),
        },
        "source_temporal_refit_comparison": {
            "far_field_temporal_refit_row_count": refit.get("far_field_temporal_refit_row_count"),
            "temporal_refit_variable_candidate_rows": refit.get("temporal_refit_variable_candidate_rows"),
            "temporal_refit_depth_threshold_met_rows": refit.get("temporal_refit_depth_threshold_met_rows"),
            "temporal_refit_bound_hit_rows": refit.get("temporal_refit_bound_hit_rows"),
            "temporal_refit_state_counts": refit.get("temporal_refit_state_counts"),
        },
        "problem_semantics": {
            "temporal_refit_reprojected_metric_depth_compatible": "the temporal refit row remains metric-depth compatible after applying the delta, reprojecting MANO, and resampling UniDepth",
            "temporal_refit_reprojected_depth_improved_residual_remaining": "the applied temporal delta improves the reprojected row but does not meet depth thresholds after resampling",
            "claim_limit": "this artifact evaluates an applied delta; it does not re-run the full joint hand/object/contact graph and does not accept annotation state",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_far_field_temporal_reprojection.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_far_field_temporal_refit_root / "v17_hand_far_field_temporal_refit_summary.json",
        "hand far-field temporal refit summary",
    )
    summary = require_dict(load_json(summary_path), "hand far-field temporal refit summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "apply_v17_hand_far_field_temporal_refit",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_far_field_temporal_refit_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_far_field_temporal_reprojection.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "temporal_refit_source_rows": require_int(
                    report.get("temporal_refit_source_rows"),
                    "temporal source rows",
                ),
                "temporal_refit_delta_applied_rows": require_int(
                    report.get("temporal_refit_delta_applied_rows"),
                    "temporal applied rows",
                ),
                "temporal_refit_reprojected_metric_depth_compatible_rows": require_int(
                    report.get("temporal_refit_reprojected_metric_depth_compatible_rows"),
                    "temporal compatible rows",
                ),
                "temporal_refit_reprojected_depth_improved_rows": require_int(
                    report.get("temporal_refit_reprojected_depth_improved_rows"),
                    "temporal improved rows",
                ),
                "metric_hand_state_accepted_rows_after_temporal_reprojection": require_int(
                    report.get("metric_hand_state_accepted_rows_after_temporal_reprojection"),
                    "accepted rows",
                ),
                "depth_repair_factor_candidate_rows_after_temporal_reprojection": require_int(
                    report.get("depth_repair_factor_candidate_rows_after_temporal_reprojection"),
                    "repair rows",
                ),
                "temporal_refit_reprojection_state_counts": require_dict(
                    report.get("temporal_refit_reprojection_state_counts"),
                    "temporal refit state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "temporal_refit_source_rows": sum(
            require_int(report.get("temporal_refit_source_rows"), "temporal source rows")
            for report in reports
        ),
        "temporal_refit_delta_applied_rows": sum(
            require_int(report.get("temporal_refit_delta_applied_rows"), "temporal applied rows")
            for report in reports
        ),
        "temporal_refit_reprojected_metric_depth_compatible_rows": sum(
            require_int(report.get("temporal_refit_reprojected_metric_depth_compatible_rows"), "compatible rows")
            for report in reports
        ),
        "temporal_refit_reprojected_depth_improved_rows": sum(
            require_int(report.get("temporal_refit_reprojected_depth_improved_rows"), "improved rows")
            for report in reports
        ),
        "metric_hand_state_accepted_rows_after_temporal_reprojection": sum(
            require_int(
                report.get("metric_hand_state_accepted_rows_after_temporal_reprojection"),
                "accepted rows",
            )
            for report in reports
        ),
        "depth_repair_factor_candidate_rows_after_temporal_reprojection": sum(
            require_int(
                report.get("depth_repair_factor_candidate_rows_after_temporal_reprojection"),
                "repair rows",
            )
            for report in reports
        ),
        "temporal_refit_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("temporal_refit_reprojection_state_counts"), "states"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_far_field_temporal_reprojection_summary.json", payload)
    return payload


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
        "--hand-metric-depth-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument(
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--hand-far-field-temporal-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--min-depth-improvement-m", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
