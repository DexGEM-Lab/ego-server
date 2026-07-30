#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, write_json


def optional_load(path: Path | None) -> Any | None:
    if path is None:
        return None
    return load_json(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    depth_registry = optional_load(args.depth_registry)
    depth_selection = optional_load(args.depth_selection)
    geometry_registry = optional_load(args.geometry_registry)
    geometry_validation = optional_load(args.geometry_validation)
    hand_shape = optional_load(args.hand_shape_report)
    contact_rows = optional_load(args.contact_render_rows)
    bundle: dict[str, Any] = {
        "schema": "v20_observation_bundle.v0",
        "claim_scope": "V20 observations selected for optimization/rendering from prediction-side measurements only. Benchmark eval refs are consumed only by evaluation.",
        "evaluation_reference_policy": "prediction_state_and_renderer_inputs_must_not_contain_evaluation_reference_sources",
        "depth": {},
        "geometry_candidates": {},
        "hand_shape": {},
        "contact_visualization": {},
    }
    if depth_registry is not None:
        bundle["depth"]["candidate_registry"] = str(args.depth_registry)
        bundle["depth"]["candidate_count"] = depth_registry.get("candidate_count", len(depth_registry.get("candidates", [])))
    if depth_selection is not None:
        bundle["depth"]["selection_report"] = str(args.depth_selection)
        bundle["depth"]["primary_depth_by_scope"] = depth_selection.get("selected")
    weak_depth_reports = []
    for path in args.weak_depth_report or []:
        payload = load_json(path)
        if payload.get("metric_depth_available") is not False:
            raise ContractError(f"weak_depth_report_must_declare_metric_depth_unavailable: {path}")
        weak_depth_reports.append({
            "path": str(path),
            "method": payload.get("method"),
            "status": payload.get("status"),
            "selection_status": payload.get("selection_status"),
            "frame_count": payload.get("frame_count"),
            "valid_fraction": payload.get("valid_fraction"),
            "claim_scope": payload.get("claim_scope"),
        })
    if weak_depth_reports:
        bundle["depth"]["weak_nonmetric_reports"] = weak_depth_reports
    if geometry_registry is not None:
        bundle["geometry_candidates"]["registry"] = str(args.geometry_registry)
        bundle["geometry_candidates"]["candidate_count"] = geometry_registry.get("candidate_count", len(geometry_registry.get("candidates", [])))
    if geometry_validation is not None:
        bundle["geometry_candidates"]["validation_report"] = str(args.geometry_validation)
        bundle["geometry_candidates"]["selected_candidate_ids"] = geometry_validation.get("promoted_candidate_ids", [])
        bundle["geometry_candidates"]["promoted_count"] = geometry_validation.get("promoted_count", 0)
    if hand_shape is not None:
        bundle["hand_shape"]["track_solve_reports"] = [str(args.hand_shape_report)]
        bundle["hand_shape"]["accepted_shape_prior_tracks"] = [
            row.get("hand_track_id")
            for row in hand_shape.get("tracks", [])
            if isinstance(row, dict) and str(row.get("promotion_status", "")).startswith("retained_shape_posterior")
        ]
    if contact_rows is not None:
        if contact_rows.get("evidence_created") is not False:
            raise ContractError("contact_point_render_rows_must_be_render_only_evidence_created_false")
        bundle["contact_visualization"]["render_rows"] = str(args.contact_render_rows)
        bundle["contact_visualization"]["evidence_created"] = False
        bundle["contact_visualization"]["row_count"] = contact_rows.get("row_count", len(contact_rows.get("rows", [])))
    ensure_no_gt_in_prediction(bundle, "v20_observation_bundle")
    write_json(args.output, bundle)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V20 prediction-side observation bundle consumed by optimization/rendering.")
    parser.add_argument("--depth-registry", type=Path, default=None)
    parser.add_argument("--depth-selection", type=Path, default=None)
    parser.add_argument("--weak-depth-report", type=Path, action="append", default=[])
    parser.add_argument("--geometry-registry", type=Path, default=None)
    parser.add_argument("--geometry-validation", type=Path, default=None)
    parser.add_argument("--hand-shape-report", type=Path, default=None)
    parser.add_argument("--contact-render-rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(result["schema"])
