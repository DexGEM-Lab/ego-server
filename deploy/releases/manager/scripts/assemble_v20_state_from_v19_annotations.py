#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, write_json


def frame_count_from_annotations(annotations: dict[str, Any]) -> int:
    frames = annotations.get("frames") if isinstance(annotations, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ContractError("annotations_have_no_frames")
    return len(frames)


def selected_frames_from_annotations(annotations: dict[str, Any]) -> list[int]:
    return [int(row.get("frame_idx", row.get("index", i))) for i, row in enumerate(annotations.get("frames", []))]


def copy_render_paths(render_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(render_summary, dict):
        return {}
    aliases = {
        "overlay": "overlay_video",
        "world": "world_video",
        "side_by_side": "side_by_side_video",
    }
    out = {}
    for canonical, alias in aliases.items():
        value = render_summary.get(canonical, render_summary.get(alias))
        if value is not None:
            out[canonical] = value
    for key in ("frame_count_match", "overlay_frame_count", "world_frame_count", "side_by_side_frame_count"):
        if key in render_summary:
            out[key] = render_summary.get(key)
    return out


def load_branch_optimization_reports(paths: list[Path] | None) -> list[dict[str, Any]]:
    reports = []
    for path in paths or []:
        payload = load_json(path)
        ensure_no_gt_in_prediction(payload, f"branch_optimization:{path}")
        reports.append({
            "path": str(path),
            "method": payload.get("method"),
            "status": payload.get("status"),
            "object_id": payload.get("object_id"),
            "annotation_ready": payload.get("annotation_ready"),
            "graph_frame_count": payload.get("graph_frame_count"),
            "claim_scope": payload.get("claim_scope"),
        })
    return reports


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    observation_bundle = load_json(args.observation_bundle) if args.observation_bundle else {}
    render_summary = load_json(args.render_summary) if args.render_summary else {}
    branch_optimization_reports = load_branch_optimization_reports(args.branch_optimization_report)
    if args.require_branch_optimization and not branch_optimization_reports:
        raise ContractError("missing_v20_branch_optimization_report: v20_sidecars_are_observations_not_final_optimized_state")
    dataset_manifest = load_json(args.dataset_manifest) if args.dataset_manifest else None
    ensure_no_gt_in_prediction(annotations, "annotations")
    ensure_no_gt_in_prediction(observation_bundle, "observation_bundle")
    frame_count = frame_count_from_annotations(annotations)
    timeline = {
        "frame_count": frame_count,
        "selected_frames": selected_frames_from_annotations(annotations),
    }
    if isinstance(dataset_manifest, dict):
        timeline.update({
            "source_frame_count": dataset_manifest.get("frame_count_total"),
            "fps": dataset_manifest.get("fps_assumed"),
            "resolution": dataset_manifest.get("resolution"),
        })
    physical_state = {
        "schema": "v20_physical_state.v0",
        "mode": args.mode,
        "benchmark_mode_detail": "prediction_eval_refs_sealed" if args.mode == "v20_benchmark" else None,
        "run_root": str(args.run_root.resolve()),
        "case_id": args.case_id,
        "dataset": {
            "name": dataset_manifest.get("dataset") if isinstance(dataset_manifest, dict) else None,
            "sample_id": dataset_manifest.get("sample_id") if isinstance(dataset_manifest, dict) else None,
            "evaluation_reference_loaded": False,
            "evaluation_reference_policy": "Eval refs are absent from prediction state; evaluation reads them separately after render.",
        } if args.mode == "v20_benchmark" else None,
        "timeline": timeline,
        "camera": {
            "state": "from_renderable_annotations_or_dataset_manifest",
            "coordinate_convention": dataset_manifest.get("coordinate_convention") if isinstance(dataset_manifest, dict) else None,
            "required_for_metric_claims": True,
        },
        "depth": observation_bundle.get("depth", {}),
        "geometry_candidates": observation_bundle.get("geometry_candidates", {}),
        "branch_optimization": {
            "required": bool(args.require_branch_optimization),
            "reports": branch_optimization_reports,
            "policy": "V20 observations are inputs to V19-style branch optimization/factor correction; sidecars alone must not close final renderable physical state.",
        },
        "hand_shape": observation_bundle.get("hand_shape", {}),
        "contact_visualization": observation_bundle.get("contact_visualization", {}),
        "occlusions": [],
        "nonpenetration": [],
        "renderer_boundary": "renders consume state/annotations_v20_renderable.json plus declared V20 sidecars only",
        "render_inputs": {
            "annotations": str(args.output_annotations),
            "observation_bundle": str(args.observation_bundle) if args.observation_bundle else None,
            "render_summary": str(args.render_summary) if args.render_summary else None,
        },
        "renders": copy_render_paths(render_summary),
    }
    uncertainty_state = {
        "schema": "v20_uncertainty_state.v0",
        "mode": args.mode,
        "benchmark_mode_detail": "prediction_eval_refs_sealed" if args.mode == "v20_benchmark" else None,
        "depth": "see v20 observation bundle selector residuals",
        "geometry": "see v20 geometry validation/promotion residuals",
        "hand_shape": "track-level posterior is a prior, not exact hand state",
        "contact_visualization": "render-only; evidence_created=false",
        "evaluation_reference_policy": "Eval refs not loaded in prediction state assembly",
    }
    annotations_out = dict(annotations)
    annotations_out["schema"] = annotations_out.get("schema", "annotations_v20_renderable.v0")
    annotations_out["v20_sidecars"] = {
        "observation_bundle": str(args.observation_bundle) if args.observation_bundle else None,
        "physical_state": str(args.output_state),
        "uncertainty_state": str(args.output_uncertainty),
    }
    ensure_no_gt_in_prediction(physical_state, "physical_state")
    ensure_no_gt_in_prediction(uncertainty_state, "uncertainty_state")
    ensure_no_gt_in_prediction(annotations_out, "annotations_v20_renderable")
    write_json(args.output_annotations, annotations_out)
    write_json(args.output_state, physical_state)
    write_json(args.output_uncertainty, uncertainty_state)
    return physical_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble V20 prediction state from V18/V19-compatible prediction-side annotations and V20 sidecars.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--observation-bundle", type=Path, default=None)
    parser.add_argument("--render-summary", type=Path, default=None)
    parser.add_argument("--branch-optimization-report", type=Path, action="append", default=[])
    parser.add_argument("--require-branch-optimization", action="store_true")
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("v20_infer", "v20_benchmark"), default="v20_infer")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-uncertainty", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = assemble(parse_args())
    print(result["schema"])
