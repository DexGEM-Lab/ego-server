#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import prepare_v20_benchmark_dataset as v20_prepare
from v20_common import ContractError, append_jsonl, load_json, write_json


BOTTLENECK_FAMILIES = ["depth_camera", "segmentation", "hand_mano"]


def update_common_manifest_fields(payload: dict[str, Any], schema: str, mode: str) -> dict[str, Any]:
    out = dict(payload)
    out["v20_adapter_source_schema"] = payload.get("schema")
    out["schema"] = schema
    out["mode"] = mode
    out["created_by"] = "scripts/prepare_v21_benchmark_dataset.py"
    out["v21_policy"] = {
        "prediction_state_boundary": "state/v21_physical_state.json",
        "gt_reference_policy": "evaluation_only_after_prediction_state_and_renders_exist",
        "public_object_roster_policy": "model_library_only_not_target_object_plan",
        "monocular_baseline_required_for_depth_or_assisted_segmentation": True,
        "bottleneck_strong_tuning_required_before_downweighting": BOTTLENECK_FAMILIES,
        "rigid_object_closure_required": "mesh_completion_or_adaptation_then_pose_fit_then_factor_correction_then_mesh_pose_render",
    }
    return out


def build_initial_state(run_root: Path, input_manifest: dict[str, Any], dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "v21_physical_state.v0",
        "mode": "v21_benchmark",
        "status": "benchmark_input_prepared_prediction_measurements_pending",
        "run_root": str(run_root.resolve()),
        "dataset": dataset_manifest.get("dataset"),
        "sample_id": dataset_manifest.get("sample_id"),
        "timeline": {
            "frame_count": input_manifest.get("frame_count"),
            "fps": input_manifest.get("fps"),
            "resolution": input_manifest.get("resolution"),
            "selected_frames": input_manifest.get("selected_frames"),
        },
        "camera": {
            "state": "dataset_prediction_intrinsics_recorded_not_yet_validated_by_v21_depth_camera_gate",
            "intrinsics_source": "input/dataset_manifest.json",
            "required_for_metric_claims": True,
        },
        "depth": {
            "state": "dataset_prediction_depth_recorded_not_yet_compared_to_monocular_baseline",
            "native_or_dataset_depth_available": bool(dataset_manifest.get("frames")),
            "monocular_baseline_required": True,
            "primary_metric_depth_selected": False,
        },
        "hands": [],
        "objects": [],
        "contacts": [],
        "occlusions": [],
        "nonpenetration": [],
        "target_object_plan": {
            "state": "unwritten",
            "public_object_roster_is_model_library_only": True,
        },
        "tuning": {
            "bottleneck_observation_policy": "strong_tune_before_downweight",
            "required_families": BOTTLENECK_FAMILIES,
            "attempt_roots": {
                "depth_camera": "tuning/depth_camera",
                "segmentation": "tuning/segmentation",
                "hand_mano": "tuning/hand_mano",
            },
        },
        "pending_measurement_and_optimization_stages": [
            "monocular_depth_camera_baseline_not_run",
            "dataset_depth_camera_not_compared_to_monocular_baseline",
            "target_object_plan_not_created_from_visual_evidence",
            "rgb_only_segmentation_baseline_not_run",
            "segmentation_contamination_review_not_run",
            "hand_mano_candidates_not_run_or_validated",
            "rigid_object_mesh_completion_pose_graph_not_run",
            "contact_occlusion_nonpenetration_not_supported_until_metric_hand_object_state_exists",
            "v21_render_not_produced",
            "gt_evaluation_not_allowed_until_prediction_state_and_render_exist",
        ],
        "renderer_boundary": "renders must consume V21 state/ and annotations_v21_renderable.json only",
    }


def build_uncertainty_state(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "v21_uncertainty_state.v0",
        "mode": "v21_benchmark",
        "status": "benchmark_input_prepared_prediction_uncertain",
        "dataset": dataset_manifest.get("dataset"),
        "sample_id": dataset_manifest.get("sample_id"),
        "camera_depth": "dataset calibration/depth recorded as prediction input, but V21 still requires monocular baseline comparison and registration/scale validation",
        "segmentation": "target object plan and RGB-only/assisted segmentation comparison not yet run",
        "hand_mano": "hand candidate streams and active MANO pose/shape/scale optimization not yet run",
        "object_geometry_pose": "public object/CAD roster is model library only; no target mesh pose state exists yet",
        "contact_occlusion_nonpenetration": "unsupported until metric MANO and object mesh pose exist",
        "gt_reference_policy": "GT reference files are sealed under evaluation/reference_manifest.json until prediction state and renders exist",
    }


def write_v21_support_files(run_root: Path, started: float) -> dict[str, Any]:
    input_manifest_path = run_root / "input" / "input_manifest.json"
    dataset_manifest_path = run_root / "input" / "dataset_manifest.json"
    reference_manifest_path = run_root / "evaluation" / "reference_manifest.json"
    input_manifest = update_common_manifest_fields(load_json(input_manifest_path), "v21_input_manifest.v0", "v21_benchmark")
    dataset_manifest = update_common_manifest_fields(load_json(dataset_manifest_path), "v21_benchmark_dataset_manifest.v0", "v21_benchmark")
    reference_manifest = update_common_manifest_fields(load_json(reference_manifest_path), "v21_benchmark_evaluation_reference_manifest.v0", "v21_benchmark")
    input_manifest["v21_state_paths"] = {
        "physical_state": "state/v21_physical_state.json",
        "uncertainty_state": "state/v21_uncertainty_state.json",
        "agent_evidence": "state/v21_agent_evidence.md",
        "observation_bundle": "state/v21_observation_bundle.json",
    }
    input_manifest["next_required_step"] = "run_v21_prediction_measurement_tuning_optimization_render_before_reference_evaluation"
    dataset_manifest["monocular_baseline_contract"] = {
        "required": True,
        "applies_to": ["dataset_native_depth", "rgbd", "stereo", "multiview", "depth_assisted_segmentation"],
        "comparison_report": "measurements/camera_depth/monocular_vs_assisted_depth_report.json",
    }
    dataset_manifest["segmentation_contract"] = {
        "target_source": "agent_visual_or_task_evidence_object_plan_not_public_object_roster",
        "rgb_only_baseline_required": True,
        "contamination_review_required": True,
    }
    dataset_manifest["hand_mano_contract"] = {
        "metric_mano_required_for_contact_claims": True,
        "active_shape_scale_optimization_required_for_exact_hand_shape_claims": True,
    }
    reference_manifest["evaluation_reference_policy"] = "evaluation_only_after_v21_prediction_state_and_renders_exist_forbidden_for_prediction_tuning_state_render_or_candidate_generation"
    write_json(input_manifest_path, input_manifest)
    write_json(dataset_manifest_path, dataset_manifest)
    write_json(reference_manifest_path, reference_manifest)
    physical_state = build_initial_state(run_root, input_manifest, dataset_manifest)
    uncertainty_state = build_uncertainty_state(dataset_manifest)
    observation_bundle = {
        "schema": "v21_observation_bundle.v0",
        "mode": "v21_benchmark",
        "status": "input_prepared_measurements_pending",
        "dataset_manifest": "input/dataset_manifest.json",
        "monocular_baseline_required": True,
        "bottleneck_tuning_required_before_downweighting": BOTTLENECK_FAMILIES,
        "accepted_measurements": [],
        "pending_measurements": physical_state["pending_measurement_and_optimization_stages"],
    }
    write_json(run_root / "state" / "v21_physical_state.json", physical_state)
    write_json(run_root / "state" / "v21_uncertainty_state.json", uncertainty_state)
    write_json(run_root / "state" / "v21_observation_bundle.json", observation_bundle)
    write_json(run_root / "state" / "annotations_v21_renderable.json", {"schema": "v21_renderable_annotations.v0", "status": "not_created_prediction_measurements_pending", "frames": []})
    evidence = "\n".join([
        "# V21 Agent Evidence",
        "",
        "Observation: Benchmark input manifests were prepared with GT references sealed under `evaluation/reference_manifest.json`.",
        "Observation: Public object rosters in `input/dataset_manifest.json` are model libraries only, not target object plans.",
        "Commitment: Next physical progress requires monocular depth/camera baseline, bottleneck tuning records when observations are weak, target segmentation review, metric MANO, rigid object mesh-pose optimization, and render from V21 state.",
        "",
    ])
    (run_root / "state" / "v21_agent_evidence.md").write_text(evidence, encoding="utf-8")
    summary = {
        "schema": "v21_benchmark_prepare_summary.v0",
        "status": "ok",
        "mode": "v21_benchmark",
        "dataset": dataset_manifest.get("dataset"),
        "sample_id": dataset_manifest.get("sample_id"),
        "run_root": str(run_root.resolve()),
        "frame_count": input_manifest.get("frame_count"),
        "prediction_manifest": str(dataset_manifest_path),
        "evaluation_reference_manifest": str(reference_manifest_path),
        "physical_state": str(run_root / "state" / "v21_physical_state.json"),
        "elapsed_s": float(time.time() - started),
        "claim_scope": "V21 benchmark input/state bootstrap only; no physical annotation measurement, optimization, render, or GT evaluation has run. The V21 state and policy fields written here are not consumed by a V21 physical mechanism until downstream V21 measurement/optimization/render adapters are implemented.",
        "next_required_step": "run_v21_prediction_measurement_tuning_optimization_render_before_reference_evaluation",
    }
    write_json(run_root / "run_summary.json", summary)
    append_jsonl(run_root / "logs" / "harness_events.jsonl", {"event": "v21_benchmark_input_prepared", **summary})
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    v20_prepare.run(args)
    return write_v21_support_files(args.run_root, started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V21 benchmark manifests and unresolved V21 state with strict eval-ref isolation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
