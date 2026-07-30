#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_runtime_manifest_scaffold"
CLAIM = (
    "This V18 artifact defines the bounded default DAG and records input/runtime contracts before any heavy "
    "perception work runs. It is a budget gate and implementation scaffold, not an annotation result."
)
FORBIDDEN_DEFAULT_TERMS = ("bundlesdf", "nerf", "neural_field_training", "sdf_training", "test_time_training")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def dag_template() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": "frame_decode_cache",
            "parallel_group": 0,
            "max_realtime_factor": 0.25,
            "default_path": True,
            "stage_kind": "io_cache",
            "forbidden_heavy_backend": False,
            "outputs": ["source_frame_manifest", "raw_video_contract"],
        },
        {
            "stage_id": "object_plan_and_physical_state",
            "parallel_group": 1,
            "max_realtime_factor": 0.50,
            "default_path": True,
            "stage_kind": "vlm_or_open_vocab_planning",
            "forbidden_heavy_backend": False,
            "outputs": ["object_roster", "model_produced_physical_state_type", "active_intervals"],
        },
        {
            "stage_id": "hand_detection_and_mano_measurements",
            "parallel_group": 1,
            "max_realtime_factor": 2.00,
            "default_path": True,
            "stage_kind": "feed_forward_hand_models",
            "forbidden_heavy_backend": False,
            "outputs": ["hand_boxes_keypoints", "visible_mano_measurements"],
        },
        {
            "stage_id": "metric_depth_and_camera",
            "parallel_group": 1,
            "max_realtime_factor": 2.00,
            "default_path": True,
            "stage_kind": "feed_forward_depth_plus_fast_camera",
            "forbidden_heavy_backend": False,
            "outputs": ["metric_depth", "depth_edge_bands", "T_world_camera"],
        },
        {
            "stage_id": "object_detection_segmentation_tracking",
            "parallel_group": 1,
            "max_realtime_factor": 2.00,
            "default_path": True,
            "stage_kind": "open_vocab_detection_plus_sam2_tracking",
            "forbidden_heavy_backend": False,
            "outputs": ["object_masks", "object_identity_tracks"],
        },
        {
            "stage_id": "visible_surface_and_fast_motion_state",
            "parallel_group": 2,
            "max_realtime_factor": 2.00,
            "default_path": True,
            "stage_kind": "bounded_depth_surface_and_track_residuals",
            "forbidden_heavy_backend": False,
            "outputs": ["visible_surfaces", "rigid_or_deformable_residuals", "unresolved_hidden_geometry"],
        },
        {
            "stage_id": "visibility_occlusion_state",
            "parallel_group": 3,
            "max_realtime_factor": 0.50,
            "default_path": True,
            "stage_kind": "single_writer_state_reducer",
            "forbidden_heavy_backend": False,
            "outputs": ["per_frame_visibility", "occluder_owner_or_unresolved", "uncertainty"],
        },
        {
            "stage_id": "bounded_consistency_graph",
            "parallel_group": 4,
            "max_realtime_factor": 1.00,
            "default_path": True,
            "stage_kind": "fixed_iteration_robust_graph",
            "forbidden_heavy_backend": False,
            "outputs": ["consistent_hand_object_state", "occlusion_depth_order_candidate_evidence", "rejected_factors", "unresolved_rows"],
        },
        {
            "stage_id": "full_duration_render",
            "parallel_group": 5,
            "max_realtime_factor": 2.00,
            "default_path": True,
            "stage_kind": "chunkable_renderer",
            "forbidden_heavy_backend": False,
            "outputs": ["overlay_video", "world_video", "side_by_side_video"],
        },
    ]


def critical_path_factor(stages: list[dict[str, Any]]) -> float:
    groups: dict[int, float] = {}
    for stage in stages:
        group = require_int(stage.get("parallel_group"), "parallel_group")
        factor = finite_float(stage.get("max_realtime_factor"), "max_realtime_factor")
        groups[group] = max(groups.get(group, 0.0), factor)
    return float(sum(groups[group] for group in sorted(groups)))


def forbidden_stage_findings(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for stage in stages:
        text = json.dumps(stage, sort_keys=True).lower().replace("-", "_")
        hits = [term for term in FORBIDDEN_DEFAULT_TERMS if term in text]
        if hits or stage.get("forbidden_heavy_backend") is True:
            findings.append({"stage_id": stage.get("stage_id"), "forbidden_terms": hits})
    return findings


def artifact_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}


def case_manifest(case: str, args: argparse.Namespace, stages: list[dict[str, Any]]) -> dict[str, Any]:
    v16_manifest_path = existing(args.v16_root / case / "v16_full_pipeline_manifest.json", f"{case} V16 manifest")
    v16 = require_dict(load_json(v16_manifest_path), f"{case} V16 manifest")
    raw = require_dict(v16.get("raw_video"), f"{case} raw_video")
    frame_count = require_int(raw.get("frame_count"), f"{case} frame_count")
    fps = finite_float(raw.get("fps"), f"{case} fps")
    duration_s = frame_count / fps
    hard_budget_s = duration_s * float(args.hard_realtime_factor)
    planned_factor = critical_path_factor(stages)
    planned_budget_s = duration_s * planned_factor
    if planned_factor > float(args.hard_realtime_factor):
        raise RuntimeError(
            f"V18 DAG planned critical path {planned_factor:.2f}x exceeds hard budget "
            f"{args.hard_realtime_factor:.2f}x"
        )
    measurement_manifest = args.measurement_store_root / case / "v17_measurement_manifest.json"
    object_timeline = args.multi_object_timeline_root / case / "v17_multi_object_timeline.json"
    visible_surface = args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json"
    pairwise_contact = args.pairwise_contact_root / case / "v17_pairwise_contact_state.json"
    interior_hand = args.interior_hand_graph_root / case / "v17_interior_owned_full_residual_hand_graph.json"
    sources = {
        "v16_manifest": artifact_status(v16_manifest_path),
        "v17_measurement_manifest": artifact_status(measurement_manifest),
        "v17_multi_object_timeline": artifact_status(object_timeline),
        "v17_visible_surface_report": artifact_status(visible_surface),
        "v17_pairwise_contact_state": artifact_status(pairwise_contact),
        "v17_interior_owned_hand_graph": artifact_status(interior_hand),
        "wilor_raw": artifact_status(Path(v16.get("hand_summary", {}).get("raw_path")) if isinstance(v16.get("hand_summary"), dict) and v16.get("hand_summary", {}).get("raw_path") else None),
    }
    missing_required = [key for key, value in sources.items() if key != "wilor_raw" and not value["exists"]]
    prior_v16_elapsed = v16.get("elapsed_s")
    manifest = {
        "method": "build_v18_runtime_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "raw_video": raw,
        "raw_frame_count": frame_count,
        "raw_fps": fps,
        "raw_duration_s": duration_s,
        "hard_realtime_factor": float(args.hard_realtime_factor),
        "hard_budget_s": hard_budget_s,
        "planned_default_dag_critical_path_factor": planned_factor,
        "planned_default_dag_critical_path_budget_s": planned_budget_s,
        "planned_default_dag_within_hard_budget": planned_budget_s <= hard_budget_s,
        "default_path_forbidden_backend_findings": forbidden_stage_findings(stages),
        "default_path_uses_bundlesdf_or_nerf": False,
        "runtime_contract_ready": True,
        "existing_source_artifacts": sources,
        "missing_required_source_artifacts": missing_required,
        "source_artifact_contract_ready": not missing_required,
        "prior_v16_elapsed_s_not_v18_budget_evidence": prior_v16_elapsed,
        "dag_stages": [
            {
                **stage,
                "budget_s_for_case": duration_s * finite_float(stage.get("max_realtime_factor"), "stage factor"),
            }
            for stage in stages
        ],
        "parallelism_semantics": {
            "groups_are_parallel_barriers": True,
            "critical_path_factor_is_sum_of_max_group_factors": True,
            "workers_write_isolated_artifacts": True,
            "reducers_are_single_writer": True,
        },
        **FALSE_READY,
    }
    if manifest["default_path_forbidden_backend_findings"]:
        raise RuntimeError(f"forbidden default backend appeared in V18 DAG: {manifest['default_path_forbidden_backend_findings']}")
    write_json(args.output_root / case / "v18_runtime_manifest.json", manifest)
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    stages = dag_template()
    cases = [case_manifest(case, args, stages) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_runtime_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "build_elapsed_s": elapsed,
        "hard_realtime_factor": float(args.hard_realtime_factor),
        "planned_default_dag_critical_path_factor": critical_path_factor(stages),
        "default_path_uses_bundlesdf_or_nerf": False,
        "default_path_forbidden_backend_findings": forbidden_stage_findings(stages),
        "runtime_contract_ready": all(case["runtime_contract_ready"] for case in cases),
        "source_artifact_contract_ready": all(case["source_artifact_contract_ready"] for case in cases),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "manifest_path": str(args.output_root / require_str(case.get("case"), "case") / "v18_runtime_manifest.json"),
                "raw_frame_count": case["raw_frame_count"],
                "raw_duration_s": case["raw_duration_s"],
                "hard_budget_s": case["hard_budget_s"],
                "planned_default_dag_critical_path_budget_s": case["planned_default_dag_critical_path_budget_s"],
                "source_artifact_contract_ready": case["source_artifact_contract_ready"],
                "missing_required_source_artifacts": case["missing_required_source_artifacts"],
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_runtime_manifest_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--multi-object-timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--visible-surface-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"))
    parser.add_argument("--pairwise-contact-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"))
    parser.add_argument("--interior-hand-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_runtime_manifest"))
    parser.add_argument("--hard-realtime-factor", type=float, default=10.0)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
