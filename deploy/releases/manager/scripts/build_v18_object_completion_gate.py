#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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

STATUS = "v18_object_completion_gate"
CLAIM = (
    "This artifact gates any future V18 object completion/pose step. It identifies bounded candidates and "
    "blocks unsuitable objects from rigid-pose completion; it does not run completion or mark any object pose complete."
)


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


def visible_geometry_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("object_id"), "geometry object_id"): row
        for row in [require_dict(raw, "geometry object row") for raw in require_list(report.get("object_rows"), "geometry object rows")]
    }


def fast_motion_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("object_id"), "motion object_id"): row
        for row in [require_dict(raw, "motion object row") for raw in require_list(report.get("object_rows"), "motion object rows")]
    }


def classify_gate(geometry: dict[str, Any], motion: dict[str, Any], physical_schema: dict[str, Any]) -> tuple[str, str, list[str], list[str]]:
    physical = str(
        physical_schema.get("model_physical_state_type")
        or motion.get("model_physical_state_type", geometry.get("model_physical_state_type", "unknown"))
    )
    fast_motion = str(motion.get("fast_motion_state", geometry.get("fast_motion_state", "motion_unresolved_no_surface")))
    requires_part_motion = bool(physical_schema.get("requires_part_or_relative_motion_model"))
    surface_frames = require_int(geometry.get("surface_frame_count", 0), "surface_frame_count")
    rejected_frames = require_int(geometry.get("rejected_visible_frame_count", 0), "rejected_visible_frame_count")
    blockers: list[str] = []
    next_evidence: list[str] = []
    if surface_frames <= 0:
        blockers.append("no_accepted_depth_backed_visible_surface")
        if rejected_frames > 0:
            blockers.append("visible_masks_failed_surface_acceptance")
        next_evidence.extend(["recover reliable metric depth for visible masks", "rerun visible-surface extraction under bounded thresholds"])
        return "blocked_no_visible_surface", "completion_not_allowed", blockers, next_evidence
    if requires_part_motion:
        blockers.append("structured_schema_requires_part_or_relative_motion_model")
        next_evidence.extend(["part-level object split", "articulation/relative-motion model", "part-wise visible geometry support"])
        return "part_motion_requires_part_split_no_single_rigid_completion", "candidate_requires_part_model_not_run", blockers, next_evidence
    if physical == "deformable":
        blockers.append("model_physical_state_deformable")
        next_evidence.extend(["deformable visible-surface tracking", "bounded nonrigid surface model or explicit hidden-geometry prior"])
        return "deformable_visible_surface_only_no_rigid_pose", "rigid_completion_not_allowed", blockers, next_evidence
    if physical == "articulated":
        blockers.append("model_physical_state_articulated")
        next_evidence.extend(["part-level articulation model", "joint/hinge evidence", "part-wise visible geometry support"])
        return "articulated_requires_part_model_no_single_pose", "single_rigid_completion_not_allowed", blockers, next_evidence
    if fast_motion == "partial_rigid_visible_surface_motion_supported":
        next_evidence.extend([
            "bounded feed-forward object prior or observed multi-view fusion",
            "canonical visible-surface fusion under time cap",
            "metric depth validation for candidate pose",
            "contact ownership validation after geometry support",
        ])
        return "bounded_rigid_completion_candidate_visible_surface_only", "candidate_not_run", blockers, next_evidence
    if fast_motion == "local_rigid_motion_only_not_pose":
        blockers.append("local_rigid_motion_does_not_compose_to_pose")
        next_evidence.extend(["persistent rigid residual support across a window", "bounded canonical visible-surface fusion"])
        return "blocked_local_motion_only_not_pose", "completion_deferred", blockers, next_evidence
    if physical == "rigid":
        blockers.append("rigid_prior_without_persistent_motion_or_completion_evidence")
        next_evidence.extend(["persistent surface-track residual check", "bounded feed-forward/observed geometry prior"])
        return "rigid_prior_visible_surface_but_pose_unresolved", "completion_deferred", blockers, next_evidence
    blockers.append("physical_state_or_motion_unresolved")
    next_evidence.extend(["model-produced physical state refinement", "accepted visible surface and motion residual support"])
    return "blocked_unknown_or_unresolved_physical_state", "completion_deferred", blockers, next_evidence


def depth_fused_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = report.get("object_rows") if isinstance(report.get("object_rows"), list) else []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        object_id = raw.get("object_id")
        mesh = raw.get("mesh_reconstruction") if isinstance(raw.get("mesh_reconstruction"), dict) else {}
        if isinstance(object_id, str) and (mesh.get("poisson_mesh_path") or mesh.get("convex_hull_mesh_path")):
            out[object_id] = raw
    return out


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    geometry_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    motion_path = args.fast_motion_root / case / "v18_fast_motion_state_report.json"
    physical_schema_path = args.physical_state_schema_root / case / "v18_physical_state_schema_report.json"
    depth_fused_path = args.depth_fused_root / case / "v18_depth_fused_reconstruction_report.json"
    geometry_report = require_dict(load_json(geometry_path), f"{case} visible geometry report")
    motion_report = require_dict(load_json(motion_path), f"{case} fast motion report")
    physical_schema_report = require_dict(load_json(physical_schema_path), f"{case} physical state schema")
    depth_fused_report = require_dict(load_json(depth_fused_path), f"{case} depth fused reconstruction report") if depth_fused_path.exists() else {"object_rows": []}
    physical_schema_index = {
        require_str(row.get("object_id"), "physical schema object_id"): row
        for row in [require_dict(raw, "physical schema object row") for raw in require_list(physical_schema_report.get("object_rows"), "physical schema object rows")]
    }
    geometry_index = visible_geometry_by_object(geometry_report)
    motion_index = fast_motion_by_object(motion_report)
    depth_fused_index = depth_fused_by_object(depth_fused_report)
    object_ids = sorted(set(geometry_index) | set(motion_index) | set(physical_schema_index) | set(depth_fused_index))
    rows: list[dict[str, Any]] = []
    gate_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for object_id in object_ids:
        geometry = geometry_index.get(object_id, {})
        motion = motion_index.get(object_id, {})
        physical_schema = physical_schema_index.get(object_id, {})
        depth_fused = depth_fused_index.get(object_id, {})
        depth_mesh = depth_fused.get("mesh_reconstruction") if isinstance(depth_fused.get("mesh_reconstruction"), dict) else {}
        depth_fused_mesh_ready = bool(depth_mesh.get("poisson_mesh_path") or depth_mesh.get("convex_hull_mesh_path"))
        gate_state, action, blockers, next_evidence = classify_gate(geometry, motion, physical_schema)
        gate_counts[gate_state] += 1
        action_counts[action] += 1
        rows.append(
            {
                "object_id": object_id,
                "track_id": motion.get("track_id", geometry.get("track_id")),
                "name": motion.get("name", geometry.get("name")),
                "model_physical_state_type": physical_schema.get("model_physical_state_type", motion.get("model_physical_state_type", geometry.get("model_physical_state_type"))),
                "physical_state_source": physical_schema.get("physical_state_source"),
                "requires_part_or_relative_motion_model": bool(physical_schema.get("requires_part_or_relative_motion_model")),
                "part_or_relative_motion_evidence_terms": physical_schema.get("part_or_relative_motion_evidence_terms", []),
                "secondary_deformable_or_surface_component": bool(physical_schema.get("secondary_deformable_or_surface_component")),
                "legacy_keyword_physical_state_type": physical_schema.get("legacy_keyword_physical_state_type"),
                "fast_motion_state": motion.get("fast_motion_state", geometry.get("fast_motion_state")),
                "visible_geometry_status": geometry.get("v18_visible_geometry_status"),
                "surface_frame_count": geometry.get("surface_frame_count", 0),
                "rejected_visible_frame_count": geometry.get("rejected_visible_frame_count", 0),
                "surface_vertex_count": geometry.get("surface_vertex_count", 0),
                "surface_face_count": geometry.get("surface_face_count", 0),
                "completion_gate_state": gate_state,
                "completion_action": action,
                "blockers": blockers,
                "required_next_evidence": next_evidence,
                "completion_run": depth_fused_mesh_ready,
                "hidden_geometry_reconstructed": depth_fused_mesh_ready,
                "canonical_mesh_ready": depth_fused_mesh_ready,
                "depth_fused_reconstruction_report": str(depth_fused_path) if depth_fused else None,
                "depth_fused_source_frame_count": depth_fused.get("source_frame_count"),
                "depth_fused_sampled_point_count": depth_fused.get("sampled_point_count"),
                "depth_fused_poisson_mesh_path": depth_mesh.get("poisson_mesh_path"),
                "depth_fused_convex_hull_mesh_path": depth_mesh.get("convex_hull_mesh_path"),
                "depth_fused_mesh_status": depth_mesh.get("status"),
                "complete_object_pose_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_object_completion_gate",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_visible_geometry_archive": str(geometry_path), "v18_fast_motion_state": str(motion_path), "v18_physical_state_schema": str(physical_schema_path), "v18_depth_fused_reconstruction": str(depth_fused_path)},
        "object_count": len(rows),
        "completion_gate_state_counts": dict(sorted(gate_counts.items())),
        "completion_action_counts": dict(sorted(action_counts.items())),
        "completion_candidate_count": gate_counts.get("bounded_rigid_completion_candidate_visible_surface_only", 0),
        "part_split_candidate_count": gate_counts.get("part_motion_requires_part_split_no_single_rigid_completion", 0),
        "completion_run_count": sum(1 for row in rows if row.get("completion_run") is True),
        "hidden_geometry_reconstructed_count": sum(1 for row in rows if row.get("hidden_geometry_reconstructed") is True),
        "canonical_mesh_ready_count": sum(1 for row in rows if row.get("canonical_mesh_ready") is True),
        "complete_object_pose_ready_count": 0,
        "object_rows": rows,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_object_completion_gate_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    gate_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for report in reports:
        gate_counts.update(report["completion_gate_state_counts"])
        action_counts.update(report["completion_action_counts"])
    summary = {
        "method": "build_v18_object_completion_gate",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count": sum(require_int(report.get("object_count"), "object_count") for report in reports),
        "completion_gate_state_counts": dict(sorted(gate_counts.items())),
        "completion_action_counts": dict(sorted(action_counts.items())),
        "completion_candidate_count": gate_counts.get("bounded_rigid_completion_candidate_visible_surface_only", 0),
        "part_split_candidate_count": gate_counts.get("part_motion_requires_part_split_no_single_rigid_completion", 0),
        "completion_run_count": sum(require_int(report.get("completion_run_count"), "completion_run_count") for report in reports),
        "hidden_geometry_reconstructed_count": sum(require_int(report.get("hidden_geometry_reconstructed_count"), "hidden_geometry_reconstructed_count") for report in reports),
        "canonical_mesh_ready_count": sum(require_int(report.get("canonical_mesh_ready_count"), "canonical_mesh_ready_count") for report in reports),
        "complete_object_pose_ready_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_object_completion_gate_report.json"),
                "object_count": report["object_count"],
                "completion_gate_state_counts": report["completion_gate_state_counts"],
                "completion_candidate_count": report["completion_candidate_count"],
                "part_split_candidate_count": report["part_split_candidate_count"],
                "completion_run_count": report["completion_run_count"],
                "hidden_geometry_reconstructed_count": report["hidden_geometry_reconstructed_count"],
                "canonical_mesh_ready_count": report["canonical_mesh_ready_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_object_completion_gate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--fast-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_fast_motion_state"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--depth-fused-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_object_completion_gate"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
