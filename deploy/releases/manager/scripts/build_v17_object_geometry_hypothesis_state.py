#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATUS = "v17_object_geometry_hypothesis_state_qc"
CLAIM = (
    "This artifact groups V17 object-geometry evidence by object id and classifies the current hypothesis state. "
    "It compares masks, visible RGBD surfaces, persistent visible-surface meshes, object-depth repair candidates, "
    "local contact patches, material-motion segments, and source-compatibility audits. It is a state-owner audit, "
    "not a mesh reconstruction solver and not object-pose closure."
)

FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


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
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def rows_by_object(rows: list[Any], *, key: str, label: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, raw in enumerate(rows):
        row = require_dict(raw, f"{label}[{i}]")
        object_id = require_str(row.get(key), f"{label}[{i}].{key}")
        out.setdefault(object_id, []).append(row)
    return out


def visible_surface_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("object_summaries"), "visible-surface object_summaries")):
        row = require_dict(raw, f"visible-surface object_summaries[{i}]")
        object_id = require_str(row.get("object_id"), f"visible-surface object_summaries[{i}].object_id")
        out[object_id] = {
            "surface_frame_count": require_int(row.get("surface_frame_count"), f"{object_id} surface_frame_count"),
            "rejected_frame_count": require_int(row.get("rejected_frame_count"), f"{object_id} rejected_frame_count"),
            "surface_vertices": require_int(row.get("surface_vertices"), f"{object_id} surface_vertices"),
            "surface_faces": require_int(row.get("surface_faces"), f"{object_id} surface_faces"),
            "object_geometry_complete": bool(row.get("object_geometry_complete") is True),
            "object_pose_requirement_met": bool(row.get("object_pose_requirement_met") is True),
        }
    return out


def geometry_audit_conflicts_by_object(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return rows_by_object(
        require_list(report.get("local_patch_visible_surface_conflicts"), "local_patch_visible_surface_conflicts"),
        key="object_id",
        label="local_patch_visible_surface_conflicts",
    )


def temporal_validation_reports_for_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    source_file = candidate.get("source_file")
    if source_file is None:
        return []
    parent = Path(require_str(source_file, "object-depth source_file")).parent
    reports: list[dict[str, Any]] = []
    for path in sorted(parent.glob("temporal_validation_*/object_depth_repair_temporal_validation_summary.json")):
        payload = require_dict(load_json(path), f"{path}")
        for row_i, raw in enumerate(require_list(payload.get("rows"), f"{path}.rows")):
            row = require_dict(raw, f"{path}.rows[{row_i}]")
            if row.get("candidate_measurement_id") != candidate.get("measurement_id"):
                continue
            reports.append(
                {
                    "path": str(path),
                    "validation": require_dict(row.get("validation"), f"{path}.rows[{row_i}].validation"),
                    "hypothesis_names": sorted(require_dict(row.get("hypotheses"), "hypotheses").keys()),
                }
            )
    return reports


def temporal_status_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        status = require_str(require_dict(report.get("validation"), "validation").get("status"), "validation.status")
        counts[status] += 1
    return dict(sorted(counts.items()))


def material_motion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "window_count": len(rows),
        "rigid_factor_ready_pair_count": sum(
            require_int(row.get("rigid_factor_ready_pairs"), "rigid_factor_ready_pairs") for row in rows
        ),
        "persistent_window_motion_candidate_count": sum(
            1 for row in rows if row.get("persistent_window_motion_candidate") is True
        ),
        "local_adjacent_material_motion_window_count": sum(
            1 for row in rows if row.get("local_adjacent_material_motion") is True
        ),
        "window_ids": [require_str(row.get("window_id"), "window_id") for row in rows],
    }


def material_pose_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_segment_count": len(rows),
        "ready_segment_count": sum(1 for row in rows if row.get("partial_material_pose_candidate") is True),
        "candidate_ids": [require_str(row.get("candidate_id"), "candidate_id") for row in rows],
        "ready_candidate_ids": [
            require_str(row.get("candidate_id"), "candidate_id")
            for row in rows
            if row.get("partial_material_pose_candidate") is True
        ],
    }


def surface_replay_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_segment_count": len(rows),
        "ready_segment_count": sum(1 for row in rows if row.get("partial_visible_surface_replay_candidate") is True),
        "candidate_ids": [require_str(row.get("candidate_id"), "candidate_id") for row in rows],
        "ready_candidate_ids": [
            require_str(row.get("candidate_id"), "candidate_id")
            for row in rows
            if row.get("partial_visible_surface_replay_candidate") is True
        ],
    }


def reconstruction_result_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted_reconstruction_result") is True]
    hidden = [row for row in rows if row.get("hidden_topology_reconstructed") is True]
    return {
        "result_count": len(rows),
        "solver_output_detected_count": sum(
            1
            for row in rows
            if require_dict(row.get("readiness_checks"), "reconstruction result readiness_checks").get(
                "solver_backend_output_detected"
            )
            is True
        ),
        "mesh_projection_qc_passed_count": sum(
            1
            for row in rows
            if require_dict(row.get("readiness_checks"), "reconstruction result readiness_checks").get(
                "mesh_projection_qc_passed"
            )
            is True
        ),
        "hidden_topology_reconstructed_count": len(hidden),
        "accepted_reconstruction_result_count": len(accepted),
        "accepted_job_ids": [require_str(row.get("job_id"), "reconstruction job_id") for row in accepted],
        "accepted_frame_ranges": [
            [
                require_int(row.get("first_frame"), "reconstruction first_frame"),
                require_int(row.get("last_frame"), "reconstruction last_frame"),
            ]
            for row in accepted
        ],
        "accepted_frame_count": sum(require_int(row.get("frame_count"), "reconstruction frame_count") for row in accepted),
        "accepted_mesh_vertices": sum(require_int(row.get("mesh_vertices"), "reconstruction mesh_vertices") for row in accepted),
        "accepted_mesh_faces": sum(require_int(row.get("mesh_faces"), "reconstruction mesh_faces") for row in accepted),
        "full_active_interval_geometry_ready_count": sum(
            1 for row in accepted if row.get("full_active_interval_geometry_ready") is True
        ),
        "contact_compatible_geometry_ready_count": sum(
            1 for row in accepted if row.get("contact_compatible_geometry_ready") is True
        ),
    }


def local_patch_summary(rows: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted_local_contact_patch_state"]
    return {
        "patch_state_count": len(rows),
        "accepted_patch_state_count": len(accepted),
        "accepted_contact_measurement_ids": [
            require_str(row.get("contact_measurement_id"), "contact_measurement_id") for row in accepted
        ],
        "visible_surface_conflict_count": sum(1 for row in conflicts if row.get("source_conflict") is True),
        "conflicts": conflicts,
    }


def object_depth_repair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    validation_statuses: Counter[str] = Counter()
    for row in rows:
        validations = temporal_validation_reports_for_candidate(row)
        validation_counts = temporal_status_counts(validations)
        validation_statuses.update(validation_counts)
        candidate_rows.append(
            {
                "measurement_id": require_str(row.get("measurement_id"), "object-depth measurement_id"),
                "frame_idx": require_int(row.get("frame_idx"), "object-depth frame_idx"),
                "validation_status": row.get("validation_status"),
                "hand_side": row.get("hand_side"),
                "vertices": require_int(row.get("vertices"), "object-depth vertices"),
                "faces": require_int(row.get("faces"), "object-depth faces"),
                "depth_scale_to_contact_anchor": row.get("depth_scale_to_contact_anchor"),
                "temporal_validation_count": len(validations),
                "temporal_validation_status_counts": validation_counts,
            }
        )
    return {
        "candidate_count": len(rows),
        "temporal_validation_count": sum(row["temporal_validation_count"] for row in candidate_rows),
        "temporal_validation_status_counts": dict(sorted(validation_statuses.items())),
        "candidates": candidate_rows,
    }


def persistent_shape_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted_persistent_visible_surface_support"]
    canonical_meshes = sorted(
        {
            require_str(row.get("canonical_mesh_npz"), "canonical_mesh_npz")
            for row in accepted
            if row.get("canonical_mesh_npz") is not None
        }
    )
    return {
        "measurement_count": len(rows),
        "accepted_measurement_count": len(accepted),
        "support_frame_indices": [require_int(row.get("frame_idx"), "persistent shape frame_idx") for row in accepted],
        "canonical_mesh_npz": canonical_meshes,
        "pose_models": sorted(
            {
                require_str(row.get("pose_model"), "pose_model")
                for row in accepted
                if row.get("pose_model") is not None
            }
        ),
    }


def status_from_hypotheses(
    *,
    persistent_shape: dict[str, Any],
    local_patch: dict[str, Any],
    object_depth_repair: dict[str, Any],
    observed_surface_seed: dict[str, Any],
    reconstruction_result: dict[str, Any],
    material_pose_replay: dict[str, Any],
    visible_surface: dict[str, Any],
) -> str:
    candidates = [
        (
            "partial_persistent_visible_surface_hypothesis",
            persistent_shape["accepted_measurement_count"] > 0,
        ),
        (
            "partial_short_segment_hidden_topology_reconstruction",
            reconstruction_result["accepted_reconstruction_result_count"] > 0,
        ),
        (
            "partial_observed_surface_geometry_seed",
            observed_surface_seed["seed_candidate_count"] > 0,
        ),
        (
            "partial_observed_surface_pose_segments",
            material_pose_replay["ready_segment_count"] > 0,
        ),
        (
            "local_contact_patch_conflicted_with_visible_surface",
            local_patch["visible_surface_conflict_count"] > 0,
        ),
        (
            "object_depth_repair_candidate_under_temporal_test",
            object_depth_repair["candidate_count"] > 0,
        ),
        (
            "visible_surface_measurements_only",
            visible_surface["surface_frame_count"] > 0,
        ),
    ]
    for status, predicate in candidates:
        if predicate:
            return status
    return "mask_only_no_geometry_hypothesis"


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    timeline_path = existing_path(
        args.multi_object_timeline_root / case / "v17_multi_object_timeline.json",
        f"{case} multi-object timeline",
    )
    visible_surface_path = existing_path(
        args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
        f"{case} visible-surface report",
    )
    geometry_audit_path = existing_path(
        args.geometry_source_audit_root / case / "v17_geometry_source_audit_report.json",
        f"{case} geometry-source audit",
    )
    motion_path = existing_path(
        args.object_material_motion_state_root / case / "v17_object_material_motion_state_report.json",
        f"{case} material-motion report",
    )
    pose_path = existing_path(
        args.object_material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json",
        f"{case} material-pose report",
    )
    replay_path = existing_path(
        args.object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
        f"{case} material-surface replay report",
    )
    observed_seed_path = existing_path(
        args.observed_surface_geometry_seed_root / case / "v17_observed_surface_geometry_seed_report.json",
        f"{case} observed-surface geometry seed report",
    )
    reconstruction_result_path = existing_path(
        args.geometry_reconstruction_results_root / case / "v17_geometry_reconstruction_results_report.json",
        f"{case} geometry reconstruction results report",
    )
    measurement_dir = args.measurement_store_root / case / "measurements_v17"
    local_patch_path = existing_path(
        measurement_dir / "local_contact_patch_state_measurements.json",
        f"{case} local contact patch states",
    )
    object_depth_path = existing_path(
        measurement_dir / "object_depth_repair_candidate_measurements.json",
        f"{case} object-depth repair candidates",
    )
    persistent_shape_path = existing_path(
        measurement_dir / "persistent_object_shape_measurements.json",
        f"{case} persistent object shape measurements",
    )

    timeline = require_dict(load_json(timeline_path), f"{case} timeline")
    visible_surface = require_dict(load_json(visible_surface_path), f"{case} visible-surface report")
    geometry_audit = require_dict(load_json(geometry_audit_path), f"{case} geometry-source audit")
    material_motion = require_dict(load_json(motion_path), f"{case} material-motion report")
    material_pose = require_dict(load_json(pose_path), f"{case} material-pose report")
    material_replay = require_dict(load_json(replay_path), f"{case} material-surface replay report")
    observed_seed = require_dict(load_json(observed_seed_path), f"{case} observed-surface geometry seed report")
    reconstruction_result = require_dict(
        load_json(reconstruction_result_path), f"{case} geometry reconstruction results report"
    )

    visible_by_object = visible_surface_by_object(visible_surface)
    local_patch_by_object = rows_by_object(load_json(local_patch_path), key="entity_id", label="local patches")
    object_depth_by_object = rows_by_object(load_json(object_depth_path), key="entity_id", label="object-depth candidates")
    persistent_shape_by_object = rows_by_object(load_json(persistent_shape_path), key="entity_id", label="persistent shapes")
    motion_by_object = rows_by_object(require_list(material_motion.get("windows"), "material-motion windows"), key="object_id", label="material-motion windows")
    pose_by_object = rows_by_object(require_list(material_pose.get("candidates"), "material-pose candidates"), key="object_id", label="material-pose candidates")
    replay_by_object = rows_by_object(require_list(material_replay.get("candidates"), "material-surface replay candidates"), key="object_id", label="material-surface replay candidates")
    observed_seed_by_object = rows_by_object(require_list(observed_seed.get("candidate_rows"), "observed-surface seed candidates"), key="object_id", label="observed-surface seed candidates")
    reconstruction_result_by_object = rows_by_object(
        require_list(reconstruction_result.get("jobs"), "geometry reconstruction result jobs"),
        key="object_id",
        label="geometry reconstruction result jobs",
    )
    conflicts_by_object = geometry_audit_conflicts_by_object(geometry_audit)

    objects: list[dict[str, Any]] = []
    for i, raw_object in enumerate(require_list(timeline.get("objects"), f"{case} timeline objects")):
        obj = require_dict(raw_object, f"{case} timeline objects[{i}]")
        object_id = require_str(obj.get("object_id"), f"{case} timeline objects[{i}].object_id")
        visible_summary = visible_by_object.get(
            object_id,
            {
                "surface_frame_count": 0,
                "rejected_frame_count": 0,
                "surface_vertices": 0,
                "surface_faces": 0,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            },
        )
        persistent_summary = persistent_shape_summary(persistent_shape_by_object.get(object_id, []))
        patch_summary = local_patch_summary(
            local_patch_by_object.get(object_id, []),
            conflicts_by_object.get(object_id, []),
        )
        depth_summary = object_depth_repair_summary(object_depth_by_object.get(object_id, []))
        motion_summary = material_motion_summary(motion_by_object.get(object_id, []))
        pose_summary = material_pose_summary(pose_by_object.get(object_id, []))
        replay_summary = surface_replay_summary(replay_by_object.get(object_id, []))
        reconstruction_summary = reconstruction_result_summary(reconstruction_result_by_object.get(object_id, []))
        observed_seed_summary = {
            "seed_candidate_count": len(observed_seed_by_object.get(object_id, [])),
            "candidate_ids": [
                require_str(row.get("candidate_id"), "observed-surface seed candidate_id")
                for row in observed_seed_by_object.get(object_id, [])
            ],
            "seed_vertices": sum(
                require_int(row.get("seed_vertices"), "observed-surface seed vertices")
                for row in observed_seed_by_object.get(object_id, [])
            ),
            "seed_faces": sum(
                require_int(row.get("seed_faces"), "observed-surface seed faces")
                for row in observed_seed_by_object.get(object_id, [])
            ),
            "complete_geometry_seed_count": sum(
                1 for row in observed_seed_by_object.get(object_id, []) if row.get("object_geometry_complete") is True
            ),
            "contact_compatible_geometry_seed_count": sum(
                1
                for row in observed_seed_by_object.get(object_id, [])
                if row.get("contact_compatible_geometry_ready") is True
            ),
            "full_active_interval_geometry_seed_count": sum(
                1
                for row in observed_seed_by_object.get(object_id, [])
                if row.get("full_active_interval_geometry_ready") is True
            ),
        }
        state = status_from_hypotheses(
            persistent_shape=persistent_summary,
            local_patch=patch_summary,
            object_depth_repair=depth_summary,
            observed_surface_seed=observed_seed_summary,
            reconstruction_result=reconstruction_summary,
            material_pose_replay=replay_summary,
            visible_surface=visible_summary,
        )
        object_row = {
            "object_id": object_id,
            "track_id": require_str(obj.get("track_id"), f"{object_id} track_id"),
            "name": require_str(obj.get("name"), f"{object_id} name"),
            "active_frame_count": require_int(obj.get("active_frame_count"), f"{object_id} active_frame_count"),
            "visible_mask_frame_count": require_int(
                obj.get("visible_mask_frame_count"), f"{object_id} visible_mask_frame_count"
            ),
            "active_without_visible_mask_frame_count": require_int(
                obj.get("active_without_visible_mask_frame_count"),
                f"{object_id} active_without_visible_mask_frame_count",
            ),
            "geometry_hypothesis_state": state,
            "visible_surface_measurement": visible_summary,
            "persistent_visible_surface_shape": persistent_summary,
            "object_depth_repair_candidates": depth_summary,
            "local_contact_patch_state": patch_summary,
            "material_motion_state": motion_summary,
            "material_pose_candidates": pose_summary,
            "material_surface_replay": replay_summary,
            "observed_surface_geometry_seed": observed_seed_summary,
            "geometry_reconstruction_result": reconstruction_summary,
            "can_own_contact_factors": False,
            "can_own_object_pose_factors": False,
            "complete_mesh_timeline_ready": False,
            **FALSE_READY,
        }
        objects.append(object_row)

    state_counts = dict(sorted(Counter(require_str(row["geometry_hypothesis_state"], "geometry state") for row in objects).items()))
    report = {
        "method": "build_v17_object_geometry_hypothesis_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "multi_object_timeline": source_summary(timeline_path, timeline),
            "multi_object_visible_surface_report": source_summary(visible_surface_path, visible_surface),
            "geometry_source_audit_report": source_summary(geometry_audit_path, geometry_audit),
            "object_material_motion_state_report": source_summary(motion_path, material_motion),
            "object_material_pose_candidate_report": source_summary(pose_path, material_pose),
            "object_material_surface_replay_report": source_summary(replay_path, material_replay),
            "observed_surface_geometry_seed_report": source_summary(observed_seed_path, observed_seed),
            "geometry_reconstruction_results_report": source_summary(reconstruction_result_path, reconstruction_result),
            "local_contact_patch_state_measurements": {"path": str(local_patch_path), "row_count": sum(len(v) for v in local_patch_by_object.values())},
            "object_depth_repair_candidate_measurements": {"path": str(object_depth_path), "row_count": sum(len(v) for v in object_depth_by_object.values())},
            "persistent_object_shape_measurements": {"path": str(persistent_shape_path), "row_count": sum(len(v) for v in persistent_shape_by_object.values())},
        },
        "frame_count": require_int(timeline.get("frame_count"), f"{case} frame_count"),
        "object_count": len(objects),
        "object_frame_rows": require_int(timeline.get("object_frame_rows"), f"{case} object_frame_rows"),
        "visible_surface_frame_rows": require_int(visible_surface.get("surface_frame_rows"), f"{case} surface_frame_rows"),
        "state_counts": state_counts,
        "objects_with_persistent_visible_surface_shape": sum(
            1 for row in objects if row["persistent_visible_surface_shape"]["accepted_measurement_count"] > 0
        ),
        "objects_with_object_depth_repair_candidates": sum(
            1 for row in objects if row["object_depth_repair_candidates"]["candidate_count"] > 0
        ),
        "objects_with_local_contact_patches": sum(
            1 for row in objects if row["local_contact_patch_state"]["accepted_patch_state_count"] > 0
        ),
        "objects_with_material_surface_replay_ready_segments": sum(
            1 for row in objects if row["material_surface_replay"]["ready_segment_count"] > 0
        ),
        "objects_with_observed_surface_geometry_seed": sum(
            1 for row in objects if row["observed_surface_geometry_seed"]["seed_candidate_count"] > 0
        ),
        "objects_with_accepted_reconstruction_results": sum(
            1 for row in objects if row["geometry_reconstruction_result"]["accepted_reconstruction_result_count"] > 0
        ),
        "observed_surface_geometry_seed_count": sum(
            row["observed_surface_geometry_seed"]["seed_candidate_count"] for row in objects
        ),
        "accepted_reconstruction_result_count": sum(
            row["geometry_reconstruction_result"]["accepted_reconstruction_result_count"] for row in objects
        ),
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        "source_incompatibility_count": require_int(
            geometry_audit.get("source_incompatibility_count"),
            f"{case} source_incompatibility_count",
        ),
        "objects": objects,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_object_geometry_hypothesis_state_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.multi_object_timeline_root / "v17_multi_object_timeline_summary.json",
        "multi-object timeline summary",
    )
    summary = require_dict(load_json(summary_path), "multi-object timeline summary")
    reports = [
        build_case(
            require_str(require_dict(raw, f"timeline summary cases[{i}]").get("case"), "timeline summary case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "timeline summary cases"))
    ]
    payload = {
        "method": "build_v17_object_geometry_hypothesis_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_multi_object_timeline_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_object_geometry_hypothesis_state_report.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "object_count": require_int(report.get("object_count"), "object_count"),
                "state_counts": require_dict(report.get("state_counts"), "state_counts"),
                "objects_with_persistent_visible_surface_shape": require_int(
                    report.get("objects_with_persistent_visible_surface_shape"),
                    "objects_with_persistent_visible_surface_shape",
                ),
                "objects_with_object_depth_repair_candidates": require_int(
                    report.get("objects_with_object_depth_repair_candidates"),
                    "objects_with_object_depth_repair_candidates",
                ),
                "objects_with_local_contact_patches": require_int(
                    report.get("objects_with_local_contact_patches"),
                    "objects_with_local_contact_patches",
                ),
                "objects_with_material_surface_replay_ready_segments": require_int(
                    report.get("objects_with_material_surface_replay_ready_segments"),
                    "objects_with_material_surface_replay_ready_segments",
                ),
                "objects_with_observed_surface_geometry_seed": require_int(
                    report.get("objects_with_observed_surface_geometry_seed"),
                    "objects_with_observed_surface_geometry_seed",
                ),
                "objects_with_accepted_reconstruction_results": require_int(
                    report.get("objects_with_accepted_reconstruction_results"),
                    "objects_with_accepted_reconstruction_results",
                ),
                "observed_surface_geometry_seed_count": require_int(
                    report.get("observed_surface_geometry_seed_count"),
                    "observed_surface_geometry_seed_count",
                ),
                "accepted_reconstruction_result_count": require_int(
                    report.get("accepted_reconstruction_result_count"),
                    "accepted_reconstruction_result_count",
                ),
                "complete_object_geometry_hypothesis_count": 0,
                "contact_compatible_object_geometry_hypothesis_count": 0,
                "object_pose_factor_ready_hypothesis_count": 0,
                "source_incompatibility_count": require_int(
                    report.get("source_incompatibility_count"),
                    "source_incompatibility_count",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "object_count": sum(require_int(report.get("object_count"), "object_count") for report in reports),
        "objects_with_observed_surface_geometry_seed": sum(
            require_int(report.get("objects_with_observed_surface_geometry_seed"), "objects_with_observed_surface_geometry_seed")
            for report in reports
        ),
        "observed_surface_geometry_seed_count": sum(
            require_int(report.get("observed_surface_geometry_seed_count"), "observed_surface_geometry_seed_count")
            for report in reports
        ),
        "objects_with_accepted_reconstruction_results": sum(
            require_int(
                report.get("objects_with_accepted_reconstruction_results"),
                "objects_with_accepted_reconstruction_results",
            )
            for report in reports
        ),
        "accepted_reconstruction_result_count": sum(
            require_int(report.get("accepted_reconstruction_result_count"), "accepted_reconstruction_result_count")
            for report in reports
        ),
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        "source_incompatibility_count": sum(
            require_int(report.get("source_incompatibility_count"), "source_incompatibility_count")
            for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_object_geometry_hypothesis_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--multi-object-timeline-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--geometry-source-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_source_audit"),
    )
    parser.add_argument(
        "--object-material-motion-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_motion_state"),
    )
    parser.add_argument(
        "--object-material-pose-candidate-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_pose_candidates"),
    )
    parser.add_argument(
        "--object-material-surface-replay-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"),
    )
    parser.add_argument(
        "--observed-surface-geometry-seed-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed"),
    )
    parser.add_argument(
        "--geometry-reconstruction-results-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
