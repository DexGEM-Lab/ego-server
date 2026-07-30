#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATUS = "v17_object_geometry_factor_problem_qc"
CLAIM = (
    "This artifact converts current V17 per-object evidence into an object-centric geometry factor problem. "
    "It defines the canonical-geometry, per-frame pose, material-correspondence, visible-surface, and contact "
    "factor blocks that a real object solver must own. It is a problem materialization, not an optimizer."
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


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if abs(out) < float("inf") else None


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def rows_by_object(rows: list[Any], *, key: str, label: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, raw in enumerate(rows):
        row = require_dict(raw, f"{label}[{i}]")
        object_id = require_str(row.get(key), f"{label}[{i}].{key}")
        out.setdefault(object_id, []).append(row)
    return out


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(float(v) for v in values if finite_float(v) is not None)
    if not vals:
        return {"count": 0}
    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac
    return {
        "count": len(vals),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "min": vals[0],
        "max": vals[-1],
    }


def finite_values(values: list[Any]) -> list[float]:
    return [value for value in (finite_float(raw) for raw in values) if value is not None]


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def visible_surface_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("object_summaries"), "visible-surface object_summaries")):
        row = require_dict(raw, f"visible-surface object_summaries[{i}]")
        object_id = require_str(row.get("object_id"), f"visible-surface object_summaries[{i}].object_id")
        out[object_id] = row
    return out


def load_case_inputs(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "object_geometry_hypothesis_state": existing_path(
            args.object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
            f"{case} object-geometry hypothesis-state report",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "material_track": existing_path(
            args.object_material_track_root / case / "v17_object_material_track_summary.json",
            f"{case} material-track summary",
        ),
        "material_motion": existing_path(
            args.object_material_motion_state_root / case / "v17_object_material_motion_state_report.json",
            f"{case} material-motion report",
        ),
        "material_pose": existing_path(
            args.object_material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json",
            f"{case} material-pose report",
        ),
        "material_surface_replay": existing_path(
            args.object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
            f"{case} material-surface replay report",
        ),
        "observed_surface_geometry_seed": existing_path(
            args.observed_surface_geometry_seed_root / case / "v17_observed_surface_geometry_seed_report.json",
            f"{case} observed-surface geometry seed report",
        ),
        "geometry_reconstruction_jobs": existing_path(
            args.geometry_reconstruction_jobs_root / case / "v17_geometry_reconstruction_jobs_report.json",
            f"{case} geometry reconstruction jobs report",
        ),
        "geometry_reconstruction_results": existing_path(
            args.geometry_reconstruction_results_root / case / "v17_geometry_reconstruction_results_report.json",
            f"{case} geometry reconstruction results report",
        ),
        "multi_object_contact_evidence": existing_path(
            args.multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
            f"{case} multi-object contact evidence report",
        ),
        "contact_ownership_problem": existing_path(
            args.contact_ownership_problem_root / case / "v17_contact_ownership_problem.json",
            f"{case} contact-ownership problem",
        ),
        "pairwise_contact_depth_gap": existing_path(
            args.pairwise_contact_depth_gap_root / case / "v17_pairwise_contact_depth_gap.json",
            f"{case} pairwise contact depth-gap report",
        ),
        "geometry_source_audit": existing_path(
            args.geometry_source_audit_root / case / "v17_geometry_source_audit_report.json",
            f"{case} geometry-source audit report",
        ),
        "depth_contact_consistency_audit": existing_path(
            args.depth_contact_consistency_audit_root / case / "v17_depth_contact_consistency_audit_report.json",
            f"{case} depth-contact consistency audit report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    return {"paths": paths, "payloads": payloads}


def material_track_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        frames = [require_int(frame, "material-track frame") for frame in require_list(row.get("frames"), "frames")]
        out.append(
            {
                "window_id": require_str(row.get("window_id"), "window_id"),
                "report_path": require_str(row.get("report_path"), "report_path"),
                "rigid_pair_report_path": row.get("rigid_pair_report_path"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "query_points": require_int(row.get("query_points"), "query_points"),
                "all_frame_accepted_tracks": require_int(
                    row.get("all_frame_accepted_tracks"),
                    "all_frame_accepted_tracks",
                ),
                "rigid_factor_ready_pairs": require_int(
                    row.get("rigid_factor_ready_pairs"),
                    "rigid_factor_ready_pairs",
                ),
                "rigid_motion_evidence_ready": bool(row.get("rigid_motion_evidence_ready") is True),
                "ready_pair_inlier_residual_m": row.get("ready_pair_inlier_residual_m"),
            }
        )
    return out


def motion_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        segments = []
        for raw in require_list(row.get("segments"), "material-motion segments"):
            segment = require_dict(raw, "material-motion segment")
            segments.append(
                {
                    "start_frame": require_int(segment.get("start_frame"), "segment start_frame"),
                    "end_frame": require_int(segment.get("end_frame"), "segment end_frame"),
                    "pair_count": require_int(segment.get("pair_count"), "segment pair_count"),
                    "ready": bool(segment.get("persistent_window_motion_candidate") is True)
                    or bool(segment.get("window_rigid_motion_candidate") is True),
                    "readiness_checks": segment.get("readiness_checks"),
                }
            )
        out.append(
            {
                "window_id": require_str(row.get("window_id"), "window_id"),
                "object_id": require_str(row.get("object_id"), "object_id"),
                "rigid_factor_ready_pairs": require_int(
                    row.get("rigid_factor_ready_pairs"),
                    "rigid_factor_ready_pairs",
                ),
                "local_adjacent_material_motion": bool(row.get("local_adjacent_material_motion") is True),
                "persistent_window_motion_candidate": bool(row.get("persistent_window_motion_candidate") is True),
                "ready_segment_count": require_int(row.get("ready_segment_count"), "ready_segment_count"),
                "candidate_segment_count": require_int(row.get("candidate_segment_count"), "candidate_segment_count"),
                "max_ready_segment_pairs": require_int(row.get("max_ready_segment_pairs"), "max_ready_segment_pairs"),
                "segments": segments,
            }
        )
    return out


def pose_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), "candidate_id"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "start_frame": require_int(row.get("start_frame"), "start_frame"),
                "end_frame": require_int(row.get("end_frame"), "end_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "track_count": require_int(row.get("track_count"), "track_count"),
                "pair_count": require_int(row.get("pair_count"), "pair_count"),
                "partial_material_pose_candidate": bool(row.get("partial_material_pose_candidate") is True),
                "residual_m": row.get("residual_m"),
                "readiness_checks": row.get("readiness_checks"),
            }
        )
    return out


def replay_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), "candidate_id"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "start_frame": require_int(row.get("start_frame"), "start_frame"),
                "end_frame": require_int(row.get("end_frame"), "end_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "partial_visible_surface_replay_candidate": bool(
                    row.get("partial_visible_surface_replay_candidate") is True
                ),
                "surface_replay_m": row.get("surface_replay_m"),
                "readiness_checks": row.get("readiness_checks"),
            }
        )
    return out


def observed_surface_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), "candidate_id"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "archive_path": require_str(row.get("archive_path"), "archive_path"),
                "start_frame": require_int(row.get("start_frame"), "start_frame"),
                "end_frame": require_int(row.get("end_frame"), "end_frame"),
                "seed_frame_count": require_int(row.get("seed_frame_count"), "seed_frame_count"),
                "seed_vertices": require_int(row.get("seed_vertices"), "seed_vertices"),
                "seed_faces": require_int(row.get("seed_faces"), "seed_faces"),
                "observed_surface_only": bool(row.get("observed_surface_only") is True),
                "hidden_topology_reconstructed": bool(row.get("hidden_topology_reconstructed") is True),
                "full_active_interval_geometry_ready": bool(row.get("full_active_interval_geometry_ready") is True),
                "contact_compatible_geometry_ready": bool(row.get("contact_compatible_geometry_ready") is True),
                "canonical_extent_m": row.get("canonical_extent_m"),
                "canonical_centroid_delta_from_source_m": row.get("canonical_centroid_delta_from_source_m"),
            }
        )
    return out


def reconstruction_jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "job_id": require_str(row.get("job_id"), "job_id"),
                "job_path": require_str(row.get("job_path"), "job_path"),
                "dataset_dir": require_str(row.get("dataset_dir"), "dataset_dir"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "first_frame": require_int(row.get("first_frame"), "first_frame"),
                "last_frame": require_int(row.get("last_frame"), "last_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "solver_job_ready": bool(row.get("solver_job_ready") is True),
                "hidden_topology_reconstructed": bool(row.get("hidden_topology_reconstructed") is True),
                "rectification_nearest_3d_residual_p95_m": row.get("rectification_nearest_3d_residual_p95_m"),
                "projected_inside_fraction": row.get("projected_inside_fraction"),
                "source_intrinsics": row.get("source_intrinsics"),
                "rectified_intrinsics_fx_fy_cx_cy": row.get("rectified_intrinsics_fx_fy_cx_cy"),
            }
        )
    return out


def reconstruction_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        checks = require_dict(row.get("readiness_checks"), "reconstruction result readiness_checks")
        out.append(
            {
                "job_id": require_str(row.get("job_id"), "job_id"),
                "status": require_str(row.get("status"), "status"),
                "bundlesdf_output_dir": require_str(row.get("bundlesdf_output_dir"), "bundlesdf_output_dir"),
                "window_id": require_str(row.get("window_id"), "window_id"),
                "first_frame": require_int(row.get("first_frame"), "first_frame"),
                "last_frame": require_int(row.get("last_frame"), "last_frame"),
                "frame_count": require_int(row.get("frame_count"), "frame_count"),
                "solver_job_ready": bool(row.get("solver_job_ready") is True),
                "solver_backend_output_detected": bool(checks.get("solver_backend_output_detected") is True),
                "mesh_file_detected": bool(checks.get("mesh_file_detected") is True),
                "pose_sequence_complete": bool(checks.get("pose_sequence_complete") is True),
                "mesh_scale_plausible_against_rectified_rgbd": bool(
                    checks.get("mesh_scale_plausible_against_rectified_rgbd") is True
                ),
                "mesh_surface_topology_plausible": bool(checks.get("mesh_surface_topology_plausible") is True),
                "mesh_projection_qc_passed": bool(checks.get("mesh_projection_qc_passed") is True),
                "hidden_topology_reconstructed": bool(row.get("hidden_topology_reconstructed") is True),
                "accepted_reconstruction_result": bool(row.get("accepted_reconstruction_result") is True),
                "mesh_path": row.get("mesh_path"),
                "mesh_vertices": row.get("mesh_vertices"),
                "mesh_faces": row.get("mesh_faces"),
                "mesh_extent_m": row.get("mesh_extent_m"),
                "mesh_scale_acceptance_range_m": row.get("mesh_scale_acceptance_range_m"),
                "projection_qc": row.get("projection_qc"),
                **FALSE_READY,
            }
        )
    return out


def depth_contact_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hand_rows = [
        require_dict(hand, "depth-contact hand row")
        for row in rows
        for hand in require_list(row.get("hand_rows"), "depth-contact hand rows")
    ]
    return {
        "evaluated_frame_count": len(rows),
        "evaluated_hand_rows": len(hand_rows),
        "near_reconstructed_mesh_hand_rows": sum(
            require_int(row.get("near_reconstructed_mesh_hand_rows"), "near reconstructed mesh hand rows")
            for row in rows
        ),
        "reconstructed_mesh_contact_candidate_rows": sum(
            require_int(
                row.get("reconstructed_mesh_contact_candidate_rows"),
                "reconstructed mesh contact candidate rows",
            )
            for row in rows
        ),
        "legacy_contact_ready_hand_rows": sum(
            require_int(row.get("legacy_contact_ready_hand_rows"), "legacy contact ready hand rows") for row in rows
        ),
        "multi_object_reconstructed_object_contact_candidate_rows": sum(
            require_int(
                row.get("multi_object_reconstructed_object_contact_candidate_rows"),
                "multi-object reconstructed object contact candidate rows",
            )
            for row in rows
        ),
        "legacy_owner_mismatch_frame_count": sum(
            1
            for row in rows
            if require_dict(row.get("object_owner_state"), "object owner state").get(
                "legacy_single_object_matches_reconstructed_object"
            )
            is False
        ),
        "shared_depth_state_ready_frame_count": sum(
            1 for row in rows if row.get("shared_depth_state_ready") is True
        ),
        "depth_owner_incompatibility_count": sum(
            1
            for row in rows
            if require_dict(row.get("same_depth_state_checks"), "same_depth_state_checks").get(
                "legacy_object_depth_matches_visible_unidepth"
            )
            is False
            or require_dict(row.get("same_depth_state_checks"), "same_depth_state_checks").get(
                "all_hand_depths_match_visible_unidepth"
            )
            is False
        ),
        "visible_unidepth_m": summarize(
            finite_values(
                [require_dict(row.get("visible_object_unidepth_m"), "visible depth").get("median") for row in rows]
            )
        ),
        "reconstructed_mesh_camera_depth_m": summarize(
            finite_values(
                [
                    require_dict(row.get("reconstructed_mesh_camera_depth_m"), "mesh camera depth").get("median")
                    for row in rows
                ]
            )
        ),
        "reconstructed_mesh_front_surface_depth_abs_p95_m": summarize(
            finite_values(
                [
                    require_dict(row.get("reconstructed_mesh_front_surface_depth_abs_m"), "front surface depth").get(
                        "p95"
                    )
                    for row in rows
                ]
            )
        ),
        "legacy_object_center_depth_m": summarize(
            finite_values([row.get("legacy_object_center_depth_m") for row in rows])
        ),
        "hand_source_depth_m": summarize(
            finite_values(
                [require_dict(hand.get("source_depth_m"), "hand source depth").get("median") for hand in hand_rows]
            )
        ),
        "reconstructed_mesh_to_hand_min_m": summarize(
            finite_values(
                [
                    require_dict(hand.get("reconstructed_mesh_distance_m"), "reconstructed mesh distance").get(
                        "min_symmetric"
                    )
                    for hand in hand_rows
                ]
            )
        ),
        "rows": rows,
    }


def contact_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("contact_mode_state") == "measured_distance_evidence"]
    min_distances = [
        float(row["min_symmetric_distance_m"])
        for row in measured
        if finite_float(row.get("min_symmetric_distance_m")) is not None
    ]
    return {
        "hand_object_rows": len(rows),
        "measured_distance_rows": len(measured),
        "unobserved_rows": sum(1 for row in rows if row.get("contact_mode_state") == "unobserved"),
        "visible_surface_distance_candidate_rows": sum(
            1 for row in rows if row.get("visible_surface_distance_candidate") is True
        ),
        "contact_distance_candidate_rows": sum(1 for row in rows if row.get("contact_distance_candidate") is True),
        "contact_factor_ready_rows": sum(1 for row in rows if row.get("contact_factor_ready") is True),
        "min_symmetric_distance_m": summarize(min_distances),
        "missing_geometry_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in require_list(row.get("missing_geometry", []), "missing_geometry")
                ).items()
            )
        ),
    }


def contact_ownership_by_object(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, raw_variable in enumerate(require_list(report.get("problem_variables"), "contact owner variables")):
        variable = require_dict(raw_variable, f"contact owner variables[{i}]")
        owner_variable_id = require_str(variable.get("owner_variable_id"), "owner_variable_id")
        frame_idx = require_int(variable.get("frame_idx"), "contact owner frame_idx")
        hand_side = require_str(variable.get("hand_side"), "contact owner hand_side")
        owner_state = require_str(variable.get("owner_variable_state"), "owner_variable_state")
        selected_state = require_str(
            variable.get("selected_measurement_candidate_state"),
            "selected_measurement_candidate_state",
        )
        for j, raw_candidate in enumerate(require_list(variable.get("candidate_objects"), "candidate_objects")):
            candidate = require_dict(raw_candidate, f"contact owner variables[{i}].candidate_objects[{j}]")
            object_id = require_str(candidate.get("object_id"), "contact owner candidate object_id")
            out.setdefault(object_id, []).append(
                {
                    "owner_variable_id": owner_variable_id,
                    "frame_idx": frame_idx,
                    "hand_side": hand_side,
                    "owner_variable_state": owner_state,
                    "selected_measurement_candidate_state": selected_state,
                    "candidate_evidence_state": require_str(
                        candidate.get("owner_evidence_state"),
                        "owner_evidence_state",
                    ),
                    "selected_measurement_supports_candidate": bool(
                        candidate.get("selected_measurement_supports_candidate") is True
                    ),
                    "owner_supported_by_current_evidence": bool(
                        candidate.get("owner_supported_by_current_evidence") is True
                    ),
                    "owner_geometrically_supported": bool(candidate.get("owner_geometrically_supported") is True),
                    "owner_image_supported": bool(candidate.get("owner_image_supported") is True),
                    "owner_metric_depth_supported": bool(candidate.get("owner_metric_depth_supported") is True),
                    "contact_owner_factor_ready": bool(candidate.get("contact_owner_factor_ready") is True),
                    "multi_object_visible_surface": require_dict(
                        candidate.get("multi_object_visible_surface"),
                        "multi_object_visible_surface",
                    ),
                    "pairwise_image_contact": require_dict(
                        candidate.get("pairwise_image_contact"),
                        "pairwise_image_contact",
                    ),
                    "pairwise_metric_depth": require_dict(
                        candidate.get("pairwise_metric_depth"),
                        "pairwise_metric_depth",
                    ),
                    "accepted_reconstruction_contact": require_dict(
                        candidate.get("accepted_reconstruction_contact"),
                        "accepted_reconstruction_contact",
                    ),
                }
            )
    return out


def contact_ownership_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variable_ids = {require_str(row.get("owner_variable_id"), "owner_variable_id") for row in rows}
    return {
        "owner_variable_rows": len(variable_ids),
        "candidate_rows": len(rows),
        "supported_candidate_rows": sum(
            1 for row in rows if row.get("owner_supported_by_current_evidence") is True
        ),
        "geometrically_supported_candidate_rows": sum(
            1 for row in rows if row.get("owner_geometrically_supported") is True
        ),
        "image_supported_candidate_rows": sum(1 for row in rows if row.get("owner_image_supported") is True),
        "metric_depth_supported_candidate_rows": sum(
            1 for row in rows if row.get("owner_metric_depth_supported") is True
        ),
        "contact_owner_factor_ready_rows": sum(1 for row in rows if row.get("contact_owner_factor_ready") is True),
        "owner_variable_state_counts": dict(
            sorted(Counter(require_str(row.get("owner_variable_state"), "owner_variable_state") for row in rows).items())
        ),
        "candidate_evidence_state_counts": dict(
            sorted(
                Counter(
                    require_str(row.get("candidate_evidence_state"), "candidate_evidence_state")
                    for row in rows
                ).items()
            )
        ),
        "candidate_rows_preview": rows[:20],
        "candidate_rows_preview_limit": 20,
        "candidate_rows_preview_truncated": len(rows) > 20,
    }


def factor_blocks(
    obj: dict[str, Any],
    *,
    track_windows: list[dict[str, Any]],
    motion: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    replays: list[dict[str, Any]],
    observed_seeds: list[dict[str, Any]],
    reconstruction_job_rows: list[dict[str, Any]],
    reconstruction_result_rows: list[dict[str, Any]],
    depth_contact: dict[str, Any],
    contacts: dict[str, Any],
    contact_ownership: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    depth_candidates = require_dict(obj.get("object_depth_repair_candidates"), "object_depth_repair_candidates")
    return [
        {
            "factor_block": "visible_surface_depth_and_mask",
            "source": "multi_object_visible_surfaces",
            "frame_rows": require_int(visible.get("surface_frame_count"), "surface_frame_count"),
            "rejected_frame_rows": require_int(visible.get("rejected_frame_count"), "rejected_frame_count"),
            "surface_vertices": require_int(visible.get("surface_vertices"), "surface_vertices"),
            "surface_faces": require_int(visible.get("surface_faces"), "surface_faces"),
            "solver_role": "observed surface and silhouette residuals for canonical geometry and per-frame pose",
        },
        {
            "factor_block": "persistent_visible_surface_shape",
            "source": "persistent_object_shape_measurements",
            "accepted_measurement_count": require_int(
                persistent.get("accepted_measurement_count"),
                "accepted_measurement_count",
            ),
            "canonical_mesh_npz": require_list(persistent.get("canonical_mesh_npz"), "canonical_mesh_npz"),
            "pose_models": require_list(persistent.get("pose_models"), "pose_models"),
            "solver_role": "canonical visible-surface geometry seed; hidden topology remains unresolved",
        },
        {
            "factor_block": "material_correspondence_rigidity",
            "source": "object_material_tracks",
            "window_count": len(track_windows),
            "rigid_factor_ready_pair_count": sum(
                require_int(row.get("rigid_factor_ready_pairs"), "rigid_factor_ready_pairs")
                for row in track_windows
            ),
            "windows": track_windows,
            "solver_role": "pairwise material-point SE(3) and rigidity residuals",
        },
        {
            "factor_block": "material_motion_segments",
            "source": "object_material_motion_state",
            "window_count": len(motion),
            "persistent_window_motion_candidate_count": sum(
                1 for row in motion if row.get("persistent_window_motion_candidate") is True
            ),
            "local_adjacent_material_motion_window_count": sum(
                1 for row in motion if row.get("local_adjacent_material_motion") is True
            ),
            "windows": motion,
            "solver_role": "temporal grouping for object pose variables",
        },
        {
            "factor_block": "partial_material_pose_segments",
            "source": "object_material_pose_candidates",
            "candidate_segment_count": len(poses),
            "ready_segment_count": sum(1 for row in poses if row.get("partial_material_pose_candidate") is True),
            "candidates": poses,
            "solver_role": "observed material-point pose priors for short segments",
        },
        {
            "factor_block": "visible_surface_replay_segments",
            "source": "object_material_surface_replay",
            "candidate_segment_count": len(replays),
            "ready_segment_count": sum(
                1 for row in replays if row.get("partial_visible_surface_replay_candidate") is True
            ),
            "candidates": replays,
            "solver_role": "surface replay constraints for short material-pose segments",
        },
        {
            "factor_block": "observed_surface_geometry_seed",
            "source": "observed_surface_geometry_seed",
            "seed_candidate_count": len(observed_seeds),
            "observed_surface_only_seed_count": sum(
                1 for row in observed_seeds if row.get("observed_surface_only") is True
            ),
            "complete_geometry_seed_count": sum(
                1 for row in observed_seeds if row.get("hidden_topology_reconstructed") is True
            ),
            "contact_compatible_geometry_seed_count": sum(
                1 for row in observed_seeds if row.get("contact_compatible_geometry_ready") is True
            ),
            "full_active_interval_geometry_seed_count": sum(
                1 for row in observed_seeds if row.get("full_active_interval_geometry_ready") is True
            ),
            "seed_candidates": observed_seeds,
            "solver_role": "canonical observed-surface geometry seed for object mesh or SDF optimization",
        },
        {
            "factor_block": "unknown_object_rgbd_reconstruction_jobs",
            "source": "geometry_reconstruction_jobs",
            "job_count": len(reconstruction_job_rows),
            "solver_job_ready_count": sum(1 for row in reconstruction_job_rows if row.get("solver_job_ready") is True),
            "hidden_topology_reconstructed_job_count": sum(
                1 for row in reconstruction_job_rows if row.get("hidden_topology_reconstructed") is True
            ),
            "jobs": reconstruction_job_rows,
            "solver_role": "constant-intrinsics RGBD job inputs for a hidden-topology reconstruction backend",
        },
        {
            "factor_block": "unknown_object_rgbd_reconstruction_results",
            "source": "geometry_reconstruction_results",
            "job_count": len(reconstruction_result_rows),
            "pending_solver_output_count": sum(
                1 for row in reconstruction_result_rows if row.get("status") == "pending_solver_output"
            ),
            "solver_output_detected_count": sum(
                1 for row in reconstruction_result_rows if row.get("solver_backend_output_detected") is True
            ),
            "mesh_file_detected_count": sum(
                1 for row in reconstruction_result_rows if row.get("mesh_file_detected") is True
            ),
            "pose_sequence_complete_count": sum(
                1 for row in reconstruction_result_rows if row.get("pose_sequence_complete") is True
            ),
            "mesh_scale_plausible_count": sum(
                1
                for row in reconstruction_result_rows
                if row.get("mesh_scale_plausible_against_rectified_rgbd") is True
            ),
            "mesh_projection_qc_passed_count": sum(
                1 for row in reconstruction_result_rows if row.get("mesh_projection_qc_passed") is True
            ),
            "hidden_topology_reconstructed_job_count": sum(
                1 for row in reconstruction_result_rows if row.get("hidden_topology_reconstructed") is True
            ),
            "accepted_reconstruction_result_count": sum(
                1 for row in reconstruction_result_rows if row.get("accepted_reconstruction_result") is True
            ),
            "results": reconstruction_result_rows,
            "solver_role": "accept-or-reject QC for hidden-topology reconstruction backend outputs",
        },
        {
            "factor_block": "depth_contact_consistency",
            "source": "depth_contact_consistency_audit",
            **depth_contact,
            "solver_role": "checks whether reconstructed object geometry, visible depth, hand geometry, and contact modes share one metric state",
        },
        {
            "factor_block": "multi_object_hand_contact_distance",
            "source": "multi_object_contact_evidence",
            **contacts,
            "solver_role": "hand-object distance and contact-mode residuals against unified object geometry",
        },
        {
            "factor_block": "object_contact_ownership",
            "source": "contact_ownership_problem",
            **contact_ownership,
            "solver_role": "discrete ownership factors that attach contact-mode hand-side states to explicit object ids",
        },
        {
            "factor_block": "object_depth_repair_temporal_validation",
            "source": "object_depth_repair_candidate_measurements",
            "candidate_count": require_int(depth_candidates.get("candidate_count"), "candidate_count"),
            "temporal_validation_status_counts": require_dict(
                depth_candidates.get("temporal_validation_status_counts"),
                "temporal_validation_status_counts",
            ),
            "solver_role": "candidate depth repairs that need temporal support before entering geometry factors",
        },
        {
            "factor_block": "geometry_source_compatibility",
            "source": "geometry_source_audit",
            "source_conflict_count": len(conflicts),
            "conflicts": conflicts,
            "solver_role": "prevents local patches or legacy single-stream geometry from silently owning object pose",
        },
    ]


def variable_blocks(obj: dict[str, Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_id = require_str(obj.get("object_id"), "object_id")
    active_frames = require_int(obj.get("active_frame_count"), f"{object_id} active_frame_count")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    canonical_meshes = require_list(persistent.get("canonical_mesh_npz"), "canonical_mesh_npz")
    material_pose_ready = sum(
        require_int(block.get("ready_segment_count"), "ready_segment_count")
        for block in blocks
        if block.get("factor_block") == "partial_material_pose_segments"
    )
    observed_seed_count = sum(
        require_int(block.get("seed_candidate_count"), "seed_candidate_count")
        for block in blocks
        if block.get("factor_block") == "observed_surface_geometry_seed"
    )
    geometry_seed = "persistent_visible_surface_mesh" if canonical_meshes else "visible_surface_samples_only"
    if observed_seed_count:
        geometry_seed = "observed_surface_geometry_seed"
    orientation_observable = not any("orientation_unobservable" in str(model) for model in persistent.get("pose_models", []))
    return [
        {
            "variable_block": "canonical_object_geometry",
            "variable_type": "mesh_or_sdf_field",
            "object_id": object_id,
            "current_seed": geometry_seed if require_int(visible.get("surface_frame_count"), "surface_frame_count") else "none",
            "canonical_mesh_npz": canonical_meshes,
            "instantiated_by_current_v17_solver": False,
            "complete_hidden_topology_ready": False,
        },
        {
            "variable_block": "object_pose_per_active_frame",
            "variable_type": "SE3_per_frame",
            "object_id": object_id,
            "active_frame_count": active_frames,
            "parameter_count": active_frames * 6,
            "orientation_observable_from_current_seed": bool(orientation_observable and material_pose_ready > 0),
            "instantiated_by_current_v17_solver": False,
        },
        {
            "variable_block": "object_deformation_or_topology_delta",
            "variable_type": "geometry_delta_field",
            "object_id": object_id,
            "reason_required": "current evidence can contain partial visible surfaces, hand occlusion, and local patches without complete hidden topology",
            "instantiated_by_current_v17_solver": False,
        },
        {
            "variable_block": "contact_attachment_per_hand_object_frame",
            "variable_type": "contact_mode_and_patch_identity",
            "object_id": object_id,
            "instantiated_by_current_v17_solver": False,
            "requires_unified_geometry_source": True,
        },
    ]


def readiness_checks(
    obj: dict[str, Any],
    contacts: dict[str, Any],
    contact_ownership: dict[str, Any],
    conflicts: list[dict[str, Any]],
    reconstruction_result_rows: list[dict[str, Any]],
    depth_contact: dict[str, Any],
) -> dict[str, bool]:
    visible = require_dict(obj.get("visible_surface_measurement"), "visible_surface_measurement")
    persistent = require_dict(obj.get("persistent_visible_surface_shape"), "persistent_visible_surface_shape")
    material_motion = require_dict(obj.get("material_motion_state"), "material_motion_state")
    material_pose = require_dict(obj.get("material_pose_candidates"), "material_pose_candidates")
    material_replay = require_dict(obj.get("material_surface_replay"), "material_surface_replay")
    observed_seed = require_dict(obj.get("observed_surface_geometry_seed"), "observed_surface_geometry_seed")
    active = require_int(obj.get("active_frame_count"), "active_frame_count")
    visible_masks = require_int(obj.get("visible_mask_frame_count"), "visible_mask_frame_count")
    visible_surfaces = require_int(visible.get("surface_frame_count"), "surface_frame_count")
    pose_models = [str(model) for model in require_list(persistent.get("pose_models"), "pose_models")]
    return {
        "mask_evidence_available": visible_masks > 0,
        "visible_surface_evidence_available": visible_surfaces > 0,
        "full_active_visible_surface_coverage": active > 0 and visible_surfaces == active,
        "canonical_geometry_seed_available": require_int(
            persistent.get("accepted_measurement_count"),
            "accepted_measurement_count",
        )
        > 0,
        "hidden_topology_reconstructed": any(
            row.get("hidden_topology_reconstructed") is True for row in reconstruction_result_rows
        ),
        "orientation_observable": not any("orientation_unobservable" in model for model in pose_models),
        "persistent_material_motion_available": require_int(
            material_motion.get("persistent_window_motion_candidate_count"),
            "persistent_window_motion_candidate_count",
        )
        > 0,
        "partial_material_pose_segments_available": require_int(
            material_pose.get("ready_segment_count"),
            "ready_segment_count",
        )
        > 0,
        "partial_visible_surface_replay_available": require_int(
            material_replay.get("ready_segment_count"),
            "ready_segment_count",
        )
        > 0,
        "observed_surface_geometry_seed_available": require_int(
            observed_seed.get("seed_candidate_count"),
            "seed_candidate_count",
        )
        > 0,
        "contact_distance_candidates_available": require_int(
            contacts.get("contact_distance_candidate_rows"),
            "contact_distance_candidate_rows",
        )
        > 0,
        "contact_factor_ready_against_multi_object_geometry": require_int(
            contacts.get("contact_factor_ready_rows"),
            "contact_factor_ready_rows",
        )
        > 0,
        "contact_owner_factor_ready_against_object_geometry": require_int(
            contact_ownership.get("contact_owner_factor_ready_rows"),
            "contact_owner_factor_ready_rows",
        )
        > 0,
        "shared_depth_contact_state_available": bool(
            require_int(
                depth_contact.get("shared_depth_state_ready_frame_count"),
                "shared_depth_state_ready_frame_count",
            )
            > 0
            and require_int(
                depth_contact.get("depth_owner_incompatibility_count"),
                "depth_owner_incompatibility_count",
            )
            == 0
            and require_int(
                depth_contact.get("reconstructed_mesh_contact_candidate_rows"),
                "reconstructed_mesh_contact_candidate_rows",
            )
            > 0
            and require_int(
                depth_contact.get("legacy_owner_mismatch_frame_count"),
                "legacy_owner_mismatch_frame_count",
            )
            == 0
        ),
        "source_compatible_with_visible_surface_geometry": len(conflicts) == 0,
    }


def blocked_reasons(checks: dict[str, bool]) -> list[str]:
    messages = {
        "visible_surface_evidence_available": "no visible RGBD surface measurements",
        "full_active_visible_surface_coverage": "visible surfaces do not cover every active object frame",
        "canonical_geometry_seed_available": "no canonical object geometry seed",
        "hidden_topology_reconstructed": "hidden topology is not reconstructed",
        "orientation_observable": "orientation is not observable from current geometry seed",
        "contact_factor_ready_against_multi_object_geometry": "no contact factors are ready against multi-object geometry",
        "contact_owner_factor_ready_against_object_geometry": "no contact-owner factors are ready against object geometry",
        "shared_depth_contact_state_available": "no shared depth/contact state links accepted reconstruction meshes to current hand geometry",
        "source_compatible_with_visible_surface_geometry": "local patch, legacy geometry, or legacy contact ownership conflicts with visible-surface geometry",
    }
    return [message for key, message in messages.items() if checks.get(key) is not True]


def build_object_row(
    obj: dict[str, Any],
    *,
    track_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]],
    pose_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    observed_seed_rows: list[dict[str, Any]],
    reconstruction_job_rows: list[dict[str, Any]],
    reconstruction_result_rows: list[dict[str, Any]],
    depth_contact_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    contact_ownership_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks = material_track_windows(track_rows)
    motion = motion_windows(motion_rows)
    poses = pose_candidates(pose_rows)
    replays = replay_candidates(replay_rows)
    observed_seeds = observed_surface_seeds(observed_seed_rows)
    reconstruction_rows = reconstruction_jobs(reconstruction_job_rows)
    reconstruction_result_payloads = reconstruction_results(reconstruction_result_rows)
    depth_contact_payload = depth_contact_consistency(depth_contact_rows)
    contacts = contact_summary(contact_rows)
    contact_ownership = contact_ownership_summary(contact_ownership_rows)
    blocks = factor_blocks(
        obj,
        track_windows=tracks,
        motion=motion,
        poses=poses,
        replays=replays,
        observed_seeds=observed_seeds,
        reconstruction_job_rows=reconstruction_rows,
        reconstruction_result_rows=reconstruction_result_payloads,
        depth_contact=depth_contact_payload,
        contacts=contacts,
        contact_ownership=contact_ownership,
        conflicts=conflict_rows,
    )
    checks = readiness_checks(
        obj,
        contacts,
        contact_ownership,
        conflict_rows,
        reconstruction_result_payloads,
        depth_contact_payload,
    )
    solve_activation_ready = all(
        checks[key]
        for key in (
            "visible_surface_evidence_available",
            "full_active_visible_surface_coverage",
            "canonical_geometry_seed_available",
            "hidden_topology_reconstructed",
            "orientation_observable",
            "shared_depth_contact_state_available",
            "contact_factor_ready_against_multi_object_geometry",
            "contact_owner_factor_ready_against_object_geometry",
            "source_compatible_with_visible_surface_geometry",
        )
    )
    return {
        "object_id": require_str(obj.get("object_id"), "object_id"),
        "track_id": require_str(obj.get("track_id"), "track_id"),
        "name": require_str(obj.get("name"), "name"),
        "geometry_hypothesis_state": require_str(obj.get("geometry_hypothesis_state"), "geometry_hypothesis_state"),
        "active_frame_count": require_int(obj.get("active_frame_count"), "active_frame_count"),
        "visible_mask_frame_count": require_int(obj.get("visible_mask_frame_count"), "visible_mask_frame_count"),
        "variable_blocks": variable_blocks(obj, blocks),
        "factor_blocks": blocks,
        "readiness_checks": checks,
        "blocked_reasons": blocked_reasons(checks),
        "object_geometry_factor_problem_materialized": True,
        "solve_activation_ready": bool(solve_activation_ready),
        "can_own_contact_factors": False,
        "can_own_object_pose_factors": False,
        "complete_mesh_timeline_ready": False,
        **FALSE_READY,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_case_inputs(case, args)
    paths: dict[str, Path] = loaded["paths"]
    payloads: dict[str, dict[str, Any]] = loaded["payloads"]

    hypothesis = payloads["object_geometry_hypothesis_state"]
    visible_surface = payloads["visible_surface"]
    material_track = payloads["material_track"]
    material_motion = payloads["material_motion"]
    material_pose = payloads["material_pose"]
    material_replay = payloads["material_surface_replay"]
    observed_seed = payloads["observed_surface_geometry_seed"]
    reconstruction_jobs_report = payloads["geometry_reconstruction_jobs"]
    reconstruction_results_report = payloads["geometry_reconstruction_results"]
    depth_contact = payloads["depth_contact_consistency_audit"]
    contact = payloads["multi_object_contact_evidence"]
    contact_ownership = payloads["contact_ownership_problem"]
    pairwise_depth_gap = payloads["pairwise_contact_depth_gap"]
    audit = payloads["geometry_source_audit"]

    objects = [require_dict(row, f"{case} objects[{i}]") for i, row in enumerate(require_list(hypothesis.get("objects"), "objects"))]
    track_by_object = rows_by_object(require_list(material_track.get("windows"), "material-track windows"), key="object_id", label="material-track windows")
    motion_by_object = rows_by_object(require_list(material_motion.get("windows"), "material-motion windows"), key="object_id", label="material-motion windows")
    pose_by_object = rows_by_object(require_list(material_pose.get("candidates"), "material-pose candidates"), key="object_id", label="material-pose candidates")
    replay_by_object = rows_by_object(require_list(material_replay.get("candidates"), "material-surface replay candidates"), key="object_id", label="material-surface replay candidates")
    observed_seed_by_object = rows_by_object(require_list(observed_seed.get("candidate_rows"), "observed-surface seed candidates"), key="object_id", label="observed-surface seed candidates")
    reconstruction_jobs_by_object = rows_by_object(require_list(reconstruction_jobs_report.get("jobs"), "geometry reconstruction jobs"), key="object_id", label="geometry reconstruction jobs")
    reconstruction_results_by_object = rows_by_object(
        require_list(reconstruction_results_report.get("jobs"), "geometry reconstruction results"),
        key="object_id",
        label="geometry reconstruction results",
    )
    depth_contact_by_object = rows_by_object(
        require_list(depth_contact.get("rows"), "depth-contact rows"),
        key="object_id",
        label="depth-contact rows",
    )
    contact_by_object = rows_by_object(require_list(contact.get("rows"), "multi-object contact rows"), key="object_id", label="multi-object contact rows")
    contact_ownership_by_obj = contact_ownership_by_object(contact_ownership)
    conflict_by_object = rows_by_object(require_list(audit.get("local_patch_visible_surface_conflicts"), "geometry-source conflicts"), key="object_id", label="geometry-source conflicts")

    visible_by_object = visible_surface_by_object(visible_surface)
    object_rows = [
        build_object_row(
            obj,
            track_rows=track_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            motion_rows=motion_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            pose_rows=pose_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            replay_rows=replay_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            observed_seed_rows=observed_seed_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            reconstruction_job_rows=reconstruction_jobs_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            reconstruction_result_rows=reconstruction_results_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            depth_contact_rows=depth_contact_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            contact_rows=contact_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
            contact_ownership_rows=contact_ownership_by_obj.get(require_str(obj.get("object_id"), "object_id"), []),
            conflict_rows=conflict_by_object.get(require_str(obj.get("object_id"), "object_id"), []),
        )
        for obj in objects
    ]
    missing_visible_objects = sorted(
        set(require_str(obj.get("object_id"), "object_id") for obj in objects).difference(visible_by_object)
    )
    if missing_visible_objects:
        for object_id in missing_visible_objects:
            visible_by_object[object_id] = {}

    summary_counts = {
        "object_count": len(object_rows),
        "solve_activation_ready_object_count": sum(1 for row in object_rows if row["solve_activation_ready"] is True),
        "visible_surface_factor_rows": sum(
            require_int(row["factor_blocks"][0].get("frame_rows"), "visible frame_rows") for row in object_rows
        ),
        "material_rigidity_pair_factor_count": sum(
            require_int(block.get("rigid_factor_ready_pair_count"), "rigid factor count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "material_correspondence_rigidity"
        ),
        "partial_material_pose_ready_segment_count": sum(
            require_int(block.get("ready_segment_count"), "ready segment count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "partial_material_pose_segments"
        ),
        "partial_visible_surface_replay_ready_segment_count": sum(
            require_int(block.get("ready_segment_count"), "ready segment count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "visible_surface_replay_segments"
        ),
        "observed_surface_geometry_seed_count": sum(
            require_int(block.get("seed_candidate_count"), "seed_candidate_count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "observed_surface_geometry_seed"
        ),
        "observed_surface_geometry_seed_vertices": sum(
            require_int(seed.get("seed_vertices"), "seed_vertices")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "observed_surface_geometry_seed"
            for seed in require_list(block.get("seed_candidates"), "seed_candidates")
        ),
        "observed_surface_geometry_seed_faces": sum(
            require_int(seed.get("seed_faces"), "seed_faces")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "observed_surface_geometry_seed"
            for seed in require_list(block.get("seed_candidates"), "seed_candidates")
        ),
        "geometry_reconstruction_job_count": sum(
            require_int(block.get("job_count"), "geometry reconstruction job count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_jobs"
        ),
        "geometry_reconstruction_solver_job_ready_count": sum(
            require_int(block.get("solver_job_ready_count"), "solver job ready count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_jobs"
        ),
        "geometry_reconstruction_hidden_topology_job_count": sum(
            require_int(block.get("hidden_topology_reconstructed_job_count"), "hidden topology job count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_jobs"
        ),
        "geometry_reconstruction_result_job_count": sum(
            require_int(block.get("job_count"), "geometry reconstruction result job count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_pending_solver_output_count": sum(
            require_int(block.get("pending_solver_output_count"), "pending solver output count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_solver_output_detected_count": sum(
            require_int(block.get("solver_output_detected_count"), "solver output detected count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_mesh_file_detected_count": sum(
            require_int(block.get("mesh_file_detected_count"), "mesh file detected count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_pose_sequence_complete_count": sum(
            require_int(block.get("pose_sequence_complete_count"), "pose sequence complete count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_mesh_scale_plausible_count": sum(
            require_int(block.get("mesh_scale_plausible_count"), "mesh scale plausible count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_mesh_projection_qc_passed_count": sum(
            require_int(block.get("mesh_projection_qc_passed_count"), "mesh projection qc passed count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_result_hidden_topology_job_count": sum(
            require_int(block.get("hidden_topology_reconstructed_job_count"), "result hidden topology job count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "geometry_reconstruction_accepted_result_count": sum(
            require_int(block.get("accepted_reconstruction_result_count"), "accepted reconstruction result count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "unknown_object_rgbd_reconstruction_results"
        ),
        "multi_object_contact_factor_ready_rows": sum(
            require_int(block.get("contact_factor_ready_rows"), "contact factor rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "multi_object_hand_contact_distance"
        ),
        "contact_owner_variable_count": require_int(
            contact_ownership.get("contact_owner_variable_count"),
            "contact owner variable count",
        ),
        "contact_owner_candidate_rows": sum(
            require_int(block.get("candidate_rows"), "contact owner candidate rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "contact_owner_supported_candidate_rows": sum(
            require_int(block.get("supported_candidate_rows"), "contact owner supported candidate rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "contact_owner_geometrically_supported_candidate_rows": sum(
            require_int(block.get("geometrically_supported_candidate_rows"), "contact owner geometrically supported rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "contact_owner_image_supported_candidate_rows": sum(
            require_int(block.get("image_supported_candidate_rows"), "contact owner image-supported rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "contact_owner_metric_depth_evaluated_rows": require_int(
            contact_ownership.get("pairwise_metric_depth_evaluated_rows"),
            "contact owner pairwise metric depth evaluated rows",
        ),
        "contact_owner_metric_depth_compatible_candidate_rows": require_int(
            contact_ownership.get("pairwise_metric_depth_compatible_candidate_rows"),
            "contact owner pairwise metric depth compatible rows",
        ),
        "contact_owner_metric_depth_supported_candidate_rows": sum(
            require_int(block.get("metric_depth_supported_candidate_rows"), "contact owner metric-depth supported rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "contact_owner_factor_ready_rows": sum(
            require_int(block.get("contact_owner_factor_ready_rows"), "contact owner factor-ready rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "object_contact_ownership"
        ),
        "depth_contact_evaluated_frame_count": sum(
            require_int(block.get("evaluated_frame_count"), "depth-contact evaluated frame count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_evaluated_hand_rows": sum(
            require_int(block.get("evaluated_hand_rows"), "depth-contact evaluated hand rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_near_reconstructed_mesh_hand_rows": sum(
            require_int(block.get("near_reconstructed_mesh_hand_rows"), "near reconstructed mesh hand rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_reconstructed_mesh_contact_candidate_rows": sum(
            require_int(block.get("reconstructed_mesh_contact_candidate_rows"), "reconstructed mesh contact candidates")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_legacy_contact_ready_hand_rows": sum(
            require_int(block.get("legacy_contact_ready_hand_rows"), "legacy contact ready hand rows")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": sum(
            require_int(
                block.get("multi_object_reconstructed_object_contact_candidate_rows"),
                "multi-object reconstructed object contact candidate rows",
            )
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_legacy_owner_mismatch_frame_count": sum(
            require_int(block.get("legacy_owner_mismatch_frame_count"), "legacy owner mismatch frame count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_shared_depth_state_ready_frame_count": sum(
            require_int(block.get("shared_depth_state_ready_frame_count"), "shared depth ready frames")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "depth_contact_owner_incompatibility_count": sum(
            require_int(block.get("depth_owner_incompatibility_count"), "depth owner incompatibility count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "depth_contact_consistency"
        ),
        "geometry_source_conflict_count": sum(
            require_int(block.get("source_conflict_count"), "source conflict count")
            for row in object_rows
            for block in row["factor_blocks"]
            if block.get("factor_block") == "geometry_source_compatibility"
        ),
    }
    if summary_counts["visible_surface_factor_rows"] != require_int(
        visible_surface.get("surface_frame_rows"),
        "visible_surface surface_frame_rows",
    ):
        raise RuntimeError(f"{case} object factor visible-surface rows disagree with visible-surface report")
    if summary_counts["multi_object_contact_factor_ready_rows"] != require_int(
        contact.get("contact_factor_ready_rows"),
        "contact factor ready rows",
    ):
        raise RuntimeError(f"{case} contact factor-ready rows disagree with multi-object contact evidence")
    if summary_counts["contact_owner_variable_count"] != require_int(
        contact_ownership.get("contact_owner_variable_count"),
        "contact owner variable count",
    ):
        raise RuntimeError(f"{case} contact owner variable count disagrees with contact-ownership report")
    if summary_counts["contact_owner_candidate_rows"] != require_int(
        contact_ownership.get("contact_owner_candidate_rows"),
        "contact owner candidate rows",
    ):
        raise RuntimeError(f"{case} contact owner candidate rows disagree with contact-ownership report")
    if summary_counts["contact_owner_factor_ready_rows"] != require_int(
        contact_ownership.get("contact_owner_factor_ready_rows"),
        "contact owner factor-ready rows",
    ):
        raise RuntimeError(f"{case} contact owner factor-ready rows disagree with contact-ownership report")
    if summary_counts["contact_owner_image_supported_candidate_rows"] != require_int(
        contact_ownership.get("contact_owner_image_supported_candidate_rows"),
        "contact owner image-supported rows",
    ):
        raise RuntimeError(f"{case} contact owner image-supported rows disagree with contact-ownership report")
    if summary_counts["contact_owner_metric_depth_evaluated_rows"] != require_int(
        pairwise_depth_gap.get("evaluated_pair_depth_rows"),
        "pairwise depth-gap evaluated rows",
    ):
        raise RuntimeError(f"{case} metric-depth evaluated rows disagree with pairwise depth-gap report")
    if summary_counts["contact_owner_metric_depth_compatible_candidate_rows"] != require_int(
        pairwise_depth_gap.get("metric_depth_compatible_candidate_rows"),
        "pairwise depth-gap compatible rows",
    ):
        raise RuntimeError(f"{case} metric-depth compatible rows disagree with pairwise depth-gap report")
    if summary_counts["contact_owner_metric_depth_supported_candidate_rows"] != require_int(
        contact_ownership.get("contact_owner_metric_depth_supported_candidate_rows"),
        "contact owner metric-depth supported rows",
    ):
        raise RuntimeError(f"{case} metric-depth supported rows disagree with contact-ownership report")
    if summary_counts["geometry_source_conflict_count"] != len(
        require_list(audit.get("local_patch_visible_surface_conflicts"), "local_patch_visible_surface_conflicts")
    ):
        raise RuntimeError(f"{case} source conflict rows disagree with geometry-source audit")
    if summary_counts["geometry_reconstruction_job_count"] != require_int(
        reconstruction_jobs_report.get("job_count"),
        "geometry reconstruction job_count",
    ):
        raise RuntimeError(f"{case} geometry reconstruction job count disagrees with report")
    if summary_counts["geometry_reconstruction_solver_job_ready_count"] != require_int(
        reconstruction_jobs_report.get("solver_job_ready_count"),
        "geometry reconstruction solver_job_ready_count",
    ):
        raise RuntimeError(f"{case} solver-ready reconstruction job count disagrees with report")
    if summary_counts["geometry_reconstruction_hidden_topology_job_count"] != require_int(
        reconstruction_jobs_report.get("hidden_topology_reconstructed_job_count"),
        "geometry reconstruction hidden_topology_reconstructed_job_count",
    ):
        raise RuntimeError(f"{case} hidden-topology reconstruction job count disagrees with report")
    if summary_counts["geometry_reconstruction_result_job_count"] != require_int(
        reconstruction_results_report.get("job_count"),
        "geometry reconstruction results job_count",
    ):
        raise RuntimeError(f"{case} geometry reconstruction result job count disagrees with report")
    if summary_counts["geometry_reconstruction_result_hidden_topology_job_count"] != require_int(
        reconstruction_results_report.get("hidden_topology_reconstructed_job_count"),
        "geometry reconstruction results hidden_topology_reconstructed_job_count",
    ):
        raise RuntimeError(f"{case} geometry reconstruction result hidden-topology count disagrees with report")
    if summary_counts["geometry_reconstruction_accepted_result_count"] != require_int(
        reconstruction_results_report.get("accepted_reconstruction_result_count"),
        "geometry reconstruction accepted_reconstruction_result_count",
    ):
        raise RuntimeError(f"{case} accepted geometry reconstruction result count disagrees with report")
    if summary_counts["depth_contact_evaluated_frame_count"] != require_int(
        depth_contact.get("evaluated_frame_count"),
        "depth-contact evaluated_frame_count",
    ):
        raise RuntimeError(f"{case} depth-contact evaluated frame count disagrees with report")
    if summary_counts["depth_contact_owner_incompatibility_count"] != require_int(
        depth_contact.get("depth_owner_incompatibility_count"),
        "depth-contact depth_owner_incompatibility_count",
    ):
        raise RuntimeError(f"{case} depth-contact incompatibility count disagrees with report")
    if summary_counts["depth_contact_legacy_owner_mismatch_frame_count"] != require_int(
        depth_contact.get("legacy_owner_mismatch_frame_count"),
        "depth-contact legacy_owner_mismatch_frame_count",
    ):
        raise RuntimeError(f"{case} depth-contact legacy owner mismatch count disagrees with report")

    report = {
        "method": "build_v17_object_geometry_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": require_int(hypothesis.get("frame_count"), "hypothesis frame_count"),
        "factor_problem_object_rows": len(object_rows),
        "state_counts": require_dict(hypothesis.get("state_counts"), "state_counts"),
        "missing_visible_surface_object_ids": missing_visible_objects,
        "object_rows": object_rows,
        **summary_counts,
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_object_geometry_factor_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.object_geometry_hypothesis_state_root / "v17_object_geometry_hypothesis_state_summary.json",
        "object-geometry hypothesis-state summary",
    )
    summary = require_dict(load_json(summary_path), "object-geometry hypothesis-state summary")
    reports = [
        build_case(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(require_dict(report.get("state_counts"), "state_counts"))
    payload = {
        "method": "build_v17_object_geometry_factor_problem",
        "status": STATUS,
        "claim": CLAIM,
        "source_object_geometry_hypothesis_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "problem_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_object_geometry_factor_problem.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "factor_problem_object_rows": require_int(
                    report.get("factor_problem_object_rows"),
                    "factor_problem_object_rows",
                ),
                "state_counts": require_dict(report.get("state_counts"), "state_counts"),
                "solve_activation_ready_object_count": require_int(
                    report.get("solve_activation_ready_object_count"),
                    "solve_activation_ready_object_count",
                ),
                "visible_surface_factor_rows": require_int(
                    report.get("visible_surface_factor_rows"),
                    "visible_surface_factor_rows",
                ),
                "material_rigidity_pair_factor_count": require_int(
                    report.get("material_rigidity_pair_factor_count"),
                    "material_rigidity_pair_factor_count",
                ),
                "partial_material_pose_ready_segment_count": require_int(
                    report.get("partial_material_pose_ready_segment_count"),
                    "partial_material_pose_ready_segment_count",
                ),
                "partial_visible_surface_replay_ready_segment_count": require_int(
                    report.get("partial_visible_surface_replay_ready_segment_count"),
                    "partial_visible_surface_replay_ready_segment_count",
                ),
                "observed_surface_geometry_seed_count": require_int(
                    report.get("observed_surface_geometry_seed_count"),
                    "observed_surface_geometry_seed_count",
                ),
                "observed_surface_geometry_seed_vertices": require_int(
                    report.get("observed_surface_geometry_seed_vertices"),
                    "observed_surface_geometry_seed_vertices",
                ),
                "observed_surface_geometry_seed_faces": require_int(
                    report.get("observed_surface_geometry_seed_faces"),
                    "observed_surface_geometry_seed_faces",
                ),
                "geometry_reconstruction_job_count": require_int(
                    report.get("geometry_reconstruction_job_count"),
                    "geometry_reconstruction_job_count",
                ),
                "geometry_reconstruction_solver_job_ready_count": require_int(
                    report.get("geometry_reconstruction_solver_job_ready_count"),
                    "geometry_reconstruction_solver_job_ready_count",
                ),
                "geometry_reconstruction_hidden_topology_job_count": require_int(
                    report.get("geometry_reconstruction_hidden_topology_job_count"),
                    "geometry_reconstruction_hidden_topology_job_count",
                ),
                "geometry_reconstruction_result_job_count": require_int(
                    report.get("geometry_reconstruction_result_job_count"),
                    "geometry_reconstruction_result_job_count",
                ),
                "geometry_reconstruction_pending_solver_output_count": require_int(
                    report.get("geometry_reconstruction_pending_solver_output_count"),
                    "geometry_reconstruction_pending_solver_output_count",
                ),
                "geometry_reconstruction_solver_output_detected_count": require_int(
                    report.get("geometry_reconstruction_solver_output_detected_count"),
                    "geometry_reconstruction_solver_output_detected_count",
                ),
                "geometry_reconstruction_mesh_file_detected_count": require_int(
                    report.get("geometry_reconstruction_mesh_file_detected_count"),
                    "geometry_reconstruction_mesh_file_detected_count",
                ),
                "geometry_reconstruction_pose_sequence_complete_count": require_int(
                    report.get("geometry_reconstruction_pose_sequence_complete_count"),
                    "geometry_reconstruction_pose_sequence_complete_count",
                ),
                "geometry_reconstruction_mesh_scale_plausible_count": require_int(
                    report.get("geometry_reconstruction_mesh_scale_plausible_count"),
                    "geometry_reconstruction_mesh_scale_plausible_count",
                ),
                "geometry_reconstruction_mesh_projection_qc_passed_count": require_int(
                    report.get("geometry_reconstruction_mesh_projection_qc_passed_count"),
                    "geometry_reconstruction_mesh_projection_qc_passed_count",
                ),
                "geometry_reconstruction_result_hidden_topology_job_count": require_int(
                    report.get("geometry_reconstruction_result_hidden_topology_job_count"),
                    "geometry_reconstruction_result_hidden_topology_job_count",
                ),
                "geometry_reconstruction_accepted_result_count": require_int(
                    report.get("geometry_reconstruction_accepted_result_count"),
                    "geometry_reconstruction_accepted_result_count",
                ),
                "depth_contact_evaluated_frame_count": require_int(
                    report.get("depth_contact_evaluated_frame_count"),
                    "depth_contact_evaluated_frame_count",
                ),
                "depth_contact_evaluated_hand_rows": require_int(
                    report.get("depth_contact_evaluated_hand_rows"),
                    "depth_contact_evaluated_hand_rows",
                ),
                "depth_contact_near_reconstructed_mesh_hand_rows": require_int(
                    report.get("depth_contact_near_reconstructed_mesh_hand_rows"),
                    "depth_contact_near_reconstructed_mesh_hand_rows",
                ),
                "depth_contact_reconstructed_mesh_contact_candidate_rows": require_int(
                    report.get("depth_contact_reconstructed_mesh_contact_candidate_rows"),
                    "depth_contact_reconstructed_mesh_contact_candidate_rows",
                ),
                "depth_contact_legacy_contact_ready_hand_rows": require_int(
                    report.get("depth_contact_legacy_contact_ready_hand_rows"),
                    "depth_contact_legacy_contact_ready_hand_rows",
                ),
                "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": require_int(
                    report.get("depth_contact_multi_object_reconstructed_object_contact_candidate_rows"),
                    "depth_contact_multi_object_reconstructed_object_contact_candidate_rows",
                ),
                "depth_contact_legacy_owner_mismatch_frame_count": require_int(
                    report.get("depth_contact_legacy_owner_mismatch_frame_count"),
                    "depth_contact_legacy_owner_mismatch_frame_count",
                ),
                "depth_contact_shared_depth_state_ready_frame_count": require_int(
                    report.get("depth_contact_shared_depth_state_ready_frame_count"),
                    "depth_contact_shared_depth_state_ready_frame_count",
                ),
                "depth_contact_owner_incompatibility_count": require_int(
                    report.get("depth_contact_owner_incompatibility_count"),
                    "depth_contact_owner_incompatibility_count",
                ),
                "multi_object_contact_factor_ready_rows": require_int(
                    report.get("multi_object_contact_factor_ready_rows"),
                    "multi_object_contact_factor_ready_rows",
                ),
                "contact_owner_variable_count": require_int(
                    report.get("contact_owner_variable_count"),
                    "contact_owner_variable_count",
                ),
                "contact_owner_candidate_rows": require_int(
                    report.get("contact_owner_candidate_rows"),
                    "contact_owner_candidate_rows",
                ),
                "contact_owner_supported_candidate_rows": require_int(
                    report.get("contact_owner_supported_candidate_rows"),
                    "contact_owner_supported_candidate_rows",
                ),
                "contact_owner_geometrically_supported_candidate_rows": require_int(
                    report.get("contact_owner_geometrically_supported_candidate_rows"),
                    "contact_owner_geometrically_supported_candidate_rows",
                ),
                "contact_owner_image_supported_candidate_rows": require_int(
                    report.get("contact_owner_image_supported_candidate_rows"),
                    "contact_owner_image_supported_candidate_rows",
                ),
                "contact_owner_metric_depth_evaluated_rows": require_int(
                    report.get("contact_owner_metric_depth_evaluated_rows"),
                    "contact_owner_metric_depth_evaluated_rows",
                ),
                "contact_owner_metric_depth_compatible_candidate_rows": require_int(
                    report.get("contact_owner_metric_depth_compatible_candidate_rows"),
                    "contact_owner_metric_depth_compatible_candidate_rows",
                ),
                "contact_owner_metric_depth_supported_candidate_rows": require_int(
                    report.get("contact_owner_metric_depth_supported_candidate_rows"),
                    "contact_owner_metric_depth_supported_candidate_rows",
                ),
                "contact_owner_factor_ready_rows": require_int(
                    report.get("contact_owner_factor_ready_rows"),
                    "contact_owner_factor_ready_rows",
                ),
                "geometry_source_conflict_count": require_int(
                    report.get("geometry_source_conflict_count"),
                    "geometry_source_conflict_count",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "object_count": sum(require_int(report.get("factor_problem_object_rows"), "object rows") for report in reports),
        "state_counts": dict(sorted(state_counts.items())),
        "solve_activation_ready_object_count": sum(
            require_int(report.get("solve_activation_ready_object_count"), "solve-ready")
            for report in reports
        ),
        "visible_surface_factor_rows": sum(
            require_int(report.get("visible_surface_factor_rows"), "visible rows") for report in reports
        ),
        "material_rigidity_pair_factor_count": sum(
            require_int(report.get("material_rigidity_pair_factor_count"), "rigid pair count")
            for report in reports
        ),
        "partial_material_pose_ready_segment_count": sum(
            require_int(report.get("partial_material_pose_ready_segment_count"), "material pose ready count")
            for report in reports
        ),
        "partial_visible_surface_replay_ready_segment_count": sum(
            require_int(report.get("partial_visible_surface_replay_ready_segment_count"), "surface replay ready count")
            for report in reports
        ),
        "observed_surface_geometry_seed_count": sum(
            require_int(report.get("observed_surface_geometry_seed_count"), "observed surface seed count")
            for report in reports
        ),
        "observed_surface_geometry_seed_vertices": sum(
            require_int(report.get("observed_surface_geometry_seed_vertices"), "observed surface seed vertices")
            for report in reports
        ),
        "observed_surface_geometry_seed_faces": sum(
            require_int(report.get("observed_surface_geometry_seed_faces"), "observed surface seed faces")
            for report in reports
        ),
        "geometry_reconstruction_job_count": sum(
            require_int(report.get("geometry_reconstruction_job_count"), "geometry reconstruction job count")
            for report in reports
        ),
        "geometry_reconstruction_solver_job_ready_count": sum(
            require_int(
                report.get("geometry_reconstruction_solver_job_ready_count"),
                "geometry reconstruction solver job ready count",
            )
            for report in reports
        ),
        "geometry_reconstruction_hidden_topology_job_count": sum(
            require_int(
                report.get("geometry_reconstruction_hidden_topology_job_count"),
                "geometry reconstruction hidden topology job count",
            )
            for report in reports
        ),
        "geometry_reconstruction_result_job_count": sum(
            require_int(report.get("geometry_reconstruction_result_job_count"), "geometry reconstruction result job count")
            for report in reports
        ),
        "geometry_reconstruction_pending_solver_output_count": sum(
            require_int(
                report.get("geometry_reconstruction_pending_solver_output_count"),
                "geometry reconstruction pending solver output count",
            )
            for report in reports
        ),
        "geometry_reconstruction_solver_output_detected_count": sum(
            require_int(
                report.get("geometry_reconstruction_solver_output_detected_count"),
                "geometry reconstruction solver output detected count",
            )
            for report in reports
        ),
        "geometry_reconstruction_mesh_file_detected_count": sum(
            require_int(
                report.get("geometry_reconstruction_mesh_file_detected_count"),
                "geometry reconstruction mesh file detected count",
            )
            for report in reports
        ),
        "geometry_reconstruction_pose_sequence_complete_count": sum(
            require_int(
                report.get("geometry_reconstruction_pose_sequence_complete_count"),
                "geometry reconstruction pose sequence complete count",
            )
            for report in reports
        ),
        "geometry_reconstruction_mesh_scale_plausible_count": sum(
            require_int(
                report.get("geometry_reconstruction_mesh_scale_plausible_count"),
                "geometry reconstruction mesh scale plausible count",
            )
            for report in reports
        ),
        "geometry_reconstruction_mesh_projection_qc_passed_count": sum(
            require_int(
                report.get("geometry_reconstruction_mesh_projection_qc_passed_count"),
                "geometry reconstruction mesh projection qc passed count",
            )
            for report in reports
        ),
        "geometry_reconstruction_result_hidden_topology_job_count": sum(
            require_int(
                report.get("geometry_reconstruction_result_hidden_topology_job_count"),
                "geometry reconstruction result hidden topology job count",
            )
            for report in reports
        ),
        "geometry_reconstruction_accepted_result_count": sum(
            require_int(
                report.get("geometry_reconstruction_accepted_result_count"),
                "geometry reconstruction accepted result count",
            )
            for report in reports
        ),
        "depth_contact_evaluated_frame_count": sum(
            require_int(report.get("depth_contact_evaluated_frame_count"), "depth-contact evaluated frame count")
            for report in reports
        ),
        "depth_contact_evaluated_hand_rows": sum(
            require_int(report.get("depth_contact_evaluated_hand_rows"), "depth-contact evaluated hand rows")
            for report in reports
        ),
        "depth_contact_near_reconstructed_mesh_hand_rows": sum(
            require_int(
                report.get("depth_contact_near_reconstructed_mesh_hand_rows"),
                "depth-contact near reconstructed mesh hand rows",
            )
            for report in reports
        ),
        "depth_contact_reconstructed_mesh_contact_candidate_rows": sum(
            require_int(
                report.get("depth_contact_reconstructed_mesh_contact_candidate_rows"),
                "depth-contact reconstructed mesh contact candidate rows",
            )
            for report in reports
        ),
        "depth_contact_legacy_contact_ready_hand_rows": sum(
            require_int(
                report.get("depth_contact_legacy_contact_ready_hand_rows"),
                "depth-contact legacy contact ready hand rows",
            )
            for report in reports
        ),
        "depth_contact_multi_object_reconstructed_object_contact_candidate_rows": sum(
            require_int(
                report.get("depth_contact_multi_object_reconstructed_object_contact_candidate_rows"),
                "depth-contact multi-object reconstructed object contact candidate rows",
            )
            for report in reports
        ),
        "depth_contact_legacy_owner_mismatch_frame_count": sum(
            require_int(
                report.get("depth_contact_legacy_owner_mismatch_frame_count"),
                "depth-contact legacy owner mismatch frame count",
            )
            for report in reports
        ),
        "depth_contact_shared_depth_state_ready_frame_count": sum(
            require_int(
                report.get("depth_contact_shared_depth_state_ready_frame_count"),
                "depth-contact shared depth state ready frame count",
            )
            for report in reports
        ),
        "depth_contact_owner_incompatibility_count": sum(
            require_int(
                report.get("depth_contact_owner_incompatibility_count"),
                "depth-contact owner incompatibility count",
            )
            for report in reports
        ),
        "multi_object_contact_factor_ready_rows": sum(
            require_int(report.get("multi_object_contact_factor_ready_rows"), "contact factor rows")
            for report in reports
        ),
        "contact_owner_variable_count": sum(
            require_int(report.get("contact_owner_variable_count"), "contact owner variable count")
            for report in reports
        ),
        "contact_owner_candidate_rows": sum(
            require_int(report.get("contact_owner_candidate_rows"), "contact owner candidate rows")
            for report in reports
        ),
        "contact_owner_supported_candidate_rows": sum(
            require_int(report.get("contact_owner_supported_candidate_rows"), "contact owner supported rows")
            for report in reports
        ),
        "contact_owner_geometrically_supported_candidate_rows": sum(
            require_int(
                report.get("contact_owner_geometrically_supported_candidate_rows"),
                "contact owner geometrically supported rows",
            )
            for report in reports
        ),
        "contact_owner_image_supported_candidate_rows": sum(
            require_int(report.get("contact_owner_image_supported_candidate_rows"), "contact owner image-supported rows")
            for report in reports
        ),
        "contact_owner_metric_depth_evaluated_rows": sum(
            require_int(report.get("contact_owner_metric_depth_evaluated_rows"), "contact owner metric-depth evaluated rows")
            for report in reports
        ),
        "contact_owner_metric_depth_compatible_candidate_rows": sum(
            require_int(
                report.get("contact_owner_metric_depth_compatible_candidate_rows"),
                "contact owner metric-depth compatible rows",
            )
            for report in reports
        ),
        "contact_owner_metric_depth_supported_candidate_rows": sum(
            require_int(
                report.get("contact_owner_metric_depth_supported_candidate_rows"),
                "contact owner metric-depth supported rows",
            )
            for report in reports
        ),
        "contact_owner_factor_ready_rows": sum(
            require_int(report.get("contact_owner_factor_ready_rows"), "contact owner factor-ready rows")
            for report in reports
        ),
        "geometry_source_conflict_count": sum(
            require_int(report.get("geometry_source_conflict_count"), "source conflict count")
            for report in reports
        ),
        "complete_object_geometry_hypothesis_count": 0,
        "contact_compatible_object_geometry_hypothesis_count": 0,
        "object_pose_factor_ready_hypothesis_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_object_geometry_factor_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--object-material-track-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
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
        "--geometry-reconstruction-jobs-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs"),
    )
    parser.add_argument(
        "--geometry-reconstruction-results-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--contact-ownership-problem-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_ownership_problem"),
    )
    parser.add_argument(
        "--pairwise-contact-depth-gap-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument(
        "--geometry-source-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_source_audit"),
    )
    parser.add_argument(
        "--depth-contact-consistency-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_factor_problem"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
