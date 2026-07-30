#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_constraints(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("constraint_rows", []):
        try:
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
        except Exception:
            continue
    return out


def full_signed_domain(row: dict[str, Any]) -> bool:
    signed_summary = row.get("signed_distance_m") if isinstance(row.get("signed_distance_m"), dict) else {}
    signed_count = signed_summary.get("count") if isinstance(signed_summary, dict) else None
    hand_vertex_count = row.get("hand_vertex_count")
    if isinstance(signed_count, (int, float)) and isinstance(hand_vertex_count, (int, float)):
        return int(signed_count) == int(hand_vertex_count)
    return False


def hand_update_from_constraint(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("candidate_application_state")
    if state == "candidate_coordinate_correction_visible_2d_compatible":
        h_state = "compact_rigid_object_nonpenetration_corrected"
        h_prime_equals_h = False
        uncertainty = [
            "metric MANO state was translated by the minimal compact-rigid object nonpenetration correction; visible 2D consistency predicate is recorded in this update"
        ]
    elif state in {"not_applied_measurement_only", "not_applied_visible_2d_conflict_or_unmeasured", "not_applied_escape_solver_failed", "not_applied_local_escape_amplified"}:
        h_state = "candidate_coordinate_correction_requires_visible_2d_review"
        h_prime_equals_h = True
        uncertainty = [
            "object mesh suggests signed nonpenetration correction, but coordinate update is held because visible 2D consistency was not established"
        ]
    elif state in {"uncertainty_only_nonwatertight_mesh_no_signed_correction", "uncertainty_sign_mesh_missing_near_surface_support", "uncertainty_signed_distance_not_evaluated_broadphase_support"} or (state == "no_penetration_no_coordinate_change_needed" and not full_signed_domain(row)):
        h_state = "unchanged_with_compact_rigid_object_overlap_uncertainty"
        h_prime_equals_h = True
        if state == "no_penetration_no_coordinate_change_needed" and not full_signed_domain(row):
            uncertainty = [
                "post-correction signed nonpenetration was reported only on a subset of bridge hand vertices",
                "hand state remains HaWoR metric MANO with object-constraint uncertainty until signed inside/outside evidence covers every bridge hand vertex",
            ]
        elif state == "uncertainty_sign_mesh_missing_near_surface_support":
            uncertainty = [
                "MANO vertices are within the observed object-surface band, but the watertight hidden-volume prior does not cover that region",
                "hand state remains HaWoR metric MANO with added object-constraint uncertainty; no coordinate move is justified by this sign mesh",
            ]
        elif state == "uncertainty_signed_distance_not_evaluated_broadphase_support":
            uncertainty = [
                "MANO vertices are within the observed object-surface band and covered by the sign-mesh AABB, but exact signed distance has not been evaluated yet",
                "hand state remains HaWoR metric MANO with object-constraint uncertainty until the local signed test is completed",
            ]
        else:
            uncertainty = [
                "compact-rigid surface evidence overlaps the hand, but no watertight sign mesh is available for a signed volume correction",
                "hand state remains HaWoR metric MANO with added object-constraint uncertainty rather than a silent coordinate move",
            ]
    else:
        h_state = "validated_no_compact_rigid_object_coordinate_change"
        h_prime_equals_h = True
        uncertainty = []
    return {
        "method": "apply_v18_mano_object_constraint_state",
        "h_prime_state": h_state,
        "h_prime_equals_input_h": h_prime_equals_h,
        "coordinate_update_applied": state == "candidate_coordinate_correction_visible_2d_compatible",
        "candidate_translation_world_m": row.get("candidate_translation_world_m"),
        "candidate_translation_norm_m": row.get("candidate_translation_norm_m"),
        "candidate_joint_reprojection_shift_px": row.get("candidate_joint_reprojection_shift_px"),
        "object_constraint": {
            "object_id": row.get("object_id"),
            "surface_mesh_path": row.get("surface_mesh_path") or row.get("mesh_path"),
            "sign_mesh_path": row.get("sign_mesh_path"),
            "sign_mesh_source_report": row.get("sign_mesh_source_report"),
            "completed_surface_mesh_watertight": row.get("completed_surface_mesh_watertight", row.get("completed_mesh_watertight")),
            "sign_mesh_watertight": row.get("sign_mesh_watertight"),
            "hand_vertex_count": row.get("hand_vertex_count"),
            "signed_distance_query_scope": row.get("signed_distance_query_scope"),
            "near_surface_gate_applied_to_signed_distance": row.get("near_surface_gate_applied_to_signed_distance"),
            "sign_aabb_gate_applied_to_signed_distance": row.get("sign_aabb_gate_applied_to_signed_distance"),
            "observed_band_m": row.get("observed_band_m"),
            "near_surface_vertex_count": row.get("near_surface_vertex_count"),
            "near_surface_vertex_fraction": row.get("near_surface_vertex_fraction"),
            "surface_aabb_candidate_vertex_count": row.get("surface_aabb_candidate_vertex_count", row.get("aabb_candidate_vertex_count")),
            "surface_aabb_candidate_vertex_fraction": row.get("surface_aabb_candidate_vertex_fraction", row.get("aabb_candidate_vertex_fraction")),
            "sign_aabb_candidate_vertex_count": row.get("sign_aabb_candidate_vertex_count"),
            "sign_aabb_candidate_vertex_fraction": row.get("sign_aabb_candidate_vertex_fraction"),
            "outside_sign_aabb_vertex_count": row.get("outside_sign_aabb_vertex_count"),
            "outside_sign_aabb_vertex_fraction": row.get("outside_sign_aabb_vertex_fraction"),
            "signed_query_candidate_vertex_count": row.get("signed_query_candidate_vertex_count"),
            "signed_query_candidate_vertex_fraction": row.get("signed_query_candidate_vertex_fraction"),
            "penetrating_vertex_count": row.get("penetrating_vertex_count"),
            "nearest_surface_unsigned_m": row.get("nearest_surface_unsigned_m"),
            "signed_distance_m": row.get("signed_distance_m"),
            "application_state": state,
            "visible_2d_consistency": row.get("candidate_visible_2d_consistency"),
            "reason": row.get("reason"),
        },
        "uncertainty_added": uncertainty,
        "scope": "MANO backing-state consumption of compact-rigid object constraint; coordinate updates are explicit H-prime translations when visible 2D consistency is established",
    }


def add_vec_to_vector(raw: Any, delta: np.ndarray) -> list[float] | None:
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return [float(x) for x in (arr + delta).tolist()]


def add_vec_to_points(raw: Any, delta: np.ndarray) -> list[list[float]] | None:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        return None
    return [[float(x) for x in row] for row in (arr + delta[None, :]).tolist()]


def frame_camera_delta(frame: dict[str, Any], delta_world: np.ndarray) -> tuple[np.ndarray | None, str]:
    camera_raw = frame.get("camera")
    camera = camera_raw if isinstance(camera_raw, dict) else {}
    transform = np.asarray(camera.get("T_world_camera_metric") or [], dtype=float)
    if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
        return delta_world @ transform[:3, :3], "annotation_frame_T_world_camera_metric"
    return None, "missing_annotation_frame_T_world_camera_metric"


def translate_mano_params(params: Any, delta_world: np.ndarray) -> Any:
    if not isinstance(params, dict):
        return params
    out = json.loads(json.dumps(params))
    translated = add_vec_to_vector(out.get("trans_world_m"), delta_world)
    if translated is not None:
        out["trans_world_m"] = translated
        out["trans_world_m_updated_by_compact_rigid_constraint"] = True
    return out


def apply_coordinate_translation(hand: dict[str, Any], frame: dict[str, Any], update: dict[str, Any], previous_update: dict[str, Any] | None = None) -> dict[str, Any]:
    delta_world = np.asarray(update.get("candidate_translation_world_m") or [], dtype=float)
    if delta_world.shape != (3,) or not np.all(np.isfinite(delta_world)):
        update["coordinate_update_applied"] = False
        update["coordinate_update_failure"] = "invalid_candidate_translation_world_m"
        update["h_prime_state"] = "candidate_coordinate_correction_requires_visible_2d_review"
        update["h_prime_equals_input_h"] = True
        return update
    delta_camera, camera_source = frame_camera_delta(frame, delta_world)
    previous_cumulative = np.zeros(3, dtype=float)
    if isinstance(previous_update, dict):
        raw_prev = previous_update.get("cumulative_translation_world_m") or previous_update.get("applied_translation_world_m")
        arr_prev = np.asarray(raw_prev or [], dtype=float)
        if arr_prev.shape == (3,) and np.all(np.isfinite(arr_prev)):
            previous_cumulative = arr_prev
            update["previous_translation_world_m"] = [float(x) for x in previous_cumulative.tolist()]
    cumulative = previous_cumulative + delta_world
    update["applied_translation_world_m"] = [float(x) for x in delta_world.tolist()]
    update["cumulative_translation_world_m"] = [float(x) for x in cumulative.tolist()]
    update["camera_translation_source"] = camera_source
    if delta_camera is not None:
        update["applied_translation_camera_m"] = [float(x) for x in delta_camera.tolist()]
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else None
    if isinstance(metric, dict):
        for key in ["vertices_world_sample_m", "joints_current_v18_world_m"]:
            translated = add_vec_to_points(metric.get(key), delta_world)
            if translated is not None:
                metric[key] = translated
        wrist = add_vec_to_vector(metric.get("wrist_current_v18_world_m"), delta_world)
        if wrist is not None:
            metric["wrist_current_v18_world_m"] = wrist
        if delta_camera is not None:
            for key in ["vertices_camera_sample_m", "joints_current_v18_camera_m"]:
                translated = add_vec_to_points(metric.get(key), delta_camera)
                if translated is not None:
                    metric[key] = translated
        metric["mano_params"] = translate_mano_params(metric.get("mano_params"), delta_world)
        metric["compact_rigid_object_corrected_h_prime"] = True
    mano_candidate = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else None
    if isinstance(mano_candidate, dict):
        mano_candidate["mano_params"] = translate_mano_params(mano_candidate.get("mano_params"), delta_world)
        # The legacy overlay candidate may live in a different camera normalization.
        # Keep its raw joints unchanged and let the renderer draw corrected metric
        # MANO from metric_mano_state when coordinate_update_applied is true.
        mano_candidate["compact_rigid_object_corrected_h_prime_available_in_metric_state"] = True
    return update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--constraint-report", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    constraints = load_json(args.constraint_report)
    by_key = index_constraints(constraints)
    applied = 0
    uncertainty = 0
    candidate_corrections = 0
    validated_no_change = 0
    for frame in annotations.get("frames", []):
        frame_idx = int(frame.get("frame_idx"))
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            side = str(hand.get("hand_side"))
            row = by_key.get((frame_idx, side))
            if row is None:
                continue
            previous_update = hand.get("compact_rigid_object_mano_constraint_update") if isinstance(hand.get("compact_rigid_object_mano_constraint_update"), dict) else None
            metric_for_previous = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            previous_corrected = bool(
                (isinstance(previous_update, dict) and previous_update.get("coordinate_update_applied") is True)
                or (isinstance(metric_for_previous, dict) and metric_for_previous.get("compact_rigid_object_corrected_h_prime") is True)
            )
            update = hand_update_from_constraint(row)
            new_coordinate_update = bool(update.get("coordinate_update_applied") is True)
            if new_coordinate_update:
                update = apply_coordinate_translation(hand, frame, update, previous_update=previous_update)
            elif previous_corrected and update.get("h_prime_state") == "validated_no_compact_rigid_object_coordinate_change" and isinstance(previous_update, dict):
                update = previous_update
                update["post_hprime_verification"] = {
                    "application_state": row.get("candidate_application_state"),
                    "nearest_surface_unsigned_m": row.get("nearest_surface_unsigned_m"),
                    "signed_distance_m": row.get("signed_distance_m"),
                    "penetrating_vertex_count": row.get("penetrating_vertex_count"),
                    "hand_vertex_count": row.get("hand_vertex_count"),
                    "signed_distance_query_scope": row.get("signed_distance_query_scope"),
                    "signed_query_candidate_vertex_count": row.get("signed_query_candidate_vertex_count"),
                    "near_surface_gate_applied_to_signed_distance": row.get("near_surface_gate_applied_to_signed_distance"),
                    "sign_aabb_gate_applied_to_signed_distance": row.get("sign_aabb_gate_applied_to_signed_distance"),
                    "verified_no_additional_coordinate_change": True,
                }
            hand["compact_rigid_object_mano_constraint_update"] = update
            # Also attach to the existing metric MANO state if present so renderers
            # and downstream consumers looking at the hand state, rather than a
            # side report, can see the changed uncertainty semantics.
            if isinstance(hand.get("metric_mano_state"), dict):
                hand["metric_mano_state"]["compact_rigid_object_constraint_update"] = update
            applied += 1
            state = update["h_prime_state"]
            if new_coordinate_update:
                candidate_corrections += 1
            elif state == "unchanged_with_compact_rigid_object_overlap_uncertainty":
                uncertainty += 1
            elif state == "candidate_coordinate_correction_requires_visible_2d_review":
                uncertainty += 1
            elif state == "validated_no_compact_rigid_object_coordinate_change":
                validated_no_change += 1

    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations.write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    summary = {
        "method": "apply_v18_mano_object_constraint_state",
        "status": "ok",
        "input_annotations": str(args.annotations),
        "constraint_report": str(args.constraint_report),
        "output_annotations": str(args.output_annotations),
        "applied_hand_rows": applied,
        "uncertainty_hand_rows": uncertainty,
        "candidate_coordinate_correction_rows": candidate_corrections,
        "validated_no_change_rows": validated_no_change,
        "claim_scope": "This file consumes compact-rigid object constraints into hand backing state. It does not claim coordinate improvement unless coordinate_update_applied is true.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
