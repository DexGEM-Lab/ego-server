#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


STATUS = "v18_status_invariant_audit"
CLAIM = (
    "This artifact audits V18 status-deliverable invariants across generated reports. Passing this audit supports "
    "the scoped status deliverable only; it does not prove final pose/object/contact completion."
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


def check(checks: list[dict[str, Any]], check_id: str, passed: bool, observed: Any, expected: Any, severity: str = "required") -> None:
    checks.append({"id": check_id, "passed": bool(passed), "observed": observed, "expected": expected, "severity": severity})


def nested_case(manifest: dict[str, Any], case: str) -> dict[str, Any]:
    for raw in manifest.get("cases", []):
        row = require_dict(raw, "manifest case")
        if row.get("case") == case:
            return row
    raise RuntimeError(f"manifest has no case {case}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    manifest_path = args.status_manifest_root / "v18_status_deliverable_manifest.json"
    runtime_path = args.measured_runtime_root / "v18_measured_status_pipeline_runtime_report.json"
    hand_baseline_path = args.hand_baseline_root / "v18_hand_baseline_branch_summary.json"
    visibility_summary_path = args.visibility_occlusion_root / "v18_visibility_occlusion_state_summary.json"
    subset_summary_path = args.visible_part_subset_root / "v18_visible_part_subset_archive_summary.json"
    occlusion_summary_path = args.occlusion_owner_candidates_root / "v18_occlusion_owner_candidates_summary.json"
    occlusion_depth_summary_path = args.occlusion_depth_evidence_root / "v18_occlusion_depth_order_evidence_summary.json"
    bounded_summary_path = args.bounded_state_root / "v18_bounded_state_solution_summary.json"
    physical_schema_path = args.physical_state_schema_root / "v18_physical_state_schema_summary.json"
    part_source_path = args.part_track_source_root / "v18_part_track_source_manifest_summary.json"
    part_split_path = args.part_split_evidence_root / "v18_part_split_evidence_summary.json"
    part_surfaces_path = args.part_visible_surfaces_root / "v18_part_visible_surfaces_summary.json"
    part_models_path = args.part_model_candidates_root / "v18_part_model_candidates_summary.json"
    articulation_fit_path = args.articulation_fit_root / "v18_articulation_fit_candidates_summary.json"
    part_se3_path = args.part_se3_root / "v18_part_se3_surface_residuals_summary.json"
    part_blockers_path = args.part_object_blockers_root / "v18_part_object_blocker_manifest_summary.json"
    acquisition_path = args.part_mask_acquisition_root / "v18_part_mask_acquisition_plan_summary.json"
    sam_promptable_path = args.sam_promptable_proposals_root / "v18_sam_promptable_part_proposals_summary.json"
    promotion_gate_path = args.part_mask_promotion_gate_root / "v18_part_mask_promotion_gate_summary.json"
    manifest = require_dict(load_json(manifest_path), "status manifest")
    runtime = require_dict(load_json(runtime_path), "runtime report")
    hand_baseline = require_dict(load_json(hand_baseline_path), "hand baseline summary")
    visibility = require_dict(load_json(visibility_summary_path), "visibility occlusion summary")
    subset = require_dict(load_json(subset_summary_path), "visible part subset summary")
    occlusion = require_dict(load_json(occlusion_summary_path), "occlusion candidates summary")
    occlusion_depth = require_dict(load_json(occlusion_depth_summary_path), "occlusion depth-order evidence summary")
    bounded = require_dict(load_json(bounded_summary_path), "bounded state summary")
    physical = require_dict(load_json(physical_schema_path), "physical schema summary")
    part_source = require_dict(load_json(part_source_path), "part source summary")
    part_split = require_dict(load_json(part_split_path), "part split summary")
    part_surfaces = require_dict(load_json(part_surfaces_path), "part visible surfaces summary")
    part_models = require_dict(load_json(part_models_path), "part model candidates summary")
    articulation_fit = require_dict(load_json(articulation_fit_path), "articulation fit summary")
    part_se3 = require_dict(load_json(part_se3_path), "part SE3 surface residual summary")
    part_blockers = require_dict(load_json(part_blockers_path), "part object blockers summary")
    acquisition = require_dict(load_json(acquisition_path), "part mask acquisition summary")
    sam_promptable = require_dict(load_json(sam_promptable_path), "SAM promptable proposals summary")
    promotion_gate = require_dict(load_json(promotion_gate_path), "part mask promotion gate summary")

    checks: list[dict[str, Any]] = []
    check(checks, "status_deliverable_ready", manifest.get("status_deliverable_ready") is True, manifest.get("status_deliverable_ready"), True)
    check(checks, "final_pose_complete_deliverable_not_ready", manifest.get("final_pose_complete_deliverable_ready") is False, manifest.get("final_pose_complete_deliverable_ready"), False)
    for flag in [
        "annotation_ready",
        "deliverable_ready",
        "accuracy_target_met",
        "object_geometry_complete",
        "object_pose_requirement_met",
        "rigid_pose_requirement_met",
        "v3_solver_complete",
    ]:
        check(checks, f"false_readiness_{flag}", manifest.get(flag) is False, manifest.get(flag), False)
    for flag in ["all_frame_counts_match_raw", "all_status_video_durations_match_raw", "all_status_video_fps_match_raw"]:
        check(checks, flag, manifest.get(flag) is True, manifest.get(flag), True)
    check(checks, "no_bundlesdf_or_nerf_default", manifest.get("default_path_uses_bundlesdf_or_nerf") is False, manifest.get("default_path_uses_bundlesdf_or_nerf"), False)
    check(checks, "object_completion_candidates_zero", manifest.get("object_completion_candidate_count") == 0, manifest.get("object_completion_candidate_count"), 0)
    check(checks, "object_completion_pose_ready_zero", manifest.get("object_completion_pose_ready_count") == 0, manifest.get("object_completion_pose_ready_count"), 0)
    check(checks, "part_pose_ready_zero", manifest.get("part_pose_ready_count") == 0, manifest.get("part_pose_ready_count"), 0)
    check(checks, "contact_factor_ready_zero", manifest.get("contact_factor_ready_rows") == 0, manifest.get("contact_factor_ready_rows"), 0)
    check(checks, "contact_ownership_ready_zero", manifest.get("contact_ownership_ready_count") == 0, manifest.get("contact_ownership_ready_count"), 0)
    check(checks, "pose_filled_through_occlusion_zero", manifest.get("pose_filled_through_occlusion_rows") == 0, manifest.get("pose_filled_through_occlusion_rows"), 0)
    check(checks, "occluder_owner_accepted_zero", manifest.get("occluder_owner_accepted_count") == 0, manifest.get("occluder_owner_accepted_count"), 0)
    check(checks, "bounded_occluder_owner_accepted_zero", manifest.get("bounded_occluder_owner_accepted_rows") == 0, manifest.get("bounded_occluder_owner_accepted_rows"), 0)
    check(checks, "occlusion_depth_order_resolved_zero", manifest.get("occlusion_depth_order_resolved_count") == 0, manifest.get("occlusion_depth_order_resolved_count"), 0)
    check(checks, "bounded_occlusion_depth_order_zero", manifest.get("bounded_occlusion_depth_order_resolved_rows") == 0, manifest.get("bounded_occlusion_depth_order_resolved_rows"), 0)
    check(checks, "generated_only_visible_part_subset_ready_zero", manifest.get("visible_part_subset_archive_ready_count") == 0, manifest.get("visible_part_subset_archive_ready_count"), 0)
    check(checks, "not_all_cases_visible_part_subset_ready", manifest.get("all_cases_visible_part_subset_archive_ready") is False, manifest.get("all_cases_visible_part_subset_archive_ready"), False)
    check(checks, "subset_summary_ready_count_matches_manifest", subset.get("visible_part_subset_archive_ready_count") == manifest.get("visible_part_subset_archive_ready_count"), subset.get("visible_part_subset_archive_ready_count"), manifest.get("visible_part_subset_archive_ready_count"))
    task5_subset = require_dict(load_json(args.visible_part_subset_root / "task5_tomato_960" / "v18_visible_part_subset_archive_report.json"), "task5 subset report")
    check(checks, "empty_task5_subset_not_ready", task5_subset.get("visible_part_subset_archive_ready") is False, task5_subset.get("visible_part_subset_archive_ready"), False)
    check(checks, "physical_schema_object_count", physical.get("object_count") == manifest.get("physical_state_schema_object_count") == 13, {"schema": physical.get("object_count"), "manifest": manifest.get("physical_state_schema_object_count")}, 13)
    check(checks, "structured_part_motion_required_three", manifest.get("structured_part_or_relative_motion_required_count") == 3, manifest.get("structured_part_or_relative_motion_required_count"), 3)
    check(checks, "wilor_hand_measurement_count", hand_baseline.get("wilor_measurement_row_count") == manifest.get("wilor_measurement_row_count") == 3361, {"hand_baseline": hand_baseline.get("wilor_measurement_row_count"), "manifest": manifest.get("wilor_measurement_row_count")}, 3361)
    check(checks, "hawor_baseline_partial_not_full_video", hand_baseline.get("hawor_measurement_row_count") == visibility.get("hawor_measurement_row_count") == manifest.get("hawor_measurement_row_count") == 182 and hand_baseline.get("hawor_available_measurement_count") == visibility.get("hawor_available_measurement_count") == manifest.get("hawor_available_measurement_count") == 132 and hand_baseline.get("hawor_motion_infill_candidate_count") == visibility.get("hawor_motion_infill_candidate_count") == manifest.get("hawor_motion_infill_candidate_count") == 50 and manifest.get("hawor_full_video_baseline_ready_all_cases") is False, {"hand_baseline": {"rows": hand_baseline.get("hawor_measurement_row_count"), "available": hand_baseline.get("hawor_available_measurement_count"), "infill": hand_baseline.get("hawor_motion_infill_candidate_count")}, "visibility": {"rows": visibility.get("hawor_measurement_row_count"), "available": visibility.get("hawor_available_measurement_count"), "infill": visibility.get("hawor_motion_infill_candidate_count")}, "manifest": {"rows": manifest.get("hawor_measurement_row_count"), "available": manifest.get("hawor_available_measurement_count"), "infill": manifest.get("hawor_motion_infill_candidate_count"), "full_video_ready": manifest.get("hawor_full_video_baseline_ready_all_cases")}}, "182 rows / 132 available / 50 infill / full-video ready false")
    check(checks, "hawor_occlusion_pose_not_accepted", hand_baseline.get("temporal_occlusion_pose_accepted_count") == visibility.get("hawor_temporal_occlusion_pose_accepted_count") == manifest.get("hawor_temporal_occlusion_pose_accepted_count") == 0 and manifest.get("pose_filled_through_occlusion_rows") == 0, {"hand_baseline": hand_baseline.get("temporal_occlusion_pose_accepted_count"), "visibility": visibility.get("hawor_temporal_occlusion_pose_accepted_count"), "manifest": manifest.get("hawor_temporal_occlusion_pose_accepted_count"), "pose_filled": manifest.get("pose_filled_through_occlusion_rows")}, 0)
    check(checks, "rtmlib_loaded_status_normalized", hand_baseline.get("rtmlib_loaded_case_count") == visibility.get("rtmlib_loaded_case_count") == manifest.get("rtmlib_loaded_case_count") == 1 and hand_baseline.get("rtmlib_frames_with_hands") == visibility.get("rtmlib_frames_with_hands") == manifest.get("rtmlib_frames_with_hands") == 1041 and hand_baseline.get("rtmlib_wilor_comparison_count") == visibility.get("rtmlib_wilor_comparison_count") == manifest.get("rtmlib_wilor_comparison_count") == 1551, {"hand_baseline": {"cases": hand_baseline.get("rtmlib_loaded_case_count"), "frames": hand_baseline.get("rtmlib_frames_with_hands"), "comparisons": hand_baseline.get("rtmlib_wilor_comparison_count")}, "visibility": {"cases": visibility.get("rtmlib_loaded_case_count"), "frames": visibility.get("rtmlib_frames_with_hands"), "comparisons": visibility.get("rtmlib_wilor_comparison_count")}, "manifest": {"cases": manifest.get("rtmlib_loaded_case_count"), "frames": manifest.get("rtmlib_frames_with_hands"), "comparisons": manifest.get("rtmlib_wilor_comparison_count")}}, "1 case / 1041 frames / 1551 comparisons")
    check(checks, "part_track_source_ready_all_cases", manifest.get("part_track_source_manifest_ready_all_cases") is True, manifest.get("part_track_source_manifest_ready_all_cases"), True)
    check(checks, "part_track_uniform_generation_ready", part_source.get("uniform_part_track_generation_ready") is True and manifest.get("uniform_part_track_generation_ready") is True, {"source": part_source.get("uniform_part_track_generation_ready"), "manifest": manifest.get("uniform_part_track_generation_ready")}, True)
    check(checks, "part_track_source_scope_generated_only", part_source.get("part_track_candidate_source_scope") == "v18_owlv2_sam2_generated_only" and part_split.get("part_track_candidate_source_scope") == "v18_owlv2_sam2_generated_only", {"source": part_source.get("part_track_candidate_source_scope"), "split": part_split.get("part_track_candidate_source_scope")}, "v18_owlv2_sam2_generated_only")
    check(checks, "part_track_source_counts_generated_only", part_source.get("root_count") == manifest.get("part_track_source_root_count") == 2 and part_source.get("usable_track_count") == manifest.get("part_track_source_usable_track_count") == 5, {"source_roots": part_source.get("root_count"), "manifest_roots": manifest.get("part_track_source_root_count"), "source_usable": part_source.get("usable_track_count"), "manifest_usable": manifest.get("part_track_source_usable_track_count")}, "2 roots, 5 usable tracks")
    check(checks, "part_split_generated_assignment_counts", part_split.get("accepted_part_track_assignment_count") == manifest.get("accepted_part_track_assignment_count") == 5, {"split": part_split.get("accepted_part_track_assignment_count"), "manifest": manifest.get("accepted_part_track_assignment_count")}, 5)
    check(checks, "part_surface_generated_counts", part_surfaces.get("surface_frame_rows") == manifest.get("part_visible_surface_frame_rows") == 809 and part_surfaces.get("total_vertices") == manifest.get("part_visible_surface_vertices") == 302902 and part_surfaces.get("total_faces") == manifest.get("part_visible_surface_faces") == 525741, {"surface_rows": part_surfaces.get("surface_frame_rows"), "manifest_rows": manifest.get("part_visible_surface_frame_rows"), "vertices": part_surfaces.get("total_vertices"), "manifest_vertices": manifest.get("part_visible_surface_vertices"), "faces": part_surfaces.get("total_faces"), "manifest_faces": manifest.get("part_visible_surface_faces")}, "809 rows / 302902 vertices / 525741 faces")
    check(checks, "part_model_probes_not_promoted", part_models.get("candidate_count") == manifest.get("part_model_candidate_count") == 0 and part_models.get("rejected_candidate_count") == manifest.get("part_model_rejected_candidate_count") == part_blockers.get("rejected_part_model_candidate_count") == manifest.get("part_object_blocker_rejected_candidate_count") == 3, {"model_candidates": part_models.get("candidate_count"), "model_non_promoted": part_models.get("rejected_candidate_count"), "manifest_non_promoted": manifest.get("part_model_rejected_candidate_count"), "blocker_non_promoted": part_blockers.get("rejected_part_model_candidate_count"), "manifest_blocker_non_promoted": manifest.get("part_object_blocker_rejected_candidate_count")}, "0 accepted / 3 non-promoted probes")
    expected_icp_states = {"surface_icp_residual_supported_visible_only_not_pose": 5}
    check(checks, "part_surface_icp_diagnostics_not_pose", part_models.get("surface_icp_probe_count") == manifest.get("part_surface_icp_probe_count") == part_blockers.get("surface_icp_probe_count") == 5 and part_models.get("surface_icp_probe_state_counts") == manifest.get("part_surface_icp_probe_state_counts") == part_blockers.get("surface_icp_probe_state_counts") == expected_icp_states and manifest.get("part_pose_ready_count") == 0 and manifest.get("object_pose_requirement_met") is False, {"model_probe_count": part_models.get("surface_icp_probe_count"), "manifest_probe_count": manifest.get("part_surface_icp_probe_count"), "blocker_probe_count": part_blockers.get("surface_icp_probe_count"), "model_states": part_models.get("surface_icp_probe_state_counts"), "manifest_states": manifest.get("part_surface_icp_probe_state_counts"), "blocker_states": part_blockers.get("surface_icp_probe_state_counts"), "part_pose_ready_count": manifest.get("part_pose_ready_count"), "object_pose_requirement_met": manifest.get("object_pose_requirement_met")}, {"count": 5, "states": expected_icp_states, "pose_ready": 0})
    expected_model_states = {"articulation_hypothesis_not_fitted": 2, "part_model_residual_probes_rejected": 1}
    expected_fit_states = {"articulation_fit_residual_rejected": 1, "articulation_fit_residual_supported_visible_center_only_not_pose": 1}
    expected_se3_pair_states = {"part_se3_not_evaluated_articulation_fit_not_supported": 1, "part_se3_surface_residual_rejected": 1}
    expected_se3_part_states = {"part_surface_se3_residual_rejected": 1, "part_surface_se3_residual_supported_visible_only_not_pose": 3}
    expected_blocker_states = {"blocked_articulation_fit_residual_rejected": 1, "blocked_part_model_residual_probes_rejected": 1, "blocked_part_se3_surface_residual_rejected": 1}
    check(checks, "part_articulation_fit_diagnostics_not_promoted", part_models.get("object_state_counts") == expected_model_states and articulation_fit.get("articulation_fit_state_counts") == expected_fit_states and part_se3.get("part_se3_pair_state_counts") == expected_se3_pair_states and part_se3.get("part_surface_se3_state_counts") == expected_se3_part_states and part_blockers.get("part_object_blocker_state_counts") == expected_blocker_states and part_models.get("articulation_hypothesis_pair_count") == part_blockers.get("articulation_hypothesis_pair_count") == manifest.get("part_articulation_hypothesis_pair_count") == articulation_fit.get("articulation_fit_probe_count") == manifest.get("articulation_fit_probe_count") == part_se3.get("part_se3_pair_count") == manifest.get("part_se3_pair_count") == 2 and articulation_fit.get("articulation_fit_supported_count") == manifest.get("articulation_fit_supported_count") == 1 and articulation_fit.get("articulation_fit_rejected_count") == manifest.get("articulation_fit_rejected_count") == 1 and manifest.get("part_se3_pair_rejected_count") == 1 and manifest.get("part_se3_surface_supported_count") == 3 and manifest.get("part_se3_surface_rejected_count") == 1 and manifest.get("articulation_model_ready_count") == 0 and manifest.get("part_pose_ready_count") == 0, {"model_states": part_models.get("object_state_counts"), "fit_states": articulation_fit.get("articulation_fit_state_counts"), "part_se3_pair_states": part_se3.get("part_se3_pair_state_counts"), "part_se3_part_states": part_se3.get("part_surface_se3_state_counts"), "blocker_states": part_blockers.get("part_object_blocker_state_counts"), "model_articulation_pairs": part_models.get("articulation_hypothesis_pair_count"), "fit_probe_count": articulation_fit.get("articulation_fit_probe_count"), "se3_pair_count": part_se3.get("part_se3_pair_count"), "manifest_se3_pair_count": manifest.get("part_se3_pair_count"), "manifest_se3_pair_rejected": manifest.get("part_se3_pair_rejected_count"), "manifest_se3_supported_parts": manifest.get("part_se3_surface_supported_count"), "manifest_se3_rejected_parts": manifest.get("part_se3_surface_rejected_count"), "articulation_ready": manifest.get("articulation_model_ready_count"), "part_pose_ready": manifest.get("part_pose_ready_count")}, {"model_states": expected_model_states, "fit_states": expected_fit_states, "se3_pair_states": expected_se3_pair_states, "se3_part_states": expected_se3_part_states, "blocker_states": expected_blocker_states, "articulation_pairs": 2, "fit_supported": 1, "fit_rejected": 1, "se3_pair_rejected": 1, "ready": 0})
    task5_se3 = require_dict(load_json(args.part_se3_root / "task5_tomato_960" / "v18_part_se3_surface_residuals_report.json"), "task5 part se3 report")
    task5_rows = [require_dict(row, "task5 se3 row") for row in task5_se3.get("rows", [])]
    faucet_rows = [row for row in task5_rows if row.get("object_id") == "object:obj_faucet_handle"]
    faucet_row = faucet_rows[0] if faucet_rows else {}
    handle_reports = [require_dict(part, "faucet part report") for part in faucet_row.get("part_surface_reports", []) if require_dict(part, "faucet part report").get("part_track_label") == "owlv2_sam2_obj_faucet_handle_handle"] if faucet_row else []
    handle_report = handle_reports[0] if handle_reports else {}
    handle_outlier_frames: list[int] = []
    if handle_report:
        for item in handle_report.get("p95_residual_outlier_frames", []):
            item_dict = require_dict(item, "handle outlier frame")
            frame_idx = item_dict.get("frame_idx")
            if not isinstance(frame_idx, int):
                raise RuntimeError("handle outlier frame_idx must be an integer")
            handle_outlier_frames.append(frame_idx)
    check(checks, "faucet_part_se3_shared_frame_outliers_preserved", faucet_row.get("pair_shared_frame_count") == 39 and handle_report.get("shared_frame_restricted") is True and handle_report.get("allowed_shared_frame_count") == 39 and handle_report.get("selected_frame_count") == 39 and handle_report.get("p95_residual_outlier_frame_count") == 8 and handle_outlier_frames == [919, 921, 922, 924, 925, 927, 928, 933] and handle_report.get("part_surface_se3_state") == "part_surface_se3_residual_rejected", {"pair_shared_frames": faucet_row.get("pair_shared_frame_count"), "shared_frame_restricted": handle_report.get("shared_frame_restricted"), "allowed_shared_frames": handle_report.get("allowed_shared_frame_count"), "selected_frames": handle_report.get("selected_frame_count"), "outlier_count": handle_report.get("p95_residual_outlier_frame_count"), "outlier_frames": handle_outlier_frames, "state": handle_report.get("part_surface_se3_state"), "components": handle_report.get("p95_residual_outlier_frame_components")}, {"shared_frames": 39, "selected_frames": 39, "outlier_count": 8, "outlier_frames": [919, 921, 922, 924, 925, 927, 928, 933], "state": "part_surface_se3_residual_rejected"})
    trash_articulation = require_dict(load_json(args.articulation_fit_root / "trash_1050" / "v18_articulation_fit_candidates_report.json"), "trash articulation fit report")
    trash_articulation_rows = [require_dict(row, "trash articulation row") for row in trash_articulation.get("rows", [])]
    offwhite_rows = [row for row in trash_articulation_rows if row.get("object_id") == "object:off_white_trash_can_first"]
    offwhite_row = offwhite_rows[0] if offwhite_rows else {}
    offwhite_worst = require_dict(offwhite_row.get("worst_residual_frames", [{}])[0], "offwhite worst residual") if offwhite_row.get("worst_residual_frames") else {}
    check(checks, "offwhite_articulation_residual_outliers_preserved", offwhite_row.get("shared_frame_count") == 259 and offwhite_row.get("articulation_fit_state") == "articulation_fit_residual_rejected" and offwhite_row.get("radial_residual_outlier_frame_count") == 37 and offwhite_row.get("plane_residual_outlier_frame_count") == 73 and offwhite_row.get("combined_residual_outlier_frame_count") == 95 and manifest.get("articulation_radial_residual_outlier_frame_count") == 37 and manifest.get("articulation_plane_residual_outlier_frame_count") == 73 and manifest.get("articulation_combined_residual_outlier_frame_count") == 95 and offwhite_worst.get("frame_idx") == 449, {"shared_frames": offwhite_row.get("shared_frame_count"), "state": offwhite_row.get("articulation_fit_state"), "radial_outliers": offwhite_row.get("radial_residual_outlier_frame_count"), "plane_outliers": offwhite_row.get("plane_residual_outlier_frame_count"), "combined_outliers": offwhite_row.get("combined_residual_outlier_frame_count"), "manifest_radial_outliers": manifest.get("articulation_radial_residual_outlier_frame_count"), "manifest_plane_outliers": manifest.get("articulation_plane_residual_outlier_frame_count"), "manifest_combined_outliers": manifest.get("articulation_combined_residual_outlier_frame_count"), "worst_frame": offwhite_worst.get("frame_idx"), "components": offwhite_row.get("combined_residual_outlier_frame_components")}, {"shared_frames": 259, "state": "articulation_fit_residual_rejected", "radial_outliers": 37, "plane_outliers": 73, "combined_outliers": 95, "worst_frame": 449})
    check(checks, "part_mask_generation_ready_for_required_objects", manifest.get("local_new_mask_generation_ready_count") == manifest.get("part_mask_acquisition_object_count") == 3, {"local_ready_count": manifest.get("local_new_mask_generation_ready_count"), "object_count": manifest.get("part_mask_acquisition_object_count")}, 3)
    check(checks, "part_mask_acquisition_no_unclassified_blockers", acquisition.get("unclassified_acquisition_blocker_count") == manifest.get("unclassified_acquisition_blocker_count") == 0, {"acquisition": acquisition.get("unclassified_acquisition_blocker_count"), "manifest": manifest.get("unclassified_acquisition_blocker_count"), "blocker_counts": acquisition.get("acquisition_blocker_counts")}, 0)
    check(
        checks,
        "owlv2_sam2_part_prompt_plan_generated_masks_ready",
        acquisition.get("environment", {}).get("promptable_segmentation_backend_available") is True
        and manifest.get("promptable_segmentation_backend_available") is True
        and acquisition.get("environment", {}).get("open_vocab_detector_backend_cached_available") is True
        and manifest.get("open_vocab_detector_backend_cached_available") is True
        and acquisition.get("environment", {}).get("open_vocab_or_referring_prompt_backend_available") is True
        and manifest.get("open_vocab_or_referring_prompt_backend_available") is True
        and acquisition.get("environment", {}).get("owlv2_sam2_part_tracks_ready") is True
        and manifest.get("model_produced_part_prompt_plan_ready") is True
        and manifest.get("local_new_mask_generation_ready") is True,
        {
            "promptable": manifest.get("promptable_segmentation_backend_available"),
            "open_vocab_detector_cached": manifest.get("open_vocab_detector_backend_cached_available"),
            "open_vocab_or_referring": manifest.get("open_vocab_or_referring_prompt_backend_available"),
            "owlv2_sam2_ready": acquisition.get("environment", {}).get("owlv2_sam2_part_tracks_ready"),
            "part_prompt_plan_ready": manifest.get("model_produced_part_prompt_plan_ready"),
            "local_ready": manifest.get("local_new_mask_generation_ready"),
        },
        "promptable true, open-vocab detector cached true, OWLv2->SAM2 tracks ready, local generation true",
    )
    check(checks, "mask_evidence_created_matches_generated_tracks", acquisition.get("mask_evidence_created_count") == manifest.get("mask_evidence_created_count") == 5, {"acquisition": acquisition.get("mask_evidence_created_count"), "manifest": manifest.get("mask_evidence_created_count")}, 5)
    check(
        checks,
        "sam_promptable_proposals_not_accepted_part_tracks",
        sam_promptable.get("saved_promptable_proposal_mask_count") == manifest.get("sam_promptable_saved_proposal_mask_count")
        and sam_promptable.get("accepted_part_track_count") == manifest.get("sam_promptable_accepted_part_track_count") == 0
        and sam_promptable.get("mask_evidence_created_count") == manifest.get("sam_promptable_mask_evidence_created_count") == 0,
        {
            "saved_proposals": manifest.get("sam_promptable_saved_proposal_mask_count"),
            "accepted_part_tracks": manifest.get("sam_promptable_accepted_part_track_count"),
            "mask_evidence_created": manifest.get("sam_promptable_mask_evidence_created_count"),
        },
        "saved proposals allowed, accepted tracks/mask evidence zero",
    )
    check(
        checks,
        "part_mask_promotion_gate_blocks_promptable_proposals",
        promotion_gate.get("saved_promptable_proposal_mask_count") == manifest.get("part_mask_promotion_gate_saved_proposal_mask_count") == manifest.get("sam_promptable_saved_proposal_mask_count")
        and promotion_gate.get("promoted_part_track_count") == manifest.get("part_mask_promotion_gate_promoted_part_track_count") == 0
        and promotion_gate.get("mask_evidence_created_count") == manifest.get("part_mask_promotion_gate_mask_evidence_created_count") == 0,
        {
            "gate_saved_proposals": promotion_gate.get("saved_promptable_proposal_mask_count"),
            "manifest_saved_proposals": manifest.get("part_mask_promotion_gate_saved_proposal_mask_count"),
            "promoted_part_tracks": manifest.get("part_mask_promotion_gate_promoted_part_track_count"),
            "mask_evidence_created": manifest.get("part_mask_promotion_gate_mask_evidence_created_count"),
        },
        "saved proposals match; promoted tracks/mask evidence zero",
    )
    check(checks, "occlusion_candidate_count_matches", occlusion.get("candidate_owner_row_count") == manifest.get("occlusion_candidate_owner_row_count") == bounded.get("occlusion_owner_candidate_rows") == manifest.get("bounded_occlusion_owner_candidate_rows"), {"occlusion": occlusion.get("candidate_owner_row_count"), "bounded": bounded.get("occlusion_owner_candidate_rows"), "manifest": manifest.get("occlusion_candidate_owner_row_count")}, 116)
    check(
        checks,
        "occlusion_depth_candidate_pairs_match",
        occlusion_depth.get("candidate_pair_count") == manifest.get("occlusion_depth_evidence_candidate_pair_rows") == bounded.get("occlusion_depth_evidence_candidate_pair_rows"),
        {
            "depth_summary": occlusion_depth.get("candidate_pair_count"),
            "manifest": manifest.get("occlusion_depth_evidence_candidate_pair_rows"),
            "bounded": bounded.get("occlusion_depth_evidence_candidate_pair_rows"),
        },
        "all equal",
    )
    depth_partition_total = sum(
        int(manifest.get(key) or 0)
        for key in [
            "occlusion_depth_evidence_foreground_support_pair_rows",
            "occlusion_depth_evidence_foreground_contradiction_pair_rows",
            "occlusion_depth_evidence_metric_compatible_pair_rows",
            "occlusion_depth_evidence_insufficient_pair_rows",
        ]
    )
    check(checks, "occlusion_depth_evidence_partition", depth_partition_total == manifest.get("occlusion_depth_evidence_candidate_pair_rows"), depth_partition_total, manifest.get("occlusion_depth_evidence_candidate_pair_rows"))
    check(
        checks,
        "occlusion_depth_evidence_no_acceptance",
        occlusion_depth.get("occluder_owner_accepted_count") == manifest.get("occlusion_depth_evidence_owner_accepted_count") == 0
        and occlusion_depth.get("depth_order_resolved_count") == manifest.get("occlusion_depth_evidence_depth_order_resolved_count") == 0
        and occlusion_depth.get("pose_filled_through_occlusion_rows") == 0,
        {
            "depth_owner_accepted": occlusion_depth.get("occluder_owner_accepted_count"),
            "manifest_owner_accepted": manifest.get("occlusion_depth_evidence_owner_accepted_count"),
            "depth_resolved": occlusion_depth.get("depth_order_resolved_count"),
            "manifest_resolved": manifest.get("occlusion_depth_evidence_depth_order_resolved_count"),
            "pose_filled": occlusion_depth.get("pose_filled_through_occlusion_rows"),
        },
        "all zero",
    )
    check(checks, "runtime_pipeline_success", runtime.get("pipeline_success") is True, runtime.get("pipeline_success"), True)
    check(checks, "runtime_stage_count_matches_manifest", runtime.get("stage_count") == manifest.get("cached_evidence_to_status_stage_count"), {"runtime": runtime.get("stage_count"), "manifest": manifest.get("cached_evidence_to_status_stage_count")}, manifest.get("cached_evidence_to_status_stage_count"))
    check(checks, "cached_runtime_measured", manifest.get("cached_evidence_to_status_runtime_measured") is True, manifest.get("cached_evidence_to_status_runtime_measured"), True)
    check(checks, "fresh_raw_status_runtime_not_measured", manifest.get("fresh_raw_video_to_status_runtime_measured") is False, manifest.get("fresh_raw_video_to_status_runtime_measured"), False)
    check(checks, "fresh_raw_final_runtime_not_measured", manifest.get("fresh_raw_video_to_final_pose_runtime_measured") is False, manifest.get("fresh_raw_video_to_final_pose_runtime_measured"), False)
    ratio = manifest.get("cached_evidence_to_status_elapsed_to_video_ratio")
    check(checks, "cached_runtime_under_10x", isinstance(ratio, (int, float)) and float(ratio) < 10.0, ratio, "<10")
    for case in args.cases:
        row = nested_case(manifest, case)
        frame_qc = require_dict(row.get("frame_count_qc"), f"{case} frame qc")
        duration_qc = require_dict(row.get("duration_qc"), f"{case} duration qc")
        check(checks, f"{case}_frame_count_match", frame_qc.get("all_match_raw") is True, frame_qc.get("all_match_raw"), True)
        check(checks, f"{case}_duration_match", duration_qc.get("all_durations_match_raw") is True, duration_qc.get("all_durations_match_raw"), True)
        check(checks, f"{case}_fps_match", duration_qc.get("all_fps_match_raw") is True, duration_qc.get("all_fps_match_raw"), True)

    failed = [row for row in checks if not row["passed"] and row.get("severity") == "required"]
    report = {
        "method": "audit_v18_status_invariants",
        "status": STATUS,
        "claim": CLAIM,
        "build_elapsed_s": time.perf_counter() - start,
        "audit_passed": not failed,
        "required_check_count": sum(1 for row in checks if row.get("severity") == "required"),
        "failed_required_check_count": len(failed),
        "checks": checks,
        "sources": {
            "status_manifest": str(manifest_path),
            "runtime_report": str(runtime_path),
            "hand_baseline_summary": str(hand_baseline_path),
            "visibility_occlusion_summary": str(visibility_summary_path),
            "visible_part_subset_summary": str(subset_summary_path),
            "occlusion_candidates_summary": str(occlusion_summary_path),
            "occlusion_depth_order_evidence_summary": str(occlusion_depth_summary_path),
            "bounded_state_summary": str(bounded_summary_path),
            "physical_schema_summary": str(physical_schema_path),
            "part_track_source_summary": str(part_source_path),
            "part_split_evidence_summary": str(part_split_path),
            "part_visible_surfaces_summary": str(part_surfaces_path),
            "part_model_candidates_summary": str(part_models_path),
            "articulation_fit_summary": str(articulation_fit_path),
            "part_se3_surface_residual_summary": str(part_se3_path),
            "part_object_blockers_summary": str(part_blockers_path),
            "part_mask_acquisition_summary": str(acquisition_path),
            "sam_promptable_proposals_summary": str(sam_promptable_path),
            "part_mask_promotion_gate_summary": str(promotion_gate_path),
        },
    }
    write_json(args.output_root / "v18_status_invariant_audit_report.json", report)
    if failed:
        print(json.dumps(report, indent=2))
        raise RuntimeError(f"V18 invariant audit failed {len(failed)} required checks")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-manifest-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_deliverable_manifest"))
    parser.add_argument("--measured-runtime-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_measured_status_pipeline_runtime"))
    parser.add_argument("--hand-baseline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--visibility-occlusion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--visible-part-subset-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_part_subset_archive"))
    parser.add_argument("--occlusion-owner-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_candidates"))
    parser.add_argument("--occlusion-depth-evidence-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence"))
    parser.add_argument("--bounded-state-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--part-track-source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_track_source_manifest"))
    parser.add_argument("--part-split-evidence-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--part-visible-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--articulation-fit-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--part-se3-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_se3_surface_residuals"))
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--part-mask-acquisition-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan"))
    parser.add_argument("--sam-promptable-proposals-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals"))
    parser.add_argument("--part-mask-promotion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_promotion_gate"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_invariant_audit"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    try:
        build(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
