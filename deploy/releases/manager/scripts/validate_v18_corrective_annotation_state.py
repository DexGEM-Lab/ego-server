#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_case(case: str, root: Path, expected_root: Path, failures: list[str]) -> dict[str, Any]:
    path = root / case / "annotations_v18_corrective_state.json"
    require(path.exists(), f"{case}: missing {path}", failures)
    if not path.exists():
        return {"case": case, "status": "missing"}
    ann = load_json(path)
    src_path = expected_root / case / "annotations_v18_full.json"
    require(src_path.exists(), f"{case}: missing source annotation {src_path}", failures)
    src = load_json(src_path) if src_path.exists() else {}
    frames = ann.get("frames", [])
    counts = ann.get("counts", {}) if isinstance(ann.get("counts"), dict) else {}
    expected_count = src.get("frame_count")
    require(isinstance(frames, list), f"{case}: frames is not a list", failures)
    require(len(frames) == expected_count, f"{case}: frame count {len(frames)} != source {expected_count}", failures)
    require(ann.get("status") == "corrective_state_delta_not_full_v18_closure", f"{case}: incorrect scoped status", failures)
    forbidden_contact_strings = ["accepted_contact_owner_by", "accepted_contact_owner_before", "accepted_before_nonpenetration_veto", "graph_accepted"]
    for rel in [
        f"{case}/contact_acceptance_audit/v18_contact_acceptance_audit_report.json",
        f"{case}/contact_nonpenetration_state/v18_contact_nonpenetration_state_report.json",
        f"{case}/nonpenetration_repair_proposal/v18_nonpenetration_repair_proposal_report.json",
    ]:
        report_path = root / rel
        if report_path.exists():
            text = report_path.read_text(encoding="utf-8")
            for forbidden in forbidden_contact_strings:
                require(forbidden not in text, f"{case}: stale contact acceptance string {forbidden} remains in {rel}", failures)
    require(int(counts.get("graph_shifted_mano_states", 0)) > 0, f"{case}: no graph-shifted MANO states", failures)
    require(int(counts.get("graph_hand_states", 0)) == int(counts.get("graph_shifted_mano_states", -1)), f"{case}: graph hand count != shifted MANO count", failures)
    require(int(counts.get("graph_object_se3_states", 0)) > 0, f"{case}: no graph object SE3 states", failures)
    require(int(counts.get("frame_local_visible_surface_states", 0)) > 0, f"{case}: no frame-local visible surface states", failures)
    stable_without_uncertainty = 0
    stable_without_residual = 0
    graph_pose_without_uncertainty = 0
    graph_pose_marked_accepted = 0
    nonrigid_graph_pose_best_current = 0
    smoothed_marked_accepted_3d = 0
    smoothed_without_uncertainty = 0
    smoothed_promoted_best_current = 0
    smoothed_applied_gate_violation = 0
    repair_bad_semantics = 0
    repair_postcheck_mismatch = 0
    occlusion_audit_bad_semantics = 0
    occlusion_audit_strict_or_accepted = 0
    contact_audit_bad_semantics = 0
    contact_audit_strict = 0
    stale_accepted_contact_semantics = 0
    bridge_quality_bad_semantics = 0
    bridge_quality_promoted_best_current = 0
    bridge_quality_rows_seen = 0
    for frame in frames if isinstance(frames, list) else []:
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            bridge_quality = hand.get("hawor_bridge_quality_candidate", {}) if isinstance(hand.get("hawor_bridge_quality_candidate"), dict) else {}
            if bridge_quality:
                bridge_quality_rows_seen += 1
                if bridge_quality.get("accepted_v18_hawor_foundation") is not False:
                    bridge_quality_bad_semantics += 1
                if bridge_quality.get("accepted_metric_hand_state") is not False:
                    bridge_quality_bad_semantics += 1
                if bridge_quality.get("accepted_contact_or_occlusion_input") is not False:
                    bridge_quality_bad_semantics += 1
                if bridge_quality.get("state_role") != "HaWoR_bridge_candidate_quality_evidence_not_foundation_acceptance_not_downstream_physics":
                    bridge_quality_bad_semantics += 1
                if "accepted" in str(bridge_quality.get("status", "")):
                    bridge_quality_bad_semantics += 1
                if str(hand.get("best_current_state", "")).startswith("hawor_bridge"):
                    bridge_quality_promoted_best_current += 1
            smoothed = hand.get("temporal_smoothed_mano2d_state", {}) if isinstance(hand.get("temporal_smoothed_mano2d_state"), dict) else {}
            if smoothed:
                if smoothed.get("accepted_3d_mano_pose") is not False:
                    smoothed_marked_accepted_3d += 1
                uncertainty = hand.get("uncertainty") if isinstance(hand.get("uncertainty"), list) else []
                if "temporal_smoothed_mano2d_is_not_3d_mano_optimization_or_physical_pose_or_best_current_state" not in uncertainty:
                    smoothed_without_uncertainty += 1
                if str(hand.get("best_current_state", "")).startswith("temporal_smoothed"):
                    smoothed_promoted_best_current += 1
                if smoothed.get("temporal_filter_applied") is True:
                    if float(smoothed.get("max_joint_shift_from_graph_shifted_input_px") or 0.0) > 120.0:
                        smoothed_applied_gate_violation += 1
                    if float(smoothed.get("centroid_shift_from_graph_shifted_input_px") or 0.0) > 80.0:
                        smoothed_applied_gate_violation += 1
                    if float(smoothed.get("root_shift_from_graph_shifted_input_px") or 0.0) > 120.0:
                        smoothed_applied_gate_violation += 1
                    if int(smoothed.get("output_out_of_source_frame_joint_count") or 0) != 0:
                        smoothed_applied_gate_violation += 1
            audit_rows = hand.get("occlusion_owner_acceptance_audit") if isinstance(hand.get("occlusion_owner_acceptance_audit"), list) else []
            for audit in audit_rows:
                if not isinstance(audit, dict):
                    continue
                if audit.get("state_role") != "occlusion_owner_acceptance_audit_not_assignment_not_pose_fill":
                    occlusion_audit_bad_semantics += 1
                if audit.get("evidence_scope") != "acceptance_audit_only_not_owner_assignment_or_pose_fill":
                    occlusion_audit_bad_semantics += 1
                if audit.get("strict_promotable_owner") is True or audit.get("accepted_occlusion_owner") is True:
                    occlusion_audit_strict_or_accepted += 1
            contact_audit_rows = hand.get("contact_acceptance_audit") if isinstance(hand.get("contact_acceptance_audit"), list) else []
            for audit in contact_audit_rows:
                if not isinstance(audit, dict):
                    continue
                if audit.get("state_role") != "contact_acceptance_audit_not_contact_assignment_not_complete_nonpenetration":
                    contact_audit_bad_semantics += 1
                if "accepted_contact_owner_before_physical_veto" in audit:
                    stale_accepted_contact_semantics += 1
                if audit.get("strict_promotable_contact") is True:
                    contact_audit_strict += 1
            contact_np = hand.get("contact_nonpenetration_state", {}) if isinstance(hand.get("contact_nonpenetration_state"), dict) else {}
            if "accepted_contact_owner_before_nonpenetration_veto" in contact_np:
                stale_accepted_contact_semantics += 1
            repair = hand.get("nonpenetration_repair_proposal", {}) if isinstance(hand.get("nonpenetration_repair_proposal"), dict) else {}
            if repair:
                if repair.get("applied_to_annotation") is not False or repair.get("proposal_complete_nonpenetration") is not False:
                    repair_bad_semantics += 1
                if repair.get("state_role") != "diagnostic_v16_local_translation_candidate_not_applied_not_complete_sdf_not_current_v18_hand_state":
                    repair_bad_semantics += 1
                if "v16" not in str(repair.get("diagnostic_geometry_basis")):
                    repair_bad_semantics += 1
                status = str(repair.get("status"))
                post_passed = repair.get("post_translation_local_metric_passed")
                if status.endswith("postcheck_pass") and post_passed is not True:
                    repair_postcheck_mismatch += 1
                if status.endswith("postcheck_failed") and post_passed is not False:
                    repair_postcheck_mismatch += 1
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            uncertainty = obj.get("uncertainty") if isinstance(obj.get("uncertainty"), list) else []
            graph_pose = obj.get("graph_object_se3", {}) if isinstance(obj.get("graph_object_se3"), dict) else {}
            if graph_pose:
                if not uncertainty:
                    graph_pose_without_uncertainty += 1
                if graph_pose.get("accepted_physical_object_pose") is not False:
                    graph_pose_marked_accepted += 1
                if obj.get("physical_state_candidate") != "rigid" and obj.get("best_current_state") == "graph_object_se3_observation":
                    nonrigid_graph_pose_best_current += 1
            attempt = obj.get("generic_rigid_se3_attempt", {}) if isinstance(obj.get("generic_rigid_se3_attempt"), dict) else {}
            if attempt.get("stable_pose6_world_from_object") is not None:
                if not uncertainty:
                    stable_without_uncertainty += 1
                if not isinstance(attempt.get("residual_check"), dict):
                    stable_without_residual += 1
    require(graph_pose_without_uncertainty == 0, f"{case}: graph object SE3 rows without uncertainty: {graph_pose_without_uncertainty}", failures)
    require(graph_pose_marked_accepted == 0, f"{case}: graph object SE3 rows missing accepted_physical_object_pose=false: {graph_pose_marked_accepted}", failures)
    require(nonrigid_graph_pose_best_current == 0, f"{case}: non-rigid graph SE3 rows still marked best-current graph_object_se3_observation: {nonrigid_graph_pose_best_current}", failures)
    require(stable_without_uncertainty == 0, f"{case}: stable rigid rows without uncertainty: {stable_without_uncertainty}", failures)
    require(stable_without_residual == 0, f"{case}: stable rigid rows without residual check: {stable_without_residual}", failures)
    require(ann.get("foundational_mano_state_valid") is False, f"{case}: foundational MANO state should remain invalid until full metric MANO is established", failures)
    require(ann.get("v18_physical_pipeline_valid_without_further_hand_work") is False, f"{case}: V18 physical pipeline should be invalid without further hand work", failures)
    mano_blockers = ann.get("mano_foundation_blocking_reasons") if isinstance(ann.get("mano_foundation_blocking_reasons"), list) else []
    require("current_v18_full_annotations_drop_mano_vertices" in mano_blockers, f"{case}: missing MANO blocker for dropped vertices", failures)
    require("current_v18_full_annotations_drop_mano_parameters" in mano_blockers, f"{case}: missing MANO blocker for dropped parameters", failures)
    require("recovered_wilor_mano_not_full_two_hand_timeline" in mano_blockers, f"{case}: missing MANO blocker for incomplete WiLoR timeline", failures)
    require("recovered_wilor_virtual_camera_not_metric_world_aligned" in mano_blockers, f"{case}: missing MANO blocker for WiLoR virtual-camera metric-world misalignment", failures)
    require(ann.get("geometry_coverage_audit_stable_pose_source") == "recomputed_from_source_annotations_factor_graph_object_se3_with_same_stable_prior_as_corrective_annotation_builder", f"{case}: geometry coverage stable-pose source is stale or missing", failures)
    geometry_summaries = ann.get("geometry_coverage_audit_object_summaries", {}) if isinstance(ann.get("geometry_coverage_audit_object_summaries"), dict) else {}
    bad_geometry_completion_flags = 0
    for summary in geometry_summaries.values():
        if isinstance(summary, dict):
            if summary.get("accepted_complete_geometry") is not False or summary.get("object_geometry_complete") is not False:
                bad_geometry_completion_flags += 1
            if summary.get("alignment_pose_scope") != "uses_uncertain_stable_rigid_prior_recomputed_from_source_factor_graph_for_diagnostic_alignment_not_accepted_pose":
                bad_geometry_completion_flags += 1
    require(bad_geometry_completion_flags == 0, f"{case}: geometry coverage summaries with accepted/completion or stale-pose semantics: {bad_geometry_completion_flags}", failures)
    require(smoothed_marked_accepted_3d == 0, f"{case}: temporal smoothed MANO2D rows marked accepted 3D: {smoothed_marked_accepted_3d}", failures)
    require(smoothed_without_uncertainty == 0, f"{case}: temporal smoothed MANO2D rows without scope uncertainty: {smoothed_without_uncertainty}", failures)
    require(smoothed_promoted_best_current == 0, f"{case}: temporal smoothed MANO2D rows promoted to best_current_state: {smoothed_promoted_best_current}", failures)
    require(smoothed_applied_gate_violation == 0, f"{case}: applied temporal smoothing rows violating anchor/bounds gates: {smoothed_applied_gate_violation}", failures)
    require(repair_bad_semantics == 0, f"{case}: repair candidate rows with applied/complete/missing-V16 semantics: {repair_bad_semantics}", failures)
    require(repair_postcheck_mismatch == 0, f"{case}: repair candidate postcheck status mismatch: {repair_postcheck_mismatch}", failures)
    require(occlusion_audit_bad_semantics == 0, f"{case}: occlusion acceptance audit rows with assignment/pose-fill semantics: {occlusion_audit_bad_semantics}", failures)
    require(occlusion_audit_strict_or_accepted == 0, f"{case}: occlusion audit rows unexpectedly strict-promotable or accepted: {occlusion_audit_strict_or_accepted}", failures)
    require(contact_audit_bad_semantics == 0, f"{case}: contact acceptance audit rows with assignment/complete-nonpenetration semantics: {contact_audit_bad_semantics}", failures)
    require(stale_accepted_contact_semantics == 0, f"{case}: stale accepted-contact pre-veto fields remain: {stale_accepted_contact_semantics}", failures)
    require(contact_audit_strict == 0, f"{case}: contact audit rows unexpectedly strict-promotable: {contact_audit_strict}", failures)
    require(bridge_quality_bad_semantics == 0, f"{case}: HaWoR bridge quality rows with acceptance/downstream semantics: {bridge_quality_bad_semantics}", failures)
    require(bridge_quality_promoted_best_current == 0, f"{case}: HaWoR bridge quality rows promoted to best_current_state: {bridge_quality_promoted_best_current}", failures)
    if case == "trash_1050":
        require(int(counts.get("hawor_prior_states", 0)) == 182, f"{case}: expected 182 HaWoR prior states", failures)
        require(int(counts.get("mano_foundation_wilor_virtual_candidate_rows", 0)) == 1617, f"{case}: expected 1617 recovered WiLoR MANO virtual-camera raw candidate rows", failures)
        require(int(counts.get("mano_foundation_wilor_virtual_unique_frame_side_rows", 0)) == 1601, f"{case}: expected 1601 recovered WiLoR MANO virtual-camera unique frame-side rows", failures)
        require(int(counts.get("mano_foundation_hawor_world_rows", 0)) == 182, f"{case}: expected 182 HaWoR MANO world rows", failures)
        require(int(counts.get("hawor_bridge_quality_candidate_rows", 0)) == 2098, f"{case}: expected 2098 HaWoR bridge quality candidate rows", failures)
        require(int(counts.get("hawor_bridge_projection_supported_candidate_rows", 0)) == 1372, f"{case}: expected 1372 HaWoR bridge projection-supported candidate rows", failures)
        require(int(counts.get("hawor_bridge_quality::projection_supported_visible_hawor_bridge_candidate", 0)) == 1365, f"{case}: expected 1365 visible projection-supported bridge candidates", failures)
        require(int(counts.get("hawor_bridge_quality::projection_supported_nonvisible_hawor_bridge_candidate", 0)) == 7, f"{case}: expected 7 nonvisible projection-supported bridge candidates", failures)
        require(int(counts.get("hawor_bridge_quality::residual_tail_hawor_out_of_frame_or_visibility_conflict", 0)) == 59, f"{case}: expected 59 residual-tail bridge candidates", failures)
        require(bridge_quality_rows_seen == 2098, f"{case}: expected 2098 bridge quality rows seen, got {bridge_quality_rows_seen}", failures)
        require(ann.get("hawor_bridge_quality_status") == "hawor_bridge_quality_candidate_state_built_not_accepted", f"{case}: unexpected bridge quality status metadata", failures)
        require(ann.get("hawor_bridge_quality_accepted_v18_hawor_foundation") is False, f"{case}: bridge quality foundation should not be accepted", failures)
        require(ann.get("hawor_bridge_quality_v18_physical_hand_state_valid") is False, f"{case}: bridge quality physical hand state should be false", failures)
        require(int(ann.get("mano_foundation_wilor_virtual_candidate_rows") or 0) == 1617, f"{case}: annotation metadata expected 1617 recovered WiLoR MANO virtual-camera raw candidate rows", failures)
        require(int(ann.get("mano_foundation_wilor_virtual_unique_frame_side_rows") or 0) == 1601, f"{case}: annotation metadata expected 1601 recovered WiLoR MANO virtual-camera unique frame-side rows", failures)
        require(ann.get("mano_foundation_wilor_metric_world_alignment_valid") is False, f"{case}: WiLoR virtual-camera MANO should not be marked metric-world aligned", failures)
        require(int(ann.get("mano_foundation_hawor_world_rows") or 0) == 182, f"{case}: annotation metadata expected 182 HaWoR MANO world rows", failures)
        require(int(counts.get("geometry_coverage_audit_objects", 0)) == 1, f"{case}: expected 1 geometry coverage audit object", failures)
        require(int(counts.get("geometry_coverage::broad_visible_coverage_but_hidden_geometry_still_unresolved", 0)) == 1, f"{case}: expected pink lid broad-visible unresolved coverage status", failures)
        require(int(counts.get("temporal_smoothed_mano2d_states", 0)) == 1901, f"{case}: expected 1901 temporal smoothed MANO2D states", failures)
        require(int(counts.get("pose_fill_best_effort_states", 0)) == 50, f"{case}: expected 50 HaWoR motion-infill pose-fill best-effort states", failures)
        require(int(counts.get("frame_local_visible_surface_states", 0)) == 232, f"{case}: expected 232 visible surface states for the rigid lid", failures)
        require(int(counts.get("occlusion_owner_best_effort_states", 0)) == 64, f"{case}: expected 64 tentative occlusion owner rows", failures)
        require(int(counts.get("occlusion_owner_acceptance_audit_rows", 0)) == 165, f"{case}: expected 165 occlusion acceptance audit rows", failures)
        require(int(counts.get("occlusion_owner_acceptance::direct_depth_mesh_support_not_temporal_selected", 0)) == 1, f"{case}: expected 1 direct-depth mesh row not selected", failures)
        require(int(counts.get("occlusion_owner_acceptance::foreground_depth_contradicts_candidate", 0)) == 27, f"{case}: expected 27 foreground-depth contradiction rows", failures)
        require(int(counts.get("occlusion_owner_acceptance::not_selected_no_direct_depth_support", 0)) == 73, f"{case}: expected 73 not-selected/no-direct-depth rows", failures)
        require(int(counts.get("occlusion_owner_acceptance::temporal_selected_mesh_margin_supported_depth_missing", 0)) == 24, f"{case}: expected 24 temporal-selected mesh/margin supported depth-missing rows", failures)
        require(int(counts.get("occlusion_owner_acceptance::temporal_selected_mesh_support_low_or_missing", 0)) == 15, f"{case}: expected 15 temporal-selected mesh-low rows", failures)
        require(int(counts.get("occlusion_owner_acceptance::temporal_selected_mesh_supported_margin_low", 0)) == 25, f"{case}: expected 25 temporal-selected margin-low rows", failures)
        require(ann.get("occlusion_owner_acceptance_audit_strict_promotable_rows") == 0, f"{case}: expected zero strict-promotable occlusion audit rows", failures)
        require(ann.get("occlusion_owner_selected_rows") == 64, f"{case}: selected owner row metadata should be 64", failures)
        require(ann.get("contact_graph_selected_rows") == 371, f"{case}: expected 371 selected contact rows", failures)
        require(int(counts.get("contact_acceptance_audit_rows", 0)) == 371, f"{case}: expected 371 contact acceptance audit rows", failures)
        require(int(counts.get("contact_acceptance::source_graph_candidate_local_penetration_veto", 0)) == 293, f"{case}: expected 293 contact audit penetration veto rows", failures)
        require(int(counts.get("contact_acceptance::source_graph_candidate_local_no_penetration_open_mesh_not_strict", 0)) == 2, f"{case}: expected 2 contact audit local-no-penetration open-mesh rows", failures)
        require(int(counts.get("contact_acceptance::graph_selected_not_contact_accepted", 0)) == 76, f"{case}: expected 76 contact audit graph-selected-not-accepted rows", failures)
        require(ann.get("contact_acceptance_audit_strict_promotable_rows") == 0, f"{case}: expected zero strict-promotable contact audit rows", failures)
        require(int(counts.get("contact_nonpenetration_states", 0)) == 371, f"{case}: expected 371 contact/nonpenetration states", failures)
        require(int(counts.get("contact_nonpenetration::source_graph_candidate_but_local_penetration_veto", 0)) == 293, f"{case}: expected 293 local penetration contact veto states", failures)
        require(int(counts.get("contact_nonpenetration::source_graph_candidate_no_local_penetration_flag", 0)) == 2, f"{case}: expected 2 source graph contact-candidate states without local penetration flag", failures)
        require(int(counts.get("nonpenetration_repair_proposal_states", 0)) == 293, f"{case}: expected 293 nonpenetration repair proposal states", failures)
        require(int(counts.get("nonpenetration_repair::large_local_translation_required", 0)) == 235, f"{case}: expected 235 large local translation states", failures)
        require(int(counts.get("nonpenetration_repair::translation_candidate_unreliable_incoherent_normals", 0)) == 54, f"{case}: expected 54 incoherent-normal translation candidates", failures)
        require(int(counts.get("nonpenetration_repair::small_coherent_translation_candidate_local_postcheck_pass", 0)) == 2, f"{case}: expected 2 small coherent candidates passing local postcheck", failures)
        require(int(counts.get("nonpenetration_repair::small_coherent_translation_candidate_local_postcheck_failed", 0)) == 2, f"{case}: expected 2 small coherent candidates failing local postcheck", failures)
        require(int(counts.get("rigid_residual_checked_states", 0)) == 232, f"{case}: expected 232 rigid residual checked states", failures)
        require(int(counts.get("rigid_residual::bidirectional_residual_supported_uncertain", 0)) == 150, f"{case}: expected 150 bidirectional residual supported states", failures)
        require(int(counts.get("rigid_residual::visible_supported_but_fused_overspread", 0)) == 82, f"{case}: expected 82 fused-overspread residual states", failures)
        require("object:pink_lid_trash_can_second" in ann.get("rigid_candidate_ids", []), f"{case}: missing pink lid rigid candidate", failures)
    if case == "task5_tomato_960":
        require(ann.get("hawor_measurement_rows") == 0, f"{case}: expected zero HaWoR measurement rows", failures)
        require(int(counts.get("pose_fill_best_effort_states", 0)) == 0, f"{case}: expected zero pose-fill best-effort rows without HaWoR", failures)
        require(ann.get("occlusion_owner_selected_rows") == 0, f"{case}: expected zero selected tentative occlusion owner rows", failures)
        require(int(counts.get("occlusion_owner_acceptance_audit_rows", 0)) == 1, f"{case}: expected 1 occlusion acceptance audit row", failures)
        require(int(counts.get("occlusion_owner_acceptance::not_selected_no_direct_depth_support", 0)) == 1, f"{case}: expected 1 not-selected/no-direct-depth audit row", failures)
        require(ann.get("occlusion_owner_acceptance_audit_strict_promotable_rows") == 0, f"{case}: expected zero strict-promotable occlusion audit rows", failures)
        require(int(counts.get("hawor_provisioning_failed_hand_states", 0)) == 1920, f"{case}: expected 1920 HaWoR provisioning-failure hand states", failures)
        require(int(counts.get("mano_foundation_wilor_virtual_candidate_rows", 0)) == 1744, f"{case}: expected 1744 recovered WiLoR MANO virtual-camera raw candidate rows", failures)
        require(int(counts.get("mano_foundation_wilor_virtual_unique_frame_side_rows", 0)) == 1733, f"{case}: expected 1733 recovered WiLoR MANO virtual-camera unique frame-side rows", failures)
        require(int(counts.get("mano_foundation_hawor_world_rows", 0)) == 0, f"{case}: expected 0 HaWoR MANO world rows", failures)
        require(int(counts.get("hawor_bridge_quality_candidate_rows", 0)) == 0, f"{case}: expected 0 HaWoR bridge quality candidate rows", failures)
        require(bridge_quality_rows_seen == 0, f"{case}: expected 0 bridge quality rows seen, got {bridge_quality_rows_seen}", failures)
        require(ann.get("hawor_bridge_quality_status") == "blocked_no_hawor_bridge_candidates_for_case", f"{case}: unexpected bridge quality status metadata", failures)
        require(ann.get("hawor_bridge_quality_accepted_v18_hawor_foundation") is False, f"{case}: bridge quality foundation should not be accepted", failures)
        require(ann.get("hawor_bridge_quality_v18_physical_hand_state_valid") is False, f"{case}: bridge quality physical hand state should be false", failures)
        require(int(ann.get("mano_foundation_wilor_virtual_candidate_rows") or 0) == 1744, f"{case}: annotation metadata expected 1744 recovered WiLoR MANO virtual-camera raw candidate rows", failures)
        require(int(ann.get("mano_foundation_wilor_virtual_unique_frame_side_rows") or 0) == 1733, f"{case}: annotation metadata expected 1733 recovered WiLoR MANO virtual-camera unique frame-side rows", failures)
        require(ann.get("mano_foundation_wilor_metric_world_alignment_valid") is False, f"{case}: WiLoR virtual-camera MANO should not be marked metric-world aligned", failures)
        require(int(ann.get("mano_foundation_hawor_world_rows") or 0) == 0, f"{case}: annotation metadata expected 0 HaWoR MANO world rows", failures)
        require("hawor_missing_for_case" in mano_blockers, f"{case}: missing task5 HaWoR MANO blocker", failures)
        require(int(counts.get("geometry_coverage_audit_objects", 0)) == 2, f"{case}: expected 2 geometry coverage audit objects", failures)
        require(int(counts.get("geometry_coverage::coverage_confounded_by_pose_alignment_overspread", 0)) == 1, f"{case}: expected tomato coverage-confounded status", failures)
        require(int(counts.get("geometry_coverage::insufficient_view_count_for_geometry_completion_claim", 0)) == 1, f"{case}: expected plastic container insufficient-view status", failures)
        require(int(counts.get("temporal_smoothed_mano2d_states", 0)) == 1859, f"{case}: expected 1859 temporal smoothed MANO2D states", failures)
        require(int(counts.get("frame_local_visible_surface_states", 0)) == 449, f"{case}: expected 449 visible surface states for rigid candidates", failures)
        require(ann.get("contact_graph_selected_rows") == 808, f"{case}: expected 808 selected contact rows", failures)
        require(int(counts.get("contact_acceptance_audit_rows", 0)) == 808, f"{case}: expected 808 contact acceptance audit rows", failures)
        require(int(counts.get("contact_acceptance::source_graph_candidate_local_penetration_veto", 0)) == 705, f"{case}: expected 705 contact audit penetration veto rows", failures)
        require(int(counts.get("contact_acceptance::source_graph_candidate_local_no_penetration_open_mesh_not_strict", 0)) == 16, f"{case}: expected 16 contact audit local-no-penetration open-mesh rows", failures)
        require(int(counts.get("contact_acceptance::graph_selected_not_contact_accepted", 0)) == 87, f"{case}: expected 87 contact audit graph-selected-not-accepted rows", failures)
        require(ann.get("contact_acceptance_audit_strict_promotable_rows") == 0, f"{case}: expected zero strict-promotable contact audit rows", failures)
        require(int(counts.get("contact_nonpenetration_states", 0)) == 808, f"{case}: expected 808 contact/nonpenetration states", failures)
        require(int(counts.get("contact_nonpenetration::source_graph_candidate_but_local_penetration_veto", 0)) == 705, f"{case}: expected 705 local penetration contact veto states", failures)
        require(int(counts.get("contact_nonpenetration::source_graph_candidate_no_local_penetration_flag", 0)) == 16, f"{case}: expected 16 source graph contact-candidate states without local penetration flag", failures)
        require(int(counts.get("nonpenetration_repair_proposal_states", 0)) == 703, f"{case}: expected 703 nonpenetration repair proposal states", failures)
        require(int(counts.get("nonpenetration_repair::large_local_translation_required", 0)) == 335, f"{case}: expected 335 large local translation states", failures)
        require(int(counts.get("nonpenetration_repair::translation_candidate_unreliable_incoherent_normals", 0)) == 336, f"{case}: expected 336 incoherent-normal translation candidates", failures)
        require(int(counts.get("nonpenetration_repair::small_coherent_translation_candidate_local_postcheck_pass", 0)) == 21, f"{case}: expected 21 small coherent candidates passing local postcheck", failures)
        require(int(counts.get("nonpenetration_repair::small_coherent_translation_candidate_local_postcheck_failed", 0)) == 11, f"{case}: expected 11 small coherent candidates failing local postcheck", failures)
        require(int(counts.get("rigid_residual_checked_states", 0)) == 449, f"{case}: expected 449 rigid residual checked states", failures)
        require(int(counts.get("rigid_residual::bidirectional_residual_supported_uncertain", 0)) == 22, f"{case}: expected 22 bidirectional residual supported states", failures)
        require(int(counts.get("rigid_residual::visible_supported_but_fused_overspread", 0)) == 425, f"{case}: expected 425 fused-overspread residual states", failures)
        require(int(counts.get("rigid_residual::visible_surface_not_explained_by_fused_pose", 0)) == 2, f"{case}: expected 2 residual rejected states", failures)
        require("object:obj_tomato" in ann.get("rigid_candidate_ids", []), f"{case}: missing tomato generic rigid candidate", failures)
        tomato_pose_rows = 0
        tomato_surface_rows = 0
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for obj in frame.get("objects", []):
                if isinstance(obj, dict) and obj.get("object_id") == "object:obj_tomato":
                    attempt = obj.get("generic_rigid_se3_attempt", {}) if isinstance(obj.get("generic_rigid_se3_attempt"), dict) else {}
                    if attempt.get("stable_pose6_world_from_object") is not None:
                        tomato_pose_rows += 1
                    if isinstance(obj.get("frame_local_visible_surface_state"), dict):
                        tomato_surface_rows += 1
        require(tomato_pose_rows > 0, f"{case}: no tomato stable rigid pose rows", failures)
        require(tomato_surface_rows == 447, f"{case}: tomato visible surface rows {tomato_surface_rows} != 447", failures)
    return {"case": case, "frame_count": len(frames), "counts": counts, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    failures: list[str] = []
    cases = [validate_case(case, args.root, args.source_root, failures) for case in args.cases]
    report = {"status": "ok" if not failures else "failed", "cases": cases, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
