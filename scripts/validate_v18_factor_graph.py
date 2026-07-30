#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE = "scene_depth_supports_accepted_foreground_occluder_owner"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(path: Path) -> dict[str, Any]:
    ann = load_json(path)
    case = ann.get("case")
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: frames missing")
    fg = ann.get("factor_graph_summary")
    require(isinstance(fg, dict), f"{case}: factor_graph_summary missing")
    objective = fg.get("objective")
    require(isinstance(objective, dict), f"{case}: objective missing")
    energy_initial = float(objective.get("energy_initial"))
    energy_after = float(objective.get("energy_after"))
    require(energy_after <= energy_initial + 1e-6, f"{case}: graph energy did not decrease")
    variable_counts = fg.get("variable_counts")
    factor_counts = fg.get("factor_counts")
    require(isinstance(variable_counts, dict) and isinstance(factor_counts, dict), f"{case}: counts missing")
    for key in ["camera_depth_correction", "hand_state", "object_se3", "part_se3", "contact_switch", "occlusion_owner"]:
        require(int(variable_counts.get(key, 0)) > 0, f"{case}: missing {key} variables")
    require(int(variable_counts.get("deformable_surface_patch", 0)) > 0, f"{case}: missing deformable surface patch variables")
    for key in ["camera_depth_correction_observation", "hand_state_observation", "object_se3_observation", "part_se3_observation", "contact_switch_discrete", "contact_switch_temporal", "contact_local_nonpenetration", "occlusion_owner_discrete"]:
        require(int(factor_counts.get(key, 0)) > 0, f"{case}: missing {key} factors")
    implemented_status = fg.get("implemented_variable_status")
    spec_gaps = fg.get("spec_factor_gaps_remaining")
    require(isinstance(implemented_status, dict), f"{case}: implemented variable status missing")
    require("camera_depth_correction" in implemented_status and "observed_depth_scale_correction" in str(implemented_status.get("camera_depth_correction")), f"{case}: camera/depth correction observation status not explicit")
    require("part_se3" in implemented_status and "pca_rotvec" in str(implemented_status.get("part_se3")), f"{case}: part SE3 PCA status not explicit")
    require(isinstance(spec_gaps, list) and len(spec_gaps) > 0, f"{case}: spec factor gaps missing")
    require(any("camera_depth_correction_is_scale_only" in str(gap) for gap in spec_gaps), f"{case}: camera/depth correction limitation not explicit")
    require(any("visible_surface_PCA" in str(gap) for gap in spec_gaps), f"{case}: visible-surface part SE3 limitation not explicit")
    anchor_fp = fg.get("contact_pose_anchor_fixed_point")
    require(isinstance(anchor_fp, dict), f"{case}: contact pose anchor fixed-point summary missing")
    require(str(anchor_fp.get("method")) == "bounded_two_stage_contact_pose_anchor_fixed_point", f"{case}: contact pose anchor fixed-point method missing")
    require(str(anchor_fp.get("semantics", "")).find("raw_contact_proposals_remain_evidence_only") >= 0, f"{case}: contact pose anchor semantics do not block raw proposal coupling")
    require(isinstance(anchor_fp.get("history"), list) and len(anchor_fp.get("history")) >= 1, f"{case}: contact pose anchor fixed-point history missing")
    require(isinstance(anchor_fp.get("emitted_anchor_factor_count"), int), f"{case}: emitted contact-pose anchor count missing")
    require(int(anchor_fp.get("emitted_anchor_factor_count", 0)) <= int(anchor_fp.get("stable_anchor_input_count", 0)), f"{case}: emitted contact-pose anchor count exceeds stable input count")
    implemented_families = fg.get("implemented_factor_families")
    require(isinstance(implemented_families, list) and any("contact_switch_temporal" in str(item) for item in implemented_families), f"{case}: contact temporal factor family missing")
    require(isinstance(implemented_families, list) and any("contact_local_nonpenetration" in str(item) for item in implemented_families), f"{case}: contact local nonpenetration factor family missing")
    require(isinstance(implemented_families, list) and any("contact_part_pose_anchor" in str(item) for item in implemented_families), f"{case}: contact-part pose anchor factor family missing")
    require(isinstance(implemented_families, list) and any("deformable_surface_patch" in str(item) for item in implemented_families), f"{case}: deformable surface patch factor family missing")
    inference = fg.get("inference")
    require(isinstance(inference, dict), f"{case}: inference missing")
    require("SciPy" in str(inference.get("continuous_method")), f"{case}: continuous solve is not SciPy-backed")
    require("viterbi" in str(inference.get("discrete_method")).lower(), f"{case}: contact switch inference is not temporal Viterbi")
    series = inference.get("series_summaries")
    require(isinstance(series, dict) and len(series) > 0, f"{case}: series summaries missing")
    object_se3_series = {k: v for k, v in series.items() if str(k).startswith("object_se3::") and isinstance(v, dict)}
    part_se3_series = {k: v for k, v in series.items() if str(k).startswith("part_se3::") and isinstance(v, dict)}
    deformable_patch_series = {k: v for k, v in series.items() if str(k).startswith("deformable_surface_patch::") and isinstance(v, dict)}
    require(len(object_se3_series) > 0, f"{case}: object SE3 series missing")
    require(len(part_se3_series) > 0, f"{case}: part SE3 series missing")
    require(len(deformable_patch_series) > 0, f"{case}: deformable surface patch series missing")
    object_6d_count = sum(1 for v in object_se3_series.values() if int(v.get("dimension", 0)) == 6)
    part_6d_count = sum(1 for v in part_se3_series.values() if int(v.get("dimension", 0)) == 6)
    require(object_6d_count > 0, f"{case}: no 6D object SE3 series")
    require(part_6d_count > 0, f"{case}: no 6D part SE3 series")
    frame_with_graph = 0
    temporal_contact_rows = 0
    contact_episode_rows = 0
    contact_episode_factor_rows = 0
    temporal_contact_factor_rows = 0
    temporal_contact_active_conflicts = 0
    temporal_contact_bad_gaps = 0
    local_temporal_factor_count_sum = 0
    local_nonpenetration_factor_count_sum = 0
    contact_nonpenetration_factor_rows = 0
    contact_object_component_rows = 0
    contact_part_component_rows = 0
    deformable_surface_patch_rows = 0
    deformable_surface_visible_factor_rows = 0
    deformable_surface_contact_factor_rows = 0
    occlusion_owner_rows = 0
    occlusion_owner_with_temporal_or_mesh = 0
    accepted_occlusion_owner_rows = 0
    hand_occlusion_pose_fill_rows = 0
    hand_occlusion_pose_fill_factor_rows = 0
    local_occlusion_factor_count_sum = 0
    for frame in frames:
        g = frame.get("factor_graph_solution")
        if isinstance(g, dict) and isinstance(g.get("variables"), dict) and isinstance(g.get("objective"), dict):
            frame_with_graph += 1
            factors_raw = g.get("factors")
            factors: dict[str, Any] = factors_raw if isinstance(factors_raw, dict) else {}
            local_temporal_factor_count_sum += int(factors.get("contact_switch_temporal", 0))
            contact_episode_factor_rows += int(factors.get("contact_episode_persistence", 0))
            local_nonpenetration_factor_count_sum += int(factors.get("contact_local_nonpenetration", 0))
            local_occlusion_factor_count_sum += int(factors.get("occlusion_owner_discrete", 0))
            variables_raw = g.get("variables")
            variables: dict[str, Any] = variables_raw if isinstance(variables_raw, dict) else {}
            occlusion_raw = variables.get("occlusion_owner")
            if isinstance(occlusion_raw, list):
                for occ_raw in occlusion_raw:
                    occ: dict[str, Any] = occ_raw if isinstance(occ_raw, dict) else {}
                    occlusion_owner_rows += 1
                    require(str(occ.get("inference_method")) in {"box_mesh_depth_temporal_energy_with_unowned_competitor", "box_mesh_depth_temporal_energy_with_unowned_competitor_support_gated_by_hawor_observation"}, f"{case}: occlusion owner lacks integrated inference method")
                    if occ.get("owner_supported_by_depth_evidence") is True or occ.get("accepted_owner") is True:
                        accepted_occlusion_owner_rows += 1
                    candidates_raw = occ.get("candidate_energies")
                    candidates: list[Any] = candidates_raw if isinstance(candidates_raw, list) else []
                    require(len(candidates) > 0, f"{case}: occlusion owner candidates missing")
                    unowned_count = 0
                    for cand_raw in candidates:
                        cand: dict[str, Any] = cand_raw if isinstance(cand_raw, dict) else {}
                        if cand.get("object_id") is None:
                            unowned_count += 1
                        else:
                            require("mesh_temporal_support" in cand and "temporal_graph_selected" in cand and "depth_evidence_state" in cand, f"{case}: occlusion candidate missing mesh/temporal/depth fields")
                            depth_state = str(cand.get("depth_evidence_state"))
                            expected_support = "foreground" in depth_state and "support" in depth_state and "no_support" not in depth_state and "contradict" not in depth_state
                            expected_contradiction = "foreground" in depth_state and "contradict" in depth_state
                            require(cand.get("foreground_depth_support") is expected_support, f"{case}: foreground support flag mismatch")
                            require(cand.get("foreground_depth_contradiction") is expected_contradiction, f"{case}: foreground contradiction flag mismatch")
                            if cand.get("temporal_graph_selected") is True or float(cand.get("mesh_temporal_support", 0.0)) > 0.0:
                                occlusion_owner_with_temporal_or_mesh += 1
                    require(unowned_count == 1, f"{case}: occlusion owner missing exactly one unowned competitor")
                    if occ.get("owner_supported_by_depth_evidence") is True or occ.get("accepted_owner") is True:
                        chosen = occ.get("chosen_owner_object_id")
                        chosen_candidates = [cand for cand in candidates if isinstance(cand, dict) and cand.get("object_id") == chosen]
                        require(any(cand.get("accepted_by_depth_evidence") is True or cand.get("temporal_graph_accepted") is True for cand in chosen_candidates), f"{case}: accepted occlusion owner lacks source support")
            hand_raw = variables.get("hand_state")
            if isinstance(hand_raw, list):
                for hand_var_raw in hand_raw:
                    hand_var: dict[str, Any] = hand_var_raw if isinstance(hand_var_raw, dict) else {}
                    components = hand_var.get("hand_occlusion_pose_fill_components") if isinstance(hand_var.get("hand_occlusion_pose_fill_components"), list) else []
                    if components:
                        hand_occlusion_pose_fill_rows += len(components)
                        hand_occlusion_pose_fill_factor_rows += int(hand_var.get("factor_family_counts", {}).get("hand_occlusion_pose_fill", 0)) if isinstance(hand_var.get("factor_family_counts"), dict) else 0
                        for comp in components:
                            require(isinstance(comp, dict) and comp.get("factor_family") == "hand_occlusion_pose_fill", f"{case}: hand pose-fill component missing factor family")
                            coupling = comp.get("coupling") if isinstance(comp.get("coupling"), dict) else {}
                            require(coupling.get("accepted_occlusion_owner") is True and coupling.get("owner_depth_order_supported") is True, f"{case}: hand pose-fill factor lacks accepted owner depth support")
                            owner_support = coupling.get("source_occlusion_owner_depth_support") if isinstance(coupling.get("source_occlusion_owner_depth_support"), dict) else {}
                            require(owner_support.get("graph_occlusion_owner_accepted") is True, f"{case}: hand pose-fill factor lacks graph-accepted owner flag")
                            require(owner_support.get("depth_pair_evidence_state") == ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE, f"{case}: hand pose-fill factor carries non-accepted depth support label")
                            require(coupling.get("observed_mano_pose_through_occlusion_accepted") is True, f"{case}: hand pose-fill factor is not observed-MANO supported")
                            require(coupling.get("final_hawor_support_state") == "observed_same_frame_detection", f"{case}: hand pose-fill factor lacks observed HaWoR support")
                            require(coupling.get("hawor_to_v18_depth_scale_status") == "depth_scaled_from_projected_hawor_vertices_to_unidepth", f"{case}: hand pose-fill factor has invalid depth-scale status")
                            require(int(coupling.get("hawor_to_v18_depth_scale_sample_count") or 0) >= 40, f"{case}: hand pose-fill factor has too few depth-scale samples")
                            require("not_temporal_hallucination" in str(coupling.get("scope")), f"{case}: hand pose-fill scope overclaims temporal fill")
            object_raw = variables.get("object_se3")
            if isinstance(object_raw, list):
                for object_var_raw in object_raw:
                    object_var: dict[str, Any] = object_var_raw if isinstance(object_var_raw, dict) else {}
                    components = object_var.get("contact_object_coupling_components")
                    if isinstance(components, list) and components:
                        contact_object_component_rows += len(components)
                        for comp in components:
                            require(isinstance(comp, dict) and comp.get("factor_family") in {"contact_object_pose_anchor", "contact_surface_changing_object_pose_anchor"}, f"{case}: object contact component missing stable contact-pose anchor family")
                            coupling = comp.get("coupling") if isinstance(comp.get("coupling"), dict) else {}
                            distance = coupling.get("pre_coupling_surface_distance_m")
                            near = isinstance(distance, (int, float)) and float(distance) <= 0.12
                            require(coupling.get("contact_switch_active") is True and coupling.get("contact_proposal_used") is True and near, f"{case}: object contact pose anchor is not solved-active and near")
                            require(coupling.get("nonpenetration_conflict") is not True, f"{case}: object contact pose anchor includes nonpenetration conflict")
                            require(coupling.get("raw_contact_switch_active") in {True, False}, f"{case}: object anchor raw diagnostic flag missing")
            patch_raw = variables.get("deformable_surface_patch")
            if isinstance(patch_raw, list):
                for patch_var_raw in patch_raw:
                    patch_var: dict[str, Any] = patch_var_raw if isinstance(patch_var_raw, dict) else {}
                    deformable_surface_patch_rows += 1
                    require(str(patch_var.get("variable_id", "")).startswith("deformable_surface_patch::"), f"{case}: deformable patch variable id invalid")
                    require(patch_var.get("unit") == "world_m_local_visible_deformable_surface_patch_xyz", f"{case}: deformable patch variable has wrong unit")
                    require(patch_var.get("dimension") == 3, f"{case}: deformable patch variable is not 3D")
                    components = patch_var.get("deformable_surface_patch_components") if isinstance(patch_var.get("deformable_surface_patch_components"), list) else []
                    families = {str(comp.get("factor_family")) for comp in components if isinstance(comp, dict)}
                    require("deformable_surface_visible_observation" in families, f"{case}: deformable patch lacks visible-surface observation")
                    require("deformable_surface_contact_anchor" in families, f"{case}: deformable patch lacks MANO contact anchor")
                    deformable_surface_visible_factor_rows += int(patch_var.get("factor_family_counts", {}).get("deformable_surface_visible_observation", 0)) if isinstance(patch_var.get("factor_family_counts"), dict) else 0
                    deformable_surface_contact_factor_rows += int(patch_var.get("factor_family_counts", {}).get("deformable_surface_contact_anchor", 0)) if isinstance(patch_var.get("factor_family_counts"), dict) else 0
                    for comp in components:
                        coupling = comp.get("coupling") if isinstance(comp, dict) and isinstance(comp.get("coupling"), dict) else {}
                        require(coupling.get("contact_switch_active") is True and coupling.get("contact_proposal_used") is True, f"{case}: deformable patch component not tied to active contact")
                        require(coupling.get("support_path") == "deformable_same_frame_visible_surface", f"{case}: deformable patch component support path invalid")
                        require("not_whole_object_pose" in str(coupling.get("scope")), f"{case}: deformable patch scope overclaims object pose")
            part_raw = variables.get("part_se3")
            if isinstance(part_raw, list):
                for part_var_raw in part_raw:
                    part_var: dict[str, Any] = part_var_raw if isinstance(part_var_raw, dict) else {}
                    components = part_var.get("contact_part_coupling_components")
                    if isinstance(components, list) and components:
                        contact_part_component_rows += len(components)
                        for comp in components:
                            require(isinstance(comp, dict) and comp.get("factor_family") == "contact_part_pose_anchor", f"{case}: part contact component missing factor family")
                            coupling = comp.get("coupling") if isinstance(comp.get("coupling"), dict) else {}
                            distance = coupling.get("pre_coupling_surface_distance_m")
                            near = isinstance(distance, (int, float)) and float(distance) <= 0.12
                            require(coupling.get("contact_switch_active") is True and coupling.get("contact_proposal_used") is True and near, f"{case}: part contact pose anchor is not solved-active and near")
            episode_raw = variables.get("contact_episode")
            if isinstance(episode_raw, list):
                for episode_var_raw in episode_raw:
                    episode_var: dict[str, Any] = episode_var_raw if isinstance(episode_var_raw, dict) else {}
                    contact_episode_rows += 1
                    require(episode_var.get("estimate") is True, f"{case}: contact episode variable is not active")
                    scope = str(episode_var.get("scope"))
                    require("contact_state_only" in scope or "contact_episode_state" in scope, f"{case}: contact episode variable scope missing")
                    require(isinstance(episode_var.get("anchor_frame_indices"), list) and len(episode_var.get("anchor_frame_indices")) > 0, f"{case}: contact episode variable lacks anchors")
                    nearest_anchor_distance = episode_var.get("nearest_anchor_frame_distance")
                    max_anchor_distance = episode_var.get("max_nearest_anchor_distance_frames")
                    candidate_score = episode_var.get("candidate_score")
                    require(isinstance(candidate_score, (int, float)) and float(candidate_score) >= 0.65, f"{case}: contact episode variable has weak candidate score")
                    require(isinstance(nearest_anchor_distance, int) and isinstance(max_anchor_distance, int) and max_anchor_distance > 0 and nearest_anchor_distance <= max_anchor_distance, f"{case}: contact episode variable exceeds nearest-anchor bound")
            contact_raw = variables.get("contact_switch")
            if isinstance(contact_raw, list):
                for row_raw in contact_raw:
                    row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
                    require(row.get("temporal_inference_method") == "gap_aware_binary_viterbi_contact_switch", f"{case}: contact switch lacks temporal inference method")
                    temporal_contact_rows += 1
                    signed_conflict = row.get("signed_nonpenetration_conflict") is True
                    triangle_conflict = row.get("triangle_nonpenetration_conflict") is True
                    union_conflict = row.get("nonpenetration_conflict") is True
                    require(union_conflict == bool(signed_conflict or triangle_conflict), f"{case}: contact nonpenetration union inconsistent")
                    if row.get("local_nonpenetration_factor_present") is True:
                        contact_nonpenetration_factor_rows += 1
                        require(row.get("local_nonpenetration_factor_complete") is False, f"{case}: local nonpenetration factor overclaims completeness")
                        require("not_watertight_sdf" in str(row.get("local_nonpenetration_factor_scope")), f"{case}: local nonpenetration factor scope missing")
                        require(row.get("signed_local_nonpenetration_factor_present") is True or row.get("triangle_local_nonpenetration_factor_present") is True, f"{case}: local nonpenetration factor lacks evidence source")
                    if row.get("estimate") is True and union_conflict:
                        temporal_contact_active_conflicts += 1
                    if row.get("estimate") is True:
                        episode_support = bool(row.get("manipulation_contact_episode_supported") is True and row.get("post_graph_manipulation_episode_support") is True)
                        require(row.get("geometry_contact_evidence_available") is True or episode_support, f"{case}: active contact lacks geometry or episode evidence")
                        require(row.get("physical_contact_claim_supported") is True or episode_support, f"{case}: active contact lacks direct physical support or manipulation episode support")
                        if episode_support:
                            require(isinstance(row.get("manipulation_contact_episode_anchor_frame_indices"), list) and len(row.get("manipulation_contact_episode_anchor_frame_indices")) > 0, f"{case}: active episode contact lacks local anchors")
                            require(float(row.get("manipulation_contact_episode_candidate_score") or 0.0) >= 0.65, f"{case}: active episode contact has weak candidate score")
                            nearest_anchor_distance = row.get("manipulation_contact_episode_nearest_anchor_frame_distance")
                            max_anchor_distance = int(row.get("manipulation_contact_episode_max_nearest_anchor_distance_frames") or 0)
                            require(isinstance(nearest_anchor_distance, int) and max_anchor_distance > 0 and nearest_anchor_distance <= max_anchor_distance, f"{case}: active episode contact is not locally bounded by an anchor")
                            role = str(row.get("manipulation_contact_episode_frame_role") or "")
                            require(role in {"direct_visible_or_validated_contact_anchor", "occluded_contact_patch_anchor", "bounded_episode_bridge_candidate"}, f"{case}: active episode contact has invalid frame role {role!r}")
                            if role == "occluded_contact_patch_anchor":
                                evidence = row.get("manipulation_contact_episode_evidence") if isinstance(row.get("manipulation_contact_episode_evidence"), dict) else {}
                                require(evidence.get("occluded_contact_patch_anchor_supported") is True, f"{case}: occluded contact anchor lacks evidence flag")
                                require(row.get("depth_contradiction") is True, f"{case}: occluded contact anchor lacks depth/contact-patch occlusion state")
                                require(row.get("accepted_contact_owner") is True, f"{case}: occluded contact anchor lacks accepted contact-owner support")
                                require(float(row.get("min_box_coverage") or 0.0) >= 0.90, f"{case}: occluded contact anchor lacks high box coverage")
                                require(float(row.get("mesh_contact_support_score") or 0.0) >= 0.90, f"{case}: occluded contact anchor lacks high mesh contact support")
                            require(row.get("nonpenetration_conflict") is not True, f"{case}: episode contact overrode nonpenetration conflict")
                        if row.get("depth_contradiction") is True:
                            prior = row.get("visual_contact_prior") if isinstance(row.get("visual_contact_prior"), dict) else {}
                            require(row.get("depth_conflict_blocks_active_contact") is not True, f"{case}: active contact still has blocking depth conflict")
                            if episode_support:
                                require(str(row.get("depth_conflict_resolution")) in {"contact_episode_persistence_through_occluded_or_unmodeled_contact_patch", "local_contact_anchor_or_bounded_gap_persistence_through_occluded_or_unmodeled_contact_patch"} or row.get("visual_contact_prior_overrode_weak_depth_conflict") is True, f"{case}: active depth-contradicted episode contact lacks episode/visual-prior resolution")
                            else:
                                require(row.get("visual_contact_prior_overrode_weak_depth_conflict") is True, f"{case}: active depth-contradicted contact lacks explicit visual-prior override")
                                require(prior.get("contact_prior_supported") is True, f"{case}: active depth-contradicted contact lacks supported visual prior")
                                require(row.get("nonpenetration_conflict") is not True, f"{case}: visual prior overrode nonpenetration conflict")
                        if row.get("deformable_visible_surface_contact_claim_supported") is True:
                            raw_distance = row.get("final_metric_contact_distance_m")
                            require(isinstance(raw_distance, (int, float)) and float(raw_distance) <= 0.05, f"{case}: deformable active contact uses non-same-frame/proxy distance")
                        effective_distance = row.get("effective_metric_contact_distance_m")
                        mesh_support = float(row.get("mesh_contact_support_score", 0.0) or 0.0)
                        near_effective = isinstance(effective_distance, (int, float)) and float(effective_distance) <= 0.20
                        require(near_effective or mesh_support > 0.5 or episode_support, f"{case}: active contact lacks near metric distance, strong mesh support, or episode support")
                    gap = row.get("temporal_contact_previous_frame_gap")
                    has_factor = row.get("temporal_contact_has_factor") is True
                    applied = row.get("temporal_contact_transition_applied") is True
                    require(has_factor == applied, f"{case}: contact temporal factor/applied mismatch")
                    if has_factor:
                        temporal_contact_factor_rows += 1
                        require(isinstance(gap, int) and gap <= int(row.get("temporal_contact_max_gap_frames", 30)), f"{case}: contact temporal factor across invalid gap")
                    elif isinstance(gap, int) and gap <= int(row.get("temporal_contact_max_gap_frames", 30)) and gap > 0:
                        temporal_contact_bad_gaps += 1
    require(frame_with_graph == len(frames), f"{case}: not every frame has graph solution")
    require(occlusion_owner_rows == int(variable_counts.get("occlusion_owner", -1)), f"{case}: occlusion owner variable count mismatch")
    require(local_occlusion_factor_count_sum == occlusion_owner_rows and occlusion_owner_rows == int(factor_counts.get("occlusion_owner_discrete", -1)), f"{case}: occlusion owner factor count mismatch")
    require(occlusion_owner_with_temporal_or_mesh > 0, f"{case}: occlusion owner variables missing temporal/mesh evidence")
    require(temporal_contact_rows == int(variable_counts.get("contact_switch", -1)), f"{case}: contact switch variable count mismatch")
    if int(variable_counts.get("contact_episode", 0)) > 0 or int(factor_counts.get("contact_episode_persistence", 0)) > 0:
        require(contact_episode_rows == int(variable_counts.get("contact_episode", -1)), f"{case}: contact episode variable count mismatch")
        require(contact_episode_factor_rows == int(factor_counts.get("contact_episode_persistence", -1)), f"{case}: contact episode factor count mismatch")
    require(temporal_contact_factor_rows == int(factor_counts.get("contact_switch_temporal", -1)), f"{case}: temporal contact factor count mismatch")
    require(contact_nonpenetration_factor_rows == int(factor_counts.get("contact_local_nonpenetration", -1)), f"{case}: contact local nonpenetration factor count mismatch")
    require(deformable_surface_patch_rows == int(variable_counts.get("deformable_surface_patch", -1)), f"{case}: deformable surface patch variable count mismatch")
    require(deformable_surface_visible_factor_rows == int(factor_counts.get("deformable_surface_visible_observation", -1)), f"{case}: deformable surface visible factor count mismatch")
    require(deformable_surface_contact_factor_rows == int(factor_counts.get("deformable_surface_contact_anchor", -1)), f"{case}: deformable surface contact factor count mismatch")
    if int(factor_counts.get("hand_occlusion_pose_fill", 0)) > 0:
        require(hand_occlusion_pose_fill_rows == int(factor_counts.get("hand_occlusion_pose_fill", -1)), f"{case}: hand occlusion pose-fill component count mismatch")
        require(hand_occlusion_pose_fill_factor_rows == int(factor_counts.get("hand_occlusion_pose_fill", -1)), f"{case}: hand occlusion pose-fill factor count mismatch")
    require(int(factor_counts.get("contact_object_nonpenetration_repel", 0)) == 0, f"{case}: nonpenetration conflict must not move object pose")
    require(contact_object_component_rows == int(factor_counts.get("contact_object_pose_anchor", 0)) + int(factor_counts.get("contact_surface_changing_object_pose_anchor", 0)), f"{case}: contact-object component count mismatch")
    require(contact_part_component_rows == int(factor_counts.get("contact_part_pose_anchor", 0)), f"{case}: contact-part component count mismatch")
    require(local_nonpenetration_factor_count_sum == contact_nonpenetration_factor_rows, f"{case}: local nonpenetration factor sum mismatch")
    require(local_temporal_factor_count_sum == temporal_contact_factor_rows, f"{case}: local temporal contact factor sum mismatch")
    require(temporal_contact_active_conflicts == 0, f"{case}: active temporal contact has nonpenetration conflict")
    require(temporal_contact_bad_gaps == 0, f"{case}: missing temporal contact factor for valid adjacent gap")
    return {
        "case": case,
        "frame_count": len(frames),
        "energy_initial": energy_initial,
        "energy_after": energy_after,
        "energy_delta": energy_initial - energy_after,
        "variable_counts": variable_counts,
        "factor_counts": factor_counts,
        "object_6d_series_count": object_6d_count,
        "part_6d_series_count": part_6d_count,
        "contact_switch_temporal_factors": int(factor_counts.get("contact_switch_temporal", 0)),
        "contact_switch_temporal_rows": temporal_contact_rows,
        "contact_episode_rows": contact_episode_rows,
        "contact_episode_factors": int(factor_counts.get("contact_episode_persistence", 0)),
        "contact_local_nonpenetration_factors": int(factor_counts.get("contact_local_nonpenetration", 0)),
        "deformable_surface_patch_rows": deformable_surface_patch_rows,
        "deformable_surface_visible_factors": int(factor_counts.get("deformable_surface_visible_observation", 0)),
        "deformable_surface_contact_factors": int(factor_counts.get("deformable_surface_contact_anchor", 0)),
        "hand_occlusion_pose_fill_factors": int(factor_counts.get("hand_occlusion_pose_fill", 0)),
        "occlusion_owner_rows": occlusion_owner_rows,
        "frame_with_graph_count": frame_with_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        rows.append(validate_case(args.root / case / "annotations_v18_full.json"))
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
