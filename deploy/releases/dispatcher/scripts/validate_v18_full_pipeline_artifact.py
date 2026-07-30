#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


FORBIDDEN_FINAL_STRINGS = [
    "not_complete",
    "not complete",
    "verification status",
    "available_partial_score_2d_terms_only",
    "partial_score",
    "candidate-only",
    "candidate_only",
    "object_pose_candidate",
]
ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE = "scene_depth_supports_accepted_foreground_occluder_owner"
RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE = "scene_depth_supports_foreground_occluder_candidate_owner_unaccepted"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def ffprobe_frame_count(path: Path) -> tuple[int, str, float]:
    data = json.loads(subprocess.check_output([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]))
    stream = data["streams"][0]
    return int(stream["nb_read_frames"]), str(stream.get("r_frame_rate")), float(stream.get("duration", 0.0))


def _semantic_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(_semantic_strings(child))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_semantic_strings(child))
        return out
    if isinstance(value, str):
        if "/" in value or value.startswith("."):
            return []
        return [value.lower()]
    return []


def serialized_contains_forbidden(report_text: str, ann_text: str) -> list[str]:
    report_semantic_lines = [line.lower() for line in report_text.splitlines() if "/" not in line]
    semantic_text = "\n".join(report_semantic_lines + _semantic_strings(json.loads(ann_text)))
    return [term for term in FORBIDDEN_FINAL_STRINGS if term.lower() in semantic_text]


def stale_unaccepted_label_paths(value: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            out.extend(stale_unaccepted_label_paths(child, child_path))
        return out
    if isinstance(value, list):
        for idx, child in enumerate(value):
            out.extend(stale_unaccepted_label_paths(child, f"{path}[{idx}]"))
        return out
    if isinstance(value, str) and "unaccepted" in value:
        allowed_raw_provenance = any(token in path for token in ("raw_", "source_depth_pair_evidence", "source_row"))
        if not allowed_raw_provenance:
            out.append(path)
    return out


def validate_case(case_report: dict[str, Any], report_text: str) -> dict[str, Any]:
    case = str(case_report.get("case"))
    expected = int(case_report.get("expected_frame_count", -1))
    fps = float(case_report.get("fps", 0.0))
    require(expected > 0, f"{case}: missing expected frame count")
    require(fps > 0, f"{case}: missing fps")
    require(case_report.get("frame_count_match") is True, f"{case}: report says frame counts do not match")
    require(int(case_report.get("overlay_frame_count", -1)) == expected, f"{case}: overlay report count mismatch")
    require(int(case_report.get("world_frame_count", -1)) == expected, f"{case}: world report count mismatch")
    require(int(case_report.get("side_by_side_frame_count", -1)) == expected, f"{case}: side-by-side report count mismatch")

    monotonicity_raw = case_report.get("monotonicity")
    monotonicity: dict[str, Any] = monotonicity_raw if isinstance(monotonicity_raw, dict) else {}
    require(monotonicity.get("preserves_v16_overlay_mano_object_render") is True, f"{case}: V16 overlay not preserved")
    require(monotonicity.get("preserves_v16_metric_world_render") is True, f"{case}: V16 world render not preserved")
    require(monotonicity.get("v18_additions_are_overlay_layers") is True, f"{case}: V18 additions not marked as additive")

    for key in ["annotations", "overlay_video", "world_video", "side_by_side_video", "base_v16_overlay", "base_v16_world"]:
        path = Path(str(case_report.get(key)))
        require(path.exists(), f"{case}: missing {key}: {path}")

    for key in ["overlay_video", "world_video", "side_by_side_video"]:
        count, rate, duration = ffprobe_frame_count(Path(str(case_report[key])))
        require(count == expected, f"{case}: ffprobe {key} frame count {count} != {expected}")
        require(abs(duration - expected / fps) < 0.05, f"{case}: ffprobe {key} duration {duration} inconsistent with expected")
    overlay_draw = case_report.get("overlay_draw_counts") if isinstance(case_report.get("overlay_draw_counts"), dict) else {}
    world_draw = case_report.get("world_draw_counts") if isinstance(case_report.get("world_draw_counts"), dict) else {}
    overlay_occ = int(overlay_draw.get("occlusion_owner_edges", 0)) + int(overlay_draw.get("occlusion_unowned_or_unresolved_labels", 0))
    world_occ = int(world_draw.get("world_occlusion_owner_edges", 0)) + int(world_draw.get("world_occlusion_unowned_or_unresolved_labels", 0))
    require(overlay_occ > 0, f"{case}: overlay rendered no occlusion-owner evidence")
    require(world_occ > 0, f"{case}: world render drew no occlusion-owner evidence")
    require(int(overlay_draw.get("pose_fill_gate_markers", 0)) > 0, f"{case}: overlay rendered no pose-fill gate markers")
    require(int(world_draw.get("world_pose_fill_gate_markers", 0)) > 0, f"{case}: world render drew no pose-fill gate markers")
    overlay_object_completed = int(overlay_draw.get("reconstructed_geometry_pose_labels_completed", 0))
    overlay_object_supported = int(overlay_draw.get("reconstructed_geometry_pose_labels_supported", 0))
    overlay_object_rejected = int(overlay_draw.get("reconstructed_geometry_pose_labels_rejected", 0))
    overlay_object_unvalidated = int(overlay_draw.get("reconstructed_geometry_pose_labels_unvalidated", 0))
    world_object_completed = int(world_draw.get("world_reconstructed_mesh_footprints_completed", 0))
    world_object_supported = int(world_draw.get("world_reconstructed_mesh_footprints_supported", 0))
    world_object_rejected = int(world_draw.get("world_reconstructed_mesh_footprints_rejected", 0))
    world_object_unvalidated = int(world_draw.get("world_reconstructed_mesh_footprints_unvalidated", 0))
    require(overlay_object_completed + overlay_object_supported + overlay_object_rejected + overlay_object_unvalidated > 0, f"{case}: overlay rendered no reconstructed geometry pose labels")
    require(world_object_completed + world_object_supported + world_object_rejected + world_object_unvalidated > 0, f"{case}: world render drew no reconstructed mesh footprints")
    require(overlay_object_rejected > 0 and world_object_rejected > 0, f"{case}: render does not expose rejected/uncertain object pose candidates separately")
    overlay_part_ready = int(overlay_draw.get("part_reconstructed_geometry_pose_labels_ready", 0))
    overlay_part_supported = int(overlay_draw.get("part_reconstructed_geometry_pose_labels_supported", 0))
    overlay_part_rejected = int(overlay_draw.get("part_reconstructed_geometry_pose_labels_rejected", 0))
    overlay_part_unvalidated = int(overlay_draw.get("part_reconstructed_geometry_pose_labels_unvalidated", 0))
    world_part_ready = int(world_draw.get("world_part_reconstructed_mesh_footprints_ready", 0))
    world_part_supported = int(world_draw.get("world_part_reconstructed_mesh_footprints_supported", 0))
    world_part_rejected = int(world_draw.get("world_part_reconstructed_mesh_footprints_rejected", 0))
    world_part_unvalidated = int(world_draw.get("world_part_reconstructed_mesh_footprints_unvalidated", 0))
    require(overlay_part_ready + overlay_part_supported + overlay_part_rejected + overlay_part_unvalidated > 0, f"{case}: overlay rendered no reconstructed part geometry pose labels")
    require(world_part_ready + world_part_supported + world_part_rejected + world_part_unvalidated > 0, f"{case}: world render drew no reconstructed part mesh footprints")
    require(overlay_part_rejected > 0 and world_part_rejected > 0, f"{case}: render does not expose rejected/uncertain part pose candidates separately")

    ann_path = Path(str(case_report.get("annotations")))
    ann_text = ann_path.read_text(encoding="utf-8")
    forbidden = serialized_contains_forbidden(report_text, ann_text)
    require(not forbidden, f"{case}: forbidden final-artifact wording present: {forbidden}")
    ann = json.loads(ann_text)
    sources = ann.get("sources") if isinstance(ann.get("sources"), dict) else {}
    part_manifest_path = Path(str(sources.get("part_object_blocker_manifest", "")))
    require(part_manifest_path.exists(), f"{case}: missing part-object blocker manifest source for global part labels")
    part_manifest = load_json(part_manifest_path)
    part_manifest_rows = part_manifest.get("object_rows") if isinstance(part_manifest, dict) else []
    require(isinstance(part_manifest_rows, list), f"{case}: part-object blocker manifest lacks object rows")
    accepted_global_labels_by_object = {
        str(row.get("object_id")): sorted(str(label) for label in row.get("accepted_part_track_labels", []) if isinstance(label, str))
        for row in part_manifest_rows
        if isinstance(row, dict)
    }
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) == expected, f"{case}: annotation frame count mismatch")
    factor_graph_summary = ann.get("factor_graph_summary") if isinstance(ann.get("factor_graph_summary"), dict) else {}
    physical_contact_state_report = factor_graph_summary.get("physical_contact_state_report") if isinstance(factor_graph_summary.get("physical_contact_state_report"), dict) else {}
    contact_pose_anchor_fixed_point = factor_graph_summary.get("contact_pose_anchor_fixed_point") if isinstance(factor_graph_summary.get("contact_pose_anchor_fixed_point"), dict) else {}
    require(contact_pose_anchor_fixed_point.get("method") == "bounded_two_stage_contact_pose_anchor_fixed_point", f"{case}: contact pose anchor fixed-point summary missing")
    require("raw_contact_proposals_remain_evidence_only" in str(contact_pose_anchor_fixed_point.get("semantics")), f"{case}: contact pose anchors do not exclude raw proposal coupling")

    modules_raw = ann.get("modules")
    modules: dict[str, Any] = modules_raw if isinstance(modules_raw, dict) else {}
    module_text = json.dumps(modules)
    for needle, label in [
        ("depth_scale_correction", "camera/depth correction"),
        ("HaWoR_metric_MANO", "HaWoR metric MANO"),
        ("WiLoR", "WiLoR hand evidence"),
        ("RTMLib", "RTMLib hand evidence"),
        ("hand_baseline_evidence", "hand baseline evidence"),
        ("pose_fill_gate", "pose fill gate"),
        ("VLM_OWLv2_SAM2", "VLM/OWLv2/SAM2 perception"),
        ("depth_visible_surface", "depth visible geometry"),
        ("object_part_SE3", "object/part SE3"),
        ("contact_owner_graph", "contact-owner graph"),
        ("signed_normal_nonpenetration", "signed normal nonpenetration"),
        ("triangle_nonpenetration", "triangle nonpenetration"),
        ("temporal_occlusion_owner_graph", "temporal occlusion owner graph"),
        ("factor_graph", "factor graph"),
    ]:
        require(needle in module_text, f"{case}: {label} not listed in modules")

    counts = {
        "hand_total": 0,
        "hawor_metric_mano": 0,
        "wilor_key_rows": 0,
        "rtmlib_key_rows": 0,
        "hand_graph_metric": 0,
        "hand_support_state_rows": 0,
        "hand_mano_surface_reference_rows": 0,
        "hand_mano_parameter_contract_rows": 0,
        "hand_support_observed_rows": 0,
        "hand_support_inferred_rows": 0,
        "hand_support_boundary_fill_rows": 0,
        "pose_fill_gate_rows": 0,
        "pose_fill_accepted_rows": 0,
        "pose_fill_observed_mano_rows": 0,
        "pose_fill_temporal_rows": 0,
        "object_states": 0,
        "object_physical_state_rows": 0,
        "object_se3_rows": 0,
        "object_visible_geometry_rows": 0,
        "object_hidden_or_unresolved_geometry_rows": 0,
        "object_reconstructed_geometry_pose_rows": 0,
        "object_renderable_reconstructed_geometry_pose_rows": 0,
        "object_depth_silhouette_pose_validation_rows": 0,
        "object_depth_silhouette_pose_supported_rows": 0,
        "object_geometry_complete_rows": 0,
        "object_pose_requirement_met_rows": 0,
        "part_structured_object_pose_state_rows": 0,
        "part_structured_object_pose_ready_rows": 0,
        "object_vertex_sample_rows": 0,
        "part_rows": 0,
        "part_reconstructed_geometry_pose_rows": 0,
        "part_renderable_reconstructed_geometry_pose_rows": 0,
        "part_silhouette_depth_pose_validation_rows": 0,
        "part_silhouette_depth_pose_supported_rows": 0,
        "frame_local_part_pose_validation_rows": 0,
        "frame_local_part_pose_validation_supported_rows": 0,
        "frame_local_part_pose_validation_rejected_rows": 0,
        "contacts": 0,
        "contacts_with_final_metric_distance": 0,
        "contacts_with_hawor_support_weight": 0,
        "contact_metric_observed_rows": 0,
        "contact_metric_inferred_rows": 0,
        "contact_metric_boundary_fill_rows": 0,
        "signed_nonpenetration_rows": 0,
        "signed_nonpenetration_watertight_rows": 0,
        "signed_nonpenetration_physical_ineligible_rows": 0,
        "signed_nonpenetration_evaluated_nonobserved_hawor_rows": 0,
        "triangle_nonpenetration_rows": 0,
        "triangle_nonpenetration_watertight_rows": 0,
        "triangle_nonpenetration_physical_ineligible_rows": 0,
        "triangle_nonpenetration_evaluated_nonobserved_hawor_rows": 0,
        "deformable_surface_patch_vars": 0,
        "active_deformable_surface_patch_coupled_rows": 0,
        "contact_switch_vars": 0,
        "active_contact_switch_vars": 0,
        "active_contact_switch_vars_with_nonobserved_hawor_hand": 0,
        "active_contact_pose_coupled_rows": 0,
        "active_contact_unstable_anchor_rows": 0,
        "active_contact_stable_anchor_not_emitted_rows": 0,
        "contact_physical_mode_active": 0,
        "contact_physical_mode_depth_occluded_possible": 0,
        "contact_physical_mode_supported_near_noncontact": 0,
        "renderable_nonactive_contact_modes": 0,
        "raw_contact_switches_gated_by_hawor_support": 0,
        "raw_contact_switches_gated_by_physical_support": 0,
        "hand_occlusion_owner_accepted_rows": 0,
        "hand_occlusion_owner_accepted_rows_with_nonobserved_hawor_hand": 0,
        "hand_raw_occlusion_owner_rows_gated_by_hawor_support": 0,
        "hand_contact_depth_order_occlusion_rows": 0,
        "occlusion_owner_vars": 0,
        "occlusion_owner_supported_vars": 0,
        "occlusion_owner_supported_vars_with_nonobserved_hawor_hand": 0,
        "raw_occlusion_owner_vars_gated_by_hawor_support": 0,
        "camera_depth_observed_rows": 0,
        "factor_frames": 0,
    }

    for frame in frames:
        require(isinstance(frame, dict), f"{case}: non-dict frame row")
        hands = frame.get("hands") if isinstance(frame.get("hands"), list) else []
        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        object_by_id = {str(o.get("object_id")): o for o in objects if isinstance(o, dict)}
        hand_support_by_side = {str(h.get("hand_side")): str(h.get("hawor_support_state", "")) for h in hands if isinstance(h, dict)}
        fg_raw = frame.get("factor_graph_solution")
        fg: dict[str, Any] = fg_raw if isinstance(fg_raw, dict) else {}
        vars_raw = fg.get("variables")
        vars: dict[str, Any] = vars_raw if isinstance(vars_raw, dict) else {}
        if vars:
            counts["factor_frames"] += 1
        camera_depth = vars.get("camera_depth_correction") if isinstance(vars.get("camera_depth_correction"), dict) else {}
        if camera_depth.get("has_direct_observation") is True:
            counts["camera_depth_observed_rows"] += 1
        hand_vars = vars.get("hand_state") if isinstance(vars.get("hand_state"), list) else []
        counts["hand_graph_metric"] += sum(1 for row in hand_vars if isinstance(row, dict) and str(row.get("source", "")).startswith("HaWoR_metric_MANO_wrist_current_V18_world_m"))
        for row in hand_vars:
            if isinstance(row, dict) and str(row.get("source", "")).startswith("HaWoR_metric_MANO_wrist_current_V18_world_m"):
                require(row.get("unit") == "world_m_wrist_xyz", f"{case}: metric hand graph variable has wrong unit")
        patch_vars = vars.get("deformable_surface_patch") if isinstance(vars.get("deformable_surface_patch"), list) else []
        counts["deformable_surface_patch_vars"] += len(patch_vars)
        deformable_patch_ids = {str(row.get("variable_id")) for row in patch_vars if isinstance(row, dict)}
        for patch in patch_vars:
            if not isinstance(patch, dict):
                continue
            require(str(patch.get("variable_id", "")).startswith("deformable_surface_patch::"), f"{case}: deformable patch variable id invalid")
            require(patch.get("unit") == "world_m_local_visible_deformable_surface_patch_xyz", f"{case}: deformable patch variable has wrong unit")
            require(patch.get("dimension") == 3, f"{case}: deformable patch variable is not 3D")
            components = patch.get("deformable_surface_patch_components") if isinstance(patch.get("deformable_surface_patch_components"), list) else []
            families = {str(comp.get("factor_family")) for comp in components if isinstance(comp, dict)}
            require("deformable_surface_visible_observation" in families and "deformable_surface_contact_anchor" in families, f"{case}: deformable patch variable lacks visible/contact components")
        contact_vars = vars.get("contact_switch") if isinstance(vars.get("contact_switch"), list) else []
        counts["contact_switch_vars"] += len(contact_vars)
        for row in contact_vars:
            if not isinstance(row, dict):
                continue
            side = str(row.get("hand_side"))
            row_support_state = str(row.get("hand_support_state") or hand_support_by_side.get(side, ""))
            mode = str(row.get("physical_contact_mode") or "")
            require(mode in {"active_physical_contact", "contact_episode_hypothesis_nonactive", "depth_occluded_contact_possible", "supported_near_noncontact", "raw_contact_proposal_without_final_validated_physical_support", "depth_contradicted_noncontact", "separated_or_unresolved_noncontact"}, f"{case}: contact switch missing/invalid physical_contact_mode {mode!r}")
            renderable_mode = row.get("physical_contact_mode_renderable") is True
            if mode == "active_physical_contact":
                counts["contact_physical_mode_active"] += 1
                require(row.get("estimate") is True, f"{case}: active physical_contact_mode without active estimate")
            elif mode == "contact_episode_hypothesis_nonactive":
                counts["renderable_nonactive_contact_modes"] += int(renderable_mode)
                require(row.get("estimate") is not True, f"{case}: episode hypothesis mode is active")
                require(renderable_mode, f"{case}: episode hypothesis mode is not renderable")
                support_paths = row.get("physical_contact_mode_support_paths") if isinstance(row.get("physical_contact_mode_support_paths"), list) else []
                require("manipulation_contact_episode_persistent_constraint" in support_paths, f"{case}: episode hypothesis lacks episode support path")
                require(row.get("post_graph_manipulation_episode_support") is True, f"{case}: episode hypothesis lacks post-graph episode flag")
                require(row.get("post_graph_direct_visible_or_validated_near_support") is not True, f"{case}: episode hypothesis has direct near support and should be classified by direct evidence")
                require(row.get("physical_contact_claim_supported") is not True, f"{case}: episode hypothesis overclaims solved contact")
            elif mode == "depth_occluded_contact_possible":
                counts["contact_physical_mode_depth_occluded_possible"] += 1
                counts["renderable_nonactive_contact_modes"] += int(renderable_mode)
                require(row.get("estimate") is not True, f"{case}: depth-occluded possible mode is active")
                require(renderable_mode, f"{case}: depth-occluded possible mode is not renderable")
                require(row.get("depth_conflict_blocks_active_contact") is True, f"{case}: depth-occluded possible mode lacks depth conflict blocker")
                require(row.get("raw_estimate_before_physical_contact_gate") is True, f"{case}: depth-occluded possible mode lacks raw contact-energy support")
                require(isinstance(row.get("physical_contact_mode_support_paths"), list) and len(row.get("physical_contact_mode_support_paths")) > 0, f"{case}: depth-occluded possible mode lacks final support path")
                require(row.get("physical_contact_mode_nearest_distance_m") is not None and float(row.get("physical_contact_mode_nearest_distance_m")) <= 0.12, f"{case}: depth-occluded possible mode is not near geometry")
            elif mode == "supported_near_noncontact":
                counts["contact_physical_mode_supported_near_noncontact"] += 1
                counts["renderable_nonactive_contact_modes"] += int(renderable_mode)
                require(row.get("estimate") is not True, f"{case}: supported near noncontact mode is active")
                require(renderable_mode, f"{case}: supported near noncontact mode is not renderable")
                require(row.get("depth_conflict_blocks_active_contact") is not True, f"{case}: supported near noncontact mode has depth conflict")
                support_paths = row.get("physical_contact_mode_support_paths") if isinstance(row.get("physical_contact_mode_support_paths"), list) else []
                require(len(support_paths) > 0, f"{case}: supported near noncontact mode lacks final support path")
                require(row.get("physical_contact_mode_nearest_distance_m") is not None and float(row.get("physical_contact_mode_nearest_distance_m")) <= 0.12, f"{case}: supported near noncontact mode is not near geometry")
                if "validated_part_visible_depth_silhouette_pose" in support_paths:
                    require(row.get("final_validated_part_metric_contact_distance_m") is not None, f"{case}: validated-part near mode missing final validated part distance")
                    require(abs(float(row.get("physical_contact_mode_nearest_distance_m")) - float(row.get("final_validated_part_metric_contact_distance_m"))) < 1e-6, f"{case}: validated-part near mode distance does not match supported part distance")
                    require(isinstance(row.get("validated_part_nearest_hand_point_world_m"), list) and len(row.get("validated_part_nearest_hand_point_world_m")) == 3, f"{case}: validated-part near mode missing metric hand endpoint")
                    require(isinstance(row.get("validated_part_nearest_part_point_world_m"), list) and len(row.get("validated_part_nearest_part_point_world_m")) == 3, f"{case}: validated-part near mode missing metric part endpoint")
                if "deformable_same_frame_visible_surface_near_noncontact" in support_paths:
                    require(row.get("final_metric_contact_distance_m") is not None, f"{case}: deformable near mode missing same-frame metric distance")
                    require(0.05 < float(row.get("final_metric_contact_distance_m")) <= 0.12, f"{case}: deformable near mode is outside non-active near band")
                    require(abs(float(row.get("physical_contact_mode_nearest_distance_m")) - float(row.get("final_metric_contact_distance_m"))) < 1e-6, f"{case}: deformable near mode distance does not match same-frame visible-surface distance")
            elif renderable_mode:
                raise AssertionError(f"{case}: unsupported physical_contact_mode is renderable: {mode}")
            solved_claim_keys = [
                "physical_contact_claim_supported",
                "rigid_pose_contact_claim_supported",
                "validated_part_pose_contact_claim_supported",
                "surface_changing_pose_contact_claim_supported",
                "deformable_visible_surface_contact_claim_supported",
            ]
            evidence_keys = [
                "physical_contact_evidence_supported",
                "rigid_pose_contact_evidence_supported",
                "validated_part_pose_contact_evidence_supported",
                "surface_changing_pose_contact_evidence_supported",
                "deformable_visible_surface_contact_evidence_supported",
            ]
            if mode != "active_physical_contact":
                for key in solved_claim_keys:
                    require(row.get(key) is not True, f"{case}: non-active contact row carries solved claim flag {key}")
            for key in evidence_keys:
                if key in row:
                    require(row.get(key) in {True, False}, f"{case}: contact evidence flag {key} is not boolean")
            if row.get("raw_estimate_before_hawor_support_gate") is True and row.get("estimate") is False and row_support_state != "observed_same_frame_detection":
                counts["raw_contact_switches_gated_by_hawor_support"] += 1
            if row.get("raw_estimate_before_physical_contact_gate") is True and row.get("physical_contact_evidence_supported") is not True:
                counts["raw_contact_switches_gated_by_physical_support"] += 1
            if row.get("estimate") is True:
                counts["active_contact_switch_vars"] += 1
                row_support_paths = row.get("physical_contact_mode_support_paths") if isinstance(row.get("physical_contact_mode_support_paths"), list) else []
                episode_support = bool(row.get("post_graph_manipulation_episode_support") is True and "manipulation_contact_episode_persistent_constraint" in row_support_paths)
                direct_support_paths = [row.get("rigid_pose_contact_claim_supported") is True, row.get("validated_part_pose_contact_claim_supported") is True, row.get("surface_changing_pose_contact_claim_supported") is True, row.get("deformable_visible_surface_contact_claim_supported") is True]
                require(row.get("physical_contact_claim_supported") is True, f"{case}: active contact lacks solved direct physical contact support")
                require(row.get("post_graph_direct_visible_or_validated_near_support") is True, f"{case}: active contact lacks direct frame-local visible/validated near support")
                require(row.get("hand_depth_scale_supported_for_contact") is True, f"{case}: active contact lacks depth-scaled HaWoR metric support")
                require(row.get("hand_depth_scale_status") == "depth_scaled_from_projected_hawor_vertices_to_unidepth", f"{case}: active contact has invalid hand depth scale status")
                require(int(row.get("hand_depth_scale_sample_count") or 0) >= 40, f"{case}: active contact has too few hand depth scale samples")
                if episode_support:
                    episode = row.get("manipulation_contact_episode_final_support") if isinstance(row.get("manipulation_contact_episode_final_support"), dict) else {}
                    episode_evidence = row.get("manipulation_contact_episode_evidence") if isinstance(row.get("manipulation_contact_episode_evidence"), dict) else {}
                    require(row.get("manipulation_contact_episode_supported") is True, f"{case}: active episode contact lacks episode support flag")
                    require(isinstance(row.get("manipulation_contact_episode_anchor_frame_indices"), list) and len(row.get("manipulation_contact_episode_anchor_frame_indices")) > 0, f"{case}: active episode contact lacks local anchor frames")
                    require(float(row.get("manipulation_contact_episode_candidate_score") or 0.0) >= 0.65, f"{case}: active episode contact lacks strong manipulation candidate score")
                    nearest_anchor_distance = row.get("manipulation_contact_episode_nearest_anchor_frame_distance")
                    max_anchor_distance = int(row.get("manipulation_contact_episode_max_nearest_anchor_distance_frames") or 0)
                    require(isinstance(nearest_anchor_distance, int) and max_anchor_distance > 0 and nearest_anchor_distance <= max_anchor_distance, f"{case}: active episode contact is not locally bounded by an anchor")
                    role = str(row.get("manipulation_contact_episode_frame_role") or "")
                    require(role in {"direct_visible_or_validated_contact_anchor", "occluded_contact_patch_anchor", "bounded_episode_bridge_candidate"}, f"{case}: active episode contact has invalid frame role {role!r}")
                    if role == "occluded_contact_patch_anchor":
                        require(episode_evidence.get("occluded_contact_patch_anchor_supported") is True, f"{case}: occluded contact anchor lacks evidence flag")
                        require(row.get("depth_contradiction") is True, f"{case}: occluded contact anchor lacks depth/contact-patch occlusion state")
                        require(row.get("accepted_contact_owner") is True, f"{case}: occluded contact anchor lacks accepted contact-owner support")
                        require(float(row.get("min_box_coverage") or 0.0) >= 0.90, f"{case}: occluded contact anchor lacks high box coverage")
                        require(float(row.get("mesh_contact_support_score") or 0.0) >= 0.90, f"{case}: occluded contact anchor lacks high mesh contact support")
                    if role == "bounded_episode_bridge_candidate":
                        require(nearest_anchor_distance > 0, f"{case}: bridge candidate cannot be zero-distance anchor")
                    require(episode.get("scope") == "contact_state_only_not_object_geometry_completion_not_hidden_pose_closure", f"{case}: active episode support scope overclaims")
                    require(row.get("nonpenetration_conflict") is not True, f"{case}: episode contact overrode nonpenetration conflict")
                if row.get("depth_contradiction") is True:
                    prior = row.get("visual_contact_prior") if isinstance(row.get("visual_contact_prior"), dict) else {}
                    require(row.get("depth_conflict_blocks_active_contact") is not True, f"{case}: active contact still has blocking depth conflict")
                    if episode_support:
                        require(str(row.get("depth_conflict_resolution")) in {"contact_episode_persistence_through_occluded_or_unmodeled_contact_patch", "local_contact_anchor_or_bounded_gap_persistence_through_occluded_or_unmodeled_contact_patch"} or row.get("visual_contact_prior_overrode_weak_depth_conflict") is True, f"{case}: active episode depth contradiction lacks episode/visual-prior resolution")
                    else:
                        require(row.get("visual_contact_prior_overrode_weak_depth_conflict") is True, f"{case}: active depth-contradicted contact lacks explicit visual-prior override")
                        require(prior.get("contact_prior_supported") is True, f"{case}: active depth-contradicted contact lacks supported visual prior evidence")
                        require(row.get("effective_metric_contact_distance_m") is not None and float(row.get("effective_metric_contact_distance_m")) <= 0.07, f"{case}: visual-prior active contact is not in close metric band")
                        require(float(row.get("mesh_contact_support_score") or 0.0) >= 0.90, f"{case}: visual-prior active contact lacks high mesh contact support")
                        require(row.get("nonpenetration_conflict") is not True, f"{case}: visual prior overrode nonpenetration conflict")
                require(any(direct_support_paths), f"{case}: active contact lacks rigid/part/surface-changing/deformable-surface support path")
                coupling = row.get("active_contact_coupling_state") if isinstance(row.get("active_contact_coupling_state"), dict) else None
                require(isinstance(coupling, dict), f"{case}: active contact lacks object/part coupling state")
                require(coupling.get("contact_state_affects_object_or_part_pose") in {True, False}, f"{case}: active contact coupling state missing boolean effect field")
                require("not_a_contact_claim_source" in str(coupling.get("scope")), f"{case}: active contact coupling state scope missing")
                if coupling.get("contact_state_affects_object_or_part_pose") is True:
                    counts["active_contact_pose_coupled_rows"] += 1
                    require(coupling.get("stable_contact_pose_anchor_factor_emitted") is True, f"{case}: active contact pose coupling lacks emitted stable anchor factor")
                    require(coupling.get("coupling_family") in {"contact_object_pose_anchor", "contact_surface_changing_object_pose_anchor", "contact_part_pose_anchor"}, f"{case}: active contact pose coupling has invalid family")
                elif coupling.get("coupling_state") == "active_contact_not_pose_coupled_unstable_anchor_fixed_point":
                    counts["active_contact_unstable_anchor_rows"] += 1
                    require(coupling.get("stable_contact_pose_anchor_factor_emitted") is False, f"{case}: unstable anchor row claims emitted factor")
                    require("direct_contact_support_is_not_a_stable_contact_pose_anchor_fixed_point" in (coupling.get("blockers") if isinstance(coupling.get("blockers"), list) else []), f"{case}: unstable anchor row lacks blocker")
                elif coupling.get("coupling_state") == "active_contact_not_pose_coupled_stable_anchor_factor_not_emitted":
                    counts["active_contact_stable_anchor_not_emitted_rows"] += 1
                    require(coupling.get("stable_contact_pose_anchor_candidate") is True, f"{case}: stable-not-emitted row lacks stable candidate flag")
                    require(coupling.get("stable_contact_pose_anchor_factor_emitted") is False, f"{case}: stable-not-emitted row claims emitted factor")
                    require("stable_contact_support_but_pose_anchor_factor_not_emitted_by_pre_solve_geometry_or_pose_precondition" in (coupling.get("blockers") if isinstance(coupling.get("blockers"), list) else []), f"{case}: stable-not-emitted row lacks precondition blocker")
                if "deformable_same_frame_visible_surface" in row_support_paths:
                    require(coupling.get("contact_state_affects_object_or_part_pose") is False, f"{case}: deformable contact falsely coupled to object pose")
                    require(coupling.get("contact_state_affects_deformable_surface_patch_state") is True, f"{case}: deformable active contact lacks local surface patch coupling")
                    patch_id = str(coupling.get("deformable_surface_patch_variable_id") or "")
                    require(patch_id in deformable_patch_ids, f"{case}: deformable active contact references missing local patch variable")
                    require(coupling.get("deformable_surface_patch_factor_emitted") is True, f"{case}: deformable active contact lacks emitted patch factor")
                    require("whole_object_pose_not_coupled_deformable_patch_state_only" in (coupling.get("blockers") if isinstance(coupling.get("blockers"), list) else []), f"{case}: deformable patch coupling scope blocker missing")
                    counts["active_deformable_surface_patch_coupled_rows"] += 1
                if row.get("surface_changing_pose_contact_claim_supported") is True:
                    obj = object_by_id.get(str(row.get("object_id")), {})
                    validation = obj.get("object_depth_silhouette_pose_validation") if isinstance(obj, dict) and isinstance(obj.get("object_depth_silhouette_pose_validation"), dict) else {}
                    row_support_paths = row.get("physical_contact_mode_support_paths") if isinstance(row.get("physical_contact_mode_support_paths"), list) else []
                    has_compact_support = validation.get("surface_changing_compact_visible_pose_supported") is True and "surface_changing_visible_depth_silhouette_pose" in row_support_paths
                    local_support = row.get("surface_changing_local_visible_contact_support") if isinstance(row.get("surface_changing_local_visible_contact_support"), dict) else {}
                    has_local_support = "surface_changing_local_visible_contact_surface" in row_support_paths
                    require(has_compact_support or has_local_support, f"{case}: active surface-changing contact lacks final compact or local visible-surface support")
                    if has_local_support:
                        require(row.get("visual_contact_prior_supported") is True, f"{case}: local surface-changing contact lacks visual prior")
                        require(float(local_support.get("observed_projection_inside_mask_fraction") or 0.0) >= 0.80, f"{case}: local surface-changing contact lacks observed mask support")
                        require(float(local_support.get("observed_to_predicted_median_m") or 999.0) <= 0.075, f"{case}: local surface-changing contact residual too high")
                        require(row.get("effective_metric_contact_distance_m") is not None and float(row.get("effective_metric_contact_distance_m")) <= 0.07, f"{case}: local surface-changing contact is outside close visual-prior band")
                    else:
                        require(row.get("effective_metric_contact_distance_m") is not None and float(row.get("effective_metric_contact_distance_m")) <= 0.12, f"{case}: active surface-changing contact is not near MANO/object geometry")
                if row.get("validated_part_pose_contact_claim_supported") is True:
                    obj = object_by_id.get(str(row.get("object_id")), {})
                    parts = obj.get("parts") if isinstance(obj, dict) and isinstance(obj.get("parts"), list) else []
                    label = str(row.get("validated_part_track_label"))
                    part = next((p for p in parts if isinstance(p, dict) and str(p.get("part_track_label")) == label), None)
                    require(isinstance(part, dict), f"{case}: active validated-part contact missing part row")
                    validation = part.get("part_silhouette_depth_pose_validation") if isinstance(part.get("part_silhouette_depth_pose_validation"), dict) else {}
                    require(validation.get("visible_depth_silhouette_pose_supported") is True, f"{case}: active validated-part contact lacks supported part validation")
                    require(row.get("validated_part_metric_contact_distance_m") is not None and float(row.get("validated_part_metric_contact_distance_m")) <= 0.12, f"{case}: active validated-part contact is not near supported part geometry")
                if row.get("deformable_visible_surface_contact_claim_supported") is True:
                    obj = object_by_id.get(str(row.get("object_id")), {})
                    schema = obj.get("physical_state_schema") if isinstance(obj, dict) and isinstance(obj.get("physical_state_schema"), dict) else {}
                    geom = obj.get("visible_geometry_candidate") if isinstance(obj, dict) and isinstance(obj.get("visible_geometry_candidate"), dict) else {}
                    physical = str(schema.get("model_physical_state_type") or obj.get("physical_state_label") or "unknown") if isinstance(obj, dict) else "unknown"
                    require(schema.get("requires_part_or_relative_motion_model") is not True, f"{case}: part-required object used whole-object deformable contact path")
                    require(physical == "deformable" or schema.get("secondary_deformable_or_surface_component") is True, f"{case}: active deformable contact is not on deformable object")
                    require(isinstance(geom.get("world_vertices_sample_m"), list) and len(geom.get("world_vertices_sample_m")) > 0, f"{case}: active deformable contact lacks visible depth surface")
                    require(row.get("final_metric_contact_distance_m") is not None and float(row.get("final_metric_contact_distance_m")) <= 0.05, f"{case}: active deformable contact is not within 5cm same-frame visible surface band")
                require(row.get("depth_contradiction") is not True or row.get("visual_contact_prior_overrode_weak_depth_conflict") is True or episode_support, f"{case}: active contact has depth contradiction without visual-prior or episode resolution")
                if row_support_state != "observed_same_frame_detection":
                    counts["active_contact_switch_vars_with_nonobserved_hawor_hand"] += 1
        occlusion_vars = vars.get("occlusion_owner") if isinstance(vars.get("occlusion_owner"), list) else []
        counts["occlusion_owner_vars"] += len(occlusion_vars)
        for row in occlusion_vars:
            if not isinstance(row, dict):
                continue
            side = str(row.get("hand_side"))
            row_support_state = str(row.get("hand_support_state") or hand_support_by_side.get(side, ""))
            if row.get("raw_owner_supported_by_depth_evidence_before_hawor_support_gate") is True and row.get("owner_supported_by_depth_evidence") is False and row_support_state != "observed_same_frame_detection":
                counts["raw_occlusion_owner_vars_gated_by_hawor_support"] += 1
            if row.get("owner_supported_by_depth_evidence") is True:
                counts["occlusion_owner_supported_vars"] += 1
                if row_support_state != "observed_same_frame_detection":
                    counts["occlusion_owner_supported_vars_with_nonobserved_hawor_hand"] += 1

        require(len(hands) == 2, f"{case}: frame {frame.get('frame_idx')} does not have two hand rows")
        for hand in hands:
            require(isinstance(hand, dict), f"{case}: non-dict hand row")
            counts["hand_total"] += 1
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            mano = hand.get("mano_candidate") if isinstance(hand.get("mano_candidate"), dict) else {}
            if hand.get("hand_geometry_source") == "HaWoR_metric_MANO_current_V18_world" or str(metric.get("source", "")).startswith("HaWoR_metric_MANO"):
                counts["hawor_metric_mano"] += 1
            surface_ref = mano.get("surface_reference") if isinstance(mano.get("surface_reference"), dict) else metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else None
            if isinstance(surface_ref, dict) and surface_ref.get("shape_vertices") == [778, 3] and isinstance(surface_ref.get("bridge_npz"), str):
                counts["hand_mano_surface_reference_rows"] += 1
            mano_params = mano.get("mano_params") if isinstance(mano.get("mano_params"), dict) else metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else None
            if isinstance(mano_params, dict) and all(isinstance(mano_params.get(k), list) and len(mano_params.get(k)) == n for k, n in [("root_orient_axis_angle", 3), ("hand_pose_axis_angle", 45), ("betas", 10), ("trans_world_m", 3)]):
                counts["hand_mano_parameter_contract_rows"] += 1
            support_state = str(hand.get("hawor_support_state", ""))
            support_weight = hand.get("hawor_physical_factor_weight")
            require(support_state in {"observed_same_frame_detection", "inferred_no_same_frame_detection", "temporal_boundary_fill", "pipeline_gap_fill", "missing_hawor_row"}, f"{case}: invalid/missing hand HaWoR support state {support_state!r}")
            require(isinstance(support_weight, (int, float)) and 0.0 <= float(support_weight) <= 1.0, f"{case}: invalid hand HaWoR support weight")
            counts["hand_support_state_rows"] += 1
            if support_state == "observed_same_frame_detection":
                counts["hand_support_observed_rows"] += 1
                require(hand.get("hawor_same_frame_detection") is True, f"{case}: observed support row missing same-frame detector flag")
            elif support_state == "inferred_no_same_frame_detection":
                counts["hand_support_inferred_rows"] += 1
            elif support_state == "temporal_boundary_fill":
                counts["hand_support_boundary_fill_rows"] += 1
                require(hand.get("hawor_temporal_boundary_filled") is True, f"{case}: boundary-fill support row missing boundary flag")
            if "wilor_or_v16_candidate_present" in hand:
                counts["wilor_key_rows"] += 1
            if "rtmlib_anchor_available" in hand:
                counts["rtmlib_key_rows"] += 1
            pose_gate = hand.get("occlusion_pose_fill_gate") if isinstance(hand.get("occlusion_pose_fill_gate"), dict) else {}
            if pose_gate:
                counts["pose_fill_gate_rows"] += 1
                if pose_gate.get("pose_fill_through_occlusion_accepted") is True:
                    counts["pose_fill_accepted_rows"] += 1
                    require(pose_gate.get("accepted_occlusion_owner") is True and pose_gate.get("owner_depth_order_supported") is True, f"{case}: accepted pose fill lacks accepted owner depth support")
                    require(pose_gate.get("final_hawor_support_state") == "observed_same_frame_detection", f"{case}: accepted pose fill lacks observed final HaWoR support")
                    require(pose_gate.get("final_hawor_same_frame_detection") is True, f"{case}: accepted pose fill lacks same-frame detector flag")
                    require(pose_gate.get("final_hawor_observed_depth_scaled_mano_supported") is True, f"{case}: accepted pose fill lacks depth-scaled MANO support")
                    require(pose_gate.get("hawor_to_v18_depth_scale_status") == "depth_scaled_from_projected_hawor_vertices_to_unidepth", f"{case}: accepted pose fill has invalid depth-scale status")
                    sample_count = int(pose_gate.get("hawor_to_v18_depth_scale_sample_count") or 0)
                    min_samples = int(pose_gate.get("min_hawor_to_v18_depth_scale_sample_count") or 0)
                    require(min_samples > 0 and sample_count >= min_samples, f"{case}: accepted pose fill has too few depth-scale samples")
                    require(not pose_gate.get("observed_pose_acceptance_blockers"), f"{case}: accepted observed pose fill has fatal blockers")
                    stale_paths = stale_unaccepted_label_paths({"pose_gate": pose_gate, "occlusion_owner_hypothesis": hand.get("occlusion_owner_hypothesis")})
                    require(not stale_paths, f"{case}: accepted pose fill contains stale non-accepted owner labels outside raw provenance: {stale_paths[:5]}")
                    owner_support = pose_gate.get("source_occlusion_owner_depth_support") if isinstance(pose_gate.get("source_occlusion_owner_depth_support"), dict) else {}
                    require(owner_support.get("graph_occlusion_owner_accepted") is True, f"{case}: accepted pose fill lacks graph-accepted owner flag")
                    require(owner_support.get("depth_pair_evidence_state") == ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE, f"{case}: accepted pose fill carries non-accepted depth support label")
                    raw_depth_state = owner_support.get("raw_depth_pair_evidence_state_before_graph_acceptance")
                    require(raw_depth_state is None or raw_depth_state == RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE, f"{case}: accepted pose fill raw depth provenance has unexpected state")
                    acceptance_type = str(pose_gate.get("pose_fill_acceptance_type") or "")
                    if acceptance_type == "observed_depth_scaled_mano_behind_accepted_occluder":
                        counts["pose_fill_observed_mano_rows"] += 1
                        require(pose_gate.get("observed_mano_pose_through_occlusion_accepted") is True, f"{case}: observed pose fill flag missing")
                    elif acceptance_type == "temporal_occlusion_pose_baseline":
                        counts["pose_fill_temporal_rows"] += 1
                        require(pose_gate.get("hand_baseline_temporal_occlusion_pose_accepted") is True, f"{case}: temporal pose fill lacks baseline acceptance")
                    else:
                        raise RuntimeError(f"{case}: unsupported pose-fill acceptance type {acceptance_type!r}")
            occ = hand.get("occlusion_owner_hypothesis") if isinstance(hand.get("occlusion_owner_hypothesis"), dict) else None
            require(isinstance(occ, dict), f"{case}: missing hand occlusion owner hypothesis")
            if isinstance(occ, dict):
                raw_count = int(occ.get("raw_accepted_occlusion_owner_count_before_hawor_support_gate") or 0)
                accepted_count = int(occ.get("accepted_occlusion_owner_count") or 0)
                if raw_count > 0 and accepted_count == 0 and support_state != "observed_same_frame_detection":
                    counts["hand_raw_occlusion_owner_rows_gated_by_hawor_support"] += 1
                if accepted_count > 0:
                    counts["hand_occlusion_owner_accepted_rows"] += 1
                    require(occ.get("state") == "accepted_occlusion_owner_by_final_graph_and_observed_hawor_support", f"{case}: accepted hand occlusion owner carries stale/non-accepted state")
                    if support_state != "observed_same_frame_detection":
                        counts["hand_occlusion_owner_accepted_rows_with_nonobserved_hawor_hand"] += 1
                depth_rows = occ.get("contact_depth_order_evidence") if isinstance(occ.get("contact_depth_order_evidence"), list) else []
                counts["hand_contact_depth_order_occlusion_rows"] += len(depth_rows)
                for depth_row in depth_rows:
                    require(isinstance(depth_row, dict), f"{case}: contact depth-order occlusion row is not dict")
                    require(depth_row.get("contact_depth_order_supported") is True, f"{case}: contact depth-order occlusion row lacks support flag")
                    require(depth_row.get("global_occlusion_owner_claim") is False, f"{case}: contact depth-order occlusion row overclaims global owner")
                    require("contact_pair_depth_order" in str(depth_row.get("scope")), f"{case}: contact depth-order occlusion row scope missing")

        objects = frame.get("objects") if isinstance(frame.get("objects"), list) else []
        require(objects, f"{case}: frame {frame.get('frame_idx')} has no object rows")
        for obj in objects:
            require(isinstance(obj, dict), f"{case}: non-dict object row")
            counts["object_states"] += 1
            if isinstance(obj.get("physical_state_decision"), dict) and obj.get("physical_state_decision", {}).get("decision"):
                counts["object_physical_state_rows"] += 1
            if obj.get("mask_path"):
                require(Path(str(obj.get("mask_path"))).exists(), f"{case}: object mask path does not exist")
            if isinstance(obj.get("object_se3_observation"), dict):
                counts["object_se3_rows"] += 1
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            if geom:
                counts["object_visible_geometry_rows"] += 1
                if isinstance(geom.get("world_vertices_sample_m"), list) and geom.get("world_vertices_sample_m"):
                    counts["object_vertex_sample_rows"] += 1
                if geom.get("weak_visible_depth_pose_candidate") is True:
                    require("weak" in str(geom.get("geometry_strength")), f"{case}: weak visible-depth row missing explicit weak geometry strength")
            hidden = obj.get("hidden_geometry_candidate")
            if hidden is not None:
                counts["object_hidden_or_unresolved_geometry_rows"] += 1
            validation = obj.get("object_depth_silhouette_pose_validation") if isinstance(obj.get("object_depth_silhouette_pose_validation"), dict) else None
            if isinstance(validation, dict):
                counts["object_depth_silhouette_pose_validation_rows"] += 1
                if validation.get("visible_depth_silhouette_pose_supported") is True:
                    counts["object_depth_silhouette_pose_supported_rows"] += 1
                assessment = validation.get("compact_multiview_geometry_completion_assessment") if isinstance(validation.get("compact_multiview_geometry_completion_assessment"), dict) else {}
                if validation.get("object_geometry_complete") is True or validation.get("object_pose_requirement_met") is True:
                    require(assessment.get("object_geometry_complete") is True and assessment.get("object_pose_requirement_met") is True, f"{case}: object completion lacks compact multiview assessment support")
                    schema = obj.get("physical_state_schema") if isinstance(obj.get("physical_state_schema"), dict) else {}
                    physical = str(schema.get("model_physical_state_type") or obj.get("physical_state_label") or "unknown")
                    require(assessment.get("schema_eligible_compact_object") is True, f"{case}: object completion schema is not compact eligible")
                    require(physical == "rigid", f"{case}: object completion is not clean rigid compact geometry")
                    require(schema.get("surface_change_without_pose_state") is not True, f"{case}: surface-changing object completion overclaims hidden geometry/pose")
                    if schema.get("surface_appearance_changes") is True:
                        require(assessment.get("surface_appearance_compatible_with_compact_completion") is True, f"{case}: surface-appearance object completion lacks structured compact-compatibility support")
                        require(schema.get("pose_model_allowed_by_structured_vlm") is True, f"{case}: surface-appearance completion lacks structured pose-model allowance")
                        require(str(schema.get("geometry_changes")) in {"none", "minor_surface_layer_or_texture_change"}, f"{case}: surface-appearance completion has non-minor geometry change")
                    require(schema.get("requires_part_or_relative_motion_model") is not True, f"{case}: part/relative-motion object completion overclaims single-object geometry")
                    require(schema.get("secondary_deformable_or_surface_component") is not True, f"{case}: deformable/surface-component object completion overclaims compact geometry")
                    require(assessment.get("current_frame_visible_depth_silhouette_pose_supported") is True, f"{case}: object completion lacks current-frame visible pose support")
                    require(int(assessment.get("source_frame_count") or 0) >= int(assessment.get("min_source_frame_count") or 100), f"{case}: object completion has too few source frames")
                    require(max(int(assessment.get("source_point_count") or 0), int(assessment.get("sampled_point_count") or 0)) >= int(assessment.get("min_depth_point_count") or 5000), f"{case}: object completion has too few depth points")
                    require(int(assessment.get("convex_hull_faces") or 0) >= int(assessment.get("min_convex_hull_faces") or 40), f"{case}: object completion hull too sparse")
                    require(int(assessment.get("poisson_vertices") or 0) >= int(assessment.get("min_poisson_vertices") or 1000), f"{case}: object completion poisson mesh too sparse")
                    require("not_category_primitive_not_centroid" in str(assessment.get("scope")), f"{case}: object completion assessment scope missing anti-proxy guarantee")
                    counts["object_geometry_complete_rows"] += int(validation.get("object_geometry_complete") is True)
                    counts["object_pose_requirement_met_rows"] += int(validation.get("object_pose_requirement_met") is True)
                require("visible_depth" in str(validation.get("scope")), f"{case}: object pose validation scope missing")
            recon = obj.get("reconstructed_geometry_pose") if isinstance(obj.get("reconstructed_geometry_pose"), dict) else None
            if isinstance(recon, dict):
                counts["object_reconstructed_geometry_pose_rows"] += 1
                if recon.get("renderable_pose_geometry") is True:
                    counts["object_renderable_reconstructed_geometry_pose_rows"] += 1
                    require(isinstance(recon.get("mesh_path"), str) and Path(str(recon.get("mesh_path"))).exists(), f"{case}: reconstructed geometry mesh path missing")
                    require(isinstance(recon.get("world_bbox_corners_m"), list) and len(recon.get("world_bbox_corners_m")) == 8, f"{case}: reconstructed geometry pose missing render corners")
                    require(isinstance(recon.get("translation_world_m"), list) and len(recon.get("translation_world_m")) == 3, f"{case}: reconstructed geometry pose missing translation")
                    recon_assessment = recon.get("compact_multiview_geometry_completion_assessment") if isinstance(recon.get("compact_multiview_geometry_completion_assessment"), dict) else {}
                    if recon.get("object_pose_requirement_met") is True or recon.get("object_geometry_complete") is True:
                        require(recon_assessment.get("object_pose_requirement_met") is True and recon_assessment.get("object_geometry_complete") is True, f"{case}: reconstructed geometry completion lacks compact multiview support")
                        require("not_category_primitive_not_centroid" in str(recon_assessment.get("scope")), f"{case}: reconstructed geometry completion scope missing anti-proxy guarantee")
                    require(recon.get("visible_depth_silhouette_pose_supported") in {True, False}, f"{case}: reconstructed geometry pose missing object depth/silhouette validation support field")
                    if geom.get("weak_visible_depth_pose_candidate") is True:
                        require(recon.get("rigid_pose_supported_visible_mesh") is not True, f"{case}: weak visible-depth row must not support strict rigid pose")
            if isinstance(obj.get("object_geometry_complete"), bool):
                require(validation is not None, f"{case}: root object geometry completion field lacks validation row")
                require(obj.get("object_geometry_complete") == bool(validation.get("object_geometry_complete") is True), f"{case}: root object geometry completion disagrees with validation")
            if isinstance(obj.get("object_pose_requirement_met"), bool):
                require(validation is not None, f"{case}: root object pose requirement field lacks validation row")
                require(obj.get("object_pose_requirement_met") == bool(validation.get("object_pose_requirement_met") is True), f"{case}: root object pose requirement disagrees with validation")
            structured = obj.get("part_structured_pose_state") if isinstance(obj.get("part_structured_pose_state"), dict) else None
            if isinstance(structured, dict):
                counts["part_structured_object_pose_state_rows"] += 1
                schema = obj.get("physical_state_schema") if isinstance(obj.get("physical_state_schema"), dict) else {}
                parts = [p for p in obj.get("parts", []) if isinstance(p, dict)] if isinstance(obj.get("parts"), list) else []
                current_frame_labels = sorted(str(p.get("part_track_label")) for p in parts if p.get("part_track_label"))
                current_frame_ready_labels = []
                for part in parts:
                    recon_part = part.get("reconstructed_part_geometry_pose") if isinstance(part.get("reconstructed_part_geometry_pose"), dict) else {}
                    if recon_part.get("part_pose_ready") is True and part.get("part_track_label"):
                        current_frame_ready_labels.append(str(part.get("part_track_label")))
                current_frame_ready_labels = sorted(current_frame_ready_labels)
                manifest_required_labels = accepted_global_labels_by_object.get(str(obj.get("object_id")), [])
                required_labels = sorted(str(label) for label in structured.get("accepted_global_part_track_labels", []) if isinstance(label, str))
                if schema.get("requires_part_or_relative_motion_model") is True:
                    require(required_labels == manifest_required_labels, f"{case}: structured part-pose accepted global labels disagree with part-object manifest")
                ready_labels = sorted(str(label) for label in structured.get("ready_part_track_labels", []) if isinstance(label, str))
                require(structured.get("current_frame_part_track_labels") == current_frame_labels, f"{case}: structured part-pose current-frame labels disagree with object parts")
                require(structured.get("current_frame_ready_part_track_labels") == current_frame_ready_labels, f"{case}: structured part-pose current-frame ready labels disagree with object parts")
                require(structured.get("required_part_track_labels") == required_labels, f"{case}: structured part-pose required labels disagree with accepted global labels")
                tracked_labels = structured.get("tracked_part_labels")
                require(tracked_labels is None or tracked_labels == required_labels, f"{case}: structured part-pose tracked labels disagree with accepted global labels")
                require(all(label in required_labels for label in ready_labels), f"{case}: structured part-pose ready labels include non-global part tracks")
                unready_labels = sorted(label for label in required_labels if label not in set(ready_labels))
                structured_unready_labels = structured.get("unready_part_track_labels")
                require(structured_unready_labels is None or structured_unready_labels == unready_labels, f"{case}: structured part-pose unready labels disagree with accepted global labels")
                missing_current_frame_labels = sorted(label for label in required_labels if label not in set(current_frame_labels))
                require(structured.get("missing_current_frame_part_track_labels") == missing_current_frame_labels, f"{case}: structured part-pose missing current-frame labels disagree with accepted global labels")
                require(structured.get("object_pose_requirement_met") is False, f"{case}: structured part-pose overclaims object pose completion")
                require(structured.get("object_geometry_complete") is False, f"{case}: structured part-pose overclaims hidden geometry completion")
                require("not_hidden_geometry_completion" in str(structured.get("scope")), f"{case}: structured part-pose scope missing hidden-geometry limit")
                supported = bool(structured.get("part_structured_pose_ready") is True)
                require(obj.get("part_structured_pose_ready") is supported, f"{case}: root structured part-pose readiness disagrees with state")
                if supported:
                    counts["part_structured_object_pose_ready_rows"] += 1
                    require(schema.get("requires_part_or_relative_motion_model") is True, f"{case}: structured part-pose ready on object without part/relative-motion schema")
                    require(structured.get("part_structured_pose_support_mode") == "visible_base_reference_plus_ready_moving_part", f"{case}: structured part-pose ready has unsupported mode")
                    require(structured.get("base_visible_surface_reference_available") is True, f"{case}: structured part-pose ready lacks visible base reference support")
                    require(structured.get("base_visible_surface_reference_not_object_pose") is True, f"{case}: structured part-pose base reference overclaims object pose")
                    require(len(required_labels) >= 1, f"{case}: structured part-pose ready without accepted global part tracks")
                    require(len(ready_labels) >= 1, f"{case}: structured part-pose ready without any ready moving part")
                    require(isinstance(structured.get("ready_parts"), list) and len(structured.get("ready_parts")) == len(ready_labels), f"{case}: structured part-pose ready missing ready part pose records")
                    residual_uncertainty = structured.get("residual_uncertainty") if isinstance(structured.get("residual_uncertainty"), list) else []
                    for label in unready_labels:
                        require(any(str(label) in str(item) for item in residual_uncertainty), f"{case}: structured part-pose ready drops residual uncertainty for {label}")
                    require("not_whole_object_pose" in str(structured.get("scope")), f"{case}: structured part-pose scope overclaims whole-object pose")
                    require(structured.get("object_pose_requirement_met") is False and structured.get("object_geometry_complete") is False, f"{case}: structured part-pose ready overclaims object completion")
            for part in obj.get("parts") if isinstance(obj.get("parts"), list) else []:
                if not isinstance(part, dict):
                    continue
                counts["part_rows"] += 1
                if part.get("part_mask_path"):
                    require(Path(str(part.get("part_mask_path"))).exists(), f"{case}: part mask path does not exist")
                validation = part.get("part_silhouette_depth_pose_validation") if isinstance(part.get("part_silhouette_depth_pose_validation"), dict) else None
                if isinstance(validation, dict):
                    counts["part_silhouette_depth_pose_validation_rows"] += 1
                    if validation.get("visible_depth_silhouette_pose_supported") is True:
                        counts["part_silhouette_depth_pose_supported_rows"] += 1
                    require(validation.get("object_pose_requirement_met") is False, f"{case}: part validation overclaims object pose")
                    require("visible_same_frame_depth" in str(validation.get("scope")), f"{case}: part validation scope missing")
                    if "frame_visible_depth_silhouette_pose_supported" in validation:
                        frame_supported = bool(validation.get("frame_visible_depth_silhouette_pose_supported") is True)
                        require(validation.get("part_pose_ready") is frame_supported, f"{case}: part validation readiness disagrees with frame-local support")
                        counts["frame_local_part_pose_validation_rows"] += 1
                        if frame_supported:
                            counts["frame_local_part_pose_validation_supported_rows"] += 1
                        else:
                            counts["frame_local_part_pose_validation_rejected_rows"] += 1
                        require(validation.get("frame_local_validation_phase") == "graph", f"{case}: final part frame-local validation is not graph-phase")
                        require("same_frame_visible_depth" in str(validation.get("frame_local_validation_scope")), f"{case}: frame-local part validation scope missing")
                        require(isinstance(validation.get("frame_observed_to_predicted_median_m"), (int, float)), f"{case}: frame-local part validation missing median residual")
                        require(isinstance(validation.get("frame_observed_to_predicted_p95_m"), (int, float)), f"{case}: frame-local part validation missing p95 residual")
                part_recon = part.get("reconstructed_part_geometry_pose") if isinstance(part.get("reconstructed_part_geometry_pose"), dict) else None
                if isinstance(part_recon, dict):
                    counts["part_reconstructed_geometry_pose_rows"] += 1
                    if part_recon.get("renderable_part_pose_geometry") is True:
                        counts["part_renderable_reconstructed_geometry_pose_rows"] += 1
                        require(isinstance(part_recon.get("mesh_path"), str) and Path(str(part_recon.get("mesh_path"))).exists(), f"{case}: reconstructed part geometry mesh path missing")
                        require(isinstance(part_recon.get("part_bbox_corners_camera_m"), list) and len(part_recon.get("part_bbox_corners_camera_m")) == 8, f"{case}: reconstructed part geometry pose missing render corners")
                        require(isinstance(part_recon.get("translation_camera_m"), list) and len(part_recon.get("translation_camera_m")) == 3, f"{case}: reconstructed part geometry pose missing translation")
                        require(part_recon.get("part_pose_ready") in {True, False}, f"{case}: reconstructed part geometry missing part_pose_ready field")
                        require(part_recon.get("object_pose_requirement_met") is False, f"{case}: reconstructed part geometry overclaims object pose")
                        require(part_recon.get("visible_depth_silhouette_pose_supported") in {True, False}, f"{case}: reconstructed part geometry missing silhouette/depth pose support field")

        for hyp in frame.get("contact_hypotheses", []) if isinstance(frame.get("contact_hypotheses"), list) else []:
            if not isinstance(hyp, dict):
                continue
            counts["contacts"] += 1
            metric_contact = hyp.get("final_metric_contact_evidence") if isinstance(hyp.get("final_metric_contact_evidence"), dict) else None
            if isinstance(metric_contact, dict):
                counts["contacts_with_final_metric_distance"] += 1
                support_state = str(metric_contact.get("hand_support_state", ""))
                support_weight = metric_contact.get("hand_physical_factor_weight")
                require(support_state in {"observed_same_frame_detection", "inferred_no_same_frame_detection", "temporal_boundary_fill", "pipeline_gap_fill"}, f"{case}: final metric contact missing HaWoR support state")
                require(isinstance(support_weight, (int, float)) and 0.0 <= float(support_weight) <= 1.0, f"{case}: final metric contact has invalid HaWoR support weight")
                counts["contacts_with_hawor_support_weight"] += 1
                if support_state == "observed_same_frame_detection":
                    counts["contact_metric_observed_rows"] += 1
                elif support_state == "inferred_no_same_frame_detection":
                    counts["contact_metric_inferred_rows"] += 1
                elif support_state == "temporal_boundary_fill":
                    counts["contact_metric_boundary_fill_rows"] += 1
            evidence = hyp.get("evidence") if isinstance(hyp.get("evidence"), dict) else {}
            signed_np = evidence.get("signed_nonpenetration_evidence") if isinstance(evidence.get("signed_nonpenetration_evidence"), dict) else None
            if isinstance(signed_np, dict):
                counts["signed_nonpenetration_rows"] += 1
                if signed_np.get("mesh_watertight_by_edges") is True:
                    counts["signed_nonpenetration_watertight_rows"] += 1
                if signed_np.get("blocker") == "object_not_strict_rigid_nonpenetration_eligible":
                    counts["signed_nonpenetration_physical_ineligible_rows"] += 1
                    require(signed_np.get("strict_nonpenetration_eligibility") == "strict_rigid_nonpenetration_not_eligible", f"{case}: signed nonpenetration physical-ineligible row missing eligibility state")
                if str(signed_np.get("signed_nonpenetration_claim", "")).startswith("depth_fused_mesh_normal_") and str(signed_np.get("hand_support_state")) != "observed_same_frame_detection":
                    counts["signed_nonpenetration_evaluated_nonobserved_hawor_rows"] += 1
            triangle_np = evidence.get("triangle_nonpenetration_evidence") if isinstance(evidence.get("triangle_nonpenetration_evidence"), dict) else None
            if isinstance(triangle_np, dict):
                counts["triangle_nonpenetration_rows"] += 1
                if triangle_np.get("mesh_watertight_by_edges") is True:
                    counts["triangle_nonpenetration_watertight_rows"] += 1
                if triangle_np.get("blocker") == "object_not_strict_rigid_nonpenetration_eligible":
                    counts["triangle_nonpenetration_physical_ineligible_rows"] += 1
                    require(triangle_np.get("strict_nonpenetration_eligibility") == "strict_rigid_nonpenetration_not_eligible", f"{case}: triangle nonpenetration physical-ineligible row missing eligibility state")
                if str(triangle_np.get("triangle_nonpenetration_claim", "")).startswith("depth_fused_mesh_triangle_") and str(triangle_np.get("hand_support_state")) != "observed_same_frame_detection":
                    counts["triangle_nonpenetration_evaluated_nonobserved_hawor_rows"] += 1

    expected_hand_rows = expected * 2
    require(counts["hand_total"] == expected_hand_rows, f"{case}: hand rows do not cover full timeline")
    require(counts["hawor_metric_mano"] == expected_hand_rows, f"{case}: HaWoR metric MANO does not cover all hand rows")
    require(counts["hand_graph_metric"] == expected_hand_rows, f"{case}: graph hand variables do not all consume HaWoR metric MANO")
    require(counts["hand_support_state_rows"] == expected_hand_rows, f"{case}: HaWoR support state does not cover all hand rows")
    require(counts["hand_mano_surface_reference_rows"] == expected_hand_rows, f"{case}: MANO surface references do not cover all hand rows")
    require(counts["hand_mano_parameter_contract_rows"] == expected_hand_rows, f"{case}: MANO parameter contracts do not cover all hand rows")
    require(counts["hand_support_observed_rows"] > 0, f"{case}: no observed same-frame HaWoR rows")
    require(counts["wilor_key_rows"] == expected_hand_rows, f"{case}: WiLoR/V16 hand evidence keys missing")
    require(counts["rtmlib_key_rows"] == expected_hand_rows, f"{case}: RTMLib hand evidence keys missing")
    require(counts["pose_fill_gate_rows"] == expected_hand_rows, f"{case}: pose fill gate rows do not cover both hands/full timeline")
    if counts["pose_fill_accepted_rows"] > 0:
        require(counts["pose_fill_accepted_rows"] == counts["pose_fill_observed_mano_rows"] + counts["pose_fill_temporal_rows"], f"{case}: accepted pose-fill rows are not classified")
        require(int(overlay_draw.get("pose_fill_accepted_markers", 0)) == counts["pose_fill_accepted_rows"], f"{case}: overlay accepted pose-fill markers do not match backing state")
        require(int(world_draw.get("world_pose_fill_accepted_markers", 0)) == counts["pose_fill_accepted_rows"], f"{case}: world accepted pose-fill markers do not match backing state")
    require(counts["object_states"] > 0, f"{case}: no object states")
    require(counts["object_physical_state_rows"] == counts["object_states"], f"{case}: physical-state decisions missing on object rows")
    require(counts["object_se3_rows"] == counts["object_states"], f"{case}: object SE3 observations missing on object rows")
    require(counts["object_visible_geometry_rows"] > 0, f"{case}: no depth-visible geometry rows")
    require(counts["object_vertex_sample_rows"] > 0, f"{case}: no visible geometry vertex samples")
    require(counts["object_hidden_or_unresolved_geometry_rows"] > 0, f"{case}: no hidden/unresolved geometry state rows")
    require(counts["object_reconstructed_geometry_pose_rows"] == counts["object_states"], f"{case}: reconstructed geometry pose state missing on object rows")
    require(counts["object_renderable_reconstructed_geometry_pose_rows"] > 0, f"{case}: no renderable reconstructed mesh pose rows")
    require(counts["object_depth_silhouette_pose_validation_rows"] > 0, f"{case}: no object depth/silhouette pose validation rows")
    require(counts["part_structured_object_pose_state_rows"] == counts["object_states"], f"{case}: structured part-pose state missing on object rows")
    if case == "task5_tomato_960":
        require(counts["object_depth_silhouette_pose_supported_rows"] > 0, f"{case}: no supported object depth/silhouette pose validation rows")
    require(counts["part_rows"] > 0, f"{case}: no part rows")
    require(counts["part_reconstructed_geometry_pose_rows"] == counts["part_rows"], f"{case}: reconstructed part geometry pose state missing on part rows")
    require(counts["part_renderable_reconstructed_geometry_pose_rows"] > 0, f"{case}: no renderable reconstructed part mesh pose rows")
    require(counts["part_silhouette_depth_pose_validation_rows"] > 0, f"{case}: no part silhouette/depth pose validation rows")
    require(counts["frame_local_part_pose_validation_rows"] == counts["part_silhouette_depth_pose_validation_rows"], f"{case}: frame-local part validation rows do not cover final part validation rows")
    require(counts["part_silhouette_depth_pose_supported_rows"] > 0, f"{case}: no supported part silhouette/depth pose validation rows")
    require(counts["factor_frames"] == expected, f"{case}: factor graph not present for every frame")
    require(counts["camera_depth_observed_rows"] > 0, f"{case}: no observed camera/depth correction rows")
    require(counts["contacts"] > 0, f"{case}: no contact hypotheses")
    require(counts["contact_switch_vars"] == counts["contacts"], f"{case}: contact switch variables do not cover contact hypotheses")
    require(counts["active_contact_pose_coupled_rows"] == int(contact_pose_anchor_fixed_point.get("emitted_anchor_factor_count", -1)), f"{case}: active pose-coupled row count does not match emitted stable anchor factors")
    require(counts["deformable_surface_patch_vars"] > 0, f"{case}: no deformable surface patch variables")
    require(counts["active_deformable_surface_patch_coupled_rows"] > 0, f"{case}: no active deformable contacts coupled to local patch state")
    require(counts["active_contact_switch_vars"] > 0 or counts["raw_contact_switches_gated_by_physical_support"] > 0, f"{case}: neither active physical contacts nor physically gated raw contact evidence exists")
    require(counts["active_contact_switch_vars_with_nonobserved_hawor_hand"] == 0, f"{case}: non-observed HaWoR hand rows still produce active contact switches")
    require(counts["contacts_with_final_metric_distance"] > 0, f"{case}: no final metric MANO-to-object-surface distances")
    require(counts["contacts_with_hawor_support_weight"] == counts["contacts_with_final_metric_distance"], f"{case}: final metric contact distances missing HaWoR support weights")
    require(counts["signed_nonpenetration_rows"] > 0, f"{case}: no signed nonpenetration evidence rows")
    require(counts["signed_nonpenetration_watertight_rows"] > 0 or counts["signed_nonpenetration_physical_ineligible_rows"] > 0, f"{case}: signed nonpenetration has neither watertight evaluation nor physical-eligibility blockers")
    require(counts["signed_nonpenetration_evaluated_nonobserved_hawor_rows"] == 0, f"{case}: evaluated signed nonpenetration rows are not support-gated to observed HaWoR hands")
    require(counts["triangle_nonpenetration_rows"] > 0, f"{case}: no triangle nonpenetration evidence rows")
    require(counts["triangle_nonpenetration_watertight_rows"] > 0 or counts["triangle_nonpenetration_physical_ineligible_rows"] > 0, f"{case}: triangle nonpenetration has neither watertight evaluation nor physical-eligibility blockers")
    require(counts["triangle_nonpenetration_evaluated_nonobserved_hawor_rows"] == 0, f"{case}: evaluated triangle nonpenetration rows are not support-gated to observed HaWoR hands")
    require(counts["occlusion_owner_vars"] > 0, f"{case}: no occlusion owner graph variables")
    require(counts["contact_physical_mode_active"] == counts["active_contact_switch_vars"], f"{case}: active contact mode count does not match active contact switches")
    contact_report_counts = physical_contact_state_report.get("counts") if isinstance(physical_contact_state_report.get("counts"), dict) else {}
    require(bool(physical_contact_state_report.get("render_counts_excluded_from_contact_semantics")) is True, f"{case}: contact state report does not exclude render counts")
    require(int(contact_report_counts.get("active_frame_pair_states", -1)) == counts["active_contact_switch_vars"], f"{case}: contact state report active frame-pair count mismatch")
    require(int(contact_report_counts.get("active_temporal_contact_episodes_consecutive", 0)) > 0, f"{case}: contact state report lacks temporal episodes")
    require(isinstance(physical_contact_state_report.get("temporal_episodes"), list) and len(physical_contact_state_report.get("temporal_episodes")) == int(contact_report_counts.get("active_temporal_contact_episodes_consecutive", -1)), f"{case}: contact temporal episode list/count mismatch")
    require(counts["hand_contact_depth_order_occlusion_rows"] == counts["contact_physical_mode_depth_occluded_possible"], f"{case}: contact depth-order occlusion hand evidence does not match depth-occluded possible contact modes")
    require(int(overlay_draw.get("contact_lines", 0)) <= counts["active_contact_switch_vars"], f"{case}: overlay draws more contact lines than active physical contacts")
    world_active_drawn = int(world_draw.get("world_contact_edges", 0)) + int(world_draw.get("world_contact_episode_state_edges", 0))
    require(world_active_drawn <= counts["active_contact_switch_vars"], f"{case}: world render draws more active contact states than solved active contacts")
    require(int(world_draw.get("world_active_contact_missing_metric_endpoints", 0)) == 0, f"{case}: direct active contact world render is missing metric endpoints")
    require(int(overlay_draw.get("contact_depth_occluded_possible_lines", 0)) <= counts["contact_physical_mode_depth_occluded_possible"], f"{case}: overlay draws more depth-occluded possible contact lines than solved modes")
    require(int(overlay_draw.get("contact_supported_near_noncontact_lines", 0)) <= counts["contact_physical_mode_supported_near_noncontact"], f"{case}: overlay draws more supported-near lines than solved modes")
    require(int(world_draw.get("world_contact_depth_occluded_possible_lines", 0)) <= counts["contact_physical_mode_depth_occluded_possible"], f"{case}: world render draws more depth-occluded possible contact edges than solved modes")
    require(int(world_draw.get("world_contact_supported_near_noncontact_lines", 0)) <= counts["contact_physical_mode_supported_near_noncontact"], f"{case}: world render draws more supported-near edges than solved modes")
    require(int(overlay_draw.get("occlusion_owner_edges", 0)) <= counts["occlusion_owner_supported_vars"], f"{case}: overlay draws unsupported occlusion owner edges")
    require(int(world_draw.get("world_occlusion_owner_edges", 0)) <= counts["occlusion_owner_supported_vars"], f"{case}: world render draws unsupported occlusion owner edges")
    require(int(overlay_draw.get("part_structured_object_pose_ready_labels", 0)) <= counts["part_structured_object_pose_ready_rows"], f"{case}: overlay draws unsupported structured part-object pose readiness")
    require(int(world_draw.get("world_part_structured_object_pose_ready_labels", 0)) <= counts["part_structured_object_pose_ready_rows"], f"{case}: world render draws unsupported structured part-object pose readiness")
    require(counts["hand_occlusion_owner_accepted_rows_with_nonobserved_hawor_hand"] == 0, f"{case}: non-observed HaWoR hand rows still produce accepted hand occlusion-owner claims")
    require(counts["occlusion_owner_supported_vars_with_nonobserved_hawor_hand"] == 0, f"{case}: non-observed HaWoR hand rows still produce supported occlusion-owner factor claims")
    return {"case": case, "expected_frame_count": expected, **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json"))
    args = parser.parse_args()
    report_text = args.report.read_text(encoding="utf-8")
    self_inspection_path = args.report.parent / "v18_completion_self_inspection.json"
    if self_inspection_path.exists():
        report_text = report_text + "\n" + self_inspection_path.read_text(encoding="utf-8")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    require(report.get("all_frame_counts_match") is True, "global frame count mismatch")
    cases = report.get("cases")
    require(isinstance(cases, list) and len(cases) == 2, "report must contain the two representative cases")
    rows = [validate_case(case_report, report_text) for case_report in cases if isinstance(case_report, dict)]
    print(json.dumps({"validation": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
