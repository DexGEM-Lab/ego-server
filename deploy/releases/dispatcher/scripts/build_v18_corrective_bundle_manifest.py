#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(st.st_mtime))}


def frame_count_key(output_key: str) -> str:
    mapping = {
        "overlay_video": "overlay",
        "world_video": "world",
        "side_by_side_video": "side_by_side",
        "video": "video",
    }
    return mapping.get(output_key, output_key.removesuffix("_video"))


def video_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), dict) else {}
    frame_counts = report.get("frame_counts", {}) if isinstance(report.get("frame_counts"), dict) else {}
    expected = report.get("frame_count")
    out = []
    for key, value in outputs.items():
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        actual = frame_counts.get(frame_count_key(key))
        out.append({**file_info(path), "role": key, "frame_count": actual, "expected_frame_count": expected, "frame_count_matches": actual == expected})
    return out


def case_bundle(case: str, root: Path) -> dict[str, Any]:
    graph = load_json(root / case / "v18_corrective_state_report.json")
    rigid = load_json(root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json")
    visible = load_json(root / case / "visible_surface_state" / "v18_visible_surface_state_report.json")
    geometry_coverage = load_json(root / case / "geometry_coverage_audit" / "v18_geometry_coverage_audit_report.json")
    hawor = load_json(root / case / "hawor_ghost_attempt" / "v18_hawor_ghost_attempt_report.json")
    hand_smoothing = load_json(root / case / "temporal_hand_pose_smoothing" / "v18_temporal_hand_pose_smoothing_report.json")
    owner = load_json(root / case / "occlusion_owner_best_effort" / "v18_occlusion_owner_best_effort_report.json")
    owner_audit = load_json(root / case / "occlusion_owner_acceptance_audit" / "v18_occlusion_owner_acceptance_audit_report.json")
    contact = load_json(root / case / "contact_nonpenetration_state" / "v18_contact_nonpenetration_state_report.json")
    contact_audit = load_json(root / case / "contact_acceptance_audit" / "v18_contact_acceptance_audit_report.json")
    residual = load_json(root / case / "rigid_se3_residual_check" / "v18_rigid_se3_residual_check_report.json")
    repair = load_json(root / case / "nonpenetration_repair_proposal" / "v18_nonpenetration_repair_proposal_report.json")
    montage = load_json(root / case / "corrective_montage" / "v18_corrective_montage_report.json")
    mano = load_json(root / "mano_foundation_audit" / case / "v18_mano_foundation_state_report.json")
    mano_overlay = load_json(root / "mano_foundation_audit" / case / "v18_mano_foundation_overlay_report.json")
    hawor_requirement_path = root / "hawor_requirement_state" / "v18_hawor_requirement_state.json"
    hawor_requirement = load_json(hawor_requirement_path) if hawor_requirement_path.exists() else {}
    hawor_requirement_cases = hawor_requirement.get("cases") if isinstance(hawor_requirement.get("cases"), list) else []
    hawor_requirement_case = next((row for row in hawor_requirement_cases if isinstance(row, dict) and row.get("case") == case), {})
    hawor_bridge_summary_path = root / "hawor_bridge_state" / "v18_hawor_bridge_state_summary.json"
    hawor_bridge = load_json(hawor_bridge_summary_path) if hawor_bridge_summary_path.exists() else {}
    hawor_bridge_cases = hawor_bridge.get("cases") if isinstance(hawor_bridge.get("cases"), list) else []
    hawor_bridge_case = next((row for row in hawor_bridge_cases if isinstance(row, dict) and row.get("case") == case), {})
    hawor_quality_summary_path = root / "hawor_bridge_state" / "v18_hawor_bridge_quality_state_summary.json"
    hawor_quality = load_json(hawor_quality_summary_path) if hawor_quality_summary_path.exists() else {}
    hawor_quality_cases = hawor_quality.get("cases") if isinstance(hawor_quality.get("cases"), list) else []
    hawor_quality_case = next((row for row in hawor_quality_cases if isinstance(row, dict) and row.get("case") == case), {})
    hawor_coverage_summary_path = root / "hawor_bridge_state" / "v18_hawor_bridge_downstream_coverage_summary.json"
    hawor_coverage = load_json(hawor_coverage_summary_path) if hawor_coverage_summary_path.exists() else {}
    hawor_coverage_cases = hawor_coverage.get("cases") if isinstance(hawor_coverage.get("cases"), list) else []
    hawor_coverage_case = next((row for row in hawor_coverage_cases if isinstance(row, dict) and row.get("case") == case), {})
    hawor_subset_summary_path = root / "hawor_bridge_state" / "v18_hawor_bridge_subset_policy_summary.json"
    hawor_subset = load_json(hawor_subset_summary_path) if hawor_subset_summary_path.exists() else {}
    hawor_subset_cases = hawor_subset.get("cases") if isinstance(hawor_subset.get("cases"), list) else []
    hawor_subset_case = next((row for row in hawor_subset_cases if isinstance(row, dict) and row.get("case") == case), {})
    ann_path = root / case / "annotations_v18_corrective_state.json"
    ann = load_json(ann_path)
    review_sheets = sorted((root / "review_sheets").glob(f"{case}_*_corrective_review.jpg"))
    reports = {
        "graph_corrective_render": graph,
        "generic_rigid_se3_attempt": rigid,
        "frame_local_visible_surface": visible,
        "geometry_coverage_audit": geometry_coverage,
        "mano_foundation_overlay": mano_overlay,
        "hawor_ghost_or_failure": hawor,
        "temporal_hand_pose_smoothing": hand_smoothing,
        "tentative_occlusion_owner": owner,
        "occlusion_owner_acceptance_audit": owner_audit,
        "contact_nonpenetration": contact,
        "contact_acceptance_audit": contact_audit,
        "rigid_se3_residual_check": residual,
        "nonpenetration_repair_proposal": repair,
        "corrective_montage": montage,
    }
    videos: list[dict[str, Any]] = []
    for name, report in reports.items():
        for entry in video_entries(report):
            entry["mechanism"] = name
            videos.append(entry)
    return {
        "case": case,
        "annotation_state": {**file_info(ann_path), "frame_count": ann.get("frame_count"), "counts": ann.get("counts"), "claim_scope": ann.get("claim_scope")},
        "mechanisms": {
            "graph_corrective_render": {"draw_counts": graph.get("draw_counts"), "jitter_probe": graph.get("jitter_probe"), "claim_scope": graph.get("claim_scope")},
            "generic_rigid_se3_attempt": {"candidate_objects": rigid.get("candidate_objects"), "claim_scope": rigid.get("claim_scope")},
            "frame_local_visible_surface": {"candidate_objects": visible.get("candidate_objects"), "claim_scope": visible.get("claim_scope")},
            "geometry_coverage_audit": {"object_count": geometry_coverage.get("object_count"), "status_counts": geometry_coverage.get("status_counts"), "object_summaries": geometry_coverage.get("object_summaries"), "claim_scope": geometry_coverage.get("claim_scope")},
            "hawor_ghost_or_failure": {"measurement_rows": hawor.get("measurement_rows"), "draw_counts": hawor.get("draw_counts"), "execution_failure_logs": hawor.get("execution_failure_logs"), "claim_scope": hawor.get("claim_scope")},
            "hawor_hard_requirement_state": {"status": hawor_requirement_case.get("status"), "hard_requirement": hawor_requirement_case.get("hard_requirement"), "accepted_v18_hawor_requirement_met": hawor_requirement_case.get("accepted_v18_hawor_requirement_met"), "accepted_metric_hand_state_from_hawor": hawor_requirement_case.get("accepted_metric_hand_state_from_hawor"), "available_hawor_frame_side_rows": hawor_requirement_case.get("available_hawor_frame_side_rows"), "expected_frame_side_rows": hawor_requirement_case.get("expected_frame_side_rows"), "full_timeline_hawor_npz_shape_valid": hawor_requirement_case.get("full_timeline_hawor_npz_shape_valid"), "current_v18_bridge_candidate": hawor_requirement_case.get("current_v18_bridge_candidate"), "blocking_reasons": hawor_requirement_case.get("blocking_reasons"), "claim_scope": hawor_requirement_case.get("claim_scope")},
            "hawor_bridge_state": {"status": hawor_bridge_case.get("status"), "bridge_candidate_rows": hawor_bridge_case.get("bridge_candidate_rows"), "expected_frame_side_rows": hawor_bridge_case.get("expected_frame_side_rows"), "accepted_v18_hawor_foundation": hawor_bridge_case.get("accepted_v18_hawor_foundation"), "reference_projection_residual_px_median_per_row": hawor_bridge_case.get("reference_projection_residual_px_median_per_row"), "reference_projection_residual_threshold_counts": hawor_bridge_case.get("reference_projection_residual_threshold_counts"), "bridge_candidate_npz": hawor_bridge_case.get("bridge_candidate_npz"), "review_report": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_review_report.json"), "review_sheet": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_residual_review_sheet.jpg"), "blocking_reasons": hawor_bridge_case.get("blocking_reasons"), "claim_scope": hawor_bridge_case.get("claim_scope")},
            "hawor_bridge_quality_state": {"status": hawor_quality_case.get("status"), "bridge_candidate_rows": hawor_quality_case.get("bridge_candidate_rows"), "expected_frame_side_rows": hawor_quality_case.get("expected_frame_side_rows"), "quality_counts": hawor_quality_case.get("quality_counts"), "accepted_v18_hawor_foundation": hawor_quality_case.get("accepted_v18_hawor_foundation"), "v18_physical_hand_state_valid_from_quality": hawor_quality_case.get("v18_physical_hand_state_valid_from_quality"), "projection_residual_px_median_per_row": hawor_quality_case.get("projection_residual_px_median_per_row"), "supported_candidate_projection_residual_px_median_per_row": hawor_quality_case.get("supported_candidate_projection_residual_px_median_per_row"), "quality_state_report": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_state.json"), "quality_overlay_report": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_overlay_report.json"), "quality_overlay_video": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_overlay.mp4"), "blocking_reasons": hawor_quality_case.get("blocking_reasons"), "claim_scope": hawor_quality_case.get("claim_scope")},
            "hawor_bridge_downstream_coverage": {"status": hawor_coverage_case.get("status"), "hawor_bridge_quality_candidate_hand_rows": hawor_coverage_case.get("hawor_bridge_quality_candidate_hand_rows"), "hawor_bridge_projection_supported_hand_rows": hawor_coverage_case.get("hawor_bridge_projection_supported_hand_rows"), "existing_contact_acceptance_audit_rows": hawor_coverage_case.get("existing_contact_acceptance_audit_rows"), "existing_contact_rows_with_projection_supported_hawor_bridge": hawor_coverage_case.get("existing_contact_rows_with_projection_supported_hawor_bridge"), "existing_occlusion_acceptance_audit_rows": hawor_coverage_case.get("existing_occlusion_acceptance_audit_rows"), "existing_occlusion_rows_with_projection_supported_hawor_bridge": hawor_coverage_case.get("existing_occlusion_rows_with_projection_supported_hawor_bridge"), "existing_contact_nonpenetration_hands": hawor_coverage_case.get("existing_contact_nonpenetration_hands"), "existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge": hawor_coverage_case.get("existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge"), "accepted_contact_or_occlusion_input_flags": hawor_coverage_case.get("accepted_contact_or_occlusion_input_flags"), "coverage_report": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_downstream_coverage_report.json"), "blocking_reasons": hawor_coverage_case.get("blocking_reasons"), "claim_scope": hawor_coverage_case.get("claim_scope")},
            "hawor_bridge_subset_policy": {"status": hawor_subset_case.get("status"), "quality_state_status": hawor_subset_case.get("quality_state_status"), "policy_counts": hawor_subset_case.get("policy_counts"), "existing_contact_acceptance_audit_rows_in_strict_candidate_queue": hawor_subset_case.get("existing_contact_acceptance_audit_rows_in_strict_candidate_queue"), "existing_contact_acceptance_audit_rows_total": hawor_subset_case.get("existing_contact_acceptance_audit_rows_total"), "existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue": hawor_subset_case.get("existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue"), "existing_occlusion_acceptance_audit_rows_total": hawor_subset_case.get("existing_occlusion_acceptance_audit_rows_total"), "existing_contact_nonpenetration_hands_in_strict_candidate_queue": hawor_subset_case.get("existing_contact_nonpenetration_hands_in_strict_candidate_queue"), "existing_contact_nonpenetration_hands_total": hawor_subset_case.get("existing_contact_nonpenetration_hands_total"), "foundation_acceptance_from_policy": hawor_subset_case.get("foundation_acceptance_from_policy"), "metric_hand_state_acceptance_from_policy": hawor_subset_case.get("metric_hand_state_acceptance_from_policy"), "downstream_acceptance_from_policy": hawor_subset_case.get("downstream_acceptance_from_policy"), "policy_report": file_info(root / "hawor_bridge_state" / case / "v18_hawor_bridge_subset_policy_report.json"), "blocking_reasons": hawor_subset_case.get("blocking_reasons"), "claim_scope": hawor_subset_case.get("claim_scope")},
            "temporal_hand_pose_smoothing": {"draw_counts": hand_smoothing.get("draw_counts"), "jitter_probe": hand_smoothing.get("jitter_probe"), "claim_scope": hand_smoothing.get("claim_scope")},
            "tentative_occlusion_owner": {"selected_tentative_owner_rows": owner.get("selected_tentative_owner_rows"), "strict_accepted_owner_rows": owner.get("strict_accepted_owner_rows"), "owner_object_counts": owner.get("owner_object_counts"), "acceptance_blocker_counts": owner.get("acceptance_blocker_counts"), "claim_scope": owner.get("claim_scope")},
            "occlusion_owner_acceptance_audit": {"candidate_rows": owner_audit.get("candidate_rows"), "strict_promotable_owner_rows": owner_audit.get("strict_promotable_owner_rows"), "category_counts": owner_audit.get("category_counts"), "claim_scope": owner_audit.get("claim_scope")},
            "contact_nonpenetration": {"contact_graph_selected_rows": contact.get("contact_graph_selected_rows"), "source_graph_contact_candidate_rows_before_nonpenetration_veto": contact.get("source_graph_contact_candidate_rows_before_nonpenetration_veto"), "signed_local_penetration_rows": contact.get("signed_local_penetration_rows"), "triangle_local_penetration_rows": contact.get("triangle_local_penetration_rows"), "mesh_watertight_rows": contact.get("mesh_watertight_rows"), "claim_scope": contact.get("claim_scope")},
            "contact_acceptance_audit": {"selected_contact_rows": contact_audit.get("selected_contact_rows"), "strict_promotable_contact_rows": contact_audit.get("strict_promotable_contact_rows"), "category_counts": contact_audit.get("category_counts"), "claim_scope": contact_audit.get("claim_scope")},
            "rigid_se3_residual_check": {"candidate_objects": residual.get("candidate_objects"), "claim_scope": residual.get("claim_scope")},
            "nonpenetration_repair_proposal": {"proposal_rows": repair.get("proposal_rows"), "proposal_status_counts": repair.get("proposal_status_counts"), "claim_scope": repair.get("claim_scope")},
            "corrective_montage": {"panels": montage.get("panels"), "claim_scope": montage.get("claim_scope")},
            "mano_foundation_state": {"foundational_mano_state_valid": mano.get("foundational_mano_state_valid"), "blocking_reasons": mano.get("blocking_reasons"), "current_v18_full_mano_storage": mano.get("current_v18_full_mano_storage"), "recovered_wilor_virtual_camera_mano_candidates": mano.get("recovered_wilor_virtual_camera_mano_candidates"), "hawor_world_mano_candidates": mano.get("hawor_world_mano_candidates"), "claim_scope": mano.get("claim_scope")},
            "mano_foundation_overlay": {"available_frame_side_rows": mano_overlay.get("available_frame_side_rows"), "draw_counts": mano_overlay.get("draw_counts"), "foundational_mano_state_valid": mano_overlay.get("foundational_mano_state_valid"), "claim_scope": mano_overlay.get("claim_scope")},
        },
        "mano_foundation_artifacts": {
            "report": file_info(root / "mano_foundation_audit" / case / "v18_mano_foundation_state_report.json"),
            "wilor_virtual_camera_npz": file_info(Path(str(mano.get("recovered_wilor_virtual_camera_mano_candidates", {}).get("npz_path")))) if isinstance(mano.get("recovered_wilor_virtual_camera_mano_candidates"), dict) and mano.get("recovered_wilor_virtual_camera_mano_candidates", {}).get("npz_path") else {"exists": False},
            "overlay_report": file_info(root / "mano_foundation_audit" / case / "v18_mano_foundation_overlay_report.json"),
        },
        "videos": videos,
        "review_sheets": [file_info(p) for p in review_sheets],
        "all_listed_video_frame_counts_match": all(v.get("frame_count_matches") for v in videos),
    }


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# V18 corrective bundle manifest",
        "",
        "This manifest indexes changed V18 corrective artifacts. It does not claim full V18 closure.",
        "",
        f"Root: `{manifest['output_root']}`",
        f"All listed video frame counts match: `{manifest['all_listed_video_frame_counts_match']}`",
        "",
        "## Global artifacts",
        "",
        f"HaWoR provisioning audit: status `{manifest.get('hawor_provisioning_audit', {}).get('status')}`; missing `{manifest.get('hawor_provisioning_audit', {}).get('missing_required')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_provisioning_audit_report', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_provisioning_audit_markdown', {}).get('path')}`",
        f"MANO foundation summary: valid `{manifest.get('mano_foundation_audit', {}).get('all_cases_foundational_mano_valid')}`; physical pipeline valid `{manifest.get('mano_foundation_audit', {}).get('v18_physical_pipeline_valid_without_further_hand_work')}`",
        f"- `{manifest.get('global_artifacts', {}).get('mano_foundation_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('mano_foundation_markdown', {}).get('path')}`",
        f"HaWoR hard requirement: status `{manifest.get('hawor_hard_requirement_state', {}).get('status')}`; all cases met `{manifest.get('hawor_hard_requirement_state', {}).get('all_cases_hawor_requirement_met')}`; physical hand state valid `{manifest.get('hawor_hard_requirement_state', {}).get('v18_physical_hand_state_valid_from_hawor')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_requirement_state', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_requirement_markdown', {}).get('path')}`",
        f"Task5 HaWoR export contract: status `{manifest.get('hawor_task5_export_contract', {}).get('status')}`; expected output exists `{manifest.get('hawor_task5_export_contract', {}).get('expected_local_output_npz', {}).get('exists')}`; accepted requirement `{manifest.get('hawor_task5_export_contract', {}).get('acceptance_flags', {}).get('accepted_v18_hawor_requirement_met')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_task5_export_contract', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_task5_export_contract_markdown', {}).get('path')}`",
        f"HaWoR bridge state: status `{manifest.get('hawor_bridge_state', {}).get('status')}`; all cases accepted `{manifest.get('hawor_bridge_state', {}).get('all_cases_hawor_bridge_accepted')}`; physical hand state valid `{manifest.get('hawor_bridge_state', {}).get('v18_physical_hand_state_valid_from_bridge')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_markdown', {}).get('path')}`",
        f"HaWoR bridge quality state: status `{manifest.get('hawor_bridge_quality_state', {}).get('status')}`; all cases accepted `{manifest.get('hawor_bridge_quality_state', {}).get('all_cases_quality_foundation_accepted')}`; physical hand state valid `{manifest.get('hawor_bridge_quality_state', {}).get('v18_physical_hand_state_valid_from_quality')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_quality_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_quality_markdown', {}).get('path')}`",
        f"HaWoR bridge downstream coverage: status `{manifest.get('hawor_bridge_downstream_coverage', {}).get('status')}`; all cases downstream accepted `{manifest.get('hawor_bridge_downstream_coverage', {}).get('all_cases_downstream_accepted')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_downstream_coverage_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_downstream_coverage_markdown', {}).get('path')}`",
        f"HaWoR bridge subset policy: status `{manifest.get('hawor_bridge_subset_policy', {}).get('status')}`; all cases policy foundation accepted `{manifest.get('hawor_bridge_subset_policy', {}).get('all_cases_policy_foundation_accepted')}`; downstream accepted from policy `{manifest.get('hawor_bridge_subset_policy', {}).get('all_cases_downstream_accepted_from_policy')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_subset_policy_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_bridge_subset_policy_markdown', {}).get('path')}`",
        f"HaWoR strict contact probe: status `{manifest.get('hawor_strict_contact_probe', {}).get('status')}`; contact accepted `{manifest.get('hawor_strict_contact_probe', {}).get('contact_acceptance_from_probe')}`; nonpenetration accepted `{manifest.get('hawor_strict_contact_probe', {}).get('nonpenetration_acceptance_from_probe')}`; distance median `{manifest.get('hawor_strict_contact_probe', {}).get('distance_summary', {}).get('median')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_strict_contact_probe_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_strict_contact_probe_markdown', {}).get('path')}`",
        f"HaWoR temporal offset probe: status `{manifest.get('hawor_temporal_offset_probe', {}).get('status')}`; interpretation `{manifest.get('hawor_temporal_offset_probe', {}).get('interpretation')}`; contact accepted `{manifest.get('hawor_temporal_offset_probe', {}).get('contact_acceptance_from_probe')}`; foundation accepted `{manifest.get('hawor_temporal_offset_probe', {}).get('foundation_acceptance_from_probe')}`; best-distance offset `{manifest.get('hawor_temporal_offset_probe', {}).get('dominant_best_distance_offset')}` fraction `{manifest.get('hawor_temporal_offset_probe', {}).get('dominant_best_distance_fraction')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_temporal_offset_probe_summary', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('hawor_temporal_offset_probe_markdown', {}).get('path')}`",
        f"Post-bridge targeted validation: status `{manifest.get('post_bridge_targeted_validation', {}).get('status')}`; long pipeline rerun after bridge changes `{manifest.get('post_bridge_targeted_validation', {}).get('long_pipeline_rerun_after_bridge_changes')}`; active partial exists `{manifest.get('post_bridge_targeted_validation', {}).get('active_partial_pipeline_report', {}).get('exists')}`",
        f"- `{manifest.get('global_artifacts', {}).get('post_bridge_targeted_validation_report', {}).get('path')}`",
        f"- `{manifest.get('global_artifacts', {}).get('pipeline_report_scope_note', {}).get('path')}`",
        "",
    ]
    for case in manifest["cases"]:
        lines += [f"## {case['case']}", ""]
        ann = case["annotation_state"]
        lines += [f"Annotation state: `{ann['path']}`", f"Counts: `{ann.get('counts')}`", ""]
        owner = case["mechanisms"]["tentative_occlusion_owner"]
        hawor = case["mechanisms"]["hawor_ghost_or_failure"]
        hawor_hard = case["mechanisms"].get("hawor_hard_requirement_state", {})
        hawor_bridge = case["mechanisms"].get("hawor_bridge_state", {})
        hawor_quality = case["mechanisms"].get("hawor_bridge_quality_state", {})
        hawor_coverage = case["mechanisms"].get("hawor_bridge_downstream_coverage", {})
        hawor_subset = case["mechanisms"].get("hawor_bridge_subset_policy", {})
        hand_smoothing = case["mechanisms"]["temporal_hand_pose_smoothing"]
        owner_audit = case["mechanisms"]["occlusion_owner_acceptance_audit"]
        contact = case["mechanisms"]["contact_nonpenetration"]
        contact_audit = case["mechanisms"]["contact_acceptance_audit"]
        mano = case["mechanisms"]["mano_foundation_state"]
        wilor_mano = mano.get("recovered_wilor_virtual_camera_mano_candidates", {}) if isinstance(mano.get("recovered_wilor_virtual_camera_mano_candidates"), dict) else {}
        hawor_mano = mano.get("hawor_world_mano_candidates", {}) if isinstance(mano.get("hawor_world_mano_candidates"), dict) else {}
        lines += [
            f"MANO foundation valid: `{mano.get('foundational_mano_state_valid')}`; recovered WiLoR virtual-camera raw candidates: `{wilor_mano.get('complete_virtual_camera_candidate_rows')}`; unique frame-side rows: `{wilor_mano.get('unique_virtual_camera_frame_side_rows')}`; metric-world aligned: `{wilor_mano.get('metric_world_alignment_valid')}`; HaWoR world rows: `{hawor_mano.get('complete_world_surface_param_rows')}`; blockers: `{mano.get('blocking_reasons')}`",
            f"HaWoR hard requirement state: status `{hawor_hard.get('status')}`; available HaWoR frame-side rows `{hawor_hard.get('available_hawor_frame_side_rows')}/{hawor_hard.get('expected_frame_side_rows')}`; requirement met `{hawor_hard.get('accepted_v18_hawor_requirement_met')}`; blockers `{hawor_hard.get('blocking_reasons')}`",
            f"HaWoR bridge state: status `{hawor_bridge.get('status')}`; candidate rows `{hawor_bridge.get('bridge_candidate_rows')}/{hawor_bridge.get('expected_frame_side_rows')}`; accepted foundation `{hawor_bridge.get('accepted_v18_hawor_foundation')}`; median residual summary `{hawor_bridge.get('reference_projection_residual_px_median_per_row')}`; blockers `{hawor_bridge.get('blocking_reasons')}`",
            f"HaWoR bridge NPZ: `{hawor_bridge.get('bridge_candidate_npz')}`",
            f"HaWoR bridge residual review sheet: `{hawor_bridge.get('review_sheet', {}).get('path')}` exists=`{hawor_bridge.get('review_sheet', {}).get('exists')}`",
            f"HaWoR bridge quality state: status `{hawor_quality.get('status')}`; quality counts `{hawor_quality.get('quality_counts')}`; accepted foundation `{hawor_quality.get('accepted_v18_hawor_foundation')}`; physical hand valid `{hawor_quality.get('v18_physical_hand_state_valid_from_quality')}`; supported residual summary `{hawor_quality.get('supported_candidate_projection_residual_px_median_per_row')}`",
            f"HaWoR bridge quality overlay: `{hawor_quality.get('quality_overlay_video', {}).get('path')}` exists=`{hawor_quality.get('quality_overlay_video', {}).get('exists')}`",
            f"HaWoR bridge downstream coverage: contact rows with projection-supported bridge `{hawor_coverage.get('existing_contact_rows_with_projection_supported_hawor_bridge')}/{hawor_coverage.get('existing_contact_acceptance_audit_rows')}`; occlusion rows `{hawor_coverage.get('existing_occlusion_rows_with_projection_supported_hawor_bridge')}/{hawor_coverage.get('existing_occlusion_acceptance_audit_rows')}`; accepted contact/occlusion input flags `{hawor_coverage.get('accepted_contact_or_occlusion_input_flags')}`",
            f"HaWoR bridge subset policy: strict candidate queue policy counts `{hawor_subset.get('policy_counts')}`; contact strict queue `{hawor_subset.get('existing_contact_acceptance_audit_rows_in_strict_candidate_queue')}/{hawor_subset.get('existing_contact_acceptance_audit_rows_total')}`; occlusion strict queue `{hawor_subset.get('existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue')}/{hawor_subset.get('existing_occlusion_acceptance_audit_rows_total')}`; downstream accepted from policy `{hawor_subset.get('downstream_acceptance_from_policy')}`",
            f"MANO NPZ: `{case.get('mano_foundation_artifacts', {}).get('wilor_virtual_camera_npz', {}).get('path')}`",
            f"MANO overlay available frame-side rows: `{case['mechanisms']['mano_foundation_overlay'].get('available_frame_side_rows')}`",
            f"Tentative owner rows: `{owner.get('selected_tentative_owner_rows')}`; strict accepted: `{owner.get('strict_accepted_owner_rows')}`",
            f"Occlusion acceptance audit rows: `{owner_audit.get('candidate_rows')}`; strict promotable: `{owner_audit.get('strict_promotable_owner_rows')}`; categories: `{owner_audit.get('category_counts')}`",
            f"Contact rows selected: `{contact.get('contact_graph_selected_rows')}`; source graph contact candidates before local veto: `{contact.get('source_graph_contact_candidate_rows_before_nonpenetration_veto')}`; signed/triangle penetration rows: `{contact.get('signed_local_penetration_rows')}` / `{contact.get('triangle_local_penetration_rows')}`",
            f"Contact acceptance audit rows: `{contact_audit.get('selected_contact_rows')}`; strict promotable: `{contact_audit.get('strict_promotable_contact_rows')}`; categories: `{contact_audit.get('category_counts')}`",
            f"Repair proposal rows: `{case['mechanisms']['nonpenetration_repair_proposal'].get('proposal_rows')}`; statuses: `{case['mechanisms']['nonpenetration_repair_proposal'].get('proposal_status_counts')}`",
            f"Temporal smoothed MANO2D draw counts: `{hand_smoothing.get('draw_counts')}`",
            f"Geometry coverage audit statuses: `{case['mechanisms']['geometry_coverage_audit'].get('status_counts')}`",
            f"HaWoR measurement rows: `{hawor.get('measurement_rows')}`",
            "",
            "Videos:",
        ]
        for video in case["videos"]:
            lines.append(f"- `{video['path']}` — {video['mechanism']} / {video['role']} / frames {video.get('frame_count')}/{video.get('expected_frame_count')}")
        lines += ["", "Review sheets:"]
        for sheet in case["review_sheets"]:
            lines.append(f"- `{sheet['path']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_bundle(case, args.output_root) for case in args.cases]
    hawor_audit_path = args.output_root / "hawor_provisioning_audit" / "v18_hawor_provisioning_audit_report.json"
    hawor_audit = load_json(hawor_audit_path) if hawor_audit_path.exists() else None
    mano_summary_path = args.output_root / "mano_foundation_audit" / "v18_mano_foundation_audit_summary.json"
    mano_summary = load_json(mano_summary_path) if mano_summary_path.exists() else None
    hawor_requirement_path = args.output_root / "hawor_requirement_state" / "v18_hawor_requirement_state.json"
    hawor_requirement = load_json(hawor_requirement_path) if hawor_requirement_path.exists() else None
    hawor_task5_contract_path = args.output_root / "hawor_task5_export_contract" / "v18_hawor_task5_export_contract.json"
    hawor_task5_contract = load_json(hawor_task5_contract_path) if hawor_task5_contract_path.exists() else None
    hawor_bridge_summary_path = args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_state_summary.json"
    hawor_bridge_summary = load_json(hawor_bridge_summary_path) if hawor_bridge_summary_path.exists() else None
    hawor_bridge_quality_summary_path = args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_quality_state_summary.json"
    hawor_bridge_quality_summary = load_json(hawor_bridge_quality_summary_path) if hawor_bridge_quality_summary_path.exists() else None
    hawor_bridge_downstream_coverage_path = args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_downstream_coverage_summary.json"
    hawor_bridge_downstream_coverage = load_json(hawor_bridge_downstream_coverage_path) if hawor_bridge_downstream_coverage_path.exists() else None
    hawor_bridge_subset_policy_path = args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_subset_policy_summary.json"
    hawor_bridge_subset_policy = load_json(hawor_bridge_subset_policy_path) if hawor_bridge_subset_policy_path.exists() else None
    hawor_strict_contact_probe_path = args.output_root / "hawor_bridge_state" / "v18_hawor_strict_contact_probe_summary.json"
    hawor_strict_contact_probe = load_json(hawor_strict_contact_probe_path) if hawor_strict_contact_probe_path.exists() else None
    hawor_temporal_offset_probe_path = args.output_root / "hawor_bridge_state" / "v18_hawor_temporal_offset_probe_summary.json"
    hawor_temporal_offset_probe = load_json(hawor_temporal_offset_probe_path) if hawor_temporal_offset_probe_path.exists() else None
    post_bridge_validation_path = args.output_root / "v18_post_bridge_targeted_validation_report.json"
    post_bridge_validation = load_json(post_bridge_validation_path) if post_bridge_validation_path.exists() else None
    manifest = {
        "method": "build_v18_corrective_bundle_manifest",
        "status": "corrective_bundle_index_not_full_v18_closure",
        "output_root": str(args.output_root),
        "global_artifacts": {
            "hawor_provisioning_audit_report": file_info(hawor_audit_path),
            "hawor_provisioning_audit_markdown": file_info(args.output_root / "hawor_provisioning_audit" / "V18_HAWOR_PROVISIONING_AUDIT.md"),
            "mano_foundation_summary": file_info(mano_summary_path),
            "mano_foundation_markdown": file_info(args.output_root / "mano_foundation_audit" / "V18_MANO_FOUNDATION_AUDIT.md"),
            "hawor_requirement_state": file_info(hawor_requirement_path),
            "hawor_requirement_markdown": file_info(args.output_root / "hawor_requirement_state" / "V18_HAWOR_REQUIREMENT_STATE.md"),
            "hawor_task5_export_contract": file_info(hawor_task5_contract_path),
            "hawor_task5_export_contract_markdown": file_info(args.output_root / "hawor_task5_export_contract" / "V18_HAWOR_TASK5_EXPORT_CONTRACT.md"),
            "hawor_bridge_summary": file_info(hawor_bridge_summary_path),
            "hawor_bridge_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_BRIDGE_STATE.md"),
            "hawor_bridge_quality_summary": file_info(hawor_bridge_quality_summary_path),
            "hawor_bridge_quality_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_BRIDGE_QUALITY_STATE.md"),
            "hawor_bridge_downstream_coverage_summary": file_info(hawor_bridge_downstream_coverage_path),
            "hawor_bridge_downstream_coverage_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_BRIDGE_DOWNSTREAM_COVERAGE.md"),
            "hawor_bridge_subset_policy_summary": file_info(hawor_bridge_subset_policy_path),
            "hawor_bridge_subset_policy_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_BRIDGE_SUBSET_POLICY.md"),
            "hawor_strict_contact_probe_summary": file_info(hawor_strict_contact_probe_path),
            "hawor_strict_contact_probe_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_STRICT_CONTACT_PROBE.md"),
            "hawor_temporal_offset_probe_summary": file_info(hawor_temporal_offset_probe_path),
            "hawor_temporal_offset_probe_markdown": file_info(args.output_root / "hawor_bridge_state" / "V18_HAWOR_TEMPORAL_OFFSET_PROBE.md"),
            "post_bridge_targeted_validation_report": file_info(post_bridge_validation_path),
            "pipeline_report_scope_note": file_info(args.output_root / "V18_PIPELINE_REPORT_SCOPE_NOTE.md"),
        },
        "hawor_provisioning_audit": {
            "status": hawor_audit.get("status") if isinstance(hawor_audit, dict) else None,
            "missing_required": hawor_audit.get("missing_required") if isinstance(hawor_audit, dict) else None,
            "claim_scope": hawor_audit.get("claim_scope") if isinstance(hawor_audit, dict) else None,
        },
        "mano_foundation_audit": {
            "all_cases_foundational_mano_valid": mano_summary.get("all_cases_foundational_mano_valid") if isinstance(mano_summary, dict) else None,
            "v18_physical_pipeline_valid_without_further_hand_work": mano_summary.get("v18_physical_pipeline_valid_without_further_hand_work") if isinstance(mano_summary, dict) else None,
            "claim_scope": mano_summary.get("claim_scope") if isinstance(mano_summary, dict) else None,
        },
        "hawor_hard_requirement_state": {
            "status": hawor_requirement.get("status") if isinstance(hawor_requirement, dict) else None,
            "all_cases_hawor_requirement_met": hawor_requirement.get("all_cases_hawor_requirement_met") if isinstance(hawor_requirement, dict) else None,
            "v18_physical_hand_state_valid_from_hawor": hawor_requirement.get("v18_physical_hand_state_valid_from_hawor") if isinstance(hawor_requirement, dict) else None,
            "blocking_reasons": hawor_requirement.get("blocking_reasons") if isinstance(hawor_requirement, dict) else None,
            "claim_scope": hawor_requirement.get("claim_scope") if isinstance(hawor_requirement, dict) else None,
        },
        "hawor_task5_export_contract": {
            "status": hawor_task5_contract.get("status") if isinstance(hawor_task5_contract, dict) else None,
            "expected_local_output_npz": hawor_task5_contract.get("expected_local_output_npz") if isinstance(hawor_task5_contract, dict) else None,
            "remote_export_command": hawor_task5_contract.get("remote_export_command") if isinstance(hawor_task5_contract, dict) else None,
            "acceptance_flags": hawor_task5_contract.get("acceptance_flags") if isinstance(hawor_task5_contract, dict) else None,
            "blocking_reasons": hawor_task5_contract.get("blocking_reasons") if isinstance(hawor_task5_contract, dict) else None,
            "claim_scope": hawor_task5_contract.get("claim_scope") if isinstance(hawor_task5_contract, dict) else None,
        },
        "hawor_bridge_state": {
            "status": hawor_bridge_summary.get("status") if isinstance(hawor_bridge_summary, dict) else None,
            "all_cases_hawor_bridge_accepted": hawor_bridge_summary.get("all_cases_hawor_bridge_accepted") if isinstance(hawor_bridge_summary, dict) else None,
            "v18_physical_hand_state_valid_from_bridge": hawor_bridge_summary.get("v18_physical_hand_state_valid_from_bridge") if isinstance(hawor_bridge_summary, dict) else None,
            "blocking_reasons": hawor_bridge_summary.get("blocking_reasons") if isinstance(hawor_bridge_summary, dict) else None,
            "claim_scope": hawor_bridge_summary.get("claim_scope") if isinstance(hawor_bridge_summary, dict) else None,
        },
        "hawor_bridge_quality_state": {
            "status": hawor_bridge_quality_summary.get("status") if isinstance(hawor_bridge_quality_summary, dict) else None,
            "all_cases_quality_foundation_accepted": hawor_bridge_quality_summary.get("all_cases_quality_foundation_accepted") if isinstance(hawor_bridge_quality_summary, dict) else None,
            "v18_physical_hand_state_valid_from_quality": hawor_bridge_quality_summary.get("v18_physical_hand_state_valid_from_quality") if isinstance(hawor_bridge_quality_summary, dict) else None,
            "blocking_reasons": hawor_bridge_quality_summary.get("blocking_reasons") if isinstance(hawor_bridge_quality_summary, dict) else None,
            "claim_scope": hawor_bridge_quality_summary.get("claim_scope") if isinstance(hawor_bridge_quality_summary, dict) else None,
        },
        "hawor_bridge_downstream_coverage": {
            "status": hawor_bridge_downstream_coverage.get("status") if isinstance(hawor_bridge_downstream_coverage, dict) else None,
            "all_cases_downstream_accepted": hawor_bridge_downstream_coverage.get("all_cases_downstream_accepted") if isinstance(hawor_bridge_downstream_coverage, dict) else None,
            "claim_scope": hawor_bridge_downstream_coverage.get("claim_scope") if isinstance(hawor_bridge_downstream_coverage, dict) else None,
        },
        "hawor_bridge_subset_policy": {
            "status": hawor_bridge_subset_policy.get("status") if isinstance(hawor_bridge_subset_policy, dict) else None,
            "all_cases_policy_foundation_accepted": hawor_bridge_subset_policy.get("all_cases_policy_foundation_accepted") if isinstance(hawor_bridge_subset_policy, dict) else None,
            "all_cases_metric_hand_state_accepted_from_policy": hawor_bridge_subset_policy.get("all_cases_metric_hand_state_accepted_from_policy") if isinstance(hawor_bridge_subset_policy, dict) else None,
            "all_cases_downstream_accepted_from_policy": hawor_bridge_subset_policy.get("all_cases_downstream_accepted_from_policy") if isinstance(hawor_bridge_subset_policy, dict) else None,
            "claim_scope": hawor_bridge_subset_policy.get("claim_scope") if isinstance(hawor_bridge_subset_policy, dict) else None,
        },
        "hawor_strict_contact_probe": {
            "status": hawor_strict_contact_probe.get("status") if isinstance(hawor_strict_contact_probe, dict) else None,
            "contact_acceptance_from_probe": hawor_strict_contact_probe.get("contact_acceptance_from_probe") if isinstance(hawor_strict_contact_probe, dict) else None,
            "nonpenetration_acceptance_from_probe": hawor_strict_contact_probe.get("nonpenetration_acceptance_from_probe") if isinstance(hawor_strict_contact_probe, dict) else None,
            "claim_scope": hawor_strict_contact_probe.get("claim_scope") if isinstance(hawor_strict_contact_probe, dict) else None,
            "distance_summary": (hawor_strict_contact_probe.get("cases") or [{}])[0].get("hawor_hand_to_visible_object_surface_min_m") if isinstance(hawor_strict_contact_probe, dict) and isinstance(hawor_strict_contact_probe.get("cases"), list) and hawor_strict_contact_probe.get("cases") else None,
        },
        "hawor_temporal_offset_probe": {
            "status": hawor_temporal_offset_probe.get("status") if isinstance(hawor_temporal_offset_probe, dict) else None,
            "contact_acceptance_from_probe": hawor_temporal_offset_probe.get("contact_acceptance_from_probe") if isinstance(hawor_temporal_offset_probe, dict) else None,
            "nonpenetration_acceptance_from_probe": hawor_temporal_offset_probe.get("nonpenetration_acceptance_from_probe") if isinstance(hawor_temporal_offset_probe, dict) else None,
            "foundation_acceptance_from_probe": hawor_temporal_offset_probe.get("foundation_acceptance_from_probe") if isinstance(hawor_temporal_offset_probe, dict) else None,
            "claim_scope": hawor_temporal_offset_probe.get("claim_scope") if isinstance(hawor_temporal_offset_probe, dict) else None,
            "interpretation": (hawor_temporal_offset_probe.get("cases") or [{}])[0].get("interpretation") if isinstance(hawor_temporal_offset_probe, dict) and isinstance(hawor_temporal_offset_probe.get("cases"), list) and hawor_temporal_offset_probe.get("cases") else None,
            "dominant_best_distance_offset": (hawor_temporal_offset_probe.get("cases") or [{}])[0].get("dominant_best_distance_offset") if isinstance(hawor_temporal_offset_probe, dict) and isinstance(hawor_temporal_offset_probe.get("cases"), list) and hawor_temporal_offset_probe.get("cases") else None,
            "dominant_best_distance_fraction": (hawor_temporal_offset_probe.get("cases") or [{}])[0].get("dominant_best_distance_fraction") if isinstance(hawor_temporal_offset_probe, dict) and isinstance(hawor_temporal_offset_probe.get("cases"), list) and hawor_temporal_offset_probe.get("cases") else None,
            "best_any_offset_distance_summary": (hawor_temporal_offset_probe.get("cases") or [{}])[0].get("best_any_offset_distance_m") if isinstance(hawor_temporal_offset_probe, dict) and isinstance(hawor_temporal_offset_probe.get("cases"), list) and hawor_temporal_offset_probe.get("cases") else None,
            "best_any_offset_abs_depth_gap_summary": (hawor_temporal_offset_probe.get("cases") or [{}])[0].get("best_any_offset_abs_depth_gap_m") if isinstance(hawor_temporal_offset_probe, dict) and isinstance(hawor_temporal_offset_probe.get("cases"), list) and hawor_temporal_offset_probe.get("cases") else None,
        },
        "post_bridge_targeted_validation": {
            "status": post_bridge_validation.get("status") if isinstance(post_bridge_validation, dict) else None,
            "claim_scope": post_bridge_validation.get("claim_scope") if isinstance(post_bridge_validation, dict) else None,
            "long_pipeline_rerun_after_bridge_changes": post_bridge_validation.get("long_pipeline_rerun_after_bridge_changes") if isinstance(post_bridge_validation, dict) else None,
            "pre_bridge_pipeline_report": post_bridge_validation.get("pre_bridge_pipeline_report") if isinstance(post_bridge_validation, dict) else None,
            "active_partial_pipeline_report": post_bridge_validation.get("active_partial_pipeline_report") if isinstance(post_bridge_validation, dict) else None,
            "stale_partial_pipeline_report_archives": post_bridge_validation.get("stale_partial_pipeline_report_archives") if isinstance(post_bridge_validation, dict) else None,
        },
        "cases": cases,
        "all_listed_video_frame_counts_match": all(case["all_listed_video_frame_counts_match"] for case in cases),
        "claim_scope": "indexes actual changed corrective V18 artifacts and failure evidence; not a readiness ledger or version-closure claim",
    }
    write_json(args.output_root / "v18_corrective_bundle_manifest.json", manifest)
    write_markdown(args.output_root / "V18_CORRECTIVE_BUNDLE.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
