#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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

STATUS = "v18_status_deliverable_manifest"
CLAIM = (
    "This manifest closes a V18 status deliverable: full-duration 2D overlay, abstract world/status, "
    "side-by-side status videos, HaWoR/WiLoR/RTMLib hand-baseline evidence, bounded state evidence, "
    "occlusion owner-candidate evidence, occlusion depth-order triage evidence, visible-surface geometry evidence, "
    "structured physical-state schema evidence, an object completion eligibility gate, generated-only part-track source "
    "manifest, part-split mask evidence audit, part visible-surface evidence, part-motion state evidence, part-motion "
    "confound QC, explicit part-object blocker records, promptable SAM proposal evidence, promptable proposal "
    "promotion-gate evidence, part-mask acquisition status, measured cached-evidence-to-status runtime evidence, "
    "and invariant audit evidence. It does not close full-video HaWoR readiness, occluded hand pose, final hidden "
    "object geometry, object pose, part pose, articulation model, physical contact requirements, occluder ownership, "
    "depth ordering, or fresh raw-video-to-final runtime."
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


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{label} must be numeric")
    return float(value)




def parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0.0:
                return None
            return float(num) / denominator
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def ffprobe_video_info(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    payload = require_dict(json.loads(proc.stdout), f"ffprobe {path}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"ffprobe found no video stream for {path}")
    stream = require_dict(streams[0], f"ffprobe stream {path}")
    duration_raw = stream.get("duration")
    duration = float(duration_raw) if duration_raw is not None else None
    fps = parse_rate(stream.get("avg_frame_rate"))
    return {"duration_s": duration, "avg_frame_rate": fps}


def read_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    solution_path = args.solution_root / case / "v18_bounded_state_solution.json"
    overlay_qc_path = args.render_root / case / "v18_status_overlay_qc.json"
    world_qc_path = args.render_root / case / "v18_world_status_qc.json"
    occlusion_candidates_path = args.occlusion_owner_candidates_root / case / "v18_occlusion_owner_candidates_report.json"
    occlusion_depth_evidence_path = args.occlusion_depth_evidence_root / case / "v18_occlusion_depth_order_evidence_report.json"
    side_qc_path = args.render_root / case / "v18_status_side_by_side_qc.json"
    visible_geometry_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    physical_schema_path = args.physical_state_schema_root / case / "v18_physical_state_schema_report.json"
    hand_baseline_path = args.hand_baseline_root / case / "v18_hand_baseline_branch_report.json"
    completion_gate_path = args.completion_gate_root / case / "v18_object_completion_gate_report.json"
    part_split_path = args.part_split_root / case / "v18_part_split_evidence_report.json"
    part_track_source_path = args.part_track_source_root / case / "v18_part_track_source_manifest_report.json"
    part_surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    part_motion_path = args.part_motion_root / case / "v18_part_motion_state_report.json"
    part_motion_qc_path = args.part_motion_qc_root / case / "v18_part_motion_qc_report.json"
    part_model_candidates_path = args.part_model_candidates_root / case / "v18_part_model_candidates_report.json"
    articulation_fit_path = args.articulation_fit_root / case / "v18_articulation_fit_candidates_report.json"
    part_se3_path = args.part_se3_root / case / "v18_part_se3_surface_residuals_report.json"
    visible_part_subset_path = args.visible_part_subset_root / case / "v18_visible_part_subset_archive_report.json"
    part_object_blockers_path = args.part_object_blockers_root / case / "v18_part_object_blocker_manifest_report.json"
    sam_promptable_proposals_path = args.sam_promptable_proposals_root / case / "v18_sam_promptable_part_proposals_report.json"
    part_mask_promotion_gate_path = args.part_mask_promotion_gate_root / case / "v18_part_mask_promotion_gate_report.json"
    part_mask_acquisition_path = args.part_mask_acquisition_root / case / "v18_part_mask_acquisition_plan_report.json"
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    solution = require_dict(load_json(solution_path), f"{case} solution")
    overlay = require_dict(load_json(overlay_qc_path), f"{case} overlay qc")
    world = require_dict(load_json(world_qc_path), f"{case} world qc")
    occlusion_candidates = require_dict(load_json(occlusion_candidates_path), f"{case} occlusion owner candidates")
    occlusion_depth_evidence = require_dict(load_json(occlusion_depth_evidence_path), f"{case} occlusion depth-order evidence")
    side = require_dict(load_json(side_qc_path), f"{case} side qc")
    visible_geometry = require_dict(load_json(visible_geometry_path), f"{case} visible geometry")
    physical_schema = require_dict(load_json(physical_schema_path), f"{case} physical state schema")
    hand_baseline = require_dict(load_json(hand_baseline_path), f"{case} hand baseline branch")
    completion_gate = require_dict(load_json(completion_gate_path), f"{case} completion gate")
    part_split = require_dict(load_json(part_split_path), f"{case} part split evidence")
    part_track_source = require_dict(load_json(part_track_source_path), f"{case} part-track source manifest")
    part_surfaces = require_dict(load_json(part_surfaces_path), f"{case} part visible surfaces")
    part_motion = require_dict(load_json(part_motion_path), f"{case} part motion state")
    part_motion_qc = require_dict(load_json(part_motion_qc_path), f"{case} part motion qc")
    part_model_candidates = require_dict(load_json(part_model_candidates_path), f"{case} part model candidates")
    articulation_fit = require_dict(load_json(articulation_fit_path), f"{case} articulation fit candidates")
    part_se3 = require_dict(load_json(part_se3_path), f"{case} part se3 surface residuals")
    visible_part_subset = require_dict(load_json(visible_part_subset_path), f"{case} visible part subset archive")
    part_object_blockers = require_dict(load_json(part_object_blockers_path), f"{case} part object blockers")
    sam_promptable_proposals = require_dict(load_json(sam_promptable_proposals_path), f"{case} SAM promptable proposals")
    part_mask_promotion_gate = require_dict(load_json(part_mask_promotion_gate_path), f"{case} part mask promotion gate")
    part_mask_acquisition = require_dict(load_json(part_mask_acquisition_path), f"{case} part mask acquisition plan")
    frame_count = require_int(annotation.get("frame_count"), "annotation frame_count")
    raw_frame_count = require_int(annotation.get("raw_frame_count"), "annotation raw_frame_count")
    raw_video = require_dict(annotation.get("raw_video"), "annotation raw_video")
    duration_s = require_float(annotation.get("duration_s"), "duration_s")
    qcs = {
        "annotation_state": bool(annotation.get("frame_count_match")),
        "bounded_state_solution": bool(solution.get("frame_count_match")),
        "status_overlay": bool(overlay.get("frame_count_match")),
        "world_status": bool(world.get("frame_count_match")),
        "side_by_side": bool(side.get("frame_count_match")),
    }
    if frame_count != raw_frame_count:
        raise RuntimeError(f"{case}: frame_count != raw_frame_count")
    for label, ok in qcs.items():
        if not ok:
            raise RuntimeError(f"{case}: {label} frame-count QC is false")
    overlay_frames = require_int(overlay.get("video_frame_count"), "overlay video_frame_count")
    world_frames = require_int(world.get("video_frame_count"), "world video_frame_count")
    side_frames = require_int(side.get("side_by_side_frame_count"), "side side_by_side_frame_count")
    if not (overlay_frames == world_frames == side_frames == frame_count):
        raise RuntimeError(f"{case}: output video frame counts do not all equal {frame_count}")
    overlay_path = Path(str(overlay.get("output_video")))
    world_path = Path(str(world.get("output_video")))
    side_path = Path(str(side.get("output_video")))
    for label, video_path in (("overlay", overlay_path), ("world", world_path), ("side_by_side", side_path)):
        if not video_path.exists():
            raise RuntimeError(f"{case}: missing {label} video at {video_path}")
    expected_fps = require_float(raw_video.get("fps"), "raw fps")
    expected_duration_s = frame_count / expected_fps
    duration_tolerance_s = max(0.05, 0.5 / expected_fps)
    fps_tolerance = 1e-3
    overlay_info = ffprobe_video_info(overlay_path)
    world_info = ffprobe_video_info(world_path)
    side_info = ffprobe_video_info(side_path)

    def duration_ok(info: dict[str, Any]) -> bool:
        duration = info.get("duration_s")
        return isinstance(duration, float) and abs(duration - expected_duration_s) <= duration_tolerance_s

    def fps_ok(info: dict[str, Any]) -> bool:
        fps = info.get("avg_frame_rate")
        return isinstance(fps, float) and abs(fps - expected_fps) <= fps_tolerance

    duration_qc = {
        "expected_fps": expected_fps,
        "expected_duration_s": expected_duration_s,
        "duration_tolerance_s": duration_tolerance_s,
        "overlay": overlay_info,
        "world_status": world_info,
        "side_by_side": side_info,
        "all_durations_match_raw": duration_ok(overlay_info) and duration_ok(world_info) and duration_ok(side_info),
        "all_fps_match_raw": fps_ok(overlay_info) and fps_ok(world_info) and fps_ok(side_info),
    }
    if not duration_qc["all_durations_match_raw"] or not duration_qc["all_fps_match_raw"]:
        raise RuntimeError(f"{case}: status video duration/FPS QC failed: {duration_qc}")
    render_elapsed_s = require_float(overlay.get("elapsed_s"), "overlay elapsed") + require_float(world.get("elapsed_s"), "world elapsed") + require_float(side.get("elapsed_s"), "side elapsed")
    return {
        "case": case,
        "raw_video": raw_video,
        "duration_s": duration_s,
        "frame_count": frame_count,
        "raw_frame_count": raw_frame_count,
        "frame_count_match": True,
        "status_outputs": {
            "annotation_state": str(annotation_path),
            "bounded_state_solution": str(solution_path),
            "status_overlay_video": overlay.get("output_video"),
            "world_status_video": world.get("output_video"),
            "side_by_side_status_video": side.get("output_video"),
            "status_overlay_qc": str(overlay_qc_path),
            "world_status_qc": str(world_qc_path),
            "occlusion_owner_candidates_report": str(occlusion_candidates_path),
            "occlusion_depth_order_evidence_report": str(occlusion_depth_evidence_path),
            "side_by_side_qc": str(side_qc_path),
            "visible_geometry_archive_report": str(visible_geometry_path),
            "visible_geometry_archive_npz": visible_geometry.get("archive_npz"),
            "physical_state_schema_report": str(physical_schema_path),
            "hand_baseline_branch_report": str(hand_baseline_path),
            "object_completion_gate_report": str(completion_gate_path),
            "part_split_evidence_report": str(part_split_path),
            "part_track_source_manifest_report": str(part_track_source_path),
            "part_visible_surfaces_report": str(part_surfaces_path),
            "part_visible_surfaces_archive_npz": part_surfaces.get("archive_npz"),
            "part_motion_state_report": str(part_motion_path),
            "part_motion_qc_report": str(part_motion_qc_path),
            "part_model_candidates_report": str(part_model_candidates_path),
            "articulation_fit_candidates_report": str(articulation_fit_path),
            "part_se3_surface_residuals_report": str(part_se3_path),
            "visible_part_subset_archive_report": str(visible_part_subset_path),
            "visible_part_subset_archive_npz": visible_part_subset.get("archive_npz"),
            "part_object_blocker_manifest": str(part_object_blockers_path),
            "sam_promptable_part_proposals": str(sam_promptable_proposals_path),
            "part_mask_promotion_gate": str(part_mask_promotion_gate_path),
            "part_mask_acquisition_plan": str(part_mask_acquisition_path),
        },
        "frame_count_qc": {
            "annotation_state_frames": frame_count,
            "bounded_solution_frames": require_int(solution.get("frame_count"), "solution frame_count"),
            "overlay_video_frames": overlay_frames,
            "world_status_video_frames": world_frames,
            "side_by_side_video_frames": side_frames,
            "all_match_raw": True,
        },
        "status_runtime_qc": {
            "measured_render_elapsed_s": render_elapsed_s,
            "duration_s": duration_s,
            "measured_render_to_video_ratio": render_elapsed_s / duration_s if duration_s > 0 else None,
            "under_10x_realtime_for_status_render": render_elapsed_s <= 10.0 * duration_s,
        },
        "duration_qc": duration_qc,
        "occlusion_owner_candidate_qc": {
            "unresolved_hand_row_count": occlusion_candidates.get("unresolved_hand_row_count"),
            "candidate_state_counts": occlusion_candidates.get("candidate_state_counts"),
            "candidate_owner_row_count": occlusion_candidates.get("candidate_owner_row_count"),
            "occluder_owner_accepted_count": occlusion_candidates.get("occluder_owner_accepted_count"),
            "depth_order_resolved_count": occlusion_candidates.get("depth_order_resolved_count"),
            "pose_filled_through_occlusion_rows": occlusion_candidates.get("pose_filled_through_occlusion_rows"),
        },
        "occlusion_depth_order_evidence_qc": {
            "candidate_owner_row_count": occlusion_depth_evidence.get("candidate_owner_row_count"),
            "candidate_pair_count": occlusion_depth_evidence.get("candidate_pair_count"),
            "same_frame_object_surface_pair_count": occlusion_depth_evidence.get("same_frame_object_surface_pair_count"),
            "candidate_pair_depth_evidence_state_counts": occlusion_depth_evidence.get("candidate_pair_depth_evidence_state_counts"),
            "foreground_occluder_support_pair_count": occlusion_depth_evidence.get("foreground_occluder_support_pair_count"),
            "foreground_occluder_contradiction_pair_count": occlusion_depth_evidence.get("foreground_occluder_contradiction_pair_count"),
            "metric_compatible_no_foreground_signal_pair_count": occlusion_depth_evidence.get("metric_compatible_no_foreground_signal_pair_count"),
            "insufficient_object_surface_depth_pair_count": occlusion_depth_evidence.get("insufficient_object_surface_depth_pair_count"),
            "insufficient_or_untrusted_hand_depth_pair_count": occlusion_depth_evidence.get("insufficient_or_untrusted_hand_depth_pair_count"),
            "occluder_owner_accepted_count": occlusion_depth_evidence.get("occluder_owner_accepted_count"),
            "depth_order_resolved_count": occlusion_depth_evidence.get("depth_order_resolved_count"),
            "pose_filled_through_occlusion_rows": occlusion_depth_evidence.get("pose_filled_through_occlusion_rows"),
        },
        "bounded_state_qc": {
            "hand_solution_state_counts": solution.get("hand_solution_state_counts"),
            "object_solution_state_counts": solution.get("object_solution_state_counts"),
            "contact_solution_state_counts": solution.get("contact_solution_state_counts"),
            "occlusion_solution_counts": solution.get("occlusion_solution_counts"),
            "contact_factor_ready_rows": solution.get("contact_factor_ready_rows"),
            "occlusion_owner_candidate_rows": solution.get("occlusion_owner_candidate_rows"),
            "occluder_owner_accepted_rows": solution.get("occluder_owner_accepted_rows"),
            "occlusion_depth_order_resolved_rows": solution.get("occlusion_depth_order_resolved_rows"),
            "occlusion_depth_evidence_candidate_pair_rows": solution.get("occlusion_depth_evidence_candidate_pair_rows"),
            "occlusion_depth_evidence_support_pair_rows": solution.get("occlusion_depth_evidence_support_pair_rows"),
            "occlusion_depth_evidence_contradiction_pair_rows": solution.get("occlusion_depth_evidence_contradiction_pair_rows"),
            "pose_filled_through_occlusion_rows": solution.get("pose_filled_through_occlusion_rows"),
        },
        "visible_geometry_qc": {
            "visible_geometry_archive_ready": visible_geometry.get("visible_geometry_archive_ready"),
            "surface_frame_rows": visible_geometry.get("surface_frame_rows"),
            "rejected_visible_object_frame_rows": visible_geometry.get("rejected_visible_object_frame_rows"),
            "total_vertices": visible_geometry.get("total_vertices"),
            "total_faces": visible_geometry.get("total_faces"),
            "v18_visible_geometry_status_counts": visible_geometry.get("v18_visible_geometry_status_counts"),
            "hidden_geometry_reconstructed": visible_geometry.get("hidden_geometry_reconstructed"),
            "canonical_mesh_ready": visible_geometry.get("canonical_mesh_ready"),
            "complete_object_pose_ready": visible_geometry.get("complete_object_pose_ready"),
        },
        "physical_state_schema_qc": {
            "object_count": physical_schema.get("object_count"),
            "model_physical_state_type_counts": physical_schema.get("model_physical_state_type_counts"),
            "legacy_keyword_physical_state_type_counts": physical_schema.get("legacy_keyword_physical_state_type_counts"),
            "part_or_relative_motion_required_count": physical_schema.get("part_or_relative_motion_required_count"),
            "secondary_deformable_or_surface_component_count": physical_schema.get("secondary_deformable_or_surface_component_count"),
            "optical_difficulty_count": physical_schema.get("optical_difficulty_count"),
            "surface_change_without_pose_state_count": physical_schema.get("surface_change_without_pose_state_count"),
            "changed_from_legacy_keyword_count": physical_schema.get("changed_from_legacy_keyword_count"),
            "part_pose_ready_count": physical_schema.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": physical_schema.get("object_pose_requirement_met_count"),
        },
        "hand_baseline_qc": {
            "hand_state_row_count": hand_baseline.get("hand_state_row_count"),
            "hand_baseline_state_counts": hand_baseline.get("hand_baseline_state_counts"),
            "wilor_measurement_row_count": hand_baseline.get("wilor_measurement_row_count"),
            "hawor_measurement_row_count": hand_baseline.get("hawor_measurement_row_count"),
            "hawor_available_measurement_count": hand_baseline.get("hawor_available_measurement_count"),
            "hawor_motion_infill_candidate_count": hand_baseline.get("hawor_motion_infill_candidate_count"),
            "hawor_unique_frame_count": hand_baseline.get("hawor_unique_frame_count"),
            "hawor_full_video_baseline_ready": hand_baseline.get("hawor_full_video_baseline_ready"),
            "hawor_full_video_blockers": hand_baseline.get("hawor_full_video_blockers"),
            "rtmlib_source_status_normalized": hand_baseline.get("rtmlib_source_status_normalized"),
            "rtmlib_manifest_status": hand_baseline.get("rtmlib_manifest_status"),
            "rtmlib_frames_with_hands": hand_baseline.get("rtmlib_frames_with_hands"),
            "rtmlib_wilor_comparison_count": hand_baseline.get("rtmlib_wilor_comparison_count"),
            "temporal_occlusion_pose_accepted_count": hand_baseline.get("temporal_occlusion_pose_accepted_count"),
            "pose_filled_through_occlusion_rows": hand_baseline.get("pose_filled_through_occlusion_rows"),
        },
        "completion_gate_qc": {
            "completion_gate_state_counts": completion_gate.get("completion_gate_state_counts"),
            "completion_action_counts": completion_gate.get("completion_action_counts"),
            "completion_candidate_count": completion_gate.get("completion_candidate_count"),
            "part_split_candidate_count": completion_gate.get("part_split_candidate_count"),
            "completion_run_count": completion_gate.get("completion_run_count"),
            "hidden_geometry_reconstructed_count": completion_gate.get("hidden_geometry_reconstructed_count"),
            "complete_object_pose_ready_count": completion_gate.get("complete_object_pose_ready_count"),
        },
        "part_track_source_qc": {
            "candidate_source_manifest_ready": part_track_source.get("candidate_source_manifest_ready"),
            "part_track_candidate_source_scope": part_track_source.get("part_track_candidate_source_scope"),
            "uniform_part_track_generation_ready": part_track_source.get("uniform_part_track_generation_ready"),
            "root_count": part_track_source.get("root_count"),
            "existing_root_count": part_track_source.get("existing_root_count"),
            "track_count": part_track_source.get("track_count"),
            "usable_track_count": part_track_source.get("usable_track_count"),
            "mask_evidence_created": part_track_source.get("mask_evidence_created"),
        },
        "part_split_evidence_qc": {
            "part_required_object_count": part_split.get("part_required_object_count"),
            "discovered_part_track_count": part_split.get("discovered_part_track_count"),
            "accepted_part_track_assignment_count": part_split.get("accepted_part_track_assignment_count"),
            "part_split_evidence_state_counts": part_split.get("part_split_evidence_state_counts"),
            "part_track_candidate_source_scope": part_split.get("part_track_candidate_source_scope"),
            "candidate_assignment_semantics": part_split.get("candidate_assignment_semantics"),
            "uniform_part_track_generation_ready": part_split.get("uniform_part_track_generation_ready"),
            "part_track_source_manifest_ready": part_split.get("part_track_source_manifest_ready"),
            "part_geometry_extraction_ready_count": part_split.get("part_geometry_extraction_ready_count"),
            "part_pose_ready_count": part_split.get("part_pose_ready_count"),
        },
        "part_visible_surface_qc": {
            "part_visible_surface_archive_ready": part_surfaces.get("part_visible_surface_archive_ready"),
            "surface_frame_rows": part_surfaces.get("surface_frame_rows"),
            "rejected_candidate_rows": part_surfaces.get("rejected_candidate_rows"),
            "total_vertices": part_surfaces.get("total_vertices"),
            "total_faces": part_surfaces.get("total_faces"),
            "surface_rows_by_object": part_surfaces.get("surface_rows_by_object"),
            "surface_rows_by_part_track": part_surfaces.get("surface_rows_by_part_track"),
            "part_geometry_completion_ready": part_surfaces.get("part_geometry_completion_ready"),
            "part_pose_ready": part_surfaces.get("part_pose_ready"),
        },
        "part_motion_qc": {
            "object_count_with_part_surfaces": part_motion.get("object_count_with_part_surfaces"),
            "part_motion_state_counts": part_motion.get("part_motion_state_counts"),
            "pair_motion_state_counts": part_motion.get("pair_motion_state_counts"),
            "part_pose_ready_count": part_motion.get("part_pose_ready_count"),
            "articulation_model_ready_count": part_motion.get("articulation_model_ready_count"),
            "object_pose_requirement_met_count": part_motion.get("object_pose_requirement_met_count"),
        },
        "part_motion_confound_qc": {
            "part_motion_qc_state_counts": part_motion_qc.get("part_motion_qc_state_counts"),
            "pair_qc_state_counts": part_motion_qc.get("pair_qc_state_counts"),
            "part_surface_quality_counts": part_motion_qc.get("part_surface_quality_counts"),
            "articulation_model_ready_count": part_motion_qc.get("articulation_model_ready_count"),
            "part_pose_ready_count": part_motion_qc.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": part_motion_qc.get("object_pose_requirement_met_count"),
        },
        "part_model_candidate_qc": {
            "candidate_count": part_model_candidates.get("candidate_count"),
            "rejected_candidate_count": part_model_candidates.get("rejected_candidate_count"),
            "surface_icp_probe_count": part_model_candidates.get("surface_icp_probe_count"),
            "surface_icp_probe_state_counts": part_model_candidates.get("surface_icp_probe_state_counts"),
            "articulation_hypothesis_pair_count": part_model_candidates.get("articulation_hypothesis_pair_count"),
            "visible_subset_model_candidate_count": part_model_candidates.get("visible_subset_model_candidate_count"),
            "hidden_geometry_completion_candidate_count": part_model_candidates.get("hidden_geometry_completion_candidate_count"),
            "articulation_model_candidate_count": part_model_candidates.get("articulation_model_candidate_count"),
            "articulation_model_ready_count": part_model_candidates.get("articulation_model_ready_count"),
            "part_pose_ready_count": part_model_candidates.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": part_model_candidates.get("object_pose_requirement_met_count"),
            "object_state_counts": part_model_candidates.get("object_state_counts"),
        },
        "articulation_fit_qc": {
            "articulation_fit_probe_count": articulation_fit.get("articulation_fit_probe_count"),
            "articulation_fit_state_counts": articulation_fit.get("articulation_fit_state_counts"),
            "articulation_fit_supported_count": articulation_fit.get("articulation_fit_supported_count"),
            "articulation_fit_rejected_count": articulation_fit.get("articulation_fit_rejected_count"),
            "articulation_fit_underconstrained_count": articulation_fit.get("articulation_fit_underconstrained_count"),
            "radial_residual_outlier_frame_count": articulation_fit.get("radial_residual_outlier_frame_count"),
            "plane_residual_outlier_frame_count": articulation_fit.get("plane_residual_outlier_frame_count"),
            "combined_residual_outlier_frame_count": articulation_fit.get("combined_residual_outlier_frame_count"),
            "articulation_model_ready_count": articulation_fit.get("articulation_model_ready_count"),
            "part_pose_ready_count": articulation_fit.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": articulation_fit.get("object_pose_requirement_met_count"),
        },
        "part_se3_surface_residual_qc": {
            "part_se3_pair_count": part_se3.get("part_se3_pair_count"),
            "part_se3_pair_state_counts": part_se3.get("part_se3_pair_state_counts"),
            "part_surface_se3_state_counts": part_se3.get("part_surface_se3_state_counts"),
            "part_pose_ready_count": part_se3.get("part_pose_ready_count"),
            "contact_ownership_ready_count": part_se3.get("contact_ownership_ready_count"),
            "object_pose_requirement_met_count": part_se3.get("object_pose_requirement_met_count"),
        },
        "visible_part_subset_archive_qc": {
            "visible_part_subset_archive_file_written": visible_part_subset.get("visible_part_subset_archive_file_written"),
            "visible_part_subset_archive_ready": visible_part_subset.get("visible_part_subset_archive_ready"),
            "visible_part_subset_archive_ready_scope": visible_part_subset.get("visible_part_subset_archive_ready_scope"),
            "candidate_count": visible_part_subset.get("candidate_count"),
            "archive_row_count": visible_part_subset.get("archive_row_count"),
            "unique_frame_count": visible_part_subset.get("unique_frame_count"),
            "total_vertices": visible_part_subset.get("total_vertices"),
            "total_faces": visible_part_subset.get("total_faces"),
            "hidden_geometry_completion_candidate_count": visible_part_subset.get("hidden_geometry_completion_candidate_count"),
            "part_pose_ready_count": visible_part_subset.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": visible_part_subset.get("object_pose_requirement_met_count"),
        },
        "part_object_blocker_qc": {
            "required_part_object_count": part_object_blockers.get("required_part_object_count"),
            "part_object_blocker_state_counts": part_object_blockers.get("part_object_blocker_state_counts"),
            "rejected_part_model_candidate_count": part_object_blockers.get("rejected_part_model_candidate_count"),
            "surface_icp_probe_count": part_object_blockers.get("surface_icp_probe_count"),
            "surface_icp_probe_state_counts": part_object_blockers.get("surface_icp_probe_state_counts"),
            "articulation_hypothesis_pair_count": part_object_blockers.get("articulation_hypothesis_pair_count"),
            "hidden_geometry_reconstructed_count": part_object_blockers.get("hidden_geometry_reconstructed_count"),
            "articulation_model_ready_count": part_object_blockers.get("articulation_model_ready_count"),
            "part_pose_ready_count": part_object_blockers.get("part_pose_ready_count"),
            "contact_ownership_ready_count": part_object_blockers.get("contact_ownership_ready_count"),
            "object_pose_requirement_met_count": part_object_blockers.get("object_pose_requirement_met_count"),
        },
        "sam_promptable_proposal_qc": {
            "object_count": sam_promptable_proposals.get("object_count"),
            "selected_frame_count": sam_promptable_proposals.get("selected_frame_count"),
            "prompt_point_count": sam_promptable_proposals.get("prompt_point_count"),
            "raw_sam_mask_candidate_count": sam_promptable_proposals.get("raw_sam_mask_candidate_count"),
            "saved_promptable_proposal_mask_count": sam_promptable_proposals.get("saved_promptable_proposal_mask_count"),
            "proposal_state_counts": sam_promptable_proposals.get("proposal_state_counts"),
            "accepted_part_track_count": sam_promptable_proposals.get("accepted_part_track_count"),
            "semantic_part_label_ready_count": sam_promptable_proposals.get("semantic_part_label_ready_count"),
            "mask_evidence_created_count": sam_promptable_proposals.get("mask_evidence_created_count"),
            "part_pose_ready_count": sam_promptable_proposals.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": sam_promptable_proposals.get("object_pose_requirement_met_count"),
        },
        "part_mask_promotion_gate_qc": {
            "object_count": part_mask_promotion_gate.get("object_count"),
            "promotion_gate_state_counts": part_mask_promotion_gate.get("promotion_gate_state_counts"),
            "saved_promptable_proposal_mask_count": part_mask_promotion_gate.get("saved_promptable_proposal_mask_count"),
            "objects_with_saved_promptable_proposals_count": part_mask_promotion_gate.get("objects_with_saved_promptable_proposals_count"),
            "promoted_part_track_count": part_mask_promotion_gate.get("promoted_part_track_count"),
            "mask_evidence_created_count": part_mask_promotion_gate.get("mask_evidence_created_count"),
            "part_geometry_extraction_ready_count": part_mask_promotion_gate.get("part_geometry_extraction_ready_count"),
            "part_pose_ready_count": part_mask_promotion_gate.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": part_mask_promotion_gate.get("object_pose_requirement_met_count"),
            "contact_ownership_ready_count": part_mask_promotion_gate.get("contact_ownership_ready_count"),
        },
        "part_mask_acquisition_qc": {
            "object_count": part_mask_acquisition.get("object_count"),
            "local_new_mask_generation_ready_count": part_mask_acquisition.get("local_new_mask_generation_ready_count"),
            "mask_evidence_created_count": part_mask_acquisition.get("mask_evidence_created_count"),
            "unclassified_acquisition_blocker_count": part_mask_acquisition.get("unclassified_acquisition_blocker_count"),
            "acquisition_blocker_counts": part_mask_acquisition.get("acquisition_blocker_counts"),
            "part_pose_ready_count": part_mask_acquisition.get("part_pose_ready_count"),
            "object_pose_requirement_met_count": part_mask_acquisition.get("object_pose_requirement_met_count"),
            "environment": part_mask_acquisition.get("environment"),
            "promptable_segmentation_backend_available": require_dict(part_mask_acquisition.get("environment"), "part mask acquisition environment").get("promptable_segmentation_backend_available"),
            "open_vocab_detector_backend_cached_available": require_dict(part_mask_acquisition.get("environment"), "part mask acquisition environment").get("open_vocab_detector_backend_cached_available"),
            "open_vocab_or_referring_prompt_backend_available": require_dict(part_mask_acquisition.get("environment"), "part mask acquisition environment").get("open_vocab_or_referring_prompt_backend_available"),
            "model_produced_part_prompt_plan_ready": require_dict(part_mask_acquisition.get("environment"), "part mask acquisition environment").get("model_produced_part_prompt_plan_ready"),
            "local_new_mask_generation_ready": require_dict(part_mask_acquisition.get("environment"), "part mask acquisition environment").get("local_new_mask_generation_ready"),
        },
        "status_deliverable_ready": True,
        "final_pose_complete_deliverable_ready": False,
        **FALSE_READY,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [read_case(case, args) for case in args.cases]
    measured_runtime_path = args.measured_status_pipeline_runtime_root / "v18_measured_status_pipeline_runtime_report.json"
    measured_runtime = require_dict(load_json(measured_runtime_path), "measured status pipeline runtime") if measured_runtime_path.exists() else {}
    invariant_audit_path = args.status_invariant_audit_root / "v18_status_invariant_audit_report.json"
    invariant_audit = require_dict(load_json(invariant_audit_path), "status invariant audit") if invariant_audit_path.exists() else {}
    elapsed = time.perf_counter() - start
    total_duration = sum(require_float(case.get("duration_s"), "case duration_s") for case in cases)
    total_render_elapsed = sum(require_float(require_dict(case.get("status_runtime_qc"), "runtime qc").get("measured_render_elapsed_s"), "render elapsed") for case in cases)
    occlusion_candidate_owner_row_count = sum(
        require_int(require_dict(case.get("occlusion_owner_candidate_qc"), "occlusion candidate qc").get("candidate_owner_row_count"), "occlusion candidate rows")
        for case in cases
    )
    occluder_owner_accepted_count = sum(
        require_int(require_dict(case.get("occlusion_owner_candidate_qc"), "occlusion candidate qc").get("occluder_owner_accepted_count"), "occluder accepted")
        for case in cases
    )
    occlusion_depth_order_resolved_count = sum(
        require_int(require_dict(case.get("occlusion_owner_candidate_qc"), "occlusion candidate qc").get("depth_order_resolved_count"), "occlusion depth order")
        for case in cases
    )
    bounded_occlusion_owner_candidate_rows = sum(
        require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("occlusion_owner_candidate_rows"), "bounded occlusion candidate rows")
        for case in cases
    )
    bounded_occluder_owner_accepted_rows = sum(
        require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("occluder_owner_accepted_rows"), "bounded occluder accepted rows")
        for case in cases
    )
    bounded_occlusion_depth_order_resolved_rows = sum(
        require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("occlusion_depth_order_resolved_rows"), "bounded occlusion depth rows")
        for case in cases
    )
    occlusion_depth_evidence_candidate_pair_rows = sum(
        require_int(require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("candidate_pair_count"), "occlusion depth candidate pairs")
        for case in cases
    )
    occlusion_depth_evidence_same_frame_surface_pair_rows = sum(
        require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("same_frame_object_surface_pair_count"),
            "occlusion depth surface pairs",
        )
        for case in cases
    )
    occlusion_depth_evidence_foreground_support_pair_rows = sum(
        require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("foreground_occluder_support_pair_count"),
            "occlusion depth support pairs",
        )
        for case in cases
    )
    occlusion_depth_evidence_foreground_contradiction_pair_rows = sum(
        require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("foreground_occluder_contradiction_pair_count"),
            "occlusion depth contradiction pairs",
        )
        for case in cases
    )
    occlusion_depth_evidence_metric_compatible_pair_rows = sum(
        require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("metric_compatible_no_foreground_signal_pair_count"),
            "occlusion depth metric compatible pairs",
        )
        for case in cases
    )
    occlusion_depth_evidence_insufficient_pair_rows = sum(
        require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("insufficient_object_surface_depth_pair_count"),
            "occlusion depth insufficient object pairs",
        )
        + require_int(
            require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("insufficient_or_untrusted_hand_depth_pair_count"),
            "occlusion depth insufficient hand pairs",
        )
        for case in cases
    )
    occlusion_depth_evidence_owner_accepted_count = sum(
        require_int(require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("occluder_owner_accepted_count"), "occlusion depth accepted count")
        for case in cases
    )
    occlusion_depth_evidence_depth_order_resolved_count = sum(
        require_int(require_dict(case.get("occlusion_depth_order_evidence_qc"), "occlusion depth evidence qc").get("depth_order_resolved_count"), "occlusion depth resolved count")
        for case in cases
    )
    visible_surface_rows = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("surface_frame_rows"), "surface rows")
        for case in cases
    )
    visible_geometry_vertices = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("total_vertices"), "vertices") for case in cases
    )
    visible_geometry_faces = sum(
        require_int(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("total_faces"), "faces") for case in cases
    )
    physical_state_schema_object_count = sum(
        require_int(require_dict(case.get("physical_state_schema_qc"), "physical schema qc").get("object_count"), "physical schema object count")
        for case in cases
    )
    structured_part_or_relative_motion_required_count = sum(
        require_int(
            require_dict(case.get("physical_state_schema_qc"), "physical schema qc").get("part_or_relative_motion_required_count"),
            "structured part motion count",
        )
        for case in cases
    )
    structured_secondary_deformable_or_surface_component_count = sum(
        require_int(
            require_dict(case.get("physical_state_schema_qc"), "physical schema qc").get("secondary_deformable_or_surface_component_count"),
            "structured secondary deformable count",
        )
        for case in cases
    )
    physical_state_changed_from_legacy_keyword_count = sum(
        require_int(
            require_dict(case.get("physical_state_schema_qc"), "physical schema qc").get("changed_from_legacy_keyword_count"),
            "physical state changed from legacy count",
        )
        for case in cases
    )
    hand_baseline_row_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("hand_state_row_count"), "hand baseline rows")
        for case in cases
    )
    wilor_measurement_row_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("wilor_measurement_row_count"), "WiLoR rows")
        for case in cases
    )
    hawor_measurement_row_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("hawor_measurement_row_count"), "HaWoR rows")
        for case in cases
    )
    hawor_available_measurement_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("hawor_available_measurement_count"), "HaWoR available")
        for case in cases
    )
    hawor_motion_infill_candidate_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("hawor_motion_infill_candidate_count"), "HaWoR infill")
        for case in cases
    )
    hawor_full_video_baseline_ready_all_cases = all(
        bool(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("hawor_full_video_baseline_ready"))
        for case in cases
    )
    rtmlib_loaded_case_count = sum(
        1 for case in cases if bool(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("rtmlib_source_status_normalized"))
    )
    rtmlib_frames_with_hands = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("rtmlib_frames_with_hands"), "RTMLib frames")
        for case in cases
    )
    rtmlib_wilor_comparison_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("rtmlib_wilor_comparison_count"), "RTMLib WiLoR comparisons")
        for case in cases
    )
    hawor_temporal_occlusion_pose_accepted_count = sum(
        require_int(require_dict(case.get("hand_baseline_qc"), "hand baseline qc").get("temporal_occlusion_pose_accepted_count"), "HaWoR accepted occlusion")
        for case in cases
    )
    completion_candidate_count = sum(
        require_int(require_dict(case.get("completion_gate_qc"), "completion gate qc").get("completion_candidate_count"), "completion candidates")
        for case in cases
    )
    completion_run_count = sum(
        require_int(require_dict(case.get("completion_gate_qc"), "completion gate qc").get("completion_run_count"), "completion run")
        for case in cases
    )
    completion_pose_ready_count = sum(
        require_int(require_dict(case.get("completion_gate_qc"), "completion gate qc").get("complete_object_pose_ready_count"), "pose ready")
        for case in cases
    )
    part_split_candidate_count = sum(
        require_int(require_dict(case.get("completion_gate_qc"), "completion gate qc").get("part_split_candidate_count"), "part split candidates")
        for case in cases
    )
    part_track_source_manifest_ready_all_cases = all(
        bool(require_dict(case.get("part_track_source_qc"), "part track source qc").get("candidate_source_manifest_ready"))
        for case in cases
    )
    part_track_source_root_count = sum(
        require_int(require_dict(case.get("part_track_source_qc"), "part track source qc").get("root_count"), "part track source root count")
        for case in cases
    )
    part_track_source_usable_track_count = sum(
        require_int(require_dict(case.get("part_track_source_qc"), "part track source qc").get("usable_track_count"), "part track source usable track count")
        for case in cases
    )
    uniform_part_track_generation_ready = all(
        bool(require_dict(case.get("part_track_source_qc"), "part track source qc").get("uniform_part_track_generation_ready"))
        for case in cases
    )
    part_required_object_count = sum(
        require_int(require_dict(case.get("part_split_evidence_qc"), "part split qc").get("part_required_object_count"), "part required")
        for case in cases
    )
    accepted_part_track_assignment_count = sum(
        require_int(require_dict(case.get("part_split_evidence_qc"), "part split qc").get("accepted_part_track_assignment_count"), "accepted part tracks")
        for case in cases
    )
    part_geometry_ready_count = sum(
        require_int(require_dict(case.get("part_split_evidence_qc"), "part split qc").get("part_geometry_extraction_ready_count"), "part geometry ready")
        for case in cases
    )
    part_pose_ready_count = sum(
        require_int(require_dict(case.get("part_split_evidence_qc"), "part split qc").get("part_pose_ready_count"), "part pose ready")
        for case in cases
    )
    part_visible_surface_rows = sum(
        require_int(require_dict(case.get("part_visible_surface_qc"), "part surface qc").get("surface_frame_rows"), "part surface rows")
        for case in cases
    )
    part_visible_surface_vertices = sum(
        require_int(require_dict(case.get("part_visible_surface_qc"), "part surface qc").get("total_vertices"), "part vertices")
        for case in cases
    )
    part_visible_surface_faces = sum(
        require_int(require_dict(case.get("part_visible_surface_qc"), "part surface qc").get("total_faces"), "part faces")
        for case in cases
    )
    part_motion_object_count = sum(
        require_int(require_dict(case.get("part_motion_qc"), "part motion qc").get("object_count_with_part_surfaces"), "part motion object count")
        for case in cases
    )
    articulation_model_ready_count = sum(
        require_int(require_dict(case.get("part_motion_qc"), "part motion qc").get("articulation_model_ready_count"), "articulation ready")
        for case in cases
    )
    part_motion_qc_object_count = sum(
        sum(require_dict(case.get("part_motion_confound_qc"), "part motion confound qc").get("part_motion_qc_state_counts", {}).values())
        for case in cases
    )
    part_model_candidate_count = sum(
        require_int(require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("candidate_count"), "part model candidate count")
        for case in cases
    )
    part_model_rejected_candidate_count = sum(
        require_int(require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("rejected_candidate_count"), "rejected part model candidate count")
        for case in cases
    )
    part_surface_icp_probe_count = sum(
        require_int(require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("surface_icp_probe_count"), "part surface icp probe count")
        for case in cases
    )
    part_surface_icp_probe_state_counts = dict(
        sorted(
            sum(
                (
                    Counter(require_dict(require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("surface_icp_probe_state_counts"), "part surface icp state counts"))
                    for case in cases
                ),
                Counter(),
            ).items()
        )
    )
    part_articulation_hypothesis_pair_count = sum(
        require_int(require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("articulation_hypothesis_pair_count"), "part articulation hypothesis pair count")
        for case in cases
    )
    articulation_fit_probe_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("articulation_fit_probe_count"), "articulation fit probe count")
        for case in cases
    )
    articulation_fit_supported_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("articulation_fit_supported_count"), "articulation fit supported count")
        for case in cases
    )
    articulation_fit_rejected_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("articulation_fit_rejected_count"), "articulation fit rejected count")
        for case in cases
    )
    articulation_radial_residual_outlier_frame_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("radial_residual_outlier_frame_count"), "articulation radial outlier count")
        for case in cases
    )
    articulation_plane_residual_outlier_frame_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("plane_residual_outlier_frame_count"), "articulation plane outlier count")
        for case in cases
    )
    articulation_combined_residual_outlier_frame_count = sum(
        require_int(require_dict(case.get("articulation_fit_qc"), "articulation fit qc").get("combined_residual_outlier_frame_count"), "articulation combined outlier count")
        for case in cases
    )
    part_se3_pair_count = sum(
        require_int(require_dict(case.get("part_se3_surface_residual_qc"), "part se3 surface residual qc").get("part_se3_pair_count"), "part se3 pair count")
        for case in cases
    )
    part_se3_surface_supported_count = sum(
        int(require_dict(require_dict(case.get("part_se3_surface_residual_qc"), "part se3 surface residual qc").get("part_surface_se3_state_counts"), "part surface se3 states").get("part_surface_se3_residual_supported_visible_only_not_pose", 0))
        for case in cases
    )
    part_se3_surface_rejected_count = sum(
        int(require_dict(require_dict(case.get("part_se3_surface_residual_qc"), "part se3 surface residual qc").get("part_surface_se3_state_counts"), "part surface se3 states").get("part_surface_se3_residual_rejected", 0))
        for case in cases
    )
    part_se3_pair_rejected_count = sum(
        int(require_dict(require_dict(case.get("part_se3_surface_residual_qc"), "part se3 surface residual qc").get("part_se3_pair_state_counts"), "part se3 pair states").get("part_se3_surface_residual_rejected", 0))
        for case in cases
    )
    visible_subset_model_candidate_count = sum(
        require_int(
            require_dict(case.get("part_model_candidate_qc"), "part model candidate qc").get("visible_subset_model_candidate_count"),
            "visible subset model candidate count",
        )
        for case in cases
    )
    visible_part_subset_archive_ready_count = sum(
        1
        for case in cases
        if bool(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("visible_part_subset_archive_ready"))
    )
    visible_part_subset_archive_ready = visible_part_subset_archive_ready_count > 0
    all_cases_visible_part_subset_archive_ready = all(
        bool(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("visible_part_subset_archive_ready"))
        for case in cases
    )
    visible_part_subset_archive_file_written_all_cases = all(
        bool(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("visible_part_subset_archive_file_written"))
        for case in cases
    )
    visible_part_subset_archive_rows = sum(
        require_int(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("archive_row_count"), "visible part subset rows")
        for case in cases
    )
    visible_part_subset_vertices = sum(
        require_int(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("total_vertices"), "visible part subset vertices")
        for case in cases
    )
    visible_part_subset_faces = sum(
        require_int(require_dict(case.get("visible_part_subset_archive_qc"), "visible part subset archive qc").get("total_faces"), "visible part subset faces")
        for case in cases
    )
    required_part_object_blocker_count = sum(
        require_int(require_dict(case.get("part_object_blocker_qc"), "part object blocker qc").get("required_part_object_count"), "required part object blocker count")
        for case in cases
    )
    part_object_blocker_rejected_candidate_count = sum(
        require_int(require_dict(case.get("part_object_blocker_qc"), "part object blocker qc").get("rejected_part_model_candidate_count"), "blocker rejected candidate count")
        for case in cases
    )
    contact_ownership_ready_count = sum(
        require_int(require_dict(case.get("part_object_blocker_qc"), "part object blocker qc").get("contact_ownership_ready_count"), "contact ownership ready count")
        for case in cases
    )
    sam_promptable_selected_frame_count = sum(
        require_int(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("selected_frame_count"), "sam selected frame count")
        for case in cases
    )
    sam_promptable_raw_candidate_count = sum(
        require_int(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("raw_sam_mask_candidate_count"), "sam raw candidate count")
        for case in cases
    )
    sam_promptable_saved_proposal_mask_count = sum(
        require_int(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("saved_promptable_proposal_mask_count"), "sam saved proposal count")
        for case in cases
    )
    sam_promptable_not_referring_part_track_count = sum(
        require_int(
            require_dict(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("proposal_state_counts"), "sam proposal state counts").get(
                "promptable_sam_proposal_not_referring_part_track",
                0,
            ),
            "sam not referring part track proposal count",
        )
        for case in cases
    )
    sam_promptable_accepted_part_track_count = sum(
        require_int(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("accepted_part_track_count"), "sam accepted part track count")
        for case in cases
    )
    sam_promptable_mask_evidence_created_count = sum(
        require_int(require_dict(case.get("sam_promptable_proposal_qc"), "sam proposal qc").get("mask_evidence_created_count"), "sam mask evidence created count")
        for case in cases
    )
    part_mask_promotion_gate_object_count = sum(
        require_int(require_dict(case.get("part_mask_promotion_gate_qc"), "part mask promotion gate qc").get("object_count"), "part mask promotion gate object count")
        for case in cases
    )
    part_mask_promotion_gate_saved_proposal_mask_count = sum(
        require_int(
            require_dict(case.get("part_mask_promotion_gate_qc"), "part mask promotion gate qc").get("saved_promptable_proposal_mask_count"),
            "part mask promotion saved proposals",
        )
        for case in cases
    )
    part_mask_promotion_gate_promoted_part_track_count = sum(
        require_int(require_dict(case.get("part_mask_promotion_gate_qc"), "part mask promotion gate qc").get("promoted_part_track_count"), "promoted part tracks")
        for case in cases
    )
    part_mask_promotion_gate_mask_evidence_created_count = sum(
        require_int(require_dict(case.get("part_mask_promotion_gate_qc"), "part mask promotion gate qc").get("mask_evidence_created_count"), "promotion mask evidence")
        for case in cases
    )
    part_mask_acquisition_object_count = sum(
        require_int(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("object_count"), "part mask acquisition object count")
        for case in cases
    )
    local_new_mask_generation_ready_count = sum(
        require_int(
            require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("local_new_mask_generation_ready_count"),
            "local new mask generation ready count",
        )
        for case in cases
    )
    mask_evidence_created_count = sum(
        require_int(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("mask_evidence_created_count"), "mask evidence created count")
        for case in cases
    )
    unclassified_acquisition_blocker_count = sum(
        require_int(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("unclassified_acquisition_blocker_count"), "unclassified acquisition blocker count")
        for case in cases
    )
    promptable_segmentation_backend_available = any(
        bool(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("promptable_segmentation_backend_available"))
        for case in cases
    )
    open_vocab_detector_backend_cached_available = any(
        bool(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("open_vocab_detector_backend_cached_available"))
        for case in cases
    )
    open_vocab_or_referring_prompt_backend_available = any(
        bool(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("open_vocab_or_referring_prompt_backend_available"))
        for case in cases
    )
    model_produced_part_prompt_plan_ready = all(
        bool(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("model_produced_part_prompt_plan_ready"))
        for case in cases
    )
    local_new_mask_generation_ready = all(
        bool(require_dict(case.get("part_mask_acquisition_qc"), "part mask acquisition qc").get("local_new_mask_generation_ready"))
        for case in cases
    )
    manifest = {
        "method": "build_v18_status_deliverable_manifest",
        "status": STATUS,
        "claim": CLAIM,
        "build_elapsed_s": elapsed,
        "case_count": len(cases),
        "status_deliverable_ready": True,
        "final_pose_complete_deliverable_ready": False,
        "all_frame_counts_match_raw": all(bool(case.get("frame_count_match")) for case in cases),
        "all_status_renders_under_10x_realtime": all(
            bool(require_dict(case.get("status_runtime_qc"), "runtime qc").get("under_10x_realtime_for_status_render")) for case in cases
        ),
        "all_status_video_durations_match_raw": all(
            bool(require_dict(case.get("duration_qc"), "duration qc").get("all_durations_match_raw")) for case in cases
        ),
        "all_status_video_fps_match_raw": all(
            bool(require_dict(case.get("duration_qc"), "duration qc").get("all_fps_match_raw")) for case in cases
        ),
        "occlusion_candidate_owner_row_count": occlusion_candidate_owner_row_count,
        "occluder_owner_accepted_count": occluder_owner_accepted_count,
        "occlusion_depth_order_resolved_count": occlusion_depth_order_resolved_count,
        "bounded_occlusion_owner_candidate_rows": bounded_occlusion_owner_candidate_rows,
        "bounded_occluder_owner_accepted_rows": bounded_occluder_owner_accepted_rows,
        "bounded_occlusion_depth_order_resolved_rows": bounded_occlusion_depth_order_resolved_rows,
        "occlusion_depth_evidence_candidate_pair_rows": occlusion_depth_evidence_candidate_pair_rows,
        "occlusion_depth_evidence_same_frame_surface_pair_rows": occlusion_depth_evidence_same_frame_surface_pair_rows,
        "occlusion_depth_evidence_foreground_support_pair_rows": occlusion_depth_evidence_foreground_support_pair_rows,
        "occlusion_depth_evidence_foreground_contradiction_pair_rows": occlusion_depth_evidence_foreground_contradiction_pair_rows,
        "occlusion_depth_evidence_metric_compatible_pair_rows": occlusion_depth_evidence_metric_compatible_pair_rows,
        "occlusion_depth_evidence_insufficient_pair_rows": occlusion_depth_evidence_insufficient_pair_rows,
        "occlusion_depth_evidence_owner_accepted_count": occlusion_depth_evidence_owner_accepted_count,
        "occlusion_depth_evidence_depth_order_resolved_count": occlusion_depth_evidence_depth_order_resolved_count,
        "visible_geometry_archive_ready": all(
            bool(require_dict(case.get("visible_geometry_qc"), "visible geometry qc").get("visible_geometry_archive_ready")) for case in cases
        ),
        "visible_geometry_surface_frame_rows": visible_surface_rows,
        "visible_geometry_vertices": visible_geometry_vertices,
        "visible_geometry_faces": visible_geometry_faces,
        "physical_state_schema_object_count": physical_state_schema_object_count,
        "structured_part_or_relative_motion_required_count": structured_part_or_relative_motion_required_count,
        "structured_secondary_deformable_or_surface_component_count": structured_secondary_deformable_or_surface_component_count,
        "physical_state_changed_from_legacy_keyword_count": physical_state_changed_from_legacy_keyword_count,
        "hand_baseline_row_count": hand_baseline_row_count,
        "wilor_measurement_row_count": wilor_measurement_row_count,
        "hawor_measurement_row_count": hawor_measurement_row_count,
        "hawor_available_measurement_count": hawor_available_measurement_count,
        "hawor_motion_infill_candidate_count": hawor_motion_infill_candidate_count,
        "hawor_full_video_baseline_ready_all_cases": hawor_full_video_baseline_ready_all_cases,
        "rtmlib_loaded_case_count": rtmlib_loaded_case_count,
        "rtmlib_frames_with_hands": rtmlib_frames_with_hands,
        "rtmlib_wilor_comparison_count": rtmlib_wilor_comparison_count,
        "hawor_temporal_occlusion_pose_accepted_count": hawor_temporal_occlusion_pose_accepted_count,
        "object_completion_candidate_count": completion_candidate_count,
        "object_part_split_candidate_count": part_split_candidate_count,
        "object_completion_run_count": completion_run_count,
        "object_completion_pose_ready_count": completion_pose_ready_count,
        "part_track_source_manifest_ready_all_cases": part_track_source_manifest_ready_all_cases,
        "part_track_source_root_count": part_track_source_root_count,
        "part_track_source_usable_track_count": part_track_source_usable_track_count,
        "uniform_part_track_generation_ready": uniform_part_track_generation_ready,
        "part_required_object_count": part_required_object_count,
        "accepted_part_track_assignment_count": accepted_part_track_assignment_count,
        "part_geometry_extraction_ready_count": part_geometry_ready_count,
        "part_pose_ready_count": part_pose_ready_count,
        "part_visible_surface_frame_rows": part_visible_surface_rows,
        "part_visible_surface_vertices": part_visible_surface_vertices,
        "part_visible_surface_faces": part_visible_surface_faces,
        "part_motion_object_count": part_motion_object_count,
        "articulation_model_ready_count": articulation_model_ready_count,
        "part_motion_qc_object_count": part_motion_qc_object_count,
        "part_model_candidate_count": part_model_candidate_count,
        "part_model_rejected_candidate_count": part_model_rejected_candidate_count,
        "part_surface_icp_probe_count": part_surface_icp_probe_count,
        "part_surface_icp_probe_state_counts": part_surface_icp_probe_state_counts,
        "part_articulation_hypothesis_pair_count": part_articulation_hypothesis_pair_count,
        "articulation_fit_probe_count": articulation_fit_probe_count,
        "articulation_fit_supported_count": articulation_fit_supported_count,
        "articulation_fit_rejected_count": articulation_fit_rejected_count,
        "articulation_radial_residual_outlier_frame_count": articulation_radial_residual_outlier_frame_count,
        "articulation_plane_residual_outlier_frame_count": articulation_plane_residual_outlier_frame_count,
        "articulation_combined_residual_outlier_frame_count": articulation_combined_residual_outlier_frame_count,
        "part_se3_pair_count": part_se3_pair_count,
        "part_se3_surface_supported_count": part_se3_surface_supported_count,
        "part_se3_surface_rejected_count": part_se3_surface_rejected_count,
        "part_se3_pair_rejected_count": part_se3_pair_rejected_count,
        "visible_subset_model_candidate_count": visible_subset_model_candidate_count,
        "visible_part_subset_archive_file_written_all_cases": visible_part_subset_archive_file_written_all_cases,
        "visible_part_subset_archive_ready": visible_part_subset_archive_ready,
        "visible_part_subset_archive_ready_scope": "one_or_more_nonempty_candidate_archives",
        "visible_part_subset_archive_ready_count": visible_part_subset_archive_ready_count,
        "all_cases_visible_part_subset_archive_ready": all_cases_visible_part_subset_archive_ready,
        "visible_part_subset_archive_rows": visible_part_subset_archive_rows,
        "visible_part_subset_vertices": visible_part_subset_vertices,
        "visible_part_subset_faces": visible_part_subset_faces,
        "required_part_object_blocker_count": required_part_object_blocker_count,
        "part_object_blocker_rejected_candidate_count": part_object_blocker_rejected_candidate_count,
        "contact_ownership_ready_count": contact_ownership_ready_count,
        "sam_promptable_selected_frame_count": sam_promptable_selected_frame_count,
        "sam_promptable_raw_mask_candidate_count": sam_promptable_raw_candidate_count,
        "sam_promptable_saved_proposal_mask_count": sam_promptable_saved_proposal_mask_count,
        "sam_promptable_not_referring_part_track_count": sam_promptable_not_referring_part_track_count,
        "sam_promptable_accepted_part_track_count": sam_promptable_accepted_part_track_count,
        "sam_promptable_mask_evidence_created_count": sam_promptable_mask_evidence_created_count,
        "part_mask_promotion_gate_object_count": part_mask_promotion_gate_object_count,
        "part_mask_promotion_gate_saved_proposal_mask_count": part_mask_promotion_gate_saved_proposal_mask_count,
        "part_mask_promotion_gate_promoted_part_track_count": part_mask_promotion_gate_promoted_part_track_count,
        "part_mask_promotion_gate_mask_evidence_created_count": part_mask_promotion_gate_mask_evidence_created_count,
        "part_mask_acquisition_object_count": part_mask_acquisition_object_count,
        "promptable_segmentation_backend_available": promptable_segmentation_backend_available,
        "open_vocab_detector_backend_cached_available": open_vocab_detector_backend_cached_available,
        "open_vocab_or_referring_prompt_backend_available": open_vocab_or_referring_prompt_backend_available,
        "model_produced_part_prompt_plan_ready": model_produced_part_prompt_plan_ready,
        "local_new_mask_generation_ready": local_new_mask_generation_ready,
        "local_new_mask_generation_ready_count": local_new_mask_generation_ready_count,
        "mask_evidence_created_count": mask_evidence_created_count,
        "unclassified_acquisition_blocker_count": unclassified_acquisition_blocker_count,
        "cached_evidence_to_status_runtime_measured": bool(measured_runtime.get("cached_evidence_to_status_runtime_measured")),
        "cached_evidence_to_status_runtime_report": str(measured_runtime_path) if measured_runtime else None,
        "cached_evidence_to_status_elapsed_s": measured_runtime.get("total_elapsed_s"),
        "cached_evidence_to_status_elapsed_to_video_ratio": measured_runtime.get("total_elapsed_to_video_ratio"),
        "cached_evidence_to_status_stage_count": measured_runtime.get("stage_count"),
        "fresh_raw_video_to_status_runtime_measured": bool(measured_runtime.get("fresh_raw_video_to_status_runtime_measured")),
        "fresh_raw_video_to_final_pose_runtime_measured": bool(measured_runtime.get("fresh_raw_video_to_final_pose_runtime_measured")),
        "status_invariant_audit_report": str(invariant_audit_path) if invariant_audit else None,
        "status_invariant_audit_passed": invariant_audit.get("audit_passed"),
        "status_invariant_failed_required_check_count": invariant_audit.get("failed_required_check_count"),
        "status_invariant_required_check_count": invariant_audit.get("required_check_count"),
        "total_duration_s": total_duration,
        "total_measured_render_elapsed_s": total_render_elapsed,
        "total_measured_render_to_video_ratio": total_render_elapsed / total_duration if total_duration > 0 else None,
        "default_path_uses_bundlesdf_or_nerf": False,
        "contact_factor_ready_rows": sum(
            require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("contact_factor_ready_rows"), "contact ready") for case in cases
        ),
        "pose_filled_through_occlusion_rows": sum(
            require_int(require_dict(case.get("bounded_state_qc"), "bounded qc").get("pose_filled_through_occlusion_rows"), "pose filled") for case in cases
        ),
        "cases": cases,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_status_deliverable_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--solution-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--render-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_renders"))
    parser.add_argument("--occlusion-owner-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_candidates"))
    parser.add_argument("--occlusion-depth-evidence-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--hand-baseline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--completion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_object_completion_gate"))
    parser.add_argument("--part-track-source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_track_source_manifest"))
    parser.add_argument("--part-split-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_split_evidence"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--part-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_state"))
    parser.add_argument("--part-motion-qc-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--articulation-fit-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--part-se3-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_se3_surface_residuals"))
    parser.add_argument("--visible-part-subset-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_part_subset_archive"))
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--sam-promptable-proposals-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals"))
    parser.add_argument("--part-mask-promotion-gate-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_promotion_gate"))
    parser.add_argument("--part-mask-acquisition-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan"))
    parser.add_argument("--measured-status-pipeline-runtime-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_measured_status_pipeline_runtime"))
    parser.add_argument("--status-invariant-audit-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_invariant_audit"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_deliverable_manifest"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
