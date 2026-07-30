#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STATUS = "v18_measured_status_pipeline_runtime"
CLAIM = (
    "This artifact measures the implemented V18 cached-evidence-to-status pipeline by running the current V18 "
    "reducers and status renderers in dependency order. It includes raw-frame status rendering and regenerates the "
    "V18 HaWoR/WiLoR/RTMLib hand-baseline reducer and OWLv2->SAM2 part-track reducer, but upstream raw hand/object/depth "
    "model outputs remain cached V16/V17/V18 inputs. It is not proof of fresh raw-video-to-final pose-complete runtime."
)

FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STAGES: list[dict[str, Any]] = [
    {"id": "runtime_manifest", "script": "scripts/build_v18_runtime_manifest.py", "source_scope": "static_design_budget"},
    {"id": "physical_state_schema", "script": "scripts/build_v18_physical_state_schema.py", "source_scope": "cached_v17_roster_and_timeline"},
    {"id": "hand_baseline_branch", "script": "scripts/build_v18_hand_baseline_branch.py", "source_scope": "cached_wilor_hawor_rtmlib_measurements_plus_interior_hand_depth"},
    {"id": "visibility_occlusion_state", "script": "scripts/build_v18_visibility_occlusion_state.py", "source_scope": "cached_measurements_plus_v18_hand_baseline_branch"},
    {"id": "fast_motion_state", "script": "scripts/build_v18_fast_motion_state.py", "source_scope": "cached_visibility_surfaces_material_tracks"},
    {"id": "consistency_graph", "script": "scripts/build_v18_consistency_graph.py", "source_scope": "cached_visibility_motion_contact_depth"},
    {"id": "annotation_state", "script": "scripts/build_v18_annotation_state.py", "source_scope": "cached_timeline_and_v18_reducers"},
    {"id": "occlusion_owner_candidates", "script": "scripts/build_v18_occlusion_owner_candidates.py", "source_scope": "annotation_state_and_visibility_short_gaps"},
    {"id": "occlusion_depth_order_evidence", "script": "scripts/build_v18_occlusion_depth_order_evidence.py", "source_scope": "occlusion_candidates_plus_cached_hand_scene_depth_and_object_visible_surfaces"},
    {"id": "bounded_state_solution", "script": "scripts/build_v18_bounded_state_solution.py", "source_scope": "cached_v18_state_reducers"},
    {"id": "status_overlay_render", "script": "scripts/render_v18_status_overlay.py", "source_scope": "raw_frames_plus_cached_annotation_state"},
    {"id": "world_status_render", "script": "scripts/render_v18_world_status.py", "source_scope": "cached_bounded_state_solution"},
    {"id": "side_by_side_render", "script": "scripts/render_v18_side_by_side.py", "source_scope": "status_overlay_and_world_status_videos"},
    {"id": "visible_geometry_archive", "script": "scripts/build_v18_visible_geometry_archive.py", "source_scope": "cached_v17_visible_surfaces"},
    {"id": "object_completion_gate", "script": "scripts/build_v18_object_completion_gate.py", "source_scope": "v18_visible_geometry_fast_motion_physical_schema"},
    {"id": "owlv2_sam2_part_tracks", "script": "scripts/build_v18_owlv2_sam2_part_tracks.py", "source_scope": "vlm_physical_notes_plus_owlv2_keyframe_boxes_plus_sam2_video_tracking"},
    {"id": "part_track_source_manifest", "script": "scripts/build_v18_part_track_source_manifest.py", "source_scope": "v18_owlv2_sam2_generated_tracks_only_by_default"},
    {"id": "part_split_evidence", "script": "scripts/build_v18_part_split_evidence.py", "source_scope": "part_source_manifest_plus_whole_object_masks"},
    {"id": "part_visible_surfaces", "script": "scripts/build_v18_part_visible_surfaces.py", "source_scope": "accepted_part_masks_plus_cached_metric_depth"},
    {"id": "part_motion_state", "script": "scripts/build_v18_part_motion_state.py", "source_scope": "part_visible_surfaces"},
    {"id": "part_motion_qc", "script": "scripts/build_v18_part_motion_qc.py", "source_scope": "part_motion_state_and_part_surfaces"},
    {"id": "part_model_candidates", "script": "scripts/build_v18_part_model_candidates.py", "source_scope": "part_motion_qc"},
    {"id": "articulation_fit_candidates", "script": "scripts/build_v18_articulation_fit_candidates.py", "source_scope": "part_model_candidates_part_surfaces_and_v16_camera_poses"},
    {"id": "part_se3_surface_residuals", "script": "scripts/build_v18_part_se3_surface_residuals.py", "source_scope": "articulation_fit_part_surfaces_and_v16_camera_poses"},
    {"id": "visible_part_subset_archive", "script": "scripts/build_v18_visible_part_subset_archive.py", "source_scope": "part_model_candidates_and_part_surfaces"},
    {"id": "part_object_blocker_manifest", "script": "scripts/build_v18_part_object_blocker_manifest.py", "source_scope": "part_evidence_articulation_fit_part_se3_and_completion_gate"},
    {"id": "sam_promptable_part_proposals", "script": "scripts/build_v18_sam_promptable_part_proposals.py", "source_scope": "promptable_sam_probe_on_selected_blocked_object_frames_not_accepted_tracks"},
    {"id": "part_mask_acquisition_plan", "script": "scripts/build_v18_part_mask_acquisition_plan.py", "source_scope": "part_object_blockers_and_backend_probe"},
    {"id": "part_mask_promotion_gate", "script": "scripts/build_v18_part_mask_promotion_gate.py", "source_scope": "promptable_proposals_plus_acquisition_status_no_promotion"},
    {"id": "status_deliverable_manifest_pre_report", "script": "scripts/build_v18_status_deliverable_manifest.py", "source_scope": "all_current_v18_status_artifacts_before_runtime_report_write"},
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_stage(stage: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    stage_id = str(stage["id"])
    script = Path(str(stage["script"]))
    if not script.exists():
        raise RuntimeError(f"missing stage script {script}")
    stdout_path = args.output_root / "stage_logs" / f"{stage_id}.stdout.json"
    stderr_path = args.output_root / "stage_logs" / f"{stage_id}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script)]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=args.repo_root, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "stage_id": stage_id,
        "script": str(script),
        "command": command,
        "source_scope": stage.get("source_scope"),
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "success": proc.returncode == 0,
    }


def final_manifest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"manifest_exists": False}
    manifest = load_json(path)
    keys = [
        "status_deliverable_ready",
        "final_pose_complete_deliverable_ready",
        "all_frame_counts_match_raw",
        "all_status_video_durations_match_raw",
        "all_status_video_fps_match_raw",
        "total_duration_s",
        "total_measured_render_elapsed_s",
        "total_measured_render_to_video_ratio",
        "physical_state_schema_object_count",
        "object_part_split_candidate_count",
        "part_required_object_count",
        "part_track_source_usable_track_count",
        "local_new_mask_generation_ready_count",
        "mask_evidence_created_count",
        "contact_ownership_ready_count",
        "contact_factor_ready_rows",
        "object_pose_requirement_met",
    ]
    return {"manifest_exists": True, **{key: manifest.get(key) for key in keys}}


def post_report_status_manifest_refresh(args: argparse.Namespace, stage_id: str, source_scope: str) -> dict[str, Any]:
    stage = {"id": stage_id, "script": "scripts/build_v18_status_deliverable_manifest.py", "source_scope": source_scope}
    return run_stage(stage, args)


def post_report_status_invariant_audit(args: argparse.Namespace) -> dict[str, Any]:
    stage = {"id": "post_report_status_invariant_audit", "script": "scripts/audit_v18_status_invariants.py", "source_scope": "audit_manifest_after_runtime_report_refresh"}
    return run_stage(stage, args)


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []
    failed_stage: str | None = None
    for stage in STAGES:
        row = run_stage(stage, args)
        stage_rows.append(row)
        partial = {
            "method": "run_v18_measured_status_pipeline",
            "status": STATUS,
            "claim": CLAIM,
            "source_scope": "cached_evidence_to_status_pipeline_not_fresh_raw_video_to_final_pose",
            "stage_rows": stage_rows,
            "failed_stage": row["stage_id"] if not row["success"] else None,
            "pipeline_success": all(bool(item.get("success")) for item in stage_rows),
            "build_elapsed_s_so_far": time.perf_counter() - start,
        }
        write_json(args.output_root / "v18_measured_status_pipeline_runtime_report.partial.json", partial)
        if not row["success"]:
            failed_stage = row["stage_id"]
            break
    elapsed = time.perf_counter() - start
    manifest_path = args.status_manifest_root / "v18_status_deliverable_manifest.json"
    manifest_summary = final_manifest_summary(manifest_path)
    total_video_duration = manifest_summary.get("total_duration_s")
    ratio = elapsed / float(total_video_duration) if isinstance(total_video_duration, (int, float)) and float(total_video_duration) > 0 else None
    report = {
        "method": "run_v18_measured_status_pipeline",
        "status": STATUS,
        "claim": CLAIM,
        "source_scope": "cached_evidence_to_status_pipeline_not_fresh_raw_video_to_final_pose",
        "pipeline_success": failed_stage is None,
        "failed_stage": failed_stage,
        "stage_count": len(stage_rows),
        "successful_stage_count": sum(1 for row in stage_rows if bool(row.get("success"))),
        "total_elapsed_s": elapsed,
        "total_video_duration_s": total_video_duration,
        "total_elapsed_to_video_ratio": ratio,
        "stage_rows": stage_rows,
        "status_manifest_summary": manifest_summary,
        "cached_evidence_to_status_runtime_measured": failed_stage is None,
        "fresh_raw_video_to_status_runtime_measured": False,
        "fresh_raw_video_to_final_pose_runtime_measured": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    report_path = args.output_root / "v18_measured_status_pipeline_runtime_report.json"
    write_json(report_path, report)
    if failed_stage is not None:
        raise RuntimeError(f"V18 measured status pipeline failed at stage {failed_stage}; see {args.output_root}")
    runtime_refresh_row = post_report_status_manifest_refresh(
        args,
        "post_report_status_manifest_refresh",
        "refresh_manifest_after_runtime_report_write",
    )
    report["post_report_status_manifest_refresh"] = runtime_refresh_row
    report["status_manifest_summary_after_report_refresh"] = final_manifest_summary(manifest_path)
    write_json(report_path, report)
    if not runtime_refresh_row.get("success"):
        raise RuntimeError(f"V18 measured status pipeline post-report manifest refresh failed; see {runtime_refresh_row.get('stderr_path')}")
    audit_refresh_row = post_report_status_invariant_audit(args)
    report["post_report_status_invariant_audit"] = audit_refresh_row
    write_json(report_path, report)
    if not audit_refresh_row.get("success"):
        raise RuntimeError(f"V18 measured status pipeline post-report invariant audit failed; see {audit_refresh_row.get('stderr_path')}")
    final_manifest_refresh_row = post_report_status_manifest_refresh(
        args,
        "post_audit_status_manifest_refresh",
        "refresh_manifest_after_post_report_audit",
    )
    report["post_audit_status_manifest_refresh"] = final_manifest_refresh_row
    report["status_manifest_summary_after_post_audit_refresh"] = final_manifest_summary(manifest_path)
    write_json(report_path, report)
    if not final_manifest_refresh_row.get("success"):
        raise RuntimeError(f"V18 measured status pipeline post-audit manifest refresh failed; see {final_manifest_refresh_row.get('stderr_path')}")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_measured_status_pipeline_runtime"))
    parser.add_argument("--status-manifest-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_status_deliverable_manifest"))
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
