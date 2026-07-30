#!/usr/bin/env python3
"""Audit V21 atomic algorithm overlay/QC/tuning coverage.

The audit is deliberately evidence-only: it records which algorithm outputs and
overlays exist, which are missing, and which artifacts are deprecated. It does
not claim that missing heavy model runs have occurred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNNER_AGENT_ID = "runner_agent"
RUNNER_POLICY = "single_runner_agent_schedules_and_executes_every_v21_atomic_algorithm; family/source fields are provenance, not runner roles"

RUNS = {
    "pico": {
        "run_root": "output/v21_infer_20260626/pico_trackers_10_2_100s_120s_v21",
        "object_id": "red_scandic_tin",
    },
    "living_room": {
        "run_root": "output/v21_infer_20260626/living_room_cleanup_multiview_v21",
        "object_id": "clear_glass_bowl",
    },
}

ALGORITHMS = [
    # Raw-video/input adapters and consumed plan/state artifacts.
    {"id": "input_manifest", "family": "input", "source": "v21_raw_video_bootstrap", "overlay_type": "report", "data": "input/input_manifest.json", "overlay": "{case}/input_manifest/overlay.mp4", "claim_scope": "Input-video contract and raw clip provenance consumed by all downstream atoms."},
    {"id": "raw_frame_manifest", "family": "input", "source": "v21_raw_video_bootstrap", "overlay_type": "report", "data": "input/raw_frame_manifest/manifest.json", "overlay": "{case}/raw_frame_manifest/overlay.mp4", "claim_scope": "Decoded/resized raw-frame timeline consumed by depth, hand, and some candidate atoms."},
    {"id": "source_frame_manifest", "family": "input", "source": "v21_source_rgb_bootstrap", "overlay_type": "report", "data": "input/source_frame_manifest/manifest.json", "overlay": "{case}/source_frame_manifest/overlay.mp4", "claim_scope": "Source-resolution RGB frame timeline consumed by source-coordinate prompt and overlay atoms."},
    {"id": "depth_modality_report", "family": "depth", "source": "v21_added", "overlay_type": "report", "data": "measurements/camera_depth/depth_modality_report.json", "overlay": "{case}/depth_modality_report/overlay.mp4"},
    {"id": "depth_candidate_registry", "family": "depth", "source": "v20_reused_as_v21_registry", "overlay_type": "report", "data": "measurements/camera_depth/v20_depth_registry/depth_candidate_registry.json", "overlay": "{case}/depth_candidate_registry/overlay.mp4", "claim_scope": "Registry of depth/camera candidates consumed by depth selection; not itself selected depth."},
    {"id": "depth_selection_bundle", "family": "depth", "source": "v20_reused_as_v21_selection_bundle", "overlay_type": "report", "data": "measurements/camera_depth/v20_depth_selection_bundle.json", "overlay": "{case}/depth_selection_bundle/overlay.mp4"},
    {"id": "depth_camera_selection", "family": "depth", "source": "v21_added", "overlay_type": "report", "data": "measurements/camera_depth/depth_camera_selection_report.json", "overlay": "{case}/depth_camera_selection/overlay.mp4"},
    {"id": "segmentation_stable_keyframes", "family": "prompt", "source": "v21_added", "overlay_type": "report", "data": "measurements/object_candidates/segmentation_stable_keyframes.json", "alt_data": "measurements/object_candidates/segmentation_stable_keyframes_review.jpg", "overlay": "{case}/segmentation_stable_keyframes/overlay.mp4", "claim_scope": "Runner/object-plan keyframe-selection atom for OWLv2 detector frames; it is not itself a detector, mask tracker, geometry, or pose solver."},
    {"id": "object_plan_agent", "family": "prompt", "source": "runner_source_keyframe_visual_review", "overlay_type": "report", "data": "measurements/object_candidates/object_plan_agent.json", "overlay": "{case}/object_plan_agent/overlay.mp4", "claim_scope": "Current target-selection and prompt-plan atom from source-frame visual review; not masks, geometry, or pose."},
    {"id": "object_plan_current", "family": "prompt", "source": "runner_source_keyframe_visual_review", "overlay_type": "report", "data": "measurements/object_candidates/object_plan_current.json", "overlay": "{case}/object_plan_current/overlay.mp4", "claim_scope": "Current object plan consumed by OWLv2 keyframe proposal and bbox approval."},
    {"id": "v18_full_mano_annotations", "family": "state", "source": "v21_to_v18_bridge", "overlay_type": "report", "data": "state/annotations_v18_full_mano.json", "overlay": "{case}/v18_full_mano_annotations/overlay.mp4"},
    {"id": "v21_renderable_annotations", "family": "state", "source": "v21_state_assembly", "overlay_type": "report", "data": "state/annotations_v21_renderable.json", "overlay": "{case}/v21_renderable_annotations/overlay.mp4"},
    {"id": "v21_physical_state", "family": "state", "source": "v21_state_assembly", "overlay_type": "report", "data": "state/v21_physical_state.json", "overlay": "{case}/v21_physical_state/overlay.mp4"},
    {"id": "v21_uncertainty_state", "family": "state", "source": "v21_state_assembly", "overlay_type": "report", "data": "state/v21_uncertainty_state.json", "overlay": "{case}/v21_uncertainty_state/overlay.mp4"},
    # Master/V19/V18 depth-camera spine plus V21 additions.
    {"id": "depthpro", "family": "depth", "source": "v21_added", "overlay_type": "depth", "data": "measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz", "overlay": "{case}/depthpro/overlay.mp4"},
    {"id": "unidepth_v2", "family": "depth", "source": "master_v19", "overlay_type": "depth", "data": "measurements/depth_candidates/unidepth_v2/unidepth_v2_depth.npz", "overlay": "{case}/unidepth_v2/overlay.mp4"},
    {"id": "depth_anything_v2", "family": "depth", "source": "v21_added", "overlay_type": "depth", "data": "measurements/depth_candidates/depth_anything_v2/depth_anything_v2_depth.npz", "overlay": "{case}/depth_anything_v2/overlay.mp4"},
    {"id": "stereo_sgbm", "family": "depth", "source": "v21_added", "overlay_type": "depth", "data": "measurements/depth_candidates/stereo_sgbm/relative_inverse_depth.npz", "overlay": "{case}/stereo_sgbm/overlay.mp4", "claim_scope": "weak relative evidence only unless calibrated"},
    {"id": "droid_or_camera_trajectory", "family": "camera", "source": "master_v19", "overlay_type": "camera", "data": "measurements/camera_depth/depthpro_as_droid.npz", "overlay": "{case}/droid_or_camera_trajectory/overlay.mp4", "optional": True},
    # Bbox / segmentation.
    {"id": "owlv2_bbox", "family": "bbox", "source": "master_aligned_replacement", "overlay_type": "detection", "data": "measurements/object_candidates/owlv2_bbox_proposals.json", "overlay": "{case}/owlv2_bbox/overlay.mp4", "required_replacement_for": "groundingdino"},
    {"id": "owlv2_bbox_approved_prompts", "family": "bbox", "source": "v21_runner_bbox_approval", "overlay_type": "detection", "data": "measurements/object_candidates/owlv2_bbox_approved_prompts.json", "overlay": "{case}/owlv2_bbox_approved_prompts/overlay.mp4", "claim_scope": "Runner-approved OWLv2 keyframe boxes that seed SAM2 proper; not masks, geometry, or pose."},
    {"id": "sam2_proper", "family": "segmentation", "source": "v21_owlv2_bbox_prompt_sam2", "overlay_type": "segmentation", "data": "measurements/object_tracks/sam2_proper/{object_id}/sam2_masks", "alt_data": "measurements/object_tracks/sam2_proper/{object_id}/sam2_track.json", "native_overlay": "measurements/object_tracks/sam2_proper/{object_id}/sam2_proper_overlay.mp4", "overlay": "{case}/sam2_proper/overlay.mp4"},
    # Hand/MANO stack.
    {"id": "rtmlib_2d", "family": "hand", "source": "master_v19", "overlay_type": "hand", "data": "measurements/hand_candidates/rtmlib_2d/rtmlib_hand2d.json", "native_overlay": "measurements/hand_candidates/rtmlib_2d/rtmlib_hand2d_overlay.mp4", "overlay": "{case}/rtmlib_2d/overlay.mp4"},
    {"id": "wilor_mano", "family": "hand", "source": "master_v19", "overlay_type": "hand", "data": "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json", "overlay": "{case}/wilor_mano/overlay.mp4"},
    {"id": "wilor_metric_refit", "family": "hand", "source": "v21_added", "overlay_type": "hand", "data": "measurements/hand_candidates/wilor_v21_metric/wilor_metric_hands.json", "overlay": "{case}/wilor_metric_refit/overlay.mp4"},
    {"id": "hamer", "family": "hand", "source": "master_v19", "overlay_type": "hand", "data": "measurements/hand_candidates/hamer/hamer_raw_hands.json", "overlay": "{case}/hamer/overlay.mp4"},
    {"id": "hawor", "family": "hand", "source": "master_v19", "overlay_type": "report", "data": "v18_hand_baseline_branch/{run_case}/v18_hand_baseline_branch.json", "overlay": "{case}/hawor/overlay.mp4", "optional": False},
    {"id": "active_mano", "family": "hand", "source": "v21_added", "overlay_type": "hand", "data": "measurements/hand_candidates/v21_active_mano/optimized_mano_state.json", "overlay": "{case}/active_mano/overlay.mp4"},
    # Geometry / pose / graph / physical variables.
    {"id": "visible_surface", "family": "geometry", "source": "master_v19", "overlay_type": "report", "data": "measurements/object_visible_surfaces/depthpro_local_grabcut/visible_surface_summary.json", "native_overlay": "renders/v21_visible_surface_overlay.mp4", "overlay": "{case}/visible_surface/overlay.mp4"},
    {"id": "heightfield_observed", "family": "geometry", "source": "master_v18", "overlay_type": "heightfield", "data": "measurements/object_geometry/heightfield_observed/{object_id}/reconstruction", "overlay": "{case}/heightfield_observed/overlay.mp4"},
    {"id": "v21_mesh_candidate", "family": "geometry", "source": "v21_added", "overlay_type": "mesh_candidate", "data": "measurements/object_geometry/v21_mesh_candidate/{object_id}/mesh_completion_report.json", "overlay": "{case}/v21_mesh_candidate/overlay.mp4"},
    {"id": "v18_compact_rigid_pose_fit", "family": "pose_fit", "source": "master_v18", "overlay_type": "pose_mesh", "data": "measurements/object_geometry_mesh_pose/{object_id}/v18_icp_fit/v18_compact_rigid_object_pose_fit_report.json", "overlay": "{case}/v18_compact_rigid_pose_fit/overlay.mp4", "claim_scope": "Per-frame completed-mesh ICP fit to visible depth samples; observations for the temporal V19 pose graph, not final corrected pose."},
    {"id": "v19_rigid_pose_graph", "family": "pose_graph", "source": "master_v19", "overlay_type": "pose_mesh", "data": "measurements/object_geometry_mesh_pose/{object_id}/v19_pose_graph/v19_rigid_object_pose_graph_report.json", "overlay": "{case}/v19_rigid_pose_graph/overlay.mp4"},
    {"id": "adopted_object_pose", "family": "pose", "source": "current_adopted_v19", "overlay_type": "pose_mesh", "data": "measurements/object_geometry_mesh_pose/{object_id}/v19_pose_graph/v19_rigid_object_pose_graph_report.json", "overlay": "{case}/adopted_object_pose/overlay.mp4", "claim_scope": "Current adopted object pose source for consumers: v19_rigid_pose_graph pose_rows."},
    {"id": "object_factor_graph", "family": "pose_graph", "source": "master_v18", "overlay_type": "report", "data": "measurements/object_geometry_mesh_pose/{object_id}/v3_object_factor_graph/qc_object_factor_graph_v3.json", "overlay": "{case}/object_factor_graph/overlay.mp4"},
    {"id": "mesh_prior_graph", "family": "pose_graph", "source": "master_v18", "overlay_type": "report", "data": "measurements/object_geometry_mesh_pose/{object_id}/v3_mesh_prior_graph/qc_mesh_prior_pose_graph_v3.json", "overlay": "{case}/mesh_prior_graph/overlay.mp4"},
    {"id": "contact_patch_pose_graph", "family": "pose_graph", "source": "master_v18", "overlay_type": "report", "data": "measurements/object_geometry_mesh_pose/{object_id}/v3_contact_patch_graph/qc_contact_patch_object_pose_graph_v3.json", "overlay": "{case}/contact_patch_pose_graph/overlay.mp4"},
    {"id": "contact_ownership_graph", "family": "contact", "source": "master_v18", "overlay_type": "report", "data": "measurements/contact_occlusion_nonpenetration/v18_contact_ownership/{run_case}/v18_contact_ownership_graph_report.json", "overlay": "{case}/contact_ownership_graph/overlay.mp4"},
    {"id": "occlusion_owner_graph", "family": "occlusion", "source": "master_v18", "overlay_type": "report", "data": "measurements/contact_occlusion_nonpenetration/v18_occlusion_owner_graph/{run_case}/v18_occlusion_owner_graph_report.json", "overlay": "{case}/occlusion_owner_graph/overlay.mp4"},
    {"id": "signed_nonpenetration", "family": "nonpenetration", "source": "master_v18", "overlay_type": "report", "data": "measurements/contact_occlusion_nonpenetration/v18_signed_nonpenetration/{run_case}/v18_signed_nonpenetration_evidence_report.json", "overlay": "{case}/signed_nonpenetration/overlay.mp4"},
    {"id": "triangle_nonpenetration", "family": "nonpenetration", "source": "master_v18", "overlay_type": "report", "data": "measurements/contact_occlusion_nonpenetration/v18_triangle_nonpenetration/{run_case}/v18_triangle_nonpenetration_evidence_report.json", "overlay": "{case}/triangle_nonpenetration/overlay.mp4"},
    # Final render deliverables.
    {"id": "v21_final_overlay", "family": "render", "source": "v21", "data": "renders/v21_overlay.mp4", "native_overlay": "renders/v21_overlay.mp4", "overlay": "{case}/v21_final_overlay/overlay.mp4"},
    {"id": "v21_final_world", "family": "render", "source": "v21", "data": "renders/v21_world.mp4", "native_overlay": "renders/v21_world.mp4", "overlay": "{case}/v21_final_world/overlay.mp4"},
    {"id": "v21_final_side_by_side", "family": "render", "source": "v21", "data": "renders/v21_side_by_side.mp4", "native_overlay": "renders/v21_side_by_side.mp4", "overlay": "{case}/v21_final_side_by_side/overlay.mp4"},
]


def materialize(pattern: str | None, case: str, run_case: str, object_id: str) -> str | None:
    if not pattern:
        return None
    return pattern.format(case=case, run_case=run_case, object_id=object_id)


def exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def read_json_object(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def rejected_by_method(path: Path | None, substrings: list[str] | None) -> tuple[bool, str | None]:
    if not path or not substrings:
        return False, None
    payload = read_json_object(path)
    if payload is None:
        return False, None
    method = str(payload.get("method", ""))
    method_lower = method.lower()
    for substring in substrings:
        if substring.lower() in method_lower:
            return True, method
    return False, method or None


def payload_requires_tuning(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    direct = payload.get("tuning_required") or payload.get("large_deviation_detected")
    if isinstance(direct, bool):
        return direct
    status = str(payload.get("status", "")).lower()
    if status in {"needs_tuning", "large_deviation", "large_deviation_detected"}:
        return True
    for key in ["qc", "quality", "diagnostics", "residuals"]:
        nested = payload.get(key)
        if isinstance(nested, dict) and payload_requires_tuning(nested):
            return True
    return False


def audit_case(case: str, info: dict[str, str], overlay_root: Path) -> list[dict[str, Any]]:
    run_root = Path(info["run_root"])
    object_id = info["object_id"]
    run_case = run_root.name
    rows: list[dict[str, Any]] = []
    for spec in ALGORITHMS:
        data_rel = materialize(spec.get("data"), case, run_case, object_id)
        alt_data_rel = materialize(spec.get("alt_data"), case, run_case, object_id)
        native_overlay_rel = materialize(spec.get("native_overlay"), case, run_case, object_id)
        overlay_rel = materialize(spec.get("overlay"), case, run_case, object_id)
        data_path = run_root / data_rel if data_rel else None
        alt_data_path = run_root / alt_data_rel if alt_data_rel else None
        native_overlay_path = run_root / native_overlay_rel if native_overlay_rel else None
        overlay_path = overlay_root / overlay_rel if overlay_rel else None
        tuning_dir = run_root / "tuning" / str(spec["family"]) / str(spec["id"])
        data_exists_raw = exists(data_path) or exists(alt_data_path)
        reject_paths = [p for p in [data_path, alt_data_path] if exists(p)]
        rejected = False
        rejected_method = None
        for candidate in reject_paths:
            rejected, rejected_method = rejected_by_method(candidate, spec.get("reject_methods_containing"))
            if rejected:
                break
        data_exists = data_exists_raw and not rejected
        overlay_exists = exists(overlay_path) or exists(native_overlay_path)
        if exists(overlay_path):
            qc_path = overlay_path.parent / "qc.json"
        elif exists(native_overlay_path):
            qc_path = native_overlay_path.with_suffix(".qc.json")
        else:
            qc_path = None
        qc_exists = exists(qc_path)
        tuning_records = sorted(str(p) for p in tuning_dir.glob("attempt_*.json")) if tuning_dir.exists() else []
        tuning_required = bool(spec.get("tuning_required", False))
        if not tuning_required:
            for candidate in [data_path, alt_data_path, qc_path]:
                if payload_requires_tuning(read_json_object(candidate)):
                    tuning_required = True
                    break
        if spec.get("deprecated"):
            status = "deprecated_present" if data_exists_raw or overlay_exists else "deprecated_absent"
        elif rejected:
            status = "invalid_data"
        elif data_exists and overlay_exists:
            status = "overlay_ready"
        elif data_exists and not overlay_exists:
            status = "missing_overlay"
        elif not data_exists and overlay_exists:
            status = "overlay_without_current_data"
        else:
            status = "missing_data"
        rows.append(
            {
                "case": case,
                "algorithm_id": spec["id"],
                "family": spec["family"],
                "source": spec.get("source"),
                "status": status,
                "runner_agent": RUNNER_AGENT_ID,
                "runner_policy": RUNNER_POLICY,
                "runner_overlay_command": "native_overlay_copy" if spec.get("native_overlay") else (f"scripts/generate_algorithm_overlay.py --type {spec.get('overlay_type')}" if spec.get("overlay_type") else None),
                "deprecated": bool(spec.get("deprecated", False)),
                "optional": bool(spec.get("optional", False)),
                "required": not bool(spec.get("deprecated", False)) and not bool(spec.get("optional", False)),
                "data_path": str(data_path) if data_path else None,
                "alt_data_path": str(alt_data_path) if alt_data_path else None,
                "native_overlay_path": str(native_overlay_path) if native_overlay_path else None,
                "overlay_path": str(overlay_path) if overlay_path else None,
                "data_exists": bool(data_exists),
                "raw_data_exists": bool(data_exists_raw),
                "invalid_data": bool(rejected),
                "invalid_data_method": rejected_method if rejected else None,
                "overlay_exists": bool(overlay_exists),
                "qc_path": str(qc_path) if qc_path else None,
                "qc_exists": bool(qc_exists),
                "tuning_dir": str(tuning_dir),
                "tuning_required": bool(tuning_required),
                "tuning_record_count": len(tuning_records),
                "tuning_records": tuning_records,
                "blocker": spec.get("blocker"),
                "claim_scope": spec.get("claim_scope"),
                "required_replacement_for": spec.get("required_replacement_for"),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [r for r in rows if not r["deprecated"] and not r["optional"]]
    deprecated_present = [r for r in rows if r["deprecated"] and r["status"] == "deprecated_present"]
    missing_data = [r for r in active if r["status"] in {"missing_data", "invalid_data", "overlay_without_current_data"}]
    missing_overlay = [r for r in active if r["status"] == "missing_overlay"]
    missing_qc = [r for r in active if r["overlay_exists"] and not r.get("qc_exists")]
    missing_tuning = [r for r in active if r.get("tuning_required") and r["tuning_record_count"] == 0]
    return {
        "active_required_count": len(active),
        "overlay_ready_count": sum(1 for r in active if r["status"] == "overlay_ready"),
        "missing_data_count": len(missing_data),
        "missing_overlay_count": len(missing_overlay),
        "deprecated_present_count": len(deprecated_present),
        "missing_qc_count": len(missing_qc),
        "missing_tuning_record_count": len(missing_tuning),
        "missing_data": [{"case": r["case"], "algorithm_id": r["algorithm_id"], "family": r["family"]} for r in missing_data],
        "missing_overlay": [{"case": r["case"], "algorithm_id": r["algorithm_id"], "family": r["family"]} for r in missing_overlay],
        "deprecated_present": [{"case": r["case"], "algorithm_id": r["algorithm_id"], "data_path": r["data_path"], "overlay_path": r["overlay_path"]} for r in deprecated_present],
        "missing_qc": [{"case": r["case"], "algorithm_id": r["algorithm_id"], "family": r["family"], "qc_path": r.get("qc_path")} for r in missing_qc],
        "missing_tuning_records": [{"case": r["case"], "algorithm_id": r["algorithm_id"], "family": r["family"]} for r in missing_tuning],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    overlay_root = Path(args.overlay_root)
    rows: list[dict[str, Any]] = []
    for case, info in RUNS.items():
        rows.extend(audit_case(case, info, overlay_root))
    report = {
        "schema": "v21_atomic_algorithm_overlay_audit.v0",
        "status": "needs_work" if any(r["status"] in {"missing_data", "invalid_data", "missing_overlay", "overlay_without_current_data", "deprecated_present"} and not r["optional"] for r in rows) else "ok",
        "method": "audit_v21_atomic_algorithm_overlays",
        "runner_agent": RUNNER_AGENT_ID,
        "runner_policy": RUNNER_POLICY,
        "overlay_root": str(overlay_root),
        "groundingdino_policy": "disabled_as_v21_default_bbox; historical artifacts are deprecated evidence only",
        "claim_scope": "File-level audit of atomic algorithm data/overlay/tuning coverage. It does not prove visual quality or that missing heavy model runs occurred.",
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "summary": report["summary"]}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, default=Path("outputs/v21_per_algorithm_results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v21_per_algorithm_results/atomic_algorithm_overlay_audit.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
