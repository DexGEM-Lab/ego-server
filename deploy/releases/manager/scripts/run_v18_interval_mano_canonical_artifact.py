#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build a canonical two-case V18 interval-MANO artifact.

This integrates the current compact-rigid object, temporal MANO uncertainty,
side-specific articulated MANO hypotheses/falsification, observed-surface
constraint provenance, and hidden-volume quarantine into one renderable two-case
output root. It does not accept coordinate corrections; it preserves the interval
hand-state evidence as the current physical V18 MANO state.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


CASE_CONFIGS: dict[str, dict[str, str]] = {
    "task5_tomato_960": {
        "object_id": "object:obj_tomato",
        "object_label": "canonical rigid tomato interval MANO left replay",
        "annotations": "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/task5_tomato_960/annotations_v18_full.json",
        "pose_report": "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/pose_fit_frame929prior_frame806scale_v1_from_tracked/v18_compact_rigid_object_pose_fit_report.json",
        "completed_mesh": "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/completed_mesh_frame929prior_frame806scale_v1/object_obj_tomato_scale_sane_completed_mesh_labeled.ply",
        "constraint_report": "/data2/ego_annotation_outputs/v18_scale_sane_tomato_completion_v1/task5_tomato_960/object_obj_tomato/scale_sane_full_bridge_initial_measure_from_tracked/v18_mano_object_constraint_state_full_bridge.json",
        "temporal_mano_state": "/data2/ego_annotation_outputs/v18_task5_tomato_temporal_mano_articulated_leftreplay_v1/task5_tomato_960/v18_temporal_mano_articulated_interval_state.json",
        "observed_surface_state": "/data2/ego_annotation_outputs/v18_task5_observed_surface_mano_constraints_leftreplay_v1/task5_tomato_960/v18_observed_surface_mano_constraint_state.json",
        "hidden_volume_validation": "/data2/ego_annotation_outputs/v18_compact_rigid_hidden_volume_depth_validation_v1/task5_tomato_960/object_obj_tomato/v18_compact_rigid_hidden_volume_depth_validation.json",
    },
    "trash_1050": {
        "object_id": "object:pink_lid_trash_can_second",
        "object_label": "canonical rigid pink lid interval MANO left replay",
        "annotations": "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/trash_1050/annotations_v18_full.json",
        "pose_report": "/data2/ego_annotation_outputs/v18_compact_rigid_completion_next_frame872/trash_1050/object_pink_lid_trash_can_second/pose_fit_seed42_v3/v18_compact_rigid_object_pose_fit_report.json",
        "completed_mesh": "/data2/ego_annotation_outputs/v18_compact_rigid_completion_next_frame872/trash_1050/object_pink_lid_trash_can_second/completed_mesh_seed42_v3/object_pink_lid_trash_can_second_compact_rigid_completed_mesh_labeled.ply",
        "constraint_report": "/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/trash_1050/object_pink_lid_trash_can_second/frame872_full_bridge_all_signed/initial_measure/v18_mano_object_constraint_state_full_bridge.json",
        "temporal_mano_state": "/data2/ego_annotation_outputs/v18_trash_lid_temporal_mano_articulated_leftreplay_v1/trash_1050/v18_temporal_mano_articulated_interval_state.json",
        "observed_surface_state": "/data2/ego_annotation_outputs/v18_trash_observed_surface_mano_constraints_leftreplay_v1/trash_1050/v18_observed_surface_mano_constraint_state.json",
        "hidden_volume_validation": "/data2/ego_annotation_outputs/v18_compact_rigid_hidden_volume_depth_validation_v1/trash_1050/object_pink_lid_trash_can_second/v18_compact_rigid_hidden_volume_depth_validation.json",
    },
}


def temporal_map(state: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in state.get("per_frame_states", []) if isinstance(state.get("per_frame_states"), list) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def hidden_map(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in state.get("frame_rows", []) if isinstance(state.get("frame_rows"), list) else []:
        if isinstance(row, dict):
            out[int(row["frame_idx"])] = row
    return out


def observed_surface_map(state: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in state.get("per_frame_states", []) if isinstance(state.get("per_frame_states"), list) else []:
        if isinstance(row, dict):
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def compact_temporal_state(row: dict[str, Any] | None, temporal_path: Path) -> dict[str, Any]:
    if row is None:
        return {
            "state": "no_interval_mano_state_for_frame_hand",
            "coordinate_correction_accepted": False,
            "source_temporal_mano_state": str(temporal_path),
        }
    keep = [
        "frame_idx",
        "hand_side",
        "interval_id",
        "temporal_mano_state",
        "coordinate_correction_accepted",
        "mano_parameterization",
        "raw_replay_vertex_error_m",
        "raw_to_current_similarity_error_m",
        "residual_penetration_after_articulated_mano_m",
        "visible_joint_shift_px",
        "joint_camera_depth_shift_m",
        "pose_delta_norm_rad",
        "hidden_volume_state",
        "reason",
    ]
    out = {key: deepcopy(row[key]) for key in keep if key in row}
    if "optimized_joints_world_m" in row:
        out["optimized_joints_world_m"] = deepcopy(row["optimized_joints_world_m"])
    if "optimized_vertices_world_sample_m" in row:
        out["optimized_vertices_world_sample_m"] = deepcopy(row["optimized_vertices_world_sample_m"])
        out["optimized_vertices_sample_ids"] = deepcopy(row.get("optimized_vertices_sample_ids"))
    out["source_temporal_mano_state"] = str(temporal_path)
    return out


def compact_hidden_state(row: dict[str, Any] | None, hidden_path: Path) -> dict[str, Any]:
    if row is None:
        return {"state": "hidden_volume_unmeasured", "source_hidden_volume_validation": str(hidden_path)}
    keep = [
        "frame_idx",
        "state",
        "free_space_conflict_fraction_projected",
        "observed_support_fraction_projected",
        "projected_vertex_count",
        "missing_depth_count",
    ]
    out = {key: deepcopy(row[key]) for key in keep if key in row}
    out["source_hidden_volume_validation"] = str(hidden_path)
    return out


def compact_observed_surface_state(row: dict[str, Any] | None, observed_path: Path | None) -> dict[str, Any]:
    if row is None or observed_path is None:
        return {
            "state": "observed_surface_mano_state_unavailable",
            "coordinate_correction_accepted": False,
            "source_observed_surface_mano_state": str(observed_path) if observed_path is not None else None,
        }
    keep = [
        "frame_idx",
        "hand_side",
        "temporal_mano_state_input",
        "observed_surface_mano_state",
        "coordinate_correction_accepted",
        "blocking_mechanisms",
        "hidden_volume_state_input",
        "candidate_reconstruction",
    ]
    out = {key: deepcopy(row[key]) for key in keep if key in row}
    candidate = row.get("candidate_full_778_measurement") if isinstance(row.get("candidate_full_778_measurement"), dict) else None
    if candidate is not None:
        out["candidate_full_778_measurement"] = {
            "hand_vertex_count": candidate.get("hand_vertex_count"),
            "penetrating_vertex_count": candidate.get("penetrating_vertex_count"),
            "max_penetration_m": candidate.get("max_penetration_m"),
            "observed_supported_strict_penetration_m": deepcopy(candidate.get("observed_supported_strict_penetration_m")),
            "free_space_conflict_penetration_m": deepcopy(candidate.get("free_space_conflict_penetration_m")),
            "hidden_or_unvalidated_penetration_m": deepcopy(candidate.get("hidden_or_unvalidated_penetration_m")),
        }
    out["source_observed_surface_mano_state"] = str(observed_path)
    return out


def build_case_annotations(case: str, cfg: dict[str, str], output_root: Path) -> dict[str, Any]:
    annotations_path = Path(cfg["annotations"])
    temporal_path = Path(cfg["temporal_mano_state"])
    observed_path = Path(cfg["observed_surface_state"]) if cfg.get("observed_surface_state") else None
    hidden_path = Path(cfg["hidden_volume_validation"])
    annotations = load_json(annotations_path)
    temporal_state = load_json(temporal_path)
    observed_state = load_json(observed_path) if observed_path is not None else None
    hidden_state = load_json(hidden_path)
    tmap = temporal_map(temporal_state)
    omap = observed_surface_map(observed_state) if isinstance(observed_state, dict) else {}
    hmap = hidden_map(hidden_state)
    out = deepcopy(annotations)
    out["v18_interval_mano_canonical_state"] = {
        "case": case,
        "object_id": cfg["object_id"],
        "coordinate_correction_accepted": False,
        "temporal_mano_state": str(temporal_path),
        "temporal_mano_summary": temporal_state.get("summary"),
        "hidden_volume_validation": str(hidden_path),
        "hidden_volume_validation_summary": hidden_state.get("summary"),
        "observed_surface_mano_state": str(observed_path) if observed_path is not None else None,
        "observed_surface_mano_summary": observed_state.get("summary") if isinstance(observed_state, dict) else None,
        "claim_scope": (
            "Canonical V18 interval-MANO state: compact-rigid object constraints, side-specific articulated MANO hypotheses, "
            "observed-surface provenance, and hidden-volume quarantine are carried as bounded/falsified uncertainty. "
            "No coordinate-level MANO correction is accepted."
        ),
    }
    attached = 0
    for frame in out.get("frames", []) if isinstance(out.get("frames"), list) else []:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", 0))
        frame["v18_hidden_volume_constraint_state"] = compact_hidden_state(hmap.get(frame_idx), hidden_path)
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            state = compact_temporal_state(tmap.get((frame_idx, side)), temporal_path)
            observed = compact_observed_surface_state(omap.get((frame_idx, side)), observed_path)
            hand["v18_interval_mano_state"] = state
            hand["v18_observed_surface_mano_constraint_state"] = observed
            metric = hand.get("metric_mano_state")
            if isinstance(metric, dict):
                metric["v18_interval_mano_state"] = deepcopy(state)
                metric["v18_observed_surface_mano_constraint_state"] = deepcopy(observed)
                metric["coordinate_correction_accepted"] = False
            attached += 1
    case_dir = output_root / case
    write_json(case_dir / "annotations_v18_interval_mano_canonical.json", out)
    return {
        "case": case,
        "annotations": str(case_dir / "annotations_v18_interval_mano_canonical.json"),
        "hand_states_attached": attached,
        "temporal_mano_summary": temporal_state.get("summary"),
        "hidden_volume_summary": hidden_state.get("summary"),
        "observed_surface_mano_summary": observed_state.get("summary") if isinstance(observed_state, dict) else None,
    }


def ffprobe_count(path: Path) -> int | None:
    try:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
        return int(raw)
    except Exception:
        return None


def render_case(case: str, cfg: dict[str, str], annotations_path: Path, output_root: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py",
        "--case",
        case,
        "--object-label",
        cfg["object_label"],
        "--annotations",
        str(annotations_path),
        "--pose-report",
        cfg["pose_report"],
        "--completed-mesh",
        cfg["completed_mesh"],
        "--constraint-report",
        cfg["constraint_report"],
        "--temporal-mano-state",
        cfg["temporal_mano_state"],
        "--hidden-volume-validation",
        cfg["hidden_volume_validation"],
        "--output-root",
        str(output_root),
        "--world-view",
        "local",
    ]
    subprocess.run(cmd, check=True)
    safe_label = cfg["object_label"].replace(" ", "_").replace(":", "_")
    case_dir = output_root / case
    overlay = case_dir / f"v18_overlay_{safe_label}.mp4"
    world = case_dir / f"v18_world_{safe_label}.mp4"
    side = case_dir / f"v18_side_by_side_{safe_label}.mp4"
    return {
        "case": case,
        "overlay": str(overlay),
        "world": str(world),
        "side_by_side": str(side),
        "overlay_frames": ffprobe_count(overlay),
        "world_frames": ffprobe_count(world),
        "side_by_side_frames": ffprobe_count(side),
        "render_manifest": str(case_dir / "v18_temporal_rigid_object_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_interval_mano_canonical_artifact_leftreplay_v1"))
    ap.add_argument("--skip-render", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    case_reports: list[dict[str, Any]] = []
    render_reports: list[dict[str, Any]] = []
    for case, cfg in CASE_CONFIGS.items():
        case_report = build_case_annotations(case, cfg, output_root)
        case_reports.append(case_report)
        if not args.skip_render:
            render_reports.append(render_case(case, cfg, Path(case_report["annotations"]), output_root))
    manifest = {
        "method": "run_v18_interval_mano_canonical_artifact",
        "status": "ok",
        "output_root": str(output_root),
        "coordinate_correction_accepted": False,
        "case_reports": case_reports,
        "render_reports": render_reports,
        "physical_claim": (
            "This canonical artifact carries the current best V18 MANO state as interval-level uncertainty/falsification. "
            "Both hands use side-specific HaWoR MANO replay where reproducible; left replay uses MANO_LEFT with the documented "
            "HaWoR shapedirs-x fix. Observed-surface MANO provenance and hidden-volume quarantine are explicit. "
            "It is not accepted coordinate-level MANO correction."
        ),
        "visual_inspection_required": True,
    }
    write_json(output_root / "v18_interval_mano_canonical_artifact_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
