#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
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

STATUS = "v18_fast_object_motion_state_scaffold"
CLAIM = (
    "This V18 artifact reduces existing visible-surface and material-track evidence into a cheap per-object "
    "motion-state hypothesis. It does not run BundleSDF/NeRF, does not reconstruct hidden geometry, and does "
    "not make complete object-pose claims."
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


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def aggregate_visibility_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for raw_frame in require_list(state.get("frames"), "visibility frames"):
        frame = require_dict(raw_frame, "visibility frame")
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            row = require_dict(raw_obj, "object state row")
            object_id = require_str(row.get("object_id"), "object_id")
            acc = objects.setdefault(
                object_id,
                {
                    "object_id": object_id,
                    "track_id": row.get("track_id"),
                    "name": row.get("name"),
                    "model_physical_state_type": row.get("model_physical_state_type", "unknown"),
                    "physical_state_source": row.get("physical_state_source"),
                    "physical_notes": row.get("physical_notes"),
                    "requires_part_or_relative_motion_model": row.get("requires_part_or_relative_motion_model"),
                    "part_or_relative_motion_evidence_terms": row.get("part_or_relative_motion_evidence_terms"),
                    "secondary_deformable_or_surface_component": row.get("secondary_deformable_or_surface_component"),
                    "optical_difficulty": row.get("optical_difficulty"),
                    "surface_change_without_pose_state": row.get("surface_change_without_pose_state"),
                    "physical_state_schema_confidence": row.get("physical_state_schema_confidence"),
                    "legacy_keyword_physical_state_type": row.get("legacy_keyword_physical_state_type"),
                    "visibility_counts": Counter(),
                    "geometry_scope_counts": Counter(),
                    "active_frames": 0,
                    "visible_frames": 0,
                    "visible_surface_depth_backed_frames": 0,
                    "visible_mask_surface_rejected_frames": 0,
                },
            )
            visibility = require_str(row.get("visibility_state"), "visibility_state")
            geometry_scope = require_str(row.get("geometry_scope"), "geometry_scope")
            acc["visibility_counts"][visibility] += 1
            acc["geometry_scope_counts"][geometry_scope] += 1
            if visibility != "out_of_frame":
                acc["active_frames"] += 1
            if visibility == "visible":
                acc["visible_frames"] += 1
            if geometry_scope == "visible_surface_depth_backed":
                acc["visible_surface_depth_backed_frames"] += 1
            if geometry_scope == "visible_mask_only_surface_rejected":
                acc["visible_mask_surface_rejected_frames"] += 1
    for acc in objects.values():
        acc["visibility_counts"] = dict(sorted(acc["visibility_counts"].items()))
        acc["geometry_scope_counts"] = dict(sorted(acc["geometry_scope_counts"].items()))
    return objects


def material_windows_by_object(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in require_list(report.get("windows"), "motion windows"):
        row = require_dict(raw, "motion window")
        out[require_str(row.get("object_id"), "motion object_id")].append(row)
    return out


def replay_candidates_by_object(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in require_list(report.get("candidates"), "replay candidates"):
        row = require_dict(raw, "replay candidate")
        out[require_str(row.get("object_id"), "replay object_id")].append(row)
    return out


def classify_motion_state(physical: str, visible_surface_frames: int, windows: list[dict[str, Any]], replay: list[dict[str, Any]]) -> tuple[str, str]:
    persistent = any(row.get("persistent_window_motion_candidate") is True for row in windows)
    local = any(row.get("local_adjacent_material_motion") is True for row in windows)
    ready_replay = any(row.get("partial_visible_surface_replay_candidate") is True for row in replay)
    if persistent and ready_replay:
        return (
            "partial_rigid_visible_surface_motion_supported",
            "Persistent material motion and visible-surface replay support a partial visible-surface rigid state; hidden geometry and full-timeline pose remain unresolved.",
        )
    if physical == "deformable":
        if local or visible_surface_frames > 0:
            return (
                "deformable_visible_surface_motion_or_surface_only",
                "Model physical state is deformable; keep per-frame visible/deformable surface state rather than forcing SE(3).",
            )
        return ("deformable_motion_unresolved_no_surface", "Model physical state is deformable but current visible-surface/motion evidence is absent.")
    if physical == "articulated":
        if local or visible_surface_frames > 0:
            return (
                "articulated_visible_surface_motion_unresolved",
                "Model physical state is articulated; local surfaces may move but no single-object rigid pose is accepted.",
            )
        return ("articulated_motion_unresolved_no_surface", "Articulated object has no usable fast surface/motion evidence yet.")
    if local:
        return (
            "local_rigid_motion_only_not_pose",
            "Adjacent material motion exists but does not compose into a persistent full-window pose.",
        )
    if visible_surface_frames > 0:
        return (
            "visible_surface_only_motion_unresolved",
            "Depth-backed visible surfaces exist but cheap material motion does not yet support rigid/deformable temporal state.",
        )
    return ("motion_unresolved_no_surface", "No visible-surface or material-motion support in the current fast evidence.")


def object_report_row(object_id: str, acc: dict[str, Any], windows: list[dict[str, Any]], replay: list[dict[str, Any]]) -> dict[str, Any]:
    physical = str(acc.get("model_physical_state_type", "unknown"))
    visible_surface_frames = require_int(acc.get("visible_surface_depth_backed_frames"), "visible surface frames")
    state, reason = classify_motion_state(physical, visible_surface_frames, windows, replay)
    rigid_ready_pairs = sum(require_int(row.get("rigid_factor_ready_pairs", 0), "rigid ready pairs") for row in windows)
    local_window_count = sum(1 for row in windows if row.get("local_adjacent_material_motion") is True)
    persistent_window_count = sum(1 for row in windows if row.get("persistent_window_motion_candidate") is True)
    ready_replay_count = sum(1 for row in replay if row.get("partial_visible_surface_replay_candidate") is True)
    return {
        "object_id": object_id,
        "track_id": acc.get("track_id"),
        "name": acc.get("name"),
        "model_physical_state_type": physical,
        "physical_state_source": acc.get("physical_state_source"),
        "physical_notes": acc.get("physical_notes"),
        "requires_part_or_relative_motion_model": bool(acc.get("requires_part_or_relative_motion_model")),
        "part_or_relative_motion_evidence_terms": acc.get("part_or_relative_motion_evidence_terms", []),
        "secondary_deformable_or_surface_component": bool(acc.get("secondary_deformable_or_surface_component")),
        "optical_difficulty": bool(acc.get("optical_difficulty")),
        "surface_change_without_pose_state": bool(acc.get("surface_change_without_pose_state")),
        "physical_state_schema_confidence": acc.get("physical_state_schema_confidence"),
        "legacy_keyword_physical_state_type": acc.get("legacy_keyword_physical_state_type"),
        "fast_motion_state": state,
        "fast_motion_state_reason": reason,
        "active_frames": acc.get("active_frames"),
        "visible_frames": acc.get("visible_frames"),
        "visible_surface_depth_backed_frames": visible_surface_frames,
        "visible_mask_surface_rejected_frames": acc.get("visible_mask_surface_rejected_frames"),
        "visibility_counts": acc.get("visibility_counts"),
        "geometry_scope_counts": acc.get("geometry_scope_counts"),
        "material_track_window_count": len(windows),
        "rigid_factor_ready_pair_count": rigid_ready_pairs,
        "local_adjacent_motion_window_count": local_window_count,
        "persistent_motion_window_count": persistent_window_count,
        "partial_visible_surface_replay_ready_count": ready_replay_count,
        "window_ids": [row.get("window_id") for row in windows],
        "ready_replay_candidate_ids": [row.get("candidate_id") for row in replay if row.get("partial_visible_surface_replay_candidate") is True],
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "pose_claim": "fast_motion_state_only_not_complete_object_pose",
    }


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    visibility_path = existing(args.visibility_root / case / "v18_visibility_occlusion_state.json", f"{case} V18 visibility state")
    motion_path = existing(args.material_motion_root / case / "v17_object_material_motion_state_report.json", f"{case} V17 material motion report")
    replay_path = existing(args.surface_replay_root / case / "v17_object_material_surface_replay_report.json", f"{case} V17 surface replay report")
    visibility = require_dict(load_json(visibility_path), f"{case} visibility state")
    motion = require_dict(load_json(motion_path), f"{case} material motion report")
    replay = require_dict(load_json(replay_path), f"{case} surface replay report")
    objects = aggregate_visibility_state(visibility)
    windows_by_object = material_windows_by_object(motion)
    replay_by_object = replay_candidates_by_object(replay)
    rows = [
        object_report_row(object_id, acc, windows_by_object.get(object_id, []), replay_by_object.get(object_id, []))
        for object_id, acc in sorted(objects.items())
    ]
    state_counts = Counter(require_str(row.get("fast_motion_state"), "fast motion state") for row in rows)
    physical_counts = Counter(require_str(row.get("model_physical_state_type"), "physical state") for row in rows)
    report = {
        "method": "build_v18_fast_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_visibility_occlusion_state": str(visibility_path),
            "v17_material_motion_state": str(motion_path),
            "v17_surface_replay": str(replay_path),
        },
        "frame_count": require_int(visibility.get("frame_count"), "visibility frame_count"),
        "object_count": len(rows),
        "fast_motion_state_counts": dict(sorted(state_counts.items())),
        "model_physical_state_type_counts": dict(sorted(physical_counts.items())),
        "object_rows": rows,
        "default_path_uses_bundlesdf_or_nerf": False,
        "object_geometry_policy": "visible_surface_and_fast_motion_only_not_hidden_geometry_completion",
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_fast_motion_state_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    total_counts: Counter[str] = Counter()
    physical_counts: Counter[str] = Counter()
    for report in reports:
        total_counts.update(report["fast_motion_state_counts"])
        physical_counts.update(report["model_physical_state_type_counts"])
    summary = {
        "method": "build_v18_fast_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "fast_motion_state_counts": dict(sorted(total_counts.items())),
        "model_physical_state_type_counts": dict(sorted(physical_counts.items())),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(args.output_root / require_str(report.get("case"), "case") / "v18_fast_motion_state_report.json"),
                "object_count": report["object_count"],
                "fast_motion_state_counts": report["fast_motion_state_counts"],
                "model_physical_state_type_counts": report["model_physical_state_type_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        "object_geometry_policy": "visible_surface_and_fast_motion_only_not_hidden_geometry_completion",
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_fast_motion_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--material-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_object_material_motion_state"))
    parser.add_argument("--surface-replay-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_fast_motion_state"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
