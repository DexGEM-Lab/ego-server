#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

STATUS = "v18_physical_state_schema"
CLAIM = (
    "This artifact consumes structured VLM physical-model fields for each object: primary whole-object state, "
    "pose-model eligibility, part/relative-motion requirement, deformable/optical secondary evidence, and surface "
    "appearance uncertainty. It does not parse free-text notes to decide physical model type, and it does not infer "
    "geometry, pose, articulation parameters, or contact ownership."
)

PRIMARY_STATES = {"rigid", "deformable", "articulated", "unknown", "unknown_optically_difficult"}
RIGID_TERMS = (
    "rigid",
    "does not deform",
    "does not change shape",
    "no object shape change",
)
DEFORMABLE_TERMS = (
    "deformable",
    "non-rigid",
    "nonrigid",
    "flexible",
    "changes shape",
    "curled",
    "folded",
    "crumpled",
    "bunched",
    "draped",
)
PRIMARY_ARTICULATION_TERMS = ("articulated", "hinged", "hinge", "lever")
PART_OR_RELATIVE_MOTION_TERMS = (
    "articulated",
    "hinged",
    "hinge",
    "lever",
    "relative motion",
    "moves relative",
    "relative to",
    "position changes",
    "changes position",
    "opens",
    "closes",
)
OPTICAL_TERMS = ("transparent", "translucent", "reflective", "reflections", "clear plastic")
SURFACE_CHANGE_TERMS = ("surface appearance changes", "skin is removed", "peeled", "detached", "surface texture")


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


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def source_from_measurement_manifest(manifest: dict[str, Any], key: str) -> Path:
    raw = manifest.get(key)
    if isinstance(raw, str):
        return Path(raw)
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, str):
            return Path(first)
        if isinstance(first, dict) and isinstance(first.get("path"), str):
            return Path(first["path"])
    raise RuntimeError(f"measurement manifest has no usable source for {key}")


def matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def structured_from_vlm_physical_model(row: dict[str, Any]) -> dict[str, Any]:
    primary = require_str(row.get("primary_physical_model"), "primary physical model")
    if primary not in PRIMARY_STATES:
        raise RuntimeError(f"unexpected primary physical model: {primary}")
    confidence_raw = row.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
    pose_allowed = bool(row.get("pose_model_allowed") is True)
    blockers: list[str] = []
    if primary == "unknown":
        blockers.append("structured_vlm_primary_physical_model_unknown")
    if not pose_allowed:
        blockers.append("structured_vlm_pose_model_not_allowed")
    return {
        "model_physical_state_type": primary,
        "physical_state_source": "structured_vlm_physical_model_fields_v1",
        "pose_model_allowed_by_structured_vlm": pose_allowed,
        "surface_appearance_changes": bool(row.get("surface_appearance_changes") is True),
        "geometry_changes": row.get("geometry_changes"),
        "requires_part_or_relative_motion_model": bool(row.get("requires_part_or_relative_motion_model") is True),
        "primary_articulation_evidence_terms": [row.get("evidence")] if row.get("requires_part_or_relative_motion_model") is True else [],
        "part_or_relative_motion_evidence_terms": [row.get("evidence")] if row.get("requires_part_or_relative_motion_model") is True else [],
        "secondary_deformable_or_surface_component": bool(row.get("secondary_deformable_or_surface_component") is True),
        "secondary_deformable_evidence_terms": [row.get("evidence")] if row.get("secondary_deformable_or_surface_component") is True else [],
        "optical_difficulty": bool(row.get("optical_difficulty") is True),
        "optical_evidence_terms": [row.get("evidence")] if row.get("optical_difficulty") is True else [],
        "surface_change_without_pose_state": bool(row.get("surface_appearance_changes") is True and not pose_allowed),
        "surface_change_evidence_terms": [row.get("evidence")] if row.get("surface_appearance_changes") is True else [],
        "structured_vlm_confidence": confidence,
        "structured_vlm_evidence": row.get("evidence"),
        "structured_vlm_uncertainty": row.get("uncertainty"),
        "schema_confidence": "high" if confidence >= 0.8 and not blockers else "medium" if confidence >= 0.5 else "low",
        "schema_blockers": blockers,
    }


def timeline_objects(timeline: dict[str, Any], roster_by_object_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for raw in require_list(timeline.get("objects"), "timeline objects"):
        row = require_dict(raw, "timeline object")
        object_id = require_str(row.get("object_id"), "object_id")
        roster_row = roster_by_object_id.get(object_id, {})
        merged = {**row}
        for key in ("physical_notes", "role_status", "source"):
            if key not in merged or merged.get(key) is None:
                merged[key] = roster_row.get(key)
        objects.append(merged)
    return objects


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    measurement_manifest_path = existing(args.measurement_store_root / case / "v17_measurement_manifest.json", f"{case} measurement manifest")
    measurement_manifest = require_dict(load_json(measurement_manifest_path), f"{case} measurement manifest")
    timeline_path = existing(args.multi_object_timeline_root / case / "v17_multi_object_timeline.json", f"{case} object timeline")
    timeline = require_dict(load_json(timeline_path), f"{case} object timeline")
    roster_path = existing(Path(require_str(measurement_manifest.get("object_roster"), f"{case} object_roster")), f"{case} object roster")
    structured_model_path = existing(args.structured_physical_model_root / case / "v18_structured_physical_model.json", f"{case} structured physical model")
    structured_model = require_dict(load_json(structured_model_path), f"{case} structured physical model")
    structured_by_track_id = {require_str(row.get("track_id"), "structured model track_id"): require_dict(row, "structured model row") for row in require_list(structured_model.get("objects"), "structured model objects")}
    roster_rows = [require_dict(row, "object roster row") for row in require_list(load_json(roster_path), f"{case} object roster")]
    roster_by_object_id = {require_str(row.get("object_id"), "roster object_id"): row for row in roster_rows}
    object_rows: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    part_motion_count = 0
    secondary_deformable_count = 0
    optical_count = 0
    surface_change_count = 0
    changed_from_legacy: list[dict[str, Any]] = []
    for obj in timeline_objects(timeline, roster_by_object_id):
        notes = obj.get("physical_notes")
        track_id = require_str(obj.get("track_id"), "timeline object track_id")
        if track_id not in structured_by_track_id:
            raise RuntimeError(f"{case}: missing structured VLM physical model for track_id {track_id}")
        structured = structured_from_vlm_physical_model(structured_by_track_id[track_id])
        primary = require_str(structured.get("model_physical_state_type"), "model physical state")
        primary_counts[primary] += 1
        legacy = "not_used_structured_vlm_required"
        legacy_counts[legacy] += 1
        if bool(structured.get("requires_part_or_relative_motion_model")):
            part_motion_count += 1
        if bool(structured.get("secondary_deformable_or_surface_component")):
            secondary_deformable_count += 1
        if bool(structured.get("optical_difficulty")):
            optical_count += 1
        if bool(structured.get("surface_change_without_pose_state")):
            surface_change_count += 1
        row = {
            "object_id": obj.get("object_id"),
            "track_id": obj.get("track_id"),
            "name": obj.get("name"),
            "physical_notes": notes,
            "role_status": obj.get("role_status"),
            "roster_source": obj.get("source"),
            "legacy_keyword_physical_state_type": legacy,
            "structured_physical_model_source": str(structured_model_path),
            **structured,
            "structured_schema_ready": True,
            "part_pose_ready": False,
            "object_pose_requirement_met": False,
        }
        if legacy != primary:
            changed_from_legacy.append(
                {
                    "object_id": obj.get("object_id"),
                    "legacy_keyword_physical_state_type": legacy,
                    "model_physical_state_type": primary,
                    "schema_reason": structured.get("schema_blockers"),
                }
            )
        object_rows.append(row)
    report = {
        "method": "build_v18_physical_state_schema",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v17_measurement_manifest": str(measurement_manifest_path),
            "v17_multi_object_timeline": str(timeline_path),
            "object_roster": str(roster_path),
            "structured_physical_model": str(structured_model_path),
        },
        "object_count": len(object_rows),
        "model_physical_state_type_counts": dict(sorted(primary_counts.items())),
        "legacy_keyword_physical_state_type_counts": dict(sorted(legacy_counts.items())),
        "part_or_relative_motion_required_count": part_motion_count,
        "secondary_deformable_or_surface_component_count": secondary_deformable_count,
        "optical_difficulty_count": optical_count,
        "surface_change_without_pose_state_count": surface_change_count,
        "changed_from_legacy_keyword_count": len(changed_from_legacy),
        "changed_from_legacy_keyword_rows": changed_from_legacy,
        "object_rows": object_rows,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_physical_state_schema_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    primary_counts: Counter[str] = Counter()
    legacy_counts: Counter[str] = Counter()
    for report in reports:
        primary_counts.update(report["model_physical_state_type_counts"])
        legacy_counts.update(report["legacy_keyword_physical_state_type_counts"])
    summary = {
        "method": "build_v18_physical_state_schema",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count": sum(int(report["object_count"]) for report in reports),
        "model_physical_state_type_counts": dict(sorted(primary_counts.items())),
        "legacy_keyword_physical_state_type_counts": dict(sorted(legacy_counts.items())),
        "part_or_relative_motion_required_count": sum(int(report["part_or_relative_motion_required_count"]) for report in reports),
        "secondary_deformable_or_surface_component_count": sum(int(report["secondary_deformable_or_surface_component_count"]) for report in reports),
        "optical_difficulty_count": sum(int(report["optical_difficulty_count"]) for report in reports),
        "surface_change_without_pose_state_count": sum(int(report["surface_change_without_pose_state_count"]) for report in reports),
        "changed_from_legacy_keyword_count": sum(int(report["changed_from_legacy_keyword_count"]) for report in reports),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_physical_state_schema_report.json"),
                "object_count": report["object_count"],
                "model_physical_state_type_counts": report["model_physical_state_type_counts"],
                "changed_from_legacy_keyword_count": report["changed_from_legacy_keyword_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_physical_state_schema_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--multi-object-timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--structured-physical-model-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_structured_physical_model"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
