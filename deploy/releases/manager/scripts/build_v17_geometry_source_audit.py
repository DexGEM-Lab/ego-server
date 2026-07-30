#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


STATUS = "v17_geometry_source_audit_qc"
CLAIM = (
    "This artifact audits which geometry source owns each current V17 object and contact claim. "
    "It does not solve object pose. It exposes whether legacy contact factors, local contact patches, "
    "multi-object visible RGBD surfaces, and partial material-pose replay candidates can be interpreted "
    "as one object-geometry state."
)

CONTACT_MEASUREMENT_FILES = (
    "contact_measurements.json",
    "hand_repair_contact_measurements.json",
    "local_contact_patch_contact_measurements.json",
    "object_depth_repair_contact_measurements.json",
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
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def optional_finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def summarize_numbers(values: list[float]) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": finite[0],
        "median": finite[len(finite) // 2],
        "max": finite[-1],
    }


def normalize_object_label(value: Any) -> str | None:
    if value is None:
        return None
    label = require_str(value, "object label")
    if label.startswith("object:"):
        return label
    return None


def contact_measurement_index(measurement_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    for name in CONTACT_MEASUREMENT_FILES:
        path = existing_path(measurement_dir / name, name)
        rows = require_list(load_json(path), str(path))
        source_counts[name] = len(rows)
        for i, raw in enumerate(rows):
            row = require_dict(raw, f"{path}[{i}]")
            measurement_id = require_str(row.get("measurement_id"), f"{path}[{i}].measurement_id")
            by_id.setdefault(measurement_id, []).append(row)
    return by_id, source_counts


def local_patch_states(path: Path) -> list[dict[str, Any]]:
    rows = require_list(load_json(existing_path(path, "local contact patch states")), str(path))
    return [require_dict(row, f"local patch state {i}") for i, row in enumerate(rows)]


def object_visible_surface_counts(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("object_summaries"), "visible surface object_summaries")):
        row = require_dict(raw, f"visible surface object_summaries[{i}]")
        object_id = require_str(row.get("object_id"), f"visible surface object_summaries[{i}].object_id")
        out[object_id] = {
            "surface_frame_count": require_int(row.get("surface_frame_count"), f"{object_id} surface_frame_count"),
            "rejected_frame_count": require_int(row.get("rejected_frame_count"), f"{object_id} rejected_frame_count"),
            "surface_vertices": require_int(row.get("surface_vertices"), f"{object_id} surface_vertices"),
            "surface_faces": require_int(row.get("surface_faces"), f"{object_id} surface_faces"),
            "object_geometry_complete": bool(row.get("object_geometry_complete") is True),
            "object_pose_requirement_met": bool(row.get("object_pose_requirement_met") is True),
            "annotation_ready": bool(row.get("annotation_ready") is True),
        }
    return out


def multi_contact_indexes(report: dict[str, Any]) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], dict[tuple[int, str, str], dict[str, Any]]]:
    by_frame_side: dict[tuple[int, str], list[dict[str, Any]]] = {}
    by_object_side: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "multi-object contact rows")):
        row = require_dict(raw, f"multi-object contact rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"multi-object contact rows[{i}].frame_idx")
        side = require_str(row.get("hand_side"), f"multi-object contact rows[{i}].hand_side")
        object_id = require_str(row.get("object_id"), f"multi-object contact rows[{i}].object_id")
        by_frame_side.setdefault((frame_idx, side), []).append(row)
        key = (frame_idx, object_id, side)
        if key in by_object_side:
            raise RuntimeError(f"duplicate multi-object contact row: {key}")
        by_object_side[key] = row
    return by_frame_side, by_object_side


def unique_field(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if len(unique) == 0:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def multi_contact_distance_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"available": False}
    return {
        "available": True,
        "contact_mode_state": row.get("contact_mode_state"),
        "visible_surface_distance_candidate": bool(row.get("visible_surface_distance_candidate") is True),
        "min_symmetric_distance_m": optional_finite_number(
            row.get("min_symmetric_distance_m"), "multi-object min_symmetric_distance_m"
        ),
        "missing_geometry": require_list(row.get("missing_geometry"), "multi-object missing_geometry"),
    }


def selected_measurement_rows(
    ready_rows: list[dict[str, Any]],
    by_measurement_id: dict[str, list[dict[str, Any]]],
    multi_by_frame_side: dict[tuple[int, str], list[dict[str, Any]]],
    multi_by_object_side: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    audited: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in ready_rows:
        frame_idx = require_int(row.get("frame_idx"), "contact-mode ready row frame_idx")
        side = require_str(row.get("side"), "contact-mode ready row side")
        selected_id = row.get("selected_measurement_id")
        same_frame_side = multi_by_frame_side.get((frame_idx, side))
        if same_frame_side is None:
            same_frame_side = []
        same_frame_measured = [
            multi_row for multi_row in same_frame_side if multi_row.get("contact_mode_state") == "measured_distance_evidence"
        ]
        same_frame_candidate = [
            multi_row for multi_row in same_frame_side if multi_row.get("visible_surface_distance_candidate") is True
        ]
        base = {
            "frame_idx": frame_idx,
            "side": side,
            "selected_measurement_id": selected_id,
            "same_frame_side_multi_object_rows": len(same_frame_side),
            "same_frame_side_measured_multi_object_rows": len(same_frame_measured),
            "same_frame_side_visible_surface_distance_candidate_rows": len(same_frame_candidate),
            "gap_min_m": optional_finite_number(row.get("gap_min_m"), "contact-mode gap_min_m"),
            "gap_p05_m": optional_finite_number(row.get("gap_p05_m"), "contact-mode gap_p05_m"),
        }
        if selected_id is None:
            counts["temporal_or_unanchored_contact_mode_rows"] += 1
            audited.append(
                {
                    **base,
                    "geometry_source": "legacy_single_object_contact_mode_inference",
                    "explicit_multi_object_id": None,
                    "object_identity_status": "no_selected_measurement_id",
                    "multi_object_visible_surface_distance": {"available": False},
                }
            )
            continue
        selected_id_str = require_str(selected_id, "selected_measurement_id")
        candidates = by_measurement_id.get(selected_id_str)
        if candidates is None:
            counts["selected_measurement_id_missing_from_measurement_store"] += 1
            audited.append(
                {
                    **base,
                    "geometry_source": "missing_measurement_record",
                    "explicit_multi_object_id": None,
                    "object_identity_status": "selected_measurement_missing",
                    "multi_object_visible_surface_distance": {"available": False},
                }
            )
            continue
        object_label = unique_field(candidates, "object_label")
        source_model = unique_field(candidates, "source_model")
        explicit_object_id = normalize_object_label(object_label)
        if explicit_object_id is None:
            counts["selected_measurement_without_explicit_multi_object_id"] += 1
            distance = {"available": False}
            identity_status = "legacy_object_label_without_multi_object_id"
        else:
            multi_row = multi_by_object_side.get((frame_idx, explicit_object_id, side))
            distance = multi_contact_distance_summary(multi_row)
            identity_status = "explicit_multi_object_id"
            if distance.get("available") is True:
                counts["selected_measurement_with_multi_object_distance_row"] += 1
                if distance.get("visible_surface_distance_candidate") is True:
                    counts["selected_measurement_with_multi_object_visible_surface_candidate"] += 1
                else:
                    counts["selected_measurement_without_multi_object_visible_surface_candidate"] += 1
            else:
                counts["selected_measurement_without_multi_object_distance_row"] += 1
        if isinstance(source_model, str):
            counts[f"selected_source_model:{source_model}"] += 1
        audited.append(
            {
                **base,
                "geometry_source": source_model,
                "selected_measurement_record_count": len(candidates),
                "object_label": object_label,
                "explicit_multi_object_id": explicit_object_id,
                "object_identity_status": identity_status,
                "multi_object_visible_surface_distance": distance,
            }
        )
    return audited, dict(sorted(counts.items()))


def local_patch_conflicts(
    patches: list[dict[str, Any]],
    multi_by_object_side: dict[tuple[int, str, str], dict[str, Any]],
    near_distance_m: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for patch in patches:
        if patch.get("status") != "accepted_local_contact_patch_state":
            continue
        frame_idx = require_int(patch.get("frame_idx"), "local patch frame_idx")
        object_id = require_str(patch.get("entity_id"), "local patch entity_id")
        side = require_str(patch.get("hand_side"), "local patch hand_side")
        metric = require_dict(patch.get("hand_object_mesh_distance_m"), "local patch hand_object_mesh_distance_m")
        local_min_m = finite_number(metric.get("min"), "local patch min distance")
        multi_row = multi_by_object_side.get((frame_idx, object_id, side))
        multi_distance = multi_contact_distance_summary(multi_row)
        multi_min = multi_distance.get("min_symmetric_distance_m")
        contradiction = bool(
            isinstance(multi_min, float)
            and local_min_m <= near_distance_m
            and multi_min > near_distance_m
        )
        out.append(
            {
                "frame_idx": frame_idx,
                "object_id": object_id,
                "hand_side": side,
                "contact_measurement_id": require_str(
                    patch.get("contact_measurement_id"), "local patch contact_measurement_id"
                ),
                "local_patch_min_distance_m": local_min_m,
                "local_patch_mesh_vertices": require_int(patch.get("mesh_vertices"), "local patch mesh_vertices"),
                "local_patch_mesh_faces": require_int(patch.get("mesh_faces"), "local patch mesh_faces"),
                "patch_object_mask_fraction": finite_number(
                    patch.get("patch_object_mask_fraction"), "local patch patch_object_mask_fraction"
                ),
                "multi_object_visible_surface_distance": multi_distance,
                "source_conflict": contradiction,
            }
        )
    return out


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    measurement_dir = args.measurement_store_root / case / "measurements_v17"
    visible_report_path = existing_path(
        args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
        f"{case} visible-surface report",
    )
    material_surface_replay_path = existing_path(
        args.object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
        f"{case} material-surface replay report",
    )
    multi_contact_path = existing_path(
        args.multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
        f"{case} multi-object contact evidence report",
    )
    contact_mode_path = existing_path(
        args.contact_mode_graph_root / case / "v17_contact_mode_graph_report.json",
        f"{case} contact-mode graph report",
    )
    sparse_report_path = existing_path(
        args.sparse_graph_root / case / "v17_full_timeline_factor_graph_report.json",
        f"{case} sparse graph report",
    )
    depth_contact_path = existing_path(
        args.depth_contact_consistency_audit_root / case / "v17_depth_contact_consistency_audit_report.json",
        f"{case} depth-contact consistency audit report",
    )
    mesh_metadata_path = existing_path(
        args.sparse_graph_root / case / "object_meshes_v17_full_timeline_graph.npz.metadata.json",
        f"{case} sparse mesh metadata",
    )
    local_patch_path = existing_path(
        measurement_dir / "local_contact_patch_state_measurements.json",
        f"{case} local patch state measurements",
    )

    visible_report = require_dict(load_json(visible_report_path), f"{case} visible-surface report")
    material_surface_replay = require_dict(load_json(material_surface_replay_path), f"{case} material-surface replay report")
    multi_contact = require_dict(load_json(multi_contact_path), f"{case} multi-object contact report")
    contact_mode = require_dict(load_json(contact_mode_path), f"{case} contact-mode report")
    sparse_report = require_dict(load_json(sparse_report_path), f"{case} sparse graph report")
    depth_contact = require_dict(load_json(depth_contact_path), f"{case} depth-contact consistency audit")
    mesh_metadata = require_dict(load_json(mesh_metadata_path), f"{case} mesh metadata")
    contact_index, contact_source_counts = contact_measurement_index(measurement_dir)
    patches = local_patch_states(local_patch_path)
    accepted_patches = [patch for patch in patches if patch.get("status") == "accepted_local_contact_patch_state"]
    multi_by_frame_side, multi_by_object_side = multi_contact_indexes(multi_contact)
    ready_rows = [
        require_dict(row, "contact-mode row")
        for row in require_list(contact_mode.get("rows"), "contact-mode rows")
        if require_dict(row, "contact-mode row").get("contact_factor_ready") is True
    ]
    audited_ready_rows, selected_counts = selected_measurement_rows(
        ready_rows,
        contact_index,
        multi_by_frame_side,
        multi_by_object_side,
    )
    patch_conflicts = local_patch_conflicts(
        accepted_patches,
        multi_by_object_side,
        finite_number(
            require_dict(multi_contact.get("parameters"), "multi-object contact parameters").get("near_distance_m"),
            "multi-object contact near_distance_m",
        ),
    )
    ready_same_frame_candidates = [
        row for row in audited_ready_rows if row["same_frame_side_visible_surface_distance_candidate_rows"] > 0
    ]
    ready_same_frame_measured = [
        row for row in audited_ready_rows if row["same_frame_side_measured_multi_object_rows"] > 0
    ]
    patch_conflict_count = sum(1 for row in patch_conflicts if row["source_conflict"] is True)
    local_patch_min_distances = [
        finite_number(
            require_dict(patch.get("hand_object_mesh_distance_m"), "local patch distance").get("min"),
            "local patch min",
        )
        for patch in accepted_patches
    ]
    patch_fractions = [
        finite_number(patch.get("patch_object_mask_fraction"), "local patch fraction")
        for patch in accepted_patches
    ]
    source_incompatibility_count = 0
    if require_int(contact_mode.get("contact_factor_ready_count"), "contact-mode contact_factor_ready_count") > 0:
        if require_int(multi_contact.get("contact_factor_ready_rows"), "multi-object contact_factor_ready_rows") == 0:
            source_incompatibility_count += 1
    if patch_conflict_count > 0:
        source_incompatibility_count += patch_conflict_count
    if require_int(
        depth_contact.get("depth_owner_incompatibility_count"),
        "depth-contact depth_owner_incompatibility_count",
    ) > 0:
        source_incompatibility_count += 1
    if require_int(mesh_metadata.get("mesh_frames"), "mesh metadata mesh_frames") != require_int(
        sparse_report.get("object_variable_frames"), "sparse report object_variable_frames"
    ):
        raise RuntimeError(f"{case} mesh frame count disagrees with sparse object-variable frames")

    report = {
        "method": "build_v17_geometry_source_audit",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "multi_object_visible_surface_report": source_summary(visible_report_path, visible_report),
            "object_material_surface_replay_report": source_summary(material_surface_replay_path, material_surface_replay),
            "multi_object_contact_evidence_report": source_summary(multi_contact_path, multi_contact),
            "contact_mode_graph_report": source_summary(contact_mode_path, contact_mode),
            "sparse_graph_report": source_summary(sparse_report_path, sparse_report),
            "depth_contact_consistency_audit_report": source_summary(depth_contact_path, depth_contact),
            "sparse_mesh_metadata": source_summary(mesh_metadata_path, mesh_metadata),
            "local_contact_patch_state_measurements": {"path": str(local_patch_path), "row_count": len(patches)},
            "contact_measurement_files": contact_source_counts,
        },
        "frame_count": require_int(sparse_report.get("frame_count"), "sparse frame_count"),
        "geometry_source_counts": {
            "multi_object_visible_surface_rows": require_int(
                visible_report.get("surface_frame_rows"), "visible surface_frame_rows"
            ),
            "multi_object_visible_surface_rejected_rows": require_int(
                visible_report.get("rejected_visible_object_frame_rows"),
                "visible rejected_visible_object_frame_rows",
            ),
            "multi_object_visible_surface_object_counts": object_visible_surface_counts(visible_report),
            "legacy_single_stream_object_variable_frames": require_int(
                sparse_report.get("object_variable_frames"), "sparse object_variable_frames"
            ),
            "legacy_single_stream_mesh_frames": require_int(mesh_metadata.get("mesh_frames"), "mesh metadata mesh_frames"),
            "legacy_single_stream_missing_mesh_frame_count": require_int(
                mesh_metadata.get("missing_mesh_frame_count"), "mesh metadata missing_mesh_frame_count"
            ),
            "local_contact_patch_state_rows": len(patches),
            "accepted_local_contact_patch_state_rows": len(accepted_patches),
            "local_contact_patch_min_distance_m": summarize_numbers(local_patch_min_distances),
            "local_contact_patch_mask_fraction": summarize_numbers(patch_fractions),
            "partial_visible_surface_replay_candidate_count": require_int(
                material_surface_replay.get("partial_visible_surface_replay_candidate_count"),
                "material surface partial_visible_surface_replay_candidate_count",
            ),
            "partial_visible_surface_replay_ready_count": require_int(
                material_surface_replay.get("partial_visible_surface_replay_ready_count"),
                "material surface partial_visible_surface_replay_ready_count",
            ),
            "partial_visible_surface_replay_ready_candidate_ids": require_list(
                material_surface_replay.get("ready_candidate_ids"),
                "material surface ready_candidate_ids",
            ),
            "accepted_reconstruction_depth_contact_evaluated_frame_count": require_int(
                depth_contact.get("evaluated_frame_count"),
                "depth-contact evaluated_frame_count",
            ),
            "accepted_reconstruction_depth_contact_evaluated_hand_rows": require_int(
                depth_contact.get("evaluated_hand_rows"),
                "depth-contact evaluated_hand_rows",
            ),
            "accepted_reconstruction_near_hand_rows": require_int(
                depth_contact.get("near_reconstructed_mesh_hand_rows"),
                "depth-contact near_reconstructed_mesh_hand_rows",
            ),
            "accepted_reconstruction_contact_candidate_rows": require_int(
                depth_contact.get("reconstructed_mesh_contact_candidate_rows"),
                "depth-contact reconstructed_mesh_contact_candidate_rows",
            ),
            "legacy_contact_ready_hand_rows_on_accepted_reconstruction_window": require_int(
                depth_contact.get("legacy_contact_ready_hand_rows"),
                "depth-contact legacy_contact_ready_hand_rows",
            ),
            "multi_object_reconstructed_object_contact_candidate_rows": require_int(
                depth_contact.get("multi_object_reconstructed_object_contact_candidate_rows"),
                "depth-contact multi_object_reconstructed_object_contact_candidate_rows",
            ),
            "legacy_owner_mismatch_frame_count": require_int(
                depth_contact.get("legacy_owner_mismatch_frame_count"),
                "depth-contact legacy_owner_mismatch_frame_count",
            ),
            "shared_depth_state_ready_frame_count": require_int(
                depth_contact.get("shared_depth_state_ready_frame_count"),
                "depth-contact shared_depth_state_ready_frame_count",
            ),
            "depth_owner_incompatibility_count": require_int(
                depth_contact.get("depth_owner_incompatibility_count"),
                "depth-contact depth_owner_incompatibility_count",
            ),
            "visible_unidepth_m": depth_contact.get("visible_unidepth_m"),
            "reconstructed_mesh_camera_depth_m": depth_contact.get("reconstructed_mesh_camera_depth_m"),
            "reconstructed_mesh_front_surface_depth_abs_p95_m": depth_contact.get(
                "reconstructed_mesh_front_surface_depth_abs_p95_m"
            ),
            "legacy_object_center_depth_m": depth_contact.get("legacy_object_center_depth_m"),
            "hand_source_depth_m": depth_contact.get("hand_source_depth_m"),
            "reconstructed_mesh_to_hand_min_m": depth_contact.get("reconstructed_mesh_to_hand_min_m"),
        },
        "contact_source_counts": {
            "contact_mode_factor_ready_rows": require_int(
                contact_mode.get("contact_factor_ready_count"), "contact-mode contact_factor_ready_count"
            ),
            "contact_mode_factor_ready_rows_with_selected_measurement": sum(
                1 for row in ready_rows if row.get("selected_measurement_id") is not None
            ),
            "contact_mode_factor_ready_rows_without_selected_measurement": sum(
                1 for row in ready_rows if row.get("selected_measurement_id") is None
            ),
            "contact_mode_ready_rows_with_same_frame_side_multi_object_measurement": len(ready_same_frame_measured),
            "contact_mode_ready_rows_with_same_frame_side_visible_surface_candidate": len(ready_same_frame_candidates),
            "multi_object_hand_object_rows": require_int(
                multi_contact.get("hand_object_rows"), "multi-object hand_object_rows"
            ),
            "multi_object_measured_distance_rows": require_int(
                multi_contact.get("measured_distance_rows"), "multi-object measured_distance_rows"
            ),
            "multi_object_unobserved_rows": require_int(multi_contact.get("unobserved_rows"), "multi-object unobserved_rows"),
            "multi_object_visible_surface_distance_candidate_rows": require_int(
                multi_contact.get("visible_surface_distance_candidate_rows"),
                "multi-object visible_surface_distance_candidate_rows",
            ),
            "multi_object_contact_factor_ready_rows": require_int(
                multi_contact.get("contact_factor_ready_rows"), "multi-object contact_factor_ready_rows"
            ),
            "selected_measurement_audit_counts": selected_counts,
        },
        "local_patch_visible_surface_conflicts": patch_conflicts,
        "selected_contact_measurement_source_audit_preview": audited_ready_rows[:50],
        "selected_contact_measurement_source_audit_preview_limit": 50,
        "selected_contact_measurement_source_audit_preview_truncated": len(audited_ready_rows) > 50,
        "source_incompatibility_count": source_incompatibility_count,
        "source_compatibility_findings": {
            "legacy_contact_factors_supported_by_multi_object_visible_surface_contact_rows": bool(
                require_int(multi_contact.get("contact_factor_ready_rows"), "multi-object contact factor rows") > 0
            ),
            "legacy_contact_factors_have_any_same_frame_side_visible_surface_candidate": bool(ready_same_frame_candidates),
            "accepted_local_patches_conflict_with_multi_object_visible_surface_distance": bool(patch_conflict_count > 0),
            "accepted_reconstruction_meshes_contact_compatible_with_current_hand_depth": bool(
                require_int(
                    depth_contact.get("reconstructed_mesh_contact_candidate_rows"),
                    "depth-contact contact candidate rows",
                )
                > 0
            ),
            "accepted_reconstruction_meshes_share_depth_state_with_current_contact_graph": bool(
                require_int(
                    depth_contact.get("shared_depth_state_ready_frame_count"),
                    "depth-contact shared depth ready frames",
                )
                > 0
                and require_int(
                    depth_contact.get("depth_owner_incompatibility_count"),
                    "depth-contact incompatibility count",
                )
                == 0
            ),
            "legacy_contact_rows_name_the_accepted_reconstruction_object": bool(
                require_int(
                    depth_contact.get("legacy_owner_mismatch_frame_count"),
                    "depth-contact legacy owner mismatch frames",
                )
                == 0
            ),
            "partial_material_pose_replay_is_complete_object_geometry": False,
            "unified_object_geometry_source_ready": False,
            "contact_factor_source_compatible_with_multi_object_geometry": False,
            "object_pose_source_compatible_with_contact_factors": False,
        },
        "unowned_state_claims": [
            "legacy contact-mode rows estimate contact over a single object stream, while the multi-object timeline has simultaneous object rows",
            "local contact-patch meshes can support contact evidence but do not reconstruct manipulated object geometry",
            "multi-object visible RGBD surfaces are object-mask measurements without canonical topology or pose variables",
            "partial material-pose replay candidates cover only short observed-surface segments and do not own hidden geometry",
            "accepted short-segment reconstruction meshes currently use the UniDepth object-depth state, while the legacy hand/contact graph uses a different source-camera depth state",
            "accepted short-segment reconstruction meshes belong to explicit multi-object ids, while legacy contact rows come from a single-object stream without object ownership",
        ],
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / case / "v17_geometry_source_audit_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.measurement_store_root / "v17_measurement_store_summary.json",
        "measurement store summary",
    )
    summary = require_dict(load_json(summary_path), "measurement store summary")
    reports = [
        build_case(
            require_str(require_dict(case_row, f"measurement summary case {i}").get("case"), "measurement summary case"),
            args,
        )
        for i, case_row in enumerate(require_list(summary.get("cases"), "measurement summary cases"))
    ]
    payload = {
        "method": "build_v17_geometry_source_audit",
        "status": STATUS,
        "claim": CLAIM,
        "measurement_store_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_geometry_source_audit_report.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "multi_object_visible_surface_rows": require_int(
                    require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                        "multi_object_visible_surface_rows"
                    ),
                    "multi_object_visible_surface_rows",
                ),
                "legacy_single_stream_object_variable_frames": require_int(
                    require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                        "legacy_single_stream_object_variable_frames"
                    ),
                    "legacy_single_stream_object_variable_frames",
                ),
                "accepted_local_contact_patch_state_rows": require_int(
                    require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                        "accepted_local_contact_patch_state_rows"
                    ),
                    "accepted_local_contact_patch_state_rows",
                ),
                "partial_visible_surface_replay_ready_count": require_int(
                    require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                        "partial_visible_surface_replay_ready_count"
                    ),
                    "partial_visible_surface_replay_ready_count",
                ),
                "contact_mode_factor_ready_rows": require_int(
                    require_dict(report.get("contact_source_counts"), "contact_source_counts").get(
                        "contact_mode_factor_ready_rows"
                    ),
                    "contact_mode_factor_ready_rows",
                ),
                "multi_object_contact_factor_ready_rows": require_int(
                    require_dict(report.get("contact_source_counts"), "contact_source_counts").get(
                        "multi_object_contact_factor_ready_rows"
                    ),
                    "multi_object_contact_factor_ready_rows",
                ),
                "source_incompatibility_count": require_int(
                    report.get("source_incompatibility_count"), "source_incompatibility_count"
                ),
                "unified_object_geometry_source_ready": False,
                "contact_factor_source_compatible_with_multi_object_geometry": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for report in reports
        ],
        "multi_object_visible_surface_rows": sum(
            require_int(
                require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                    "multi_object_visible_surface_rows"
                ),
                "multi_object_visible_surface_rows",
            )
            for report in reports
        ),
        "accepted_local_contact_patch_state_rows": sum(
            require_int(
                require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                    "accepted_local_contact_patch_state_rows"
                ),
                "accepted_local_contact_patch_state_rows",
            )
            for report in reports
        ),
        "partial_visible_surface_replay_ready_count": sum(
            require_int(
                require_dict(report.get("geometry_source_counts"), "geometry_source_counts").get(
                    "partial_visible_surface_replay_ready_count"
                ),
                "partial_visible_surface_replay_ready_count",
            )
            for report in reports
        ),
        "contact_mode_factor_ready_rows": sum(
            require_int(
                require_dict(report.get("contact_source_counts"), "contact_source_counts").get(
                    "contact_mode_factor_ready_rows"
                ),
                "contact_mode_factor_ready_rows",
            )
            for report in reports
        ),
        "multi_object_contact_factor_ready_rows": sum(
            require_int(
                require_dict(report.get("contact_source_counts"), "contact_source_counts").get(
                    "multi_object_contact_factor_ready_rows"
                ),
                "multi_object_contact_factor_ready_rows",
            )
            for report in reports
        ),
        "source_incompatibility_count": sum(
            require_int(report.get("source_incompatibility_count"), "source_incompatibility_count")
            for report in reports
        ),
        "unified_object_geometry_source_ready": False,
        "contact_factor_source_compatible_with_multi_object_geometry": False,
        "object_pose_source_compatible_with_contact_factors": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_geometry_source_audit_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--sparse-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--depth-contact-consistency-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit"),
    )
    parser.add_argument(
        "--contact-mode-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_graph"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--object-material-surface-replay-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_source_audit"),
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
