#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


STATUS = "v17_contact_ownership_problem_qc"
CLAIM = (
    "This artifact materializes the missing object-contact ownership variables for V17. "
    "It creates one owner-domain row for every contact-mode-ready hand-side frame and evaluates each active "
    "multi-object candidate against current geometry evidence. It is a problem materialization, not an optimizer."
)
FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}
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


def optional_str(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return require_str(value, label)


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


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(value for value in values if math.isfinite(value))
    if not vals:
        return {"count": 0}

    def pct(q: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = q * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "count": len(vals),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "min": vals[0],
        "max": vals[-1],
    }


def normalize_object_id(value: Any) -> str | None:
    if value is None:
        return None
    raw = require_str(value, "object label")
    return raw if raw.startswith("object:") else None


def load_case_inputs(case: str, args: argparse.Namespace) -> dict[str, Any]:
    measurement_dir = args.measurement_store_root / case / "measurements_v17"
    paths = {
        "measurement_store_dir": measurement_dir,
        "contact_mode": existing_path(
            args.contact_mode_graph_root / case / "v17_contact_mode_graph_report.json",
            f"{case} contact-mode graph report",
        ),
        "multi_object_timeline": existing_path(
            args.multi_object_timeline_root / case / "v17_multi_object_timeline.json",
            f"{case} multi-object timeline",
        ),
        "multi_object_contact": existing_path(
            args.multi_object_contact_evidence_root / case / "v17_multi_object_contact_evidence_report.json",
            f"{case} multi-object contact evidence report",
        ),
        "pairwise_contact_state": existing_path(
            args.pairwise_contact_state_root / case / "v17_pairwise_contact_state.json",
            f"{case} pairwise contact state report",
        ),
        "pairwise_contact_depth_gap": existing_path(
            args.pairwise_contact_depth_gap_root / case / "v17_pairwise_contact_depth_gap.json",
            f"{case} pairwise contact depth-gap report",
        ),
        "object_geometry_hypothesis_state": existing_path(
            args.object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
            f"{case} object-geometry hypothesis state",
        ),
        "geometry_source_audit": existing_path(
            args.geometry_source_audit_root / case / "v17_geometry_source_audit_report.json",
            f"{case} geometry-source audit report",
        ),
        "depth_contact_consistency": existing_path(
            args.depth_contact_consistency_audit_root / case / "v17_depth_contact_consistency_audit_report.json",
            f"{case} depth-contact consistency audit report",
        ),
    }
    payloads = {
        name: require_dict(load_json(path), f"{case} {name}")
        for name, path in paths.items()
        if name != "measurement_store_dir"
    }
    return {"paths": paths, "payloads": payloads}


def contact_measurement_index(measurement_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for name in CONTACT_MEASUREMENT_FILES:
        path = existing_path(measurement_dir / name, name)
        rows = require_list(load_json(path), str(path))
        counts[name] = len(rows)
        for i, raw in enumerate(rows):
            row = require_dict(raw, f"{path}[{i}]")
            measurement_id = require_str(row.get("measurement_id"), f"{path}[{i}].measurement_id")
            by_id.setdefault(measurement_id, []).append(row)
    return by_id, counts


def contact_ready_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, raw in enumerate(require_list(report.get("rows"), "contact-mode rows")):
        if not isinstance(raw, dict):
            continue
        row = require_dict(raw, f"contact-mode rows[{i}]")
        if row.get("contact_factor_ready") is True:
            rows.append(row)
    return rows


def timeline_by_frame(report: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], int, int]:
    frames = require_list(report.get("frames"), "multi-object timeline frames")
    frame_count = require_int(report.get("frame_count"), "multi-object timeline frame_count")
    object_frame_rows = require_int(report.get("object_frame_rows"), "multi-object timeline object_frame_rows")
    if len(frames) != frame_count:
        raise RuntimeError("multi-object timeline frame_count disagrees with frames length")
    out: dict[int, list[dict[str, Any]]] = {}
    counted = 0
    for i, raw in enumerate(frames):
        frame = require_dict(raw, f"multi-object timeline frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"multi-object timeline frames[{i}].frame_idx")
        objects = [
            require_dict(obj, f"multi-object timeline frame {frame_idx}.objects[{j}]")
            for j, obj in enumerate(require_list(frame.get("objects"), f"timeline frame {frame_idx}.objects"))
        ]
        counted += len(objects)
        out[frame_idx] = objects
    if counted != object_frame_rows:
        raise RuntimeError("multi-object timeline object_frame_rows disagrees with frame objects")
    return out, frame_count, object_frame_rows


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


def pairwise_contact_index(report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "pairwise contact rows")):
        row = require_dict(raw, f"pairwise contact rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"pairwise contact rows[{i}].frame_idx")
        object_id = require_str(row.get("object_id"), f"pairwise contact rows[{i}].object_id")
        side = require_str(row.get("hand_side"), f"pairwise contact rows[{i}].hand_side")
        key = (frame_idx, object_id, side)
        if key in out:
            raise RuntimeError(f"duplicate pairwise contact row: {key}")
        out[key] = row
    return out


def pairwise_depth_gap_index(report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "pairwise depth-gap rows")):
        row = require_dict(raw, f"pairwise depth-gap rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"pairwise depth-gap rows[{i}].frame_idx")
        side = require_str(row.get("hand_side"), f"pairwise depth-gap rows[{i}].hand_side")
        object_id = require_str(row.get("object_id"), f"pairwise depth-gap rows[{i}].object_id")
        out[(frame_idx, object_id, side)] = row
    return out


def object_state_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("objects"), "object hypothesis rows")):
        row = require_dict(raw, f"object hypothesis rows[{i}]")
        object_id = require_str(row.get("object_id"), f"object hypothesis rows[{i}].object_id")
        if object_id in out:
            raise RuntimeError(f"duplicate object hypothesis row: {object_id}")
        out[object_id] = row
    return out


def contact_measurement_payload(selected_id: str | None, measurements: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if selected_id is None:
        return {
            "selected_measurement_id": None,
            "selected_measurement_record_count": 0,
            "selected_measurement_object_id": None,
            "selected_measurement_object_label": None,
            "selected_measurement_source_models": [],
            "selected_measurement_min_mesh_distance_m": None,
            "selected_measurement_owner_state": "no_selected_measurement_id",
        }
    rows = measurements.get(selected_id)
    if rows is None:
        return {
            "selected_measurement_id": selected_id,
            "selected_measurement_record_count": 0,
            "selected_measurement_object_id": None,
            "selected_measurement_object_label": None,
            "selected_measurement_source_models": [],
            "selected_measurement_min_mesh_distance_m": None,
            "selected_measurement_owner_state": "selected_measurement_missing",
        }
    labels = [require_str(row.get("object_label"), "selected measurement object_label") for row in rows if row.get("object_label") is not None]
    object_ids = [object_id for object_id in (normalize_object_id(label) for label in labels) if object_id is not None]
    source_models = sorted(
        {require_str(row.get("source_model"), "selected measurement source_model") for row in rows if row.get("source_model") is not None}
    )
    min_distances = []
    for row in rows:
        distance = row.get("hand_object_mesh_distance_m")
        if isinstance(distance, dict) and distance.get("min") is not None:
            min_distances.append(finite_number(distance.get("min"), "selected measurement mesh min distance"))
    if object_ids:
        state = "explicit_multi_object_id"
    else:
        state = "legacy_label_without_multi_object_id"
    return {
        "selected_measurement_id": selected_id,
        "selected_measurement_record_count": len(rows),
        "selected_measurement_object_id": object_ids[0] if len(set(object_ids)) == 1 else None,
        "selected_measurement_object_label": labels[0] if len(set(labels)) == 1 else (sorted(set(labels)) if labels else None),
        "selected_measurement_source_models": source_models,
        "selected_measurement_min_mesh_distance_m": min(min_distances) if min_distances else None,
        "selected_measurement_owner_state": state,
    }


def depth_contact_index(report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "depth-contact rows")):
        row = require_dict(raw, f"depth-contact rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"depth-contact rows[{i}].frame_idx")
        object_id = require_str(row.get("object_id"), f"depth-contact rows[{i}].object_id")
        for j, hand_raw in enumerate(require_list(row.get("hand_rows"), f"depth-contact rows[{i}].hand_rows")):
            hand = require_dict(hand_raw, f"depth-contact rows[{i}].hand_rows[{j}]")
            side = require_str(hand.get("side"), f"depth-contact rows[{i}].hand_rows[{j}].side")
            out[(frame_idx, object_id, side)] = hand
    return out


def candidate_row(
    obj: dict[str, Any],
    *,
    frame_idx: int,
    side: str,
    selected_measurement: dict[str, Any],
    multi_contact: dict[str, Any] | None,
    pairwise_contact: dict[str, Any] | None,
    pairwise_depth_gap: dict[str, Any] | None,
    object_state: dict[str, Any] | None,
    depth_contact: dict[str, Any] | None,
    near_distance_m: float,
) -> dict[str, Any]:
    object_id = require_str(obj.get("object_id"), "timeline object_id")
    selected_object_id = selected_measurement.get("selected_measurement_object_id")
    selected_support = bool(selected_object_id == object_id)
    if multi_contact is None:
        multi_state = "missing_multi_object_contact_row"
        visible_distance_candidate = False
        contact_factor_ready = False
        min_distance = None
        missing_geometry = ["multi_object_contact_row"]
    else:
        multi_state = optional_str(multi_contact.get("contact_mode_state"), "multi-object contact_mode_state")
        visible_distance_candidate = bool(multi_contact.get("visible_surface_distance_candidate") is True)
        contact_factor_ready = bool(multi_contact.get("contact_factor_ready") is True)
        min_distance = optional_finite_number(multi_contact.get("min_symmetric_distance_m"), "multi-object min distance")
        missing_geometry = [
            require_str(reason, "multi-object missing geometry reason")
            for reason in require_list(multi_contact.get("missing_geometry"), "multi-object missing_geometry")
        ]
    reconstructed_near = bool(depth_contact is not None and depth_contact.get("near_reconstructed_mesh") is True)
    reconstructed_contact = bool(depth_contact is not None and depth_contact.get("reconstructed_mesh_contact_candidate") is True)
    reconstructed_distance = None
    if depth_contact is not None:
        reconstructed_distance = optional_finite_number(
            require_dict(depth_contact.get("reconstructed_mesh_distance_m"), "reconstructed mesh distance").get("min_symmetric"),
            "reconstructed mesh min distance",
        )
    geometry_reconstruction = (
        require_dict(object_state.get("geometry_reconstruction_result"), "geometry_reconstruction_result")
        if object_state
        else {}
    )
    accepted_reconstruction_count = (
        require_int(
            geometry_reconstruction.get("accepted_reconstruction_result_count"),
            "accepted_reconstruction_result_count",
        )
        if geometry_reconstruction
        else 0
    )
    owner_supported = bool(
        contact_factor_ready
        or visible_distance_candidate
        or (pairwise_contact is not None and pairwise_contact.get("contact_owner_image_supported") is True)
        or selected_support
        or reconstructed_contact
    )
    owner_geometrically_supported = bool(contact_factor_ready or visible_distance_candidate or reconstructed_contact)
    owner_image_supported = bool(pairwise_contact is not None and pairwise_contact.get("contact_owner_image_supported") is True)
    owner_metric_depth_supported = bool(
        pairwise_depth_gap is not None and pairwise_depth_gap.get("metric_depth_compatible_candidate") is True
    )
    contact_compatible_geometry = bool(
        object_state is not None and object_state.get("can_own_contact_factors") is True
    )
    evidence_state = "unsupported_candidate"
    if owner_geometrically_supported:
        evidence_state = "geometry_supported_owner_candidate"
    elif selected_support:
        evidence_state = "selected_measurement_names_candidate_without_geometry_support"
    elif owner_image_supported and pairwise_depth_gap is not None:
        evidence_state = "image_supported_candidate_with_metric_depth_contradiction"
    elif owner_image_supported:
        evidence_state = "image_supported_candidate_without_metric_geometry"
    elif multi_state == "unobserved":
        evidence_state = "unobserved_candidate"
    elif min_distance is not None and min_distance > near_distance_m:
        evidence_state = "geometry_contradicted_candidate"
    return {
        "object_id": object_id,
        "track_id": require_str(obj.get("track_id"), "timeline track_id"),
        "name": optional_str(obj.get("name"), "timeline name"),
        "active": bool(obj.get("active") is True),
        "visible": bool(obj.get("visible") is True),
        "mask_evidence_status": optional_str(obj.get("mask_evidence_status"), "timeline mask_evidence_status"),
        "geometry_hypothesis_state": object_state.get("geometry_hypothesis_state") if object_state else None,
        "selected_measurement_supports_candidate": selected_support,
        "multi_object_visible_surface": {
            "state": multi_state,
            "visible_surface_distance_candidate": visible_distance_candidate,
            "contact_factor_ready": contact_factor_ready,
            "min_symmetric_distance_m": min_distance,
            "missing_geometry": missing_geometry,
        },
        "accepted_reconstruction_contact": {
            "available": depth_contact is not None,
            "near_reconstructed_mesh": reconstructed_near,
            "reconstructed_mesh_contact_candidate": reconstructed_contact,
            "min_symmetric_distance_m": reconstructed_distance,
        },
        "pairwise_image_contact": {
            "available": pairwise_contact is not None,
            "image_overlap_candidate": bool(
                pairwise_contact is not None and pairwise_contact.get("image_overlap_candidate") is True
            ),
            "pair_contact_image_candidate": bool(
                pairwise_contact is not None and pairwise_contact.get("pair_contact_image_candidate") is True
            ),
            "contact_owner_image_supported": owner_image_supported,
            "physical_contact_factor_ready": bool(
                pairwise_contact is not None and pairwise_contact.get("physical_contact_factor_ready") is True
            ),
            "pair_contact_state": pairwise_contact.get("pair_contact_state") if pairwise_contact else None,
            "mask_distance_p05_px": (
                optional_finite_number(
                    require_dict(
                        pairwise_contact.get("image_plane_hand_mask_evidence"),
                        "pairwise image evidence",
                    ).get("mask_distance_p05_px"),
                    "pairwise mask_distance_p05_px",
                )
                if pairwise_contact
                else None
            ),
        },
        "pairwise_metric_depth": {
            "available": pairwise_depth_gap is not None,
            "depth_gap_state": pairwise_depth_gap.get("depth_gap_state") if pairwise_depth_gap else None,
            "metric_depth_compatible_candidate": owner_metric_depth_supported,
            "physical_contact_factor_ready": bool(
                pairwise_depth_gap is not None and pairwise_depth_gap.get("physical_contact_factor_ready") is True
            ),
            "hand_minus_object_depth_m": pairwise_depth_gap.get("hand_minus_object_depth_m")
            if pairwise_depth_gap
            else None,
            "abs_hand_minus_object_depth_m": pairwise_depth_gap.get("abs_hand_minus_object_depth_m")
            if pairwise_depth_gap
            else None,
        },
        "object_readiness_checks": {
            "hidden_topology_reconstructed": accepted_reconstruction_count > 0,
            "can_own_contact_factors": contact_compatible_geometry,
            "complete_mesh_timeline_ready": bool(object_state is not None and object_state.get("complete_mesh_timeline_ready") is True),
            "object_geometry_complete": bool(object_state is not None and object_state.get("object_geometry_complete") is True),
        },
        "owner_supported_by_current_evidence": owner_supported,
        "owner_geometrically_supported": owner_geometrically_supported,
        "owner_image_supported": owner_image_supported,
        "owner_metric_depth_supported": owner_metric_depth_supported,
        "owner_has_contact_compatible_geometry": contact_compatible_geometry,
        "owner_evidence_state": evidence_state,
        "contact_owner_factor_ready": False,
        "frame_idx": frame_idx,
        "hand_side": side,
    }


def owner_variable_row(
    row: dict[str, Any],
    *,
    timeline: dict[int, list[dict[str, Any]]],
    multi_by_object_side: dict[tuple[int, str, str], dict[str, Any]],
    pairwise_by_object_side: dict[tuple[int, str, str], dict[str, Any]],
    pairwise_depth_by_object_side: dict[tuple[int, str, str], dict[str, Any]],
    measurements: dict[str, list[dict[str, Any]]],
    object_states: dict[str, dict[str, Any]],
    depth_contact: dict[tuple[int, str, str], dict[str, Any]],
    near_distance_m: float,
) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "contact-mode frame_idx")
    side = require_str(row.get("side"), "contact-mode side")
    selected_id = optional_str(row.get("selected_measurement_id"), "selected_measurement_id")
    selected = contact_measurement_payload(selected_id, measurements)
    objects = timeline.get(frame_idx)
    if objects is None:
        raise RuntimeError(f"contact-mode frame {frame_idx} missing from multi-object timeline")
    candidates = [
        candidate_row(
            obj,
            frame_idx=frame_idx,
            side=side,
            selected_measurement=selected,
            multi_contact=multi_by_object_side.get((frame_idx, require_str(obj.get("object_id"), "timeline object_id"), side)),
            pairwise_contact=pairwise_by_object_side.get((frame_idx, require_str(obj.get("object_id"), "timeline object_id"), side)),
            pairwise_depth_gap=pairwise_depth_by_object_side.get((frame_idx, require_str(obj.get("object_id"), "timeline object_id"), side)),
            object_state=object_states.get(require_str(obj.get("object_id"), "timeline object_id")),
            depth_contact=depth_contact.get((frame_idx, require_str(obj.get("object_id"), "timeline object_id"), side)),
            near_distance_m=near_distance_m,
        )
        for obj in objects
    ]
    ready_candidates = [
        candidate
        for candidate in candidates
        if candidate["owner_geometrically_supported"] is True
        and candidate["owner_has_contact_compatible_geometry"] is True
    ]
    candidate_factor_ready = len(ready_candidates) == 1
    if candidate_factor_ready:
        ready_object_id = require_str(ready_candidates[0].get("object_id"), "ready contact-owner candidate object_id")
        candidates = [
            {**candidate, "contact_owner_factor_ready": candidate["object_id"] == ready_object_id}
            for candidate in candidates
        ]
    supported = [candidate for candidate in candidates if candidate["owner_supported_by_current_evidence"] is True]
    geometrically_supported = [candidate for candidate in candidates if candidate["owner_geometrically_supported"] is True]
    selected_owner_id = selected.get("selected_measurement_object_id")
    if selected_owner_id is None:
        selected_state = selected.get("selected_measurement_owner_state")
    elif not any(candidate["object_id"] == selected_owner_id for candidate in candidates):
        selected_state = "selected_object_not_active_in_multi_object_timeline"
    else:
        selected_state = "selected_object_in_candidate_domain"
    if not candidates:
        owner_state = "empty_candidate_domain"
    elif len(geometrically_supported) == 1:
        owner_state = "single_geometry_supported_candidate"
    elif len(geometrically_supported) > 1:
        owner_state = "ambiguous_geometry_supported_candidates"
    elif len(supported) == 1:
        owner_state = "single_non_geometric_supported_candidate"
    elif len(supported) > 1:
        owner_state = "ambiguous_non_geometric_supported_candidates"
    else:
        owner_state = "no_supported_candidate"
    return {
        "owner_variable_id": f"contact_owner:v17:{frame_idx:06d}:{side}",
        "frame_idx": frame_idx,
        "hand_side": side,
        "contact_mode": {
            "mode": optional_str(row.get("mode"), "contact-mode mode"),
            "contact_factor_ready": True,
            "gap_min_m": optional_finite_number(row.get("gap_min_m"), "contact-mode gap_min_m"),
            "gap_p05_m": optional_finite_number(row.get("gap_p05_m"), "contact-mode gap_p05_m"),
            "mask_distance_median_px": optional_finite_number(
                row.get("mask_distance_median_px"), "contact-mode mask_distance_median_px"
            ),
        },
        "selected_measurement": selected,
        "selected_measurement_candidate_state": selected_state,
        "candidate_object_count": len(candidates),
        "supported_candidate_count": len(supported),
        "geometrically_supported_candidate_count": len(geometrically_supported),
        "candidate_objects": candidates,
        "owner_variable_state": owner_state,
        "contact_owner_factor_ready": candidate_factor_ready,
        **FALSE_READY,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_case_inputs(case, args)
    paths: dict[str, Path] = loaded["paths"]
    payloads: dict[str, dict[str, Any]] = loaded["payloads"]
    measurements, measurement_file_counts = contact_measurement_index(paths["measurement_store_dir"])
    timeline, frame_count, object_frame_rows = timeline_by_frame(payloads["multi_object_timeline"])
    _, multi_by_object_side = multi_contact_indexes(payloads["multi_object_contact"])
    pairwise_by_object_side = pairwise_contact_index(payloads["pairwise_contact_state"])
    pairwise_depth_by_object_side = pairwise_depth_gap_index(payloads["pairwise_contact_depth_gap"])
    object_states = object_state_index(payloads["object_geometry_hypothesis_state"])
    depth_by_object_side = depth_contact_index(payloads["depth_contact_consistency"])
    ready_rows = contact_ready_rows(payloads["contact_mode"])
    near_distance_m = finite_number(
        require_dict(payloads["multi_object_contact"].get("parameters"), "multi-object contact parameters").get("near_distance_m"),
        "multi-object contact near_distance_m",
    )
    variables = [
        owner_variable_row(
            row,
            timeline=timeline,
            multi_by_object_side=multi_by_object_side,
            pairwise_by_object_side=pairwise_by_object_side,
            pairwise_depth_by_object_side=pairwise_depth_by_object_side,
            measurements=measurements,
            object_states=object_states,
            depth_contact=depth_by_object_side,
            near_distance_m=near_distance_m,
        )
        for row in ready_rows
    ]
    state_counts = Counter(row["owner_variable_state"] for row in variables)
    candidate_state_counts = Counter(
        candidate["owner_evidence_state"]
        for row in variables
        for candidate in require_list(row.get("candidate_objects"), "candidate_objects")
    )
    selected_state_counts = Counter(row["selected_measurement_candidate_state"] for row in variables)
    ready_with_selected = [row for row in variables if row["selected_measurement"]["selected_measurement_id"] is not None]
    ready_without_selected = len(variables) - len(ready_with_selected)
    supported_variables = [row for row in variables if row["supported_candidate_count"] > 0]
    geometry_supported_variables = [row for row in variables if row["geometrically_supported_candidate_count"] > 0]
    metric_depth_supported_candidate_rows = sum(
        1
        for row in variables
        for candidate in require_list(row.get("candidate_objects"), "candidate_objects")
        if candidate.get("owner_metric_depth_supported") is True
    )
    factor_ready_rows = sum(
        1
        for row in variables
        for candidate in require_list(row.get("candidate_objects"), "candidate_objects")
        if candidate.get("contact_owner_factor_ready") is True
    )
    candidate_distances = [
        finite_number(
            require_dict(candidate.get("multi_object_visible_surface"), "multi-object visible surface").get("min_symmetric_distance_m"),
            "multi-object visible-surface min distance",
        )
        for row in variables
        for candidate in require_list(row.get("candidate_objects"), "candidate_objects")
        if require_dict(candidate.get("multi_object_visible_surface"), "multi-object visible surface").get("min_symmetric_distance_m")
        is not None
    ]
    report = {
        "method": "build_v17_contact_ownership_problem",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            name: source_summary(path, payloads[name])
            for name, path in paths.items()
            if name != "measurement_store_dir"
        }
        | {"measurement_files": measurement_file_counts},
        "frame_count": frame_count,
        "object_frame_rows": object_frame_rows,
        "contact_owner_variable_count": len(variables),
        "contact_mode_ready_rows": len(ready_rows),
        "contact_owner_candidate_rows": sum(row["candidate_object_count"] for row in variables),
        "contact_owner_variables_with_selected_measurement": len(ready_with_selected),
        "contact_owner_variables_without_selected_measurement": ready_without_selected,
        "contact_owner_variables_with_supported_candidate": len(supported_variables),
        "contact_owner_variables_with_geometry_supported_candidate": len(geometry_supported_variables),
        "contact_owner_variables_without_supported_candidate": len(variables) - len(supported_variables),
        "contact_owner_factor_ready_rows": factor_ready_rows,
        "contact_owner_image_supported_candidate_rows": require_int(
            payloads["pairwise_contact_state"].get("contact_owner_image_supported_candidate_rows"),
            "pairwise contact owner image-supported rows",
        ),
        "pairwise_metric_depth_evaluated_rows": require_int(
            payloads["pairwise_contact_depth_gap"].get("evaluated_pair_depth_rows"),
            "pairwise depth-gap evaluated rows",
        ),
        "pairwise_metric_depth_compatible_candidate_rows": require_int(
            payloads["pairwise_contact_depth_gap"].get("metric_depth_compatible_candidate_rows"),
            "pairwise depth-gap compatible rows",
        ),
        "contact_owner_metric_depth_supported_candidate_rows": metric_depth_supported_candidate_rows,
        "owner_image_variables_with_single_supported_candidate": require_int(
            payloads["pairwise_contact_state"].get("owner_image_variables_with_single_supported_candidate"),
            "pairwise owner image single-supported variables",
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": require_int(
            payloads["pairwise_contact_state"].get("owner_image_variables_with_ambiguous_supported_candidates"),
            "pairwise owner image ambiguous-supported variables",
        ),
        "multi_object_visible_surface_distance_m": summarize(candidate_distances),
        "owner_variable_state_counts": dict(sorted(state_counts.items())),
        "candidate_evidence_state_counts": dict(sorted(candidate_state_counts.items())),
        "selected_measurement_candidate_state_counts": dict(sorted(selected_state_counts.items())),
        "problem_variables": variables,
        "problem_semantics": {
            "variable": "contact_owner[frame_idx, hand_side]",
            "domain": "active multi-object timeline objects in the same frame",
            "unary_evidence": [
                "legacy contact-mode readiness over frame and hand side",
                "selected contact measurement object label when it is an explicit multi-object id",
                "pairwise image-plane hand/object mask support",
                "pairwise hand/object metric depth compatibility at image-contact pixels",
                "multi-object visible-surface hand distance",
                "accepted reconstruction hand distance for matching reconstructed object id",
            ],
            "factor_ready_rule": "Exactly one candidate must be geometrically supported and backed by a contact-compatible object geometry state.",
        },
        **FALSE_READY,
    }
    if len(variables) != require_int(payloads["contact_mode"].get("contact_factor_ready_count"), "contact-mode ready count"):
        raise RuntimeError(f"{case} owner variables disagree with contact-mode ready rows")
    if report["contact_owner_variable_count"] != len(ready_rows):
        raise RuntimeError(f"{case} owner variable count disagrees with ready rows")
    write_json(args.output_root / case / "v17_contact_ownership_problem.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.measurement_store_root / "v17_measurement_store_summary.json",
        "measurement store summary",
    )
    summary = require_dict(load_json(summary_path), "measurement store summary")
    cases = [
        require_str(require_dict(row, f"measurement store summary cases[{i}]").get("case"), "case")
        for i, row in enumerate(require_list(summary.get("cases"), "measurement store summary cases"))
    ]
    reports = [case_problem(case, args) for case in cases]
    payload = {
        "method": "build_v17_contact_ownership_problem",
        "status": STATUS,
        "claim": CLAIM,
        "measurement_store_summary": str(summary_path),
        "contact_mode_graph_root": str(args.contact_mode_graph_root),
        "multi_object_timeline_root": str(args.multi_object_timeline_root),
        "multi_object_contact_evidence_root": str(args.multi_object_contact_evidence_root),
        "object_geometry_hypothesis_state_root": str(args.object_geometry_hypothesis_state_root),
        "geometry_source_audit_root": str(args.geometry_source_audit_root),
        "depth_contact_consistency_audit_root": str(args.depth_contact_consistency_audit_root),
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "problem_path": str(args.output_root / report["case"] / "v17_contact_ownership_problem.json"),
                "frame_count": report["frame_count"],
                "contact_owner_variable_count": report["contact_owner_variable_count"],
                "contact_owner_candidate_rows": report["contact_owner_candidate_rows"],
                "contact_owner_variables_with_selected_measurement": report[
                    "contact_owner_variables_with_selected_measurement"
                ],
                "contact_owner_variables_without_selected_measurement": report[
                    "contact_owner_variables_without_selected_measurement"
                ],
                "contact_owner_variables_with_supported_candidate": report[
                    "contact_owner_variables_with_supported_candidate"
                ],
                "contact_owner_variables_with_geometry_supported_candidate": report[
                    "contact_owner_variables_with_geometry_supported_candidate"
                ],
                "contact_owner_variables_without_supported_candidate": report[
                    "contact_owner_variables_without_supported_candidate"
                ],
                "contact_owner_factor_ready_rows": report["contact_owner_factor_ready_rows"],
                "contact_owner_image_supported_candidate_rows": report[
                    "contact_owner_image_supported_candidate_rows"
                ],
                "owner_image_variables_with_single_supported_candidate": report[
                    "owner_image_variables_with_single_supported_candidate"
                ],
                "owner_image_variables_with_ambiguous_supported_candidates": report[
                    "owner_image_variables_with_ambiguous_supported_candidates"
                ],
                "owner_variable_state_counts": report["owner_variable_state_counts"],
                "candidate_evidence_state_counts": report["candidate_evidence_state_counts"],
                "selected_measurement_candidate_state_counts": report[
                    "selected_measurement_candidate_state_counts"
                ],
                **FALSE_READY,
            }
            for report in reports
        ],
        "contact_owner_variable_count": sum(report["contact_owner_variable_count"] for report in reports),
        "contact_owner_candidate_rows": sum(report["contact_owner_candidate_rows"] for report in reports),
        "contact_owner_variables_with_selected_measurement": sum(
            report["contact_owner_variables_with_selected_measurement"] for report in reports
        ),
        "contact_owner_variables_without_selected_measurement": sum(
            report["contact_owner_variables_without_selected_measurement"] for report in reports
        ),
        "contact_owner_variables_with_supported_candidate": sum(
            report["contact_owner_variables_with_supported_candidate"] for report in reports
        ),
        "contact_owner_variables_with_geometry_supported_candidate": sum(
            report["contact_owner_variables_with_geometry_supported_candidate"] for report in reports
        ),
        "contact_owner_variables_without_supported_candidate": sum(
            report["contact_owner_variables_without_supported_candidate"] for report in reports
        ),
        "contact_owner_factor_ready_rows": sum(report["contact_owner_factor_ready_rows"] for report in reports),
        "contact_owner_image_supported_candidate_rows": sum(
            report["contact_owner_image_supported_candidate_rows"] for report in reports
        ),
        "owner_image_variables_with_single_supported_candidate": sum(
            report["owner_image_variables_with_single_supported_candidate"] for report in reports
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": sum(
            report["owner_image_variables_with_ambiguous_supported_candidates"] for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_contact_ownership_problem_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--contact-mode-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_graph"),
    )
    parser.add_argument(
        "--multi-object-timeline-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    parser.add_argument(
        "--multi-object-contact-evidence-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument(
        "--pairwise-contact-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"),
    )
    parser.add_argument(
        "--pairwise-contact-depth-gap-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument(
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--geometry-source-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_source_audit"),
    )
    parser.add_argument(
        "--depth-contact-consistency-audit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_ownership_problem"),
    )
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
