#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

STATUS = "v18_visibility_occlusion_state_scaffold"
CLAIM = (
    "This V18 artifact materializes explicit full-timeline hand/object visibility and occlusion-state rows "
    "from existing fast evidence. It does not infer certain poses through occlusion and does not claim object "
    "geometry completion."
)
HAND_SIDES = ("left", "right")


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


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    xs = sorted(float(v) for v in values)
    def percentile(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {
        "count": len(xs),
        "median": percentile(50.0),
        "p05": percentile(5.0),
        "p95": percentile(95.0),
        "min": xs[0],
        "max": xs[-1],
    }


def physical_state_from_notes(notes: Any) -> str:
    if not isinstance(notes, str) or not notes.strip():
        return "unknown"
    text = notes.lower()
    # The text is model-produced VLM physical evidence. This parser is category-agnostic: it only maps
    # explicit physical words to state classes and never branches on object names/colors/actions.
    if any(word in text for word in ("hinged", "lever", "articulated")):
        return "articulated"
    if any(word in text for word in ("deformable", "non-rigid", "nonrigid", "flexible", "thin", "changes shape", "curled")):
        return "deformable"
    if "rigid" in text:
        return "rigid"
    if any(word in text for word in ("transparent", "translucent", "reflective")):
        return "unknown_optically_difficult"
    return "unknown"


def best_wilor_by_frame_side(wilor_path: Path, frame_count: int) -> dict[tuple[int, str], dict[str, Any]]:
    payload = require_dict(load_json(wilor_path), f"WiLoR raw {wilor_path}")
    frames = require_list(payload.get("frames"), f"WiLoR frames {wilor_path}")
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in frames:
        frame = require_dict(raw_frame, "WiLoR frame")
        frame_idx = require_int(frame.get("frame_idx"), "WiLoR frame_idx")
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        for raw_hand in require_list(frame.get("raw_hands", []), "WiLoR raw_hands"):
            hand = require_dict(raw_hand, "WiLoR hand")
            side_raw = hand.get("side")
            if side_raw not in HAND_SIDES:
                continue
            score = finite_float(hand.get("detector_score", 0.0), "WiLoR detector_score")
            key = (frame_idx, str(side_raw))
            current = best.get(key)
            if current is None or score > finite_float(current.get("score", 0.0), "current score"):
                best[key] = {
                    "source": "wilor_raw",
                    "score": score,
                    "bbox_xyxy": hand.get("bbox_xyxy"),
                    "filter_status": hand.get("filter_status"),
                    "has_mano": True,
                }
    return best


def rtmlib_by_frame(rtmlib_path: Path | None, frame_count: int) -> dict[int, list[dict[str, Any]]]:
    if rtmlib_path is None or not rtmlib_path.exists():
        return {}
    payload = require_dict(load_json(rtmlib_path), f"RTMLib {rtmlib_path}")
    out: dict[int, list[dict[str, Any]]] = {}
    for raw_frame in require_list(payload.get("frames"), "RTMLib frames"):
        frame = require_dict(raw_frame, "RTMLib frame")
        frame_idx = require_int(frame.get("frame_idx"), "RTMLib frame_idx")
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        rows: list[dict[str, Any]] = []
        for raw_hand in require_list(frame.get("hands", []), "RTMLib hands"):
            hand = require_dict(raw_hand, "RTMLib hand")
            rows.append(
                {
                    "source": "rtmlib_hand2d",
                    "score": finite_float(hand.get("mean_score", 0.0), "RTMLib mean_score"),
                    "valid_keypoints": hand.get("valid_keypoints"),
                    "bbox_xyxy": hand.get("bbox_xyxy"),
                }
            )
        if rows:
            out[frame_idx] = rows
    return out


def source_from_measurement_manifest(manifest: dict[str, Any], source_key: str) -> Path | None:
    rows = manifest.get(source_key)
    if not isinstance(rows, list):
        return None
    for raw in rows:
        row = require_dict(raw, source_key)
        if row.get("status") in {"ok", "loaded"} and row.get("path"):
            return Path(require_str(row.get("path"), f"{source_key}.path"))
    return None


def visible_window_flags(frame_count: int, observed_frames_by_side: dict[str, set[int]], max_gap: int) -> dict[tuple[int, str], dict[str, Any]]:
    flags: dict[tuple[int, str], dict[str, Any]] = {}
    for side, observed in observed_frames_by_side.items():
        sorted_frames = sorted(observed)
        for a, b in zip(sorted_frames, sorted_frames[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for frame_idx in range(a + 1, b):
                    flags[(frame_idx, side)] = {
                        "bounded_gap_between_observations": True,
                        "previous_observed_frame": a,
                        "next_observed_frame": b,
                        "gap_length_frames": gap,
                    }
    return flags


def object_lookup(
    timeline: dict[str, Any],
    roster_by_object_id: dict[str, dict[str, Any]],
    physical_schema_by_object_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    objects = []
    for row in [require_dict(raw, "timeline object") for raw in require_list(timeline.get("objects"), "timeline objects")]:
        object_id = require_str(row.get("object_id"), "timeline object_id")
        # The V17 timeline intentionally kept mask/interval state small and dropped VLM physical notes.
        # V18 restores those model-produced physical notes from the roster instead of branching on names.
        roster_row = roster_by_object_id.get(object_id, {})
        schema_row = physical_schema_by_object_id.get(object_id, {})
        merged = {**row}
        for key in ("physical_notes", "role_status", "source"):
            if key not in merged or merged.get(key) is None:
                merged[key] = roster_row.get(key)
        for key in (
            "model_physical_state_type",
            "physical_state_source",
            "requires_part_or_relative_motion_model",
            "part_or_relative_motion_evidence_terms",
            "primary_articulation_evidence_terms",
            "secondary_deformable_or_surface_component",
            "secondary_deformable_evidence_terms",
            "optical_difficulty",
            "optical_evidence_terms",
            "surface_change_without_pose_state",
            "surface_change_evidence_terms",
            "schema_confidence",
            "schema_blockers",
            "legacy_keyword_physical_state_type",
        ):
            if key in schema_row:
                merged[key] = schema_row.get(key)
        objects.append(merged)
    by_frame_object: dict[tuple[int, str], dict[str, Any]] = {}
    active_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_frame in require_list(timeline.get("frames"), "timeline frames"):
        frame = require_dict(raw_frame, "timeline frame")
        frame_idx = require_int(frame.get("frame_idx"), "timeline frame_idx")
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            obj = require_dict(raw_obj, "frame object")
            object_id = require_str(obj.get("object_id"), "object_id")
            by_frame_object[(frame_idx, object_id)] = obj
            active_by_frame[frame_idx].append(obj)
    return objects, by_frame_object, active_by_frame


def surface_lookup(report: dict[str, Any]) -> tuple[set[tuple[int, str]], dict[tuple[int, str], str]]:
    surface_rows: set[tuple[int, str]] = set()
    rejected: dict[tuple[int, str], str] = {}
    for raw in require_list(report.get("surface_rows"), "surface_rows"):
        row = require_dict(raw, "surface row")
        surface_rows.add((require_int(row.get("frame_idx"), "surface frame_idx"), require_str(row.get("object_id"), "surface object_id")))
    for raw in require_list(report.get("rejected_rows"), "rejected_rows"):
        row = require_dict(raw, "rejected row")
        rejected[(require_int(row.get("frame_idx"), "rejected frame_idx"), require_str(row.get("object_id"), "rejected object_id"))] = str(row.get("reason"))
    return surface_rows, rejected


def interior_hand_lookup(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = require_dict(load_json(path), f"interior hand graph {path}")
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(payload.get("rows"), "interior rows"):
        row = require_dict(raw, "interior row")
        frame_idx = require_int(row.get("frame_idx"), "interior frame_idx")
        side = require_str(row.get("hand_side"), "interior hand_side")
        if side not in HAND_SIDES:
            continue
        key = (frame_idx, side)
        compatible = bool(row.get("interior_metric_depth_compatible") is True)
        score = 1 if compatible else 0
        current = best.get(key)
        current_score = 1 if current and current.get("interior_metric_depth_compatible") is True else 0
        if current is None or score > current_score:
            best[key] = {
                "source": "v17_interior_owned_hand_graph",
                "interior_state": row.get("interior_state"),
                "interior_metric_depth_compatible": compatible,
                "interior_valid_pixels": row.get("interior_valid_pixels"),
                "hand_index": row.get("hand_index"),
            }
    return best


def hand_baseline_index(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = require_dict(load_json(path), f"hand baseline {path}")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(payload.get("frames"), "hand baseline frames"):
        frame = require_dict(raw_frame, "hand baseline frame")
        frame_idx = require_int(frame.get("frame_idx"), "hand baseline frame_idx")
        for raw_hand in require_list(frame.get("hands"), "hand baseline hands"):
            hand = require_dict(raw_hand, "hand baseline hand")
            side = require_str(hand.get("hand_side"), "hand baseline hand_side")
            if side in HAND_SIDES:
                out[(frame_idx, side)] = hand
    return out


def hand_rows(
    frame_count: int,
    wilor: dict[tuple[int, str], dict[str, Any]],
    rtmlib: dict[int, list[dict[str, Any]]],
    interior: dict[tuple[int, str], dict[str, Any]],
    hand_baseline: dict[tuple[int, str], dict[str, Any]],
    active_objects_by_frame: dict[int, list[dict[str, Any]]],
    max_gap: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    observed_by_side = {side: {frame for (frame, s), row in wilor.items() if s == side and row.get("score", 0.0) is not None} for side in HAND_SIDES}
    gap_flags = visible_window_flags(frame_count, observed_by_side, max_gap)
    rows: list[dict[str, Any]] = []
    visibility_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for frame_idx in range(frame_count):
        active_objects = active_objects_by_frame.get(frame_idx, [])
        active_visible_objects = [obj for obj in active_objects if obj.get("visible") is True]
        for side in HAND_SIDES:
            w = wilor.get((frame_idx, side))
            metric = interior.get((frame_idx, side))
            rtmlib_rows = rtmlib.get(frame_idx, [])
            baseline = hand_baseline.get((frame_idx, side), {})
            evidence_sources: list[str] = []
            if w is not None:
                evidence_sources.append("wilor_raw")
            if metric is not None:
                evidence_sources.append("v17_interior_owned_hand_graph")
            if rtmlib_rows:
                evidence_sources.append("rtmlib_hand2d_unsided")
            if baseline.get("hawor_candidate_present") is True:
                evidence_sources.append("hawor_temporal_baseline")
            if baseline.get("rtmlib_wilor_comparison_available") is True:
                evidence_sources.append("rtmlib_wilor_2d_anchor")
            score = finite_float(w.get("score"), "WiLoR score") if w is not None else None
            if score is not None and score >= 0.35:
                visibility = "visible"
                occlusion_state = "not_occluded_observed_hand"
                uncertainty = "observed_model_score"
                inferred_pose_status = "observed_mano_measurement_available"
            elif score is not None:
                visibility = "partially_visible"
                occlusion_state = "not_occluded_low_confidence_observed_hand"
                uncertainty = "high_from_low_detector_score"
                inferred_pose_status = "observed_low_confidence_mano_measurement"
            else:
                flag = gap_flags.get((frame_idx, side))
                if flag is not None and active_visible_objects:
                    visibility = "unresolved"
                    occlusion_state = "possible_short_occlusion_unowned_no_depth_order"
                    uncertainty = "high_no_pose_filled"
                    inferred_pose_status = "not_filled_requires_occlusion_solver"
                elif flag is not None:
                    visibility = "unresolved"
                    occlusion_state = "short_detector_gap_without_visible_occluder"
                    uncertainty = "high_no_pose_filled"
                    inferred_pose_status = "not_filled_requires_temporal_validation"
                else:
                    visibility = "unresolved"
                    occlusion_state = "missing_detector_evidence_or_out_of_frame_unresolved"
                    uncertainty = "unknown"
                    inferred_pose_status = "not_filled"
            visibility_counts[visibility] += 1
            occlusion_counts[occlusion_state] += 1
            row = {
                "frame_idx": frame_idx,
                "hand_side": side,
                "visibility_state": visibility,
                "occlusion_state": occlusion_state,
                "occluder_owner": None,
                "occluder_owner_status": "unresolved_no_depth_order" if "possible" in occlusion_state else "not_required_or_unresolved",
                "uncertainty_state": uncertainty,
                "inferred_pose_status": inferred_pose_status,
                "evidence_sources": evidence_sources,
                "wilor_detector_score": score,
                "wilor_bbox_xyxy": w.get("bbox_xyxy") if w is not None else None,
                "rtmlib_unsided_detection_count": len(rtmlib_rows),
                "active_object_count": len(active_objects),
                "active_visible_object_count": len(active_visible_objects),
                "bounded_gap_evidence": gap_flags.get((frame_idx, side)),
                "metric_depth_state": metric.get("interior_state") if metric is not None else "not_evaluated_in_interior_owned_graph",
                "metric_depth_compatible": metric.get("interior_metric_depth_compatible") if metric is not None else False,
                "hand_baseline_state": baseline.get("hand_baseline_state"),
                "hand_baseline_acceptance_blockers": baseline.get("acceptance_blockers", []),
                "hawor_candidate_present": baseline.get("hawor_candidate_present", False),
                "hawor_measurement_available": baseline.get("hawor_measurement_available", False),
                "hawor_evidence_role": baseline.get("hawor_evidence_role"),
                "hawor_projection_residual_px_median": baseline.get("hawor_projection_residual_px_median"),
                "rtmlib_wilor_comparison_available": baseline.get("rtmlib_wilor_comparison_available", False),
                "rtmlib_wilor_median_keypoint_delta_px": baseline.get("rtmlib_wilor_median_keypoint_delta_px"),
                "hawor_temporal_occlusion_pose_accepted": baseline.get("temporal_occlusion_pose_accepted", False),
                "pose_claim": "no_certain_pose_if_unobserved" if visibility == "unresolved" else "observed_or_partially_observed_measurement",
            }
            rows.append(row)
    return rows, visibility_counts, occlusion_counts


def object_rows(
    frame_count: int,
    objects: list[dict[str, Any]],
    by_frame_object: dict[tuple[int, str], dict[str, Any]],
    surface_rows: set[tuple[int, str]],
    rejected_surfaces: dict[tuple[int, str], str],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], Counter[str]]:
    rows: list[dict[str, Any]] = []
    visibility_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    physical_counts: Counter[str] = Counter()
    for obj in objects:
        object_id = require_str(obj.get("object_id"), "object_id")
        physical_state = str(obj.get("model_physical_state_type") or physical_state_from_notes(obj.get("physical_notes")))
        physical_counts[physical_state] += 1
        for frame_idx in range(frame_count):
            source = by_frame_object.get((frame_idx, object_id))
            key = (frame_idx, object_id)
            if source is None:
                visibility = "out_of_frame"
                mask_state = "inactive_no_mask_expected"
            elif source.get("visible") is True:
                visibility = "visible"
                mask_state = "visible_mask"
            else:
                visibility = "unresolved"
                mask_state = str(source.get("mask_evidence_status", "active_without_visible_mask"))
            if key in surface_rows:
                geometry_scope = "visible_surface_depth_backed"
                hidden_geometry_state = "hidden_geometry_unresolved"
                surface_status = "surface_extracted"
            elif key in rejected_surfaces:
                geometry_scope = "visible_mask_only_surface_rejected"
                hidden_geometry_state = "hidden_geometry_unresolved"
                surface_status = rejected_surfaces[key]
            elif visibility == "visible":
                geometry_scope = "visible_mask_only_no_depth_surface"
                hidden_geometry_state = "hidden_geometry_unresolved"
                surface_status = "no_surface_row"
            else:
                geometry_scope = "no_visible_geometry"
                hidden_geometry_state = "hidden_geometry_unresolved"
                surface_status = "not_applicable"
            visibility_counts[visibility] += 1
            geometry_counts[geometry_scope] += 1
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "object_id": object_id,
                    "track_id": obj.get("track_id"),
                    "name": obj.get("name"),
                    "model_physical_state_type": physical_state,
                    "physical_state_source": obj.get("physical_state_source") or ("vlm_physical_notes_keyword_mapping" if obj.get("physical_notes") else "unknown_no_model_notes"),
                    "physical_notes": obj.get("physical_notes"),
                    "requires_part_or_relative_motion_model": bool(obj.get("requires_part_or_relative_motion_model")),
                    "part_or_relative_motion_evidence_terms": obj.get("part_or_relative_motion_evidence_terms", []),
                    "primary_articulation_evidence_terms": obj.get("primary_articulation_evidence_terms", []),
                    "secondary_deformable_or_surface_component": bool(obj.get("secondary_deformable_or_surface_component")),
                    "optical_difficulty": bool(obj.get("optical_difficulty")),
                    "surface_change_without_pose_state": bool(obj.get("surface_change_without_pose_state")),
                    "physical_state_schema_confidence": obj.get("schema_confidence"),
                    "legacy_keyword_physical_state_type": obj.get("legacy_keyword_physical_state_type"),
                    "visibility_state": visibility,
                    "mask_evidence_state": mask_state,
                    "occlusion_state": "observed_or_inactive" if visibility in {"visible", "out_of_frame"} else "active_interval_missing_mask_unresolved_possible_occlusion",
                    "occluder_owner": None,
                    "occluder_owner_status": "unresolved_no_depth_order" if visibility == "unresolved" else "not_required",
                    "geometry_scope": geometry_scope,
                    "surface_status": surface_status,
                    "hidden_geometry_state": hidden_geometry_state,
                    "object_pose_claim": "not_complete_object_pose",
                    "object_geometry_complete": False,
                    "object_pose_requirement_met": False,
                }
            )
    return rows, visibility_counts, geometry_counts, physical_counts


def case_state(case: str, args: argparse.Namespace) -> dict[str, Any]:
    v16_manifest_path = existing(args.v16_root / case / "v16_full_pipeline_manifest.json", f"{case} V16 manifest")
    v16 = require_dict(load_json(v16_manifest_path), f"{case} V16 manifest")
    raw = require_dict(v16.get("raw_video"), f"{case} raw_video")
    frame_count = require_int(raw.get("frame_count"), f"{case} frame_count")
    fps = finite_float(raw.get("fps"), f"{case} fps")
    measurement_manifest_path = existing(args.measurement_store_root / case / "v17_measurement_manifest.json", f"{case} V17 measurement manifest")
    measurement_manifest = require_dict(load_json(measurement_manifest_path), f"{case} V17 measurement manifest")
    wilor_path = existing(Path(require_str(measurement_manifest.get("wilor_raw"), f"{case} wilor_raw")), f"{case} WiLoR raw")
    rtmlib_path = source_from_measurement_manifest(measurement_manifest, "rtmlib_hand2d_sources")
    timeline_path = existing(args.multi_object_timeline_root / case / "v17_multi_object_timeline.json", f"{case} object timeline")
    timeline = require_dict(load_json(timeline_path), f"{case} object timeline")
    if require_int(timeline.get("frame_count"), f"{case} timeline frame_count") != frame_count:
        raise RuntimeError(f"{case} timeline frame count disagrees with raw video")
    roster_path = existing(Path(require_str(measurement_manifest.get("object_roster"), f"{case} object_roster")), f"{case} object roster")
    roster_rows = [require_dict(row, "object roster row") for row in require_list(load_json(roster_path), f"{case} object roster")]
    roster_by_object_id = {require_str(row.get("object_id"), "roster object_id"): row for row in roster_rows}
    physical_schema_path = existing(args.physical_state_schema_root / case / "v18_physical_state_schema_report.json", f"{case} physical state schema")
    physical_schema = require_dict(load_json(physical_schema_path), f"{case} physical state schema")
    physical_schema_by_object_id = {
        require_str(row.get("object_id"), "schema object_id"): row
        for row in [require_dict(raw, "schema object row") for raw in require_list(physical_schema.get("object_rows"), "schema object rows")]
    }
    visible_surface_path = existing(args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json", f"{case} visible surface report")
    visible_surface = require_dict(load_json(visible_surface_path), f"{case} visible surface report")
    interior_path = existing(args.interior_hand_graph_root / case / "v17_interior_owned_full_residual_hand_graph.json", f"{case} interior hand graph")
    hand_baseline_path = existing(args.hand_baseline_root / case / "v18_hand_baseline_branch.json", f"{case} V18 hand baseline branch")
    hand_baseline_report_path = existing(args.hand_baseline_root / case / "v18_hand_baseline_branch_report.json", f"{case} V18 hand baseline report")
    hand_baseline_report = require_dict(load_json(hand_baseline_report_path), f"{case} hand baseline report")

    objects, by_frame_object, active_objects_by_frame = object_lookup(timeline, roster_by_object_id, physical_schema_by_object_id)
    surfaces, rejected_surfaces = surface_lookup(visible_surface)
    wilor = best_wilor_by_frame_side(wilor_path, frame_count)
    rtmlib = rtmlib_by_frame(rtmlib_path, frame_count)
    interior = interior_hand_lookup(interior_path)
    hand_baseline = hand_baseline_index(hand_baseline_path)
    hands, hand_visibility_counts, hand_occlusion_counts = hand_rows(
        frame_count=frame_count,
        wilor=wilor,
        rtmlib=rtmlib,
        interior=interior,
        hand_baseline=hand_baseline,
        active_objects_by_frame=active_objects_by_frame,
        max_gap=int(args.max_short_occlusion_gap_frames),
    )
    objs, obj_visibility_counts, obj_geometry_counts, obj_physical_counts = object_rows(
        frame_count=frame_count,
        objects=objects,
        by_frame_object=by_frame_object,
        surface_rows=surfaces,
        rejected_surfaces=rejected_surfaces,
    )
    hand_rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in hands:
        hand_rows_by_frame[require_int(row.get("frame_idx"), "hand frame_idx")].append(row)
    object_rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in objs:
        object_rows_by_frame[require_int(row.get("frame_idx"), "object frame_idx")].append(row)
    frames = [
        {
            "frame_idx": frame_idx,
            "hands": hand_rows_by_frame.get(frame_idx, []),
            "objects": object_rows_by_frame.get(frame_idx, []),
        }
        for frame_idx in range(frame_count)
    ]
    output_dir = args.output_root / case
    state_path = output_dir / "v18_visibility_occlusion_state.json"
    report = {
        "method": "build_v18_visibility_occlusion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "raw_video": raw,
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps,
        "sources": {
            "v16_manifest": str(v16_manifest_path),
            "v17_measurement_manifest": str(measurement_manifest_path),
            "wilor_raw": str(wilor_path),
            "rtmlib_hand2d": str(rtmlib_path) if rtmlib_path else None,
            "v17_multi_object_timeline": str(timeline_path),
            "v18_physical_state_schema": str(physical_schema_path),
            "v17_visible_surface_report": str(visible_surface_path),
            "v17_interior_owned_hand_graph": str(interior_path),
            "v18_hand_baseline_branch": str(hand_baseline_path),
            "v18_hand_baseline_branch_report": str(hand_baseline_report_path),
        },
        "state_path": str(state_path),
        "hand_track_count": len(HAND_SIDES),
        "hand_state_row_count": len(hands),
        "object_count": len(objects),
        "object_state_row_count": len(objs),
        "hand_visibility_state_counts": dict(sorted(hand_visibility_counts.items())),
        "hand_occlusion_state_counts": dict(sorted(hand_occlusion_counts.items())),
        "object_visibility_state_counts": dict(sorted(obj_visibility_counts.items())),
        "object_geometry_scope_counts": dict(sorted(obj_geometry_counts.items())),
        "object_model_physical_state_type_counts": dict(sorted(obj_physical_counts.items())),
        "metric_depth_compatible_hand_rows": sum(1 for row in hands if row.get("metric_depth_compatible") is True),
        "hawor_measurement_row_count": hand_baseline_report.get("hawor_measurement_row_count"),
        "hawor_available_measurement_count": hand_baseline_report.get("hawor_available_measurement_count"),
        "hawor_motion_infill_candidate_count": hand_baseline_report.get("hawor_motion_infill_candidate_count"),
        "hawor_full_video_baseline_ready": hand_baseline_report.get("hawor_full_video_baseline_ready"),
        "rtmlib_source_status_normalized": hand_baseline_report.get("rtmlib_source_status_normalized"),
        "rtmlib_frames_with_hands": hand_baseline_report.get("rtmlib_frames_with_hands"),
        "rtmlib_wilor_comparison_count": hand_baseline_report.get("rtmlib_wilor_comparison_count"),
        "hawor_temporal_occlusion_pose_accepted_count": hand_baseline_report.get("temporal_occlusion_pose_accepted_count"),
        "unresolved_hand_rows": hand_visibility_counts.get("unresolved", 0),
        "unresolved_object_rows": obj_visibility_counts.get("unresolved", 0),
        "occlusion_pose_policy": "no_pose_filled_for_unobserved_rows_in_this_scaffold",
        "object_geometry_policy": "visible_surface_only_is_not_complete_object_pose",
        **FALSE_READY,
    }
    state = {
        **report,
        "frames": frames,
    }
    write_json(state_path, state)
    write_json(output_dir / "v18_visibility_occlusion_state_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_state(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_visibility_occlusion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(args.output_root / require_str(report.get("case"), "case") / "v18_visibility_occlusion_state_report.json"),
                "state_path": require_str(report.get("state_path"), "state path"),
                "frame_count": report["frame_count"],
                "hand_state_row_count": report["hand_state_row_count"],
                "object_state_row_count": report["object_state_row_count"],
                "hand_visibility_state_counts": report["hand_visibility_state_counts"],
                "object_visibility_state_counts": report["object_visibility_state_counts"],
                "object_geometry_scope_counts": report["object_geometry_scope_counts"],
                "object_model_physical_state_type_counts": report["object_model_physical_state_type_counts"],
                "hawor_measurement_row_count": report.get("hawor_measurement_row_count"),
                "hawor_available_measurement_count": report.get("hawor_available_measurement_count"),
                "hawor_motion_infill_candidate_count": report.get("hawor_motion_infill_candidate_count"),
                "hawor_full_video_baseline_ready": report.get("hawor_full_video_baseline_ready"),
                "rtmlib_source_status_normalized": report.get("rtmlib_source_status_normalized"),
                "rtmlib_frames_with_hands": report.get("rtmlib_frames_with_hands"),
                "unresolved_hand_rows": report["unresolved_hand_rows"],
                "unresolved_object_rows": report["unresolved_object_rows"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "total_unresolved_hand_rows": sum(require_int(report.get("unresolved_hand_rows"), "unresolved hand rows") for report in reports),
        "total_unresolved_object_rows": sum(require_int(report.get("unresolved_object_rows"), "unresolved object rows") for report in reports),
        "hawor_measurement_row_count": sum(require_int(report.get("hawor_measurement_row_count"), "HaWoR rows") for report in reports),
        "hawor_available_measurement_count": sum(require_int(report.get("hawor_available_measurement_count"), "HaWoR available") for report in reports),
        "hawor_motion_infill_candidate_count": sum(require_int(report.get("hawor_motion_infill_candidate_count"), "HaWoR infill") for report in reports),
        "hawor_full_video_baseline_ready_all_cases": all(report.get("hawor_full_video_baseline_ready") is True for report in reports),
        "rtmlib_loaded_case_count": sum(1 for report in reports if report.get("rtmlib_source_status_normalized") is True),
        "rtmlib_frames_with_hands": sum(require_int(report.get("rtmlib_frames_with_hands"), "RTMLib frames") for report in reports),
        "rtmlib_wilor_comparison_count": sum(require_int(report.get("rtmlib_wilor_comparison_count"), "RTMLib comparisons") for report in reports),
        "hawor_temporal_occlusion_pose_accepted_count": sum(require_int(report.get("hawor_temporal_occlusion_pose_accepted_count"), "HaWoR accepted occlusion") for report in reports),
        "occlusion_pose_policy": "no_pose_filled_for_unobserved_rows_in_this_scaffold",
        "object_geometry_policy": "visible_surface_only_is_not_complete_object_pose",
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_visibility_occlusion_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--multi-object-timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--physical-state-schema-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_physical_state_schema"))
    parser.add_argument("--visible-surface-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"))
    parser.add_argument("--interior-hand-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"))
    parser.add_argument("--hand-baseline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--max-short-occlusion-gap-frames", type=int, default=15)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
