#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

STATUS = "v18_hand_baseline_branch"
CLAIM = (
    "This artifact restores the V18 hand-branch baseline as explicit evidence: HaWoR temporal hand measurements, "
    "WiLoR visible-frame measurements, RTMLib 2D keypoint anchors, and interior hand/depth state are joined on the "
    "full timeline. It does not accept occluded hand pose, contact, or ownership; missing coverage and missing score "
    "components remain explicit blockers."
)
HAND_SIDES = ("left", "right")
ACCEPT_MEDIAN_2D_PX = 35.0
ACCEPT_P95_2D_PX = 90.0
ACCEPT_RTMLIB_MEDIAN_DELTA_PX = 35.0
ACCEPT_METRIC_DEPTH_ABS_RESIDUAL_M = 0.05
ACCEPT_TEMPORAL_ACCELERATION_M_PER_FRAME2 = 0.05
ACCEPT_HAND_BONE_SCALE_ERROR_M = 0.025
HAND_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
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


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(clean),
        "median": percentile(clean, 50.0),
        "p05": percentile(clean, 5.0),
        "p95": percentile(clean, 95.0),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def source_from_measurement_manifest(manifest: dict[str, Any], source_key: str) -> tuple[Path | None, str | None]:
    rows = manifest.get(source_key)
    if not isinstance(rows, list):
        return None, None
    for raw in rows:
        row = require_dict(raw, source_key)
        status = str(row.get("status"))
        if status in {"ok", "loaded"} and row.get("path"):
            return Path(require_str(row.get("path"), f"{source_key}.path")), status
    return None, None


def entity_side(entity_id: Any, fallback: Any = None) -> str | None:
    if isinstance(entity_id, str) and entity_id.startswith("hand:"):
        side = entity_id.split(":", 1)[1]
        if side in HAND_SIDES:
            return side
    if fallback in HAND_SIDES:
        return str(fallback)
    return None


def best_measurements_by_frame_side(rows: list[Any], label: str) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in rows:
        row = require_dict(raw, f"{label} row")
        frame_idx = row.get("frame_idx")
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            continue
        side = entity_side(row.get("entity_id"), row.get("side"))
        if side is None:
            continue
        confidence = finite_float_or_none(row.get("confidence"))
        available = row.get("measurement_available") is not False
        score = (1.0 if available else 0.0, confidence if confidence is not None else -1.0)
        key = (frame_idx, side)
        current = out.get(key)
        if current is None:
            out[key] = row
            continue
        current_conf = finite_float_or_none(current.get("confidence"))
        current_score = (1.0 if current.get("measurement_available") is not False else 0.0, current_conf if current_conf is not None else -1.0)
        if score > current_score:
            out[key] = row
    return out


def rtmlib_index(path: Path | None, frame_count: int) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    if path is None or not path.exists():
        return {}, {}
    payload = require_dict(load_json(path), f"RTMLib {path}")
    by_frame: dict[int, dict[str, Any]] = {}
    by_frame_side: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(payload.get("frames"), "RTMLib frames"):
        frame = require_dict(raw_frame, "RTMLib frame")
        frame_idx = require_int(frame.get("frame_idx"), "RTMLib frame_idx")
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        hands = [require_dict(raw, "RTMLib hand") for raw in require_list(frame.get("hands", []), "RTMLib hands")]
        by_frame[frame_idx] = {
            "frame_idx": frame_idx,
            "hand_detection_count": len(hands),
            "max_mean_score": max([finite_float_or_none(row.get("mean_score")) or 0.0 for row in hands], default=None),
            "valid_keypoint_count_max": max([int(row.get("valid_keypoints") or 0) for row in hands], default=0),
        }
        for raw_cmp in require_list(frame.get("wilor_comparisons", []), "RTMLib WiLoR comparisons"):
            cmp_row = require_dict(raw_cmp, "RTMLib WiLoR comparison")
            side = cmp_row.get("wilor_side")
            if side not in HAND_SIDES:
                continue
            key = (frame_idx, str(side))
            current = by_frame_side.get(key)
            matched = int(cmp_row.get("matched_keypoints") or 0)
            median_delta = finite_float_or_none(cmp_row.get("median_keypoint_delta_px"))
            score = (matched, -(median_delta if median_delta is not None else 1e9))
            if current is None:
                by_frame_side[key] = cmp_row
                continue
            current_matched = int(current.get("matched_keypoints") or 0)
            current_delta = finite_float_or_none(current.get("median_keypoint_delta_px"))
            current_score = (current_matched, -(current_delta if current_delta is not None else 1e9))
            if score > current_score:
                by_frame_side[key] = cmp_row
    return by_frame, by_frame_side


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
        compatible = row.get("interior_metric_depth_compatible") is True
        score = 1 if compatible else 0
        current = best.get(key)
        current_score = 1 if current and current.get("interior_metric_depth_compatible") is True else 0
        if current is None or score > current_score:
            best[key] = row
    return best


def measurement_paths(case: str, args: argparse.Namespace) -> dict[str, Path]:
    root = args.measurement_store_root / case
    return {
        "wilor": root / "measurements_v17" / "wilor_measurements.json",
        "hawor": root / "measurements_v17" / "hawor_measurements.json",
        "manifest": root / "v17_measurement_manifest.json",
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def point3_list(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) == 0:
        return None
    out: list[list[float]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) < 3:
            return None
        x = finite_float_or_none(raw[0])
        y = finite_float_or_none(raw[1])
        z = finite_float_or_none(raw[2])
        if x is None or y is None or z is None:
            return None
        out.append([x, y, z])
    return out


def centroid3(points: list[list[float]]) -> list[float]:
    n = float(len(points))
    return [sum(p[axis] for p in points) / n for axis in range(3)]


def norm3(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def bone_lengths(joints: list[list[float]]) -> list[float] | None:
    if len(joints) <= max(max(edge) for edge in HAND_BONE_EDGES):
        return None
    lengths: list[float] = []
    for a, b in HAND_BONE_EDGES:
        lengths.append(norm3([joints[b][axis] - joints[a][axis] for axis in range(3)]))
    return lengths


def hawor_source_annotation_paths(hawor_rows: list[Any]) -> list[Path]:
    paths: set[Path] = set()
    for raw in hawor_rows:
        row = require_dict(raw, "HaWoR row")
        path_raw = row.get("source_annotation")
        if isinstance(path_raw, str) and path_raw:
            paths.add(Path(path_raw))
    return sorted(paths)


def load_hawor_source_geometry(hawor_rows: list[Any], frame_count: int) -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    geometry: dict[tuple[int, str], dict[str, Any]] = {}
    source_reports: list[dict[str, Any]] = []
    for path in hawor_source_annotation_paths(hawor_rows):
        report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "loaded": False, "geometry_rows": 0}
        if not path.exists():
            source_reports.append(report)
            continue
        report["sha256"] = file_sha256(path)
        payload = require_dict(load_json(path), f"HaWoR source annotation {path}")
        frames = require_list(payload.get("frames"), f"HaWoR source frames {path}")
        report["frame_count"] = len(frames)
        report["loaded"] = True
        for raw_frame in frames:
            frame = require_dict(raw_frame, "HaWoR source frame")
            frame_idx_raw = frame.get("frame_idx")
            if isinstance(frame_idx_raw, bool) or not isinstance(frame_idx_raw, int) or frame_idx_raw < 0 or frame_idx_raw >= frame_count:
                continue
            for raw_hand in require_list(frame.get("hands", []), "HaWoR source hands"):
                hand = require_dict(raw_hand, "HaWoR source hand")
                backend = str(hand.get("source_backend") or hand.get("backend") or "")
                if backend != "HaWoR":
                    continue
                side = hand.get("side")
                if side not in HAND_SIDES:
                    continue
                joints = point3_list(hand.get("joints3d_world_m"))
                coordinate_frame = "world_m"
                if joints is None:
                    joints = point3_list(hand.get("joints3d_camera"))
                    coordinate_frame = "camera_m"
                if joints is None:
                    continue
                lengths = bone_lengths(joints)
                key = (frame_idx_raw, str(side))
                detector_score = finite_float_or_none(hand.get("detector_score"))
                candidate = {
                    "frame_idx": frame_idx_raw,
                    "hand_side": str(side),
                    "source_annotation": str(path),
                    "source_annotation_sha256": report.get("sha256"),
                    "source_backend": backend,
                    "coordinate_frame_for_temporal_geometry": coordinate_frame,
                    "joint_count": len(joints),
                    "center_m": centroid3(joints),
                    "bone_lengths_m": lengths,
                    "detector_score": detector_score,
                }
                current = geometry.get(key)
                current_score = finite_float_or_none(current.get("detector_score")) if current is not None else None
                if current is None or (detector_score or -1.0) > (current_score or -1.0):
                    geometry[key] = candidate
        report["geometry_rows"] = sum(1 for row in geometry.values() if row.get("source_annotation") == str(path))
        source_reports.append(report)
    return geometry, source_reports


def derive_hawor_geometry_components(geometry: dict[tuple[int, str], dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    components: dict[tuple[int, str], dict[str, Any]] = {}
    for key, row in geometry.items():
        components[key] = {
            "hawor_source_annotation": row.get("source_annotation"),
            "hawor_source_annotation_sha256": row.get("source_annotation_sha256"),
            "hawor_geometry_coordinate_frame": row.get("coordinate_frame_for_temporal_geometry"),
            "hawor_geometry_joint_count": row.get("joint_count"),
        }
    for (frame_idx, side), row in geometry.items():
        prev = geometry.get((frame_idx - 1, side))
        nxt = geometry.get((frame_idx + 1, side))
        center = row.get("center_m")
        if isinstance(prev, dict) and isinstance(nxt, dict) and isinstance(center, list):
            prev_center = prev.get("center_m")
            next_center = nxt.get("center_m")
            if isinstance(prev_center, list) and isinstance(next_center, list) and len(prev_center) == 3 and len(next_center) == 3 and len(center) == 3:
                acceleration = [float(next_center[axis]) - 2.0 * float(center[axis]) + float(prev_center[axis]) for axis in range(3)]
                components[(frame_idx, side)]["temporal_acceleration_m_per_frame2"] = norm3(acceleration)
                components[(frame_idx, side)]["temporal_acceleration_source"] = "second_difference_of_HaWoR_3D_joint_centroid_requires_adjacent_frames"
    reference_by_side: dict[str, list[float]] = {}
    for side in HAND_SIDES:
        rows = [row.get("bone_lengths_m") for (frame_idx, row_side), row in geometry.items() if row_side == side and isinstance(row.get("bone_lengths_m"), list)]
        typed_rows: list[list[float]] = [[float(v) for v in row] for row in rows if isinstance(row, list) and len(row) == len(HAND_BONE_EDGES)]
        if len(typed_rows) >= 3:
            reference_by_side[side] = [percentile([row[i] for row in typed_rows], 50.0) or 0.0 for i in range(len(HAND_BONE_EDGES))]
    for (frame_idx, side), row in geometry.items():
        lengths_raw = row.get("bone_lengths_m")
        reference = reference_by_side.get(side)
        if isinstance(lengths_raw, list) and reference is not None and len(lengths_raw) == len(reference):
            diffs = [abs(float(lengths_raw[i]) - reference[i]) for i in range(len(reference))]
            components[(frame_idx, side)]["hand_bone_scale_median_abs_error_m"] = percentile(diffs, 50.0)
            components[(frame_idx, side)]["hand_bone_scale_reference"] = "per_side_median_HaWoR_3D_bone_lengths_within_source_annotation"
    return components


def component_state(
    hawor: dict[str, Any] | None,
    wilor: dict[str, Any] | None,
    rtmlib_cmp: dict[str, Any] | None,
    interior: dict[str, Any] | None,
    geometry_components: dict[str, Any] | None,
) -> tuple[str, list[str], dict[str, Any]]:
    blockers: list[str] = []
    med_proj = finite_float_or_none(hawor.get("projection_residual_px_median")) if hawor is not None else None
    p95_proj = finite_float_or_none(hawor.get("projection_residual_px_p95")) if hawor is not None else None
    rtmlib_delta = finite_float_or_none(rtmlib_cmp.get("median_keypoint_delta_px")) if rtmlib_cmp is not None else None
    metric_depth_gap = finite_float_or_none(interior.get("interior_median_gap_m")) if interior is not None else None
    metric_depth_abs = abs(metric_depth_gap) if metric_depth_gap is not None else None
    metric_depth_p95_abs = finite_float_or_none(interior.get("interior_p95_abs_gap_m")) if interior is not None else None
    temporal_acceleration = finite_float_or_none(geometry_components.get("temporal_acceleration_m_per_frame2")) if geometry_components is not None else None
    bone_scale_error = finite_float_or_none(geometry_components.get("hand_bone_scale_median_abs_error_m")) if geometry_components is not None else None
    hawor_available = hawor is not None and hawor.get("measurement_available") is True
    hawor_infill = hawor is not None and hawor.get("measurement_available") is False and hawor.get("evidence_role") == "hawor_motion_infill_candidate"
    projection_ok = med_proj is not None and med_proj <= ACCEPT_MEDIAN_2D_PX and (p95_proj is None or p95_proj <= ACCEPT_P95_2D_PX)
    rtmlib_ok = rtmlib_delta is not None and rtmlib_delta <= ACCEPT_RTMLIB_MEDIAN_DELTA_PX
    depth_compatible = interior is not None and interior.get("interior_metric_depth_compatible") is True
    metric_depth_ok = metric_depth_abs is not None and metric_depth_abs <= ACCEPT_METRIC_DEPTH_ABS_RESIDUAL_M
    temporal_ok = temporal_acceleration is not None and temporal_acceleration <= ACCEPT_TEMPORAL_ACCELERATION_M_PER_FRAME2
    bone_scale_ok = bone_scale_error is not None and bone_scale_error <= ACCEPT_HAND_BONE_SCALE_ERROR_M
    if hawor is None:
        blockers.append("hawor_missing_for_frame_side")
    elif not hawor_available:
        blockers.append("hawor_temporal_infill_candidate_not_measurement")
    if med_proj is None:
        blockers.append("hawor_projection_residual_missing")
    elif not projection_ok:
        blockers.append("hawor_projection_residual_above_threshold")
    if rtmlib_cmp is None:
        blockers.append("rtmlib_wilor_comparison_missing")
    elif not rtmlib_ok:
        blockers.append("rtmlib_wilor_2d_delta_above_threshold")
    if interior is None:
        blockers.append("interior_hand_depth_state_missing")
    elif not depth_compatible:
        blockers.append("interior_hand_depth_not_metric_compatible")
    score_contract_missing_components: list[str] = []
    if metric_depth_abs is None:
        blockers.append("median_metric_depth_abs_residual_component_missing")
        score_contract_missing_components.append("median_metric_depth_abs_m/0.05")
    elif not metric_depth_ok:
        blockers.append("median_metric_depth_abs_residual_above_threshold")
    if temporal_acceleration is None:
        blockers.append("temporal_acceleration_component_missing")
        score_contract_missing_components.append("temporal_acceleration_m_per_frame2/0.05")
    elif not temporal_ok:
        blockers.append("temporal_acceleration_above_threshold")
    if bone_scale_error is None:
        blockers.append("hand_bone_scale_component_missing")
        score_contract_missing_components.append("hand_bone_scale_error_m/0.025")
    elif not bone_scale_ok:
        blockers.append("hand_bone_scale_error_above_threshold")
    score_terms: dict[str, float] = {}
    if med_proj is not None:
        score_terms["hawor_projection_residual_px_median/25"] = med_proj / 25.0
    if rtmlib_delta is not None:
        score_terms["rtmlib_wilor_median_keypoint_delta_px/25"] = rtmlib_delta / 25.0
    if metric_depth_abs is not None:
        score_terms["median_metric_depth_abs_m/0.05"] = metric_depth_abs / ACCEPT_METRIC_DEPTH_ABS_RESIDUAL_M
    if temporal_acceleration is not None:
        score_terms["temporal_acceleration_m_per_frame2/0.05"] = temporal_acceleration / ACCEPT_TEMPORAL_ACCELERATION_M_PER_FRAME2
    if bone_scale_error is not None:
        score_terms["hand_bone_scale_error_m/0.025"] = bone_scale_error / ACCEPT_HAND_BONE_SCALE_ERROR_M
    score = sum(score_terms.values()) if score_terms else None
    missing_required_score_components = len(score_contract_missing_components) > 0
    full_score_components_ok = not missing_required_score_components and metric_depth_ok and temporal_ok and bone_scale_ok
    if hawor_available and projection_ok and full_score_components_ok and len(blockers) == 0:
        state = "hawor_visible_measurement_score_components_supported_no_occluded_pose_acceptance"
    elif hawor_available and projection_ok and not missing_required_score_components:
        state = "hawor_visible_measurement_score_components_present_but_blocked"
    elif hawor_available and projection_ok:
        state = "hawor_visible_measurement_partial_score_components_missing"
    elif hawor_infill:
        state = "hawor_motion_infill_candidate_not_accepted"
    elif wilor is not None:
        state = "wilor_visible_candidate_no_accepted_hawor"
    else:
        state = "no_hand_baseline_candidate"
    components = {
        "hawor_projection_residual_px_median": med_proj,
        "hawor_projection_residual_px_p95": p95_proj,
        "rtmlib_wilor_median_keypoint_delta_px": rtmlib_delta,
        "interior_metric_depth_compatible": depth_compatible,
        "median_metric_depth_abs_residual_m": metric_depth_abs,
        "metric_depth_p95_abs_residual_m": metric_depth_p95_abs,
        "temporal_acceleration_m_per_frame2": temporal_acceleration,
        "hand_bone_scale_median_abs_error_m": bone_scale_error,
        "available_partial_score_2d_terms_only": ((med_proj or 0.0) / 25.0 + (rtmlib_delta or 0.0) / 25.0) if med_proj is not None or rtmlib_delta is not None else None,
        "available_score_terms": score_terms,
        "available_score_sum": score,
        "score_contract_missing_components": score_contract_missing_components,
        "score_contract_thresholds": {
            "hawor_projection_residual_px_median": ACCEPT_MEDIAN_2D_PX,
            "hawor_projection_residual_px_p95": ACCEPT_P95_2D_PX,
            "rtmlib_wilor_median_keypoint_delta_px": ACCEPT_RTMLIB_MEDIAN_DELTA_PX,
            "median_metric_depth_abs_residual_m": ACCEPT_METRIC_DEPTH_ABS_RESIDUAL_M,
            "temporal_acceleration_m_per_frame2": ACCEPT_TEMPORAL_ACCELERATION_M_PER_FRAME2,
            "hand_bone_scale_median_abs_error_m": ACCEPT_HAND_BONE_SCALE_ERROR_M,
        },
        "score_component_sources": {
            "metric_depth_residual": "v17_interior_owned_full_residual_hand_graph.interior_median_gap_m_abs",
            "temporal_acceleration": geometry_components.get("temporal_acceleration_source") if geometry_components is not None else None,
            "hand_bone_scale": geometry_components.get("hand_bone_scale_reference") if geometry_components is not None else None,
            "hawor_source_annotation": geometry_components.get("hawor_source_annotation") if geometry_components is not None else None,
            "hawor_source_annotation_sha256": geometry_components.get("hawor_source_annotation_sha256") if geometry_components is not None else None,
            "hawor_geometry_coordinate_frame": geometry_components.get("hawor_geometry_coordinate_frame") if geometry_components is not None else None,
        },
    }
    return state, sorted(set(blockers)), components


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    v16_manifest_path = args.v16_root / case / "v16_full_pipeline_manifest.json"
    v16 = require_dict(load_json(v16_manifest_path), f"{case} V16 manifest")
    raw = require_dict(v16.get("raw_video"), f"{case} raw video")
    frame_count = require_int(raw.get("frame_count"), f"{case} frame count")
    paths = measurement_paths(case, args)
    measurement_manifest = require_dict(load_json(paths["manifest"]), f"{case} measurement manifest")
    rtmlib_path, rtmlib_manifest_status = source_from_measurement_manifest(measurement_manifest, "rtmlib_hand2d_sources")
    if not paths["wilor"].exists():
        raise RuntimeError(f"{case} WiLoR measurement file missing: {paths['wilor']}")
    if not paths["hawor"].exists():
        raise RuntimeError(f"{case} HaWoR measurement file missing: {paths['hawor']}")
    wilor_rows = require_list(load_json(paths["wilor"]), f"{case} WiLoR measurements")
    hawor_rows = require_list(load_json(paths["hawor"]), f"{case} HaWoR measurements")
    wilor_by_side = best_measurements_by_frame_side(wilor_rows, "WiLoR")
    hawor_by_side = best_measurements_by_frame_side(hawor_rows, "HaWoR")
    rtmlib_by_frame, rtmlib_by_side = rtmlib_index(rtmlib_path, frame_count)
    interior_path = args.interior_hand_graph_root / case / "v17_interior_owned_full_residual_hand_graph.json"
    interior_by_side = interior_hand_lookup(interior_path)
    hawor_source_geometry, hawor_source_reports = load_hawor_source_geometry(hawor_rows, frame_count)
    hawor_geometry_components = derive_hawor_geometry_components(hawor_source_geometry)
    hand_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    missing_score_component_counts: Counter[str] = Counter()
    metric_depth_abs_values: list[float] = []
    temporal_acceleration_values: list[float] = []
    bone_scale_error_values: list[float] = []
    for frame_idx in range(frame_count):
        for side in HAND_SIDES:
            key = (frame_idx, side)
            wilor = wilor_by_side.get(key)
            hawor = hawor_by_side.get(key)
            rtmlib_cmp = rtmlib_by_side.get(key)
            interior = interior_by_side.get(key)
            geometry = hawor_geometry_components.get(key)
            state, blockers, components = component_state(hawor, wilor, rtmlib_cmp, interior, geometry)
            state_counts[state] += 1
            blocker_counts.update(blockers)
            missing_score_component_counts.update([str(v) for v in components.get("score_contract_missing_components", []) if isinstance(v, str)])
            metric_depth_abs = finite_float_or_none(components.get("median_metric_depth_abs_residual_m"))
            temporal_acceleration = finite_float_or_none(components.get("temporal_acceleration_m_per_frame2"))
            bone_scale_error = finite_float_or_none(components.get("hand_bone_scale_median_abs_error_m"))
            if metric_depth_abs is not None:
                metric_depth_abs_values.append(metric_depth_abs)
            if temporal_acceleration is not None:
                temporal_acceleration_values.append(temporal_acceleration)
            if bone_scale_error is not None:
                bone_scale_error_values.append(bone_scale_error)
            hand_rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "hand_baseline_state": state,
                    "acceptance_blockers": blockers,
                    "baseline_score_components": components,
                    "wilor_measurement_available": wilor is not None,
                    "wilor_confidence": finite_float_or_none(wilor.get("confidence")) if wilor is not None else None,
                    "wilor_bbox_xyxy": wilor.get("bbox_xyxy") if wilor is not None else None,
                    "hawor_candidate_present": hawor is not None,
                    "hawor_measurement_available": hawor.get("measurement_available") is True if hawor is not None else False,
                    "hawor_evidence_role": hawor.get("evidence_role") if hawor is not None else None,
                    "hawor_confidence": finite_float_or_none(hawor.get("confidence")) if hawor is not None else None,
                    "hawor_projection_residual_px_median": components["hawor_projection_residual_px_median"],
                    "hawor_projection_residual_px_p95": components["hawor_projection_residual_px_p95"],
                    "rtmlib_frame_detection_count": int(rtmlib_by_frame.get(frame_idx, {}).get("hand_detection_count") or 0),
                    "rtmlib_wilor_comparison_available": rtmlib_cmp is not None,
                    "rtmlib_wilor_median_keypoint_delta_px": components["rtmlib_wilor_median_keypoint_delta_px"],
                    "interior_metric_depth_state": interior.get("interior_state") if interior is not None else None,
                    "interior_metric_depth_compatible": components["interior_metric_depth_compatible"],
                    "median_metric_depth_abs_residual_m": components["median_metric_depth_abs_residual_m"],
                    "metric_depth_p95_abs_residual_m": components["metric_depth_p95_abs_residual_m"],
                    "temporal_acceleration_m_per_frame2": components["temporal_acceleration_m_per_frame2"],
                    "hand_bone_scale_median_abs_error_m": components["hand_bone_scale_median_abs_error_m"],
                    "score_contract_missing_components": components["score_contract_missing_components"],
                    "temporal_occlusion_pose_accepted": False,
                    "pose_claim": "no_occluded_pose_accepted_from_current_hand_baseline",
                }
            )
    frames: list[dict[str, Any]] = []
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in hand_rows:
        by_frame[require_int(row.get("frame_idx"), "hand row frame")].append(row)
    for frame_idx in range(frame_count):
        frames.append({"frame_idx": frame_idx, "hands": by_frame.get(frame_idx, [])})
    hawor_available_rows = [row for row in hawor_rows if require_dict(row, "HaWoR row").get("measurement_available") is True]
    hawor_infill_rows = [row for row in hawor_rows if require_dict(row, "HaWoR row").get("measurement_available") is False and require_dict(row, "HaWoR row").get("evidence_role") == "hawor_motion_infill_candidate"]
    hawor_frames = sorted({require_int(require_dict(row, "HaWoR row").get("frame_idx"), "HaWoR frame_idx") for row in hawor_rows})
    required_frame_side_keys = {(frame_idx, side) for frame_idx in range(frame_count) for side in HAND_SIDES}
    hawor_available_frame_side_keys = {
        (require_int(require_dict(row, "HaWoR row").get("frame_idx"), "HaWoR frame_idx"), require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").split(":", 1)[1])
        for row in hawor_rows
        if require_dict(row, "HaWoR row").get("measurement_available") is True
        and isinstance(require_dict(row, "HaWoR row").get("entity_id"), str)
        and require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").startswith("hand:")
        and require_str(require_dict(row, "HaWoR row").get("entity_id"), "HaWoR entity_id").split(":", 1)[1] in HAND_SIDES
    }
    hawor_projection_residuals = [
        value
        for row in hawor_rows
        for value in [finite_float_or_none(require_dict(row, "HaWoR row").get("projection_residual_px_median"))]
        if value is not None
    ]
    hawor_confidences = [
        value
        for row in hawor_rows
        for value in [finite_float_or_none(require_dict(row, "HaWoR row").get("confidence"))]
        if value is not None
    ]
    rtmlib_frames_with_hands = sum(1 for row in rtmlib_by_frame.values() if int(row.get("hand_detection_count") or 0) > 0)
    full_video_hawor_ready = required_frame_side_keys.issubset(hawor_available_frame_side_keys)
    output_dir = args.output_root / case
    state_path = output_dir / "v18_hand_baseline_branch.json"
    report = {
        "method": "build_v18_hand_baseline_branch",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "raw_video": raw,
        "frame_count": frame_count,
        "state_path": str(state_path),
        "sources": {
            "v16_manifest": str(v16_manifest_path),
            "v17_measurement_manifest": str(paths["manifest"]),
            "wilor_measurements": str(paths["wilor"]),
            "hawor_measurements": str(paths["hawor"]),
            "rtmlib_hand2d": str(rtmlib_path) if rtmlib_path is not None else None,
            "rtmlib_manifest_status": rtmlib_manifest_status,
            "interior_hand_graph": str(interior_path),
            "hawor_source_annotations": hawor_source_reports,
        },
        "hand_state_row_count": len(hand_rows),
        "hand_baseline_state_counts": dict(sorted(state_counts.items())),
        "acceptance_blocker_counts": dict(sorted(blocker_counts.items())),
        "score_contract_missing_component_counts": dict(sorted(missing_score_component_counts.items())),
        "baseline_score_component_stats": {
            "median_metric_depth_abs_residual_m": stats(metric_depth_abs_values),
            "temporal_acceleration_m_per_frame2": stats(temporal_acceleration_values),
            "hand_bone_scale_median_abs_error_m": stats(bone_scale_error_values),
        },
        "hawor_source_geometry_row_count": len(hawor_source_geometry),
        "hawor_geometry_component_row_count": len(hawor_geometry_components),
        "score_component_policy": "Metric depth uses absolute V17 interior median gap when present; temporal acceleration uses adjacent-frame second difference of HaWoR 3D joint centroids; bone scale is within-source per-side median 3D bone-length consistency. These evidence terms do not accept occluded pose.",
        "wilor_measurement_row_count": len(wilor_rows),
        "wilor_frame_side_count": len(wilor_by_side),
        "hawor_measurement_row_count": len(hawor_rows),
        "hawor_available_measurement_count": len(hawor_available_rows),
        "hawor_motion_infill_candidate_count": len(hawor_infill_rows),
        "hawor_frame_min": min(hawor_frames) if hawor_frames else None,
        "hawor_frame_max": max(hawor_frames) if hawor_frames else None,
        "hawor_unique_frame_count": len(hawor_frames),
        "hawor_required_frame_side_count": len(required_frame_side_keys),
        "hawor_available_frame_side_count": len(hawor_available_frame_side_keys),
        "hawor_missing_available_frame_side_count": len(required_frame_side_keys - hawor_available_frame_side_keys),
        "hawor_projection_residual_px_median_stats": stats(hawor_projection_residuals),
        "hawor_confidence_stats": stats(hawor_confidences),
        "hawor_full_video_baseline_ready": full_video_hawor_ready,
        "hawor_full_video_blockers": [] if full_video_hawor_ready else ["hawor_measurements_do_not_cover_full_video_all_frame_sides"],
        "rtmlib_manifest_status": rtmlib_manifest_status,
        "rtmlib_source_status_normalized": rtmlib_path is not None,
        "rtmlib_frame_count": len(rtmlib_by_frame),
        "rtmlib_frames_with_hands": rtmlib_frames_with_hands,
        "rtmlib_wilor_comparison_count": len(rtmlib_by_side),
        "temporal_occlusion_pose_accepted_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "acceptance_policy": "HaWoR/WiLoR/RTMLib rows are measurement evidence only until all score components and boundary/occlusion checks pass; no current row fills occluded pose.",
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    state = {**report, "frames": frames}
    write_json(state_path, state)
    write_json(output_dir / "v18_hand_baseline_branch_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_hand_baseline_branch",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "hand_state_row_count": sum(require_int(report.get("hand_state_row_count"), "hand rows") for report in reports),
        "wilor_measurement_row_count": sum(require_int(report.get("wilor_measurement_row_count"), "WiLoR rows") for report in reports),
        "hawor_measurement_row_count": sum(require_int(report.get("hawor_measurement_row_count"), "HaWoR rows") for report in reports),
        "hawor_available_measurement_count": sum(require_int(report.get("hawor_available_measurement_count"), "HaWoR available") for report in reports),
        "hawor_motion_infill_candidate_count": sum(require_int(report.get("hawor_motion_infill_candidate_count"), "HaWoR infill") for report in reports),
        "hawor_full_video_ready_case_count": sum(1 for report in reports if report.get("hawor_full_video_baseline_ready") is True),
        "hawor_full_video_baseline_ready_all_cases": all(report.get("hawor_full_video_baseline_ready") is True for report in reports),
        "rtmlib_loaded_case_count": sum(1 for report in reports if report.get("rtmlib_source_status_normalized") is True),
        "rtmlib_frames_with_hands": sum(require_int(report.get("rtmlib_frames_with_hands"), "RTMLib frames with hands") for report in reports),
        "rtmlib_wilor_comparison_count": sum(require_int(report.get("rtmlib_wilor_comparison_count"), "RTMLib WiLoR comparisons") for report in reports),
        "hawor_source_geometry_row_count": sum(require_int(report.get("hawor_source_geometry_row_count"), "HaWoR source geometry rows") for report in reports),
        "hawor_geometry_component_row_count": sum(require_int(report.get("hawor_geometry_component_row_count"), "HaWoR geometry component rows") for report in reports),
        "score_contract_missing_component_counts_by_case": {str(report.get("case")): report.get("score_contract_missing_component_counts") for report in reports},
        "baseline_score_component_stats_by_case": {str(report.get("case")): report.get("baseline_score_component_stats") for report in reports},
        "temporal_occlusion_pose_accepted_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_hand_baseline_branch_report.json"),
                "state_path": str(args.output_root / str(report["case"]) / "v18_hand_baseline_branch.json"),
                "hawor_measurement_row_count": report.get("hawor_measurement_row_count"),
                "hawor_available_measurement_count": report.get("hawor_available_measurement_count"),
                "hawor_motion_infill_candidate_count": report.get("hawor_motion_infill_candidate_count"),
                "hawor_full_video_baseline_ready": report.get("hawor_full_video_baseline_ready"),
                "rtmlib_source_status_normalized": report.get("rtmlib_source_status_normalized"),
                "rtmlib_frames_with_hands": report.get("rtmlib_frames_with_hands"),
                "hawor_source_geometry_row_count": report.get("hawor_source_geometry_row_count"),
                "hawor_geometry_component_row_count": report.get("hawor_geometry_component_row_count"),
                "score_contract_missing_component_counts": report.get("score_contract_missing_component_counts"),
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_hand_baseline_branch_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--measurement-store-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--interior-hand-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
