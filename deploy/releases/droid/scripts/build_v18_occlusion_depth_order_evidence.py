#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_occlusion_depth_order_evidence"
CLAIM = (
    "This artifact evaluates occlusion-owner box candidates with scene-depth hand states and same-frame object "
    "visible-surface depth summaries. A hand-behind-metric-depth state with an object surface resolves foreground "
    "depth order for that candidate; downstream temporal graph and HaWoR support still gate final ownership."
)
SUPPORT_STATE = "scene_depth_supports_foreground_occluder_candidate_owner_unaccepted"
CONTRADICTION_STATE = "scene_depth_contradicts_foreground_occluder_candidate"
METRIC_COMPATIBLE_STATE = "hand_scene_depth_metric_compatible_no_foreground_occluder_signal"
INSUFFICIENT_OBJECT_STATE = "insufficient_object_surface_depth"
INSUFFICIENT_HAND_STATE = "insufficient_or_untrusted_hand_depth_state"
HAWOR_SUPPORT_STATE = "hawor_mano_same_frame_object_surface_depth_supports_foreground_occluder"
HAWOR_CONTRADICTION_STATE = "hawor_mano_same_frame_object_surface_depth_contradicts_foreground_occluder"
HAWOR_COMPATIBLE_STATE = "hawor_mano_same_frame_object_surface_depth_metric_compatible_no_foreground_signal"
HAWOR_INSUFFICIENT_STATE = "hawor_mano_depth_order_insufficient_or_not_same_frame_supported"
DEFAULT_HAWOR_OUTPUTS = {
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/trash_1050_tailrepair_padded/hawor_world_hands_trimmed_1050_with_track_support.npz"),
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands_with_track_support.npz"),
}
HAWOR_OCCLUSION_SUPPORT_MARGIN_M = 0.02
HAWOR_OCCLUSION_MIN_OVERLAP_VERTICES = 20
HAWOR_OCCLUSION_OVERLAP_PAD_PX = 35.0


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


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_box(value: Any, label: str) -> list[float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [optional_float(v) for v in value]
    if any(v is None for v in vals):
        return None
    out = [float(v) for v in vals if v is not None]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def load_hawor_case(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    frame_index = {int(frame_idx): idx for idx, frame_idx in enumerate(data["frame_idx"])}
    return {"path": str(path), "data": data, "frame_index": frame_index}


def projection_scale(surface_row: dict[str, Any], hand_box: list[float], object_box: list[float]) -> tuple[float, float]:
    shape = surface_row.get("depth_pixel_shape_hw")
    if isinstance(shape, list) and len(shape) == 2:
        h = optional_float(shape[0]) or 0.0
        w = optional_float(shape[1]) or 0.0
    else:
        h = w = 0.0
    max_x = max(hand_box[2], object_box[2])
    max_y = max(hand_box[3], object_box[3])
    sx = 2.0 if w > 0 and max_x > w + 1.0 else 1.0
    sy = 2.0 if h > 0 and max_y > h + 1.0 else 1.0
    return sx, sy


def project_world_to_source(points_world: np.ndarray, rotation_c2w: np.ndarray, trans_c2w: np.ndarray, intrinsics: list[float], scale_xy: tuple[float, float]) -> np.ndarray:
    if points_world.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    cam = (points_world.astype(np.float64) - trans_c2w.astype(np.float64)) @ rotation_c2w.astype(np.float64)
    z = cam[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    cam = cam[valid]
    z = z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    u = (cam[:, 0] / z * fx + cx) * scale_xy[0]
    v = (cam[:, 1] / z * fy + cy) * scale_xy[1]
    return np.stack([u, v, z], axis=1)


def inside_box(projected_xyz: np.ndarray, box_xyxy: list[float], pad_px: float) -> np.ndarray:
    if projected_xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (projected_xyz[:, 0] >= box_xyxy[0] - pad_px)
        & (projected_xyz[:, 0] <= box_xyxy[2] + pad_px)
        & (projected_xyz[:, 1] >= box_xyxy[1] - pad_px)
        & (projected_xyz[:, 1] <= box_xyxy[3] + pad_px)
    )


def intersection_box(a: list[float], b: list[float]) -> list[float] | None:
    out = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def hawor_depth_order_evidence(row: dict[str, Any], candidate: dict[str, Any], surface_row: dict[str, Any] | None, hawor: dict[str, Any] | None) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "candidate frame_idx")
    hand_side = require_str(row.get("hand_side"), "candidate hand_side")
    out: dict[str, Any] = {
        "source": "HaWoR_metric_MANO_vertices_vs_candidate_object_visible_surface_depth",
        "hawor_npz": hawor.get("path") if isinstance(hawor, dict) else None,
        "state": HAWOR_INSUFFICIENT_STATE,
        "blockers": [],
        "same_frame_detector_supported": False,
        "accepted_as_depth_order_support": False,
    }
    if surface_row is None:
        out["blockers"].append("missing_candidate_object_surface_depth")
        return out
    if hawor is None:
        out["blockers"].append("missing_hawor_metric_mano_npz")
        return out
    data = hawor["data"]
    idx = hawor["frame_index"].get(frame_idx)
    if idx is None:
        out["blockers"].append("missing_hawor_frame")
        return out
    valid_key = f"{hand_side}_valid"
    detected_key = f"{hand_side}_detected_same_frame"
    vertices_key = f"{hand_side}_vertices_world_m"
    if valid_key not in data.files or vertices_key not in data.files:
        out["blockers"].append("missing_hawor_hand_vertices")
        return out
    if not bool(data[valid_key][idx]):
        out["blockers"].append("hawor_hand_invalid")
        return out
    same_frame_detected = bool(data[detected_key][idx]) if detected_key in data.files else False
    out["same_frame_detector_supported"] = same_frame_detected
    if not same_frame_detected:
        out["blockers"].append("hawor_no_same_frame_detector_support")
        return out
    hand_box = finite_box(row.get("interpolated_hand_box_xyxy"), "interpolated hand box")
    object_box = finite_box(candidate.get("bbox_xyxy"), "candidate object box")
    overlap = intersection_box(hand_box, object_box) if hand_box is not None and object_box is not None else None
    if overlap is None:
        out["blockers"].append("candidate_hand_object_box_overlap_missing")
        return out
    intr = surface_row.get("depth_intrinsics_fx_fy_cx_cy")
    if not (isinstance(intr, list) and len(intr) == 4 and all(optional_float(v) is not None for v in intr)):
        out["blockers"].append("missing_surface_depth_intrinsics")
        return out
    object_low = optional_float(surface_row.get("depth_low_m"))
    object_high = optional_float(surface_row.get("depth_high_m"))
    object_median = optional_float(surface_row.get("depth_median_m"))
    if object_low is None or object_high is None or object_median is None:
        out["blockers"].append("missing_surface_depth_quantiles")
        return out
    projected = project_world_to_source(
        np.asarray(data[vertices_key][idx], dtype=np.float64),
        np.asarray(data["R_c2w"][idx], dtype=np.float64),
        np.asarray(data["t_c2w"][idx], dtype=np.float64),
        [float(v) for v in intr],
        projection_scale(surface_row, hand_box, object_box),
    )
    mask = inside_box(projected, overlap, HAWOR_OCCLUSION_OVERLAP_PAD_PX)
    overlap_depths = projected[mask, 2]
    out["overlap_box_xyxy"] = overlap
    out["hawor_overlap_vertex_count"] = int(overlap_depths.shape[0])
    out["min_required_overlap_vertices"] = HAWOR_OCCLUSION_MIN_OVERLAP_VERTICES
    if overlap_depths.shape[0] < HAWOR_OCCLUSION_MIN_OVERLAP_VERTICES:
        out["blockers"].append("too_few_hawor_hand_vertices_in_candidate_overlap")
        return out
    hand_median = float(np.median(overlap_depths))
    hand_q25 = float(np.quantile(overlap_depths, 0.25))
    hand_q75 = float(np.quantile(overlap_depths, 0.75))
    foreground_margin = hand_median - object_high
    hand_front_margin = object_low - hand_q25
    out.update(
        {
            "hawor_hand_depth_median_m": hand_median,
            "hawor_hand_depth_q25_m": hand_q25,
            "hawor_hand_depth_q75_m": hand_q75,
            "object_depth_low_m": object_low,
            "object_depth_median_m": object_median,
            "object_depth_high_m": object_high,
            "object_front_margin_m": foreground_margin,
            "hand_front_margin_m": hand_front_margin,
            "support_margin_threshold_m": HAWOR_OCCLUSION_SUPPORT_MARGIN_M,
        }
    )
    if foreground_margin > HAWOR_OCCLUSION_SUPPORT_MARGIN_M:
        out["state"] = HAWOR_SUPPORT_STATE
        out["accepted_as_depth_order_support"] = True
    elif hand_front_margin > HAWOR_OCCLUSION_SUPPORT_MARGIN_M:
        out["state"] = HAWOR_CONTRADICTION_STATE
    else:
        out["state"] = HAWOR_COMPATIBLE_STATE
    return out


def hand_depth_index(visibility: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(visibility.get("frames"), "visibility frames"):
        frame = require_dict(raw_frame, "visibility frame")
        frame_idx = require_int(frame.get("frame_idx"), "visibility frame_idx")
        for raw_hand in require_list(frame.get("hands"), "visibility hands"):
            hand = require_dict(raw_hand, "visibility hand")
            side = require_str(hand.get("hand_side"), "hand_side")
            out[(frame_idx, side)] = hand
    return out


def object_surface_index(surface_report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(surface_report.get("surface_rows"), "surface rows"):
        row = require_dict(raw, "surface row")
        frame_idx = require_int(row.get("frame_idx"), "surface frame_idx")
        object_id = require_str(row.get("object_id"), "surface object_id")
        out[(frame_idx, object_id)] = row
    return out


def pair_depth_state(hand_depth_state: str, surface_row: dict[str, Any] | None) -> str:
    if surface_row is None:
        return INSUFFICIENT_OBJECT_STATE
    if hand_depth_state == "interior_hand_behind_metric_depth":
        return SUPPORT_STATE
    if hand_depth_state == "interior_hand_in_front_of_metric_depth":
        return CONTRADICTION_STATE
    if hand_depth_state == "interior_metric_depth_compatible":
        return METRIC_COMPATIBLE_STATE
    return INSUFFICIENT_HAND_STATE


def row_depth_state(pair_states: Counter[str], candidate_count: int) -> str:
    if candidate_count == 0:
        return "no_box_overlap_owner_candidate"
    if pair_states.get(SUPPORT_STATE, 0) > 0:
        return "row_scene_depth_supports_at_least_one_foreground_candidate_owner_unaccepted"
    if pair_states.get(CONTRADICTION_STATE, 0) > 0:
        return "row_scene_depth_contradicts_foreground_candidate_no_support"
    if pair_states.get(METRIC_COMPATIBLE_STATE, 0) > 0:
        return "row_metric_compatible_no_foreground_occluder_signal"
    if pair_states and sum(pair_states.values()) == pair_states.get(INSUFFICIENT_OBJECT_STATE, 0):
        return "row_insufficient_object_surface_depth"
    return "row_insufficient_or_untrusted_hand_depth_state"


def candidate_pair_record(
    row: dict[str, Any],
    candidate: dict[str, Any],
    hand: dict[str, Any] | None,
    surface_row: dict[str, Any] | None,
    hawor: dict[str, Any] | None,
) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "candidate frame_idx")
    hand_side = require_str(row.get("hand_side"), "candidate hand_side")
    object_id = require_str(candidate.get("object_id"), "candidate object_id")
    hand_state = str(hand.get("metric_depth_state")) if hand is not None else "missing_v18_hand_depth_row"
    legacy_state = pair_depth_state(hand_state, surface_row)
    hawor_evidence = hawor_depth_order_evidence(row, candidate, surface_row, hawor)
    hawor_state = str(hawor_evidence.get("state"))
    if hawor_state == HAWOR_SUPPORT_STATE:
        state = SUPPORT_STATE
    elif hawor_state == HAWOR_CONTRADICTION_STATE:
        state = CONTRADICTION_STATE
    elif hawor_state == HAWOR_COMPATIBLE_STATE:
        state = METRIC_COMPATIBLE_STATE
    else:
        state = legacy_state
    depth_order_resolved = state == SUPPORT_STATE
    out = {
        "frame_idx": frame_idx,
        "hand_side": hand_side,
        "object_id": object_id,
        "track_id": candidate.get("track_id"),
        "name": candidate.get("name"),
        "source_candidate_state": row.get("candidate_state"),
        "box_iou": candidate.get("iou"),
        "hand_box_coverage_by_object_box": candidate.get("hand_box_coverage_by_object_box"),
        "object_box_coverage_by_hand_box": candidate.get("object_box_coverage_by_hand_box"),
        "hand_metric_depth_state": hand_state,
        "hand_metric_depth_compatible": bool(hand.get("metric_depth_compatible") is True) if hand is not None else False,
        "object_surface_depth_available": surface_row is not None,
        "legacy_scene_depth_evidence_state": legacy_state,
        "hawor_mano_depth_order_evidence": hawor_evidence,
        "depth_evidence_state": state,
        "depth_order_resolved": depth_order_resolved,
        "occluder_owner_accepted": depth_order_resolved,
        "pose_filled_through_occlusion": False,
        "evidence_scope": "scene_depth_and_visible_surface_depth_order_evidence_temporal_graph_and_hawor_gate_final_owner_assignment",
    }
    if surface_row is not None:
        out.update(
            {
                "object_depth_low_m": optional_float(surface_row.get("depth_low_m")),
                "object_depth_high_m": optional_float(surface_row.get("depth_high_m")),
                "object_depth_median_m": optional_float(surface_row.get("depth_median_m")),
                "object_surface_vertices": surface_row.get("vertices"),
                "object_surface_faces": surface_row.get("faces"),
                "object_geometry_state": surface_row.get("geometry_state"),
                "object_pose_state": surface_row.get("pose_state"),
            }
        )
    return out


def case_hawor_path(case: str, args: argparse.Namespace) -> Path | None:
    if case == "trash_1050" and args.trash_hawor_npz is not None:
        return args.trash_hawor_npz
    if case == "task5_tomato_960" and args.task5_hawor_npz is not None:
        return args.task5_hawor_npz
    return DEFAULT_HAWOR_OUTPUTS.get(case)


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    occlusion_path = args.occlusion_owner_candidates_root / case / "v18_occlusion_owner_candidates_report.json"
    visibility_path = args.visibility_root / case / "v18_visibility_occlusion_state.json"
    surface_path = args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json"
    hawor_path = case_hawor_path(case, args)
    occlusion = require_dict(load_json(occlusion_path), f"{case} occlusion candidates")
    visibility = require_dict(load_json(visibility_path), f"{case} visibility state")
    surface_report = require_dict(load_json(surface_path), f"{case} visible surface report")
    hawor = load_hawor_case(hawor_path)
    hands = hand_depth_index(visibility)
    surfaces = object_surface_index(surface_report)
    row_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    row_state_counts: Counter[str] = Counter()
    pair_state_counts: Counter[str] = Counter()
    object_pair_state_counts: Counter[str] = Counter()
    for raw in require_list(occlusion.get("row_records"), "occlusion row_records"):
        row = require_dict(raw, "occlusion candidate row")
        frame_idx = require_int(row.get("frame_idx"), "row frame_idx")
        hand_side = require_str(row.get("hand_side"), "row hand_side")
        candidates = [require_dict(candidate, "candidate object") for candidate in require_list(row.get("candidate_objects"), "candidate objects")]
        hand = hands.get((frame_idx, hand_side))
        row_pairs: list[dict[str, Any]] = []
        row_pair_states: Counter[str] = Counter()
        for candidate in candidates:
            object_id = require_str(candidate.get("object_id"), "candidate object_id")
            surface_row = surfaces.get((frame_idx, object_id))
            pair = candidate_pair_record(row, candidate, hand, surface_row, hawor)
            row_pairs.append(pair)
            pair_records.append(pair)
            state = require_str(pair.get("depth_evidence_state"), "depth_evidence_state")
            row_pair_states[state] += 1
            pair_state_counts[state] += 1
            object_pair_state_counts[f"{object_id}|{state}"] += 1
        state = row_depth_state(row_pair_states, len(candidates))
        row_state_counts[state] += 1
        row_depth_resolved = any(pair.get("depth_order_resolved") is True for pair in row_pairs)
        row_owner_accepted = any(pair.get("occluder_owner_accepted") is True for pair in row_pairs)
        row_records.append(
            {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "source_candidate_state": row.get("candidate_state"),
                "source_candidate_count": row.get("candidate_count"),
                "hand_metric_depth_state": hand.get("metric_depth_state") if hand is not None else "missing_v18_hand_depth_row",
                "row_depth_evidence_state": state,
                "candidate_pair_count": len(row_pairs),
                "candidate_pair_depth_state_counts": dict(sorted(row_pair_states.items())),
                "candidate_pair_depth_evidence": row_pairs,
                "depth_order_resolved": row_depth_resolved,
                "occluder_owner_accepted": row_owner_accepted,
                "pose_filled_through_occlusion": False,
            }
        )
    same_frame_surface_pair_count = sum(1 for pair in pair_records if pair.get("object_surface_depth_available") is True)
    hawor_state_counts: Counter[str] = Counter()
    for pair in pair_records:
        evidence = pair.get("hawor_mano_depth_order_evidence") if isinstance(pair, dict) else None
        if isinstance(evidence, dict):
            hawor_state_counts[str(evidence.get("state"))] += 1
    report = {
        "method": "build_v18_occlusion_depth_order_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_occlusion_owner_candidates": str(occlusion_path),
            "v18_visibility_occlusion_state": str(visibility_path),
            "v17_visible_surface_report": str(surface_path),
            "support_aware_hawor_metric_mano_npz": str(hawor_path) if hawor_path is not None else None,
        },
        "unresolved_hand_row_count": len(row_records),
        "candidate_owner_row_count": require_int(occlusion.get("candidate_owner_row_count"), "candidate owner row count"),
        "candidate_pair_count": len(pair_records),
        "same_frame_object_surface_pair_count": same_frame_surface_pair_count,
        "candidate_row_depth_evidence_state_counts": dict(sorted(row_state_counts.items())),
        "candidate_pair_depth_evidence_state_counts": dict(sorted(pair_state_counts.items())),
        "object_pair_depth_evidence_state_counts": dict(sorted(object_pair_state_counts.items())),
        "hawor_mano_depth_order_state_counts": dict(sorted(hawor_state_counts.items())),
        "hawor_mano_foreground_support_pair_count": hawor_state_counts.get(HAWOR_SUPPORT_STATE, 0),
        "hawor_mano_foreground_contradiction_pair_count": hawor_state_counts.get(HAWOR_CONTRADICTION_STATE, 0),
        "hawor_mano_metric_compatible_pair_count": hawor_state_counts.get(HAWOR_COMPATIBLE_STATE, 0),
        "foreground_occluder_support_pair_count": pair_state_counts.get(SUPPORT_STATE, 0),
        "foreground_occluder_contradiction_pair_count": pair_state_counts.get(CONTRADICTION_STATE, 0),
        "metric_compatible_no_foreground_signal_pair_count": pair_state_counts.get(METRIC_COMPATIBLE_STATE, 0),
        "insufficient_object_surface_depth_pair_count": pair_state_counts.get(INSUFFICIENT_OBJECT_STATE, 0),
        "insufficient_or_untrusted_hand_depth_pair_count": pair_state_counts.get(INSUFFICIENT_HAND_STATE, 0),
        "occluder_owner_accepted_count": sum(1 for row in row_records if row.get("occluder_owner_accepted") is True),
        "depth_order_resolved_count": sum(1 for row in row_records if row.get("depth_order_resolved") is True),
        "pose_filled_through_occlusion_rows": 0,
        "acceptance_policy": "pair_depth_order_support_is_input_evidence_only_final_owner_requires_temporal_graph_selection_and_observed_hawor_support",
        "row_records": row_records,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_occlusion_depth_order_evidence_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    row_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    hawor_counts: Counter[str] = Counter()
    for report in reports:
        row_counts.update(require_dict(report.get("candidate_row_depth_evidence_state_counts"), "row depth counts"))
        pair_counts.update(require_dict(report.get("candidate_pair_depth_evidence_state_counts"), "pair depth counts"))
        object_counts.update(require_dict(report.get("object_pair_depth_evidence_state_counts"), "object depth counts"))
        hawor_counts.update(require_dict(report.get("hawor_mano_depth_order_state_counts"), "hawor depth counts"))
    summary = {
        "method": "build_v18_occlusion_depth_order_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "unresolved_hand_row_count": sum(require_int(report.get("unresolved_hand_row_count"), "unresolved hand rows") for report in reports),
        "candidate_owner_row_count": sum(require_int(report.get("candidate_owner_row_count"), "candidate owner rows") for report in reports),
        "candidate_pair_count": sum(require_int(report.get("candidate_pair_count"), "candidate pair count") for report in reports),
        "same_frame_object_surface_pair_count": sum(require_int(report.get("same_frame_object_surface_pair_count"), "surface pair count") for report in reports),
        "candidate_row_depth_evidence_state_counts": dict(sorted(row_counts.items())),
        "candidate_pair_depth_evidence_state_counts": dict(sorted(pair_counts.items())),
        "object_pair_depth_evidence_state_counts": dict(sorted(object_counts.items())),
        "hawor_mano_depth_order_state_counts": dict(sorted(hawor_counts.items())),
        "hawor_mano_foreground_support_pair_count": sum(require_int(report.get("hawor_mano_foreground_support_pair_count"), "hawor support pairs") for report in reports),
        "hawor_mano_foreground_contradiction_pair_count": sum(require_int(report.get("hawor_mano_foreground_contradiction_pair_count"), "hawor contradiction pairs") for report in reports),
        "hawor_mano_metric_compatible_pair_count": sum(require_int(report.get("hawor_mano_metric_compatible_pair_count"), "hawor compatible pairs") for report in reports),
        "foreground_occluder_support_pair_count": sum(require_int(report.get("foreground_occluder_support_pair_count"), "support pairs") for report in reports),
        "foreground_occluder_contradiction_pair_count": sum(require_int(report.get("foreground_occluder_contradiction_pair_count"), "contradiction pairs") for report in reports),
        "metric_compatible_no_foreground_signal_pair_count": sum(require_int(report.get("metric_compatible_no_foreground_signal_pair_count"), "metric compatible pairs") for report in reports),
        "insufficient_object_surface_depth_pair_count": sum(require_int(report.get("insufficient_object_surface_depth_pair_count"), "insufficient object pairs") for report in reports),
        "insufficient_or_untrusted_hand_depth_pair_count": sum(require_int(report.get("insufficient_or_untrusted_hand_depth_pair_count"), "insufficient hand pairs") for report in reports),
        "occluder_owner_accepted_count": sum(require_int(report.get("occluder_owner_accepted_count"), "accepted occluder owners") for report in reports),
        "depth_order_resolved_count": sum(require_int(report.get("depth_order_resolved_count"), "depth order resolved") for report in reports),
        "pose_filled_through_occlusion_rows": 0,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_occlusion_depth_order_evidence_report.json"),
                "candidate_pair_count": report["candidate_pair_count"],
                "same_frame_object_surface_pair_count": report["same_frame_object_surface_pair_count"],
                "candidate_pair_depth_evidence_state_counts": report["candidate_pair_depth_evidence_state_counts"],
                "hawor_mano_depth_order_state_counts": report["hawor_mano_depth_order_state_counts"],
                "hawor_mano_foreground_support_pair_count": report["hawor_mano_foreground_support_pair_count"],
                "foreground_occluder_support_pair_count": report["foreground_occluder_support_pair_count"],
                "foreground_occluder_contradiction_pair_count": report["foreground_occluder_contradiction_pair_count"],
                "occluder_owner_accepted_count": report["occluder_owner_accepted_count"],
                "depth_order_resolved_count": report["depth_order_resolved_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_occlusion_depth_order_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occlusion-owner-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_candidates"))
    parser.add_argument("--visibility-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--visible-surface-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"))
    parser.add_argument("--trash-hawor-npz", type=Path, default=DEFAULT_HAWOR_OUTPUTS["trash_1050"])
    parser.add_argument("--task5-hawor-npz", type=Path, default=DEFAULT_HAWOR_OUTPUTS["task5_tomato_960"])
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
