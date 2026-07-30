#!/usr/bin/env python3
"""Build a full-timeline V18 corrective annotation-state delta.

This converts the corrective V18 mechanisms from visual-only attempts into a
machine-readable annotation artifact. It does not duplicate the full V18 source
annotation and does not claim full V18 closure. It records only states actually
changed or attempted by the corrective work: factor-graph-driven hand state,
graph-shifted MANO projections, gated temporal MANO2D filtering, generic rigid
SE(3) object priors, visible-surface/residual diagnostics, HaWoR prior/provisioning
status, tentative occlusion ownership, contact/nonpenetration evidence, and V16-local
nonpenetration translation candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits) if math.isfinite(value) else None
    if isinstance(value, np.floating):
        x = float(value)
        return round(x, ndigits) if math.isfinite(x) else None
    if isinstance(value, list):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return [rounded(v, ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), ndigits)
    return value


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_center(value: Any) -> tuple[float, float] | None:
    box = bbox_tuple(value)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def shift_box_to_center(box: tuple[float, float, float, float], center: tuple[float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx, cy = center
    return cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h


def project_mano_joints_source_px(mano: dict[str, Any], source_w: float, source_h: float) -> list[tuple[float, float]]:
    joints = mano.get("joints3d_camera")
    cam_t = mano.get("cam_t")
    intr = mano.get("source_intrinsics") or [2304.0, 2304.0, source_w / 2.0, source_h / 2.0]
    if not (isinstance(joints, list) and isinstance(cam_t, list) and len(cam_t) == 3 and isinstance(intr, list) and len(intr) == 4):
        return []
    fx, fy, cx, cy = [finite_float(v) for v in intr]
    pts: list[tuple[float, float]] = []
    for raw in joints:
        if not (isinstance(raw, list) and len(raw) == 3):
            return []
        x = finite_float(raw[0]) + finite_float(cam_t[0])
        y = finite_float(raw[1]) + finite_float(cam_t[1])
        z = finite_float(raw[2]) + finite_float(cam_t[2])
        if z <= 1e-6:
            return []
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not (math.isfinite(u) and math.isfinite(v)):
            return []
        pts.append((u, v))
    return pts


def graph_hand_estimates(frame: dict[str, Any], source_w: float, source_h: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("hand_state", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("hand::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 2:
            side = vid.split("::", 1)[1]
            out[side] = {
                "variable_id": vid,
                "estimate_normalized_xy": [finite_float(est[0]), finite_float(est[1])],
                "center_source_px": [finite_float(est[0]) * source_w, finite_float(est[1]) * source_h],
                "source": row.get("source"),
                "observation_residual_norm": row.get("observation_residual_norm"),
            }
    return out


def graph_object_poses(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    rows = frame.get("factor_graph_solution", {}).get("variables", {}).get("object_se3", [])
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        vid = str(row.get("variable_id", ""))
        if not vid.startswith("object_se3::"):
            continue
        est = row.get("estimate")
        if isinstance(est, list) and len(est) >= 6:
            oid = vid.split("::", 1)[1]
            out[oid] = {
                "variable_id": vid,
                "pose6_world_from_object": [finite_float(v) for v in est[:6]],
                "source": row.get("source"),
                "observation_residual_norm": row.get("observation_residual_norm"),
            }
    return out


def load_hawor_measurement_index(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], list[Path]]:
    if not path.exists():
        return {}, []
    rows = load_json(path)
    index: dict[tuple[int, str], dict[str, Any]] = {}
    sources: set[Path] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        entity = str(row.get("entity_id", ""))
        if not entity.startswith("hand:"):
            continue
        frame_idx = int(row.get("frame_idx", -1))
        side = entity.split(":", 1)[1]
        index[(frame_idx, side)] = row
        src = row.get("source_annotation")
        if isinstance(src, str) and Path(src).exists():
            sources.add(Path(src))
    return index, sorted(sources)


def load_hawor_source_hands(paths: list[Path]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        frames = payload.get("frames", [])
        for raw_frame in frames if isinstance(frames, list) else []:
            if not isinstance(raw_frame, dict):
                continue
            frame_idx = int(raw_frame.get("frame_idx", -1))
            for hand in raw_frame.get("hands", []):
                if not isinstance(hand, dict) or hand.get("backend") != "HaWoR":
                    continue
                side = str(hand.get("hand_side") or hand.get("side"))
                if side in {"left", "right"}:
                    out[(frame_idx, side)] = hand
    return out


def hawor_bridge_quality_index(path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {}
    report = load_json(path)
    rows = report.get("quality_rows", []) if isinstance(report.get("quality_rows"), list) else []
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        side = str(row.get("side"))
        if side not in {"left", "right"}:
            continue
        out[(int(row.get("frame_idx", -1)), side)] = row
    return out, report


def rigid_candidates_from_report(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    report = load_json(path)
    candidates = report.get("candidate_objects", {})
    return {str(k): v for k, v in candidates.items() if isinstance(v, dict)} if isinstance(candidates, dict) else {}


def visible_surface_row_index(report_path: Path, candidate_ids: set[str]) -> tuple[dict[tuple[int, str], dict[str, Any]], str | None]:
    if not report_path.exists():
        return {}, None
    report = load_json(report_path)
    rows = report.get("surface_archive_rows", [])
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        oid = str(row.get("object_id"))
        if oid in candidate_ids:
            out[(int(row.get("frame_idx", -1)), oid)] = row
    archive_npz = report.get("archive_npz")
    return out, str(archive_npz) if isinstance(archive_npz, str) else None


def selected_occlusion_owner_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        if not isinstance(hand_graph, dict):
            continue
        for row in hand_graph.get("assignments", []):
            if not isinstance(row, dict):
                continue
            owner = row.get("chosen_owner_object_id")
            if isinstance(owner, str) and owner.startswith("object:"):
                out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return out, report


def occlusion_acceptance_audit_index(report_path: Path) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))].append(row)
    return out, report


def selected_contact_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for hand_graph in report.get("hand_graphs", []) if isinstance(report.get("hand_graphs"), list) else []:
        if not isinstance(hand_graph, dict):
            continue
        for row in hand_graph.get("assignments", []):
            if not isinstance(row, dict):
                continue
            owner = row.get("chosen_owner_object_id")
            if isinstance(owner, str) and owner.startswith("object:"):
                out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return out, report


def contact_acceptance_audit_index(report_path: Path) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))].append(row)
    return out, report


def nonpenetration_row_index(report_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")), str(row.get("object_id")))] = row
    return out


def rigid_residual_row_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("residual_rows", []) if isinstance(report.get("residual_rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("object_id")))] = row
    return out, report


def nonpenetration_repair_row_index(report_path: Path) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in report.get("rows", []) if isinstance(report.get("rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")), str(row.get("object_id")))] = row
    return out, report


def temporal_smoothed_hand_row_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {}
    report = load_json(report_path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in report.get("smoothed_rows", []) if isinstance(report.get("smoothed_rows"), list) else []:
        if isinstance(row, dict):
            out[(int(row.get("frame_idx", -1)), str(row.get("hand_side")))] = row
    return out, report


def geometry_coverage_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    report = load_json(report_path)
    return report if isinstance(report, dict) else {}


def stable_rigid_pose_index(frames: list[Any], candidate_ids: set[str], radius: int) -> dict[tuple[int, str], list[float]]:
    raw: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", 0))
        poses = graph_object_poses(frame)
        for oid in candidate_ids:
            pose = poses.get(oid)
            if pose is not None:
                raw[oid].append((frame_idx, np.array(pose["pose6_world_from_object"], dtype=np.float64)))
    stable: dict[tuple[int, str], list[float]] = {}
    for oid, rows in raw.items():
        ordered = sorted(rows, key=lambda x: x[0])
        if not ordered:
            continue
        translations = [pose[:3] for _frame_idx, pose in ordered]
        median_rot = np.median(np.stack([pose[3:6] for _frame_idx, pose in ordered], axis=0), axis=0)
        for i, (frame_idx, pose6) in enumerate(ordered):
            lo = max(0, i - radius)
            hi = min(len(ordered), i + radius + 1)
            stable_pose = pose6.copy()
            stable_pose[:3] = np.mean(np.stack(translations[lo:hi], axis=0), axis=0)
            stable_pose[3:6] = median_rot
            stable[(frame_idx, oid)] = stable_pose.tolist()
    return stable


def hand_corrective_state(
    frame_idx: int,
    hand: dict[str, Any],
    graph_est: dict[str, Any] | None,
    hawor_row: dict[str, Any] | None,
    hawor_hand: dict[str, Any] | None,
    hawor_available_for_case: bool,
    occlusion_owner_row: dict[str, Any] | None,
    occlusion_acceptance_rows: list[dict[str, Any]],
    contact_row: dict[str, Any] | None,
    contact_acceptance_rows: list[dict[str, Any]],
    signed_nonpenetration_row: dict[str, Any] | None,
    triangle_nonpenetration_row: dict[str, Any] | None,
    nonpenetration_repair_row: dict[str, Any] | None,
    temporal_smoothed_hand_row: dict[str, Any] | None,
    hawor_bridge_quality_row: dict[str, Any] | None,
    source_w: float,
    source_h: float,
) -> dict[str, Any]:
    side = str(hand.get("hand_side") or hand.get("side"))
    bbox = bbox_tuple(hand.get("bbox_xyxy"))
    obs_center = bbox_center(hand.get("bbox_xyxy"))
    out: dict[str, Any] = {
        "hand_side": side,
        "visibility_state": hand.get("visibility_state"),
        "source_bbox_xyxy": rounded(list(bbox) if bbox else None, 3),
        "source_confidence": hand.get("confidence"),
        "best_current_state": "source_visible_hand_no_graph_estimate",
        "uncertainty": [],
    }
    if graph_est is not None:
        center = tuple(float(v) for v in graph_est["center_source_px"])
        graph: dict[str, Any] = {
            "variable_id": graph_est.get("variable_id"),
            "center_source_px": rounded(center, 3),
            "estimate_normalized_xy": rounded(graph_est.get("estimate_normalized_xy"), 6),
            "source": graph_est.get("source"),
            "observation_residual_norm": graph_est.get("observation_residual_norm"),
            "state_role": "render_driver_best_current_image_space_hand_state",
        }
        if bbox is not None:
            shifted_bbox = shift_box_to_center(bbox, center)
            graph["shifted_bbox_xyxy"] = rounded(shifted_bbox, 3)
        if obs_center is not None:
            dx = center[0] - obs_center[0]
            dy = center[1] - obs_center[1]
            graph["center_shift_from_observation_px"] = rounded([dx, dy], 3)
            graph["center_residual_px"] = round(math.hypot(dx, dy), 3)
            mano = hand.get("mano_candidate", {}) if isinstance(hand.get("mano_candidate"), dict) else {}
            pts = project_mano_joints_source_px(mano, source_w, source_h)
            if len(pts) >= 21:
                graph["shifted_mano_joints2d_source_px"] = rounded([(x + dx, y + dy) for x, y in pts[:21]], 2)
                graph["mano_state_scope"] = "source_mano_projection_shifted_by_graph_center_not_solved_articulation"
            else:
                out["uncertainty"].append("no_projectable_mano_candidate_for_graph_shifted_skeleton")
        out["graph_hand_state"] = graph
        out["best_current_state"] = "graph_shifted_mano_if_available_else_graph_shifted_bbox"
    else:
        out["uncertainty"].append("missing_factor_graph_hand_state")

    if hawor_bridge_quality_row is not None:
        quality_state = str(hawor_bridge_quality_row.get("quality_state"))
        out["hawor_bridge_quality_candidate"] = {
            "status": quality_state,
            "candidate_source": "HaWoR_current_V18_camera_local_bridge",
            "projection_residual_px_median": hawor_bridge_quality_row.get("projection_residual_px_median"),
            "projection_residual_px_p95": hawor_bridge_quality_row.get("projection_residual_px_p95"),
            "current_visibility_state": hawor_bridge_quality_row.get("current_visibility_state"),
            "hawor_projected_inside_image_fraction": hawor_bridge_quality_row.get("hawor_projected_inside_image_fraction"),
            "reference_projected_inside_image_fraction": hawor_bridge_quality_row.get("reference_projected_inside_image_fraction"),
            "hawor_projected_inside_current_bbox_fraction": hawor_bridge_quality_row.get("hawor_projected_inside_current_bbox_fraction"),
            "reference_projection_source_family": hawor_bridge_quality_row.get("reference_projection_source_family"),
            "reference_projection_source_backend": hawor_bridge_quality_row.get("reference_projection_source_backend"),
            "quality_blockers": hawor_bridge_quality_row.get("quality_blockers") if isinstance(hawor_bridge_quality_row.get("quality_blockers"), list) else [],
            "accepted_v18_hawor_foundation": False,
            "accepted_metric_hand_state": False,
            "accepted_contact_or_occlusion_input": False,
            "state_role": "HaWoR_bridge_candidate_quality_evidence_not_foundation_acceptance_not_downstream_physics",
        }
        if quality_state.startswith("projection_supported"):
            out["uncertainty"].append("hawor_bridge_projection_supported_candidate_not_foundation_accepted")
        else:
            out["uncertainty"].append("hawor_bridge_candidate_not_projection_supported_for_physical_use")

    if temporal_smoothed_hand_row is not None:
        out["temporal_smoothed_mano2d_state"] = {
            "status": temporal_smoothed_hand_row.get("status"),
            "joints2d_source_px": temporal_smoothed_hand_row.get("joints2d_source_px"),
            "temporal_filter_applied": temporal_smoothed_hand_row.get("temporal_filter_applied"),
            "reject_reasons": temporal_smoothed_hand_row.get("reject_reasons"),
            "max_joint_shift_from_graph_shifted_input_px": temporal_smoothed_hand_row.get("max_joint_shift_from_graph_shifted_input_px"),
            "centroid_shift_from_graph_shifted_input_px": temporal_smoothed_hand_row.get("centroid_shift_from_graph_shifted_input_px"),
            "root_shift_from_graph_shifted_input_px": temporal_smoothed_hand_row.get("root_shift_from_graph_shifted_input_px"),
            "candidate_out_of_source_frame_joint_count": temporal_smoothed_hand_row.get("candidate_out_of_source_frame_joint_count"),
            "raw_out_of_source_frame_joint_count": temporal_smoothed_hand_row.get("raw_out_of_source_frame_joint_count"),
            "output_out_of_source_frame_joint_count": temporal_smoothed_hand_row.get("output_out_of_source_frame_joint_count"),
            "accepted_3d_mano_pose": False,
            "state_role": temporal_smoothed_hand_row.get("state_role") or "image_space_temporal_filter_with_anchor_and_bounds_gate_not_3d_mano_optimization",
        }
        out["uncertainty"].append("temporal_smoothed_mano2d_is_not_3d_mano_optimization_or_physical_pose_or_best_current_state")

    if hawor_row is not None:
        prior: dict[str, Any] = {
            "status": "available_uncertain_prior",
            "measurement_id": hawor_row.get("measurement_id"),
            "evidence_role": hawor_row.get("evidence_role"),
            "measurement_available": hawor_row.get("measurement_available"),
            "visibility_state": hawor_row.get("visibility_state"),
            "coordinate_frame": hawor_row.get("coordinate_frame"),
            "projection_residual_px_median": hawor_row.get("projection_residual_px_median"),
            "projection_residual_px_p95": hawor_row.get("projection_residual_px_p95"),
            "bbox_xyxy": rounded(hawor_row.get("bbox_xyxy"), 3),
            "source_annotation": hawor_row.get("source_annotation"),
            "state_role": "uncertain_temporal_motion_prior_not_accepted_occlusion_solution",
        }
        if hawor_hand is not None and isinstance(hawor_hand.get("joints2d"), list):
            prior["joints2d_source_px"] = rounded(hawor_hand.get("joints2d")[:21], 2)
        out["hawor_temporal_prior"] = prior
        if "infill" in str(hawor_row.get("evidence_role")):
            out["pose_fill_best_effort"] = {
                "status": "hawor_motion_infill_candidate_not_accepted_pose_fill",
                "source": "HaWoR",
                "measurement_id": hawor_row.get("measurement_id"),
                "evidence_role": hawor_row.get("evidence_role"),
                "accepted_pose_fill": False,
                "uncertainty": "temporal_motion_prior_without_accepted_occlusion_owner_or_depth_order",
                "state_role": "no_gating_uncertain_pose_fill_evidence_not_solution",
            }
            if hawor_hand is not None and isinstance(hawor_hand.get("joints2d"), list):
                out["pose_fill_best_effort"]["joints2d_source_px"] = rounded(hawor_hand.get("joints2d")[:21], 2)
    elif not hawor_available_for_case:
        out["hawor_temporal_prior"] = {"status": "provisioning_failed_no_case_measurements", "state_role": "required_baseline_missing_execution_not_silent_absence"}
        out["uncertainty"].append("hawor_not_executed_or_not_provisioned_for_case")
    else:
        out["hawor_temporal_prior"] = {"status": "not_in_hawor_measurement_window"}
    if occlusion_owner_row is not None:
        source_row = occlusion_owner_row.get("source_row", {}) if isinstance(occlusion_owner_row.get("source_row"), dict) else {}
        out["occlusion_owner_best_effort"] = {
            "status": "temporal_graph_selected_not_strictly_accepted",
            "chosen_owner_object_id": occlusion_owner_row.get("chosen_owner_object_id"),
            "chosen_unary_energy": occlusion_owner_row.get("chosen_unary_energy"),
            "next_best_unary_energy": occlusion_owner_row.get("next_best_unary_energy"),
            "unary_energy_margin": occlusion_owner_row.get("unary_energy_margin"),
            "accepted_occlusion_owner": False,
            "acceptance_blockers": occlusion_owner_row.get("acceptance_blockers"),
            "depth_pair_evidence_state": occlusion_owner_row.get("depth_pair_evidence_state"),
            "mesh_temporal_support": source_row.get("mesh_contact_temporal_support"),
            "state_role": "best_current_tentative_owner_with_blockers_not_pose_fill_acceptance",
        }
    if occlusion_acceptance_rows:
        audit_states = []
        for row in occlusion_acceptance_rows:
            audit_states.append({
                "object_id": row.get("object_id"),
                "category": row.get("category"),
                "strict_promotable_owner": bool(row.get("strict_promotable_owner")),
                "accepted_occlusion_owner": bool(row.get("accepted_occlusion_owner")),
                "selected_by_temporal_graph": bool(row.get("selected_by_temporal_graph")),
                "exact_foreground_depth_support": bool(row.get("exact_foreground_depth_support")),
                "same_frame_foreground_contradiction_count": row.get("same_frame_foreground_contradiction_count"),
                "mesh_temporal_support": row.get("mesh_temporal_support"),
                "temporal_graph_margin": row.get("temporal_graph_margin"),
                "source_depth_order_resolved": bool(row.get("source_depth_order_resolved")),
                "source_occluder_owner_accepted": bool(row.get("source_occluder_owner_accepted")),
                "depth_pair_evidence_state": row.get("depth_pair_evidence_state"),
                "acceptance_blockers": row.get("acceptance_blockers") if isinstance(row.get("acceptance_blockers"), list) else [],
                "evidence_scope": row.get("evidence_scope"),
                "state_role": "occlusion_owner_acceptance_audit_not_assignment_not_pose_fill",
            })
        out["occlusion_owner_acceptance_audit"] = audit_states
        out["uncertainty"].append("occlusion_owner_acceptance_audit_does_not_assign_owner_or_pose_fill")
    if contact_acceptance_rows:
        audit_states = []
        for row in contact_acceptance_rows:
            audit_states.append({
                "object_id": row.get("object_id"),
                "category": row.get("category"),
                "strict_promotable_contact": bool(row.get("strict_promotable_contact")),
                "source_graph_contact_candidate_before_physical_veto": bool(row.get("source_graph_contact_candidate_before_physical_veto")),
                "source_graph_contact_candidate_before_physical_veto": bool(row.get("source_graph_contact_candidate_before_physical_veto")),
                "contact_owner_claim_context": row.get("contact_owner_claim_context"),
                "signed_local_penetration_detected": bool(row.get("signed_local_penetration_detected")),
                "triangle_local_penetration_detected": bool(row.get("triangle_local_penetration_detected")),
                "triangle_mesh_watertight_by_edges": bool(row.get("triangle_mesh_watertight_by_edges")),
                "signed_complete": bool(row.get("signed_complete")),
                "triangle_complete": bool(row.get("triangle_complete")),
                "signed_min_local_distance_m": row.get("signed_min_local_distance_m"),
                "triangle_min_local_distance_m": row.get("triangle_min_local_distance_m"),
                "triangle_boundary_edge_count": row.get("triangle_boundary_edge_count"),
                "state_role": "contact_acceptance_audit_not_contact_assignment_not_complete_nonpenetration",
            })
        out["contact_acceptance_audit"] = audit_states
        out["uncertainty"].append("contact_acceptance_audit_does_not_assign_contact_or_complete_nonpenetration")
    if contact_row is not None:
        oid = str(contact_row.get("chosen_owner_object_id"))
        signed_pen = bool(signed_nonpenetration_row and signed_nonpenetration_row.get("local_penetration_detected"))
        tri_pen = bool(triangle_nonpenetration_row and triangle_nonpenetration_row.get("local_triangle_penetration_detected"))
        if not contact_row.get("accepted_contact_owner"):
            status = "graph_selected_not_accepted"
        elif signed_pen or tri_pen:
            status = "source_graph_candidate_but_local_penetration_veto"
        else:
            status = "source_graph_candidate_no_local_penetration_flag"
        out["contact_nonpenetration_state"] = {
            "status": status,
            "chosen_contact_object_id": oid,
            "source_graph_contact_candidate_before_nonpenetration_veto": bool(contact_row.get("accepted_contact_owner")),
            "min_hand_surface_to_object_mesh_m": contact_row.get("min_hand_surface_to_object_mesh_m"),
            "unary_energy_margin": contact_row.get("unary_energy_margin"),
            "source_row_blockers": contact_row.get("source_row_blockers"),
            "signed_nonpenetration": {
                "available": signed_nonpenetration_row is not None,
                "complete": False,
                "local_penetration_detected": signed_nonpenetration_row.get("local_penetration_detected") if signed_nonpenetration_row else None,
                "min_local_signed_distance_m": signed_nonpenetration_row.get("min_local_signed_distance_m") if signed_nonpenetration_row else None,
                "semantics": signed_nonpenetration_row.get("local_signed_distance_semantics") if signed_nonpenetration_row else None,
            },
            "triangle_nonpenetration": {
                "available": triangle_nonpenetration_row is not None,
                "complete": False,
                "mesh_watertight_by_edges": triangle_nonpenetration_row.get("mesh_watertight_by_edges") if triangle_nonpenetration_row else None,
                "local_triangle_penetration_detected": triangle_nonpenetration_row.get("local_triangle_penetration_detected") if triangle_nonpenetration_row else None,
                "min_local_triangle_signed_distance_m": triangle_nonpenetration_row.get("min_local_triangle_signed_distance_m") if triangle_nonpenetration_row else None,
                "semantics": triangle_nonpenetration_row.get("local_triangle_signed_distance_semantics") if triangle_nonpenetration_row else None,
            },
            "state_role": "contact_graph_selection_with_local_nonpenetration_evidence_not_complete_sdf_solution",
        }
        if nonpenetration_repair_row is not None:
            out["nonpenetration_repair_proposal"] = {
                "status": nonpenetration_repair_row.get("status"),
                "proposed_translation_world_m": rounded(nonpenetration_repair_row.get("proposed_translation_world_m"), 6),
                "proposed_translation_norm_m": nonpenetration_repair_row.get("proposed_translation_norm_m"),
                "penetrated_point_fraction": nonpenetration_repair_row.get("penetrated_point_fraction"),
                "penetration_normal_alignment": nonpenetration_repair_row.get("penetration_normal_alignment"),
                "post_translation_local_check_status": nonpenetration_repair_row.get("post_translation_local_check_status"),
                "post_translation_min_signed_m": nonpenetration_repair_row.get("post_translation_min_signed_m"),
                "post_translation_penetrated_point_count": nonpenetration_repair_row.get("post_translation_penetrated_point_count"),
                "post_translation_local_metric_passed": nonpenetration_repair_row.get("post_translation_local_metric_passed"),
                "proposal_complete_nonpenetration": False,
                "applied_to_annotation": False,
                "diagnostic_geometry_basis": nonpenetration_repair_row.get("diagnostic_geometry_basis"),
                "source_contact_owner_claim_context": nonpenetration_repair_row.get("source_contact_owner_claim_context"),
                "semantics": nonpenetration_repair_row.get("semantics"),
                "state_role": "diagnostic_v16_local_translation_candidate_not_applied_not_complete_sdf_not_current_v18_hand_state",
            }
    return out


def object_corrective_state(frame_idx: int, obj: dict[str, Any], graph_pose: dict[str, Any] | None, rigid_candidate: dict[str, Any] | None, stable_pose: list[float] | None, visible_surface_row: dict[str, Any] | None, residual_row: dict[str, Any] | None) -> dict[str, Any]:
    oid = str(obj.get("object_id"))
    out: dict[str, Any] = {
        "object_id": oid,
        "name": obj.get("name"),
        "physical_state_candidate": obj.get("physical_state_candidate"),
        "visibility_state": obj.get("visibility_state"),
        "source_bbox_xyxy": rounded(obj.get("bbox_xyxy"), 3),
        "best_current_state": "source_object_state_no_graph_pose",
        "uncertainty": [],
    }
    if graph_pose is not None:
        graph_source = str(graph_pose.get("source"))
        out["graph_object_se3"] = {
            "variable_id": graph_pose.get("variable_id"),
            "pose6_world_from_object": rounded(graph_pose.get("pose6_world_from_object"), 6),
            "source": graph_pose.get("source"),
            "observation_residual_norm": graph_pose.get("observation_residual_norm"),
            "accepted_physical_object_pose": False,
            "state_role": "factor_graph_object_pose_observation_not_accepted_physical_pose",
        }
        out["uncertainty"].append("graph_object_se3_is_visible_surface_observation_not_accepted_pose")
        if obj.get("physical_state_candidate") != "rigid":
            out["uncertainty"].append("non_rigid_or_unknown_object_state_not_validated_for_rigid_se3_pose")
            out["best_current_state"] = "approximate_visible_surface_pose_observation_not_physical_pose"
        elif "centroid" in graph_source or "pca" in graph_source:
            out["uncertainty"].append("object_se3_source_is_centroid_pca_proxy")
            out["best_current_state"] = "approximate_visible_surface_pose_observation_not_accepted_pose"
        else:
            out["best_current_state"] = "graph_object_se3_observation_uncertain_not_accepted_pose"
    else:
        out["uncertainty"].append("missing_factor_graph_object_se3_for_frame")
    if visible_surface_row is not None:
        out["frame_local_visible_surface_state"] = {
            "status": "available_best_visible_geometry_evidence",
            "vertex_count": visible_surface_row.get("vertex_count"),
            "face_count": visible_surface_row.get("face_count"),
            "center_world_m": rounded(visible_surface_row.get("center_world_m"), 6),
            "bbox_world_min_m": rounded(visible_surface_row.get("bbox_world_min_m"), 6),
            "bbox_world_max_m": rounded(visible_surface_row.get("bbox_world_max_m"), 6),
            "world_extent_m": rounded(visible_surface_row.get("world_extent_m"), 6),
            "mask_path": visible_surface_row.get("mask_path"),
            "state_role": "frame_local_rgbd_visible_surface_not_hidden_completion",
        }
        if out["best_current_state"] == "source_object_state_no_graph_pose":
            out["best_current_state"] = "frame_local_visible_surface_geometry"
    if rigid_candidate is not None:
        attempt: dict[str, Any] = {
            "status": "candidate_selected_by_generic_metadata",
            "selection_reasons": rigid_candidate.get("selection_reasons"),
            "model_physical_state_type": rigid_candidate.get("model_physical_state_type"),
            "fast_motion_state": rigid_candidate.get("fast_motion_state"),
            "fused_point_cloud_path": rigid_candidate.get("fused_point_cloud_path"),
            "hidden_geometry_status": rigid_candidate.get("hidden_geometry_status"),
            "object_geometry_complete": False,
            "stable_rigid_prior_method": "componentwise_median_rotation_vector_plus_local_mean_translation_no_object_name_branch",
            "state_role": "generic_rigid_se3_render_driver_attempt",
        }
        out["uncertainty"].append("rigid_candidate_geometry_not_complete")
        if str(rigid_candidate.get("model_physical_state_type")) != "rigid":
            out["uncertainty"].append("rigid_candidate_selected_by_motion_metadata_not_model_rigid_state")
        if stable_pose is not None:
            attempt["stable_pose6_world_from_object"] = rounded(stable_pose, 6)
            attempt["status"] = "stable_pose_available_uncertain_render_driver"
            out["best_current_state"] = "uncertain_rigid_prior_with_frame_local_visible_surface_when_available"
        else:
            attempt["status"] = "selected_but_no_graph_pose_this_frame"
            out["uncertainty"].append("rigid_candidate_without_pose_this_frame")
        if residual_row is not None:
            residual_status = str(residual_row.get("status"))
            attempt["residual_check"] = {
                "status": residual_status,
                "visible_to_fused_median_m": residual_row.get("visible_to_fused_median_m"),
                "visible_to_fused_p95_m": residual_row.get("visible_to_fused_p95_m"),
                "fused_to_visible_p95_m": residual_row.get("fused_to_visible_p95_m"),
                "thresholds_m": residual_row.get("thresholds_m"),
                "state_role": "bidirectional_residual_check_not_pose_acceptance",
            }
            if residual_status == "visible_supported_but_fused_overspread":
                out["best_current_state"] = "frame_local_visible_surface_preferred_fused_geometry_overspread"
                out["uncertainty"].append("fused_canonical_geometry_overspread_relative_to_visible_surface")
            elif residual_status == "visible_surface_not_explained_by_fused_pose":
                out["best_current_state"] = "frame_local_visible_surface_preferred_rigid_pose_residual_rejected"
                out["uncertainty"].append("rigid_pose_residual_rejected_by_visible_surface")
            elif residual_status == "bidirectional_residual_supported_uncertain":
                out["uncertainty"].append("rigid_residual_supported_but_pose_not_accepted")
        else:
            out["uncertainty"].append("rigid_residual_check_missing_for_frame")
        out["generic_rigid_se3_attempt"] = attempt
    return out


def source_dimensions(ann: dict[str, Any]) -> tuple[float, float]:
    raw = ann.get("raw_video", {}) if isinstance(ann.get("raw_video"), dict) else {}
    return finite_float(raw.get("width"), 1920.0), finite_float(raw.get("height"), 1080.0)


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    source_path = args.source_root / case / "annotations_v18_full.json"
    ann = load_json(source_path)
    frames = ann.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    source_w, source_h = source_dimensions(ann)
    hawor_measurement_path = args.measurement_root / case / "measurements_v17" / "hawor_measurements.json"
    hawor_index, hawor_sources = load_hawor_measurement_index(hawor_measurement_path)
    hawor_hands = load_hawor_source_hands(hawor_sources)
    rigid_report_path = args.corrective_root / case / "rigid_se3_attempt" / "v18_rigid_se3_attempt_report.json"
    rigid_candidates = rigid_candidates_from_report(rigid_report_path)
    visible_rows, visible_archive_npz = visible_surface_row_index(args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json", set(rigid_candidates))
    occlusion_owner_rows, occlusion_owner_report = selected_occlusion_owner_index(args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json")
    occlusion_audit_rows, occlusion_audit_report = occlusion_acceptance_audit_index(args.corrective_root / case / "occlusion_owner_acceptance_audit" / "v18_occlusion_owner_acceptance_audit_report.json")
    contact_rows, contact_report = selected_contact_index(args.contact_graph_root / case / "v18_contact_ownership_graph_report.json")
    contact_audit_rows, contact_audit_report = contact_acceptance_audit_index(args.corrective_root / case / "contact_acceptance_audit" / "v18_contact_acceptance_audit_report.json")
    signed_rows = nonpenetration_row_index(args.signed_nonpenetration_root / case / "v18_signed_nonpenetration_evidence_report.json")
    triangle_rows = nonpenetration_row_index(args.triangle_nonpenetration_root / case / "v18_triangle_nonpenetration_evidence_report.json")
    residual_rows, residual_report = rigid_residual_row_index(args.corrective_root / case / "rigid_se3_residual_check" / "v18_rigid_se3_residual_check_report.json")
    repair_rows, repair_report = nonpenetration_repair_row_index(args.corrective_root / case / "nonpenetration_repair_proposal" / "v18_nonpenetration_repair_proposal_report.json")
    smoothed_hand_rows, smoothed_hand_report = temporal_smoothed_hand_row_index(args.corrective_root / case / "temporal_hand_pose_smoothing" / "v18_temporal_hand_pose_smoothing_report.json")
    hawor_bridge_quality_rows, hawor_bridge_quality_report = hawor_bridge_quality_index(args.corrective_root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_state.json")
    geometry_coverage = geometry_coverage_report(args.corrective_root / case / "geometry_coverage_audit" / "v18_geometry_coverage_audit_report.json")
    geometry_summaries = geometry_coverage.get("object_summaries", {}) if isinstance(geometry_coverage.get("object_summaries"), dict) else {}
    mano_foundation_path = args.corrective_root / "mano_foundation_audit" / case / "v18_mano_foundation_state_report.json"
    mano_foundation = load_json(mano_foundation_path) if mano_foundation_path.exists() else {}
    wilor_foundation = mano_foundation.get("recovered_wilor_virtual_camera_mano_candidates", {}) if isinstance(mano_foundation.get("recovered_wilor_virtual_camera_mano_candidates"), dict) else {}
    hawor_foundation = mano_foundation.get("hawor_world_mano_candidates", {}) if isinstance(mano_foundation.get("hawor_world_mano_candidates"), dict) else {}
    stable_pose = stable_rigid_pose_index(frames, set(rigid_candidates), args.translation_smoothing_radius)
    counts: Counter[str] = Counter()
    counts["mano_foundation_wilor_virtual_candidate_rows"] = int(wilor_foundation.get("complete_virtual_camera_candidate_rows", 0)) if isinstance(wilor_foundation.get("complete_virtual_camera_candidate_rows", 0), int) else int(float(wilor_foundation.get("complete_virtual_camera_candidate_rows", 0) or 0))
    counts["mano_foundation_wilor_virtual_unique_frame_side_rows"] = int(wilor_foundation.get("unique_virtual_camera_frame_side_rows", 0)) if isinstance(wilor_foundation.get("unique_virtual_camera_frame_side_rows", 0), int) else int(float(wilor_foundation.get("unique_virtual_camera_frame_side_rows", 0) or 0))
    counts["mano_foundation_hawor_world_rows"] = int(hawor_foundation.get("complete_world_surface_param_rows", 0)) if isinstance(hawor_foundation.get("complete_world_surface_param_rows", 0), int) else int(float(hawor_foundation.get("complete_world_surface_param_rows", 0) or 0))
    for summary in geometry_summaries.values():
        if isinstance(summary, dict):
            counts["geometry_coverage_audit_objects"] += 1
            status = summary.get("status")
            if isinstance(status, str):
                counts[f"geometry_coverage::{status}"] += 1
    out_frames: list[dict[str, Any]] = []
    for raw_frame in frames:
        frame = raw_frame if isinstance(raw_frame, dict) else {}
        frame_idx = int(frame.get("frame_idx", len(out_frames)))
        graph_hands = graph_hand_estimates(frame, source_w, source_h)
        graph_objects = graph_object_poses(frame)
        hand_states = []
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side"))
            state = hand_corrective_state(
                frame_idx,
                hand,
                graph_hands.get(side),
                hawor_index.get((frame_idx, side)),
                hawor_hands.get((frame_idx, side)),
                bool(hawor_index),
                occlusion_owner_rows.get((frame_idx, side)),
                occlusion_audit_rows.get((frame_idx, side), []),
                contact_rows.get((frame_idx, side)),
                contact_audit_rows.get((frame_idx, side), []),
                signed_rows.get((frame_idx, side, str(contact_rows.get((frame_idx, side), {}).get("chosen_owner_object_id")))) if contact_rows.get((frame_idx, side)) else None,
                triangle_rows.get((frame_idx, side, str(contact_rows.get((frame_idx, side), {}).get("chosen_owner_object_id")))) if contact_rows.get((frame_idx, side)) else None,
                repair_rows.get((frame_idx, side, str(contact_rows.get((frame_idx, side), {}).get("chosen_owner_object_id")))) if contact_rows.get((frame_idx, side)) else None,
                smoothed_hand_rows.get((frame_idx, side)),
                hawor_bridge_quality_rows.get((frame_idx, side)),
                source_w,
                source_h,
            )
            if "graph_hand_state" in state:
                counts["graph_hand_states"] += 1
                if state["graph_hand_state"].get("shifted_mano_joints2d_source_px") is not None:
                    counts["graph_shifted_mano_states"] += 1
            bridge_quality_state = state.get("hawor_bridge_quality_candidate", {}).get("status") if isinstance(state.get("hawor_bridge_quality_candidate"), dict) else None
            if isinstance(bridge_quality_state, str):
                counts["hawor_bridge_quality_candidate_rows"] += 1
                counts[f"hawor_bridge_quality::{bridge_quality_state}"] += 1
                if bridge_quality_state.startswith("projection_supported"):
                    counts["hawor_bridge_projection_supported_candidate_rows"] += 1
            prior_status = state.get("hawor_temporal_prior", {}).get("status") if isinstance(state.get("hawor_temporal_prior"), dict) else None
            if prior_status == "available_uncertain_prior":
                counts["hawor_prior_states"] += 1
            if prior_status == "provisioning_failed_no_case_measurements":
                counts["hawor_provisioning_failed_hand_states"] += 1
            if "pose_fill_best_effort" in state:
                counts["pose_fill_best_effort_states"] += 1
            if "occlusion_owner_best_effort" in state:
                counts["occlusion_owner_best_effort_states"] += 1
            audit_states = state.get("occlusion_owner_acceptance_audit") if isinstance(state.get("occlusion_owner_acceptance_audit"), list) else []
            for audit_state in audit_states:
                if isinstance(audit_state, dict):
                    counts["occlusion_owner_acceptance_audit_rows"] += 1
                    category = audit_state.get("category")
                    if isinstance(category, str):
                        counts[f"occlusion_owner_acceptance::{category}"] += 1
            contact_audit_states = state.get("contact_acceptance_audit") if isinstance(state.get("contact_acceptance_audit"), list) else []
            for audit_state in contact_audit_states:
                if isinstance(audit_state, dict):
                    counts["contact_acceptance_audit_rows"] += 1
                    category = audit_state.get("category")
                    if isinstance(category, str):
                        counts[f"contact_acceptance::{category}"] += 1
            if "contact_nonpenetration_state" in state:
                counts["contact_nonpenetration_states"] += 1
                contact_status = state["contact_nonpenetration_state"].get("status")
                if isinstance(contact_status, str):
                    counts[f"contact_nonpenetration::{contact_status}"] += 1
            repair_state = state.get("nonpenetration_repair_proposal", {}) if isinstance(state.get("nonpenetration_repair_proposal"), dict) else {}
            repair_status = repair_state.get("status")
            if isinstance(repair_status, str):
                counts["nonpenetration_repair_proposal_states"] += 1
                counts[f"nonpenetration_repair::{repair_status}"] += 1
            smooth_state = state.get("temporal_smoothed_mano2d_state", {}) if isinstance(state.get("temporal_smoothed_mano2d_state"), dict) else {}
            smooth_status = smooth_state.get("status")
            if isinstance(smooth_status, str):
                counts["temporal_smoothed_mano2d_states"] += 1
                counts[f"temporal_smoothed_mano2d::{smooth_status}"] += 1
            hand_states.append(state)
        object_states = []
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            oid = str(obj.get("object_id"))
            state = object_corrective_state(frame_idx, obj, graph_objects.get(oid), rigid_candidates.get(oid), stable_pose.get((frame_idx, oid)), visible_rows.get((frame_idx, oid)), residual_rows.get((frame_idx, oid)))
            if "graph_object_se3" in state:
                counts["graph_object_se3_states"] += 1
            if "frame_local_visible_surface_state" in state:
                counts["frame_local_visible_surface_states"] += 1
            rigid_attempt = state.get("generic_rigid_se3_attempt", {}) if isinstance(state.get("generic_rigid_se3_attempt"), dict) else {}
            if rigid_attempt.get("stable_pose6_world_from_object") is not None:
                counts["generic_rigid_stable_pose_states"] += 1
            residual_check = rigid_attempt.get("residual_check", {}) if isinstance(rigid_attempt.get("residual_check"), dict) else {}
            residual_status = residual_check.get("status")
            if isinstance(residual_status, str):
                counts["rigid_residual_checked_states"] += 1
                counts[f"rigid_residual::{residual_status}"] += 1
            object_states.append(state)
        out_frames.append({
            "frame_idx": frame_idx,
            "time_s": frame.get("time_s"),
            "raw_frame_path": frame.get("raw_frame_path"),
            "hands": hand_states,
            "objects": object_states,
            "state_scope": "corrective_delta_only_refs_source_full_annotation_for_other_fields",
        })
    output_path = args.output_root / case / "annotations_v18_corrective_state.json"
    payload = {
        "method": "build_v18_corrective_annotation_state",
        "status": "corrective_state_delta_not_full_v18_closure",
        "case": case,
        "source_annotation": str(source_path),
        "frame_count": len(out_frames),
        "fps": ann.get("fps"),
        "duration_s": ann.get("duration_s"),
        "source_dimensions": {"width": source_w, "height": source_h},
        "claim_scope": "corrective_annotation_delta_for_graph_hand_state_gated_temporal_mano2d_filter_rigid_object_visible_surface_geometry_coverage_residuals_hawor_occlusion_contact_and_v16_local_nonpenetration_candidates; does_not_claim_solved_occlusion_contact_complete_geometry_3d_mano_or_nonpenetration",
        "corrective_sources": {
            "graph_render_report": str(args.corrective_root / case / "v18_corrective_state_report.json"),
            "rigid_se3_report": str(rigid_report_path),
            "hawor_measurement_file": str(hawor_measurement_path),
            "hawor_source_annotations": [str(p) for p in hawor_sources],
            "hawor_execution_failure_logs": {
                "task5_export_attempt": str(args.corrective_root / "hawor_execution_attempt" / "task5_tomato_960" / "export_hawor_world_attempt.log"),
                "setup_preflight_attempt": str(args.corrective_root / "hawor_execution_attempt" / "setup_preflight" / "remote_setup_hawor_local_attempt.log"),
            },
            "visible_surface_archive_npz": visible_archive_npz,
            "visible_surface_state_report": str(args.corrective_root / case / "visible_surface_state" / "v18_visible_surface_state_report.json"),
            "geometry_coverage_audit_report": str(args.corrective_root / case / "geometry_coverage_audit" / "v18_geometry_coverage_audit_report.json"),
            "occlusion_owner_graph_report": str(args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json"),
            "occlusion_owner_best_effort_report": str(args.corrective_root / case / "occlusion_owner_best_effort" / "v18_occlusion_owner_best_effort_report.json"),
            "occlusion_owner_acceptance_audit_report": str(args.corrective_root / case / "occlusion_owner_acceptance_audit" / "v18_occlusion_owner_acceptance_audit_report.json"),
            "contact_ownership_graph_report": str(args.contact_graph_root / case / "v18_contact_ownership_graph_report.json"),
            "signed_nonpenetration_report": str(args.signed_nonpenetration_root / case / "v18_signed_nonpenetration_evidence_report.json"),
            "triangle_nonpenetration_report": str(args.triangle_nonpenetration_root / case / "v18_triangle_nonpenetration_evidence_report.json"),
            "contact_nonpenetration_state_report": str(args.corrective_root / case / "contact_nonpenetration_state" / "v18_contact_nonpenetration_state_report.json"),
            "contact_acceptance_audit_report": str(args.corrective_root / case / "contact_acceptance_audit" / "v18_contact_acceptance_audit_report.json"),
            "rigid_se3_residual_check_report": str(args.corrective_root / case / "rigid_se3_residual_check" / "v18_rigid_se3_residual_check_report.json"),
            "nonpenetration_repair_proposal_report": str(args.corrective_root / case / "nonpenetration_repair_proposal" / "v18_nonpenetration_repair_proposal_report.json"),
            "temporal_hand_pose_smoothing_report": str(args.corrective_root / case / "temporal_hand_pose_smoothing" / "v18_temporal_hand_pose_smoothing_report.json"),
            "mano_foundation_report": str(mano_foundation_path),
            "mano_foundation_wilor_virtual_camera_npz": wilor_foundation.get("npz_path") if isinstance(wilor_foundation, dict) else None,
            "hawor_bridge_quality_state_report": str(args.corrective_root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_state.json"),
            "hawor_bridge_quality_overlay_report": str(args.corrective_root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_overlay_report.json"),
        },
        "occlusion_owner_selected_rows": len(occlusion_owner_rows),
        "occlusion_owner_strict_accepted_rows": 0,
        "occlusion_owner_acceptance_blocker_counts": occlusion_owner_report.get("acceptance_blocker_counts") if isinstance(occlusion_owner_report, dict) else None,
        "occlusion_owner_acceptance_audit_category_counts": occlusion_audit_report.get("category_counts") if isinstance(occlusion_audit_report, dict) else None,
        "occlusion_owner_acceptance_audit_strict_promotable_rows": occlusion_audit_report.get("strict_promotable_owner_rows") if isinstance(occlusion_audit_report, dict) else None,
        "contact_graph_selected_rows": len(contact_rows),
        "source_graph_contact_candidate_rows_before_nonpenetration_veto": contact_report.get("contact_ownership_accepted_rows") if isinstance(contact_report, dict) else None,
        "contact_acceptance_audit_category_counts": contact_audit_report.get("category_counts") if isinstance(contact_audit_report, dict) else None,
        "contact_acceptance_audit_strict_promotable_rows": contact_audit_report.get("strict_promotable_contact_rows") if isinstance(contact_audit_report, dict) else None,
        "rigid_residual_candidate_objects": residual_report.get("candidate_objects") if isinstance(residual_report, dict) else None,
        "geometry_coverage_audit_status_counts": geometry_coverage.get("status_counts") if isinstance(geometry_coverage, dict) else None,
        "geometry_coverage_audit_stable_pose_source": geometry_coverage.get("stable_pose_source") if isinstance(geometry_coverage, dict) else None,
        "geometry_coverage_audit_object_summaries": geometry_summaries,
        "foundational_mano_state_valid": mano_foundation.get("foundational_mano_state_valid") if isinstance(mano_foundation, dict) else None,
        "v18_physical_pipeline_valid_without_further_hand_work": mano_foundation.get("v18_physical_pipeline_valid_without_further_hand_work") if isinstance(mano_foundation, dict) else None,
        "mano_foundation_blocking_reasons": mano_foundation.get("blocking_reasons") if isinstance(mano_foundation, dict) else None,
        "mano_foundation_wilor_virtual_candidate_rows": wilor_foundation.get("complete_virtual_camera_candidate_rows") if isinstance(wilor_foundation, dict) else None,
        "mano_foundation_wilor_virtual_unique_frame_side_rows": wilor_foundation.get("unique_virtual_camera_frame_side_rows") if isinstance(wilor_foundation, dict) else None,
        "mano_foundation_wilor_internal_projection_residual_px_median": wilor_foundation.get("wilor_internal_projection_residual_px_median") if isinstance(wilor_foundation, dict) else None,
        "mano_foundation_wilor_metric_world_alignment_valid": wilor_foundation.get("metric_world_alignment_valid") if isinstance(wilor_foundation, dict) else None,
        "mano_foundation_hawor_world_rows": hawor_foundation.get("complete_world_surface_param_rows") if isinstance(hawor_foundation, dict) else None,
        "hawor_bridge_quality_status": hawor_bridge_quality_report.get("status") if isinstance(hawor_bridge_quality_report, dict) else None,
        "hawor_bridge_quality_counts": hawor_bridge_quality_report.get("quality_counts") if isinstance(hawor_bridge_quality_report, dict) else None,
        "hawor_bridge_quality_accepted_v18_hawor_foundation": hawor_bridge_quality_report.get("accepted_v18_hawor_foundation") if isinstance(hawor_bridge_quality_report, dict) else None,
        "hawor_bridge_quality_v18_physical_hand_state_valid": hawor_bridge_quality_report.get("v18_physical_hand_state_valid_from_quality") if isinstance(hawor_bridge_quality_report, dict) else None,
        "nonpenetration_repair_proposal_status_counts": repair_report.get("proposal_status_counts") if isinstance(repair_report, dict) else None,
        "temporal_hand_pose_smoothing_draw_counts": smoothed_hand_report.get("draw_counts") if isinstance(smoothed_hand_report, dict) else None,
        "temporal_hand_pose_smoothing_jitter_probe": smoothed_hand_report.get("jitter_probe") if isinstance(smoothed_hand_report, dict) else None,
        "counts": dict(sorted(counts.items())),
        "rigid_candidate_ids": sorted(rigid_candidates),
        "hawor_measurement_rows": len(hawor_index),
        "frames": out_frames,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(output_path, payload)
    return {
        "case": case,
        "output_path": str(output_path),
        "frame_count": len(out_frames),
        "expected_frame_count": ann.get("frame_count"),
        "counts": dict(sorted(counts.items())),
        "rigid_candidate_ids": sorted(rigid_candidates),
        "hawor_measurement_rows": len(hawor_index),
        "elapsed_s": payload["elapsed_s"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_corrective_annotation_state",
        "status": "corrective_annotation_state_delta_not_full_v18_closure",
        "output_root": str(args.output_root),
        "source_root": str(args.source_root),
        "cases": reports,
        "all_frame_counts_match_source": all(r["frame_count"] == r["expected_frame_count"] for r in reports),
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_corrective_annotation_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--measurement-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_measurement_store"))
    parser.add_argument("--corrective-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--occlusion-owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--contact-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--signed-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--triangle-nonpenetration-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--translation-smoothing-radius", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
