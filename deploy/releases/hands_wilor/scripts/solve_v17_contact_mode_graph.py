#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

from run_v16_full_pipeline import load_mesh_archive

CONTACT_MODE_QC_STATUS = "contact_mode_qc_structurally_consistent"
CONTACT_MODE_REJECTED_STATUS = "contact_mode_qc_rejected"
CONTACT_MODE_ARTIFACT_KIND = "contact_mode_qc_graph"
DELIVERY_ROLE = "qc_only_not_v17_closure"

OBJECT_LIMIT_FLAGS = {
    "multi_object_timeline_ready": False,
    "object_schema_status": "single_manipulated_object_qc",
    "missing_multi_object_roster_required": True,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "object_geometry_status": "partial_visible_surface_or_local_patch_qc",
}

CONTACT_MODE_QC_BANNER_LINES = (
    "V17 CONTACT-MODE QC ONLY: annotation deliverable remains open",
    "annotation_ready=false | v3_solver_complete=false | object_geometry_complete=false",
)


@dataclass(frozen=True)
class ContactObs:
    frame_idx: int
    side: str
    active: bool
    unary_logit: float
    anchor_logit: float
    gap_min_m: float | None
    gap_p05_m: float | None
    gap_median_m: float | None
    gap_p95_m: float | None
    mask_distance_median_px: float | None
    mask_close_fraction: float | None
    hand_residual_median_px: float | None
    hand_residual_p95_px: float | None
    sparse_graph_hand_ready: bool
    source_contact_state: str | None
    selected_measurement_id: str | None
    reason: str | None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fitted_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    *,
    max_scale: float = 0.9,
    min_scale: float = 0.42,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> int:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max_scale
    while scale > min_scale:
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        if width <= max_width:
            break
        scale -= 0.04
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)
    return int(height + baseline + 8)


def draw_qc_banner(image: np.ndarray) -> int:
    banner_height = 96
    cv2.rectangle(image, (0, 0), (image.shape[1], banner_height), (12, 12, 12), thickness=-1)
    y = 34
    for i, line in enumerate(CONTACT_MODE_QC_BANNER_LINES):
        step = fitted_text(
            image,
            line,
            (24, y),
            image.shape[1] - 48,
            max_scale=0.94 if i == 0 else 0.7,
            min_scale=0.44,
            thickness=2,
        )
        y += step
    cv2.rectangle(image, (0, banner_height - 4), (image.shape[1], banner_height - 1), (255, 255, 255), thickness=-1)
    return banner_height


def mode_color(row: dict[str, Any]) -> tuple[int, int, int]:
    if row["contact_factor_ready"]:
        return (55, 150, 55)
    if row["mode"] == "contact":
        return (40, 125, 235)
    if row["mode"] == "no_contact":
        return (205, 120, 45)
    return (165, 165, 165)


def write_contact_mode_sheet(
    path: Path,
    *,
    case: str,
    frame_count: int,
    solved_rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    active_contact_rows: list[dict[str, Any]],
    unobserved_rows: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
    anchor_errors: list[dict[str, Any]],
    rejection_reasons: list[str],
) -> dict[str, Any]:
    width = 1800
    height = 860
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    banner_height = draw_qc_banner(image)

    left = 170
    right = width - 70
    timeline_width = right - left
    y = banner_height + 54
    fitted_text(image, f"case={case}", (32, y), width - 64, color=(25, 25, 25), max_scale=0.82)
    y += 44
    status = "structurally_consistent" if not rejection_reasons else "rejected"
    metric_line = (
        f"contact-mode graph status={status} | frames={frame_count} | rows={len(solved_rows)} | "
        f"active={len(active_rows)} | contact={len(active_contact_rows)} | unobserved={len(unobserved_rows)} | "
        f"ready_factors={len(ready_rows)} | anchor_errors={len(anchor_errors)}"
    )
    fitted_text(image, metric_line, (32, y), width - 64, color=(25, 25, 25), max_scale=0.62, thickness=1)
    y += 36
    fitted_text(
        image,
        "legend: gray=unobserved, blue=no-contact, orange=contact, green=contact-factor-ready",
        (32, y),
        width - 64,
        color=(25, 25, 25),
        max_scale=0.58,
        thickness=1,
    )
    y += 54

    sides = sorted({str(row["side"]) for row in solved_rows}, key=lambda side: (side not in ("left", "right"), side))
    frame_den = max(1, frame_count - 1)
    for side in sides:
        side_rows = sorted([row for row in solved_rows if str(row["side"]) == side], key=lambda row: int(row["frame_idx"]))
        cv2.putText(image, side, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.rectangle(image, (left, y - 4), (right, y + 26), (218, 218, 218), thickness=-1)
        for row in side_rows:
            idx = int(row["frame_idx"])
            x0 = left + int(round(idx * timeline_width / frame_den))
            x1 = left + int(round((idx + 1) * timeline_width / frame_den))
            if x1 <= x0:
                x1 = x0 + 1
            cv2.rectangle(image, (x0, y - 4), (min(right, x1), y + 26), mode_color(row), thickness=-1)
        cv2.rectangle(image, (left, y - 4), (right, y + 26), (50, 50, 50), thickness=1)
        y += 76

    tick_values = [0, frame_count // 4, frame_count // 2, (3 * frame_count) // 4, max(0, frame_count - 1)]
    axis_y = y - 32
    for tick in tick_values:
        x = left + int(round(tick * timeline_width / frame_den))
        cv2.line(image, (x, axis_y), (x, axis_y + 16), (40, 40, 40), 1)
        cv2.putText(image, str(tick), (x - 20, axis_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)

    y += 54
    interval_text = ", ".join(
        f"{item['side']}:{item['start_frame']}-{item['end_frame']}" for item in intervals(solved_rows, "contact")[:16]
    )
    if not interval_text:
        interval_text = "none"
    fitted_text(image, f"contact intervals preview: {interval_text}", (32, y), width - 64, color=(25, 25, 25), max_scale=0.55, thickness=1)
    y += 38
    fitted_text(
        image,
        "This sheet is a row-level contact-mode QC artifact; Stage 9 visual rendering and object mesh reconstruction remain open.",
        (32, y),
        width - 64,
        color=(25, 25, 25),
        max_scale=0.55,
        thickness=1,
    )
    y += 38
    if rejection_reasons:
        fitted_text(
            image,
            f"rejection_reasons={','.join(rejection_reasons)}",
            (32, y),
            width - 64,
            color=(20, 20, 160),
            max_scale=0.55,
            thickness=1,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write contact-mode review sheet {path}")
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": "contact_mode_interval_qc_sheet",
        "delivery_role": DELIVERY_ROLE,
        "source": "generated_from_contact_mode_rows",
        "visible_qc_banner": True,
        "qc_banner_lines": list(CONTACT_MODE_QC_BANNER_LINES),
        "visual_qc_scope": "contact_mode_interval_qc_only_not_visual_quality",
        "visual_quality_qc_pass": False,
        "stage9_visual_deliverable_ready": False,
        "width_px": int(width),
        "height_px": int(height),
    }


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def neg_log_sigmoid(x: float) -> float:
    if x >= 0.0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def finite_points(value: object) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    points = points[np.isfinite(points).all(axis=1)]
    return points if len(points) else None


def side_from_measurement_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    for part in re.split(r"[:/]", value):
        if part in ("left", "right"):
            return part
    return None


def side_key(hand: dict[str, Any], fallback: int) -> str:
    side = hand.get("side")
    if side in ("left", "right"):
        return str(side)
    entity_id = hand.get("entity_id")
    if entity_id == "hand:left":
        return "left"
    if entity_id == "hand:right":
        return "right"
    return f"hand_{fallback}"


def camera_world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    camera = frame.get("camera")
    matrix = camera.get("T_world_camera_metric") if isinstance(camera, dict) else None
    try:
        transform = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frame has no valid T_world_camera_metric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("frame has no valid T_world_camera_metric")
    return (points_world - transform[:3, 3][None, :]) @ transform[:3, :3]


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=float)
    valid = z > 1e-6
    uv[valid, 0] = intrinsics[0] * points_camera[valid, 0] / z[valid] + intrinsics[2]
    uv[valid, 1] = intrinsics[1] * points_camera[valid, 1] / z[valid] + intrinsics[3]
    return uv, valid


def source_size(intrinsics: np.ndarray) -> tuple[float, float]:
    return 2.0 * float(intrinsics[2]), 2.0 * float(intrinsics[3])


def load_mask_distance(path: str, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object mask {path}")
    mask_bool = mask > 0
    outside = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    src_w, src_h = source_size(intrinsics)
    scale = np.asarray([mask_bool.shape[1] / src_w, mask_bool.shape[0] / src_h], dtype=float)
    return outside, scale


def anchor_logit(frame: dict[str, Any], side: str, strength: float) -> tuple[float, str | None, str | None]:
    contact = frame.get("v17_contact_state")
    if not isinstance(contact, dict):
        return 0.0, None, None
    status = contact.get("status")
    measurement_id = contact.get("selected_measurement_id")
    selected_side = side_from_measurement_id(measurement_id) or side_from_measurement_id(contact.get("local_patch_state_id"))
    if selected_side != side:
        return 0.0, None, None
    if status == "accepted_contact":
        return strength, str(status), str(measurement_id) if isinstance(measurement_id, str) else None
    if status == "accepted_no_contact":
        return -strength, str(status), str(measurement_id) if isinstance(measurement_id, str) else None
    return 0.0, str(status) if isinstance(status, str) else None, str(measurement_id) if isinstance(measurement_id, str) else None


def hand_intrinsics(hand: dict[str, Any]) -> np.ndarray | None:
    try:
        intr = np.asarray(hand.get("source_intrinsics"), dtype=float)
    except (TypeError, ValueError):
        return None
    if intr.shape != (4,) or not np.all(np.isfinite(intr)) or intr[0] <= 0.0 or intr[1] <= 0.0:
        return None
    return intr


def finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def hand_residual_ok(hand: dict[str, Any], max_median_px: float, max_p95_px: float) -> bool:
    median, p95 = hand_residual_values(hand)
    if median is None or p95 is None:
        return False
    return median <= max_median_px and p95 <= max_p95_px


def hand_residual_values(hand: dict[str, Any]) -> tuple[float | None, float | None]:
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return None, None
    return finite_float(residual.get("median")), finite_float(residual.get("p95"))


def contact_observation(
    frame: dict[str, Any],
    hand: dict[str, Any],
    hand_i: int,
    mesh_vertices: np.ndarray | None,
    args: argparse.Namespace,
) -> ContactObs:
    idx = int(frame["frame_idx"])
    side = side_key(hand, hand_i)
    anchor, source_state, selected_id = anchor_logit(frame, side, float(args.anchor_logit))
    points = finite_points(hand.get("vertices_world_m"))
    if points is None:
        return ContactObs(idx, side, False, anchor, anchor, None, None, None, None, None, None, None, None, False, source_state, selected_id, "missing_world_hand_vertices")
    if mesh_vertices is None or len(mesh_vertices) == 0:
        return ContactObs(idx, side, False, anchor, anchor, None, None, None, None, None, None, None, None, False, source_state, selected_id, "missing_object_mesh")
    distances, _nearest = cKDTree(mesh_vertices).query(points, k=1)
    distances = np.asarray(distances, dtype=float)
    order = np.argsort(distances)[: min(len(distances), int(args.max_contact_points))]
    near_points = points[order]
    gap_min = float(np.min(distances))
    gap_p05 = float(np.percentile(distances, 5.0))
    gap_median = float(np.median(distances))
    gap_p95 = float(np.percentile(distances, 95.0))
    gap_logit = (float(args.contact_gap_p05_m) - gap_p05) / float(args.gap_scale_m)
    mask_logit = -float(args.missing_mask_logit)
    mask_median: float | None = None
    mask_close_fraction: float | None = None
    intr = hand_intrinsics(hand)
    hand_residual_median, hand_residual_p95 = hand_residual_values(hand)
    sparse_graph_hand_ready = hand_residual_ok(hand, float(args.max_hand_median_px), float(args.max_hand_p95_px))
    obj = frame.get("object")
    mask_path = obj.get("mask_path") if isinstance(obj, dict) else None
    if isinstance(mask_path, str) and mask_path and intr is not None:
        mask_distance, scale = load_mask_distance(mask_path, intr)
        near_cam = camera_world_to_camera(near_points, frame)
        uv, valid_z = project(near_cam, intr)
        xy = uv * scale[None, :]
        valid = (
            valid_z
            & np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] < mask_distance.shape[1])
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] < mask_distance.shape[0])
        )
        if np.any(valid):
            x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, mask_distance.shape[1] - 1)
            y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, mask_distance.shape[0] - 1)
            d = mask_distance[y, x].astype(float)
            d_source_px = d / float(np.mean(scale))
            mask_median = float(np.median(d_source_px))
            mask_close_fraction = float(np.mean(d_source_px <= float(args.mask_close_px)))
            mask_logit = (float(args.mask_close_px) - mask_median) / float(args.mask_scale_px)
    unary = float(args.base_contact_logit) + float(args.gap_weight) * gap_logit + float(args.mask_weight) * mask_logit + anchor
    return ContactObs(
        idx,
        side,
        True,
        unary,
        anchor,
        gap_min,
        gap_p05,
        gap_median,
        gap_p95,
        mask_median,
        mask_close_fraction,
        hand_residual_median,
        hand_residual_p95,
        sparse_graph_hand_ready,
        source_state,
        selected_id,
        None if sparse_graph_hand_ready else "hand_residual_rejected_for_sparse_graph_contact_factor",
    )


def missing_hand_observation(frame: dict[str, Any], side: str, args: argparse.Namespace) -> ContactObs:
    idx = int(frame["frame_idx"])
    anchor, source_state, selected_id = anchor_logit(frame, side, float(args.anchor_logit))
    return ContactObs(
        idx,
        side,
        False,
        anchor,
        anchor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        source_state,
        selected_id,
        "missing_hand_state",
    )


def factor_ready_checks(row: ContactObs, mode: str, confidence: float, args: argparse.Namespace) -> dict[str, bool]:
    return {
        "active_geometry": bool(row.active),
        "contact_mode": mode == "contact",
        "positive_contact_confidence": bool(confidence >= float(args.factor_ready_min_confidence)),
        "hand_residual_ok": bool(row.sparse_graph_hand_ready),
        "gap_p05_available": row.gap_p05_m is not None,
        "gap_p05_within_threshold": bool(
            row.gap_p05_m is not None and row.gap_p05_m <= float(args.factor_ready_max_gap_p05_m)
        ),
        "mask_distance_available": row.mask_distance_median_px is not None,
        "mask_distance_within_threshold": bool(
            row.mask_distance_median_px is not None
            and row.mask_distance_median_px <= float(args.factor_ready_max_mask_px)
        ),
        "no_rejection_reason": row.reason is None,
    }


def solved_row_payload(
    row: ContactObs,
    mode: str,
    confidence: float,
    contact_factor_ready: bool,
    readiness_checks: dict[str, bool],
) -> dict[str, Any]:
    return {
        "frame_idx": int(row.frame_idx),
        "side": row.side,
        "active": bool(row.active),
        "mode": mode,
        "contact_score": float(sigmoid(row.unary_logit)),
        "confidence_score": float(confidence),
        "contact_factor_ready": bool(contact_factor_ready),
        "contact_factor_readiness_checks": readiness_checks,
        "unary_logit": float(row.unary_logit),
        "anchor_logit": float(row.anchor_logit),
        "gap_min_m": row.gap_min_m,
        "gap_p05_m": row.gap_p05_m,
        "gap_median_m": row.gap_median_m,
        "gap_p95_m": row.gap_p95_m,
        "mask_distance_median_px": row.mask_distance_median_px,
        "mask_close_fraction": row.mask_close_fraction,
        "hand_residual_median_px": row.hand_residual_median_px,
        "hand_residual_p95_px": row.hand_residual_p95_px,
        "sparse_graph_hand_ready": bool(row.sparse_graph_hand_ready),
        "source_contact_state": row.source_contact_state,
        "selected_measurement_id": row.selected_measurement_id,
        "reason": row.reason,
    }


def solve_modes(rows: list[ContactObs], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: row.frame_idx)
    if not rows:
        return []
    observed = [(i, row) for i, row in enumerate(rows) if row.active or row.anchor_logit != 0.0]
    if not observed:
        return [
            solved_row_payload(row, "unobserved", 0.0, False, factor_ready_checks(row, "unobserved", 0.0, args))
            for row in rows
        ]
    observed_rows = [row for _i, row in observed]
    dp = np.full((len(observed_rows), 2), np.inf, dtype=float)
    prev = np.full((len(observed_rows), 2), -1, dtype=np.int8)
    first = observed_rows[0].unary_logit
    dp[0, 0] = neg_log_sigmoid(-first)
    dp[0, 1] = neg_log_sigmoid(first)
    for i in range(1, len(observed_rows)):
        logit = observed_rows[i].unary_logit
        unary = np.asarray([neg_log_sigmoid(-logit), neg_log_sigmoid(logit)], dtype=float)
        gap = observed_rows[i].frame_idx - observed_rows[i - 1].frame_idx
        link = 0.0 if gap > int(args.max_temporal_link_gap) else float(args.switch_penalty) / math.sqrt(float(max(1, gap)))
        for state in (0, 1):
            costs = np.asarray([dp[i - 1, old] + (0.0 if old == state else link) for old in (0, 1)], dtype=float)
            best = int(np.argmin(costs))
            dp[i, state] = unary[state] + costs[best]
            prev[i, state] = best
    state = int(np.argmin(dp[-1]))
    states = [state]
    for i in range(len(observed_rows) - 1, 0, -1):
        state = int(prev[i, state])
        states.append(state)
    states.reverse()
    state_by_row_index = {row_i: state for (row_i, _row), state in zip(observed, states)}
    out: list[dict[str, Any]] = []
    for row_i, row in enumerate(rows):
        state = state_by_row_index.get(row_i)
        contact_score = sigmoid(row.unary_logit)
        if state is None:
            mode = "unobserved"
            confidence = 0.0
        else:
            mode = "contact" if state == 1 else "no_contact"
            confidence = contact_score if state == 1 else sigmoid(-row.unary_logit)
        checks = factor_ready_checks(row, mode, confidence, args)
        factor_ready = all(checks.values())
        out.append(solved_row_payload(row, mode, confidence, factor_ready, checks))
    return out


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def intervals(rows: list[dict[str, Any]], mode: str) -> list[dict[str, int | str]]:
    out: list[dict[str, int | str]] = []
    by_side: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["mode"] == mode:
            by_side.setdefault(str(row["side"]), []).append(row)
    for side, side_rows in by_side.items():
        side_rows.sort(key=lambda row: row["frame_idx"])
        start = end = int(side_rows[0]["frame_idx"])
        for row in side_rows[1:]:
            idx = int(row["frame_idx"])
            if idx == end + 1:
                end = idx
            else:
                out.append({"side": side, "start_frame": start, "end_frame": end})
                start = end = idx
        out.append({"side": side, "start_frame": start, "end_frame": end})
    return out


def solve_case(args: argparse.Namespace, manifest: Path) -> dict[str, Any]:
    state = load_json(manifest)
    case = str(state["case"])
    annotations = Path(state["annotations"])
    mesh_archive = Path(state["object_mesh_archive"])
    payload = load_json(annotations)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise RuntimeError(f"{annotations} must contain frames")
    mesh_by_frame = load_mesh_archive(mesh_archive)
    observations: list[ContactObs] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise RuntimeError(f"{annotations} contains a non-object frame")
        idx = frame.get("frame_idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise RuntimeError(f"{annotations} contains invalid frame_idx {idx!r}")
        mesh = mesh_by_frame.get(idx)
        mesh_vertices = mesh[0] if mesh is not None else None
        hands = frame.get("hands")
        if not isinstance(hands, list):
            hands = []
        observed_standard_sides: set[str] = set()
        for hand_i, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            side = side_key(hand, hand_i)
            if side in ("left", "right"):
                observed_standard_sides.add(side)
            observations.append(contact_observation(frame, hand, hand_i, mesh_vertices, args))
        for side in ("left", "right"):
            if side not in observed_standard_sides:
                observations.append(missing_hand_observation(frame, side, args))
    solved_rows: list[dict[str, Any]] = []
    for side in sorted({row.side for row in observations}):
        side_rows = [row for row in observations if row.side == side]
        solved_rows.extend(solve_modes(side_rows, args))
    solved_rows.sort(key=lambda row: (int(row["frame_idx"]), str(row["side"])))
    active_rows = [row for row in solved_rows if row["active"]]
    active_contact_rows = [row for row in active_rows if row["mode"] == "contact"]
    unobserved_rows = [row for row in solved_rows if row["mode"] == "unobserved"]
    ready_rows = [row for row in solved_rows if row["contact_factor_ready"]]
    anchor_rows = [row for row in solved_rows if row["source_contact_state"] in ("accepted_contact", "accepted_no_contact")]
    anchor_errors = [
        row
        for row in anchor_rows
        if (row["source_contact_state"] == "accepted_contact" and row["mode"] != "contact")
        or (row["source_contact_state"] == "accepted_no_contact" and row["mode"] != "no_contact")
    ]
    rejection_reasons: list[str] = []
    if not active_rows:
        rejection_reasons.append("no_active_hand_object_geometry_observations")
    if not ready_rows:
        rejection_reasons.append("no_contact_factor_ready_rows")
    if anchor_errors:
        rejection_reasons.append("anchor_mode_contradictions")
    case_dir = Path(args.output_root) / case
    sheet_path = case_dir / "contact_mode_interval_review_sheet.jpg"
    sheet_report = write_contact_mode_sheet(
        sheet_path,
        case=case,
        frame_count=len(frames),
        solved_rows=solved_rows,
        active_rows=active_rows,
        active_contact_rows=active_contact_rows,
        unobserved_rows=unobserved_rows,
        ready_rows=ready_rows,
        anchor_errors=anchor_errors,
        rejection_reasons=rejection_reasons,
    )
    report_path = case_dir / "v17_contact_mode_graph_report.json"
    report = {
        "case": case,
        "status": CONTACT_MODE_QC_STATUS if not rejection_reasons else CONTACT_MODE_REJECTED_STATUS,
        "artifact_status": "partial",
        "artifact_kind": CONTACT_MODE_ARTIFACT_KIND,
        "delivery_role": DELIVERY_ROLE,
        "rejection_reasons": rejection_reasons,
        "annotation_ready": False,
        "deliverable_ready": False,
        "visual_quality_qc_pass": False,
        "stage9_visual_deliverable_ready": False,
        "method": "solve_v17_contact_mode_graph",
        "solver_completeness": "contact_mode_latent_only",
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "visual_artifacts": {
            "contact_mode_interval_review_sheet": sheet_report,
            "visible_qc_banner": True,
            "qc_banner_lines": list(CONTACT_MODE_QC_BANNER_LINES),
            "visual_qc_scope": "contact_mode_interval_qc_only_not_visual_quality",
            "visual_quality_qc_pass": False,
            "stage9_visual_deliverable_ready": False,
        },
        "semantics": {
            "optimized_variables": ["per-frame per-hand binary contact/no-contact mode"],
            "fixed_variables": ["input camera trajectory", "input MANO geometry", "input object mesh geometry", "input object pose"],
            "input_geometry_dependency": "Contact modes are estimated from the annotations and mesh archive named by source_manifest. The default manifests point at the anchor-only sparse graph outputs.",
            "claim_limit": "This graph estimates contact modes from fixed input V17 geometry and image evidence. It does not optimize camera, MANO articulation, object geometry, object pose, or depth.",
        },
        "source_manifest": str(manifest),
        "source_annotations": str(annotations),
        "source_mesh_archive": str(mesh_archive),
        "frame_count": int(len(frames)),
        "observation_count": int(len(solved_rows)),
        "active_observation_count": int(len(active_rows)),
        "contact_mode_count": int(sum(1 for row in solved_rows if row["mode"] == "contact")),
        "active_contact_mode_count": int(len(active_contact_rows)),
        "unobserved_row_count": int(len(unobserved_rows)),
        "contact_factor_ready_count": int(len(ready_rows)),
        "contact_factor_readiness_semantics": (
            "Rows become sparse-graph contact factors only when every row-level "
            "contact_factor_readiness_checks predicate is true, including explicit "
            "mask-distance evidence from the object mask."
        ),
        "anchor_count": int(len(anchor_rows)),
        "anchor_error_count": int(len(anchor_errors)),
        "anchor_errors": anchor_errors[:40],
        "contact_intervals": intervals(solved_rows, "contact")[:80],
        "factor_ready_rows_source": "filter rows where contact_factor_ready is true",
        "factor_ready_rows_preview_limit": 200,
        "factor_ready_rows_preview_truncated": len(ready_rows) > 200,
        "factor_ready_rows_preview": ready_rows[:200],
        "gap_p05_m": summarize([float(row["gap_p05_m"]) for row in active_rows if row["gap_p05_m"] is not None]),
        "mask_distance_median_px": summarize(
            [float(row["mask_distance_median_px"]) for row in active_rows if row["mask_distance_median_px"] is not None]
        ),
        "parameters": {
            "base_contact_logit": float(args.base_contact_logit),
            "anchor_logit": float(args.anchor_logit),
            "contact_gap_p05_m": float(args.contact_gap_p05_m),
            "gap_scale_m": float(args.gap_scale_m),
            "mask_close_px": float(args.mask_close_px),
            "mask_scale_px": float(args.mask_scale_px),
            "switch_penalty": float(args.switch_penalty),
            "max_temporal_link_gap": int(args.max_temporal_link_gap),
            "factor_ready_min_confidence": float(args.factor_ready_min_confidence),
            "factor_ready_max_gap_p05_m": float(args.factor_ready_max_gap_p05_m),
            "factor_ready_max_mask_px": float(args.factor_ready_max_mask_px),
            "max_hand_median_px": float(args.max_hand_median_px),
            "max_hand_p95_px": float(args.max_hand_p95_px),
        },
        "rows": solved_rows,
    }
    write_json(report_path, report)
    return report


def solve(args: argparse.Namespace) -> dict[str, Any]:
    reports = [solve_case(args, manifest) for manifest in args.case_manifests]
    latent_graph_status = "structurally_consistent" if all(report["status"] == CONTACT_MODE_QC_STATUS for report in reports) else "rejected"
    summary = {
        "status": "contact_mode_qc_structurally_consistent_collection" if latent_graph_status == "structurally_consistent" else "contact_mode_qc_rejected_collection",
        "latent_graph_status": latent_graph_status,
        "artifact_status": "partial",
        "artifact_kind": "contact_mode_qc_graph_collection",
        "delivery_role": DELIVERY_ROLE,
        "annotation_ready": False,
        "deliverable_ready": False,
        "visual_quality_qc_pass": False,
        "stage9_visual_deliverable_ready": False,
        "method": "solve_v17_contact_mode_graph",
        "solver_completeness": "contact_mode_latent_only",
        "v3_solver_complete": False,
        **OBJECT_LIMIT_FLAGS,
        "visual_qc_scope": "contact_mode_interval_qc_only_not_visual_quality",
        "visible_qc_banner": all(
            bool(report.get("visual_artifacts", {}).get("visible_qc_banner")) for report in reports
        ),
        "qc_banner_lines": list(CONTACT_MODE_QC_BANNER_LINES),
        "cases": [
            {
                "case": report["case"],
                "status": report["status"],
                "artifact_status": report["artifact_status"],
                "artifact_kind": report["artifact_kind"],
                "delivery_role": report["delivery_role"],
                "annotation_ready": report["annotation_ready"],
                "deliverable_ready": report["deliverable_ready"],
                "visual_quality_qc_pass": report["visual_quality_qc_pass"],
                "stage9_visual_deliverable_ready": report["stage9_visual_deliverable_ready"],
                "solver_completeness": report["solver_completeness"],
                "v3_solver_complete": report["v3_solver_complete"],
                "multi_object_timeline_ready": report["multi_object_timeline_ready"],
                "object_schema_status": report["object_schema_status"],
                "missing_multi_object_roster_required": report["missing_multi_object_roster_required"],
                "object_geometry_complete": report["object_geometry_complete"],
                "object_pose_requirement_met": report["object_pose_requirement_met"],
                "object_geometry_status": report["object_geometry_status"],
                "visual_artifacts": report["visual_artifacts"],
                "active_observation_count": report["active_observation_count"],
                "contact_mode_count": report["contact_mode_count"],
                "contact_factor_ready_count": report["contact_factor_ready_count"],
                "anchor_error_count": report["anchor_error_count"],
            }
            for report in reports
        ],
    }
    write_json(Path(args.output_root) / "v17_contact_mode_graph_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_contact_mode_graph"))
    parser.add_argument(
        "--case-manifests",
        type=Path,
        nargs="+",
        default=[
            Path("/data2/ego_annotation_outputs/v17_full_timeline_factor_graph/trash_1050/v17_full_timeline_factor_graph_manifest.json"),
            Path("/data2/ego_annotation_outputs/v17_full_timeline_factor_graph/task5_tomato_960/v17_full_timeline_factor_graph_manifest.json"),
        ],
    )
    parser.add_argument("--max-contact-points", type=int, default=80)
    parser.add_argument("--base-contact-logit", type=float, default=-2.2)
    parser.add_argument("--anchor-logit", type=float, default=7.0)
    parser.add_argument("--contact-gap-p05-m", type=float, default=0.030)
    parser.add_argument("--gap-scale-m", type=float, default=0.010)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--mask-close-px", type=float, default=28.0)
    parser.add_argument("--mask-scale-px", type=float, default=18.0)
    parser.add_argument("--mask-weight", type=float, default=0.75)
    parser.add_argument("--missing-mask-logit", type=float, default=1.0)
    parser.add_argument("--switch-penalty", type=float, default=2.2)
    parser.add_argument("--max-temporal-link-gap", type=int, default=10)
    parser.add_argument("--factor-ready-min-confidence", type=float, default=0.72)
    parser.add_argument("--factor-ready-max-gap-p05-m", type=float, default=0.035)
    parser.add_argument("--factor-ready-max-mask-px", type=float, default=45.0)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
