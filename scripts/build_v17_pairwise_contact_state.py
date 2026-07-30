#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np


STATUS = "v17_pairwise_contact_state_qc"
CLAIM = (
    "This artifact materializes V17 hand-object pair contact variables from image-plane hand/mask evidence. "
    "It creates one contact_pair[frame, hand, object] row for every active multi-object timeline row and hand side. "
    "Image evidence can support contact ownership, but physical contact factors remain blocked until the same object "
    "has compatible metric geometry, depth, and hand state."
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
SIDES = ("left", "right")


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


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def optional_finite_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_case_inputs(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
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
        "object_geometry_hypothesis_state": existing_path(
            args.object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
            f"{case} object-geometry hypothesis state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    return {"paths": paths, "payloads": payloads}


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "status": payload.get("status"),
        "method": payload.get("method"),
    }


def annotation_by_frame(report: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], int]:
    frames = require_list(report.get("frames"), "annotation frames")
    out: dict[int, dict[str, Any]] = {}
    for i, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"annotation frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"annotation frames[{i}].frame_idx")
        if frame_idx in out:
            raise RuntimeError(f"duplicate annotation frame {frame_idx}")
        out[frame_idx] = frame
    return out, len(frames)


def timeline_by_frame(report: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], int, int]:
    frames = require_list(report.get("frames"), "multi-object timeline frames")
    frame_count = require_int(report.get("frame_count"), "multi-object timeline frame_count")
    object_frame_rows = require_int(report.get("object_frame_rows"), "multi-object timeline object_frame_rows")
    if len(frames) != frame_count:
        raise RuntimeError("multi-object timeline frame_count disagrees with frames length")
    out: dict[int, list[dict[str, Any]]] = {}
    counted = 0
    for i, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"multi-object timeline frames[{i}]")
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


def contact_mode_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "contact-mode rows")):
        row = require_dict(raw, f"contact-mode rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"contact-mode rows[{i}].frame_idx")
        side = require_str(row.get("side"), f"contact-mode rows[{i}].side")
        key = (frame_idx, side)
        if key in out:
            raise RuntimeError(f"duplicate contact-mode row: {key}")
        out[key] = row
    return out


def multi_contact_index(report: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("rows"), "multi-object contact rows")):
        row = require_dict(raw, f"multi-object contact rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"multi-object contact rows[{i}].frame_idx")
        object_id = require_str(row.get("object_id"), f"multi-object contact rows[{i}].object_id")
        side = require_str(row.get("hand_side"), f"multi-object contact rows[{i}].hand_side")
        key = (frame_idx, object_id, side)
        if key in out:
            raise RuntimeError(f"duplicate multi-object contact evidence row: {key}")
        out[key] = row
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


def hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    hands = frame.get("hands")
    if not isinstance(hands, list):
        return None
    for i, raw_hand in enumerate(hands):
        hand = require_dict(raw_hand, f"frame {frame.get('frame_idx')}.hands[{i}]")
        hand_side = hand.get("side")
        entity_id = hand.get("entity_id")
        if hand_side == side or entity_id == f"hand:{side}":
            return hand
    return None


def hand_points(hand: dict[str, Any]) -> np.ndarray | None:
    for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
        if key not in hand:
            continue
        try:
            points = np.asarray(hand.get(key), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            return points
    return None


def hand_intrinsics(hand: dict[str, Any]) -> np.ndarray | None:
    try:
        intr = np.asarray(hand.get("source_intrinsics"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if intr.shape != (4,) or not np.all(np.isfinite(intr)) or intr[0] <= 0.0 or intr[1] <= 0.0:
        return None
    return intr


def hand_residuals(hand: dict[str, Any]) -> tuple[float | None, float | None]:
    residual = hand.get("projection_residual_to_measurement_px")
    if not isinstance(residual, dict):
        return None, None
    return (
        optional_finite_float(residual.get("median"), "hand residual median"),
        optional_finite_float(residual.get("p95"), "hand residual p95"),
    )


def hand_residual_ok(hand: dict[str, Any], args: argparse.Namespace) -> bool:
    median, p95 = hand_residuals(hand)
    return bool(
        median is not None
        and p95 is not None
        and median <= float(args.max_hand_median_px)
        and p95 <= float(args.max_hand_p95_px)
    )


def camera_world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    camera = require_dict(frame.get("camera"), "frame camera")
    try:
        transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frame has invalid T_world_camera_metric") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("frame has invalid T_world_camera_metric")
    return (points_world - transform[:3, 3][None, :]) @ transform[:3, :3]


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    uv[valid, 0] = intrinsics[0] * points_camera[valid, 0] / z[valid] + intrinsics[2]
    uv[valid, 1] = intrinsics[1] * points_camera[valid, 1] / z[valid] + intrinsics[3]
    return uv, valid


def source_size(intrinsics: np.ndarray) -> tuple[float, float]:
    return 2.0 * float(intrinsics[2]), 2.0 * float(intrinsics[3])


def load_mask_distance(path: str) -> tuple[np.ndarray, np.ndarray]:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read object mask {path}")
    mask_bool = mask > 0
    outside = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    return outside, mask_bool


def image_contact_evidence(
    *,
    frame: dict[str, Any],
    side: str,
    obj: dict[str, Any],
    mask_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hand = hand_by_side(frame, side)
    mask_path = obj.get("mask_path")
    missing: list[str] = []
    if hand is None:
        missing.append("hand_state")
    if not isinstance(mask_path, str) or not mask_path:
        missing.append("visible_object_mask")
    points = hand_points(hand) if hand is not None else None
    intr = hand_intrinsics(hand) if hand is not None else None
    if hand is not None and points is None:
        missing.append("hand_world_points")
    if hand is not None and intr is None:
        missing.append("hand_source_intrinsics")
    median_residual, p95_residual = hand_residuals(hand) if hand is not None else (None, None)
    residual_ok = bool(hand is not None and hand_residual_ok(hand, args))
    base = {
        "image_evidence_state": "unobserved_image_pair" if missing else "measured_image_pair",
        "missing_image_evidence": missing,
        "hand_residual_median_px": median_residual,
        "hand_residual_p95_px": p95_residual,
        "hand_residual_ok": residual_ok,
        "mask_path": mask_path if isinstance(mask_path, str) else None,
        "valid_projected_hand_vertices": 0,
        "valid_projected_hand_fraction": 0.0,
        "inside_mask_fraction": None,
        "mask_distance_min_px": None,
        "mask_distance_p05_px": None,
        "mask_distance_median_px": None,
        "mask_distance_p95_px": None,
        "mask_close_fraction_5px": None,
        "mask_close_fraction_20px": None,
        "image_overlap_candidate": False,
    }
    if missing:
        return base
    if points is None or intr is None or not isinstance(mask_path, str):
        raise RuntimeError("missing image evidence branch failed to return")
    if mask_path not in mask_cache:
        mask_cache[mask_path] = load_mask_distance(mask_path)
    distance_image, mask_bool = mask_cache[mask_path]
    points_camera = camera_world_to_camera(points, frame)
    uv, valid_z = project(points_camera, intr)
    src_w, src_h = source_size(intr)
    scale = np.asarray([mask_bool.shape[1] / src_w, mask_bool.shape[0] / src_h], dtype=np.float64)
    xy = uv * scale[None, :]
    valid = (
        valid_z
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < mask_bool.shape[1])
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < mask_bool.shape[0])
    )
    if not np.any(valid):
        return {**base, "image_evidence_state": "unobserved_image_pair", "missing_image_evidence": ["projected_hand_vertices_inside_image"]}
    x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, mask_bool.shape[1] - 1)
    y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, mask_bool.shape[0] - 1)
    distance_source_px = distance_image[y, x].astype(np.float64) / float(np.mean(scale))
    inside = mask_bool[y, x]
    valid_count = int(len(distance_source_px))
    min_px = float(np.min(distance_source_px))
    p05_px = float(np.percentile(distance_source_px, 5.0))
    median_px = float(np.median(distance_source_px))
    p95_px = float(np.percentile(distance_source_px, 95.0))
    close_5 = float(np.mean(distance_source_px <= float(args.close_mask_px)))
    close_20 = float(np.mean(distance_source_px <= float(args.near_mask_px)))
    overlap_checks = {
        "hand_residual_ok": residual_ok,
        "valid_projected_vertices_met": valid_count >= int(args.min_projected_vertices),
        "mask_min_distance_met": min_px <= float(args.near_mask_px),
        "mask_p05_distance_met": p05_px <= float(args.image_candidate_p05_px),
        "mask_close_fraction_met": close_20 >= float(args.min_close_fraction_20px),
    }
    return {
        **base,
        "valid_projected_hand_vertices": valid_count,
        "valid_projected_hand_fraction": float(valid_count / max(1, len(points))),
        "inside_mask_fraction": float(np.mean(inside)),
        "mask_distance_min_px": min_px,
        "mask_distance_p05_px": p05_px,
        "mask_distance_median_px": median_px,
        "mask_distance_p95_px": p95_px,
        "mask_close_fraction_5px": close_5,
        "mask_close_fraction_20px": close_20,
        "image_overlap_checks": overlap_checks,
        "image_overlap_candidate": bool(all(overlap_checks.values())),
    }


def pairwise_row(
    *,
    case: str,
    frame: dict[str, Any],
    obj: dict[str, Any],
    side: str,
    contact_mode: dict[str, Any] | None,
    multi_contact: dict[str, Any] | None,
    object_state: dict[str, Any] | None,
    mask_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
    object_id = require_str(obj.get("object_id"), "timeline object_id")
    image_evidence = image_contact_evidence(
        frame=frame,
        side=side,
        obj=obj,
        mask_cache=mask_cache,
        args=args,
    )
    hand_mode = optional_str(contact_mode.get("mode"), "contact-mode mode") if contact_mode else None
    hand_contact_factor_ready = bool(contact_mode is not None and contact_mode.get("contact_factor_ready") is True)
    image_overlap = bool(image_evidence.get("image_overlap_candidate") is True)
    pair_contact_image_candidate = bool(hand_mode == "contact" and image_overlap)
    contact_owner_image_supported = bool(hand_contact_factor_ready and image_overlap)
    visible_distance = (
        optional_finite_float(multi_contact.get("min_symmetric_distance_m"), "multi-object min distance")
        if multi_contact
        else None
    )
    visible_surface_candidate = bool(multi_contact is not None and multi_contact.get("visible_surface_distance_candidate") is True)
    contact_compatible_geometry = bool(
        object_state is not None and object_state.get("can_own_contact_factors") is True
    )
    if image_evidence.get("image_evidence_state") != "measured_image_pair":
        pair_state = "unobserved_pair"
    elif hand_mode != "contact":
        pair_state = "image_overlap_without_hand_side_contact" if image_overlap else "no_hand_side_contact"
    elif image_overlap:
        pair_state = "image_supported_pair_contact_without_metric_geometry"
    else:
        pair_state = "hand_side_contact_without_object_image_support"
    return {
        "case": case,
        "pair_contact_variable_id": f"contact_pair:v17:{frame_idx:06d}:{side}:{object_id}",
        "frame_idx": frame_idx,
        "hand_side": side,
        "object_id": object_id,
        "track_id": require_str(obj.get("track_id"), "timeline track_id"),
        "name": optional_str(obj.get("name"), "timeline name"),
        "active": bool(obj.get("active") is True),
        "visible": bool(obj.get("visible") is True),
        "mask_evidence_status": optional_str(obj.get("mask_evidence_status"), "timeline mask_evidence_status"),
        "hand_side_contact_mode": {
            "available": contact_mode is not None,
            "mode": hand_mode,
            "contact_factor_ready": hand_contact_factor_ready,
            "contact_score": optional_finite_float(contact_mode.get("contact_score"), "contact score") if contact_mode else None,
            "confidence_score": optional_finite_float(contact_mode.get("confidence_score"), "confidence score") if contact_mode else None,
            "selected_measurement_id": optional_str(contact_mode.get("selected_measurement_id"), "selected measurement id")
            if contact_mode
            else None,
        },
        "image_plane_hand_mask_evidence": image_evidence,
        "multi_object_visible_surface_contact": {
            "available": multi_contact is not None,
            "visible_surface_distance_candidate": visible_surface_candidate,
            "min_symmetric_distance_m": visible_distance,
            "contact_factor_ready": bool(multi_contact is not None and multi_contact.get("contact_factor_ready") is True),
        },
        "object_readiness_checks": {
            "can_own_contact_factors": contact_compatible_geometry,
            "object_geometry_complete": bool(object_state is not None and object_state.get("object_geometry_complete") is True),
            "complete_mesh_timeline_ready": bool(object_state is not None and object_state.get("complete_mesh_timeline_ready") is True),
        },
        "image_overlap_candidate": image_overlap,
        "pair_contact_image_candidate": pair_contact_image_candidate,
        "contact_owner_image_supported": contact_owner_image_supported,
        "pair_contact_state": pair_state,
        "physical_contact_factor_ready": False,
        "physical_contact_factor_blockers": [
            "image-plane overlap does not define metric contact depth",
            "object geometry/contact ownership is not compatible with the current metric visible-surface state",
        ],
        **FALSE_READY,
    }


def owner_image_state(rows: list[dict[str, Any]], ready_keys: set[tuple[int, str]]) -> dict[str, Any]:
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = {key: [] for key in ready_keys}
    for row in rows:
        key = (require_int(row.get("frame_idx"), "frame_idx"), require_str(row.get("hand_side"), "hand_side"))
        if key in by_key:
            by_key[key].append(row)
    states: list[str] = []
    supported_rows = 0
    for key in sorted(by_key):
        supported = [row for row in by_key[key] if row.get("contact_owner_image_supported") is True]
        supported_rows += len(supported)
        if len(supported) == 0:
            states.append("no_image_supported_candidate")
        elif len(supported) == 1:
            states.append("single_image_supported_candidate")
        else:
            states.append("ambiguous_image_supported_candidates")
    counts = Counter(states)
    return {
        "owner_image_variable_count": len(ready_keys),
        "owner_image_supported_candidate_rows": supported_rows,
        "owner_image_variables_with_supported_candidate": sum(1 for state in states if state != "no_image_supported_candidate"),
        "owner_image_variables_with_single_supported_candidate": int(counts.get("single_image_supported_candidate", 0)),
        "owner_image_variables_with_ambiguous_supported_candidates": int(counts.get("ambiguous_image_supported_candidates", 0)),
        "owner_image_variables_without_supported_candidate": int(counts.get("no_image_supported_candidate", 0)),
        "owner_image_state_counts": dict(sorted(counts.items())),
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_case_inputs(case, args)
    paths: dict[str, Path] = loaded["paths"]
    payloads: dict[str, dict[str, Any]] = loaded["payloads"]
    annotations, annotation_frame_count = annotation_by_frame(payloads["annotations"])
    timeline, frame_count, object_frame_rows = timeline_by_frame(payloads["multi_object_timeline"])
    if annotation_frame_count != frame_count:
        raise RuntimeError(f"{case} annotation frame count disagrees with multi-object timeline")
    contact_modes = contact_mode_index(payloads["contact_mode"])
    multi_contacts = multi_contact_index(payloads["multi_object_contact"])
    object_states = object_state_index(payloads["object_geometry_hypothesis_state"])
    mask_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for frame_idx in sorted(timeline):
        frame = annotations.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} frame {frame_idx} missing from annotations")
        for obj in timeline[frame_idx]:
            object_id = require_str(obj.get("object_id"), "timeline object_id")
            for side in SIDES:
                rows.append(
                    pairwise_row(
                        case=case,
                        frame=frame,
                        obj=obj,
                        side=side,
                        contact_mode=contact_modes.get((frame_idx, side)),
                        multi_contact=multi_contacts.get((frame_idx, object_id, side)),
                        object_state=object_states.get(object_id),
                        mask_cache=mask_cache,
                        args=args,
                    )
                )
    expected_rows = object_frame_rows * len(SIDES)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{case} pairwise row count disagrees with object_frame_rows")
    ready_keys = {
        (require_int(row.get("frame_idx"), "contact-mode frame_idx"), require_str(row.get("side"), "contact-mode side"))
        for row in contact_modes.values()
        if row.get("contact_factor_ready") is True
    }
    owner_state = owner_image_state(rows, ready_keys)
    measured = [row for row in rows if row["image_plane_hand_mask_evidence"]["image_evidence_state"] == "measured_image_pair"]
    overlap = [row for row in rows if row["image_overlap_candidate"] is True]
    pair_candidates = [row for row in rows if row["pair_contact_image_candidate"] is True]
    owner_candidates = [row for row in rows if row["contact_owner_image_supported"] is True]
    physical_ready = [row for row in rows if row["physical_contact_factor_ready"] is True]
    state_counts = Counter(require_str(row.get("pair_contact_state"), "pair_contact_state") for row in rows)
    missing_counts: Counter[str] = Counter()
    for row in rows:
        for reason in require_list(
            row["image_plane_hand_mask_evidence"].get("missing_image_evidence"),
            "missing image evidence",
        ):
            missing_counts[require_str(reason, "missing image evidence reason")] += 1
    distances = [
        finite_float(
            row["image_plane_hand_mask_evidence"].get("mask_distance_p05_px"),
            "mask_distance_p05_px",
        )
        for row in measured
        if row["image_plane_hand_mask_evidence"].get("mask_distance_p05_px") is not None
    ]
    report = {
        "method": "build_v17_pairwise_contact_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "object_frame_rows": object_frame_rows,
        "pairwise_contact_variable_count": len(rows),
        "measured_image_pair_rows": len(measured),
        "unobserved_image_pair_rows": len(rows) - len(measured),
        "image_overlap_candidate_rows": len(overlap),
        "pair_contact_image_candidate_rows": len(pair_candidates),
        "contact_owner_image_supported_candidate_rows": len(owner_candidates),
        "physical_contact_factor_ready_rows": len(physical_ready),
        "pair_contact_state_counts": dict(sorted(state_counts.items())),
        "missing_image_evidence_reason_counts": dict(sorted(missing_counts.items())),
        "mask_distance_p05_px": summarize(distances),
        **owner_state,
        "rows": rows,
        "problem_semantics": {
            "variable": "contact_pair[frame_idx, hand_side, object_id]",
            "domain": "every active multi-object timeline row crossed with left/right hand side",
            "image_unary_evidence": [
                "projected MANO vertices",
                "active object SAM2 mask distance transform",
                "hand-side contact-mode state",
                "hand projection residual",
            ],
            "physical_factor_rule": "Image-supported pairs cannot become physical contact factors until metric object geometry, depth, and hand state agree for the same object id.",
        },
        "parameters": {
            "close_mask_px": float(args.close_mask_px),
            "near_mask_px": float(args.near_mask_px),
            "image_candidate_p05_px": float(args.image_candidate_p05_px),
            "min_close_fraction_20px": float(args.min_close_fraction_20px),
            "min_projected_vertices": int(args.min_projected_vertices),
            "max_hand_median_px": float(args.max_hand_median_px),
            "max_hand_p95_px": float(args.max_hand_p95_px),
        },
        **FALSE_READY,
    }
    if report["physical_contact_factor_ready_rows"] != 0:
        raise RuntimeError(f"{case} physical contact factor readiness must stay false in this image-evidence layer")
    write_json(args.output_root / case / "v17_pairwise_contact_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.multi_object_timeline_root / "v17_multi_object_timeline_summary.json",
        "multi-object timeline summary",
    )
    summary = require_dict(load_json(summary_path), "multi-object timeline summary")
    cases = [
        require_str(require_dict(raw, f"timeline summary cases[{i}]").get("case"), "timeline summary case")
        for i, raw in enumerate(require_list(summary.get("cases"), "timeline summary cases"))
    ]
    reports = [case_problem(case, args) for case in cases]
    payload = {
        "method": "build_v17_pairwise_contact_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_multi_object_timeline_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / report["case"] / "v17_pairwise_contact_state.json"),
                "frame_count": report["frame_count"],
                "pairwise_contact_variable_count": report["pairwise_contact_variable_count"],
                "measured_image_pair_rows": report["measured_image_pair_rows"],
                "image_overlap_candidate_rows": report["image_overlap_candidate_rows"],
                "pair_contact_image_candidate_rows": report["pair_contact_image_candidate_rows"],
                "contact_owner_image_supported_candidate_rows": report[
                    "contact_owner_image_supported_candidate_rows"
                ],
                "owner_image_variables_with_single_supported_candidate": report[
                    "owner_image_variables_with_single_supported_candidate"
                ],
                "owner_image_variables_with_ambiguous_supported_candidates": report[
                    "owner_image_variables_with_ambiguous_supported_candidates"
                ],
                "physical_contact_factor_ready_rows": report["physical_contact_factor_ready_rows"],
                "pair_contact_state_counts": report["pair_contact_state_counts"],
                "owner_image_state_counts": report["owner_image_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "pairwise_contact_variable_count": sum(report["pairwise_contact_variable_count"] for report in reports),
        "measured_image_pair_rows": sum(report["measured_image_pair_rows"] for report in reports),
        "unobserved_image_pair_rows": sum(report["unobserved_image_pair_rows"] for report in reports),
        "image_overlap_candidate_rows": sum(report["image_overlap_candidate_rows"] for report in reports),
        "pair_contact_image_candidate_rows": sum(report["pair_contact_image_candidate_rows"] for report in reports),
        "contact_owner_image_supported_candidate_rows": sum(
            report["contact_owner_image_supported_candidate_rows"] for report in reports
        ),
        "owner_image_variable_count": sum(report["owner_image_variable_count"] for report in reports),
        "owner_image_supported_candidate_rows": sum(report["owner_image_supported_candidate_rows"] for report in reports),
        "owner_image_variables_with_supported_candidate": sum(
            report["owner_image_variables_with_supported_candidate"] for report in reports
        ),
        "owner_image_variables_with_single_supported_candidate": sum(
            report["owner_image_variables_with_single_supported_candidate"] for report in reports
        ),
        "owner_image_variables_with_ambiguous_supported_candidates": sum(
            report["owner_image_variables_with_ambiguous_supported_candidates"] for report in reports
        ),
        "owner_image_variables_without_supported_candidate": sum(
            report["owner_image_variables_without_supported_candidate"] for report in reports
        ),
        "physical_contact_factor_ready_rows": sum(report["physical_contact_factor_ready_rows"] for report in reports),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_pairwise_contact_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
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
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"),
    )
    parser.add_argument("--close-mask-px", type=float, default=5.0)
    parser.add_argument("--near-mask-px", type=float, default=20.0)
    parser.add_argument("--image-candidate-p05-px", type=float, default=12.0)
    parser.add_argument("--min-close-fraction-20px", type=float, default=0.10)
    parser.add_argument("--min-projected-vertices", type=int, default=21)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
