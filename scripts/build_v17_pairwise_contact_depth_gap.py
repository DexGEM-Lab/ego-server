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


STATUS = "v17_pairwise_contact_depth_gap_qc"
CLAIM = (
    "This artifact measures metric depth compatibility for V17 image-supported hand-object contact pairs. "
    "It samples projected MANO vertices near each active object mask and compares their camera depth with "
    "the UniDepth object depth at the same pixels. It is diagnostic evidence for contact factors, not a solver."
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


def optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: list[float]) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5.0)),
        "p95": float(np.percentile(vals, 95.0)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def source_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if payload is not None:
        out["status"] = payload.get("status")
        out["method"] = payload.get("method")
    return out


def annotation_by_frame(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("frames"), "annotation frames")):
        frame = require_dict(raw, f"annotation frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"annotation frames[{i}].frame_idx")
        if frame_idx in out:
            raise RuntimeError(f"duplicate annotation frame {frame_idx}")
        out[frame_idx] = frame
    return out


def depth_archive(path: Path) -> dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    required = {"frame_idx", "depth", "source_size", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(npz.files))
    if missing:
        raise RuntimeError(f"{path} missing depth archive arrays: {missing}")
    frame_idx = np.asarray(npz["frame_idx"], dtype=np.int32)
    depth = np.asarray(npz["depth"], dtype=np.float32)
    intrinsics = np.asarray(npz["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    source_size = np.asarray(npz["source_size"], dtype=np.int32)
    if depth.ndim != 3:
        raise RuntimeError(f"{path} depth must be [frame,height,width]")
    if len(frame_idx) != depth.shape[0] or intrinsics.shape != (depth.shape[0], 4):
        raise RuntimeError(f"{path} depth arrays disagree in frame dimension")
    if source_size.shape != (2,):
        raise RuntimeError(f"{path} source_size must be [width,height]")
    return {
        "path": path,
        "frame_to_i": {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())},
        "depth": depth,
        "intrinsics": intrinsics,
        "source_size": (int(source_size[0]), int(source_size[1])),
    }


def hand_by_side(frame: dict[str, Any], side: str) -> dict[str, Any] | None:
    for i, raw in enumerate(require_list(frame.get("hands", []), "frame hands")):
        hand = require_dict(raw, f"frame hands[{i}]")
        if hand.get("side") == side:
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


def hand_residual_ok(hand: dict[str, Any], args: argparse.Namespace) -> bool:
    residual = require_dict(hand.get("projection_residual_to_measurement_px"), "hand projection residual")
    median = optional_float(residual.get("median"), "hand residual median")
    p95 = optional_float(residual.get("p95"), "hand residual p95")
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


def load_mask_distance(mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    mask_bool = mask > 0
    inv = (~mask_bool).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return dist.astype(np.float32), mask_bool


def measured_depth_gap(
    *,
    row: dict[str, Any],
    frame: dict[str, Any],
    depth: dict[str, Any],
    mask_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(row.get("frame_idx"), "row frame_idx")
    side = require_str(row.get("hand_side"), "row hand_side")
    object_id = require_str(row.get("object_id"), "row object_id")
    image_evidence = require_dict(row.get("image_plane_hand_mask_evidence"), "image evidence")
    mask_path = require_str(image_evidence.get("mask_path"), "image evidence mask_path")
    hand = hand_by_side(frame, side)
    missing: list[str] = []
    if hand is None:
        missing.append("hand_state")
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        missing.append("metric_depth_frame")
    points = hand_points(hand) if hand is not None else None
    intr = hand_intrinsics(hand) if hand is not None else None
    if hand is not None and points is None:
        missing.append("hand_world_points")
    if hand is not None and intr is None:
        missing.append("hand_source_intrinsics")
    if missing:
        return {
            "case": row.get("case"),
            "pair_contact_variable_id": require_str(row.get("pair_contact_variable_id"), "pair_contact_variable_id"),
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": object_id,
            "depth_gap_state": "unobserved_pair_depth",
            "missing_depth_evidence": missing,
            "metric_depth_compatible_candidate": False,
            "physical_contact_factor_ready": False,
            **FALSE_READY,
        }
    if points is None or intr is None or depth_i is None:
        raise RuntimeError("missing depth branch failed to return")
    if hand is None:
        raise RuntimeError("hand state vanished after missing-evidence check")
    if mask_path not in mask_cache:
        mask_cache[mask_path] = load_mask_distance(mask_path)
    distance_image, mask_bool = mask_cache[mask_path]
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    if mask_bool.shape != depth_m.shape:
        distance_image = cv2.resize(distance_image, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    points_camera = camera_world_to_camera(points, frame)
    uv, valid_z = project(points_camera, intr)
    hand_w, hand_h = source_size(intr)
    depth_w, depth_h = depth["source_size"]
    scale = np.asarray([depth_w / hand_w, depth_h / hand_h], dtype=np.float64)
    xy = uv * scale[None, :]
    valid = (
        valid_z
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < depth_m.shape[1])
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < depth_m.shape[0])
    )
    if not np.any(valid):
        return {
            "case": row.get("case"),
            "pair_contact_variable_id": require_str(row.get("pair_contact_variable_id"), "pair_contact_variable_id"),
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": object_id,
            "depth_gap_state": "unobserved_pair_depth",
            "missing_depth_evidence": ["projected_hand_vertices_inside_depth_image"],
            "metric_depth_compatible_candidate": False,
            "physical_contact_factor_ready": False,
            **FALSE_READY,
        }
    valid_ids = np.flatnonzero(valid)
    x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, depth_m.shape[1] - 1)
    y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, depth_m.shape[0] - 1)
    distance_source_px = distance_image[y, x].astype(np.float64) / float(np.mean(scale))
    selected = distance_source_px <= float(args.near_mask_px)
    if int(np.count_nonzero(selected)) < int(args.min_depth_vertices):
        return {
            "case": row.get("case"),
            "pair_contact_variable_id": require_str(row.get("pair_contact_variable_id"), "pair_contact_variable_id"),
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": object_id,
            "depth_gap_state": "unobserved_pair_depth",
            "missing_depth_evidence": ["near_mask_hand_vertices_for_depth"],
            "projected_vertices": int(len(valid_ids)),
            "near_mask_projected_vertices": int(np.count_nonzero(selected)),
            "metric_depth_compatible_candidate": False,
            "physical_contact_factor_ready": False,
            **FALSE_READY,
        }
    selected_ids = valid_ids[selected]
    sx = x[selected]
    sy = y[selected]
    hand_z = points_camera[selected_ids, 2].astype(np.float64)
    object_z = depth_m[sy, sx].astype(np.float64)
    depth_valid = np.isfinite(object_z) & (object_z >= float(args.min_depth_m)) & (object_z <= float(args.max_depth_m))
    if int(np.count_nonzero(depth_valid)) < int(args.min_depth_vertices):
        return {
            "case": row.get("case"),
            "pair_contact_variable_id": require_str(row.get("pair_contact_variable_id"), "pair_contact_variable_id"),
            "frame_idx": frame_idx,
            "hand_side": side,
            "object_id": object_id,
            "depth_gap_state": "unobserved_pair_depth",
            "missing_depth_evidence": ["valid_unidepth_at_near_mask_hand_vertices"],
            "projected_vertices": int(len(valid_ids)),
            "near_mask_projected_vertices": int(np.count_nonzero(selected)),
            "valid_depth_vertices": int(np.count_nonzero(depth_valid)),
            "metric_depth_compatible_candidate": False,
            "physical_contact_factor_ready": False,
            **FALSE_READY,
        }
    hand_z = hand_z[depth_valid]
    object_z = object_z[depth_valid]
    distances = distance_source_px[selected][depth_valid]
    gap = hand_z - object_z
    abs_gap = np.abs(gap)
    compatible = bool(
        hand_residual_ok(hand, args)
        and len(gap) >= int(args.min_depth_vertices)
        and abs(float(np.median(gap))) <= float(args.max_median_abs_depth_gap_m)
        and float(np.percentile(abs_gap, 95.0)) <= float(args.max_p95_abs_depth_gap_m)
    )
    if compatible:
        state = "image_supported_metric_depth_compatible_without_contact_geometry"
    elif float(np.median(gap)) > float(args.max_median_abs_depth_gap_m):
        state = "hand_behind_object_depth"
    elif float(np.median(gap)) < -float(args.max_median_abs_depth_gap_m):
        state = "hand_in_front_of_object_depth"
    else:
        state = "depth_tail_incompatible"
    object_ready = require_dict(row.get("object_readiness_checks"), "object_readiness_checks")
    physical_ready = bool(compatible and object_ready.get("can_own_contact_factors") is True)
    blockers = []
    if not compatible:
        blockers.append("hand/object camera depths at contact pixels are incompatible")
    if object_ready.get("can_own_contact_factors") is not True:
        blockers.append("object geometry cannot own contact factors")
    return {
        "case": row.get("case"),
        "pair_contact_variable_id": require_str(row.get("pair_contact_variable_id"), "pair_contact_variable_id"),
        "frame_idx": frame_idx,
        "hand_side": side,
        "object_id": object_id,
        "track_id": require_str(row.get("track_id"), "track_id"),
        "name": optional_str(row.get("name"), "name"),
        "pair_contact_state": require_str(row.get("pair_contact_state"), "pair_contact_state"),
        "contact_owner_image_supported": bool(row.get("contact_owner_image_supported") is True),
        "pair_contact_image_candidate": bool(row.get("pair_contact_image_candidate") is True),
        "hand_side_contact_factor_ready": bool(
            require_dict(row.get("hand_side_contact_mode"), "hand_side_contact_mode").get("contact_factor_ready") is True
        ),
        "depth_gap_state": state,
        "missing_depth_evidence": [],
        "projected_vertices": int(len(valid_ids)),
        "near_mask_projected_vertices": int(len(distances)),
        "valid_depth_vertices": int(len(gap)),
        "near_mask_distance_px": summarize(distances.astype(float).tolist()),
        "hand_source_depth_m": summarize(hand_z.astype(float).tolist()),
        "object_unidepth_m": summarize(object_z.astype(float).tolist()),
        "hand_minus_object_depth_m": summarize(gap.astype(float).tolist()),
        "abs_hand_minus_object_depth_m": summarize(abs_gap.astype(float).tolist()),
        "hand_projection_grid": {
            "hand_source_size_wh": [float(hand_w), float(hand_h)],
            "depth_source_size_wh": [int(depth_w), int(depth_h)],
            "source_to_depth_scale_xy": [float(scale[0]), float(scale[1])],
            "hand_source_intrinsics_fx_fy_cx_cy": intr.astype(float).tolist(),
            "depth_intrinsics_fx_fy_cx_cy": depth["intrinsics"][int(depth_i)].astype(float).tolist(),
        },
        "metric_depth_compatible_candidate": compatible,
        "physical_contact_factor_ready": physical_ready,
        "physical_contact_factor_blockers": blockers,
        **FALSE_READY,
    }


def object_state_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("objects"), "object hypothesis rows")):
        row = require_dict(raw, f"object hypothesis rows[{i}]")
        object_id = require_str(row.get("object_id"), f"object hypothesis rows[{i}].object_id")
        out[object_id] = row
    return out


def rows_by_object(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(require_str(row.get("object_id"), "object_id"), []).append(row)
    return out


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "pairwise_contact_state": existing_path(
            args.pairwise_contact_state_root / case / "v17_pairwise_contact_state.json",
            f"{case} pairwise contact state",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "object_geometry_hypothesis_state": existing_path(
            args.object_geometry_hypothesis_state_root / case / "v17_object_geometry_hypothesis_state_report.json",
            f"{case} object-geometry hypothesis state",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_by_frame(payloads["annotations"])
    pairwise = payloads["pairwise_contact_state"]
    visible = payloads["visible_surface"]
    object_states = object_state_index(payloads["object_geometry_hypothesis_state"])
    frame_count = require_int(pairwise.get("frame_count"), "pairwise frame_count")
    if frame_count != len(frames):
        raise RuntimeError(f"{case} pairwise frame_count disagrees with annotation frames")
    depth_path = existing_path(Path(require_str(visible.get("metric_depth_npz"), "visible metric_depth_npz")), "metric depth npz")
    depth = depth_archive(depth_path)
    mask_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for i, raw in enumerate(require_list(pairwise.get("rows"), "pairwise rows")):
        row = require_dict(raw, f"pairwise rows[{i}]")
        if row.get("pair_contact_image_candidate") is True:
            candidate_rows.append(row)
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        frame_idx = require_int(row.get("frame_idx"), "pairwise row frame_idx")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} frame {frame_idx} missing from annotations")
        object_id = require_str(row.get("object_id"), "pairwise row object_id")
        object_state = object_states.get(object_id)
        if object_state is None:
            raise RuntimeError(f"{case} object {object_id} missing from object-geometry state")
        measured = measured_depth_gap(row=row, frame=frame, depth=depth, mask_cache=mask_cache, args=args)
        measured["object_geometry_hypothesis_state"] = require_str(
            object_state.get("geometry_hypothesis_state"),
            "geometry_hypothesis_state",
        )
        rows.append(measured)
    state_counts = Counter(require_str(row.get("depth_gap_state"), "depth_gap_state") for row in rows)
    missing_counts: Counter[str] = Counter()
    for row in rows:
        for reason in require_list(row.get("missing_depth_evidence"), "missing_depth_evidence"):
            missing_counts[require_str(reason, "missing_depth_evidence reason")] += 1
    measured_rows = [row for row in rows if not row.get("missing_depth_evidence")]
    compatible_rows = [row for row in rows if row.get("metric_depth_compatible_candidate") is True]
    physical_ready = [row for row in rows if row.get("physical_contact_factor_ready") is True]
    by_object = rows_by_object(rows)
    object_summaries = [
        {
            "object_id": object_id,
            "pair_depth_rows": len(object_rows),
            "metric_depth_compatible_candidate_rows": sum(
                1 for row in object_rows if row.get("metric_depth_compatible_candidate") is True
            ),
            "physical_contact_factor_ready_rows": sum(
                1 for row in object_rows if row.get("physical_contact_factor_ready") is True
            ),
            "depth_gap_state_counts": dict(
                sorted(Counter(require_str(row.get("depth_gap_state"), "depth_gap_state") for row in object_rows).items())
            ),
            "abs_hand_minus_object_depth_m": summarize(
                [
                    finite_float(require_dict(row.get("abs_hand_minus_object_depth_m"), "abs depth").get("median"), "abs median")
                    for row in object_rows
                    if isinstance(row.get("abs_hand_minus_object_depth_m"), dict)
                    and require_dict(row.get("abs_hand_minus_object_depth_m"), "abs depth").get("median") is not None
                ]
            ),
        }
        for object_id, object_rows in sorted(by_object.items())
    ]
    report = {
        "method": "build_v17_pairwise_contact_depth_gap",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(paths[name], payloads[name]) for name in payloads},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "pairwise_contact_variable_count": require_int(
            pairwise.get("pairwise_contact_variable_count"),
            "pairwise contact variable count",
        ),
        "pair_contact_image_candidate_rows": require_int(
            pairwise.get("pair_contact_image_candidate_rows"),
            "pairwise contact image candidate rows",
        ),
        "evaluated_pair_depth_rows": len(rows),
        "measured_pair_depth_rows": len(measured_rows),
        "unobserved_pair_depth_rows": len(rows) - len(measured_rows),
        "metric_depth_compatible_candidate_rows": len(compatible_rows),
        "physical_contact_factor_ready_rows": len(physical_ready),
        "depth_gap_state_counts": dict(sorted(state_counts.items())),
        "missing_depth_evidence_reason_counts": dict(sorted(missing_counts.items())),
        "hand_minus_object_depth_m": summarize(
            [
                finite_float(require_dict(row.get("hand_minus_object_depth_m"), "depth gap").get("median"), "gap median")
                for row in measured_rows
                if isinstance(row.get("hand_minus_object_depth_m"), dict)
                and require_dict(row.get("hand_minus_object_depth_m"), "depth gap").get("median") is not None
            ]
        ),
        "abs_hand_minus_object_depth_m": summarize(
            [
                finite_float(require_dict(row.get("abs_hand_minus_object_depth_m"), "abs depth gap").get("median"), "abs gap median")
                for row in measured_rows
                if isinstance(row.get("abs_hand_minus_object_depth_m"), dict)
                and require_dict(row.get("abs_hand_minus_object_depth_m"), "abs depth gap").get("median") is not None
            ]
        ),
        "object_summaries": object_summaries,
        "rows": rows,
        "problem_semantics": {
            "variable": "contact_pair_depth_gap[frame_idx, hand_side, object_id]",
            "domain": "pair_contact_image_candidate rows from v17_pairwise_contact_state",
            "signed_gap": "hand_camera_z_minus_object_unidepth_z_at_near-mask hand vertices",
            "positive_gap_meaning": "hand surface is behind the object visible depth at the sampled pixels",
            "physical_factor_rule": "A metric-depth-compatible image pair is still not a physical contact factor unless the same object has contact-compatible geometry and pose state.",
        },
        "parameters": {
            "near_mask_px": float(args.near_mask_px),
            "min_depth_vertices": int(args.min_depth_vertices),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
            "max_hand_median_px": float(args.max_hand_median_px),
            "max_hand_p95_px": float(args.max_hand_p95_px),
        },
        **FALSE_READY,
    }
    if report["evaluated_pair_depth_rows"] != report["pair_contact_image_candidate_rows"]:
        raise RuntimeError(f"{case} pair-depth rows must match pair_contact_image_candidate rows")
    if report["physical_contact_factor_ready_rows"] != 0:
        raise RuntimeError(f"{case} pair-depth layer must not emit physical factors before object geometry is ready")
    write_json(args.output_root / case / "v17_pairwise_contact_depth_gap.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.pairwise_contact_state_root / "v17_pairwise_contact_state_summary.json",
        "pairwise contact state summary",
    )
    summary = require_dict(load_json(summary_path), "pairwise contact state summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "summary case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_pairwise_contact_depth_gap",
        "status": STATUS,
        "claim": CLAIM,
        "source_pairwise_contact_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_pairwise_contact_depth_gap.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "evaluated_pair_depth_rows": require_int(
                    report.get("evaluated_pair_depth_rows"),
                    "evaluated pair-depth rows",
                ),
                "measured_pair_depth_rows": require_int(
                    report.get("measured_pair_depth_rows"),
                    "measured pair-depth rows",
                ),
                "metric_depth_compatible_candidate_rows": require_int(
                    report.get("metric_depth_compatible_candidate_rows"),
                    "metric-depth-compatible rows",
                ),
                "physical_contact_factor_ready_rows": require_int(
                    report.get("physical_contact_factor_ready_rows"),
                    "physical factor ready rows",
                ),
                "depth_gap_state_counts": require_dict(
                    report.get("depth_gap_state_counts"),
                    "depth gap state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "evaluated_pair_depth_rows": sum(
            require_int(report.get("evaluated_pair_depth_rows"), "evaluated pair-depth rows")
            for report in reports
        ),
        "measured_pair_depth_rows": sum(
            require_int(report.get("measured_pair_depth_rows"), "measured pair-depth rows")
            for report in reports
        ),
        "unobserved_pair_depth_rows": sum(
            require_int(report.get("unobserved_pair_depth_rows"), "unobserved pair-depth rows")
            for report in reports
        ),
        "metric_depth_compatible_candidate_rows": sum(
            require_int(report.get("metric_depth_compatible_candidate_rows"), "metric-compatible rows")
            for report in reports
        ),
        "physical_contact_factor_ready_rows": sum(
            require_int(report.get("physical_contact_factor_ready_rows"), "physical factor rows")
            for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_pairwise_contact_depth_gap_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--pairwise-contact-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--object-geometry-hypothesis-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"),
    )
    parser.add_argument("--near-mask-px", type=float, default=20.0)
    parser.add_argument("--min-depth-vertices", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
