#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_v17_hand_intrinsics_depth_counterfactual import (
    annotation_hand_index,
    front_surface_depth_samples,
    local_hand_geometry,
    projection_residual,
    scale_depth_intrinsics,
    solve_translation,
    source_intrinsics,
    source_size_from_intrinsics,
)
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    active_object_distance,
    annotation_frames,
    depth_archive,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)


STATUS = "v17_hand_tail_support_state_qc"
CLAIM = (
    "This artifact tests whether residual hand-depth tail pixels, after the intrinsics and per-row "
    "scale counterfactuals, are supported by model-produced 2D hand evidence. Unsupported tails point to "
    "local MANO surface or pose projection error; supported tails point to depth-observation or local hand-surface "
    "metric mismatch."
)


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def source_summary(path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if payload is not None:
        out["status"] = payload.get("status")
        out["method"] = payload.get("method")
    return out


def optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_box(value: Any, label: str) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    out: list[float] = []
    for i, raw in enumerate(value):
        v = optional_finite(raw)
        if v is None:
            raise RuntimeError(f"{label}[{i}] must be finite")
        out.append(v)
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def array2d(value: Any, width: int, label: str) -> np.ndarray | None:
    if not isinstance(value, list):
        return None
    if value and all(isinstance(item, dict) for item in value):
        rows: list[list[float]] = []
        for i, item in enumerate(value):
            row = require_dict(item, f"{label}[{i}]")
            x = optional_finite(row.get("x"))
            y = optional_finite(row.get("y"))
            if x is None or y is None:
                return None
            rows.append([x, y])
        return np.asarray(rows, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < width:
        return None
    arr = arr[:, :width]
    if not np.isfinite(arr).all():
        return None
    return arr


def bbox_center(box: list[float]) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def load_measurements(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [require_dict(row, f"{path.name} row") for row in require_list(payload, path.name)]


def index_measurements_by_frame(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_idx = row.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        out.setdefault(frame_idx, []).append(row)
    return out


def measurement_side(row: dict[str, Any]) -> str | None:
    side = row.get("side")
    if isinstance(side, str) and side:
        return side
    entity_id = row.get("entity_id")
    if isinstance(entity_id, str):
        if entity_id.endswith(":left") or entity_id == "hand:left":
            return "left"
        if entity_id.endswith(":right") or entity_id == "hand:right":
            return "right"
    side_hint = row.get("side_hint")
    if isinstance(side_hint, str) and side_hint in {"left", "right"}:
        return side_hint
    return None


def shape_from_measurement(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    box = finite_box(row.get("bbox_xyxy"), f"{source_name} bbox")
    if box is None:
        return None
    keypoints = array2d(row.get("keypoints"), 2, f"{source_name} keypoints")
    if keypoints is None:
        keypoints = array2d(row.get("positive_points"), 2, f"{source_name} positive_points")
    return {
        "source": source_name,
        "measurement_id": row.get("measurement_id"),
        "side": measurement_side(row),
        "confidence": optional_finite(row.get("confidence")),
        "bbox_xyxy": box,
        "keypoints": None if keypoints is None else keypoints,
    }


def annotation_shape(hand: dict[str, Any]) -> dict[str, Any]:
    box = finite_box(hand.get("bbox_xyxy"), "annotation hand bbox")
    if box is None:
        joints = array2d(hand.get("joints2d_raw"), 2, "annotation joints2d_raw")
        if joints is None:
            joints = array2d(hand.get("joints2d"), 2, "annotation joints2d")
        if joints is None:
            raise RuntimeError("annotation hand has neither bbox nor 2D joints")
        x0, y0 = np.min(joints, axis=0)
        x1, y1 = np.max(joints, axis=0)
        box = [float(x0), float(y0), float(x1), float(y1)]
    keypoints = array2d(hand.get("joints2d_raw"), 2, "annotation joints2d_raw")
    if keypoints is None:
        keypoints = array2d(hand.get("joints2d"), 2, "annotation joints2d")
    return {
        "source": "selected_annotation_hand",
        "measurement_id": "selected_annotation_hand",
        "side": hand.get("side"),
        "confidence": optional_finite(hand.get("detector_score")),
        "bbox_xyxy": box,
        "keypoints": None if keypoints is None else keypoints,
    }


def assign_side(
    shape: dict[str, Any],
    frame_hands: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str | None, int | None]:
    explicit_side = shape.get("side")
    if isinstance(explicit_side, str) and explicit_side in {"left", "right"}:
        candidates = [
            (i, hand)
            for i, hand in enumerate(frame_hands)
            if require_str(hand.get("side"), "hand side") == explicit_side
        ]
    else:
        candidates = [(i, hand) for i, hand in enumerate(frame_hands)]
    if not candidates:
        return explicit_side if isinstance(explicit_side, str) else None, None
    center = bbox_center(require_list(shape["bbox_xyxy"], "shape bbox"))
    best_i: int | None = None
    best_dist = float("inf")
    for hand_i, hand in candidates:
        try:
            hand_box = annotation_shape(hand)["bbox_xyxy"]
        except RuntimeError:
            continue
        dist = float(np.linalg.norm(center - bbox_center(require_list(hand_box, "hand bbox"))))
        if dist < best_dist:
            best_dist = dist
            best_i = hand_i
    if best_i is None or best_dist > float(args.max_assign_center_px):
        return explicit_side if isinstance(explicit_side, str) else None, None
    side = require_str(frame_hands[best_i].get("side"), "assigned hand side")
    return side, best_i


def case_support_sources(case: str, args: argparse.Namespace) -> dict[str, Any]:
    root = args.measurement_store_root / case / "measurements_v17"
    paths = {
        "rtmlib_hand2d": existing_path(root / "rtmlib_hand2d_measurements.json", f"{case} RTMLib hand 2D"),
        "wilor": existing_path(root / "wilor_measurements.json", f"{case} WiLoR measurements"),
        "hamer": existing_path(root / "hamer_measurements.json", f"{case} HaMeR measurements"),
        "vlm_hand_box": existing_path(root / "vlm_hand_box_measurements.json", f"{case} VLM hand boxes"),
    }
    return {
        "paths": paths,
        "rows_by_source": {name: index_measurements_by_frame(load_measurements(path)) for name, path in paths.items()},
    }


def support_shapes_for_row(
    *,
    frame: dict[str, Any],
    hand_i: int,
    support_sources: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    hands = [require_dict(raw, "frame hand") for raw in require_list(frame.get("hands"), "frame hands")]
    hand = hands[hand_i]
    side = require_str(hand.get("side"), "hand side")
    exact = annotation_shape(hand)
    exact["assigned_side"] = side
    exact["assigned_hand_index"] = hand_i
    same_side_independent: list[dict[str, Any]] = []
    any_hand_independent: list[dict[str, Any]] = []
    frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
    for source_name, by_frame in require_dict(support_sources["rows_by_source"], "support sources").items():
        for raw in by_frame.get(frame_idx, []):
            shape = shape_from_measurement(require_dict(raw, "support row"), source_name)
            if shape is None:
                continue
            assigned_side, assigned_i = assign_side(shape, hands, args)
            shape = {**shape, "assigned_side": assigned_side, "assigned_hand_index": assigned_i}
            any_hand_independent.append(shape)
            if assigned_side == side and assigned_i in {None, hand_i}:
                same_side_independent.append(shape)
    return {
        "selected_annotation_hand": [exact],
        "same_side_independent_models": same_side_independent,
        "all_independent_models": any_hand_independent,
    }


def scale_box_to_depth(
    box: list[float],
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    depth_h, depth_w = depth_shape
    sx = float(depth_w) / float(projection_source_size[0])
    sy = float(depth_h) / float(projection_source_size[1])
    return np.asarray([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], dtype=np.float64)


def scale_points_to_depth(
    points: np.ndarray,
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    depth_h, depth_w = depth_shape
    scale = np.asarray(
        [float(depth_w) / float(projection_source_size[0]), float(depth_h) / float(projection_source_size[1])],
        dtype=np.float64,
    )
    return points[:, :2] * scale[None, :]


def bbox_distance(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    dx = np.maximum(np.maximum(box[0] - points[:, 0], 0.0), points[:, 0] - box[2])
    dy = np.maximum(np.maximum(box[1] - points[:, 1], 0.0), points[:, 1] - box[3])
    return np.sqrt(dx * dx + dy * dy)


def min_bbox_distance(
    points: np.ndarray,
    shapes: list[dict[str, Any]],
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    if not shapes:
        return np.full(len(points), np.inf, dtype=np.float64)
    distances = []
    for shape in shapes:
        box = scale_box_to_depth(require_list(shape["bbox_xyxy"], "shape bbox"), projection_source_size, depth_shape)
        distances.append(bbox_distance(points, box))
    return np.min(np.stack(distances, axis=0), axis=0)


def min_keypoint_distance(
    points: np.ndarray,
    shapes: list[dict[str, Any]],
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    keypoints = []
    for shape in shapes:
        raw = shape.get("keypoints")
        if isinstance(raw, np.ndarray) and len(raw) > 0:
            keypoints.append(scale_points_to_depth(raw, projection_source_size, depth_shape))
    if not keypoints:
        return np.full(len(points), np.inf, dtype=np.float64)
    pts = np.concatenate(keypoints, axis=0)
    delta = points[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2)).min(axis=1)


def support_summary(
    *,
    points: np.ndarray,
    shapes: list[dict[str, Any]],
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "sample_count": 0,
            "shape_count": len(shapes),
            "inside_bbox_fraction": None,
            "near_bbox_fraction": None,
            "near_keypoint_fraction": None,
            "bbox_distance_px": summarize([]),
            "keypoint_distance_px": summarize([]),
        }
    bbox_dist = min_bbox_distance(points, shapes, projection_source_size, depth_shape)
    keypoint_dist = min_keypoint_distance(points, shapes, projection_source_size, depth_shape)
    finite_bbox = bbox_dist[np.isfinite(bbox_dist)]
    finite_keypoint = keypoint_dist[np.isfinite(keypoint_dist)]
    return {
        "sample_count": int(len(points)),
        "shape_count": len(shapes),
        "inside_bbox_fraction": float(np.mean(bbox_dist <= 0.0)) if len(finite_bbox) else None,
        "near_bbox_fraction": float(np.mean(bbox_dist <= float(args.near_support_bbox_px))) if len(finite_bbox) else None,
        "near_keypoint_fraction": float(np.mean(keypoint_dist <= float(args.near_support_keypoint_px)))
        if len(finite_keypoint)
        else None,
        "bbox_distance_px": summarize(finite_bbox.astype(float).tolist()),
        "keypoint_distance_px": summarize(finite_keypoint.astype(float).tolist()),
    }


def subset_support(
    *,
    x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    shapes: dict[str, list[dict[str, Any]]],
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    points = np.stack([x[selected].astype(np.float64), y[selected].astype(np.float64)], axis=1)
    return {
        name: support_summary(
            points=points,
            shapes=shape_rows,
            projection_source_size=projection_source_size,
            depth_shape=depth_shape,
            args=args,
        )
        for name, shape_rows in shapes.items()
    }


def selected_support_state(row: dict[str, Any], subset: dict[str, Any]) -> str:
    if row.get("tail_factor_candidate") is not True:
        return "not_tail_factor_candidate"
    exact = require_dict(subset.get("selected_annotation_hand"), "selected annotation hand support")
    exact_inside = optional_finite(exact.get("inside_bbox_fraction"))
    exact_near = optional_finite(exact.get("near_bbox_fraction"))
    if exact_inside is not None and exact_inside >= 0.75:
        return "tail_pixels_inside_selected_hand_box"
    if exact_near is not None and exact_near >= 0.75:
        return "tail_pixels_near_selected_hand_box"
    return "tail_pixels_outside_selected_hand_box"


def independent_support_state(row: dict[str, Any], subset: dict[str, Any]) -> str:
    if row.get("tail_factor_candidate") is not True:
        return "not_tail_factor_candidate"
    same_side = require_dict(subset.get("same_side_independent_models"), "same-side independent support")
    all_models = require_dict(subset.get("all_independent_models"), "all independent support")
    same_inside = optional_finite(same_side.get("inside_bbox_fraction"))
    same_near = optional_finite(same_side.get("near_bbox_fraction"))
    all_inside = optional_finite(all_models.get("inside_bbox_fraction"))
    all_near = optional_finite(all_models.get("near_bbox_fraction"))
    if same_inside is not None and same_inside >= 0.75:
        return "tail_pixels_inside_same_side_independent_model_box"
    if same_near is not None and same_near >= 0.75:
        return "tail_pixels_near_same_side_independent_model_box"
    if all_inside is not None and all_inside >= 0.75:
        return "tail_pixels_inside_other_independent_model_box"
    if all_near is not None and all_near >= 0.75:
        return "tail_pixels_near_other_independent_model_box"
    return "tail_pixels_unsupported_by_independent_model_boxes"


def recompute_row_samples(
    *,
    case: str,
    tail_row: dict[str, Any],
    frame: dict[str, Any],
    hand: dict[str, Any],
    depth: dict[str, Any],
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    frame_idx = require_int(tail_row.get("frame_idx"), "tail row frame_idx")
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        return None
    hand_intr = source_intrinsics(hand)
    geometry = local_hand_geometry(hand)
    if hand_intr is None or geometry is None:
        return None
    local_joints, local_vertices, keypoints2d = geometry
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    depth_intr = depth["intrinsics"][int(depth_i)].astype(np.float64)
    projection_source_size = source_size_from_intrinsics(hand_intr)
    candidate_intrinsics = scale_depth_intrinsics(depth_intr, depth["source_size"], projection_source_size)
    translation = solve_translation(local_joints, keypoints2d, candidate_intrinsics)
    source_vertices = local_vertices + translation[None, :]
    residual = projection_residual(local_joints, keypoints2d, translation, candidate_intrinsics, args)
    samples = front_surface_depth_samples(
        source_vertices,
        candidate_intrinsics,
        projection_source_size,
        (depth_m.shape[0], depth_m.shape[1]),
    )
    if samples is None:
        return None
    x = samples["x"].astype(np.int32)
    y = samples["y"].astype(np.int32)
    hand_z = samples["hand_z"].astype(np.float64)
    scale = optional_finite(tail_row.get("per_row_scale"))
    if scale is None or scale <= 0.0:
        return None
    hand_z = hand_z * scale
    metric_z = depth_m[y, x].astype(np.float64)
    object_distance = active_object_distance(frame, (depth_m.shape[0], depth_m.shape[1]), mask_cache)
    object_distance_px = object_distance["distance_source_px"][y, x].astype(np.float64)
    finite_distance = np.isfinite(object_distance_px)
    far = (~finite_distance) | (object_distance_px >= float(args.far_object_mask_px))
    owner_label = tail_row.get("owner_sample_partition")
    if owner_label == "far_from_active_object_masks":
        selected = far
    elif owner_label == "all_projected_hand_pixels":
        selected = np.ones(len(hand_z), dtype=bool)
    else:
        selected = np.zeros(len(hand_z), dtype=bool)
    valid = (
        selected
        & np.isfinite(hand_z)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    if int(np.count_nonzero(valid)) < int(args.min_depth_pixels):
        return None
    gap = hand_z - metric_z
    return {
        "x": x,
        "y": y,
        "hand_z": hand_z,
        "metric_z": metric_z,
        "gap": gap,
        "valid": valid,
        "projection_source_size": projection_source_size,
        "depth_shape": (int(depth_m.shape[0]), int(depth_m.shape[1])),
        "projection_residual_to_measurement_px": residual,
    }


def case_problem(case: str, args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "annotations": existing_path(
            args.graph_root / case / "annotations_v17_full_timeline_graph.json",
            f"{case} graph annotations",
        ),
        "visible_surface": existing_path(
            args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
            f"{case} visible-surface report",
        ),
        "tail_state": existing_path(
            args.hand_surface_depth_tail_state_root / case / "v17_hand_surface_depth_tail_state.json",
            f"{case} hand surface-depth tail state report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    tail_state_payload = payloads["tail_state"]
    depth = depth_archive(existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive"))
    support_sources = case_support_sources(case, args)
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    for raw in require_list(tail_state_payload.get("rows"), "tail rows"):
        tail_row = require_dict(raw, "tail row")
        frame_idx = require_int(tail_row.get("frame_idx"), "tail row frame_idx")
        side = require_str(tail_row.get("hand_side"), "tail row hand_side")
        hand_i = require_int(tail_row.get("hand_index"), "tail row hand_index")
        frame = frames.get(frame_idx)
        hand = hand_index.get((frame_idx, side, hand_i))
        base = {
            "case": case,
            "hand_tail_support_variable_id": require_str(
                tail_row.get("hand_surface_depth_tail_variable_id"),
                "tail variable id",
            ).replace("hand_surface_depth_tail:", "hand_tail_support:", 1),
            "source_hand_surface_depth_tail_variable_id": require_str(
                tail_row.get("hand_surface_depth_tail_variable_id"),
                "tail variable id",
            ),
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_i,
            "tail_state": tail_row.get("tail_state"),
            "tail_factor_candidate": bool(tail_row.get("tail_factor_candidate") is True),
            "tail_pattern": tail_row.get("tail_pattern"),
            "owner_sample_partition": tail_row.get("owner_sample_partition"),
            **FALSE_READY,
        }
        if frame is None or hand is None:
            rows.append(
                {
                    **base,
                    "selected_support_state": "missing_annotation_hand",
                    "independent_support_state": "missing_annotation_hand",
                    "missing_support_inputs": ["annotation_hand"],
                }
            )
            continue
        samples = recompute_row_samples(
            case=case,
            tail_row=tail_row,
            frame=frame,
            hand=hand,
            depth=depth,
            mask_cache=mask_cache,
            args=args,
        )
        if samples is None:
            rows.append(
                {
                    **base,
                    "selected_support_state": "unobserved_tail_pixels_for_support",
                    "independent_support_state": "unobserved_tail_pixels_for_support",
                    "missing_support_inputs": ["recomputed_tail_pixels"],
                }
            )
            continue
        shapes = support_shapes_for_row(
            frame=frame,
            hand_i=hand_i,
            support_sources=support_sources,
            args=args,
        )
        valid = samples["valid"]
        gap = samples["gap"]
        abs_tail = valid & (np.abs(gap) > float(args.max_p95_abs_depth_gap_m))
        negative_tail = valid & (gap < -float(args.max_p95_abs_depth_gap_m))
        positive_tail = valid & (gap > float(args.max_p95_abs_depth_gap_m))
        support = {
            "valid_owner_samples": subset_support(
                x=samples["x"],
                y=samples["y"],
                selected=valid,
                shapes=shapes,
                projection_source_size=samples["projection_source_size"],
                depth_shape=samples["depth_shape"],
                args=args,
            ),
            "abs_tail_samples": subset_support(
                x=samples["x"],
                y=samples["y"],
                selected=abs_tail,
                shapes=shapes,
                projection_source_size=samples["projection_source_size"],
                depth_shape=samples["depth_shape"],
                args=args,
            ),
            "negative_tail_samples": subset_support(
                x=samples["x"],
                y=samples["y"],
                selected=negative_tail,
                shapes=shapes,
                projection_source_size=samples["projection_source_size"],
                depth_shape=samples["depth_shape"],
                args=args,
            ),
            "positive_tail_samples": subset_support(
                x=samples["x"],
                y=samples["y"],
                selected=positive_tail,
                shapes=shapes,
                projection_source_size=samples["projection_source_size"],
                depth_shape=samples["depth_shape"],
                args=args,
            ),
        }
        rows.append(
            {
                **base,
                "selected_support_state": selected_support_state(
                    base,
                    require_dict(support["abs_tail_samples"], "abs tail support"),
                ),
                "independent_support_state": independent_support_state(
                    base,
                    require_dict(support["abs_tail_samples"], "abs tail support"),
                ),
                "valid_owner_sample_count": int(np.count_nonzero(valid)),
                "abs_tail_sample_count": int(np.count_nonzero(abs_tail)),
                "negative_tail_sample_count": int(np.count_nonzero(negative_tail)),
                "positive_tail_sample_count": int(np.count_nonzero(positive_tail)),
                "projection_residual_to_measurement_px": samples["projection_residual_to_measurement_px"],
                "support_shape_counts": {name: len(value) for name, value in shapes.items()},
                "support": support,
                "missing_support_inputs": [],
            }
        )
    tail_rows = [row for row in rows if row.get("tail_factor_candidate") is True]
    report = {
        "method": "build_v17_hand_tail_support_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            **{name: source_summary(path, payloads[name]) for name, path in paths.items()},
            **{
                f"support_{name}": source_summary(path)
                for name, path in require_dict(support_sources["paths"], "support paths").items()
            },
        },
        "frame_count": require_int(tail_state_payload.get("frame_count"), f"{case} tail frame_count"),
        "hand_tail_support_variable_count": len(rows),
        "tail_factor_candidate_rows": len(tail_rows),
        "selected_support_state_counts": dict(
            sorted(Counter(require_str(row.get("selected_support_state"), "selected support state") for row in rows).items())
        ),
        "independent_support_state_counts": dict(
            sorted(
                Counter(require_str(row.get("independent_support_state"), "independent support state") for row in rows).items()
            )
        ),
        "tail_selected_support_state_counts": dict(
            sorted(
                Counter(require_str(row.get("selected_support_state"), "selected support state") for row in tail_rows).items()
            )
        ),
        "tail_independent_support_state_counts": dict(
            sorted(
                Counter(
                    require_str(row.get("independent_support_state"), "independent support state") for row in tail_rows
                ).items()
            )
        ),
        "tail_abs_sample_count": sum(require_int(row.get("abs_tail_sample_count", 0), "abs tail samples") for row in tail_rows),
        "tail_negative_sample_count": sum(
            require_int(row.get("negative_tail_sample_count", 0), "negative tail samples") for row in tail_rows
        ),
        "tail_positive_sample_count": sum(
            require_int(row.get("positive_tail_sample_count", 0), "positive tail samples") for row in tail_rows
        ),
        "problem_semantics": {
            "selected_support_state": "location of residual tail pixels relative to the selected annotation hand shape",
            "independent_support_state": "location of residual tail pixels relative to model-produced hand shapes excluding the selected annotation hand shape",
            "tail_pixels_inside_selected_hand_box": "most residual tail pixels lie inside the selected hand detector box",
            "tail_pixels_unsupported_by_independent_model_boxes": "most residual tail pixels lie outside independent model-produced hand boxes",
            "tail_pixels_inside_same_side_independent_model_box": "most residual tail pixels lie inside an independent hand box assigned to the same side",
            "tail_pixels_inside_other_independent_model_box": "most residual tail pixels lie inside an independent hand box assigned to another hand",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_tail_support_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_surface_depth_tail_state_root / "v17_hand_surface_depth_tail_state_summary.json",
        "hand surface-depth tail summary",
    )
    summary = require_dict(load_json(summary_path), "hand surface-depth tail summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_hand_tail_support_state",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_surface_depth_tail_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_tail_support_state.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_tail_support_variable_count": require_int(
                    report.get("hand_tail_support_variable_count"),
                    "support variable count",
                ),
                "tail_factor_candidate_rows": require_int(
                    report.get("tail_factor_candidate_rows"),
                    "tail candidate rows",
                ),
                "tail_selected_support_state_counts": require_dict(
                    report.get("tail_selected_support_state_counts"),
                    "tail selected support state counts",
                ),
                "tail_independent_support_state_counts": require_dict(
                    report.get("tail_independent_support_state_counts"),
                    "tail independent support state counts",
                ),
                "tail_abs_sample_count": require_int(report.get("tail_abs_sample_count"), "tail abs sample count"),
                "tail_negative_sample_count": require_int(
                    report.get("tail_negative_sample_count"),
                    "tail negative sample count",
                ),
                "tail_positive_sample_count": require_int(
                    report.get("tail_positive_sample_count"),
                    "tail positive sample count",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_tail_support_variable_count": sum(
            require_int(report.get("hand_tail_support_variable_count"), "support variable count")
            for report in reports
        ),
        "tail_factor_candidate_rows": sum(
            require_int(report.get("tail_factor_candidate_rows"), "tail candidate rows")
            for report in reports
        ),
        "tail_selected_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("tail_selected_support_state_counts"), "tail selected support counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "tail_independent_support_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("tail_independent_support_state_counts"),
                                "tail independent support counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "tail_abs_sample_count": sum(
            require_int(report.get("tail_abs_sample_count"), "tail abs sample count") for report in reports
        ),
        "tail_negative_sample_count": sum(
            require_int(report.get("tail_negative_sample_count"), "tail negative sample count") for report in reports
        ),
        "tail_positive_sample_count": sum(
            require_int(report.get("tail_positive_sample_count"), "tail positive sample count") for report in reports
        ),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_tail_support_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--hand-surface-depth-tail-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_tail_support_state"),
    )
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--max-assign-center-px", type=float, default=260.0)
    parser.add_argument("--near-support-bbox-px", type=float, default=8.0)
    parser.add_argument("--near-support-keypoint-px", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
