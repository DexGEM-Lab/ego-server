#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    active_object_distance,
    annotation_frames,
    depth_archive,
    finite_float,
    load_json,
    partition_state,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)


STATUS = "v17_hand_intrinsics_depth_counterfactual_qc"
CLAIM = (
    "This artifact tests a specific hand-depth counterfactual: recompute each source-camera hand translation "
    "with UniDepth intrinsics scaled to the hand image size, then measure front-surface MANO depth against the "
    "same UniDepth raster. It isolates an intrinsics mechanism and does not update accepted hand state."
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


def annotation_hand_index(frames: dict[int, dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for frame_idx, frame in frames.items():
        for hand_i, raw in enumerate(require_list(frame.get("hands", []), f"frame {frame_idx} hands")):
            hand = require_dict(raw, f"frame {frame_idx} hand {hand_i}")
            side = hand.get("side")
            side_key = side if isinstance(side, str) and side else f"hand_{hand_i}"
            key = (frame_idx, side_key, hand_i)
            if key in out:
                raise RuntimeError(f"duplicate hand row {key}")
            out[key] = hand
    return out


def array2d(value: Any, shape1: int, label: str) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != shape1 or not np.all(np.isfinite(arr)):
        return None
    return arr


def source_intrinsics(hand: dict[str, Any]) -> np.ndarray | None:
    try:
        intr = np.asarray(hand.get("source_intrinsics"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if intr.shape != (4,) or not np.all(np.isfinite(intr)) or intr[0] <= 0.0 or intr[1] <= 0.0:
        return None
    return intr


def local_hand_geometry(hand: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    joints = array2d(hand.get("joints3d_camera"), 3, "joints3d_camera")
    if joints is None or len(joints) != 21:
        return None
    vertices = array2d(hand.get("vertices_camera"), 3, "vertices_camera")
    if vertices is None or len(vertices) == 0:
        return None
    keypoints = array2d(hand.get("joints2d_raw"), 2, "joints2d_raw")
    if keypoints is None or len(keypoints) != 21:
        return None
    return joints, vertices, keypoints


def solve_translation(local_points_m: np.ndarray, points2d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    qx = (points2d[:, 0] - cx) / fx
    qy = (points2d[:, 1] - cy) / fy
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y, z), u, v in zip(local_points_m, qx, qy):
        rows.append([1.0, 0.0, -float(u)])
        rhs.append(float(u * z - x))
        rows.append([0.0, 1.0, -float(v)])
        rhs.append(float(v * z - y))
    trans, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None)
    return trans.astype(np.float64)


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-6)
    uv[valid, 0] = intrinsics[0] * points_camera[valid, 0] / z[valid] + intrinsics[2]
    uv[valid, 1] = intrinsics[1] * points_camera[valid, 1] / z[valid] + intrinsics[3]
    return uv, valid


def source_size_from_intrinsics(intrinsics: np.ndarray) -> tuple[float, float]:
    return 2.0 * float(intrinsics[2]), 2.0 * float(intrinsics[3])


def scale_depth_intrinsics(depth_intrinsics: np.ndarray, depth_source_size: tuple[int, int], target_size: tuple[float, float]) -> np.ndarray:
    sx = float(target_size[0]) / float(depth_source_size[0])
    sy = float(target_size[1]) / float(depth_source_size[1])
    return np.asarray(
        [
            float(depth_intrinsics[0]) * sx,
            float(depth_intrinsics[1]) * sy,
            float(depth_intrinsics[2]) * sx,
            float(depth_intrinsics[3]) * sy,
        ],
        dtype=np.float64,
    )


def front_surface_depth_samples(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    depth_h, depth_w = depth_shape
    uv, valid_z = project(points_camera, intrinsics)
    scale = np.asarray(
        [
            float(depth_w) / float(projection_source_size[0]),
            float(depth_h) / float(projection_source_size[1]),
        ],
        dtype=np.float64,
    )
    xy = uv * scale[None, :]
    valid = (
        valid_z
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < depth_w)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < depth_h)
    )
    if not np.any(valid):
        return None
    valid_ids = np.flatnonzero(valid)
    x = np.clip(np.rint(xy[valid, 0]).astype(np.int32), 0, depth_w - 1)
    y = np.clip(np.rint(xy[valid, 1]).astype(np.int32), 0, depth_h - 1)
    hand_z = points_camera[valid_ids, 2].astype(np.float64)
    lin = y.astype(np.int64) * int(depth_w) + x.astype(np.int64)
    order = np.lexsort((hand_z, lin))
    sorted_lin = lin[order]
    keep = np.r_[True, sorted_lin[1:] != sorted_lin[:-1]]
    front = order[keep]
    return {"x": x[front], "y": y[front], "hand_z": hand_z[front]}


def projection_residual(
    local_joints: np.ndarray,
    keypoints2d: np.ndarray,
    translation: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    projected, valid = project(local_joints + translation[None, :], intrinsics)
    if not np.all(valid):
        return {"median": None, "p95": None, "residual_ok": False}
    residual = np.linalg.norm(projected - keypoints2d, axis=1)
    median = float(np.median(residual))
    p95 = float(np.percentile(residual, 95.0))
    return {
        "median": median,
        "p95": p95,
        "residual_ok": bool(median <= float(args.max_hand_median_px) and p95 <= float(args.max_hand_p95_px)),
        "max_median_px": float(args.max_hand_median_px),
        "max_p95_px": float(args.max_hand_p95_px),
    }


def median_gap(partition: dict[str, Any]) -> float | None:
    if partition.get("measured") is not True:
        return None
    summary = partition.get("hand_minus_unidepth_depth_m")
    if not isinstance(summary, dict):
        return None
    return finite_float(summary.get("median"), "median gap")


def owner_partition(row: dict[str, Any]) -> dict[str, Any] | None:
    partitions = row.get("sample_partitions")
    if not isinstance(partitions, dict):
        return None
    far = partitions.get("far_from_active_object_masks")
    all_pixels = partitions.get("all_projected_hand_pixels")
    if isinstance(far, dict) and far.get("measured") is True:
        return far
    if isinstance(all_pixels, dict):
        return all_pixels
    return None


def optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def row_measured(row: dict[str, Any]) -> bool:
    owner = owner_partition(row)
    return bool(owner is not None and owner.get("measured") is True)


def row_residual_ok(row: dict[str, Any]) -> bool:
    residual = row.get("projection_residual_to_measurement_px")
    return bool(isinstance(residual, dict) and residual.get("residual_ok") is True)


def row_repair_candidate(row: dict[str, Any]) -> bool:
    return bool(row_measured(row) and row_residual_ok(row) and row.get("metric_depth_compatible") is not True)


def measure_counterfactual_row(
    *,
    case: str,
    frame: dict[str, Any],
    metric_row: dict[str, Any],
    hand: dict[str, Any] | None,
    depth: dict[str, Any],
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
    side = require_str(metric_row.get("hand_side"), "metric row hand_side")
    hand_index = require_int(metric_row.get("hand_index"), "metric row hand_index")
    row_id = require_str(metric_row.get("hand_metric_depth_variable_id"), "hand_metric_depth_variable_id")
    depth_i = depth["frame_to_i"].get(frame_idx)
    missing: list[str] = []
    if hand is None:
        missing.append("annotation_hand")
    if depth_i is None:
        missing.append("metric_depth_frame")
    hand_intr = source_intrinsics(hand) if hand is not None else None
    if hand_intr is None:
        missing.append("hand_source_intrinsics")
    geometry = local_hand_geometry(hand) if hand is not None else None
    if geometry is None:
        missing.append("local_hand_geometry_or_2d_keypoints")
    if missing:
        return {
            "case": case,
            "hand_intrinsics_counterfactual_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "counterfactual_state": "unobserved_intrinsics_counterfactual",
            "counterfactual_owner_depth_state": "unobserved_intrinsics_counterfactual",
            "missing_counterfactual_inputs": missing,
            "metric_depth_compatible": False,
            **FALSE_READY,
        }
    if hand is None or hand_intr is None or geometry is None or depth_i is None:
        raise RuntimeError("missing counterfactual inputs branch failed to return")
    local_joints, local_vertices, keypoints2d = geometry
    depth_m = depth["depth"][int(depth_i)].astype(np.float64)
    depth_intr = depth["intrinsics"][int(depth_i)].astype(np.float64)
    projection_source_size = source_size_from_intrinsics(hand_intr)
    candidate_intrinsics = scale_depth_intrinsics(
        depth_intr,
        depth["source_size"],
        projection_source_size,
    )
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
        return {
            "case": case,
            "hand_intrinsics_counterfactual_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "counterfactual_state": "unobserved_intrinsics_counterfactual",
            "counterfactual_owner_depth_state": "unobserved_intrinsics_counterfactual",
            "missing_counterfactual_inputs": ["projected_hand_surface_inside_depth_image"],
            "projection_residual_to_measurement_px": residual,
            "metric_depth_compatible": False,
            **FALSE_READY,
        }
    x = samples["x"].astype(np.int32)
    y = samples["y"].astype(np.int32)
    hand_z = samples["hand_z"].astype(np.float64)
    metric_z = depth_m[y, x].astype(np.float64)
    object_distance = active_object_distance(frame, (depth_m.shape[0], depth_m.shape[1]), mask_cache)
    object_distance_px = object_distance["distance_source_px"][y, x].astype(np.float64)
    finite_distance = np.isfinite(object_distance_px)
    near = finite_distance & (object_distance_px <= float(args.near_object_mask_px))
    far = (~finite_distance) | (object_distance_px >= float(args.far_object_mask_px))
    residual_ok = bool(residual.get("residual_ok") is True)
    partitions = {
        "all_projected_hand_pixels": partition_state(
            label="all_projected_hand_pixels",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=np.ones(len(hand_z), dtype=bool),
            residual_ok=residual_ok,
            args=args,
        ),
        "near_active_object_masks": partition_state(
            label="near_active_object_masks",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=near,
            residual_ok=residual_ok,
            args=args,
        ),
        "far_from_active_object_masks": partition_state(
            label="far_from_active_object_masks",
            hand_z=hand_z,
            metric_z=metric_z,
            object_distance_px=object_distance_px,
            selected=far,
            residual_ok=residual_ok,
            args=args,
        ),
    }
    far_part = partitions["far_from_active_object_masks"]
    all_part = partitions["all_projected_hand_pixels"]
    owner = far_part if far_part.get("measured") is True else all_part
    state = require_str(owner.get("state"), "probe owner state")
    compatible = bool(owner.get("metric_depth_compatible") is True)
    old_owner = owner_partition(metric_row)
    old_gap = median_gap(old_owner) if old_owner is not None else None
    new_gap = median_gap(owner)
    improved = bool(
        old_gap is not None
        and new_gap is not None
        and abs(float(new_gap)) < abs(float(old_gap))
    )
    if compatible:
        counterfactual_state = "metric_depth_compatible_under_unidepth_intrinsics"
    elif residual_ok and owner.get("measured") is True and improved:
        counterfactual_state = "improved_but_depth_incompatible_under_unidepth_intrinsics"
    elif residual_ok and owner.get("measured") is True:
        counterfactual_state = "measured_but_unimproved_under_unidepth_intrinsics"
    elif owner.get("measured") is True:
        counterfactual_state = "metric_depth_measured_projection_untrusted_under_unidepth_intrinsics"
    else:
        counterfactual_state = "unobserved_intrinsics_counterfactual"
    return {
        "case": case,
        "hand_intrinsics_counterfactual_variable_id": row_id,
        "frame_idx": frame_idx,
        "hand_side": side,
        "hand_index": hand_index,
        "counterfactual_state": counterfactual_state,
        "counterfactual_owner_depth_state": state,
        "metric_depth_compatible": compatible,
        "counterfactual_depth_repair_factor_candidate": row_repair_candidate(
            {
                "sample_partitions": partitions,
                "projection_residual_to_measurement_px": residual,
                "metric_depth_compatible": compatible,
            }
        ),
        "median_gap_improved_vs_current": improved,
        "current_owner_median_gap_m": old_gap,
        "counterfactual_owner_median_gap_m": new_gap,
        "projection_residual_to_measurement_px": residual,
        "current_source_intrinsics_fx_fy_cx_cy": hand_intr.astype(float).tolist(),
        "unidepth_scaled_intrinsics_fx_fy_cx_cy": candidate_intrinsics.astype(float).tolist(),
        "intrinsics_focal_ratio_fx": float(candidate_intrinsics[0] / hand_intr[0]),
        "counterfactual_translation_m": translation.astype(float).tolist(),
        "counterfactual_hand_depth_m": summarize(source_vertices[:, 2].astype(float).tolist()),
        "sample_partitions": partitions,
        **FALSE_READY,
    }


def rows_by_state(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def partition_case_summary(rows: list[dict[str, Any]], partition: str) -> dict[str, Any]:
    gaps: list[float] = []
    abs_gaps: list[float] = []
    states: Counter[str] = Counter()
    measured = 0
    compatible = 0
    signal_compatible = 0
    for row in rows:
        parts = row.get("sample_partitions")
        if not isinstance(parts, dict) or partition not in parts:
            states["unobserved_intrinsics_counterfactual"] += 1
            continue
        part = require_dict(parts[partition], f"sample_partitions.{partition}")
        states[require_str(part.get("state"), "partition state")] += 1
        if part.get("measured") is True:
            measured += 1
            gap_summary = require_dict(part.get("hand_minus_unidepth_depth_m"), "hand_minus_unidepth_depth_m")
            abs_summary = require_dict(part.get("abs_hand_minus_unidepth_depth_m"), "abs_hand_minus_unidepth_depth_m")
            gaps.append(finite_float(gap_summary.get("median"), "gap median"))
            abs_gaps.append(finite_float(abs_summary.get("median"), "abs gap median"))
        if part.get("metric_depth_signal_compatible") is True:
            signal_compatible += 1
        if part.get("metric_depth_compatible") is True:
            compatible += 1
    return {
        "partition": partition,
        "hand_rows": len(rows),
        "measured_hand_rows": measured,
        "unobserved_hand_rows": len(rows) - measured,
        "metric_depth_signal_compatible_hand_rows": signal_compatible,
        "metric_depth_compatible_hand_rows": compatible,
        "state_counts": dict(sorted(states.items())),
        "hand_minus_unidepth_depth_m": summarize(gaps),
        "abs_hand_minus_unidepth_depth_m": summarize(abs_gaps),
    }


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals: list[float] = []
    for row in rows:
        value = optional_finite(row.get(key))
        if value is not None:
            vals.append(value)
    return summarize(vals)


def nested_median_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals: list[float] = []
    for row in rows:
        summary = row.get(key)
        if not isinstance(summary, dict):
            continue
        value = optional_finite(summary.get("median"))
        if value is not None:
            vals.append(value)
    return summarize(vals)


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
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    frame_count = len(frames)
    if frame_count != require_int(visible.get("frame_count"), f"{case} visible frame_count"):
        raise RuntimeError(f"{case} graph annotations disagree with visible-surface report")
    if frame_count != require_int(hand_metric.get("frame_count"), f"{case} hand metric frame_count"):
        raise RuntimeError(f"{case} graph annotations disagree with hand metric-depth report")
    depth = depth_archive(existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive"))
    rows: list[dict[str, Any]] = []
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    for raw in require_list(hand_metric.get("rows"), f"{case} hand metric rows"):
        metric_row = require_dict(raw, "hand metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
        side = require_str(metric_row.get("hand_side"), "metric row hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "metric row hand_index")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} missing annotation frame {frame_idx}")
        rows.append(
            measure_counterfactual_row(
                case=case,
                frame=frame,
                metric_row=metric_row,
                hand=hand_index.get((frame_idx, side, hand_i)),
                depth=depth,
                mask_cache=mask_cache,
                args=args,
            )
        )
    measured_rows = [row for row in rows if row_measured(row)]
    projection_ready_rows = [row for row in rows if row_residual_ok(row)]
    compatible_rows = [row for row in rows if row.get("metric_depth_compatible") is True]
    improved_rows = [row for row in rows if row.get("median_gap_improved_vs_current") is True]
    repair_candidate_rows = [row for row in rows if row_repair_candidate(row)]
    partitions = ["all_projected_hand_pixels", "near_active_object_masks", "far_from_active_object_masks"]
    report = {
        "method": "build_v17_hand_intrinsics_depth_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": visible.get("metric_depth_npz"),
        "frame_count": frame_count,
        "hand_intrinsics_counterfactual_variable_count": len(rows),
        "counterfactual_metric_depth_measured_rows": len(measured_rows),
        "counterfactual_projection_factor_ready_rows": len(projection_ready_rows),
        "counterfactual_depth_repair_factor_candidate_rows": len(repair_candidate_rows),
        "counterfactual_median_gap_improved_rows": len(improved_rows),
        "counterfactual_metric_hand_state_accepted_rows": len(compatible_rows),
        "counterfactual_state_counts": rows_by_state(rows, "counterfactual_state"),
        "counterfactual_owner_depth_state_counts": rows_by_state(rows, "counterfactual_owner_depth_state"),
        "partition_summaries": {partition: partition_case_summary(rows, partition) for partition in partitions},
        "intrinsics_focal_ratio_fx": numeric_summary(rows, "intrinsics_focal_ratio_fx"),
        "counterfactual_owner_median_gap_m": numeric_summary(rows, "counterfactual_owner_median_gap_m"),
        "counterfactual_hand_depth_m": nested_median_summary(rows, "counterfactual_hand_depth_m"),
        "problem_semantics": {
            "tested_mechanism": "replace hand source intrinsics with UniDepth intrinsics scaled to the hand source image size, then re-solve source-camera hand translation from local MANO-family geometry and 2D keypoints",
            "acceptance_condition": "same median and p95 absolute hand-minus-UniDepth depth thresholds as v17_hand_metric_depth_state, plus projection residual thresholds",
            "interpretation": "If this counterfactual improves most median gaps while depth-compatible rows stay near zero, intrinsics mismatch explains part of the hand-depth error and leaves additional hand/depth variables unresolved.",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_intrinsics_depth_counterfactual.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_metric_depth_state_root / "v17_hand_metric_depth_state_summary.json",
        "hand metric-depth state summary",
    )
    summary = require_dict(load_json(summary_path), "hand metric-depth state summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    counterfactual_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    for report in reports:
        counterfactual_counts.update(require_dict(report.get("counterfactual_state_counts"), "counterfactual state counts"))
        owner_counts.update(require_dict(report.get("counterfactual_owner_depth_state_counts"), "owner state counts"))
    payload = {
        "method": "build_v17_hand_intrinsics_depth_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_metric_depth_state_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_intrinsics_depth_counterfactual.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_intrinsics_counterfactual_variable_count": require_int(
                    report.get("hand_intrinsics_counterfactual_variable_count"),
                    "counterfactual variable count",
                ),
                "counterfactual_metric_depth_measured_rows": require_int(
                    report.get("counterfactual_metric_depth_measured_rows"),
                    "counterfactual measured rows",
                ),
                "counterfactual_projection_factor_ready_rows": require_int(
                    report.get("counterfactual_projection_factor_ready_rows"),
                    "projection-ready rows",
                ),
                "counterfactual_depth_repair_factor_candidate_rows": require_int(
                    report.get("counterfactual_depth_repair_factor_candidate_rows"),
                    "counterfactual repair candidate rows",
                ),
                "counterfactual_median_gap_improved_rows": require_int(
                    report.get("counterfactual_median_gap_improved_rows"),
                    "improved rows",
                ),
                "counterfactual_metric_hand_state_accepted_rows": require_int(
                    report.get("counterfactual_metric_hand_state_accepted_rows"),
                    "compatible rows",
                ),
                "counterfactual_state_counts": require_dict(
                    report.get("counterfactual_state_counts"),
                    "counterfactual state counts",
                ),
                "counterfactual_owner_depth_state_counts": require_dict(
                    report.get("counterfactual_owner_depth_state_counts"),
                    "owner depth state counts",
                ),
                "intrinsics_focal_ratio_fx": require_dict(
                    report.get("intrinsics_focal_ratio_fx"),
                    "intrinsics focal ratio",
                ),
                "counterfactual_owner_median_gap_m": require_dict(
                    report.get("counterfactual_owner_median_gap_m"),
                    "counterfactual owner gap",
                ),
                "counterfactual_hand_depth_m": require_dict(
                    report.get("counterfactual_hand_depth_m"),
                    "counterfactual hand depth",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_intrinsics_counterfactual_variable_count": sum(
            require_int(report.get("hand_intrinsics_counterfactual_variable_count"), "counterfactual variable count")
            for report in reports
        ),
        "counterfactual_metric_depth_measured_rows": sum(
            require_int(report.get("counterfactual_metric_depth_measured_rows"), "counterfactual measured rows")
            for report in reports
        ),
        "counterfactual_projection_factor_ready_rows": sum(
            require_int(report.get("counterfactual_projection_factor_ready_rows"), "projection-ready rows")
            for report in reports
        ),
        "counterfactual_depth_repair_factor_candidate_rows": sum(
            require_int(report.get("counterfactual_depth_repair_factor_candidate_rows"), "repair candidate rows")
            for report in reports
        ),
        "counterfactual_median_gap_improved_rows": sum(
            require_int(report.get("counterfactual_median_gap_improved_rows"), "improved rows")
            for report in reports
        ),
        "counterfactual_metric_hand_state_accepted_rows": sum(
            require_int(report.get("counterfactual_metric_hand_state_accepted_rows"), "compatible rows")
            for report in reports
        ),
        "counterfactual_state_counts": dict(sorted(counterfactual_counts.items())),
        "counterfactual_owner_depth_state_counts": dict(sorted(owner_counts.items())),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_intrinsics_depth_counterfactual_summary.json", payload)
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
        "--hand-metric-depth-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_metric_depth_state"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual"),
    )
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
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
