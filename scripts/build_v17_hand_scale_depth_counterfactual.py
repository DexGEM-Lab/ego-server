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


STATUS = "v17_hand_scale_depth_counterfactual_qc"
CLAIM = (
    "This artifact tests the hand-scale/depth degeneracy left after the intrinsics counterfactual. "
    "It keeps 2D hand evidence fixed, uses UniDepth-aligned intrinsics, then evaluates global and per-row "
    "positive hand-geometry scale factors against the same UniDepth depth acceptance thresholds."
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


def mode_state(prefix: str, measured: bool, residual_ok: bool, compatible: bool) -> str:
    if compatible:
        return f"metric_depth_compatible_under_{prefix}"
    if measured and residual_ok:
        return f"depth_repair_candidate_under_{prefix}"
    if measured:
        return f"metric_depth_measured_projection_untrusted_under_{prefix}"
    return f"unobserved_under_{prefix}"


def partition_owner(partitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    far = partitions["far_from_active_object_masks"]
    if far.get("measured") is True:
        return far
    return partitions["all_projected_hand_pixels"]


def selected_valid_mask(
    *,
    hand_z: np.ndarray,
    metric_z: np.ndarray,
    selected: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    return (
        selected
        & np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )


def row_scale_candidate(
    *,
    hand_z: np.ndarray,
    metric_z: np.ndarray,
    selected: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid = selected_valid_mask(hand_z=hand_z, metric_z=metric_z, selected=selected, args=args)
    if int(np.count_nonzero(valid)) < int(args.min_depth_pixels):
        return {
            "available": False,
            "scale": None,
            "valid_depth_pixels": int(np.count_nonzero(valid)),
        }
    ratio = metric_z[valid] / hand_z[valid]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if len(ratio) < int(args.min_depth_pixels):
        return {
            "available": False,
            "scale": None,
            "valid_depth_pixels": int(len(ratio)),
        }
    scale = float(np.median(ratio))
    return {
        "available": True,
        "scale": scale,
        "valid_depth_pixels": int(len(ratio)),
        "sample_ratio_summary": summarize(ratio.astype(float).tolist()),
    }


def evaluate_scaled(
    *,
    prefix: str,
    scale: float | None,
    base: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if scale is None or not math.isfinite(float(scale)) or scale <= 0.0:
        return {
            "scale_mode": prefix,
            "scale": None,
            "state": f"unobserved_under_{prefix}",
            "owner_depth_state": "unobserved_hand_metric_depth",
            "owner_sample_partition": None,
            "metric_depth_compatible": False,
            "depth_repair_factor_candidate": False,
            "owner_median_gap_m": None,
        }
    hand_z = np.asarray(base["hand_z"], dtype=np.float64) * float(scale)
    metric_z = np.asarray(base["metric_z"], dtype=np.float64)
    object_distance_px = np.asarray(base["object_distance_px"], dtype=np.float64)
    near = np.asarray(base["near"], dtype=bool)
    far = np.asarray(base["far"], dtype=bool)
    residual_ok = bool(base["residual_ok"])
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
    owner = partition_owner(partitions)
    measured = bool(owner.get("measured") is True)
    compatible = bool(owner.get("metric_depth_compatible") is True)
    gap_summary = owner.get("hand_minus_unidepth_depth_m")
    owner_gap = None
    if isinstance(gap_summary, dict):
        owner_gap = optional_finite(gap_summary.get("median"))
    return {
        "scale_mode": prefix,
        "scale": float(scale),
        "state": mode_state(prefix, measured, residual_ok, compatible),
        "owner_depth_state": require_str(owner.get("state"), f"{prefix} owner state"),
        "owner_sample_partition": require_str(owner.get("sample_partition"), f"{prefix} owner partition"),
        "metric_depth_compatible": compatible,
        "depth_repair_factor_candidate": bool(measured and residual_ok and not compatible),
        "owner_median_gap_m": owner_gap,
        "scaled_wrist_to_middle_tip_m": float(scale) * float(base["wrist_to_middle_tip_m"]),
        "partitions": partitions,
    }


def row_mode_counts(rows: list[dict[str, Any]], mode_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        mode = require_dict(row.get(mode_key), mode_key)
        counts[require_str(mode.get("state"), f"{mode_key}.state")] += 1
    return dict(sorted(counts.items()))


def row_mode_count(rows: list[dict[str, Any]], mode_key: str, field: str) -> int:
    total = 0
    for row in rows:
        mode = require_dict(row.get(mode_key), mode_key)
        if mode.get(field) is True:
            total += 1
    return total


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        value = optional_finite(value)
        if value is not None:
            values.append(value)
    return summarize(values)


def mode_numeric_summary(rows: list[dict[str, Any]], mode_key: str, field: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        mode = row.get(mode_key)
        if not isinstance(mode, dict):
            continue
        value = optional_finite(mode.get(field))
        if value is not None:
            values.append(value)
    return summarize(values)


def side_scale_candidates(base_rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    by_side: dict[str, list[float]] = {}
    for base in base_rows:
        if base.get("residual_ok") is not True:
            continue
        candidate = base.get("row_scale_candidate")
        if not isinstance(candidate, dict) or candidate.get("available") is not True:
            continue
        scale = optional_finite(candidate.get("scale"))
        if scale is None:
            continue
        by_side.setdefault(require_str(base.get("hand_side"), "hand_side"), []).append(scale)
    for side, values in sorted(by_side.items()):
        out[side] = float(np.median(np.asarray(values, dtype=np.float64)))
    return out


def case_scale_candidate(base_rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for base in base_rows:
        if base.get("residual_ok") is not True:
            continue
        candidate = base.get("row_scale_candidate")
        if not isinstance(candidate, dict) or candidate.get("available") is not True:
            continue
        scale = optional_finite(candidate.get("scale"))
        if scale is not None:
            values.append(scale)
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def measure_base_row(
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
            "hand_scale_counterfactual_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "base_available": False,
            "missing_counterfactual_inputs": missing,
            "row_scale_candidate": {"available": False, "scale": None, "valid_depth_pixels": 0},
            **FALSE_READY,
        }
    if hand is None or hand_intr is None or geometry is None or depth_i is None:
        raise RuntimeError("missing scale counterfactual inputs branch failed to return")
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
        return {
            "case": case,
            "hand_scale_counterfactual_variable_id": row_id,
            "frame_idx": frame_idx,
            "hand_side": side,
            "hand_index": hand_index,
            "base_available": False,
            "missing_counterfactual_inputs": ["projected_hand_surface_inside_depth_image"],
            "projection_residual_to_measurement_px": residual,
            "row_scale_candidate": {"available": False, "scale": None, "valid_depth_pixels": 0},
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
    base_partitions = {
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
    owner = partition_owner(base_partitions)
    owner_label = require_str(owner.get("sample_partition"), "base owner partition")
    selected = far if owner_label == "far_from_active_object_masks" else np.ones(len(hand_z), dtype=bool)
    scale_candidate = row_scale_candidate(hand_z=hand_z, metric_z=metric_z, selected=selected, args=args)
    wrist_to_middle = float(np.linalg.norm(local_joints[12] - local_joints[0]))
    return {
        "case": case,
        "hand_scale_counterfactual_variable_id": row_id,
        "frame_idx": frame_idx,
        "hand_side": side,
        "hand_index": hand_index,
        "base_available": True,
        "projection_residual_to_measurement_px": residual,
        "residual_ok": residual_ok,
        "hand_z": hand_z,
        "metric_z": metric_z,
        "object_distance_px": object_distance_px,
        "near": near,
        "far": far,
        "base_owner_sample_partition": owner_label,
        "base_owner_depth_state": require_str(owner.get("state"), "base owner state"),
        "base_metric_depth_compatible": bool(owner.get("metric_depth_compatible") is True),
        "base_owner_median_gap_m": optional_finite(
            require_dict(owner.get("hand_minus_unidepth_depth_m"), "base gap").get("median")
        )
        if owner.get("measured") is True
        else None,
        "current_intrinsics_focal_ratio_fx": float(candidate_intrinsics[0] / hand_intr[0]),
        "wrist_to_middle_tip_m": wrist_to_middle,
        "row_scale_candidate": scale_candidate,
        **FALSE_READY,
    }


def json_row(base: dict[str, Any], case_scale: float | None, side_scale: float | None, args: argparse.Namespace) -> dict[str, Any]:
    row_scale_raw = base.get("row_scale_candidate")
    row_scale = None
    if isinstance(row_scale_raw, dict) and row_scale_raw.get("available") is True:
        row_scale = optional_finite(row_scale_raw.get("scale"))
    if base.get("base_available") is not True:
        empty = evaluate_scaled(prefix="case_global_scale", scale=None, base={}, args=args)
        return {
            "case": base["case"],
            "hand_scale_counterfactual_variable_id": base["hand_scale_counterfactual_variable_id"],
            "frame_idx": base["frame_idx"],
            "hand_side": base["hand_side"],
            "hand_index": base["hand_index"],
            "base_available": False,
            "missing_counterfactual_inputs": base.get("missing_counterfactual_inputs", []),
            "row_scale_candidate": base.get("row_scale_candidate"),
            "case_global_scale": empty,
            "side_global_scale": empty | {"scale_mode": "side_global_scale", "state": "unobserved_under_side_global_scale"},
            "per_row_scale_oracle": empty | {"scale_mode": "per_row_scale_oracle", "state": "unobserved_under_per_row_scale_oracle"},
            **FALSE_READY,
        }
    case_eval = evaluate_scaled(prefix="case_global_scale", scale=case_scale, base=base, args=args)
    side_eval = evaluate_scaled(prefix="side_global_scale", scale=side_scale, base=base, args=args)
    row_eval = evaluate_scaled(prefix="per_row_scale_oracle", scale=row_scale, base=base, args=args)
    return {
        "case": base["case"],
        "hand_scale_counterfactual_variable_id": base["hand_scale_counterfactual_variable_id"],
        "frame_idx": base["frame_idx"],
        "hand_side": base["hand_side"],
        "hand_index": base["hand_index"],
        "base_available": True,
        "projection_residual_to_measurement_px": base["projection_residual_to_measurement_px"],
        "base_owner_sample_partition": base["base_owner_sample_partition"],
        "base_owner_depth_state": base["base_owner_depth_state"],
        "base_metric_depth_compatible": base["base_metric_depth_compatible"],
        "base_owner_median_gap_m": base["base_owner_median_gap_m"],
        "current_intrinsics_focal_ratio_fx": base["current_intrinsics_focal_ratio_fx"],
        "wrist_to_middle_tip_m": base["wrist_to_middle_tip_m"],
        "row_scale_candidate": base["row_scale_candidate"],
        "case_global_scale": case_eval,
        "side_global_scale": side_eval,
        "per_row_scale_oracle": row_eval,
        **FALSE_READY,
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
        "hand_metric_depth_state": existing_path(
            args.hand_metric_depth_state_root / case / "v17_hand_metric_depth_state.json",
            f"{case} hand metric-depth state report",
        ),
        "hand_intrinsics_depth_counterfactual": existing_path(
            args.hand_intrinsics_depth_counterfactual_root / case / "v17_hand_intrinsics_depth_counterfactual.json",
            f"{case} hand intrinsics-depth counterfactual report",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hand_index = annotation_hand_index(frames)
    visible = payloads["visible_surface"]
    hand_metric = payloads["hand_metric_depth_state"]
    intrinsics_cf = payloads["hand_intrinsics_depth_counterfactual"]
    frame_count = len(frames)
    if frame_count != require_int(visible.get("frame_count"), f"{case} visible frame_count"):
        raise RuntimeError(f"{case} graph annotations disagree with visible-surface report")
    if frame_count != require_int(hand_metric.get("frame_count"), f"{case} hand metric frame_count"):
        raise RuntimeError(f"{case} graph annotations disagree with hand metric-depth report")
    if frame_count != require_int(intrinsics_cf.get("frame_count"), f"{case} intrinsics counterfactual frame_count"):
        raise RuntimeError(f"{case} graph annotations disagree with hand intrinsics-depth counterfactual")
    if require_int(
        intrinsics_cf.get("hand_intrinsics_counterfactual_variable_count"),
        f"{case} intrinsics counterfactual variable count",
    ) != require_int(hand_metric.get("hand_metric_depth_variable_count"), f"{case} hand metric variable count"):
        raise RuntimeError(f"{case} intrinsics counterfactual variable count disagrees with hand metric-depth report")
    depth = depth_archive(existing_path(Path(require_str(visible.get("metric_depth_npz"), "metric_depth_npz")), "metric depth archive"))
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    base_rows: list[dict[str, Any]] = []
    for raw in require_list(hand_metric.get("rows"), f"{case} hand metric rows"):
        metric_row = require_dict(raw, "hand metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "metric row frame_idx")
        side = require_str(metric_row.get("hand_side"), "metric row hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "metric row hand_index")
        frame = frames.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"{case} missing annotation frame {frame_idx}")
        base_rows.append(
            measure_base_row(
                case=case,
                frame=frame,
                metric_row=metric_row,
                hand=hand_index.get((frame_idx, side, hand_i)),
                depth=depth,
                mask_cache=mask_cache,
                args=args,
            )
        )
    if len(base_rows) != require_int(hand_metric.get("hand_metric_depth_variable_count"), f"{case} hand variable count"):
        raise RuntimeError(f"{case} scale counterfactual rows disagree with hand metric-depth rows")
    case_scale = case_scale_candidate(base_rows)
    side_scales = side_scale_candidates(base_rows)
    rows = [
        json_row(
            base,
            case_scale=case_scale,
            side_scale=side_scales.get(require_str(base.get("hand_side"), "hand_side")),
            args=args,
        )
        for base in base_rows
    ]
    mode_keys = ["case_global_scale", "side_global_scale", "per_row_scale_oracle"]
    scale_candidate_rows = [
        row for row in rows if require_dict(row.get("row_scale_candidate"), "row_scale_candidate").get("available") is True
    ]
    report = {
        "method": "build_v17_hand_scale_depth_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "frame_count": frame_count,
        "hand_scale_counterfactual_variable_count": len(rows),
        "base_available_rows": sum(1 for row in rows if row.get("base_available") is True),
        "scale_candidate_rows": len(scale_candidate_rows),
        "case_global_scale": case_scale,
        "side_global_scales": side_scales,
        "row_scale_candidate_summary": numeric_summary(scale_candidate_rows, "row_scale_candidate.scale"),
        "current_wrist_to_middle_tip_m": numeric_summary(rows, "wrist_to_middle_tip_m"),
        "case_global_scaled_wrist_to_middle_tip_m": mode_numeric_summary(
            rows, "case_global_scale", "scaled_wrist_to_middle_tip_m"
        ),
        "side_global_scaled_wrist_to_middle_tip_m": mode_numeric_summary(
            rows, "side_global_scale", "scaled_wrist_to_middle_tip_m"
        ),
        "per_row_scaled_wrist_to_middle_tip_m": mode_numeric_summary(
            rows, "per_row_scale_oracle", "scaled_wrist_to_middle_tip_m"
        ),
        "mode_summaries": {
            mode: {
                "state_counts": row_mode_counts(rows, mode),
                "metric_hand_state_accepted_rows": row_mode_count(rows, mode, "metric_depth_compatible"),
                "depth_repair_factor_candidate_rows": row_mode_count(rows, mode, "depth_repair_factor_candidate"),
                "owner_median_gap_m": mode_numeric_summary(rows, mode, "owner_median_gap_m"),
            }
            for mode in mode_keys
        },
        "source_intrinsics_counterfactual_comparison": {
            "intrinsics_counterfactual_metric_hand_state_accepted_rows": require_int(
                intrinsics_cf.get("counterfactual_metric_hand_state_accepted_rows"),
                f"{case} intrinsics accepted rows",
            ),
            "intrinsics_counterfactual_depth_repair_factor_candidate_rows": require_int(
                intrinsics_cf.get("counterfactual_depth_repair_factor_candidate_rows"),
                f"{case} intrinsics repair rows",
            ),
            "intrinsics_counterfactual_median_gap_improved_rows": require_int(
                intrinsics_cf.get("counterfactual_median_gap_improved_rows"),
                f"{case} intrinsics improved rows",
            ),
            "intrinsics_counterfactual_owner_median_gap_m": require_dict(
                intrinsics_cf.get("counterfactual_owner_median_gap_m"),
                f"{case} intrinsics owner gap",
            ),
        },
        "problem_semantics": {
            "scale_degeneracy": "With fixed intrinsics and fixed 2D keypoints, positive local-hand scaling and source-camera translation scaling preserve projected 2D keypoints while changing metric depth.",
            "case_global_scale": "one robust scale for all hand rows in the case, estimated from residual-ok rows with depth samples",
            "side_global_scale": "one robust scale per hand side in the case, estimated from residual-ok rows with depth samples",
            "per_row_scale_oracle": "row-specific depth scale lower bound; it is evidence for the degeneracy and not a physically valid time-varying hand-size state",
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_hand_scale_depth_counterfactual.json", report)
    return report


def nested_case_summary(report: dict[str, Any], mode: str, field: str) -> Any:
    modes = require_dict(report.get("mode_summaries"), "mode_summaries")
    mode_summary = require_dict(modes.get(mode), mode)
    return mode_summary.get(field)


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.hand_intrinsics_depth_counterfactual_root / "v17_hand_intrinsics_depth_counterfactual_summary.json",
        "hand intrinsics-depth counterfactual summary",
    )
    summary = require_dict(load_json(summary_path), "hand intrinsics-depth counterfactual summary")
    reports = [
        case_problem(
            require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"),
            args,
        )
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    mode_keys = ["case_global_scale", "side_global_scale", "per_row_scale_oracle"]
    payload = {
        "method": "build_v17_hand_scale_depth_counterfactual",
        "status": STATUS,
        "claim": CLAIM,
        "source_hand_intrinsics_depth_counterfactual_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_hand_scale_depth_counterfactual.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "hand_scale_counterfactual_variable_count": require_int(
                    report.get("hand_scale_counterfactual_variable_count"),
                    "scale variable count",
                ),
                "base_available_rows": require_int(report.get("base_available_rows"), "base rows"),
                "scale_candidate_rows": require_int(report.get("scale_candidate_rows"), "scale candidate rows"),
                "case_global_scale": report.get("case_global_scale"),
                "side_global_scales": require_dict(report.get("side_global_scales"), "side scales"),
                "row_scale_candidate_summary": require_dict(
                    report.get("row_scale_candidate_summary"),
                    "row scale summary",
                ),
                "current_wrist_to_middle_tip_m": require_dict(
                    report.get("current_wrist_to_middle_tip_m"),
                    "current wrist-middle summary",
                ),
                "case_global_scaled_wrist_to_middle_tip_m": require_dict(
                    report.get("case_global_scaled_wrist_to_middle_tip_m"),
                    "case scaled wrist-middle summary",
                ),
                "side_global_scaled_wrist_to_middle_tip_m": require_dict(
                    report.get("side_global_scaled_wrist_to_middle_tip_m"),
                    "side scaled wrist-middle summary",
                ),
                "per_row_scaled_wrist_to_middle_tip_m": require_dict(
                    report.get("per_row_scaled_wrist_to_middle_tip_m"),
                    "per-row scaled wrist-middle summary",
                ),
                "mode_summaries": require_dict(report.get("mode_summaries"), "mode summaries"),
                "source_intrinsics_counterfactual_comparison": require_dict(
                    report.get("source_intrinsics_counterfactual_comparison"),
                    "intrinsics comparison",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "hand_scale_counterfactual_variable_count": sum(
            require_int(report.get("hand_scale_counterfactual_variable_count"), "scale variable count")
            for report in reports
        ),
        "base_available_rows": sum(require_int(report.get("base_available_rows"), "base rows") for report in reports),
        "scale_candidate_rows": sum(
            require_int(report.get("scale_candidate_rows"), "scale candidate rows")
            for report in reports
        ),
        "mode_totals": {
            mode: {
                "metric_hand_state_accepted_rows": sum(
                    require_int(
                        nested_case_summary(report, mode, "metric_hand_state_accepted_rows"),
                        f"{mode} accepted rows",
                    )
                    for report in reports
                ),
                "depth_repair_factor_candidate_rows": sum(
                    require_int(
                        nested_case_summary(report, mode, "depth_repair_factor_candidate_rows"),
                        f"{mode} repair rows",
                    )
                    for report in reports
                ),
            }
            for mode in mode_keys
        },
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_hand_scale_depth_counterfactual_summary.json", payload)
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
        "--hand-intrinsics-depth-counterfactual-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual"),
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
