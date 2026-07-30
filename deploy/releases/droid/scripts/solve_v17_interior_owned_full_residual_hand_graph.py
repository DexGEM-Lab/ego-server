#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from apply_v17_hand_far_field_temporal_refit import finite_float
from build_v17_depth_edge_ownership_counterfactual import depth_edge_band, interior_state
from build_v17_hand_intrinsics_depth_counterfactual import annotation_hand_index
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
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
from refit_mano_pose_contact_v3 import robust_l1
from solve_v17_hand_depth_repair_graph import build_base_row, evaluate_row
from solve_v17_mano_articulation_local import (
    corrected_replayed_state,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
)
from solve_v17_relinearized_hand_surface_observation_graph import replay_vertices


STATUS = "v17_interior_owned_full_residual_hand_graph_qc"
CLAIM = (
    "This artifact tests whether hand-independent depth-observation ownership closes the persistent "
    "full-residual hand-depth class. It keeps the pose-enabled full-residual MANO pose deltas fixed, "
    "re-solves per-row camera-ray depth shifts against interior-owned UniDepth samples that exclude "
    "depth-discontinuity bands, and measures the result through full MANO reprojection and UniDepth "
    "resampling under both the legacy all-pixel predicate and the interior-owned predicate. It is a "
    "hand-depth ownership solver iteration, not V3 solver closure: object geometry, object pose, "
    "contact ownership, and camera trajectory remain outside this graph."
)


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def finite_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(finite_float(value, key))
    return summarize(values)


def source_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "method": payload.get("method"),
        "status": payload.get("status"),
    }


def thin(values: np.ndarray, max_count: int) -> np.ndarray:
    if values.size <= max_count:
        return values
    pick = np.linspace(0, values.size - 1, max_count).round().astype(np.int64)
    return values[pick]


def pose_delta_array(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("relinearized_pose_delta_rotvec")
    if raw is None:
        return np.zeros((15, 3), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (15, 3) or not np.all(np.isfinite(arr)):
        raise RuntimeError("relinearized_pose_delta_rotvec must be a finite 15x3 array")
    return arr


def interior_measurement(
    evaluated: dict[str, Any],
    edge: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Interior-owned depth measurement on all valid projected-hand pixels minus the edge band."""
    if evaluated.get("partitions") is None:
        return {
            "interior_measured": False,
            "interior_valid_pixels": 0,
            "edge_owner_pixels": 0,
            "interior_median_gap_m": None,
            "interior_p95_abs_gap_m": None,
            "interior_gap_values_m": np.asarray([], dtype=np.float64),
            "interior_state": "interior_unobserved",
        }
    x = np.asarray(evaluated.get("x"), dtype=np.int32)
    y = np.asarray(evaluated.get("y"), dtype=np.int32)
    hand_z = np.asarray(evaluated.get("hand_z"), dtype=np.float64)
    metric_z = np.asarray(evaluated.get("metric_z"), dtype=np.float64)
    depth_shape_raw = evaluated.get("depth_shape")
    if not isinstance(depth_shape_raw, list) or len(depth_shape_raw) != 2:
        raise RuntimeError("evaluated row depth_shape must be a two-item list")
    if edge.shape != (int(depth_shape_raw[0]), int(depth_shape_raw[1])):
        raise RuntimeError("edge band shape disagrees with evaluated depth shape")
    valid = (
        np.isfinite(hand_z)
        & (hand_z > 1e-6)
        & np.isfinite(metric_z)
        & (metric_z >= float(args.min_depth_m))
        & (metric_z <= float(args.max_depth_m))
    )
    on_edge = edge[y, x]
    interior = valid & ~on_edge
    gap = hand_z - metric_z
    interior_count = int(np.count_nonzero(interior))
    edge_count = int(np.count_nonzero(valid & on_edge))
    median_gap: float | None = None
    p95_abs_gap: float | None = None
    if interior_count > 0:
        interior_gap = gap[interior]
        median_gap = float(np.median(interior_gap))
        p95_abs_gap = float(np.percentile(np.abs(interior_gap), 95.0))
    projection = require_dict(evaluated.get("projection_residual_to_measurement_px"), "projection residual")
    residual_ok = bool(projection.get("residual_ok") is True)
    return {
        "interior_measured": bool(interior_count >= int(args.min_depth_pixels)),
        "interior_valid_pixels": interior_count,
        "edge_owner_pixels": edge_count,
        "interior_median_gap_m": median_gap,
        "interior_p95_abs_gap_m": p95_abs_gap,
        "interior_gap_values_m": gap[interior].astype(np.float64),
        "interior_state": interior_state(
            interior_count=interior_count,
            median_gap=median_gap,
            p95_abs_gap=p95_abs_gap,
            residual_ok=residual_ok,
            args=args,
        ),
    }


def case_problem(case: str, model: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
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
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
        "pose_full_residual_graph": existing_path(
            args.pose_full_residual_graph_root
            / case
            / "v17_full_residual_relinearized_hand_surface_observation_graph.json",
            f"{case} pose full-residual graph",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    for name in ["visible_surface", "hand_metric_depth_state", "hand_depth_repair_graph", "pose_full_residual_graph"]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count disagrees with {name}")
    hands = annotation_hand_index(frames)
    depth_path = existing_path(
        Path(require_str(payloads["visible_surface"].get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth archive",
    )
    depth = depth_archive(depth_path)
    repair = payloads["hand_depth_repair_graph"]
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    pose_graph = payloads["pose_full_residual_graph"]
    pose_rows = [require_dict(raw, "pose row") for raw in require_list(pose_graph.get("rows"), f"{case} pose rows")]
    pose_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "pose graph id"): row for row in pose_rows
    }
    if len(pose_by_id) != len(pose_rows):
        raise RuntimeError(f"{case} duplicate pose graph row ids")
    variable_inputs = sorted(
        [
            row
            for row in pose_rows
            if row.get("relinearized_delta_applied") is True
            and row.get("relinearized_total_hand_ray_shift_m") is not None
        ],
        key=lambda row: (
            require_str(row.get("hand_side"), "hand_side"),
            require_int(row.get("hand_index"), "hand_index"),
            require_int(row.get("frame_idx"), "frame_idx"),
        ),
    )
    expected_variables = require_int(pose_graph.get("relinearized_variable_rows"), f"{case} pose variables")
    if len(variable_inputs) != expected_variables:
        raise RuntimeError(
            f"{case} variable inputs {len(variable_inputs)} disagree with pose graph variables {expected_variables}"
        )
    var_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "variable id"): i
        for i, row in enumerate(variable_inputs)
    }
    baseline_np = np.asarray(
        [
            finite_float(row.get("relinearized_total_hand_ray_shift_m"), "baseline shift")
            for row in variable_inputs
        ],
        dtype=np.float64,
    )
    lower_np = np.maximum(
        -float(args.max_abs_hand_ray_shift_m) - baseline_np,
        -float(args.max_abs_interior_delta_m),
    ).astype(np.float32)
    upper_np = np.minimum(
        float(args.max_abs_hand_ray_shift_m) - baseline_np,
        float(args.max_abs_interior_delta_m),
    ).astype(np.float32)
    if np.any(lower_np > upper_np):
        raise RuntimeError(f"{case} interior scalar bounds are inconsistent")
    lower = torch.tensor(lower_np, dtype=torch.float32, device=device)
    upper = torch.tensor(upper_np, dtype=torch.float32, device=device)
    delta = torch.zeros(len(variable_inputs), dtype=torch.float32, device=device, requires_grad=True)
    pose_delta_np = np.stack([pose_delta_array(row) for row in variable_inputs]).astype(np.float32)
    pose_delta = torch.tensor(pose_delta_np, dtype=torch.float32, device=device)
    zero_ray = torch.zeros((), dtype=torch.float32, device=device)

    metric_rows = [
        require_dict(raw, "metric row")
        for raw in require_list(payloads["hand_metric_depth_state"].get("rows"), f"{case} metric rows")
    ]
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    base_by_id: dict[str, dict[str, Any]] = {}
    state_by_id: dict[str, dict[str, Any]] = {}
    for metric_row in metric_rows:
        graph_id = require_str(metric_row.get("hand_metric_depth_variable_id"), "metric id").replace(
            "hand_metric_depth:",
            "hand_depth_repair_graph:",
            1,
        )
        if graph_id not in var_by_id:
            continue
        frame_idx = require_int(metric_row.get("frame_idx"), "frame_idx")
        side = require_str(metric_row.get("hand_side"), "hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "hand_index")
        frame = require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}")
        hand = require_dict(hands.get((frame_idx, side, hand_i)), f"{case} annotation hand {graph_id}")
        base_by_id[graph_id] = build_base_row(
            case=case,
            frame=frame,
            metric_row=metric_row,
            hand=hand,
            depth=depth,
            mask_cache=mask_cache,
            args=args,
        )
        repair_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
        shifted_repair = {
            **repair_row,
            "hand_ray_shift_m": float(baseline_np[var_by_id[graph_id]]),
        }
        state_by_id[graph_id] = corrected_replayed_state(
            model=model,
            hand=hand,
            graph_row=shifted_repair,
            depth=depth,
            device=device,
        )
    missing_states = [graph_id for graph_id in var_by_id if graph_id not in state_by_id]
    if missing_states:
        raise RuntimeError(f"{case} missing replay state for variables: {missing_states[:3]}")

    edge_cache: dict[int, np.ndarray] = {}

    def edge_for_frame(frame_idx: int) -> np.ndarray:
        cached = edge_cache.get(frame_idx)
        if cached is not None:
            return cached
        depth_i = depth["frame_to_i"].get(frame_idx)
        if depth_i is None:
            raise RuntimeError(f"{case} frame {frame_idx} missing from metric depth archive")
        edge = depth_edge_band(
            np.asarray(depth["depth"][depth_i], dtype=np.float64),
            edge_window_px=int(args.edge_window_px),
            edge_depth_range_m=float(args.edge_depth_range_m),
            edge_band_dilation_px=int(args.edge_band_dilation_px),
        )
        edge_cache[frame_idx] = edge
        return edge

    eval_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}

    def evaluate_variables() -> list[dict[str, Any]]:
        delta_np = delta.detach().cpu().numpy().astype(np.float64)
        items: list[dict[str, Any]] = []
        with torch.no_grad():
            for graph_id, var_i in var_by_id.items():
                final_shift = float(baseline_np[var_i] + delta_np[var_i])
                base = base_by_id[graph_id]
                state = state_by_id[graph_id]
                _, _, source_vertices, source_joints = replay_vertices(
                    model=model,
                    state=state,
                    pose_delta=pose_delta[var_i : var_i + 1],
                    ray_delta=zero_ray,
                )
                eval_base = {
                    **base,
                    "source_vertices": source_vertices[0].detach().cpu().numpy().astype(np.float64),
                    "source_joints": source_joints[0].detach().cpu().numpy().astype(np.float64),
                }
                evaluated = evaluate_row(eval_base, scale, final_shift, eval_cache, args)
                frame_idx = require_int(evaluated.get("frame_idx"), "frame_idx")
                interior = interior_measurement(evaluated, edge_for_frame(frame_idx), args)
                items.append(
                    {
                        "graph_id": graph_id,
                        "var_i": var_i,
                        "final_shift_m": final_shift,
                        "evaluated": evaluated,
                        "interior": interior,
                    }
                )
        return items

    smooth_pairs: list[tuple[int, int, int]] = []
    by_hand: dict[tuple[str, int], list[int]] = {}
    for var_i, row in enumerate(variable_inputs):
        by_hand.setdefault(
            (require_str(row.get("hand_side"), "hand_side"), require_int(row.get("hand_index"), "hand_index")),
            [],
        ).append(var_i)
    for row_ids in by_hand.values():
        ordered = sorted(row_ids, key=lambda i: require_int(variable_inputs[i].get("frame_idx"), "frame_idx"))
        for a, b in zip(ordered[:-1], ordered[1:]):
            dt = max(
                1,
                require_int(variable_inputs[b].get("frame_idx"), "frame_idx")
                - require_int(variable_inputs[a].get("frame_idx"), "frame_idx"),
            )
            if dt <= int(args.max_temporal_smooth_gap_frames):
                smooth_pairs.append((a, b, dt))

    outer_reports: list[dict[str, Any]] = []
    for outer_i in range(int(args.outer_iters)):
        current_np = delta.detach().cpu().numpy().astype(np.float64)
        items = evaluate_variables()
        factor_targets: dict[int, torch.Tensor] = {}
        factor_rows = 0
        prior_smooth_rows = 0
        for item in items:
            var_i = require_int(item.get("var_i"), "var_i")
            evaluated = require_dict(item.get("evaluated"), "evaluated")
            interior = require_dict(item.get("interior"), "interior")
            projection = require_dict(
                evaluated.get("projection_residual_to_measurement_px"),
                "projection residual",
            )
            gap_values = cast(np.ndarray, interior["interior_gap_values_m"])
            if (
                projection.get("residual_ok") is True
                and interior.get("interior_measured") is True
                and gap_values.size >= int(args.min_depth_pixels)
            ):
                factor_targets[var_i] = torch.tensor(
                    thin(gap_values, int(args.max_interior_samples_per_row)).astype(np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                factor_rows += 1
            else:
                prior_smooth_rows += 1
        current = torch.tensor(current_np.astype(np.float32), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([delta], lr=float(args.lr))
        loss_history: list[float] = []
        for _ in range(int(args.inner_iters)):
            optimizer.zero_grad(set_to_none=True)
            terms: list[torch.Tensor] = [robust_l1(delta / float(args.sigma_delta_prior_m)).mean()]
            for var_i, gaps in factor_targets.items():
                terms.append(
                    float(args.w_interior_depth)
                    * robust_l1(
                        (delta[var_i] - current[var_i] + gaps) / float(args.sigma_interior_depth_m)
                    ).mean()
                )
            for a, b, dt in smooth_pairs:
                terms.append(
                    float(args.w_delta_smooth)
                    * robust_l1(
                        (delta[b] - delta[a]) / (float(args.sigma_delta_step_m) * float(dt))
                    ).mean()
                )
            total = torch.stack(terms).mean()
            total.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(torch.minimum(torch.maximum(delta, lower), upper))
            loss_history.append(float(total.detach().cpu()))
        outer_reports.append(
            {
                "outer_iteration": outer_i,
                "interior_depth_factor_rows": factor_rows,
                "interior_prior_smooth_rows": prior_smooth_rows,
                "inner_loss_first": loss_history[:3],
                "inner_loss_last": loss_history[-3:],
                **FALSE_READY,
            }
        )

    final_items = evaluate_variables()
    delta_np = delta.detach().cpu().numpy().astype(np.float64)
    variable_rows: list[dict[str, Any]] = []
    for item in final_items:
        graph_id = require_str(item.get("graph_id"), "graph id")
        var_i = require_int(item.get("var_i"), "var_i")
        evaluated = require_dict(item.get("evaluated"), "evaluated")
        interior = require_dict(item.get("interior"), "interior")
        source_row = variable_inputs[var_i]
        source_gap = finite_or_none(source_row.get("owner_median_gap_m"), "source gap")
        new_gap = finite_or_none(evaluated.get("owner_median_gap_m"), "new gap")
        variable_rows.append(
            {
                "case": case,
                "interior_owned_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "interior_owned_full_residual_graph:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "frame_idx": require_int(evaluated.get("frame_idx"), "frame_idx"),
                "hand_side": require_str(evaluated.get("hand_side"), "hand_side"),
                "hand_index": require_int(evaluated.get("hand_index"), "hand_index"),
                "source_relinearized_total_hand_ray_shift_m": float(baseline_np[var_i]),
                "interior_delta_shift_m": float(delta_np[var_i]),
                "interior_total_hand_ray_shift_m": float(item["final_shift_m"]),
                "interior_delta_bound_hit": bool(
                    math.isclose(float(delta_np[var_i]), float(lower_np[var_i]), abs_tol=1e-5)
                    or math.isclose(float(delta_np[var_i]), float(upper_np[var_i]), abs_tol=1e-5)
                ),
                "legacy_solver_state": require_str(evaluated.get("solver_state"), "solver state"),
                "legacy_owner_depth_state": require_str(evaluated.get("owner_depth_state"), "owner depth state"),
                "legacy_metric_depth_compatible": bool(evaluated.get("metric_depth_compatible") is True),
                "legacy_depth_repair_factor_candidate": bool(
                    evaluated.get("depth_repair_factor_candidate") is True
                ),
                "legacy_owner_median_gap_m": new_gap,
                "source_legacy_metric_depth_compatible": bool(
                    source_row.get("metric_depth_compatible") is True
                ),
                "source_legacy_owner_median_gap_m": source_gap,
                "interior_state": require_str(interior.get("interior_state"), "interior state"),
                "interior_metric_depth_compatible": bool(
                    interior.get("interior_state") == "interior_metric_depth_compatible"
                ),
                "interior_valid_pixels": require_int(interior.get("interior_valid_pixels"), "interior pixels"),
                "edge_owner_pixels": require_int(interior.get("edge_owner_pixels"), "edge pixels"),
                "interior_median_gap_m": interior.get("interior_median_gap_m"),
                "interior_p95_abs_gap_m": interior.get("interior_p95_abs_gap_m"),
                "projection_residual_to_measurement_px": evaluated.get(
                    "projection_residual_to_measurement_px"
                ),
                **FALSE_READY,
            }
        )

    nonvariable_rows: list[dict[str, Any]] = []
    for row in pose_rows:
        graph_id = require_str(row.get("hand_depth_repair_graph_variable_id"), "pose graph id")
        if graph_id in var_by_id:
            continue
        interior = interior_measurement(row, edge_for_frame(require_int(row.get("frame_idx"), "frame_idx")), args) if row.get("partitions") is not None else {
            "interior_state": "interior_unobserved",
            "interior_valid_pixels": 0,
            "edge_owner_pixels": 0,
        }
        nonvariable_rows.append(
            {
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "legacy_metric_depth_compatible": bool(row.get("metric_depth_compatible") is True),
                "interior_state": require_str(interior.get("interior_state"), "interior state"),
                "interior_metric_depth_compatible": bool(
                    interior.get("interior_state") == "interior_metric_depth_compatible"
                ),
            }
        )

    legacy_accepted_variables = bool_count(variable_rows, "legacy_metric_depth_compatible")
    interior_accepted_variables = bool_count(variable_rows, "interior_metric_depth_compatible")
    legacy_accepted_nonvariables = bool_count(nonvariable_rows, "legacy_metric_depth_compatible")
    interior_accepted_nonvariables = bool_count(nonvariable_rows, "interior_metric_depth_compatible")
    report = {
        "method": "solve_v17_interior_owned_full_residual_hand_graph",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "interior_owned_variable_rows": len(variable_rows),
        "interior_outer_iterations": int(args.outer_iters),
        "interior_inner_iterations_per_outer": int(args.inner_iters),
        "interior_delta_bound_hit_rows": bool_count(variable_rows, "interior_delta_bound_hit"),
        "interior_delta_shift_m": numeric_summary(variable_rows, "interior_delta_shift_m"),
        "source_legacy_metric_depth_compatible_variable_rows": bool_count(
            variable_rows,
            "source_legacy_metric_depth_compatible",
        ),
        "legacy_metric_depth_compatible_variable_rows": legacy_accepted_variables,
        "legacy_depth_repair_factor_candidate_variable_rows": bool_count(
            variable_rows,
            "legacy_depth_repair_factor_candidate",
        ),
        "interior_metric_depth_compatible_variable_rows": interior_accepted_variables,
        "interior_state_counts_variable_rows": state_counts(variable_rows, "interior_state"),
        "legacy_owner_depth_state_counts_variable_rows": state_counts(
            variable_rows,
            "legacy_owner_depth_state",
        ),
        "interior_median_gap_m_variable_rows": numeric_summary(variable_rows, "interior_median_gap_m"),
        "nonvariable_rows_total": len(nonvariable_rows),
        "legacy_metric_depth_compatible_nonvariable_rows": legacy_accepted_nonvariables,
        "interior_metric_depth_compatible_nonvariable_rows": interior_accepted_nonvariables,
        "interior_state_counts_nonvariable_rows": state_counts(nonvariable_rows, "interior_state"),
        "metric_hand_state_accepted_rows_legacy_predicate": legacy_accepted_variables
        + legacy_accepted_nonvariables,
        "metric_hand_state_accepted_rows_interior_predicate": interior_accepted_variables
        + interior_accepted_nonvariables,
        "source_pose_graph_accepted_rows_legacy_predicate": require_int(
            pose_graph.get("metric_hand_state_accepted_rows_after_relinearized_graph"),
            f"{case} pose graph accepted rows",
        ),
        "outer_iterations": outer_reports,
        "parameters": {
            "edge_window_px": int(args.edge_window_px),
            "edge_depth_range_m": float(args.edge_depth_range_m),
            "edge_band_dilation_px": int(args.edge_band_dilation_px),
            "max_abs_interior_delta_m": float(args.max_abs_interior_delta_m),
            "max_abs_hand_ray_shift_m": float(args.max_abs_hand_ray_shift_m),
            "sigma_interior_depth_m": float(args.sigma_interior_depth_m),
            "sigma_delta_prior_m": float(args.sigma_delta_prior_m),
            "sigma_delta_step_m": float(args.sigma_delta_step_m),
            "max_interior_samples_per_row": int(args.max_interior_samples_per_row),
            "min_depth_pixels": int(args.min_depth_pixels),
            "max_median_abs_depth_gap_m": float(args.max_median_abs_depth_gap_m),
            "max_p95_abs_depth_gap_m": float(args.max_p95_abs_depth_gap_m),
        },
        "problem_semantics": {
            "ownership": (
                "depth-observation factors use only UniDepth pixels outside hand-independent "
                "depth-discontinuity bands; edge-band pixels are excluded as unowned observations"
            ),
            "fixed_state": "MANO pose deltas, case-global scale, camera trajectory, object state remain fixed",
            "acceptance": (
                "legacy predicate keeps the historical all-pixel owner-partition measurement for "
                "comparability; interior predicate applies the same thresholds to interior-owned pixels"
            ),
            "claim_limit": (
                "this graph repairs hand depth-observation ownership only; object geometry, object pose, "
                "and contact ownership remain open V17 requirements"
            ),
        },
        "rows": variable_rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_interior_owned_full_residual_hand_graph.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    patch_legacy_mano_loader()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    mano_model_path = args.wilor_mano_right
    if mano_model_path is None:
        mano_model_path = args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_model_path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO_RIGHT model: {mano_model_path}")
    mano_cls = load_wilor_mano_class(args.wilor_root)
    with contextlib.redirect_stdout(sys.stderr):
        model = mano_cls(
            model_path=str(mano_model_path),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False,
            batch_size=1,
        ).to(device)
    reports = [case_problem(case, model, args, device) for case in args.cases]
    rows = [
        require_dict(row, "variable row")
        for report in reports
        for row in require_list(report.get("rows"), "rows")
    ]
    summary = {
        "method": "solve_v17_interior_owned_full_residual_hand_graph",
        "status": STATUS,
        "claim": CLAIM,
        "device": str(device),
        "case_count": len(reports),
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "interior_owned_variable_rows": len(rows),
        "interior_delta_bound_hit_rows": bool_count(rows, "interior_delta_bound_hit"),
        "interior_delta_shift_m": numeric_summary(rows, "interior_delta_shift_m"),
        "source_legacy_metric_depth_compatible_variable_rows": bool_count(
            rows,
            "source_legacy_metric_depth_compatible",
        ),
        "legacy_metric_depth_compatible_variable_rows": bool_count(rows, "legacy_metric_depth_compatible"),
        "legacy_depth_repair_factor_candidate_variable_rows": bool_count(
            rows,
            "legacy_depth_repair_factor_candidate",
        ),
        "interior_metric_depth_compatible_variable_rows": bool_count(
            rows,
            "interior_metric_depth_compatible",
        ),
        "interior_state_counts_variable_rows": state_counts(rows, "interior_state"),
        "interior_median_gap_m_variable_rows": numeric_summary(rows, "interior_median_gap_m"),
        "metric_hand_state_accepted_rows_legacy_predicate": sum(
            require_int(report.get("metric_hand_state_accepted_rows_legacy_predicate"), "legacy accepted")
            for report in reports
        ),
        "metric_hand_state_accepted_rows_interior_predicate": sum(
            require_int(report.get("metric_hand_state_accepted_rows_interior_predicate"), "interior accepted")
            for report in reports
        ),
        "source_pose_graph_accepted_rows_legacy_predicate": sum(
            require_int(report.get("source_pose_graph_accepted_rows_legacy_predicate"), "source accepted")
            for report in reports
        ),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "interior_owned_variable_rows": require_int(
                    report.get("interior_owned_variable_rows"),
                    "variables",
                ),
                "legacy_metric_depth_compatible_variable_rows": require_int(
                    report.get("legacy_metric_depth_compatible_variable_rows"),
                    "legacy compatible variables",
                ),
                "interior_metric_depth_compatible_variable_rows": require_int(
                    report.get("interior_metric_depth_compatible_variable_rows"),
                    "interior compatible variables",
                ),
                "metric_hand_state_accepted_rows_legacy_predicate": require_int(
                    report.get("metric_hand_state_accepted_rows_legacy_predicate"),
                    "legacy accepted",
                ),
                "metric_hand_state_accepted_rows_interior_predicate": require_int(
                    report.get("metric_hand_state_accepted_rows_interior_predicate"),
                    "interior accepted",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_interior_owned_full_residual_hand_graph_summary.json", summary)
    return summary


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
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--pose-full-residual-graph-root",
        type=Path,
        default=Path(
            "/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--outer-iters", type=int, default=3)
    parser.add_argument("--inner-iters", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--edge-window-px", type=int, default=7)
    parser.add_argument("--edge-depth-range-m", type=float, default=0.10)
    parser.add_argument("--edge-band-dilation-px", type=int, default=6)
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-interior-delta-m", type=float, default=0.10)
    parser.add_argument("--max-interior-samples-per-row", type=int, default=48)
    parser.add_argument("--sigma-interior-depth-m", type=float, default=0.02)
    parser.add_argument("--sigma-delta-prior-m", type=float, default=0.08)
    parser.add_argument("--sigma-delta-step-m", type=float, default=0.03)
    parser.add_argument("--w-interior-depth", type=float, default=2.0)
    parser.add_argument("--w-delta-smooth", type=float, default=1.0)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
