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
from build_v17_hand_depth_repair_residual_owner_state import row_samples, selected_residual
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
from build_v17_hand_tail_support_state import (
    case_support_sources,
    existing_path,
    independent_support_state,
    selected_support_state,
    source_summary,
    subset_support,
    support_shapes_for_row,
)
from build_v17_mano_articulation_factor_input import (
    assignment_pairs as mano_assignment_pairs,
    front_surface_vertex_samples,
)
from build_v17_post_temporal_depth_observation_support_state import (
    SAME_SIDE_INDEPENDENT_SUPPORT_STATES,
    independent_keypoint_support_state,
)
from refit_mano_pose_contact_v3 import apply_side_sign, hand_span_torch, robust_l1, rotvec_to_matrix
from solve_v17_hand_depth_repair_graph import build_base_row, evaluate_row, numeric_summary
from solve_v17_hand_temporal_owner_weighted_refit import (
    assignment_pairs,
    compatible_anchor_gaps,
    public_assignment,
    thin,
)
from solve_v17_mano_articulation_local import (
    corrected_replayed_state,
    factor_arrays,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    project_depth_torch,
    project_torch,
)
from solve_v17_post_temporal_depth_observation_weighted_refit import keypoint_sigma


STATUS = "v17_relinearized_hand_surface_observation_graph_qc"
CLAIM = (
    "This artifact tests whether stale surface ownership explains the failed coupled hand-depth graph. "
    "Each outer pass replays the current MANO and scalar depth state, rebuilds owner partitions, "
    "refreshes residual-to-compatible-surface assignments and supported UniDepth observation factors, "
    "then optimizes bounded scalar depth and per-row MANO pose deltas under the refreshed factors. "
    "The output is a diagnostic for ownership relinearization, not V3 solver closure."
)
FULL_RESIDUAL_STATUS = "v17_full_residual_relinearized_hand_surface_observation_graph_qc"
FULL_RESIDUAL_CLAIM = (
    "This artifact tests whether the current relinearized graph failed because residual rows with "
    "valid scalar hand-depth state were left outside the variable set. It starts from the sparse "
    "relinearized state, promotes every residual hand-depth row with a finite hand-ray shift into "
    "the same scalar factor graph, and measures the result through full MANO reprojection and "
    "UniDepth resampling. The output tests scalar coverage only unless geometry pose loss is enabled."
)

OBSERVATION_FACTOR_STATES = {
    "same_side_independent_keypoint_partial",
    "same_side_independent_keypoint_strong",
}


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[require_str(row.get(key), key)] += 1
    return dict(sorted(counts.items()))


def finite_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, label)


def status_for_scope(scope: str) -> str:
    if scope == "sparse_applied":
        return STATUS
    if scope == "full_residual_coverage":
        return FULL_RESIDUAL_STATUS
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def claim_for_scope(scope: str) -> str:
    if scope == "sparse_applied":
        return CLAIM
    if scope == "full_residual_coverage":
        return FULL_RESIDUAL_CLAIM
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def variable_graph_id(row: dict[str, Any]) -> str:
    return require_str(row.get("hand_depth_repair_graph_variable_id"), "variable graph id")


def baseline_shift(row: dict[str, Any], scope: str) -> float:
    if scope == "sparse_applied":
        return finite_float(row.get("post_temporal_observation_total_hand_ray_shift_m"), "baseline shift")
    if scope == "full_residual_coverage":
        return finite_float(row.get("relinearized_total_hand_ray_shift_m"), "source relinearized shift")
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def pose_delta_array(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("relinearized_pose_delta_rotvec")
    if raw is None:
        return np.zeros((15, 3), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != (15, 3) or not np.all(np.isfinite(arr)):
        raise RuntimeError("relinearized_pose_delta_rotvec must be a finite 15x3 array")
    return arr


def optimize_geometry_pose(scope: str, args: argparse.Namespace) -> bool:
    if scope == "sparse_applied":
        return bool(args.optimize_geometry_pose)
    if scope == "full_residual_coverage":
        return bool(args.full_residual_optimize_geometry_pose)
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def replay_vertices(
    *,
    model: Any,
    state: dict[str, Any],
    pose_delta: torch.Tensor,
    ray_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hand_pose = rotvec_to_matrix(pose_delta) @ state["hand_pose"]
    out = model(
        global_orient=state["global_orient"],
        hand_pose=hand_pose,
        betas=state["betas"],
        return_verts=True,
        pose2rot=False,
    )
    canonical_vertices = apply_side_sign(out.vertices, float(state["sign"]))
    canonical_joints = apply_side_sign(out.joints, float(state["sign"]))
    local_vertices = (
        state["local_scale"] * torch.matmul(canonical_vertices, state["local_rotation"].T)
        + state["local_translation"]
    )
    local_joints = (
        state["local_scale"] * torch.matmul(canonical_joints, state["local_rotation"].T)
        + state["local_translation"]
    )
    source_vertices = local_vertices + state["translation"]
    source_joints = local_joints + state["translation"]
    ray_shift = state["ray_shift"] + ray_delta
    corrected_vertices = state["graph_scale"] * source_vertices + ray_shift * state["center_ray"]
    corrected_joints = state["graph_scale"] * source_joints + ray_shift * state["center_ray"]
    return corrected_vertices, corrected_joints, source_vertices, source_joints


def geometry_eval_metrics(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    state: dict[str, Any],
    factors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    residual_vertices = vertices[0, factors["residual_vertex_id"]]
    depth_gap = residual_vertices[:, 2] - factors["target_depth"]
    uv = project_depth_torch(
        residual_vertices,
        state["intrinsics"],
        state["projection_source_size"],
        state["depth_shape"],
    )
    projection_to_seed = uv - factors["target_xy"]
    joint_uv = project_torch(joints[0], state["intrinsics"])
    joint_residual = torch.linalg.norm(joint_uv - state["keypoints2d"], dim=1)
    depth_np = depth_gap.detach().cpu().numpy()
    projection_np = torch.linalg.norm(projection_to_seed, dim=1).detach().cpu().numpy()
    joint_np = joint_residual.detach().cpu().numpy()
    span = hand_span_torch(joints).detach().cpu()
    return {
        "factor_sample_count": int(factors["sample_count"].detach().cpu()),
        "residual_vertex_minus_seed_metric_depth_m": summarize(depth_np.astype(float).tolist()),
        "residual_projection_to_seed_pixel_px": summarize(projection_np.astype(float).tolist()),
        "joint_reprojection_px": summarize(joint_np.astype(float).tolist()),
        "hand_span_m": float(span),
        "depth_abs_median_m": float(np.median(np.abs(depth_np))),
        "depth_abs_p95_m": float(np.percentile(np.abs(depth_np), 95.0)),
        "projection_to_seed_median_px": float(np.median(projection_np)),
        "joint_reprojection_median_px": float(np.median(joint_np)),
        "joint_reprojection_p95_px": float(np.percentile(joint_np, 95.0)),
    }


def local_geometry_loss(
    *,
    vertices: torch.Tensor,
    joints: torch.Tensor,
    state: dict[str, Any],
    factors: dict[str, torch.Tensor],
    pose_delta: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    residual_vertices = vertices[0, factors["residual_vertex_id"]]
    depth_loss = robust_l1(
        (residual_vertices[:, 2] - factors["target_depth"]) / float(args.sigma_geometry_depth_m)
    ).mean()
    uv = project_depth_torch(
        residual_vertices,
        state["intrinsics"],
        state["projection_source_size"],
        state["depth_shape"],
    )
    projection_loss = robust_l1((uv - factors["target_xy"]) / float(args.sigma_geometry_projection_px)).mean()
    joint_uv = project_torch(joints[0], state["intrinsics"])
    joint_loss = robust_l1((joint_uv - state["keypoints2d"]) / float(args.sigma_joint_px)).mean()
    pose_loss = robust_l1(pose_delta / float(args.sigma_pose_delta_rad)).mean()
    span = hand_span_torch(joints)
    span_loss = robust_l1(torch.relu(float(args.min_span_m) - span) / float(args.sigma_span_m)) + robust_l1(
        torch.relu(span - float(args.max_span_m)) / float(args.sigma_span_m)
    )
    return (
        float(args.w_geometry_depth) * depth_loss
        + float(args.w_geometry_projection) * projection_loss
        + float(args.w_joint) * joint_loss
        + float(args.w_pose) * pose_loss
        + float(args.w_span) * span_loss
    )


def tensor_values(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(values.astype(np.float32), dtype=torch.float32, device=device)


def relinearized_owner_state(
    row: dict[str, Any],
    assignment: dict[str, Any] | None,
    args: argparse.Namespace,
) -> str:
    if row.get("metric_depth_compatible") is True:
        return "relinearized_reprojected_metric_depth_compatible"
    projection = require_dict(row.get("projection_residual_to_measurement_px"), "projection residual")
    if projection.get("residual_ok") is not True:
        return "relinearized_reprojected_projection_untrusted"
    if assignment is None:
        return "relinearized_reprojected_residual_unobserved"
    fraction = assignment.get("nearby_compatible_assignment_fraction")
    if fraction is None:
        return "relinearized_reprojected_residual_unobserved"
    if float(fraction) >= float(args.min_local_projection_candidate_fraction):
        return "relinearized_reprojected_local_surface_factor_candidate"
    if float(fraction) >= float(args.min_mixed_projection_depth_fraction):
        return "relinearized_reprojected_mixed_surface_depth_owner"
    return "relinearized_reprojected_depth_observation_owner"


def factor_state(row: dict[str, Any]) -> str:
    state = require_str(row.get("relinearized_reprojection_state"), "relinearized state")
    if row.get("relinearized_surface_factor_row") is True:
        return "relinearized_surface_factor_variable"
    if row.get("relinearized_depth_observation_factor_row") is True:
        support = require_str(row.get("independent_keypoint_support_state"), "keypoint support")
        return f"relinearized_depth_observation_{support}_factor_variable"
    if row.get("relinearized_compatible_anchor_row") is True:
        return "relinearized_compatible_anchor_variable"
    if state == "relinearized_reprojected_depth_observation_owner":
        support = require_str(row.get("independent_keypoint_support_state"), "keypoint support")
        return f"relinearized_depth_observation_{support}_prior_smooth_variable"
    if state == "relinearized_reprojected_projection_untrusted":
        return "relinearized_projection_untrusted_prior_smooth_variable"
    if state == "relinearized_reprojected_metric_depth_compatible":
        return "relinearized_compatible_without_anchor_prior_smooth_variable"
    if state == "relinearized_reprojected_residual_unobserved":
        return "relinearized_unobserved_prior_smooth_variable"
    return "relinearized_sparse_owner_prior_smooth_variable"


def report_filename(scope: str) -> str:
    if scope == "sparse_applied":
        return "v17_relinearized_hand_surface_observation_graph.json"
    if scope == "full_residual_coverage":
        return "v17_full_residual_relinearized_hand_surface_observation_graph.json"
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def summary_filename(scope: str) -> str:
    if scope == "sparse_applied":
        return "v17_relinearized_hand_surface_observation_graph_summary.json"
    if scope == "full_residual_coverage":
        return "v17_full_residual_relinearized_hand_surface_observation_graph_summary.json"
    raise RuntimeError(f"unknown relinearized variable scope: {scope}")


def vertex_ids_for_current_surface(
    *,
    source_vertices: np.ndarray,
    base: dict[str, Any],
    evaluated: dict[str, Any],
    scale: float,
    shift: float,
    args: argparse.Namespace,
) -> np.ndarray:
    ray = np.asarray(base["center_ray"], dtype=np.float64)
    corrected = float(scale) * source_vertices + float(shift) * ray[None, :]
    depth_shape_raw = evaluated.get("depth_shape")
    projection_size_raw = evaluated.get("projection_source_size")
    if not isinstance(depth_shape_raw, list) or len(depth_shape_raw) != 2:
        raise RuntimeError("evaluated row depth_shape must be a two-item list")
    if not isinstance(projection_size_raw, list) or len(projection_size_raw) != 2:
        raise RuntimeError("evaluated row projection_source_size must be a two-item list")
    samples = front_surface_vertex_samples(
        corrected,
        np.asarray(base["intrinsics"], dtype=np.float64),
        (float(projection_size_raw[0]), float(projection_size_raw[1])),
        (int(depth_shape_raw[0]), int(depth_shape_raw[1])),
    )
    if samples is None:
        raise RuntimeError("current MANO surface does not project into the depth image")
    x = np.asarray(evaluated.get("x"), dtype=np.int32)
    y = np.asarray(evaluated.get("y"), dtype=np.int32)
    hand_z = np.asarray(evaluated.get("hand_z"), dtype=np.float64)
    sx = cast(np.ndarray, samples["x"]).astype(np.int32)
    sy = cast(np.ndarray, samples["y"]).astype(np.int32)
    sz = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
    if len(x) != len(sx) or not np.array_equal(x, sx) or not np.array_equal(y, sy):
        raise RuntimeError("current vertex-owner samples disagree with evaluated hand-surface pixels")
    max_depth_error = float(np.max(np.abs(hand_z - sz))) if len(hand_z) else 0.0
    if max_depth_error > float(args.max_surface_depth_reconstruction_error_m):
        raise RuntimeError(
            f"current vertex-owner depth disagrees with evaluated row by {max_depth_error:.9f} m"
        )
    return cast(np.ndarray, samples["vertex_id"]).astype(np.int32)


def support_for_selected_pixels(
    *,
    row: dict[str, Any],
    samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]],
    selected: np.ndarray,
    frame: dict[str, Any],
    support_sources: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hand_i = require_int(row.get("hand_index"), "hand_index")
    shapes = support_shapes_for_row(
        frame=frame,
        hand_i=hand_i,
        support_sources=support_sources,
        args=args,
    )
    support = subset_support(
        x=cast(np.ndarray, samples["x"]).astype(np.int32),
        y=cast(np.ndarray, samples["y"]).astype(np.int32),
        selected=selected,
        shapes=shapes,
        projection_source_size=cast(tuple[float, float], samples["projection_source_size"]),
        depth_shape=cast(tuple[int, int], samples["depth_shape"]),
        args=args,
    )
    return {
        "selected_support_state": selected_support_state({**row, "tail_factor_candidate": True}, support),
        "independent_support_state": independent_support_state({**row, "tail_factor_candidate": True}, support),
        "independent_keypoint_support_state": independent_keypoint_support_state(support, args),
        "support_shape_counts": {name: len(value) for name, value in shapes.items()},
        "support": support,
    }


def current_factor_targets(
    *,
    case: str,
    evaluated_items: list[dict[str, Any]],
    frames: dict[int, dict[str, Any]],
    support_sources: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    factor_rows: list[dict[str, Any]] = []
    geometry_items: list[dict[str, Any]] = []
    scalar_geometry_targets: dict[int, torch.Tensor] = {}
    observation_targets: dict[int, tuple[torch.Tensor, float, str]] = {}
    anchor_targets: dict[int, torch.Tensor] = {}
    for item in evaluated_items:
        var_i = require_int(item.get("var_i"), "var_i")
        graph_id = require_str(item.get("graph_id"), "graph id")
        evaluated = require_dict(item.get("evaluated"), "evaluated row")
        state = require_str(evaluated.get("relinearized_reprojection_state"), "relinearized state")
        samples: dict[str, np.ndarray | tuple[int, int] | tuple[float, float]] | None = None
        selected: np.ndarray | None = None
        surface_factor = False
        anchor_factor = False
        observation_factor = False
        support: dict[str, Any] = {
            "selected_support_state": "not_depth_observation_owner",
            "independent_support_state": "not_depth_observation_owner",
            "independent_keypoint_support_state": "same_side_independent_keypoints_unmeasured",
        }
        assignment_public = None
        surface_assignment = None
        scalar_target = np.asarray([], dtype=np.float64)
        observation_target = np.asarray([], dtype=np.float64)
        anchor_target = np.asarray([], dtype=np.float64)
        if evaluated.get("partitions") is not None:
            samples = row_samples(evaluated)
            selected = selected_residual(evaluated, samples, args)
        if state in {
            "relinearized_reprojected_local_surface_factor_candidate",
            "relinearized_reprojected_mixed_surface_depth_owner",
        } and samples is not None and selected is not None:
            vertex_id = vertex_ids_for_current_surface(
                source_vertices=cast(np.ndarray, item["source_vertices"]),
                base=require_dict(item.get("base"), "base"),
                evaluated=evaluated,
                scale=finite_float(item.get("scale"), "scale"),
                shift=finite_float(item.get("final_shift_m"), "final shift"),
                args=args,
            )
            surface_assignment = mano_assignment_pairs(evaluated, samples, selected, vertex_id, args)
            assignment_public = surface_assignment
            pairs = require_dict(surface_assignment.get("factor_pair_arrays"), "surface factor arrays")
            residual_hand = np.asarray(pairs.get("residual_hand_depth_m"), dtype=np.float64)
            seed_metric = np.asarray(pairs.get("seed_metric_depth_m"), dtype=np.float64)
            scalar_target = thin(
                (seed_metric - residual_hand).astype(np.float64),
                int(args.max_factor_samples_per_row),
            )
            surface_factor = bool(
                require_int(surface_assignment.get("assigned_residual_sample_count"), "assigned count")
                >= int(args.min_post_temporal_factor_samples)
            )
            if surface_factor:
                row_for_factors = {"assignment": surface_assignment}
                geometry_items.append(
                    {
                        "graph_id": graph_id,
                        "var_i": var_i,
                        "state": item["state"],
                        "factors": factor_arrays(row_for_factors, args, device),
                    }
                )
                scalar_geometry_targets[var_i] = tensor_values(scalar_target, device)
        elif (
            state == "relinearized_reprojected_metric_depth_compatible"
            and samples is not None
            and selected is not None
        ):
            anchor_target = thin(
                compatible_anchor_gaps(evaluated, samples, args),
                int(args.max_factor_samples_per_row),
            )
            anchor_factor = bool(anchor_target.size >= int(args.min_post_temporal_factor_samples))
            if anchor_factor:
                anchor_targets[var_i] = tensor_values(anchor_target, device)
        elif (
            state == "relinearized_reprojected_depth_observation_owner"
            and samples is not None
            and selected is not None
        ):
            frame_idx = require_int(evaluated.get("frame_idx"), "frame_idx")
            support = support_for_selected_pixels(
                row=evaluated,
                samples=samples,
                selected=selected,
                frame=require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}"),
                support_sources=support_sources,
                args=args,
            )
            keypoint_state = require_str(
                support.get("independent_keypoint_support_state"),
                "keypoint support state",
            )
            box_state = require_str(support.get("independent_support_state"), "box support state")
            hand_z = cast(np.ndarray, samples["hand_z"]).astype(np.float64)
            metric_z = cast(np.ndarray, samples["metric_z"]).astype(np.float64)
            if box_state in SAME_SIDE_INDEPENDENT_SUPPORT_STATES and keypoint_state in OBSERVATION_FACTOR_STATES:
                observation_target = thin(
                    (metric_z[selected] - hand_z[selected]).astype(np.float64),
                    int(args.max_depth_observation_samples_per_row),
                )
            observation_factor = bool(
                observation_target.size >= int(args.min_post_temporal_factor_samples)
            )
            if observation_factor:
                observation_targets[var_i] = (
                    tensor_values(observation_target, device),
                    keypoint_sigma(keypoint_state, args),
                    keypoint_state,
                )
        factor_row = {
            "case": case,
            "relinearized_factor_variable_id": graph_id.replace(
                "hand_depth_repair_graph:",
                "relinearized_hand_surface_observation_graph:",
                1,
            ),
            "source_hand_depth_repair_graph_variable_id": graph_id,
            "frame_idx": require_int(evaluated.get("frame_idx"), "frame_idx"),
            "hand_side": require_str(evaluated.get("hand_side"), "hand_side"),
            "hand_index": require_int(evaluated.get("hand_index"), "hand_index"),
            "relinearized_reprojection_state": state,
            "relinearized_surface_factor_row": surface_factor,
            "relinearized_compatible_anchor_row": anchor_factor,
            "relinearized_depth_observation_factor_row": observation_factor,
            "relinearized_prior_smooth_only_row": bool(
                not surface_factor and not anchor_factor and not observation_factor
            ),
            "selected_residual_sample_count": 0
            if selected is None
            else int(np.count_nonzero(selected)),
            "surface_assignment": None if assignment_public is None else public_assignment(assignment_public),
            "surface_scalar_target_delta_m": summarize(scalar_target.astype(float).tolist()),
            "compatible_anchor_gap_m": summarize(anchor_target.astype(float).tolist()),
            "depth_observation_target_delta_m": summarize(observation_target.astype(float).tolist()),
            **support,
            **FALSE_READY,
        }
        factor_rows.append({**factor_row, "relinearized_input_factor_state": factor_state(factor_row)})
    return {
        "factor_rows": factor_rows,
        "geometry_items": geometry_items,
        "scalar_geometry_targets": scalar_geometry_targets,
        "observation_targets": observation_targets,
        "anchor_targets": anchor_targets,
    }


def case_problem(case: str, model: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scope = require_str(args.variable_scope, "variable scope")
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
        "post_temporal_depth_observation_weighted_refit": existing_path(
            args.post_temporal_depth_observation_weighted_refit_root
            / case
            / "v17_post_temporal_depth_observation_weighted_refit.json",
            f"{case} post-temporal depth-observation weighted refit",
        ),
        "coupled_hand_depth_mano_observation_graph": existing_path(
            args.coupled_hand_depth_mano_observation_graph_root
            / case
            / "v17_coupled_hand_depth_mano_observation_graph.json",
            f"{case} coupled hand-depth MANO observation graph",
        ),
    }
    if scope == "full_residual_coverage":
        paths["source_relinearized_hand_surface_observation_graph"] = existing_path(
            args.source_relinearized_hand_surface_observation_graph_root
            / case
            / "v17_relinearized_hand_surface_observation_graph.json",
            f"{case} source sparse relinearized hand surface-observation graph",
        )
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    for name in [
        "visible_surface",
        "hand_metric_depth_state",
        "hand_depth_repair_graph",
        "post_temporal_depth_observation_weighted_refit",
        "coupled_hand_depth_mano_observation_graph",
        *(
            ["source_relinearized_hand_surface_observation_graph"]
            if scope == "full_residual_coverage"
            else []
        ),
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count disagrees with {name}")
    support_sources = case_support_sources(case, args)
    hands = annotation_hand_index(frames)
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth archive",
    )
    depth = depth_archive(depth_path)
    repair = payloads["hand_depth_repair_graph"]
    weighted = payloads["post_temporal_depth_observation_weighted_refit"]
    coupled = payloads["coupled_hand_depth_mano_observation_graph"]
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    weighted_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "weighted graph id"): row
        for row in [
            require_dict(raw, "weighted row")
            for raw in require_list(weighted.get("rows"), f"{case} weighted rows")
        ]
    }
    if scope == "sparse_applied":
        variable_inputs = [
            row
            for row in weighted_by_id.values()
            if row.get("post_temporal_observation_delta_applied") is True
            and row.get("post_temporal_observation_total_hand_ray_shift_m") is not None
        ]
    elif scope == "full_residual_coverage":
        source_relinearized = payloads["source_relinearized_hand_surface_observation_graph"]
        variable_inputs = [
            row
            for row in [
                require_dict(raw, "source relinearized row")
                for raw in require_list(source_relinearized.get("rows"), f"{case} source relinearized rows")
            ]
            if (
                row.get("depth_repair_factor_candidate") is True
                or row.get("relinearized_delta_applied") is True
            )
            and row.get("relinearized_total_hand_ray_shift_m") is not None
        ]
    else:
        raise RuntimeError(f"{case} unknown relinearized variable scope: {scope}")
    variable_inputs = sorted(
        variable_inputs,
        key=lambda row: (
            require_str(row.get("hand_side"), "hand_side"),
            require_int(row.get("hand_index"), "hand_index"),
            require_int(row.get("frame_idx"), "frame_idx"),
        ),
    )
    var_by_id = {
        variable_graph_id(row): i
        for i, row in enumerate(variable_inputs)
    }
    lower_np = np.asarray(
        [
            max(
                -float(args.max_abs_hand_ray_shift_m)
                - baseline_shift(row, scope),
                -float(args.max_abs_relinearized_delta_m),
            )
            for row in variable_inputs
        ],
        dtype=np.float32,
    )
    upper_np = np.asarray(
        [
            min(
                float(args.max_abs_hand_ray_shift_m)
                - baseline_shift(row, scope),
                float(args.max_abs_relinearized_delta_m),
            )
            for row in variable_inputs
        ],
        dtype=np.float32,
    )
    if np.any(lower_np > upper_np):
        bad = np.flatnonzero(lower_np > upper_np)
        raise RuntimeError(f"{case} relinearized scalar bounds are inconsistent at {bad[:12].tolist()}")
    lower = torch.tensor(lower_np, dtype=torch.float32, device=device)
    upper = torch.tensor(upper_np, dtype=torch.float32, device=device)
    scalar_delta = torch.zeros(len(variable_inputs), dtype=torch.float32, device=device, requires_grad=True)
    init_pose_delta_np = np.stack([pose_delta_array(row) for row in variable_inputs]).astype(np.float32)
    pose_delta = torch.tensor(init_pose_delta_np, dtype=torch.float32, device=device, requires_grad=True)
    pose_optimization_enabled = optimize_geometry_pose(scope, args)
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    base_by_id: dict[str, dict[str, Any]] = {}
    state_by_id: dict[str, dict[str, Any]] = {}
    for graph_id, var_i in var_by_id.items():
        row = variable_inputs[var_i]
        frame_idx = require_int(row.get("frame_idx"), "frame_idx")
        side = require_str(row.get("hand_side"), "hand_side")
        hand_i = require_int(row.get("hand_index"), "hand_index")
        hand = require_dict(hands.get((frame_idx, side, hand_i)), f"{case} annotation hand {graph_id}")
        repair_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
        shifted_repair = {
            **repair_row,
            "hand_ray_shift_m": baseline_shift(row, scope),
        }
        state_by_id[graph_id] = corrected_replayed_state(
            model=model,
            hand=hand,
            graph_row=shifted_repair,
            depth=depth,
            device=device,
        )
    metric_rows = [
        require_dict(raw, "metric row")
        for raw in require_list(payloads["hand_metric_depth_state"].get("rows"), f"{case} metric rows")
    ]
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}

    def base_for_metric_row(metric_row: dict[str, Any]) -> dict[str, Any]:
        graph_id = require_str(
            metric_row.get("hand_metric_depth_variable_id"),
            "metric id",
        ).replace("hand_metric_depth:", "hand_depth_repair_graph:", 1)
        if graph_id in base_by_id:
            return base_by_id[graph_id]
        frame_idx = require_int(metric_row.get("frame_idx"), "frame_idx")
        side = require_str(metric_row.get("hand_side"), "hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "hand_index")
        frame = require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}")
        base = build_base_row(
            case=case,
            frame=frame,
            metric_row=metric_row,
            hand=hands.get((frame_idx, side, hand_i)),
            depth=depth,
            mask_cache=mask_cache,
            args=args,
        )
        base_by_id[graph_id] = base
        return base

    for row in metric_rows:
        graph_id = require_str(
            row.get("hand_metric_depth_variable_id"),
            "metric id",
        ).replace("hand_metric_depth:", "hand_depth_repair_graph:", 1)
        if graph_id in var_by_id:
            base_for_metric_row(row)

    def evaluate_variables() -> list[dict[str, Any]]:
        scalar_np = scalar_delta.detach().cpu().numpy().astype(np.float64)
        items: list[dict[str, Any]] = []
        with torch.no_grad():
            for graph_id, var_i in var_by_id.items():
                source_row = variable_inputs[var_i]
                source_shift = baseline_shift(source_row, scope)
                final_shift = source_shift + float(scalar_np[var_i])
                base = require_dict(base_by_id.get(graph_id), f"{case} base {graph_id}")
                state = require_dict(state_by_id.get(graph_id), f"{case} state {graph_id}")
                _, _, source_vertices, source_joints = replay_vertices(
                    model=model,
                    state=state,
                    pose_delta=pose_delta[var_i : var_i + 1],
                    ray_delta=scalar_delta[var_i],
                )
                source_vertices_np = source_vertices[0].detach().cpu().numpy().astype(np.float64)
                source_joints_np = source_joints[0].detach().cpu().numpy().astype(np.float64)
                eval_base = {
                    **base,
                    "source_vertices": source_vertices_np,
                    "source_joints": source_joints_np,
                }
                evaluated = evaluate_row(eval_base, scale, final_shift, eval_mask_cache, args)
                assignment = None
                if evaluated.get("owner_sample_partition") is not None and isinstance(
                    evaluated.get("partitions"),
                    dict,
                ):
                    samples = row_samples(evaluated)
                    selected = selected_residual(evaluated, samples, args)
                    assignment = assignment_pairs(evaluated, samples, selected, args)
                enriched = {
                    **evaluated,
                    "source_post_temporal_observation_weighted_refit_variable_id": source_row.get(
                        "post_temporal_depth_observation_weighted_refit_variable_id"
                    ),
                    "source_temporal_refit_state": source_row.get("source_temporal_refit_state"),
                    "source_post_temporal_observation_reprojection_state": source_row.get(
                        "post_temporal_observation_reprojection_state"
                    ),
                    "source_post_temporal_observation_owner_median_gap_m": source_row.get("owner_median_gap_m"),
                    "source_post_temporal_observation_total_hand_ray_shift_m": source_row.get(
                        "post_temporal_observation_total_hand_ray_shift_m"
                    ),
                    "source_relinearized_total_hand_ray_shift_m": source_shift,
                    "relinearized_delta_shift_m": float(scalar_np[var_i]),
                    "relinearized_total_hand_ray_shift_m": final_shift,
                    "relinearized_delta_applied": True,
                    "relinearized_pose_delta_applied": bool(
                        float(np.max(np.abs(pose_delta[var_i].detach().cpu().numpy()))) > 0.0
                    ),
                    "relinearized_pose_delta_rotvec": pose_delta[var_i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist(),
                    "relinearized_reprojection_assignment": None
                    if assignment is None
                    else public_assignment(assignment),
                    **FALSE_READY,
                }
                items.append(
                    {
                        "graph_id": graph_id,
                        "var_i": var_i,
                        "base": eval_base,
                        "state": state,
                        "scale": scale,
                        "final_shift_m": final_shift,
                        "source_vertices": source_vertices_np,
                        "source_joints": source_joints_np,
                        "evaluated": {
                            **enriched,
                            "relinearized_reprojection_state": relinearized_owner_state(
                                enriched,
                                assignment,
                                args,
                            ),
                        },
                    }
                )
        return items

    smooth_pairs: list[tuple[int, int, int]] = []
    by_hand: dict[tuple[str, int], list[int]] = {}
    for var_i, row in enumerate(variable_inputs):
        by_hand.setdefault(
            (
                require_str(row.get("hand_side"), "hand_side"),
                require_int(row.get("hand_index"), "hand_index"),
            ),
            [],
        ).append(var_i)
    for row_ids in by_hand.values():
        ordered = sorted(row_ids, key=lambda i: require_int(variable_inputs[i].get("frame_idx"), "frame_idx"))
        for a, b in zip(ordered[:-1], ordered[1:]):
            frame_a = require_int(variable_inputs[a].get("frame_idx"), "frame_idx")
            frame_b = require_int(variable_inputs[b].get("frame_idx"), "frame_idx")
            dt = max(1, frame_b - frame_a)
            if dt <= int(args.max_temporal_smooth_gap_frames):
                smooth_pairs.append((a, b, dt))
    outer_reports: list[dict[str, Any]] = []
    latest_factor_targets: dict[str, Any] | None = None
    for outer_i in range(int(args.outer_iters)):
        current_scalar_np = scalar_delta.detach().cpu().numpy().astype(np.float64)
        evaluated_items = evaluate_variables()
        factor_targets = current_factor_targets(
            case=case,
            evaluated_items=evaluated_items,
            frames=frames,
            support_sources=support_sources,
            device=device,
            args=args,
        )
        latest_factor_targets = factor_targets
        optimizer = torch.optim.Adam([scalar_delta, pose_delta], lr=float(args.lr))
        inner_loss_history: list[float] = []
        for _ in range(int(args.inner_iters)):
            optimizer.zero_grad(set_to_none=True)
            scalar_terms: list[torch.Tensor] = [
                robust_l1(scalar_delta / float(args.sigma_delta_prior_m)).mean()
            ]
            current_scalar = torch.tensor(current_scalar_np, dtype=torch.float32, device=device)
            for var_i, targets in cast(dict[int, torch.Tensor], factor_targets["scalar_geometry_targets"]).items():
                scalar_terms.append(
                    float(args.w_geometry_scalar)
                    * robust_l1(
                        (scalar_delta[var_i] - current_scalar[var_i] - targets)
                        / float(args.sigma_geometry_depth_m)
                    ).mean()
                )
            for var_i, (targets, sigma, _) in cast(
                dict[int, tuple[torch.Tensor, float, str]],
                factor_targets["observation_targets"],
            ).items():
                scalar_terms.append(
                    float(args.w_depth_observation)
                    * robust_l1((scalar_delta[var_i] - current_scalar[var_i] - targets) / sigma).mean()
                )
            for var_i, anchors in cast(dict[int, torch.Tensor], factor_targets["anchor_targets"]).items():
                scalar_terms.append(
                    float(args.w_compatible_anchor)
                    * robust_l1(
                        (anchors + scalar_delta[var_i] - current_scalar[var_i])
                        / float(args.sigma_compatible_anchor_m)
                    ).mean()
                )
            for a, b, dt in smooth_pairs:
                scalar_terms.append(
                    float(args.w_delta_smooth)
                    * robust_l1(
                        (scalar_delta[b] - scalar_delta[a])
                        / (float(args.sigma_delta_step_m) * float(dt))
                    ).mean()
                )
            total_loss = torch.stack(scalar_terms).mean()
            total_loss.backward()
            geometry_items = cast(list[dict[str, Any]], factor_targets["geometry_items"])
            if pose_optimization_enabled:
                for item in geometry_items:
                    var_i = require_int(item.get("var_i"), "geometry var_i")
                    vertices, joints, _, _ = replay_vertices(
                        model=model,
                        state=require_dict(item.get("state"), "geometry state"),
                        pose_delta=pose_delta[var_i : var_i + 1],
                        ray_delta=scalar_delta[var_i],
                    )
                    row_loss = local_geometry_loss(
                        vertices=vertices,
                        joints=joints,
                        state=require_dict(item.get("state"), "geometry state"),
                        factors=cast(dict[str, torch.Tensor], item["factors"]),
                        pose_delta=pose_delta[var_i : var_i + 1],
                        args=args,
                    )
                    (row_loss / max(1, len(geometry_items))).backward()
            optimizer.step()
            with torch.no_grad():
                scalar_delta.copy_(torch.minimum(torch.maximum(scalar_delta, lower), upper))
                if pose_optimization_enabled:
                    pose_delta.clamp_(-float(args.max_pose_delta_rad), float(args.max_pose_delta_rad))
                else:
                    pose_delta.copy_(torch.tensor(init_pose_delta_np, dtype=torch.float32, device=device))
                inner_loss_history.append(float(total_loss.detach().cpu()))
        after_items = evaluate_variables()
        after_rows = [require_dict(item.get("evaluated"), "after evaluated") for item in after_items]
        factor_rows = cast(list[dict[str, Any]], factor_targets["factor_rows"])
        outer_reports.append(
            {
                "outer_iteration": outer_i,
                "factor_state_counts": state_counts(factor_rows, "relinearized_input_factor_state"),
                "reprojection_state_counts_after_inner_solve": state_counts(
                    after_rows,
                    "relinearized_reprojection_state",
                ),
                "surface_factor_rows": bool_count(factor_rows, "relinearized_surface_factor_row"),
                "depth_observation_factor_rows": bool_count(
                    factor_rows,
                    "relinearized_depth_observation_factor_row",
                ),
                "compatible_anchor_rows": bool_count(factor_rows, "relinearized_compatible_anchor_row"),
                "metric_depth_compatible_rows_after_inner_solve": bool_count(
                    after_rows,
                    "metric_depth_compatible",
                ),
                "depth_repair_factor_candidate_rows_after_inner_solve": bool_count(
                    after_rows,
                    "depth_repair_factor_candidate",
                ),
                "inner_loss_history_first": inner_loss_history[:5],
                "inner_loss_history_last": inner_loss_history[-5:],
                **FALSE_READY,
            }
        )
    final_variable_items = evaluate_variables()
    final_variable_by_id = {
        require_str(item.get("graph_id"), "graph id"): item for item in final_variable_items
    }
    variable_input_by_id = {
        variable_graph_id(row): row
        for row in variable_inputs
    }
    scalar_delta_np = scalar_delta.detach().cpu().numpy().astype(np.float64)
    pose_delta_np = pose_delta.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    final_eval_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    for metric_row in metric_rows:
        base = base_for_metric_row(metric_row)
        graph_id = require_str(base.get("hand_depth_repair_graph_variable_id"), "graph id")
        weighted_row = weighted_by_id.get(graph_id)
        source_variable_row = variable_input_by_id.get(graph_id)
        final_shift = None
        source_vertices_np = None
        source_joints_np = None
        if weighted_row is not None and weighted_row.get("post_temporal_observation_total_hand_ray_shift_m") is not None:
            final_shift = finite_float(
                weighted_row.get("post_temporal_observation_total_hand_ray_shift_m"),
                "weighted shift",
            )
        if graph_id in final_variable_by_id:
            item = final_variable_by_id[graph_id]
            final_shift = finite_float(item.get("final_shift_m"), "final variable shift")
            source_vertices_np = cast(np.ndarray, item["source_vertices"])
            source_joints_np = cast(np.ndarray, item["source_joints"])
        eval_base = base
        if source_vertices_np is not None and source_joints_np is not None:
            eval_base = {**base, "source_vertices": source_vertices_np, "source_joints": source_joints_np}
        evaluated = (
            evaluate_row(eval_base, None, None, final_eval_cache, args)
            if final_shift is None
            else evaluate_row(eval_base, scale, final_shift, final_eval_cache, args)
        )
        if source_variable_row is not None:
            source_gap = finite_or_none(source_variable_row.get("owner_median_gap_m"), "source variable gap")
        elif weighted_row is not None:
            source_gap = finite_or_none(weighted_row.get("owner_median_gap_m"), "source gap")
        else:
            source_gap = None
        new_gap = finite_or_none(evaluated.get("owner_median_gap_m"), "new gap")
        assignment = None
        if graph_id in var_by_id and evaluated.get("owner_sample_partition") is not None and isinstance(
            evaluated.get("partitions"),
            dict,
        ):
            samples = row_samples(evaluated)
            selected = selected_residual(evaluated, samples, args)
            assignment = assignment_pairs(evaluated, samples, selected, args)
        enriched = {
            **evaluated,
            "source_post_temporal_observation_weighted_refit_variable_id": None
            if weighted_row is None
            else weighted_row.get("post_temporal_depth_observation_weighted_refit_variable_id"),
            "source_temporal_refit_state": None
            if weighted_row is None
            else weighted_row.get("source_temporal_refit_state"),
            "source_post_temporal_observation_reprojection_state": None
            if weighted_row is None
            else weighted_row.get("post_temporal_observation_reprojection_state"),
            "source_post_temporal_observation_owner_median_gap_m": source_gap,
            "source_post_temporal_observation_total_hand_ray_shift_m": None
            if weighted_row is None
            else weighted_row.get("post_temporal_observation_total_hand_ray_shift_m"),
            "source_relinearized_total_hand_ray_shift_m": None
            if source_variable_row is None
            else baseline_shift(source_variable_row, scope),
            "relinearized_delta_shift_m": None
            if graph_id not in var_by_id
            else float(scalar_delta_np[var_by_id[graph_id]]),
            "relinearized_total_hand_ray_shift_m": final_shift,
            "relinearized_delta_applied": bool(graph_id in var_by_id),
            "relinearized_pose_delta_abs_max_rad": None
            if graph_id not in var_by_id
            else float(np.max(np.abs(pose_delta_np[var_by_id[graph_id]]))),
            "relinearized_pose_delta_rotvec": None
            if graph_id not in var_by_id
            else pose_delta_np[var_by_id[graph_id]].astype(float).tolist(),
            "relinearized_reprojected_depth_improved": bool(
                source_gap is not None
                and new_gap is not None
                and abs(source_gap) - abs(new_gap) >= float(args.min_depth_median_improvement_m)
            ),
            "relinearized_reprojection_assignment": None if assignment is None else public_assignment(assignment),
            **FALSE_READY,
        }
        state = (
            "not_relinearized_hand_depth_row"
            if graph_id not in var_by_id
            else relinearized_owner_state(enriched, assignment, args)
        )
        rows.append({**enriched, "relinearized_reprojection_state": state})
    temporal_rows = [row for row in rows if row.get("source_temporal_refit_state") is not None]
    applied_rows = [row for row in rows if row.get("relinearized_delta_applied") is True]
    residual_rows = [
        row
        for row in applied_rows
        if row["relinearized_reprojection_state"]
        not in {
            "relinearized_reprojected_metric_depth_compatible",
            "relinearized_reprojected_projection_untrusted",
            "relinearized_reprojected_residual_unobserved",
        }
    ]
    source_nonapplied_variable_rows = (
        sum(1 for row in variable_inputs if row.get("relinearized_delta_applied") is not True)
        if scope == "full_residual_coverage"
        else 0
    )
    source_residual_variable_rows = sum(
        1
        for row in variable_inputs
        if row.get("depth_repair_factor_candidate") is True
    )
    if latest_factor_targets is None:
        raise RuntimeError("relinearized graph did not build factor targets")
    final_factor_rows = cast(list[dict[str, Any]], latest_factor_targets["factor_rows"])
    geometry_items_final = cast(list[dict[str, Any]], latest_factor_targets["geometry_items"])
    geometry_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in geometry_items_final:
            var_i = require_int(item.get("var_i"), "geometry var_i")
            graph_id = require_str(item.get("graph_id"), "geometry graph id")
            vertices, joints, _, _ = replay_vertices(
                model=model,
                state=require_dict(item.get("state"), "geometry state"),
                pose_delta=pose_delta[var_i : var_i + 1],
                ray_delta=scalar_delta[var_i],
            )
            geometry_rows.append(
                {
                    "case": case,
                    "source_hand_depth_repair_graph_variable_id": graph_id,
                    "frame_idx": require_int(variable_inputs[var_i].get("frame_idx"), "frame_idx"),
                    "hand_side": require_str(variable_inputs[var_i].get("hand_side"), "hand_side"),
                    "hand_index": require_int(variable_inputs[var_i].get("hand_index"), "hand_index"),
                    "after": geometry_eval_metrics(
                        vertices,
                        joints,
                        require_dict(item.get("state"), "geometry state"),
                        cast(dict[str, torch.Tensor], item["factors"]),
                    ),
                    "pose_delta_abs_max_rad": float(np.max(np.abs(pose_delta_np[var_i]))),
                    **FALSE_READY,
                }
            )
    report = {
        "method": "solve_v17_relinearized_hand_surface_observation_graph",
        "status": status_for_scope(scope),
        "claim": claim_for_scope(scope),
        "case": case,
        "sources": {
            **{name: source_summary(path, payloads[name]) for name, path in paths.items()},
            **{
                f"support_{name}": source_summary(path)
                for name, path in require_dict(support_sources["paths"], "support paths").items()
            },
        },
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "relinearized_variable_scope": scope,
        "relinearized_variable_rows": len(variable_inputs),
        "relinearized_source_nonapplied_variable_rows": source_nonapplied_variable_rows,
        "relinearized_source_residual_variable_rows": source_residual_variable_rows,
        "relinearized_geometry_pose_optimization_enabled": pose_optimization_enabled,
        "relinearized_outer_iterations": int(args.outer_iters),
        "relinearized_inner_iterations_per_outer": int(args.inner_iters),
        "relinearized_surface_factor_rows": bool_count(final_factor_rows, "relinearized_surface_factor_row"),
        "relinearized_depth_observation_factor_rows": bool_count(
            final_factor_rows,
            "relinearized_depth_observation_factor_row",
        ),
        "relinearized_compatible_anchor_rows": bool_count(
            final_factor_rows,
            "relinearized_compatible_anchor_row",
        ),
        "relinearized_input_factor_state_counts": state_counts(
            final_factor_rows,
            "relinearized_input_factor_state",
        ),
        "relinearized_scalar_delta_bound_hit_rows": sum(
            1
            for var_i, delta in enumerate(scalar_delta_np)
            if math.isclose(float(delta), float(lower_np[var_i]), abs_tol=float(args.bound_tolerance_m))
            or math.isclose(float(delta), float(upper_np[var_i]), abs_tol=float(args.bound_tolerance_m))
        ),
        "relinearized_geometry_pose_delta_clamp_hit_rows": sum(
            1
            for var_i in range(len(variable_inputs))
            if float(np.max(np.abs(pose_delta_np[var_i]))) >= float(args.max_pose_delta_rad) - 1.0e-5
        ),
        "relinearized_reprojected_metric_depth_compatible_rows": bool_count(
            applied_rows,
            "metric_depth_compatible",
        ),
        "relinearized_reprojected_depth_improved_rows": sum(
            1 for row in applied_rows if row.get("relinearized_reprojected_depth_improved") is True
        ),
        "metric_hand_state_accepted_rows_after_relinearized_graph": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows_after_relinearized_graph": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "relinearized_reprojection_residual_owner_rows": len(residual_rows),
        "relinearized_reprojection_local_surface_factor_candidate_rows": sum(
            1
            for row in residual_rows
            if row["relinearized_reprojection_state"]
            == "relinearized_reprojected_local_surface_factor_candidate"
        ),
        "relinearized_reprojection_mixed_surface_depth_owner_rows": sum(
            1
            for row in residual_rows
            if row["relinearized_reprojection_state"] == "relinearized_reprojected_mixed_surface_depth_owner"
        ),
        "relinearized_reprojection_depth_observation_owner_rows": sum(
            1
            for row in residual_rows
            if row["relinearized_reprojection_state"] == "relinearized_reprojected_depth_observation_owner"
        ),
        "relinearized_reprojection_projection_untrusted_rows": sum(
            1
            for row in applied_rows
            if row["relinearized_reprojection_state"] == "relinearized_reprojected_projection_untrusted"
        ),
        "relinearized_temporal_reprojection_state_counts": state_counts(
            temporal_rows,
            "relinearized_reprojection_state",
        ),
        "relinearized_owner_depth_state_counts_after_reprojection": state_counts(rows, "owner_depth_state"),
        "relinearized_owner_median_gap_m_after_reprojection": numeric_summary(rows, "owner_median_gap_m"),
        "pose_delta_abs_max_rad": numeric_summary(geometry_rows, "pose_delta_abs_max_rad"),
        "geometry_after_depth_abs_median_m": numeric_summary(geometry_rows, "after.depth_abs_median_m"),
        "source_weighted_refit_comparison": {
            "post_temporal_observation_weighted_variable_rows": weighted.get(
                "post_temporal_observation_weighted_variable_rows"
            ),
            "post_temporal_observation_reprojected_metric_depth_compatible_rows": weighted.get(
                "post_temporal_observation_reprojected_metric_depth_compatible_rows"
            ),
            "metric_hand_state_accepted_rows_after_post_temporal_observation_refit": weighted.get(
                "metric_hand_state_accepted_rows_after_post_temporal_observation_refit"
            ),
            "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit": weighted.get(
                "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit"
            ),
            "post_temporal_observation_reprojection_depth_observation_owner_rows": weighted.get(
                "post_temporal_observation_reprojection_depth_observation_owner_rows"
            ),
            "post_temporal_observation_temporal_reprojection_state_counts": weighted.get(
                "post_temporal_observation_temporal_reprojection_state_counts"
            ),
        },
        "source_fixed_coupled_graph_comparison": {
            "coupled_variable_rows": coupled.get("coupled_variable_rows"),
            "coupled_reprojected_metric_depth_compatible_rows": coupled.get(
                "coupled_reprojected_metric_depth_compatible_rows"
            ),
            "metric_hand_state_accepted_rows_after_coupled_graph": coupled.get(
                "metric_hand_state_accepted_rows_after_coupled_graph"
            ),
            "depth_repair_factor_candidate_rows_after_coupled_graph": coupled.get(
                "depth_repair_factor_candidate_rows_after_coupled_graph"
            ),
            "coupled_reprojection_depth_observation_owner_rows": coupled.get(
                "coupled_reprojection_depth_observation_owner_rows"
            ),
            "coupled_temporal_reprojection_state_counts": coupled.get(
                "coupled_temporal_reprojection_state_counts"
            ),
        },
        "outer_iterations": outer_reports,
        "problem_semantics": {
            "relinearization": "surface owner pixels, MANO vertex ids, compatible seed pixels, and depth-observation support are rebuilt from the current replayed state at every outer pass",
            "variable_scope": (
                "sparse_applied keeps the original post-temporal applied subset; "
                "full_residual_coverage starts from the sparse relinearized graph and promotes every "
                "residual row with a finite scalar hand-depth state"
            ),
            "full_reprojection_oracle": "the same evaluate_row owner measurement path used by earlier V17 hand-depth artifacts decides compatibility after the solve",
            "claim_limit": "camera trajectory, MANO shape, object geometry, object pose, contact, and dense depth remain fixed outside this diagnostic",
        },
        "parameters": {
            "variable_scope": scope,
            "optimize_geometry_pose": bool(args.optimize_geometry_pose),
            "full_residual_optimize_geometry_pose": bool(args.full_residual_optimize_geometry_pose),
            "outer_iters": int(args.outer_iters),
            "inner_iters": int(args.inner_iters),
            "lr": float(args.lr),
            "max_abs_relinearized_delta_m": float(args.max_abs_relinearized_delta_m),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "local_projection_search_radius_px": float(args.local_projection_search_radius_px),
            "compatible_depth_abs_m": float(args.compatible_depth_abs_m),
        },
        "factor_rows": final_factor_rows,
        "geometry_rows": geometry_rows,
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / report_filename(scope), report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    scope = require_str(args.variable_scope, "variable scope")
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
    cases = ["trash_1050", "task5_tomato_960"]
    reports = [case_problem(case, model, args, device) for case in cases]
    rows = [
        require_dict(row, "relinearized row")
        for report in reports
        for row in require_list(report.get("rows"), "rows")
    ]
    factor_rows = [
        require_dict(row, "factor row")
        for report in reports
        for row in require_list(report.get("factor_rows"), "factor rows")
    ]
    geometry_rows = [
        require_dict(row, "geometry row")
        for report in reports
        for row in require_list(report.get("geometry_rows"), "geometry rows")
    ]
    summary = {
        "method": "solve_v17_relinearized_hand_surface_observation_graph",
        "status": status_for_scope(scope),
        "claim": claim_for_scope(scope),
        "wilor_root": str(args.wilor_root),
        "wilor_mano_right": str(mano_model_path),
        "device": str(device),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "relinearized_variable_rows": require_int(
                    report.get("relinearized_variable_rows"),
                    "variables",
                ),
                "relinearized_reprojected_metric_depth_compatible_rows": require_int(
                    report.get("relinearized_reprojected_metric_depth_compatible_rows"),
                    "compatible rows",
                ),
                "metric_hand_state_accepted_rows_after_relinearized_graph": require_int(
                    report.get("metric_hand_state_accepted_rows_after_relinearized_graph"),
                    "accepted rows",
                ),
                "depth_repair_factor_candidate_rows_after_relinearized_graph": require_int(
                    report.get("depth_repair_factor_candidate_rows_after_relinearized_graph"),
                    "residual rows",
                ),
                "relinearized_variable_scope": require_str(
                    report.get("relinearized_variable_scope"),
                    "variable scope",
                ),
                "relinearized_source_nonapplied_variable_rows": require_int(
                    report.get("relinearized_source_nonapplied_variable_rows"),
                    "source nonapplied variables",
                ),
                "relinearized_source_residual_variable_rows": require_int(
                    report.get("relinearized_source_residual_variable_rows"),
                    "source residual variables",
                ),
                "relinearized_temporal_reprojection_state_counts": require_dict(
                    report.get("relinearized_temporal_reprojection_state_counts"),
                    "temporal state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "relinearized_variable_scope": scope,
        "relinearized_variable_rows": sum(
            require_int(report.get("relinearized_variable_rows"), "variables") for report in reports
        ),
        "relinearized_source_nonapplied_variable_rows": sum(
            require_int(report.get("relinearized_source_nonapplied_variable_rows"), "source nonapplied variables")
            for report in reports
        ),
        "relinearized_source_residual_variable_rows": sum(
            require_int(report.get("relinearized_source_residual_variable_rows"), "source residual variables")
            for report in reports
        ),
        "relinearized_geometry_pose_optimization_enabled": bool(
            any(report.get("relinearized_geometry_pose_optimization_enabled") is True for report in reports)
        ),
        "relinearized_surface_factor_rows": bool_count(factor_rows, "relinearized_surface_factor_row"),
        "relinearized_depth_observation_factor_rows": bool_count(
            factor_rows,
            "relinearized_depth_observation_factor_row",
        ),
        "relinearized_compatible_anchor_rows": bool_count(
            factor_rows,
            "relinearized_compatible_anchor_row",
        ),
        "relinearized_input_factor_state_counts": state_counts(
            factor_rows,
            "relinearized_input_factor_state",
        ),
        "relinearized_scalar_delta_bound_hit_rows": sum(
            require_int(report.get("relinearized_scalar_delta_bound_hit_rows"), "scalar bound rows")
            for report in reports
        ),
        "relinearized_geometry_pose_delta_clamp_hit_rows": sum(
            require_int(report.get("relinearized_geometry_pose_delta_clamp_hit_rows"), "pose clamp rows")
            for report in reports
        ),
        "relinearized_reprojected_metric_depth_compatible_rows": sum(
            require_int(report.get("relinearized_reprojected_metric_depth_compatible_rows"), "compatible rows")
            for report in reports
        ),
        "metric_hand_state_accepted_rows_after_relinearized_graph": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows_after_relinearized_graph": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "relinearized_reprojection_residual_owner_rows": sum(
            require_int(report.get("relinearized_reprojection_residual_owner_rows"), "residual owners")
            for report in reports
        ),
        "relinearized_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(report.get("relinearized_reprojection_local_surface_factor_candidate_rows"), "local")
            for report in reports
        ),
        "relinearized_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(report.get("relinearized_reprojection_mixed_surface_depth_owner_rows"), "mixed")
            for report in reports
        ),
        "relinearized_reprojection_depth_observation_owner_rows": sum(
            require_int(report.get("relinearized_reprojection_depth_observation_owner_rows"), "depth")
            for report in reports
        ),
        "relinearized_reprojection_projection_untrusted_rows": sum(
            require_int(report.get("relinearized_reprojection_projection_untrusted_rows"), "projection")
            for report in reports
        ),
        "relinearized_temporal_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("relinearized_temporal_reprojection_state_counts"),
                                "temporal counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "pose_delta_abs_max_rad": numeric_summary(geometry_rows, "pose_delta_abs_max_rad"),
        "geometry_after_depth_abs_median_m": numeric_summary(geometry_rows, "after.depth_abs_median_m"),
        **FALSE_READY,
    }
    write_json(args.output_root / summary_filename(scope), summary)
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
        "--post-temporal-depth-observation-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit"),
    )
    parser.add_argument(
        "--coupled-hand-depth-mano-observation-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_coupled_hand_depth_mano_observation_graph"),
    )
    parser.add_argument(
        "--source-relinearized-hand-surface-observation-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--variable-scope",
        choices=["sparse_applied", "full_residual_coverage"],
        default="sparse_applied",
    )
    parser.add_argument("--optimize-geometry-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-residual-optimize-geometry-pose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--outer-iters", type=int, default=3)
    parser.add_argument("--inner-iters", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--near-object-mask-px", type=float, default=20.0)
    parser.add_argument("--far-object-mask-px", type=float, default=80.0)
    parser.add_argument("--min-depth-pixels", type=int, default=12)
    parser.add_argument("--max-depth-samples-per-row", type=int, default=48)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-median-abs-depth-gap-m", type=float, default=0.03)
    parser.add_argument("--max-p95-abs-depth-gap-m", type=float, default=0.08)
    parser.add_argument("--compatible-depth-abs-m", type=float, default=0.03)
    parser.add_argument("--local-projection-search-radius-px", type=float, default=8.0)
    parser.add_argument("--min-local-projection-candidate-fraction", type=float, default=0.75)
    parser.add_argument("--min-mixed-projection-depth-fraction", type=float, default=0.25)
    parser.add_argument("--min-post-temporal-factor-samples", type=int, default=3)
    parser.add_argument("--max-factor-samples-per-row", type=int, default=64)
    parser.add_argument("--max-pairs-per-row", type=int, default=64)
    parser.add_argument("--max-depth-observation-samples-per-row", type=int, default=64)
    parser.add_argument("--max-hand-median-px", type=float, default=45.0)
    parser.add_argument("--max-hand-p95-px", type=float, default=95.0)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-relinearized-delta-m", type=float, default=0.10)
    parser.add_argument("--min-corrected-hand-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-geometry-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-geometry-projection-px", type=float, default=6.0)
    parser.add_argument("--sigma-joint-px", type=float, default=18.0)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.18)
    parser.add_argument("--sigma-span-m", type=float, default=0.02)
    parser.add_argument("--sigma-depth-observation-strong-m", type=float, default=0.03)
    parser.add_argument("--sigma-depth-observation-partial-m", type=float, default=0.05)
    parser.add_argument("--sigma-compatible-anchor-m", type=float, default=0.03)
    parser.add_argument("--sigma-delta-prior-m", type=float, default=0.08)
    parser.add_argument("--sigma-delta-step-m", type=float, default=0.03)
    parser.add_argument("--w-geometry-depth", type=float, default=2.0)
    parser.add_argument("--w-geometry-scalar", type=float, default=1.0)
    parser.add_argument("--w-geometry-projection", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.6)
    parser.add_argument("--w-pose", type=float, default=0.25)
    parser.add_argument("--w-span", type=float, default=0.25)
    parser.add_argument("--w-depth-observation", type=float, default=1.0)
    parser.add_argument("--w-compatible-anchor", type=float, default=1.0)
    parser.add_argument("--w-delta-smooth", type=float, default=1.0)
    parser.add_argument("--max-temporal-smooth-gap-frames", type=int, default=45)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.35)
    parser.add_argument("--min-span-m", type=float, default=0.10)
    parser.add_argument("--max-span-m", type=float, default=0.22)
    parser.add_argument("--max-surface-depth-reconstruction-error-m", type=float, default=1e-4)
    parser.add_argument("--min-keypoint-supported-fraction", type=float, default=0.05)
    parser.add_argument("--strong-keypoint-supported-fraction", type=float, default=0.25)
    parser.add_argument("--max-assign-center-px", type=float, default=150.0)
    parser.add_argument("--near-support-bbox-px", type=float, default=20.0)
    parser.add_argument("--near-support-keypoint-px", type=float, default=20.0)
    parser.add_argument("--min-depth-median-improvement-m", type=float, default=0.005)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
