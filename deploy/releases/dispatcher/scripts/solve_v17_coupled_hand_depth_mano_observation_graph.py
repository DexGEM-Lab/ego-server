#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
from build_v17_hand_tail_support_state import existing_path, source_summary
from build_v17_hand_temporal_reprojection_residual_owner_state import temporal_owner_state
from refit_mano_pose_contact_v3 import apply_side_sign, hand_span_torch, robust_l1, rotvec_to_matrix
from solve_v17_hand_depth_repair_graph import build_base_row, evaluate_row, numeric_summary
from solve_v17_hand_temporal_owner_weighted_refit import assignment_pairs, public_assignment, thin
from solve_v17_mano_articulation_local import (
    corrected_replayed_state,
    factor_arrays,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    project_depth_torch,
    project_torch,
)
from solve_v17_post_temporal_depth_observation_weighted_refit import (
    input_factor_state,
    keypoint_sigma,
    post_temporal_inputs,
)


STATUS = "v17_coupled_hand_depth_mano_observation_graph_qc"
CLAIM = (
    "This artifact tests one coupled hand-depth graph after the owner-weighted temporal refit. "
    "It optimizes bounded temporal camera-ray hand-depth variables, per-row MANO pose deltas for "
    "current local and mixed surface-owner rows, and supported UniDepth observation factors in one "
    "objective, then reprojects MANO and resamples UniDepth. It is a causal diagnostic, not annotation closure."
)

LOCAL_STATE = "owner_weighted_reprojected_local_surface_factor_candidate"
MIXED_STATE = "owner_weighted_reprojected_mixed_surface_depth_owner"
DEPTH_STATE = "owner_weighted_reprojected_depth_observation_owner"
UNTRUSTED_STATE = "owner_weighted_reprojected_projection_untrusted"
COMPATIBLE_STATE = "owner_weighted_reprojected_metric_depth_compatible"


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


def replay_coupled_vertices(
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


def tensor_targets(values: Any, device: torch.device, max_count: int) -> torch.Tensor:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise RuntimeError("target values must be a vector")
    arr = thin(arr.astype(np.float64), max_count).astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def output_state(row: dict[str, Any], assignment: dict[str, Any] | None, args: argparse.Namespace) -> str:
    if row.get("source_temporal_refit_state") is None:
        return "not_coupled_hand_depth_row"
    if row.get("coupled_delta_applied") is not True:
        return "coupled_delta_not_applied"
    if row.get("owner_sample_partition") is None or not isinstance(row.get("partitions"), dict):
        return "coupled_reprojected_unobserved"
    if assignment is None:
        raise RuntimeError("applied coupled row needs an assignment state")
    temporal_state = temporal_owner_state(row, assignment, args)
    mapping = {
        "temporal_reprojection_metric_depth_compatible": "coupled_reprojected_metric_depth_compatible",
        "temporal_reprojection_projection_untrusted": "coupled_reprojected_projection_untrusted",
        "temporal_reprojection_residual_unobserved": "coupled_reprojected_residual_unobserved",
        "temporal_reprojection_local_surface_factor_candidate": "coupled_reprojected_local_surface_factor_candidate",
        "temporal_reprojection_mixed_surface_depth_owner": "coupled_reprojected_mixed_surface_depth_owner",
        "temporal_reprojection_depth_observation_owner": "coupled_reprojected_depth_observation_owner",
        "temporal_delta_not_applied": "coupled_delta_not_applied",
    }
    if temporal_state not in mapping:
        raise RuntimeError(f"unknown coupled reprojection owner state: {temporal_state}")
    return mapping[temporal_state]


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
        "hand_temporal_owner_weighted_refit": existing_path(
            args.hand_temporal_owner_weighted_refit_root
            / case
            / "v17_hand_temporal_owner_weighted_refit.json",
            f"{case} hand temporal owner-weighted refit",
        ),
        "post_temporal_mano_factor_input": existing_path(
            args.post_temporal_mano_factor_input_root
            / case
            / "v17_post_temporal_mano_factor_input.json",
            f"{case} post-temporal MANO factor input",
        ),
        "post_temporal_depth_observation_support": existing_path(
            args.post_temporal_depth_observation_support_state_root
            / case
            / "v17_post_temporal_depth_observation_support_state.json",
            f"{case} post-temporal depth-observation support state",
        ),
        "post_temporal_depth_observation_weighted_refit": existing_path(
            args.post_temporal_depth_observation_weighted_refit_root
            / case
            / "v17_post_temporal_depth_observation_weighted_refit.json",
            f"{case} post-temporal depth-observation weighted refit",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    frame_count = len(frames)
    for name in [
        "visible_surface",
        "hand_metric_depth_state",
        "hand_depth_repair_graph",
        "hand_temporal_owner_weighted_refit",
        "post_temporal_mano_factor_input",
        "post_temporal_depth_observation_support",
        "post_temporal_depth_observation_weighted_refit",
    ]:
        if frame_count != require_int(payloads[name].get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame_count disagrees with {name}")
    owner_weighted = payloads["hand_temporal_owner_weighted_refit"]
    support = payloads["post_temporal_depth_observation_support"]
    weighted_refit = payloads["post_temporal_depth_observation_weighted_refit"]
    scalar_inputs = post_temporal_inputs(case, owner_weighted, support, args)
    variable_inputs = [
        row for row in scalar_inputs if row.get("post_temporal_observation_variable_candidate") is True
    ]
    var_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "scalar graph id"): i
        for i, row in enumerate(variable_inputs)
    }
    lower_np = np.asarray(
        [finite_float(row.get("post_temporal_delta_lower_bound_m"), "lower bound") for row in variable_inputs],
        dtype=np.float32,
    )
    upper_np = np.asarray(
        [finite_float(row.get("post_temporal_delta_upper_bound_m"), "upper bound") for row in variable_inputs],
        dtype=np.float32,
    )
    if np.any(lower_np > upper_np):
        bad = np.flatnonzero(lower_np > upper_np)
        raise RuntimeError(f"{case} coupled scalar bounds are inconsistent at {bad[:12].tolist()}")
    lower = torch.tensor(lower_np, dtype=torch.float32, device=device)
    upper = torch.tensor(upper_np, dtype=torch.float32, device=device)
    scalar_delta = torch.zeros(len(variable_inputs), dtype=torch.float32, device=device, requires_grad=True)
    geometry_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "factor graph id"): row
        for row in [
            require_dict(raw, "post-temporal MANO factor row")
            for raw in require_list(payloads["post_temporal_mano_factor_input"].get("rows"), "MANO factor rows")
            if require_dict(raw, "post-temporal MANO factor row").get(
                "post_temporal_mano_factor_input_materialized"
            )
            is True
        ]
    }
    hands = annotation_hand_index(frames)
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth archive",
    )
    depth = depth_archive(depth_path)
    repair = payloads["hand_depth_repair_graph"]
    repair_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "repair graph id"): row
        for row in [require_dict(raw, "repair row") for raw in require_list(repair.get("rows"), "repair rows")]
    }
    owner_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "owner graph id"): row
        for row in [
            require_dict(raw, "owner row")
            for raw in require_list(owner_weighted.get("rows"), f"{case} owner-weighted rows")
        ]
    }
    geometry_states: list[dict[str, Any]] = []
    pose_ids: list[str] = []
    for graph_id, factor_row in geometry_by_id.items():
        if graph_id not in var_by_id:
            raise RuntimeError(f"{case} geometry row has no coupled scalar variable: {graph_id}")
        frame_idx = require_int(factor_row.get("frame_idx"), "factor frame_idx")
        side = require_str(factor_row.get("hand_side"), "factor hand side")
        hand_i = require_int(factor_row.get("hand_index"), "factor hand index")
        hand = require_dict(hands.get((frame_idx, side, hand_i)), f"{case} annotation hand {graph_id}")
        graph_row = require_dict(repair_by_id.get(graph_id), f"{case} repair row {graph_id}")
        owner_row = require_dict(owner_by_id.get(graph_id), f"{case} owner row {graph_id}")
        shifted_graph_row = {
            **graph_row,
            "hand_ray_shift_m": owner_row.get("owner_weighted_total_hand_ray_shift_m"),
        }
        state = corrected_replayed_state(
            model=model,
            hand=hand,
            graph_row=shifted_graph_row,
            depth=depth,
            device=device,
        )
        factors = factor_arrays(factor_row, args, device)
        geometry_states.append(
            {
                "graph_id": graph_id,
                "var_i": var_by_id[graph_id],
                "factor_row": factor_row,
                "state": state,
                "factors": factors,
            }
        )
        pose_ids.append(graph_id)
    pose_delta = torch.zeros((len(geometry_states), 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    observation_targets: dict[int, tuple[torch.Tensor, float, str]] = {}
    anchor_targets: dict[int, torch.Tensor] = {}
    fixed_targets_before: dict[int, np.ndarray] = {}
    for graph_id, var_i in var_by_id.items():
        row = variable_inputs[var_i]
        observation = tensor_targets(
            row.get("_depth_observation_target_delta_m"),
            device,
            int(args.max_depth_observation_samples_per_row),
        )
        anchor = tensor_targets(
            row.get("_compatible_anchor_gap_m"),
            device,
            int(args.max_factor_samples_per_row),
        )
        geometry = thin(
            np.asarray(row.get("_geometry_target_delta_m"), dtype=np.float64),
            int(args.max_factor_samples_per_row),
        )
        fixed_targets_before[var_i] = np.concatenate(
            [
                -geometry,
                anchor.detach().cpu().numpy().astype(np.float64),
                -observation.detach().cpu().numpy().astype(np.float64),
            ]
        )
        if observation.numel():
            support_state = require_str(row.get("independent_keypoint_support_state"), "keypoint support state")
            observation_targets[var_i] = (observation, keypoint_sigma(support_state, args), support_state)
        if anchor.numel():
            anchor_targets[var_i] = anchor
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
    for rows_for_hand in by_hand.values():
        ordered = sorted(rows_for_hand, key=lambda i: require_int(variable_inputs[i].get("frame_idx"), "frame_idx"))
        for a, b in zip(ordered[:-1], ordered[1:]):
            frame_a = require_int(variable_inputs[a].get("frame_idx"), "frame_idx")
            frame_b = require_int(variable_inputs[b].get("frame_idx"), "frame_idx")
            dt = max(1, frame_b - frame_a)
            if dt <= int(args.max_temporal_smooth_gap_frames):
                smooth_pairs.append((a, b, dt))
    optimizer = torch.optim.Adam([scalar_delta, pose_delta], lr=float(args.lr))
    loss_history: list[float] = []
    before_geometry: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for pose_i, item in enumerate(geometry_states):
            vertices, joints, _, _ = replay_coupled_vertices(
                model=model,
                state=item["state"],
                pose_delta=torch.zeros((1, 15, 3), dtype=torch.float32, device=device),
                ray_delta=torch.tensor(0.0, dtype=torch.float32, device=device),
            )
            before_geometry[require_str(item["graph_id"], "graph id")] = geometry_eval_metrics(
                vertices,
                joints,
                item["state"],
                item["factors"],
            )
    for _ in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        scalar_loss_terms: list[torch.Tensor] = []
        if scalar_delta.numel():
            scalar_loss_terms.append(robust_l1(scalar_delta / float(args.sigma_delta_prior_m)).mean())
        for var_i, (targets, sigma, _) in observation_targets.items():
            scalar_loss_terms.append(
                float(args.w_depth_observation) * robust_l1((scalar_delta[var_i] - targets) / sigma).mean()
            )
        for var_i, anchors in anchor_targets.items():
            scalar_loss_terms.append(
                float(args.w_compatible_anchor) * robust_l1(
                    (anchors + scalar_delta[var_i]) / float(args.sigma_compatible_anchor_m)
                ).mean()
            )
        for a, b, dt in smooth_pairs:
            scalar_loss_terms.append(
                float(args.w_delta_smooth)
                * robust_l1(
                    (scalar_delta[b] - scalar_delta[a]) / (float(args.sigma_delta_step_m) * float(dt))
                ).mean()
            )
        if scalar_loss_terms:
            total_loss = total_loss + torch.stack(scalar_loss_terms).mean()
        total_loss.backward()
        for pose_i, item in enumerate(geometry_states):
            vertices, joints, _, _ = replay_coupled_vertices(
                model=model,
                state=item["state"],
                pose_delta=pose_delta[pose_i : pose_i + 1],
                ray_delta=scalar_delta[int(item["var_i"])],
            )
            row_loss = local_geometry_loss(
                vertices=vertices,
                joints=joints,
                state=item["state"],
                factors=item["factors"],
                pose_delta=pose_delta[pose_i : pose_i + 1],
                args=args,
            )
            (row_loss / max(1, len(geometry_states))).backward()
        optimizer.step()
        with torch.no_grad():
            scalar_delta.copy_(torch.minimum(torch.maximum(scalar_delta, lower), upper))
            pose_delta.clamp_(-float(args.max_pose_delta_rad), float(args.max_pose_delta_rad))
            loss_history.append(float(total_loss.detach().cpu()))
    scalar_delta_np = scalar_delta.detach().cpu().numpy().astype(np.float64)
    pose_delta_np = pose_delta.detach().cpu().numpy().astype(np.float64)
    geometry_rows: list[dict[str, Any]] = []
    pose_source_vertices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        for pose_i, item in enumerate(geometry_states):
            graph_id = require_str(item["graph_id"], "graph id")
            vertices, joints, source_vertices, source_joints = replay_coupled_vertices(
                model=model,
                state=item["state"],
                pose_delta=pose_delta[pose_i : pose_i + 1],
                ray_delta=scalar_delta[int(item["var_i"])],
            )
            after = geometry_eval_metrics(vertices, joints, item["state"], item["factors"])
            before = before_geometry[graph_id]
            depth_improved = bool(
                before["depth_abs_median_m"] - after["depth_abs_median_m"]
                >= float(args.min_depth_median_improvement_m)
            )
            depth_ok = bool(
                after["depth_abs_median_m"] <= float(args.accept_depth_median_m)
                and after["depth_abs_p95_m"] <= float(args.accept_depth_p95_m)
            )
            joint_ok = bool(
                after["joint_reprojection_median_px"] <= float(args.accept_joint_median_px)
                and after["joint_reprojection_p95_px"] <= float(args.accept_joint_p95_px)
            )
            state_name = "coupled_geometry_no_depth_gain"
            if depth_improved:
                state_name = "coupled_geometry_reduces_depth_residual"
            if depth_improved and joint_ok and depth_ok:
                state_name = "coupled_geometry_solved_under_local_thresholds"
            pose_source_vertices[graph_id] = (
                source_vertices[0].detach().cpu().numpy().astype(np.float64),
                source_joints[0].detach().cpu().numpy().astype(np.float64),
            )
            geometry_rows.append(
                {
                    "case": case,
                    "coupled_geometry_variable_id": graph_id.replace(
                        "hand_depth_repair_graph:",
                        "coupled_hand_depth_mano_observation_graph:",
                        1,
                    ),
                    "source_hand_depth_repair_graph_variable_id": graph_id,
                    "source_post_temporal_mano_factor_input_id": item["factor_row"].get(
                        "post_temporal_mano_factor_input_id"
                    ),
                    "frame_idx": require_int(item["factor_row"].get("frame_idx"), "frame_idx"),
                    "hand_side": require_str(item["factor_row"].get("hand_side"), "hand_side"),
                    "hand_index": require_int(item["factor_row"].get("hand_index"), "hand_index"),
                    "source_owner_weighted_reprojection_state": item["factor_row"].get(
                        "source_owner_weighted_reprojection_state"
                    ),
                    "coupled_delta_shift_m": float(scalar_delta_np[int(item["var_i"])]),
                    "coupled_total_hand_ray_shift_m": float(
                        finite_float(
                            variable_inputs[int(item["var_i"])].get("current_owner_weighted_hand_ray_shift_m"),
                            "current shift",
                        )
                        + scalar_delta_np[int(item["var_i"])]
                    ),
                    "coupled_geometry_solve_state": state_name,
                    "coupled_geometry_depth_improved": depth_improved,
                    "coupled_geometry_depth_threshold_met": depth_ok,
                    "coupled_geometry_projection_trusted": joint_ok,
                    "pose_delta_abs_max_rad": float(np.max(np.abs(pose_delta_np[pose_i]))),
                    "before": before,
                    "after": after,
                    **FALSE_READY,
                }
            )
    scalar_rows: list[dict[str, Any]] = []
    for var_i, row in enumerate(variable_inputs):
        delta = float(scalar_delta_np[var_i])
        current_shift = finite_float(row.get("current_owner_weighted_hand_ray_shift_m"), "current shift")
        fixed_before = fixed_targets_before[var_i]
        fixed_after = fixed_before + delta
        scalar_rows.append(
            {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "coupled_delta_shift_m": delta,
                "coupled_total_hand_ray_shift_m": current_shift + delta,
                "post_temporal_observation_factor_state": input_factor_state(row),
                "coupled_delta_bound_hit": bool(
                    math.isclose(delta, float(lower_np[var_i]), abs_tol=float(args.bound_tolerance_m))
                    or math.isclose(delta, float(upper_np[var_i]), abs_tol=float(args.bound_tolerance_m))
                ),
                "coupled_fixed_factor_residual": {
                    "before": {
                        "signed_gap_m": summarize(fixed_before.astype(float).tolist()),
                        "abs_gap_m": summarize(np.abs(fixed_before).astype(float).tolist()),
                    },
                    "after": {
                        "signed_gap_m": summarize(fixed_after.astype(float).tolist()),
                        "abs_gap_m": summarize(np.abs(fixed_after).astype(float).tolist()),
                    },
                },
                "coupled_fixed_factor_depth_improved": bool(
                    len(fixed_before)
                    and float(np.median(np.abs(fixed_before))) - float(np.median(np.abs(fixed_after)))
                    >= float(args.min_depth_median_improvement_m)
                ),
                "coupled_fixed_factor_depth_threshold_met": bool(
                    len(fixed_after)
                    and float(np.median(np.abs(fixed_after))) <= float(args.accept_depth_median_m)
                    and float(np.percentile(np.abs(fixed_after), 95.0)) <= float(args.accept_depth_p95_m)
                ),
                **FALSE_READY,
            }
        )
    solved_by_id = {
        require_str(row.get("source_hand_depth_repair_graph_variable_id"), "scalar source id"): row
        for row in scalar_rows
    }
    scale = finite_float(repair.get("case_global_scale"), f"{case} repair graph scale")
    hand_index = annotation_hand_index(frames)
    mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    eval_mask_cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    for raw in require_list(payloads["hand_metric_depth_state"].get("rows"), f"{case} metric rows"):
        metric_row = require_dict(raw, "metric row")
        frame_idx = require_int(metric_row.get("frame_idx"), "frame_idx")
        side = require_str(metric_row.get("hand_side"), "hand_side")
        hand_i = require_int(metric_row.get("hand_index"), "hand_index")
        frame = require_dict(frames.get(frame_idx), f"{case} frame {frame_idx}")
        base = build_base_row(
            case=case,
            frame=frame,
            metric_row=metric_row,
            hand=hand_index.get((frame_idx, side, hand_i)),
            depth=depth,
            mask_cache=mask_cache,
            args=args,
        )
        graph_id = require_str(base.get("hand_depth_repair_graph_variable_id"), "repair graph id")
        owner_row = owner_by_id.get(graph_id)
        scalar_row = solved_by_id.get(graph_id)
        current_shift = finite_or_none(
            None if owner_row is None else owner_row.get("owner_weighted_total_hand_ray_shift_m"),
            "current owner-weighted shift",
        )
        delta = finite_or_none(None if scalar_row is None else scalar_row.get("coupled_delta_shift_m"), "coupled delta")
        final_shift = None
        if current_shift is not None and base.get("base_available") is True:
            final_shift = current_shift + (0.0 if delta is None else delta)
        eval_base = base
        if graph_id in pose_source_vertices and base.get("base_available") is True:
            source_vertices_np, source_joints_np = pose_source_vertices[graph_id]
            eval_base = {**base, "source_vertices": source_vertices_np, "source_joints": source_joints_np}
        evaluated = (
            evaluate_row(eval_base, None, None, eval_mask_cache, args)
            if final_shift is None
            else evaluate_row(eval_base, scale, final_shift, eval_mask_cache, args)
        )
        source_abs_gap = None
        if owner_row is not None and owner_row.get("owner_median_gap_m") is not None:
            source_abs_gap = abs(finite_float(owner_row.get("owner_median_gap_m"), "source owner gap"))
        new_abs_gap = None
        if evaluated.get("owner_median_gap_m") is not None:
            new_abs_gap = abs(finite_float(evaluated.get("owner_median_gap_m"), "new owner gap"))
        assignment = None
        if (
            scalar_row is not None
            and evaluated.get("owner_sample_partition") is not None
            and isinstance(evaluated.get("partitions"), dict)
        ):
            samples = row_samples(evaluated)
            selected = selected_residual(evaluated, samples, args)
            assignment = assignment_pairs(evaluated, samples, selected, args)
        enriched = {
            **evaluated,
            "source_hand_temporal_owner_weighted_refit_variable_id": None
            if owner_row is None
            else owner_row.get("hand_temporal_owner_weighted_refit_variable_id"),
            "source_temporal_refit_state": None if owner_row is None else owner_row.get("source_temporal_refit_state"),
            "source_owner_weighted_reprojection_state": None
            if owner_row is None
            else owner_row.get("owner_weighted_reprojection_state"),
            "source_owner_weighted_owner_median_gap_m": None
            if owner_row is None
            else owner_row.get("owner_median_gap_m"),
            "source_owner_weighted_total_hand_ray_shift_m": current_shift,
            "coupled_delta_shift_m": delta,
            "coupled_delta_applied": bool(delta is not None),
            "temporal_refit_delta_applied": bool(delta is not None),
            "coupled_pose_delta_applied": bool(graph_id in pose_source_vertices),
            "coupled_total_hand_ray_shift_m": final_shift,
            "coupled_reprojected_depth_improved": bool(
                source_abs_gap is not None
                and new_abs_gap is not None
                and source_abs_gap - new_abs_gap >= float(args.min_depth_median_improvement_m)
            ),
            "coupled_reprojection_assignment": None if assignment is None else public_assignment(assignment),
            **FALSE_READY,
        }
        rows.append({**enriched, "coupled_reprojection_state": output_state(enriched, assignment, args)})
    temporal_rows = [row for row in rows if row.get("source_temporal_refit_state") is not None]
    applied_rows = [row for row in temporal_rows if row.get("coupled_delta_applied") is True]
    residual_rows = [
        row
        for row in applied_rows
        if row["coupled_reprojection_state"]
        not in {
            "coupled_reprojected_metric_depth_compatible",
            "coupled_reprojected_projection_untrusted",
            "coupled_reprojected_unobserved",
        }
    ]
    report = {
        "method": "solve_v17_coupled_hand_depth_mano_observation_graph",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "coupled_variable_rows": len(variable_inputs),
        "coupled_geometry_pose_variable_rows": len(geometry_states),
        "coupled_depth_observation_factor_rows": len(observation_targets),
        "coupled_compatible_anchor_rows": len(anchor_targets),
        "coupled_scalar_delta_bound_hit_rows": bool_count(scalar_rows, "coupled_delta_bound_hit"),
        "coupled_fixed_factor_depth_improved_rows": bool_count(
            scalar_rows,
            "coupled_fixed_factor_depth_improved",
        ),
        "coupled_fixed_factor_depth_threshold_met_rows": bool_count(
            scalar_rows,
            "coupled_fixed_factor_depth_threshold_met",
        ),
        "coupled_geometry_depth_improved_rows": bool_count(
            geometry_rows,
            "coupled_geometry_depth_improved",
        ),
        "coupled_geometry_depth_threshold_met_rows": bool_count(
            geometry_rows,
            "coupled_geometry_depth_threshold_met",
        ),
        "coupled_geometry_projection_trusted_rows": bool_count(
            geometry_rows,
            "coupled_geometry_projection_trusted",
        ),
        "coupled_geometry_pose_delta_clamp_hit_rows": sum(
            1
            for row in geometry_rows
            if float(row.get("pose_delta_abs_max_rad", 0.0)) >= float(args.max_pose_delta_rad) - 1.0e-5
        ),
        "coupled_reprojected_metric_depth_compatible_rows": bool_count(applied_rows, "metric_depth_compatible"),
        "coupled_reprojected_depth_improved_rows": bool_count(
            applied_rows,
            "coupled_reprojected_depth_improved",
        ),
        "metric_hand_state_accepted_rows_after_coupled_graph": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows_after_coupled_graph": bool_count(rows, "depth_repair_factor_candidate"),
        "coupled_reprojection_residual_owner_rows": len(residual_rows),
        "coupled_reprojection_local_surface_factor_candidate_rows": sum(
            1
            for row in residual_rows
            if row["coupled_reprojection_state"] == "coupled_reprojected_local_surface_factor_candidate"
        ),
        "coupled_reprojection_mixed_surface_depth_owner_rows": sum(
            1
            for row in residual_rows
            if row["coupled_reprojection_state"] == "coupled_reprojected_mixed_surface_depth_owner"
        ),
        "coupled_reprojection_depth_observation_owner_rows": sum(
            1
            for row in residual_rows
            if row["coupled_reprojection_state"] == "coupled_reprojected_depth_observation_owner"
        ),
        "coupled_reprojection_projection_untrusted_rows": sum(
            1
            for row in applied_rows
            if row["coupled_reprojection_state"] == "coupled_reprojected_projection_untrusted"
        ),
        "coupled_input_factor_state_counts": state_counts(scalar_rows, "post_temporal_observation_factor_state"),
        "coupled_geometry_solve_state_counts": state_counts(geometry_rows, "coupled_geometry_solve_state")
        if geometry_rows
        else {},
        "coupled_reprojection_state_counts": state_counts(rows, "coupled_reprojection_state"),
        "coupled_temporal_reprojection_state_counts": state_counts(temporal_rows, "coupled_reprojection_state"),
        "coupled_owner_depth_state_counts_after_reprojection": state_counts(rows, "owner_depth_state"),
        "coupled_owner_median_gap_m_after_reprojection": numeric_summary(rows, "owner_median_gap_m"),
        "geometry_before_depth_abs_median_m": numeric_summary(geometry_rows, "before.depth_abs_median_m"),
        "geometry_after_depth_abs_median_m": numeric_summary(geometry_rows, "after.depth_abs_median_m"),
        "pose_delta_abs_max_rad": numeric_summary(geometry_rows, "pose_delta_abs_max_rad"),
        "source_weighted_refit_comparison": {
            "post_temporal_observation_weighted_variable_rows": weighted_refit.get(
                "post_temporal_observation_weighted_variable_rows"
            ),
            "post_temporal_observation_depth_factor_rows": weighted_refit.get(
                "post_temporal_observation_depth_factor_rows"
            ),
            "post_temporal_observation_reprojected_metric_depth_compatible_rows": weighted_refit.get(
                "post_temporal_observation_reprojected_metric_depth_compatible_rows"
            ),
            "metric_hand_state_accepted_rows_after_post_temporal_observation_refit": weighted_refit.get(
                "metric_hand_state_accepted_rows_after_post_temporal_observation_refit"
            ),
            "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit": weighted_refit.get(
                "depth_repair_factor_candidate_rows_after_post_temporal_observation_refit"
            ),
            "post_temporal_observation_reprojection_state_counts": weighted_refit.get(
                "post_temporal_observation_temporal_reprojection_state_counts"
            ),
        },
        "solver": {
            "device": str(device),
            "iters": int(args.iters),
            "lr": float(args.lr),
            "loss_history_first": loss_history[:5],
            "loss_history_last": loss_history[-5:],
            "smoothness_factor_count": len(smooth_pairs),
        },
        "problem_semantics": {
            "optimized_variables": "bounded scalar camera-ray depth deltas for all temporal rows plus per-row MANO hand-pose deltas for current local and mixed owner rows",
            "depth_observation_factor": "same-side keypoint-supported UniDepth rows pull the scalar hand-depth variable",
            "geometry_factor": "post-temporal MANO residual vertices are pulled toward same-hand compatible-depth seed pixels while scalar depth and pose move together",
            "claim_limit": "camera trajectory, MANO shape, object geometry, object pose, contact, and full nonlinear ownership remain outside this diagnostic",
        },
        "parameters": {
            "max_pairs_per_row": int(args.max_pairs_per_row),
            "max_depth_observation_samples_per_row": int(args.max_depth_observation_samples_per_row),
            "max_abs_post_temporal_delta_m": float(args.max_abs_post_temporal_delta_m),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "accept_depth_median_m": float(args.accept_depth_median_m),
            "accept_depth_p95_m": float(args.accept_depth_p95_m),
            "accept_joint_median_px": float(args.accept_joint_median_px),
            "accept_joint_p95_px": float(args.accept_joint_p95_px),
        },
        "scalar_rows": scalar_rows,
        "geometry_rows": geometry_rows,
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_coupled_hand_depth_mano_observation_graph.json", report)
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
    cases = ["trash_1050", "task5_tomato_960"]
    reports = [case_problem(case, model, args, device) for case in cases]
    rows = [
        require_dict(row, "coupled graph row")
        for report in reports
        for row in require_list(report.get("rows"), "coupled rows")
    ]
    geometry_rows = [
        require_dict(row, "coupled geometry row")
        for report in reports
        for row in require_list(report.get("geometry_rows"), "geometry rows")
    ]
    summary = {
        "method": "solve_v17_coupled_hand_depth_mano_observation_graph",
        "status": STATUS,
        "claim": CLAIM,
        "wilor_root": str(args.wilor_root),
        "wilor_mano_right": str(mano_model_path),
        "device": str(device),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "coupled_variable_rows": require_int(report.get("coupled_variable_rows"), "variables"),
                "coupled_geometry_pose_variable_rows": require_int(
                    report.get("coupled_geometry_pose_variable_rows"),
                    "geometry rows",
                ),
                "coupled_depth_observation_factor_rows": require_int(
                    report.get("coupled_depth_observation_factor_rows"),
                    "observation rows",
                ),
                "coupled_reprojected_metric_depth_compatible_rows": require_int(
                    report.get("coupled_reprojected_metric_depth_compatible_rows"),
                    "compatible rows",
                ),
                "metric_hand_state_accepted_rows_after_coupled_graph": require_int(
                    report.get("metric_hand_state_accepted_rows_after_coupled_graph"),
                    "accepted rows",
                ),
                "depth_repair_factor_candidate_rows_after_coupled_graph": require_int(
                    report.get("depth_repair_factor_candidate_rows_after_coupled_graph"),
                    "residual rows",
                ),
                "coupled_temporal_reprojection_state_counts": require_dict(
                    report.get("coupled_temporal_reprojection_state_counts"),
                    "temporal state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "frame_count": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "coupled_variable_rows": sum(
            require_int(report.get("coupled_variable_rows"), "variables") for report in reports
        ),
        "coupled_geometry_pose_variable_rows": sum(
            require_int(report.get("coupled_geometry_pose_variable_rows"), "geometry variables")
            for report in reports
        ),
        "coupled_depth_observation_factor_rows": sum(
            require_int(report.get("coupled_depth_observation_factor_rows"), "observation rows")
            for report in reports
        ),
        "coupled_compatible_anchor_rows": sum(
            require_int(report.get("coupled_compatible_anchor_rows"), "anchors") for report in reports
        ),
        "coupled_scalar_delta_bound_hit_rows": sum(
            require_int(report.get("coupled_scalar_delta_bound_hit_rows"), "bound rows")
            for report in reports
        ),
        "coupled_fixed_factor_depth_improved_rows": sum(
            require_int(report.get("coupled_fixed_factor_depth_improved_rows"), "fixed improved")
            for report in reports
        ),
        "coupled_fixed_factor_depth_threshold_met_rows": sum(
            require_int(report.get("coupled_fixed_factor_depth_threshold_met_rows"), "fixed threshold")
            for report in reports
        ),
        "coupled_geometry_depth_improved_rows": bool_count(geometry_rows, "coupled_geometry_depth_improved"),
        "coupled_geometry_depth_threshold_met_rows": bool_count(
            geometry_rows,
            "coupled_geometry_depth_threshold_met",
        ),
        "coupled_geometry_projection_trusted_rows": bool_count(
            geometry_rows,
            "coupled_geometry_projection_trusted",
        ),
        "coupled_geometry_pose_delta_clamp_hit_rows": sum(
            require_int(report.get("coupled_geometry_pose_delta_clamp_hit_rows"), "pose clamp rows")
            for report in reports
        ),
        "coupled_reprojected_metric_depth_compatible_rows": sum(
            require_int(report.get("coupled_reprojected_metric_depth_compatible_rows"), "compatible rows")
            for report in reports
        ),
        "coupled_reprojected_depth_improved_rows": sum(
            require_int(report.get("coupled_reprojected_depth_improved_rows"), "improved rows")
            for report in reports
        ),
        "metric_hand_state_accepted_rows_after_coupled_graph": bool_count(rows, "metric_depth_compatible"),
        "depth_repair_factor_candidate_rows_after_coupled_graph": bool_count(
            rows,
            "depth_repair_factor_candidate",
        ),
        "coupled_reprojection_residual_owner_rows": sum(
            require_int(report.get("coupled_reprojection_residual_owner_rows"), "residual owners")
            for report in reports
        ),
        "coupled_reprojection_local_surface_factor_candidate_rows": sum(
            require_int(report.get("coupled_reprojection_local_surface_factor_candidate_rows"), "local rows")
            for report in reports
        ),
        "coupled_reprojection_mixed_surface_depth_owner_rows": sum(
            require_int(report.get("coupled_reprojection_mixed_surface_depth_owner_rows"), "mixed rows")
            for report in reports
        ),
        "coupled_reprojection_depth_observation_owner_rows": sum(
            require_int(report.get("coupled_reprojection_depth_observation_owner_rows"), "depth rows")
            for report in reports
        ),
        "coupled_reprojection_projection_untrusted_rows": sum(
            require_int(report.get("coupled_reprojection_projection_untrusted_rows"), "projection rows")
            for report in reports
        ),
        "coupled_temporal_reprojection_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(
                                report.get("coupled_temporal_reprojection_state_counts"),
                                "temporal counts",
                            )
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "coupled_geometry_solve_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(
                            require_dict(report.get("coupled_geometry_solve_state_counts"), "geometry counts")
                        )
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "geometry_before_depth_abs_median_m": numeric_summary(geometry_rows, "before.depth_abs_median_m"),
        "geometry_after_depth_abs_median_m": numeric_summary(geometry_rows, "after.depth_abs_median_m"),
        "pose_delta_abs_max_rad": numeric_summary(geometry_rows, "pose_delta_abs_max_rad"),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_coupled_hand_depth_mano_observation_graph_summary.json", summary)
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
        "--hand-temporal-owner-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit"),
    )
    parser.add_argument(
        "--post-temporal-mano-factor-input-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input"),
    )
    parser.add_argument(
        "--post-temporal-depth-observation-support-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state"),
    )
    parser.add_argument(
        "--post-temporal-depth-observation-weighted-refit-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_coupled_hand_depth_mano_observation_graph"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.015)
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
    parser.add_argument("--max-abs-post-temporal-delta-m", type=float, default=0.12)
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
    parser.add_argument("--accept-depth-median-m", type=float, default=0.030)
    parser.add_argument("--accept-depth-p95-m", type=float, default=0.080)
    parser.add_argument("--accept-joint-median-px", type=float, default=45.0)
    parser.add_argument("--accept-joint-p95-px", type=float, default=95.0)
    parser.add_argument("--min-depth-median-improvement-m", type=float, default=0.005)
    parser.add_argument("--solver-tol", type=float, default=1e-8)
    parser.add_argument("--max-solver-iter", type=int, default=500)
    parser.add_argument("--bound-tolerance-m", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
