#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import inspect
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_v17_hand_intrinsics_depth_counterfactual import (
    annotation_hand_index,
    local_hand_geometry,
    scale_depth_intrinsics,
    solve_translation,
    source_intrinsics,
    source_size_from_intrinsics,
)
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
from refit_mano_pose_contact_v3 import (
    apply_side_sign,
    hand_span_torch,
    robust_l1,
    rotvec_to_matrix,
    side_sign,
    similarity_from_to,
)


STATUS = "v17_mano_articulation_local_solve_qc"
CLAIM = (
    "This artifact tests a local MANO articulation mechanism for V17 hand-depth residuals. It consumes "
    "the materialized MANO articulation factor inputs, keeps the solved hand-depth graph scale and "
    "camera-ray shift fixed, optimizes small MANO pose deltas against residual-to-compatible-seed "
    "surface pairs, and reports whether local articulation reduces the remaining residual. It does not "
    "write corrected annotations or complete the V3 joint solver."
)


def patch_legacy_mano_loader() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if name not in np.__dict__:
            setattr(np, name, value)


def load_wilor_mano_class(wilor_root: Path) -> Any:
    path = wilor_root / "wilor" / "models" / "mano_wrapper.py"
    if not path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO wrapper: {path}")
    spec = importlib.util.spec_from_file_location("wilor_mano_wrapper_for_v17_articulation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WiLoR MANO wrapper spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO


def state_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(require_str(row.get(key), key) for row in rows).items()))


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return summarize(values)


def finite_number(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def array2d(value: Any, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise RuntimeError(f"{name} must be an Nx{width} array")
    if np.any(~np.isfinite(arr)):
        raise RuntimeError(f"{name} contains nonfinite values")
    return arr


def mano_param_tensors(hand: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    params = require_dict(hand.get("mano_params"), "mano_params")
    global_orient = np.asarray(params.get("global_orient"), dtype=np.float32)
    hand_pose = np.asarray(params.get("hand_pose"), dtype=np.float32)
    betas = np.asarray(params.get("betas"), dtype=np.float32)
    if global_orient.shape != (1, 3, 3):
        raise RuntimeError("global_orient must be 1x3x3 rotation matrix")
    if hand_pose.shape != (15, 3, 3):
        raise RuntimeError("hand_pose must be 15x3x3 rotation matrices")
    if betas.shape != (10,):
        raise RuntimeError("betas must contain 10 shape coefficients")
    return (
        torch.tensor(global_orient[None], dtype=torch.float32, device=device),
        torch.tensor(hand_pose[None], dtype=torch.float32, device=device),
        torch.tensor(betas[None], dtype=torch.float32, device=device),
    )


def project_torch(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points[..., 2].clamp_min(1.0e-6)
    fx, fy, cx, cy = intrinsics
    return torch.stack([fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy], dim=-1)


def project_depth_torch(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    projection_source_size: tuple[float, float],
    depth_shape: tuple[int, int],
) -> torch.Tensor:
    uv = project_torch(points, intrinsics)
    depth_h, depth_w = depth_shape
    scale = torch.tensor(
        [
            float(depth_w) / float(projection_source_size[0]),
            float(depth_h) / float(projection_source_size[1]),
        ],
        dtype=uv.dtype,
        device=uv.device,
    )
    return uv * scale


def corrected_replayed_state(
    *,
    model: Any,
    hand: dict[str, Any],
    graph_row: dict[str, Any],
    depth: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    frame_idx = require_int(graph_row.get("frame_idx"), "graph frame_idx")
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        raise RuntimeError(f"missing depth frame for {frame_idx}")
    hand_intrinsics = source_intrinsics(hand)
    if hand_intrinsics is None:
        raise RuntimeError(f"missing source intrinsics for frame {frame_idx}")
    geometry = local_hand_geometry(hand)
    if geometry is None:
        raise RuntimeError(f"missing local hand geometry for frame {frame_idx}")
    local_joints_np, local_vertices_np, keypoints2d_np = geometry
    depth_intrinsics = np.asarray(depth["intrinsics"][int(depth_i)], dtype=np.float64)
    projection_source_size = source_size_from_intrinsics(hand_intrinsics)
    intrinsics_np = scale_depth_intrinsics(
        depth_intrinsics,
        depth["source_size"],
        projection_source_size,
    )
    translation = solve_translation(local_joints_np, keypoints2d_np, intrinsics_np)
    source_joints_np = local_joints_np + translation[None, :]
    source_vertices_np = local_vertices_np + translation[None, :]
    center = np.median(source_joints_np, axis=0)
    if not np.all(np.isfinite(center)) or float(center[2]) <= 1e-6:
        raise RuntimeError(f"invalid source hand center for frame {frame_idx}")
    center_ray_np = center / float(center[2])
    solved_scale = finite_number(graph_row.get("solved_scale"), "solved_scale")
    ray_shift = finite_number(graph_row.get("hand_ray_shift_m"), "hand_ray_shift_m")
    global_orient, hand_pose, betas = mano_param_tensors(hand, device)
    sign = side_sign(require_str(hand.get("side"), "hand side"))
    with torch.no_grad():
        base_out = model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            return_verts=True,
            pose2rot=False,
        )
        base_vertices = apply_side_sign(base_out.vertices, sign)[0].detach().cpu().numpy()
        base_joints = apply_side_sign(base_out.joints, sign)[0].detach().cpu().numpy()
    local_scale, local_rotation, local_translation, vertex_error = similarity_from_to(
        base_vertices,
        local_vertices_np,
    )
    aligned_joints = local_scale * (base_joints @ local_rotation.T) + local_translation[None, :]
    joint_error = np.linalg.norm(aligned_joints - local_joints_np, axis=1)
    return {
        "global_orient": global_orient,
        "hand_pose": hand_pose,
        "betas": betas,
        "sign": sign,
        "local_scale": torch.tensor(float(local_scale), dtype=torch.float32, device=device),
        "local_rotation": torch.tensor(local_rotation, dtype=torch.float32, device=device),
        "local_translation": torch.tensor(local_translation, dtype=torch.float32, device=device).reshape(1, 1, 3),
        "graph_scale": torch.tensor(float(solved_scale), dtype=torch.float32, device=device),
        "ray_shift": torch.tensor(float(ray_shift), dtype=torch.float32, device=device),
        "center_ray": torch.tensor(center_ray_np, dtype=torch.float32, device=device).reshape(1, 1, 3),
        "translation": torch.tensor(translation, dtype=torch.float32, device=device).reshape(1, 1, 3),
        "intrinsics": torch.tensor(intrinsics_np, dtype=torch.float32, device=device),
        "projection_source_size": tuple(float(value) for value in projection_source_size),
        "depth_shape": (int(depth["depth"][int(depth_i)].shape[0]), int(depth["depth"][int(depth_i)].shape[1])),
        "keypoints2d": torch.tensor(keypoints2d_np, dtype=torch.float32, device=device),
        "source_joints_np": source_joints_np,
        "source_vertices_np": source_vertices_np,
        "zero_state_vertex_error_median_m": float(np.median(vertex_error)),
        "zero_state_vertex_error_p95_m": float(np.percentile(vertex_error, 95.0)),
        "zero_state_joint_error_median_m": float(np.median(joint_error)),
        "zero_state_joint_error_p95_m": float(np.percentile(joint_error, 95.0)),
    }


def replay_corrected_vertices(
    *,
    model: Any,
    state: dict[str, Any],
    pose_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_orient = state["global_orient"]
    hand_pose = rotvec_to_matrix(pose_delta) @ state["hand_pose"]
    out = model(
        global_orient=global_orient,
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
    corrected_vertices = (
        state["graph_scale"] * source_vertices
        + state["ray_shift"] * state["center_ray"]
    )
    corrected_joints = (
        state["graph_scale"] * source_joints
        + state["ray_shift"] * state["center_ray"]
    )
    return corrected_vertices, corrected_joints, local_vertices


def factor_arrays(row: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    arrays = require_dict(require_dict(row.get("assignment"), "assignment").get("factor_pair_arrays"), "factor pairs")
    residual_vertex = np.asarray(arrays.get("residual_vertex_id"), dtype=np.int64)
    seed_vertex = np.asarray(arrays.get("seed_vertex_id"), dtype=np.int64)
    if residual_vertex.ndim != 1 or seed_vertex.ndim != 1 or len(residual_vertex) != len(seed_vertex):
        raise RuntimeError("factor vertex arrays must be same-length vectors")
    if len(residual_vertex) == 0:
        raise RuntimeError("local articulation solve requires assigned factor pairs")
    take = np.arange(len(residual_vertex), dtype=np.int64)
    if len(take) > int(args.max_pairs_per_row):
        take = np.linspace(0, len(residual_vertex) - 1, int(args.max_pairs_per_row), dtype=np.int64)
    residual_xy = np.stack(
        [
            np.asarray(arrays.get("residual_x"), dtype=np.float32)[take],
            np.asarray(arrays.get("residual_y"), dtype=np.float32)[take],
        ],
        axis=1,
    )
    seed_xy = np.stack(
        [
            np.asarray(arrays.get("seed_x"), dtype=np.float32)[take],
            np.asarray(arrays.get("seed_y"), dtype=np.float32)[take],
        ],
        axis=1,
    )
    return {
        "residual_vertex_id": torch.tensor(residual_vertex[take], dtype=torch.long, device=device),
        "seed_vertex_id": torch.tensor(seed_vertex[take], dtype=torch.long, device=device),
        "target_depth": torch.tensor(
            np.asarray(arrays.get("seed_metric_depth_m"), dtype=np.float32)[take],
            dtype=torch.float32,
            device=device,
        ),
        "target_xy": torch.tensor(seed_xy, dtype=torch.float32, device=device),
        "residual_xy": torch.tensor(residual_xy, dtype=torch.float32, device=device),
        "sample_count": torch.tensor(int(len(take)), dtype=torch.int32, device=device),
    }


def eval_metrics(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    state: dict[str, Any],
    factors: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    residual_vertices = vertices[0, factors["residual_vertex_id"]]
    depth_gap = residual_vertices[:, 2] - factors["target_depth"]
    uv = project_depth_torch(
        residual_vertices,
        state["intrinsics"],
        state["projection_source_size"],
        state["depth_shape"],
    )
    source_uv = project_depth_torch(
        vertices[0, factors["seed_vertex_id"]],
        state["intrinsics"],
        state["projection_source_size"],
        state["depth_shape"],
    )
    projection_to_seed = uv - factors["target_xy"]
    projection_to_source_seed = uv - source_uv
    joint_uv = project_torch(joints[0], state["intrinsics"])
    joint_residual = torch.linalg.norm(joint_uv - state["keypoints2d"], dim=1)
    depth_gap_np = depth_gap.detach().cpu().numpy()
    projection_np = torch.linalg.norm(projection_to_seed, dim=1).detach().cpu().numpy()
    projection_seed_np = torch.linalg.norm(projection_to_source_seed, dim=1).detach().cpu().numpy()
    joint_np = joint_residual.detach().cpu().numpy()
    span = hand_span_torch(joints).detach().cpu()
    return {
        "factor_sample_count": int(factors["sample_count"].detach().cpu()),
        "residual_vertex_minus_seed_metric_depth_m": summarize(depth_gap_np.astype(float).tolist()),
        "residual_projection_to_seed_pixel_px": summarize(projection_np.astype(float).tolist()),
        "residual_projection_to_source_seed_vertex_px": summarize(projection_seed_np.astype(float).tolist()),
        "joint_reprojection_px": summarize(joint_np.astype(float).tolist()),
        "hand_span_m": float(span),
        "depth_abs_median_m": float(np.median(np.abs(depth_gap_np))),
        "depth_abs_p95_m": float(np.percentile(np.abs(depth_gap_np), 95.0)),
        "projection_to_seed_median_px": float(np.median(projection_np)),
        "joint_reprojection_median_px": float(np.median(joint_np)),
        "joint_reprojection_p95_px": float(np.percentile(joint_np, 95.0)),
    }


def solve_row(
    *,
    model: Any,
    hand: dict[str, Any],
    graph_row: dict[str, Any],
    factor_row: dict[str, Any],
    depth: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    state = corrected_replayed_state(model=model, hand=hand, graph_row=graph_row, depth=depth, device=device)
    factors = factor_arrays(factor_row, args, device)
    pose_delta = torch.zeros((1, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pose_delta], lr=float(args.lr))
    with torch.no_grad():
        base_vertices, base_joints, _ = replay_corrected_vertices(model=model, state=state, pose_delta=pose_delta)
        before = eval_metrics(base_vertices, base_joints, state, factors, args)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)
        vertices, joints, _ = replay_corrected_vertices(model=model, state=state, pose_delta=pose_delta)
        residual_vertices = vertices[0, factors["residual_vertex_id"]]
        depth_loss = robust_l1(
            (residual_vertices[:, 2] - factors["target_depth"]) / float(args.sigma_depth_m)
        ).mean()
        uv = project_depth_torch(
            residual_vertices,
            state["intrinsics"],
            state["projection_source_size"],
            state["depth_shape"],
        )
        projection_loss = robust_l1((uv - factors["target_xy"]) / float(args.sigma_projection_px)).mean()
        joint_uv = project_torch(joints[0], state["intrinsics"])
        joint_loss = robust_l1((joint_uv - state["keypoints2d"]) / float(args.sigma_joint_px)).mean()
        pose_loss = robust_l1(pose_delta / float(args.sigma_pose_delta_rad)).mean()
        span = hand_span_torch(joints)
        span_loss = robust_l1(torch.relu(float(args.min_span_m) - span) / float(args.sigma_span_m)) + robust_l1(
            torch.relu(span - float(args.max_span_m)) / float(args.sigma_span_m)
        )
        loss = (
            float(args.w_depth) * depth_loss
            + float(args.w_projection) * projection_loss
            + float(args.w_joint) * joint_loss
            + float(args.w_pose) * pose_loss
            + float(args.w_span) * span_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pose_delta.clamp_(-float(args.max_pose_delta_rad), float(args.max_pose_delta_rad))
            if float(loss.detach().cpu()) < best_loss:
                best_loss = float(loss.detach().cpu())
                best_state = {
                    "pose_delta": pose_delta.detach().clone(),
                    "vertices": vertices.detach().clone(),
                    "joints": joints.detach().clone(),
                    "depth_loss": depth_loss.detach().clone(),
                    "projection_loss": projection_loss.detach().clone(),
                    "joint_loss": joint_loss.detach().clone(),
                    "pose_loss": pose_loss.detach().clone(),
                    "span_loss": span_loss.detach().clone(),
                }
    if best_state is None:
        raise RuntimeError("MANO local articulation optimizer produced no state")
    after = eval_metrics(best_state["vertices"], best_state["joints"], state, factors, args)
    depth_improvement = before["depth_abs_median_m"] - after["depth_abs_median_m"]
    projection_change = after["projection_to_seed_median_px"] - before["projection_to_seed_median_px"]
    joint_ok = bool(
        after["joint_reprojection_median_px"] <= float(args.accept_joint_median_px)
        and after["joint_reprojection_p95_px"] <= float(args.accept_joint_p95_px)
    )
    depth_ok = bool(
        after["depth_abs_median_m"] <= float(args.accept_depth_median_m)
        and after["depth_abs_p95_m"] <= float(args.accept_depth_p95_m)
    )
    improved = bool(depth_improvement >= float(args.min_depth_median_improvement_m))
    state_name = "local_articulation_solve_reduces_depth_residual" if improved else "local_articulation_solve_no_depth_gain"
    if improved and not joint_ok:
        state_name = "local_articulation_depth_gain_projection_untrusted"
    if improved and joint_ok and depth_ok:
        state_name = "local_articulation_factor_solved_under_local_thresholds"
    return {
        "local_articulation_solve_state": state_name,
        "local_articulation_depth_improved": improved,
        "local_articulation_depth_threshold_met": depth_ok,
        "local_articulation_projection_trusted": joint_ok,
        "before": before,
        "after": after,
        "depth_abs_median_improvement_m": float(depth_improvement),
        "projection_to_seed_median_change_px": float(projection_change),
        "loss": best_loss,
        "loss_terms": {
            "depth_loss": float(best_state["depth_loss"].detach().cpu()),
            "projection_loss": float(best_state["projection_loss"].detach().cpu()),
            "joint_loss": float(best_state["joint_loss"].detach().cpu()),
            "pose_loss": float(best_state["pose_loss"].detach().cpu()),
            "span_loss": float(best_state["span_loss"].detach().cpu()),
        },
        "pose_delta_abs_max_rad": float(torch.max(torch.abs(best_state["pose_delta"])).detach().cpu()),
        "zero_state_vertex_error_median_m": state["zero_state_vertex_error_median_m"],
        "zero_state_vertex_error_p95_m": state["zero_state_vertex_error_p95_m"],
        "zero_state_joint_error_median_m": state["zero_state_joint_error_median_m"],
        "zero_state_joint_error_p95_m": state["zero_state_joint_error_p95_m"],
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
        "hand_depth_repair_graph": existing_path(
            args.hand_depth_repair_graph_root / case / "v17_hand_depth_repair_graph.json",
            f"{case} hand depth repair graph",
        ),
        "mano_articulation_factor_input": existing_path(
            args.mano_articulation_factor_input_root / case / "v17_mano_articulation_factor_input.json",
            f"{case} MANO articulation factor input",
        ),
    }
    payloads = {name: require_dict(load_json(path), f"{case} {name}") for name, path in paths.items()}
    frames = annotation_frames(payloads["annotations"])
    hands = annotation_hand_index(frames)
    visible_surface = payloads["visible_surface"]
    depth_path = existing_path(
        Path(require_str(visible_surface.get("metric_depth_npz"), "metric_depth_npz")),
        "metric depth npz",
    )
    depth = depth_archive(depth_path)
    graph_report = payloads["hand_depth_repair_graph"]
    factor_report = payloads["mano_articulation_factor_input"]
    frame_count = len(frames)
    for name, report in [
        ("hand depth repair graph", graph_report),
        ("MANO articulation factor input", factor_report),
    ]:
        if frame_count != require_int(report.get("frame_count"), f"{case} {name} frame_count"):
            raise RuntimeError(f"{case} frame count disagrees with {name}")
    graph_by_id = {
        require_str(row.get("hand_depth_repair_graph_variable_id"), "graph id"): row
        for row in [require_dict(raw, "graph row") for raw in require_list(graph_report.get("rows"), "graph rows")]
    }
    rows: list[dict[str, Any]] = []
    for raw in require_list(factor_report.get("rows"), f"{case} factor input rows"):
        factor_row = require_dict(raw, "factor input row")
        graph_id = require_str(
            factor_row.get("source_hand_depth_repair_graph_variable_id"),
            "factor source graph id",
        )
        graph_row = require_dict(graph_by_id.get(graph_id), f"{case} graph row {graph_id}")
        frame_idx = require_int(factor_row.get("frame_idx"), "factor frame_idx")
        side = require_str(factor_row.get("hand_side"), "factor hand_side")
        hand_index = require_int(factor_row.get("hand_index"), "factor hand_index")
        hand = require_dict(hands.get((frame_idx, side, hand_index)), f"{case} hand {graph_id}")
        fit = solve_row(model=model, hand=hand, graph_row=graph_row, factor_row=factor_row, depth=depth, args=args, device=device)
        rows.append(
            {
                "case": case,
                "mano_local_articulation_solve_variable_id": graph_id.replace(
                    "hand_depth_repair_graph:",
                    "mano_local_articulation_solve:",
                    1,
                ),
                "source_hand_depth_repair_graph_variable_id": graph_id,
                "source_mano_articulation_factor_input_id": factor_row.get(
                    "mano_articulation_factor_input_id"
                ),
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_index": hand_index,
                **fit,
                **FALSE_READY,
            }
        )
    solved_rows = [row for row in rows if row.get("local_articulation_depth_improved") is True]
    threshold_rows = [row for row in rows if row.get("local_articulation_depth_threshold_met") is True]
    trusted_rows = [row for row in rows if row.get("local_articulation_projection_trusted") is True]
    clamp_hit_rows = [
        row
        for row in rows
        if float(row.get("pose_delta_abs_max_rad", 0.0)) >= float(args.max_pose_delta_rad) - 1.0e-5
    ]
    report = {
        "method": "solve_v17_mano_articulation_local",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {name: source_summary(path, payloads[name]) for name, path in paths.items()},
        "metric_depth_npz": str(depth_path),
        "frame_count": frame_count,
        "mano_local_articulation_solve_candidate_rows": len(rows),
        "local_articulation_depth_improved_rows": len(solved_rows),
        "local_articulation_depth_threshold_met_rows": len(threshold_rows),
        "local_articulation_projection_trusted_rows": len(trusted_rows),
        "local_articulation_pose_delta_clamp_hit_rows": len(clamp_hit_rows),
        "local_articulation_solve_state_counts": state_counts(rows, "local_articulation_solve_state"),
        "before_depth_abs_median_m": numeric_summary(rows, "before.depth_abs_median_m"),
        "after_depth_abs_median_m": numeric_summary(rows, "after.depth_abs_median_m"),
        "depth_abs_median_improvement_m": numeric_summary(rows, "depth_abs_median_improvement_m"),
        "after_joint_reprojection_median_px": numeric_summary(rows, "after.joint_reprojection_median_px"),
        "after_joint_reprojection_p95_px": numeric_summary(rows, "after.joint_reprojection_p95_px"),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        "source_mano_articulation_factor_input_comparison": {
            "mano_articulation_factor_input_candidate_rows": factor_report.get(
                "mano_articulation_factor_input_candidate_rows"
            ),
            "mano_articulation_factor_input_materialized_rows": factor_report.get(
                "mano_articulation_factor_input_materialized_rows"
            ),
            "assigned_factor_sample_count": factor_report.get("assigned_factor_sample_count"),
        },
        "problem_semantics": {
            "optimized_variables": "per-row MANO hand_pose rotation deltas only",
            "fixed_state": "saved MANO shape, global orientation, graph scale, graph ray shift, and graph source-camera translation",
            "local_factor": "residual surface vertices are pulled toward compatible-depth seed pixels from the same repaired hand surface",
            "solver_scope": "local diagnostic solve; no full-timeline temporal coupling and no annotation update",
        },
        "parameters": {
            "device": str(device),
            "iters": int(args.iters),
            "lr": float(args.lr),
            "max_pairs_per_row": int(args.max_pairs_per_row),
            "sigma_depth_m": float(args.sigma_depth_m),
            "sigma_projection_px": float(args.sigma_projection_px),
            "sigma_joint_px": float(args.sigma_joint_px),
            "sigma_pose_delta_rad": float(args.sigma_pose_delta_rad),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "accept_depth_median_m": float(args.accept_depth_median_m),
            "accept_depth_p95_m": float(args.accept_depth_p95_m),
            "accept_joint_median_px": float(args.accept_joint_median_px),
            "accept_joint_p95_px": float(args.accept_joint_p95_px),
        },
        "rows": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_mano_articulation_local_solve.json", report)
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
    summary_path = existing_path(
        args.mano_articulation_factor_input_root / "v17_mano_articulation_factor_input_summary.json",
        "MANO articulation factor input summary",
    )
    factor_summary = require_dict(load_json(summary_path), "MANO articulation factor input summary")
    reports = [
        case_problem(require_str(require_dict(raw, f"summary case {i}").get("case"), "case"), model, args, device)
        for i, raw in enumerate(require_list(factor_summary.get("cases"), "summary cases"))
    ]
    rows = [
        require_dict(row, "local articulation row")
        for report in reports
        for row in require_list(report.get("rows"), "local articulation rows")
    ]
    payload = {
        "method": "solve_v17_mano_articulation_local",
        "status": STATUS,
        "claim": CLAIM,
        "source_mano_articulation_factor_input_summary": str(summary_path),
        "wilor_root": str(args.wilor_root),
        "wilor_mano_right": str(mano_model_path),
        "device": str(device),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_mano_articulation_local_solve.json"
                ),
                "frame_count": require_int(report.get("frame_count"), "frame_count"),
                "mano_local_articulation_solve_candidate_rows": require_int(
                    report.get("mano_local_articulation_solve_candidate_rows"),
                    "candidate rows",
                ),
                "local_articulation_depth_improved_rows": require_int(
                    report.get("local_articulation_depth_improved_rows"),
                    "improved rows",
                ),
                "local_articulation_depth_threshold_met_rows": require_int(
                    report.get("local_articulation_depth_threshold_met_rows"),
                    "threshold rows",
                ),
                "local_articulation_projection_trusted_rows": require_int(
                    report.get("local_articulation_projection_trusted_rows"),
                    "projection trusted rows",
                ),
                "local_articulation_pose_delta_clamp_hit_rows": require_int(
                    report.get("local_articulation_pose_delta_clamp_hit_rows"),
                    "pose clamp hit rows",
                ),
                "local_articulation_solve_state_counts": require_dict(
                    report.get("local_articulation_solve_state_counts"),
                    "state counts",
                ),
                **FALSE_READY,
            }
            for report in reports
        ],
        "mano_local_articulation_solve_candidate_rows": len(rows),
        "local_articulation_depth_improved_rows": bool_count(rows, "local_articulation_depth_improved"),
        "local_articulation_depth_threshold_met_rows": bool_count(rows, "local_articulation_depth_threshold_met"),
        "local_articulation_projection_trusted_rows": bool_count(rows, "local_articulation_projection_trusted"),
        "local_articulation_pose_delta_clamp_hit_rows": sum(
            1
            for row in rows
            if float(row.get("pose_delta_abs_max_rad", 0.0)) >= float(args.max_pose_delta_rad) - 1.0e-5
        ),
        "local_articulation_solve_state_counts": dict(
            sorted(
                sum(
                    (
                        Counter(require_dict(report.get("local_articulation_solve_state_counts"), "state counts"))
                        for report in reports
                    ),
                    Counter(),
                ).items()
            )
        ),
        "before_depth_abs_median_m": numeric_summary(rows, "before.depth_abs_median_m"),
        "after_depth_abs_median_m": numeric_summary(rows, "after.depth_abs_median_m"),
        "depth_abs_median_improvement_m": numeric_summary(rows, "depth_abs_median_improvement_m"),
        "after_joint_reprojection_median_px": numeric_summary(rows, "after.joint_reprojection_median_px"),
        "after_joint_reprojection_p95_px": numeric_summary(rows, "after.joint_reprojection_p95_px"),
        "pose_delta_abs_max_rad": numeric_summary(rows, "pose_delta_abs_max_rad"),
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_mano_articulation_local_solve_summary.json", payload)
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
        "--hand-depth-repair-graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_hand_depth_repair_graph"),
    )
    parser.add_argument(
        "--mano-articulation-factor-input-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_articulation_factor_input"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_mano_articulation_local_solve"),
    )
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max-pairs-per-row", type=int, default=96)
    parser.add_argument("--sigma-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-projection-px", type=float, default=6.0)
    parser.add_argument("--sigma-joint-px", type=float, default=18.0)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.18)
    parser.add_argument("--sigma-span-m", type=float, default=0.02)
    parser.add_argument("--w-depth", type=float, default=2.0)
    parser.add_argument("--w-projection", type=float, default=1.0)
    parser.add_argument("--w-joint", type=float, default=0.6)
    parser.add_argument("--w-pose", type=float, default=0.25)
    parser.add_argument("--w-span", type=float, default=0.25)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.35)
    parser.add_argument("--min-span-m", type=float, default=0.10)
    parser.add_argument("--max-span-m", type=float, default=0.22)
    parser.add_argument("--accept-depth-median-m", type=float, default=0.030)
    parser.add_argument("--accept-depth-p95-m", type=float, default=0.080)
    parser.add_argument("--accept-joint-median-px", type=float, default=45.0)
    parser.add_argument("--accept-joint-p95-px", type=float, default=95.0)
    parser.add_argument("--min-depth-median-improvement-m", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
