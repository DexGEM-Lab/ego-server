#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build interval-level rigid-SE(3) MANO pose hypotheses or uncertainty.

This is the next physical mechanism after translation-only interval corrections.
For each interaction interval, it optimizes a small per-frame rigid transform of
the current V18 bridge MANO surface: translation plus a world-frame rotation
vector about the current hand center. The MANO articulation/shape is preserved;
only global hand pose is perturbed. This tests whether a wrist/global-pose error,
rather than pure translation, can explain object nonpenetration conflicts while
preserving visible 2D compatibility.

The output is conservative: optimized transforms are hypotheses, not accepted
coordinate corrections, unless residual penetration, 2D shift, temporal coherence,
and hidden-volume validity all support acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_mano_object_constraint_state import frame_intrinsics, project  # noqa: E402
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    DEFAULT_ANNOTATIONS,
    DEFAULT_CASE,
    DEFAULT_COMPLETION_REPORT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_REPORT,
    DEFAULT_SIGN_MESH,
    as_list,
    frame_camera_pose,
    inverse_object,
    load_json,
    load_mesh,
    numeric_summary,
    object_vec_to_world,
    pose_map,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    parser.add_argument("--completion-report", type=Path, default=DEFAULT_COMPLETION_REPORT)
    parser.add_argument("--sign-mesh", type=Path, default=DEFAULT_SIGN_MESH)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--object-id", default="object:obj_tomato")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-constraints-per-frame", type=int, default=96)
    parser.add_argument("--penetration-epsilon-m", type=float, default=1.0e-5)
    parser.add_argument("--max-translation-m", type=float, default=0.035)
    parser.add_argument("--max-rotation-rad", type=float, default=0.20)
    parser.add_argument("--accepted-residual-m", type=float, default=0.0015)
    parser.add_argument("--visible-shift-limit-px", type=float, default=8.0)
    parser.add_argument("--translation-prior-weight", type=float, default=1.0e4)
    parser.add_argument("--rotation-prior-weight", type=float, default=2.0e3)
    parser.add_argument("--smooth-weight", type=float, default=5.0e4)
    parser.add_argument("--accel-weight", type=float, default=1.0e5)
    parser.add_argument("--penetration-weight", type=float, default=1.0e6)
    parser.add_argument("--max-optimizer-iterations", type=int, default=280)
    return parser.parse_args()


def load_bridge_array(cache: dict[Path, Any], bridge_path: Path, array_name: str, row_index: int) -> np.ndarray:
    if bridge_path not in cache:
        cache[bridge_path] = np.load(bridge_path, allow_pickle=True)
    return np.asarray(cache[bridge_path][array_name][row_index], dtype=float)


def bridge_vertices_and_joints(hand: dict[str, Any], bridge_cache: dict[Path, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    metric_raw = hand.get("metric_mano_state")
    metric: dict[str, Any] = metric_raw if isinstance(metric_raw, dict) else {}
    reference_raw = metric.get("vertices_reference")
    reference: dict[str, Any] = reference_raw if isinstance(reference_raw, dict) else {}
    bridge_path_raw = reference.get("bridge_npz")
    vertices_array = reference.get("bridge_vertices_world_array")
    row_index_raw = reference.get("bridge_row_index")
    if not isinstance(bridge_path_raw, str) or not isinstance(vertices_array, str) or row_index_raw is None:
        return None
    bridge_path = Path(bridge_path_raw)
    if not bridge_path.exists():
        return None
    row_index = int(row_index_raw)
    vertices_world = load_bridge_array(bridge_cache, bridge_path, vertices_array, row_index)
    joints_array = "joints_current_v18_world_from_hawor_projection_relift_m"
    joints_world = load_bridge_array(bridge_cache, bridge_path, joints_array, row_index)
    return vertices_world, joints_world


def project_se3_shift_px(
    frame: dict[str, Any],
    side: str,
    joints_world: np.ndarray,
    center_world: np.ndarray,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> dict[str, Any]:
    intr = frame_intrinsics(frame, side)
    if intr is None:
        return {"state": "missing_intrinsics", "count": 0, "max": None, "median": None}
    r_c2w, t_c2w = frame_camera_pose(frame)
    before = project(joints_world, r_c2w, t_c2w, intr)
    rel = joints_world - center_world[None, :]
    after_world = joints_world + translation_world[None, :] + np.cross(rotation_world[None, :], rel)
    after = project(after_world, r_c2w, t_c2w, intr)
    shift = np.linalg.norm(after - before, axis=1)
    return {
        "state": "evaluated",
        "count": int(len(shift)),
        "median": float(np.median(shift)),
        "p95": float(np.percentile(shift, 95)),
        "max": float(np.max(shift)),
    }


def build_frame_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    sign_mesh = load_mesh(args.sign_mesh)
    sign_scene = o3d.t.geometry.RaycastingScene()
    sign_scene.add_triangles(
        o3d.core.Tensor(np.asarray(sign_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(sign_mesh.faces, dtype=np.uint32)),
    )
    bridge_cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    frames_by_idx: dict[int, dict[str, Any]] = {}

    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame["frame_idx"])
        frames_by_idx[frame_idx] = frame
        pose = poses.get(frame_idx)
        if pose is None:
            continue
        r_obj, t_obj = pose
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            arrays = bridge_vertices_and_joints(hand, bridge_cache)
            if arrays is None:
                continue
            vertices_world, joints_world = arrays
            center_world = np.mean(joints_world, axis=0)
            vertices_object = inverse_object(vertices_world, r_obj, t_obj)
            signed = -sign_scene.compute_signed_distance(
                o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))
            ).numpy().astype(float)
            penetrating_idx = np.where(signed > float(args.penetration_epsilon_m))[0]
            depths_all = signed[penetrating_idx]
            constraint_idx = penetrating_idx
            clipped = 0
            if len(constraint_idx) > int(args.max_constraints_per_frame):
                order = np.argsort(depths_all)[::-1]
                keep = order[: int(args.max_constraints_per_frame)]
                constraint_idx = penetrating_idx[keep]
                clipped = int(len(penetrating_idx) - len(constraint_idx))
            normals_world = np.zeros((0, 3), dtype=float)
            depths = np.zeros((0,), dtype=float)
            constraint_vertices_world = np.zeros((0, 3), dtype=float)
            if len(constraint_idx) > 0:
                closest = sign_scene.compute_closest_points(
                    o3d.core.Tensor(np.asarray(vertices_object[constraint_idx], dtype=np.float32))
                )["points"].numpy().astype(float)
                disp = closest - vertices_object[constraint_idx]
                norms = np.linalg.norm(disp, axis=1)
                valid = norms > 1.0e-12
                normals_object = disp[valid] / norms[valid, None]
                normals_world = object_vec_to_world(normals_object, r_obj)
                depths = signed[constraint_idx][valid]
                constraint_vertices_world = vertices_world[constraint_idx][valid]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "constraint_count": int(len(depths)),
                    "constraint_clipped_count": clipped,
                    "penetrating_vertex_count": int(len(penetrating_idx)),
                    "penetration_depth_m": numeric_summary(depths_all),
                    "normals_world": normals_world.astype(float),
                    "depths_m": depths.astype(float),
                    "constraint_vertices_world": constraint_vertices_world.astype(float),
                    "hand_center_world": center_world.astype(float),
                    "joints_world": joints_world.astype(float),
                }
            )
    return rows, frames_by_idx


def contiguous_segments(frame_indices: list[int]) -> list[tuple[int, int]]:
    if not frame_indices:
        return []
    frames = sorted(set(frame_indices))
    segments: list[tuple[int, int]] = []
    start = end = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx <= end + 1:
            end = frame_idx
        else:
            segments.append((start, end))
            start = end = frame_idx
    segments.append((start, end))
    return segments


def row_design_matrix(row: dict[str, Any]) -> np.ndarray:
    normals = np.asarray(row["normals_world"], dtype=float)
    vertices = np.asarray(row["constraint_vertices_world"], dtype=float)
    center = np.asarray(row["hand_center_world"], dtype=float)
    if len(normals) == 0:
        return np.zeros((0, 6), dtype=float)
    rel = vertices - center[None, :]
    rot_coeff = np.cross(rel, normals)
    return np.hstack([normals, rot_coeff])


def objective_and_grad(
    flat_x: np.ndarray,
    matrices: list[np.ndarray],
    depths: list[np.ndarray],
    translation_prior_weight: float,
    rotation_prior_weight: float,
    smooth_weight: float,
    accel_weight: float,
    penetration_weight: float,
) -> tuple[float, np.ndarray]:
    x = flat_x.reshape((-1, 6))
    grad = np.zeros_like(x)
    value = 0.0

    translation = x[:, :3]
    rotation = x[:, 3:]
    value += 0.5 * translation_prior_weight * float(np.sum(translation * translation))
    value += 0.5 * rotation_prior_weight * float(np.sum(rotation * rotation))
    grad[:, :3] += translation_prior_weight * translation
    grad[:, 3:] += rotation_prior_weight * rotation

    # Adjacent frames outside the interval are assumed zero-perturbation current MANO observations.
    for endpoint in (0, len(x) - 1):
        value += 0.5 * smooth_weight * float(np.dot(x[endpoint], x[endpoint]))
        grad[endpoint] += smooth_weight * x[endpoint]

    for i in range(1, len(x)):
        diff = x[i] - x[i - 1]
        value += 0.5 * smooth_weight * float(np.dot(diff, diff))
        grad[i] += smooth_weight * diff
        grad[i - 1] -= smooth_weight * diff

    for i in range(1, len(x) - 1):
        acc = x[i + 1] - 2.0 * x[i] + x[i - 1]
        value += 0.5 * accel_weight * float(np.dot(acc, acc))
        grad[i + 1] += accel_weight * acc
        grad[i] -= 2.0 * accel_weight * acc
        grad[i - 1] += accel_weight * acc

    for i, (matrix_i, depth_i) in enumerate(zip(matrices, depths)):
        if len(depth_i) == 0:
            continue
        residual = depth_i - matrix_i @ x[i]
        active = residual > 0.0
        if not np.any(active):
            continue
        res_active = residual[active]
        matrix_active = matrix_i[active]
        scale = penetration_weight / max(1, len(depth_i))
        value += 0.5 * scale * float(np.dot(res_active, res_active))
        grad[i] -= scale * (matrix_active.T @ res_active)

    return value, grad.reshape(-1)


def optimize_segment(
    segment_rows: list[dict[str, Any]],
    frames_by_idx: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = [int(row["frame_idx"]) for row in segment_rows]
    side = str(segment_rows[0]["hand_side"])
    matrices = [row_design_matrix(row) for row in segment_rows]
    depths = [np.asarray(row["depths_m"], dtype=float) for row in segment_rows]
    x0 = np.zeros((len(segment_rows), 6), dtype=float)
    translation_component_bound = float(args.max_translation_m) / math.sqrt(3.0)
    rotation_component_bound = float(args.max_rotation_rad) / math.sqrt(3.0)
    # Bounds must be interleaved per frame as [tx,ty,tz,rx,ry,rz].
    per_frame_bounds = [
        (-translation_component_bound, translation_component_bound),
        (-translation_component_bound, translation_component_bound),
        (-translation_component_bound, translation_component_bound),
        (-rotation_component_bound, rotation_component_bound),
        (-rotation_component_bound, rotation_component_bound),
        (-rotation_component_bound, rotation_component_bound),
    ]
    bounds = per_frame_bounds * len(segment_rows)

    result = minimize(
        lambda flat: objective_and_grad(
            flat,
            matrices,
            depths,
            float(args.translation_prior_weight),
            float(args.rotation_prior_weight),
            float(args.smooth_weight),
            float(args.accel_weight),
            float(args.penetration_weight),
        ),
        x0.reshape(-1),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(args.max_optimizer_iterations), "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    se3 = np.asarray(result.x, dtype=float).reshape((-1, 6)) if result.x is not None else x0

    frame_states: list[dict[str, Any]] = []
    residual_max_values: list[float] = []
    residual_rms_values: list[float] = []
    translation_norm_values: list[float] = []
    rotation_norm_values: list[float] = []
    visible_shift_values: list[float] = []
    state_counter: Counter[str] = Counter()

    for row, state_vec, matrix_i, depth_i in zip(segment_rows, se3, matrices, depths):
        frame_idx = int(row["frame_idx"])
        translation = state_vec[:3]
        rotation = state_vec[3:]
        if len(depth_i) > 0:
            residual = np.maximum(0.0, depth_i - matrix_i @ state_vec)
        else:
            residual = np.zeros((0,), dtype=float)
        residual_max = float(np.max(residual)) if residual.size else 0.0
        residual_rms = float(math.sqrt(float(np.mean(residual * residual)))) if residual.size else 0.0
        translation_norm = float(np.linalg.norm(translation))
        rotation_norm = float(np.linalg.norm(rotation))
        shift = project_se3_shift_px(
            frames_by_idx[frame_idx],
            side,
            np.asarray(row["joints_world"], dtype=float),
            np.asarray(row["hand_center_world"], dtype=float),
            translation,
            rotation,
        )
        shift_max = shift.get("max") if isinstance(shift.get("max"), (float, int)) else None
        residual_max_values.append(residual_max)
        residual_rms_values.append(residual_rms)
        translation_norm_values.append(translation_norm)
        rotation_norm_values.append(rotation_norm)
        if shift_max is not None:
            visible_shift_values.append(float(shift_max))

        residual_ok = residual_max <= float(args.accepted_residual_m)
        shift_ok = shift_max is not None and float(shift_max) <= float(args.visible_shift_limit_px)
        translation_bounded = translation_norm <= float(args.max_translation_m) + 1.0e-12
        rotation_bounded = rotation_norm <= float(args.max_rotation_rad) + 1.0e-12
        if residual_ok and shift_ok and translation_bounded and rotation_bounded:
            temporal_state = "bounded_se3_candidate_hidden_volume_unaccepted"
        elif not residual_ok:
            temporal_state = "unresolved_residual_penetration_after_temporal_se3"
        elif not shift_ok:
            temporal_state = "unresolved_visible_2d_shift_after_temporal_se3"
        elif not translation_bounded:
            temporal_state = "unresolved_translation_bound_after_temporal_se3"
        else:
            temporal_state = "unresolved_rotation_bound_after_temporal_se3"
        state_counter[temporal_state] += 1
        frame_states.append(
            {
                "frame_idx": frame_idx,
                "hand_side": side,
                "hand_center_world_m": np.asarray(row["hand_center_world"], dtype=float).tolist(),
                "optimized_translation_world_m": translation.astype(float).tolist(),
                "optimized_rotation_vector_world_rad": rotation.astype(float).tolist(),
                "translation_norm_m": translation_norm,
                "rotation_norm_rad": rotation_norm,
                "residual_penetration_after_se3_m": {
                    "max": residual_max,
                    "rms": residual_rms,
                    "active_constraint_count": int(np.count_nonzero(residual > 0.0)),
                    "constraint_count": int(len(depth_i)),
                    "constraint_clipped_count": int(row.get("constraint_clipped_count") or 0),
                },
                # Renderer compatibility: existing renderer reads this key for labels.
                "residual_penetration_after_translation_m": {
                    "max": residual_max,
                    "rms": residual_rms,
                    "active_constraint_count": int(np.count_nonzero(residual > 0.0)),
                    "constraint_count": int(len(depth_i)),
                    "constraint_clipped_count": int(row.get("constraint_clipped_count") or 0),
                },
                "visible_joint_shift_px": shift,
                "temporal_mano_state": temporal_state,
                "coordinate_correction_accepted": False,
            }
        )

    if all(state == "bounded_se3_candidate_hidden_volume_unaccepted" for state in state_counter):
        interval_state = "bounded_se3_trajectory_candidate_hidden_volume_unaccepted"
    elif any("residual_penetration" in state for state in state_counter):
        interval_state = "se3_trajectory_blocked_by_residual_penetration"
    elif any("visible_2d" in state for state in state_counter):
        interval_state = "se3_trajectory_blocked_by_visible_2d_shift"
    else:
        interval_state = "se3_trajectory_unaccepted_uncertain"

    interval = {
        "hand_side": side,
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "frame_count": int(len(frames)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", 0) or 0),
        "temporal_mano_interval_state": interval_state,
        "coordinate_correction_accepted": False,
        "state_counts": dict(state_counter),
        "translation_norm_m": numeric_summary(np.asarray(translation_norm_values, dtype=float)),
        "rotation_norm_rad": numeric_summary(np.asarray(rotation_norm_values, dtype=float)),
        "residual_penetration_after_se3_m": numeric_summary(np.asarray(residual_max_values, dtype=float)),
        "residual_penetration_after_translation_m": numeric_summary(np.asarray(residual_max_values, dtype=float)),
        "residual_rms_after_se3_m": numeric_summary(np.asarray(residual_rms_values, dtype=float)),
        "visible_joint_shift_max_px": numeric_summary(np.asarray(visible_shift_values, dtype=float)),
        "blocking_mechanisms": [],
    }
    blockers: list[str] = []
    if interval["residual_penetration_after_se3_m"]["max"] is not None and float(interval["residual_penetration_after_se3_m"]["max"]) > float(args.accepted_residual_m):
        blockers.append("temporal_se3_leaves_residual_penetration")
    if interval["visible_joint_shift_max_px"]["max"] is not None and float(interval["visible_joint_shift_max_px"]["max"]) > float(args.visible_shift_limit_px):
        blockers.append("temporal_se3_exceeds_visible_2d_shift_limit")
    blockers.append("object_hidden_volume_uncertain_not_observed_depth_overwritten_or_free_space_validated")
    interval["blocking_mechanisms"] = blockers
    return interval, frame_states


def main() -> None:
    args = parse_args()
    completion = load_json(args.completion_report)
    frame_rows, frames_by_idx = build_frame_rows(args)
    rows_by_side: dict[str, list[dict[str, Any]]] = {}
    for row in frame_rows:
        if int(row["penetrating_vertex_count"]) <= 0:
            continue
        rows_by_side.setdefault(str(row["hand_side"]), []).append(row)

    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    for side, side_rows in sorted(rows_by_side.items()):
        rows_by_frame = {int(row["frame_idx"]): row for row in side_rows}
        for start, end in contiguous_segments(list(rows_by_frame)):
            segment_rows = [rows_by_frame[idx] for idx in range(start, end + 1) if idx in rows_by_frame]
            if not segment_rows:
                continue
            interval, frame_states = optimize_segment(segment_rows, frames_by_idx, args)
            interval["interval_id"] = f"{side}_{start:04d}_{end:04d}"
            for state in frame_states:
                state["interval_id"] = interval["interval_id"]
            intervals.append(interval)
            per_frame_states.extend(frame_states)

    summary_counts = Counter(interval["temporal_mano_interval_state"] for interval in intervals)
    report = {
        "method": "build_v18_temporal_mano_se3_interval_state",
        "status": "ok",
        "case": str(args.case),
        "object_id": args.object_id,
        "claim_scope": (
            "Temporal rigid-SE(3) optimizer for MANO interval uncertainty. The current MANO surface is globally "
            "translated and rotated about its hand center as a hypothesis; coordinate corrections are not accepted "
            "when residual penetration, visible 2D shift, or hidden-volume uncertainty remains."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report),
            "sign_mesh": str(args.sign_mesh),
        },
        "object_hypothesis_scope": completion.get("claim_scope"),
        "parameters": {
            "max_constraints_per_frame": int(args.max_constraints_per_frame),
            "penetration_epsilon_m": float(args.penetration_epsilon_m),
            "max_translation_m": float(args.max_translation_m),
            "max_rotation_rad": float(args.max_rotation_rad),
            "translation_bound_semantics": "euclidean_norm_conservative_box_bound",
            "rotation_bound_semantics": "euclidean_norm_conservative_box_bound",
            "accepted_residual_m": float(args.accepted_residual_m),
            "visible_shift_limit_px": float(args.visible_shift_limit_px),
            "translation_prior_weight": float(args.translation_prior_weight),
            "rotation_prior_weight": float(args.rotation_prior_weight),
            "smooth_weight": float(args.smooth_weight),
            "accel_weight": float(args.accel_weight),
            "penetration_weight": float(args.penetration_weight),
        },
        "summary": {
            "interval_count": int(len(intervals)),
            "per_frame_state_count": int(len(per_frame_states)),
            "interval_state_counts": dict(summary_counts),
            "coordinate_correction_accepted": False,
        },
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "physical_conclusion": (
            "A smooth rigid-SE(3) MANO trajectory can be computed as a global pose hypothesis. It is not accepted "
            "as corrected hand state unless residual penetration is cleared within tolerance, visible 2D shift remains "
            "bounded, temporal motion is coherent, and object hidden volume is validated."
        ),
    }
    out_path = args.output_dir / "v18_temporal_mano_se3_interval_state.json"
    write_json(out_path, report)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "summary": report["summary"],
                "first_intervals": intervals[:3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
