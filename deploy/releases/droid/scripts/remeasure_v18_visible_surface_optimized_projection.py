#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Remeasure visible-surface depth order at optimized MANO projections.

The joint MANO interval solver stores a selected visible-surface depth-order
residual from the pre-optimization projection, then reports how the optimized
hand changes that fixed selected residual.  Large occlusion hypotheses can move
MANO far enough in image space that those fixed correspondences are stale.  This
script rebuilds the same interval rows, reconstructs full optimized MANO
vertices from the saved optimized deltas, and remeasures visible first-surface
mask/depth constraints at the optimized projection.

This is a geometric verification of an interval hand-state mechanism.  It does
not run vision-model inference, produce labels, or decide contact/object pose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_v18_joint_mano_interval_trajectory as solve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Joint MANO interval solver state JSON to remeasure.")
    parser.add_argument("--output", type=Path, required=True, help="Output remeasurement report JSON.")
    parser.add_argument(
        "--compute-note",
        default="local_cpu_light_geometry_replay_no_model_inference_no_gpu",
        help="Compute provenance string recorded in the report.",
    )
    parser.add_argument(
        "solver_args",
        nargs=argparse.REMAINDER,
        help="Pass the original solver arguments after '--'. If --device is omitted, CPU is forced by default.",
    )
    args = parser.parse_args()
    if args.solver_args and args.solver_args[0] == "--":
        args.solver_args = args.solver_args[1:]
    if not args.solver_args:
        raise ValueError("solver arguments are required after '--' so the same interval/factors can be rebuilt")
    return args


def parse_solver_args(solver_args: list[str]) -> argparse.Namespace:
    actual = list(solver_args)
    if "--device" not in actual:
        actual = ["--device", "cpu", *actual]
    old_argv = sys.argv
    try:
        sys.argv = ["solve_v18_joint_mano_interval_trajectory.py", *actual]
        return solve.parse_args()
    finally:
        sys.argv = old_argv


def numeric_summary(values: list[float | int | None] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray([v for v in np.ravel(values).tolist() if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "min": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def load_binary_mask(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"visible-surface mask does not exist: {mask_path}")
    return np.asarray(Image.open(mask_path).convert("L")) > 0


def float_or_default(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def reconstruct_optimized_vertices(
    rows: list[Any],
    states_by_key: dict[tuple[int, str], dict[str, Any]],
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if not rows:
        return np.zeros((0, 0, 3), dtype=float), np.zeros((0, 0, 3), dtype=float), []
    batch = len(rows)
    root = torch.tensor(np.stack([r.root_orient_axis_angle for r in rows]).reshape(batch, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r.hand_pose_axis_angle for r in rows]).reshape(batch, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r.betas for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r.trans_world_m for r in rows]), dtype=torch.float32, device=device)
    base_root_mat = solve.rotvec_to_matrix(root)
    base_pose_mat = solve.rotvec_to_matrix(pose)
    with torch.no_grad():
        base_out = model(global_orient=base_root_mat, hand_pose=base_pose_mat, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_base_vertices = base_out.vertices.detach().cpu().numpy().astype(float)
        raw_base_joints = base_out.joints.detach().cpu().numpy().astype(float)

    root_delta_rows: list[np.ndarray] = []
    pose_delta_rows: list[np.ndarray] = []
    trans_delta_rows: list[np.ndarray] = []
    for row in rows:
        key = (int(row.frame_idx), str(row.side))
        if key not in states_by_key:
            raise KeyError(f"state JSON lacks optimized row for frame/side {key}")
        state = states_by_key[key]
        root_delta_rows.append(np.asarray(state["optimized_root_delta_axis_angle_rad"], dtype=float).reshape(1, 3))
        pose_delta_rows.append(np.asarray(state["optimized_hand_pose_delta_axis_angle_rad"], dtype=float).reshape(15, 3))
        trans_delta_rows.append(np.asarray(state["optimized_translation_world_m"], dtype=float).reshape(3))

    root_delta = torch.tensor(np.stack(root_delta_rows), dtype=torch.float32, device=device)
    pose_delta = torch.tensor(np.stack(pose_delta_rows), dtype=torch.float32, device=device)
    trans_delta = torch.tensor(np.stack(trans_delta_rows), dtype=torch.float32, device=device)
    sim_scale_np = np.asarray([r.similarity_scale for r in rows], dtype=float).reshape(batch, 1, 1)
    sim_rot_np = np.stack([r.similarity_rotation_raw_to_current for r in rows]).astype(float)
    sim_trans_np = np.stack([r.similarity_translation_raw_to_current for r in rows]).astype(float)
    sim_scale_t = torch.tensor(sim_scale_np, dtype=torch.float32, device=device)
    sim_rot_t = torch.tensor(sim_rot_np, dtype=torch.float32, device=device)
    sim_trans_t = torch.tensor(sim_trans_np, dtype=torch.float32, device=device).reshape(batch, 1, 3)
    current_vertices_t = torch.tensor(np.stack([r.current_vertices_world for r in rows]).astype(float), dtype=torch.float32, device=device)
    current_joints_t = torch.tensor(np.stack([r.current_joints_world for r in rows]).astype(float), dtype=torch.float32, device=device)
    raw_base_vertices_t = torch.tensor(raw_base_vertices, dtype=torch.float32, device=device)
    raw_base_joints_t = torch.tensor(raw_base_joints, dtype=torch.float32, device=device)

    with torch.no_grad():
        new_root = solve.rotvec_to_matrix(root_delta) @ base_root_mat
        new_pose = solve.rotvec_to_matrix(pose_delta) @ base_pose_mat
        out = model(global_orient=new_root, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        if str(args.zero_surface_mode) == "similarity_mapped_raw":
            mapped_vertices = sim_scale_t * torch.matmul(out.vertices, sim_rot_t.transpose(1, 2)) + sim_trans_t
            mapped_joints = sim_scale_t * torch.matmul(out.joints, sim_rot_t.transpose(1, 2)) + sim_trans_t
            vertices = mapped_vertices + trans_delta[:, None, :]
            joints = mapped_joints + trans_delta[:, None, :]
        else:
            raw_delta_vertices = out.vertices - raw_base_vertices_t
            raw_delta_joints = out.joints - raw_base_joints_t
            mapped_vertices = sim_scale_t * torch.matmul(raw_delta_vertices, sim_rot_t.transpose(1, 2))
            mapped_joints = sim_scale_t * torch.matmul(raw_delta_joints, sim_rot_t.transpose(1, 2))
            vertices = current_vertices_t + mapped_vertices + trans_delta[:, None, :]
            joints = current_joints_t + mapped_joints + trans_delta[:, None, :]

    vertices_np = vertices.detach().cpu().numpy().astype(float)
    joints_np = joints.detach().cpu().numpy().astype(float)
    reconstruction_errors: list[float] = []
    for i, row in enumerate(rows):
        state = states_by_key[(int(row.frame_idx), str(row.side))]
        sample_ids = np.asarray(state.get("optimized_vertices_sample_ids") or [], dtype=int)
        sample_vertices = np.asarray(state.get("optimized_vertices_world_sample_m") or [], dtype=float)
        if sample_ids.size and sample_vertices.shape == (sample_ids.size, 3):
            reconstruction_errors.append(float(np.max(np.linalg.norm(vertices_np[i, sample_ids] - sample_vertices, axis=1))))
    return vertices_np, joints_np, reconstruction_errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "current_reproject_all_in_front_sum": int(sum(r["current_reproject_all_in_front_count"] for r in rows)),
        "fixed_selected_initial_in_front_sum": int(sum(r["fixed_selected_initial_in_front_count"] for r in rows)),
        "fixed_selected_final_in_front_sum": int(sum(r["fixed_selected_final_in_front_count"] for r in rows)),
        "optimized_reproject_all_inside_sum": int(sum(r["optimized_reproject_all_inside_count"] for r in rows)),
        "optimized_reproject_all_in_front_sum": int(sum(r["optimized_reproject_all_in_front_count"] for r in rows)),
        "optimized_reproject_selected_in_front_sum": int(sum(r["optimized_reproject_selected_in_front_count"] for r in rows)),
        "optimized_reproject_all_delta_median_m": numeric_summary([
            (r.get("optimized_reproject_all_delta_hand_minus_surface_m") or {}).get("median") for r in rows
        ]),
        "optimized_reproject_selected_delta_min_m": numeric_summary([
            (r.get("optimized_reproject_selected_delta_hand_minus_surface_m") or {}).get("min") for r in rows
        ]),
        "optimized_projection_shift_px_max_summary": numeric_summary([
            (r.get("optimized_projection_shift_px") or {}).get("max") for r in rows
        ]),
    }


def main() -> None:
    cli = parse_args()
    solver_args = parse_solver_args(list(cli.solver_args))
    if str(solver_args.device) != "cpu":
        # This script is often run locally as a light geometry check; forcing a
        # CUDA replay on a workstation would violate the project compute policy.
        # Server/A800 users can still pass --device cuda intentionally and the
        # provenance will record it.
        pass
    device = torch.device(str(solver_args.device))
    state = json.loads(cli.state.read_text())
    state_rows = [row for row in state.get("per_frame_states", []) if isinstance(row, dict)]
    states_by_key = {(int(row["frame_idx"]), str(row["hand_side"])): row for row in state_rows}
    if len(states_by_key) != len(state_rows):
        raise ValueError("state JSON has duplicate frame/hand rows")

    models = solve.load_models(solver_args, device)
    depth_rows = solve.load_depth_sources(list(solver_args.depth_npz or [solve.DEFAULT_DEPTH]))
    report_rows: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    build_meta: dict[str, Any] = {}

    for side in solver_args.sides:
        rows, meta, _scene = solve.build_rows(solver_args, side)
        build_meta[str(side)] = meta
        vertices, joints, side_errors = reconstruct_optimized_vertices(rows, states_by_key, models[side], solver_args, device)
        reconstruction_errors.extend(side_errors)
        for i, row in enumerate(rows):
            key = (int(row.frame_idx), str(row.side))
            state_row = states_by_key[key]
            mask_path = row.visible_surface_track_mask_path or row.visible_object_mask_path
            mask = load_binary_mask(mask_path)
            depth_row = depth_rows.get(int(row.frame_idx))
            _cur_ids, _cur_depth, _cur_delta, current_measure = solve.visible_surface_depth_order_constraints(
                frame=row.frame,
                side=row.side,
                vertices_world=row.current_vertices_world,
                mask=mask,
                depth_row=depth_row,
                args=solver_args,
                enabled=True,
            )
            selected_ids, _selected_depths, selected_delta, final_measure = solve.visible_surface_depth_order_constraints(
                frame=row.frame,
                side=row.side,
                vertices_world=vertices[i],
                mask=mask,
                depth_row=depth_row,
                args=solver_args,
                enabled=True,
            )
            margin = float(solver_args.visible_surface_depth_order_margin_m)
            optimized_selected_in_front = int(np.count_nonzero(np.asarray(selected_delta, dtype=float) < -margin))
            optimized_uv = solve.project_world(vertices[i], row.frame, row.side)
            current_uv = solve.project_world(row.current_vertices_world, row.frame, row.side)
            projection_shift = np.zeros((0,), dtype=float)
            if optimized_uv is not None and current_uv is not None and optimized_uv.shape == current_uv.shape:
                projection_shift = np.linalg.norm(optimized_uv - current_uv, axis=1)
            report_rows.append(
                {
                    "frame_idx": int(row.frame_idx),
                    "hand_side": str(row.side),
                    "visible_surface_state": row.visible_surface_track_factor_state,
                    "mask_path": mask_path,
                    "current_reproject_all_inside_count": int(current_measure.get("finite_inside_count") or 0),
                    "current_reproject_all_in_front_count": int(current_measure.get("hand_in_front_of_observed_surface_count") or 0),
                    "fixed_selected_initial_in_front_count": int(state_row.get("visible_surface_depth_order_selected_initial_in_front_count") or 0),
                    "fixed_selected_final_in_front_count": int(state_row.get("visible_surface_depth_order_selected_final_in_front_count") or 0),
                    "optimized_reproject_all_inside_count": int(final_measure.get("finite_inside_count") or 0),
                    "optimized_reproject_all_in_front_count": int(final_measure.get("hand_in_front_of_observed_surface_count") or 0),
                    "optimized_reproject_all_behind_count": int(final_measure.get("hand_behind_observed_surface_count") or 0),
                    "optimized_reproject_all_near_count": int(final_measure.get("hand_near_observed_surface_depth_count") or 0),
                    "optimized_reproject_all_delta_hand_minus_surface_m": final_measure.get("depth_delta_hand_minus_surface_m"),
                    "optimized_reproject_selected_vertex_count": int(len(selected_ids)),
                    "optimized_reproject_selected_in_front_count": optimized_selected_in_front,
                    "optimized_reproject_selected_delta_hand_minus_surface_m": numeric_summary(selected_delta),
                    "optimized_projection_shift_px": numeric_summary(projection_shift),
                    "hand_observation_visibility_weight_multiplier": float_or_default(state_row.get("hand_observation_visibility_weight_multiplier"), 1.0),
                    "visible_joint_shift_px": state_row.get("visible_joint_shift_px"),
                    "joint_camera_depth_shift_m": state_row.get("joint_camera_depth_shift_m"),
                }
            )

    def select_rows(*, side: str | None = None, latent: bool | None = None) -> list[dict[str, Any]]:
        out = report_rows
        if side is not None:
            out = [row for row in out if row["hand_side"] == side]
        if latent is not None:
            if latent:
                out = [row for row in out if float(row["hand_observation_visibility_weight_multiplier"]) <= 1.0e-6]
            else:
                out = [row for row in out if float(row["hand_observation_visibility_weight_multiplier"]) > 1.0e-6]
        return out

    summary = {
        "all": summarize_rows(report_rows),
        "left": summarize_rows(select_rows(side="left")),
        "right": summarize_rows(select_rows(side="right")),
        "latent_zero_observation": summarize_rows(select_rows(latent=True)),
        "nonlatent": summarize_rows(select_rows(latent=False)),
        "hard_frames": [row for row in report_rows if row["hand_side"] == "left" and row["frame_idx"] in (988, 1000, 1002)],
        "reconstruction_max_error_m": numeric_summary(reconstruction_errors),
    }
    out = {
        "method": "remeasure_v18_visible_surface_optimized_projection",
        "claim_scope": "Remeasures optimized MANO vertices against visible first-surface mask/depth at optimized projections; distinguishes physical depth-order improvement from fixed-correspondence residual reduction.",
        "input_state": str(cli.state),
        "compute": str(cli.compute_note),
        "parameters": {
            "case": str(solver_args.case),
            "object_id": str(solver_args.object_id),
            "start_frame": int(solver_args.start_frame),
            "end_frame": int(solver_args.end_frame),
            "sides": list(solver_args.sides),
            "device": str(solver_args.device),
            "visible_surface_depth_order_margin_m": float(solver_args.visible_surface_depth_order_margin_m),
            "max_visible_surface_depth_vertices": int(solver_args.max_visible_surface_depth_vertices),
        },
        "build_meta": build_meta,
        "summary": summary,
        "rows": report_rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({"output": str(cli.output), "summary": summary}, indent=2)[:8000])


if __name__ == "__main__":
    main()
