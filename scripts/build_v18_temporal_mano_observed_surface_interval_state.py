#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Optimize interval MANO pose against observed-supported object surface only.

This is the task5 follow-up to observed-vs-hidden separation. It reuses the
right-hand HaWoR/WiLoR MANO replay and temporal articulated optimizer, but the
object residuals are built only from hand vertices whose closest object face is
supported by metric depth in the current frame. Hidden, behind-depth, and
free-space-conflicted object faces are recorded as uncertainty but do not push
MANO coordinates.

The output is not accepted coordinate correction. It tests whether remaining
observed-surface blockers are caused by correctable MANO articulation error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_compact_rigid_hidden_volume_depth_validation import load_depth_sources  # noqa: E402
from build_v18_observed_surface_mano_constraint_state import (  # noqa: E402
    classify_object_vertices_against_depth,
    face_provenance,
)
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    ReplayFrame,
    bridge_vertices_and_joints,
    load_source_arrays,
    load_wilor_mano_class,
    optimize_segment,
    patch_legacy_mano_loader,
    source_npz_for_hand,
)
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
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
    parser.add_argument("--case", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completed-mesh", type=Path, required=True)
    parser.add_argument("--depth-npz", type=Path, action="append", required=True)
    parser.add_argument("--source-articulated-state", type=Path, required=True)
    parser.add_argument("--hidden-volume-validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-margin-m", type=float, default=0.015)
    parser.add_argument("--free-space-margin-m", type=float, default=0.025)
    parser.add_argument("--max-constraints-per-frame", type=int, default=64)
    parser.add_argument("--penetration-epsilon-m", type=float, default=1.0e-5)
    parser.add_argument("--accepted-residual-m", type=float, default=0.0015)
    parser.add_argument("--visible-shift-limit-px", type=float, default=8.0)
    parser.add_argument("--depth-shift-limit-m", type=float, default=0.025)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.35)
    parser.add_argument("--pose-prior-weight", type=float, default=3.0e2)
    parser.add_argument("--smooth-weight", type=float, default=1.5e3)
    parser.add_argument("--accel-weight", type=float, default=3.0e3)
    parser.add_argument("--penetration-weight", type=float, default=2.5e5)
    parser.add_argument("--visible-hinge-weight", type=float, default=4.0e2)
    parser.add_argument("--depth-hinge-weight", type=float, default=2.0e4)
    parser.add_argument("--max-optimizer-iterations", type=int, default=80)
    parser.add_argument("--sample-vertex-count-for-render", type=int, default=96)
    return parser.parse_args()


def hidden_state_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("frame_rows", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict):
            out[int(row["frame_idx"])] = row
    return out


def right_intervals(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = []
    for interval in payload.get("intervals", []) if isinstance(payload, dict) else []:
        if isinstance(interval, dict) and interval.get("hand_side") == "right":
            rows.append(interval)
    return sorted(rows, key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))


def source_left_states(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(path)
    left_states = []
    left_intervals = []
    for row in payload.get("per_frame_states", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("hand_side") == "left":
            copy = dict(row)
            copy["temporal_mano_state"] = "observed_surface_mano_replay_ineligible_missing_left_mano_model"
            copy["coordinate_correction_accepted"] = False
            left_states.append(copy)
    for row in payload.get("intervals", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("hand_side") == "left":
            copy = dict(row)
            copy["temporal_mano_interval_state"] = "observed_surface_mano_replay_ineligible_missing_left_mano_model"
            copy["coordinate_correction_accepted"] = False
            left_intervals.append(copy)
    return left_intervals, left_states


def build_observed_replay_frame(
    *,
    frame_idx: int,
    frame: dict[str, Any],
    hand: dict[str, Any],
    pose: tuple[np.ndarray, np.ndarray],
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    scene: Any,
    depth_row: dict[str, Any] | None,
    bridge_cache: dict[Path, Any],
    source_cache: dict[Path, Any],
    hidden_rows: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[ReplayFrame | None, dict[str, Any] | None]:
    arrays = bridge_vertices_and_joints(hand, bridge_cache)
    if arrays is None:
        return None, {"frame_idx": frame_idx, "hand_side": "right", "reason": "missing_current_v18_bridge_surface"}
    current_vertices, current_joints = arrays
    source_info = source_npz_for_hand(hand)
    if source_info is None:
        return None, {"frame_idx": frame_idx, "hand_side": "right", "reason": "missing_hawor_source_npz_for_mano_replay"}
    source_path, source_frame = source_info
    source = load_source_arrays(source_cache, source_path)
    required = [
        "right_vertices_world_m",
        "right_joints_world_m",
        "right_root_orient_axis_angle",
        "right_hand_pose_axis_angle",
        "right_betas",
        "right_trans_world_m",
    ]
    missing = [key for key in required if key not in source]
    if missing:
        return None, {"frame_idx": frame_idx, "hand_side": "right", "reason": "source_npz_missing_arrays", "missing": missing}
    raw_vertices = np.asarray(source["right_vertices_world_m"][source_frame], dtype=float)
    raw_joints = np.asarray(source["right_joints_world_m"][source_frame], dtype=float)
    from build_v18_temporal_mano_articulated_interval_state import similarity_from_to  # local import avoids pyright cycle noise

    scale, rot, _trans, sim_err = similarity_from_to(raw_vertices, current_vertices)
    vertex_classes, _depth_summary = classify_object_vertices_against_depth(
        frame=frame,
        vertices_object=object_vertices,
        pose=pose,
        depth_row=depth_row,
        support_margin_m=float(args.support_margin_m),
        free_space_margin_m=float(args.free_space_margin_m),
    )
    prov = face_provenance(vertex_classes, object_faces)
    r_obj, t_obj = pose
    vertices_object_hand = inverse_object(current_vertices, r_obj, t_obj)
    signed = -scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object_hand, dtype=np.float32))).numpy().astype(float)
    penetrating_idx = np.where(signed > float(args.penetration_epsilon_m))[0]
    all_depths = signed[penetrating_idx]
    observed_idx = np.zeros((0,), dtype=np.int64)
    hidden_depths = np.zeros((0,), dtype=float)
    if penetrating_idx.size:
        closest_all = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object_hand[penetrating_idx], dtype=np.float32)))
        prim = closest_all["primitive_ids"].numpy().astype(np.int64)
        valid = (prim >= 0) & (prim < len(object_faces))
        observed_mask = np.zeros_like(valid, dtype=bool)
        observed_mask[valid] = prov["observed_supported_strict"][prim[valid]]
        observed_idx = penetrating_idx[observed_mask]
        hidden_depths = signed[penetrating_idx[~observed_mask]]
    constraint_idx = observed_idx.astype(np.int64)
    clipped = 0
    if len(constraint_idx) > int(args.max_constraints_per_frame):
        order = np.argsort(signed[constraint_idx])[::-1]
        keep = order[: int(args.max_constraints_per_frame)]
        clipped = int(len(constraint_idx) - len(keep))
        constraint_idx = constraint_idx[keep]
    normals_world = np.zeros((0, 3), dtype=float)
    depths = np.zeros((0,), dtype=float)
    if len(constraint_idx):
        closest = scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object_hand[constraint_idx], dtype=np.float32)))["points"].numpy().astype(float)
        disp = closest - vertices_object_hand[constraint_idx]
        norms = np.linalg.norm(disp, axis=1)
        valid = norms > 1.0e-12
        normals_world = object_vec_to_world(disp[valid] / norms[valid, None], r_obj)
        depths = signed[constraint_idx][valid]
        constraint_idx = constraint_idx[valid]
    volume_row = hidden_rows.get(frame_idx)
    volume_state = str(volume_row.get("state", "hidden_volume_unvalidated")) if isinstance(volume_row, dict) else "hidden_volume_unvalidated"
    row = ReplayFrame(
        frame_idx=frame_idx,
        hand_side="right",
        current_vertices_world=current_vertices,
        current_joints_world=current_joints,
        raw_vertices_world=raw_vertices,
        raw_joints_world=raw_joints,
        root_orient_axis_angle=np.asarray(source["right_root_orient_axis_angle"][source_frame], dtype=float),
        hand_pose_axis_angle=np.asarray(source["right_hand_pose_axis_angle"][source_frame], dtype=float),
        betas=np.asarray(source["right_betas"][source_frame], dtype=float),
        trans_world_m=np.asarray(source["right_trans_world_m"][source_frame], dtype=float),
        source_hawor_npz=source_path,
        source_frame_index=int(source_frame),
        similarity_scale=float(scale),
        similarity_rotation_raw_to_current=rot.astype(float),
        similarity_error_median_m=float(np.median(sim_err)),
        similarity_error_p95_m=float(np.percentile(sim_err, 95)),
        raw_replay_vertex_error_median_m=float("nan"),
        raw_replay_joint_error_median_m=float("nan"),
        frame=frame,
        constraint_indices=constraint_idx.astype(np.int64),
        constraint_normals_world=normals_world.astype(float),
        constraint_depths_m=depths.astype(float),
        penetration_depths_all_m=all_depths.astype(float),
        constraint_clipped_count=clipped,
        hidden_volume_state=volume_state,
    )
    extra = {
        "observed_supported_constraint_count": int(len(depths)),
        "hidden_or_unvalidated_penetration_count_initial": int(len(hidden_depths)),
        "full_initial_penetration_depth_m": numeric_summary(all_depths),
        "observed_initial_penetration_depth_m": numeric_summary(depths),
        "hidden_or_unvalidated_initial_penetration_depth_m": numeric_summary(hidden_depths),
    }
    return row, extra


def rename_state(state: str, constraint_count: int) -> str:
    if constraint_count == 0:
        return "observed_surface_no_active_supported_constraint_hidden_unresolved"
    return state.replace("temporal_articulated_mano", "observed_surface_articulated_mano").replace(
        "bounded_articulated_mano", "bounded_observed_surface_articulated_mano"
    )


def main() -> None:
    args = parse_args()
    patch_legacy_mano_loader()
    mano_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_path.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT model: {mano_path}")
    mano_cls = load_wilor_mano_class(args.wilor_root)
    device = torch.device(args.device)
    model = mano_cls(model_path=str(mano_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    model.eval()

    annotations = load_json(args.annotations)
    frames = [frame for frame in as_list(annotations.get("frames")) if isinstance(frame, dict)]
    frames_by_idx = {int(frame["frame_idx"]): frame for frame in frames if frame.get("frame_idx") is not None}
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    mesh = load_mesh(args.completed_mesh)
    object_vertices = np.asarray(mesh.vertices, dtype=float)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    depth_by_frame = load_depth_sources(args.depth_npz)
    hidden_rows = hidden_state_map(args.hidden_volume_validation)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(object_vertices.astype(np.float32)), o3d.core.Tensor(object_faces.astype(np.uint32)))
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    skipped: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    left_intervals, left_states = source_left_states(args.source_articulated_state)
    intervals.extend(left_intervals)
    per_frame_states.extend(left_states)
    right_source_intervals = right_intervals(args.source_articulated_state)
    optimized_frame_count = 0
    observed_constraint_counts: list[int] = []
    hidden_initial_counts: list[int] = []
    for src_interval in right_source_intervals:
        start = int(src_interval["start_frame"])
        end = int(src_interval["end_frame"])
        rows: list[ReplayFrame] = []
        extras: dict[int, dict[str, Any]] = {}
        for frame_idx in range(start, end + 1):
            frame = frames_by_idx.get(frame_idx)
            pose = poses.get(frame_idx)
            if frame is None or pose is None:
                continue
            right_hand = None
            for hand in as_list(frame.get("hands")):
                if isinstance(hand, dict) and hand.get("hand_side") == "right":
                    right_hand = hand
                    break
            if right_hand is None:
                continue
            row, extra = build_observed_replay_frame(
                frame_idx=frame_idx,
                frame=frame,
                hand=right_hand,
                pose=pose,
                object_vertices=object_vertices,
                object_faces=object_faces,
                scene=scene,
                depth_row=depth_by_frame.get(frame_idx),
                bridge_cache=bridge_cache,
                source_cache=source_cache,
                hidden_rows=hidden_rows,
                args=args,
            )
            if row is None:
                if extra is not None:
                    skipped.append(extra)
                continue
            rows.append(row)
            if extra is not None:
                extras[frame_idx] = extra
                observed_constraint_counts.append(int(extra["observed_supported_constraint_count"]))
                hidden_initial_counts.append(int(extra["hidden_or_unvalidated_penetration_count_initial"]))
        if not rows:
            continue
        interval, states = optimize_segment(model=model, rows=rows, args=args, device=device)
        interval_id = f"right_observed_{rows[0].frame_idx:04d}_{rows[-1].frame_idx:04d}"
        interval["interval_id"] = interval_id
        interval["source_interval_id"] = src_interval.get("interval_id")
        interval["constraint_scope"] = "observed_supported_object_faces_only"
        interval["temporal_mano_interval_state"] = interval["temporal_mano_interval_state"].replace(
            "articulated_mano", "observed_surface_articulated_mano"
        )
        interval["coordinate_correction_accepted"] = False
        interval["observed_supported_constraint_count"] = numeric_summary(np.asarray([extras[r.frame_idx]["observed_supported_constraint_count"] for r in rows if r.frame_idx in extras], dtype=float))
        interval["hidden_or_unvalidated_initial_penetration_count"] = numeric_summary(np.asarray([extras[r.frame_idx]["hidden_or_unvalidated_penetration_count_initial"] for r in rows if r.frame_idx in extras], dtype=float))
        new_state_counts: Counter[str] = Counter()
        for state in states:
            frame_idx = int(state["frame_idx"])
            extra = extras.get(frame_idx, {})
            constraint_count = int(extra.get("observed_supported_constraint_count", 0))
            state["source_interval_id"] = src_interval.get("interval_id")
            state["interval_id"] = interval_id
            state["constraint_scope"] = "observed_supported_object_faces_only"
            state["temporal_mano_state"] = rename_state(str(state.get("temporal_mano_state")), constraint_count)
            state["coordinate_correction_accepted"] = False
            state["observed_surface_constraint_input"] = extra
            new_state_counts[state["temporal_mano_state"]] += 1
        interval["state_counts"] = dict(new_state_counts)
        intervals.append(interval)
        per_frame_states.extend(states)
        optimized_frame_count += len(rows)

    interval_counts = Counter(str(row.get("temporal_mano_interval_state")) for row in intervals)
    frame_counts = Counter(str(row.get("temporal_mano_state")) for row in per_frame_states)
    report = {
        "method": "build_v18_temporal_mano_observed_surface_interval_state",
        "status": "ok",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Interval-level right-hand MANO pose optimization using only depth-observed-supported object faces as object constraints. "
            "Hidden/free-space object volume is excluded from the optimizer and remains uncertainty. No coordinate correction is accepted."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
            "depth_npz": [str(path) for path in args.depth_npz],
            "source_articulated_state": str(args.source_articulated_state),
            "hidden_volume_validation": str(args.hidden_volume_validation) if args.hidden_volume_validation else None,
            "wilor_mano_right": str(mano_path),
        },
        "parameters": {
            "support_margin_m": float(args.support_margin_m),
            "free_space_margin_m": float(args.free_space_margin_m),
            "max_constraints_per_frame": int(args.max_constraints_per_frame),
            "accepted_residual_m": float(args.accepted_residual_m),
            "visible_shift_limit_px": float(args.visible_shift_limit_px),
            "depth_shift_limit_m": float(args.depth_shift_limit_m),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "constraint_scope": "strict observed-supported faces: closest face has at least two depth-supported vertices and no free-space vertex",
        },
        "summary": {
            "annotation_frame_count": int(len(frames)),
            "pose_frame_count": int(len(poses)),
            "source_right_interval_count": int(len(right_source_intervals)),
            "optimized_right_frame_count": int(optimized_frame_count),
            "interval_count": int(len(intervals)),
            "per_frame_state_count": int(len(per_frame_states)),
            "interval_state_counts": dict(interval_counts),
            "per_frame_state_counts": dict(frame_counts),
            "observed_supported_constraint_count": numeric_summary(np.asarray(observed_constraint_counts, dtype=float)),
            "hidden_or_unvalidated_initial_penetration_count": numeric_summary(np.asarray(hidden_initial_counts, dtype=float)),
            "coordinate_correction_accepted": False,
        },
        "skipped": skipped[:200],
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "physical_conclusion": (
            "Observed-surface-only optimization tests whether visible object surface conflicts can be fixed by MANO articulation. "
            "Any remaining hidden/free-space residual is not optimized and cannot support coordinate closure."
        ),
    }
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_temporal_mano_observed_surface_interval_state.json"
    write_json(out_path, report)
    print(json.dumps({"output": str(out_path), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
