#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Build interval-level articulated MANO hand-pose hypotheses or uncertainty.

This is the physical mechanism after translation and global rigid-SE(3) failed.
For each current V18 compact-rigid object conflict interval, it tests whether a
small temporally smooth MANO hand-pose perturbation can reduce object/hand
penetration while preserving visible 2D/depth compatibility.

Important scope:
- The zero state is the current V18 bridge MANO surface.
- HaWoR MANO replay is required to reproduce the saved HaWoR surface before
  any articulated hypothesis is eligible.
- Left-hand replay is eligible only when a real MANO_LEFT.pkl is supplied and
  the documented HaWoR left shapedirs-x fix reproduces saved left surfaces.
- Free-space-conflicted or otherwise hidden-only object volume cannot support an
  accepted coordinate correction. It can only produce bounded hypotheses or
  interval uncertainty/falsification.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch

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

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass(frozen=True)
class ReplayFrame:
    frame_idx: int
    hand_side: str
    current_vertices_world: np.ndarray
    current_joints_world: np.ndarray
    raw_vertices_world: np.ndarray
    raw_joints_world: np.ndarray
    root_orient_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    betas: np.ndarray
    trans_world_m: np.ndarray
    source_hawor_npz: Path
    source_frame_index: int
    similarity_scale: float
    similarity_rotation_raw_to_current: np.ndarray
    similarity_error_median_m: float
    similarity_error_p95_m: float
    raw_replay_vertex_error_median_m: float
    raw_replay_joint_error_median_m: float
    frame: dict[str, Any]
    constraint_indices: np.ndarray
    constraint_normals_world: np.ndarray
    constraint_depths_m: np.ndarray
    penetration_depths_all_m: np.ndarray
    constraint_clipped_count: int
    hidden_volume_state: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pose-report", type=Path, default=DEFAULT_POSE_REPORT)
    parser.add_argument("--completion-report", type=Path, default=DEFAULT_COMPLETION_REPORT)
    parser.add_argument("--sign-mesh", type=Path, default=DEFAULT_SIGN_MESH)
    parser.add_argument("--hidden-volume-validation", type=Path, default=None)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--object-id", default="object:obj_tomato")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--wilor-mano-right", type=Path, default=None)
    parser.add_argument("--wilor-mano-left", type=Path, default=None)
    parser.add_argument(
        "--hawor-left-shapedirs-x-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply HaWoR's MANO_LEFT shapedirs[:,0,:] *= -1 convention before left replay.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eligible-side", choices=("right", "both"), default="right")
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
    parser.add_argument("--max-articulated-frames", type=int, default=None)
    parser.add_argument("--sample-vertex-count-for-render", type=int, default=96)
    return parser.parse_args()


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
        if not hasattr(np, name):
            setattr(np, name, value)


def load_wilor_mano_class(wilor_root: Path):
    path = wilor_root / "wilor" / "models" / "mano_wrapper.py"
    if not path.exists():
        raise FileNotFoundError(f"missing WiLoR MANO wrapper: {path}")
    spec = importlib.util.spec_from_file_location("wilor_mano_wrapper_v18_articulated", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load WiLoR MANO wrapper spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MANO


def skew(rotvec: torch.Tensor) -> torch.Tensor:
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(skew(rotvec))


def similarity_from_to(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("similarity expects matching Nx3 arrays")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    src = source - source_center
    tgt = target - target_center
    covariance = src.T @ tgt / len(source)
    u, s, vt = np.linalg.svd(covariance)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    var = float(np.sum(src * src) / len(source))
    scale = float(np.sum(s) / max(var, 1.0e-12))
    trans = target_center - scale * (source_center @ rot.T)
    pred = scale * (source @ rot.T) + trans[None, :]
    err = np.linalg.norm(pred - target, axis=1)
    return scale, rot, trans, err


def load_bridge_array(cache: dict[Path, Any], bridge_path: Path, array_name: str, row_index: int) -> np.ndarray:
    if bridge_path not in cache:
        with np.load(bridge_path, allow_pickle=True) as z:
            needed = {
                "vertices_current_v18_world_from_hawor_projection_relift_m",
                "joints_current_v18_world_from_hawor_projection_relift_m",
            }
            cache[bridge_path] = {key: np.asarray(z[key]) for key in z.files if key in needed or key == array_name}
    return np.asarray(cache[bridge_path][array_name][row_index], dtype=float)


def load_hidden_validation(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("frame_rows", []) if isinstance(payload.get("frame_rows"), list) else []:
        if isinstance(row, dict):
            out[int(row["frame_idx"])] = row
    return out


def source_npz_for_hand(hand: dict[str, Any]) -> tuple[Path, int] | None:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    params = metric.get("mano_params") if isinstance(metric.get("mano_params"), dict) else {}
    source = params.get("source_hawor_npz")
    source_frame = params.get("source_frame_index")
    if not isinstance(source, str) or source_frame is None:
        ref = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
        source = ref.get("source_hawor_npz")
        source_frame = ref.get("source_frame_index")
    if not isinstance(source, str) or source_frame is None:
        return None
    path = Path(source)
    if path.is_dir():
        # Prefer support-augmented exports where present.
        candidates = sorted(path.glob("*with_track_support*.npz")) + sorted(path.glob("*.npz"))
        if not candidates:
            return None
        path = candidates[0]
    if not path.exists():
        return None
    return path, int(source_frame)


def bridge_vertices_and_joints(hand: dict[str, Any], bridge_cache: dict[Path, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
    reference = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
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
    joints_world = load_bridge_array(bridge_cache, bridge_path, "joints_current_v18_world_from_hawor_projection_relift_m", row_index)
    return vertices_world, joints_world


def project_world(points_world: np.ndarray, frame: dict[str, Any], side: str) -> np.ndarray | None:
    intr = frame_intrinsics(frame, side)
    if intr is None:
        return None
    r_c2w, t_c2w = frame_camera_pose(frame)
    return project(points_world, r_c2w, t_c2w, intr)


def world_to_camera(points_world: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    r_c2w, t_c2w = frame_camera_pose(frame)
    return (points_world - t_c2w[None, :]) @ r_c2w


def load_source_arrays(cache: dict[Path, Any], path: Path) -> dict[str, np.ndarray]:
    if path not in cache:
        with np.load(path, allow_pickle=True) as z:
            cache[path] = {key: np.asarray(z[key]) for key in z.files}
    return cache[path]


def build_replay_frame(
    *,
    frame_idx: int,
    frame: dict[str, Any],
    hand: dict[str, Any],
    side: str,
    pose: tuple[np.ndarray, np.ndarray],
    sign_scene: Any,
    bridge_cache: dict[Path, Any],
    source_cache: dict[Path, Any],
    hidden_rows: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[ReplayFrame | None, dict[str, Any] | None]:
    arrays = bridge_vertices_and_joints(hand, bridge_cache)
    if arrays is None:
        return None, {"frame_idx": frame_idx, "hand_side": side, "reason": "missing_current_v18_bridge_surface"}
    current_vertices, current_joints = arrays
    source_info = source_npz_for_hand(hand)
    if source_info is None:
        return None, {"frame_idx": frame_idx, "hand_side": side, "reason": "missing_hawor_source_npz_for_mano_replay"}
    source_path, source_frame = source_info
    source = load_source_arrays(source_cache, source_path)
    required = [
        f"{side}_vertices_world_m",
        f"{side}_joints_world_m",
        f"{side}_root_orient_axis_angle",
        f"{side}_hand_pose_axis_angle",
        f"{side}_betas",
        f"{side}_trans_world_m",
    ]
    missing = [key for key in required if key not in source]
    if missing:
        return None, {"frame_idx": frame_idx, "hand_side": side, "reason": "source_npz_missing_arrays", "missing": missing}
    raw_vertices = np.asarray(source[f"{side}_vertices_world_m"][source_frame], dtype=float)
    raw_joints = np.asarray(source[f"{side}_joints_world_m"][source_frame], dtype=float)
    scale, rot, _trans, sim_err = similarity_from_to(raw_vertices, current_vertices)

    r_obj, t_obj = pose
    vertices_object = inverse_object(current_vertices, r_obj, t_obj)
    signed = -sign_scene.compute_signed_distance(o3d.core.Tensor(np.asarray(vertices_object, dtype=np.float32))).numpy().astype(float)
    penetrating_idx = np.where(signed > float(args.penetration_epsilon_m))[0]
    all_depths = signed[penetrating_idx]
    constraint_idx = penetrating_idx
    clipped = 0
    if len(constraint_idx) > int(args.max_constraints_per_frame):
        order = np.argsort(all_depths)[::-1]
        keep = order[: int(args.max_constraints_per_frame)]
        constraint_idx = penetrating_idx[keep]
        clipped = int(len(penetrating_idx) - len(constraint_idx))
    normals_world = np.zeros((0, 3), dtype=float)
    depths = np.zeros((0,), dtype=float)
    if len(constraint_idx) > 0:
        closest = sign_scene.compute_closest_points(o3d.core.Tensor(np.asarray(vertices_object[constraint_idx], dtype=np.float32)))["points"].numpy().astype(float)
        disp = closest - vertices_object[constraint_idx]
        norms = np.linalg.norm(disp, axis=1)
        valid = norms > 1.0e-12
        normals_world = object_vec_to_world(disp[valid] / norms[valid, None], r_obj)
        depths = signed[constraint_idx][valid]
        constraint_idx = constraint_idx[valid]
    volume_row = hidden_rows.get(frame_idx)
    volume_state = str(volume_row.get("state", "hidden_volume_unvalidated")) if isinstance(volume_row, dict) else "hidden_volume_unvalidated"
    return ReplayFrame(
        frame_idx=frame_idx,
        hand_side=side,
        current_vertices_world=current_vertices,
        current_joints_world=current_joints,
        raw_vertices_world=raw_vertices,
        raw_joints_world=raw_joints,
        root_orient_axis_angle=np.asarray(source[f"{side}_root_orient_axis_angle"][source_frame], dtype=float),
        hand_pose_axis_angle=np.asarray(source[f"{side}_hand_pose_axis_angle"][source_frame], dtype=float),
        betas=np.asarray(source[f"{side}_betas"][source_frame], dtype=float),
        trans_world_m=np.asarray(source[f"{side}_trans_world_m"][source_frame], dtype=float),
        source_hawor_npz=source_path,
        source_frame_index=source_frame,
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
    ), None


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


def sample_ids(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def optimize_segment(
    *,
    model: Any,
    rows: list[ReplayFrame],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b = len(rows)
    root = torch.tensor(np.stack([r.root_orient_axis_angle for r in rows]).reshape(b, 1, 3), dtype=torch.float32, device=device)
    pose = torch.tensor(np.stack([r.hand_pose_axis_angle for r in rows]).reshape(b, 15, 3), dtype=torch.float32, device=device)
    betas = torch.tensor(np.stack([r.betas for r in rows]), dtype=torch.float32, device=device)
    trans = torch.tensor(np.stack([r.trans_world_m for r in rows]), dtype=torch.float32, device=device)
    base_root_mat = rotvec_to_matrix(root)
    base_pose_mat = rotvec_to_matrix(pose)
    with torch.no_grad():
        base_out = model(global_orient=base_root_mat, hand_pose=base_pose_mat, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_base_vertices = base_out.vertices.detach().cpu().numpy().astype(float)
        raw_base_joints = base_out.joints.detach().cpu().numpy().astype(float)
    raw_vertex_errors = [np.linalg.norm(raw_base_vertices[i] - rows[i].raw_vertices_world, axis=1) for i in range(b)]
    raw_joint_errors = [np.linalg.norm(raw_base_joints[i] - rows[i].raw_joints_world, axis=1) for i in range(b)]
    max_replay_median = max(float(np.median(e)) for e in raw_vertex_errors) if raw_vertex_errors else float("inf")
    max_joint_replay_median = max(float(np.median(e)) for e in raw_joint_errors) if raw_joint_errors else float("inf")
    replay_ok = max_replay_median <= 1.0e-5 and max_joint_replay_median <= 1.0e-5

    pose_delta = torch.zeros((b, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([pose_delta], lr=0.25, max_iter=int(args.max_optimizer_iterations), line_search_fn="strong_wolfe")

    current_vertices_t = [torch.tensor(r.current_vertices_world, dtype=torch.float32, device=device) for r in rows]
    current_joints_t = [torch.tensor(r.current_joints_world, dtype=torch.float32, device=device) for r in rows]
    raw_base_vertices_t = torch.tensor(raw_base_vertices, dtype=torch.float32, device=device)
    raw_base_joints_t = torch.tensor(raw_base_joints, dtype=torch.float32, device=device)
    sim_scale_t = torch.tensor([r.similarity_scale for r in rows], dtype=torch.float32, device=device).reshape(b, 1, 1)
    sim_rot_t = torch.tensor(np.stack([r.similarity_rotation_raw_to_current for r in rows]), dtype=torch.float32, device=device)

    # Projection/depth compatibility is measured on joints; object residuals use constrained vertices.
    base_uv: list[torch.Tensor | None] = []
    base_depth: list[torch.Tensor] = []
    intr_t: list[torch.Tensor | None] = []
    r_c2w_t: list[torch.Tensor] = []
    t_c2w_t: list[torch.Tensor] = []
    for row in rows:
        intr = frame_intrinsics(row.frame, row.hand_side)
        intr_t.append(None if intr is None else torch.tensor(intr, dtype=torch.float32, device=device))
        r_c2w, t_c2w = frame_camera_pose(row.frame)
        r_c2w_t.append(torch.tensor(r_c2w, dtype=torch.float32, device=device))
        t_c2w_t.append(torch.tensor(t_c2w, dtype=torch.float32, device=device))
        uv_np = project_world(row.current_joints_world, row.frame, row.hand_side)
        base_uv.append(None if uv_np is None else torch.tensor(uv_np, dtype=torch.float32, device=device))
        base_cam = world_to_camera(row.current_joints_world, row.frame)
        base_depth.append(torch.tensor(base_cam[:, 2], dtype=torch.float32, device=device))

    def current_hypothesis() -> tuple[torch.Tensor, torch.Tensor]:
        new_pose = rotvec_to_matrix(pose_delta) @ base_pose_mat
        out = model(global_orient=base_root_mat, hand_pose=new_pose, betas=betas, transl=trans, return_verts=True, pose2rot=False)
        raw_delta_vertices = out.vertices - raw_base_vertices_t
        raw_delta_joints = out.joints - raw_base_joints_t
        mapped_delta_vertices = sim_scale_t * torch.matmul(raw_delta_vertices, sim_rot_t.transpose(1, 2))
        mapped_delta_joints = sim_scale_t * torch.matmul(raw_delta_joints, sim_rot_t.transpose(1, 2))
        current_vertices = torch.stack(current_vertices_t, dim=0) + mapped_delta_vertices
        current_joints = torch.stack(current_joints_t, dim=0) + mapped_delta_joints
        return current_vertices, current_joints

    def project_torch(points_world: torch.Tensor, row_i: int) -> torch.Tensor | None:
        intr = intr_t[row_i]
        if intr is None:
            return None
        cam = torch.matmul(points_world - t_c2w_t[row_i].reshape(1, 3), r_c2w_t[row_i])
        z = cam[:, 2].clamp_min(1.0e-5)
        fx, fy, cx, cy = intr
        return torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], dim=-1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        hyp_vertices, hyp_joints = current_hypothesis()
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = loss + float(args.pose_prior_weight) * torch.mean(pose_delta * pose_delta)
        if b > 1:
            vel = pose_delta[1:] - pose_delta[:-1]
            loss = loss + float(args.smooth_weight) * torch.mean(vel * vel)
        if b > 2:
            acc = pose_delta[2:] - 2.0 * pose_delta[1:-1] + pose_delta[:-2]
            loss = loss + float(args.accel_weight) * torch.mean(acc * acc)
        for i, row in enumerate(rows):
            if len(row.constraint_indices) > 0:
                ids = torch.tensor(row.constraint_indices, dtype=torch.long, device=device)
                normals = torch.tensor(row.constraint_normals_world, dtype=torch.float32, device=device)
                depths = torch.tensor(row.constraint_depths_m, dtype=torch.float32, device=device)
                moved = hyp_vertices[i, ids] - current_vertices_t[i][ids]
                residual = torch.relu(depths - torch.sum(normals * moved, dim=1))
                loss = loss + float(args.penetration_weight) * torch.mean(residual * residual)
            uv = project_torch(hyp_joints[i], i)
            if uv is not None and base_uv[i] is not None:
                shift = torch.linalg.norm(uv - base_uv[i], dim=1)
                excess = torch.relu(shift - float(args.visible_shift_limit_px))
                loss = loss + float(args.visible_hinge_weight) * torch.mean((excess / max(1.0, float(args.visible_shift_limit_px))) ** 2)
            cam = torch.matmul(hyp_joints[i] - t_c2w_t[i].reshape(1, 3), r_c2w_t[i])
            depth_shift = torch.abs(cam[:, 2] - base_depth[i])
            depth_excess = torch.relu(depth_shift - float(args.depth_shift_limit_m))
            loss = loss + float(args.depth_hinge_weight) * torch.mean(depth_excess * depth_excess)
        loss.backward()
        return loss

    if replay_ok:
        optimizer.step(closure)
    with torch.no_grad():
        hyp_vertices_t, hyp_joints_t = current_hypothesis()
        pose_delta_np = pose_delta.detach().cpu().numpy().astype(float)
        hyp_vertices = hyp_vertices_t.detach().cpu().numpy().astype(float)
        hyp_joints = hyp_joints_t.detach().cpu().numpy().astype(float)

    frame_states: list[dict[str, Any]] = []
    state_counter: Counter[str] = Counter()
    residual_max_values: list[float] = []
    residual_rms_values: list[float] = []
    visible_shift_values: list[float] = []
    depth_shift_values: list[float] = []
    pose_delta_values: list[float] = []
    similarity_values: list[float] = []
    render_ids = sample_ids(778, int(args.sample_vertex_count_for_render))
    for i, row in enumerate(rows):
        if len(row.constraint_indices) > 0:
            moved_np = hyp_vertices[i, row.constraint_indices] - row.current_vertices_world[row.constraint_indices]
            residual = np.maximum(0.0, row.constraint_depths_m - np.sum(row.constraint_normals_world * moved_np, axis=1))
        else:
            residual = np.zeros((0,), dtype=float)
        residual_max = float(np.max(residual)) if residual.size else 0.0
        residual_rms = float(math.sqrt(float(np.mean(residual * residual)))) if residual.size else 0.0
        uv0 = project_world(row.current_joints_world, row.frame, row.hand_side)
        uv1 = project_world(hyp_joints[i], row.frame, row.hand_side)
        if uv0 is not None and uv1 is not None:
            visible_shift = np.linalg.norm(uv1 - uv0, axis=1)
            visible_shift_max: float | None = float(np.max(visible_shift))
            visible_shift_median: float | None = float(np.median(visible_shift))
        else:
            visible_shift = np.zeros((0,), dtype=float)
            visible_shift_max = None
            visible_shift_median = None
        cam0 = world_to_camera(row.current_joints_world, row.frame)
        cam1 = world_to_camera(hyp_joints[i], row.frame)
        depth_shift = np.abs(cam1[:, 2] - cam0[:, 2])
        depth_shift_max = float(np.max(depth_shift))
        pose_norms = np.linalg.norm(pose_delta_np[i], axis=1)
        pose_delta_max = float(np.max(pose_norms))
        residual_ok = residual_max <= float(args.accepted_residual_m)
        visible_ok = visible_shift_max is not None and visible_shift_max <= float(args.visible_shift_limit_px)
        depth_ok = depth_shift_max <= float(args.depth_shift_limit_m)
        pose_ok = pose_delta_max <= float(args.max_pose_delta_rad)
        hidden_ok = row.hidden_volume_state == "observed_depth_support_with_hidden_uncertainty"
        if not replay_ok:
            temporal_state = "articulated_mano_replay_failed_unoptimized"
        elif not residual_ok:
            temporal_state = "unresolved_residual_penetration_after_temporal_articulated_mano"
        elif not visible_ok:
            temporal_state = "unresolved_visible_2d_shift_after_temporal_articulated_mano"
        elif not depth_ok:
            temporal_state = "unresolved_depth_shift_after_temporal_articulated_mano"
        elif not pose_ok:
            temporal_state = "unresolved_pose_delta_bound_after_temporal_articulated_mano"
        elif hidden_ok:
            temporal_state = "bounded_articulated_mano_candidate_hidden_volume_unaccepted"
        else:
            temporal_state = "bounded_articulated_mano_candidate_hidden_volume_quarantined"
        state_counter[temporal_state] += 1
        residual_max_values.append(residual_max)
        residual_rms_values.append(residual_rms)
        if visible_shift_max is not None:
            visible_shift_values.append(visible_shift_max)
        depth_shift_values.append(depth_shift_max)
        pose_delta_values.append(pose_delta_max)
        similarity_values.append(float(row.similarity_error_median_m))
        frame_states.append(
            {
                "frame_idx": int(row.frame_idx),
                "hand_side": row.hand_side,
                "source_hawor_npz": str(row.source_hawor_npz),
                "source_frame_index": int(row.source_frame_index),
                "mano_parameterization": "HaWoR_axis_angle_replayed_by_WiLoR_MANO_rotation_matrices_raw_hand_pose",
                "optimized_hand_pose_delta_axis_angle_rad": pose_delta_np[i].reshape(-1).astype(float).tolist(),
                "optimized_joints_world_m": hyp_joints[i].astype(float).tolist(),
                "optimized_vertices_world_sample_m": hyp_vertices[i, render_ids].astype(float).tolist(),
                "optimized_vertices_sample_ids": render_ids.astype(int).tolist(),
                "raw_replay_vertex_error_m": {
                    "median": float(np.median(raw_vertex_errors[i])),
                    "p95": float(np.percentile(raw_vertex_errors[i], 95)),
                    "max": float(np.max(raw_vertex_errors[i])),
                },
                "raw_replay_joint_error_m": {
                    "median": float(np.median(raw_joint_errors[i])),
                    "p95": float(np.percentile(raw_joint_errors[i], 95)),
                    "max": float(np.max(raw_joint_errors[i])),
                },
                "raw_to_current_similarity_error_m": {
                    "median": float(row.similarity_error_median_m),
                    "p95": float(row.similarity_error_p95_m),
                    "scope": "used_only_to_map_MANO_surface_deltas; zero_state_anchored_to_current_V18_bridge_vertices",
                },
                "residual_penetration_after_articulated_mano_m": {
                    "max": residual_max,
                    "rms": residual_rms,
                    "active_constraint_count": int(np.count_nonzero(residual > 0.0)),
                    "constraint_count": int(len(row.constraint_depths_m)),
                    "constraint_clipped_count": int(row.constraint_clipped_count),
                },
                "residual_penetration_after_translation_m": {
                    "max": residual_max,
                    "rms": residual_rms,
                    "active_constraint_count": int(np.count_nonzero(residual > 0.0)),
                    "constraint_count": int(len(row.constraint_depths_m)),
                    "constraint_clipped_count": int(row.constraint_clipped_count),
                },
                "visible_joint_shift_px": {
                    "state": "evaluated" if visible_shift_max is not None else "missing_intrinsics",
                    "count": int(len(visible_shift)),
                    "median": visible_shift_median,
                    "max": visible_shift_max,
                },
                "joint_camera_depth_shift_m": {
                    "count": int(len(depth_shift)),
                    "median": float(np.median(depth_shift)),
                    "p95": float(np.percentile(depth_shift, 95)),
                    "max": depth_shift_max,
                },
                "pose_delta_norm_rad": {
                    "count": int(len(pose_norms)),
                    "median": float(np.median(pose_norms)),
                    "max": pose_delta_max,
                },
                "hidden_volume_state": row.hidden_volume_state,
                "temporal_mano_state": temporal_state,
                "coordinate_correction_accepted": False,
            }
        )

    if not replay_ok:
        interval_state = "articulated_mano_replay_failed"
    elif any("residual_penetration" in state for state in state_counter):
        interval_state = "articulated_mano_trajectory_blocked_by_residual_penetration"
    elif any("visible_2d" in state for state in state_counter):
        interval_state = "articulated_mano_trajectory_blocked_by_visible_2d_shift"
    elif any("depth_shift" in state for state in state_counter):
        interval_state = "articulated_mano_trajectory_blocked_by_depth_shift"
    elif any("pose_delta_bound" in state for state in state_counter):
        interval_state = "articulated_mano_trajectory_blocked_by_pose_delta_bound"
    else:
        interval_state = "bounded_articulated_mano_trajectory_candidate_hidden_volume_unaccepted"
    blockers: list[str] = []
    if not replay_ok:
        blockers.append(f"{rows[0].hand_side}_hand_mano_replay_not_exact_enough")
    if residual_max_values and max(residual_max_values) > float(args.accepted_residual_m):
        blockers.append("temporal_articulated_mano_leaves_residual_penetration")
    if visible_shift_values and max(visible_shift_values) > float(args.visible_shift_limit_px):
        blockers.append("temporal_articulated_mano_exceeds_visible_2d_shift_limit")
    if depth_shift_values and max(depth_shift_values) > float(args.depth_shift_limit_m):
        blockers.append("temporal_articulated_mano_exceeds_depth_shift_limit")
    if pose_delta_values and max(pose_delta_values) > float(args.max_pose_delta_rad):
        blockers.append("temporal_articulated_mano_exceeds_pose_delta_bound")
    hidden_states = sorted(set(row.hidden_volume_state for row in rows))
    if any(state != "observed_depth_support_with_hidden_uncertainty" for state in hidden_states):
        blockers.append("object_hidden_volume_free_space_conflicted_or_unvalidated")
    blockers.append("hidden_inferred_volume_not_accepted_for_coordinate_level_nonpenetration_proof")
    interval = {
        "hand_side": rows[0].hand_side,
        "start_frame": int(rows[0].frame_idx),
        "end_frame": int(rows[-1].frame_idx),
        "frame_count": int(len(rows)),
        "optimizer_ran": bool(replay_ok),
        "raw_replay_vertex_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in raw_vertex_errors], dtype=float)),
        "raw_replay_joint_error_median_m": numeric_summary(np.asarray([float(np.median(e)) for e in raw_joint_errors], dtype=float)),
        "raw_to_current_similarity_error_median_m": numeric_summary(np.asarray(similarity_values, dtype=float)),
        "temporal_mano_interval_state": interval_state,
        "coordinate_correction_accepted": False,
        "state_counts": dict(state_counter),
        "residual_penetration_after_articulated_mano_m": numeric_summary(np.asarray(residual_max_values, dtype=float)),
        "residual_penetration_after_translation_m": numeric_summary(np.asarray(residual_max_values, dtype=float)),
        "residual_rms_after_articulated_mano_m": numeric_summary(np.asarray(residual_rms_values, dtype=float)),
        "visible_joint_shift_max_px": numeric_summary(np.asarray(visible_shift_values, dtype=float)),
        "joint_camera_depth_shift_max_m": numeric_summary(np.asarray(depth_shift_values, dtype=float)),
        "pose_delta_max_rad": numeric_summary(np.asarray(pose_delta_values, dtype=float)),
        "hidden_volume_states": hidden_states,
        "blocking_mechanisms": blockers,
    }
    return interval, frame_states


def build_rows(args: argparse.Namespace) -> tuple[dict[str, list[ReplayFrame]], list[dict[str, Any]], dict[str, Any]]:
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    poses = pose_map(pose_report)
    hidden_rows = load_hidden_validation(args.hidden_volume_validation)
    sign_mesh = load_mesh(args.sign_mesh)
    sign_scene = o3d.t.geometry.RaycastingScene()
    sign_scene.add_triangles(
        o3d.core.Tensor(np.asarray(sign_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(sign_mesh.faces, dtype=np.uint32)),
    )
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    rows_by_side: dict[str, list[ReplayFrame]] = {}
    skipped: list[dict[str, Any]] = []
    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame["frame_idx"])
        pose = poses.get(frame_idx)
        if pose is None:
            continue
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            replay_row, skip = build_replay_frame(
                frame_idx=frame_idx,
                frame=frame,
                hand=hand,
                side=side,
                pose=pose,
                sign_scene=sign_scene,
                bridge_cache=bridge_cache,
                source_cache=source_cache,
                hidden_rows=hidden_rows,
                args=args,
            )
            if skip is not None:
                skipped.append(skip)
                continue
            if replay_row is None:
                continue
            if int(len(replay_row.penetration_depths_all_m)) <= 0:
                continue
            rows_by_side.setdefault(side, []).append(replay_row)
    metadata = {
        "annotation_frame_count": len(as_list(annotations.get("frames"))),
        "pose_frame_count": len(poses),
        "hidden_volume_validation_frame_count": len(hidden_rows),
    }
    return rows_by_side, skipped, metadata


def ineligible_intervals(
    rows: list[ReplayFrame],
    *,
    side: str,
    temporal_state: str,
    reason: str,
    blocker: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intervals: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    rows_by_frame = {row.frame_idx: row for row in rows}
    for start, end in contiguous_segments(list(rows_by_frame)):
        seg = [rows_by_frame[idx] for idx in range(start, end + 1) if idx in rows_by_frame]
        if not seg:
            continue
        for row in seg:
            state = {
                "frame_idx": int(row.frame_idx),
                "hand_side": side,
                "temporal_mano_state": temporal_state,
                "coordinate_correction_accepted": False,
                "reason": reason,
                "residual_penetration_after_articulated_mano_m": {
                    "max": float(np.max(row.penetration_depths_all_m)) if len(row.penetration_depths_all_m) else 0.0,
                    "constraint_count": int(len(row.constraint_depths_m)),
                },
                "residual_penetration_after_translation_m": {
                    "max": float(np.max(row.penetration_depths_all_m)) if len(row.penetration_depths_all_m) else 0.0,
                    "constraint_count": int(len(row.constraint_depths_m)),
                },
                "hidden_volume_state": row.hidden_volume_state,
            }
            states.append(state)
        intervals.append(
            {
                "hand_side": side,
                "start_frame": int(start),
                "end_frame": int(end),
                "frame_count": int(len(seg)),
                "temporal_mano_interval_state": temporal_state,
                "coordinate_correction_accepted": False,
                "state_counts": {temporal_state: int(len(seg))},
                "blocking_mechanisms": [blocker],
                "residual_penetration_initial_m": numeric_summary(np.concatenate([row.penetration_depths_all_m for row in seg]) if seg else np.asarray([], dtype=float)),
            }
        )
    return intervals, states


def main() -> None:
    args = parse_args()
    patch_legacy_mano_loader()
    mano_right_path = args.wilor_mano_right if args.wilor_mano_right is not None else args.wilor_root / "mano_data" / "MANO_RIGHT.pkl"
    if not mano_right_path.exists():
        raise FileNotFoundError(f"missing MANO_RIGHT model: {mano_right_path}")
    mano_cls = load_wilor_mano_class(args.wilor_root)
    device = torch.device(args.device)
    models: dict[str, Any] = {
        "right": mano_cls(model_path=str(mano_right_path), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    }
    model_paths: dict[str, str] = {"right": str(mano_right_path)}
    left_model_status = "not_requested_or_not_provided"
    mano_left_path = args.wilor_mano_left
    if mano_left_path is not None:
        if not mano_left_path.exists():
            left_model_status = "missing_mano_left_path"
        else:
            left_model = mano_cls(model_path=str(mano_left_path), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
            if bool(args.hawor_left_shapedirs_x_fix):
                with torch.no_grad():
                    left_model.shapedirs[:, 0, :] *= -1
            models["left"] = left_model
            model_paths["left"] = str(mano_left_path)
            left_model_status = "loaded_with_hawor_shapedirs_x_fix" if bool(args.hawor_left_shapedirs_x_fix) else "loaded_without_hawor_shapedirs_x_fix"
    for model in models.values():
        model.eval()

    completion = load_json(args.completion_report)
    rows_by_side, skipped, metadata = build_rows(args)
    intervals: list[dict[str, Any]] = []
    per_frame_states: list[dict[str, Any]] = []
    eligible_sides = {"right"} if args.eligible_side == "right" else {"left", "right"}
    optimized_frame_count_by_side: Counter[str] = Counter()
    for side in sorted(rows_by_side):
        side_rows = sorted(rows_by_side.get(side, []), key=lambda row: row.frame_idx)
        if side not in eligible_sides:
            ineligible, states = ineligible_intervals(
                side_rows,
                side=side,
                temporal_state="articulated_mano_not_requested_for_side",
                reason=f"{side} optimization was not requested by --eligible-side={args.eligible_side}",
                blocker="side_not_requested_for_articulated_mano_optimization",
            )
            intervals.extend(ineligible)
            per_frame_states.extend(states)
            continue
        model = models.get(side)
        if model is None:
            state_name = "articulated_mano_replay_ineligible_missing_left_mano_model" if side == "left" else "articulated_mano_replay_ineligible_missing_mano_model"
            reason = (
                "MANO_LEFT.pkl was not supplied or could not be loaded with the HaWoR shapedirs-x replay convention"
                if side == "left"
                else "MANO model for this side was not supplied"
            )
            ineligible, states = ineligible_intervals(
                side_rows,
                side=side,
                temporal_state=state_name,
                reason=reason,
                blocker="missing_or_unloaded_side_specific_mano_model",
            )
            intervals.extend(ineligible)
            per_frame_states.extend(states)
            continue
        rows_by_frame = {row.frame_idx: row for row in side_rows}
        side_optimized = 0
        for start, end in contiguous_segments(list(rows_by_frame)):
            seg = [rows_by_frame[idx] for idx in range(start, end + 1) if idx in rows_by_frame]
            if not seg:
                continue
            if args.max_articulated_frames is not None and side_optimized >= int(args.max_articulated_frames):
                for row in seg:
                    per_frame_states.append(
                        {
                            "frame_idx": int(row.frame_idx),
                            "hand_side": side,
                            "temporal_mano_state": "articulated_mano_not_run_max_frame_budget",
                            "coordinate_correction_accepted": False,
                        }
                    )
                intervals.append(
                    {
                        "hand_side": side,
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "frame_count": int(len(seg)),
                        "temporal_mano_interval_state": "articulated_mano_not_run_max_frame_budget",
                        "coordinate_correction_accepted": False,
                        "state_counts": {"articulated_mano_not_run_max_frame_budget": int(len(seg))},
                        "blocking_mechanisms": ["max_articulated_frames_budget"],
                    }
                )
                continue
            if args.max_articulated_frames is not None:
                remaining = int(args.max_articulated_frames) - side_optimized
                seg = seg[:remaining]
            interval, states = optimize_segment(model=model, rows=seg, args=args, device=device)
            interval["interval_id"] = f"{side}_{seg[0].frame_idx:04d}_{seg[-1].frame_idx:04d}"
            for state in states:
                state["interval_id"] = interval["interval_id"]
            intervals.append(interval)
            per_frame_states.extend(states)
            side_optimized += len(seg)
            optimized_frame_count_by_side[side] += len(seg)
    summary_counts = Counter(interval["temporal_mano_interval_state"] for interval in intervals)
    frame_counts = Counter(state.get("temporal_mano_state", "unknown") for state in per_frame_states)
    report = {
        "method": "build_v18_temporal_mano_articulated_interval_state",
        "status": "ok",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Interval-level articulated MANO hand-pose mechanism test. Side-specific HaWoR MANO replay must be exact "
            "before pose-delta hypotheses are eligible; left replay uses MANO_LEFT with the documented HaWoR shapedirs-x fix when supplied. "
            "Coordinate corrections remain unaccepted unless residual, visible/depth compatibility, temporal coherence, and hidden-volume evidence all pass."
        ),
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report),
            "sign_mesh": str(args.sign_mesh),
            "hidden_volume_validation": str(args.hidden_volume_validation) if args.hidden_volume_validation else None,
            "wilor_mano_right": str(mano_right_path),
            "wilor_mano_left": str(mano_left_path) if mano_left_path is not None else None,
            "loaded_mano_models": model_paths,
            "left_model_status": left_model_status,
        },
        "object_hypothesis_scope": completion.get("claim_scope") if isinstance(completion, dict) else None,
        "parameters": {
            "eligible_side": str(args.eligible_side),
            "hawor_left_shapedirs_x_fix": bool(args.hawor_left_shapedirs_x_fix),
            "max_constraints_per_frame": int(args.max_constraints_per_frame),
            "accepted_residual_m": float(args.accepted_residual_m),
            "visible_shift_limit_px": float(args.visible_shift_limit_px),
            "depth_shift_limit_m": float(args.depth_shift_limit_m),
            "max_pose_delta_rad": float(args.max_pose_delta_rad),
            "zero_state": "current_V18_bridge_vertices; MANO pose deltas mapped as raw-HaWoR displacement through row-wise similarity",
        },
        "summary": {
            "interval_count": int(len(intervals)),
            "per_frame_state_count": int(len(per_frame_states)),
            "optimized_frame_count_by_side": {k: int(v) for k, v in sorted(optimized_frame_count_by_side.items())},
            "optimized_right_frame_count": int(optimized_frame_count_by_side.get("right", 0)),
            "optimized_left_frame_count": int(optimized_frame_count_by_side.get("left", 0)),
            "interval_state_counts": dict(summary_counts),
            "per_frame_state_counts": dict(frame_counts),
            "coordinate_correction_accepted": False,
            **metadata,
        },
        "skipped": skipped[:200],
        "intervals": intervals,
        "per_frame_states": per_frame_states,
        "physical_conclusion": (
            "This artifact tests finger/wrist articulation as a causal mechanism for the remaining MANO/object conflicts. "
            "Any cyan articulated hypothesis is a bounded or falsified MANO hand-pose perturbation, not an accepted correction, "
            "because current object hidden volumes remain quarantined and each side must still pass residual, visible/depth, and temporal checks."
        ),
    }
    out_dir = args.output_dir / str(args.case)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v18_temporal_mano_articulated_interval_state.json"
    write_json(out_path, report)
    print(json.dumps({"output": str(out_path), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
