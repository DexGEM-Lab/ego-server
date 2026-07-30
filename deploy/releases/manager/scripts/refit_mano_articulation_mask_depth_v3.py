#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame
from fit_mano_to_hand_mask_depth_v3 import distance_map, mask_center, mask_depth_stats, mask_for_frame, sample_mask_depth
from optimize_hand_translation_contact_v3 import source_to_world
from refit_mano_pose_contact_v3 import (
    apply_side_sign,
    hand_span_torch,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
    project_torch,
    robust_l1,
    rotvec_to_matrix,
    side_sign,
    similarity_from_to,
)


HAND_EDGES = np.asarray(
    [
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
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class ArticulationInput:
    frame_idx: int
    hand_index: int
    side: str
    track_id: str
    intrinsics: torch.Tensor
    metric_depth: torch.Tensor
    depth_valid_mask: torch.Tensor
    mask: np.ndarray
    mask_depth_median_m: float
    mask_depth_iqr_m: float
    mask_center_xy: np.ndarray
    mask_distance: torch.Tensor
    mask_inside_distance: torch.Tensor
    base_global_orient: torch.Tensor
    base_hand_pose: torch.Tensor
    betas: torch.Tensor
    base_cam_t: torch.Tensor
    base_local_joints: np.ndarray
    base_local_vertices: np.ndarray
    base_source_joints: np.ndarray
    base_source_vertices: np.ndarray
    base_joints2d: np.ndarray
    rtmlib_joints2d: np.ndarray | None
    rtmlib_scores: np.ndarray | None
    T_world_camera: np.ndarray


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def frame_map(annotations: dict) -> dict[int, dict]:
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations must contain a non-empty frames list")
    out = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate frame_idx {frame_idx}")
        out[frame_idx] = frame
    return out


def source_vertex_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    raise RuntimeError("articulation refit requires full vertices_source_camera_m")


def local_vertex_key(hand: dict) -> str:
    if "vertices_camera" in hand:
        return "vertices_camera"
    raise RuntimeError("articulation refit requires full vertices_camera")


def project_torch_to_mask(points: torch.Tensor, intrinsics: torch.Tensor, depth_shape: tuple[int, int], source_size: tuple[int, int]) -> torch.Tensor:
    uv = project_torch(points, intrinsics)
    scale_x = float(depth_shape[1]) / float(source_size[0])
    scale_y = float(depth_shape[0]) / float(source_size[1])
    return torch.stack([uv[..., 0] * scale_x, uv[..., 1] * scale_y], dim=-1)


def bilinear_sample(image: torch.Tensor, xy: torch.Tensor, invalid_value: float) -> torch.Tensor:
    height, width = image.shape
    x = xy[:, 0]
    y = xy[:, 1]
    valid = torch.isfinite(x) & torch.isfinite(y) & (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)
    x0 = torch.floor(x.clamp(0, width - 1)).long()
    y0 = torch.floor(y.clamp(0, height - 1)).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x.clamp(0, width - 1) - x0.float()
    wy = y.clamp(0, height - 1) - y0.float()
    values = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    return torch.where(valid, values, torch.full_like(values, float(invalid_value)))


def depth_error_summary_for_vertices(vertices: np.ndarray, item: ArticulationInput, args: argparse.Namespace, vertex_ids: np.ndarray) -> dict:
    source_size = (int(args.source_width), int(args.source_height))
    depth = item.metric_depth.cpu().numpy()
    uv = project_points(vertices[vertex_ids], item.intrinsics.cpu().numpy())
    targets = sample_mask_depth(depth, uv, source_size)
    valid = np.isfinite(targets) & (targets > float(args.min_depth_m))
    if not np.any(valid):
        return {
            "sampled_vertex_minus_metric_depth_median_m": None,
            "sampled_vertex_minus_metric_depth_p95_abs_m": None,
            "sampled_vertex_depth_valid_count": 0,
            "interior_vertex_minus_metric_depth_p95_abs_m": None,
            "interior_vertex_minus_metric_depth_median_abs_m": None,
            "interior_vertex_depth_valid_count": 0,
            "interior_vertex_depth_margin_px": float(args.interior_depth_margin_px),
        }
    err = vertices[vertex_ids][valid, 2] - targets[valid]
    h, w = item.mask.shape
    xy = np.rint(uv).astype(int)
    in_image = (
        valid
        & np.isfinite(uv).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )
    inside = item.mask_inside_distance.cpu().numpy()
    interior = np.zeros(len(vertex_ids), dtype=bool)
    if np.any(in_image):
        interior[in_image] = inside[xy[in_image, 1], xy[in_image, 0]] >= float(args.interior_depth_margin_px)
    interior_valid = interior & valid
    if int(np.count_nonzero(interior_valid)) >= int(args.min_interior_depth_vertices):
        interior_err = np.abs(vertices[vertex_ids][interior_valid, 2] - targets[interior_valid])
        interior_p95 = float(np.percentile(interior_err, 95.0))
        interior_median = float(np.median(interior_err))
        interior_count = int(interior_err.size)
    else:
        interior_p95 = None
        interior_median = None
        interior_count = int(np.count_nonzero(interior_valid))
    return {
        "sampled_vertex_minus_metric_depth_median_m": float(np.median(err)),
        "sampled_vertex_minus_metric_depth_p95_abs_m": float(np.percentile(np.abs(err), 95.0)),
        "sampled_vertex_depth_valid_count": int(np.count_nonzero(valid)),
        "interior_vertex_minus_metric_depth_p95_abs_m": interior_p95,
        "interior_vertex_minus_metric_depth_median_abs_m": interior_median,
        "interior_vertex_depth_valid_count": interior_count,
        "interior_vertex_depth_margin_px": float(args.interior_depth_margin_px),
    }


def sample_ids(count: int, max_count: int) -> np.ndarray:
    if count <= 0:
        raise RuntimeError("cannot sample empty vertex set")
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_count, dtype=np.int64)


def inside_distance_map(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)


def candidate_score(hand: dict, track_id: str, side: str) -> tuple[int, int]:
    hand_track = str(hand.get("track_id", ""))
    hand_side = str(hand.get("side", ""))
    return (0 if hand_track == track_id else 1, 0 if hand_side == side else 1)


def load_rtmlib_targets(args: argparse.Namespace) -> dict[int, dict]:
    if args.rtmlib_json is None:
        return {}
    if args.rtmlib_prompts is None:
        raise RuntimeError("--rtmlib-prompts is required when --rtmlib-json is provided")
    rtm = load_json(args.rtmlib_json)
    prompt = load_json(args.rtmlib_prompts)
    frames = rtm.get("frames")
    diagnostics = prompt.get("diagnostics")
    if not isinstance(frames, list) or not isinstance(diagnostics, list):
        raise RuntimeError("RTMLib target construction requires frames and prompt diagnostics")
    selected_by_frame: dict[int, int] = {}
    for row in diagnostics:
        frame_idx = int(row["frame_idx"])
        if int(args.frame_start) <= frame_idx <= int(args.frame_end):
            selected_by_frame[frame_idx] = int(row["selected_rtmlib_hand_idx"])
    targets: dict[int, dict] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        if frame_idx not in selected_by_frame:
            raise RuntimeError(f"RTMLib prompt diagnostics missing selected hand for frame {frame_idx}")
        selected_idx = selected_by_frame[frame_idx]
        hands = frame.get("hands")
        if not isinstance(hands, list):
            raise RuntimeError(f"RTMLib frame {frame_idx} hands field must be a list")
        matches = [hand for hand in hands if int(hand.get("hand_idx", -1)) == selected_idx]
        if len(matches) != 1:
            raise RuntimeError(f"RTMLib frame {frame_idx} has {len(matches)} hands with selected index {selected_idx}")
        hand = matches[0]
        keypoints = np.asarray(hand.get("keypoints", []), dtype=float)
        scores = np.asarray(hand.get("scores", []), dtype=float)
        if keypoints.shape != (21, 2) or scores.shape != (21,):
            raise RuntimeError(f"RTMLib frame {frame_idx} selected hand has invalid keypoint fields")
        valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores) & (scores >= float(args.rtmlib_min_score))
        if int(np.count_nonzero(valid)) < int(args.rtmlib_min_keypoints):
            raise RuntimeError(
                f"RTMLib frame {frame_idx} selected hand has {int(np.count_nonzero(valid))} valid keypoints; "
                f"required {int(args.rtmlib_min_keypoints)}"
            )
        targets[frame_idx] = {
            "keypoints": keypoints,
            "scores": scores,
            "hand_idx": selected_idx,
            "mean_score": float(hand.get("mean_score", np.mean(scores[valid]))),
        }
    missing = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))) if idx not in targets]
    if missing:
        raise RuntimeError(f"RTMLib targets missing frames {missing[:20]}")
    return targets


def source_size_for_frame(frame: dict, args: argparse.Namespace) -> tuple[int, int]:
    obj = frame.get("object", {})
    size = np.asarray(obj.get("source_image_size", []), dtype=float)
    if size.shape == (2,) and np.isfinite(size).all() and np.all(size > 0):
        return int(size[0]), int(size[1])
    return int(args.source_width), int(args.source_height)


def initial_cam_t_from_mask(local_joints: np.ndarray, mask_center_xy: np.ndarray, mask_depth: float, intr: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intr.astype(float)
    center = np.asarray(
        [
            (float(mask_center_xy[0]) - cx) * float(mask_depth) / fx,
            (float(mask_center_xy[1]) - cy) * float(mask_depth) / fy,
            float(mask_depth),
        ],
        dtype=float,
    )
    return center - np.median(local_joints, axis=0)


def build_inputs(args: argparse.Namespace) -> tuple[dict, list[ArticulationInput], list[dict]]:
    annotations = load_json(args.annotations)
    track = load_json(args.mask_track)
    rtmlib_targets = load_rtmlib_targets(args)
    frames = frame_map(annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    frame_to_depth = {int(idx): i for i, idx in enumerate(depth_indices)}
    inputs: list[ArticulationInput] = []
    skipped: list[dict] = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation_frame"})
            continue
        if frame_idx not in frame_to_depth:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_depth_frame"})
            continue
        source_size = source_size_for_frame(frame, args)
        mask = mask_for_frame(track, frame_idx, source_size, args)
        if mask is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_visible_mask"})
            continue
        depth = depth_frame(depths, frame_to_depth, frame_idx)
        try:
            mask_depth, mask_iqr, mask_depth_valid = mask_depth_stats(depth, mask, source_size)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        hands = list(frame.get("hands", []))
        order = sorted(range(len(hands)), key=lambda i: candidate_score(hands[i], args.track_id, args.side))
        for hand_idx in order[: max(1, int(args.max_hypotheses_per_frame))]:
            hand = hands[hand_idx]
            try:
                side = str(hand.get("side"))
                if args.side != "any" and side != args.side:
                    continue
                params = hand["mano_params"]
                global_orient = np.asarray(params["global_orient"], dtype=float)
                hand_pose = np.asarray(params["hand_pose"], dtype=float)
                betas = np.asarray(params["betas"], dtype=float)
                if global_orient.shape != (1, 3, 3) or hand_pose.shape != (15, 3, 3) or betas.shape != (10,):
                    raise RuntimeError("MANO params must contain rotation matrices and 10 betas")
                intr = np.asarray(hand["source_intrinsics"], dtype=float)
                local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                local_vertices = np.asarray(hand[local_vertex_key(hand)], dtype=float)
                source_joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                source_vertices = np.asarray(hand[source_vertex_key(hand)], dtype=float)
                joints2d = np.asarray(hand["joints2d"], dtype=float)
                target = rtmlib_targets.get(frame_idx)
                T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
                if intr.shape != (4,) or local_joints.shape != (21, 3) or source_joints.shape != (21, 3):
                    raise RuntimeError("invalid hand geometry fields")
                if local_vertices.ndim != 2 or local_vertices.shape[1] != 3 or source_vertices.shape != local_vertices.shape:
                    raise RuntimeError("invalid MANO vertices")
                if joints2d.shape != (21, 2) or T_world_camera.shape != (4, 4):
                    raise RuntimeError("invalid projection or camera fields")
                cam_t0 = initial_cam_t_from_mask(local_joints, mask_center(mask), mask_depth, intr)
                inputs.append(
                    ArticulationInput(
                        frame_idx=frame_idx,
                        hand_index=int(hand_idx),
                        side=side,
                        track_id=str(hand.get("track_id", "")),
                        intrinsics=torch.tensor(intr, dtype=torch.float32),
                        metric_depth=torch.tensor(np.nan_to_num(depth, nan=0.0), dtype=torch.float32),
                        depth_valid_mask=torch.tensor(mask_depth_valid, dtype=torch.bool),
                        mask=mask,
                        mask_depth_median_m=float(mask_depth),
                        mask_depth_iqr_m=float(mask_iqr),
                        mask_center_xy=mask_center(mask),
                        mask_distance=torch.tensor(distance_map(mask), dtype=torch.float32),
                        mask_inside_distance=torch.tensor(inside_distance_map(mask), dtype=torch.float32),
                        base_global_orient=torch.tensor(global_orient[None], dtype=torch.float32),
                        base_hand_pose=torch.tensor(hand_pose[None], dtype=torch.float32),
                        betas=torch.tensor(betas[None], dtype=torch.float32),
                        base_cam_t=torch.tensor(cam_t0[None], dtype=torch.float32),
                        base_local_joints=local_joints,
                        base_local_vertices=local_vertices,
                        base_source_joints=source_joints,
                        base_source_vertices=source_vertices,
                        base_joints2d=joints2d,
                        rtmlib_joints2d=None if target is None else np.asarray(target["keypoints"], dtype=float),
                        rtmlib_scores=None if target is None else np.asarray(target["scores"], dtype=float),
                        T_world_camera=T_world_camera,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_idx, "reason": str(exc)})
    if len(inputs) < int(args.min_observations):
        raise RuntimeError(f"insufficient articulation observations {len(inputs)}; skipped={skipped[:80]}")
    return annotations, inputs, skipped


def metric_rows(
    joints: np.ndarray,
    vertices: np.ndarray,
    item: ArticulationInput,
    args: argparse.Namespace,
    sampled_vertex_ids: np.ndarray,
) -> dict:
    uv_joints = project_points(joints, item.intrinsics.cpu().numpy())
    uv_vertices = project_points(vertices[sampled_vertex_ids], item.intrinsics.cpu().numpy())
    h, w = item.mask.shape
    x = np.clip(np.rint(uv_vertices[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(uv_vertices[:, 1]).astype(int), 0, h - 1)
    dist = item.mask_distance.cpu().numpy()[y, x]
    depth_summary = depth_error_summary_for_vertices(vertices, item, args, sampled_vertex_ids)
    joint_reproj = np.linalg.norm(uv_joints - item.base_joints2d, axis=1)
    rtmlib_reproj = None
    rtmlib_valid_keypoints = 0
    if item.rtmlib_joints2d is not None and item.rtmlib_scores is not None:
        valid_rtm = (
            np.isfinite(item.rtmlib_joints2d).all(axis=1)
            & np.isfinite(item.rtmlib_scores)
            & (item.rtmlib_scores >= float(args.rtmlib_min_score))
        )
        rtmlib_valid_keypoints = int(np.count_nonzero(valid_rtm))
        if rtmlib_valid_keypoints:
            rtmlib_reproj = np.linalg.norm(uv_joints[valid_rtm] - item.rtmlib_joints2d[valid_rtm], axis=1)
    return {
        "frame_idx": int(item.frame_idx),
        "hand_index": int(item.hand_index),
        "side": item.side,
        "track_id": item.track_id,
        "silhouette_inside_fraction": float(np.mean(dist <= 0.5)),
        "silhouette_distance_median_px": float(np.median(dist)),
        "silhouette_distance_p95_px": float(np.percentile(dist, 95.0)),
        "mano_minus_mask_depth_median_m": float(np.median(joints[:, 2]) - item.mask_depth_median_m),
        **depth_summary,
        "joint_reprojection_to_base_median_px": float(np.median(joint_reproj)),
        "joint_reprojection_to_base_p95_px": float(np.percentile(joint_reproj, 95.0)),
        "rtmlib_joint_reprojection_median_px": None if rtmlib_reproj is None else float(np.median(rtmlib_reproj)),
        "rtmlib_joint_reprojection_p95_px": None if rtmlib_reproj is None else float(np.percentile(rtmlib_reproj, 95.0)),
        "rtmlib_valid_keypoints": int(rtmlib_valid_keypoints),
        "hand_bone_m": float(hand_bone_scale_m(joints)),
        "hand_span_m": float(hand_span_torch(torch.tensor(joints[None], dtype=torch.float32))),
    }


def fit_one(model, item: ArticulationInput, args: argparse.Namespace) -> tuple[dict, dict]:
    device = torch.device(args.device)
    model = model.to(device)
    sign = side_sign(item.side)
    with torch.no_grad():
        base_out = model(
            global_orient=item.base_global_orient.to(device),
            hand_pose=item.base_hand_pose.to(device),
            betas=item.betas.to(device),
            return_verts=True,
            pose2rot=False,
        )
        base_vertices = apply_side_sign(base_out.vertices, sign)[0].cpu().numpy()
        base_joints = apply_side_sign(base_out.joints, sign)[0].cpu().numpy()
    local_scale, local_rotation, local_translation, vertex_error = similarity_from_to(base_vertices, item.base_local_vertices)
    aligned_joints = local_scale * (base_joints @ local_rotation.T) + local_translation[None, :]
    joint_error = np.linalg.norm(aligned_joints - item.base_local_joints, axis=1)
    if float(np.median(vertex_error)) > float(args.max_zero_state_vertex_error_m):
        raise RuntimeError(f"zero-state vertex mismatch {float(np.median(vertex_error)):.6f}m")
    if float(np.median(joint_error)) > float(args.max_zero_state_joint_error_m):
        raise RuntimeError(f"zero-state joint mismatch {float(np.median(joint_error)):.6f}m")

    local_rotation_t = torch.tensor(local_rotation, dtype=torch.float32, device=device)
    local_translation_t = torch.tensor(local_translation, dtype=torch.float32, device=device).reshape(1, 1, 3)
    local_scale_t = torch.tensor(float(local_scale), dtype=torch.float32, device=device)
    vertex_ids_np = sample_ids(int(item.base_local_vertices.shape[0]), int(args.max_sampled_vertices))
    depth_ids_np = sample_ids(int(item.base_local_vertices.shape[0]), int(args.max_depth_vertices))
    vertex_ids = torch.tensor(vertex_ids_np, dtype=torch.long, device=device)
    depth_ids = torch.tensor(depth_ids_np, dtype=torch.long, device=device)
    mask_distance = item.mask_distance.to(device)
    metric_depth = item.metric_depth.to(device)
    depth_valid = item.depth_valid_mask.to(device)
    intr = item.intrinsics.to(device)
    source_size = (int(args.source_width), int(args.source_height))
    depth_shape = tuple(int(x) for x in metric_depth.shape)

    pose_delta = torch.zeros((1, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    orient_delta = torch.zeros((1, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros_like(item.base_cam_t, device=device, requires_grad=True)
    log_scale = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pose_delta, orient_delta, trans_delta, log_scale], lr=float(args.lr))

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)
        global_orient = rotvec_to_matrix(orient_delta) @ item.base_global_orient.to(device)
        hand_pose = rotvec_to_matrix(pose_delta) @ item.base_hand_pose.to(device)
        out = model(global_orient=global_orient, hand_pose=hand_pose, betas=item.betas.to(device), return_verts=True, pose2rot=False)
        canonical_vertices = apply_side_sign(out.vertices, sign)
        canonical_joints = apply_side_sign(out.joints, sign)
        local_vertices = local_scale_t * torch.matmul(canonical_vertices, local_rotation_t.T) + local_translation_t
        local_joints = local_scale_t * torch.matmul(canonical_joints, local_rotation_t.T) + local_translation_t
        scale = torch.exp(log_scale).reshape(1, 1, 1)
        vertices = scale * local_vertices + item.base_cam_t[:, None, :].to(device) + trans_delta[:, None, :]
        joints = scale * local_joints + item.base_cam_t[:, None, :].to(device) + trans_delta[:, None, :]

        uv_mask = project_torch(vertices[0, vertex_ids], intr)
        dist = bilinear_sample(mask_distance, uv_mask, float(args.max_silhouette_distance_px)).clamp_max(float(args.max_silhouette_distance_px))
        silhouette_loss = robust_l1(dist / float(args.sigma_silhouette_px)).mean()

        uv_depth = project_torch_to_mask(vertices[0, depth_ids], intr, depth_shape, source_size)
        target_depth = bilinear_sample(metric_depth, uv_depth, 0.0)
        target_ok = bilinear_sample(depth_valid.float(), uv_depth, 0.0) > 0.5
        valid_vertex_depth = target_ok & (target_depth > float(args.min_depth_m))
        if bool(valid_vertex_depth.any()):
            depth_loss = robust_l1((vertices[0, depth_ids][valid_vertex_depth, 2] - target_depth[valid_vertex_depth]) / float(args.sigma_vertex_depth_m)).mean()
        else:
            depth_loss = torch.tensor(float(args.invalid_depth_penalty), dtype=torch.float32, device=device)

        uv_joints = project_torch(joints[0], intr)
        base_uv = torch.tensor(item.base_joints2d, dtype=torch.float32, device=device)
        joint_prior_loss = robust_l1((uv_joints - base_uv) / float(args.sigma_joint_prior_px)).mean()
        if item.rtmlib_joints2d is not None and item.rtmlib_scores is not None:
            target_uv = torch.tensor(item.rtmlib_joints2d, dtype=torch.float32, device=device)
            target_scores = torch.tensor(item.rtmlib_scores, dtype=torch.float32, device=device)
            valid_rtm = torch.isfinite(target_uv).all(dim=1) & torch.isfinite(target_scores) & (target_scores >= float(args.rtmlib_min_score))
            if int(valid_rtm.sum().detach().cpu()) < int(args.rtmlib_min_keypoints):
                keypoint_loss = torch.tensor(float(args.invalid_keypoint_penalty), dtype=torch.float32, device=device)
            else:
                weights = target_scores[valid_rtm].clamp_min(0.0)
                weights = weights / weights.mean().clamp_min(1.0e-6)
                per_joint = robust_l1((uv_joints[valid_rtm] - target_uv[valid_rtm]) / float(args.sigma_rtmlib_keypoint_px)).mean(dim=1)
                keypoint_loss = (per_joint * weights).mean()
        else:
            keypoint_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        mask_center = torch.tensor(item.mask_center_xy, dtype=torch.float32, device=device)
        projected_center = torch.median(uv_joints, dim=0).values
        center_loss = robust_l1((projected_center - mask_center) / float(args.sigma_center_px)).mean()
        mask_depth_loss = robust_l1((torch.median(joints[0, :, 2]) - float(item.mask_depth_median_m)) / float(args.sigma_mask_depth_m))
        span = hand_span_torch(joints)
        span_loss = robust_l1(torch.relu(float(args.min_span_m) - span) / float(args.sigma_span_m)) + robust_l1(torch.relu(span - float(args.max_span_m)) / float(args.sigma_span_m))
        pose_loss = robust_l1(pose_delta / float(args.sigma_pose_delta_rad)).mean()
        orient_loss = robust_l1(orient_delta / float(args.sigma_orient_delta_rad)).mean()
        trans_loss = robust_l1(trans_delta / float(args.sigma_translation_m)).mean()
        scale_loss = robust_l1(log_scale / float(args.sigma_log_scale)).mean()
        loss = (
            float(args.w_silhouette) * silhouette_loss
            + float(args.w_depth) * depth_loss
            + float(args.w_rtmlib_keypoints) * keypoint_loss
            + float(args.w_joint_prior) * joint_prior_loss
            + float(args.w_center) * center_loss
            + float(args.w_mask_depth) * mask_depth_loss
            + float(args.w_span) * span_loss
            + float(args.w_pose) * pose_loss
            + float(args.w_orient) * orient_loss
            + float(args.w_translation) * trans_loss
            + float(args.w_scale) * scale_loss
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(np.log(float(args.min_scale)), np.log(float(args.max_scale)))
            pose_delta.clamp_(-float(args.max_pose_delta_rad), float(args.max_pose_delta_rad))
            orient_delta.clamp_(-float(args.max_orient_delta_rad), float(args.max_orient_delta_rad))
            trans_delta.clamp_(-float(args.max_translation_m), float(args.max_translation_m))
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_state = {
                    "global_orient": global_orient.detach().clone(),
                    "hand_pose": hand_pose.detach().clone(),
                    "local_vertices": local_vertices.detach().clone(),
                    "local_joints": local_joints.detach().clone(),
                    "vertices": vertices.detach().clone(),
                    "joints": joints.detach().clone(),
                    "scale": torch.exp(log_scale.detach()).clone(),
                    "trans_delta": trans_delta.detach().clone(),
                    "pose_delta": pose_delta.detach().clone(),
                    "orient_delta": orient_delta.detach().clone(),
                    "loss": loss.detach().clone(),
                    "silhouette_loss": silhouette_loss.detach().clone(),
                    "depth_loss": depth_loss.detach().clone(),
                    "rtmlib_keypoint_loss": keypoint_loss.detach().clone(),
                    "joint_prior_loss": joint_prior_loss.detach().clone(),
                }
    if best_state is None:
        raise RuntimeError("optimizer produced no state")
    vertices_np = best_state["vertices"][0].cpu().numpy()
    joints_np = best_state["joints"][0].cpu().numpy()
    base_metrics = metric_rows(item.base_source_joints, item.base_source_vertices, item, args, vertex_ids_np)
    fit_metrics = metric_rows(joints_np, vertices_np, item, args, vertex_ids_np)
    fit = {
        "vertices": vertices_np,
        "joints": joints_np,
        "joints2d": project_points(joints_np, item.intrinsics.cpu().numpy()),
        "global_orient": best_state["global_orient"].cpu().numpy(),
        "hand_pose": best_state["hand_pose"].cpu().numpy(),
        "local_vertices": best_state["local_vertices"][0].cpu().numpy(),
        "local_joints": best_state["local_joints"][0].cpu().numpy(),
        "scale": float(best_state["scale"]),
        "trans_delta": best_state["trans_delta"][0].cpu().numpy(),
    }
    metrics = {
        **fit_metrics,
        "loss": best_loss,
        "base_silhouette_inside_fraction": base_metrics["silhouette_inside_fraction"],
        "base_silhouette_distance_p95_px": base_metrics["silhouette_distance_p95_px"],
        "base_mano_minus_mask_depth_median_m": base_metrics["mano_minus_mask_depth_median_m"],
        "base_sampled_vertex_minus_metric_depth_p95_abs_m": base_metrics["sampled_vertex_minus_metric_depth_p95_abs_m"],
        "scale": fit["scale"],
        "translation_delta_norm_m": float(np.linalg.norm(fit["trans_delta"])),
        "pose_delta_abs_max_rad": float(torch.max(torch.abs(best_state["pose_delta"]))),
        "orient_delta_abs_max_rad": float(torch.max(torch.abs(best_state["orient_delta"]))),
        "zero_state_vertex_error_median_m": float(np.median(vertex_error)),
        "zero_state_joint_error_median_m": float(np.median(joint_error)),
        "sampled_vertices": int(len(vertex_ids_np)),
    }
    return fit, metrics


def select_best_per_frame(fits: dict[tuple[int, int], dict], metrics: list[dict], args: argparse.Namespace) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for row in metrics:
        by_frame.setdefault(int(row["frame_idx"]), []).append(row)
    selected: dict[tuple[int, int], dict] = {}
    selected_rows = []
    for frame_idx, rows in sorted(by_frame.items()):
        ranked = sorted(
            rows,
            key=lambda r: (
                -float(r["silhouette_inside_fraction"]),
                float(r["silhouette_distance_p95_px"]),
                abs(float(r["mano_minus_mask_depth_median_m"])),
                float("inf") if r.get("rtmlib_joint_reprojection_median_px") is None else float(r["rtmlib_joint_reprojection_median_px"]),
                float(r["joint_reprojection_to_base_median_px"]),
            ),
        )
        best = ranked[0]
        key = (int(best["frame_idx"]), int(best["hand_index"]))
        selected[key] = fits[key]
        out = dict(best)
        out["selected_for_annotation_stream"] = True
        out["selection_rank_count"] = len(ranked)
        selected_rows.append(out)
    return selected, selected_rows


def apply_fits(annotations: dict, fits: dict[tuple[int, int], dict], rows: list[dict], args: argparse.Namespace) -> dict:
    out = copy.deepcopy(annotations)
    metrics_by_key = {(int(row["frame_idx"]), int(row["hand_index"])): row for row in rows}
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        new_hands = []
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, hand_i)
            fit = fits.get(key)
            if fit is None:
                continue
            row = metrics_by_key[key]
            fitted = copy.deepcopy(hand)
            fitted["vertices_camera"] = fit["local_vertices"].astype(float).tolist()
            fitted["joints3d_camera"] = fit["local_joints"].astype(float).tolist()
            fitted["vertices_source_camera_m"] = fit["vertices"].astype(float).tolist()
            fitted["joints3d_source_camera_m"] = fit["joints"].astype(float).tolist()
            fitted["joints2d"] = fit["joints2d"].astype(float).tolist()
            fitted["joints3d_world_m"] = source_to_world(fit["joints"], T).astype(float).tolist()
            fitted["vertices_world_m"] = source_to_world(fit["vertices"], T).astype(float).tolist()
            fitted["mano_params"]["global_orient"] = fit["global_orient"].reshape(1, 3, 3).astype(float).tolist()
            fitted["mano_params"]["hand_pose"] = fit["hand_pose"].reshape(15, 3, 3).astype(float).tolist()
            fitted["mano_params"]["rotation_convention"] = "wilor_rotation_matrix_pose2rot_false_with_side_x_sign"
            vertex_depth_p95 = row["sampled_vertex_minus_metric_depth_p95_abs_m"]
            interior_depth_p95 = row.get("interior_vertex_minus_metric_depth_p95_abs_m")
            if interior_depth_p95 is not None and int(row.get("interior_vertex_depth_valid_count") or 0) >= int(args.min_interior_depth_vertices):
                depth_acceptance_source = "hand_mask_interior_vertices"
                depth_acceptance_value = float(interior_depth_p95)
            else:
                depth_acceptance_source = "all_sampled_projected_vertices"
                depth_acceptance_value = None if vertex_depth_p95 is None else float(vertex_depth_p95)
            vertex_depth_ok = depth_acceptance_value is not None and float(depth_acceptance_value) <= float(args.accept_vertex_depth_p95_abs_m)
            pose_ok = float(row["pose_delta_abs_max_rad"]) <= float(args.accept_pose_delta_abs_max_rad)
            scale_ok = float(args.accept_min_scale) <= float(row["scale"]) <= float(args.accept_max_scale)
            rtmlib_required = row.get("rtmlib_joint_reprojection_median_px") is not None
            rtmlib_ok = True
            if rtmlib_required:
                rtmlib_ok = (
                    int(row.get("rtmlib_valid_keypoints") or 0) >= int(args.rtmlib_min_keypoints)
                    and float(row["rtmlib_joint_reprojection_median_px"]) <= float(args.accept_rtmlib_reprojection_median_px)
                )
            fitted["source_hand_index_before_refit"] = int(hand_i)
            fitted["measurement_available"] = bool(
                float(row["silhouette_inside_fraction"]) >= float(args.accept_inside_fraction)
                and float(row["silhouette_distance_p95_px"]) <= float(args.accept_p95_silhouette_px)
                and abs(float(row["mano_minus_mask_depth_median_m"])) <= float(args.accept_mask_depth_abs_m)
                and vertex_depth_ok
                and pose_ok
                and scale_ok
                and rtmlib_ok
                and float(row["hand_bone_m"]) >= float(args.min_span_m)
            )
            fitted["filter_status"] = "v3_mano_articulation_mask_depth_refit" if fitted["measurement_available"] else "v3_mano_articulation_mask_depth_rejected"
            fitted["world_coordinate_status"] = "v3_mano_articulation_mask_depth_source_camera_to_existing_world_camera"
            fitted["v3_mano_articulation_mask_depth_refit"] = {
                "silhouette_inside_fraction": row["silhouette_inside_fraction"],
                "silhouette_distance_median_px": row["silhouette_distance_median_px"],
                "silhouette_distance_p95_px": row["silhouette_distance_p95_px"],
                "mano_minus_mask_depth_median_m": row["mano_minus_mask_depth_median_m"],
                "sampled_vertex_minus_metric_depth_p95_abs_m": row["sampled_vertex_minus_metric_depth_p95_abs_m"],
                "sampled_vertex_depth_valid_count": row.get("sampled_vertex_depth_valid_count"),
                "interior_vertex_minus_metric_depth_p95_abs_m": row.get("interior_vertex_minus_metric_depth_p95_abs_m"),
                "interior_vertex_minus_metric_depth_median_abs_m": row.get("interior_vertex_minus_metric_depth_median_abs_m"),
                "interior_vertex_depth_valid_count": row.get("interior_vertex_depth_valid_count"),
                "interior_vertex_depth_margin_px": row.get("interior_vertex_depth_margin_px"),
                "depth_acceptance_source": depth_acceptance_source,
                "depth_acceptance_value_m": depth_acceptance_value,
                "rtmlib_joint_reprojection_median_px": row.get("rtmlib_joint_reprojection_median_px"),
                "rtmlib_joint_reprojection_p95_px": row.get("rtmlib_joint_reprojection_p95_px"),
                "rtmlib_valid_keypoints": row.get("rtmlib_valid_keypoints"),
                "hand_bone_m": row["hand_bone_m"],
                "hand_span_m": row["hand_span_m"],
                "pose_delta_abs_max_rad": row["pose_delta_abs_max_rad"],
                "translation_delta_norm_m": row["translation_delta_norm_m"],
                "scale": row["scale"],
                "acceptance": {
                    "vertex_depth_ok": bool(vertex_depth_ok),
                    "pose_ok": bool(pose_ok),
                    "scale_ok": bool(scale_ok),
                    "rtmlib_ok": bool(rtmlib_ok),
                    "measurement_available": bool(fitted["measurement_available"]),
                },
            }
            new_hands.append(fitted)
        frame["hands"] = new_hands
    return out


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return summarize(np.asarray(values, dtype=float))


def render_review(args: argparse.Namespace, annotations: dict, selected_rows: list[dict]) -> dict:
    if args.video is None or args.review_dir is None:
        return {"status": "skipped"}
    frames = frame_map(annotations)
    selected = {int(row["frame_idx"]): row for row in selected_rows}
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    args.review_dir.mkdir(parents=True, exist_ok=True)
    stills = []
    writer = None
    try:
        for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.review_stride))):
            if frame_idx not in frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_idx}")
            frame = frames[frame_idx]
            if frame.get("hands"):
                selected_hand_index = int(selected[frame_idx]["hand_index"]) if frame_idx in selected else None
                selected_hands = [
                    hand for hand in frame["hands"] if int(hand.get("source_hand_index_before_refit", -1)) == selected_hand_index
                ]
                hand = selected_hands[0] if selected_hands else frame["hands"][0]
                joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                intr = np.asarray(hand["source_intrinsics"], dtype=float)
                uv = project_points(joints, intr)
                color = (70, 235, 70) if bool(hand.get("measurement_available")) else (80, 80, 230)
                for a, b in HAND_EDGES:
                    cv2.line(image, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 3, cv2.LINE_AA)
                for point in uv:
                    cv2.circle(image, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)
            row = selected.get(frame_idx)
            label = f"frame {frame_idx}"
            if row is not None:
                label += (
                    f" inside {float(row['silhouette_inside_fraction']):.2f}"
                    f" p95 {float(row['silhouette_distance_p95_px']):.0f}px"
                    f" dz {1000.0 * float(row['mano_minus_mask_depth_median_m']):.0f}mm"
                )
            cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(image, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            if int(args.render_width) > 0 and image.shape[1] != int(args.render_width):
                h = int(round(int(args.render_width) * image.shape[0] / image.shape[1]))
                image = cv2.resize(image, (int(args.render_width), h), interpolation=cv2.INTER_AREA)
            if writer is None:
                path = args.review_dir / "mano_articulation_mask_depth_review.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.review_fps), (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open writer {path}")
            writer.write(image)
            if row is not None or frame_idx in set(int(x) for x in args.still_frames):
                still = args.review_dir / f"frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(still), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {still}")
                stills.append(str(still))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {"status": "ok", "video": str(args.review_dir / "mano_articulation_mask_depth_review.mp4"), "stills": stills}


def run(args: argparse.Namespace) -> dict:
    patch_legacy_mano_loader()
    annotations, inputs, skipped = build_inputs(args)
    mano_class = load_wilor_mano_class(args.mano_wrapper_root)
    model = mano_class(
        model_path=str(args.mano_model_root),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    )
    fits: dict[tuple[int, int], dict] = {}
    rows: list[dict] = []
    fit_errors: list[dict] = []
    for item in inputs:
        try:
            fit, metrics = fit_one(model, item, args)
            key = (int(item.frame_idx), int(item.hand_index))
            fits[key] = fit
            rows.append(metrics)
        except Exception as exc:
            fit_errors.append({"frame_idx": int(item.frame_idx), "hand_index": int(item.hand_index), "side": item.side, "reason": str(exc)})
    if not rows:
        raise RuntimeError(f"all articulation fits failed; skipped={skipped[:30]} fit_errors={fit_errors[:30]}")
    selected_fits, selected_rows = select_best_per_frame(fits, rows, args)
    output = apply_fits(annotations, fits, rows, args)
    save_json(args.output_annotations, output)
    review = render_review(args, output, selected_rows)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "refit_mano_articulation_mask_depth_v3",
        "annotations": str(args.annotations),
        "mask_track": str(args.mask_track),
        "metric_depth_npz": str(args.metric_depth_npz),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "track_id": args.track_id,
        "side": args.side,
        "fit_rows": int(len(rows)),
        "selected_rows": int(len(selected_rows)),
        "skipped_count": int(len(skipped)),
        "fit_error_count": int(len(fit_errors)),
        "summary": {
            "silhouette_inside_fraction": summarize_key(selected_rows, "silhouette_inside_fraction"),
            "silhouette_distance_p95_px": summarize_key(selected_rows, "silhouette_distance_p95_px"),
            "mano_minus_mask_depth_median_m": summarize_key(selected_rows, "mano_minus_mask_depth_median_m"),
            "sampled_vertex_minus_metric_depth_p95_abs_m": summarize_key(selected_rows, "sampled_vertex_minus_metric_depth_p95_abs_m"),
            "sampled_vertex_depth_valid_count": summarize_key(selected_rows, "sampled_vertex_depth_valid_count"),
            "interior_vertex_minus_metric_depth_p95_abs_m": summarize_key(selected_rows, "interior_vertex_minus_metric_depth_p95_abs_m"),
            "interior_vertex_minus_metric_depth_median_abs_m": summarize_key(selected_rows, "interior_vertex_minus_metric_depth_median_abs_m"),
            "interior_vertex_depth_valid_count": summarize_key(selected_rows, "interior_vertex_depth_valid_count"),
            "joint_reprojection_to_base_median_px": summarize_key(selected_rows, "joint_reprojection_to_base_median_px"),
            "rtmlib_joint_reprojection_median_px": summarize_key(selected_rows, "rtmlib_joint_reprojection_median_px"),
            "rtmlib_joint_reprojection_p95_px": summarize_key(selected_rows, "rtmlib_joint_reprojection_p95_px"),
            "rtmlib_valid_keypoints": summarize_key(selected_rows, "rtmlib_valid_keypoints"),
            "hand_bone_m": summarize_key(selected_rows, "hand_bone_m"),
            "hand_span_m": summarize_key(selected_rows, "hand_span_m"),
            "pose_delta_abs_max_rad": summarize_key(selected_rows, "pose_delta_abs_max_rad"),
            "translation_delta_norm_m": summarize_key(selected_rows, "translation_delta_norm_m"),
            "scale": summarize_key(selected_rows, "scale"),
        },
        "acceptance_thresholds": {
            "accept_inside_fraction": float(args.accept_inside_fraction),
            "accept_p95_silhouette_px": float(args.accept_p95_silhouette_px),
            "accept_mask_depth_abs_m": float(args.accept_mask_depth_abs_m),
            "accept_vertex_depth_p95_abs_m": float(args.accept_vertex_depth_p95_abs_m),
            "accept_pose_delta_abs_max_rad": float(args.accept_pose_delta_abs_max_rad),
            "accept_min_scale": float(args.accept_min_scale),
            "accept_max_scale": float(args.accept_max_scale),
            "accept_rtmlib_reprojection_median_px": float(args.accept_rtmlib_reprojection_median_px),
            "interior_depth_margin_px": float(args.interior_depth_margin_px),
            "min_interior_depth_vertices": int(args.min_interior_depth_vertices),
        },
        "rtmlib_keypoint_factor": {
            "enabled": bool(args.rtmlib_json is not None),
            "rtmlib_json": None if args.rtmlib_json is None else str(args.rtmlib_json),
            "rtmlib_prompts": None if args.rtmlib_prompts is None else str(args.rtmlib_prompts),
            "min_score": float(args.rtmlib_min_score),
            "min_keypoints": int(args.rtmlib_min_keypoints),
            "sigma_px": float(args.sigma_rtmlib_keypoint_px),
            "weight": float(args.w_rtmlib_keypoints),
        },
        "review": review,
        "rows_preview": selected_rows[:160],
        "all_rows_preview": rows[:160],
        "skipped_preview": skipped[:160],
        "fit_errors_preview": fit_errors[:160],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "all_rows_preview", "skipped_preview", "fit_errors_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--rtmlib-json", type=Path)
    parser.add_argument("--rtmlib-prompts", type=Path)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--mano-wrapper-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--mano-model-root", type=Path, default=Path("/data/dex_home/yiwen/arctic/data/body_models/mano"))
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--track-id", default="left_visible_gloved_hand")
    parser.add_argument("--side", choices=["left", "right", "any"], default="left")
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/dev/shm/ego_annotation_keyboard_hand_masks/outputs"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs/representative_keyboard/v3_keyboard_hand_sam2_visual_tracks_60_75"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-hypotheses-per-frame", type=int, default=4)
    parser.add_argument("--max-sampled-vertices", type=int, default=384)
    parser.add_argument("--max-depth-vertices", type=int, default=256)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--sigma-silhouette-px", type=float, default=8.0)
    parser.add_argument("--sigma-vertex-depth-m", type=float, default=0.050)
    parser.add_argument("--sigma-rtmlib-keypoint-px", type=float, default=18.0)
    parser.add_argument("--sigma-joint-prior-px", type=float, default=70.0)
    parser.add_argument("--sigma-center-px", type=float, default=60.0)
    parser.add_argument("--sigma-mask-depth-m", type=float, default=0.050)
    parser.add_argument("--sigma-span-m", type=float, default=0.020)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.45)
    parser.add_argument("--sigma-orient-delta-rad", type=float, default=0.35)
    parser.add_argument("--sigma-translation-m", type=float, default=0.12)
    parser.add_argument("--sigma-log-scale", type=float, default=0.20)
    parser.add_argument("--min-span-m", type=float, default=0.110)
    parser.add_argument("--max-span-m", type=float, default=0.230)
    parser.add_argument("--min-scale", type=float, default=0.80)
    parser.add_argument("--max-scale", type=float, default=1.20)
    parser.add_argument("--max-pose-delta-rad", type=float, default=1.10)
    parser.add_argument("--max-orient-delta-rad", type=float, default=0.80)
    parser.add_argument("--max-translation-m", type=float, default=0.45)
    parser.add_argument("--invalid-depth-penalty", type=float, default=8.0)
    parser.add_argument("--max-silhouette-distance-px", type=float, default=160.0)
    parser.add_argument("--max-zero-state-vertex-error-m", type=float, default=0.030)
    parser.add_argument("--max-zero-state-joint-error-m", type=float, default=0.030)
    parser.add_argument("--interior-depth-margin-px", type=float, default=20.0)
    parser.add_argument("--min-interior-depth-vertices", type=int, default=120)
    parser.add_argument("--rtmlib-min-score", type=float, default=0.30)
    parser.add_argument("--rtmlib-min-keypoints", type=int, default=12)
    parser.add_argument("--invalid-keypoint-penalty", type=float, default=8.0)
    parser.add_argument("--w-silhouette", type=float, default=1.2)
    parser.add_argument("--w-depth", type=float, default=0.7)
    parser.add_argument("--w-rtmlib-keypoints", type=float, default=0.0)
    parser.add_argument("--w-joint-prior", type=float, default=0.10)
    parser.add_argument("--w-center", type=float, default=0.4)
    parser.add_argument("--w-mask-depth", type=float, default=0.7)
    parser.add_argument("--w-span", type=float, default=1.0)
    parser.add_argument("--w-pose", type=float, default=0.16)
    parser.add_argument("--w-orient", type=float, default=0.18)
    parser.add_argument("--w-translation", type=float, default=0.20)
    parser.add_argument("--w-scale", type=float, default=0.25)
    parser.add_argument("--accept-inside-fraction", type=float, default=0.55)
    parser.add_argument("--accept-p95-silhouette-px", type=float, default=50.0)
    parser.add_argument("--accept-mask-depth-abs-m", type=float, default=0.035)
    parser.add_argument("--accept-vertex-depth-p95-abs-m", type=float, default=0.060)
    parser.add_argument("--accept-pose-delta-abs-max-rad", type=float, default=0.75)
    parser.add_argument("--accept-min-scale", type=float, default=0.88)
    parser.add_argument("--accept-max-scale", type=float, default=1.12)
    parser.add_argument("--accept-rtmlib-reprojection-median-px", type=float, default=35.0)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--iters", type=int, default=180)
    parser.add_argument("--review-stride", type=int, default=1)
    parser.add_argument("--review-fps", type=float, default=8.0)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[60, 67, 75])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
