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
import trimesh

from diagnose_contact_kinematics_v3 import selected_vertex_ids
from diagnose_metric_depth_alignment_v3 import depth_frame
from diagnose_volume_sdf_contact_v3 import crop_mesh_around_points, voxel_sdf
from fit_mano_to_hand_mask_depth_v3 import distance_map
from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask
from optimize_hand_translation_contact_v3 import source_to_world
from refit_mano_articulation_mask_depth_v3 import source_size_for_frame
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
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def frame_map(annotations: dict) -> dict[int, dict]:
    out = {}
    for frame in annotations.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate frame_idx {frame_idx}")
        out[frame_idx] = frame
    if not out:
        raise RuntimeError("annotations contain no frames")
    return out


def hand_vertex_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    raise RuntimeError("V8 graph requires full vertices_source_camera_m")


def as_matrix_stack(raw: object, shape: tuple[int, int, int], key: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape != shape:
        raise RuntimeError(f"{key} has shape {arr.shape}, expected {shape}")
    return arr


def as_vector(raw: object, length: int, key: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape != (length,):
        raise RuntimeError(f"{key} has shape {arr.shape}, expected {(length,)}")
    return arr


def object_mask_for_frame(frame: dict, source_size: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    obj = frame.get("object", {})
    mask_path_raw = obj.get("mask_path")
    if not isinstance(mask_path_raw, str) or not mask_path_raw:
        raise RuntimeError("frame object lacks mask_path")
    mask_size = np.asarray(obj.get("mask_image_size", []), dtype=int)
    if mask_size.shape != (2,) or np.any(mask_size <= 0):
        raise RuntimeError("frame object lacks valid mask_image_size")
    mask_path = localize_path(mask_path_raw, args.remote_output_root, args.local_output_root)
    mask = resize_bool_mask(mask_path, (int(mask_size[0]), int(mask_size[1])))
    if mask.shape[1] != int(source_size[0]) or mask.shape[0] != int(source_size[1]):
        mask = cv2.resize(mask.astype(np.uint8), (int(source_size[0]), int(source_size[1])), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def load_hand_mask_evidence(path: Path | None, args: argparse.Namespace) -> dict[tuple[int, str, str], dict]:
    if path is None:
        return {}
    evidence = load_json(path)
    rows: dict[tuple[int, str, str], dict] = {}
    for frame in evidence.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        for hand in frame.get("hands", []):
            key = (frame_idx, str(hand.get("side", "")), str(hand.get("track_id", "")))
            if key in rows:
                raise RuntimeError(f"duplicate hand mask evidence for {key}")
            rows[key] = hand
    return rows


def hand_mask_for_observation(hand_mask_rows: dict[tuple[int, str, str], dict], frame_idx: int, side: str, track_id: str, source_size: tuple[int, int], args: argparse.Namespace) -> np.ndarray | None:
    row = hand_mask_rows.get((int(frame_idx), str(side), str(track_id)))
    if row is None:
        return None
    mask_path_raw = row.get("mask_path")
    if not isinstance(mask_path_raw, str) or not mask_path_raw:
        raise RuntimeError(f"hand mask evidence for {(frame_idx, side, track_id)} lacks mask_path")
    mask_path = localize_path(mask_path_raw, args.hand_mask_remote_root, args.hand_mask_local_root)
    mask = resize_bool_mask(mask_path, (int(args.hand_mask_width), int(args.hand_mask_height)))
    if mask.shape[1] != int(source_size[0]) or mask.shape[0] != int(source_size[1]):
        mask = cv2.resize(mask.astype(np.uint8), (int(source_size[0]), int(source_size[1])), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def supported_contact_patch(
    patch_ids: np.ndarray,
    source_vertices: np.ndarray,
    intrinsics: np.ndarray,
    hand_mask: np.ndarray | None,
    rtmlib: dict | None,
    source_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    uv = project_torch(torch.tensor(source_vertices[patch_ids], dtype=torch.float32), torch.tensor(intrinsics, dtype=torch.float32)).numpy()
    xy = np.rint(uv).astype(int)
    in_image = (
        np.isfinite(uv).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < int(source_size[0]))
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < int(source_size[1]))
    )
    support: dict = {
        "patch_vertices": int(len(patch_ids)),
        "in_image_fraction": float(np.mean(in_image)) if len(in_image) else 0.0,
    }
    checks = []
    if hand_mask is not None and len(patch_ids):
        dist = distance_map(hand_mask)
        mask_d = np.full(len(patch_ids), np.inf, dtype=np.float32)
        mask_d[in_image] = dist[xy[in_image, 1], xy[in_image, 0]]
        support["hand_mask_distance_median_px"] = float(np.median(mask_d[np.isfinite(mask_d)])) if np.any(np.isfinite(mask_d)) else None
        support["hand_mask_inside_fraction"] = float(np.mean(mask_d <= float(args.contact_patch_hand_mask_max_px)))
        checks.append(support["hand_mask_inside_fraction"] >= float(args.contact_patch_min_hand_mask_inside_fraction))
    if rtmlib is not None and len(patch_ids):
        keypoints = np.asarray(rtmlib["keypoints"], dtype=np.float32)
        scores = np.asarray(rtmlib["scores"], dtype=np.float32)
        valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores) & (scores >= float(args.rtmlib_min_score))
        if np.any(valid):
            nearest = np.min(np.linalg.norm(uv[:, None, :] - keypoints[valid][None, :, :], axis=2), axis=1)
            support["nearest_rtmlib_distance_median_px"] = float(np.median(nearest))
            checks.append(support["nearest_rtmlib_distance_median_px"] <= float(args.contact_patch_max_nearest_rtmlib_px))
    support["supported"] = bool(checks and all(checks))
    return bool(support["supported"]), support


def torch_sample(image: torch.Tensor, xy: torch.Tensor, invalid_value: float) -> torch.Tensor:
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


def torch_sample_sdf(points: torch.Tensor, sdf: torch.Tensor, transform: torch.Tensor, invalid_value: float) -> torch.Tensor:
    pitch = transform[0, 0]
    origin = transform[:3, 3]
    coords = (points - origin[None, :]) / pitch
    base = torch.floor(coords).long()
    frac = coords - base.float()
    shape = torch.tensor(sdf.shape, dtype=torch.long, device=points.device)
    valid = (
        torch.isfinite(coords).all(dim=1)
        & (base[:, 0] >= 0)
        & (base[:, 1] >= 0)
        & (base[:, 2] >= 0)
        & (base[:, 0] + 1 < shape[0])
        & (base[:, 1] + 1 < shape[1])
        & (base[:, 2] + 1 < shape[2])
    )
    values = torch.full((points.shape[0],), float(invalid_value), dtype=torch.float32, device=points.device)
    if bool(valid.any()):
        b = base[valid]
        f = frac[valid]
        x0, y0, z0 = b[:, 0], b[:, 1], b[:, 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = f[:, 0], f[:, 1], f[:, 2]
        c000 = sdf[x0, y0, z0]
        c100 = sdf[x1, y0, z0]
        c010 = sdf[x0, y1, z0]
        c110 = sdf[x1, y1, z0]
        c001 = sdf[x0, y0, z1]
        c101 = sdf[x1, y0, z1]
        c011 = sdf[x0, y1, z1]
        c111 = sdf[x1, y1, z1]
        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd
        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd
        values[valid] = c0 * (1.0 - zd) + c1 * zd
    return values


def summarize(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def contact_rows_by_frame(path: Path, frame_start: int, frame_end: int) -> dict[tuple[int, int], dict]:
    report = load_json(path)
    rows = {}
    for row in report.get("rows_detail", []):
        frame_idx = int(row["frame_idx"])
        if frame_idx < int(frame_start) or frame_idx > int(frame_end):
            continue
        if bool(row.get("reliable_for_contact", False)) or bool(row.get("geometry_backed_temporal_contact", False)):
            rows[(frame_idx, int(row["hand_idx"]))] = row
    return rows


def load_rtmlib_targets(path: Path | None, prompts_path: Path | None, frame_start: int, frame_end: int, args: argparse.Namespace) -> dict[int, dict]:
    if path is None:
        return {}
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("RTMLib JSON must contain a frames list")
    selected_by_frame: dict[int, int] = {}
    if prompts_path is not None:
        prompts = load_json(prompts_path)
        diagnostics = prompts.get("diagnostics")
        if not isinstance(diagnostics, list):
            raise RuntimeError("RTMLib prompts JSON must contain diagnostics")
        selected_by_frame = {
            int(row["frame_idx"]): int(row["selected_rtmlib_hand_idx"])
            for row in diagnostics
            if int(frame_start) <= int(row["frame_idx"]) <= int(frame_end)
        }
    targets: dict[int, dict] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < int(frame_start) or frame_idx > int(frame_end):
            continue
        hands = frame.get("hands")
        if not isinstance(hands, list):
            raise RuntimeError(f"RTMLib frame {frame_idx} hands field must be a list")
        if frame_idx in selected_by_frame:
            selected = [hand for hand in hands if int(hand.get("hand_idx", -1)) == int(selected_by_frame[frame_idx])]
        elif len(hands) == 1:
            selected = [hands[0]]
        else:
            raise RuntimeError(f"RTMLib frame {frame_idx} has {len(hands)} hands and no prompt-selected hand")
        if len(selected) != 1:
            raise RuntimeError(f"RTMLib frame {frame_idx} selected-hand match count is {len(selected)}")
        hand = selected[0]
        keypoints = np.asarray(hand.get("keypoints", []), dtype=np.float32)
        scores = np.asarray(hand.get("scores", []), dtype=np.float32)
        if keypoints.shape != (21, 2) or scores.shape != (21,):
            raise RuntimeError(f"RTMLib frame {frame_idx} selected hand has invalid keypoints or scores")
        valid = np.isfinite(keypoints).all(axis=1) & np.isfinite(scores) & (scores >= float(args.rtmlib_min_score))
        if int(np.count_nonzero(valid)) < int(args.rtmlib_min_keypoints):
            raise RuntimeError(
                f"RTMLib frame {frame_idx} selected hand has {int(np.count_nonzero(valid))} valid keypoints; required {int(args.rtmlib_min_keypoints)}"
            )
        targets[frame_idx] = {
            "keypoints": keypoints,
            "scores": scores,
            "hand_idx": int(hand.get("hand_idx", 0)),
            "mean_score": float(hand.get("mean_score", np.mean(scores[valid]))),
        }
    return targets


@dataclass(frozen=True)
class V8Obs:
    frame_idx: int
    hand_idx: int
    side: str
    track_id: str
    frame_order: int
    intrinsics: torch.Tensor
    source_size: tuple[int, int]
    hand_mask_distance: torch.Tensor | None
    metric_depth: torch.Tensor
    depth_valid: torch.Tensor
    base_global_orient: torch.Tensor
    base_hand_pose: torch.Tensor
    betas: torch.Tensor
    base_cam_t: torch.Tensor
    base_local_vertices: np.ndarray
    base_local_joints: np.ndarray
    base_source_vertices: np.ndarray
    base_source_joints: np.ndarray
    base_joints2d: np.ndarray
    base_zero_local_rotation: torch.Tensor
    base_zero_local_translation: torch.Tensor
    base_zero_local_scale: torch.Tensor
    sign: int
    rtmlib_joints2d: torch.Tensor | None
    rtmlib_scores: torch.Tensor | None
    hand_mask: np.ndarray | None
    contact_patch_ids: torch.Tensor
    contact_seed: float
    has_contact_evidence: bool
    contact_lookup_key: tuple[int, int]
    contact_row_track_id: str | None
    contact_support: dict
    sdf: torch.Tensor
    sdf_transform: torch.Tensor
    T_world_camera: np.ndarray


def build_local_sdf(
    mesh_world: np.ndarray,
    faces: np.ndarray,
    T_world_camera: np.ndarray,
    cover_points_camera: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_camera = camera_points(mesh_world, T_world_camera).astype(np.float32)
    faces_i = np.asarray(faces, dtype=np.int32)
    mesh = crop_mesh_around_points(
        mesh_camera,
        faces_i,
        cover_points_camera.astype(np.float32),
        float(args.local_sdf_crop_margin_m),
        int(args.local_sdf_min_faces),
    )
    sdf, transform, _ = voxel_sdf(mesh, float(args.sdf_pitch_m), int(args.sdf_pad_voxels), cover_points_camera)
    return sdf, transform


def build_observations(model, annotations: dict, args: argparse.Namespace) -> tuple[list[V8Obs], list[dict]]:
    frames = frame_map(annotations)
    meshes = load_mesh_archive(args.object_mesh_npz)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_indices)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    contact_rows = contact_rows_by_frame(args.contact_report, int(args.frame_start), int(args.frame_end))
    hand_mask_rows = load_hand_mask_evidence(args.hand_mask_evidence_json, args)
    rtmlib_targets = load_rtmlib_targets(args.rtmlib_json, args.rtmlib_prompts, int(args.frame_start), int(args.frame_end), args)
    observations: list[V8Obs] = []
    skipped: list[dict] = []
    device = torch.device(args.device)
    for order, frame_idx in enumerate(range(int(args.frame_start), int(args.frame_end) + 1)):
        if frame_idx not in frames:
            skipped.append({"frame_idx": frame_idx, "reason": "frame_missing_from_annotations"})
            continue
        if frame_idx not in meshes:
            skipped.append({"frame_idx": frame_idx, "reason": "frame_missing_from_object_mesh"})
            continue
        frame = frames[frame_idx]
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = source_size_for_frame(frame, args)
            object_mask = object_mask_for_frame(frame, source_size, args)
            depth_valid = np.isfinite(depth) & (depth > float(args.min_depth_m))
            mesh_vertices, mesh_faces = meshes[frame_idx]
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_idx, hand in enumerate(frame.get("hands", [])):
            try:
                if args.side is not None and str(hand.get("side")) != str(args.side):
                    raise RuntimeError("side_filtered")
                if args.track_id is not None and str(hand.get("track_id")) != str(args.track_id):
                    raise RuntimeError("track_filtered")
                if not bool(hand.get("measurement_available", False)):
                    raise RuntimeError("hand_measurement_unavailable")
                mano = hand.get("mano_params")
                if not isinstance(mano, dict):
                    raise RuntimeError("hand lacks mano_params")
                base_global_orient = as_matrix_stack(mano.get("global_orient"), (1, 3, 3), "global_orient")
                base_hand_pose = as_matrix_stack(mano.get("hand_pose"), (15, 3, 3), "hand_pose").reshape(1, 15, 3, 3)
                betas = as_vector(mano.get("betas"), 10, "betas").reshape(1, 10)
                local_vertices = np.asarray(hand["vertices_camera"], dtype=np.float32)
                local_joints = np.asarray(hand["joints3d_camera"], dtype=np.float32)
                source_vertices = np.asarray(hand[hand_vertex_key(hand)], dtype=np.float32)
                source_joints = np.asarray(hand["joints3d_source_camera_m"], dtype=np.float32)
                base_cam_t = np.asarray(hand["cam_t"], dtype=np.float32).reshape(1, 3)
                joints2d = np.asarray(hand["joints2d_raw"], dtype=np.float32)
                intr = np.asarray(hand["source_intrinsics"], dtype=np.float32)
                if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid local vertices")
                if source_vertices.shape != local_vertices.shape or local_joints.shape != (21, 3) or source_joints.shape != (21, 3):
                    raise RuntimeError("invalid source/local geometry")
                if joints2d.shape != (21, 2) or intr.shape != (4,):
                    raise RuntimeError("invalid 2D or intrinsics")
                side = str(hand.get("side", "unknown"))
                track_id = str(hand.get("track_id", ""))
                hand_mask = hand_mask_for_observation(hand_mask_rows, frame_idx, side, track_id, source_size, args)
                hand_mask_distance = None if hand_mask is None else distance_map(hand_mask)
                rtmlib = rtmlib_targets.get(frame_idx)
                sign = side_sign(str(hand.get("side", "right")))
                with torch.no_grad():
                    base_out = model(
                        global_orient=torch.tensor(base_global_orient[None], dtype=torch.float32, device=device),
                        hand_pose=torch.tensor(base_hand_pose, dtype=torch.float32, device=device),
                        betas=torch.tensor(betas, dtype=torch.float32, device=device),
                        return_verts=True,
                        pose2rot=False,
                    )
                    canonical_vertices = apply_side_sign(base_out.vertices, sign)[0].detach().cpu().numpy()
                    canonical_joints = apply_side_sign(base_out.joints, sign)[0].detach().cpu().numpy()
                local_scale, local_rotation, local_translation, vertex_error = similarity_from_to(canonical_vertices, local_vertices)
                aligned_joints = local_scale * (canonical_joints @ local_rotation.T) + local_translation[None, :]
                joint_error = np.linalg.norm(aligned_joints - local_joints, axis=1)
                if float(np.median(vertex_error)) > float(args.max_zero_state_vertex_error_m):
                    raise RuntimeError(f"zero-state vertex mismatch {float(np.median(vertex_error)):.6f}m")
                if float(np.median(joint_error)) > float(args.max_zero_state_joint_error_m):
                    raise RuntimeError(f"zero-state joint mismatch {float(np.median(joint_error)):.6f}m")
                row = contact_rows.get((frame_idx, hand_idx))
                if row is not None:
                    patch_ids = selected_vertex_ids(row).astype(np.int64)
                    has_contact, contact_support = supported_contact_patch(patch_ids, source_vertices, intr, hand_mask, rtmlib, source_size, args)
                    contact_seed = 0.90 if has_contact else 0.05
                    contact_row_track_id = str(row.get("track_id", ""))
                else:
                    uv = project_torch(torch.tensor(source_vertices, dtype=torch.float32), torch.tensor(intr, dtype=torch.float32)).numpy()
                    dist = cv2.distanceTransform(object_mask.astype(np.uint8), cv2.DIST_L2, 3)
                    xy = np.rint(uv).astype(int)
                    valid = (
                        np.isfinite(uv).all(axis=1)
                        & (xy[:, 0] >= 0)
                        & (xy[:, 0] < source_size[0])
                        & (xy[:, 1] >= 0)
                        & (xy[:, 1] < source_size[1])
                    )
                    near = np.flatnonzero(valid & (dist[xy[:, 1], xy[:, 0]] <= float(args.contact_distance_px)))
                    if len(near) > int(args.max_contact_vertices):
                        near = near[np.linspace(0, len(near) - 1, int(args.max_contact_vertices), dtype=int)]
                    patch_ids = near.astype(np.int64)
                    contact_seed = 0.05
                    has_contact = False
                    contact_row_track_id = None
                    contact_support = {"supported": False, "source": "near_object_fallback", "patch_vertices": int(len(patch_ids))}
                if row is None and len(patch_ids) < int(args.min_patch_vertices):
                    patch_ids = np.linspace(0, len(source_vertices) - 1, int(args.min_patch_vertices), dtype=np.int64)
                    contact_seed = 0.01
                    has_contact = False
                    contact_row_track_id = None
                    contact_support = {"supported": False, "source": "uniform_fallback", "patch_vertices": int(len(patch_ids))}
                cover = source_vertices[patch_ids]
                sdf, sdf_transform = build_local_sdf(mesh_vertices, mesh_faces, T_world_camera, cover, args)
                if rtmlib is None:
                    rtmlib_joints = None
                    rtmlib_scores = None
                else:
                    rtmlib_joints = torch.tensor(rtmlib["keypoints"], dtype=torch.float32, device=device)
                    rtmlib_scores = torch.tensor(rtmlib["scores"], dtype=torch.float32, device=device)
                observations.append(
                    V8Obs(
                        frame_idx=frame_idx,
                        hand_idx=hand_idx,
                        side=side,
                        track_id=track_id,
                        frame_order=order,
                        intrinsics=torch.tensor(intr, dtype=torch.float32, device=device),
                        source_size=(int(source_size[0]), int(source_size[1])),
                        hand_mask_distance=None if hand_mask_distance is None else torch.tensor(hand_mask_distance, dtype=torch.float32, device=device),
                        metric_depth=torch.tensor(depth, dtype=torch.float32, device=device),
                        depth_valid=torch.tensor(depth_valid, dtype=torch.bool, device=device),
                        base_global_orient=torch.tensor(base_global_orient[None], dtype=torch.float32, device=device),
                        base_hand_pose=torch.tensor(base_hand_pose, dtype=torch.float32, device=device),
                        betas=torch.tensor(betas, dtype=torch.float32, device=device),
                        base_cam_t=torch.tensor(base_cam_t, dtype=torch.float32, device=device),
                        base_local_vertices=local_vertices,
                        base_local_joints=local_joints,
                        base_source_vertices=source_vertices,
                        base_source_joints=source_joints,
                        base_joints2d=joints2d,
                        base_zero_local_rotation=torch.tensor(local_rotation, dtype=torch.float32, device=device),
                        base_zero_local_translation=torch.tensor(local_translation, dtype=torch.float32, device=device).reshape(1, 1, 3),
                        base_zero_local_scale=torch.tensor(float(local_scale), dtype=torch.float32, device=device),
                        sign=int(sign),
                        rtmlib_joints2d=rtmlib_joints,
                        rtmlib_scores=rtmlib_scores,
                        hand_mask=hand_mask,
                        contact_patch_ids=torch.tensor(patch_ids, dtype=torch.long, device=device),
                        contact_seed=float(contact_seed),
                        has_contact_evidence=bool(has_contact),
                        contact_lookup_key=(int(frame_idx), int(hand_idx)),
                        contact_row_track_id=contact_row_track_id,
                        contact_support=contact_support,
                        sdf=torch.tensor(sdf, dtype=torch.float32, device=device),
                        sdf_transform=torch.tensor(sdf_transform, dtype=torch.float32, device=device),
                        T_world_camera=T_world_camera,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_idx, "side": hand.get("side"), "reason": str(exc)})
    if not observations:
        raise RuntimeError(f"no V8 observations built; skipped={skipped[:20]}")
    return observations, skipped


def contact_prior_logit(seed: float) -> float:
    p = float(np.clip(seed, 1e-4, 1.0 - 1e-4))
    return float(np.log(p / (1.0 - p)))


def sample_vertex_ids(count: int, max_count: int) -> np.ndarray:
    if count <= int(max_count):
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, int(max_count), dtype=np.int64)


def solve(model, observations: list[V8Obs], args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], list[dict]]:
    device = torch.device(args.device)
    n = len(observations)
    pose_delta = torch.zeros((n, 1, 15, 3), dtype=torch.float32, device=device, requires_grad=True)
    orient_delta = torch.zeros((n, 1, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    trans_delta = torch.zeros((n, 1, 3), dtype=torch.float32, device=device, requires_grad=True)
    log_scale = torch.zeros((n, 1, 1, 1), dtype=torch.float32, device=device, requires_grad=True)
    contact_logit = torch.tensor([contact_prior_logit(obs.contact_seed) for obs in observations], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([pose_delta, orient_delta, trans_delta, log_scale, contact_logit], lr=float(args.lr))
    vertex_ids = [torch.tensor(sample_vertex_ids(len(obs.base_source_vertices), int(args.max_sampled_vertices)), dtype=torch.long, device=device) for obs in observations]
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    loss_rows: list[dict] = []
    model = model.to(device)
    for step in range(int(args.iters)):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        row = {"step": int(step)}
        solved = forward_state(model, observations, pose_delta, orient_delta, trans_delta, log_scale, args)
        for i, obs in enumerate(observations):
            vertices = solved["vertices"][i]
            joints = solved["joints"][i]
            uv_joints = project_torch(joints, obs.intrinsics)
            base_uv = torch.tensor(obs.base_joints2d, dtype=torch.float32, device=device)
            keypoint_loss = robust_l1((uv_joints - base_uv) / float(args.sigma_keypoint_px)).mean()
            losses.append(float(args.w_keypoint) * keypoint_loss)
            if obs.rtmlib_joints2d is not None and obs.rtmlib_scores is not None:
                valid = torch.isfinite(obs.rtmlib_joints2d).all(dim=1) & torch.isfinite(obs.rtmlib_scores) & (obs.rtmlib_scores >= float(args.rtmlib_min_score))
                if int(valid.sum().detach().cpu()) >= int(args.rtmlib_min_keypoints):
                    weights = obs.rtmlib_scores[valid].clamp_min(0.0)
                    weights = weights / weights.mean().clamp_min(1e-6)
                    per_joint = robust_l1((uv_joints[valid] - obs.rtmlib_joints2d[valid]) / float(args.sigma_rtmlib_keypoint_px)).mean(dim=1)
                    losses.append(float(args.w_rtmlib_keypoint) * (per_joint * weights).mean())
            ids = vertex_ids[i]
            uv_vertices = project_torch(vertices[ids], obs.intrinsics)
            if obs.hand_mask_distance is not None:
                silhouette = torch_sample(obs.hand_mask_distance, uv_vertices, float(args.max_silhouette_distance_px)).clamp_max(float(args.max_silhouette_distance_px))
                losses.append(float(args.w_silhouette) * robust_l1(silhouette / float(args.sigma_silhouette_px)).mean())
            uv_depth = uv_vertices.clone()
            scale_x = float(obs.metric_depth.shape[1]) / float(obs.source_size[0])
            scale_y = float(obs.metric_depth.shape[0]) / float(obs.source_size[1])
            uv_depth[:, 0] *= scale_x
            uv_depth[:, 1] *= scale_y
            target_depth = torch_sample(obs.metric_depth, uv_depth, 0.0)
            target_ok = torch_sample(obs.depth_valid.float(), uv_depth, 0.0) > 0.5
            if bool(target_ok.any()):
                losses.append(float(args.w_depth) * robust_l1((vertices[ids][target_ok, 2] - target_depth[target_ok]) / float(args.sigma_depth_m)).mean())
            span = hand_span_torch(joints.reshape(1, 21, 3))
            losses.append(float(args.w_span) * robust_l1(torch.relu(float(args.min_span_m) - span) / float(args.sigma_span_m)).mean())
            losses.append(float(args.w_span) * robust_l1(torch.relu(span - float(args.max_span_m)) / float(args.sigma_span_m)).mean())
            patch = vertices[obs.contact_patch_ids]
            sdf = torch_sample_sdf(patch, obs.sdf, obs.sdf_transform, float(args.invalid_sdf_m))
            finite = torch.isfinite(sdf) & (torch.abs(sdf) < float(args.invalid_sdf_m) * 0.5)
            if bool(finite.any()):
                contact_prob = torch.sigmoid(contact_logit[i])
                contact_term = torch.sqrt(contact_prob.clamp_min(1e-5)) * torch.abs(sdf[finite]) / float(args.sigma_contact_sdf_m)
                penetration_term = torch.sqrt((1.0 - contact_prob).clamp_min(1e-5)) * torch.relu(-sdf[finite]) / float(args.sigma_penetration_m)
                losses.append(float(args.w_contact) * robust_l1(contact_term).mean())
                if float(args.w_contact_tail) > 0.0:
                    contact_tail = torch.max(contact_term)
                    losses.append(float(args.w_contact_tail) * robust_l1(contact_tail))
                losses.append(float(args.w_penetration) * robust_l1(penetration_term).mean())
            prior = torch.tensor(contact_prior_logit(obs.contact_seed), dtype=torch.float32, device=device)
            losses.append(float(args.w_contact_prior) * robust_l1((contact_logit[i] - prior) / float(args.sigma_contact_logit)).mean())
            losses.append(float(args.w_pose_prior) * robust_l1(pose_delta[i] / float(args.sigma_pose_delta_rad)).mean())
            losses.append(float(args.w_orient_prior) * robust_l1(orient_delta[i] / float(args.sigma_orient_delta_rad)).mean())
            losses.append(float(args.w_translation_prior) * robust_l1(trans_delta[i] / float(args.sigma_translation_m)).mean())
            losses.append(float(args.w_scale_prior) * robust_l1(log_scale[i] / float(args.sigma_log_scale)).mean())
        by_track: dict[tuple[str, str], list[int]] = {}
        for i, obs in enumerate(observations):
            by_track.setdefault((obs.side, obs.track_id), []).append(i)
        for indices in by_track.values():
            indices.sort(key=lambda i: observations[i].frame_idx)
            for a, b in zip(indices[:-1], indices[1:]):
                losses.append(float(args.w_temporal_translation) * robust_l1((trans_delta[b] - trans_delta[a]) / float(args.sigma_temporal_translation_m)).mean())
                losses.append(float(args.w_temporal_pose) * robust_l1((pose_delta[b] - pose_delta[a]) / float(args.sigma_temporal_pose_rad)).mean())
                losses.append(float(args.w_temporal_contact) * robust_l1((contact_logit[b] - contact_logit[a]) / float(args.sigma_temporal_contact_logit)).mean())
        loss = torch.stack([x.reshape(()) for x in losses]).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pose_delta.clamp_(-float(args.max_pose_delta_rad), float(args.max_pose_delta_rad))
            orient_delta.clamp_(-float(args.max_orient_delta_rad), float(args.max_orient_delta_rad))
            trans_delta.clamp_(-float(args.max_translation_m), float(args.max_translation_m))
            log_scale.clamp_(np.log(float(args.min_scale)), np.log(float(args.max_scale)))
            contact_logit.clamp_(-float(args.max_abs_contact_logit), float(args.max_abs_contact_logit))
            if float(loss.detach().cpu()) < best_loss:
                best_loss = float(loss.detach().cpu())
                best_state = {
                    "pose_delta": pose_delta.detach().clone(),
                    "orient_delta": orient_delta.detach().clone(),
                    "trans_delta": trans_delta.detach().clone(),
                    "log_scale": log_scale.detach().clone(),
                    "contact_logit": contact_logit.detach().clone(),
                    "loss": loss.detach().clone(),
                }
        if step == 0 or step == int(args.iters) - 1 or (int(args.log_every) > 0 and step % int(args.log_every) == 0):
            row["loss"] = float(loss.detach().cpu())
            loss_rows.append(row)
    if best_state is None:
        raise RuntimeError("V8 optimizer produced no state")
    return best_state, loss_rows


def forward_state(
    model,
    observations: list[V8Obs],
    pose_delta: torch.Tensor,
    orient_delta: torch.Tensor,
    trans_delta: torch.Tensor,
    log_scale: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, list[torch.Tensor]]:
    vertices_out: list[torch.Tensor] = []
    joints_out: list[torch.Tensor] = []
    local_vertices_out: list[torch.Tensor] = []
    local_joints_out: list[torch.Tensor] = []
    global_orient_out: list[torch.Tensor] = []
    hand_pose_out: list[torch.Tensor] = []
    for i, obs in enumerate(observations):
        global_orient = rotvec_to_matrix(orient_delta[i]) @ obs.base_global_orient
        hand_pose = rotvec_to_matrix(pose_delta[i]) @ obs.base_hand_pose
        model_out = model(global_orient=global_orient, hand_pose=hand_pose, betas=obs.betas, return_verts=True, pose2rot=False)
        canonical_vertices = apply_side_sign(model_out.vertices, obs.sign)
        canonical_joints = apply_side_sign(model_out.joints, obs.sign)
        local_vertices = obs.base_zero_local_scale * torch.matmul(canonical_vertices, obs.base_zero_local_rotation.T) + obs.base_zero_local_translation
        local_joints = obs.base_zero_local_scale * torch.matmul(canonical_joints, obs.base_zero_local_rotation.T) + obs.base_zero_local_translation
        scale = torch.exp(log_scale[i])
        delta = trans_delta[i].reshape(1, 1, 3)
        vertices = scale * local_vertices + obs.base_cam_t[:, None, :] + delta
        joints = scale * local_joints + obs.base_cam_t[:, None, :] + delta
        vertices_out.append(vertices[0])
        joints_out.append(joints[0])
        local_vertices_out.append(local_vertices[0])
        local_joints_out.append(local_joints[0])
        global_orient_out.append(global_orient)
        hand_pose_out.append(hand_pose)
    return {
        "vertices": vertices_out,
        "joints": joints_out,
        "local_vertices": local_vertices_out,
        "local_joints": local_joints_out,
        "global_orient": global_orient_out,
        "hand_pose": hand_pose_out,
    }


def metrics_for_state(model, observations: list[V8Obs], state: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[list[dict], dict[str, list[np.ndarray]]]:
    solved = forward_state(model, observations, state["pose_delta"], state["orient_delta"], state["trans_delta"], state["log_scale"], args)
    rows = []
    arrays: dict[str, list[np.ndarray]] = {"vertices": [], "joints": [], "local_vertices": [], "local_joints": [], "global_orient": [], "hand_pose": []}
    contact_prob = torch.sigmoid(state["contact_logit"]).detach().cpu().numpy()
    for i, obs in enumerate(observations):
        vertices = solved["vertices"][i].detach().cpu().numpy()
        joints = solved["joints"][i].detach().cpu().numpy()
        uv = project_torch(solved["joints"][i], obs.intrinsics).detach().cpu().numpy()
        reproj = np.linalg.norm(uv - obs.base_joints2d, axis=1)
        if obs.rtmlib_joints2d is None or obs.rtmlib_scores is None:
            rtmlib_reproj = np.asarray([], dtype=np.float32)
        else:
            rtmlib_xy = obs.rtmlib_joints2d.detach().cpu().numpy()
            rtmlib_scores = obs.rtmlib_scores.detach().cpu().numpy()
            valid_rtmlib = np.isfinite(rtmlib_xy).all(axis=1) & np.isfinite(rtmlib_scores) & (rtmlib_scores >= float(args.rtmlib_min_score))
            rtmlib_reproj = np.linalg.norm(uv[valid_rtmlib] - rtmlib_xy[valid_rtmlib], axis=1)
        uv_vertices = project_torch(solved["vertices"][i], obs.intrinsics).detach()
        if obs.hand_mask_distance is None:
            hand_silhouette = np.asarray([], dtype=np.float32)
        else:
            hand_silhouette = torch_sample(obs.hand_mask_distance, uv_vertices, float(args.max_silhouette_distance_px)).clamp_max(float(args.max_silhouette_distance_px)).detach().cpu().numpy()
        patch = solved["vertices"][i][obs.contact_patch_ids]
        sdf = torch_sample_sdf(patch, obs.sdf, obs.sdf_transform, float(args.invalid_sdf_m)).detach().cpu().numpy()
        sdf = sdf[np.isfinite(sdf) & (np.abs(sdf) < float(args.invalid_sdf_m) * 0.5)]
        rows.append(
            {
                "frame_idx": int(obs.frame_idx),
                "hand_idx": int(obs.hand_idx),
                "side": obs.side,
                "track_id": obs.track_id,
                "has_contact_evidence": bool(obs.has_contact_evidence),
                "contact_lookup_key": [int(obs.contact_lookup_key[0]), int(obs.contact_lookup_key[1])],
                "contact_row_track_id": obs.contact_row_track_id,
                "contact_support": obs.contact_support,
                "contact_probability": float(contact_prob[i]),
                "translation_delta_norm_m": float(torch.linalg.norm(state["trans_delta"][i]).detach().cpu()),
                "pose_delta_abs_max_rad": float(torch.max(torch.abs(state["pose_delta"][i])).detach().cpu()),
                "orient_delta_abs_max_rad": float(torch.max(torch.abs(state["orient_delta"][i])).detach().cpu()),
                "scale": float(torch.exp(state["log_scale"][i]).detach().cpu()),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "has_rtmlib_evidence": bool(len(rtmlib_reproj) > 0),
                "rtmlib_keypoint_reprojection_px": summarize(rtmlib_reproj),
                "has_hand_mask_evidence": bool(obs.hand_mask_distance is not None),
                "hand_silhouette_distance_px": summarize(hand_silhouette),
                "hand_silhouette_inside_fraction": None if len(hand_silhouette) == 0 else float(np.mean(hand_silhouette <= 0.5)),
                "contact_abs_sdf_m": summarize(np.abs(sdf)),
                "contact_sdf_m": summarize(sdf),
                "contact_near_surface_fraction": None if len(sdf) == 0 else float(np.mean(np.abs(sdf) <= float(args.accept_contact_abs_sdf_m))),
                "contact_penetration_fraction": None if len(sdf) == 0 else float(np.mean(sdf < -float(args.penetration_tolerance_m))),
            }
        )
        arrays["vertices"].append(vertices)
        arrays["joints"].append(joints)
        arrays["local_vertices"].append(solved["local_vertices"][i].detach().cpu().numpy())
        arrays["local_joints"].append(solved["local_joints"][i].detach().cpu().numpy())
        arrays["global_orient"].append(solved["global_orient"][i].detach().cpu().numpy())
        arrays["hand_pose"].append(solved["hand_pose"][i].detach().cpu().numpy())
    return rows, arrays


def summarize_rows(rows: list[dict]) -> dict:
    contact_rows = [row for row in rows if bool(row.get("has_contact_evidence"))]
    non_contact_rows = [row for row in rows if not bool(row.get("has_contact_evidence"))]
    contact_sdf = [
        row["contact_abs_sdf_m"].get("p95")
        for row in contact_rows
        if isinstance(row.get("contact_abs_sdf_m"), dict)
    ]
    non_contact_sdf = [
        row["contact_abs_sdf_m"].get("p95")
        for row in non_contact_rows
        if isinstance(row.get("contact_abs_sdf_m"), dict)
    ]
    hand_mask_rows = [row for row in rows if bool(row.get("has_hand_mask_evidence"))]
    rtmlib_rows = [row for row in rows if bool(row.get("has_rtmlib_evidence"))]
    return {
        "rows": int(len(rows)),
        "contact_evidence_rows": int(len(contact_rows)),
        "non_contact_rows": int(len(non_contact_rows)),
        "hand_mask_evidence_rows": int(len(hand_mask_rows)),
        "rtmlib_evidence_rows": int(len(rtmlib_rows)),
        "contact_probability": summarize([row["contact_probability"] for row in rows]),
        "translation_delta_norm_m": summarize([row["translation_delta_norm_m"] for row in rows]),
        "pose_delta_abs_max_rad": summarize([row["pose_delta_abs_max_rad"] for row in rows]),
        "scale": summarize([row["scale"] for row in rows]),
        "keypoint_reprojection_median_px": summarize([row["keypoint_reprojection_median_px"] for row in rows]),
        "hand_silhouette_distance_p95_px": summarize(
            [
                row["hand_silhouette_distance_px"].get("p95")
                for row in hand_mask_rows
                if isinstance(row.get("hand_silhouette_distance_px"), dict)
            ]
        ),
        "hand_silhouette_inside_fraction": summarize(
            [row["hand_silhouette_inside_fraction"] for row in hand_mask_rows if row["hand_silhouette_inside_fraction"] is not None]
        ),
        "rtmlib_keypoint_reprojection_median_px": summarize(
            [
                row["rtmlib_keypoint_reprojection_px"].get("median")
                for row in rtmlib_rows
                if isinstance(row.get("rtmlib_keypoint_reprojection_px"), dict)
            ]
        ),
        "rtmlib_keypoint_reprojection_p95_px": summarize(
            [
                row["rtmlib_keypoint_reprojection_px"].get("p95")
                for row in rtmlib_rows
                if isinstance(row.get("rtmlib_keypoint_reprojection_px"), dict)
            ]
        ),
        "contact_evidence_abs_sdf_p95_m": summarize(contact_sdf),
        "non_contact_sample_abs_sdf_p95_m": summarize(non_contact_sdf),
        "contact_near_surface_fraction": summarize([row["contact_near_surface_fraction"] for row in rows if row["contact_near_surface_fraction"] is not None]),
        "contact_penetration_fraction": summarize([row["contact_penetration_fraction"] for row in rows if row["contact_penetration_fraction"] is not None]),
    }


def apply_solution(annotations: dict, observations: list[V8Obs], state: dict[str, torch.Tensor], arrays: dict[str, list[np.ndarray]]) -> dict:
    out = copy.deepcopy(annotations)
    obs_index = {(obs.frame_idx, obs.hand_idx): i for i, obs in enumerate(observations)}
    contact_prob = torch.sigmoid(state["contact_logit"]).detach().cpu().numpy()
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        for hand_idx, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, hand_idx)
            if key not in obs_index:
                continue
            i = obs_index[key]
            joints = arrays["joints"][i]
            vertices = arrays["vertices"][i]
            local_joints = arrays["local_joints"][i]
            local_vertices = arrays["local_vertices"][i]
            intr = observations[i].intrinsics.detach().cpu().numpy()
            joints2d = project_torch(torch.tensor(joints, dtype=torch.float32), torch.tensor(intr, dtype=torch.float32)).numpy()
            hand["joints3d_source_camera_m"] = joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = vertices.astype(float).tolist()
            hand["joints3d_camera"] = local_joints.astype(float).tolist()
            hand["vertices_camera"] = local_vertices.astype(float).tolist()
            hand["joints2d"] = joints2d.astype(float).tolist()
            hand["joints3d_world_m"] = source_to_world(joints, T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(vertices, T_world_camera).astype(float).tolist()
            hand["mano_params"]["global_orient"] = arrays["global_orient"][i].reshape(1, 3, 3).astype(float).tolist()
            hand["mano_params"]["hand_pose"] = arrays["hand_pose"][i][0].astype(float).tolist()
            hand["v8_contact_aware_mano_graph"] = {
                "contact_probability": float(contact_prob[i]),
                "translation_delta_m": state["trans_delta"][i].detach().cpu().numpy().reshape(-1).astype(float).tolist(),
                "pose_delta_abs_max_rad": float(torch.max(torch.abs(state["pose_delta"][i])).detach().cpu()),
                "orient_delta_abs_max_rad": float(torch.max(torch.abs(state["orient_delta"][i])).detach().cpu()),
                "scale": float(torch.exp(state["log_scale"][i]).detach().cpu()),
            }
    return out


def render_review(annotations_before: dict, annotations_after: dict, observations: list[V8Obs], args: argparse.Namespace) -> dict:
    if args.output_review_video is None:
        return {"enabled": False}
    args.output_review_video.parent.mkdir(parents=True, exist_ok=True)
    frames_before = frame_map(annotations_before)
    frames_after = frame_map(annotations_after)
    writer = cv2.VideoWriter(
        str(args.output_review_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.output_fps),
        (int(args.review_width) * 2, int(args.review_height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open review video writer: {args.output_review_video}")
    obs_by_frame = {obs.frame_idx: obs for obs in observations}
    written = 0
    try:
        for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
            if frame_idx not in obs_by_frame:
                continue
            before = draw_review_frame(frames_before[frame_idx], obs_by_frame[frame_idx], "before V8", args)
            after = draw_review_frame(frames_after[frame_idx], obs_by_frame[frame_idx], "after V8", args)
            writer.write(np.hstack([before, after]))
            written += 1
    finally:
        writer.release()
    return {"enabled": True, "video": str(args.output_review_video), "frames": int(written)}


def rgb_path_for_frame(frame: dict, frame_idx: int, args: argparse.Namespace) -> Path | None:
    raw = frame.get("rgb_path") or frame.get("image_path")
    if isinstance(raw, str):
        path = Path(raw)
        if path.exists():
            return path
    if args.review_rgb_dir is None:
        return None
    path = args.review_rgb_dir / (str(args.review_rgb_pattern).format(frame_idx=int(frame_idx)))
    if path.exists():
        return path
    return None


def draw_review_frame(frame: dict, obs: V8Obs, label: str, args: argparse.Namespace) -> np.ndarray:
    image = np.zeros((int(args.review_height), int(args.review_width), 3), dtype=np.uint8)
    rgb_path = rgb_path_for_frame(frame, int(obs.frame_idx), args)
    if rgb_path is not None:
        raw = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if raw is not None:
            image = cv2.resize(raw, (int(args.review_width), int(args.review_height)), interpolation=cv2.INTER_AREA)
    else:
        mask = object_mask_for_frame(frame, obs.source_size, args)
        image = cv2.cvtColor((mask.astype(np.uint8) * 180), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (int(args.review_width), int(args.review_height)), interpolation=cv2.INTER_NEAREST)
    hand = frame["hands"][obs.hand_idx]
    joints = np.asarray(hand["joints2d"], dtype=np.float32)
    scale = np.asarray([int(args.review_width) / float(obs.source_size[0]), int(args.review_height) / float(obs.source_size[1])], dtype=np.float32)
    if obs.hand_mask is not None:
        mask = cv2.resize(obs.hand_mask.astype(np.uint8), (int(args.review_width), int(args.review_height)), interpolation=cv2.INTER_NEAREST) > 0
        color = np.zeros_like(image)
        color[:, :, 1] = 180
        image = np.where(mask[..., None], (0.75 * image + 0.25 * color).astype(np.uint8), image)
    if obs.rtmlib_joints2d is not None and obs.rtmlib_scores is not None:
        rtm = obs.rtmlib_joints2d.detach().cpu().numpy()
        scores = obs.rtmlib_scores.detach().cpu().numpy()
        valid = np.isfinite(rtm).all(axis=1) & np.isfinite(scores) & (scores >= float(args.rtmlib_min_score))
        for xy in rtm[valid] * scale[None, :]:
            cv2.circle(image, (int(round(xy[0])), int(round(xy[1]))), 3, (255, 210, 40), -1, cv2.LINE_AA)
    draw_skeleton(image, joints * scale[None, :], (80, 255, 80))
    cv2.putText(image, f"{label} frame {obs.frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_skeleton(image: np.ndarray, joints: np.ndarray, color: tuple[int, int, int]) -> None:
    edges = [
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
    if joints.shape != (21, 2):
        return
    for a, b in edges:
        cv2.line(image, tuple(np.rint(joints[a]).astype(int)), tuple(np.rint(joints[b]).astype(int)), color, 3, cv2.LINE_AA)
    for point in joints:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict:
    patch_legacy_mano_loader()
    annotations = load_json(args.annotations)
    mano_class = load_wilor_mano_class(args.mano_wrapper_root)
    model = mano_class(
        model_path=str(args.mano_model_root),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
        batch_size=1,
    ).to(torch.device(args.device))
    observations, skipped = build_observations(model, annotations, args)
    initial_state = {
        "pose_delta": torch.zeros((len(observations), 1, 15, 3), dtype=torch.float32, device=torch.device(args.device)),
        "orient_delta": torch.zeros((len(observations), 1, 1, 3), dtype=torch.float32, device=torch.device(args.device)),
        "trans_delta": torch.zeros((len(observations), 1, 3), dtype=torch.float32, device=torch.device(args.device)),
        "log_scale": torch.zeros((len(observations), 1, 1, 1), dtype=torch.float32, device=torch.device(args.device)),
        "contact_logit": torch.tensor([contact_prior_logit(obs.contact_seed) for obs in observations], dtype=torch.float32, device=torch.device(args.device)),
    }
    initial_rows, _ = metrics_for_state(model, observations, initial_state, args)
    best_state, loss_rows = solve(model, observations, args)
    final_rows, arrays = metrics_for_state(model, observations, best_state, args)
    output_annotations = apply_solution(annotations, observations, best_state, arrays)
    save_json(args.output_annotations, output_annotations)
    review = render_review(annotations, output_annotations, observations, args)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_contact_aware_mano_graph_v8",
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "object_mesh_npz": str(args.object_mesh_npz),
        "metric_depth_npz": str(args.metric_depth_npz),
        "contact_report": str(args.contact_report),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "observations": int(len(observations)),
        "skipped_count": int(len(skipped)),
        "contact_observations": int(sum(1 for obs in observations if obs.has_contact_evidence)),
        "loss_history": loss_rows,
        "initial": summarize_rows(initial_rows),
        "final": summarize_rows(final_rows),
        "rows_initial": initial_rows,
        "rows_final": final_rows,
        "skipped": skipped,
        "review": review,
        "acceptance_note": "V8 graph output is a candidate annotation stream. It becomes deliverable only after unchanged V7 replay, track-surface, physics, and visual render inspection pass.",
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_initial", "rows_final", "skipped"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--output-review-video", type=Path)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--rtmlib-json", type=Path)
    parser.add_argument("--rtmlib-prompts", type=Path)
    parser.add_argument("--hand-mask-evidence-json", type=Path)
    parser.add_argument("--hand-mask-remote-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote"))
    parser.add_argument("--hand-mask-local-root", type=Path, default=Path("/data2/ego_annotation_outputs/representative_box_books"))
    parser.add_argument("--hand-mask-width", type=int, default=960)
    parser.add_argument("--hand-mask-height", type=int, default=540)
    parser.add_argument("--mano-wrapper-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--mano-model-root", type=Path, default=Path("third_party/WiLoR/mano_data"))
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--side")
    parser.add_argument("--track-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--max-sampled-vertices", type=int, default=900)
    parser.add_argument("--max-contact-vertices", type=int, default=160)
    parser.add_argument("--min-patch-vertices", type=int, default=16)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--contact-patch-hand-mask-max-px", type=float, default=4.0)
    parser.add_argument("--contact-patch-min-hand-mask-inside-fraction", type=float, default=0.50)
    parser.add_argument("--contact-patch-max-nearest-rtmlib-px", type=float, default=35.0)
    parser.add_argument("--local-sdf-crop-margin-m", type=float, default=0.050)
    parser.add_argument("--local-sdf-min-faces", type=int, default=512)
    parser.add_argument("--sdf-pitch-m", type=float, default=0.002)
    parser.add_argument("--sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-zero-state-vertex-error-m", type=float, default=0.004)
    parser.add_argument("--max-zero-state-joint-error-m", type=float, default=0.004)
    parser.add_argument("--sigma-keypoint-px", type=float, default=20.0)
    parser.add_argument("--sigma-rtmlib-keypoint-px", type=float, default=18.0)
    parser.add_argument("--sigma-silhouette-px", type=float, default=12.0)
    parser.add_argument("--sigma-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-contact-sdf-m", type=float, default=0.010)
    parser.add_argument("--sigma-penetration-m", type=float, default=0.006)
    parser.add_argument("--sigma-contact-logit", type=float, default=1.25)
    parser.add_argument("--sigma-pose-delta-rad", type=float, default=0.25)
    parser.add_argument("--sigma-orient-delta-rad", type=float, default=0.20)
    parser.add_argument("--sigma-translation-m", type=float, default=0.050)
    parser.add_argument("--sigma-log-scale", type=float, default=0.08)
    parser.add_argument("--sigma-temporal-translation-m", type=float, default=0.020)
    parser.add_argument("--sigma-temporal-pose-rad", type=float, default=0.18)
    parser.add_argument("--sigma-temporal-contact-logit", type=float, default=1.0)
    parser.add_argument("--sigma-span-m", type=float, default=0.015)
    parser.add_argument("--min-span-m", type=float, default=0.115)
    parser.add_argument("--max-span-m", type=float, default=0.240)
    parser.add_argument("--w-keypoint", type=float, default=1.0)
    parser.add_argument("--w-rtmlib-keypoint", type=float, default=0.8)
    parser.add_argument("--w-silhouette", type=float, default=0.5)
    parser.add_argument("--w-depth", type=float, default=0.5)
    parser.add_argument("--w-contact", type=float, default=1.4)
    parser.add_argument("--w-contact-tail", type=float, default=0.0)
    parser.add_argument("--w-penetration", type=float, default=2.0)
    parser.add_argument("--w-contact-prior", type=float, default=0.4)
    parser.add_argument("--w-span", type=float, default=0.4)
    parser.add_argument("--w-pose-prior", type=float, default=0.5)
    parser.add_argument("--w-orient-prior", type=float, default=0.4)
    parser.add_argument("--w-translation-prior", type=float, default=0.4)
    parser.add_argument("--w-scale-prior", type=float, default=0.3)
    parser.add_argument("--w-temporal-translation", type=float, default=1.0)
    parser.add_argument("--w-temporal-pose", type=float, default=0.6)
    parser.add_argument("--w-temporal-contact", type=float, default=0.25)
    parser.add_argument("--rtmlib-min-score", type=float, default=0.20)
    parser.add_argument("--rtmlib-min-keypoints", type=int, default=8)
    parser.add_argument("--max-silhouette-distance-px", type=float, default=64.0)
    parser.add_argument("--invalid-sdf-m", type=float, default=1.0)
    parser.add_argument("--accept-contact-abs-sdf-m", type=float, default=0.006)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.002)
    parser.add_argument("--max-pose-delta-rad", type=float, default=0.65)
    parser.add_argument("--max-orient-delta-rad", type=float, default=0.45)
    parser.add_argument("--max-translation-m", type=float, default=0.060)
    parser.add_argument("--min-scale", type=float, default=0.92)
    parser.add_argument("--max-scale", type=float, default=1.08)
    parser.add_argument("--max-abs-contact-logit", type=float, default=5.0)
    parser.add_argument("--output-fps", type=float, default=6.0)
    parser.add_argument("--review-width", type=int, default=960)
    parser.add_argument("--review-height", type=int, default=540)
    parser.add_argument("--review-rgb-dir", type=Path)
    parser.add_argument("--review-rgb-pattern", default="{frame_idx:06d}.jpg")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
