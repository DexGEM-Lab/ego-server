#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from diagnose_contact_depth_conflict_v3 import mesh_frame_vertices, summarize
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


@dataclass(frozen=True)
class SurfaceObs:
    track_id: str
    object_depth_m: float
    near_vertex_indices: np.ndarray
    contact_seed: float


@dataclass(frozen=True)
class HandObs:
    key: tuple[int, str, int]
    frame_idx: int
    side: str
    hand_index: int
    detector_score: float
    base_cam_t: np.ndarray
    local_joints: np.ndarray
    local_vertices: np.ndarray
    wilor2d: np.ndarray
    wilor_weight: np.ndarray
    rtmlib2d: np.ndarray
    rtmlib_weight: np.ndarray
    intrinsics: np.ndarray
    metric_depth: np.ndarray
    depth_weight: np.ndarray
    surfaces: tuple[SurfaceObs, ...]
    surface_reports: tuple[dict, ...]
    contract_error_m: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def source_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def hand_vertex_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    if "vertices_camera" in hand:
        return "vertices_camera"
    if "vertices_source_camera_m_sample" in hand:
        return "vertices_source_camera_m_sample"
    if "vertices_camera_sample" in hand:
        return "vertices_camera_sample"
    raise RuntimeError("hand has no source-camera vertices")


def bone_scale(joints: np.ndarray) -> float:
    lengths = []
    for chain in FINGER_CHAINS:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def depth_patch_iqr_ratio(depth: np.ndarray, xy: np.ndarray, radius: int) -> float:
    x = int(np.clip(round(float(xy[0])), 0, depth.shape[1] - 1))
    y = int(np.clip(round(float(xy[1])), 0, depth.shape[0] - 1))
    patch = depth[
        max(0, y - radius) : min(depth.shape[0], y + radius + 1),
        max(0, x - radius) : min(depth.shape[1], x + radius + 1),
    ]
    vals = patch[np.isfinite(patch) & (patch > 0.0)]
    if len(vals) == 0:
        return float("nan")
    med = float(np.median(vals))
    return float((np.percentile(vals, 75.0) - np.percentile(vals, 25.0)) / max(1e-6, med))


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def object_camera_depth(vertices_world: np.ndarray, T_world_camera: np.ndarray) -> float:
    homog = np.c_[vertices_world, np.ones(len(vertices_world), dtype=float)]
    camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if len(z) == 0:
        raise RuntimeError("object mesh has no positive camera depth")
    return float(np.median(z))


def surface_vertices_near_mask(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    source_size: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if np.any(vertices[:, 2] <= 0.0):
        return np.empty(0, dtype=int)
    dist = mask_distance_map(resize_mask_to_depth(mask, depth))
    scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    uv = project_points(vertices, intrinsics) * scale[None, :]
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = np.flatnonzero(valid & (dist[y, x] <= args.contact_distance_px))
    if len(near) <= args.max_contact_vertices:
        return near.astype(int)
    z = vertices[near, 2]
    keep = np.argsort(np.abs(z - np.median(z)))[: args.max_contact_vertices]
    return near[keep].astype(int)


def load_rtmlib_matches(path: Path | None, good_match_px: float) -> dict[tuple[int, str, int], dict]:
    if path is None:
        return {}
    data = load_json(path)
    out: dict[tuple[int, str, int], dict] = {}
    for row in data.get("rows_preview", []):
        frame_idx = int(row["frame_idx"])
        for match in row.get("matches", []):
            if float(match["median_keypoint_delta_px"]) > good_match_px:
                continue
            key = (frame_idx, str(match.get("wilor_side", "unknown")), int(match.get("wilor_list_idx", -1)))
            if key not in out or float(match["median_keypoint_delta_px"]) < float(out[key]["median_keypoint_delta_px"]):
                out[key] = match
    return out


def load_rtmlib_frames(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    data = load_json(path)
    return {int(frame["frame_idx"]): frame for frame in data.get("frames", [])}


def target_from_rtmlib(
    frame_idx: int,
    hand_i: int,
    side: str,
    matches: dict[tuple[int, str, int], dict],
    rtmlib_frames: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    match = matches.get((frame_idx, side, hand_i))
    if match is None:
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float)
    frame = rtmlib_frames.get(frame_idx)
    if frame is None:
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float)
    hands = frame.get("hands", [])
    list_idx = int(match["rtmlib_list_idx"])
    if list_idx < 0 or list_idx >= len(hands):
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float)
    points = np.asarray(hands[list_idx].get("keypoints", []), dtype=float)
    scores = np.asarray(hands[list_idx].get("scores", []), dtype=float)
    if points.shape != (21, 2) or scores.shape != (21,):
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float)
    good = np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= args.min_rtmlib_keypoint_score)
    weights = np.zeros(21, dtype=float)
    weights[good] = 1.0
    return points, weights


def surface_jobs(args: argparse.Namespace) -> list[dict]:
    manifest = load_json(args.postprocess_manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"postprocess manifest has no jobs: {args.postprocess_manifest}")
    return jobs


def surface_frame_tables(jobs: list[dict]) -> list[tuple[dict, dict[int, dict]]]:
    tables = []
    for job in jobs:
        annotations = load_json(Path(job["adapted_annotations"]))
        tables.append((job, {int(frame["frame_idx"]): frame for frame in annotations["frames"]}))
    return tables


def frame_surface_obs(
    frame_idx: int,
    source_vertices: np.ndarray,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    T_world_camera: np.ndarray,
    surfaces: list[tuple[dict, dict[int, dict]]],
    args: argparse.Namespace,
) -> tuple[tuple[SurfaceObs, ...], tuple[dict, ...]]:
    out = []
    reports = []
    for job, frames in surfaces:
        track_id = str(job["track_id"])
        frame = frames.get(frame_idx)
        if frame is None:
            reports.append({"track_id": track_id, "status": "not_available", "reason": "missing_surface_annotation_frame"})
            continue
        obj = frame.get("object", {})
        if not obj.get("mask_path"):
            reports.append({"track_id": track_id, "status": "not_available", "reason": str(obj.get("status", "no_mask_path"))})
            continue
        source_size = np.asarray(obj["source_image_size"], dtype=float)
        mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
        mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
        object_vertices = mesh_frame_vertices(Path(job["mesh_dir"]) / "dynamic_object_meshes.npz", frame_idx)
        object_depth = object_camera_depth(object_vertices, T_world_camera)
        near = surface_vertices_near_mask(source_vertices, intrinsics, mask, depth, source_size, args)
        if len(near) < args.min_near_vertices:
            reports.append({"track_id": track_id, "status": "underconstrained", "reason": "too_few_near_vertices", "near_vertices": int(len(near))})
            continue
        gap = source_vertices[near, 2] - object_depth
        seed = float(np.exp(-abs(float(np.median(gap))) / max(args.contact_seed_scale_m, 1e-6)))
        out.append(SurfaceObs(track_id, object_depth, near, seed))
        reports.append({"track_id": track_id, "status": "available", "near_vertices": int(len(near)), "object_depth_m": float(object_depth), "contact_seed": seed})
    return tuple(out), tuple(reports)


def build_observations(args: argparse.Namespace) -> tuple[dict, list[HandObs], list[dict]]:
    base = load_json(args.base_annotations)
    frames = {int(frame["frame_idx"]): frame for frame in base["frames"]}
    jobs = surface_jobs(args)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_indices)) != len(depth_indices):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_indices)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    matches = load_rtmlib_matches(args.rtmlib_wilor_qc, args.max_rtmlib_wilor_delta_px)
    rtmlib_frames = load_rtmlib_frames(args.rtmlib_json)
    surfaces_by_track = surface_frame_tables(jobs)
    observations: list[HandObs] = []
    skipped: list[dict] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_base_frame"})
            continue
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        source_size = np.asarray(frame.get("object", {}).get("source_image_size", [1920, 1080]), dtype=float)
        scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
        for hand_i, hand in enumerate(frame.get("hands", [])):
            side = str(hand.get("side", "unknown"))
            try:
                measured = bool(hand.get("measurement_available", False))
                score = float(hand.get("detector_score", np.nan))
                if not measured or not np.isfinite(score) or score < args.min_detector_score:
                    raise RuntimeError("hand_not_measured_or_low_score")
                local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                local_vertices = np.asarray(hand["vertices_camera"], dtype=float)
                source_joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                source_vertices = np.asarray(hand[hand_vertex_key(hand)], dtype=float)
                cam_t = np.asarray(hand["cam_t"], dtype=float)
                raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
                intrinsics = np.asarray(hand["source_intrinsics"], dtype=float)
                if local_joints.shape != (21, 3) or source_joints.shape != (21, 3) or cam_t.shape != (3,):
                    raise RuntimeError("invalid hand joint geometry")
                if local_vertices.ndim != 2 or local_vertices.shape[1] != 3 or source_vertices.shape != local_vertices.shape:
                    raise RuntimeError("invalid hand vertex geometry")
                if raw2d.shape != (21, 2) or intrinsics.shape != (4,):
                    raise RuntimeError("invalid hand keypoint geometry")
                contract = max(
                    float(np.max(np.linalg.norm(local_joints + cam_t[None, :] - source_joints, axis=1))),
                    float(np.max(np.linalg.norm(local_vertices + cam_t[None, :] - source_vertices, axis=1))),
                )
                if contract > args.max_contract_error_m:
                    raise RuntimeError(f"hand_source_contract_error_{contract:.4f}m")
                projected = project_points(source_joints, intrinsics)
                reproj = np.linalg.norm(projected - raw2d, axis=1)
                metric_depth = sample_depth(depth, raw2d, source_size)
                patch = np.asarray([depth_patch_iqr_ratio(depth, xy * scale, args.patch_radius) for xy in raw2d], dtype=float)
                depth_weight = np.zeros(21, dtype=float)
                depth_weight[
                    np.isfinite(metric_depth)
                    & (metric_depth > 0.0)
                    & (reproj <= args.good_joint_reprojection_px)
                    & np.isfinite(patch)
                    & (patch <= args.max_depth_iqr_ratio)
                ] = 1.0
                if int(np.count_nonzero(depth_weight)) < args.min_depth_keypoints:
                    raise RuntimeError("too_few_stable_depth_keypoints")
                rtmlib2d, rtmlib_weight = target_from_rtmlib(frame_idx, hand_i, side, matches, rtmlib_frames, args)
                surfaces, surface_reports = frame_surface_obs(frame_idx, source_vertices, intrinsics, depth, T_world_camera, surfaces_by_track, args)
                observations.append(
                    HandObs(
                        key=(frame_idx, side, hand_i),
                        frame_idx=frame_idx,
                        side=side,
                        hand_index=hand_i,
                        detector_score=score,
                        base_cam_t=cam_t,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        wilor2d=raw2d,
                        wilor_weight=np.ones(21, dtype=float),
                        rtmlib2d=rtmlib2d,
                        rtmlib_weight=rtmlib_weight,
                        intrinsics=intrinsics,
                        metric_depth=metric_depth,
                        depth_weight=depth_weight,
                        surfaces=surfaces,
                        surface_reports=surface_reports,
                        contract_error_m=contract,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": side, "reason": str(exc)})
    if len(observations) < args.min_observations:
        raise RuntimeError(f"insufficient observations: {len(observations)}; skipped={skipped[:12]}")
    return base, observations, skipped


def pack_initial(observations: list[HandObs]) -> np.ndarray:
    n = len(observations)
    shifts = np.zeros((n, 3), dtype=float)
    velocities = np.zeros((n, 3), dtype=float)
    logits = []
    for obs in observations:
        if obs.surfaces:
            seed = max(surface.contact_seed for surface in obs.surfaces)
            p = np.clip(0.05 + 0.90 * seed, 1e-4, 1.0 - 1e-4)
        else:
            p = 0.02
        logits.append(math.log(p / (1.0 - p)))
    return np.r_[0.0, shifts.reshape(-1), velocities.reshape(-1), np.asarray(logits, dtype=float)]


def unpack(params: np.ndarray, n: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale_log = float(params[0])
    offset = 1
    shifts = np.asarray(params[offset : offset + 3 * n], dtype=float).reshape(n, 3)
    offset += 3 * n
    velocities = np.asarray(params[offset : offset + 3 * n], dtype=float).reshape(n, 3)
    offset += 3 * n
    logits = np.asarray(params[offset : offset + n], dtype=float)
    contact = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return scale_log, shifts, velocities, logits, contact


def corrected(obs: HandObs, scale_log: float, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = math.exp(float(scale_log))
    cam_t = scale * obs.base_cam_t + shift
    joints = scale * obs.local_joints + cam_t[None, :]
    vertices = scale * obs.local_vertices + cam_t[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("nonpositive corrected hand depth")
    return cam_t, joints, vertices


def residual(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    scale_log, shifts, velocities, logits, contact = unpack(params, len(observations))
    out: list[np.ndarray] = [np.asarray([scale_log / args.sigma_hand_log_scale], dtype=float)]
    for i, obs in enumerate(observations):
        try:
            _, joints, vertices = corrected(obs, scale_log, shifts[i])
        except RuntimeError:
            out.append(np.full(42, args.invalid_penalty, dtype=float))
            if np.any(obs.rtmlib_weight > 0.0):
                out.append(np.full(42, args.invalid_penalty, dtype=float))
            out.append(np.full(int(np.count_nonzero(obs.depth_weight > 0.0)), args.invalid_penalty, dtype=float))
            out.append(np.full(7, args.invalid_penalty, dtype=float))
            if obs.surfaces:
                for surface in obs.surfaces:
                    n_gap = min(len(surface.near_vertex_indices), args.max_contact_vertices)
                    out.append(np.full(2 * n_gap, args.invalid_penalty, dtype=float))
                out.append(np.asarray([args.invalid_penalty], dtype=float))
            else:
                out.append(np.asarray([args.invalid_penalty], dtype=float))
            continue
        uv = project_points(joints, obs.intrinsics)
        wilor = ((uv - obs.wilor2d) / args.sigma_wilor_reprojection_px) * obs.wilor_weight[:, None]
        out.append(wilor.reshape(-1))
        if np.any(obs.rtmlib_weight > 0.0):
            rtm = ((uv - obs.rtmlib2d) / args.sigma_rtmlib_reprojection_px) * obs.rtmlib_weight[:, None]
            out.append(rtm.reshape(-1))
        valid_depth = obs.depth_weight > 0.0
        if np.any(valid_depth):
            out.append((joints[valid_depth, 2] - obs.metric_depth[valid_depth]) / args.sigma_metric_depth_m)
        out.append(shifts[i] / args.sigma_translation_prior_m)
        out.append(velocities[i] / args.sigma_velocity_prior_m)
        out.append(np.asarray([(bone_scale(joints) - args.hand_bone_scale_prior_m) / args.sigma_bone_scale_m]))
        if obs.surfaces:
            p0 = np.clip(0.05 + 0.90 * max(surface.contact_seed for surface in obs.surfaces), 1e-4, 1.0 - 1e-4)
            surface_medians = []
            surface_terms = []
            for surface in obs.surfaces:
                gap = vertices[surface.near_vertex_indices, 2] - surface.object_depth_m
                if len(gap) > args.max_contact_vertices:
                    gap = gap[np.linspace(0, len(gap) - 1, args.max_contact_vertices).astype(int)]
                med = float(np.median(gap))
                surface_medians.append(abs(med))
                surface_terms.append((abs(med), gap))
            best_i = int(np.argmin(np.asarray(surface_medians, dtype=float)))
            for surface_i, (_, gap) in enumerate(surface_terms):
                use_weight = 1.0 if surface_i == best_i else args.nonselected_surface_weight
                attraction = use_weight * math.sqrt(max(float(contact[i]), 1e-6)) * gap / args.sigma_contact_depth_m
                out.append(np.clip(attraction, -args.clip_residual, args.clip_residual))
                penetration = use_weight * np.minimum(gap, 0.0) / args.sigma_penetration_m
                out.append(np.clip(penetration, -args.clip_residual, args.clip_residual))
            out.append(np.asarray([(logits[i] - math.log(p0 / (1.0 - p0))) / args.sigma_contact_logit]))
        else:
            out.append(np.asarray([contact[i] / args.sigma_no_surface_contact]))
    by_side: dict[str, list[int]] = {}
    for i, obs in enumerate(observations):
        by_side.setdefault(obs.side, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda i: observations[i].frame_idx)
        for a, b in zip(indices[:-1], indices[1:]):
            dt = max(1.0, float(observations[b].frame_idx - observations[a].frame_idx) / float(args.fps))
            predicted = shifts[a] + velocities[a] * dt
            out.append((shifts[b] - predicted) / args.sigma_motion_m)
            out.append((velocities[b] - velocities[a]) / args.sigma_acceleration_m)
            out.append(np.asarray([(logits[b] - logits[a]) / (args.sigma_contact_logit_step * max(1.0, dt))]))
    return np.concatenate([np.ravel(x).astype(float) for x in out])


def solve(observations: list[HandObs], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    x0 = pack_initial(observations)
    n = len(observations)
    lower = np.r_[
        np.log(args.min_hand_scale),
        np.full(3 * n, -args.max_abs_translation_m),
        np.full(3 * n, -args.max_abs_velocity_mps),
        np.full(n, -args.max_abs_contact_logit),
    ]
    upper = np.r_[
        np.log(args.max_hand_scale),
        np.full(3 * n, args.max_abs_translation_m),
        np.full(3 * n, args.max_abs_velocity_mps),
        np.full(n, args.max_abs_contact_logit),
    ]
    before = residual(x0, observations, args)
    result = least_squares(
        lambda x: residual(x, observations, args),
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
        verbose=0,
    )
    after = residual(result.x, observations, args)
    return result.x, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "rms_before": float(np.sqrt(np.mean(before * before))),
        "rms_after": float(np.sqrt(np.mean(after * after))),
    }


def best_surface_gap(obs: HandObs, vertices: np.ndarray) -> tuple[str | None, np.ndarray]:
    if not obs.surfaces:
        return None, np.asarray([], dtype=float)
    rows = []
    for surface in obs.surfaces:
        gap = vertices[surface.near_vertex_indices, 2] - surface.object_depth_m
        rows.append((str(surface.track_id), gap, abs(float(np.median(gap))) if len(gap) else float("inf")))
    track, gap, _ = min(rows, key=lambda row: row[2])
    return track, gap


def row_metrics(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> list[dict]:
    scale_log, shifts, velocities, _, contact = unpack(params, len(observations))
    rows = []
    for i, obs in enumerate(observations):
        _, joints, vertices = corrected(obs, scale_log, shifts[i])
        uv = project_points(joints, obs.intrinsics)
        reproj = np.linalg.norm(uv - obs.wilor2d, axis=1)
        rtm_valid = obs.rtmlib_weight > 0.0
        rtm_reproj = np.linalg.norm(uv[rtm_valid] - obs.rtmlib2d[rtm_valid], axis=1)
        valid_depth = obs.depth_weight > 0.0
        depth_gap = joints[valid_depth, 2] - obs.metric_depth[valid_depth]
        track_id, contact_gap = best_surface_gap(obs, vertices)
        rows.append(
            {
                "frame_idx": int(obs.frame_idx),
                "side": obs.side,
                "hand_index": int(obs.hand_index),
                "detector_score": float(obs.detector_score),
                "hand_scale": float(math.exp(scale_log)),
                "translation_shift_norm_m": float(np.linalg.norm(shifts[i])),
                "velocity_norm_mps": float(np.linalg.norm(velocities[i])),
                "contact_probability": float(contact[i]),
                "surfaces_available": int(len(obs.surfaces)),
                "surface_reports": list(obs.surface_reports),
                "selected_surface": track_id,
                "contract_error_m": float(obs.contract_error_m),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "rtmlib_keypoint_reprojection_median_px": None if len(rtm_reproj) == 0 else float(np.median(rtm_reproj)),
                "metric_depth_keypoints": int(np.count_nonzero(valid_depth)),
                "mano_minus_metric_depth_median_m": None if len(depth_gap) == 0 else float(np.median(depth_gap)),
                "mano_minus_metric_depth_p95_abs_m": None if len(depth_gap) == 0 else float(np.percentile(np.abs(depth_gap), 95.0)),
                "contact_gap_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
                "contact_gap_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
                "hand_bone_scale_m": bone_scale(joints),
            }
        )
    return rows


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values.append(value_f)
    return summarize(np.asarray(values, dtype=float))


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "rows": int(len(rows)),
        "hand_scale": summarize_key(rows, "hand_scale"),
        "translation_shift_norm_m": summarize_key(rows, "translation_shift_norm_m"),
        "velocity_norm_mps": summarize_key(rows, "velocity_norm_mps"),
        "contact_probability": summarize_key(rows, "contact_probability"),
        "keypoint_reprojection_median_px": summarize_key(rows, "keypoint_reprojection_median_px"),
        "rtmlib_keypoint_reprojection_median_px": summarize_key(rows, "rtmlib_keypoint_reprojection_median_px"),
        "mano_minus_metric_depth_median_m": summarize_key(rows, "mano_minus_metric_depth_median_m"),
        "mano_minus_metric_depth_p95_abs_m": summarize_key(rows, "mano_minus_metric_depth_p95_abs_m"),
        "contact_gap_median_m": summarize_key(rows, "contact_gap_median_m"),
        "contact_gap_p95_abs_m": summarize_key(rows, "contact_gap_p95_abs_m"),
        "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
    }


def status_for(rows: list[dict], params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> str:
    scale_log, shifts, velocities, _, contact = unpack(params, len(observations))
    scale = math.exp(scale_log)
    shift_bound = bool(np.any(np.abs(shifts) >= args.max_abs_translation_m - args.bound_tolerance))
    velocity_bound = bool(np.any(np.abs(velocities) >= args.max_abs_velocity_mps - args.bound_tolerance))
    scale_bound = bool(scale <= args.min_hand_scale + args.bound_tolerance or scale >= args.max_hand_scale - args.bound_tolerance)
    high_contact = [row for row in rows if float(row["contact_probability"]) >= args.accept_contact_probability]
    reliable_like = [
        row
        for row in high_contact
        if row["contact_gap_median_m"] is not None
        and row["mano_minus_metric_depth_median_m"] is not None
        and abs(float(row["contact_gap_median_m"])) <= args.accept_contact_gap_m
        and abs(float(row["mano_minus_metric_depth_median_m"])) <= args.accept_depth_gap_m
        and float(row["keypoint_reprojection_median_px"]) <= args.accept_reprojection_px
    ]
    if len(reliable_like) == 0:
        return "diagnostic_no_reliable_contact_rows"
    if scale_bound or shift_bound or velocity_bound:
        return "diagnostic_solution_requires_bound_saturation"
    return "diagnostic_multisurface_graph_candidate_needs_external_reliability_qc"


def apply_solution(annotations: dict, observations: list[HandObs], params: np.ndarray, args: argparse.Namespace) -> dict:
    scale_log, shifts, velocities, _, contact = unpack(params, len(observations))
    obs_by_key = {obs.key: (i, obs) for i, obs in enumerate(observations)}
    out = copy.deepcopy(annotations)
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, str(hand.get("side")), hand_i)
            if key not in obs_by_key:
                continue
            i, obs = obs_by_key[key]
            cam_t, joints, vertices = corrected(obs, scale_log, shifts[i])
            uv = project_points(joints, obs.intrinsics)
            track_id, gap = best_surface_gap(obs, vertices)
            hand["cam_t"] = cam_t.astype(float).tolist()
            hand["joints3d_source_camera_m"] = joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = vertices.astype(float).tolist()
            hand["joints2d"] = uv.astype(float).tolist()
            hand["joints3d_world_m"] = source_to_world(joints, T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(vertices, T_world_camera).astype(float).tolist()
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_multisurface_contact_graph"
            hand["world_coordinate_status"] = "v3_multisurface_contact_graph_source_camera_mano_transformed_by_existing_camera_pose"
            hand["v3_multisurface_contact"] = {
                "hand_scale": float(math.exp(scale_log)),
                "translation_shift_m": shifts[i].astype(float).tolist(),
                "velocity_mps": velocities[i].astype(float).tolist(),
                "contact_probability": float(contact[i]),
                "selected_surface": track_id,
                "selected_contact_gap_median_m": None if len(gap) == 0 else float(np.median(gap)),
                "surfaces_available": int(len(obs.surfaces)),
            }
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    base, observations, skipped = build_observations(args)
    x0 = pack_initial(observations)
    before_rows = row_metrics(x0, observations, args)
    params, solver = solve(observations, args)
    after_rows = row_metrics(params, observations, args)
    output = apply_solution(base, observations, params, args)
    save_json(args.output_annotations, output)
    report = {
        "status": status_for(after_rows, params, observations, args),
        "annotation_ready": False,
        "diagnostic_only": True,
        "model": "global_hand_scale_temporal_3d_shift_multisurface_latent_contact",
        "base_annotations": str(args.base_annotations),
        "postprocess_manifest": str(args.postprocess_manifest),
        "metric_depth_npz": str(args.metric_depth_npz),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "observations": int(len(observations)),
        "skipped_rows": int(len(skipped)),
        "variables": int(len(params)),
        "solver": solver,
        "before_summary": summarize_rows(before_rows),
        "after_summary": summarize_rows(after_rows),
        "rows_preview_before": before_rows[:180],
        "rows_preview_after": after_rows[:180],
        "skipped_preview": skipped[:180],
        "elapsed_s": float(time.time() - started),
        "interpretation": (
            "This graph tests whether a shared hand scale, temporal 3D hand shifts, metric depth, 2D keypoints, "
            "and model-produced surface meshes can jointly explain contact. It remains diagnostic until an external "
            "contact-reliability pass validates the corrected annotations."
        ),
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview_before", "rows_preview_after", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-annotations", type=Path, required=True)
    parser.add_argument("--postprocess-manifest", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--rtmlib-json", type=Path)
    parser.add_argument("--rtmlib-wilor-qc", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-observations", type=int, default=12)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=24.0)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--min-depth-keypoints", type=int, default=8)
    parser.add_argument("--contact-distance-px", type=float, default=10.0)
    parser.add_argument("--min-near-vertices", type=int, default=40)
    parser.add_argument("--max-contact-vertices", type=int, default=160)
    parser.add_argument("--max-contract-error-m", type=float, default=0.025)
    parser.add_argument("--max-rtmlib-wilor-delta-px", type=float, default=30.0)
    parser.add_argument("--min-rtmlib-keypoint-score", type=float, default=0.2)
    parser.add_argument("--min-hand-scale", type=float, default=0.80)
    parser.add_argument("--max-hand-scale", type=float, default=1.20)
    parser.add_argument("--max-abs-translation-m", type=float, default=0.240)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=2.0)
    parser.add_argument("--max-abs-contact-logit", type=float, default=6.0)
    parser.add_argument("--sigma-wilor-reprojection-px", type=float, default=18.0)
    parser.add_argument("--sigma-rtmlib-reprojection-px", type=float, default=18.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-contact-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-penetration-m", type=float, default=0.015)
    parser.add_argument("--sigma-translation-prior-m", type=float, default=0.120)
    parser.add_argument("--sigma-velocity-prior-m", type=float, default=1.000)
    parser.add_argument("--sigma-motion-m", type=float, default=0.040)
    parser.add_argument("--sigma-acceleration-m", type=float, default=0.700)
    parser.add_argument("--sigma-hand-log-scale", type=float, default=0.100)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.030)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.205)
    parser.add_argument("--sigma-contact-logit", type=float, default=1.5)
    parser.add_argument("--sigma-contact-logit-step", type=float, default=2.0)
    parser.add_argument("--sigma-no-surface-contact", type=float, default=0.1)
    parser.add_argument("--nonselected-surface-weight", type=float, default=0.05)
    parser.add_argument("--contact-seed-scale-m", type=float, default=0.080)
    parser.add_argument("--accept-contact-probability", type=float, default=0.50)
    parser.add_argument("--accept-contact-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-depth-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-reprojection-px", type=float, default=12.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-5)
    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--clip-residual", type=float, default=50.0)
    parser.add_argument("--max-nfev", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
