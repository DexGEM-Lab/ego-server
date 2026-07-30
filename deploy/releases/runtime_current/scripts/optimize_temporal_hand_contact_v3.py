#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
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
class HandObs:
    key: tuple[int, str, int]
    frame_idx: int
    frame_order: int
    side: str
    hand_index: int
    detector_score: float
    base_cam_t: np.ndarray
    local_joints: np.ndarray
    local_vertices: np.ndarray
    source_joints: np.ndarray
    source_vertices: np.ndarray
    wilor2d: np.ndarray
    wilor_weight: np.ndarray
    rtmlib2d: np.ndarray
    rtmlib_weight: np.ndarray
    intrinsics: np.ndarray
    metric_depth: np.ndarray
    depth_weight: np.ndarray
    near_vertex_indices: np.ndarray
    object_depth_m: float
    contact_seed: float
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


def bone_scale(joints: np.ndarray) -> float:
    lengths = []
    for chain in FINGER_CHAINS:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def object_camera_depth(vertices_world: np.ndarray, T_world_camera: np.ndarray) -> float:
    homog = np.c_[vertices_world, np.ones(len(vertices_world), dtype=float)]
    camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if len(z) == 0:
        raise RuntimeError("object mesh has no positive camera depth")
    return float(np.median(z))


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def contact_vertices(vertices: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray, depth: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> np.ndarray:
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
    wilor_target: np.ndarray,
    matches: dict[tuple[int, str, int], dict],
    rtmlib_frames: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    match = matches.get((frame_idx, side, hand_i))
    if match is None:
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float), float("nan"), 0.0
    frame = rtmlib_frames.get(frame_idx)
    if frame is None:
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float), float(match["median_keypoint_delta_px"]), 0.0
    hands = frame.get("hands", [])
    list_idx = int(match["rtmlib_list_idx"])
    if list_idx < 0 or list_idx >= len(hands):
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float), float(match["median_keypoint_delta_px"]), 0.0
    rtm = np.asarray(hands[list_idx].get("keypoints", []), dtype=float)
    scores = np.asarray(hands[list_idx].get("scores", []), dtype=float)
    if rtm.shape != (21, 2) or scores.shape != (21,):
        return np.zeros((21, 2), dtype=float), np.zeros(21, dtype=float), float(match["median_keypoint_delta_px"]), 0.0
    good = np.isfinite(rtm).all(axis=1) & np.isfinite(scores) & (scores >= args.min_rtmlib_keypoint_score)
    weight = np.zeros(21, dtype=float)
    weight[good] = 1.0
    return rtm, weight, float(match["median_keypoint_delta_px"]), float(np.mean(scores[good])) if np.any(good) else 0.0


def build_observations(args: argparse.Namespace) -> tuple[dict, list[HandObs], list[dict]]:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_indices)) != len(depth_indices):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_indices)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    matches = load_rtmlib_matches(args.rtmlib_wilor_qc, args.max_rtmlib_wilor_delta_px)
    rtmlib_frames = load_rtmlib_frames(args.rtmlib_json)
    observations: list[HandObs] = []
    skipped: list[dict] = []
    frame_order = 0
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        obj = frame.get("object", {})
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            object_depth = object_camera_depth(object_vertices, np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float))
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            frame_order += 1
            continue
        scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                side = str(hand.get("side", "unknown"))
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
                    raise RuntimeError("invalid hand image observation")
                contract = max(
                    float(np.max(np.linalg.norm(local_joints + cam_t[None, :] - source_joints, axis=1))),
                    float(np.max(np.linalg.norm(local_vertices + cam_t[None, :] - source_vertices, axis=1))),
                )
                if contract > args.max_contract_error_m:
                    raise RuntimeError(f"hand_source_contract_error_{contract:.4f}m")
                projected = project_points(source_joints, intrinsics)
                reproj = np.linalg.norm(projected - raw2d, axis=1)
                rtmlib2d, rtmlib_weight, match_delta, rtm_score = target_from_rtmlib(
                    frame_idx, hand_i, side, raw2d, matches, rtmlib_frames, args
                )
                metric_depth = sample_depth(depth, raw2d, source_size)
                valid_depth = np.isfinite(metric_depth) & (metric_depth > 0.0) & (reproj <= args.good_joint_reprojection_px)
                patch = np.asarray(
                    [depth_patch_iqr_ratio(depth, xy * scale, args.patch_radius) for xy in raw2d],
                    dtype=float,
                )
                stable = np.isfinite(patch) & (patch <= args.max_depth_iqr_ratio)
                depth_weight = np.zeros(21, dtype=float)
                depth_weight[valid_depth & stable] = 1.0
                near = contact_vertices(source_vertices, intrinsics, mask, depth, source_size, args)
                contact_seed = 0.0
                if len(near) >= args.min_seed_near_vertices:
                    gap = source_vertices[near, 2] - object_depth
                    contact_seed = float(np.exp(-abs(float(np.median(gap))) / max(args.contact_seed_scale_m, 1e-6)))
                observations.append(
                    HandObs(
                        key=(frame_idx, side, hand_i),
                        frame_idx=frame_idx,
                        frame_order=frame_order,
                        side=side,
                        hand_index=hand_i,
                        detector_score=score,
                        base_cam_t=cam_t,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        source_joints=source_joints,
                        source_vertices=source_vertices,
                        wilor2d=raw2d,
                        wilor_weight=np.ones(21, dtype=float),
                        rtmlib2d=rtmlib2d,
                        rtmlib_weight=rtmlib_weight,
                        intrinsics=intrinsics,
                        metric_depth=metric_depth,
                        depth_weight=depth_weight,
                        near_vertex_indices=near,
                        object_depth_m=object_depth,
                        contact_seed=contact_seed,
                        contract_error_m=contract,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
        frame_order += 1
    if not observations:
        raise RuntimeError(f"no usable temporal hand observations; skipped={skipped[:12]}")
    return annotations, observations, skipped


def pack_initial(observations: list[HandObs]) -> np.ndarray:
    shifts = np.zeros((len(observations), 3), dtype=float)
    velocities = np.zeros((len(observations), 3), dtype=float)
    logits = np.asarray([math.log((0.05 + 0.90 * obs.contact_seed) / (0.95 - 0.90 * obs.contact_seed)) for obs in observations], dtype=float)
    return np.r_[shifts.reshape(-1), velocities.reshape(-1), logits]


def unpack(params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifts = np.asarray(params[: 3 * n], dtype=float).reshape(n, 3)
    velocities = np.asarray(params[3 * n : 6 * n], dtype=float).reshape(n, 3)
    logits = np.asarray(params[6 * n : 7 * n], dtype=float)
    contact = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return shifts, velocities, logits, contact


def corrected(obs: HandObs, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_t = obs.base_cam_t + shift
    joints = obs.local_joints + cam_t[None, :]
    vertices = obs.local_vertices + cam_t[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("nonpositive corrected hand depth")
    return cam_t, joints, vertices


def residual(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    shifts, velocities, logits, contact = unpack(params, len(observations))
    out: list[np.ndarray] = []
    for i, obs in enumerate(observations):
        try:
            _, joints, vertices = corrected(obs, shifts[i])
        except RuntimeError:
            out.append(np.full(16, args.invalid_penalty, dtype=float))
            continue
        projected = project_points(joints, obs.intrinsics)
        wilor_reproj = ((projected - obs.wilor2d) / args.sigma_wilor_reprojection_px).reshape(21, 2)
        out.append((wilor_reproj * obs.wilor_weight[:, None]).reshape(-1))
        if np.any(obs.rtmlib_weight > 0.0):
            rtm_reproj = ((projected - obs.rtmlib2d) / args.sigma_rtmlib_reprojection_px).reshape(21, 2)
            out.append((rtm_reproj * obs.rtmlib_weight[:, None]).reshape(-1))
        valid_depth = obs.depth_weight > 0.0
        if np.any(valid_depth):
            out.append((joints[valid_depth, 2] - obs.metric_depth[valid_depth]) / args.sigma_metric_depth_m)
        out.append(shifts[i] / args.sigma_translation_prior_m)
        out.append(velocities[i] / args.sigma_velocity_prior_m)
        out.append(np.asarray([(bone_scale(joints) - args.hand_bone_scale_prior_m) / args.sigma_bone_scale_m]))
        p0 = np.clip(0.05 + 0.90 * obs.contact_seed, 1e-4, 1.0 - 1e-4)
        out.append(np.asarray([(logits[i] - math.log(p0 / (1.0 - p0))) / args.sigma_contact_logit]))
        if len(obs.near_vertex_indices) >= args.min_near_vertices:
            gap = vertices[obs.near_vertex_indices, 2] - obs.object_depth_m
            if len(gap) > args.max_contact_vertices:
                gap = gap[np.linspace(0, len(gap) - 1, args.max_contact_vertices).astype(int)]
            attraction = math.sqrt(max(float(contact[i]), 1e-6)) * gap / args.sigma_contact_depth_m
            out.append(np.clip(attraction, -args.clip_residual, args.clip_residual))
            penetration = np.minimum(gap, 0.0) / args.sigma_penetration_m
            out.append(np.clip(penetration, -args.clip_residual, args.clip_residual))
        else:
            out.append(np.asarray([contact[i] / args.sigma_no_evidence_contact]))
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
        np.full(3 * n, -args.max_abs_translation_m),
        np.full(3 * n, -args.max_abs_velocity_mps),
        np.full(n, -args.max_abs_contact_logit),
    ]
    upper = np.r_[
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
        max_nfev=args.max_nfev,
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


def row_metrics(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> list[dict]:
    shifts, velocities, logits, contact = unpack(params, len(observations))
    rows = []
    for i, obs in enumerate(observations):
        _, joints, vertices = corrected(obs, shifts[i])
        uv = project_points(joints, obs.intrinsics)
        reproj = np.linalg.norm(uv - obs.wilor2d, axis=1)
        rtm_valid = obs.rtmlib_weight > 0.0
        rtm_reproj = np.linalg.norm(uv[rtm_valid] - obs.rtmlib2d[rtm_valid], axis=1)
        valid_depth = obs.depth_weight > 0.0
        depth_gap = joints[valid_depth, 2] - obs.metric_depth[valid_depth]
        contact_gap = np.asarray([], dtype=float)
        if len(obs.near_vertex_indices) > 0:
            contact_gap = vertices[obs.near_vertex_indices, 2] - obs.object_depth_m
        rows.append(
            {
                "frame_idx": int(obs.frame_idx),
                "side": obs.side,
                "hand_index": int(obs.hand_index),
                "detector_score": float(obs.detector_score),
                "translation_shift_norm_m": float(np.linalg.norm(shifts[i])),
                "velocity_norm_mps": float(np.linalg.norm(velocities[i])),
                "contact_probability": float(contact[i]),
                "contact_seed": float(obs.contact_seed),
                "contract_error_m": float(obs.contract_error_m),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "rtmlib_keypoint_reprojection_median_px": None if len(rtm_reproj) == 0 else float(np.median(rtm_reproj)),
                "rtmlib_keypoint_reprojection_p95_px": None if len(rtm_reproj) == 0 else float(np.percentile(rtm_reproj, 95.0)),
                "rtmlib_keypoints": int(np.count_nonzero(rtm_valid)),
                "metric_depth_keypoints": int(np.count_nonzero(valid_depth)),
                "mano_minus_metric_depth_median_m": None if len(depth_gap) == 0 else float(np.median(depth_gap)),
                "mano_minus_metric_depth_p95_abs_m": None if len(depth_gap) == 0 else float(np.percentile(np.abs(depth_gap), 95.0)),
                "near_mask_vertices": int(len(obs.near_vertex_indices)),
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
        "contract_error_m": summarize_key(rows, "contract_error_m"),
    }


def status_for(rows: list[dict], params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> str:
    shifts, velocities, _, contact = unpack(params, len(observations))
    shift_bound = bool(np.any(np.abs(shifts) >= args.max_abs_translation_m - args.bound_tolerance))
    velocity_bound = bool(np.any(np.abs(velocities) >= args.max_abs_velocity_mps - args.bound_tolerance))
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
    if shift_bound or velocity_bound:
        return "diagnostic_solution_requires_bound_saturation"
    return "diagnostic_temporal_graph_candidate_needs_external_reliability_qc"


def apply_solution(annotations: dict, observations: list[HandObs], params: np.ndarray, args: argparse.Namespace) -> dict:
    shifts, velocities, _, contact = unpack(params, len(observations))
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
            cam_t, joints, vertices = corrected(obs, shifts[i])
            joints2d = project_points(joints, obs.intrinsics)
            reproj = np.linalg.norm(joints2d - obs.wilor2d, axis=1)
            hand["cam_t"] = cam_t.astype(float).tolist()
            hand["joints3d_source_camera_m"] = joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = vertices.astype(float).tolist()
            hand["joints2d"] = joints2d.astype(float).tolist()
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(reproj)),
                "p95": float(np.percentile(reproj, 95.0)),
            }
            hand["joints3d_world_m"] = source_to_world(joints, T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(vertices, T_world_camera).astype(float).tolist()
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_temporal_contact_graph"
            hand["world_coordinate_status"] = "v3_temporal_contact_graph_source_camera_mano_transformed_by_existing_camera_pose"
            hand["v3_temporal_contact"] = {
                "translation_shift_m": shifts[i].astype(float).tolist(),
                "velocity_mps": velocities[i].astype(float).tolist(),
                "contact_probability": float(contact[i]),
                "contact_seed": float(obs.contact_seed),
                "contract_error_m": float(obs.contract_error_m),
            }
    return out


def run(args: argparse.Namespace) -> dict:
    args.intrinsics = np.asarray(args.intrinsics, dtype=float)
    if args.intrinsics.shape != (4,):
        raise RuntimeError("--intrinsics must have four values")
    annotations, observations, skipped = build_observations(args)
    x0 = pack_initial(observations)
    initial_rows = row_metrics(x0, observations, args)
    params, solver = solve(observations, args)
    final_rows = row_metrics(params, observations, args)
    output = apply_solution(annotations, observations, params, args)
    save_json(args.output_annotations, output)
    report = {
        "status": status_for(final_rows, params, observations, args),
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "observations": int(len(observations)),
        "variables": int(len(params)),
        "solver": solver,
        "initial": summarize_rows(initial_rows),
        "final": summarize_rows(final_rows),
        "rows_preview_initial": initial_rows[:180],
        "rows_preview_final": final_rows[:180],
        "skipped_preview": skipped[:180],
        "interpretation": (
            "This diagnostic optimizes a temporal hand translation and contact-probability state over measured fused hand geometry. "
            "It is accepted only if the external contact-reliability diagnostic later finds reliable rows; optimizer residual reduction alone is insufficient."
        ),
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview_initial", "rows_preview_final", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--rtmlib-json", type=Path)
    parser.add_argument("--rtmlib-wilor-qc", type=Path)
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[2304.0, 2304.0, 960.0, 540.0])
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=24.0)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--min-seed-near-vertices", type=int, default=20)
    parser.add_argument("--max-contact-vertices", type=int, default=180)
    parser.add_argument("--max-contract-error-m", type=float, default=0.025)
    parser.add_argument("--max-rtmlib-wilor-delta-px", type=float, default=30.0)
    parser.add_argument("--min-rtmlib-keypoint-score", type=float, default=0.2)
    parser.add_argument("--sigma-wilor-reprojection-px", type=float, default=20.0)
    parser.add_argument("--sigma-rtmlib-reprojection-px", type=float, default=20.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-contact-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-penetration-m", type=float, default=0.015)
    parser.add_argument("--sigma-translation-prior-m", type=float, default=0.100)
    parser.add_argument("--sigma-velocity-prior-m", type=float, default=1.000)
    parser.add_argument("--sigma-motion-m", type=float, default=0.030)
    parser.add_argument("--sigma-acceleration-m", type=float, default=0.600)
    parser.add_argument("--sigma_bone_scale_m", "--sigma-bone-scale-m", dest="sigma_bone_scale_m", type=float, default=0.030)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.170)
    parser.add_argument("--sigma-contact-logit", type=float, default=1.5)
    parser.add_argument("--sigma-contact-logit-step", type=float, default=2.0)
    parser.add_argument("--sigma-no-evidence-contact", type=float, default=0.1)
    parser.add_argument("--contact-seed-scale-m", type=float, default=0.060)
    parser.add_argument("--max-abs-translation-m", type=float, default=0.180)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=1.5)
    parser.add_argument("--max-abs-contact-logit", type=float, default=6.0)
    parser.add_argument("--accept-contact-probability", type=float, default=0.50)
    parser.add_argument("--accept-contact-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-depth-gap-m", type=float, default=0.030)
    parser.add_argument("--accept-reprojection-px", type=float, default=12.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-5)
    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--clip-residual", type=float, default=50.0)
    parser.add_argument("--max-nfev", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
