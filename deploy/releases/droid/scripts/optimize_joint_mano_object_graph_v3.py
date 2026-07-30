#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from align_mesh_prior_v3 import load_observed_frame, sample_mesh_surface
from diagnose_hand_reprojection_depth_v3 import bbox_corners, project_points
from optimize_object_factor_graph_v3 import (
    MEASURED_STATUSES,
    camera_axis_world,
    load_initial_sim3,
    localize_path,
    mask_distance_map,
    resize_bool_mask,
    sample_rows,
    save_mesh_archive,
    silhouette_residual,
    summarize_array,
    transform_points,
)


@dataclass(frozen=True)
class HandFactor:
    frame_i: int
    frame_idx: int
    side: str
    detector_score: float
    weight: float
    bbox_xyxy: np.ndarray
    intrinsics: np.ndarray
    bbox_points_camera: np.ndarray
    contact_points_camera: np.ndarray
    center_ray: np.ndarray


@dataclass(frozen=True)
class FrameFactor:
    frame_idx: int
    T_world_camera: np.ndarray
    observed_points: np.ndarray
    observed_tree: cKDTree
    mask: np.ndarray
    mask_distance: np.ndarray
    mask_size: tuple[int, int]
    source_size: tuple[int, int]
    camera_axis_world: np.ndarray


@dataclass(frozen=True)
class BuildResult:
    frames: list[FrameFactor]
    hands: list[HandFactor]
    skipped: list[dict]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def active_frame_records(annotations: dict, start: int, end: int) -> list[dict]:
    out = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        obj = frame.get("object", {})
        if start <= idx <= end and obj.get("status") in MEASURED_STATUSES and obj.get("mask_path"):
            out.append(frame)
    if not out:
        raise RuntimeError("no active measured object frames selected")
    return out


def camera_to_world(points_camera_m: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    return np.asarray(points_camera_m, dtype=float) @ R.T + t[None, :]


def bbox_residual(points_camera_m: np.ndarray, intrinsics: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    uv = project_points(points_camera_m, intrinsics)
    proj_min = uv.min(axis=0)
    proj_max = uv.max(axis=0)
    target = bbox_corners(bbox_xyxy.astype(float).tolist())
    return np.r_[proj_min - target[0], proj_max - target[1]].astype(float)


def source_points(hand: dict, key: str) -> np.ndarray:
    if key not in hand:
        raise RuntimeError(f"missing hand field {key}")
    points = np.asarray(hand[key], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"{key} must have shape Nx3")
    if len(points) == 0 or not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 0.0):
        raise RuntimeError(f"{key} has invalid camera points")
    return points


def source_intrinsics(hand: dict, expected: np.ndarray, tolerance: float) -> np.ndarray:
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if intr.shape != (4,) or not np.all(np.isfinite(intr)):
        raise RuntimeError("invalid source_intrinsics")
    if float(np.max(np.abs(intr - expected))) > tolerance:
        raise RuntimeError("hand source_intrinsics disagree with DROID source intrinsics")
    return intr


def near_mask(vertices_camera_m: np.ndarray, intrinsics: np.ndarray, frame: FrameFactor, distance_px: float) -> np.ndarray:
    uv = project_points(vertices_camera_m, intrinsics)
    scale = np.asarray([frame.mask.shape[1], frame.mask.shape[0]], dtype=float) / np.asarray(frame.source_size, dtype=float)
    xy = uv * scale[None, :]
    valid = np.isfinite(xy).all(axis=1) & (vertices_camera_m[:, 2] > 0.0)
    if not np.any(valid):
        return np.zeros(len(vertices_camera_m), dtype=bool)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, frame.mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, frame.mask.shape[0] - 1)
    return valid & (frame.mask_distance[y, x] <= distance_px * float(scale.mean()))


def hand_weight(score: float, current_bbox_l2_px: float, args: argparse.Namespace) -> float:
    if not np.isfinite(score):
        return 0.0
    score_part = min(1.0, max(0.0, float(score) / args.hand_score_full))
    reproj_part = math.exp(-0.5 * (float(current_bbox_l2_px) / args.contact_reprojection_sigma_px) ** 2)
    return float(score_part * reproj_part)


def build_data(args: argparse.Namespace, selected_frames: list[dict], intrinsics: np.ndarray) -> BuildResult:
    frames: list[FrameFactor] = []
    hands: list[HandFactor] = []
    skipped: list[dict] = []
    for frame in selected_frames:
        idx = int(frame["frame_idx"])
        obj = frame["object"]
        try:
            observed, _ = load_observed_frame(args.observed_mesh_npz, idx)
            observed = sample_rows(observed, args.max_observed_points, args.seed + idx)
            mask_size = tuple(int(x) for x in obj["mask_image_size"])
            source_size = tuple(int(x) for x in obj["source_image_size"])
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, mask_size)
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            if T_world_camera.shape != (4, 4):
                raise RuntimeError("invalid T_world_camera_metric")
            frame_record = FrameFactor(
                frame_idx=idx,
                T_world_camera=T_world_camera,
                observed_points=observed,
                observed_tree=cKDTree(observed),
                mask=mask,
                mask_distance=mask_distance_map(mask),
                mask_size=mask_size,
                source_size=source_size,
                camera_axis_world=camera_axis_world(T_world_camera),
            )
            frame_i = len(frames)
            frames.append(frame_record)
        except Exception as exc:
            skipped.append({"frame_idx": idx, "reason": str(exc)})
            continue

        for hand_i, hand in enumerate(frame.get("hands", [])):
            try:
                vertices = source_points(hand, "vertices_source_camera_m")
                joints = source_points(hand, "joints3d_source_camera_m")
                intr = source_intrinsics(hand, intrinsics, args.intrinsics_tolerance)
                bbox = np.asarray(hand.get("bbox_xyxy", []), dtype=float)
                if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                    raise RuntimeError("invalid bbox_xyxy")
                near = near_mask(vertices, intr, frame_record, args.contact_distance_px)
                near_count = int(np.count_nonzero(near))
                if near_count < args.min_near_vertices:
                    skipped.append(
                        {
                            "frame_idx": idx,
                            "hand_idx": hand_i,
                            "side": hand.get("side"),
                            "reason": "too_few_near_mask_vertices",
                            "near_vertices": near_count,
                        }
                    )
                    continue
                bbox_points = sample_rows(np.vstack([joints, vertices]), args.max_hand_reprojection_points, args.seed + idx + hand_i)
                contact_points = sample_rows(vertices[near], args.max_contact_points_per_hand, args.seed + idx + 31 + hand_i)
                center = np.median(np.vstack([joints, vertices]), axis=0)
                if center[2] <= 0.0:
                    raise RuntimeError("hand center has non-positive depth")
                current_l2 = float(np.linalg.norm(bbox_residual(bbox_points, intr, bbox)))
                score = float(hand.get("detector_score", np.nan))
                weight = hand_weight(score, current_l2, args)
                if weight < args.min_hand_weight:
                    skipped.append(
                        {
                            "frame_idx": idx,
                            "hand_idx": hand_i,
                            "side": hand.get("side"),
                            "reason": "low_hand_weight",
                            "weight": weight,
                            "detector_score": score,
                            "bbox_l2_px": current_l2,
                        }
                    )
                    continue
                hands.append(
                    HandFactor(
                        frame_i=frame_i,
                        frame_idx=idx,
                        side=str(hand.get("side", "unknown")),
                        detector_score=score,
                        weight=weight,
                        bbox_xyxy=bbox,
                        intrinsics=intr,
                        bbox_points_camera=bbox_points,
                        contact_points_camera=contact_points,
                        center_ray=center / center[2],
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
    if len(frames) < 2:
        raise RuntimeError(f"too few usable factor frames; skipped={skipped[:8]}")
    if len(hands) == 0:
        raise RuntimeError(f"no usable hand contact factors; skipped={skipped[:8]}")
    return BuildResult(frames=frames, hands=hands, skipped=skipped)


def unpack(params: np.ndarray, frame_count: int, hand_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    object_width = 7
    object_params = params[: frame_count * object_width].reshape(frame_count, object_width)
    rotvecs = object_params[:, :3]
    translations = object_params[:, 3:6]
    depth_offsets = object_params[:, 6]
    hand_log_scale = float(params[frame_count * object_width])
    hand_ray_shift = params[frame_count * object_width + 1 : frame_count * object_width + 1 + hand_count]
    return rotvecs, translations, depth_offsets, hand_log_scale, hand_ray_shift


def corrected_hand_camera(hand: HandFactor, hand_log_scale: float, hand_shift_m: float, contact: bool) -> np.ndarray:
    points = hand.contact_points_camera if contact else hand.bbox_points_camera
    scale = math.exp(hand_log_scale)
    corrected = scale * points + float(hand_shift_m) * hand.center_ray[None, :]
    if np.any(corrected[:, 2] <= 0.0):
        raise RuntimeError("hand correction produced non-positive depth")
    return corrected


def residual_vector(
    params: np.ndarray,
    frames: list[FrameFactor],
    hands: list[HandFactor],
    base_surface: np.ndarray,
    base_silhouette: np.ndarray,
    pivot: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rotvecs, translations, depth_offsets, hand_log_scale, hand_ray_shift = unpack(params, len(frames), len(hands))
    residuals: list[np.ndarray] = []
    residuals.append(np.asarray([hand_log_scale / args.sigma_hand_log_scale], dtype=float))
    object_trees: list[cKDTree] = []
    surfaces: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        translation = translations[i] + depth_offsets[i] * frame.camera_axis_world
        surface = transform_points(base_surface, pivot, rotvecs[i], translation)
        surfaces.append(surface)
        tree = cKDTree(surface)
        object_trees.append(tree)
        d_obs, _ = tree.query(frame.observed_points, k=1)
        d_prior, _ = frame.observed_tree.query(surface, k=1)
        residuals.append(np.clip(d_obs, 0.0, args.max_surface_residual_m) / args.sigma_observed_m)
        residuals.append(np.clip(d_prior, 0.0, args.max_surface_residual_m) / args.sigma_prior_surface_m)
        silhouette_points = transform_points(base_silhouette, pivot, rotvecs[i], translation)
        residuals.append(silhouette_residual(silhouette_points, frame, intrinsics, args.sigma_silhouette_px, args.max_silhouette_px))
        residuals.append(np.asarray([depth_offsets[i] / args.sigma_object_depth_offset_m], dtype=float))
    for h, hand in enumerate(hands):
        frame = frames[hand.frame_i]
        contact_camera = corrected_hand_camera(hand, hand_log_scale, hand_ray_shift[h], contact=True)
        contact_world = camera_to_world(contact_camera, frame.T_world_camera)
        d_contact, _ = object_trees[hand.frame_i].query(contact_world, k=1)
        contact_sigma = args.sigma_contact_m / max(args.min_hand_weight, hand.weight)
        residuals.append(np.clip(d_contact, 0.0, args.max_contact_residual_m) / contact_sigma)
        reproj_camera = corrected_hand_camera(hand, hand_log_scale, hand_ray_shift[h], contact=False)
        reproj = bbox_residual(reproj_camera, hand.intrinsics, hand.bbox_xyxy)
        reproj_sigma = args.sigma_hand_reprojection_px / max(args.min_hand_weight, hand.weight)
        residuals.append(np.clip(reproj, -args.max_reprojection_residual_px, args.max_reprojection_residual_px) / reproj_sigma)
        residuals.append(np.asarray([hand_ray_shift[h] / args.sigma_hand_ray_shift_m], dtype=float))
    for i in range(1, len(frames)):
        residuals.append((translations[i] - translations[i - 1]) / args.sigma_translation_step_m)
        residuals.append((rotvecs[i] - rotvecs[i - 1]) / args.sigma_rotation_step_rad)
        residuals.append(np.asarray([(depth_offsets[i] - depth_offsets[i - 1]) / args.sigma_depth_step_m], dtype=float))
    for i in range(1, len(frames) - 1):
        residuals.append((translations[i + 1] - 2.0 * translations[i] + translations[i - 1]) / args.sigma_translation_accel_m)
        residuals.append((rotvecs[i + 1] - 2.0 * rotvecs[i] + rotvecs[i - 1]) / args.sigma_rotation_accel_rad)
        residuals.append(np.asarray([(depth_offsets[i + 1] - 2.0 * depth_offsets[i] + depth_offsets[i - 1]) / args.sigma_depth_accel_m], dtype=float))
    by_side: dict[str, list[tuple[int, int]]] = {}
    for h, hand in enumerate(hands):
        by_side.setdefault(hand.side, []).append((hand.frame_idx, h))
    for seq in by_side.values():
        seq.sort()
        for (_, h0), (_, h1) in zip(seq, seq[1:]):
            residuals.append(np.asarray([(hand_ray_shift[h1] - hand_ray_shift[h0]) / args.sigma_hand_ray_shift_step_m], dtype=float))
        for (_, h0), (_, h1), (_, h2) in zip(seq, seq[1:], seq[2:]):
            residuals.append(np.asarray([(hand_ray_shift[h2] - 2.0 * hand_ray_shift[h1] + hand_ray_shift[h0]) / args.sigma_hand_ray_shift_accel_m], dtype=float))
    anchor = int(np.argmin([abs(frame.frame_idx - args.anchor_frame) for frame in frames]))
    residuals.append(translations[anchor] / args.sigma_anchor_translation_m)
    residuals.append(rotvecs[anchor] / args.sigma_anchor_rotation_rad)
    residuals.append(np.asarray([depth_offsets[anchor] / args.sigma_anchor_depth_offset_m], dtype=float))
    return np.concatenate([r.reshape(-1) for r in residuals])


def frame_metrics(
    params: np.ndarray,
    frames: list[FrameFactor],
    hands: list[HandFactor],
    base_surface: np.ndarray,
    base_silhouette: np.ndarray,
    pivot: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict]:
    rotvecs, translations, depth_offsets, hand_log_scale, hand_ray_shift = unpack(params, len(frames), len(hands))
    metrics: dict[str, dict] = {}
    object_trees: list[cKDTree] = []
    surfaces: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        translation = translations[i] + depth_offsets[i] * frame.camera_axis_world
        surface = transform_points(base_surface, pivot, rotvecs[i], translation)
        surfaces.append(surface)
        tree = cKDTree(surface)
        object_trees.append(tree)
        d_obs, _ = tree.query(frame.observed_points, k=1)
        d_prior, _ = frame.observed_tree.query(surface, k=1)
        silhouette_points = transform_points(base_silhouette, pivot, rotvecs[i], translation)
        sil = silhouette_residual(silhouette_points, frame, intrinsics, 1.0, args.max_silhouette_px)
        metrics[str(frame.frame_idx)] = {
            "observed_to_prior_median_m": float(np.median(d_obs)),
            "observed_to_prior_p95_m": float(np.percentile(d_obs, 95.0)),
            "prior_to_observed_median_m": float(np.median(d_prior)),
            "prior_to_observed_p95_m": float(np.percentile(d_prior, 95.0)),
            "silhouette_outside_median_px": float(np.median(sil)),
            "silhouette_outside_p95_px": float(np.percentile(sil, 95.0)),
            "translation_delta_m": translations[i].astype(float).tolist(),
            "rotation_delta_rad": rotvecs[i].astype(float).tolist(),
            "depth_axis_offset_m": float(depth_offsets[i]),
            "hand_count": 0,
            "contact_median_m": None,
            "contact_p95_m": None,
            "hand_reprojection_l2_px_median": None,
            "hand_ray_shift_m_median": None,
        }
    hand_rows: dict[int, list[dict]] = {i: [] for i in range(len(frames))}
    for h, hand in enumerate(hands):
        frame = frames[hand.frame_i]
        contact_camera = corrected_hand_camera(hand, hand_log_scale, hand_ray_shift[h], contact=True)
        contact_world = camera_to_world(contact_camera, frame.T_world_camera)
        d_contact, _ = object_trees[hand.frame_i].query(contact_world, k=1)
        reproj_camera = corrected_hand_camera(hand, hand_log_scale, hand_ray_shift[h], contact=False)
        reproj = bbox_residual(reproj_camera, hand.intrinsics, hand.bbox_xyxy)
        hand_rows[hand.frame_i].append(
            {
                "side": hand.side,
                "weight": float(hand.weight),
                "contact_median_m": float(np.median(d_contact)),
                "contact_p95_m": float(np.percentile(d_contact, 95.0)),
                "contact_min_m": float(np.min(d_contact)),
                "reprojection_l2_px": float(np.linalg.norm(reproj)),
                "reprojection_max_abs_px": float(np.max(np.abs(reproj))),
                "ray_shift_m": float(hand_ray_shift[h]),
            }
        )
    for frame_i, rows in hand_rows.items():
        if not rows:
            continue
        row = metrics[str(frames[frame_i].frame_idx)]
        row["hand_count"] = int(len(rows))
        row["contact_median_m"] = float(np.median([r["contact_median_m"] for r in rows]))
        row["contact_p95_m"] = float(np.percentile([r["contact_p95_m"] for r in rows], 95.0))
        row["hand_reprojection_l2_px_median"] = float(np.median([r["reprojection_l2_px"] for r in rows]))
        row["hand_reprojection_l2_px_p95"] = float(np.percentile([r["reprojection_l2_px"] for r in rows], 95.0))
        row["hand_ray_shift_m_median"] = float(np.median([r["ray_shift_m"] for r in rows]))
        row["hand_rows"] = rows
    return metrics


def summarize_metrics(metrics: dict[str, dict]) -> dict:
    summary = {}
    keys = (
        "observed_to_prior_median_m",
        "prior_to_observed_median_m",
        "silhouette_outside_median_px",
        "contact_median_m",
        "contact_p95_m",
        "hand_reprojection_l2_px_median",
        "hand_ray_shift_m_median",
        "depth_axis_offset_m",
    )
    for key in keys:
        vals = [row[key] for row in metrics.values() if row.get(key) is not None]
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            summary[f"{key}_median"] = float(np.median(arr))
            summary[f"{key}_p95"] = float(np.percentile(arr, 95.0))
            summary[f"{key}_max"] = float(np.max(arr))
        else:
            summary[f"{key}_median"] = None
            summary[f"{key}_p95"] = None
            summary[f"{key}_max"] = None
    return summary


def residual_blocks(
    params: np.ndarray,
    frames: list[FrameFactor],
    hands: list[HandFactor],
    base_surface: np.ndarray,
    base_silhouette: np.ndarray,
    pivot: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    metrics = summarize_metrics(frame_metrics(params, frames, hands, base_surface, base_silhouette, pivot, intrinsics, args))
    _, _, depth_offsets, hand_log_scale, hand_ray_shift = unpack(params, len(frames), len(hands))
    metrics["hand_scale"] = float(math.exp(hand_log_scale))
    metrics["hand_ray_shift_abs_m"] = summarize_array(np.abs(hand_ray_shift))
    metrics["object_depth_offset_abs_m"] = summarize_array(np.abs(depth_offsets))
    return metrics


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    selected = active_frame_records(annotations, args.frame_start, args.frame_end)
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    if intrinsics.shape != (4,):
        raise RuntimeError("DROID intrinsics_source must have shape 4")
    build = build_data(args, selected, intrinsics)
    mesh = trimesh.load(args.mesh_prior, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh prior: {args.mesh_prior}")
    initial = load_initial_sim3(args.initial_alignment_qc)
    local_surface = sample_mesh_surface(mesh, args.max_prior_surface_points, args.seed)
    local_silhouette = sample_mesh_surface(mesh, args.max_silhouette_points, args.seed + 101)
    base_vertices = initial.apply(np.asarray(mesh.vertices, dtype=float))
    base_surface = initial.apply(local_surface)
    base_silhouette = initial.apply(local_silhouette)
    pivot = base_surface.mean(axis=0)
    frame_count = len(build.frames)
    hand_count = len(build.hands)
    x0 = np.zeros(frame_count * 7 + 1 + hand_count, dtype=float)
    lower = np.full_like(x0, -np.inf, dtype=float)
    upper = np.full_like(x0, np.inf, dtype=float)
    lower[frame_count * 7] = math.log(args.min_hand_scale)
    upper[frame_count * 7] = math.log(args.max_hand_scale)
    lower[frame_count * 7 + 1 :] = -args.max_abs_hand_ray_shift_m
    upper[frame_count * 7 + 1 :] = args.max_abs_hand_ray_shift_m
    before_vec = residual_vector(x0, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args)
    result = least_squares(
        lambda x: residual_vector(x, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args),
        x0,
        bounds=(lower, upper),
        max_nfev=args.max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    after_vec = residual_vector(result.x, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args)
    before_metrics = frame_metrics(x0, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args)
    after_metrics = frame_metrics(result.x, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args)
    before_summary = summarize_metrics(before_metrics)
    after_summary = summarize_metrics(after_metrics)
    rotvecs, translations, depth_offsets, hand_log_scale, hand_ray_shift = unpack(result.x, frame_count, hand_count)
    hand_scale = float(math.exp(hand_log_scale))
    vertices_per_frame = [
        transform_points(base_vertices, pivot, rotvecs[i], translations[i] + depth_offsets[i] * build.frames[i].camera_axis_world).astype(np.float32)
        for i in range(frame_count)
    ]
    mesh_archive = args.output_dir / "joint_graph_object_meshes.npz"
    save_mesh_archive(mesh_archive, build.frames, vertices_per_frame, np.asarray(mesh.faces, dtype=np.int32))
    contact_before = before_summary.get("contact_median_m_median")
    contact_after = after_summary.get("contact_median_m_median")
    surface_before = before_summary.get("observed_to_prior_median_m_median")
    surface_after = after_summary.get("observed_to_prior_median_m_median")
    contact_improved = contact_before is not None and contact_after is not None and float(contact_after) < float(contact_before)
    surface_improved = surface_before is not None and surface_after is not None and float(surface_after) < float(surface_before)
    hand_scale_at_bound = (
        abs(hand_scale - args.min_hand_scale) <= args.bound_tolerance
        or abs(hand_scale - args.max_hand_scale) <= args.bound_tolerance
    )
    hand_shift_at_bound = bool(np.max(np.abs(hand_ray_shift)) >= args.max_abs_hand_ray_shift_m - args.bound_tolerance)
    contact_p95_after = after_summary.get("contact_p95_m_median")
    contact_median_after = after_summary.get("contact_median_m_median")
    reproj_after = after_summary.get("hand_reprojection_l2_px_median")
    status = "diagnostic_joint_contact_not_solved"
    if surface_improved and contact_improved:
        status = "diagnostic_joint_surface_and_contact_improved"
    if contact_median_after is not None and float(contact_median_after) > args.contact_solved_median_m:
        status = "diagnostic_joint_surface_improved_contact_remains_large"
    if contact_p95_after is not None and float(contact_p95_after) <= args.contact_solved_p95_m:
        status = "diagnostic_joint_contact_p95_solved"
    if hand_scale_at_bound or hand_shift_at_bound:
        status = "diagnostic_joint_contact_requires_hand_bound_saturation"
    if reproj_after is not None and float(reproj_after) > args.max_acceptable_reprojection_l2_px:
        status = "diagnostic_joint_reprojection_conflict_remains"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "mesh_prior": str(args.mesh_prior),
        "initial_alignment_qc": str(args.initial_alignment_qc),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "used_frames": [frame.frame_idx for frame in build.frames],
        "candidate_frames": [int(frame["frame_idx"]) for frame in selected],
        "skipped": build.skipped,
        "skipped_count": int(len(build.skipped)),
        "hand_factors": int(hand_count),
        "frame_count": int(frame_count),
        "variables": int(result.x.size),
        "model": "object_pose_depth_offsets_plus_global_hand_scale_plus_per_hand_center_ray_shifts",
        "max_nfev": int(args.max_nfev),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before_vec * before_vec))),
        "residual_rms_after": float(np.sqrt(np.mean(after_vec * after_vec))),
        "before_summary": before_summary,
        "after_summary": after_summary,
        "before_blocks": residual_blocks(x0, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args),
        "after_blocks": residual_blocks(result.x, build.frames, build.hands, base_surface, base_silhouette, pivot, intrinsics, args),
        "contact_improved": bool(contact_improved),
        "surface_improved": bool(surface_improved),
        "contact_solved_median_m": float(args.contact_solved_median_m),
        "contact_solved_p95_m": float(args.contact_solved_p95_m),
        "hand_scale": hand_scale,
        "hand_scale_at_bound": bool(hand_scale_at_bound),
        "hand_ray_shift_m": summarize_array(hand_ray_shift),
        "hand_ray_shift_abs_m": summarize_array(np.abs(hand_ray_shift)),
        "hand_shift_at_bound": bool(hand_shift_at_bound),
        "hand_factor_weights": summarize_array(np.asarray([hand.weight for hand in build.hands], dtype=float)),
        "bounds": {
            "hand_scale": [float(args.min_hand_scale), float(args.max_hand_scale)],
            "max_abs_hand_ray_shift_m": float(args.max_abs_hand_ray_shift_m),
            "bound_tolerance": float(args.bound_tolerance),
        },
        "priors": {
            "sigma_hand_log_scale": float(args.sigma_hand_log_scale),
            "sigma_hand_ray_shift_m": float(args.sigma_hand_ray_shift_m),
            "sigma_hand_ray_shift_step_m": float(args.sigma_hand_ray_shift_step_m),
            "sigma_hand_ray_shift_accel_m": float(args.sigma_hand_ray_shift_accel_m),
            "sigma_hand_reprojection_px": float(args.sigma_hand_reprojection_px),
            "sigma_contact_m": float(args.sigma_contact_m),
            "sigma_observed_m": float(args.sigma_observed_m),
            "sigma_prior_surface_m": float(args.sigma_prior_surface_m),
        },
        "mesh_archive": str(mesh_archive),
        "frame_metrics_before": before_metrics,
        "frame_metrics_after": after_metrics,
        "signed_penetration_supported": False,
        "penetration_note": "This diagnostic still uses unsigned proximity because the TripoSR prior is not guaranteed watertight.",
        "elapsed_s": time.time() - started,
    }
    qc_path = args.output_dir / "qc_joint_mano_object_graph_v3.json"
    qc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after", "skipped"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--initial-alignment-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, default=858)
    parser.add_argument("--max-observed-points", type=int, default=260)
    parser.add_argument("--max-prior-surface-points", type=int, default=360)
    parser.add_argument("--max-silhouette-points", type=int, default=260)
    parser.add_argument("--max-contact-points-per-hand", type=int, default=80)
    parser.add_argument("--max-hand-reprojection-points", type=int, default=220)
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--hand-score-full", type=float, default=0.80)
    parser.add_argument("--contact-reprojection-sigma-px", type=float, default=120.0)
    parser.add_argument("--min-hand-weight", type=float, default=0.15)
    parser.add_argument("--sigma-observed-m", type=float, default=0.040)
    parser.add_argument("--sigma-prior-surface-m", type=float, default=0.055)
    parser.add_argument("--sigma-silhouette-px", type=float, default=5.0)
    parser.add_argument("--sigma-contact-m", type=float, default=0.025)
    parser.add_argument("--sigma-hand-reprojection-px", type=float, default=70.0)
    parser.add_argument("--sigma-hand-log-scale", type=float, default=0.050)
    parser.add_argument("--sigma-hand-ray-shift-m", type=float, default=0.090)
    parser.add_argument("--sigma-hand-ray-shift-step-m", type=float, default=0.035)
    parser.add_argument("--sigma-hand-ray-shift-accel-m", type=float, default=0.025)
    parser.add_argument("--sigma-object-depth-offset-m", type=float, default=0.12)
    parser.add_argument("--sigma-translation-step-m", type=float, default=0.045)
    parser.add_argument("--sigma-rotation-step-rad", type=float, default=0.28)
    parser.add_argument("--sigma-depth-step-m", type=float, default=0.10)
    parser.add_argument("--sigma-translation-accel-m", type=float, default=0.025)
    parser.add_argument("--sigma-rotation-accel-rad", type=float, default=0.18)
    parser.add_argument("--sigma-depth-accel-m", type=float, default=0.06)
    parser.add_argument("--sigma-anchor-translation-m", type=float, default=0.035)
    parser.add_argument("--sigma-anchor-rotation-rad", type=float, default=0.18)
    parser.add_argument("--sigma-anchor-depth-offset-m", type=float, default=0.10)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.20)
    parser.add_argument("--max-contact-residual-m", type=float, default=0.20)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-reprojection-residual-px", type=float, default=400.0)
    parser.add_argument("--min-hand-scale", type=float, default=0.75)
    parser.add_argument("--max-hand-scale", type=float, default=1.15)
    parser.add_argument("--max-abs-hand-ray-shift-m", type=float, default=0.25)
    parser.add_argument("--contact-solved-p95-m", type=float, default=0.010)
    parser.add_argument("--contact-solved-median-m", type=float, default=0.015)
    parser.add_argument("--max-acceptable-reprojection-l2-px", type=float, default=160.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--intrinsics-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
