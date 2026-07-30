#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m
from diagnose_intrinsics_focal_sweep_v3 import solve_source_camera_translation, source_local_vertices
from diagnose_vggt_focal_sweep_v3 import source_focal_to_vggt_intrinsics, summarize
from diagnose_vggt_fragment_contact_v3 import (
    gap_stats,
    load_json,
    point_extent,
    surface_stats,
    track_fragments,
)
from diagnose_vggt_mano_contact_v3 import points_to_vggt_frame
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_surface_fragment_contact_v3 import load_tracks


@dataclass(frozen=True)
class HandObs:
    key: tuple[int, str, int]
    frame_idx: int
    side: str
    hand_index: int
    detector_score: float
    local_joints: np.ndarray
    local_vertices: np.ndarray
    target2d_vggt: np.ndarray


@dataclass(frozen=True)
class PatchObs:
    index: int
    hand_key: tuple[int, str, int]
    frame_idx: int
    track_id: str
    fragment: dict
    vertex_ids: np.ndarray
    local_center: np.ndarray
    seed_probability: float
    diagnostic_row: dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: float) -> float:
    q = float(np.clip(p, 1e-5, 1.0 - 1e-5))
    return math.log(q / (1.0 - q))


def patch_seed(row: dict, args: argparse.Namespace) -> float:
    distance = row.get("best_patch_surface_distance_p95_m")
    gap = row.get("best_patch_gap_median_m")
    spread = row.get("best_patch_local_spread_m")
    if distance is None or gap is None or spread is None:
        return float(args.min_contact_seed)
    distance_score = math.exp(-float(distance) / max(float(args.seed_distance_scale_m), 1e-6))
    gap_score = math.exp(-abs(float(gap)) / max(float(args.seed_gap_scale_m), 1e-6))
    spread_score = math.exp(-max(0.0, float(spread) - float(args.accept_patch_spread_m)) / max(float(args.seed_spread_scale_m), 1e-6))
    return float(np.clip(float(args.min_contact_seed) + float(args.max_contact_seed_boost) * distance_score * gap_score * spread_score, 1e-5, 1.0 - 1e-5))


def build_focal_problem(args: argparse.Namespace, focal_key: str) -> tuple[list[HandObs], list[PatchObs], dict]:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    archive = np.load(args.vggt_archive)
    archive_frames = archive["frame_idx"].astype(int)
    tracks = load_tracks(args.sam2_root)
    fragments = track_fragments(
        tracks,
        archive_frames,
        int(args.target_size),
        args.remote_output_root,
        args.local_output_root,
        args,
    )
    fragment_by_key = {(int(fragment["frame_idx"]), str(fragment["track_id"])): fragment for fragment in fragments}
    diagnostic = load_json(args.patch_diagnostic)
    rows = diagnostic.get("rows_by_focal", {}).get(focal_key)
    if rows is None:
        raise RuntimeError(f"patch diagnostic has no rows_by_focal entry for {focal_key}")

    hand_obs: dict[tuple[int, str, int], HandObs] = {}
    patch_obs: list[PatchObs] = []
    skipped = []
    for row in rows:
        if int(row.get("best_patch_vertices", 0)) < int(args.min_patch_vertices):
            continue
        if float(row.get("detector_score", 0.0)) < float(args.min_detector_score):
            continue
        if float(row.get("median_joint_reprojection_px_vggt", math.inf)) > float(args.max_seed_reprojection_px):
            continue
        frame_idx = int(row["frame_idx"])
        hand_index = int(row["hand_index"])
        side = str(row.get("side", "unknown"))
        hand_key = (frame_idx, side, hand_index)
        frame = frames.get(frame_idx)
        fragment = fragment_by_key.get((frame_idx, str(row["track_id"])))
        if frame is None or fragment is None:
            skipped.append({"frame_idx": frame_idx, "track_id": row.get("track_id"), "reason": "missing_frame_or_fragment"})
            continue
        hands = frame.get("hands", [])
        if hand_index < 0 or hand_index >= len(hands):
            skipped.append({"frame_idx": frame_idx, "hand_index": hand_index, "reason": "missing_hand"})
            continue
        hand = hands[hand_index]
        try:
            local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
            local_vertices = source_local_vertices(hand)
            raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
            target2d = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], int(args.target_size))
            vertex_ids = np.asarray(row["best_patch_vertex_ids"], dtype=int)
            if local_joints.shape != (21, 3) or raw2d.shape != (21, 2) or local_vertices.ndim != 2:
                raise RuntimeError("invalid_hand_geometry")
            if vertex_ids.size < int(args.min_patch_vertices) or np.any(vertex_ids < 0) or np.any(vertex_ids >= len(local_vertices)):
                raise RuntimeError("invalid_patch_vertices")
            hand_obs.setdefault(
                hand_key,
                HandObs(
                    key=hand_key,
                    frame_idx=frame_idx,
                    side=side,
                    hand_index=hand_index,
                    detector_score=float(hand.get("detector_score", row.get("detector_score", 0.0))),
                    local_joints=local_joints,
                    local_vertices=local_vertices,
                    target2d_vggt=target2d,
                ),
            )
            patch_obs.append(
                PatchObs(
                    index=len(patch_obs),
                    hand_key=hand_key,
                    frame_idx=frame_idx,
                    track_id=str(row["track_id"]),
                    fragment=fragment,
                    vertex_ids=vertex_ids,
                    local_center=np.mean(local_vertices[vertex_ids], axis=0),
                    seed_probability=patch_seed(row, args),
                    diagnostic_row=row,
                )
            )
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "hand_index": hand_index, "track_id": row.get("track_id"), "reason": str(exc)})

    hands = list(hand_obs.values())
    hands.sort(key=lambda item: (item.frame_idx, item.side, item.hand_index))
    if len(hands) < int(args.min_hands) or len(patch_obs) < int(args.min_patch_rows):
        raise RuntimeError(f"insufficient patch graph observations: hands={len(hands)} patches={len(patch_obs)} skipped={skipped[:20]}")
    return hands, patch_obs, {"skipped": skipped, "diagnostic": diagnostic}


def unpack(params: np.ndarray, hand_count: int, patch_count: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    hand_scale = math.exp(float(params[0]))
    shifts = params[1 : 1 + hand_count].astype(float)
    velocities = params[1 + hand_count : 1 + 2 * hand_count].astype(float)
    logits = params[1 + 2 * hand_count : 1 + 2 * hand_count + patch_count].astype(float)
    return hand_scale, shifts, velocities, logits


def hand_geometry(hand: HandObs, hand_scale: float, shift_z: float, K4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_t = solve_source_camera_translation(hand_scale * hand.local_joints, hand.target2d_vggt, K4)
    delta = np.asarray([0.0, 0.0, shift_z], dtype=float)
    joints = hand_scale * hand.local_joints + base_t[None, :] + delta[None, :]
    vertices = hand_scale * hand.local_vertices + base_t[None, :] + delta[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("nonpositive_hand_depth")
    return joints, vertices


def patch_residuals(vertices: np.ndarray, patch: PatchObs, K4: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    selected = vertices[patch.vertex_ids]
    uv = project_points(selected, K4)
    surface = patch.fragment["surface"]
    yx = surface["yx"]
    z = surface["depth_values"]
    n = int(len(patch.vertex_ids))
    gap = np.full(n, float(args.missing_patch_penalty_m), dtype=float)
    distance = np.full(n, float(args.missing_patch_penalty_m), dtype=float)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(selected).all(axis=1) & (selected[:, 2] > 0.0)
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= int(args.target_size) - 1)
    valid &= (uv[:, 1] >= 0.0) & (uv[:, 1] <= int(args.target_size) - 1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0 or yx.size == 0:
        return gap, distance, 0

    xy_surface = np.c_[yx[:, 1], yx[:, 0]].astype(float)
    xy = uv[valid_indices].astype(float)
    d2 = np.sum((xy[:, None, :] - xy_surface[None, :, :]) ** 2, axis=2)
    surface_idx = np.argmin(d2, axis=1)
    dist_px = np.sqrt(d2[np.arange(len(valid_indices)), surface_idx])
    near = dist_px <= float(args.contact_distance_px)
    if not np.any(near):
        return gap, distance, 0

    patch_indices = valid_indices[near]
    surface_idx = surface_idx[near]
    local_z = z[surface_idx].astype(float)
    fx, fy, cx, cy = K4.astype(float)
    u = xy_surface[surface_idx, 0]
    v = xy_surface[surface_idx, 1]
    surface_points = np.c_[(u - cx) / fx * local_z, (v - cy) / fy * local_z, local_z]
    gap[patch_indices] = selected[patch_indices, 2] - local_z
    distance[patch_indices] = np.linalg.norm(selected[patch_indices] - surface_points, axis=1)
    return gap, distance, int(np.count_nonzero(near))


def residual(params: np.ndarray, hands: list[HandObs], patches: list[PatchObs], hand_to_i: dict[tuple[int, str, int], int], K4: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hand_scale, shifts, velocities, logits = unpack(params, len(hands), len(patches))
    contact = sigmoid(logits)
    out: list[np.ndarray] = [np.asarray([math.log(hand_scale) / float(args.sigma_log_hand_scale)], dtype=float)]
    hand_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, hand in enumerate(hands):
        try:
            joints, vertices = hand_geometry(hand, hand_scale, shifts[i], K4)
            hand_cache[i] = (joints, vertices)
        except RuntimeError:
            out.append(np.full(45, float(args.invalid_penalty), dtype=float))
            continue
        uv = project_points(joints, K4)
        score_weight = min(1.0, max(0.0, hand.detector_score / float(args.detector_score_full)))
        out.append(score_weight * (uv - hand.target2d_vggt).reshape(-1) / float(args.sigma_reprojection_px))
        out.append(np.asarray([shifts[i] / float(args.sigma_shift_m)], dtype=float))
        out.append(np.asarray([velocities[i] / float(args.sigma_velocity_mps)], dtype=float))
        out.append(np.asarray([(hand_bone_scale_m(joints) - float(args.hand_bone_scale_prior_m)) / float(args.sigma_bone_scale_m)], dtype=float))

    for j, patch in enumerate(patches):
        hand_i = hand_to_i[patch.hand_key]
        if hand_i not in hand_cache:
            out.append(np.full(2 + 2 * int(len(patch.vertex_ids)), float(args.invalid_penalty), dtype=float))
            continue
        _, vertices = hand_cache[hand_i]
        gap, distance, count = patch_residuals(vertices, patch, K4, args)
        out.append(np.asarray([(logits[j] - logit(patch.seed_probability)) / float(args.sigma_contact_logit)], dtype=float))
        out.append(np.asarray([max(0, int(args.min_patch_vertices) - int(count)) / float(args.sigma_patch_coverage_vertices)], dtype=float))
        attraction = math.sqrt(max(float(contact[j]), 1e-6)) * gap / float(args.sigma_contact_gap_m)
        surface = math.sqrt(max(float(contact[j]), 1e-6)) * distance / float(args.sigma_contact_distance_m)
        out.append(np.clip(attraction, -float(args.clip_residual), float(args.clip_residual)))
        out.append(np.clip(surface, 0.0, float(args.clip_residual)))

    by_side: dict[str, list[int]] = {}
    for i, hand in enumerate(hands):
        by_side.setdefault(hand.side, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda i: (hands[i].frame_idx, hands[i].hand_index))
        for a, b in zip(indices[:-1], indices[1:]):
            dt = max(1.0 / float(args.fps), float(hands[b].frame_idx - hands[a].frame_idx) / float(args.fps))
            out.append(np.asarray([(shifts[b] - shifts[a] - velocities[a] * dt) / float(args.sigma_motion_m)], dtype=float))
            out.append(np.asarray([(velocities[b] - velocities[a]) / float(args.sigma_accel_mps2)], dtype=float))

    by_patch_track: dict[tuple[str, str, int], list[int]] = {}
    for j, patch in enumerate(patches):
        _, side, hand_index = patch.hand_key
        by_patch_track.setdefault((patch.track_id, side, hand_index), []).append(j)
    for indices in by_patch_track.values():
        indices.sort(key=lambda j: patches[j].frame_idx)
        for a, b in zip(indices[:-1], indices[1:]):
            dt_frames = max(1, patches[b].frame_idx - patches[a].frame_idx)
            if dt_frames <= int(args.max_contact_logit_gap_frames):
                out.append(np.asarray([(logits[b] - logits[a]) / float(args.sigma_contact_logit_step)], dtype=float))
    return np.concatenate([np.ravel(item).astype(float) for item in out])


def solve_graph(hands: list[HandObs], patches: list[PatchObs], K4: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    hand_to_i = {hand.key: i for i, hand in enumerate(hands)}
    n_h = len(hands)
    n_p = len(patches)
    x0 = np.zeros(1 + 2 * n_h + n_p, dtype=float)
    x0[1 + 2 * n_h :] = np.asarray([logit(patch.seed_probability) for patch in patches], dtype=float)
    lower = np.r_[
        np.log(float(args.min_hand_scale)),
        np.full(n_h, -float(args.max_abs_shift_m)),
        np.full(n_h, -float(args.max_abs_velocity_mps)),
        np.full(n_p, -float(args.max_abs_contact_logit)),
    ]
    upper = np.r_[
        np.log(float(args.max_hand_scale)),
        np.full(n_h, float(args.max_abs_shift_m)),
        np.full(n_h, float(args.max_abs_velocity_mps)),
        np.full(n_p, float(args.max_abs_contact_logit)),
    ]
    before = residual(x0, hands, patches, hand_to_i, K4, args)
    result = least_squares(
        lambda x: residual(x, hands, patches, hand_to_i, K4, args),
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
    )
    after = residual(result.x, hands, patches, hand_to_i, K4, args)
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


def patch_metrics(params: np.ndarray, hands: list[HandObs], patches: list[PatchObs], K4: np.ndarray, source_focal: float, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    hand_scale, shifts, velocities, logits = unpack(params, len(hands), len(patches))
    contact = sigmoid(logits)
    hand_to_i = {hand.key: i for i, hand in enumerate(hands)}
    hand_rows = []
    hand_cache = {}
    for i, hand in enumerate(hands):
        joints, vertices = hand_geometry(hand, hand_scale, shifts[i], K4)
        hand_cache[i] = (joints, vertices)
        uv = project_points(joints, K4)
        reproj = np.linalg.norm(uv - hand.target2d_vggt, axis=1)
        hand_rows.append(
            {
                "frame_idx": int(hand.frame_idx),
                "side": hand.side,
                "hand_index": int(hand.hand_index),
                "detector_score": float(hand.detector_score),
                "source_focal_px": float(source_focal),
                "hand_scale": float(hand_scale),
                "shift_z_m": float(shifts[i]),
                "velocity_z_mps": float(velocities[i]),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "hand_bone_scale_m": float(hand_bone_scale_m(joints)),
            }
        )
    patch_rows = []
    for j, patch in enumerate(patches):
        hand_i = hand_to_i[patch.hand_key]
        _, vertices = hand_cache[hand_i]
        gap, distance, count = patch_residuals(vertices, patch, K4, args)
        gstats = gap_stats(gap)
        dstats = surface_stats(distance)
        patch_rows.append(
            {
                "frame_idx": int(patch.frame_idx),
                "track_id": patch.track_id,
                "side": patch.hand_key[1],
                "hand_index": int(patch.hand_key[2]),
                "source_focal_px": float(source_focal),
                "contact_probability": float(contact[j]),
                "seed_probability": float(patch.seed_probability),
                "patch_vertices": int(count),
                "patch_vertex_ids": [int(v) for v in patch.vertex_ids.tolist()],
                "patch_local_center_m": [float(v) for v in patch.local_center.tolist()],
                "patch_local_spread_m": point_extent(vertices[patch.vertex_ids]),
                **{f"patch_{key}": value for key, value in gstats.items()},
                **{f"patch_{key}": value for key, value in dstats.items()},
                "hand_keypoint_reprojection_median_px": hand_rows[hand_i]["keypoint_reprojection_median_px"],
                "hand_shift_z_m": hand_rows[hand_i]["shift_z_m"],
            }
        )
    annotate_patch_support(patch_rows, args)
    return hand_rows, patch_rows


def annotate_patch_support(rows: list[dict], args: argparse.Namespace) -> None:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        row["patch_geometry_ok"] = bool(
            float(row["contact_probability"]) >= float(args.accept_contact_probability)
            and int(row["patch_vertices"]) >= int(args.min_patch_vertices)
            and row["patch_gap_median_m"] is not None
            and abs(float(row["patch_gap_median_m"])) <= float(args.accept_patch_gap_m)
            and row["patch_gap_p95_abs_m"] is not None
            and float(row["patch_gap_p95_abs_m"]) <= float(args.accept_patch_p95_gap_m)
            and row["patch_surface_distance_p95_m"] is not None
            and float(row["patch_surface_distance_p95_m"]) <= float(args.accept_patch_p95_distance_m)
            and float(row["hand_keypoint_reprojection_median_px"]) <= float(args.accept_reprojection_px)
        )
        row["patch_temporal_support_frames"] = 0
        row["patch_temporal_local_drift_m"] = None
        row["patch_temporal_support_ok"] = False
        row["reliable_patch_contact"] = False
        if row["patch_geometry_ok"]:
            groups.setdefault((row["track_id"], row["side"], int(row["hand_index"])), []).append(row)

    for candidates in groups.values():
        candidates.sort(key=lambda row: int(row["frame_idx"]))
        clusters: list[list[dict]] = []
        current: list[dict] = []
        for row in candidates:
            if not current or int(row["frame_idx"]) - int(current[-1]["frame_idx"]) <= int(args.max_temporal_patch_gap_frames):
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
        if current:
            clusters.append(current)
        for cluster in clusters:
            frames = [int(row["frame_idx"]) for row in cluster]
            centers = [np.asarray(row["patch_local_center_m"], dtype=float) for row in cluster]
            drift = point_extent(np.stack(centers, axis=0)) if len(centers) >= 2 else None
            ok = bool(len(set(frames)) >= int(args.min_temporal_patch_frames) and drift is not None and drift <= float(args.accept_temporal_patch_local_drift_m))
            for row in cluster:
                row["patch_temporal_support_frames"] = int(len(set(frames)))
                row["patch_temporal_local_drift_m"] = None if drift is None else float(drift)
                row["patch_temporal_support_ok"] = ok
                row["reliable_patch_contact"] = bool(row["patch_geometry_ok"] and ok)


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
    return summarize(values)


def status_for(params: np.ndarray, hand_rows: list[dict], patch_rows: list[dict], args: argparse.Namespace) -> str:
    hand_scale, shifts, velocities, logits = unpack(params, len(hand_rows), len(patch_rows))
    if hand_scale <= float(args.min_hand_scale) + float(args.bound_tolerance) or hand_scale >= float(args.max_hand_scale) - float(args.bound_tolerance):
        return "diagnostic_solution_requires_hand_scale_bound"
    if np.any(np.abs(shifts) >= float(args.max_abs_shift_m) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_shift_bound"
    if np.any(np.abs(velocities) >= float(args.max_abs_velocity_mps) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_velocity_bound"
    if any(row["reliable_patch_contact"] for row in patch_rows):
        return "diagnostic_fragment_patch_contact_candidate"
    return "diagnostic_no_reliable_fragment_patch_contact_rows"


def run_one_focal(args: argparse.Namespace, source_focal: float) -> dict:
    focal_key = f"{float(source_focal):.6f}"
    hands, patches, meta = build_focal_problem(args, focal_key)
    K4 = source_focal_to_vggt_intrinsics(
        float(source_focal),
        int(args.width),
        int(args.height),
        float(args.cx),
        float(args.cy),
        int(args.target_size),
    )
    params, solver = solve_graph(hands, patches, K4, args)
    hand_rows, patch_rows = patch_metrics(params, hands, patches, K4, float(source_focal), args)
    return {
        "source_focal_px": float(source_focal),
        "status": status_for(params, hand_rows, patch_rows, args),
        "observed_hands": int(len(hands)),
        "patch_rows": int(len(patches)),
        "variables": int(len(params)),
        "solver": solver,
        "hand_summary": {
            "hand_scale": summarize_key(hand_rows, "hand_scale"),
            "shift_abs_m": summarize([abs(row["shift_z_m"]) for row in hand_rows]),
            "velocity_abs_mps": summarize([abs(row["velocity_z_mps"]) for row in hand_rows]),
            "keypoint_reprojection_median_px": summarize_key(hand_rows, "keypoint_reprojection_median_px"),
            "hand_bone_scale_m": summarize_key(hand_rows, "hand_bone_scale_m"),
        },
        "patch_summary": {
            "contact_probability": summarize_key(patch_rows, "contact_probability"),
            "patch_gap_median_m": summarize_key(patch_rows, "patch_gap_median_m"),
            "patch_gap_p95_abs_m": summarize_key(patch_rows, "patch_gap_p95_abs_m"),
            "patch_surface_distance_p95_m": summarize_key(patch_rows, "patch_surface_distance_p95_m"),
            "geometry_ok_rows": int(sum(row["patch_geometry_ok"] for row in patch_rows)),
            "temporal_support_ok_rows": int(sum(row["patch_temporal_support_ok"] for row in patch_rows)),
            "reliable_patch_contact_rows": int(sum(row["reliable_patch_contact"] for row in patch_rows)),
        },
        "hand_rows": hand_rows,
        "patch_rows_detail": patch_rows,
        "skipped_preview": meta["skipped"][:80],
    }


def run(args: argparse.Namespace) -> dict:
    diagnostic = load_json(args.patch_diagnostic)
    available = {float(key): key for key in diagnostic.get("rows_by_focal", {}).keys()}
    source_focals = [float(value) for value in args.source_focals]
    if args.include_vggt_predicted_focal:
        predicted = diagnostic["vggt_predicted_source_focal"]["median_mean_source_focal_px"]
        if all(abs(float(predicted) - value) > 1e-6 for value in source_focals):
            source_focals.append(float(predicted))
    source_focals = sorted(source_focals)
    reports = []
    skipped = []
    for source_focal in source_focals:
        key = f"{float(source_focal):.6f}"
        if key not in diagnostic.get("rows_by_focal", {}):
            skipped.append({"source_focal_px": float(source_focal), "reason": "missing_focal_in_patch_diagnostic"})
            continue
        try:
            reports.append(run_one_focal(args, float(source_focal)))
        except Exception as exc:
            skipped.append({"source_focal_px": float(source_focal), "reason": str(exc)})
    if not reports:
        raise RuntimeError(f"no focal reports produced; skipped={skipped}")
    ranked = sorted(
        reports,
        key=lambda report: (
            -int(report["patch_summary"]["reliable_patch_contact_rows"]),
            float(report["patch_summary"]["patch_surface_distance_p95_m"].get("median", math.inf)),
            float(report["hand_summary"]["keypoint_reprojection_median_px"].get("median", math.inf)),
        ),
    )
    report = {
        "status": "diagnostic_fragment_patch_contact_candidate"
        if any(item["patch_summary"]["reliable_patch_contact_rows"] for item in reports)
        else "diagnostic_no_reliable_fragment_patch_contact_rows",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_vggt_fragment_patch_hand_graph_v3",
        "annotations": str(args.annotations),
        "patch_diagnostic": str(args.patch_diagnostic),
        "vggt_archive": str(args.vggt_archive),
        "sam2_root": str(args.sam2_root),
        "source_focals": source_focals,
        "focal_reports": reports,
        "ranked_focals": [
            {
                "source_focal_px": item["source_focal_px"],
                "status": item["status"],
                "patch_summary": item["patch_summary"],
                "hand_summary": item["hand_summary"],
            }
            for item in ranked
        ],
        "skipped_focals": skipped,
        "parameters": vars(args),
        "interpretation": (
            "This graph tests whether the local VGGT/SAM2 fragment patch candidates survive temporal MANO depth shifts. "
            "It keeps contact as a latent switch: patch attraction is weighted by contact probability, while reprojection, "
            "hand scale, shift bounds, and MANO-local temporal continuity decide whether a row becomes reliable contact."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"focal_reports", "parameters"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--patch-diagnostic", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-focals", type=float, nargs="*", default=[1200.0, 1400.0, 1600.0, 1800.0, 2000.0])
    parser.add_argument("--include-vggt-predicted-focal", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-output-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    parser.add_argument("--min-depth-conf", type=float, default=0.0)
    parser.add_argument("--conf-quantile", type=float, default=0.0)
    parser.add_argument("--min-vggt-mask-pixels", type=int, default=20)
    parser.add_argument("--min-depth-pixels", type=int, default=20)
    parser.add_argument("--min-hands", type=int, default=2)
    parser.add_argument("--min-patch-rows", type=int, default=4)
    parser.add_argument("--min-patch-vertices", type=int, default=8)
    parser.add_argument("--min-detector-score", type=float, default=0.45)
    parser.add_argument("--max-seed-reprojection-px", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--detector-score-full", type=float, default=0.80)
    parser.add_argument("--sigma-log-hand-scale", type=float, default=0.16)
    parser.add_argument("--sigma-reprojection-px", type=float, default=5.0)
    parser.add_argument("--sigma-shift-m", type=float, default=0.10)
    parser.add_argument("--sigma-velocity-mps", type=float, default=0.70)
    parser.add_argument("--sigma-motion-m", type=float, default=0.020)
    parser.add_argument("--sigma-accel-mps2", type=float, default=1.20)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.205)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.025)
    parser.add_argument("--sigma-contact-logit", type=float, default=1.60)
    parser.add_argument("--sigma-contact-logit-step", type=float, default=0.90)
    parser.add_argument("--sigma-contact-without-patch", type=float, default=0.25)
    parser.add_argument("--sigma-patch-coverage-vertices", type=float, default=4.0)
    parser.add_argument("--sigma-contact-gap-m", type=float, default=0.020)
    parser.add_argument("--sigma-contact-distance-m", type=float, default=0.030)
    parser.add_argument("--missing-patch-penalty-m", type=float, default=0.250)
    parser.add_argument("--min-hand-scale", type=float, default=0.86)
    parser.add_argument("--max-hand-scale", type=float, default=1.10)
    parser.add_argument("--max-abs-shift-m", type=float, default=0.16)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=1.00)
    parser.add_argument("--max-abs-contact-logit", type=float, default=5.0)
    parser.add_argument("--seed-distance-scale-m", type=float, default=0.060)
    parser.add_argument("--seed-gap-scale-m", type=float, default=0.060)
    parser.add_argument("--seed-spread-scale-m", type=float, default=0.050)
    parser.add_argument("--min-contact-seed", type=float, default=0.02)
    parser.add_argument("--max-contact-seed-boost", type=float, default=0.78)
    parser.add_argument("--accept-contact-probability", type=float, default=0.50)
    parser.add_argument("--accept-patch-gap-m", type=float, default=0.020)
    parser.add_argument("--accept-patch-p95-gap-m", type=float, default=0.040)
    parser.add_argument("--accept-patch-p95-distance-m", type=float, default=0.040)
    parser.add_argument("--accept-patch-spread-m", type=float, default=0.050)
    parser.add_argument("--accept-reprojection-px", type=float, default=12.0)
    parser.add_argument("--min-temporal-patch-frames", type=int, default=2)
    parser.add_argument("--max-temporal-patch-gap-frames", type=int, default=8)
    parser.add_argument("--max-contact-logit-gap-frames", type=int, default=8)
    parser.add_argument("--accept-temporal-patch-local-drift-m", type=float, default=0.030)
    parser.add_argument("--invalid-penalty", type=float, default=1e3)
    parser.add_argument("--clip-residual", type=float, default=60.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-nfev", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
