#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m
from diagnose_intrinsics_focal_sweep_v3 import mask_distance_map, project_points, solve_source_camera_translation, source_local_vertices
from diagnose_vggt_focal_sweep_v3 import source_focal_to_vggt_intrinsics, vggt_predicted_source_focals
from diagnose_vggt_mano_contact_v3 import points_to_vggt_frame, resize_mask, vggt_frame_points


@dataclass(frozen=True)
class Obs:
    frame_idx: int
    side: str
    hand_index: int
    detector_score: float
    local_joints: np.ndarray
    local_vertices: np.ndarray
    target2d_vggt: np.ndarray
    object_depth_vggt: float
    mask_distance: np.ndarray
    source_width: int
    source_height: int
    source_cx: float
    source_cy: float
    initial_contact_seed: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def object_depth_vggt(archive: np.lib.npyio.NpzFile, frame_i: int) -> float:
    points = vggt_frame_points(archive, frame_i)
    extrinsic = archive["extrinsic"].astype(float)[frame_i]
    camera = (points @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if z.size == 0:
        raise RuntimeError(f"VGGT object points have no positive camera depth at index {frame_i}")
    return float(np.median(z))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: float) -> float:
    q = float(np.clip(p, 1e-5, 1.0 - 1e-5))
    return math.log(q / (1.0 - q))


def initial_contact_seed(vertices: np.ndarray, K4: np.ndarray, mask_distance: np.ndarray, object_depth: float, args: argparse.Namespace) -> float:
    if np.any(vertices[:, 2] <= 0.0):
        return 0.05
    uv = project_points(vertices, K4)
    valid = np.isfinite(uv).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, int(args.target_size) - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, int(args.target_size) - 1)
    near = valid & (mask_distance[y, x] <= float(args.contact_distance_px))
    if int(np.count_nonzero(near)) < int(args.min_near_vertices):
        return 0.05
    gap = vertices[near, 2] - float(object_depth)
    closest = float(np.percentile(np.abs(gap), 15.0))
    overlap = min(1.0, float(np.count_nonzero(near)) / float(args.contact_seed_full_vertices))
    depth_term = math.exp(-closest / max(float(args.contact_seed_scale_m), 1e-6))
    return float(np.clip(0.05 + 0.90 * overlap * depth_term, 0.05, 0.95))


def build_obs(args: argparse.Namespace) -> tuple[list[Obs], dict, list[dict]]:
    annotations = load_json(args.annotations)
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    mask_by_idx = {int(row["frame_idx"]): Path(row["mask"]) for row in load_json(args.dataset_manifest)["frames"]}
    archive = np.load(args.vggt_archive)
    frames = archive["frame_idx"].astype(int)
    predicted = vggt_predicted_source_focals(archive, int(args.width), int(args.height), int(args.target_size))
    K_seed = source_focal_to_vggt_intrinsics(
        float(args.initial_source_focal_px),
        int(args.width),
        int(args.height),
        float(args.cx),
        float(args.cy),
        int(args.target_size),
    )
    out: list[Obs] = []
    skipped: list[dict] = []
    for i, frame_idx in enumerate(frames.tolist()):
        frame = frame_by_idx.get(int(frame_idx))
        if frame is None:
            skipped.append({"frame_idx": int(frame_idx), "reason": "missing_annotation"})
            continue
        mask_path = mask_by_idx.get(int(frame_idx))
        if mask_path is None:
            skipped.append({"frame_idx": int(frame_idx), "reason": "missing_mask"})
            continue
        try:
            dist = mask_distance_map(resize_mask(mask_path, int(args.target_size)))
            object_depth = object_depth_vggt(archive, i)
        except Exception as exc:
            skipped.append({"frame_idx": int(frame_idx), "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            side = str(hand.get("side", "unknown"))
            try:
                if not bool(hand.get("measurement_available", False)):
                    raise RuntimeError("predicted_hand")
                score = float(hand.get("detector_score", np.nan))
                if not np.isfinite(score) or score < float(args.min_detector_score):
                    raise RuntimeError("low_detector_score")
                local_joints = np.asarray(hand.get("joints3d_camera", []), dtype=float)
                local_vertices = source_local_vertices(hand)
                raw2d = np.asarray(hand.get("joints2d_raw", []), dtype=float)
                if local_joints.shape != (21, 3) or raw2d.shape != (21, 2):
                    raise RuntimeError("invalid_hand_joints")
                if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid_hand_vertices")
                target2d = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], int(args.target_size))
                base_t = solve_source_camera_translation(local_joints, target2d, K_seed)
                seed_vertices = local_vertices + base_t[None, :]
                seed = initial_contact_seed(seed_vertices, K_seed, dist, object_depth, args)
                out.append(
                    Obs(
                        frame_idx=int(frame_idx),
                        side=side,
                        hand_index=int(hand_i),
                        detector_score=score,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        target2d_vggt=target2d,
                        object_depth_vggt=object_depth,
                        mask_distance=dist,
                        source_width=int(args.width),
                        source_height=int(args.height),
                        source_cx=float(args.cx),
                        source_cy=float(args.cy),
                        initial_contact_seed=seed,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": int(frame_idx), "hand_index": int(hand_i), "side": side, "reason": str(exc)})
    if len(out) < int(args.min_rows):
        raise RuntimeError(f"insufficient observations: {len(out)}; skipped={skipped[:20]}")
    return out, predicted, skipped


def unpack(params: np.ndarray, n: int, args: argparse.Namespace) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    source_focal = float(args.initial_source_focal_px) * math.exp(float(params[0]))
    hand_scale = math.exp(float(params[1]))
    shifts = params[2 : 2 + n].astype(float)
    velocities = params[2 + n : 2 + 2 * n].astype(float)
    contact_logits = params[2 + 2 * n : 2 + 3 * n].astype(float)
    return source_focal, hand_scale, shifts, velocities, contact_logits


def geometry(obs: Obs, source_focal: float, hand_scale: float, shift_z: float, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K4 = source_focal_to_vggt_intrinsics(
        source_focal,
        obs.source_width,
        obs.source_height,
        obs.source_cx,
        obs.source_cy,
        int(args.target_size),
    )
    base_t = solve_source_camera_translation(hand_scale * obs.local_joints, obs.target2d_vggt, K4)
    delta = np.asarray([0.0, 0.0, shift_z], dtype=float)
    joints = hand_scale * obs.local_joints + base_t[None, :] + delta[None, :]
    vertices = hand_scale * obs.local_vertices + base_t[None, :] + delta[None, :]
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError("nonpositive hand depth")
    return K4, joints, vertices


def contact_arrays(obs: Obs, vertices: np.ndarray, K4: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    uv = project_points(vertices, K4)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, int(args.target_size) - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, int(args.target_size) - 1)
    dist = obs.mask_distance[y, x]
    near = valid & (dist <= float(args.contact_distance_px))
    gap = vertices[:, 2] - obs.object_depth_vggt
    return gap[near], dist[near]


def selected_contact_gap(gap: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if gap.size == 0:
        return np.zeros(0, dtype=float)
    count = min(int(args.contact_patch_vertices), gap.size)
    order = np.argsort(np.abs(gap))[:count]
    return gap[order]


def residual(params: np.ndarray, obs: list[Obs], args: argparse.Namespace) -> np.ndarray:
    source_focal, hand_scale, shifts, velocities, contact_logits = unpack(params, len(obs), args)
    contacts = sigmoid(contact_logits)
    out: list[np.ndarray] = [
        np.asarray([math.log(source_focal / float(args.initial_source_focal_px)) / float(args.sigma_log_focal)], dtype=float),
        np.asarray([math.log(hand_scale) / float(args.sigma_log_hand_scale)], dtype=float),
    ]
    for i, row in enumerate(obs):
        try:
            K4, joints, vertices = geometry(row, source_focal, hand_scale, shifts[i], args)
        except RuntimeError:
            out.append(np.full(16, float(args.invalid_penalty), dtype=float))
            continue
        uv = project_points(joints, K4)
        score_weight = min(1.0, max(0.0, row.detector_score / float(args.detector_score_full)))
        out.append(score_weight * (uv - row.target2d_vggt).reshape(-1) / float(args.sigma_reprojection_px))
        out.append(np.asarray([shifts[i] / float(args.sigma_shift_m)], dtype=float))
        out.append(np.asarray([velocities[i] / float(args.sigma_velocity_mps)], dtype=float))
        out.append(np.asarray([(hand_bone_scale_m(joints) - float(args.hand_bone_scale_prior_m)) / float(args.sigma_bone_scale_m)], dtype=float))
        prior = logit(row.initial_contact_seed)
        out.append(np.asarray([(contact_logits[i] - prior) / float(args.sigma_contact_logit)], dtype=float))
        gap, _ = contact_arrays(row, vertices, K4, args)
        if gap.size == 0:
            out.append(np.asarray([contacts[i] / float(args.sigma_contact_without_overlap)], dtype=float))
            continue
        patch_gap = selected_contact_gap(gap, args)
        if patch_gap.size:
            out.append(math.sqrt(max(float(contacts[i]), 1e-6)) * patch_gap / float(args.sigma_contact_m))
        penetration = np.maximum(gap - float(args.penetration_margin_m), 0.0)
        if penetration.size > int(args.max_penetration_vertices):
            order = np.argsort(penetration)[-int(args.max_penetration_vertices) :]
            penetration = penetration[order]
        out.append(np.clip(penetration / float(args.sigma_penetration_m), 0.0, float(args.clip_residual)))
    by_side: dict[str, list[int]] = {}
    for i, row in enumerate(obs):
        by_side.setdefault(row.side, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda j: obs[j].frame_idx)
        for a, b in zip(indices[:-1], indices[1:]):
            dt = max(1.0 / float(args.fps), float(obs[b].frame_idx - obs[a].frame_idx) / float(args.fps))
            out.append(np.asarray([(shifts[b] - shifts[a] - velocities[a] * dt) / float(args.sigma_motion_m)], dtype=float))
            out.append(np.asarray([(velocities[b] - velocities[a]) / float(args.sigma_accel_mps2)], dtype=float))
            out.append(np.asarray([(contact_logits[b] - contact_logits[a]) / float(args.sigma_contact_logit_step)], dtype=float))
    return np.concatenate([np.ravel(item).astype(float) for item in out])


def solve(obs: list[Obs], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    n = len(obs)
    x0 = np.zeros(2 + 3 * n, dtype=float)
    x0[2 + 2 * n : 2 + 3 * n] = np.asarray([logit(row.initial_contact_seed) for row in obs], dtype=float)
    lower = np.r_[
        np.log(float(args.min_source_focal_px) / float(args.initial_source_focal_px)),
        np.log(float(args.min_hand_scale)),
        np.full(n, -float(args.max_abs_shift_m)),
        np.full(n, -float(args.max_abs_velocity_mps)),
        np.full(n, -float(args.max_abs_contact_logit)),
    ]
    upper = np.r_[
        np.log(float(args.max_source_focal_px) / float(args.initial_source_focal_px)),
        np.log(float(args.max_hand_scale)),
        np.full(n, float(args.max_abs_shift_m)),
        np.full(n, float(args.max_abs_velocity_mps)),
        np.full(n, float(args.max_abs_contact_logit)),
    ]
    before = residual(x0, obs, args)
    result = least_squares(
        lambda x: residual(x, obs, args),
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
    )
    after = residual(result.x, obs, args)
    return result.x, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "rms_before": float(np.sqrt(np.mean(before * before))),
        "rms_after": float(np.sqrt(np.mean(after * after))),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
    }


def gap_metrics(gap: np.ndarray) -> dict:
    if gap.size == 0:
        return {
            "near_mask_vertices": 0,
            "gap_median_m": None,
            "gap_p95_abs_m": None,
            "contact_vertices_010m": 0,
            "contact_vertices_030m": 0,
            "contact_fraction_010m": 0.0,
            "contact_fraction_030m": 0.0,
            "penetration_fraction_010m": 0.0,
            "penetration_fraction_030m": 0.0,
            "positive_gap_p95_m": None,
            "negative_gap_p05_m": None,
        }
    abs_gap = np.abs(gap)
    positive = np.maximum(gap, 0.0)
    return {
        "near_mask_vertices": int(gap.size),
        "gap_median_m": float(np.median(gap)),
        "gap_p95_abs_m": float(np.percentile(abs_gap, 95.0)),
        "contact_vertices_010m": int(np.count_nonzero(abs_gap <= 0.010)),
        "contact_vertices_030m": int(np.count_nonzero(abs_gap <= 0.030)),
        "contact_fraction_010m": float(np.mean(abs_gap <= 0.010)),
        "contact_fraction_030m": float(np.mean(abs_gap <= 0.030)),
        "penetration_fraction_010m": float(np.mean(gap > 0.010)),
        "penetration_fraction_030m": float(np.mean(gap > 0.030)),
        "positive_gap_p95_m": float(np.percentile(positive, 95.0)),
        "negative_gap_p05_m": float(np.percentile(gap, 5.0)),
    }


def row_metrics(params: np.ndarray, obs: list[Obs], args: argparse.Namespace) -> list[dict]:
    source_focal, hand_scale, shifts, velocities, contact_logits = unpack(params, len(obs), args)
    contacts = sigmoid(contact_logits)
    rows = []
    for i, row in enumerate(obs):
        K4, joints, vertices = geometry(row, source_focal, hand_scale, shifts[i], args)
        uv = project_points(joints, K4)
        reproj = np.linalg.norm(uv - row.target2d_vggt, axis=1)
        gap, dist = contact_arrays(row, vertices, K4, args)
        stats = gap_metrics(gap)
        rows.append(
            {
                "frame_idx": int(row.frame_idx),
                "side": row.side,
                "hand_index": int(row.hand_index),
                "detector_score": float(row.detector_score),
                "source_focal_px": float(source_focal),
                "hand_scale": float(hand_scale),
                "shift_z_m": float(shifts[i]),
                "velocity_z_mps": float(velocities[i]),
                "contact_probability": float(contacts[i]),
                "initial_contact_seed": float(row.initial_contact_seed),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "hand_bone_scale_m": float(hand_bone_scale_m(joints)),
                "object_depth_vggt": float(row.object_depth_vggt),
                "hand_depth_median_m": float(np.median(vertices[:, 2])),
                **stats,
            }
        )
    return rows


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values.append(value_f)
    return summarize(values)


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "rows": int(len(rows)),
        "source_focal_px": summarize_key(rows, "source_focal_px"),
        "hand_scale": summarize_key(rows, "hand_scale"),
        "shift_abs_m": summarize([abs(row["shift_z_m"]) for row in rows]),
        "velocity_abs_mps": summarize([abs(row["velocity_z_mps"]) for row in rows]),
        "contact_probability": summarize_key(rows, "contact_probability"),
        "keypoint_reprojection_median_px": summarize_key(rows, "keypoint_reprojection_median_px"),
        "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
        "near_mask_vertices": summarize_key(rows, "near_mask_vertices"),
        "gap_median_m": summarize_key(rows, "gap_median_m"),
        "gap_p95_abs_m": summarize_key(rows, "gap_p95_abs_m"),
        "contact_vertices_010m": summarize_key(rows, "contact_vertices_010m"),
        "contact_vertices_030m": summarize_key(rows, "contact_vertices_030m"),
        "contact_fraction_010m": summarize_key(rows, "contact_fraction_010m"),
        "contact_fraction_030m": summarize_key(rows, "contact_fraction_030m"),
        "penetration_fraction_010m": summarize_key(rows, "penetration_fraction_010m"),
        "penetration_fraction_030m": summarize_key(rows, "penetration_fraction_030m"),
        "positive_gap_p95_m": summarize_key(rows, "positive_gap_p95_m"),
        "negative_gap_p05_m": summarize_key(rows, "negative_gap_p05_m"),
    }


def status_for(params: np.ndarray, rows: list[dict], args: argparse.Namespace) -> str:
    source_focal, hand_scale, shifts, velocities, logits = unpack(params, len(rows), args)
    if source_focal <= float(args.min_source_focal_px) + float(args.bound_tolerance) or source_focal >= float(args.max_source_focal_px) - float(args.bound_tolerance):
        return "diagnostic_solution_requires_focal_bound"
    if hand_scale <= float(args.min_hand_scale) + float(args.bound_tolerance) or hand_scale >= float(args.max_hand_scale) - float(args.bound_tolerance):
        return "diagnostic_solution_requires_hand_scale_bound"
    if np.any(np.abs(shifts) >= float(args.max_abs_shift_m) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_shift_bound"
    if np.any(np.abs(velocities) >= float(args.max_abs_velocity_mps) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_velocity_bound"
    reproj = np.asarray([row["keypoint_reprojection_median_px"] for row in rows], dtype=float)
    penetration = np.asarray([row["positive_gap_p95_m"] for row in rows if row["positive_gap_p95_m"] is not None], dtype=float)
    contact30 = np.asarray([row["contact_fraction_030m"] for row in rows], dtype=float)
    if float(np.median(reproj)) > float(args.accept_reprojection_median_px):
        return "diagnostic_reprojection_residual_too_large"
    if penetration.size and float(np.percentile(penetration, 95.0)) > float(args.accept_penetration_p95_m):
        return "diagnostic_penetration_residual_remains"
    if contact30.size and float(np.median(contact30)) < float(args.accept_contact_fraction_030):
        return "diagnostic_contact_support_too_sparse"
    return "diagnostic_vggt_focal_hand_contact_candidate"


def run(args: argparse.Namespace) -> dict:
    obs, predicted, skipped = build_obs(args)
    n = len(obs)
    x0 = np.zeros(2 + 3 * n, dtype=float)
    x0[2 + 2 * n : 2 + 3 * n] = np.asarray([logit(row.initial_contact_seed) for row in obs], dtype=float)
    before_rows = row_metrics(x0, obs, args)
    params, solver = solve(obs, args)
    after_rows = row_metrics(params, obs, args)
    source_focal, hand_scale, shifts, velocities, logits = unpack(params, len(obs), args)
    report = {
        "status": status_for(params, after_rows, args),
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_vggt_focal_hand_contact_graph_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "observations": int(len(obs)),
        "skipped_rows": int(len(skipped)),
        "variables": int(len(params)),
        "solver": solver,
        "vggt_predicted_source_focal_px": predicted,
        "source_focal_px": float(source_focal),
        "hand_scale": float(hand_scale),
        "before_summary": summarize_rows(before_rows),
        "after_summary": summarize_rows(after_rows),
        "rows": after_rows,
        "skipped_preview": skipped[:160],
        "parameters": vars(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qc_vggt_focal_hand_contact_graph_v3.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "skipped_preview", "parameters"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--cx", type=float, default=960.0)
    parser.add_argument("--cy", type=float, default=540.0)
    parser.add_argument("--initial-source-focal-px", type=float, default=1400.0)
    parser.add_argument("--min-source-focal-px", type=float, default=900.0)
    parser.add_argument("--max-source-focal-px", type=float, default=2600.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--detector-score-full", type=float, default=0.80)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--contact-seed-full-vertices", type=int, default=300)
    parser.add_argument("--contact-seed-scale-m", type=float, default=0.060)
    parser.add_argument("--contact-patch-vertices", type=int, default=80)
    parser.add_argument("--max-penetration-vertices", type=int, default=180)
    parser.add_argument("--sigma-log-focal", type=float, default=0.45)
    parser.add_argument("--sigma-log-hand-scale", type=float, default=0.18)
    parser.add_argument("--sigma-reprojection-px", type=float, default=5.0)
    parser.add_argument("--sigma-shift-m", type=float, default=0.10)
    parser.add_argument("--sigma-velocity-mps", type=float, default=0.55)
    parser.add_argument("--sigma-motion-m", type=float, default=0.018)
    parser.add_argument("--sigma-accel-mps2", type=float, default=1.00)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.210)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.020)
    parser.add_argument("--sigma-contact-logit", type=float, default=1.60)
    parser.add_argument("--sigma-contact-logit-step", type=float, default=0.80)
    parser.add_argument("--sigma-contact-without-overlap", type=float, default=0.35)
    parser.add_argument("--sigma-contact-m", type=float, default=0.015)
    parser.add_argument("--penetration-margin-m", type=float, default=0.004)
    parser.add_argument("--sigma-penetration-m", type=float, default=0.010)
    parser.add_argument("--min-hand-scale", type=float, default=0.86)
    parser.add_argument("--max-hand-scale", type=float, default=1.10)
    parser.add_argument("--max-abs-shift-m", type=float, default=0.18)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=0.90)
    parser.add_argument("--max-abs-contact-logit", type=float, default=5.0)
    parser.add_argument("--max-nfev", type=int, default=180)
    parser.add_argument("--invalid-penalty", type=float, default=1e3)
    parser.add_argument("--clip-residual", type=float, default=60.0)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--accept-reprojection-median-px", type=float, default=12.0)
    parser.add_argument("--accept-penetration-p95-m", type=float, default=0.030)
    parser.add_argument("--accept-contact-fraction-030", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
