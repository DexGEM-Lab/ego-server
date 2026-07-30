#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_contact_depth_scale_v3 import summarize
from optimize_hand_translation_contact_v3 import (
    HandObs,
    build_observations,
    hand_span,
    source_to_world,
)


OBS_RESIDUAL_COUNT = 42 + 21 + 3 + 1 + 2 + 1


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def local_center(obs: HandObs) -> np.ndarray:
    return obs.local_joints[0].astype(float)


def scaled_local(points: np.ndarray, obs: HandObs, log_scale: float) -> np.ndarray:
    center = local_center(obs)
    scale = float(np.exp(log_scale))
    return center[None, :] + scale * (points - center[None, :])


def corrected_joints(obs: HandObs, cam_t: np.ndarray, log_scale: float) -> np.ndarray:
    pts = scaled_local(obs.local_joints, obs, log_scale) + cam_t[None, :]
    if np.any(pts[:, 2] <= 0.0):
        raise RuntimeError("corrected joints have non-positive depth")
    return pts


def corrected_vertices(obs: HandObs, cam_t: np.ndarray, log_scale: float) -> np.ndarray:
    pts = scaled_local(obs.local_vertices, obs, log_scale) + cam_t[None, :]
    if np.any(pts[:, 2] <= 0.0):
        raise RuntimeError("corrected vertices have non-positive depth")
    return pts


def initial_log_scale(obs: HandObs, args: argparse.Namespace) -> float:
    span = hand_span(obs.local_joints + obs.base_cam_t[None, :])
    if not np.isfinite(span) or span <= 1e-6:
        return 0.0
    if span < args.min_span_m:
        return float(np.log(args.min_span_m / span))
    if span > args.max_span_m:
        return float(np.log(args.max_span_m / span))
    return 0.0


def variables0(observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    rows = []
    for obs in observations:
        rows.append(np.r_[obs.base_cam_t, initial_log_scale(obs, args)])
    return np.asarray(rows, dtype=float).reshape(-1)


def unpack(params: np.ndarray, observations: list[HandObs]) -> dict[tuple[int, str, int], tuple[np.ndarray, float]]:
    arr = np.asarray(params, dtype=float).reshape(len(observations), 4)
    return {obs.key: (arr[i, :3], float(arr[i, 3])) for i, obs in enumerate(observations)}


def residual(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    x = np.asarray(params, dtype=float).reshape(len(observations), 4)
    out: list[float] = []
    for i, obs in enumerate(observations):
        cam_t = x[i, :3]
        log_scale = float(x[i, 3])
        try:
            joints = corrected_joints(obs, cam_t, log_scale)
            vertices = corrected_vertices(obs, cam_t, log_scale)
        except RuntimeError:
            out.extend([args.invalid_penalty] * OBS_RESIDUAL_COUNT)
            continue
        projected = project_points(joints, obs.intrinsics)
        out.extend(np.clip(((projected - obs.raw2d) / args.sigma_reprojection_px).ravel(), -args.clip_residual, args.clip_residual))
        good_depth = obs.depth_valid & obs.stable_depth
        for joint_i in range(21):
            if good_depth[joint_i]:
                value = (joints[joint_i, 2] - obs.metric_depth[joint_i]) / args.sigma_metric_depth_m
                out.append(float(np.clip(value, -args.clip_residual, args.clip_residual)))
            else:
                out.append(0.0)
        out.extend((cam_t - obs.base_cam_t) / args.sigma_translation_prior_m)
        out.append(log_scale / args.sigma_log_scale)
        span = hand_span(joints)
        out.append(max(0.0, args.min_span_m - span) / args.sigma_span_m)
        out.append(max(0.0, span - args.max_span_m) / args.sigma_span_m)
        if len(obs.contact_vertex_ids) >= args.min_near_vertices:
            contact_z = vertices[obs.contact_vertex_ids, 2] - obs.object_depth_m
            contact_stat = float(np.median(contact_z))
            out.append(np.clip(contact_stat / args.sigma_contact_depth_m, -args.clip_residual, args.clip_residual))
        else:
            out.append(0.0)

    by_side: dict[str, list[tuple[int, np.ndarray]]] = {}
    for i, obs in enumerate(observations):
        by_side.setdefault(obs.side, []).append((obs.frame_order, x[i]))
    for items in by_side.values():
        items.sort(key=lambda t: t[0])
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            out.extend((b[:3] - a[:3]) / args.sigma_temporal_translation_m)
            out.append((b[3] - a[3]) / args.sigma_temporal_log_scale)
    return np.asarray(out, dtype=float)


def row_metric(obs: HandObs, cam_t: np.ndarray, log_scale: float) -> dict:
    joints = corrected_joints(obs, cam_t, log_scale)
    vertices = corrected_vertices(obs, cam_t, log_scale)
    projected = project_points(joints, obs.intrinsics)
    reproj = np.linalg.norm(projected - obs.raw2d, axis=1)
    depth_valid = obs.depth_valid & obs.stable_depth
    depth_err = joints[depth_valid, 2] - obs.metric_depth[depth_valid]
    contact_gap = np.asarray([], dtype=float)
    if len(obs.contact_vertex_ids):
        contact_gap = vertices[obs.contact_vertex_ids, 2] - obs.object_depth_m
    scale = float(np.exp(log_scale))
    return {
        "frame_idx": int(obs.frame_idx),
        "side": obs.side,
        "hand_index": int(obs.hand_index),
        "detector_score": float(obs.score),
        "translation_shift_m": (cam_t - obs.base_cam_t).astype(float).tolist(),
        "translation_shift_norm_m": float(np.linalg.norm(cam_t - obs.base_cam_t)),
        "local_scale": scale,
        "joint_reprojection_px_median": float(np.median(reproj)),
        "joint_reprojection_px_p95": float(np.percentile(reproj, 95.0)),
        "depth_joints": int(np.count_nonzero(depth_valid)),
        "mano_minus_metric_depth_median_m": None if len(depth_err) == 0 else float(np.median(depth_err)),
        "mano_minus_metric_depth_p95_abs_m": None if len(depth_err) == 0 else float(np.percentile(np.abs(depth_err), 95.0)),
        "hand_span_m": hand_span(joints),
        "near_mask_vertices": int(len(obs.contact_vertex_ids)),
        "contact_gap_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "contact_gap_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
    }


def summarize_key(rows: list[dict], key: str) -> dict:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            vals.append(value)
    return summarize(np.asarray(vals, dtype=float))


def apply_solution(
    annotations: dict,
    observations: list[HandObs],
    params_by_key: dict[tuple[int, str, int], tuple[np.ndarray, float]],
    args: argparse.Namespace,
) -> dict:
    out = copy.deepcopy(annotations)
    obs_by_key = {obs.key: obs for obs in observations}
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, str(hand.get("side")), hand_i)
            if key not in params_by_key or key not in obs_by_key:
                continue
            obs = obs_by_key[key]
            cam_t, log_scale = params_by_key[key]
            local_joints = scaled_local(obs.local_joints, obs, log_scale)
            local_vertices = scaled_local(obs.local_vertices, obs, log_scale)
            joints = local_joints + cam_t[None, :]
            vertices = local_vertices + cam_t[None, :]
            joints2d = project_points(joints, obs.intrinsics)
            reproj = np.linalg.norm(joints2d - obs.raw2d, axis=1)
            hand["cam_t"] = cam_t.astype(float).tolist()
            hand["joints3d_camera"] = local_joints.astype(float).tolist()
            hand["vertices_camera"] = local_vertices.astype(float).tolist()
            hand["joints3d_source_camera_m"] = joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = vertices.astype(float).tolist()
            hand["joints2d"] = joints2d.astype(float).tolist()
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(reproj)),
                "p95": float(np.percentile(reproj, 95.0)),
            }
            hand["joints3d_world_m"] = source_to_world(joints, T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(vertices, T_world_camera).astype(float).tolist()
            hand["v3_similarity_contact_refit"] = {
                "local_scale": float(np.exp(log_scale)),
                "translation_delta_m": (cam_t - obs.base_cam_t).astype(float).tolist(),
                "geometry_source": "fused_wilor_local_geometry_scaled_about_wrist",
            }
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_similarity_contact_refit"
            hand["world_coordinate_status"] = "v3_similarity_contact_refit_source_camera_geometry_transformed_by_existing_camera_pose"
    return out


def bounds(observations: list[HandObs], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for obs in observations:
        lower.extend((obs.base_cam_t - args.max_translation_m).tolist())
        upper.extend((obs.base_cam_t + args.max_translation_m).tolist())
        lower.append(float(np.log(args.min_local_scale)))
        upper.append(float(np.log(args.max_local_scale)))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def run(args: argparse.Namespace) -> dict:
    annotations, observations, skipped = build_observations(args)
    x0 = variables0(observations, args)
    lo, hi = bounds(observations, args)
    x0 = np.clip(x0, lo, hi)
    initial_rows = [row_metric(obs, obs.base_cam_t, 0.0) for obs in observations]
    result = least_squares(
        residual,
        x0,
        args=(observations, args),
        bounds=(lo, hi),
        loss="soft_l1",
        max_nfev=args.max_nfev,
        x_scale="jac",
        verbose=0,
    )
    params_by_key = unpack(result.x, observations)
    final_rows = [row_metric(obs, *params_by_key[obs.key]) for obs in observations]
    output = apply_solution(annotations, observations, params_by_key, args)
    save_json(args.output_annotations, output)
    report = {
        "status": "diagnostic_similarity_contact_refit",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "observations": int(len(observations)),
        "variables": int(len(result.x)),
        "solver": {
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
        },
        "initial": {
            "joint_reprojection_px": summarize_key(initial_rows, "joint_reprojection_px_median"),
            "mano_minus_metric_depth_m": summarize_key(initial_rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(initial_rows, "contact_gap_median_m"),
            "hand_span_m": summarize_key(initial_rows, "hand_span_m"),
            "local_scale": summarize_key(initial_rows, "local_scale"),
        },
        "final": {
            "joint_reprojection_px": summarize_key(final_rows, "joint_reprojection_px_median"),
            "mano_minus_metric_depth_m": summarize_key(final_rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(final_rows, "contact_gap_median_m"),
            "hand_span_m": summarize_key(final_rows, "hand_span_m"),
            "local_scale": summarize_key(final_rows, "local_scale"),
            "translation_shift_norm_m": summarize_key(final_rows, "translation_shift_norm_m"),
        },
        "thresholds": {
            "sigma_reprojection_px": float(args.sigma_reprojection_px),
            "sigma_metric_depth_m": float(args.sigma_metric_depth_m),
            "sigma_contact_depth_m": float(args.sigma_contact_depth_m),
            "min_span_m": float(args.min_span_m),
            "max_span_m": float(args.max_span_m),
            "min_local_scale": float(args.min_local_scale),
            "max_local_scale": float(args.max_local_scale),
        },
        "rows_preview_initial": initial_rows[:120],
        "rows_preview_final": final_rows[:120],
        "skipped_preview": skipped[:120],
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
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--good-joint-reprojection-px", type=float, default=20.0)
    parser.add_argument("--min-depth-joints", type=int, default=8)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--max-contact-vertices", type=int, default=180)
    parser.add_argument("--sigma-reprojection-px", type=float, default=25.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-contact-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-translation-prior-m", type=float, default=0.080)
    parser.add_argument("--sigma-temporal-translation-m", type=float, default=0.040)
    parser.add_argument("--sigma-temporal-log-scale", type=float, default=0.25)
    parser.add_argument("--sigma-log-scale", type=float, default=0.70)
    parser.add_argument("--sigma-span-m", type=float, default=0.020)
    parser.add_argument("--min-span-m", type=float, default=0.110)
    parser.add_argument("--max-span-m", type=float, default=0.210)
    parser.add_argument("--min-local-scale", type=float, default=0.50)
    parser.add_argument("--max-local-scale", type=float, default=4.00)
    parser.add_argument("--max-translation-m", type=float, default=0.35)
    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--clip-residual", type=float, default=50.0)
    parser.add_argument("--max-nfev", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
