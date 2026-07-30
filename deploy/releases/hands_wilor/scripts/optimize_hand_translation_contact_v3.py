#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from diagnose_hand_contact_reliability_v3 import camera_points_from_hand
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_contact_depth_scale_v3 import summarize
from optimize_object_factor_graph_v3 import localize_path, mask_distance_map, resize_bool_mask


TIP_IDS = np.asarray([4, 8, 12, 16, 20], dtype=int)
OBS_RESIDUAL_COUNT = 42 + 21 + 3 + 2 + 1


@dataclass(frozen=True)
class HandObs:
    key: tuple[int, str, int]
    frame_idx: int
    side: str
    frame_order: int
    hand_index: int
    score: float
    base_cam_t: np.ndarray
    local_joints: np.ndarray
    local_vertices: np.ndarray
    raw2d: np.ndarray
    intrinsics: np.ndarray
    metric_depth: np.ndarray
    depth_valid: np.ndarray
    stable_depth: np.ndarray
    contact_vertex_ids: np.ndarray
    object_depth_m: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def source_to_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def mesh_vertices_by_frame(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    frame_idx = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(int)
    vertices = np.asarray(blob["vertices"], dtype=float)
    return {
        int(frame): vertices[int(offsets[i]) : int(offsets[i + 1])]
        for i, frame in enumerate(frame_idx)
    }


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def depth_patch_iqr_ratio(depth: np.ndarray, xy: np.ndarray, radius: int) -> float:
    x = int(np.clip(round(float(xy[0])), 0, depth.shape[1] - 1))
    y = int(np.clip(round(float(xy[1])), 0, depth.shape[0] - 1))
    patch = depth[max(0, y - radius) : min(depth.shape[0], y + radius + 1), max(0, x - radius) : min(depth.shape[1], x + radius + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0.0)]
    if len(vals) == 0:
        return float("nan")
    med = float(np.median(vals))
    return float((np.percentile(vals, 75.0) - np.percentile(vals, 25.0)) / max(1e-6, med))


def hand_span(joints_source: np.ndarray) -> float:
    tips = joints_source[TIP_IDS]
    return float(np.max(np.linalg.norm(tips[:, None, :] - tips[None, :, :], axis=2)))


def object_camera_depth(vertices_world: np.ndarray, T_world_camera: np.ndarray) -> float:
    homog = np.c_[vertices_world, np.ones(len(vertices_world), dtype=float)]
    camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if len(z) == 0:
        raise RuntimeError("object mesh has no positive source-camera depth")
    return float(np.median(z))


def contact_vertex_ids(
    vertices_source: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    source_size: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if np.any(vertices_source[:, 2] <= 0.0):
        return np.empty(0, dtype=int)
    depth_mask = resize_mask_to_depth(mask, depth)
    dist = mask_distance_map(depth_mask)
    depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    uv = project_points(vertices_source, intrinsics)
    xy = uv * depth_scale[None, :]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(vertices_source).all(axis=1) & (vertices_source[:, 2] > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    near = np.flatnonzero(valid & (dist[y, x] <= args.contact_distance_px))
    if len(near) <= args.max_contact_vertices:
        return near.astype(int)
    z = vertices_source[near, 2]
    order = np.argsort(np.abs(z - np.median(z)))[: args.max_contact_vertices]
    return near[order].astype(int)


def build_observations(args: argparse.Namespace) -> tuple[dict, list[HandObs], list[dict]]:
    annotations = load_json(args.annotations)
    depth_blob = np.load(args.metric_depth_npz)
    depth_frame_idx = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_frame_idx)) != len(depth_frame_idx):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_frame_idx)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    object_vertices = mesh_vertices_by_frame(args.object_mesh_npz)

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
            T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            obj_depth = object_camera_depth(object_vertices[frame_idx], T_world_camera)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            frame_order += 1
            continue
        depth_scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
        for hand_i, hand in enumerate(frame.get("hands", [])):
            score = float(hand.get("detector_score", np.nan))
            if not hand.get("measurement_available", False) or not np.isfinite(score) or score < args.min_detector_score:
                continue
            try:
                local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                local_vertices = np.asarray(hand["vertices_camera"], dtype=float)
                cam_t = np.asarray(hand["cam_t"], dtype=float)
                raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
                intr = np.asarray(hand["source_intrinsics"], dtype=float)
                if local_joints.shape != (21, 3) or local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid local MANO geometry")
                if cam_t.shape != (3,) or raw2d.shape != (21, 2) or intr.shape != (4,):
                    raise RuntimeError("invalid hand observation fields")
                base_joints = local_joints + cam_t[None, :]
                base_vertices = local_vertices + cam_t[None, :]
                projected = project_points(base_joints, intr)
                reproj = np.linalg.norm(projected - raw2d, axis=1)
                samples = sample_depth(depth, raw2d, source_size)
                valid_depth = np.isfinite(samples) & (samples > 0.0) & (reproj <= args.good_joint_reprojection_px)
                patch_ratios = np.asarray(
                    [depth_patch_iqr_ratio(depth, xy * depth_scale, args.patch_radius) for xy in raw2d],
                    dtype=float,
                )
                stable = np.isfinite(patch_ratios) & (patch_ratios <= args.max_depth_iqr_ratio)
                contact_ids = contact_vertex_ids(base_vertices, intr, mask, depth, source_size, args)
                observations.append(
                    HandObs(
                        key=(frame_idx, str(hand.get("side")), hand_i),
                        frame_idx=frame_idx,
                        side=str(hand.get("side")),
                        frame_order=frame_order,
                        hand_index=hand_i,
                        score=score,
                        base_cam_t=cam_t,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        raw2d=raw2d,
                        intrinsics=intr,
                        metric_depth=samples,
                        depth_valid=valid_depth,
                        stable_depth=stable,
                        contact_vertex_ids=contact_ids,
                        object_depth_m=obj_depth,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
        frame_order += 1
    if not observations:
        raise RuntimeError("no measured hands for translation refit")
    return annotations, observations, skipped


def variables0(observations: list[HandObs]) -> np.ndarray:
    return np.concatenate([obs.base_cam_t for obs in observations]).astype(float)


def unpack(params: np.ndarray, observations: list[HandObs]) -> dict[tuple[int, str, int], np.ndarray]:
    arr = np.asarray(params, dtype=float).reshape(len(observations), 3)
    return {obs.key: arr[i] for i, obs in enumerate(observations)}


def corrected_joints(obs: HandObs, cam_t: np.ndarray) -> np.ndarray:
    pts = obs.local_joints + cam_t[None, :]
    if np.any(pts[:, 2] <= 0.0):
        raise RuntimeError("corrected joints have non-positive depth")
    return pts


def corrected_vertices(obs: HandObs, cam_t: np.ndarray) -> np.ndarray:
    pts = obs.local_vertices + cam_t[None, :]
    if np.any(pts[:, 2] <= 0.0):
        raise RuntimeError("corrected vertices have non-positive depth")
    return pts


def residual(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    cam_ts = np.asarray(params, dtype=float).reshape(len(observations), 3)
    out: list[float] = []
    for i, obs in enumerate(observations):
        cam_t = cam_ts[i]
        try:
            joints = corrected_joints(obs, cam_t)
            vertices = corrected_vertices(obs, cam_t)
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
        shift = cam_t - obs.base_cam_t
        out.extend(shift / args.sigma_translation_prior_m)
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
        by_side.setdefault(obs.side, []).append((obs.frame_order, cam_ts[i]))
    for items in by_side.values():
        items.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            out.extend((b - a) / args.sigma_temporal_translation_m)
    return np.asarray(out, dtype=float)


def row_metric(obs: HandObs, cam_t: np.ndarray) -> dict:
    joints = corrected_joints(obs, cam_t)
    vertices = corrected_vertices(obs, cam_t)
    projected = project_points(joints, obs.intrinsics)
    reproj = np.linalg.norm(projected - obs.raw2d, axis=1)
    depth_valid = obs.depth_valid & obs.stable_depth
    depth_err = joints[depth_valid, 2] - obs.metric_depth[depth_valid]
    contact_gap = np.asarray([], dtype=float)
    if len(obs.contact_vertex_ids):
        contact_gap = vertices[obs.contact_vertex_ids, 2] - obs.object_depth_m
    return {
        "frame_idx": int(obs.frame_idx),
        "side": obs.side,
        "detector_score": float(obs.score),
        "translation_shift_m": (cam_t - obs.base_cam_t).astype(float).tolist(),
        "translation_shift_norm_m": float(np.linalg.norm(cam_t - obs.base_cam_t)),
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


def apply_solution(annotations: dict, observations: list[HandObs], cam_t_by_key: dict[tuple[int, str, int], np.ndarray], args: argparse.Namespace) -> dict:
    out = copy.deepcopy(annotations)
    obs_by_key = {obs.key: obs for obs in observations}
    for frame in out["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (frame_idx, str(hand.get("side")), hand_i)
            if key not in cam_t_by_key or key not in obs_by_key:
                continue
            obs = obs_by_key[key]
            cam_t = cam_t_by_key[key]
            joints = corrected_joints(obs, cam_t)
            vertices = corrected_vertices(obs, cam_t)
            joints2d = project_points(joints, obs.intrinsics)
            reproj = np.linalg.norm(joints2d - obs.raw2d, axis=1)
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
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_translation_contact_refit"
            hand["world_coordinate_status"] = "v3_translation_contact_refit_source_camera_mano_transformed_by_existing_camera_pose"
    return out


def run(args: argparse.Namespace) -> dict:
    annotations, observations, skipped = build_observations(args)
    x0 = variables0(observations)
    initial_rows = [row_metric(obs, obs.base_cam_t) for obs in observations]
    result = least_squares(
        residual,
        x0,
        args=(observations, args),
        loss="soft_l1",
        max_nfev=args.max_nfev,
        x_scale="jac",
        verbose=0,
    )
    cam_t_by_key = unpack(result.x, observations)
    final_rows = [row_metric(obs, cam_t_by_key[obs.key]) for obs in observations]
    output = apply_solution(annotations, observations, cam_t_by_key, args)
    save_json(args.output_annotations, output)
    report = {
        "status": "diagnostic_translation_contact_refit",
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
        },
        "final": {
            "joint_reprojection_px": summarize_key(final_rows, "joint_reprojection_px_median"),
            "mano_minus_metric_depth_m": summarize_key(final_rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(final_rows, "contact_gap_median_m"),
            "hand_span_m": summarize_key(final_rows, "hand_span_m"),
            "translation_shift_norm_m": summarize_key(final_rows, "translation_shift_norm_m"),
        },
        "thresholds": {
            "sigma_reprojection_px": float(args.sigma_reprojection_px),
            "sigma_metric_depth_m": float(args.sigma_metric_depth_m),
            "sigma_contact_depth_m": float(args.sigma_contact_depth_m),
            "min_span_m": float(args.min_span_m),
            "max_span_m": float(args.max_span_m),
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
    parser.add_argument("--sigma-span-m", type=float, default=0.020)
    parser.add_argument("--min-span-m", type=float, default=0.110)
    parser.add_argument("--max-span-m", type=float, default=0.210)
    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--clip-residual", type=float, default=50.0)
    parser.add_argument("--max-nfev", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
