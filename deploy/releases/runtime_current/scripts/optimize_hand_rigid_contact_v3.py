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
from scipy.spatial.transform import Rotation

from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m, hand_tip_spread_m
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth
from optimize_contact_depth_scale_v3 import summarize
from optimize_hand_translation_contact_v3 import (
    contact_vertex_ids,
    depth_patch_iqr_ratio,
    mesh_vertices_by_frame,
    object_camera_depth,
    resize_mask_to_depth,
    source_to_world,
)
from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask


OBS_RESIDUAL_COUNT = 42 + 21 + 6 + 2 + 1


@dataclass(frozen=True)
class HandObs:
    key: tuple[int, str, int]
    frame_idx: int
    side: str
    hand_index: int
    frame_order: int
    score: float
    center: np.ndarray
    local_joints: np.ndarray
    local_vertices: np.ndarray
    base_cam_t: np.ndarray
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
                source_joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                source_vertices = np.asarray(hand["vertices_source_camera_m"], dtype=float)
                raw2d = np.asarray(hand["joints2d_raw"], dtype=float)
                intr = np.asarray(hand["source_intrinsics"], dtype=float)
                if source_joints.shape != (21, 3) or source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
                    raise RuntimeError("invalid source-camera hand geometry")
                if raw2d.shape != (21, 2) or intr.shape != (4,):
                    raise RuntimeError("invalid 2D/intrinsics fields")
                center = source_joints[0].copy()
                local_joints = source_joints - center[None, :]
                local_vertices = source_vertices - center[None, :]
                projected = project_points(source_joints, intr)
                reproj = np.linalg.norm(projected - raw2d, axis=1)
                samples = sample_depth(depth, raw2d, source_size)
                valid_depth = np.isfinite(samples) & (samples > 0.0) & (reproj <= args.good_joint_reprojection_px)
                patch_ratios = np.asarray(
                    [depth_patch_iqr_ratio(depth, xy * depth_scale, args.patch_radius) for xy in raw2d],
                    dtype=float,
                )
                stable = np.isfinite(patch_ratios) & (patch_ratios <= args.max_depth_iqr_ratio)
                contact_ids = contact_vertex_ids(source_vertices, intr, mask, depth, source_size, args)
                observations.append(
                    HandObs(
                        key=(frame_idx, str(hand.get("side")), hand_i),
                        frame_idx=frame_idx,
                        side=str(hand.get("side")),
                        hand_index=hand_i,
                        frame_order=frame_order,
                        score=score,
                        center=center,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        base_cam_t=np.zeros(3, dtype=float),
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
        raise RuntimeError("no measured hands for rigid hand refit")
    return annotations, observations, skipped


def transform_points(obs: HandObs, rotvec: np.ndarray, trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    pts = obs.center[None, :] + points @ rotation.T + trans[None, :]
    if np.any(pts[:, 2] <= 0.0):
        raise RuntimeError("hand points have non-positive depth")
    return pts


def variables0(observations: list[HandObs]) -> np.ndarray:
    return np.zeros((len(observations), 6), dtype=float).reshape(-1)


def unpack(params: np.ndarray, observations: list[HandObs]) -> dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(params, dtype=float).reshape(len(observations), 6)
    return {obs.key: (arr[i, :3], arr[i, 3:]) for i, obs in enumerate(observations)}


def residual(params: np.ndarray, observations: list[HandObs], args: argparse.Namespace) -> np.ndarray:
    x = np.asarray(params, dtype=float).reshape(len(observations), 6)
    out: list[float] = []
    for i, obs in enumerate(observations):
        rotvec = x[i, :3]
        trans = x[i, 3:]
        try:
            joints = transform_points(obs, rotvec, trans, obs.local_joints)
            vertices = transform_points(obs, rotvec, trans, obs.local_vertices)
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
        out.extend(rotvec / args.sigma_rotation_rad)
        out.extend(trans / args.sigma_translation_m)
        bone_scale = hand_bone_scale_m(joints)
        out.append(max(0.0, args.min_bone_scale_m - bone_scale) / args.sigma_bone_scale_m)
        out.append(max(0.0, bone_scale - args.max_bone_scale_m) / args.sigma_bone_scale_m)
        if len(obs.contact_vertex_ids) >= args.min_near_vertices:
            contact_z = vertices[obs.contact_vertex_ids, 2] - obs.object_depth_m
            out.append(float(np.clip(np.median(contact_z) / args.sigma_contact_depth_m, -args.clip_residual, args.clip_residual)))
        else:
            out.append(0.0)

    by_side: dict[str, list[tuple[int, np.ndarray]]] = {}
    for i, obs in enumerate(observations):
        by_side.setdefault(obs.side, []).append((obs.frame_order, x[i]))
    for items in by_side.values():
        items.sort(key=lambda item: item[0])
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            out.extend((b[:3] - a[:3]) / args.sigma_temporal_rotation_rad)
            out.extend((b[3:] - a[3:]) / args.sigma_temporal_translation_m)
    return np.asarray(out, dtype=float)


def row_metric(obs: HandObs, rotvec: np.ndarray, trans: np.ndarray) -> dict:
    joints = transform_points(obs, rotvec, trans, obs.local_joints)
    vertices = transform_points(obs, rotvec, trans, obs.local_vertices)
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
        "hand_index": int(obs.hand_index),
        "detector_score": float(obs.score),
        "rotation_norm_rad": float(np.linalg.norm(rotvec)),
        "translation_norm_m": float(np.linalg.norm(trans)),
        "translation_z_m": float(trans[2]),
        "joint_reprojection_px_median": float(np.median(reproj)),
        "joint_reprojection_px_p95": float(np.percentile(reproj, 95.0)),
        "depth_joints": int(np.count_nonzero(depth_valid)),
        "mano_minus_metric_depth_median_m": None if len(depth_err) == 0 else float(np.median(depth_err)),
        "mano_minus_metric_depth_p95_abs_m": None if len(depth_err) == 0 else float(np.percentile(np.abs(depth_err), 95.0)),
        "hand_bone_scale_m": hand_bone_scale_m(joints),
        "hand_tip_spread_m": hand_tip_spread_m(joints),
        "near_mask_vertices": int(len(obs.contact_vertex_ids)),
        "contact_gap_median_m": None if len(contact_gap) == 0 else float(np.median(contact_gap)),
        "contact_gap_p95_abs_m": None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0)),
    }


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


def bounds(observations: list[HandObs], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lo = []
    hi = []
    for _ in observations:
        lo.extend([-args.max_rotation_rad] * 3)
        hi.extend([args.max_rotation_rad] * 3)
        lo.extend([-args.max_translation_m] * 3)
        hi.extend([args.max_translation_m] * 3)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def apply_solution(
    annotations: dict,
    observations: list[HandObs],
    params_by_key: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]],
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
            if key not in params_by_key:
                continue
            obs = obs_by_key[key]
            rotvec, trans = params_by_key[key]
            joints = transform_points(obs, rotvec, trans, obs.local_joints)
            vertices = transform_points(obs, rotvec, trans, obs.local_vertices)
            joints2d = project_points(joints, obs.intrinsics)
            reproj = np.linalg.norm(joints2d - obs.raw2d, axis=1)
            cam_t = joints[0].copy()
            hand["cam_t"] = cam_t.astype(float).tolist()
            hand["joints3d_camera"] = (joints - cam_t[None, :]).astype(float).tolist()
            hand["vertices_camera"] = (vertices - cam_t[None, :]).astype(float).tolist()
            hand["joints3d_source_camera_m"] = joints.astype(float).tolist()
            hand["vertices_source_camera_m"] = vertices.astype(float).tolist()
            hand["joints2d"] = joints2d.astype(float).tolist()
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(reproj)),
                "p95": float(np.percentile(reproj, 95.0)),
            }
            hand["joints3d_world_m"] = source_to_world(joints, T_world_camera).astype(float).tolist()
            hand["vertices_world_m"] = source_to_world(vertices, T_world_camera).astype(float).tolist()
            hand["v3_rigid_contact_refit"] = {
                "rotation_delta_axis_angle": rotvec.astype(float).tolist(),
                "translation_delta_m": trans.astype(float).tolist(),
                "geometry_source": str(hand.get("backend", "unknown")),
            }
            hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_rigid_contact_refit"
            hand["world_coordinate_status"] = "v3_rigid_contact_refit_source_camera_geometry_transformed_by_existing_camera_pose"
    return out


def run(args: argparse.Namespace) -> dict:
    annotations, observations, skipped = build_observations(args)
    x0 = variables0(observations)
    lo, hi = bounds(observations, args)
    initial_rows = [row_metric(obs, np.zeros(3), np.zeros(3)) for obs in observations]
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
        "status": "diagnostic_rigid_contact_refit",
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
            "hand_bone_scale_m": summarize_key(initial_rows, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(initial_rows, "hand_tip_spread_m"),
        },
        "final": {
            "joint_reprojection_px": summarize_key(final_rows, "joint_reprojection_px_median"),
            "mano_minus_metric_depth_m": summarize_key(final_rows, "mano_minus_metric_depth_median_m"),
            "contact_gap_m": summarize_key(final_rows, "contact_gap_median_m"),
            "hand_bone_scale_m": summarize_key(final_rows, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(final_rows, "hand_tip_spread_m"),
            "rotation_norm_rad": summarize_key(final_rows, "rotation_norm_rad"),
            "translation_norm_m": summarize_key(final_rows, "translation_norm_m"),
            "translation_z_m": summarize_key(final_rows, "translation_z_m"),
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
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.040)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--max-contact-vertices", type=int, default=180)
    parser.add_argument("--sigma-reprojection-px", type=float, default=25.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-contact-depth-m", type=float, default=0.030)
    parser.add_argument("--sigma-rotation-rad", type=float, default=0.35)
    parser.add_argument("--sigma-translation-m", type=float, default=0.100)
    parser.add_argument("--sigma-temporal-rotation-rad", type=float, default=0.20)
    parser.add_argument("--sigma-temporal-translation-m", type=float, default=0.060)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.020)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--max-rotation-rad", type=float, default=0.80)
    parser.add_argument("--max-translation-m", type=float, default=0.35)
    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--clip-residual", type=float, default=50.0)
    parser.add_argument("--max-nfev", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
