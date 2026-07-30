#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame
from optimize_object_factor_graph_v3 import localize_path, resize_bool_mask


@dataclass(frozen=True)
class MaskObs:
    frame_idx: int
    hand_index: int
    side: str
    local_joints: np.ndarray
    local_vertices: np.ndarray
    joints2d_prior: np.ndarray
    source_joints0: np.ndarray
    source_vertices0: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    mask_distance: np.ndarray
    mask_depth_median: float
    mask_depth_iqr_m: float
    mask_center_xy: np.ndarray
    silhouette_vertices: np.ndarray
    depth_vertices: np.ndarray
    init_params: np.ndarray
    base_bone_m: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def monkeypatch_chumpy_numpy() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def load_mano_faces(path: Path) -> np.ndarray:
    monkeypatch_chumpy_numpy()
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    faces = np.asarray(data.get("f"), dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0:
        raise RuntimeError(f"invalid MANO face topology in {path}")
    return faces


def frames_by_index(annotations: dict) -> dict[int, dict]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations must contain frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def local_vertex_key(hand: dict) -> str:
    for key in ("vertices_camera", "vertices_camera_sample"):
        if key in hand:
            return key
    raise RuntimeError("hand has no local MANO vertices")


def source_vertex_key(hand: dict) -> str:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        if key in hand:
            return key
    raise RuntimeError("hand has no source-camera MANO vertices")


def source_to_world(points: np.ndarray, frame: dict) -> np.ndarray:
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T @ homog.T).T[:, :3]


def rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def transform_points(local: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(params[:3])
    t = params[3:6]
    scale = math.exp(float(params[6]))
    return scale * (local @ rot.T) + t[None, :]


def mask_for_frame(track: dict, frame_idx: int, source_size: tuple[int, int], args: argparse.Namespace) -> np.ndarray | None:
    row = track.get(str(frame_idx))
    if not isinstance(row, dict) or not row.get("visible") or not row.get("mask_path"):
        return None
    mask_path = localize_path(str(row["mask_path"]), args.remote_output_root, args.local_output_root)
    mask = resize_bool_mask(mask_path, (960, 540))
    return cv2.resize(mask.astype(np.uint8), source_size, interpolation=cv2.INTER_NEAREST) > 0


def mask_depth_stats(depth: np.ndarray, mask: np.ndarray, source_size: tuple[int, int]) -> tuple[float, float, np.ndarray]:
    mask_depth = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    values = depth[mask_depth]
    values = values[np.isfinite(values) & (values > 0.05)]
    if len(values) < 50:
        raise RuntimeError("hand mask has too few metric depth pixels")
    q25, q50, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return float(q50), float(q75 - q25), mask_depth


def initial_params(local_joints: np.ndarray, mask_center_xy: np.ndarray, mask_depth_median: float, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(float)
    z = float(mask_depth_median)
    if not np.isfinite(z) or z <= 0.05:
        raise RuntimeError("invalid mask depth for MANO initialization")
    center_camera = np.asarray([(mask_center_xy[0] - cx) * z / fx, (mask_center_xy[1] - cy) * z / fy, z], dtype=float)
    local_center = np.median(local_joints, axis=0)
    return np.r_[np.zeros(3, dtype=float), center_camera - local_center, 0.0]


def distance_map(mask: np.ndarray) -> np.ndarray:
    inv = (~mask).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)


def mask_center(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("empty hand mask")
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=float)


def silhouette_vertices(vertices: np.ndarray, faces: np.ndarray, max_count: int) -> np.ndarray:
    used = np.unique(faces.reshape(-1))
    if len(used) == 0:
        raise RuntimeError("MANO faces reference no vertices")
    if len(used) <= max_count:
        return used.astype(np.int32)
    center = np.mean(vertices[used], axis=0)
    dist = np.linalg.norm(vertices[used] - center[None, :], axis=1)
    order = np.argsort(dist)[-max_count:]
    return used[order].astype(np.int32)


def build_obs(args: argparse.Namespace, faces: np.ndarray) -> tuple[dict, list[dict], list[MaskObs], list[dict]]:
    annotations = load_json(args.hand_annotations)
    output = copy.deepcopy(annotations)
    hand_frames = frames_by_index(annotations)
    output_frames = frames_by_index(output)
    track = load_json(args.mask_track)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    frame_to_depth = {int(frame_idx): i for i, frame_idx in enumerate(depth_indices)}
    obs: list[MaskObs] = []
    skipped: list[dict] = []
    source_size = (int(args.source_width), int(args.source_height))
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))):
        frame = hand_frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_hand_annotation_frame"})
            continue
        if frame_idx not in frame_to_depth:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_depth_frame"})
            continue
        mask = mask_for_frame(track, frame_idx, source_size, args)
        if mask is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_visible_hand_mask"})
            continue
        depth = depth_frame(depths, frame_to_depth, frame_idx)
        try:
            mask_depth, mask_iqr, _ = mask_depth_stats(depth, mask, source_size)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        hands = frame.get("hands", [])
        for hand_index, hand in enumerate(hands):
            try:
                side = str(hand.get("side", "unknown"))
                if args.side != "any" and side != args.side:
                    continue
                local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                local_vertices = np.asarray(hand[local_vertex_key(hand)], dtype=float)
                source_joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                source_vertices = np.asarray(hand[source_vertex_key(hand)], dtype=float)
                joints2d = np.asarray(hand.get("joints2d", []), dtype=float)
                intrinsics = np.asarray(hand.get("source_intrinsics", []), dtype=float)
                if local_joints.shape != (21, 3) or source_joints.shape != (21, 3):
                    raise RuntimeError("invalid MANO joints")
                if local_vertices.ndim != 2 or source_vertices.shape != local_vertices.shape or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid MANO vertices")
                if joints2d.shape != (21, 2) or intrinsics.shape != (4,):
                    raise RuntimeError("invalid projection fields")
                base_bone = hand_bone_scale_m(local_joints)
                if not np.isfinite(base_bone) or base_bone <= 0.0:
                    raise RuntimeError("invalid local hand bone scale")
                obs.append(
                    MaskObs(
                        frame_idx=frame_idx,
                        hand_index=hand_index,
                        side=side,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        joints2d_prior=joints2d,
                        source_joints0=source_joints,
                        source_vertices0=source_vertices,
                        intrinsics=intrinsics,
                        depth=depth,
                        mask=mask,
                        mask_distance=distance_map(mask),
                        mask_depth_median=mask_depth,
                        mask_depth_iqr_m=mask_iqr,
                        mask_center_xy=mask_center(mask),
                        silhouette_vertices=silhouette_vertices(local_vertices, faces, int(args.max_silhouette_vertices)),
                        depth_vertices=silhouette_vertices(local_vertices, faces, int(args.max_depth_vertices)),
                        init_params=initial_params(local_joints, mask_center(mask), mask_depth, intrinsics),
                        base_bone_m=base_bone,
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_index": hand_index, "reason": str(exc)})
    if len(obs) < int(args.min_observations):
        raise RuntimeError(f"insufficient mask MANO observations {len(obs)}; skipped={skipped[:40]}")
    return output, output["frames"], obs, skipped


def pack_initial(obs: list[MaskObs]) -> np.ndarray:
    return np.concatenate([o.init_params for o in obs]).astype(float)


def unpack(params: np.ndarray) -> np.ndarray:
    if len(params) % 7 != 0:
        raise RuntimeError("parameter vector is not 7N")
    return params.reshape((-1, 7))


def sample_mask_depth(depth: np.ndarray, points: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    scale = np.asarray([depth.shape[1] / source_size[0], depth.shape[0] / source_size[1]], dtype=float)
    xy = points * scale[None, :]
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return depth[y, x]


def fixed_depth_residual(values: np.ndarray, targets: np.ndarray, sigma: float, invalid_penalty: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    valid = np.isfinite(values) & np.isfinite(targets) & (targets > 0.05)
    out = np.full(values.shape, float(invalid_penalty), dtype=float)
    out[valid] = (values[valid] - targets[valid]) / float(sigma)
    return out


def residual(params: np.ndarray, obs: list[MaskObs], args: argparse.Namespace) -> np.ndarray:
    p = unpack(params)
    out: list[np.ndarray] = []
    centers = []
    source_size = (int(args.source_width), int(args.source_height))
    for i, o in enumerate(obs):
        joints = transform_points(o.local_joints, p[i])
        vertices = transform_points(o.local_vertices, p[i])
        min_depth_terms = np.r_[
            np.clip(float(args.min_depth_m) - joints[:, 2], 0.0, None),
            np.clip(float(args.min_depth_m) - vertices[o.depth_vertices, 2], 0.0, None),
        ]
        out.append(min_depth_terms / float(args.sigma_min_depth_m))
        proj_joints = project_points(joints, o.intrinsics)
        proj_vertices = project_points(vertices[o.silhouette_vertices], o.intrinsics)
        x = np.clip(np.rint(proj_vertices[:, 0]).astype(int), 0, o.mask_distance.shape[1] - 1)
        y = np.clip(np.rint(proj_vertices[:, 1]).astype(int), 0, o.mask_distance.shape[0] - 1)
        dist = o.mask_distance[y, x]
        out.append(np.clip(dist, 0.0, float(args.max_silhouette_distance_px)) / float(args.sigma_silhouette_px))
        joint_depth = sample_mask_depth(o.depth, proj_joints, source_size)
        out.append(fixed_depth_residual(joints[:, 2], joint_depth, float(args.sigma_joint_depth_m), float(args.invalid_depth_penalty)))
        proj_depth_vertices = project_points(vertices[o.depth_vertices], o.intrinsics)
        vertex_depth = sample_mask_depth(o.depth, proj_depth_vertices, source_size)
        out.append(
            fixed_depth_residual(
                vertices[o.depth_vertices, 2],
                vertex_depth,
                float(args.sigma_vertex_depth_m),
                float(args.invalid_depth_penalty),
            )
        )
        mask_depth_gap = np.median(joints[:, 2]) - o.mask_depth_median
        out.append(np.asarray([mask_depth_gap / float(args.sigma_mask_depth_m)]))
        if bool(args.use_joint_prior):
            out.append((proj_joints - o.joints2d_prior).reshape(-1) / float(args.sigma_joint_prior_px))
        scale = math.exp(float(p[i, 6]))
        bone = scale * o.base_bone_m
        out.append(np.asarray([(bone - float(args.hand_bone_scale_prior_m)) / float(args.sigma_bone_prior_m)]))
        out.append(np.asarray([np.clip(float(args.min_bone_scale_m) - bone, 0.0, None) / float(args.sigma_bone_bound_m)]))
        out.append(np.asarray([np.clip(bone - float(args.max_bone_scale_m), 0.0, None) / float(args.sigma_bone_bound_m)]))
        out.append(p[i, :3] / float(args.sigma_rotation_prior_rad))
        out.append(np.asarray([p[i, 6] / float(args.sigma_log_scale_prior)]))
        out.append((p[i, 3:6] - o.init_params[3:6]) / float(args.sigma_translation_init_m))
        centers.append(np.median(joints, axis=0))
    for a, b in zip(range(len(obs) - 1), range(1, len(obs))):
        if obs[b].frame_idx - obs[a].frame_idx > int(args.max_temporal_gap_frames):
            continue
        dt = max(1, obs[b].frame_idx - obs[a].frame_idx)
        out.append((centers[b] - centers[a]) / (float(args.sigma_center_step_m) * dt))
        out.append((p[b, :3] - p[a, :3]) / (float(args.sigma_rotation_step_rad) * dt))
        out.append(np.asarray([(p[b, 6] - p[a, 6]) / (float(args.sigma_log_scale_step) * dt)]))
    return np.concatenate([np.ravel(x).astype(float) for x in out])


def solve(obs: list[MaskObs], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    x0 = pack_initial(obs)
    lo = np.tile(
        np.asarray(
            [
                -float(args.max_rotation_rad),
                -float(args.max_rotation_rad),
                -float(args.max_rotation_rad),
                -2.0,
                -2.0,
                float(args.min_depth_m),
                math.log(float(args.min_scale)),
            ]
        ),
        len(obs),
    )
    hi = np.tile(
        np.asarray(
            [
                float(args.max_rotation_rad),
                float(args.max_rotation_rad),
                float(args.max_rotation_rad),
                2.0,
                2.0,
                float(args.max_depth_m),
                math.log(float(args.max_scale)),
            ]
        ),
        len(obs),
    )
    before = residual(x0, obs, args)
    result = least_squares(
        residual,
        x0,
        args=(obs, args),
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        verbose=0,
    )
    after = residual(result.x, obs, args)
    return result.x, {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
    }


def row_metrics(obs: list[MaskObs], params: np.ndarray, args: argparse.Namespace) -> list[dict]:
    p = unpack(params)
    rows = []
    source_size = (int(args.source_width), int(args.source_height))
    for i, o in enumerate(obs):
        joints0 = transform_points(o.local_joints, o.init_params)
        vertices0 = transform_points(o.local_vertices, o.init_params)
        joints = transform_points(o.local_joints, p[i])
        vertices = transform_points(o.local_vertices, p[i])
        proj = project_points(vertices[o.silhouette_vertices], o.intrinsics)
        x = np.clip(np.rint(proj[:, 0]).astype(int), 0, o.mask_distance.shape[1] - 1)
        y = np.clip(np.rint(proj[:, 1]).astype(int), 0, o.mask_distance.shape[0] - 1)
        dist = o.mask_distance[y, x]
        proj_joints = project_points(joints, o.intrinsics)
        joint_depth = sample_mask_depth(o.depth, proj_joints, source_size)
        valid_depth = np.isfinite(joint_depth) & (joint_depth > float(args.min_depth_m))
        joint_reproj_prior = np.linalg.norm(proj_joints - o.joints2d_prior, axis=1)
        scale = math.exp(float(p[i, 6]))
        rows.append(
            {
                "frame_idx": int(o.frame_idx),
                "hand_index": int(o.hand_index),
                "side": o.side,
                "mask_depth_median_m": float(o.mask_depth_median),
                "mask_depth_iqr_m": float(o.mask_depth_iqr_m),
                "silhouette_vertices": int(len(o.silhouette_vertices)),
                "silhouette_inside_fraction": float(np.mean(dist <= 0.5)),
                "silhouette_distance_median_px": float(np.median(dist)),
                "silhouette_distance_p95_px": float(np.percentile(dist, 95.0)),
                "joint_prior_reprojection_median_px": float(np.median(joint_reproj_prior)),
                "joint_prior_reprojection_p95_px": float(np.percentile(joint_reproj_prior, 95.0)),
                "mano_minus_mask_depth_median_m": float(np.median(joints[:, 2]) - o.mask_depth_median),
                "mano_joint_minus_depth_median_m": float(np.median(joints[valid_depth, 2] - joint_depth[valid_depth])) if np.any(valid_depth) else None,
                "hand_bone_m": float(scale * o.base_bone_m),
                "scale": float(scale),
                "rotation_norm_rad": float(np.linalg.norm(p[i, :3])),
                "center_before_m": np.median(joints0, axis=0).astype(float).tolist(),
                "center_after_m": np.median(joints, axis=0).astype(float).tolist(),
                "center_shift_from_init_m": float(np.linalg.norm(np.median(joints, axis=0) - np.median(joints0, axis=0))),
                "vertex_depth_before_median_m": float(np.median(vertices0[:, 2])),
                "vertex_depth_after_median_m": float(np.median(vertices[:, 2])),
            }
        )
    return rows


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


def apply_solution(output_frames: list[dict], obs: list[MaskObs], params: np.ndarray, rows: list[dict]) -> None:
    frame_map = {int(frame["frame_idx"]): frame for frame in output_frames}
    p = unpack(params)
    row_map = {(row["frame_idx"], row["hand_index"]): row for row in rows}
    for i, o in enumerate(obs):
        frame = frame_map[o.frame_idx]
        hand = frame["hands"][o.hand_index]
        source_joints = transform_points(o.local_joints, p[i])
        source_vertices = transform_points(o.local_vertices, p[i])
        projected = project_points(source_joints, o.intrinsics)
        row = row_map[(o.frame_idx, o.hand_index)]
        hand["joints3d_source_camera_m"] = source_joints.astype(float).tolist()
        hand[source_vertex_key(hand)] = source_vertices.astype(float).tolist()
        hand["joints2d"] = projected.astype(float).tolist()
        hand["source_intrinsics"] = o.intrinsics.astype(float).tolist()
        hand["joints3d_world_m"] = source_to_world(source_joints, frame).astype(float).tolist()
        hand["vertices_world_m"] = source_to_world(source_vertices, frame).astype(float).tolist()
        hand["measurement_available"] = True
        hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_mask_depth_similarity_refit"
        hand["world_coordinate_status"] = "v3_mask_depth_similarity_refit_world_from_camera"
        hand["v3_mask_depth_refit"] = {
            "status": "applied",
            "silhouette_inside_fraction": row["silhouette_inside_fraction"],
            "silhouette_distance_p95_px": row["silhouette_distance_p95_px"],
            "mano_minus_mask_depth_median_m": row["mano_minus_mask_depth_median_m"],
            "hand_bone_m": row["hand_bone_m"],
            "scale": row["scale"],
            "rotation_norm_rad": row["rotation_norm_rad"],
            "center_shift_from_init_m": row["center_shift_from_init_m"],
        }


def run(args: argparse.Namespace) -> dict:
    faces = load_mano_faces(args.mano_model)
    output, output_frames, obs, skipped = build_obs(args, faces)
    params, solve_info = solve(obs, args)
    rows = row_metrics(obs, params, args)
    apply_solution(output_frames, obs, params, rows)
    save_json(args.output_annotations, output)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "hand_annotations": str(args.hand_annotations),
        "mask_track": str(args.mask_track),
        "metric_depth_npz": str(args.metric_depth_npz),
        "mano_model": str(args.mano_model),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "side": args.side,
        "observations": int(len(obs)),
        "skipped_count": int(len(skipped)),
        "solve": solve_info,
        "summary": {
            "silhouette_inside_fraction": summarize_key(rows, "silhouette_inside_fraction"),
            "silhouette_distance_p95_px": summarize_key(rows, "silhouette_distance_p95_px"),
            "mano_minus_mask_depth_median_m": summarize_key(rows, "mano_minus_mask_depth_median_m"),
            "mano_joint_minus_depth_median_m": summarize_key(rows, "mano_joint_minus_depth_median_m"),
            "joint_prior_reprojection_median_px": summarize_key(rows, "joint_prior_reprojection_median_px"),
            "hand_bone_m": summarize_key(rows, "hand_bone_m"),
            "center_shift_from_init_m": summarize_key(rows, "center_shift_from_init_m"),
            "scale": summarize_key(rows, "scale"),
            "rotation_norm_rad": summarize_key(rows, "rotation_norm_rad"),
        },
        "thresholds": {
            "min_bone_scale_m": float(args.min_bone_scale_m),
            "max_bone_scale_m": float(args.max_bone_scale_m),
            "sigma_silhouette_px": float(args.sigma_silhouette_px),
            "sigma_mask_depth_m": float(args.sigma_mask_depth_m),
        },
        "rows_preview": rows[:160],
        "skipped_preview": skipped[:160],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-annotations", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--side", choices=["left", "right", "any"], default="left")
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--remote-output-root", type=Path, default=Path("/dev/shm/ego_annotation_keyboard_hand_masks/outputs"))
    parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/representative_keyboard/v3_keyboard_hand_sam2_visual_tracks_60_75"),
    )
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-silhouette-vertices", type=int, default=256)
    parser.add_argument("--max-depth-vertices", type=int, default=128)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.165)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.12)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.24)
    parser.add_argument("--min-scale", type=float, default=0.65)
    parser.add_argument("--max-scale", type=float, default=1.45)
    parser.add_argument("--max-rotation-rad", type=float, default=1.2)
    parser.add_argument("--max-silhouette-distance-px", type=float, default=80.0)
    parser.add_argument("--sigma-silhouette-px", type=float, default=6.0)
    parser.add_argument("--sigma-joint-depth-m", type=float, default=0.040)
    parser.add_argument("--sigma-vertex-depth-m", type=float, default=0.050)
    parser.add_argument("--sigma-mask-depth-m", type=float, default=0.050)
    parser.add_argument("--use-joint-prior", action="store_true")
    parser.add_argument("--sigma-joint-prior-px", type=float, default=40.0)
    parser.add_argument("--sigma-bone-prior-m", type=float, default=0.030)
    parser.add_argument("--sigma-bone-bound-m", type=float, default=0.010)
    parser.add_argument("--sigma-rotation-prior-rad", type=float, default=0.70)
    parser.add_argument("--sigma-log-scale-prior", type=float, default=0.30)
    parser.add_argument("--sigma-translation-init-m", type=float, default=0.25)
    parser.add_argument("--sigma-center-step-m", type=float, default=0.080)
    parser.add_argument("--sigma-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--sigma-log-scale-step", type=float, default=0.080)
    parser.add_argument("--max-temporal-gap-frames", type=int, default=3)
    parser.add_argument("--sigma-min-depth-m", type=float, default=0.010)
    parser.add_argument("--invalid-penalty", type=float, default=1e3)
    parser.add_argument("--invalid-depth-penalty", type=float, default=12.0)
    parser.add_argument("--max-nfev", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
