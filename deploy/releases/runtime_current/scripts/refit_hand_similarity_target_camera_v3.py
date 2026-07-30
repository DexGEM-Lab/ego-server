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
from scipy.spatial.transform import Rotation

from compare_hand_streams_scale055_v3 import load_depth_archive, load_frame_window
from diagnose_contact_depth_conflict_v3 import summarize
from diagnose_hand_contact_reliability_v3 import depth_patch_iqr_ratio, hand_bone_scale_m, hand_tip_spread_m
from diagnose_hand_reprojection_depth_v3 import project_points
from diagnose_metric_depth_alignment_v3 import depth_frame, sample_depth


@dataclass(frozen=True)
class Obs:
    frame_idx: int
    frame_order: int
    hand_idx: int
    side: str
    track_id: str | None
    detector_score: float
    hand: dict
    target_frame: dict
    local_joints: np.ndarray
    local_vertices: np.ndarray
    joints2d: np.ndarray
    target_intrinsics: np.ndarray
    metric_depth: np.ndarray
    depth_weight: np.ndarray
    init_params: np.ndarray
    base_bone_m: float
    base_tip_spread_m: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def target_intrinsics(frame: dict, args: argparse.Namespace) -> np.ndarray:
    if args.target_intrinsics_source == "annotation-vggt":
        intr = np.asarray(frame.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=float)
    elif args.target_intrinsics_source == "cli":
        intr = np.asarray(args.intrinsics, dtype=float)
    else:
        raise RuntimeError(f"unsupported target intrinsics source {args.target_intrinsics_source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid target intrinsics for frame {frame.get('frame_idx')}")
    return intr


def target_source_size(frame: dict, args: argparse.Namespace) -> np.ndarray:
    size = np.asarray(frame.get("object", {}).get("source_image_size", []), dtype=float)
    if size.shape == (2,) and np.isfinite(size).all() and np.all(size > 0.0):
        return size
    return np.asarray([float(args.source_width), float(args.source_height)], dtype=float)


def local_vertices_key(hand: dict) -> str:
    if "vertices_camera" in hand:
        return "vertices_camera"
    if "vertices_camera_sample" in hand:
        return "vertices_camera_sample"
    raise RuntimeError("hand has no local MANO vertices")


def source_vertices_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    if "vertices_source_camera_m_sample" in hand:
        return "vertices_source_camera_m_sample"
    raise RuntimeError("hand has no source MANO vertices")


def source_to_world(points: np.ndarray, target_frame: dict) -> np.ndarray:
    T = np.asarray(target_frame["camera"]["T_world_camera_metric"], dtype=float)
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T @ homog.T).T[:, :3]


def rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def transform_points(local: np.ndarray, params: np.ndarray) -> np.ndarray:
    rotvec = params[:3]
    t = params[3:6]
    scale = math.exp(float(params[6]))
    return scale * (local @ rotation_matrix(rotvec).T) + t[None, :]


def stable_depth_weight(depth: np.ndarray, keypoints: np.ndarray, source_size: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
    ratios = np.asarray([depth_patch_iqr_ratio(depth, xy * scale, args.patch_radius) for xy in keypoints], dtype=float)
    return np.isfinite(ratios) & (ratios <= args.max_depth_iqr_ratio)


def initial_params(
    local_joints: np.ndarray,
    keypoints: np.ndarray,
    intrinsics: np.ndarray,
    metric_depth: np.ndarray,
    valid_depth: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    base_bone = hand_bone_scale_m(local_joints)
    if not np.isfinite(base_bone) or base_bone <= 0.0:
        raise RuntimeError("invalid local hand bone scale")
    scale = float(np.clip(args.hand_bone_scale_prior_m / base_bone, args.min_scale, args.max_scale))
    z = float(np.median(metric_depth[valid_depth]))
    if not np.isfinite(z) or z <= args.min_depth_m:
        raise RuntimeError("invalid target metric hand depth")
    center_px = np.median(keypoints[valid_depth], axis=0)
    fx, fy, cx, cy = intrinsics
    center_camera = np.asarray([(center_px[0] - cx) * z / fx, (center_px[1] - cy) * z / fy, z], dtype=float)
    local_center = np.median(scale * local_joints, axis=0)
    t = center_camera - local_center
    return np.r_[np.zeros(3, dtype=float), t, math.log(scale)]


def build_obs(args: argparse.Namespace) -> tuple[list[dict], list[Obs], list[dict]]:
    target_frames = load_frame_window(args.target_annotations, args.frame_start, args.frame_end)
    hand_frames = load_frame_window(args.hand_annotations, args.frame_start, args.frame_end)
    frame_to_depth_i, depths = load_depth_archive(args.metric_depth_npz)
    obs: list[Obs] = []
    skipped: list[dict] = []
    output_frames = []
    for order, frame_idx in enumerate(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride))):
        if frame_idx not in target_frames:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_target_frame"})
            continue
        if frame_idx not in hand_frames:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_hand_frame"})
            continue
        target = copy.deepcopy(target_frames[frame_idx])
        target["hands"] = copy.deepcopy(hand_frames[frame_idx].get("hands", []))
        output_frames.append(target)
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = target_source_size(target, args)
            intr = target_intrinsics(target, args)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(target.get("hands", [])):
            side = str(hand.get("side", "unknown"))
            try:
                if not bool(hand.get("measurement_available", False)):
                    raise RuntimeError("not_measured")
                score = float(hand.get("detector_score", np.nan))
                if not np.isfinite(score) or score < args.min_detector_score:
                    raise RuntimeError("low_detector_score")
                local_joints = np.asarray(hand["joints3d_camera"], dtype=float)
                local_vertices = np.asarray(hand[local_vertices_key(hand)], dtype=float)
                keypoints = np.asarray(hand["joints2d_raw"], dtype=float)
                if local_joints.shape != (21, 3) or local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid local MANO geometry")
                if keypoints.shape != (21, 2):
                    raise RuntimeError("invalid 2D hand keypoints")
                metric = sample_depth(depth, keypoints, source_size)
                valid = np.isfinite(metric) & (metric > args.min_depth_m)
                stable = stable_depth_weight(depth, keypoints, source_size, args)
                depth_weight = valid & stable
                if int(np.count_nonzero(depth_weight)) < args.min_depth_keypoints:
                    depth_weight = valid
                if int(np.count_nonzero(depth_weight)) < args.min_depth_keypoints:
                    raise RuntimeError("too_few_metric_depth_keypoints")
                init = initial_params(local_joints, keypoints, intr, metric, depth_weight, args)
                obs.append(
                    Obs(
                        frame_idx=frame_idx,
                        frame_order=order,
                        hand_idx=hand_i,
                        side=side,
                        track_id=hand.get("track_id"),
                        detector_score=score,
                        hand=hand,
                        target_frame=target,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        joints2d=keypoints,
                        target_intrinsics=intr,
                        metric_depth=metric,
                        depth_weight=depth_weight.astype(float),
                        init_params=init,
                        base_bone_m=hand_bone_scale_m(local_joints),
                        base_tip_spread_m=hand_tip_spread_m(local_joints),
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": side, "reason": str(exc)})
    if len(obs) < args.min_observations:
        raise RuntimeError(f"insufficient observations {len(obs)}; skipped={skipped[:20]}")
    return output_frames, obs, skipped


def pack_initial(obs: list[Obs]) -> np.ndarray:
    return np.concatenate([o.init_params for o in obs]).astype(float)


def unpack(params: np.ndarray) -> np.ndarray:
    if len(params) % 7 != 0:
        raise RuntimeError("parameter vector is not 7N")
    return params.reshape((-1, 7))


def residual(params: np.ndarray, obs: list[Obs], args: argparse.Namespace) -> np.ndarray:
    p = unpack(params)
    out: list[np.ndarray] = []
    centers = []
    for i, o in enumerate(obs):
        joints = transform_points(o.local_joints, p[i])
        min_depth_violation = np.clip(args.min_depth_m - joints[:, 2], 0.0, None)
        out.append(np.asarray([float(np.max(min_depth_violation)) / args.sigma_min_depth_m]))
        joints_for_terms = joints.copy()
        joints_for_terms[:, 2] = np.clip(joints_for_terms[:, 2], args.min_depth_m, None)
        projected = project_points(joints_for_terms, o.target_intrinsics)
        out.append(((projected - o.joints2d) / args.sigma_reprojection_px).reshape(-1))
        valid_depth = o.depth_weight > 0.0
        if np.any(valid_depth):
            out.append((joints_for_terms[valid_depth, 2] - o.metric_depth[valid_depth]) / args.sigma_metric_depth_m)
        scale = math.exp(float(p[i, 6]))
        bone = scale * o.base_bone_m
        out.append(
            np.asarray(
                [
                    np.clip(args.min_bone_scale_m - bone, 0.0, None) / args.sigma_bone_scale_m,
                    np.clip(bone - args.max_bone_scale_m, 0.0, None) / args.sigma_bone_scale_m,
                ],
                dtype=float,
            )
        )
        out.append(np.asarray([(bone - args.hand_bone_scale_prior_m) / args.sigma_bone_prior_m]))
        out.append(p[i, :3] / args.sigma_rotation_prior_rad)
        out.append(np.asarray([p[i, 6] / args.sigma_log_scale_prior]))
        out.append((p[i, 3:6] - o.init_params[3:6]) / args.sigma_translation_init_m)
        centers.append(np.median(joints_for_terms, axis=0))

    by_side: dict[str, list[int]] = {}
    for i, o in enumerate(obs):
        key = str(o.track_id) if o.track_id is not None else o.side
        by_side.setdefault(key, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda i: (obs[i].frame_idx, obs[i].hand_idx))
        for a, b in zip(indices[:-1], indices[1:]):
            if obs[b].frame_idx - obs[a].frame_idx > args.max_temporal_gap_frames:
                continue
            if not np.isfinite(centers[a]).all() or not np.isfinite(centers[b]).all():
                continue
            dt_frames = max(1, obs[b].frame_idx - obs[a].frame_idx)
            out.append((centers[b] - centers[a]) / (args.sigma_center_step_m * dt_frames))
            out.append((p[b, :3] - p[a, :3]) / (args.sigma_rotation_step_rad * dt_frames))
            out.append(np.asarray([(p[b, 6] - p[a, 6]) / (args.sigma_log_scale_step * dt_frames)]))
        for a, b, c in zip(indices[:-2], indices[1:-1], indices[2:]):
            if obs[c].frame_idx - obs[a].frame_idx > 2 * args.max_temporal_gap_frames:
                continue
            if not np.isfinite(centers[a]).all() or not np.isfinite(centers[b]).all() or not np.isfinite(centers[c]).all():
                continue
            out.append((centers[c] - 2.0 * centers[b] + centers[a]) / args.sigma_center_accel_m)
    return np.concatenate([np.ravel(x).astype(float) for x in out])


def solve(obs: list[Obs], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    x0 = pack_initial(obs)
    lo = np.tile(np.asarray([-args.max_rotation_rad, -args.max_rotation_rad, -args.max_rotation_rad, -2.0, -2.0, args.min_depth_m, math.log(args.min_scale)]), len(obs))
    hi = np.tile(np.asarray([args.max_rotation_rad, args.max_rotation_rad, args.max_rotation_rad, 2.0, 2.0, args.max_depth_m, math.log(args.max_scale)]), len(obs))
    result = least_squares(
        residual,
        x0,
        args=(obs, args),
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=args.max_nfev,
        verbose=0,
    )
    return result.x, {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
    }


def row_metrics(obs: list[Obs], params: np.ndarray) -> list[dict]:
    p = unpack(params)
    rows = []
    for i, o in enumerate(obs):
        before = o.init_params
        joints0 = transform_points(o.local_joints, before)
        vertices0 = transform_points(o.local_vertices, before)
        joints1 = transform_points(o.local_joints, p[i])
        vertices1 = transform_points(o.local_vertices, p[i])
        reproj0 = np.linalg.norm(project_points(joints0, o.target_intrinsics) - o.joints2d, axis=1)
        reproj1 = np.linalg.norm(project_points(joints1, o.target_intrinsics) - o.joints2d, axis=1)
        depth_valid = o.depth_weight > 0.0
        scale0 = math.exp(float(before[6]))
        scale1 = math.exp(float(p[i, 6]))
        center0 = np.median(joints0, axis=0)
        center1 = np.median(joints1, axis=0)
        rows.append(
            {
                "frame_idx": o.frame_idx,
                "hand_idx": o.hand_idx,
                "side": o.side,
                "track_id": o.track_id,
                "detector_score": o.detector_score,
                "depth_keypoints": int(np.count_nonzero(depth_valid)),
                "median_reprojection_before_px": float(np.median(reproj0)),
                "median_reprojection_after_px": float(np.median(reproj1)),
                "p95_reprojection_after_px": float(np.percentile(reproj1, 95.0)),
                "mano_minus_unidepth_before_m": float(np.median(joints0[depth_valid, 2] - o.metric_depth[depth_valid])),
                "mano_minus_unidepth_after_m": float(np.median(joints1[depth_valid, 2] - o.metric_depth[depth_valid])),
                "hand_bone_before_m": float(scale0 * o.base_bone_m),
                "hand_bone_after_m": float(scale1 * o.base_bone_m),
                "scale_before": float(scale0),
                "scale_after": float(scale1),
                "rotation_norm_after_rad": float(np.linalg.norm(p[i, :3])),
                "center_shift_from_init_m": float(np.linalg.norm(center1 - center0)),
                "center_after_m": center1.astype(float).tolist(),
                "vertex_count": int(len(vertices1)),
                "tip_spread_after_m": hand_tip_spread_m(joints1),
                "source_vertices_key": source_vertices_key(o.hand),
                "local_vertices_key": local_vertices_key(o.hand),
            }
        )
        _ = vertices0
    return rows


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            values.append(f)
    return summarize(np.asarray(values, dtype=float))


def temporal_summary(rows: list[dict], fps: float) -> dict:
    speeds = []
    by_side: dict[str, list[dict]] = {}
    for row in rows:
        by_side.setdefault(str(row["side"]), []).append(row)
    for side_rows in by_side.values():
        side_rows.sort(key=lambda r: (int(r["frame_idx"]), int(r["hand_idx"])))
        for a, b in zip(side_rows[:-1], side_rows[1:]):
            dt = max(1.0 / fps, (int(b["frame_idx"]) - int(a["frame_idx"])) / fps)
            ca = np.asarray(a["center_after_m"], dtype=float)
            cb = np.asarray(b["center_after_m"], dtype=float)
            speeds.append(float(np.linalg.norm(cb - ca) / dt))
    return {"center_speed_mps": summarize(np.asarray(speeds, dtype=float))}


def apply_solution(output_frames: list[dict], obs: list[Obs], params: np.ndarray, rows: list[dict]) -> None:
    frame_map = {int(frame["frame_idx"]): frame for frame in output_frames}
    p = unpack(params)
    row_by_key = {(int(r["frame_idx"]), int(r["hand_idx"]), str(r["side"])): r for r in rows}
    for i, o in enumerate(obs):
        frame = frame_map[o.frame_idx]
        hand = frame["hands"][o.hand_idx]
        local_joints = transform_points(o.local_joints, np.r_[p[i, :3], np.zeros(3), p[i, 6]])
        local_vertices = transform_points(o.local_vertices, np.r_[p[i, :3], np.zeros(3), p[i, 6]])
        t = p[i, 3:6]
        source_joints = local_joints + t[None, :]
        source_vertices = local_vertices + t[None, :]
        projected = project_points(source_joints, o.target_intrinsics)
        row = row_by_key[(o.frame_idx, o.hand_idx, o.side)]
        hand["joints3d_camera"] = local_joints.astype(float).tolist()
        hand[local_vertices_key(hand)] = local_vertices.astype(float).tolist()
        hand["cam_t"] = t.astype(float).tolist()
        hand["joints3d_source_camera_m"] = source_joints.astype(float).tolist()
        hand[source_vertices_key(hand)] = source_vertices.astype(float).tolist()
        hand["joints2d"] = projected.astype(float).tolist()
        hand["source_intrinsics"] = o.target_intrinsics.astype(float).tolist()
        hand["joints3d_world_m"] = source_to_world(source_joints, frame).astype(float).tolist()
        hand["vertices_world_m"] = source_to_world(source_vertices, frame).astype(float).tolist()
        hand["world_coordinate_status"] = "v3_target_camera_similarity_refit_world_from_target_camera"
        hand["filter_status"] = str(hand.get("filter_status", "")) + "_v3_target_similarity_refit"
        hand["v3_target_similarity_refit"] = {
            "status": "applied",
            "target_intrinsics_source": "annotation-vggt",
            "track_id": o.track_id,
            "scale": row["scale_after"],
            "rotation_norm_rad": row["rotation_norm_after_rad"],
            "center_shift_from_init_m": row["center_shift_from_init_m"],
            "median_reprojection_after_px": row["median_reprojection_after_px"],
            "p95_reprojection_after_px": row["p95_reprojection_after_px"],
            "mano_minus_unidepth_after_m": row["mano_minus_unidepth_after_m"],
            "hand_bone_after_m": row["hand_bone_after_m"],
        }


def report_for(rows: list[dict], solve_info: dict, skipped: list[dict], args: argparse.Namespace) -> dict:
    return {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "hand_annotations": str(args.hand_annotations),
        "target_annotations": str(args.target_annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "target_intrinsics_source": args.target_intrinsics_source,
        "solve": solve_info,
        "observations": int(len(rows)),
        "skipped_count": int(len(skipped)),
        "summary": {
            "median_reprojection_before_px": summarize_key(rows, "median_reprojection_before_px"),
            "median_reprojection_after_px": summarize_key(rows, "median_reprojection_after_px"),
            "p95_reprojection_after_px": summarize_key(rows, "p95_reprojection_after_px"),
            "mano_minus_unidepth_before_m": summarize_key(rows, "mano_minus_unidepth_before_m"),
            "mano_minus_unidepth_after_m": summarize_key(rows, "mano_minus_unidepth_after_m"),
            "hand_bone_after_m": summarize_key(rows, "hand_bone_after_m"),
            "scale_after": summarize_key(rows, "scale_after"),
            "rotation_norm_after_rad": summarize_key(rows, "rotation_norm_after_rad"),
            "center_shift_from_init_m": summarize_key(rows, "center_shift_from_init_m"),
            **temporal_summary(rows, args.fps),
        },
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "min_depth_keypoints": int(args.min_depth_keypoints),
            "min_bone_scale_m": float(args.min_bone_scale_m),
            "max_bone_scale_m": float(args.max_bone_scale_m),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
        },
        "interpretation": (
            "This stage tests whether an existing posed MANO hand can be re-anchored into the target VGGT camera "
            "with image and UniDepth evidence. Contact is not optimized here; a later diagnostic must still test "
            "object distance, penetration, and temporal support."
        ),
        "rows_preview": rows[:120],
        "skipped_preview": skipped[:120],
    }


def run(args: argparse.Namespace) -> dict:
    output_frames, obs, skipped = build_obs(args)
    params, solve_info = solve(obs, args)
    rows = row_metrics(obs, params)
    apply_solution(output_frames, obs, params, rows)
    output = {"frames": output_frames}
    save_json(args.output_annotations, output)
    report = report_for(rows, solve_info, skipped, args)
    report["output_annotations"] = str(args.output_annotations)
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--hand-annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-intrinsics-source", choices=["annotation-vggt", "cli"], default="annotation-vggt")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[2304.0, 2304.0, 960.0, 540.0])
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-detector-score", type=float, default=0.5)
    parser.add_argument("--min-depth-keypoints", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--max-depth-iqr-ratio", type=float, default=0.08)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.165)
    parser.add_argument("--min-bone-scale-m", type=float, default=0.12)
    parser.add_argument("--max-bone-scale-m", type=float, default=0.24)
    parser.add_argument("--min-scale", type=float, default=0.65)
    parser.add_argument("--max-scale", type=float, default=1.45)
    parser.add_argument("--max-rotation-rad", type=float, default=1.2)
    parser.add_argument("--sigma-reprojection-px", type=float, default=8.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.040)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.020)
    parser.add_argument("--sigma-bone-prior-m", type=float, default=0.050)
    parser.add_argument("--sigma-rotation-prior-rad", type=float, default=0.70)
    parser.add_argument("--sigma-log-scale-prior", type=float, default=0.30)
    parser.add_argument("--sigma-translation-init-m", type=float, default=0.25)
    parser.add_argument("--sigma-center-step-m", type=float, default=0.070)
    parser.add_argument("--sigma-center-accel-m", type=float, default=0.060)
    parser.add_argument("--sigma-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--sigma-log-scale-step", type=float, default=0.080)
    parser.add_argument("--sigma-min-depth-m", type=float, default=0.010)
    parser.add_argument("--max-temporal-gap-frames", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--invalid-penalty", type=float, default=1e3)
    parser.add_argument("--max-nfev", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
