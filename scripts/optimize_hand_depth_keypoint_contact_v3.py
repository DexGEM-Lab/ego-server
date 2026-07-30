#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


@dataclass(frozen=True)
class HandObs:
    frame_idx: int
    side: str
    detector_score: float
    match_delta_px: float
    rtmlib_score: float
    joints0: np.ndarray
    vertices0: np.ndarray
    joints2d_target: np.ndarray
    target_weight: np.ndarray
    intrinsics: np.ndarray
    metric_depth: np.ndarray
    object_depth: float
    near_vertex_indices: np.ndarray
    center_ray: np.ndarray


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def intrinsics_for_hand(frame: dict, hand: dict, args: argparse.Namespace) -> np.ndarray:
    if args.intrinsics_source == "hand":
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    elif args.intrinsics_source == "annotation-vggt":
        camera = frame.get("camera")
        if not isinstance(camera, dict) or "vggt_source_intrinsics_fx_fy_cx_cy" not in camera:
            raise RuntimeError(f"frame {frame.get('frame_idx')} missing camera.vggt_source_intrinsics_fx_fy_cx_cy")
        intr = np.asarray(camera["vggt_source_intrinsics_fx_fy_cx_cy"], dtype=float)
    elif args.intrinsics_source == "cli":
        intr = np.asarray(args.intrinsics, dtype=float)
    else:
        raise RuntimeError(f"unsupported intrinsics source: {args.intrinsics_source}")
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"invalid intrinsics for frame {frame.get('frame_idx')}: {intr}")
    return intr


def hand_vertex_key(hand: dict) -> str:
    if "vertices_source_camera_m" in hand:
        return "vertices_source_camera_m"
    if "vertices_camera" in hand:
        return "vertices_camera"
    if "vertices_source_camera_m_sample" in hand:
        return "vertices_source_camera_m_sample"
    if "vertices_camera_sample" in hand:
        return "vertices_camera_sample"
    raise RuntimeError("hand has no vertices")


def bone_scale(joints: np.ndarray) -> float:
    lengths = []
    for chain in FINGER_CHAINS:
        length = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            length += float(np.linalg.norm(joints[b] - joints[a]))
        lengths.append(length)
    return float(np.median(lengths))


def load_rtmlib_matches(path: Path, good_match_px: float) -> dict[tuple[int, str], dict]:
    data = load_json(path)
    matches: dict[tuple[int, str], dict] = {}
    for frame in data["rows_preview"]:
        frame_idx = int(frame["frame_idx"])
        for match in frame.get("matches", []):
            if float(match["median_keypoint_delta_px"]) > good_match_px:
                continue
            side = str(match.get("wilor_side", "unknown"))
            key = (frame_idx, side)
            if key not in matches or float(match["median_keypoint_delta_px"]) < float(matches[key]["median_keypoint_delta_px"]):
                matches[key] = match
    return matches


def resize_mask_to_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if mask.shape == depth.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def object_depth_camera(object_vertices_world: np.ndarray, T_world_camera: np.ndarray) -> float:
    homog = np.c_[object_vertices_world, np.ones(len(object_vertices_world), dtype=float)]
    camera = (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0.0)]
    if len(z) == 0:
        raise RuntimeError("object mesh has no positive camera depth")
    return float(np.median(z))


def build_obs(args: argparse.Namespace) -> tuple[list[HandObs], list[dict]]:
    annotations = load_json(args.annotations)
    frames = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    rtm_matches = load_rtmlib_matches(args.rtmlib_wilor_qc, args.max_rtmlib_wilor_delta_px)
    depth_blob = np.load(args.metric_depth_npz)
    depth_indices = depth_blob["frame_idx"].astype(int)
    if len(set(int(x) for x in depth_indices)) != len(depth_indices):
        raise RuntimeError("metric depth archive has duplicate frame_idx entries")
    frame_to_depth_i = {int(frame_idx): i for i, frame_idx in enumerate(depth_indices)}
    depths = np.asarray(depth_blob["depth"], dtype=np.float32)
    obs: list[HandObs] = []
    skipped: list[dict] = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames.get(frame_idx)
        if frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation"})
            continue
        obj = frame.get("object", {})
        if not obj.get("mask_path"):
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        try:
            depth = depth_frame(depths, frame_to_depth_i, frame_idx)
            source_size = np.asarray(obj["source_image_size"], dtype=float)
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, tuple(int(x) for x in obj["mask_image_size"]))
            mask_depth = resize_mask_to_depth(mask, depth)
            dist = mask_distance_map(mask_depth)
            object_vertices = mesh_frame_vertices(args.object_mesh_npz, frame_idx)
            object_depth = object_depth_camera(object_vertices, np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float))
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})
            continue
        for hand_i, hand in enumerate(frame.get("hands", [])):
            side = str(hand.get("side", "unknown"))
            try:
                if not bool(hand.get("measurement_available", False)):
                    raise RuntimeError("predicted_hand")
                detector_score = float(hand.get("detector_score", np.nan))
                if not np.isfinite(detector_score) or detector_score < args.min_detector_score:
                    raise RuntimeError("low_wilor_score")
                match = rtm_matches.get((frame_idx, side))
                if match is None:
                    raise RuntimeError("missing_good_rtmlib_wilor_match")
                joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
                vertices = np.asarray(hand[hand_vertex_key(hand)], dtype=float)
                target = np.asarray(hand["joints2d_raw"], dtype=float)
                intr = intrinsics_for_hand(frame, hand, args)
                if joints.shape != (21, 3) or vertices.ndim != 2 or vertices.shape[1] != 3:
                    raise RuntimeError("invalid hand geometry")
                if target.shape != (21, 2):
                    raise RuntimeError("invalid hand keypoints")
                metric = sample_depth(depth, target, source_size)
                valid_depth = np.isfinite(metric) & (metric > 0.0)
                if int(np.count_nonzero(valid_depth)) < args.min_depth_keypoints:
                    raise RuntimeError("too_few_metric_depth_keypoints")
                projected = project_points(joints, intr)
                current_reproj = np.linalg.norm(projected - target, axis=1)
                valid_2d = np.isfinite(current_reproj) & (current_reproj <= args.max_initial_keypoint_reprojection_px)
                if int(np.count_nonzero(valid_2d)) < args.min_reprojection_keypoints:
                    raise RuntimeError("too_few_initial_2d_keypoints")
                scale = np.asarray([depth.shape[1], depth.shape[0]], dtype=float) / source_size
                xy = project_points(vertices, intr) * scale[None, :]
                valid_uv = np.isfinite(xy).all(axis=1) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.0)
                x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1)
                y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
                near = np.flatnonzero(valid_uv & (dist[y, x] <= args.contact_distance_px))
                if len(near) < args.min_near_vertices:
                    raise RuntimeError("too_few_near_mask_vertices")
                center = np.median(joints, axis=0)
                if center[2] <= 0.0:
                    raise RuntimeError("nonpositive_hand_center")
                weight = np.zeros(21, dtype=float)
                weight[valid_2d] = 1.0
                weight[valid_depth] *= 1.0
                obs.append(
                    HandObs(
                        frame_idx=frame_idx,
                        side=side,
                        detector_score=detector_score,
                        match_delta_px=float(match["median_keypoint_delta_px"]),
                        rtmlib_score=float(match["rtmlib_mean_score"]),
                        joints0=joints,
                        vertices0=vertices,
                        joints2d_target=target,
                        target_weight=weight,
                        intrinsics=intr,
                        metric_depth=metric,
                        object_depth=object_depth,
                        near_vertex_indices=near,
                        center_ray=center / center[2],
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": side, "reason": str(exc)})
    if len(obs) < args.min_rows:
        args._last_skipped = skipped
        raise RuntimeError(f"insufficient hand observations: {len(obs)}; skipped={skipped[:12]}")
    return obs, skipped


def unpack(params: np.ndarray, n: int) -> tuple[float, np.ndarray, np.ndarray]:
    hand_log_scale = float(params[0])
    hand_shift = np.asarray(params[1 : 1 + n], dtype=float)
    object_shift = np.asarray(params[1 + n : 1 + 2 * n], dtype=float)
    return hand_log_scale, hand_shift, object_shift


def correct(points: np.ndarray, obs: HandObs, hand_log_scale: float, hand_shift: float) -> np.ndarray:
    out = math.exp(hand_log_scale) * points + float(hand_shift) * obs.center_ray[None, :]
    if np.any(out[:, 2] <= 0.0):
        raise RuntimeError("nonpositive corrected hand depth")
    return out


def residual(params: np.ndarray, obs: list[HandObs], args: argparse.Namespace, use_contact: bool) -> np.ndarray:
    hand_log_scale, hand_shift, object_shift = unpack(params, len(obs))
    residuals: list[np.ndarray] = [np.asarray([hand_log_scale / args.sigma_hand_log_scale], dtype=float)]
    for i, row in enumerate(obs):
        joints = correct(row.joints0, row, hand_log_scale, hand_shift[i])
        uv = project_points(joints, row.intrinsics)
        valid_2d = row.target_weight > 0.0
        reproj = (uv[valid_2d] - row.joints2d_target[valid_2d]).reshape(-1) / args.sigma_keypoint_px
        residuals.append(reproj * min(1.0, max(0.0, row.detector_score / args.detector_score_full)))
        valid_depth = np.isfinite(row.metric_depth) & (row.metric_depth > 0.0)
        residuals.append((joints[valid_depth, 2] - row.metric_depth[valid_depth]) / args.sigma_metric_depth_m)
        if use_contact and len(row.near_vertex_indices) > 0:
            vertices = correct(row.vertices0[row.near_vertex_indices], row, hand_log_scale, hand_shift[i])
            contact_gap = vertices[:, 2] - (row.object_depth + object_shift[i])
            if len(contact_gap) > args.max_contact_vertices:
                contact_gap = contact_gap[np.linspace(0, len(contact_gap) - 1, args.max_contact_vertices).astype(int)]
            residuals.append(contact_gap / args.sigma_contact_m)
            residuals.append(np.asarray([object_shift[i] / args.sigma_object_shift_m], dtype=float))
        else:
            residuals.append(np.asarray([object_shift[i] / args.sigma_object_shift_m], dtype=float))
        residuals.append(np.asarray([hand_shift[i] / args.sigma_hand_shift_m], dtype=float))
        scale_now = bone_scale(joints)
        residuals.append(np.asarray([(scale_now - args.hand_bone_scale_prior_m) / args.sigma_bone_scale_m], dtype=float))
    order = sorted(range(len(obs)), key=lambda i: (obs[i].side, obs[i].frame_idx))
    for a, b in zip(order[:-1], order[1:]):
        if obs[a].side != obs[b].side:
            continue
        dt = max(1, obs[b].frame_idx - obs[a].frame_idx)
        residuals.append(np.asarray([(hand_shift[b] - hand_shift[a]) / (args.sigma_hand_shift_step_m * dt)], dtype=float))
        residuals.append(np.asarray([(object_shift[b] - object_shift[a]) / (args.sigma_object_shift_step_m * dt)], dtype=float))
    return np.concatenate(residuals)


def solve(obs: list[HandObs], args: argparse.Namespace, use_contact: bool) -> tuple[np.ndarray, dict]:
    n = len(obs)
    x0 = np.zeros(1 + 2 * n, dtype=float)
    lower = np.r_[
        np.log(args.min_hand_scale),
        np.full(n, -args.max_abs_hand_shift_m),
        np.full(n, -args.max_abs_object_shift_m),
    ]
    upper = np.r_[
        np.log(args.max_hand_scale),
        np.full(n, args.max_abs_hand_shift_m),
        np.full(n, args.max_abs_object_shift_m),
    ]
    before = residual(x0, obs, args, use_contact)
    result = least_squares(
        lambda x: residual(x, obs, args, use_contact),
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(args.max_nfev),
        x_scale="jac",
    )
    after = residual(result.x, obs, args, use_contact)
    return result.x, {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
    }


def metrics(params: np.ndarray, obs: list[HandObs], args: argparse.Namespace) -> list[dict]:
    hand_log_scale, hand_shift, object_shift = unpack(params, len(obs))
    rows = []
    for i, row in enumerate(obs):
        joints = correct(row.joints0, row, hand_log_scale, hand_shift[i])
        vertices = correct(row.vertices0[row.near_vertex_indices], row, hand_log_scale, hand_shift[i])
        uv = project_points(joints, row.intrinsics)
        valid_2d = row.target_weight > 0.0
        valid_depth = np.isfinite(row.metric_depth) & (row.metric_depth > 0.0)
        reproj = np.linalg.norm(uv[valid_2d] - row.joints2d_target[valid_2d], axis=1)
        depth_gap = joints[valid_depth, 2] - row.metric_depth[valid_depth]
        contact_gap = vertices[:, 2] - (row.object_depth + object_shift[i]) if len(vertices) else np.asarray([], dtype=float)
        contact_median = None if len(contact_gap) == 0 else float(np.median(contact_gap))
        contact_p95 = None if len(contact_gap) == 0 else float(np.percentile(np.abs(contact_gap), 95.0))
        rows.append(
            {
                "frame_idx": row.frame_idx,
                "side": row.side,
                "detector_score": row.detector_score,
                "rtmlib_score": row.rtmlib_score,
                "rtmlib_wilor_delta_px": row.match_delta_px,
                "hand_shift_m": float(hand_shift[i]),
                "object_shift_m": float(object_shift[i]),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95.0)),
                "mano_minus_metric_depth_median_m": float(np.median(depth_gap)),
                "mano_minus_metric_depth_p95_abs_m": float(np.percentile(np.abs(depth_gap), 95.0)),
                "near_mask_vertices": int(len(row.near_vertex_indices)),
                "hand_minus_object_depth_median_m": contact_median,
                "hand_minus_object_depth_p95_abs_m": contact_p95,
                "hand_bone_scale_m": bone_scale(joints),
            }
        )
    return rows


def summarize_key(rows: list[dict], key: str) -> dict:
    values = []
    for row in rows:
        if key not in row:
            continue
        value = row[key]
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values.append(value_f)
    vals = np.asarray(values, dtype=float)
    return summarize(vals)


def row_summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "keypoint_reprojection_median_px": summarize_key(rows, "keypoint_reprojection_median_px"),
        "mano_minus_metric_depth_median_m": summarize_key(rows, "mano_minus_metric_depth_median_m"),
        "mano_minus_metric_depth_p95_abs_m": summarize_key(rows, "mano_minus_metric_depth_p95_abs_m"),
        "hand_minus_object_depth_median_m": summarize_key(rows, "hand_minus_object_depth_median_m"),
        "hand_minus_object_depth_p95_abs_m": summarize_key(rows, "hand_minus_object_depth_p95_abs_m"),
        "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
        "hand_shift_abs_m": summarize(np.abs(np.asarray([row["hand_shift_m"] for row in rows], dtype=float))),
        "object_shift_abs_m": summarize(np.abs(np.asarray([row["object_shift_m"] for row in rows], dtype=float))),
    }


def status_for(rows: list[dict], params: np.ndarray, args: argparse.Namespace, use_contact: bool) -> str:
    hand_scale = float(np.exp(params[0]))
    _, hand_shift, object_shift = unpack(params, len(rows))
    reproj = np.asarray([row["keypoint_reprojection_median_px"] for row in rows], dtype=float)
    depth_abs = np.asarray([abs(row["mano_minus_metric_depth_median_m"]) for row in rows], dtype=float)
    contact_abs = np.asarray(
        [abs(row["hand_minus_object_depth_median_m"]) for row in rows if row["hand_minus_object_depth_median_m"] is not None],
        dtype=float,
    )
    hand_bound = np.max(np.abs(hand_shift)) >= args.max_abs_hand_shift_m - args.bound_tolerance
    object_bound = np.max(np.abs(object_shift)) >= args.max_abs_object_shift_m - args.bound_tolerance
    scale_bound = hand_scale <= args.min_hand_scale + args.bound_tolerance or hand_scale >= args.max_hand_scale - args.bound_tolerance
    if float(np.median(reproj)) > args.solved_keypoint_median_px:
        return "diagnostic_keypoint_reprojection_residual_too_large"
    if float(np.median(depth_abs)) > args.solved_depth_median_m:
        return "diagnostic_metric_depth_residual_remains"
    if use_contact and len(contact_abs) == 0:
        return "diagnostic_no_contact_supported_rows"
    if use_contact and float(np.percentile(contact_abs, 95.0)) > args.solved_contact_p95_m:
        return "diagnostic_contact_residual_remains"
    if hand_bound or object_bound or scale_bound:
        return "diagnostic_solution_requires_bound_saturation"
    return "diagnostic_factors_fit_with_current_parameterization"


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.intrinsics = np.asarray(args.intrinsics, dtype=float)
    if args.intrinsics.shape != (4,):
        raise RuntimeError("--intrinsics must have four values")
    try:
        obs, skipped = build_obs(args)
    except RuntimeError as exc:
        if not args.allow_empty_observations or "insufficient hand observations" not in str(exc):
            raise
        obs = []
        skipped = getattr(args, "_last_skipped", [])
    x0 = np.zeros(1 + 2 * len(obs), dtype=float)
    if not obs:
        skipped_counts: dict[str, int] = {}
        for row in skipped:
            reason = str(row.get("reason", "unknown"))
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
        report = {
            "status": "diagnostic_no_hand_observations",
            "annotation_ready": False,
            "diagnostic_only": True,
            "model": "global_hand_scale_per_hand_ray_shift_per_row_object_depth_shift_with_2d_depth_contact_residuals",
            "annotations": str(args.annotations),
            "rtmlib_wilor_qc": str(args.rtmlib_wilor_qc),
            "metric_depth_npz": str(args.metric_depth_npz),
            "object_mesh_npz": str(args.object_mesh_npz),
            "frame_start": int(args.frame_start),
            "frame_end": int(args.frame_end),
            "observations": 0,
            "skipped_rows": len(skipped),
            "skipped_reason_counts": skipped_counts,
            "skipped_preview": skipped[:240],
            "elapsed_s": float(time.time() - started),
            "interpretation": "No hand rows satisfied the detector, 2D association, metric-depth, and near-object observation contract.",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "skipped_preview"}, indent=2))
        return report
    before_rows = metrics(x0, obs, args)
    depth_params, depth_solver = solve(obs, args, use_contact=False)
    depth_rows = metrics(depth_params, obs, args)
    contact_params, contact_solver = solve(obs, args, use_contact=True)
    contact_rows = metrics(contact_params, obs, args)
    report = {
        "status": status_for(contact_rows, contact_params, args, use_contact=True),
        "annotation_ready": False,
        "diagnostic_only": True,
        "model": "global_hand_scale_per_hand_ray_shift_per_row_object_depth_shift_with_2d_depth_contact_residuals",
        "annotations": str(args.annotations),
        "rtmlib_wilor_qc": str(args.rtmlib_wilor_qc),
        "metric_depth_npz": str(args.metric_depth_npz),
        "object_mesh_npz": str(args.object_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "observations": len(obs),
        "skipped_rows": len(skipped),
        "variables": int(1 + 2 * len(obs)),
        "before_summary": row_summary(before_rows),
        "depth_only_status": status_for(depth_rows, depth_params, args, use_contact=False),
        "depth_only_solver": depth_solver,
        "depth_only_hand_scale": float(np.exp(depth_params[0])),
        "depth_only_summary": row_summary(depth_rows),
        "contact_status": status_for(contact_rows, contact_params, args, use_contact=True),
        "contact_solver": contact_solver,
        "contact_hand_scale": float(np.exp(contact_params[0])),
        "contact_summary": row_summary(contact_rows),
        "thresholds": {
            "max_rtmlib_wilor_delta_px": float(args.max_rtmlib_wilor_delta_px),
            "solved_keypoint_median_px": float(args.solved_keypoint_median_px),
            "solved_depth_median_m": float(args.solved_depth_median_m),
            "solved_contact_p95_m": float(args.solved_contact_p95_m),
        },
        "rows_preview": contact_rows[:160],
        "skipped_preview": skipped[:160],
        "elapsed_s": float(time.time() - started),
        "interpretation": (
            "This diagnostic tests whether associated RTMLib/WiLoR 2D hand evidence, metric depth, and object contact "
            "can be satisfied by bounded hand scale and ray-depth shifts. A residual or bound-saturation status means "
            "the current hand/object/camera state remains physically inconsistent."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rtmlib-wilor-qc", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--object-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[2304.0, 2304.0, 960.0, 540.0])
    parser.add_argument("--intrinsics-source", choices=["hand", "annotation-vggt", "cli"], default="hand")
    parser.add_argument("--allow-empty-observations", action="store_true")
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--min-detector-score", type=float, default=0.5)
    parser.add_argument("--detector-score-full", type=float, default=0.8)
    parser.add_argument("--max-rtmlib-wilor-delta-px", type=float, default=30.0)
    parser.add_argument("--min-depth-keypoints", type=int, default=12)
    parser.add_argument("--min-reprojection-keypoints", type=int, default=12)
    parser.add_argument("--max-initial-keypoint-reprojection-px", type=float, default=40.0)
    parser.add_argument("--contact-distance-px", type=float, default=16.0)
    parser.add_argument("--min-near-vertices", type=int, default=20)
    parser.add_argument("--max-contact-vertices", type=int, default=80)
    parser.add_argument("--min-hand-scale", type=float, default=0.85)
    parser.add_argument("--max-hand-scale", type=float, default=1.15)
    parser.add_argument("--max-abs-hand-shift-m", type=float, default=0.15)
    parser.add_argument("--max-abs-object-shift-m", type=float, default=0.08)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.205)
    parser.add_argument("--sigma-keypoint-px", type=float, default=6.0)
    parser.add_argument("--sigma-metric-depth-m", type=float, default=0.035)
    parser.add_argument("--sigma-contact-m", type=float, default=0.015)
    parser.add_argument("--sigma-hand-log-scale", type=float, default=0.07)
    parser.add_argument("--sigma-hand-shift-m", type=float, default=0.06)
    parser.add_argument("--sigma-object-shift-m", type=float, default=0.04)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.025)
    parser.add_argument("--sigma-hand-shift-step-m", type=float, default=0.01)
    parser.add_argument("--sigma-object-shift-step-m", type=float, default=0.01)
    parser.add_argument("--solved-keypoint-median-px", type=float, default=20.0)
    parser.add_argument("--solved-depth-median-m", type=float, default=0.025)
    parser.add_argument("--solved-contact-p95-m", type=float, default=0.025)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-nfev", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
