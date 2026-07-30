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
from diagnose_vggt_mano_contact_v3 import points_to_vggt_frame, resize_mask, vggt_frame_points


@dataclass(frozen=True)
class Obs:
    frame_idx: int
    side: str
    hand_index: int
    detector_score: float
    local_joints: np.ndarray
    local_vertices: np.ndarray
    target2d: np.ndarray
    intrinsics4: np.ndarray
    base_translation: np.ndarray
    object_depth: float
    near_indices: np.ndarray


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


def object_depth_vggt(blob: np.lib.npyio.NpzFile, index: int) -> float:
    points = vggt_frame_points(blob, index)
    extrinsic = blob["extrinsic"][index].astype(float)
    camera = (points @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]
    z = camera[:, 2]
    z = z[np.isfinite(z) & (z > 0)]
    if z.size == 0:
        raise RuntimeError("VGGT object points have no positive depth")
    return float(np.median(z))


def build_obs(args: argparse.Namespace) -> tuple[list[Obs], list[dict]]:
    annotations = load_json(args.annotations)
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    mask_by_idx = {int(row["frame_idx"]): Path(row["mask"]) for row in load_json(args.dataset_manifest)["frames"]}
    vggt = np.load(args.vggt_archive)
    frames = vggt["frame_idx"].astype(int)
    intrinsics = vggt["intrinsic"].astype(float)
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
        mask = resize_mask(mask_path, int(args.target_size))
        dist = mask_distance_map(mask)
        K = intrinsics[i]
        K4 = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=float)
        try:
            obj_depth = object_depth_vggt(vggt, i)
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
                if local_joints.shape != (21, 3) or local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                    raise RuntimeError("invalid_hand_geometry")
                if raw2d.shape != (21, 2):
                    raise RuntimeError("invalid_keypoints")
                target2d = points_to_vggt_frame(raw2d, frame["object"]["source_image_size"], int(args.target_size))
                base_t = solve_source_camera_translation(local_joints, target2d, K4)
                vertices = local_vertices + base_t[None, :]
                uv = project_points(vertices, K4)
                valid = np.isfinite(uv).all(axis=1) & (vertices[:, 2] > 0)
                x = np.clip(np.rint(uv[:, 0]).astype(int), 0, int(args.target_size) - 1)
                y = np.clip(np.rint(uv[:, 1]).astype(int), 0, int(args.target_size) - 1)
                near = np.flatnonzero(valid & (dist[y, x] <= float(args.contact_distance_px)))
                if len(near) < int(args.min_near_vertices):
                    raise RuntimeError("too_few_near_vertices")
                out.append(
                    Obs(
                        frame_idx=int(frame_idx),
                        side=side,
                        hand_index=int(hand_i),
                        detector_score=score,
                        local_joints=local_joints,
                        local_vertices=local_vertices,
                        target2d=target2d,
                        intrinsics4=K4,
                        base_translation=base_t,
                        object_depth=obj_depth,
                        near_indices=near.astype(int),
                    )
                )
            except Exception as exc:
                skipped.append({"frame_idx": int(frame_idx), "side": side, "hand_index": int(hand_i), "reason": str(exc)})
    if len(out) < int(args.min_rows):
        raise RuntimeError(f"insufficient observations: {len(out)}; skipped={skipped[:12]}")
    return out, skipped


def unpack(params: np.ndarray, n: int) -> tuple[float, np.ndarray, np.ndarray]:
    log_scale = float(params[0])
    shifts = params[1 : 1 + n].astype(float)
    velocities = params[1 + n : 1 + 2 * n].astype(float)
    return log_scale, shifts, velocities


def corrected(obs: Obs, log_scale: float, shift_z: float) -> tuple[np.ndarray, np.ndarray]:
    scale = math.exp(float(log_scale))
    joints = scale * obs.local_joints + obs.base_translation[None, :] + np.asarray([0.0, 0.0, shift_z], dtype=float)[None, :]
    vertices = scale * obs.local_vertices + obs.base_translation[None, :] + np.asarray([0.0, 0.0, shift_z], dtype=float)[None, :]
    if np.any(joints[:, 2] <= 0) or np.any(vertices[:, 2] <= 0):
        raise RuntimeError("nonpositive corrected hand depth")
    return joints, vertices


def residual(params: np.ndarray, obs: list[Obs], args: argparse.Namespace) -> np.ndarray:
    log_scale, shifts, velocities = unpack(params, len(obs))
    out: list[np.ndarray] = [np.asarray([log_scale / float(args.sigma_log_scale)], dtype=float)]
    for i, row in enumerate(obs):
        try:
            joints, vertices = corrected(row, log_scale, shifts[i])
        except RuntimeError:
            out.append(np.full(16, float(args.invalid_penalty), dtype=float))
            continue
        uv = project_points(joints, row.intrinsics4)
        score_weight = min(1.0, max(0.0, row.detector_score / float(args.detector_score_full)))
        out.append(score_weight * (uv - row.target2d).reshape(-1) / float(args.sigma_reprojection_px))
        gap = vertices[row.near_indices, 2] - row.object_depth
        if len(gap) > int(args.max_contact_vertices):
            gap = gap[np.linspace(0, len(gap) - 1, int(args.max_contact_vertices), dtype=int)]
        out.append(gap / float(args.sigma_contact_m))
        out.append(np.asarray([shifts[i] / float(args.sigma_shift_m)], dtype=float))
        out.append(np.asarray([velocities[i] / float(args.sigma_velocity_mps)], dtype=float))
        out.append(np.asarray([(hand_bone_scale_m(joints) - float(args.hand_bone_scale_prior_m)) / float(args.sigma_bone_scale_m)], dtype=float))
    by_side: dict[str, list[int]] = {}
    for i, row in enumerate(obs):
        by_side.setdefault(row.side, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda i: obs[i].frame_idx)
        for a, b in zip(indices[:-1], indices[1:]):
            dt = max(1.0 / float(args.fps), float(obs[b].frame_idx - obs[a].frame_idx) / float(args.fps))
            out.append(np.asarray([(shifts[b] - shifts[a] - velocities[a] * dt) / float(args.sigma_motion_m)], dtype=float))
            out.append(np.asarray([(velocities[b] - velocities[a]) / float(args.sigma_accel_mps2)], dtype=float))
    return np.concatenate([np.ravel(x).astype(float) for x in out])


def solve(obs: list[Obs], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    n = len(obs)
    x0 = np.zeros(1 + 2 * n, dtype=float)
    lower = np.r_[np.log(float(args.min_hand_scale)), np.full(n, -float(args.max_abs_shift_m)), np.full(n, -float(args.max_abs_velocity_mps))]
    upper = np.r_[np.log(float(args.max_hand_scale)), np.full(n, float(args.max_abs_shift_m)), np.full(n, float(args.max_abs_velocity_mps))]
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


def row_metrics(params: np.ndarray, obs: list[Obs]) -> list[dict]:
    log_scale, shifts, velocities = unpack(params, len(obs))
    rows = []
    for i, row in enumerate(obs):
        joints, vertices = corrected(row, log_scale, shifts[i])
        uv = project_points(joints, row.intrinsics4)
        reproj = np.linalg.norm(uv - row.target2d, axis=1)
        gap = vertices[row.near_indices, 2] - row.object_depth
        rows.append(
            {
                "frame_idx": int(row.frame_idx),
                "side": row.side,
                "hand_index": int(row.hand_index),
                "detector_score": float(row.detector_score),
                "shift_z_m": float(shifts[i]),
                "velocity_z_mps": float(velocities[i]),
                "keypoint_reprojection_median_px": float(np.median(reproj)),
                "keypoint_reprojection_p95_px": float(np.percentile(reproj, 95)),
                "near_mask_vertices": int(len(row.near_indices)),
                "contact_gap_median_m": float(np.median(gap)),
                "contact_gap_p95_abs_m": float(np.percentile(np.abs(gap), 95)),
                "hand_bone_scale_m": float(hand_bone_scale_m(joints)),
                "object_depth_vggt": float(row.object_depth),
                "hand_depth_median_m": float(np.median(vertices[row.near_indices, 2])),
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "rows": int(len(rows)),
        "shift_abs_m": summarize([abs(row["shift_z_m"]) for row in rows]),
        "velocity_abs_mps": summarize([abs(row["velocity_z_mps"]) for row in rows]),
        "keypoint_reprojection_median_px": summarize([row["keypoint_reprojection_median_px"] for row in rows]),
        "contact_gap_median_m": summarize([row["contact_gap_median_m"] for row in rows]),
        "contact_gap_p95_abs_m": summarize([row["contact_gap_p95_abs_m"] for row in rows]),
        "hand_bone_scale_m": summarize([row["hand_bone_scale_m"] for row in rows]),
    }


def status_for(params: np.ndarray, rows: list[dict], args: argparse.Namespace) -> str:
    log_scale, shifts, velocities = unpack(params, len(rows))
    if np.any(np.abs(shifts) >= float(args.max_abs_shift_m) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_shift_bound"
    if np.any(np.abs(velocities) >= float(args.max_abs_velocity_mps) - float(args.bound_tolerance)):
        return "diagnostic_solution_requires_velocity_bound"
    if math.exp(log_scale) <= float(args.min_hand_scale) + float(args.bound_tolerance):
        return "diagnostic_solution_requires_scale_bound"
    if math.exp(log_scale) >= float(args.max_hand_scale) - float(args.bound_tolerance):
        return "diagnostic_solution_requires_scale_bound"
    contact_abs = np.asarray([abs(row["contact_gap_median_m"]) for row in rows], dtype=float)
    reproj = np.asarray([row["keypoint_reprojection_median_px"] for row in rows], dtype=float)
    if float(np.percentile(contact_abs, 95)) > float(args.accept_contact_p95_m):
        return "diagnostic_contact_residual_remains"
    if float(np.median(reproj)) > float(args.accept_reprojection_median_px):
        return "diagnostic_reprojection_residual_too_large"
    return "diagnostic_vggt_mano_depth_candidate"


def run(args: argparse.Namespace) -> dict:
    obs, skipped = build_obs(args)
    x0 = np.zeros(1 + 2 * len(obs), dtype=float)
    before_rows = row_metrics(x0, obs)
    params, solver = solve(obs, args)
    rows = row_metrics(params, obs)
    report = {
        "status": status_for(params, rows, args),
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_vggt_mano_depth_contact_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "observations": int(len(obs)),
        "skipped_rows": int(len(skipped)),
        "variables": int(1 + 2 * len(obs)),
        "solver": solver,
        "hand_scale": float(math.exp(params[0])),
        "before_summary": summarize_rows(before_rows),
        "after_summary": summarize_rows(rows),
        "rows": rows,
        "skipped_preview": skipped[:120],
        "parameters": vars(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qc_vggt_mano_depth_contact_v3.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "skipped_preview", "parameters"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-detector-score", type=float, default=0.50)
    parser.add_argument("--contact-distance-px", type=float, default=8.0)
    parser.add_argument("--min-near-vertices", type=int, default=80)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--max-contact-vertices", type=int, default=220)
    parser.add_argument("--detector-score-full", type=float, default=0.80)
    parser.add_argument("--sigma-log-scale", type=float, default=0.25)
    parser.add_argument("--sigma-reprojection-px", type=float, default=5.0)
    parser.add_argument("--sigma-contact-m", type=float, default=0.020)
    parser.add_argument("--sigma-shift-m", type=float, default=0.18)
    parser.add_argument("--sigma-velocity-mps", type=float, default=0.80)
    parser.add_argument("--sigma-motion-m", type=float, default=0.025)
    parser.add_argument("--sigma-accel-mps2", type=float, default=1.50)
    parser.add_argument("--hand-bone-scale-prior-m", type=float, default=0.210)
    parser.add_argument("--sigma-bone-scale-m", type=float, default=0.020)
    parser.add_argument("--min-hand-scale", type=float, default=0.80)
    parser.add_argument("--max-hand-scale", type=float, default=1.12)
    parser.add_argument("--max-abs-shift-m", type=float, default=0.35)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=1.20)
    parser.add_argument("--max-nfev", type=int, default=160)
    parser.add_argument("--invalid-penalty", type=float, default=1e3)
    parser.add_argument("--bound-tolerance", type=float, default=1e-4)
    parser.add_argument("--accept-contact-p95-m", type=float, default=0.030)
    parser.add_argument("--accept-reprojection-median-px", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
