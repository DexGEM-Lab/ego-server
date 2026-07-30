#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from diagnose_hand_reprojection_depth_v3 import project_points


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise RuntimeError("Sim3 alignment requires matching Nx3 arrays with at least 3 points")
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    X = source - src_mean
    Y = target - tgt_mean
    src_var = float(np.mean(np.sum(X * X, axis=1)))
    if src_var <= 0.0:
        raise RuntimeError("source camera trajectory has zero variance")
    cov = (Y.T @ X) / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / src_var)
    t = tgt_mean - scale * (R @ src_mean)
    return scale, R, t


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (np.asarray(points, dtype=float) @ rotation.T) + translation


def camera_points(world_points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    T_camera_world = np.linalg.inv(T_world_camera)
    homog = np.c_[world_points, np.ones(len(world_points), dtype=float)]
    return (T_camera_world @ homog.T).T[:, :3]


def source_hand(hand_npz: dict, side: str, frame_i: int) -> dict:
    valid = np.asarray(hand_npz[f"{side}_valid"], dtype=np.uint8)
    if frame_i >= len(valid) or int(valid[frame_i]) == 0:
        raise RuntimeError(f"HaWoR {side} hand invalid at frame {frame_i}")
    return {
        "vertices": np.asarray(hand_npz[f"{side}_vertices_world_m"], dtype=float)[frame_i],
        "joints": np.asarray(hand_npz[f"{side}_joints_world_m"], dtype=float)[frame_i],
        "trans": np.asarray(hand_npz[f"{side}_trans_world_m"], dtype=float)[frame_i],
        "root_orient": np.asarray(hand_npz[f"{side}_root_orient_axis_angle"], dtype=float)[frame_i],
        "hand_pose": np.asarray(hand_npz[f"{side}_hand_pose_axis_angle"], dtype=float)[frame_i],
        "betas": np.asarray(hand_npz[f"{side}_betas"], dtype=float)[frame_i],
    }


def observed_hand(frame: dict, side: str) -> dict | None:
    candidates = [
        hand
        for hand in frame.get("hands", [])
        if str(hand.get("side", "")).lower() == side and bool(hand.get("measurement_available", False))
    ]
    if candidates:
        candidates.sort(key=lambda hand: float(hand.get("detector_score", 0.0)), reverse=True)
        return candidates[0]
    return None


def fallback_intrinsics(frame: dict, observed: dict | None) -> np.ndarray:
    if observed is not None and "source_intrinsics" in observed:
        intr = np.asarray(observed["source_intrinsics"], dtype=float)
        if intr.shape == (4,):
            return intr
    for hand in frame.get("hands", []):
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
        if intr.shape == (4,):
            return intr
    raise RuntimeError("no source intrinsics available for HaWoR hand adaptation")


def adapt_hand(frame: dict, side: str, hawor: dict, frame_i: int, scale: float, rotation: np.ndarray, translation: np.ndarray) -> dict:
    src = source_hand(hawor, side, frame_i)
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    observed = observed_hand(frame, side)
    intr = fallback_intrinsics(frame, observed)
    vertices_world = apply_sim3(src["vertices"], scale, rotation, translation)
    joints_world = apply_sim3(src["joints"], scale, rotation, translation)
    vertices_camera = camera_points(vertices_world, T)
    joints_camera = camera_points(joints_world, T)
    if np.any(vertices_camera[:, 2] <= 0.0) or np.any(joints_camera[:, 2] <= 0.0):
        raise RuntimeError(f"adapted HaWoR {side} hand has non-positive source-camera depth")
    joints2d = project_points(joints_camera, intr)
    if observed is None:
        raw2d = joints2d
        bbox = None
        detector_score = 0.0
        measurement_available = False
    else:
        raw2d = np.asarray(observed.get("joints2d_raw", []), dtype=float)
        if raw2d.shape != (21, 2):
            raise RuntimeError(f"observed {side} hand has invalid joints2d_raw")
        bbox = observed.get("bbox_xyxy")
        detector_score = float(observed.get("detector_score", 0.0))
        measurement_available = True
    reproj = np.linalg.norm(joints2d - raw2d, axis=1)
    out = {
        **({"bbox_xyxy": [float(v) for v in bbox]} if bbox is not None else {}),
        "source_observation_backend": None if observed is None else observed.get("backend"),
        "source_observation_filter_status": None if observed is None else observed.get("filter_status"),
        "projection_residual_to_measurement_px": {
            "median": float(np.median(reproj)),
            "p95": float(np.percentile(reproj, 95.0)),
        },
    }
    return {
        **out,
        "backend": "HaWoR",
        "side": side,
        "measurement_available": measurement_available,
        "detector_score": detector_score,
        "filter_status": "hawor_world_export",
        "source_intrinsics": intr.astype(float).tolist(),
        "cam_t": np.median(joints_camera, axis=0).astype(float).tolist(),
        "joints2d": joints2d.astype(float).tolist(),
        "joints2d_raw": raw2d.astype(float).tolist(),
        "joints3d_source_camera_m": joints_camera.astype(float).tolist(),
        "joints3d_world_m": joints_world.astype(float).tolist(),
        "vertices_source_camera_m": vertices_camera.astype(float).tolist(),
        "vertices_world_m": vertices_world.astype(float).tolist(),
        "mano_params": {
            "global_orient_axis_angle": src["root_orient"].astype(float).tolist(),
            "hand_pose_axis_angle": src["hand_pose"].astype(float).tolist(),
            "betas": src["betas"].astype(float).tolist(),
        },
        "world_coordinate_status": "hawor_world_aligned_to_existing_camera_trajectory_sim3",
        "mano_surface_status": "full_vertices",
        "mano_vertex_count": int(len(vertices_world)),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    hawor_npz = np.load(args.hawor_npz, allow_pickle=True)
    hawor_frames = np.asarray(hawor_npz["frame_idx"], dtype=int)
    if np.any(hawor_frames != np.arange(len(hawor_frames))):
        raise RuntimeError("HaWoR frame_idx must be contiguous source-video frame indices")
    current_camera = []
    hawor_camera = []
    frames_by_idx = {int(frame["frame_idx"]): frame for frame in annotations["frames"]}
    t_c2w_h = np.asarray(hawor_npz["t_c2w"], dtype=float)
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.alignment_stride)):
        frame = frames_by_idx.get(frame_idx)
        if frame is None or frame_idx >= len(t_c2w_h):
            continue
        current_camera.append(np.asarray(frame["camera"]["position_world_m"], dtype=float))
        hawor_camera.append(t_c2w_h[frame_idx])
    current_camera_arr = np.asarray(current_camera, dtype=float)
    hawor_camera_arr = np.asarray(hawor_camera, dtype=float)
    scale, rotation, translation = estimate_sim3(hawor_camera_arr, current_camera_arr)
    aligned_cam = apply_sim3(hawor_camera_arr, scale, rotation, translation)
    cam_err = np.linalg.norm(aligned_cam - current_camera_arr, axis=1)

    out_frames = []
    adapted = []
    skipped = []
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            out_frames.append(frame)
            continue
        new_frame = copy.deepcopy(frame)
        new_hands = []
        for side in args.sides:
            try:
                new_hands.append(adapt_hand(new_frame, side, hawor_npz, frame_idx, scale, rotation, translation))
                adapted.append({"frame_idx": frame_idx, "side": side})
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": str(exc)})
        if new_hands:
            new_frame["hands"] = new_hands
        out_frames.append(new_frame)

    output = {"frames": out_frames}
    save_json(args.output_annotations, output)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "hawor_npz": str(args.hawor_npz),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "sides": list(args.sides),
        "alignment_frames": int(len(cam_err)),
        "alignment_camera_error_m": {
            "median": float(np.median(cam_err)),
            "p95": float(np.percentile(cam_err, 95.0)),
            "max": float(np.max(cam_err)),
        },
        "sim3": {
            "scale": float(scale),
            "rotation": rotation.astype(float).tolist(),
            "translation": translation.astype(float).tolist(),
        },
        "adapted_hands": int(len(adapted)),
        "skipped_hands": int(len(skipped)),
        "skipped_preview": skipped[:120],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k != "skipped_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--alignment-stride", type=int, default=5)
    parser.add_argument("--sides", nargs="+", default=["left", "right"], choices=["left", "right"])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
