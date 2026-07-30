#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from diagnose_hand_contact_reliability_v3 import hand_bone_scale_m, hand_tip_spread_m
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_contact_depth_scale_v3 import summarize
from optimize_hand_translation_contact_v3 import source_to_world


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def hawor_camera_points(blob: np.lib.npyio.NpzFile, side: str, frame_idx: int, field: str) -> np.ndarray:
    world = np.asarray(blob[f"{side}_{field}_world_m"], dtype=float)[frame_idx]
    rotation = np.asarray(blob["R_c2w"], dtype=float)[frame_idx]
    translation = np.asarray(blob["t_c2w"], dtype=float)[frame_idx]
    return (rotation.T @ (world - translation[None, :]).T).T


def observed_hand(frame: dict, side: str) -> dict | None:
    candidates = [
        hand
        for hand in frame.get("hands", [])
        if str(hand.get("side", "")).lower() == side and bool(hand.get("measurement_available", False))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda hand: float(hand.get("detector_score", 0.0)))


def fallback_intrinsics(frame: dict, observed: dict | None) -> np.ndarray:
    if observed is not None:
        intr = np.asarray(observed.get("source_intrinsics", []), dtype=float)
        if intr.shape == (4,):
            return intr
    for hand in frame.get("hands", []):
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
        if intr.shape == (4,):
            return intr
    intr = np.asarray(frame.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=float)
    if intr.shape == (4,):
        return intr
    raise RuntimeError("no source intrinsics available")


def adapt_hand(
    frame: dict,
    blob: np.lib.npyio.NpzFile,
    side: str,
    source_frame_idx: int,
    hawor_frame_idx: int,
) -> tuple[dict, dict]:
    valid = np.asarray(blob[f"{side}_valid"], dtype=np.uint8)
    if hawor_frame_idx < 0 or hawor_frame_idx >= len(valid) or int(valid[hawor_frame_idx]) == 0:
        raise RuntimeError(f"HaWoR {side} hand invalid at source frame {source_frame_idx} / HaWoR frame {hawor_frame_idx}")
    joints_camera = hawor_camera_points(blob, side, hawor_frame_idx, "joints")
    vertices_camera = hawor_camera_points(blob, side, hawor_frame_idx, "vertices")
    if np.any(joints_camera[:, 2] <= 0.0) or np.any(vertices_camera[:, 2] <= 0.0):
        raise RuntimeError(
            f"HaWoR {side} camera-local hand has non-positive depth at source frame {source_frame_idx} "
            f"/ HaWoR frame {hawor_frame_idx}"
        )
    observed = observed_hand(frame, side)
    intr = fallback_intrinsics(frame, observed)
    joints2d = project_points(joints_camera, intr)
    if observed is None:
        raw2d = joints2d
        detector_score = 0.0
        measurement_available = False
        bbox = None
    else:
        raw2d = np.asarray(observed.get("joints2d_raw", []), dtype=float)
        if raw2d.shape != (21, 2):
            raise RuntimeError(f"observed {side} hand has invalid joints2d_raw at frame {source_frame_idx}")
        detector_score = float(observed.get("detector_score", 0.0))
        measurement_available = True
        bbox = observed.get("bbox_xyxy")
    reproj = np.linalg.norm(joints2d - raw2d, axis=1)
    T_world_camera = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    joints_world = source_to_world(joints_camera, T_world_camera)
    vertices_world = source_to_world(vertices_camera, T_world_camera)
    cam_t = joints_camera[0].astype(float)
    local_joints = joints_camera - cam_t[None, :]
    local_vertices = vertices_camera - cam_t[None, :]
    root_key = f"{side}_root_orient_axis_angle"
    pose_key = f"{side}_hand_pose_axis_angle"
    betas_key = f"{side}_betas"
    hand = {
        "backend": "HaWoR",
        "side": side,
        "measurement_available": measurement_available,
        "detector_score": detector_score,
        "filter_status": "hawor_camera_local",
        "source_intrinsics": intr.astype(float).tolist(),
        "cam_t": cam_t.astype(float).tolist(),
        "joints3d_camera": local_joints.astype(float).tolist(),
        "vertices_camera": local_vertices.astype(float).tolist(),
        "joints3d_source_camera_m": joints_camera.astype(float).tolist(),
        "vertices_source_camera_m": vertices_camera.astype(float).tolist(),
        "joints3d_world_m": joints_world.astype(float).tolist(),
        "vertices_world_m": vertices_world.astype(float).tolist(),
        "joints2d": joints2d.astype(float).tolist(),
        "joints2d_raw": raw2d.astype(float).tolist(),
        "projection_residual_to_measurement_px": {
            "median": float(np.median(reproj)),
            "p95": float(np.percentile(reproj, 95.0)),
        },
        "mano_params": {
            "global_orient_axis_angle": np.asarray(blob[root_key], dtype=float)[hawor_frame_idx].astype(float).tolist(),
            "hand_pose_axis_angle": np.asarray(blob[pose_key], dtype=float)[hawor_frame_idx].astype(float).tolist(),
            "betas": np.asarray(blob[betas_key], dtype=float)[hawor_frame_idx].astype(float).tolist(),
        },
        "world_coordinate_status": "hawor_camera_local_transformed_by_existing_annotation_camera_pose",
        "mano_surface_status": "full_vertices",
        "mano_vertex_count": int(len(vertices_camera)),
    }
    if bbox is not None:
        hand["bbox_xyxy"] = [float(v) for v in bbox]
    row = {
        "frame_idx": int(source_frame_idx),
        "hawor_frame_idx": int(hawor_frame_idx),
        "side": side,
        "measurement_available": bool(measurement_available),
        "detector_score": float(detector_score),
        "joint_reprojection_px_median": float(np.median(reproj)),
        "joint_reprojection_px_p95": float(np.percentile(reproj, 95.0)),
        "hand_bone_scale_m": hand_bone_scale_m(joints_camera),
        "hand_tip_spread_m": hand_tip_spread_m(joints_camera),
        "median_camera_depth_m": float(np.median(joints_camera[:, 2])),
    }
    return hand, row


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    blob = np.load(args.hawor_npz, allow_pickle=True)
    hawor_frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    if np.any(hawor_frame_idx != np.arange(len(hawor_frame_idx))):
        raise RuntimeError("HaWoR frame_idx must be contiguous zero-based indices in its processed video")
    output = copy.deepcopy(annotations)
    rows: list[dict] = []
    skipped: list[dict] = []
    for frame in output["frames"]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        hawor_idx = frame_idx - int(args.source_frame_offset)
        if hawor_idx < 0 or hawor_idx >= len(hawor_frame_idx):
            skipped.append(
                {
                    "frame_idx": frame_idx,
                    "side": "all",
                    "reason": f"source frame maps outside HaWoR window: {frame_idx} -> {hawor_idx}",
                }
            )
            continue
        new_hands = []
        for side in args.sides:
            try:
                hand, row = adapt_hand(frame, blob, side, frame_idx, hawor_idx)
                new_hands.append(hand)
                rows.append(row)
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": str(exc)})
        if new_hands:
            frame["hands"] = new_hands
    save_json(args.output_annotations, output)
    report = {
        "status": "diagnostic_hawor_camera_local_adapter",
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "hawor_npz": str(args.hawor_npz),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "source_frame_offset": int(args.source_frame_offset),
        "sides": list(args.sides),
        "adapted_hands": int(len(rows)),
        "skipped_hands": int(len(skipped)),
        "summary": {
            "joint_reprojection_px": summarize_key(rows, "joint_reprojection_px_median"),
            "hand_bone_scale_m": summarize_key(rows, "hand_bone_scale_m"),
            "hand_tip_spread_m": summarize_key(rows, "hand_tip_spread_m"),
            "median_camera_depth_m": summarize_key(rows, "median_camera_depth_m"),
        },
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:120],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-frame-offset", type=int, default=0)
    parser.add_argument("--sides", nargs="+", choices=["left", "right"], default=["left", "right"])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
