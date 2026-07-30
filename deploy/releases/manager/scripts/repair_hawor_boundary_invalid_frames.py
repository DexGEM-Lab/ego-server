#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SIDES = ("left", "right")
POINT_KEYS = ("vertices_world_m", "joints_world_m", "trans_world_m")
POSE_KEYS = ("root_orient_axis_angle", "hand_pose_axis_angle", "betas")


def axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    mat, _ = cv2.Rodrigues(np.asarray(vec, dtype=np.float64).reshape(3, 1))
    return mat.astype(np.float64)


def matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    vec, _ = cv2.Rodrigues(np.asarray(mat, dtype=np.float64).reshape(3, 3))
    return vec.reshape(3).astype(np.float32)


def boundary_invalid_indices(valid: np.ndarray, max_gap: int) -> tuple[list[int], str | None]:
    invalid = np.where(~valid.astype(bool))[0].astype(int).tolist()
    if not invalid:
        return [], None
    n = int(len(valid))
    prefix = list(range(0, len(invalid))) if invalid and invalid == list(range(0, len(invalid))) else []
    suffix_start = n - len(invalid)
    suffix = list(range(suffix_start, n)) if invalid and invalid == list(range(suffix_start, n)) else []
    if prefix and len(prefix) <= max_gap:
        return prefix, "prefix"
    if suffix and len(suffix) <= max_gap:
        return suffix, "suffix"
    return [], None


def fill_side(arrays: dict[str, np.ndarray], side: str, max_gap: int, hold: str) -> dict[str, Any]:
    valid_key = f"{side}_valid"
    valid = arrays[valid_key].astype(bool).copy()
    fill_indices, boundary = boundary_invalid_indices(valid, max_gap)
    report: dict[str, Any] = {
        "side": side,
        "original_valid_count": int(np.count_nonzero(valid)),
        "invalid_frames": np.where(~valid)[0].astype(int).tolist(),
        "filled_frames": [],
        "boundary": boundary,
        "status": "no_fill_needed" if not fill_indices else "filled_boundary_invalid_frames",
    }
    if np.any(~valid) and not fill_indices:
        report["status"] = "blocked_invalid_frames_not_fillable_boundary_gap"
        return report
    if not fill_indices:
        return report
    frame_count = int(len(valid))
    R = arrays["R_c2w"].astype(np.float64)
    t = arrays["t_c2w"].astype(np.float64)
    if boundary == "suffix":
        src = min(fill_indices) - 1
    elif boundary == "prefix":
        src = max(fill_indices) + 1
    else:
        report["status"] = "blocked_unknown_boundary"
        return report
    if src < 0 or src >= frame_count or not valid[src]:
        report["status"] = "blocked_no_valid_source_neighbor"
        report["source_frame"] = int(src)
        return report
    report["source_frame"] = int(src)
    source_cam: dict[str, np.ndarray] = {}
    for key in ("vertices_world_m", "joints_world_m"):
        world = arrays[f"{side}_{key}"][src].astype(np.float64)
        source_cam[key] = (R[src].T @ (world - t[src][None, :]).T).T
    trans_world = arrays[f"{side}_trans_world_m"][src].astype(np.float64)
    source_cam["trans_world_m"] = R[src].T @ (trans_world - t[src])
    root_world = axis_angle_to_matrix(arrays[f"{side}_root_orient_axis_angle"][src])
    root_cam = R[src].T @ root_world
    state_source_key = f"{side}_state_source"
    boundary_fill_key = f"{side}_temporal_boundary_filled"
    if state_source_key not in arrays:
        state_source = np.full(frame_count, "hawor_export", dtype="<U96")
    else:
        state_source = arrays[state_source_key].astype("<U96", copy=True)
    boundary_fill = arrays[boundary_fill_key].astype(bool, copy=True) if boundary_fill_key in arrays else np.zeros(frame_count, dtype=bool)
    for dst in fill_indices:
        for key in ("vertices_world_m", "joints_world_m"):
            arrays[f"{side}_{key}"][dst] = (R[dst] @ source_cam[key].T).T + t[dst][None, :]
        arrays[f"{side}_trans_world_m"][dst] = R[dst] @ source_cam["trans_world_m"] + t[dst]
        root_dst = R[dst] @ root_cam
        arrays[f"{side}_root_orient_axis_angle"][dst] = matrix_to_axis_angle(root_dst)
        for key in ("hand_pose_axis_angle", "betas"):
            arrays[f"{side}_{key}"][dst] = arrays[f"{side}_{key}"][src]
        valid[dst] = True
        if f"{side}_detected_same_frame" in arrays:
            arrays[f"{side}_detected_same_frame"][dst] = 0
        if f"{side}_det_box_xyxyscore" in arrays:
            arrays[f"{side}_det_box_xyxyscore"][dst] = np.full(5, np.nan, dtype=np.float32)
        if f"{side}_track_id" in arrays:
            arrays[f"{side}_track_id"][dst] = ""
        state_source[dst] = f"temporal_boundary_fill_{hold}_from_frame_{src}"
        boundary_fill[dst] = True
        report["filled_frames"].append(int(dst))
    arrays[valid_key] = valid.astype(arrays[valid_key].dtype)
    arrays[state_source_key] = state_source
    arrays[boundary_fill_key] = boundary_fill.astype(np.uint8)
    report["filled_count"] = int(len(fill_indices))
    report["final_valid_count"] = int(np.count_nonzero(valid))
    report["hold"] = hold
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    z = np.load(args.input_npz, allow_pickle=True)
    arrays: dict[str, np.ndarray] = {name: np.asarray(z[name]).copy() for name in z.files}
    frame_count = int(len(arrays["frame_idx"]))
    reports = [fill_side(arrays, side, args.max_gap, args.hold) for side in SIDES]
    blocked = [r for r in reports if str(r.get("status", "")).startswith("blocked")]
    arrays["boundary_fill_status"] = np.asarray(["blocked" if blocked else "ok"])
    arrays["boundary_fill_source_npz"] = np.asarray([str(args.input_npz)])
    arrays["boundary_fill_hold"] = np.asarray([args.hold])
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    report = {
        "method": "repair_hawor_boundary_invalid_frames",
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "frame_count": frame_count,
        "max_gap": int(args.max_gap),
        "hold": args.hold,
        "side_reports": reports,
        "status": "blocked" if blocked else "ok",
        "claim_scope": "explicit_temporal_boundary_fill_for_invalid_boundary_frames_no_same_frame_detection_no_observed_HaWoR_claim",
    }
    args.qc_json.parent.mkdir(parents=True, exist_ok=True)
    args.qc_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--hold", choices=["camera_local"], default="camera_local")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
