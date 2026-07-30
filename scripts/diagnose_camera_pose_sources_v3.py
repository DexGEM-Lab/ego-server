#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def annotation_camera(path: Path) -> dict[int, np.ndarray]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames list")
    out = {}
    for frame in frames:
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid camera transform in frame {frame.get('frame_idx')}")
        out[int(frame["frame_idx"])] = T
    return out


def vggt_centers(path: Path | None) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    blob = np.load(path)
    required = {"frame_idx", "camera_centers_aligned"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    centers = blob["camera_centers_aligned"].astype(np.float64)
    return {int(idx): centers[i] for i, idx in enumerate(frame_idx)}


def summarize(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> dict:
    cameras = annotation_camera(args.annotations)
    centers_vggt = vggt_centers(args.vggt_archive)
    frame_indices = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in cameras]
    if len(frame_indices) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_indices)} camera frames")
    rows = []
    droid_centers = []
    droid_rotations = []
    vggt_center_rows = []
    for idx in frame_indices:
        T = cameras[idx]
        center = T[:3, 3].astype(float)
        rotation = T[:3, :3].astype(float)
        row = {"frame_idx": int(idx), "droid_center_world_m": center.tolist()}
        if idx in centers_vggt:
            diff = center - centers_vggt[idx]
            row["vggt_center_world_m"] = centers_vggt[idx].astype(float).tolist()
            row["droid_vggt_center_error_m"] = float(np.linalg.norm(diff))
            vggt_center_rows.append(centers_vggt[idx])
        rows.append(row)
        droid_centers.append(center)
        droid_rotations.append(rotation)
    pair_rows = []
    for i in range(1, len(frame_indices)):
        dt = (frame_indices[i] - frame_indices[i - 1]) / float(args.fps)
        if dt <= 0.0:
            raise RuntimeError("frame indices must increase")
        center_step = float(np.linalg.norm(droid_centers[i] - droid_centers[i - 1]))
        angular_step = float(np.linalg.norm(Rotation.from_matrix(droid_rotations[i - 1].T @ droid_rotations[i]).as_rotvec()))
        pair = {
            "from_frame": int(frame_indices[i - 1]),
            "to_frame": int(frame_indices[i]),
            "droid_center_step_m": center_step,
            "droid_center_speed_m_s": center_step / dt,
            "droid_angular_speed_rad_s": angular_step / dt,
        }
        if frame_indices[i] in centers_vggt and frame_indices[i - 1] in centers_vggt:
            vstep = float(np.linalg.norm(centers_vggt[frame_indices[i]] - centers_vggt[frame_indices[i - 1]]))
            pair["vggt_center_step_m"] = vstep
            pair["vggt_center_speed_m_s"] = vstep / dt
        pair_rows.append(pair)
    report = {
        "status": "ok",
        "method": "camera_pose_source_diagnostic_v3",
        "annotations": str(args.annotations),
        "vggt_archive": str(args.vggt_archive) if args.vggt_archive is not None else None,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(frame_indices)),
        "droid_center_speed_m_s": summarize([row["droid_center_speed_m_s"] for row in pair_rows]),
        "droid_angular_speed_rad_s": summarize([row["droid_angular_speed_rad_s"] for row in pair_rows]),
        "vggt_center_speed_m_s": summarize([row["vggt_center_speed_m_s"] for row in pair_rows if "vggt_center_speed_m_s" in row]),
        "droid_vggt_center_error_m": summarize([row["droid_vggt_center_error_m"] for row in rows if "droid_vggt_center_error_m" in row]),
        "rows": rows,
        "pairs": pair_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "pairs"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--vggt-archive", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
