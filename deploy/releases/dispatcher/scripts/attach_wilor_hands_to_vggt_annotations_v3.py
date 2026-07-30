#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from build_vggt_wilor_object_annotations_v3 import summarize, transform_hand_to_world
from fuse_v1_full_fidelity import (
    choose_hand_by_side,
    hand_metric_scale_from_raw,
    hand_vector,
    kalman_rts,
    normalize_hand_to_source_camera,
    project_points,
    vector_to_hand,
)


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_raw_wilor(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    out: dict[int, dict] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate WiLoR frame {frame_idx}")
        out[frame_idx] = frame
    return out


def intrinsics_for(frame: dict) -> np.ndarray:
    intr = np.asarray(frame.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
    if intr.shape != (4,) or not np.isfinite(intr).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} missing valid VGGT source intrinsics")
    return intr


def transform_for(frame: dict) -> np.ndarray:
    transform = np.asarray(frame.get("camera", {}).get("T_world_camera_metric", []), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} missing valid T_world_camera_metric")
    return transform


def normalize_frames(
    frames: list[dict],
    raw_by_frame: dict[int, dict],
    wilor_to_meters: float,
) -> tuple[list[dict], dict]:
    normalized_by_frame = []
    rejected = {"left": 0, "right": 0, "unknown": 0}
    raw_count = 0
    normalized_count = 0
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        raw_frame = raw_by_frame.get(frame_idx)
        if raw_frame is None:
            raise RuntimeError(f"WiLoR raw missing frame {frame_idx}")
        intr = intrinsics_for(frame)
        normalized = []
        for raw_hand in raw_frame.get("raw_hands", []):
            raw_count += 1
            hand = normalize_hand_to_source_camera(raw_hand, intr, float(wilor_to_meters))
            if hand is None:
                side = str(raw_hand.get("side", "unknown"))
                rejected[side] = rejected.get(side, 0) + 1
                continue
            normalized.append(hand)
            normalized_count += 1
        normalized_by_frame.append({"frame_idx": frame_idx, "raw_hands": normalized})
    return normalized_by_frame, {
        "raw_hand_detections": int(raw_count),
        "normalized_hands": int(normalized_count),
        "rejected_source_camera_solves": rejected,
    }


def smooth_side(
    frames: list[dict],
    normalized_by_frame: list[dict],
    chosen: list[dict[str, dict]],
    side: str,
    fps: float,
) -> tuple[dict[int, dict], dict]:
    templates = [row[side] for row in chosen if side in row]
    if not templates:
        return {}, {"measured_frames": 0, "predicted_frames": 0, "coverage": 0.0}
    template = templates[0]
    measurements = []
    confs = []
    best_by_frame = []
    for row in chosen:
        hand = row.get(side)
        best_by_frame.append(hand)
        measurements.append(hand_vector(hand) if hand is not None else None)
        confs.append(float(hand.get("detector_score", 0.0)) if hand is not None else 0.0)
    measured_indices = [i for i, measurement in enumerate(measurements) if measurement is not None]
    if not measured_indices:
        return {}, {"measured_frames": 0, "predicted_frames": 0, "coverage": 0.0}
    first_measured = measured_indices[0]
    last_measured = measured_indices[-1]
    dim = int(measurements[first_measured].shape[0])  # type: ignore[union-attr]
    meas_sigma = np.full(dim, 0.010, dtype=float)
    proc_pos = np.full(dim, 0.006, dtype=float)
    proc_vel = np.full(dim, 0.025, dtype=float)
    smoothed, statuses = kalman_rts(measurements, confs, fps, meas_sigma, proc_pos, proc_vel)
    out: dict[int, dict] = {}
    measured_count = 0
    predicted_count = 0
    projection_medians = []
    detector_scores = []
    for i, vec in enumerate(smoothed):
        if i < first_measured or i > last_measured:
            continue
        frame = frames[i]
        frame_idx = int(frame["frame_idx"])
        intr = intrinsics_for(frame)
        source = best_by_frame[i] or template
        hand = vector_to_hand(source, vec, statuses[i], confs[i], intr)
        hand["measurement_available"] = best_by_frame[i] is not None
        hand["track_id"] = f"wilor_{side}"
        hand["track_source"] = "wilor_side_kalman_rts"
        if hand["measurement_available"]:
            measured_count += 1
            detector_scores.append(float(hand.get("detector_score", 0.0)))
            raw = np.asarray(hand["joints2d_raw"], dtype=float)
            joints = np.asarray(hand["joints3d_source_camera_m"], dtype=float)
            err = np.linalg.norm(project_points(joints, intr) - raw, axis=1)
            hand["projection_residual_to_measurement_px"] = {
                "median": float(np.median(err)),
                "p95": float(np.percentile(err, 95)),
            }
            projection_medians.append(float(np.median(err)))
        else:
            predicted_count += 1
        transform_hand_to_world(hand, transform_for(frame))
        out[frame_idx] = hand
    return out, {
        "measured_frames": int(measured_count),
        "predicted_frames": int(predicted_count),
        "coverage": float(measured_count / max(1, len(frames))),
        "detector_score": summarize(detector_scores),
        "median_projection_residual_px": summarize(projection_medians),
    }


def video_fps_from_annotations(frames: list[dict], fallback: float) -> float:
    times = np.asarray([float(frame.get("time_s", np.nan)) for frame in frames], dtype=float)
    times = times[np.isfinite(times)]
    if len(times) >= 2:
        dt = np.diff(times)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(dt):
            return float(1.0 / np.median(dt))
    if not math.isfinite(fallback) or fallback <= 0.0:
        raise RuntimeError("could not infer FPS from annotations or fallback")
    return float(fallback)


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = [copy.deepcopy(frame) for frame in annotations.get("frames", [])]
    if not frames:
        raise RuntimeError(f"{args.annotations} contains no frames")
    frames = [frame for frame in frames if int(args.frame_start) <= int(frame["frame_idx"]) <= int(args.frame_end)]
    if not frames:
        raise RuntimeError("no annotation frames remain after frame filtering")
    raw_by_frame = load_raw_wilor(args.wilor_raw)
    raw_window = [raw_by_frame[int(frame["frame_idx"])] for frame in frames]
    hand_scale = hand_metric_scale_from_raw(raw_window)
    normalized_by_frame, normalize_stats = normalize_frames(frames, raw_by_frame, float(hand_scale["wilor_local_to_meters"]))
    chosen = [choose_hand_by_side(row["raw_hands"]) for row in normalized_by_frame]
    fps = video_fps_from_annotations(frames, float(args.fps))
    by_side: dict[str, dict[int, dict]] = {}
    side_stats: dict[str, dict] = {}
    for side in ("left", "right"):
        by_side[side], side_stats[side] = smooth_side(frames, normalized_by_frame, chosen, side, fps)
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        hands = [by_side[side][frame_idx] for side in ("left", "right") if frame_idx in by_side[side]]
        frame["hands"] = hands
    measured = [
        hand
        for frame in frames
        for hand in frame.get("hands", [])
        if bool(hand.get("measurement_available", False))
    ]
    if len(measured) < int(args.min_measured_hands):
        raise RuntimeError(f"only {len(measured)} measured WiLoR hands, min_measured_hands={args.min_measured_hands}")
    reproj = [
        float(hand["projection_residual_to_measurement_px"]["median"])
        for hand in measured
        if "projection_residual_to_measurement_px" in hand
    ]
    output = {"frames": frames}
    save_json(args.output_annotations, output)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "attach_wilor_hands_to_vggt_annotations_v3",
        "annotations": str(args.annotations),
        "wilor_raw": str(args.wilor_raw),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(frames[0]["frame_idx"]),
        "frame_end": int(frames[-1]["frame_idx"]),
        "frames": int(len(frames)),
        "fps": float(fps),
        "hand_scale": hand_scale,
        "normalize": normalize_stats,
        "sides": side_stats,
        "measured_hand_rows": int(len(measured)),
        "measured_reprojection_median_px": summarize(reproj),
        "interpretation": (
            "This attaches WiLoR MANO hands to an existing VGGT/object annotation skeleton. "
            "The output is a hand observation stream; object contact and physical consistency must be tested by later "
            "mesh-surface diagnostics under the same camera and depth contract."
        ),
    }
    save_json(args.output_qc, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--wilor-raw", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-measured-hands", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
