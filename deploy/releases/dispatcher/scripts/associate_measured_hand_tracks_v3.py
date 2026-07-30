#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from compare_hand_streams_scale055_v3 import load_frame_window


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def hand_keypoints(hand: dict) -> np.ndarray:
    keypoints = np.asarray(hand.get("joints2d_raw", []), dtype=float)
    if keypoints.shape != (21, 2) or not np.isfinite(keypoints).all():
        raise RuntimeError("measured hand has invalid joints2d_raw")
    return keypoints


def measured_entries(frames: dict[int, dict], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    entries = []
    skipped = []
    for frame_idx in sorted(frames):
        for hand_i, hand in enumerate(frames[frame_idx].get("hands", [])):
            if not bool(hand.get("measurement_available", False)):
                continue
            score = float(hand.get("detector_score", np.nan))
            if not np.isfinite(score) or score < float(args.min_detector_score):
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": "low_detector_score"})
                continue
            try:
                keypoints = hand_keypoints(hand)
            except Exception as exc:
                skipped.append({"frame_idx": frame_idx, "hand_idx": hand_i, "side": hand.get("side"), "reason": str(exc)})
                continue
            entries.append(
                {
                    "frame_idx": int(frame_idx),
                    "hand_idx": int(hand_i),
                    "side": str(hand.get("side", "unknown")),
                    "score": score,
                    "keypoints": keypoints,
                    "center": np.median(keypoints, axis=0),
                }
            )
    return entries, skipped


def keypoint_distance(a: dict, b: dict) -> dict:
    delta = np.linalg.norm(a["keypoints"] - b["keypoints"], axis=1)
    center_delta = float(np.linalg.norm(a["center"] - b["center"]))
    return {
        "median_keypoint_delta_px": float(np.median(delta)),
        "p95_keypoint_delta_px": float(np.percentile(delta, 95.0)),
        "center_delta_px": center_delta,
    }


def assign_tracks(entries: list[dict], args: argparse.Namespace) -> tuple[dict[tuple[int, int], str], list[dict]]:
    tracks: dict[str, dict] = {}
    assignments: dict[tuple[int, int], str] = {}
    links = []
    next_track = 0
    for entry in sorted(entries, key=lambda item: (int(item["frame_idx"]), -float(item["score"]))):
        candidates = []
        for track_id, last in tracks.items():
            dt = int(entry["frame_idx"]) - int(last["frame_idx"])
            if dt <= 0 or dt > int(args.max_gap_frames):
                continue
            dist = keypoint_distance(last, entry)
            if dist["center_delta_px"] > float(args.max_center_delta_px):
                continue
            if dist["median_keypoint_delta_px"] > float(args.max_median_keypoint_delta_px):
                continue
            if dist["p95_keypoint_delta_px"] > float(args.max_p95_keypoint_delta_px):
                continue
            cost = dist["median_keypoint_delta_px"] + 0.25 * dist["center_delta_px"] + 0.02 * dist["p95_keypoint_delta_px"]
            candidates.append((cost, track_id, dist, dt))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _, track_id, dist, dt = candidates[0]
        else:
            track_id = f"track_{next_track:02d}"
            next_track += 1
            dist = None
            dt = None
        tracks[track_id] = entry
        assignments[(int(entry["frame_idx"]), int(entry["hand_idx"]))] = track_id
        links.append(
            {
                "frame_idx": int(entry["frame_idx"]),
                "hand_idx": int(entry["hand_idx"]),
                "side": entry["side"],
                "score": float(entry["score"]),
                "track_id": track_id,
                "link_dt_frames": None if dt is None else int(dt),
                "link_distance": dist,
            }
        )
    return assignments, links


def apply_tracks(frames: dict[int, dict], assignments: dict[tuple[int, int], str]) -> list[dict]:
    out = []
    for frame_idx in sorted(frames):
        frame = copy.deepcopy(frames[frame_idx])
        for hand_i, hand in enumerate(frame.get("hands", [])):
            key = (int(frame_idx), int(hand_i))
            if key in assignments:
                hand["track_id"] = assignments[key]
                hand["track_source"] = "v3_2d_measured_keypoint_continuity"
        out.append(frame)
    return out


def run(args: argparse.Namespace) -> dict:
    frames = load_frame_window(args.annotations, int(args.frame_start), int(args.frame_end))
    entries, skipped = measured_entries(frames, args)
    assignments, links = assign_tracks(entries, args)
    output = {"frames": apply_tracks(frames, assignments)}
    save_json(args.output_annotations, output)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "associate_measured_hand_tracks_v3",
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "measured_entries": int(len(entries)),
        "tracks": sorted(set(assignments.values())),
        "links": links,
        "skipped_count": int(len(skipped)),
        "skipped_preview": skipped[:120],
        "thresholds": {
            "min_detector_score": float(args.min_detector_score),
            "max_gap_frames": int(args.max_gap_frames),
            "max_center_delta_px": float(args.max_center_delta_px),
            "max_median_keypoint_delta_px": float(args.max_median_keypoint_delta_px),
            "max_p95_keypoint_delta_px": float(args.max_p95_keypoint_delta_px),
        },
        "interpretation": (
            "This diagnostic assigns temporal identity to measured hand detections by 2D keypoint continuity. "
            "It preserves the original side label as metadata because side labels can flip under ego-view occlusion."
        ),
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"links", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-detector-score", type=float, default=0.30)
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--max-center-delta-px", type=float, default=120.0)
    parser.add_argument("--max-median-keypoint-delta-px", type=float, default=120.0)
    parser.add_argument("--max-p95-keypoint-delta-px", type=float, default=220.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
