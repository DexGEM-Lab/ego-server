#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SIDE_FROM_HANDEDNESS = {0: "left", 1: "right"}


def best_track_support(tracks_npy: Path, frame_count: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    support: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        support[side] = {
            "detected_same_frame": np.zeros(frame_count, dtype=np.uint8),
            "det_box_xyxyscore": np.full((frame_count, 5), np.nan, dtype=np.float32),
            "track_id": np.full(frame_count, "", dtype="<U64"),
        }
    report: dict[str, Any] = {
        "tracks_npy": str(tracks_npy),
        "tracks_file_exists": tracks_npy.exists(),
        "side_handedness_mapping": {"left": 0, "right": 1},
        "records_read": 0,
        "records_used_as_best_same_frame_detection": 0,
        "dropped_records": 0,
    }
    if not tracks_npy.exists():
        report["status"] = "tracks_file_missing"
        return support, report
    tracks_obj = np.load(tracks_npy, allow_pickle=True)
    tracks = tracks_obj.item() if getattr(tracks_obj, "shape", None) == () else tracks_obj
    if not isinstance(tracks, dict):
        report["status"] = "tracks_file_not_dict"
        report["tracks_object_type"] = str(type(tracks))
        return support, report
    for track_id, records in tracks.items():
        for rec in records:
            report["records_read"] += 1
            try:
                frame_idx = int(rec.get("frame", -1))
                if frame_idx < 0 or frame_idx >= frame_count:
                    report["dropped_records"] += 1
                    continue
                handed_arr = np.asarray(rec.get("det_handedness"), dtype=np.float32).reshape(-1)
                if handed_arr.size == 0:
                    report["dropped_records"] += 1
                    continue
                side = SIDE_FROM_HANDEDNESS.get(int(round(float(handed_arr[0]))))
                if side is None:
                    report["dropped_records"] += 1
                    continue
                box = np.asarray(rec.get("det_box"), dtype=np.float32).reshape(-1)
                if box.size < 5 or not np.isfinite(box[:5]).all():
                    report["dropped_records"] += 1
                    continue
                current_box = support[side]["det_box_xyxyscore"][frame_idx]
                current_score = float(current_box[4]) if np.isfinite(current_box[4]) else -np.inf
                if support[side]["detected_same_frame"][frame_idx] == 0 or float(box[4]) > current_score:
                    support[side]["detected_same_frame"][frame_idx] = 1
                    support[side]["det_box_xyxyscore"][frame_idx] = box[:5]
                    support[side]["track_id"][frame_idx] = str(track_id)
                    report["records_used_as_best_same_frame_detection"] += 1
            except Exception:
                report["dropped_records"] += 1
                continue
    report["status"] = "ok"
    report["detected_same_frame_counts"] = {side: int(np.count_nonzero(support[side]["detected_same_frame"])) for side in support}
    return support, report


def longest_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(args.hawor_npz, allow_pickle=True)
    frame_idx = np.asarray(z["frame_idx"], dtype=np.int32)
    frame_count = int(len(frame_idx))
    support, support_report = best_track_support(args.tracks_npy, frame_count)
    arrays: dict[str, np.ndarray] = {name: np.asarray(z[name]) for name in z.files}
    for side in ("left", "right"):
        arrays[f"{side}_detected_same_frame"] = support[side]["detected_same_frame"]
        arrays[f"{side}_det_box_xyxyscore"] = support[side]["det_box_xyxyscore"]
        arrays[f"{side}_track_id"] = support[side]["track_id"]
    arrays["track_support_status"] = np.asarray([support_report.get("status", "unknown")])
    arrays["track_support_path"] = np.asarray([str(args.tracks_npy)])
    arrays["support_augmented_from_npz"] = np.asarray([str(args.hawor_npz)])
    np.savez_compressed(args.output_npz, **arrays)
    per_side: dict[str, Any] = {}
    for side in ("left", "right"):
        valid_key = f"{side}_valid"
        valid = np.asarray(z[valid_key]).astype(bool) if valid_key in z.files else np.ones(frame_count, dtype=bool)
        detected = support[side]["detected_same_frame"].astype(bool)
        per_side[side] = {
            "valid_count": int(np.count_nonzero(valid)),
            "same_frame_detection_count": int(np.count_nonzero(detected)),
            "valid_with_same_frame_detection": int(np.count_nonzero(valid & detected)),
            "valid_without_same_frame_detection": int(np.count_nonzero(valid & ~detected)),
            "longest_valid_without_detection_run": longest_run(valid & ~detected),
        }
    report = {
        "method": "augment_hawor_track_support",
        "hawor_npz": str(args.hawor_npz),
        "tracks_npy": str(args.tracks_npy),
        "output_npz": str(args.output_npz),
        "frame_count": frame_count,
        "track_support": support_report,
        "per_side": per_side,
        "claim_scope": "post_hoc_HaWoR_track_support_provenance_no_new_MANO_inference_no_foundation_acceptance",
    }
    args.qc_json.parent.mkdir(parents=True, exist_ok=True)
    args.qc_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--tracks-npy", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--qc-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
