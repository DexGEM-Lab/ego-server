#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def hand_delta(rtm_hand: dict, wilor_hand: dict, min_score: float, min_points: int) -> dict | None:
    rtm = np.asarray(rtm_hand["keypoints"], dtype=float)
    scores = np.asarray(rtm_hand["scores"], dtype=float)
    wilor = np.asarray(wilor_hand.get("joints2d_raw", []), dtype=float)
    if rtm.shape != (21, 2) or scores.shape != (21,) or wilor.shape != (21, 2):
        return None
    valid = np.isfinite(rtm).all(axis=1) & np.isfinite(wilor).all(axis=1) & np.isfinite(scores) & (scores >= min_score)
    if int(np.count_nonzero(valid)) < min_points:
        return None
    delta = np.linalg.norm(rtm[valid] - wilor[valid], axis=1)
    return {
        "matched_keypoints": int(np.count_nonzero(valid)),
        "median_keypoint_delta_px": float(np.median(delta)),
        "p95_keypoint_delta_px": float(np.percentile(delta, 95.0)),
    }


def best_assignment(rtm_hands: list[dict], wilor_hands: list[dict], args: argparse.Namespace) -> tuple[list[dict], list[int], list[int]]:
    candidates = {}
    for ri, rtm_hand in enumerate(rtm_hands):
        for wi, wilor_hand in enumerate(wilor_hands):
            row = hand_delta(rtm_hand, wilor_hand, args.min_rtmlib_keypoint_score, args.min_matched_keypoints)
            if row is not None:
                candidates[(ri, wi)] = row
    if not candidates:
        return [], list(range(len(rtm_hands))), list(range(len(wilor_hands)))
    n = min(len(rtm_hands), len(wilor_hands))
    best: list[dict] = []
    best_cost = float("inf")
    rtm_indices = range(len(rtm_hands))
    wilor_indices = range(len(wilor_hands))
    for k in range(1, n + 1):
        for rsel in itertools.combinations(rtm_indices, k):
            for wsel in itertools.permutations(wilor_indices, k):
                rows = []
                cost = 0.0
                ok = True
                for ri, wi in zip(rsel, wsel):
                    row = candidates.get((ri, wi))
                    if row is None:
                        ok = False
                        break
                    cost += float(row["median_keypoint_delta_px"])
                    rows.append(
                        {
                            "rtmlib_hand_idx": int(rtm_hands[ri]["hand_idx"]),
                            "rtmlib_list_idx": int(ri),
                            "rtmlib_mean_score": float(rtm_hands[ri]["mean_score"]),
                            "wilor_list_idx": int(wi),
                            "wilor_side": wilor_hands[wi].get("side"),
                            "wilor_score": float(wilor_hands[wi].get("detector_score", np.nan)),
                            **row,
                        }
                    )
                if ok and (len(rows) > len(best) or (len(rows) == len(best) and cost < best_cost)):
                    best = rows
                    best_cost = cost
    used_rtm = {int(row["rtmlib_list_idx"]) for row in best}
    used_wilor = {int(row["wilor_list_idx"]) for row in best}
    return best, [i for i in rtm_indices if i not in used_rtm], [i for i in wilor_indices if i not in used_wilor]


def run(args: argparse.Namespace) -> dict:
    rtm = load_json(args.rtmlib_json)
    wilor = load_json(args.wilor_raw)
    rtm_frames = {int(frame["frame_idx"]): frame for frame in rtm["frames"]}
    wilor_frames = {int(frame["frame_idx"]): frame for frame in wilor["frames"]}
    rows = []
    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        rtm_frame = rtm_frames.get(frame_idx)
        wilor_frame = wilor_frames.get(frame_idx)
        if rtm_frame is None or wilor_frame is None:
            continue
        matches, unmatched_rtm, unmatched_wilor = best_assignment(rtm_frame.get("hands", []), wilor_frame.get("raw_hands", []), args)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "rtmlib_hands": int(len(rtm_frame.get("hands", []))),
                "wilor_hands": int(len(wilor_frame.get("raw_hands", []))),
                "matches": matches,
                "unmatched_rtmlib": [int(i) for i in unmatched_rtm],
                "unmatched_wilor": [int(i) for i in unmatched_wilor],
            }
        )
    match_rows = [match for row in rows for match in row["matches"]]
    good_matches = [match for match in match_rows if match["median_keypoint_delta_px"] <= args.good_match_px]
    report = {
        "status": "ok",
        "diagnostic_only": True,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": len(rows),
        "frames_with_match": int(sum(1 for row in rows if row["matches"])),
        "frames_with_good_match": int(sum(1 for row in rows if any(m["median_keypoint_delta_px"] <= args.good_match_px for m in row["matches"]))),
        "matches": len(match_rows),
        "good_matches": len(good_matches),
        "matched_delta_px": summarize([float(match["median_keypoint_delta_px"]) for match in match_rows]),
        "good_match_delta_px": summarize([float(match["median_keypoint_delta_px"]) for match in good_matches]),
        "rtmlib_mean_score_matched": summarize([float(match["rtmlib_mean_score"]) for match in match_rows]),
        "rows_preview": rows[:160],
        "interpretation": (
            "This is a 2D hand-observation diagnostic. Matched RTMLib/WiLoR landmarks support a live hand-keypoint factor; "
            "unmatched or high-delta detections must not be converted into metric contact constraints."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtmlib-json", type=Path, required=True)
    parser.add_argument("--wilor-raw", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--min-rtmlib-keypoint-score", type=float, default=0.2)
    parser.add_argument("--min-matched-keypoints", type=int, default=8)
    parser.add_argument("--good-match-px", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
