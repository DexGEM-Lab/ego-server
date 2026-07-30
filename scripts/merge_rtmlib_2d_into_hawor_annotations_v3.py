#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from diagnose_hand_reprojection_depth_v3 import project_points


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
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


def rtmlib_by_frame(path: Path) -> dict[int, list[dict]]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{path} must contain a frames list")
    return {int(frame["frame_idx"]): list(frame.get("hands", [])) for frame in frames}


def hand_projection(hand: dict) -> np.ndarray:
    joints = np.asarray(hand.get("joints3d_source_camera_m", []), dtype=float)
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    if joints.shape != (21, 3) or intr.shape != (4,):
        raise RuntimeError("hand is missing source-camera joints or intrinsics")
    return project_points(joints, intr)


def detection_points(det: dict) -> np.ndarray:
    pts = np.asarray(det.get("keypoints", []), dtype=float)
    if pts.shape != (21, 2):
        raise RuntimeError("RTMLib detection has invalid keypoints")
    return pts


def detection_scores(det: dict) -> np.ndarray:
    scores = np.asarray(det.get("scores", []), dtype=float)
    if scores.shape != (21,):
        raise RuntimeError("RTMLib detection has invalid scores")
    return scores


def match_frame(frame: dict, detections: list[dict], args: argparse.Namespace) -> list[dict]:
    hands = list(frame.get("hands", []))
    if not hands or not detections:
        for hand in hands:
            hand["measurement_available"] = False
            hand["filter_status"] = f"{hand.get('filter_status', 'hawor')}_no_rtmlib_match"
        return []

    cost = np.full((len(hands), len(detections)), float(args.unmatched_cost_px), dtype=float)
    projections = [hand_projection(hand) for hand in hands]
    det_points = [detection_points(det) for det in detections]
    det_scores = [detection_scores(det) for det in detections]
    for i, proj in enumerate(projections):
        for j, pts in enumerate(det_points):
            valid = np.isfinite(pts).all(axis=1) & np.isfinite(det_scores[j]) & (det_scores[j] >= float(args.min_keypoint_score))
            if int(np.count_nonzero(valid)) < int(args.min_keypoints):
                continue
            cost[i, j] = float(np.median(np.linalg.norm(proj[valid] - pts[valid], axis=1)))

    rows, cols = linear_sum_assignment(cost)
    matched: set[int] = set()
    match_rows = []
    for i, j in zip(rows, cols):
        value = float(cost[i, j])
        if value > float(args.max_match_median_px):
            continue
        hand = hands[i]
        det = detections[j]
        pts = det_points[j]
        scores = det_scores[j]
        valid = np.isfinite(pts).all(axis=1) & np.isfinite(scores) & (scores >= float(args.min_keypoint_score))
        raw = pts.copy()
        raw[~valid] = projections[i][~valid]
        hand["measurement_available"] = True
        hand["joints2d_raw"] = raw.astype(float).tolist()
        hand["rtmlib_scores"] = scores.astype(float).tolist()
        hand["detector_score"] = float(det.get("mean_score", 0.0))
        hand["bbox_xyxy"] = [float(v) for v in det.get("bbox_xyxy", [])]
        hand["filter_status"] = f"{hand.get('filter_status', 'hawor')}_rtmlib_2d_matched"
        hand["rtmlib_match"] = {
            "hand_idx": int(det.get("hand_idx", j)),
            "median_projection_delta_px": value,
            "valid_keypoints": int(np.count_nonzero(valid)),
        }
        matched.add(i)
        match_rows.append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "side": str(hand.get("side", "unknown")),
                "rtmlib_hand_idx": int(det.get("hand_idx", j)),
                "median_projection_delta_px": value,
                "valid_keypoints": int(np.count_nonzero(valid)),
                "mean_score": float(det.get("mean_score", 0.0)),
            }
        )

    for i, hand in enumerate(hands):
        if i in matched:
            continue
        hand["measurement_available"] = False
        hand["filter_status"] = f"{hand.get('filter_status', 'hawor')}_unmatched_rtmlib"
    return match_rows


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    detections_by_frame = rtmlib_by_frame(args.rtmlib_json)
    rows = []
    for frame in annotations.get("frames", []):
        frame_idx = int(frame["frame_idx"])
        if frame_idx < args.frame_start or frame_idx > args.frame_end:
            continue
        rows.extend(match_frame(frame, detections_by_frame.get(frame_idx, []), args))
    save_json(args.output_annotations, annotations)
    report = {
        "status": "ok",
        "method": "merge_rtmlib_2d_into_hawor_annotations_v3",
        "annotations": str(args.annotations),
        "rtmlib_json": str(args.rtmlib_json),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "matched_hands": int(len(rows)),
        "median_projection_delta_px": summarize([float(row["median_projection_delta_px"]) for row in rows]),
        "mean_score": summarize([float(row["mean_score"]) for row in rows]),
        "rows_preview": rows[:180],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows_preview"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rtmlib-json", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-keypoints", type=int, default=8)
    parser.add_argument("--min-keypoint-score", type=float, default=0.30)
    parser.add_argument("--max-match-median-px", type=float, default=160.0)
    parser.add_argument("--unmatched-cost-px", type=float, default=1e6)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
