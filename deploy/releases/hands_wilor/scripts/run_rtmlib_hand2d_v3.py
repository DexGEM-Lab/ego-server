#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def require_rtmlib():
    try:
        from rtmlib import Hand, PoseTracker
    except Exception as exc:
        raise RuntimeError("rtmlib is required for this runner; install rtmlib and onnxruntime-gpu in the remote env") from exc
    return Hand, PoseTracker


def to_hands(keypoints: np.ndarray, scores: np.ndarray, min_points: int) -> list[dict]:
    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if keypoints.size == 0 and scores.size == 0:
        return []
    if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 2):
        raise RuntimeError(f"RTMLib returned invalid keypoint shape {keypoints.shape}")
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim != 2 or scores.shape != keypoints.shape[:2]:
        raise RuntimeError(f"RTMLib returned invalid score shape {scores.shape}")
    hands = []
    for hand_idx, (pts, conf) in enumerate(zip(keypoints, scores)):
        finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
        valid = finite & (conf > 0.0)
        if int(np.count_nonzero(valid)) < min_points:
            continue
        valid_pts = pts[valid]
        hands.append(
            {
                "hand_idx": int(hand_idx),
                "keypoints": pts.tolist(),
                "scores": conf.tolist(),
                "valid_keypoints": int(np.count_nonzero(valid)),
                "mean_score": float(np.mean(conf[valid])),
                "median_score": float(np.median(conf[valid])),
                "bbox_xyxy": [
                    float(np.min(valid_pts[:, 0])),
                    float(np.min(valid_pts[:, 1])),
                    float(np.max(valid_pts[:, 0])),
                    float(np.max(valid_pts[:, 1])),
                ],
            }
        )
    return hands


def draw_hands(frame: np.ndarray, hands: list[dict], score_thr: float) -> np.ndarray:
    out = frame.copy()
    colors = [(60, 220, 255), (255, 120, 80), (120, 255, 120), (220, 120, 255)]
    for hand in hands:
        pts = np.asarray(hand["keypoints"], dtype=float)
        scores = np.asarray(hand["scores"], dtype=float)
        color = colors[int(hand["hand_idx"]) % len(colors)]
        for a, b in HAND_EDGES:
            if scores[a] < score_thr or scores[b] < score_thr:
                continue
            pa = tuple(np.round(pts[a]).astype(int))
            pb = tuple(np.round(pts[b]).astype(int))
            cv2.line(out, pa, pb, color, 2, cv2.LINE_AA)
        for i, (pt, score) in enumerate(zip(pts, scores)):
            if score < score_thr:
                continue
            radius = 4 if i in {4, 8, 12, 16, 20} else 3
            cv2.circle(out, tuple(np.round(pt).astype(int)), radius, color, -1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(round(x)) for x in hand["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"RTM hand {hand['hand_idx']} {hand['mean_score']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def compare_to_wilor(frame_idx: int, hands: list[dict], wilor_frames: dict[int, dict] | None) -> list[dict]:
    if wilor_frames is None:
        return []
    wilor_frame = wilor_frames.get(frame_idx)
    if wilor_frame is None:
        return []
    raw_hands = wilor_frame.get("raw_hands", [])
    rows = []
    for rtm_hand in hands:
        rtm = np.asarray(rtm_hand["keypoints"], dtype=float)
        rtm_scores = np.asarray(rtm_hand["scores"], dtype=float)
        for wi, wh in enumerate(raw_hands):
            raw = np.asarray(wh.get("joints2d_raw", []), dtype=float)
            if raw.shape != (21, 2):
                continue
            valid = np.isfinite(rtm).all(axis=1) & np.isfinite(raw).all(axis=1) & np.isfinite(rtm_scores) & (rtm_scores > 0.2)
            if np.count_nonzero(valid) < 8:
                continue
            delta = np.linalg.norm(rtm[valid] - raw[valid], axis=1)
            rows.append(
                {
                    "rtmlib_hand_idx": int(rtm_hand["hand_idx"]),
                    "wilor_hand_idx": int(wi),
                    "wilor_side": wh.get("side"),
                    "wilor_score": float(wh.get("detector_score", np.nan)),
                    "matched_keypoints": int(np.count_nonzero(valid)),
                    "median_keypoint_delta_px": float(np.median(delta)),
                    "p95_keypoint_delta_px": float(np.percentile(delta, 95.0)),
                }
            )
    return rows


def load_wilor(path: Path | None) -> dict[int, dict] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(frame["frame_idx"]): frame for frame in data["frames"]}


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


def run(args: argparse.Namespace) -> dict:
    Hand, PoseTracker = require_rtmlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "review_stills"
    review_dir.mkdir(exist_ok=True)
    wilor_frames = load_wilor(args.wilor_raw)

    tracker = PoseTracker(
        Hand,
        det_frequency=args.det_frequency,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
        to_openpose=False,
    )
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame_start < 0 or args.frame_end >= frame_count or args.frame_start > args.frame_end:
        raise RuntimeError(f"invalid frame window {args.frame_start}-{args.frame_end} for {frame_count} frames")

    overlay_path = args.output_dir / "rtmlib_hand2d_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {overlay_path}")

    source_frame_offset = int(args.source_frame_offset)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_start)
    frames = []
    comparisons = []
    started = time.time()
    for frame_idx in range(args.frame_start, args.frame_end + 1):
        source_frame_idx = int(frame_idx + source_frame_offset)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_idx}")
        keypoints, scores = tracker(frame)
        hands = to_hands(keypoints, scores, args.min_keypoints)
        frame_comparisons = compare_to_wilor(source_frame_idx, hands, wilor_frames)
        comparisons.extend({"frame_idx": source_frame_idx, **row} for row in frame_comparisons)
        rendered = draw_hands(frame, hands, args.draw_score_thr)
        writer.write(rendered)
        if source_frame_idx in args.review_frames:
            cv2.imwrite(str(review_dir / f"frame_{source_frame_idx:06d}.jpg"), rendered)
        frames.append(
            {
                "frame_idx": int(source_frame_idx),
                "local_frame_idx": int(frame_idx),
                "time_s": float(source_frame_idx / fps),
                "hands": hands,
                "wilor_comparisons": frame_comparisons,
            }
        )
    writer.release()
    cap.release()

    detected = [len(frame["hands"]) for frame in frames]
    score_values = [hand["mean_score"] for frame in frames for hand in frame["hands"]]
    delta_values = [row["median_keypoint_delta_px"] for row in comparisons]
    output_json = args.output_dir / "rtmlib_hand2d.json"
    output_json.write_text(
        json.dumps(
            {
                "clip": str(args.clip),
                "video": {"fps": fps, "width": width, "height": height, "frame_count": frame_count},
                "frame_start": int(args.frame_start + source_frame_offset),
                "frame_end": int(args.frame_end + source_frame_offset),
                "local_frame_start": int(args.frame_start),
                "local_frame_end": int(args.frame_end),
                "source_frame_offset": int(source_frame_offset),
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    qc = {
        "status": "ok",
        "clip": str(args.clip),
        "frame_start": int(args.frame_start + source_frame_offset),
        "frame_end": int(args.frame_end + source_frame_offset),
        "local_frame_start": int(args.frame_start),
        "local_frame_end": int(args.frame_end),
        "source_frame_offset": int(source_frame_offset),
        "processed_frames": len(frames),
        "frames_with_hands": int(sum(1 for n in detected if n > 0)),
        "hand_detection_rate": float(sum(1 for n in detected if n > 0) / max(1, len(frames))),
        "hands_per_frame": summarize([float(n) for n in detected]),
        "hand_mean_score": summarize(score_values),
        "wilor_keypoint_delta_px": summarize(delta_values),
        "elapsed_s": float(time.time() - started),
        "output_json": str(output_json),
        "overlay_video": str(overlay_path),
        "review_stills": str(review_dir),
        "runtime": {"backend": args.backend, "device": args.device, "mode": args.mode, "det_frequency": args.det_frequency},
    }
    (args.output_dir / "qc_rtmlib_hand2d.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--source-frame-offset", type=int, default=0)
    parser.add_argument("--wilor-raw", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--mode", default="lightweight", choices=["lightweight"])
    parser.add_argument("--det-frequency", type=int, default=1)
    parser.add_argument("--min-keypoints", type=int, default=8)
    parser.add_argument("--draw-score-thr", type=float, default=0.3)
    parser.add_argument("--review-frames", type=int, nargs="*", default=[840, 858, 875, 880, 903, 930])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
