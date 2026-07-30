#!/usr/bin/env python3
"""Render WiLoR hand candidate overlay from full-frame raw hand JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_codec_utils import H264VideoWriter

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return payload


def resolve_rgb(run_root: Path, repo_root: Path, frame_row: dict[str, Any]) -> Path | None:
    fidx = int(frame_row["frame_idx"])
    candidates = [run_root / f"input/source_frame_manifest/rgb/{fidx:06d}.jpg"]
    if frame_row.get("rgb"):
        raw = Path(str(frame_row["rgb"]))
        candidates.extend([raw if raw.is_absolute() else repo_root / raw, run_root / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def draw_hand(image: np.ndarray, hand: dict[str, Any], hand_idx: int, scale_xy: tuple[float, float]) -> None:
    sx, sy = scale_xy
    color = (80, 220, 255) if str(hand.get("side", "")).lower() == "right" else (255, 160, 80)
    bbox = hand.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(v) * (sx if i % 2 == 0 else sy))) for i, v in enumerate(bbox)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        score = hand.get("detector_score")
        label = f"WiLoR {hand.get('side','?')} {float(score):.2f}" if score is not None else f"WiLoR {hand.get('side','?')}"
        cv2.putText(image, label, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    joints = hand.get("joints2d")
    if isinstance(joints, list) and len(joints) >= 21:
        pts = []
        for joint in joints[:21]:
            if not isinstance(joint, list) or len(joint) < 2:
                pts.append(None)
                continue
            x, y = float(joint[0]), float(joint[1])
            if not np.isfinite([x, y]).all():
                pts.append(None)
            else:
                pts.append((int(round(x * sx)), int(round(y * sy))))
        for a, b in EDGES:
            if pts[a] is not None and pts[b] is not None:
                cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for pt in pts:
            if pt is not None:
                cv2.circle(image, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 2, color, -1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    raw = load_json(Path(args.raw_hands))
    manifest = load_json(run_root / "input/raw_frame_manifest/manifest.json")
    by_frame = {int(row["frame_idx"]): row.get("raw_hands", []) for row in raw.get("frames", []) if isinstance(row, dict) and row.get("frame_idx") is not None}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    review_dir = Path(args.review_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    review_frames = {int(v) for v in args.review_frames}
    fps = float(manifest.get("fps", 30.0) or 30.0)
    writer = None
    frames_written = 0
    frames_with_hands = 0
    total_hands = 0
    try:
        for frame_row in manifest.get("frames", []):
            if not isinstance(frame_row, dict) or frame_row.get("frame_idx") is None:
                continue
            fidx = int(frame_row["frame_idx"])
            rgb_path = resolve_rgb(run_root, repo_root, frame_row)
            if rgb_path is None:
                continue
            image = cv2.imread(str(rgb_path))
            if image is None:
                continue
            hands = by_frame.get(fidx, [])
            if hands:
                frames_with_hands += 1
                total_hands += len(hands)
            coord_w = float(frame_row.get("manifest_width") or image.shape[1])
            coord_h = float(frame_row.get("manifest_height") or image.shape[0])
            scale_xy = (float(image.shape[1]) / max(1.0, coord_w), float(image.shape[0]) / max(1.0, coord_h))
            for hand_idx, hand in enumerate(hands):
                if isinstance(hand, dict):
                    draw_hand(image, hand, hand_idx, scale_xy)
            cv2.putText(image, f"WiLoR frame {fidx}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, f"WiLoR frame {fidx}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = H264VideoWriter(output, fps, (image.shape[1], image.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open writer: {output}")
            writer.write(image)
            if fidx in review_frames:
                cv2.imwrite(str(review_dir / f"frame_{fidx:06d}.jpg"), image)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    if frames_written == 0:
        raise RuntimeError("no frames written")
    qc = {
        "status": "ok",
        "method": "render_v22_wilor_hand_overlay",
        "frames_written": frames_written,
        "frames_with_hands": frames_with_hands,
        "total_hands": total_hands,
        "overlay_video": str(output),
        "review_stills": str(review_dir),
        "claim_scope": "Visualization of WiLoR raw MANO candidate detections; not an accepted optimized V22 hand state.",
    }
    qc_path = output.parent / "wilor_overlay_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--raw-hands", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--review-dir", required=True)
    ap.add_argument("--review-frames", type=int, nargs="*", default=[])
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
