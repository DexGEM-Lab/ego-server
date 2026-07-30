#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def run(args: argparse.Namespace) -> dict:
    if args.frame_end < args.frame_start:
        raise RuntimeError(f"invalid frame interval {args.frame_start}:{args.frame_end}")
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open clip: {args.clip}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid clip metadata for {args.clip}")
    if args.frame_end >= frame_count:
        raise RuntimeError(f"frame_end {args.frame_end} exceeds clip frame count {frame_count}")
    for subdir in ("rgb", "masks"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    mask = np.full((height, width), 255, dtype=np.uint8)
    entries = []
    try:
        for out_i, frame_idx in enumerate(range(int(args.frame_start), int(args.frame_end) + 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to read frame {frame_idx}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"frame {frame_idx} shape {frame.shape[:2]} does not match clip metadata {(height, width)}")
            stem = f"{out_i:06d}"
            rgb_path = args.output_dir / "rgb" / f"{stem}.png"
            mask_path = args.output_dir / "masks" / f"{stem}.png"
            if not cv2.imwrite(str(rgb_path), frame):
                raise RuntimeError(f"failed to write {rgb_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"failed to write {mask_path}")
            entries.append(
                {
                    "index": int(out_i),
                    "frame_idx": int(frame_idx),
                    "rgb": str(rgb_path),
                    "mask": str(mask_path),
                    "mask_area_px": int(mask.size),
                    "track_id": "full_scene",
                    "label": "full scene",
                }
            )
    finally:
        cap.release()
    if len(entries) != int(args.frame_end) - int(args.frame_start) + 1:
        raise RuntimeError("exported frame count mismatch")
    manifest = {"frames": entries}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "method": "full_scene_vggt_dataset_export_v3",
        "clip": str(args.clip),
        "output_dir": str(args.output_dir),
        "manifest": str(args.output_dir / "manifest.json"),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(entries)),
        "source_size": [width, height],
        "fps": fps,
        "full_scene_mask_area_px": int(mask.size),
    }
    (args.output_dir / "qc_full_scene_vggt_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
