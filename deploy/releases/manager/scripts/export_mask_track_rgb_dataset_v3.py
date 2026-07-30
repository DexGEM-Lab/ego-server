#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx}")
    return frame


def run(args: argparse.Namespace) -> dict:
    track = load_json(args.mask_track)
    clip = cv2.VideoCapture(str(args.clip))
    if not clip.isOpened():
        raise RuntimeError(f"failed to open clip: {args.clip}")
    width = int(clip.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(clip.get(cv2.CAP_PROP_FPS))
    frame_count = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb"
    mask_dir = args.output_dir / "masks"
    rgb_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    rows = []
    skipped = []
    try:
        for raw_frame, row in sorted(track.items(), key=lambda item: int(item[0])):
            frame_idx = int(raw_frame)
            if args.frame_start is not None and frame_idx < args.frame_start:
                continue
            if args.frame_end is not None and frame_idx > args.frame_end:
                continue
            if not row.get("visible"):
                skipped.append({"frame_idx": frame_idx, "reason": "track_visible_false"})
                continue
            mask_path = Path(str(row.get("mask_path", "")))
            if not mask_path.exists():
                raise RuntimeError(f"missing mask for frame {frame_idx}: {mask_path}")
            frame = read_frame(clip, frame_idx)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read mask: {mask_path}")
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0
            mask_pixels = int(mask_bool.sum())
            if mask_pixels < int(args.min_mask_pixels):
                skipped.append({"frame_idx": frame_idx, "reason": "too_few_mask_pixels", "mask_pixels": mask_pixels})
                continue
            index = len(rows)
            rgb_path = rgb_dir / f"{index:06d}.jpg"
            out_mask_path = mask_dir / f"{index:06d}.png"
            if not cv2.imwrite(str(rgb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write {rgb_path}")
            if not cv2.imwrite(str(out_mask_path), mask_bool.astype("uint8") * 255):
                raise RuntimeError(f"failed to write {out_mask_path}")
            rows.append(
                {
                    "index": index,
                    "frame_idx": frame_idx,
                    "rgb": str(rgb_path),
                    "mask": str(out_mask_path),
                    "mask_pixels": mask_pixels,
                    "selection": row.get("selection"),
                    "candidate": row.get("candidate"),
                }
            )
    finally:
        clip.release()
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} frames exported, min_frames={args.min_frames}; skipped={skipped}")
    manifest = {
        "status": "ok",
        "backend": "export_mask_track_rgb_dataset_v3",
        "clip": str(args.clip),
        "mask_track": str(args.mask_track),
        "dataset_dir": str(args.output_dir),
        "video": {"width": width, "height": height, "fps": fps, "frame_count": frame_count},
        "frames": rows,
        "skipped": skipped,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path), "frames": len(rows), "skipped": len(skipped)}, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
