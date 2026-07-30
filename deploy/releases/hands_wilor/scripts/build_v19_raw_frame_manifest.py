#!/usr/bin/env python3
"""Build a V19-owned raw-frame manifest directly from an input video.

This is the first executable component of the V19 pipeline. It extracts the
source timeline into one row per source frame under the current run root. It has
no dependency on V16/V17/V18 cached artifacts.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return float(self.frame_count / self.fps) if self.fps > 0 else 0.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open input video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0.0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata for {path}: {info}")
    return cap, info


def even_height_for_width(width: int, source_width: int, source_height: int) -> int:
    height = int(round(float(width) * float(source_height) / float(source_width)))
    if height % 2:
        height += 1
    return max(2, height)


def build(args: argparse.Namespace) -> dict:
    started = time.time()
    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(f"missing input video: {video}")

    cap, info = open_video(video)
    if args.render_width is None or int(args.render_width) <= 0:
        manifest_width = int(info.width)
        manifest_height = int(info.height)
    else:
        manifest_width = int(args.render_width)
        manifest_height = even_height_for_width(manifest_width, info.width, info.height)

    frame_start = 0 if args.frame_start is None else int(args.frame_start)
    frame_end = int(info.frame_count) - 1 if args.frame_end is None else int(args.frame_end)
    if frame_start < 0 or frame_end < frame_start or frame_end >= int(info.frame_count):
        cap.release()
        raise RuntimeError(
            f"invalid frame range {frame_start}:{frame_end} for {info.frame_count}-frame video"
        )

    output_dir = args.output_dir.resolve()
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict] = []
    extracted = 0
    try:
        for frame_idx in range(int(info.frame_count)):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {frame_idx}, expected {info.frame_count}")
            if frame_idx < frame_start or frame_idx > frame_end:
                continue
            if manifest_width != int(info.width) or manifest_height != int(info.height):
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (manifest_width, manifest_height),
                    interpolation=cv2.INTER_AREA,
                )
            path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f"failed to write frame image: {path}")
            frames.append(
                {
                    "index": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx / info.fps),
                    "rgb": str(path),
                    "raw_frame_path": str(path),
                    "source_width": int(info.width),
                    "source_height": int(info.height),
                    "manifest_width": int(manifest_width),
                    "manifest_height": int(manifest_height),
                    "source_video": str(video),
                }
            )
            extracted += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("no frames extracted")

    manifest = {
        "status": "ok",
        "method": "build_v19_raw_frame_manifest",
        "claim_scope": "fresh V19 source-timeline extraction; frame rows are measurements of video timing and image paths only",
        "input_video": str(video),
        "video": {**asdict(info), "duration_s": info.duration_s},
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_count": int(extracted),
        "full_source_timeline": bool(frame_start == 0 and frame_end == int(info.frame_count) - 1),
        "manifest_width": int(manifest_width),
        "manifest_height": int(manifest_height),
        "rgb_dir": str(rgb_dir),
        "frames": frames,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    report = {
        "status": "ok",
        "method": "build_v19_raw_frame_manifest",
        "input_video": str(video),
        "manifest": str(manifest_path),
        "rgb_dir": str(rgb_dir),
        "source_frame_count": int(info.frame_count),
        "extracted_frame_count": int(extracted),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "full_source_timeline": bool(manifest["full_source_timeline"]),
        "source_width": int(info.width),
        "source_height": int(info.height),
        "manifest_width": int(manifest_width),
        "manifest_height": int(manifest_height),
        "fps": float(info.fps),
        "elapsed_s": float(time.time() - started),
    }
    write_json(output_dir / "v19_raw_frame_manifest_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output raw_frame_manifest directory")
    parser.add_argument("--render-width", type=int, default=960, help="Extracted frame width; <=0 preserves source resolution")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
