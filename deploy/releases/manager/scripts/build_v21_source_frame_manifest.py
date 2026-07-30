#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    video = Path(str(input_manifest["primary_video"]))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0 or fps <= 0:
        cap.release()
        raise ContractError(f"invalid_video_metadata: {video}")
    rgb_dir = args.output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    try:
        for frame_idx in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: frame={frame_idx}")
            path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise ContractError(f"could_not_write_frame: {path}")
            frames.append(
                {
                    "index": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx / fps),
                    "rgb": str(path),
                    "source_width": int(width),
                    "source_height": int(height),
                    "manifest_width": int(width),
                    "manifest_height": int(height),
                    "coordinate_semantics": "source_video_pixel_coordinates",
                }
            )
    finally:
        cap.release()
    manifest = {
        "schema": "v21_source_frame_manifest.v0",
        "status": "ok",
        "method": "build_v21_source_frame_manifest",
        "input_manifest": str(args.input_manifest),
        "clip": str(video),
        "video": {"fps": fps, "width": width, "height": height, "frame_count": frame_count, "duration_s": float(frame_count / fps)},
        "frames": frames,
        "claim_scope": "Source-resolution decoded RGB frames for algorithms whose prompts are in source pixel coordinates. This is an input adapter, not a physical measurement.",
    }
    write_json(args.output_manifest, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2, ensure_ascii=False))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-resolution frame manifest for V21 image-space algorithms.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
