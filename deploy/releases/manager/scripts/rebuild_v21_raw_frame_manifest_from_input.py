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


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {path}")
    meta = {
        "path": str(path),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if meta["fps"] <= 0 or meta["width"] <= 0 or meta["height"] <= 0 or meta["frame_count"] <= 0:
        raise ContractError(f"invalid_video_metadata: {path} {meta}")
    meta["duration_s"] = float(meta["frame_count"] / meta["fps"])
    return meta


def even_height(width: int, source_width: int, source_height: int) -> int:
    height = int(round(float(width) * float(source_height) / float(source_width)))
    return height + height % 2


def existing_frame_extras(existing_manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    keep = {}
    for row in existing_manifest.get("frames", []) if isinstance(existing_manifest.get("frames"), list) else []:
        if not isinstance(row, dict) or row.get("frame_idx") is None:
            continue
        extras = {k: v for k, v in row.items() if k not in {
            "index", "frame_idx", "source_frame_idx", "time_s", "source_time_s", "rgb", "raw_frame_path",
            "source_width", "source_height", "manifest_width", "manifest_height", "source_video",
        }}
        keep[int(row["frame_idx"])] = extras
    return keep


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    video = Path(str(input_manifest["primary_video"]))
    meta = video_metadata(video)
    existing = load_json(args.output_manifest) if args.output_manifest.exists() else {}
    if args.render_width and args.render_width > 0:
        render_width = int(args.render_width)
    elif existing.get("render_width"):
        render_width = int(existing["render_width"])
    elif existing.get("manifest_width"):
        render_width = int(existing["manifest_width"])
    elif isinstance(existing.get("frames"), list) and existing["frames"]:
        render_width = int(existing["frames"][0].get("manifest_width") or 960)
    else:
        render_width = 960
    render_height = even_height(render_width, int(meta["width"]), int(meta["height"]))
    source_start = int((input_manifest.get("source_span") or {}).get("start_frame") or 0)
    extras = existing_frame_extras(existing)
    rgb_dir = args.output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ContractError(f"could_not_open_video: {video}")
    frames = []
    try:
        for frame_idx in range(int(meta["frame_count"])):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: {video} frame={frame_idx}")
            resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            frame_path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise ContractError(f"could_not_write_frame: {frame_path}")
            row = {
                "index": int(frame_idx),
                "frame_idx": int(frame_idx),
                "source_frame_idx": int(source_start + frame_idx),
                "time_s": float(frame_idx / meta["fps"]),
                "source_time_s": float((source_start + frame_idx) / meta["fps"]),
                "rgb": str(frame_path),
                "raw_frame_path": str(frame_path),
                "source_width": int(meta["width"]),
                "source_height": int(meta["height"]),
                "manifest_width": int(render_width),
                "manifest_height": int(render_height),
                "coordinate_semantics": "raw_manifest_pixel_coordinates",
            }
            row.update(extras.get(frame_idx, {}))
            frames.append(row)
    finally:
        cap.release()
    manifest = {
        "schema": "v21_raw_frame_manifest.v0",
        "status": "ok",
        "method": "rebuild_v21_raw_frame_manifest_from_input",
        "input_manifest": str(args.input_manifest),
        "clip": str(video),
        "video": meta,
        "render_width": int(render_width),
        "render_height": int(render_height),
        "frames": frames,
        "claim_scope": "Decoded raw-frame timeline from the current input manifest primary video. This is an input/timeline atom only, not a physical measurement.",
    }
    write_json(args.output_manifest, manifest)
    report = {k: v for k, v in manifest.items() if k != "frames"}
    report["frame_count"] = len(frames)
    write_json(args.output_dir / "raw_frame_manifest_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild only the V21 raw-frame manifest atom from an existing input manifest.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    return parser.parse_args()


if __name__ == "__main__":
    rebuild(parse_args())
