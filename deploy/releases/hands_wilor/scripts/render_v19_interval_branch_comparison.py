#!/usr/bin/env python3
"""Render a clear comparison video/contact sheet for V19 interval branches.

This visualization consumes already-rendered overlay/world frames and interval
state JSONs.  It does not create physical state; it makes the physical evidence
and quantitative branch differences readable.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Branch:
    name: str
    render_root: Path
    interval_state: Path
    summary: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_branch(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise RuntimeError("--branch must be NAME=RENDER_ROOT:INTERVAL_STATE")
    name = parts[0].strip()
    rhs = parts[1]
    render_raw, sep, interval_raw = rhs.rpartition(":")
    if not sep:
        raise RuntimeError("--branch must be NAME=RENDER_ROOT:INTERVAL_STATE")
    return name, Path(render_raw), Path(interval_raw)


def metric_value(summary: dict[str, Any], key: str, stat: str = "median") -> float | None:
    value = summary.get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get(stat)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def interval_summary(path: Path, side: str) -> dict[str, Any]:
    payload = load_json(path)
    for row in payload.get("intervals", []) if isinstance(payload.get("intervals"), list) else []:
        if isinstance(row, dict) and str(row.get("interval_id", "")).startswith(f"{side}_"):
            return row
    raise RuntimeError(f"interval state {path} has no {side} interval")


def read_frame(root: Path, kind: str, frame_idx: int, size: tuple[int, int]) -> np.ndarray:
    path = root / kind / f"{frame_idx:06d}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def branch_label(branch: Branch, side: str) -> str:
    s = branch.summary
    gap = metric_value(s, "contact_patch_final_abs_normal_gap_m")
    shift = metric_value(s, "visible_joint_shift_max_px")
    trans = metric_value(s, "translation_delta_norm_m")
    active = s.get("active_set_closed")
    gap_txt = "gap=?" if gap is None else f"gap={gap*1000:.1f}mm"
    shift_txt = "shift=?" if shift is None else f"shift={shift:.1f}px"
    trans_txt = "trans=?" if trans is None else f"trans={trans*1000:.1f}mm"
    closed_txt = f"closed={active}"
    return f"{branch.name} | {side}: {gap_txt} {shift_txt} {trans_txt} {closed_txt}"


def make_tile(branch: Branch, frame_idx: int, side: str, tile_width: int, tile_height: int) -> np.ndarray:
    banner_h = 44
    image_h = tile_height - banner_h
    half_w = tile_width // 2
    overlay = read_frame(branch.render_root, "overlay_frames", frame_idx, (half_w, image_h))
    world = read_frame(branch.render_root, "world_frames", frame_idx, (tile_width - half_w, image_h))
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    tile[banner_h:, :half_w] = overlay
    tile[banner_h:, half_w:] = world
    cv2.rectangle(tile, (0, 0), (tile_width, banner_h), (8, 8, 8), -1)
    cv2.putText(tile, branch_label(branch, side), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(tile, f"frame {frame_idx}", (tile_width - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 230, 255), 2, cv2.LINE_AA)
    cv2.line(tile, (half_w, banner_h), (half_w, tile_height - 1), (80, 80, 80), 1)
    return tile


def build(args: argparse.Namespace) -> dict[str, Any]:
    branches: list[Branch] = []
    for raw in args.branch:
        name, render_root, interval_state = parse_branch(raw)
        if not render_root.exists():
            raise FileNotFoundError(render_root)
        if not interval_state.exists():
            raise FileNotFoundError(interval_state)
        branches.append(Branch(name=name, render_root=render_root, interval_state=interval_state, summary=interval_summary(interval_state, args.side)))
    if not branches:
        raise RuntimeError("at least one --branch is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    frame_w = tile_w
    frame_h = tile_h * len(branches)
    video_path = args.output_dir / "v19_interval_branch_comparison.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    written = 0
    still_paths: list[str] = []
    try:
        for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))):
            tiles = [make_tile(branch, frame_idx, args.side, tile_w, tile_h) for branch in branches]
            canvas = np.vstack(tiles)
            writer.write(canvas)
            written += 1
            if frame_idx in set(int(x) for x in args.still_frames):
                out = args.output_dir / f"comparison_frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                    raise RuntimeError(f"failed to write {out}")
                still_paths.append(str(out))
    finally:
        writer.release()
    report = {
        "status": "ok",
        "method": "render_v19_interval_branch_comparison",
        "claim_scope": "visual comparison of already-rendered physical-state branches; not a state producer",
        "side": args.side,
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "video": str(video_path),
        "stills": still_paths,
        "frames_written": int(written),
        "branches": [
            {
                "name": b.name,
                "render_root": str(b.render_root),
                "interval_state": str(b.interval_state),
                "summary": {
                    "contact_patch_final_abs_normal_gap_m": b.summary.get("contact_patch_final_abs_normal_gap_m"),
                    "visible_joint_shift_max_px": b.summary.get("visible_joint_shift_max_px"),
                    "translation_delta_norm_m": b.summary.get("translation_delta_norm_m"),
                    "active_set_closed": b.summary.get("active_set_closed"),
                },
            }
            for b in branches
        ],
    }
    write_json(args.output_dir / "v19_interval_branch_comparison_report.json", report)
    print(json.dumps(report, indent=2)[:20000])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", action="append", required=True, help="NAME=RENDER_ROOT:INTERVAL_STATE")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-width", type=int, default=1280)
    parser.add_argument("--tile-height", type=int, default=300)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--still-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
