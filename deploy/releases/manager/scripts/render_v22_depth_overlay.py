#!/usr/bin/env python3
"""Render a full-video color depth overlay from a depth NPZ and run-root frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_codec_utils import H264VideoWriter


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    manifest = load_json(run_root / "input/raw_frame_manifest/manifest.json")
    data = np.load(str(args.depth_npz))
    depth = np.asarray(data["depth"], dtype=np.float32)
    sample = depth[:: max(1, len(depth) // 120)]
    valid = sample[np.isfinite(sample) & (sample > 0.1)]
    if valid.size == 0:
        raise RuntimeError("no valid depth values")
    lo = float(np.percentile(valid, args.low_percentile))
    hi = float(np.percentile(valid, args.high_percentile))
    if hi <= lo:
        hi = lo + 1.0
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames_written = 0
    fps = float(manifest.get("fps", 30.0) or 30.0)
    try:
        for row in manifest.get("frames", []):
            if not isinstance(row, dict) or row.get("frame_idx") is None:
                continue
            fidx = int(row["frame_idx"])
            rgb_path = resolve_rgb(run_root, repo_root, row)
            if rgb_path is None or fidx >= len(depth):
                continue
            image = cv2.imread(str(rgb_path))
            if image is None:
                continue
            d = depth[fidx]
            if d.shape[:2] != image.shape[:2]:
                d = cv2.resize(d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            norm = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
            color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            valid_mask = np.isfinite(d) & (d > 0.1)
            overlay = image.copy()
            overlay[valid_mask] = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)[valid_mask]
            cv2.putText(overlay, f"UniDepth v2 frame {fidx} depth_m [{lo:.2f},{hi:.2f}]", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, f"UniDepth v2 frame {fidx} depth_m [{lo:.2f},{hi:.2f}]", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = H264VideoWriter(output, fps, (overlay.shape[1], overlay.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"could not open writer: {output}")
            writer.write(overlay)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    qc = {"status": "ok", "method": "render_v22_depth_overlay", "frames_written": frames_written, "output_video": str(output), "depth_npz": str(args.depth_npz), "depth_window_m": [lo, hi]}
    (output.parent / "unidepth_v2_depth_overlay_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--depth-npz", required=True, type=Path)
    ap.add_argument("--output", required=True)
    ap.add_argument("--low-percentile", type=float, default=1.0)
    ap.add_argument("--high-percentile", type=float, default=99.0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
