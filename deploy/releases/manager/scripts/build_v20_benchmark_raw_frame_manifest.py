#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, write_json


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.dataset_manifest)
    ensure_no_gt_in_prediction(manifest, "dataset_manifest")
    frames_in = manifest.get("frames") if isinstance(manifest, dict) else None
    if not isinstance(frames_in, list) or not frames_in:
        raise ContractError("dataset_manifest_has_no_frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    first_size = None
    for index, row in enumerate(frames_in):
        if not isinstance(row, dict):
            raise ContractError(f"dataset_manifest_frame_not_object: {index}")
        rgb_path = Path(str(row.get("rgb_path") or row.get("rgb") or ""))
        if not rgb_path.exists():
            raise ContractError(f"missing_prediction_rgb_frame: {rgb_path}")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ContractError(f"could_not_read_prediction_rgb_frame: {rgb_path}")
        height, width = image.shape[:2]
        size = (width, height)
        if first_size is None:
            first_size = size
        elif size != first_size:
            raise ContractError(f"benchmark_rgb_resolution_mismatch: {rgb_path} {size} expected={first_size}")
        frame_idx = int(row.get("frame_index", row.get("frame_idx", index)))
        frames.append({
            "index": int(index),
            "frame_idx": frame_idx,
            "source_frame_idx": frame_idx,
            "time_s": float(index / float(args.fps or manifest.get("fps_assumed", 10.0))),
            "rgb": str(rgb_path),
            "rgb_path": str(rgb_path),
            "depth_path": row.get("depth_path"),
            "source_width": int(width),
            "source_height": int(height),
            "manifest_width": int(width),
            "manifest_height": int(height),
        })
    output = {
        "schema": "v20_benchmark_raw_frame_manifest.v0",
        "status": "ok",
        "method": "build_v20_benchmark_raw_frame_manifest",
        "dataset_manifest": str(args.dataset_manifest),
        "dataset": manifest.get("dataset"),
        "sample_id": manifest.get("sample_id"),
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "frame_count": len(frames),
        "fps": float(args.fps or manifest.get("fps_assumed", 10.0)),
        "video": {
            "frame_count": len(frames),
            "fps": float(args.fps or manifest.get("fps_assumed", 10.0)),
            "width": int(first_size[0]) if first_size else None,
            "height": int(first_size[1]) if first_size else None,
        },
        "frames": frames,
        "evaluation_reference_policy": "Eval refs are not read by this prediction-side raw-frame manifest builder.",
    }
    ensure_no_gt_in_prediction(output, "benchmark_raw_frame_manifest")
    write_json(args.output_manifest, output)
    summary = {
        "status": "ok",
        "method": "build_v20_benchmark_raw_frame_manifest",
        "output_manifest": str(args.output_manifest),
        "frame_count": len(frames),
        "eval_refs_loaded": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V20 benchmark raw-frame manifest from prediction-side dataset RGB rows.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
