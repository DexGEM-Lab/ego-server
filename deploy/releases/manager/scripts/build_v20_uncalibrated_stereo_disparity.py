#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, numeric_summary, write_json


def video_meta(path: Path) -> dict[str, Any]:
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
    return meta


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"raw_frame_manifest_has_no_frames: {path}")
    return payload


def resize_pair(left: np.ndarray, right: np.ndarray, max_width: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = left.shape[:2]
    scale = min(1.0, float(max_width) / float(width))
    out_w = max(16, int(round(width * scale / 2.0)) * 2)
    out_h = max(16, int(round(height * scale / 2.0)) * 2)
    left_small = cv2.resize(left, (out_w, out_h), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return left_small, right_small


def normalize_preview(relative_inverse_depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(relative_inverse_depth) & (relative_inverse_depth > 0)
    preview = np.zeros(relative_inverse_depth.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(relative_inverse_depth[valid], [2.0, 98.0])
        if hi > lo:
            preview[valid] = np.clip((relative_inverse_depth[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.raw_frame_manifest)
    frames = manifest["frames"]
    left_meta = video_meta(args.left_video)
    right_meta = video_meta(args.right_video)
    cap_right = cv2.VideoCapture(str(args.right_video))
    if not cap_right.isOpened():
        raise ContractError(f"could_not_open_right_video: {args.right_video}")
    num_disparities = int(args.num_disparities)
    num_disparities = max(16, (num_disparities // 16) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=int(args.block_size),
        P1=8 * 3 * int(args.block_size) ** 2,
        P2=32 * 3 * int(args.block_size) ** 2,
        uniquenessRatio=int(args.uniqueness_ratio),
        speckleWindowSize=80,
        speckleRange=2,
        disp12MaxDiff=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparities = []
    valid_fractions = []
    medians = []
    frame_indices = []
    preview_frames = set([0, len(frames) // 2, len(frames) - 1])
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        for row in frames:
            frame_idx = int(row.get("frame_idx", row.get("index", len(frame_indices))))
            ok_right, right = cap_right.read()
            if not ok_right:
                break
            left = cv2.imread(str(row["rgb"]), cv2.IMREAD_COLOR)
            if left is None:
                raise ContractError(f"could_not_read_left_manifest_frame: {row['rgb']}")
            if right.shape[:2] != left.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            left_small, right_small = resize_pair(left, right, int(args.max_width))
            left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
            disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
            valid = np.isfinite(disp) & (disp > float(args.min_disparity))
            rel = np.zeros(disp.shape, dtype=np.float32)
            if np.any(valid):
                rel[valid] = disp[valid] / max(1.0, float(num_disparities))
                valid_fractions.append(float(np.mean(valid)))
                medians.append(float(np.median(disp[valid])))
            else:
                valid_fractions.append(0.0)
            disparities.append(rel.astype(np.float16))
            frame_indices.append(frame_idx)
            if frame_idx in preview_frames:
                preview = normalize_preview(rel)
                joined = np.hstack([left_small, right_small, preview])
                cv2.imwrite(str(args.preview_dir / f"stereo_disparity_{frame_idx:06d}.jpg"), joined)
    finally:
        cap_right.release()
    if not disparities:
        raise ContractError("uncalibrated_stereo_disparity_produced_no_frames")
    rel_stack = np.stack(disparities, axis=0)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        relative_inverse_depth=rel_stack,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        value_semantics=np.asarray(["relative_inverse_depth_from_uncalibrated_stereo_not_metric"], dtype=object),
        output_size_hw=np.asarray(rel_stack.shape[1:], dtype=np.int32),
    )
    report = {
        "schema": "v20_uncalibrated_stereo_disparity_report.v0",
        "status": "ok",
        "method": "opencv_sgbm_uncalibrated_stereo_disparity",
        "left_video": left_meta,
        "right_video": right_meta,
        "output_npz": str(args.output_npz),
        "preview_dir": str(args.preview_dir),
        "frame_count": int(rel_stack.shape[0]),
        "output_size_hw": [int(rel_stack.shape[1]), int(rel_stack.shape[2])],
        "valid_fraction": numeric_summary(valid_fractions),
        "median_disparity_px": numeric_summary(medians),
        "metric_depth_available": False,
        "calibration_available": False,
        "selection_status": "retained_weak_nonmetric_not_primary_depth",
        "claim_scope": "OpenCV SGBM disparity from synchronized stereo frames without calibration or rectification evidence. Values are relative inverse-depth cues only and must not support metric geometry/contact claims.",
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak non-metric stereo disparity evidence for V20 infer when calibration is unavailable.")
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--num-disparities", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=8)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
