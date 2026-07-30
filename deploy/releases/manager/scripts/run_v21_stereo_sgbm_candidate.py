#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def numeric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


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
    meta["duration_s"] = float(meta["frame_count"] / meta["fps"]) if meta["fps"] > 0 else None
    return meta


def resize_pair(left: np.ndarray, right: np.ndarray, max_width: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = left.shape[:2]
    scale = min(1.0, float(max_width) / float(width))
    out_w = max(16, int(round(width * scale / 2.0)) * 2)
    out_h = max(16, int(round(height * scale / 2.0)) * 2)
    return cv2.resize(left, (out_w, out_h), interpolation=cv2.INTER_AREA), cv2.resize(right, (out_w, out_h), interpolation=cv2.INTER_AREA)


def colorize(relative_inverse_depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(relative_inverse_depth) & (relative_inverse_depth > 0)
    preview = np.zeros(relative_inverse_depth.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(relative_inverse_depth[valid], [2.0, 98.0])
        if hi > lo:
            preview[valid] = np.clip((relative_inverse_depth[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.raw_frame_manifest)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"raw_manifest_has_no_frames: {args.raw_frame_manifest}")
    left_meta = video_meta(args.left_video)
    right_meta = video_meta(args.right_video)
    cap_right = cv2.VideoCapture(str(args.right_video))
    if not cap_right.isOpened():
        raise ContractError(f"could_not_open_right_video: {args.right_video}")
    num_disparities = max(16, (int(args.num_disparities) // 16) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=int(args.block_size),
        P1=8 * 3 * int(args.block_size) ** 2,
        P2=32 * 3 * int(args.block_size) ** 2,
        uniquenessRatio=int(args.uniqueness_ratio),
        speckleWindowSize=int(args.speckle_window_size),
        speckleRange=int(args.speckle_range),
        disp12MaxDiff=int(args.disp12_max_diff),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparities = []
    frame_indices = []
    valid_fractions = []
    medians = []
    preview_sources = {0, max(0, len(frames) // 2), max(0, len(frames) - 1)}
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        for local_i, row in enumerate(frames):
            ok_right, right = cap_right.read()
            if not ok_right:
                raise ContractError(f"right_video_ended_early: {local_i}")
            left = cv2.imread(str(row["rgb"]), cv2.IMREAD_COLOR)
            if left is None:
                raise ContractError(f"could_not_read_left_frame: {row['rgb']}")
            if right.shape[:2] != left.shape[:2]:
                right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            left_small, right_small = resize_pair(left, right, int(args.max_width))
            disp = matcher.compute(cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
            valid = np.isfinite(disp) & (disp > float(args.min_disparity))
            rel = np.zeros_like(disp, dtype=np.float32)
            if np.any(valid):
                rel[valid] = disp[valid] / float(num_disparities)
                valid_fractions.append(float(np.mean(valid)))
                medians.append(float(np.median(disp[valid])))
            else:
                valid_fractions.append(0.0)
            disparities.append(rel.astype(np.float16))
            frame_idx = int(row.get("frame_idx", local_i))
            frame_indices.append(frame_idx)
            if local_i in preview_sources or frame_idx in preview_sources:
                preview = colorize(rel)
                joined = np.hstack([left_small, right_small, preview])
                cv2.putText(joined, f"V21 SGBM relative inverse depth frame {frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(joined, f"V21 SGBM relative inverse depth frame {frame_idx}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(args.preview_dir / f"stereo_sgbm_{frame_idx:06d}.jpg"), joined)
    finally:
        cap_right.release()
    if not disparities:
        raise ContractError("no_disparity_frames")
    stack = np.stack(disparities, axis=0)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        relative_inverse_depth=stack,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        output_size_hw=np.asarray(stack.shape[1:], dtype=np.int32),
        value_semantics=np.asarray(["relative_inverse_depth_from_uncalibrated_stereo_not_metric"], dtype=object),
    )
    report = {
        "schema": "v21_stereo_sgbm_candidate_report.v0",
        "status": "ok",
        "method": "run_v21_stereo_sgbm_candidate",
        "candidate_id": "stereo_sgbm_relative_inverse_depth",
        "left_video": left_meta,
        "right_video": right_meta,
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "output_npz": str(args.output_npz),
        "preview_dir": str(args.preview_dir),
        "frame_count": int(stack.shape[0]),
        "output_size_hw": [int(stack.shape[1]), int(stack.shape[2])],
        "parameters": {
            "max_width": int(args.max_width),
            "num_disparities": int(num_disparities),
            "block_size": int(args.block_size),
            "uniqueness_ratio": int(args.uniqueness_ratio),
            "speckle_window_size": int(args.speckle_window_size),
            "speckle_range": int(args.speckle_range),
            "disp12_max_diff": int(args.disp12_max_diff),
            "min_disparity": float(args.min_disparity),
        },
        "valid_fraction": numeric_summary(valid_fractions),
        "median_disparity_px": numeric_summary(medians),
        "metric_depth_available": False,
        "calibration_available": False,
        "selection_status": "retained_as_assisted_relative_depth_not_primary_metric_depth",
        "claim_scope": "OpenCV SGBM on synchronized views without calibration/rectification. This can diagnose stereo signal and support qualitative assisted segmentation cues, but cannot support metric object/hand/contact claims.",
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V21 uncalibrated stereo SGBM as nonmetric relative inverse-depth evidence.")
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=480)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=8)
    parser.add_argument("--speckle-window-size", type=int, default=80)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--disp12-max-diff", type=int, default=2)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
