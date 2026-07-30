#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from fuse_v1_full_fidelity import DEFAULT_CLIP, load_json, open_video, read_video_frame


MEASURED_STATUSES = {
    "measured_sam_kalman",
    "measured_plan_sam",
    "measured_plan_sam_vlm_verified",
    "measured_sam2_vlm_points",
}


def selected_frames(annotations: dict, frame_start: int | None, frame_end: int | None, stride: int) -> list[int]:
    frames = annotations["frames"]
    start = int(frames[0]["frame_idx"]) if frame_start is None else int(frame_start)
    end = int(frames[-1]["frame_idx"]) if frame_end is None else int(frame_end)
    out = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        if idx < start or idx > end or idx % stride:
            continue
        obj = frame.get("object") or {}
        if obj.get("status") in MEASURED_STATUSES and obj.get("mask_path"):
            out.append(idx)
    if not out:
        raise RuntimeError("no measured object-mask frames selected for metric depth")
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    indices = selected_frames(annotations, args.frame_start, args.frame_end, max(1, int(args.frame_stride)))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    local_only = args.local_files_only or os.environ.get("EGO_LOCAL_FILES_ONLY", "0") == "1"
    processor = AutoImageProcessor.from_pretrained(args.model_id, local_files_only=local_only)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id, local_files_only=local_only).to(device).eval()
    cap, info = open_video(args.clip)
    depth_maps = []
    try:
        for idx in tqdm(indices, desc="metric_depth_v2"):
            frame = read_video_frame(cap, idx)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred[:, None],
                size=(args.depth_height, args.depth_width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            depth = pred.detach().float().cpu().numpy()
            depth = np.maximum(depth, 0.0).astype(np.float16)
            depth_maps.append(depth)
    finally:
        cap.release()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "metric_depth_v2.npz"
    np.savez_compressed(
        archive,
        frame_idx=np.asarray(indices, dtype=np.int32),
        depth=np.stack(depth_maps, axis=0),
        depth_size=np.asarray([args.depth_width, args.depth_height], dtype=np.int32),
        source_size=np.asarray([info.width, info.height], dtype=np.int32),
        model_id=np.asarray([args.model_id]),
    )
    depths = np.asarray(depth_maps, dtype=np.float32)
    qc = {
        "status": "ok",
        "backend": "Depth Anything V2 metric indoor",
        "model_id": args.model_id,
        "clip": str(args.clip),
        "annotations": str(args.annotations),
        "selected_frames": len(indices),
        "first_frame": int(indices[0]),
        "last_frame": int(indices[-1]),
        "depth_width": int(args.depth_width),
        "depth_height": int(args.depth_height),
        "archive": str(archive),
        "depth_median_m": float(np.median(depths)),
        "depth_p05_m": float(np.percentile(depths, 5)),
        "depth_p95_m": float(np.percentile(depths, 95)),
        "elapsed_s": time.time() - started,
    }
    (args.output_dir / "qc_metric_depth_v2.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--depth-width", type=int, default=960)
    parser.add_argument("--depth-height", type=int, default=540)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
