#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from run_unidepth_metric_source_v3 import infer_unidepth, load_model, localize_path, read_manifest, resize_depth, summarize


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.unidepth_repo is not None:
        sys.path.insert(0, str(args.unidepth_repo))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("UniDepth full-frame export requires CUDA unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = load_model(args.model_id, device)
    rows_in = read_manifest(args.manifest, int(args.frame_start), int(args.frame_end))
    if not rows_in:
        raise RuntimeError("manifest contains no selected frames")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    depth_png_dir = args.output_dir / "depth"
    depth_png_dir.mkdir(exist_ok=True)

    depth_stack = []
    frame_indices = []
    focal_px = []
    intrinsics_stack = []
    rows = []
    for out_i, entry in enumerate(rows_in):
        frame_idx = int(entry["frame_idx"])
        rgb_path = localize_path(str(entry["rgb"]), args.remote_root, args.local_root)
        image = Image.open(rgb_path).convert("RGB")
        depth_raw, intrinsics = infer_unidepth(model, image, device)
        depth = resize_depth(depth_raw, (int(args.source_height), int(args.source_width)))
        if intrinsics is None:
            raise RuntimeError(f"UniDepth returned no intrinsics for frame {frame_idx}")
        fx = float(intrinsics[0, 0]) * (int(args.source_width) / float(depth_raw.shape[1]))
        fy = float(intrinsics[1, 1]) * (int(args.source_height) / float(depth_raw.shape[0]))
        cx = float(intrinsics[0, 2]) * (int(args.source_width) / float(depth_raw.shape[1]))
        cy = float(intrinsics[1, 2]) * (int(args.source_height) / float(depth_raw.shape[0]))
        focal = float(np.sqrt(max(1e-9, fx * fy)))
        valid = np.isfinite(depth) & (depth > 0.0)
        if int(np.count_nonzero(valid)) < int(args.min_valid_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few valid UniDepth pixels")
        depth_png_path = depth_png_dir / f"{out_i:06d}.png"
        depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if not cv2.imwrite(str(depth_png_path), depth_mm):
            raise RuntimeError(f"failed to write {depth_png_path}")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read RGB {rgb_path}")
        norm = depth.copy()
        lo, hi = np.percentile(depth[valid], [5.0, 95.0])
        norm = np.clip((norm - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        review = cv2.addWeighted(rgb, 0.55, color, 0.45, 0.0)
        cv2.imwrite(str(still_dir / f"frame_{frame_idx:06d}.png"), review)

        frame_indices.append(frame_idx)
        focal_px.append(focal)
        intrinsics_stack.append([fx, fy, cx, cy])
        depth_stack.append(depth.astype(np.float16))
        values = depth[valid].astype(np.float64)
        rows.append(
            {
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "depth_png": str(depth_png_path),
                "unidepth_focal_px": focal,
                "unidepth_fx_px": fx,
                "unidepth_fy_px": fy,
                "unidepth_cx_px": cx,
                "unidepth_cy_px": cy,
                "depth_median_m": float(np.median(values)),
                "depth_p05_m": float(np.percentile(values, 5.0)),
                "depth_p95_m": float(np.percentile(values, 95.0)),
            }
        )

    depth_archive = args.output_dir / "unidepth_full_frame_depth_v3.npz"
    np.savez_compressed(
        depth_archive,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        depth=np.stack(depth_stack, axis=0),
        source_size=np.asarray([int(args.source_width), int(args.source_height)], dtype=np.int32),
        focal_px=np.asarray(focal_px, dtype=np.float32),
        intrinsics_fx_fy_cx_cy=np.asarray(intrinsics_stack, dtype=np.float32),
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "run_unidepth_full_frame_v3",
        "model_id": str(args.model_id),
        "manifest": str(args.manifest),
        "frames": int(len(rows)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "depth_archive": str(depth_archive),
        "stills_dir": str(still_dir),
        "unidepth_focal_px": summarize([row["unidepth_focal_px"] for row in rows]),
        "depth_median_m": summarize([row["depth_median_m"] for row in rows]),
        "rows": rows,
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_unidepth_full_frame_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--unidepth-repo", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--min-valid-pixels", type=int, default=100000)
    parser.add_argument("--model-id", default="lpiccinelli/unidepth-v2-vitl14")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
