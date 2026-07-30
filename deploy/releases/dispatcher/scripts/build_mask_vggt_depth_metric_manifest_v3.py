#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_vggt_native_camera_v3 import source_intrinsics


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def resize_vggt_depth_to_source(depth: np.ndarray, source_width: int, source_height: int, target_size: int) -> np.ndarray:
    if depth.shape != (target_size, target_size):
        raise RuntimeError(f"expected VGGT depth {target_size}x{target_size}, got {depth.shape}")
    if source_width >= source_height:
        new_width = int(target_size)
        new_height = round(source_height * (new_width / source_width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(source_width * (new_height / source_height) / 14) * 14
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2
    crop = depth[pad_top : pad_top + new_height, pad_left : pad_left + new_width]
    return cv2.resize(crop.astype(np.float32), (source_width, source_height), interpolation=cv2.INTER_LINEAR)


def write_depth_mm(path: Path, depth_m: np.ndarray) -> None:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        raise RuntimeError("depth map has no finite positive values")
    depth_mm = np.clip(depth_m.astype(np.float64) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"failed to write depth map {path}")


def run(args: argparse.Namespace) -> dict:
    mask_payload = load_json(args.mask_manifest)
    mask_frames = mask_payload.get("frames")
    if not isinstance(mask_frames, list) or not mask_frames:
        raise RuntimeError(f"{args.mask_manifest} must contain nonempty frames")
    blob = np.load(args.vggt_archive)
    required = {"frame_idx", "depth", "intrinsic", "vggt_to_meters"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{args.vggt_archive} missing keys: {sorted(missing)}")
    frame_to_i = {int(idx): i for i, idx in enumerate(blob["frame_idx"].astype(int).tolist())}
    scale = float(blob["vggt_to_meters"][0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid VGGT metric scale: {scale}")
    depth = blob["depth"].astype(np.float32)
    intrinsic = blob["intrinsic"].astype(np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    rows = []
    for row in mask_frames:
        frame_idx = int(row["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        i = frame_to_i.get(frame_idx)
        if i is None:
            raise RuntimeError(f"mask frame {frame_idx} has no VGGT depth row")
        mask_path = Path(str(row["mask"]))
        rgb_path = Path(str(row["rgb"]))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")
        source_depth = resize_vggt_depth_to_source(
            depth[int(i)] * scale,
            int(args.source_width),
            int(args.source_height),
            int(args.target_size),
        )
        if mask.shape != source_depth.shape:
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}: {mask.shape} vs {source_depth.shape}")
        object_pixels = (mask > 0) & np.isfinite(source_depth) & (source_depth > 0.0)
        if int(np.count_nonzero(object_pixels)) < int(args.min_mask_depth_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few masked VGGT depth pixels")
        out_i = len(rows)
        depth_path = depth_dir / f"{out_i:06d}.png"
        write_depth_mm(depth_path, source_depth)
        values = source_depth[object_pixels].astype(np.float64)
        rows.append(
            {
                "index": int(out_i),
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "depth": str(depth_path),
                "source_mask_manifest": str(args.mask_manifest),
                "source_vggt_archive": str(args.vggt_archive),
                "depth_source": "vggt_depth_scaled_to_metric",
                "vggt_to_meters": float(scale),
                "intrinsics_fx_fy_cx_cy": source_intrinsics(
                    intrinsic[int(i)],
                    int(args.source_width),
                    int(args.source_height),
                    int(args.target_size),
                ),
                "mask_pixels": int(np.count_nonzero(mask > 0)),
                "valid_mask_depth_pixels": int(np.count_nonzero(object_pixels)),
                "mask_depth_median_m": float(np.median(values)),
                "mask_depth_p05_m": float(np.percentile(values, 5.0)),
                "mask_depth_p95_m": float(np.percentile(values, 95.0)),
            }
        )
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} VGGT metric rows joined, min_frames={args.min_frames}")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "build_mask_vggt_depth_metric_manifest_v3",
        "mask_manifest": str(args.mask_manifest),
        "vggt_archive": str(args.vggt_archive),
        "manifest": str(manifest_path),
        "frames": int(len(rows)),
        "frame_start": int(rows[0]["frame_idx"]),
        "frame_end": int(rows[-1]["frame_idx"]),
        "vggt_to_meters": float(scale),
        "mask_depth_median_m": {
            "median": float(np.median([row["mask_depth_median_m"] for row in rows])),
            "p05": float(np.percentile([row["mask_depth_median_m"] for row in rows], 5.0)),
            "p95": float(np.percentile([row["mask_depth_median_m"] for row in rows], 95.0)),
        },
    }
    (args.output_dir / "qc_mask_vggt_depth_metric_manifest_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-mask-depth-pixels", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
