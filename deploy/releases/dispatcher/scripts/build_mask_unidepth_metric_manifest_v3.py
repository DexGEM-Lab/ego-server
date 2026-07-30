#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_depth(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy", "source_size"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    if len(set(frame_idx.tolist())) != len(frame_idx):
        raise RuntimeError(f"{path} contains duplicate frame_idx values")
    return {
        "frame_to_i": {int(idx): i for i, idx in enumerate(frame_idx.tolist())},
        "depth": blob["depth"].astype(np.float32),
        "intrinsics": blob["intrinsics_fx_fy_cx_cy"].astype(np.float64),
        "source_size": blob["source_size"].astype(int).tolist(),
    }


def write_depth_mm(path: Path, depth_m: np.ndarray) -> None:
    if depth_m.ndim != 2:
        raise RuntimeError(f"depth map must be HxW, got {depth_m.shape}")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        raise RuntimeError("depth map has no positive finite values")
    depth_mm = np.clip(depth_m.astype(np.float64) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"failed to write depth map {path}")


def run(args: argparse.Namespace) -> dict:
    mask_manifest = load_json(args.mask_manifest)
    frames = mask_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{args.mask_manifest} must contain nonempty frames")
    depth = load_depth(args.unidepth_npz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    rows = []
    for row in frames:
        frame_idx = int(row["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        depth_i = depth["frame_to_i"].get(frame_idx)
        if depth_i is None:
            raise RuntimeError(f"measured mask frame {frame_idx} has no UniDepth row")
        mask_path = Path(str(row["mask"]))
        rgb_path = Path(str(row["rgb"]))
        if not mask_path.exists():
            raise RuntimeError(f"mask path does not exist for frame {frame_idx}: {mask_path}")
        if not rgb_path.exists():
            raise RuntimeError(f"rgb path does not exist for frame {frame_idx}: {rgb_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {mask_path}")
        depth_m = depth["depth"][int(depth_i)]
        if tuple(mask.shape) != tuple(depth_m.shape):
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}: {mask.shape} vs {depth_m.shape}")
        object_pixels = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0)
        if int(np.count_nonzero(object_pixels)) < int(args.min_mask_depth_pixels):
            raise RuntimeError(f"frame {frame_idx} has too few valid masked depth pixels")
        out_i = len(rows)
        depth_path = depth_dir / f"{out_i:06d}.png"
        write_depth_mm(depth_path, depth_m)
        fx, fy, cx, cy = depth["intrinsics"][int(depth_i)].astype(float).tolist()
        values = depth_m[object_pixels].astype(np.float64)
        rows.append(
            {
                "index": int(out_i),
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "depth": str(depth_path),
                "source_mask_manifest": str(args.mask_manifest),
                "source_unidepth_npz": str(args.unidepth_npz),
                "intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
                "mask_pixels": int(np.count_nonzero(mask > 0)),
                "valid_mask_depth_pixels": int(np.count_nonzero(object_pixels)),
                "mask_depth_median_m": float(np.median(values)),
                "mask_depth_p05_m": float(np.percentile(values, 5.0)),
                "mask_depth_p95_m": float(np.percentile(values, 95.0)),
            }
        )
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} metric rows joined, min_frames={args.min_frames}")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": rows}, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "build_mask_unidepth_metric_manifest_v3",
        "mask_manifest": str(args.mask_manifest),
        "unidepth_npz": str(args.unidepth_npz),
        "manifest": str(manifest_path),
        "frames": int(len(rows)),
        "frame_start": int(rows[0]["frame_idx"]),
        "frame_end": int(rows[-1]["frame_idx"]),
        "source_size": depth["source_size"],
        "mask_depth_median_m": {
            "median": float(np.median([row["mask_depth_median_m"] for row in rows])),
            "p05": float(np.percentile([row["mask_depth_median_m"] for row in rows], 5.0)),
            "p95": float(np.percentile([row["mask_depth_median_m"] for row in rows], 95.0)),
        },
    }
    (args.output_dir / "qc_mask_unidepth_metric_manifest_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--unidepth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--min-mask-depth-pixels", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
