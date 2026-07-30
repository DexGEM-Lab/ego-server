#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.source_manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"{args.source_manifest} must contain nonempty frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "depth", "masks"):
        (args.output_dir / sub).mkdir(exist_ok=True)
    cam_k = args.source_dataset / "cam_K.txt"
    if not cam_k.exists():
        raise RuntimeError(f"missing {cam_k}")
    shutil.copy2(cam_k, args.output_dir / "cam_K.txt")
    rows = []
    out_entries = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        rgb_path = Path(entry["rgb"])
        depth_path = Path(entry["depth"])
        mask_path = Path(entry["mask"])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise RuntimeError(f"failed to read frame {frame_idx} rgb/depth/mask")
        mask_bool = mask > 0
        depth_m = depth.astype(np.float64) / 1000.0
        values = depth_m[mask_bool & np.isfinite(depth_m) & (depth_m > 0.0)]
        if values.size < int(args.min_source_pixels):
            rows.append({"frame_idx": frame_idx, "status": "skipped_too_few_source_pixels", "source_pixels": int(values.size)})
            continue
        center = float(np.quantile(values, float(args.center_quantile)))
        band = float(args.depth_band_m)
        filtered = mask_bool & np.isfinite(depth_m) & (np.abs(depth_m - center) <= band)
        if args.keep_largest_component:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered.astype(np.uint8), 8)
            if count > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest = int(np.argmax(areas) + 1)
                filtered = labels == largest
        area = int(np.count_nonzero(filtered))
        if area < int(args.min_mask_pixels):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "status": "skipped_too_few_filtered_pixels",
                    "source_pixels": int(values.size),
                    "filtered_pixels": area,
                    "center_depth_m": center,
                }
            )
            continue
        out_index = len(out_entries)
        stem = f"{out_index:06d}"
        rgb_out = args.output_dir / "rgb" / f"{stem}.png"
        depth_out = args.output_dir / "depth" / f"{stem}.png"
        mask_out = args.output_dir / "masks" / f"{stem}.png"
        if not cv2.imwrite(str(rgb_out), rgb):
            raise RuntimeError(f"failed to write {rgb_out}")
        if not cv2.imwrite(str(depth_out), depth):
            raise RuntimeError(f"failed to write {depth_out}")
        if not cv2.imwrite(str(mask_out), filtered.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_out}")
        kept = depth_m[filtered]
        out_entry = dict(entry)
        out_entry.update(
            {
                "index": out_index,
                "source_index": int(entry.get("index", out_index)),
                "rgb": str(rgb_out),
                "depth": str(depth_out),
                "mask": str(mask_out),
                "mask_area_px": area,
                "mask_depth_median_m": float(np.median(kept)),
                "mask_depth_p05_m": float(np.percentile(kept, 5.0)),
                "mask_depth_p95_m": float(np.percentile(kept, 95.0)),
            }
        )
        out_entries.append(out_entry)
        rows.append(
            {
                "frame_idx": frame_idx,
                "status": "ok",
                "source_pixels": int(values.size),
                "filtered_pixels": area,
                "center_depth_m": center,
                "depth_p05_m": float(np.percentile(kept, 5.0)),
                "depth_p95_m": float(np.percentile(kept, 95.0)),
            }
        )
    if len(out_entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(out_entries)} quantile-depth frames survived")
    out_manifest = {"frames": out_entries}
    (args.output_dir / "manifest.json").write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "method": "filter_mask_depth_quantile_dataset_v3",
        "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest),
        "output_dir": str(args.output_dir),
        "manifest": str(args.output_dir / "manifest.json"),
        "center_quantile": float(args.center_quantile),
        "depth_band_m": float(args.depth_band_m),
        "frames": int(len(out_entries)),
        "rows": rows,
    }
    (args.output_dir / "qc_depth_quantile_filter_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "rows"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-quantile", type=float, default=0.50)
    parser.add_argument("--depth-band-m", type=float, default=0.05)
    parser.add_argument("--min-source-pixels", type=int, default=500)
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
