#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def square_crop_bounds(box: tuple[int, int, int, int], image_shape: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    side = int(round(max(width, height) * (1.0 + margin)))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    xx0 = int(round(cx - 0.5 * side))
    yy0 = int(round(cy - 0.5 * side))
    xx1 = xx0 + side
    yy1 = yy0 + side
    pad_left = max(0, -xx0)
    pad_top = max(0, -yy0)
    pad_right = max(0, xx1 - image_shape[1])
    pad_bottom = max(0, yy1 - image_shape[0])
    xx0 = max(0, xx0)
    yy0 = max(0, yy0)
    xx1 = min(image_shape[1], xx1)
    yy1 = min(image_shape[0], yy1)
    return xx0, yy0, xx1, yy1, pad_left, pad_top, pad_right, pad_bottom


def crop_with_padding(image: np.ndarray, bounds: tuple[int, int, int, int, int, int, int, int], value: tuple[int, ...]) -> np.ndarray:
    x0, y0, x1, y1, pad_left, pad_top, pad_right, pad_bottom = bounds
    crop = image[y0:y1, x0:x1]
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=value)


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("manifest must contain nonempty frames list")
    matches = [entry for entry in entries if int(entry["frame_idx"]) == int(args.frame_idx)]
    if len(matches) != 1:
        raise RuntimeError(f"frame {args.frame_idx} appears {len(matches)} times in {args.manifest}")
    entry = matches[0]
    rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        raise RuntimeError(f"failed to read RGB/mask for frame {args.frame_idx}")
    mask_bool = mask > 0
    bounds = square_crop_bounds(bbox(mask_bool), rgb.shape[:2], float(args.margin))
    rgb_crop = crop_with_padding(rgb, bounds, (int(args.background), int(args.background), int(args.background)))
    mask_crop = crop_with_padding((mask_bool.astype(np.uint8) * 255), bounds, (0,))
    isolated = np.full_like(rgb_crop, int(args.background))
    isolated[mask_crop > 0] = rgb_crop[mask_crop > 0]
    rgba = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_crop
    if args.output_size:
        size = (int(args.output_size), int(args.output_size))
        isolated = cv2.resize(isolated, size, interpolation=cv2.INTER_AREA)
        rgba = cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(mask_crop, size, interpolation=cv2.INTER_NEAREST)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = args.output_dir / f"{args.name}_rgba.png"
    rgb_path = args.output_dir / f"{args.name}_gray.png"
    mask_path = args.output_dir / f"{args.name}_mask.png"
    review_path = args.output_dir / f"{args.name}_review.png"
    if not cv2.imwrite(str(rgba_path), rgba):
        raise RuntimeError(f"failed to write {rgba_path}")
    if not cv2.imwrite(str(rgb_path), isolated):
        raise RuntimeError(f"failed to write {rgb_path}")
    if not cv2.imwrite(str(mask_path), mask_crop):
        raise RuntimeError(f"failed to write {mask_path}")
    review = isolated.copy()
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(review, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(review_path), review):
        raise RuntimeError(f"failed to write {review_path}")
    report = {
        "status": "ok",
        "method": "object_crop_from_mask_v3",
        "manifest": str(args.manifest),
        "frame_idx": int(args.frame_idx),
        "source_index": int(entry.get("source_index", entry["index"])),
        "mask_area_px": int(np.count_nonzero(mask_bool)),
        "output_dir": str(args.output_dir),
        "rgba": str(rgba_path),
        "rgb_gray_background": str(rgb_path),
        "mask": str(mask_path),
        "review": str(review_path),
        "crop_bounds_with_padding": [int(v) for v in bounds],
    }
    (args.output_dir / f"qc_{args.name}_crop_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="object_crop")
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--background", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
