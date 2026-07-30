#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise RuntimeError(f"{path} must contain a JSON object with a frames list")
    return payload


def frame_entry(manifest: dict, frame_idx: int) -> dict:
    entries = [entry for entry in manifest["frames"] if int(entry["frame_idx"]) == int(frame_idx)]
    if len(entries) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(entries)} times in manifest")
    return entries[0]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def square_bounds(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin: float,
) -> tuple[int, int, int, int, int, int, int, int]:
    x0, y0, x1, y1 = box
    side = int(round(max(x1 - x0, y1 - y0) * (1.0 + margin)))
    if side <= 0:
        raise RuntimeError("crop side is non-positive")
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    raw_x0 = int(round(cx - 0.5 * side))
    raw_y0 = int(round(cy - 0.5 * side))
    raw_x1 = raw_x0 + side
    raw_y1 = raw_y0 + side
    h, w = image_shape
    pad_left = max(0, -raw_x0)
    pad_top = max(0, -raw_y0)
    pad_right = max(0, raw_x1 - w)
    pad_bottom = max(0, raw_y1 - h)
    return (
        max(0, raw_x0),
        max(0, raw_y0),
        min(w, raw_x1),
        min(h, raw_y1),
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    )


def crop_pad(image: np.ndarray, bounds: tuple[int, int, int, int, int, int, int, int], value: tuple[int, ...]) -> np.ndarray:
    x0, y0, x1, y1, pad_left, pad_top, pad_right, pad_bottom = bounds
    crop = image[y0:y1, x0:x1]
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=value)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def run(args: argparse.Namespace) -> dict:
    manifest = load_manifest(args.manifest)
    entry = frame_entry(manifest, args.frame_idx)
    rgb_bgr = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
    if rgb_bgr is None:
        raise RuntimeError(f"failed to read RGB image {entry['rgb']}")
    if mask_gray is None:
        raise RuntimeError(f"failed to read mask image {entry['mask']}")
    mask = mask_gray > 0
    if rgb_bgr.shape[:2] != mask.shape:
        raise RuntimeError(f"RGB shape {rgb_bgr.shape[:2]} differs from mask shape {mask.shape}")

    bounds = square_bounds(mask_bbox(mask), rgb_bgr.shape[:2], float(args.margin))
    crop_bgr = crop_pad(rgb_bgr, bounds, (int(args.background), int(args.background), int(args.background)))
    crop_mask = crop_pad((mask.astype(np.uint8) * 255), bounds, (0,))
    crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_rgba[:, :, 3] = crop_mask
    if args.crop_size:
        size = (int(args.crop_size), int(args.crop_size))
        crop_bgr = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST)
        crop_rgba = cv2.resize(crop_rgba, size, interpolation=cv2.INTER_AREA)
        crop_rgba[:, :, 3] = crop_mask

    full_image = args.output_dir / f"frame_{args.frame_idx:06d}_image.png"
    full_mask = args.output_dir / f"frame_{args.frame_idx:06d}_mask.png"
    crop_rgb = args.output_dir / f"frame_{args.frame_idx:06d}_crop_rgb.png"
    crop_mask_path = args.output_dir / f"frame_{args.frame_idx:06d}_crop_mask.png"
    crop_rgba_path = args.output_dir / f"frame_{args.frame_idx:06d}_crop_rgba.png"
    review = rgb_bgr.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(review, contours, -1, (0, 255, 255), 3, cv2.LINE_AA)
    review_path = args.output_dir / f"frame_{args.frame_idx:06d}_mask_review.png"

    write_image(full_image, rgb_bgr)
    write_image(full_mask, mask.astype(np.uint8) * 255)
    write_image(crop_rgb, crop_bgr)
    write_image(crop_mask_path, crop_mask)
    write_image(crop_rgba_path, crop_rgba)
    write_image(review_path, review)

    report = {
        "status": "ok",
        "method": "prepare_sam3d_object_input_v3",
        "manifest": str(args.manifest),
        "frame_idx": int(args.frame_idx),
        "source_index": int(entry.get("source_index", entry["index"])),
        "source_rgb": str(entry["rgb"]),
        "source_mask": str(entry["mask"]),
        "image_shape_h_w": [int(rgb_bgr.shape[0]), int(rgb_bgr.shape[1])],
        "mask_area_px": int(np.count_nonzero(mask)),
        "mask_bbox_xyxy": list(mask_bbox(mask)),
        "crop_bounds_with_padding": [int(v) for v in bounds],
        "full_image": str(full_image),
        "full_mask": str(full_mask),
        "crop_rgb": str(crop_rgb),
        "crop_mask": str(crop_mask_path),
        "crop_rgba": str(crop_rgba_path),
        "review": str(review_path),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = args.output_dir / f"qc_sam3d_input_frame_{args.frame_idx:06d}.json"
    qc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--background", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
