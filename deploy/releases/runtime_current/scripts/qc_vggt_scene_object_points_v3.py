#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    rows = {}
    for row in payload["frames"]:
        frame_idx = int(row["frame_idx"])
        if frame_idx in rows:
            raise RuntimeError(f"duplicate manifest frame {frame_idx}")
        rows[frame_idx] = row
    return rows


def vggt_resize_pad(
    image: np.ndarray,
    target_size: int,
    interpolation: int,
    fill_value: int | float,
) -> np.ndarray:
    height, width = image.shape[:2]
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    if new_width <= 0 or new_height <= 0:
        raise RuntimeError("invalid VGGT resize dimensions")
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2
    if image.ndim == 2:
        out = np.full((target_size, target_size), fill_value, dtype=image.dtype)
        out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    else:
        out = np.full((target_size, target_size, image.shape[2]), fill_value, dtype=image.dtype)
        out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    return out


def draw_points(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    for x, y in zip(xs.astype(int), ys.astype(int)):
        cv2.circle(out, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)
    return out


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "stills"
    review_dir.mkdir(exist_ok=True)
    archive = np.load(args.vggt_archive)
    required = {"frame_idx", "vertex_offsets", "object_points_vggt", "object_colors", "extrinsic", "intrinsic"}
    missing = required.difference(archive.files)
    if missing:
        raise RuntimeError(f"VGGT archive missing keys: {sorted(missing)}")
    frame_idx = archive["frame_idx"].astype(int)
    offsets = archive["vertex_offsets"].astype(np.int64)
    points = archive["object_points_vggt"].astype(float)
    colors = archive["object_colors"].astype(np.uint8)
    extrinsic = archive["extrinsic"].astype(float)
    intrinsic = archive["intrinsic"].astype(float)
    manifest = read_manifest(args.dataset_manifest)
    rows = []
    for i, idx in enumerate(frame_idx.tolist()):
        row = manifest.get(int(idx))
        if row is None:
            raise RuntimeError(f"manifest lacks frame {idx}")
        mask = cv2.imread(str(Path(row["mask"])), cv2.IMREAD_GRAYSCALE)
        depth_raw = cv2.imread(str(Path(row["depth"])), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(Path(row["rgb"])), cv2.IMREAD_COLOR)
        if mask is None or depth_raw is None or rgb is None:
            raise RuntimeError(f"could not read manifest assets for frame {idx}")
        mask_small = vggt_resize_pad(mask, int(args.target_size), cv2.INTER_NEAREST, 0) > 0
        depth_small = vggt_resize_pad(depth_raw.astype(np.float32) / 1000.0, int(args.target_size), cv2.INTER_NEAREST, 0.0)
        rgb_small = vggt_resize_pad(rgb, int(args.target_size), cv2.INTER_AREA, 0)
        start = int(offsets[i])
        end = int(offsets[i + 1])
        pts = points[start:end]
        cols = colors[start:end]
        pts_cam = (pts @ extrinsic[i, :3, :3].T) + extrinsic[i, :3, 3][None, :]
        z = pts_cam[:, 2]
        valid_z = np.isfinite(z) & (z > 1e-6)
        uv_h = (pts_cam @ intrinsic[i].T)
        xs = uv_h[:, 0] / z
        ys = uv_h[:, 1] / z
        in_image = (
            valid_z
            & np.isfinite(xs)
            & np.isfinite(ys)
            & (xs >= 0)
            & (ys >= 0)
            & (xs < int(args.target_size))
            & (ys < int(args.target_size))
        )
        xi = np.clip(np.rint(xs[in_image]).astype(int), 0, int(args.target_size) - 1)
        yi = np.clip(np.rint(ys[in_image]).astype(int), 0, int(args.target_size) - 1)
        projected = np.zeros((int(args.target_size), int(args.target_size)), dtype=bool)
        projected[yi, xi] = True
        union = int(np.logical_or(projected, mask_small).sum())
        intersection = int(np.logical_and(projected, mask_small).sum())
        in_mask_fraction = float(mask_small[yi, xi].mean()) if len(xi) else 0.0
        depth_values = depth_small[yi, xi] if len(xi) else np.zeros(0, dtype=np.float32)
        depth_delta = z[in_image] - depth_values
        valid_depth = np.isfinite(depth_delta) & (depth_values > 0)
        if i % int(args.review_stride) == 0:
            overlay = np.zeros_like(rgb_small)
            overlay[:, :, 1] = np.where(mask_small, 255, 0).astype(np.uint8)
            base = cv2.addWeighted(rgb_small, 0.75, overlay, 0.25, 0.0)
            if len(xi):
                if len(xi) > int(args.max_review_points):
                    keep = np.linspace(0, len(xi) - 1, int(args.max_review_points), dtype=int)
                else:
                    keep = np.arange(len(xi), dtype=int)
                base = draw_points(base, xi[keep], yi[keep], (0, 0, 255))
            cv2.imwrite(str(review_dir / f"frame_{idx:06d}.jpg"), base)
        rows.append(
            {
                "frame_idx": int(idx),
                "points": int(len(pts)),
                "projected_points": int(in_image.sum()),
                "point_in_mask_fraction": in_mask_fraction,
                "projected_mask_iou": float(intersection / union) if union else 0.0,
                "median_vggt_minus_depth_anything_m": float(np.median(depth_delta[valid_depth]))
                if valid_depth.any()
                else None,
                "p95_abs_vggt_minus_depth_anything_m": float(np.percentile(np.abs(depth_delta[valid_depth]), 95))
                if valid_depth.any()
                else None,
                "median_vggt_camera_z": float(np.median(z[valid_z])) if valid_z.any() else None,
                "median_depth_anything_z": float(np.median(depth_values[valid_depth])) if valid_depth.any() else None,
            }
        )
    report = {
        "status": "ok",
        "method": "qc_vggt_scene_object_points_v3",
        "vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "frames": int(len(rows)),
        "point_in_mask_fraction": summarize([row["point_in_mask_fraction"] for row in rows]),
        "projected_mask_iou": summarize([row["projected_mask_iou"] for row in rows]),
        "median_vggt_minus_depth_anything_m": summarize(
            [row["median_vggt_minus_depth_anything_m"] for row in rows if row["median_vggt_minus_depth_anything_m"] is not None]
        ),
        "p95_abs_vggt_minus_depth_anything_m": summarize(
            [row["p95_abs_vggt_minus_depth_anything_m"] for row in rows if row["p95_abs_vggt_minus_depth_anything_m"] is not None]
        ),
        "rows": rows,
        "outputs": {"review_stills": str(review_dir)},
    }
    (args.output_dir / "qc_vggt_scene_object_points_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--review-stride", type=int, default=4)
    parser.add_argument("--max-review-points", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
