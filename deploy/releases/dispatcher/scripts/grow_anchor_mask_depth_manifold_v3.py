#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_cam_k(path: Path, values: list[float]) -> None:
    fx, fy, cx, cy = values
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, K, fmt="%.10f")


def load_intrinsics(dataset: Path, manifest: dict) -> list[float]:
    qc_path = dataset / "qc_bundlesdf_dataset_v3.json"
    if qc_path.exists():
        values = load_json(qc_path).get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return [float(v) for v in values]
    values = manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return [float(v) for v in values]
    frames = manifest.get("frames")
    if isinstance(frames, list) and frames:
        rows = []
        for row in frames:
            raw = row.get("intrinsics_fx_fy_cx_cy")
            if raw is None:
                continue
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != (4,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid per-frame intrinsics for frame {row.get('frame_idx')}: {raw}")
            rows.append(values)
        if rows and len(rows) == len(frames):
            return np.median(np.stack(rows, axis=0), axis=0).astype(float).tolist()
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3):
        raise RuntimeError(f"{dataset / 'cam_K.txt'} must be 3x3")
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def frame_prompts(path: Path) -> tuple[dict[int, dict], tuple[int, int]]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks point_prompts")
    width = int(payload["prompt_image_width"])
    height = int(payload.get("prompt_image_height", round(width * 9 / 16)))
    return {int(row["frame_idx"]): row for row in rows}, (width, height)


def seed_pixels(prompt: dict, prompt_size: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    scale = np.asarray([shape[1] / prompt_size[0], shape[0] / prompt_size[1]], dtype=np.float64)
    seeds = []
    for point in prompt.get("positive_points", []):
        xy = np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64) * scale
        x = int(np.clip(round(float(xy[0])), 0, shape[1] - 1))
        y = int(np.clip(round(float(xy[1])), 0, shape[0] - 1))
        seeds.append((y, x))
    return seeds


def local_depth(depth_m: np.ndarray, y: int, x: int, radius: int) -> float | None:
    y0 = max(0, y - radius)
    y1 = min(depth_m.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(depth_m.shape[1], x + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0.05)]
    if values.size == 0:
        return None
    return float(np.median(values))


def grow(mask: np.ndarray, depth_m: np.ndarray, seeds: list[tuple[int, int]], args: argparse.Namespace) -> np.ndarray:
    allowed = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.05)
    out = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    seed_depths = []
    for y, x in seeds:
        if not allowed[y, x]:
            continue
        z = local_depth(depth_m, y, x, int(args.seed_depth_radius_px))
        if z is None:
            continue
        seed_depths.append(z)
        out[y, x] = True
        queue.append((y, x))
    if len(seed_depths) < int(args.min_seed_depths):
        return out
    depth_center = float(np.median(seed_depths))
    lower = depth_center - float(args.depth_band_m)
    upper = depth_center + float(args.depth_band_m)
    while queue:
        y, x = queue.popleft()
        current_z = float(depth_m[y, x])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= mask.shape[0] or xx < 0 or xx >= mask.shape[1]:
                continue
            if out[yy, xx] or not allowed[yy, xx]:
                continue
            z = float(depth_m[yy, xx])
            if z < lower or z > upper:
                continue
            if abs(z - current_z) > float(args.local_depth_step_m):
                continue
            out[yy, xx] = True
            queue.append((yy, xx))
    if args.close_px > 0:
        kernel = np.ones((2 * args.close_px + 1, 2 * args.close_px + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        out &= allowed & (depth_m >= lower) & (depth_m <= upper)
    return out


def run(args: argparse.Namespace) -> dict:
    source_manifest = load_json(args.source_manifest)
    entries = source_manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("source manifest must contain nonempty frames")
    prompts, prompt_size = frame_prompts(args.point_prompts)
    intrinsics = load_intrinsics(args.source_dataset, source_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "depth", "masks"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_cam_k(args.output_dir / "cam_K.txt", intrinsics)

    out_entries = []
    rows = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        prompt = prompts.get(frame_idx)
        if not prompt or not prompt.get("target_visible"):
            rows.append({"frame_idx": frame_idx, "reason": "no_prompt"})
            continue
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise RuntimeError(f"failed to read RGB/depth/mask for frame {frame_idx}")
        depth_m = depth.astype(np.float64) / 1000.0
        seeds = seed_pixels(prompt, prompt_size, mask.shape)
        grown = grow(mask, depth_m, seeds, args)
        area = int(np.count_nonzero(grown))
        if area < int(args.min_mask_pixels):
            rows.append({"frame_idx": frame_idx, "reason": "grown_area_too_small", "grown_area_px": area, "seeds": len(seeds)})
            continue
        out_index = len(out_entries)
        stem = f"{out_index:06d}"
        rgb_path = args.output_dir / "rgb" / f"{stem}.png"
        depth_path = args.output_dir / "depth" / f"{stem}.png"
        mask_path = args.output_dir / "masks" / f"{stem}.png"
        if not cv2.imwrite(str(rgb_path), rgb):
            raise RuntimeError(f"failed to write {rgb_path}")
        if not cv2.imwrite(str(depth_path), depth):
            raise RuntimeError(f"failed to write {depth_path}")
        if not cv2.imwrite(str(mask_path), grown.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        valid_depth = depth_m[grown]
        source_index = int(entry.get("source_index", entry["index"]))
        out_entries.append(
            {
                "index": out_index,
                "source_index": source_index,
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "depth": str(depth_path),
                "mask": str(mask_path),
                "mask_area_px": area,
                "mask_depth_median_m": float(np.median(valid_depth)),
                "mask_depth_p05_m": float(np.percentile(valid_depth, 5)),
                "mask_depth_p95_m": float(np.percentile(valid_depth, 95)),
                "track_id": args.track_id,
                "label": args.label,
            }
        )
        rows.append(
            {
                "frame_idx": frame_idx,
                "reason": "ok",
                "source_index": source_index,
                "original_area_px": int(np.count_nonzero(mask)),
                "grown_area_px": area,
                "seeds": len(seeds),
            }
        )
    if len(out_entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(out_entries)} grown-mask frames survived")

    (args.output_dir / "manifest.json").write_text(json.dumps({"frames": out_entries}, indent=2), encoding="utf-8")
    areas = np.asarray([entry["mask_area_px"] for entry in out_entries], dtype=np.float64)
    med_depths = np.asarray([entry["mask_depth_median_m"] for entry in out_entries], dtype=np.float64)
    qc = {
        "status": "ok",
        "method": "anchor_seed_depth_manifold_growth_v3",
        "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest),
        "point_prompts": str(args.point_prompts),
        "output_dir": str(args.output_dir),
        "frames": int(len(out_entries)),
        "first_frame": int(out_entries[0]["frame_idx"]),
        "last_frame": int(out_entries[-1]["frame_idx"]),
        "track_id": args.track_id,
        "depth_band_m": float(args.depth_band_m),
        "local_depth_step_m": float(args.local_depth_step_m),
        "mask_area_median_px": float(np.median(areas)),
        "mask_area_min_px": int(np.min(areas)),
        "mask_area_max_px": int(np.max(areas)),
        "depth_median_m": float(np.median(med_depths)),
        "depth_p05_m": float(np.percentile(med_depths, 5)),
        "depth_p95_m": float(np.percentile(med_depths, 95)),
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "manifest": str(args.output_dir / "manifest.json"),
        "rows": rows,
    }
    (args.output_dir / "qc_bundlesdf_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k != "rows"}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--depth-band-m", type=float, default=0.06)
    parser.add_argument("--local-depth-step-m", type=float, default=0.015)
    parser.add_argument("--seed-depth-radius-px", type=int, default=2)
    parser.add_argument("--min-seed-depths", type=int, default=3)
    parser.add_argument("--min-mask-pixels", type=int, default=5000)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
