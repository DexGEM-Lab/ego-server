#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


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


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def rows_from_manifest(path: Path, start: int, end: int, stride: int) -> list[dict[str, Any]]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"manifest_has_no_frames: {path}")
    rows = [row for row in frames if start <= int(row["frame_idx"]) <= end and (int(row["frame_idx"]) - start) % max(1, stride) == 0]
    rows.sort(key=lambda row: int(row["frame_idx"]))
    if not rows:
        raise ContractError("no_frames_selected")
    return rows


def to_depth_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ContractError(f"invalid_depth_shape: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ContractError("depth_contains_nonfinite")
    return arr.astype(np.float32)


def scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().reshape(-1)[0])
    arr = np.asarray(value)
    if arr.size != 1:
        raise ContractError(f"expected_scalar: shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def resize_depth(depth: np.ndarray, source_height: int, source_width: int) -> np.ndarray:
    if depth.shape == (source_height, source_width):
        return depth
    return cv2.resize(depth, (source_width, source_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def render_depth(rgb_path: Path, depth: np.ndarray, out_path: Path) -> None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ContractError(f"could_not_read_rgb: {rgb_path}")
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ContractError("depth_has_no_valid_pixels")
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    color = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    if color.shape[:2] != rgb.shape[:2]:
        color = cv2.resize(color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    review = cv2.addWeighted(rgb, 0.55, color, 0.45, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), review):
        raise ContractError(f"could_not_write_review: {out_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.depthpro_repo is not None:
        sys.path.insert(0, str(args.depthpro_repo / "src"))
    import depth_pro  # noqa: PLC0415

    if not args.cpu and not torch.cuda.is_available():
        raise ContractError("Depth Pro requires CUDA unless --cpu is explicit")
    device = torch.device("cpu" if args.cpu else "cuda")
    previous_cwd = Path.cwd()
    if args.depthpro_repo is not None:
        os.chdir(args.depthpro_repo)
    try:
        model, transform = depth_pro.create_model_and_transforms()
    finally:
        os.chdir(previous_cwd)
    model.eval().to(device)
    rows = rows_from_manifest(args.raw_frame_manifest, int(args.frame_start), int(args.frame_end), int(args.frame_stride))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    still_dir = args.output_dir / "stills"
    depth_png_dir = args.output_dir / "depth_png"
    depth_stack = []
    frame_indices = []
    focal_values = []
    intrinsics_stack = []
    row_reports = []
    for local_i, row in enumerate(rows):
        frame_idx = int(row["frame_idx"])
        rgb_path = Path(str(row["rgb"]))
        image_pil, _, exif_focal = depth_pro.load_rgb(rgb_path)
        image_tensor = transform(image_pil).to(device)
        with torch.no_grad():
            pred = model.infer(image_tensor, f_px=exif_focal)
        depth = resize_depth(to_depth_array(pred["depth"]), int(args.source_height), int(args.source_width))
        focal_px = scalar(pred["focallength_px"])
        valid = np.isfinite(depth) & (depth > 0)
        if np.count_nonzero(valid) < int(args.min_valid_pixels):
            raise ContractError(f"too_few_valid_depth_pixels: frame={frame_idx}")
        depth_stack.append(depth.astype(np.float16))
        frame_indices.append(frame_idx)
        focal_values.append(float(focal_px))
        intrinsics_stack.append([float(focal_px), float(focal_px), float(args.source_width) / 2.0, float(args.source_height) / 2.0])
        depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
        depth_png = depth_png_dir / f"{frame_idx:06d}.png"
        depth_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(depth_png), depth_mm)
        if local_i in {0, len(rows) // 2, len(rows) - 1}:
            render_depth(rgb_path, depth, still_dir / f"frame_{frame_idx:06d}.jpg")
        vals = depth[valid].astype(np.float64)
        row_reports.append(
            {
                "frame_idx": frame_idx,
                "rgb": str(rgb_path),
                "depth_png": str(depth_png),
                "depthpro_focal_px": float(focal_px),
                "depth_median_m": float(np.median(vals)),
                "depth_p05_m": float(np.percentile(vals, 5)),
                "depth_p95_m": float(np.percentile(vals, 95)),
                "valid_fraction": float(np.mean(valid)),
            }
        )
    archive = args.output_dir / "depthpro_full_frame_depth_v21.npz"
    np.savez_compressed(
        archive,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        depth=np.stack(depth_stack, axis=0),
        source_size=np.asarray([int(args.source_width), int(args.source_height)], dtype=np.int32),
        focal_px=np.asarray(focal_values, dtype=np.float32),
        intrinsics_fx_fy_cx_cy=np.asarray(intrinsics_stack, dtype=np.float32),
    )
    report = {
        "schema": "v21_depthpro_full_frame_candidate.v0",
        "status": "ok",
        "annotation_ready": False,
        "method": "run_v21_depthpro_full_frame_candidate",
        "candidate_id": "depthpro_rgb_metric_depth_baseline",
        "raw_frame_manifest": str(args.raw_frame_manifest),
        "frame_stride": int(args.frame_stride),
        "frame_count": int(len(row_reports)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "depth_archive": str(archive),
        "stills_dir": str(still_dir),
        "depthpro_focal_px": summarize(focal_values),
        "depth_median_m": summarize([row["depth_median_m"] for row in row_reports]),
        "valid_fraction": summarize([row["valid_fraction"] for row in row_reports]),
        "rows": row_reports,
        "elapsed_s": float(time.time() - started),
        "claim_scope": "Depth Pro RGB-only metric depth/focal baseline candidate. It must be compared with UniDepth/native/stereo/multiview evidence before depth/camera selection.",
    }
    write_json(args.output_dir / "qc_depthpro_full_frame_v21.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V21 Depth Pro full-frame monocular depth candidate on raw-frame manifest frames.")
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depthpro-repo", type=Path)
    parser.add_argument("--source-width", type=int, required=True)
    parser.add_argument("--source-height", type=int, required=True)
    parser.add_argument("--min-valid-pixels", type=int, default=100000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
