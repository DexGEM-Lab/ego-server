#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_intrinsics(source_dataset: Path, source_manifest: dict) -> list[float]:
    qc_path = source_dataset / "qc_bundlesdf_dataset_v3.json"
    if qc_path.exists():
        values = load_json(qc_path).get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return [float(v) for v in values]
    values = source_manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return [float(v) for v in values]
    K = np.loadtxt(source_dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3):
        raise RuntimeError(f"{source_dataset / 'cam_K.txt'} must be a 3x3 matrix")
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def write_cam_k(path: Path, intrinsics: list[float]) -> None:
    fx, fy, cx, cy = intrinsics
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, K, fmt="%.10f")


def local_mask_path(mask_path: str, remote_root: Path, local_root: Path) -> Path:
    path = Path(mask_path)
    if path.exists():
        return path
    if remote_root and local_root:
        try:
            rel = path.resolve().relative_to(remote_root.resolve())
            candidate = local_root / rel
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    raise RuntimeError(f"mask path does not exist locally: {mask_path}")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.shape == (height, width):
        return (mask > 0).astype(np.uint8) * 255
    return cv2.resize((mask > 0).astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)


def run(args: argparse.Namespace) -> dict:
    source_manifest = load_json(args.source_manifest)
    source_entries = source_manifest.get("frames")
    if not isinstance(source_entries, list) or not source_entries:
        raise RuntimeError("source manifest must contain a nonempty frames list")
    track = load_json(args.mask_track)
    intrinsics = load_intrinsics(args.source_dataset, source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "depth", "masks"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_cam_k(args.output_dir / "cam_K.txt", intrinsics)

    manifest_entries = []
    mask_areas = []
    depth_medians = []
    for source_entry in source_entries:
        frame_idx = int(source_entry["frame_idx"])
        result = track.get(str(frame_idx))
        if not result or not result.get("visible") or not result.get("mask_path"):
            continue
        src_index = int(source_entry["index"])
        out_index = len(manifest_entries)
        rgb_src = Path(source_entry["rgb"])
        depth_src = Path(source_entry["depth"])
        if not rgb_src.exists() or not depth_src.exists():
            raise RuntimeError(f"missing source RGB/depth for frame {frame_idx}")
        rgb = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError(f"failed to read source RGB/depth for frame {frame_idx}")
        mask_path = local_mask_path(str(result["mask_path"]), args.remote_data_root, args.local_data_root)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise RuntimeError(f"failed to read mask-track mask {mask_path}")
        mask = resize_mask(mask_raw, (rgb.shape[1], rgb.shape[0]))
        mask_area = int(np.count_nonzero(mask))
        if mask_area < int(args.min_mask_pixels):
            continue
        depth_m = depth.astype(np.float64) / 1000.0
        valid_depth = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0.05)]
        if valid_depth.size < int(args.min_depth_pixels):
            continue

        stem = f"{out_index:06d}"
        rgb_out = args.output_dir / "rgb" / f"{stem}.png"
        depth_out = args.output_dir / "depth" / f"{stem}.png"
        mask_out = args.output_dir / "masks" / f"{stem}.png"
        shutil.copy2(rgb_src, rgb_out)
        shutil.copy2(depth_src, depth_out)
        if not cv2.imwrite(str(mask_out), mask):
            raise RuntimeError(f"failed to write {mask_out}")
        mask_areas.append(mask_area)
        depth_medians.append(float(np.median(valid_depth)))
        manifest_entries.append(
            {
                "index": out_index,
                "source_index": src_index,
                "frame_idx": frame_idx,
                "rgb": str(rgb_out),
                "depth": str(depth_out),
                "mask": str(mask_out),
                "mask_area_px": mask_area,
                "mask_depth_median_m": float(np.median(valid_depth)),
                "mask_depth_p05_m": float(np.percentile(valid_depth, 5)),
                "mask_depth_p95_m": float(np.percentile(valid_depth, 95)),
                "source_track_area_px": float(result.get("area_px", 0.0)),
                "track_id": args.track_id,
                "label": args.label,
            }
        )
    if len(manifest_entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(manifest_entries)} mask-track frames survived export")

    manifest = {"frames": manifest_entries}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    qc = {
        "status": "ok",
        "method": "mask_track_to_bundlesdf_dataset_v3",
        "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest),
        "mask_track": str(args.mask_track),
        "output_dir": str(args.output_dir),
        "frames": int(len(manifest_entries)),
        "first_frame": int(manifest_entries[0]["frame_idx"]),
        "last_frame": int(manifest_entries[-1]["frame_idx"]),
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "mask_area_median_px": float(np.median(mask_areas)),
        "mask_area_min_px": int(np.min(mask_areas)),
        "mask_area_max_px": int(np.max(mask_areas)),
        "depth_median_m": float(np.median(depth_medians)),
        "depth_p05_m": float(np.percentile(depth_medians, 5)),
        "depth_p95_m": float(np.percentile(depth_medians, 95)),
        "manifest": str(args.output_dir / "manifest.json"),
    }
    (args.output_dir / "qc_bundlesdf_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mask-track", "--sam2-track", dest="mask_track", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--remote-data-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/data"))
    parser.add_argument("--local-data-root", type=Path, default=Path("/data2/ego_annotation_outputs"))
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--min-depth-pixels", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
