#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fuse_v1_full_fidelity import load_json, open_video, read_video_frame
from reconstruct_object_mesh_v2 import localize_mask_path


def resize_metric_depth_to_source(depth: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    width, height = source_size
    if depth.shape == (height, width):
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def mask_to_source(mask: np.ndarray, mask_size: tuple[int, int], source_size: tuple[int, int]) -> np.ndarray:
    source_w, source_h = source_size
    if mask.shape == (source_h, source_w):
        return (mask > 0).astype(np.uint8) * 255
    if mask.shape != (mask_size[1], mask_size[0]):
        raise RuntimeError(f"mask shape {mask.shape} does not match declared mask size {mask_size}")
    return cv2.resize((mask > 0).astype(np.uint8) * 255, (source_w, source_h), interpolation=cv2.INTER_NEAREST)


def load_metric_depth(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    if depth.ndim != 3 or len(frames) != depth.shape[0]:
        raise RuntimeError(f"{path} has invalid frame/depth shapes: {frames.shape}, {depth.shape}")
    return {
        "frame_to_i": {int(frame_idx): i for i, frame_idx in enumerate(frames)},
        "depth": depth,
        "source_size": tuple(int(v) for v in blob["source_size"].tolist()) if "source_size" in blob.files else None,
    }


def selected_frames(annotations: dict, frame_start: int, frame_end: int, track_id: str | None) -> list[dict]:
    frames = []
    for frame in annotations["frames"]:
        frame_idx = int(frame["frame_idx"])
        obj = frame.get("object") or {}
        if frame_idx < frame_start or frame_idx > frame_end:
            continue
        status = str(obj.get("status") or "")
        if not status.startswith("measured_") or not obj.get("mask_path"):
            continue
        if track_id is not None and obj.get("track_id") != track_id:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("no measured object-mask frames selected for BundleSDF export")
    return frames


def write_cam_k(path: Path, intrinsics: np.ndarray) -> None:
    fx, fy, cx, cy = intrinsics.astype(float)
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    np.savetxt(path, K, fmt="%.10f")


def load_npz_intrinsics(path: Path) -> np.ndarray:
    blob = np.load(path)
    for key in ("intrinsics_source", "intrinsics"):
        if key in blob.files:
            intrinsics = np.asarray(blob[key], dtype=float)
            if intrinsics.shape == (4,) and np.isfinite(intrinsics).all():
                return intrinsics
    raise RuntimeError(f"{path} lacks finite source intrinsics")


def load_annotation_intrinsics(frames: list[dict]) -> np.ndarray:
    rows = []
    for frame in frames:
        camera = frame.get("camera")
        if not isinstance(camera, dict) or "vggt_source_intrinsics_fx_fy_cx_cy" not in camera:
            raise RuntimeError(
                f"frame {frame.get('frame_idx')} lacks camera.vggt_source_intrinsics_fx_fy_cx_cy"
            )
        vals = np.asarray(camera["vggt_source_intrinsics_fx_fy_cx_cy"], dtype=np.float64)
        if vals.shape != (4,) or not np.isfinite(vals).all():
            raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid annotation intrinsics: {vals}")
        rows.append(vals)
    intrinsics = np.asarray(rows, dtype=np.float64)
    spread = np.ptp(intrinsics, axis=0)
    if np.max(spread) > 1e-5:
        raise RuntimeError(f"BundleSDF export requires constant intrinsics, got spread {spread.tolist()}")
    return intrinsics[0]


def load_intrinsics(args: argparse.Namespace, frames: list[dict]) -> tuple[np.ndarray, str]:
    if args.intrinsics_source == "droid-npz":
        if args.droid_npz is None:
            raise RuntimeError("--droid-npz is required when --intrinsics-source=droid-npz")
        return load_npz_intrinsics(args.droid_npz), f"droid-npz:{args.droid_npz}"
    if args.intrinsics_source == "annotation-vggt":
        return load_annotation_intrinsics(frames), "annotation-vggt"
    raise RuntimeError(f"unsupported intrinsics source: {args.intrinsics_source}")


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = selected_frames(annotations, int(args.frame_start), int(args.frame_end), args.track_id)
    metric_depth = load_metric_depth(args.metric_depth_npz)
    intrinsics, intrinsics_source = load_intrinsics(args, frames)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise RuntimeError(f"invalid source intrinsics: {intrinsics}")
    cap, info = open_video(args.clip)
    source_size = (int(info.width), int(info.height))
    out = args.output_dir
    for sub in ("rgb", "depth", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    write_cam_k(out / "cam_K.txt", intrinsics)
    manifest = []
    mask_areas = []
    depth_medians = []
    try:
        for out_i, frame in enumerate(frames):
            frame_idx = int(frame["frame_idx"])
            obj = frame["object"]
            if frame_idx not in metric_depth["frame_to_i"]:
                raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
            rgb = read_video_frame(cap, frame_idx)
            depth = metric_depth["depth"][metric_depth["frame_to_i"][frame_idx]]
            depth_source = resize_metric_depth_to_source(depth, source_size)
            depth_mm = np.clip(np.rint(depth_source * 1000.0), 0, 65535).astype(np.uint16)
            mask_path = localize_mask_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"failed to read mask {mask_path}")
            mask_size = tuple(int(v) for v in obj.get("mask_image_size", source_size))
            mask_source = mask_to_source(mask, mask_size, source_size)
            stem = f"{out_i:06d}"
            rgb_path = out / "rgb" / f"{stem}.png"
            depth_path = out / "depth" / f"{stem}.png"
            mask_out = out / "masks" / f"{stem}.png"
            if not cv2.imwrite(str(rgb_path), rgb):
                raise RuntimeError(f"failed to write {rgb_path}")
            if not cv2.imwrite(str(depth_path), depth_mm):
                raise RuntimeError(f"failed to write {depth_path}")
            if not cv2.imwrite(str(mask_out), mask_source):
                raise RuntimeError(f"failed to write {mask_out}")
            valid_depth = depth_source[(mask_source > 0) & np.isfinite(depth_source) & (depth_source > 0.05)]
            if valid_depth.size == 0:
                raise RuntimeError(f"mask has no valid metric depth at frame {frame_idx}")
            mask_area = int((mask_source > 0).sum())
            mask_areas.append(mask_area)
            depth_medians.append(float(np.median(valid_depth)))
            manifest.append(
                {
                    "index": int(out_i),
                    "frame_idx": frame_idx,
                    "rgb": str(rgb_path),
                    "depth": str(depth_path),
                    "mask": str(mask_out),
                    "mask_area_px": mask_area,
                    "mask_depth_median_m": float(np.median(valid_depth)),
                    "mask_depth_p05_m": float(np.percentile(valid_depth, 5)),
                    "mask_depth_p95_m": float(np.percentile(valid_depth, 95)),
                    "track_id": obj.get("track_id"),
                    "label": obj.get("label"),
                }
            )
    finally:
        cap.release()
    qc = {
        "status": "ok",
        "method": "bundlesdf_dataset_export",
        "clip": str(args.clip),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "droid_npz": str(args.droid_npz) if args.droid_npz is not None else None,
        "output_dir": str(out),
        "frames": len(manifest),
        "first_frame": int(manifest[0]["frame_idx"]),
        "last_frame": int(manifest[-1]["frame_idx"]),
        "source_size": [int(source_size[0]), int(source_size[1])],
        "intrinsics_fx_fy_cx_cy": intrinsics.astype(float).tolist(),
        "intrinsics_source": intrinsics_source,
        "mask_area_median_px": float(np.median(mask_areas)),
        "mask_area_min_px": int(np.min(mask_areas)),
        "depth_median_m": float(np.median(depth_medians)),
        "depth_p05_m": float(np.percentile(depth_medians, 5)),
        "depth_p95_m": float(np.percentile(depth_medians, 95)),
        "manifest": str(out / "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps({"frames": manifest}, indent=2), encoding="utf-8")
    (out / "qc_bundlesdf_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path)
    parser.add_argument("--intrinsics-source", choices=("droid-npz", "annotation-vggt"), default="droid-npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--track-id")
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
