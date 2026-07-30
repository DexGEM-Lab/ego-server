#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_depth(path: Path) -> dict:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy", "source_size"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    return {
        "frame_to_i": {int(idx): i for i, idx in enumerate(frame_idx.tolist())},
        "depth": blob["depth"].astype(np.float32),
        "intrinsics": blob["intrinsics_fx_fy_cx_cy"].astype(np.float64),
        "source_size": blob["source_size"].astype(int),
    }


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video}")
    return frame


def save_depth_mm(path: Path, depth_m: np.ndarray) -> None:
    if not np.isfinite(depth_m).any():
        raise RuntimeError("depth map contains no finite values")
    depth_mm = np.clip(depth_m.astype(np.float64) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"failed to write {path}")


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    depth = load_depth(args.metric_depth_npz)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{args.annotations} must contain nonempty frames list")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb"
    mask_dir = args.output_dir / "masks"
    depth_dir = args.output_dir / "depth"
    for directory in (rgb_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)
    entries = []
    rows = []
    cam_k_values = []
    out_i = 0
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        obj = frame.get("object", {})
        if not str(obj.get("status", "")).startswith("measured") or not obj.get("mask_path"):
            rows.append({"frame_idx": frame_idx, "status": "skipped_unobserved_object"})
            continue
        if frame_idx not in depth["frame_to_i"]:
            rows.append({"frame_idx": frame_idx, "status": "skipped_missing_depth"})
            continue
        mask = cv2.imread(str(obj["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask {obj['mask_path']}")
        depth_i = int(depth["frame_to_i"][frame_idx])
        depth_m = depth["depth"][depth_i]
        if mask.shape != depth_m.shape:
            mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb = read_frame(args.video, frame_idx)
        if rgb.shape[:2] != depth_m.shape:
            rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)
        rgb_path = rgb_dir / f"{out_i:06d}.jpg"
        mask_path = mask_dir / f"{out_i:06d}.png"
        depth_path = depth_dir / f"{out_i:06d}.png"
        if not cv2.imwrite(str(rgb_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to write {rgb_path}")
        if not cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        save_depth_mm(depth_path, depth_m)
        fx, fy, cx, cy = depth["intrinsics"][depth_i].tolist()
        K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        cam_k_values.append(K)
        entries.append(
            {
                "index": int(out_i),
                "frame_idx": int(frame_idx),
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "depth": str(depth_path),
                "source_mask": str(obj["mask_path"]),
                "intrinsics_fx_fy_cx_cy": [float(fx), float(fy), float(cx), float(cy)],
            }
        )
        rows.append({"frame_idx": frame_idx, "status": "ok", "index": int(out_i), "mask_pixels": int(np.count_nonzero(mask))})
        out_i += 1
    if len(entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(entries)} measured frames exported")
    cam_k = np.median(np.stack(cam_k_values, axis=0), axis=0)
    np.savetxt(args.output_dir / "cam_K.txt", cam_k, fmt="%.10f")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": entries}, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "export_annotation_mask_unidepth_dataset_v3",
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "manifest": str(manifest_path),
        "frames": int(len(entries)),
        "first_frame": int(entries[0]["frame_idx"]),
        "last_frame": int(entries[-1]["frame_idx"]),
        "intrinsics_median_fx_fy_cx_cy": [
            float(cam_k[0, 0]),
            float(cam_k[1, 1]),
            float(cam_k[0, 2]),
            float(cam_k[1, 2]),
        ],
        "rows": rows,
    }
    (args.output_dir / "qc_annotation_mask_unidepth_dataset_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
