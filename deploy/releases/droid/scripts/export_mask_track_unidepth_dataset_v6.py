#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def load_depth(path: Path) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=np.int64)
    depth = np.asarray(blob["depth"], dtype=np.float32)
    intrinsics = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    if depth.ndim != 3 or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(f"invalid depth/intrinsics shapes in {path}: {depth.shape}, {intrinsics.shape}")
    frame_to_i = {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())}
    if len(frame_to_i) != len(frame_idx):
        raise RuntimeError(f"duplicate frame ids in {path}")
    return frame_to_i, depth, intrinsics


def read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video}")
    return frame


def write_depth_mm(path: Path, depth_m: np.ndarray) -> None:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float64) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"failed to write {path}")


def mask_row(track: dict, frame_idx: int) -> dict:
    key = str(int(frame_idx))
    if key not in track:
        raise RuntimeError(f"track lacks frame {frame_idx}")
    row = track[key]
    if not isinstance(row, dict):
        raise RuntimeError(f"invalid track row for frame {frame_idx}")
    if not row.get("visible"):
        raise RuntimeError(f"track row for frame {frame_idx} is not visible")
    if not row.get("mask_path"):
        raise RuntimeError(f"visible track row for frame {frame_idx} lacks mask_path")
    return row


def run(args: argparse.Namespace) -> dict:
    track = load_json(args.mask_track)
    frame_to_depth_i, depths, intrinsics = load_depth(args.metric_depth_npz)
    frame_ids = list(range(int(args.frame_start), int(args.frame_end) + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.output_dir / "rgb"
    mask_dir = args.output_dir / "masks"
    depth_dir = args.output_dir / "depth"
    for directory in (rgb_dir, mask_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries = []
    rows = []
    for out_i, frame_idx in enumerate(frame_ids):
        row = mask_row(track, frame_idx)
        if frame_idx not in frame_to_depth_i:
            raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask for frame {frame_idx}: {row['mask_path']}")
        depth_i = int(frame_to_depth_i[frame_idx])
        depth_m = depths[depth_i]
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
        write_depth_mm(depth_path, depth_m)
        fx, fy, cx, cy = [float(v) for v in intrinsics[depth_i].tolist()]
        entries.append(
            {
                "index": int(out_i),
                "frame_idx": int(frame_idx),
                "rgb": str(rgb_path),
                "mask": str(mask_path),
                "depth": str(depth_path),
                "source_mask": str(row["mask_path"]),
                "source_track": str(args.mask_track),
                "source": str(row.get("source", "mask_track")),
                "intrinsics_fx_fy_cx_cy": [fx, fy, cx, cy],
            }
        )
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "status": "ok",
                "index": int(out_i),
                "mask_pixels": int(np.count_nonzero(mask)),
                "source_mask": str(row["mask_path"]),
            }
        )
    if len(entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(entries)} frames exported")
    cam = np.median(np.asarray([[e["intrinsics_fx_fy_cx_cy"][0], e["intrinsics_fx_fy_cx_cy"][1], e["intrinsics_fx_fy_cx_cy"][2], e["intrinsics_fx_fy_cx_cy"][3]] for e in entries], dtype=np.float64), axis=0)
    cam_k = np.asarray([[cam[0], 0.0, cam[2]], [0.0, cam[1], cam[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(args.output_dir / "cam_K.txt", cam_k, fmt="%.10f")
    manifest = {
        "status": "ok",
        "method": "export_mask_track_unidepth_dataset_v6",
        "video": str(args.video),
        "mask_track": str(args.mask_track),
        "metric_depth_npz": str(args.metric_depth_npz),
        "dataset_dir": str(args.output_dir),
        "frames": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "export_mask_track_unidepth_dataset_v6",
        "manifest": str(manifest_path),
        "frames": int(len(entries)),
        "first_frame": int(entries[0]["frame_idx"]),
        "last_frame": int(entries[-1]["frame_idx"]),
        "rows": rows,
    }
    (args.output_dir / "qc_mask_track_unidepth_dataset_v6.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask-track", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
