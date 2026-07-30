#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_vggt_native_camera_v3 import mask_depth_rows, make_local_world_poses, summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def run(args: argparse.Namespace) -> dict:
    blob = np.load(args.vggt_archive)
    required = {
        "frame_idx",
        "extrinsic",
        "intrinsic",
        "depth",
        "depth_conf",
        "camera_centers_vggt",
        "source_intrinsics_fx_fy_cx_cy",
        "anchor_frame",
    }
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{args.vggt_archive} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(np.int32)
    if int(args.frame_start) not in frame_idx or int(args.frame_end) not in frame_idx:
        raise RuntimeError("requested scale interval must be contained in VGGT archive")
    keep = (frame_idx >= int(args.frame_start)) & (frame_idx <= int(args.frame_end))
    rows_all = load_json(args.dataset_manifest).get("frames")
    if not isinstance(rows_all, list) or not rows_all:
        raise RuntimeError(f"{args.dataset_manifest} must contain nonempty frames")
    rows = [row for row in rows_all if int(args.frame_start) <= int(row["frame_idx"]) <= int(args.frame_end)]
    if len(rows) != int(np.count_nonzero(keep)):
        raise RuntimeError("VGGT frame interval and dataset manifest interval differ")
    frame_idx_out = frame_idx[keep]
    extrinsic = blob["extrinsic"][keep].astype(np.float64)
    intrinsic = blob["intrinsic"][keep].astype(np.float64)
    depth = blob["depth"][keep].astype(np.float32)
    depth_conf = blob["depth_conf"][keep].astype(np.float32)
    centers = blob["camera_centers_vggt"][keep].astype(np.float64)
    anchor_frame = int(args.anchor_frame) if args.anchor_frame is not None else int(blob["anchor_frame"][0])
    if anchor_frame not in frame_idx_out.tolist():
        raise RuntimeError(f"anchor frame {anchor_frame} is absent from requested interval")
    anchor_i = frame_idx_out.tolist().index(anchor_frame)
    depth_rows, scale = mask_depth_rows(rows, depth, intrinsic, args)
    transforms = make_local_world_poses(extrinsic, centers, float(scale), anchor_i)
    source_intrinsics = blob["source_intrinsics_fx_fy_cx_cy"][keep].astype(np.float32)
    center_speed = []
    if len(transforms) > 1:
        dt = np.diff(frame_idx_out.astype(np.float64)) / float(args.fps)
        steps = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1)
        center_speed = (steps / np.maximum(dt, 1e-9)).astype(float).tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "vggt_native_camera_v3.npz"
    passthrough = {
        key: blob[key][keep]
        for key in ("points_vggt", "masks", "rgbs")
        if key in blob.files
    }
    np.savez_compressed(
        archive,
        frame_idx=frame_idx_out,
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth.astype(np.float16),
        depth_conf=depth_conf.astype(np.float16),
        camera_centers_vggt=centers.astype(np.float32),
        T_world_camera_metric=transforms.astype(np.float32),
        source_intrinsics_fx_fy_cx_cy=source_intrinsics.astype(np.float32),
        vggt_to_meters=np.asarray([float(scale)], dtype=np.float32),
        anchor_frame=np.asarray([anchor_frame], dtype=np.int32),
        **passthrough,
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "patch_vggt_native_camera_scale_v3",
        "source_vggt_archive": str(args.vggt_archive),
        "dataset_manifest": str(args.dataset_manifest),
        "metric_depth_manifest": str(args.metric_depth_manifest),
        "archive": str(archive),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": frame_idx_out.astype(int).tolist(),
        "anchor_frame": int(anchor_frame),
        "vggt_to_meters": float(scale),
        "center_speed_m_s": summarize(center_speed),
        "metric_depth_rows": depth_rows,
    }
    (args.output_dir / "qc_patch_vggt_native_camera_scale_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "metric_depth_rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--metric-depth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-remote-root", type=Path)
    parser.add_argument("--metric-local-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-depth-pixels", type=int, default=500)
    parser.add_argument("--vggt-to-meters", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
