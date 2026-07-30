#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from patch_annotations_with_vggt_poses_v3 import source_intrinsics
from run_vggt_object_geometry_v3 import camera_centers_from_vggt, run_vggt
from run_vggt_scene_geometry_v3 import load_scene_views, read_manifest


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def localize(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src, dst in ((local_root, remote_root), (remote_root, local_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def source_mask_to_vggt(mask: np.ndarray, source_width: int, source_height: int, target_size: int) -> np.ndarray:
    if mask.shape != (source_height, source_width):
        raise RuntimeError(f"source mask shape {mask.shape} does not match {(source_height, source_width)}")
    if source_width >= source_height:
        new_width = int(target_size)
        new_height = round(source_height * (new_width / source_width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(source_width * (new_height / source_height) / 14) * 14
    if new_width <= 0 or new_height <= 0:
        raise RuntimeError("invalid VGGT mask preprocessing dimensions")
    small = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    pad_top = (target_size - new_height) // 2
    pad_left = (target_size - new_width) // 2
    out = np.zeros((target_size, target_size), dtype=np.uint8)
    out[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = np.where(small > 0, 255, 0).astype(np.uint8)
    return out > 0


def mask_depth_rows(
    rows: list[dict],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], float]:
    if args.metric_depth_manifest is None:
        return [], float(args.vggt_to_meters)
    metric_payload = load_json(args.metric_depth_manifest)
    metric_rows = metric_payload.get("frames")
    if not isinstance(metric_rows, list) or not metric_rows:
        raise RuntimeError(f"{args.metric_depth_manifest} must contain nonempty frames")
    metric_by_frame = {int(row["frame_idx"]): row for row in metric_rows}
    out = []
    ratios = []
    for i, row in enumerate(rows):
        frame_idx = int(row["frame_idx"])
        metric = metric_by_frame.get(frame_idx)
        if metric is None:
            continue
        depth_path = localize(str(metric["depth"]), args.metric_remote_root, args.metric_local_root)
        metric_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if metric_depth is None:
            raise RuntimeError(f"failed to read metric depth: {depth_path}")
        metric_depth_m = metric_depth.astype(np.float64) / 1000.0
        mask_path = localize(str(metric["mask"]), args.metric_remote_root, args.metric_local_root)
        source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source_mask is None:
            raise RuntimeError(f"failed to read metric mask: {mask_path}")
        if source_mask.shape != metric_depth_m.shape:
            raise RuntimeError(f"metric mask/depth shape mismatch for frame {frame_idx}")
        mask_small = source_mask_to_vggt(
            source_mask,
            int(args.source_width),
            int(args.source_height),
            int(args.target_size),
        )
        vggt_depth = depth[i].astype(np.float64)
        vggt_vals = vggt_depth[mask_small & np.isfinite(vggt_depth) & (vggt_depth > 0.0)]
        metric_vals = metric_depth_m[(source_mask > 0) & np.isfinite(metric_depth_m) & (metric_depth_m > 0.0)]
        if len(vggt_vals) < int(args.min_depth_pixels) or len(metric_vals) < int(args.min_depth_pixels):
            raise RuntimeError(f"too few depth pixels for metric scale on frame {frame_idx}")
        vggt_median = float(np.median(vggt_vals))
        metric_median = float(np.median(metric_vals))
        ratio = metric_median / vggt_median
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise RuntimeError(f"invalid metric/VGGT depth ratio on frame {frame_idx}: {ratio}")
        out.append(
            {
                "frame_idx": frame_idx,
                "vggt_mask_depth_median_native": vggt_median,
                "metric_mask_depth_median_m": metric_median,
                "metric_depth_source": str(depth_path),
                "source_intrinsics_fx_fy_cx_cy": source_intrinsics(
                    intrinsics[i],
                    int(args.source_width),
                    int(args.source_height),
                    int(args.target_size),
                ),
                "metric_to_vggt_depth_ratio": float(ratio),
            }
        )
        ratios.append(float(ratio))
    if not ratios:
        raise RuntimeError("metric depth manifest supplied, but no overlapping frames were found")
    return out, float(np.median(np.asarray(ratios, dtype=np.float64)))


def make_local_world_poses(extrinsic: np.ndarray, centers: np.ndarray, scale: float, anchor_i: int) -> np.ndarray:
    R_anchor = extrinsic[anchor_i, :3, :3]
    c_anchor = centers[anchor_i]
    poses = []
    for i in range(len(extrinsic)):
        R_world_camera = R_anchor @ extrinsic[i, :3, :3].T
        t_world_camera = scale * (R_anchor @ (centers[i] - c_anchor))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_world_camera
        T[:3, 3] = t_world_camera
        poses.append(T)
    return np.stack(poses, axis=0)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    rows = read_manifest(args.dataset_manifest, int(args.frame_start), int(args.frame_end))
    frame_indices = [int(row["frame_idx"]) for row in rows]
    images, masks, rgbs, view_reports = load_scene_views(
        rows,
        int(args.target_size),
        args.remote_output_root,
        args.local_output_root,
    )
    extrinsic, intrinsic, depth, depth_conf, points_vggt = run_vggt(args, images)
    centers = camera_centers_from_vggt(extrinsic)
    depth_rows, scale = mask_depth_rows(rows, depth, intrinsic, args)
    anchor_frame = int(args.anchor_frame) if args.anchor_frame is not None else int(frame_indices[0])
    if anchor_frame not in frame_indices:
        raise RuntimeError(f"anchor frame {anchor_frame} is absent from VGGT frames")
    anchor_i = frame_indices.index(anchor_frame)
    T_world_camera = make_local_world_poses(extrinsic, centers, float(scale), anchor_i)
    source_intr = np.asarray(
        [
            source_intrinsics(intrinsic[i], int(args.source_width), int(args.source_height), int(args.target_size))
            for i in range(len(frame_indices))
        ],
        dtype=np.float32,
    )
    center_speed = []
    if len(T_world_camera) > 1:
        dt = np.diff(np.asarray(frame_indices, dtype=np.float64)) / float(args.fps)
        steps = np.linalg.norm(np.diff(T_world_camera[:, :3, 3], axis=0), axis=1)
        center_speed = (steps / np.maximum(dt, 1e-9)).astype(float).tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "vggt_native_camera_v3.npz"
    np.savez_compressed(
        archive,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth.astype(np.float16),
        depth_conf=depth_conf.astype(np.float16),
        camera_centers_vggt=centers.astype(np.float32),
        T_world_camera_metric=T_world_camera.astype(np.float32),
        source_intrinsics_fx_fy_cx_cy=source_intr.astype(np.float32),
        vggt_to_meters=np.asarray([float(scale)], dtype=np.float32),
        anchor_frame=np.asarray([anchor_frame], dtype=np.int32),
        points_vggt=points_vggt.astype(np.float16),
        masks=masks.astype(np.uint8),
        rgbs=rgbs.astype(np.uint8),
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "run_vggt_native_camera_v3",
        "dataset_manifest": str(args.dataset_manifest),
        "archive": str(archive),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": frame_indices,
        "anchor_frame": anchor_frame,
        "scale_status": "metric_depth_ratio" if depth_rows else "explicit_cli_scale",
        "vggt_to_meters": float(scale),
        "center_speed_m_s": summarize(center_speed),
        "source_intrinsics_fx_px": summarize(source_intr[:, 0].astype(float).tolist()),
        "source_intrinsics_fy_px": summarize(source_intr[:, 1].astype(float).tolist()),
        "metric_depth_rows": depth_rows,
        "view_reports": view_reports,
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_vggt_native_camera_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"metric_depth_rows", "view_reports"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--metric-depth-manifest", type=Path)
    parser.add_argument("--metric-remote-root", type=Path)
    parser.add_argument("--metric-local-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1080)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--model-file", default="model.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--vggt-to-meters", type=float, default=1.0)
    parser.add_argument("--min-depth-pixels", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
