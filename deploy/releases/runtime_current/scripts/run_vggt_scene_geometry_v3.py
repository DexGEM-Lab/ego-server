#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vggt_object_geometry_v3 import (
    apply_sim3,
    camera_centers_from_annotations,
    camera_centers_from_vggt,
    frame_map,
    import_vggt,
    load_json,
    run_vggt,
    umeyama_similarity,
)


def read_manifest(path: Path, frame_start: int, frame_end: int) -> list[dict]:
    payload = load_json(path)
    rows = []
    for row in payload["frames"]:
        frame_idx = int(row["frame_idx"])
        if frame_start <= frame_idx <= frame_end:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no dataset rows in requested interval {frame_start}:{frame_end}")
    rows.sort(key=lambda item: int(item["frame_idx"]))
    actual = np.asarray([int(row["frame_idx"]) for row in rows], dtype=int)
    expected = np.arange(actual[0], actual[-1] + 1, dtype=int)
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"dataset rows are not contiguous: {actual.tolist()}")
    return rows


def localize_path(path: str, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src_root, dst_root in ((local_root, remote_root), (remote_root, local_root)):
            try:
                rel = direct.relative_to(src_root)
            except ValueError:
                rel = None
            if rel is not None:
                candidate = dst_root / rel
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(path)


def preprocess_full_image_and_mask(
    image_path: Path,
    mask_path: Path,
    target_size: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        raise RuntimeError(f"could not read image: {image_path}")
    if mask is None:
        raise RuntimeError(f"could not read mask: {mask_path}")
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise RuntimeError(f"image and mask size mismatch: {image_bgr.shape[:2]} vs {mask.shape[:2]}")
    height, width = image_bgr.shape[:2]
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    if new_width <= 0 or new_height <= 0:
        raise RuntimeError("invalid preprocessed VGGT dimensions")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_small = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    mask_small = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left
    if min(pad_top, pad_bottom, pad_left, pad_right) < 0:
        raise RuntimeError("preprocessed image is larger than target size")
    image_pad = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    mask_pad = np.zeros((target_size, target_size), dtype=np.uint8)
    image_pad[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = image_small
    mask_pad[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = np.where(
        mask_small > 0, 255, 0
    ).astype(np.uint8)
    tensor = torch.from_numpy(image_pad.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor, mask_pad, image_pad


def load_scene_views(
    rows: list[dict],
    target_size: int,
    remote_root: Path | None,
    local_root: Path | None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[dict]]:
    images = []
    masks = []
    rgbs = []
    reports = []
    for row in rows:
        image_path = localize_path(str(row["rgb"]), remote_root, local_root)
        mask_path = localize_path(str(row["mask"]), remote_root, local_root)
        image, mask, rgb = preprocess_full_image_and_mask(image_path, mask_path, target_size)
        if int((mask > 0).sum()) < 100:
            raise RuntimeError(f"frame {row['frame_idx']} has underconstrained mask after VGGT resize")
        images.append(image)
        masks.append(mask)
        rgbs.append(rgb)
        reports.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_pixels_vggt": int((mask > 0).sum()),
            }
        )
    return torch.stack(images, dim=0), np.stack(masks, axis=0), np.stack(rgbs, axis=0), reports


def robust_extent(points: np.ndarray, q: float) -> np.ndarray:
    if len(points) == 0:
        raise RuntimeError("cannot compute extent for empty point set")
    lo = np.quantile(points, q, axis=0)
    hi = np.quantile(points, 1.0 - q, axis=0)
    return hi - lo


def transform_world_to_camera(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    return (np.asarray(points, dtype=float) @ rotation.T) + translation[None, :]


def summarize_values(values: list[float]) -> dict:
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


def select_object_points(
    frame_indices: list[int],
    points_world: np.ndarray,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    masks: np.ndarray,
    rgbs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(int(args.seed))
    all_points = []
    all_colors = []
    vertex_offsets = [0]
    frame_reports = []
    camera_extents = []
    world_extents = []
    center_world = []
    center_camera = []
    for i, frame_idx in enumerate(frame_indices):
        mask = masks[i] > 0
        frame_points = points_world[i]
        frame_depth = depth[i]
        frame_conf = depth_conf[i]
        valid = (
            mask
            & np.isfinite(frame_points).all(axis=2)
            & np.isfinite(frame_depth)
            & (frame_depth > 1e-5)
            & np.isfinite(frame_conf)
        )
        conf_values = frame_conf[mask & np.isfinite(frame_conf)]
        if conf_values.size == 0:
            raise RuntimeError(f"frame {frame_idx} has no finite VGGT confidence on object mask")
        threshold = max(float(args.min_depth_conf), float(np.quantile(conf_values, float(args.conf_quantile))))
        valid &= frame_conf >= threshold
        valid_count = int(valid.sum())
        if valid_count < int(args.min_points_per_frame):
            raise RuntimeError(f"frame {frame_idx} has only {valid_count} valid VGGT object points")
        points = frame_points[valid].astype(np.float32)
        colors = rgbs[i][valid].astype(np.uint8)
        if len(points) > int(args.max_points_per_frame):
            chosen = rng.choice(len(points), size=int(args.max_points_per_frame), replace=False)
            points = points[chosen]
            colors = colors[chosen]
        points_cam = transform_world_to_camera(points, extrinsic[i])
        cam_extent = robust_extent(points_cam, float(args.robust_quantile))
        world_extent = robust_extent(points, float(args.robust_quantile))
        camera_extents.append(cam_extent)
        world_extents.append(world_extent)
        center_world.append(np.median(points, axis=0))
        center_camera.append(np.median(points_cam, axis=0))
        all_points.append(points)
        all_colors.append(colors)
        vertex_offsets.append(vertex_offsets[-1] + len(points))
        frame_reports.append(
            {
                "frame_idx": int(frame_idx),
                "mask_pixels": int(mask.sum()),
                "confidence_threshold": float(threshold),
                "valid_points_before_sampling": valid_count,
                "sampled_points": int(len(points)),
                "center_world_vggt": np.asarray(center_world[-1], dtype=float).tolist(),
                "center_camera_vggt": np.asarray(center_camera[-1], dtype=float).tolist(),
                "robust_extent_world_vggt": np.asarray(world_extent, dtype=float).tolist(),
                "robust_extent_camera_vggt": np.asarray(cam_extent, dtype=float).tolist(),
            }
        )
    return (
        np.vstack(all_points).astype(np.float32),
        np.vstack(all_colors).astype(np.uint8),
        np.asarray(vertex_offsets, dtype=np.int64),
        np.asarray(camera_extents, dtype=float),
        frame_reports,
    )


def export_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.dataset_manifest, int(args.frame_start), int(args.frame_end))
    frame_indices = [int(row["frame_idx"]) for row in rows]
    frame_by_idx = frame_map(args.annotations)
    missing = [idx for idx in frame_indices if idx not in frame_by_idx]
    if missing:
        raise RuntimeError(f"annotation frames missing for VGGT rows: {missing}")
    images, masks, rgbs, view_reports = load_scene_views(
        rows, int(args.target_size), args.remote_output_root, args.local_output_root
    )
    extrinsic, intrinsic, depth, depth_conf, points_vggt = run_vggt(args, images)
    vggt_centers = camera_centers_from_vggt(extrinsic)
    target_centers = camera_centers_from_annotations(frame_by_idx, frame_indices)
    scale, rotation, translation = umeyama_similarity(vggt_centers, target_centers)
    aligned_centers = apply_sim3(vggt_centers, scale, rotation, translation)
    center_error = np.linalg.norm(aligned_centers - target_centers, axis=1)
    object_points_vggt, object_colors, vertex_offsets, camera_extents, point_reports = select_object_points(
        frame_indices, points_vggt, depth, depth_conf, extrinsic, masks, rgbs, args
    )
    object_points_aligned = apply_sim3(object_points_vggt, scale, rotation, translation).astype(np.float32)
    point_path = args.output_dir / "vggt_scene_object_points_aligned.ply"
    export_point_ply(point_path, object_points_aligned, object_colors)
    archive_path = args.output_dir / "vggt_scene_object_points_v3.npz"
    np.savez_compressed(
        archive_path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=vertex_offsets,
        object_points_vggt=object_points_vggt.astype(np.float32),
        object_points_aligned=object_points_aligned.astype(np.float32),
        object_colors=object_colors.astype(np.uint8),
        masks=masks.astype(np.uint8),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
        camera_centers_vggt=vggt_centers.astype(np.float32),
        camera_centers_droid_metric=target_centers.astype(np.float32),
        camera_centers_aligned=aligned_centers.astype(np.float32),
        sim3_scale=np.asarray([scale], dtype=np.float32),
        sim3_rotation=rotation.astype(np.float32),
        sim3_translation=translation.astype(np.float32),
    )
    median_extent = np.median(camera_extents, axis=0)
    ratios = camera_extents / np.maximum(median_extent[None, :], 1e-9)
    pair_center_speed_vggt = []
    centers = np.asarray([row["center_world_vggt"] for row in point_reports], dtype=float)
    if len(centers) > 1:
        pair_center_speed_vggt = (np.linalg.norm(np.diff(centers, axis=0), axis=1) * float(args.fps)).tolist()
    report = {
        "status": "ok",
        "diagnostic_only": True,
        "method": "vggt_full_scene_samwise_object_geometry",
        "dataset_manifest": str(args.dataset_manifest),
        "annotations": str(args.annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": frame_indices,
        "model_id": str(args.model_id),
        "target_size": int(args.target_size),
        "sim3_vggt_to_droid_scale": float(scale),
        "camera_alignment_median_m": float(np.median(center_error)),
        "camera_alignment_p95_m": float(np.percentile(center_error, 95)),
        "camera_alignment_max_m": float(np.max(center_error)),
        "object_points": int(len(object_points_vggt)),
        "robust_camera_extent_median_vggt": median_extent.astype(float).tolist(),
        "robust_camera_extent_ratio_to_median_vggt": {
            "x": summarize_values(ratios[:, 0].tolist()),
            "y": summarize_values(ratios[:, 1].tolist()),
            "z": summarize_values(ratios[:, 2].tolist()),
            "max_abs_log": summarize_values(np.max(np.abs(np.log(np.maximum(ratios, 1e-9))), axis=1).tolist()),
        },
        "pair_center_speed_vggt_units_per_s": summarize_values(pair_center_speed_vggt),
        "predicted_intrinsics": {
            "fx_median": float(np.median(intrinsic[:, 0, 0])),
            "fy_median": float(np.median(intrinsic[:, 1, 1])),
            "cx_median": float(np.median(intrinsic[:, 0, 2])),
            "cy_median": float(np.median(intrinsic[:, 1, 2])),
            "fx_range": [float(np.min(intrinsic[:, 0, 0])), float(np.max(intrinsic[:, 0, 0]))],
            "fy_range": [float(np.min(intrinsic[:, 1, 1])), float(np.max(intrinsic[:, 1, 1]))],
        },
        "view_reports": view_reports,
        "point_reports": point_reports,
        "outputs": {
            "point_cloud_aligned": str(point_path),
            "archive": str(archive_path),
        },
        "parameters": {
            "conf_quantile": float(args.conf_quantile),
            "min_depth_conf": float(args.min_depth_conf),
            "max_points_per_frame": int(args.max_points_per_frame),
            "robust_quantile": float(args.robust_quantile),
        },
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_vggt_scene_geometry_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"view_reports", "point_reports"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--model-file", default="model.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--min-depth-conf", type=float, default=0.0)
    parser.add_argument("--conf-quantile", type=float, default=0.30)
    parser.add_argument("--min-points-per-frame", type=int, default=900)
    parser.add_argument("--max-points-per-frame", type=int, default=9000)
    parser.add_argument("--robust-quantile", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
