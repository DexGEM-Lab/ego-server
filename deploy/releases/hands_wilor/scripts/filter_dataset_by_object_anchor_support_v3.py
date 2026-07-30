#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_intrinsics(dataset: Path, manifest: dict) -> tuple[float, float, float, float]:
    qc = dataset / "qc_bundlesdf_dataset_v3.json"
    if qc.exists():
        values = load_json(qc).get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return tuple(float(v) for v in values)
    values = manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return tuple(float(v) for v in values)
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3):
        raise RuntimeError(f"{dataset / 'cam_K.txt'} must be a 3x3 matrix")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def write_cam_k(path: Path, intrinsics: tuple[float, float, float, float]) -> None:
    fx, fy, cx, cy = intrinsics
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, K, fmt="%.10f")


def source_pose_index(entry: dict) -> int:
    return int(entry.get("source_index", entry["index"]))


def load_pose(bundlesdf_output: Path, index: int) -> np.ndarray:
    path = bundlesdf_output / "ob_in_cam" / f"{index:06d}.txt"
    if not path.exists():
        raise RuntimeError(f"missing BundleSDF pose {path}")
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise RuntimeError(f"BundleSDF pose must be finite 4x4: {path}")
    return pose


def unproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.c_[points.astype(np.float64), np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def sample_mask_points(mask: np.ndarray, depth_m: np.ndarray, intrinsics: tuple[float, float, float, float], pose: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.nonzero(mask & np.isfinite(depth_m) & (depth_m > 0.05))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(xs) > max_points:
        order = np.linspace(0, len(xs) - 1, max_points, dtype=np.int64)
        xs = xs[order]
        ys = ys[order]
    cam = unproject_pixels(xs, ys, depth_m, intrinsics)
    return transform_points(np.linalg.inv(pose), cam)


def anchor_points(path: Path) -> np.ndarray:
    payload = load_json(path)
    anchors = payload.get("anchor_points")
    if not isinstance(anchors, list) or not anchors:
        raise RuntimeError(f"{path} lacks anchor_points")
    points = np.asarray([row["object_xyz"] for row in anchors], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("anchor_points must be finite Nx3")
    return points


def support_tree_from_observations(
    args: argparse.Namespace,
    entries: list[dict],
    intrinsics: tuple[float, float, float, float],
    anchor_xyz: np.ndarray,
) -> tuple[cKDTree, dict]:
    support_parts = [anchor_xyz]
    per_frame = []
    anchor_tree = cKDTree(anchor_xyz)
    for entry in entries:
        index = int(entry["index"])
        pose_index = source_pose_index(entry)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise RuntimeError(f"failed to read mask/depth for frame {entry['frame_idx']}")
        depth_m = depth.astype(np.float64) / 1000.0
        pose = load_pose(args.bundlesdf_output, pose_index)
        obj_points = sample_mask_points(mask > 0, depth_m, intrinsics, pose, int(args.max_support_points_per_frame))
        if len(obj_points) == 0:
            continue
        nearest, _ = anchor_tree.query(obj_points, k=1)
        supported = obj_points[nearest <= float(args.anchor_bootstrap_radius_m)]
        if len(supported):
            support_parts.append(supported)
        per_frame.append({"frame_idx": int(entry["frame_idx"]), "index": index, "source_index": pose_index, "support_points": int(len(supported))})
    support = np.vstack(support_parts)
    if len(support) < int(args.min_support_points):
        raise RuntimeError(f"only {len(support)} support points")
    return cKDTree(support), {"support_points": int(len(support)), "support_frames": per_frame}


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.source_manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("source manifest must contain a nonempty frames list")
    intrinsics = load_intrinsics(args.source_dataset, manifest)
    anchors = anchor_points(args.anchor_prompts)
    support_tree, support_report = support_tree_from_observations(args, entries, intrinsics, anchors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "depth", "masks"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_cam_k(args.output_dir / "cam_K.txt", intrinsics)

    output_entries = []
    rows = []
    for entry in entries:
        src_index = int(entry["index"])
        pose_index = source_pose_index(entry)
        frame_idx = int(entry["frame_idx"])
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or mask is None:
            raise RuntimeError(f"failed to read RGB/depth/mask for frame {frame_idx}")
        depth_m = depth.astype(np.float64) / 1000.0
        valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.05)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            rows.append({"frame_idx": frame_idx, "reason": "no_valid_mask_depth"})
            continue
        pose = load_pose(args.bundlesdf_output, pose_index)
        obj = transform_points(np.linalg.inv(pose), unproject_pixels(xs, ys, depth_m, intrinsics))
        nearest, _ = support_tree.query(obj, k=1)
        keep_flat = nearest <= float(args.support_radius_m)
        keep = np.zeros(mask.shape, dtype=bool)
        keep[ys[keep_flat], xs[keep_flat]] = True
        if args.mask_open_px > 0:
            kernel = np.ones((2 * args.mask_open_px + 1, 2 * args.mask_open_px + 1), dtype=np.uint8)
            keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) > 0
        area = int(np.count_nonzero(keep))
        if area < int(args.min_mask_pixels):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "reason": "support_area_too_small",
                    "source_index": pose_index,
                    "original_area_px": int(np.count_nonzero(mask)),
                    "filtered_area_px": area,
                }
            )
            continue
        out_index = len(output_entries)
        stem = f"{out_index:06d}"
        rgb_path = args.output_dir / "rgb" / f"{stem}.png"
        depth_path = args.output_dir / "depth" / f"{stem}.png"
        mask_path = args.output_dir / "masks" / f"{stem}.png"
        if not cv2.imwrite(str(rgb_path), rgb):
            raise RuntimeError(f"failed to write {rgb_path}")
        if not cv2.imwrite(str(depth_path), depth):
            raise RuntimeError(f"failed to write {depth_path}")
        if not cv2.imwrite(str(mask_path), keep.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        valid_depth = depth_m[keep]
        output_entries.append(
            {
                "index": out_index,
                "source_index": pose_index,
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
                "source_index": pose_index,
                "original_area_px": int(np.count_nonzero(mask)),
                "filtered_area_px": area,
            }
        )
    if len(output_entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(output_entries)} filtered frames survived")

    (args.output_dir / "manifest.json").write_text(json.dumps({"frames": output_entries}, indent=2), encoding="utf-8")
    areas = np.asarray([entry["mask_area_px"] for entry in output_entries], dtype=np.float64)
    med_depth = np.asarray([entry["mask_depth_median_m"] for entry in output_entries], dtype=np.float64)
    qc = {
        "status": "ok",
        "method": "object_anchor_support_mask_filter_v3",
        "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest),
        "anchor_prompts": str(args.anchor_prompts),
        "bundlesdf_output": str(args.bundlesdf_output),
        "output_dir": str(args.output_dir),
        "frames": int(len(output_entries)),
        "first_frame": int(output_entries[0]["frame_idx"]),
        "last_frame": int(output_entries[-1]["frame_idx"]),
        "track_id": args.track_id,
        "intrinsics_fx_fy_cx_cy": [float(x) for x in intrinsics],
        "support_radius_m": float(args.support_radius_m),
        "anchor_bootstrap_radius_m": float(args.anchor_bootstrap_radius_m),
        "mask_area_median_px": float(np.median(areas)),
        "mask_area_min_px": int(np.min(areas)),
        "mask_area_max_px": int(np.max(areas)),
        "depth_median_m": float(np.median(med_depth)),
        "depth_p05_m": float(np.percentile(med_depth, 5)),
        "depth_p95_m": float(np.percentile(med_depth, 95)),
        "manifest": str(args.output_dir / "manifest.json"),
        **support_report,
        "rows": rows,
    }
    (args.output_dir / "qc_bundlesdf_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k not in {"rows", "support_frames"}}, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--anchor-prompts", type=Path, required=True)
    parser.add_argument("--bundlesdf-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--support-radius-m", type=float, default=0.035)
    parser.add_argument("--anchor-bootstrap-radius-m", type=float, default=0.08)
    parser.add_argument("--max-support-points-per-frame", type=int, default=30000)
    parser.add_argument("--min-support-points", type=int, default=50)
    parser.add_argument("--min-mask-pixels", type=int, default=5000)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--mask-open-px", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
