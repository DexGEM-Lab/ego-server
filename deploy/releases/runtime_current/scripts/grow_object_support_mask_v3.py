#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
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
    qc_path = dataset / "qc_bundlesdf_dataset_v3.json"
    if qc_path.exists():
        values = load_json(qc_path).get("intrinsics_fx_fy_cx_cy")
        if isinstance(values, list) and len(values) == 4:
            return tuple(float(v) for v in values)
    values = manifest.get("intrinsics_fx_fy_cx_cy")
    if isinstance(values, list) and len(values) == 4:
        return tuple(float(v) for v in values)
    k = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if k.shape != (3, 3):
        raise RuntimeError(f"{dataset / 'cam_K.txt'} must be a 3x3 matrix")
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def write_cam_k(path: Path, intrinsics: tuple[float, float, float, float]) -> None:
    fx, fy, cx, cy = intrinsics
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, k, fmt="%.10f")


def source_pose_index(entry: dict) -> int:
    return int(entry.get("source_index", entry["index"]))


def load_pose(bundlesdf_output: Path, pose_index: int) -> np.ndarray:
    path = bundlesdf_output / "ob_in_cam" / f"{pose_index:06d}.txt"
    if not path.exists():
        raise RuntimeError(f"missing BundleSDF pose {path}")
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise RuntimeError(f"BundleSDF pose must be finite 4x4: {path}")
    return pose


def unproject_xy_depth(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.column_stack(((xs.astype(np.float64) - cx) * zs / fx, (ys.astype(np.float64) - cy) * zs / fy, zs))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.column_stack((points.astype(np.float64), np.ones(len(points), dtype=np.float64)))
    return (transform @ homog.T).T[:, :3]


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform @ np.r_[point.astype(np.float64), 1.0])[:3]


def project(point_cam: np.ndarray, intrinsics: tuple[float, float, float, float]) -> tuple[float, float] | None:
    if not np.isfinite(point_cam).all() or point_cam[2] <= 0.05:
        return None
    fx, fy, cx, cy = intrinsics
    return float(fx * point_cam[0] / point_cam[2] + cx), float(fy * point_cam[1] / point_cam[2] + cy)


def local_depth(depth_m: np.ndarray, x: float, y: float, radius: int) -> float | None:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x0 = max(0, xi - radius)
    x1 = min(depth_m.shape[1], xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(depth_m.shape[0], yi + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[np.isfinite(patch) & (patch > 0.05)]
    if values.size == 0:
        return None
    return float(np.median(values))


def robust_anchor_rows(payload: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = payload.get("anchor_points")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{args.anchor_prompts} lacks anchor_points")
    points = np.asarray([row["object_xyz"] for row in rows], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("anchor_points must contain finite object_xyz vectors")
    tree = cKDTree(points)
    parents = list(range(len(points)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i: int, j: int) -> None:
        a = find(i)
        b = find(j)
        if a != b:
            parents[b] = a

    for i, j in tree.query_pairs(float(args.anchor_cluster_radius_m)):
        union(int(i), int(j))
    components: dict[int, list[int]] = {}
    for i in range(len(points)):
        components.setdefault(find(i), []).append(i)
    inlier_idx = max(components.values(), key=len)
    if len(inlier_idx) < int(args.min_inlier_anchors):
        raise RuntimeError(f"largest anchor cluster has only {len(inlier_idx)} anchors")
    cluster = points[inlier_idx]
    extent = cluster.max(axis=0) - cluster.min(axis=0)
    if np.max(extent) > float(args.max_anchor_cluster_extent_m):
        raise RuntimeError(f"largest anchor cluster extent is too large: {extent.tolist()}")
    rejected = sorted(set(range(len(rows))).difference(inlier_idx))
    report = {
        "anchor_count_raw": int(len(rows)),
        "anchor_count_inlier": int(len(inlier_idx)),
        "anchor_cluster_radius_m": float(args.anchor_cluster_radius_m),
        "anchor_cluster_extent_m": extent.astype(float).tolist(),
        "anchor_cluster_center_median": np.median(cluster, axis=0).astype(float).tolist(),
        "rejected_anchor_indices": rejected,
        "rejected_anchors": [
            {
                "anchor_index_global": int(i),
                "track_id": str(rows[i].get("track_id", "")),
                "anchor_frame_idx": int(rows[i].get("anchor_frame_idx", -1)),
                "object_xyz": [float(v) for v in rows[i]["object_xyz"]],
                "evidence": str(rows[i].get("evidence", "")),
            }
            for i in rejected
        ],
    }
    return [rows[i] for i in sorted(inlier_idx)], report


def support_tree_from_parent_masks(
    args: argparse.Namespace,
    entries: list[dict],
    intrinsics: tuple[float, float, float, float],
    anchor_xyz: np.ndarray,
) -> tuple[cKDTree, dict]:
    anchor_tree = cKDTree(anchor_xyz)
    support_parts = [anchor_xyz]
    per_frame = []
    for entry in entries:
        pose_index = source_pose_index(entry)
        pose = load_pose(args.bundlesdf_output, pose_index)
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise RuntimeError(f"failed to read parent mask/depth for frame {entry['frame_idx']}")
        valid = (mask > 0) & np.isfinite(depth.astype(np.float64)) & (depth > 50)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            per_frame.append({"frame_idx": int(entry["frame_idx"]), "support_points": 0})
            continue
        if len(xs) > int(args.max_bootstrap_pixels_per_frame):
            stride = int(np.ceil(len(xs) / int(args.max_bootstrap_pixels_per_frame)))
            xs = xs[::stride]
            ys = ys[::stride]
        zs = depth[ys, xs].astype(np.float64) / 1000.0
        object_points = transform_points(np.linalg.inv(pose), unproject_xy_depth(xs, ys, zs, intrinsics))
        nearest, _ = anchor_tree.query(object_points, k=1)
        supported = object_points[nearest <= float(args.anchor_bootstrap_radius_m)]
        if len(supported):
            support_parts.append(supported)
        per_frame.append(
            {
                "frame_idx": int(entry["frame_idx"]),
                "source_index": int(pose_index),
                "sampled_parent_points": int(len(xs)),
                "support_points": int(len(supported)),
            }
        )
    support = np.vstack(support_parts)
    if len(support) < int(args.min_support_points):
        raise RuntimeError(f"only {len(support)} object support points")
    return cKDTree(support), {"support_points": int(len(support)), "support_frames": per_frame}


def frame_seed_pixels(
    anchor_xyz: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_m: np.ndarray,
    parent: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int]], list[float], list[dict]]:
    seeds: list[tuple[int, int]] = []
    seed_depths: list[float] = []
    report = []
    h, w = depth_m.shape
    for i, anchor in enumerate(anchor_xyz):
        cam = transform_point(pose, anchor)
        xy = project(cam, intrinsics)
        if xy is None:
            report.append({"anchor": int(i), "reason": "behind_camera"})
            continue
        x, y = xy
        if x < 0 or x >= w or y < 0 or y >= h:
            report.append({"anchor": int(i), "reason": "outside_image", "x": float(x), "y": float(y)})
            continue
        observed = local_depth(depth_m, x, y, int(args.reprojection_depth_radius_px))
        if observed is None:
            report.append({"anchor": int(i), "reason": "no_depth", "x": float(x), "y": float(y)})
            continue
        depth_error = abs(observed - float(cam[2]))
        if depth_error > float(args.reprojection_depth_tolerance_m):
            report.append(
                {"anchor": int(i), "reason": "depth_residual", "x": float(x), "y": float(y), "depth_error_m": float(depth_error)}
            )
            continue
        xi = int(round(x))
        yi = int(round(y))
        if not parent[yi, xi]:
            report.append({"anchor": int(i), "reason": "outside_parent_mask", "x": float(x), "y": float(y)})
            continue
        seeds.append((yi, xi))
        seed_depths.append(observed)
        report.append({"anchor": int(i), "reason": "ok", "x": float(x), "y": float(y), "depth_error_m": float(depth_error)})
    return seeds, seed_depths, report


def support_allowed_mask(
    parent: np.ndarray,
    depth_m: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    support_tree: cKDTree,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    valid = parent & np.isfinite(depth_m) & (depth_m > 0.05)
    ys, xs = np.nonzero(valid)
    out = np.zeros(parent.shape, dtype=bool)
    if len(xs) == 0:
        return out, {"parent_valid_pixels": 0, "support_allowed_pixels": 0}
    points_cam = unproject_xy_depth(xs, ys, depth_m[ys, xs], intrinsics)
    object_points = transform_points(np.linalg.inv(pose), points_cam)
    nearest, _ = support_tree.query(object_points, k=1)
    keep = nearest <= float(args.support_radius_m)
    out[ys[keep], xs[keep]] = True
    return out, {
        "parent_valid_pixels": int(len(xs)),
        "support_allowed_pixels": int(np.count_nonzero(out)),
        "support_nearest_median_m": float(np.median(nearest)) if len(nearest) else None,
        "support_nearest_p05_m": float(np.percentile(nearest, 5)) if len(nearest) else None,
        "support_nearest_p95_m": float(np.percentile(nearest, 95)) if len(nearest) else None,
    }


def grow_connected(
    allowed: np.ndarray,
    depth_m: np.ndarray,
    seeds: list[tuple[int, int]],
    seed_depths: list[float],
    args: argparse.Namespace,
) -> np.ndarray:
    out = np.zeros(allowed.shape, dtype=bool)
    if len(seed_depths) < int(args.min_seed_depths):
        return out
    lo = float(np.percentile(seed_depths, 5)) - float(args.seed_depth_band_m)
    hi = float(np.percentile(seed_depths, 95)) + float(args.seed_depth_band_m)
    queue: deque[tuple[int, int]] = deque()
    for y, x in seeds:
        if allowed[y, x] and lo <= float(depth_m[y, x]) <= hi:
            out[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        current = float(depth_m[y, x])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= allowed.shape[0] or xx < 0 or xx >= allowed.shape[1]:
                continue
            if out[yy, xx] or not allowed[yy, xx]:
                continue
            z = float(depth_m[yy, xx])
            if z < lo or z > hi:
                continue
            if abs(z - current) > float(args.local_depth_step_m):
                continue
            out[yy, xx] = True
            queue.append((yy, xx))
    if args.close_px > 0:
        kernel = np.ones((2 * args.close_px + 1, 2 * args.close_px + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
        out &= allowed & (depth_m >= lo) & (depth_m <= hi)
    if args.open_px > 0:
        kernel = np.ones((2 * args.open_px + 1, 2 * args.open_px + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) > 0
    return out


def render_review(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[:, :, 1] = 220
    overlay[mask] = cv2.addWeighted(rgb[mask], 0.45, color[mask], 0.55, 0.0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"failed to write {path}")


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.source_manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("source manifest must contain nonempty frames")
    selected = [
        entry
        for entry in entries
        if int(entry["frame_idx"]) >= int(args.frame_start) and int(entry["frame_idx"]) <= int(args.frame_end)
    ]
    if not selected:
        raise RuntimeError("no source frames selected")
    intrinsics = load_intrinsics(args.source_dataset, manifest)
    prompt_payload = load_json(args.anchor_prompts)
    inlier_rows, anchor_report = robust_anchor_rows(prompt_payload, args)
    anchor_xyz = np.asarray([row["object_xyz"] for row in inlier_rows], dtype=np.float64)
    support_tree, support_report = support_tree_from_parent_masks(args, selected, intrinsics, anchor_xyz)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "depth", "masks", "review"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_cam_k(args.output_dir / "cam_K.txt", intrinsics)

    review_frames = set(int(x) for x in args.review_frames)
    output_entries = []
    area_values = []
    depth_values = []
    rows = []
    for entry in selected:
        frame_idx = int(entry["frame_idx"])
        pose_index = source_pose_index(entry)
        rgb = cv2.imread(str(Path(entry["rgb"])), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(Path(entry["depth"])), cv2.IMREAD_UNCHANGED)
        parent_raw = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if rgb is None or depth is None or parent_raw is None:
            raise RuntimeError(f"failed to read RGB/depth/mask for frame {frame_idx}")
        depth_m = depth.astype(np.float64) / 1000.0
        parent = parent_raw > 0
        pose = load_pose(args.bundlesdf_output, pose_index)
        seeds, seed_depths, seed_report = frame_seed_pixels(anchor_xyz, pose, intrinsics, depth_m, parent, args)
        allowed, support_stats = support_allowed_mask(parent, depth_m, pose, intrinsics, support_tree, args)
        grown = grow_connected(allowed, depth_m, seeds, seed_depths, args)
        area = int(np.count_nonzero(grown))
        row = {
            "frame_idx": frame_idx,
            "source_index": pose_index,
            "parent_area_px": int(np.count_nonzero(parent)),
            "valid_seeds": int(len(seeds)),
            "seed_depth_median_m": float(np.median(seed_depths)) if seed_depths else None,
            "grown_area_px": area,
            **support_stats,
            "seed_report": seed_report,
        }
        if area < int(args.min_mask_pixels):
            row["reason"] = "grown_area_too_small"
            rows.append(row)
            continue
        valid_depth = depth_m[grown]
        if valid_depth.size < int(args.min_depth_pixels):
            row["reason"] = "insufficient_depth_pixels"
            rows.append(row)
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
        if not cv2.imwrite(str(mask_path), grown.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {mask_path}")
        if frame_idx in review_frames:
            render_review(rgb, grown, args.output_dir / "review" / f"frame_{frame_idx:06d}.png")
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
        area_values.append(area)
        depth_values.append(float(np.median(valid_depth)))
        row.update(
            {
                "reason": "ok",
                "output_index": out_index,
                "mask_depth_median_m": float(np.median(valid_depth)),
                "mask_depth_p05_m": float(np.percentile(valid_depth, 5)),
                "mask_depth_p95_m": float(np.percentile(valid_depth, 95)),
            }
        )
        rows.append(row)
    if len(output_entries) < int(args.min_frames):
        raise RuntimeError(f"only {len(output_entries)} object-support frames survived")

    manifest_out = {"frames": output_entries}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    areas = np.asarray(area_values, dtype=np.float64)
    depths = np.asarray(depth_values, dtype=np.float64)
    qc = {
        "status": "ok",
        "method": "robust_anchor_object_support_depth_growth_v3",
        "source_dataset": str(args.source_dataset),
        "source_manifest": str(args.source_manifest),
        "anchor_prompts": str(args.anchor_prompts),
        "bundlesdf_output": str(args.bundlesdf_output),
        "output_dir": str(args.output_dir),
        "frames": int(len(output_entries)),
        "first_frame": int(output_entries[0]["frame_idx"]),
        "last_frame": int(output_entries[-1]["frame_idx"]),
        "track_id": args.track_id,
        "label": args.label,
        "intrinsics_fx_fy_cx_cy": [float(x) for x in intrinsics],
        "support_radius_m": float(args.support_radius_m),
        "anchor_bootstrap_radius_m": float(args.anchor_bootstrap_radius_m),
        "seed_depth_band_m": float(args.seed_depth_band_m),
        "local_depth_step_m": float(args.local_depth_step_m),
        "mask_area_median_px": float(np.median(areas)),
        "mask_area_min_px": int(np.min(areas)),
        "mask_area_max_px": int(np.max(areas)),
        "depth_median_m": float(np.median(depths)),
        "depth_p05_m": float(np.percentile(depths, 5)),
        "depth_p95_m": float(np.percentile(depths, 95)),
        "manifest": str(args.output_dir / "manifest.json"),
        **anchor_report,
        **support_report,
        "rows": rows,
    }
    (args.output_dir / "qc_bundlesdf_dataset_v3.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in qc.items() if k not in {"rows", "support_frames", "rejected_anchors"}}, indent=2))
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
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-cluster-radius-m", type=float, default=0.08)
    parser.add_argument("--max-anchor-cluster-extent-m", type=float, default=0.45)
    parser.add_argument("--min-inlier-anchors", type=int, default=8)
    parser.add_argument("--anchor-bootstrap-radius-m", type=float, default=0.025)
    parser.add_argument("--support-radius-m", type=float, default=0.025)
    parser.add_argument("--max-bootstrap-pixels-per-frame", type=int, default=80000)
    parser.add_argument("--min-support-points", type=int, default=20)
    parser.add_argument("--reprojection-depth-radius-px", type=int, default=2)
    parser.add_argument("--reprojection-depth-tolerance-m", type=float, default=0.04)
    parser.add_argument("--seed-depth-band-m", type=float, default=0.12)
    parser.add_argument("--local-depth-step-m", type=float, default=0.018)
    parser.add_argument("--min-seed-depths", type=int, default=4)
    parser.add_argument("--min-mask-pixels", type=int, default=1500)
    parser.add_argument("--min-depth-pixels", type=int, default=1500)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=2)
    parser.add_argument("--open-px", type=int, default=1)
    parser.add_argument("--review-frames", type=int, nargs="*", default=[858, 866, 878, 880])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
