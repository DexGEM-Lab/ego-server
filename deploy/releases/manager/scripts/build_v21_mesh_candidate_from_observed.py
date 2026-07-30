#!/usr/bin/env python3
"""V21 object geometry candidate from selected multi-source depth and mask backprojection.

Generates a mesh candidate for a rigid object by:
1. Resolving the selected depth/camera bundle and depth candidate registry
2. Finding the visible mask frame with the largest valid primary-depth support
3. Scale-aligning available depth sources on that anchor mask
4. Backprojecting the fused anchor-frame depth evidence to 3D camera-space points
5. Creating a heightfield mesh candidate and canonicalizing it by PCA

The mesh is a candidate prior, not accepted geometry.
It must be SE(3) pose-fitted per-frame against observed surfaces.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v21_mask_sources import resolve_current_object_mask_dir

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh required", file=sys.stderr)
    sys.exit(1)


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def resolve_existing_path(run_root: Path, raw: Any) -> Path:
    if raw is None:
        raise ContractError("missing_path_value")
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([run_root / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ContractError(f"path_does_not_exist: {raw}")


def load_depth_archive(npz_path: Path) -> dict[str, Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "depth" not in data.files:
        raise ContractError(f"depth_npz_missing_depth: {npz_path}")
    depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim != 3:
        raise ContractError(f"depth_npz_depth_must_be_NHW: {npz_path} shape={depth.shape}")
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64) if "frame_idx" in data.files else np.arange(depth.shape[0], dtype=np.int64)
    if len(frame_idx) != depth.shape[0]:
        raise ContractError(f"depth_npz_frame_count_mismatch: {npz_path}")
    intr = None
    if "intrinsics_fx_fy_cx_cy" in data.files:
        intr = np.asarray(data["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    elif "intrinsics" in data.files:
        raw_intr = np.asarray(data["intrinsics"], dtype=np.float64)
        if raw_intr.ndim == 3 and raw_intr.shape[1:] == (3, 3):
            intr = np.stack([raw_intr[:, 0, 0], raw_intr[:, 1, 1], raw_intr[:, 0, 2], raw_intr[:, 1, 2]], axis=1)
    if intr is not None and len(intr) != depth.shape[0]:
        raise ContractError(f"depth_intrinsics_frame_count_mismatch: {npz_path}")
    return {
        "path": str(npz_path),
        "depth": depth,
        "frame_idx": frame_idx,
        "frame_to_i": {int(frame): int(i) for i, frame in enumerate(frame_idx)},
        "intrinsics_fx_fy_cx_cy": intr,
    }


def load_registry_sources(run_root: Path, registry_path: Path | None) -> list[dict[str, Any]]:
    if registry_path is None or not registry_path.exists():
        return []
    registry = load_json(registry_path)
    rows = registry.get("candidates")
    if not isinstance(rows, list):
        return []
    sources: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("depth_npz") or row.get("depth_candidate_npz") or row.get("npz_path")
        if raw_path is None:
            continue
        try:
            path = resolve_existing_path(run_root, raw_path)
        except ContractError:
            continue
        sources.append(
            {
                "source_id": str(row.get("candidate_id") or path.parent.name),
                "path": path,
                "method_family": str(row.get("method_family") or "unknown_depth"),
                "role": str(row.get("candidate_role_hint") or "registry_candidate"),
                "prior_weight": float(row.get("prior_weight", 1.0) or 1.0),
                "registry_candidate": row,
            }
        )
    return sources


def selected_primary_source(run_root: Path, selection_report_path: Path | None) -> dict[str, Any] | None:
    if selection_report_path is None or not selection_report_path.exists():
        return None
    report = load_json(selection_report_path)
    primary = report.get("selected_primary_depth_camera")
    if not isinstance(primary, dict):
        return None
    archive = primary.get("depth_archive")
    if not archive:
        return None
    return {
        "source_id": str(primary.get("candidate_id") or "selected_primary_depth"),
        "path": resolve_existing_path(run_root, archive),
        "method_family": str(primary.get("kind") or "selected_depth_camera"),
        "role": "selected_primary_depth_camera",
        "prior_weight": 1.0,
        "selection_primary": primary,
    }


def resolve_depth_sources(run_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    explicit = Path(args.depth_npz) if args.depth_npz else None
    if explicit is not None:
        return [{"source_id": "explicit_depth_npz", "path": resolve_existing_path(run_root, explicit), "method_family": "explicit", "role": "explicit_primary", "prior_weight": 1.0}]
    selection_path = Path(args.depth_selection_report) if args.depth_selection_report else run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json"
    registry_path = Path(args.depth_registry) if args.depth_registry else run_root / "measurements" / "camera_depth" / "v20_depth_registry" / "depth_candidate_registry.json"
    primary = selected_primary_source(run_root, selection_path)
    registry_sources = load_registry_sources(run_root, registry_path)
    if primary is None:
        raise ContractError(f"mesh_depth_selection_missing_primary: {selection_path}")
    ordered: list[dict[str, Any]] = [primary]
    seen = {str(primary["path"])}
    for source in registry_sources:
        key = str(source["path"])
        if key in seen:
            continue
        ordered.append(source)
        seen.add(key)
    return ordered


def find_best_mask_frame(mask_dir: Path, total_frames: int) -> tuple[int | None, int, int]:
    mask_files = sorted(glob.glob(str(mask_dir / "*.png")))
    best_frame = None
    best_area = 0
    for mf in mask_files:
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        area = int((mask > 127).sum())
        if area > best_area:
            best_area = area
            best_frame = int(Path(mf).stem)
    return best_frame, best_area, len(mask_files)


def scaled_intrinsics(intrinsics: np.ndarray, from_hw: tuple[int, int], to_hw: tuple[int, int]) -> np.ndarray:
    if from_hw == to_hw:
        return np.asarray(intrinsics, dtype=np.float64).copy()
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    sx = float(to_w) / max(1.0, float(from_w))
    sy = float(to_h) / max(1.0, float(from_h))
    intr = np.asarray(intrinsics, dtype=np.float64).copy()
    intr[0] *= sx
    intr[2] *= sx
    intr[1] *= sy
    intr[3] *= sy
    return intr


def resize_depth_to_mask(depth_frame: np.ndarray, intrinsics: np.ndarray, mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    depth_hw = tuple(int(v) for v in depth_frame.shape[:2])
    if depth_hw == mask_shape:
        return depth_frame, np.asarray(intrinsics, dtype=np.float64)
    resized = cv2.resize(depth_frame, (int(mask_shape[1]), int(mask_shape[0])), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32), scaled_intrinsics(intrinsics, depth_hw, mask_shape)


def foreground_depth_support(
    mask: np.ndarray,
    depth_frame: np.ndarray,
    core_quantile: float = 0.65,
    window_m: float = 0.12,
    window_rel: float = 0.06,
    min_points: int = 80,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = (mask > 127) & np.isfinite(depth_frame) & (depth_frame > 0.1)
    valid_count = int(valid.sum())
    stats: dict[str, Any] = {
        "valid_mask_depth_pixels": valid_count,
        "foreground_depth_center_m": None,
        "foreground_depth_window_m": None,
        "foreground_depth_pixels": 0,
        "foreground_filter_status": "no_valid_mask_depth",
    }
    if valid_count == 0:
        return valid, stats

    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 3)
    valid_dist = dist[valid]
    core = valid.copy()
    if len(valid_dist) >= min_points:
        threshold = float(np.quantile(valid_dist, np.clip(core_quantile, 0.0, 1.0)))
        core = valid & (dist >= threshold)
    core_depth = depth_frame[core]
    core_depth = core_depth[np.isfinite(core_depth) & (core_depth > 0.1)]
    if len(core_depth) < min_points:
        core_depth = depth_frame[valid]
        core_depth = core_depth[np.isfinite(core_depth) & (core_depth > 0.1)]
    if len(core_depth) == 0:
        return valid, stats

    center = float(np.median(core_depth))
    window = float(max(window_m, abs(center) * window_rel))
    foreground = valid & (np.abs(depth_frame.astype(np.float64) - center) <= window)
    if int(foreground.sum()) < min_points:
        foreground = valid & (np.abs(depth_frame.astype(np.float64) - center) <= window * 2.0)
        stats["foreground_filter_status"] = "foreground_window_widened"
    else:
        stats["foreground_filter_status"] = "foreground_depth_mode_window"

    if int(foreground.sum()) >= min_points:
        component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
        if component_count > 1:
            largest_label = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            foreground = labels == largest_label
            stats["foreground_filter_status"] += "+largest_connected_depth_component"

    stats.update(
        {
            "foreground_depth_center_m": center,
            "foreground_depth_window_m": window,
            "foreground_depth_pixels": int(foreground.sum()),
            "foreground_keep_fraction": float(int(foreground.sum()) / max(1, valid_count)),
        }
    )
    return foreground, stats


def backproject_mask_depth(
    mask: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: np.ndarray,
    outlier_removal: bool = True,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Backproject mask pixels with depth to 3D camera-space points."""
    fx, fy, cx, cy = intrinsics
    foreground_mask, stats = foreground_depth_support(mask, depth_frame)
    ys, xs = np.where(foreground_mask)
    if len(ys) == 0:
        empty = np.zeros((0, 3))
        return (empty, stats) if return_stats else empty
    zs = np.asarray(depth_frame[ys, xs], dtype=np.float64)
    valid = np.isfinite(zs) & (zs > 0.1)
    xs_v, ys_v, zs_v = xs[valid], ys[valid], zs[valid]

    if outlier_removal and len(zs_v) > 50:
        z_median = np.median(zs_v)
        z_iqr = np.percentile(zs_v, 75) - np.percentile(zs_v, 25)
        z_lo = z_median - 3 * max(z_iqr, 0.01)
        z_hi = z_median + 3 * max(z_iqr, 0.01)
        inlier = (zs_v >= z_lo) & (zs_v <= z_hi)
        xs_v, ys_v, zs_v = xs_v[inlier], ys_v[inlier], zs_v[inlier]

    xc = (xs_v.astype(np.float64) - cx) * zs_v / fx
    yc = (ys_v.astype(np.float64) - cy) * zs_v / fy
    points = np.column_stack([xc, yc, zs_v])
    stats["backprojected_point_count"] = int(len(points))
    stats["backprojected_extent_m"] = point_extent(points).astype(float).tolist() if len(points) else [0.0, 0.0, 0.0]
    return (points, stats) if return_stats else points


def filter_points_to_reference_support(points: np.ndarray, reference: np.ndarray, margin_m: float = 0.03) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) == 0 or len(reference) == 0:
        return points, {"support_filter_status": "empty_points_or_reference", "support_filtered_point_count": int(len(points))}
    lo = np.percentile(reference, 2.0, axis=0) - float(margin_m)
    hi = np.percentile(reference, 98.0, axis=0) + float(margin_m)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    filtered = points[keep]
    if len(filtered) < min(80, len(points)):
        return points, {
            "support_filter_status": "support_filter_rejected_too_many_points_kept_original",
            "support_reference_bounds_min_m": lo.astype(float).tolist(),
            "support_reference_bounds_max_m": hi.astype(float).tolist(),
            "support_filtered_point_count": int(len(filtered)),
            "support_original_point_count": int(len(points)),
        }
    return filtered, {
        "support_filter_status": "filtered_to_primary_spatial_support",
        "support_reference_bounds_min_m": lo.astype(float).tolist(),
        "support_reference_bounds_max_m": hi.astype(float).tolist(),
        "support_filtered_point_count": int(len(filtered)),
        "support_original_point_count": int(len(points)),
        "support_keep_fraction": float(len(filtered) / max(1, len(points))),
    }


def point_extent(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64)
    return points.max(axis=0) - points.min(axis=0)


def select_anchor_mask_frame(mask_dir: Path, depth: np.ndarray, per_frame_intrinsics: np.ndarray, frame_to_i: dict[int, int], max_extent_m: float | None = None, min_points: int = 100, preferred_frames: set[int] | None = None) -> dict[str, Any]:
    mask_files = sorted(glob.glob(str(mask_dir / "*.png")))
    candidates = []
    rejected_by_extent = []
    for mf in mask_files:
        frame_idx = int(Path(mf).stem)
        depth_i = frame_to_i.get(frame_idx)
        if depth_i is None:
            continue
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        area = int((mask > 127).sum())
        if area <= 0:
            continue
        depth_frame, intr = resize_depth_to_mask(depth[depth_i], per_frame_intrinsics[depth_i], mask.shape[:2])
        points, foreground_stats = backproject_mask_depth(mask, depth_frame, intr, return_stats=True)
        if len(points) < int(min_points):
            continue
        extent = point_extent(points)
        max_axis = float(np.max(extent)) if len(extent) else 0.0
        row = {
            "frame_idx": frame_idx,
            "area_px": area,
            "valid_point_count": int(len(points)),
            "extent_m": extent.astype(float).tolist(),
            "max_extent_axis_m": max_axis,
            "mask_path": str(mf),
            "foreground_depth_filter": foreground_stats,
        }
        if max_extent_m is not None and max_axis > float(max_extent_m):
            rejected_by_extent.append(row)
            continue
        candidates.append((area, len(points), frame_idx, mask, points, row, depth_frame, intr))
    if not candidates:
        if max_extent_m is not None:
            return select_anchor_mask_frame(mask_dir, depth, per_frame_intrinsics, frame_to_i, None, min_points, preferred_frames)
        raise ContractError(f"no_usable_anchor_mask_frame: {mask_dir}")
    preferred_candidates = [item for item in candidates if preferred_frames and int(item[2]) in preferred_frames]
    ranking_pool = preferred_candidates if preferred_candidates else candidates
    max_area = max(item[0] for item in ranking_pool)
    support_floor = max(float(min_points), 0.5 * float(max_area))
    rankable = [item for item in ranking_pool if float(item[0]) >= support_floor]
    if not rankable:
        rankable = candidates
    rankable.sort(key=lambda item: (float(item[5]["max_extent_axis_m"]), -int(item[0]), -int(item[1])))
    area, _points_count, frame_idx, mask, points, row, depth_frame, intr = rankable[0]
    return {
        "frame_idx": int(frame_idx),
        "area_px": int(area),
        "mask": mask,
        "primary_points": points,
        "primary_depth_frame": depth_frame,
        "primary_intrinsics": intr,
        "selected": row,
        "candidate_count": len(candidates),
        "preferred_keyframe_candidate_count": len(preferred_candidates),
        "ranking_pool": "segmentation_stable_keyframes" if preferred_candidates else "all_masks",
        "rankable_candidate_count": len(rankable),
        "rankable_area_floor_px": float(support_floor),
        "rejected_by_extent_count": len(rejected_by_extent),
        "rejected_by_extent_examples": rejected_by_extent[:10],
        "selection_policy": "preferred_keyframe_compact_3d_extent_with_large_mask_support_after_depth_extent_filter" if max_extent_m is not None and preferred_candidates else ("compact_3d_extent_with_large_mask_support_after_depth_extent_filter" if max_extent_m is not None else "compact_3d_extent_with_large_mask_support"),
        "max_anchor_extent_m": None if max_extent_m is None else float(max_extent_m),
    }


def source_points_for_anchor(
    source: dict[str, Any],
    frame_idx: int,
    mask: np.ndarray,
    primary_depth_frame: np.ndarray,
    primary_intrinsics: np.ndarray,
    primary_path: Path,
    primary_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_path = Path(source["path"])
    data = load_depth_archive(source_path)
    depth_i = data["frame_to_i"].get(int(frame_idx))
    if depth_i is None:
        return np.zeros((0, 3)), {"source_id": source["source_id"], "path": str(source_path), "status": "missing_anchor_frame"}
    intr_all = data.get("intrinsics_fx_fy_cx_cy")
    intr = intr_all[depth_i] if intr_all is not None else np.asarray(primary_intrinsics, dtype=np.float64)
    depth_frame, intr = resize_depth_to_mask(data["depth"][depth_i], intr, mask.shape[:2])
    scale = 1.0
    scale_status = "identity_primary" if source_path == primary_path else "scale_aligned_to_primary_mask_median"
    if source_path != primary_path:
        valid = (mask > 127) & np.isfinite(primary_depth_frame) & np.isfinite(depth_frame) & (primary_depth_frame > 0.1) & (depth_frame > 0.1)
        if int(valid.sum()) >= 64:
            ratio = np.median(primary_depth_frame[valid].astype(np.float64) / np.maximum(depth_frame[valid].astype(np.float64), 1.0e-6))
            if np.isfinite(ratio) and 0.02 <= float(ratio) <= 50.0:
                scale = float(ratio)
                depth_frame = (depth_frame.astype(np.float32) * scale).astype(np.float32)
            else:
                scale_status = "invalid_scale_ratio_source_skipped"
                return np.zeros((0, 3)), {"source_id": source["source_id"], "path": str(source_path), "status": scale_status, "scale_to_primary": None}
        else:
            scale_status = "insufficient_overlap_for_scale_source_skipped"
            return np.zeros((0, 3)), {"source_id": source["source_id"], "path": str(source_path), "status": scale_status, "scale_to_primary": None, "overlap_pixels": int(valid.sum())}
    points, foreground_stats = backproject_mask_depth(mask, depth_frame, intr, return_stats=True)
    points, support_stats = filter_points_to_reference_support(points, primary_points)
    status = "used" if len(points) else "no_valid_points_after_backprojection"
    if support_stats.get("support_filter_status") == "support_filter_rejected_too_many_points_kept_original":
        points = np.zeros((0, 3), dtype=np.float64)
        status = "primary_support_mismatch_source_skipped"
    return points, {
        "source_id": source["source_id"],
        "path": str(source_path),
        "method_family": source.get("method_family"),
        "role": source.get("role"),
        "status": status,
        "scale_status": scale_status,
        "scale_to_primary": float(scale),
        "point_count": int(len(points)),
        "depth_shape_hw": [int(depth_frame.shape[0]), int(depth_frame.shape[1])],
        "foreground_depth_filter": foreground_stats,
        "primary_support_filter": support_stats,
    }


def create_heightfield_mesh(points: np.ndarray, grid_res: int = 80, padding: float = 0.002) -> tuple[Any, np.ndarray, np.ndarray]:
    """Create a mesh from a point cloud using a heightmap approach."""
    center = points.mean(axis=0)
    pts = points - center

    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    pts_aligned = pts @ evecs

    x_min, x_max = pts_aligned[:, 0].min() - padding, pts_aligned[:, 0].max() + padding
    y_min, y_max = pts_aligned[:, 1].min() - padding, pts_aligned[:, 1].max() + padding

    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    grid_x, grid_y = np.meshgrid(xs, ys)

    grid_z_front = np.full_like(grid_x, np.nan)
    grid_z_back = np.full_like(grid_x, np.nan)

    x_idx = np.clip(((pts_aligned[:, 0] - x_min) / (x_max - x_min) * (grid_res - 1)).astype(int), 0, grid_res - 1)
    y_idx = np.clip(((pts_aligned[:, 1] - y_min) / (y_max - y_min) * (grid_res - 1)).astype(int), 0, grid_res - 1)

    for i in range(len(pts_aligned)):
        gx, gy = x_idx[i], y_idx[i]
        z = pts_aligned[i, 2]
        if np.isnan(grid_z_front[gy, gx]) or z < grid_z_front[gy, gx]:
            grid_z_front[gy, gx] = z
        if np.isnan(grid_z_back[gy, gx]) or z > grid_z_back[gy, gx]:
            grid_z_back[gy, gx] = z

    from scipy.ndimage import distance_transform_edt
    for grid_z in [grid_z_front, grid_z_back]:
        nan_mask = np.isnan(grid_z)
        if nan_mask.any() and not nan_mask.all():
            indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            filled = grid_z[tuple(indices)]
            grid_z[nan_mask] = filled[nan_mask]

    med_z = np.nanmedian(pts_aligned[:, 2])
    grid_z_front = np.nan_to_num(grid_z_front, nan=med_z)
    grid_z_back = np.nan_to_num(grid_z_back, nan=med_z)

    verts_front = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z_front.ravel()])
    verts_back = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z_back.ravel()])

    faces_front = []
    for i in range(grid_res - 1):
        for j in range(grid_res - 1):
            v0 = i * grid_res + j
            v1 = i * grid_res + j + 1
            v2 = (i + 1) * grid_res + j
            v3 = (i + 1) * grid_res + j + 1
            faces_front.append([v0, v1, v3])
            faces_front.append([v0, v3, v2])

    all_verts = np.vstack([verts_front, verts_back])
    n_front = len(verts_front)
    faces_back = [[f[0] + n_front, f[2] + n_front, f[1] + n_front] for f in faces_front]

    side_faces = []
    for i in range(grid_res - 1):
        j = grid_res - 1
        v0 = i * grid_res + j
        v1 = (i + 1) * grid_res + j
        v0b = v0 + n_front
        v1b = v1 + n_front
        side_faces.append([v0, v1b, v1])
        side_faces.append([v0, v0b, v1b])
        j = 0
        v0 = i * grid_res + j
        v1 = (i + 1) * grid_res + j
        v0b = v0 + n_front
        v1b = v1 + n_front
        side_faces.append([v0, v1, v1b])
        side_faces.append([v0, v1b, v0b])
    for j in range(grid_res - 1):
        i = 0
        v0 = i * grid_res + j
        v1 = i * grid_res + j + 1
        v0b = v0 + n_front
        v1b = v1 + n_front
        side_faces.append([v0, v1, v1b])
        side_faces.append([v0, v1b, v0b])
        i = grid_res - 1
        v0 = i * grid_res + j
        v1 = i * grid_res + j + 1
        v0b = v0 + n_front
        v1b = v1 + n_front
        side_faces.append([v0, v1b, v1])
        side_faces.append([v0, v0b, v1b])

    all_faces = np.array(faces_front + faces_back + side_faces, dtype=np.int64)
    # The exported OBJ is a canonical object mesh, not an absolute anchor-frame point cloud.
    # Downstream pose rows map these centered coordinates into each frame's metric camera/world coordinates.
    all_verts_canonical = all_verts @ evecs.T
    mesh = trimesh.Trimesh(vertices=all_verts_canonical, faces=all_faces, process=True)
    return mesh, center, evecs


def load_preferred_keyframes(run_root: Path) -> set[int]:
    path = run_root / "measurements" / "object_candidates" / "segmentation_stable_keyframes.json"
    if not path.exists():
        return set()
    try:
        report = load_json(path)
    except Exception:
        return set()
    frames: set[int] = set()
    for row in report.get("selected_keyframes", []) if isinstance(report.get("selected_keyframes"), list) else []:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            frames.add(int(row["frame_idx"]))
    return frames


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    object_id = args.object_id
    depth_sources = resolve_depth_sources(run_root, args)
    primary_source = depth_sources[0]
    primary_data = load_depth_archive(Path(primary_source["path"]))
    primary_intrinsics = primary_data.get("intrinsics_fx_fy_cx_cy")
    if primary_intrinsics is None:
        raise ContractError(f"primary_depth_source_missing_intrinsics: {primary_source['path']}")

    try:
        mask_dir = resolve_current_object_mask_dir(run_root, object_id)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    mask_count = len(list(mask_dir.glob("*.png")))
    preferred_frames = load_preferred_keyframes(run_root)
    anchor = select_anchor_mask_frame(
        mask_dir,
        primary_data["depth"],
        primary_intrinsics,
        primary_data["frame_to_i"],
        max_extent_m=args.max_anchor_extent_m,
        min_points=args.min_anchor_points,
        preferred_frames=preferred_frames,
    )
    best_frame = int(anchor["frame_idx"])
    best_area = int(anchor["area_px"])
    mask = anchor["mask"]
    intr = anchor["primary_intrinsics"]
    primary_depth_frame = anchor["primary_depth_frame"]
    print(
        f"Best frame: {best_frame} (mask area={best_area}px, total masks={mask_count}, "
        f"mask_source={mask_dir}, policy={anchor['selection_policy']}, primary_depth={primary_source['source_id']})",
        flush=True,
    )

    primary_path = Path(primary_source["path"])
    primary_points = np.asarray(anchor["primary_points"], dtype=np.float64)
    per_source_reports: list[dict[str, Any]] = [
        {
            "source_id": primary_source["source_id"],
            "path": str(primary_path),
            "method_family": primary_source.get("method_family"),
            "role": primary_source.get("role"),
            "status": "used",
            "scale_status": "identity_primary",
            "scale_to_primary": 1.0,
            "point_count": int(len(primary_points)),
            "depth_shape_hw": [int(primary_depth_frame.shape[0]), int(primary_depth_frame.shape[1])],
            "foreground_depth_filter": anchor["selected"].get("foreground_depth_filter"),
            "primary_support_filter": {"support_filter_status": "identity_primary_reference"},
        }
    ]
    point_sets: list[np.ndarray] = [primary_points]
    for source in depth_sources[1:]:
        points_i, report_i = source_points_for_anchor(source, best_frame, mask, primary_depth_frame, intr, primary_path, primary_points)
        per_source_reports.append(report_i)
        if len(points_i):
            point_sets.append(points_i)
    if not point_sets:
        raise ContractError("no_depth_source_backprojected_points")
    points = np.vstack(point_sets)
    if len(points) < int(args.min_anchor_points):
        raise ContractError(f"too_few_fused_anchor_points: {len(points)}")

    print(f"Backprojected {len(points)} fused points at frame {best_frame}", flush=True)
    extent = point_extent(points)
    print(f"Object extent: x={extent[0]:.4f}m, y={extent[1]:.4f}m, z={extent[2]:.4f}m", flush=True)

    mesh, center, evecs = create_heightfield_mesh(points, grid_res=args.grid_res)
    print(f"Heightfield mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)

    output_dir = run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / "mesh_candidate.obj"
    mesh.export(str(mesh_path))

    pcd = trimesh.PointCloud(points)
    point_cloud_path = output_dir / "best_frame_pointcloud.ply"
    pcd.export(str(point_cloud_path))

    summary = {
        "schema": "v21_mesh_candidate.v1",
        "status": "ok",
        "method": "selected_depth_bundle_anchor_frame_multisource_mask_depth_centered_heightfield",
        "object_id": object_id,
        "run_root": str(run_root),
        "anchor_frame": best_frame,
        "anchor_mask_area_px": best_area,
        "total_mask_frames": mask_count,
        "mask_source": str(mask_dir),
        "mask_source_policy": "active_v21_object_mask_source",
        "anchor_selection_policy": anchor["selection_policy"],
        "max_anchor_extent_m": anchor["max_anchor_extent_m"],
        "anchor_candidate_count_after_filter": anchor["candidate_count"],
        "anchor_preferred_keyframe_candidates": anchor["preferred_keyframe_candidate_count"],
        "anchor_ranking_pool": anchor["ranking_pool"],
        "preferred_keyframes": sorted(int(v) for v in preferred_frames),
        "anchor_rankable_candidate_count": anchor["rankable_candidate_count"],
        "anchor_rankable_area_floor_px": anchor["rankable_area_floor_px"],
        "anchor_rejected_by_extent_count": anchor["rejected_by_extent_count"],
        "anchor_rejected_by_extent_examples": anchor["rejected_by_extent_examples"],
        "backprojected_points": int(len(points)),
        "object_extent_m": extent.tolist(),
        "object_center_m": center.tolist(),
        "canonical_coordinate_source": "anchor_frame_fused_points_minus_object_center_m_in_camera_world_axes",
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_face_count": int(len(mesh.faces)),
        "mesh_path": str(mesh_path),
        "point_cloud_path": str(point_cloud_path),
        "depth_selection_report": str(args.depth_selection_report) if args.depth_selection_report else str(run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json"),
        "depth_registry": str(args.depth_registry) if args.depth_registry else str(run_root / "measurements" / "camera_depth" / "v20_depth_registry" / "depth_candidate_registry.json"),
        "primary_depth_source": {k: str(v) if isinstance(v, Path) else v for k, v in primary_source.items() if k != "registry_candidate"},
        "depth_sources": per_source_reports,
        "used_depth_source_count": int(sum(1 for row in per_source_reports if row.get("status") == "used")),
        "fusion_method": "anchor_frame_mask_median_scale_alignment_then_point_cloud_union",
        "intrinsics_at_anchor": intr.tolist(),
        "candidate_state": "mesh_candidate_not_yet_pose_fitted",
        "next_required_step": "per_frame_se3_pose_fitting_against_observed_surfaces",
    }
    completion_report = dict(summary)
    completion_report["outputs"] = {
        "completed_mesh_labeled": str(mesh_path),
        "point_cloud": str(point_cloud_path),
    }
    completion_report["claim_scope"] = "Single-anchor mask/depth heightfield mesh candidate using the selected depth bundle and scale-aligned auxiliary depth sources. This is a candidate prior for pose fitting, not accepted complete object geometry."
    summary_path = output_dir / "mesh_candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "mesh_completion_report.json").write_text(json.dumps(completion_report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--depth-selection-report", default=None)
    ap.add_argument("--depth-registry", default=None)
    ap.add_argument("--depth-npz", default=None, help="Explicit single depth NPZ for debugging; bypasses selected depth bundle.")
    ap.add_argument("--grid-res", type=int, default=80)
    ap.add_argument("--max-anchor-extent-m", type=float, default=0.5, help="Max per-axis 3D extent for selecting a sane mesh anchor frame; falls back to unfiltered selection if no frame passes.")
    ap.add_argument("--min-anchor-points", type=int, default=100)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
