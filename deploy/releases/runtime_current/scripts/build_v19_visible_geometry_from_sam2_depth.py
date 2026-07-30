#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def export_anchor_visible_surface_mesh(
    *,
    output_dir: Path,
    object_id: str,
    anchor_frame: int,
    anchor_centroid_world_m: np.ndarray,
    anchor_points_world_m: np.ndarray,
    min_voxel_m: float,
    voxel_divisor: float,
    poisson_depth: int,
    poisson_density_quantile: float,
) -> dict[str, Any]:
    """Export the selected anchor's visible surfels in object-canonical coordinates.

    The downstream rigid completion/pose components assume the completed mesh is
    in an object-canonical frame and per-frame pose rows map that canonical mesh
    into world coordinates.  Therefore the anchor point cloud is centered at the
    anchor-frame visible centroid rather than written in world coordinates.
    """
    points_world = np.asarray(anchor_points_world_m, dtype=np.float64)
    centroid = np.asarray(anchor_centroid_world_m, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 30:
        return {
            "status": "too_few_anchor_visible_points_for_mesh_export",
            "anchor_frame_idx": int(anchor_frame),
            "point_count_input": int(len(points_world)) if points_world.ndim == 2 else 0,
            "blockers": ["anchor visible surfels missing or below 30 points"],
        }
    if centroid.shape != (3,) or not np.isfinite(centroid).all() or not np.isfinite(points_world).all():
        raise RuntimeError("anchor visible surface contains invalid coordinates")

    points_canonical = points_world - centroid[None, :]
    object_dir = output_dir / "anchor_visible_surface_mesh" / safe_name(object_id.replace("object:", "object_"))
    object_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_canonical.astype(np.float64))
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    voxel_size = max(float(min_voxel_m), diag / max(float(voxel_divisor), 1.0)) if diag > 0.0 else float(min_voxel_m)
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(pcd.points, dtype=np.float64)
    out: dict[str, Any] = {
        "status": "anchor_visible_surface_mesh_exported",
        "coordinate_frame": "object_canonical_anchor_centroid_frame",
        "canonical_coordinate_source": "selected_anchor_frame_visible_surfels_minus_anchor_centroid_world_m",
        "anchor_frame_idx": int(anchor_frame),
        "anchor_centroid_world_m": centroid.astype(float).tolist(),
        "point_count_input": int(points_canonical.shape[0]),
        "point_count_downsampled": int(down_points.shape[0]),
        "voxel_size_m": float(voxel_size),
        "canonical_bbox_min_m": down_points.min(axis=0).astype(float).tolist() if len(down_points) else None,
        "canonical_bbox_max_m": down_points.max(axis=0).astype(float).tolist() if len(down_points) else None,
        "fused_point_cloud_path": None,
        "poisson_mesh_path": None,
        "convex_hull_mesh_path": None,
        "blockers": [],
        "claim_scope": "selected-frame metric visible surface for TRELLIS alignment; not complete hidden object geometry",
    }
    pcd_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_visible_points_canonical.ply"
    if not o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write anchor visible point cloud: {pcd_path}")
    out["fused_point_cloud_path"] = str(pcd_path)
    if down_points.shape[0] < 30:
        out["status"] = "too_few_downsampled_anchor_points_for_mesh"
        out["blockers"].append("too_few_downsampled_anchor_points_for_mesh")
        return out
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        hull_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_convex_hull_visible_candidate.ply"
        if o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False):
            out["convex_hull_mesh_path"] = str(hull_path)
            out["convex_hull_vertices"] = int(np.asarray(hull.vertices).shape[0])
            out["convex_hull_faces"] = int(np.asarray(hull.triangles).shape[0])
    except Exception as exc:
        out["blockers"].append(f"convex_hull_failed:{type(exc).__name__}:{exc}")
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 4.0, float(min_voxel_m) * 4.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(poisson_depth))
        mesh = mesh.crop(bbox)
        densities_np = np.asarray(densities)
        if densities_np.size and np.asarray(mesh.vertices).shape[0] == densities_np.shape[0]:
            keep_threshold = float(np.quantile(densities_np, float(poisson_density_quantile)))
            mesh.remove_vertices_by_mask(densities_np < keep_threshold)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        poisson_path = object_dir / f"frame_{anchor_frame:06d}_{safe_name(object_id)}_anchor_poisson_visible_mesh.ply"
        if not o3d.io.write_triangle_mesh(str(poisson_path), mesh, write_ascii=False, compressed=False):
            raise RuntimeError("Open3D write_triangle_mesh returned false")
        out["poisson_mesh_path"] = str(poisson_path)
        out["poisson_vertices"] = int(np.asarray(mesh.vertices).shape[0])
        out["poisson_faces"] = int(np.asarray(mesh.triangles).shape[0])
    except Exception as exc:
        out["blockers"].append(f"poisson_reconstruction_failed:{type(exc).__name__}:{exc}")
        if not out.get("convex_hull_mesh_path"):
            out["status"] = "anchor_visible_surface_point_cloud_only_mesh_failed"
        else:
            out["status"] = "anchor_visible_surface_hull_only_poisson_failed"
    return out


def raw_frame_map(manifest_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = load_json(manifest_path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{manifest_path} must contain a nonempty frames list")
    out: dict[int, dict[str, Any]] = {}
    for row_raw in frames:
        if not isinstance(row_raw, dict):
            continue
        idx = int(row_raw.get("frame_idx", row_raw.get("index", -1)))
        if idx < 0:
            raise RuntimeError(f"raw frame row lacks frame_idx/index: {row_raw}")
        if idx in out:
            raise RuntimeError(f"duplicate raw frame {idx} in {manifest_path}")
        out[idx] = row_raw
    return out, payload


def load_depth_npz(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{path} missing depth keys: {missing}")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    depth = np.asarray(blob["depth"], dtype=np.float32)
    intr = np.asarray(blob["intrinsics_fx_fy_cx_cy"], dtype=np.float64)
    if depth.ndim != 3:
        raise RuntimeError(f"{path} depth must have shape [N,H,W], got {depth.shape}")
    if len(frame_idx) != depth.shape[0] or len(frame_idx) != intr.shape[0]:
        raise RuntimeError(f"{path} has inconsistent frame/depth/intrinsics rows")
    source_size = np.asarray(blob["source_size"], dtype=float) if "source_size" in blob.files else None
    return {
        "path": str(path),
        "frame_idx": frame_idx,
        "depth": depth,
        "intrinsics": intr,
        "source_size": source_size,
        "frame_to_i": {int(idx): int(i) for i, idx in enumerate(frame_idx)},
    }


def load_calibration_contract(path: Path | None) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None]:
    if path is None:
        return None, None, None
    if not path.exists():
        raise FileNotFoundError(f"missing calibration contract: {path}")
    payload = load_json(path)
    intr = np.asarray(payload.get("intrinsics_fx_fy_cx_cy"), dtype=float).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or float(intr[0]) <= 0.0 or float(intr[1]) <= 0.0:
        raise RuntimeError(f"calibration contract has invalid intrinsics_fx_fy_cx_cy: {path}")
    source = str(payload.get("intrinsics_source") or payload.get("method") or "v19_calibration_contract")
    source = f"calibration_contract:{source}"
    summary = {
        "path": str(path),
        "intrinsics_fx_fy_cx_cy": [float(v) for v in intr.tolist()],
        "intrinsics_source": source,
        "fov_degrees": payload.get("fov_degrees"),
        "aggregation": payload.get("aggregation"),
    }
    return intr.astype(float), source, summary


def load_camera_npz(path: Path | None) -> dict[int, tuple[np.ndarray, str]]:
    if path is None:
        return {}
    blob = np.load(path, allow_pickle=True)
    if "frame_idx" not in blob.files:
        raise RuntimeError(f"camera npz {path} lacks frame_idx")
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    poses: dict[int, tuple[np.ndarray, str]] = {}
    if "T_world_camera" in blob.files:
        mats = np.asarray(blob["T_world_camera"], dtype=float)
        source = "camera_npz_T_world_camera"
    elif "T_world_camera_metric_current_v18" in blob.files:
        mats = np.asarray(blob["T_world_camera_metric_current_v18"], dtype=float)
        source = "camera_npz_T_world_camera_metric_current_v18"
    elif "R_c2w" in blob.files and "t_c2w" in blob.files:
        r = np.asarray(blob["R_c2w"], dtype=float)
        t = np.asarray(blob["t_c2w"], dtype=float)
        if len(r) != len(frame_idx) or len(t) != len(frame_idx):
            raise RuntimeError(f"camera npz {path} has inconsistent R_c2w/t_c2w rows")
        for i, idx in enumerate(frame_idx):
            T = np.eye(4, dtype=float)
            T[:3, :3] = r[i]
            T[:3, 3] = t[i]
            if T.shape != (4, 4) or not np.isfinite(T).all():
                raise RuntimeError(f"camera npz {path} invalid pose for frame {idx}")
            poses[int(idx)] = (T, "camera_npz_R_c2w_t_c2w")
        return poses
    else:
        raise RuntimeError(
            f"camera npz {path} lacks a supported camera pose key; expected T_world_camera, "
            "T_world_camera_metric_current_v18, or R_c2w/t_c2w"
        )
    if mats.shape[0] != len(frame_idx) or mats.shape[1:] != (4, 4):
        raise RuntimeError(f"camera npz {path} pose shape mismatch: {mats.shape}")
    for i, idx in enumerate(frame_idx):
        T = np.asarray(mats[i], dtype=float)
        if not np.isfinite(T).all():
            raise RuntimeError(f"camera npz {path} invalid pose for frame {idx}")
        poses[int(idx)] = (T, source)
    return poses


def load_base_annotations(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{path} must contain frames list")
    out: dict[int, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("frame_idx", -1))
        if idx < 0:
            raise RuntimeError(f"base annotation row lacks frame_idx: {row}")
        if idx in out:
            raise RuntimeError(f"duplicate base annotation frame {idx}")
        out[idx] = row
    return out


def load_object_plan_record(path: Path | None, track_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload, dict) else None
    if not isinstance(plan, dict):
        plan = payload if isinstance(payload, dict) else {}
    for row in as_list(plan.get("objects")):
        if isinstance(row, dict) and str(row.get("track_id")) == str(track_id):
            return row
    return None


def load_sam2_track(args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], Path]:
    if args.sam2_track_json is not None:
        path = args.sam2_track_json
    else:
        path = args.sam2_root / args.track_id / "sam2" / "sam2_track.json"
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"SAM2 track must be an object: {path}")
    out: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        try:
            idx = int(key)
        except ValueError as exc:
            raise RuntimeError(f"SAM2 track key is not a frame index: {key}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"SAM2 track row for frame {idx} is not an object")
        out[idx] = value
    return out, path


def frame_range(raw_frames: dict[int, dict[str, Any]], start: int | None, end: int | None) -> list[int]:
    indices = sorted(raw_frames)
    lo = indices[0] if start is None else int(start)
    hi = indices[-1] if end is None else int(end)
    if hi < lo:
        raise RuntimeError(f"invalid frame range {lo}:{hi}")
    selected = [idx for idx in indices if lo <= idx <= hi]
    if not selected:
        raise RuntimeError(f"no raw frames in range {lo}:{hi}")
    return selected


def base_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    for key in ("T_world_camera_metric", "T_world_camera", "T_world_camera_metric_current_v18"):
        value = camera.get(key)
        arr = np.asarray(value if value is not None else [], dtype=float)
        if arr.shape == (4, 4) and np.isfinite(arr).all():
            return arr, f"base_annotations_camera_{key}"
    return None


def base_camera_intrinsics(frame: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    value = camera.get("intrinsics_fx_fy_cx_cy")
    arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    if arr.shape == (4,) and np.isfinite(arr).all() and float(arr[0]) > 0.0 and float(arr[1]) > 0.0:
        source = str(camera.get("intrinsics_source") or "base_annotations_camera_intrinsics_fx_fy_cx_cy")
        return arr.astype(float), f"base_annotations:{source}"
    return None


def resolve_camera_pose(
    frame_idx: int,
    frame: dict[str, Any],
    camera_poses: dict[int, tuple[np.ndarray, str]],
    allow_camera_frame_world: bool,
) -> tuple[np.ndarray, str]:
    base = base_camera_pose(frame)
    if base is not None:
        return base
    if frame_idx in camera_poses:
        return camera_poses[frame_idx]
    if allow_camera_frame_world:
        return np.eye(4, dtype=float), "explicit_camera_frame_world_identity_not_metric_temporal_world"
    raise RuntimeError(
        f"frame {frame_idx} lacks camera/world pose. Provide --base-annotations with camera fields, "
        "--camera-npz, or explicit --allow-camera-frame-world."
    )


def localize_path(path: str | Path, remote_root: Path | None, local_root: Path | None) -> Path:
    direct = Path(path)
    if direct.exists():
        return direct
    if remote_root is not None and local_root is not None:
        for src, dst in ((remote_root, local_root), (local_root, remote_root)):
            try:
                rel = direct.relative_to(src)
            except ValueError:
                continue
            candidate = dst / rel
            if candidate.exists():
                return candidate
    raise FileNotFoundError(str(path))


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {path}")
    return mask > 0


def bbox_xyxy_from_mask(mask: np.ndarray, source_width: int, source_height: int) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    sx = float(source_width) / float(mask.shape[1])
    sy = float(source_height) / float(mask.shape[0])
    return [float(xs.min() * sx), float(ys.min() * sy), float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]


def scaled_intrinsics_for_depth(intr: np.ndarray, depth_shape: tuple[int, int], source_size: np.ndarray | None) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(intr, dtype=float).tolist()
    h, w = depth_shape
    if source_size is not None and source_size.size >= 2:
        source_w = float(source_size[0])
        source_h = float(source_size[1])
        if source_w > 0 and source_h > 0 and (abs(source_w - w) > 1e-6 or abs(source_h - h) > 1e-6):
            sx = float(w) / source_w
            sy = float(h) / source_h
            return np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=float)
    return np.asarray([fx, fy, cx, cy], dtype=float)


def subtract_hand_owned_bbox_regions(
    mask: np.ndarray,
    base_frame: dict[str, Any],
    *,
    source_width: int,
    source_height: int,
    pad_px: int,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove image regions owned by visible hands before lifting object depth.

    A semantic object mask can cover pixels where a hand occludes the object. Those
    pixels are not visible object surface.  This conservative ownership filter
    uses available hand detections from base annotations; it is intentionally
    category-agnostic and records how much support was removed.
    """
    if not enabled:
        return mask, {"state": "disabled", "input_mask_pixels": int(mask.sum()), "output_mask_pixels": int(mask.sum())}
    out = mask.copy()
    h, w = out.shape[:2]
    sx = float(w) / float(max(1, int(source_width)))
    sy = float(h) / float(max(1, int(source_height)))
    removed_total = 0
    boxes: list[dict[str, Any]] = []
    for hand in as_list(base_frame.get("hands")):
        if not isinstance(hand, dict):
            continue
        if hand.get("same_frame_detection") is False and not hand.get("hawor_candidate_present"):
            continue
        box = hand.get("bbox_xyxy")
        if not (isinstance(box, list) and len(box) >= 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        pad = int(max(0, pad_px))
        xi1 = max(0, int(np.floor(x1 * sx - pad)))
        yi1 = max(0, int(np.floor(y1 * sy - pad)))
        xi2 = min(w, int(np.ceil(x2 * sx + pad)))
        yi2 = min(h, int(np.ceil(y2 * sy + pad)))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        before = int(out.sum())
        out[yi1:yi2, xi1:xi2] = False
        removed = before - int(out.sum())
        removed_total += int(removed)
        boxes.append({
            "hand_side": hand.get("hand_side"),
            "source_bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "mask_bbox_xyxy": [int(xi1), int(yi1), int(xi2), int(yi2)],
            "removed_mask_pixels": int(removed),
        })
    return out, {
        "state": "hand_owned_bbox_regions_subtracted" if boxes else "no_hand_bboxes_available",
        "input_mask_pixels": int(mask.sum()),
        "output_mask_pixels": int(out.sum()),
        "removed_mask_pixels": int(removed_total),
        "hand_boxes": boxes,
        "pad_px_in_mask_coordinates": int(max(0, pad_px)),
        "claim_scope": "Pixels inside visible hand support are not lifted as visible object surface; uncertain occluded object surface remains unobserved.",
    }


def choose_visible_points(
    valid: np.ndarray,
    depth: np.ndarray,
    intr: np.ndarray,
    T_world_camera: np.ndarray,
    *,
    pixel_stride: int,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stride = max(1, int(pixel_stride))
    sampled = np.zeros_like(valid, dtype=bool)
    sampled[::stride, ::stride] = valid[::stride, ::stride]
    ys, xs = np.where(sampled)
    if len(xs) == 0:
        ys, xs = np.where(valid)
    if len(xs) > int(max_points):
        order = rng.choice(len(xs), size=int(max_points), replace=False)
        ys = ys[order]
        xs = xs[order]
    z = depth[ys, xs].astype(float)
    fx, fy, cx, cy = intr.astype(float).tolist()
    X = (xs.astype(float) - cx) * z / fx
    Y = (ys.astype(float) - cy) * z / fy
    cam = np.column_stack([X, Y, z])
    hom = np.column_stack([cam, np.ones(len(cam), dtype=float)])
    world = (hom @ T_world_camera.T)[:, :3]
    keep = np.isfinite(world).all(axis=1) & np.isfinite(cam).all(axis=1)
    cam = cam[keep]
    world = world[keep]
    summary = {
        "valid_depth_mask_pixels": int(valid.sum()),
        "sampled_points_before_finite_filter": int(len(z)),
        "sampled_points": int(len(world)),
        "pixel_stride": int(stride),
        "max_points": int(max_points),
    }
    return cam.astype(float), world.astype(float), summary


def normalize01(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def raw_image_path_from_row(raw_row: dict[str, Any]) -> str:
    return str(raw_row.get("raw_frame_path") or raw_row.get("rgb") or raw_row.get("path") or "")


def candidate_mask_summary(mask: np.ndarray, source_width: int, source_height: int) -> dict[str, Any]:
    bbox = bbox_xyxy_from_mask(mask, source_width, source_height)
    area = int(mask.sum())
    h, w = mask.shape[:2]
    component_count = 0
    largest_component_px = 0
    if area > 0:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_count = max(0, int(n) - 1)
        if n > 1:
            counts = np.bincount(labels.reshape(-1))[1:]
            largest_component_px = int(counts.max()) if counts.size else 0
    if len(bbox) >= 4:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        margin = 0.03 * float(min(max(1, source_width), max(1, source_height)))
        touches_border = bool(x0 <= margin or y0 <= margin or x1 >= float(source_width) - margin or y1 >= float(source_height) - margin)
        bbox_area = float(bw * bh)
    else:
        x0 = y0 = x1 = y1 = bw = bh = bbox_area = 0.0
        touches_border = True
    return {
        "mask_area_px": area,
        "component_count": int(component_count),
        "largest_component_px": int(largest_component_px),
        "bbox_xyxy": bbox,
        "bbox_width_px": float(bw),
        "bbox_height_px": float(bh),
        "bbox_area_px": float(bbox_area),
        "touches_or_near_image_border": bool(touches_border),
    }


def build_anchor_candidate_proposals(
    *,
    args: argparse.Namespace,
    object_id: str,
    visible_data: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Write anchor candidate evidence without choosing the anchor.

    The score is a proposal heuristic only.  It intentionally exposes the raw
    factors and review sheet so the runtime agent can choose, reject, or request
    another proposal after subjective visual/geometric inspection.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, vis in sorted(visible_data.items()):
        mask = read_mask(Path(vis["mask_path"]))
        source_width = int(vis.get("source_width") or mask.shape[1])
        source_height = int(vis.get("source_height") or mask.shape[0])
        mask_stats = candidate_mask_summary(mask, source_width, source_height)
        world_points = np.asarray(vis.get("world_points"), dtype=float)
        extent = world_points.max(axis=0) - world_points.min(axis=0) if world_points.ndim == 2 and len(world_points) else np.zeros(3, dtype=float)
        diag = float(np.linalg.norm(extent))
        ownership = vis.get("object_surface_ownership_filter") if isinstance(vis.get("object_surface_ownership_filter"), dict) else {}
        input_mask_px = int(ownership.get("input_mask_pixels") or mask_stats["mask_area_px"] or 0)
        output_mask_px = int(ownership.get("output_mask_pixels") or mask_stats["mask_area_px"] or 0)
        removed_px = int(ownership.get("removed_mask_pixels") or max(0, input_mask_px - output_mask_px))
        removed_frac = float(removed_px / input_mask_px) if input_mask_px > 0 else 0.0
        depth_p05 = float(vis.get("depth_p05_m") or np.nan)
        depth_p95 = float(vis.get("depth_p95_m") or np.nan)
        depth_spread = float(depth_p95 - depth_p05) if np.isfinite(depth_p05) and np.isfinite(depth_p95) else float("nan")
        sample_summary = vis.get("sample_summary") if isinstance(vis.get("sample_summary"), dict) else {}
        rows.append(
            {
                "frame_idx": int(idx),
                "raw_frame_path": raw_image_path_from_row(vis.get("raw_row") if isinstance(vis.get("raw_row"), dict) else {}),
                "object_owned_mask_path": str(vis.get("mask_path")),
                "raw_sam2_mask_path": str(vis.get("raw_sam2_mask_path")),
                "visible_depth_vertex_count": int(len(world_points)),
                "valid_depth_mask_pixels": int(sample_summary.get("valid_depth_mask_pixels") or 0),
                "sampled_points": int(sample_summary.get("sampled_points") or len(world_points)),
                "mask_area_px": int(mask_stats["mask_area_px"]),
                "raw_mask_area_px": int(input_mask_px),
                "hand_owned_removed_px": int(removed_px),
                "hand_owned_removed_fraction": float(removed_frac),
                "component_count": int(mask_stats["component_count"]),
                "largest_component_px": int(mask_stats["largest_component_px"]),
                "bbox_xyxy": mask_stats["bbox_xyxy"],
                "bbox_width_px": float(mask_stats["bbox_width_px"]),
                "bbox_height_px": float(mask_stats["bbox_height_px"]),
                "bbox_area_px": float(mask_stats["bbox_area_px"]),
                "touches_or_near_image_border": bool(mask_stats["touches_or_near_image_border"]),
                "depth_median_m": float(vis.get("depth_median_m") or np.nan),
                "depth_p05_m": depth_p05,
                "depth_p95_m": depth_p95,
                "depth_spread_p95_p05_m": depth_spread,
                "world_extent_m": extent.astype(float).tolist(),
                "world_extent_diag_m": float(diag),
                "intrinsics_source": str(vis.get("intrinsics_source")),
                "camera_source": str(vis.get("camera_source")),
            }
        )
    if not rows:
        raise RuntimeError("no visible rows available for anchor candidate proposal")
    area_values = np.asarray([float(r["mask_area_px"]) for r in rows], dtype=float)
    point_values = np.asarray([float(r["visible_depth_vertex_count"]) for r in rows], dtype=float)
    diag_values = np.asarray([float(r["world_extent_diag_m"]) for r in rows if float(r["world_extent_diag_m"]) > 0.0], dtype=float)
    depth_spreads = np.asarray([float(r["depth_spread_p95_p05_m"]) for r in rows if np.isfinite(float(r["depth_spread_p95_p05_m"]))], dtype=float)
    median_diag = float(np.median(diag_values)) if diag_values.size else 0.0
    depth_spread_hi = float(np.percentile(depth_spreads, 90.0)) if depth_spreads.size else 1.0
    area_hi = float(area_values.max()) if area_values.size else 1.0
    point_hi = float(point_values.max()) if point_values.size else 1.0
    for r in rows:
        area_score = float(r["mask_area_px"]) / max(area_hi, 1.0)
        point_score = float(r["visible_depth_vertex_count"]) / max(point_hi, 1.0)
        hand_clean_score = 1.0 - float(np.clip(r["hand_owned_removed_fraction"], 0.0, 1.0))
        border_score = 0.0 if bool(r["touches_or_near_image_border"]) else 1.0
        comp = max(1, int(r["component_count"]))
        component_score = float(max(0.0, 1.0 - 0.25 * (comp - 1)))
        diag = float(r["world_extent_diag_m"])
        extent_score = float(np.exp(-abs(np.log(max(diag, 1.0e-9) / max(median_diag, 1.0e-9))))) if median_diag > 0 else 0.0
        spread = float(r["depth_spread_p95_p05_m"])
        depth_score = 1.0 - normalize01(spread, 0.0, max(depth_spread_hi, 1.0e-6)) if np.isfinite(spread) else 0.0
        proposal_score = (
            0.24 * area_score
            + 0.16 * point_score
            + 0.18 * hand_clean_score
            + 0.16 * border_score
            + 0.14 * extent_score
            + 0.07 * component_score
            + 0.05 * depth_score
        )
        r["proposal_score"] = float(proposal_score)
        r["proposal_score_terms"] = {
            "mask_area_score": float(area_score),
            "visible_depth_vertex_count_score": float(point_score),
            "hand_clean_score": float(hand_clean_score),
            "not_near_image_border_score": float(border_score),
            "extent_consistency_score": float(extent_score),
            "single_component_score": float(component_score),
            "depth_stability_score": float(depth_score),
        }
        r["score_interpretation"] = "heuristic proposal score only; agent visual/geometric judgment must choose the anchor"
    ranked = sorted(rows, key=lambda r: (-float(r["proposal_score"]), int(r["frame_idx"])))
    top_k = max(1, int(args.anchor_candidate_count))
    min_gap = max(0, int(args.anchor_candidate_min_gap))
    review_rows: list[dict[str, Any]] = []
    for row in ranked:
        if all(abs(int(row["frame_idx"]) - int(prev["frame_idx"])) >= min_gap for prev in review_rows):
            review_rows.append(row)
        if len(review_rows) >= top_k:
            break
    if len(review_rows) < top_k:
        seen = {int(r["frame_idx"]) for r in review_rows}
        for row in ranked:
            if int(row["frame_idx"]) not in seen:
                review_rows.append(row)
                seen.add(int(row["frame_idx"]))
            if len(review_rows) >= top_k:
                break
    report_path = args.output_dir / "anchor_candidate_proposals.json"
    review_path = args.output_dir / "anchor_candidate_review.jpg"
    review_status = render_anchor_candidate_review(
        review_rows=review_rows,
        review_path=review_path,
        panel_width=int(args.anchor_candidate_panel_width),
    )
    report = {
        "method": "build_v19_visible_geometry_from_sam2_depth_anchor_candidate_proposals",
        "status": "ok",
        "object_id": object_id,
        "track_id": args.track_id,
        "claim_scope": "Anchor candidates for agent subjective selection. This report does not choose or accept an anchor frame.",
        "selection_required": True,
        "selection_instruction": "Inspect anchor_candidate_review.jpg and ranked_candidates; write an agent anchor decision before running P09/P11 with --anchor-frame/--selected-frame-idx.",
        "proposal_score_policy": "weighted heuristic over owned mask area, visible depth support, low hand-owned removal, non-border support, metric extent consistency, component count, and depth stability; score is not an acceptance gate.",
        "candidate_count": int(len(rows)),
        "review_candidate_count": int(len(review_rows)),
        "review_diversity_min_frame_gap": int(min_gap),
        "outputs": {
            "anchor_candidate_proposals": str(report_path),
            "anchor_candidate_review": str(review_path) if review_status.get("status") == "ok" else None,
        },
        "review_status": review_status,
        "ranked_candidates": ranked,
        "review_candidates": review_rows,
        "score_population_summary": {
            "max_mask_area_px": int(area_hi),
            "max_visible_depth_vertex_count": int(point_hi),
            "median_world_extent_diag_m": float(median_diag),
            "depth_spread_p90_m": float(depth_spread_hi),
        },
    }
    write_json(report_path, report)
    return report


def render_anchor_candidate_review(*, review_rows: list[dict[str, Any]], review_path: Path, panel_width: int) -> dict[str, Any]:
    panels: list[np.ndarray] = []
    blockers: list[str] = []
    panel_width = max(220, int(panel_width))
    for rank, row in enumerate(review_rows, start=1):
        raw_path = Path(str(row.get("raw_frame_path") or ""))
        mask_path = Path(str(row.get("object_owned_mask_path") or ""))
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR) if raw_path.is_file() else None
        if raw is None:
            blockers.append(f"missing_raw_frame:{row.get('frame_idx')}:{raw_path}")
            raw = np.full((480, 480, 3), 240, dtype=np.uint8)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None:
            blockers.append(f"missing_object_owned_mask:{row.get('frame_idx')}:{mask_path}")
            mask_bool_img = np.zeros(raw.shape[:2], dtype=bool)
        else:
            if mask.shape[:2] != raw.shape[:2]:
                mask = cv2.resize(mask, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool_img = mask > 0
        overlay = raw.copy()
        tint = overlay.copy()
        tint[mask_bool_img] = (40, 220, 70)
        overlay[mask_bool_img] = cv2.addWeighted(tint, 0.45, overlay, 0.55, 0)[mask_bool_img]
        cnts, _ = cv2.findContours(mask_bool_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
        scale = panel_width / float(max(1, overlay.shape[1]))
        ph = int(round(overlay.shape[0] * scale))
        panel_img = cv2.resize(overlay, (panel_width, ph), interpolation=cv2.INTER_AREA)
        label_h = 112
        panel = np.full((ph + label_h, panel_width, 3), 255, dtype=np.uint8)
        panel[:ph] = panel_img
        def put(line: str, y: int, color: tuple[int, int, int] = (0, 0, 0)) -> None:
            cv2.putText(panel, line[:72], (6, ph + y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
        put(f"rank {rank} frame {row.get('frame_idx')} score {float(row.get('proposal_score',0.0)):.3f}", 16)
        put(f"mask {int(row.get('mask_area_px',0))}px pts {int(row.get('visible_depth_vertex_count',0))} border {bool(row.get('touches_or_near_image_border'))}", 34)
        put(f"hand_removed {100.0*float(row.get('hand_owned_removed_fraction',0.0)):.1f}% comps {int(row.get('component_count',0))}", 52)
        ext = row.get("world_extent_m") if isinstance(row.get("world_extent_m"), list) else []
        if len(ext) >= 3:
            put(f"extent m {float(ext[0]):.3f},{float(ext[1]):.3f},{float(ext[2]):.3f}", 70)
        put(f"depth p05-p95 {float(row.get('depth_spread_p95_p05_m',0.0)):.3f}m", 88)
        put("agent must inspect; score is not acceptance", 106, (40, 40, 180))
        panels.append(panel)
    if not panels:
        return {"status": "no_review_candidates", "blockers": blockers}
    cols = min(3, len(panels))
    rows_img: list[np.ndarray] = []
    for start in range(0, len(panels), cols):
        chunk = panels[start:start + cols]
        max_h = max(p.shape[0] for p in chunk)
        padded = []
        for p in chunk:
            if p.shape[0] < max_h:
                pad = np.full((max_h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        while len(padded) < cols:
            padded.append(np.full((max_h, panel_width, 3), 255, dtype=np.uint8))
        rows_img.append(np.hstack(padded))
    sheet = np.vstack(rows_img)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(review_path), sheet)
    return {"status": "ok" if ok else "write_failed", "path": str(review_path), "blockers": blockers[:20]}


def object_row_not_visible(object_id: str, track_id: str, status: str, reason: str, mask_path: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "object_id": object_id,
        "track_id": track_id,
        "label": track_id,
        "status": status,
        "visible": False,
        "v19_visible_geometry_adapter": {
            "state": status,
            "reason": reason,
        },
    }
    if mask_path:
        row["mask_path"] = mask_path
    return row


def remove_existing_object(objects: list[Any], object_id: str, track_id: str) -> list[Any]:
    out = []
    for obj in objects:
        if not isinstance(obj, dict):
            out.append(obj)
            continue
        if obj.get("object_id") == object_id or obj.get("track_id") == track_id:
            continue
        out.append(obj)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_frames, raw_payload = raw_frame_map(args.raw_frame_manifest)
    depth = load_depth_npz(args.depth_npz)
    calibration_intrinsics, calibration_source, calibration_summary = load_calibration_contract(args.calibration_contract)
    camera_poses = load_camera_npz(args.camera_npz)
    base_frames = load_base_annotations(args.base_annotations)
    sam2, sam2_path = load_sam2_track(args)
    object_plan_record = load_object_plan_record(args.object_plan, args.track_id)
    object_id = args.object_id or f"object:{args.track_id}"
    indices = frame_range(raw_frames, args.frame_start, args.frame_end)
    output_indices = sorted(raw_frames) if bool(args.preserve_source_index) else list(indices)
    rng = np.random.default_rng(int(args.seed))

    visible_data: dict[int, dict[str, Any]] = {}
    skipped_rows: list[dict[str, Any]] = []
    for idx in indices:
        track_row = sam2.get(idx, {})
        if not track_row.get("visible") or not track_row.get("mask_path"):
            skipped_rows.append({"frame_idx": idx, "status": "not_visible_in_sam2"})
            continue
        depth_i = depth["frame_to_i"].get(idx)
        if depth_i is None:
            skipped_rows.append({"frame_idx": idx, "status": "missing_depth_row"})
            continue
        source_mask_path = str(track_row["mask_path"])
        mask_path = localize_path(source_mask_path, args.remote_root, args.local_root)
        mask = read_mask(mask_path)
        depth_m = np.asarray(depth["depth"][depth_i], dtype=float)
        raw_row = raw_frames[idx]
        base_frame = copy.deepcopy(base_frames.get(idx, {"frame_idx": idx}))
        source_width = int(raw_row.get("source_width") or raw_payload.get("video", {}).get("width") or raw_row.get("manifest_width") or mask.shape[1])
        source_height = int(raw_row.get("source_height") or raw_payload.get("video", {}).get("height") or raw_row.get("manifest_height") or mask.shape[0])
        mask_owned, ownership_summary = subtract_hand_owned_bbox_regions(
            mask,
            base_frame,
            source_width=source_width,
            source_height=source_height,
            pad_px=int(args.hand_bbox_exclusion_pad_px),
            enabled=bool(args.exclude_hand_bboxes),
        )
        owned_mask_path = args.output_dir / "object_owned_masks" / f"{idx:06d}_{safe_name(object_id)}_object_owned_mask.png"
        owned_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(owned_mask_path), mask_owned.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write object-owned mask: {owned_mask_path}")
        if mask_owned.shape != depth_m.shape:
            mask_depth_owned = cv2.resize(mask_owned.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask_depth_owned = mask_owned
        if calibration_intrinsics is not None:
            raw_intrinsics = calibration_intrinsics
            intrinsics_source = calibration_source or "calibration_contract"
        else:
            base_intr = base_camera_intrinsics(base_frame)
            if base_intr is not None:
                raw_intrinsics, intrinsics_source = base_intr
            else:
                raw_intrinsics = depth["intrinsics"][depth_i]
                intrinsics_source = "depth_npz_intrinsics_fx_fy_cx_cy"
        intr = scaled_intrinsics_for_depth(raw_intrinsics, depth_m.shape, depth["source_size"])
        valid = mask_depth_owned & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
        if int(valid.sum()) < int(args.min_valid_points):
            skipped_rows.append({"frame_idx": idx, "status": "too_few_valid_mask_depth_pixels", "valid_pixels": int(valid.sum())})
            continue
        T_world_camera, camera_source = resolve_camera_pose(idx, base_frame, camera_poses, bool(args.allow_camera_frame_world))
        camera_points, world_points, sample_summary = choose_visible_points(
            valid,
            depth_m,
            intr,
            T_world_camera,
            pixel_stride=int(args.pixel_stride),
            max_points=int(args.max_points),
            rng=rng,
        )
        if len(world_points) < int(args.min_valid_points):
            skipped_rows.append({"frame_idx": idx, "status": "too_few_sampled_visible_points", "sampled_points": int(len(world_points))})
            continue
        visible_data[idx] = {
            "mask_path": str(owned_mask_path),
            "source_mask_path": source_mask_path,
            "raw_sam2_mask_path": str(mask_path),
            "track_row": track_row,
            "raw_row": raw_row,
            "camera_points": camera_points,
            "world_points": world_points,
            "intrinsics": intr,
            "intrinsics_source": intrinsics_source,
            "T_world_camera": T_world_camera,
            "camera_source": camera_source,
            "sample_summary": sample_summary,
            "depth_median_m": float(np.median(depth_m[valid])),
            "depth_p05_m": float(np.percentile(depth_m[valid], 5.0)),
            "depth_p95_m": float(np.percentile(depth_m[valid], 95.0)),
            "mask_shape": list(mask_owned.shape),
            "raw_sam2_mask_shape": list(mask.shape),
            "depth_shape": list(depth_m.shape),
            "object_surface_ownership_filter": ownership_summary,
            "source_width": int(source_width),
            "source_height": int(source_height),
        }

    if not visible_data:
        raise RuntimeError(
            f"no visible metric geometry rows for track_id={args.track_id}. "
            "Check SAM2 masks, depth archive coverage, frame range, and camera-pose availability."
        )

    anchor_candidate_report = build_anchor_candidate_proposals(args=args, object_id=object_id, visible_data=visible_data)
    if bool(args.propose_anchor_candidates_only):
        return {
            "method": "build_v19_visible_geometry_from_sam2_depth",
            "status": "anchor_candidates_proposed_only",
            "case": args.case,
            "track_id": args.track_id,
            "object_id": object_id,
            "claim_scope": "Candidate anchor evidence only. No canonical anchor mesh, visible-geometry annotations, completion, pose, or render state was produced.",
            "outputs": anchor_candidate_report.get("outputs", {}),
            "candidate_count": anchor_candidate_report.get("candidate_count"),
            "review_candidate_count": anchor_candidate_report.get("review_candidate_count"),
        }

    if args.anchor_frame is not None:
        anchor = int(args.anchor_frame)
        if anchor not in visible_data:
            raise RuntimeError(f"--anchor-frame {anchor} has no visible metric geometry")
    else:
        if bool(args.require_anchor_frame):
            outputs = anchor_candidate_report.get("outputs", {}) if isinstance(anchor_candidate_report, dict) else {}
            raise RuntimeError(
                "--anchor-frame is required for this run. Inspect anchor candidates and rerun with an agent-selected anchor. "
                f"candidate_report={outputs.get('anchor_candidate_proposals')} review={outputs.get('anchor_candidate_review')}"
            )
        anchor = max(visible_data, key=lambda idx: len(visible_data[idx]["world_points"]))
    anchor_points = np.asarray(visible_data[anchor]["world_points"], dtype=float)
    anchor_centroid = anchor_points.mean(axis=0)
    anchor_extent_m = anchor_points.max(axis=0) - anchor_points.min(axis=0)
    anchor_diag_m = float(np.linalg.norm(anchor_extent_m))
    if anchor_diag_m <= 0.0 or not np.isfinite(anchor_diag_m):
        raise RuntimeError(f"anchor frame {anchor} has invalid metric extent")
    anchor_mesh_reconstruction = export_anchor_visible_surface_mesh(
        output_dir=args.output_dir,
        object_id=object_id,
        anchor_frame=int(anchor),
        anchor_centroid_world_m=anchor_centroid,
        anchor_points_world_m=anchor_points,
        min_voxel_m=float(args.anchor_mesh_min_voxel_m),
        voxel_divisor=float(args.anchor_mesh_voxel_divisor),
        poisson_depth=int(args.anchor_mesh_poisson_depth),
        poisson_density_quantile=float(args.anchor_mesh_poisson_density_quantile),
    )

    last_pose: dict[str, Any] | None = None
    output_frames: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    camera_source_counts: Counter[str] = Counter()
    for idx in output_indices:
        raw_row = raw_frames[idx]
        frame = copy.deepcopy(base_frames.get(idx, {"frame_idx": idx}))
        frame["frame_idx"] = int(idx)
        raw_text = str(frame.get("raw_frame_path") or raw_row.get("rgb") or "")
        frame["raw_frame_path"] = str(localize_path(raw_text, args.remote_root, args.local_root)) if raw_text else raw_text
        frame["source_width"] = int(raw_row.get("source_width") or raw_payload.get("video", {}).get("width") or raw_row.get("manifest_width") or 0)
        frame["source_height"] = int(raw_row.get("source_height") or raw_payload.get("video", {}).get("height") or raw_row.get("manifest_height") or 0)
        try:
            T_world_camera, camera_source = resolve_camera_pose(idx, frame, camera_poses, bool(args.allow_camera_frame_world))
            camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
            camera = dict(camera)
            camera["T_world_camera_metric"] = T_world_camera.astype(float).tolist()
            camera["position_world_m"] = T_world_camera[:3, 3].astype(float).tolist()
            camera["v19_camera_pose_source"] = camera_source
            frame["camera"] = camera
            camera_source_counts[camera_source] += 1
        except RuntimeError:
            if idx in visible_data:
                raise
        objects = remove_existing_object(as_list(frame.get("objects")), object_id, args.track_id)
        vis = visible_data.get(idx)
        if vis is None:
            track_row = sam2.get(idx, {})
            row = object_row_not_visible(
                object_id,
                args.track_id,
                "not_visible_or_no_metric_depth",
                "SAM2/depth did not provide a visible metric surface for this frame",
                str(track_row.get("mask_path")) if track_row.get("mask_path") else None,
            )
            if bool(args.carry_invisible_pose) and last_pose is not None:
                row["reconstructed_geometry_pose"] = {
                    **last_pose,
                    "pose_source": "carried_nearest_visible_centroid_uncertain_v19_adapter",
                    "pose_uncertainty": "no current visible metric surface; pose is a carried prior for rendering/optimization initialization only",
                }
            objects.append(row)
            frame["objects"] = objects
            output_frames.append(frame)
            rows.append({"frame_idx": idx, "status": "not_visible_or_no_metric_depth"})
            continue
        world_points = np.asarray(vis["world_points"], dtype=float)
        cam_points = np.asarray(vis["camera_points"], dtype=float)
        centroid = world_points.mean(axis=0)
        world_extent_m = world_points.max(axis=0) - world_points.min(axis=0)
        extent_ratio_diag = float(np.linalg.norm(world_extent_m) / max(anchor_diag_m, 1.0e-9))
        extent_ratio_axis = np.divide(
            world_extent_m,
            np.maximum(anchor_extent_m, 1.0e-6),
            out=np.full(3, np.inf, dtype=float),
            where=np.maximum(anchor_extent_m, 1.0e-6) > 0,
        )
        max_extent_ratio_axis = float(np.max(extent_ratio_axis))
        rigid_pose_observation_eligible = bool(
            np.isfinite(extent_ratio_diag)
            and np.isfinite(max_extent_ratio_axis)
            and extent_ratio_diag <= float(args.rigid_extent_ratio_max)
            and max_extent_ratio_axis <= float(args.rigid_extent_axis_ratio_max)
        )
        rigid_pose_observation_reason = (
            "metric_extent_consistent_with_selected_anchor_rigid_object"
            if rigid_pose_observation_eligible
            else "systematic_mask_extent_inconsistent_with_selected_anchor_rigid_object_probable_hand_background_leakage"
        )
        raw_mask = read_mask(Path(vis["mask_path"]))
        bbox = vis["track_row"].get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            bbox = bbox_xyxy_from_mask(raw_mask, int(frame["source_width"]), int(frame["source_height"]))
        pose = {
            "rotation_world_from_canonical_matrix": np.eye(3, dtype=float).tolist(),
            "translation_world_m": centroid.astype(float).tolist(),
            "pose_source": "v19_visible_geometry_adapter_centroid_initial_pose_not_final_rigid_pose",
            "pose_uncertainty": "centroid-only initialization from current visible SAM2/depth surfels; rigid completion and pose fitting must refine it",
            "anchor_frame_idx": int(anchor),
            "anchor_centroid_world_m": anchor_centroid.astype(float).tolist(),
        }
        last_pose = pose
        geom = {
            "status": "visible_surface_from_sam2_mask_metric_depth",
            "source": "build_v19_visible_geometry_from_sam2_depth",
            "frame_idx": int(idx),
            "object_id": object_id,
            "track_id": args.track_id,
            "mask_path": str(vis["mask_path"]),
            "source_mask_path": str(vis.get("source_mask_path")),
            "source_width": int(vis.get("source_width") or frame.get("source_width") or 0),
            "source_height": int(vis.get("source_height") or frame.get("source_height") or 0),
            "depth_npz": str(args.depth_npz),
            "depth_frame_index": int(idx),
            "camera_pose_source": vis["camera_source"],
            "intrinsics_fx_fy_cx_cy": np.asarray(vis["intrinsics"], dtype=float).tolist(),
            "intrinsics_source": vis.get("intrinsics_source"),
            "vertex_count": int(len(world_points)),
            "world_vertices_sample_m": world_points.astype(float).tolist(),
            "camera_vertices_sample_m": cam_points.astype(float).tolist(),
            "centroid_world_m": centroid.astype(float).tolist(),
            "world_extent_m": world_extent_m.astype(float).tolist(),
            "anchor_extent_world_m": anchor_extent_m.astype(float).tolist(),
            "extent_ratio_to_anchor_diag": extent_ratio_diag,
            "extent_ratio_to_anchor_axis": extent_ratio_axis.astype(float).tolist(),
            "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
            "rigid_pose_observation_reason": rigid_pose_observation_reason,
            "depth_median_m": float(vis["depth_median_m"]),
            "depth_p05_m": float(vis["depth_p05_m"]),
            "depth_p95_m": float(vis["depth_p95_m"]),
            "sample_summary": vis["sample_summary"],
            "object_surface_ownership_filter": vis.get("object_surface_ownership_filter"),
            "claim_scope": "visible metric surface measurement only; hand-owned pixels are excluded from object surface before metric lifting; not hidden geometry and not final object pose",
        }
        row_obj = {
            "object_id": object_id,
            "track_id": args.track_id,
            "label": args.track_id,
            "description": (object_plan_record or {}).get("description"),
            "status": "visible_metric_surface_measurement",
            "visible": True,
            "mask_path": str(vis["mask_path"]),
            "bbox_xyxy": [float(x) for x in bbox[:4]],
            "area_px": float(vis["track_row"].get("area_px", raw_mask.sum())),
            "depth_m": float(vis["depth_median_m"]),
            "visible_geometry_candidate": geom,
            "reconstructed_geometry_pose": pose,
            "v19_physical_model": (object_plan_record or {}).get("physical_model"),
            "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
            "rigid_pose_observation_reason": rigid_pose_observation_reason,
        }
        objects.append(row_obj)
        frame["objects"] = objects
        output_frames.append(frame)
        rows.append(
            {
                "frame_idx": int(idx),
                "status": "visible_metric_surface_measurement",
                "vertex_count": int(len(world_points)),
                "mask_path": str(vis["mask_path"]),
                "depth_median_m": float(vis["depth_median_m"]),
                "centroid_world_m": centroid.astype(float).tolist(),
                "world_extent_m": world_extent_m.astype(float).tolist(),
                "extent_ratio_to_anchor_diag": extent_ratio_diag,
                "extent_ratio_to_anchor_axis_max": max_extent_ratio_axis,
                "rigid_pose_observation_eligible": rigid_pose_observation_eligible,
                "rigid_pose_observation_reason": rigid_pose_observation_reason,
                "camera_pose_source": vis["camera_source"],
                "intrinsics_source": vis.get("intrinsics_source"),
            }
        )

    annotations = {
        "frames": output_frames,
        "v19_visible_geometry_adapter": {
            "method": "build_v19_visible_geometry_from_sam2_depth",
            "case": args.case,
            "track_id": args.track_id,
            "object_id": object_id,
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "sam2_track": str(sam2_path),
            "depth_npz": str(args.depth_npz),
            "base_annotations": str(args.base_annotations) if args.base_annotations else None,
            "camera_npz": str(args.camera_npz) if args.camera_npz else None,
            "calibration_contract": str(args.calibration_contract) if args.calibration_contract else None,
            "anchor_frame_idx": int(anchor),
            "claim_scope": "visible metric surfel and initial-pose adapter for rigid branch; downstream completion/pose/interval solvers must produce the physical object pose claim",
        },
    }
    if isinstance(raw_payload.get("video"), dict):
        annotations["raw_video"] = raw_payload["video"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "annotations_v19_visible_geometry.json"
    write_json(annotations_path, annotations)

    visible_mask_rows = [
        {
            "frame_idx": int(row["frame_idx"]),
            "target_entity_id": object_id,
            "object_id": object_id,
            "track_id": args.track_id,
            "status": "v19_visible_metric_surface_mask",
            "measurement_type": "sam2_mask_with_metric_depth_support",
            "mask_path": str(row["mask_path"]),
            "saved_mask_path": str(row["mask_path"]),
            "mask_area_px": None if sam2.get(int(row["frame_idx"]), {}).get("area_px") is None else float(sam2[int(row["frame_idx"])] ["area_px"]),
            "depth_median_m": row.get("depth_median_m"),
            "visible_vertex_count": row.get("vertex_count"),
            "coordinate_frame": "source_image_mask_plus_metric_depth",
            "claim_scope": "visible model mask measurement with metric-depth support; not hidden geometry or object pose",
        }
        for row in rows
        if row.get("status") == "visible_metric_surface_measurement" and row.get("mask_path")
    ]
    visible_mask_report_path = args.output_dir / "v19_visible_mask_report.json"
    visible_mask_report = {
        "method": "build_v19_visible_geometry_from_sam2_depth_visible_mask_report",
        "status": "ok",
        "case": args.case,
        "target_entity_id": object_id,
        "track_id": args.track_id,
        "claim_scope": "SAM2 mask rows with metric depth support for downstream visible-surface/ownership factor builders. This report is a mask measurement index, not object pose.",
        "surface_rows": visible_mask_rows,
    }
    write_json(visible_mask_report_path, visible_mask_report)

    report = {
        "method": "build_v19_visible_geometry_from_sam2_depth",
        "status": "ok",
        "case": args.case,
        "track_id": args.track_id,
        "object_id": object_id,
        "claim_scope": "SAM2 masks plus metric depth/camera are lifted to visible object surfels and centroid initial poses. This is measurement/adaptation input for the rigid branch, not final object pose.",
        "inputs": {
            "raw_frame_manifest": str(args.raw_frame_manifest),
            "sam2_track": str(sam2_path),
            "depth_npz": str(args.depth_npz),
            "base_annotations": str(args.base_annotations) if args.base_annotations else None,
            "camera_npz": str(args.camera_npz) if args.camera_npz else None,
            "object_plan": str(args.object_plan) if args.object_plan else None,
            "remote_root": str(args.remote_root) if args.remote_root else None,
            "local_root": str(args.local_root) if args.local_root else None,
        },
        "outputs": {
            "annotations": str(annotations_path),
            "depth_fused_report": str(args.output_dir / "v19_visible_geometry_depth_fused_report.json"),
            "visible_mask_report": str(visible_mask_report_path),
            "anchor_candidate_proposals": anchor_candidate_report.get("outputs", {}).get("anchor_candidate_proposals") if isinstance(anchor_candidate_report, dict) else None,
            "anchor_candidate_review": anchor_candidate_report.get("outputs", {}).get("anchor_candidate_review") if isinstance(anchor_candidate_report, dict) else None,
            "anchor_visible_surface_mesh": anchor_mesh_reconstruction.get("poisson_mesh_path") or anchor_mesh_reconstruction.get("convex_hull_mesh_path") or anchor_mesh_reconstruction.get("fused_point_cloud_path"),
        },
        "requested_frame_start": int(indices[0]),
        "requested_frame_end": int(indices[-1]),
        "requested_frame_count": int(len(indices)),
        "output_frame_count": int(len(output_indices)),
        "preserve_source_index": bool(args.preserve_source_index),
        "visible_metric_frame_count": int(sum(1 for row in rows if row.get("status") == "visible_metric_surface_measurement")),
        "anchor_frame_idx": int(anchor),
        "anchor_centroid_world_m": anchor_centroid.astype(float).tolist(),
        "anchor_extent_world_m": anchor_extent_m.astype(float).tolist(),
        "anchor_visible_surface_mesh_reconstruction": anchor_mesh_reconstruction,
        "camera_pose_source_counts": dict(camera_source_counts),
        "intrinsics_source_counts": dict(Counter(str(vis.get("intrinsics_source")) for vis in visible_data.values())),
        "calibration_contract": calibration_summary,
        "parameters": {
            "pixel_stride": int(args.pixel_stride),
            "max_points": int(args.max_points),
            "min_valid_points": int(args.min_valid_points),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "allow_camera_frame_world": bool(args.allow_camera_frame_world),
            "carry_invisible_pose": bool(args.carry_invisible_pose),
            "preserve_source_index": bool(args.preserve_source_index),
            "exclude_hand_bboxes": bool(args.exclude_hand_bboxes),
            "hand_bbox_exclusion_pad_px": int(args.hand_bbox_exclusion_pad_px),
            "rigid_extent_ratio_max": float(args.rigid_extent_ratio_max),
            "rigid_extent_axis_ratio_max": float(args.rigid_extent_axis_ratio_max),
            "anchor_candidate_count": int(args.anchor_candidate_count),
            "anchor_candidate_min_gap": int(args.anchor_candidate_min_gap),
            "require_anchor_frame": bool(args.require_anchor_frame),
        },
        "rows": rows,
        "skipped_rows_preview": skipped_rows[:200],
    }
    report_path = args.output_dir / "v19_visible_geometry_adapter_report.json"
    write_json(report_path, report)

    depth_fused = {
        "method": "build_v19_visible_geometry_from_sam2_depth_depth_fused_compat_report",
        "status": "ok",
        "case": args.case,
        "claim": "Compatibility report for V18 rigid evidence bundle. Visible geometry lives in annotations; no hidden geometry is claimed here.",
        "object_rows": [
            {
                "object_id": object_id,
                "track_id": args.track_id,
                "frame_surface_rows": report["visible_metric_frame_count"],
                "visible_geometry_adapter_report": str(report_path),
                "visible_mask_report": str(visible_mask_report_path),
                "annotations": str(annotations_path),
                "mesh_reconstruction": anchor_mesh_reconstruction,
                "object_geometry_complete": False,
                "hidden_geometry_reconstructed": False,
                "complete_object_pose_ready": False,
            }
        ],
    }
    write_json(args.output_dir / "v19_visible_geometry_depth_fused_report.json", depth_fused)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, default=Path("."))
    parser.add_argument("--sam2-track-json", type=Path, default=None)
    parser.add_argument("--depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-annotations", type=Path, default=None)
    parser.add_argument("--camera-npz", type=Path, default=None)
    parser.add_argument("--calibration-contract", type=Path, default=None, help="V19 camera calibration contract JSON. When supplied, its constant intrinsics override base/depth per-frame intrinsics for mask-depth lifting.")
    parser.add_argument("--object-plan", type=Path, default=None)
    parser.add_argument("--remote-root", type=Path, default=None, help="Remote path prefix to localize mask/raw paths from server-produced manifests")
    parser.add_argument("--local-root", type=Path, default=None, help="Local path prefix corresponding to --remote-root")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--anchor-frame", type=int, default=None)
    parser.add_argument("--propose-anchor-candidates-only", action="store_true", help="Lift visible mask/depth rows, write anchor_candidate_proposals.json and anchor_candidate_review.jpg, then stop before exporting canonical anchor geometry. The agent must inspect and choose an anchor.")
    parser.add_argument("--require-anchor-frame", action="store_true", help="Fail instead of falling back to the max-point frame when --anchor-frame is absent. Use after candidate proposal so anchor choice is explicit.")
    parser.add_argument("--anchor-candidate-count", type=int, default=12, help="Number of diversified candidate frames to show in the visual anchor review sheet.")
    parser.add_argument("--anchor-candidate-min-gap", type=int, default=8, help="Minimum frame gap used when diversifying review-sheet anchor candidates; ranked JSON still contains every candidate.")
    parser.add_argument("--anchor-candidate-panel-width", type=int, default=360, help="Width in pixels for each candidate panel in anchor_candidate_review.jpg.")
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--min-valid-points", type=int, default=50)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--allow-camera-frame-world", action="store_true", help="Explicitly use each camera frame as its own world frame when no world camera pose is available. This is not valid for temporal metric world claims.")
    parser.add_argument("--carry-invisible-pose", action="store_true", help="Carry the nearest visible centroid pose into invisible frames as an uncertain initialization only.")
    parser.add_argument("--preserve-source-index", action=argparse.BooleanOptionalAction, default=True, help="Write one output frame row per raw source frame so annotations['frames'][frame_idx] remains valid for V18 rigid tools.")
    parser.add_argument("--exclude-hand-bboxes", action=argparse.BooleanOptionalAction, default=True, help="Subtract same-frame hand support boxes from object masks before lifting visible object depth; hand-owned pixels remain occlusion/uncertainty, not object surface.")
    parser.add_argument("--hand-bbox-exclusion-pad-px", type=int, default=12, help="Padding, in mask/depth pixels, around projected hand boxes removed from object visible-surface support.")
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--anchor-mesh-min-voxel-m", type=float, default=0.002)
    parser.add_argument("--anchor-mesh-voxel-divisor", type=float, default=80.0)
    parser.add_argument("--anchor-mesh-poisson-depth", type=int, default=7)
    parser.add_argument("--anchor-mesh-poisson-density-quantile", type=float, default=0.02)
    parser.add_argument("--rigid-extent-ratio-max", type=float, default=2.75, help="Mark visible surfaces with diagonal extent more than this multiple of the selected anchor as ineligible for rigid pose fitting; they remain mask/depth measurements with systematic leakage uncertainty.")
    parser.add_argument("--rigid-extent-axis-ratio-max", type=float, default=3.25, help="Axis-wise companion to --rigid-extent-ratio-max for detecting elongated hand/background leakage.")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "skipped_rows_preview"}}, indent=2))


if __name__ == "__main__":
    main()
