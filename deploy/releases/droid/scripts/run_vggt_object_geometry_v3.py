#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def import_vggt(repo_root: Path) -> tuple[object, object, object]:
    vggt_root = repo_root / "third_party" / "vggt"
    if not vggt_root.exists():
        raise RuntimeError(f"VGGT checkout missing: {vggt_root}")
    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    return VGGT, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map


def require_open3d():
    import open3d as o3d

    return o3d


def frame_map(annotations: Path) -> dict[int, dict]:
    frames = load_json(annotations)["frames"]
    out: dict[int, dict] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx in out:
            raise RuntimeError(f"duplicate annotation frame {frame_idx}")
        out[frame_idx] = frame
    return out


def read_manifest(path: Path, frame_start: int, frame_end: int) -> list[dict]:
    payload = load_json(path)
    rows = []
    for row in payload["frame_reports"]:
        frame_idx = int(row["frame_idx"])
        if frame_start <= frame_idx <= frame_end:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no VGGT view rows in requested interval {frame_start}:{frame_end}")
    rows.sort(key=lambda item: int(item["frame_idx"]))
    expected = np.arange(rows[0]["frame_idx"], rows[-1]["frame_idx"] + 1, dtype=int)
    actual = np.asarray([int(row["frame_idx"]) for row in rows], dtype=int)
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"VGGT view rows are not contiguous: {actual.tolist()}")
    return rows


def localize_manifest_paths(rows: list[dict], remote_root: Path | None, local_root: Path | None) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("image_path", "mask_path"):
            path = Path(str(item[key]))
            if not path.exists() and remote_root is not None and local_root is not None:
                try:
                    rel = path.relative_to(local_root)
                except ValueError:
                    rel = None
                if rel is not None:
                    path = remote_root / rel
            if not path.exists():
                raise FileNotFoundError(f"manifest {key} does not exist: {path}")
            item[key] = str(path)
        out.append(item)
    return out


def preprocess_image_and_mask(image_path: Path, mask_path: Path, target_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
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
    image_pad = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    mask_pad = np.zeros((target_size, target_size), dtype=np.uint8)
    image_pad[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = image_small
    mask_pad[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = np.where(mask_small > 0, 255, 0).astype(np.uint8)
    tensor = torch.from_numpy(image_pad.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor, mask_pad, image_pad


def load_views(rows: list[dict], target_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    images = []
    masks = []
    rgbs = []
    for row in rows:
        image, mask, rgb = preprocess_image_and_mask(Path(row["image_path"]), Path(row["mask_path"]), target_size)
        images.append(image)
        masks.append(mask)
        rgbs.append(rgb)
    return torch.stack(images, dim=0), np.stack(masks, axis=0), np.stack(rgbs, axis=0)


def camera_centers_from_vggt(extrinsic: np.ndarray) -> np.ndarray:
    centers = []
    for row in extrinsic:
        R = row[:3, :3]
        t = row[:3, 3]
        centers.append(-R.T @ t)
    return np.asarray(centers, dtype=float)


def camera_centers_from_annotations(frame_by_idx: dict[int, dict], frame_indices: list[int]) -> np.ndarray:
    centers = []
    for frame_idx in frame_indices:
        frame = frame_by_idx[int(frame_idx)]
        T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"frame {frame_idx} has invalid T_world_camera_metric")
        centers.append(T[:3, 3])
    return np.asarray(centers, dtype=float)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise RuntimeError(f"invalid Sim3 inputs: {src.shape}, {dst.shape}")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    var_src = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if var_src <= 1e-12:
        raise RuntimeError("source camera centers are degenerate")
    cov = (dst_centered.T @ src_centered) / len(src)
    U, singular, Vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=float)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1.0
    R = U @ np.diag(sign) @ Vt
    scale = float(np.sum(singular * sign) / var_src)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid Sim3 scale: {scale}")
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def apply_sim3(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (float(scale) * (np.asarray(points, dtype=float) @ R.T)) + np.asarray(t, dtype=float)[None, :]


def run_vggt(args: argparse.Namespace, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT object geometry requires CUDA")
    VGGT, pose_encoding_to_extri_intri, unproject_depth_map_to_point_map = import_vggt(args.repo_root)
    device = torch.device(f"cuda:{int(args.gpu)}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model = VGGT()
    checkpoint_url = f"https://huggingface.co/{args.model_id}/resolve/main/{args.model_file}"
    state = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth = predictions["depth"].squeeze(0).float().cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).float().cpu().numpy()
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extrinsic_np, intrinsic_np)
    if depth.ndim == 4:
        depth = depth[..., 0]
    return extrinsic_np, intrinsic_np, depth, depth_conf, points


def select_object_points(
    frame_indices: list[int],
    points_vggt: np.ndarray,
    depth: np.ndarray,
    depth_conf: np.ndarray,
    masks: np.ndarray,
    rgbs: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(int(args.seed))
    points = []
    colors = []
    reports = []
    for i, frame_idx in enumerate(frame_indices):
        mask = masks[i] > 0
        frame_points = points_vggt[i]
        frame_depth = depth[i]
        frame_conf = depth_conf[i]
        valid = mask & np.isfinite(frame_points).all(axis=2) & np.isfinite(frame_depth) & (frame_depth > 1e-5) & np.isfinite(frame_conf)
        object_conf = frame_conf[mask & np.isfinite(frame_conf)]
        if object_conf.size == 0:
            raise RuntimeError(f"frame {frame_idx} has no finite VGGT confidence on object mask")
        threshold = max(float(args.min_depth_conf), float(np.quantile(object_conf, float(args.conf_quantile))))
        valid &= frame_conf >= threshold
        valid_count = int(valid.sum())
        if valid_count < int(args.min_points_per_frame):
            raise RuntimeError(f"frame {frame_idx} has only {valid_count} valid VGGT object points")
        flat_points = frame_points[valid]
        flat_colors = rgbs[i][valid]
        if len(flat_points) > int(args.max_points_per_frame):
            chosen = rng.choice(len(flat_points), size=int(args.max_points_per_frame), replace=False)
            flat_points = flat_points[chosen]
            flat_colors = flat_colors[chosen]
        points.append(flat_points.astype(np.float32))
        colors.append(flat_colors.astype(np.uint8))
        reports.append(
            {
                "frame_idx": int(frame_idx),
                "mask_pixels": int(mask.sum()),
                "confidence_threshold": float(threshold),
                "valid_points_before_sampling": valid_count,
                "sampled_points": int(len(flat_points)),
            }
        )
    return np.vstack(points), np.vstack(colors), reports


def to_pcd(points: np.ndarray, colors: np.ndarray | None, voxel_size: float, normal_radius: float, normal_max_nn: int) -> o3d.geometry.PointCloud:
    o3d = require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float) / 255.0)
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    if len(pcd.points) == 0:
        raise RuntimeError("point cloud is empty after downsampling")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_max_nn)))
    return pcd


def mesh_from_points(points: np.ndarray, colors: np.ndarray, args: argparse.Namespace) -> tuple[trimesh.Trimesh, o3d.geometry.PointCloud]:
    o3d = require_open3d()
    pcd = to_pcd(points, colors, args.voxel_size_m, args.normal_radius_m, args.normal_max_nn)
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=int(args.outlier_neighbors), std_ratio=float(args.outlier_std_ratio))
    if len(clean.points) >= int(args.min_fused_points):
        pcd = clean
    if args.orient_normals:
        pcd.orient_normals_consistent_tangent_plane(int(args.normal_orientation_k))
    radii = [float(args.voxel_size_m) * float(scale) for scale in args.bpa_radius_scales]
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float32)
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)
    if len(vertices) < int(args.min_mesh_vertices) or len(faces) < int(args.min_mesh_faces):
        raise RuntimeError(f"VGGT BPA mesh underconstrained: vertices={len(vertices)} faces={len(faces)}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), pcd


def load_mesh_archive_points(path: Path, frame_indices: list[int], max_points: int, seed: int) -> np.ndarray:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "vertices"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"observed mesh archive missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    pieces = []
    for frame_idx in frame_indices:
        hits = np.where(frames == int(frame_idx))[0]
        if len(hits) != 1:
            raise RuntimeError(f"observed mesh archive lacks frame {frame_idx}")
        i = int(hits[0])
        pieces.append(vertices[int(offsets[i]) : int(offsets[i + 1])])
    points = np.vstack(pieces)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < int(max_points * 0.05):
        raise RuntimeError("observed mesh points are underconstrained")
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def distance_summary(source: np.ndarray, target: np.ndarray) -> dict:
    tree = cKDTree(np.asarray(target, dtype=float))
    dists = tree.query(np.asarray(source, dtype=float), k=1)[0]
    return {
        "median_m": float(np.median(dists)),
        "p95_m": float(np.percentile(dists, 95)),
        "max_m": float(np.max(dists)),
    }


def hand_points(frame: dict) -> np.ndarray:
    chunks = []
    for hand in frame.get("hands", []):
        for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
            if key in hand:
                arr = np.asarray(hand[key], dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    chunks.append(arr)
                break
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    points = np.vstack(chunks)
    return points[np.isfinite(points).all(axis=1)]


def hand_mesh_summary(mesh_vertices: np.ndarray, frame_by_idx: dict[int, dict], frame_indices: list[int]) -> dict:
    tree = cKDTree(np.asarray(mesh_vertices, dtype=float))
    rows = []
    for frame_idx in frame_indices:
        points = hand_points(frame_by_idx[int(frame_idx)])
        if len(points) == 0:
            continue
        if len(points) > 1800:
            points = points[np.linspace(0, len(points) - 1, 1800, dtype=int)]
        min_dist = float(np.min(tree.query(points, k=1)[0]))
        rows.append({"frame_idx": int(frame_idx), "min_hand_mesh_distance_m": min_dist})
    values = np.asarray([row["min_hand_mesh_distance_m"] for row in rows], dtype=float)
    return {
        "frames": rows,
        "median_m": float(np.median(values)) if len(values) else None,
        "p95_m": float(np.percentile(values, 95)) if len(values) else None,
        "min_m": float(np.min(values)) if len(values) else None,
    }


def save_mesh_archive(path: Path, frame_indices: list[int], mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for _ in frame_indices:
        vertices_all.append(vertices)
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def export_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    cloud = trimesh.points.PointCloud(points, colors=colors)
    cloud.export(path)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.views_dir / "vggt_object_views_manifest.json"
    rows = read_manifest(manifest_path, int(args.frame_start), int(args.frame_end))
    rows = localize_manifest_paths(rows, args.remote_output_root, args.local_output_root)
    frame_indices = [int(row["frame_idx"]) for row in rows]
    frame_by_idx = frame_map(args.annotations)
    missing = [idx for idx in frame_indices if idx not in frame_by_idx]
    if missing:
        raise RuntimeError(f"annotation frames missing for VGGT rows: {missing}")

    images, masks, rgbs = load_views(rows, int(args.target_size))
    extrinsic, intrinsic, depth, depth_conf, points_vggt = run_vggt(args, images)
    vggt_centers = camera_centers_from_vggt(extrinsic)
    target_centers = camera_centers_from_annotations(frame_by_idx, frame_indices)
    scale, R, t = umeyama_similarity(vggt_centers, target_centers)
    aligned_centers = apply_sim3(vggt_centers, scale, R, t)
    center_error = np.linalg.norm(aligned_centers - target_centers, axis=1)
    if float(np.median(center_error)) > float(args.max_camera_align_median_m):
        raise RuntimeError(f"VGGT camera Sim3 median error is too high: {float(np.median(center_error)):.4f}m")

    object_points_vggt, object_colors, point_reports = select_object_points(
        frame_indices, points_vggt, depth, depth_conf, masks, rgbs, args
    )
    object_points_metric = apply_sim3(object_points_vggt, scale, R, t)
    extent = object_points_metric.max(axis=0) - object_points_metric.min(axis=0)
    if float(np.max(extent)) > float(args.max_object_extent_m):
        raise RuntimeError(f"VGGT object point extent is implausible after camera Sim3: {extent.tolist()}")
    mesh, pcd = mesh_from_points(object_points_metric, object_colors, args)
    mesh_extent = np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(axis=0)
    if float(np.max(mesh_extent)) > float(args.max_mesh_extent_m):
        raise RuntimeError(f"VGGT object mesh extent is implausible: {mesh_extent.tolist()}")

    observed_points = load_mesh_archive_points(args.observed_mesh_npz, frame_indices, int(args.max_observed_points), int(args.seed) + 91)
    pcd_points = np.asarray(pcd.points, dtype=float)
    dist_vggt_to_observed = distance_summary(pcd_points, observed_points)
    dist_observed_to_vggt = distance_summary(observed_points, pcd_points)
    hand_summary = hand_mesh_summary(np.asarray(mesh.vertices, dtype=float), frame_by_idx, frame_indices)

    canonical_points_path = args.output_dir / "vggt_object_points_metric.ply"
    export_point_cloud(canonical_points_path, pcd_points, (np.asarray(pcd.colors) * 255.0).astype(np.uint8) if len(pcd.colors) else None)
    mesh_path = args.output_dir / "vggt_object_mesh_metric.obj"
    mesh.export(mesh_path)
    archive_path = args.output_dir / "vggt_object_meshes.npz"
    save_mesh_archive(archive_path, frame_indices, mesh)
    np.savez_compressed(
        args.output_dir / "vggt_predictions_object_geometry.npz",
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        masks=masks.astype(np.uint8),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
        camera_centers_vggt=vggt_centers.astype(np.float32),
        camera_centers_metric=target_centers.astype(np.float32),
        sim3_scale=np.asarray([scale], dtype=np.float32),
        sim3_rotation=R.astype(np.float32),
        sim3_translation=t.astype(np.float32),
    )
    report = {
        "status": "ok",
        "method": "vggt_masked_multiview_object_geometry",
        "views_manifest": str(manifest_path),
        "annotations": str(args.annotations),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": frame_indices,
        "model_id": str(args.model_id),
        "target_size": int(args.target_size),
        "sim3_scale": float(scale),
        "camera_alignment_median_m": float(np.median(center_error)),
        "camera_alignment_p95_m": float(np.percentile(center_error, 95)),
        "camera_alignment_max_m": float(np.max(center_error)),
        "object_points_sampled": int(len(object_points_metric)),
        "object_point_extent_m": extent.astype(float).tolist(),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_m": mesh_extent.astype(float).tolist(),
        "watertight": bool(mesh.is_watertight),
        "penetration_supported": bool(mesh.is_watertight),
        "vggt_to_observed_distance": dist_vggt_to_observed,
        "observed_to_vggt_distance": dist_observed_to_vggt,
        "hand_mesh_distance": hand_summary,
        "point_reports": point_reports,
        "outputs": {
            "point_cloud": str(canonical_points_path),
            "mesh": str(mesh_path),
            "mesh_archive": str(archive_path),
            "predictions": str(args.output_dir / "vggt_predictions_object_geometry.npz"),
        },
        "parameters": {
            "conf_quantile": float(args.conf_quantile),
            "min_depth_conf": float(args.min_depth_conf),
            "max_points_per_frame": int(args.max_points_per_frame),
            "voxel_size_m": float(args.voxel_size_m),
            "bpa_radius_scales": [float(v) for v in args.bpa_radius_scales],
        },
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_vggt_object_geometry_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"point_reports", "hand_mesh_distance"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
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
    parser.add_argument("--max-points-per-frame", type=int, default=7000)
    parser.add_argument("--voxel-size-m", type=float, default=0.006)
    parser.add_argument("--normal-radius-m", type=float, default=0.030)
    parser.add_argument("--normal-max-nn", type=int, default=35)
    parser.add_argument("--orient-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-orientation-k", type=int, default=28)
    parser.add_argument("--outlier-neighbors", type=int, default=24)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.2)
    parser.add_argument("--min-fused-points", type=int, default=1800)
    parser.add_argument("--min-mesh-vertices", type=int, default=500)
    parser.add_argument("--min-mesh-faces", type=int, default=700)
    parser.add_argument("--bpa-radius-scales", type=float, nargs="+", default=[1.5, 2.5, 4.0, 6.0])
    parser.add_argument("--max-observed-points", type=int, default=50000)
    parser.add_argument("--max-object-extent-m", type=float, default=0.90)
    parser.add_argument("--max-mesh-extent-m", type=float, default=0.90)
    parser.add_argument("--max-camera-align-median-m", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
