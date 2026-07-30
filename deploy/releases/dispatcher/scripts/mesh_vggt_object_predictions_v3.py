#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def load_observed_points(path: Path, frame_indices: list[int], max_points: int, seed: int) -> np.ndarray:
    blob = np.load(path)
    frames = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    pieces = []
    for frame_idx in frame_indices:
        hits = np.where(frames == int(frame_idx))[0]
        if len(hits) != 1:
            raise RuntimeError(f"observed archive lacks frame {frame_idx}")
        i = int(hits[0])
        pieces.append(vertices[int(offsets[i]) : int(offsets[i + 1])])
    points = np.vstack(pieces)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("observed point set is empty")
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def robust_extent(points: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.percentile(points, hi, axis=0) - np.percentile(points, lo, axis=0)


def centered_scale(points: np.ndarray, target_center: np.ndarray, scale: float) -> np.ndarray:
    center = points.mean(axis=0)
    return (points - center[None, :]) * float(scale) + target_center[None, :]


def to_pcd(points: np.ndarray, colors: np.ndarray, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float) / 255.0)
    if args.voxel_size_m > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size_m))
    if len(pcd.points) == 0:
        raise RuntimeError("VGGT point cloud empty after voxel downsample")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.normal_radius_m), max_nn=int(args.normal_max_nn))
    )
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=int(args.outlier_neighbors), std_ratio=float(args.outlier_std_ratio))
    if len(clean.points) >= int(args.min_fused_points):
        pcd = clean
    return pcd


def mesh_from_pcd(pcd: o3d.geometry.PointCloud, args: argparse.Namespace) -> trimesh.Trimesh:
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
        raise RuntimeError(f"VGGT mesh underconstrained: vertices={len(vertices)} faces={len(faces)}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def save_mesh_archive(path: Path, frame_indices: list[int], mesh: trimesh.Trimesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_offsets = [0]
    face_offsets = [0]
    all_vertices = []
    all_faces = []
    for _ in frame_indices:
        all_vertices.append(vertices)
        all_faces.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(all_vertices).astype(np.float32),
        faces=np.vstack(all_faces).astype(np.int32),
    )


def distance_summary(source: np.ndarray, target: np.ndarray) -> dict:
    dists = cKDTree(target).query(source, k=1)[0]
    return {
        "median_m": float(np.median(dists)),
        "p95_m": float(np.percentile(dists, 95)),
        "max_m": float(np.max(dists)),
    }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(args.predictions)
    frame_indices = [int(v) for v in pred["frame_idx"].astype(int).tolist()]
    points_cam_sim3 = np.asarray(pred["object_points_metric"], dtype=float)
    colors = np.asarray(pred["object_colors"], dtype=np.uint8)
    scale_cam = float(pred["sim3_scale"][0])
    R = np.asarray(pred["sim3_rotation"], dtype=float)
    t = np.asarray(pred["sim3_translation"], dtype=float)
    points_vggt = ((points_cam_sim3 - t[None, :]) @ R) / scale_cam
    observed = load_observed_points(args.observed_mesh_npz, frame_indices, int(args.max_observed_points), int(args.seed) + 17)
    vggt_extent = robust_extent(points_vggt, args.extent_percentile_low, args.extent_percentile_high)
    observed_extent = robust_extent(observed, args.extent_percentile_low, args.extent_percentile_high)
    valid = (vggt_extent > 1e-6) & np.isfinite(vggt_extent) & np.isfinite(observed_extent)
    if int(valid.sum()) < 2:
        raise RuntimeError(f"insufficient extent axes for scale calibration: vggt={vggt_extent}, observed={observed_extent}")
    scale_object = float(np.median(observed_extent[valid] / vggt_extent[valid]))
    if not math.isfinite(scale_object) or scale_object <= 0.0:
        raise RuntimeError(f"invalid object scale: {scale_object}")
    points_metric = centered_scale(points_vggt, observed.mean(axis=0), scale_object)
    extent = points_metric.max(axis=0) - points_metric.min(axis=0)
    if float(np.max(extent)) > float(args.max_mesh_extent_m):
        raise RuntimeError(f"object-scaled VGGT point extent implausible: {extent.tolist()}")
    pcd = to_pcd(points_metric, colors, args)
    pcd_points = np.asarray(pcd.points, dtype=float)
    mesh = mesh_from_pcd(pcd, args)
    mesh_path = args.output_dir / "vggt_object_mesh_metric_object_scaled.obj"
    mesh.export(mesh_path)
    point_path = args.output_dir / "vggt_object_points_metric_object_scaled.ply"
    trimesh.points.PointCloud(pcd_points, colors=(np.asarray(pcd.colors) * 255.0).astype(np.uint8)).export(point_path)
    archive_path = args.output_dir / "vggt_object_meshes.npz"
    save_mesh_archive(archive_path, frame_indices, mesh)
    report = {
        "status": "ok",
        "method": "vggt_object_predictions_object_extent_scaled_bpa",
        "predictions": str(args.predictions),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frames": frame_indices,
        "camera_sim3_scale": scale_cam,
        "object_extent_scale": scale_object,
        "raw_vggt_extent": (points_vggt.max(axis=0) - points_vggt.min(axis=0)).astype(float).tolist(),
        "robust_vggt_extent": vggt_extent.astype(float).tolist(),
        "robust_observed_extent": observed_extent.astype(float).tolist(),
        "point_extent_m": extent.astype(float).tolist(),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_m": (np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(axis=0)).astype(float).tolist(),
        "watertight": bool(mesh.is_watertight),
        "penetration_supported": bool(mesh.is_watertight),
        "vggt_to_observed_distance": distance_summary(pcd_points, observed),
        "observed_to_vggt_distance": distance_summary(observed, pcd_points),
        "outputs": {
            "mesh": str(mesh_path),
            "points": str(point_path),
            "mesh_archive": str(archive_path),
        },
    }
    (args.output_dir / "qc_vggt_object_mesh_from_predictions_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-observed-points", type=int, default=50000)
    parser.add_argument("--extent-percentile-low", type=float, default=5.0)
    parser.add_argument("--extent-percentile-high", type=float, default=95.0)
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
    parser.add_argument("--max-mesh-extent-m", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
