#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, load_depth_archive, manifest_by_frame
from optimize_mesh_prior_pose_graph_v3 import load_mesh, sample_mesh_surface, sample_rows
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {path}")
    if mask.shape != expected_shape:
        mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def intrinsics_for(annotation: dict, depth_intrinsics: np.ndarray, source: str) -> np.ndarray:
    if source == "annotation-vggt":
        intrinsics = np.asarray(annotation.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
    elif source == "metric-depth":
        intrinsics = np.asarray(depth_intrinsics, dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported intrinsics source {source}")
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise RuntimeError(f"invalid {source} intrinsics for frame {annotation.get('frame_idx')}")
    return intrinsics


def mask_depth_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int,
    seed: int,
    min_depth_m: float,
) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m > float(min_depth_m))
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError("mask has no valid metric-depth pixels")
    coords = np.c_[xs, ys]
    if len(coords) > int(max_points):
        rng = np.random.default_rng(int(seed))
        coords = coords[rng.choice(len(coords), size=int(max_points), replace=False)]
    z = depth_m[coords[:, 1], coords[:, 0]].astype(np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    x = (coords[:, 0].astype(np.float64) - cx) * z / fx
    y = (coords[:, 1].astype(np.float64) - cy) * z / fy
    points = np.c_[x, y, z]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise RuntimeError("sampled mask-depth points are invalid")
    return points.astype(np.float64)


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise RuntimeError("rigid transform inputs must be matching Nx3 arrays")
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    src0 = source - src_center
    tgt0 = target - tgt_center
    u, _s, vh = np.linalg.svd(src0.T @ tgt0)
    r = vh.T @ u.T
    if np.linalg.det(r) < 0.0:
        vh[-1, :] *= -1.0
        r = vh.T @ u.T
    t = tgt_center - src_center @ r.T
    err = np.linalg.norm(source @ r.T + t - target, axis=1)
    return r.astype(np.float64), t.astype(np.float64), float(np.median(err))


def compute_frame_pose(
    prior_vertices: np.ndarray,
    graph_vertices_world: np.ndarray,
    annotation: dict,
    max_correspondences: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(graph_vertices_world) != len(prior_vertices):
        raise RuntimeError("graph mesh and prior mesh do not share vertex count")
    T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
    graph_vertices_camera = camera_points(graph_vertices_world, T_world_camera)
    indices = np.arange(len(prior_vertices), dtype=np.int64)
    if len(indices) > int(max_correspondences):
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(indices, size=int(max_correspondences), replace=False)
    r, t, median_error = rigid_transform(prior_vertices[indices], graph_vertices_camera[indices])
    if median_error > 1e-4:
        raise RuntimeError(f"recovered graph pose has {median_error:.6f} m median correspondence error")
    return r, t, {"pose_recovery_median_error_m": median_error, "pose_correspondences": int(len(indices))}


def camera_to_prior(points_camera: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points_camera - t[None, :]) @ r


def prior_to_camera(points_prior: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points_prior @ r.T + t[None, :]


def to_point_cloud(points: np.ndarray, voxel_size: float, normal_radius: float, normal_max_nn: int) -> o3d.geometry.PointCloud:
    if len(points) == 0:
        raise RuntimeError("cannot build point cloud from zero points")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(float(voxel_size))
    if len(pcd.points) == 0:
        raise RuntimeError("point cloud is empty after voxel downsample")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_max_nn)))
    return pcd


def reconstruct_poisson_mesh(points: np.ndarray, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    pcd = to_point_cloud(points, float(args.voxel_size_m), float(args.normal_radius_m), int(args.normal_max_nn))
    pcd.orient_normals_consistent_tangent_plane(int(args.normal_orientation_k))
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(args.poisson_depth))
    densities_arr = np.asarray(densities, dtype=np.float64)
    if len(densities_arr) != len(mesh_o3d.vertices):
        raise RuntimeError("Poisson density length mismatch")
    if len(densities_arr) == 0:
        raise RuntimeError("Poisson reconstruction returned an empty mesh")
    density_threshold = float(np.quantile(densities_arr, float(args.poisson_density_trim_quantile)))
    mesh_o3d.remove_vertices_by_mask(densities_arr < density_threshold)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()
    vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64)
    faces = np.asarray(mesh_o3d.triangles, dtype=np.int32)
    if len(vertices) < int(args.min_mesh_vertices) or len(faces) < int(args.min_mesh_faces):
        raise RuntimeError(f"fused mesh underconstrained: vertices={len(vertices)} faces={len(faces)}")
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), process=True)
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        raise RuntimeError("fused mesh has no connected components")
    mesh = max(components, key=lambda comp: float(comp.area))
    mesh.remove_unreferenced_vertices()
    extent = mesh.extents.astype(np.float64)
    if not np.isfinite(extent).all() or np.any(extent <= 0.0):
        raise RuntimeError("fused mesh extent is invalid")
    if float(np.max(extent)) > float(args.max_mesh_extent_m):
        raise RuntimeError(f"fused mesh extent is implausible: {extent.tolist()}")
    return mesh, {
        "input_points": int(len(points)),
        "pcd_points_after_voxel": int(len(pcd.points)),
        "poisson_density_trim_quantile": float(args.poisson_density_trim_quantile),
        "poisson_density_threshold": density_threshold,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_m": extent.astype(float).tolist(),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
    }


def save_camera_obj(path: Path, mesh: trimesh.Trimesh, r: np.ndarray, t: np.ndarray) -> None:
    vertices = prior_to_camera(np.asarray(mesh.vertices, dtype=np.float64), r, t)
    out = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.graph_mesh_archive)
    prior_mesh = load_mesh(args.mesh_prior_camera)
    prior_vertices = np.asarray(prior_mesh.vertices, dtype=np.float64)
    prior_sample = sample_mesh_surface(prior_mesh, int(args.max_prior_points), int(args.seed) + 800)
    observed_prior_points = []
    frame_rows = []
    anchor_pose: tuple[np.ndarray, np.ndarray] | None = None
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in manifest or frame_idx not in depths or frame_idx not in graph_meshes:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        r, t, pose_row = compute_frame_pose(
            prior_vertices,
            graph_meshes[frame_idx][0],
            annotation,
            int(args.max_pose_correspondences),
            int(args.seed) + frame_idx,
        )
        if int(frame_idx) == int(args.anchor_frame):
            anchor_pose = (r, t)
        camera_points_frame = mask_depth_points(
            mask,
            depth_m,
            intrinsics,
            int(args.max_observed_points_per_frame),
            int(args.seed) + 2000 + frame_idx,
            float(args.min_depth_m),
        )
        prior_points_frame = camera_to_prior(camera_points_frame, r, t)
        observed_prior_points.append(prior_points_frame)
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "observed_points": int(len(prior_points_frame)),
                "observed_prior_extent_m": (prior_points_frame.max(axis=0) - prior_points_frame.min(axis=0)).astype(float).tolist(),
                "object_translation_camera_m": t.astype(float).tolist(),
                "object_rotation_delta_rad": Rotation.from_matrix(r).as_rotvec().astype(float).tolist(),
                **pose_row,
            }
        )
    if len(observed_prior_points) < int(args.min_frames):
        raise RuntimeError(f"only {len(observed_prior_points)} usable frames for fusion")
    if anchor_pose is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} was not available")
    observed = np.vstack(observed_prior_points)
    observed_sample = sample_rows(observed, int(args.max_observed_fusion_points), int(args.seed) + 3000)
    prior_weight = max(0.0, float(args.prior_weight))
    if prior_weight > 0.0:
        prior_count = max(1, int(round(len(prior_sample) * prior_weight)))
        prior_points = sample_rows(prior_sample, prior_count, int(args.seed) + 4000)
        fusion_points = np.vstack([observed_sample, prior_points])
    else:
        fusion_points = observed_sample
    mesh_prior, mesh_report = reconstruct_poisson_mesh(fusion_points, args)
    r_anchor, t_anchor = anchor_pose
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_mesh_path = args.output_dir / "observed_prior_fused_mesh_anchor_camera.obj"
    save_camera_obj(anchor_mesh_path, mesh_prior, r_anchor, t_anchor)
    prior_mesh_path = args.output_dir / "observed_prior_fused_mesh_prior_frame.obj"
    mesh_prior.export(prior_mesh_path)
    observed_extent = observed_sample.max(axis=0) - observed_sample.min(axis=0)
    fusion_extent = fusion_points.max(axis=0) - fusion_points.min(axis=0)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "fuse_observed_surface_with_complete_prior_v3",
        "claim_tested": "visible SAMWISE plus UniDepth surface can repair the complete prior missing rim/flange geometry",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "graph_mesh_archive": str(args.graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "anchor_frame": int(args.anchor_frame),
        "frames": [row["frame_idx"] for row in frame_rows],
        "prior_frame_mesh": str(prior_mesh_path),
        "anchor_camera_mesh": str(anchor_mesh_path),
        "observed_sample_points": int(len(observed_sample)),
        "fusion_points": int(len(fusion_points)),
        "observed_sample_extent_m": observed_extent.astype(float).tolist(),
        "fusion_points_extent_m": fusion_extent.astype(float).tolist(),
        "frame_rows": frame_rows,
        "mesh": mesh_report,
        "parameters": {
            "intrinsics_source": str(args.intrinsics_source),
            "prior_weight": float(args.prior_weight),
            "max_observed_points_per_frame": int(args.max_observed_points_per_frame),
            "max_observed_fusion_points": int(args.max_observed_fusion_points),
            "max_prior_points": int(args.max_prior_points),
            "voxel_size_m": float(args.voxel_size_m),
            "poisson_depth": int(args.poisson_depth),
        },
    }
    report_path = args.output_dir / "qc_observed_prior_fused_mesh_v3.json"
    save_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "frame_rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--graph-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--max-observed-points-per-frame", type=int, default=16000)
    parser.add_argument("--max-observed-fusion-points", type=int, default=70000)
    parser.add_argument("--max-prior-points", type=int, default=30000)
    parser.add_argument("--prior-weight", type=float, default=0.6)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--voxel-size-m", type=float, default=0.003)
    parser.add_argument("--normal-radius-m", type=float, default=0.025)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    parser.add_argument("--normal-orientation-k", type=int, default=32)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-density-trim-quantile", type=float, default=0.015)
    parser.add_argument("--min-mesh-vertices", type=int, default=500)
    parser.add_argument("--min-mesh-faces", type=int, default=1000)
    parser.add_argument("--max-mesh-extent-m", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=991)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
