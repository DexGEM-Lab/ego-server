#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from align_mesh_prior_v3 import load_observed_frame


def sample_rows(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (np.asarray(transform, dtype=float) @ homog.T).T[:, :3]


def to_pcd(points: np.ndarray, voxel_size: float, normal_radius: float, max_nn: int) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    if len(pcd.points) == 0:
        raise RuntimeError("point cloud is empty after voxel downsample")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(max_nn))
    )
    return pcd


def merged_cloud(
    frame_indices: list[int],
    frame_points: dict[int, np.ndarray],
    poses_to_anchor: dict[int, np.ndarray],
    voxel_size: float,
    normal_radius: float,
    normal_max_nn: int,
) -> o3d.geometry.PointCloud:
    points = []
    for idx in frame_indices:
        points.append(transform_points(frame_points[idx], poses_to_anchor[idx]))
    return to_pcd(np.vstack(points), voxel_size, normal_radius, normal_max_nn)


def register_to_target(
    source_points: np.ndarray,
    target: o3d.geometry.PointCloud,
    init: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    source = to_pcd(source_points, args.voxel_size, args.normal_radius, args.normal_max_nn)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(args.icp_threshold_m),
        np.asarray(init, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(args.icp_iterations),
            relative_fitness=float(args.icp_relative_fitness),
            relative_rmse=float(args.icp_relative_rmse),
        ),
    )
    report = {
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
    }
    if report["correspondence_count"] < int(args.min_icp_correspondences) or report["fitness"] < float(args.min_icp_fitness):
        report["accepted"] = False
        return np.asarray(init, dtype=float), report
    delta = np.asarray(result.transformation, dtype=float) @ np.linalg.inv(np.asarray(init, dtype=float))
    shift = float(np.linalg.norm(delta[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))
    report["delta_translation_m"] = shift
    report["delta_rotation_rad"] = angle
    if shift > float(args.max_update_translation_m) or angle > float(args.max_update_rotation_rad):
        report["accepted"] = False
        report["reject_reason"] = "update_exceeds_motion_bound"
        return np.asarray(init, dtype=float), report
    report["accepted"] = True
    return np.asarray(result.transformation, dtype=float), report


def sequential_poses(
    frame_indices: list[int],
    frame_points: dict[int, np.ndarray],
    anchor_frame: int,
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    poses = {int(anchor_frame): np.eye(4, dtype=float)}
    reports: list[dict] = []
    target_indices = [int(anchor_frame)]
    target = merged_cloud(target_indices, frame_points, poses, args.voxel_size, args.normal_radius, args.normal_max_nn)
    forward = [idx for idx in frame_indices if idx > anchor_frame]
    for idx in forward:
        prev = max(k for k in poses if k < idx)
        pose, report = register_to_target(frame_points[idx], target, poses[prev], args)
        poses[idx] = pose
        report.update({"frame_idx": int(idx), "pass": "forward", "init_frame": int(prev)})
        reports.append(report)
        target_indices.append(idx)
        target = merged_cloud(target_indices, frame_points, poses, args.voxel_size, args.normal_radius, args.normal_max_nn)
    backward = [idx for idx in reversed(frame_indices) if idx < anchor_frame]
    for idx in backward:
        next_idx = min(k for k in poses if k > idx)
        pose, report = register_to_target(frame_points[idx], target, poses[next_idx], args)
        poses[idx] = pose
        report.update({"frame_idx": int(idx), "pass": "backward", "init_frame": int(next_idx)})
        reports.append(report)
        target_indices.append(idx)
        target = merged_cloud(target_indices, frame_points, poses, args.voxel_size, args.normal_radius, args.normal_max_nn)
    return poses, reports


def refine_poses(
    frame_indices: list[int],
    frame_points: dict[int, np.ndarray],
    poses: dict[int, np.ndarray],
    anchor_frame: int,
    args: argparse.Namespace,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    reports: list[dict] = []
    for iteration in range(int(args.refine_passes)):
        target = merged_cloud(frame_indices, frame_points, poses, args.voxel_size, args.normal_radius, args.normal_max_nn)
        next_poses = dict(poses)
        for idx in frame_indices:
            if idx == anchor_frame and args.lock_anchor:
                continue
            pose, report = register_to_target(frame_points[idx], target, poses[idx], args)
            next_poses[idx] = pose
            report.update({"frame_idx": int(idx), "pass": "refine", "iteration": int(iteration)})
            reports.append(report)
        poses = next_poses
    return poses, reports


def reconstruct_bpa_mesh(pcd: o3d.geometry.PointCloud, args: argparse.Namespace) -> trimesh.Trimesh:
    if args.orient_normals:
        pcd.orient_normals_consistent_tangent_plane(int(args.normal_orientation_k))
    radii = [float(args.voxel_size) * float(scale) for scale in args.bpa_radius_scales]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) < args.min_mesh_vertices or len(faces) < args.min_mesh_faces:
        raise RuntimeError(f"BPA mesh underconstrained: vertices={len(vertices)} faces={len(faces)}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def save_mesh_archive(path: Path, frame_indices: list[int], canonical_mesh: trimesh.Trimesh, poses_to_anchor: dict[int, np.ndarray]) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    base_vertices = np.asarray(canonical_mesh.vertices, dtype=float)
    faces = np.asarray(canonical_mesh.faces, dtype=np.int32)
    for idx in frame_indices:
        T_world_from_anchor = np.linalg.inv(poses_to_anchor[int(idx)])
        vertices = transform_points(base_vertices, T_world_from_anchor).astype(np.float32)
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


def summarize_points(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=float)
    return {
        "count": int(len(pts)),
        "extent_m": (pts.max(axis=0) - pts.min(axis=0)).astype(float).tolist(),
        "center": pts.mean(axis=0).astype(float).tolist(),
    }


def extent_report(points: np.ndarray, max_extent_m: float) -> dict:
    pts = np.asarray(points, dtype=float)
    extent = pts.max(axis=0) - pts.min(axis=0)
    return {
        "extent_m": extent.astype(float).tolist(),
        "max_extent_m": float(np.max(extent)),
        "passes": bool(float(np.max(extent)) <= float(max_extent_m)),
    }


def load_frame_points(args: argparse.Namespace) -> tuple[list[int], dict[int, np.ndarray], list[dict]]:
    blob = np.load(args.observed_mesh_npz)
    available = set(int(v) for v in blob["frame_idx"].astype(int).tolist())
    frame_indices = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in available]
    if args.anchor_frame not in frame_indices:
        raise RuntimeError(f"anchor frame {args.anchor_frame} is not present in observed mesh archive")
    if len(frame_indices) < 2:
        raise RuntimeError("too few observed mesh frames for multiview reconstruction")
    frame_points = {}
    reports = []
    for idx in frame_indices:
        vertices, _ = load_observed_frame(args.observed_mesh_npz, idx)
        sampled = sample_rows(vertices, int(args.max_points_per_frame), int(args.seed) + idx)
        frame_points[idx] = sampled
        reports.append({"frame_idx": int(idx), "raw_vertices": int(len(vertices)), "sampled_points": int(len(sampled))})
    return frame_indices, frame_points, reports


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices, frame_points, input_reports = load_frame_points(args)
    poses, seq_reports = sequential_poses(frame_indices, frame_points, int(args.anchor_frame), args)
    poses, refine_reports = refine_poses(frame_indices, frame_points, poses, int(args.anchor_frame), args)
    fused = merged_cloud(frame_indices, frame_points, poses, args.fusion_voxel_size, args.normal_radius, args.normal_max_nn)
    clean, keep = fused.remove_statistical_outlier(nb_neighbors=int(args.outlier_neighbors), std_ratio=float(args.outlier_std_ratio))
    if len(clean.points) >= int(args.min_fused_points):
        fused = clean
    canonical_points = np.asarray(fused.points, dtype=float)
    fused_extent = extent_report(canonical_points, args.max_canonical_extent_m)
    if not fused_extent["passes"]:
        raise RuntimeError(f"fused canonical object extent is implausible: {fused_extent}")
    mesh = reconstruct_bpa_mesh(fused, args)
    canonical_mesh_path = args.output_dir / "canonical_multiview_object_mesh.obj"
    mesh.export(canonical_mesh_path)
    archive_path = args.output_dir / "multiview_object_meshes.npz"
    save_mesh_archive(archive_path, frame_indices, mesh, poses)
    pose_payload = {
        str(idx): {
            "T_anchor_from_frame_world": poses[idx].astype(float).tolist(),
            "T_frame_world_from_anchor": np.linalg.inv(poses[idx]).astype(float).tolist(),
        }
        for idx in frame_indices
    }
    (args.output_dir / "object_poses_to_anchor.json").write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "method": "observed_metric_depth_multiview_icp_bpa",
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "anchor_frame": int(args.anchor_frame),
        "frames": frame_indices,
        "input_frames": input_reports,
        "sequential_registration": seq_reports,
        "refine_registration": refine_reports,
        "fused_point_cloud": summarize_points(canonical_points),
        "fused_extent_check": fused_extent,
        "fused_points_after_outlier_filter": int(len(canonical_points)),
        "canonical_mesh": str(canonical_mesh_path),
        "mesh_archive": str(archive_path),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_m": (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).astype(float).tolist(),
        "watertight": bool(mesh.is_watertight),
        "penetration_supported": bool(mesh.is_watertight),
        "penetration_note": "Signed penetration requires watertight mesh; non-watertight BPA meshes support observed-surface contact checks only.",
        "parameters": {
            "voxel_size": float(args.voxel_size),
            "fusion_voxel_size": float(args.fusion_voxel_size),
            "icp_threshold_m": float(args.icp_threshold_m),
            "icp_iterations": int(args.icp_iterations),
            "refine_passes": int(args.refine_passes),
            "bpa_radius_scales": [float(v) for v in args.bpa_radius_scales],
        },
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_multiview_object_mesh_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"sequential_registration", "refine_registration", "input_frames"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points-per-frame", type=int, default=2200)
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--fusion-voxel-size", type=float, default=0.004)
    parser.add_argument("--normal-radius", type=float, default=0.030)
    parser.add_argument("--normal-max-nn", type=int, default=35)
    parser.add_argument("--normal-orientation-k", type=int, default=24)
    parser.add_argument("--orient-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--icp-threshold-m", type=float, default=0.045)
    parser.add_argument("--icp-iterations", type=int, default=60)
    parser.add_argument("--icp-relative-fitness", type=float, default=1e-7)
    parser.add_argument("--icp-relative-rmse", type=float, default=1e-7)
    parser.add_argument("--min-icp-correspondences", type=int, default=80)
    parser.add_argument("--min-icp-fitness", type=float, default=0.05)
    parser.add_argument("--max-update-translation-m", type=float, default=0.080)
    parser.add_argument("--max-update-rotation-rad", type=float, default=0.35)
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--lock-anchor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--min-fused-points", type=int, default=500)
    parser.add_argument("--min-mesh-vertices", type=int, default=200)
    parser.add_argument("--min-mesh-faces", type=int, default=300)
    parser.add_argument("--max-canonical-extent-m", type=float, default=0.80)
    parser.add_argument("--bpa-radius-scales", type=float, nargs="+", default=[1.5, 2.5, 4.0, 6.0])
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
