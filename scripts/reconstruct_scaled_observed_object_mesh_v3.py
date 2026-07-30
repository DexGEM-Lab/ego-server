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
from reconstruct_multiview_object_mesh_v3 import (
    merged_cloud,
    reconstruct_bpa_mesh,
    refine_poses,
    sample_rows,
    sequential_poses,
    transform_points,
)


def robust_extent_xy(points: np.ndarray, quantile: float) -> np.ndarray:
    q = float(quantile)
    lo = np.quantile(points[:, :2], q, axis=0)
    hi = np.quantile(points[:, :2], 1.0 - q, axis=0)
    extent = hi - lo
    if not np.isfinite(extent).all() or np.any(extent <= 1e-5):
        raise RuntimeError("degenerate robust xy extent")
    return extent.astype(np.float64)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_frame_points(args: argparse.Namespace) -> tuple[list[int], dict[int, np.ndarray], list[dict]]:
    blob = np.load(args.observed_mesh_npz)
    available = set(int(v) for v in blob["frame_idx"].astype(int).tolist())
    frame_indices = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in available]
    if int(args.anchor_frame) not in frame_indices:
        raise RuntimeError(f"anchor frame {args.anchor_frame} is absent")
    if len(frame_indices) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_indices)} observed frames")

    raw_points: dict[int, np.ndarray] = {}
    rows = []
    extents = []
    for frame_idx in frame_indices:
        vertices, _ = load_observed_frame(args.observed_mesh_npz, frame_idx)
        vertices = sample_rows(vertices, int(args.max_points_per_frame), int(args.seed) + frame_idx)
        extent_xy = robust_extent_xy(vertices, float(args.extent_quantile))
        raw_points[frame_idx] = vertices
        extents.append(extent_xy)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "raw_points": int(len(vertices)),
                "raw_extent_xy_m": extent_xy.astype(float).tolist(),
                "raw_depth_median_m": float(np.median(vertices[:, 2])),
            }
        )

    if args.reference_extent_xy_m:
        reference_xy = np.asarray(args.reference_extent_xy_m, dtype=np.float64)
        if reference_xy.shape != (2,) or not np.isfinite(reference_xy).all() or np.any(reference_xy <= 0.0):
            raise RuntimeError("--reference-extent-xy-m must contain two positive finite values")
    elif args.reference == "anchor":
        reference_xy = robust_extent_xy(raw_points[int(args.anchor_frame)], float(args.extent_quantile))
    elif args.reference == "median":
        reference_xy = np.median(np.asarray(extents, dtype=np.float64), axis=0)
    else:
        raise RuntimeError(f"unsupported reference: {args.reference}")

    corrected: dict[int, np.ndarray] = {}
    for row in rows:
        frame_idx = int(row["frame_idx"])
        points = raw_points[frame_idx]
        extent_xy = np.asarray(row["raw_extent_xy_m"], dtype=np.float64)
        scale = float(np.median(reference_xy / extent_xy))
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"invalid scale for frame {frame_idx}: {scale}")
        if scale < float(args.min_depth_scale) or scale > float(args.max_depth_scale):
            raise RuntimeError(f"scale {scale:.4f} for frame {frame_idx} outside [{args.min_depth_scale}, {args.max_depth_scale}]")
        scaled = points * scale
        corrected[frame_idx] = scaled
        row["depth_scale"] = scale
        row["corrected_extent_xy_m"] = robust_extent_xy(scaled, float(args.extent_quantile)).astype(float).tolist()
        row["corrected_depth_median_m"] = float(np.median(scaled[:, 2]))
    return frame_indices, corrected, rows


def summarize_points(points: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float64)
    return {
        "count": int(len(pts)),
        "extent_m": (pts.max(axis=0) - pts.min(axis=0)).astype(float).tolist(),
        "robust_extent_xy_5_95_m": robust_extent_xy(pts, 0.05).astype(float).tolist(),
        "center_m": np.median(pts, axis=0).astype(float).tolist(),
    }


def frame_annotations(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def save_archives(
    output_dir: Path,
    frame_indices: list[int],
    mesh: trimesh.Trimesh,
    poses: dict[int, np.ndarray],
    input_rows: list[dict],
    annotations_path: Path | None,
) -> tuple[Path, Path | None]:
    base_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    depth_scales = {int(row["frame_idx"]): float(row["depth_scale"]) for row in input_rows}
    vertex_offsets = [0]
    face_offsets = [0]
    camera_vertices_all = []
    world_vertices_all = []
    faces_all = []
    annotations = frame_annotations(annotations_path) if annotations_path is not None else None
    for idx in frame_indices:
        camera_scaled = transform_points(base_vertices, np.linalg.inv(poses[int(idx)]))
        camera_unscaled = camera_scaled / depth_scales[int(idx)]
        camera_vertices_all.append(camera_unscaled.astype(np.float32))
        if annotations is not None:
            if int(idx) not in annotations:
                raise RuntimeError(f"annotations missing frame {idx}")
            T_world_camera = np.asarray(annotations[int(idx)]["camera"]["T_world_camera_metric"], dtype=np.float64)
            homog = np.c_[camera_unscaled, np.ones(len(camera_unscaled), dtype=np.float64)]
            world_vertices_all.append((T_world_camera @ homog.T).T[:, :3].astype(np.float32))
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(camera_unscaled))
        face_offsets.append(face_offsets[-1] + len(faces))
    archive_kwargs = {
        "frame_idx": np.asarray(frame_indices, dtype=np.int32),
        "vertex_offsets": np.asarray(vertex_offsets, dtype=np.int64),
        "face_offsets": np.asarray(face_offsets, dtype=np.int64),
        "faces": np.vstack(faces_all).astype(np.int32),
        "depth_scale": np.asarray([depth_scales[int(idx)] for idx in frame_indices], dtype=np.float32),
    }
    camera_archive = output_dir / "scaled_observed_object_meshes_camera_unscaled.npz"
    np.savez_compressed(camera_archive, vertices=np.vstack(camera_vertices_all).astype(np.float32), **archive_kwargs)
    world_archive = None
    if annotations is not None:
        world_archive = output_dir / "scaled_observed_object_meshes_world.npz"
        np.savez_compressed(world_archive, vertices=np.vstack(world_vertices_all).astype(np.float32), **archive_kwargs)
    return camera_archive, world_archive


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices, frame_points, input_rows = load_frame_points(args)
    poses, seq_reports = sequential_poses(frame_indices, frame_points, int(args.anchor_frame), args)
    poses, refine_reports = refine_poses(frame_indices, frame_points, poses, int(args.anchor_frame), args)
    fused = merged_cloud(frame_indices, frame_points, poses, args.fusion_voxel_size, args.normal_radius, args.normal_max_nn)
    clean, _ = fused.remove_statistical_outlier(nb_neighbors=int(args.outlier_neighbors), std_ratio=float(args.outlier_std_ratio))
    if len(clean.points) >= int(args.min_fused_points):
        fused = clean
    canonical_points = np.asarray(fused.points, dtype=np.float64)
    extent = canonical_points.max(axis=0) - canonical_points.min(axis=0)
    if float(np.max(extent)) > float(args.max_canonical_extent_m):
        raise RuntimeError(f"scaled fused extent is implausible: {extent.tolist()}")
    mesh = reconstruct_bpa_mesh(fused, args)
    canonical_mesh_path = args.output_dir / "canonical_scaled_observed_object_mesh.obj"
    mesh.export(canonical_mesh_path)
    camera_archive_path, world_archive_path = save_archives(args.output_dir, frame_indices, mesh, poses, input_rows, args.annotations)
    pose_payload = {
        str(idx): {
            "T_anchor_from_frame_camera_scaled": poses[idx].astype(float).tolist(),
            "T_frame_camera_scaled_from_anchor": np.linalg.inv(poses[idx]).astype(float).tolist(),
        }
        for idx in frame_indices
    }
    (args.output_dir / "object_poses_to_anchor_camera_scaled.json").write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "scaled_camera_depth_observed_surface_multiview_icp_bpa_v3",
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "anchor_frame": int(args.anchor_frame),
        "frames": [int(v) for v in frame_indices],
        "input_frames": input_rows,
        "sequential_registration": seq_reports,
        "refine_registration": refine_reports,
        "fused_point_cloud": summarize_points(canonical_points),
        "canonical_mesh": str(canonical_mesh_path),
        "camera_mesh_archive": str(camera_archive_path),
        "world_mesh_archive": str(world_archive_path) if world_archive_path is not None else None,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_m": (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).astype(float).tolist(),
        "watertight": bool(mesh.is_watertight),
        "penetration_supported": bool(mesh.is_watertight),
        "parameters": {
            "reference": args.reference,
            "reference_extent_xy_m": [float(x) for x in (args.reference_extent_xy_m or [])],
            "extent_quantile": float(args.extent_quantile),
            "voxel_size": float(args.voxel_size),
            "fusion_voxel_size": float(args.fusion_voxel_size),
            "icp_threshold_m": float(args.icp_threshold_m),
            "refine_passes": int(args.refine_passes),
            "max_canonical_extent_m": float(args.max_canonical_extent_m),
        },
        "elapsed_s": float(time.time() - started),
    }
    (args.output_dir / "qc_scaled_observed_object_mesh_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"sequential_registration", "refine_registration", "input_frames"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-points-per-frame", type=int, default=2500)
    parser.add_argument("--reference", choices=("anchor", "median"), default="median")
    parser.add_argument("--reference-extent-xy-m", type=float, nargs=2)
    parser.add_argument("--extent-quantile", type=float, default=0.05)
    parser.add_argument("--min-depth-scale", type=float, default=0.55)
    parser.add_argument("--max-depth-scale", type=float, default=1.80)
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
    parser.add_argument("--max-canonical-extent-m", type=float, default=0.55)
    parser.add_argument("--bpa-radius-scales", type=float, nargs="+", default=[1.5, 2.5, 4.0, 6.0])
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
