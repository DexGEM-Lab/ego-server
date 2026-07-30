#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def load_depth_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    intrinsics = blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if depth.ndim != 3 or len(frame_idx) != depth.shape[0] or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(f"{path} has invalid depth or intrinsics shapes")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx.tolist()):
        if int(idx) in out:
            raise RuntimeError(f"{path} has duplicate frame {idx}")
        out[int(idx)] = (depth[i], intrinsics[i])
    return out


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid triangle mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def sample_mask_depth(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError("mask/depth has no valid object pixels")
    coords = np.c_[xs, ys]
    if len(coords) > int(max_points):
        rng = np.random.default_rng(int(seed))
        coords = coords[rng.choice(len(coords), size=int(max_points), replace=False)]
    z = depth[coords[:, 1], coords[:, 0]].astype(np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    x = (coords[:, 0].astype(np.float64) - cx) * z / fx
    y = (coords[:, 1].astype(np.float64) - cy) * z / fy
    points = np.c_[x, y, z]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("sampled mask-depth points are invalid")
    return points


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(faces) == 0 or not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        raise RuntimeError("mesh faces have invalid area")
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(len(faces), size=int(count), replace=True, p=areas / areas.sum())
    tri = vertices[faces[face_ids]]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def sample_rows(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("cannot sample empty point set")
    if len(points) <= int(count):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(count), replace=False)]


def prior_from_aligned(points_aligned: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rotation = np.asarray(sim3["rotation"], dtype=np.float64)
    translation = np.asarray(sim3["translation_m"], dtype=np.float64)
    if scale <= 0.0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("invalid Sim3 row")
    return ((np.asarray(points_aligned, dtype=np.float64) - translation[None, :]) / scale) @ rotation


def aligned_from_prior(points_prior: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rotation = np.asarray(sim3["rotation"], dtype=np.float64)
    translation = np.asarray(sim3["translation_m"], dtype=np.float64)
    return scale * (np.asarray(points_prior, dtype=np.float64) @ rotation.T) + translation[None, :]


def reconstruct_mesh(points: np.ndarray, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if float(args.voxel_size_m) > 0.0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size_m))
    if len(pcd.points) < int(args.min_pcd_points):
        raise RuntimeError(f"too few fusion points after voxel downsample: {len(pcd.points)}")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.normal_radius_m), max_nn=int(args.normal_max_nn)))
    pcd.orient_normals_consistent_tangent_plane(int(args.normal_orientation_k))
    mesh_o3d, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(args.poisson_depth))
    densities = np.asarray(density, dtype=np.float64)
    if len(densities) == 0 or len(densities) != len(mesh_o3d.vertices):
        raise RuntimeError("Poisson reconstruction returned invalid density values")
    threshold = float(np.quantile(densities, float(args.poisson_density_trim_quantile)))
    mesh_o3d.remove_vertices_by_mask(densities < threshold)
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
    if not components:
        raise RuntimeError("fused mesh has no connected components")
    mesh = max(components, key=lambda item: float(item.area))
    mesh.remove_unreferenced_vertices()
    extent = np.asarray(mesh.extents, dtype=np.float64)
    if not np.isfinite(extent).all() or np.any(extent <= 0.0) or float(np.max(extent)) > float(args.max_prior_extent_m):
        raise RuntimeError(f"fused mesh extent is invalid: {extent.tolist()}")
    return mesh, {
        "pcd_points": int(len(pcd.points)),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_prior_units": extent.astype(float).tolist(),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
        "poisson_density_threshold": threshold,
    }


def write_archive(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    frame_idx = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame, vertices, faces in rows:
        frame_idx.append(int(frame))
        vertices_all.append(np.asarray(vertices, dtype=np.float32))
        faces_all.append(np.asarray(faces, dtype=np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(vertices_all[-1]))
        face_offsets.append(face_offsets[-1] + len(faces_all[-1]))
    if not frame_idx:
        raise RuntimeError("cannot write empty mesh archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_idx, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.concatenate(vertices_all, axis=0),
        faces=np.concatenate(faces_all, axis=0),
    )


def run(args: argparse.Namespace) -> dict:
    alignment = load_json(args.alignment_report)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    prior = load_mesh(args.mesh_prior)
    prior_points = sample_mesh_surface(prior, int(args.max_prior_points), int(args.seed) + 101)
    rows_by_frame = {int(row["frame_idx"]): row for row in alignment.get("rows", [])}
    selected_frames = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in rows_by_frame and idx in manifest and idx in depths]
    if len(selected_frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(selected_frames)} frames available for Sim3 fusion")

    observed_prior = []
    frame_rows = []
    for idx in selected_frames:
        depth, intrinsics = depths[idx]
        mask = read_mask(Path(manifest[idx]["mask"]), depth.shape)
        points = sample_mask_depth(mask, depth, intrinsics, int(args.max_observed_points_per_frame), int(args.seed) + idx)
        prior_space = prior_from_aligned(points, rows_by_frame[idx]["sim3"])
        observed_prior.append(prior_space)
        frame_rows.append(
            {
                "frame_idx": int(idx),
                "observed_points": int(len(points)),
                "sim3_scale": float(rows_by_frame[idx]["sim3"]["scale"]),
                "visible_surface_coverage_p95_m": float(rows_by_frame[idx]["visible_surface_coverage_p95_m"]),
            }
        )
    observed = np.vstack(observed_prior)
    observed_sample = sample_rows(observed, int(args.max_observed_fusion_points), int(args.seed) + 202)
    observed_extent = np.percentile(observed_sample, 95.0, axis=0) - np.percentile(observed_sample, 5.0, axis=0)
    scales = np.asarray([row["sim3_scale"] for row in frame_rows], dtype=np.float64)
    if not np.isfinite(observed_extent).all() or float(np.max(observed_extent)) > float(args.max_observed_prior_extent_m):
        raise RuntimeError(f"observed points are unstable in prior coordinates: extent={observed_extent.astype(float).tolist()}")
    if float(np.max(scales) / max(float(np.min(scales)), 1e-9)) > float(args.max_sim3_scale_ratio):
        raise RuntimeError(f"Sim3 scale is unstable across frames: scales={scales.astype(float).tolist()}")
    prior_count = int(round(len(prior_points) * max(0.0, float(args.prior_weight))))
    if prior_count > 0:
        fusion_points = np.vstack([observed_sample, sample_rows(prior_points, prior_count, int(args.seed) + 303)])
    else:
        fusion_points = observed_sample
    fused_prior, mesh_report = reconstruct_mesh(fusion_points, args)
    faces = np.asarray(fused_prior.faces, dtype=np.int32)
    archive_rows = []
    for idx in selected_frames:
        archive_rows.append((idx, aligned_from_prior(np.asarray(fused_prior.vertices, dtype=np.float64), rows_by_frame[idx]["sim3"]), faces))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_mesh_path = args.output_dir / "fused_prior_canonical_mesh.ply"
    fused_prior.export(prior_mesh_path)
    archive_path = args.output_dir / "fused_prior_meshes_world.npz"
    write_archive(archive_path, archive_rows)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "fuse_v7_sim3_prior_observed_surfaces",
        "claim_tested": "model-generated prior and model-produced mask/depth observations can be fused in the prior canonical frame using the same Sim3 rows that replay alignment used",
        "mesh_prior": str(args.mesh_prior),
        "alignment_report": str(args.alignment_report),
        "manifest": str(args.manifest),
        "metric_depth_npz": str(args.metric_depth_npz),
        "prior_mesh": str(prior_mesh_path),
        "mesh_archive": str(archive_path),
        "frames": selected_frames,
        "observed_points_total": int(len(observed)),
        "observed_points_used": int(len(observed_sample)),
        "observed_prior_extent_m": observed_extent.astype(float).tolist(),
        "fusion_points": int(len(fusion_points)),
        "sim3_scale": summarize(scales),
        "input_visible_surface_coverage_p95_m": summarize(np.asarray([row["visible_surface_coverage_p95_m"] for row in frame_rows], dtype=np.float64)),
        "frame_rows": frame_rows,
        "mesh": mesh_report,
        "parameters": {
            "prior_weight": float(args.prior_weight),
            "max_observed_points_per_frame": int(args.max_observed_points_per_frame),
            "max_observed_fusion_points": int(args.max_observed_fusion_points),
            "max_prior_points": int(args.max_prior_points),
            "voxel_size_m": float(args.voxel_size_m),
            "poisson_depth": int(args.poisson_depth),
        },
    }
    output_json = args.output_dir / "qc_fuse_v7_sim3_prior_observed_surfaces.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frame_rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-observed-points-per-frame", type=int, default=12000)
    parser.add_argument("--max-observed-fusion-points", type=int, default=70000)
    parser.add_argument("--max-prior-points", type=int, default=30000)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--voxel-size-m", type=float, default=0.003)
    parser.add_argument("--normal-radius-m", type=float, default=0.025)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    parser.add_argument("--normal-orientation-k", type=int, default=32)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-density-trim-quantile", type=float, default=0.015)
    parser.add_argument("--min-pcd-points", type=int, default=1000)
    parser.add_argument("--min-mesh-vertices", type=int, default=500)
    parser.add_argument("--min-mesh-faces", type=int, default=1000)
    parser.add_argument("--max-observed-prior-extent-m", type=float, default=1.0)
    parser.add_argument("--max-prior-extent-m", type=float, default=2.0)
    parser.add_argument("--max-sim3-scale-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7401)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
