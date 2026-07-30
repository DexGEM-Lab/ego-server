#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from archive_aligned_mesh_prior_v7 import load_triangle_mesh, write_mesh_archive
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"invalid vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"invalid faces: {faces.shape}")
    tri = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    if not bool(valid.any()):
        raise RuntimeError("mesh has no positive-area faces")
    tri = tri[valid]
    areas = areas[valid]
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(len(tri), size=int(count), replace=True, p=areas / areas.sum())
    chosen = tri[face_ids]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    if not np.isfinite(points).all():
        raise RuntimeError("sampled surface points contain non-finite values")
    return points


def sample_rows(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("cannot sample empty point set")
    if len(points) <= int(count):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(count), replace=False)]


def rows_by_frame(alignment_report: dict) -> dict[int, dict]:
    rows = alignment_report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("alignment report must contain nonempty rows")
    out: dict[int, dict] = {}
    for row in rows:
        idx = int(row["frame_idx"])
        if idx in out:
            raise RuntimeError(f"alignment report has duplicate frame {idx}")
        sim3 = row.get("sim3")
        if not isinstance(sim3, dict):
            raise RuntimeError(f"alignment row lacks sim3 for frame {idx}")
        out[idx] = row
    return out


def prior_from_aligned(points_aligned: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rotation = np.asarray(sim3["rotation"], dtype=np.float64)
    translation = np.asarray(sim3["translation_m"], dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0.0 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise RuntimeError("invalid Sim3 row")
    points = ((np.asarray(points_aligned, dtype=np.float64) - translation[None, :]) / scale) @ rotation
    if not np.isfinite(points).all():
        raise RuntimeError("inverse Sim3 produced non-finite points")
    return points


def aligned_from_prior(points_prior: np.ndarray, sim3: dict) -> np.ndarray:
    scale = float(sim3["scale"])
    rotation = np.asarray(sim3["rotation"], dtype=np.float64)
    translation = np.asarray(sim3["translation_m"], dtype=np.float64)
    points = scale * (np.asarray(points_prior, dtype=np.float64) @ rotation.T) + translation[None, :]
    if not np.isfinite(points).all():
        raise RuntimeError("Sim3 produced non-finite points")
    return points


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
        raise RuntimeError(f"fused mesh extent is invalid: {extent.astype(float).tolist()}")
    return mesh, {
        "pcd_points": int(len(pcd.points)),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_extent_prior_units": extent.astype(float).tolist(),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
        "poisson_density_threshold": threshold,
    }


def surface_distance_row(
    frame_idx: int,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    observed_vertices: np.ndarray,
    observed_faces: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    candidate_points = sample_surface(
        candidate_vertices,
        candidate_faces,
        int(args.eval_candidate_surface_points),
        int(args.seed) + int(frame_idx) + 401,
    )
    observed_points = sample_surface(
        observed_vertices,
        observed_faces,
        int(args.eval_observed_surface_points),
        int(args.seed) + int(frame_idx) + 503,
    )
    observed_tree = cKDTree(observed_points)
    candidate_tree = cKDTree(candidate_points)
    prior_to_observed = observed_tree.query(candidate_points, workers=-1)[0]
    observed_to_prior = candidate_tree.query(observed_points, workers=-1)[0]
    p_to_o = summarize(prior_to_observed.astype(np.float64))
    o_to_p = summarize(observed_to_prior.astype(np.float64))
    return {
        "frame_idx": int(frame_idx),
        "candidate_vertices": int(len(candidate_vertices)),
        "candidate_faces": int(len(candidate_faces)),
        "observed_vertices": int(len(observed_vertices)),
        "observed_faces": int(len(observed_faces)),
        "visible_surface_coverage_p95_m": float(o_to_p["p95"]),
        "hidden_surface_conflict_p95_m": float(p_to_o["p95"]),
        "bidirectional_p95_m": float(max(float(o_to_p["p95"]), float(p_to_o["p95"]))),
        "prior_to_observed_m": p_to_o,
        "observed_to_prior_m": o_to_p,
    }


def run(args: argparse.Namespace) -> dict:
    prior = load_triangle_mesh(args.mesh_prior)
    observed_meshes = load_mesh_archive(args.observed_mesh_archive)
    alignment = load_json(args.alignment_report)
    align_rows = rows_by_frame(alignment)
    selected_frames = [
        idx
        for idx in range(int(args.frame_start), int(args.frame_end) + 1)
        if idx in observed_meshes and idx in align_rows
    ]
    if len(selected_frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(selected_frames)} frames have both observed mesh and alignment rows")

    prior_vertices = np.asarray(prior.vertices, dtype=np.float64)
    prior_faces = np.asarray(prior.faces, dtype=np.int32)
    prior_extent = np.asarray(prior.extents, dtype=np.float64)
    prior_points = sample_surface(
        prior_vertices,
        prior_faces,
        int(args.max_prior_points),
        int(args.seed) + 101,
    )

    observed_prior_points = []
    frame_rows = []
    for idx in selected_frames:
        observed_vertices, observed_faces = observed_meshes[idx]
        observed_world = sample_surface(
            observed_vertices,
            observed_faces,
            int(args.max_observed_surface_points_per_frame),
            int(args.seed) + idx,
        )
        observed_prior = prior_from_aligned(observed_world, align_rows[idx]["sim3"])
        observed_prior_points.append(observed_prior)
        frame_rows.append(
            {
                "frame_idx": int(idx),
                "observed_surface_points": int(len(observed_world)),
                "input_visible_surface_coverage_p95_m": float(align_rows[idx]["visible_surface_coverage_p95_m"]),
                "input_hidden_surface_conflict_p95_m": float(align_rows[idx]["hidden_surface_conflict_p95_m"]),
                "sim3_scale": float(align_rows[idx]["sim3"]["scale"]),
            }
        )

    observed_prior_all = np.vstack(observed_prior_points)
    observed_sample = sample_rows(observed_prior_all, int(args.max_observed_fusion_points), int(args.seed) + 202)
    observed_extent = np.percentile(observed_sample, 95.0, axis=0) - np.percentile(observed_sample, 5.0, axis=0)
    extent_ratio = observed_extent / np.maximum(prior_extent, 1e-9)
    scales = np.asarray([row["sim3_scale"] for row in frame_rows], dtype=np.float64)
    if not np.isfinite(observed_extent).all():
        raise RuntimeError("observed prior-coordinate extent is non-finite")
    if float(np.max(observed_extent)) > float(args.max_observed_prior_extent_m):
        raise RuntimeError(f"observed surface is unstable in prior coordinates: extent={observed_extent.astype(float).tolist()}")
    if float(np.max(extent_ratio)) > float(args.max_observed_prior_extent_ratio):
        raise RuntimeError(
            "observed surface extent is inconsistent with prior extent: "
            f"observed={observed_extent.astype(float).tolist()} prior={prior_extent.astype(float).tolist()}"
        )
    if float(np.max(scales) / max(float(np.min(scales)), 1e-9)) > float(args.max_sim3_scale_ratio):
        raise RuntimeError(f"Sim3 scale is unstable across frames: scales={scales.astype(float).tolist()}")

    prior_count = int(round(len(prior_points) * max(0.0, float(args.prior_weight))))
    fusion_points = observed_sample if prior_count == 0 else np.vstack(
        [observed_sample, sample_rows(prior_points, prior_count, int(args.seed) + 303)]
    )
    fused_prior, mesh_report = reconstruct_mesh(fusion_points, args)
    fused_vertices = np.asarray(fused_prior.vertices, dtype=np.float64)
    fused_faces = np.asarray(fused_prior.faces, dtype=np.int32)

    archive_rows = []
    eval_rows = []
    for idx in selected_frames:
        candidate_vertices = aligned_from_prior(fused_vertices, align_rows[idx]["sim3"])
        archive_rows.append((idx, candidate_vertices, fused_faces))
        observed_vertices, observed_faces = observed_meshes[idx]
        eval_rows.append(surface_distance_row(idx, candidate_vertices, fused_faces, observed_vertices, observed_faces, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_mesh_path = args.output_dir / "fused_prior_canonical_mesh.ply"
    fused_prior.export(prior_mesh_path)
    archive_path = args.output_dir / "fused_prior_meshes_world.npz"
    write_mesh_archive(archive_path, archive_rows)

    visible_p95 = np.asarray([row["visible_surface_coverage_p95_m"] for row in eval_rows], dtype=np.float64)
    hidden_p95 = np.asarray([row["hidden_surface_conflict_p95_m"] for row in eval_rows], dtype=np.float64)
    bidirectional_p95 = np.asarray([row["bidirectional_p95_m"] for row in eval_rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "fuse_v9_prior_observed_mesh_surfaces",
        "claim_tested": "a complete generated object mesh can be used as a canonical prior while accepted measured mesh surfaces dominate the visible geometry before replay and physics checks",
        "mesh_prior": str(args.mesh_prior),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "alignment_report": str(args.alignment_report),
        "prior_mesh": str(prior_mesh_path),
        "output_mesh_archive": str(archive_path),
        "mesh_archive": str(archive_path),
        "frame_count": int(len(selected_frames)),
        "first_frame": int(selected_frames[0]),
        "last_frame": int(selected_frames[-1]),
        "alignment_bidirectional_p95_m": summarize(bidirectional_p95),
        "visible_surface_coverage_p95_m": summarize(visible_p95),
        "hidden_surface_conflict_p95_m": summarize(hidden_p95),
        "alignment_scale": summarize(scales),
        "input_visible_surface_coverage_p95_m": summarize(
            np.asarray([row["input_visible_surface_coverage_p95_m"] for row in frame_rows], dtype=np.float64)
        ),
        "input_hidden_surface_conflict_p95_m": summarize(
            np.asarray([row["input_hidden_surface_conflict_p95_m"] for row in frame_rows], dtype=np.float64)
        ),
        "observed_prior_extent_m": observed_extent.astype(float).tolist(),
        "observed_prior_extent_ratio": extent_ratio.astype(float).tolist(),
        "mesh": mesh_report,
        "acceptance_policy": {
            "strict_full_surface_alignment_pass": bool(np.all(bidirectional_p95 <= float(args.max_bidirectional_p95_m))),
            "visible_surface_coverage_pass": bool(np.all(visible_p95 <= float(args.max_visible_surface_p95_m))),
            "max_bidirectional_p95_m": float(args.max_bidirectional_p95_m),
            "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
            "delivery_requires_downstream_replay": [
                "z-buffer silhouette and depth",
                "track-surface temporal support",
                "mesh-surface contact",
                "selected-contact SDF",
                "full-hand SDF",
                "visual review",
            ],
        },
        "frame_rows": frame_rows,
        "rows": eval_rows,
        "parameters": {
            "prior_weight": float(args.prior_weight),
            "max_observed_surface_points_per_frame": int(args.max_observed_surface_points_per_frame),
            "max_observed_fusion_points": int(args.max_observed_fusion_points),
            "max_prior_points": int(args.max_prior_points),
            "voxel_size_m": float(args.voxel_size_m),
            "poisson_depth": int(args.poisson_depth),
        },
    }
    output_json = args.output_dir / "qc_fuse_v9_prior_observed_mesh_surfaces.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_rows", "rows"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-observed-surface-points-per-frame", type=int, default=18000)
    parser.add_argument("--max-observed-fusion-points", type=int, default=90000)
    parser.add_argument("--max-prior-points", type=int, default=45000)
    parser.add_argument("--prior-weight", type=float, default=0.20)
    parser.add_argument("--voxel-size-m", type=float, default=0.0025)
    parser.add_argument("--normal-radius-m", type=float, default=0.025)
    parser.add_argument("--normal-max-nn", type=int, default=40)
    parser.add_argument("--normal-orientation-k", type=int, default=32)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-density-trim-quantile", type=float, default=0.02)
    parser.add_argument("--min-pcd-points", type=int, default=1000)
    parser.add_argument("--min-mesh-vertices", type=int, default=500)
    parser.add_argument("--min-mesh-faces", type=int, default=1000)
    parser.add_argument("--max-observed-prior-extent-m", type=float, default=3.0)
    parser.add_argument("--max-observed-prior-extent-ratio", type=float, default=1.6)
    parser.add_argument("--max-prior-extent-m", type=float, default=2.5)
    parser.add_argument("--max-sim3-scale-ratio", type=float, default=2.0)
    parser.add_argument("--eval-candidate-surface-points", type=int, default=40000)
    parser.add_argument("--eval-observed-surface-points", type=int, default=40000)
    parser.add_argument("--max-bidirectional-p95-m", type=float, default=0.010)
    parser.add_argument("--max-visible-surface-p95-m", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=9901)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
