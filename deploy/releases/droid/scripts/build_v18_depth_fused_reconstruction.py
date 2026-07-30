#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[reportMissingTypeStubs]
from scipy.spatial.transform import Rotation  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_depth_fused_reconstruction"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def vector(value: Any, dim: int) -> np.ndarray | None:
    if not (isinstance(value, list) and len(value) == dim):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def object_graph_estimates(annotation_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    ann = load_json(annotation_path)
    estimates: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in ann.get("frames", []):
        if not isinstance(raw_frame, dict):
            continue
        frame_idx_raw = raw_frame.get("frame_idx")
        if not isinstance(frame_idx_raw, int):
            continue
        frame_idx = frame_idx_raw
        graph = raw_frame.get("factor_graph_solution")
        if not isinstance(graph, dict):
            continue
        variables = graph.get("variables")
        if not isinstance(variables, dict):
            continue
        for raw_var in variables.get("object_se3", []):
            if not isinstance(raw_var, dict):
                continue
            variable_id = str(raw_var.get("variable_id"))
            if not variable_id.startswith("object_se3::"):
                continue
            object_id = variable_id[len("object_se3::"):]
            est = raw_var.get("estimate")
            init = raw_var.get("initial")
            if not (isinstance(est, list) and len(est) >= 3):
                continue
            estimates[(frame_idx, object_id)] = {
                "estimate": est,
                "initial": init,
                "source": raw_var.get("source"),
                "dimension": raw_var.get("dimension"),
                "observation_residual_norm": raw_var.get("observation_residual_norm"),
            }
    return estimates, ann.get("factor_graph_summary", {}) if isinstance(ann.get("factor_graph_summary"), dict) else {}


def transform_world_to_object(points_world: np.ndarray, pose: dict[str, Any]) -> tuple[np.ndarray, str]:
    est = pose.get("estimate")
    t = vector(est[:3] if isinstance(est, list) else None, 3)
    if t is None:
        return points_world.copy(), "world_frame_no_pose"
    if isinstance(est, list) and len(est) >= 6:
        rotvec = vector(est[3:6], 3)
        if rotvec is not None:
            rotation = Rotation.from_rotvec(rotvec).as_matrix()
            return (points_world - t[None, :]) @ rotation, "graph_se3_inverse"
    return points_world - t[None, :], "graph_translation_only_inverse"


def maybe_downsample(points: np.ndarray, target_points: int) -> np.ndarray:
    if points.shape[0] <= target_points:
        return points
    step = max(1, int(math.ceil(points.shape[0] / target_points)))
    return points[::step]


def reconstruct_mesh(points: np.ndarray, object_dir: Path, object_safe: str, args: argparse.Namespace) -> dict[str, Any]:
    object_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "point_count_input": int(points.shape[0]),
        "poisson_mesh_path": None,
        "convex_hull_mesh_path": None,
        "status": "unresolved",
        "blockers": [],
    }
    if points.shape[0] < 30:
        out["blockers"].append("too_few_depth_fused_points_for_surface_reconstruction")
        return out
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    voxel_size = max(args.min_voxel_m, diag / args.voxel_divisor) if diag > 0 else args.min_voxel_m
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    down_points = np.asarray(pcd.points)
    out["point_count_downsampled"] = int(down_points.shape[0])
    out["voxel_size_m"] = float(voxel_size)
    pcd_path = object_dir / f"{object_safe}_fused_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)
    out["fused_point_cloud_path"] = str(pcd_path)
    if down_points.shape[0] < 30:
        out["blockers"].append("too_few_downsampled_points_for_mesh")
        return out
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        hull_path = object_dir / f"{object_safe}_convex_hull_candidate.ply"
        o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False)
        out["convex_hull_mesh_path"] = str(hull_path)
        out["convex_hull_vertices"] = int(np.asarray(hull.vertices).shape[0])
        out["convex_hull_faces"] = int(np.asarray(hull.triangles).shape[0])
    except Exception as exc:  # Open3D/QHull can fail for near-planar surfaces.
        out["blockers"].append(f"convex_hull_failed:{type(exc).__name__}")
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 4.0, args.min_voxel_m * 4.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
        mesh = mesh.crop(bbox)
        densities_np = np.asarray(densities)
        if densities_np.size and np.asarray(mesh.vertices).shape[0] == densities_np.shape[0]:
            keep_threshold = float(np.quantile(densities_np, args.poisson_density_quantile))
            mesh.remove_vertices_by_mask(densities_np < keep_threshold)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        poisson_path = object_dir / f"{object_safe}_poisson_visible_completion_candidate.ply"
        o3d.io.write_triangle_mesh(str(poisson_path), mesh, write_ascii=False, compressed=False)
        out["poisson_mesh_path"] = str(poisson_path)
        out["poisson_vertices"] = int(np.asarray(mesh.vertices).shape[0])
        out["poisson_faces"] = int(np.asarray(mesh.triangles).shape[0])
        out["status"] = "depth_fused_visible_geometry_mesh_candidate"
    except Exception as exc:
        out["blockers"].append(f"poisson_reconstruction_failed:{type(exc).__name__}")
        if out.get("convex_hull_mesh_path"):
            out["status"] = "depth_fused_visible_geometry_hull_candidate_poisson_failed"
    return out


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.visible_geometry_root / case / "v18_visible_geometry_archive_report.json"
    visible_report = load_json(report_path)
    archive_path = Path(str(visible_report.get("archive_npz")))
    annotation_path = args.full_pipeline_root / case / "annotations_v18_full.json"
    estimates, graph_summary = object_graph_estimates(annotation_path)
    data = np.load(archive_path, allow_pickle=True)
    frame_idx = data["frame_idx"]
    object_ids = data["object_id"]
    vertex_offsets = data["vertex_offsets"]
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    fused_by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_pose_rows = 0
    for row_idx in range(len(frame_idx)):
        fidx = int(frame_idx[row_idx])
        object_id = str(object_ids[row_idx])
        start = int(vertex_offsets[row_idx])
        end = int(vertex_offsets[row_idx + 1])
        if end <= start:
            continue
        pts_world = vertices[start:end]
        pose = estimates.get((fidx, object_id))
        if pose is None:
            missing_pose_rows += 1
            pts_obj = pts_world.copy()
            transform_status = "missing_graph_pose_world_frame_preserved"
        else:
            pts_obj, transform_status = transform_world_to_object(pts_world, pose)
        fused_by_object[object_id].append(pts_obj.astype(np.float32))
        source_rows[object_id].append(
            {
                "frame_idx": fidx,
                "archive_row_index": row_idx,
                "source_vertex_count": int(pts_world.shape[0]),
                "transform_status": transform_status,
                "graph_pose_dimension": pose.get("dimension") if pose else None,
                "graph_pose_source": pose.get("source") if pose else None,
                "graph_observation_residual_norm": pose.get("observation_residual_norm") if pose else None,
            }
        )
    case_dir = args.output_root / case
    object_rows: list[dict[str, Any]] = []
    for object_id, chunks in sorted(fused_by_object.items()):
        points = np.concatenate(chunks, axis=0).astype(np.float64)
        sampled_points = maybe_downsample(points, args.max_reconstruction_points)
        object_safe = safe_name(object_id)
        mesh_info = reconstruct_mesh(sampled_points, case_dir / object_safe, object_safe, args)
        mn_arr = sampled_points.min(axis=0) if sampled_points.size else np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        mx_arr = sampled_points.max(axis=0) if sampled_points.size else np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        row = {
            "object_id": object_id,
            "source_frame_count": len(chunks),
            "source_point_count": int(points.shape[0]),
            "sampled_point_count": int(sampled_points.shape[0]),
            "canonical_coordinate_source": "inverse_graph_object_se3_when_available_else_world_frame_preserved",
            "canonical_bbox_min_m": [float(v) for v in mn_arr.tolist()],
            "canonical_bbox_max_m": [float(v) for v in mx_arr.tolist()],
            "mesh_reconstruction": mesh_info,
            "source_rows_sample": source_rows[object_id][:20],
            "source_rows_total": len(source_rows[object_id]),
            "hidden_geometry_status": "candidate_surface_completion_from_depth_fused_visible_points_not_accepted_complete_geometry",
            "object_geometry_complete": False,
            "acceptance_scope": "real_depth_fused_visible_surface_reconstruction_with_uncertain_completion_candidate",
        }
        object_rows.append(row)
    out = {
        "method": "build_v18_depth_fused_reconstruction",
        "status": STATUS,
        "claim": "Depth-backed visible object surfaces are fused into graph-object coordinates and reconstructed as point clouds plus Poisson/hull candidate meshes. This is a real geometry artifact, but it is not accepted as complete hidden object geometry.",
        "case": case,
        "sources": {
            "visible_geometry_archive_report": str(report_path),
            "visible_geometry_npz": str(archive_path),
            "v18_full_annotations": str(annotation_path),
        },
        "frame_surface_rows": int(len(frame_idx)),
        "object_count": len(object_rows),
        "missing_graph_pose_rows": missing_pose_rows,
        "factor_graph_solver": graph_summary.get("solver"),
        "factor_graph_object_se3_count": graph_summary.get("variable_counts", {}).get("object_se3") if isinstance(graph_summary.get("variable_counts"), dict) else None,
        "parameters": {
            "max_reconstruction_points": args.max_reconstruction_points,
            "min_voxel_m": args.min_voxel_m,
            "voxel_divisor": args.voxel_divisor,
            "poisson_depth": args.poisson_depth,
            "poisson_density_quantile": args.poisson_density_quantile,
        },
        "object_rows": object_rows,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
        "object_geometry_complete": False,
        "hidden_geometry_reconstructed": False,
        "complete_object_pose_ready": False,
    }
    write_json(case_dir / "v18_depth_fused_reconstruction_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    outputs = []
    for case in args.cases:
        outputs.append(build_case(case, args))
    summary = {
        "method": "build_v18_depth_fused_reconstruction",
        "status": STATUS,
        "case_count": len(outputs),
        "cases": [
            {
                "case": out["case"],
                "object_count": out["object_count"],
                "frame_surface_rows": out["frame_surface_rows"],
                "missing_graph_pose_rows": out["missing_graph_pose_rows"],
                "objects_with_poisson_mesh": sum(1 for row in out["object_rows"] if row.get("mesh_reconstruction", {}).get("poisson_mesh_path")),
                "objects_with_hull_mesh": sum(1 for row in out["object_rows"] if row.get("mesh_reconstruction", {}).get("convex_hull_mesh_path")),
            }
            for out in outputs
        ],
        "claim_scope": "real_depth_fused_visible_geometry_reconstruction_artifact_not_complete_hidden_geometry_acceptance",
        "default_path_uses_bundlesdf_or_nerf": False,
    }
    write_json(args.output_root / "v18_depth_fused_reconstruction_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-geometry-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-reconstruction-points", type=int, default=60000)
    parser.add_argument("--min-voxel-m", type=float, default=0.006)
    parser.add_argument("--voxel-divisor", type=float, default=160.0)
    parser.add_argument("--poisson-depth", type=int, default=7)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
