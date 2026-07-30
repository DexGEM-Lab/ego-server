#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[reportMissingTypeStubs]
from scipy.spatial.transform import Rotation  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_part_depth_fused_reconstruction"
CLAIM = (
    "Depth-backed visible part surfaces for part-SE3-supported articulated objects are fused into graph-part "
    "coordinates and reconstructed as candidate point clouds plus Poisson/hull meshes. This is hidden/part geometry "
    "candidate evidence only; it does not mark part pose, contact ownership, or object pose complete."
)

FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "part"


def supported_part_labels(part_se3_report: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for raw in require_list(part_se3_report.get("rows"), "part se3 rows"):
        row = require_dict(raw, "part se3 row")
        if row.get("part_se3_pair_state") != "part_se3_surface_residual_supported_visible_only_not_pose":
            continue
        object_id = str(row.get("object_id"))
        for label in require_list(row.get("part_track_labels"), "part labels"):
            out.add((object_id, str(label)))
    return out


def part_graph_estimates(annotation_path: Path) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any]]:
    ann = require_dict(load_json(annotation_path), f"annotations {annotation_path}")
    estimates: dict[tuple[int, str, str], dict[str, Any]] = {}
    prefix = "part_se3::"
    for raw_frame in require_list(ann.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        variables = require_dict(require_dict(frame.get("factor_graph_solution"), "factor graph").get("variables"), "variables")
        for raw_var in require_list(variables.get("part_se3"), "part se3 variables"):
            var = require_dict(raw_var, "part se3 variable")
            variable_id = str(var.get("variable_id"))
            if not variable_id.startswith(prefix):
                continue
            rest = variable_id[len(prefix) :]
            object_id, sep, label = rest.partition("::")
            if not sep or not label:
                continue
            est = var.get("estimate")
            if not (isinstance(est, list) and len(est) >= 3):
                continue
            estimates[(frame_idx, object_id, label)] = {
                "estimate": est,
                "initial": var.get("initial"),
                "dimension": var.get("dimension"),
                "source": var.get("source"),
                "observation_residual_norm": var.get("observation_residual_norm"),
            }
    return estimates, ann.get("factor_graph_summary", {}) if isinstance(ann.get("factor_graph_summary"), dict) else {}


def transform_camera_to_part(points_camera: np.ndarray, pose: dict[str, Any]) -> tuple[np.ndarray, str]:
    est = pose.get("estimate")
    t = vector(est[:3] if isinstance(est, list) else None, 3)
    if t is None:
        return points_camera.copy(), "camera_frame_no_part_pose"
    if isinstance(est, list) and len(est) >= 6:
        rotvec = vector(est[3:6], 3)
        if rotvec is not None:
            rotation = Rotation.from_rotvec(rotvec).as_matrix()
            return (points_camera - t[None, :]) @ rotation, "graph_part_se3_inverse"
    return points_camera - t[None, :], "graph_part_translation_only_inverse"


def maybe_downsample(points: np.ndarray, target_points: int) -> np.ndarray:
    if points.shape[0] <= target_points:
        return points
    step = max(1, int(math.ceil(points.shape[0] / target_points)))
    return points[::step]


def reconstruct_mesh(points: np.ndarray, part_dir: Path, part_safe: str, args: argparse.Namespace) -> dict[str, Any]:
    part_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "point_count_input": int(points.shape[0]),
        "poisson_mesh_path": None,
        "convex_hull_mesh_path": None,
        "status": "unresolved",
        "blockers": [],
    }
    if points.shape[0] < 30:
        out["blockers"].append("too_few_depth_fused_points_for_part_surface_reconstruction")
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
    pcd_path = part_dir / f"{part_safe}_fused_points.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)
    out["fused_point_cloud_path"] = str(pcd_path)
    if down_points.shape[0] < 30:
        out["blockers"].append("too_few_downsampled_points_for_part_mesh")
        return out
    try:
        hull, _ = pcd.compute_convex_hull()
        hull.compute_vertex_normals()
        hull_path = part_dir / f"{part_safe}_convex_hull_candidate.ply"
        o3d.io.write_triangle_mesh(str(hull_path), hull, write_ascii=False, compressed=False)
        out["convex_hull_mesh_path"] = str(hull_path)
        out["convex_hull_vertices"] = int(np.asarray(hull.vertices).shape[0])
        out["convex_hull_faces"] = int(np.asarray(hull.triangles).shape[0])
    except Exception as exc:
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
        poisson_path = part_dir / f"{part_safe}_poisson_visible_completion_candidate.ply"
        o3d.io.write_triangle_mesh(str(poisson_path), mesh, write_ascii=False, compressed=False)
        out["poisson_mesh_path"] = str(poisson_path)
        out["poisson_vertices"] = int(np.asarray(mesh.vertices).shape[0])
        out["poisson_faces"] = int(np.asarray(mesh.triangles).shape[0])
        out["status"] = "part_depth_fused_visible_geometry_mesh_candidate"
    except Exception as exc:
        out["blockers"].append(f"poisson_reconstruction_failed:{type(exc).__name__}")
        if out.get("convex_hull_mesh_path"):
            out["status"] = "part_depth_fused_visible_geometry_hull_candidate_poisson_failed"
    return out


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    part_se3_path = args.part_se3_root / case / "v18_part_se3_surface_residuals_report.json"
    annotation_path = args.full_pipeline_root / case / "annotations_v18_full.json"
    surfaces = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    part_se3 = require_dict(load_json(part_se3_path), f"{case} part se3")
    labels = supported_part_labels(part_se3)
    estimates, graph_summary = part_graph_estimates(annotation_path)
    archive_path = Path(str(surfaces.get("archive_npz")))
    data = np.load(archive_path, allow_pickle=True)
    frame_idx = data["frame_idx"]
    object_ids = data["object_id"]
    part_labels = data["part_track_label"]
    vertex_offsets = data["vertex_offsets"]
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    fused_by_part: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    source_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped_unsupported_rows = 0
    missing_pose_rows = 0
    for row_idx in range(len(frame_idx)):
        fidx = int(frame_idx[row_idx])
        object_id = str(object_ids[row_idx])
        label = str(part_labels[row_idx])
        key = (object_id, label)
        if key not in labels:
            skipped_unsupported_rows += 1
            continue
        start = int(vertex_offsets[row_idx])
        end = int(vertex_offsets[row_idx + 1])
        if end <= start:
            continue
        pts_camera = vertices[start:end]
        pose = estimates.get((fidx, object_id, label))
        if pose is None:
            missing_pose_rows += 1
            pts_part = pts_camera.copy()
            transform_status = "missing_graph_part_pose_camera_frame_preserved"
        else:
            pts_part, transform_status = transform_camera_to_part(pts_camera, pose)
        fused_by_part[key].append(pts_part.astype(np.float32))
        source_rows[key].append(
            {
                "frame_idx": fidx,
                "archive_row_index": row_idx,
                "source_vertex_count": int(pts_camera.shape[0]),
                "transform_status": transform_status,
                "graph_pose_dimension": pose.get("dimension") if pose else None,
                "graph_pose_source": pose.get("source") if pose else None,
                "graph_observation_residual_norm": pose.get("observation_residual_norm") if pose else None,
            }
        )
    case_dir = args.output_root / case
    part_rows: list[dict[str, Any]] = []
    for (object_id, label), chunks in sorted(fused_by_part.items()):
        points = np.concatenate(chunks, axis=0).astype(np.float64)
        sampled_points = maybe_downsample(points, args.max_reconstruction_points)
        part_safe = safe_name(f"{object_id}_{label}")
        mesh_info = reconstruct_mesh(sampled_points, case_dir / part_safe, part_safe, args)
        mn_arr = sampled_points.min(axis=0) if sampled_points.size else np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        mx_arr = sampled_points.max(axis=0) if sampled_points.size else np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        mesh_ready = bool(mesh_info.get("poisson_mesh_path") or mesh_info.get("convex_hull_mesh_path"))
        part_rows.append(
            {
                "object_id": object_id,
                "part_track_label": label,
                "source_frame_count": len(chunks),
                "source_point_count": int(points.shape[0]),
                "sampled_point_count": int(sampled_points.shape[0]),
                "canonical_coordinate_source": "inverse_graph_part_se3_when_available_else_camera_frame_preserved",
                "canonical_bbox_min_m": [float(v) for v in mn_arr.tolist()],
                "canonical_bbox_max_m": [float(v) for v in mx_arr.tolist()],
                "mesh_reconstruction": mesh_info,
                "source_rows_sample": source_rows[(object_id, label)][:20],
                "source_rows_total": len(source_rows[(object_id, label)]),
                "supported_part_se3_source": str(part_se3_path),
                "hidden_part_geometry_reconstructed": mesh_ready,
                "part_geometry_complete": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
                "acceptance_scope": "part_depth_fused_visible_surface_reconstruction_candidate_with_hidden_surface_uncertainty",
            }
        )
    mesh_ready_count = sum(1 for row in part_rows if row.get("hidden_part_geometry_reconstructed") is True)
    out = {
        "method": "build_v18_part_depth_fused_reconstruction",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "part_visible_surfaces": str(surfaces_path),
            "part_surface_npz": str(archive_path),
            "part_se3_surface_residuals": str(part_se3_path),
            "v18_full_annotations": str(annotation_path),
        },
        "supported_part_label_count": len(labels),
        "part_count": len(part_rows),
        "frame_surface_rows_consumed": sum(int(row.get("source_frame_count", 0)) for row in part_rows),
        "skipped_unsupported_part_surface_rows": skipped_unsupported_rows,
        "missing_graph_part_pose_rows": missing_pose_rows,
        "part_rows": part_rows,
        "part_mesh_candidate_count": mesh_ready_count,
        "hidden_part_geometry_reconstructed_count": mesh_ready_count,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "factor_graph_solver": graph_summary.get("solver"),
        "parameters": {
            "max_reconstruction_points": args.max_reconstruction_points,
            "min_voxel_m": args.min_voxel_m,
            "voxel_divisor": args.voxel_divisor,
            "poisson_depth": args.poisson_depth,
            "poisson_density_quantile": args.poisson_density_quantile,
        },
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(case_dir / "v18_part_depth_fused_reconstruction_report.json", out)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [build_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts = Counter()
    for report in reports:
        for row in report.get("part_rows", []):
            mesh = row.get("mesh_reconstruction") if isinstance(row, dict) else {}
            if isinstance(mesh, dict):
                state_counts[str(mesh.get("status"))] += 1
    summary = {
        "method": "build_v18_part_depth_fused_reconstruction",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "part_count": sum(require_int(report.get("part_count"), "part count") for report in reports),
        "part_mesh_candidate_count": sum(require_int(report.get("part_mesh_candidate_count"), "mesh count") for report in reports),
        "hidden_part_geometry_reconstructed_count": sum(require_int(report.get("hidden_part_geometry_reconstructed_count"), "hidden part geometry count") for report in reports),
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "mesh_status_counts": dict(sorted(state_counts.items())),
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_depth_fused_reconstruction_report.json"),
                "part_count": report["part_count"],
                "part_mesh_candidate_count": report["part_mesh_candidate_count"],
                "missing_graph_part_pose_rows": report["missing_graph_part_pose_rows"],
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_depth_fused_reconstruction_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--part-se3-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_se3_surface_residuals"))
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_depth_fused_reconstruction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-reconstruction-points", type=int, default=60000)
    parser.add_argument("--min-voxel-m", type=float, default=0.002)
    parser.add_argument("--voxel-divisor", type=float, default=80.0)
    parser.add_argument("--poisson-depth", type=int, default=7)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
