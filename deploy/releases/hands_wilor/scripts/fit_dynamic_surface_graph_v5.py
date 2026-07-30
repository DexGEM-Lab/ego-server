#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def state_rows(path: Path) -> dict[int, dict]:
    payload = load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"state package lacks rows list: {path}")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            raise RuntimeError(f"invalid state row in {path}")
        out[int(row["frame_idx"])] = row
    return out


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    out_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    out_faces = np.asarray(mesh.triangles, dtype=np.int32)
    if out_vertices.ndim != 2 or out_vertices.shape[1] != 3 or len(out_vertices) < 16:
        raise RuntimeError("template simplification produced invalid vertices")
    if out_faces.ndim != 2 or out_faces.shape[1] != 3 or len(out_faces) < 16:
        raise RuntimeError("template simplification produced invalid faces")
    if out_faces.min() < 0 or out_faces.max() >= len(out_vertices):
        raise RuntimeError("template simplification produced invalid face indices")
    return out_vertices, out_faces


def sample_vertices(vertices: np.ndarray, count: int) -> np.ndarray:
    if len(vertices) <= int(count):
        return np.asarray(vertices, dtype=np.float64)
    ids = np.linspace(0, len(vertices) - 1, int(count), dtype=np.int64)
    return np.asarray(vertices[ids], dtype=np.float64)


def pcd(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    if len(cloud.points) == 0:
        raise RuntimeError("empty point cloud after voxel downsample")
    return cloud


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ homog.T).T[:, :3]


def rigid_initialization(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = np.median(target, axis=0) - np.median(source, axis=0)
    result = o3d.pipelines.registration.registration_icp(
        pcd(source, float(args.icp_voxel_m)),
        pcd(target, float(args.icp_voxel_m)),
        float(args.icp_threshold_m),
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(args.icp_iterations)),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    return transform, {
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
        "correspondence_count": int(len(result.correspondence_set)),
        "translation_m": transform[:3, 3].astype(float).tolist(),
    }


def adjacency_from_faces(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    sets = [set() for _ in range(int(vertex_count))]
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        sets[a].update((b, c))
        sets[b].update((a, c))
        sets[c].update((a, b))
    neighbors = [np.asarray(sorted(items), dtype=np.int64) for items in sets]
    isolated = [i for i, items in enumerate(neighbors) if len(items) == 0]
    if isolated:
        raise RuntimeError(f"template mesh has isolated vertices: {isolated[:8]}")
    return neighbors


def laplacian(vertices: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    out = np.empty_like(vertices, dtype=np.float64)
    for i, nbr in enumerate(neighbors):
        out[i] = vertices[i] - vertices[nbr].mean(axis=0)
    return out


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("cannot summarize empty array")
    return {
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def nearest_metrics(source: np.ndarray, target: np.ndarray) -> dict:
    distances = cKDTree(np.asarray(target, dtype=np.float64)).query(np.asarray(source, dtype=np.float64), k=1)[0]
    return summarize(distances)


def build_system(
    init_vertices: np.ndarray,
    data_targets: np.ndarray,
    data_weights: np.ndarray,
    neighbors: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[sparse.csr_matrix, list[tuple[str, int, int]]]:
    frame_count, vertex_count = data_weights.shape
    rows = []
    cols = []
    vals = []
    meta: list[tuple[str, int, int]] = []
    row_id = 0

    def var(t: int, i: int) -> int:
        return int(t) * int(vertex_count) + int(i)

    for t in range(frame_count):
        for i in range(vertex_count):
            rows.append(row_id)
            cols.append(var(t, i))
            vals.append(float(np.sqrt(float(args.data_weight) * data_weights[t, i])))
            meta.append(("data", t, i))
            row_id += 1

    for t in range(frame_count):
        for i, nbr in enumerate(neighbors):
            weight = float(np.sqrt(float(args.laplacian_weight)))
            rows.append(row_id)
            cols.append(var(t, i))
            vals.append(weight)
            coeff = -weight / float(len(nbr))
            for j in nbr:
                rows.append(row_id)
                cols.append(var(t, int(j)))
                vals.append(coeff)
            meta.append(("laplacian", t, i))
            row_id += 1

    for t in range(1, frame_count):
        for i in range(vertex_count):
            weight = float(np.sqrt(float(args.temporal_weight)))
            rows.extend((row_id, row_id))
            cols.extend((var(t, i), var(t - 1, i)))
            vals.extend((weight, -weight))
            meta.append(("temporal", t, i))
            row_id += 1

    matrix = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(row_id, frame_count * vertex_count),
    ).tocsr()
    return matrix, meta


def solve_vertices(
    matrix: sparse.csr_matrix,
    meta: list[tuple[str, int, int]],
    init_vertices: np.ndarray,
    data_targets: np.ndarray,
    data_weights: np.ndarray,
    neighbors: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    frame_count, vertex_count, _ = data_targets.shape
    init_lap = np.stack([laplacian(init_vertices[t], neighbors) for t in range(frame_count)], axis=0)
    solved = np.empty((frame_count, vertex_count, 3), dtype=np.float64)
    solver_rows = []
    for axis in range(3):
        rhs = np.empty(len(meta), dtype=np.float64)
        for row_id, (kind, t, i) in enumerate(meta):
            if kind == "data":
                rhs[row_id] = float(np.sqrt(float(args.data_weight) * data_weights[t, i])) * data_targets[t, i, axis]
            elif kind == "laplacian":
                rhs[row_id] = float(np.sqrt(float(args.laplacian_weight))) * init_lap[t, i, axis]
            elif kind == "temporal":
                rhs[row_id] = 0.0
            else:
                raise RuntimeError(f"unknown residual kind: {kind}")
        result = lsmr(matrix, rhs, atol=float(args.lsmr_tol), btol=float(args.lsmr_tol), maxiter=int(args.lsmr_maxiter))
        solved[:, :, axis] = result[0].reshape(frame_count, vertex_count)
        solver_rows.append(
            {
                "axis": int(axis),
                "istop": int(result[1]),
                "iterations": int(result[2]),
                "normr": float(result[3]),
                "normar": float(result[4]),
                "conda": float(result[6]),
            }
        )
    return solved, solver_rows


def save_mesh_archive(path: Path, frames: list[int], vertices_by_frame: np.ndarray, faces: np.ndarray) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for vertices in vertices_by_frame:
        vertices_all.append(np.asarray(vertices, dtype=np.float32))
        faces_all.append(np.asarray(faces, dtype=np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frames, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    states = state_rows(args.v5_state_json)
    frames = list(range(int(args.frame_start), int(args.frame_end) + 1))
    missing_mesh = [frame for frame in frames if frame not in meshes]
    if missing_mesh:
        raise RuntimeError(f"mesh archive missing frames: {missing_mesh[:8]}")
    missing_state = [frame for frame in frames if frame not in states]
    if missing_state:
        raise RuntimeError(f"state package missing frames: {missing_state[:8]}")
    bad_state = [frame for frame in frames if states[frame].get("geometry_state") != "map_observable_measured_geometry"]
    if bad_state:
        raise RuntimeError(f"dynamic surface fit requires map-observable frames, got non-observable frames: {bad_state[:8]}")

    anchor = int(args.anchor_frame)
    if anchor not in frames:
        raise RuntimeError(f"anchor frame {anchor} is outside requested window")
    anchor_vertices, anchor_faces = meshes[anchor]
    template_vertices, template_faces = simplify_mesh(anchor_vertices, anchor_faces, int(args.template_faces))
    neighbors = adjacency_from_faces(len(template_vertices), template_faces)

    init_vertices = []
    data_targets = []
    data_weights = []
    target_samples: dict[int, np.ndarray] = {}
    initial_reports = {}
    for frame in frames:
        target_vertices, _ = meshes[frame]
        target = sample_vertices(target_vertices, int(args.target_sample_points))
        target_samples[frame] = target
        transform, icp_report = rigid_initialization(template_vertices, target, args)
        initial = transform_points(template_vertices, transform)
        distances, ids = cKDTree(target).query(initial, k=1)
        weights = 1.0 / (1.0 + np.square(distances / float(args.robust_sigma_m)))
        init_vertices.append(initial)
        data_targets.append(target[ids])
        data_weights.append(weights)
        initial_reports[int(frame)] = {
            "icp": icp_report,
            "initial_graph_to_target_m": summarize(distances),
            "initial_target_to_graph_m": nearest_metrics(target, initial),
            "robust_data_weight": summarize(weights),
        }

    init_arr = np.stack(init_vertices, axis=0)
    targets_arr = np.stack(data_targets, axis=0)
    weights_arr = np.stack(data_weights, axis=0)
    matrix, meta = build_system(init_arr, targets_arr, weights_arr, neighbors, args)
    solved, solver_rows = solve_vertices(matrix, meta, init_arr, targets_arr, weights_arr, neighbors, args)

    frame_reports = []
    for t, frame in enumerate(frames):
        target = target_samples[frame]
        displacement = np.linalg.norm(solved[t] - init_arr[t], axis=1)
        frame_reports.append(
            {
                "frame_idx": int(frame),
                "state": str(states[frame]["geometry_state"]),
                "graph_to_target_m": nearest_metrics(solved[t], target),
                "target_to_graph_m": nearest_metrics(target, solved[t]),
                "deformation_from_rigid_init_m": summarize(displacement),
            }
        )
    temporal_steps = []
    for t in range(1, len(frames)):
        step = np.linalg.norm(solved[t] - solved[t - 1], axis=1)
        temporal_steps.append({"from_frame": int(frames[t - 1]), "to_frame": int(frames[t]), "vertex_step_m": summarize(step)})

    mesh_path = args.output_dir / "dynamic_surface_meshes_world.npz"
    save_mesh_archive(mesh_path, frames, solved, template_faces)
    report = {
        "status": "ok",
        "method": "fit_dynamic_surface_graph_v5",
        "claim_tested": "one deformable category-agnostic topology can explain a V5 map-observable measured-mesh window without using a rigid canonical object map",
        "mesh_archive": str(args.mesh_archive),
        "v5_state_json": str(args.v5_state_json),
        "output_mesh_archive": str(mesh_path),
        "frames": frames,
        "anchor_frame": int(anchor),
        "template_vertices": int(len(template_vertices)),
        "template_faces": int(len(template_faces)),
        "linear_system": {"rows": int(matrix.shape[0]), "cols": int(matrix.shape[1]), "nnz": int(matrix.nnz)},
        "parameters": {
            "template_faces": int(args.template_faces),
            "target_sample_points": int(args.target_sample_points),
            "data_weight": float(args.data_weight),
            "laplacian_weight": float(args.laplacian_weight),
            "temporal_weight": float(args.temporal_weight),
            "robust_sigma_m": float(args.robust_sigma_m),
            "icp_threshold_m": float(args.icp_threshold_m),
            "icp_voxel_m": float(args.icp_voxel_m),
        },
        "solver": solver_rows,
        "initial_reports": initial_reports,
        "frame_reports": frame_reports,
        "temporal_steps": temporal_steps,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "qc_dynamic_surface_graph_v5.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"initial_reports", "frame_reports", "temporal_steps"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--v5-state-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--template-faces", type=int, default=8000)
    parser.add_argument("--target-sample-points", type=int, default=60000)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--laplacian-weight", type=float, default=0.35)
    parser.add_argument("--temporal-weight", type=float, default=0.08)
    parser.add_argument("--robust-sigma-m", type=float, default=0.010)
    parser.add_argument("--icp-threshold-m", type=float, default=0.040)
    parser.add_argument("--icp-voxel-m", type=float, default=0.004)
    parser.add_argument("--icp-iterations", type=int, default=80)
    parser.add_argument("--lsmr-tol", type=float, default=1e-8)
    parser.add_argument("--lsmr-maxiter", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
