#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def state_rows(path: Path) -> dict[int, dict]:
    rows = load_json(path).get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"state package lacks rows: {path}")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            raise RuntimeError(f"invalid state row in {path}")
        out[int(row["frame_idx"])] = row
    return out


def sample_points(vertices: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= int(count):
        ids = np.arange(len(vertices), dtype=np.int64)
    else:
        ids = np.linspace(0, len(vertices) - 1, int(count), dtype=np.int64)
    return np.asarray(vertices[ids], dtype=np.float64), ids


def pcd(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    if len(cloud.points) == 0:
        raise RuntimeError("empty point cloud after downsampling")
    return cloud


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (np.asarray(transform, dtype=np.float64) @ homog.T).T[:, :3]


def pair_transform(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
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


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError("cannot summarize empty values")
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def build_pair(frame_a: int, frame_b: int, meshes: dict[int, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> dict:
    vertices_a, _ = meshes[frame_a]
    vertices_b, _ = meshes[frame_b]
    sample_a, ids_a = sample_points(vertices_a, int(args.sample_points))
    sample_b, ids_b = sample_points(vertices_b, int(args.sample_points))
    transform, icp = pair_transform(sample_a, sample_b, args)
    aligned_a = transform_points(sample_a, transform)
    tree_b = cKDTree(sample_b)
    dist_ab, nearest_b = tree_b.query(aligned_a, k=1)
    tree_a = cKDTree(aligned_a)
    dist_ba, nearest_a = tree_a.query(sample_b, k=1)
    mutual = np.arange(len(sample_a), dtype=np.int64) == nearest_a[nearest_b]
    accepted = (
        mutual
        & (dist_ab <= float(args.max_edge_distance_m))
        & (dist_ba[nearest_b] <= float(args.max_edge_distance_m))
    )
    accepted_ids = np.nonzero(accepted)[0]
    if len(accepted_ids) > int(args.max_edges_per_pair):
        keep = accepted_ids[np.argsort(dist_ab[accepted_ids])[: int(args.max_edges_per_pair)]]
        accepted_ids = np.sort(keep)
    edges = []
    for local_id in accepted_ids:
        target_local = int(nearest_b[int(local_id)])
        edges.append(
            {
                "source_vertex_id": int(ids_a[int(local_id)]),
                "target_vertex_id": int(ids_b[target_local]),
                "source_sample_id": int(local_id),
                "target_sample_id": int(target_local),
                "distance_m": float(dist_ab[int(local_id)]),
            }
        )
    overlap_fraction = float(len(edges) / max(len(sample_a), 1))
    stable = overlap_fraction >= float(args.min_overlap_fraction)
    return {
        "from_frame": int(frame_a),
        "to_frame": int(frame_b),
        "icp": icp,
        "source_to_target_distance_m": summarize(dist_ab),
        "target_to_source_distance_m": summarize(dist_ba),
        "mutual_fraction": float(np.mean(mutual)),
        "accepted_edge_count": int(len(edges)),
        "sample_count": int(len(sample_a)),
        "accepted_overlap_fraction": overlap_fraction,
        "stable_transport_pair": bool(stable),
        "edges": edges,
    }


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
    allowed_states = {str(item) for item in args.allowed_geometry_state}
    bad_state = [frame for frame in frames if states[frame].get("geometry_state") not in allowed_states]
    if bad_state:
        raise RuntimeError(f"transport edge build got disallowed geometry states: {bad_state[:8]}")
    pairs = [build_pair(a, b, meshes, args) for a, b in zip(frames[:-1], frames[1:])]
    stable_pairs = [pair for pair in pairs if bool(pair["stable_transport_pair"])]
    payload = {
        "status": "ok",
        "method": "build_dynamic_surface_transport_edges_v5",
        "claim_tested": "stable local surface correspondences can be stored as temporal edges while preserving the accepted per-frame measured meshes",
        "mesh_archive": str(args.mesh_archive),
        "v5_state_json": str(args.v5_state_json),
        "frames": frames,
        "parameters": {
            "sample_points": int(args.sample_points),
            "max_edge_distance_m": float(args.max_edge_distance_m),
            "min_overlap_fraction": float(args.min_overlap_fraction),
            "max_edges_per_pair": int(args.max_edges_per_pair),
            "icp_threshold_m": float(args.icp_threshold_m),
            "icp_voxel_m": float(args.icp_voxel_m),
            "allowed_geometry_state": sorted(allowed_states),
        },
        "pair_count": int(len(pairs)),
        "stable_pair_count": int(len(stable_pairs)),
        "accepted_edge_count": int(sum(int(pair["accepted_edge_count"]) for pair in pairs)),
        "accepted_overlap_fraction": summarize(np.asarray([pair["accepted_overlap_fraction"] for pair in pairs], dtype=np.float64)),
        "source_to_target_p95_m": summarize(np.asarray([pair["source_to_target_distance_m"]["p95"] for pair in pairs], dtype=np.float64)),
        "target_to_source_p95_m": summarize(np.asarray([pair["target_to_source_distance_m"]["p95"] for pair in pairs], dtype=np.float64)),
        "pairs": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "dynamic_surface_transport_edges_v5.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--v5-state-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--sample-points", type=int, default=80000)
    parser.add_argument("--max-edge-distance-m", type=float, default=0.006)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.45)
    parser.add_argument("--max-edges-per-pair", type=int, default=12000)
    parser.add_argument("--icp-threshold-m", type=float, default=0.035)
    parser.add_argument("--icp-voxel-m", type=float, default=0.003)
    parser.add_argument("--icp-iterations", type=int, default=80)
    parser.add_argument(
        "--allowed-geometry-state",
        action="append",
        default=["map_observable_measured_geometry"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
