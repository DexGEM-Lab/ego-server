#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from align_mesh_prior_v3 import sample_mesh_surface
from archive_aligned_mesh_prior_v7 import load_triangle_mesh
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= int(target_faces):
        return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("simplification produced invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("simplification produced invalid faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError("simplification produced out-of-range faces")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def bidirectional_surface_error(source: trimesh.Trimesh, target: trimesh.Trimesh, samples: int, seed: int) -> dict:
    source_points = sample_mesh_surface(source, int(samples), int(seed))
    target_points = sample_mesh_surface(target, int(samples), int(seed) + 17_071)
    source_to_target, _ = cKDTree(target_points).query(source_points, k=1)
    target_to_source, _ = cKDTree(source_points).query(target_points, k=1)
    bidirectional = np.concatenate([source_to_target, target_to_source])
    return {
        "source_to_simplified_m": summarize(np.asarray(source_to_target, dtype=np.float64)),
        "simplified_to_source_m": summarize(np.asarray(target_to_source, dtype=np.float64)),
        "bidirectional_m": summarize(np.asarray(bidirectional, dtype=np.float64)),
    }


def run(args: argparse.Namespace) -> dict:
    source = load_triangle_mesh(args.input_mesh)
    simplified = simplify_mesh(source, int(args.target_faces))
    if len(simplified.faces) > len(source.faces):
        raise RuntimeError("simplified mesh has more faces than source mesh")
    error = bidirectional_surface_error(source, simplified, int(args.samples), int(args.seed))
    bidirectional = error["bidirectional_m"]
    p95 = float(bidirectional.get("p95", np.inf))
    max_error = float(bidirectional.get("max", np.inf))
    accepted = bool(p95 <= float(args.max_bidirectional_p95_m) and max_error <= float(args.max_bidirectional_max_m))
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    simplified.export(args.output_mesh)
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": False,
        "method": "simplify_mesh_prior_with_surface_certificate_v7",
        "claim_tested": "a reduced triangle mesh represents the source generated prior surface within a sampled bidirectional surface-distance tolerance",
        "input_mesh": str(args.input_mesh),
        "output_mesh": str(args.output_mesh),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "simplified_vertices": int(len(simplified.vertices)),
        "simplified_faces": int(len(simplified.faces)),
        "face_reduction_fraction": float(1.0 - (len(simplified.faces) / max(1, len(source.faces)))),
        "samples_per_direction": int(args.samples),
        "surface_error": error,
        "thresholds": {
            "max_bidirectional_p95_m": float(args.max_bidirectional_p95_m),
            "max_bidirectional_max_m": float(args.max_bidirectional_max_m),
        },
        "pass": {
            "bidirectional_p95": bool(p95 <= float(args.max_bidirectional_p95_m)),
            "bidirectional_max": bool(max_error <= float(args.max_bidirectional_max_m)),
        },
        "use_policy": "Only an accepted certificate can justify replaying this reduced mesh as a computational proxy for the source prior; final delivery still requires guarded replay and physics QC.",
    }
    save_json(args.output_json, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--target-faces", type=int, default=200000)
    parser.add_argument("--samples", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--max-bidirectional-p95-m", type=float, default=0.0015)
    parser.add_argument("--max-bidirectional-max-m", type=float, default=0.0060)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
