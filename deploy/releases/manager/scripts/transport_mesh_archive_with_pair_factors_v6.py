#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def mesh_sample(vertices: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices
    ids = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
    return vertices[ids]


def surface_residual(source_vertices: np.ndarray, target_vertices: np.ndarray, max_points: int) -> dict:
    source = mesh_sample(source_vertices, max_points)
    target = mesh_sample(target_vertices, max_points)
    source_to_target = cKDTree(target).query(source, k=1)[0]
    target_to_source = cKDTree(source).query(target, k=1)[0]
    return {
        "source_to_target_m": summarize(source_to_target),
        "target_to_source_m": summarize(target_to_source),
        "bidirectional_abs_m": summarize(np.concatenate([source_to_target, target_to_source])),
    }


def pair_factor_rows(report: dict, path: Path) -> list[dict]:
    method = report.get("method")
    rows = report.get("pair_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"pair factor report has no pair_rows list: {path}")
    if method not in {
        "fit_cotracker_pairwise_rigid_factors_v6",
        "merge_cotracker_pair_factors_v6",
    }:
        raise RuntimeError(f"unsupported pair factor report method {method!r}: {path}")
    return rows


def write_mesh_archive(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    if not rows:
        raise RuntimeError("no transported meshes to write")
    frame_idx = []
    vertices_all = []
    faces_all = []
    vertex_offsets = [0]
    face_offsets = [0]
    for frame, vertices, faces in rows:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(f"invalid vertices for frame {frame}: {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError(f"invalid faces for frame {frame}: {faces.shape}")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise RuntimeError(f"face index out of range for frame {frame}")
        frame_idx.append(int(frame))
        vertices_all.append(vertices.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
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
    pair_report = load_json(args.pair_factors_json)
    source_meshes = load_mesh_archive(args.mesh_archive)
    target_meshes = load_mesh_archive(args.mesh_archive)
    transported_rows = []
    report_rows = []

    for row in pair_factor_rows(pair_report, args.pair_factors_json):
        if not row.get("rigid_factor_ready") and not args.include_rejected_pairs:
            continue
        source_frame = int(row["source_frame"])
        target_frame = int(row["target_frame"])
        if source_frame not in source_meshes or target_frame not in target_meshes:
            raise RuntimeError(f"mesh archive lacks source/target pair {source_frame}->{target_frame}")
        vertices, faces = source_meshes[source_frame]
        rot = np.asarray(row["rotation"], dtype=np.float64)
        trans = np.asarray(row["translation_m"], dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,):
            raise RuntimeError(f"invalid factor transform for pair {source_frame}->{target_frame}")
        transported = vertices @ rot + trans
        target_vertices, _target_faces = target_meshes[target_frame]
        residual = surface_residual(transported, target_vertices, int(args.max_surface_points))
        report_rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "source_vertices": int(len(vertices)),
                "target_vertices": int(len(target_vertices)),
                "rigid_factor_ready": bool(row.get("rigid_factor_ready")),
                "factor_inlier_residual_m": row.get("inlier_residual_m", {}),
                "transported_surface_residual_m": residual,
            }
        )
        transported_rows.append((target_frame, transported, faces))

    write_mesh_archive(args.output_mesh_archive, transported_rows)
    pair_medians = [
        item["transported_surface_residual_m"]["bidirectional_abs_m"].get("median", np.nan)
        for item in report_rows
    ]
    pair_p95 = [
        item["transported_surface_residual_m"]["bidirectional_abs_m"].get("p95", np.nan)
        for item in report_rows
    ]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "transport_mesh_archive_with_pair_factors_v6",
        "claim_tested": "ready CoTracker pairwise SE3 factors can rigidly transport the accepted source-frame mesh into the target-frame object geometry",
        "mesh_archive": str(args.mesh_archive),
        "pair_factors_json": str(args.pair_factors_json),
        "output_mesh_archive": str(args.output_mesh_archive),
        "transported_pair_count": int(len(report_rows)),
        "pair_bidirectional_median_m": summarize(np.asarray(pair_medians, dtype=np.float64)),
        "pair_bidirectional_p95_m": summarize(np.asarray(pair_p95, dtype=np.float64)),
        "rows": report_rows,
        "parameters": {
            "include_rejected_pairs": bool(args.include_rejected_pairs),
            "max_surface_points": int(args.max_surface_points),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--output-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-surface-points", type=int, default=50000)
    parser.add_argument("--include-rejected-pairs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
