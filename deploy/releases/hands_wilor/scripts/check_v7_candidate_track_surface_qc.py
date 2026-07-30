#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_factor_graph_v6 import load_pair_edges, report_pair_rows, sparse_edges_path
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def frame_set(meshes: dict[int, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> set[int]:
    lo = min(meshes) if args.frame_start is None else int(args.frame_start)
    hi = max(meshes) if args.frame_end is None else int(args.frame_end)
    frames = {idx for idx in meshes if lo <= idx <= hi}
    if not frames:
        raise RuntimeError(f"no candidate mesh frames selected in [{lo}, {hi}]")
    return frames


def source_key(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def observed_mesh_archive_from_pair_rows(pair_report: dict, pair_report_path: Path, rows: list[dict]) -> Path:
    observed_paths = set()
    for row in rows:
        edge_json = sparse_edges_path(row, pair_report, pair_report_path)
        edge_report = load_json(edge_json)
        mesh_archive = edge_report.get("mesh_archive")
        if not isinstance(mesh_archive, str) or not mesh_archive:
            raise RuntimeError(f"sparse edge report lacks mesh_archive provenance: {edge_json}")
        observed_paths.add(str(Path(mesh_archive).resolve()))
    if len(observed_paths) != 1:
        raise RuntimeError(f"track QC needs one observed mesh archive, got {sorted(observed_paths)}")
    return Path(next(iter(observed_paths)))


def nearest_surface_rows(
    candidate_meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    observed_meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    pair_report: dict,
    pair_report_path: Path,
    frames: set[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trees = {frame: cKDTree(candidate_meshes[frame][0]) for frame in frames}
    rows: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []
    for row in report_pair_rows(pair_report, pair_report_path):
        source_frame = int(row["source_frame"])
        target_frame = int(row["target_frame"])
        if source_frame not in frames or target_frame not in frames:
            continue
        if not row.get("rigid_factor_ready"):
            rejected_pairs.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "reason": "pair_factor_not_ready",
                    "inlier_residual_m": row.get("inlier_residual_m", {}),
                }
            )
            continue
        if source_frame not in observed_meshes or target_frame not in observed_meshes:
            raise RuntimeError(f"observed archive lacks ready pair {source_frame}->{target_frame}")
        source_observed_vertices = observed_meshes[source_frame][0]
        target_observed_vertices = observed_meshes[target_frame][0]
        edge_json = sparse_edges_path(row, pair_report, pair_report_path)
        accepted = 0
        for edge in load_pair_edges(edge_json, source_frame, target_frame):
            source_vertex = int(edge["source_vertex"])
            target_vertex = int(edge["target_vertex"])
            source_track_point = source_observed_vertices[source_vertex]
            target_track_point = target_observed_vertices[target_vertex]
            rot = np.asarray(row["rotation"], dtype=np.float64)
            trans = np.asarray(row["translation_m"], dtype=np.float64)
            factor_residual = float(np.linalg.norm(source_track_point @ rot + trans - target_track_point))
            if factor_residual > float(args.max_pair_factor_residual_m):
                continue
            source_dist, source_candidate_vertex = trees[source_frame].query(source_track_point)
            target_dist, target_candidate_vertex = trees[target_frame].query(target_track_point)
            if max(float(source_dist), float(target_dist)) > float(args.max_track_surface_distance_m):
                continue
            source_candidate_point = candidate_meshes[source_frame][0][int(source_candidate_vertex)]
            target_candidate_point = candidate_meshes[target_frame][0][int(target_candidate_vertex)]
            transformed = source_candidate_point @ rot + trans
            rows.append(
                {
                    "track_source": source_key(edge_json),
                    "track_id": int(edge["track_id"]),
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "source_vertex": source_vertex,
                    "target_vertex": target_vertex,
                    "source_candidate_vertex": int(source_candidate_vertex),
                    "target_candidate_vertex": int(target_candidate_vertex),
                    "source_surface_distance_m": float(source_dist),
                    "target_surface_distance_m": float(target_dist),
                    "factor_residual_m": factor_residual,
                    "pair_residual_m": float(np.linalg.norm(transformed - target_candidate_point)),
                    "source_correction_m": (source_candidate_point - source_track_point).astype(float).tolist(),
                    "target_correction_m": (target_candidate_point - target_track_point).astype(float).tolist(),
                    "factor_inlier_residual_m": row.get("inlier_residual_m", {}),
                }
            )
            accepted += 1
        if accepted == 0:
            rejected_pairs.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "reason": "no_edges_below_surface_distance_threshold",
                }
            )
    return rows, rejected_pairs


def finite_summary(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return summarize(values.astype(np.float64))


def run(args: argparse.Namespace) -> dict:
    candidate_meshes = load_mesh_archive(args.candidate_mesh_archive)
    frames = frame_set(candidate_meshes, args)
    pair_report = load_json(args.pair_factors_json)
    selected_rows = [
        row
        for row in report_pair_rows(pair_report, args.pair_factors_json)
        if int(row["source_frame"]) in frames and int(row["target_frame"]) in frames
    ]
    if not selected_rows:
        raise RuntimeError("no pair-factor rows selected for candidate track QC")
    observed_mesh_archive = observed_mesh_archive_from_pair_rows(pair_report, args.pair_factors_json, selected_rows)
    observed_meshes = load_mesh_archive(observed_mesh_archive)
    missing_observed = sorted(frame for frame in frames if frame not in observed_meshes)
    if missing_observed:
        raise RuntimeError(f"observed mesh archive lacks selected frames: {missing_observed}")
    rows, rejected_pairs = nearest_surface_rows(candidate_meshes, observed_meshes, pair_report, args.pair_factors_json, frames, args)
    if len(rows) < int(args.min_edges):
        raise RuntimeError(f"only {len(rows)} accepted track-surface edges")
    pair_residuals = np.asarray([row["pair_residual_m"] for row in rows], dtype=np.float64)
    factor_residuals = np.asarray([row["factor_residual_m"] for row in rows], dtype=np.float64)
    source_dist = np.asarray([row["source_surface_distance_m"] for row in rows], dtype=np.float64)
    target_dist = np.asarray([row["target_surface_distance_m"] for row in rows], dtype=np.float64)
    correction = np.asarray(
        [
            np.linalg.norm(np.asarray(row["source_correction_m"], dtype=np.float64))
            for row in rows
        ]
        + [
            np.linalg.norm(np.asarray(row["target_correction_m"], dtype=np.float64))
            for row in rows
        ],
        dtype=np.float64,
    )
    source_tracks = defaultdict(set)
    for row in rows:
        source_tracks[(row["track_source"], int(row["track_id"]))].update([int(row["source_frame"]), int(row["target_frame"])])
    track_count = int(len(source_tracks))
    accepted_edge_count = int(len(rows))
    factor_summary = finite_summary(factor_residuals)
    pair_summary = finite_summary(pair_residuals)
    source_summary = finite_summary(source_dist)
    target_summary = finite_summary(target_dist)
    correction_summary = finite_summary(correction)
    checks = {
        "min_tracks": track_count >= int(args.min_tracks),
        "min_edges": accepted_edge_count >= int(args.min_edges),
        "pair_residual_p95": True
        if args.max_pair_residual_p95_m is None
        else float(pair_summary["p95"]) <= float(args.max_pair_residual_p95_m),
        "correction_displacement_p95": True
        if args.max_correction_displacement_p95_m is None
        else float(correction_summary["p95"]) <= float(args.max_correction_displacement_p95_m),
    }
    accepted = all(checks.values())
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "check_v7_candidate_track_surface_qc",
        "claim_tested": "candidate mesh surface must support the model-produced 3D track observations without sharing measured-mesh topology",
        "candidate_mesh_archive": str(args.candidate_mesh_archive),
        "observed_mesh_archive": str(observed_mesh_archive),
        "pair_factors_json": str(args.pair_factors_json),
        "frame_start": min(frames),
        "frame_end": max(frames),
        "track_count": track_count,
        "accepted_edge_count": accepted_edge_count,
        "rejected_pair_count": int(len(rejected_pairs)),
        "factor_residual_m": factor_summary,
        "pair_residual_m": pair_summary,
        "source_surface_distance_m": source_summary,
        "target_surface_distance_m": target_summary,
        "correction_displacement_m": correction_summary,
        "pass": checks,
        "parameters": {
            "max_pair_factor_residual_m": float(args.max_pair_factor_residual_m),
            "max_track_surface_distance_m": float(args.max_track_surface_distance_m),
            "min_edges": int(args.min_edges),
            "min_tracks": int(args.min_tracks),
            "max_pair_residual_p95_m": None
            if args.max_pair_residual_p95_m is None
            else float(args.max_pair_residual_p95_m),
            "max_correction_displacement_p95_m": None
            if args.max_correction_displacement_p95_m is None
            else float(args.max_correction_displacement_p95_m),
        },
        "edges": rows,
        "rejected_pairs": rejected_pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "qc_v7_candidate_track_surface.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"edges", "rejected_pairs"}}, indent=2))
    if args.fail_on_rejected and not accepted:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"track surface QC rejected candidate; failed checks: {failed}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-mesh-archive", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-track-surface-distance-m", type=float, default=0.012)
    parser.add_argument("--max-pair-factor-residual-m", type=float, default=0.012)
    parser.add_argument("--min-edges", type=int, default=24)
    parser.add_argument("--min-tracks", type=int, default=1)
    parser.add_argument("--max-pair-residual-p95-m", type=float)
    parser.add_argument("--max-correction-displacement-p95-m", type=float)
    parser.add_argument("--fail-on-rejected", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
