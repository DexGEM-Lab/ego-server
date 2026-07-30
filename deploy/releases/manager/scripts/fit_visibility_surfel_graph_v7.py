#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_factor_graph_v6 import edge_residual, load_pair_edges, report_pair_rows, sparse_edges_path
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def frame_list(meshes: dict[int, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> set[int]:
    lo = min(meshes) if args.frame_start is None else int(args.frame_start)
    hi = max(meshes) if args.frame_end is None else int(args.frame_end)
    frames = {idx for idx in meshes if lo <= idx <= hi}
    if not frames:
        raise RuntimeError(f"no mesh frames selected in [{lo}, {hi}]")
    return frames


def source_key(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def node_key(edge_json: Path, track_id: int, frame_idx: int) -> tuple[str, int, int]:
    return source_key(edge_json), int(track_id), int(frame_idx)


def node_spread(points: list[np.ndarray]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) <= 1:
        return 0.0
    center = np.median(arr, axis=0)
    return float(np.max(np.linalg.norm(arr - center, axis=1)))


def collect_observations(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    pair_report: dict,
    pair_report_path: Path,
    frames: set[int],
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict], list[dict]]:
    node_points: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
    node_vertices: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    raw_edges: list[dict] = []
    rejected_pairs: list[dict] = []

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
        if source_frame not in meshes or target_frame not in meshes:
            raise RuntimeError(f"mesh archive lacks ready pair {source_frame}->{target_frame}")
        edge_json = sparse_edges_path(row, pair_report, pair_report_path)
        edge_report = load_json(edge_json)
        edge_mesh_archive = edge_report.get("mesh_archive")
        if edge_mesh_archive is None:
            raise RuntimeError(f"sparse edge report lacks mesh_archive provenance: {edge_json}")
        if Path(edge_mesh_archive).resolve() != args.mesh_archive.resolve():
            raise RuntimeError(
                "sparse edge vertex indices were built against a different mesh archive: "
                f"{edge_mesh_archive} vs {args.mesh_archive}"
            )
        source_vertices, _ = meshes[source_frame]
        target_vertices, _ = meshes[target_frame]
        accepted_count = 0
        rejected_count = 0
        for edge in load_pair_edges(edge_json, source_frame, target_frame):
            residual = edge_residual(source_vertices, target_vertices, row, edge)
            if residual > float(args.max_pair_residual_m):
                rejected_count += 1
                continue
            track_id = int(edge["track_id"])
            source_vertex = int(edge["source_vertex"])
            target_vertex = int(edge["target_vertex"])
            source_node = node_key(edge_json, track_id, source_frame)
            target_node = node_key(edge_json, track_id, target_frame)
            node_points[source_node].append(source_vertices[source_vertex].astype(np.float64))
            node_points[target_node].append(target_vertices[target_vertex].astype(np.float64))
            node_vertices[source_node].append(source_vertex)
            node_vertices[target_node].append(target_vertex)
            raw_edges.append(
                {
                    "source_node": source_node,
                    "target_node": target_node,
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "track_source": source_key(edge_json),
                    "track_id": track_id,
                    "source_vertex": source_vertex,
                    "target_vertex": target_vertex,
                    "pair_residual_m": float(residual),
                    "rotation": row["rotation"],
                    "translation_m": row["translation_m"],
                    "factor_inlier_residual_m": row.get("inlier_residual_m", {}),
                }
            )
            accepted_count += 1
        if accepted_count == 0:
            rejected_pairs.append(
                {
                    "source_frame": source_frame,
                    "target_frame": target_frame,
                    "reason": "no_edges_below_pair_residual_threshold",
                    "raw_rejected_edge_count": int(rejected_count),
                }
            )

    nodes: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, points in node_points.items():
        spread = node_spread(points)
        arr = np.asarray(points, dtype=np.float64)
        nodes[key] = {
            "key": key,
            "accepted": bool(spread <= float(args.max_duplicate_spread_m)),
            "measured_position_m": np.median(arr, axis=0),
            "duplicate_count": int(len(points)),
            "duplicate_spread_m": float(spread),
            "vertices": sorted({int(v) for v in node_vertices[key]}),
        }
    return nodes, raw_edges, rejected_pairs


def accepted_graph(nodes: dict[tuple[str, int, int], dict[str, Any]], raw_edges: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    node_keys = sorted([key for key, value in nodes.items() if value["accepted"]], key=lambda x: (x[0], x[1], x[2]))
    node_index = {key: i for i, key in enumerate(node_keys)}
    node_rows = []
    for key in node_keys:
        source, track_id, frame_idx = key
        value = nodes[key]
        node_rows.append(
            {
                "node_id": int(node_index[key]),
                "track_source": source,
                "track_id": int(track_id),
                "frame_idx": int(frame_idx),
                "measured_position_m": value["measured_position_m"].astype(float).tolist(),
                "duplicate_count": int(value["duplicate_count"]),
                "duplicate_spread_m": float(value["duplicate_spread_m"]),
                "vertices": value["vertices"],
            }
        )

    graph_edges = []
    rejected_edges = []
    for edge in raw_edges:
        src = edge["source_node"]
        dst = edge["target_node"]
        if src not in node_index or dst not in node_index:
            rejected_edges.append(
                {
                    "source_node": list(src),
                    "target_node": list(dst),
                    "reason": "duplicate_observation_conflict",
                    "pair_residual_m": float(edge["pair_residual_m"]),
                }
            )
            continue
        graph_edges.append(
            {
                "source_node_id": int(node_index[src]),
                "target_node_id": int(node_index[dst]),
                "source_frame": int(edge["source_frame"]),
                "target_frame": int(edge["target_frame"]),
                "track_source": edge["track_source"],
                "track_id": int(edge["track_id"]),
                "pair_residual_m": float(edge["pair_residual_m"]),
                "rotation": edge["rotation"],
                "translation_m": edge["translation_m"],
                "factor_inlier_residual_m": edge["factor_inlier_residual_m"],
            }
        )
    return node_rows, graph_edges, rejected_edges


def smooth_triples(node_rows: list[dict], max_gap: int) -> list[dict]:
    by_track: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in node_rows:
        by_track[(str(row["track_source"]), int(row["track_id"]))].append(row)
    triples = []
    for (track_source, track_id), rows in by_track.items():
        ordered = sorted(rows, key=lambda item: int(item["frame_idx"]))
        for a, b, c in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
            gap_ab = int(b["frame_idx"]) - int(a["frame_idx"])
            gap_bc = int(c["frame_idx"]) - int(b["frame_idx"])
            if gap_ab <= int(max_gap) and gap_bc <= int(max_gap):
                triples.append(
                    {
                        "track_source": track_source,
                        "track_id": int(track_id),
                        "node_a": int(a["node_id"]),
                        "node_b": int(b["node_id"]),
                        "node_c": int(c["node_id"]),
                        "frame_a": int(a["frame_idx"]),
                        "frame_b": int(b["frame_idx"]),
                        "frame_c": int(c["frame_idx"]),
                    }
                )
    return triples


def unpack(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, 3)


def residual_vector(
    x: np.ndarray,
    measured: np.ndarray,
    graph_edges: list[dict],
    triples: list[dict],
    args: argparse.Namespace,
) -> np.ndarray:
    pos = unpack(x)
    residuals = [((pos - measured) / float(args.observation_sigma_m)).reshape(-1)]
    for edge in graph_edges:
        src = int(edge["source_node_id"])
        dst = int(edge["target_node_id"])
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        residuals.append(((pos[src] @ rot + trans - pos[dst]) / float(args.factor_sigma_m)).reshape(-1))
    for triple in triples:
        a = int(triple["node_a"])
        b = int(triple["node_b"])
        c = int(triple["node_c"])
        residuals.append(((pos[c] - 2.0 * pos[b] + pos[a]) / float(args.smooth_sigma_m)).reshape(-1))
    return np.concatenate(residuals).astype(np.float64)


def pair_residuals(positions: np.ndarray, graph_edges: list[dict]) -> np.ndarray:
    values = []
    for edge in graph_edges:
        src = int(edge["source_node_id"])
        dst = int(edge["target_node_id"])
        rot = np.asarray(edge["rotation"], dtype=np.float64)
        trans = np.asarray(edge["translation_m"], dtype=np.float64)
        values.append(float(np.linalg.norm(positions[src] @ rot + trans - positions[dst])))
    return np.asarray(values, dtype=np.float64)


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    pair_report = load_json(args.pair_factors_json)
    frames = frame_list(meshes, args)
    nodes, raw_edges, rejected_pairs = collect_observations(meshes, pair_report, args.pair_factors_json, frames, args)
    node_rows, graph_edges, rejected_edges = accepted_graph(nodes, raw_edges)
    if len(node_rows) < int(args.min_nodes):
        raise RuntimeError(f"only {len(node_rows)} accepted surfel nodes")
    if len(graph_edges) < int(args.min_edges):
        raise RuntimeError(f"only {len(graph_edges)} accepted surfel edges")
    triples = smooth_triples(node_rows, int(args.max_smooth_gap_frames))
    measured = np.asarray([row["measured_position_m"] for row in node_rows], dtype=np.float64)
    before = measured.copy()
    result = least_squares(
        residual_vector,
        before.reshape(-1),
        args=(measured, graph_edges, triples, args),
        loss=args.loss,
        f_scale=float(args.loss_f_scale),
        max_nfev=int(args.max_nfev),
        xtol=float(args.xtol),
        ftol=float(args.ftol),
        gtol=float(args.gtol),
    )
    after = unpack(result.x)
    before_pair = pair_residuals(before, graph_edges)
    after_pair = pair_residuals(after, graph_edges)
    correction = np.linalg.norm(after - before, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions_npz = args.output_dir / "visibility_surfel_positions_v7.npz"
    np.savez_compressed(
        positions_npz,
        node_id=np.asarray([row["node_id"] for row in node_rows], dtype=np.int32),
        frame_idx=np.asarray([row["frame_idx"] for row in node_rows], dtype=np.int32),
        track_id=np.asarray([row["track_id"] for row in node_rows], dtype=np.int32),
        measured_position_m=before.astype(np.float32),
        solved_position_m=after.astype(np.float32),
    )
    for row, solved in zip(node_rows, after, strict=True):
        row["solved_position_m"] = solved.astype(float).tolist()

    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "fit_visibility_surfel_graph_v7",
        "claim_tested": "mesh-attached learned point tracks can form a visibility-aware temporal surfel state without replacing measured per-frame object meshes",
        "mesh_archive": str(args.mesh_archive),
        "pair_factors_json": str(args.pair_factors_json),
        "frame_start": min(frames),
        "frame_end": max(frames),
        "node_count": int(len(node_rows)),
        "raw_edge_count": int(len(raw_edges)),
        "accepted_edge_count": int(len(graph_edges)),
        "smooth_triple_count": int(len(triples)),
        "rejected_pair_count": int(len(rejected_pairs)),
        "rejected_edge_count": int(len(rejected_edges)),
        "pair_residual_before_m": summarize(before_pair),
        "pair_residual_after_m": summarize(after_pair),
        "correction_displacement_m": summarize(correction),
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "parameters": {
            "max_pair_residual_m": float(args.max_pair_residual_m),
            "max_duplicate_spread_m": float(args.max_duplicate_spread_m),
            "observation_sigma_m": float(args.observation_sigma_m),
            "factor_sigma_m": float(args.factor_sigma_m),
            "smooth_sigma_m": float(args.smooth_sigma_m),
            "max_smooth_gap_frames": int(args.max_smooth_gap_frames),
            "loss": str(args.loss),
            "loss_f_scale": float(args.loss_f_scale),
        },
        "positions_npz": str(positions_npz),
        "nodes": node_rows,
        "edges": graph_edges,
        "smooth_triples": triples,
        "rejected_pairs": rejected_pairs,
        "rejected_edges": rejected_edges,
    }
    output_json = args.output_dir / "qc_visibility_surfel_graph_v7.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"nodes", "edges", "smooth_triples", "rejected_pairs", "rejected_edges"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-pair-residual-m", type=float, default=0.012)
    parser.add_argument("--max-duplicate-spread-m", type=float, default=0.002)
    parser.add_argument("--observation-sigma-m", type=float, default=0.002)
    parser.add_argument("--factor-sigma-m", type=float, default=0.006)
    parser.add_argument("--smooth-sigma-m", type=float, default=0.020)
    parser.add_argument("--max-smooth-gap-frames", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=24)
    parser.add_argument("--min-edges", type=int, default=24)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--loss-f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--xtol", type=float, default=1e-10)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--gtol", type=float, default=1e-10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
