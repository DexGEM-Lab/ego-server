#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def transform_points(points: np.ndarray, pivot: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot = rodrigues(np.asarray(params[:3], dtype=np.float64))
    trans = np.asarray(params[3:6], dtype=np.float64)
    return (points - pivot) @ rot + pivot + trans


def write_mesh_archive(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    if not rows:
        raise RuntimeError("no meshes to write")
    frame_idx = []
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for frame, vertices, faces in rows:
        frame_idx.append(int(frame))
        vertices_all.append(np.asarray(vertices, dtype=np.float32))
        faces_all.append(np.asarray(faces, dtype=np.int32))
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


def sampled_vertices(vertices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(vertices) <= limit:
        return vertices
    ids = np.linspace(0, len(vertices) - 1, limit, dtype=np.int64)
    return vertices[ids]


def report_pair_rows(pair_report: dict, path: Path) -> list[dict]:
    method = pair_report.get("method")
    rows = pair_report.get("pair_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"pair factor report has no pair_rows list: {path}")
    if method not in {
        "fit_cotracker_pairwise_rigid_factors_v6",
        "merge_cotracker_pair_factors_v6",
    }:
        raise RuntimeError(f"unsupported pair factor report method {method!r}: {path}")
    return rows


def sparse_edges_path(pair_row: dict, root_report: dict, root_path: Path) -> Path:
    source_report = Path(pair_row.get("source_report") or root_path)
    report = root_report if source_report == root_path else load_json(source_report)
    path = report.get("sparse_edges_json")
    if not path:
        raise RuntimeError(f"pair source report lacks sparse_edges_json: {source_report}")
    return Path(path)


def load_pair_edges(edge_json: Path, source_frame: int, target_frame: int) -> list[dict]:
    data = load_json(edge_json)
    if data.get("status") != "ok":
        raise RuntimeError(f"sparse edge report is not ok: {edge_json}")
    edges = data.get("edges")
    if not isinstance(edges, list):
        raise RuntimeError(f"sparse edge report has no edges list: {edge_json}")
    return [
        edge
        for edge in edges
        if int(edge["source_frame"]) == int(source_frame)
        and int(edge["target_frame"]) == int(target_frame)
    ]


def edge_residual(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    row: dict,
    edge: dict,
) -> float:
    source_vertex = int(edge["source_vertex"])
    target_vertex = int(edge["target_vertex"])
    rot = np.asarray(row["rotation"], dtype=np.float64)
    trans = np.asarray(row["translation_m"], dtype=np.float64)
    pred = source_vertices[source_vertex] @ rot + trans
    return float(np.linalg.norm(pred - target_vertices[target_vertex]))


def selected_graph_edges(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    pair_report: dict,
    pair_report_path: Path,
    frames: set[int],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    for row in report_pair_rows(pair_report, pair_report_path):
        if not row.get("rigid_factor_ready"):
            continue
        source_frame = int(row["source_frame"])
        target_frame = int(row["target_frame"])
        if source_frame not in frames or target_frame not in frames:
            continue
        if source_frame not in meshes or target_frame not in meshes:
            raise RuntimeError(f"mesh archive lacks pair {source_frame}->{target_frame}")
        edge_json = sparse_edges_path(row, pair_report, pair_report_path)
        pair_edges = load_pair_edges(edge_json, source_frame, target_frame)
        source_vertices, _ = meshes[source_frame]
        target_vertices, _ = meshes[target_frame]
        kept = []
        for edge in pair_edges:
            residual = edge_residual(source_vertices, target_vertices, row, edge)
            if residual <= float(args.max_edge_residual_m):
                item = dict(edge)
                item["pair_residual_m"] = residual
                kept.append(item)
        if len(kept) > int(args.max_edges_per_pair):
            ids = np.linspace(0, len(kept) - 1, int(args.max_edges_per_pair), dtype=np.int64)
            kept = [kept[int(i)] for i in ids]
        summary = {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "source_anchor": str(row.get("source_anchor", "")),
            "sparse_edges_json": str(edge_json),
            "raw_edge_count": int(len(pair_edges)),
            "accepted_edge_count": int(len(kept)),
            "edge_pair_residual_m": summarize(np.asarray([edge["pair_residual_m"] for edge in kept], dtype=np.float64)),
            "factor_inlier_residual_m": row.get("inlier_residual_m", {}),
        }
        if len(kept) < int(args.min_edges_per_pair):
            rejected.append(summary)
            continue
        accepted.append(
            {
                **summary,
                "rotation": row["rotation"],
                "translation_m": row["translation_m"],
                "edges": kept,
            }
        )
    if not accepted:
        raise RuntimeError("no graph edges survived CoTracker factor and mesh-edge validation")
    return accepted, rejected


def unpack_params(x: np.ndarray, frame_count: int) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(frame_count, 6)


def graph_residual(
    x: np.ndarray,
    frames: list[int],
    frame_to_t: dict[int, int],
    pivots: dict[int, np.ndarray],
    observation_points: dict[int, np.ndarray],
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    graph_edges: list[dict],
    args: argparse.Namespace,
) -> np.ndarray:
    params = unpack_params(x, len(frames))
    residuals = []
    obs_sigma = float(args.observation_sigma_m)
    factor_sigma = float(args.factor_sigma_m)
    trans_sigma = float(args.smooth_translation_sigma_m)
    rot_sigma = float(args.smooth_rotation_sigma_rad)

    for frame in frames:
        t = frame_to_t[frame]
        points = observation_points[frame]
        moved = transform_points(points, pivots[frame], params[t])
        residuals.append(((moved - points) / obs_sigma).reshape(-1))

    for item in graph_edges:
        source_frame = int(item["source_frame"])
        target_frame = int(item["target_frame"])
        source_t = frame_to_t[source_frame]
        target_t = frame_to_t[target_frame]
        source_vertices, _ = meshes[source_frame]
        target_vertices, _ = meshes[target_frame]
        rot = np.asarray(item["rotation"], dtype=np.float64)
        trans = np.asarray(item["translation_m"], dtype=np.float64)
        source_ids = np.asarray([int(edge["source_vertex"]) for edge in item["edges"]], dtype=np.int64)
        target_ids = np.asarray([int(edge["target_vertex"]) for edge in item["edges"]], dtype=np.int64)
        source = transform_points(source_vertices[source_ids], pivots[source_frame], params[source_t])
        target = transform_points(target_vertices[target_ids], pivots[target_frame], params[target_t])
        residuals.append(((source @ rot + trans - target) / factor_sigma).reshape(-1))

    for source_frame, target_frame in zip(frames[:-1], frames[1:]):
        if target_frame - source_frame != 1:
            continue
        source_t = frame_to_t[source_frame]
        target_t = frame_to_t[target_frame]
        residuals.append((params[target_t, :3] - params[source_t, :3]) / rot_sigma)
        residuals.append((params[target_t, 3:6] - params[source_t, 3:6]) / trans_sigma)

    return np.concatenate(residuals).astype(np.float64)


def edge_metrics(
    x: np.ndarray,
    frames: list[int],
    frame_to_t: dict[int, int],
    pivots: dict[int, np.ndarray],
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    graph_edges: list[dict],
) -> list[dict]:
    params = unpack_params(x, len(frames))
    rows = []
    for item in graph_edges:
        source_frame = int(item["source_frame"])
        target_frame = int(item["target_frame"])
        source_t = frame_to_t[source_frame]
        target_t = frame_to_t[target_frame]
        source_vertices, _ = meshes[source_frame]
        target_vertices, _ = meshes[target_frame]
        rot = np.asarray(item["rotation"], dtype=np.float64)
        trans = np.asarray(item["translation_m"], dtype=np.float64)
        source_ids = np.asarray([int(edge["source_vertex"]) for edge in item["edges"]], dtype=np.int64)
        target_ids = np.asarray([int(edge["target_vertex"]) for edge in item["edges"]], dtype=np.int64)
        source = transform_points(source_vertices[source_ids], pivots[source_frame], params[source_t])
        target = transform_points(target_vertices[target_ids], pivots[target_frame], params[target_t])
        residual = np.linalg.norm(source @ rot + trans - target, axis=1)
        rows.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "source_anchor": str(item.get("source_anchor", "")),
                "edge_count": int(len(residual)),
                "residual_m": summarize(residual),
            }
        )
    return rows


def correction_metrics(
    x: np.ndarray,
    frames: list[int],
    pivots: dict[int, np.ndarray],
    observation_points: dict[int, np.ndarray],
) -> list[dict]:
    params = unpack_params(x, len(frames))
    rows = []
    for t, frame in enumerate(frames):
        points = observation_points[frame]
        moved = transform_points(points, pivots[frame], params[t])
        displacement = np.linalg.norm(moved - points, axis=1)
        rows.append(
            {
                "frame_idx": int(frame),
                "rotation_rad": float(np.linalg.norm(params[t, :3])),
                "translation_m": float(np.linalg.norm(params[t, 3:6])),
                "sample_displacement_m": summarize(displacement),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    pair_report = load_json(args.pair_factors_json)
    pair_rows = report_pair_rows(pair_report, args.pair_factors_json)
    ready_frames = {
        int(frame)
        for row in pair_rows
        if row.get("rigid_factor_ready")
        for frame in (row["source_frame"], row["target_frame"])
    }
    if args.frame_start is not None:
        ready_frames = {frame for frame in ready_frames if frame >= int(args.frame_start)}
    if args.frame_end is not None:
        ready_frames = {frame for frame in ready_frames if frame <= int(args.frame_end)}
    frames = sorted(frame for frame in ready_frames if frame in meshes)
    if not frames:
        raise RuntimeError("no mesh-backed ready-factor frames selected")
    frame_to_t = {frame: i for i, frame in enumerate(frames)}
    pivots = {frame: np.asarray(meshes[frame][0], dtype=np.float64).mean(axis=0) for frame in frames}
    observation_points = {
        frame: sampled_vertices(np.asarray(meshes[frame][0], dtype=np.float64), int(args.max_observation_vertices))
        for frame in frames
    }
    graph_edges, rejected_edges = selected_graph_edges(meshes, pair_report, args.pair_factors_json, set(frames), args)
    x0 = np.zeros(len(frames) * 6, dtype=np.float64)
    before = graph_residual(x0, frames, frame_to_t, pivots, observation_points, meshes, graph_edges, args)
    result = least_squares(
        lambda x: graph_residual(x, frames, frame_to_t, pivots, observation_points, meshes, graph_edges, args),
        x0,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=int(args.max_nfev),
        verbose=2 if args.verbose else 0,
    )
    after = graph_residual(result.x, frames, frame_to_t, pivots, observation_points, meshes, graph_edges, args)

    params = unpack_params(result.x, len(frames))
    archive_rows = []
    for t, frame in enumerate(frames):
        vertices, faces = meshes[frame]
        corrected = transform_points(np.asarray(vertices, dtype=np.float64), pivots[frame], params[t])
        archive_rows.append((frame, corrected.astype(np.float32), faces))
    mesh_path = args.output_dir / "cotracker_factor_graph_meshes_world.npz"
    write_mesh_archive(mesh_path, archive_rows)

    before_edge_rows = edge_metrics(x0, frames, frame_to_t, pivots, meshes, graph_edges)
    after_edge_rows = edge_metrics(result.x, frames, frame_to_t, pivots, meshes, graph_edges)
    correction_rows = correction_metrics(result.x, frames, pivots, observation_points)
    correction_p95 = [
        row["sample_displacement_m"].get("p95", np.nan)
        for row in correction_rows
    ]
    before_edge_residuals = [
        row["residual_m"].get("p95", np.nan)
        for row in before_edge_rows
    ]
    after_edge_residuals = [
        row["residual_m"].get("p95", np.nan)
        for row in after_edge_rows
    ]
    correction_summary = summarize(np.asarray(correction_p95, dtype=np.float64))
    before_edge_summary = summarize(np.asarray(before_edge_residuals, dtype=np.float64))
    after_edge_summary = summarize(np.asarray(after_edge_residuals, dtype=np.float64))
    before_median = before_edge_summary.get("median")
    after_median = after_edge_summary.get("median")
    correction_max = correction_summary.get("max", np.inf)
    material_improvement = (
        before_median is not None
        and after_median is not None
        and float(after_median) <= 0.90 * float(before_median)
        and float(correction_max) >= float(args.min_material_correction_m)
    )
    status = "diagnostic_factor_compatible_no_material_correction"
    if not result.success:
        status = "diagnostic_optimizer_incomplete"
    elif material_improvement:
        status = "diagnostic_factor_reduced_edge_residual"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "fit_cotracker_factor_graph_v6",
        "claim_tested": "small per-frame SE3 corrections can use graph-ready CoTracker factors as temporal priors while preserving measured per-frame object meshes",
        "mesh_archive": str(args.mesh_archive),
        "pair_factors_json": str(args.pair_factors_json),
        "output_mesh_archive": str(mesh_path),
        "frames": [int(frame) for frame in frames],
        "frame_count": int(len(frames)),
        "accepted_graph_pair_count": int(len(graph_edges)),
        "rejected_graph_pair_count": int(len(rejected_edges)),
        "accepted_graph_edge_count": int(sum(len(item["edges"]) for item in graph_edges)),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        "edge_p95_before_m": before_edge_summary,
        "edge_p95_after_m": after_edge_summary,
        "correction_sample_displacement_p95_m": correction_summary,
        "material_improvement": bool(material_improvement),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "parameters": {
            "max_observation_vertices": int(args.max_observation_vertices),
            "observation_sigma_m": float(args.observation_sigma_m),
            "factor_sigma_m": float(args.factor_sigma_m),
            "max_edge_residual_m": float(args.max_edge_residual_m),
            "min_edges_per_pair": int(args.min_edges_per_pair),
            "max_edges_per_pair": int(args.max_edges_per_pair),
            "smooth_translation_sigma_m": float(args.smooth_translation_sigma_m),
            "smooth_rotation_sigma_rad": float(args.smooth_rotation_sigma_rad),
            "min_material_correction_m": float(args.min_material_correction_m),
            "max_nfev": int(args.max_nfev),
        },
        "accepted_graph_pairs": [
            {key: value for key, value in item.items() if key not in {"edges", "rotation", "translation_m"}}
            for item in graph_edges
        ],
        "rejected_graph_pairs": rejected_edges,
        "edge_metrics_before": before_edge_rows,
        "edge_metrics_after": after_edge_rows,
        "correction_metrics": correction_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = args.output_dir / "qc_cotracker_factor_graph_v6.json"
    qc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    omitted = {"accepted_graph_pairs", "rejected_graph_pairs", "edge_metrics_before", "edge_metrics_after", "correction_metrics"}
    print(json.dumps({k: v for k, v in report.items() if k not in omitted}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-observation-vertices", type=int, default=2000)
    parser.add_argument("--observation-sigma-m", type=float, default=0.003)
    parser.add_argument("--factor-sigma-m", type=float, default=0.006)
    parser.add_argument("--max-edge-residual-m", type=float, default=0.012)
    parser.add_argument("--min-edges-per-pair", type=int, default=8)
    parser.add_argument("--max-edges-per-pair", type=int, default=64)
    parser.add_argument("--smooth-translation-sigma-m", type=float, default=0.010)
    parser.add_argument("--smooth-rotation-sigma-rad", type=float, default=0.050)
    parser.add_argument("--min-material-correction-m", type=float, default=0.001)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
