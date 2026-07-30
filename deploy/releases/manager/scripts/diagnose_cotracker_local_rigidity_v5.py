#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_edges(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("edges"), list):
        raise RuntimeError(f"sparse edge report has no edges list: {path}")
    return data


def kabsch_rmsd(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("invalid Kabsch input")
    if len(source) < 3:
        raise RuntimeError("Kabsch requires at least 3 points")
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    src = source - src_center
    tgt = target - tgt_center
    u, _s, vt = np.linalg.svd(src.T @ tgt)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    trans = tgt_center - src_center @ rot
    aligned = source @ rot + trans
    residual = np.linalg.norm(aligned - target, axis=1)
    return rot, trans, float(np.sqrt(np.mean(np.square(residual)))), residual


def run(args: argparse.Namespace) -> dict:
    tracks = np.load(args.cotracker_npz)
    frame_idx = np.asarray(tracks["frame_idx"], dtype=np.int64)
    accepted = np.asarray(tracks["accepted"], dtype=bool)
    world = np.asarray(tracks["world_xyz"], dtype=np.float64)
    edges_report = load_edges(args.sparse_edges_json)
    edge_track_ids = sorted({int(edge["track_id"]) for edge in edges_report["edges"]})
    if not edge_track_ids:
        raise RuntimeError("no sparse edge track ids")
    usable = np.zeros((accepted.shape[1],), dtype=bool)
    usable[np.asarray(edge_track_ids, dtype=np.int64)] = True

    pair_rows = []
    residuals_all = []
    length_errors_all = []
    for i in range(len(frame_idx) - 1):
        a = int(frame_idx[i])
        b = int(frame_idx[i + 1])
        ids = np.where(usable & accepted[i] & accepted[i + 1] & np.all(np.isfinite(world[i]), axis=1) & np.all(np.isfinite(world[i + 1]), axis=1))[0]
        if len(ids) < int(args.min_pair_tracks):
            pair_rows.append({"source_frame": a, "target_frame": b, "track_count": int(len(ids)), "status": "too_few_tracks"})
            continue
        src = world[i, ids]
        tgt = world[i + 1, ids]
        _rot, _trans, rmsd, residual = kabsch_rmsd(src, tgt)
        residuals_all.extend(residual.astype(float).tolist())
        if len(ids) > int(args.max_pairwise_tracks):
            ids_for_pairs = ids[: int(args.max_pairwise_tracks)]
            src_pairs = world[i, ids_for_pairs]
            tgt_pairs = world[i + 1, ids_for_pairs]
        else:
            src_pairs = src
            tgt_pairs = tgt
        pair_i, pair_j = np.triu_indices(len(src_pairs), k=1)
        src_len = np.linalg.norm(src_pairs[pair_i] - src_pairs[pair_j], axis=1)
        tgt_len = np.linalg.norm(tgt_pairs[pair_i] - tgt_pairs[pair_j], axis=1)
        ok = (src_len >= float(args.min_pair_distance_m)) & (tgt_len >= float(args.min_pair_distance_m))
        length_error = np.abs(tgt_len[ok] - src_len[ok])
        length_errors_all.extend(length_error.astype(float).tolist())
        pair_rows.append(
            {
                "source_frame": a,
                "target_frame": b,
                "track_count": int(len(ids)),
                "rigid_rmsd_m": rmsd,
                "rigid_residual_m": summarize(residual),
                "pairwise_length_error_m": summarize(length_error),
                "status": "ok",
            }
        )

    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "diagnose_cotracker_local_rigidity_v5",
        "claim_tested": "sparse learned object tracks preserve enough local metric structure to justify use as material-correspondence factors",
        "cotracker_npz": str(args.cotracker_npz),
        "sparse_edges_json": str(args.sparse_edges_json),
        "frames": [int(frame) for frame in frame_idx.tolist()],
        "usable_track_count": int(np.count_nonzero(usable)),
        "pair_rows": pair_rows,
        "rigid_residual_m": summarize(np.asarray(residuals_all, dtype=np.float64)),
        "pairwise_length_error_m": summarize(np.asarray(length_errors_all, dtype=np.float64)),
        "parameters": {
            "min_pair_tracks": int(args.min_pair_tracks),
            "max_pairwise_tracks": int(args.max_pairwise_tracks),
            "min_pair_distance_m": float(args.min_pair_distance_m),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument("--sparse-edges-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-pair-tracks", type=int, default=12)
    parser.add_argument("--max-pairwise-tracks", type=int, default=80)
    parser.add_argument("--min-pair-distance-m", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
