#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    pair_report_in = require_file(args.pair_factors_json, "pair_factors_json")
    sparse_edges_in = require_file(args.sparse_edges_json, "sparse_edges_json")
    observed_mesh_archive = require_file(args.observed_mesh_archive, "observed_mesh_archive")
    tracks_npz = require_file(args.cotracker_npz, "cotracker_npz")

    pair_report = load_json(pair_report_in)
    sparse_edges = load_json(sparse_edges_in)
    if pair_report.get("method") != "fit_cotracker_pairwise_rigid_factors_v6":
        raise RuntimeError(f"unsupported pair report method: {pair_report.get('method')!r}")
    if sparse_edges.get("method") != "build_cotracker_sparse_correspondence_edges_v5":
        raise RuntimeError(f"unsupported sparse edge method: {sparse_edges.get('method')!r}")

    sparse_edges_out = args.output_dir / "sparse_edges" / sparse_edges_in.name
    pair_report_out = args.output_dir / "pair_factors" / pair_report_in.name
    sparse_edges["cotracker_npz"] = str(tracks_npz)
    sparse_edges["mesh_archive"] = str(observed_mesh_archive)
    pair_report["cotracker_npz"] = str(tracks_npz)
    pair_report["sparse_edges_json"] = str(sparse_edges_out)
    write_json(sparse_edges_out, sparse_edges)
    write_json(pair_report_out, pair_report)

    report = {
        "status": "ok",
        "method": "localize_v7_cotracker_factor_report",
        "claim_tested": "synced remote CoTracker factor reports can be made local without changing track, edge, or pair-factor measurements",
        "inputs": {
            "pair_factors_json": str(pair_report_in),
            "sparse_edges_json": str(sparse_edges_in),
            "cotracker_npz": str(tracks_npz),
            "observed_mesh_archive": str(observed_mesh_archive),
        },
        "outputs": {
            "pair_factors_json": str(pair_report_out),
            "sparse_edges_json": str(sparse_edges_out),
        },
        "pair_count": pair_report.get("pair_count"),
        "rigid_factor_ready_pairs": pair_report.get("rigid_factor_ready_pairs"),
        "edge_count": sparse_edges.get("edge_count"),
    }
    output_json = args.output_dir / "qc_localized_cotracker_factor_report_v7.json"
    write_json(output_json, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-factors-json", type=Path, required=True)
    parser.add_argument("--sparse-edges-json", type=Path, required=True)
    parser.add_argument("--cotracker-npz", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
