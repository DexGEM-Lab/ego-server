#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from align_mesh_prior_v3 import choose_alignment, sample_mesh_surface
from archive_aligned_mesh_prior_v7 import load_triangle_mesh, summarize, write_mesh_archive
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def mesh4d_output_root(report: dict) -> Path:
    initial_mesh = Path(str(report.get("initial_mesh", "")))
    if not initial_mesh.name:
        raise RuntimeError("Mesh4D report has no initial_mesh path for output-root mapping")
    return initial_mesh.parent


def resolve_mesh_path(mesh_path: Path, remote_output_root: Path, local_output_root: Path) -> tuple[Path, dict]:
    if mesh_path.exists():
        return mesh_path, {"path_resolution": "as_reported", "reported_mesh_path": str(mesh_path)}
    try:
        relative = mesh_path.relative_to(remote_output_root)
    except ValueError as exc:
        raise RuntimeError(f"Mesh4D mesh path is outside reported output root: {mesh_path}") from exc
    local_mesh_path = local_output_root / relative
    if not local_mesh_path.exists():
        raise RuntimeError(f"Mesh4D mesh path does not exist after output-root relocation: {local_mesh_path}")
    return local_mesh_path, {
        "path_resolution": "relocated_output_root",
        "reported_mesh_path": str(mesh_path),
        "reported_output_root": str(remote_output_root),
        "local_output_root": str(local_output_root),
    }


def mesh4d_frames(report: dict, report_path: Path) -> list[dict]:
    if report.get("status") != "ok":
        raise RuntimeError(f"Mesh4D report status is not ok: {report.get('status')}")
    rows = report.get("frames")
    if not isinstance(rows, list) or len(rows) != 6:
        raise RuntimeError("Mesh4D report must contain exactly six frame rows")
    remote_output_root = mesh4d_output_root(report)
    local_output_root = report_path.parent
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Mesh4D frame row is not an object")
        frame_idx = int(row["frame_idx"])
        mesh_path, path_row = resolve_mesh_path(Path(str(row["mesh"])), remote_output_root, local_output_root)
        parsed.append({**row, **path_row, "frame_idx": frame_idx, "mesh_path": mesh_path})
    return sorted(parsed, key=lambda row: int(row["sequence_index"]))


def align_frame(prior_mesh: trimesh.Trimesh, observed_vertices: np.ndarray, observed_faces: np.ndarray, frame_idx: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    observed_mesh = trimesh.Trimesh(vertices=observed_vertices, faces=observed_faces, process=False)
    prior_points = sample_mesh_surface(prior_mesh, int(args.samples), int(args.seed) + int(frame_idx))
    observed_count = min(int(args.samples), max(int(args.samples) // 2, len(observed_faces) * 4))
    observed_points = sample_mesh_surface(observed_mesh, observed_count, int(args.seed) + int(frame_idx) + 100_000)
    sim, alignment = choose_alignment(prior_points, observed_points)
    transformed = sim.apply(np.asarray(prior_mesh.vertices, dtype=np.float64))
    selected = alignment.get("selected", {})
    prior_to_observed_p95 = float(selected.get("p95_prior_to_observed_m", np.nan))
    observed_to_prior_p95 = float(selected.get("p95_observed_to_prior_m", np.nan))
    row = {
        "frame_idx": int(frame_idx),
        "observed_vertices": int(len(observed_vertices)),
        "observed_faces": int(len(observed_faces)),
        "prior_vertices": int(len(prior_mesh.vertices)),
        "prior_faces": int(len(prior_mesh.faces)),
        "selected_alignment": selected,
        "candidate_count": int(len(alignment.get("candidates", []))),
        "visible_surface_coverage_p95_m": observed_to_prior_p95,
        "hidden_surface_conflict_p95_m": prior_to_observed_p95,
        "bidirectional_p95_m": float(np.nanmax([prior_to_observed_p95, observed_to_prior_p95])),
        "sim3": {
            "scale": float(sim.scale),
            "rotation": np.asarray(sim.rotation, dtype=np.float64).tolist(),
            "translation_m": np.asarray(sim.translation, dtype=np.float64).tolist(),
        },
    }
    return transformed, row


def run(args: argparse.Namespace) -> dict:
    mesh4d_report = load_json(args.mesh4d_json)
    observed_meshes = load_mesh_archive(args.observed_mesh_archive)
    frame_rows = mesh4d_frames(mesh4d_report, args.mesh4d_json)

    rows = []
    archive_rows = []
    for frame_row in frame_rows:
        frame_idx = int(frame_row["frame_idx"])
        if frame_idx not in observed_meshes:
            raise RuntimeError(f"observed archive has no frame {frame_idx}")
        prior_mesh = load_triangle_mesh(frame_row["mesh_path"])
        observed_vertices, observed_faces = observed_meshes[frame_idx]
        aligned_vertices, row = align_frame(prior_mesh, observed_vertices, observed_faces, frame_idx, args)
        row["sequence_index"] = int(frame_row["sequence_index"])
        row["source_index"] = int(frame_row["source_index"])
        row["mesh4d_mesh"] = str(frame_row["mesh_path"])
        row["mesh4d_reported_mesh"] = str(frame_row["reported_mesh_path"])
        row["mesh4d_path_resolution"] = frame_row["path_resolution"]
        rows.append(row)
        archive_rows.append((frame_idx, aligned_vertices, np.asarray(prior_mesh.faces, dtype=np.int32)))

    write_mesh_archive(args.output_mesh_archive, archive_rows)
    p95 = np.asarray([row["bidirectional_p95_m"] for row in rows], dtype=np.float64)
    visible_p95 = np.asarray([row["visible_surface_coverage_p95_m"] for row in rows], dtype=np.float64)
    hidden_p95 = np.asarray([row["hidden_surface_conflict_p95_m"] for row in rows], dtype=np.float64)
    scales = np.asarray([row["sim3"]["scale"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "archive_mesh4d_sequence_prior_v7",
        "claim_tested": "a Mesh4D generated animated object mesh can be aligned per frame to measured object surfaces and then replayed through standard checks",
        "mesh4d_json": str(args.mesh4d_json),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "output_mesh_archive": str(args.output_mesh_archive),
        "frame_count": int(len(rows)),
        "first_frame": int(rows[0]["frame_idx"]),
        "last_frame": int(rows[-1]["frame_idx"]),
        "alignment_bidirectional_p95_m": summarize(p95),
        "visible_surface_coverage_p95_m": summarize(visible_p95),
        "hidden_surface_conflict_p95_m": summarize(hidden_p95),
        "alignment_scale": summarize(scales),
        "acceptance_policy": {
            "strict_full_surface_alignment_pass": bool(np.all(p95 <= float(args.max_bidirectional_p95_m))),
            "visible_surface_coverage_pass": bool(np.all(visible_p95 <= float(args.max_visible_surface_p95_m))),
            "max_bidirectional_p95_m": float(args.max_bidirectional_p95_m),
            "max_visible_surface_p95_m": float(args.max_visible_surface_p95_m),
            "delivery_requires_downstream_replay": [
                "z-buffer silhouette and depth",
                "mesh-surface contact",
                "selected-contact SDF",
                "full-hand SDF",
                "visual review",
            ],
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh4d-json", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-mesh-archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-bidirectional-p95-m", type=float, default=0.01)
    parser.add_argument("--max-visible-surface-p95-m", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
