#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(points, dtype=float) @ self.rotation.T) + self.translation


def load_observed_frame(mesh_npz: Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(mesh_npz)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"mesh archive missing keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    matches = np.flatnonzero(frames == int(frame_idx))
    if len(matches) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(matches)} times in {mesh_npz}")
    i = int(matches[0])
    v0, v1 = int(blob["vertex_offsets"][i]), int(blob["vertex_offsets"][i + 1])
    f0, f1 = int(blob["face_offsets"][i]), int(blob["face_offsets"][i + 1])
    vertices = blob["vertices"][v0:v1].astype(float)
    faces = blob["faces"][f0:f1].astype(np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"observed mesh for frame {frame_idx} is empty")
    return vertices, faces


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        raise RuntimeError("sample count must be positive")
    rng = np.random.default_rng(seed)
    if len(mesh.faces) == 0:
        raise RuntimeError("mesh has no faces")
    areas = mesh.area_faces.astype(float)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        raise RuntimeError("mesh face areas are invalid")
    face_ids = rng.choice(len(mesh.faces), size=count, replace=True, p=areas / areas.sum())
    tri = mesh.vertices[mesh.faces[face_ids]].astype(float)
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1.0
    return center, axes, singular


def nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy is required for nearest-neighbor alignment") from exc
    tree = cKDTree(dst)
    d, _ = tree.query(src, k=1)
    return d.astype(float)


def robust_extent(points: np.ndarray) -> np.ndarray:
    lo = np.percentile(points, 5.0, axis=0)
    hi = np.percentile(points, 95.0, axis=0)
    extent = hi - lo
    if np.any(~np.isfinite(extent)) or np.any(extent <= 1e-8):
        raise RuntimeError("point extent is degenerate")
    return extent


def candidate_sim3(prior_points: np.ndarray, observed_points: np.ndarray) -> list[Sim3]:
    prior_center, prior_axes, prior_singular = pca_frame(prior_points)
    observed_center, observed_axes, observed_singular = pca_frame(observed_points)
    scale = float(np.median(robust_extent(observed_points)) / np.median(robust_extent(prior_points)))
    sims = []
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        sign = np.diag(np.asarray(signs, dtype=float))
        rotation = observed_axes.T @ sign @ prior_axes
        if np.linalg.det(rotation) < 0:
            continue
        translation = observed_center - scale * (prior_center @ rotation.T)
        sims.append(Sim3(scale=scale, rotation=rotation, translation=translation))
    if not sims:
        raise RuntimeError("no valid similarity candidates")
    return sims


def choose_alignment(prior_points: np.ndarray, observed_points: np.ndarray) -> tuple[Sim3, dict]:
    best = None
    records = []
    for sim in candidate_sim3(prior_points, observed_points):
        transformed = sim.apply(prior_points)
        d_prior_to_obs = nearest_distances(transformed, observed_points)
        d_obs_to_prior = nearest_distances(observed_points, transformed)
        score = float(np.median(d_prior_to_obs) + np.median(d_obs_to_prior))
        record = {
            "scale": float(sim.scale),
            "median_prior_to_observed_m": float(np.median(d_prior_to_obs)),
            "median_observed_to_prior_m": float(np.median(d_obs_to_prior)),
            "p95_prior_to_observed_m": float(np.percentile(d_prior_to_obs, 95.0)),
            "p95_observed_to_prior_m": float(np.percentile(d_obs_to_prior, 95.0)),
            "score": score,
        }
        records.append(record)
        if best is None or score < best[0]:
            best = (score, sim, record)
    assert best is not None
    return best[1], {"selected": best[2], "candidates": records}


def run(args: argparse.Namespace) -> dict:
    prior_mesh = trimesh.load(args.mesh_prior, force="mesh", process=False)
    if not isinstance(prior_mesh, trimesh.Trimesh) or len(prior_mesh.vertices) == 0 or len(prior_mesh.faces) == 0:
        raise RuntimeError(f"invalid prior mesh: {args.mesh_prior}")
    observed_vertices, observed_faces = load_observed_frame(args.observed_mesh_npz, args.frame_idx)
    observed_mesh = trimesh.Trimesh(vertices=observed_vertices, faces=observed_faces, process=False)
    prior_points = sample_mesh_surface(prior_mesh, args.samples, args.seed)
    observed_points = sample_mesh_surface(observed_mesh, min(args.samples, max(args.samples // 2, len(observed_faces) * 4)), args.seed + 1)
    sim, report = choose_alignment(prior_points, observed_points)
    aligned_vertices = sim.apply(np.asarray(prior_mesh.vertices, dtype=float))
    aligned_mesh = trimesh.Trimesh(vertices=aligned_vertices, faces=np.asarray(prior_mesh.faces), process=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = args.output_dir / f"aligned_prior_frame_{args.frame_idx:06d}.obj"
    aligned_mesh.export(out_mesh)
    observed_out = args.output_dir / f"observed_surface_frame_{args.frame_idx:06d}.ply"
    observed_mesh.export(observed_out)
    report.update(
        {
            "status": "ok",
            "mesh_prior": str(args.mesh_prior),
            "observed_mesh_npz": str(args.observed_mesh_npz),
            "frame_idx": int(args.frame_idx),
            "prior_vertices": int(len(prior_mesh.vertices)),
            "prior_faces": int(len(prior_mesh.faces)),
            "observed_vertices": int(len(observed_vertices)),
            "observed_faces": int(len(observed_faces)),
            "aligned_mesh": str(out_mesh),
            "observed_surface_mesh": str(observed_out),
            "sim3": {
                "scale": float(sim.scale),
                "rotation": sim.rotation.astype(float).tolist(),
                "translation": sim.translation.astype(float).tolist(),
            },
        }
    )
    (args.output_dir / "qc_align_mesh_prior_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=858)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
