#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return mesh


def view_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh
    if np.linalg.det(basis) < 0:
        basis[-1] *= -1.0
    q = centered @ basis.T
    radius = float(np.percentile(np.linalg.norm(q, axis=1), 98.0))
    return center, basis, max(radius, 1e-4)


def project(points: np.ndarray, center: np.ndarray, basis: np.ndarray, radius: float, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    q = (points - center) @ basis.T
    scale = 0.42 * min(width, height) / radius
    x = width * 0.5 + q[:, 0] * scale
    y = height * 0.5 - q[:, 1] * scale
    return np.c_[x, y].astype(np.int32), q[:, 2]


def draw_mesh(
    image: np.ndarray,
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    alpha: float,
    edge_budget: int,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    xy, depth = project(vertices, center, basis, radius, (image.shape[1], image.shape[0]))
    face_depth = depth[faces].mean(axis=1)
    order = np.argsort(face_depth)
    overlay = image.copy()
    for face_id in order:
        poly = xy[faces[int(face_id)]]
        if np.any(poly[:, 0] < -500) or np.any(poly[:, 0] > image.shape[1] + 500):
            continue
        if np.any(poly[:, 1] < -500) or np.any(poly[:, 1] > image.shape[0] + 500):
            continue
        cv2.fillConvexPoly(overlay, poly, color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
    edge_count = min(len(faces), edge_budget)
    for face_id in np.linspace(0, len(faces) - 1, edge_count, dtype=int):
        poly = xy[faces[int(face_id)]]
        cv2.polylines(image, [poly], True, color, 1, cv2.LINE_AA)


def panel(title: str, meshes: list[tuple[str, trimesh.Trimesh]], center: np.ndarray, basis: np.ndarray, radius: float, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), (244, 245, 240), dtype=np.uint8)
    palette = {
        "prior": ((45, 145, 220), 0.18, 2200),
        "observed": ((45, 165, 60), 0.38, 1600),
    }
    for name, mesh in meshes:
        color, alpha, edge_budget = palette[name]
        draw_mesh(image, mesh, center, basis, radius, color, alpha, edge_budget)
    cv2.putText(image, title, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def render(args: argparse.Namespace) -> dict:
    prior = load_mesh(args.aligned_prior)
    observed = load_mesh(args.observed_surface)
    points = np.vstack([np.asarray(prior.vertices), np.asarray(observed.vertices)])
    center, basis, radius = view_basis(points)
    panel_width = args.width // 3
    panel_height = args.height
    panels = [
        panel("observed metric-depth surface", [("observed", observed)], center, basis, radius, panel_width, panel_height),
        panel("TripoSR complete prior", [("prior", prior)], center, basis, radius, panel_width, panel_height),
        panel("overlay", [("observed", observed), ("prior", prior)], center, basis, radius, panel_width, panel_height),
    ]
    image = np.concatenate(panels, axis=1)
    cv2.putText(image, "green = observed surface", (24, args.height - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 165, 60), 2, cv2.LINE_AA)
    cv2.putText(image, "orange = complete mesh prior", (24, args.height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 145, 220), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    report = {
        "status": "ok",
        "aligned_prior": str(args.aligned_prior),
        "observed_surface": str(args.observed_surface),
        "output": str(args.output),
        "prior_vertices": int(len(prior.vertices)),
        "prior_faces": int(len(prior.faces)),
        "observed_vertices": int(len(observed.vertices)),
        "observed_faces": int(len(observed.faces)),
    }
    (args.output.parent / "qc_render_mesh_alignment_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-prior", type=Path, required=True)
    parser.add_argument("--observed-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
