#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from render_mesh_alignment_v3 import draw_mesh, load_mesh, view_basis


def load_point_cloud(path: Path) -> np.ndarray:
    cloud = trimesh.load(path, process=False)
    if isinstance(cloud, trimesh.PointCloud):
        points = np.asarray(cloud.vertices, dtype=float)
    elif isinstance(cloud, trimesh.Trimesh):
        points = np.asarray(cloud.vertices, dtype=float)
    else:
        raise RuntimeError(f"unsupported point cloud object from {path}: {type(cloud)}")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError(f"point cloud is empty: {path}")
    return points


def load_observed_points(path: Path, frame_idx: int) -> np.ndarray:
    blob = np.load(path)
    frames = blob["frame_idx"].astype(int)
    hits = np.where(frames == int(frame_idx))[0]
    if len(hits) != 1:
        raise RuntimeError(f"observed mesh archive lacks frame {frame_idx}")
    i = int(hits[0])
    offsets = blob["vertex_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(float)
    points = vertices[int(offsets[i]) : int(offsets[i + 1])]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError(f"observed mesh frame {frame_idx} is empty")
    return points


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    color: tuple[int, int, int],
    max_points: int,
) -> None:
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    q = (points - center) @ basis.T
    scale = 0.42 * min(image.shape[1], image.shape[0]) / radius
    xy = np.c_[image.shape[1] * 0.5 + q[:, 0] * scale, image.shape[0] * 0.5 - q[:, 1] * scale]
    xy = np.rint(xy).astype(int)
    for x, y in xy:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)


def panel(
    title: str,
    mesh: trimesh.Trimesh | None,
    points: list[tuple[np.ndarray, tuple[int, int, int]]],
    center: np.ndarray,
    basis: np.ndarray,
    radius: float,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.full((height, width, 3), (244, 245, 240), dtype=np.uint8)
    for point_set, color in points:
        draw_points(image, point_set, center, basis, radius, color, 6000)
    if mesh is not None:
        draw_mesh(image, mesh, center, basis, radius, (70, 92, 220), 0.26, 2500)
    cv2.putText(image, title, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (28, 28, 28), 2, cv2.LINE_AA)
    return image


def render(args: argparse.Namespace) -> dict:
    mesh = load_mesh(args.vggt_mesh)
    vggt_points = load_point_cloud(args.vggt_points)
    observed_points = load_observed_points(args.observed_mesh_npz, int(args.observed_frame))
    all_points = np.vstack([np.asarray(mesh.vertices, dtype=float), vggt_points, observed_points])
    center, basis, radius = view_basis(all_points)
    panel_width = args.width // 3
    panels = [
        panel("VGGT mesh", mesh, [], center, basis, radius, panel_width, args.height),
        panel("observed metric-depth surface", None, [(observed_points, (45, 165, 60))], center, basis, radius, panel_width, args.height),
        panel("overlay", mesh, [(observed_points, (45, 165, 60)), (vggt_points, (215, 115, 45))], center, basis, radius, panel_width, args.height),
    ]
    image = np.concatenate(panels, axis=1)
    cv2.putText(image, f"observed frame {int(args.observed_frame):06d}", (24, args.height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"failed to write {args.output}")
    report = {
        "status": "ok",
        "vggt_mesh": str(args.vggt_mesh),
        "vggt_points": str(args.vggt_points),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "observed_frame": int(args.observed_frame),
        "output": str(args.output),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "vggt_points_count": int(len(vggt_points)),
        "observed_points_count": int(len(observed_points)),
    }
    (args.output.parent / "qc_vggt_mesh_review_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-mesh", type=Path, required=True)
    parser.add_argument("--vggt-points", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--observed-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
