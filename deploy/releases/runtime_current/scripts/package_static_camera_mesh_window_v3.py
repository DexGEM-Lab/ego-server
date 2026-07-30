#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )


def annotations_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def robust_extent(vertices: np.ndarray) -> np.ndarray:
    return np.quantile(vertices, 0.95, axis=0) - np.quantile(vertices, 0.05, axis=0)


def save_archive(path: Path, frame_indices: list[int], vertices_world: list[np.ndarray], faces: np.ndarray) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    faces_all = []
    for vertices in vertices_world:
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
        faces_all.append(faces)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_world).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    mesh = load_mesh(args.mesh_camera)
    vertices_camera = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    annotations = annotations_by_frame(args.annotations)
    frame_indices = list(range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))))
    if not frame_indices:
        raise RuntimeError("no frames selected")

    rows = []
    vertices_world = []
    for frame_idx in frame_indices:
        frame = annotations.get(frame_idx)
        if frame is None:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        camera = frame.get("camera")
        if not isinstance(camera, dict) or "T_world_camera_metric" not in camera:
            raise RuntimeError(f"frame {frame_idx} missing camera transform")
        transform = np.asarray(camera["T_world_camera_metric"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"frame {frame_idx} camera transform must be finite 4x4")
        world = transform_points(vertices_camera, transform)
        vertices_world.append(world)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "center_world_m": np.median(world, axis=0).astype(float).tolist(),
                "extent_world_m": (world.max(axis=0) - world.min(axis=0)).astype(float).tolist(),
                "robust_extent_world_m": robust_extent(world).astype(float).tolist(),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "static_camera_mesh_window_world.npz"
    save_archive(archive, frame_indices, vertices_world, faces)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "package_static_camera_mesh_window_v3",
        "mesh_camera": str(args.mesh_camera),
        "annotations": str(args.annotations),
        "archive": str(archive),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "camera_mesh_vertices": int(len(vertices_camera)),
        "camera_mesh_faces": int(len(faces)),
        "camera_mesh_robust_extent_m": robust_extent(vertices_camera).astype(float).tolist(),
        "rows": rows,
    }
    report_path = args.output_dir / "qc_static_camera_mesh_window_v3.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-camera", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
