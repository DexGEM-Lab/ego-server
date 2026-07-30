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
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"{path} did not load as a triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"{path} contains no 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"{path} contains no triangular faces")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError(f"{path} face indices are outside the vertex array")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def select_manifest_frame(manifest: dict, frame_idx: int) -> dict:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("manifest must contain a nonempty frames list")
    selected = [frame for frame in frames if int(frame["frame_idx"]) == int(frame_idx)]
    if len(selected) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(selected)} times in manifest")
    out = dict(manifest)
    out["frames"] = selected
    return out


def robust_extent(vertices: np.ndarray) -> np.ndarray:
    return np.quantile(vertices, 0.95, axis=0) - np.quantile(vertices, 0.05, axis=0)


def frame_annotation(annotations_path: Path, frame_idx: int) -> dict:
    frames = load_json(annotations_path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{annotations_path} must contain a nonempty frames list")
    selected = [frame for frame in frames if int(frame["frame_idx"]) == int(frame_idx)]
    if len(selected) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(selected)} times in annotations")
    return selected[0]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def run(args: argparse.Namespace) -> dict:
    mesh = load_mesh(args.mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    manifest = select_manifest_frame(load_json(args.manifest), int(args.frame_idx))
    coordinate = "world"
    if args.mesh_coordinate == "camera":
        annotation = frame_annotation(args.annotations, int(args.frame_idx))
        transform = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"frame {args.frame_idx} T_world_camera_metric must be a finite 4x4 matrix")
        vertices = transform_points(vertices, transform)
        coordinate = "camera_to_world"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / f"world_mesh_frame_{args.frame_idx:06d}.npz"
    np.savez_compressed(
        archive,
        frame_idx=np.asarray([int(args.frame_idx)], dtype=np.int32),
        vertex_offsets=np.asarray([0, len(vertices)], dtype=np.int64),
        face_offsets=np.asarray([0, len(faces)], dtype=np.int64),
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
    )

    manifest_path = args.output_dir / f"manifest_frame_{args.frame_idx:06d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "status": "ok",
        "method": "package_world_mesh_frame_v3",
        "mesh": str(args.mesh),
        "mesh_coordinate": args.mesh_coordinate,
        "archive_coordinate": coordinate,
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "frame_idx": int(args.frame_idx),
        "archive": str(archive),
        "filtered_manifest": str(manifest_path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "center_world_m": np.median(vertices, axis=0).astype(float).tolist(),
        "extent_world_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
        "robust_extent_world_m": robust_extent(vertices).astype(float).tolist(),
    }
    (args.output_dir / "qc_package_world_mesh_frame_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-coordinate", choices=["world", "camera"], default="world")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
