#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"bad vertex line in {path}: {line.strip()}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                face = [int(part.split("/")[0]) - 1 for part in parts]
                for j in range(1, len(face) - 1):
                    faces.append([face[0], face[j], face[j + 1]])
    v = np.asarray(vertices, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise RuntimeError(f"{path} contains no 3D vertices")
    if f_arr.ndim != 2 or f_arr.shape[1] != 3 or len(f_arr) == 0:
        raise RuntimeError(f"{path} contains no triangular faces")
    if f_arr.min() < 0 or f_arr.max() >= len(v):
        raise RuntimeError(f"{path} has face indices outside the vertex array")
    return v, f_arr


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homog.T).T[:, :3]


def save_mesh_archive(
    path: Path,
    frame_indices: list[int],
    vertices_per_frame: list[np.ndarray],
    faces_per_frame: list[np.ndarray],
) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    for vertices, faces in zip(vertices_per_frame, faces_per_frame):
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_per_frame).astype(np.float32),
        faces=np.vstack(faces_per_frame).astype(np.int32),
    )


def mesh_extent(vertices: np.ndarray) -> np.ndarray:
    return vertices.max(axis=0) - vertices.min(axis=0)


def run(args: argparse.Namespace) -> None:
    annotations = load_json(args.annotations)
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("annotations JSON must contain nonempty frames list")
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}

    manifest = load_json(args.manifest)
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("BundleSDF manifest must contain nonempty frames list")

    mesh_path = args.mesh
    if mesh_path is None:
        candidates = [
            args.bundlesdf_output / "mesh" / "mesh_real_scale.obj",
            args.bundlesdf_output / "textured_mesh.obj",
            args.bundlesdf_output / "mesh_cleaned.obj",
            args.bundlesdf_output / "mesh" / "mesh_biggest_component_smoothed.obj",
            args.bundlesdf_output / "mesh" / "mesh_biggest_component.obj",
        ]
        mesh_path = next((path for path in candidates if path.exists()), None)
        if mesh_path is None:
            raise RuntimeError(f"no BundleSDF mesh found under {args.bundlesdf_output}")
    if not mesh_path.exists():
        raise RuntimeError(f"BundleSDF mesh does not exist: {mesh_path}")
    canonical_vertices, canonical_faces = load_obj_mesh(mesh_path)

    frame_indices: list[int] = []
    vertices_per_frame: list[np.ndarray] = []
    faces_per_frame: list[np.ndarray] = []
    pose_files: list[str] = []
    missing_pose: list[str] = []
    extents: list[np.ndarray] = []

    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        pose_index = int(entry.get("source_index", entry["index"]))
        annotation = frame_by_idx.get(frame_idx)
        if annotation is None:
            raise RuntimeError(f"frame {frame_idx} from manifest is missing in annotations")
        pose_path = args.bundlesdf_output / "ob_in_cam" / f"{pose_index:06d}.txt"
        if not pose_path.exists():
            missing_pose.append(str(pose_path))
            continue
        ob_in_cam = np.loadtxt(pose_path).astype(np.float64)
        if ob_in_cam.shape != (4, 4):
            raise RuntimeError(f"pose file {pose_path} must be 4x4")
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T_world_camera.shape != (4, 4):
            raise RuntimeError(f"frame {frame_idx} T_world_camera_metric must be 4x4")
        T_world_object = T_world_camera @ ob_in_cam
        world_vertices = transform_points(canonical_vertices, T_world_object)
        frame_indices.append(frame_idx)
        vertices_per_frame.append(world_vertices)
        faces_per_frame.append(canonical_faces)
        pose_files.append(str(pose_path))
        extents.append(mesh_extent(world_vertices))

    if missing_pose:
        raise RuntimeError(f"missing BundleSDF pose files: {missing_pose[:5]}")
    if len(frame_indices) != len(entries):
        raise RuntimeError(f"adapted {len(frame_indices)} frames but manifest has {len(entries)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "bundlesdf_object_meshes_world.npz"
    save_mesh_archive(archive_path, frame_indices, vertices_per_frame, faces_per_frame)

    ext = np.asarray(extents, dtype=np.float64)
    report = {
        "status": "ok",
        "method": "bundlesdf_to_world_mesh_archive_v3",
        "bundlesdf_output": str(args.bundlesdf_output),
        "mesh": str(mesh_path),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "archive": str(archive_path),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "canonical_vertices": int(len(canonical_vertices)),
        "canonical_faces": int(len(canonical_faces)),
        "canonical_extent_m": mesh_extent(canonical_vertices).astype(float).tolist(),
        "world_extent_median_m": np.median(ext, axis=0).astype(float).tolist(),
        "world_extent_min_m": ext.min(axis=0).astype(float).tolist(),
        "world_extent_max_m": ext.max(axis=0).astype(float).tolist(),
        "pose_files": pose_files,
    }
    (args.output_dir / "qc_bundlesdf_mesh_archive_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundlesdf-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
