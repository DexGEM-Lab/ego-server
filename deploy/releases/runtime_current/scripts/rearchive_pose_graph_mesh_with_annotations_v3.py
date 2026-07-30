#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


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
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)


def annotation_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def transform_camera(points: np.ndarray, pivot: np.ndarray, rotvec: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    return float(scale) * ((points - pivot) @ Rotation.from_rotvec(rotvec).as_matrix().T) + pivot + translation


def transform_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def save_archive(path: Path, frame_indices: list[int], vertices_per_frame: list[np.ndarray], faces: np.ndarray) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    face_blocks = []
    for vertices in vertices_per_frame:
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
        face_blocks.append(faces.astype(np.int32))
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_per_frame).astype(np.float32),
        faces=np.vstack(face_blocks).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    qc = load_json(args.pose_graph_qc)
    metrics = qc.get(args.metrics_key)
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError(f"{args.pose_graph_qc} missing metrics object {args.metrics_key}")
    mesh_path = Path(str(qc.get("mesh_prior_camera", args.mesh_prior_camera or "")))
    if args.mesh_prior_camera is not None:
        mesh_path = args.mesh_prior_camera
    if not mesh_path.exists():
        raise RuntimeError(f"mesh prior does not exist: {mesh_path}")
    mesh = load_mesh(mesh_path)
    vertices_camera_anchor = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pivot = np.median(vertices_camera_anchor, axis=0)
    annotations = annotation_by_frame(args.annotations)
    scale = float(qc.get("scale", 1.0))
    frame_indices = [int(idx) for idx in qc.get("used_frames", [])]
    if not frame_indices:
        frame_indices = sorted(int(idx) for idx in metrics)
    vertices_world = []
    rows = []
    for frame_idx in frame_indices:
        key = str(frame_idx)
        if key not in metrics:
            raise RuntimeError(f"metrics missing frame {frame_idx}")
        if frame_idx not in annotations:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        row = metrics[key]
        rotvec = np.asarray(row["rotation_delta_rad"], dtype=np.float64)
        translation = np.asarray(row["translation_camera_delta_m"], dtype=np.float64)
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
            raise RuntimeError(f"frame {frame_idx} camera transform must be finite 4x4")
        camera_vertices = transform_camera(vertices_camera_anchor, pivot, rotvec, translation, scale)
        world_vertices = transform_world(camera_vertices, T_world_camera)
        vertices_world.append(world_vertices.astype(np.float32))
        rows.append(
            {
                "frame_idx": frame_idx,
                "center_world_m": np.median(world_vertices, axis=0).astype(float).tolist(),
                "extent_world_m": (world_vertices.max(axis=0) - world_vertices.min(axis=0)).astype(float).tolist(),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "rearchived_pose_graph_object_meshes_world.npz"
    save_archive(archive, frame_indices, vertices_world, faces)
    report = {
        "status": "ok",
        "method": "rearchive_pose_graph_mesh_with_annotations_v3",
        "pose_graph_qc": str(args.pose_graph_qc),
        "annotations": str(args.annotations),
        "mesh_prior_camera": str(mesh_path),
        "metrics_key": str(args.metrics_key),
        "archive": str(archive),
        "frames": int(len(frame_indices)),
        "scale": float(scale),
        "rows": rows,
    }
    (args.output_dir / "qc_rearchive_pose_graph_mesh_with_annotations_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-graph-qc", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-prior-camera", type=Path)
    parser.add_argument("--metrics-key", default="frame_metrics_after")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
