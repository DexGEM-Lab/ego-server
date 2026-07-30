#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def annotations_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames")
    return {int(frame["frame_idx"]): frame for frame in frames}


def transform_points(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def robust_extent(vertices: np.ndarray) -> np.ndarray:
    return np.quantile(vertices, 0.95, axis=0) - np.quantile(vertices, 0.05, axis=0)


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    blob = np.load(args.camera_mesh_archive)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{args.camera_mesh_archive} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices_camera_all = blob["vertices"].astype(np.float64)
    faces_all = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError("archive offsets do not match frame count")
    out_vertices = []
    out_faces = []
    out_vertex_offsets = [0]
    out_face_offsets = [0]
    rows = []
    for i, idx in enumerate(frame_idx.tolist()):
        if idx < int(args.frame_start) or idx > int(args.frame_end):
            continue
        if idx not in annotations:
            raise RuntimeError(f"annotations missing frame {idx}")
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        vertices_camera = vertices_camera_all[v0:v1]
        faces = faces_all[f0:f1]
        if len(vertices_camera) == 0 or len(faces) == 0:
            raise RuntimeError(f"empty camera mesh for frame {idx}")
        if faces.min() < 0 or faces.max() >= len(vertices_camera):
            raise RuntimeError(f"face index out of range for frame {idx}")
        T = np.asarray(annotations[idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"frame {idx} camera transform must be finite 4x4")
        vertices_world = transform_points(vertices_camera, T)
        out_vertices.append(vertices_world.astype(np.float32))
        out_faces.append(faces.astype(np.int32))
        out_vertex_offsets.append(out_vertex_offsets[-1] + len(vertices_world))
        out_face_offsets.append(out_face_offsets[-1] + len(faces))
        rows.append(
            {
                "frame_idx": int(idx),
                "vertices": int(len(vertices_world)),
                "faces": int(len(faces)),
                "center_world_m": np.median(vertices_world, axis=0).astype(float).tolist(),
                "robust_extent_world_m": robust_extent(vertices_world).astype(float).tolist(),
                "camera_robust_extent_m": robust_extent(vertices_camera).astype(float).tolist(),
            }
        )
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} mesh frames selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "camera_mesh_archive_world.npz"
    np.savez_compressed(
        archive,
        frame_idx=np.asarray([row["frame_idx"] for row in rows], dtype=np.int32),
        vertex_offsets=np.asarray(out_vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(out_face_offsets, dtype=np.int64),
        vertices=np.vstack(out_vertices).astype(np.float32),
        faces=np.vstack(out_faces).astype(np.int32),
    )
    centers = np.asarray([row["center_world_m"] for row in rows], dtype=np.float64)
    speeds = []
    if len(centers) > 1:
        frame_steps = np.diff(np.asarray([row["frame_idx"] for row in rows], dtype=np.float64))
        speeds = (np.linalg.norm(np.diff(centers, axis=0), axis=1) * float(args.fps) / np.maximum(frame_steps, 1e-9)).tolist()
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "transform_camera_mesh_archive_to_world_v3",
        "camera_mesh_archive": str(args.camera_mesh_archive),
        "annotations": str(args.annotations),
        "archive": str(archive),
        "frames": int(len(rows)),
        "first_frame": int(rows[0]["frame_idx"]),
        "last_frame": int(rows[-1]["frame_idx"]),
        "object_center_speed_m_s": summarize(speeds),
        "rows": rows,
    }
    (args.output_dir / "qc_transform_camera_mesh_archive_to_world_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
