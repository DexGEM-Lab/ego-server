#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mesh_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    if len(vertex_offsets) != len(frame_idx) + 1 or len(face_offsets) != len(frame_idx) + 1:
        raise RuntimeError("mesh archive offsets do not match frame_idx length")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        frame_vertices = vertices[v0:v1]
        frame_faces = faces[f0:f1]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"empty mesh for frame {idx}")
        if frame_faces.min() < 0 or frame_faces.max() >= len(frame_vertices):
            raise RuntimeError(f"face index out of range for frame {idx}")
        out[int(idx)] = (frame_vertices, frame_faces)
    return out


def frame_map(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def to_camera(vertices_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[vertices_world, np.ones(len(vertices_world), dtype=np.float64)]
    return (np.linalg.inv(T_world_camera) @ homog.T).T[:, :3]


def robust_extent(points: np.ndarray, q: float) -> np.ndarray:
    lo = np.quantile(points, float(q), axis=0)
    hi = np.quantile(points, 1.0 - float(q), axis=0)
    return hi - lo


def summarize(values: list[float] | np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def rows_from_meshes(meshes: dict[int, tuple[np.ndarray, np.ndarray]], annotations: dict[int, dict], args: argparse.Namespace) -> list[dict]:
    rows = []
    for frame_idx in sorted(meshes):
        vertices, faces = meshes[frame_idx]
        if args.annotations is not None:
            if frame_idx not in annotations:
                raise RuntimeError(f"annotations missing frame {frame_idx}")
            T = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
            camera_vertices = to_camera(vertices, T)
        else:
            camera_vertices = vertices
        row = {
            "frame_idx": int(frame_idx),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "center_world_m": np.median(vertices, axis=0).astype(float).tolist(),
            "extent_world_m": (vertices.max(axis=0) - vertices.min(axis=0)).astype(float).tolist(),
            "robust_extent_world_m": robust_extent(vertices, float(args.robust_quantile)).astype(float).tolist(),
            "center_camera_m": np.median(camera_vertices, axis=0).astype(float).tolist(),
            "extent_camera_m": (camera_vertices.max(axis=0) - camera_vertices.min(axis=0)).astype(float).tolist(),
            "robust_extent_camera_m": robust_extent(camera_vertices, float(args.robust_quantile)).astype(float).tolist(),
        }
        rows.append(row)
    return rows


def add_pairwise(rows: list[dict], fps: float) -> list[dict]:
    pairs = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        a = np.asarray(prev["center_world_m"], dtype=np.float64)
        b = np.asarray(cur["center_world_m"], dtype=np.float64)
        dt = (int(cur["frame_idx"]) - int(prev["frame_idx"])) / float(fps)
        if dt <= 0.0:
            raise RuntimeError("frame indices must be strictly increasing")
        extent_prev = np.asarray(prev["robust_extent_camera_m"], dtype=np.float64)
        extent_cur = np.asarray(cur["robust_extent_camera_m"], dtype=np.float64)
        scale_ratio = extent_cur / np.maximum(extent_prev, 1e-9)
        pairs.append(
            {
                "from_frame": int(prev["frame_idx"]),
                "to_frame": int(cur["frame_idx"]),
                "center_step_world_m": float(np.linalg.norm(b - a)),
                "center_speed_world_m_s": float(np.linalg.norm(b - a) / dt),
                "robust_camera_extent_ratio_xyz": scale_ratio.astype(float).tolist(),
                "robust_camera_extent_ratio_max_abs_log": float(np.max(np.abs(np.log(scale_ratio)))),
            }
        )
    return pairs


def run(args: argparse.Namespace) -> dict:
    meshes = load_mesh_archive(args.mesh_archive)
    annotations = frame_map(args.annotations)
    rows = rows_from_meshes(meshes, annotations, args)
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} mesh frames")
    pairs = add_pairwise(rows, float(args.fps))
    camera_extent = np.asarray([row["robust_extent_camera_m"] for row in rows], dtype=np.float64)
    extent_median = np.median(camera_extent, axis=0)
    extent_ratio = camera_extent / np.maximum(extent_median[None, :], 1e-9)
    pair_extent_logs = [row["robust_camera_extent_ratio_max_abs_log"] for row in pairs]
    center_speeds = [row["center_speed_world_m_s"] for row in pairs]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "object_mesh_temporal_consistency_v3",
        "mesh_archive": str(args.mesh_archive),
        "annotations": str(args.annotations) if args.annotations is not None else None,
        "frames": int(len(rows)),
        "first_frame": int(rows[0]["frame_idx"]),
        "last_frame": int(rows[-1]["frame_idx"]),
        "robust_camera_extent_median_m": extent_median.astype(float).tolist(),
        "robust_camera_extent_ratio_to_median": {
            "x": summarize(extent_ratio[:, 0]),
            "y": summarize(extent_ratio[:, 1]),
            "z": summarize(extent_ratio[:, 2]),
            "max_abs_log": summarize(np.max(np.abs(np.log(extent_ratio)), axis=1)),
        },
        "pair_center_speed_world_m_s": summarize(center_speeds),
        "pair_robust_extent_ratio_max_abs_log": summarize(pair_extent_logs),
        "rows": rows,
        "pairs": pairs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "pairs"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robust-quantile", type=float, default=0.05)
    parser.add_argument("--min-frames", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
