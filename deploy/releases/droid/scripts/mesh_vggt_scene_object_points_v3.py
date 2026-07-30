#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def build_frame_mesh(
    points_vggt: np.ndarray,
    points_world: np.ndarray,
    colors: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    sim3_scale: float,
    sim3_rotation: np.ndarray,
    sim3_translation: np.ndarray,
    target_size: int,
    grid: int,
    max_edge_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    points_cam = (points_vggt @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    uv_h = points_cam @ intrinsic.T
    uv = np.c_[uv_h[:, 0] / z, uv_h[:, 1] / z]
    valid &= np.isfinite(uv).all(axis=1)
    if int(valid.sum()) < 4:
        raise RuntimeError("too few valid VGGT object points for meshing")
    uv = uv[valid]
    points = points_world[valid]
    colors = colors[valid]
    cols = np.floor(np.clip(uv[:, 0], 0, target_size - 1) / float(grid)).astype(int)
    rows = np.floor(np.clip(uv[:, 1], 0, target_size - 1) / float(grid)).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(rows.tolist(), cols.tolist(), strict=True)):
        cells.setdefault(key, []).append(i)
    cell_keys = sorted(cells)
    vertex_index: dict[tuple[int, int], int] = {}
    vertices = []
    for key in cell_keys:
        idx = np.asarray(cells[key], dtype=int)
        vertex_index[key] = len(vertices)
        vertices.append(np.median(points_vggt[valid][idx], axis=0))
    vertices_vggt = np.asarray(vertices, dtype=np.float32)
    faces = []
    for r, c in cell_keys:
        square = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        if all(key in vertex_index for key in square):
            for tri_keys in ((square[0], square[1], square[2]), (square[1], square[3], square[2])):
                tri = [vertex_index[key] for key in tri_keys]
                tri_pts_vggt = vertices_vggt[np.asarray(tri, dtype=np.int32)]
                tri_pts_cam = (tri_pts_vggt @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]
                edge = max(
                    float(np.linalg.norm(tri_pts_cam[0] - tri_pts_cam[1])),
                    float(np.linalg.norm(tri_pts_cam[1] - tri_pts_cam[2])),
                    float(np.linalg.norm(tri_pts_cam[2] - tri_pts_cam[0])),
                )
                if edge <= float(max_edge_m):
                    faces.append(tri)
    faces_arr = np.asarray(faces, dtype=np.int32)
    if len(vertices_vggt) == 0 or len(faces_arr) == 0:
        raise RuntimeError("VGGT object point mesh is empty")
    used = np.unique(faces_arr.reshape(-1))
    remap = np.full(len(vertices_vggt), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    vertices_vggt = vertices_vggt[used]
    vertices_arr = (float(sim3_scale) * (vertices_vggt.astype(float) @ sim3_rotation.T)) + sim3_translation[None, :]
    faces_arr = remap[faces_arr]
    report = {
        "points_input": int(len(points_world)),
        "points_projected": int(valid.sum()),
        "vertices": int(len(vertices_arr)),
        "faces": int(len(faces_arr)),
        "extent_world_m": (vertices_arr.max(axis=0) - vertices_arr.min(axis=0)).astype(float).tolist(),
    }
    return vertices_arr.astype(np.float32), faces_arr.astype(np.int32), report


def nearest_summary(source: np.ndarray, target: np.ndarray) -> dict:
    tree = cKDTree(np.asarray(target, dtype=float))
    dist = tree.query(np.asarray(source, dtype=float), k=1)[0]
    return {
        "median_m": float(np.median(dist)),
        "p95_m": float(np.percentile(dist, 95)),
        "max_m": float(np.max(dist)),
    }


def run(args: argparse.Namespace) -> dict:
    blob = np.load(args.vggt_archive)
    required = {"frame_idx", "vertex_offsets", "object_points_aligned", "object_points_vggt", "object_colors", "extrinsic", "intrinsic"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"VGGT archive missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    points_world_all = blob["object_points_aligned"].astype(np.float32)
    points_vggt_all = blob["object_points_vggt"].astype(np.float32)
    colors_all = blob["object_colors"].astype(np.uint8)
    extrinsic = blob["extrinsic"].astype(float)
    intrinsic = blob["intrinsic"].astype(float)
    sim3_scale = float(blob["sim3_scale"][0])
    sim3_rotation = blob["sim3_rotation"].astype(float)
    sim3_translation = blob["sim3_translation"].astype(float)
    vertices_all = []
    faces_all = []
    rows = []
    vertex_offsets = [0]
    face_offsets = [0]
    for i, idx in enumerate(frame_idx.tolist()):
        start, end = int(offsets[i]), int(offsets[i + 1])
        vertices, faces, row = build_frame_mesh(
            points_vggt_all[start:end],
            points_world_all[start:end],
            colors_all[start:end],
            extrinsic[i],
            intrinsic[i],
            sim3_scale,
            sim3_rotation,
            sim3_translation,
            int(args.target_size),
            int(args.grid_px),
            float(args.max_triangle_edge_m),
        )
        vertices_all.append(vertices)
        faces_all.append(faces)
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
        row["frame_idx"] = int(idx)
        row["mesh_to_vggt_points"] = nearest_summary(vertices, points_world_all[start:end])
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "vggt_scene_object_meshes.npz"
    np.savez_compressed(
        archive_path,
        frame_idx=frame_idx.astype(np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "mesh_vggt_scene_object_points_v3",
        "vggt_archive": str(args.vggt_archive),
        "mesh_archive": str(archive_path),
        "frames": int(len(frame_idx)),
        "vertices": int(sum(len(v) for v in vertices_all)),
        "faces": int(sum(len(f) for f in faces_all)),
        "mesh_to_vggt_points_median_m": summarize([row["mesh_to_vggt_points"]["median_m"] for row in rows]),
        "rows": rows,
        "parameters": {
            "target_size": int(args.target_size),
            "grid_px": int(args.grid_px),
            "max_triangle_edge_m": float(args.max_triangle_edge_m),
        },
    }
    (args.output_dir / "qc_vggt_scene_object_meshes_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--grid-px", type=int, default=5)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
