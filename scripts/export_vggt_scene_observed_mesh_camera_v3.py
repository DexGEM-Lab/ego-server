#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_vggt_scene_object_points_v3 import build_frame_mesh, nearest_summary, summarize


def camera_points(points_vggt: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    return (np.asarray(points_vggt, dtype=np.float64) @ extrinsic[:3, :3].T) + extrinsic[:3, 3][None, :]


def save_archive(
    path: Path,
    frame_idx: np.ndarray,
    vertices: list[np.ndarray],
    faces: list[np.ndarray],
) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    for v, f in zip(vertices, faces, strict=True):
        vertex_offsets.append(vertex_offsets[-1] + len(v))
        face_offsets.append(face_offsets[-1] + len(f))
    np.savez_compressed(
        path,
        frame_idx=frame_idx.astype(np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices).astype(np.float32),
        faces=np.vstack(faces).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    blob = np.load(args.vggt_archive)
    required = {
        "frame_idx",
        "vertex_offsets",
        "object_points_aligned",
        "object_points_vggt",
        "object_colors",
        "extrinsic",
        "intrinsic",
        "sim3_scale",
        "sim3_rotation",
        "sim3_translation",
    }
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{args.vggt_archive} missing keys: {sorted(missing)}")
    all_frame_idx = blob["frame_idx"].astype(int)
    offsets = blob["vertex_offsets"].astype(np.int64)
    points_aligned = blob["object_points_aligned"].astype(np.float32)
    points_vggt = blob["object_points_vggt"].astype(np.float32)
    colors = blob["object_colors"].astype(np.uint8)
    extrinsic = blob["extrinsic"].astype(np.float64)
    intrinsic = blob["intrinsic"].astype(np.float64)
    sim3_scale = float(blob["sim3_scale"][0])
    if args.camera_scale_mode == "sim3":
        camera_scale = sim3_scale
    elif args.camera_scale_mode == "custom":
        camera_scale = float(args.custom_scale)
        if not np.isfinite(camera_scale) or camera_scale <= 0.0:
            raise RuntimeError(f"custom scale must be positive, got {camera_scale}")
    else:
        raise RuntimeError(f"unsupported camera scale mode: {args.camera_scale_mode}")
    sim3_rotation = blob["sim3_rotation"].astype(np.float64)
    sim3_translation = blob["sim3_translation"].astype(np.float64)

    frame_indices = []
    vertices_camera_all = []
    faces_all = []
    rows = []
    for i, idx in enumerate(all_frame_idx.tolist()):
        if idx < int(args.frame_start) or idx > int(args.frame_end):
            continue
        start, end = int(offsets[i]), int(offsets[i + 1])
        vertices_aligned, faces, row = build_frame_mesh(
            points_vggt[start:end],
            points_aligned[start:end],
            colors[start:end],
            extrinsic[i],
            intrinsic[i],
            sim3_scale,
            sim3_rotation,
            sim3_translation,
            int(args.target_size),
            int(args.grid_px),
            float(args.max_triangle_edge_m),
        )
        vertices_vggt_mesh = (vertices_aligned.astype(np.float64) - sim3_translation[None, :]) @ sim3_rotation / sim3_scale
        vertices_camera = float(camera_scale) * camera_points(vertices_vggt_mesh, extrinsic[i])
        if np.count_nonzero(vertices_camera[:, 2] > 0.0) < max(10, len(vertices_camera) // 2):
            raise RuntimeError(f"frame {idx} VGGT camera mesh has too few positive-depth vertices")
        row["frame_idx"] = int(idx)
        row["camera_extent_m"] = (vertices_camera.max(axis=0) - vertices_camera.min(axis=0)).astype(float).tolist()
        row["camera_robust_extent_m"] = (
            np.quantile(vertices_camera, 0.95, axis=0) - np.quantile(vertices_camera, 0.05, axis=0)
        ).astype(float).tolist()
        row["camera_depth_median_m"] = float(np.median(vertices_camera[:, 2]))
        row["mesh_to_vggt_points"] = nearest_summary(vertices_aligned, points_aligned[start:end])
        rows.append(row)
        frame_indices.append(int(idx))
        vertices_camera_all.append(vertices_camera.astype(np.float32))
        faces_all.append(faces.astype(np.int32))

    if len(frame_indices) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_indices)} frames selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "vggt_scene_observed_meshes_camera.npz"
    save_archive(archive, np.asarray(frame_indices, dtype=np.int32), vertices_camera_all, faces_all)
    camera_extent = np.asarray([row["camera_robust_extent_m"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "export_vggt_scene_observed_mesh_camera_v3",
        "vggt_archive": str(args.vggt_archive),
        "archive": str(archive),
        "frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "robust_camera_extent_median_m": np.median(camera_extent, axis=0).astype(float).tolist(),
        "camera_coordinate_scale": str(args.camera_scale_mode),
        "sim3_scale": float(sim3_scale),
        "custom_scale": float(args.custom_scale) if args.camera_scale_mode == "custom" else None,
        "applied_camera_scale": float(camera_scale),
        "mesh_to_vggt_points_median_m": summarize([row["mesh_to_vggt_points"]["median_m"] for row in rows]),
        "camera_depth_median_m": summarize([row["camera_depth_median_m"] for row in rows]),
        "rows": rows,
        "parameters": {
            "target_size": int(args.target_size),
            "grid_px": int(args.grid_px),
            "max_triangle_edge_m": float(args.max_triangle_edge_m),
        },
    }
    (args.output_dir / "qc_vggt_scene_observed_mesh_camera_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--grid-px", type=int, default=5)
    parser.add_argument("--max-triangle-edge-m", type=float, default=0.08)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--camera-scale-mode", choices=["sim3", "custom"], default="sim3")
    parser.add_argument("--custom-scale", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
