#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from close_mesh_archive_with_voxel_fill_v3 import save_archive, transform_points
from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json, load_mesh_archive
from render_mesh_zbuffer_qc_v3 import triangle_zbuffer


def summarize(values: np.ndarray | list[float]) -> dict:
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


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    entries = manifest.get("frames")
    frames = annotations.get("frames")
    if not isinstance(entries, list) or not isinstance(frames, list):
        raise RuntimeError("manifest and annotations must contain frames lists")
    annotation_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    mesh_archive = load_mesh_archive(args.mesh_archive)
    depth_archive = load_depth_archive(args.metric_depth_npz)
    corrected = []
    frame_ids = []
    rows = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        if frame_idx not in annotation_by_idx:
            raise RuntimeError(f"annotations lack frame {frame_idx}")
        if frame_idx not in mesh_archive:
            raise RuntimeError(f"mesh archive lacks frame {frame_idx}")
        if frame_idx not in depth_archive:
            raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask for frame {frame_idx}")
        object_mask = mask > 0
        depth_m = np.asarray(depth_archive[frame_idx], dtype=np.float64)
        if depth_m.shape != object_mask.shape:
            raise RuntimeError(f"depth shape {depth_m.shape} does not match mask shape {object_mask.shape}")
        vertices_world, faces = mesh_archive[frame_idx]
        annotation = annotation_by_idx[frame_idx]
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        K = intrinsics_for_frame(args, entry, annotation)
        vertices_camera = camera_points(vertices_world, T_world_camera)
        z = vertices_camera[:, 2]
        uv = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
        positive = z > 0.0
        uv[positive, 0] = K[0, 0] * vertices_camera[positive, 0] / z[positive] + K[0, 2]
        uv[positive, 1] = K[1, 1] * vertices_camera[positive, 1] / z[positive] + K[1, 2]
        zbuf = triangle_zbuffer(object_mask.shape, uv, z, faces, args.max_faces)
        valid = np.isfinite(zbuf) & object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
        residual = zbuf[valid].astype(np.float64) - depth_m[valid]
        if len(residual) < int(args.min_samples):
            raise RuntimeError(f"frame {frame_idx} has only {len(residual)} valid z-buffer residual samples")
        shift_m = -float(np.median(residual))
        shift_m = float(np.clip(shift_m, -float(args.max_abs_shift_m), float(args.max_abs_shift_m)))
        shifted_camera = vertices_camera.copy()
        shifted_camera[:, 2] += shift_m
        shifted_world = transform_points(shifted_camera, T_world_camera)
        corrected.append(
            trimesh.Trimesh(
                vertices=shifted_world.astype(np.float32),
                faces=np.asarray(faces, dtype=np.int32),
                process=False,
            )
        )
        frame_ids.append(frame_idx)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "valid_samples": int(len(residual)),
                "pre_shift_residual_m": summarize(residual),
                "camera_z_shift_m": shift_m,
            }
        )
    if not corrected:
        raise RuntimeError("no frames corrected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "zbuffer_depth_shift_meshes_world.npz"
    save_archive(
        archive_path,
        frame_ids,
        corrected,
    )
    report = {
        "status": "ok",
        "method": "zbuffer_depth_shift_v3",
        "diagnostic_only": True,
        "claim_tested": "per-frame camera-z translation can explain the delivered mesh visible-depth residual",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "corrected_mesh_archive": str(archive_path),
        "camera_z_shift_m": summarize([row["camera_z_shift_m"] for row in rows]),
        "rows": rows,
        "parameters": {
            "max_abs_shift_m": float(args.max_abs_shift_m),
            "max_faces": None if args.max_faces is None else int(args.max_faces),
            "min_samples": int(args.min_samples),
        },
    }
    (args.output_dir / "qc_zbuffer_depth_shift_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--max-abs-shift-m", type=float, default=0.080)
    parser.add_argument("--min-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_faces is not None and int(args.max_faces) <= 0:
        args.max_faces = None
    run(args)


if __name__ == "__main__":
    main()
