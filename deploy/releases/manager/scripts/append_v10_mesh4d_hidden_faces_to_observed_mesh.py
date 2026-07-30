#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from append_v9_hidden_prior_faces_to_observed_mesh import (
    annotations_by_frame,
    append_faces,
    manifest_by_frame,
    projection_filter,
    simplify_prior_mesh,
)
from archive_aligned_mesh_prior_v7 import write_mesh_archive
from diagnose_object_mesh_temporal_consistency_v3 import load_mesh_archive
from fit_cotracker_pairwise_rigid_factors_v6 import summarize
from render_bundlesdf_mesh_qc_v3 import load_depth_archive


def run(args: argparse.Namespace) -> dict:
    observed = load_mesh_archive(args.observed_mesh_archive)
    mesh4d = load_mesh_archive(args.mesh4d_mesh_archive)
    manifest = manifest_by_frame(args.manifest)
    annotations = annotations_by_frame(args.annotations)
    depths = load_depth_archive(args.metric_depth_npz)
    frames = [
        idx
        for idx in range(int(args.frame_start), int(args.frame_end) + 1)
        if idx in observed and idx in mesh4d and idx in manifest and idx in annotations and idx in depths
    ]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames have observed mesh, Mesh4D mesh, manifest, annotations, and depth")

    rows = []
    archive_rows = []
    for frame_idx in frames:
        observed_vertices, observed_faces = observed[frame_idx]
        mesh4d_vertices_raw, mesh4d_faces_raw = mesh4d[frame_idx]
        mesh4d_vertices, mesh4d_faces, simplification = simplify_prior_mesh(
            mesh4d_vertices_raw,
            mesh4d_faces_raw,
            int(args.max_mesh4d_faces_per_frame),
        )
        face_keep, row = projection_filter(
            mesh4d_vertices,
            mesh4d_faces,
            observed_vertices,
            observed_faces,
            manifest[frame_idx],
            annotations[frame_idx],
            np.asarray(depths[frame_idx], dtype=np.float64),
            args,
        )
        kept_faces = mesh4d_faces[face_keep]
        vertices, faces = append_faces(observed_vertices, observed_faces, mesh4d_vertices, kept_faces)
        archive_rows.append((frame_idx, vertices, faces))
        row.update(
            {
                "frame_idx": int(frame_idx),
                "observed_vertices": int(len(observed_vertices)),
                "observed_faces": int(len(observed_faces)),
                "mesh4d_vertices_raw": int(len(mesh4d_vertices_raw)),
                "mesh4d_faces_raw": int(len(mesh4d_faces_raw)),
                "mesh4d_vertices_filtered": int(len(mesh4d_vertices)),
                "mesh4d_faces_filtered": int(len(mesh4d_faces)),
                "kept_mesh4d_faces": int(len(kept_faces)),
                "archive_vertices": int(len(vertices)),
                "archive_faces": int(len(faces)),
                "mesh4d_simplification": simplification,
            }
        )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observed_plus_mesh4d_hidden_meshes_world.npz"
    write_mesh_archive(archive, archive_rows)
    kept = np.asarray([row["kept_mesh4d_faces"] for row in rows], dtype=np.float64)
    raw_faces = np.asarray([row["mesh4d_faces_raw"] for row in rows], dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "method": "append_v10_mesh4d_hidden_faces_to_observed_mesh",
        "claim_tested": (
            "measured visible object meshes remain exact while Mesh4D generated faces are appended only when mask, "
            "metric depth, camera free space, z-buffer, and measured-surface distance support them as hidden geometry"
        ),
        "mesh4d_mesh_archive": str(args.mesh4d_mesh_archive),
        "observed_mesh_archive": str(args.observed_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "mesh_archive": str(archive),
        "output_mesh_archive": str(archive),
        "frame_count": int(len(frames)),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "kept_mesh4d_faces": summarize(kept),
        "raw_mesh4d_faces": summarize(raw_faces),
        "parameters": {
            "max_mesh4d_faces_per_frame": int(args.max_mesh4d_faces_per_frame),
            "min_hidden_distance_m": float(args.min_hidden_distance_m),
            "max_visible_depth_abs_m": float(args.max_visible_depth_abs_m),
            "max_front_free_space_m": float(args.max_front_free_space_m),
            "min_hidden_behind_observed_m": float(args.min_hidden_behind_observed_m),
            "max_raster_filter_iters": int(args.max_raster_filter_iters),
            "allow_visible_mask_fill": bool(args.allow_visible_mask_fill),
            "keep_inside_image_hidden_faces": bool(args.keep_inside_image_hidden_faces),
            "min_component_faces": int(args.min_component_faces),
        },
        "rows": rows,
    }
    out = args.output_dir / "qc_append_v10_mesh4d_hidden_faces_to_observed_mesh.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh4d-mesh-archive", type=Path, required=True)
    parser.add_argument("--observed-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--intrinsics-source", choices=("annotation-vggt", "manifest"), default="annotation-vggt")
    parser.add_argument("--max-mesh4d-faces-per-frame", type=int, default=60000)
    parser.add_argument("--min-hidden-distance-m", type=float, default=0.004)
    parser.add_argument("--max-visible-depth-abs-m", type=float, default=0.008)
    parser.add_argument("--max-front-free-space-m", type=float, default=0.004)
    parser.add_argument("--min-hidden-behind-observed-m", type=float, default=0.003)
    parser.add_argument("--max-raster-filter-iters", type=int, default=12)
    parser.add_argument("--allow-visible-mask-fill", action="store_true")
    parser.add_argument("--keep-inside-image-hidden-faces", action="store_true")
    parser.add_argument("--min-component-faces", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
