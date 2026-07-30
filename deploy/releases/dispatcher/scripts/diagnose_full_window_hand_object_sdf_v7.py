#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from diagnose_hand_object_sdf_penetration_v3 import region_vertex_labels
from diagnose_mesh_surface_contact_v3 import hand_camera_vertices
from diagnose_volume_sdf_contact_v3 import sample_sdf, summarize, voxel_sdf
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def frame_sdf(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_idx: int,
    T_world_camera: np.ndarray,
    args: argparse.Namespace,
    cover_points: np.ndarray,
):
    mesh_world, mesh_faces = meshes[frame_idx]
    mesh_camera = camera_points(mesh_world, T_world_camera)
    mesh = trimesh.Trimesh(vertices=mesh_camera.astype(np.float32), faces=np.asarray(mesh_faces, dtype=np.int32), process=True)
    return voxel_sdf(mesh, float(args.pitch_m), int(args.pad_voxels), cover_points=cover_points)


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    meshes = load_mesh_archive(args.mesh_archive)
    rows = []
    all_sdf = []
    sdf_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in meshes:
            continue
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        frame_hands = []
        for hand in annotations[frame_idx].get("hands", []):
            try:
                frame_hands.append((hand, hand_camera_vertices(hand, T_world_camera)))
            except RuntimeError:
                continue
        if not frame_hands:
            continue
        cover_points = np.concatenate([vertices for _, vertices in frame_hands], axis=0)
        if frame_idx not in sdf_by_frame:
            sdf_by_frame[frame_idx] = frame_sdf(meshes, frame_idx, T_world_camera, args, cover_points)
        sdf, transform, occ = sdf_by_frame[frame_idx]
        for hand_idx, (hand, hand_vertices) in enumerate(frame_hands):
            labels = region_vertex_labels(hand, len(hand_vertices))
            values = sample_sdf(hand_vertices, sdf, transform)
            finite_mask = np.isfinite(values)
            finite = values[finite_mask]
            if len(finite) == 0:
                continue
            all_sdf.extend(finite.astype(float).tolist())
            region_rows = []
            for region in sorted(set(labels[finite_mask].tolist())):
                take = finite_mask & (labels == region)
                vals = values[take]
                if len(vals) == 0:
                    continue
                region_rows.append(
                    {
                        "region": str(region),
                        "vertices": int(len(vals)),
                        "sdf_m": summarize(vals),
                        "penetration_fraction": float(np.mean(vals < -float(args.penetration_tolerance_m))),
                        "near_surface_fraction": float(np.mean(np.abs(vals) <= float(args.near_surface_m))),
                    }
                )
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "hand_idx": int(hand_idx),
                    "vertices": int(len(finite)),
                    "sdf_m": summarize(finite),
                    "penetration_fraction": float(np.mean(finite < -float(args.penetration_tolerance_m))),
                    "near_surface_fraction": float(np.mean(np.abs(finite) <= float(args.near_surface_m))),
                    "regions": region_rows,
                    "voxel_occupied": int(np.count_nonzero(occ)),
                    "voxel_shape": [int(v) for v in occ.shape],
                }
            )
    if not rows:
        raise RuntimeError("no hand-object SDF rows diagnosed over the requested frame window")
    all_arr = np.asarray(all_sdf, dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "full_window_hand_object_sdf_v7",
        "annotations": str(args.annotations),
        "mesh_archive": str(args.mesh_archive),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": sorted({int(row["frame_idx"]) for row in rows}),
        "summary": {
            "sdf_m": summarize(all_arr),
            "penetration_fraction": float(np.mean(all_arr < -float(args.penetration_tolerance_m))),
            "near_surface_fraction": float(np.mean(np.abs(all_arr) <= float(args.near_surface_m))),
        },
        "rows": rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "pad_voxels": int(args.pad_voxels),
            "penetration_tolerance_m": float(args.penetration_tolerance_m),
            "near_surface_m": float(args.near_surface_m),
        },
    }
    save_json(args.output_json, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--pad-voxels", type=int, default=8)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.003)
    parser.add_argument("--near-surface-m", type=float, default=0.006)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
