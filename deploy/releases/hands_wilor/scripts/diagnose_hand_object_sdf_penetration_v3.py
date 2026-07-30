#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from diagnose_mesh_surface_contact_v3 import hand_camera_vertices, hand_local_joints, hand_local_vertices
from diagnose_volume_sdf_contact_v3 import sample_sdf, summarize, voxel_sdf
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, contact_rows
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


JOINT_REGION = np.asarray(
    [
        "palm",
        "thumb",
        "thumb",
        "thumb",
        "thumb",
        "index",
        "index",
        "index",
        "index",
        "middle",
        "middle",
        "middle",
        "middle",
        "ring",
        "ring",
        "ring",
        "ring",
        "pinky",
        "pinky",
        "pinky",
        "pinky",
    ],
    dtype=object,
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def region_vertex_labels(hand: dict, vertex_count: int) -> np.ndarray:
    local_vertices = hand_local_vertices(hand)
    local_joints = hand_local_joints(hand)
    if local_vertices.shape[0] != vertex_count or local_joints.shape != (21, 3):
        return np.full(vertex_count, "unknown", dtype=object)
    nearest_joint = np.argmin(np.linalg.norm(local_vertices[:, None, :] - local_joints[None, :, :], axis=2), axis=1)
    return JOINT_REGION[nearest_joint]


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    meshes = load_mesh_archive(args.mesh_archive)
    rows = []
    all_sdf = []
    sdf_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    contact_keys = {(int(row["frame_idx"]), int(row["hand_idx"])) for row in contact_rows(args.contact_report)}
    for frame_idx, hand_idx in sorted(contact_keys):
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        if frame_idx not in annotations or frame_idx not in meshes:
            continue
        hand = annotations[frame_idx]["hands"][hand_idx]
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        hand_vertices = hand_camera_vertices(hand, T_world_camera)
        labels = region_vertex_labels(hand, len(hand_vertices))
        if frame_idx not in sdf_by_frame:
            mesh_world, mesh_faces = meshes[frame_idx]
            mesh_camera = camera_points(mesh_world, T_world_camera)
            mesh = trimesh.Trimesh(vertices=mesh_camera.astype(np.float32), faces=np.asarray(mesh_faces, dtype=np.int32), process=True)
            sdf_by_frame[frame_idx] = voxel_sdf(mesh, float(args.pitch_m), int(args.pad_voxels))
        sdf, transform, occ = sdf_by_frame[frame_idx]
        values = sample_sdf(hand_vertices, sdf, transform)
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        if len(finite) == 0:
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} has no in-bounds SDF samples")
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
        raise RuntimeError("no hand-object SDF rows diagnosed")
    all_arr = np.asarray(all_sdf, dtype=np.float64)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "hand_object_sdf_penetration_v3",
        "annotations": str(args.annotations),
        "mesh_archive": str(args.mesh_archive),
        "contact_report": str(args.contact_report),
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
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--pad-voxels", type=int, default=8)
    parser.add_argument("--penetration-tolerance-m", type=float, default=0.002)
    parser.add_argument("--near-surface-m", type=float, default=0.006)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
