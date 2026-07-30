#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from diagnose_hand_reprojection_depth_v3 import project_points
from fuse_observed_surface_with_complete_prior_v3 import compute_frame_pose, intrinsics_for, prior_to_camera, read_mask
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, load_depth_archive, manifest_by_frame
from optimize_mesh_prior_pose_graph_v3 import load_mesh
from render_bundlesdf_mesh_qc_v3 import load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def vertex_mask_support(
    vertices_prior: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    min_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_camera = prior_to_camera(vertices_prior, r, t)
    positive = vertices_camera[:, 2] > float(min_depth_m)
    uv = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
    if np.any(positive):
        uv[positive] = project_points(vertices_camera[positive], intrinsics)
    rounded = np.rint(uv).astype(np.int64)
    in_bounds = (
        positive
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    supported = np.zeros(len(vertices_camera), dtype=bool)
    if np.any(in_bounds):
        y = rounded[in_bounds, 1]
        x = rounded[in_bounds, 0]
        supported[in_bounds] = mask[y, x]
    return supported, in_bounds


def filter_faces(mesh: trimesh.Trimesh, support_count: np.ndarray, visible_count: np.ndarray, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_supported = support_count >= int(args.min_supported_views)
    vertex_visible = visible_count >= int(args.min_visible_views)
    face_supported = np.sum(vertex_supported[faces], axis=1) >= int(args.min_supported_vertices_per_face)
    face_visible = np.sum(vertex_visible[faces], axis=1) >= int(args.min_visible_vertices_per_face)
    keep_faces = face_supported & face_visible
    if np.count_nonzero(keep_faces) < int(args.min_faces):
        raise RuntimeError(f"mask filtering kept only {np.count_nonzero(keep_faces)} faces")
    filtered = mesh.submesh([np.flatnonzero(keep_faces)], append=True, repair=False)
    filtered.remove_unreferenced_vertices()
    filtered.update_faces(filtered.unique_faces())
    filtered.update_faces(filtered.nondegenerate_faces())
    filtered.remove_unreferenced_vertices()
    components = filtered.split(only_watertight=False)
    if len(components) == 0:
        raise RuntimeError("mask-filtered mesh has no connected components")
    filtered = max(components, key=lambda comp: float(comp.area))
    filtered.remove_unreferenced_vertices()
    if len(filtered.vertices) < int(args.min_vertices) or len(filtered.faces) < int(args.min_faces):
        raise RuntimeError(f"largest mask-filtered component underconstrained: vertices={len(filtered.vertices)} faces={len(filtered.faces)}")
    return filtered, {
        "input_vertices": int(len(mesh.vertices)),
        "input_faces": int(len(mesh.faces)),
        "kept_faces_before_component": int(np.count_nonzero(keep_faces)),
        "output_vertices": int(len(filtered.vertices)),
        "output_faces": int(len(filtered.faces)),
        "output_extent_m": filtered.extents.astype(float).tolist(),
        "output_watertight": bool(filtered.is_watertight),
        "output_winding_consistent": bool(filtered.is_winding_consistent),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    reference_meshes = load_mesh_archive(args.reference_graph_mesh_archive)
    reference_prior = load_mesh(args.reference_mesh_prior_camera)
    reference_vertices = np.asarray(reference_prior.vertices, dtype=np.float64)
    mesh = load_mesh(args.mesh_prior_frame)
    vertices_prior = np.asarray(mesh.vertices, dtype=np.float64)
    support_count = np.zeros(len(vertices_prior), dtype=np.int32)
    visible_count = np.zeros(len(vertices_prior), dtype=np.int32)
    frame_rows = []
    anchor_pose: tuple[np.ndarray, np.ndarray] | None = None
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in manifest or frame_idx not in depths or frame_idx not in reference_meshes:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        r, t, pose_row = compute_frame_pose(
            reference_vertices,
            reference_meshes[frame_idx][0],
            annotation,
            int(args.max_pose_correspondences),
            int(args.seed) + frame_idx,
        )
        if int(frame_idx) == int(args.anchor_frame):
            anchor_pose = (r, t)
        supported, visible = vertex_mask_support(vertices_prior, r, t, mask, intrinsics, float(args.min_depth_m))
        support_count += supported.astype(np.int32)
        visible_count += visible.astype(np.int32)
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "visible_fraction": float(np.mean(visible)),
                "supported_fraction": float(np.mean(supported)),
                "supported_given_visible_fraction": float(np.mean(supported[visible])) if np.any(visible) else 0.0,
                "object_translation_camera_m": t.astype(float).tolist(),
                "object_rotation_delta_rad": Rotation.from_matrix(r).as_rotvec().astype(float).tolist(),
                **pose_row,
            }
        )
    if len(frame_rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_rows)} usable frames for mask support")
    if anchor_pose is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} was not available")
    filtered, mesh_report = filter_faces(mesh, support_count, visible_count, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_frame_path = args.output_dir / "mask_supported_mesh_prior_frame.obj"
    filtered.export(prior_frame_path)
    anchor_path = args.output_dir / "mask_supported_mesh_anchor_camera.obj"
    r_anchor, t_anchor = anchor_pose
    anchor_vertices = prior_to_camera(np.asarray(filtered.vertices, dtype=np.float64), r_anchor, t_anchor)
    trimesh.Trimesh(vertices=anchor_vertices.astype(np.float32), faces=np.asarray(filtered.faces, dtype=np.int32), process=False).export(anchor_path)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "filter_mesh_by_multiview_mask_support_v3",
        "mesh_prior_frame": str(args.mesh_prior_frame),
        "reference_mesh_prior_camera": str(args.reference_mesh_prior_camera),
        "reference_graph_mesh_archive": str(args.reference_graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "frames": [int(row["frame_idx"]) for row in frame_rows],
        "anchor_frame": int(args.anchor_frame),
        "prior_frame_mesh": str(prior_frame_path),
        "anchor_camera_mesh": str(anchor_path),
        "support_count": summarize(support_count),
        "visible_count": summarize(visible_count),
        "frame_rows": frame_rows,
        "mesh": mesh_report,
        "parameters": {
            "min_supported_views": int(args.min_supported_views),
            "min_visible_views": int(args.min_visible_views),
            "min_supported_vertices_per_face": int(args.min_supported_vertices_per_face),
            "min_visible_vertices_per_face": int(args.min_visible_vertices_per_face),
        },
    }
    save_json(args.output_dir / "qc_mask_supported_mesh_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "frame_rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-frame", type=Path, required=True)
    parser.add_argument("--reference-mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--reference-graph-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--min-supported-views", type=int, default=1)
    parser.add_argument("--min-visible-views", type=int, default=1)
    parser.add_argument("--min-supported-vertices-per-face", type=int, default=2)
    parser.add_argument("--min-visible-vertices-per-face", type=int, default=2)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--min-vertices", type=int, default=500)
    parser.add_argument("--min-faces", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1103)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
