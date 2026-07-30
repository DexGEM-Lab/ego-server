#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
import trimesh
from trimesh.sample import sample_surface

from close_mesh_archive_with_voxel_fill_v3 import save_archive, topology, transform_points
from diagnose_contact_kinematics_v3 import selected_vertex_ids
from fit_mano_to_hand_mask_depth_v3 import load_mano_faces
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, contact_rows, hand_vertices_camera
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive


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
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def contact_points_camera(annotations: dict[int, dict], contact_report: Path, frame_idx: int) -> np.ndarray:
    points = []
    for row in contact_rows(contact_report):
        if int(row["frame_idx"]) != int(frame_idx):
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        ids = selected_vertex_ids(row)
        if int(ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} contact vertex id exceeds MANO vertex count")
        points.append(vertices[ids])
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(points).astype(np.float64)


def hand_surface_camera(
    annotations: dict[int, dict],
    contact_report: Path,
    frame_idx: int,
    faces: np.ndarray,
    samples_per_hand: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    points = []
    total_vertices = 0
    total_samples = 0
    rng = np.random.default_rng(int(seed) + int(frame_idx))
    seen: set[int] = set()
    for row in contact_rows(contact_report):
        if int(row["frame_idx"]) != int(frame_idx):
            continue
        hand_idx = int(row["hand_idx"])
        if hand_idx in seen:
            continue
        seen.add(hand_idx)
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        if len(vertices) <= int(faces.max()):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} has too few vertices for MANO faces")
        mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), process=True)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} produced empty MANO mesh")
        points.append(np.asarray(mesh.vertices, dtype=np.float64))
        total_vertices += int(len(mesh.vertices))
        if int(samples_per_hand) > 0:
            state = np.random.get_state()
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            try:
                samples, _face_ids = sample_surface(mesh, int(samples_per_hand))
            finally:
                np.random.set_state(state)
            points.append(np.asarray(samples, dtype=np.float64))
            total_samples += int(samples_per_hand)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), {
            "contact_hands": 0,
            "hand_surface_vertices": 0,
            "hand_surface_samples": 0,
            "hand_surface_points": 0,
        }
    surface = np.vstack(points).astype(np.float64)
    return surface, {
        "contact_hands": int(len(seen)),
        "hand_surface_vertices": int(total_vertices),
        "hand_surface_samples": int(total_samples),
        "hand_surface_points": int(len(surface)),
    }


def voxel_centers(mesh: trimesh.Trimesh, pitch_m: float) -> np.ndarray:
    vox = mesh.voxelized(float(pitch_m)).fill()
    points = np.asarray(vox.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("voxelization produced no occupied centers")
    return points


def add_points_to_grid(grid: np.ndarray, points: np.ndarray, origin: np.ndarray, pitch_m: float) -> int:
    if len(points) == 0:
        return 0
    idx = np.rint((points - origin[None, :]) / float(pitch_m)).astype(np.int64)
    valid = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < grid.shape[0])
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < grid.shape[1])
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < grid.shape[2])
    )
    idx = idx[valid]
    if len(idx) == 0:
        return 0
    before = int(np.count_nonzero(grid))
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return int(np.count_nonzero(grid) - before)


def remove_hand_voxels(
    grid: np.ndarray,
    origin: np.ndarray,
    pitch_m: float,
    hand_surface: np.ndarray,
    contact_points: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    if len(hand_surface) == 0 or float(args.hand_surface_exclusion_m) < 0.0:
        return {"hand_surface_excluded_voxels": 0}
    occupied_idx = np.argwhere(grid)
    if len(occupied_idx) == 0:
        raise RuntimeError("cannot remove hand voxels from empty occupancy")
    centers = origin[None, :] + occupied_idx.astype(np.float64) * float(pitch_m)
    hand_distance = cKDTree(hand_surface).query(centers, k=1)[0]
    if len(contact_points):
        contact_distance = cKDTree(contact_points).query(centers, k=1)[0]
    else:
        contact_distance = np.full(len(centers), np.inf, dtype=np.float64)
    remove = (hand_distance <= float(args.hand_surface_exclusion_m)) & (contact_distance > float(args.contact_exemption_radius_m))
    if np.any(remove):
        doomed = occupied_idx[remove]
        grid[doomed[:, 0], doomed[:, 1], doomed[:, 2]] = False
    return {
        "hand_surface_excluded_voxels": int(np.count_nonzero(remove)),
        "hand_surface_exclusion_m": float(args.hand_surface_exclusion_m),
        "contact_exemption_radius_m": float(args.contact_exemption_radius_m),
    }


def mesh_from_grid(grid: np.ndarray, origin: np.ndarray, pitch_m: float, args: argparse.Namespace) -> trimesh.Trimesh:
    if int(np.count_nonzero(grid)) == 0:
        raise RuntimeError("fused visible/contact occupancy is empty")
    occ = binary_closing(grid, iterations=int(args.close_iterations))
    pad = int(args.sdf_pad_voxels)
    if pad < 2:
        raise RuntimeError("SDF extraction requires at least two pad voxels")
    occ = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ, sampling=[float(pitch_m)] * 3)
    inside = distance_transform_edt(occ, sampling=[float(pitch_m)] * 3)
    sdf = outside - inside
    if float(args.sdf_smooth_sigma_voxels) > 0.0:
        sdf = gaussian_filter(sdf, sigma=float(args.sdf_smooth_sigma_voxels), mode="nearest")
    vertices, faces, normals, _ = measure.marching_cubes(
        sdf.astype(np.float32),
        level=0.0,
        spacing=(float(pitch_m), float(pitch_m), float(pitch_m)),
        allow_degenerate=False,
    )
    vertices = vertices + origin[None, :] - float(pitch_m) * pad
    mesh = trimesh.Trimesh(
        vertices=vertices.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        vertex_normals=np.asarray(normals, dtype=np.float64),
        process=True,
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError("fused visible/contact mesh has no components")
    mesh = max(components, key=lambda component: float(component.area))
    trimesh.repair.fix_normals(mesh)
    topo = topology(mesh)
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"fused visible/contact mesh is not topologically closed: {topo}")
    return mesh


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    visible_meshes = load_mesh_archive(args.visible_mesh_archive)
    contact_meshes = load_mesh_archive(args.contact_mesh_archive)
    mano_faces = load_mano_faces(args.mano_model)
    frames = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in visible_meshes and idx in contact_meshes and idx in annotations]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames available for fusion")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meshes_world = []
    rows = []
    for frame_idx in frames:
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        visible_world, visible_faces = visible_meshes[frame_idx]
        contact_world, contact_faces = contact_meshes[frame_idx]
        visible_camera = camera_points(visible_world, T_world_camera)
        contact_camera = camera_points(contact_world, T_world_camera)
        visible_mesh = trimesh.Trimesh(vertices=visible_camera.astype(np.float32), faces=np.asarray(visible_faces, dtype=np.int32), process=True)
        contact_mesh = trimesh.Trimesh(vertices=contact_camera.astype(np.float32), faces=np.asarray(contact_faces, dtype=np.int32), process=True)
        contacts = contact_points_camera(annotations, args.contact_report, frame_idx)
        hand_surface, hand_row = hand_surface_camera(
            annotations,
            args.contact_report,
            frame_idx,
            mano_faces,
            int(args.hand_surface_samples_per_hand),
            int(args.seed),
        )
        if len(contacts) == 0:
            raise RuntimeError(f"frame {frame_idx} has no reliable contact points")
        lo = np.minimum(visible_mesh.bounds[0], contacts.min(axis=0) - float(args.contact_source_radius_m)) - float(args.grid_pad_m)
        hi = np.maximum(visible_mesh.bounds[1], contacts.max(axis=0) + float(args.contact_source_radius_m)) + float(args.grid_pad_m)
        pitch = float(args.pitch_m)
        shape = tuple((np.ceil((hi - lo) / pitch).astype(int) + 1).tolist())
        if int(np.prod(shape)) > int(args.max_grid_voxels):
            raise RuntimeError(f"frame {frame_idx} grid has {int(np.prod(shape))} voxels, above max")
        grid = np.zeros(shape, dtype=bool)
        visible_centers = voxel_centers(visible_mesh, pitch)
        visible_added = add_points_to_grid(grid, visible_centers, lo, pitch)
        contact_centers_all = voxel_centers(contact_mesh, pitch)
        contact_distance = cKDTree(contacts).query(contact_centers_all, k=1)[0]
        contact_centers = contact_centers_all[contact_distance <= float(args.contact_source_radius_m)]
        contact_added = add_points_to_grid(grid, contact_centers, lo, pitch)
        removal_row = remove_hand_voxels(grid, lo, pitch, hand_surface, contacts, args)
        fused_camera = mesh_from_grid(grid, lo, pitch, args)
        fused_world = trimesh.Trimesh(
            vertices=transform_points(np.asarray(fused_camera.vertices, dtype=np.float64), T_world_camera).astype(np.float32),
            faces=np.asarray(fused_camera.faces, dtype=np.int32),
            process=False,
        )
        meshes_world.append(fused_world)
        if int(frame_idx) in set(int(v) for v in args.export_frames):
            fused_camera.export(args.output_dir / f"fused_visible_contact_frame_{frame_idx:06d}_camera.obj")
            fused_world.export(args.output_dir / f"fused_visible_contact_frame_{frame_idx:06d}_world.obj")
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "grid_shape": [int(v) for v in shape],
                "grid_voxels": int(np.prod(shape)),
                "visible_voxels_added": int(visible_added),
                "contact_source_voxels_available": int(len(contact_centers)),
                "contact_voxels_added": int(contact_added),
                "occupied_after_hand_exclusion": int(np.count_nonzero(grid)),
                "contact_points": int(len(contacts)),
                **hand_row,
                **removal_row,
                "mesh_topology": topology(fused_camera),
            }
        )
    archive_path = args.output_dir / "fused_visible_contact_meshes_world.npz"
    save_archive(archive_path, frames, meshes_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "fuse_visible_contact_sdf_mesh_v3",
        "claim_tested": "visible-depth carved volume plus contact-preserving mesh volume can satisfy the same mesh archive QC under MANO surface exclusion",
        "visible_mesh_archive": str(args.visible_mesh_archive),
        "contact_mesh_archive": str(args.contact_mesh_archive),
        "annotations": str(args.annotations),
        "contact_report": str(args.contact_report),
        "mesh_archive_world": str(archive_path),
        "frames": [int(v) for v in frames],
        "rows": rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "grid_pad_m": float(args.grid_pad_m),
            "contact_source_radius_m": float(args.contact_source_radius_m),
            "hand_surface_exclusion_m": float(args.hand_surface_exclusion_m),
            "contact_exemption_radius_m": float(args.contact_exemption_radius_m),
            "sdf_smooth_sigma_voxels": float(args.sdf_smooth_sigma_voxels),
        },
    }
    save_json(args.output_dir / "qc_fused_visible_contact_sdf_mesh_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-mesh-archive", type=Path, required=True)
    parser.add_argument("--contact-mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--pitch-m", type=float, default=0.004)
    parser.add_argument("--grid-pad-m", type=float, default=0.012)
    parser.add_argument("--max-grid-voxels", type=int, default=5000000)
    parser.add_argument("--contact-source-radius-m", type=float, default=0.026)
    parser.add_argument("--hand-surface-exclusion-m", type=float, default=0.004)
    parser.add_argument("--contact-exemption-radius-m", type=float, default=0.014)
    parser.add_argument("--hand-surface-samples-per-hand", type=int, default=30000)
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument("--sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--sdf-smooth-sigma-voxels", type=float, default=0.5)
    parser.add_argument("--export-frames", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=313)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
