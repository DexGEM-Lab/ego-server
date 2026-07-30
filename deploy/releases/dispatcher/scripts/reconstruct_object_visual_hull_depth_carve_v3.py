#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage import measure
import trimesh
from trimesh.sample import sample_surface

from close_mesh_archive_with_voxel_fill_v3 import save_archive, topology, transform_points
from fuse_observed_surface_with_complete_prior_v3 import compute_frame_pose, intrinsics_for, read_mask
from diagnose_contact_kinematics_v3 import selected_vertex_ids
from fit_mano_to_hand_mask_depth_v3 import load_mano_faces
from optimize_contact_patch_object_pose_graph_v3 import contact_rows, hand_vertices_camera
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame, load_depth_archive, manifest_by_frame
from optimize_mesh_prior_pose_graph_v3 import load_mesh
from render_bundlesdf_mesh_qc_v3 import load_mesh_archive


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = z > 0.0
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    uv[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return uv, valid


def carve_volume(
    grid_points: np.ndarray,
    grid_shape: tuple[int, int, int],
    frame_rows: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    keep_votes = np.zeros(len(grid_points), dtype=np.int16)
    depth_hits = np.zeros(len(grid_points), dtype=np.int16)
    rows = []
    for row in frame_rows:
        r = row["rotation_prior_to_camera"]
        t = row["translation_prior_to_camera"]
        points_camera = grid_points @ r.T + t[None, :]
        uv, positive = project(points_camera, row["intrinsics"])
        xy = np.rint(uv).astype(np.int64)
        in_bounds = (
            positive
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < row["mask"].shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < row["mask"].shape[0])
        )
        frame_keep = np.zeros(len(grid_points), dtype=bool)
        if np.any(in_bounds):
            x = xy[in_bounds, 0]
            y = xy[in_bounds, 1]
            in_mask = row["mask"][y, x]
            depth = row["depth_m"][y, x].astype(np.float64)
            z = points_camera[in_bounds, 2]
            depth_valid = np.isfinite(depth) & (depth > float(args.min_depth_m))
            near_or_behind = z >= depth - float(args.depth_front_tolerance_m)
            not_too_far = z <= depth + float(args.depth_back_tolerance_m)
            accepted = in_mask & depth_valid & near_or_behind & not_too_far
            frame_keep[in_bounds] = accepted
            depth_hit = np.zeros(len(grid_points), dtype=bool)
            depth_hit[in_bounds] = in_mask & depth_valid
            depth_hits += depth_hit.astype(np.int16)
        keep_votes += frame_keep.astype(np.int16)
        rows.append(
            {
                "frame_idx": int(row["frame_idx"]),
                "mask_depth_voxels": int(np.count_nonzero(frame_keep)),
                "visible_depth_voxels": int(np.count_nonzero(in_bounds)),
            }
        )
    min_votes = int(args.min_carve_votes)
    min_depth_hits = int(args.min_depth_hits)
    occupied = (keep_votes >= min_votes) & (depth_hits >= min_depth_hits)
    return occupied.reshape(grid_shape), rows


def contact_constraints_prior(
    annotations: dict[int, dict],
    contact_report: Path | None,
    frame_rows: list[dict],
) -> np.ndarray:
    if contact_report is None:
        return np.zeros((0, 3), dtype=np.float64)
    pose_by_frame = {int(row["frame_idx"]): row for row in frame_rows}
    points = []
    for row in contact_rows(contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        patch_ids = selected_vertex_ids(row)
        if int(patch_ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} patch id exceeds MANO vertex count")
        patch_camera = vertices[patch_ids]
        pose = pose_by_frame[frame_idx]
        r = pose["rotation_prior_to_camera"]
        t = pose["translation_prior_to_camera"]
        points.append((patch_camera - t[None, :]) @ r)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(points).astype(np.float64)


def hand_meshes_prior(
    annotations: dict[int, dict],
    contact_report: Path | None,
    frame_rows: list[dict],
    faces: np.ndarray | None,
) -> list[trimesh.Trimesh]:
    if contact_report is None or faces is None:
        return []
    pose_by_frame = {int(row["frame_idx"]): row for row in frame_rows}
    meshes = []
    for row in contact_rows(contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices_camera = hand_vertices_camera(hand)
        if len(vertices_camera) <= int(faces.max()):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} has too few MANO vertices for face topology")
        pose = pose_by_frame[frame_idx]
        r = pose["rotation_prior_to_camera"]
        t = pose["translation_prior_to_camera"]
        vertices_prior = (vertices_camera - t[None, :]) @ r
        mesh = trimesh.Trimesh(vertices=vertices_prior.astype(np.float32), faces=faces.astype(np.int32), process=True)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} MANO mesh is empty")
        meshes.append(mesh)
    return meshes


def hand_surface_points_prior(
    hand_meshes: list[trimesh.Trimesh],
    samples_per_mesh: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if not hand_meshes:
        return np.zeros((0, 3), dtype=np.float64), {
            "hand_meshes": 0,
            "hand_surface_vertices": 0,
            "hand_surface_samples": 0,
        }
    rng = np.random.default_rng(int(seed))
    points = []
    total_vertices = 0
    total_samples = 0
    for hand_mesh in hand_meshes:
        vertices = np.asarray(hand_mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            raise RuntimeError("empty MANO hand mesh reached hand surface sampling")
        points.append(vertices)
        total_vertices += int(len(vertices))
        sample_count = int(samples_per_mesh)
        if sample_count > 0:
            state = np.random.get_state()
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            try:
                samples, _face_ids = sample_surface(hand_mesh, sample_count)
            finally:
                np.random.set_state(state)
            points.append(np.asarray(samples, dtype=np.float64))
            total_samples += sample_count
    surface = np.vstack(points).astype(np.float64)
    return surface, {
        "hand_meshes": int(len(hand_meshes)),
        "hand_surface_vertices": int(total_vertices),
        "hand_surface_samples": int(total_samples),
        "hand_surface_points": int(len(surface)),
    }


def apply_contact_constraints(
    occupied: np.ndarray,
    grid_points: np.ndarray,
    contact_points: np.ndarray,
    hand_surface: np.ndarray,
    hand_surface_row: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    flat = occupied.reshape(-1).copy()
    protect = np.zeros(len(grid_points), dtype=bool)
    if len(contact_points):
        diff = grid_points[:, None, :] - contact_points[None, :, :]
        nearest = np.linalg.norm(diff, axis=2).min(axis=1)
        protect = nearest <= float(args.contact_protect_radius_m)
        flat[protect] = True
    hand_excluded = np.zeros(len(grid_points), dtype=bool)
    if len(hand_surface) and float(args.hand_surface_exclusion_m) >= 0.0:
        margin = float(args.hand_surface_exclusion_m)
        if len(hand_surface) > int(args.max_hand_surface_points):
            raise RuntimeError(f"sampled hand surface has {len(hand_surface)} points, above max {args.max_hand_surface_points}")
        distance = cKDTree(hand_surface).query(grid_points, k=1)[0]
        hand_excluded = distance <= margin
        flat[hand_excluded & ~protect] = False
    return flat.reshape(occupied.shape), {
        "contact_points": int(len(contact_points)),
        "protected_voxels": int(np.count_nonzero(protect)),
        **hand_surface_row,
        "hand_surface_excluded_voxels": int(np.count_nonzero(hand_excluded & ~protect)),
        "contact_protect_radius_m": float(args.contact_protect_radius_m),
        "hand_surface_exclusion_m": float(args.hand_surface_exclusion_m),
        "hand_surface_samples_per_mesh": int(args.hand_surface_samples_per_mesh),
    }


def padded_grid_points(shape: tuple[int, int, int], origin: np.ndarray, pitch_m: float, pad: int) -> np.ndarray:
    axes = [
        origin[i] + (np.arange(shape[i], dtype=np.float64) - float(pad)) * float(pitch_m)
        for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]


def sample_sdf_grid(points: np.ndarray, sdf: np.ndarray, origin: np.ndarray, pitch_m: float, pad: int) -> np.ndarray:
    coords = ((np.asarray(points, dtype=np.float64) - origin[None, :]) / float(pitch_m)) + float(pad)
    return map_coordinates(
        sdf,
        [coords[:, 0], coords[:, 1], coords[:, 2]],
        order=1,
        mode="constant",
        cval=np.nan,
    ).astype(np.float64)


def max_gaussian_contact_pull(
    grid_points: np.ndarray,
    contact_points: np.ndarray,
    contact_values: np.ndarray,
    sigma_m: float,
    max_shift_m: float,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    sigma = float(sigma_m)
    if sigma <= 0.0:
        return np.zeros(len(grid_points), dtype=np.float32), {"active_contact_points": 0}
    finite = np.isfinite(contact_values)
    needed = np.clip(contact_values[finite], 0.0, float(max_shift_m))
    contacts = np.asarray(contact_points, dtype=np.float64)[finite]
    active = needed > 0.0
    needed = needed[active]
    contacts = contacts[active]
    if len(contacts) == 0:
        return np.zeros(len(grid_points), dtype=np.float32), {"active_contact_points": 0}
    pull = np.zeros(len(grid_points), dtype=np.float32)
    for start in range(0, len(grid_points), int(chunk_size)):
        stop = min(len(grid_points), start + int(chunk_size))
        diff = grid_points[start:stop, None, :] - contacts[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        weighted = needed[None, :] * np.exp(-0.5 * dist2 / (sigma * sigma))
        pull[start:stop] = np.max(weighted, axis=1).astype(np.float32)
    return pull, {
        "active_contact_points": int(len(contacts)),
        "contact_pull_needed_m": {
            "median": float(np.median(needed)),
            "p95": float(np.percentile(needed, 95.0)),
            "max": float(np.max(needed)),
        },
    }


def apply_sdf_boundary_constraints(
    sdf: np.ndarray,
    origin: np.ndarray,
    pitch_m: float,
    pad: int,
    contact_points: np.ndarray,
    hand_surface: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    row = {
        "enabled": False,
        "contact_points": int(len(contact_points)),
        "hand_surface_points": int(len(hand_surface)),
    }
    contact_sigma = float(args.sdf_contact_boundary_sigma_m)
    hand_clearance = float(args.sdf_hand_clearance_m)
    if contact_sigma <= 0.0 and hand_clearance < 0.0:
        return sdf, row
    if contact_sigma > 0.0 and len(contact_points) == 0:
        raise RuntimeError("sdf contact-boundary repair requested without contact points")
    if hand_clearance >= 0.0 and len(hand_surface) == 0:
        raise RuntimeError("sdf hand-clearance lower bound requested without sampled MANO hand surface")
    grid = padded_grid_points(tuple(int(v) for v in sdf.shape), origin, float(pitch_m), int(pad))
    repaired = sdf.reshape(-1).astype(np.float32).copy()
    contact_values_before = (
        sample_sdf_grid(contact_points, sdf, origin, pitch_m, pad) - float(args.sdf_contact_boundary_target_m)
        if len(contact_points)
        else np.zeros(0, dtype=np.float64)
    )
    if contact_sigma > 0.0:
        pull, pull_row = max_gaussian_contact_pull(
            grid,
            contact_points,
            contact_values_before,
            contact_sigma,
            float(args.sdf_contact_boundary_max_shift_m),
            int(args.sdf_constraint_chunk_size),
        )
        repaired -= pull
        row.update(pull_row)
    if hand_clearance >= 0.0:
        distance_to_hand = cKDTree(hand_surface).query(grid, k=1)[0]
        if len(contact_points):
            contact_distance = cKDTree(contact_points).query(grid, k=1)[0]
            contact_exemption = np.exp(
                -0.5 * (contact_distance / float(args.sdf_hand_clearance_contact_sigma_m)) ** 2
            )
        else:
            contact_exemption = np.zeros(len(grid), dtype=np.float64)
        lower = (hand_clearance - distance_to_hand) * (1.0 - contact_exemption)
        repaired = np.maximum(repaired, lower.astype(np.float32))
        row["hand_clearance_active_points"] = int(np.count_nonzero(lower > 0.0))
    repaired_sdf = repaired.reshape(sdf.shape)
    contact_values_after = (
        sample_sdf_grid(contact_points, repaired_sdf, origin, pitch_m, pad)
        if len(contact_points)
        else np.zeros(0, dtype=np.float64)
    )
    row.update(
        {
            "enabled": True,
            "contact_boundary_sigma_m": float(args.sdf_contact_boundary_sigma_m),
            "contact_boundary_target_m": float(args.sdf_contact_boundary_target_m),
            "contact_boundary_max_shift_m": float(args.sdf_contact_boundary_max_shift_m),
            "hand_clearance_m": float(args.sdf_hand_clearance_m),
            "hand_clearance_contact_sigma_m": float(args.sdf_hand_clearance_contact_sigma_m),
            "contact_sdf_before_m": summarize(contact_values_before + float(args.sdf_contact_boundary_target_m)),
            "contact_sdf_after_m": summarize(contact_values_after),
        }
    )
    return repaired_sdf, row


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


def mesh_from_occupancy(
    occupied: np.ndarray,
    origin: np.ndarray,
    pitch_m: float,
    contact_points: np.ndarray,
    hand_surface: np.ndarray,
    args: argparse.Namespace,
) -> tuple[trimesh.Trimesh, dict]:
    if int(np.count_nonzero(occupied)) == 0:
        raise RuntimeError("depth-carved occupancy is empty")
    occ = binary_closing(occupied, iterations=int(args.close_iterations))
    pad = int(args.sdf_pad_voxels)
    if pad < 2:
        raise RuntimeError("SDF extraction requires at least two pad voxels")
    occ_pad = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ_pad, sampling=[float(pitch_m)] * 3)
    inside = distance_transform_edt(occ_pad, sampling=[float(pitch_m)] * 3)
    sdf = outside - inside
    if float(args.sdf_smooth_sigma_voxels) > 0.0:
        sdf = gaussian_filter(sdf, sigma=float(args.sdf_smooth_sigma_voxels), mode="nearest")
    sdf, sdf_constraint_row = apply_sdf_boundary_constraints(
        sdf.astype(np.float32),
        origin,
        float(pitch_m),
        pad,
        contact_points,
        hand_surface,
        args,
    )
    vertices, faces, normals, _values = measure.marching_cubes(
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
        raise RuntimeError("depth-carved mesh has no connected components")
    mesh = max(components, key=lambda component: float(component.area))
    trimesh.repair.fix_normals(mesh)
    topo = topology(mesh)
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"depth-carved mesh is not topologically closed: {topo}")
    return mesh, sdf_constraint_row


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.graph_mesh_archive)
    prior_mesh = load_mesh(args.mesh_prior_camera)
    prior_vertices = np.asarray(prior_mesh.vertices, dtype=np.float64)
    bounds = prior_mesh.bounds.astype(np.float64)
    pad = float(args.grid_pad_m)
    lo = bounds[0] - pad
    hi = bounds[1] + pad
    pitch = float(args.pitch_m)
    axes = [np.arange(lo[i], hi[i] + 0.5 * pitch, pitch, dtype=np.float64) for i in range(3)]
    grid_shape = tuple(int(len(axis)) for axis in axes)
    if int(np.prod(grid_shape)) > int(args.max_grid_voxels):
        raise RuntimeError(f"grid has {int(np.prod(grid_shape))} voxels, above max {args.max_grid_voxels}")
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_points = np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]
    frame_rows = []
    pose_internal_rows = []
    pose_rows = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in manifest or frame_idx not in depths or frame_idx not in graph_meshes:
            continue
        depth_m, depth_intrinsics = depths[frame_idx]
        annotation = annotations[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        r, t, pose_row = compute_frame_pose(
            prior_vertices,
            graph_meshes[frame_idx][0],
            annotation,
            int(args.max_pose_correspondences),
            int(args.seed) + frame_idx,
        )
        frame_rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_prior_to_camera": r,
                "translation_prior_to_camera": t,
                "intrinsics": intrinsics,
                "mask": mask,
                "depth_m": depth_m,
            }
        )
        pose_rows.append(
            {
                "frame_idx": int(frame_idx),
                "object_translation_camera_m": t.astype(float).tolist(),
                "object_rotation_delta_rad": Rotation.from_matrix(r).as_rotvec().astype(float).tolist(),
                **pose_row,
            }
        )
        pose_internal_rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_prior_to_camera": r,
                "translation_prior_to_camera": t,
            }
        )
    if len(frame_rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(frame_rows)} carving frames available")
    mano_faces = load_mano_faces(args.mano_model) if args.mano_model is not None else None
    hand_meshes = hand_meshes_prior(annotations, args.contact_report, pose_internal_rows, mano_faces)
    hand_surface, hand_surface_row = hand_surface_points_prior(hand_meshes, int(args.hand_surface_samples_per_mesh), int(args.seed))
    if len(hand_surface) > int(args.max_hand_surface_points):
        raise RuntimeError(f"sampled hand surface has {len(hand_surface)} points, above max {args.max_hand_surface_points}")
    occupied, carve_rows = carve_volume(grid_points, grid_shape, frame_rows, args)
    contact_points = contact_constraints_prior(annotations, args.contact_report, pose_internal_rows)
    occupied, contact_constraint_row = apply_contact_constraints(occupied, grid_points, contact_points, hand_surface, hand_surface_row, args)
    mesh_prior, sdf_constraint_row = mesh_from_occupancy(occupied, lo, pitch, contact_points, hand_surface, args)
    meshes_world = []
    frame_ids = []
    for row in frame_rows:
        vertices_camera = np.asarray(mesh_prior.vertices, dtype=np.float64) @ row["rotation_prior_to_camera"].T + row["translation_prior_to_camera"][None, :]
        T_world_camera = np.asarray(annotations[row["frame_idx"]]["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_world = transform_points(vertices_camera, T_world_camera)
        meshes_world.append(trimesh.Trimesh(vertices=vertices_world.astype(np.float32), faces=np.asarray(mesh_prior.faces, dtype=np.int32), process=False))
        frame_ids.append(int(row["frame_idx"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = args.output_dir / "depth_carved_mesh_prior_frame.obj"
    mesh_prior.export(prior_path)
    archive_path = args.output_dir / "depth_carved_meshes_world.npz"
    save_archive(archive_path, frame_ids, meshes_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "object_visual_hull_depth_carve_v3",
        "claim_tested": "model-produced masks and metric depth can carve a closed object-centric mesh under solved object poses",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "graph_mesh_archive": str(args.graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "prior_frame_mesh": str(prior_path),
        "mesh_archive_world": str(archive_path),
        "frames": frame_ids,
        "grid_shape": [int(v) for v in grid_shape],
        "grid_voxels": int(np.prod(grid_shape)),
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "mesh_topology": topology(mesh_prior),
        "carve_rows": carve_rows,
        "contact_constraints": contact_constraint_row,
        "sdf_boundary_constraints": sdf_constraint_row,
        "pose_rows": pose_rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "grid_pad_m": float(args.grid_pad_m),
            "depth_front_tolerance_m": float(args.depth_front_tolerance_m),
            "depth_back_tolerance_m": float(args.depth_back_tolerance_m),
            "min_carve_votes": int(args.min_carve_votes),
            "min_depth_hits": int(args.min_depth_hits),
            "close_iterations": int(args.close_iterations),
            "sdf_smooth_sigma_voxels": float(args.sdf_smooth_sigma_voxels),
            "sdf_contact_boundary_sigma_m": float(args.sdf_contact_boundary_sigma_m),
            "sdf_contact_boundary_target_m": float(args.sdf_contact_boundary_target_m),
            "sdf_contact_boundary_max_shift_m": float(args.sdf_contact_boundary_max_shift_m),
            "sdf_hand_clearance_m": float(args.sdf_hand_clearance_m),
            "sdf_hand_clearance_contact_sigma_m": float(args.sdf_hand_clearance_contact_sigma_m),
        },
    }
    save_json(args.output_dir / "qc_depth_carved_mesh_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"carve_rows", "pose_rows"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--graph-mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path)
    parser.add_argument("--mano-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--grid-pad-m", type=float, default=0.012)
    parser.add_argument("--max-grid-voxels", type=int, default=5000000)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--depth-back-tolerance-m", type=float, default=0.030)
    parser.add_argument("--min-carve-votes", type=int, default=2)
    parser.add_argument("--min-depth-hits", type=int, default=2)
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument("--contact-protect-radius-m", type=float, default=0.0)
    parser.add_argument("--hand-surface-exclusion-m", type=float, default=-1.0)
    parser.add_argument("--hand-surface-samples-per-mesh", type=int, default=30000)
    parser.add_argument("--max-hand-surface-points", type=int, default=500000)
    parser.add_argument("--sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--sdf-smooth-sigma-voxels", type=float, default=0.5)
    parser.add_argument("--sdf-contact-boundary-sigma-m", type=float, default=0.0)
    parser.add_argument("--sdf-contact-boundary-target-m", type=float, default=0.0)
    parser.add_argument("--sdf-contact-boundary-max-shift-m", type=float, default=0.030)
    parser.add_argument("--sdf-hand-clearance-m", type=float, default=-1.0)
    parser.add_argument("--sdf-hand-clearance-contact-sigma-m", type=float, default=0.010)
    parser.add_argument("--sdf-constraint-chunk-size", type=int, default=200000)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=181)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
