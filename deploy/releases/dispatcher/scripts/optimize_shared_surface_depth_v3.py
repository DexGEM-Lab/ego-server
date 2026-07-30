#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from close_mesh_archive_with_voxel_fill_v3 import save_archive, topology, transform_points
from diagnose_contact_kinematics_v3 import selected_vertex_ids
from fit_mano_to_hand_mask_depth_v3 import load_mano_faces
from fuse_observed_surface_with_complete_prior_v3 import compute_frame_pose, intrinsics_for, read_mask
from optimize_contact_patch_object_pose_graph_v3 import (
    annotations_by_frame,
    contact_rows,
    hand_vertices_camera,
    load_depth_archive,
    manifest_by_frame,
)
from optimize_mesh_prior_pose_graph_v3 import load_mesh
from render_bundlesdf_mesh_qc_v3 import load_mesh_archive
from render_mesh_zbuffer_qc_v3 import triangle_zbuffer
from trimesh.sample import sample_surface


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    z = points_camera[:, 2].astype(np.float64)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    positive = z > 0.0
    uv[positive, 0] = fx * points_camera[positive, 0] / z[positive] + cx
    uv[positive, 1] = fy * points_camera[positive, 1] / z[positive] + cy
    return uv, positive


def unique_edges(faces: np.ndarray) -> np.ndarray:
    edge_sets = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64)
    edge_sets.sort(axis=1)
    return np.unique(edge_sets, axis=0)


def sparse_rows(row_count: int, cols: np.ndarray, vals: np.ndarray, n_vars: int) -> coo_matrix:
    rows = np.arange(row_count, dtype=np.int64)
    return coo_matrix((vals.astype(np.float64), (rows, cols.astype(np.int64))), shape=(row_count, n_vars))


def build_pose_rows(args: argparse.Namespace, annotations: dict[int, dict], graph_meshes: dict[int, tuple[np.ndarray, np.ndarray]], prior_vertices: np.ndarray) -> list[dict]:
    rows = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in annotations or frame_idx not in graph_meshes:
            continue
        r, t, pose_row = compute_frame_pose(
            prior_vertices,
            graph_meshes[frame_idx][0],
            annotations[frame_idx],
            int(args.max_pose_correspondences),
            int(args.seed) + frame_idx,
        )
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "rotation_prior_to_camera": r,
                "translation_prior_to_camera": t,
                "object_translation_camera_m": t.astype(float).tolist(),
                "object_rotation_delta_rad": Rotation.from_matrix(r).as_rotvec().astype(float).tolist(),
                **pose_row,
            }
        )
    if len(rows) < int(args.min_frames):
        raise RuntimeError(f"only {len(rows)} frame poses recovered")
    return rows


def depth_observation_block(
    args: argparse.Namespace,
    prior_vertices: np.ndarray,
    prior_normals: np.ndarray,
    faces: np.ndarray,
    annotations: dict[int, dict],
    manifest: dict[int, dict],
    depths: dict[int, tuple[np.ndarray, np.ndarray]],
    pose_rows: list[dict],
) -> tuple[coo_matrix, np.ndarray, dict]:
    cols_all = []
    vals_all = []
    rhs_all = []
    residual_before = []
    alpha_all = []
    frame_reports = []
    rng = np.random.default_rng(int(args.seed) + 17)
    n_vars = len(prior_vertices)
    row_offset = 0
    for pose in pose_rows:
        frame_idx = int(pose["frame_idx"])
        if frame_idx not in manifest or frame_idx not in depths:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        mask_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        vertices_camera = prior_vertices @ r.T + t[None, :]
        normals_camera = prior_normals @ r.T
        uv, positive = project(vertices_camera, intrinsics)
        z = vertices_camera[:, 2]
        zbuf = triangle_zbuffer(mask.shape, uv, z, faces, args.max_zbuffer_faces)
        xy = np.rint(uv).astype(np.int64)
        in_bounds = (
            positive
            & np.isfinite(uv).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < mask.shape[0])
        )
        candidate = np.zeros(len(prior_vertices), dtype=bool)
        if np.any(in_bounds):
            idx = np.flatnonzero(in_bounds)
            x = xy[idx, 0]
            y = xy[idx, 1]
            depth = depth_m[y, x].astype(np.float64)
            z_visible = zbuf[y, x].astype(np.float64)
            residual = z[idx].astype(np.float64) - depth
            alpha = normals_camera[idx, 2].astype(np.float64)
            keep = (
                mask[y, x]
                & (mask_distance[y, x] >= float(args.min_mask_distance_px))
                & np.isfinite(depth)
                & (depth > float(args.min_depth_m))
                & np.isfinite(z_visible)
                & (np.abs(z[idx] - z_visible) <= float(args.visibility_depth_tolerance_m))
                & (np.abs(alpha) >= float(args.min_depth_alpha))
                & (np.abs(residual) <= float(args.max_depth_residual_m))
            )
            candidate[idx[keep]] = True
        ids = np.flatnonzero(candidate)
        if len(ids) > int(args.max_depth_vertices_per_frame):
            ids = np.sort(rng.choice(ids, size=int(args.max_depth_vertices_per_frame), replace=False))
        if len(ids) == 0:
            frame_reports.append({"frame_idx": frame_idx, "depth_vertices": 0})
            continue
        x = xy[ids, 0]
        y = xy[ids, 1]
        depth = depth_m[y, x].astype(np.float64)
        alpha = normals_camera[ids, 2].astype(np.float64)
        before = z[ids].astype(np.float64) - depth
        cols_all.append(ids.astype(np.int64))
        vals_all.append(alpha / float(args.sigma_depth_m))
        rhs_all.append((depth - z[ids].astype(np.float64)) / float(args.sigma_depth_m))
        residual_before.append(before)
        alpha_all.append(alpha)
        frame_reports.append(
            {
                "frame_idx": frame_idx,
                "depth_vertices": int(len(ids)),
                "depth_residual_before_m": summarize(before),
                "normal_z_alpha": summarize(alpha),
            }
        )
        row_offset += len(ids)
    if row_offset == 0:
        raise RuntimeError("no visible depth observations selected")
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    rhs = np.concatenate(rhs_all)
    block = sparse_rows(row_offset, cols, vals, n_vars)
    report = {
        "rows": int(row_offset),
        "residual_before_m": summarize(np.concatenate(residual_before)),
        "normal_z_alpha": summarize(np.concatenate(alpha_all)),
        "frames": frame_reports,
    }
    return block, rhs, report


def contact_block(
    args: argparse.Namespace,
    prior_vertices: np.ndarray,
    prior_normals: np.ndarray,
    annotations: dict[int, dict],
    pose_rows: list[dict],
) -> tuple[coo_matrix | None, np.ndarray, dict]:
    if args.contact_report is None:
        return None, np.zeros(0, dtype=np.float64), {"rows": 0}
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows}
    tree = cKDTree(prior_vertices)
    cols = []
    rhs = []
    before = []
    frames = []
    for row in contact_rows(args.contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand = annotations[frame_idx]["hands"][int(row["hand_idx"])]
        patch_ids = selected_vertex_ids(row)
        vertices = hand_vertices_camera(hand)
        if int(patch_ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} contact row references invalid MANO vertex id")
        patch_camera = vertices[patch_ids]
        pose = pose_by_frame[frame_idx]
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        patch_prior = (patch_camera - t[None, :]) @ r
        _dist, nearest = tree.query(patch_prior, k=1)
        target = np.einsum("ij,ij->i", patch_prior - prior_vertices[nearest], prior_normals[nearest])
        cols.extend([int(v) for v in nearest.tolist()])
        rhs.extend([float(v) / float(args.sigma_contact_m) for v in target.tolist()])
        before.extend([-float(v) for v in target.tolist()])
        frames.append({"frame_idx": frame_idx, "contact_points": int(len(patch_prior)), "normal_target_m": summarize(target)})
    if not cols:
        return None, np.zeros(0, dtype=np.float64), {"rows": 0}
    vals = np.full(len(cols), 1.0 / float(args.sigma_contact_m), dtype=np.float64)
    block = sparse_rows(len(cols), np.asarray(cols, dtype=np.int64), vals, len(prior_vertices))
    return block, np.asarray(rhs, dtype=np.float64), {
        "rows": int(len(cols)),
        "residual_before_m": summarize(before),
        "frames": frames,
    }


def contact_patch_prior_points(args: argparse.Namespace, annotations: dict[int, dict], pose_rows: list[dict]) -> np.ndarray:
    if args.contact_report is None:
        return np.zeros((0, 3), dtype=np.float64)
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows}
    points = []
    for row in contact_rows(args.contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand = annotations[frame_idx]["hands"][int(row["hand_idx"])]
        patch_ids = selected_vertex_ids(row)
        vertices = hand_vertices_camera(hand)
        if int(patch_ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} contact row references invalid MANO vertex id")
        pose = pose_by_frame[frame_idx]
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        points.append((vertices[patch_ids] - t[None, :]) @ r)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(points).astype(np.float64)


def hand_surface_prior_points(
    args: argparse.Namespace,
    annotations: dict[int, dict],
    pose_rows: list[dict],
) -> tuple[np.ndarray, dict]:
    if args.contact_report is None or args.mano_model is None:
        return np.zeros((0, 3), dtype=np.float64), {"hand_meshes": 0, "hand_surface_points": 0}
    faces = load_mano_faces(args.mano_model)
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows}
    rng = np.random.default_rng(int(args.seed) + 71)
    points = []
    mesh_rows = []
    for row in contact_rows(args.contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices_camera = hand_vertices_camera(hand)
        if len(vertices_camera) <= int(faces.max()):
            raise RuntimeError(f"frame {frame_idx} hand {hand_idx} has too few MANO vertices for face topology")
        pose = pose_by_frame[frame_idx]
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        vertices_prior = (vertices_camera - t[None, :]) @ r
        mesh = trimesh.Trimesh(vertices=vertices_prior.astype(np.float32), faces=faces.astype(np.int32), process=False)
        points.append(vertices_prior.astype(np.float64))
        sample_count = int(args.hand_surface_samples_per_mesh)
        if sample_count > 0:
            state = np.random.get_state()
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            try:
                samples, _face_ids = sample_surface(mesh, sample_count)
            finally:
                np.random.set_state(state)
            points.append(np.asarray(samples, dtype=np.float64))
        mesh_rows.append({"frame_idx": frame_idx, "hand_idx": hand_idx, "vertices": int(len(vertices_prior)), "surface_samples": sample_count})
    if not points:
        return np.zeros((0, 3), dtype=np.float64), {"hand_meshes": 0, "hand_surface_points": 0}
    surface = np.vstack(points).astype(np.float64)
    if len(surface) > int(args.max_hand_surface_points):
        keep = np.sort(rng.choice(np.arange(len(surface), dtype=np.int64), size=int(args.max_hand_surface_points), replace=False))
        surface = surface[keep]
    return surface, {
        "hand_meshes": int(len(mesh_rows)),
        "hand_surface_points": int(len(surface)),
        "frames": mesh_rows,
    }


def hand_clearance_block(
    args: argparse.Namespace,
    prior_vertices: np.ndarray,
    prior_normals: np.ndarray,
    annotations: dict[int, dict],
    pose_rows: list[dict],
) -> tuple[coo_matrix | None, np.ndarray, dict]:
    hand_points, hand_report = hand_surface_prior_points(args, annotations, pose_rows)
    if len(hand_points) == 0:
        return None, np.zeros(0, dtype=np.float64), {"rows": 0, **hand_report}
    contact_points = contact_patch_prior_points(args, annotations, pose_rows)
    contact_tree = cKDTree(contact_points) if len(contact_points) else None
    tree = cKDTree(prior_vertices)
    distances, nearest = tree.query(hand_points, k=1)
    candidate = distances <= float(args.hand_clearance_radius_m)
    if contact_tree is not None:
        contact_distance, _ = contact_tree.query(hand_points, k=1)
        candidate &= contact_distance >= float(args.hand_contact_exclusion_radius_m)
    if not np.any(candidate):
        return None, np.zeros(0, dtype=np.float64), {"rows": 0, **hand_report}
    points = hand_points[candidate]
    vertex_ids = nearest[candidate].astype(np.int64)
    signed = np.einsum("ij,ij->i", points - prior_vertices[vertex_ids], prior_normals[vertex_ids])
    violating = signed < float(args.hand_clearance_m)
    if not np.any(violating):
        return None, np.zeros(0, dtype=np.float64), {
            **hand_report,
            "rows": 0,
            "candidate_pairs": int(len(points)),
            "signed_gap_before_m": summarize(signed),
        }
    vertex_ids = vertex_ids[violating]
    signed = signed[violating]
    if len(vertex_ids) > int(args.max_hand_clearance_pairs):
        order = np.argsort(signed)[: int(args.max_hand_clearance_pairs)]
        vertex_ids = vertex_ids[order]
        signed = signed[order]
    vals = np.full(len(vertex_ids), 1.0 / float(args.sigma_hand_clearance_m), dtype=np.float64)
    rhs = (signed - float(args.hand_clearance_m)) / float(args.sigma_hand_clearance_m)
    block = sparse_rows(len(vertex_ids), vertex_ids.astype(np.int64), vals, len(prior_vertices))
    return block, rhs.astype(np.float64), {
        **hand_report,
        "rows": int(len(vertex_ids)),
        "candidate_pairs": int(len(points)),
        "signed_gap_before_m": summarize(signed),
        "hand_clearance_m": float(args.hand_clearance_m),
        "hand_clearance_radius_m": float(args.hand_clearance_radius_m),
        "hand_contact_exclusion_radius_m": float(args.hand_contact_exclusion_radius_m),
    }


def boundary_vertex_ids(
    args: argparse.Namespace,
    prior_vertices: np.ndarray,
    faces: np.ndarray,
    annotations: dict[int, dict],
    manifest: dict[int, dict],
    depths: dict[int, tuple[np.ndarray, np.ndarray]],
    pose_rows: list[dict],
) -> tuple[np.ndarray, dict]:
    selected = np.zeros(len(prior_vertices), dtype=bool)
    frame_reports = []
    for pose in pose_rows:
        frame_idx = int(pose["frame_idx"])
        if frame_idx not in manifest or frame_idx not in depths:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        inside_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside_distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        signed_distance = np.where(mask, inside_distance, -outside_distance)
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        vertices_camera = prior_vertices @ r.T + t[None, :]
        uv, positive = project(vertices_camera, intrinsics)
        z = vertices_camera[:, 2]
        zbuf = triangle_zbuffer(mask.shape, uv, z, faces, args.max_zbuffer_faces)
        xy = np.rint(uv).astype(np.int64)
        in_bounds = (
            positive
            & np.isfinite(uv).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < mask.shape[0])
        )
        frame_selected = np.zeros(len(prior_vertices), dtype=bool)
        if np.any(in_bounds):
            idx = np.flatnonzero(in_bounds)
            x = xy[idx, 0]
            y = xy[idx, 1]
            visible = np.isfinite(zbuf[y, x]) & (np.abs(z[idx] - zbuf[y, x].astype(np.float64)) <= float(args.visibility_depth_tolerance_m))
            near_boundary = np.abs(signed_distance[y, x]) <= float(args.boundary_preserve_distance_px)
            frame_selected[idx[visible & near_boundary]] = True
        selected |= frame_selected
        frame_reports.append({"frame_idx": frame_idx, "boundary_vertices": int(np.count_nonzero(frame_selected))})
    ids = np.flatnonzero(selected)
    return ids.astype(np.int64), {
        "rows": int(len(ids)),
        "frames": frame_reports,
        "boundary_preserve_distance_px": float(args.boundary_preserve_distance_px),
    }


def displacement_block(n_vars: int, sigma_m: float) -> tuple[coo_matrix, np.ndarray]:
    cols = np.arange(n_vars, dtype=np.int64)
    vals = np.full(n_vars, 1.0 / float(sigma_m), dtype=np.float64)
    return sparse_rows(n_vars, cols, vals, n_vars), np.zeros(n_vars, dtype=np.float64)


def smoothness_block(edges: np.ndarray, n_vars: int, sigma_m: float) -> tuple[coo_matrix, np.ndarray]:
    row = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
    col = edges.reshape(-1).astype(np.int64)
    val = np.tile(np.asarray([1.0, -1.0], dtype=np.float64), len(edges)) / float(sigma_m)
    block = coo_matrix((val, (row, col)), shape=(len(edges), n_vars))
    return block, np.zeros(len(edges), dtype=np.float64)


def selected_displacement_block(ids: np.ndarray, n_vars: int, sigma_m: float) -> tuple[coo_matrix, np.ndarray]:
    vals = np.full(len(ids), 1.0 / float(sigma_m), dtype=np.float64)
    return sparse_rows(len(ids), ids.astype(np.int64), vals, n_vars), np.zeros(len(ids), dtype=np.float64)


def apply_mesh_to_world_archive(
    args: argparse.Namespace,
    mesh_prior: trimesh.Trimesh,
    annotations: dict[int, dict],
    pose_rows: list[dict],
) -> Path:
    meshes_world = []
    frame_ids = []
    prior_vertices = np.asarray(mesh_prior.vertices, dtype=np.float64)
    faces = np.asarray(mesh_prior.faces, dtype=np.int32)
    for pose in pose_rows:
        frame_idx = int(pose["frame_idx"])
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        vertices_camera = prior_vertices @ r.T + t[None, :]
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_world = transform_points(vertices_camera, T_world_camera)
        meshes_world.append(trimesh.Trimesh(vertices=vertices_world.astype(np.float32), faces=faces, process=False))
        frame_ids.append(frame_idx)
    archive_path = args.output_dir / "shared_surface_depth_meshes_world.npz"
    save_archive(archive_path, frame_ids, meshes_world)
    return archive_path


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.graph_mesh_archive)
    prior_mesh = load_mesh(args.mesh_prior_camera)
    prior_vertices = np.asarray(prior_mesh.vertices, dtype=np.float64)
    faces = np.asarray(prior_mesh.faces, dtype=np.int32)
    if len(prior_vertices) == 0 or len(faces) == 0:
        raise RuntimeError("prior mesh is empty")
    prior_normals = np.asarray(prior_mesh.vertex_normals, dtype=np.float64)
    norm = np.linalg.norm(prior_normals, axis=1)
    if prior_normals.shape != prior_vertices.shape or np.any(norm < 1e-8):
        raise RuntimeError("prior mesh has invalid vertex normals")
    prior_normals = prior_normals / norm[:, None]
    pose_rows = build_pose_rows(args, annotations, graph_meshes, prior_vertices)
    depth_A, depth_b, depth_report = depth_observation_block(
        args,
        prior_vertices,
        prior_normals,
        faces,
        annotations,
        manifest,
        depths,
        pose_rows,
    )
    blocks = [depth_A]
    rhs = [depth_b]
    contact_A, contact_b, contact_report = contact_block(args, prior_vertices, prior_normals, annotations, pose_rows)
    if contact_A is not None:
        blocks.append(contact_A)
        rhs.append(contact_b)
    hand_A, hand_b, hand_report = hand_clearance_block(args, prior_vertices, prior_normals, annotations, pose_rows)
    if hand_A is not None:
        blocks.append(hand_A)
        rhs.append(hand_b)
    boundary_ids, boundary_report = boundary_vertex_ids(args, prior_vertices, faces, annotations, manifest, depths, pose_rows)
    if len(boundary_ids) and float(args.sigma_boundary_displacement_m) > 0.0:
        boundary_A, boundary_b = selected_displacement_block(
            boundary_ids,
            len(prior_vertices),
            float(args.sigma_boundary_displacement_m),
        )
        blocks.append(boundary_A)
        rhs.append(boundary_b)
    else:
        boundary_A = None
    disp_A, disp_b = displacement_block(len(prior_vertices), float(args.sigma_displacement_m))
    blocks.append(disp_A)
    rhs.append(disp_b)
    edges = unique_edges(faces)
    if len(edges) > int(args.max_smooth_edges):
        rng = np.random.default_rng(int(args.seed) + 41)
        keep = np.sort(rng.choice(np.arange(len(edges), dtype=np.int64), size=int(args.max_smooth_edges), replace=False))
        edges = edges[keep]
    smooth_A, smooth_b = smoothness_block(edges, len(prior_vertices), float(args.sigma_edge_smooth_m))
    blocks.append(smooth_A)
    rhs.append(smooth_b)
    A = vstack(blocks, format="csr")
    b = np.concatenate(rhs)
    result = lsq_linear(
        A,
        b,
        bounds=(-float(args.max_normal_displacement_m), float(args.max_normal_displacement_m)),
        lsmr_tol="auto",
        max_iter=int(args.max_iter),
        verbose=0,
    )
    displacement = np.asarray(result.x, dtype=np.float64)
    optimized_vertices = prior_vertices + prior_normals * displacement[:, None]
    optimized_mesh = trimesh.Trimesh(vertices=optimized_vertices.astype(np.float32), faces=faces, process=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = args.output_dir / "shared_surface_depth_prior_frame.obj"
    optimized_mesh.export(prior_path)
    archive_path = apply_mesh_to_world_archive(args, optimized_mesh, annotations, pose_rows)
    depth_after = (depth_A @ displacement - depth_b) * float(args.sigma_depth_m)
    contact_after = np.zeros(0, dtype=np.float64)
    if contact_A is not None:
        contact_after = (contact_A @ displacement - contact_b) * float(args.sigma_contact_m)
    hand_after = np.zeros(0, dtype=np.float64)
    if hand_A is not None:
        hand_after = (hand_A @ displacement - hand_b) * float(args.sigma_hand_clearance_m)
    boundary_after = np.zeros(0, dtype=np.float64)
    if boundary_A is not None:
        boundary_after = (boundary_A @ displacement) * float(args.sigma_boundary_displacement_m)
    smooth_after = (smooth_A @ displacement) * float(args.sigma_edge_smooth_m)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_shared_surface_depth_v3",
        "claim_tested": "one shared object-frame surface displacement field can reduce visible-depth residual while preserving strict MANO contact evidence",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "graph_mesh_archive": str(args.graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "contact_report": None if args.contact_report is None else str(args.contact_report),
        "intrinsics_source": str(args.intrinsics_source),
        "prior_frame_mesh": str(prior_path),
        "mesh_archive_world": str(archive_path),
        "frames": [int(row["frame_idx"]) for row in pose_rows],
        "solver": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nit": int(result.nit),
            "matrix_shape": [int(v) for v in A.shape],
        },
        "mesh_topology": topology(optimized_mesh),
        "depth_observations": {
            **depth_report,
            "residual_after_m": summarize(depth_after),
        },
        "contact_observations": {
            **contact_report,
            "residual_after_m": summarize(contact_after),
        },
        "hand_clearance_observations": {
            **hand_report,
            "residual_after_m": summarize(hand_after),
        },
        "boundary_preservation": {
            **boundary_report,
            "normal_displacement_after_m": summarize(boundary_after),
            "normal_displacement_abs_after_m": summarize(np.abs(boundary_after)),
        },
        "regularization": {
            "normal_displacement_m": summarize(displacement),
            "normal_displacement_abs_m": summarize(np.abs(displacement)),
            "edge_delta_m": summarize(smooth_after),
            "smooth_edges": int(len(edges)),
        },
        "pose_rows": [
            {
                "frame_idx": int(row["frame_idx"]),
                "object_translation_camera_m": row["object_translation_camera_m"],
                "object_rotation_delta_rad": row["object_rotation_delta_rad"],
                "pose_recovery_median_error_m": row["pose_recovery_median_error_m"],
                "pose_correspondences": row["pose_correspondences"],
            }
            for row in pose_rows
        ],
        "parameters": {
            "sigma_depth_m": float(args.sigma_depth_m),
            "sigma_contact_m": float(args.sigma_contact_m),
            "sigma_hand_clearance_m": float(args.sigma_hand_clearance_m),
            "sigma_boundary_displacement_m": float(args.sigma_boundary_displacement_m),
            "sigma_displacement_m": float(args.sigma_displacement_m),
            "sigma_edge_smooth_m": float(args.sigma_edge_smooth_m),
            "max_normal_displacement_m": float(args.max_normal_displacement_m),
            "min_mask_distance_px": float(args.min_mask_distance_px),
            "visibility_depth_tolerance_m": float(args.visibility_depth_tolerance_m),
            "min_depth_alpha": float(args.min_depth_alpha),
            "max_depth_residual_m": float(args.max_depth_residual_m),
            "boundary_preserve_distance_px": float(args.boundary_preserve_distance_px),
            "hand_clearance_m": float(args.hand_clearance_m),
            "hand_clearance_radius_m": float(args.hand_clearance_radius_m),
            "hand_contact_exclusion_radius_m": float(args.hand_contact_exclusion_radius_m),
            "hand_surface_samples_per_mesh": int(args.hand_surface_samples_per_mesh),
            "max_depth_vertices_per_frame": int(args.max_depth_vertices_per_frame),
            "max_smooth_edges": int(args.max_smooth_edges),
        },
    }
    save_json(args.output_dir / "qc_shared_surface_depth_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"pose_rows"}}, indent=2))
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
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--max-zbuffer-faces", type=int, default=0)
    parser.add_argument("--min-depth-m", type=float, default=0.050)
    parser.add_argument("--min-mask-distance-px", type=float, default=3.0)
    parser.add_argument("--visibility-depth-tolerance-m", type=float, default=0.010)
    parser.add_argument("--min-depth-alpha", type=float, default=0.10)
    parser.add_argument("--max-depth-residual-m", type=float, default=0.150)
    parser.add_argument("--max-depth-vertices-per-frame", type=int, default=14000)
    parser.add_argument("--sigma-depth-m", type=float, default=0.006)
    parser.add_argument("--sigma-contact-m", type=float, default=0.003)
    parser.add_argument("--sigma-hand-clearance-m", type=float, default=0.003)
    parser.add_argument("--sigma-boundary-displacement-m", type=float, default=0.006)
    parser.add_argument("--sigma-displacement-m", type=float, default=0.040)
    parser.add_argument("--sigma-edge-smooth-m", type=float, default=0.006)
    parser.add_argument("--max-normal-displacement-m", type=float, default=0.060)
    parser.add_argument("--boundary-preserve-distance-px", type=float, default=8.0)
    parser.add_argument("--hand-clearance-m", type=float, default=0.004)
    parser.add_argument("--hand-clearance-radius-m", type=float, default=0.020)
    parser.add_argument("--hand-contact-exclusion-radius-m", type=float, default=0.018)
    parser.add_argument("--hand-surface-samples-per-mesh", type=int, default=900)
    parser.add_argument("--max-hand-surface-points", type=int, default=12000)
    parser.add_argument("--max-hand-clearance-pairs", type=int, default=6000)
    parser.add_argument("--max-smooth-edges", type=int, default=120000)
    parser.add_argument("--max-iter", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_zbuffer_faces is not None and int(args.max_zbuffer_faces) <= 0:
        args.max_zbuffer_faces = None
    run(args)


if __name__ == "__main__":
    main()
