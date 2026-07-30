#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage import measure
import trimesh
from trimesh.sample import sample_surface

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


def backproject_pixels(xy: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    z = depth.astype(np.float64)
    return np.c_[(xy[:, 0].astype(np.float64) - cx) * z / fx, (xy[:, 1].astype(np.float64) - cy) * z / fy, z]


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


def camera_to_prior(points_camera: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points_camera - t[None, :]) @ r


def sdf_from_mesh(mesh: trimesh.Trimesh, pitch: float, pad_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vox = mesh.voxelized(pitch=float(pitch)).fill()
    occ = np.asarray(vox.matrix, dtype=bool)
    if np.count_nonzero(occ) == 0:
        raise RuntimeError("voxelized mesh has no occupied cells")
    pad = int(pad_voxels)
    occ_pad = np.pad(occ, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occ_pad, sampling=[float(pitch)] * 3)
    inside = distance_transform_edt(occ_pad, sampling=[float(pitch)] * 3)
    sdf = outside - inside
    transform = np.asarray(vox.transform, dtype=np.float64).copy()
    transform[:3, 3] -= float(pitch) * pad
    return sdf.astype(np.float32), transform, occ_pad


def grid_points_from_transform(shape: tuple[int, int, int], transform: np.ndarray) -> np.ndarray:
    pitch = float(transform[0, 0])
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    axes = [origin[i] + pitch * np.arange(int(shape[i]), dtype=np.float64) for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]


def interp_rows(points: np.ndarray, sdf: np.ndarray, transform: np.ndarray, var_lookup: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    shape = tuple(int(v) for v in sdf.shape)
    pitch = float(transform[0, 0])
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    coords = (np.asarray(points, dtype=np.float64) - origin[None, :]) / pitch
    base = np.floor(coords).astype(np.int64)
    frac = coords - base.astype(np.float64)
    in_bounds = (
        (base[:, 0] >= 0)
        & (base[:, 0] + 1 < shape[0])
        & (base[:, 1] >= 0)
        & (base[:, 1] + 1 < shape[1])
        & (base[:, 2] >= 0)
        & (base[:, 2] + 1 < shape[2])
    )
    corners = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.int64,
    )
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    fixed_values = []
    kept = []
    for point_idx in np.flatnonzero(in_bounds):
        b = base[point_idx]
        f = frac[point_idx]
        corner_weights = np.asarray(
            [
                (1.0 - f[0]) * (1.0 - f[1]) * (1.0 - f[2]),
                f[0] * (1.0 - f[1]) * (1.0 - f[2]),
                (1.0 - f[0]) * f[1] * (1.0 - f[2]),
                f[0] * f[1] * (1.0 - f[2]),
                (1.0 - f[0]) * (1.0 - f[1]) * f[2],
                f[0] * (1.0 - f[1]) * f[2],
                (1.0 - f[0]) * f[1] * f[2],
                f[0] * f[1] * f[2],
            ],
            dtype=np.float64,
        )
        row_cols = []
        row_weights = []
        fixed = 0.0
        for corner, weight in zip(corners, corner_weights, strict=True):
            ijk = b + corner
            local_col = int(var_lookup[ijk[0], ijk[1], ijk[2]])
            if local_col >= 0:
                row_cols.append(local_col)
                row_weights.append(float(weight))
            fixed += float(weight) * float(sdf[ijk[0], ijk[1], ijk[2]])
        if row_cols:
            cols.append(np.asarray(row_cols, dtype=np.int64))
            weights.append(np.asarray(row_weights, dtype=np.float64))
            fixed_values.append(fixed)
            kept.append(int(point_idx))
    return cols, weights, np.asarray(fixed_values, dtype=np.float64), np.asarray(kept, dtype=np.int64)


def append_interp_equations(
    rows: list[int],
    cols: list[int],
    data: list[float],
    rhs: list[float],
    row_idx: int,
    interp_cols: list[np.ndarray],
    interp_weights: list[np.ndarray],
    fixed: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> int:
    inv = 1.0 / float(scale)
    for local_cols, local_weights, fixed_value, target_value in zip(interp_cols, interp_weights, fixed, target, strict=True):
        for col, weight in zip(local_cols, local_weights, strict=True):
            rows.append(row_idx)
            cols.append(int(col))
            data.append(float(weight) * inv)
        rhs.append((float(target_value) - float(fixed_value)) * inv)
        row_idx += 1
    return row_idx


def neighbor_edges(var_lookup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for axis in range(3):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(0, -1)
        slicer_b[axis] = slice(1, None)
        a = var_lookup[tuple(slicer_a)]
        b = var_lookup[tuple(slicer_b)]
        mask = (a >= 0) & (b >= 0)
        if np.any(mask):
            pairs.append(np.c_[a[mask], b[mask]])
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edges = np.vstack(pairs).astype(np.int64)
    return edges[:, 0], edges[:, 1]


def contact_points_prior(args: argparse.Namespace, annotations: dict[int, dict], pose_rows: list[dict]) -> tuple[np.ndarray, dict]:
    if args.contact_report is None:
        return np.zeros((0, 3), dtype=np.float64), {"contact_points": 0}
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows}
    points = []
    rows = []
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
        patch_prior = camera_to_prior(vertices[patch_ids], r, t)
        points.append(patch_prior)
        rows.append({"frame_idx": frame_idx, "hand_idx": int(row["hand_idx"]), "points": int(len(patch_prior))})
    if not points:
        return np.zeros((0, 3), dtype=np.float64), {"contact_points": 0}
    out = np.vstack(points).astype(np.float64)
    return out, {"contact_points": int(len(out)), "frames": rows}


def hand_points_prior(args: argparse.Namespace, annotations: dict[int, dict], pose_rows: list[dict], contact_points: np.ndarray) -> tuple[np.ndarray, dict]:
    if args.contact_report is None or args.mano_model is None:
        return np.zeros((0, 3), dtype=np.float64), {"hand_points": 0}
    faces = load_mano_faces(args.mano_model)
    pose_by_frame = {int(row["frame_idx"]): row for row in pose_rows}
    contact_tree = cKDTree(contact_points) if len(contact_points) else None
    rng = np.random.default_rng(int(args.seed) + 91)
    points = []
    rows = []
    for row in contact_rows(args.contact_report):
        frame_idx = int(row["frame_idx"])
        if frame_idx not in pose_by_frame or frame_idx not in annotations:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices_camera = hand_vertices_camera(hand)
        pose = pose_by_frame[frame_idx]
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        vertices_prior = camera_to_prior(vertices_camera, r, t)
        mesh = trimesh.Trimesh(vertices=vertices_prior.astype(np.float32), faces=faces.astype(np.int32), process=False)
        hand_points = [vertices_prior.astype(np.float64)]
        sample_count = int(args.hand_surface_samples_per_mesh)
        if sample_count > 0:
            state = np.random.get_state()
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            try:
                samples, _ = sample_surface(mesh, sample_count)
            finally:
                np.random.set_state(state)
            hand_points.append(np.asarray(samples, dtype=np.float64))
        frame_points = np.vstack(hand_points).astype(np.float64)
        if contact_tree is not None:
            contact_distance, _ = contact_tree.query(frame_points, k=1)
            frame_points = frame_points[contact_distance >= float(args.hand_contact_exclusion_radius_m)]
        points.append(frame_points)
        rows.append({"frame_idx": frame_idx, "hand_idx": hand_idx, "points": int(len(frame_points))})
    if not points:
        return np.zeros((0, 3), dtype=np.float64), {"hand_points": 0}
    out = np.vstack(points).astype(np.float64)
    if len(out) > int(args.max_hand_points):
        keep = np.sort(rng.choice(np.arange(len(out), dtype=np.int64), size=int(args.max_hand_points), replace=False))
        out = out[keep]
    return out, {"hand_points": int(len(out)), "frames": rows}


def visible_depth_points_prior(
    args: argparse.Namespace,
    annotations: dict[int, dict],
    manifest: dict[int, dict],
    depths: dict[int, tuple[np.ndarray, np.ndarray]],
    pose_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(int(args.seed) + 121)
    surface_points = []
    free_space_points = []
    rows = []
    for pose in pose_rows:
        frame_idx = int(pose["frame_idx"])
        if frame_idx not in manifest or frame_idx not in depths:
            continue
        annotation = annotations[frame_idx]
        depth_m, depth_intrinsics = depths[frame_idx]
        intrinsics = intrinsics_for(annotation, depth_intrinsics, str(args.intrinsics_source))
        mask = read_mask(Path(manifest[frame_idx]["mask"]), depth_m.shape)
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        valid = mask & (dist >= float(args.min_mask_distance_px)) & np.isfinite(depth_m) & (depth_m > float(args.min_depth_m))
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            rows.append({"frame_idx": frame_idx, "visible_points": 0})
            continue
        xy = np.c_[xs, ys]
        if len(xy) > int(args.max_visible_points_per_frame):
            xy = xy[np.sort(rng.choice(np.arange(len(xy), dtype=np.int64), size=int(args.max_visible_points_per_frame), replace=False))]
        depth = depth_m[xy[:, 1], xy[:, 0]].astype(np.float64)
        camera_points = backproject_pixels(xy, depth, intrinsics)
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        prior_points = camera_to_prior(camera_points, r, t)
        surface_points.append(prior_points)
        frame_free_points = []
        if int(args.free_space_samples_per_visible) > 0:
            offsets = np.linspace(
                float(args.free_space_min_offset_m),
                float(args.free_space_max_offset_m),
                int(args.free_space_samples_per_visible),
                dtype=np.float64,
            )
            for offset in offsets:
                free_depth = depth - float(offset)
                free_ok = free_depth > float(args.min_depth_m)
                if np.any(free_ok):
                    frame_free_points.append(camera_to_prior(backproject_pixels(xy[free_ok], free_depth[free_ok], intrinsics), r, t))
        if frame_free_points:
            free_space_points.append(np.vstack(frame_free_points).astype(np.float64))
        rows.append(
            {
                "frame_idx": frame_idx,
                "visible_points": int(len(prior_points)),
                "free_space_points": int(sum(len(v) for v in frame_free_points)),
            }
        )
    if not surface_points:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, {"visible_points": 0, "free_space_points": 0}
    surface_out = np.vstack(surface_points).astype(np.float64)
    free_out = np.vstack(free_space_points).astype(np.float64) if free_space_points else np.zeros((0, 3), dtype=np.float64)
    return surface_out, free_out, {"visible_points": int(len(surface_out)), "free_space_points": int(len(free_out)), "frames": rows}


def variable_mask_for_points(sdf: np.ndarray, transform: np.ndarray, points: np.ndarray, radius_m: float) -> tuple[np.ndarray, dict]:
    grid = grid_points_from_transform(tuple(int(v) for v in sdf.shape), transform)
    if len(points) == 0:
        raise RuntimeError("variable mask needs at least one point")
    distance = cKDTree(points).query(grid, k=1, workers=-1)[0]
    mask = distance <= float(radius_m)
    if int(np.count_nonzero(mask)) == 0:
        raise RuntimeError("shared SDF variable mask is empty")
    return mask.reshape(sdf.shape), {
        "anchor_points": int(len(points)),
        "edit_radius_m": float(radius_m),
        "variable_nodes": int(np.count_nonzero(mask)),
    }


def solve_sdf_delta(
    args: argparse.Namespace,
    sdf: np.ndarray,
    transform: np.ndarray,
    var_mask: np.ndarray,
    visible_points: np.ndarray,
    free_space_points: np.ndarray,
    contact_points: np.ndarray,
    hand_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    var_flat = np.flatnonzero(var_mask.reshape(-1))
    if len(var_flat) > int(args.max_variables):
        raise RuntimeError(f"shared SDF solve has {len(var_flat)} variables, above max {args.max_variables}")
    var_lookup = np.full(sdf.shape, -1, dtype=np.int64)
    var_lookup.reshape(-1)[var_flat] = np.arange(len(var_flat), dtype=np.int64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_idx = 0

    visible_cols, visible_weights, visible_fixed, visible_kept = interp_rows(visible_points, sdf, transform, var_lookup)
    if len(visible_cols):
        target = np.full(len(visible_cols), float(args.visible_sdf_target_m), dtype=np.float64)
        row_idx = append_interp_equations(rows, cols, data, rhs, row_idx, visible_cols, visible_weights, visible_fixed, target, float(args.visible_residual_scale_m))
    free_cols, free_weights, free_fixed, free_kept = interp_rows(free_space_points, sdf, transform, var_lookup)
    contact_cols, contact_weights, contact_fixed, contact_kept = interp_rows(contact_points, sdf, transform, var_lookup)
    if len(contact_points) and len(contact_cols) < int(args.min_controllable_contact_points):
        raise RuntimeError(f"only {len(contact_cols)} controllable contact points")
    if len(contact_cols):
        target = np.full(len(contact_cols), float(args.contact_sdf_target_m), dtype=np.float64)
        row_idx = append_interp_equations(rows, cols, data, rhs, row_idx, contact_cols, contact_weights, contact_fixed, target, float(args.contact_residual_scale_m))
    hand_cols, hand_weights, hand_fixed, hand_kept = interp_rows(hand_points, sdf, transform, var_lookup)
    left, right = neighbor_edges(var_lookup)

    def values_after(row_cols: list[np.ndarray], row_weights: list[np.ndarray], fixed: np.ndarray) -> np.ndarray:
        if not row_cols:
            return np.zeros(0, dtype=np.float64)
        return fixed + np.asarray([float(np.dot(w, solution[c])) for c, w in zip(row_cols, row_weights, strict=True)], dtype=np.float64)

    solution = np.zeros(len(var_flat), dtype=np.float64)
    current_free = free_fixed.copy()
    current_hand = hand_fixed.copy()
    iterations = []
    preserve = 1.0 / float(args.preserve_scale_m)
    smooth = 1.0 / float(args.smooth_delta_scale_m)
    for iteration in range(int(args.active_set_iterations)):
        iter_rows = list(rows)
        iter_cols = list(cols)
        iter_data = list(data)
        iter_rhs = list(rhs)
        iter_row_idx = row_idx
        free_active = current_free < float(args.free_space_sdf_target_m) + float(args.free_space_active_margin_m)
        if len(free_cols) and np.any(free_active):
            ids = np.flatnonzero(free_active)
            target = np.full(len(ids), float(args.free_space_sdf_target_m), dtype=np.float64)
            iter_row_idx = append_interp_equations(
                iter_rows,
                iter_cols,
                iter_data,
                iter_rhs,
                iter_row_idx,
                [free_cols[i] for i in ids],
                [free_weights[i] for i in ids],
                free_fixed[ids],
                target,
                float(args.free_space_residual_scale_m),
            )
        hand_active = current_hand < float(args.hand_clearance_m) + float(args.hand_active_margin_m)
        if len(hand_cols) and np.any(hand_active):
            ids = np.flatnonzero(hand_active)
            target = np.full(len(ids), float(args.hand_clearance_m), dtype=np.float64)
            iter_row_idx = append_interp_equations(
                iter_rows,
                iter_cols,
                iter_data,
                iter_rhs,
                iter_row_idx,
                [hand_cols[i] for i in ids],
                [hand_weights[i] for i in ids],
                hand_fixed[ids],
                target,
                float(args.hand_clearance_residual_scale_m),
            )
        for col in range(len(var_flat)):
            iter_rows.append(iter_row_idx)
            iter_cols.append(col)
            iter_data.append(preserve)
            iter_rhs.append(0.0)
            iter_row_idx += 1
        for a, b in zip(left, right, strict=True):
            iter_rows.append(iter_row_idx)
            iter_cols.append(int(a))
            iter_data.append(smooth)
            iter_rows.append(iter_row_idx)
            iter_cols.append(int(b))
            iter_data.append(-smooth)
            iter_rhs.append(0.0)
            iter_row_idx += 1

        system = sparse.coo_matrix(
            (
                np.asarray(iter_data, dtype=np.float64),
                (np.asarray(iter_rows, dtype=np.int64), np.asarray(iter_cols, dtype=np.int64)),
            ),
            shape=(iter_row_idx, len(var_flat)),
        ).tocsr()
        solution = lsmr(
            system,
            np.asarray(iter_rhs, dtype=np.float64),
            atol=float(args.lsmr_tol),
            btol=float(args.lsmr_tol),
            maxiter=int(args.lsmr_maxiter),
        )[0]
        if float(args.max_delta_m) > 0.0:
            solution = np.clip(solution, -float(args.max_delta_m), float(args.max_delta_m))
        current_free = values_after(free_cols, free_weights, free_fixed)
        current_hand = values_after(hand_cols, hand_weights, hand_fixed)
        iterations.append(
            {
                "iteration": int(iteration),
                "rows": int(iter_row_idx),
                "active_free_space_points": int(np.count_nonzero(free_active)) if len(free_cols) else 0,
                "free_space_sdf_m": summarize(current_free),
                "active_hand_points": int(np.count_nonzero(hand_active)) if len(hand_cols) else 0,
                "hand_sdf_m": summarize(current_hand),
                "delta_m": summarize(solution),
            }
        )

    visible_after = values_after(visible_cols, visible_weights, visible_fixed)
    free_after = values_after(free_cols, free_weights, free_fixed)
    contact_after = values_after(contact_cols, contact_weights, contact_fixed)
    hand_after = values_after(hand_cols, hand_weights, hand_fixed)
    delta = np.zeros(sdf.size, dtype=np.float32)
    delta[var_flat] = solution.astype(np.float32)
    return delta.reshape(sdf.shape), {
        "variables": int(len(var_flat)),
        "system_rows": int(row_idx),
        "visible_points": int(len(visible_points)),
        "controllable_visible_points": int(len(visible_cols)),
        "visible_sdf_before_m": summarize(visible_fixed),
        "visible_sdf_after_m": summarize(visible_after),
        "free_space_points": int(len(free_space_points)),
        "controllable_free_space_points": int(len(free_cols)),
        "free_space_sdf_before_m": summarize(free_fixed),
        "free_space_sdf_after_m": summarize(free_after),
        "free_space_violation_fraction_after": float(np.mean(free_after < float(args.free_space_sdf_target_m))) if len(free_after) else 0.0,
        "contact_points": int(len(contact_points)),
        "controllable_contact_points": int(len(contact_cols)),
        "contact_sdf_before_m": summarize(contact_fixed),
        "contact_sdf_after_m": summarize(contact_after),
        "hand_points": int(len(hand_points)),
        "controllable_hand_points": int(len(hand_cols)),
        "active_hand_points": int(np.count_nonzero(hand_active)) if len(hand_cols) else 0,
        "hand_sdf_before_m": summarize(hand_fixed),
        "hand_sdf_after_m": summarize(hand_after),
        "hand_clearance_violation_fraction_after": float(np.mean(hand_after < float(args.hand_clearance_m))) if len(hand_after) else 0.0,
        "delta_m": summarize(solution),
        "smooth_edges": int(len(left)),
        "active_set_iterations": iterations,
    }


def extract_mesh(sdf: np.ndarray, transform: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    pitch = float(transform[0, 0])
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    vertices, faces, normals, _values = measure.marching_cubes(sdf.astype(np.float32), level=0.0, spacing=(pitch, pitch, pitch), allow_degenerate=False)
    vertices = vertices + origin[None, :]
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=np.asarray(faces, dtype=np.int32), vertex_normals=np.asarray(normals, dtype=np.float64), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError("optimized shared SDF mesh has no connected components")
    areas = [float(component.area) for component in components]
    mesh = components[int(np.argmax(areas))]
    trimesh.repair.fix_normals(mesh)
    topo = topology(mesh)
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"optimized shared SDF mesh is not topologically closed: {topo}")
    return mesh, {"components": int(len(components)), "component_area_m2": summarize(areas), "topology": topo}


def save_world_archive(args: argparse.Namespace, mesh_prior: trimesh.Trimesh, annotations: dict[int, dict], pose_rows: list[dict]) -> Path:
    vertices = np.asarray(mesh_prior.vertices, dtype=np.float64)
    faces = np.asarray(mesh_prior.faces, dtype=np.int32)
    frame_ids = []
    meshes_world = []
    for pose in pose_rows:
        frame_idx = int(pose["frame_idx"])
        r = np.asarray(pose["rotation_prior_to_camera"], dtype=np.float64)
        t = np.asarray(pose["translation_prior_to_camera"], dtype=np.float64)
        camera_vertices = vertices @ r.T + t[None, :]
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        world_vertices = transform_points(camera_vertices, T_world_camera)
        meshes_world.append(trimesh.Trimesh(vertices=world_vertices.astype(np.float32), faces=faces, process=False))
        frame_ids.append(frame_idx)
    archive_path = args.output_dir / "shared_sdf_visible_contact_meshes_world.npz"
    save_archive(archive_path, frame_ids, meshes_world)
    return archive_path


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    graph_meshes = load_mesh_archive(args.graph_mesh_archive)
    prior_mesh = load_mesh(args.mesh_prior_camera)
    prior_vertices = np.asarray(prior_mesh.vertices, dtype=np.float64)
    pose_rows = build_pose_rows(args, annotations, graph_meshes, prior_vertices)
    sdf, transform, occ = sdf_from_mesh(prior_mesh, float(args.pitch_m), int(args.pad_voxels))
    visible_points, free_space_points, visible_report = visible_depth_points_prior(args, annotations, manifest, depths, pose_rows)
    contact_points, contact_report = contact_points_prior(args, annotations, pose_rows)
    hand_points, hand_report = hand_points_prior(args, annotations, pose_rows, contact_points)
    anchors = [visible_points]
    if len(free_space_points):
        anchors.append(free_space_points)
    if len(contact_points):
        anchors.append(contact_points)
    if len(hand_points):
        anchors.append(hand_points)
    var_mask, variable_report = variable_mask_for_points(sdf, transform, np.vstack(anchors), float(args.edit_radius_m))
    delta, solve_report = solve_sdf_delta(args, sdf, transform, var_mask, visible_points, free_space_points, contact_points, hand_points)
    edited_sdf = sdf.astype(np.float32) + delta.astype(np.float32)
    mesh_prior, extract_report = extract_mesh(edited_sdf, transform)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = args.output_dir / "shared_sdf_visible_contact_prior_frame.obj"
    mesh_prior.export(prior_path)
    archive_path = save_world_archive(args, mesh_prior, annotations, pose_rows)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_shared_sdf_visible_contact_v3",
        "claim_tested": "one shared object-frame implicit SDF can place visible depth, selected contact, and full-hand clearance on the delivered zero surface",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "graph_mesh_archive": str(args.graph_mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "contact_report": None if args.contact_report is None else str(args.contact_report),
        "mano_model": None if args.mano_model is None else str(args.mano_model),
        "prior_frame_mesh": str(prior_path),
        "mesh_archive_world": str(archive_path),
        "frames": [int(row["frame_idx"]) for row in pose_rows],
        "base_voxel_occupied": int(np.count_nonzero(occ)),
        "base_voxel_shape": [int(v) for v in sdf.shape],
        "visible_points": visible_report,
        "contact_points": contact_report,
        "hand_points": hand_report,
        "variable_mask": variable_report,
        "solve": solve_report,
        "extract": extract_report,
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
            "pitch_m": float(args.pitch_m),
            "pad_voxels": int(args.pad_voxels),
            "edit_radius_m": float(args.edit_radius_m),
            "visible_sdf_target_m": float(args.visible_sdf_target_m),
            "free_space_sdf_target_m": float(args.free_space_sdf_target_m),
            "contact_sdf_target_m": float(args.contact_sdf_target_m),
            "hand_clearance_m": float(args.hand_clearance_m),
            "preserve_scale_m": float(args.preserve_scale_m),
            "smooth_delta_scale_m": float(args.smooth_delta_scale_m),
            "max_delta_m": float(args.max_delta_m),
        },
    }
    save_json(args.output_dir / "qc_shared_sdf_visible_contact_v3.json", report)
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
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-pose-correspondences", type=int, default=12000)
    parser.add_argument("--pitch-m", type=float, default=0.004)
    parser.add_argument("--pad-voxels", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.050)
    parser.add_argument("--min-mask-distance-px", type=float, default=3.0)
    parser.add_argument("--max-visible-points-per-frame", type=int, default=2500)
    parser.add_argument("--free-space-samples-per-visible", type=int, default=2)
    parser.add_argument("--free-space-min-offset-m", type=float, default=0.006)
    parser.add_argument("--free-space-max-offset-m", type=float, default=0.030)
    parser.add_argument("--free-space-sdf-target-m", type=float, default=0.008)
    parser.add_argument("--free-space-active-margin-m", type=float, default=0.006)
    parser.add_argument("--free-space-residual-scale-m", type=float, default=0.003)
    parser.add_argument("--visible-sdf-target-m", type=float, default=0.0)
    parser.add_argument("--visible-residual-scale-m", type=float, default=0.006)
    parser.add_argument("--contact-sdf-target-m", type=float, default=0.0)
    parser.add_argument("--contact-residual-scale-m", type=float, default=0.002)
    parser.add_argument("--min-controllable-contact-points", type=int, default=12)
    parser.add_argument("--hand-clearance-m", type=float, default=0.004)
    parser.add_argument("--hand-active-margin-m", type=float, default=0.004)
    parser.add_argument("--hand-clearance-residual-scale-m", type=float, default=0.002)
    parser.add_argument("--hand-contact-exclusion-radius-m", type=float, default=0.018)
    parser.add_argument("--hand-surface-samples-per-mesh", type=int, default=900)
    parser.add_argument("--max-hand-points", type=int, default=12000)
    parser.add_argument("--edit-radius-m", type=float, default=0.020)
    parser.add_argument("--max-variables", type=int, default=500000)
    parser.add_argument("--preserve-scale-m", type=float, default=0.010)
    parser.add_argument("--smooth-delta-scale-m", type=float, default=0.004)
    parser.add_argument("--max-delta-m", type=float, default=0.030)
    parser.add_argument("--active-set-iterations", type=int, default=3)
    parser.add_argument("--lsmr-tol", type=float, default=1e-5)
    parser.add_argument("--lsmr-maxiter", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
