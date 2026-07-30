#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree
from skimage import measure

from close_mesh_archive_with_voxel_fill_v3 import save_archive, topology, transform_points
from deform_visible_mesh_contact_surface_v3 import frame_contact_points, frame_noncontact_hand_points
from diagnose_volume_sdf_contact_v3 import summarize, voxel_sdf
from render_bundlesdf_mesh_qc_v3 import camera_points, load_mesh_archive
from optimize_contact_patch_object_pose_graph_v3 import annotations_by_frame
from fit_mano_to_hand_mask_depth_v3 import load_mano_faces


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def grid_points_from_transform(shape: tuple[int, int, int], transform: np.ndarray) -> np.ndarray:
    pitch = float(transform[0, 0])
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    axes = [origin[i] + np.arange(shape[i], dtype=np.float64) * pitch for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.c_[gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)]


def flat_index(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    return (ix * int(shape[1]) + iy) * int(shape[2]) + iz


def local_variable_mask(
    sdf: np.ndarray,
    transform: np.ndarray,
    vertices: np.ndarray,
    contact_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if len(contact_points) == 0:
        raise RuntimeError("SDF contact optimization needs contact points")
    nearest_distance, nearest_idx = cKDTree(vertices).query(contact_points, k=1)
    anchors = []
    for anchor, contact in zip(vertices[nearest_idx], contact_points, strict=True):
        for alpha in np.linspace(0.0, 1.0, int(args.segment_samples), dtype=np.float64):
            anchors.append((1.0 - alpha) * anchor + alpha * contact)
    anchors = np.asarray(anchors, dtype=np.float64)
    grid = grid_points_from_transform(tuple(int(v) for v in sdf.shape), transform)
    distance = cKDTree(anchors).query(grid, k=1, workers=-1)[0]
    mask = distance <= float(args.edit_radius_m)
    if int(np.count_nonzero(mask)) == 0:
        raise RuntimeError("local SDF edit mask is empty")
    return mask.reshape(sdf.shape), {
        "contact_points": int(len(contact_points)),
        "segment_anchors": int(len(anchors)),
        "nearest_surface_distance_m": summarize(nearest_distance),
        "edit_radius_m": float(args.edit_radius_m),
        "variable_nodes": int(np.count_nonzero(mask)),
    }


def interp_rows(
    points: np.ndarray,
    sdf: np.ndarray,
    transform: np.ndarray,
    var_lookup: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
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
    cols: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    fixed_values = []
    kept = []
    corners = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
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
        fixed = 0.0
        row_cols = []
        row_weights = []
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
    matrix_rows: list[int],
    matrix_cols: list[int],
    matrix_data: list[float],
    rhs: list[float],
    row_base: int,
    cols: list[np.ndarray],
    weights: list[np.ndarray],
    fixed: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> int:
    row_idx = int(row_base)
    for c, w, base_value, target_value in zip(cols, weights, fixed, target, strict=True):
        for col, weight in zip(c, w, strict=True):
            matrix_rows.append(row_idx)
            matrix_cols.append(int(col))
            matrix_data.append(float(weight) / float(scale))
        rhs.append((float(target_value) - float(base_value)) / float(scale))
        row_idx += 1
    return row_idx


def neighbor_edges(var_lookup: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for axis in range(3):
        a = [slice(None), slice(None), slice(None)]
        b = [slice(None), slice(None), slice(None)]
        a[axis] = slice(0, -1)
        b[axis] = slice(1, None)
        left = var_lookup[tuple(a)]
        right = var_lookup[tuple(b)]
        keep = (left >= 0) & (right >= 0)
        if np.any(keep):
            pairs.append(np.c_[left[keep], right[keep]])
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    stacked = np.vstack(pairs).astype(np.int64)
    return stacked[:, 0], stacked[:, 1]


def solve_delta(
    sdf: np.ndarray,
    transform: np.ndarray,
    var_mask: np.ndarray,
    contact_points: np.ndarray,
    noncontact_hand_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    var_flat = np.flatnonzero(var_mask.reshape(-1))
    if len(var_flat) > int(args.max_variables):
        raise RuntimeError(f"local SDF solve has {len(var_flat)} variables, above max {args.max_variables}")
    var_lookup = np.full(sdf.shape, -1, dtype=np.int64)
    var_lookup.reshape(-1)[var_flat] = np.arange(len(var_flat), dtype=np.int64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_idx = 0

    c_cols, c_weights, c_fixed, c_kept = interp_rows(contact_points, sdf, transform, var_lookup)
    if len(c_cols) != len(contact_points):
        raise RuntimeError(f"only {len(c_cols)}/{len(contact_points)} contact points are controllable")
    contact_target = np.full(len(c_cols), float(args.contact_sdf_target_m), dtype=np.float64)
    row_idx = append_interp_equations(
        rows,
        cols,
        data,
        rhs,
        row_idx,
        c_cols,
        c_weights,
        c_fixed,
        contact_target,
        float(args.contact_residual_scale_m),
    )

    h_cols, h_weights, h_fixed, h_kept = interp_rows(noncontact_hand_points, sdf, transform, var_lookup)
    hand_points_kept = noncontact_hand_points[h_kept] if len(h_kept) else np.zeros((0, 3), dtype=np.float64)
    current_hand = h_fixed.copy()
    active_hand = np.ones(len(h_fixed), dtype=bool)
    hand_iterations = []
    solution = np.zeros(len(var_flat), dtype=np.float64)
    left_edges, right_edges = neighbor_edges(var_lookup)
    preserve_weight = 1.0 / float(args.preserve_scale_m)
    smooth_weight = 1.0 / float(args.smooth_delta_scale_m)
    for iteration in range(int(args.active_set_iterations)):
        iter_rows = list(rows)
        iter_cols = list(cols)
        iter_data = list(data)
        iter_rhs = list(rhs)
        iter_row_idx = row_idx
        if len(h_cols):
            active_hand = current_hand < float(args.hand_clearance_m) + float(args.hand_active_margin_m)
            active_cols = [h_cols[i] for i in np.flatnonzero(active_hand)]
            active_weights = [h_weights[i] for i in np.flatnonzero(active_hand)]
            active_fixed = h_fixed[active_hand]
            hand_target = np.full(len(active_cols), float(args.hand_clearance_m), dtype=np.float64)
            iter_row_idx = append_interp_equations(
                iter_rows,
                iter_cols,
                iter_data,
                iter_rhs,
                iter_row_idx,
                active_cols,
                active_weights,
                active_fixed,
                hand_target,
                float(args.hand_clearance_residual_scale_m),
            )
        for col in range(len(var_flat)):
            iter_rows.append(iter_row_idx)
            iter_cols.append(col)
            iter_data.append(preserve_weight)
            iter_rhs.append(0.0)
            iter_row_idx += 1
        for left, right in zip(left_edges, right_edges, strict=True):
            iter_rows.append(iter_row_idx)
            iter_cols.append(int(left))
            iter_data.append(smooth_weight)
            iter_rows.append(iter_row_idx)
            iter_cols.append(int(right))
            iter_data.append(-smooth_weight)
            iter_rhs.append(0.0)
            iter_row_idx += 1
        system = sparse.coo_matrix(
            (np.asarray(iter_data, dtype=np.float64), (np.asarray(iter_rows), np.asarray(iter_cols))),
            shape=(iter_row_idx, len(var_flat)),
        ).tocsr()
        solution = lsmr(system, np.asarray(iter_rhs, dtype=np.float64), atol=float(args.lsmr_tol), btol=float(args.lsmr_tol), maxiter=int(args.lsmr_maxiter))[0]
        if float(args.max_delta_m) > 0.0:
            solution = np.clip(solution, -float(args.max_delta_m), float(args.max_delta_m))
        if len(h_cols):
            current_hand = h_fixed + np.asarray(
                [float(np.dot(w, solution[c])) for c, w in zip(h_cols, h_weights, strict=True)],
                dtype=np.float64,
            )
        hand_iterations.append(
            {
                "iteration": int(iteration),
                "rows": int(iter_row_idx),
                "active_hand_points": int(np.count_nonzero(active_hand)) if len(h_cols) else 0,
                "solution_delta_m": summarize(solution),
                "hand_sdf_m": summarize(current_hand),
                "hand_clearance_violation_fraction": float(np.mean(current_hand < float(args.hand_clearance_m))) if len(current_hand) else 0.0,
            }
        )
    contact_values = c_fixed + np.asarray(
        [float(np.dot(w, solution[c])) for c, w in zip(c_cols, c_weights, strict=True)],
        dtype=np.float64,
    )
    delta = np.zeros(sdf.size, dtype=np.float32)
    delta[var_flat] = solution.astype(np.float32)
    return delta.reshape(sdf.shape), {
        "variables": int(len(var_flat)),
        "contact_points": int(len(contact_points)),
        "contact_sdf_before_m": summarize(c_fixed),
        "contact_sdf_after_m": summarize(contact_values),
        "noncontact_hand_points": int(len(noncontact_hand_points)),
        "controllable_hand_points": int(len(h_cols)),
        "hand_points_kept": int(len(hand_points_kept)),
        "noncontact_hand_sdf_after_m": summarize(current_hand),
        "noncontact_hand_clearance_violation_fraction": float(np.mean(current_hand < float(args.hand_clearance_m))) if len(current_hand) else 0.0,
        "delta_m": summarize(solution),
        "smooth_edges": int(len(left_edges)),
        "active_set_iterations": hand_iterations,
    }


def extract_mesh(sdf: np.ndarray, transform: np.ndarray, args: argparse.Namespace) -> tuple[trimesh.Trimesh, dict]:
    pitch = float(transform[0, 0])
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    vertices, faces, normals, _values = measure.marching_cubes(
        sdf.astype(np.float32),
        level=0.0,
        spacing=(pitch, pitch, pitch),
        allow_degenerate=False,
    )
    vertices = vertices + origin[None, :]
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
        raise RuntimeError("optimized SDF mesh has no connected components")
    component_areas = [float(component.area) for component in components]
    mesh = components[int(np.argmax(component_areas))]
    trimesh.repair.fix_normals(mesh)
    topo = topology(mesh)
    if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
        raise RuntimeError(f"optimized SDF mesh is not topologically closed: {topo}")
    return mesh, {
        "components": int(len(components)),
        "component_area_m2": summarize(component_areas),
        "selected_component_area_m2": float(max(component_areas)),
        "topology": topo,
    }


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    meshes = load_mesh_archive(args.visible_mesh_archive)
    mano_faces = load_mano_faces(args.mano_model) if args.mano_model is not None else None
    frames = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in meshes and idx in annotations]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_rows = []
    meshes_world = []
    for frame_idx in frames:
        vertices_world, faces = meshes[frame_idx]
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_camera = camera_points(vertices_world, T_world_camera)
        base_mesh = trimesh.Trimesh(vertices=vertices_camera.astype(np.float32), faces=np.asarray(faces, dtype=np.int32), process=True)
        trimesh.repair.fix_normals(base_mesh)
        base_topology = topology(base_mesh)
        if not base_topology["watertight"] or base_topology["boundary_edges"] != 0 or base_topology["nonmanifold_edges"] != 0:
            raise RuntimeError(f"source mesh frame {frame_idx} is not closed: {base_topology}")
        sdf, transform, occ = voxel_sdf(base_mesh, float(args.pitch_m), int(args.pad_voxels))
        contact_points, contact_rows = frame_contact_points(annotations, args.contact_report, frame_idx, args)
        noncontact_hand_points, hand_row = frame_noncontact_hand_points(
            annotations,
            args.contact_report,
            frame_idx,
            contact_points,
            mano_faces,
            args,
        )
        var_mask, mask_row = local_variable_mask(
            sdf,
            transform,
            np.asarray(base_mesh.vertices, dtype=np.float64),
            contact_points,
            args,
        )
        delta, solve_row = solve_delta(
            sdf.astype(np.float32),
            transform,
            var_mask,
            contact_points,
            noncontact_hand_points,
            args,
        )
        edited_sdf = sdf.astype(np.float32) + delta.astype(np.float32)
        mesh_camera, extract_row = extract_mesh(edited_sdf, transform, args)
        mesh_world = trimesh.Trimesh(
            vertices=transform_points(np.asarray(mesh_camera.vertices, dtype=np.float64), T_world_camera).astype(np.float32),
            faces=np.asarray(mesh_camera.faces, dtype=np.int32),
            process=False,
        )
        meshes_world.append(mesh_world)
        if int(frame_idx) in set(int(v) for v in args.export_frames):
            mesh_camera.export(args.output_dir / f"sdf_contact_surface_frame_{frame_idx:06d}_camera.obj")
            mesh_world.export(args.output_dir / f"sdf_contact_surface_frame_{frame_idx:06d}_world.obj")
        mesh_rows.append(
            {
                "frame_idx": int(frame_idx),
                "base_topology": base_topology,
                "contact_rows": contact_rows,
                "hand_surface": hand_row,
                "variable_mask": mask_row,
                "solve": solve_row,
                "extract": extract_row,
                "voxel_occupied": int(np.count_nonzero(occ)),
                "voxel_shape": [int(v) for v in sdf.shape],
            }
        )
    archive = args.output_dir / "optimized_sdf_contact_meshes_world.npz"
    save_archive(archive, frames, meshes_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_object_sdf_contact_surface_v3",
        "claim_tested": "a local SDF least-squares edit can place active contact on the zero surface while preserving non-contact hand clearance",
        "visible_mesh_archive": str(args.visible_mesh_archive),
        "annotations": str(args.annotations),
        "contact_report": str(args.contact_report),
        "mesh_archive_world": str(archive),
        "frames": [int(v) for v in frames],
        "rows": mesh_rows,
        "parameters": {
            "pitch_m": float(args.pitch_m),
            "pad_voxels": int(args.pad_voxels),
            "edit_radius_m": float(args.edit_radius_m),
            "contact_sdf_target_m": float(args.contact_sdf_target_m),
            "hand_clearance_m": float(args.hand_clearance_m),
            "preserve_scale_m": float(args.preserve_scale_m),
            "smooth_delta_scale_m": float(args.smooth_delta_scale_m),
        },
    }
    save_json(args.output_dir / "qc_optimized_sdf_contact_surface_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--pitch-m", type=float, default=0.003)
    parser.add_argument("--pad-voxels", type=int, default=8)
    parser.add_argument("--edit-radius-m", type=float, default=0.018)
    parser.add_argument("--segment-samples", type=int, default=7)
    parser.add_argument("--max-variables", type=int, default=140000)
    parser.add_argument("--contact-sdf-target-m", type=float, default=0.0015)
    parser.add_argument("--max-contact-source-p95-m", type=float, default=-1.0)
    parser.add_argument("--max-contact-source-signed-p95-m", type=float, default=-1.0)
    parser.add_argument("--contact-residual-scale-m", type=float, default=0.0008)
    parser.add_argument("--hand-surface-samples-per-hand", type=int, default=1500)
    parser.add_argument("--active-contact-exclusion-radius-m", type=float, default=0.012)
    parser.add_argument("--hand-clearance-m", type=float, default=0.004)
    parser.add_argument("--hand-active-margin-m", type=float, default=0.003)
    parser.add_argument("--hand-clearance-residual-scale-m", type=float, default=0.0012)
    parser.add_argument("--preserve-scale-m", type=float, default=0.030)
    parser.add_argument("--smooth-delta-scale-m", type=float, default=0.008)
    parser.add_argument("--max-delta-m", type=float, default=0.030)
    parser.add_argument("--active-set-iterations", type=int, default=4)
    parser.add_argument("--lsmr-tol", type=float, default=1e-6)
    parser.add_argument("--lsmr-maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--export-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
