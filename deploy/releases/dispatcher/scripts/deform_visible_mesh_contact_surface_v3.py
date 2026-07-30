#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
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


def selected_patch_metric(row: dict, suffix: str) -> float:
    source = str(row.get("selected_patch_source"))
    if source == "anatomical_patch":
        key = f"anatomical_patch_{suffix}"
    elif source == "best_patch":
        key = f"best_patch_{suffix}"
    else:
        raise RuntimeError(f"unsupported selected patch source {source!r}")
    value = row.get(key)
    if value is None:
        raise RuntimeError(f"row {row.get('frame_idx')} missing selected patch metric {key}")
    return float(value)


def contact_row_enabled(row: dict, args: argparse.Namespace) -> bool:
    max_distance = float(args.max_contact_source_p95_m)
    max_signed = float(args.max_contact_source_signed_p95_m)
    if max_distance >= 0.0 and selected_patch_metric(row, "distance_p95_m") > max_distance:
        return False
    if max_signed >= 0.0 and selected_patch_metric(row, "signed_gap_p95_abs_m") > max_signed:
        return False
    return True


def frame_contact_points(annotations: dict[int, dict], contact_report: Path, frame_idx: int, args: argparse.Namespace) -> tuple[np.ndarray, list[dict]]:
    points = []
    rows = []
    for row in contact_rows(contact_report):
        if int(row["frame_idx"]) != int(frame_idx):
            continue
        enabled = contact_row_enabled(row, args)
        hand_idx = int(row["hand_idx"])
        hand = annotations[frame_idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        ids = selected_vertex_ids(row)
        if int(ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {frame_idx} contact vertex id exceeds MANO vertex count")
        patch = vertices[ids]
        if enabled:
            points.append(patch)
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "hand_idx": int(hand_idx),
                "track_id": row.get("track_id"),
                "selected_patch_source": row.get("selected_patch_source"),
                "selected_patch_region": row.get("selected_patch_region"),
                "selected_source_distance_p95_m": selected_patch_metric(row, "distance_p95_m"),
                "selected_source_signed_gap_p95_abs_m": selected_patch_metric(row, "signed_gap_p95_abs_m"),
                "contact_constraint_enabled": bool(enabled),
                "patch_points": int(len(patch)),
            }
        )
    if not points:
        return np.zeros((0, 3), dtype=np.float64), rows
    return np.vstack(points).astype(np.float64), rows


def frame_contact_hand_indices(contact_report: Path, frame_idx: int) -> list[int]:
    indices = sorted({int(row["hand_idx"]) for row in contact_rows(contact_report) if int(row["frame_idx"]) == int(frame_idx)})
    return indices


def sampled_hand_surface_camera(
    hand: dict,
    mano_faces: np.ndarray | None,
    samples: int,
    seed: int,
) -> np.ndarray:
    vertices = hand_vertices_camera(hand).astype(np.float64)
    if int(samples) <= 0:
        return vertices
    if mano_faces is None:
        raise RuntimeError("hand surface sampling requires --mano-model")
    mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=mano_faces.astype(np.int32), process=True)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("empty MANO mesh reached hand surface sampling")
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        sampled, _face_ids = sample_surface(mesh, int(samples))
    finally:
        np.random.set_state(state)
    return np.vstack([vertices, np.asarray(sampled, dtype=np.float64)]).astype(np.float64)


def frame_noncontact_hand_points(
    annotations: dict[int, dict],
    contact_report: Path,
    frame_idx: int,
    contact_points: np.ndarray,
    mano_faces: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    hand_points = []
    hand_indices = frame_contact_hand_indices(contact_report, frame_idx)
    for hand_idx in hand_indices:
        hand = annotations[frame_idx]["hands"][hand_idx]
        hand_points.append(
            sampled_hand_surface_camera(
                hand,
                mano_faces,
                int(args.hand_surface_samples_per_hand),
                int(args.seed) + int(frame_idx) * 1009 + int(hand_idx),
            )
        )
    if not hand_points:
        return np.zeros((0, 3), dtype=np.float64), {"contact_hands": 0, "hand_surface_points": 0}
    points = np.vstack(hand_points).astype(np.float64)
    if len(contact_points):
        distance = cKDTree(contact_points).query(points, k=1)[0]
        keep = distance > float(args.active_contact_exclusion_radius_m)
        points = points[keep]
    return points, {
        "contact_hands": int(len(hand_indices)),
        "noncontact_hand_points": int(len(points)),
        "hand_surface_samples_per_hand": int(args.hand_surface_samples_per_hand),
        "active_contact_exclusion_radius_m": float(args.active_contact_exclusion_radius_m),
    }


def deform_vertices(vertices: np.ndarray, contact_points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if len(contact_points) == 0:
        return vertices.copy(), {"contact_points": 0}
    tree = cKDTree(vertices)
    nearest_distance, nearest_idx = tree.query(contact_points, k=1)
    anchors = vertices[nearest_idx]
    direction = contact_points - anchors
    norm = np.linalg.norm(direction, axis=1)
    valid = norm > 1e-8
    if not np.any(valid):
        return vertices.copy(), {
            "contact_points": int(len(contact_points)),
            "active_contact_points": 0,
            "initial_contact_vertex_distance_m": summarize(nearest_distance),
        }
    contact_points = contact_points[valid]
    anchors = anchors[valid]
    norm = norm[valid]
    direction = direction[valid] / norm[:, None]
    target = contact_points - float(args.target_contact_clearance_m) * direction
    delta = target - anchors
    delta_norm = np.linalg.norm(delta, axis=1)
    max_delta = float(args.max_contact_displacement_m)
    if max_delta > 0.0:
        scale = np.minimum(1.0, max_delta / np.maximum(delta_norm, 1e-8))
        delta = delta * scale[:, None]
        delta_norm = delta_norm * scale
    out = vertices.copy()
    total_weight = np.zeros(len(vertices), dtype=np.float64)
    weighted_delta = np.zeros_like(vertices, dtype=np.float64)
    sigma2 = float(args.contact_sigma_m) ** 2
    radius = float(args.contact_radius_m)
    for anchor, shift in zip(anchors, delta, strict=True):
        nearby = tree.query_ball_point(anchor, r=radius)
        if not nearby:
            continue
        ids = np.asarray(nearby, dtype=np.int64)
        dist2 = np.sum((vertices[ids] - anchor[None, :]) ** 2, axis=1)
        weight = np.exp(-0.5 * dist2 / sigma2)
        weighted_delta[ids] += weight[:, None] * shift[None, :]
        total_weight[ids] += weight
    moved = total_weight > 1e-8
    out[moved] += weighted_delta[moved] / total_weight[moved, None]
    return out, {
        "contact_points": int(len(contact_points)),
        "active_contact_points": int(np.count_nonzero(valid)),
        "moved_vertices": int(np.count_nonzero(moved)),
        "initial_contact_vertex_distance_m": summarize(norm),
        "applied_contact_delta_m": summarize(delta_norm),
        "target_contact_clearance_m": float(args.target_contact_clearance_m),
        "contact_sigma_m": float(args.contact_sigma_m),
        "contact_radius_m": float(args.contact_radius_m),
        "max_contact_displacement_m": float(args.max_contact_displacement_m),
    }


def candidate_vertex_weights(
    vertices: np.ndarray,
    anchors: np.ndarray,
    sigma_m: float,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(anchors) == 0:
        raise RuntimeError("candidate weights requested without anchors")
    tree = cKDTree(vertices)
    ids = sorted({int(idx) for anchor in anchors for idx in tree.query_ball_point(anchor, r=float(radius_m))})
    if not ids:
        raise RuntimeError("contact radius selects zero mesh vertices")
    candidate_ids = np.asarray(ids, dtype=np.int64)
    diff = vertices[candidate_ids, None, :] - anchors[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    weights = np.exp(-0.5 * dist2 / (float(sigma_m) ** 2))
    weights[dist2 > float(radius_m) ** 2] = 0.0
    row_sum = weights.sum(axis=1)
    active = row_sum > 1e-10
    if not np.any(active):
        raise RuntimeError("contact weights are all zero")
    candidate_ids = candidate_ids[active]
    weights = weights[active] / row_sum[active, None]
    return candidate_ids, weights.astype(np.float64)


def optimize_contact_clearance_vertices(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    contact_points: np.ndarray,
    noncontact_hand_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if len(contact_points) == 0:
        return vertices.copy(), {"contact_points": 0}
    nearest_distance, nearest_idx = cKDTree(vertices).query(contact_points, k=1)
    anchors = vertices[nearest_idx].astype(np.float64)
    direction = contact_points - anchors
    norm = np.linalg.norm(direction, axis=1)
    valid = norm > 1e-8
    if not np.any(valid):
        return vertices.copy(), {
            "contact_points": int(len(contact_points)),
            "active_contact_points": 0,
            "initial_contact_vertex_distance_m": summarize(nearest_distance),
        }
    contact_points = contact_points[valid].astype(np.float64)
    anchors = anchors[valid]
    nearest_idx = nearest_idx[valid]
    direction = direction[valid] / norm[valid, None]
    target = contact_points - float(args.target_contact_clearance_m) * direction
    candidate_ids, weights = candidate_vertex_weights(vertices, anchors, float(args.contact_sigma_m), float(args.contact_radius_m))
    candidate_index = {int(vertex_id): i for i, vertex_id in enumerate(candidate_ids.tolist())}
    contact_candidate_rows = []
    for vertex_id, point, tgt in zip(nearest_idx, contact_points, target, strict=True):
        if int(vertex_id) not in candidate_index:
            raise RuntimeError("nearest contact vertex is absent from contact candidate set")
        contact_candidate_rows.append((candidate_index[int(vertex_id)], point, tgt))
    if vertex_normals.shape != vertices.shape:
        raise RuntimeError("vertex normal array shape does not match vertices")
    candidate_normals = vertex_normals[candidate_ids].astype(np.float64)
    normal_norm = np.linalg.norm(candidate_normals, axis=1)
    if np.any(normal_norm < 1e-8):
        raise RuntimeError("candidate mesh contains zero normals")
    candidate_normals = candidate_normals / normal_norm[:, None]
    hand_pairs = []
    if len(noncontact_hand_points):
        candidate_tree = cKDTree(vertices[candidate_ids])
        distance, local_idx = candidate_tree.query(noncontact_hand_points, k=1)
        near = distance <= float(args.hand_constraint_radius_m)
        for point, idx in zip(noncontact_hand_points[near], local_idx[near], strict=True):
            hand_pairs.append((int(idx), point.astype(np.float64)))
    if not hand_pairs and float(args.hand_clearance_m) >= 0.0:
        raise RuntimeError("least-squares clearance requested but no non-contact hand points are near the moved surface")
    x0 = np.zeros((len(anchors), 3), dtype=np.float64)
    initial_shift = target - anchors
    shift_norm = np.linalg.norm(initial_shift, axis=1)
    max_shift = float(args.max_contact_displacement_m)
    if max_shift > 0.0:
        scale = np.minimum(1.0, max_shift / np.maximum(shift_norm, 1e-8))
        initial_shift *= scale[:, None]
    x0[:] = initial_shift
    lower = np.full(x0.size, -float(args.max_contact_displacement_m), dtype=np.float64)
    upper = np.full(x0.size, float(args.max_contact_displacement_m), dtype=np.float64)

    base_candidates = vertices[candidate_ids].astype(np.float64)

    def moved_candidates(shifts: np.ndarray) -> np.ndarray:
        return base_candidates + weights @ shifts.reshape(len(anchors), 3)

    def residual(flat: np.ndarray) -> np.ndarray:
        shifts = flat.reshape(len(anchors), 3)
        moved = moved_candidates(shifts)
        parts = []
        contact_scale = float(args.contact_residual_scale_m)
        for local_idx, _point, tgt in contact_candidate_rows:
            parts.append((moved[local_idx] - tgt) / contact_scale)
        if hand_pairs:
            clearance = float(args.hand_clearance_m)
            clear_scale = float(args.hand_clearance_residual_scale_m)
            local_ids = np.asarray([idx for idx, _point in hand_pairs], dtype=np.int64)
            points = np.vstack([point for _idx, point in hand_pairs]).astype(np.float64)
            if str(args.clearance_mode) == "unsigned":
                gap = np.linalg.norm(points - moved[local_ids], axis=1)
            elif str(args.clearance_mode) == "signed_normal":
                gap = np.einsum("ij,ij->i", points - moved[local_ids], candidate_normals[local_ids])
            else:
                raise RuntimeError(f"unknown clearance mode: {args.clearance_mode}")
            hinge = np.maximum(0.0, clearance - gap) / clear_scale
            parts.append(hinge[:, None])
        preserve_scale = float(args.surface_preserve_scale_m)
        if preserve_scale > 0.0:
            parts.append((moved - base_candidates) / preserve_scale)
        shift_scale = float(args.anchor_shift_scale_m)
        if shift_scale > 0.0:
            parts.append(shifts / shift_scale)
        return np.concatenate([part.reshape(-1) for part in parts])

    result = least_squares(
        residual,
        x0.reshape(-1),
        bounds=(lower, upper),
        loss=str(args.loss),
        f_scale=float(args.loss_f_scale),
        max_nfev=int(args.max_nfev),
        verbose=0,
    )
    shifts = result.x.reshape(len(anchors), 3)
    out = vertices.copy()
    out[candidate_ids] = moved_candidates(shifts)
    hand_gap = np.zeros(0, dtype=np.float64)
    hand_hinge = np.zeros(0, dtype=np.float64)
    if hand_pairs:
        moved = out[candidate_ids]
        local_ids = np.asarray([idx for idx, _point in hand_pairs], dtype=np.int64)
        points = np.vstack([point for _idx, point in hand_pairs]).astype(np.float64)
        if str(args.clearance_mode) == "unsigned":
            hand_gap = np.linalg.norm(points - moved[local_ids], axis=1)
        elif str(args.clearance_mode) == "signed_normal":
            hand_gap = np.einsum("ij,ij->i", points - moved[local_ids], candidate_normals[local_ids])
        else:
            raise RuntimeError(f"unknown clearance mode: {args.clearance_mode}")
        hand_hinge = np.maximum(0.0, float(args.hand_clearance_m) - hand_gap)
    final_contact = np.linalg.norm(out[nearest_idx] - target, axis=1)
    return out, {
        "contact_points": int(len(contact_points)),
        "active_contact_points": int(len(contact_points)),
        "moved_vertices": int(len(candidate_ids)),
        "noncontact_hand_pairs": int(len(hand_pairs)),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_cost": float(result.cost),
        "solver_nfev": int(result.nfev),
        "initial_contact_vertex_distance_m": summarize(norm[valid]),
        "initial_contact_shift_m": summarize(np.linalg.norm(initial_shift, axis=1)),
        "final_contact_target_error_m": summarize(final_contact),
        "final_anchor_shift_m": summarize(np.linalg.norm(shifts, axis=1)),
        "noncontact_hand_gap_m": summarize(hand_gap),
        "noncontact_clearance_violation_m": summarize(hand_hinge),
        "clearance_mode": str(args.clearance_mode),
        "target_contact_clearance_m": float(args.target_contact_clearance_m),
        "hand_clearance_m": float(args.hand_clearance_m),
        "contact_sigma_m": float(args.contact_sigma_m),
        "contact_radius_m": float(args.contact_radius_m),
        "max_contact_displacement_m": float(args.max_contact_displacement_m),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = annotations_by_frame(args.annotations)
    meshes = load_mesh_archive(args.visible_mesh_archive)
    frames = [idx for idx in range(int(args.frame_start), int(args.frame_end) + 1) if idx in meshes and idx in annotations]
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} mesh frames available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mano_faces = load_mano_faces(args.mano_model) if args.mano_model is not None else None
    meshes_world = []
    rows = []
    for frame_idx in frames:
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_world, faces = meshes[frame_idx]
        vertices_camera = camera_points(vertices_world, T_world_camera)
        contact_points, contact_rows_used = frame_contact_points(annotations, args.contact_report, frame_idx, args)
        if len(contact_points) == 0:
            deformed_camera = vertices_camera.astype(np.float64)
            deform_row = {
                "contact_points": 0,
                "active_contact_points": 0,
                "moved_vertices": 0,
                "contact_constraints_enabled": False,
            }
            hand_row = {"solver": str(args.solver), "contact_gated_off": True}
        elif str(args.solver) == "direct":
            deformed_camera, deform_row = deform_vertices(vertices_camera.astype(np.float64), contact_points, args)
            hand_row = {"solver": "direct"}
        elif str(args.solver) == "least_squares":
            source_mesh = trimesh.Trimesh(
                vertices=vertices_camera.astype(np.float32),
                faces=np.asarray(faces, dtype=np.int32),
                process=False,
            )
            vertex_normals = np.asarray(source_mesh.vertex_normals, dtype=np.float64)
            noncontact_hand_points, hand_row = frame_noncontact_hand_points(
                annotations,
                args.contact_report,
                frame_idx,
                contact_points,
                mano_faces,
                args,
            )
            deformed_camera, deform_row = optimize_contact_clearance_vertices(
                vertices_camera.astype(np.float64),
                vertex_normals,
                contact_points,
                noncontact_hand_points,
                args,
            )
            hand_row["solver"] = "least_squares"
        else:
            raise RuntimeError(f"unknown solver: {args.solver}")
        mesh_camera = trimesh.Trimesh(vertices=deformed_camera.astype(np.float32), faces=np.asarray(faces, dtype=np.int32), process=True)
        mesh_camera.update_faces(mesh_camera.nondegenerate_faces())
        mesh_camera.update_faces(mesh_camera.unique_faces())
        mesh_camera.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh_camera)
        topo = topology(mesh_camera)
        if not topo["watertight"] or topo["boundary_edges"] != 0 or topo["nonmanifold_edges"] != 0:
            raise RuntimeError(f"frame {frame_idx} deformed mesh is not topologically closed: {topo}")
        mesh_world = trimesh.Trimesh(
            vertices=transform_points(np.asarray(mesh_camera.vertices, dtype=np.float64), T_world_camera).astype(np.float32),
            faces=np.asarray(mesh_camera.faces, dtype=np.int32),
            process=False,
        )
        meshes_world.append(mesh_world)
        if int(frame_idx) in set(int(v) for v in args.export_frames):
            mesh_camera.export(args.output_dir / f"deformed_contact_surface_frame_{frame_idx:06d}_camera.obj")
            mesh_world.export(args.output_dir / f"deformed_contact_surface_frame_{frame_idx:06d}_world.obj")
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "topology": topo,
                "contact_rows": contact_rows_used,
                **hand_row,
                **deform_row,
            }
        )
    archive_path = args.output_dir / "deformed_contact_surface_meshes_world.npz"
    save_archive(archive_path, frames, meshes_world)
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "deform_visible_mesh_contact_surface_v3",
        "claim_tested": "directly optimizing the delivered visible mesh surface near active MANO contact can satisfy contact without object-volume union",
        "visible_mesh_archive": str(args.visible_mesh_archive),
        "annotations": str(args.annotations),
        "contact_report": str(args.contact_report),
        "mesh_archive_world": str(archive_path),
        "frames": [int(v) for v in frames],
        "rows": rows,
        "parameters": {
            "target_contact_clearance_m": float(args.target_contact_clearance_m),
            "contact_sigma_m": float(args.contact_sigma_m),
            "contact_radius_m": float(args.contact_radius_m),
            "max_contact_displacement_m": float(args.max_contact_displacement_m),
            "solver": str(args.solver),
            "max_contact_source_p95_m": float(args.max_contact_source_p95_m),
            "max_contact_source_signed_p95_m": float(args.max_contact_source_signed_p95_m),
        },
    }
    save_json(args.output_dir / "qc_deformed_contact_surface_v3.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-mesh-archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--target-contact-clearance-m", type=float, default=0.001)
    parser.add_argument("--contact-sigma-m", type=float, default=0.010)
    parser.add_argument("--contact-radius-m", type=float, default=0.030)
    parser.add_argument("--max-contact-displacement-m", type=float, default=0.030)
    parser.add_argument("--max-contact-source-p95-m", type=float, default=-1.0)
    parser.add_argument("--max-contact-source-signed-p95-m", type=float, default=-1.0)
    parser.add_argument("--solver", choices=["direct", "least_squares"], default="direct")
    parser.add_argument("--hand-surface-samples-per-hand", type=int, default=0)
    parser.add_argument("--active-contact-exclusion-radius-m", type=float, default=0.010)
    parser.add_argument("--hand-clearance-m", type=float, default=0.004)
    parser.add_argument("--hand-constraint-radius-m", type=float, default=0.040)
    parser.add_argument("--clearance-mode", choices=["signed_normal", "unsigned"], default="signed_normal")
    parser.add_argument("--contact-residual-scale-m", type=float, default=0.0015)
    parser.add_argument("--hand-clearance-residual-scale-m", type=float, default=0.0015)
    parser.add_argument("--surface-preserve-scale-m", type=float, default=0.030)
    parser.add_argument("--anchor-shift-scale-m", type=float, default=0.030)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="soft_l1")
    parser.add_argument("--loss-f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument("--export-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
