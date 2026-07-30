#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[import]
import trimesh  # type: ignore[import]
from scipy.spatial import cKDTree  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_mano_object_constraint_state import (  # noqa: E402
    frame_camera_pose,
    frame_intrinsics,
    least_norm_halfspace_escape,
    project,
    visible_2d_consistency,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [m for m in geom.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"{path}: scene contains no triangle meshes")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"{path}: did not load as triangle mesh")
    return geom


def inverse_object(points_world: np.ndarray, r_world_from_object: np.ndarray, t_world: np.ndarray) -> np.ndarray:
    return (points_world - t_world[None, :]) @ r_world_from_object


def forward_object_vec(vec_object: np.ndarray, r_world_from_object: np.ndarray) -> np.ndarray:
    return np.asarray(vec_object, dtype=float) @ r_world_from_object.T


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "min": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def existing_hprime_delta(hand: dict[str, Any], metric: dict[str, Any]) -> tuple[np.ndarray, str]:
    update_raw = hand.get("compact_rigid_object_mano_constraint_update")
    update = update_raw if isinstance(update_raw, dict) else metric.get("compact_rigid_object_constraint_update")
    if isinstance(update, dict) and update.get("coordinate_update_applied") is True:
        for field in ("cumulative_translation_world_m", "applied_translation_world_m", "candidate_translation_world_m"):
            arr = np.asarray(update.get(field) or [], dtype=float)
            if arr.shape == (3,) and np.isfinite(arr).all():
                return arr, field
    return np.zeros(3, dtype=float), "no_existing_hprime_translation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--sign-mesh", type=Path, required=True)
    parser.add_argument("--sign-mesh-source-report", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    completion = load_json(args.completion_report)
    pose_by_idx = {int(row["frame_idx"]): row for row in as_list(pose_report.get("pose_rows")) if isinstance(row, dict) and row.get("status") == "fit_to_visible_depth_samples"}

    surface_mesh = load_mesh(Path(completion["outputs"]["completed_mesh_labeled"]))
    surface_bounds_min = np.asarray(surface_mesh.bounds[0], dtype=float)
    surface_bounds_max = np.asarray(surface_mesh.bounds[1], dtype=float)
    observed_band_m = float(completion.get("observed_band_m") or 0.0)
    surface_samples, _ = trimesh.sample.sample_surface(surface_mesh, min(12000, max(1, len(surface_mesh.faces))), seed=np.random.default_rng(1818))
    surface_tree = cKDTree(np.asarray(surface_samples, dtype=float))

    sign_mesh = load_mesh(args.sign_mesh)
    sign_bounds_min = np.asarray(sign_mesh.bounds[0], dtype=float)
    sign_bounds_max = np.asarray(sign_mesh.bounds[1], dtype=float)
    sign_scene = o3d.t.geometry.RaycastingScene()
    sign_scene.add_triangles(
        o3d.core.Tensor(np.asarray(sign_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(sign_mesh.faces, dtype=np.uint32)),
    )

    hawor_npz = np.load(args.hawor_npz, allow_pickle=True)
    arrays = {key: np.asarray(hawor_npz[key]) for key in hawor_npz.files}
    frame_pos_by_idx = {int(frame_idx): i for i, frame_idx in enumerate(np.asarray(arrays["frame_idx"], dtype=int))}
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    bridge_cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []

    for frame_raw in as_list(annotations.get("frames")):
        if not isinstance(frame_raw, dict):
            continue
        frame = frame_raw
        frame_idx = int(frame["frame_idx"])
        pose = pose_by_idx.get(frame_idx)
        if pose is None or frame_idx not in frame_pos_by_idx:
            continue
        frame_pos = frame_pos_by_idx[frame_idx]
        r_obj = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=float)
        t_obj = np.asarray(pose["translation_world_m"], dtype=float)
        if "R_c2w" in arrays and "t_c2w" in arrays:
            fallback_r_c2w = np.asarray(arrays["R_c2w"][frame_pos], dtype=float)
            fallback_t_c2w = np.asarray(arrays["t_c2w"][frame_pos], dtype=float)
        elif "T_world_camera_metric_current_v18" in arrays:
            fallback_t_world_camera = np.asarray(arrays["T_world_camera_metric_current_v18"][frame_pos], dtype=float)
            if fallback_t_world_camera.shape != (4, 4):
                raise RuntimeError(f"T_world_camera_metric_current_v18 row has shape {fallback_t_world_camera.shape}, expected (4, 4)")
            fallback_r_c2w = fallback_t_world_camera[:3, :3]
            fallback_t_c2w = fallback_t_world_camera[:3, 3]
        else:
            raise RuntimeError("bridge NPZ lacks camera pose arrays: expected R_c2w/t_c2w or T_world_camera_metric_current_v18")
        r_c2w, t_c2w, camera_pose_source = frame_camera_pose(
            frame,
            fallback_r_c2w,
            fallback_t_c2w,
        )
        for hand_raw in as_list(frame.get("hands")):
            if not isinstance(hand_raw, dict):
                continue
            hand = hand_raw
            side = str(hand.get("hand_side"))
            metric = as_dict(hand.get("metric_mano_state"))
            reference = as_dict(metric.get("vertices_reference"))
            bridge_path = Path(str(reference.get("bridge_npz") or ""))
            vertices_array = reference.get("bridge_vertices_world_array")
            bridge_row_index = reference.get("bridge_row_index")
            if not bridge_path.exists() or not isinstance(vertices_array, str) or bridge_row_index is None:
                continue
            if bridge_path not in bridge_cache:
                bridge_cache[bridge_path] = np.load(bridge_path, allow_pickle=True)
            bridge = bridge_cache[bridge_path]
            existing_delta, existing_delta_source = existing_hprime_delta(hand, metric)
            verts_w = np.asarray(bridge[vertices_array][int(bridge_row_index)], dtype=float) + existing_delta[None, :]
            joints_w = np.asarray(bridge["joints_current_v18_world_from_hawor_projection_relift_m"][int(bridge_row_index)], dtype=float) + existing_delta[None, :]

            verts_obj = inverse_object(verts_w, r_obj, t_obj)
            unsigned_surface_dist, _ = surface_tree.query(verts_obj, k=1, workers=-1)
            near_surface = unsigned_surface_dist <= observed_band_m if observed_band_m > 0 else np.zeros(len(verts_obj), dtype=bool)
            surface_aabb = ((verts_obj >= surface_bounds_min) & (verts_obj <= surface_bounds_max)).all(axis=1)
            sign_aabb = ((verts_obj >= sign_bounds_min) & (verts_obj <= sign_bounds_max)).all(axis=1)
            signed_query = np.ones(len(verts_obj), dtype=bool)
            signed = np.full(len(verts_obj), np.nan, dtype=float)
            penetrating = np.zeros(len(verts_obj), dtype=bool)
            correction_obj = np.zeros(3, dtype=float)
            solver: dict[str, Any] = {"solver": "not_needed_no_penetration", "success": True, "constraint_count": 0}
            if signed_query.any():
                query_idx = np.where(signed_query)[0]
                signed[query_idx] = -sign_scene.compute_signed_distance(o3d.core.Tensor(np.asarray(verts_obj[query_idx], dtype=np.float32))).numpy().astype(float)
                penetrating[query_idx] = signed[query_idx] > 0.0
                if penetrating.any():
                    closest = sign_scene.compute_closest_points(o3d.core.Tensor(np.asarray(verts_obj[penetrating], dtype=np.float32)))["points"].numpy().astype(float)
                    correction_obj, solver = least_norm_halfspace_escape(verts_obj[penetrating], closest, signed[penetrating])
            correction_w = forward_object_vec(correction_obj, r_obj)
            correction_norm_m = float(np.linalg.norm(correction_w))
            max_penetration_m = float(np.nanmax(signed[penetrating])) if penetrating.any() else 0.0
            constraint_count = int(solver.get("constraint_count") or 0)
            amplification_bound_m = float(np.sqrt(max(1, constraint_count)) * max_penetration_m) if penetrating.any() else 0.0
            solver["max_penetration_depth_m"] = max_penetration_m
            solver["translation_to_max_penetration_ratio"] = float(correction_norm_m / max_penetration_m) if max_penetration_m > 0.0 else None
            solver["orthogonal_independent_constraints_bound_m"] = amplification_bound_m
            solver["bounded_local_escape"] = bool((not penetrating.any()) or (solver.get("success") is True and correction_norm_m <= amplification_bound_m + 1e-12))

            intr = frame_intrinsics(frame, side)
            proj0 = proj1 = None
            if intr is not None:
                proj0 = project(joints_w, r_c2w, t_c2w, intr)
                proj1 = project(joints_w + correction_w[None, :], r_c2w, t_c2w, intr)
            same_frame = bool(hand.get("same_frame_detection")) if hand.get("same_frame_detection") is not None else False
            visible = visible_2d_consistency(frame, hand, intr, proj0, proj1, same_frame, raw_video)

            if penetrating.any():
                if not solver.get("success"):
                    app_state = "not_applied_escape_solver_failed"
                    reason = "full bridge surface penetration exists but halfspace solver failed"
                elif not solver.get("bounded_local_escape"):
                    app_state = "not_applied_local_escape_amplified"
                    reason = "full bridge surface local escape translation is amplified beyond penetration-depth bound"
                elif not visible.get("compatible_with_visible_2d"):
                    app_state = "not_applied_visible_2d_conflict_or_unmeasured"
                    reason = "full bridge surface correction degrades visible 2D evidence"
                else:
                    app_state = "candidate_coordinate_correction_visible_2d_compatible"
                    reason = "full bridge surface signed penetration has bounded correction that preserves visible 2D evidence"
            elif near_surface.any() and not sign_aabb.any():
                app_state = "uncertainty_sign_mesh_missing_near_surface_support"
                reason = "full bridge surface is near observed surface but sign mesh lacks support"
            else:
                app_state = "no_penetration_no_coordinate_change_needed"
                reason = "full bridge surface sign-supporting compact-rigid mesh does not require correction"

            rows.append({
                "frame_idx": frame_idx,
                "hand_side": side,
                "same_frame_detection": same_frame,
                "object_id": args.object_id,
                "status": "full_bridge_mano_object_constraint_measured",
                "surface_mesh_path": str(Path(completion["outputs"]["completed_mesh_labeled"])),
                "sign_mesh_path": str(args.sign_mesh),
                "sign_mesh_source_report": str(args.sign_mesh_source_report),
                "signed_distance_semantics": "positive_inside_negative_outside_zero_on_surface_open3d_raycasting_converted_from_negative_inside",
                "signed_distance_query_scope": "all_full_bridge_hand_vertices",
                "near_surface_gate_applied_to_signed_distance": False,
                "sign_aabb_gate_applied_to_signed_distance": False,
                "completed_surface_mesh_watertight": bool(surface_mesh.is_watertight),
                "sign_mesh_watertight": bool(sign_mesh.is_watertight),
                "hand_geometry_source": "full_bridge_vertices_current_v18_world_from_hawor_projection_relift_m",
                "camera_pose_source_for_reprojection": camera_pose_source,
                "hand_vertex_count": int(len(verts_w)),
                "existing_hprime_translation_world_m": [float(x) for x in existing_delta.tolist()],
                "existing_hprime_translation_source": existing_delta_source,
                "observed_band_m": observed_band_m,
                "near_surface_vertex_count": int(near_surface.sum()),
                "near_surface_vertex_fraction": float(near_surface.mean()),
                "surface_aabb_candidate_vertex_count": int(surface_aabb.sum()),
                "surface_aabb_candidate_vertex_fraction": float(surface_aabb.mean()),
                "sign_aabb_candidate_vertex_count": int(sign_aabb.sum()),
                "sign_aabb_candidate_vertex_fraction": float(sign_aabb.mean()),
                "outside_sign_aabb_vertex_count": int((~sign_aabb).sum()),
                "outside_sign_aabb_vertex_fraction": float((~sign_aabb).mean()),
                "signed_query_candidate_vertex_count": int(signed_query.sum()),
                "signed_query_candidate_vertex_fraction": float(signed_query.mean()),
                "penetrating_vertex_count": int(penetrating.sum()),
                "penetrating_vertex_fraction": float(penetrating.mean()),
                "penetration_depth_m": numeric_summary(signed[penetrating]),
                "signed_distance_m": numeric_summary(signed[signed_query]),
                "nearest_surface_unsigned_m": numeric_summary(unsigned_surface_dist),
                "candidate_translation_world_m": [float(x) for x in correction_w.tolist()],
                "candidate_translation_norm_m": correction_norm_m,
                "candidate_translation_solver": solver,
                "candidate_joint_reprojection_shift_px": numeric_summary(np.linalg.norm((proj1 - proj0), axis=1)) if proj0 is not None and proj1 is not None else None,
                "candidate_visible_2d_consistency": visible,
                "candidate_application_state": app_state,
                "reason": reason,
                "bridge_npz": str(bridge_path),
                "bridge_row_index": int(bridge_row_index),
                "bridge_vertices_world_array": vertices_array,
            })

    counts = Counter(str(row.get("candidate_application_state")) for row in rows)
    report = {
        "method": "build_v18_full_bridge_mano_object_constraint_state",
        "status": "ok",
        "claim_scope": "Measures compact-rigid nonpenetration on full current-V18 bridge MANO surfaces by signed inside/outside evaluation for every bridge hand vertex, not a near-surface subset or 64-vertex annotation sample. Coordinate corrections are candidates until post-remeasured on the same full bridge surface.",
        "inputs": {
            "annotations": str(args.annotations),
            "hawor_npz": str(args.hawor_npz),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report),
            "sign_mesh": str(args.sign_mesh),
            "sign_mesh_source_report": str(args.sign_mesh_source_report),
        },
        "constraint_rows": rows,
        "summary": {
            "measured_pair_count": len(rows),
            "candidate_application_state_counts": dict(sorted(counts.items())),
        },
    }
    write_json(args.output_dir / "v18_mano_object_constraint_state_full_bridge.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
