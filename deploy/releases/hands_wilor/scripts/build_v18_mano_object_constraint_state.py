#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import cKDTree
import trimesh
import open3d as o3d


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(str(path), process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh):
        raise RuntimeError(f"not mesh: {path}")
    if len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"empty mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def nearest_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "min": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def project(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray, intr: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intr]
    cam = (points_world - t_c2w) @ r_c2w
    z = np.maximum(cam[:, 2], 1e-9)
    return np.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], axis=1)


def detector_bbox_in_intrinsics_grid(frame: dict[str, Any], hand_ann: dict[str, Any], intr: list[float], raw_video: dict[str, Any]) -> np.ndarray | None:
    bbox = hand_ann.get("bbox_xyxy") if isinstance(hand_ann, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    raw_w = float(raw_video.get("width") or 0.0)
    raw_h = float(raw_video.get("height") or 0.0)
    if raw_w <= 0.0 or raw_h <= 0.0:
        return None
    _, _, cx, cy = [float(x) for x in intr]
    scaled = np.asarray(bbox, dtype=float).copy()
    scaled[[0, 2]] *= (2.0 * cx) / raw_w
    scaled[[1, 3]] *= (2.0 * cy) / raw_h
    return scaled


def bbox_inside_fraction(uv: np.ndarray, bbox: np.ndarray) -> float:
    if uv.ndim != 2 or uv.shape[0] == 0:
        return 0.0
    inside = (
        (uv[:, 0] >= bbox[0])
        & (uv[:, 0] <= bbox[2])
        & (uv[:, 1] >= bbox[1])
        & (uv[:, 1] <= bbox[3])
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
    )
    return float(np.mean(inside))


def visible_2d_consistency(frame: dict[str, Any], hand_ann: dict[str, Any], intr: list[float] | None, proj0: np.ndarray | None, proj1: np.ndarray | None, same_frame: bool, raw_video: dict[str, Any]) -> dict[str, Any]:
    if intr is None or proj0 is None or proj1 is None:
        return {"state": "not_evaluated_missing_projection", "compatible_with_visible_2d": False}
    bbox = detector_bbox_in_intrinsics_grid(frame, hand_ann, intr, raw_video)
    if bbox is None or not same_frame:
        return {"state": "no_same_frame_detector_box_constraint", "compatible_with_visible_2d": True}
    before = bbox_inside_fraction(proj0, bbox)
    after = bbox_inside_fraction(proj1, bbox)
    diag = float(np.linalg.norm([bbox[2] - bbox[0], bbox[3] - bbox[1]]))
    compatible = bool(after + 1e-12 >= before)
    return {
        "state": "preserves_or_improves_detector_box_containment" if compatible else "degrades_detector_box_containment",
        "compatible_with_visible_2d": compatible,
        "detector_bbox_intrinsics_grid_xyxy": [float(x) for x in bbox.tolist()],
        "projected_joint_inside_detector_box_fraction_before": before,
        "projected_joint_inside_detector_box_fraction_after": after,
        "detector_bbox_diag_px_in_intrinsics_grid": diag,
    }


def least_norm_halfspace_escape(points: np.ndarray, closest: np.ndarray, signed_depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Least-norm rigid translation satisfying local escape halfspaces.

    For each penetrating vertex p_i with closest surface point q_i, n_i=(q_i-p_i)/||q_i-p_i||
    is the local outward escape direction. A rigid translation t should satisfy
    n_i dot t >= signed_depth_i in the first-order signed-distance model. The
    solution is the projection of zero onto the intersection of those halfspaces.
    """
    disp = np.asarray(closest, dtype=float) - np.asarray(points, dtype=float)
    signed_depth = np.asarray(signed_depth, dtype=float)
    norms = np.linalg.norm(disp, axis=1)
    valid = (norms > 1e-12) & np.isfinite(norms) & np.isfinite(signed_depth) & (signed_depth > 0.0)
    if not np.any(valid):
        return np.zeros(3, dtype=float), {"solver": "no_valid_penetrating_halfspaces", "success": False, "constraint_count": 0}
    normals = disp[valid] / norms[valid, None]
    depths = signed_depth[valid]
    x0 = np.average(disp[valid], axis=0, weights=np.maximum(depths, 1e-12))
    constraint = LinearConstraint(normals, depths, np.full_like(depths, np.inf))
    res = minimize(lambda x: 0.5 * float(np.dot(x, x)), x0=x0, jac=lambda x: np.asarray(x, dtype=float), constraints=[constraint], method="SLSQP", options={"ftol": 1e-12, "maxiter": 200, "disp": False})
    if res.success and np.all(np.isfinite(res.x)):
        slack = normals @ res.x - depths
        return np.asarray(res.x, dtype=float), {
            "solver": "least_norm_local_halfspace_escape",
            "success": True,
            "constraint_count": int(len(depths)),
            "min_constraint_slack_m": float(np.min(slack)) if slack.size else None,
            "max_constraint_violation_m": float(max(0.0, -float(np.min(slack)))) if slack.size else 0.0,
        }
    # Fall back to the weighted mean but mark that the first-order constraints
    # were not solved; downstream must not treat this as validated H'.
    fallback = np.average(disp[valid], axis=0, weights=np.maximum(depths, 1e-12))
    slack = normals @ fallback - depths
    return np.asarray(fallback, dtype=float), {
        "solver": "weighted_mean_escape_fallback",
        "success": False,
        "constraint_count": int(len(depths)),
        "message": str(getattr(res, "message", "unknown")),
        "min_constraint_slack_m": float(np.min(slack)) if slack.size else None,
        "max_constraint_violation_m": float(max(0.0, -float(np.min(slack)))) if slack.size else None,
    }


def inverse_object(points_world: np.ndarray, r_w_from_c: np.ndarray, t_w: np.ndarray) -> np.ndarray:
    return (points_world - t_w) @ r_w_from_c


def forward_object_vec(vec_canonical: np.ndarray, r_w_from_c: np.ndarray) -> np.ndarray:
    return vec_canonical @ r_w_from_c.T


def hand_arrays(arrays: dict[str, np.ndarray], side: str, frame_pos: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(arrays[f"{side}_vertices_world_m"][frame_pos], dtype=float), np.asarray(arrays[f"{side}_joints_world_m"][frame_pos], dtype=float)


def frame_intrinsics(frame: dict[str, Any], side: str) -> list[float] | None:
    for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
        if hand.get("hand_side") == side:
            metric_state = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            intr = hand.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or metric_state.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if intr and len(intr) == 4:
                return [float(x) for x in intr]
    return None


def frame_camera_pose(frame: dict[str, Any], fallback_r_c2w: np.ndarray, fallback_t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    transform = np.asarray(camera.get("T_world_camera_metric") or [], dtype=float)
    if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
        return transform[:3, :3], transform[:3, 3], "annotation_frame_T_world_camera_metric"
    return fallback_r_c2w, fallback_t_c2w, "hawor_npz_camera_pose_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--pose-report", type=Path, required=True)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--sign-mesh", type=Path, default=None, help="optional watertight aligned hidden-volume mesh for signed nonpenetration hypotheses")
    parser.add_argument("--sign-mesh-source-report", type=Path, default=None)
    parser.add_argument("--skip-signed-distance", action="store_true", help="broadphase only: do not run expensive exact signed distance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    args = parser.parse_args()

    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    completion = load_json(args.completion_report)
    mesh_path = Path(completion["outputs"]["completed_mesh_labeled"])
    observed_band_m = float(completion.get("observed_band_m") or 0.0)
    surface_mesh = load_mesh(mesh_path)
    surface_mesh_watertight = bool(surface_mesh.is_watertight)
    surface_bounds_min = np.asarray(surface_mesh.bounds[0], dtype=float)
    surface_bounds_max = np.asarray(surface_mesh.bounds[1], dtype=float)
    surface_samples, _ = trimesh.sample.sample_surface(surface_mesh, min(12000, max(1, len(surface_mesh.faces))), seed=np.random.default_rng(1818))
    surface_tree = cKDTree(np.asarray(surface_samples, dtype=float))

    sign_mesh_path = args.sign_mesh if args.sign_mesh is not None else mesh_path
    sign_mesh = load_mesh(sign_mesh_path)
    sign_query = None
    sign_scene = None
    if not args.skip_signed_distance:
        sign_scene = o3d.t.geometry.RaycastingScene()
        sign_scene.add_triangles(
            o3d.core.Tensor(np.asarray(sign_mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(sign_mesh.faces, dtype=np.uint32)),
        )
    sign_mesh_watertight = bool(sign_mesh.is_watertight)
    sign_bounds_min = np.asarray(sign_mesh.bounds[0], dtype=float)
    sign_bounds_max = np.asarray(sign_mesh.bounds[1], dtype=float)

    npz = np.load(args.hawor_npz, allow_pickle=True)
    arrays = {key: np.asarray(npz[key]) for key in npz.files}
    frame_indices = np.asarray(arrays["frame_idx"], dtype=int)
    frame_pos_by_idx = {int(f): i for i, f in enumerate(frame_indices)}
    frames_by_idx = {int(f.get("frame_idx")): f for f in annotations.get("frames", [])}
    raw_video_meta = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    accepted_pose_statuses = {
        "fit_to_visible_depth_samples",
        "fit_to_visible_depth_archive_vertices",
        "corrected_temporal_rigid_pose_graph",
        "completed_temporal_rigid_pose_uncertain",
    }
    pose_by_idx = {int(row["frame_idx"]): row for row in pose_report.get("pose_rows", []) if row.get("status") in accepted_pose_statuses}

    rows = []
    corrective_rows = []
    for frame_idx, pose in pose_by_idx.items():
        if frame_idx not in frame_pos_by_idx:
            continue
        frame_pos = frame_pos_by_idx[frame_idx]
        frame = frames_by_idx.get(frame_idx, {})
        r_obj = np.asarray(pose["rotation_world_from_completed_canonical_matrix"], dtype=float)
        t_obj = np.asarray(pose["translation_world_m"], dtype=float)
        r_c2w_npz = np.asarray(arrays["R_c2w"][frame_pos], dtype=float)
        t_c2w_npz = np.asarray(arrays["t_c2w"][frame_pos], dtype=float)
        r_c2w, t_c2w, camera_pose_source = frame_camera_pose(frame, r_c2w_npz, t_c2w_npz)
        hands_by_side = {str(h.get("hand_side")): h for h in frame.get("hands", []) if isinstance(h, dict)}
        for side in ["left", "right"]:
            hand_ann = hands_by_side.get(side, {})
            metric_state = hand_ann.get("metric_mano_state") if isinstance(hand_ann.get("metric_mano_state"), dict) else {}
            ann_vertices_raw = hand_ann.get("vertices_world_sample_m") or metric_state.get("vertices_world_sample_m") or []
            ann_joints_raw = hand_ann.get("joints_current_v18_world_m") or metric_state.get("joints_current_v18_world_m") or []
            ann_vertices = np.asarray(ann_vertices_raw, dtype=float) if isinstance(hand_ann, dict) else np.asarray([], dtype=float)
            ann_joints = np.asarray(ann_joints_raw, dtype=float) if isinstance(hand_ann, dict) else np.asarray([], dtype=float)
            if ann_vertices.ndim == 2 and ann_vertices.shape[1] == 3 and len(ann_vertices) > 0:
                verts_w = ann_vertices
                if ann_joints.ndim == 2 and ann_joints.shape[1] == 3 and len(ann_joints) > 0:
                    joints_w = ann_joints
                else:
                    _, joints_w = hand_arrays(arrays, side, frame_pos)
                hand_geometry_source = "delivered_annotation_metric_mano_vertices_world_sample_m"
            else:
                if int(arrays[f"{side}_valid"][frame_pos]) == 0:
                    continue
                verts_w, joints_w = hand_arrays(arrays, side, frame_pos)
                hand_geometry_source = "hawor_npz_full_vertices_world_m_fallback"
            same_frame = bool(hand_ann.get("same_frame_detection")) if isinstance(hand_ann, dict) and hand_ann.get("same_frame_detection") is not None else (bool(int(arrays[f"{side}_detected_same_frame"][frame_pos])) if f"{side}_detected_same_frame" in arrays else False)
            verts_c = inverse_object(verts_w, r_obj, t_obj)
            surface_aabb = ((verts_c >= surface_bounds_min) & (verts_c <= surface_bounds_max)).all(axis=1)
            sign_aabb = ((verts_c >= sign_bounds_min) & (verts_c <= sign_bounds_max)).all(axis=1)
            unsigned_surface_dist, _ = surface_tree.query(verts_c, k=1, workers=-1)
            near_surface = unsigned_surface_dist <= observed_band_m if observed_band_m > 0 else np.zeros(len(verts_c), dtype=bool)
            signed = np.full(len(verts_c), np.nan, dtype=float)
            penetrating = np.zeros(len(verts_c), dtype=bool)
            correction_c = np.zeros(3, dtype=float)
            correction_solver = {"solver": "not_needed_no_penetration", "success": True, "constraint_count": 0}
            signed_query_candidate = sign_aabb & near_surface
            if sign_mesh_watertight and signed_query_candidate.any() and sign_scene is not None:
                candidate_idx = np.where(signed_query_candidate)[0]
                query_tensor = o3d.core.Tensor(np.asarray(verts_c[candidate_idx], dtype=np.float32))
                # Open3D convention is negative inside, positive outside. Convert
                # to this report's convention: positive inside/penetrating.
                candidate_signed = -sign_scene.compute_signed_distance(query_tensor).numpy().astype(float)
                signed[candidate_idx] = candidate_signed
                penetrating[candidate_idx] = candidate_signed > 0.0
                if penetrating.any():
                    penetrating_points = o3d.core.Tensor(np.asarray(verts_c[penetrating], dtype=np.float32))
                    closest = sign_scene.compute_closest_points(penetrating_points)["points"].numpy().astype(float)
                    correction_c, correction_solver = least_norm_halfspace_escape(verts_c[penetrating], closest, signed[penetrating])
            correction_w = forward_object_vec(correction_c, r_obj)
            correction_norm_m = float(np.linalg.norm(correction_w))
            max_penetration_m = float(np.nanmax(signed[penetrating])) if penetrating.any() else 0.0
            constraint_count = int(correction_solver.get("constraint_count") or 0)
            amplification_bound_m = float(np.sqrt(max(1, constraint_count)) * max_penetration_m) if penetrating.any() else 0.0
            amplification_ratio = float(correction_norm_m / max_penetration_m) if max_penetration_m > 0.0 else None
            bounded_local_escape = bool((not penetrating.any()) or (correction_solver.get("success") is True and correction_norm_m <= amplification_bound_m + 1e-12))
            correction_solver["max_penetration_depth_m"] = max_penetration_m
            correction_solver["translation_to_max_penetration_ratio"] = amplification_ratio
            correction_solver["orthogonal_independent_constraints_bound_m"] = amplification_bound_m
            correction_solver["bounded_local_escape"] = bounded_local_escape
            correction_solver["bounded_local_escape_semantics"] = "accepted only when ||translation|| <= sqrt(num_penetrating_vertices) * max_penetration_depth; larger amplification indicates conflicting local escape directions or invalid sign support for a rigid hand correction"
            joints_corr_w = joints_w + correction_w
            intr = frame_intrinsics(frame, side)
            reproj_summary = None
            proj0 = None
            proj1 = None
            visible_consistency = {"state": "not_evaluated_missing_projection", "compatible_with_visible_2d": False}
            if intr is not None:
                proj0 = project(joints_w, r_c2w, t_c2w, intr)
                proj1 = project(joints_corr_w, r_c2w, t_c2w, intr)
                reproj_summary = nearest_summary(np.linalg.norm(proj1 - proj0, axis=1))
                visible_consistency = visible_2d_consistency(frame, hand_ann, intr, proj0, proj1, same_frame, raw_video_meta)
            if penetrating.any() and bounded_local_escape and visible_consistency.get("compatible_with_visible_2d") is True:
                app_state = "candidate_coordinate_correction_visible_2d_compatible"
                reason = "watertight sign mesh predicts local MANO/object penetration; least-norm local halfspace translation satisfies penetrating-vertex escape constraints and does not contradict available same-frame 2D hand-box evidence"
            elif penetrating.any() and correction_solver.get("success") is not True:
                app_state = "not_applied_escape_solver_failed"
                reason = "watertight sign mesh predicts MANO/object penetration, but the rigid hand translation halfspace solver failed; coordinate update is held as uncertainty"
            elif penetrating.any() and not bounded_local_escape:
                app_state = "not_applied_local_escape_amplified"
                reason = "watertight sign mesh predicts MANO/object penetration, but satisfying local escape halfspaces requires translation larger than the penetration-depth-derived orthogonal constraint bound; this is treated as sign-support inconsistency, not H-prime"
            elif penetrating.any():
                app_state = "not_applied_visible_2d_conflict_or_unmeasured"
                reason = "watertight sign mesh predicts MANO/object penetration, but available visible 2D consistency is missing or degraded; coordinate update is held"
            elif signed_query_candidate.any() and args.skip_signed_distance:
                app_state = "uncertainty_signed_distance_not_evaluated_broadphase_support"
                reason = "MANO vertices are within the observed-surface band and inside the sign-mesh AABB, but exact signed distance was intentionally skipped for broadphase measurement"
            elif near_surface.any() and sign_mesh_watertight and not sign_aabb.any():
                app_state = "uncertainty_sign_mesh_missing_near_surface_support"
                reason = "MANO vertices are within the observed-surface voxel band, but the watertight sign mesh does not cover those vertices; this falsifies using the sign mesh as a nonpenetration validator for this contact region"
            elif surface_aabb.any() and not sign_mesh_watertight:
                app_state = "uncertainty_only_nonwatertight_mesh_no_signed_correction"
                reason = "surface overlap exists but no watertight sign mesh is available, so this changes uncertainty only"
            else:
                app_state = "no_penetration_no_coordinate_change_needed"
                reason = "sign-supporting compact-rigid mesh does not require nonpenetration correction for this hand/frame"
            row = {
                "frame_idx": int(frame_idx),
                "hand_side": side,
                "same_frame_detection": same_frame,
                "object_id": args.object_id,
                "status": "mano_object_constraint_measured",
                "surface_mesh_path": str(mesh_path),
                "sign_mesh_path": str(sign_mesh_path),
                "sign_mesh_source_report": str(args.sign_mesh_source_report) if args.sign_mesh_source_report else None,
                "signed_distance_semantics": "positive_inside_negative_outside_zero_on_surface_open3d_raycasting_converted_from_negative_inside",
                "completed_surface_mesh_watertight": surface_mesh_watertight,
                "sign_mesh_watertight": sign_mesh_watertight,
                "hand_geometry_source": hand_geometry_source,
                "camera_pose_source_for_reprojection": camera_pose_source,
                "hand_vertex_count": int(len(verts_w)),
                "observed_band_m": observed_band_m,
                "near_surface_vertex_count": int(near_surface.sum()),
                "near_surface_vertex_fraction": float(near_surface.mean()),
                "surface_aabb_candidate_vertex_count": int(surface_aabb.sum()),
                "surface_aabb_candidate_vertex_fraction": float(surface_aabb.mean()),
                "sign_aabb_candidate_vertex_count": int(sign_aabb.sum()),
                "sign_aabb_candidate_vertex_fraction": float(sign_aabb.mean()),
                "signed_query_candidate_vertex_count": int(signed_query_candidate.sum()),
                "signed_query_candidate_vertex_fraction": float(signed_query_candidate.mean()),
                "penetrating_vertex_count": int(penetrating.sum()),
                "penetrating_vertex_fraction": float(penetrating.mean()),
                "penetration_depth_m": nearest_summary(signed[penetrating]) if sign_mesh_watertight else nearest_summary(np.asarray([], dtype=float)),
                "signed_distance_m": nearest_summary(signed[np.isfinite(signed)]),
                "nearest_surface_unsigned_m": nearest_summary(np.asarray(unsigned_surface_dist, dtype=float)),
                "candidate_translation_world_m": correction_w.astype(float).tolist(),
                "candidate_translation_norm_m": correction_norm_m,
                "candidate_translation_solver": correction_solver,
                "candidate_joint_reprojection_shift_px": reproj_summary,
                "candidate_visible_2d_consistency": visible_consistency,
                "candidate_application_state": app_state,
                "reason": reason,
            }
            rows.append(row)
            if penetrating.any():
                corrective_rows.append(row)

    by_side = {}
    for side in ["left", "right"]:
        side_rows = [r for r in rows if r["hand_side"] == side]
        correction_norms = np.asarray([r["candidate_translation_norm_m"] for r in side_rows], dtype=float)
        penetrations = np.asarray([r["penetrating_vertex_count"] for r in side_rows], dtype=float)
        by_side[side] = {
            "measured_frames": len(side_rows),
            "frames_with_any_penetration": int(np.sum(penetrations > 0)) if len(penetrations) else 0,
            "candidate_translation_norm_m": nearest_summary(correction_norms),
        }
    report = {
        "method": "build_v18_mano_object_constraint_state",
        "status": "ok",
        "claim_scope": "Completed posed object mesh supplies surface/overlap measurements; an optional watertight aligned sign mesh supplies signed nonpenetration hypotheses. Coordinate corrections remain candidates until visible 2D consistency is inspected.",
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "hawor_npz": str(args.hawor_npz),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report),
            "completed_surface_mesh": str(mesh_path),
            "sign_mesh": str(sign_mesh_path),
            "sign_mesh_source_report": str(args.sign_mesh_source_report) if args.sign_mesh_source_report else None,
            "skip_signed_distance": bool(args.skip_signed_distance),
        },
        "completed_surface_mesh_watertight": surface_mesh_watertight,
        "sign_mesh_watertight": sign_mesh_watertight,
        "measured_pair_count": len(rows),
        "candidate_correction_count": len(corrective_rows),
        "summary_by_side": by_side,
        "constraint_rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "v18_mano_object_constraint_state.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "object_id", "completed_surface_mesh_watertight", "sign_mesh_watertight", "measured_pair_count", "candidate_correction_count", "summary_by_side"]}, indent=2))


if __name__ == "__main__":
    main()
