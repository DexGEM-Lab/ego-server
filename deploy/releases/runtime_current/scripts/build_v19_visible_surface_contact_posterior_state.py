#!/usr/bin/env python3
"""Build a V19 contact posterior from observed object-owned visible surfels.

Completed object meshes can contain hidden/backside or broad prior surfaces.  For
contact support, a stricter measurement is the per-frame object-owned visible
surface lifted from SAM/depth.  This builder keeps metric MANO fixed and writes
posterior targets on those visible object surfels.  Missing or distant surfels are
evidence for low/absent visible contact support, not a reason to move the hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from build_v19_mano_surface_hypothesis_state import hawor_joint_map, numeric_summary, zero_summary
from refit_v19_mano_contact_similarity_interval import (
    as_list,
    camera_to_world,
    distance_to_mask,
    full_vertices_camera,
    frame_camera_pose,
    mask_membership,
    project_camera,
    sample_distance_map,
    target_object,
    uv_source_to_mask_xy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--hawor-npz", type=Path, required=True, help="Metric MANO joint source to preserve exactly; may be a hybrid NPZ")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--object-mask-dilation-px", type=int, default=8)
    p.add_argument("--object-proximity-px", type=float, default=65.0)
    p.add_argument("--target-locality-px", type=float, default=40.0)
    p.add_argument("--max-current-surface-distance-m", type=float, default=0.22)
    p.add_argument("--max-hand-behind-surface-m", type=float, default=0.08)
    p.add_argument("--contact-proximity-weight-px", type=float, default=45.0)
    p.add_argument("--contact-distance-weight-m", type=float, default=0.12)
    p.add_argument("--min-contact-vertices", type=int, default=16)
    p.add_argument("--max-contact-vertices", type=int, default=96)
    p.add_argument("--max-visible-surfels", type=int, default=2500)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def visible_surfels_camera(obj: dict[str, Any], max_visible_surfels: int) -> np.ndarray:
    cand = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    pts = np.asarray(cand.get("camera_vertices_sample_m") or [], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1.0e-5)
    pts = pts[valid]
    if len(pts) > int(max_visible_surfels):
        ids = np.linspace(0, len(pts) - 1, int(max_visible_surfels), dtype=np.int64)
        pts = pts[ids]
    return pts


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tree = cKDTree(pts)
    normals = np.zeros_like(pts)
    kk = min(max(3, int(k)), len(pts))
    for i, p in enumerate(pts):
        _dist, ids = tree.query(p, k=kk)
        nb = pts[np.atleast_1d(ids)]
        centered = nb - np.mean(nb, axis=0, keepdims=True)
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            n = vh[-1]
        except np.linalg.LinAlgError:
            n = np.asarray([0.0, 0.0, 1.0])
        # Orient approximately toward camera so normal-component magnitudes remain meaningful.
        if np.dot(n, p) > 0:
            n = -n
        norm = np.linalg.norm(n)
        normals[i] = n / max(norm, 1.0e-12)
    return normals


def source_gap_summaries(source: np.ndarray, target: np.ndarray, normals: np.ndarray) -> dict[str, Any]:
    if len(source) == 0 or len(target) == 0:
        empty = numeric_summary([])
        return {"distance": empty, "normal_abs": empty, "tangent": empty}
    n = min(len(source), len(target), len(normals))
    diff = source[:n] - target[:n]
    dist = np.linalg.norm(diff, axis=1)
    normal_abs = np.abs(np.sum(diff * normals[:n], axis=1))
    tangent = np.sqrt(np.maximum(0.0, dist * dist - normal_abs * normal_abs))
    return {
        "distance": numeric_summary(dist.astype(float).tolist()),
        "normal_abs": numeric_summary(normal_abs.astype(float).tolist()),
        "tangent": numeric_summary(tangent.astype(float).tolist()),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations = load_json(args.annotations)
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    source_joints = hawor_joint_map(args.hawor_npz)
    side_set = set(str(s) for s in args.sides)
    bridge_cache: dict[Path, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    distance_medians: list[float] = []
    normal_medians: list[float] = []
    tangent_medians: list[float] = []
    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.start_frame) or frame_idx > int(args.end_frame):
            continue
        source_w = int(frame.get("source_width") or raw_video.get("width") or 0)
        source_h = int(frame.get("source_height") or raw_video.get("height") or 0)
        if source_w <= 0 or source_h <= 0:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_source_size"})
            continue
        obj = target_object(frame, str(args.object_id))
        if not isinstance(obj, dict) or not obj.get("visible", False):
            skipped.append({"frame_idx": frame_idx, "reason": "missing_visible_object"})
            continue
        mask_path = obj.get("mask_path")
        if not isinstance(mask_path, str) or not Path(mask_path).exists():
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        surfels = visible_surfels_camera(obj, int(args.max_visible_surfels))
        if len(surfels) == 0:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_visible_object_surfels"})
            continue
        surfel_normals = estimate_normals(surfels)
        surfel_tree = cKDTree(surfels)
        obj_mask = dilate(load_mask(Path(mask_path)), int(args.object_mask_dilation_px))
        surfel_uv = None
        surfel_xy = None
        surfel_xy_valid = None
        surfel_2d_tree = None
        r_c2w, t_c2w = frame_camera_pose(frame)
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side") or "")
            if side not in side_set:
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=np.float64)
            verts = full_vertices_camera(metric, bridge_cache)
            intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or metric.get("v19_camera_intrinsics_fx_fy_cx_cy")
            if joints.shape != (21, 3) or verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0 or not isinstance(intr, list) or len(intr) != 4:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": "invalid_hand_metric_state"})
                continue
            intr_float = [float(x) for x in intr]
            if surfel_uv is None:
                surfel_uv = project_camera(surfels, intr_float)[0]
                surfel_xy, surfel_xy_valid = uv_source_to_mask_xy(surfel_uv, (source_w, source_h), obj_mask.shape)
                if np.any(surfel_xy_valid):
                    surfel_2d_tree = cKDTree(surfel_xy[np.flatnonzero(surfel_xy_valid)])
            uv_verts = project_camera(verts, intr_float)[0]
            xy_verts, valid_uv = uv_source_to_mask_xy(uv_verts, (source_w, source_h), obj_mask.shape)
            inside_mask = mask_membership(obj_mask, uv_verts, (source_w, source_h))
            proximity_dist_px = sample_distance_map(distance_to_mask(obj_mask), xy_verts, valid_uv)
            if float(args.target_locality_px) > 0.0 and surfel_2d_tree is not None and surfel_xy_valid is not None:
                valid_surfel_ids = np.flatnonzero(surfel_xy_valid)
                surface_dist_m = np.full((len(verts),), np.inf, dtype=np.float64)
                target_ids = np.full((len(verts),), -1, dtype=np.int64)
                targets = np.zeros_like(verts, dtype=np.float64)
                target_normals = np.zeros_like(verts, dtype=np.float64)
                for hand_id in np.flatnonzero(valid_uv):
                    local_ids = surfel_2d_tree.query_ball_point(xy_verts[hand_id], r=float(args.target_locality_px))
                    if not local_ids:
                        continue
                    sids = valid_surfel_ids[np.asarray(local_ids, dtype=np.int64)]
                    delta = surfels[sids] - verts[hand_id][None, :]
                    dist = np.linalg.norm(delta, axis=1)
                    best = int(np.argmin(dist))
                    sid = int(sids[best])
                    target_ids[hand_id] = sid
                    surface_dist_m[hand_id] = float(dist[best])
                    targets[hand_id] = surfels[sid]
                    target_normals[hand_id] = surfel_normals[sid]
            else:
                surface_dist_m, target_ids = surfel_tree.query(verts, k=1)
                targets = surfels[target_ids]
                target_normals = surfel_normals[target_ids]
            depth_delta_m = verts[:, 2] - targets[:, 2]
            selected = valid_uv & inside_mask
            selected |= valid_uv & (proximity_dist_px <= float(args.object_proximity_px))
            selected &= np.isfinite(surface_dist_m) & (surface_dist_m <= float(args.max_current_surface_distance_m))
            selected &= depth_delta_m <= float(args.max_hand_behind_surface_m)
            ids = np.where(selected)[0]
            if len(ids) < int(args.min_contact_vertices):
                skipped.append({
                    "frame_idx": frame_idx,
                    "side": side,
                    "reason": "too_few_visible_surface_support_vertices",
                    "count": int(len(ids)),
                    "inside_object_mask_vertices": int(np.count_nonzero(inside_mask)),
                    "visible_surfels": int(len(surfels)),
                    "surface_dist_m_p10": float(np.percentile(surface_dist_m[np.isfinite(surface_dist_m)], 10.0)) if np.any(np.isfinite(surface_dist_m)) else None,
                    "proximity_px_p10": float(np.percentile(proximity_dist_px[np.isfinite(proximity_dist_px)], 10.0)) if np.any(np.isfinite(proximity_dist_px)) else None,
                })
                continue
            score = (surface_dist_m[ids] / max(1.0e-6, float(args.max_current_surface_distance_m))) + (
                proximity_dist_px[ids] / max(1.0, float(args.object_proximity_px))
            )
            if len(ids) > int(args.max_contact_vertices):
                ids = ids[np.argsort(score)[: int(args.max_contact_vertices)]]
            selected_source = verts[ids]
            selected_targets = targets[ids]
            selected_normals = target_normals[ids]
            weights = np.exp(-0.5 * (proximity_dist_px[ids] / max(1.0, float(args.contact_proximity_weight_px))) ** 2)
            weights *= np.exp(-0.5 * (surface_dist_m[ids] / max(1.0e-6, float(args.contact_distance_weight_m))) ** 2)
            weights = np.clip(weights, 0.10, 1.0)
            gaps = source_gap_summaries(selected_source, selected_targets, selected_normals)
            for key, dest in (("distance", distance_medians), ("normal_abs", normal_medians), ("tangent", tangent_medians)):
                med = gaps[key].get("median") if isinstance(gaps[key], dict) else None
                if isinstance(med, (int, float)) and np.isfinite(float(med)):
                    dest.append(float(med))
            joints_world = source_joints.get((frame_idx, side))
            if joints_world is None:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_source_npz_joints"})
                continue
            source_world = camera_to_world(selected_source, r_c2w, t_c2w)
            target_world = camera_to_world(selected_targets, r_c2w, t_c2w)
            rows.append({
                "frame_idx": frame_idx,
                "source_frame_index": frame_idx,
                "hand_side": side,
                "interval_id": f"{side}_{frame_idx:04d}_visible_object_surface_posterior",
                "temporal_mano_state": "v19_source_metric_mano_plus_visible_object_surface_contact_posterior",
                "joint_state_policy": "hawor_npz_metric_mano_preserved",
                "optimized_joints_world_m": np.asarray(joints_world, dtype=np.float64).tolist(),
                "optimized_vertices_world_sample_m": target_world.astype(float).tolist(),
                "optimized_vertices_sample_ids": [],
                "object_surface_posterior_source_mano_vertex_ids": [int(x) for x in ids.tolist()],
                "source_contact_vertices_world_sample_m": source_world.astype(float).tolist(),
                "contact_surface_vertices_world_sample_m": target_world.astype(float).tolist(),
                "contact_surface_hypothesis_state": "uncertain_visible_object_surface_posterior_not_contact_ownership",
                "source_metric_mano_state": {"kind": "hawor_npz_or_hybrid_npz", "path": str(args.hawor_npz)},
                "source_hawor_npz": str(args.hawor_npz),
                "optimized_translation_world_m": [0.0, 0.0, 0.0],
                "optimized_translation_camera_m": [0.0, 0.0, 0.0],
                "optimized_rotation_vector_camera_rad": [0.0, 0.0, 0.0],
                "optimized_rotation_vector_world_rad": [0.0, 0.0, 0.0],
                "optimized_similarity_scale": 1.0,
                "optimized_rotation_norm_rad": 0.0,
                "metric_joint_shift_px": zero_summary(),
                "visible_joint_shift_px": zero_summary(),
                "contact_similarity_refit": {
                    "contact_residual_mode": "direct_object_surface_posterior",
                    "target_surface_source": "object_owned_visible_depth_surfels",
                    "solver_stage": "none_source_gaps_only",
                    "contact_solver_applied": False,
                    "contact_vertex_count": int(len(ids)),
                    "source_hand_to_object_surface_distance_m": gaps["distance"],
                    "source_hand_to_object_surface_normal_abs_m": gaps["normal_abs"],
                    "source_hand_to_object_surface_tangent_m": gaps["tangent"],
                    "contact_distance_after_m": gaps["distance"],
                    "contact_normal_abs_after_m": gaps["normal_abs"],
                    "contact_tangent_after_m": gaps["tangent"],
                    "contact_weight": numeric_summary(weights.astype(float).tolist()),
                    "candidate_stats": {
                        "candidate_mode": "object_owned_visible_depth_surfels",
                        "selected_vertices": int(len(ids)),
                        "inside_object_mask_vertices": int(np.count_nonzero(inside_mask)),
                        "visible_surfels": int(len(surfels)),
                        "selected_proximity_px": numeric_summary(proximity_dist_px[ids].astype(float).tolist()),
                        "selected_current_surface_distance_m": numeric_summary(surface_dist_m[ids].astype(float).tolist()),
                        "selected_depth_delta_m": numeric_summary(depth_delta_m[ids].astype(float).tolist()),
                        "selected_weight": numeric_summary(weights.astype(float).tolist()),
                        "target_locality_px": float(args.target_locality_px),
                    },
                },
                "direct_object_surface_source_gap_m": gaps["distance"],
                "full_observed_surface_penetration_after_solver_m": {"count": 0, "not_applicable": True, "reason": "visible-surface posterior does not solve nonpenetration"},
                "final_active_constraint_residual_after_solver_m": {"count": 0, "not_applicable": True, "reason": "visible-surface posterior does not solve nonpenetration"},
            })
    if not rows:
        raise RuntimeError(f"no visible-surface posterior rows; skipped={skipped[:20]}")
    payload = {
        "method": "v19_visible_object_surface_contact_posterior_state",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": (
            "Metric MANO joints are preserved. Posterior targets come only from per-frame object-owned visible depth surfels, "
            "so missing/high-gap rows are evidence about visible contact support, not hidden contact ownership or nonpenetration."
        ),
        "inputs": {"annotations": str(args.annotations), "hawor_npz": str(args.hawor_npz)},
        "parameters": vars(args) | {"annotations": str(args.annotations), "hawor_npz": str(args.hawor_npz), "output": str(args.output)},
        "summary": {
            "rows_out": len(rows),
            "skipped_count": len(skipped),
            "source_hand_to_object_surface_distance_median": numeric_summary(distance_medians),
            "source_hand_to_object_surface_normal_abs_median": numeric_summary(normal_medians),
            "source_hand_to_object_surface_tangent_median": numeric_summary(tangent_medians),
            "contact_distance_after_median": numeric_summary(distance_medians),
            "contact_normal_abs_after_median": numeric_summary(normal_medians),
            "contact_tangent_after_median": numeric_summary(tangent_medians),
            "metric_joint_shift_px": zero_summary(),
        },
        "skipped_preview": skipped[:180],
        "per_frame_states": rows,
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = build(args)
    write_json(args.output, payload)
    write_json(args.output.with_name(args.output.stem + "_report.json"), {k: v for k, v in payload.items() if k != "per_frame_states"})
    print(json.dumps({k: v for k, v in payload.items() if k != "per_frame_states"}, indent=2)[:20000])


if __name__ == "__main__":
    main()
