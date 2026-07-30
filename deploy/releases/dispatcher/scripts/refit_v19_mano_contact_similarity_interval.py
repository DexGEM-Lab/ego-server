#!/usr/bin/env python3
"""Contact-aware V19 MANO similarity refit for a bounded interval.

This repairs a measurement, not the rigid object: for each frame/side it keeps the
current MANO articulation as the local hand shape, optimizes a small Sim(3) in
the frame camera, preserves the current image-space hand projection, and pulls
image-adjacent hand vertices toward the fitted rigid object surface as uncertain
contact evidence.  The output is a temporal MANO state compatible with the V18
interval renderer.  It is not an accepted contact anchor or MANO parameter solve;
scale/2D/contact residuals decide whether the mechanism is physically plausible.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--object-mask-dilation-px", type=int, default=18)
    p.add_argument("--candidate-mode", choices=("object_mask", "near_surface", "object_mask_or_near_surface"), default="object_mask_or_near_surface")
    p.add_argument("--object-proximity-px", type=float, default=55.0)
    p.add_argument("--mesh-projection-dilation-px", type=int, default=8)
    p.add_argument("--max-current-surface-distance-m", type=float, default=0.22)
    p.add_argument("--max-hand-behind-surface-m", type=float, default=0.08)
    p.add_argument("--contact-proximity-weight-px", type=float, default=45.0)
    p.add_argument("--contact-distance-weight-m", type=float, default=0.12)
    p.add_argument("--min-contact-vertices", type=int, default=6)
    p.add_argument("--max-contact-vertices", type=int, default=48)
    p.add_argument("--mesh-sample-stride", type=int, default=32)
    p.add_argument(
        "--target-locality-px",
        type=float,
        default=0.0,
        help=(
            "If positive, choose each contact target from object mesh samples whose projected location is within this "
            "many mask pixels of the source MANO vertex, instead of using the global nearest object surface point."
        ),
    )
    p.add_argument("--sigma-reprojection-px", type=float, default=8.0)
    p.add_argument("--contact-residual-mode", choices=("point_to_point", "point_to_plane"), default="point_to_point")
    p.add_argument("--sigma-contact-m", type=float, default=0.045)
    p.add_argument("--sigma-translation-m", type=float, default=0.16)
    p.add_argument("--sigma-rotation-rad", type=float, default=0.45)
    p.add_argument("--sigma-log-scale", type=float, default=0.18)
    p.add_argument("--sigma-temporal-center-m", type=float, default=0.055)
    p.add_argument("--sigma-temporal-log-scale", type=float, default=0.06)
    p.add_argument("--min-scale", type=float, default=0.75)
    p.add_argument("--max-scale", type=float, default=1.35)
    p.add_argument("--max-translation-m", type=float, default=0.28)
    p.add_argument("--max-rotation-rad", type=float, default=0.85)
    p.add_argument("--max-nfev", type=int, default=100)
    return p.parse_args()


def frame_camera_pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cam = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
    raw = cam.get("T_world_camera_metric") or cam.get("T_world_camera")
    mat = np.asarray(raw, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"frame {frame.get('frame_idx')} lacks valid camera matrix")
    return mat[:3, :3], mat[:3, 3]


def world_to_camera(points_world: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return (points_world - t_c2w[None, :]) @ r_c2w


def camera_to_world(points_camera: np.ndarray, r_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    return points_camera @ r_c2w.T + t_c2w[None, :]


def project_camera(points_camera: np.ndarray, intr: list[float], width: int | None = None, height: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.asarray(points_camera, dtype=float)
    fx, fy, cx, cy = [float(x) for x in intr]
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1.0e-5)
    u = fx * pts[:, 0] / np.maximum(z, 1.0e-6) + cx
    v = fy * pts[:, 1] / np.maximum(z, 1.0e-6) + cy
    uv = np.stack([u, v], axis=1)
    if width is not None and height is not None:
        scale_x = width / max(1.0, 2.0 * cx)
        scale_y = height / max(1.0, 2.0 * cy)
        uv_mask = np.stack([u * scale_x, v * scale_y], axis=1)
        valid &= (uv_mask[:, 0] >= 0) & (uv_mask[:, 0] < width) & (uv_mask[:, 1] >= 0) & (uv_mask[:, 1] < height)
        return uv, valid
    return uv, valid


def mask_membership(mask: np.ndarray, uv_source: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    source_w, source_h = source_size
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"invalid source_size {source_size}")
    scale_x = float(w) / float(source_w)
    scale_y = float(h) / float(source_h)
    u = np.rint(uv_source[:, 0] * scale_x).astype(int)
    v = np.rint(uv_source[:, 1] * scale_y).astype(int)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.zeros((len(uv_source),), dtype=bool)
    out[valid] = mask[v[valid], u[valid]]
    return out


def uv_source_to_mask_xy(uv_source: np.ndarray, source_size: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask_shape
    source_w, source_h = source_size
    scale_x = float(w) / float(source_w)
    scale_y = float(h) / float(source_h)
    xy = np.stack([uv_source[:, 0] * scale_x, uv_source[:, 1] * scale_y], axis=1)
    valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0.0) & (xy[:, 0] < w) & (xy[:, 1] >= 0.0) & (xy[:, 1] < h)
    return xy, valid


def rasterize_projected_points(points_camera: np.ndarray, intr: list[float], source_size: tuple[int, int], mask_shape: tuple[int, int], dilation_px: int) -> np.ndarray:
    mask = np.zeros(mask_shape, dtype=np.uint8)
    uv_source, _valid_z = project_camera(points_camera, intr)
    xy, valid = uv_source_to_mask_xy(uv_source, source_size, mask_shape)
    if np.any(valid):
        pts = np.rint(xy[valid]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, mask_shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, mask_shape[0] - 1)
        mask[pts[:, 1], pts[:, 0]] = 1
    if int(dilation_px) > 0:
        radius = int(dilation_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)


def sample_distance_map(distance_map: np.ndarray, xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full((len(xy),), np.inf, dtype=np.float64)
    if not np.any(valid):
        return out
    h, w = distance_map.shape
    xi = np.rint(xy[valid, 0]).astype(int)
    yi = np.rint(xy[valid, 1]).astype(int)
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    out[np.flatnonzero(valid)] = distance_map[yi, xi].astype(float)
    return out


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def full_vertices_camera(metric: dict[str, Any], bridge_cache: dict[Path, dict[str, np.ndarray]]) -> np.ndarray:
    ref = metric.get("vertices_reference") if isinstance(metric.get("vertices_reference"), dict) else {}
    raw_path = ref.get("bridge_npz")
    raw_row = ref.get("bridge_row_index")
    if isinstance(raw_path, str) and raw_row is not None:
        path = Path(raw_path)
        if path.exists():
            if path not in bridge_cache:
                with np.load(path, allow_pickle=True) as z:
                    bridge_cache[path] = {key: np.asarray(z[key]) for key in z.files}
            z = bridge_cache[path]
            key = str(ref.get("bridge_vertices_camera_array") or "vertices_current_v18_camera_m")
            if key in z:
                verts = np.asarray(z[key][int(raw_row)], dtype=float)
                if verts.ndim == 2 and verts.shape[1] == 3 and len(verts) >= 64:
                    return verts
    verts = np.asarray(metric.get("vertices_camera_sample_m") or [], dtype=float)
    if verts.ndim == 2 and verts.shape[1] == 3 and len(verts) > 0:
        return verts
    return np.zeros((0, 3), dtype=float)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def target_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    bare = object_id.split(":", 1)[-1]
    for obj in as_list(frame.get("objects")):
        if not isinstance(obj, dict):
            continue
        ids = {str(obj.get("object_id")), str(obj.get("track_id")), f"object:{obj.get('object_id')}", f"object:{obj.get('track_id')}"}
        if object_id in ids or bare in ids:
            return obj
    return None


def pose_map(report: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for row in as_list(report.get("pose_rows")):
        if not isinstance(row, dict) or not isinstance(row.get("frame_idx"), int):
            continue
        rot = row.get("rotation_world_from_completed_canonical_matrix") or row.get("rotation_world_from_object_matrix")
        trans = row.get("translation_world_m")
        if rot is None or trans is None:
            continue
        r = np.asarray(rot, dtype=float)
        t = np.asarray(trans, dtype=float)
        if r.shape == (3, 3) and t.shape == (3,):
            out[int(row["frame_idx"])] = (r, t)
    return out


def numeric_summary(vals: list[float]) -> dict[str, Any]:
    finite = sorted(float(v) for v in vals if np.isfinite(v))
    if not finite:
        return {"count": 0}
    def q(frac: float) -> float:
        idx = min(len(finite) - 1, max(0, int(round(frac * (len(finite) - 1)))))
        return finite[idx]
    return {"count": len(finite), "min": finite[0], "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": finite[-1], "mean": float(np.mean(finite))}


class Obs:
    def __init__(
        self,
        *,
        frame_idx: int,
        side: str,
        joints_cam: np.ndarray,
        verts_cam: np.ndarray,
        intr: list[float],
        r_c2w: np.ndarray,
        t_c2w: np.ndarray,
        contact_idx: np.ndarray,
        contact_targets_cam: np.ndarray,
        contact_normals_cam: np.ndarray,
        contact_weights: np.ndarray,
        candidate_stats: dict[str, Any],
    ):
        self.frame_idx = frame_idx
        self.side = side
        self.joints_cam = joints_cam
        self.verts_cam = verts_cam
        self.intr = intr
        self.r_c2w = r_c2w
        self.t_c2w = t_c2w
        self.center = np.median(joints_cam, axis=0)
        self.local_joints = joints_cam - self.center[None, :]
        self.local_verts = verts_cam - self.center[None, :]
        self.base_uv = project_camera(joints_cam, intr)[0]
        self.contact_idx = contact_idx.astype(int)
        self.contact_targets_cam = contact_targets_cam.astype(float)
        normals = np.asarray(contact_normals_cam, dtype=float)
        if normals.shape != self.contact_targets_cam.shape:
            raise ValueError("contact_normals_cam must match contact_targets_cam")
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        self.contact_normals_cam = normals / np.maximum(norm, 1.0e-12)
        weights = np.asarray(contact_weights, dtype=float).reshape(-1)
        if weights.shape != (len(self.contact_idx),):
            raise ValueError("contact_weights must match contact_idx")
        self.contact_weights = np.clip(weights, 0.05, 1.0)
        self.candidate_stats = dict(candidate_stats)


def transform(local: np.ndarray, center: np.ndarray, params: np.ndarray) -> np.ndarray:
    rot = Rotation.from_rotvec(params[:3]).as_matrix()
    scale = math.exp(float(params[6]))
    return scale * (local @ rot.T) + center[None, :] + params[3:6][None, :]


def build_observations(args: argparse.Namespace) -> tuple[list[Obs], list[dict[str, Any]]]:
    annotations = load_json(args.annotations)
    raw_video = annotations.get("raw_video") if isinstance(annotations.get("raw_video"), dict) else {}
    poses = pose_map(load_json(args.pose_report))
    mesh = trimesh.load(args.completed_mesh, process=False)
    verts_obj = np.asarray(mesh.vertices, dtype=float)
    normals_obj = np.asarray(mesh.vertex_normals, dtype=float)
    if len(verts_obj) > 0:
        sample_stride = max(1, int(args.mesh_sample_stride))
        verts_obj_sample = verts_obj[::sample_stride]
        normals_obj_sample = normals_obj[::sample_stride] if normals_obj.shape == verts_obj.shape else np.zeros_like(verts_obj_sample)
    else:
        raise ValueError(f"mesh has no vertices: {args.completed_mesh}")
    obs: list[Obs] = []
    skipped: list[dict[str, Any]] = []
    bridge_cache: dict[Path, dict[str, np.ndarray]] = {}
    side_set = set(str(s) for s in args.sides)
    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.start_frame) or frame_idx > int(args.end_frame):
            continue
        if frame_idx not in poses:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_pose"})
            continue
        source_w = int(frame.get("source_width") or raw_video.get("width") or 0)
        source_h = int(frame.get("source_height") or raw_video.get("height") or 0)
        if source_w <= 0 or source_h <= 0:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_source_size_for_mask_scaling"})
            continue
        obj = target_object(frame, str(args.object_id))
        if not isinstance(obj, dict) or obj.get("rigid_pose_observation_eligible") is False:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_or_ineligible_object_mask"})
            continue
        mask_path = obj.get("mask_path")
        if not isinstance(mask_path, str) or not Path(mask_path).exists():
            skipped.append({"frame_idx": frame_idx, "reason": "missing_object_mask"})
            continue
        obj_mask = dilate(load_mask(Path(mask_path)), int(args.object_mask_dilation_px))
        r_c2w, t_c2w = frame_camera_pose(frame)
        r_obj, t_obj = poses[frame_idx]
        obj_world = verts_obj_sample @ r_obj.T + t_obj[None, :]
        obj_normals_world = normals_obj_sample @ r_obj.T
        obj_cam_full = world_to_camera(obj_world, r_c2w, t_c2w)
        obj_normals_cam_full = obj_normals_world @ r_c2w
        valid_obj = np.isfinite(obj_cam_full).all(axis=1) & (obj_cam_full[:, 2] > 1.0e-5) & np.isfinite(obj_normals_cam_full).all(axis=1)
        obj_cam = obj_cam_full[valid_obj]
        obj_normals_cam = obj_normals_cam_full[valid_obj]
        nrm = np.linalg.norm(obj_normals_cam, axis=1, keepdims=True)
        obj_normals_cam = obj_normals_cam / np.maximum(nrm, 1.0e-12)
        if len(obj_cam) == 0:
            skipped.append({"frame_idx": frame_idx, "reason": "empty_object_camera_points"})
            continue
        tree = cKDTree(obj_cam)
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side") or hand.get("side") or "")
            if side not in side_set:
                continue
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            joints = np.asarray(metric.get("joints_current_v18_camera_m") or [], dtype=float)
            verts = full_vertices_camera(metric, bridge_cache)
            intr = metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or metric.get("v19_camera_intrinsics_fx_fy_cx_cy")
            if joints.shape != (21, 3) or verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0 or not isinstance(intr, list) or len(intr) != 4:
                skipped.append({"frame_idx": frame_idx, "side": side, "reason": "invalid_hand_metric_state"})
                continue
            intr_float = [float(x) for x in intr]
            uv_verts = project_camera(verts, intr_float)[0]
            xy_verts, valid_uv = uv_source_to_mask_xy(uv_verts, (source_w, source_h), obj_mask.shape)
            inside_mask = mask_membership(obj_mask, uv_verts, (source_w, source_h))
            mesh_projection = rasterize_projected_points(
                obj_cam,
                intr_float,
                (source_w, source_h),
                obj_mask.shape,
                int(args.mesh_projection_dilation_px),
            )
            proximity_mask = obj_mask | mesh_projection
            proximity_dist_px = sample_distance_map(distance_to_mask(proximity_mask), xy_verts, valid_uv)
            target_locality_px = float(getattr(args, "target_locality_px", 0.0) or 0.0)
            if target_locality_px > 0.0:
                obj_uv_source = project_camera(obj_cam, intr_float)[0]
                obj_xy, obj_xy_valid = uv_source_to_mask_xy(obj_uv_source, (source_w, source_h), obj_mask.shape)
                surface_dist_m = np.full((len(verts),), np.inf, dtype=np.float64)
                nn_all = np.full((len(verts),), -1, dtype=np.int64)
                targets_all = np.zeros_like(verts, dtype=np.float64)
                target_normals_all = np.zeros_like(verts, dtype=np.float64)
                if np.any(obj_xy_valid) and np.any(valid_uv):
                    obj_valid_ids = np.flatnonzero(obj_xy_valid)
                    obj_2d_tree = cKDTree(obj_xy[obj_valid_ids])
                    hand_valid_ids = np.flatnonzero(valid_uv)
                    candidates_by_hand = obj_2d_tree.query_ball_point(xy_verts[hand_valid_ids], r=target_locality_px)
                    for hand_id, local_obj_ids in zip(hand_valid_ids, candidates_by_hand):
                        if not local_obj_ids:
                            continue
                        obj_ids = obj_valid_ids[np.asarray(local_obj_ids, dtype=np.int64)]
                        delta = obj_cam[obj_ids] - verts[hand_id][None, :]
                        dist = np.linalg.norm(delta, axis=1)
                        best_pos = int(np.argmin(dist))
                        best_obj_id = int(obj_ids[best_pos])
                        nn_all[hand_id] = best_obj_id
                        surface_dist_m[hand_id] = float(dist[best_pos])
                        targets_all[hand_id] = obj_cam[best_obj_id]
                        target_normals_all[hand_id] = obj_normals_cam[best_obj_id]
            else:
                surface_dist_m, nn_all = tree.query(verts, k=1)
                targets_all = obj_cam[nn_all]
                target_normals_all = obj_normals_cam[nn_all]
            depth_delta_m = verts[:, 2] - targets_all[:, 2]
            near_surface = (
                valid_uv
                & (proximity_dist_px <= float(args.object_proximity_px))
                & (surface_dist_m <= float(args.max_current_surface_distance_m))
                & (depth_delta_m <= float(args.max_hand_behind_surface_m))
            )
            if args.candidate_mode == "object_mask":
                selected = inside_mask
            elif args.candidate_mode == "near_surface":
                selected = near_surface
            else:
                selected = inside_mask | near_surface
            selected &= np.isfinite(surface_dist_m) & (surface_dist_m <= float(args.max_current_surface_distance_m))
            selected &= depth_delta_m <= float(args.max_hand_behind_surface_m)
            ids = np.where(selected)[0]
            if len(ids) < int(args.min_contact_vertices):
                skipped.append(
                    {
                        "frame_idx": frame_idx,
                        "side": side,
                        "reason": "too_few_near_surface_contact_vertices",
                        "count": int(len(ids)),
                        "inside_object_mask_vertices": int(np.count_nonzero(inside_mask)),
                        "near_surface_vertices": int(np.count_nonzero(near_surface)),
                        "proximity_px_p10": float(np.percentile(proximity_dist_px[np.isfinite(proximity_dist_px)], 10.0)) if np.any(np.isfinite(proximity_dist_px)) else None,
                        "surface_dist_m_p10": float(np.percentile(surface_dist_m[np.isfinite(surface_dist_m)], 10.0)) if np.any(np.isfinite(surface_dist_m)) else None,
                    }
                )
                continue
            score = (surface_dist_m[ids] / max(1.0e-6, float(args.max_current_surface_distance_m))) + (
                proximity_dist_px[ids] / max(1.0, float(args.object_proximity_px))
            )
            if len(ids) > int(args.max_contact_vertices):
                order = np.argsort(score)[: int(args.max_contact_vertices)]
                ids = ids[order]
            targets = targets_all[ids]
            target_normals = target_normals_all[ids]
            weights = np.exp(-0.5 * (proximity_dist_px[ids] / max(1.0, float(args.contact_proximity_weight_px))) ** 2)
            weights *= np.exp(-0.5 * (surface_dist_m[ids] / max(1.0e-6, float(args.contact_distance_weight_m))) ** 2)
            weights = np.clip(weights, 0.10, 1.0)
            candidate_stats = {
                "candidate_mode": str(args.candidate_mode),
                "selected_vertices": int(len(ids)),
                "inside_object_mask_vertices": int(np.count_nonzero(inside_mask)),
                "near_surface_vertices": int(np.count_nonzero(near_surface)),
                "selected_proximity_px": numeric_summary(proximity_dist_px[ids].astype(float).tolist()),
                "selected_current_surface_distance_m": numeric_summary(surface_dist_m[ids].astype(float).tolist()),
                "selected_depth_delta_m": numeric_summary(depth_delta_m[ids].astype(float).tolist()),
                "selected_weight": numeric_summary(weights.astype(float).tolist()),
                "target_locality_px": float(target_locality_px),
                "localized_target_vertices": int(np.count_nonzero(nn_all >= 0)) if target_locality_px > 0.0 else None,
            }
            obs.append(
                Obs(
                    frame_idx=frame_idx,
                    side=side,
                    joints_cam=joints,
                    verts_cam=verts,
                    intr=intr_float,
                    r_c2w=r_c2w,
                    t_c2w=t_c2w,
                    contact_idx=ids,
                    contact_targets_cam=targets,
                    contact_normals_cam=target_normals,
                    contact_weights=weights,
                    candidate_stats=candidate_stats,
                )
            )
    return obs, skipped


def pack(obs: list[Obs]) -> np.ndarray:
    return np.zeros((len(obs), 7), dtype=float).reshape(-1)


def unpack(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape((-1, 7))


def residual(x: np.ndarray, obs: list[Obs], args: argparse.Namespace) -> np.ndarray:
    p = unpack(x)
    out: list[np.ndarray] = []
    centers = []
    for i, o in enumerate(obs):
        params = p[i]
        joints = transform(o.local_joints, o.center, params)
        verts = transform(o.local_verts, o.center, params)
        uv = project_camera(joints, o.intr)[0]
        out.append(((uv - o.base_uv) / float(args.sigma_reprojection_px)).reshape(-1))
        if len(o.contact_idx):
            contact = verts[o.contact_idx]
            diff = contact - o.contact_targets_cam
            if str(args.contact_residual_mode) == "point_to_plane":
                signed = np.sum(diff * o.contact_normals_cam, axis=1)
                out.append((np.sqrt(o.contact_weights) * signed / float(args.sigma_contact_m)).reshape(-1))
            else:
                weighted = np.sqrt(o.contact_weights)[:, None] * diff
                out.append((weighted / float(args.sigma_contact_m)).reshape(-1))
        out.append(params[:3] / float(args.sigma_rotation_rad))
        out.append(params[3:6] / float(args.sigma_translation_m))
        out.append(np.asarray([params[6] / float(args.sigma_log_scale)]))
        centers.append(np.median(joints, axis=0))
    by_side: dict[str, list[int]] = {}
    for i, o in enumerate(obs):
        by_side.setdefault(o.side, []).append(i)
    for indices in by_side.values():
        indices.sort(key=lambda idx: obs[idx].frame_idx)
        for a, b in zip(indices[:-1], indices[1:]):
            gap = max(1, obs[b].frame_idx - obs[a].frame_idx)
            if gap <= 3:
                out.append((centers[b] - centers[a]) / (float(args.sigma_temporal_center_m) * gap))
                out.append(np.asarray([(p[b, 6] - p[a, 6]) / (float(args.sigma_temporal_log_scale) * gap)]))
    return np.concatenate([np.ravel(v).astype(float) for v in out]) if out else np.zeros((0,), dtype=float)


def solve(obs: list[Obs], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    x0 = pack(obs)
    lo = np.tile(np.asarray([-args.max_rotation_rad, -args.max_rotation_rad, -args.max_rotation_rad, -args.max_translation_m, -args.max_translation_m, -args.max_translation_m, math.log(args.min_scale)], dtype=float), len(obs))
    hi = np.tile(np.asarray([args.max_rotation_rad, args.max_rotation_rad, args.max_rotation_rad, args.max_translation_m, args.max_translation_m, args.max_translation_m, math.log(args.max_scale)], dtype=float), len(obs))
    result = least_squares(residual, x0, args=(obs, args), bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=int(args.max_nfev), verbose=0)
    return result.x, {"success": bool(result.success), "message": str(result.message), "cost": float(result.cost), "optimality": float(result.optimality), "nfev": int(result.nfev)}


def make_rows(obs: list[Obs], params_flat: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    params = unpack(params_flat)
    rows = []
    for i, o in enumerate(obs):
        p = params[i]
        joints = transform(o.local_joints, o.center, p)
        verts = transform(o.local_verts, o.center, p)
        joints_world = camera_to_world(joints, o.r_c2w, o.t_c2w)
        verts_world = camera_to_world(verts, o.r_c2w, o.t_c2w)
        base_joints_world = camera_to_world(o.joints_cam, o.r_c2w, o.t_c2w)
        uv_after = project_camera(joints, o.intr)[0]
        uv_shift = np.linalg.norm(uv_after - o.base_uv, axis=1)
        center_shift = np.linalg.norm(np.median(joints, axis=0) - o.center)
        contact_diff_before = o.verts_cam[o.contact_idx] - o.contact_targets_cam if len(o.contact_idx) else np.zeros((0, 3), dtype=float)
        contact_diff_after = verts[o.contact_idx] - o.contact_targets_cam if len(o.contact_idx) else np.zeros((0, 3), dtype=float)
        contact_before = np.linalg.norm(contact_diff_before, axis=1) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        contact_after = np.linalg.norm(contact_diff_after, axis=1) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        normal_before = np.abs(np.sum(contact_diff_before * o.contact_normals_cam, axis=1)) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        normal_after = np.abs(np.sum(contact_diff_after * o.contact_normals_cam, axis=1)) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        tangent_before = np.sqrt(np.maximum(0.0, contact_before * contact_before - normal_before * normal_before)) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        tangent_after = np.sqrt(np.maximum(0.0, contact_after * contact_after - normal_after * normal_after)) if len(o.contact_idx) else np.zeros((0,), dtype=float)
        center_world = camera_to_world(o.center[None, :], o.r_c2w, o.t_c2w)[0]
        rotation_world = (p[:3] @ o.r_c2w.T).astype(float)
        rows.append({
            "frame_idx": int(o.frame_idx),
            "source_frame_index": int(o.frame_idx),
            "hand_side": o.side,
            "interval_id": f"{o.side}_{o.frame_idx:04d}_contact_similarity_refit",
            "temporal_mano_state": "v19_contact_similarity_metric_refit_candidate",
            "optimized_joints_world_m": joints_world.astype(float).tolist(),
            "optimized_vertices_world_sample_m": verts_world.astype(float).tolist(),
            "optimized_vertices_sample_ids": list(range(len(verts_world))),
            "optimized_translation_world_m": (np.median(joints_world, axis=0) - np.median(base_joints_world, axis=0)).astype(float).tolist(),
            "optimized_translation_camera_m": p[3:6].astype(float).tolist(),
            "optimized_rotation_vector_camera_rad": p[:3].astype(float).tolist(),
            "optimized_rotation_vector_world_rad": rotation_world.astype(float).tolist(),
            "hand_center_world_m": center_world.astype(float).tolist(),
            "optimized_similarity_scale": float(math.exp(float(p[6]))),
            "optimized_rotation_norm_rad": float(np.linalg.norm(p[:3])),
            "contact_similarity_refit": {
                "contact_vertex_count": int(len(o.contact_idx)),
                "contact_residual_mode": str(args.contact_residual_mode),
                "contact_distance_before_m": numeric_summary(contact_before.astype(float).tolist()),
                "contact_distance_after_m": numeric_summary(contact_after.astype(float).tolist()),
                "contact_normal_abs_before_m": numeric_summary(normal_before.astype(float).tolist()),
                "contact_normal_abs_after_m": numeric_summary(normal_after.astype(float).tolist()),
                "contact_tangent_before_m": numeric_summary(tangent_before.astype(float).tolist()),
                "contact_tangent_after_m": numeric_summary(tangent_after.astype(float).tolist()),
                "contact_weight": numeric_summary(o.contact_weights.astype(float).tolist()),
                "candidate_stats": o.candidate_stats,
                "center_shift_camera_m": float(center_shift),
                "scale": float(math.exp(float(p[6]))),
                "rotation_norm_rad": float(np.linalg.norm(p[:3])),
            },
            "visible_joint_shift_px": numeric_summary(uv_shift.astype(float).tolist()),
            "full_observed_surface_penetration_after_solver_m": numeric_summary(contact_after.astype(float).tolist()),
            "final_active_constraint_residual_after_solver_m": numeric_summary(contact_after.astype(float).tolist()),
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    obs, skipped = build_observations(args)
    if not obs:
        raise RuntimeError(f"no contact-similarity observations; skipped={skipped[:20]}")
    params, solve_info = solve(obs, args)
    rows = make_rows(obs, params, args)
    report = {
        "method": "v19_mano_contact_similarity_interval_refit",
        "case": str(args.case),
        "object_id": str(args.object_id),
        "claim_scope": "Bounded Sim(3) refit of current MANO camera geometry using image projection and rigid-object contact targets. Candidate measurement repair only; not accepted MANO parameters or persistent contact anchors.",
        "inputs": {"annotations": str(args.annotations), "pose_report": str(args.pose_report), "completed_mesh": str(args.completed_mesh)},
        "parameters": vars(args) | {"output_dir": str(args.output_dir), "annotations": str(args.annotations), "pose_report": str(args.pose_report), "completed_mesh": str(args.completed_mesh)},
        "solve": solve_info,
        "summary": {
            "row_count": len(rows),
            "scale": numeric_summary([float(r["optimized_similarity_scale"]) for r in rows]),
            "rotation_norm_rad": numeric_summary([float(r["optimized_rotation_norm_rad"]) for r in rows]),
            "visible_joint_shift_px_median": numeric_summary([float((r["visible_joint_shift_px"] or {}).get("median", 0.0)) for r in rows]),
            "contact_after_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_distance_after_m") or {}).get("median", 0.0)) for r in rows]),
            "contact_before_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_distance_before_m") or {}).get("median", 0.0)) for r in rows]),
            "contact_normal_abs_after_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_normal_abs_after_m") or {}).get("median", 0.0)) for r in rows]),
            "contact_normal_abs_before_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_normal_abs_before_m") or {}).get("median", 0.0)) for r in rows]),
            "contact_tangent_after_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_tangent_after_m") or {}).get("median", 0.0)) for r in rows]),
            "contact_tangent_before_median": numeric_summary([float(((r["contact_similarity_refit"] or {}).get("contact_tangent_before_m") or {}).get("median", 0.0)) for r in rows]),
        },
        "per_frame_states": rows,
        "skipped_preview": skipped[:160],
    }
    out_dir = args.output_dir / str(args.case)
    write_json(out_dir / "v18_joint_mano_interval_trajectory_state.json", report)
    write_json(out_dir / "v19_mano_contact_similarity_refit_report.json", {k: v for k, v in report.items() if k != "per_frame_states"})
    print(json.dumps({"status": "ok", "output": str(out_dir / "v18_joint_mano_interval_trajectory_state.json"), "summary": report["summary"]}, indent=2))
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
