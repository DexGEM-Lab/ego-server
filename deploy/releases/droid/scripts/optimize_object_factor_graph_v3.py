#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from align_mesh_prior_v3 import Sim3, load_observed_frame, sample_mesh_surface


MEASURED_STATUSES = {
    "measured_plan_sam",
    "measured_plan_sam_vlm_verified",
    "measured_sam_kalman",
    "measured_sam2_vlm_points",
}


@dataclass(frozen=True)
class FrameFactorData:
    frame_idx: int
    T_world_camera: np.ndarray
    observed_points: np.ndarray
    observed_tree: cKDTree
    mask: np.ndarray
    mask_distance: np.ndarray
    mask_size: tuple[int, int]
    source_size: tuple[int, int]
    camera_axis_world: np.ndarray
    contact_points: np.ndarray
    has_contact_evidence: bool
    contact_weight: float


@dataclass(frozen=True)
class FrameBuildResult:
    records: list[FrameFactorData]
    skipped: list[dict]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rodrigues(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    x, y, z = axis
    K = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def load_initial_sim3(path: Path) -> Sim3:
    report = load_json(path)
    sim = report["sim3"]
    return Sim3(
        scale=float(sim["scale"]),
        rotation=np.asarray(sim["rotation"], dtype=float),
        translation=np.asarray(sim["translation"], dtype=float),
    )


def localize_path(path_str: str, remote_root: Path | None, local_root: Path | None) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    if remote_root is None or local_root is None:
        raise FileNotFoundError(path_str)
    try:
        rel = path.relative_to(remote_root)
    except ValueError as exc:
        raise FileNotFoundError(path_str) from exc
    candidate = local_root / rel
    if not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


def resize_bool_mask(path: Path, mask_size: tuple[int, int]) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"failed to read mask image: {path}")
    width, height = mask_size
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    return raw > 0


def mask_distance_map(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3).astype(np.float32)


def summarize_array(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def project_world(points_world: np.ndarray, T_world_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_camera_world = np.linalg.inv(T_world_camera)
    homog = np.c_[points_world, np.ones(len(points_world), dtype=float)]
    camera = (T_camera_world @ homog.T).T[:, :3]
    z = camera[:, 2]
    uv = np.full((len(points_world), 2), np.nan, dtype=float)
    valid = z > 1e-5
    fx, fy, cx, cy = intrinsics.astype(float)
    uv[valid, 0] = camera[valid, 0] / z[valid] * fx + cx
    uv[valid, 1] = camera[valid, 1] / z[valid] * fy + cy
    return uv, z


def sample_rows(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def hand_surface_points(frame: dict) -> np.ndarray:
    points = []
    for hand in frame.get("hands", []):
        for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
            if key not in hand:
                continue
            arr = np.asarray(hand[key], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3 and len(arr):
                points.append(arr)
            break
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(points)


def hand_reprojection_residual_px(hand: dict) -> float | None:
    box = hand.get("bbox_xyxy")
    intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
    cam_t = np.asarray(hand.get("cam_t", [0.0, 0.0, 0.0]), dtype=float)
    if box is None or intr.shape != (4,) or cam_t.shape != (3,):
        return None
    points = []
    for key in ("joints3d_camera", "vertices_camera"):
        arr = np.asarray(hand.get(key, []), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr):
            points.append(arr + cam_t[None, :])
    if not points:
        return None
    cloud = np.vstack(points)
    if not np.all(np.isfinite(cloud)) or np.any(cloud[:, 2] <= 0.0):
        return None
    fx, fy, cx, cy = intr
    uv = np.c_[fx * cloud[:, 0] / cloud[:, 2] + cx, fy * cloud[:, 1] / cloud[:, 2] + cy]
    proj_min = uv.min(axis=0)
    proj_max = uv.max(axis=0)
    x0, y0, x1, y1 = [float(v) for v in box]
    residual = np.r_[proj_min - np.asarray([x0, y0]), proj_max - np.asarray([x1, y1])]
    return float(np.linalg.norm(residual))


def frame_contact_weight(frame: dict, args: argparse.Namespace) -> float:
    weights = []
    for hand in frame.get("hands", []):
        score = float(hand.get("detector_score", 0.0))
        score_weight = min(1.0, max(0.0, score / args.contact_score_full))
        reproj = hand_reprojection_residual_px(hand)
        if reproj is None:
            reproj_weight = args.contact_missing_reprojection_weight
        else:
            reproj_weight = float(np.exp(-0.5 * (reproj / args.contact_reprojection_sigma_px) ** 2))
        weights.append(score_weight * reproj_weight)
    if not weights:
        return 0.0
    return float(max(weights))


def contact_points_for_frame(frame: dict, mask: np.ndarray, intrinsics: np.ndarray, max_points: int, distance_px: float) -> np.ndarray:
    obj = frame.get("object", {})
    has_contact = float(obj.get("contact_ratio", 0.0)) > 0.0 or float(obj.get("min_tip_dist_px", math.inf)) <= distance_px
    if not has_contact:
        return np.zeros((0, 3), dtype=float)
    hands = hand_surface_points(frame)
    if len(hands) == 0:
        return hands
    T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
    uv, depth = project_world(hands, T, intrinsics)
    scale = np.asarray([mask.shape[1], mask.shape[0]], dtype=float) / np.asarray(obj["source_image_size"], dtype=float)
    xy = uv * scale
    valid = np.isfinite(xy).all(axis=1) & (depth > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    dist = mask_distance_map(mask)
    near = valid & (dist[y, x] <= distance_px * scale.mean())
    selected = hands[near]
    if len(selected) == 0:
        return selected
    return sample_rows(selected, max_points, int(frame["frame_idx"]) + 17)


def camera_axis_world(T_world_camera: np.ndarray) -> np.ndarray:
    axis = np.asarray(T_world_camera[:3, :3], dtype=float) @ np.asarray([0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError("invalid camera optical axis")
    return axis / norm


def active_frame_records(annotations: dict, start: int, end: int) -> list[dict]:
    out = []
    for frame in annotations["frames"]:
        idx = int(frame["frame_idx"])
        obj = frame.get("object", {})
        if start <= idx <= end and obj.get("status") in MEASURED_STATUSES and obj.get("mask_path"):
            out.append(frame)
    if not out:
        raise RuntimeError("no active measured object frames selected")
    return out


def build_frame_data(args: argparse.Namespace, frames: list[dict], intrinsics: np.ndarray) -> FrameBuildResult:
    records = []
    skipped = []
    for frame in frames:
        idx = int(frame["frame_idx"])
        obj = frame["object"]
        try:
            observed, _ = load_observed_frame(args.observed_mesh_npz, idx)
            observed = sample_rows(observed, args.max_observed_points, args.seed + idx)
            mask_size = tuple(int(x) for x in obj["mask_image_size"])
            source_size = tuple(int(x) for x in obj["source_image_size"])
            mask_path = localize_path(str(obj["mask_path"]), args.remote_output_root, args.local_output_root)
            mask = resize_bool_mask(mask_path, mask_size)
            contacts = contact_points_for_frame(frame, mask, intrinsics, args.max_contact_points, args.contact_distance_px)
            contact_weight = frame_contact_weight(frame, args)
            records.append(
                FrameFactorData(
                    frame_idx=idx,
                    T_world_camera=np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float),
                    observed_points=observed,
                    observed_tree=cKDTree(observed),
                    mask=mask,
                    mask_distance=mask_distance_map(mask),
                    mask_size=mask_size,
                    source_size=source_size,
                    camera_axis_world=camera_axis_world(np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)),
                    contact_points=contacts,
                    has_contact_evidence=len(contacts) > 0 and contact_weight >= args.min_contact_weight,
                    contact_weight=contact_weight,
                )
            )
        except Exception as exc:
            skipped.append({"frame_idx": idx, "reason": str(exc)})
    if len(records) < 2:
        raise RuntimeError(f"too few usable factor frames; skipped={skipped[:8]}")
    return FrameBuildResult(records=records, skipped=skipped)


def unpack_params(params: np.ndarray, n: int, depth_axis_enabled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 7 if depth_axis_enabled else 6
    pose = params.reshape(n, width)
    depth = pose[:, 6] if depth_axis_enabled else np.zeros(n, dtype=float)
    return pose[:, :3], pose[:, 3:6], depth


def transform_points(base_points: np.ndarray, pivot: np.ndarray, rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    R = rodrigues(rotvec)
    return (base_points - pivot) @ R.T + pivot + trans


def silhouette_residual(
    points_world: np.ndarray,
    frame: FrameFactorData,
    intrinsics: np.ndarray,
    sigma_px: float,
    max_px: float,
) -> np.ndarray:
    uv, depth = project_world(points_world, frame.T_world_camera, intrinsics)
    scale = np.asarray(frame.mask_size, dtype=float) / np.asarray(frame.source_size, dtype=float)
    xy = uv * scale
    valid = np.isfinite(xy).all(axis=1) & (depth > 0.0)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, frame.mask.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, frame.mask.shape[0] - 1)
    dist = frame.mask_distance[y, x].astype(float)
    dist[~valid] = max_px
    outside_image = valid & (
        (xy[:, 0] < 0.0)
        | (xy[:, 0] >= frame.mask.shape[1])
        | (xy[:, 1] < 0.0)
        | (xy[:, 1] >= frame.mask.shape[0])
    )
    dist[outside_image] = max_px
    return np.clip(dist, 0.0, max_px) / sigma_px


def residual_vector(
    params: np.ndarray,
    frames: list[FrameFactorData],
    base_surface: np.ndarray,
    base_silhouette: np.ndarray,
    pivot: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rotvecs, translations, depth_offsets = unpack_params(params, len(frames), args.enable_depth_axis)
    residuals = []
    for i, frame in enumerate(frames):
        translation = translations[i] + depth_offsets[i] * frame.camera_axis_world
        surface = transform_points(base_surface, pivot, rotvecs[i], translation)
        tree = cKDTree(surface)
        d_obs, _ = tree.query(frame.observed_points, k=1)
        d_prior, _ = frame.observed_tree.query(surface, k=1)
        residuals.append(np.clip(d_obs, 0.0, args.max_surface_residual_m) / args.sigma_observed_m)
        residuals.append(np.clip(d_prior, 0.0, args.max_surface_residual_m) / args.sigma_prior_surface_m)
        silhouette_points = transform_points(base_silhouette, pivot, rotvecs[i], translation)
        residuals.append(silhouette_residual(silhouette_points, frame, intrinsics, args.sigma_silhouette_px, args.max_silhouette_px))
        if frame.has_contact_evidence:
            d_contact, _ = tree.query(frame.contact_points, k=1)
            contact_sigma = args.sigma_contact_m / max(args.min_contact_weight, frame.contact_weight)
            residuals.append(np.clip(d_contact, 0.0, args.max_contact_residual_m) / contact_sigma)
            if args.enable_depth_axis:
                residuals.append(np.asarray([depth_offsets[i] / args.sigma_contact_depth_offset_m], dtype=float))
        elif args.enable_depth_axis:
            residuals.append(np.asarray([depth_offsets[i] / args.sigma_noncontact_depth_offset_m], dtype=float))
    for i in range(1, len(frames)):
        residuals.append((translations[i] - translations[i - 1]) / args.sigma_translation_step_m)
        residuals.append((rotvecs[i] - rotvecs[i - 1]) / args.sigma_rotation_step_rad)
        if args.enable_depth_axis:
            residuals.append(np.asarray([(depth_offsets[i] - depth_offsets[i - 1]) / args.sigma_depth_step_m], dtype=float))
    for i in range(1, len(frames) - 1):
        residuals.append((translations[i + 1] - 2.0 * translations[i] + translations[i - 1]) / args.sigma_translation_accel_m)
        residuals.append((rotvecs[i + 1] - 2.0 * rotvecs[i] + rotvecs[i - 1]) / args.sigma_rotation_accel_rad)
        if args.enable_depth_axis:
            residuals.append(np.asarray([(depth_offsets[i + 1] - 2.0 * depth_offsets[i] + depth_offsets[i - 1]) / args.sigma_depth_accel_m], dtype=float))
    anchor = int(np.argmin([abs(frame.frame_idx - args.anchor_frame) for frame in frames]))
    residuals.append(translations[anchor] / args.sigma_anchor_translation_m)
    residuals.append(rotvecs[anchor] / args.sigma_anchor_rotation_rad)
    if args.enable_depth_axis:
        residuals.append(np.asarray([depth_offsets[anchor] / args.sigma_anchor_depth_offset_m], dtype=float))
    return np.concatenate([r.reshape(-1) for r in residuals])


def add_block(entries: list[tuple[int, int]], rows: range, cols: range) -> None:
    for r in rows:
        for c in cols:
            entries.append((r, c))


def residual_sparsity(frames: list[FrameFactorData], base_surface: np.ndarray, base_silhouette: np.ndarray, args: argparse.Namespace) -> sparse.csr_matrix:
    n = len(frames)
    width = 7 if args.enable_depth_axis else 6
    total_cols = n * width
    entries: list[tuple[int, int]] = []
    row = 0
    for i, frame in enumerate(frames):
        cols = range(i * width, (i + 1) * width)
        for count in (len(frame.observed_points), len(base_surface), len(base_silhouette)):
            rows = range(row, row + int(count))
            add_block(entries, rows, cols)
            row += int(count)
        if frame.has_contact_evidence:
            rows = range(row, row + len(frame.contact_points))
            add_block(entries, rows, cols)
            row += len(frame.contact_points)
            if args.enable_depth_axis:
                add_block(entries, range(row, row + 1), cols)
                row += 1
        elif args.enable_depth_axis:
            add_block(entries, range(row, row + 1), cols)
            row += 1
    for i in range(1, n):
        cols = range((i - 1) * width, (i + 1) * width)
        add_block(entries, range(row, row + 3), cols)
        row += 3
        add_block(entries, range(row, row + 3), cols)
        row += 3
        if args.enable_depth_axis:
            add_block(entries, range(row, row + 1), cols)
            row += 1
    for i in range(1, n - 1):
        cols = range((i - 1) * width, (i + 2) * width)
        add_block(entries, range(row, row + 3), cols)
        row += 3
        add_block(entries, range(row, row + 3), cols)
        row += 3
        if args.enable_depth_axis:
            add_block(entries, range(row, row + 1), cols)
            row += 1
    anchor = int(np.argmin([abs(frame.frame_idx - args.anchor_frame) for frame in frames]))
    cols = range(anchor * width, (anchor + 1) * width)
    add_block(entries, range(row, row + 3), cols)
    row += 3
    add_block(entries, range(row, row + 3), cols)
    row += 3
    if args.enable_depth_axis:
        add_block(entries, range(row, row + 1), cols)
        row += 1
    if not entries:
        raise RuntimeError("empty sparsity pattern")
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, total_cols))


def frame_metrics(
    params: np.ndarray,
    frames: list[FrameFactorData],
    base_surface: np.ndarray,
    base_silhouette: np.ndarray,
    pivot: np.ndarray,
    intrinsics: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict]:
    rotvecs, translations, depth_offsets = unpack_params(params, len(frames), args.enable_depth_axis)
    metrics = {}
    for i, frame in enumerate(frames):
        translation = translations[i] + depth_offsets[i] * frame.camera_axis_world
        surface = transform_points(base_surface, pivot, rotvecs[i], translation)
        tree = cKDTree(surface)
        d_obs, _ = tree.query(frame.observed_points, k=1)
        d_prior, _ = frame.observed_tree.query(surface, k=1)
        silhouette_points = transform_points(base_silhouette, pivot, rotvecs[i], translation)
        sil = silhouette_residual(silhouette_points, frame, intrinsics, 1.0, args.max_silhouette_px)
        row = {
            "observed_to_prior_median_m": float(np.median(d_obs)),
            "observed_to_prior_p95_m": float(np.percentile(d_obs, 95.0)),
            "prior_to_observed_median_m": float(np.median(d_prior)),
            "prior_to_observed_p95_m": float(np.percentile(d_prior, 95.0)),
            "silhouette_outside_median_px": float(np.median(sil)),
            "silhouette_outside_p95_px": float(np.percentile(sil, 95.0)),
            "translation_delta_m": translations[i].astype(float).tolist(),
            "rotation_delta_rad": rotvecs[i].astype(float).tolist(),
            "depth_axis_offset_m": float(depth_offsets[i]),
            "contact_points": int(len(frame.contact_points)),
            "contact_weight": float(frame.contact_weight),
        }
        if frame.has_contact_evidence:
            d_contact, _ = tree.query(frame.contact_points, k=1)
            row["contact_median_m"] = float(np.median(d_contact))
            row["contact_p95_m"] = float(np.percentile(d_contact, 95.0))
            row["contact_min_m"] = float(np.min(d_contact))
        else:
            row["contact_median_m"] = None
            row["contact_p95_m"] = None
            row["contact_min_m"] = None
        metrics[str(frame.frame_idx)] = row
    return metrics


def summarize_metrics(metrics: dict[str, dict]) -> dict:
    def values(key: str) -> np.ndarray:
        vals = [row[key] for row in metrics.values() if row.get(key) is not None]
        return np.asarray(vals, dtype=float)

    summary = {}
    for key in (
        "observed_to_prior_median_m",
        "prior_to_observed_median_m",
        "silhouette_outside_median_px",
        "contact_median_m",
        "contact_p95_m",
        "depth_axis_offset_m",
    ):
        arr = values(key)
        if arr.size:
            summary[f"{key}_median"] = float(np.median(arr))
            summary[f"{key}_p95"] = float(np.percentile(arr, 95.0))
            summary[f"{key}_max"] = float(np.max(arr))
        else:
            summary[f"{key}_median"] = None
            summary[f"{key}_p95"] = None
            summary[f"{key}_max"] = None
    return summary


def save_mesh_archive(path: Path, frames: list[FrameFactorData], vertices_per_frame: list[np.ndarray], faces: np.ndarray) -> None:
    vertex_offsets = [0]
    face_offsets = [0]
    faces_per_frame = []
    for vertices in vertices_per_frame:
        vertex_offsets.append(vertex_offsets[-1] + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
        faces_per_frame.append(faces.astype(np.int32))
    np.savez_compressed(
        path,
        frame_idx=np.asarray([frame.frame_idx for frame in frames], dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_per_frame).astype(np.float32),
        faces=np.vstack(faces_per_frame).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    annotations = load_json(args.annotations)
    selected_frames = active_frame_records(annotations, args.frame_start, args.frame_end)
    droid = np.load(args.droid_npz)
    intrinsics = np.asarray(droid["intrinsics_source"], dtype=float)
    build_result = build_frame_data(args, selected_frames, intrinsics)
    frames = build_result.records
    skipped_frames = build_result.skipped
    skipped_fraction = len(skipped_frames) / max(1, len(selected_frames))
    mesh = trimesh.load(args.mesh_prior, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh prior: {args.mesh_prior}")
    initial = load_initial_sim3(args.initial_alignment_qc)
    local_surface = sample_mesh_surface(mesh, args.max_prior_surface_points, args.seed)
    local_silhouette = sample_mesh_surface(mesh, args.max_silhouette_points, args.seed + 101)
    base_vertices = initial.apply(np.asarray(mesh.vertices, dtype=float))
    base_surface = initial.apply(local_surface)
    base_silhouette = initial.apply(local_silhouette)
    pivot = base_surface.mean(axis=0)
    width = 7 if args.enable_depth_axis else 6
    x0 = np.zeros(len(frames) * width, dtype=float)
    before_vec = residual_vector(x0, frames, base_surface, base_silhouette, pivot, intrinsics, args)
    jac_pattern = residual_sparsity(frames, base_surface, base_silhouette, args)
    if jac_pattern.shape != (len(before_vec), len(x0)):
        raise RuntimeError(f"jacobian sparsity shape {jac_pattern.shape} disagrees with residual/vector {(len(before_vec), len(x0))}")
    result = least_squares(
        lambda x: residual_vector(x, frames, base_surface, base_silhouette, pivot, intrinsics, args),
        x0,
        jac_sparsity=jac_pattern,
        max_nfev=args.max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    after_vec = residual_vector(result.x, frames, base_surface, base_silhouette, pivot, intrinsics, args)
    before_metrics = frame_metrics(x0, frames, base_surface, base_silhouette, pivot, intrinsics, args)
    after_metrics = frame_metrics(result.x, frames, base_surface, base_silhouette, pivot, intrinsics, args)
    rotvecs, translations, depth_offsets = unpack_params(result.x, len(frames), args.enable_depth_axis)
    vertices_per_frame = [
        transform_points(base_vertices, pivot, rotvecs[i], translations[i] + depth_offsets[i] * frames[i].camera_axis_world).astype(np.float32)
        for i in range(len(frames))
    ]
    mesh_archive = args.output_dir / "factor_graph_object_meshes.npz"
    save_mesh_archive(mesh_archive, frames, vertices_per_frame, np.asarray(mesh.faces, dtype=np.int32))
    before_summary = summarize_metrics(before_metrics)
    after_summary = summarize_metrics(after_metrics)
    contact_before = before_summary.get("contact_median_m_median")
    contact_after = after_summary.get("contact_median_m_median")
    surface_before = before_summary.get("observed_to_prior_median_m_median")
    surface_after = after_summary.get("observed_to_prior_median_m_median")
    contact_improved = (
        contact_before is not None
        and contact_after is not None
        and float(contact_after) < float(contact_before)
    )
    surface_improved = (
        surface_before is not None
        and surface_after is not None
        and float(surface_after) < float(surface_before)
    )
    status = "diagnostic_contact_not_solved"
    if skipped_fraction > args.max_skipped_fraction:
        status = "diagnostic_failed_too_many_skipped_frames"
    elif result.success and contact_improved and surface_improved:
        status = "diagnostic_surface_and_contact_improved"
    elif result.success and surface_improved:
        status = "diagnostic_surface_improved_contact_not_solved"
    elif not result.success and surface_improved:
        status = "diagnostic_surface_improved_optimizer_incomplete_contact_not_solved"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "annotations": str(args.annotations),
        "mesh_prior": str(args.mesh_prior),
        "initial_alignment_qc": str(args.initial_alignment_qc),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "used_frames": [frame.frame_idx for frame in frames],
        "candidate_frames": [int(frame["frame_idx"]) for frame in selected_frames],
        "skipped_frames": skipped_frames,
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_fraction": float(skipped_fraction),
        "max_skipped_fraction": float(args.max_skipped_fraction),
        "variables": int(len(result.x)),
        "max_nfev": int(args.max_nfev),
        "depth_axis_enabled": bool(args.enable_depth_axis),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before_vec * before_vec))),
        "residual_rms_after": float(np.sqrt(np.mean(after_vec * after_vec))),
        "before_summary": before_summary,
        "after_summary": after_summary,
        "contact_weight_summary": summarize_array(np.asarray([frame.contact_weight for frame in frames], dtype=float)),
        "contact_improved": bool(contact_improved),
        "surface_improved": bool(surface_improved),
        "frame_metrics_before": before_metrics,
        "frame_metrics_after": after_metrics,
        "mesh_archive": str(mesh_archive),
        "signed_penetration_supported": False,
        "penetration_note": "The TripoSR prior is not watertight, so this prototype reports contact proximity only. Signed penetration needs a watertight mesh or robust signed-distance field.",
        "elapsed_s": time.time() - started,
    }
    qc_path = args.output_dir / "qc_object_factor_graph_v3.json"
    qc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--droid-npz", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--initial-alignment-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path)
    parser.add_argument("--local-output-root", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, default=858)
    parser.add_argument("--max-observed-points", type=int, default=300)
    parser.add_argument("--max-prior-surface-points", type=int, default=450)
    parser.add_argument("--max-silhouette-points", type=int, default=360)
    parser.add_argument("--max-contact-points", type=int, default=120)
    parser.add_argument("--contact-distance-px", type=float, default=18.0)
    parser.add_argument("--sigma-observed-m", type=float, default=0.040)
    parser.add_argument("--sigma-prior-surface-m", type=float, default=0.055)
    parser.add_argument("--sigma-contact-m", type=float, default=0.025)
    parser.add_argument("--contact-score-full", type=float, default=0.80)
    parser.add_argument("--contact-reprojection-sigma-px", type=float, default=120.0)
    parser.add_argument("--contact-missing-reprojection-weight", type=float, default=0.25)
    parser.add_argument("--min-contact-weight", type=float, default=0.15)
    parser.add_argument("--sigma-silhouette-px", type=float, default=5.0)
    parser.add_argument("--sigma-translation-step-m", type=float, default=0.045)
    parser.add_argument("--sigma-rotation-step-rad", type=float, default=0.28)
    parser.add_argument("--sigma-translation-accel-m", type=float, default=0.025)
    parser.add_argument("--sigma-rotation-accel-rad", type=float, default=0.18)
    parser.add_argument("--sigma-anchor-translation-m", type=float, default=0.035)
    parser.add_argument("--sigma-anchor-rotation-rad", type=float, default=0.18)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.20)
    parser.add_argument("--max-contact-residual-m", type=float, default=0.20)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-skipped-fraction", type=float, default=0.05)
    parser.add_argument("--enable-depth-axis", action="store_true")
    parser.add_argument("--sigma-contact-depth-offset-m", type=float, default=0.75)
    parser.add_argument("--sigma-noncontact-depth-offset-m", type=float, default=0.08)
    parser.add_argument("--sigma-depth-step-m", type=float, default=0.10)
    parser.add_argument("--sigma-depth-accel-m", type=float, default=0.06)
    parser.add_argument("--sigma-anchor-depth-offset-m", type=float, default=0.10)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
