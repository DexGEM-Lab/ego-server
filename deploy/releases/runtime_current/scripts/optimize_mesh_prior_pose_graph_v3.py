#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    T_world_camera: np.ndarray
    K: np.ndarray
    observed_points_camera: np.ndarray
    mask: np.ndarray
    mask_distance: np.ndarray
    depth_m: np.ndarray


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx):
        v0, v1 = int(vertex_offsets[i]), int(vertex_offsets[i + 1])
        f0, f1 = int(face_offsets[i]), int(face_offsets[i + 1])
        frame_vertices = vertices[v0:v1]
        frame_faces = faces[f0:f1]
        if len(frame_vertices) == 0 or len(frame_faces) == 0:
            raise RuntimeError(f"empty observed mesh frame {idx}")
        out[int(idx)] = (frame_vertices, frame_faces)
    return out


def annotation_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(entry["frame_idx"]): entry for entry in frames}


def load_intrinsics(dataset: Path) -> np.ndarray:
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"invalid intrinsics matrix: {dataset / 'cam_K.txt'}")
    return K


def annotation_intrinsics(annotation: dict) -> np.ndarray:
    values = annotation.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", [])
    intrinsics = np.asarray(values, dtype=np.float64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise RuntimeError(f"annotation frame {annotation.get('frame_idx')} has invalid VGGT intrinsics")
    fx, fy, cx, cy = intrinsics.tolist()
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid triangle mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int32), process=False)


def sample_rows(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RuntimeError("cannot sample an empty point set")
    if len(points) <= int(count):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(count), replace=False)]


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tri = vertices[faces]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0.0:
        raise RuntimeError("mesh face areas are invalid")
    face_ids = rng.choice(len(faces), size=int(count), replace=True, p=areas / areas.sum())
    chosen = tri[face_ids]
    u = rng.random(int(count))
    v = rng.random(int(count))
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])


def anchor_visible_patch_surface(prior_surface: np.ndarray, anchor_points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    tree = cKDTree(np.asarray(anchor_points, dtype=np.float64))
    distances = tree.query(np.asarray(prior_surface, dtype=np.float64), k=1)[0]
    patch = prior_surface[distances <= float(args.anchor_patch_max_distance_m)]
    if len(patch) < int(args.min_anchor_patch_points):
        raise RuntimeError(
            f"anchor visible patch has only {len(patch)} points within "
            f"{float(args.anchor_patch_max_distance_m):.4f}m"
        )
    if len(patch) > int(args.max_anchor_patch_points):
        patch = sample_rows(patch, int(args.max_anchor_patch_points), int(args.seed) + 1100)
    return patch


def mask_distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)


def load_frames(args: argparse.Namespace) -> list[FrameData]:
    observed_archive = load_archive(args.observed_mesh_npz)
    annotations = annotation_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    dataset_K = load_intrinsics(args.dataset)
    frames: list[FrameData] = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in observed_archive:
            continue
        if frame_idx not in annotations:
            raise RuntimeError(f"annotations missing frame {frame_idx}")
        if frame_idx not in manifest:
            raise RuntimeError(f"manifest missing frame {frame_idx}")
        mask = cv2.imread(str(manifest[frame_idx]["mask"]), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(manifest[frame_idx]["depth"]), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise RuntimeError(f"failed to read mask/depth for frame {frame_idx}")
        if mask.shape != depth.shape:
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}")
        T_world_camera = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
            raise RuntimeError(f"invalid T_world_camera_metric for frame {frame_idx}")
        if args.intrinsics_source == "annotation-vggt":
            K = annotation_intrinsics(annotations[frame_idx])
        elif args.intrinsics_source == "dataset":
            K = dataset_K
        else:
            raise RuntimeError(f"unsupported intrinsics source {args.intrinsics_source}")
        observed, _ = observed_archive[frame_idx]
        frames.append(
            FrameData(
                frame_idx=frame_idx,
                T_world_camera=T_world_camera,
                K=K,
                observed_points_camera=sample_rows(observed, int(args.max_observed_points), int(args.seed) + frame_idx),
                mask=mask > 0,
                mask_distance=mask_distance(mask > 0),
                depth_m=depth.astype(np.float64) / 1000.0,
            )
        )
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames selected")
    return frames


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    axes = vh.astype(np.float64)
    if np.linalg.det(axes) < 0.0:
        axes[-1] *= -1.0
    return center, axes


def transform_camera(points: np.ndarray, pivot: np.ndarray, rotvec: np.ndarray, translation: np.ndarray, log_scale: float) -> np.ndarray:
    scale = float(np.exp(log_scale))
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return scale * ((points - pivot) @ R.T) + pivot + translation


def pca_initial_pose(anchor_points: np.ndarray, current_points: np.ndarray, pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor_center, anchor_axes = pca_axes(anchor_points)
    current_center, current_axes = pca_axes(current_points)
    candidates = []
    anchor_sample = sample_rows(anchor_points, min(1200, len(anchor_points)), 101)
    current_sample = sample_rows(current_points, min(1200, len(current_points)), 103)
    current_tree = cKDTree(current_sample)
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        sign = np.diag(np.asarray(signs, dtype=np.float64))
        R = current_axes.T @ sign @ anchor_axes
        if np.linalg.det(R) < 0.0:
            continue
        translation = current_center - anchor_center @ R.T + pivot @ R.T - pivot
        transformed = (anchor_sample - pivot) @ R.T + pivot + translation
        distances, _ = current_tree.query(transformed, k=1)
        candidates.append((float(np.median(distances)), R, translation))
    if not candidates:
        raise RuntimeError("no valid PCA pose initialization candidates")
    _, rotation, translation = min(candidates, key=lambda item: item[0])
    return Rotation.from_matrix(rotation).as_rotvec(), translation.astype(np.float64)


def initial_params(frames: list[FrameData], anchor_frame: int, anchor_points: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    pose = np.zeros((len(frames), 6), dtype=np.float64)
    anchor_index = next((i for i, frame in enumerate(frames) if frame.frame_idx == int(anchor_frame)), None)
    if anchor_index is None:
        raise RuntimeError(f"anchor frame {anchor_frame} absent from selected frames")
    for i, frame in enumerate(frames):
        if i == anchor_index:
            continue
        rotvec, translation = pca_initial_pose(anchor_points, frame.observed_points_camera, pivot)
        pose[i, :3] = rotvec
        pose[i, 3:6] = translation
    return np.r_[pose.reshape(-1), 0.0]


def unpack(params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    pose = params[:-1].reshape(n, 6)
    return pose[:, :3], pose[:, 3:6], float(params[-1])


def project(points_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = z > 1e-5
    uv[valid, 0] = K[0, 0] * points_camera[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points_camera[valid, 1] / z[valid] + K[1, 2]
    return uv, z, valid


def projection_terms(points_camera: np.ndarray, frame: FrameData, K: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv, z, valid_z = project(points_camera, K)
    height, width = frame.mask.shape
    finite_uv = np.isfinite(uv).all(axis=1)
    in_image = valid_z & finite_uv & (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
    x = np.clip(np.rint(np.nan_to_num(uv[:, 0], nan=0.0)).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(np.nan_to_num(uv[:, 1], nan=0.0)).astype(np.int32), 0, height - 1)

    outside = frame.mask_distance[y, x].astype(np.float64)
    outside[~in_image] = float(args.max_silhouette_px)
    outside = np.clip(outside, 0.0, float(args.max_silhouette_px)) / float(args.sigma_silhouette_px)

    measured = frame.depth_m[y, x]
    valid_depth = in_image & frame.mask[y, x] & np.isfinite(measured) & (measured > float(args.min_depth_m))
    front_error = np.zeros(len(points_camera), dtype=np.float64)
    raw_front_error = measured - z - float(args.depth_front_tolerance_m)
    front_error[valid_depth] = np.maximum(0.0, raw_front_error[valid_depth])
    front_error = np.clip(front_error, 0.0, float(args.max_front_depth_residual_m)) / float(args.sigma_front_depth_m)

    inside = np.zeros(len(points_camera), dtype=np.float64)
    inside[in_image & frame.mask[y, x]] = 1.0
    return outside, front_error, inside


def world_points(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def rotation_world(rotvec: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return np.asarray(T_world_camera[:3, :3], dtype=np.float64) @ Rotation.from_rotvec(rotvec).as_matrix()


def rotation_log_vector(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def residual_vector(
    params: np.ndarray,
    frames: list[FrameData],
    prior_surface: np.ndarray,
    prior_projection: np.ndarray,
    anchor_points: np.ndarray,
    pivot: np.ndarray,
    anchor_index: int,
    args: argparse.Namespace,
) -> np.ndarray:
    rotvecs, translations, log_scale = unpack(params, len(frames))
    residuals = []
    transformed_centers = []
    transformed_rotvecs_camera = []
    for i, frame in enumerate(frames):
        surface = transform_camera(prior_surface, pivot, rotvecs[i], translations[i], log_scale)
        tree = cKDTree(surface)
        d_observed, _ = tree.query(frame.observed_points_camera, k=1)
        residuals.append(np.clip(d_observed, 0.0, float(args.max_surface_residual_m)) / float(args.sigma_observed_m))

        projection_points = transform_camera(prior_projection, pivot, rotvecs[i], translations[i], log_scale)
        outside, front_depth, _ = projection_terms(projection_points, frame, frame.K, args)
        residuals.append(outside)
        residuals.append(front_depth)

        center_camera = transform_camera(pivot[None, :], pivot, rotvecs[i], translations[i], log_scale)[0]
        transformed_centers.append(world_points(center_camera[None, :], frame.T_world_camera)[0])
        transformed_rotvecs_camera.append(rotvecs[i])

    centers = np.asarray(transformed_centers, dtype=np.float64)
    for i in range(1, len(frames)):
        dt = max(1, frames[i].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append((centers[i] - centers[i - 1]) / (float(args.sigma_world_velocity_m_s) * dt))
        residuals.append((transformed_rotvecs_camera[i] - transformed_rotvecs_camera[i - 1]) / (float(args.sigma_camera_rotation_step_rad_s) * dt))
    for i in range(1, len(frames) - 1):
        dt = max(1, frames[i + 1].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append((centers[i + 1] - 2.0 * centers[i] + centers[i - 1]) / (float(args.sigma_world_accel_m_s2) * dt * dt))
        residuals.append(
            (transformed_rotvecs_camera[i + 1] - 2.0 * transformed_rotvecs_camera[i] + transformed_rotvecs_camera[i - 1])
            / (float(args.sigma_camera_rotation_accel_rad_s2) * dt * dt)
        )

    residuals.append(translations[anchor_index] / float(args.sigma_anchor_translation_m))
    residuals.append(rotvecs[anchor_index] / float(args.sigma_anchor_rotation_rad))
    residuals.append(np.asarray([log_scale / float(args.sigma_log_scale)], dtype=np.float64))
    anchor_fit = transform_camera(anchor_points, pivot, rotvecs[anchor_index], translations[anchor_index], log_scale)
    residuals.append((anchor_fit - anchor_points).reshape(-1) / float(args.sigma_anchor_surface_lock_m))
    return np.concatenate([part.reshape(-1) for part in residuals])


def residual_sparsity(frames: list[FrameData], prior_surface: np.ndarray, prior_projection: np.ndarray, anchor_points: np.ndarray, anchor_index: int) -> sparse.csr_matrix:
    n = len(frames)
    cols_total = n * 6 + 1
    entries: list[tuple[int, int]] = []
    row = 0
    for i, frame in enumerate(frames):
        local_cols = list(range(i * 6, (i + 1) * 6)) + [cols_total - 1]
        for count in (len(frame.observed_points_camera), len(prior_projection), len(prior_projection)):
            for r in range(row, row + int(count)):
                for c in local_cols:
                    entries.append((r, c))
            row += int(count)
    for i in range(1, n):
        local_cols = list(range((i - 1) * 6, (i + 1) * 6)) + [cols_total - 1]
        for count in (3, 3):
            for r in range(row, row + count):
                for c in local_cols:
                    entries.append((r, c))
            row += count
    for i in range(1, n - 1):
        local_cols = list(range((i - 1) * 6, (i + 2) * 6)) + [cols_total - 1]
        for count in (3, 3):
            for r in range(row, row + count):
                for c in local_cols:
                    entries.append((r, c))
            row += count
    local_cols = list(range(anchor_index * 6, (anchor_index + 1) * 6)) + [cols_total - 1]
    for count in (3, 3, 1, len(anchor_points) * 3):
        for r in range(row, row + int(count)):
            for c in local_cols:
                entries.append((r, c))
        row += int(count)
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols_total))


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


def frame_metrics(
    params: np.ndarray,
    frames: list[FrameData],
    prior_surface: np.ndarray,
    prior_projection: np.ndarray,
    pivot: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict]:
    rotvecs, translations, log_scale = unpack(params, len(frames))
    rows = {}
    centers_world = []
    rotations_world = []
    for i, frame in enumerate(frames):
        surface = transform_camera(prior_surface, pivot, rotvecs[i], translations[i], log_scale)
        tree = cKDTree(surface)
        d_observed, _ = tree.query(frame.observed_points_camera, k=1)
        projection_points = transform_camera(prior_projection, pivot, rotvecs[i], translations[i], log_scale)
        outside, front_depth, inside = projection_terms(projection_points, frame, frame.K, argparse.Namespace(
            max_silhouette_px=float(args.max_silhouette_px),
            sigma_silhouette_px=1.0,
            min_depth_m=float(args.min_depth_m),
            depth_front_tolerance_m=0.0,
            max_front_depth_residual_m=10.0,
            sigma_front_depth_m=1.0,
        ))
        center_camera = transform_camera(pivot[None, :], pivot, rotvecs[i], translations[i], log_scale)[0]
        center_world = world_points(center_camera[None, :], frame.T_world_camera)[0]
        R_world = rotation_world(rotvecs[i], frame.T_world_camera)
        centers_world.append(center_world)
        rotations_world.append(R_world)
        rows[str(frame.frame_idx)] = {
            "observed_to_prior_median_m": float(np.median(d_observed)),
            "observed_to_prior_p95_m": float(np.percentile(d_observed, 95.0)),
            "silhouette_outside_median_px": float(np.median(outside)),
            "silhouette_outside_p95_px": float(np.percentile(outside, 95.0)),
            "front_depth_violation_median_m": float(np.median(front_depth)),
            "front_depth_violation_p95_m": float(np.percentile(front_depth, 95.0)),
            "projection_inside_mask_fraction": float(np.mean(inside)),
            "center_world_m": center_world.astype(float).tolist(),
            "translation_camera_delta_m": translations[i].astype(float).tolist(),
            "rotation_delta_rad": rotvecs[i].astype(float).tolist(),
        }
    speeds = []
    angle_speeds = []
    for prev, cur, prev_rot, cur_rot, prev_frame, cur_frame in zip(
        centers_world[:-1],
        centers_world[1:],
        rotations_world[:-1],
        rotations_world[1:],
        frames[:-1],
        frames[1:],
    ):
        dt = max(1, cur_frame.frame_idx - prev_frame.frame_idx) / float(args.fps)
        speeds.append(float(np.linalg.norm(cur - prev) / dt))
        angle_speeds.append(float(np.linalg.norm(rotation_log_vector(prev_rot.T @ cur_rot)) / dt))
    for row, speed, angle_speed in zip(list(rows.values())[1:], speeds, angle_speeds):
        row["world_center_speed_m_s_from_prev"] = speed
        row["world_angular_speed_rad_s_from_prev"] = angle_speed
    return rows


def choose_surface_observation(
    full_surface: np.ndarray,
    anchor_points: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    if args.surface_match_mode == "whole_mesh":
        return full_surface, {
            "surface_match_mode": "whole_mesh",
            "surface_points": int(len(full_surface)),
        }
    if args.surface_match_mode == "anchor_visible_patch":
        patch = anchor_visible_patch_surface(full_surface, anchor_points, args)
        return patch, {
            "surface_match_mode": "anchor_visible_patch",
            "surface_points": int(len(patch)),
            "full_surface_points": int(len(full_surface)),
            "anchor_patch_max_distance_m": float(args.anchor_patch_max_distance_m),
        }
    raise RuntimeError(f"unsupported surface_match_mode: {args.surface_match_mode}")


def save_world_archive(
    path: Path,
    frames: list[FrameData],
    mesh_vertices_camera_anchor: np.ndarray,
    mesh_faces: np.ndarray,
    pivot: np.ndarray,
    params: np.ndarray,
) -> None:
    rotvecs, translations, log_scale = unpack(params, len(frames))
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for i, frame in enumerate(frames):
        camera_vertices = transform_camera(mesh_vertices_camera_anchor, pivot, rotvecs[i], translations[i], log_scale)
        world_vertices = world_points(camera_vertices, frame.T_world_camera)
        vertices_all.append(world_vertices.astype(np.float32))
        faces_all.append(mesh_faces.astype(np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(world_vertices))
        face_offsets.append(face_offsets[-1] + len(mesh_faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray([frame.frame_idx for frame in frames], dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def summary_from_rows(rows: dict[str, dict]) -> dict:
    keys = (
        "observed_to_prior_median_m",
        "observed_to_prior_p95_m",
        "silhouette_outside_p95_px",
        "front_depth_violation_p95_m",
        "projection_inside_mask_fraction",
        "world_center_speed_m_s_from_prev",
        "world_angular_speed_rad_s_from_prev",
    )
    return {key: summarize([row[key] for row in rows.values() if key in row]) for key in keys}


def run(args: argparse.Namespace) -> dict:
    frames = load_frames(args)
    anchor_index = next((i for i, frame in enumerate(frames) if frame.frame_idx == int(args.anchor_frame)), None)
    if anchor_index is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from selected frames")
    mesh = load_mesh(args.mesh_prior_camera)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
    pivot = np.median(mesh_vertices, axis=0)
    full_prior_surface = sample_mesh_surface(mesh, int(args.max_prior_surface_points), int(args.seed) + 700)
    prior_projection = sample_mesh_surface(mesh, int(args.max_projection_points), int(args.seed) + 900)
    anchor_points = frames[anchor_index].observed_points_camera
    prior_surface, surface_report = choose_surface_observation(full_prior_surface, anchor_points, args)
    x0 = initial_params(frames, int(args.anchor_frame), anchor_points, pivot)
    before_vec = residual_vector(x0, frames, prior_surface, prior_projection, anchor_points, pivot, anchor_index, args)
    pattern = residual_sparsity(frames, prior_surface, prior_projection, anchor_points, anchor_index)
    if pattern.shape != (len(before_vec), len(x0)):
        raise RuntimeError(f"sparsity shape {pattern.shape} does not match residual/vector {(len(before_vec), len(x0))}")
    result = least_squares(
        lambda x: residual_vector(x, frames, prior_surface, prior_projection, anchor_points, pivot, anchor_index, args),
        x0,
        jac_sparsity=pattern,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after_vec = residual_vector(result.x, frames, prior_surface, prior_projection, anchor_points, pivot, anchor_index, args)
    before_rows = frame_metrics(x0, frames, prior_surface, prior_projection, pivot, args)
    after_rows = frame_metrics(result.x, frames, prior_surface, prior_projection, pivot, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "mesh_prior_pose_graph_object_meshes_world.npz"
    save_world_archive(archive_path, frames, mesh_vertices, mesh_faces, pivot, result.x)
    report = {
        "status": "ok" if result.success else "optimizer_incomplete",
        "annotation_ready": False,
        "method": "mesh_prior_camera_pose_graph_v3",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "intrinsics_source": str(args.intrinsics_source),
        "anchor_frame": int(args.anchor_frame),
        "used_frames": [int(frame.frame_idx) for frame in frames],
        "mesh_archive_world": str(archive_path),
        "variables": int(len(result.x)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "log_scale": float(result.x[-1]),
        "scale": float(np.exp(result.x[-1])),
        "surface_observation": surface_report,
        "residual_rms_before": float(np.sqrt(np.mean(before_vec * before_vec))),
        "residual_rms_after": float(np.sqrt(np.mean(after_vec * after_vec))),
        "before_summary": summary_from_rows(before_rows),
        "after_summary": summary_from_rows(after_rows),
        "frame_metrics_before": before_rows,
        "frame_metrics_after": after_rows,
    }
    (args.output_dir / "qc_mesh_prior_pose_graph_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["dataset", "annotation-vggt"], default="dataset")
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-observed-points", type=int, default=900)
    parser.add_argument("--max-prior-surface-points", type=int, default=1100)
    parser.add_argument("--max-projection-points", type=int, default=1300)
    parser.add_argument("--surface-match-mode", choices=["whole_mesh", "anchor_visible_patch"], default="whole_mesh")
    parser.add_argument("--anchor-patch-max-distance-m", type=float, default=0.040)
    parser.add_argument("--min-anchor-patch-points", type=int, default=80)
    parser.add_argument("--max-anchor-patch-points", type=int, default=900)
    parser.add_argument("--sigma-observed-m", type=float, default=0.030)
    parser.add_argument("--sigma-silhouette-px", type=float, default=5.0)
    parser.add_argument("--sigma-front-depth-m", type=float, default=0.020)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--sigma-world-velocity-m-s", type=float, default=0.80)
    parser.add_argument("--sigma-world-accel-m-s2", type=float, default=3.5)
    parser.add_argument("--sigma-camera-rotation-step-rad-s", type=float, default=2.5)
    parser.add_argument("--sigma-camera-rotation-accel-rad-s2", type=float, default=4.0)
    parser.add_argument("--sigma-anchor-translation-m", type=float, default=0.012)
    parser.add_argument("--sigma-anchor-rotation-rad", type=float, default=0.060)
    parser.add_argument("--sigma-anchor-surface-lock-m", type=float, default=0.020)
    parser.add_argument("--sigma-log-scale", type=float, default=0.090)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.120)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-front-depth-residual-m", type=float, default=0.120)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
