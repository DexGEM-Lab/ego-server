#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    T_world_camera_prior: np.ndarray
    observed_points_camera: np.ndarray
    mask: np.ndarray
    mask_distance: np.ndarray
    depth_m: np.ndarray
    vggt_center_world: np.ndarray | None


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def annotation_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def load_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
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
            raise RuntimeError(f"empty archive frame {idx}")
        out[int(idx)] = (frame_vertices, frame_faces)
    return out


def load_vggt_centers(path: Path | None) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    blob = np.load(path)
    required = {"frame_idx", "camera_centers_aligned"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    centers = blob["camera_centers_aligned"].astype(np.float64)
    if centers.shape != (len(frame_idx), 3):
        raise RuntimeError(f"{path} camera_centers_aligned shape mismatch")
    return {int(idx): centers[i] for i, idx in enumerate(frame_idx)}


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


def load_intrinsics(dataset: Path) -> np.ndarray:
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"invalid intrinsics matrix: {dataset / 'cam_K.txt'}")
    return K


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


def mask_distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)


def load_frames(args: argparse.Namespace) -> list[FrameData]:
    annotations = annotation_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    observed_archive = load_archive(args.observed_mesh_npz)
    vggt_centers = load_vggt_centers(args.vggt_archive)
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
        T = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid camera transform for frame {frame_idx}")
        observed, _ = observed_archive[frame_idx]
        frames.append(
            FrameData(
                frame_idx=frame_idx,
                T_world_camera_prior=T,
                observed_points_camera=sample_rows(observed, int(args.max_observed_points), int(args.seed) + frame_idx),
                mask=mask > 0,
                mask_distance=mask_distance(mask > 0),
                depth_m=depth.astype(np.float64) / 1000.0,
                vggt_center_world=vggt_centers.get(frame_idx),
            )
        )
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames selected")
    return frames


def load_initial_object_params(path: Path, frame_indices: list[int]) -> tuple[np.ndarray, float]:
    qc = load_json(path)
    metrics = qc.get("frame_metrics_after")
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError(f"{path} missing frame_metrics_after")
    object_pose = np.zeros((len(frame_indices), 6), dtype=np.float64)
    for i, frame_idx in enumerate(frame_indices):
        row = metrics.get(str(frame_idx))
        if not isinstance(row, dict):
            raise RuntimeError(f"{path} missing frame {frame_idx} metrics")
        object_pose[i, :3] = np.asarray(row["rotation_delta_rad"], dtype=np.float64)
        object_pose[i, 3:6] = np.asarray(row["translation_camera_delta_m"], dtype=np.float64)
    scale = float(qc.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"invalid initial scale: {scale}")
    return object_pose, float(np.log(scale))


def pack(camera_pose: np.ndarray, object_pose: np.ndarray, log_scale: float) -> np.ndarray:
    return np.r_[camera_pose.reshape(-1), object_pose.reshape(-1), float(log_scale)]


def unpack(params: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    split = n * 6
    camera_pose = params[:split].reshape(n, 6)
    object_pose = params[split : split * 2].reshape(n, 6)
    return camera_pose, object_pose, float(params[-1])


def transform_object_camera(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray, log_scale: float) -> np.ndarray:
    rotvec = pose[:3]
    translation = pose[3:6]
    scale = float(np.exp(log_scale))
    return scale * ((points - pivot) @ Rotation.from_rotvec(rotvec).as_matrix().T) + pivot + translation


def corrected_camera(frame: FrameData, camera_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_delta = Rotation.from_rotvec(camera_pose[:3]).as_matrix()
    trans_delta = camera_pose[3:6]
    R_prior = np.asarray(frame.T_world_camera_prior[:3, :3], dtype=np.float64)
    t_prior = np.asarray(frame.T_world_camera_prior[:3, 3], dtype=np.float64)
    R = rot_delta @ R_prior
    t = t_prior + trans_delta
    return R, t


def world_from_camera_points(points_camera: np.ndarray, R_world_camera: np.ndarray, t_world_camera: np.ndarray) -> np.ndarray:
    return points_camera @ R_world_camera.T + t_world_camera[None, :]


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


def rotation_log_vector(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def residual_vector(
    params: np.ndarray,
    frames: list[FrameData],
    prior_surface: np.ndarray,
    prior_projection: np.ndarray,
    pivot: np.ndarray,
    K: np.ndarray,
    anchor_index: int,
    args: argparse.Namespace,
) -> np.ndarray:
    camera_pose, object_pose, log_scale = unpack(params, len(frames))
    residuals = []
    object_centers_world = []
    object_rotations_world = []
    camera_centers_world = []
    camera_rotations_world = []
    for i, frame in enumerate(frames):
        surface_camera = transform_object_camera(prior_surface, pivot, object_pose[i], log_scale)
        tree = cKDTree(surface_camera)
        d_observed, _ = tree.query(frame.observed_points_camera, k=1)
        residuals.append(np.clip(d_observed, 0.0, float(args.max_surface_residual_m)) / float(args.sigma_observed_m))

        projection_camera = transform_object_camera(prior_projection, pivot, object_pose[i], log_scale)
        outside, front_depth, _ = projection_terms(projection_camera, frame, K, args)
        residuals.append(outside)
        residuals.append(front_depth)

        R_wc, t_wc = corrected_camera(frame, camera_pose[i])
        center_camera = transform_object_camera(pivot[None, :], pivot, object_pose[i], log_scale)[0]
        object_centers_world.append(world_from_camera_points(center_camera[None, :], R_wc, t_wc)[0])
        object_rotations_world.append(R_wc @ Rotation.from_rotvec(object_pose[i, :3]).as_matrix())
        camera_centers_world.append(t_wc)
        camera_rotations_world.append(R_wc)

        residuals.append(camera_pose[i, 3:6] / float(args.sigma_camera_translation_prior_m))
        residuals.append(camera_pose[i, :3] / float(args.sigma_camera_rotation_prior_rad))
        if frame.vggt_center_world is not None:
            residuals.append((t_wc - frame.vggt_center_world) / float(args.sigma_vggt_center_m))

    object_centers_world = np.asarray(object_centers_world, dtype=np.float64)
    camera_centers_world = np.asarray(camera_centers_world, dtype=np.float64)
    for i in range(1, len(frames)):
        dt = max(1, frames[i].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append((object_centers_world[i] - object_centers_world[i - 1]) / (float(args.sigma_object_world_velocity_m_s) * dt))
        residuals.append(
            rotation_log_vector(object_rotations_world[i - 1].T @ object_rotations_world[i])
            / (float(args.sigma_object_world_angular_velocity_rad_s) * dt)
        )
        residuals.append((camera_centers_world[i] - camera_centers_world[i - 1]) / (float(args.sigma_camera_world_velocity_m_s) * dt))
        residuals.append(
            rotation_log_vector(camera_rotations_world[i - 1].T @ camera_rotations_world[i])
            / (float(args.sigma_camera_world_angular_velocity_rad_s) * dt)
        )
    for i in range(1, len(frames) - 1):
        dt = max(1, frames[i + 1].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append(
            (object_centers_world[i + 1] - 2.0 * object_centers_world[i] + object_centers_world[i - 1])
            / (float(args.sigma_object_world_accel_m_s2) * dt * dt)
        )
        residuals.append(
            (camera_centers_world[i + 1] - 2.0 * camera_centers_world[i] + camera_centers_world[i - 1])
            / (float(args.sigma_camera_world_accel_m_s2) * dt * dt)
        )

    residuals.append(camera_pose[anchor_index, 3:6] / float(args.sigma_anchor_camera_translation_m))
    residuals.append(camera_pose[anchor_index, :3] / float(args.sigma_anchor_camera_rotation_rad))
    residuals.append(object_pose[anchor_index, 3:6] / float(args.sigma_anchor_object_translation_m))
    residuals.append(object_pose[anchor_index, :3] / float(args.sigma_anchor_object_rotation_rad))
    residuals.append(np.asarray([log_scale / float(args.sigma_log_scale)], dtype=np.float64))
    return np.concatenate([part.reshape(-1) for part in residuals])


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


def metrics(
    params: np.ndarray,
    frames: list[FrameData],
    prior_surface: np.ndarray,
    prior_projection: np.ndarray,
    pivot: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict]:
    camera_pose, object_pose, log_scale = unpack(params, len(frames))
    rows: dict[str, dict] = {}
    object_centers = []
    object_rotations = []
    camera_centers = []
    camera_rotations = []
    for i, frame in enumerate(frames):
        surface_camera = transform_object_camera(prior_surface, pivot, object_pose[i], log_scale)
        d_observed, _ = cKDTree(surface_camera).query(frame.observed_points_camera, k=1)
        projection_camera = transform_object_camera(prior_projection, pivot, object_pose[i], log_scale)
        outside, front_depth, inside = projection_terms(
            projection_camera,
            frame,
            K,
            argparse.Namespace(
                max_silhouette_px=float(args.max_silhouette_px),
                sigma_silhouette_px=1.0,
                min_depth_m=float(args.min_depth_m),
                depth_front_tolerance_m=0.0,
                max_front_depth_residual_m=10.0,
                sigma_front_depth_m=1.0,
            ),
        )
        R_wc, t_wc = corrected_camera(frame, camera_pose[i])
        center_camera = transform_object_camera(pivot[None, :], pivot, object_pose[i], log_scale)[0]
        center_world = world_from_camera_points(center_camera[None, :], R_wc, t_wc)[0]
        R_object_world = R_wc @ Rotation.from_rotvec(object_pose[i, :3]).as_matrix()
        object_centers.append(center_world)
        object_rotations.append(R_object_world)
        camera_centers.append(t_wc)
        camera_rotations.append(R_wc)
        row = {
            "observed_to_prior_median_m": float(np.median(d_observed)),
            "observed_to_prior_p95_m": float(np.percentile(d_observed, 95.0)),
            "silhouette_outside_p95_px": float(np.percentile(outside, 95.0)),
            "front_depth_violation_p95_m": float(np.percentile(front_depth, 95.0)),
            "projection_inside_mask_fraction": float(np.mean(inside)),
            "object_center_world_m": center_world.astype(float).tolist(),
            "camera_center_world_m": t_wc.astype(float).tolist(),
            "camera_translation_delta_m": camera_pose[i, 3:6].astype(float).tolist(),
            "camera_rotation_delta_rad": camera_pose[i, :3].astype(float).tolist(),
            "object_translation_camera_delta_m": object_pose[i, 3:6].astype(float).tolist(),
            "object_rotation_camera_delta_rad": object_pose[i, :3].astype(float).tolist(),
        }
        if frame.vggt_center_world is not None:
            row["vggt_center_error_m"] = float(np.linalg.norm(t_wc - frame.vggt_center_world))
        rows[str(frame.frame_idx)] = row
    for i in range(1, len(frames)):
        dt = max(1, frames[i].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        rows[str(frames[i].frame_idx)]["object_world_center_speed_m_s_from_prev"] = float(
            np.linalg.norm(object_centers[i] - object_centers[i - 1]) / dt
        )
        rows[str(frames[i].frame_idx)]["object_world_angular_speed_rad_s_from_prev"] = float(
            np.linalg.norm(rotation_log_vector(object_rotations[i - 1].T @ object_rotations[i])) / dt
        )
        rows[str(frames[i].frame_idx)]["camera_world_center_speed_m_s_from_prev"] = float(
            np.linalg.norm(camera_centers[i] - camera_centers[i - 1]) / dt
        )
        rows[str(frames[i].frame_idx)]["camera_world_angular_speed_rad_s_from_prev"] = float(
            np.linalg.norm(rotation_log_vector(camera_rotations[i - 1].T @ camera_rotations[i])) / dt
        )
    return rows


def summary_from_rows(rows: dict[str, dict]) -> dict:
    keys = (
        "observed_to_prior_median_m",
        "observed_to_prior_p95_m",
        "silhouette_outside_p95_px",
        "front_depth_violation_p95_m",
        "projection_inside_mask_fraction",
        "object_world_center_speed_m_s_from_prev",
        "object_world_angular_speed_rad_s_from_prev",
        "camera_world_center_speed_m_s_from_prev",
        "camera_world_angular_speed_rad_s_from_prev",
        "vggt_center_error_m",
    )
    return {key: summarize([row[key] for row in rows.values() if key in row]) for key in keys}


def save_outputs(
    output_dir: Path,
    frames: list[FrameData],
    mesh_vertices_camera_anchor: np.ndarray,
    mesh_faces: np.ndarray,
    pivot: np.ndarray,
    params: np.ndarray,
) -> tuple[Path, Path]:
    camera_pose, object_pose, log_scale = unpack(params, len(frames))
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    patched_annotations = {"frames": []}
    for i, frame in enumerate(frames):
        R_wc, t_wc = corrected_camera(frame, camera_pose[i])
        camera_vertices = transform_object_camera(mesh_vertices_camera_anchor, pivot, object_pose[i], log_scale)
        world_vertices = world_from_camera_points(camera_vertices, R_wc, t_wc)
        vertices_all.append(world_vertices.astype(np.float32))
        faces_all.append(mesh_faces.astype(np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(world_vertices))
        face_offsets.append(face_offsets[-1] + len(mesh_faces))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_wc
        T[:3, 3] = t_wc
        patched_annotations["frames"].append(
            {
                "frame_idx": int(frame.frame_idx),
                "camera": {"T_world_camera_metric": T.astype(float).tolist()},
            }
        )
    archive_path = output_dir / "joint_camera_object_meshes_world.npz"
    np.savez_compressed(
        archive_path,
        frame_idx=np.asarray([frame.frame_idx for frame in frames], dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )
    camera_path = output_dir / "joint_camera_patch_frames.json"
    camera_path.write_text(json.dumps(patched_annotations, indent=2), encoding="utf-8")
    return archive_path, camera_path


def run(args: argparse.Namespace) -> dict:
    frames = load_frames(args)
    frame_indices = [int(frame.frame_idx) for frame in frames]
    anchor_index = next((i for i, frame in enumerate(frames) if frame.frame_idx == int(args.anchor_frame)), None)
    if anchor_index is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from selected frames")
    mesh = load_mesh(args.mesh_prior_camera)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
    pivot = np.median(mesh_vertices, axis=0)
    prior_surface = sample_mesh_surface(mesh, int(args.max_prior_surface_points), int(args.seed) + 211)
    prior_projection = sample_mesh_surface(mesh, int(args.max_projection_points), int(args.seed) + 307)
    K = load_intrinsics(args.dataset)
    initial_object_pose, initial_log_scale = load_initial_object_params(args.initial_object_pose_qc, frame_indices)
    initial_camera_pose = np.zeros_like(initial_object_pose)
    x0 = pack(initial_camera_pose, initial_object_pose, initial_log_scale)
    before_vec = residual_vector(x0, frames, prior_surface, prior_projection, pivot, K, anchor_index, args)
    result = least_squares(
        lambda x: residual_vector(x, frames, prior_surface, prior_projection, pivot, K, anchor_index, args),
        x0,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after_vec = residual_vector(result.x, frames, prior_surface, prior_projection, pivot, K, anchor_index, args)
    before_rows = metrics(x0, frames, prior_surface, prior_projection, pivot, K, args)
    after_rows = metrics(result.x, frames, prior_surface, prior_projection, pivot, K, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path, camera_path = save_outputs(args.output_dir, frames, mesh_vertices, mesh_faces, pivot, result.x)
    report = {
        "status": "ok" if result.success else "optimizer_incomplete",
        "annotation_ready": False,
        "method": "joint_camera_object_factor_graph_v3",
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "initial_object_pose_qc": str(args.initial_object_pose_qc),
        "vggt_archive": str(args.vggt_archive) if args.vggt_archive is not None else None,
        "used_frames": frame_indices,
        "anchor_frame": int(args.anchor_frame),
        "variables": int(len(result.x)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before_vec * before_vec))),
        "residual_rms_after": float(np.sqrt(np.mean(after_vec * after_vec))),
        "log_scale": float(unpack(result.x, len(frames))[2]),
        "scale": float(np.exp(unpack(result.x, len(frames))[2])),
        "before_summary": summary_from_rows(before_rows),
        "after_summary": summary_from_rows(after_rows),
        "frame_metrics_before": before_rows,
        "frame_metrics_after": after_rows,
        "mesh_archive_world": str(archive_path),
        "camera_patch_frames": str(camera_path),
    }
    (args.output_dir / "qc_joint_camera_object_graph_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--initial-object-pose-qc", type=Path, required=True)
    parser.add_argument("--vggt-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--max-observed-points", type=int, default=900)
    parser.add_argument("--max-prior-surface-points", type=int, default=1100)
    parser.add_argument("--max-projection-points", type=int, default=1300)
    parser.add_argument("--sigma-observed-m", type=float, default=0.040)
    parser.add_argument("--sigma-silhouette-px", type=float, default=3.5)
    parser.add_argument("--sigma-front-depth-m", type=float, default=0.025)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--sigma_camera_translation_prior_m", "--sigma-camera-translation-prior-m", dest="sigma_camera_translation_prior_m", type=float, default=0.040)
    parser.add_argument("--sigma_camera_rotation_prior_rad", "--sigma-camera-rotation-prior-rad", dest="sigma_camera_rotation_prior_rad", type=float, default=0.080)
    parser.add_argument("--sigma-vggt-center-m", type=float, default=0.010)
    parser.add_argument("--sigma-object-world-velocity-m-s", type=float, default=0.35)
    parser.add_argument("--sigma-object-world-angular-velocity-rad-s", type=float, default=2.0)
    parser.add_argument("--sigma-object-world-accel-m-s2", type=float, default=1.5)
    parser.add_argument("--sigma-camera-world-velocity-m-s", type=float, default=0.75)
    parser.add_argument("--sigma-camera-world-angular-velocity-rad-s", type=float, default=6.0)
    parser.add_argument("--sigma-camera-world-accel-m-s2", type=float, default=4.0)
    parser.add_argument("--sigma-anchor-camera-translation-m", type=float, default=0.006)
    parser.add_argument("--sigma-anchor-camera-rotation-rad", type=float, default=0.030)
    parser.add_argument("--sigma-anchor-object-translation-m", type=float, default=0.006)
    parser.add_argument("--sigma-anchor-object-rotation-rad", type=float, default=0.030)
    parser.add_argument("--sigma-log-scale", type=float, default=0.025)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.120)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-front-depth-residual-m", type=float, default=0.120)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
