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
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from diagnose_contact_kinematics_v3 import selected_vertex_ids
from diagnose_hand_reprojection_depth_v3 import project_points
from optimize_mesh_prior_pose_graph_v3 import load_mesh, mask_distance, sample_mesh_surface, sample_rows


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    T_world_camera: np.ndarray
    intrinsics: np.ndarray
    mask: np.ndarray
    mask_distance: np.ndarray
    observed_points_camera: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class ContactData:
    frame_i: int
    frame_idx: int
    hand_idx: int
    track_id: str
    source: str
    region: str
    hand_patch_camera: np.ndarray


@dataclass(frozen=True)
class VolumeSDF:
    sdf: np.ndarray
    transform: np.ndarray
    pitch_m: float


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def annotations_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    return {int(row["frame_idx"]): row for row in frames}


def load_depth_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "depth", "intrinsics_fx_fy_cx_cy"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing keys: {sorted(missing)}")
    frame_idx = blob["frame_idx"].astype(int)
    depth = blob["depth"].astype(np.float64)
    intrinsics = blob["intrinsics_fx_fy_cx_cy"].astype(np.float64)
    if depth.ndim != 3 or len(frame_idx) != depth.shape[0] or intrinsics.shape != (len(frame_idx), 4):
        raise RuntimeError(f"{path} has invalid frame/depth/intrinsics shapes")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, idx in enumerate(frame_idx.tolist()):
        if int(idx) in out:
            raise RuntimeError(f"{path} has duplicate frame {idx}")
        frame_depth = depth[i]
        if frame_depth.ndim != 2 or not np.isfinite(frame_depth).all():
            raise RuntimeError(f"{path} frame {idx} has invalid depth")
        frame_intrinsics = intrinsics[i]
        if frame_intrinsics.shape != (4,) or not np.isfinite(frame_intrinsics).all():
            raise RuntimeError(f"{path} frame {idx} has invalid intrinsics")
        out[int(idx)] = (frame_depth, frame_intrinsics)
    return out


def read_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {path}")
    if mask.shape != expected_shape:
        mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def sample_mask_depth_points(mask: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray, count: int, seed: int) -> np.ndarray:
    if mask.shape != depth_m.shape:
        raise RuntimeError("mask and metric depth shape mismatch")
    ys, xs = np.nonzero(mask & np.isfinite(depth_m) & (depth_m > 0.0))
    if len(xs) == 0:
        raise RuntimeError("object mask has no valid metric depth pixels")
    coords = np.c_[xs, ys]
    if len(coords) > int(count):
        rng = np.random.default_rng(int(seed))
        coords = coords[rng.choice(len(coords), size=int(count), replace=False)]
    z = depth_m[coords[:, 1], coords[:, 0]].astype(np.float64)
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    x = (coords[:, 0].astype(np.float64) - cx) * z / fx
    y = (coords[:, 1].astype(np.float64) - cy) * z / fy
    points = np.c_[x, y, z]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise RuntimeError("sampled object metric-depth points are invalid")
    return points.astype(np.float64)


def hand_vertices_camera(hand: dict) -> np.ndarray:
    for key in ("vertices_source_camera_m", "vertices_source_camera_m_sample"):
        arr = np.asarray(hand.get(key, []), dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0 and np.isfinite(arr).all():
            return arr
    raise RuntimeError("hand has no usable source-camera MANO vertices")


def contact_rows(path: Path) -> list[dict]:
    report = load_json(path)
    rows = [
        row
        for row in report.get("rows_detail", [])
        if bool(row.get("reliable_for_contact", False))
        or bool(row.get("geometry_backed_temporal_contact", False))
    ]
    if not rows:
        raise RuntimeError(f"{path} contains no reliable or geometry-backed temporal contact rows")
    return sorted(rows, key=lambda row: (int(row["frame_idx"]), int(row["hand_idx"])))


def build_frames_and_contacts(args: argparse.Namespace) -> tuple[list[FrameData], list[ContactData], list[dict]]:
    annotations = annotations_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    depths = load_depth_archive(args.metric_depth_npz)
    selected_frames = []
    for idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if idx not in annotations or idx not in manifest or idx not in depths:
            continue
        selected_frames.append(idx)
    if len(selected_frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(selected_frames)} usable frames selected")
    frames: list[FrameData] = []
    frame_i_by_idx: dict[int, int] = {}
    for idx in selected_frames:
        depth_m, depth_intrinsics = depths[idx]
        if args.intrinsics_source == "annotation-vggt":
            intrinsics = np.asarray(annotations[idx].get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
        elif args.intrinsics_source == "metric-depth":
            intrinsics = depth_intrinsics
        else:
            raise RuntimeError(f"unsupported intrinsics source {args.intrinsics_source}")
        if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
            raise RuntimeError(f"frame {idx} has invalid {args.intrinsics_source} intrinsics")
        mask = read_mask(Path(manifest[idx]["mask"]), depth_m.shape)
        T_world_camera = np.asarray(annotations[idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
            raise RuntimeError(f"frame {idx} has invalid T_world_camera_metric")
        observed = sample_mask_depth_points(mask, depth_m, intrinsics, int(args.max_observed_points), int(args.seed) + idx)
        frame_i_by_idx[idx] = len(frames)
        frames.append(
            FrameData(
                frame_idx=idx,
                T_world_camera=T_world_camera,
                intrinsics=intrinsics,
                mask=mask,
                mask_distance=mask_distance(mask),
                observed_points_camera=observed,
                depth_m=depth_m,
            )
        )

    contacts: list[ContactData] = []
    contact_build_rows = []
    for row in contact_rows(args.contact_report):
        idx = int(row["frame_idx"])
        if idx not in frame_i_by_idx:
            continue
        hand_idx = int(row["hand_idx"])
        hand = annotations[idx]["hands"][hand_idx]
        vertices = hand_vertices_camera(hand)
        patch_ids = selected_vertex_ids(row)
        if int(patch_ids.max()) >= len(vertices):
            raise RuntimeError(f"frame {idx} hand {hand_idx} patch id exceeds MANO vertex count")
        patch = vertices[patch_ids]
        if len(patch) > int(args.max_contact_points_per_row):
            patch = sample_rows(patch, int(args.max_contact_points_per_row), int(args.seed) + idx + hand_idx + 9000)
        contacts.append(
            ContactData(
                frame_i=frame_i_by_idx[idx],
                frame_idx=idx,
                hand_idx=hand_idx,
                track_id=str(row.get("track_id")),
                source=str(row.get("selected_patch_source")),
                region=str(row.get("selected_patch_region")),
                hand_patch_camera=patch,
            )
        )
        contact_build_rows.append(
            {
                "frame_idx": idx,
                "hand_idx": hand_idx,
                "track_id": row.get("track_id"),
                "selected_patch_source": row.get("selected_patch_source"),
                "selected_patch_region": row.get("selected_patch_region"),
                "patch_points": int(len(patch)),
            }
        )
    if len(contacts) < int(args.min_contact_rows):
        raise RuntimeError(f"only {len(contacts)} contact rows available")
    return frames, contacts, contact_build_rows


def transform_camera(points: np.ndarray, pivot: np.ndarray, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(rotvec).as_matrix()
    return (points - pivot) @ R.T + pivot + translation


def camera_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def rotation_world(rotvec: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return T_world_camera[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_rotvec()


def project_camera(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(np.float64).tolist()
    return project_points(points, np.asarray([fx, fy, cx, cy], dtype=np.float64))


def projection_terms(points: np.ndarray, frame: FrameData, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = points[:, 2] > float(args.min_depth_m)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if np.any(positive):
        uv[positive] = project_camera(points[positive], frame.intrinsics)
    rounded = np.rint(uv).astype(np.int64)
    in_bounds = (
        positive
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < frame.mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < frame.mask.shape[0])
    )
    outside = np.full(len(points), float(args.max_silhouette_px), dtype=np.float64)
    inside = np.zeros(len(points), dtype=bool)
    if np.any(in_bounds):
        y = rounded[in_bounds, 1]
        x = rounded[in_bounds, 0]
        outside[in_bounds] = np.minimum(frame.mask_distance[y, x], float(args.max_silhouette_px))
        inside[in_bounds] = frame.mask[y, x]
    front = np.zeros(len(points), dtype=np.float64)
    if np.any(in_bounds):
        y = rounded[in_bounds, 1]
        x = rounded[in_bounds, 0]
        metric_depth = frame.depth_m[y, x]
        object_z = points[in_bounds, 2]
        violation = object_z - metric_depth - float(args.depth_front_tolerance_m)
        front[in_bounds] = np.clip(violation, 0.0, float(args.max_front_depth_residual_m))
    return outside / float(args.sigma_silhouette_px), front / float(args.sigma_front_depth_m), inside


def visible_depth_terms(points: np.ndarray, frame: FrameData, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if int(args.max_visible_depth_pixels) <= 0:
        return np.zeros(0, dtype=np.float64), {"samples": 0}
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64), {"samples": 0}
    if len(points) > int(args.max_visible_depth_pixels):
        rng = np.random.default_rng(int(args.seed) + int(frame.frame_idx) + 17000)
        point_ids = np.sort(rng.choice(len(points), size=int(args.max_visible_depth_pixels), replace=False))
    else:
        point_ids = np.arange(len(points), dtype=np.int64)
    selected = points[point_ids]
    residual = np.zeros(len(selected), dtype=np.float64)
    positive = selected[:, 2] > float(args.min_depth_m)
    if not np.any(positive):
        return residual, {"samples": 0}
    uv = project_camera(selected[positive], frame.intrinsics)
    z = selected[positive, 2].astype(np.float64)
    xy = np.rint(uv).astype(np.int64)
    in_bounds = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < frame.mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < frame.mask.shape[0])
    )
    if not np.any(in_bounds):
        return residual, {"samples": 0}
    positive_ids = np.flatnonzero(positive)
    xy_valid = xy[in_bounds]
    z_valid = z[in_bounds]
    selected_valid = positive_ids[in_bounds]
    mask_hit = frame.mask[xy_valid[:, 1], xy_valid[:, 0]]
    if not np.any(mask_hit):
        return residual, {"samples": 0}
    xy_valid = xy_valid[mask_hit]
    z_valid = z_valid[mask_hit]
    selected_valid = selected_valid[mask_hit]
    x = xy_valid[:, 0]
    y = xy_valid[:, 1]
    depth = frame.depth_m[y, x].astype(np.float64)
    valid = np.isfinite(depth) & (depth > float(args.min_depth_m))
    if not np.any(valid):
        return residual, {"samples": 0}
    err = z_valid[valid] - depth[valid]
    residual[selected_valid[valid]] = np.clip(err, -float(args.max_visible_depth_residual_m), float(args.max_visible_depth_residual_m))
    return residual / float(args.sigma_visible_depth_m), {
        "samples": int(len(err)),
        "signed_median_m": float(np.median(err)),
        "signed_p05_m": float(np.percentile(err, 5.0)),
        "signed_p95_m": float(np.percentile(err, 95.0)),
        "abs_median_m": float(np.median(np.abs(err))),
        "abs_p95_m": float(np.percentile(np.abs(err), 95.0)),
        "closer_than_depth_fraction_5mm": float(np.mean(err < -0.005)),
        "farther_than_depth_fraction_5mm": float(np.mean(err > 0.005)),
    }


def unpack(params: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    pose = params.reshape(frame_count, 6)
    return pose[:, :3], pose[:, 3:6]


def append_sample_block(residuals: list[np.ndarray], block: np.ndarray) -> None:
    values = np.asarray(block, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise RuntimeError("empty sampled residual block")
    residuals.append(values / math.sqrt(float(len(values))))


def build_volume_sdf(mesh: trimesh.Trimesh, args: argparse.Namespace) -> VolumeSDF | None:
    if not bool(args.use_volume_sdf):
        return None
    vox = mesh.voxelized(pitch=float(args.volume_sdf_pitch_m)).fill()
    occupied = np.asarray(vox.matrix, dtype=bool)
    if np.count_nonzero(occupied) == 0:
        raise RuntimeError("volume SDF voxelization produced no occupied cells")
    pad = int(args.volume_sdf_pad_voxels)
    occupied = np.pad(occupied, pad_width=pad, mode="constant", constant_values=False)
    outside = distance_transform_edt(~occupied, sampling=[float(args.volume_sdf_pitch_m)] * 3)
    inside = distance_transform_edt(occupied, sampling=[float(args.volume_sdf_pitch_m)] * 3)
    transform = np.asarray(vox.transform, dtype=np.float64).copy()
    transform[:3, 3] -= float(args.volume_sdf_pitch_m) * pad
    return VolumeSDF(sdf=(outside - inside).astype(np.float32), transform=transform, pitch_m=float(args.volume_sdf_pitch_m))


def sample_volume_sdf(points: np.ndarray, volume_sdf: VolumeSDF) -> np.ndarray:
    origin = volume_sdf.transform[:3, 3]
    coords = (points - origin[None, :]) / float(volume_sdf.pitch_m)
    base = np.floor(coords).astype(np.int64)
    frac = coords - base.astype(np.float64)
    in_bounds = (
        (base[:, 0] >= 0)
        & (base[:, 0] + 1 < volume_sdf.sdf.shape[0])
        & (base[:, 1] >= 0)
        & (base[:, 1] + 1 < volume_sdf.sdf.shape[1])
        & (base[:, 2] >= 0)
        & (base[:, 2] + 1 < volume_sdf.sdf.shape[2])
    )
    values = np.full(len(points), np.nan, dtype=np.float64)
    if np.any(in_bounds):
        b = base[in_bounds]
        f = frac[in_bounds]
        x0, y0, z0 = b[:, 0], b[:, 1], b[:, 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        xd, yd, zd = f[:, 0], f[:, 1], f[:, 2]
        c000 = volume_sdf.sdf[x0, y0, z0]
        c100 = volume_sdf.sdf[x1, y0, z0]
        c010 = volume_sdf.sdf[x0, y1, z0]
        c110 = volume_sdf.sdf[x1, y1, z0]
        c001 = volume_sdf.sdf[x0, y0, z1]
        c101 = volume_sdf.sdf[x1, y0, z1]
        c011 = volume_sdf.sdf[x0, y1, z1]
        c111 = volume_sdf.sdf[x1, y1, z1]
        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd
        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd
        values[in_bounds] = c0 * (1.0 - zd) + c1 * zd
    return values


def camera_to_local(points: np.ndarray, rotvec: np.ndarray, translation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return (points - translation[None, :] - pivot[None, :]) @ rotation + pivot[None, :]


def residual_vector(
    params: np.ndarray,
    frames: list[FrameData],
    contacts: list[ContactData],
    mesh_surface: np.ndarray,
    contact_surface: np.ndarray,
    projection_surface: np.ndarray,
    pivot: np.ndarray,
    volume_sdf: VolumeSDF | None,
    args: argparse.Namespace,
) -> np.ndarray:
    rotvecs, translations = unpack(params, len(frames))
    residuals: list[np.ndarray] = []
    object_centers_world = []
    object_rotations_world = []
    object_trees = []
    for i, frame in enumerate(frames):
        surface = transform_camera(mesh_surface, pivot, rotvecs[i], translations[i])
        tree = cKDTree(surface)
        object_trees.append(tree)
        d_obs, _ = tree.query(frame.observed_points_camera, k=1)
        append_sample_block(residuals, np.clip(d_obs, 0.0, float(args.max_surface_residual_m)) / float(args.sigma_observed_m))
        projection_points = transform_camera(projection_surface, pivot, rotvecs[i], translations[i])
        outside, front, _inside = projection_terms(projection_points, frame, args)
        append_sample_block(residuals, outside)
        append_sample_block(residuals, front)
        visible_depth, _visible_depth_metrics = visible_depth_terms(projection_points, frame, args)
        if len(visible_depth):
            append_sample_block(residuals, visible_depth)
        center_camera = transform_camera(pivot[None, :], pivot, rotvecs[i], translations[i])[0]
        object_centers_world.append(camera_to_world(center_camera[None, :], frame.T_world_camera)[0])
        object_rotations_world.append(rotation_world(rotvecs[i], frame.T_world_camera))
        residuals.append(translations[i] / float(args.sigma_object_translation_prior_m))
        residuals.append(rotvecs[i] / float(args.sigma_object_rotation_prior_rad))

    for contact in contacts:
        transformed_contact_mesh = transform_camera(contact_surface, pivot, rotvecs[contact.frame_i], translations[contact.frame_i])
        tree = cKDTree(transformed_contact_mesh)
        distances, _ = tree.query(contact.hand_patch_camera, k=1)
        append_sample_block(residuals, np.clip(distances, 0.0, float(args.max_contact_residual_m)) / float(args.sigma_contact_m))
        if volume_sdf is not None:
            local = camera_to_local(contact.hand_patch_camera, rotvecs[contact.frame_i], translations[contact.frame_i], pivot)
            signed = sample_volume_sdf(local, volume_sdf)
            signed = signed[np.isfinite(signed)]
            if len(signed) == 0:
                raise RuntimeError(f"contact row {contact.frame_idx} has no in-bounds volume SDF samples")
            penetration = np.clip(float(args.volume_sdf_surface_m) - signed, 0.0, float(args.max_volume_sdf_penetration_m))
            append_sample_block(residuals, penetration / float(args.sigma_volume_sdf_penetration_m))

    centers = np.asarray(object_centers_world, dtype=np.float64)
    for i in range(1, len(frames)):
        dt = max(1, frames[i].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append((centers[i] - centers[i - 1]) / (float(args.sigma_object_world_velocity_m_s) * dt))
        residuals.append(
            rotation_log_vector(object_rotations_world[i - 1].T @ object_rotations_world[i])
            / (float(args.sigma_object_world_angular_velocity_rad_s) * dt)
        )
    for i in range(1, len(frames) - 1):
        dt = max(1, frames[i + 1].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        residuals.append(
            (centers[i + 1] - 2.0 * centers[i] + centers[i - 1])
            / (float(args.sigma_object_world_accel_m_s2) * dt * dt)
        )
    anchor_i = next((i for i, frame in enumerate(frames) if frame.frame_idx == int(args.anchor_frame)), None)
    if anchor_i is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from selected frames")
    residuals.append(translations[anchor_i] / float(args.sigma_anchor_translation_m))
    residuals.append(rotvecs[anchor_i] / float(args.sigma_anchor_rotation_rad))
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


def frame_metrics(
    params: np.ndarray,
    frames: list[FrameData],
    contacts: list[ContactData],
    mesh_surface: np.ndarray,
    contact_surface: np.ndarray,
    projection_surface: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    pivot: np.ndarray,
    signed_mesh: trimesh.Trimesh | None,
    volume_sdf: VolumeSDF | None,
    args: argparse.Namespace,
) -> dict[str, dict]:
    rotvecs, translations = unpack(params, len(frames))
    rows: dict[str, dict] = {}
    object_trees = []
    centers_world = []
    rotations_world = []
    transformed_meshes = []
    for i, frame in enumerate(frames):
        surface = transform_camera(mesh_surface, pivot, rotvecs[i], translations[i])
        projection_points = transform_camera(projection_surface, pivot, rotvecs[i], translations[i])
        tree = cKDTree(surface)
        object_trees.append(tree)
        transformed_meshes.append(transform_camera(mesh_vertices, pivot, rotvecs[i], translations[i]))
        d_obs, _ = tree.query(frame.observed_points_camera, k=1)
        metric_args = argparse.Namespace(**vars(args))
        metric_args.sigma_silhouette_px = 1.0
        metric_args.sigma_front_depth_m = 1.0
        outside, front, inside = projection_terms(projection_points, frame, metric_args)
        _visible_depth_residual, visible_depth_metrics = visible_depth_terms(projection_points, frame, metric_args)
        center_camera = transform_camera(pivot[None, :], pivot, rotvecs[i], translations[i])[0]
        center_world = camera_to_world(center_camera[None, :], frame.T_world_camera)[0]
        R_world = rotation_world(rotvecs[i], frame.T_world_camera)
        centers_world.append(center_world)
        rotations_world.append(R_world)
        rows[str(frame.frame_idx)] = {
            "observed_to_prior_median_m": float(np.median(d_obs)),
            "observed_to_prior_p95_m": float(np.percentile(d_obs, 95.0)),
            "silhouette_outside_median_px": float(np.median(outside)),
            "silhouette_outside_p95_px": float(np.percentile(outside, 95.0)),
            "front_depth_violation_median_m": float(np.median(front)),
            "front_depth_violation_p95_m": float(np.percentile(front, 95.0)),
            "visible_depth_samples": int(visible_depth_metrics.get("samples", 0)),
            "visible_depth_signed_median_m": visible_depth_metrics.get("signed_median_m"),
            "visible_depth_signed_p05_m": visible_depth_metrics.get("signed_p05_m"),
            "visible_depth_signed_p95_m": visible_depth_metrics.get("signed_p95_m"),
            "visible_depth_abs_median_m": visible_depth_metrics.get("abs_median_m"),
            "visible_depth_abs_p95_m": visible_depth_metrics.get("abs_p95_m"),
            "visible_depth_closer_than_depth_fraction_5mm": visible_depth_metrics.get("closer_than_depth_fraction_5mm"),
            "visible_depth_farther_than_depth_fraction_5mm": visible_depth_metrics.get("farther_than_depth_fraction_5mm"),
            "projection_inside_mask_fraction": float(np.mean(inside)),
            "center_world_m": center_world.astype(float).tolist(),
            "translation_camera_delta_m": translations[i].astype(float).tolist(),
            "rotation_delta_rad": rotvecs[i].astype(float).tolist(),
            "contact_count": 0,
            "contact_distance_median_m": None,
            "contact_distance_p95_m": None,
            "contact_signed_distance_median_m": None,
            "contact_penetration_fraction": None,
            "contact_volume_sdf_median_m": None,
            "contact_volume_sdf_penetration_fraction": None,
        }
    for contact in contacts:
        row = rows[str(contact.frame_idx)]
        transformed_contact_mesh = transform_camera(contact_surface, pivot, rotvecs[contact.frame_i], translations[contact.frame_i])
        tree = cKDTree(transformed_contact_mesh)
        distances, _ = tree.query(contact.hand_patch_camera, k=1)
        signed_distances = None
        if signed_mesh is not None:
            local = camera_to_local(contact.hand_patch_camera, rotvecs[contact.frame_i], translations[contact.frame_i], pivot)
            signed_distances = trimesh.proximity.signed_distance(signed_mesh, local)
        volume_sdf_values = None
        if volume_sdf is not None:
            local = camera_to_local(contact.hand_patch_camera, rotvecs[contact.frame_i], translations[contact.frame_i], pivot)
            sampled = sample_volume_sdf(local, volume_sdf)
            volume_sdf_values = sampled[np.isfinite(sampled)]
        contact_row = {
            "frame_idx": int(contact.frame_idx),
            "hand_idx": int(contact.hand_idx),
            "track_id": contact.track_id,
            "selected_patch_source": contact.source,
            "selected_patch_region": contact.region,
            "distance_median_m": float(np.median(distances)),
            "distance_p95_m": float(np.percentile(distances, 95.0)),
            "distance_max_m": float(np.max(distances)),
        }
        if signed_distances is not None:
            contact_row.update(
                {
                    "signed_distance_median_m": float(np.median(signed_distances)),
                    "signed_distance_p05_m": float(np.percentile(signed_distances, 5.0)),
                    "penetration_fraction": float(np.mean(signed_distances > float(args.signed_penetration_positive_m))),
                }
            )
        if volume_sdf_values is not None:
            contact_row.update(
                {
                    "volume_sdf_median_m": float(np.median(volume_sdf_values)),
                    "volume_sdf_p05_m": float(np.percentile(volume_sdf_values, 5.0)),
                    "volume_sdf_penetration_fraction": float(
                        np.mean(volume_sdf_values < -float(args.volume_sdf_penetration_tolerance_m))
                    ),
                    "volume_sdf_near_surface_fraction": float(np.mean(np.abs(volume_sdf_values) <= float(args.volume_sdf_near_surface_m))),
                }
            )
        row.setdefault("contact_rows", []).append(contact_row)
        row["contact_count"] = int(row["contact_count"]) + 1
        row["contact_distance_median_m"] = float(np.median([r["distance_median_m"] for r in row["contact_rows"]]))
        row["contact_distance_p95_m"] = float(np.percentile([r["distance_p95_m"] for r in row["contact_rows"]], 95.0))
        if signed_distances is not None:
            row["contact_signed_distance_median_m"] = float(np.median([r["signed_distance_median_m"] for r in row["contact_rows"]]))
            row["contact_penetration_fraction"] = float(np.max([r["penetration_fraction"] for r in row["contact_rows"]]))
        if volume_sdf_values is not None:
            row["contact_volume_sdf_median_m"] = float(np.median([r["volume_sdf_median_m"] for r in row["contact_rows"]]))
            row["contact_volume_sdf_penetration_fraction"] = float(
                np.max([r["volume_sdf_penetration_fraction"] for r in row["contact_rows"]])
            )

    centers = np.asarray(centers_world, dtype=np.float64)
    for i in range(1, len(frames)):
        dt = max(1, frames[i].frame_idx - frames[i - 1].frame_idx) / float(args.fps)
        rows[str(frames[i].frame_idx)]["world_center_speed_m_s_from_prev"] = float(np.linalg.norm(centers[i] - centers[i - 1]) / dt)
        rows[str(frames[i].frame_idx)]["world_angular_speed_rad_s_from_prev"] = float(
            np.linalg.norm(rotation_log_vector(rotations_world[i - 1].T @ rotations_world[i])) / dt
        )
    return rows


def summary_from_rows(rows: dict[str, dict]) -> dict:
    keys = [
        "observed_to_prior_median_m",
        "observed_to_prior_p95_m",
        "silhouette_outside_p95_px",
        "front_depth_violation_p95_m",
        "visible_depth_abs_median_m",
        "visible_depth_abs_p95_m",
        "visible_depth_closer_than_depth_fraction_5mm",
        "visible_depth_farther_than_depth_fraction_5mm",
        "projection_inside_mask_fraction",
        "world_center_speed_m_s_from_prev",
        "world_angular_speed_rad_s_from_prev",
        "contact_distance_median_m",
        "contact_distance_p95_m",
        "contact_penetration_fraction",
        "contact_volume_sdf_median_m",
        "contact_volume_sdf_penetration_fraction",
    ]
    return {key: summarize([row[key] for row in rows.values() if row.get(key) is not None]) for key in keys}


def save_world_archive(
    path: Path,
    frames: list[FrameData],
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    pivot: np.ndarray,
    params: np.ndarray,
) -> None:
    rotvecs, translations = unpack(params, len(frames))
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for i, frame in enumerate(frames):
        camera_vertices = transform_camera(mesh_vertices, pivot, rotvecs[i], translations[i])
        world_vertices = camera_to_world(camera_vertices, frame.T_world_camera)
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


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    frames, contacts, contact_build_rows = build_frames_and_contacts(args)
    mesh = load_mesh(args.mesh_prior_camera)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
    pivot = np.median(mesh_vertices, axis=0)
    mesh_surface = sample_mesh_surface(mesh, int(args.max_prior_surface_points), int(args.seed) + 100)
    contact_surface = sample_mesh_surface(mesh, int(args.max_contact_surface_points), int(args.seed) + 150)
    projection_surface = sample_mesh_surface(mesh, int(args.max_projection_points), int(args.seed) + 200)
    volume_sdf = build_volume_sdf(mesh, args)
    x0 = np.zeros(len(frames) * 6, dtype=np.float64)
    before_vec = residual_vector(x0, frames, contacts, mesh_surface, contact_surface, projection_surface, pivot, volume_sdf, args)
    result = least_squares(
        lambda x: residual_vector(x, frames, contacts, mesh_surface, contact_surface, projection_surface, pivot, volume_sdf, args),
        x0,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after_vec = residual_vector(result.x, frames, contacts, mesh_surface, contact_surface, projection_surface, pivot, volume_sdf, args)
    signed_mesh = mesh if bool(mesh.is_watertight) and bool(mesh.is_winding_consistent) else None
    before_rows = frame_metrics(
        x0, frames, contacts, mesh_surface, contact_surface, projection_surface, mesh_vertices, mesh_faces, pivot, signed_mesh, volume_sdf, args
    )
    after_rows = frame_metrics(
        result.x, frames, contacts, mesh_surface, contact_surface, projection_surface, mesh_vertices, mesh_faces, pivot, signed_mesh, volume_sdf, args
    )
    before_summary = summary_from_rows(before_rows)
    after_summary = summary_from_rows(after_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "contact_patch_object_pose_graph_meshes_world.npz"
    save_world_archive(archive_path, frames, mesh_vertices, mesh_faces, pivot, result.x)
    contact_p95 = after_summary["contact_distance_p95_m"].get("median")
    surface_p95 = after_summary["observed_to_prior_p95_m"].get("median")
    visible_depth_p95 = after_summary["visible_depth_abs_p95_m"].get("median")
    contact_volume_penetration_p95 = after_summary["contact_volume_sdf_penetration_fraction"].get("p95")
    speed = after_summary["world_center_speed_m_s_from_prev"].get("median")
    projection_inside = after_summary["projection_inside_mask_fraction"].get("median")
    status = "diagnostic_contact_patch_object_pose_unsolved"
    if contact_p95 is not None and surface_p95 is not None and speed is not None and projection_inside is not None:
        visible_depth_ok = visible_depth_p95 is None or float(visible_depth_p95) <= float(args.accept_visible_depth_p95_m)
        contact_volume_ok = contact_volume_penetration_p95 is None or float(contact_volume_penetration_p95) <= float(
            args.accept_contact_volume_penetration_p95
        )
        if (
            float(contact_p95) <= float(args.accept_contact_p95_m)
            and float(surface_p95) <= float(args.accept_surface_p95_m)
            and visible_depth_ok
            and contact_volume_ok
            and float(speed) <= float(args.accept_world_speed_m_s)
            and float(projection_inside) >= float(args.accept_projection_inside_fraction)
        ):
            status = "diagnostic_contact_patch_object_pose_consistent"
    report = {
        "status": status,
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "contact_patch_complete_mesh_object_pose_graph_v3",
        "annotations": str(args.annotations),
        "manifest": str(args.manifest),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "contact_report": str(args.contact_report),
        "mesh_prior_camera": str(args.mesh_prior_camera),
        "mesh_archive_world": str(archive_path),
        "used_frames": [int(frame.frame_idx) for frame in frames],
        "contact_rows": contact_build_rows,
        "variables": int(len(result.x)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "mesh_watertight": bool(mesh.is_watertight),
        "signed_penetration_supported": signed_mesh is not None,
        "volume_sdf_supported": volume_sdf is not None,
        "residual_rms_before": float(np.sqrt(np.mean(before_vec * before_vec))),
        "residual_rms_after": float(np.sqrt(np.mean(after_vec * after_vec))),
        "before_summary": before_summary,
        "after_summary": after_summary,
        "frame_metrics_before": before_rows,
        "frame_metrics_after": after_rows,
        "acceptance": {
            "accept_contact_p95_m": float(args.accept_contact_p95_m),
            "accept_surface_p95_m": float(args.accept_surface_p95_m),
            "accept_visible_depth_p95_m": float(args.accept_visible_depth_p95_m),
            "accept_contact_volume_penetration_p95": float(args.accept_contact_volume_penetration_p95),
            "accept_world_speed_m_s": float(args.accept_world_speed_m_s),
            "accept_projection_inside_fraction": float(args.accept_projection_inside_fraction),
        },
        "priors": {
            "sampled_observation_blocks": "normalized_by_sqrt_block_length",
            "sigma_observed_m": float(args.sigma_observed_m),
            "sigma_silhouette_px": float(args.sigma_silhouette_px),
            "sigma_front_depth_m": float(args.sigma_front_depth_m),
            "sigma_visible_depth_m": float(args.sigma_visible_depth_m),
            "max_visible_depth_pixels": int(args.max_visible_depth_pixels),
            "sigma_contact_m": float(args.sigma_contact_m),
            "sigma_volume_sdf_penetration_m": float(args.sigma_volume_sdf_penetration_m),
            "sigma_object_translation_prior_m": float(args.sigma_object_translation_prior_m),
            "sigma_object_rotation_prior_rad": float(args.sigma_object_rotation_prior_rad),
            "sigma_object_world_velocity_m_s": float(args.sigma_object_world_velocity_m_s),
            "sigma_object_world_angular_velocity_rad_s": float(args.sigma_object_world_angular_velocity_rad_s),
            "sigma_object_world_accel_m_s2": float(args.sigma_object_world_accel_m_s2),
        },
        "elapsed_s": float(time.time() - started),
    }
    qc_path = args.output_dir / "qc_contact_patch_object_pose_graph_v3.json"
    qc_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--mesh-prior-camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--min-contact-rows", type=int, default=2)
    parser.add_argument("--max-observed-points", type=int, default=700)
    parser.add_argument("--max-prior-surface-points", type=int, default=5000)
    parser.add_argument("--max-contact-surface-points", type=int, default=8000)
    parser.add_argument("--max-projection-points", type=int, default=900)
    parser.add_argument("--max-contact-points-per-row", type=int, default=80)
    parser.add_argument("--sigma-observed-m", type=float, default=0.020)
    parser.add_argument("--sigma-silhouette-px", type=float, default=6.0)
    parser.add_argument("--sigma-front-depth-m", type=float, default=0.020)
    parser.add_argument("--sigma-visible-depth-m", type=float, default=0.020)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.008)
    parser.add_argument("--sigma-contact-m", type=float, default=0.006)
    parser.add_argument("--use-volume-sdf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--volume-sdf-pitch-m", type=float, default=0.003)
    parser.add_argument("--volume-sdf-pad-voxels", type=int, default=8)
    parser.add_argument("--volume-sdf-surface-m", type=float, default=0.0)
    parser.add_argument("--volume-sdf-penetration-tolerance-m", type=float, default=0.002)
    parser.add_argument("--volume-sdf-near-surface-m", type=float, default=0.006)
    parser.add_argument("--sigma-volume-sdf-penetration-m", type=float, default=0.006)
    parser.add_argument("--max-volume-sdf-penetration-m", type=float, default=0.030)
    parser.add_argument("--sigma-object-translation-prior-m", type=float, default=0.045)
    parser.add_argument("--sigma-object-rotation-prior-rad", type=float, default=0.20)
    parser.add_argument("--sigma-object-world-velocity-m-s", type=float, default=0.55)
    parser.add_argument("--sigma-object-world-angular-velocity-rad-s", type=float, default=3.0)
    parser.add_argument("--sigma-object-world-accel-m-s2", type=float, default=5.0)
    parser.add_argument("--sigma-anchor-translation-m", type=float, default=0.012)
    parser.add_argument("--sigma-anchor-rotation-rad", type=float, default=0.060)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.100)
    parser.add_argument("--max-contact-residual-m", type=float, default=0.060)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-front-depth-residual-m", type=float, default=0.120)
    parser.add_argument("--max-visible-depth-residual-m", type=float, default=0.120)
    parser.add_argument("--max-visible-depth-pixels", type=int, default=0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--signed-penetration-positive-m", type=float, default=0.001)
    parser.add_argument("--accept-contact-p95-m", type=float, default=0.006)
    parser.add_argument("--accept-surface-p95-m", type=float, default=0.030)
    parser.add_argument("--accept-visible-depth-p95-m", type=float, default=0.030)
    parser.add_argument("--accept-contact-volume-penetration-p95", type=float, default=0.05)
    parser.add_argument("--accept-world-speed-m-s", type=float, default=0.35)
    parser.add_argument("--accept-projection-inside-fraction", type=float, default=0.80)
    parser.add_argument("--max-nfev", type=int, default=70)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
