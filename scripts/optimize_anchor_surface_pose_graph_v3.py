#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FrameData:
    frame_idx: int
    T_world_camera: np.ndarray
    observed_points: np.ndarray
    observed_tree: cKDTree
    mask: np.ndarray
    mask_distance: np.ndarray
    depth_m: np.ndarray


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def load_archive(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    required = {"frame_idx", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = required.difference(blob.files)
    if missing:
        raise RuntimeError(f"{path} missing archive keys: {sorted(missing)}")
    frames = blob["frame_idx"].astype(int)
    vo = blob["vertex_offsets"].astype(np.int64)
    fo = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int32)
    out = {}
    for i, frame_idx in enumerate(frames):
        v = vertices[int(vo[i]) : int(vo[i + 1])]
        f = faces[int(fo[i]) : int(fo[i + 1])]
        if len(v) == 0 or len(f) == 0:
            raise RuntimeError(f"archive frame {frame_idx} has empty mesh")
        out[int(frame_idx)] = (v, f)
    return out


def annotation_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def manifest_by_frame(path: Path) -> dict[int, dict]:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain nonempty frames list")
    return {int(entry["frame_idx"]): entry for entry in frames}


def load_intrinsics(dataset: Path) -> np.ndarray:
    K = np.loadtxt(dataset / "cam_K.txt").astype(np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise RuntimeError(f"invalid intrinsics: {dataset / 'cam_K.txt'}")
    return K


def sample_rows(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    return points[rng.choice(len(points), size=int(max_points), replace=False)]


def transform(points: np.ndarray, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ Rotation.from_rotvec(rotvec).as_matrix().T + translation


def transform_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def project(points_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = z > 1e-5
    uv[valid, 0] = K[0, 0] * points_camera[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points_camera[valid, 1] / z[valid] + K[1, 2]
    return uv, z, valid


def mask_distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)


def plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - np.median(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    center = np.median(points, axis=0)
    if float(normal @ center) > 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return normal / norm


def rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / max(float(np.linalg.norm(a)), 1e-8)
    b = b / max(float(np.linalg.norm(b)), 1e-8)
    cross = np.cross(a, b)
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.0:
            return np.zeros(3, dtype=np.float64)
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(axis @ a)) > 0.9:
            axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - a * float(axis @ a)
        axis /= np.linalg.norm(axis)
        return axis * np.pi
    return Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.arccos(dot)).as_rotvec()


def load_frame_data(
    args: argparse.Namespace,
    archive: dict[int, tuple[np.ndarray, np.ndarray]],
    annotations: dict[int, dict],
    manifest: dict[int, dict],
) -> list[FrameData]:
    frames = []
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        if frame_idx not in archive:
            continue
        if frame_idx not in annotations:
            raise RuntimeError(f"missing annotation frame {frame_idx}")
        if frame_idx not in manifest:
            raise RuntimeError(f"missing manifest frame {frame_idx}")
        entry = manifest[frame_idx]
        observed, _ = archive[frame_idx]
        observed = sample_rows(observed, int(args.max_observed_points), int(args.seed) + frame_idx)
        mask = cv2.imread(str(entry["mask"]), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(entry["depth"]), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise RuntimeError(f"failed to read mask/depth for frame {frame_idx}")
        if mask.shape != depth.shape:
            raise RuntimeError(f"mask/depth shape mismatch for frame {frame_idx}")
        T = np.asarray(annotations[frame_idx]["camera"]["T_world_camera_metric"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise RuntimeError(f"invalid T_world_camera_metric for frame {frame_idx}")
        mask_bool = mask > 0
        frames.append(
            FrameData(
                frame_idx=frame_idx,
                T_world_camera=T,
                observed_points=observed,
                observed_tree=cKDTree(observed),
                mask=mask_bool,
                mask_distance=mask_distance(mask_bool),
                depth_m=depth.astype(np.float64) / 1000.0,
            )
        )
    if len(frames) < int(args.min_frames):
        raise RuntimeError(f"only {len(frames)} frames available")
    return frames


def initial_params(frames: list[FrameData], anchor_points: np.ndarray, anchor_frame: int) -> np.ndarray:
    anchor_center = np.median(anchor_points, axis=0)
    anchor_normal = plane_normal(anchor_points)
    params = np.zeros((len(frames), 6), dtype=np.float64)
    for i, frame in enumerate(frames):
        if frame.frame_idx == anchor_frame:
            continue
        center = np.median(frame.observed_points, axis=0)
        normal = plane_normal(frame.observed_points)
        params[i, :3] = rotation_align(anchor_normal, normal)
        rotated_center = transform(anchor_center[None, :], params[i, :3], np.zeros(3, dtype=np.float64))[0]
        params[i, 3:6] = center - rotated_center
    return params.reshape(-1)


def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = params.reshape(-1, 6)
    return pose[:, :3], pose[:, 3:6]


def projection_residuals(points_camera: np.ndarray, frame: FrameData, K: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    uv, z, valid_z = project(points_camera, K)
    height, width = frame.mask.shape
    in_image = valid_z & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
    x = np.clip(np.rint(np.nan_to_num(uv[:, 0], nan=0.0)).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(np.nan_to_num(uv[:, 1], nan=0.0)).astype(np.int32), 0, height - 1)
    sil = frame.mask_distance[y, x].astype(np.float64)
    sil[~in_image] = float(args.max_silhouette_px)
    sil = np.clip(sil, 0.0, float(args.max_silhouette_px)) / float(args.sigma_silhouette_px)
    valid_depth = in_image & frame.mask[y, x] & np.isfinite(frame.depth_m[y, x]) & (frame.depth_m[y, x] > 0.05)
    depth = np.zeros(len(points_camera), dtype=np.float64)
    raw_depth = z - frame.depth_m[y, x]
    depth[valid_depth] = np.clip(raw_depth[valid_depth], -args.max_depth_residual_m, args.max_depth_residual_m) / float(args.sigma_depth_m)
    return sil, depth


def residual_vector(
    params: np.ndarray,
    frames: list[FrameData],
    anchor_points: np.ndarray,
    depth_points: np.ndarray,
    K: np.ndarray,
    anchor_index: int,
    args: argparse.Namespace,
) -> np.ndarray:
    rotvecs, translations = unpack(params)
    residuals = []
    for i, frame in enumerate(frames):
        points = transform(anchor_points, rotvecs[i], translations[i])
        tree = cKDTree(points)
        d_obs_to_anchor, _ = tree.query(frame.observed_points, k=1)
        d_anchor_to_obs, _ = frame.observed_tree.query(points, k=1)
        residuals.append(np.clip(d_obs_to_anchor, 0.0, args.max_surface_residual_m) / float(args.sigma_observed_m))
        residuals.append(np.clip(d_anchor_to_obs, 0.0, args.max_surface_residual_m) / float(args.sigma_anchor_surface_m))
        proj_points = transform(depth_points, rotvecs[i], translations[i])
        sil, depth = projection_residuals(proj_points, frame, K, args)
        residuals.append(sil)
        residuals.append(depth)
    for i in range(1, len(frames)):
        residuals.append((translations[i] - translations[i - 1]) / float(args.sigma_translation_step_m))
        residuals.append((rotvecs[i] - rotvecs[i - 1]) / float(args.sigma_rotation_step_rad))
    for i in range(1, len(frames) - 1):
        residuals.append((translations[i + 1] - 2.0 * translations[i] + translations[i - 1]) / float(args.sigma_translation_accel_m))
        residuals.append((rotvecs[i + 1] - 2.0 * rotvecs[i] + rotvecs[i - 1]) / float(args.sigma_rotation_accel_rad))
    residuals.append(translations[anchor_index] / float(args.sigma_anchor_translation_m))
    residuals.append(rotvecs[anchor_index] / float(args.sigma_anchor_rotation_rad))
    return np.concatenate([part.reshape(-1) for part in residuals])


def residual_sparsity(frames: list[FrameData], anchor_points: np.ndarray, depth_points: np.ndarray, anchor_index: int) -> sparse.csr_matrix:
    n = len(frames)
    cols = n * 6
    entries: list[tuple[int, int]] = []
    row = 0
    for i, frame in enumerate(frames):
        c = range(i * 6, (i + 1) * 6)
        for count in (len(frame.observed_points), len(anchor_points), len(depth_points), len(depth_points)):
            for r in range(row, row + int(count)):
                for col in c:
                    entries.append((r, col))
            row += int(count)
    for i in range(1, n):
        c = range((i - 1) * 6, (i + 1) * 6)
        for count in (3, 3):
            for r in range(row, row + count):
                for col in c:
                    entries.append((r, col))
            row += count
    for i in range(1, n - 1):
        c = range((i - 1) * 6, (i + 2) * 6)
        for count in (3, 3):
            for r in range(row, row + count):
                for col in c:
                    entries.append((r, col))
            row += count
    c = range(anchor_index * 6, (anchor_index + 1) * 6)
    for count in (3, 3):
        for r in range(row, row + count):
            for col in c:
                entries.append((r, col))
        row += count
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols))


def frame_metrics(
    params: np.ndarray,
    frames: list[FrameData],
    anchor_points: np.ndarray,
    depth_points: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict]:
    rotvecs, translations = unpack(params)
    rows = {}
    for i, frame in enumerate(frames):
        points = transform(anchor_points, rotvecs[i], translations[i])
        tree = cKDTree(points)
        d_obs, _ = tree.query(frame.observed_points, k=1)
        d_anchor, _ = frame.observed_tree.query(points, k=1)
        proj_points = transform(depth_points, rotvecs[i], translations[i])
        sil, depth = projection_residuals(proj_points, frame, K, args)
        rows[str(frame.frame_idx)] = {
            "observed_to_anchor_median_m": float(np.median(d_obs)),
            "observed_to_anchor_p95_m": float(np.percentile(d_obs, 95.0)),
            "anchor_to_observed_median_m": float(np.median(d_anchor)),
            "anchor_to_observed_p95_m": float(np.percentile(d_anchor, 95.0)),
            "silhouette_outside_median_px": float(np.median(sil) * args.sigma_silhouette_px),
            "silhouette_outside_p95_px": float(np.percentile(sil, 95.0) * args.sigma_silhouette_px),
            "depth_residual_abs_median_m": float(np.median(np.abs(depth)) * args.sigma_depth_m),
            "depth_residual_abs_p95_m": float(np.percentile(np.abs(depth), 95.0) * args.sigma_depth_m),
            "translation_m": translations[i].astype(float).tolist(),
            "rotation_rad": rotvecs[i].astype(float).tolist(),
            "camera_extent_m": (points.max(axis=0) - points.min(axis=0)).astype(float).tolist(),
        }
    return rows


def summarize(rows: dict[str, dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows.values()], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def save_world_archive(path: Path, frames: list[FrameData], anchor_vertices: np.ndarray, faces: np.ndarray, params: np.ndarray) -> None:
    rotvecs, translations = unpack(params)
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_all = []
    faces_all = []
    for i, frame in enumerate(frames):
        camera_vertices = transform(anchor_vertices, rotvecs[i], translations[i])
        world_vertices = transform_world(camera_vertices, frame.T_world_camera)
        vertices_all.append(world_vertices.astype(np.float32))
        faces_all.append(faces.astype(np.int32))
        vertex_offsets.append(vertex_offsets[-1] + len(world_vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    np.savez_compressed(
        path,
        frame_idx=np.asarray([frame.frame_idx for frame in frames], dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=np.vstack(vertices_all).astype(np.float32),
        faces=np.vstack(faces_all).astype(np.int32),
    )


def run(args: argparse.Namespace) -> dict:
    archive = load_archive(args.observed_mesh_npz)
    if int(args.anchor_frame) not in archive:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from observed archive")
    annotations = annotation_by_frame(args.annotations)
    manifest = manifest_by_frame(args.manifest)
    frames = load_frame_data(args, archive, annotations, manifest)
    anchor_index = next((i for i, frame in enumerate(frames) if frame.frame_idx == int(args.anchor_frame)), None)
    if anchor_index is None:
        raise RuntimeError(f"anchor frame {args.anchor_frame} absent from selected frames")
    anchor_vertices, anchor_faces = archive[int(args.anchor_frame)]
    anchor_points = sample_rows(anchor_vertices, int(args.max_anchor_points), int(args.seed) + 101)
    depth_points = sample_rows(anchor_vertices, int(args.max_depth_points), int(args.seed) + 202)
    K = load_intrinsics(args.dataset)
    x0 = initial_params(frames, anchor_points, int(args.anchor_frame))
    x0[anchor_index * 6 : (anchor_index + 1) * 6] = 0.0
    before = residual_vector(x0, frames, anchor_points, depth_points, K, anchor_index, args)
    pattern = residual_sparsity(frames, anchor_points, depth_points, anchor_index)
    result = least_squares(
        lambda x: residual_vector(x, frames, anchor_points, depth_points, K, anchor_index, args),
        x0,
        jac_sparsity=pattern,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after = residual_vector(result.x, frames, anchor_points, depth_points, K, anchor_index, args)
    before_rows = frame_metrics(x0, frames, anchor_points, depth_points, K, args)
    after_rows = frame_metrics(result.x, frames, anchor_points, depth_points, K, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "anchor_surface_pose_graph_object_meshes_world.npz"
    save_world_archive(archive_path, frames, anchor_vertices, anchor_faces, result.x)
    report = {
        "status": "ok" if result.success else "optimizer_incomplete",
        "annotation_ready": False,
        "method": "anchor_surface_camera_pose_graph_v3",
        "observed_mesh_npz": str(args.observed_mesh_npz),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "anchor_frame": int(args.anchor_frame),
        "used_frames": [int(frame.frame_idx) for frame in frames],
        "mesh_archive": str(archive_path),
        "variables": int(len(result.x)),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "residual_rms_before": float(np.sqrt(np.mean(before * before))),
        "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        "before_summary": {
            key: summarize(before_rows, key)
            for key in (
                "observed_to_anchor_median_m",
                "anchor_to_observed_median_m",
                "silhouette_outside_median_px",
                "depth_residual_abs_median_m",
            )
        },
        "after_summary": {
            key: summarize(after_rows, key)
            for key in (
                "observed_to_anchor_median_m",
                "anchor_to_observed_median_m",
                "silhouette_outside_median_px",
                "depth_residual_abs_median_m",
            )
        },
        "frame_metrics_before": before_rows,
        "frame_metrics_after": after_rows,
    }
    (args.output_dir / "qc_anchor_surface_pose_graph_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"frame_metrics_before", "frame_metrics_after"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-mesh-npz", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--anchor-frame", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-observed-points", type=int, default=1500)
    parser.add_argument("--max-anchor-points", type=int, default=1500)
    parser.add_argument("--max-depth-points", type=int, default=1200)
    parser.add_argument("--sigma-observed-m", type=float, default=0.020)
    parser.add_argument("--sigma-anchor-surface-m", type=float, default=0.025)
    parser.add_argument("--sigma-depth-m", type=float, default=0.018)
    parser.add_argument("--sigma-silhouette-px", type=float, default=4.0)
    parser.add_argument("--sigma-translation-step-m", type=float, default=0.060)
    parser.add_argument("--sigma-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--sigma-translation-accel-m", type=float, default=0.040)
    parser.add_argument("--sigma-rotation-accel-rad", type=float, default=0.22)
    parser.add_argument("--sigma-anchor-translation-m", type=float, default=0.010)
    parser.add_argument("--sigma-anchor-rotation-rad", type=float, default=0.060)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.12)
    parser.add_argument("--max-depth-residual-m", type=float, default=0.18)
    parser.add_argument("--max-silhouette-px", type=float, default=80.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
