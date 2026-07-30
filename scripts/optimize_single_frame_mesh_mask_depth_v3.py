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
class FrameEvidence:
    frame_idx: int
    mask: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    observed_points_camera: np.ndarray
    mask_pixels: np.ndarray
    outside_distance_px: np.ndarray


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


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
        "max": float(np.max(arr)),
    }


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"{path} scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"{path} did not load as a triangle mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"{path} contains no 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"{path} contains no triangular faces")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError(f"{path} face indices are outside the vertex array")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


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
        frame_intrinsics = intrinsics[i]
        if frame_depth.ndim != 2 or not np.isfinite(frame_depth).all():
            raise RuntimeError(f"{path} frame {idx} has invalid depth")
        if frame_intrinsics.shape != (4,) or not np.isfinite(frame_intrinsics).all():
            raise RuntimeError(f"{path} frame {idx} has invalid intrinsics")
        out[int(idx)] = (frame_depth, frame_intrinsics)
    return out


def frame_from_list(path: Path, frame_idx: int) -> dict:
    frames = load_json(path).get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"{path} must contain a nonempty frames list")
    selected = [frame for frame in frames if int(frame["frame_idx"]) == int(frame_idx)]
    if len(selected) != 1:
        raise RuntimeError(f"frame {frame_idx} appears {len(selected)} times in {path}")
    return selected[0]


def manifest_frame(path: Path, frame_idx: int) -> dict:
    return frame_from_list(path, frame_idx)


def intrinsics_for(annotation: dict, depth_intrinsics: np.ndarray, source: str) -> np.ndarray:
    if source == "annotation-vggt":
        intrinsics = np.asarray(annotation.get("camera", {}).get("vggt_source_intrinsics_fx_fy_cx_cy", []), dtype=np.float64)
    elif source == "metric-depth":
        intrinsics = np.asarray(depth_intrinsics, dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported intrinsics source {source}")
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise RuntimeError(f"invalid {source} intrinsics for frame {annotation.get('frame_idx')}")
    return intrinsics


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


def project(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    z = points_camera[:, 2].astype(np.float64)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    positive = z > 1e-6
    uv[positive, 0] = fx * points_camera[positive, 0] / z[positive] + cx
    uv[positive, 1] = fy * points_camera[positive, 1] / z[positive] + cy
    return uv, positive


def backproject_mask(mask: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError("mask has no valid depth samples")
    coords = np.c_[xs, ys].astype(np.float64)
    if len(coords) > int(max_points):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(len(coords), size=int(max_points), replace=False)
        coords = coords[take]
        xs = coords[:, 0].astype(np.int64)
        ys = coords[:, 1].astype(np.int64)
    z = depth_m[ys, xs].astype(np.float64)
    fx, fy, cx, cy = intrinsics
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points = np.c_[x, y, z]
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.0):
        raise RuntimeError("backprojected mask-depth points are invalid")
    return points, coords.astype(np.float64)


def load_frame_evidence(args: argparse.Namespace) -> FrameEvidence:
    manifest = manifest_frame(args.manifest, int(args.frame_idx))
    annotation = frame_from_list(args.annotations, int(args.frame_idx))
    depths = load_depth_archive(args.metric_depth_npz)
    if int(args.frame_idx) not in depths:
        raise RuntimeError(f"{args.metric_depth_npz} lacks frame {args.frame_idx}")
    depth_m, depth_intrinsics = depths[int(args.frame_idx)]
    mask = cv2.imread(str(manifest["mask"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read mask {manifest['mask']}")
    if mask.shape != depth_m.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0
    intrinsics = intrinsics_for(annotation, depth_intrinsics, args.intrinsics_source)
    T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
    if T_world_camera.shape != (4, 4) or not np.isfinite(T_world_camera).all():
        raise RuntimeError(f"frame {args.frame_idx} has invalid T_world_camera_metric")
    observed_points, mask_pixels = backproject_mask(mask_bool, depth_m, intrinsics, int(args.max_observed_points), int(args.seed) + 1000)
    coverage_pixels = sample_rows(mask_pixels, int(args.max_mask_pixels), int(args.seed) + 2000)
    outside_distance = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
    return FrameEvidence(
        frame_idx=int(args.frame_idx),
        mask=mask_bool,
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_world_camera=T_world_camera,
        observed_points_camera=observed_points,
        mask_pixels=coverage_pixels,
        outside_distance_px=outside_distance,
    )


def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    axes = vh.astype(np.float64)
    if np.linalg.det(axes) < 0.0:
        axes[-1] *= -1.0
    return center, axes


def robust_extent(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    extent = np.percentile(points, 95.0, axis=0) - np.percentile(points, 5.0, axis=0)
    if extent.shape != (3,) or np.any(~np.isfinite(extent)) or np.any(extent <= 1e-8):
        raise RuntimeError("point extent is degenerate")
    return extent.astype(np.float64)


def transform(points: np.ndarray, pivot: np.ndarray, params: np.ndarray) -> np.ndarray:
    rotvec = params[:3]
    translation = params[3:6]
    scale = float(np.exp(params[6]))
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return scale * ((points - pivot) @ rotation.T) + translation


def in_image(uv: np.ndarray, positive: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    finite = np.isfinite(uv).all(axis=1)
    return positive & finite & (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)


def normalized_block(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise RuntimeError("cannot normalize an empty residual block")
    return values / np.sqrt(float(len(values)))


def residual_vector(
    params: np.ndarray,
    prior_surface: np.ndarray,
    prior_projection: np.ndarray,
    pivot: np.ndarray,
    frame: FrameEvidence,
    log_scale_prior: float,
    args: argparse.Namespace,
) -> np.ndarray:
    surface = transform(prior_surface, pivot, params)
    projection_points = transform(prior_projection, pivot, params)
    residuals = []

    surface_tree = cKDTree(surface)
    d_observed, _ = surface_tree.query(frame.observed_points_camera, k=1)
    residuals.append(normalized_block(np.clip(d_observed, 0.0, float(args.max_surface_residual_m)) / float(args.sigma_surface_m)))

    uv, positive = project(projection_points, frame.intrinsics)
    in_bounds = in_image(uv, positive, frame.mask.shape)
    safe_uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.rint(safe_uv).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, frame.mask.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, frame.mask.shape[0] - 1)
    outside = frame.outside_distance_px[rounded[:, 1], rounded[:, 0]]
    outside[~in_bounds] = float(args.max_projection_residual_px)
    residuals.append(normalized_block(np.clip(outside, 0.0, float(args.max_projection_residual_px)) / float(args.sigma_projection_px)))

    projected_valid = uv[in_bounds]
    if len(projected_valid) < int(args.min_projected_points):
        coverage = np.full(len(frame.mask_pixels), float(args.max_coverage_residual_px), dtype=np.float64)
    else:
        coverage, _ = cKDTree(projected_valid).query(frame.mask_pixels, k=1)
    residuals.append(normalized_block(np.clip(coverage, 0.0, float(args.max_coverage_residual_px)) / float(args.sigma_coverage_px)))

    mask_hit = np.zeros(len(projection_points), dtype=bool)
    if np.any(in_bounds):
        mask_hit[in_bounds] = frame.mask[rounded[in_bounds, 1], rounded[in_bounds, 0]]
    valid_depth = in_bounds & mask_hit
    front = np.zeros(len(projection_points), dtype=np.float64)
    if np.any(valid_depth):
        measured = frame.depth_m[rounded[valid_depth, 1], rounded[valid_depth, 0]]
        raw_front = measured - projection_points[valid_depth, 2] - float(args.depth_front_tolerance_m)
        front[valid_depth] = np.maximum(0.0, raw_front)
    residuals.append(normalized_block(np.clip(front, 0.0, float(args.max_front_depth_residual_m)) / float(args.sigma_front_depth_m)))

    residuals.append(np.asarray([(params[6] - log_scale_prior) / float(args.sigma_log_scale)], dtype=np.float64))
    return np.concatenate(residuals)


def initial_params(mesh_points: np.ndarray, observed_points: np.ndarray, pivot: np.ndarray, scale0: float, args: argparse.Namespace) -> list[np.ndarray]:
    prior_center, prior_axes = pca_axes(mesh_points)
    observed_center, observed_axes = pca_axes(observed_points)
    starts = []
    signs_list = (
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    )
    scale_factors = np.asarray([float(x) for x in str(args.scale_factors).split(",")], dtype=np.float64)
    if scale_factors.ndim != 1 or len(scale_factors) == 0 or np.any(~np.isfinite(scale_factors)) or np.any(scale_factors <= 0.0):
        raise RuntimeError("--scale-factors must contain positive comma-separated floats")
    for signs in signs_list:
        sign = np.diag(np.asarray(signs, dtype=np.float64))
        rotation = observed_axes.T @ sign @ prior_axes
        if np.linalg.det(rotation) < 0.0:
            continue
        for factor in scale_factors:
            scale = float(scale0 * factor)
            translation = observed_center - scale * ((prior_center - pivot) @ rotation.T)
            starts.append(np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation, np.log(scale)])
    if not starts:
        raise RuntimeError("no valid initial pose candidates")
    return starts


def world_points(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points_camera, np.ones(len(points_camera), dtype=np.float64)]
    return (T_world_camera @ homog.T).T[:, :3]


def save_world_archive(path: Path, frame: FrameEvidence, vertices_camera: np.ndarray, faces: np.ndarray) -> None:
    vertices_world = world_points(vertices_camera, frame.T_world_camera)
    np.savez_compressed(
        path,
        frame_idx=np.asarray([frame.frame_idx], dtype=np.int32),
        vertex_offsets=np.asarray([0, len(vertices_world)], dtype=np.int64),
        face_offsets=np.asarray([0, len(faces)], dtype=np.int64),
        vertices=vertices_world.astype(np.float32),
        faces=faces.astype(np.int32),
    )


def write_filtered_manifest(path: Path, manifest_path: Path, frame_idx: int) -> None:
    selected = manifest_frame(manifest_path, frame_idx)
    path.write_text(json.dumps({"frames": [selected]}, indent=2), encoding="utf-8")


def metrics(params: np.ndarray, prior_surface: np.ndarray, prior_projection: np.ndarray, pivot: np.ndarray, frame: FrameEvidence) -> dict:
    surface = transform(prior_surface, pivot, params)
    projection_points = transform(prior_projection, pivot, params)
    d_observed, _ = cKDTree(surface).query(frame.observed_points_camera, k=1)
    uv, positive = project(projection_points, frame.intrinsics)
    in_bounds = in_image(uv, positive, frame.mask.shape)
    safe_uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.rint(safe_uv).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, frame.mask.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, frame.mask.shape[0] - 1)
    mask_hit = np.zeros(len(projection_points), dtype=bool)
    if np.any(in_bounds):
        mask_hit[in_bounds] = frame.mask[rounded[in_bounds, 1], rounded[in_bounds, 0]]
    projected_valid = uv[in_bounds]
    if len(projected_valid) == 0:
        coverage = np.full(len(frame.mask_pixels), np.nan, dtype=np.float64)
    else:
        coverage, _ = cKDTree(projected_valid).query(frame.mask_pixels, k=1)
    depth_error = np.full(len(projection_points), np.nan, dtype=np.float64)
    valid_depth = in_bounds & mask_hit
    if np.any(valid_depth):
        measured = frame.depth_m[rounded[valid_depth, 1], rounded[valid_depth, 0]]
        depth_error[valid_depth] = projection_points[valid_depth, 2] - measured
    return {
        "observed_to_prior_surface_m": summarize(d_observed),
        "mask_to_projected_mesh_coverage_px": summarize(coverage),
        "projected_in_bounds_fraction": float(np.mean(in_bounds)),
        "projected_inside_mask_fraction": float(np.mean(mask_hit[in_bounds])) if np.any(in_bounds) else 0.0,
        "inside_mask_depth_error_m": summarize(depth_error[valid_depth]),
        "inside_mask_abs_depth_error_m": summarize(np.abs(depth_error[valid_depth])),
        "translation_camera_m": params[3:6].astype(float).tolist(),
        "rotation_camera_rotvec": params[:3].astype(float).tolist(),
        "scale": float(np.exp(params[6])),
    }


def run(args: argparse.Namespace) -> dict:
    mesh = load_mesh(args.mesh_prior)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
    pivot = np.median(mesh_vertices, axis=0)
    frame = load_frame_evidence(args)
    prior_surface = sample_mesh_surface(mesh, int(args.max_prior_surface_points), int(args.seed) + 3000)
    prior_projection = sample_mesh_surface(mesh, int(args.max_projection_points), int(args.seed) + 4000)
    prior_extent = robust_extent(prior_surface)
    observed_extent = robust_extent(frame.observed_points_camera)
    scale0 = float(np.max(observed_extent) / np.max(prior_extent))
    if not np.isfinite(scale0) or scale0 <= 0.0:
        raise RuntimeError("computed invalid scale prior")
    log_scale_prior = float(np.log(scale0))
    starts = initial_params(prior_surface, frame.observed_points_camera, pivot, scale0, args)
    lower = np.asarray(
        [
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            max(float(args.min_depth_m), float(np.percentile(frame.observed_points_camera[:, 2], 5.0)) - float(args.translation_z_margin_m)),
            np.log(scale0 * float(args.min_scale_factor)),
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            float(np.percentile(frame.observed_points_camera[:, 2], 95.0)) + float(args.translation_z_margin_m),
            np.log(scale0 * float(args.max_scale_factor)),
        ],
        dtype=np.float64,
    )
    before_records = []
    best = None
    for start_i, start in enumerate(starts):
        clipped = np.minimum(np.maximum(start, lower + 1e-9), upper - 1e-9)
        before = residual_vector(clipped, prior_surface, prior_projection, pivot, frame, log_scale_prior, args)
        result = least_squares(
            lambda x: residual_vector(x, prior_surface, prior_projection, pivot, frame, log_scale_prior, args),
            clipped,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=int(args.max_nfev),
            verbose=0,
        )
        after = residual_vector(result.x, prior_surface, prior_projection, pivot, frame, log_scale_prior, args)
        record = {
            "start_index": int(start_i),
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "residual_rms_before": float(np.sqrt(np.mean(before * before))),
            "residual_rms_after": float(np.sqrt(np.mean(after * after))),
            "scale_start": float(np.exp(clipped[6])),
            "scale_after": float(np.exp(result.x[6])),
        }
        before_records.append(record)
        score = record["residual_rms_after"]
        if best is None or score < best[0]:
            best = (score, result, record)
    if best is None:
        raise RuntimeError("optimizer produced no candidates")
    _score, result, selected_record = best

    vertices_camera = transform(mesh_vertices, pivot, result.x)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_camera = trimesh.Trimesh(vertices=vertices_camera, faces=mesh_faces, process=False)
    camera_mesh_path = args.output_dir / f"aligned_mask_depth_frame_{frame.frame_idx:06d}.obj"
    mesh_camera.export(camera_mesh_path)
    world_archive = args.output_dir / f"world_mesh_frame_{frame.frame_idx:06d}.npz"
    save_world_archive(world_archive, frame, vertices_camera, mesh_faces)
    manifest_out = args.output_dir / f"manifest_frame_{frame.frame_idx:06d}.json"
    write_filtered_manifest(manifest_out, args.manifest, frame.frame_idx)
    after_metrics = metrics(result.x, prior_surface, prior_projection, pivot, frame)
    vertex_extent = vertices_camera.max(axis=0) - vertices_camera.min(axis=0)
    vertex_robust_extent = robust_extent(vertices_camera)
    report = {
        "status": "ok" if bool(result.success) else "optimizer_incomplete",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "optimize_single_frame_mesh_mask_depth_v3",
        "claim_tested": "whether the raw complete mesh can satisfy one-frame mask, metric depth, and observed surface evidence under a bounded Sim3 camera pose",
        "mesh_prior": str(args.mesh_prior),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "frame_idx": int(frame.frame_idx),
        "camera_mesh": str(camera_mesh_path),
        "world_archive": str(world_archive),
        "filtered_manifest": str(manifest_out),
        "prior_vertices": int(len(mesh_vertices)),
        "prior_faces": int(len(mesh_faces)),
        "prior_robust_extent_model": prior_extent.astype(float).tolist(),
        "observed_robust_extent_camera_m": observed_extent.astype(float).tolist(),
        "data_derived_scale_prior": float(scale0),
        "scale_bounds": [float(np.exp(lower[6])), float(np.exp(upper[6]))],
        "translation_z_bounds_m": [float(lower[5]), float(upper[5])],
        "selected_start": selected_record,
        "candidate_records": before_records,
        "metrics_after": after_metrics,
        "center_camera_m": np.median(vertices_camera, axis=0).astype(float).tolist(),
        "extent_camera_m": vertex_extent.astype(float).tolist(),
        "robust_extent_camera_m": vertex_robust_extent.astype(float).tolist(),
    }
    (args.output_dir / "qc_single_frame_mesh_mask_depth_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_records"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-prior", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--intrinsics-source", choices=["annotation-vggt", "metric-depth"], default="annotation-vggt")
    parser.add_argument("--max-observed-points", type=int, default=2500)
    parser.add_argument("--max-mask-pixels", type=int, default=2500)
    parser.add_argument("--max-prior-surface-points", type=int, default=2500)
    parser.add_argument("--max-projection-points", type=int, default=2500)
    parser.add_argument("--min-projected-points", type=int, default=40)
    parser.add_argument("--sigma-surface-m", type=float, default=0.012)
    parser.add_argument("--sigma-projection-px", type=float, default=4.0)
    parser.add_argument("--sigma-coverage-px", type=float, default=6.0)
    parser.add_argument("--sigma-front-depth-m", type=float, default=0.020)
    parser.add_argument("--sigma-log-scale", type=float, default=0.32)
    parser.add_argument("--depth-front-tolerance-m", type=float, default=0.006)
    parser.add_argument("--max-surface-residual-m", type=float, default=0.080)
    parser.add_argument("--max-projection-residual-px", type=float, default=80.0)
    parser.add_argument("--max-coverage-residual-px", type=float, default=80.0)
    parser.add_argument("--max-front-depth-residual-m", type=float, default=0.080)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--translation-z-margin-m", type=float, default=0.25)
    parser.add_argument("--min-scale-factor", type=float, default=0.25)
    parser.add_argument("--max-scale-factor", type=float, default=3.0)
    parser.add_argument("--scale-factors", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
