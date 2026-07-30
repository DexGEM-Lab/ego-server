#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
import yaml  # type: ignore[reportMissingImports]


STATUS = "v17_geometry_reconstruction_results_qc"
CLAIM = (
    "This artifact evaluates hidden-topology solver outputs for the V17 RGBD reconstruction jobs. "
    "It accepts only solver outputs whose mesh, pose sequence, object-scale, projection, depth, and "
    "surface-topology evidence are compatible with the rectified RGBD job. It is a solver-output QC "
    "layer, not full active-interval object-pose closure."
)

FALSE_READY = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be a finite number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be a finite number")
    return out


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read grayscale image: {path}")
    return image


def read_depth_m(path: Path) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise RuntimeError(f"failed to read depth image: {path}")
    if depth_mm.ndim != 2:
        raise RuntimeError(f"depth image must be single-channel: {path}")
    return depth_mm.astype(np.float64) / 1000.0


def image_points_to_xyz(u: np.ndarray, v: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intrinsics.tolist()]
    z = depth.astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intrinsics.tolist()]
    z = points[:, 2]
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 0.0)
    out[valid, 0] = fx * points[valid, 0] / z[valid] + cx
    out[valid, 1] = fy * points[valid, 1] / z[valid] + cy
    return out


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (hom @ matrix.T)[:, :3]


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise RuntimeError(f"invalid vertex row in {path}: {line.strip()}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for part in line.split()[1:]:
                    raw = part.split("/")[0]
                    if not raw:
                        raise RuntimeError(f"invalid face row in {path}: {line.strip()}")
                    value = int(raw)
                    idx.append(value - 1 if value > 0 else len(vertices) + value)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    faces.extend([[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)])
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise RuntimeError(f"{path} contains no valid vertices")
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise RuntimeError(f"{path} contains no valid triangular faces")
    if np.any(tri < 0) or np.any(tri >= len(verts)):
        raise RuntimeError(f"{path} contains face indices outside the vertex array")
    return verts, tri


def mesh_extent(vertices: np.ndarray) -> np.ndarray:
    return np.ptp(vertices, axis=0)


def mesh_topology_stats(faces: np.ndarray) -> dict[str, Any]:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    packed = edges[:, 0].astype(np.int64) << np.int64(32)
    packed = packed | edges[:, 1].astype(np.int64)
    unique, counts = np.unique(packed, return_counts=True)
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    total = int(len(unique))
    return {
        "unique_edge_count": total,
        "boundary_edge_count": boundary,
        "nonmanifold_edge_count": nonmanifold,
        "boundary_edge_fraction": float(boundary / total) if total else 1.0,
        "nonmanifold_edge_fraction": float(nonmanifold / total) if total else 1.0,
    }


def candidate_mesh_path(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "textured_mesh.obj",
        output_dir / "mesh" / "mesh_real_scale.obj",
        output_dir / "mesh" / "mesh_biggest_component_smoothed.obj",
        output_dir / "mesh" / "mesh_biggest_component.obj",
        output_dir / "mesh_cleaned.obj",
    ]
    return next((path for path in candidates if path.exists()), None)


def write_obj_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in vertices:
            f.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
        for a, b, c in faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"BundleSDF normalization file must be a mapping: {path}")
    return data


def bundlesdf_normalization(output_dir: Path) -> tuple[Path, float, np.ndarray]:
    candidates = [
        output_dir / "final" / "nerf" / "config.yml",
        output_dir / "final" / "nerf" / "normalization.yml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        data = read_yaml_mapping(path)
        sc_factor = data.get("sc_factor")
        translation = data.get("translation", data.get("translation_cvcam"))
        if sc_factor is None or translation is None:
            continue
        sc = finite_float(sc_factor, f"{path} sc_factor")
        if sc <= 0.0:
            raise RuntimeError(f"BundleSDF sc_factor must be positive: {path}")
        tr = np.asarray(translation, dtype=np.float64)
        if tr.shape != (3,) or not np.isfinite(tr).all():
            raise RuntimeError(f"BundleSDF translation must be a finite 3-vector: {path}")
        return path, sc, tr
    raise RuntimeError(f"missing BundleSDF normalization config for normalized mesh: {output_dir}")


def metric_mesh_for_qc(
    output_dir: Path,
    source_mesh_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    metric_mesh_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_extent = mesh_extent(vertices)
    if source_mesh_path.name != "mesh_cleaned.obj":
        return vertices, {
            "source_mesh_path": str(source_mesh_path),
            "mesh_coordinate_contract": "bundlesdf_metric_mesh",
            "metric_mesh_path": str(source_mesh_path),
            "source_mesh_extent_native_units": [float(v) for v in source_extent.tolist()],
            "metric_transform_applied": False,
        }
    normalization_path, sc_factor, translation = bundlesdf_normalization(output_dir)
    metric_vertices = vertices / sc_factor - translation.reshape(1, 3)
    write_obj_mesh(metric_mesh_path, metric_vertices, faces)
    return metric_vertices, {
        "source_mesh_path": str(source_mesh_path),
        "mesh_coordinate_contract": "bundlesdf_mesh_cleaned_normalized_before_texture_stage",
        "metric_mesh_path": str(metric_mesh_path),
        "source_mesh_extent_native_units": [float(v) for v in source_extent.tolist()],
        "metric_transform_applied": True,
        "metric_transform_source": str(normalization_path),
        "bundlesdf_sc_factor": float(sc_factor),
        "bundlesdf_translation_cvcam": [float(v) for v in translation.tolist()],
        "bundlesdf_pose_offset_applied": False,
        "bundlesdf_pose_offset_persisted": False,
        "coordinate_contract_note": (
            "BundleSDF exports mesh_cleaned.obj in normalized NeRF coordinates before the texture-stage "
            "mesh_to_real_world conversion. V17 applies the persisted scale and translation only, then "
            "requires projection and depth QC before accepting the recovered metric mesh."
        ),
    }


def observed_extent_rows(frames: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(frames):
        frame = require_dict(raw, f"job frames[{i}]")
        mask = read_gray(Path(require_str(frame.get("mask"), f"job frames[{i}].mask"))) > 0
        depth = read_depth_m(Path(require_str(frame.get("depth"), f"job frames[{i}].depth")))
        if mask.shape != depth.shape:
            raise RuntimeError(f"job frame {i} mask/depth shape mismatch")
        valid = mask & np.isfinite(depth) & (depth > 0.05)
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            raise RuntimeError(f"job frame {i} has no valid masked depth")
        if len(xs) > max_points:
            selected = np.linspace(0, len(xs) - 1, max_points, dtype=np.int64)
            xs = xs[selected]
            ys = ys[selected]
        intrinsics = np.asarray(frame.get("rectified_intrinsics_fx_fy_cx_cy"), dtype=np.float64)
        if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
            raise RuntimeError(f"job frame {i} has invalid rectified intrinsics")
        pts = image_points_to_xyz(xs.astype(np.float64), ys.astype(np.float64), depth[ys, xs], intrinsics)
        q = np.percentile(pts, [5.0, 95.0], axis=0)
        extent = q[1] - q[0]
        rows.append(
            {
                "frame_idx": require_int(frame.get("frame_idx"), f"job frames[{i}].frame_idx"),
                "sample_count": int(len(pts)),
                "extent_p05_p95_m": [float(v) for v in extent.tolist()],
                "max_extent_p05_p95_m": float(np.max(extent)),
            }
        )
    return rows


def render_silhouette(
    shape: tuple[int, int],
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    height, width = shape
    silhouette = np.zeros((height, width), dtype=np.uint8)
    valid = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid)
    if max_faces > 0 and len(face_ids) > max_faces:
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, max_faces, dtype=np.int64)]
    order = np.argsort(z[faces[face_ids]].mean(axis=1))[::-1]
    for face_id in face_ids[order]:
        poly = uv[faces[int(face_id)]]
        if np.any(poly[:, 0] < -width) or np.any(poly[:, 0] > 2 * width):
            continue
        if np.any(poly[:, 1] < -height) or np.any(poly[:, 1] > 2 * height):
            continue
        cv2.fillConvexPoly(silhouette, np.round(poly).astype(np.int32), 255, cv2.LINE_AA)
    return silhouette > 0


def rasterized_front_depth_errors(
    shape: tuple[int, int],
    uv: np.ndarray,
    z: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    max_faces: int,
) -> np.ndarray:
    height, width = shape
    valid = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(z[faces] > 0.0, axis=1)
    face_ids = np.flatnonzero(valid)
    if len(face_ids) == 0:
        return np.asarray([], dtype=np.float64)
    if max_faces > 0 and len(face_ids) > max_faces:
        face_ids = face_ids[np.linspace(0, len(face_ids) - 1, max_faces, dtype=np.int64)]
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    order = np.argsort(z[faces[face_ids]].mean(axis=1))[::-1]
    for face_id in face_ids[order]:
        face = faces[int(face_id)]
        poly = uv[face]
        if np.any(poly[:, 0] < -width) or np.any(poly[:, 0] > 2 * width):
            continue
        if np.any(poly[:, 1] < -height) or np.any(poly[:, 1] > 2 * height):
            continue
        cv2.fillConvexPoly(zbuf, np.round(poly).astype(np.int32), float(np.min(z[face])), cv2.LINE_AA)
    keep = mask & np.isfinite(zbuf) & np.isfinite(depth) & (depth > 0.05)
    if not np.any(keep):
        return np.asarray([], dtype=np.float64)
    return zbuf[keep].astype(np.float64) - depth[keep]


def projection_rows(
    frames: list[dict[str, Any]],
    output_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pose_dir = output_dir / "ob_in_cam"
    for i, raw in enumerate(frames):
        frame = require_dict(raw, f"job frames[{i}]")
        index = require_int(frame.get("index"), f"job frames[{i}].index")
        pose_path = pose_dir / f"{index:06d}.txt"
        if not pose_path.exists():
            raise RuntimeError(f"missing BundleSDF pose file: {pose_path}")
        pose = np.loadtxt(pose_path).astype(np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RuntimeError(f"BundleSDF pose must be finite 4x4: {pose_path}")
        mask = read_gray(Path(require_str(frame.get("mask"), f"job frames[{i}].mask"))) > 0
        depth = read_depth_m(Path(require_str(frame.get("depth"), f"job frames[{i}].depth")))
        if mask.shape != depth.shape:
            raise RuntimeError(f"job frame {i} mask/depth shape mismatch")
        intrinsics = np.asarray(frame.get("rectified_intrinsics_fx_fy_cx_cy"), dtype=np.float64)
        cam_vertices = transform_points(vertices, pose)
        uv = project(cam_vertices, intrinsics)
        projectable_faces = np.all(np.isfinite(uv[faces]), axis=(1, 2)) & np.all(cam_vertices[:, 2][faces] > 0.0, axis=1)
        projectable_face_count = int(np.count_nonzero(projectable_faces))
        face_limit = int(args.max_projection_faces)
        rasterized_face_count = (
            min(projectable_face_count, face_limit) if face_limit > 0 else projectable_face_count
        )
        silhouette = render_silhouette(mask.shape, uv, cam_vertices[:, 2], faces, int(args.max_projection_faces))
        intersection = int(np.count_nonzero(silhouette & mask))
        union = int(np.count_nonzero(silhouette | mask))
        if union == 0:
            raise RuntimeError(f"frame {frame.get('frame_idx')} has empty mask/silhouette union")
        errors = rasterized_front_depth_errors(
            mask.shape, uv, cam_vertices[:, 2], faces, mask, depth, int(args.max_projection_faces)
        )
        rows.append(
            {
                "frame_idx": require_int(frame.get("frame_idx"), f"job frames[{i}].frame_idx"),
                "dataset_index": index,
                "pose_path": str(pose_path),
                "silhouette_mask_iou": float(intersection / union),
                "silhouette_area_px": int(np.count_nonzero(silhouette)),
                "mask_area_px": int(np.count_nonzero(mask)),
                "projectable_mesh_face_count": projectable_face_count,
                "rasterized_mesh_face_count": rasterized_face_count,
                "mesh_face_subsampling_used": bool(rasterized_face_count < projectable_face_count),
                "front_surface_depth_sample_count": int(len(errors)),
                "front_surface_depth_signed_median_m": float(np.median(errors)) if len(errors) else None,
                "front_surface_depth_signed_p05_m": float(np.percentile(errors, 5.0)) if len(errors) else None,
                "front_surface_depth_signed_p95_m": float(np.percentile(errors, 95.0)) if len(errors) else None,
                "front_surface_depth_positive_fraction": float(np.mean(errors > 0.0)) if len(errors) else None,
                "front_surface_depth_abs_median_m": float(np.median(np.abs(errors))) if len(errors) else None,
                "front_surface_depth_abs_p95_m": float(np.percentile(np.abs(errors), 95.0)) if len(errors) else None,
            }
        )
    return rows


def projection_pass(rows: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    if not rows:
        return False
    ious = [finite_float(row.get("silhouette_mask_iou"), "silhouette_mask_iou") for row in rows]
    p95s = [
        finite_float(row.get("front_surface_depth_abs_p95_m"), "front_surface_depth_abs_p95_m")
        for row in rows
        if row.get("front_surface_depth_abs_p95_m") is not None
    ]
    medians = [
        finite_float(row.get("front_surface_depth_abs_median_m"), "front_surface_depth_abs_median_m")
        for row in rows
        if row.get("front_surface_depth_abs_median_m") is not None
    ]
    samples = [require_int(row.get("front_surface_depth_sample_count"), "front_surface_depth_sample_count") for row in rows]
    return bool(
        ious
        and p95s
        and medians
        and min(samples) >= int(args.min_front_depth_samples)
        and float(np.percentile(ious, 5.0)) >= float(args.min_silhouette_iou_p05)
        and float(np.median(ious)) >= float(args.min_silhouette_iou_median)
        and float(np.percentile(p95s, 95.0)) <= float(args.max_front_depth_abs_p95_m)
        and float(np.median(medians)) <= float(args.max_front_depth_abs_median_m)
    )


def evaluate_job(case: str, job_row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    job_id = require_str(job_row.get("job_id"), "job_id")
    job_path = existing_path(Path(require_str(job_row.get("job_path"), "job_path")), f"{case} {job_id} job manifest")
    job = require_dict(load_json(job_path), f"{case} {job_id} job")
    frames = [require_dict(row, f"{job_id} frames[{i}]") for i, row in enumerate(require_list(job.get("frames"), "job frames"))]
    observed_rows = observed_extent_rows(frames, int(args.max_observed_extent_points))
    observed_maxima = [finite_float(row["max_extent_p05_p95_m"], "observed max extent") for row in observed_rows]
    output_dir = args.bundlesdf_output_root / case / job_id
    mesh_path = candidate_mesh_path(output_dir)
    base = {
        "case": case,
        "job_id": job_id,
        "object_id": require_str(job_row.get("object_id"), "object_id"),
        "track_id": require_str(job_row.get("track_id"), "track_id"),
        "window_id": require_str(job_row.get("window_id"), "window_id"),
        "source_job_path": str(job_path),
        "source_dataset_dir": require_str(job_row.get("dataset_dir"), "dataset_dir"),
        "bundlesdf_output_dir": str(output_dir),
        "frame_count": require_int(job.get("frame_count"), "job frame_count"),
        "first_frame": require_int(job.get("first_frame"), "job first_frame"),
        "last_frame": require_int(job.get("last_frame"), "job last_frame"),
        "solver_job_ready": bool(job_row.get("solver_job_ready") is True),
        "observed_rectified_extent_m": {
            "rows": observed_rows,
            "max_extent_summary": summarize(observed_maxima),
        },
    }
    if not output_dir.exists():
        return {
            **base,
            "status": "pending_solver_output",
            "readiness_checks": {
                "solver_backend_output_detected": False,
                "mesh_file_detected": False,
                "pose_sequence_complete": False,
                "mesh_scale_plausible_against_rectified_rgbd": False,
                "mesh_surface_topology_plausible": False,
                "mesh_projection_qc_passed": False,
            },
            "hidden_topology_reconstructed": False,
            "accepted_reconstruction_result": False,
            **FALSE_READY,
        }
    if mesh_path is None:
        return {
            **base,
            "status": "rejected_missing_solver_mesh",
            "readiness_checks": {
                "solver_backend_output_detected": True,
                "mesh_file_detected": False,
                "pose_sequence_complete": False,
                "mesh_scale_plausible_against_rectified_rgbd": False,
                "mesh_surface_topology_plausible": False,
                "mesh_projection_qc_passed": False,
            },
            "hidden_topology_reconstructed": False,
            "accepted_reconstruction_result": False,
            **FALSE_READY,
        }
    raw_vertices, faces = load_obj_mesh(mesh_path)
    metric_mesh_path = args.output_root / case / job_id / "bundlesdf_metric_mesh_for_qc.obj"
    vertices, mesh_contract = metric_mesh_for_qc(output_dir, mesh_path, raw_vertices, faces, metric_mesh_path)
    extent = mesh_extent(vertices)
    max_extent = float(np.max(extent))
    observed_limit = float(np.percentile(observed_maxima, 95.0)) * float(args.max_mesh_extent_ratio)
    observed_floor = float(np.percentile(observed_maxima, 5.0)) * float(args.min_mesh_extent_ratio)
    scale_ok = bool(observed_floor <= max_extent <= observed_limit)
    topology = mesh_topology_stats(faces)
    topology_ok = bool(
        finite_float(topology["boundary_edge_fraction"], "boundary_edge_fraction")
        <= float(args.max_boundary_edge_fraction)
        and finite_float(topology["nonmanifold_edge_fraction"], "nonmanifold_edge_fraction")
        <= float(args.max_nonmanifold_edge_fraction)
    )
    pose_dir = output_dir / "ob_in_cam"
    pose_files = [pose_dir / f"{require_int(frame.get('index'), 'frame.index'):06d}.txt" for frame in frames]
    pose_sequence_complete = all(path.exists() for path in pose_files)
    projection = None
    projection_ok = False
    projection_error = None
    if pose_sequence_complete:
        try:
            projection_rows_payload = projection_rows(frames, output_dir, vertices, faces, args)
            projection = {
                "silhouette_mask_iou": summarize([row["silhouette_mask_iou"] for row in projection_rows_payload]),
                "projectable_mesh_face_count": summarize(
                    [
                        float(require_int(row["projectable_mesh_face_count"], "projectable mesh face count"))
                        for row in projection_rows_payload
                    ]
                ),
                "rasterized_mesh_face_count": summarize(
                    [
                        float(require_int(row["rasterized_mesh_face_count"], "rasterized mesh face count"))
                        for row in projection_rows_payload
                    ]
                ),
                "mesh_face_subsampling_used_frame_count": sum(
                    1 for row in projection_rows_payload if row["mesh_face_subsampling_used"] is True
                ),
                "front_surface_depth_abs_median_m": summarize(
                    [
                        finite_float(row["front_surface_depth_abs_median_m"], "front depth median")
                        for row in projection_rows_payload
                        if row["front_surface_depth_abs_median_m"] is not None
                    ]
                ),
                "front_surface_depth_abs_p95_m": summarize(
                    [
                        finite_float(row["front_surface_depth_abs_p95_m"], "front depth p95")
                        for row in projection_rows_payload
                        if row["front_surface_depth_abs_p95_m"] is not None
                    ]
                ),
                "front_surface_depth_signed_median_m": summarize(
                    [
                        finite_float(row["front_surface_depth_signed_median_m"], "front signed depth median")
                        for row in projection_rows_payload
                        if row["front_surface_depth_signed_median_m"] is not None
                    ]
                ),
                "front_surface_depth_signed_p05_m": summarize(
                    [
                        finite_float(row["front_surface_depth_signed_p05_m"], "front signed depth p05")
                        for row in projection_rows_payload
                        if row["front_surface_depth_signed_p05_m"] is not None
                    ]
                ),
                "front_surface_depth_signed_p95_m": summarize(
                    [
                        finite_float(row["front_surface_depth_signed_p95_m"], "front signed depth p95")
                        for row in projection_rows_payload
                        if row["front_surface_depth_signed_p95_m"] is not None
                    ]
                ),
                "front_surface_depth_positive_fraction": summarize(
                    [
                        finite_float(row["front_surface_depth_positive_fraction"], "front positive depth fraction")
                        for row in projection_rows_payload
                        if row["front_surface_depth_positive_fraction"] is not None
                    ]
                ),
                "front_surface_depth_sample_count": summarize(
                    [
                        float(require_int(row["front_surface_depth_sample_count"], "front depth samples"))
                        for row in projection_rows_payload
                    ]
                ),
                "rows": projection_rows_payload,
            }
            projection_ok = projection_pass(projection_rows_payload, args)
        except RuntimeError as exc:
            projection_error = str(exc)
    checks = {
        "solver_backend_output_detected": True,
        "mesh_file_detected": True,
        "pose_sequence_complete": pose_sequence_complete,
        "mesh_scale_plausible_against_rectified_rgbd": scale_ok,
        "mesh_surface_topology_plausible": topology_ok,
        "mesh_projection_qc_passed": projection_ok,
    }
    accepted = all(checks.values())
    if accepted:
        status = "accepted_hidden_topology_segment_qc"
    elif not pose_sequence_complete:
        status = "rejected_incomplete_pose_sequence"
    elif not scale_ok:
        status = "rejected_mesh_scale_inconsistent_with_rectified_rgbd"
    elif not topology_ok:
        status = "rejected_open_or_nonmanifold_mesh_topology"
    elif not projection_ok:
        status = "rejected_mesh_projection_qc"
    else:
        status = "rejected_reconstruction_result"
    return {
        **base,
        "status": status,
        "mesh_path": str(mesh_contract["metric_mesh_path"]),
        "source_solver_mesh_path": str(mesh_path),
        "mesh_coordinate_contract": mesh_contract,
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "mesh_extent_m": [float(v) for v in extent.tolist()],
        "mesh_max_extent_m": max_extent,
        "mesh_scale_acceptance_range_m": [observed_floor, observed_limit],
        "mesh_topology": topology,
        "pose_files_expected": len(pose_files),
        "pose_files_found": int(sum(1 for path in pose_files if path.exists())),
        "projection_qc": projection,
        "projection_qc_error": projection_error,
        "readiness_checks": checks,
        "hidden_topology_reconstructed": accepted,
        "accepted_reconstruction_result": accepted,
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    jobs_report_path = existing_path(
        args.geometry_reconstruction_jobs_root / case / "v17_geometry_reconstruction_jobs_report.json",
        f"{case} geometry reconstruction jobs report",
    )
    jobs_report = require_dict(load_json(jobs_report_path), f"{case} geometry reconstruction jobs report")
    rows = [evaluate_job(case, require_dict(row, f"{case} jobs[{i}]"), args) for i, row in enumerate(require_list(jobs_report.get("jobs"), "jobs"))]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = require_str(row.get("status"), "result status")
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "method": "build_v17_geometry_reconstruction_results",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_geometry_reconstruction_jobs_report": str(jobs_report_path),
        "bundlesdf_output_root": str(args.bundlesdf_output_root),
        "job_count": len(rows),
        "solver_job_ready_count": sum(1 for row in rows if row.get("solver_job_ready") is True),
        "pending_solver_output_count": sum(1 for row in rows if row.get("status") == "pending_solver_output"),
        "solver_output_detected_count": sum(
            1 for row in rows if require_dict(row.get("readiness_checks"), "readiness_checks")["solver_backend_output_detected"] is True
        ),
        "mesh_file_detected_count": sum(
            1 for row in rows if require_dict(row.get("readiness_checks"), "readiness_checks")["mesh_file_detected"] is True
        ),
        "pose_sequence_complete_count": sum(
            1 for row in rows if require_dict(row.get("readiness_checks"), "readiness_checks")["pose_sequence_complete"] is True
        ),
        "mesh_scale_plausible_count": sum(
            1
            for row in rows
            if require_dict(row.get("readiness_checks"), "readiness_checks")[
                "mesh_scale_plausible_against_rectified_rgbd"
            ]
            is True
        ),
        "mesh_projection_qc_passed_count": sum(
            1 for row in rows if require_dict(row.get("readiness_checks"), "readiness_checks")["mesh_projection_qc_passed"] is True
        ),
        "hidden_topology_reconstructed_job_count": sum(1 for row in rows if row.get("hidden_topology_reconstructed") is True),
        "accepted_reconstruction_result_count": sum(1 for row in rows if row.get("accepted_reconstruction_result") is True),
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        "status_counts": dict(sorted(status_counts.items())),
        "jobs": rows,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_geometry_reconstruction_results_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.geometry_reconstruction_jobs_root / "v17_geometry_reconstruction_jobs_summary.json",
        "geometry reconstruction jobs summary",
    )
    jobs_summary = require_dict(load_json(summary_path), "geometry reconstruction jobs summary")
    reports = [
        case_report(
            require_str(require_dict(row, f"summary cases[{i}]").get("case"), f"summary cases[{i}].case"),
            args,
        )
        for i, row in enumerate(require_list(jobs_summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_geometry_reconstruction_results",
        "status": STATUS,
        "claim": CLAIM,
        "source_geometry_reconstruction_jobs_summary": str(summary_path),
        "bundlesdf_output_root": str(args.bundlesdf_output_root),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(args.output_root / require_str(report.get("case"), "case") / "v17_geometry_reconstruction_results_report.json"),
                "job_count": require_int(report.get("job_count"), "job_count"),
                "solver_job_ready_count": require_int(report.get("solver_job_ready_count"), "solver_job_ready_count"),
                "pending_solver_output_count": require_int(report.get("pending_solver_output_count"), "pending_solver_output_count"),
                "solver_output_detected_count": require_int(report.get("solver_output_detected_count"), "solver_output_detected_count"),
                "mesh_file_detected_count": require_int(report.get("mesh_file_detected_count"), "mesh_file_detected_count"),
                "pose_sequence_complete_count": require_int(report.get("pose_sequence_complete_count"), "pose_sequence_complete_count"),
                "mesh_scale_plausible_count": require_int(report.get("mesh_scale_plausible_count"), "mesh_scale_plausible_count"),
                "mesh_projection_qc_passed_count": require_int(report.get("mesh_projection_qc_passed_count"), "mesh_projection_qc_passed_count"),
                "hidden_topology_reconstructed_job_count": require_int(
                    report.get("hidden_topology_reconstructed_job_count"),
                    "hidden_topology_reconstructed_job_count",
                ),
                "accepted_reconstruction_result_count": require_int(
                    report.get("accepted_reconstruction_result_count"),
                    "accepted_reconstruction_result_count",
                ),
                "complete_geometry_seed_count": 0,
                "contact_compatible_geometry_seed_count": 0,
                "full_active_interval_geometry_seed_count": 0,
                **FALSE_READY,
            }
            for report in reports
        ],
        "job_count": sum(require_int(report.get("job_count"), "job_count") for report in reports),
        "solver_job_ready_count": sum(require_int(report.get("solver_job_ready_count"), "solver_job_ready_count") for report in reports),
        "pending_solver_output_count": sum(require_int(report.get("pending_solver_output_count"), "pending_solver_output_count") for report in reports),
        "solver_output_detected_count": sum(require_int(report.get("solver_output_detected_count"), "solver_output_detected_count") for report in reports),
        "mesh_file_detected_count": sum(require_int(report.get("mesh_file_detected_count"), "mesh_file_detected_count") for report in reports),
        "pose_sequence_complete_count": sum(require_int(report.get("pose_sequence_complete_count"), "pose_sequence_complete_count") for report in reports),
        "mesh_scale_plausible_count": sum(require_int(report.get("mesh_scale_plausible_count"), "mesh_scale_plausible_count") for report in reports),
        "mesh_projection_qc_passed_count": sum(require_int(report.get("mesh_projection_qc_passed_count"), "mesh_projection_qc_passed_count") for report in reports),
        "hidden_topology_reconstructed_job_count": sum(
            require_int(report.get("hidden_topology_reconstructed_job_count"), "hidden_topology_reconstructed_job_count")
            for report in reports
        ),
        "accepted_reconstruction_result_count": sum(
            require_int(report.get("accepted_reconstruction_result_count"), "accepted_reconstruction_result_count")
            for report in reports
        ),
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_geometry_reconstruction_results_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-reconstruction-jobs-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs"),
    )
    parser.add_argument(
        "--bundlesdf-output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_bundlesdf_outputs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results"),
    )
    parser.add_argument("--max-observed-extent-points", type=int, default=5000)
    parser.add_argument("--min-mesh-extent-ratio", type=float, default=0.20)
    parser.add_argument("--max-mesh-extent-ratio", type=float, default=3.0)
    parser.add_argument("--max-boundary-edge-fraction", type=float, default=0.25)
    parser.add_argument("--max-nonmanifold-edge-fraction", type=float, default=0.02)
    parser.add_argument("--max-projection-faces", type=int, default=0)
    parser.add_argument("--min-silhouette-iou-p05", type=float, default=0.35)
    parser.add_argument("--min-silhouette-iou-median", type=float, default=0.45)
    parser.add_argument("--min-front-depth-samples", type=int, default=200)
    parser.add_argument("--max-front-depth-abs-median-m", type=float, default=0.02)
    parser.add_argument("--max-front-depth-abs-p95-m", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
