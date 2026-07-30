#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingImports]


STATUS = "v17_geometry_reconstruction_jobs_qc"
CLAIM = (
    "This artifact prepares auditable RGBD reconstruction jobs for hidden-topology object solvers. "
    "It rectifies per-frame metric RGBD observations into one constant-intrinsics camera contract and "
    "measures whether the rectified depth preserves the original object rays. It is a solver input layer, "
    "not object geometry reconstruction."
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


def stable_seed(*parts: Any) -> int:
    total = 0
    for ch in "|".join(str(part) for part in parts).encode("utf-8"):
        total = (total * 131 + int(ch)) % (2**32 - 1)
    return total


def sample_rows(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to read grayscale image: {path}")
    return image


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read color image: {path}")
    return image


def read_depth_m(path: Path) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise RuntimeError(f"failed to read depth image: {path}")
    if depth_mm.ndim != 2:
        raise RuntimeError(f"depth image must be single-channel: {path}")
    return depth_mm.astype(np.float32) / 1000.0


def write_cam_k(path: Path, intrinsics: np.ndarray) -> None:
    fx, fy, cx, cy = [float(v) for v in intrinsics.tolist()]
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savetxt(path, K, fmt="%.10f")


def image_points_to_xyz(u: np.ndarray, v: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intrinsics.tolist()]
    z = depth.astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def project_xyz(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = [float(x) for x in intrinsics.tolist()]
    z = points[:, 2]
    if np.any(z <= 0.0) or not np.isfinite(points).all():
        raise RuntimeError("cannot project invalid or non-positive-depth points")
    u = fx * points[:, 0] / z + cx
    v = fy * points[:, 1] / z + cy
    return u, v


def object_points_from_frame(frame: dict[str, Any], *, max_points: int, seed: int) -> np.ndarray:
    mask = read_gray(Path(require_str(frame.get("mask"), "frame.mask"))) > 0
    depth = read_depth_m(Path(require_str(frame.get("depth"), "frame.depth")))
    if depth.shape != mask.shape:
        raise RuntimeError(f"mask/depth shape mismatch at frame {frame.get('frame_idx')}: {mask.shape} vs {depth.shape}")
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no valid object-mask depth")
    selected = np.column_stack([xs, ys])
    selected = sample_rows(selected, max_points, seed).astype(np.int64)
    intrinsics = np.asarray(frame.get("intrinsics_fx_fy_cx_cy"), dtype=np.float64)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise RuntimeError(f"frame {frame.get('frame_idx')} has invalid intrinsics")
    return image_points_to_xyz(selected[:, 0], selected[:, 1], depth[selected[:, 1], selected[:, 0]], intrinsics)


def target_intrinsics(frames: list[dict[str, Any]]) -> np.ndarray:
    intrinsics = np.asarray([frame["intrinsics_fx_fy_cx_cy"] for frame in frames], dtype=np.float64)
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4 or not np.isfinite(intrinsics).all():
        raise RuntimeError("frame intrinsics must be finite Nx4")
    return np.median(intrinsics, axis=0)


def rectify_frame(
    frame: dict[str, Any],
    *,
    output_index: int,
    target_k: np.ndarray,
    output_rgb: Path,
    output_mask: Path,
    output_depth: Path,
    max_eval_points: int,
    seed: int,
    raster_scale: int,
) -> dict[str, Any]:
    rgb = read_color(Path(require_str(frame.get("rgb"), "frame.rgb")))
    mask = read_gray(Path(require_str(frame.get("mask"), "frame.mask"))) > 0
    depth = read_depth_m(Path(require_str(frame.get("depth"), "frame.depth")))
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise RuntimeError(f"input RGB/mask/depth shape mismatch at frame {frame.get('frame_idx')}")
    source_k = np.asarray(frame.get("intrinsics_fx_fy_cx_cy"), dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no valid object-mask depth")
    points = image_points_to_xyz(xs.astype(np.float64), ys.astype(np.float64), depth[ys, xs], source_k)
    scaled_k = target_k.astype(np.float64).copy()
    scaled_k *= float(raster_scale)
    h, w = depth.shape
    out_h = h * int(raster_scale)
    out_w = w * int(raster_scale)
    u_float, v_float = project_xyz(points, scaled_k)
    u = np.rint(u_float).astype(np.int64)
    v = np.rint(v_float).astype(np.int64)
    inside = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
    inside_count = int(np.count_nonzero(inside))
    if inside_count == 0:
        raise RuntimeError(f"frame {frame.get('frame_idx')} has no projected object points inside target camera")

    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    map_x = ((xx - float(scaled_k[2])) / float(scaled_k[0])) * float(source_k[0]) + float(source_k[2])
    map_y = ((yy - float(scaled_k[3])) / float(scaled_k[1])) * float(source_k[1]) + float(source_k[3])
    mask_u8 = mask.astype(np.uint8) * 255
    rect_rgb = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_depth = cv2.remap(
        depth.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    rect_mask = cv2.remap(
        mask_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rect_mask = ((rect_mask > 0) & np.isfinite(rect_depth) & (rect_depth > 0.05)).astype(np.uint8) * 255
    rect_depth = np.where(rect_mask > 0, rect_depth, 0.0).astype(np.float32)
    output_rgb.parent.mkdir(parents=True, exist_ok=True)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_rgb), rect_rgb):
        raise RuntimeError(f"failed to write rectified RGB: {output_rgb}")
    if not cv2.imwrite(str(output_mask), rect_mask):
        raise RuntimeError(f"failed to write rectified mask: {output_mask}")
    depth_mm = np.clip(np.rint(rect_depth.astype(np.float64) * 1000.0), 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(output_depth), depth_mm):
        raise RuntimeError(f"failed to write rectified depth: {output_depth}")

    rect_valid = rect_depth > 0.05
    yy, xx = np.nonzero(rect_valid)
    rect_points = image_points_to_xyz(xx.astype(np.float64), yy.astype(np.float64), rect_depth[yy, xx], scaled_k)
    eval_source = sample_rows(
        points[inside],
        max_eval_points,
        seed,
    )
    distances = cKDTree(rect_points).query(eval_source, k=1, workers=-1)[0]
    source_index = require_int(frame.get("index"), "frame.index")
    return {
        "index": int(output_index),
        "source_object_track_index": source_index,
        "frame_idx": require_int(frame.get("frame_idx"), "frame.frame_idx"),
        "rgb": str(output_rgb),
        "mask": str(output_mask),
        "depth": str(output_depth),
        "source_rgb": require_str(frame.get("rgb"), "frame.rgb"),
        "source_mask": require_str(frame.get("mask"), "frame.mask"),
        "source_depth": require_str(frame.get("depth"), "frame.depth"),
        "source_intrinsics_fx_fy_cx_cy": [float(v) for v in source_k.tolist()],
        "rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in scaled_k.tolist()],
        "base_rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in target_k.tolist()],
        "raster_scale": int(raster_scale),
        "source_mask_depth_points": int(len(points)),
        "projected_inside_points": inside_count,
        "rectified_depth_points": int(len(rect_points)),
        "projected_inside_fraction": float(np.count_nonzero(inside) / len(points)),
        "rectification_nearest_3d_residual_m": summarize(distances),
        **FALSE_READY,
    }


def seed_frame_ranges(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, raw in enumerate(require_list(report.get("candidate_rows"), "observed-surface seed candidates")):
        row = require_dict(raw, f"observed-surface seed candidates[{i}]")
        if row.get("observed_surface_only") is not True:
            raise RuntimeError(f"seed row {i} must be observed-surface-only under the current contract")
        rows.append(
            {
                "candidate_id": require_str(row.get("candidate_id"), f"seed row {i}.candidate_id"),
                "object_id": require_str(row.get("object_id"), f"seed row {i}.object_id"),
                "track_id": require_str(row.get("track_id"), f"seed row {i}.track_id"),
                "window_id": require_str(row.get("window_id"), f"seed row {i}.window_id"),
                "archive_path": require_str(row.get("archive_path"), f"seed row {i}.archive_path"),
                "start_frame": require_int(row.get("start_frame"), f"seed row {i}.start_frame"),
                "end_frame": require_int(row.get("end_frame"), f"seed row {i}.end_frame"),
                "seed_frame_count": require_int(row.get("seed_frame_count"), f"seed row {i}.seed_frame_count"),
                "seed_vertices": require_int(row.get("seed_vertices"), f"seed row {i}.seed_vertices"),
                "seed_faces": require_int(row.get("seed_faces"), f"seed row {i}.seed_faces"),
            }
        )
    return rows


def frame_subset(manifest: dict[str, Any], start_frame: int, end_frame: int) -> list[dict[str, Any]]:
    frames = []
    for i, raw in enumerate(require_list(manifest.get("frames"), "object-track frames")):
        frame = require_dict(raw, f"object-track frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"object-track frames[{i}].frame_idx")
        if start_frame <= frame_idx <= end_frame:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"no object-track frames in {start_frame}-{end_frame}")
    return frames


def intrinsics_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    intr = np.asarray([frame["intrinsics_fx_fy_cx_cy"] for frame in frames], dtype=np.float64)
    return {
        "count": int(len(intr)),
        "min": intr.min(axis=0).astype(float).tolist(),
        "max": intr.max(axis=0).astype(float).tolist(),
        "spread": np.ptp(intr, axis=0).astype(float).tolist(),
        "median": np.median(intr, axis=0).astype(float).tolist(),
    }


def build_job(
    *,
    case: str,
    seed: dict[str, Any],
    object_manifest: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = require_int(seed.get("start_frame"), "seed start_frame")
    end = require_int(seed.get("end_frame"), "seed end_frame")
    frames = frame_subset(object_manifest, start, end)
    target_k = target_intrinsics(frames)
    job_id = require_str(seed.get("candidate_id"), "seed candidate_id")
    job_dir = output_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    rgb_dir = job_dir / "rgb"
    mask_dir = job_dir / "masks"
    depth_dir = job_dir / "depth"
    rectified_rows = [
        rectify_frame(
            frame,
            output_index=out_i,
            target_k=target_k,
            output_rgb=rgb_dir / f"{out_i:06d}.png",
            output_mask=mask_dir / f"{out_i:06d}.png",
            output_depth=depth_dir / f"{out_i:06d}.png",
            max_eval_points=int(args.max_eval_points),
            seed=stable_seed(case, job_id, frame["frame_idx"]),
            raster_scale=int(args.raster_scale),
        )
        for out_i, frame in enumerate(frames)
    ]
    scaled_k = target_k.astype(np.float64).copy()
    scaled_k *= float(args.raster_scale)
    write_cam_k(job_dir / "cam_K.txt", scaled_k)
    residual_p95 = [
        finite_float(row["rectification_nearest_3d_residual_m"].get("p95"), "rectification p95")
        for row in rectified_rows
    ]
    inside_fraction = [finite_float(row.get("projected_inside_fraction"), "projected_inside_fraction") for row in rectified_rows]
    ray_preserving = bool(
        residual_p95
        and max(residual_p95) <= float(args.max_rectification_residual_p95_m)
        and min(inside_fraction) >= float(args.min_projected_inside_fraction)
    )
    job_status = "ready_for_unknown_object_rgbd_solver" if ray_preserving else "rejected_rectification_residual"
    manifest = {
        "method": "build_v17_geometry_reconstruction_jobs",
        "status": job_status,
        "claim": CLAIM,
        "case": case,
        "job_id": job_id,
        "object_id": require_str(seed.get("object_id"), "seed object_id"),
        "track_id": require_str(seed.get("track_id"), "seed track_id"),
        "window_id": require_str(seed.get("window_id"), "seed window_id"),
        "solver_backend_contract": "BundleSDF-compatible RGBD folder with constant cam_K.txt; no solver output included",
        "dataset_dir": str(job_dir),
        "cam_k": str(job_dir / "cam_K.txt"),
        "source_object_track_manifest": require_str(object_manifest.get("dataset_dir"), "object manifest dataset_dir") + "/manifest.json",
        "source_observed_surface_seed_archive": require_str(seed.get("archive_path"), "seed archive_path"),
        "frame_count": len(rectified_rows),
        "first_frame": start,
        "last_frame": end,
        "source_seed_frame_count": require_int(seed.get("seed_frame_count"), "seed frame count"),
        "source_seed_vertices": require_int(seed.get("seed_vertices"), "seed vertices"),
        "source_seed_faces": require_int(seed.get("seed_faces"), "seed faces"),
        "source_intrinsics": intrinsics_summary(frames),
        "rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in scaled_k.tolist()],
        "base_rectified_intrinsics_fx_fy_cx_cy": [float(v) for v in target_k.tolist()],
        "raster_scale": int(args.raster_scale),
        "frames": rectified_rows,
        "rectification_nearest_3d_residual_p95_m": summarize(residual_p95),
        "projected_inside_fraction": summarize(inside_fraction),
        "readiness_checks": {
            "constant_intrinsics_written": True,
            "source_rays_preserved_by_rectified_depth": ray_preserving,
            "hidden_topology_solver_has_run": False,
            "hidden_topology_reconstructed": False,
            "mesh_projection_qc_passed": False,
        },
        "solver_job_ready": ray_preserving,
        "hidden_topology_reconstructed": False,
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(job_dir / "manifest.json", {"frames": rectified_rows})
    write_json(job_dir / "v17_geometry_reconstruction_job.json", manifest)
    return manifest


def object_manifest_path(args: argparse.Namespace, case: str, track_id: str) -> Path:
    return existing_path(
        args.object_track_dataset_root / case / track_id / "manifest.json",
        f"{case} {track_id} object-track manifest",
    )


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    seed_report_path = existing_path(
        args.observed_surface_geometry_seed_root / case / "v17_observed_surface_geometry_seed_report.json",
        f"{case} observed-surface geometry seed report",
    )
    seed_report = require_dict(load_json(seed_report_path), f"{case} observed-surface geometry seed report")
    jobs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for seed in seed_frame_ranges(seed_report):
        track_id = require_str(seed.get("track_id"), "seed track_id")
        manifest_path = object_manifest_path(args, case, track_id)
        object_manifest = require_dict(load_json(manifest_path), f"{case} {track_id} object manifest")
        try:
            jobs.append(
                build_job(
                    case=case,
                    seed=seed,
                    object_manifest=object_manifest,
                    output_dir=args.output_root / case,
                    args=args,
                )
            )
        except RuntimeError as exc:
            skipped.append(
                {
                    "candidate_id": require_str(seed.get("candidate_id"), "seed candidate_id"),
                    "object_id": require_str(seed.get("object_id"), "seed object_id"),
                    "track_id": track_id,
                    "reason": str(exc),
                    **FALSE_READY,
                }
            )
    ready_jobs = [job for job in jobs if job.get("solver_job_ready") is True]
    residuals = [
        finite_float(frame["rectification_nearest_3d_residual_m"].get("p95"), "rectification frame p95")
        for job in jobs
        for frame in require_list(job.get("frames"), "job frames")
    ]
    report = {
        "method": "build_v17_geometry_reconstruction_jobs",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_observed_surface_geometry_seed_report": str(seed_report_path),
        "job_count": len(jobs),
        "solver_job_ready_count": len(ready_jobs),
        "skipped_job_count": len(skipped),
        "jobs": [
            {
                "job_id": require_str(job.get("job_id"), "job_id"),
                "object_id": require_str(job.get("object_id"), "object_id"),
                "track_id": require_str(job.get("track_id"), "track_id"),
                "window_id": require_str(job.get("window_id"), "window_id"),
                "job_path": str(Path(require_str(job.get("dataset_dir"), "dataset_dir")) / "v17_geometry_reconstruction_job.json"),
                "dataset_dir": require_str(job.get("dataset_dir"), "dataset_dir"),
                "frame_count": require_int(job.get("frame_count"), "frame_count"),
                "first_frame": require_int(job.get("first_frame"), "first_frame"),
                "last_frame": require_int(job.get("last_frame"), "last_frame"),
                "solver_job_ready": bool(job.get("solver_job_ready") is True),
                "rectification_nearest_3d_residual_p95_m": require_dict(
                    job.get("rectification_nearest_3d_residual_p95_m"),
                    "job rectification residual",
                ),
                "projected_inside_fraction": require_dict(job.get("projected_inside_fraction"), "job projected fraction"),
                "source_intrinsics": require_dict(job.get("source_intrinsics"), "job source intrinsics"),
                "rectified_intrinsics_fx_fy_cx_cy": require_list(
                    job.get("rectified_intrinsics_fx_fy_cx_cy"),
                    "job rectified intrinsics",
                ),
                "hidden_topology_reconstructed": False,
                **FALSE_READY,
            }
            for job in jobs
        ],
        "skipped_jobs": skipped,
        "rectification_nearest_3d_residual_p95_m": summarize(residuals),
        "hidden_topology_reconstructed_job_count": 0,
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_geometry_reconstruction_jobs_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.observed_surface_geometry_seed_root / "v17_observed_surface_geometry_seed_summary.json",
        "observed-surface geometry seed summary",
    )
    summary = require_dict(load_json(summary_path), "observed-surface geometry seed summary")
    reports = [
        build_case(require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_geometry_reconstruction_jobs",
        "status": STATUS,
        "claim": CLAIM,
        "source_observed_surface_geometry_seed_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_geometry_reconstruction_jobs_report.json"
                ),
                "job_count": require_int(report.get("job_count"), "job_count"),
                "solver_job_ready_count": require_int(report.get("solver_job_ready_count"), "solver_job_ready_count"),
                "skipped_job_count": require_int(report.get("skipped_job_count"), "skipped_job_count"),
                "hidden_topology_reconstructed_job_count": 0,
                "complete_geometry_seed_count": 0,
                "contact_compatible_geometry_seed_count": 0,
                "full_active_interval_geometry_seed_count": 0,
                **FALSE_READY,
            }
            for report in reports
        ],
        "job_count": sum(require_int(report.get("job_count"), "job_count") for report in reports),
        "solver_job_ready_count": sum(
            require_int(report.get("solver_job_ready_count"), "solver_job_ready_count") for report in reports
        ),
        "skipped_job_count": sum(require_int(report.get("skipped_job_count"), "skipped_job_count") for report in reports),
        "hidden_topology_reconstructed_job_count": 0,
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_geometry_reconstruction_jobs_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observed-surface-geometry-seed-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed"),
    )
    parser.add_argument(
        "--object-track-dataset-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs"),
    )
    parser.add_argument("--max-eval-points", type=int, default=5000)
    parser.add_argument("--max-rectification-residual-p95-m", type=float, default=0.003)
    parser.add_argument("--min-projected-inside-fraction", type=float, default=0.995)
    parser.add_argument("--raster-scale", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    payload = build(parse_args())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
