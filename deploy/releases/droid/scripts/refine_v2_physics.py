#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from fuse_v1_full_fidelity import (
    DEFAULT_CLIP,
    DEFAULT_MANO_RIGHT,
    RenderSpec,
    hand_vertices,
    load_json,
    open_video,
    project_points,
    render_outputs,
    source_camera_ray,
)

TIP_IDS = [4, 8, 12, 16, 20]


@dataclass(frozen=True)
class FrameFactors:
    source_idx: int
    T_world_camera: np.ndarray
    ray_source: np.ndarray
    depth0_m: float
    radius0_m: float
    radius_per_depth: float
    depth_sigma_m: float
    radius_sigma_m: float
    contact_points_world: np.ndarray
    penetration_points_world: np.ndarray
    measured: bool


def finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{name} has invalid shape or non-finite values: got {arr.shape}, expected {shape}")
    return arr


def first_intrinsics(frames: list[dict]) -> np.ndarray:
    for frame in frames:
        for hand in frame.get("hands", []):
            if "source_intrinsics" in hand:
                return finite_array(hand["source_intrinsics"], (4,), "source_intrinsics")
    raise RuntimeError("annotations contain no hand source_intrinsics")


def active_object_indices(frames: list[dict]) -> list[int]:
    active = []
    for i, frame in enumerate(frames):
        obj = frame.get("object", {})
        if obj.get("center_xy") is None:
            continue
        if obj.get("depth_m") is None or obj.get("radius_m") is None or obj.get("center_world_m") is None:
            raise RuntimeError(f"frame {i} has object image state but missing world fields")
        active.append(i)
    if not active:
        raise RuntimeError("annotations contain no active object world states")
    return active


def object_radius_px(obj: dict) -> float:
    return math.sqrt(max(1.0, float(obj["area_px"])) / math.pi)


def contact_points(frame: dict, center_xy: np.ndarray, radius_px: float, contact_px: float) -> np.ndarray:
    points = []
    for hand in frame.get("hands", []):
        joints2d = finite_array(hand["joints2d"], (21, 2), "joints2d")
        joints3d = finite_array(hand["joints3d_world_m"], (21, 3), "joints3d_world_m")
        dists = np.linalg.norm(joints2d[TIP_IDS] - center_xy[None, :], axis=1)
        limit = max(contact_px, radius_px + 35.0)
        for dist, point in zip(dists, joints3d[TIP_IDS]):
            if float(dist) <= limit:
                points.append((float(dist), point))
    if not points:
        return np.zeros((0, 3), dtype=float)
    points.sort(key=lambda item: item[0])
    return np.asarray([point for _, point in points[:4]], dtype=float)


def penetration_points(frame: dict, center_xy: np.ndarray, radius_px: float, intrinsics: np.ndarray, max_points: int) -> np.ndarray:
    candidates = []
    limit = radius_px + 70.0
    for hand in frame.get("hands", []):
        verts_source = hand_vertices(hand, "_source_camera_m")
        verts_world = hand_vertices(hand, "_world_m")
        if len(verts_source) != len(verts_world):
            raise RuntimeError("source/world MANO vertex counts differ")
        projected = project_points(np.asarray(verts_source, dtype=float), intrinsics)
        dists = np.linalg.norm(projected - center_xy[None, :], axis=1)
        valid = np.isfinite(dists) & (dists <= limit)
        for dist, point in zip(dists[valid], np.asarray(verts_world, dtype=float)[valid]):
            candidates.append((float(dist), point))
    if not candidates:
        return np.zeros((0, 3), dtype=float)
    candidates.sort(key=lambda item: item[0])
    return np.asarray([point for _, point in candidates[:max_points]], dtype=float)


def depth_sigma(obj: dict) -> float:
    evidence = obj.get("depth_evidence", {})
    has_depth = bool(evidence.get("droid_depth", False))
    has_contact = bool(evidence.get("contact_anchor", False))
    measured = bool(obj.get("measurement_available", False))
    if has_depth and has_contact:
        return 0.055
    if has_depth or has_contact:
        return 0.085 if measured else 0.14
    return 0.18 if measured else 0.25


def build_factors(
    frames: list[dict],
    active: list[int],
    intrinsics: np.ndarray,
    contact_px: float,
    max_penetration_points: int,
) -> list[FrameFactors]:
    fx, fy, _, _ = intrinsics
    focal = 0.5 * (float(fx) + float(fy))
    factors = []
    for i in active:
        frame = frames[i]
        obj = frame["object"]
        center_xy = finite_array(obj["center_xy"], (2,), "object center_xy")
        T = finite_array(frame["camera"]["T_world_camera_metric"], (4, 4), "T_world_camera_metric")
        radius_px = object_radius_px(obj)
        contacts = contact_points(frame, center_xy, radius_px, contact_px)
        penetrations = penetration_points(frame, center_xy, radius_px, intrinsics, max_penetration_points)
        measured = bool(obj.get("measurement_available", False))
        factors.append(
            FrameFactors(
                source_idx=int(frame["frame_idx"]),
                T_world_camera=T,
                ray_source=source_camera_ray(center_xy, intrinsics),
                depth0_m=float(obj["depth_m"]),
                radius0_m=float(obj["radius_m"]),
                radius_per_depth=float(radius_px / focal),
                depth_sigma_m=depth_sigma(obj),
                radius_sigma_m=0.010 if measured else 0.024,
                contact_points_world=contacts,
                penetration_points_world=penetrations,
                measured=measured,
            )
        )
    return factors


def centers_world(factors: list[FrameFactors], depths: np.ndarray) -> np.ndarray:
    centers = []
    for factor, depth in zip(factors, depths):
        point = factor.ray_source * float(depth)
        centers.append((factor.T_world_camera @ np.r_[point, 1.0])[:3])
    return np.asarray(centers, dtype=float)


def residual_groups(
    factors: list[FrameFactors],
    depths: np.ndarray,
    radii: np.ndarray,
    accel_sigma_m: float,
    radius_accel_sigma_m: float,
    contact_sigma_m: float,
    penetration_sigma_m: float,
) -> dict[str, np.ndarray]:
    centers = centers_world(factors, depths)
    groups: dict[str, list[float]] = {
        "depth_prior": [],
        "radius_mask": [],
        "center_acceleration": [],
        "radius_acceleration": [],
        "contact_surface": [],
        "nonpenetration": [],
    }
    for i, factor in enumerate(factors):
        groups["depth_prior"].append((depths[i] - factor.depth0_m) / factor.depth_sigma_m)
        groups["radius_mask"].append((radii[i] - factor.radius_per_depth * depths[i]) / factor.radius_sigma_m)
        for point in factor.contact_points_world:
            dist = float(np.linalg.norm(centers[i] - point))
            groups["contact_surface"].append((dist - radii[i]) / contact_sigma_m)
        for point in factor.penetration_points_world:
            dist = float(np.linalg.norm(centers[i] - point))
            groups["nonpenetration"].append(max(0.0, radii[i] - dist) / penetration_sigma_m)
    for i in range(1, len(factors) - 1):
        if factors[i].source_idx - factors[i - 1].source_idx != 1 or factors[i + 1].source_idx - factors[i].source_idx != 1:
            continue
        groups["center_acceleration"].extend(((centers[i - 1] - 2.0 * centers[i] + centers[i + 1]) / accel_sigma_m).tolist())
        groups["radius_acceleration"].append((radii[i - 1] - 2.0 * radii[i] + radii[i + 1]) / radius_accel_sigma_m)
    return {name: np.asarray(values, dtype=float) for name, values in groups.items()}


def flatten_groups(groups: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [values for values in groups.values() if values.size]
    if not arrays:
        raise RuntimeError("factor graph produced no residuals")
    return np.concatenate(arrays)


def jacobian_sparsity(factors: list[FrameFactors]) -> sparse.csr_matrix:
    n = len(factors)
    rows: list[int] = []
    cols: list[int] = []
    row = 0

    def mark(columns: list[int]) -> None:
        nonlocal row
        rows.extend([row] * len(columns))
        cols.extend(columns)
        row += 1

    for i in range(n):
        mark([i])
    for i in range(n):
        mark([i, n + i])
    for i in range(1, n - 1):
        if factors[i].source_idx - factors[i - 1].source_idx != 1 or factors[i + 1].source_idx - factors[i].source_idx != 1:
            continue
        for _ in range(3):
            mark([i - 1, i, i + 1])
    for i in range(1, n - 1):
        if factors[i].source_idx - factors[i - 1].source_idx != 1 or factors[i + 1].source_idx - factors[i].source_idx != 1:
            continue
        mark([n + i - 1, n + i, n + i + 1])
    for i, factor in enumerate(factors):
        for _ in factor.contact_points_world:
            mark([i, n + i])
    for i, factor in enumerate(factors):
        for _ in factor.penetration_points_world:
            mark([i, n + i])
    if row == 0:
        raise RuntimeError("factor graph produced no Jacobian rows")
    return sparse.coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(row, 2 * n)).tocsr()


def rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values * values)))


def raw_physics_metrics(factors: list[FrameFactors], depths: np.ndarray, radii: np.ndarray) -> dict:
    centers = centers_world(factors, depths)
    contact_abs = []
    penetration = []
    contact_frames = 0
    penetration_frames = 0
    for i, factor in enumerate(factors):
        if len(factor.contact_points_world):
            contact_frames += 1
        if len(factor.penetration_points_world):
            penetration_frames += 1
        for point in factor.contact_points_world:
            contact_abs.append(abs(float(np.linalg.norm(centers[i] - point)) - radii[i]))
        for point in factor.penetration_points_world:
            penetration.append(max(0.0, radii[i] - float(np.linalg.norm(centers[i] - point))))
    contact_arr = np.asarray(contact_abs, dtype=float)
    penetration_arr = np.asarray(penetration, dtype=float)
    return {
        "contact_frames": contact_frames,
        "penetration_candidate_frames": penetration_frames,
        "contact_pairs": int(contact_arr.size),
        "penetration_candidate_points": int(penetration_arr.size),
        "median_abs_contact_surface_error_m": float(np.median(contact_arr)) if contact_arr.size else None,
        "p95_abs_contact_surface_error_m": float(np.percentile(contact_arr, 95)) if contact_arr.size else None,
        "max_hand_object_penetration_m": float(np.max(penetration_arr)) if penetration_arr.size else 0.0,
        "p95_hand_object_penetration_m": float(np.percentile(penetration_arr, 95)) if penetration_arr.size else 0.0,
    }


def optimize(args: argparse.Namespace, frames: list[dict], factors: list[FrameFactors]) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(factors)
    depth0 = np.asarray([factor.depth0_m for factor in factors], dtype=float)
    radius0 = np.asarray([factor.radius0_m for factor in factors], dtype=float)
    x0 = np.r_[depth0, radius0]
    lower = np.r_[np.maximum(0.20, depth0 * 0.55), np.maximum(0.006, radius0 * 0.45)]
    upper = np.r_[np.minimum(3.20, depth0 * 1.45), np.minimum(0.22, np.maximum(radius0 * 1.85, radius0 + 0.030))]

    def residual(x: np.ndarray) -> np.ndarray:
        depths = x[:n]
        radii = x[n:]
        groups = residual_groups(
            factors,
            depths,
            radii,
            args.accel_sigma_m,
            args.radius_accel_sigma_m,
            args.contact_sigma_m,
            args.penetration_sigma_m,
        )
        return flatten_groups(groups)

    initial_residual = residual(x0)
    sparsity = jacobian_sparsity(factors)
    if sparsity.shape[0] != initial_residual.size:
        raise RuntimeError(f"Jacobian rows {sparsity.shape[0]} do not match residual length {initial_residual.size}")
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        jac_sparsity=sparsity,
        tr_solver="lsmr",
        max_nfev=args.max_nfev,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        verbose=0,
    )
    depth = np.asarray(result.x[:n], dtype=float)
    radius = np.asarray(result.x[n:], dtype=float)
    before_groups = residual_groups(
        factors,
        depth0,
        radius0,
        args.accel_sigma_m,
        args.radius_accel_sigma_m,
        args.contact_sigma_m,
        args.penetration_sigma_m,
    )
    after_groups = residual_groups(
        factors,
        depth,
        radius,
        args.accel_sigma_m,
        args.radius_accel_sigma_m,
        args.contact_sigma_m,
        args.penetration_sigma_m,
    )
    qc = {
        "optimizer": {
            "method": "scipy.optimize.least_squares",
            "loss": "soft_l1",
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "initial_cost": float(0.5 * np.dot(initial_residual, initial_residual)),
            "final_cost": float(result.cost),
            "optimality": float(result.optimality),
            "active_frames": n,
            "residual_count": int(initial_residual.size),
            "jacobian_nonzeros": int(sparsity.nnz),
            "jacobian_density": float(sparsity.nnz / (sparsity.shape[0] * sparsity.shape[1])),
        },
        "residual_rms_before": {name: rms(values) for name, values in before_groups.items()},
        "residual_rms_after": {name: rms(values) for name, values in after_groups.items()},
        "physics_metrics_before": raw_physics_metrics(factors, depth0, radius0),
        "physics_metrics_after": raw_physics_metrics(factors, depth, radius),
        "depth_delta_m_iqr": [
            float(np.percentile(depth - depth0, 25)),
            float(np.percentile(depth - depth0, 75)),
        ],
        "radius_delta_m_iqr": [
            float(np.percentile(radius - radius0, 25)),
            float(np.percentile(radius - radius0, 75)),
        ],
        "max_abs_depth_delta_m": float(np.max(np.abs(depth - depth0))),
        "max_abs_radius_delta_m": float(np.max(np.abs(radius - radius0))),
    }
    if not result.success:
        qc["status"] = "incomplete_optimizer_not_converged"
        qc["reason"] = str(result.message)
        raise RuntimeError(json.dumps(qc, indent=2))
    return depth, radius, qc


def apply_refinement(frames: list[dict], active: list[int], factors: list[FrameFactors], depths: np.ndarray, radii: np.ndarray) -> None:
    centers = centers_world(factors, depths)
    for j, frame_idx in enumerate(active):
        frame = frames[frame_idx]
        obj = frame["object"]
        old = {
            "depth_m": float(obj["depth_m"]),
            "radius_m": float(obj["radius_m"]),
            "center_world_m": obj["center_world_m"],
            "pose_status": obj.get("pose_status"),
        }
        point_source = factors[j].ray_source * float(depths[j])
        obj["center_source_camera_m"] = point_source.astype(float).tolist()
        obj["center_world_m"] = centers[j].astype(float).tolist()
        obj["depth_m"] = float(depths[j])
        obj["radius_m"] = float(radii[j])
        obj["pose_status"] = "v2_factor_refined_contact_nonpenetration_temporal"
        obj["pose_type"] = "deformable_object_centroid_with_spherical_extent"
        obj["v2_physics"] = {
            "previous_v1": old,
            "depth_delta_m": float(depths[j] - factors[j].depth0_m),
            "radius_delta_m": float(radii[j] - factors[j].radius0_m),
            "contact_points": int(len(factors[j].contact_points_world)),
            "penetration_candidate_points": int(len(factors[j].penetration_points_world)),
            "terms": [
                "depth_prior",
                "mask_radius",
                "temporal_center_acceleration",
                "temporal_radius_acceleration",
                "contact_surface",
                "hand_object_nonpenetration",
            ],
        }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.annotations)
    frames = payload["frames"]
    cap, info = open_video(args.clip)
    cap.release()
    active = active_object_indices(frames)
    intrinsics = first_intrinsics(frames)
    factors = build_factors(frames, active, intrinsics, args.contact_px, args.max_penetration_points)
    depths, radii, qc = optimize(args, frames, factors)
    apply_refinement(frames, active, factors, depths, radii)
    annotations_path = args.output_dir / "annotations_v2_physics.json"
    write_json(annotations_path, {"frames": frames})
    render = RenderSpec(args.render_width, int(round(args.render_width * info.height / info.width)), info.fps)
    if args.render:
        render_outputs(args, frames, render)
    qc.update(
        {
            "status": "ok",
            "clip": str(args.clip),
            "source_annotations": str(args.annotations),
            "output_annotations": str(annotations_path),
            "render": render.__dict__,
            "elapsed_s": time.time() - started,
            "outputs": {
                "annotations": str(annotations_path),
                "overlay": str(args.output_dir / "overlay_mano_object.mp4"),
                "reconstruction_3d": str(args.output_dir / "reconstruction_3d_world.mp4"),
                "side_by_side": str(args.output_dir / "side_by_side.mp4"),
            },
        }
    )
    (args.output_dir / "qc_v2_physics.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, default=Path("outputs/examples/tomato_v1_full/fused/annotations_v1_full.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/examples/tomato_v2_physics"))
    parser.add_argument("--mano-right", type=Path, default=DEFAULT_MANO_RIGHT)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-px", type=float, default=55.0)
    parser.add_argument("--max-penetration-points", type=int, default=64)
    parser.add_argument("--accel-sigma-m", type=float, default=0.030)
    parser.add_argument("--radius-accel-sigma-m", type=float, default=0.006)
    parser.add_argument("--contact-sigma-m", type=float, default=0.010)
    parser.add_argument("--penetration-sigma-m", type=float, default=0.020)
    parser.add_argument("--max-nfev", type=int, default=240)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
