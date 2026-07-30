#!/usr/bin/env python3
"""Build a motion-aware V19 rigid-object pose trajectory from noisy visible-pose rows.

This script is prediction-side: it consumes only V19 visible-depth object pose
measurements, annotations, and the completed object mesh.  It differs from
``solve_v19_rigid_object_pose_graph.py`` because it smooths the physical SE(3)
trajectory itself.  The existing graph smooths correction deltas around zero,
which returns the raw per-frame visible ICP pose when no trusted contact or
nonpenetration target exists.  Here the measured pose is a noisy observation and
the physical variable is a temporally smooth object trajectory.

The prior is acceleration, not stationarity.  Constant translation/rotation
velocity is allowed; high second-difference motion is penalized at the scale of
the pose measurement noise estimated from visible-depth residuals.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

POSE_MEASUREMENT_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
    "corrected_temporal_rigid_pose_graph",
}
DIRECT_MEASUREMENT_STATUSES = {
    "fit_to_visible_depth_samples",
    "fit_to_visible_depth_archive_vertices",
}
# Render-state construction accepts corrected temporal rigid-pose rows under the
# stable V19 status below.  The smoother-specific mechanism is recorded in
# temporal_pose_graph.pose_source/method, not by inventing a renderer-invisible
# status string.
SMOOTHED_POSE_STATUS = "corrected_temporal_rigid_pose_graph"
COMPLETED_POSE_STATUS = "completed_temporal_rigid_pose_uncertain"


@dataclass(frozen=True)
class PoseMeas:
    frame_idx: int
    source_row: dict[str, Any]
    R_meas: np.ndarray
    t_meas: np.ndarray
    sigma_t: float
    sigma_r: float
    visible_sample_count: int
    observed_points_world: np.ndarray


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
    }


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh geometry in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid mesh {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise RuntimeError("sampled mesh points invalid")
    return pts


def as_pose(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    R = np.asarray(row.get("rotation_world_from_completed_canonical_matrix"), dtype=float)
    t = np.asarray(row.get("translation_world_m"), dtype=float)
    if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
        return R, t
    return None


def apply_pose(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)[None, :]


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    query = np.asarray(query, dtype=float)
    target = np.asarray(target, dtype=float)
    if query.ndim != 2 or target.ndim != 2 or query.shape[1:] != (3,) or target.shape[1:] != (3,) or len(query) == 0 or len(target) == 0:
        return {"count": int(len(query)) if query.ndim == 2 else 0, "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
    d, _ = cKDTree(target).query(query, k=1, workers=-1)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90.0)),
        "p95_m": float(np.percentile(d, 95.0)),
        "mean_m": float(np.mean(d)),
        "max_m": float(np.max(d)),
    }


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def observed_points(frame: dict[str, Any], object_id: str) -> np.ndarray:
    obj = annotation_object(frame, object_id)
    if obj is None:
        return np.empty((0, 3), dtype=float)
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    pts = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
    if pts.ndim == 2 and pts.shape[1] == 3 and np.isfinite(pts).all():
        return pts
    return np.empty((0, 3), dtype=float)


def pose_sigma(row: dict[str, Any], object_radius_m: float, args: argparse.Namespace) -> tuple[float, float]:
    final = row.get("observed_to_mesh_final") if isinstance(row.get("observed_to_mesh_final"), dict) else {}
    finite = []
    for key in ("median_m", "p95_m"):
        value = final.get(key)
        if value is not None and math.isfinite(float(value)):
            finite.append(float(value))
    residual_scale = max(finite) if finite else float(args.default_pose_sigma_m)
    visible_n = int(row.get("visible_sample_count") or final.get("count") or 0)
    sample_factor = math.sqrt(max(1.0, min(float(visible_n), 400.0)) / 100.0)
    sigma_t = residual_scale / max(1.0, sample_factor)
    sigma_t = float(np.clip(max(float(args.min_pose_sigma_m), sigma_t), float(args.min_pose_sigma_m), float(args.max_pose_sigma_m)))
    sigma_r = sigma_t / max(float(object_radius_m), 1.0e-3)
    sigma_r = float(np.clip(sigma_r, float(args.min_pose_rotation_sigma_rad), float(args.max_pose_rotation_sigma_rad)))
    return sigma_t, sigma_r


def is_direct_measurement(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") in DIRECT_MEASUREMENT_STATUSES:
        return True
    if str(row.get("pose_measurement_status") or "") in DIRECT_MEASUREMENT_STATUSES:
        return True
    tpg = row.get("temporal_pose_graph") if isinstance(row.get("temporal_pose_graph"), dict) else {}
    return bool(tpg.get("direct_visible_measurement"))


def build_measurements(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any], mesh: trimesh.Trimesh) -> tuple[list[PoseMeas], list[dict[str, Any]]]:
    frames = {int(fr.get("frame_idx")): fr for fr in annotations.get("frames", []) if isinstance(fr, dict) and fr.get("frame_idx") is not None}
    radius = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)) / 2.0)
    out: list[PoseMeas] = []
    skipped: list[dict[str, Any]] = []
    for row in pose_report.get("pose_rows", []) if isinstance(pose_report.get("pose_rows"), list) else []:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        idx = int(row["frame_idx"])
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        if not is_direct_measurement(row):
            skipped.append({"frame_idx": idx, "reason": "not_direct_visible_pose_measurement", "status": row.get("status"), "pose_measurement_status": row.get("pose_measurement_status")})
            continue
        pose = as_pose(row)
        if pose is None:
            skipped.append({"frame_idx": idx, "reason": "missing_pose_matrix"})
            continue
        frame = frames.get(idx)
        if frame is None:
            skipped.append({"frame_idx": idx, "reason": "missing_annotation_frame"})
            continue
        obs = observed_points(frame, args.object_id)
        if len(obs) < int(args.min_visible_points):
            skipped.append({"frame_idx": idx, "reason": "insufficient_visible_points", "count": int(len(obs))})
            continue
        sigma_t, sigma_r = pose_sigma(row, radius, args)
        R, t = pose
        out.append(PoseMeas(idx, row, R, t, sigma_t, sigma_r, int(row.get("visible_sample_count") or len(obs)), obs.astype(float)))
    out.sort(key=lambda m: m.frame_idx)
    if len(out) < int(args.min_graph_frames):
        raise RuntimeError(f"only {len(out)} usable direct pose measurements; skipped={skipped[:8]}")
    return out, skipped


def unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    arr = x.reshape(n, 6)
    return arr[:, :3], arr[:, 3:]


def corrected_arrays(meas: list[PoseMeas], x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    rot_delta, trans_delta = unpack(x, len(meas))
    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for i, m in enumerate(meas):
        Rs.append(Rotation.from_rotvec(rot_delta[i]).as_matrix() @ m.R_meas)
        ts.append(m.t_meas + trans_delta[i])
    return Rs, np.vstack(ts)


def rot_step(R_next: np.ndarray, R_prev: np.ndarray, gap: float) -> np.ndarray:
    return Rotation.from_matrix(R_next @ R_prev.T).as_rotvec() / float(max(gap, 1.0))


def residual_vector(x: np.ndarray, meas: list[PoseMeas], args: argparse.Namespace) -> np.ndarray:
    rot_delta, trans_delta = unpack(x, len(meas))
    Rs, ts = corrected_arrays(meas, x)
    residuals: list[np.ndarray] = []
    for i, m in enumerate(meas):
        residuals.append(trans_delta[i] / float(m.sigma_t))
        residuals.append(rot_delta[i] / float(m.sigma_r))
    sigma_t_acc = float(args.sigma_translation_accel_m)
    sigma_r_acc = float(args.sigma_rotation_accel_rad)
    if sigma_t_acc <= 0 or sigma_r_acc <= 0:
        raise RuntimeError("acceleration sigmas must be positive")
    for i in range(1, len(meas) - 1):
        gap0 = max(1, meas[i].frame_idx - meas[i - 1].frame_idx)
        gap1 = max(1, meas[i + 1].frame_idx - meas[i].frame_idx)
        v_prev_t = (ts[i] - ts[i - 1]) / float(gap0)
        v_next_t = (ts[i + 1] - ts[i]) / float(gap1)
        accel_t = v_next_t - v_prev_t
        v_prev_r = rot_step(Rs[i], Rs[i - 1], float(gap0))
        v_next_r = rot_step(Rs[i + 1], Rs[i], float(gap1))
        accel_r = v_next_r - v_prev_r
        scale = math.sqrt(float(max(gap0, gap1)))
        residuals.append(accel_t / (sigma_t_acc * scale))
        residuals.append(accel_r / (sigma_r_acc * scale))
    return np.concatenate([r.reshape(-1) for r in residuals]).astype(float)


def residual_sparsity(n: int) -> sparse.csr_matrix:
    cols = n * 6
    entries: list[tuple[int, int]] = []
    row = 0
    def add(rows: range, frame_ids: list[int]) -> None:
        for rr in rows:
            for fi in frame_ids:
                for cc in range(fi * 6, fi * 6 + 6):
                    entries.append((rr, cc))
    for i in range(n):
        add(range(row, row + 3), [i]); row += 3
        add(range(row, row + 3), [i]); row += 3
    for i in range(1, n - 1):
        add(range(row, row + 3), [i - 1, i, i + 1]); row += 3
        add(range(row, row + 3), [i - 1, i, i + 1]); row += 3
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols))


def surface_metrics(meas: list[PoseMeas], mesh_samples: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    Rs, ts = corrected_arrays(meas, x)
    obs_meds: list[float] = []
    mesh_meds: list[float] = []
    per_frame: dict[str, Any] = {}
    for m, R, t in zip(meas, Rs, ts):
        mesh_world = apply_pose(mesh_samples, R, t)
        o2m = nearest_summary(m.observed_points_world, mesh_world)
        m2o = nearest_summary(mesh_world, m.observed_points_world)
        if o2m["median_m"] is not None:
            obs_meds.append(float(o2m["median_m"]))
        if m2o["median_m"] is not None:
            mesh_meds.append(float(m2o["median_m"]))
        per_frame[str(m.frame_idx)] = {"observed_to_mesh": o2m, "mesh_to_observed": m2o}
    return {
        "observed_to_mesh_median_m": numeric_summary(obs_meds),
        "mesh_to_observed_median_m": numeric_summary(mesh_meds),
        "per_frame": per_frame,
    }


def delta_summary(x: np.ndarray, meas: list[PoseMeas]) -> dict[str, Any]:
    rot_delta, trans_delta = unpack(x, len(meas))
    trans_norm = np.linalg.norm(trans_delta, axis=1)
    rot_norm = np.linalg.norm(rot_delta, axis=1)
    Rs, ts = corrected_arrays(meas, x)
    accel_t: list[float] = []
    accel_r: list[float] = []
    for i in range(1, len(meas) - 1):
        gap0 = max(1, meas[i].frame_idx - meas[i - 1].frame_idx)
        gap1 = max(1, meas[i + 1].frame_idx - meas[i].frame_idx)
        accel_t.append(float(np.linalg.norm((ts[i + 1] - ts[i]) / gap1 - (ts[i] - ts[i - 1]) / gap0)))
        accel_r.append(float(np.linalg.norm(rot_step(Rs[i + 1], Rs[i], gap1) - rot_step(Rs[i], Rs[i - 1], gap0))))
    return {
        "translation_delta_norm_m": numeric_summary(trans_norm),
        "rotation_delta_norm_rad": numeric_summary(rot_norm),
        "rotation_delta_norm_deg": numeric_summary([math.degrees(x) for x in rot_norm]),
        "physical_translation_accel_m_per_frame2": numeric_summary(accel_t),
        "physical_rotation_accel_rad_per_frame2": numeric_summary(accel_r),
    }


def build_pose_rows(original_rows: list[dict[str, Any]], meas: list[PoseMeas], x: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_idx = {m.frame_idx: i for i, m in enumerate(meas)}
    Rs, ts = corrected_arrays(meas, x)
    direct_frames = sorted(by_idx)
    out: list[dict[str, Any]] = []
    completed = {"enabled": True, "direct_row_count": len(direct_frames), "completed_row_count": 0, "mode_counts": {}}
    def add_mode(mode: str) -> None:
        completed["mode_counts"][mode] = int(completed["mode_counts"].get(mode, 0)) + 1
    rots_for_slerp = Rotation.from_matrix(np.stack([Rs[by_idx[f]] for f in direct_frames], axis=0))
    slerp = Slerp(direct_frames, rots_for_slerp)
    t_stack = np.stack([ts[by_idx[f]] for f in direct_frames], axis=0)
    for row in original_rows:
        if not isinstance(row, dict) or "frame_idx" not in row:
            continue
        idx = int(row["frame_idx"])
        new_row = dict(row)
        if idx in by_idx:
            i = by_idx[idx]
            new_row["status"] = SMOOTHED_POSE_STATUS
            new_row["pose_measurement_status"] = row.get("status")
            new_row["rotation_world_from_completed_canonical_matrix"] = Rs[i].astype(float).tolist()
            new_row["translation_world_m"] = ts[i].astype(float).tolist()
            new_row["temporal_pose_graph"] = {
                "pose_source": "physical_se3_acceleration_smoother_direct_visible_measurement",
                "method": "smooth_v19_rigid_object_pose_trajectory",
                "direct_visible_measurement": True,
                "original_pose_status": row.get("status"),
                "original_pose_measurement_status": row.get("pose_measurement_status"),
                "uncertainty": "visible-depth object pose measurement smoothed by physical acceleration prior; no hand/contact/GT used",
            }
        else:
            # Fill non-direct frames for render continuity only.
            if not direct_frames:
                out.append(new_row); continue
            if idx <= direct_frames[0]:
                use = direct_frames[0]; R_fill = Rs[by_idx[use]]; t_fill = ts[by_idx[use]]; mode = "nearest_visible_pose_hold"; bracket = [use]
            elif idx >= direct_frames[-1]:
                use = direct_frames[-1]; R_fill = Rs[by_idx[use]]; t_fill = ts[by_idx[use]]; mode = "nearest_visible_pose_hold"; bracket = [use]
            else:
                hi_pos = int(np.searchsorted(np.asarray(direct_frames), idx, side="right"))
                lo = direct_frames[hi_pos - 1]; hi = direct_frames[hi_pos]
                alpha = (idx - lo) / float(max(1, hi - lo))
                R_fill = slerp([float(idx)]).as_matrix()[0]
                t_fill = (1.0 - alpha) * t_stack[hi_pos - 1] + alpha * t_stack[hi_pos]
                mode = "interpolated_between_visible_pose_observations"; bracket = [lo, hi]
            new_row["status"] = COMPLETED_POSE_STATUS
            new_row["pose_measurement_status"] = row.get("status")
            new_row["rotation_world_from_completed_canonical_matrix"] = R_fill.astype(float).tolist()
            new_row["translation_world_m"] = t_fill.astype(float).tolist()
            new_row["temporal_pose_graph"] = {
                "pose_source": mode,
                "method": "smooth_v19_rigid_object_pose_trajectory",
                "bracket_visible_pose_frames": bracket,
                "direct_visible_measurement": False,
                "uncertainty": "temporal completion from smoothed direct visible object poses; not a direct object mask/depth observation",
            }
            completed["completed_row_count"] = int(completed["completed_row_count"]) + 1
            add_mode(mode)
        out.append(new_row)
    return out, completed


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    mesh = load_mesh(args.completed_mesh)
    meas, skipped = build_measurements(args, annotations, pose_report, mesh)
    sigma_t_med = float(np.median([m.sigma_t for m in meas]))
    sigma_r_med = float(np.median([m.sigma_r for m in meas]))
    translation_accel_auto = args.sigma_translation_accel_m is None
    rotation_accel_auto = args.sigma_rotation_accel_rad is None
    if translation_accel_auto:
        args.sigma_translation_accel_m = sigma_t_med
    if rotation_accel_auto:
        args.sigma_rotation_accel_rad = sigma_r_med
    x0 = np.zeros(len(meas) * 6, dtype=float)
    before = residual_vector(x0, meas, args)
    jac = residual_sparsity(len(meas))
    result = least_squares(
        lambda x: residual_vector(x, meas, args),
        x0,
        jac_sparsity=jac,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after = residual_vector(result.x, meas, args)
    mesh_samples = deterministic_sample_mesh(mesh, int(args.surface_metric_sample_count), int(args.seed) + 91)
    before_surface = surface_metrics(meas, mesh_samples, x0)
    after_surface = surface_metrics(meas, mesh_samples, result.x)
    pose_rows, completion = build_pose_rows(pose_report.get("pose_rows", []), meas, result.x, args)
    surface_before = before_surface["observed_to_mesh_median_m"].get("median")
    surface_after = after_surface["observed_to_mesh_median_m"].get("median")
    surface_degradation = None
    if surface_before is not None and surface_after is not None:
        surface_degradation = float(surface_after) - float(surface_before)
    report = dict(pose_report)
    report.update({
        "method": "smooth_v19_rigid_object_pose_trajectory",
        "status": "smoothed_physical_se3_acceleration_pose_candidate" if result.success else "smoothed_physical_se3_acceleration_optimizer_incomplete",
        "annotation_ready": bool(result.success),
        "claim_scope": "Prediction-side physical SE(3) acceleration smoothing of visible-depth rigid-object pose observations. Allows real object motion; suppresses high-acceleration ICP jitter. No hand/contact/GT consumed.",
        "object_id": str(args.object_id),
        "inputs": {
            **(pose_report.get("inputs") if isinstance(pose_report.get("inputs"), dict) else {}),
            "annotations": str(args.annotations),
            "source_pose_report": str(args.pose_report),
            "completed_mesh": str(args.completed_mesh),
        },
        "graph_frames": [m.frame_idx for m in meas],
        "graph_frame_count": len(meas),
        "skipped_pose_observations": skipped,
        "parameters": {
            "sigma_translation_accel_m_per_frame2": float(args.sigma_translation_accel_m),
            "sigma_rotation_accel_rad_per_frame2": float(args.sigma_rotation_accel_rad),
            "sigma_translation_accel_source": "median_visible_pose_sigma" if translation_accel_auto else "explicit_arg",
            "sigma_rotation_accel_source": "median_visible_pose_rotation_sigma" if rotation_accel_auto else "explicit_arg",
            "median_pose_sigma_t_m": sigma_t_med,
            "median_pose_sigma_r_rad": sigma_r_med,
            "surface_metric_sample_count": int(args.surface_metric_sample_count),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "residual_rms_before": float(np.sqrt(np.mean(before * before))),
            "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        },
        "correction_summary": delta_summary(result.x, meas),
        "surface_before": before_surface,
        "surface_after": after_surface,
        "surface_observed_to_mesh_median_degradation_m": surface_degradation,
        "full_timeline_rigid_pose_completion": completion,
        "pose_rows": pose_rows,
        "outputs": {"pose_report": str(args.output_report)},
        "elapsed_s": float(time.time() - started),
    })
    write_json(args.output_report, report)
    print(json.dumps({
        "status": report["status"],
        "graph_frame_count": report["graph_frame_count"],
        "parameters": report["parameters"],
        "optimizer": report["optimizer"],
        "correction_summary": report["correction_summary"],
        "surface_observed_to_mesh_median_degradation_m": surface_degradation,
    }, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--min-graph-frames", type=int, default=8)
    p.add_argument("--min-visible-points", type=int, default=20)
    p.add_argument("--min-pose-sigma-m", type=float, default=0.004)
    p.add_argument("--max-pose-sigma-m", type=float, default=0.045)
    p.add_argument("--default-pose-sigma-m", type=float, default=0.018)
    p.add_argument("--min-pose-rotation-sigma-rad", type=float, default=0.035)
    p.add_argument("--max-pose-rotation-sigma-rad", type=float, default=0.35)
    p.add_argument("--sigma-translation-accel-m", type=float, default=None, help="Physical translation acceleration sigma in m/frame^2. Default uses median visible-pose sigma.")
    p.add_argument("--sigma-rotation-accel-rad", type=float, default=None, help="Physical rotation acceleration sigma in rad/frame^2. Default uses median visible-pose rotation sigma.")
    p.add_argument("--surface-metric-sample-count", type=int, default=2500)
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--seed", type=int, default=2203)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
