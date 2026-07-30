#!/usr/bin/env python3
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
}
CORRECTED_POSE_STATUS = "corrected_temporal_rigid_pose_graph"
COMPLETED_POSE_STATUS = "completed_temporal_rigid_pose_uncertain"


@dataclass(frozen=True)
class PoseObservation:
    frame_idx: int
    source_row: dict[str, Any]
    rotation_world_from_canonical: np.ndarray
    translation_world_m: np.ndarray
    translation_sigma_m: float
    rotation_sigma_rad: float
    visible_sample_count: int
    observed_points_world: np.ndarray
    nonpenetration_target_world_m: np.ndarray | None
    nonpenetration_weight: float
    nonpenetration_source_rows: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=float)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"invalid {name}: expected {shape}, got {arr.shape}")
    return arr


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no triangle mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0 or len(geom.faces) == 0:
        raise RuntimeError(f"invalid mesh: {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def deterministic_sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, int(count), seed=rng)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise RuntimeError("sampled mesh points invalid")
    return pts


def nearest_summary(query: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
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


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
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
    }


def apply_pose(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation[None, :]


def corrected_pose(obs: PoseObservation, rot_delta: np.ndarray, trans_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_corr = Rotation.from_rotvec(rot_delta).as_matrix() @ obs.rotation_world_from_canonical
    t_corr = obs.translation_world_m + trans_delta
    return r_corr, t_corr


def pose_row_sigma(row: dict[str, Any], object_radius_m: float, args: argparse.Namespace) -> tuple[float, float]:
    final = row.get("observed_to_mesh_final") if isinstance(row.get("observed_to_mesh_final"), dict) else {}
    median = final.get("median_m")
    p95 = final.get("p95_m")
    visible_n = int(row.get("visible_sample_count") or final.get("count") or 0)
    finite = [float(x) for x in (median, p95) if x is not None and math.isfinite(float(x))]
    residual_scale = max(finite) if finite else float(args.default_pose_sigma_m)
    # ICP pose is a measurement of the visible surface. More visible points tighten it, but not below
    # the physical depth/camera floor; high residuals widen it so the graph can expose systematic conflict.
    sample_factor = math.sqrt(max(1.0, min(float(visible_n), 400.0)) / 100.0)
    sigma_t = residual_scale / max(1.0, sample_factor)
    sigma_t = float(np.clip(max(float(args.min_pose_sigma_m), sigma_t), float(args.min_pose_sigma_m), float(args.max_pose_sigma_m)))
    sigma_r = sigma_t / max(float(object_radius_m), 1.0e-3)
    sigma_r = float(np.clip(sigma_r, float(args.min_pose_rotation_sigma_rad), float(args.max_pose_rotation_sigma_rad)))
    return sigma_t, sigma_r


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    object_aliases = {object_id, f"object:{object_id}"}
    for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
        if obj.get("object_id") in object_aliases or obj.get("track_id") == object_id:
            return obj
    legacy = frame.get("object") if isinstance(frame.get("object"), dict) else None
    if legacy and (legacy.get("object_id") in (None, *object_aliases) or legacy.get("track_id") == object_id):
        return legacy
    return None


def constraint_targets(path: Path | None, max_target_m: float, accepted_states: set[str]) -> dict[int, tuple[np.ndarray, float, int]]:
    if path is None:
        return {}
    data = load_json(path)
    by_frame: dict[int, list[np.ndarray]] = {}
    for row in data.get("constraint_rows", []) if isinstance(data, dict) else []:
        state = str(row.get("candidate_application_state") or "")
        if state not in accepted_states:
            continue
        cand = np.asarray(row.get("candidate_translation_world_m") or [], dtype=float)
        if cand.shape != (3,) or not np.isfinite(cand).all():
            continue
        norm = float(np.linalg.norm(cand))
        if norm <= 0.0:
            continue
        # The row's vector moves the hand out of the object. The object-pose graph can only use the
        # opposite direction as bounded pressure, never as a hard truth about the object state.
        target = -cand
        tnorm = float(np.linalg.norm(target))
        if tnorm > max_target_m:
            target = target * (max_target_m / max(tnorm, 1.0e-12))
        by_frame.setdefault(int(row["frame_idx"]), []).append(target.astype(float))
    out: dict[int, tuple[np.ndarray, float, int]] = {}
    for idx, targets in by_frame.items():
        stack = np.vstack(targets)
        med = np.median(stack, axis=0)
        weight = min(1.0, math.sqrt(len(targets)))
        out[int(idx)] = (med.astype(float), float(weight), int(len(targets)))
    return out


def build_observations(args: argparse.Namespace, annotations: dict[str, Any], pose_report: dict[str, Any], mesh: trimesh.Trimesh) -> tuple[list[PoseObservation], list[dict[str, Any]], dict[int, tuple[np.ndarray, float, int]]]:
    frames_by_idx = {int(frame.get("frame_idx")): frame for frame in annotations.get("frames", []) if isinstance(frame, dict) and frame.get("frame_idx") is not None}
    radius = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)) / 2.0)
    accepted = set(args.nonpenetration_states)
    targets = constraint_targets(args.constraint_report, float(args.max_nonpenetration_target_m), accepted)
    observations: list[PoseObservation] = []
    skipped: list[dict[str, Any]] = []
    for row in pose_report.get("pose_rows", []) if isinstance(pose_report.get("pose_rows"), list) else []:
        if str(row.get("status") or "") not in POSE_MEASUREMENT_STATUSES:
            continue
        idx = int(row["frame_idx"])
        if args.frame_start is not None and idx < int(args.frame_start):
            continue
        if args.frame_end is not None and idx > int(args.frame_end):
            continue
        try:
            rot = as_array(row.get("rotation_world_from_completed_canonical_matrix"), (3, 3), f"pose rotation frame {idx}")
            trans = as_array(row.get("translation_world_m"), (3,), f"pose translation frame {idx}")
            frame = frames_by_idx[idx]
            obj = annotation_object(frame, args.object_id)
            if obj is None:
                raise RuntimeError("annotation object missing")
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            observed = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
            if observed.ndim != 2 or observed.shape[1] != 3 or len(observed) < int(args.min_visible_points) or not np.isfinite(observed).all():
                raise RuntimeError(f"insufficient visible surfels: {observed.shape}")
            sigma_t, sigma_r = pose_row_sigma(row, radius, args)
            target_tuple = targets.get(idx)
            if target_tuple is None:
                target, weight, target_rows = None, 0.0, 0
            else:
                target, weight, target_rows = target_tuple
            observations.append(
                PoseObservation(
                    frame_idx=idx,
                    source_row=row,
                    rotation_world_from_canonical=rot,
                    translation_world_m=trans,
                    translation_sigma_m=sigma_t,
                    rotation_sigma_rad=sigma_r,
                    visible_sample_count=int(row.get("visible_sample_count") or len(observed)),
                    observed_points_world=observed.astype(float),
                    nonpenetration_target_world_m=target,
                    nonpenetration_weight=weight,
                    nonpenetration_source_rows=target_rows,
                )
            )
        except Exception as exc:
            skipped.append({"frame_idx": idx, "reason": str(exc)})
    if len(observations) < int(args.min_graph_frames):
        raise RuntimeError(f"only {len(observations)} usable pose observations; skipped={skipped[:8]}")
    observations.sort(key=lambda obs: obs.frame_idx)
    return observations, skipped, targets


def unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    arr = x.reshape(n, 6)
    return arr[:, :3], arr[:, 3:6]


def residual_vector(x: np.ndarray, observations: list[PoseObservation], args: argparse.Namespace) -> np.ndarray:
    rot_delta, trans_delta = unpack(x, len(observations))
    residuals: list[np.ndarray] = []
    for i, obs in enumerate(observations):
        residuals.append(trans_delta[i] / obs.translation_sigma_m)
        residuals.append(rot_delta[i] / obs.rotation_sigma_rad)
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            residuals.append(
                math.sqrt(obs.nonpenetration_weight)
                * (trans_delta[i] - obs.nonpenetration_target_world_m)
                / float(args.sigma_nonpenetration_target_m)
            )
    # Smooth the correction field, not the physical object trajectory, so real object motion measured by ICP is preserved.
    for i in range(1, len(observations)):
        gap = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        scale = math.sqrt(float(gap))
        residuals.append((trans_delta[i] - trans_delta[i - 1]) / (float(args.sigma_translation_delta_step_m) * scale))
        residuals.append((rot_delta[i] - rot_delta[i - 1]) / (float(args.sigma_rotation_delta_step_rad) * scale))
    for i in range(1, len(observations) - 1):
        gap0 = max(1, observations[i].frame_idx - observations[i - 1].frame_idx)
        gap1 = max(1, observations[i + 1].frame_idx - observations[i].frame_idx)
        # Use a gap-scaled second difference of the correction field as a soft acceleration prior.
        prev_v_t = (trans_delta[i] - trans_delta[i - 1]) / float(gap0)
        next_v_t = (trans_delta[i + 1] - trans_delta[i]) / float(gap1)
        prev_v_r = (rot_delta[i] - rot_delta[i - 1]) / float(gap0)
        next_v_r = (rot_delta[i + 1] - rot_delta[i]) / float(gap1)
        scale = math.sqrt(float(max(gap0, gap1)))
        residuals.append((next_v_t - prev_v_t) / (float(args.sigma_translation_delta_accel_m) * scale))
        residuals.append((next_v_r - prev_v_r) / (float(args.sigma_rotation_delta_accel_rad) * scale))
    return np.concatenate([r.reshape(-1) for r in residuals]).astype(float)


def residual_sparsity(observations: list[PoseObservation]) -> sparse.csr_matrix:
    n = len(observations)
    cols = n * 6
    entries: list[tuple[int, int]] = []
    row = 0

    def add(rows: range, frame_ids: list[int]) -> None:
        for rr in rows:
            for fi in frame_ids:
                for cc in range(fi * 6, fi * 6 + 6):
                    entries.append((rr, cc))

    for i, obs in enumerate(observations):
        add(range(row, row + 3), [i])
        row += 3
        add(range(row, row + 3), [i])
        row += 3
        if obs.nonpenetration_target_world_m is not None and obs.nonpenetration_weight > 0.0:
            add(range(row, row + 3), [i])
            row += 3
    for i in range(1, n):
        add(range(row, row + 3), [i - 1, i])
        row += 3
        add(range(row, row + 3), [i - 1, i])
        row += 3
    for i in range(1, n - 1):
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
        add(range(row, row + 3), [i - 1, i, i + 1])
        row += 3
    rr, cc = np.asarray(entries, dtype=np.int64).T
    return sparse.csr_matrix((np.ones(len(entries), dtype=bool), (rr, cc)), shape=(row, cols))


def target_residual_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    _, trans_delta = unpack(x, len(observations))
    vals = []
    norms = []
    for i, obs in enumerate(observations):
        if obs.nonpenetration_target_world_m is None or obs.nonpenetration_weight <= 0.0:
            continue
        diff = trans_delta[i] - obs.nonpenetration_target_world_m
        vals.append(float(np.linalg.norm(diff)))
        norms.append(float(np.linalg.norm(obs.nonpenetration_target_world_m)))
    return {"target_residual_norm_m": numeric_summary(vals), "target_norm_m": numeric_summary(norms), "target_frame_count": len(vals)}


def correction_summary(x: np.ndarray, observations: list[PoseObservation]) -> dict[str, Any]:
    rot_delta, trans_delta = unpack(x, len(observations))
    gaps = np.asarray([observations[i].frame_idx - observations[i - 1].frame_idx for i in range(1, len(observations))], dtype=float)
    trans_norm = np.linalg.norm(trans_delta, axis=1)
    rot_norm = np.linalg.norm(rot_delta, axis=1)
    step_t = np.linalg.norm(np.diff(trans_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    step_r = np.linalg.norm(np.diff(rot_delta, axis=0), axis=1) if len(observations) > 1 else np.asarray([])
    return {
        "translation_delta_norm_m": numeric_summary(trans_norm),
        "rotation_delta_norm_rad": numeric_summary(rot_norm),
        "translation_delta_step_norm_m": numeric_summary(step_t),
        "rotation_delta_step_norm_rad": numeric_summary(step_r),
        "visible_frame_gap": numeric_summary(gaps),
    }


def surface_metrics(observations: list[PoseObservation], mesh_samples: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    rot_delta, trans_delta = unpack(x, len(observations))
    obs_to_mesh_medians: list[float] = []
    mesh_to_obs_medians: list[float] = []
    per_frame: dict[str, Any] = {}
    for i, obs in enumerate(observations):
        r, t = corrected_pose(obs, rot_delta[i], trans_delta[i])
        mesh_world = apply_pose(mesh_samples, r, t)
        o2m = nearest_summary(obs.observed_points_world, mesh_world)
        m2o = nearest_summary(mesh_world, obs.observed_points_world)
        if o2m["median_m"] is not None:
            obs_to_mesh_medians.append(float(o2m["median_m"]))
        if m2o["median_m"] is not None:
            mesh_to_obs_medians.append(float(m2o["median_m"]))
        per_frame[str(obs.frame_idx)] = {"observed_to_mesh": o2m, "mesh_to_observed": m2o}
    return {
        "observed_to_mesh_median_m": numeric_summary(obs_to_mesh_medians),
        "mesh_to_observed_median_m": numeric_summary(mesh_to_obs_medians),
        "per_frame": per_frame,
    }


def build_pose_rows(
    original_pose_rows: list[dict[str, Any]],
    observations: list[PoseObservation],
    x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs_by_idx = {obs.frame_idx: i for i, obs in enumerate(observations)}
    rot_delta, trans_delta = unpack(x, len(observations))
    corrected_by_idx: dict[int, tuple[np.ndarray, np.ndarray, PoseObservation, np.ndarray, np.ndarray]] = {}
    out: list[dict[str, Any]] = []
    for row in original_pose_rows:
        idx = int(row.get("frame_idx", -1))
        if idx not in obs_by_idx:
            out.append(dict(row))
            continue
        i = obs_by_idx[idx]
        obs = observations[i]
        r, t = corrected_pose(obs, rot_delta[i], trans_delta[i])
        corrected_by_idx[idx] = (r, t, obs, rot_delta[i], trans_delta[i])
        new_row = dict(row)
        new_row["status"] = CORRECTED_POSE_STATUS
        new_row["pose_measurement_status"] = row.get("status")
        new_row["rotation_world_from_completed_canonical_matrix"] = r.astype(float).tolist()
        new_row["translation_world_m"] = t.astype(float).tolist()
        new_row["temporal_pose_graph"] = {
            "pose_source": "direct_visible_pose_observation_corrected",
            "rotation_delta_rotvec_rad": rot_delta[i].astype(float).tolist(),
            "translation_delta_world_m": trans_delta[i].astype(float).tolist(),
            "translation_prior_sigma_m": float(obs.translation_sigma_m),
            "rotation_prior_sigma_rad": float(obs.rotation_sigma_rad),
            "nonpenetration_target_world_m": None if obs.nonpenetration_target_world_m is None else obs.nonpenetration_target_world_m.astype(float).tolist(),
            "nonpenetration_weight": float(obs.nonpenetration_weight),
            "nonpenetration_source_rows": int(obs.nonpenetration_source_rows),
        }
        out.append(new_row)

    summary: dict[str, Any] = {
        "enabled": bool(args.complete_full_timeline_rigid_pose),
        "completed_row_count": 0,
        "direct_row_count": int(len(corrected_by_idx)),
        "mode_counts": {},
        "max_interpolation_gap_frames": int(args.max_rigid_pose_interpolation_gap_frames),
        "max_extrapolation_gap_frames": int(args.max_rigid_pose_extrapolation_gap_frames),
    }
    if not args.complete_full_timeline_rigid_pose or not corrected_by_idx:
        return out, summary

    key_frames = sorted(corrected_by_idx)
    key_rots = Rotation.from_matrix([corrected_by_idx[idx][0] for idx in key_frames])
    slerp = Slerp(np.asarray(key_frames, dtype=float), key_rots)
    key_trans = np.asarray([corrected_by_idx[idx][1] for idx in key_frames], dtype=float)

    def add_mode(mode: str) -> None:
        counts = summary.setdefault("mode_counts", {})
        counts[mode] = int(counts.get(mode, 0)) + 1

    for row in out:
        idx = int(row.get("frame_idx", -1))
        if idx in corrected_by_idx or idx < 0:
            continue
        before = [f for f in key_frames if f < idx]
        after = [f for f in key_frames if f > idx]
        mode: str | None = None
        r_fill: np.ndarray | None = None
        t_fill: np.ndarray | None = None
        gap_frames: int | None = None
        bracket: list[int] | None = None
        if before and after:
            lo = before[-1]
            hi = after[0]
            gap_frames = int(hi - lo)
            if gap_frames <= int(args.max_rigid_pose_interpolation_gap_frames):
                alpha = float(idx - lo) / float(max(1, hi - lo))
                r_fill = slerp([float(idx)]).as_matrix()[0]
                t_lo = key_trans[key_frames.index(lo)]
                t_hi = key_trans[key_frames.index(hi)]
                t_fill = (1.0 - alpha) * t_lo + alpha * t_hi
                mode = "interpolated_between_visible_pose_observations"
                bracket = [int(lo), int(hi)]
        if mode is None:
            nearest = min(key_frames, key=lambda f: abs(f - idx))
            dist = abs(nearest - idx)
            if dist <= int(args.max_rigid_pose_extrapolation_gap_frames):
                r_fill = corrected_by_idx[nearest][0]
                t_fill = corrected_by_idx[nearest][1]
                gap_frames = int(dist)
                mode = "nearest_visible_pose_hold"
                bracket = [int(nearest)]
        if mode is None or r_fill is None or t_fill is None:
            continue
        old_status = row.get("status")
        row["status"] = COMPLETED_POSE_STATUS
        row["pose_measurement_status"] = old_status
        row["rotation_world_from_completed_canonical_matrix"] = r_fill.astype(float).tolist()
        row["translation_world_m"] = t_fill.astype(float).tolist()
        row["temporal_pose_graph"] = {
            "pose_source": mode,
            "bracket_visible_pose_frames": bracket,
            "gap_frames": gap_frames,
            "direct_visible_measurement": False,
            "uncertainty": "rigid-body temporal completion; not a direct object mask/depth observation",
        }
        summary["completed_row_count"] = int(summary["completed_row_count"]) + 1
        add_mode(mode)
    return out, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    completion = load_json(args.completion_report) if args.completion_report else {}
    mesh_path = args.completed_mesh or Path(completion.get("outputs", {}).get("completed_mesh_labeled", ""))
    if not mesh_path:
        raise RuntimeError("completed mesh path missing; pass --completed-mesh or --completion-report")
    mesh = load_mesh(Path(mesh_path))
    observations, skipped, targets = build_observations(args, annotations, pose_report, mesh)
    x0 = np.zeros(len(observations) * 6, dtype=float)
    before = residual_vector(x0, observations, args)
    jac = residual_sparsity(observations)
    if jac.shape != (len(before), len(x0)):
        raise RuntimeError(f"sparsity shape {jac.shape} != residual/vector {(len(before), len(x0))}")
    result = least_squares(
        lambda x: residual_vector(x, observations, args),
        x0,
        jac_sparsity=jac,
        max_nfev=int(args.max_nfev),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        verbose=2 if args.verbose else 0,
    )
    after = residual_vector(result.x, observations, args)
    mesh_samples = deterministic_sample_mesh(mesh, int(args.surface_metric_sample_count), int(args.seed) + 73)
    before_surface = surface_metrics(observations, mesh_samples, x0)
    after_surface = surface_metrics(observations, mesh_samples, result.x)
    before_target = target_residual_summary(x0, observations)
    after_target = target_residual_summary(result.x, observations)
    pose_rows, full_timeline_completion = build_pose_rows(pose_report.get("pose_rows", []), observations, result.x, args)
    surface_before_med = before_surface["observed_to_mesh_median_m"]["median"]
    surface_after_med = after_surface["observed_to_mesh_median_m"]["median"]
    target_before_med = before_target["target_residual_norm_m"]["median"]
    target_after_med = after_target["target_residual_norm_m"]["median"]
    surface_degraded_m = None
    if surface_before_med is not None and surface_after_med is not None:
        surface_degraded_m = float(surface_after_med) - float(surface_before_med)
    target_improved = target_before_med is not None and target_after_med is not None and float(target_after_med) < float(target_before_med)
    surface_preserved = surface_degraded_m is None or surface_degraded_m <= float(args.max_surface_median_degradation_m)
    if result.success and surface_preserved and target_improved:
        status = "corrected_pose_graph_surface_preserved_nonpenetration_pressure_improved"
    elif result.success and surface_preserved:
        status = "corrected_pose_graph_surface_preserved_no_nonpenetration_gain"
    elif result.success:
        status = "corrected_pose_graph_surface_degraded_untrusted"
    else:
        status = "corrected_pose_graph_optimizer_incomplete"
    report = {
        "method": "solve_v19_rigid_object_pose_graph",
        "status": status,
        "annotation_ready": bool(result.success and surface_preserved),
        "claim_scope": "Temporal rigid-object pose correction over visible-frame SE(3) measurements. Visible-surface ICP rows are pose observations; nonpenetration rows exert only clipped soft pressure. Remaining penetration after remeasurement is systematic hand/object/camera conflict, not solved by this graph.",
        "object_id": args.object_id,
        "inputs": {
            "annotations": str(args.annotations),
            "pose_report": str(args.pose_report),
            "completion_report": str(args.completion_report) if args.completion_report else None,
            "completed_mesh": str(mesh_path),
            "constraint_report": str(args.constraint_report) if args.constraint_report else None,
        },
        "graph_frames": [obs.frame_idx for obs in observations],
        "graph_frame_count": int(len(observations)),
        "skipped_pose_observations": skipped,
        "nonpenetration_target_frame_count": int(sum(1 for obs in observations if obs.nonpenetration_target_world_m is not None)),
        "nonpenetration_target_source_frame_count": int(len(targets)),
        "parameters": {
            "max_nonpenetration_target_m": float(args.max_nonpenetration_target_m),
            "sigma_nonpenetration_target_m": float(args.sigma_nonpenetration_target_m),
            "sigma_translation_delta_step_m": float(args.sigma_translation_delta_step_m),
            "sigma_rotation_delta_step_rad": float(args.sigma_rotation_delta_step_rad),
            "sigma_translation_delta_accel_m": float(args.sigma_translation_delta_accel_m),
            "sigma_rotation_delta_accel_rad": float(args.sigma_rotation_delta_accel_rad),
            "max_surface_median_degradation_m": float(args.max_surface_median_degradation_m),
            "nonpenetration_states": list(args.nonpenetration_states),
            "complete_full_timeline_rigid_pose": bool(args.complete_full_timeline_rigid_pose),
            "max_rigid_pose_interpolation_gap_frames": int(args.max_rigid_pose_interpolation_gap_frames),
            "max_rigid_pose_extrapolation_gap_frames": int(args.max_rigid_pose_extrapolation_gap_frames),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "residual_rms_before": float(np.sqrt(np.mean(before * before))),
            "residual_rms_after": float(np.sqrt(np.mean(after * after))),
        },
        "correction_summary": correction_summary(result.x, observations),
        "nonpenetration_target_before": before_target,
        "nonpenetration_target_after": after_target,
        "surface_before": before_surface,
        "surface_after": after_surface,
        "surface_observed_to_mesh_median_degradation_m": surface_degraded_m,
        "full_timeline_rigid_pose_completion": full_timeline_completion,
        "pose_rows": pose_rows,
        "outputs": {
            "pose_graph_report": str(args.output_dir / "v19_rigid_object_pose_graph_report.json"),
            "v18_compatible_pose_report": str(args.output_dir / "v18_compact_rigid_object_pose_fit_report.json"),
        },
        "elapsed_s": float(time.time() - started),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "v19_rigid_object_pose_graph_report.json", report)
    # Same payload under the legacy name lets existing V18 render/constraint tools consume corrected rows.
    write_json(args.output_dir / "v18_compact_rigid_object_pose_fit_report.json", report)
    print(json.dumps({k: report[k] for k in ["status", "annotation_ready", "graph_frame_count", "nonpenetration_target_frame_count", "optimizer", "correction_summary", "full_timeline_rigid_pose_completion", "nonpenetration_target_before", "nonpenetration_target_after", "surface_observed_to_mesh_median_degradation_m"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completion-report", type=Path, default=None)
    p.add_argument("--completed-mesh", type=Path, default=None)
    p.add_argument("--constraint-report", type=Path, default=None)
    p.add_argument("--object-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame-start", type=int, default=None)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--min-graph-frames", type=int, default=8)
    p.add_argument("--min-visible-points", type=int, default=20)
    p.add_argument("--min-pose-sigma-m", type=float, default=0.004)
    p.add_argument("--max-pose-sigma-m", type=float, default=0.045)
    p.add_argument("--default-pose-sigma-m", type=float, default=0.018)
    p.add_argument("--min-pose-rotation-sigma-rad", type=float, default=0.035)
    p.add_argument("--max-pose-rotation-sigma-rad", type=float, default=0.35)
    p.add_argument("--max-nonpenetration-target-m", type=float, default=0.015)
    p.add_argument("--sigma-nonpenetration-target-m", type=float, default=0.030)
    p.add_argument("--nonpenetration-states", nargs="+", default=["candidate_coordinate_correction_visible_2d_compatible"])
    p.add_argument("--sigma-translation-delta-step-m", type=float, default=0.010)
    p.add_argument("--sigma-rotation-delta-step-rad", type=float, default=0.080)
    p.add_argument("--sigma-translation-delta-accel-m", type=float, default=0.006)
    p.add_argument("--sigma-rotation-delta-accel-rad", type=float, default=0.050)
    p.add_argument("--max-surface-median-degradation-m", type=float, default=0.003)
    p.add_argument("--complete-full-timeline-rigid-pose", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-rigid-pose-interpolation-gap-frames", type=int, default=240)
    p.add_argument("--max-rigid-pose-extrapolation-gap-frames", type=int, default=240)
    p.add_argument("--surface-metric-sample-count", type=int, default=2500)
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument("--seed", type=int, default=1907)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
