#!/usr/bin/env python3
"""Build a static/global rigid-object pose candidate from a trusted anchor pose.

This is a prediction-side correction for objects whose rendered trajectory is
physically stationary but whose per-frame visible-depth ICP pose jitters because
partial planar surface observations underconstrain SE(3).  It does not use GT,
contact, or hand state.  It copies one support pose across the timeline and
recomputes visible-depth residual summaries so the candidate can be accepted or
rejected by physical evidence rather than by the existence of rows.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ACCEPTED_STATIC_STATUS = "corrected_temporal_rigid_pose_graph"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_R_t(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    R = np.asarray(row.get("rotation_world_from_completed_canonical_matrix"), dtype=float)
    t = np.asarray(row.get("translation_world_m"), dtype=float)
    if R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all():
        return R, t
    return None


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


def pose_support_sigma_m(row: dict[str, Any], *, floor_m: float, cap_m: float, default_m: float) -> float:
    """Prediction-side uncertainty for one visible-depth pose measurement.

    The measured quantity is the residual between observed object-owned depth
    surfels and the fitted completed mesh.  A lower residual and more support
    points make that pose translation more credible; the floor/cap prevent a
    single sharp but partial planar fit from becoming certain.
    """

    obs = row.get("observed_to_mesh_final") if isinstance(row.get("observed_to_mesh_final"), dict) else {}
    vals = []
    for key in ("median_m", "p90_m", "p95_m"):
        value = obs.get(key)
        if value is not None and math.isfinite(float(value)):
            vals.append(float(value))
    residual = max(vals) if vals else float(default_m)
    visible_n = int(row.get("visible_sample_count") or obs.get("count") or 0)
    sample_factor = math.sqrt(max(1.0, min(float(visible_n), 400.0)) / 100.0)
    sigma = residual / max(sample_factor, 1.0)
    return float(np.clip(max(float(floor_m), sigma), float(floor_m), float(cap_m)))


def weighted_geometric_median(points: np.ndarray, weights: np.ndarray, *, eps: float = 1.0e-9, max_iter: int = 256) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("weighted_geometric_median expects Nx3 points")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        raise RuntimeError("all translation weights are zero")
    x = np.average(points, axis=0, weights=weights)
    for _ in range(int(max_iter)):
        d = np.linalg.norm(points - x[None, :], axis=1)
        hit = np.where(d < eps)[0]
        if hit.size:
            return points[int(hit[0])].astype(float)
        inv = weights / np.maximum(d, eps)
        new_x = np.sum(points * inv[:, None], axis=0) / float(np.sum(inv))
        if float(np.linalg.norm(new_x - x)) < eps:
            return new_x.astype(float)
        x = new_x
    return x.astype(float)


def pose_measurement_weights(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[int], list[int]]:
    valid_rows: list[dict[str, Any]] = []
    sigmas: list[float] = []
    weights: list[float] = []
    frame_ids: list[int] = []
    visible_counts: list[int] = []
    for row in rows:
        if as_R_t(row) is None:
            continue
        sigma = pose_support_sigma_m(
            row,
            floor_m=float(args.translation_weight_floor_m),
            cap_m=float(args.translation_weight_cap_m),
            default_m=float(args.translation_weight_default_m),
        )
        valid_rows.append(row)
        sigmas.append(sigma)
        weights.append(1.0 / max(sigma, 1.0e-6) ** 2)
        frame_ids.append(int(row.get("frame_idx", len(frame_ids))))
        visible_counts.append(int(row.get("visible_sample_count") or 0))
    return valid_rows, np.asarray(sigmas, dtype=float), np.asarray(weights, dtype=float), frame_ids, visible_counts


def weighted_quaternion_average(rotations: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not rotations:
        raise RuntimeError("no rotations for weighted average")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(rotations),) or float(np.sum(weights)) <= 0.0:
        raise RuntimeError("rotation weights must be positive and match rotations")
    quats_xyzw = Rotation.from_matrix(np.stack(rotations, axis=0)).as_quat()
    ref = quats_xyzw[0].copy()
    A = np.zeros((4, 4), dtype=float)
    for q, w in zip(quats_xyzw, weights):
        q = np.asarray(q, dtype=float)
        q /= np.linalg.norm(q)
        if float(np.dot(q, ref)) < 0.0:
            q = -q
        A += float(w) * np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q_mean = vecs[:, int(np.argmax(vals))]
    q_mean /= np.linalg.norm(q_mean)
    if float(np.dot(q_mean, ref)) < 0.0:
        q_mean = -q_mean
    return Rotation.from_quat(q_mean).as_matrix().astype(float)


def rotation_estimate(
    rows: list[dict[str, Any]],
    anchor_R: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(args.rotation_source)
    if source == "anchor":
        return anchor_R.astype(float), {
            "rotation_source": "anchor",
            "row_count": 1,
            "claim_scope": "single reviewed anchor rotation",
        }
    valid_rows, sigma_arr, weights, frame_ids, visible_counts = pose_measurement_weights(rows, args)
    rotations: list[np.ndarray] = []
    for row in valid_rows:
        pose = as_R_t(row)
        if pose is None:
            continue
        R, _ = pose
        rotations.append(R)
    if not rotations:
        raise RuntimeError("no valid pose rotations for stationary rotation estimate")
    if source != "support_weighted_quat_mean":  # pragma: no cover
        raise RuntimeError(f"unknown rotation source {source}")
    R_est = weighted_quaternion_average(rotations, weights)
    deltas = np.asarray([rotation_delta_rad(R_est, R) for R in rotations], dtype=float)
    return R_est, {
        "rotation_source": source,
        "row_count": int(len(rotations)),
        "frame_idx_min": int(min(frame_ids)),
        "frame_idx_max": int(max(frame_ids)),
        "weight_model": "inverse squared visible-depth pose sigma from observed-to-mesh residual and visible support count",
        "rotation_sigma_m_proxy": numeric_summary(sigma_arr),
        "rotation_weight": numeric_summary(weights),
        "visible_sample_count": numeric_summary(np.asarray(visible_counts, dtype=float)),
        "source_rotation_angle_to_estimate_rad": numeric_summary(deltas),
        "source_rotation_angle_to_estimate_deg": numeric_summary([math.degrees(x) for x in deltas]),
        "estimate_rotation_world_from_completed_canonical_matrix": R_est.astype(float).tolist(),
        "claim_scope": "stationary rigid-object rotation posterior; uses prediction-side visible-depth pose measurements only; no hand/contact/GT",
    }


def translation_estimate(
    rows: list[dict[str, Any]],
    anchor_t: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(args.translation_source)
    valid_rows = [row for row in rows if as_R_t(row) is not None]
    if source == "anchor":
        return anchor_t.astype(float), {
            "translation_source": "anchor",
            "row_count": 1,
            "claim_scope": "single reviewed anchor translation",
        }
    weighted_rows, sigma_arr, weights, frame_ids, visible_counts = pose_measurement_weights(rows, args)
    translations = []
    for row in weighted_rows:
        pose = as_R_t(row)
        if pose is None:
            continue
        _, t = pose
        translations.append(t)
    if not translations:
        raise RuntimeError("no valid pose translations for stationary translation estimate")
    pts = np.vstack(translations).astype(float)
    if source == "support_weighted_mean":
        t_est = np.average(pts, axis=0, weights=weights).astype(float)
    elif source == "support_weighted_geomedian":
        t_est = weighted_geometric_median(pts, weights).astype(float)
    else:  # pragma: no cover
        raise RuntimeError(f"unknown translation source {source}")
    deltas = np.linalg.norm(pts - t_est[None, :], axis=1)
    return t_est, {
        "translation_source": source,
        "row_count": int(len(pts)),
        "frame_idx_min": int(min(frame_ids)),
        "frame_idx_max": int(max(frame_ids)),
        "weight_model": "inverse squared visible-depth pose sigma from observed-to-mesh residual and visible support count",
        "translation_sigma_m": numeric_summary(sigma_arr),
        "translation_weight": numeric_summary(weights),
        "visible_sample_count": numeric_summary(np.asarray(visible_counts, dtype=float)),
        "source_translation_distance_to_estimate_m": numeric_summary(deltas),
        "estimate_m": t_est.astype(float).tolist(),
        "claim_scope": "stationary rigid-object translation posterior; uses prediction-side visible-depth pose measurements only; no hand/contact/GT",
    }


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


def load_mesh(path: Path) -> trimesh.Trimesh:
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene):
        meshes = [m for m in geom.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"no mesh in {path}")
        geom = trimesh.util.concatenate(meshes)
    if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
        raise RuntimeError(f"invalid mesh {path}")
    return trimesh.Trimesh(vertices=np.asarray(geom.vertices, dtype=float), faces=np.asarray(geom.faces, dtype=np.int64), process=False)


def apply_pose(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ R.T + np.asarray(t, dtype=float)[None, :]


def annotation_frames_by_idx(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for pos, fr in enumerate(annotations.get("frames", [])):
        if isinstance(fr, dict):
            out[int(fr.get("frame_idx", pos))] = fr
    return out


def annotation_object(frame: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    objs = frame.get("objects") if isinstance(frame.get("objects"), list) else []
    for obj in objs:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj
    return None


def observed_points(frame: dict[str, Any], object_id: str) -> np.ndarray:
    obj = annotation_object(frame, object_id)
    if not isinstance(obj, dict):
        return np.empty((0, 3), dtype=float)
    geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
    pts = np.asarray(geom.get("world_vertices_sample_m") or [], dtype=float)
    if pts.ndim == 2 and pts.shape[1] == 3 and np.isfinite(pts).all():
        return pts
    return np.empty((0, 3), dtype=float)


def rotation_delta_rad(R_static: np.ndarray, R_original: np.ndarray) -> float:
    R_delta = np.asarray(R_static, dtype=float) @ np.asarray(R_original, dtype=float).T
    return float(np.linalg.norm(Rotation.from_matrix(R_delta).as_rotvec()))


def select_anchor_row(rows: list[dict[str, Any]], annotations: dict[str, Any], object_id: str, requested_anchor: int | None) -> dict[str, Any]:
    by_idx = {int(r["frame_idx"]): r for r in rows if isinstance(r.get("frame_idx"), int) and as_R_t(r) is not None}
    if requested_anchor is None:
        adapter = annotations.get("v19_visible_geometry_adapter") if isinstance(annotations.get("v19_visible_geometry_adapter"), dict) else {}
        value = adapter.get("anchor_frame_idx")
        requested_anchor = int(value) if isinstance(value, int) else None
    if requested_anchor is not None and requested_anchor in by_idx:
        return by_idx[requested_anchor]
    # Fallback is support-driven, not category-driven: maximize visible point count and minimize bidirectional residual if present.
    def score(row: dict[str, Any]) -> tuple[float, int]:
        n = int(row.get("visible_sample_count") or 0)
        obs = row.get("observed_to_mesh_final") if isinstance(row.get("observed_to_mesh_final"), dict) else {}
        mesh = row.get("mesh_to_observed_final") if isinstance(row.get("mesh_to_observed_final"), dict) else {}
        obs_p95 = float(obs.get("p95_m") if obs.get("p95_m") is not None else 1.0)
        mesh_p90 = float(mesh.get("p90_m") if mesh.get("p90_m") is not None else 1.0)
        return (obs_p95 + 0.25 * mesh_p90 - 1.0e-5 * n, -n)
    candidates = [r for r in rows if as_R_t(r) is not None]
    if not candidates:
        raise RuntimeError("pose report has no usable pose rows")
    return sorted(candidates, key=score)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--pose-report", type=Path, required=True)
    p.add_argument("--completed-mesh", type=Path, required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--output-report", type=Path, required=True)
    p.add_argument("--anchor-frame", type=int, default=None)
    p.add_argument("--mesh-sample-count", type=int, default=6000)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--hold-components",
        choices=("both", "translation", "rotation"),
        default="both",
        help="Which pose components to hold fixed. Translation-only stabilization preserves per-frame rotations.",
    )
    p.add_argument(
        "--translation-source",
        choices=("anchor", "support_weighted_geomedian", "support_weighted_mean"),
        default="anchor",
        help="Source for stationary translation when --hold-components includes translation. Anchor preserves legacy behavior; support-weighted modes estimate translation from visible-depth pose observations.",
    )
    p.add_argument(
        "--rotation-source",
        choices=("anchor", "support_weighted_quat_mean"),
        default="anchor",
        help="Source for stationary rotation when --hold-components includes rotation. support_weighted_quat_mean estimates one rotation from visible-depth pose observations using the same support weights as translation.",
    )
    p.add_argument("--translation-weight-floor-m", type=float, default=0.006)
    p.add_argument("--translation-weight-cap-m", type=float, default=0.080)
    p.add_argument("--translation-weight-default-m", type=float, default=0.025)
    return p.parse_args()


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    annotations = load_json(args.annotations)
    pose_report = load_json(args.pose_report)
    if not isinstance(annotations, dict) or not isinstance(pose_report, dict):
        raise RuntimeError("annotations and pose-report must be JSON objects")
    rows_in = [r for r in pose_report.get("pose_rows", []) if isinstance(r, dict)]
    anchor_row = select_anchor_row(rows_in, annotations, args.object_id, args.anchor_frame)
    anchor_idx = int(anchor_row["frame_idx"])
    anchor_pose = as_R_t(anchor_row)
    if anchor_pose is None:
        raise RuntimeError(f"anchor row {anchor_idx} lacks pose")
    R_anchor, t_anchor = anchor_pose
    t_static, translation_source_summary = translation_estimate(rows_in, t_anchor, args)
    R_static, rotation_source_summary = rotation_estimate(rows_in, R_anchor, args)
    mesh = load_mesh(args.completed_mesh)
    rng = np.random.default_rng(int(args.sample_seed))
    sample_count = min(int(args.mesh_sample_count), max(1, len(mesh.faces) * 2))
    mesh_sample_obj, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    mesh_sample_obj = np.asarray(mesh_sample_obj, dtype=float)
    frames_by_idx = annotation_frames_by_idx(annotations)

    pose_rows: list[dict[str, Any]] = []
    trans_deltas: list[float] = []
    rot_deltas: list[float] = []
    obs_to_mesh_medians: list[float] = []
    mesh_to_obs_medians: list[float] = []
    visible_counts: list[int] = []
    for raw in rows_in:
        if "frame_idx" not in raw:
            continue
        idx = int(raw["frame_idx"])
        original_pose = as_R_t(raw)
        if original_pose is None:
            continue
        R_orig, t_orig = original_pose
        if args.hold_components == "both":
            R_out, t_out = R_static, t_static
        elif args.hold_components == "translation":
            R_out, t_out = R_orig, t_static
        elif args.hold_components == "rotation":
            R_out, t_out = R_static, t_orig
        else:  # pragma: no cover
            raise RuntimeError(f"unknown hold-components {args.hold_components}")
        trans_delta = float(np.linalg.norm(t_out - t_orig))
        rot_delta = rotation_delta_rad(R_out, R_orig)
        trans_deltas.append(trans_delta)
        rot_deltas.append(rot_delta)
        frame = frames_by_idx.get(idx, {})
        obs = observed_points(frame, args.object_id) if isinstance(frame, dict) else np.empty((0, 3), dtype=float)
        visible_counts.append(int(len(obs)))
        mesh_sample_world = apply_pose(mesh_sample_obj, R_out, t_out)
        if len(obs) > 0:
            obs_to_mesh = nearest_summary(obs, mesh_sample_world)
            mesh_to_obs = nearest_summary(mesh_sample_world, obs)
            med = obs_to_mesh.get("median_m")
            if med is not None and math.isfinite(float(med)):
                obs_to_mesh_medians.append(float(med))
            med2 = mesh_to_obs.get("median_m")
            if med2 is not None and math.isfinite(float(med2)):
                mesh_to_obs_medians.append(float(med2))
        else:
            obs_to_mesh = {"count": 0, "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
            mesh_to_obs = {"count": int(len(mesh_sample_world)), "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
        row = dict(raw)
        row["status"] = ACCEPTED_STATIC_STATUS
        row["pose_measurement_status"] = "static_anchor_pose_candidate"
        row["rotation_world_from_completed_canonical_matrix"] = R_out.astype(float).tolist()
        row["translation_world_m"] = t_out.astype(float).tolist()
        row["static_pose_candidate"] = {
            "method": "stationary_pose_component_hold_static_rigid_object",
            "hold_components": str(args.hold_components),
            "translation_source": str(args.translation_source),
            "translation_source_summary": translation_source_summary,
            "rotation_source": str(args.rotation_source),
            "rotation_source_summary": rotation_source_summary,
            "anchor_frame_idx": anchor_idx,
            "anchor_translation_world_m": t_anchor.astype(float).tolist(),
            "anchor_selection": "requested_or_visible_geometry_adapter_anchor_then_support_score",
            "source_pose_report": str(args.pose_report),
            "source_frame_original_translation_delta_m": trans_delta,
            "source_frame_original_rotation_delta_rad": rot_delta,
            "visible_point_count": int(len(obs)),
            "observed_to_static_mesh": obs_to_mesh,
            "static_mesh_to_observed": mesh_to_obs,
            "claim_scope": "static object pose candidate; does not use hand/contact/GT; must be validated by render and prediction/evaluation residuals",
        }
        row["temporal_pose_graph"] = {
            "pose_source": f"static_stationary_pose_{args.hold_components}_hold",
            "translation_source": str(args.translation_source),
            "translation_source_summary": translation_source_summary,
            "rotation_source": str(args.rotation_source),
            "rotation_source_summary": rotation_source_summary,
            "static_anchor_frame_idx": anchor_idx,
            "direct_visible_measurement": idx == anchor_idx,
            "original_pose_status": raw.get("status"),
            "original_pose_measurement_status": raw.get("pose_measurement_status"),
            "original_to_static_translation_delta_m": trans_delta,
            "original_to_static_rotation_delta_rad": rot_delta,
            "uncertainty": "stationary-rigid-object hypothesis; rejects per-frame planar ICP jitter when visible support is partial",
        }
        pose_rows.append(row)

    frame_ids = sorted(int(fr.get("frame_idx", pos)) for pos, fr in enumerate(annotations.get("frames", [])) if isinstance(fr, dict))
    pose_ids = {int(r["frame_idx"]) for r in pose_rows}
    report = dict(pose_report)
    report.update(
        {
            "method": "build_v19_static_rigid_pose_report",
            "status": "static_rigid_pose_candidate_built",
            "annotation_ready": True,
            "claim_scope": "prediction-side static/global rigid object pose candidate; hand/MANO/contact are not used; GT is not used",
            "inputs": {
                **(pose_report.get("inputs") if isinstance(pose_report.get("inputs"), dict) else {}),
                "annotations": str(args.annotations),
                "source_pose_report": str(args.pose_report),
                "completed_mesh": str(args.completed_mesh),
            },
            "static_pose_candidate": {
                "anchor_frame_idx": anchor_idx,
                "anchor_pose_translation_world_m": t_anchor.astype(float).tolist(),
                "stationary_translation_world_m": t_static.astype(float).tolist(),
                "translation_source": str(args.translation_source),
                "translation_source_summary": translation_source_summary,
                "anchor_pose_rotation_world_from_completed_canonical_matrix": R_anchor.astype(float).tolist(),
                "stationary_rotation_world_from_completed_canonical_matrix": R_static.astype(float).tolist(),
                "rotation_source": str(args.rotation_source),
                "rotation_source_summary": rotation_source_summary,
                "hold_components": str(args.hold_components),
                "mesh_sample_count": int(len(mesh_sample_obj)),
                "frame_count": len(pose_rows),
                "missing_pose_frames": sorted(set(frame_ids).difference(pose_ids)),
                "original_to_static_translation_delta_m": numeric_summary(trans_deltas),
                "original_to_static_rotation_delta_rad": numeric_summary(rot_deltas),
                "original_to_static_rotation_delta_deg": numeric_summary([math.degrees(x) for x in rot_deltas]),
                "visible_point_count": numeric_summary(visible_counts),
                "static_observed_to_mesh_median_m": numeric_summary(obs_to_mesh_medians),
                "static_mesh_to_observed_median_m": numeric_summary(mesh_to_obs_medians),
            },
            "correction_summary": {
                "status": "static_stationary_pose_hold_candidate",
                "anchor_frame_idx": anchor_idx,
                "translation_source": str(args.translation_source),
                "translation_source_summary": translation_source_summary,
                "rotation_source": str(args.rotation_source),
                "rotation_source_summary": rotation_source_summary,
                "translation_delta_norm_m": numeric_summary(trans_deltas),
                "rotation_delta_norm_rad": numeric_summary(rot_deltas),
                "static_observed_to_mesh_median_m": numeric_summary(obs_to_mesh_medians),
                "static_mesh_to_observed_median_m": numeric_summary(mesh_to_obs_medians),
            },
            "full_timeline_rigid_pose_completion": {
                "state": f"static_stationary_pose_{args.hold_components}_{args.translation_source}_{args.rotation_source}_full_timeline_candidate",
                "frame_count": len(frame_ids),
                "pose_frame_count": len(pose_ids),
                "missing_pose_frames": sorted(set(frame_ids).difference(pose_ids)),
            },
            "pose_rows": pose_rows,
            "elapsed_s": time.time() - started,
            "outputs": {"pose_report": str(args.output_report)},
        }
    )
    return report


def main() -> None:
    args = parse_args()
    report = build(args)
    write_json(args.output_report, report)
    print(json.dumps({
        "status": report.get("status"),
        "anchor_frame_idx": report.get("static_pose_candidate", {}).get("anchor_frame_idx"),
        "frame_count": len(report.get("pose_rows", [])),
        "correction_summary": report.get("correction_summary"),
    }, indent=2))


if __name__ == "__main__":
    main()
