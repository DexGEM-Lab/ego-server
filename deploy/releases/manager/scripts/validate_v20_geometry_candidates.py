#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - dependency-light fallback for harness smoke environments
    cKDTree = None

from v20_common import ContractError, load_json, load_mask, numeric_summary, project_points, safe_id, write_json


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ContractError(f"invalid_mesh_candidate: {path}")
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=float), faces=np.asarray(mesh.faces, dtype=np.int64), process=False)


def sample_mesh(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points, _ = trimesh.sample.sample_surface(mesh, min(int(count), max(1, len(mesh.faces) * 2)), seed=rng)
    return np.asarray(points, dtype=float)


def transform_candidate_points(points: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    for key in ("matrix_model_to_canonical", "T_canonical_model", "T_world_model", "T_object_model"):
        raw = candidate.get(key)
        arr = np.asarray(raw if raw is not None else [], dtype=float)
        if arr.shape == (4, 4):
            hom = np.concatenate([points, np.ones((len(points), 1), dtype=float)], axis=1)
            return (hom @ arr.T)[:, :3]
    return points


def collect_visible_surfaces(annotations: dict[str, Any], object_id: str, max_frames: int, max_points: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks = []
    rows = []
    for frame in annotations.get("frames", []) if isinstance(annotations, dict) else []:
        if len(rows) >= max_frames:
            break
        frame_idx = int(frame.get("frame_idx", frame.get("index", -1)))
        for obj in frame.get("objects", []) if isinstance(frame.get("objects"), list) else []:
            if str(obj.get("object_id")) != str(object_id):
                continue
            geom = obj.get("visible_geometry_candidate") if isinstance(obj.get("visible_geometry_candidate"), dict) else {}
            pts = np.asarray(geom.get("world_vertices_sample_m") or geom.get("camera_vertices_sample_m") or [], dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 3 and np.isfinite(pts).all():
                if len(pts) > max_points:
                    pts = pts[np.linspace(0, len(pts) - 1, max_points, dtype=int)]
                chunks.append(pts)
                rows.append({"frame_idx": frame_idx, "frame": frame, "object": obj, "points": pts})
            break
    if not chunks:
        return np.zeros((0, 3), dtype=float), rows
    return np.vstack(chunks), rows


def nearest_distances(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    if cKDTree is not None:
        d, _ = cKDTree(target).query(query, k=1, workers=-1)
        return np.asarray(d, dtype=float)
    chunk = 1024
    distances = []
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        diff = q[:, None, :] - target[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))
    return np.concatenate(distances, axis=0) if distances else np.zeros((0,), dtype=float)


def nearest_stats(query: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(query) == 0 or len(target) == 0:
        return {"count": int(len(query)), "median_m": None, "p90_m": None, "p95_m": None, "mean_m": None, "max_m": None}
    d = nearest_distances(query, target)
    return {
        "count": int(len(query)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90.0)),
        "p95_m": float(np.percentile(d, 95.0)),
        "mean_m": float(np.mean(d)),
        "max_m": float(np.max(d)),
    }


def silhouette_residual(candidate_points: np.ndarray, visible_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    distances = []
    evaluated = 0
    for row in visible_rows[: int(args.max_silhouette_frames)]:
        obj = row["object"]
        frame = row["frame"]
        mask_path = obj.get("mask_path") or (obj.get("visible_geometry_candidate") or {}).get("mask_path")
        intr = (obj.get("visible_geometry_candidate") or {}).get("intrinsics_fx_fy_cx_cy")
        camera = frame.get("camera") if isinstance(frame.get("camera"), dict) else {}
        T_world_camera = np.asarray(camera.get("T_world_camera_metric") or camera.get("T_world_camera") or [], dtype=float)
        if not mask_path or intr is None or T_world_camera.shape != (4, 4):
            continue
        mask = load_mask(Path(mask_path), None)
        R = T_world_camera[:3, :3]
        t = T_world_camera[:3, 3]
        points_camera = (candidate_points - t[None, :]) @ R
        depth_valid = points_camera[:, 2] > 0
        if np.count_nonzero(depth_valid) < 3:
            continue
        uv = project_points(points_camera[depth_valid], intr, "opencv_positive_z")
        inside_image = (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        if np.count_nonzero(inside_image) < 3:
            continue
        rounded = np.rint(uv[inside_image]).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, mask.shape[1] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, mask.shape[0] - 1)
        inside_mask = mask[rounded[:, 1], rounded[:, 0]]
        distances.append(1.0 - float(np.mean(inside_mask)))
        evaluated += 1
    return {"evaluated_frames": evaluated, "projected_candidate_outside_mask_fraction": numeric_summary(distances)}


def free_space_and_nonpenetration(candidate_points: np.ndarray, validation_report: dict[str, Any] | None, nonpenetration_report: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"has_hidden_volume_validation": validation_report is not None, "has_nonpenetration_report": nonpenetration_report is not None}
    if isinstance(validation_report, dict):
        out["hidden_volume_status"] = validation_report.get("status")
        for key in ("free_space_violation_count", "free_space_violation_fraction", "hidden_volume_violation_count"):
            if key in validation_report:
                out[key] = validation_report[key]
    if isinstance(nonpenetration_report, dict):
        rows = nonpenetration_report.get("rows") or nonpenetration_report.get("constraint_rows") or []
        magnitudes = []
        for row in rows if isinstance(rows, list) else []:
            for key in ("penetration_depth_m", "signed_distance_m", "nonpenetration_residual_m"):
                if key in row:
                    try:
                        magnitudes.append(abs(float(row[key])))
                    except Exception:
                        pass
        out["nonpenetration_abs_m"] = numeric_summary(magnitudes)
    return out


def contact_compatibility(candidate_points: np.ndarray, contact_report: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(contact_report, dict):
        return {"evaluated_rows": 0, "reason": "no_contact_report"}
    rows = contact_report.get("rows") or contact_report.get("contact_rows") or contact_report.get("constraint_rows")
    if not isinstance(rows, list):
        return {"evaluated_rows": 0, "reason": "contact_report_has_no_rows"}
    distances = []
    for row in rows[: int(args.max_contact_rows)]:
        if not isinstance(row, dict):
            continue
        point = row.get("render_point_world_m") or row.get("object_point_world_m") or row.get("contact_point_world_m")
        if point is None or len(candidate_points) == 0:
            continue
        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.shape[0] != 3 or not np.isfinite(arr).all():
            continue
        dist = nearest_distances(arr.reshape(1, 3), candidate_points)
        distances.append(float(dist[0]))
    return {"evaluated_rows": len(distances), "contact_point_to_candidate_surface_m": numeric_summary(distances)}


def promotion_status(residuals: dict[str, Any], args: argparse.Namespace) -> str:
    align_p95 = residuals["visible_surface_alignment"].get("observed_to_candidate_m", {}).get("p95_m")
    reverse_p95 = residuals["visible_surface_alignment"].get("candidate_to_observed_m", {}).get("p95_m")
    outside = residuals["silhouette_projection"].get("projected_candidate_outside_mask_fraction", {}).get("median")
    nonpen = residuals["free_space_nonpenetration"].get("nonpenetration_abs_m", {}).get("p95")
    if align_p95 is None:
        return "unresolved_no_visible_surface_validation"
    if float(align_p95) > float(args.max_visible_alignment_p95_m):
        return "rejected_visible_depth_conflict"
    if reverse_p95 is not None and float(reverse_p95) > float(args.max_candidate_to_observed_p95_m):
        return "downweighted_shape_prior_partial_view_mismatch"
    if outside is not None and float(outside) > float(args.max_outside_mask_fraction):
        return "rejected_silhouette_conflict"
    if nonpen is not None and float(nonpen) > float(args.max_nonpenetration_p95_m):
        return "rejected_nonpenetration_conflict"
    return "promoted_geometry_observation"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_json(args.registry)
    candidates = registry.get("candidates") if isinstance(registry, dict) else None
    if not isinstance(candidates, list):
        raise ContractError(f"geometry_registry_has_no_candidates: {args.registry}")
    annotations = load_json(args.annotations)
    hidden_report = load_json(args.hidden_volume_validation) if args.hidden_volume_validation else None
    nonpenetration_report = load_json(args.nonpenetration_report) if args.nonpenetration_report else None
    contact_report = load_json(args.contact_report) if args.contact_report else None
    rows = []
    promoted = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("evaluation_reference_allowed_in_prediction") is True or "gt" in str(candidate.get("candidate_id", "")).lower() or "ground_truth" in str(candidate.get("candidate_id", "")).lower() or "oracle" in str(candidate.get("candidate_id", "")).lower():
            raise ContractError(f"reference_geometry_candidate_forbidden_in_validator: {candidate.get('candidate_id')}")
        mesh = load_mesh(Path(candidate["mesh_path"]))
        sampled = sample_mesh(mesh, int(args.sample_count), int(args.seed))
        candidate_points = transform_candidate_points(sampled, candidate)
        visible_points, visible_rows = collect_visible_surfaces(annotations, str(candidate["object_id"]), int(args.max_visible_frames), int(args.max_visible_points_per_frame))
        alignment = {
            "visible_surface_point_count": int(len(visible_points)),
            "candidate_sample_count": int(len(candidate_points)),
            "observed_to_candidate_m": nearest_stats(visible_points, candidate_points),
            "candidate_to_observed_m": nearest_stats(candidate_points, visible_points),
        }
        residuals = {
            "visible_surface_alignment": alignment,
            "silhouette_projection": silhouette_residual(candidate_points, visible_rows, args),
            "free_space_nonpenetration": free_space_and_nonpenetration(candidate_points, hidden_report, nonpenetration_report),
            "contact_compatibility": contact_compatibility(candidate_points, contact_report, args),
        }
        status = promotion_status(residuals, args)
        row = dict(candidate)
        row.update({"validation_residuals": residuals, "promotion_status": status, "accepted_geometry": status == "promoted_geometry_observation"})
        rows.append(row)
        if row["accepted_geometry"]:
            promoted.append(str(row["candidate_id"]))
        candidate_dir = args.output_dir / safe_id(str(candidate["object_id"])) / safe_id(str(candidate["candidate_id"]))
        write_json(candidate_dir / "geometry_candidate_validated.json", row)
    report = {
        "schema": "v20_geometry_validation_report.v0",
        "claim_scope": "Candidates are promoted only when prediction-side generated geometry is compatible with visible surfels and available physical residuals.",
        "candidate_count": len(rows),
        "promoted_count": len(promoted),
        "promoted_candidate_ids": promoted,
        "candidates": rows,
    }
    bundle_update = {
        "schema": "v20_geometry_observation_bundle.v0",
        "geometry_candidates": {
            "registry": str(args.registry),
            "validation_report": str(args.output_report),
            "selected_candidate_ids": promoted,
            "selection_scope": "prediction_side_geometry_validation",
        },
    }
    write_json(args.output_report, report)
    write_json(args.output_bundle, bundle_update)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and promote V20 geometry candidates against prediction-side physical evidence.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--hidden-volume-validation", type=Path, default=None)
    parser.add_argument("--nonpenetration-report", type=Path, default=None)
    parser.add_argument("--contact-report", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=202620)
    parser.add_argument("--max-visible-frames", type=int, default=80)
    parser.add_argument("--max-visible-points-per-frame", type=int, default=512)
    parser.add_argument("--max-silhouette-frames", type=int, default=20)
    parser.add_argument("--max-contact-rows", type=int, default=1000)
    parser.add_argument("--max-visible-alignment-p95-m", type=float, default=0.045)
    parser.add_argument("--max-candidate-to-observed-p95-m", type=float, default=0.20)
    parser.add_argument("--max-outside-mask-fraction", type=float, default=0.35)
    parser.add_argument("--max-nonpenetration-p95-m", type=float, default=0.025)
    return parser.parse_args()


if __name__ == "__main__":
    result = validate(parse_args())
    print(result["promoted_candidate_ids"])
