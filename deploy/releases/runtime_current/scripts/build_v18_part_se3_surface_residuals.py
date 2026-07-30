#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_se3_surface_residuals"
CLAIM = (
    "This artifact evaluates whether supported V18 articulation center fits also have enough world-frame part-surface "
    "rigid SE(3) residual support. It transforms depth-backed part vertices to the V16 world frame, aligns selected "
    "frames of each part surface to a reference surface by rigid ICP, and records residual blockers. It does not mark "
    "part pose, contact ownership, or object pose ready."
)


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


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def stats(values: Any) -> dict[str, Any]:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * p / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {"count": len(xs), "median": pct(50.0), "p05": pct(5.0), "p95": pct(95.0), "min": xs[0], "max": xs[-1]}


def finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if int(points.shape[0]) <= max_points:
        return points
    return points[np.linspace(0, int(points.shape[0]) - 1, max_points, dtype=np.int64)]


def rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    centered_source = source - source_centroid
    centered_target = target - target_centroid
    h = centered_source.T @ centered_target
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = target_centroid - r @ source_centroid
    return r, t


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = max(-1.0, min(1.0, float((np.trace(rotation) - 1.0) / 2.0)))
    return math.degrees(math.acos(value))


def icp_residual(source_points: np.ndarray, target_points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if int(source_points.shape[0]) < int(args.min_icp_points) or int(target_points.shape[0]) < int(args.min_icp_points):
        raise RuntimeError("too_few_points_for_part_se3_icp")
    source = sample_points(source_points, int(args.max_icp_points))
    target = sample_points(target_points, int(args.max_icp_points))
    transformed = source.copy()
    tree = KDTree(target)
    previous_median: float | None = None
    rotation_total = np.eye(3, dtype=np.float64)
    translation_total = np.zeros(3, dtype=np.float64)
    iterations = 0
    for iterations in range(1, int(args.icp_iterations) + 1):
        distances, indices = tree.query(transformed, k=1)
        matched = target[np.asarray(indices, dtype=np.int64)]
        r, t = rigid_transform(transformed, matched)
        transformed = (r @ transformed.T).T + t
        rotation_total = r @ rotation_total
        translation_total = r @ translation_total + t
        median = float(np.median(distances)) if len(distances) else 0.0
        if previous_median is not None and abs(previous_median - median) < float(args.icp_tolerance_m):
            break
        previous_median = median
    distances, _ = tree.query(transformed, k=1)
    return {
        "source_point_count": int(source_points.shape[0]),
        "target_point_count": int(target_points.shape[0]),
        "sampled_source_point_count": int(source.shape[0]),
        "sampled_target_point_count": int(target.shape[0]),
        "iterations": iterations,
        "residual_m": stats(np.asarray(distances, dtype=np.float64)),
        "icp_rotation_angle_deg": rotation_angle_deg(rotation_total),
        "icp_translation_norm_m": float(np.linalg.norm(translation_total)),
    }


def frame_transforms(annotation_path: Path) -> dict[int, np.ndarray]:
    payload = require_dict(load_json(annotation_path), f"annotations {annotation_path}")
    out: dict[int, np.ndarray] = {}
    for raw in require_list(payload.get("frames"), "annotation frames"):
        frame = require_dict(raw, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        transform = np.asarray(require_dict(frame.get("camera"), "frame camera").get("T_world_camera_metric"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"frame {frame_idx} has invalid T_world_camera_metric")
        out[frame_idx] = transform
    return out


def load_world_part_surfaces(surfaces_report: dict[str, Any], transforms: dict[int, np.ndarray]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    archive_path = Path(str(surfaces_report.get("archive_npz")))
    arrays = np.load(archive_path, allow_pickle=True)
    rows = [require_dict(raw, "surface row") for raw in require_list(surfaces_report.get("surface_rows"), "surface rows")]
    if int(arrays["frame_idx"].shape[0]) != len(rows):
        raise RuntimeError("part surface archive/report row count mismatch")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        frame_idx = require_int(row.get("frame_idx"), "surface frame_idx")
        object_id = str(row.get("object_id"))
        label = str(row.get("part_track_label"))
        if int(arrays["frame_idx"][index]) != frame_idx or str(arrays["object_id"][index]) != object_id or str(arrays["part_track_label"][index]) != label:
            raise RuntimeError(f"part surface archive/report mismatch at row {index}")
        transform = transforms.get(frame_idx)
        if transform is None:
            continue
        start = int(arrays["vertex_offsets"][index])
        end = int(arrays["vertex_offsets"][index + 1])
        vertices_camera = np.asarray(arrays["vertices"][start:end], dtype=np.float64)
        vertices_world = (transform @ np.c_[vertices_camera, np.ones(vertices_camera.shape[0])].T).T[:, :3]
        grouped.setdefault((object_id, label), []).append(
            {
                "archive_row_index": index,
                "frame_idx": frame_idx,
                "vertices_world_m": vertices_world,
                "vertex_count": int(vertices_world.shape[0]),
                "face_count": require_int(row.get("faces"), "faces"),
                "part_containment_in_object": row.get("part_containment_in_object"),
                "depth_median_m": row.get("depth_median_m"),
                "mask_stride_used": row.get("mask_stride_used"),
                "mask_sampling_target_met": row.get("mask_sampling_target_met"),
            }
        )
    return grouped


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace, allowed_frames: set[int] | None = None) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in sorted(rows, key=lambda item: require_int(item.get("frame_idx"), "frame_idx"))
        if int(row.get("vertex_count", 0)) >= int(args.min_icp_points)
        and (allowed_frames is None or require_int(row.get("frame_idx"), "frame_idx") in allowed_frames)
    ]
    if allowed_frames is not None and len(eligible) <= int(args.max_exhaustive_shared_frames):
        return eligible
    if len(eligible) <= int(args.max_probe_frames):
        return eligible
    return [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, int(args.max_probe_frames), dtype=np.int64)]


def contiguous_components(frame_indices: list[int]) -> list[dict[str, int]]:
    if not frame_indices:
        return []
    frames = sorted(set(int(frame) for frame in frame_indices))
    components: list[dict[str, int]] = []
    start = frames[0]
    prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        components.append({"frame_start": start, "frame_end": prev, "frame_count": prev - start + 1})
        start = frame
        prev = frame
    components.append({"frame_start": start, "frame_end": prev, "frame_count": prev - start + 1})
    return components


def classify_part(part_row: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    blockers: list[str] = []
    successful = require_int(part_row.get("successful_icp_frame_count"), "successful_icp_frame_count")
    median_residual = finite_float_or_none(require_dict(part_row.get("per_frame_median_residual_m"), "median residual stats").get("median"))
    p95_of_p95 = finite_float_or_none(require_dict(part_row.get("per_frame_p95_residual_m"), "p95 residual stats").get("p95"))
    if successful < int(args.min_successful_frames):
        blockers.append("too_few_successful_part_se3_icp_frames")
    if median_residual is None or median_residual > float(args.max_median_residual_m):
        blockers.append("part_se3_median_residual_above_threshold")
    if p95_of_p95 is None or p95_of_p95 > float(args.max_p95_of_p95_residual_m):
        blockers.append("part_se3_p95_residual_outlier_above_threshold")
    if blockers:
        return "part_surface_se3_residual_rejected", blockers
    return "part_surface_se3_residual_supported_visible_only_not_pose", blockers


def robust_surface_inlier_refit(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    frame_results = [require_dict(raw, "frame result") for raw in require_list(payload.get("frame_results"), "frame results")]
    if not frame_results:
        return {"robust_surface_inlier_refit_applied": False, "robust_surface_inlier_state": "no_successful_icp_frames"}
    inliers: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for result in frame_results:
        residual = require_dict(result.get("residual_m"), "frame residual")
        median = finite_float_or_none(residual.get("median"))
        p95 = finite_float_or_none(residual.get("p95"))
        keep = (
            median is not None
            and p95 is not None
            and median <= float(args.max_median_residual_m)
            and p95 <= float(args.max_p95_of_p95_residual_m)
        )
        if keep:
            inliers.append(result)
        else:
            outliers.append(result)
    inlier_ratio = len(inliers) / float(len(frame_results)) if frame_results else 0.0
    robust_payload = dict(payload)
    medians = [float(result["residual_m"]["median"]) for result in inliers if result.get("residual_m", {}).get("median") is not None]
    p95s = [float(result["residual_m"]["p95"]) for result in inliers if result.get("residual_m", {}).get("p95") is not None]
    rotations = [float(result["icp_rotation_angle_deg"]) for result in inliers]
    translations = [float(result["icp_translation_norm_m"]) for result in inliers]
    robust_payload.update(
        {
            "successful_icp_frame_count": len(inliers),
            "selected_frame_count": len(inliers),
            "per_frame_median_residual_m": stats(medians),
            "per_frame_p95_residual_m": stats(p95s),
            "icp_rotation_angle_deg": stats(rotations),
            "icp_translation_norm_m": stats(translations),
            "p95_residual_outlier_frame_count": 0,
            "p95_residual_outlier_frames": [],
            "p95_residual_outlier_frame_components": [],
            "worst_p95_residual_frames": [],
        }
    )
    state, blockers = classify_part(robust_payload, args)
    if inlier_ratio < float(args.min_robust_surface_inlier_ratio):
        state = "part_surface_se3_residual_rejected"
        blockers = sorted(set(blockers + ["robust_surface_inlier_ratio_below_threshold"]))
    return {
        "robust_surface_inlier_refit_applied": state == "part_surface_se3_residual_supported_visible_only_not_pose",
        "robust_surface_inlier_state": state,
        "robust_surface_inlier_blockers": blockers,
        "robust_surface_inlier_frame_count": len(inliers),
        "robust_surface_inlier_ratio": inlier_ratio,
        "robust_surface_excluded_frame_count": len(outliers),
        "robust_surface_excluded_frames": [require_int(row.get("frame_idx"), "outlier frame_idx") for row in outliers],
        "robust_surface_excluded_frame_components": contiguous_components([require_int(row.get("frame_idx"), "outlier frame_idx") for row in outliers]),
        "robust_surface_inlier_stats": {
            "per_frame_median_residual_m": robust_payload["per_frame_median_residual_m"],
            "per_frame_p95_residual_m": robust_payload["per_frame_p95_residual_m"],
            "icp_rotation_angle_deg": robust_payload["icp_rotation_angle_deg"],
            "icp_translation_norm_m": robust_payload["icp_translation_norm_m"],
        },
        "robust_payload_fields": {k: robust_payload[k] for k in ["successful_icp_frame_count", "selected_frame_count", "per_frame_median_residual_m", "per_frame_p95_residual_m", "icp_rotation_angle_deg", "icp_translation_norm_m", "p95_residual_outlier_frame_count", "p95_residual_outlier_frames", "p95_residual_outlier_frame_components", "worst_p95_residual_frames"]},
    }


def part_se3_probe(object_id: str, label: str, rows: list[dict[str, Any]], args: argparse.Namespace, allowed_frames: set[int] | None = None) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"{object_id} {label}: no world surface rows")
    candidate_rows = [row for row in rows if allowed_frames is None or require_int(row.get("frame_idx"), "frame_idx") in allowed_frames]
    if not candidate_rows:
        raise RuntimeError(f"{object_id} {label}: no rows after shared-frame restriction")
    reference = max(candidate_rows, key=lambda row: int(row.get("vertex_count", 0)))
    selected = select_rows(rows, args, allowed_frames)
    frame_results: list[dict[str, Any]] = []
    rejected_frames: list[dict[str, Any]] = []
    for row in selected:
        if int(row["archive_row_index"]) == int(reference["archive_row_index"]):
            continue
        try:
            result = icp_residual(row["vertices_world_m"], reference["vertices_world_m"], args)
        except RuntimeError as exc:
            rejected_frames.append({"frame_idx": row.get("frame_idx"), "reason": str(exc)})
            continue
        frame_results.append(
            {
                "frame_idx": row.get("frame_idx"),
                "archive_row_index": row.get("archive_row_index"),
                "vertex_count": row.get("vertex_count"),
                "face_count": row.get("face_count"),
                "part_containment_in_object": row.get("part_containment_in_object"),
                "depth_median_m": row.get("depth_median_m"),
                "mask_stride_used": row.get("mask_stride_used"),
                "mask_sampling_target_met": row.get("mask_sampling_target_met"),
                **result,
            }
        )
    medians = [float(result["residual_m"]["median"]) for result in frame_results if result.get("residual_m", {}).get("median") is not None]
    p95s = [float(result["residual_m"]["p95"]) for result in frame_results if result.get("residual_m", {}).get("p95") is not None]
    rotations = [float(result["icp_rotation_angle_deg"]) for result in frame_results]
    translations = [float(result["icp_translation_norm_m"]) for result in frame_results]
    vertex_counts = [int(row.get("vertex_count", 0)) for row in rows]
    face_counts = [int(row.get("face_count", 0)) for row in rows]
    outlier_frames = [
        {
            "frame_idx": require_int(result.get("frame_idx"), "outlier frame_idx"),
            "archive_row_index": result.get("archive_row_index"),
            "residual_median_m": result.get("residual_m", {}).get("median"),
            "residual_p95_m": result.get("residual_m", {}).get("p95"),
            "vertex_count": result.get("vertex_count"),
            "face_count": result.get("face_count"),
            "part_containment_in_object": result.get("part_containment_in_object"),
            "depth_median_m": result.get("depth_median_m"),
            "mask_stride_used": result.get("mask_stride_used"),
            "mask_sampling_target_met": result.get("mask_sampling_target_met"),
        }
        for result in frame_results
        if (result.get("residual_m", {}).get("p95") is not None and float(result["residual_m"]["p95"]) > float(args.max_p95_of_p95_residual_m))
    ]
    outlier_frames_sorted = sorted(outlier_frames, key=lambda item: float(item.get("residual_p95_m") or 0.0), reverse=True)
    payload = {
        "object_id": object_id,
        "part_track_label": label,
        "probe_type": "world_frame_part_surface_rigid_icp_to_reference",
        "coordinate_frame": "V16 T_world_camera_metric world frame",
        "surface_frame_count": len(rows),
        "shared_frame_restricted": allowed_frames is not None,
        "allowed_shared_frame_count": len(allowed_frames) if allowed_frames is not None else None,
        "candidate_frame_count_after_shared_restriction": len(candidate_rows),
        "selected_frame_count": len(selected),
        "successful_icp_frame_count": len(frame_results),
        "rejected_icp_frame_count": len(rejected_frames),
        "reference_frame_idx": reference.get("frame_idx"),
        "reference_archive_row_index": reference.get("archive_row_index"),
        "reference_vertex_count": reference.get("vertex_count"),
        "vertex_count": stats(vertex_counts),
        "face_count": stats(face_counts),
        "per_frame_median_residual_m": stats(medians),
        "per_frame_p95_residual_m": stats(p95s),
        "icp_rotation_angle_deg": stats(rotations),
        "icp_translation_norm_m": stats(translations),
        "p95_residual_outlier_threshold_m": float(args.max_p95_of_p95_residual_m),
        "p95_residual_outlier_frame_count": len(outlier_frames),
        "p95_residual_outlier_frames": sorted(outlier_frames, key=lambda item: int(item["frame_idx"])),
        "p95_residual_outlier_frame_components": contiguous_components([int(item["frame_idx"]) for item in outlier_frames]),
        "worst_p95_residual_frames": outlier_frames_sorted[: int(args.max_reported_outlier_frames)],
        "frame_results": frame_results,
        "rejected_frames": rejected_frames,
        "acceptance_thresholds": {
            "min_successful_frames": int(args.min_successful_frames),
            "max_median_residual_m": float(args.max_median_residual_m),
            "max_p95_of_p95_residual_m": float(args.max_p95_of_p95_residual_m),
        },
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }
    state, blockers = classify_part(payload, args)
    payload["initial_full_frame_part_surface_se3_state"] = state
    payload["initial_full_frame_part_surface_se3_blockers"] = list(blockers)
    payload["initial_full_frame_residual_stats"] = {
        "successful_icp_frame_count": payload["successful_icp_frame_count"],
        "per_frame_median_residual_m": payload["per_frame_median_residual_m"],
        "per_frame_p95_residual_m": payload["per_frame_p95_residual_m"],
        "icp_rotation_angle_deg": payload["icp_rotation_angle_deg"],
        "icp_translation_norm_m": payload["icp_translation_norm_m"],
        "p95_residual_outlier_frame_count": payload["p95_residual_outlier_frame_count"],
        "p95_residual_outlier_frames": payload["p95_residual_outlier_frames"],
        "worst_p95_residual_frames": payload["worst_p95_residual_frames"],
    }
    robust_report: dict[str, Any] = {"robust_surface_inlier_refit_applied": False, "robust_surface_inlier_state": "not_needed_or_not_applicable"}
    if state == "part_surface_se3_residual_rejected":
        robust_report = robust_surface_inlier_refit(payload, args)
        if robust_report.get("robust_surface_inlier_refit_applied") is True:
            for key, value in require_dict(robust_report.get("robust_payload_fields"), "robust payload fields").items():
                payload[key] = value
            state = "part_surface_se3_residual_supported_visible_only_not_pose"
            blockers = []
    payload["robust_surface_inlier_refit"] = {k: v for k, v in robust_report.items() if k != "robust_payload_fields"}
    payload["part_surface_se3_state"] = state
    payload["part_surface_se3_blockers"] = blockers
    return payload


def pair_state(articulation_row: dict[str, Any], part_rows: list[dict[str, Any]], shared_frame_count: int, args: argparse.Namespace) -> tuple[str, list[str]]:
    if articulation_row.get("articulation_fit_state") != "articulation_fit_residual_supported_visible_center_only_not_pose":
        return "part_se3_not_evaluated_articulation_fit_not_supported", ["articulation_fit_not_supported"]
    if shared_frame_count < int(args.min_successful_frames):
        return "part_se3_surface_residual_underconstrained", ["too_few_shared_frames_for_pair_se3_surface_residual"]
    rejected_parts = [row for row in part_rows if row.get("part_surface_se3_state") != "part_surface_se3_residual_supported_visible_only_not_pose"]
    if rejected_parts:
        blockers = ["one_or_more_part_surface_se3_residuals_rejected"]
        for row in rejected_parts:
            blockers.extend(str(item) for item in row.get("part_surface_se3_blockers", []))
        return "part_se3_surface_residual_rejected", sorted(set(blockers))
    return "part_se3_surface_residual_supported_visible_only_not_pose", ["silhouette_residual_not_evaluated", "hidden_geometry_not_completed", "contact_not_validated"]


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    articulation_path = args.articulation_fit_root / case / "v18_articulation_fit_candidates_report.json"
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    annotation_path = args.v16_root / case / "annotations_v16_full.json"
    articulation = require_dict(load_json(articulation_path), f"{case} articulation fits")
    surfaces = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    transforms = frame_transforms(annotation_path)
    grouped = load_world_part_surfaces(surfaces, transforms)
    rows: list[dict[str, Any]] = []
    for raw in require_list(articulation.get("rows"), "articulation rows"):
        art_row = require_dict(raw, "articulation row")
        object_id = str(art_row.get("object_id"))
        labels = [str(item) for item in require_list(art_row.get("part_track_labels"), "part labels")]
        frame_sets = [
            {require_int(row.get("frame_idx"), "frame_idx") for row in grouped.get((object_id, label), [])}
            for label in labels
        ]
        shared_frames = set.intersection(*frame_sets) if frame_sets else set()
        part_reports = [part_se3_probe(object_id, label, grouped.get((object_id, label), []), args, shared_frames) for label in labels]
        state, blockers = pair_state(art_row, part_reports, len(shared_frames), args)
        rows.append(
            {
                "object_id": object_id,
                "source_articulation_fit_state": art_row.get("articulation_fit_state"),
                "part_track_labels": labels,
                "part_surface_reports": part_reports,
                "pair_shared_frame_count": len(shared_frames),
                "part_se3_pair_state": state,
                "part_se3_pair_blockers": blockers,
                "part_se3_surface_supported_count": sum(1 for row in part_reports if row.get("part_surface_se3_state") == "part_surface_se3_residual_supported_visible_only_not_pose"),
                "part_se3_surface_rejected_count": sum(1 for row in part_reports if row.get("part_surface_se3_state") != "part_surface_se3_residual_supported_visible_only_not_pose"),
                "part_pose_ready": False,
                "contact_ownership_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    pair_counts = Counter(str(row.get("part_se3_pair_state")) for row in rows)
    part_counts = Counter(str(part.get("part_surface_se3_state")) for row in rows for part in row.get("part_surface_reports", []))
    report = {
        "method": "build_v18_part_se3_surface_residuals",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "articulation_fit_candidates": str(articulation_path),
            "part_visible_surfaces": str(surfaces_path),
            "v16_annotations": str(annotation_path),
        },
        "part_se3_pair_count": len(rows),
        "part_se3_pair_state_counts": dict(sorted(pair_counts.items())),
        "part_surface_se3_state_counts": dict(sorted(part_counts.items())),
        "rows": rows,
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_se3_surface_residuals_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    pair_counts: Counter[str] = Counter()
    part_counts: Counter[str] = Counter()
    for report in reports:
        pair_counts.update(report["part_se3_pair_state_counts"])
        part_counts.update(report["part_surface_se3_state_counts"])
    summary = {
        "method": "build_v18_part_se3_surface_residuals",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "part_se3_pair_count": sum(require_int(report.get("part_se3_pair_count"), "pair count") for report in reports),
        "part_se3_pair_state_counts": dict(sorted(pair_counts.items())),
        "part_surface_se3_state_counts": dict(sorted(part_counts.items())),
        "part_pose_ready_count": 0,
        "contact_ownership_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_se3_surface_residuals_report.json"),
                "part_se3_pair_count": report["part_se3_pair_count"],
                "part_se3_pair_state_counts": report["part_se3_pair_state_counts"],
                "part_surface_se3_state_counts": report["part_surface_se3_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_se3_surface_residuals_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articulation-fit-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_articulation_fit_candidates"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_se3_surface_residuals"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-icp-points", type=int, default=30)
    parser.add_argument("--max-icp-points", type=int, default=512)
    parser.add_argument("--max-probe-frames", type=int, default=24)
    parser.add_argument("--max-exhaustive-shared-frames", type=int, default=64)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--icp-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--min-successful-frames", type=int, default=8)
    parser.add_argument("--max-median-residual-m", type=float, default=0.02)
    parser.add_argument("--max-p95-of-p95-residual-m", type=float, default=0.06)
    parser.add_argument("--min-robust-surface-inlier-ratio", type=float, default=0.5)
    parser.add_argument("--max-reported-outlier-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
