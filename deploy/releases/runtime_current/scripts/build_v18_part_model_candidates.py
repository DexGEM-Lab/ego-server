#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict, deque
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

STATUS = "v18_part_model_candidates"
CLAIM = (
    "This artifact records bounded visible part-model candidates from robust stable part-surface relationships. "
    "It does not complete hidden geometry, fit articulation, estimate part pose, or satisfy object-pose requirements."
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


def finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "count": len(clean),
        "median": percentile(clean, 50.0),
        "p05": percentile(clean, 5.0),
        "p95": percentile(clean, 95.0),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def stable_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        component: list[str] = []
        q: deque[str] = deque([start])
        seen.add(start)
        while q:
            item = q.popleft()
            component.append(item)
            for nxt in sorted(graph[item]):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(component) >= 2:
            components.append(sorted(component))
    return components


def rows_for_component(surface_rows: list[dict[str, Any]], object_id: str, labels: set[str]) -> list[dict[str, Any]]:
    return [row for row in surface_rows if str(row.get("object_id")) == object_id and str(row.get("part_track_label")) in labels]


def archive_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["frame_idx", "object_id", "part_track_label", "vertex_offsets", "face_offsets", "vertices", "faces"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"part surface archive {path} missing keys: {missing}")
    frame_idx = data["frame_idx"]
    object_id = data["object_id"]
    part_track_label = data["part_track_label"]
    vertex_offsets = data["vertex_offsets"]
    face_offsets = data["face_offsets"]
    vertices = data["vertices"]
    faces = data["faces"]
    row_count = int(frame_idx.shape[0])
    if int(object_id.shape[0]) != row_count or int(part_track_label.shape[0]) != row_count:
        raise RuntimeError("part surface archive row metadata length mismatch")
    if int(vertex_offsets.shape[0]) != row_count + 1 or int(face_offsets.shape[0]) != row_count + 1:
        raise RuntimeError("part surface archive offsets must have row_count+1 entries")
    if vertices.ndim != 2 or int(vertices.shape[1]) != 3:
        raise RuntimeError("part surface archive vertices must be Nx3")
    if faces.ndim != 2 or int(faces.shape[1]) != 3:
        raise RuntimeError("part surface archive faces must be Mx3")
    return {
        "frame_idx": frame_idx,
        "object_id": object_id,
        "part_track_label": part_track_label,
        "vertex_offsets": vertex_offsets,
        "face_offsets": face_offsets,
        "vertices": vertices,
        "faces": faces,
    }


def row_vertices(arrays: dict[str, np.ndarray], row_index: int) -> np.ndarray:
    start = int(arrays["vertex_offsets"][row_index])
    end = int(arrays["vertex_offsets"][row_index + 1])
    return np.asarray(arrays["vertices"][start:end], dtype=np.float64)


def deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if int(points.shape[0]) <= max_points:
        return points
    idx = np.linspace(0, int(points.shape[0]) - 1, max_points, dtype=np.int64)
    return points[idx]


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


def icp_residual(source_points: np.ndarray, target_points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    min_points = int(args.min_surface_icp_points)
    if int(source_points.shape[0]) < min_points or int(target_points.shape[0]) < min_points:
        raise RuntimeError("too_few_points_for_surface_icp")
    source = deterministic_sample(source_points, int(args.max_surface_icp_points))
    target = deterministic_sample(target_points, int(args.max_surface_icp_points))
    transformed = source.copy()
    tree = KDTree(target)
    previous_median: float | None = None
    iterations = 0
    for iterations in range(1, int(args.surface_icp_iterations) + 1):
        distances, indices = tree.query(transformed, k=1)
        matched = target[np.asarray(indices, dtype=np.int64)]
        r, t = rigid_transform(transformed, matched)
        transformed = (r @ transformed.T).T + t
        median = float(np.median(distances)) if len(distances) else 0.0
        if previous_median is not None and abs(previous_median - median) < float(args.surface_icp_tolerance_m):
            break
        previous_median = median
    distances, _ = tree.query(transformed, k=1)
    residuals = [float(v) for v in np.asarray(distances, dtype=np.float64).tolist()]
    return {
        "source_point_count": int(source_points.shape[0]),
        "target_point_count": int(target_points.shape[0]),
        "sampled_source_point_count": int(source.shape[0]),
        "sampled_target_point_count": int(target.shape[0]),
        "iterations": iterations,
        "residual_m": stats(residuals),
    }


def surface_icp_probe(
    object_id: str,
    part_label: str,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sorted_rows = sorted(indexed_rows, key=lambda item: (require_int(item[1].get("frame_idx"), "frame_idx"), item[0]))
    vertex_counts = [require_int(row.get("vertices"), "vertices") for _, row in sorted_rows]
    face_counts = [require_int(row.get("faces"), "faces") for _, row in sorted_rows]
    containment = [finite_float(row.get("part_containment_in_object"), "containment") for _, row in sorted_rows]
    if not sorted_rows:
        raise RuntimeError("surface_icp_probe requires at least one surface row")
    reference_index, reference_row = max(sorted_rows, key=lambda item: require_int(item[1].get("vertices"), "vertices"))
    reference_vertices = row_vertices(arrays, reference_index)
    eligible_rows = [(idx, row) for idx, row in sorted_rows if require_int(row.get("vertices"), "vertices") >= int(args.min_surface_icp_points)]
    if len(eligible_rows) > int(args.max_surface_icp_probe_frames):
        selected_indices = np.linspace(0, len(eligible_rows) - 1, int(args.max_surface_icp_probe_frames), dtype=np.int64)
        eligible_rows = [eligible_rows[int(i)] for i in selected_indices]
    frame_results: list[dict[str, Any]] = []
    rejected_frames: list[dict[str, Any]] = []
    for row_index, row in eligible_rows:
        if row_index == reference_index:
            continue
        try:
            result = icp_residual(row_vertices(arrays, row_index), reference_vertices, args)
        except RuntimeError as exc:
            rejected_frames.append({"frame_idx": row.get("frame_idx"), "reason": str(exc)})
            continue
        frame_results.append({"frame_idx": row.get("frame_idx"), "archive_row_index": row_index, **result})
    median_residuals = [finite_float(result["residual_m"].get("median"), "median residual") for result in frame_results if result.get("residual_m", {}).get("median") is not None]
    p95_residuals = [finite_float(result["residual_m"].get("p95"), "p95 residual") for result in frame_results if result.get("residual_m", {}).get("p95") is not None]
    blocker_reasons: list[str] = []
    median_vertices = float(np.median(vertex_counts)) if vertex_counts else 0.0
    median_faces = float(np.median(face_counts)) if face_counts else 0.0
    successful = len(frame_results)
    residual_median_stats = stats(median_residuals)
    residual_p95_stats = stats(p95_residuals)
    residual_median = residual_median_stats.get("median")
    residual_p95 = residual_p95_stats.get("median")
    if successful < int(args.min_surface_icp_successful_frames):
        blocker_reasons.append("too_few_successful_surface_icp_frames")
    if median_vertices < float(args.min_surface_icp_robust_median_vertices):
        blocker_reasons.append("surface_icp_sparse_median_vertices")
    if median_faces < float(args.min_surface_icp_robust_median_faces):
        blocker_reasons.append("surface_icp_sparse_median_faces")
    if residual_median is None or float(residual_median) > float(args.accept_surface_icp_median_residual_m):
        blocker_reasons.append("surface_icp_median_residual_above_threshold")
    if residual_p95 is None or float(residual_p95) > float(args.accept_surface_icp_p95_residual_m):
        blocker_reasons.append("surface_icp_p95_residual_above_threshold")
    if not blocker_reasons:
        state = "surface_icp_residual_supported_visible_only_not_pose"
    elif successful < int(args.min_surface_icp_successful_frames):
        state = "surface_icp_underconstrained"
    elif "surface_icp_sparse_median_vertices" in blocker_reasons or "surface_icp_sparse_median_faces" in blocker_reasons:
        state = "surface_icp_sparse_or_unstable_not_pose"
    else:
        state = "surface_icp_residual_rejected"
    return {
        "object_id": object_id,
        "part_track_label": part_label,
        "probe_type": "visible_part_surface_icp_to_reference_frame",
        "coordinate_frame": "per-frame metric depth camera; ICP estimates arbitrary rigid alignment to the reference surface for shape consistency only",
        "surface_frame_count": len(sorted_rows),
        "eligible_frame_count": len(eligible_rows),
        "successful_icp_frame_count": successful,
        "rejected_icp_frame_count": len(rejected_frames),
        "reference_frame_idx": reference_row.get("frame_idx"),
        "reference_archive_row_index": reference_index,
        "reference_vertex_count": int(reference_vertices.shape[0]),
        "vertices": {"median": median_vertices, "p05": percentile([float(v) for v in vertex_counts], 5.0), "p95": percentile([float(v) for v in vertex_counts], 95.0)},
        "faces": {"median": median_faces, "p05": percentile([float(v) for v in face_counts], 5.0), "p95": percentile([float(v) for v in face_counts], 95.0)},
        "part_containment_in_object": stats(containment),
        "median_icp_residual_m": residual_median_stats,
        "p95_icp_residual_m": residual_p95_stats,
        "surface_icp_probe_state": state,
        "surface_icp_blockers": blocker_reasons,
        "acceptance_thresholds": {
            "min_successful_frames": int(args.min_surface_icp_successful_frames),
            "min_median_vertices": float(args.min_surface_icp_robust_median_vertices),
            "min_median_faces": float(args.min_surface_icp_robust_median_faces),
            "median_residual_m": float(args.accept_surface_icp_median_residual_m),
            "p95_residual_m": float(args.accept_surface_icp_p95_residual_m),
        },
        "frame_results": frame_results,
        "rejected_frames": rejected_frames,
        "eligible_for_hidden_geometry_completion": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def build_surface_icp_probes(surface_rows: list[dict[str, Any]], archive_path: Path, args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Any]]:
    arrays = archive_arrays(archive_path)
    if int(arrays["frame_idx"].shape[0]) != len(surface_rows):
        raise RuntimeError("part surface archive row count does not match report surface_rows")
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(surface_rows):
        frame_idx = require_int(row.get("frame_idx"), "surface frame_idx")
        object_id = str(row.get("object_id"))
        part_label = str(row.get("part_track_label"))
        if int(arrays["frame_idx"][index]) != frame_idx or str(arrays["object_id"][index]) != object_id or str(arrays["part_track_label"][index]) != part_label:
            raise RuntimeError(f"part surface archive/report mismatch at row {index}")
        grouped[(object_id, part_label)].append((index, row))
    return {key: surface_icp_probe(key[0], key[1], rows, arrays, args) for key, rows in sorted(grouped.items())}


def pair_probe_payload(index: int, object_id: str, pair: dict[str, Any], surface_probes: dict[tuple[str, str], dict[str, Any]], *, state: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "candidate_id": f"{object_id}::pair_surface_residual_probe::{index:02d}",
        "object_id": object_id,
        "candidate_type": "two_part_relative_motion_residual_probe",
        "candidate_state": state,
        "part_track_labels": [str(pair.get("part_a")), str(pair.get("part_b"))],
        "shared_frame_count": pair.get("shared_frame_count"),
        "frame_min": pair.get("frame_min"),
        "frame_max": pair.get("frame_max"),
        "center_distance_m": pair.get("center_distance_m"),
        "p95_minus_p05_distance_m": pair.get("p95_minus_p05_distance_m"),
        "pair_motion_state": pair.get("pair_motion_state"),
        "pair_qc_state": pair.get("pair_qc_state"),
        "rejection_reasons": reasons,
        "surface_icp_probes": [surface_probes.get((object_id, str(label))) for label in [pair.get("part_a"), pair.get("part_b")] if surface_probes.get((object_id, str(label))) is not None],
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def rejected_probe_from_pair(index: int, object_id: str, pair: dict[str, Any], surface_probes: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    reasons = list(pair.get("qc_blockers", [])) if isinstance(pair.get("qc_blockers"), list) else [str(pair.get("pair_qc_state"))]
    if not reasons:
        reasons = [str(pair.get("pair_qc_state"))]
    return pair_probe_payload(index, object_id, pair, surface_probes, state="rejected_residual_probe_not_part_model", reasons=reasons)


def articulation_hypothesis_probe_from_pair(index: int, object_id: str, pair: dict[str, Any], surface_probes: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    return pair_probe_payload(
        index,
        object_id,
        pair,
        surface_probes,
        state="articulation_hypothesis_not_fitted_no_pose",
        reasons=["articulation_parameter_fit_not_implemented", "joint_axis_not_estimated", "hidden_geometry_not_completed"],
    )


def rejected_probe_from_single(index: int, object_id: str, part: dict[str, Any], surface_probes: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_id": f"{object_id}::rejected_single_part_surface_probe::{index:02d}",
        "object_id": object_id,
        "candidate_type": "single_part_visible_surface_residual_probe",
        "candidate_state": "rejected_single_part_not_split_model",
        "part_track_labels": [str(part.get("part_track_label"))],
        "part_surface_quality": part.get("part_surface_quality"),
        "quality_metrics": part.get("quality_metrics"),
        "rejection_reasons": ["requires_at_least_two_semantic_part_tracks_for_part_or_articulation_model"],
        "surface_icp_probes": [surface_probes[(object_id, str(part.get("part_track_label")))] ] if (object_id, str(part.get("part_track_label"))) in surface_probes else [],
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }


def candidate_from_component(index: int, object_id: str, labels: list[str], rows: list[dict[str, Any]], source_edges: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sorted({require_int(row.get("frame_idx"), "frame_idx") for row in rows})
    blockers = [
        "visible_surface_subset_only",
        "hidden_geometry_not_completed",
        "whole_object_not_modeled",
        "confounded_variable_part_pairs_excluded",
        "no_part_pose_estimator_applied",
        "no_articulation_parameter_fit",
    ]
    return {
        "candidate_id": f"{object_id}::visible_rigid_subset::{index:02d}",
        "object_id": object_id,
        "candidate_type": "robust_stable_visible_part_subset",
        "model_scope": "visible_surface_subset_only",
        "part_track_labels": labels,
        "stable_pair_edges": [
            {"part_a": edge.get("part_a"), "part_b": edge.get("part_b"), "pair_qc_state": edge.get("pair_qc_state")}
            for edge in source_edges
        ],
        "surface_frame_count": len(rows),
        "unique_frame_count": len(frames),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "total_vertices": sum(require_int(row.get("vertices"), "vertices") for row in rows),
        "total_faces": sum(require_int(row.get("faces"), "faces") for row in rows),
        "candidate_ready_for_bounded_visible_subset_model": True,
        "eligible_for_hidden_geometry_completion": False,
        "articulation_model_ready": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
        "completion_blockers": blockers,
    }


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    qc_path = args.part_motion_qc_root / case / "v18_part_motion_qc_report.json"
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    qc = require_dict(load_json(qc_path), f"{case} part motion qc")
    surfaces = require_dict(load_json(surfaces_path), f"{case} part surfaces")
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(surfaces.get("surface_rows"), "surface rows")]
    archive_path = Path(str(surfaces.get("archive_npz")))
    surface_icp_probes = build_surface_icp_probes(surface_rows, archive_path, args)
    surface_icp_state_counts = Counter(str(probe.get("surface_icp_probe_state")) for probe in surface_icp_probes.values())
    object_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    object_state_counts: Counter[str] = Counter()
    for raw_obj in require_list(qc.get("object_rows"), "qc object rows"):
        obj = require_dict(raw_obj, "qc object row")
        object_id = str(obj.get("object_id"))
        stable_edges: list[tuple[str, str]] = []
        stable_edge_rows: list[dict[str, Any]] = []
        confounded_variable_count = 0
        articulation_hypothesis_pair_count = 0
        rejected_obj_candidates: list[dict[str, Any]] = []
        pair_rows = [require_dict(raw_pair, "qc pair row") for raw_pair in require_list(obj.get("pair_rows"), "qc pair rows")]
        for raw_pair in pair_rows:
            pair = require_dict(raw_pair, "qc pair row")
            pair = require_dict(raw_pair, "qc pair row")
            if pair.get("pair_qc_state") == "stable_pair_supported_by_robust_surfaces":
                a = str(pair.get("part_a"))
                b = str(pair.get("part_b"))
                stable_edges.append((a, b))
                stable_edge_rows.append(pair)
            else:
                if pair.get("pair_qc_state") == "variable_pair_confounded_by_part_surface_quality":
                    confounded_variable_count += 1
                    rejected_obj_candidates.append(rejected_probe_from_pair(len(rejected_obj_candidates) + 1, object_id, pair, surface_icp_probes))
                elif pair.get("pair_qc_state") == "variable_pair_between_robust_surfaces_articulation_hypothesis":
                    articulation_hypothesis_pair_count += 1
                    rejected_obj_candidates.append(articulation_hypothesis_probe_from_pair(len(rejected_obj_candidates) + 1, object_id, pair, surface_icp_probes))
                else:
                    rejected_obj_candidates.append(rejected_probe_from_pair(len(rejected_obj_candidates) + 1, object_id, pair, surface_icp_probes))
        components = stable_components(stable_edges)
        obj_candidates: list[dict[str, Any]] = []
        for index, labels in enumerate(components, start=1):
            component_edges = [
                edge
                for edge in stable_edge_rows
                if str(edge.get("part_a")) in labels and str(edge.get("part_b")) in labels
            ]
            candidate = candidate_from_component(index, object_id, labels, rows_for_component(surface_rows, object_id, set(labels)), component_edges)
            obj_candidates.append(candidate)
            candidates.append(candidate)
        if not pair_rows:
            for raw_part in require_list(obj.get("part_rows"), "qc part rows"):
                rejected_obj_candidates.append(rejected_probe_from_single(len(rejected_obj_candidates) + 1, object_id, require_dict(raw_part, "qc part row"), surface_icp_probes))
        rejected_candidates.extend(rejected_obj_candidates)
        if obj_candidates:
            state = "visible_stable_part_subset_candidates_only"
        elif articulation_hypothesis_pair_count:
            state = "articulation_hypothesis_not_fitted"
        elif rejected_obj_candidates:
            state = "part_model_residual_probes_rejected"
        elif confounded_variable_count:
            state = "no_model_candidate_due_confounded_variable_pairs"
        else:
            state = "no_part_model_candidate"
        object_state_counts[state] += 1
        object_surface_probe_states = [
            str(probe.get("surface_icp_probe_state"))
            for (probe_object_id, _), probe in surface_icp_probes.items()
            if probe_object_id == object_id
        ]
        object_rows.append(
            {
                "object_id": object_id,
                "source_part_motion_qc_state": obj.get("part_motion_qc_state"),
                "part_model_candidate_state": state,
                "stable_component_count": len(components),
                "confounded_variable_pair_count": confounded_variable_count,
                "articulation_hypothesis_pair_count": articulation_hypothesis_pair_count,
                "candidate_ids": [candidate["candidate_id"] for candidate in obj_candidates],
                "rejected_candidate_ids": [candidate["candidate_id"] for candidate in rejected_obj_candidates],
                "rejected_candidate_count": len(rejected_obj_candidates),
                "surface_icp_probe_count": len(object_surface_probe_states),
                "surface_icp_probe_state_counts": dict(sorted(Counter(object_surface_probe_states).items())),
                "hidden_geometry_reconstructed": False,
                "articulation_model_ready": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_part_model_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_part_motion_qc": str(qc_path), "v18_part_visible_surfaces": str(surfaces_path), "v18_part_visible_surfaces_archive": str(archive_path)},
        "candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "surface_icp_probe_count": len(surface_icp_probes),
        "surface_icp_probe_state_counts": dict(sorted(surface_icp_state_counts.items())),
        "articulation_hypothesis_pair_count": sum(require_int(row.get("articulation_hypothesis_pair_count"), "articulation hypothesis pair count") for row in object_rows),
        "object_state_counts": dict(sorted(object_state_counts.items())),
        "object_rows": object_rows,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "surface_icp_probes": list(surface_icp_probes.values()),
        "visible_subset_model_candidate_count": len(candidates),
        "hidden_geometry_completion_candidate_count": 0,
        "articulation_model_candidate_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_model_candidates_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    surface_icp_state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["object_state_counts"])
        surface_icp_state_counts.update(report["surface_icp_probe_state_counts"])
    summary = {
        "method": "build_v18_part_model_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "candidate_count": sum(require_int(report.get("candidate_count"), "candidate_count") for report in reports),
        "rejected_candidate_count": sum(require_int(report.get("rejected_candidate_count"), "rejected_candidate_count") for report in reports),
        "surface_icp_probe_count": sum(require_int(report.get("surface_icp_probe_count"), "surface_icp_probe_count") for report in reports),
        "surface_icp_probe_state_counts": dict(sorted(surface_icp_state_counts.items())),
        "articulation_hypothesis_pair_count": sum(require_int(report.get("articulation_hypothesis_pair_count"), "articulation hypothesis pair count") for report in reports),
        "visible_subset_model_candidate_count": sum(require_int(report.get("visible_subset_model_candidate_count"), "visible_subset_count") for report in reports),
        "hidden_geometry_completion_candidate_count": 0,
        "articulation_model_candidate_count": 0,
        "articulation_model_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "object_state_counts": dict(sorted(state_counts.items())),
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_model_candidates_report.json"),
                "candidate_count": report["candidate_count"],
                "rejected_candidate_count": report["rejected_candidate_count"],
                "surface_icp_probe_count": report["surface_icp_probe_count"],
                "surface_icp_probe_state_counts": report["surface_icp_probe_state_counts"],
                "articulation_hypothesis_pair_count": report["articulation_hypothesis_pair_count"],
                "object_state_counts": report["object_state_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_model_candidates_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-motion-qc-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_motion_qc"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-surface-icp-points", type=int, default=8)
    parser.add_argument("--max-surface-icp-points", type=int, default=512)
    parser.add_argument("--max-surface-icp-probe-frames", type=int, default=24)
    parser.add_argument("--surface-icp-iterations", type=int, default=12)
    parser.add_argument("--surface-icp-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--min-surface-icp-successful-frames", type=int, default=8)
    parser.add_argument("--min-surface-icp-robust-median-vertices", type=float, default=30.0)
    parser.add_argument("--min-surface-icp-robust-median-faces", type=float, default=30.0)
    parser.add_argument("--accept-surface-icp-median-residual-m", type=float, default=0.02)
    parser.add_argument("--accept-surface-icp-p95-residual-m", type=float, default=0.06)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
