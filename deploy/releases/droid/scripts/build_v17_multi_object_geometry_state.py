#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree


STATUS = "multi_object_geometry_state_diagnostic_not_pose_solver"
CLAIM = (
    "This artifact measures center-normalized visible-surface envelope stability from the V17 multi-object RGBD surfaces. "
    "It identifies surface evidence available to future geometry variables but does not solve canonical geometry or object pose."
)


@dataclass(frozen=True)
class Surface:
    frame_idx: int
    object_id: str
    row: dict[str, Any]
    vertices_world: np.ndarray
    points_local: np.ndarray
    center_world: np.ndarray


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


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be numeric") from exc
    if arr.shape != shape or not np.isfinite(arr).all():
        raise RuntimeError(f"{label} must have shape {shape} with finite values")
    return arr


def finite_optional_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= int(max_points):
        return points.astype(np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(points), size=int(max_points), replace=False)
    return points[idx].astype(np.float64)


def stable_seed(case: str, object_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{case}|{object_id}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def archive_surfaces(report: dict[str, Any], archive_path: Path, args: argparse.Namespace) -> list[Surface]:
    rows = [require_dict(row, f"surface_rows[{i}]") for i, row in enumerate(require_list(report.get("surface_rows"), "surface_rows"))]
    blob = np.load(archive_path, allow_pickle=False)
    required = {"frame_idx", "object_id", "vertex_offsets", "vertices"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{archive_path} missing archive keys: {missing}")
    frame_idx = blob["frame_idx"].astype(np.int64)
    object_id = blob["object_id"].astype(str)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    if len(rows) != len(frame_idx) or len(vertex_offsets) != len(rows) + 1:
        raise RuntimeError(f"{archive_path} row count disagrees with visible-surface report")
    surfaces: list[Surface] = []
    for i, row in enumerate(rows):
        row_frame = require_int(row.get("frame_idx"), f"surface_rows[{i}].frame_idx")
        row_object = require_str(row.get("object_id"), f"surface_rows[{i}].object_id")
        if row_frame != int(frame_idx[i]) or row_object != str(object_id[i]):
            raise RuntimeError(f"{archive_path} archive row {i} disagrees with report")
        start = int(vertex_offsets[i])
        end = int(vertex_offsets[i + 1])
        surface_vertices = vertices[start:end]
        if len(surface_vertices) == 0 or not np.isfinite(surface_vertices).all():
            raise RuntimeError(f"{archive_path} row {i} has invalid vertices")
        center = finite_array(row.get("center_world_m"), (3,), f"surface_rows[{i}].center_world_m")
        sampled_world = sample_points(surface_vertices, int(args.max_points_per_surface), int(args.seed) + row_frame + i)
        surfaces.append(
            Surface(
                frame_idx=row_frame,
                object_id=row_object,
                row=row,
                vertices_world=surface_vertices,
                points_local=sampled_world - center[None, :],
                center_world=center,
            )
        )
    return surfaces


def pca_shape(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 3:
        return {"status": "insufficient_points"}
    centered = points - np.median(points, axis=0)
    cov = centered.T @ centered / max(float(len(centered) - 1), 1.0)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(np.maximum(eig, 0.0))[::-1]
    if not np.isfinite(eig).all() or eig[0] <= 1e-12:
        return {"status": "degenerate_points", "eigenvalues": eig.astype(float).tolist()}
    return {
        "status": "ok",
        "eigenvalues": eig.astype(float).tolist(),
        "lambda2_over_lambda1": float(eig[1] / eig[0]),
        "lambda3_over_lambda1": float(eig[2] / eig[0]),
    }


def cross_surface_residuals(surfaces: list[Surface], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parity_points: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for surface in surfaces:
        parity_points[surface.frame_idx % 2].append(surface.points_local)
    envelope: dict[int, np.ndarray] = {}
    for parity, chunks in parity_points.items():
        if chunks:
            pts = np.vstack(chunks)
            envelope[parity] = sample_points(
                pts,
                int(args.max_canonical_points_per_partition),
                int(args.seed) + 101 * parity,
            )
        else:
            envelope[parity] = np.zeros((0, 3), dtype=np.float64)
    if min(len(envelope[0]), len(envelope[1])) < int(args.min_cross_partition_points):
        return (
            {
                "status": "rejected_insufficient_cross_partition_points",
                "partition_points": {str(key): int(len(value)) for key, value in envelope.items()},
            },
            [],
        )
    trees = {parity: KDTree(points) for parity, points in envelope.items()}
    all_distances: list[float] = []
    rows: list[dict[str, Any]] = []
    for surface in surfaces:
        target_parity = 1 - (surface.frame_idx % 2)
        distances = trees[target_parity].query(surface.points_local, k=1)[0].astype(np.float64)
        all_distances.extend(distances.astype(float).tolist())
        rows.append(
            {
                "frame_idx": surface.frame_idx,
                "object_id": surface.object_id,
                "sampled_points": int(len(surface.points_local)),
                "query_partition": int(target_parity),
                "surface_to_cross_partition_canonical_m": finite_optional_summary(distances),
            }
        )
    return (
        {
            "status": "accepted_cross_partition_residuals",
            "partition_points": {str(key): int(len(value)) for key, value in envelope.items()},
            "surface_to_cross_partition_canonical_m": finite_optional_summary(
                np.asarray(all_distances, dtype=np.float64)
            ),
        },
        rows,
    )


def object_report(
    case: str,
    object_id: str,
    surfaces: list[Surface],
    rejected_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray]:
    if not surfaces:
        return (
            {
                "object_id": object_id,
                "status": "rejected_no_visible_surface_measurements",
                "surface_frame_count": 0,
                "rejected_visible_surface_frame_count": len(rejected_rows),
                "center_normalized_visible_surface_points": 0,
                "cross_surface_residual": {"status": "rejected_no_visible_surface_measurements"},
                "visible_surface_envelope_candidate": False,
                "persistent_visible_surface_candidate": False,
                "rigid_pose_candidate": False,
                "geometry_state": "no_visible_surface_geometry_measurement",
                "pose_state": "no_object_pose_variable",
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "annotation_ready": False,
                "deliverable_ready": False,
                "v3_solver_complete": False,
            },
            np.zeros((0, 3), dtype=np.float32),
        )
    centers = np.asarray([surface.center_world for surface in surfaces], dtype=np.float64)
    extents = np.asarray([surface.row["world_extent_m"] for surface in surfaces], dtype=np.float64)
    if extents.ndim != 2 or extents.shape[1] != 3 or not np.isfinite(extents).all():
        raise RuntimeError(f"{case} {object_id} has invalid extent rows")
    center_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros((0,), dtype=np.float64)
    frame_gaps = np.diff(np.asarray([surface.frame_idx for surface in surfaces], dtype=np.int64)) if len(surfaces) > 1 else np.zeros((0,), dtype=np.int64)
    sampled_points = np.vstack([surface.points_local for surface in surfaces])
    envelope_points = sample_points(
        sampled_points,
        int(args.max_canonical_points_per_object),
        stable_seed(case, object_id, int(args.seed)),
    )
    residual_summary, residual_rows = cross_surface_residuals(surfaces, args)
    residual_p95 = None
    if residual_summary.get("status") == "accepted_cross_partition_residuals":
        residual = require_dict(
            residual_summary.get("surface_to_cross_partition_canonical_m"),
            f"{case} {object_id} residual summary",
        )
        value = residual.get("p95")
        residual_p95 = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
    extent_cv = np.std(extents, axis=0) / np.maximum(np.mean(extents, axis=0), 1e-9)
    max_extent_cv = float(np.max(extent_cv))
    enough_surfaces = len(surfaces) >= int(args.min_surface_frames)
    cross_residual_ready = residual_p95 is not None and residual_p95 <= float(args.max_persistent_surface_p95_m)
    stable_extent = max_extent_cv <= float(args.max_extent_cv)
    envelope_candidate = bool(cross_residual_ready and stable_extent)
    if not enough_surfaces:
        status = "rejected_insufficient_visible_surface_frames"
    elif residual_summary.get("status") != "accepted_cross_partition_residuals":
        status = str(residual_summary.get("status"))
    elif envelope_candidate:
        status = "center_normalized_visible_surface_envelope_available"
    else:
        status = "deformable_or_inconsistent_visible_surface_evidence"
    return (
        {
            "object_id": object_id,
            "status": status,
            "surface_frame_count": len(surfaces),
            "rejected_visible_surface_frame_count": len(rejected_rows),
            "first_surface_frame": int(surfaces[0].frame_idx),
            "last_surface_frame": int(surfaces[-1].frame_idx),
            "frame_gap": finite_optional_summary(frame_gaps.astype(np.float64)),
            "center_step_m": finite_optional_summary(center_steps),
            "center_span_m": (centers.max(axis=0) - centers.min(axis=0)).astype(float).tolist(),
            "extent_m": {
                "median": np.median(extents, axis=0).astype(float).tolist(),
                "p05": np.percentile(extents, 5.0, axis=0).astype(float).tolist(),
                "p95": np.percentile(extents, 95.0, axis=0).astype(float).tolist(),
                "cv_by_axis": extent_cv.astype(float).tolist(),
                "max_cv": max_extent_cv,
            },
            "center_normalized_visible_surface_points": int(len(envelope_points)),
            "center_normalized_visible_surface_shape": pca_shape(envelope_points),
            "cross_surface_residual": residual_summary,
            "cross_surface_residual_rows_preview": residual_rows[: int(args.residual_rows_preview)],
            "visible_surface_envelope_candidate": envelope_candidate,
            "persistent_visible_surface_candidate": False,
            "rigid_pose_candidate": False,
            "geometry_state": "center_normalized_visible_surface_envelope_not_canonical_object_mesh",
            "pose_state": "no_object_pose_variable",
            "test_limitations": [
                "surface rows are center-normalized before comparison",
                "the residual measures envelope repeatability, not material correspondence",
                "deformation, symmetric rigid rotation, and repeated depth silhouettes can produce the same residual",
                "object pose and deformation variables remain absent",
            ],
            "object_geometry_complete": False,
            "object_pose_requirement_met": False,
            "annotation_ready": False,
            "deliverable_ready": False,
            "v3_solver_complete": False,
        },
        envelope_points.astype(np.float32),
    )


def save_center_normalized_archive(path: Path, object_reports: list[dict[str, Any]], envelope_points: list[np.ndarray]) -> None:
    offsets = [0]
    chunks: list[np.ndarray] = []
    for points in envelope_points:
        chunks.append(points.astype(np.float32))
        offsets.append(offsets[-1] + int(len(points)))
    payload = {
        "status": STATUS,
        "claim": CLAIM,
        "object_count": len(object_reports),
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_id=np.asarray([require_str(row.get("object_id"), "object_id") for row in object_reports]),
        point_offsets=np.asarray(offsets, dtype=np.int64),
        points_local=np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, 3), dtype=np.float32),
        v17_archive_metadata_json=json.dumps(payload),
    )


def build_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    case_root = args.output_root / case
    report_path = args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json"
    report = require_dict(load_json(report_path), f"{case} visible-surface report")
    archive_path = Path(require_str(report.get("mesh_archive"), f"{case} mesh_archive"))
    surfaces = archive_surfaces(report, archive_path, args)
    rejected_rows = [require_dict(row, f"{case} rejected_rows[{i}]") for i, row in enumerate(require_list(report.get("rejected_rows"), f"{case} rejected_rows"))]
    object_ids = sorted({surface.object_id for surface in surfaces} | {require_str(row.get("object_id"), "rejected object_id") for row in rejected_rows})
    reports: list[dict[str, Any]] = []
    envelope_chunks: list[np.ndarray] = []
    for object_id in object_ids:
        object_surfaces = sorted([surface for surface in surfaces if surface.object_id == object_id], key=lambda surface: surface.frame_idx)
        object_rejections = [row for row in rejected_rows if require_str(row.get("object_id"), "rejected object_id") == object_id]
        row, points = object_report(case, object_id, object_surfaces, object_rejections, args)
        reports.append(row)
        envelope_chunks.append(points)
    legacy_archive = case_root / "multi_object_canonical_visible_surface_points.npz"
    if legacy_archive.exists():
        legacy_archive.unlink()
    envelope_archive = case_root / "multi_object_center_normalized_visible_surface_points.npz"
    save_center_normalized_archive(envelope_archive, reports, envelope_chunks)
    candidate_count = sum(1 for row in reports if row.get("visible_surface_envelope_candidate") is True)
    payload = {
        "method": "build_v17_multi_object_geometry_state",
        "case": case,
        "status": STATUS,
        "claim": CLAIM,
        "source_visible_surface_report": str(report_path),
        "source_visible_surface_archive": str(archive_path),
        "center_normalized_visible_surface_points_archive": str(envelope_archive),
        "frame_count": require_int(report.get("frame_count"), f"{case} frame_count"),
        "object_count": len(reports),
        "surface_frame_rows": require_int(report.get("surface_frame_rows"), f"{case} surface_frame_rows"),
        "rejected_visible_object_frame_rows": require_int(report.get("rejected_visible_object_frame_rows"), f"{case} rejected_visible_object_frame_rows"),
        "visible_surface_envelope_candidate_count": int(candidate_count),
        "persistent_visible_surface_candidate_count": 0,
        "rigid_pose_candidate_count": 0,
        "objects": reports,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
        "thresholds": {
            "min_surface_frames": int(args.min_surface_frames),
            "max_points_per_surface": int(args.max_points_per_surface),
            "max_canonical_points_per_object": int(args.max_canonical_points_per_object),
            "max_canonical_points_per_partition": int(args.max_canonical_points_per_partition),
            "min_cross_partition_points": int(args.min_cross_partition_points),
            "max_persistent_surface_p95_m": float(args.max_persistent_surface_p95_m),
            "max_extent_cv": float(args.max_extent_cv),
        },
    }
    write_json(case_root / "v17_multi_object_geometry_state_report.json", payload)
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [build_case(args, case) for case in args.case]
    payload = {
        "method": "build_v17_multi_object_geometry_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "cases": [
            {
                "case": case["case"],
                "report": str(args.output_root / case["case"] / "v17_multi_object_geometry_state_report.json"),
                "center_normalized_visible_surface_points_archive": case["center_normalized_visible_surface_points_archive"],
                "object_count": case["object_count"],
                "surface_frame_rows": case["surface_frame_rows"],
                "visible_surface_envelope_candidate_count": case["visible_surface_envelope_candidate_count"],
                "persistent_visible_surface_candidate_count": 0,
                "rigid_pose_candidate_count": 0,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
            for case in cases
        ],
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_multi_object_geometry_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_geometry_state"),
    )
    parser.add_argument("--case", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-surface-frames", type=int, default=30)
    parser.add_argument("--max-points-per-surface", type=int, default=256)
    parser.add_argument("--max-canonical-points-per-object", type=int, default=50000)
    parser.add_argument("--max-canonical-points-per-partition", type=int, default=40000)
    parser.add_argument("--min-cross-partition-points", type=int, default=1024)
    parser.add_argument("--max-persistent-surface-p95-m", type=float, default=0.030)
    parser.add_argument("--max-extent-cv", type=float, default=0.60)
    parser.add_argument("--residual-rows-preview", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
