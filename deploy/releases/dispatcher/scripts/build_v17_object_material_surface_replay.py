#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_object_material_surface_replay_qc"
CLAIM = (
    "This artifact replays visible object surfaces through partial material-pose candidates and compares them "
    "with observed RGBD visible surfaces. It tests surface consistency only; it does not reconstruct hidden "
    "geometry, full object topology, or full-timeline object pose."
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


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty JSON string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be a JSON integer")
    return value


def summarize(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RuntimeError("invalid surface point cloud")
    if len(points) <= max_points:
        return points.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[idx].astype(np.float64, copy=False)


def nearest_distances(source: np.ndarray, target: np.ndarray, chunk: int) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        raise RuntimeError("nearest distance requires non-empty point clouds")
    out = np.empty((len(source),), dtype=np.float64)
    for start in range(0, len(source), chunk):
        block = source[start : start + chunk]
        diff = block[:, None, :] - target[None, :, :]
        out[start : start + len(block)] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def load_visible_surfaces(report_path: Path, archive_path: Path) -> dict[tuple[str, int], np.ndarray]:
    report = require_dict(load_json(report_path), f"{report_path}")
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
    surfaces: dict[tuple[str, int], np.ndarray] = {}
    for i, row in enumerate(rows):
        row_frame = require_int(row.get("frame_idx"), f"surface_rows[{i}].frame_idx")
        row_object = require_str(row.get("object_id"), f"surface_rows[{i}].object_id")
        if row_frame != int(frame_idx[i]) or row_object != str(object_id[i]):
            raise RuntimeError(f"{archive_path} archive row {i} disagrees with report")
        start = int(vertex_offsets[i])
        end = int(vertex_offsets[i + 1])
        points = vertices[start:end]
        if len(points) == 0 or not np.isfinite(points).all():
            raise RuntimeError(f"{archive_path} row {i} has invalid vertices")
        key = (row_object, row_frame)
        if key in surfaces:
            raise RuntimeError(f"duplicate visible surface row: {key}")
        surfaces[key] = points
    return surfaces


def candidate_replay(
    *,
    case: str,
    candidate: dict[str, Any],
    surfaces: dict[tuple[str, int], np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_id = require_str(candidate.get("candidate_id"), "candidate_id")
    object_id = require_str(candidate.get("object_id"), "candidate object_id")
    start_frame = require_int(candidate.get("start_frame"), "candidate start_frame")
    source_surface = surfaces.get((object_id, start_frame))
    if source_surface is None:
        raise RuntimeError(f"{case} candidate {candidate_id} missing source visible surface for {object_id} frame {start_frame}")
    source_sample = sample_points(source_surface, int(args.max_source_points), stable_seed(case, candidate_id, "source"))
    frame_rows: list[dict[str, Any]] = []
    all_symmetric: list[float] = []
    for frame_i, raw_frame in enumerate(require_list(candidate.get("frame_rows"), f"{candidate_id}.frame_rows")):
        frame = require_dict(raw_frame, f"{candidate_id}.frame_rows[{frame_i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"{candidate_id}.frame_rows[{frame_i}].frame_idx")
        target_surface = surfaces.get((object_id, frame_idx))
        if target_surface is None:
            raise RuntimeError(f"{case} candidate {candidate_id} missing target visible surface for {object_id} frame {frame_idx}")
        target_sample = sample_points(target_surface, int(args.max_target_points), stable_seed(case, candidate_id, frame_idx, "target"))
        rot = np.asarray(frame.get("rotation"), dtype=np.float64)
        trans = np.asarray(frame.get("translation_m"), dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,) or not np.isfinite(rot).all() or not np.isfinite(trans).all():
            raise RuntimeError(f"{case} candidate {candidate_id} frame {frame_idx} has invalid transform")
        replay = source_sample @ rot + trans
        source_to_target = nearest_distances(replay, target_sample, int(args.distance_chunk))
        target_to_source = nearest_distances(target_sample, replay, int(args.distance_chunk))
        symmetric = np.concatenate([source_to_target, target_to_source])
        symmetric_summary = summarize(symmetric)
        all_symmetric.extend(symmetric.astype(float).tolist())
        checks = {
            "surface_replay_symmetric_p95_met": bool(
                float(symmetric_summary.get("p95", float("inf"))) <= float(args.accept_symmetric_p95_m)
            ),
            "surface_replay_median_met": bool(
                float(symmetric_summary.get("median", float("inf"))) <= float(args.accept_symmetric_median_m)
            ),
            "min_source_surface_points_met": bool(len(source_surface) >= int(args.min_surface_vertices)),
            "min_target_surface_points_met": bool(len(target_surface) >= int(args.min_surface_vertices)),
        }
        frame_rows.append(
            {
                "frame_idx": frame_idx,
                "source_surface_vertices": int(len(source_surface)),
                "target_surface_vertices": int(len(target_surface)),
                "sampled_source_points": int(len(source_sample)),
                "sampled_target_points": int(len(target_sample)),
                "source_to_target_m": summarize(source_to_target),
                "target_to_source_m": summarize(target_to_source),
                "symmetric_surface_replay_m": symmetric_summary,
                "readiness_checks": checks,
                "partial_visible_surface_replay_candidate": bool(all(checks.values())),
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
        )
    all_checks = [bool(row["partial_visible_surface_replay_candidate"]) for row in frame_rows]
    candidate_checks = {
        "source_pose_candidate_ready": bool(candidate.get("partial_material_pose_candidate") is True),
        "all_frame_surface_replay_checks_met": bool(all(all_checks)),
    }
    return {
        "candidate_id": candidate_id,
        "case": case,
        "object_id": object_id,
        "track_id": require_str(candidate.get("track_id"), "candidate track_id"),
        "window_id": require_str(candidate.get("window_id"), "candidate window_id"),
        "start_frame": start_frame,
        "end_frame": require_int(candidate.get("end_frame"), "candidate end_frame"),
        "frame_count": require_int(candidate.get("frame_count"), "candidate frame_count"),
        "track_count": require_int(candidate.get("track_count"), "candidate track_count"),
        "surface_replay_m": summarize(np.asarray(all_symmetric, dtype=np.float64)),
        "readiness_checks": candidate_checks,
        "partial_visible_surface_replay_candidate": bool(all(candidate_checks.values())),
        "frame_rows": frame_rows,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    pose_report_path = args.material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json"
    visible_report_path = args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json"
    visible_archive_path = args.visible_surface_root / case / "multi_object_visible_surfaces_world.npz"
    pose_report = require_dict(load_json(pose_report_path), f"{case} material-pose report")
    surfaces = load_visible_surfaces(visible_report_path, visible_archive_path)
    candidate_rows = [
        candidate_replay(case=case, candidate=require_dict(raw, f"candidates[{i}]"), surfaces=surfaces, args=args)
        for i, raw in enumerate(require_list(pose_report.get("candidates"), "pose candidates"))
    ]
    ready = [row for row in candidate_rows if row["partial_visible_surface_replay_candidate"]]
    report = {
        "method": "build_v17_object_material_surface_replay",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_material_pose_candidate_report": str(pose_report_path),
        "source_visible_surface_report": str(visible_report_path),
        "source_visible_surface_archive": str(visible_archive_path),
        "partial_material_pose_candidate_segment_count": require_int(
            pose_report.get("partial_material_pose_candidate_segment_count"),
            "pose partial_material_pose_candidate_segment_count",
        ),
        "partial_material_pose_candidate_ready_segment_count": require_int(
            pose_report.get("partial_material_pose_candidate_ready_segment_count"),
            "pose partial_material_pose_candidate_ready_segment_count",
        ),
        "partial_visible_surface_replay_candidate_count": len(candidate_rows),
        "partial_visible_surface_replay_ready_count": len(ready),
        "ready_candidate_ids": [require_str(row.get("candidate_id"), "ready candidate_id") for row in ready],
        "candidates": candidate_rows,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
        "parameters": {
            "max_source_points": int(args.max_source_points),
            "max_target_points": int(args.max_target_points),
            "min_surface_vertices": int(args.min_surface_vertices),
            "accept_symmetric_p95_m": float(args.accept_symmetric_p95_m),
            "accept_symmetric_median_m": float(args.accept_symmetric_median_m),
        },
    }
    write_json(args.output_root / case / "v17_object_material_surface_replay_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    pose_summary_path = args.material_pose_candidate_root / "v17_object_material_pose_candidate_summary.json"
    pose_summary = require_dict(load_json(pose_summary_path), "material-pose candidate summary")
    cases = [
        require_str(require_dict(raw, f"pose summary cases[{i}]").get("case"), "pose summary case")
        for i, raw in enumerate(require_list(pose_summary.get("cases"), "pose summary cases"))
    ]
    reports = [build_case(case, args) for case in cases]
    payload = {
        "method": "build_v17_object_material_surface_replay",
        "status": STATUS,
        "claim": CLAIM,
        "source_material_pose_candidate_summary": str(pose_summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_object_material_surface_replay_report.json"
                ),
                "partial_material_pose_candidate_segment_count": report[
                    "partial_material_pose_candidate_segment_count"
                ],
                "partial_material_pose_candidate_ready_segment_count": report[
                    "partial_material_pose_candidate_ready_segment_count"
                ],
                "partial_visible_surface_replay_candidate_count": report[
                    "partial_visible_surface_replay_candidate_count"
                ],
                "partial_visible_surface_replay_ready_count": report[
                    "partial_visible_surface_replay_ready_count"
                ],
                "ready_candidate_ids": report["ready_candidate_ids"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for report in reports
        ],
        "partial_material_pose_candidate_segment_count": sum(
            require_int(report.get("partial_material_pose_candidate_segment_count"), "partial material pose candidate count")
            for report in reports
        ),
        "partial_visible_surface_replay_candidate_count": sum(
            require_int(report.get("partial_visible_surface_replay_candidate_count"), "partial visible replay candidate count")
            for report in reports
        ),
        "partial_visible_surface_replay_ready_count": sum(
            require_int(report.get("partial_visible_surface_replay_ready_count"), "partial visible replay ready count")
            for report in reports
        ),
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_object_material_surface_replay_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material-pose-candidate-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_pose_candidates"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"),
    )
    parser.add_argument("--max-source-points", type=int, default=1024)
    parser.add_argument("--max-target-points", type=int, default=1024)
    parser.add_argument("--min-surface-vertices", type=int, default=50)
    parser.add_argument("--accept-symmetric-p95-m", type=float, default=0.03)
    parser.add_argument("--accept-symmetric-median-m", type=float, default=0.015)
    parser.add_argument("--distance-chunk", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
