#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_observed_surface_geometry_seed_qc"
CLAIM = (
    "This artifact builds canonical observed-surface geometry seeds from replay-passing material-pose segments. "
    "It maps RGBD visible surfaces back into the segment source frame using material-pose transforms. "
    "The output is an observed-surface seed, not hidden topology reconstruction and not object-pose closure."
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


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def summarize(values: np.ndarray) -> dict[str, Any]:
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
    text = "|".join(str(part) for part in parts)
    total = 0
    for ch in text.encode("utf-8"):
        total = (total * 131 + int(ch)) % (2**32 - 1)
    return total


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def load_visible_surfaces(report_path: Path, archive_path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    report = require_dict(load_json(report_path), f"{report_path}")
    rows = [require_dict(row, f"surface_rows[{i}]") for i, row in enumerate(require_list(report.get("surface_rows"), "surface_rows"))]
    blob = np.load(archive_path, allow_pickle=False)
    required = {"frame_idx", "object_id", "vertex_offsets", "face_offsets", "vertices", "faces"}
    missing = sorted(required.difference(blob.files))
    if missing:
        raise RuntimeError(f"{archive_path} missing archive keys: {missing}")
    frame_idx = blob["frame_idx"].astype(np.int64)
    object_id = blob["object_id"].astype(str)
    vertex_offsets = blob["vertex_offsets"].astype(np.int64)
    face_offsets = blob["face_offsets"].astype(np.int64)
    vertices = blob["vertices"].astype(np.float64)
    faces = blob["faces"].astype(np.int64)
    if len(rows) != len(frame_idx) or len(vertex_offsets) != len(rows) + 1 or len(face_offsets) != len(rows) + 1:
        raise RuntimeError(f"{archive_path} row count disagrees with visible-surface report")
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for i, row in enumerate(rows):
        row_frame = require_int(row.get("frame_idx"), f"surface_rows[{i}].frame_idx")
        row_object = require_str(row.get("object_id"), f"surface_rows[{i}].object_id")
        if row_frame != int(frame_idx[i]) or row_object != str(object_id[i]):
            raise RuntimeError(f"{archive_path} archive row {i} disagrees with report")
        v_start = int(vertex_offsets[i])
        v_end = int(vertex_offsets[i + 1])
        f_start = int(face_offsets[i])
        f_end = int(face_offsets[i + 1])
        row_vertices = vertices[v_start:v_end]
        row_faces = faces[f_start:f_end] - v_start
        if len(row_vertices) == 0 or not np.isfinite(row_vertices).all():
            raise RuntimeError(f"{archive_path} row {i} has invalid vertices")
        if row_faces.size and (row_faces.min() < 0 or row_faces.max() >= len(row_vertices)):
            raise RuntimeError(f"{archive_path} row {i} has invalid local face indices")
        key = (row_object, row_frame)
        if key in out:
            raise RuntimeError(f"duplicate visible surface row: {key}")
        out[key] = (row_vertices, row_faces.astype(np.int32))
    return out


def transform_to_canonical(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points - translation[None, :]) @ rotation.T


def candidate_pose_by_frame(candidate: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, raw in enumerate(require_list(candidate.get("frame_rows"), "candidate frame_rows")):
        row = require_dict(raw, f"candidate frame_rows[{i}]")
        frame_idx = require_int(row.get("frame_idx"), f"candidate frame_rows[{i}].frame_idx")
        rotation = np.asarray(row.get("rotation"), dtype=np.float64)
        translation = np.asarray(row.get("translation_m"), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise RuntimeError(f"candidate frame {frame_idx} has invalid material pose")
        out[frame_idx] = (rotation, translation)
    return out


def replay_candidate_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(report.get("candidates"), "surface replay candidates")):
        row = require_dict(raw, f"surface replay candidates[{i}]")
        candidate_id = require_str(row.get("candidate_id"), f"surface replay candidates[{i}].candidate_id")
        if candidate_id in out:
            raise RuntimeError(f"duplicate replay candidate id: {candidate_id}")
        out[candidate_id] = row
    return out


def save_seed_npz(
    path: Path,
    *,
    frame_idx: list[int],
    vertices: np.ndarray,
    faces: np.ndarray,
    frame_vertex_start: list[int],
    frame_vertex_end: list[int],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "frame_idx": np.asarray(frame_idx, dtype=np.int32),
        "vertices": vertices.astype(np.float32),
        "faces": faces.astype(np.int32),
        "frame_vertex_start": np.asarray(frame_vertex_start, dtype=np.int64),
        "frame_vertex_end": np.asarray(frame_vertex_end, dtype=np.int64),
        "v17_archive_metadata_json": json.dumps(metadata, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def build_seed_for_candidate(
    *,
    case: str,
    pose_candidate: dict[str, Any],
    replay_candidate: dict[str, Any],
    surfaces: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_id = require_str(pose_candidate.get("candidate_id"), "pose candidate_id")
    object_id = require_str(pose_candidate.get("object_id"), "pose object_id")
    if require_str(replay_candidate.get("candidate_id"), "replay candidate_id") != candidate_id:
        raise RuntimeError(f"{case} replay candidate mismatch for {candidate_id}")
    if replay_candidate.get("partial_visible_surface_replay_candidate") is not True:
        raise RuntimeError(f"{case} candidate {candidate_id} is not replay-ready")
    pose_by_frame = candidate_pose_by_frame(pose_candidate)
    frame_vertices: list[np.ndarray] = []
    frame_faces: list[np.ndarray] = []
    frame_indices: list[int] = []
    frame_vertex_start: list[int] = []
    frame_vertex_end: list[int] = []
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    source_vertices: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    vertex_cursor = 0
    for raw in require_list(replay_candidate.get("frame_rows"), f"{candidate_id} replay frame_rows"):
        replay_frame = require_dict(raw, "replay frame")
        frame_idx = require_int(replay_frame.get("frame_idx"), "replay frame_idx")
        if replay_frame.get("partial_visible_surface_replay_candidate") is not True:
            continue
        surface = surfaces.get((object_id, frame_idx))
        if surface is None:
            raise RuntimeError(f"{case} candidate {candidate_id} missing visible surface at frame {frame_idx}")
        rotation, translation = pose_by_frame[frame_idx]
        vertices_world, faces = surface
        canonical_vertices = transform_to_canonical(vertices_world, rotation, translation)
        if source_vertices is None:
            source_vertices = canonical_vertices
        sampled_a = sample_points(
            canonical_vertices,
            int(args.max_eval_points),
            stable_seed(case, candidate_id, frame_idx, "canonical"),
        )
        sampled_b = sample_points(
            source_vertices,
            int(args.max_eval_points),
            stable_seed(case, candidate_id, frame_idx, "source"),
        )
        diff = np.linalg.norm(sampled_a.mean(axis=0) - sampled_b.mean(axis=0))
        frame_indices.append(frame_idx)
        frame_vertex_start.append(vertex_cursor)
        frame_vertex_end.append(vertex_cursor + len(canonical_vertices))
        all_vertices.append(canonical_vertices)
        all_faces.append(faces + vertex_cursor)
        vertex_cursor += len(canonical_vertices)
        frame_vertices.append(canonical_vertices)
        frame_faces.append(faces)
        rows.append(
            {
                "frame_idx": frame_idx,
                "source_surface_vertices": int(len(vertices_world)),
                "canonical_seed_vertices": int(len(canonical_vertices)),
                "canonical_centroid_delta_from_source_m": float(diff),
                "surface_replay_frame_ready": True,
            }
        )
    if len(frame_indices) < int(args.min_seed_frames):
        raise RuntimeError(f"{case} candidate {candidate_id} has only {len(frame_indices)} replay-ready frames")
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int32)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    metadata = {
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "candidate_id": candidate_id,
        "object_id": object_id,
        "track_id": require_str(pose_candidate.get("track_id"), "track_id"),
        "window_id": require_str(pose_candidate.get("window_id"), "window_id"),
        "seed_frame_count": len(frame_indices),
        "seed_vertices": int(len(vertices)),
        "seed_faces": int(len(faces)),
        "observed_surface_only": True,
        "hidden_topology_reconstructed": False,
        "full_active_interval_geometry_ready": False,
        "contact_compatible_geometry_ready": False,
        **FALSE_READY,
    }
    archive_path = output_dir / f"{candidate_id}_observed_surface_seed.npz"
    save_seed_npz(
        archive_path,
        frame_idx=frame_indices,
        vertices=vertices,
        faces=faces,
        frame_vertex_start=frame_vertex_start,
        frame_vertex_end=frame_vertex_end,
        metadata=metadata,
    )
    return {
        **metadata,
        "archive_path": str(archive_path),
        "start_frame": min(frame_indices),
        "end_frame": max(frame_indices),
        "canonical_extent_m": extent.astype(float).tolist(),
        "canonical_centroid_delta_from_source_m": summarize(
            np.asarray([row["canonical_centroid_delta_from_source_m"] for row in rows], dtype=np.float64)
        ),
        "frame_rows": rows,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    pose_path = existing_path(
        args.object_material_pose_candidate_root / case / "v17_object_material_pose_candidate_report.json",
        f"{case} material-pose report",
    )
    replay_path = existing_path(
        args.object_material_surface_replay_root / case / "v17_object_material_surface_replay_report.json",
        f"{case} material-surface replay report",
    )
    visible_report_path = existing_path(
        args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json",
        f"{case} visible-surface report",
    )
    visible_archive_path = existing_path(
        args.visible_surface_root / case / "multi_object_visible_surfaces_world.npz",
        f"{case} visible-surface archive",
    )
    pose_report = require_dict(load_json(pose_path), f"{case} pose report")
    replay_report = require_dict(load_json(replay_path), f"{case} replay report")
    surfaces = load_visible_surfaces(visible_report_path, visible_archive_path)
    replay_by_id = replay_candidate_by_id(replay_report)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for i, raw in enumerate(require_list(pose_report.get("candidates"), "pose candidates")):
        pose = require_dict(raw, f"pose candidates[{i}]")
        candidate_id = require_str(pose.get("candidate_id"), f"pose candidates[{i}].candidate_id")
        replay = replay_by_id.get(candidate_id)
        if replay is None:
            skipped.append({"candidate_id": candidate_id, "reason": "missing_surface_replay_candidate"})
            continue
        if replay.get("partial_visible_surface_replay_candidate") is not True:
            skipped.append({"candidate_id": candidate_id, "reason": "surface_replay_not_ready"})
            continue
        candidates.append(
            build_seed_for_candidate(
                case=case,
                pose_candidate=pose,
                replay_candidate=replay,
                surfaces=surfaces,
                output_dir=args.output_root / case,
                args=args,
            )
        )
    report = {
        "method": "build_v17_observed_surface_geometry_seed",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "object_material_pose_candidate_report": str(pose_path),
            "object_material_surface_replay_report": str(replay_path),
            "visible_surface_report": str(visible_report_path),
            "visible_surface_archive": str(visible_archive_path),
        },
        "seed_candidate_count": len(candidates),
        "skipped_candidate_count": len(skipped),
        "candidate_rows": candidates,
        "skipped_candidates": skipped,
        "observed_surface_only_seed_count": len(candidates),
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v17_observed_surface_geometry_seed_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = existing_path(
        args.object_material_surface_replay_root / "v17_object_material_surface_replay_summary.json",
        "material-surface replay summary",
    )
    summary = require_dict(load_json(summary_path), "material-surface replay summary")
    reports = [
        build_case(require_str(require_dict(raw, f"summary cases[{i}]").get("case"), "case"), args)
        for i, raw in enumerate(require_list(summary.get("cases"), "summary cases"))
    ]
    payload = {
        "method": "build_v17_observed_surface_geometry_seed",
        "status": STATUS,
        "claim": CLAIM,
        "source_material_surface_replay_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_observed_surface_geometry_seed_report.json"
                ),
                "seed_candidate_count": require_int(report.get("seed_candidate_count"), "seed_candidate_count"),
                "skipped_candidate_count": require_int(report.get("skipped_candidate_count"), "skipped_candidate_count"),
                "observed_surface_only_seed_count": require_int(
                    report.get("observed_surface_only_seed_count"),
                    "observed_surface_only_seed_count",
                ),
                "complete_geometry_seed_count": 0,
                "contact_compatible_geometry_seed_count": 0,
                "full_active_interval_geometry_seed_count": 0,
                **FALSE_READY,
            }
            for report in reports
        ],
        "seed_candidate_count": sum(require_int(report.get("seed_candidate_count"), "seed_candidate_count") for report in reports),
        "skipped_candidate_count": sum(require_int(report.get("skipped_candidate_count"), "skipped_candidate_count") for report in reports),
        "observed_surface_only_seed_count": sum(
            require_int(report.get("observed_surface_only_seed_count"), "observed_surface_only_seed_count")
            for report in reports
        ),
        "complete_geometry_seed_count": 0,
        "contact_compatible_geometry_seed_count": 0,
        "full_active_interval_geometry_seed_count": 0,
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_observed_surface_geometry_seed_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-material-pose-candidate-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_pose_candidates"),
    )
    parser.add_argument(
        "--object-material-surface-replay-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_surface_replay"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed"),
    )
    parser.add_argument("--max-eval-points", type=int, default=1024)
    parser.add_argument("--min-seed-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
