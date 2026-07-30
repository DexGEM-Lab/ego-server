#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_visible_part_subset_archive"
CLAIM = (
    "This artifact materializes robust stable part-subset candidates as visible-surface mesh archives. It copies "
    "only observed depth-backed part surfaces, rebases face indices, and preserves provenance. It does not complete "
    "hidden geometry, fit part pose, or satisfy object-pose requirements."
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


def string_array(values: list[str]) -> np.ndarray:
    max_len = max([1] + [len(value) for value in values])
    return np.asarray(values, dtype=f"<U{max_len}")


def copy_surface_row(
    source_index: int,
    source_npz: np.lib.npyio.NpzFile,
    out_vertex_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_offsets = source_npz["vertex_offsets"]
    face_offsets = source_npz["face_offsets"]
    src_v0 = int(vertex_offsets[source_index])
    src_v1 = int(vertex_offsets[source_index + 1])
    src_f0 = int(face_offsets[source_index])
    src_f1 = int(face_offsets[source_index + 1])
    vertices = np.asarray(source_npz["vertices"][src_v0:src_v1], dtype=np.float32)
    faces = np.asarray(source_npz["faces"][src_f0:src_f1], dtype=np.int64)
    if faces.size:
        local_faces = faces - src_v0
        if int(local_faces.min()) < 0 or int(local_faces.max()) >= len(vertices):
            raise RuntimeError(f"source row {source_index} face indices are outside its vertex slice")
        rebased = local_faces + out_vertex_offset
    else:
        rebased = faces.reshape((0, 3))
    return vertices, np.asarray(rebased, dtype=np.int32)


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = args.part_model_candidates_root / case / "v18_part_model_candidates_report.json"
    surfaces_path = args.part_surfaces_root / case / "v18_part_visible_surfaces_report.json"
    candidates_report = require_dict(load_json(candidates_path), f"{case} candidates")
    surfaces_report = require_dict(load_json(surfaces_path), f"{case} surfaces")
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(surfaces_report.get("surface_rows"), "surface rows")]
    archive_path = Path(str(surfaces_report.get("archive_npz")))
    source_npz = np.load(archive_path)
    selected_indices: list[int] = []
    selected_candidate_ids: list[str] = []
    selected_object_ids: list[str] = []
    selected_part_labels: list[str] = []
    selected_frame_idx: list[int] = []
    selected_source_indices: list[int] = []
    row_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_offsets = [0]
    face_offsets = [0]
    for candidate in [require_dict(raw, "candidate") for raw in require_list(candidates_report.get("candidates"), "candidates")]:
        labels = {str(label) for label in require_list(candidate.get("part_track_labels"), "part labels")}
        object_id = str(candidate.get("object_id"))
        candidate_id = str(candidate.get("candidate_id"))
        before_rows = len(selected_indices)
        before_vertices = vertex_offsets[-1]
        before_faces = face_offsets[-1]
        for source_index, row in enumerate(surface_rows):
            if str(row.get("object_id")) != object_id:
                continue
            if str(row.get("part_track_label")) not in labels:
                continue
            out_v0 = vertex_offsets[-1]
            vertices, faces = copy_surface_row(source_index, source_npz, out_v0)
            vertices_parts.append(vertices)
            faces_parts.append(faces)
            selected_indices.append(len(selected_indices))
            selected_candidate_ids.append(candidate_id)
            selected_object_ids.append(object_id)
            part_label = str(row.get("part_track_label"))
            selected_part_labels.append(part_label)
            frame_idx = require_int(row.get("frame_idx"), "frame_idx")
            selected_frame_idx.append(frame_idx)
            selected_source_indices.append(source_index)
            vertex_offsets.append(vertex_offsets[-1] + int(vertices.shape[0]))
            face_offsets.append(face_offsets[-1] + int(faces.shape[0]))
            row_records.append(
                {
                    "candidate_id": candidate_id,
                    "object_id": object_id,
                    "part_track_label": part_label,
                    "frame_idx": frame_idx,
                    "source_surface_row_index": source_index,
                    "npz_row_index": len(selected_indices) - 1,
                    "npz_vertex_slice": [out_v0, vertex_offsets[-1]],
                    "npz_face_slice": [face_offsets[-2], face_offsets[-1]],
                    "vertices": int(vertices.shape[0]),
                    "faces": int(faces.shape[0]),
                    "geometry_claim": "visible_part_subset_surface_only",
                    "hidden_geometry_reconstructed": False,
                    "part_pose_ready": False,
                    "object_pose_requirement_met": False,
                }
            )
        after_rows = len(selected_indices)
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "object_id": object_id,
                "part_track_labels": sorted(labels),
                "archive_row_count": after_rows - before_rows,
                "unique_frame_count": len(set(selected_frame_idx[before_rows:after_rows])),
                "frame_min": min(selected_frame_idx[before_rows:after_rows]) if after_rows > before_rows else None,
                "frame_max": max(selected_frame_idx[before_rows:after_rows]) if after_rows > before_rows else None,
                "vertex_count": vertex_offsets[-1] - before_vertices,
                "face_count": face_offsets[-1] - before_faces,
                "model_scope": "visible_part_subset_surface_only",
                "hidden_geometry_reconstructed": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
            }
        )
    out_vertices = np.concatenate(vertices_parts, axis=0).astype(np.float32) if vertices_parts else np.zeros((0, 3), dtype=np.float32)
    out_faces = np.concatenate(faces_parts, axis=0).astype(np.int32) if faces_parts else np.zeros((0, 3), dtype=np.int32)
    if out_faces.size:
        if int(out_faces.min()) < 0 or int(out_faces.max()) >= int(out_vertices.shape[0]):
            raise RuntimeError("rebased output faces exceed output vertex array")
    metadata = {
        "status": STATUS,
        "claim": CLAIM,
        "archive_format": "v18_visible_part_subset_rows",
        "face_indices": "global_vertex_indices",
        "local_face_conversion": "subtract npz_vertex_slice[0] for a row-local mesh",
        "hidden_geometry_reconstructed": False,
        "part_pose_ready": False,
        "object_pose_requirement_met": False,
    }
    output_npz = args.output_root / case / "v18_visible_part_subset_camera.npz"
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        candidate_id=string_array(selected_candidate_ids),
        frame_idx=np.asarray(selected_frame_idx, dtype=np.int32),
        object_id=string_array(selected_object_ids),
        part_track_label=string_array(selected_part_labels),
        source_surface_row_index=np.asarray(selected_source_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices=out_vertices,
        faces=out_faces,
        v18_archive_metadata_json=np.asarray(json.dumps(metadata)),
    )
    label_counts = Counter(selected_part_labels)
    nonempty_ready = bool(candidate_records and row_records and int(out_vertices.shape[0]) > 0 and int(out_faces.shape[0]) > 0)
    report = {
        "method": "build_v18_visible_part_subset_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_part_model_candidates": str(candidates_path), "v18_part_visible_surfaces": str(surfaces_path), "source_archive_npz": str(archive_path)},
        "archive_npz": str(output_npz),
        "candidate_count": len(candidate_records),
        "archive_row_count": len(row_records),
        "unique_frame_count": len(set(selected_frame_idx)),
        "total_vertices": int(out_vertices.shape[0]),
        "total_faces": int(out_faces.shape[0]),
        "rows_by_part_track": dict(sorted(label_counts.items())),
        "candidate_records": candidate_records,
        "row_records": row_records,
        "visible_part_subset_archive_file_written": True,
        "visible_part_subset_archive_ready": nonempty_ready,
        "visible_part_subset_archive_ready_scope": "nonempty_candidate_archive_only",
        "hidden_geometry_completion_candidate_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "hidden_geometry_reconstructed": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_visible_part_subset_archive_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    summary = {
        "method": "build_v18_visible_part_subset_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "candidate_count": sum(require_int(report.get("candidate_count"), "candidate_count") for report in reports),
        "archive_row_count": sum(require_int(report.get("archive_row_count"), "archive_row_count") for report in reports),
        "unique_frame_count_sum_by_case": sum(require_int(report.get("unique_frame_count"), "unique_frame_count") for report in reports),
        "total_vertices": sum(require_int(report.get("total_vertices"), "total_vertices") for report in reports),
        "total_faces": sum(require_int(report.get("total_faces"), "total_faces") for report in reports),
        "visible_part_subset_archive_file_written_all_cases": all(bool(report.get("visible_part_subset_archive_file_written")) for report in reports),
        "visible_part_subset_archive_ready": any(bool(report.get("visible_part_subset_archive_ready")) for report in reports),
        "visible_part_subset_archive_ready_scope": "one_or_more_nonempty_candidate_archives",
        "visible_part_subset_archive_ready_count": sum(1 for report in reports if bool(report.get("visible_part_subset_archive_ready"))),
        "all_cases_visible_part_subset_archive_ready": all(bool(report.get("visible_part_subset_archive_ready")) for report in reports),
        "hidden_geometry_completion_candidate_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "hidden_geometry_reconstructed": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_visible_part_subset_archive_report.json"),
                "archive_npz": report["archive_npz"],
                "candidate_count": report["candidate_count"],
                "archive_row_count": report["archive_row_count"],
                "total_vertices": report["total_vertices"],
                "total_faces": report["total_faces"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_visible_part_subset_archive_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-model-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_model_candidates"))
    parser.add_argument("--part-surfaces-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_visible_surfaces"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_part_subset_archive"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
