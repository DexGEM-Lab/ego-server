#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
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

STATUS = "v18_visible_geometry_archive"
CLAIM = (
    "This artifact stages depth-backed visible object surfaces as V18 geometry evidence. It preserves NPZ "
    "surface meshes and per-frame offsets, but it does not reconstruct hidden geometry, canonical object meshes, "
    "or complete object poses."
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


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def fast_motion_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("object_id"), "fast motion object_id"): row
        for row in [require_dict(raw, "fast motion row") for raw in require_list(report.get("object_rows"), "fast motion object_rows")]
    }


def bounded_object_state_index(solution: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(solution.get("frames"), "solution frames"):
        frame = require_dict(raw_frame, "solution frame")
        frame_idx = require_int(frame.get("frame_idx"), "solution frame_idx")
        for raw_obj in require_list(frame.get("objects"), "solution objects"):
            obj = require_dict(raw_obj, "solution object")
            object_id = require_str(obj.get("object_id"), "solution object_id")
            out[(frame_idx, object_id)] = obj
    return out


def archive_arrays(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    required = ["frame_idx", "object_id", "vertex_offsets", "face_offsets", "vertices", "faces"]
    for key in required:
        if key not in data.files:
            raise RuntimeError(f"archive {path} missing key {key}")
    frame_idx = data["frame_idx"]
    object_id = data["object_id"]
    vertex_offsets = data["vertex_offsets"]
    face_offsets = data["face_offsets"]
    vertices = data["vertices"]
    faces = data["faces"]
    if frame_idx.ndim != 1 or object_id.ndim != 1:
        raise RuntimeError("frame_idx/object_id arrays must be 1D")
    row_count = int(frame_idx.shape[0])
    if int(object_id.shape[0]) != row_count:
        raise RuntimeError("object_id row count mismatch")
    if int(vertex_offsets.shape[0]) != row_count + 1 or int(face_offsets.shape[0]) != row_count + 1:
        raise RuntimeError("offset arrays must have row_count+1 entries")
    if vertices.ndim != 2 or int(vertices.shape[1]) != 3:
        raise RuntimeError("vertices must be Nx3")
    if faces.ndim != 2 or int(faces.shape[1]) != 3:
        raise RuntimeError("faces must be Mx3")
    if int(vertex_offsets[-1]) != int(vertices.shape[0]):
        raise RuntimeError("vertex offset terminal does not match vertices length")
    if int(face_offsets[-1]) != int(faces.shape[0]):
        raise RuntimeError("face offset terminal does not match faces length")
    return {
        "frame_idx": frame_idx,
        "object_id": object_id,
        "vertex_offsets": vertex_offsets,
        "face_offsets": face_offsets,
        "vertices": vertices,
        "faces": faces,
        "row_count": row_count,
    }


def object_status(surface_frames: int, rejected_frames: int, fast_motion_state: str | None) -> str:
    if surface_frames > 0 and fast_motion_state == "partial_rigid_visible_surface_motion_supported":
        return "partial_rigid_visible_surface_archive_ready_not_complete_pose"
    if surface_frames > 0:
        return "visible_surface_archive_ready_hidden_geometry_unresolved"
    if rejected_frames > 0:
        return "visible_mask_without_accepted_surface"
    return "no_visible_geometry_evidence"


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    visible_report_path = existing(args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json", f"{case} V17 visible surface report")
    visible_archive_path = existing(args.visible_surface_root / case / "multi_object_visible_surfaces_world.npz", f"{case} V17 visible surface archive")
    fast_motion_path = existing(args.fast_motion_root / case / "v18_fast_motion_state_report.json", f"{case} V18 fast motion")
    solution_path = existing(args.solution_root / case / "v18_bounded_state_solution.json", f"{case} V18 bounded solution")
    report = require_dict(load_json(visible_report_path), f"{case} visible report")
    fast_motion = require_dict(load_json(fast_motion_path), f"{case} fast motion")
    solution = require_dict(load_json(solution_path), f"{case} bounded solution")
    arrays = archive_arrays(visible_archive_path)
    surface_rows = [require_dict(raw, "surface row") for raw in require_list(report.get("surface_rows"), "surface_rows")]
    rejected_rows = [require_dict(raw, "rejected row") for raw in require_list(report.get("rejected_rows"), "rejected_rows")]
    if len(surface_rows) != int(arrays["row_count"]):
        raise RuntimeError(f"{case}: surface_rows length does not match archive row count")
    motion_index = fast_motion_by_object(fast_motion)
    solution_index = bounded_object_state_index(solution)
    output_case_dir = args.output_root / case
    output_case_dir.mkdir(parents=True, exist_ok=True)
    output_archive_path = output_case_dir / "v18_visible_surfaces_world.npz"
    shutil.copy2(visible_archive_path, output_archive_path)
    object_surface_counts: Counter[str] = Counter()
    object_rejected_counts: Counter[str] = Counter()
    object_vertices: Counter[str] = Counter()
    object_faces: Counter[str] = Counter()
    archive_rows: list[dict[str, Any]] = []
    for i, row in enumerate(surface_rows):
        frame_idx = require_int(row.get("frame_idx"), "surface frame_idx")
        object_id = require_str(row.get("object_id"), "surface object_id")
        archive_frame = int(arrays["frame_idx"][i])
        archive_object = str(arrays["object_id"][i])
        if frame_idx != archive_frame or object_id != archive_object:
            raise RuntimeError(f"{case}: archive/report row mismatch at row {i}")
        vertex_start = int(arrays["vertex_offsets"][i])
        vertex_end = int(arrays["vertex_offsets"][i + 1])
        face_start = int(arrays["face_offsets"][i])
        face_end = int(arrays["face_offsets"][i + 1])
        vertex_count = vertex_end - vertex_start
        face_count = face_end - face_start
        object_surface_counts[object_id] += 1
        object_vertices[object_id] += vertex_count
        object_faces[object_id] += face_count
        bounded_obj = solution_index.get((frame_idx, object_id), {})
        archive_rows.append(
            {
                "archive_row_index": i,
                "frame_idx": frame_idx,
                "object_id": object_id,
                "measurement_type": row.get("measurement_type"),
                "coordinate_frame": row.get("coordinate_frame"),
                "npz_vertex_slice": [vertex_start, vertex_end],
                "npz_face_slice": [face_start, face_end],
                "face_index_convention": "global_vertex_indices",
                "local_face_index_conversion": "subtract npz_vertex_slice[0] from each face index after slicing",
                "vertex_count": vertex_count,
                "face_count": face_count,
                "center_world_m": row.get("center_world_m"),
                "bbox_world_min_m": row.get("bbox_world_min_m"),
                "bbox_world_max_m": row.get("bbox_world_max_m"),
                "world_extent_m": row.get("world_extent_m"),
                "mask_path": row.get("mask_path"),
                "bounded_object_solution_state": bounded_obj.get("solution_state"),
                "geometry_state": "visible_surface_only_hidden_geometry_unresolved",
                "pose_state": "no_complete_object_pose_variable",
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )
    for row in rejected_rows:
        object_rejected_counts[require_str(row.get("object_id"), "rejected object_id")] += 1
    object_rows: list[dict[str, Any]] = []
    object_ids = sorted(set(object_surface_counts) | set(object_rejected_counts) | set(motion_index))
    for object_id in object_ids:
        motion = motion_index.get(object_id, {})
        surface_count = int(object_surface_counts.get(object_id, 0))
        rejected_count = int(object_rejected_counts.get(object_id, 0))
        fast_motion_state = motion.get("fast_motion_state") if isinstance(motion.get("fast_motion_state"), str) else None
        status = object_status(surface_count, rejected_count, fast_motion_state)
        object_rows.append(
            {
                "object_id": object_id,
                "track_id": motion.get("track_id"),
                "name": motion.get("name"),
                "model_physical_state_type": motion.get("model_physical_state_type"),
                "fast_motion_state": fast_motion_state,
                "v18_visible_geometry_status": status,
                "surface_frame_count": surface_count,
                "rejected_visible_frame_count": rejected_count,
                "surface_vertex_count": int(object_vertices.get(object_id, 0)),
                "surface_face_count": int(object_faces.get(object_id, 0)),
                "geometry_claim": "depth_backed_visible_surface_only" if surface_count > 0 else "no_accepted_visible_surface",
                "hidden_geometry_state": "hidden_geometry_unresolved",
                "canonical_mesh_ready": False,
                "complete_object_pose_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )
    status_counts = Counter(require_str(row.get("v18_visible_geometry_status"), "visible geometry status") for row in object_rows)
    geometry_claim_counts = Counter(require_str(row.get("geometry_claim"), "geometry claim") for row in object_rows)
    archive_metadata = {
        "method": "build_v18_visible_geometry_archive",
        "status": STATUS,
        "case": case,
        "source_visible_surface_archive": str(visible_archive_path),
        "source_visible_surface_report": str(visible_report_path),
        "row_count": int(arrays["row_count"]),
        "total_vertices": int(arrays["vertices"].shape[0]),
        "total_faces": int(arrays["faces"].shape[0]),
        "geometry_claim": "visible_surface_only_hidden_geometry_unresolved",
        "face_index_convention": "global_vertex_indices",
        "consumer_note": "Per-row face slices retain global indices into the archive vertices array. Convert to local row vertices by subtracting npz_vertex_slice[0].",
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
    }
    write_json(output_case_dir / "v18_visible_geometry_archive_metadata.json", archive_metadata)
    case_payload = {
        "method": "build_v18_visible_geometry_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v17_visible_surface_report": str(visible_report_path),
            "v17_visible_surface_archive": str(visible_archive_path),
            "v18_fast_motion_state": str(fast_motion_path),
            "v18_bounded_state_solution": str(solution_path),
        },
        "archive_npz": str(output_archive_path),
        "archive_metadata": str(output_case_dir / "v18_visible_geometry_archive_metadata.json"),
        "frame_count": require_int(report.get("frame_count"), "frame_count"),
        "surface_frame_rows": len(archive_rows),
        "rejected_visible_object_frame_rows": len(rejected_rows),
        "object_count": len(object_rows),
        "total_vertices": int(arrays["vertices"].shape[0]),
        "total_faces": int(arrays["faces"].shape[0]),
        "v18_visible_geometry_status_counts": dict(sorted(status_counts.items())),
        "geometry_claim_counts": dict(sorted(geometry_claim_counts.items())),
        "object_rows": object_rows,
        "surface_archive_rows": archive_rows,
        "rejection_reason_counts": report.get("rejection_reason_counts"),
        "visible_geometry_archive_ready": True,
        "hidden_geometry_reconstructed": False,
        "canonical_mesh_ready": False,
        "complete_object_pose_ready": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "face_index_convention": "global_vertex_indices",
        "consumer_note": "Per-row face slices retain global indices into the archive vertices array. Convert to local row vertices by subtracting npz_vertex_slice[0].",
        **FALSE_READY,
    }
    write_json(output_case_dir / "v18_visible_geometry_archive_report.json", case_payload)
    return case_payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    status_counts: Counter[str] = Counter()
    geometry_claim_counts: Counter[str] = Counter()
    for report in reports:
        status_counts.update(report["v18_visible_geometry_status_counts"])
        geometry_claim_counts.update(report["geometry_claim_counts"])
    summary = {
        "method": "build_v18_visible_geometry_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "surface_frame_rows": sum(require_int(report.get("surface_frame_rows"), "surface rows") for report in reports),
        "rejected_visible_object_frame_rows": sum(require_int(report.get("rejected_visible_object_frame_rows"), "rejected rows") for report in reports),
        "total_vertices": sum(require_int(report.get("total_vertices"), "vertices") for report in reports),
        "total_faces": sum(require_int(report.get("total_faces"), "faces") for report in reports),
        "v18_visible_geometry_status_counts": dict(sorted(status_counts.items())),
        "geometry_claim_counts": dict(sorted(geometry_claim_counts.items())),
        "visible_geometry_archive_ready": True,
        "hidden_geometry_reconstructed": False,
        "canonical_mesh_ready": False,
        "complete_object_pose_ready": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_visible_geometry_archive_report.json"),
                "archive_npz": report["archive_npz"],
                "surface_frame_rows": report["surface_frame_rows"],
                "rejected_visible_object_frame_rows": report["rejected_visible_object_frame_rows"],
                "total_vertices": report["total_vertices"],
                "total_faces": report["total_faces"],
                "v18_visible_geometry_status_counts": report["v18_visible_geometry_status_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_visible_geometry_archive_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-surface-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"))
    parser.add_argument("--fast-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_fast_motion_state"))
    parser.add_argument("--solution-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visible_geometry_archive"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
