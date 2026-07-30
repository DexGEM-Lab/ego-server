#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapt_bundlesdf_to_mesh_archive_v3 import load_obj_mesh, transform_points
from build_v17_geometry_reconstruction_results import (
    bundlesdf_normalization,
    candidate_mesh_path,
)
from build_v17_hand_metric_depth_state import (
    FALSE_READY,
    load_json,
    require_dict,
    require_int,
    require_list,
    require_str,
    summarize,
    write_json,
)
from run_v16_full_pipeline import save_mesh_archive


STATUS = "v17_multi_object_world_mesh_archive_qc"
CLAIM = (
    "This artifact converts accepted full-interval BundleSDF reconstructions into one per-frame "
    "multi-object world mesh archive per case. Per frame, every accepted object with a pose file is "
    "transformed by T_world_camera @ ob_in_cam and concatenated into a single frame mesh. It is the "
    "object stream for V17 full-duration rendering; frames outside every object's reconstructed "
    "interval carry no object mesh, and acceptance comes solely from the reconstruction-results "
    "evaluator. It does not change V17 readiness."
)


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    out = float(value)
    if not np.isfinite(out):
        raise RuntimeError(f"{label} must be finite")
    return out


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    return path


def annotation_camera_transforms(annotations_path: Path) -> dict[int, np.ndarray]:
    payload = require_dict(load_json(annotations_path), "annotations payload")
    out: dict[int, np.ndarray] = {}
    for i, raw in enumerate(require_list(payload.get("frames"), "annotation frames")):
        frame = require_dict(raw, f"annotation frames[{i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"annotation frames[{i}].frame_idx")
        camera = require_dict(frame.get("camera"), f"frame {frame_idx} camera")
        transform = np.asarray(camera.get("T_world_camera_metric"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"frame {frame_idx} T_world_camera_metric must be a finite 4x4")
        out[frame_idx] = transform
    if not out:
        raise RuntimeError(f"{annotations_path} contains no camera frames")
    return out


def accepted_result_rows(results_report: dict[str, Any], case: str) -> list[dict[str, Any]]:
    rows = []
    for i, raw in enumerate(require_list(results_report.get("jobs"), f"{case} result jobs")):
        row = require_dict(raw, f"{case} result jobs[{i}]")
        if row.get("accepted_reconstruction_result") is True:
            rows.append(row)
    return rows


def object_world_meshes(
    *,
    case: str,
    result_row: dict[str, Any],
    cameras: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    job_id = require_str(result_row.get("job_id"), "job_id")
    object_id = require_str(result_row.get("object_id"), "object_id")
    job_path = existing_path(Path(require_str(result_row.get("source_job_path"), "source_job_path")), f"{job_id} job manifest")
    job = require_dict(load_json(job_path), f"{job_id} job manifest")
    output_dir = existing_path(
        Path(require_str(result_row.get("bundlesdf_output_dir"), "bundlesdf_output_dir")),
        f"{job_id} BundleSDF output",
    )
    mesh_path = candidate_mesh_path(output_dir)
    if mesh_path is None:
        raise RuntimeError(f"{job_id} accepted result has no mesh file")
    vertices, faces = load_obj_mesh(mesh_path)
    if mesh_path.name == "mesh_cleaned.obj":
        _, sc_factor, translation = bundlesdf_normalization(output_dir)
        vertices = vertices / sc_factor - translation.reshape(1, 3)
    frames = [require_dict(raw, f"{job_id} job frame") for raw in require_list(job.get("frames"), f"{job_id} job frames")]
    per_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    missing_pose = 0
    missing_camera = 0
    for frame in frames:
        frame_idx = require_int(frame.get("frame_idx"), "job frame_idx")
        pose_index = require_int(frame.get("index"), "job frame index")
        pose_path = output_dir / "ob_in_cam" / f"{pose_index:06d}.txt"
        if not pose_path.exists():
            missing_pose += 1
            continue
        camera = cameras.get(frame_idx)
        if camera is None:
            missing_camera += 1
            continue
        ob_in_cam = np.loadtxt(pose_path).astype(np.float64)
        if ob_in_cam.shape != (4, 4) or not np.isfinite(ob_in_cam).all():
            raise RuntimeError(f"{pose_path} must be a finite 4x4 pose")
        per_frame[frame_idx] = (transform_points(vertices, camera @ ob_in_cam), faces)
    if not per_frame:
        raise RuntimeError(f"{job_id} produced no world-frame meshes")
    extents = np.asarray(
        [mesh.max(axis=0) - mesh.min(axis=0) for mesh, _ in per_frame.values()],
        dtype=np.float64,
    )
    return {
        "job_id": job_id,
        "object_id": object_id,
        "mesh_path": str(mesh_path),
        "canonical_vertices": int(len(vertices)),
        "canonical_faces": int(len(faces)),
        "world_frame_count": len(per_frame),
        "first_world_frame": min(per_frame),
        "last_world_frame": max(per_frame),
        "missing_pose_frames": missing_pose,
        "missing_camera_frames": missing_camera,
        "world_extent_max_m": summarize(extents.max(axis=1).astype(float).tolist()),
        "_per_frame": per_frame,
    }


def case_archive(case: str, args: argparse.Namespace) -> dict[str, Any]:
    results_path = existing_path(
        args.geometry_reconstruction_results_root / case / "v17_geometry_reconstruction_results_report.json",
        f"{case} reconstruction results report",
    )
    results_report = require_dict(load_json(results_path), f"{case} reconstruction results report")
    annotations_path = existing_path(
        Path(
            require_str(
                require_dict(
                    load_json(args.v16_root / case / "v16_full_pipeline_manifest.json"),
                    f"{case} v16 manifest",
                ).get("annotations"),
                "v16 annotations path",
            )
        ),
        f"{case} v16 annotations",
    )
    cameras = annotation_camera_transforms(annotations_path)
    accepted = accepted_result_rows(results_report, case)
    objects = [
        object_world_meshes(case=case, result_row=row, cameras=cameras, args=args)
        for row in accepted
    ]
    merged: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for obj in objects:
        for frame_idx, (verts, faces) in obj["_per_frame"].items():
            merged.setdefault(frame_idx, ([], []))
            merged[frame_idx][0].append(verts)
            merged[frame_idx][1].append(faces)
    frame_indices = sorted(merged)
    vertices_per_frame: list[np.ndarray] = []
    faces_per_frame: list[np.ndarray] = []
    for frame_idx in frame_indices:
        vert_lists, face_lists = merged[frame_idx]
        offset = 0
        shifted_faces = []
        for verts, faces in zip(vert_lists, face_lists):
            shifted_faces.append(faces + offset)
            offset += len(verts)
        vertices_per_frame.append(np.vstack(vert_lists))
        faces_per_frame.append(np.vstack(shifted_faces))
    case_dir = args.output_root / case
    archive_path = case_dir / "v17_multi_object_world_meshes.npz"
    if frame_indices:
        case_dir.mkdir(parents=True, exist_ok=True)
        save_mesh_archive(archive_path, frame_indices, vertices_per_frame, faces_per_frame)
    report = {
        "method": "build_v17_multi_object_world_mesh_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "geometry_reconstruction_results_report": str(results_path),
            "v16_annotations": str(annotations_path),
        },
        "accepted_reconstruction_count": len(accepted),
        "archived_object_count": len(objects),
        "archive_path": str(archive_path) if frame_indices else None,
        "archive_frame_count": len(frame_indices),
        "first_archive_frame": frame_indices[0] if frame_indices else None,
        "last_archive_frame": frame_indices[-1] if frame_indices else None,
        "annotation_frame_count": len(cameras),
        "archive_coverage_fraction_of_annotation": (
            float(len(frame_indices) / len(cameras)) if cameras else None
        ),
        "objects": [
            {key: value for key, value in obj.items() if not key.startswith("_")} for obj in objects
        ],
        **FALSE_READY,
    }
    write_json(case_dir / "v17_multi_object_world_mesh_archive_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    cases = [case_archive(case, args) for case in args.cases]
    summary = {
        "method": "build_v17_multi_object_world_mesh_archive",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(cases),
        "accepted_reconstruction_count": sum(
            require_int(case.get("accepted_reconstruction_count"), "case accepted count") for case in cases
        ),
        "archived_object_count": sum(
            require_int(case.get("archived_object_count"), "case archived count") for case in cases
        ),
        "cases": [
            {
                "case": require_str(case.get("case"), "case"),
                "accepted_reconstruction_count": case["accepted_reconstruction_count"],
                "archived_object_count": case["archived_object_count"],
                "archive_frame_count": case["archive_frame_count"],
                "annotation_frame_count": case["annotation_frame_count"],
                "archive_coverage_fraction_of_annotation": case["archive_coverage_fraction_of_annotation"],
                **FALSE_READY,
            }
            for case in cases
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v17_multi_object_world_mesh_archive_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-reconstruction-results-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_geometry_reconstruction_results_full_interval"),
    )
    parser.add_argument(
        "--v16-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_world_mesh_archive"),
    )
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
