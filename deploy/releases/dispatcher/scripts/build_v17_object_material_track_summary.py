#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATUS = "v17_object_material_track_summary"
CLAIM = (
    "This artifact summarizes CoTracker material-correspondence measurements and rigid-pair diagnostics "
    "for V17 object-track datasets. It is not object-pose annotation and cannot close the V3 solver."
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


def case_dataset_summary(dataset_root: Path, case: str) -> dict[str, Any]:
    path = dataset_root / case / "v17_object_track_dataset_summary.json"
    if not path.exists():
        raise RuntimeError(f"object-track dataset summary missing: {path}")
    payload = require_dict(load_json(path), f"{path}")
    return payload


def case_track_reports(track_root: Path, case: str) -> list[Path]:
    case_dir = track_root / case
    if not case_dir.exists():
        return []
    return sorted(case_dir.glob("*/v17_object_material_track_report.json"))


def matching_rigid_report(track_report_path: Path) -> dict[str, Any] | None:
    path = track_report_path.parent / "rigid_pair_factors.json"
    if not path.exists():
        return None
    return require_dict(load_json(path), f"{path}")


def source_dataset_row(dataset: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    for i, raw in enumerate(require_list(dataset.get("objects"), "dataset.objects")):
        row = require_dict(raw, f"dataset.objects[{i}]")
        if row.get("object_id") == object_id:
            return row
    return None


def track_window_row(report_path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    track = require_dict(load_json(report_path), f"{report_path}")
    rigid = matching_rigid_report(report_path)
    object_id = require_str(track.get("object_id"), f"{report_path}.object_id")
    dataset_row = source_dataset_row(dataset, object_id)
    if dataset_row is None:
        raise RuntimeError(f"track report object {object_id} not found in object-track dataset summary")
    rigid_ready_pairs = 0
    rigid_pair_count = 0
    ready_residual = {"count": 0}
    if rigid is not None:
        rigid_ready_pairs = require_int(rigid.get("rigid_factor_ready_pairs"), f"{report_path}.rigid ready pairs")
        rigid_pair_count = require_int(rigid.get("pair_count"), f"{report_path}.rigid pair_count")
        ready_residual = require_dict(
            rigid.get("ready_pair_inlier_residual_m"), f"{report_path}.ready_pair_inlier_residual_m"
        )
    return {
        "object_id": object_id,
        "track_id": require_str(track.get("track_id"), f"{report_path}.track_id"),
        "window_id": report_path.parent.name,
        "report_path": str(report_path),
        "rigid_pair_report_path": str(report_path.parent / "rigid_pair_factors.json") if rigid is not None else None,
        "source_dataset_manifest": dataset_row.get("manifest"),
        "source_dataset_frame_count": require_int(dataset_row.get("frame_count"), "dataset frame_count"),
        "frame_count": require_int(track.get("frame_count"), f"{report_path}.frame_count"),
        "frames": require_list(track.get("frames"), f"{report_path}.frames"),
        "query_points": require_int(track.get("query_points"), f"{report_path}.query_points"),
        "all_frame_accepted_tracks": require_int(
            track.get("all_frame_accepted_tracks"), f"{report_path}.all_frame_accepted_tracks"
        ),
        "valid_frames_per_track": require_dict(
            track.get("valid_frames_per_track"), f"{report_path}.valid_frames_per_track"
        ),
        "world_step_m": require_dict(track.get("world_step_m"), f"{report_path}.world_step_m"),
        "rigid_pair_count": rigid_pair_count,
        "rigid_factor_ready_pairs": rigid_ready_pairs,
        "ready_pair_inlier_residual_m": ready_residual,
        "material_track_measurement_ready": bool(track.get("status") == "v17_object_material_track_measurement"),
        "rigid_motion_evidence_ready": rigid_ready_pairs > 0,
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }


def case_summary(track_root: Path, dataset_root: Path, case: str) -> dict[str, Any]:
    dataset = case_dataset_summary(dataset_root, case)
    dataset_objects = require_list(dataset.get("objects"), f"{case} dataset objects")
    exported_objects = [
        require_dict(row, f"{case} dataset object {i}")
        for i, row in enumerate(dataset_objects)
        if require_int(require_dict(row, f"{case} dataset object {i}").get("frame_count"), "frame_count") > 0
    ]
    windows = [track_window_row(path, dataset) for path in case_track_reports(track_root, case)]
    tracked_object_ids = sorted({require_str(row.get("object_id"), "window.object_id") for row in windows})
    ready_windows = [row for row in windows if bool(row["rigid_motion_evidence_ready"])]
    rigid_ready_pairs = sum(require_int(row.get("rigid_factor_ready_pairs"), "window rigid ready pairs") for row in windows)
    return {
        "method": "build_v17_object_material_track_summary",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "track_root": str(track_root / case),
        "object_track_dataset_summary": str(dataset_root / case / "v17_object_track_dataset_summary.json"),
        "dataset_object_count": require_int(dataset.get("object_count"), f"{case} dataset object_count"),
        "dataset_exported_object_count": require_int(
            dataset.get("exported_object_count"), f"{case} dataset exported_object_count"
        ),
        "dataset_exported_frames": require_int(
            dataset.get("total_exported_frames"), f"{case} dataset total_exported_frames"
        ),
        "material_track_window_count": len(windows),
        "material_tracked_object_count": len(tracked_object_ids),
        "material_tracked_object_ids": tracked_object_ids,
        "exported_object_ids_without_material_tracks": sorted(
            {
                require_str(row.get("object_id"), f"{case} exported object object_id")
                for row in exported_objects
            }.difference(tracked_object_ids)
        ),
        "rigid_motion_ready_window_count": len(ready_windows),
        "rigid_factor_ready_pair_count": rigid_ready_pairs,
        "windows": windows,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    measurement_summary_path = args.measurement_store_root / "v17_measurement_store_summary.json"
    summary = require_dict(load_json(measurement_summary_path), f"{measurement_summary_path}")
    cases = [
        require_str(require_dict(row, f"measurement summary case {i}").get("case"), "case")
        for i, row in enumerate(require_list(summary.get("cases"), "measurement summary cases"))
    ]
    outputs = [case_summary(args.track_root, args.object_track_dataset_root, case) for case in cases]
    for case_output in outputs:
        write_json(
            args.output_root / require_str(case_output.get("case"), "case output case") / "v17_object_material_track_summary.json",
            case_output,
        )
    payload = {
        "method": "build_v17_object_material_track_summary",
        "status": STATUS,
        "claim": CLAIM,
        "measurement_store_summary": str(measurement_summary_path),
        "track_root": str(args.track_root),
        "object_track_dataset_root": str(args.object_track_dataset_root),
        "case_count": len(outputs),
        "cases": [
            {
                "case": case["case"],
                "summary_path": str(
                    args.output_root
                    / require_str(case.get("case"), "case")
                    / "v17_object_material_track_summary.json"
                ),
                "dataset_exported_object_count": case["dataset_exported_object_count"],
                "dataset_exported_frames": case["dataset_exported_frames"],
                "material_track_window_count": case["material_track_window_count"],
                "material_tracked_object_count": case["material_tracked_object_count"],
                "rigid_motion_ready_window_count": case["rigid_motion_ready_window_count"],
                "rigid_factor_ready_pair_count": case["rigid_factor_ready_pair_count"],
                "exported_object_ids_without_material_tracks": case["exported_object_ids_without_material_tracks"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for case in outputs
        ],
        "material_track_window_count": sum(require_int(case.get("material_track_window_count"), "material window count") for case in outputs),
        "rigid_factor_ready_pair_count": sum(require_int(case.get("rigid_factor_ready_pair_count"), "rigid ready pair count") for case in outputs),
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_object_material_track_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-store-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_measurement_store"),
    )
    parser.add_argument(
        "--object-track-dataset-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_track_datasets"),
    )
    parser.add_argument(
        "--track-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
