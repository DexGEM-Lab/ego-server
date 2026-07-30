#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_object_material_motion_state_qc"
CLAIM = (
    "This artifact tests whether local material-track rigid factors compose into window-level object "
    "motion candidates. It does not create full-timeline object pose variables and cannot close the V3 solver."
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


def ready_pair_segments(pair_rows: list[Any]) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for i, raw in enumerate(pair_rows):
        row = require_dict(raw, f"pair_rows[{i}]")
        if row.get("rigid_factor_ready") is not True:
            if current:
                segments.append(current)
                current = []
            continue
        if current and require_int(current[-1].get("target_frame"), "target_frame") != require_int(row.get("source_frame"), "source_frame"):
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def compose_pair_chain(segment: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    rot_total = np.eye(3, dtype=np.float64)
    trans_total = np.zeros((3,), dtype=np.float64)
    chain: list[tuple[np.ndarray, np.ndarray]] = [(rot_total.copy(), trans_total.copy())]
    for i, row in enumerate(segment):
        rot = np.asarray(row.get("rotation"), dtype=np.float64)
        trans = np.asarray(row.get("translation_m"), dtype=np.float64)
        if rot.shape != (3, 3) or trans.shape != (3,) or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
            raise RuntimeError(f"invalid transform in segment row {i}")
        trans_total = trans_total @ rot + trans
        rot_total = rot_total @ rot
        chain.append((rot_total.copy(), trans_total.copy()))
    return rot_total, trans_total, chain


def transform(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return points @ rot + trans


def segment_state(
    segment: list[dict[str, Any]],
    frame_idx: np.ndarray,
    frame_to_i: dict[int, int],
    accepted: np.ndarray,
    world: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_frame = require_int(segment[0].get("source_frame"), "segment source_frame")
    end_frame = require_int(segment[-1].get("target_frame"), "segment target_frame")
    frame_positions = [frame_to_i[start_frame]]
    for row_i, row in enumerate(segment):
        source_frame = require_int(row.get("source_frame"), f"segment[{row_i}].source_frame")
        target_frame = require_int(row.get("target_frame"), f"segment[{row_i}].target_frame")
        if source_frame not in frame_to_i or target_frame not in frame_to_i:
            raise RuntimeError(f"segment row references frame absent from track archive: {source_frame}->{target_frame}")
        source_i = frame_to_i[source_frame]
        target_i = frame_to_i[target_frame]
        if source_i != frame_positions[-1]:
            raise RuntimeError(f"material-motion segment is not a composed chain at frame {source_frame}")
        if target_i <= source_i:
            raise RuntimeError(f"material-motion segment is not forward in track archive at frame {source_frame}")
        frame_positions.append(target_i)
    start_i = frame_positions[0]
    end_i = frame_positions[-1]
    rot_total, trans_total, chain = compose_pair_chain(segment)
    endpoint_keep = (
        accepted[start_i]
        & accepted[end_i]
        & np.all(np.isfinite(world[start_i]), axis=1)
        & np.all(np.isfinite(world[end_i]), axis=1)
    )
    all_segment_keep = np.all(accepted[frame_positions], axis=0) & np.all(
        np.all(np.isfinite(world[frame_positions]), axis=2),
        axis=0,
    )
    endpoint_residual = np.array([], dtype=np.float64)
    if np.any(endpoint_keep):
        predicted = transform(world[start_i, endpoint_keep], rot_total, trans_total)
        endpoint_residual = np.linalg.norm(predicted - world[end_i, endpoint_keep], axis=1)

    all_segment_residuals: list[float] = []
    endpoint_residual_by_frame: list[dict[str, Any]] = []
    for frame_i, (rot, trans) in zip(frame_positions, chain):
        frame = int(frame_idx[frame_i])
        endpoint_frame_keep = (
            accepted[start_i]
            & accepted[frame_i]
            & np.all(np.isfinite(world[start_i]), axis=1)
            & np.all(np.isfinite(world[frame_i]), axis=1)
        )
        frame_residual = np.array([], dtype=np.float64)
        if np.any(endpoint_frame_keep):
            predicted = transform(world[start_i, endpoint_frame_keep], rot, trans)
            frame_residual = np.linalg.norm(predicted - world[frame_i, endpoint_frame_keep], axis=1)
        if np.any(all_segment_keep):
            predicted_all = transform(world[start_i, all_segment_keep], rot, trans)
            residual_all = np.linalg.norm(predicted_all - world[frame_i, all_segment_keep], axis=1)
            all_segment_residuals.extend(residual_all.astype(float).tolist())
        endpoint_residual_by_frame.append(
            {
                "frame_idx": int(frame),
                "track_count": int(np.count_nonzero(endpoint_frame_keep)),
                "residual_m": summarize(frame_residual),
            }
        )

    endpoint_summary = summarize(endpoint_residual)
    all_segment_summary = summarize(np.asarray(all_segment_residuals, dtype=np.float64))
    endpoint_p95 = float(endpoint_summary.get("p95", float("inf")))
    all_segment_p95 = float(all_segment_summary.get("p95", float("inf")))
    checks = {
        "min_segment_pairs_met": bool(len(segment) >= int(args.min_segment_pairs)),
        "min_endpoint_tracks_met": bool(np.count_nonzero(endpoint_keep) >= int(args.min_endpoint_tracks)),
        "min_all_segment_tracks_met": bool(np.count_nonzero(all_segment_keep) >= int(args.min_all_segment_tracks)),
        "endpoint_chain_p95_met": bool(endpoint_p95 <= float(args.accept_chain_p95_m)),
        "all_segment_chain_p95_met": bool(all_segment_p95 <= float(args.accept_all_segment_p95_m)),
    }
    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "pair_count": len(segment),
        "track_archive_frame_positions": [int(i) for i in frame_positions],
        "endpoint_track_count": int(np.count_nonzero(endpoint_keep)),
        "all_segment_track_count": int(np.count_nonzero(all_segment_keep)),
        "endpoint_chain_residual_m": endpoint_summary,
        "all_segment_chain_residual_m": all_segment_summary,
        "endpoint_residual_by_frame": endpoint_residual_by_frame,
        "readiness_checks": checks,
        "window_rigid_motion_candidate": bool(all(checks.values())),
        "persistent_window_motion_candidate": bool(all(checks.values())),
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }


def window_state(window: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(require_str(window.get("report_path"), "window.report_path"))
    track_report = require_dict(load_json(report_path), f"{report_path}")
    npz_path = Path(require_str(require_dict(track_report.get("outputs"), "track outputs").get("tracks_npz"), "track outputs.tracks_npz"))
    data = np.load(npz_path)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    accepted = np.asarray(data["accepted"], dtype=bool)
    world = np.asarray(data["world_xyz"], dtype=np.float64)
    if accepted.shape[:2] != world.shape[:2] or world.shape[2] != 3:
        raise RuntimeError(f"track archive has inconsistent accepted/world arrays: {npz_path}")
    frame_to_i = {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())}
    rigid_pair_count = require_int(window.get("rigid_pair_count"), "window.rigid_pair_count")
    rigid_ready_count = require_int(window.get("rigid_factor_ready_pairs"), "window.rigid_factor_ready_pairs")
    rigid_path_raw = window.get("rigid_pair_report_path")
    if rigid_path_raw is None:
        if rigid_pair_count != 0 or rigid_ready_count != 0:
            raise RuntimeError(
                f"window {require_str(window.get('window_id'), 'window.window_id')} has rigid evidence without a rigid-pair report"
            )
        pair_rows: list[Any] = []
    else:
        rigid_path = Path(require_str(rigid_path_raw, "window.rigid_pair_report_path"))
        rigid_report = require_dict(load_json(rigid_path), f"{rigid_path}")
        pair_rows = require_list(rigid_report.get("pair_rows"), "rigid pair_rows")
        if len(pair_rows) != rigid_pair_count:
            raise RuntimeError(
                f"window {require_str(window.get('window_id'), 'window.window_id')} rigid pair count disagrees with report"
            )
        if sum(1 for row in pair_rows if require_dict(row, "pair row").get("rigid_factor_ready") is True) != rigid_ready_count:
            raise RuntimeError(
                f"window {require_str(window.get('window_id'), 'window.window_id')} ready pair count disagrees with report"
            )
    segments = [
        segment_state(segment, frame_idx, frame_to_i, accepted, world, args)
        for segment in ready_pair_segments(pair_rows)
    ]
    candidate_segments = [row for row in segments if row["window_rigid_motion_candidate"]]
    has_local_motion = bool(segments)
    motion_state = (
        "persistent_window_motion_candidate"
        if candidate_segments
        else ("local_adjacent_material_motion_only" if has_local_motion else "no_ready_material_motion")
    )
    return {
        "object_id": require_str(window.get("object_id"), "window.object_id"),
        "track_id": require_str(window.get("track_id"), "window.track_id"),
        "window_id": require_str(window.get("window_id"), "window.window_id"),
        "frame_count": require_int(window.get("frame_count"), "window.frame_count"),
        "query_points": require_int(window.get("query_points"), "window.query_points"),
        "all_frame_accepted_tracks": require_int(window.get("all_frame_accepted_tracks"), "window.all_frame_accepted_tracks"),
        "rigid_factor_ready_pairs": require_int(window.get("rigid_factor_ready_pairs"), "window.rigid_factor_ready_pairs"),
        "ready_segment_count": len(segments),
        "candidate_segment_count": len(candidate_segments),
        "max_ready_segment_pairs": max((require_int(row.get("pair_count"), "segment pair_count") for row in segments), default=0),
        "max_candidate_segment_pairs": max((require_int(row.get("pair_count"), "candidate pair_count") for row in candidate_segments), default=0),
        "window_rigid_motion_candidate": bool(candidate_segments),
        "persistent_window_motion_candidate": bool(candidate_segments),
        "local_adjacent_material_motion": has_local_motion,
        "material_motion_state": motion_state,
        "segments": segments,
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }


def case_state(summary_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = require_dict(load_json(summary_path), f"{summary_path}")
    windows = [window_state(require_dict(raw, f"windows[{i}]"), args) for i, raw in enumerate(require_list(summary.get("windows"), "windows"))]
    candidate_windows = [row for row in windows if row["window_rigid_motion_candidate"]]
    local_motion_windows = [row for row in windows if row["local_adjacent_material_motion"]]
    no_ready_windows = [row for row in windows if row["material_motion_state"] == "no_ready_material_motion"]
    return {
        "method": "build_v17_object_material_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": require_str(summary.get("case"), "case summary case"),
        "source_material_track_summary": str(summary_path),
        "material_track_window_count": len(windows),
        "material_tracked_object_count": require_int(summary.get("material_tracked_object_count"), "material tracked object count"),
        "rigid_factor_ready_pair_count": require_int(summary.get("rigid_factor_ready_pair_count"), "rigid ready pair count"),
        "window_rigid_motion_candidate_count": len(candidate_windows),
        "persistent_window_motion_candidate_count": len(candidate_windows),
        "local_adjacent_material_motion_window_count": len(local_motion_windows),
        "noncandidate_local_adjacent_material_motion_window_count": len(local_motion_windows) - len(candidate_windows),
        "no_ready_material_motion_window_count": len(no_ready_windows),
        "candidate_window_ids": [require_str(row.get("window_id"), "candidate window_id") for row in candidate_windows],
        "max_candidate_segment_pairs": max((require_int(row.get("max_candidate_segment_pairs"), "candidate segment pairs") for row in candidate_windows), default=0),
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
    material_summary = require_dict(load_json(args.material_track_root / "v17_object_material_track_summary.json"), "material track summary")
    cases = [
        require_str(require_dict(raw, f"material summary cases[{i}]").get("case"), "case")
        for i, raw in enumerate(require_list(material_summary.get("cases"), "material summary cases"))
    ]
    outputs = [
        case_state(args.material_track_root / case / "v17_object_material_track_summary.json", args)
        for case in cases
    ]
    for case_output in outputs:
        write_json(
            args.output_root / require_str(case_output.get("case"), "case") / "v17_object_material_motion_state_report.json",
            case_output,
        )
    payload = {
        "method": "build_v17_object_material_motion_state",
        "status": STATUS,
        "claim": CLAIM,
        "material_track_summary": str(args.material_track_root / "v17_object_material_track_summary.json"),
        "case_count": len(outputs),
        "cases": [
            {
                "case": case["case"],
                "report_path": str(
                    args.output_root
                    / require_str(case.get("case"), "case")
                    / "v17_object_material_motion_state_report.json"
                ),
                "material_track_window_count": case["material_track_window_count"],
                "rigid_factor_ready_pair_count": case["rigid_factor_ready_pair_count"],
                "window_rigid_motion_candidate_count": case["window_rigid_motion_candidate_count"],
                "persistent_window_motion_candidate_count": case["persistent_window_motion_candidate_count"],
                "local_adjacent_material_motion_window_count": case["local_adjacent_material_motion_window_count"],
                "noncandidate_local_adjacent_material_motion_window_count": case[
                    "noncandidate_local_adjacent_material_motion_window_count"
                ],
                "no_ready_material_motion_window_count": case["no_ready_material_motion_window_count"],
                "candidate_window_ids": case["candidate_window_ids"],
                "max_candidate_segment_pairs": case["max_candidate_segment_pairs"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for case in outputs
        ],
        "material_track_window_count": sum(require_int(case.get("material_track_window_count"), "window count") for case in outputs),
        "rigid_factor_ready_pair_count": sum(require_int(case.get("rigid_factor_ready_pair_count"), "ready pair count") for case in outputs),
        "window_rigid_motion_candidate_count": sum(require_int(case.get("window_rigid_motion_candidate_count"), "candidate count") for case in outputs),
        "persistent_window_motion_candidate_count": sum(
            require_int(case.get("persistent_window_motion_candidate_count"), "persistent candidate count")
            for case in outputs
        ),
        "local_adjacent_material_motion_window_count": sum(
            require_int(case.get("local_adjacent_material_motion_window_count"), "local motion window count")
            for case in outputs
        ),
        "noncandidate_local_adjacent_material_motion_window_count": sum(
            require_int(
                case.get("noncandidate_local_adjacent_material_motion_window_count"),
                "noncandidate local motion window count",
            )
            for case in outputs
        ),
        "no_ready_material_motion_window_count": sum(
            require_int(case.get("no_ready_material_motion_window_count"), "no-ready motion window count")
            for case in outputs
        ),
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
        "parameters": {
            "min_segment_pairs": int(args.min_segment_pairs),
            "min_endpoint_tracks": int(args.min_endpoint_tracks),
            "min_all_segment_tracks": int(args.min_all_segment_tracks),
            "accept_chain_p95_m": float(args.accept_chain_p95_m),
            "accept_all_segment_p95_m": float(args.accept_all_segment_p95_m),
        },
    }
    write_json(args.output_root / "v17_object_material_motion_state_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material-track-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_motion_state"),
    )
    parser.add_argument("--min-segment-pairs", type=int, default=5)
    parser.add_argument("--min-endpoint-tracks", type=int, default=12)
    parser.add_argument("--min-all-segment-tracks", type=int, default=12)
    parser.add_argument("--accept-chain-p95-m", type=float, default=0.015)
    parser.add_argument("--accept-all-segment-p95-m", type=float, default=0.015)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
