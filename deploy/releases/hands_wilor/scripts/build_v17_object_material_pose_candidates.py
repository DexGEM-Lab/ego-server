#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_object_material_pose_candidates_partial_qc"
CLAIM = (
    "This artifact fits partial SE(3) material-point pose candidates only for persistent material-motion "
    "segments. It does not reconstruct object geometry, does not create full-timeline object pose variables, "
    "and cannot close the V3 solver."
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


def rotation_angle_rad(rot: np.ndarray) -> float:
    cos_theta = float((np.trace(rot) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def cloud_support(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError("invalid point cloud support input")
    center = np.mean(points, axis=0)
    centered = points - center
    radial = np.linalg.norm(centered, axis=1)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size < 3:
        singular = np.pad(singular, (0, 3 - singular.size), constant_values=0.0)
    rank2_ratio = float(singular[1] / singular[0]) if singular[0] > 1e-12 else 0.0
    rank3_ratio = float(singular[2] / singular[0]) if singular[0] > 1e-12 else 0.0
    return {
        "center_world_m": center.astype(float).tolist(),
        "aabb_extent_m": (np.max(points, axis=0) - np.min(points, axis=0)).astype(float).tolist(),
        "radial_extent_m": summarize(radial),
        "singular_values_m": singular.astype(float).tolist(),
        "rank2_ratio": rank2_ratio,
        "rank3_ratio": rank3_ratio,
    }


def finite_summary_value(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        return float("nan")
    return float(value)


def weighted_kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RuntimeError("invalid Kabsch inputs")
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(source):
        raise RuntimeError("weight length mismatch")
    total = float(np.sum(weights))
    if total <= 1e-12:
        raise RuntimeError("degenerate Kabsch weights")
    src_center = np.sum(source * weights[:, None], axis=0) / total
    tgt_center = np.sum(target * weights[:, None], axis=0) / total
    src = source - src_center
    tgt = target - tgt_center
    cov = (src * weights[:, None]).T @ tgt
    u, _s, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    trans = tgt_center - src_center @ rot
    if np.linalg.norm(src_center @ rot + trans - tgt_center) > 1e-8:
        raise RuntimeError("Kabsch transform failed centroid consistency")
    return rot, trans


def robust_fit(source: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    weights = np.ones((len(source),), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    rot = np.eye(3, dtype=np.float64)
    trans = np.zeros((3,), dtype=np.float64)
    residual = np.full((len(source),), np.nan, dtype=np.float64)
    for iteration in range(int(args.irls_iterations)):
        rot, trans = weighted_kabsch(source, target, weights)
        aligned = source @ rot + trans
        residual = np.linalg.norm(aligned - target, axis=1)
        delta = float(args.huber_delta_m)
        new_weights = np.minimum(1.0, delta / np.maximum(residual, 1e-9))
        rows.append({"iteration": int(iteration), "weight": summarize(weights), "residual_m": summarize(residual)})
        if np.max(np.abs(new_weights - weights)) < 1e-4:
            weights = new_weights
            break
        weights = new_weights
    return rot, trans, residual, rows


def material_track_windows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(require_list(summary.get("windows"), "material-track windows")):
        row = require_dict(raw, f"material-track windows[{i}]")
        window_id = require_str(row.get("window_id"), f"material-track windows[{i}].window_id")
        if window_id in out:
            raise RuntimeError(f"duplicate material-track window_id: {window_id}")
        out[window_id] = row
    return out


def frame_positions(segment: dict[str, Any], frame_to_i: dict[int, int]) -> list[int]:
    raw = require_list(segment.get("track_archive_frame_positions"), "segment.track_archive_frame_positions")
    positions = [require_int(value, f"track_archive_frame_positions[{i}]") for i, value in enumerate(raw)]
    if len(positions) < 2:
        raise RuntimeError("material-pose segment must span at least two frames")
    for left, right in zip(positions, positions[1:]):
        if right <= left:
            raise RuntimeError("material-pose segment frame positions are not strictly increasing")
    start_frame = require_int(segment.get("start_frame"), "segment.start_frame")
    end_frame = require_int(segment.get("end_frame"), "segment.end_frame")
    if frame_to_i.get(start_frame) != positions[0] or frame_to_i.get(end_frame) != positions[-1]:
        raise RuntimeError("material-pose segment frame positions disagree with segment frame ids")
    return positions


def save_candidate_npz(path: Path, payload: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_payload: dict[str, Any] = dict(payload)
    archive_payload["v17_archive_metadata_json"] = json.dumps(metadata)
    np.savez_compressed(str(path), **archive_payload)


def pose_candidate(
    *,
    case: str,
    window: dict[str, Any],
    segment: dict[str, Any],
    material_row: dict[str, Any],
    candidate_index: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    track_report_path = Path(require_str(material_row.get("report_path"), "material row report_path"))
    track_report = require_dict(load_json(track_report_path), f"{track_report_path}")
    npz_path = Path(require_str(require_dict(track_report.get("outputs"), "track outputs").get("tracks_npz"), "track outputs.tracks_npz"))
    data = np.load(npz_path)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    accepted = np.asarray(data["accepted"], dtype=bool)
    world = np.asarray(data["world_xyz"], dtype=np.float64)
    if accepted.shape[:2] != world.shape[:2] or world.shape[2] != 3:
        raise RuntimeError(f"track archive has inconsistent accepted/world arrays: {npz_path}")
    frame_to_i = {int(frame): int(i) for i, frame in enumerate(frame_idx.tolist())}
    positions = frame_positions(segment, frame_to_i)
    keep = np.all(accepted[positions], axis=0) & np.all(np.all(np.isfinite(world[positions]), axis=2), axis=0)
    track_ids = np.nonzero(keep)[0].astype(np.int64)
    if len(track_ids) != require_int(segment.get("all_segment_track_count"), "segment all_segment_track_count"):
        raise RuntimeError("material-pose track count disagrees with material-motion segment")
    canonical = world[positions[0], track_ids]
    support = cloud_support(canonical)
    support_checks = {
        "min_pose_tracks_met": bool(len(track_ids) >= int(args.min_pose_tracks)),
        "min_radial_extent_met": bool(
            finite_summary_value(support["radial_extent_m"], "p95") >= float(args.min_radial_extent_m)
        ),
        "min_rank2_ratio_met": bool(float(support["rank2_ratio"]) >= float(args.min_rank2_ratio)),
    }
    frame_rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    observed_blocks: list[np.ndarray] = []
    for pos in positions:
        observed = world[pos, track_ids]
        rot, trans, residual, solver_rows = robust_fit(canonical, observed, args)
        frame_checks = {
            "frame_residual_p95_met": bool(float(summarize(residual).get("p95", float("inf"))) <= float(args.accept_frame_p95_m)),
            "max_rotation_from_reference_met": bool(rotation_angle_rad(rot) <= float(args.max_rotation_from_reference_rad)),
            "max_translation_from_reference_met": bool(np.linalg.norm(trans) <= float(args.max_translation_from_reference_m)),
        }
        frame_rows.append(
            {
                "frame_idx": int(frame_idx[pos]),
                "track_count": int(len(track_ids)),
                "rotation": rot.astype(float).tolist(),
                "translation_m": trans.astype(float).tolist(),
                "rotation_angle_from_reference_rad": rotation_angle_rad(rot),
                "translation_norm_from_reference_m": float(np.linalg.norm(trans)),
                "residual_m": summarize(residual),
                "solver": solver_rows,
                "readiness_checks": frame_checks,
                "partial_material_pose_frame_candidate": bool(all(frame_checks.values())),
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
        )
        residuals.extend(residual.astype(float).tolist())
        rotations.append(rot)
        translations.append(trans)
        observed_blocks.append(observed)
    all_frame_checks = [bool(row["partial_material_pose_frame_candidate"]) for row in frame_rows]
    candidate_checks = {
        **support_checks,
        "all_pose_frame_checks_met": bool(all(all_frame_checks)),
        "segment_motion_candidate_source_met": bool(segment.get("persistent_window_motion_candidate") is True),
    }
    window_id = require_str(window.get("window_id"), "window.window_id")
    candidate_id = f"{window_id}_segment_{require_int(segment.get('start_frame'), 'segment.start_frame')}_{require_int(segment.get('end_frame'), 'segment.end_frame')}"
    archive_path = output_dir / f"{candidate_id}.npz"
    metadata = {
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "candidate_id": candidate_id,
        "window_id": window_id,
        "object_id": require_str(window.get("object_id"), "window.object_id"),
        "track_id": require_str(window.get("track_id"), "window.track_id"),
        "frame_count": len(positions),
        "track_count": int(len(track_ids)),
        "partial_material_pose_candidate": bool(all(candidate_checks.values())),
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "v3_solver_complete": False,
    }
    save_candidate_npz(
        archive_path,
        {
            "frame_idx": frame_idx[positions].astype(np.int32),
            "track_id": track_ids.astype(np.int32),
            "canonical_points_world_m": canonical.astype(np.float32),
            "observed_points_world_m": np.stack(observed_blocks, axis=0).astype(np.float32),
            "rotation": np.stack(rotations, axis=0).astype(np.float64),
            "translation_m": np.stack(translations, axis=0).astype(np.float64),
        },
        metadata,
    )
    return {
        **metadata,
        "candidate_index": int(candidate_index),
        "archive_path": str(archive_path),
        "source_track_report": str(track_report_path),
        "source_tracks_npz": str(npz_path),
        "start_frame": require_int(segment.get("start_frame"), "segment.start_frame"),
        "end_frame": require_int(segment.get("end_frame"), "segment.end_frame"),
        "pair_count": require_int(segment.get("pair_count"), "segment.pair_count"),
        "canonical_support": support,
        "residual_m": summarize(np.asarray(residuals, dtype=np.float64)),
        "readiness_checks": candidate_checks,
        "frame_rows": frame_rows,
        "partial_material_pose_candidate": bool(all(candidate_checks.values())),
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "v3_solver_complete": False,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    motion_path = args.material_motion_state_root / case / "v17_object_material_motion_state_report.json"
    material_path = args.object_material_track_root / case / "v17_object_material_track_summary.json"
    motion = require_dict(load_json(motion_path), f"{case} material-motion state report")
    material = require_dict(load_json(material_path), f"{case} material-track summary")
    material_by_window = material_track_windows(material)
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for window_i, raw_window in enumerate(require_list(motion.get("windows"), "motion windows")):
        window = require_dict(raw_window, f"motion windows[{window_i}]")
        if window.get("persistent_window_motion_candidate") is not True:
            continue
        window_id = require_str(window.get("window_id"), f"motion windows[{window_i}].window_id")
        material_row = material_by_window.get(window_id)
        if material_row is None:
            raise RuntimeError(f"{case} material-motion candidate missing material-track window: {window_id}")
        for segment_i, raw_segment in enumerate(require_list(window.get("segments"), f"{window_id}.segments")):
            segment = require_dict(raw_segment, f"{window_id}.segments[{segment_i}]")
            if segment.get("persistent_window_motion_candidate") is not True:
                continue
            candidates.append(
                pose_candidate(
                    case=case,
                    window=window,
                    segment=segment,
                    material_row=material_row,
                    candidate_index=candidate_index,
                    output_dir=args.output_root / case / "candidate_archives",
                    args=args,
                )
            )
            candidate_index += 1
    ready_candidates = [candidate for candidate in candidates if candidate["partial_material_pose_candidate"]]
    report = {
        "method": "build_v17_object_material_pose_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_material_motion_state_report": str(motion_path),
        "source_object_material_track_summary": str(material_path),
        "material_track_window_count": require_int(motion.get("material_track_window_count"), "motion material_track_window_count"),
        "persistent_window_motion_candidate_count": require_int(
            motion.get("persistent_window_motion_candidate_count"),
            "motion persistent_window_motion_candidate_count",
        ),
        "partial_material_pose_candidate_segment_count": len(candidates),
        "partial_material_pose_candidate_ready_segment_count": len(ready_candidates),
        "candidate_window_ids": sorted({require_str(candidate.get("window_id"), "candidate window_id") for candidate in candidates}),
        "candidate_segment_ids": [require_str(candidate.get("candidate_id"), "candidate candidate_id") for candidate in candidates],
        "candidates": candidates,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "rigid_pose_requirement_met": False,
        "v3_solver_complete": False,
        "parameters": {
            "min_pose_tracks": int(args.min_pose_tracks),
            "min_radial_extent_m": float(args.min_radial_extent_m),
            "min_rank2_ratio": float(args.min_rank2_ratio),
            "accept_frame_p95_m": float(args.accept_frame_p95_m),
            "max_rotation_from_reference_rad": float(args.max_rotation_from_reference_rad),
            "max_translation_from_reference_m": float(args.max_translation_from_reference_m),
            "huber_delta_m": float(args.huber_delta_m),
            "irls_iterations": int(args.irls_iterations),
        },
    }
    write_json(args.output_root / case / "v17_object_material_pose_candidate_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    motion_summary_path = args.material_motion_state_root / "v17_object_material_motion_state_summary.json"
    motion_summary = require_dict(load_json(motion_summary_path), "material-motion state summary")
    cases = [
        require_str(require_dict(raw, f"motion summary cases[{i}]").get("case"), "motion summary case")
        for i, raw in enumerate(require_list(motion_summary.get("cases"), "motion summary cases"))
    ]
    reports = [build_case(case, args) for case in cases]
    payload = {
        "method": "build_v17_object_material_pose_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "source_material_motion_state_summary": str(motion_summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "report_path": str(
                    args.output_root
                    / require_str(report.get("case"), "case")
                    / "v17_object_material_pose_candidate_report.json"
                ),
                "persistent_window_motion_candidate_count": report["persistent_window_motion_candidate_count"],
                "partial_material_pose_candidate_segment_count": report["partial_material_pose_candidate_segment_count"],
                "partial_material_pose_candidate_ready_segment_count": report[
                    "partial_material_pose_candidate_ready_segment_count"
                ],
                "candidate_window_ids": report["candidate_window_ids"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for report in reports
        ],
        "persistent_window_motion_candidate_count": sum(
            require_int(report.get("persistent_window_motion_candidate_count"), "persistent window candidate count")
            for report in reports
        ),
        "partial_material_pose_candidate_segment_count": sum(
            require_int(report.get("partial_material_pose_candidate_segment_count"), "candidate segment count")
            for report in reports
        ),
        "partial_material_pose_candidate_ready_segment_count": sum(
            require_int(report.get("partial_material_pose_candidate_ready_segment_count"), "ready candidate segment count")
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
    write_json(args.output_root / "v17_object_material_pose_candidate_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--material-motion-state-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_motion_state"),
    )
    parser.add_argument(
        "--object-material-track-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_tracks"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_object_material_pose_candidates"),
    )
    parser.add_argument("--min-pose-tracks", type=int, default=50)
    parser.add_argument("--min-radial-extent-m", type=float, default=0.03)
    parser.add_argument("--min-rank2-ratio", type=float, default=0.20)
    parser.add_argument("--accept-frame-p95-m", type=float, default=0.015)
    parser.add_argument("--max-rotation-from-reference-rad", type=float, default=1.5)
    parser.add_argument("--max-translation-from-reference-m", type=float, default=0.8)
    parser.add_argument("--huber-delta-m", type=float, default=0.015)
    parser.add_argument("--irls-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
