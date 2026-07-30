#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "v17_multi_object_contact_evidence_qc"
CLAIM = (
    "This artifact builds hand-side/object contact evidence rows for every active multi-object timeline row. "
    "Rows with hand and visible-surface geometry carry nearest-distance measurements; rows without geometry are "
    "explicitly unobserved. Visible-surface distances measure object-mask RGBD surfaces only, so accepted local "
    "contact-patch states can disagree with this layer. It is not a contact-mode optimizer and cannot close the V3 solver."
)
SIDES = ("left", "right")


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
        raise RuntimeError("invalid point cloud")
    if len(points) <= max_points:
        return points.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[idx].astype(np.float64, copy=False)


def nearest_summary(a: np.ndarray, b: np.ndarray, chunk: int) -> dict[str, Any]:
    if len(a) == 0 or len(b) == 0:
        raise RuntimeError("nearest distance requires non-empty point clouds")
    out = np.empty((len(a),), dtype=np.float64)
    for start in range(0, len(a), chunk):
        block = a[start : start + chunk]
        diff = block[:, None, :] - b[None, :, :]
        out[start : start + len(block)] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return summarize(out)


def hand_points(hand: dict[str, Any], max_points: int, seed: int) -> np.ndarray | None:
    for key in ("vertices_world_m", "vertices_sample_world_m", "joints3d_world_m"):
        if key not in hand:
            continue
        points = np.asarray(hand.get(key), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            continue
        return sample_points(points, max_points, seed)
    return None


def load_annotation_hands(path: Path) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    payload = require_dict(load_json(path), f"{path}")
    frames = require_list(payload.get("frames"), "annotation frames")
    out: dict[int, dict[str, np.ndarray]] = {}
    for row_i, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"annotation frames[{row_i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"annotation frames[{row_i}].frame_idx")
        hands = require_list(frame.get("hands"), f"annotation frame {frame_idx}.hands")
        side_points: dict[str, np.ndarray] = {}
        for hand_i, raw_hand in enumerate(hands):
            hand = require_dict(raw_hand, f"annotation frame {frame_idx}.hands[{hand_i}]")
            side = require_str(hand.get("side"), f"annotation frame {frame_idx}.hands[{hand_i}].side")
            if side not in SIDES:
                continue
            points = hand_points(hand, 1024, stable_seed(path, frame_idx, side))
            if points is not None:
                side_points[side] = points
        out[frame_idx] = side_points
    return out, len(frames)


def load_timeline(path: Path) -> tuple[dict[int, list[dict[str, Any]]], int, int]:
    payload = require_dict(load_json(path), f"{path}")
    frames = require_list(payload.get("frames"), "timeline frames")
    frame_count = require_int(payload.get("frame_count"), "timeline frame_count")
    object_frame_rows = require_int(payload.get("object_frame_rows"), "timeline object_frame_rows")
    if len(frames) != frame_count:
        raise RuntimeError(f"{path} frame_count disagrees with frame array length")
    out: dict[int, list[dict[str, Any]]] = {}
    counted = 0
    for row_i, raw_frame in enumerate(frames):
        frame = require_dict(raw_frame, f"timeline frames[{row_i}]")
        frame_idx = require_int(frame.get("frame_idx"), f"timeline frames[{row_i}].frame_idx")
        objects = [require_dict(raw, f"timeline frame {frame_idx}.objects[{i}]") for i, raw in enumerate(require_list(frame.get("objects"), f"timeline frame {frame_idx}.objects"))]
        counted += len(objects)
        out[frame_idx] = objects
    if counted != object_frame_rows:
        raise RuntimeError(f"{path} object_frame_rows disagrees with frame objects")
    return out, frame_count, object_frame_rows


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
    out: dict[tuple[str, int], np.ndarray] = {}
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
        out[(row_object, row_frame)] = points
    return out


def evidence_row(
    *,
    case: str,
    frame_idx: int,
    obj: dict[str, Any],
    side: str,
    hand_by_side: dict[str, np.ndarray],
    surfaces: dict[tuple[str, int], np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_id = require_str(obj.get("object_id"), "timeline object_id")
    track_id = require_str(obj.get("track_id"), "timeline track_id")
    hand = hand_by_side.get(side)
    surface = surfaces.get((object_id, frame_idx))
    missing: list[str] = []
    if hand is None:
        missing.append("hand_world_geometry")
    if surface is None:
        missing.append("visible_object_surface")
    base = {
        "case": case,
        "frame_idx": frame_idx,
        "object_id": object_id,
        "track_id": track_id,
        "hand_side": side,
        "measurement_type": "multi_object_hand_object_distance_evidence",
        "coordinate_frame": "world_metric",
        "contact_mode_state": "unobserved" if missing else "measured_distance_evidence",
        "geometry_source": "multi_object_visible_surface_only",
        "missing_geometry": missing,
        "contact_factor_ready": False,
        "contact_mode_variable_estimated": False,
        "annotation_ready": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    if missing:
        return base
    if hand is None or surface is None:
        raise RuntimeError("missing geometry branch failed to return")
    hand_sample = sample_points(hand, int(args.max_hand_points), stable_seed(case, frame_idx, object_id, side, "hand"))
    surface_sample = sample_points(surface, int(args.max_surface_points), stable_seed(case, frame_idx, object_id, side, "surface"))
    hand_to_object = nearest_summary(hand_sample, surface_sample, int(args.distance_chunk))
    object_to_hand = nearest_summary(surface_sample, hand_sample, int(args.distance_chunk))
    min_distance = min(float(hand_to_object.get("min", float("inf"))), float(object_to_hand.get("min", float("inf"))))
    checks = {
        "near_visible_surface_distance_candidate": bool(min_distance <= float(args.near_distance_m)),
        "hand_sample_points_met": bool(len(hand_sample) >= int(args.min_hand_points)),
        "surface_sample_points_met": bool(len(surface_sample) >= int(args.min_surface_points)),
    }
    return {
        **base,
        "hand_points": int(len(hand_sample)),
        "surface_points": int(len(surface_sample)),
        "hand_to_object_m": hand_to_object,
        "object_to_hand_m": object_to_hand,
        "min_symmetric_distance_m": min_distance,
        "evidence_checks": checks,
        "visible_surface_distance_candidate": bool(all(checks.values())),
        "contact_distance_candidate": False,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.graph_root / case / "annotations_v17_full_timeline_graph.json"
    timeline_path = args.multi_object_timeline_root / case / "v17_multi_object_timeline.json"
    visible_report_path = args.visible_surface_root / case / "v17_multi_object_visible_surface_report.json"
    visible_archive_path = args.visible_surface_root / case / "multi_object_visible_surfaces_world.npz"
    hands, annotation_frame_count = load_annotation_hands(annotation_path)
    timeline, frame_count, object_frame_rows = load_timeline(timeline_path)
    if annotation_frame_count != frame_count:
        raise RuntimeError(f"{case} annotation frame count disagrees with multi-object timeline")
    surfaces = load_visible_surfaces(visible_report_path, visible_archive_path)
    rows: list[dict[str, Any]] = []
    for frame_idx in sorted(timeline):
        hand_by_side = hands.get(frame_idx, {})
        for obj in timeline[frame_idx]:
            for side in SIDES:
                rows.append(
                    evidence_row(
                        case=case,
                        frame_idx=frame_idx,
                        obj=obj,
                        side=side,
                        hand_by_side=hand_by_side,
                        surfaces=surfaces,
                        args=args,
                    )
                )
    expected_rows = 2 * object_frame_rows
    if len(rows) != expected_rows:
        raise RuntimeError(f"{case} row count {len(rows)} does not equal two hand sides times object frames {expected_rows}")
    measured_rows = [row for row in rows if row["contact_mode_state"] == "measured_distance_evidence"]
    near_rows = [row for row in measured_rows if row.get("visible_surface_distance_candidate") is True]
    missing_counts: dict[str, int] = {}
    for row in rows:
        for reason in require_list(row.get("missing_geometry"), "missing_geometry"):
            key = require_str(reason, "missing geometry reason")
            missing_counts[key] = missing_counts.get(key, 0) + 1
    report = {
        "method": "build_v17_multi_object_contact_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "source_annotations": str(annotation_path),
        "source_multi_object_timeline": str(timeline_path),
        "source_visible_surface_report": str(visible_report_path),
        "source_visible_surface_archive": str(visible_archive_path),
        "frame_count": frame_count,
        "object_frame_rows": object_frame_rows,
        "expected_hand_object_rows": expected_rows,
        "hand_object_rows": len(rows),
        "measured_distance_rows": len(measured_rows),
        "unobserved_rows": len(rows) - len(measured_rows),
        "visible_surface_distance_candidate_rows": len(near_rows),
        "contact_distance_candidate_rows": 0,
        "contact_factor_ready_rows": 0,
        "contact_evidence_semantics": (
            "Rows measure distance from MANO world geometry to multi-object visible RGBD surfaces. "
            "They do not include accepted local contact-patch meshes or legacy corrected single-object surfaces."
        ),
        "missing_geometry_reason_counts": missing_counts,
        "rows": rows,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
        "parameters": {
            "max_hand_points": int(args.max_hand_points),
            "max_surface_points": int(args.max_surface_points),
            "min_hand_points": int(args.min_hand_points),
            "min_surface_points": int(args.min_surface_points),
            "near_distance_m": float(args.near_distance_m),
        },
    }
    write_json(args.output_root / case / "v17_multi_object_contact_evidence_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.multi_object_timeline_root / "v17_multi_object_timeline_summary.json"
    summary = require_dict(load_json(summary_path), "multi-object timeline summary")
    cases = [
        require_str(require_dict(raw, f"timeline summary cases[{i}]").get("case"), "timeline summary case")
        for i, raw in enumerate(require_list(summary.get("cases"), "timeline summary cases"))
    ]
    reports = [build_case(case, args) for case in cases]
    payload = {
        "method": "build_v17_multi_object_contact_evidence",
        "status": STATUS,
        "claim": CLAIM,
        "source_multi_object_timeline_summary": str(summary_path),
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / require_str(report.get("case"), "case") / "v17_multi_object_contact_evidence_report.json"),
                "frame_count": report["frame_count"],
                "object_frame_rows": report["object_frame_rows"],
                "hand_object_rows": report["hand_object_rows"],
                "measured_distance_rows": report["measured_distance_rows"],
                "unobserved_rows": report["unobserved_rows"],
                "visible_surface_distance_candidate_rows": report["visible_surface_distance_candidate_rows"],
                "contact_distance_candidate_rows": report["contact_distance_candidate_rows"],
                "contact_factor_ready_rows": report["contact_factor_ready_rows"],
                "annotation_ready": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
                "v3_solver_complete": False,
            }
            for report in reports
        ],
        "object_frame_rows": sum(require_int(report.get("object_frame_rows"), "object_frame_rows") for report in reports),
        "hand_object_rows": sum(require_int(report.get("hand_object_rows"), "hand_object_rows") for report in reports),
        "measured_distance_rows": sum(require_int(report.get("measured_distance_rows"), "measured_distance_rows") for report in reports),
        "unobserved_rows": sum(require_int(report.get("unobserved_rows"), "unobserved_rows") for report in reports),
        "visible_surface_distance_candidate_rows": sum(
            require_int(report.get("visible_surface_distance_candidate_rows"), "visible surface candidate rows")
            for report in reports
        ),
        "contact_distance_candidate_rows": 0,
        "contact_factor_ready_rows": 0,
        "annotation_ready": False,
        "deliverable_ready": False,
        "accuracy_target_met": False,
        "object_geometry_complete": False,
        "object_pose_requirement_met": False,
        "v3_solver_complete": False,
    }
    write_json(args.output_root / "v17_multi_object_contact_evidence_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_contact_mode_factor_graph"),
    )
    parser.add_argument(
        "--multi-object-timeline-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"),
    )
    parser.add_argument(
        "--visible-surface-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data2/ego_annotation_outputs/v17_multi_object_contact_evidence"),
    )
    parser.add_argument("--max-hand-points", type=int, default=256)
    parser.add_argument("--max-surface-points", type=int, default=512)
    parser.add_argument("--min-hand-points", type=int, default=21)
    parser.add_argument("--min-surface-points", type=int, default=50)
    parser.add_argument("--near-distance-m", type=float, default=0.03)
    parser.add_argument("--distance-chunk", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
