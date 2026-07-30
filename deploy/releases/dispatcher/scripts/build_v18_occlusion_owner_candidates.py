#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_occlusion_owner_candidates"
CLAIM = (
    "This artifact proposes bounded occlusion-owner candidates for unresolved hand rows by interpolating observed "
    "hand boxes across short detector gaps and testing overlap with current visible object boxes. It does not assign "
    "occluder ownership, infer depth ordering, fill hand pose, or validate contact."
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


def finite_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    out = [float(v) for v in value]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def interpolate_box(prev_box: list[float], next_box: list[float], prev_frame: int, next_frame: int, frame_idx: int) -> list[float]:
    if next_frame <= prev_frame:
        return prev_box
    alpha = (frame_idx - prev_frame) / float(next_frame - prev_frame)
    return [prev_box[i] * (1.0 - alpha) + next_box[i] * alpha for i in range(4)]


def box_metrics(a: list[float], b: list[float]) -> dict[str, float]:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return {
        "intersection_area_px": inter,
        "hand_box_area_px": area_a,
        "object_box_area_px": area_b,
        "iou": inter / union if union > 0 else 0.0,
        "hand_box_coverage_by_object_box": inter / area_a if area_a > 0 else 0.0,
        "object_box_coverage_by_hand_box": inter / area_b if area_b > 0 else 0.0,
    }


def hand_index(annotation: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_hand in require_list(frame.get("hands"), "hands"):
            hand = require_dict(raw_hand, "hand")
            side = str(hand.get("hand_side"))
            out[(frame_idx, side)] = hand
    return out


def visible_objects_by_frame(annotation: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for raw_frame in require_list(annotation.get("frames"), "annotation frames"):
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        rows: list[dict[str, Any]] = []
        for raw_obj in require_list(frame.get("objects"), "objects"):
            obj = require_dict(raw_obj, "object")
            if obj.get("visibility_state") == "visible" and obj.get("renderable_bbox") is True and finite_box(obj.get("bbox_xyxy")) is not None:
                rows.append(obj)
        out[frame_idx] = rows
    return out


def candidate_state(candidate_rows: list[dict[str, Any]], gap: dict[str, Any] | None, prev_box: list[float] | None, next_box: list[float] | None) -> str:
    if gap is None:
        return "unbounded_unresolved_no_temporal_gap"
    if prev_box is None or next_box is None:
        return "short_gap_missing_neighbor_hand_box"
    if candidate_rows:
        return "short_gap_visible_object_overlap_owner_candidates"
    return "short_gap_no_visible_object_overlap_candidate"


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.annotation_root / case / "v18_annotation_state.json"
    visibility_path = args.visibility_root / case / "v18_visibility_occlusion_state.json"
    annotation = require_dict(load_json(annotation_path), f"{case} annotation")
    visibility = require_dict(load_json(visibility_path), f"{case} visibility")
    hands = hand_index(annotation)
    objects_by_frame = visible_objects_by_frame(annotation)
    row_records: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    candidate_object_counts: Counter[str] = Counter()
    for raw_frame in require_list(visibility.get("frames"), "visibility frames"):
        frame = require_dict(raw_frame, "visibility frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_hand in require_list(frame.get("hands"), "visibility hands"):
            hand = require_dict(raw_hand, "visibility hand")
            if hand.get("visibility_state") != "unresolved":
                continue
            side = str(hand.get("hand_side"))
            gap = hand.get("bounded_gap_evidence") if isinstance(hand.get("bounded_gap_evidence"), dict) else None
            prev_box = next_box = interp_box = None
            candidate_rows: list[dict[str, Any]] = []
            if gap is not None:
                prev_frame = require_int(gap.get("previous_observed_frame"), "previous observed frame")
                next_frame = require_int(gap.get("next_observed_frame"), "next observed frame")
                prev_box = finite_box(hands.get((prev_frame, side), {}).get("bbox_xyxy"))
                next_box = finite_box(hands.get((next_frame, side), {}).get("bbox_xyxy"))
                if prev_box is not None and next_box is not None:
                    interp_box = interpolate_box(prev_box, next_box, prev_frame, next_frame, frame_idx)
                    for obj in objects_by_frame.get(frame_idx, []):
                        obj_box = finite_box(obj.get("bbox_xyxy"))
                        if obj_box is None:
                            continue
                        metrics = box_metrics(interp_box, obj_box)
                        if metrics["iou"] >= float(args.min_box_iou) or metrics["hand_box_coverage_by_object_box"] >= float(args.min_hand_coverage):
                            candidate = {
                                "object_id": obj.get("object_id"),
                                "track_id": obj.get("track_id"),
                                "name": obj.get("name"),
                                "model_physical_state_type": obj.get("model_physical_state_type"),
                                "geometry_scope": obj.get("geometry_scope"),
                                "bbox_xyxy": obj_box,
                                **metrics,
                            }
                            candidate_rows.append(candidate)
                            candidate_object_counts[str(obj.get("object_id"))] += 1
                    candidate_rows.sort(key=lambda row: (float(row["hand_box_coverage_by_object_box"]), float(row["iou"])), reverse=True)
            state = candidate_state(candidate_rows, gap, prev_box, next_box)
            state_counts[state] += 1
            row_records.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "source_visibility_occlusion_state": hand.get("occlusion_state"),
                    "bounded_gap_evidence": gap,
                    "interpolated_hand_box_xyxy": interp_box,
                    "candidate_state": state,
                    "candidate_count": len(candidate_rows),
                    "candidate_objects": candidate_rows,
                    "occluder_owner_accepted": False,
                    "occluder_owner": None,
                    "depth_order_resolved": False,
                    "pose_filled_through_occlusion": False,
                }
            )
    report = {
        "method": "build_v18_occlusion_owner_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {"v18_annotation_state": str(annotation_path), "v18_visibility_occlusion_state": str(visibility_path)},
        "unresolved_hand_row_count": len(row_records),
        "candidate_state_counts": dict(sorted(state_counts.items())),
        "candidate_object_counts": dict(sorted(candidate_object_counts.items())),
        "candidate_owner_row_count": state_counts.get("short_gap_visible_object_overlap_owner_candidates", 0),
        "occluder_owner_accepted_count": 0,
        "depth_order_resolved_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "row_records": row_records,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_occlusion_owner_candidates_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["candidate_state_counts"])
        object_counts.update(report["candidate_object_counts"])
    summary = {
        "method": "build_v18_occlusion_owner_candidates",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "unresolved_hand_row_count": sum(int(report["unresolved_hand_row_count"]) for report in reports),
        "candidate_state_counts": dict(sorted(state_counts.items())),
        "candidate_object_counts": dict(sorted(object_counts.items())),
        "candidate_owner_row_count": sum(int(report["candidate_owner_row_count"]) for report in reports),
        "occluder_owner_accepted_count": 0,
        "depth_order_resolved_count": 0,
        "pose_filled_through_occlusion_rows": 0,
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_occlusion_owner_candidates_report.json"),
                "unresolved_hand_row_count": report["unresolved_hand_row_count"],
                "candidate_state_counts": report["candidate_state_counts"],
                "candidate_owner_row_count": report["candidate_owner_row_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_occlusion_owner_candidates_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--visibility-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_candidates"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--min-box-iou", type=float, default=0.01)
    parser.add_argument("--min-hand-coverage", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
