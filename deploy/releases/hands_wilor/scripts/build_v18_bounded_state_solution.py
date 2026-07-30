#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
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

STATUS = "v18_bounded_fixed_pass_state_solution"
CLAIM = (
    "This artifact is a bounded fixed-pass V18 state solution over renderable annotation evidence. "
    "It classifies observation, unresolved gaps, occlusion owner candidates, occlusion depth-order triage evidence, "
    "object geometry scope, and contact status without accepting occluder ownership, filling occluded poses, or "
    "promoting visible masks/surfaces to complete object pose."
)
HAND_SIDES = ("left", "right")


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


def bbox_center(value: Any) -> list[float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return [(x0 + x1) * 0.5, (y0 + y1) * 0.5]


def bbox_area(value: Any) -> float | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    return (x1 - x0) * (y1 - y0)


def visible_object_candidates(frame: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw_obj in require_list(frame.get("objects"), "frame objects"):
        obj = require_dict(raw_obj, "object")
        if obj.get("visibility_state") != "visible":
            continue
        center = bbox_center(obj.get("bbox_xyxy"))
        candidates.append(
            {
                "object_id": obj.get("object_id"),
                "name": obj.get("name"),
                "geometry_scope": obj.get("geometry_scope"),
                "fast_motion_state": obj.get("fast_motion_state"),
                "bbox_center_xy": center,
                "candidate_status": "visible_object_candidate_only_no_depth_order_owner",
            }
        )
    return candidates


def hand_gap_classes(frames: list[dict[str, Any]], max_gap_frames: int) -> dict[tuple[int, str], dict[str, Any]]:
    by_side: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {side: [] for side in HAND_SIDES}
    for raw_frame in frames:
        frame = require_dict(raw_frame, "annotation frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        by_hand = {str(require_dict(raw, "hand").get("hand_side")): require_dict(raw, "hand") for raw in require_list(frame.get("hands"), "hands")}
        for side in HAND_SIDES:
            if side in by_hand:
                by_side[side].append((frame_idx, frame, by_hand[side]))
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for side, rows in by_side.items():
        run: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        previous_observed: int | None = None
        def flush(next_observed: int | None) -> None:
            nonlocal run, previous_observed
            if not run:
                return
            gap_len = len(run)
            has_visible_objects = any(require_dict(row_frame.get("frame_summary"), "frame_summary").get("visible_objects", 0) for _idx, row_frame, _hand in run)
            if gap_len <= max_gap_frames and previous_observed is not None and next_observed is not None and has_visible_objects:
                gap_class = "short_gap_possible_occlusion_unfilled"
                bounded_candidate = True
                reason = "bounded detector gap between observed hand states with visible object candidates"
            elif gap_len <= max_gap_frames and previous_observed is not None and next_observed is not None:
                gap_class = "short_gap_no_visible_occluder_evidence_unfilled"
                bounded_candidate = False
                reason = "bounded detector gap but no visible object candidate in the gap"
            else:
                gap_class = "long_or_open_unresolved_gap_unfilled"
                bounded_candidate = False
                reason = "gap exceeds bounded infill window or lacks pre/post observations"
            for idx, row_frame, _hand in run:
                out[(idx, side)] = {
                    "gap_class": gap_class,
                    "gap_length_frames": gap_len,
                    "previous_observed_frame": previous_observed,
                    "next_observed_frame": next_observed,
                    "bounded_occlusion_candidate": bounded_candidate,
                    "occluder_owner_status": "candidate_only_unowned" if bounded_candidate else "unowned_unresolved",
                    "occluder_owner_candidates": visible_object_candidates(row_frame) if bounded_candidate else [],
                    "pose_filled_through_occlusion": False,
                    "classification_reason": reason,
                }
            run = []
        for frame_idx, frame, hand in rows:
            visibility_state = str(hand.get("visibility_state"))
            if visibility_state == "unresolved":
                run.append((frame_idx, frame, hand))
                continue
            flush(frame_idx)
            previous_observed = frame_idx
        flush(None)
    return out


def hand_solution_state(hand: dict[str, Any], gap_info: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    visibility_state = str(hand.get("visibility_state"))
    if visibility_state == "visible" and hand.get("metric_depth_compatible") is True:
        return "observed_hand_depth_consistent", {"pose_filled_through_occlusion": False}
    if visibility_state == "visible":
        return "observed_hand_depth_unchecked", {"pose_filled_through_occlusion": False}
    if visibility_state == "partially_visible":
        return "partial_hand_observation_uncertain", {"pose_filled_through_occlusion": False}
    if gap_info is not None:
        return str(gap_info["gap_class"]), gap_info
    return "unresolved_hand_unclassified_unfilled", {"pose_filled_through_occlusion": False}


def object_solution_state(obj: dict[str, Any]) -> str:
    visibility_state = str(obj.get("visibility_state"))
    geometry_scope = str(obj.get("geometry_scope"))
    if visibility_state == "visible" and geometry_scope == "visible_surface_depth_backed":
        return "visible_surface_only_hidden_geometry_unresolved"
    if visibility_state == "visible" and geometry_scope == "visible_mask_only_surface_rejected":
        return "visible_mask_only_surface_rejected_hidden_geometry_unresolved"
    if visibility_state == "visible":
        return "visible_without_metric_geometry_unresolved"
    if visibility_state == "unresolved":
        return "active_object_visibility_unresolved_unfilled"
    return "inactive_or_out_of_frame"


def occlusion_candidate_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(report.get("row_records"), "occlusion candidate row_records"):
        row = require_dict(raw, "occlusion candidate row")
        frame_idx = require_int(row.get("frame_idx"), "occlusion candidate frame_idx")
        hand_side = require_str(row.get("hand_side"), "occlusion candidate hand_side")
        out[(frame_idx, hand_side)] = row
    return out


def occlusion_depth_evidence_index(report: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in require_list(report.get("row_records"), "occlusion depth row_records"):
        row = require_dict(raw, "occlusion depth evidence row")
        frame_idx = require_int(row.get("frame_idx"), "occlusion depth frame_idx")
        hand_side = require_str(row.get("hand_side"), "occlusion depth hand_side")
        out[(frame_idx, hand_side)] = row
    return out


def contact_solution_state(contact: dict[str, Any]) -> tuple[str, str, list[str]]:
    state = str(contact.get("v18_consistency_state"))
    blockers_raw = contact.get("blockers", [])
    blockers = [str(x) for x in blockers_raw] if isinstance(blockers_raw, list) else []
    if state == "image_contact_rejected_by_metric_depth":
        return "rejected_contact_current_metric_depth", "not_contact_under_current_depth_evidence", blockers
    if state == "image_overlap_only":
        return "near_image_overlap_only", "near_or_overlap_unresolved_not_contact", blockers
    if state == "unobserved_pair":
        return "unobserved_pair_unresolved", "unresolved", blockers
    if state == "no_contact_image_evidence":
        return "no_contact_image_evidence", "not_contact_by_image_evidence", blockers
    if state == "metric_depth_compatible_but_geometry_incomplete":
        return "metric_candidate_geometry_blocked", "unresolved_geometry_blocked", blockers
    return "unresolved_contact_state", "unresolved", blockers


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.annotation_root / case / "v18_annotation_state.json"
    occlusion_candidates_path = args.occlusion_owner_candidates_root / case / "v18_occlusion_owner_candidates_report.json"
    occlusion_depth_path = args.occlusion_depth_evidence_root / case / "v18_occlusion_depth_order_evidence_report.json"
    state = require_dict(load_json(state_path), f"{case} annotation state")
    occlusion_candidates = require_dict(load_json(occlusion_candidates_path), f"{case} occlusion owner candidates")
    occlusion_depth_evidence = require_dict(load_json(occlusion_depth_path), f"{case} occlusion depth-order evidence")
    occlusion_candidate_by_hand = occlusion_candidate_index(occlusion_candidates)
    occlusion_depth_by_hand = occlusion_depth_evidence_index(occlusion_depth_evidence)
    frames_raw = [require_dict(raw, "annotation frame") for raw in require_list(state.get("frames"), "annotation frames")]
    frame_count = require_int(state.get("frame_count"), "frame_count")
    if len(frames_raw) != frame_count:
        raise RuntimeError(f"{case}: annotation frames length {len(frames_raw)} != frame_count {frame_count}")
    gap_index = hand_gap_classes(frames_raw, args.max_hand_gap_frames)
    frames: list[dict[str, Any]] = []
    hand_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    contact_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    contact_ready_rows = 0
    pose_filled_rows = 0
    occlusion_owner_candidate_rows = 0
    occluder_owner_accepted_rows = 0
    occlusion_depth_order_resolved_rows = 0
    occlusion_depth_evidence_candidate_pair_rows = 0
    occlusion_depth_evidence_support_pair_rows = 0
    occlusion_depth_evidence_contradiction_pair_rows = 0
    for frame in frames_raw:
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        hand_rows: list[dict[str, Any]] = []
        object_rows: list[dict[str, Any]] = []
        contact_rows: list[dict[str, Any]] = []
        for raw_hand in require_list(frame.get("hands"), "hands"):
            hand = require_dict(raw_hand, "hand")
            side = require_str(hand.get("hand_side"), "hand_side")
            gap_info = gap_index.get((frame_idx, side))
            solution_state, extra = hand_solution_state(hand, gap_info)
            occlusion_candidate = occlusion_candidate_by_hand.get((frame_idx, side))
            if occlusion_candidate is not None:
                extra = {
                    **extra,
                    "owner_candidate_state": occlusion_candidate.get("candidate_state"),
                    "owner_candidate_count": occlusion_candidate.get("candidate_count"),
                    "owner_candidate_objects": occlusion_candidate.get("candidate_objects", []),
                    "owner_candidate_source": "v18_occlusion_owner_candidates",
                    "occluder_owner_accepted": False,
                    "depth_order_resolved": False,
                }
                if require_int(occlusion_candidate.get("candidate_count"), "candidate count") > 0:
                    occlusion_owner_candidate_rows += 1
                if occlusion_candidate.get("occluder_owner_accepted") is True:
                    occluder_owner_accepted_rows += 1
                if occlusion_candidate.get("depth_order_resolved") is True:
                    occlusion_depth_order_resolved_rows += 1
            depth_evidence = occlusion_depth_by_hand.get((frame_idx, side))
            if depth_evidence is not None:
                pair_evidence = [require_dict(raw_pair, "depth evidence pair") for raw_pair in require_list(depth_evidence.get("candidate_pair_depth_evidence"), "candidate pair depth evidence")]
                pair_state_counts = require_dict(depth_evidence.get("candidate_pair_depth_state_counts"), "candidate pair depth state counts")
                occlusion_depth_evidence_candidate_pair_rows += len(pair_evidence)
                occlusion_depth_evidence_support_pair_rows += require_int(
                    pair_state_counts.get("scene_depth_supports_foreground_occluder_candidate_owner_unaccepted", 0),
                    "support pair count",
                )
                occlusion_depth_evidence_contradiction_pair_rows += require_int(
                    pair_state_counts.get("scene_depth_contradicts_foreground_occluder_candidate", 0),
                    "contradiction pair count",
                )
                extra = {
                    **extra,
                    "depth_order_evidence_source": "v18_occlusion_depth_order_evidence",
                    "depth_order_evidence_state": depth_evidence.get("row_depth_evidence_state"),
                    "depth_order_candidate_pair_count": len(pair_evidence),
                    "depth_order_candidate_pair_state_counts": pair_state_counts,
                    "depth_order_candidate_pair_evidence": pair_evidence,
                    "depth_order_resolved": False,
                }
            hand_counts[solution_state] += 1
            if extra.get("bounded_occlusion_candidate") is True:
                occlusion_counts["short_gap_possible_occlusion_unfilled"] += 1
            center = bbox_center(hand.get("bbox_xyxy"))
            hand_rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": side,
                    "visibility_state": hand.get("visibility_state"),
                    "solution_state": solution_state,
                    "bbox_xyxy": hand.get("bbox_xyxy"),
                    "bbox_center_xy": center,
                    "metric_depth_state": hand.get("metric_depth_state"),
                    "metric_depth_compatible": hand.get("metric_depth_compatible"),
                    "uncertainty_state": hand.get("uncertainty_state"),
                    "occlusion_solution": extra,
                    "pose_filled_through_occlusion": False,
                    "mano_pose_complete": False,
                }
            )
        for raw_obj in require_list(frame.get("objects"), "objects"):
            obj = require_dict(raw_obj, "object")
            solution_state = object_solution_state(obj)
            object_counts[solution_state] += 1
            center = bbox_center(obj.get("bbox_xyxy"))
            area = bbox_area(obj.get("bbox_xyxy"))
            per_object_contacts: list[dict[str, Any]] = []
            for raw_contact in require_list(obj.get("contact_rows", []), "contact_rows"):
                contact = require_dict(raw_contact, "contact")
                contact_solution, contact_mode, blockers = contact_solution_state(contact)
                contact_counts[contact_solution] += 1
                if contact.get("v18_contact_factor_ready") is True:
                    contact_ready_rows += 1
                row = {
                    "frame_idx": frame_idx,
                    "object_id": obj.get("object_id"),
                    "hand_side": contact.get("hand_side"),
                    "solution_state": contact_solution,
                    "contact_mode": contact_mode,
                    "source_consistency_state": contact.get("v18_consistency_state"),
                    "pair_depth_gap_state": contact.get("pair_depth_gap_state"),
                    "metric_depth_compatible_candidate": contact.get("metric_depth_compatible_candidate"),
                    "contact_factor_ready": False,
                    "blockers": blockers,
                }
                per_object_contacts.append(row)
                contact_rows.append(row)
            object_rows.append(
                {
                    "frame_idx": frame_idx,
                    "object_id": obj.get("object_id"),
                    "track_id": obj.get("track_id"),
                    "name": obj.get("name"),
                    "visibility_state": obj.get("visibility_state"),
                    "solution_state": solution_state,
                    "model_physical_state_type": obj.get("model_physical_state_type"),
                    "fast_motion_state": obj.get("fast_motion_state"),
                    "geometry_scope": obj.get("geometry_scope"),
                    "hidden_geometry_state": obj.get("hidden_geometry_state"),
                    "bbox_xyxy": obj.get("bbox_xyxy"),
                    "bbox_center_xy": center,
                    "bbox_area_px": area,
                    "mask_path": obj.get("mask_path"),
                    "renderable_mask": obj.get("renderable_mask"),
                    "contact_rows": per_object_contacts,
                    "object_geometry_complete": False,
                    "object_pose_requirement_met": False,
                    "hidden_geometry_reconstructed": False,
                }
            )
        if any(row.get("pose_filled_through_occlusion") is True for row in hand_rows):
            pose_filled_rows += 1
        frames.append(
            {
                "frame_idx": frame_idx,
                "raw_frame_path": frame.get("raw_frame_path"),
                "hands": hand_rows,
                "objects": object_rows,
                "contacts": contact_rows,
                "frame_solution_summary": {
                    "hand_solution_state_counts": dict(sorted(Counter(str(row["solution_state"]) for row in hand_rows).items())),
                    "object_solution_state_counts": dict(sorted(Counter(str(row["solution_state"]) for row in object_rows).items())),
                    "contact_solution_state_counts": dict(sorted(Counter(str(row["solution_state"]) for row in contact_rows).items())),
                    "contact_factor_ready": 0,
                    "pose_filled_through_occlusion": False,
                    "short_gap_possible_occlusion_unfilled": sum(
                        1
                        for row in hand_rows
                        if require_dict(row.get("occlusion_solution"), "occlusion_solution").get("bounded_occlusion_candidate") is True
                    ),
                    "occlusion_owner_candidate_rows": sum(
                        1
                        for row in hand_rows
                        if require_int(require_dict(row.get("occlusion_solution"), "occlusion_solution").get("owner_candidate_count", 0), "owner candidate count") > 0
                    ),
                    "occlusion_depth_evidence_candidate_pair_rows": sum(
                        require_int(require_dict(row.get("occlusion_solution"), "occlusion_solution").get("depth_order_candidate_pair_count", 0), "depth candidate pair count")
                        for row in hand_rows
                    ),
                    "status": "bounded_state_classified_no_pose_fill",
                },
            }
        )
    report = {
        "method": "build_v18_bounded_state_solution",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_annotation_state": str(state_path),
            "v18_occlusion_owner_candidates": str(occlusion_candidates_path),
            "v18_occlusion_depth_order_evidence": str(occlusion_depth_path),
        },
        "frame_count": frame_count,
        "raw_frame_count": state.get("raw_frame_count"),
        "frame_count_match": frame_count == state.get("raw_frame_count") == len(frames),
        "bounded_solver": {
            "type": "deterministic_fixed_pass_state_update",
            "passes": [
                "hand_unresolved_gap_classification",
                "object_geometry_scope_projection",
                "contact_mode_projection_from_consistency_depth_evidence",
                "occlusion_owner_candidate_projection_without_owner_acceptance",
                "occlusion_depth_order_evidence_projection_without_resolution",
            ],
            "max_hand_gap_frames": args.max_hand_gap_frames,
            "pose_fill_policy": "never_fill_pose_without_depth_order_owner_and_validated_temporal_model",
            "contact_promotion_policy": "never_promote_image_overlap_to_contact_without_metric_depth_and_geometry_support",
            "object_pose_policy": "visible_surface_or_mask_only_is_not_complete_object_pose",
        },
        "hand_solution_state_counts": dict(sorted(hand_counts.items())),
        "object_solution_state_counts": dict(sorted(object_counts.items())),
        "contact_solution_state_counts": dict(sorted(contact_counts.items())),
        "occlusion_solution_counts": dict(sorted(occlusion_counts.items())),
        "contact_factor_ready_rows": contact_ready_rows,
        "occlusion_owner_candidate_rows": occlusion_owner_candidate_rows,
        "occluder_owner_accepted_rows": occluder_owner_accepted_rows,
        "occlusion_depth_order_resolved_rows": occlusion_depth_order_resolved_rows,
        "occlusion_depth_evidence_candidate_pair_rows": occlusion_depth_evidence_candidate_pair_rows,
        "occlusion_depth_evidence_support_pair_rows": occlusion_depth_evidence_support_pair_rows,
        "occlusion_depth_evidence_contradiction_pair_rows": occlusion_depth_evidence_contradiction_pair_rows,
        "pose_filled_through_occlusion_rows": pose_filled_rows,
        "ready_for_world_status_render": True,
        "frames": frames,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_bounded_state_solution.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [build_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    hand_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    contact_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for report in reports:
        hand_counts.update(report["hand_solution_state_counts"])
        object_counts.update(report["object_solution_state_counts"])
        contact_counts.update(report["contact_solution_state_counts"])
        occlusion_counts.update(report["occlusion_solution_counts"])
    summary = {
        "method": "build_v18_bounded_state_solution",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "frame_count_total": sum(require_int(report.get("frame_count"), "frame_count") for report in reports),
        "all_frame_counts_match_raw": all(bool(report.get("frame_count_match")) for report in reports),
        "hand_solution_state_counts": dict(sorted(hand_counts.items())),
        "object_solution_state_counts": dict(sorted(object_counts.items())),
        "contact_solution_state_counts": dict(sorted(contact_counts.items())),
        "occlusion_solution_counts": dict(sorted(occlusion_counts.items())),
        "contact_factor_ready_rows": sum(require_int(report.get("contact_factor_ready_rows"), "contact ready") for report in reports),
        "occlusion_owner_candidate_rows": sum(require_int(report.get("occlusion_owner_candidate_rows"), "occlusion owner candidates") for report in reports),
        "occluder_owner_accepted_rows": sum(require_int(report.get("occluder_owner_accepted_rows"), "occluder accepted") for report in reports),
        "occlusion_depth_order_resolved_rows": sum(require_int(report.get("occlusion_depth_order_resolved_rows"), "occlusion depth order") for report in reports),
        "occlusion_depth_evidence_candidate_pair_rows": sum(
            require_int(report.get("occlusion_depth_evidence_candidate_pair_rows"), "occlusion depth evidence pair rows") for report in reports
        ),
        "occlusion_depth_evidence_support_pair_rows": sum(
            require_int(report.get("occlusion_depth_evidence_support_pair_rows"), "occlusion depth support pair rows") for report in reports
        ),
        "occlusion_depth_evidence_contradiction_pair_rows": sum(
            require_int(report.get("occlusion_depth_evidence_contradiction_pair_rows"), "occlusion depth contradiction pair rows") for report in reports
        ),
        "pose_filled_through_occlusion_rows": sum(require_int(report.get("pose_filled_through_occlusion_rows"), "pose filled") for report in reports),
        "ready_for_world_status_render": True,
        "cases": [
            {
                "case": report["case"],
                "solution_path": str(args.output_root / str(report["case"]) / "v18_bounded_state_solution.json"),
                "frame_count": report["frame_count"],
                "frame_count_match": report["frame_count_match"],
                "hand_solution_state_counts": report["hand_solution_state_counts"],
                "object_solution_state_counts": report["object_solution_state_counts"],
                "contact_solution_state_counts": report["contact_solution_state_counts"],
                "occlusion_owner_candidate_rows": report["occlusion_owner_candidate_rows"],
                "occluder_owner_accepted_rows": report["occluder_owner_accepted_rows"],
                "occlusion_depth_order_resolved_rows": report["occlusion_depth_order_resolved_rows"],
                "occlusion_depth_evidence_candidate_pair_rows": report["occlusion_depth_evidence_candidate_pair_rows"],
                "occlusion_depth_evidence_support_pair_rows": report["occlusion_depth_evidence_support_pair_rows"],
                "occlusion_depth_evidence_contradiction_pair_rows": report["occlusion_depth_evidence_contradiction_pair_rows"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_bounded_state_solution_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--occlusion-owner-candidates-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_candidates"))
    parser.add_argument("--occlusion-depth-evidence-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-hand-gap-frames", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
