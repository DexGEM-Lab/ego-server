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

STATUS = "v18_consistency_graph_scaffold"
CLAIM = (
    "This V18 artifact is a bounded consistency/contact reducer over existing visibility, fast object-motion, "
    "and pairwise contact/depth evidence. It exposes blockers and unresolved ownership; it does not solve a "
    "nonlinear graph or fill occluded poses."
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
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label} missing: {path}")
    return path


def visibility_indexes(state: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    hand_index: dict[tuple[int, str], dict[str, Any]] = {}
    object_index: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_frame in require_list(state.get("frames"), "visibility frames"):
        frame = require_dict(raw_frame, "visibility frame")
        frame_idx = require_int(frame.get("frame_idx"), "frame_idx")
        for raw_hand in require_list(frame.get("hands"), "hands"):
            hand = require_dict(raw_hand, "hand row")
            hand_index[(frame_idx, require_str(hand.get("hand_side"), "hand_side"))] = hand
        for raw_obj in require_list(frame.get("objects"), "objects"):
            obj = require_dict(raw_obj, "object row")
            object_index[(frame_idx, require_str(obj.get("object_id"), "object_id"))] = obj
    return hand_index, object_index


def fast_motion_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("object_id"), "fast object_id"): row
        for row in [require_dict(raw, "fast motion row") for raw in require_list(report.get("object_rows"), "object_rows")]
    }


def depth_by_pair(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("pair_contact_variable_id"), "depth pair id"): row
        for row in [require_dict(raw, "depth row") for raw in require_list(report.get("rows"), "depth rows")]
    }


def consistency_state(pair: dict[str, Any], depth: dict[str, Any] | None, hand_visibility: str, object_visibility: str) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if hand_visibility == "unresolved":
        blockers.append("hand_visibility_unresolved")
    if object_visibility == "unresolved":
        blockers.append("object_visibility_unresolved")
    if pair.get("physical_contact_factor_ready") is True:
        return "contact_factor_ready", blockers
    if pair.get("pair_contact_image_candidate") is True:
        if depth is None:
            blockers.append("missing_pair_depth_check")
            blockers.append("object_geometry_incomplete")
            return "image_contact_without_metric_depth", blockers
        if depth.get("metric_depth_compatible_candidate") is True:
            blockers.append("object_geometry_incomplete")
            return "metric_depth_compatible_but_geometry_incomplete", blockers
        blockers.append("metric_depth_contradiction")
        blockers.append("object_geometry_incomplete")
        return "image_contact_rejected_by_metric_depth", blockers
    if pair.get("image_overlap_candidate") is True:
        blockers.append("image_overlap_is_not_contact")
        return "image_overlap_only", blockers
    pair_state = pair.get("pair_contact_state")
    if pair_state == "unobserved_pair":
        blockers.append("pair_unobserved")
        return "unobserved_pair", blockers
    return "no_contact_image_evidence", blockers


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    visibility_path = existing(args.visibility_root / case / "v18_visibility_occlusion_state.json", f"{case} V18 visibility state")
    fast_motion_path = existing(args.fast_motion_root / case / "v18_fast_motion_state_report.json", f"{case} V18 fast motion report")
    pairwise_path = existing(args.pairwise_contact_root / case / "v17_pairwise_contact_state.json", f"{case} V17 pairwise contact")
    depth_path = existing(args.pairwise_depth_root / case / "v17_pairwise_contact_depth_gap.json", f"{case} V17 pairwise depth")
    visibility = require_dict(load_json(visibility_path), f"{case} visibility")
    fast_motion = require_dict(load_json(fast_motion_path), f"{case} fast motion")
    pairwise = require_dict(load_json(pairwise_path), f"{case} pairwise")
    depth_report = require_dict(load_json(depth_path), f"{case} depth")
    hand_index, object_index = visibility_indexes(visibility)
    motion_by_object = fast_motion_by_object(fast_motion)
    depth_rows = depth_by_pair(depth_report)
    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for raw in require_list(pairwise.get("rows"), "pairwise rows"):
        pair = require_dict(raw, "pair row")
        pair_id = require_str(pair.get("pair_contact_variable_id"), "pair id")
        frame_idx = require_int(pair.get("frame_idx"), "pair frame_idx")
        hand_side = require_str(pair.get("hand_side"), "pair hand_side")
        object_id = require_str(pair.get("object_id"), "pair object_id")
        hand = hand_index.get((frame_idx, hand_side), {})
        obj = object_index.get((frame_idx, object_id), {})
        motion = motion_by_object.get(object_id, {})
        hand_visibility = str(hand.get("visibility_state", "unresolved"))
        object_visibility = str(obj.get("visibility_state", "unresolved"))
        depth = depth_rows.get(pair_id)
        state, blockers = consistency_state(pair, depth, hand_visibility, object_visibility)
        state_counts[state] += 1
        blocker_counts.update(blockers)
        rows.append(
            {
                "case": case,
                "pair_contact_variable_id": pair_id,
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "object_id": object_id,
                "hand_visibility_state": hand_visibility,
                "object_visibility_state": object_visibility,
                "object_fast_motion_state": motion.get("fast_motion_state"),
                "object_model_physical_state_type": motion.get("model_physical_state_type"),
                "object_geometry_scope": obj.get("geometry_scope"),
                "pair_contact_state_v17": pair.get("pair_contact_state"),
                "image_overlap_candidate": bool(pair.get("image_overlap_candidate") is True),
                "pair_contact_image_candidate": bool(pair.get("pair_contact_image_candidate") is True),
                "contact_owner_image_supported": bool(pair.get("contact_owner_image_supported") is True),
                "pair_depth_gap_state": depth.get("depth_gap_state") if depth is not None else None,
                "metric_depth_compatible_candidate": bool(depth.get("metric_depth_compatible_candidate") is True) if depth is not None else False,
                "v18_consistency_state": state,
                "v18_contact_factor_ready": False,
                "v18_contact_mode": "unresolved_or_rejected" if state != "no_contact_image_evidence" else "not_contact_by_image_evidence",
                "blockers": sorted(set(blockers + ["complete_object_geometry_absent", "bounded_optimizer_not_run"])),
                "pose_filled_through_occlusion": False,
                "object_geometry_complete": False,
                "object_pose_requirement_met": False,
            }
        )
    report = {
        "method": "build_v18_consistency_graph",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "v18_visibility_occlusion_state": str(visibility_path),
            "v18_fast_motion_state": str(fast_motion_path),
            "v17_pairwise_contact_state": str(pairwise_path),
            "v17_pairwise_contact_depth_gap": str(depth_path),
        },
        "frame_count": require_int(visibility.get("frame_count"), "visibility frame_count"),
        "pair_row_count": len(rows),
        "consistency_state_counts": dict(sorted(state_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "contact_factor_ready_rows": 0,
        "pose_filled_through_occlusion_rows": 0,
        "rows": rows,
        "bounded_optimizer_status": "not_run_scaffold_only",
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_consistency_graph_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(report["consistency_state_counts"])
        blocker_counts.update(report["blocker_counts"])
    summary = {
        "method": "build_v18_consistency_graph",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "pair_row_count": sum(require_int(report.get("pair_row_count"), "pair row count") for report in reports),
        "consistency_state_counts": dict(sorted(state_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "contact_factor_ready_rows": 0,
        "pose_filled_through_occlusion_rows": 0,
        "bounded_optimizer_status": "not_run_scaffold_only",
        "default_path_uses_bundlesdf_or_nerf": False,
        "cases": [
            {
                "case": require_str(report.get("case"), "case"),
                "report_path": str(args.output_root / require_str(report.get("case"), "case") / "v18_consistency_graph_report.json"),
                "pair_row_count": report["pair_row_count"],
                "consistency_state_counts": report["consistency_state_counts"],
                "blocker_counts": report["blocker_counts"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_consistency_graph_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--fast-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_fast_motion_state"))
    parser.add_argument("--pairwise-contact-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_state"))
    parser.add_argument("--pairwise-depth-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_consistency_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
