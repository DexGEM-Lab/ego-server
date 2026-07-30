#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

STATUS = "v18_renderable_annotation_state"
CLAIM = (
    "This artifact joins V18 visibility, fast motion, and consistency evidence into a full-timeline "
    "renderable annotation state. It is an honest status annotation: unresolved hands/objects, rejected "
    "contact, and incomplete object geometry remain explicit."
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


def raw_frame_dir_for(case: str, args: argparse.Namespace) -> Path:
    return existing(args.v16_root / case / "raw_frame_manifest" / "rgb", f"{case} V16 raw rgb frame directory")


def frame_path(raw_frame_dir: Path, frame_idx: int) -> str:
    return str(raw_frame_dir / f"{frame_idx:06d}.jpg")


def v16_manifest_for(case: str, args: argparse.Namespace) -> dict[str, Any]:
    return require_dict(load_json(existing(args.v16_root / case / "v16_full_pipeline_manifest.json", f"{case} V16 manifest")), "V16 manifest")


def timeline_indexes(timeline: dict[str, Any]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_frame in require_list(timeline.get("frames"), "timeline frames"):
        frame = require_dict(raw_frame, "timeline frame")
        frame_idx = require_int(frame.get("frame_idx"), "timeline frame_idx")
        for raw_obj in require_list(frame.get("objects"), "timeline objects"):
            obj = require_dict(raw_obj, "timeline object")
            object_id = require_str(obj.get("object_id"), "timeline object_id")
            by_key[(frame_idx, object_id)] = obj
            by_frame[frame_idx].append(obj)
    return by_key, by_frame


def motion_by_object(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        require_str(row.get("object_id"), "motion object_id"): row
        for row in [require_dict(raw, "motion object row") for raw in require_list(report.get("object_rows"), "motion object rows")]
    }


def consistency_by_frame(report: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], Counter[str]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for raw in require_list(report.get("rows"), "consistency rows"):
        row = require_dict(raw, "consistency row")
        frame_idx = require_int(row.get("frame_idx"), "consistency frame_idx")
        state = require_str(row.get("v18_consistency_state"), "v18_consistency_state")
        by_frame[frame_idx].append(row)
        counts[state] += 1
    return by_frame, counts


def contact_rows_for_object(rows: list[dict[str, Any]], object_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("object_id") != object_id:
            continue
        out.append(
            {
                "hand_side": row.get("hand_side"),
                "v18_consistency_state": row.get("v18_consistency_state"),
                "v18_contact_mode": row.get("v18_contact_mode"),
                "pair_depth_gap_state": row.get("pair_depth_gap_state"),
                "image_overlap_candidate": row.get("image_overlap_candidate"),
                "pair_contact_image_candidate": row.get("pair_contact_image_candidate"),
                "metric_depth_compatible_candidate": row.get("metric_depth_compatible_candidate"),
                "v18_contact_factor_ready": False,
                "blockers": row.get("blockers", []),
            }
        )
    return out


def hand_render_style(visibility_state: str, metric_depth_compatible: bool) -> dict[str, Any]:
    if visibility_state == "visible" and metric_depth_compatible:
        return {"color_bgr": [80, 220, 80], "line_style": "solid", "label_prefix": "HAND observed-depth-ok"}
    if visibility_state == "visible":
        return {"color_bgr": [80, 200, 255], "line_style": "solid", "label_prefix": "HAND observed-depth-unchecked"}
    if visibility_state == "partially_visible":
        return {"color_bgr": [0, 215, 255], "line_style": "solid", "label_prefix": "HAND partial"}
    return {"color_bgr": [160, 160, 160], "line_style": "dashed", "label_prefix": "HAND unresolved"}


def object_render_style(visibility_state: str, physical_state: str, geometry_scope: str) -> dict[str, Any]:
    if visibility_state == "visible" and geometry_scope == "visible_surface_depth_backed":
        return {"color_bgr": [255, 180, 70], "alpha": 0.28, "label_prefix": f"OBJ {physical_state} surface-only"}
    if visibility_state == "visible":
        return {"color_bgr": [255, 120, 220], "alpha": 0.20, "label_prefix": f"OBJ {physical_state} mask-only"}
    if visibility_state == "unresolved":
        return {"color_bgr": [128, 128, 128], "alpha": 0.10, "label_prefix": f"OBJ {physical_state} unresolved"}
    return {"color_bgr": [90, 90, 90], "alpha": 0.08, "label_prefix": f"OBJ {physical_state} inactive"}


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    raw_frame_dir = raw_frame_dir_for(case, args)
    v16_manifest = v16_manifest_for(case, args)
    visibility_path = existing(args.visibility_root / case / "v18_visibility_occlusion_state.json", f"{case} visibility")
    fast_motion_path = existing(args.fast_motion_root / case / "v18_fast_motion_state_report.json", f"{case} fast motion")
    consistency_path = existing(args.consistency_root / case / "v18_consistency_graph_report.json", f"{case} consistency")
    timeline_path = existing(args.timeline_root / case / "v17_multi_object_timeline.json", f"{case} V17 timeline")
    visibility = require_dict(load_json(visibility_path), "visibility")
    fast_motion = require_dict(load_json(fast_motion_path), "fast motion")
    consistency = require_dict(load_json(consistency_path), "consistency")
    timeline = require_dict(load_json(timeline_path), "timeline")
    motion_index = motion_by_object(fast_motion)
    timeline_index, _timeline_by_frame = timeline_indexes(timeline)
    consistency_frame_index, consistency_counts = consistency_by_frame(consistency)
    raw_video = require_dict(visibility.get("raw_video"), "visibility raw_video")
    frame_count = require_int(visibility.get("frame_count"), "visibility frame_count")
    raw_frame_count = require_int(v16_manifest.get("raw_frame_count"), "v16 raw_frame_count")
    if frame_count != raw_frame_count:
        raise RuntimeError(f"{case}: V18 frame_count {frame_count} does not match raw frame_count {raw_frame_count}")
    frames: list[dict[str, Any]] = []
    hand_visibility_counts: Counter[str] = Counter()
    object_visibility_counts: Counter[str] = Counter()
    object_geometry_counts: Counter[str] = Counter()
    contact_state_counts: Counter[str] = Counter(consistency_counts)
    renderable_hand_box_rows = 0
    renderable_object_mask_rows = 0
    unresolved_frame_count = 0
    for raw_frame in require_list(visibility.get("frames"), "visibility frames"):
        frame = require_dict(raw_frame, "visibility frame")
        frame_idx = require_int(frame.get("frame_idx"), "visibility frame_idx")
        if not (raw_frame_dir / f"{frame_idx:06d}.jpg").exists():
            raise RuntimeError(f"{case}: missing raw frame {frame_idx:06d}.jpg in {raw_frame_dir}")
        frame_consistency_rows = consistency_frame_index.get(frame_idx, [])
        hand_rows: list[dict[str, Any]] = []
        object_rows: list[dict[str, Any]] = []
        frame_has_unresolved = False
        for raw_hand in require_list(frame.get("hands"), "frame hands"):
            hand = require_dict(raw_hand, "hand row")
            visibility_state = require_str(hand.get("visibility_state"), "hand visibility_state")
            hand_visibility_counts[visibility_state] += 1
            bbox = hand.get("wilor_bbox_xyxy")
            has_bbox = isinstance(bbox, list) and len(bbox) == 4
            if has_bbox:
                renderable_hand_box_rows += 1
            if visibility_state == "unresolved":
                frame_has_unresolved = True
            hand_rows.append(
                {
                    "frame_idx": frame_idx,
                    "hand_side": hand.get("hand_side"),
                    "visibility_state": visibility_state,
                    "occlusion_state": hand.get("occlusion_state"),
                    "occluder_owner": hand.get("occluder_owner"),
                    "uncertainty_state": hand.get("uncertainty_state"),
                    "bbox_xyxy": bbox,
                    "renderable_bbox": has_bbox,
                    "metric_depth_state": hand.get("metric_depth_state"),
                    "metric_depth_compatible": hand.get("metric_depth_compatible"),
                    "hand_baseline_state": hand.get("hand_baseline_state"),
                    "hand_baseline_acceptance_blockers": hand.get("hand_baseline_acceptance_blockers", []),
                    "hawor_candidate_present": hand.get("hawor_candidate_present", False),
                    "hawor_measurement_available": hand.get("hawor_measurement_available", False),
                    "hawor_evidence_role": hand.get("hawor_evidence_role"),
                    "hawor_projection_residual_px_median": hand.get("hawor_projection_residual_px_median"),
                    "rtmlib_wilor_comparison_available": hand.get("rtmlib_wilor_comparison_available", False),
                    "rtmlib_wilor_median_keypoint_delta_px": hand.get("rtmlib_wilor_median_keypoint_delta_px"),
                    "pose_claim": hand.get("pose_claim"),
                    "pose_filled_through_occlusion": False,
                    "render_style": hand_render_style(visibility_state, bool(hand.get("metric_depth_compatible") is True)),
                }
            )
        for raw_obj in require_list(frame.get("objects"), "frame objects"):
            obj = require_dict(raw_obj, "object row")
            object_id = require_str(obj.get("object_id"), "object_id")
            timeline_obj = timeline_index.get((frame_idx, object_id), {})
            motion = motion_index.get(object_id, {})
            visibility_state = require_str(obj.get("visibility_state"), "object visibility_state")
            physical_state = str(obj.get("model_physical_state_type", "unknown"))
            geometry_scope = str(obj.get("geometry_scope", "no_visible_geometry"))
            object_visibility_counts[visibility_state] += 1
            object_geometry_counts[geometry_scope] += 1
            if visibility_state == "unresolved":
                frame_has_unresolved = True
            mask_path = timeline_obj.get("mask_path") if isinstance(timeline_obj, dict) else None
            bbox = timeline_obj.get("bbox_xyxy") if isinstance(timeline_obj, dict) else None
            has_mask = isinstance(mask_path, str) and Path(mask_path).exists()
            has_bbox = isinstance(bbox, list) and len(bbox) == 4
            if has_mask:
                renderable_object_mask_rows += 1
            contact_rows = contact_rows_for_object(frame_consistency_rows, object_id)
            object_rows.append(
                {
                    "frame_idx": frame_idx,
                    "object_id": object_id,
                    "track_id": obj.get("track_id"),
                    "name": obj.get("name"),
                    "visibility_state": visibility_state,
                    "mask_evidence_state": obj.get("mask_evidence_state"),
                    "occlusion_state": obj.get("occlusion_state"),
                    "occluder_owner": obj.get("occluder_owner"),
                    "model_physical_state_type": physical_state,
                    "fast_motion_state": motion.get("fast_motion_state"),
                    "geometry_scope": geometry_scope,
                    "surface_status": obj.get("surface_status"),
                    "hidden_geometry_state": obj.get("hidden_geometry_state"),
                    "object_pose_claim": obj.get("object_pose_claim"),
                    "object_geometry_complete": False,
                    "object_pose_requirement_met": False,
                    "bbox_xyxy": bbox,
                    "mask_path": mask_path,
                    "renderable_bbox": has_bbox,
                    "renderable_mask": has_mask,
                    "contact_rows": contact_rows,
                    "render_style": object_render_style(visibility_state, physical_state, geometry_scope),
                }
            )
        if frame_has_unresolved:
            unresolved_frame_count += 1
        frames.append(
            {
                "frame_idx": frame_idx,
                "raw_frame_path": frame_path(raw_frame_dir, frame_idx),
                "hands": hand_rows,
                "objects": object_rows,
                "frame_summary": {
                    "visible_or_partial_hands": sum(1 for h in hand_rows if h["visibility_state"] in ("visible", "partially_visible")),
                    "unresolved_hands": sum(1 for h in hand_rows if h["visibility_state"] == "unresolved"),
                    "visible_objects": sum(1 for o in object_rows if o["visibility_state"] == "visible"),
                    "unresolved_objects": sum(1 for o in object_rows if o["visibility_state"] == "unresolved"),
                    "image_contact_rejected_by_metric_depth": sum(
                        1 for row in frame_consistency_rows if row.get("v18_consistency_state") == "image_contact_rejected_by_metric_depth"
                    ),
                    "image_overlap_only": sum(1 for row in frame_consistency_rows if row.get("v18_consistency_state") == "image_overlap_only"),
                    "contact_factor_ready": 0,
                    "has_unresolved_state": frame_has_unresolved,
                },
            }
        )
    report = {
        "method": "build_v18_annotation_state",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "raw_frame_dir": str(raw_frame_dir),
            "v16_manifest": str(args.v16_root / case / "v16_full_pipeline_manifest.json"),
            "v18_visibility_occlusion_state": str(visibility_path),
            "v18_fast_motion_state": str(fast_motion_path),
            "v18_consistency_graph": str(consistency_path),
            "v17_multi_object_timeline": str(timeline_path),
        },
        "raw_video": raw_video,
        "frame_count": frame_count,
        "raw_frame_count": raw_frame_count,
        "frame_count_match": frame_count == raw_frame_count == len(frames),
        "fps": raw_video.get("fps"),
        "duration_s": visibility.get("duration_s"),
        "renderable_status_annotation_ready": True,
        "render_contract": {
            "full_duration": True,
            "same_frame_count_as_raw": frame_count == raw_frame_count == len(frames),
            "object_geometry_claim_policy": "visible_surface_or_mask_only_until_complete_geometry_exists",
            "contact_claim_policy": "image_contact_candidates_rejected_or_unresolved_unless_metric_depth_and_geometry_support_contact",
            "pose_filled_through_occlusion": False,
        },
        "hand_visibility_state_counts": dict(sorted(hand_visibility_counts.items())),
        "object_visibility_state_counts": dict(sorted(object_visibility_counts.items())),
        "object_geometry_scope_counts": dict(sorted(object_geometry_counts.items())),
        "contact_consistency_state_counts": dict(sorted(contact_state_counts.items())),
        "renderable_hand_box_rows": renderable_hand_box_rows,
        "renderable_object_mask_rows": renderable_object_mask_rows,
        "unresolved_frame_count": unresolved_frame_count,
        "contact_factor_ready_rows": 0,
        "pose_filled_through_occlusion_rows": 0,
        "frames": frames,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_annotation_state.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [build_case(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    hand_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    contact_counts: Counter[str] = Counter()
    for report in reports:
        hand_counts.update(report["hand_visibility_state_counts"])
        object_counts.update(report["object_visibility_state_counts"])
        geometry_counts.update(report["object_geometry_scope_counts"])
        contact_counts.update(report["contact_consistency_state_counts"])
    summary = {
        "method": "build_v18_annotation_state",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "frame_count_total": sum(require_int(report.get("frame_count"), "report frame_count") for report in reports),
        "all_frame_counts_match_raw": all(bool(report.get("frame_count_match")) for report in reports),
        "renderable_status_annotation_ready": True,
        "hand_visibility_state_counts": dict(sorted(hand_counts.items())),
        "object_visibility_state_counts": dict(sorted(object_counts.items())),
        "object_geometry_scope_counts": dict(sorted(geometry_counts.items())),
        "contact_consistency_state_counts": dict(sorted(contact_counts.items())),
        "contact_factor_ready_rows": 0,
        "pose_filled_through_occlusion_rows": 0,
        "cases": [
            {
                "case": report["case"],
                "annotation_state_path": str(args.output_root / str(report["case"]) / "v18_annotation_state.json"),
                "frame_count": report["frame_count"],
                "frame_count_match": report["frame_count_match"],
                "renderable_hand_box_rows": report["renderable_hand_box_rows"],
                "renderable_object_mask_rows": report["renderable_object_mask_rows"],
                "unresolved_frame_count": report["unresolved_frame_count"],
                **FALSE_READY,
            }
            for report in reports
        ],
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_annotation_state_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--visibility-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_visibility_occlusion_state"))
    parser.add_argument("--fast-motion-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_fast_motion_state"))
    parser.add_argument("--consistency-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_consistency_graph"))
    parser.add_argument("--timeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v17_multi_object_timeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_annotation_state"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
