#!/usr/bin/env python3
"""Build V21 OWLv2 detector keyframes from the agent object plan.

This writes the keyframe list consumed by OWLv2 in the active V21
raw-to-segmentation chain. It writes detector keyframes only and does not read
prior SAM2 tracks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def object_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = root.get("objects") if isinstance(root, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_objects")
    return [row for row in rows if isinstance(row, dict)]


def object_matches(row: dict[str, Any], track_id: str) -> bool:
    candidates = {
        str(row.get("track_id") or ""),
        str(row.get("object_id") or ""),
        str(row.get("target_object_id") or ""),
        str(row.get("model_object_id") or ""),
    }
    normalized = {value.replace("object:", "") for value in candidates}
    return track_id in candidates or track_id in normalized


def frame_count_from_manifest(path: Path) -> int:
    manifest = load_json(path)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError(f"raw_frame_manifest_has_no_frames: {path}")
    return len(frames)


def roi_by_frame(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for roi in row.get("local_mask_rois", []) if isinstance(row.get("local_mask_rois"), list) else []:
        if isinstance(roi, dict) and roi.get("frame_idx") is not None and isinstance(roi.get("bbox_xyxy"), list) and len(roi["bbox_xyxy"]) >= 4:
            out[int(roi["frame_idx"])] = {
                "frame_idx": int(roi["frame_idx"]),
                "bbox_xyxy": [float(v) for v in roi["bbox_xyxy"][:4]],
                "source": str(roi.get("source") or "agent_object_plan_roi"),
            }
    return out


def active_interval_midpoints(row: dict[str, Any], frame_count: int) -> list[int]:
    out: list[int] = []
    for interval in row.get("active_intervals", []) if isinstance(row.get("active_intervals"), list) else []:
        if not isinstance(interval, dict):
            continue
        start = int(interval.get("start_frame", interval.get("frame_start", 0)) or 0)
        end = int(interval.get("end_frame", interval.get("frame_end", frame_count - 1)) or frame_count - 1)
        start = max(0, min(frame_count - 1, start))
        end = max(0, min(frame_count - 1, end))
        if end < start:
            start, end = end, start
        out.append((start + end) // 2)
    return out


def dedupe_frames(frames: list[int], frame_count: int, max_keyframes: int) -> list[int]:
    out: list[int] = []
    for frame_idx in frames:
        frame = max(0, min(frame_count - 1, int(frame_idx)))
        if frame not in out:
            out.append(frame)
        if len(out) >= int(max_keyframes):
            break
    if not out:
        raise ContractError("no_agent_keyframes_selected")
    return out


def select_frames(plan: dict[str, Any], target: dict[str, Any], frame_count: int, args: argparse.Namespace) -> tuple[list[int], str]:
    top = plan.get("reviewed_source_frames")
    if isinstance(top, list) and top:
        return dedupe_frames([int(v) for v in top], frame_count, args.max_keyframes), "object_plan_reviewed_source_frames"
    detector_rows = target.get("owlv2_detector_keyframes")
    if isinstance(detector_rows, list) and detector_rows:
        frames = [int(row["frame_idx"]) for row in detector_rows if isinstance(row, dict) and row.get("frame_idx") is not None]
        if frames:
            return dedupe_frames(frames, frame_count, args.max_keyframes), "object_plan_owlv2_detector_keyframes"
    rois = roi_by_frame(target)
    if rois:
        return dedupe_frames(sorted(rois), frame_count, args.max_keyframes), "object_plan_local_mask_rois"
    frames = active_interval_midpoints(target, frame_count)
    if frames:
        return dedupe_frames(frames, frame_count, args.max_keyframes), "object_plan_active_interval_midpoints"
    return dedupe_frames([frame_count // 2], frame_count, args.max_keyframes), "video_midpoint_fallback_from_object_plan_absence"


def scrub_point_prompt_fields(target: dict[str, Any]) -> None:
    for key in [
        "point_prompts",
        "point_prompt_coordinate_semantics",
        "point_prompt_source_width",
        "point_prompt_source_height",
        "prompt_source",
    ]:
        target.pop(key, None)


def scrub_deprecated_source_inputs(payload: dict[str, Any]) -> None:
    source_inputs = payload.get("source_inputs")
    if not isinstance(source_inputs, dict):
        return
    for key in list(source_inputs):
        lowered = key.lower() + " " + str(source_inputs[key]).lower()
        if "sam2_rgb_baseline" in lowered or "deprecated_seed" in lowered or "segmentation_stable_keyframe_selection" in lowered:
            source_inputs.pop(key, None)


def update_plan(plan: dict[str, Any], track_id: str, frames: list[int], frame_source: str, source_report: Path) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    out["schema"] = str(out.get("schema") or "v21_object_plan_agent.v1")
    out["method"] = "agent_keyframes_for_owlv2_bbox_prompt_sam2"
    out["producer"] = "Pi/Codex agent keyframe selector for V21 OWLv2 bbox-prompt SAM2"
    out["reviewed_source_frames"] = [int(v) for v in frames]
    out["keyframe_selection_policy"] = {
        "method": frame_source,
        "selected_frames": [int(v) for v in frames],
        "source_report": str(source_report),
        "sam2_prompt_type": "approved_owlv2_bbox_only",
    }
    scrub_deprecated_source_inputs(out)
    rows = object_rows(out)
    target = next((row for row in rows if object_matches(row, track_id)), None)
    if target is None:
        raise ContractError(f"track_id_not_found_in_object_plan: {track_id}")
    scrub_point_prompt_fields(target)
    target.pop("local_mask_rois", None)
    target.pop("stable_keyframe_segments", None)
    detector_keyframes = []
    for frame_idx in frames:
        detector_keyframes.append(
            {
                "frame_idx": int(frame_idx),
                "target_visible": True,
                "keyframe_source": frame_source,
                "coordinate_semantics": "source_rgb_pixel_coordinates",
            }
        )
    target["owlv2_detector_keyframes"] = detector_keyframes
    target["measurement_policy"] = "Run OWLv2 on these agent-selected keyframes, approve target boxes, and seed SAM2 only with approved bbox prompts."
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_manifest = Path(args.raw_frame_manifest)
    frame_count = frame_count_from_manifest(raw_manifest)
    plan = load_json(Path(args.object_plan))
    target = next((row for row in object_rows(plan) if object_matches(row, args.track_id)), None)
    if target is None:
        raise ContractError(f"track_id_not_found_in_object_plan: {args.track_id}")
    frames, frame_source = select_frames(plan, target, frame_count, args)
    selected_keyframes = [
        {
            "frame_idx": int(frame_idx),
            "interaction_class": "agent_selected_target_keyframe",
            "segment_frame_start": int(frame_idx),
            "segment_frame_end": int(frame_idx),
            "stability_score": 0.0,
            "keyframe_source": frame_source,
        }
        for frame_idx in frames
    ]
    report = {
        "schema": "v21_agent_keyframes_for_owlv2_bbox.v0",
        "status": "ok",
        "method": "agent_keyframes_for_owlv2_bbox_prompt_sam2",
        "run_root": str(Path(args.run_root).resolve()),
        "track_id": args.track_id,
        "object_plan": str(args.object_plan),
        "raw_frame_manifest": str(raw_manifest),
        "source_frame_manifest": str(args.source_frame_manifest) if args.source_frame_manifest else None,
        "frame_source": frame_source,
        "selected_keyframes": selected_keyframes,
        "frames_csv": ",".join(str(row["frame_idx"]) for row in selected_keyframes),
        "claim_scope": "Agent-selected detector keyframes for OWLv2 bbox proposals. This file contains keyframe indices only, not masks, geometry, object pose, or contact evidence.",
    }
    write_json(args.output, report)
    if args.object_plan_output:
        updated = update_plan(plan, args.track_id, frames, frame_source, Path(args.output))
        write_json(args.object_plan_output, updated)
        report["object_plan_output"] = str(args.object_plan_output)
        write_json(args.output, report)
    print(json.dumps({"status": "ok", "selected_keyframes": selected_keyframes, "output": str(args.output)}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-plan-output", type=Path)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument("--source-frame-manifest", type=Path)
    parser.add_argument("--max-keyframes", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
