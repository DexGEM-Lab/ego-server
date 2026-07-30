#!/usr/bin/env python3
"""Select V21 OWLv2 keyframe boxes as SAM2 bbox prompts.

This is the V21 target-box approval boundary between OWLv2 proposals and SAM2
video propagation. It does not generate point prompts and does not fall back to
SAM2 RGB baseline artifacts.

Output:
  measurements/object_candidates/owlv2_bbox_approved_prompts.json
"""
from __future__ import annotations

import argparse
import json
import math
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


def selected_keyframes(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("selected_keyframes")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"keyframe_selection_report_has_no_selected_keyframes: {path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            out.append(dict(row))
        elif isinstance(row, int):
            out.append({"frame_idx": int(row)})
        else:
            raise ContractError(f"invalid_keyframe_selection_row: {row}")
    deduped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in out:
        frame_idx = int(row["frame_idx"])
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        row["frame_idx"] = frame_idx
        deduped.append(row)
    return deduped


def sanitize_box(raw: Any, width: int | None = None, height: int | None = None) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw[:4]]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
        return None
    if width is not None:
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    if height is not None:
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def normalize_frame_map(proposals: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames = proposals.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("owlv2_proposals_have_no_frames")
    out: dict[int, dict[str, Any]] = {}
    for row in frames:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            out[int(row["frame_idx"])] = row
    return out


def usable_detections(frame: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    width = int(frame.get("image_width") or 0) or None
    height = int(frame.get("image_height") or 0) or None
    image_area = float(width * height) if width and height else None
    out: list[dict[str, Any]] = []
    detections = frame.get("detections") if isinstance(frame.get("detections"), list) else []
    for rank, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        score = float(det.get("score", det.get("owlv2_score", 0.0)) or 0.0)
        if score < float(args.min_score):
            continue
        box = sanitize_box(det.get("bbox_xyxy") or det.get("bbox"), width, height)
        if box is None:
            continue
        area_fraction = det.get("box_area_fraction")
        if area_fraction is None and image_area:
            area_fraction = box_area(box) / max(1.0, image_area)
        area_fraction_f = float(area_fraction or 0.0)
        if args.max_area_fraction is not None and area_fraction_f > float(args.max_area_fraction):
            continue
        row = dict(det)
        row.update(
            {
                "bbox_xyxy": [float(v) for v in box],
                "score": score,
                "owlv2_score": score,
                "box_area_fraction": area_fraction_f,
                "source_detection_rank": int(rank),
            }
        )
        out.append(row)
    return out


def select_detection(frame_idx: int, frame: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = usable_detections(frame, args)
    if not candidates:
        return None, []
    candidates.sort(key=lambda det: float(det.get("score", 0.0) or 0.0), reverse=True)
    selected = dict(candidates[0])
    selected["selection_policy"] = "highest_owlv2_score_under_area_cap"
    selected["frame_idx"] = int(frame_idx)
    return selected, candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    proposals_path = Path(args.owlv2_proposals)
    keyframe_path = Path(args.keyframe_selection_report)
    proposals = load_json(proposals_path)
    method = str(proposals.get("method", ""))
    if "groundingdino" in method.lower() or "grounding_dino" in method.lower():
        raise ContractError(f"groundingdino_proposals_not_allowed_for_v21_bbox_prompts: {proposals_path}")
    frame_map = normalize_frame_map(proposals)
    keyframes = selected_keyframes(keyframe_path)

    prompts: list[dict[str, Any]] = []
    frames_out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for keyframe in keyframes:
        frame_idx = int(keyframe["frame_idx"])
        frame = frame_map.get(frame_idx)
        if frame is None:
            rejected.append({"frame_idx": frame_idx, "reason": "missing_owlv2_proposal_frame", "keyframe": keyframe})
            continue
        selected, ranked = select_detection(frame_idx, frame, args)
        if selected is None:
            rejected.append({"frame_idx": frame_idx, "reason": "no_usable_owlv2_detection", "keyframe": keyframe})
            continue
        prompt = {
            "frame_idx": frame_idx,
            "track_id": args.object_id,
            "object_id": args.object_id,
            "target_object_id": args.object_id,
            "bbox_xyxy": [float(v) for v in selected["bbox_xyxy"]],
            "label": selected.get("label"),
            "text_label": selected.get("text_label", selected.get("label")),
            "score": float(selected.get("score", selected.get("owlv2_score", 0.0)) or 0.0),
            "box_area_fraction": float(selected.get("box_area_fraction", 0.0) or 0.0),
            "approval_source": "agent_selected_owlv2_keyframe_bbox",
            "prompt_type": "sam2_box_prompt",
            "selection_policy": selected.get("selection_policy"),
            "keyframe": keyframe,
            "source_detection_rank": selected.get("source_detection_rank"),
            "source_detection": selected,
        }
        prompts.append(prompt)
        approved_detection = dict(selected)
        approved_detection["approval_state"] = "approved_target_bbox_prompt"
        approved_detection["bbox_prompt_for_track_id"] = args.object_id
        frames_out.append(
            {
                "frame_idx": frame_idx,
                "image_path": frame.get("image_path"),
                "image_width": frame.get("image_width"),
                "image_height": frame.get("image_height"),
                "detections": [approved_detection],
                "candidate_count_after_filter": int(len(ranked)),
            }
        )
    if not prompts:
        report = {
            "schema": "v21_owlv2_bbox_approved_prompts.v0",
            "status": "failed",
            "failure_reason": "no_approved_owlv2_bbox_prompts",
            "run_root": str(run_root),
            "object_id": args.object_id,
            "owlv2_proposals": str(proposals_path),
            "keyframe_selection_report": str(keyframe_path),
            "rejected_keyframes": rejected,
        }
        write_json(args.output, report)
        raise ContractError(f"no_approved_owlv2_bbox_prompts: wrote {args.output}")
    report = {
        "schema": "v21_owlv2_bbox_approved_prompts.v0",
        "status": "ok",
        "method": "agent_approved_owlv2_keyframe_bbox_prompts_for_sam2",
        "claim_scope": "Approved OWLv2 keyframe boxes used as SAM2 video box prompts. This is target-box selection evidence only, not a mask, geometry, pose, or contact claim.",
        "run_root": str(run_root),
        "object_id": args.object_id,
        "track_id": args.object_id,
        "owlv2_proposals": str(proposals_path),
        "keyframe_selection_report": str(keyframe_path),
        "min_score": float(args.min_score),
        "max_area_fraction": None if args.max_area_fraction is None else float(args.max_area_fraction),
        "bbox_prompt_frames": [int(row["frame_idx"]) for row in prompts],
        "prompt_count": int(len(prompts)),
        "rejected_keyframes": rejected,
        "prompts": prompts,
        "frames": frames_out,
    }
    write_json(args.output, report)
    print(json.dumps({"status": "ok", "output": str(args.output), "prompt_count": len(prompts)}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--owlv2-proposals", type=Path, required=True)
    parser.add_argument("--keyframe-selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-area-fraction", type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
