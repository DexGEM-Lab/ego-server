#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "object"


def point_xy(point: Any) -> dict[str, float]:
    if isinstance(point, dict) and "x" in point and "y" in point:
        return {"x": float(point["x"]), "y": float(point["y"])}
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return {"x": float(point[0]), "y": float(point[1])}
    raise ContractError(f"invalid_point: {point}")


def object_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_objects")
    return [row for row in rows if isinstance(row, dict)]


def box_contains_point(box: list[float], point: dict[str, float]) -> bool:
    return float(box[0]) <= float(point["x"]) <= float(box[2]) and float(box[1]) <= float(point["y"]) <= float(box[3])


def box_prompt_points(box: list[float]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    positives = [
        {"x": cx, "y": cy},
        {"x": cx - 0.18 * w, "y": cy},
        {"x": cx + 0.18 * w, "y": cy},
        {"x": cx, "y": cy - 0.18 * h},
        {"x": cx, "y": cy + 0.18 * h},
    ]
    negatives = [
        {"x": x0 - 0.35 * w, "y": cy},
        {"x": x1 + 0.35 * w, "y": cy},
        {"x": cx, "y": y0 - 0.35 * h},
        {"x": cx, "y": y1 + 0.35 * h},
    ]
    return positives, negatives


def load_owlv2_boxes(path: Path | None) -> dict[int, list[float]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out: dict[int, list[float]] = {}
    for frame in payload.get("frames", []) if isinstance(payload.get("frames"), list) else []:
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        detections = frame.get("detections") if isinstance(frame.get("detections"), list) else []
        usable = [d for d in detections if isinstance(d, dict) and isinstance(d.get("bbox_xyxy"), list) and len(d["bbox_xyxy"]) >= 4 and float(d.get("box_area_fraction", 0.0) or 0.0) <= 0.08]
        if not usable:
            continue
        usable.sort(key=lambda d: float(d.get("score", d.get("owlv2_score", 0.0)) or 0.0), reverse=True)
        out[int(frame["frame_idx"])] = [float(v) for v in usable[0]["bbox_xyxy"][:4]]
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    plan = load_json(args.object_plan)
    owlv2_boxes = load_owlv2_boxes(args.owlv2_proposals)
    out_targets: list[dict[str, Any]] = []
    for row in object_rows(plan):
        object_id = str(row.get("object_id") or row.get("target_object_id") or row.get("model_object_id") or "")
        if not object_id:
            raise ContractError("object_row_missing_object_id")
        track_id = safe_id(str(row.get("track_id") or object_id))
        prompts = row.get("point_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ContractError(f"object_missing_point_prompts: {object_id}")
        roi_by_frame = {}
        for roi in row.get("local_mask_rois", []) if isinstance(row.get("local_mask_rois"), list) else []:
            if isinstance(roi, dict) and roi.get("frame_idx") is not None and isinstance(roi.get("bbox_xyxy"), list) and len(roi["bbox_xyxy"]) >= 4:
                roi_by_frame[int(roi["frame_idx"])] = [float(v) for v in roi["bbox_xyxy"][:4]]
        point_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise ContractError(f"invalid_prompt_row: {object_id}")
            positives = prompt.get("positive_points")
            if not isinstance(positives, list) or not positives:
                raise ContractError(f"prompt_missing_positive_points: {object_id}")
            negatives = prompt.get("negative_points", [])
            if not isinstance(negatives, list):
                raise ContractError(f"prompt_negative_points_not_list: {object_id}")
            frame_idx = int(prompt.get("frame_idx", 0))
            positive_points = [point_xy(point) for point in positives]
            negative_points = [point_xy(point) for point in negatives]
            prompt_source = str(prompt.get("prompt_source", "agent_visual_review_object_plan"))
            raw_box = prompt.get("bbox_xyxy") if isinstance(prompt.get("bbox_xyxy"), list) else roi_by_frame.get(frame_idx)
            raw_box_source = "prompt_bbox_xyxy" if isinstance(prompt.get("bbox_xyxy"), list) else "object_plan_local_mask_roi"
            detector_box = owlv2_boxes.get(frame_idx)
            if detector_box is not None and positive_points and not box_contains_point(detector_box, positive_points[0]):
                positive_points, negative_points = box_prompt_points(detector_box)
                raw_box = detector_box
                raw_box_source = "owlv2_bbox_relocalized_prompt"
                prompt_source = prompt_source + "+owlv2_bbox_relocalized_points"
            prompt_row = {
                "frame_idx": frame_idx,
                "target_visible": bool(prompt.get("target_visible", True)),
                "positive_points": positive_points,
                "negative_points": negative_points,
                "prompt_source": prompt_source,
            }
            if isinstance(raw_box, list) and len(raw_box) >= 4:
                prompt_row["bbox_xyxy"] = [float(v) for v in raw_box[:4]]
                prompt_row["bbox_source"] = raw_box_source
            point_prompts.append(prompt_row)
        prompt_image_width = int(row.get("point_prompt_source_width") or args.prompt_image_width)
        output = {
            "schema": "v21_object_point_prompts.v0",
            "mode": "v21_infer",
            "backend": "agent_or_vlm_object_plan_points_for_sam2_rgb_baseline",
            "case_id": input_manifest.get("case_id"),
            "input_manifest": str(args.input_manifest),
            "object_plan": str(args.object_plan),
            "track_id": track_id,
            "target_object_id": object_id,
            "description": row.get("description") or object_id,
            "prompt_image_width": prompt_image_width,
            "coordinate_semantics": "pixel_coordinates_in_prompt_image_width; consumers must scale to their working image size",
            "active_intervals": row.get("active_intervals", []),
            "physical_model": row.get("physical_model"),
            "point_prompts": point_prompts,
            "claim_scope": "Prediction-side target prompts for RGB-only SAM2 baseline. These prompts select the manipulated target; they are not masks, geometry, pose, or GT labels.",
        }
        path = args.output_root / track_id / "object_point_prompts_v21.json"
        write_json(path, output)
        out_targets.append({"object_id": object_id, "track_id": track_id, "prompt_path": str(path), "prompt_count": len(point_prompts)})
    summary = {
        "schema": "v21_object_prompt_summary.v0",
        "status": "ok",
        "method": "build_v21_object_prompts_from_plan",
        "input_manifest": str(args.input_manifest),
        "object_plan": str(args.object_plan),
        "output_root": str(args.output_root),
        "target_count": len(out_targets),
        "targets": out_targets,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V21 SAM2 point-prompt files from a prediction-side object plan.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--prompt-image-width", type=int, default=960)
    parser.add_argument("--owlv2-proposals", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
