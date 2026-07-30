#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v20_common import ContractError, ensure_no_gt_in_prediction, safe_id, write_json, load_json


def point_xy(point: Any) -> dict[str, float]:
    if isinstance(point, dict) and "x" in point and "y" in point:
        return {"x": float(point["x"]), "y": float(point["y"])}
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return {"x": float(point[0]), "y": float(point[1])}
    raise ContractError(f"invalid_point_prompt: {point}")


def object_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError("object_plan_has_no_target_objects")
    return [row for row in rows if isinstance(row, dict)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.object_plan)
    ensure_no_gt_in_prediction(payload, "object_plan")
    out_rows = []
    for row in object_rows(payload):
        object_id = str(row.get("object_id") or row.get("target_object_id") or row.get("model_object_id") or "")
        if not object_id:
            raise ContractError("object_plan_row_missing_object_id")
        track_id = safe_id(str(row.get("track_id") or object_id))
        prompts = row.get("point_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ContractError(f"object_plan_target_missing_point_prompts: {object_id}")
        point_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise ContractError(f"invalid_prompt_row_for_object: {object_id}")
            positives = prompt.get("positive_points")
            if not isinstance(positives, list) or not positives:
                raise ContractError(f"missing_positive_points_for_object: {object_id}")
            negatives = prompt.get("negative_points", [])
            if not isinstance(negatives, list):
                raise ContractError(f"negative_points_not_list_for_object: {object_id}")
            point_prompts.append({
                "frame_idx": int(prompt.get("frame_idx", args.frame_start)),
                "target_visible": bool(prompt.get("target_visible", True)),
                "positive_points": [point_xy(point) for point in positives],
                "negative_points": [point_xy(point) for point in negatives],
                "prompt_source": str(prompt.get("prompt_source", "agent_visual_keyframe_sheet")),
            })
        output = {
            "schema": "v20_prediction_visual_object_point_prompts.v0",
            "backend": "agent_visual_inspection_points_from_object_plan_not_eval_refs",
            "track_id": track_id,
            "target_object_id": object_id,
            "description": row.get("description") or object_id,
            "prompt_image_width": int(args.prompt_image_width),
            "object_plan_payload": {
                "object_plan": str(args.object_plan),
                "target_selection_source": "object_plan_not_public_model_roster",
                "active_intervals": row.get("active_intervals", []),
                "physical_model": row.get("physical_model"),
                "physical_notes": row.get("physical_notes"),
            },
            "point_prompts": point_prompts,
        }
        ensure_no_gt_in_prediction(output, f"point_prompts:{object_id}")
        path = args.output_root / track_id / "object_point_prompts_vlm.json"
        write_json(path, output)
        out_rows.append({"object_id": object_id, "track_id": track_id, "path": str(path), "prompt_count": len(point_prompts)})
    summary = {
        "status": "ok",
        "method": "build_v20_infer_point_prompts_from_object_plan",
        "object_plan": str(args.object_plan),
        "output_root": str(args.output_root),
        "targets": out_rows,
        "target_count": len(out_rows),
        "eval_refs_loaded": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SAM2 point prompt files from a V20 infer object plan.")
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--prompt-image-width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
