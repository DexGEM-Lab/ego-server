#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from v20_common import ContractError, ensure_no_gt_in_prediction, load_json, safe_id, write_json


def object_plan_rows(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return [], None
    payload = load_json(path)
    plan = payload.get("plan") if isinstance(payload, dict) and isinstance(payload.get("plan"), dict) else payload
    rows = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"v20_object_plan_has_no_targets: {path}")
    return [row for row in rows if isinstance(row, dict)], payload


def public_roster(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("public_object_model_roster") or manifest.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ContractError("dataset_manifest_has_no_public_object_model_roster")
    return {str(row.get("object_id")): row for row in rows if isinstance(row, dict)}


def make_point(point: dict[str, Any] | list[Any] | tuple[Any, ...]) -> dict[str, float]:
    if isinstance(point, dict):
        return {"x": float(point["x"]), "y": float(point["y"])}
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return {"x": float(point[0]), "y": float(point[1])}
    raise ContractError(f"invalid_point_prompt_point: {point}")


def normalize_prompt_row(row: dict[str, Any], frame_start: int) -> dict[str, Any]:
    positives = row.get("positive_points")
    negatives = row.get("negative_points", [])
    if not isinstance(positives, list) or not positives:
        raise ContractError("object_plan_point_prompt_missing_positive_points")
    if not isinstance(negatives, list):
        raise ContractError("object_plan_point_prompt_negative_points_must_be_list")
    return {
        "frame_idx": int(row.get("frame_idx", frame_start)),
        "target_visible": bool(row.get("target_visible", True)),
        "positive_points": [make_point(point) for point in positives],
        "negative_points": [make_point(point) for point in negatives],
        "prompt_source": str(row.get("prompt_source", "object_plan_visual_evidence_not_eval_ref")),
    }


def build_video(manifest: dict[str, Any], output_video: Path, fps: float) -> dict[str, Any]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("dataset_manifest_has_no_frames")
    first = cv2.imread(str(frames[0]["rgb_path"]), cv2.IMREAD_COLOR)
    if first is None:
        raise ContractError(f"could_not_read_first_rgb: {frames[0]['rgb_path']}")
    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ContractError(f"could_not_open_video_writer: {output_video}")
    try:
        for row in frames:
            image = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ContractError(f"could_not_read_rgb: {row['rgb_path']}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()
    return {"video": str(output_video), "frame_count": len(frames), "width": width, "height": height, "fps": fps}


def write_prompts(manifest: dict[str, Any], object_plan: Path, output_root: Path, prompt_width: int) -> list[dict[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ContractError("dataset_manifest_has_no_frames")
    frame_start = int(frames[0]["frame_index"])
    frame_end = int(frames[-1]["frame_index"])
    roster = public_roster(manifest)
    targets, plan_payload = object_plan_rows(object_plan)
    rows = []
    for target in targets:
        object_id = str(target.get("object_id") or target.get("target_object_id") or target.get("model_object_id") or "")
        if object_id not in roster:
            raise ContractError(f"object_plan_target_not_in_public_model_roster: {object_id}")
        prompt_rows = target.get("point_prompts")
        if not isinstance(prompt_rows, list) or not prompt_rows:
            raise ContractError(f"object_plan_missing_point_prompts_for_target: {object_id}")
        track_id = safe_id(str(target.get("track_id") or object_id))
        payload = {
            "schema": "v20_prediction_visual_object_point_prompts.v0",
            "backend": "agent_visual_inspection_points_from_object_plan_not_eval_refs",
            "track_id": track_id,
            "target_object_id": object_id,
            "description": target.get("description") or roster[object_id].get("object_name") or object_id,
            "prompt_image_width": int(prompt_width),
            "object_plan_payload": {
                "active_intervals": target.get("active_intervals") or [{"start_frame": frame_start, "end_frame": frame_end}],
                "object_plan": str(object_plan),
                "target_selection_source": "object_plan_not_public_model_roster",
                "visual_evidence": "point prompts supplied by object plan from prediction-side visual inspection",
            },
            "point_prompts": [normalize_prompt_row(row, frame_start) for row in prompt_rows],
        }
        ensure_no_gt_in_prediction(payload, f"point_prompt:{track_id}")
        path = output_root / track_id / "object_point_prompts_vlm.json"
        write_json(path, payload)
        rows.append({"object_id": object_id, "track_id": track_id, "prompt_path": str(path)})
    ensure_no_gt_in_prediction(plan_payload, "object_plan")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.dataset_manifest)
    ensure_no_gt_in_prediction(manifest, "dataset_manifest")
    fps = float(args.fps or manifest.get("fps_assumed", 10.0))
    video_info = build_video(manifest, args.output_video, fps)
    prompt_rows: list[dict[str, Any]] = []
    if not args.skip_prompts:
        if args.object_plan is None:
            raise ContractError("missing_object_plan_for_prompt_generation: public_object_model_roster_must_not_drive_prompts")
        prompt_rows = write_prompts(manifest, args.object_plan, args.point_root, int(args.prompt_image_width))
    summary = {
        "status": "ok",
        "method": "build_v20_benchmark_video_and_object_prompts",
        "dataset_manifest": str(args.dataset_manifest),
        "video": video_info,
        "point_root": str(args.point_root) if not args.skip_prompts else None,
        "object_plan": str(args.object_plan) if args.object_plan else None,
        "prompts": prompt_rows,
        "eval_refs_loaded": False,
        "claim_scope": "Builds an MP4 from prediction-side RGB frames and, only when an object plan supplies target point prompts, writes target prompt files. The public model roster is never a target list.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark MP4 and optional target prompts from prediction-side inputs.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--prompt-image-width", type=int, default=640)
    parser.add_argument("--skip-prompts", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
