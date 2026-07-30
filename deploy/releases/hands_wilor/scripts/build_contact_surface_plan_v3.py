#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from build_object_plan_vlm import load_env_file
from fuse_v1_full_fidelity import read_video_frame
from run_v1_wilor_colmap import open_video
from segment_object_plan_v2 import plan_objects


SURFACE_OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "track_id": {"type": "string"},
        "description": {"type": "string"},
        "open_vocabulary_prompts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "active_intervals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_frame": {"type": "integer"},
                    "end_frame": {"type": "integer"},
                    "evidence": {"type": "string"},
                },
                "required": ["start_frame", "end_frame", "evidence"],
            },
            "minItems": 1,
        },
        "physical_notes": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "track_id",
        "description",
        "open_vocabulary_prompts",
        "active_intervals",
        "physical_notes",
        "confidence",
    ],
}


SURFACE_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_summary": {"type": "string"},
        "objects": {"type": "array", "items": SURFACE_OBJECT_SCHEMA, "minItems": 1},
        "contact_state_notes": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task_summary", "objects", "contact_state_notes", "uncertainties"],
}


def parse_frame_indices(raw_items: list[str] | None, frame_start: int, frame_end: int, max_images: int) -> list[int]:
    if raw_items:
        indices = sorted({int(part) for raw in raw_items for part in raw.split(",") if part.strip()})
    else:
        if frame_end < frame_start:
            raise RuntimeError(f"invalid frame window {frame_start}:{frame_end}")
        count = min(max_images, frame_end - frame_start + 1)
        if count <= 1:
            indices = [frame_start]
        else:
            indices = [round(frame_start + i * (frame_end - frame_start) / (count - 1)) for i in range(count)]
    return indices


def image_data_url(frame: np.ndarray, width: int) -> str:
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def parent_object(args: argparse.Namespace, objects: list[dict]) -> dict:
    if args.parent_track_id:
        matches = [obj for obj in objects if obj.get("track_id") == args.parent_track_id]
        if not matches:
            raise RuntimeError(f"parent track id not found in object plan: {args.parent_track_id}")
        return matches[0]
    if args.parent_object_index < 0 or args.parent_object_index >= len(objects):
        raise RuntimeError(f"--parent-object-index {args.parent_object_index} out of range for {len(objects)} objects")
    return objects[args.parent_object_index]


def response_text(body: dict) -> str:
    output_text = body.get("output_text")
    if output_text is not None:
        return output_text
    return "\n".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def call_responses(
    args: argparse.Namespace,
    parent: dict,
    source_plan: dict,
    images: list[tuple[int, str]],
    video_info: dict,
) -> dict:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    content = [
        {
            "type": "input_text",
            "text": (
                "Return a contact-surface plan for a dexterous egocentric manipulation annotation pipeline. "
                "The downstream system will segment each returned track with open-vocabulary masks, reconstruct an observed surface mesh from metric depth, "
                "and attach hand-object contact factors only to surfaces whose pixels are actually visible and contact-relevant.\n\n"
                "A valid track is a visually segmentable physical surface or material region of the manipulated object context. "
                "Split regions when they can be contacted independently or have different depth geometry. "
                "Examples of valid distinctions are a top surface, rim/edge, inner liner/sheet, handle, opening lip, or inserted material, but include only regions supported by these frames. "
                "Do not create a mask track for a 'no contact' state; put no-contact observations in contact_state_notes. "
                "Do not merge a container surface and an inserted liner if their visible pixels can be separated. "
                "Do not write category rules or object-family state. Describe what the vision model should segment.\n\n"
                f"Video info: {json.dumps(video_info, ensure_ascii=True)}\n\n"
                f"Frame window: {args.frame_start} to {args.frame_end}\n\n"
                f"Parent manipulated object track:\n{json.dumps(parent, ensure_ascii=True)}\n\n"
                f"Full object plan context:\n{json.dumps(source_plan, ensure_ascii=True)}\n\n"
                "Known geometry evidence for this window: the accepted single pink-lid observed-surface mesh is visually real, "
                "but corrected contact reliability has zero reliable rows. Measured high-score MANO rows have about 11 px median reprojection, "
                "about 173 mm MANO-minus-metric-depth residual, and about 269 mm median hand-to-lid contact gap. "
                "This means the next plan must identify separate visible contact surfaces rather than treating every near-mask hand vertex as contact with the same lid surface."
            ),
        }
    ]
    for idx, url in images:
        content.append({"type": "input_text", "text": f"source frame {idx}"})
        content.append({"type": "input_image", "image_url": url, "detail": args.detail})
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_contact_surface_plan",
                "strict": True,
                "schema": SURFACE_PLAN_SCHEMA,
            }
        },
    }
    with httpx.Client(timeout=float(args.timeout_s)) as client:
        response = client.post(
            f"{args.base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API failed {response.status_code}: {response.text[:1000]}")
    text = response_text(response.json())
    if not text:
        raise RuntimeError("Responses API returned no output text")
    return json.loads(text)


def validate_plan(plan: dict, frame_count: int, frame_start: int, frame_end: int) -> None:
    objects = plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("surface plan contains no objects")
    track_ids = set()
    for obj in objects:
        track_id = obj.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            raise RuntimeError(f"surface track has invalid track_id: {obj}")
        if track_id in track_ids:
            raise RuntimeError(f"duplicate surface track_id: {track_id}")
        track_ids.add(track_id)
        prompts = obj.get("open_vocabulary_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"surface track lacks prompts: {track_id}")
        intervals = obj.get("active_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise RuntimeError(f"surface track lacks active intervals: {track_id}")
        for interval in intervals:
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
            if start < 0 or end < start or end >= frame_count:
                raise RuntimeError(f"surface interval out of video range for {track_id}: {interval}")
            if end < frame_start or start > frame_end:
                raise RuntimeError(f"surface interval outside requested window for {track_id}: {interval}")


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    source = json.loads(args.object_plan.read_text(encoding="utf-8"))
    objects = plan_objects(args.object_plan)
    parent = parent_object(args, objects)
    cap, info = open_video(args.clip)
    indices = parse_frame_indices(args.frame_indices, int(args.frame_start), int(args.frame_end), int(args.max_images))
    if any(idx < 0 or idx >= info.frame_count for idx in indices):
        raise RuntimeError(f"selected frames outside video range: {indices}, frame_count={info.frame_count}")
    images = []
    try:
        for idx in indices:
            frame = read_video_frame(cap, idx)
            images.append((idx, image_data_url(frame, int(args.image_width))))
    finally:
        cap.release()
    surface_plan = call_responses(args, parent, source.get("plan", source), images, info.__dict__)
    validate_plan(surface_plan, info.frame_count, int(args.frame_start), int(args.frame_end))
    output = {
        "status": "ok",
        "backend": "OpenAI Responses vision structured output",
        "model": args.model,
        "clip": str(args.clip),
        "object_plan": str(args.object_plan),
        "parent_track_id": parent["track_id"],
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "sampled_frames": indices,
        "video": info.__dict__,
        "elapsed_s": time.time() - started,
        "plan": {
            "task_summary": surface_plan["task_summary"],
            "objects": surface_plan["objects"],
            "uncertainties": surface_plan["uncertainties"],
        },
        "contact_state_notes": surface_plan["contact_state_notes"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_json": str(args.output_json),
                "sampled_frames": indices,
                "surface_tracks": [obj["track_id"] for obj in surface_plan["objects"]],
            },
            indent=2,
        )
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--parent-track-id")
    parser.add_argument("--parent-object-index", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-indices", nargs="*")
    parser.add_argument("--max-images", type=int, default=14)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
