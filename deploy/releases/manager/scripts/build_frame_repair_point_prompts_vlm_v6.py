#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import httpx

from build_object_plan_vlm import load_env_file
from build_visual_track_point_prompts_vlm import (
    POINT_SCHEMA,
    image_data_url,
    open_video,
    read_video_frame,
    render_review,
    validate_response,
)


def response_text(body: dict) -> str:
    output_text = body.get("output_text")
    if output_text is not None:
        return str(output_text)
    texts = []
    for item in body.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "\n".join(texts)


def call_responses(args: argparse.Namespace, image_url: str, prompt_size: tuple[int, int]) -> list[dict]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    base_url = args.base_url
    if args.base_url_env:
        base_url = os.environ.get(args.base_url_env, "")
        if not base_url:
            raise RuntimeError(f"{args.base_url_env} is not set")
    base_url = str(base_url).strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"invalid Responses API base URL: {base_url!r}")
    width, height = prompt_size
    text = (
        "Return SAM point prompts for one ambiguous egocentric manipulation frame. "
        f"Coordinates must use the resized image coordinate system, width={width}, height={height}. "
        "Positive points must lie on visible pixels of one continuous physical target only. "
        "Negative points must mark nearby parallel stems, detached sheath strips, hands, sleeves, countertop glare, and background objects that SAM might merge with the target. "
        "If the visible target surface cannot be separated from adjacent similar stems or peel strips, set target_visible false. "
        "The downstream system will reconstruct metric object geometry from the mask; a mask that merges two physical stems is invalid even if both are the same object category. "
        "Use at least four positive points distributed along the target when it is separable, and at least six negative points around confusing adjacent surfaces.\n\n"
        f"Track id: {args.track_id}\n"
        f"Track description: {args.track_description}\n"
        f"Frame index: {args.frame_idx}\n"
        f"Frame-specific ambiguity and prior failed evidence: {args.ambiguity_note}"
    )
    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_text", "text": f"source frame {int(args.frame_idx)}"},
                    {"type": "input_image", "image_url": image_url, "detail": args.detail},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_visual_track_point_prompts",
                "strict": True,
                "schema": POINT_SCHEMA,
            }
        },
    }
    with httpx.Client(timeout=float(args.timeout_s)) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API failed {response.status_code}: {response.text[:1000]}")
    body = response.json()
    text_out = response_text(body)
    if not text_out:
        raise RuntimeError(f"Responses API returned no output text: {json.dumps(body)[:1000]}")
    return validate_response(json.loads(text_out), [int(args.frame_idx)], prompt_size)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(args.clip)
    try:
        frame = read_video_frame(cap, int(args.frame_idx))
    finally:
        cap.release()
    image_url, prompt_size = image_data_url(frame, int(args.image_width))
    prompts = call_responses(args, image_url, prompt_size)
    review = render_review(args, prompts, prompt_size, info)
    payload = {
        "status": "ok",
        "backend": "VLM frame-repair point prompts for SAM",
        "model": args.model,
        "clip": str(args.clip),
        "video": info.__dict__,
        "track_id": args.track_id,
        "target_track_id": args.track_id,
        "description": args.track_description,
        "ambiguity_note": args.ambiguity_note,
        "prompt_image_width": int(args.image_width),
        "frames_prompted": 1,
        "visible_frames": sum(1 for row in prompts if row["target_visible"]),
        "point_prompts": prompts,
        "batches": [{"batch": 0, "frames": [int(args.frame_idx)], "prompt_image_size": list(prompt_size)}],
        "review_video": str(review),
    }
    output_json = args.output_dir / "visual_track_point_prompts_vlm.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"point_prompts", "batches"}}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--track-description", required=True)
    parser.add_argument("--ambiguity-note", required=True)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
