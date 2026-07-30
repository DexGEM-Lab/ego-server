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

from fuse_v1_full_fidelity import read_video_frame
from run_v1_wilor_colmap import load_actions, open_video


PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_summary": {"type": "string"},
        "objects": {
            "type": "array",
            "items": {
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
                "required": ["track_id", "description", "open_vocabulary_prompts", "active_intervals", "physical_notes", "confidence"],
            },
            "minItems": 1,
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["task_summary", "objects", "uncertainties"],
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if sep and key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def frame_indices(actions: list[dict], frame_count: int, max_images: int) -> list[int]:
    candidates = {0, max(0, frame_count - 1)}
    for action in actions:
        start = int(action["start_frame"])
        end = min(frame_count - 1, int(action["end_frame"]) - 1)
        mid = (start + end) // 2
        candidates.update({max(0, start), max(0, mid), max(0, end)})
    ordered = sorted(idx for idx in candidates if 0 <= idx < frame_count)
    if len(ordered) <= max_images:
        return ordered
    keep = [ordered[0], ordered[-1]]
    interior = ordered[1:-1]
    ids = [round(i * (len(interior) - 1) / max(1, max_images - 3)) for i in range(max_images - 2)]
    keep.extend(interior[i] for i in ids)
    return sorted(set(keep))


def image_data_url(frame, width: int) -> str:
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def action_payload(actions: list[dict]) -> list[dict]:
    fields = ("start_frame", "end_frame", "hand", "scene_semantic", "camera_motion", "action", "description")
    return [{key: action.get(key) for key in fields if key in action} for action in actions]


def call_responses(args: argparse.Namespace, actions: list[dict], images: list[tuple[int, str]]) -> dict:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    sampled = [idx for idx, _ in images]
    last_valid_frame = max(sampled + [int(action["end_frame"]) - 1 for action in actions])
    content = [
        {
            "type": "input_text",
            "text": (
                "Return JSON for the manipulated object plan. Use the action metadata and sampled egocentric frames. "
                "Identify the physical object or objects being manipulated, write open-vocabulary prompts suitable for OWLv2/Grounded-SAM style segmentation, "
                "and give frame intervals where each object should be tracked. Do not invent geometry. "
                f"Use inclusive video frame indices for active_intervals; the last valid video frame index is {last_valid_frame}. "
                "If an object changes shape, describe that in physical_notes; the downstream stage reconstructs mesh from masks and depth."
                "\n\nAction metadata:\n"
                + json.dumps(action_payload(actions), ensure_ascii=True)
            ),
        }
    ]
    for idx, url in images:
        content.append({"type": "input_text", "text": f"sampled source frame {idx}"})
        content.append({"type": "input_image", "image_url": url, "detail": args.detail})
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_object_plan",
                "strict": True,
                "schema": PLAN_SCHEMA,
            }
        },
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{args.base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI Responses API failed {response.status_code}: {response.text[:1000]}")
    body = response.json()
    output_text = body.get("output_text")
    if output_text is None:
        texts = []
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        output_text = "\n".join(texts)
    if not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    return json.loads(output_text)


def validate_plan(plan: dict, frame_count: int) -> None:
    if not plan.get("objects"):
        raise RuntimeError("VLM plan contains no objects")
    for obj in plan["objects"]:
        if not obj.get("open_vocabulary_prompts"):
            raise RuntimeError(f"object plan lacks prompts: {obj}")
        for interval in obj.get("active_intervals", []):
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
            if start < 0 or end < start or end >= frame_count:
                raise RuntimeError(f"object interval out of video range: {interval}, frame_count={frame_count}")


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    actions = load_actions(args.actions_json if args.actions_json is not None else args.clip.with_suffix(".json"))
    cap, info = open_video(args.clip)
    indices = frame_indices(actions, info.frame_count, args.max_images)
    images = []
    try:
        for idx in indices:
            frame = read_video_frame(cap, idx)
            images.append((idx, image_data_url(frame, args.image_width)))
    finally:
        cap.release()
    plan = call_responses(args, actions, images)
    validate_plan(plan, info.frame_count)
    out = {
        "status": "ok",
        "backend": "OpenAI Responses vision structured output",
        "model": args.model,
        "clip": str(args.clip),
        "video": info.__dict__,
        "sampled_frames": indices,
        "elapsed_s": time.time() - started,
        "plan": plan,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "sampled_frames": indices, "objects": len(plan["objects"])}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--actions-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-images", type=int, default=14)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--detail", default="high")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
