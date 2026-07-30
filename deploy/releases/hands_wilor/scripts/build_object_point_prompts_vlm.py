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
from fuse_v1_full_fidelity import DEFAULT_CLIP, load_json, read_video_frame
from run_v1_wilor_colmap import open_video
from segment_object_plan_v2 import intervals_for_object, plan_objects


POINT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_track_id": {"type": "string"},
        "prompt_image_width": {"type": "integer"},
        "prompt_image_height": {"type": "integer"},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_idx": {"type": "integer"},
                    "target_visible": {"type": "boolean"},
                    "positive_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["x", "y", "evidence"],
                        },
                    },
                    "negative_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["x", "y", "evidence"],
                        },
                    },
                    "bbox_xyxy": {"type": "array", "items": {"type": "number"}},
                    "visual_evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "frame_idx",
                    "target_visible",
                    "positive_points",
                    "negative_points",
                    "bbox_xyxy",
                    "visual_evidence",
                    "confidence",
                ],
            },
        },
    },
    "required": ["target_track_id", "prompt_image_width", "prompt_image_height", "frames"],
}


def image_data_url(frame: np.ndarray, width: int) -> tuple[str, tuple[int, int]]:
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), (width, height)


def selected_indices(args: argparse.Namespace, objects: list[dict], frame_count: int) -> list[int]:
    if args.frame_indices:
        indices = sorted({int(part) for raw in args.frame_indices for part in raw.split(",") if part.strip()})
    else:
        obj = objects[args.object_index]
        intervals = intervals_for_object(obj, args.frame_start or 0, args.frame_end if args.frame_end is not None else frame_count - 1)
        indices = []
        for start, end in intervals:
            indices.extend(range(start, end + 1, max(1, int(args.frame_stride))))
    return [idx for idx in indices if 0 <= idx < frame_count]


def clamp_points(points: list[dict], width: int, height: int) -> list[dict]:
    out = []
    for point in points:
        x = float(point["x"])
        y = float(point["y"])
        if not np.isfinite([x, y]).all():
            continue
        q = dict(point)
        q["x"] = float(np.clip(x, 0.0, width - 1.0))
        q["y"] = float(np.clip(y, 0.0, height - 1.0))
        out.append(q)
    return out


def validate_response(payload: dict, object_plan: dict, requested: list[int], prompt_size: tuple[int, int]) -> list[dict]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"VLM response has no frame list: {payload}")
    by_idx = {int(row["frame_idx"]): row for row in frames}
    missing = [idx for idx in requested if idx not in by_idx]
    if missing:
        raise RuntimeError(f"VLM omitted frame point prompts: {missing}")
    width, height = prompt_size
    out = []
    for idx in requested:
        row = dict(by_idx[idx])
        row["frame_idx"] = int(idx)
        row["positive_points"] = clamp_points(row.get("positive_points", []), width, height)
        row["negative_points"] = clamp_points(row.get("negative_points", []), width, height)
        bbox = [float(x) for x in row.get("bbox_xyxy", [])]
        row["bbox_xyxy"] = bbox[:4] if len(bbox) >= 4 else []
        if bool(row["target_visible"]) and len(row["positive_points"]) == 0:
            raise RuntimeError(f"VLM marked target visible but returned no positive points for frame {idx}")
        out.append(row)
    return out


def call_responses(args: argparse.Namespace, object_plan: dict, images: list[tuple[int, str]], prompt_size: tuple[int, int]) -> list[dict]:
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
    requested = [idx for idx, _ in images]
    content = [
        {
            "type": "input_text",
            "text": (
                "Return SAM point prompts for the target object track in the sampled egocentric frames. "
                f"Coordinates must be in the resized image coordinate system, width={width}, height={height}. "
                "Put positive points only on visible pixels of the target object. Put negative points on visually confusing nearby objects, hands, supports, or background. "
                "If the target is hidden or too ambiguous, set target_visible false and return empty positive_points and negative_points. "
                "For any deformable, articulated, or partially occluded target, use visible surface regions that belong to the same physical object instance. "
                "Do not place points on a container, lid, floor, wall, or hand unless that pixel is the target object itself.\n\n"
                f"Target object plan:\n{json.dumps(object_plan, ensure_ascii=True)}\n\n"
                f"Requested frame indices: {requested}"
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
                "name": "ego_object_point_prompts",
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
    output_text = body.get("output_text")
    if output_text is None:
        output_text = "\n".join(
            part.get("text", "")
            for item in body.get("output", [])
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    if not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    parsed = json.loads(output_text)
    return validate_response(parsed, object_plan, requested, prompt_size)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    objects = plan_objects(args.object_plan)
    if args.object_index >= len(objects):
        raise RuntimeError(f"--object-index {args.object_index} out of range for {len(objects)} planned objects")
    object_plan = objects[args.object_index]
    cap, info = open_video(args.clip)
    indices = selected_indices(args, objects, info.frame_count)
    if not indices:
        raise RuntimeError("no source frames selected for point prompting")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "object_point_prompts_vlm.json"
    partial_json = args.output_dir / "object_point_prompts_vlm.partial.json"
    all_prompts: list[dict] = []
    batches = []
    selected_set = set(indices)

    def payload(status: str, elapsed_s: float | None = None) -> dict:
        return {
            "status": status,
            "backend": "VLM object point prompts for SAM",
            "model": args.model,
            "clip": str(args.clip),
            "video": info.__dict__,
            "object_plan": str(args.object_plan),
            "object_index": int(args.object_index),
            "track_id": object_plan["track_id"],
            "description": object_plan["description"],
            "object_plan_record": object_plan,
            "prompt_image_width": int(args.image_width),
            "frames_prompted": len(all_prompts),
            "visible_frames": sum(1 for row in all_prompts if row["target_visible"]),
            "point_prompts": all_prompts,
            "batches": batches,
            "elapsed_s": time.time() - started if elapsed_s is None else elapsed_s,
        }

    if args.resume_partial:
        resume_path = partial_json if partial_json.exists() else output_json if output_json.exists() else None
        if resume_path is not None:
            previous = load_json(resume_path)
            required = {
                "model": args.model,
                "clip": str(args.clip),
                "object_plan": str(args.object_plan),
                "object_index": int(args.object_index),
                "track_id": object_plan["track_id"],
                "prompt_image_width": int(args.image_width),
            }
            mismatches = {
                key: {"expected": expected, "observed": previous.get(key)}
                for key, expected in required.items()
                if previous.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"cannot resume point prompts from incompatible file {resume_path}: {mismatches}")
            all_prompts = [
                dict(row)
                for row in previous.get("point_prompts", [])
                if int(row.get("frame_idx", -1)) in selected_set
            ]
            batches = [
                {**dict(row), "frames": [int(frame) for frame in row.get("frames", []) if int(frame) in selected_set]}
                for row in previous.get("batches", [])
            ]
            batches = [row for row in batches if row["frames"]]
    completed = {int(row["frame_idx"]) for row in all_prompts}
    remaining_indices = [idx for idx in indices if idx not in completed]
    try:
        for start in range(0, len(remaining_indices), max(1, int(args.batch_size))):
            batch = remaining_indices[start : start + int(args.batch_size)]
            images = []
            prompt_size = None
            for idx in batch:
                frame = read_video_frame(cap, idx)
                url, size = image_data_url(frame, int(args.image_width))
                images.append((idx, url))
                prompt_size = size
            if prompt_size is None:
                continue
            prompts = call_responses(args, object_plan, images, prompt_size)
            all_prompts.extend(prompts)
            batches.append({"batch": len(batches), "frames": batch, "prompt_image_size": list(prompt_size)})
            partial_json.write_text(json.dumps(payload("partial"), indent=2), encoding="utf-8")
    finally:
        cap.release()
    final_payload = payload("ok")
    output_json.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in final_payload.items() if k not in {"point_prompts", "batches"}}, indent=2))
    return final_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--frame-indices", nargs="*")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--resume-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
