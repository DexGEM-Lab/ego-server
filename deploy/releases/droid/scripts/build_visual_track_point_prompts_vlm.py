#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from build_object_plan_vlm import load_env_file


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


@dataclass(frozen=True)
class ClipInfo:
    fps: float
    width: int
    height: int
    frame_count: int


def open_video(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    info = ClipInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.fps <= 0.0 or info.width <= 0 or info.height <= 0 or info.frame_count <= 0:
        raise RuntimeError(f"invalid video metadata: {info}")
    return cap, info


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame_idx}")
    return frame


def image_data_url(frame: np.ndarray, width: int) -> tuple[str, tuple[int, int]]:
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), (width, height)


def frame_indices(args: argparse.Namespace, frame_count: int) -> list[int]:
    if args.frame_indices:
        indices = sorted({int(part) for raw in args.frame_indices for part in raw.split(",") if part.strip()})
    else:
        indices = list(range(int(args.frame_start), int(args.frame_end) + 1, max(1, int(args.frame_stride))))
    out = [idx for idx in indices if 0 <= idx < frame_count]
    if not out:
        raise RuntimeError("no valid frame indices selected")
    return out


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


def validate_response(payload: dict, requested: list[int], prompt_size: tuple[int, int]) -> list[dict]:
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


def call_responses(args: argparse.Namespace, images: list[tuple[int, str]], prompt_size: tuple[int, int]) -> list[dict]:
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
    text = (
        "Return SAM point prompts for one visual track in the sampled egocentric frames. "
        f"Coordinates must use the resized image coordinate system, width={width}, height={height}. "
        "Positive points must lie only on visible pixels of the target visual track. "
        "Negative points should lie on nearby confusing pixels: manipulated objects, sleeves, cloth, the other hand, tools, or background. "
        "If the target is hidden, too ambiguous, or visually merged with another object so that a mask would be misleading, set target_visible false. "
        "The mask target is the visible surface of the described physical track only; do not include occluded parts or adjacent objects.\n\n"
        f"Track id: {args.track_id}\n"
        f"Track description: {args.track_description}\n"
        f"Requested frame indices: {requested}"
    )
    content = [{"type": "input_text", "text": text}]
    for idx, url in images:
        content.append({"type": "input_text", "text": f"source frame {idx}"})
        content.append({"type": "input_image", "image_url": url, "detail": args.detail})
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
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
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    return validate_response(json.loads(text_out), requested, prompt_size)


def render_review(args: argparse.Namespace, prompts: list[dict], prompt_size: tuple[int, int], info: ClipInfo) -> Path:
    cap, _ = open_video(args.clip)
    height = int(round(args.render_width * info.height / info.width))
    output = args.output_dir / "visual_track_point_prompts_review.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (int(args.render_width), height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create {output}")
    scale = np.asarray([args.render_width / prompt_size[0], height / prompt_size[1]], dtype=float)
    try:
        for row in prompts:
            frame_idx = int(row["frame_idx"])
            frame = read_video_frame(cap, frame_idx)
            frame = cv2.resize(frame, (int(args.render_width), height), interpolation=cv2.INTER_AREA)
            if bool(row.get("target_visible")):
                for point in row.get("positive_points", []):
                    p = np.asarray([float(point["x"]), float(point["y"])], dtype=float) * scale
                    cv2.circle(frame, tuple(np.rint(p).astype(int)), 6, (0, 255, 0), -1, cv2.LINE_AA)
                for point in row.get("negative_points", []):
                    p = np.asarray([float(point["x"]), float(point["y"])], dtype=float) * scale
                    cv2.drawMarker(frame, tuple(np.rint(p).astype(int)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
                bbox = row.get("bbox_xyxy", [])
                if len(bbox) >= 4:
                    a = np.asarray([float(bbox[0]), float(bbox[1])], dtype=float) * scale
                    b = np.asarray([float(bbox[2]), float(bbox[3])], dtype=float) * scale
                    cv2.rectangle(frame, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
            label = f"{frame_idx} {args.track_id} conf={float(row.get('confidence', 0.0)):.2f}"
            cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return output


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap, info = open_video(args.clip)
    indices = frame_indices(args, info.frame_count)
    all_prompts: list[dict] = []
    batches = []
    try:
        for batch_id, start in enumerate(range(0, len(indices), max(1, int(args.batch_size)))):
            batch = indices[start : start + int(args.batch_size)]
            images = []
            prompt_size = None
            for idx in batch:
                url, size = image_data_url(read_video_frame(cap, idx), int(args.image_width))
                images.append((idx, url))
                prompt_size = size
            if prompt_size is None:
                raise RuntimeError("empty VLM batch")
            prompts = call_responses(args, images, prompt_size)
            all_prompts.extend(prompts)
            batches.append({"batch": batch_id, "frames": batch, "prompt_image_size": list(prompt_size)})
    finally:
        cap.release()
    review = render_review(args, all_prompts, tuple(batches[0]["prompt_image_size"]), info)
    payload = {
        "status": "ok",
        "backend": "VLM visual-track point prompts for SAM",
        "model": args.model,
        "clip": str(args.clip),
        "video": info.__dict__,
        "track_id": args.track_id,
        "target_track_id": args.track_id,
        "description": args.track_description,
        "prompt_image_width": int(args.image_width),
        "frames_prompted": len(all_prompts),
        "visible_frames": sum(1 for row in all_prompts if row["target_visible"]),
        "point_prompts": all_prompts,
        "batches": batches,
        "review_video": str(review),
        "elapsed_s": time.time() - started,
    }
    output_json = args.output_dir / "visual_track_point_prompts_vlm.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"point_prompts", "batches"}}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--track-description", required=True)
    parser.add_argument("--frame-indices", nargs="*")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.frame_indices is None and (args.frame_start is None or args.frame_end is None):
        raise RuntimeError("pass --frame-indices or both --frame-start and --frame-end")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
