#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import httpx
import numpy as np

from build_object_plan_vlm import load_env_file


VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_track_id": {"type": "string"},
        "frame_idx": {"type": "integer"},
        "accepted": {"type": "boolean"},
        "target_visible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict_reason": {"type": "string"},
        "visual_evidence": {"type": "string"},
        "revised_bbox_xyxy": {"type": "array", "items": {"type": "number"}},
        "replacement_positive_points": {
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
        "negative_points_to_add": {
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
    },
    "required": [
        "target_track_id",
        "frame_idx",
        "accepted",
        "target_visible",
        "confidence",
        "verdict_reason",
        "visual_evidence",
        "revised_bbox_xyxy",
        "replacement_positive_points",
        "negative_points_to_add",
    ],
}


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return data


def encode_image(path: Path, width: int) -> tuple[str, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    height = int(round(width * image.shape[0] / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), (width, height)


def response_text(body: dict) -> str:
    output_text = body.get("output_text")
    if output_text is not None:
        return str(output_text)
    texts = []
    for item in body.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                texts.append(str(part.get("text", "")))
    return "\n".join(texts)


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


def call_vlm(args: argparse.Namespace, raw_url: str, overlay_url: str, image_size: tuple[int, int], base_prompt: dict) -> dict:
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
    width, height = image_size
    text = (
        "Inspect a candidate SAM mask for one egocentric manipulation frame. "
        f"All coordinates must use the raw frame coordinate system, width={width}, height={height}. "
        "The second image shows the candidate mask in yellow over the same frame. "
        "Decide whether the yellow mask covers only the visible surface of the target physical track. "
        "Reject the mask if it includes an adjacent same-category object, a detached peel strip, a hand, countertop, or background region. "
        "If rejected and the target is still separable, return replacement SAM positive points for the corrected target surface, plus negative points on leaked yellow pixels or nearby confusing pixels. "
        "If the target cannot be separated in this frame, set target_visible false. "
        "Do not accept a mask only because it has good category identity; the object geometry pipeline needs one physical object surface.\n\n"
        f"Track id: {args.track_id}\n"
        f"Track description: {args.track_description}\n"
        f"Frame index: {int(args.frame_idx)}\n"
        f"Existing prompt evidence: {json.dumps(base_prompt, ensure_ascii=False)}"
    )
    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_text", "text": "raw source frame"},
                    {"type": "input_image", "image_url": raw_url, "detail": args.detail},
                    {"type": "input_text", "text": "candidate mask overlay"},
                    {"type": "input_image", "image_url": overlay_url, "detail": args.detail},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_mask_repair_verdict",
                "strict": True,
                "schema": VERDICT_SCHEMA,
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
    text_out = response_text(response.json())
    if not text_out:
        raise RuntimeError("Responses API returned no output text")
    verdict = json.loads(text_out)
    if int(verdict["frame_idx"]) != int(args.frame_idx):
        raise RuntimeError(f"VLM returned wrong frame_idx: {verdict['frame_idx']}")
    verdict["replacement_positive_points"] = clamp_points(verdict.get("replacement_positive_points", []), width, height)
    verdict["negative_points_to_add"] = clamp_points(verdict.get("negative_points_to_add", []), width, height)
    bbox = [float(v) for v in verdict.get("revised_bbox_xyxy", [])]
    verdict["revised_bbox_xyxy"] = bbox[:4] if len(bbox) >= 4 else []
    return verdict


def base_prompt(payload: dict, frame_idx: int) -> dict:
    rows = payload.get("point_prompts")
    if not isinstance(rows, list):
        raise RuntimeError("base point prompt file lacks point_prompts list")
    for row in rows:
        if int(row["frame_idx"]) == int(frame_idx):
            return dict(row)
    raise RuntimeError(f"base point prompt file lacks frame {frame_idx}")


def revised_payload(args: argparse.Namespace, base_payload: dict, prompt: dict, verdict: dict, image_size: tuple[int, int]) -> dict:
    row = dict(prompt)
    if bool(verdict["target_visible"]):
        row["target_visible"] = True
        row["positive_points"] = list(verdict["replacement_positive_points"])
        row["negative_points"] = list(row.get("negative_points", [])) + list(verdict["negative_points_to_add"])
        if len(verdict.get("revised_bbox_xyxy", [])) >= 4:
            row["bbox_xyxy"] = list(verdict["revised_bbox_xyxy"][:4])
    else:
        row["target_visible"] = False
        row["positive_points"] = []
        row["negative_points"] = []
    row["visual_evidence"] = str(verdict["visual_evidence"])
    row["confidence"] = float(verdict["confidence"])
    out = dict(base_payload)
    out["backend"] = "VLM mask verification and point repair for SAM"
    out["model"] = args.model
    out["track_id"] = args.track_id
    out["target_track_id"] = args.track_id
    out["description"] = args.track_description
    out["prompt_image_width"] = int(image_size[0])
    out["frames_prompted"] = 1
    out["visible_frames"] = 1 if bool(row["target_visible"]) else 0
    out["point_prompts"] = [row]
    out["batches"] = [{"batch": 0, "frames": [int(args.frame_idx)], "prompt_image_size": [int(image_size[0]), int(image_size[1])]}]
    out["mask_verdict"] = verdict
    return out


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_url, raw_size = encode_image(args.raw_frame, int(args.image_width))
    overlay_url, overlay_size = encode_image(args.candidate_overlay, int(args.image_width))
    if raw_size != overlay_size:
        raise RuntimeError(f"raw and overlay sizes differ after resize: {raw_size} vs {overlay_size}")
    base_payload = load_json(args.base_point_prompts)
    prompt = base_prompt(base_payload, int(args.frame_idx))
    verdict = call_vlm(args, raw_url, overlay_url, raw_size, prompt)
    repaired = revised_payload(args, base_payload, prompt, verdict, raw_size)
    verdict_path = args.output_dir / "mask_repair_verdict_vlm.json"
    prompt_path = args.output_dir / "visual_track_point_prompts_vlm.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    prompt_path.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
    report = {
        "status": "ok",
        "backend": "VLM mask verification and point repair for SAM",
        "verdict": str(verdict_path),
        "repaired_point_prompts": str(prompt_path),
        "accepted": bool(verdict["accepted"]),
        "target_visible": bool(verdict["target_visible"]),
        "replacement_positive_points": int(len(verdict["replacement_positive_points"])),
        "negative_points_added": int(len(verdict["negative_points_to_add"])),
    }
    (args.output_dir / "qc_mask_repair_verdict_vlm_v6.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-frame", type=Path, required=True)
    parser.add_argument("--candidate-overlay", type=Path, required=True)
    parser.add_argument("--base-point-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--track-description", required=True)
    parser.add_argument("--image-width", type=int, default=960)
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
