#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import httpx
import numpy as np

from build_hand_mask_box_evidence_v3 import synthetic_keypoints
from build_object_plan_vlm import load_env_file


HAND_BOX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "prompt_image_width": {"type": "integer"},
        "prompt_image_height": {"type": "integer"},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_idx": {"type": "integer"},
                    "hands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "hand_id": {"type": "string"},
                                "side_hint": {"type": "string", "enum": ["left", "right", "unknown"]},
                                "visibility": {
                                    "type": "string",
                                    "enum": ["clear", "partial", "severe_occlusion", "ambiguous"],
                                },
                                "bbox_xyxy": {"type": "array", "items": {"type": "number"}},
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
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "visual_evidence": {"type": "string"},
                            },
                            "required": [
                                "hand_id",
                                "side_hint",
                                "visibility",
                                "bbox_xyxy",
                                "positive_points",
                                "negative_points",
                                "confidence",
                                "visual_evidence",
                            ],
                        },
                    },
                },
                "required": ["frame_idx", "hands"],
            },
        },
    },
    "required": ["prompt_image_width", "prompt_image_height", "frames"],
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def response_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if not isinstance(output_text, str) or not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    return output_text


def frame_rows(manifest: dict[str, Any], frame_indices: list[int]) -> list[dict[str, Any]]:
    rows = manifest.get("frames")
    if not isinstance(rows, list):
        raise RuntimeError("frame manifest has no frames list")
    by_idx = {int(row["frame_idx"]): row for row in rows if isinstance(row, dict)}
    missing = [idx for idx in frame_indices if idx not in by_idx]
    if missing:
        raise RuntimeError(f"frame manifest is missing frames: {missing}")
    return [by_idx[idx] for idx in frame_indices]


def image_data_url(path: Path, width: int) -> tuple[str, tuple[int, int], tuple[int, int], np.ndarray]:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"failed to read image {path}")
    source_size = (int(frame.shape[1]), int(frame.shape[0]))
    height = int(round(width * frame.shape[0] / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError(f"failed to encode image {path}")
    url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    return url, (width, height), source_size, frame


def clamp_box(box: Any, width: int, height: int) -> list[float]:
    if not isinstance(box, list) or len(box) < 4:
        raise RuntimeError(f"invalid hand box: {box!r}")
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    x0, x1 = sorted((max(0.0, min(float(width - 1), x0)), max(0.0, min(float(width - 1), x1))))
    y0, y1 = sorted((max(0.0, min(float(height - 1), y0)), max(0.0, min(float(height - 1), y1))))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"collapsed hand box after clamping: {box!r}")
    return [x0, y0, x1, y1]


def clamp_points(points: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        raise RuntimeError(f"points must be a list, got {type(points).__name__}")
    out: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            raise RuntimeError(f"point must be an object: {point!r}")
        out.append(
            {
                "x": max(0.0, min(float(width - 1), float(point["x"]))),
                "y": max(0.0, min(float(height - 1), float(point["y"]))),
                "evidence": str(point["evidence"]),
            }
        )
    return out


def scale_box(box: list[float], sx: float, sy: float, width: int, height: int, margin_ratio: float) -> list[float]:
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1) * sx
    cy = 0.5 * (y0 + y1) * sy
    bw = max(1.0, (x1 - x0) * sx)
    bh = max(1.0, (y1 - y0) * sy)
    bw *= 1.0 + 2.0 * margin_ratio
    bh *= 1.0 + 2.0 * margin_ratio
    return [
        max(0.0, cx - 0.5 * bw),
        max(0.0, cy - 0.5 * bh),
        min(float(width - 1), cx + 0.5 * bw),
        min(float(height - 1), cy + 0.5 * bh),
    ]


def scale_points(points: list[dict[str, Any]], sx: float, sy: float) -> list[dict[str, Any]]:
    return [{"x": float(point["x"]) * sx, "y": float(point["y"]) * sy, "evidence": point["evidence"]} for point in points]


def call_responses(args: argparse.Namespace, images: list[tuple[int, str]], prompt_size: tuple[int, int]) -> dict[str, Any]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    base_url = os.environ.get(args.base_url_env, "") if args.base_url_env else args.base_url
    base_url = str(base_url).strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"invalid Responses API base URL: {base_url!r}")
    width, height = prompt_size
    requested = [idx for idx, _ in images]
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Return visible human hand localization for egocentric trash-can manipulation frames. "
                f"Coordinates must use the resized image coordinate system, width={width}, height={height}. "
                "Create one hand record for each visible physical hand region, including partial hands and hands mostly hidden by the trash can lid, bag, sleeve, or frame edge. "
                "The box must enclose visible skin pixels of that hand and enough adjacent wrist/palm context for a hand-pose crop. "
                "Do not put the box around sleeves, the trash bag, the bin, the lid, shoes, floor, cabinet, or plant. "
                "Positive points must lie on visible skin of the hand. Negative points must mark nearby sleeves, lid, bag, or background that a segmenter might merge with the hand. "
                "Use side_hint only when the image evidence supports left or right; use unknown for ambiguous egocentric orientation. "
                "For severe occlusion, return the visible skin region with visibility severe_occlusion and lower confidence. "
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
                "name": "ego_hand_box_prompts",
                "strict": True,
                "schema": HAND_BOX_SCHEMA,
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
    text = response_text(response.json())
    if not text:
        raise RuntimeError("Responses API returned empty output text")
    return json.loads(text)


def validate_and_scale(
    payload: dict[str, Any],
    frame_meta: list[dict[str, Any]],
    prompt_size: tuple[int, int],
    source_size: tuple[int, int],
    margin_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_w, prompt_h = prompt_size
    source_w, source_h = source_size
    sx = float(source_w) / float(prompt_w)
    sy = float(source_h) / float(prompt_h)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"VLM response has no frames list: {payload}")
    by_idx = {int(frame["frame_idx"]): frame for frame in frames if isinstance(frame, dict)}
    measurements: list[dict[str, Any]] = []
    rtmlib_frames: list[dict[str, Any]] = []
    for frame in frame_meta:
        idx = int(frame["frame_idx"])
        raw = by_idx.get(idx)
        if raw is None:
            raise RuntimeError(f"VLM omitted frame {idx}")
        hands = raw.get("hands")
        if not isinstance(hands, list):
            raise RuntimeError(f"VLM frame {idx} hands must be a list")
        rtmlib_hands: list[dict[str, Any]] = []
        for hand_i, hand in enumerate(hands):
            if not isinstance(hand, dict):
                raise RuntimeError(f"VLM frame {idx} hand row must be an object")
            box_prompt = clamp_box(hand.get("bbox_xyxy"), prompt_w, prompt_h)
            box_source = scale_box(box_prompt, sx, sy, source_w, source_h, margin_ratio)
            positives = clamp_points(hand.get("positive_points"), prompt_w, prompt_h)
            negatives = clamp_points(hand.get("negative_points"), prompt_w, prompt_h)
            confidence = float(hand.get("confidence", 0.0))
            if confidence < 0.0 or confidence > 1.0:
                raise RuntimeError(f"VLM confidence outside [0,1] at frame {idx}: {confidence}")
            keypoints = synthetic_keypoints(box_source)
            scores = [confidence for _ in keypoints]
            measurement = {
                "measurement_id": f"vlm_hand_box:{idx}:{hand_i}",
                "frame_idx": idx,
                "entity_type": "hand",
                "entity_id": f"hand_vlm:{idx}:{hand_i}",
                "measurement_type": "vlm_visible_hand_box",
                "source_model": "VLM hand box localization",
                "coordinate_frame": "source_image_px",
                "confidence": confidence,
                "side_hint": str(hand.get("side_hint")),
                "visibility": str(hand.get("visibility")),
                "bbox_xyxy_prompt": box_prompt,
                "bbox_xyxy": box_source,
                "positive_points_prompt": positives,
                "negative_points_prompt": negatives,
                "positive_points": scale_points(positives, sx, sy),
                "negative_points": scale_points(negatives, sx, sy),
                "visual_evidence": str(hand.get("visual_evidence")),
                "crop_contract": "synthetic_keypoints_fill_box_for_hamer_crop_only",
            }
            measurements.append(measurement)
            rtmlib_hands.append(
                {
                    "hand_idx": int(hand_i),
                    "bbox_xyxy": box_source,
                    "keypoints": keypoints,
                    "scores": scores,
                    "mean_score": confidence,
                    "median_score": confidence,
                    "valid_keypoints": len(keypoints),
                    "source_measurement_id": measurement["measurement_id"],
                    "source_model": "VLM hand box localization",
                }
            )
        rtmlib_frames.append(
            {
                "frame_idx": idx,
                "time_s": frame.get("time_s"),
                "hands": rtmlib_hands,
            }
        )
    return measurements, rtmlib_frames


def render_review(frame_images: dict[int, np.ndarray], measurements: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in measurements:
        by_frame.setdefault(int(row["frame_idx"]), []).append(row)
    for idx, frame in frame_images.items():
        canvas = frame.copy()
        for row in by_frame.get(idx, []):
            x0, y0, x1, y1 = [int(round(float(v))) for v in row["bbox_xyxy"]]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 3)
            label = f"{row['measurement_id']} {row['visibility']} {row['confidence']:.2f}"
            cv2.putText(canvas, label, (x0, max(30, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            for point in row["positive_points"]:
                cv2.circle(canvas, (int(round(point["x"])), int(round(point["y"]))), 6, (0, 255, 0), -1)
            for point in row["negative_points"]:
                cv2.circle(canvas, (int(round(point["x"])), int(round(point["y"]))), 6, (0, 0, 255), -1)
        cv2.imwrite(str(output_dir / f"{idx:06d}.jpg"), canvas)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.frame_manifest)
    indices = [int(idx) for idx in args.frame_indices]
    frames = frame_rows(manifest, indices)
    images: list[tuple[int, str]] = []
    frame_images: dict[int, np.ndarray] = {}
    prompt_size: tuple[int, int] | None = None
    source_size: tuple[int, int] | None = None
    for frame in frames:
        idx = int(frame["frame_idx"])
        url, current_prompt_size, current_source_size, image = image_data_url(Path(frame["rgb"]), int(args.image_width))
        if prompt_size is None:
            prompt_size = current_prompt_size
            source_size = current_source_size
        elif prompt_size != current_prompt_size or source_size != current_source_size:
            raise RuntimeError("all prompted frames must share prompt and source size")
        images.append((idx, url))
        frame_images[idx] = image
    if prompt_size is None or source_size is None:
        raise RuntimeError("no frames selected")
    response = call_responses(args, images, prompt_size)
    measurements, rtmlib_frames = validate_and_scale(response, frames, prompt_size, source_size, float(args.box_margin_ratio))
    output_dir = args.output_dir
    write_json(
        output_dir / "vlm_hand_box_measurements.json",
        {
            "status": "ok",
            "method": "build_v17_hand_box_prompts_vlm",
            "model": args.model,
            "frame_manifest": str(args.frame_manifest),
            "frame_indices": indices,
            "prompt_image_size": list(prompt_size),
            "source_image_size": list(source_size),
            "box_margin_ratio": float(args.box_margin_ratio),
            "measurements": measurements,
        },
    )
    write_json(
        output_dir / "rtmlib_from_vlm_hand_boxes.json",
        {
            "status": "ok",
            "method": "build_v17_hand_box_prompts_vlm",
            "video": manifest.get("video"),
            "frames": rtmlib_frames,
            "crop_contract": "synthetic keypoints fill VLM boxes; hamer-full-projection must ignore them for metric translation",
        },
    )
    render_review(frame_images, measurements, output_dir / "review")
    report = {
        "status": "ok",
        "method": "build_v17_hand_box_prompts_vlm",
        "output_dir": str(output_dir),
        "frames": len(indices),
        "hand_box_measurements": len(measurements),
        "rtmlib_json": str(output_dir / "rtmlib_from_vlm_hand_boxes.json"),
        "measurements_json": str(output_dir / "vlm_hand_box_measurements.json"),
        "review_dir": str(output_dir / "review"),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OCC_API_KEY")
    parser.add_argument("--base-url-env", default="OCC_BASEURL")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--detail", default="high")
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--box-margin-ratio", type=float, default=0.12)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
