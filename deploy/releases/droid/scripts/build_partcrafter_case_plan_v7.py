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


PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "num_parts": {"type": "integer", "minimum": 1, "maximum": 16},
                    "visual_reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "num_parts", "visual_reason", "confidence"],
            },
            "minItems": 1,
        },
    },
    "required": ["cases"],
}


def image_data_url(path: Path, width: int) -> tuple[str, tuple[int, int], int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"expected RGBA crop: {path}")
    alpha_pixels = int(np.count_nonzero(image[..., 3] > 0))
    if alpha_pixels <= 0:
        raise RuntimeError(f"empty alpha mask: {path}")
    height = int(round(width * image.shape[0] / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), (width, height), alpha_pixels


def parse_case(raw: str) -> tuple[str, Path, int]:
    parts = raw.split("|")
    if len(parts) != 3:
        raise RuntimeError("--case must have format name|rgba_path|seed")
    name, rgba, seed = [part.strip() for part in parts]
    if not name:
        raise RuntimeError("case name is empty")
    return name, Path(rgba), int(seed)


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


def call_responses(args: argparse.Namespace, cases: list[dict]) -> dict:
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

    text = (
        "For each masked object crop, choose the number of visible semantic-geometric parts for PartCrafter. "
        "The part count is a conditioning variable for a part-level 3D mesh generator, not a category label. "
        "Use visible geometry in the crop: long handles, stems, lids, caps, main bodies, attached tips, or separated sub-surfaces can be parts when they are visually distinct. "
        "Return one integer from 1 to 16 for every case. Do not use object-category rules. "
        "Use conservative counts when evidence is ambiguous, and explain the visual reason."
    )
    content = [{"type": "input_text", "text": text}]
    for case in cases:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"case={case['name']} seed={case['seed']} "
                    f"alpha_pixels={case['alpha_pixels']} prompt_size={case['prompt_size']}"
                ),
            }
        )
        content.append({"type": "input_image", "image_url": case["image_url"], "detail": args.detail})
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "partcrafter_case_plan",
                "strict": True,
                "schema": PLAN_SCHEMA,
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
    return json.loads(text_out)


def validate_plan(plan: dict, expected_names: list[str]) -> list[dict]:
    rows = plan.get("cases")
    if not isinstance(rows, list):
        raise RuntimeError(f"VLM plan has no cases list: {plan}")
    by_name = {str(row.get("name")): row for row in rows}
    missing = [name for name in expected_names if name not in by_name]
    extra = [name for name in by_name if name not in set(expected_names)]
    if missing or extra:
        raise RuntimeError(f"VLM case plan name mismatch: missing={missing}, extra={extra}")
    out = []
    for name in expected_names:
        row = dict(by_name[name])
        n = int(row["num_parts"])
        if not 1 <= n <= 16:
            raise RuntimeError(f"{name}: invalid num_parts {n}")
        conf = float(row["confidence"])
        if not np.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise RuntimeError(f"{name}: invalid confidence {conf}")
        row["name"] = name
        row["num_parts"] = n
        row["confidence"] = conf
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    parsed = [parse_case(raw) for raw in args.case]
    cases = []
    for name, rgba_path, seed in parsed:
        image_url, prompt_size, alpha_pixels = image_data_url(rgba_path, int(args.image_width))
        cases.append(
            {
                "name": name,
                "rgba_path": str(rgba_path),
                "seed": int(seed),
                "image_url": image_url,
                "prompt_size": [int(prompt_size[0]), int(prompt_size[1])],
                "alpha_pixels": int(alpha_pixels),
            }
        )
    plan = call_responses(args, cases)
    rows = validate_plan(plan, [name for name, _, _ in parsed])
    case_args = []
    case_reports = []
    for row, (_, rgba_path, seed) in zip(rows, parsed, strict=True):
        case_args.append(f"{row['name']}|{rgba_path}|{int(seed)}|{int(row['num_parts'])}")
        case_reports.append(
            {
                "name": row["name"],
                "rgba_path": str(rgba_path),
                "seed": int(seed),
                "num_parts": int(row["num_parts"]),
                "visual_reason": str(row["visual_reason"]),
                "confidence": float(row["confidence"]),
            }
        )
    report = {
        "status": "ok",
        "method": "build_partcrafter_case_plan_v7",
        "backend": "Responses API vision structured output",
        "model": args.model,
        "base_url": args.base_url_env or args.base_url,
        "elapsed_s": float(time.time() - started),
        "cases": case_reports,
        "case_args": case_args,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_args is not None:
        args.output_args.parent.mkdir(parents=True, exist_ok=True)
        args.output_args.write_text("\n".join(case_args) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "cases": case_reports}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-args", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--base-url-env")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--detail", default="high")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
