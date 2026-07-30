#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from build_object_plan_vlm import load_env_file


COVERAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "expected_label": {"type": "string"},
                    "coverage_status": {"type": "string", "enum": ["covered", "ambiguous", "missing"]},
                    "covered_by_object_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["expected_label", "coverage_status", "covered_by_object_ids", "reason"],
            },
        },
    },
    "required": ["coverage"],
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plan_objects(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = plan_payload.get("plan") if isinstance(plan_payload.get("plan"), dict) else plan_payload
    objects = plan.get("objects") if isinstance(plan, dict) else None
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"object plan has no objects")
    return objects


def call_responses(args: argparse.Namespace, objects: list[dict[str, Any]]) -> dict[str, Any]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    base_url = str(args.base_url).strip()
    if args.base_url_env:
        base_url = os.environ.get(args.base_url_env, "").strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"invalid Responses API base URL: {base_url!r}")
    content = [
        {
            "type": "input_text",
            "text": (
                "Map each expected coarse object label to the object ids in the VLM object plan. "
                "Use semantic equivalence, action context, descriptions, intervals, prompts, and physical notes. "
                "Return covered when the plan clearly includes the expected object under a more specific or differently named id. "
                "Return ambiguous when the plan may include it but the object identity is uncertain. "
                "Return missing when the plan does not contain an object corresponding to the expected label. "
                "Do not create new object ids.\n\n"
                f"Expected labels: {json.dumps(args.expected_labels, ensure_ascii=True)}\n\n"
                f"Object plan entries: {json.dumps(objects, ensure_ascii=True)}"
            ),
        }
    ]
    payload = {
        "model": args.model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ego_expected_object_coverage",
                "strict": True,
                "schema": COVERAGE_SCHEMA,
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
    return json.loads(output_text)


def validate_coverage(payload: dict[str, Any], expected_labels: list[str], objects: list[dict[str, Any]]) -> None:
    rows = payload.get("coverage")
    if not isinstance(rows, list):
        raise RuntimeError("coverage payload has no coverage list")
    by_label = {str(row.get("expected_label")): row for row in rows if isinstance(row, dict)}
    missing_labels = [label for label in expected_labels if label not in by_label]
    if missing_labels:
        raise RuntimeError(f"coverage omitted expected labels: {missing_labels}")
    object_ids = {str(obj.get("track_id")) for obj in objects}
    for row in rows:
        if row.get("coverage_status") not in {"covered", "ambiguous", "missing"}:
            raise RuntimeError(f"invalid coverage status: {row}")
        unknown = [obj_id for obj_id in row.get("covered_by_object_ids", []) if str(obj_id) not in object_ids]
        if unknown:
            raise RuntimeError(f"coverage references unknown object ids: {unknown}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    plan_payload = load_json(args.object_plan)
    objects = plan_objects(plan_payload)
    result = call_responses(args, objects)
    validate_coverage(result, args.expected_labels, objects)
    output = {
        "status": "ok",
        "backend": "OpenAI Responses structured text output",
        "model": args.model,
        "object_plan": str(args.object_plan),
        "expected_labels": args.expected_labels,
        "coverage": result["coverage"],
        "elapsed_s": time.time() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": str(args.output_json), "labels": len(args.expected_labels)}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-plan", type=Path, required=True)
    parser.add_argument("--expected-label", dest="expected_labels", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--base-url-env")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
