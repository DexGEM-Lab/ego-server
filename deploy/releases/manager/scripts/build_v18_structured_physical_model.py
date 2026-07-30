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

PHYSICAL_MODEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "track_id": {"type": "string"},
                    "primary_physical_model": {"type": "string", "enum": ["rigid", "deformable", "articulated", "unknown", "unknown_optically_difficult"]},
                    "pose_model_allowed": {"type": "boolean"},
                    "surface_appearance_changes": {"type": "boolean"},
                    "geometry_changes": {"type": "string", "enum": ["none", "minor_surface_layer_or_texture_change", "nonrigid_deformation", "articulation_or_part_motion", "unknown"]},
                    "requires_part_or_relative_motion_model": {"type": "boolean"},
                    "secondary_deformable_or_surface_component": {"type": "boolean"},
                    "optical_difficulty": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string"},
                    "uncertainty": {"type": "string"},
                },
                "required": [
                    "track_id",
                    "primary_physical_model",
                    "pose_model_allowed",
                    "surface_appearance_changes",
                    "geometry_changes",
                    "requires_part_or_relative_motion_model",
                    "secondary_deformable_or_surface_component",
                    "optical_difficulty",
                    "confidence",
                    "evidence",
                    "uncertainty",
                ],
            },
            "minItems": 1,
        },
    },
    "required": ["objects"],
}

DEFAULT_OBJECT_PLANS = {
    "trash_1050": Path("/data2/ego_annotation_outputs/representative_trash/v2_object_plan/object_plan_vlm.json"),
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v17_object_plan/task5_tomato_960/object_plan_vlm.json"),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def call_responses(args: argparse.Namespace, case: str, object_plan: dict[str, Any]) -> dict[str, Any]:
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    plan = object_plan.get("plan") if isinstance(object_plan.get("plan"), dict) else object_plan
    objects = plan.get("objects") if isinstance(plan.get("objects"), list) else []
    if not objects:
        raise RuntimeError(f"{case}: object plan has no objects")
    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Return structured physical-model fields for each object in this egocentric object plan. "
                            "Do not use keyword matching. Interpret the physical object described by the plan. "
                            "Surface appearance changes on a compact near-rigid object do not by themselves forbid a pose model. "
                            "Do not mark the main object as secondary_deformable_or_surface_component just because texture, skin, or coating appearance changes; detached fragments can be separate deformable objects. "
                            "Use secondary_deformable_or_surface_component only when a flexible/deformable component is physically attached to or part of the manipulated object state, such as wrap or a liner around a rigid item. "
                            "A thin flexible object with substantial shape change is deformable. A hinged or levered mechanism requires part/relative motion. "
                            "Return exactly one output object per input track_id.\n\n"
                            f"Case: {case}\nObject plan JSON:\n{json.dumps(plan, ensure_ascii=True)}"
                        ),
                    }
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "v18_structured_physical_model",
                "strict": True,
                "schema": PHYSICAL_MODEL_SCHEMA,
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
    body = response.json()
    output_text = body.get("output_text")
    if output_text is None:
        chunks: list[str] = []
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    chunks.append(str(part.get("text") or ""))
        output_text = "\n".join(chunks)
    if not output_text:
        raise RuntimeError(f"Responses API returned no output_text: {json.dumps(body)[:1000]}")
    parsed = json.loads(output_text)
    expected = {str(obj.get("track_id")) for obj in objects if obj.get("track_id")}
    got = {str(obj.get("track_id")) for obj in parsed.get("objects", []) if isinstance(obj, dict)}
    if expected != got:
        raise RuntimeError(f"{case}: structured model track_ids mismatch expected={sorted(expected)} got={sorted(got)}")
    return parsed


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    object_plan_path = DEFAULT_OBJECT_PLANS.get(case)
    if object_plan_path is None or not object_plan_path.exists():
        raise RuntimeError(f"{case}: missing object plan {object_plan_path}")
    object_plan = load_json(object_plan_path)
    structured = call_responses(args, case, object_plan)
    rows = []
    for row in structured["objects"]:
        rows.append({**row, "source_object_plan": str(object_plan_path)})
    report = {
        "method": "build_v18_structured_physical_model",
        "status": "ok",
        "case": case,
        "backend": "OpenAI Responses structured output",
        "model": args.model,
        "source_object_plan": str(object_plan_path),
        "object_count": len(rows),
        "objects": rows,
    }
    out_path = args.output_root / case / "v18_structured_physical_model.json"
    write_json(out_path, report)
    return {"case": case, "report_path": str(out_path), "object_count": len(rows)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cases = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_structured_physical_model",
        "status": "ok",
        "case_count": len(cases),
        "cases": cases,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output_root / "v18_structured_physical_model_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_structured_physical_model"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
