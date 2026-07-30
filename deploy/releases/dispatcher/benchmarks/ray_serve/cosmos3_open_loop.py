#!/usr/bin/env python3
"""Open-loop Cosmos3 multimodal sweep over distinct inline-image payloads.

The manifest is created next to a benchmark run and names the exact media hashes,
source timestamps, prompts, endpoint, rates, and per-level request count.  Results
retain both HTTP timing and the resident adapter's token/timing/batch trace so a
reviewer can distinguish client latency from model-observed work.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import httpx

from ego_annotation.serving.contracts import Cosmos3Response
from ego_annotation.serving.transport import build_cosmos3_request, parse_cosmos3_response


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]


def wire(value: Any) -> Any:
    """Preserve typed local values and already-decoded multipart mappings alike."""
    return value.to_wire() if hasattr(value, "to_wire") else value


def svg(path: Path, title: str, series: dict[str, list[tuple[float, float]]], xlab: str, ylab: str) -> None:
    width, height, margin_left, margin_bottom = 760, 460, 75, 70
    all_points = [point for points in series.values() for point in points]
    max_x = max((x for x, _ in all_points), default=1.0) or 1.0
    max_y = max((y for _, y in all_points), default=1.0) or 1.0
    palette = ["#1677ff", "#d4380d", "#389e0d", "#722ed1"]
    fragments: list[str] = []
    for index, (name, points) in enumerate(series.items()):
        color = palette[index % len(palette)]
        coords = [
            (margin_left + x / max_x * (width - margin_left - 25), height - margin_bottom - y / max_y * (height - margin_bottom - 40))
            for x, y in points
        ]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>' for x, y in coords)
        fragments.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>{circles}<text x="{margin_left + index * 145}" y="48" fill="{color}" font-size="13">{name}</text>')
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{margin_left}" y="27" font-size="18">{title}</text><line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-25}" y2="{height-margin_bottom}" stroke="black"/><line x1="{margin_left}" y1="{height-margin_bottom}" x2="{margin_left}" y2="45" stroke="black"/>{''.join(fragments)}<text x="{width/2}" y="{height-18}" text-anchor="middle">{xlab}</text><text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">{ylab}</text></svg>''',
        encoding="utf-8",
    )


async def scheduled_request(client: httpx.AsyncClient, endpoint: str, item: dict[str, Any], deadline: float) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    gate = loop.create_future()
    loop.call_at(deadline, gate.set_result, None)
    await gate
    arrived = time.monotonic()
    metadata = {
        "ownership": {key: item[key] for key in ("request_id", "job_id", "item_id", "stage_id", "source_id", "source_timestamp_s")},
        "prompt": item["prompt"],
        "generation": {"max_tokens": 96, "temperature": 0.0, "top_p": 1.0},
    }
    data = Path(item["path"]).read_bytes()
    body, content_type = build_cosmos3_request(metadata, [(data, "image", item["media_type"], 0)])
    try:
        response = await client.post(endpoint, content=body, headers={"Content-Type": content_type})
        completed = time.monotonic()
        parsed = None
        if response.status_code == 200:
            parsed = Cosmos3Response.from_wire(parse_cosmos3_response(response.content, response.headers.get("content-type", "")))
        ok = response.status_code == 200 and parsed is not None and parsed.error is None and parsed.result is not None and bool(parsed.result.text.strip())
        result = parsed.result if ok and parsed is not None else None
        return {
            "request_id": item["request_id"],
            "job_id": item["job_id"],
            "payload_sha256": item["sha256"],
            "scheduled_monotonic_s": deadline,
            "arrived_monotonic_s": arrived,
            "completed_monotonic_s": completed,
            "http_latency_s": completed - arrived,
            "status_code": response.status_code,
            "ok": ok,
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
            "model_revision": result.model_revision if result else None,
            "prompt_tokens": result.prompt_tokens if result else None,
            "completion_tokens": result.completion_tokens if result else None,
            "total_tokens": result.total_tokens if result else None,
            "model_timings": wire(result.timings) if result else None,
            "batch_trace": wire(result.trace) if result else None,
            "error": None if ok else (parsed.error.to_wire() if parsed and parsed.error else response.text[:1000]),
        }
    except Exception as exc:
        completed = time.monotonic()
        return {
            "request_id": item["request_id"], "job_id": item["job_id"], "payload_sha256": item["sha256"],
            "scheduled_monotonic_s": deadline, "arrived_monotonic_s": arrived, "completed_monotonic_s": completed,
            "http_latency_s": completed - arrived, "status_code": None, "ok": False, "error": repr(exc),
        }


def summarize(rows: list[dict[str, Any]], rates: list[float]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for rate in rates:
        level = [row for row in rows if row["offered_rps"] == rate]
        successes = [row for row in level if row["ok"]]
        latencies = [row["http_latency_s"] for row in successes]
        span = max(row["completed_monotonic_s"] for row in level) - min(row["arrived_monotonic_s"] for row in level)
        summary.append({
            "offered_rps": rate,
            "requests": len(level), "successes": len(successes), "errors": len(level) - len(successes),
            "achieved_rps": len(successes) / span if span else 0.0,
            "p50_http_latency_s": percentile(latencies, 0.50), "p95_http_latency_s": percentile(latencies, 0.95), "p99_http_latency_s": percentile(latencies, 0.99),
            "payload_hashes": len({row["payload_sha256"] for row in level}),
            "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in successes),
            "completion_tokens": sum(row.get("completion_tokens") or 0 for row in successes),
            "total_tokens": sum(row.get("total_tokens") or 0 for row in successes),
            "revisions": sorted({row["model_revision"] for row in successes if row.get("model_revision")}),
        })
    return summary


async def run(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["media_items"]
    sweep = manifest["sweep"]
    levels = sweep["offered_rates_rps"]
    per_level = sweep["requests_per_level"]
    required = len(levels) * per_level
    if len(items) < required or len({item["sha256"] for item in items[:required]}) != required:
        raise ValueError("open-loop sweep requires a distinct image hash for every scheduled request")
    raw = root / "raw"
    plots = root / "plots"
    raw.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        for rate in levels:
            level_items = items[offset:offset + per_level]
            offset += per_level
            base = asyncio.get_running_loop().time() + 0.2
            level_rows = await asyncio.gather(*[scheduled_request(client, sweep["endpoint"], item, base + index / rate) for index, item in enumerate(level_items)])
            for row in level_rows:
                row["offered_rps"] = rate
            rows.extend(level_rows)
    (raw / "open_loop_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize(rows, levels)
    (root / "open_loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    successful = [row for row in rows if row["ok"]]
    (root / "token_metrics.json").write_text(json.dumps({"requests": len(successful), "prompt_tokens": sum(row["prompt_tokens"] for row in successful), "completion_tokens": sum(row["completion_tokens"] for row in successful), "total_tokens": sum(row["total_tokens"] for row in successful)}, indent=2) + "\n", encoding="utf-8")
    traces = [row["batch_trace"] for row in successful]
    (root / "batch_trace.json").write_text(json.dumps({"request_count": len(traces), "effective_work_units": [trace["effective_work_units"] for trace in traces], "request_counts": [trace["request_count"] for trace in traces], "forward_counts": [trace["forward_count"] for trace in traces], "model_load_counts": sorted({trace["model_load_count"] for trace in traces}), "traces": traces}, indent=2) + "\n", encoding="utf-8")
    svg(plots / "throughput_vs_offered.svg", "Cosmos3 achieved throughput", {"achieved": [(point["offered_rps"], point["achieved_rps"]) for point in summary]}, "offered requests/s", "achieved requests/s")
    svg(plots / "latency_vs_offered.svg", "Cosmos3 endpoint latency", {"p50": [(point["offered_rps"], point["p50_http_latency_s"] or 0.0) for point in summary], "p95": [(point["offered_rps"], point["p95_http_latency_s"] or 0.0) for point in summary], "p99": [(point["offered_rps"], point["p99_http_latency_s"] or 0.0) for point in summary]}, "offered requests/s", "seconds")
    svg(plots / "tokens_vs_offered.svg", "Cosmos3 tokens completed", {"completion": [(point["offered_rps"], float(point["completion_tokens"])) for point in summary], "total": [(point["offered_rps"], float(point["total_tokens"])) for point in summary]}, "offered requests/s", "tokens per level")
    svg(plots / "batch_work_vs_offered.svg", "Cosmos3 adapter work units", {"request count": [(point["offered_rps"], float(point["requests"])) for point in summary], "successes": [(point["offered_rps"], float(point["successes"])) for point in summary]}, "offered requests/s", "requests")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.root, args.manifest))


if __name__ == "__main__":
    main()
