#!/usr/bin/env python3
"""Observe complete-video rates without controlling the full-dataset producer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_fps_production_condition import stability_windows

REQUIRED_SERVICES = ("unidepth", "hands.detect", "wilor", "droid", "hawor.track", "hawor.infiller")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def completed_rows(condition_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(condition_root / "dataset_request_events.jsonl")
    completed = [
        row for row in rows
        if row.get("event") == "terminal"
        and row.get("status") == "completed"
        and row.get("measurement_phase") in {"producer", "measurement"}
        and isinstance(row.get("finished_at_unix"), (int, float))
    ]
    return sorted(completed, key=lambda row: float(row["finished_at_unix"]))


def lane_times(rows: list[dict[str, Any]], service: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        lanes = row.get("service_lane_traces")
        lane = lanes.get(service) if isinstance(lanes, dict) else None
        value = lane.get("completed_monotonic_s") if isinstance(lane, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def evaluate(rows: list[dict[str, Any]], *, discard: int, window_size: int, tolerance: float) -> dict[str, Any]:
    options = {"warmup_count": discard, "window_size": window_size, "tolerance": tolerance}
    overall = stability_windows([float(row["finished_at_unix"]) for row in rows], **options)
    services = {service: stability_windows(lane_times(rows, service), **options) for service in REQUIRED_SERVICES}
    missing = {service: len(rows) - len(lane_times(rows, service)) for service in REQUIRED_SERVICES}
    all_stable = bool(overall["stable"] and all(value["stable"] for value in services.values()) and not any(missing.values()))
    return {"discard_completion_count": discard, "overall": overall, "services": services, "missing_lane_markers": missing, "all_stable": all_stable}


def observe(condition_root: Path, *, minimum_discard: int, window_size: int, tolerance: float) -> dict[str, Any]:
    rows = completed_rows(condition_root)
    selected = evaluate(rows, discard=minimum_discard, window_size=window_size, tolerance=tolerance)
    earliest_stable: dict[str, Any] | None = None
    required_after_boundary = 3 * window_size
    for discard in range(minimum_discard, max(minimum_discard, len(rows) - required_after_boundary) + 1):
        candidate = evaluate(rows, discard=discard, window_size=window_size, tolerance=tolerance)
        if candidate["all_stable"]:
            earliest_stable = candidate
            selected = candidate
            break
    return {
        "schema": "ego.annotation.api_ify_video_rate_observer.v1",
        "condition_root": str(condition_root),
        "producer_control": "read_only; this observer never submits, cancels, or stops requests",
        "completed_video_count": len(rows),
        "minimum_discard_completion_count": minimum_discard,
        "window_size": window_size,
        "tolerance": tolerance,
        "selection_rule": "earliest completion boundary at or after minimum_discard where overall and all six service lanes have three stable windows",
        "stable_boundary_found": earliest_stable is not None,
        "selected": selected,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-discard-completions", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()
    if args.minimum_discard_completions < 1:
        raise ValueError("--minimum-discard-completions must be positive")
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive")
    payload = observe(args.condition_root.expanduser().resolve(), minimum_discard=args.minimum_discard_completions, window_size=args.window_size, tolerance=args.tolerance)
    write_json_atomic(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
