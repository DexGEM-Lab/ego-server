#!/usr/bin/env python3
"""Summarize production FPS telemetry and per-item pipeline evidence.

This report is descriptive. Missing or malformed evidence is represented as
null values and explicit ``missing_fields`` entries; no metric is an
acceptance gate.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROUTE_TO_SERVICE = {
    "/unidepth.infer": "unidepth",
    "/hands.detect": "hands.detect",
    "/wilor.reconstruct": "wilor",
    "/droid.create_session": "droid",
    "/droid.push_frame": "droid",
    "/droid.finalize": "droid",
    "/hawor.infer_tracks": "hawor.track",
    "/hawor_infiller.fill": "hawor.infiller",
    "/cosmos3.reason": "cosmos3",
}

EVENT_FIELDS = (
    "video_job_id",
    "video_item_id",
    "received_at_unix",
    "upstream_started_at_unix",
    "upstream_finished_at_unix",
    "finished_at_unix",
    "wait_s",
    "upstream_wall_s",
    "total_wall_s",
    "status",
    "request_bytes",
    "response_bytes",
    "route",
    "configured_limit",
)
LATENCY_FIELDS = ("total_wall_s", "upstream_wall_s", "wait_s")
TRACE_LATENCY_KEYS = (
    "service_wall_s",
    "service_elapsed_s",
    "latency_s",
    "inference_s",
    "forward_s",
    "duration_s",
    "wall_s",
    "batch_elapsed_s",
    "inference_wall_s",
    "request_wall_s",
    "wall_time_s",
    "elapsed_s",
)
BATCH_SIZE_KEYS = ("request_count", "batch_size", "batch_items", "num_requests")
FORWARD_COUNT_KEYS = ("forward_count", "num_forwards", "model_forward_count", "forward_calls")
BATCH_ID_KEYS = ("batch_id", "native_batch_id", "forward_id")


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def iso_from_unix(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: Any) -> float | None:
    numeric = finite(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def load_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], 1
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            malformed += 1
    return rows, malformed


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values if finite(value) is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    values = list(values)
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def stability_windows(completion_times: Iterable[float], *, warmup_count: int = 5, window_size: int = 5, tolerance: float = 0.10) -> dict[str, Any]:
    ordered = sorted(float(value) for value in completion_times if finite(value) is not None)
    effective_warmup = min(warmup_count, len(ordered))
    measured = ordered[effective_warmup:]
    boundary = ordered[effective_warmup - 1] if effective_warmup > 0 else (measured[0] if measured else None)
    windows: list[dict[str, Any]] = []
    for offset in range(0, len(measured) - window_size + 1, window_size):
        values = measured[offset : offset + window_size]
        if boundary is None:
            break
        duration = values[-1] - boundary
        rate = None if duration <= 0.0 else len(values) / duration
        windows.append({"index": len(windows), "completion_count": len(values), "window_started_at_unix": boundary, "first_finished_at_unix": values[0], "last_finished_at_unix": values[-1], "duration_s": duration, "video_req_s": rate})
        boundary = values[-1]
    recent = [float(row["video_req_s"]) for row in windows[-3:] if finite(row.get("video_req_s")) is not None]
    center = percentile(recent, 0.50)
    max_relative_deviation = None if center is None or center <= 0.0 or len(recent) < 3 else max(abs(value - center) / center for value in recent)
    stable = bool(len(recent) == 3 and max_relative_deviation is not None and max_relative_deviation <= tolerance)
    post_warmup_duration_s = None if boundary is None or not measured else measured[-1] - (ordered[effective_warmup - 1] if effective_warmup > 0 else measured[0])
    post_warmup_average_video_s = None if post_warmup_duration_s is None or post_warmup_duration_s <= 0.0 else len(measured) / post_warmup_duration_s
    return {
        "warmup_completion_count": effective_warmup,
        "window_size": window_size,
        "tolerance": tolerance,
        "total_completion_count": len(ordered),
        "post_warmup_completion_count": len(measured),
        "post_warmup_duration_s": post_warmup_duration_s,
        "post_warmup_average_video_s": post_warmup_average_video_s,
        "windows": windows,
        "recent_three_max_relative_deviation": max_relative_deviation,
        "stable": stable,
        "stable_video_req_s": post_warmup_average_video_s if stable else None,
        "recent_window_median_video_s": center,
    }


def metric_value(row: Mapping[str, Any], field: str) -> float | None:
    value = finite(row.get(field))
    if value is not None:
        return value
    if field == "total_wall_s":
        start = first_time(row.get("received_at_unix"), row.get("received_at_iso"))
        end = first_time(row.get("finished_at_unix"), row.get("finished_at_iso"))
        if start is not None and end is not None and end >= start:
            return end - start
    if field == "upstream_wall_s":
        start = first_time(row.get("upstream_started_at_unix"), row.get("upstream_started_at_iso"))
        end = first_time(row.get("upstream_finished_at_unix"), row.get("upstream_finished_at_iso"))
        if start is not None and end is not None and end >= start:
            return end - start
    return None


def first_time(*values: Any) -> float | None:
    for value in values:
        parsed = parse_time(value)
        if parsed is not None:
            return parsed
    return None


def event_time(row: Mapping[str, Any], prefix: str) -> float | None:
    return first_time(row.get(f"{prefix}_at_unix"), row.get(f"{prefix}_at_iso"), row.get(f"{prefix}_at"))


def is_success(row: Mapping[str, Any]) -> bool:
    status = integer(row.get("status"))
    if status is not None:
        return 200 <= status < 300
    return str(row.get("status") or "").lower() in {"ok", "success", "completed", "succeeded"}


def is_backpressure(row: Mapping[str, Any]) -> bool:
    status = integer(row.get("status"))
    text = f"{row.get('status', '')} {row.get('error', '')}".lower()
    return status in {425, 429} or any(token in text for token in ("backpressure", "rate_limit", "rate limited", "resource_exhausted", "too_many_requests"))


def is_rejected(row: Mapping[str, Any]) -> bool:
    status = integer(row.get("status"))
    text = f"{row.get('status', '')} {row.get('error', '')}".lower()
    return status in {408, 425, 429} or any(token in text for token in ("reject", "backpressure", "rate_limit", "rate limited", "resource_exhausted", "too_many_requests"))


def concurrency_max(rows: Iterable[Mapping[str, Any]]) -> int | None:
    points: list[tuple[float, int]] = []
    for row in rows:
        started = event_time(row, "upstream_started")
        finished = event_time(row, "upstream_finished")
        if started is None or finished is None or finished < started:
            continue
        points.extend(((started, 1), (finished, -1)))
    if not points:
        return None
    active = 0
    maximum = 0
    for timestamp, delta in sorted(points, key=lambda item: (item[0], 0 if item[1] < 0 else 1)):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _group_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    received = [event_time(row, "received") for row in rows]
    finished = [event_time(row, "finished") for row in rows]
    received = [value for value in received if value is not None]
    finished = [value for value in finished if value is not None]
    first_received = min(received) if received else None
    last_finished = max(finished) if finished else None
    window_s = None if first_received is None or last_finished is None else max(0.0, last_finished - first_received)
    return {
        "first_received_at_unix": first_received,
        "first_received_at_iso": iso_from_unix(first_received),
        "last_finished_at_unix": last_finished,
        "last_finished_at_iso": iso_from_unix(last_finished),
        "window_s": window_s,
    }


def summarize_events(
    rows: list[dict[str, Any]],
    *,
    name: str,
    service: str | None = None,
    completed_video_ids: set[str] | None = None,
    stability_warmup_count: int = 5,
    stability_window_size: int = 5,
    stability_tolerance: float = 0.10,
) -> dict[str, Any]:
    missing = sorted({field for row in rows for field in EVENT_FIELDS if row.get(field) is None})
    timing = _group_timing(rows)
    total = len(rows)
    success = sum(1 for row in rows if is_success(row))
    rejected = sum(1 for row in rows if is_rejected(row) and not is_success(row))
    failed = total - success - rejected
    window_s = timing["window_s"]
    latencies = {
        field: quantiles(metric_value(row, field) for row in rows if metric_value(row, field) is not None)
        for field in LATENCY_FIELDS
    }
    errors = [
        {"route": row.get("route"), "status": row.get("status"), "error": row.get("error")}
        for row in rows
        if row.get("error") not in (None, "")
    ]
    error_counts = Counter(str(row.get("error")) for row in rows if row.get("error") not in (None, ""))
    offered = None if window_s is None or window_s <= 0.0 else total / window_s
    completed = None if window_s is None or window_s <= 0.0 else success / window_s
    identity_rows = [row for row in rows if row.get("video_item_id") or row.get("video_job_id")]
    identity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_rows:
        identity_groups[str(row.get("video_item_id") or row.get("video_job_id"))].append(row)
    identities = {identity for identity, group in identity_groups.items() if group and all(is_success(row) for row in group)}
    if completed_video_ids is not None:
        identities &= completed_video_ids
    completion_by_identity: dict[str, float] = {}
    video_lane_rows: list[dict[str, float]] = []
    for identity in sorted(identities):
        group = identity_groups[identity]
        received = [event_time(row, "received") for row in group]
        finished = [event_time(row, "finished") for row in group]
        received = [value for value in received if value is not None]
        finished = [value for value in finished if value is not None]
        if not received or not finished:
            continue
        first_received = min(received)
        last_finished = max(finished)
        completion_by_identity[identity] = last_finished
        video_lane_rows.append(
            {
                "total_wall_s": max(0.0, last_finished - first_received),
                "wait_s": sum(metric_value(row, "wait_s") or 0.0 for row in group),
                "upstream_wall_s": sum(metric_value(row, "upstream_wall_s") or 0.0 for row in group),
                "work_unit_count": float(len(group)),
                "first_received_at_unix": first_received,
                "last_finished_at_unix": last_finished,
            }
        )
    video_first_received = min((row["first_received_at_unix"] for row in video_lane_rows), default=None)
    video_last_finished = max((row["last_finished_at_unix"] for row in video_lane_rows), default=None)
    video_window_s = None if video_first_received is None or video_last_finished is None else max(0.0, video_last_finished - video_first_received)
    video_stream_rate = None if video_window_s in (None, 0.0) or not video_lane_rows else len(video_lane_rows) / float(video_window_s)
    video_latencies = {
        field: quantiles(row[field] for row in video_lane_rows)
        for field in LATENCY_FIELDS
    }
    stable_video_stream = stability_windows(
        completion_by_identity.values(),
        warmup_count=stability_warmup_count,
        window_size=stability_window_size,
        tolerance=stability_tolerance,
    )
    target_video_count = (
        len({identity.removesuffix(":item-0") for identity in completed_video_ids})
        if completed_video_ids is not None
        else len(identity_groups)
    )
    return {
        "name": name,
        "service": service,
        "route": None if service is not None else name,
        "aggregation_level": "internal_frame_work_unit",
        "native_batch_trace_level": "separate_traces_section",
        "video_stream": {
            "req_definition": "one complete video stream whose service events carry one video identity",
            "completed_video_count": len(video_lane_rows),
            "completed_video_req_s": video_stream_rate,
            "success_total": {
                "success": len(video_lane_rows),
                "total": target_video_count,
                "ratio": None if target_video_count == 0 else len(video_lane_rows) / target_video_count,
            },
            "identity_fields": ["video_item_id", "video_job_id"],
            "missing_identity_count": sum(1 for row in rows if not (row.get("video_item_id") or row.get("video_job_id"))),
            "completed_video_filter_applied": completed_video_ids is not None,
            "timing_window": {
                "first_received_at_unix": video_first_received,
                "first_received_at_iso": iso_from_unix(video_first_received),
                "last_finished_at_unix": video_last_finished,
                "last_finished_at_iso": iso_from_unix(video_last_finished),
                "window_s": video_window_s,
            },
            "latency_s": video_latencies,
            "work_units_per_video": quantiles(row["work_unit_count"] for row in video_lane_rows),
            "stable_window": stable_video_stream,
        },
        "total_count": total,
        "internal_work_unit_count": total,
        "success_count": success,
        "failed_count": failed,
        "rejected_count": rejected,
        "backpressure_count": sum(1 for row in rows if is_backpressure(row) and not is_success(row)),
        "success_total": {"success": success, "total": total, "ratio": None if total == 0 else success / total},
        "observed_offered_arrival_rate_req_s": offered,
        "offered_arrival_rate_req_s": offered,
        "achieved_success_completion_rate_req_s": completed,
        "success_completion_rate_req_s": completed,
        "latency_s": latencies,
        "p50_total_wall_s": latencies["total_wall_s"]["p50"],
        "p95_total_wall_s": latencies["total_wall_s"]["p95"],
        "p99_total_wall_s": latencies["total_wall_s"]["p99"],
        "p50_upstream_wall_s": latencies["upstream_wall_s"]["p50"],
        "p95_upstream_wall_s": latencies["upstream_wall_s"]["p95"],
        "p99_upstream_wall_s": latencies["upstream_wall_s"]["p99"],
        "p50_wait_s": latencies["wait_s"]["p50"],
        "p95_wait_s": latencies["wait_s"]["p95"],
        "p99_wait_s": latencies["wait_s"]["p99"],
        "max_upstream_concurrency": concurrency_max(rows),
        "configured_limits": sorted({integer(row.get("configured_limit")) for row in rows if integer(row.get("configured_limit")) is not None}),
        "limit_names": sorted({str(row.get("limit_name") or row.get("limit")) for row in rows if row.get("limit_name") or row.get("limit")}),
        "timing_window": timing,
        "errors": errors,
        "error_counts": dict(sorted(error_counts.items())),
        "missing_fields": missing,
    }


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "trace" and isinstance(child, dict):
                yield child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _trace_latency(trace: Mapping[str, Any]) -> float | None:
    for key in TRACE_LATENCY_KEYS:
        value = finite(trace.get(key))
        if value is not None and value >= 0.0:
            return value
    for key in ("latency_ms", "duration_ms", "inference_ms"):
        value = finite(trace.get(key))
        if value is not None and value >= 0.0:
            return value / 1000.0
    return None


def _trace_number(trace: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = finite(trace.get(key))
        if value is not None and value >= 0.0:
            return value
    return None


def _unique_batch_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate response traces before counting native forwards.

    A native batch can serve requests belonging to different videos, so the same
    batch_id may occur in several per-item artifacts. Request latency remains
    per request; batch size and forward counts must be computed once per batch.
    Traces without an identity are retained as independent observations and are
    reported with the corresponding missing provenance.
    """
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, trace in enumerate(traces):
        identity = next((str(trace.get(key)) for key in BATCH_ID_KEYS if trace.get(key) not in (None, "")), None)
        key = ("id", identity) if identity is not None else ("index", str(index))
        if key in seen:
            continue
        seen.add(key)
        unique.append(trace)
    return unique


def _unit_bytes(value: Any, key: str) -> float | None:
    number = finite(value)
    if number is None:
        return None
    lowered = key.lower()
    if "gb" in lowered:
        return number * 1024.0 * 1024.0 * 1024.0
    if "mb" in lowered:
        return number * 1024.0 * 1024.0
    if "kb" in lowered:
        return number * 1024.0
    return number


def _memory_values(value: Any, prefix: str = "") -> dict[str, list[float]]:
    found: dict[str, list[float]] = defaultdict(list)
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            lowered = str(raw_key).lower()
            if any(token in lowered for token in ("memory", "cuda")) and any(token in lowered for token in ("allocated", "reserved", "memory_max", "max_memory")):
                kind = "reserved_bytes" if "reserved" in lowered else "allocated_bytes"
                converted = _unit_bytes(child, str(raw_key))
                if converted is not None:
                    found[kind].append(converted)
            for child_kind, values in _memory_values(child, key).items():
                found[child_kind].extend(values)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            for child_kind, values in _memory_values(child, f"{prefix}[{index}]").items():
                found[child_kind].extend(values)
    return found


def _read_trace_payloads(run_root: Path, model: str) -> list[dict[str, Any]]:
    names = {
        "unidepth": (
            "qc_unidepth_v2.json",
            "qc_unidepth_v2_resident.json",
            "resident_unidepth_worker_report.json",
            "wilor_qc.json",
            "hands_detector_timeline.json",
        ),
        "droid": (
            "droid_shared_geometry.json",
            "droid_qc.json",
            "v22_camera_trajectory_stage.json",
            "droid_keyframes.json",
        ),
    }[model]
    payloads: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for name in names:
        paths.update(run_root.rglob(name))
    paths.update(path for path in run_root.rglob("*.json") if "trace" in path.name.lower() or "worker_report" in path.name.lower())
    for path in sorted(paths):
        payload = load_object(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def summarize_trace(run_roots: Iterable[Path], model: str) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for run_root in run_roots:
        if run_root.is_dir():
            payloads.extend(_read_trace_payloads(run_root, model))
    for payload in payloads:
        traces.extend(_walk(payload))
    batch_traces = _unique_batch_traces(traces)
    latencies = [value for trace in traces for value in [_trace_latency(trace)] if value is not None]
    trace_work_units: list[float] = []
    batch_work_units: list[float] = []
    batch_sizes: list[float] = []
    forward_counts: list[float] = []
    artifact_work_units: list[float] = []
    load_counts: list[int] = []
    memory: dict[str, list[float]] = defaultdict(list)
    for payload in batch_traces:
        batch_size = _trace_number(payload, BATCH_SIZE_KEYS)
        if batch_size is not None:
            batch_sizes.append(batch_size)
        forward_count = _trace_number(payload, FORWARD_COUNT_KEYS)
        if forward_count is not None:
            forward_counts.append(forward_count)
        work_units = _trace_number(payload, ("effective_work_units", "work_units", "rows_inferred", "frame_count", "processed_frames"))
        if work_units is not None:
            batch_work_units.append(work_units if batch_size is None else max(work_units, batch_size))
    for payload in traces:
        work_units = _trace_number(payload, ("effective_work_units", "work_units", "rows_inferred", "frame_count", "processed_frames"))
        if work_units is not None:
            trace_work_units.append(work_units)
    for payload in payloads:
        work_units = _trace_number(payload, ("effective_work_units", "work_units", "rows_inferred", "frame_count", "processed_frames"))
        if work_units is not None:
            artifact_work_units.append(work_units)
        count = integer(payload.get("model_load_count"))
        if count is not None:
            load_counts.append(count)
        for kind, values in _memory_values(payload).items():
            memory[kind].extend(values)
    effective_work_units = sum(batch_work_units) if batch_work_units else (sum(trace_work_units) if trace_work_units else (max(artifact_work_units) if artifact_work_units else None))
    result: dict[str, Any] = {
        "model": model,
        "aggregation_level": "native_batch_trace",
        "trace_count": len(traces),
        "request_trace_count": len(traces),
        "native_batch_count": len(batch_traces),
        "native_batch_size": quantiles(batch_sizes),
        "native_batch_size_total": sum(batch_sizes) if batch_sizes else None,
        "native_forward_count": sum(forward_counts) if forward_counts else None,
        "native_forward_count_observations": len(forward_counts),
        "service_trace_latency_s": quantiles(latencies),
        "trace_latency_s": quantiles(latencies),
        "service_trace_p50_s": percentile(latencies, 0.50),
        "service_trace_p95_s": percentile(latencies, 0.95),
        "service_trace_p99_s": percentile(latencies, 0.99),
        "effective_work_units": effective_work_units,
        "model_load_count": max(load_counts) if load_counts else None,
        "cuda_memory_maxima": {kind: max(values) if values else None for kind, values in sorted(memory.items())},
        "missing_fields": [],
    }
    if not traces:
        result["missing_fields"].append("service_trace")
    if not latencies:
        result["missing_fields"].append("service_trace_latency_s")
    if not batch_sizes:
        result["missing_fields"].append("native_batch_size")
    if not forward_counts:
        result["missing_fields"].append("native_forward_count")
    if effective_work_units is None:
        result["missing_fields"].append("effective_work_units")
    if not load_counts:
        result["missing_fields"].append("model_load_count")
    if not memory:
        result["missing_fields"].append("cuda_memory_maxima")
    if model == "droid":
        caller_timing: dict[str, Any] = {}
        keyframe_counts: list[int] = []
        keyframe_indices: list[int] = []
        invocation: dict[str, Any] = {}
        for payload in payloads:
            if isinstance(payload.get("caller_timing"), dict):
                caller_timing.update(payload["caller_timing"])
            if isinstance(payload.get("droid_caller_timing"), dict):
                caller_timing.update(payload["droid_caller_timing"])
            if isinstance(payload.get("droid_invocation"), dict):
                invocation.update(payload["droid_invocation"])
                invocation_count = integer(payload["droid_invocation"].get("keyframe_count"))
                if invocation_count is not None:
                    keyframe_counts.append(invocation_count)
            count = integer(payload.get("keyframe_count"))
            if count is not None:
                keyframe_counts.append(count)
            keyframes = payload.get("keyframes")
            if isinstance(keyframes, list):
                keyframe_counts.append(len(keyframes))
            elif isinstance(keyframes, dict) and isinstance(keyframes.get("keyframes"), list):
                keyframe_counts.append(len(keyframes["keyframes"]))
            if isinstance(payload.get("keyframe_mapping"), list):
                keyframe_counts.append(len(payload["keyframe_mapping"]))
            for row in payload.get("keyframes", []) if isinstance(payload.get("keyframes"), list) else []:
                if isinstance(row, dict) and integer(row.get("source_frame_idx")) is not None:
                    keyframe_indices.append(integer(row["source_frame_idx"]))  # type: ignore[arg-type]
        result["caller_timing"] = caller_timing or None
        result["droid_invocation"] = invocation or None
        result["keyframe_count"] = max(keyframe_counts) if keyframe_counts else None
        result["keyframe_source_frame_indices"] = sorted(set(keyframe_indices)) or None
        for field in ("caller_timing", "keyframe_count"):
            if result[field] is None:
                result["missing_fields"].append(field)
    return result


def raw_duration(raw: Mapping[str, Any]) -> float | None:
    video = raw.get("video") if isinstance(raw.get("video"), dict) else {}
    for value in (video.get("duration_s"), raw.get("duration_s")):
        duration = finite(value)
        if duration is not None and duration > 0.0:
            return duration
    fps = finite(raw.get("fps")) or finite(video.get("fps"))
    count = integer(raw.get("frame_count")) or integer(video.get("frame_count"))
    return None if fps is None or fps <= 0.0 or count is None else count / fps


def _artifact_count(run_root: Path, relative_names: tuple[str, ...], keys: tuple[str, ...]) -> int | None:
    for relative in relative_names:
        payload = load_object(run_root / relative)
        if payload is None:
            continue
        for key in keys:
            value = integer(payload.get(key))
            if value is not None:
                return value
    return None


def summarize_item(item_root: Path, item_result: dict[str, Any]) -> dict[str, Any]:
    run_root_value = item_result.get("run_root")
    run_root = Path(str(run_root_value)).expanduser() if run_root_value else None
    if run_root is not None and not run_root.is_absolute():
        run_root = (item_root / run_root).resolve()
    if run_root is not None and not run_root.is_dir():
        run_root = None
    missing: list[str] = []
    pipeline = load_object(run_root / "annotation_pipeline_manifest.json") if run_root else None
    raw = load_object(run_root / "input" / "raw_frame_manifest" / "manifest.json") if run_root else None
    if run_root is None:
        missing.append("run_root")
    if pipeline is None:
        missing.append("annotation_pipeline_manifest")
    if raw is None:
        missing.append("raw_frame_manifest")
    steps = pipeline.get("steps") if isinstance(pipeline, dict) and isinstance(pipeline.get("steps"), list) else []
    stage_times: dict[str, float | None] = {}
    stage_status: Counter[str] = Counter()
    stage_counts: dict[str, dict[str, Any]] = {}
    for row in steps:
        if not isinstance(row, dict):
            continue
        name = str(row.get("step") or "unknown")
        status = str(row.get("status") or "unknown")
        stage_times[name] = finite(row.get("elapsed_s"))
        stage_status[status] += 1
        stage_counts[name] = {"count": 1, "status": status, "elapsed_s": stage_times[name]}
    duration = raw_duration(raw or {})
    pipeline_row = item_result.get("pipeline") if isinstance(item_result.get("pipeline"), dict) else {}
    pipeline_wall = finite(pipeline_row.get("elapsed_s"))
    if pipeline_wall is None:
        pipeline_wall = finite(item_result.get("pipeline_elapsed_s"))
    selected: dict[str, int | None] = {
        "unidepth": _artifact_count(run_root, ("measurements/depth_candidates/unidepth_v2/qc_unidepth_v2.json", "measurements/depth_candidates/unidepth_v2_sparse/qc_unidepth_v2.json"), ("frame_count", "processed_frames")) if run_root else None,
        "droid": _artifact_count(run_root, ("measurements/camera_trajectory/droid_full_frame/droid_shared_geometry.json", "measurements/camera_trajectory/droid_sparse/droid_shared_geometry.json"), ("processed_frames", "frame_count")) if run_root else None,
    }
    selected["source"] = integer(raw.get("frame_count")) if raw else None
    source_fps = {name: (None if value is None or duration is None or duration <= 0.0 else value / duration) for name, value in selected.items()}
    realtime = None if pipeline_wall is None or pipeline_wall <= 0.0 or duration is None else duration / pipeline_wall
    for field, value in (("pipeline_wall_s", pipeline_wall), ("source_duration_s", duration), ("selected_frame_counts", selected)):
        if value is None:
            missing.append(field)
    if selected["unidepth"] is None:
        missing.append("selected_frame_counts.unidepth")
    if selected["droid"] is None:
        missing.append("selected_frame_counts.droid")
    if pipeline_wall is None:
        missing.append("total_pipeline_wall_s")
    if realtime is None:
        missing.append("realtime_factor")
    return {
        "item_index": integer(item_result.get("item_index")) if integer(item_result.get("item_index")) is not None else integer(item_root.name.rsplit("_", 1)[-1]),
        "case_id": item_result.get("case_id"),
        "status": item_result.get("status"),
        "run_root": str(run_root) if run_root else run_root_value,
        "stage_counts": {"total": sum(stage_status.values()), "by_status": dict(sorted(stage_status.items())), "by_stage": stage_counts},
        "stage_times_s": stage_times,
        "selected_frame_counts": selected,
        "source_time_fps": source_fps,
        "source_duration_s": duration,
        "realtime_factor": realtime,
        "total_pipeline_wall_s": pipeline_wall,
        "missing_fields": sorted(set(missing)),
    }


def _discover_items(condition_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(condition_root.glob("items/item_*/item_result.json"))
    if (condition_root / "item_result.json").is_file():
        paths.append(condition_root / "item_result.json")
    result: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        payload = load_object(path)
        if payload is not None:
            result.append((path.parent, payload))
    return result


def summarize_top_level_video_events(
    rows: list[dict[str, Any]],
    *,
    stability_warmup_count: int = 5,
    stability_window_size: int = 5,
    stability_tolerance: float = 0.10,
) -> dict[str, Any]:
    all_terminals = [row for row in rows if str(row.get("event") or "") == "terminal"]
    terminals = [row for row in all_terminals if str(row.get("measurement_phase") or "measurement") in {"measurement", "producer"}]
    drain_terminals = [row for row in all_terminals if str(row.get("measurement_phase") or "measurement") == "drain_after_stability"]
    completed = [row for row in terminals if str(row.get("status") or "").lower() == "completed"]
    completion_times = [first_time(row.get("finished_at_unix"), row.get("finished_at")) for row in completed]
    completion_times = [value for value in completion_times if value is not None]
    elapsed = [value for row in terminals if (value := finite(row.get("elapsed_s"))) is not None]
    measurement_tokens = {str(row.get("request_token")) for row in terminals if row.get("request_token")}
    measurement_starts = [
        first_time(row.get("request_started_at_unix"), row.get("request_started_at"))
        for row in rows
        if str(row.get("event") or "") == "request_started"
        and (not measurement_tokens or str(row.get("request_token")) in measurement_tokens)
    ]
    measurement_starts = [value for value in measurement_starts if value is not None]
    queue_wait = [
        value
        for row in rows
        if str(row.get("event") or "") == "request_started"
        and (not measurement_tokens or str(row.get("request_token")) in measurement_tokens)
        and (value := finite(row.get("client_queue_wait_s"))) is not None
    ]
    timing = _group_timing([{"received_at_unix": min(measurement_starts) if measurement_starts else (min(completion_times) if completion_times else None), "finished_at_unix": max(completion_times) if completion_times else None}])
    stable = stability_windows(
        completion_times,
        warmup_count=stability_warmup_count,
        window_size=stability_window_size,
        tolerance=stability_tolerance,
    )
    return {
        "aggregation_level": "top_level_complete_video_request",
        "req_definition": "one public file-only POST /v1/annotation-jobs video request",
        "completed_video_count": len(completed),
        "terminal_count": len(terminals),
        "archived_terminal_count": len(all_terminals),
        "drain_after_stability_count": len(drain_terminals),
        "success_total": {"success": len(completed), "total": len(terminals), "ratio": None if not terminals else len(completed) / len(terminals)},
        "completion_window": timing,
        "observed_from_start_video_s": None if timing["window_s"] in (None, 0.0) or len(completed) < 2 else len(completed) / float(timing["window_s"]),
        "video_req_s": stable.get("post_warmup_average_video_s"),
        "stable_video_req_s": stable.get("stable_video_req_s"),
        "stable_window": stable,
        "latency_s": quantiles(elapsed),
        "queue_wait_s": quantiles(queue_wait),
        "status_counts": dict(sorted(Counter(str(row.get("status") or "unknown") for row in terminals).items())),
    }


def summarize_pipeline_service_lanes(
    manager_events: list[dict[str, Any]],
    *,
    stability_warmup_count: int,
    stability_window_size: int,
    stability_tolerance: float,
) -> dict[str, dict[str, Any]]:
    completed = [
        row
        for row in manager_events
        if str(row.get("event") or "") == "terminal"
        and str(row.get("measurement_phase") or "measurement") in {"measurement", "producer"}
        and str(row.get("status") or "").lower() == "completed"
    ]
    service_names = sorted({
        str(service)
        for row in completed
        for service in ((row.get("service_lane_traces") or {}).keys() if isinstance(row.get("service_lane_traces"), dict) else ())
    })
    reports: dict[str, dict[str, Any]] = {}
    for service in service_names:
        lanes = [
            row["service_lane_traces"][service]
            for row in completed
            if isinstance(row.get("service_lane_traces"), dict)
            and isinstance(row["service_lane_traces"].get(service), dict)
            and finite(row["service_lane_traces"][service].get("completed_monotonic_s")) is not None
        ]
        completion_times = [float(lane["completed_monotonic_s"]) for lane in lanes]
        stable = stability_windows(
            completion_times,
            warmup_count=stability_warmup_count,
            window_size=stability_window_size,
            tolerance=stability_tolerance,
        )
        latency = quantiles(finite(lane.get("total_wall_s")) for lane in lanes if finite(lane.get("total_wall_s")) is not None)
        work_units = quantiles(finite(lane.get("request_count")) for lane in lanes if finite(lane.get("request_count")) is not None)
        reports[service] = {
            "name": service,
            "service": service,
            "aggregation_level": "complete_video_pipeline_service_lane",
            "req_definition": "one complete video lane from its first stage trace start to final stage trace completion",
            "video_stream": {
                "completed_video_count": len(lanes),
                "completed_video_req_s": stable.get("post_warmup_average_video_s"),
                "stable_video_req_s": stable.get("stable_video_req_s"),
                "success_total": {
                    "success": len(lanes),
                    "total": len(completed),
                    "ratio": None if not completed else len(lanes) / len(completed),
                },
                "latency_s": {"total_wall_s": latency},
                "work_units_per_video": work_units,
                "stable_window": stable,
            },
            "stage_ids": sorted({stage for lane in lanes for stage in lane.get("stage_ids", [])}),
            "missing_lane_marker_count": len(completed) - len(lanes),
        }
    return reports


def summarize_condition(condition_root: Path) -> dict[str, Any]:
    condition_root = condition_root.expanduser().resolve()
    raw_events, malformed_events = read_jsonl(condition_root / "algorithm_admission_events.jsonl")
    # A client-side 429 retry emits one attempt row for each submission.  Use
    # terminal rows for request-rate/latency summaries, while retaining the
    # complete attempt stream as explicit retry telemetry.  Old event files do
    # not have ``terminal`` and remain backward-compatible.
    attempt_events = [row for row in raw_events if str(row.get("event") or "") == "algorithm_request_forwarded"]
    events = [row for row in raw_events if str(row.get("event") or "") != "algorithm_request_forwarded" or row.get("terminal", True) is not False]
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        route = str(row.get("route") or "unknown")
        service = ROUTE_TO_SERVICE.get(route, str(row.get("service") or "unknown"))
        by_route[route].append(row)
        by_service[service].append(row)
    manager_events, malformed_manager_events = read_jsonl(condition_root / "dataset_request_events.jsonl")
    batch_summary = load_object(condition_root / "dataset_batch_summary.json") or {}
    stability_config = batch_summary.get("stability_window") if isinstance(batch_summary.get("stability_window"), dict) else {}
    configured_warmup = integer(stability_config.get("warmup_video_count"))
    configured_window_size = integer(stability_config.get("window_size"))
    configured_tolerance = finite(stability_config.get("tolerance"))
    stability_warmup_count = 5 if configured_warmup is None else configured_warmup
    stability_window_size = 5 if configured_window_size is None else configured_window_size
    stability_tolerance = 0.10 if configured_tolerance is None else configured_tolerance
    overall = summarize_top_level_video_events(
        manager_events,
        stability_warmup_count=stability_warmup_count,
        stability_window_size=stability_window_size,
        stability_tolerance=stability_tolerance,
    )
    item_records = [summarize_item(item_root, payload) for item_root, payload in _discover_items(condition_root)]
    if not item_records:
        item_records = [
            {"item_index": integer(row.get("item_index")), "case_id": row.get("job_id") or row.get("request_token"), "status": row.get("status"), "run_root": row.get("api_run_root") or row.get("remote_run_root"), "total_pipeline_wall_s": finite(row.get("elapsed_s")), "missing_fields": []}
            for row in manager_events
            if str(row.get("event") or "") == "terminal"
        ]
    run_roots = [Path(record["run_root"]) for record in item_records if isinstance(record.get("run_root"), str) and Path(record["run_root"]).is_dir()]
    completed_video_ids: set[str] = set()
    measurement_completed = [
        row
        for row in manager_events
        if str(row.get("event") or "") == "terminal"
        and str(row.get("measurement_phase") or "measurement") in {"measurement", "producer"}
        and str(row.get("status") or "").lower() == "completed"
    ]
    if measurement_completed:
        for row in measurement_completed:
            case_id = row.get("job_id") or row.get("request_token")
            if case_id:
                completed_video_ids.add(str(case_id))
                completed_video_ids.add(f"{case_id}:item-0")
    else:
        for record in item_records:
            case_id = record.get("case_id")
            if case_id and str(record.get("status") or "").lower() in {"completed", "physical_delivery_complete", "ok"}:
                completed_video_ids.add(str(case_id))
                completed_video_ids.add(f"{case_id}:item-0")
    event_summary_options = {
        "completed_video_ids": completed_video_ids,
        "stability_warmup_count": stability_warmup_count,
        "stability_window_size": stability_window_size,
        "stability_tolerance": stability_tolerance,
    }
    route_reports = {route: summarize_events(rows, name=route, **event_summary_options) for route, rows in sorted(by_route.items())}
    service_work_unit_reports = {service: summarize_events(rows, name=service, service=service, **event_summary_options) for service, rows in sorted(by_service.items())}
    pipeline_service_reports = summarize_pipeline_service_lanes(
        manager_events,
        stability_warmup_count=stability_warmup_count,
        stability_window_size=stability_window_size,
        stability_tolerance=stability_tolerance,
    )
    service_reports = pipeline_service_reports or service_work_unit_reports
    current_service_rates = {
        service: finite(report.get("video_stream", {}).get("completed_video_req_s"))
        for service, report in service_reports.items()
    }
    current_service_rates = {service: rate for service, rate in current_service_rates.items() if rate is not None and rate > 0.0}
    stable_service_rates = {
        service: finite(report.get("video_stream", {}).get("stable_video_req_s"))
        for service, report in service_reports.items()
    }
    stable_service_rates = {service: rate for service, rate in stable_service_rates.items() if rate is not None and rate > 0.0}
    balance = None
    if current_service_rates:
        minimum = min(current_service_rates.values())
        maximum = max(current_service_rates.values())
        balance = {
            "rate_basis": "post_warmup_average_video_s",
            "service_rates": current_service_rates,
            "spread_video_s": maximum - minimum,
            "ratio": maximum / minimum,
            "stable": len(stable_service_rates) == len(service_reports) and bool(service_reports),
            "stable_service_rates": stable_service_rates,
        }
    report_missing = []
    if not events:
        report_missing.append("algorithm_admission_events")
    if malformed_events:
        report_missing.append("algorithm_admission_events.malformed_lines")
    if not item_records:
        report_missing.append("item_results")
    return {
        "schema": "ego.annotation.fps_production_condition_metrics.v1",
        "status": "observed_with_missing" if report_missing or any(record["missing_fields"] for record in item_records) else "ok",
        "condition_root": str(condition_root),
        "acceptance_gates": [],
        "event_source": {
            "path": str(condition_root / "algorithm_admission_events.jsonl"),
            "event_count": len(events),
            "attempt_event_count": len(attempt_events),
            "retry_attempt_count": sum(1 for row in attempt_events if integer(row.get("retry_count")) not in (None, 0)),
            "malformed_line_count": malformed_events,
            "missing_fields": report_missing,
        },
        "top_level_event_source": {"path": str(condition_root / "dataset_request_events.jsonl"), "event_count": len(manager_events), "malformed_line_count": malformed_manager_events},
        "stability_control": batch_summary.get("stability_control"),
        "overall": overall,
        "routes": route_reports,
        "services": service_reports,
        "service_work_unit_diagnostics": service_work_unit_reports,
        "balance": balance,
        "aggregation_levels": {
            "top_level_video_requests": {"count": len(item_records), "source": "items/*/item_result.json"},
            "internal_frame_work_units": {"source": "algorithm_admission_events.jsonl", "routes": sorted(route_reports), "services": sorted(service_reports)},
            "native_batch_traces": {"source": "per-item service trace artifacts", "models": ["unidepth", "droid"]},
        },
        "full_pipeline": {
            "aggregation_level": "top_level_video_request",
            "item_count": len(item_records),
            "items": item_records,
            "status_counts": dict(sorted(Counter(str(record.get("status") or "unknown") for record in item_records).items())),
            "missing_fields": sorted({field for record in item_records for field in record["missing_fields"]}),
        },
        "traces": {
            "unidepth": summarize_trace(run_roots, "unidepth") if run_roots else {"model": "unidepth", "missing_fields": ["item_run_roots"]},
            "droid": summarize_trace(run_roots, "droid") if run_roots else {"model": "droid", "missing_fields": ["item_run_roots"]},
        },
    }


def build_report(condition_root: Path) -> dict[str, Any]:
    """Compatibility alias for callers of the repository's summary scripts."""
    return summarize_condition(condition_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = summarize_condition(args.condition_root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
