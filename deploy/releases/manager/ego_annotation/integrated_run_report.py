"""Create one-item client/service timing reports without inventing missing telemetry."""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ADMISSION_EVENTS_ENV = "EGO_ANNOTATION_ADMISSION_EVENTS_PATH"
REPORT_JSON_NAME = "integrated_run_report.json"
REPORT_TEXT_NAME = "integrated_run_report.md"

MODULE_STAGES: Mapping[str, tuple[str, ...]] = {
    "frame_store": (),
    "unidepth": ("unidepth.infer",),
    "hands": ("hands.detect",),
    "wilor_build": (),
    "wilor_service": ("wilor.reconstruct",),
    "droid": ("droid.create_session", "droid.push_frame", "droid.finalize"),
    "hawor": ("hawor.infer_tracks",),
    "infiller": ("hawor_infiller.fill",),
    "cosmos": ("cosmos3.reason",),
    "render": (),
}


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def admission_events_path_from_environment(environ: Mapping[str, str] | None = None) -> Path | None:
    value = (environ or os.environ).get(ADMISSION_EVENTS_ENV, "").strip()
    return Path(value).expanduser() if value else None


def _event_value(row: Mapping[str, object], name: str) -> object:
    if name == "finished_at_unix":
        return row.get("finished_at_unix", row.get("finished_at"))
    return row.get(name)


def summarize_client_queue_events(
    events_path: Path | None,
    *,
    case_id: str,
) -> dict[str, object]:
    """Summarize actual proxy forward events in the scheduler's release order."""
    if events_path is None:
        return {
            "status": "unavailable",
            "reason": f"{ADMISSION_EVENTS_ENV} was not supplied by the admission wrapper",
            "path": None,
            "client_release_sequence": [],
            "client_batch_distributions": [],
        }
    if not events_path.is_file():
        return {
            "status": "unavailable",
            "reason": "admission events path does not exist",
            "path": str(events_path),
            "client_release_sequence": [],
            "client_batch_distributions": [],
        }

    malformed_lines = 0
    selected: list[dict[str, object]] = []
    for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            malformed_lines += 1
            continue
        if event.get("event") != "algorithm_request_forwarded" or event.get("video_job_id") != case_id:
            continue
        selected.append(event)

    def release_order(row: Mapping[str, object]) -> tuple[float, float, str, int]:
        started = _finite_number(row.get("upstream_started_at_unix"))
        received = _finite_number(row.get("received_at_unix"))
        return (
            started if started is not None else float("inf"),
            received if received is not None else float("inf"),
            str(row.get("logical_request_id") or ""),
            _nonnegative_int(row.get("attempt")) or 0,
        )

    selected.sort(key=release_order)
    sequence = [
        {
            "release_index": index,
            "route": row.get("route"),
            "logical_request_id": row.get("logical_request_id"),
            "client_batch_id": row.get("batch_id"),
            "client_batch_size": row.get("batch_size"),
            "received_at_unix": row.get("received_at_unix"),
            "upstream_started_at_unix": row.get("upstream_started_at_unix"),
            "upstream_finished_at_unix": row.get("upstream_finished_at_unix"),
            "finished_at_unix": _event_value(row, "finished_at_unix"),
            "wait_s": row.get("wait_s"),
            "attempt": row.get("attempt"),
            "retry_count": row.get("retry_count"),
            "status": row.get("status"),
            "terminal": row.get("terminal"),
        }
        for index, row in enumerate(selected, start=1)
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in sequence:
        route = row.get("route")
        batch_id = row.get("client_batch_id")
        if isinstance(route, str) and route and isinstance(batch_id, str) and batch_id:
            grouped[(route, batch_id)].append(row)
    distributions = [
        {
            "route": route,
            "client_batch_id": batch_id,
            "client_batch_size": rows[0].get("client_batch_size"),
            "forwarded_attempt_count": len(rows),
            "logical_request_ids": sorted({str(row["logical_request_id"]) for row in rows if row.get("logical_request_id")}),
            "statuses": [row.get("status") for row in rows],
        }
        for (route, batch_id), rows in sorted(grouped.items())
    ]
    if not sequence:
        return {
            "status": "unavailable",
            "reason": "no admission forward events matched this case_id",
            "path": str(events_path),
            "malformed_line_count": malformed_lines,
            "client_release_sequence": [],
            "client_batch_distributions": [],
        }
    return {
        "status": "available",
        "path": str(events_path),
        "malformed_line_count": malformed_lines,
        "client_release_sequence": sequence,
        "client_batch_distributions": distributions,
    }


def summarize_service_batch_traces(
    records: Sequence[Mapping[str, object]],
    *,
    client_release_sequence: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Deduplicate service callbacks and pair them by returned request IDs."""
    request_to_client_batch: dict[str, tuple[str, object]] = {}
    for row in client_release_sequence:
        request_id = row.get("logical_request_id")
        batch_id = row.get("client_batch_id")
        if isinstance(request_id, str) and isinstance(batch_id, str):
            request_to_client_batch[request_id] = (batch_id, row.get("client_batch_size"))
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        stage_id = record.get("stage_id")
        trace = record.get("trace")
        if not isinstance(stage_id, str) or not isinstance(trace, Mapping):
            continue
        batch_id = trace.get("batch_id")
        if isinstance(batch_id, str) and batch_id:
            grouped[(stage_id, batch_id)].append(record)
    distributions: list[dict[str, object]] = []
    for (stage_id, batch_id), rows in sorted(grouped.items()):
        trace = rows[0]["trace"]
        assert isinstance(trace, Mapping)
        request_ids = tuple(
            str(value) for row in rows for value in (trace.get("request_ids", ()) if isinstance(trace.get("request_ids", ()), (list, tuple)) else ())
        )
        paired = [request_to_client_batch[value] for value in request_ids if value in request_to_client_batch]
        distributions.append(
            {
                "stage_id": stage_id,
                "batch_id": batch_id,
                "response_record_count": len(rows),
                "request_count": trace.get("request_count"),
                "forward_count": trace.get("forward_count", trace.get("fnet_forward_count")),
                "fnet_forward_count": trace.get("fnet_forward_count"),
                "session_local_forward_count": trace.get("session_local_forward_count"),
                "effective_work_units": trace.get("effective_work_units"),
                "replica_id": trace.get("replica_id"),
                "request_ids": list(dict.fromkeys(request_ids)),
                "client_batch_ids": sorted({item[0] for item in paired}),
                "client_batch_sizes": sorted({item[1] for item in paired}),
                "pairing_status": "available" if paired else "unavailable_without_request_ids",
                "traces": [dict(row["trace"]) for row in rows if isinstance(row.get("trace"), Mapping)],
            }
        )
    return {
        "status": "available" if distributions else "unavailable",
        "reason": None if distributions else "no decoded live response contained a complete service batch trace",
        "response_record_count": len(records),
        "deduplicated_batch_distributions": distributions,
    }


def _module_performance(
    timing_breakdowns: Mapping[str, object],
    distributions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    units_by_stage: dict[str, int] = defaultdict(int)
    for distribution in distributions:
        stage_id = distribution.get("stage_id")
        work = _nonnegative_int(distribution.get("effective_work_units"))
        if isinstance(stage_id, str) and work is not None:
            units_by_stage[stage_id] += work
    rows: list[dict[str, object]] = []
    for module, stages in MODULE_STAGES.items():
        raw = timing_breakdowns.get(module)
        timing = raw if isinstance(raw, Mapping) else {}
        wall_s = _finite_number(timing.get("total_wall_s"))
        request_count = _nonnegative_int(timing.get("request_count"))
        work_units = sum(units_by_stage.get(stage, 0) for stage in stages)
        service_available = any(stage in units_by_stage for stage in stages)
        rows.append(
            {
                "module": module,
                "wall_s": wall_s,
                "request_count": request_count,
                "requests_per_s": (request_count / wall_s) if wall_s and request_count is not None else None,
                "equivalent_img_per_s": (work_units / wall_s) if wall_s and service_available else None,
                "equivalent_work_units": work_units if service_available else None,
                "equivalent_img_status": "available" if service_available else "unavailable_without_service_batch_trace",
            }
        )
    return rows


def _human_report(report: Mapping[str, object]) -> str:
    lines = ["# Integrated run timing report", "", f"Run root: `{report['run_root']}`", "", "## Modules", "", "| Module | Wall time (s) | Requests | req/s | equivalent img/s |", "|---|---:|---:|---:|---:|"]
    for row in report["modules"]:  # type: ignore[index]
        assert isinstance(row, Mapping)
        def shown(value: object) -> str:
            return "unavailable" if value is None else f"{float(value):.3f}" if isinstance(value, float) else str(value)
        lines.append(f"| {row['module']} | {shown(row['wall_s'])} | {shown(row['request_count'])} | {shown(row['requests_per_s'])} | {shown(row['equivalent_img_per_s'])} |")
    service = report["service_batches"]
    client = report["client_scheduler"]
    assert isinstance(service, Mapping) and isinstance(client, Mapping)
    lines.extend(["", "## Client scheduler", "", f"Status: **{client['status']}**. {client.get('reason') or ''}".rstrip(), f"Release attempts: {len(client['client_release_sequence'])}", "", "## Service batches", "", f"Status: **{service['status']}**. {service.get('reason') or ''}".rstrip(), f"Deduplicated batches: {len(service['deduplicated_batch_distributions'])}", "", "## Artifacts", ""])
    artifacts = report["artifacts"]
    assert isinstance(artifacts, Mapping)
    for label, path in artifacts.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def write_integrated_run_report(
    run_root: Path,
    *,
    case_id: str,
    performance: Mapping[str, object],
    service_batch_traces: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, object],
    admission_events_path: Path | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown reports after a one-item run completes."""
    run_root = run_root.resolve()
    json_path = run_root / REPORT_JSON_NAME
    text_path = run_root / REPORT_TEXT_NAME
    client = summarize_client_queue_events(admission_events_path, case_id=case_id)
    sequence = client.get("client_release_sequence", [])
    service = summarize_service_batch_traces(
        service_batch_traces,
        client_release_sequence=sequence if isinstance(sequence, list) else (),
    )
    distributions = service["deduplicated_batch_distributions"]
    assert isinstance(distributions, list)
    timings = performance.get("module_timing_breakdown_s")
    report = {
        "schema": "ego.annotation.integrated_run_report.v2",
        "case_id": case_id,
        "run_root": str(run_root),
        "modules": _module_performance(timings if isinstance(timings, Mapping) else {}, distributions),
        "client_scheduler": client,
        "service_batches": service,
        "client_service_pairing": {
            "status": "available" if any(row.get("pairing_status") == "available" for row in distributions if isinstance(row, Mapping)) else "unavailable",
            "basis": "service-returned request_ids mapped to client logical_request_id and client_batch_id",
        },
        "artifacts": {**dict(artifacts), "machine_readable_report": str(json_path), "human_readable_report": str(text_path)},
    }
    _atomic_json(json_path, report)
    _atomic_text(text_path, _human_report(report))
    return {"json": str(json_path), "text": str(text_path)}


__all__ = [
    "ADMISSION_EVENTS_ENV",
    "REPORT_JSON_NAME",
    "REPORT_TEXT_NAME",
    "admission_events_path_from_environment",
    "summarize_client_queue_events",
    "summarize_service_batch_traces",
    "write_integrated_run_report",
]
