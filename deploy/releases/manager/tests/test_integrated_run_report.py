from __future__ import annotations

import json
from pathlib import Path

from ego_annotation.integrated_run_report import (
    ADMISSION_EVENTS_ENV,
    admission_events_path_from_environment,
    summarize_client_queue_events,
    write_integrated_run_report,
)


def _forwarded(*, batch_id: str, attempt: int, status: int, started: float, terminal: bool) -> dict[str, object]:
    return {
        "event": "algorithm_request_forwarded",
        "route": "/hands.detect",
        "video_job_id": "case-1",
        "logical_request_id": "request-1",
        "batch_id": batch_id,
        "batch_size": 2,
        "received_at_unix": 10.0,
        "upstream_started_at_unix": started,
        "upstream_finished_at_unix": started + 0.5,
        "finished_at_unix": started + 0.6,
        "wait_s": started - 10.0,
        "attempt": attempt,
        "retry_count": attempt - 1,
        "status": status,
        "terminal": terminal,
    }


def test_client_queue_summary_preserves_scheduler_release_sequence(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    rows = [
        _forwarded(batch_id="batch-2", attempt=2, status=200, started=12.0, terminal=True),
        _forwarded(batch_id="batch-1", attempt=1, status=429, started=11.0, terminal=False),
        {"event": "algorithm_request_forwarded", "video_job_id": "other"},
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot-json\n", encoding="utf-8")

    summary = summarize_client_queue_events(events, case_id="case-1")

    assert summary["status"] == "available"
    sequence = summary["client_release_sequence"]
    assert [row["client_batch_id"] for row in sequence] == ["batch-1", "batch-2"]
    assert sequence[0]["wait_s"] == 1.0
    assert sequence[1]["attempt"] == 2
    assert sequence[1]["status"] == 200
    assert summary["malformed_line_count"] == 1
    assert admission_events_path_from_environment({ADMISSION_EVENTS_ENV: str(events)}) == events


def test_report_creation_keeps_trace_and_unavailability_explicit(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(_forwarded(batch_id="client-batch", attempt=1, status=200, started=11.0, terminal=True)) + "\n", encoding="utf-8")
    traces = [
        {
            "stage_id": "hands.detect",
            "case_id": "case-1",
            "trace": {
                "batch_id": "service-batch",
                "request_count": 2,
                "forward_count": 1,
                "effective_work_units": 4,
                "replica_id": "gpu-0",
                "fnet_count": 7,
                "session_count": 3,
            },
        },
        {
            "stage_id": "hands.detect",
            "case_id": "case-1",
            "trace": {
                "batch_id": "service-batch",
                "request_count": 2,
                "forward_count": 1,
                "effective_work_units": 4,
                "replica_id": "gpu-0",
            },
        },
    ]
    monkeypatch.setenv(ADMISSION_EVENTS_ENV, str(events))
    paths = write_integrated_run_report(
        tmp_path / "run",
        case_id="case-1",
        performance={"module_timing_breakdown_s": {"hands": {"total_wall_s": 2.0, "request_count": 2}}},
        service_batch_traces=traces,
        artifacts={"combined_video": "renders/video.mp4"},
        admission_events_path=admission_events_path_from_environment(),
    )

    report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    hands = next(row for row in report["modules"] if row["module"] == "hands")
    assert hands["requests_per_s"] == 1.0
    assert hands["equivalent_img_per_s"] == 2.0
    distribution = report["service_batches"]["deduplicated_batch_distributions"][0]
    assert distribution["response_record_count"] == 2
    assert distribution["traces"][0]["fnet_count"] == 7
    assert report["client_scheduler"]["client_release_sequence"][0]["client_batch_size"] == 2
    assert "Client scheduler" in Path(paths["text"]).read_text(encoding="utf-8")

    unavailable = json.loads(
        Path(
            write_integrated_run_report(
                tmp_path / "without-service",
                case_id="case-1",
                performance={"module_timing_breakdown_s": {}},
                service_batch_traces=[],
                artifacts={},
            )["json"]
        ).read_text(encoding="utf-8")
    )
    assert unavailable["service_batches"]["status"] == "unavailable"
    assert unavailable["modules"][0]["equivalent_img_status"] == "unavailable_without_service_batch_trace"
