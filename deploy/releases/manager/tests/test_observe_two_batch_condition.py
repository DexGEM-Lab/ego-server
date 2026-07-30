import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "observe_two_batch_condition.py"
SPEC = importlib.util.spec_from_file_location("observe_two_batch_condition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def test_service_statistics_routes_events_and_computes_percentiles(tmp_path: Path) -> None:
    events = tmp_path / "admission.jsonl"
    rows = [
        {"route": "/unidepth.infer", "finished_at_unix": 101.0, "total_wall_s": 2.0, "wait_s": 0.1, "status": 200},
        {"route": "/unidepth.infer", "finished_at_unix": 102.0, "total_wall_s": 4.0, "wait_s": 0.3, "status": 200},
        {"route": "/droid.finalize", "finished_at_unix": 102.0, "total_wall_s": 8.0, "wait_s": 1.0, "status": 500},
        {"route": "/wilor.reconstruct", "finished_at_unix": 102.0, "total_wall_s": 6.0, "wait_s": 0.2, "status": 200},
        {"route": "/unidepth.infer", "finished_at_unix": 99.0, "total_wall_s": 999.0, "wait_s": 999.0, "status": 200},
    ]
    events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    stats = observer.service_statistics(events, started_at_unix=100.0, interval_started_at_unix=100.0)

    assert stats["28000"]["admission_events_since_start"] == 2
    assert stats["28000"]["completed_events_this_interval"] == 2
    assert stats["28000"]["p50_wall_s"] == 3.0
    assert stats["28000"]["p95_wall_s"] == 3.9
    assert stats["28002"]["status_counts_since_start"] == {"500": 1}
    assert stats["28004"]["admission_events_since_start"] == 1
    assert stats["28004"]["p50_wall_s"] == 6.0


def test_admission_event_tail_ignores_preexisting_manager_history(tmp_path: Path) -> None:
    events = tmp_path / "admission.jsonl"
    events.write_text(json.dumps({"route": "/unidepth.infer", "finished_at_unix": 1.0, "total_wall_s": 999.0}) + "\n", encoding="utf-8")
    tail = observer.AdmissionEventTail(events, started_at_unix=100.0)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"route": "/unidepth.infer", "finished_at_unix": 101.0, "total_wall_s": 2.0, "wait_s": 0.1, "status": 200}) + "\n")

    stats = tail.update(interval_started_at_unix=100.0)

    assert stats["28000"]["admission_events_since_start"] == 1
    assert stats["28000"]["p50_wall_s"] == 2.0
