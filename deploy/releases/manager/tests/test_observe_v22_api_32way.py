import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "observe_v22_api_32way.py"
SPEC = importlib.util.spec_from_file_location("observe_v22_api_32way", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


class NoSSH:
    host = "fixture-a800"

    def read_json(self, path: str):
        return None, {"status": "fixture_no_remote_read", "path": path}


def test_event_parsing_merges_lifecycle_and_submitter_timings(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"1234567")
    rows = [
        {
            "event": "queued",
            "status": "queued",
            "item_index": 4,
            "request_token": "request-4",
            "video": str(video),
        },
        {
            "event": "request_started",
            "status": "request_started",
            "item_index": 4,
            "request_token": "request-4",
            "request_started_at": "2026-07-26T01:02:03Z",
            "request_started_at_unix": 1785027723.0,
        },
        {
            "event": "terminal",
            "status": "completed",
            "item_index": 4,
            "request_token": "request-4",
            "request_id": "request-4",
            "job_id": "annotation_4",
            "item_id": "item_4",
            "upload_prepare_s": 0.25,
            "manager_http_wait_s": 8.5,
            "submitter_response_decode_s": 0.05,
            "total_submit_wall_s": 8.9,
            "package_path": "/remote/package.zip",
            "finished_at_unix": 1785027732.0,
        },
    ]

    parsed = observer.parse_batch_events(rows, batch_root=tmp_path)

    assert len(parsed) == 1
    job = parsed[0]
    assert job["request_id"] == "request-4"
    assert job["job_id"] == "annotation_4"
    assert job["item_id"] == "item_4"
    assert job["source_path"] == str(video)
    assert job["source_size_bytes"] == 7
    assert job["submitted_at"] == "2026-07-26T01:02:03Z"
    assert job["upload_prepare_s"] == 0.25
    assert job["manager_http_wait_s"] == 8.5
    assert job["response_decode_s"] == 0.05
    assert job["total_submit_wall_s"] == 8.9
    assert job["terminal_status"] == "completed"
    assert job["package_path"] == "/remote/package.zip"


def test_percentiles_are_deterministic_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert observer.percentile(values, 0.50) == 2.5
    assert observer.percentile(values, 0.95) == 3.8499999999999996
    assert observer.percentile([], 0.95) is None


def test_completed_with_uncertainty_is_success_but_failure_aggregate_is_not() -> None:
    assert observer.is_success({"terminal": True, "terminal_status": "completed_with_partial_camera_coverage"})
    assert observer.is_failure({"terminal": True, "terminal_status": "completed_with_failures"})


def test_capacity_comparison_reports_reached_not_reached_and_unknown() -> None:
    assert observer.compare_capacity(32, [2, 32, 16]) == {
        "status": "reached",
        "design_capacity": 32,
        "observed_max": 32.0,
        "denominator_observation_count": 3,
    }
    assert observer.compare_capacity(32, [2, 31]) == {
        "status": "not_reached",
        "design_capacity": 32,
        "observed_max": 31.0,
        "denominator_observation_count": 2,
    }
    assert observer.compare_capacity(32, [None, "missing"]) == {
        "status": "unknown",
        "design_capacity": 32,
        "observed_max": None,
        "denominator_observation_count": 0,
    }


def test_missing_breakdown_fields_remain_none_and_are_not_invented() -> None:
    normalized = observer.normalize_module_breakdown({
        "unidepth": {
            "client_prepare_s": 1.0,
            "transport_wait_s": 2.0,
            "total_wall_s": 4.0,
            "request_count": 3,
        },
        "legacy_scalar": 9.0,
    })

    unidepth = normalized["unidepth"]
    assert unidepth["client_decode_postprocess"] is None
    assert unidepth["client_decode_postprocess_s"] is None
    assert unidepth["local_assembly_write"] is None
    assert unidepth["local_assembly_write_s"] is None
    assert unidepth["own"] is None
    assert unidepth["wait"] == 2.0
    assert unidepth["total"] == 4.0
    assert unidepth["missing_fields"] == ["client_decode_postprocess", "local_assembly_write"]
    assert normalized["legacy_scalar"]["total"] is None
    assert normalized["legacy_scalar"]["request_count"] is None
    assert normalized["legacy_scalar"]["missing_fields"] == list(observer.BREAKDOWN_FIELDS)


def test_hydration_reads_run_result_and_preserves_explicit_missing_fields(tmp_path: Path) -> None:
    run_root = tmp_path / "jobs" / "annotation_9"
    run_root.mkdir(parents=True)
    run_result = {
        "frame_count": 120,
        "module_timings_s": {"unidepth": 9.0},
        "module_timing_breakdown_s": {
            "unidepth": {
                "client_prepare_s": 1.0,
                "transport_wait_s": 6.0,
                "client_decode_postprocess_s": 0.5,
                "local_assembly_write_s": 0.25,
                "total_wall_s": 8.0,
                "request_count": 12,
            }
        },
    }
    (run_root / "run_result.json").write_text(json.dumps(run_result), encoding="utf-8")
    (run_root / "annotation_pipeline_manifest.json").write_text(json.dumps({"frame_count": 120}), encoding="utf-8")
    jobs = observer.parse_batch_events([
        {
            "event": "terminal",
            "status": "completed",
            "request_id": "req-9",
            "job_id": "annotation_9",
            "item_id": "item-9",
            "remote_run_root": str(run_root),
            "total_submit_wall_s": 10.0,
            "finished_at_unix": 200.0,
        }
    ])
    loader = observer.ArtifactLoader(NoSSH())

    observer.hydrate_job_artifacts(jobs[0], job_root=tmp_path / "jobs", loader=loader)

    job = jobs[0]
    assert job["frame_count"] == 120
    assert job["module_timings_s"] == {"unidepth": 9.0}
    assert job["module_timing_breakdown_s"]["unidepth"]["own"] == 1.75
    assert job["module_timing_breakdown_s"]["unidepth"]["wait"] == 6.0
    assert job["module_timing_breakdown_s"]["unidepth"]["total"] == 8.0
    assert job["artifact_payloads"] == {"run_result": "present", "manifest": "present"}
    assert "manager_http_wait_s" in job["missing_fields"]
    assert job["manager_http_wait_s"] is None


def test_throughput_rates_include_numerators_denominators_and_module_quantiles() -> None:
    breakdown = {
        "unidepth": {
            "client_prepare": 1.0,
            "transport_wait": 2.0,
            "client_decode_postprocess": 3.0,
            "local_assembly_write": 4.0,
            "total_wall": 12.0,
            "request_count": 5,
            "own": 8.0,
            "wait": 2.0,
            "total": 12.0,
        }
    }
    jobs = [
        {
            "terminal": True,
            "terminal_status": "completed",
            "submitted_at_unix": 100.0,
            "total_submit_wall_s": 15.0,
            "frame_count": 120,
            "module_timing_breakdown_s": breakdown,
        },
        {
            "terminal": True,
            "terminal_status": "failed_http",
            "submitted_at_unix": 110.0,
            "total_submit_wall_s": 5.0,
            "frame_count": None,
            "module_timing_breakdown_s": {},
        },
    ]

    summary = observer.throughput_summary(jobs, observer_started_unix=90.0, observed_at_unix=3700.0)

    assert summary["videos_per_hour"] == {
        "value": 1.0,
        "numerator_completed_videos": 1,
        "denominator_window_hours": 1.0,
    }
    assert summary["counts"]["outer_observed_max_inflight"] == 2
    assert summary["wall_requests_per_s"]["numerator_terminal_requests"] == 2
    assert summary["wall_requests_per_s"]["denominator_window_s"] == 3600.0
    assert summary["completed_images_per_s"]["numerator_completed_images"] == 120
    assert summary["completed_images_per_s"]["denominator_window_s"] == 3600.0
    assert summary["job_submit_wall"]["p50_s"] == 10.0
    assert summary["module_timing_breakdown"]["unidepth"]["own"]["p95_s"] == 8.0
    assert summary["module_timing_breakdown"]["unidepth"]["request_count"] == {
        "numerator_requests": 5,
        "denominator_job_count_with_request_count": 1,
    }
    assert summary["module_timing_breakdown"]["unidepth"]["wall_requests_per_s"] == {
        "value": 5 / 3600.0,
        "numerator_requests": 5,
        "denominator_window_s": 3600.0,
    }
