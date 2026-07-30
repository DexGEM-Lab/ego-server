from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ego_annotation.fps_config import DEFAULT_FPS_CONDITION, FPS_CONDITION_BY_NAME, get_fps_condition
from scripts.run_api_ify_fps_autoresearch import (
    DEFAULT_STABILITY_VIDEO_LIMIT,
    PAIRED_CONDITIONS,
    configured_output_root,
    configured_stability_video_limit,
)
from scripts import run_v22_api_egoscale30h_batch as batch_manager
from scripts.observe_api_ify_video_rates import observe
from scripts.run_v22_api_egoscale30h_batch import archive_condition_algorithm_events
from scripts.summarize_fps_production_condition import stability_windows, summarize_events, summarize_pipeline_service_lanes, summarize_top_level_video_events


def test_autoresearch_runs_only_the_four_displayed_paired_conditions() -> None:
    assert PAIRED_CONDITIONS == (
        "unidepth_full__droid_full",
        "unidepth_20fps__droid_20fps",
        "unidepth_15fps__droid_15fps",
        "unidepth_10fps__droid_10fps",
    )
    assert DEFAULT_STABILITY_VIDEO_LIMIT == 30


def test_autoresearch_environment_controls_are_applied(tmp_path: Path) -> None:
    environ = {
        "EGO_API_IFY_OUTPUT_ROOT": str(tmp_path / "research"),
        "EGO_API_IFY_STABILITY_VIDEO_LIMIT": "18",
    }
    assert configured_output_root(environ) == tmp_path / "research"
    assert configured_stability_video_limit(environ) == 18


def test_zero_argument_manager_imports_from_outside_repo(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_v22_api_egoscale30h_batch.py"
    completed = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_api_ify_manager_has_one_builtin_source_for_full_10_15_20() -> None:
    assert DEFAULT_FPS_CONDITION == "unidepth_full__droid_full"
    assert len(FPS_CONDITION_BY_NAME) == 10
    assert get_fps_condition("unidepth_10fps__droid_full").unidepth_fps == 10.0
    assert get_fps_condition("unidepth_full__droid_15fps").droid_fps == 15.0
    assert get_fps_condition("unidepth_20fps__droid_20fps").unidepth_fps == 20.0


def test_condition_archive_filters_global_route_events_by_video_identity(tmp_path: Path) -> None:
    source = tmp_path / "global.jsonl"
    rows = [
        {"video_job_id": "job-a", "route": "/unidepth.infer", "status": 200},
        {"video_job_id": "job-b", "route": "/unidepth.infer", "status": 200},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = archive_condition_algorithm_events(tmp_path / "condition", [{"job_id": "job-a"}], source=source)

    archived = [json.loads(line) for line in (tmp_path / "condition/algorithm_admission_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["matched_event_count"] == 1
    assert archived == [rows[0]]


def test_api_http_manager_stops_replenishing_when_stable(tmp_path: Path, monkeypatch) -> None:
    class Handler(BaseHTTPRequestHandler):
        counter = 0

        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            type(self).counter += 1
            marker = float(type(self).counter)
            traces = [
                {"stage_id": "unidepth.infer", "request_count": 1, "started_monotonic_s": marker, "completed_monotonic_s": marker + 0.1},
                {"stage_id": "hands.detect", "request_count": 1, "started_monotonic_s": marker, "completed_monotonic_s": marker + 0.2},
                {"stage_id": "wilor.reconstruct", "request_count": 1, "started_monotonic_s": marker + 0.2, "completed_monotonic_s": marker + 0.3},
                {"stage_id": "droid.create_session", "request_count": 1, "started_monotonic_s": marker + 0.2, "completed_monotonic_s": marker + 0.21},
                {"stage_id": "droid.push_frame", "request_count": 1, "started_monotonic_s": marker + 0.21, "completed_monotonic_s": marker + 0.4},
                {"stage_id": "droid.finalize", "request_count": 1, "started_monotonic_s": marker + 0.4, "completed_monotonic_s": marker + 0.41},
                {"stage_id": "hawor.infer_tracks", "request_count": 1, "started_monotonic_s": marker + 0.41, "completed_monotonic_s": marker + 0.5},
                {"stage_id": "hawor_infiller.fill", "request_count": 1, "started_monotonic_s": marker + 0.5, "completed_monotonic_s": marker + 0.6},
                {"stage_id": "cosmos3.reason", "request_count": 2, "started_monotonic_s": marker, "completed_monotonic_s": marker + 0.55},
            ]
            payload = json.dumps({
                "job_id": f"job-{type(self).counter}",
                "status": "ok",
                "summary": {
                    "performance": {"request_traces": traces},
                    "cosmos": {
                        "status": "enabled",
                        "request_count": 2,
                        "semantic_row_count": 1,
                        "review_json": "/remote/review.json",
                        "captioned_combined_video": "/remote/v22_combined.mp4",
                    },
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def fake_stability(values, **_kwargs):
        count = len(list(values))
        return {"stable": count >= 2, "stable_video_req_s": 1.0 if count >= 2 else None, "windows": [], "total_completion_count": count}

    monkeypatch.setattr(batch_manager, "stability_windows", fake_stability)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    videos = []
    for index in range(10):
        path = tmp_path / f"video-{index}.mp4"
        path.write_bytes(b"video")
        videos.append(path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    args = Namespace(
        output_root=output_root,
        api_base_url=f"http://127.0.0.1:{server.server_address[1]}",
        api_client_concurrency=2,
        api_request_timeout_s=30.0,
        api_job_prefix=None,
        fps_condition=get_fps_condition(DEFAULT_FPS_CONDITION),
        stability_warmup_count=0,
        stability_window_size=2,
        stability_tolerance=0.1,
        stability_video_limit=10,
        api_model_backend="api_ify",
        api_diagnostic_monocular=True,
        total_request_limit=128,
        algorithm_inflight_multiplier=2,
        dataset_root=tmp_path,
    )
    try:
        result = batch_manager.run_api_http_requests(args, videos, started=time.time())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    rows = [json.loads(line) for line in (output_root / "dataset_request_events.jsonl").read_text().splitlines()]
    queued = [row for row in rows if row["event"] == "queued"]
    terminals = [row for row in rows if row["event"] == "terminal"]
    assert result == 0
    assert 2 <= len(queued) < len(videos)
    assert all(row["measurement_phase"] in {"measurement", "drain_after_stability"} for row in terminals)
    summary = json.loads((output_root / "dataset_batch_summary.json").read_text())
    assert summary["stability_control"]["stability_reached"] is True
    assert summary["stability_control"]["stable_at_completion_count"] == 2
    assert set(summary["stability_control"]["service_stability"]) == {"unidepth", "hands.detect", "wilor", "droid", "hawor.track", "hawor.infiller", "cosmos3"}
    assert all(row["stable"] for row in summary["stability_control"]["service_stability"].values())
    assert all(row["service_lane_traces"] for row in terminals if row["status"] == "completed")


def test_read_only_observer_selects_stable_boundary_without_controlling_producer(tmp_path: Path) -> None:
    root = tmp_path / "condition"
    root.mkdir()
    rows = []
    services = ("unidepth", "hands.detect", "wilor", "droid", "hawor.track", "hawor.infiller", "cosmos3")
    for index in range(16):
        rows.append({
            "event": "terminal",
            "measurement_phase": "producer",
            "status": "completed",
            "finished_at_unix": float(index + 1),
            "service_lane_traces": {
                service: {"completed_monotonic_s": float(index + 1) + lane_index / 100.0, "total_wall_s": 0.5, "request_count": 1, "stage_ids": [service]}
                for lane_index, service in enumerate(services)
            },
        })
    (root / "dataset_request_events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = observe(root, minimum_discard=4, window_size=4, tolerance=0.1)
    assert result["producer_control"].startswith("read_only")
    assert result["completed_video_count"] == 16
    assert result["stable_boundary_found"] is True
    assert result["selected"]["discard_completion_count"] == 4
    assert result["selected"]["overall"]["post_warmup_average_video_s"] == 1.0
    assert result["selected"]["all_stable"] is True


def test_stability_rule_requires_three_consistent_windows_after_warmup() -> None:
    result = stability_windows(float(index) for index in range(30))
    assert result["warmup_completion_count"] == 5
    assert len(result["windows"]) == 5
    assert result["windows"][0]["duration_s"] == 5.0
    assert result["windows"][1]["window_started_at_unix"] == result["windows"][0]["last_finished_at_unix"]
    assert result["stable"] is True
    assert result["stable_video_req_s"] == 1.0


def test_pipeline_service_lane_summary_uses_trace_completion_average() -> None:
    rows = []
    for index in range(5):
        rows.append({
            "event": "terminal",
            "measurement_phase": "measurement",
            "status": "completed",
            "service_lane_traces": {
                "unidepth": {
                    "stage_ids": ["unidepth.infer"],
                    "request_count": 10,
                    "started_monotonic_s": float(index),
                    "completed_monotonic_s": float(index + 1),
                    "total_wall_s": 1.0,
                }
            },
        })
    report = summarize_pipeline_service_lanes(rows, stability_warmup_count=1, stability_window_size=2, stability_tolerance=0.1)
    stream = report["unidepth"]["video_stream"]
    assert stream["completed_video_count"] == 5
    assert stream["completed_video_req_s"] == 1.0
    assert stream["latency_s"]["total_wall_s"]["p50"] == 1.0
    assert stream["work_units_per_video"]["p50"] == 10.0


def test_summary_excludes_post_stability_drain_from_measurement() -> None:
    rows = [
        {"event": "terminal", "measurement_phase": "measurement", "request_token": "a", "status": "completed", "finished_at_unix": 10.0, "elapsed_s": 5.0},
        {"event": "terminal", "measurement_phase": "drain_after_stability", "request_token": "b", "status": "completed", "finished_at_unix": 20.0, "elapsed_s": 7.0},
    ]
    result = summarize_top_level_video_events(rows, stability_warmup_count=0, stability_window_size=2)
    assert result["completed_video_count"] == 1
    assert result["terminal_count"] == 1
    assert result["archived_terminal_count"] == 2
    assert result["drain_after_stability_count"] == 1


def test_service_video_stream_requires_all_identity_events_to_succeed() -> None:
    rows = [
        {"video_item_id": "video-a", "received_at_unix": 1.0, "finished_at_unix": 2.0, "wait_s": 0.1, "upstream_wall_s": 0.8, "total_wall_s": 1.0, "status": 200},
        {"video_item_id": "video-a", "received_at_unix": 2.0, "finished_at_unix": 4.0, "wait_s": 0.2, "upstream_wall_s": 1.7, "total_wall_s": 2.0, "status": 200},
        {"video_item_id": "video-b", "received_at_unix": 3.0, "finished_at_unix": 5.0, "wait_s": 0.3, "upstream_wall_s": 1.5, "total_wall_s": 2.0, "status": 500},
    ]
    result = summarize_events(rows, name="unidepth", service="unidepth", completed_video_ids={"video-a", "video-b"})
    stream = result["video_stream"]
    assert stream["completed_video_count"] == 1
    assert stream["success_total"] == {"success": 1, "total": 2, "ratio": 0.5}
    assert stream["latency_s"]["total_wall_s"]["p50"] == 3.0
    assert stream["work_units_per_video"]["p50"] == 2.0
