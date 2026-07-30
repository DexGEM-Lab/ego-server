from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ego_annotation.serving.benchmark.manifest import build_synthetic_unidepth_manifest
from ego_annotation.serving.benchmark.unidepth_service_burst import (
    ChildSubmission,
    IncompleteWaveEvidenceError,
    _multipart_for_item,
    _response_from_submission,
    prebuild_wave_bodies,
    run_fixed_process_service_waves,
    validate_complete_wave_evidence,
)
from ego_annotation.serving.contracts import ErrorCode
from ego_annotation.serving.transport import build_multipart_response


class _WaveServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, batch_size: int):
        super().__init__(address, handler)
        self.batch_size = batch_size
        self.arrival_times: list[float] = []
        self.arrival_lock = threading.Lock()


class _WaveHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with self.server.arrival_lock:  # type: ignore[attr-defined]
            self.server.arrival_times.append(time.monotonic())  # type: ignore[attr-defined]
        # The burst runner only needs a server trace; this small deterministic HTTP
        # server keeps the test independent of aiohttp/Ray and GPU hardware.
        now = time.monotonic()
        ownership = {"request_id": "unknown", "job_id": "unknown", "item_id": "unknown", "stage_id": "unknown", "source_id": "unknown"}
        marker = b'"ownership":'
        if marker in body:
            # Request ownership is parsed enough to satisfy the response wire contract.
            import json
            from ego_annotation.serving.benchmark.fakeserver import _parse_generic_multipart
            metadata, _ = _parse_generic_multipart(body, self.headers["Content-Type"])
            ownership = metadata["ownership"]
        trace = {
            "batch_id": "server-wave-batch",
            "replica_id": "test-replica",
            "admitted_monotonic_s": now,
            "dispatched_monotonic_s": now,
            "forward_started_monotonic_s": now,
            "completed_monotonic_s": now + 0.001,
            "effective_work_units": self.server.batch_size,  # type: ignore[attr-defined]
            "request_count": self.server.batch_size,  # type: ignore[attr-defined]
            "forward_count": 1,
            "model_load_count": 1,
        }
        response, content_type = build_multipart_response({"result": {
            "ownership": ownership,
            "phase_timing": {"admission_ms": 0.0, "queue_ms": 0.0, "dispatch_ms": 0.0, "forward_ms": 1.0, "encoding_ms": 0.0},
            "trace": trace,
            "batch_diagnostics": {"allocator_memory": {"allocated_bytes": 1, "reserved_bytes": 2, "max_allocated_bytes": 3, "max_reserved_bytes": 4}},
        }}, {})
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _start_wave_server(batch_size: int) -> tuple[_WaveServer, threading.Thread]:
    server = _WaveServer(("127.0.0.1", 0), _WaveHandler, batch_size=batch_size)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_prebuild_finishes_every_multipart_body_before_release_clock() -> None:
    manifest = build_synthetic_unidepth_manifest(manifest_id="prebuilt", count=8, height=512, width=1024)

    def deliberately_expensive_build(item):
        deadline = time.monotonic() + 0.003
        while time.monotonic() < deadline:
            pass
        return _multipart_for_item(item)

    prepared = prebuild_wave_bodies(manifest.items, build_body=deliberately_expensive_build)
    release_clock = time.monotonic()
    assert len(prepared) == 8
    assert all(request.build_completed_s <= release_clock for request in prepared)
    assert prepared[-1].build_completed_s - prepared[0].build_started_s >= 0.020
    assert all(len(request.body) > 1_500_000 for request in prepared)


def test_prebuild_envelope_records_format_and_preserves_large_tensor_vectors() -> None:
    manifest = build_synthetic_unidepth_manifest(manifest_id="prebuilt-envelope", count=1, height=512, width=1024)
    prepared = prebuild_wave_bodies(manifest.items, wire_format="envelope")
    assert prepared[0].wire_format == "envelope"
    assert prepared[0].content_type == "application/vnd.ego.binary-envelope"
    assert isinstance(prepared[0].body, tuple)
    assert len(prepared[0].body) == 4  # HTTP prefix, codec header, metadata, RGB
    assert prepared[0].to_dict()["wire_format"] == "envelope"
    assert prepared[0].body[-1].nbytes == 512 * 1024 * 3


def test_fixed_process_wave_synchronizes_prebuilt_submissions_and_preserves_batch_evidence() -> None:
    server, thread = _start_wave_server(batch_size=8)
    try:
        manifest = build_synthetic_unidepth_manifest(manifest_id="sync", count=8, height=512, width=1024)
        run = run_fixed_process_service_waves(
            f"http://127.0.0.1:{server.server_port}/unidepth.infer", manifest,
            wave_size=8, wave_count=1, release_lead_s=0.05, timeout_s=10.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert len(run.records) == 8
    wave = run.waves[0]
    submit_times = [record.submit_s for record in wave.request_evidence]
    assert max(submit_times) - min(submit_times) < 0.020
    # Serve receives a complete multipart request after the OS/network has drained
    # its bytes; batch admission is controlled by the synchronized write clocks.
    assert wave.submit_spread_s < 0.020
    assert wave.submissions_within_synchronization_window is True
    assert all(record.build_completed_s < wave.release_target_s for record in wave.request_evidence)
    assert wave.dominant_observed_batch_size == 8
    assert wave.requested_wave_matches_dominant_batch_size is True
    assert {record.allocator_memory["reserved_bytes"] for record in wave.request_evidence} == {2}


def test_burst_parent_preserves_http_and_multipart_failure_evidence() -> None:
    item = build_synthetic_unidepth_manifest(manifest_id="error-evidence", count=1, height=8, width=8).items[0]

    http_body, http_content_type = build_multipart_response({"error": {
        "code": "validation", "message": "rgb payload rejected", "retryable": False,
    }}, {})
    http_response = _response_from_submission(item, ChildSubmission(
        child_id="http-error", ready_s=1.0, submit_s=1.1, response_s=1.2, status_code=400,
        response_headers={"Content-Type": http_content_type}, response_body=http_body,
    ))
    assert http_response.error is not None
    assert http_response.error.code is ErrorCode.TRANSPORT
    assert f"http_status=400 content_type={http_content_type!r} body_length={len(http_body)}" in http_response.error.message
    assert f"body_prefix={http_body[:512].decode('utf-8', errors='replace')!r}" in http_response.error.message
    assert "parsed_error_code='validation' parsed_error_message='rgb payload rejected'" in http_response.error.message

    malformed_body = b"not a multipart response: diagnostic payload " + b"x" * 480 + b" AFTER_PREFIX"
    malformed_content_type = "multipart/form-data; boundary=missing"
    malformed_response = _response_from_submission(item, ChildSubmission(
        child_id="malformed", ready_s=1.0, submit_s=1.1, response_s=1.2, status_code=200,
        response_headers={"content-type": malformed_content_type}, response_body=malformed_body,
    ))
    assert malformed_response.error is not None
    assert malformed_response.error.code is ErrorCode.TRANSPORT
    assert "parse_exception=ValueError: multipart response missing 'metadata' part" in malformed_response.error.message
    assert f"content_type={malformed_content_type!r} body_length={len(malformed_body)}" in malformed_response.error.message
    assert f"body_prefix={malformed_body[:512].decode('utf-8')!r}" in malformed_response.error.message
    assert "AFTER_PREFIX" not in malformed_response.error.message

    meta_body, meta_content_type = build_multipart_response({"error": {
        "code": "backpressure", "message": "queue full", "retryable": True,
    }}, {})
    meta_response = _response_from_submission(item, ChildSubmission(
        child_id="meta-error", ready_s=1.0, submit_s=1.1, response_s=1.2, status_code=200,
        response_headers={"Content-Type": meta_content_type}, response_body=meta_body,
    ))
    assert meta_response.error is not None
    assert meta_response.error.code is ErrorCode.BACKPRESSURE
    assert meta_response.error.retryable is True
    assert "parsed_error_code='backpressure' parsed_error_message='queue full'" in meta_response.error.message
    assert f"http_status=200 content_type={meta_content_type!r} body_length={len(meta_body)}" in meta_response.error.message
    assert f"body_prefix={meta_body[:512].decode('utf-8', errors='replace')!r}" in meta_response.error.message


def test_parent_rejects_incomplete_wave_evidence() -> None:
    submission = ChildSubmission(
        child_id="child-0", ready_s=1.0, submit_s=1.1, response_s=1.2, status_code=200,
        response_headers={}, response_body=b"",
    )
    with pytest.raises(IncompleteWaveEvidenceError, match="missing"):
        validate_complete_wave_evidence(
            ("child-0", "child-1"), {"child-0": submission}, {"child-0": 0, "child-1": None},
        )
