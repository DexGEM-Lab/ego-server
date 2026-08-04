from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import scripts.annotation_admission_proxy as proxy_module
from scripts.annotation_admission_proxy import (
    AdmissionServer,
    BatchScheduler,
    IMAGE_QUEUE_BUDGET,
    PendingRequest,
    ROUTE_FORWARD_LIMITS,
    ROUTE_IMAGE_UNITS,
    UpstreamResult,
)


def _request(route: str, *, body: bytes = b"x") -> PendingRequest:
    return PendingRequest(
        route=route,
        service="hands_wilor",
        body=body,
        headers=(("Content-Type", "application/octet-stream"),),
        video_job_id=None,
        video_item_id=None,
        received_at_unix=time.time(),
        received_at_mono=time.monotonic(),
        queued_at_mono=time.monotonic(),
        logical_id=f"id-{id(body)}-{time.monotonic_ns()}",
        body_size=len(body),
        work_units=ROUTE_IMAGE_UNITS[route],
    )


def test_image_budget_conversions_are_conservative() -> None:
    assert IMAGE_QUEUE_BUDGET == 8000
    assert ROUTE_IMAGE_UNITS == {
        "/unidepth.infer": 1,
        "/hands.detect": 1,
        "/wilor.reconstruct": 1,
        "/droid.infer": 256,
        "/hawor.infer_tracks": 16,
        "/hawor_infiller.fill": 120,
        "/cosmos3.reason": 8,
    }
    assert {r: (IMAGE_QUEUE_BUDGET + n - 1) // n for r, n in ROUTE_IMAGE_UNITS.items()} == {
        "/unidepth.infer": 8000,
        "/hands.detect": 8000,
        "/wilor.reconstruct": 8000,
        "/droid.infer": 32,
        "/hawor.infer_tracks": 500,
        "/hawor_infiller.fill": 67,
        "/cosmos3.reason": 1000,
    }


def test_weighted_queue_blocks_at_8000_until_terminal_completion() -> None:
    scheduler = BatchScheduler(
        "/hawor.infer_tracks", max_inflight=1, work_units=16, queue_budget=8000,
    )
    requests = [_request("/hawor.infer_tracks") for _ in range(501)]
    try:
        for request in requests[:500]:
            scheduler.enqueue(request)
        assert scheduler.admitted_work_units == 8000
        blocked = threading.Thread(target=scheduler.enqueue, args=(requests[500],))
        blocked.start()
        time.sleep(0.05)
        assert blocked.is_alive(), "a 501st 16-frame request bypassed the 8000-frame queue"
        requests[0].wait_for_dispatch()
        scheduler.complete_batch_member(requests[0].batch_id, terminal=True)
        blocked.join(timeout=2.0)
        assert not blocked.is_alive()
        assert scheduler.admitted_work_units == 8000
    finally:
        scheduler.stop()


def test_burst_opens_only_internal_forwarding_workers_and_spools_bodies(tmp_path: Path) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            nonlocal active, peak
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert body
            with lock:
                active += 1
                peak = max(peak, active)
            release.wait(timeout=5.0)
            with lock:
                active -= 1
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address
    proxy = AdmissionServer(
        ("127.0.0.1", 0),
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        events_path=tmp_path / "events.jsonl",
        route_inflight_limits={**ROUTE_FORWARD_LIMITS, "/hands.detect": 2},
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()

    def post(body: bytes) -> tuple[int, bytes]:
        h, p = proxy.server_address[:2]
        conn = http.client.HTTPConnection(h, p, timeout=5)
        conn.request("POST", "/hands.detect", body=body, headers={"Content-Length": str(len(body))})
        response = conn.getresponse()
        result = int(response.status), response.read()
        conn.close()
        return result

    clients = [threading.Thread(target=post, args=(f"body-{i}".encode(),)) for i in range(20)]
    try:
        for client in clients:
            client.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                if peak == 2:
                    break
            time.sleep(0.01)
        with lock:
            assert peak == 2, f"proxy opened {peak} upstream connections instead of the internal worker bound"
        assert len(list((tmp_path / ".request-bodies").glob("*.body"))) <= 20
        release.set()
        for client in clients:
            client.join(timeout=5.0)
            assert not client.is_alive()
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        terminal = [row for row in events if row["event"] == "algorithm_request_terminal"]
        assert len(terminal) == 20
        assert {row["status"] for row in terminal} == {200}
        assert not list((tmp_path / ".request-bodies").glob("*.body"))
    finally:
        release.set()
        for client in clients:
            client.join(timeout=5.0)
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=5.0)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5.0)


def test_droid_atomic_route_keeps_frame_weight_and_internal_limit(tmp_path: Path) -> None:
    received: list[bytes] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address
    proxy = AdmissionServer(
        ("127.0.0.1", 0),
        upstreams={"droid": f"http://{host}:{port}"},
        events_path=tmp_path / "events.jsonl",
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        h, p = proxy.server_address[:2]
        conn = http.client.HTTPConnection(h, p, timeout=5.0)
        conn.request("POST", "/droid.infer", body=b"atomic", headers={"Content-Length": "6"})
        response = conn.getresponse()
        assert response.status == 200 and response.read() == b"ok"
        conn.close()
        assert received == [b"atomic"]
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        event = next(row for row in events if row["event"] == "algorithm_request_forwarded")
        assert event["request_image_units"] == 256
        assert event["configured_limit"] == 6
        assert event["queue_image_budget"] == 8000
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=5.0)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5.0)


def test_full_weighted_queue_blocks_before_spooling_offered_body(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            release.wait(timeout=5.0)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address
    proxy = AdmissionServer(
        ("127.0.0.1", 0),
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        events_path=tmp_path / "events.jsonl",
        route_inflight_limits={"/hands.detect": 1},
        route_queue_image_budgets={"/hands.detect": 1},
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    results: list[tuple[int, bytes]] = []

    def post(body: bytes) -> None:
        h, p = proxy.server_address[:2]
        conn = http.client.HTTPConnection(h, p, timeout=5.0)
        conn.request("POST", "/hands.detect", body=body, headers={"Content-Length": str(len(body))})
        response = conn.getresponse()
        results.append((response.status, response.read()))
        conn.close()

    first = threading.Thread(target=post, args=(b"first",))
    second = threading.Thread(target=post, args=(b"blocked-before-spool",))
    first.start()
    try:
        assert entered.wait(timeout=2.0)
        second.start()
        time.sleep(0.1)
        assert second.is_alive()
        assert len(list((tmp_path / ".request-bodies").glob("*.body"))) == 1
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert not first.is_alive() and not second.is_alive()
        assert sorted(results) == [(200, b"ok"), (200, b"ok")]
        assert not list((tmp_path / ".request-bodies").glob("*.body"))
    finally:
        release.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=5.0)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5.0)


def test_transport_outcome_is_terminal_without_automatic_replay() -> None:
    request = _request("/wilor.reconstruct")
    result = UpstreamResult(
        status=502,
        reason="Bad Gateway",
        headers=(),
        body=b"reset",
        error="ConnectionResetError(104, 'Connection reset by peer')",
        started_at_unix=time.time(),
        started_at_mono=time.monotonic(),
        finished_at_unix=time.time(),
        finished_at_mono=time.monotonic(),
    )
    reason = AdmissionServer.retry_failure_reason(request, result)
    assert reason is not None and "ambiguous" in reason
    request.failure_reason = reason
    terminal = AdmissionServer.annotate_retry_failure(request, result)
    assert terminal.status == 502
    assert b"reconcile before retrying" in terminal.body


def test_route_forward_limits_are_internal_not_external_quota() -> None:
    assert ROUTE_FORWARD_LIMITS["/hands.detect"] == 32
    assert ROUTE_FORWARD_LIMITS["/wilor.reconstruct"] == 32
    assert ROUTE_FORWARD_LIMITS["/droid.infer"] == 6
    assert proxy_module.AdmissionServer.request_queue_size >= 8000
    assert proxy_module.AdmissionServer.max_handler_threads == 256
