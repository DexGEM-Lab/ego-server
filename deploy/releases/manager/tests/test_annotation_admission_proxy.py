from __future__ import annotations

import http.client
import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import scripts.annotation_admission_proxy as proxy_module
from scripts.annotation_admission_proxy import (
    AdmissionServer,
    CLIENT_BATCH_CAPS,
    CLIENT_BATCH_WAIT_S,
    ROUTE_TO_SERVICE,
    route_uses_cross_process_slot,
)


def _start_proxy(
    tmp_path: Path,
    *,
    upstreams: dict[str, str],
    batch_caps: dict[str, int] | None = None,
    batch_waits: dict[str, float] | None = None,
) -> tuple[AdmissionServer, threading.Thread]:
    proxy = AdmissionServer(
        ("127.0.0.1", 0),
        upstreams=upstreams,
        events_path=tmp_path / "proxy_events.jsonl",
        batch_caps=batch_caps,
        batch_waits=batch_waits,
    )
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    return proxy, thread


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def _post(proxy: AdmissionServer, route: str, body: bytes, *, job_id: str = "job") -> tuple[int, bytes, dict[str, str]]:
    host, port = proxy.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5.0)
    connection.request(
        "POST",
        route,
        body=body,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
            "X-Ego-Video-Job-Id": job_id,
        },
    )
    response = connection.getresponse()
    result = int(response.status), response.read(), {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return result


def test_client_scheduler_uses_model_batch_caps_without_route_semaphores() -> None:
    assert CLIENT_BATCH_CAPS == {
        "/unidepth.infer": 8,
        "/hands.detect": 8,
        "/wilor.reconstruct": 16,
        "/droid.create_session": 1,
        "/droid.push_frame": 1,
        "/droid.finalize": 1,
        "/hawor.infer_tracks": 8,
        "/hawor_infiller.fill": 4,
        "/cosmos3.reason": 1,
    }
    assert CLIENT_BATCH_WAIT_S["/hawor.infer_tracks"] == 0.250
    assert CLIENT_BATCH_WAIT_S["/cosmos3.reason"] == 0.0
    assert not route_uses_cross_process_slot("unidepth.infer")
    assert not route_uses_cross_process_slot("droid.session")


def test_fully_buffered_group_is_forwarded_concurrently(tmp_path: Path) -> None:
    entered = 0
    entered_lock = threading.Lock()
    both_entered = threading.Event()
    release = threading.Event()
    arrivals: list[float] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            nonlocal entered
            self.rfile.read(int(self.headers["Content-Length"]))
            with entered_lock:
                entered += 1
                arrivals.append(time.monotonic())
                if entered == 2:
                    both_entered.set()
            release.wait(timeout=5.0)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(
        tmp_path,
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        batch_caps={"/hands.detect": 2},
        batch_waits={"/hands.detect": 1.0},
    )
    responses: list[tuple[int, bytes, dict[str, str]]] = []
    clients = [threading.Thread(target=lambda body=body: responses.append(_post(proxy, "/hands.detect", body))) for body in (b"a", b"b")]
    try:
        for client in clients:
            client.start()
        assert both_entered.wait(timeout=2.0), "a serial forwarding loop cannot enter both blocking upstream requests"
        assert len(arrivals) == 2 and max(arrivals) - min(arrivals) < 0.1
        release.set()
        for client in clients:
            client.join(timeout=5.0)
            assert not client.is_alive()
        assert sorted((status, body) for status, body, _headers in responses) == [(200, b"ok"), (200, b"ok")]
        events = [json.loads(line) for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()]
        attempts = [row for row in events if row["event"] == "algorithm_request_forwarded"]
        assert len(attempts) == 2
        assert {row["batch_size"] for row in attempts} == {2}
        assert len({row["batch_id"] for row in attempts}) == 1
        assert {row["configured_limit"] for row in attempts} == {None}
    finally:
        release.set()
        for client in clients:
            client.join(timeout=5.0)
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_next_group_waits_for_previous_group_completion(tmp_path: Path) -> None:
    entered = 0
    entered_lock = threading.Lock()
    first_group_entered = threading.Event()
    release = threading.Event()

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            nonlocal entered
            self.rfile.read(int(self.headers["Content-Length"]))
            with entered_lock:
                entered += 1
                if entered == 2:
                    first_group_entered.set()
            release.wait(timeout=5.0)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(
        tmp_path,
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        batch_caps={"/hands.detect": 2},
        batch_waits={"/hands.detect": 0.1},
    )
    responses: list[tuple[int, bytes, dict[str, str]]] = []
    clients = [
        threading.Thread(target=lambda body=body: responses.append(_post(proxy, "/hands.detect", body)))
        for body in (b"a", b"b", b"c", b"d")
    ]
    try:
        for client in clients:
            client.start()
        assert first_group_entered.wait(timeout=2.0)
        time.sleep(0.1)
        assert entered == 2, "the scheduler released a second group before the first completed"
        release.set()
        for client in clients:
            client.join(timeout=5.0)
            assert not client.is_alive()
        assert len(responses) == 4 and all(status == 200 for status, _body, _headers in responses)
        events = [json.loads(line) for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()]
        attempts = [row for row in events if row["event"] == "algorithm_request_forwarded"]
        assert len(attempts) == 4
        assert {row["batch_size"] for row in attempts} == {2}
        assert len({row["batch_id"] for row in attempts}) == 2
    finally:
        release.set()
        for client in clients:
            client.join(timeout=5.0)
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_partial_tail_flushes_after_group_deadline(tmp_path: Path) -> None:
    received: list[float] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append(time.monotonic())
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(
        tmp_path,
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        batch_caps={"/hands.detect": 4},
        batch_waits={"/hands.detect": 0.050},
    )
    started = time.monotonic()
    try:
        assert _post(proxy, "/hands.detect", b"tail")[0] == 201
        elapsed = received[0] - started
        assert 0.035 <= elapsed < 0.5
        event = next(
            json.loads(line)
            for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "algorithm_request_forwarded"
        )
        assert event["batch_size"] == 1
    finally:
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_only_429_members_reenter_retry_queue(tmp_path: Path) -> None:
    attempts: Counter[bytes] = Counter()

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            attempts[body] += 1
            status = 429 if body == b"retry" and attempts[body] == 1 else 200
            response_body = b"busy" if status == 429 else b"ok-" + body
            self.send_response(status)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(
        tmp_path,
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        batch_caps={"/hands.detect": 2},
        batch_waits={"/hands.detect": 0.020},
    )
    responses: dict[bytes, tuple[int, bytes, dict[str, str]]] = {}
    clients = [threading.Thread(target=lambda body=body: responses.__setitem__(body, _post(proxy, "/hands.detect", body))) for body in (b"retry", b"accepted")]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(timeout=5.0)
            assert not client.is_alive()
        assert attempts == Counter({b"retry": 2, b"accepted": 1})
        assert responses[b"retry"][:2] == (200, b"ok-retry")
        assert responses[b"accepted"][:2] == (200, b"ok-accepted")
        events = [json.loads(line) for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()]
        forwarded = [row for row in events if row["event"] == "algorithm_request_forwarded"]
        retry_rows = [row for row in forwarded if row["retry_count"] > 0]
        assert len(forwarded) == 3 and len(retry_rows) == 2
        assert [row["status"] for row in retry_rows] == [429, 200]
        assert [row["terminal"] for row in retry_rows] == [False, True]
    finally:
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_retry_deadline_surfaces_final_429_and_marks_proxy_retry_complete(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(proxy_module, "RETRY_DEADLINE_S", 0.040)
    monkeypatch.setattr(proxy_module, "RETRY_INITIAL_DELAY_S", 0.005)
    attempts = 0

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            nonlocal attempts
            attempts += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"busy")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(
        tmp_path,
        upstreams={"hands_wilor": f"http://{host}:{port}"},
        batch_caps={"/hands.detect": 1},
        batch_waits={"/hands.detect": 0.0},
    )
    try:
        status, body, headers = _post(proxy, "/hands.detect", b"always-busy")
        assert (status, body) == (429, b"busy")
        assert headers["x-ego-admission-retry-complete"] == "1"
        assert attempts >= 2
    finally:
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_droid_keeps_local_protocol_ownership_without_capacity_slots(tmp_path: Path) -> None:
    received: list[str] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            received.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(201)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(tmp_path, upstreams={"droid": f"http://{host}:{port}"})
    try:
        assert _post(proxy, "/droid.create_session", b"create", job_id="job-a")[0] == 201
        assert proxy.droid_session_is_active("job-a")
        assert _post(proxy, "/droid.push_frame", b"wrong", job_id="job-b")[0] == 409
        assert _post(proxy, "/droid.push_frame", b"push", job_id="job-a")[0] == 201
        assert _post(proxy, "/droid.finalize", b"finish", job_id="job-a")[0] == 201
        assert not proxy.droid_session_is_active("job-a")
        assert received == ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
        assert not hasattr(proxy, "semaphores")
    finally:
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)


def test_cosmos_remains_direct_and_native_body_is_unchanged(tmp_path: Path) -> None:
    received: list[tuple[str, bytes]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, body))
            self.send_response(201)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    host, port = upstream.server_address[:2]
    proxy, proxy_thread = _start_proxy(tmp_path, upstreams={"cosmos3": f"http://{host}:{port}"})
    try:
        body = b"native-cosmos-multipart"
        assert _post(proxy, "/cosmos3.reason", body)[0] == 201
        assert ROUTE_TO_SERVICE["/cosmos3.reason"] == "cosmos3"
        assert received == [("/cosmos3.reason", body)]
        event = next(
            json.loads(line)
            for line in (tmp_path / "proxy_events.jsonl").read_text().splitlines()
            if json.loads(line)["event"] == "algorithm_request_forwarded"
        )
        assert event["batch_cap"] == 1 and event["batch_size"] == 1
    finally:
        _stop_server(proxy, proxy_thread)
        _stop_server(upstream, upstream_thread)
