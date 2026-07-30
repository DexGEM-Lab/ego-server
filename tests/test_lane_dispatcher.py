from __future__ import annotations

import http.client
import json
import os
import socket
import sqlite3
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from ego_annotation.serving.binary_envelope import binary_envelope_iovecs, build_binary_envelope
from ego_annotation.serving.lane_dispatcher import (
    DispatcherProcess,
    LaneDispatcher,
    LeaseRegistry,
    _DispatcherHandler,
)
from ego_annotation.serving.transport import build_multipart_request_fields


@dataclass
class FakeBackend:
    name: str
    health_status: int = 200
    response_status: dict[str, int] = field(default_factory=dict)
    drop_paths: set[str] = field(default_factory=set)
    post_gate: threading.Semaphore | None = None
    posts: list[tuple[str, bytes, str]] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _post_condition: threading.Condition = field(default_factory=threading.Condition, init=False)
    _post_count: int = field(default=0, init=False)

    def start(self) -> None:
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                status = backend.health_status if self.path == "/status" else 404
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                body = self.rfile.read(length)
                with backend._post_condition:
                    backend.posts.append((self.path, body, self.headers.get("Content-Type", "")))
                    backend._post_count += 1
                    backend._post_condition.notify_all()
                if self.path in backend.drop_paths:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if backend.post_gate is not None:
                    backend.post_gate.acquire()
                status = backend.response_status.get(self.path, 200)
                if self.path == "/droid.create_session" and status == 200:
                    payload = json.dumps({"metadata": {"session_id": f"session-{backend.name}"}}).encode()
                else:
                    payload = f"{backend.name}:{self.path}".encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        import threading

        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    def wait_for_posts(self, count: int) -> None:
        with self._post_condition:
            while self._post_count < count:
                if not self._post_condition.wait(timeout=5.0):
                    raise AssertionError(f"{self.name} received {self._post_count} posts, expected {count}")


def _post(address: tuple[str, int], path: str, body: bytes, content_type: str = "application/json") -> tuple[int, bytes, str]:
    connection = http.client.HTTPConnection(*address, timeout=5)
    try:
        connection.request("POST", path, body=body, headers={"Content-Type": content_type, "Content-Length": str(len(body))})
        response = connection.getresponse()
        return response.status, response.read(), response.getheader("Content-Type", "")
    finally:
        connection.close()


@contextmanager
def _running(
    tmp_path: Path,
    *backends: FakeBackend,
    ttl_s: float = 3600.0,
    unidepth_replica_capacity: int = 8,
    health_timeout_s: float = 2.0,
) -> Iterator[tuple[DispatcherProcess, LeaseRegistry]]:
    stack = ExitStack()
    for backend in backends:
        backend.start()
        stack.callback(backend.close)
    registry = LeaseRegistry(str(tmp_path / "leases.sqlite3"), ttl_s=ttl_s)
    dispatcher = LaneDispatcher(
        registry=registry,
        droid_replicas=tuple(backend.url for backend in backends),
        unidepth_replicas=tuple(backend.url for backend in backends),
        unidepth_replica_capacity=unidepth_replica_capacity,
        backend_timeout_s=10.0,
        health_timeout_s=health_timeout_s,
    )
    process = DispatcherProcess(dispatcher, droid_port=0, unidepth_port=0)
    process.start()
    stack.callback(process.shutdown)
    try:
        yield process, registry
    finally:
        stack.close()


def test_droid_create_binds_replica(tmp_path: Path) -> None:
    first, second = FakeBackend("first"), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, registry):
        status, body, _content_type = _post(process.droid_address, "/droid.create_session", b"{}")
        assert status == 200
        assert json.loads(body) == {"metadata": {"session_id": "session-first"}}
        assert registry.lookup_session("session-first") == first.url
        assert [path for path, _body, _type in first.posts] == ["/droid.create_session"]
        assert second.posts == []


def test_droid_push_finalize_sticky(tmp_path: Path) -> None:
    first, second = FakeBackend("first"), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, registry):
        create_status, _body, _ = _post(process.droid_address, "/droid.create_session", b"{}")
        assert create_status == 200
        large_tensor = os.urandom(1024 * 1024)
        metadata = {"ownership": {"request_id": "r"}, "metadata": {"session_id": "session-first", "frame_id": "f"}}
        multipart, content_type = build_multipart_request_fields(
            metadata, {"rgb": (large_tensor, (len(large_tensor),), "uint8")},
        )
        push_status, _push_body, _ = _post(process.droid_address, "/droid.push_frame", multipart, content_type)
        assert push_status == 200
        assert first.posts[-1] == ("/droid.push_frame", multipart, content_type)
        assert not any(path == "/droid.push_frame" for path, _body, _type in second.posts)

        envelope_metadata = json.dumps({"metadata": {"session_id": "session-first"}}, separators=(",", ":")).encode()
        envelope = build_binary_envelope({
            "metadata": (envelope_metadata, (), "application/json"),
            "rgb": (large_tensor, (len(large_tensor),), "uint8"),
        })
        envelope_body = b"".join(binary_envelope_iovecs(envelope))
        finalize_status, _finalize_body, _ = _post(
            process.droid_address, "/droid.finalize", envelope_body, "application/vnd.ego.binary-envelope",
        )
        assert finalize_status == 200
        assert first.posts[-1] == ("/droid.finalize", envelope_body, "application/vnd.ego.binary-envelope")
        assert registry.lookup_session("session-first") is None


def test_droid_all_full_returns_429(tmp_path: Path) -> None:
    first, second = FakeBackend("first"), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, registry):
        for replica in (first.url, second.url):
            for index in range(8):
                assert registry.bind_session(f"{replica}-{index}", replica)
        status, body, content_type = _post(process.droid_address, "/droid.create_session", b"{}")
        assert status == 429
        assert content_type == "application/json"
        assert json.loads(body) == {"error": "droid_capacity_exhausted", "retryable": True}
        assert first.posts == second.posts == []


def test_droid_finalize_releases(tmp_path: Path) -> None:
    first = FakeBackend("first")
    with _running(tmp_path, first) as (process, registry):
        assert registry.bind_session("known-session", first.url)
        status, _body, _content_type = _post(
            process.droid_address, "/droid.finalize", json.dumps({"session_id": "known-session"}).encode(),
        )
        assert status == 200
        assert registry.lookup_session("known-session") is None
        with sqlite3.connect(registry.db_path) as connection:
            assert connection.execute("SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (first.url,)).fetchone() == (0,)


def _launch_posts(
    address: tuple[str, int], count: int, *, body_prefix: bytes = b"image",
) -> tuple[list[threading.Thread], list[tuple[int, bytes, str] | None]]:
    barrier = threading.Barrier(count + 1)
    results: list[tuple[int, bytes, str] | None] = [None] * count

    def request(index: int) -> None:
        barrier.wait()
        results[index] = _post(address, "/unidepth.infer", body_prefix + str(index).encode())

    threads = [threading.Thread(target=request, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    barrier.wait()
    return threads, results


def _join_posts(threads: list[threading.Thread], results: list[tuple[int, bytes, str] | None]) -> None:
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    assert all(result is not None and result[0] == 200 for result in results)


class _ObservedCondition(threading.Condition):
    """Test-only Condition observer that does not alter dispatcher notification flow."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_entered = threading.Semaphore(0)
        self.notify_all_calls = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.wait_entered.release()
        return super().wait(timeout)

    def notify_all(self) -> None:
        self.notify_all_calls += 1
        super().notify_all()


def _assert_unidepth_idle(dispatcher: LaneDispatcher) -> None:
    with dispatcher._unidepth_condition:
        assert dispatcher._unidepth_active == {replica: 0 for replica in dispatcher.unidepth_replicas}


def test_unidepth_sequential_requests_fill_first_lane(tmp_path: Path) -> None:
    first, second = FakeBackend("first"), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, _registry):
        responses = [_post(process.unidepth_address, "/unidepth.infer", f"image-{index}".encode()) for index in range(3)]
        assert responses == [(200, b"first:/unidepth.infer", "application/octet-stream")] * 3
        assert [body for _path, body, _content_type in first.posts] == [b"image-0", b"image-1", b"image-2"]
        assert second.posts == []


def test_unidepth_ninth_concurrent_request_uses_second_lane(tmp_path: Path) -> None:
    first, second = FakeBackend("first", post_gate=threading.Semaphore(0)), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, _registry):
        threads, results = _launch_posts(process.unidepth_address, 8)
        first.wait_for_posts(8)
        assert _post(process.unidepth_address, "/unidepth.infer", b"ninth")[1] == b"second:/unidepth.infer"
        assert len(first.posts) == 8
        assert second.posts[-1][:2] == ("/unidepth.infer", b"ninth")
        for _ in range(8):
            first.post_gate.release()
        _join_posts(threads, results)
        _assert_unidepth_idle(process.dispatcher)


def test_unidepth_reuses_first_lane_after_release(tmp_path: Path) -> None:
    first, second = FakeBackend("first", post_gate=threading.Semaphore(0)), FakeBackend("second")
    with _running(tmp_path, first, second, unidepth_replica_capacity=1) as (process, _registry):
        first_thread, first_results = _launch_posts(process.unidepth_address, 1)
        first.wait_for_posts(1)
        assert _post(process.unidepth_address, "/unidepth.infer", b"overflow")[1] == b"second:/unidepth.infer"
        first.post_gate.release()
        _join_posts(first_thread, first_results)
        reused_thread, reused_results = _launch_posts(process.unidepth_address, 1, body_prefix=b"reused")
        first.wait_for_posts(2)
        first.post_gate.release()
        _join_posts(reused_thread, reused_results)
        assert reused_results == [(200, b"first:/unidepth.infer", "application/octet-stream")]
        assert [body for _path, body, _content_type in first.posts] == [b"image0", b"reused0"]
        _assert_unidepth_idle(process.dispatcher)


def test_unidepth_seventeenth_request_waits_for_capacity(tmp_path: Path) -> None:
    first = FakeBackend("first", post_gate=threading.Semaphore(0))
    second = FakeBackend("second", post_gate=threading.Semaphore(0))
    with _running(tmp_path, first, second) as (process, _registry):
        threads, results = _launch_posts(process.unidepth_address, 16)
        first.wait_for_posts(8)
        second.wait_for_posts(8)
        observed_condition = _ObservedCondition()
        process.dispatcher._unidepth_condition = observed_condition
        with observed_condition:
            assert process.dispatcher._unidepth_active == {first.url: 8, second.url: 8}

        seventeenth_result: list[tuple[int, bytes, str]] = []

        def seventeenth_request() -> None:
            seventeenth_result.append(_post(process.unidepth_address, "/unidepth.infer", b"seventeenth"))

        seventeenth = threading.Thread(target=seventeenth_request)
        seventeenth.start()
        assert observed_condition.wait_entered.acquire(timeout=5.0)

        first.post_gate.release()
        first.wait_for_posts(9)
        for _ in range(8):
            first.post_gate.release()
        for _ in range(8):
            second.post_gate.release()
        seventeenth.join(timeout=5.0)
        assert not seventeenth.is_alive()
        assert seventeenth_result == [(200, b"first:/unidepth.infer", "application/octet-stream")]
        _join_posts(threads, results)
        _assert_unidepth_idle(process.dispatcher)


def test_unidepth_two_waiters_sleep_until_releases_without_peer_wakes(tmp_path: Path) -> None:
    first = FakeBackend("first", post_gate=threading.Semaphore(0))
    second = FakeBackend("second", post_gate=threading.Semaphore(0))
    with _running(
        tmp_path, first, second, unidepth_replica_capacity=1, health_timeout_s=30.0,
    ) as (process, _registry):
        occupied_threads, occupied_results = _launch_posts(process.unidepth_address, 2)
        first.wait_for_posts(1)
        second.wait_for_posts(1)

        observed_condition = _ObservedCondition()
        process.dispatcher._unidepth_condition = observed_condition
        healthy_replicas = process.dispatcher._healthy_replicas
        health_probe_lock = threading.Lock()
        health_probe_count = 0

        def count_health_probes(replicas: tuple[str, ...]) -> tuple[str, ...]:
            nonlocal health_probe_count
            with health_probe_lock:
                health_probe_count += 1
            return healthy_replicas(replicas)

        process.dispatcher._healthy_replicas = count_health_probes  # type: ignore[method-assign]
        waiting_threads, waiting_results = _launch_posts(process.unidepth_address, 2, body_prefix=b"waiting")
        assert observed_condition.wait_entered.acquire(timeout=5.0)
        assert observed_condition.wait_entered.acquire(timeout=5.0)
        with observed_condition:
            assert observed_condition.notify_all_calls == 0
            assert process.dispatcher._unidepth_active == {first.url: 1, second.url: 1}
        with health_probe_lock:
            assert health_probe_count == 2
        assert len(first.posts) == len(second.posts) == 1

        first.post_gate.release()
        first.wait_for_posts(2)
        assert len(second.posts) == 1
        first.post_gate.release()
        first.wait_for_posts(3)

        first.post_gate.release()
        second.post_gate.release()
        _join_posts(waiting_threads, waiting_results)
        _join_posts(occupied_threads, occupied_results)
        _assert_unidepth_idle(process.dispatcher)


def test_unidepth_retry_releases_primary_before_reserving_alternate(tmp_path: Path) -> None:
    first = FakeBackend("first")
    second = FakeBackend("second", post_gate=threading.Semaphore(0))
    first.response_status["/unidepth.infer"] = 503
    with _running(tmp_path, first, second, unidepth_replica_capacity=1) as (process, _registry):
        results: list[tuple[int, bytes, str]] = []
        request = threading.Thread(
            target=lambda: results.append(_post(process.unidepth_address, "/unidepth.infer", b"retry")),
        )
        request.start()
        first.wait_for_posts(1)
        second.wait_for_posts(1)
        with process.dispatcher._unidepth_condition:
            assert process.dispatcher._unidepth_active == {first.url: 0, second.url: 1}
        second.post_gate.release()
        request.join(timeout=5.0)
        assert not request.is_alive()
        assert results == [(200, b"second:/unidepth.infer", "application/octet-stream")]
        assert first.posts[-1][:2] == ("/unidepth.infer", b"retry")
        assert second.posts[-1][:2] == ("/unidepth.infer", b"retry")
        _assert_unidepth_idle(process.dispatcher)


def test_unidepth_connection_http_and_relay_failures_release_capacity(tmp_path: Path, monkeypatch: object) -> None:
    first, second = FakeBackend("first"), FakeBackend("second")
    first.drop_paths.add("/unidepth.infer")
    with _running(tmp_path, first, second, unidepth_replica_capacity=1) as (process, _registry):
        assert _post(process.unidepth_address, "/unidepth.infer", b"connection")[1] == b"second:/unidepth.infer"
        _assert_unidepth_idle(process.dispatcher)

        first.drop_paths.clear()
        first.response_status["/unidepth.infer"] = 500
        assert _post(process.unidepth_address, "/unidepth.infer", b"http-error")[0] == 500
        _assert_unidepth_idle(process.dispatcher)

        first.response_status.clear()
        original_send = _DispatcherHandler._send

        def fail_relay(self: _DispatcherHandler, status: int, headers: object, body: bytes) -> None:
            raise BrokenPipeError("test client relay failure")

        monkeypatch.setattr(_DispatcherHandler, "_send", fail_relay)
        process.unidepth_server.handle_error = lambda _request, _client_address: None
        try:
            _post(process.unidepth_address, "/unidepth.infer", b"relay")
        except http.client.RemoteDisconnected:
            pass
        else:
            raise AssertionError("client should not receive a relayed response")
        finally:
            monkeypatch.setattr(_DispatcherHandler, "_send", original_send)
        _assert_unidepth_idle(process.dispatcher)


def test_replica_down_skip(tmp_path: Path) -> None:
    down, healthy = FakeBackend("down", health_status=503), FakeBackend("healthy")
    with _running(tmp_path, down, healthy) as (process, registry):
        droid_status, _droid_body, _ = _post(process.droid_address, "/droid.create_session", b"{}")
        unidepth_status, unidepth_body, _ = _post(process.unidepth_address, "/unidepth.infer", b"pixels")
        assert droid_status == 200
        assert registry.lookup_session("session-healthy") == healthy.url
        assert (unidepth_status, unidepth_body) == (200, b"healthy:/unidepth.infer")
        assert down.posts == []


def test_ttl_cleanup(tmp_path: Path) -> None:
    backend = FakeBackend("first")
    with _running(tmp_path, backend, ttl_s=1.0) as (process, registry):
        assert registry.bind_session("expired", backend.url, created_monotonic=0.0)
        assert registry.cleanup_expired(now_monotonic=2.0) == 1
        assert registry.lookup_session("expired") is None
        with sqlite3.connect(registry.db_path) as connection:
            assert connection.execute("SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (backend.url,)).fetchone() == (0,)
        status, _body, _ = _post(process.droid_address, "/droid.create_session", b"{}")
        assert status == 200
