from __future__ import annotations

import http.client
import json
import socket
import sqlite3
import sys
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

# The deployment bundle is the artifact under test; do not accidentally import
# the similarly named development package from the repository root.
RELEASE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "ego_annotation" or module_name.startswith("ego_annotation."):
        sys.modules.pop(module_name)
from ego_annotation.serving.lane_dispatcher import (  # noqa: E402
    DROID_INFER_BACKEND_TIMEOUT_S,
    DispatcherProcess,
    LaneDispatcher,
    LeaseRegistry,
    _DispatcherHandler,
)


@dataclass
class FakeBackend:
    name: str
    gate: threading.Semaphore | None = None
    drop_paths: set[str] = field(default_factory=set)
    posts: list[str] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _condition: threading.Condition = field(default_factory=threading.Condition, init=False)

    def start(self) -> None:
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                self.send_response(200 if self.path == "/status" else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                with backend._condition:
                    backend.posts.append(self.path)
                    backend._condition.notify_all()
                if self.path in backend.drop_paths:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if backend.gate is not None:
                    backend.gate.acquire()
                body = f"{backend.name}:{self.path}".encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def wait_for_posts(self, count: int) -> None:
        with self._condition:
            while len(self.posts) < count:
                if not self._condition.wait(5.0):
                    raise AssertionError(f"{self.name} has {len(self.posts)} posts, expected {count}")

    def close(self) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()


def _post(address: tuple[str, int], path: str) -> tuple[int, bytes, str]:
    connection = http.client.HTTPConnection(*address, timeout=5)
    try:
        connection.request("POST", path, body=b"request", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read(), response.getheader("Content-Type", "")
    finally:
        connection.close()


@contextmanager
def _running(tmp_path: Path, *backends: FakeBackend) -> Iterator[tuple[DispatcherProcess, LeaseRegistry]]:
    stack = ExitStack()
    for backend in backends:
        backend.start()
        stack.callback(backend.close)
    registry = LeaseRegistry(str(tmp_path / "leases.sqlite3"))
    process = DispatcherProcess(LaneDispatcher(
        registry=registry,
        droid_replicas=tuple(backend.url for backend in backends),
        unidepth_replicas=tuple(backend.url for backend in backends),
        health_timeout_s=1.0,
    ), droid_port=0, unidepth_port=0)
    process.start()
    stack.callback(process.shutdown)
    try:
        yield process, registry
    finally:
        stack.close()


def _slots(registry: LeaseRegistry, replica: str) -> int:
    with sqlite3.connect(registry.db_path) as connection:
        row = connection.execute("SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (replica,)).fetchone()
    assert row is not None
    return int(row[0])


def test_infer_least_active_alias_and_release(tmp_path: Path) -> None:
    first, second = FakeBackend("first", gate=threading.Semaphore(0)), FakeBackend("second")
    with _running(tmp_path, first, second) as (process, registry):
        result: list[tuple[int, bytes, str]] = []
        thread = threading.Thread(target=lambda: result.append(_post(process.droid_address, "/droid.infer")))
        thread.start()
        first.wait_for_posts(1)
        assert _slots(registry, first.url) == 1
        assert registry.lookup_session("any-session") is None

        assert _post(process.droid_address, "/infer")[:2] == (200, b"second:/infer")
        first.gate.release()
        thread.join(5.0)
        assert not thread.is_alive()
        assert result == [(200, b"first:/droid.infer", "application/octet-stream")]
        assert _slots(registry, first.url) == _slots(registry, second.url) == 0


def test_infer_backpressure_and_upstream_disconnect_release_slots(tmp_path: Path) -> None:
    backends = tuple(FakeBackend(str(index)) for index in range(6))
    with _running(tmp_path, *backends) as (process, registry):
        reservations = [registry.reserve_replica_for_infer(tuple(backend.url for backend in backends)) for _ in backends]
        assert all(reservation is not None for reservation in reservations)
        response, status, payload = process.dispatcher.dispatch("/droid.infer", b"overflow", ())
        assert response is None
        assert status == 429
        assert payload == {"error": "droid_infer_capacity_exhausted", "retryable": True}
        for reservation in reservations:
            assert reservation is not None
            registry.release_replica_infer(reservation)
        assert all(_slots(registry, backend.url) == 0 for backend in backends)

    failed = FakeBackend("failed", drop_paths={"/droid.infer"})
    with _running(tmp_path, failed) as (process, registry):
        assert _post(process.droid_address, "/droid.infer")[:2] == (
            503, b'{"error":"droid_infer_unavailable","retryable":true}',
        )
        assert _slots(registry, failed.url) == 0


def test_infer_timeout_and_client_relay_disconnect_do_not_strand_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeBackend("one")
    with _running(tmp_path, backend) as (process, registry):
        forward = process.dispatcher._forward
        timeouts: list[float | None] = []

        def observe(replica: str, path: str, body: bytes, headers: object, *, timeout_s: float | None = None) -> object:
            timeouts.append(timeout_s)
            return forward(replica, path, body, headers, timeout_s=timeout_s)  # type: ignore[arg-type]

        monkeypatch.setattr(process.dispatcher, "_forward", observe)
        assert _post(process.droid_address, "/droid.infer")[0] == 200
        assert timeouts == [DROID_INFER_BACKEND_TIMEOUT_S]
        assert _slots(registry, backend.url) == 0

        original_send = _DispatcherHandler._send
        monkeypatch.setattr(_DispatcherHandler, "_send", lambda *_args: (_ for _ in ()).throw(BrokenPipeError("disconnect")))
        process.droid_server.handle_error = lambda _request, _client_address: None
        with pytest.raises(http.client.RemoteDisconnected):
            _post(process.droid_address, "/droid.infer")
        monkeypatch.setattr(_DispatcherHandler, "_send", original_send)
        assert _slots(registry, backend.url) == 0
