#!/usr/bin/env python3
"""Client-side batch-aware admission proxy for the frozen service routes.

Every incoming HTTP body is fully buffered before it enters a route-local ready
queue.  The queue groups compatible requests up to the service's native batch
cap and releases one group at a time: each request remains an independent HTTP
POST and each handler forwards its own request concurrently.  Waiting for all
members of a group to complete prevents client flooding of the service queue
without adding a fixed per-algorithm multiplier.  A 429 is a
transient service-capacity signal: that request is returned to a bounded-time
retry queue and is never duplicated with the other members of the group.

This module deliberately has no per-algorithm ``*2`` semaphore.  Ray Serve (or
vLLM for Cosmos) remains the service-side queue owner.  The old multiplier and
limit helpers remain import-compatible for older launchers, but are not used to
admit ordinary requests.
"""
from __future__ import annotations

import heapq
import http.client
import json
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit

# DROID's stateful protocol remains serial per video/session.  These values are
# compatibility metadata only; they are not request semaphores.
DROID_REPLICA_COUNT = 6
DROID_RESIDENT_SESSIONS_PER_REPLICA = 8
DROID_SESSION_NODE_CAPACITY = DROID_REPLICA_COUNT * DROID_RESIDENT_SESSIONS_PER_REPLICA

# Retained as a compatibility view for old reports and callers.  No ordinary
# route uses these values as a B-layer admission limit anymore.
RAY_SERVE_ONGOING_LIMITS = {
    "unidepth.infer": 16,
    "hands.detect": 16,
    "wilor.reconstruct": 32,
    "droid.session": DROID_SESSION_NODE_CAPACITY,
    "hawor.infer_tracks": 8,
    "hawor_infiller.fill": 4,
    "cosmos3.reason": 16,
}

# Native model batch caps.  DROID and Cosmos are deliberately direct: DROID
# pushes are protocol-serial per session and Cosmos owns its vLLM scheduler.
CLIENT_BATCH_CAPS = {
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

# A partial group must eventually flush.  These are client grouping deadlines,
# not service compute or Ray Serve wait-time claims.
CLIENT_BATCH_WAIT_S = {
    "/unidepth.infer": 0.040,
    "/hands.detect": 0.040,
    "/wilor.reconstruct": 0.040,
    "/droid.create_session": 0.0,
    "/droid.push_frame": 0.0,
    "/droid.finalize": 0.0,
    "/hawor.infer_tracks": 0.250,
    "/hawor_infiller.fill": 0.200,
    "/cosmos3.reason": 0.0,
}

# A proxy handler waits on the same connection, so retries must terminate
# before the established client timeout.  Retry timing is intentionally short
# and capped; successful requests wake the retry scheduler immediately.
# The client does not group or throttle first attempts. Only requests that
# the service rejects as capacity/transient failures wait in the retry queue.
# 250ms is the retry cadence; Retry-After is honored when provided.
RETRY_RETRY_DELAY_S = 0.250
RETRY_AFTER_MAX_S = 5.0

ROUTE_TO_SERVICE = {
    "/unidepth.infer": "unidepth",
    "/hands.detect": "hands_wilor",
    "/wilor.reconstruct": "wilor",
    "/droid.create_session": "droid",
    "/droid.push_frame": "droid",
    "/droid.finalize": "droid",
    "/hawor.infer_tracks": "hawor",
    "/hawor_infiller.fill": "hawor",
    "/cosmos3.reason": "cosmos3",
}

ROUTE_TO_LIMIT_NAME = {
    "/unidepth.infer": "unidepth.infer",
    "/hands.detect": "hands.detect",
    "/wilor.reconstruct": "wilor.reconstruct",
    "/droid.create_session": "droid.session",
    "/droid.push_frame": "droid.session",
    "/droid.finalize": "droid.session",
    "/hawor.infer_tracks": "hawor.infer_tracks",
    "/hawor_infiller.fill": "hawor_infiller.fill",
    "/cosmos3.reason": "cosmos3.reason",
}


def utc_iso_from_unix(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def admission_limits(multiplier: int) -> dict[str, int]:
    """Return no B-layer limits; retained only for launcher compatibility."""
    if multiplier <= 0:
        raise ValueError("algorithm inflight multiplier must be positive")
    return {}


def route_uses_cross_process_slot(limit_name: str) -> bool:
    """The batch scheduler no longer uses request-scoped file slots."""
    return False


def load_upstreams(profile: Path, overrides: dict[str, str] | None = None) -> dict[str, str]:
    payload = json.loads(profile.read_text(encoding="utf-8"))
    services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(services, dict):
        raise ValueError(f"invalid service profile: {profile}")
    result: dict[str, str] = {}
    for service in set(ROUTE_TO_SERVICE.values()):
        row = services.get(service)
        base_url = (overrides or {}).get(service) or (row.get("base_url") if isinstance(row, dict) else None)
        if isinstance(base_url, str) and base_url.startswith(("http://", "https://")):
            result[service] = base_url.rstrip("/")
        elif service != "cosmos3":
            raise ValueError(f"missing service base URL: {service}")
    return result


@dataclass(frozen=True)
class UpstreamResult:
    status: int
    reason: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    error: str | None
    started_at_unix: float
    started_at_mono: float
    finished_at_unix: float
    finished_at_mono: float


@dataclass
class PendingRequest:
    route: str
    service: str
    body: bytes
    headers: tuple[tuple[str, str], ...]
    video_job_id: str | None
    video_item_id: str | None
    received_at_unix: float
    received_at_mono: float
    queued_at_mono: float
    logical_id: str
    retry_count: int = 0
    attempt: int = 0
    batch_id: str | None = None
    batch_size: int = 1
    dispatch_event: threading.Event = field(default_factory=threading.Event)
    dispatch_guard: threading.Lock = field(default_factory=threading.Lock)
    result: UpstreamResult | None = None
    final: bool = False
    next_retry_at_mono: float | None = None

    def arm_dispatch(self) -> None:
        with self.dispatch_guard:
            self.dispatch_event = threading.Event()

    def release_dispatch(self) -> None:
        with self.dispatch_guard:
            self.dispatch_event.set()

    def wait_for_dispatch(self) -> None:
        with self.dispatch_guard:
            event = self.dispatch_event
        event.wait()


class BatchScheduler:
    """Route-local ready/retry queue that releases compatible groups together."""

    def __init__(self, route: str):
        self.route = route
        self._ready: deque[PendingRequest] = deque()
        self._retry_heap: list[tuple[float, int, PendingRequest]] = []
        self._sequence = 0
        self._batch_sequence = 0
        self._active_batch_id: str | None = None
        self._active_batch_remaining = 0
        self._scheduler_id = uuid.uuid4().hex[:12]
        self._condition = threading.Condition()
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name=f"batch-scheduler-{route}", daemon=True)
        self._thread.start()

    def enqueue(self, request: PendingRequest) -> None:
        with self._condition:
            request.queued_at_mono = time.monotonic()
            request.next_retry_at_mono = None
            self._ready.append(request)
            self._condition.notify_all()

    def requeue_429(self, request: PendingRequest, retry_after_s: float | None) -> bool:
        now = time.monotonic()
        request.retry_count += 1
        request.arm_dispatch()
        request.next_retry_at_mono = now + self._retry_delay(request.retry_count, retry_after_s)
        with self._condition:
            self._sequence += 1
            heapq.heappush(self._retry_heap, (request.next_retry_at_mono, self._sequence, request))
            self._condition.notify_all()
        return True

    def complete_batch_member(self, batch_id: str | None) -> None:
        """Open the next group only after every member has completed an attempt."""
        if not batch_id:
            return
        with self._condition:
            if batch_id != self._active_batch_id or self._active_batch_remaining <= 0:
                return
            self._active_batch_remaining -= 1
            if self._active_batch_remaining == 0:
                self._active_batch_id = None
                self._condition.notify_all()

    def wake_retry(self) -> None:
        """Use a successful forward as an immediate capacity hint."""
        with self._condition:
            if self._retry_heap:
                _when, sequence, request = heapq.heappop(self._retry_heap)
                self._sequence += 1
                request.next_retry_at_mono = time.monotonic()
                heapq.heappush(self._retry_heap, (request.next_retry_at_mono, self._sequence, request))
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    @staticmethod
    def _retry_delay(retry_count: int, retry_after_s: float | None) -> float:
        if retry_after_s is not None:
            return min(RETRY_AFTER_MAX_S, max(0.0, retry_after_s))
        return RETRY_RETRY_DELAY_S

    def _promote_due_retries(self, now: float) -> None:
        while self._retry_heap and self._retry_heap[0][0] <= now:
            _when, _sequence, request = heapq.heappop(self._retry_heap)
            request.queued_at_mono = now
            request.next_retry_at_mono = None
            self._ready.append(request)

    def _take_group(self) -> list[PendingRequest]:
        request = self._ready.popleft()
        self._batch_sequence += 1
        batch_id = f"{self.route.replace('.', '_').replace('/', '_')}-{self._scheduler_id}-{self._batch_sequence:08d}"
        self._active_batch_id = batch_id
        self._active_batch_remaining = 1
        request.batch_id = batch_id
        request.batch_size = 1
        request.release_dispatch()
        return [request]

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    now = time.monotonic()
                    self._promote_due_retries(now)
                    if self._stopping:
                        return
                    if self._active_batch_id is not None:
                        self._condition.wait()
                        continue
                    if self._ready:
                        self._take_group()
                        break
                    elif self._retry_heap:
                        timeout = max(0.0, self._retry_heap[0][0] - now)
                    else:
                        timeout = None
                    if self._stopping:
                        return
                    self._condition.wait(timeout=timeout)


class AdmissionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstreams: dict[str, str],
        events_path: Path,
        batch_caps: dict[str, int] | None = None,
        batch_waits: dict[str, float] | None = None,
        # Compatibility parameters accepted by older launchers; intentionally unused.
        limits: dict[str, int] | None = None,
        lock_root: Path | None = None,
    ):
        super().__init__(address, AdmissionHandler)
        self.upstreams = upstreams
        self.events_path = events_path
        self.lock_root = lock_root
        self.batch_caps = dict(CLIENT_BATCH_CAPS if batch_caps is None else batch_caps)
        self.batch_waits = dict(CLIENT_BATCH_WAIT_S if batch_waits is None else batch_waits)
        self.schedulers = {route: BatchScheduler(route) for route in ROUTE_TO_SERVICE}
        # Local lifecycle ownership only; no capacity pool and no hash slots.
        self.droid_session_locks: dict[str, object] = {}
        self.droid_session_locks_guard = threading.Lock()
        self.events_lock = threading.Lock()

    def droid_session_is_active(self, job_id: str) -> bool:
        with self.droid_session_locks_guard:
            return job_id in self.droid_session_locks

    def acquire_droid_session_lock(self, job_id: str) -> tuple[object, None]:
        with self.droid_session_locks_guard:
            if job_id in self.droid_session_locks:
                raise RuntimeError(f"DROID session is already active for job {job_id!r}")
            marker = object()
            self.droid_session_locks[job_id] = marker
            return marker, None

    def release_droid_session_lock(self, job_id: str) -> None:
        with self.droid_session_locks_guard:
            self.droid_session_locks.pop(job_id, None)

    def enqueue(self, request: PendingRequest) -> None:
        self.schedulers[request.route].enqueue(request)

    def retry_429(self, request: PendingRequest, retry_after_s: float | None) -> bool:
        return self.schedulers[request.route].requeue_429(request, retry_after_s)

    def complete_batch_member(self, route: str, batch_id: str | None) -> None:
        self.schedulers[route].complete_batch_member(batch_id)

    def wake_retry(self, route: str) -> None:
        self.schedulers[route].wake_retry()

    def record(self, row: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()

    def forward(self, request: PendingRequest) -> UpstreamResult:
        upstream = urlsplit(self.upstreams[request.service])
        connection: http.client.HTTPConnection | None = None
        started_unix = time.time()
        started_mono = time.monotonic()
        status = 502
        reason = "Bad Gateway"
        response_body = b""
        response_headers: tuple[tuple[str, str], ...] = ()
        error: str | None = None
        try:
            connection_class = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
            connection = connection_class(upstream.hostname, upstream.port, timeout=86400.0)
            headers = {
                key: value
                for key, value in request.headers
                if key.lower() not in {"host", "connection", "proxy-connection", "transfer-encoding", "content-length"}
            }
            headers["Content-Length"] = str(len(request.body))
            target = f"{upstream.path.rstrip('/')}{request.route}" if upstream.path else request.route
            connection.request("POST", target, body=request.body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            status = int(response.status)
            reason = str(response.reason or "")
            response_headers = tuple(
                (key, value)
                for key, value in response.getheaders()
                if key.lower() not in {"connection", "transfer-encoding", "content-length"}
            )
        except Exception as exc:
            error = repr(exc)
            response_body = json.dumps({"error": "admission_proxy_upstream_failure", "detail": error}).encode("utf-8")
            response_headers = (("Content-Type", "application/json"),)
        finally:
            if connection is not None:
                connection.close()
        return UpstreamResult(
            status=status,
            reason=reason,
            headers=response_headers,
            body=response_body,
            error=error,
            started_at_unix=started_unix,
            started_at_mono=started_mono,
            finished_at_unix=time.time(),
            finished_at_mono=time.monotonic(),
        )

    def shutdown(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.stop()
        super().shutdown()


class AdmissionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        server = self.server
        if not isinstance(server, AdmissionServer):
            self.send_error(500, "invalid admission server")
            return
        route = self.path.split("?", 1)[0]
        service = ROUTE_TO_SERVICE.get(route)
        if service is None or service not in server.upstreams:
            self.send_error(404, "unknown algorithm route")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if length < 0:
            self.send_error(400, "invalid request body length")
            return
        received_at_unix = time.time()
        received_at_mono = time.monotonic()
        body = self.rfile.read(length)
        if len(body) != length:
            self.send_error(400, "truncated request body")
            return
        video_job_id = self.headers.get("X-Ego-Video-Job-Id")
        video_item_id = self.headers.get("X-Ego-Video-Item-Id")
        if route == "/droid.create_session":
            if not video_job_id:
                self.send_error(409, "DROID create_session requires X-Ego-Video-Job-Id")
                return
            try:
                server.acquire_droid_session_lock(video_job_id)
            except Exception as exc:
                self.send_error(409, str(exc))
                return
        elif route in {"/droid.push_frame", "/droid.finalize"}:
            if not video_job_id:
                self.send_error(409, f"DROID {route.rsplit('/', 1)[-1]} requires X-Ego-Video-Job-Id")
                return
            if not server.droid_session_is_active(video_job_id):
                self.send_error(409, f"DROID session is not active for job {video_job_id!r}")
                return

        request = PendingRequest(
            route=route,
            service=service,
            body=body,
            headers=tuple(self.headers.items()),
            video_job_id=video_job_id,
            video_item_id=video_item_id,
            received_at_unix=received_at_unix,
            received_at_mono=received_at_mono,
            queued_at_mono=time.monotonic(),
            logical_id=uuid.uuid4().hex,
        )
        server.enqueue(request)
        while True:
            request.wait_for_dispatch()
            result = server.forward(request)
            request.attempt += 1
            batch_id = request.batch_id
            retrying = (result.status == 429 or result.status >= 500 or result.error is not None) and server.retry_429(request, _retry_after_seconds(result.headers))
            server.complete_batch_member(route, batch_id)
            server.record(
                {
                    "event": "algorithm_request_forwarded",
                    "route": route,
                    "limit_name": ROUTE_TO_LIMIT_NAME[route],
                    "configured_limit": None,
                    "batch_cap": server.batch_caps.get(route, 1),
                    "batch_id": request.batch_id,
                    "batch_size": request.batch_size,
                    "logical_request_id": request.logical_id,
                    "attempt": request.attempt,
                    "retry_count": request.retry_count,
                    "video_job_id": video_job_id,
                    "video_item_id": video_item_id,
                    "received_at_unix": float(received_at_unix),
                    "received_at_iso": utc_iso_from_unix(received_at_unix),
                    "upstream_started_at_unix": float(result.started_at_unix),
                    "upstream_started_at_iso": utc_iso_from_unix(result.started_at_unix),
                    "upstream_finished_at_unix": float(result.finished_at_unix),
                    "upstream_finished_at_iso": utc_iso_from_unix(result.finished_at_unix),
                    "finished_at_unix": float(time.time()),
                    "finished_at_iso": utc_iso_from_unix(time.time()),
                    "wait_s": float(max(0.0, result.started_at_mono - received_at_mono)),
                    "upstream_wall_s": float(result.finished_at_mono - result.started_at_mono),
                    "total_wall_s": float(time.monotonic() - received_at_mono),
                    "status": int(result.status),
                    "error": result.error,
                    "request_bytes": int(len(body)),
                    "response_bytes": int(len(result.body)),
                    "terminal": not retrying,
                }
            )
            if retrying:
                continue
            request.result = result
            request.final = True
            if result.status != 429:
                # A completed non-backpressure request is evidence that some
                # service capacity may have freed. It wakes one retry queue.
                server.wake_retry(route)
            if video_job_id and route == "/droid.finalize":
                server.release_droid_session_lock(video_job_id)
            if video_job_id and route == "/droid.create_session" and result.status >= 300:
                server.release_droid_session_lock(video_job_id)
            break

        assert request.result is not None
        result = request.result
        # Mark the final attempt separately without fabricating service-compute time.
        server.record({
            "event": "algorithm_request_terminal",
            "route": route,
            "logical_request_id": request.logical_id,
            "attempt": request.attempt,
            "retry_count": request.retry_count,
            "batch_id": request.batch_id,
            "batch_size": request.batch_size,
            "status": int(result.status),
            "finished_at": float(time.time()),
        })
        response_headers = list(result.headers)
        if result.status == 429:
            response_headers.append(("X-Ego-Admission-Retry-Complete", "1"))
        self.send_response(result.status, result.reason)
        for key, value in response_headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(result.body)))
        self.end_headers()
        self.wfile.write(result.body)


def _retry_after_seconds(headers: Iterable[tuple[str, str]]) -> float | None:
    for key, value in headers:
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value.strip()))
            except ValueError:
                return None
    return None


@contextmanager
def running_proxy(
    *,
    host: str,
    port: int,
    profile: Path,
    multiplier: int,
    events_path: Path,
    lock_root: Path | None = None,
    upstream_overrides: dict[str, str] | None = None,
) -> Iterator[str]:
    # ``multiplier`` remains accepted so old manager/job invocations do not
    # change their CLI contract.  It has no effect on client scheduling.
    if multiplier <= 0:
        raise ValueError("algorithm inflight multiplier must be positive")
    server = AdmissionServer(
        (host, port),
        upstreams=load_upstreams(profile, upstream_overrides),
        events_path=events_path,
        lock_root=lock_root,
    )
    thread = threading.Thread(target=server.serve_forever, name="annotation-admission-proxy", daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    try:
        yield f"http://{bound_host}:{bound_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


__all__ = [
    "AdmissionHandler", "AdmissionServer", "BatchScheduler", "CLIENT_BATCH_CAPS", "CLIENT_BATCH_WAIT_S",
    "DROID_RESIDENT_SESSIONS_PER_REPLICA", "DROID_REPLICA_COUNT", "DROID_SESSION_NODE_CAPACITY",
    "RAY_SERVE_ONGOING_LIMITS", "ROUTE_TO_LIMIT_NAME", "ROUTE_TO_SERVICE",
    "admission_limits", "load_upstreams", "route_uses_cross_process_slot", "running_proxy",
]
