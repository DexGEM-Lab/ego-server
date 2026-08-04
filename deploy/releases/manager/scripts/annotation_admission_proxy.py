#!/usr/bin/env python3
"""Internal image-weighted admission queue for model-service routes.

Each incoming body is streamed to a private spool file, assigned a conservative
image/frame weight, and admitted to a route-local queue holding 8000
image-equivalents. Bounded internal forwarding workers open upstream connections;
external callers are not assigned concurrency slots or rejected because workers are
busy. A full internal queue applies backpressure while retaining the spooled request.

Ray Serve (or vLLM for Cosmos) remains the model execution owner. Retryable
service backpressure keeps the same request in the weighted queue until one terminal
response, without duplicating its queue weight or request body.
"""
from __future__ import annotations

import heapq
import http.client
import json
import random
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

# Each atomic DROID inference owns one dispatcher session for the lifetime of
# its HTTP request.  The deployed service has six replicas with max_sessions=1.
DROID_REPLICA_COUNT = 6
DROID_SESSIONS_PER_REPLICA = 1
DROID_SESSION_NODE_CAPACITY = DROID_REPLICA_COUNT * DROID_SESSIONS_PER_REPLICA

# Internal queue budget. A request is assigned a conservative fixed work weight
# before it is admitted. Handlers wait here when the route queue is full; this is
# service-side buffering, not a caller/external-concurrency quota.
IMAGE_QUEUE_BUDGET = 8000
ROUTE_IMAGE_UNITS = {
    "/unidepth.infer": 1,
    "/hands.detect": 1,
    "/wilor.reconstruct": 1,
    "/droid.infer": 256,
    "/hawor.infer_tracks": 16,
    "/hawor_infiller.fill": 120,
    "/cosmos3.reason": 8,
}
# These are service-facing forwarding workers. They avoid opening thousands of
# simultaneous sockets while the internal weighted queue retains the work.
ROUTE_FORWARD_LIMITS = {
    "/unidepth.infer": 16,
    "/hands.detect": 32,
    "/wilor.reconstruct": 32,
    "/droid.infer": DROID_SESSION_NODE_CAPACITY,
    "/hawor.infer_tracks": 16,
    "/hawor_infiller.fill": 8,
    "/cosmos3.reason": 16,
}

# Retained for compatibility with existing reports and callers. Internal
# forwarding workers are defined separately in ROUTE_FORWARD_LIMITS.
RAY_SERVE_ONGOING_LIMITS = {
    "unidepth.infer": 16,
    "hands.detect": 16,
    "wilor.reconstruct": 32,
    "droid.infer": DROID_SESSION_NODE_CAPACITY,
    "hawor.infer_tracks": 8,
    "hawor_infiller.fill": 4,
    "cosmos3.reason": 16,
}

# Native model batch caps. DROID is one full-video single-push request; Cosmos
# owns its vLLM scheduler.
CLIENT_BATCH_CAPS = {
    "/unidepth.infer": 8,
    "/hands.detect": 8,
    "/wilor.reconstruct": 16,
    "/droid.infer": 1,
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
    "/droid.infer": 0.0,
    "/hawor.infer_tracks": 0.250,
    "/hawor_infiller.fill": 0.200,
    "/cosmos3.reason": 0.0,
}

# A proxy handler waits on the same connection, so retries must terminate
# before the established client timeout.  Retry timing is intentionally short
# and capped; successful requests wake the retry scheduler immediately.
# Normal submissions go out immediately (burst). RETRY_CYCLE_S is only the
# scheduler poll interval; individual retries use exponential backoff so a
# rejected full-video DROID upload cannot be replayed in a tight loop.
RETRY_CYCLE_S = 0.020
RETRY_INITIAL_DELAY_S = 0.5
RETRY_MAX_DELAY_S = 10.0
RETRY_JITTER_FRACTION = 0.10
RETRY_AFTER_MAX_S = 5.0
# No retry attempt limit or deadline: capacity failures stay in the queue until
# the service accepts them or the pipeline is explicitly stopped.

ROUTE_TO_SERVICE = {
    "/unidepth.infer": "unidepth",
    "/hands.detect": "hands_wilor",
    "/wilor.reconstruct": "wilor",
    "/droid.infer": "droid",
    "/hawor.infer_tracks": "hawor",
    "/hawor_infiller.fill": "hawor",
    "/cosmos3.reason": "cosmos3",
}

ROUTE_TO_LIMIT_NAME = {
    "/unidepth.infer": "unidepth.infer",
    "/hands.detect": "hands.detect",
    "/wilor.reconstruct": "wilor.reconstruct",
    "/droid.infer": "droid.infer",
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
    body: bytes | None
    headers: tuple[tuple[str, str], ...]
    video_job_id: str | None
    video_item_id: str | None
    received_at_unix: float
    received_at_mono: float
    queued_at_mono: float
    logical_id: str
    body_path: Path | None = None
    body_size: int = 0
    work_units: int = 1
    retry_count: int = 0
    attempt: int = 0
    batch_id: str | None = None
    batch_size: int = 1
    dispatch_event: threading.Event = field(default_factory=threading.Event)
    dispatch_guard: threading.Lock = field(default_factory=threading.Lock)
    result: UpstreamResult | None = None
    final: bool = False
    next_retry_at_mono: float | None = None
    failure_reason: str | None = None
    reserved: bool = False

    def read_body(self) -> bytes:
        if self.body is not None:
            return self.body
        if self.body_path is None:
            raise RuntimeError("pending request has no body")
        return self.body_path.read_bytes()

    def cleanup_body(self) -> None:
        if self.body_path is not None:
            try:
                self.body_path.unlink()
            except FileNotFoundError:
                pass

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
    """Route-local weighted queue with bounded forwarding workers.

    Each admitted request occupies ``work_units`` until its terminal response.
    When the weighted queue reaches 8000 image-equivalents, the HTTP handler
    remains attached to its already-spooled body and waits for capacity; it is
    never rejected merely because the internal queue is full. Only the bounded
    forwarding workers open upstream connections.
    """

    def __init__(
        self, route: str, *, max_inflight: int, work_units: int, queue_budget: int = IMAGE_QUEUE_BUDGET,
    ):
        if max_inflight <= 0 or work_units <= 0 or queue_budget <= 0:
            raise ValueError("scheduler capacities must be positive")
        self.route = route
        self.max_inflight = max_inflight
        self.work_units = work_units
        self.queue_budget = queue_budget
        self._ready: deque[PendingRequest] = deque()
        self._retry_queue: deque[PendingRequest] = deque()
        self._sequence = 0
        self._batch_sequence = 0
        self._inflight = 0
        self._admitted_work = 0
        self._scheduler_id = uuid.uuid4().hex[:12]
        self._condition = threading.Condition()
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name=f"batch-scheduler-{route}", daemon=True)
        self._thread.start()

    def reserve(self, request: PendingRequest) -> None:
        if request.work_units != self.work_units:
            raise ValueError(f"{self.route} request weight {request.work_units} != configured {self.work_units}")
        with self._condition:
            while not self._stopping and self._admitted_work + request.work_units > self.queue_budget:
                self._condition.wait()
            if self._stopping:
                raise RuntimeError(f"{self.route} scheduler is stopping")
            request.queued_at_mono = time.monotonic()
            request.next_retry_at_mono = None
            request.reserved = True
            self._admitted_work += request.work_units
            self._condition.notify_all()

    def enqueue_reserved(self, request: PendingRequest) -> None:
        with self._condition:
            if not request.reserved:
                raise RuntimeError(f"{self.route} request was not reserved")
            self._ready.append(request)
            self._condition.notify_all()

    def cancel_reservation(self, request: PendingRequest) -> None:
        with self._condition:
            if request.reserved:
                request.reserved = False
                self._admitted_work -= request.work_units
                if self._admitted_work < 0:
                    raise RuntimeError(f"{self.route} weighted queue accounting underflow")
                self._condition.notify_all()

    def enqueue(self, request: PendingRequest) -> None:
        self.reserve(request)
        self.enqueue_reserved(request)

    def requeue_429(self, request: PendingRequest, retry_after_s: float | None) -> bool:
        now = time.monotonic()
        request.retry_count += 1
        request.arm_dispatch()
        request.next_retry_at_mono = now + self._retry_delay(request.retry_count, retry_after_s)
        with self._condition:
            self._sequence += 1
            self._retry_queue.append(request)
            self._condition.notify_all()
        return True

    def complete_batch_member(self, batch_id: str | None, *, terminal: bool) -> None:
        with self._condition:
            if self._inflight <= 0:
                raise RuntimeError(f"{self.route} completed without an admitted request")
            self._inflight -= 1
            if terminal:
                self._admitted_work -= self.work_units
                if self._admitted_work < 0:
                    raise RuntimeError(f"{self.route} weighted queue accounting underflow")
            self._condition.notify_all()

    @property
    def inflight_count(self) -> int:
        with self._condition:
            return self._inflight

    @property
    def admitted_work_units(self) -> int:
        with self._condition:
            return self._admitted_work

    def wake_retry(self) -> None:
        with self._condition:
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
        delay = min(RETRY_MAX_DELAY_S, RETRY_INITIAL_DELAY_S * (2 ** max(0, retry_count - 1)))
        jitter = random.uniform(-RETRY_JITTER_FRACTION, RETRY_JITTER_FRACTION)
        return max(0.0, delay * (1.0 + jitter))

    def _promote_due_retries(self, now: float) -> None:
        due: list[PendingRequest] = []
        remaining: deque[PendingRequest] = deque()
        for request in self._retry_queue:
            if request.next_retry_at_mono is not None and request.next_retry_at_mono <= now:
                request.queued_at_mono = now
                request.next_retry_at_mono = None
                due.append(request)
            else:
                remaining.append(request)
        self._retry_queue = remaining
        self._ready.extend(due)

    def _release_all_ready(self) -> int:
        released = 0
        while self._ready and self._inflight < self.max_inflight:
            request = self._ready.popleft()
            self._batch_sequence += 1
            request.batch_id = f"{self.route.replace('.', '_').replace('/', '_')}-{self._scheduler_id}-{self._batch_sequence:08d}"
            request.batch_size = 1
            self._inflight += 1
            request.release_dispatch()
            released += 1
        return released

    def _run(self) -> None:
        while True:
            with self._condition:
                now = time.monotonic()
                self._promote_due_retries(now)
                if self._stopping:
                    return
                released = self._release_all_ready()
                if released:
                    self._condition.wait(timeout=0.001)
                    continue
                timeout = RETRY_CYCLE_S if self._retry_queue else None
                self._condition.wait(timeout=timeout)


class AdmissionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Secondary listener protection. The weighted internal queues, not this
    # socket backlog, define service capacity.
    request_queue_size = 8192
    # This bounds Python handler threads and therefore resident request state.
    # It is an internal resource boundary; pending clients remain in the kernel
    # listener backlog rather than consuming one thread or spool file each.
    max_handler_threads = 256

    def __init__(
        self,
        address: tuple[str, int],
        *,
        upstreams: dict[str, str],
        events_path: Path,
        batch_caps: dict[str, int] | None = None,
        batch_waits: dict[str, float] | None = None,
        # Compatibility parameters accepted by older launchers; per-route
        # capacity is defined by the deployed DROID topology below.
        limits: dict[str, int] | None = None,
        lock_root: Path | None = None,
        route_inflight_limits: dict[str, int] | None = None,
        route_image_units: dict[str, int] | None = None,
        route_queue_image_budgets: dict[str, int] | None = None,
    ):
        super().__init__(address, AdmissionHandler)
        self._handler_slots = threading.BoundedSemaphore(self.max_handler_threads)
        self.upstreams = upstreams
        self.events_path = events_path
        self.lock_root = lock_root
        self.batch_caps = dict(CLIENT_BATCH_CAPS if batch_caps is None else batch_caps)
        self.batch_waits = dict(CLIENT_BATCH_WAIT_S if batch_waits is None else batch_waits)
        self.route_inflight_limits = dict(ROUTE_FORWARD_LIMITS)
        if route_inflight_limits is not None:
            self.route_inflight_limits.update(route_inflight_limits)
        self.route_image_units = dict(ROUTE_IMAGE_UNITS)
        if route_image_units is not None:
            self.route_image_units.update(route_image_units)
        self.route_queue_image_budgets = {
            route: IMAGE_QUEUE_BUDGET for route in ROUTE_TO_SERVICE
        }
        if route_queue_image_budgets is not None:
            self.route_queue_image_budgets.update(route_queue_image_budgets)
        self.spool_root = events_path.parent / ".request-bodies"
        self.spool_root.mkdir(parents=True, exist_ok=True)
        self.schedulers = {
            route: BatchScheduler(
                route,
                max_inflight=self.route_inflight_limits[route],
                work_units=self.route_image_units[route],
                queue_budget=self.route_queue_image_budgets[route],
            )
            for route in ROUTE_TO_SERVICE
        }
        self.events_lock = threading.Lock()

    def spool_body(self, reader: Any, length: int) -> tuple[Path, int]:
        path = self.spool_root / f"{uuid.uuid4().hex}.body"
        received = 0
        try:
            with path.open("xb") as handle:
                while received < length:
                    chunk = reader.read(min(1024 * 1024, length - received))
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if received != length:
            path.unlink(missing_ok=True)
            raise ValueError("truncated request body")
        return path, received

    def reserve(self, request: PendingRequest) -> None:
        self.schedulers[request.route].reserve(request)

    def enqueue_reserved(self, request: PendingRequest) -> None:
        self.schedulers[request.route].enqueue_reserved(request)

    def cancel_reservation(self, request: PendingRequest) -> None:
        self.schedulers[request.route].cancel_reservation(request)

    def enqueue(self, request: PendingRequest) -> None:
        self.schedulers[request.route].enqueue(request)

    def process_request(self, request: Any, client_address: Any) -> None:
        self._handler_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()

    def retry_429(self, request: PendingRequest, retry_after_s: float | None) -> bool:
        return self.schedulers[request.route].requeue_429(request, retry_after_s)

    @staticmethod
    def retry_failure_reason(request: PendingRequest, result: UpstreamResult) -> str | None:
        """Return a terminal diagnostic when retrying would be unsafe or futile.

        A 429 is an explicit admission rejection and can be retried while the
        same weighted queue entry remains owned. A transport failure or 5xx is
        ambiguous: the upstream may have accepted the request before the
        response was lost, so replaying it automatically could duplicate work.
        Those attempts terminate and require operator reconciliation.
        """
        if result.status == 429:
            return None
        if result.status == 413:
            return "upstream rejected the request as payload_too_large (HTTP 413)"
        if result.error is not None:
            return "upstream transport outcome is ambiguous; reconcile before retrying"
        if result.status >= 500:
            return f"upstream returned ambiguous HTTP {result.status}; reconcile before retrying"
        return None

    @staticmethod
    def annotate_retry_failure(request: PendingRequest, result: UpstreamResult) -> UpstreamResult:
        """Surface an unsafe-to-retry outcome as an explicit gateway error."""
        if request.failure_reason is None:
            return result
        body = json.dumps({
            "error": "admission_proxy_retry_not_safe",
            "reason": request.failure_reason,
            "last_upstream_error": result.error,
            "last_upstream_status": result.status,
            "attempts": request.attempt,
        }, sort_keys=True).encode("utf-8")
        return UpstreamResult(
            status=502,
            reason="Bad Gateway",
            headers=(("Content-Type", "application/json"),),
            body=body,
            error=(f"{result.error}; " if result.error else "") + request.failure_reason,
            started_at_unix=result.started_at_unix,
            started_at_mono=result.started_at_mono,
            finished_at_unix=result.finished_at_unix,
            finished_at_mono=result.finished_at_mono,
        )

    def complete_batch_member(self, route: str, batch_id: str | None, *, terminal: bool) -> None:
        self.schedulers[route].complete_batch_member(batch_id, terminal=terminal)

    def queue_metrics(self, route: str) -> dict[str, int]:
        scheduler = self.schedulers[route]
        return {
            "queue_image_budget": IMAGE_QUEUE_BUDGET,
            "request_image_units": ROUTE_IMAGE_UNITS[route],
            "admitted_image_units": scheduler.admitted_work_units,
            "forward_inflight": scheduler.inflight_count,
        }

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
            body = request.read_body()
            headers["Content-Length"] = str(len(body))
            target = f"{upstream.path.rstrip('/')}{request.route}" if upstream.path else request.route
            connection.request("POST", target, body=body, headers=headers)
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
        video_job_id = self.headers.get("X-Ego-Video-Job-Id")
        video_item_id = self.headers.get("X-Ego-Video-Item-Id")
        request = PendingRequest(
            route=route,
            service=service,
            body=None,
            body_path=None,
            body_size=0,
            work_units=ROUTE_IMAGE_UNITS[route],
            headers=tuple(self.headers.items()),
            video_job_id=video_job_id,
            video_item_id=video_item_id,
            received_at_unix=received_at_unix,
            received_at_mono=received_at_mono,
            queued_at_mono=time.monotonic(),
            logical_id=uuid.uuid4().hex,
        )
        # Reserve image-weighted capacity before reading the body. When the
        # internal budget is full, the client remains at the HTTP boundary and
        # no spool file or full body is allocated for the blocked request.
        server.reserve(request)
        try:
            body_path, body_size = server.spool_body(self.rfile, length)
        except ValueError:
            server.cancel_reservation(request)
            self.send_error(400, "truncated request body")
            return
        except OSError:
            server.cancel_reservation(request)
            self.send_error(507, "admission spool unavailable")
            return
        request.body_path = body_path
        request.body_size = body_size
        server.enqueue_reserved(request)
        while True:
            request.wait_for_dispatch()
            result = server.forward(request)
            request.attempt += 1
            batch_id = request.batch_id
            retryable = result.status == 429 or result.status >= 500 or result.error is not None
            failure_reason = server.retry_failure_reason(request, result)
            retrying = retryable and failure_reason is None and server.retry_429(request, _retry_after_seconds(result.headers))
            if failure_reason is not None:
                request.failure_reason = failure_reason
                result = server.annotate_retry_failure(request, result)
            server.complete_batch_member(route, batch_id, terminal=not retrying)
            server.record(
                {
                    "event": "algorithm_request_forwarded",
                    "route": route,
                    "limit_name": ROUTE_TO_LIMIT_NAME[route],
                    "configured_limit": server.route_inflight_limits.get(route),
                    "queue_image_budget": IMAGE_QUEUE_BUDGET,
                    "request_image_units": ROUTE_IMAGE_UNITS[route],
                    "admitted_image_units": server.schedulers[route].admitted_work_units,
                    "forward_inflight": server.schedulers[route].inflight_count,
                    "batch_cap": server.batch_caps.get(route, 1),
                    "batch_id": request.batch_id,
                    "batch_size": request.batch_size,
                    "logical_request_id": request.logical_id,
                    "attempt": request.attempt,
                    "retry_count": request.retry_count,
                    "failure_reason": request.failure_reason,
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
                    "request_bytes": int(request.body_size),
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
            "failure_reason": request.failure_reason,
            "batch_id": request.batch_id,
            "batch_size": request.batch_size,
            "status": int(result.status),
            "finished_at": float(time.time()),
        })
        response_headers = list(result.headers)
        if result.status == 429:
            response_headers.append(("X-Ego-Admission-Retry-Complete", "1"))
        if request.failure_reason is not None:
            response_headers.append(("X-Ego-Admission-Retry-Exhausted", request.failure_reason))
        self.send_response(result.status, result.reason)
        for key, value in response_headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(result.body)))
        self.end_headers()
        try:
            self.wfile.write(result.body)
        finally:
            request.cleanup_body()


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
    "DROID_REPLICA_COUNT", "DROID_SESSIONS_PER_REPLICA", "DROID_SESSION_NODE_CAPACITY",
    "IMAGE_QUEUE_BUDGET", "ROUTE_IMAGE_UNITS", "ROUTE_FORWARD_LIMITS",
    "RAY_SERVE_ONGOING_LIMITS", "ROUTE_TO_LIMIT_NAME", "ROUTE_TO_SERVICE",
    "admission_limits", "load_upstreams", "route_uses_cross_process_slot", "running_proxy",
]
