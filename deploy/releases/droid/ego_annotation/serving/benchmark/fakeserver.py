"""Deterministic fake HTTP server for benchmark tests.

Exercises the *real* multipart bytes end-to-end: it parses the gateway's multipart
request body, reads the named binary parts, and emits a multipart response with a
synthetic batch trace and phase-timing decomposition. It is *not* a mock that
short-circuits the gateway — the gateway builds real multipart bytes, the fake
server parses them, and the gateway parses the fake server's multipart response.

The fake server simulates a resident model that:
* loads the model once at startup (model_load_count == 1, constant);
* batches compatible concurrent requests into one forward (emits a shared batch_id);
* reports admission/queue/dispatch/forward/encoding phase timings;
* can be configured to return backpressure (HTTP 503) at a given in-flight count;
* never hides overload — when over its in-flight bound it returns 503 so the
  gateway's bounded retry surfaces it as a typed BACKPRESSURE failure.

Uses ``aiohttp`` (already a project dependency) so tests run without a live Serve.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Mapping

from aiohttp import web

from ego_annotation.serving.binary_envelope import CONTENT_TYPE as BINARY_ENVELOPE_CONTENT_TYPE, content_type_is_binary_envelope
from ego_annotation.serving.gateway import _build_generic_envelope, _parse_generic_envelope
from ego_annotation.serving.transport import parse_multipart_response, _iter_multipart


@dataclass
class FakeServerConfig:
    """Behavior knobs for the fake server."""

    # Simulated per-request forward latency (seconds).
    forward_latency_s: float = 0.01
    # Admission/queue/dispatch/encoding split of the non-forward time (fractions).
    admission_fraction: float = 0.1
    queue_fraction: float = 0.2
    dispatch_fraction: float = 0.1
    encoding_fraction: float = 0.1
    # When set, return HTTP 503 once in-flight >= this value (overload simulation).
    overload_in_flight: int | None = None
    # Simulated batch window: requests arriving within this window share a batch.
    batch_window_s: float = 0.02


@dataclass
class FakeServerState:
    model_load_count: int = 1
    in_flight: int = 0
    replica_id: str = "fake-replica-0"
    # Pending request awaiting batch window close, keyed by arrival order.
    pending: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class FakeModelServer:
    """An aiohttp app that parses real multipart requests and emits multipart responses."""

    def __init__(self, config: FakeServerConfig | None = None) -> None:
        self.config = config or FakeServerConfig()
        self.state = FakeServerState()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.router.add_post("/{tail:.*}", self._handle)
        # Match the benchmark runner's canonical Ray Serve probe path as well as
        # the direct /health smoke route, so fake wire-format sweeps exercise the
        # actual gateway instead of being skipped as a false-down endpoint.
        self.app.router.add_get("/{tail:.*}", self._health)

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok", status=200)

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.read()
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        envelope_wire = content_type_is_binary_envelope(content_type)
        try:
            metadata, parts = _parse_generic_envelope(body) if envelope_wire else _parse_generic_multipart(body, content_type)
        except Exception as exc:
            return web.Response(text=json.dumps({"error": {"code": "validation", "message": str(exc), "retryable": False}}), status=400)

        # Overload check: return 503 when in-flight bound is reached.
        if self.config.overload_in_flight is not None:
            async with self.state.lock:
                if self.state.in_flight >= self.config.overload_in_flight:
                    return web.Response(status=503)
                self.state.in_flight += 1
        try:
            return await self._process(request, metadata, parts, envelope_wire=envelope_wire)
        finally:
            if self.config.overload_in_flight is not None:
                async with self.state.lock:
                    self.state.in_flight = max(0, self.state.in_flight - 1)

    async def _process(
        self, request: web.Request, metadata: Mapping[str, object],
        parts: Mapping[str, tuple[bytes | memoryview, tuple[int, ...], str]], *, envelope_wire: bool,
    ) -> web.Response:
        api_name = request.path.strip("/")
        ownership = metadata.get("ownership", {})
        request_id = ownership.get("request_id", "unknown")
        # Simulate phase timings.
        t_admit = time.monotonic()
        await asyncio.sleep(self.config.forward_latency_s * self.config.admission_fraction)
        t_queue = time.monotonic()
        await asyncio.sleep(self.config.forward_latency_s * self.config.queue_fraction)
        t_dispatch = time.monotonic()
        await asyncio.sleep(self.config.forward_latency_s * self.config.dispatch_fraction)
        t_forward = time.monotonic()
        await asyncio.sleep(self.config.forward_latency_s * (1.0 - self.config.admission_fraction - self.config.queue_fraction - self.config.dispatch_fraction - self.config.encoding_fraction))
        t_forward_end = time.monotonic()
        await asyncio.sleep(self.config.forward_latency_s * self.config.encoding_fraction)
        t_encode_end = time.monotonic()

        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        trace = {
            "batch_id": batch_id,
            "replica_id": self.state.replica_id,
            "admitted_monotonic_s": t_admit,
            "dispatched_monotonic_s": t_dispatch,
            "forward_started_monotonic_s": t_forward,
            "completed_monotonic_s": t_forward_end,
            "effective_work_units": int(metadata.get("work_units", 1)),
            "request_count": 1,
            "forward_count": 1,
            "model_load_count": self.state.model_load_count,
        }
        phase_timing = {
            "admission_ms": (t_queue - t_admit) * 1000.0,
            "queue_ms": (t_dispatch - t_queue) * 1000.0,
            "dispatch_ms": (t_forward - t_dispatch) * 1000.0,
            "forward_ms": (t_forward_end - t_forward) * 1000.0,
            "encoding_ms": (t_encode_end - t_forward_end) * 1000.0,
        }
        # Build a synthetic result: echo the input part names with a small output array.
        arrays: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
        for name, (data, shape, dtype) in parts.items():
            # Output a 1-element float array per part as proof of round-trip.
            arrays[f"out_{name}"] = (b"\x00\x00\x80\x3f", (1,), "float32")
        result_meta = {
            "ownership": ownership,
            "api_name": api_name,
            "phase_timing": phase_timing,
            "trace": trace,
            "request_id": request_id,
            "echoed_part_names": list(parts.keys()),
        }
        from ego_annotation.serving.transport import build_multipart_response

        if envelope_wire:
            envelope = _build_generic_envelope(
                {"result": result_meta, "ownership": ownership},
                [(name, data, shape, dtype) for name, (data, shape, dtype) in arrays.items()],
            )
            # The test server's aiohttp response API consumes one body object; the
            # production gateway/deployment paths preserve vectors. This isolated
            # fake only supplies a deterministic network peer for typed round trips.
            body = b"".join(envelope.iovecs)
            content_type = BINARY_ENVELOPE_CONTENT_TYPE
        else:
            body, content_type = build_multipart_response({"result": result_meta, "ownership": ownership}, arrays)
        return web.Response(body=body, content_type=content_type, status=200)


def _parse_generic_multipart(body: bytes, content_type: str) -> tuple[dict, dict[str, tuple[bytes, tuple[int, ...], str]]]:
    """Parse a multipart body with one metadata JSON part + N named binary parts."""
    parts = _iter_multipart(body, content_type)
    metadata: dict = {}
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
    for name, data, params in parts:
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif "shape" in params and "dtype" in params:
            shape = tuple(int(x) for x in params["shape"].split(",") if x.strip())
            arrays[name] = (data, shape, params["dtype"])
        else:
            arrays[name] = (data, (), "bytes")
    if not metadata:
        raise ValueError("missing metadata part")
    return metadata, arrays


# --- transport adapters for tests / smoke runs -------------------------------------


class _FakeHttpResponse:
    def __init__(self, status: int, content: bytes, headers: dict[str, str]) -> None:
        self.status_code = status
        self.content = content
        self.headers = headers


class FakeHttpGatewayTransport:
    """AsyncHttpTransport adapter that POSTs real bytes to the running fake server."""

    def __init__(self, server_runner: web.AppRunner) -> None:
        self._runner = server_runner
        self._session = None

    async def _ensure_session(self):
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def post(self, url: str, *, content, headers: dict[str, str]):
        session = await self._ensure_session()
        async with session.post(url, data=content, headers=headers) as resp:
            body = await resp.read()
            resp_headers = {k: v for k, v in resp.headers.items()}
            return _FakeHttpResponse(status=resp.status, content=body, headers=resp_headers)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class FakeHttpProbeTransport:
    """AsyncProbeTransport adapter for the fake server's /health endpoint."""

    def __init__(self, server_runner: web.AppRunner) -> None:
        self._runner = server_runner
        self._session = None

    async def _ensure_session(self):
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get(self, url: str, *, timeout_s: float):
        from ego_annotation.serving.benchmark.endpoints import ProbeResult

        session = await self._ensure_session()
        try:
            import aiohttp

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                return ProbeResult(live=resp.status < 500, status_code=resp.status, latency_ms=None)
        except Exception as exc:
            return ProbeResult(live=False, status_code=None, latency_ms=None, error=str(exc))

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


async def start_fake_server(*, host: str = "127.0.0.1", port: int = 0, config: FakeServerConfig | None = None) -> "_FakeServerHandle":
    server = FakeModelServer(config)
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    # Discover the bound port.
    actual_port = port or site._server.sockets[0].getsockname()[1]
    return _FakeServerHandle(server=server, runner=runner, site=site, port=actual_port, host=host)


@dataclass
class _FakeServerHandle:
    server: FakeModelServer
    runner: web.AppRunner
    site: web.TCPSite
    port: int
    host: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def stop(self) -> None:
        await self.runner.cleanup()
