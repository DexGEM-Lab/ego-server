"""Endpoint runner: check live lane endpoints once, preserve a run manifest.

The benchmark must not wait for model lanes to become available. The endpoint runner
performs a **single** liveness probe against each configured Serve endpoint (one
GET/HEAD to a health path, or a minimal OPTIONS), records which endpoints were live
at probe time, and writes a run manifest. The harness can be invoked again later
(after more lanes come up) and will re-probe once; endpoints that were down are
recorded as ``down`` and skipped for that run, not polled.

This keeps Ray internal: the runner only touches the public Serve HTTP endpoints.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

from ego_annotation.serving.router import ModelApiName, ModelServiceRouter, ServeEndpoint


from typing import Protocol


class AsyncProbeTransport(Protocol):
    async def get(self, url: str, *, timeout_s: float) -> "ProbeResult": ...


@dataclass(frozen=True)
class ProbeResult:
    live: bool
    status_code: int | None
    latency_ms: float | None
    error: str | None = None


@dataclass(frozen=True)
class EndpointObservation:
    api_name: ModelApiName
    url: str
    live: bool
    status_code: int | None
    latency_ms: float | None
    error: str | None

    def to_manifest(self) -> dict[str, object]:
        return {
            "api_name": self.api_name.value,
            "url": self.url,
            "live": self.live,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class EndpointProbeConfig:
    """How to probe one endpoint. Defaults to Ray's exact proxy health route."""

    health_path: str = "/-/healthz"
    timeout_s: float = 2.0


@dataclass
class RunManifest:
    """Preserved run manifest so the harness can be re-invoked as lanes come up."""

    run_id: str
    created_at_s: float
    probe_config: EndpointProbeConfig
    observations: tuple[EndpointObservation, ...]
    live_apis: tuple[str, ...]
    down_apis: tuple[str, ...]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at_s": self.created_at_s,
            "probe_config": {
                "health_path": self.probe_config.health_path,
                "timeout_s": self.probe_config.timeout_s,
            },
            "observations": [o.to_manifest() for o in self.observations],
            "live_apis": list(self.live_apis),
            "down_apis": list(self.down_apis),
            "artifact_paths": dict(self.artifact_paths),
            "notes": list(self.notes),
        }


async def _probe_one(
    router: ModelServiceRouter,
    endpoint: ServeEndpoint,
    transport: AsyncProbeTransport,
    config: EndpointProbeConfig,
    clock: Callable[[], float],
) -> EndpointObservation:
    # Honor per-API URL overrides (e.g. fake server or a lane discovered at run
    # time) so the probe targets the same origin the API is routed to.
    base = router.base_url_for(endpoint.api_name)
    url = f"{base}{config.health_path}"
    t0 = clock()
    try:
        result = await transport.get(url, timeout_s=config.timeout_s)
    except Exception as exc:
        return EndpointObservation(
            api_name=endpoint.api_name, url=url, live=False, status_code=None,
            latency_ms=(clock() - t0) * 1000.0, error=str(exc),
        )
    latency_ms = (clock() - t0) * 1000.0
    # A 404/redirect/error is not liveness. Ray's /-/healthz is an exact 2xx
    # contract; typed application probes may use a method-specific validator.
    live = result.live and result.status_code is not None and 200 <= result.status_code < 300
    return EndpointObservation(
        api_name=endpoint.api_name, url=url, live=live,
        status_code=result.status_code, latency_ms=latency_ms, error=result.error,
    )


async def probe_endpoints_once(
    router: ModelServiceRouter,
    transport: AsyncProbeTransport,
    *,
    apis: Sequence[ModelApiName | str] | None = None,
    config: EndpointProbeConfig | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[EndpointObservation, ...]:
    """Probe each endpoint exactly once. Never poll, never wait for lanes.

    ``apis`` selects which APIs to probe (default: all configured). Returns one
    observation per probed API. A down endpoint is recorded and skipped; it is not
    retried here. Re-running the harness re-probes once.
    """
    cfg = config or EndpointProbeConfig()
    clk = clock or time.monotonic
    target_apis = [ModelApiName(a) if isinstance(a, str) else a for a in (apis or router.all_apis())]
    endpoints = [router.endpoint_for(a) for a in target_apis]
    # Probe all in parallel, each exactly once.
    observations = await asyncio.gather(*(_probe_one(router, ep, transport, cfg, clk) for ep in endpoints))
    return tuple(observations)


def build_run_manifest(
    *,
    run_id: str,
    observations: Sequence[EndpointObservation],
    probe_config: EndpointProbeConfig,
    artifact_paths: Mapping[str, str] | None = None,
    notes: Sequence[str] | None = None,
    clock: Callable[[], float] | None = None,
) -> RunManifest:
    clk = clock or time.monotonic
    live = tuple(o.api_name.value for o in observations if o.live)
    down = tuple(o.api_name.value for o in observations if not o.live)
    return RunManifest(
        run_id=run_id,
        created_at_s=clk(),
        probe_config=probe_config,
        observations=tuple(observations),
        live_apis=live,
        down_apis=down,
        artifact_paths=dict(artifact_paths or {}),
        notes=tuple(notes or ()),
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
