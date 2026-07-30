"""Live (httpx) transport adapters for benchmark endpoint probing.

Imported lazily by ``scripts.ray_serve_benchmark`` only for live runs so the test
environment (which uses the fake server) does not require httpx to be importable at
module load time. httpx is already a project dependency.
"""
from __future__ import annotations

import httpx

from ego_annotation.serving.benchmark.endpoints import ProbeResult


class HttpxProbeTransport:
    """Probe a Serve endpoint's health path once with httpx."""

    async def get(self, url: str, *, timeout_s: float) -> ProbeResult:
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.get(url)
                return ProbeResult(live=resp.status_code < 500, status_code=resp.status_code, latency_ms=None)
        except Exception as exc:
            return ProbeResult(live=False, status_code=None, latency_ms=None, error=str(exc))
