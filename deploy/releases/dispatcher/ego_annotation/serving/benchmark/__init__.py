"""Open-loop offered-load benchmark harness for the typed model-service gateway.

Public surface:
* ``manifest``     — distinct real payload manifests with content hashes.
* ``metrics``      — per-item records and four-rate aggregate summaries.
* ``generator``    — open-loop offered-load generator (arrivals independent of
  completions).
* ``endpoints``    — probe-once endpoint runner and run manifest.
* ``artifacts``    — raw JSONL/CSV writers.
* ``plotting``     — throughput-latency and batch-distribution plots.
* ``runner``       — orchestrates a full benchmark run (probe, sweep, write).
* ``fakeserver``   — deterministic fake HTTP server for tests (real multipart bytes).
"""
from ego_annotation.serving.benchmark import (  # noqa: F401
    artifacts,
    endpoints,
    fakeserver,
    generator,
    manifest,
    metrics,
    plotting,
    runner,
)

__all__ = [
    "artifacts", "endpoints", "fakeserver", "generator", "manifest", "metrics",
    "plotting", "runner",
]
