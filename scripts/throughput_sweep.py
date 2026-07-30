#!/usr/bin/env python3
"""Concurrent multi-service throughput sweep with saturation-knee detection.

At every configured level, every selected resident service is driven at the same
moment with its own offered concurrency.  Stateless services repeat a verified
model-native request; DROID workers each own a complete create -> ordered pushes ->
finalize lifecycle.  The client never imports Ray and never changes a deployment.

Live use (only when explicitly authorized):

    python -m scripts.throughput_sweep --apis unidepth,hands,droid --duration-s 60

No-GPU validation:

    python -m scripts.throughput_sweep --fake-server --levels 1,2 --requests-per-worker 2
    python -m scripts.throughput_sweep --dry-run

Artifacts are written below /home/zjh/ray_serve_benchmarks/throughput_sweep_<ts> by
default.  This command is a load generator in live mode; do not use it against a
service until a load window has been authorized.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit
import uuid

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx

from ego_annotation.serving.benchmark.metrics import _percentile as benchmark_percentile
from ego_annotation.serving.benchmark.plotting import plot_throughput_latency
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import build_multipart_request_fields
from scripts.parallel_stress_test import (
    ARTIFACT_BASE,
    DEFAULT_DROID_MANIFEST,
    PreparedOperation,
    droid_camera_contract,
    env_path,
    normalize_apis,
    ownership,
    post_raw,
    prepare_operations,
    read_json,
    resolve_existing,
)


SERVICE_UNITS = {
    "unidepth": "img/s",
    "hands": "img/s",
    "wilor": "crop/s",
    "droid": "frame/s",
    "hawor": "chunk/s",
    "infiller": "window/s",
    "cosmos": "request/s",
}
DROID_MAX_SESSIONS = 48
DEFAULT_DROID_FRAMES_PER_SESSION = 16
# GPU ownership is deliberately expressed here, rather than inferred from HTTP
# ports.  Dispatchers route UniDepth across GPU0/GPU5 and DROID across GPU2/GPU7;
# WiLoR has its own GPU4 resident actor rather than sharing Hands' GPU1.
SERVICE_EXPECTED_GPUS: dict[str, frozenset[int]] = {
    "unidepth": frozenset({0, 5}),
    "hands": frozenset({1}),
    "wilor": frozenset({4}),
    "droid": frozenset({2, 7}),
    "hawor": frozenset({3}),
    "infiller": frozenset({3}),
    "cosmos": frozenset({6}),
}
EXTERNAL_GPU_MEMORY_DELTA_MIB = 256.0
SIGNIFICANT_BACKPRESSURE_RATE = 0.05


class AsyncHttpClient(Protocol):
    async def post(self, url: str, *, content: bytes, headers: Mapping[str, str]) -> Any: ...
    async def get(self, url: str) -> Any: ...


@dataclass(frozen=True)
class CycleResult:
    rows: tuple[Mapping[str, Any], ...]
    work_units: int

    @property
    def success(self) -> bool:
        return bool(self.rows) and all(bool(row.get("success")) for row in self.rows)


@dataclass
class LevelResult:
    service: str
    offered_concurrency: int
    duration_s: float
    completed_requests: int
    attempted_requests: int
    failed_requests: int
    work_units: int
    latencies_ms: list[float] = field(default_factory=list)
    batch_sizes: list[float] = field(default_factory=list)
    status_429: int = 0
    status_503: int = 0
    cycles_completed: int = 0
    cycles_failed: int = 0
    requested_concurrency: int | None = None
    execution_traces: list[Mapping[str, Any]] = field(default_factory=list)
    queue_depths: list[float] = field(default_factory=list)
    batch_snapshot_before: Mapping[str, Any] | None = None
    batch_snapshot_after: Mapping[str, Any] | None = None
    droid_leases_after_level: Mapping[str, Any] | None = None
    memory_before_level: Mapping[str, Any] | None = None
    memory_after_level: Mapping[str, Any] | None = None

    @property
    def req_s(self) -> float:
        return self.completed_requests / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def img_s(self) -> float:
        return self.work_units / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.failed_requests / self.attempted_requests if self.attempted_requests else 0.0

    @property
    def latency_p50_ms(self) -> float | None:
        return percentile(self.latencies_ms, 0.50)

    @property
    def latency_p95_ms(self) -> float | None:
        return percentile(self.latencies_ms, 0.95)

    @property
    def latency_p99_ms(self) -> float | None:
        return percentile(self.latencies_ms, 0.99)

    @property
    def batch_size_mean(self) -> float | None:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else None

    @property
    def queue_depth_mean(self) -> float | None:
        return sum(self.queue_depths) / len(self.queue_depths) if self.queue_depths else None

    @property
    def queue_depth_max(self) -> float | None:
        return max(self.queue_depths) if self.queue_depths else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "work_unit": SERVICE_UNITS[self.service],
            "offered_concurrency": self.offered_concurrency,
            "requested_concurrency": self.requested_concurrency or self.offered_concurrency,
            "duration_s": self.duration_s,
            "completed_requests": self.completed_requests,
            "attempted_requests": self.attempted_requests,
            "failed_requests": self.failed_requests,
            "req_s": self.req_s,
            "work_units_completed": self.work_units,
            "img_s": self.img_s,
            "error_rate": self.error_rate,
            "http_429_count": self.status_429,
            "http_503_count": self.status_503,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "observed_batch_size_mean": self.batch_size_mean,
            "observed_batch_sizes": self.batch_sizes,
            "queue_depth_mean": self.queue_depth_mean,
            "queue_depth_max": self.queue_depth_max,
            "execution_traces": self.execution_traces,
            "cycles_completed": self.cycles_completed,
            "cycles_failed": self.cycles_failed,
            "batch_snapshot_before": self.batch_snapshot_before,
            "batch_snapshot_after": self.batch_snapshot_after,
            "droid_leases_after_level": self.droid_leases_after_level,
            "memory_before_level": self.memory_before_level,
            "memory_after_level": self.memory_after_level,
        }


class SweepWorkload:
    service: str
    endpoint: str

    async def run_cycle(self, client: AsyncHttpClient, run_id: str, worker_index: int) -> CycleResult:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticWorkload(SweepWorkload):
    operation: PreparedOperation

    @property
    def service(self) -> str:
        return self.operation.service

    @property
    def endpoint(self) -> str:
        return self.operation.endpoint

    async def run_cycle(self, client: AsyncHttpClient, run_id: str, worker_index: int) -> CycleResult:
        row = await self.operation.invoke(client, f"{run_id}-w{worker_index}-{uuid.uuid4().hex[:8]}")
        return CycleResult((row,), 1 if row.get("success") else 0)


@dataclass(frozen=True)
class DroidFrame:
    rgb: bytes
    rgb_shape: tuple[int, ...]
    mask: bytes
    mask_shape: tuple[int, ...]
    source_frame_index: int
    timestamp_s: float


@dataclass(frozen=True)
class DroidWorkload(SweepWorkload):
    endpoint_base: str
    model_revision: str
    frames: tuple[DroidFrame, ...]

    service: str = "droid"

    @property
    def endpoint(self) -> str:
        return self.endpoint_base

    async def run_cycle(self, client: AsyncHttpClient, run_id: str, worker_index: int) -> CycleResult:
        rows: list[Mapping[str, Any]] = []
        session_id: str | None = None
        source_id = f"throughput-sweep-droid-w{worker_index}"
        create_meta = {
            "ownership": ownership(run_id, "droid.create_session", source_id, source_id),
            "camera": droid_camera_contract(),
            "image_shape": {"height": 320, "width": 568},
            "options": {"buffer": 256, "filter_thresh": 1.0, "keyframe_thresh": 2.0, "warmup": 8},
            "model_revision": self.model_revision,
        }
        create_body, create_ct = build_multipart_request_fields(create_meta, {})
        create = await post_raw(
            client, service="droid.create_session", endpoint=f"{self.endpoint_base}/droid.create_session",
            body=create_body, content_type=create_ct,
        )
        rows.append(create)
        response = create.get("response")
        if isinstance(response, Mapping) and isinstance(response.get("session_id"), str):
            session_id = str(response["session_id"])

        pushed = 0
        try:
            if session_id is not None and create.get("success"):
                for frame in self.frames:
                    metadata = {
                        "ownership": ownership(
                            run_id, "droid.push_frame", f"frame-{frame.source_frame_index}",
                            f"{source_id}:{frame.source_frame_index}", frame.timestamp_s,
                        ),
                        "session_id": session_id,
                        "frame_id": f"frame-{frame.source_frame_index}",
                        "source_timestamp_s": frame.timestamp_s,
                        "model_revision": self.model_revision,
                    }
                    body, content_type = build_multipart_request_fields(
                        metadata,
                        {
                            "rgb": (frame.rgb, frame.rgb_shape, "uint8"),
                            "static_confidence_mask": (frame.mask, frame.mask_shape, "float32"),
                        },
                    )
                    push = await post_raw(
                        client, service="droid.push_frame", endpoint=f"{self.endpoint_base}/droid.push_frame",
                        body=body, content_type=content_type,
                    )
                    rows.append(push)
                    if not push.get("success"):
                        break
                    pushed += 1
        finally:
            # Dispatcher affinity is released by a terminal finalize.  Run it even
            # after a failed frame so a rejected level cannot retain a leased session.
            if session_id is not None:
                finalize_meta = {
                    "ownership": ownership(run_id, "droid.finalize", source_id, source_id),
                    "session_id": session_id,
                    "model_revision": self.model_revision,
                }
                body, content_type = build_multipart_request_fields(finalize_meta, {})
                rows.append(await post_raw(
                    client, service="droid.finalize", endpoint=f"{self.endpoint_base}/droid.finalize",
                    body=body, content_type=content_type,
                ))
        return CycleResult(tuple(rows), pushed)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Reuse the benchmark harness's percentile definition for comparable p50/p95."""
    return benchmark_percentile(values, fraction)


def extract_trace(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        response = row.get("response")
        if isinstance(response, Mapping):
            result = response.get("result")
            if isinstance(result, Mapping):
                trace = result.get("trace")
    return dict(trace) if isinstance(trace, Mapping) else None


def trace_batch_size(row: Mapping[str, Any]) -> float | None:
    trace = extract_trace(row)
    if trace is None:
        return None
    for key in ("request_count", "batch_size", "effective_batch_size"):
        value = trace.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def rows_into_level(level: LevelResult, result: CycleResult) -> None:
    if result.success:
        level.cycles_completed += 1
    else:
        level.cycles_failed += 1
    level.work_units += result.work_units
    for row in result.rows:
        level.attempted_requests += 1
        if row.get("success"):
            level.completed_requests += 1
        else:
            level.failed_requests += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            level.latencies_ms.append(float(latency))
        status = row.get("http_status")
        if status == 429:
            level.status_429 += 1
        elif status == 503:
            level.status_503 += 1
        trace = extract_trace(row)
        if trace is not None:
            level.execution_traces.append(trace)
            for key in ("queue_depth", "queue_size", "queued_requests"):
                depth = trace.get(key)
                if isinstance(depth, (int, float)) and not isinstance(depth, bool) and depth >= 0:
                    level.queue_depths.append(float(depth))
                    break
        batch_size = trace_batch_size(row)
        if batch_size is not None:
            level.batch_sizes.append(batch_size)


async def snapshot_batches(client: AsyncHttpClient, endpoint: str) -> Mapping[str, Any] | None:
    parts = urlsplit(endpoint)
    origin = f"{parts.scheme}://{parts.netloc}"
    candidates = (f"{origin}/-/batch-snapshot", f"{endpoint.rstrip('/')}/-/batch-snapshot")
    observations: list[dict[str, Any]] = []
    for candidate in dict.fromkeys(candidates):
        try:
            response = await client.get(candidate)
            payload = response.json()
            observations.append({"url": candidate, "http_status": response.status_code, "body": payload})
        except Exception as exc:
            observations.append({"url": candidate, "error": repr(exc)})
    return {"observations": observations}


async def run_service_level(
    workload: SweepWorkload,
    client: AsyncHttpClient,
    *,
    offered_concurrency: int,
    requested_concurrency: int | None,
    duration_s: float | None,
    requests_per_worker: int | None,
    run_id: str,
    ready: asyncio.Event,
    release: asyncio.Event,
) -> LevelResult:
    ready.set()
    await release.wait()
    before = await snapshot_batches(client, workload.endpoint)
    started = time.monotonic()
    result = LevelResult(service=workload.service, offered_concurrency=offered_concurrency, duration_s=0.0, completed_requests=0, attempted_requests=0, failed_requests=0, work_units=0, requested_concurrency=requested_concurrency, batch_snapshot_before=before)
    deadline = started + duration_s if duration_s is not None else None

    async def worker(worker_index: int) -> None:
        completed_cycles = 0
        while True:
            if requests_per_worker is not None:
                if completed_cycles >= requests_per_worker:
                    return
            elif deadline is not None:
                if completed_cycles and time.monotonic() >= deadline:
                    return
            else:
                raise RuntimeError("one stopping condition is required")
            cycle = await workload.run_cycle(client, run_id, worker_index)
            rows_into_level(result, cycle)
            completed_cycles += 1

    await asyncio.gather(*(worker(worker_index) for worker_index in range(offered_concurrency)))
    result.duration_s = max(time.monotonic() - started, 1e-9)
    result.batch_snapshot_after = await snapshot_batches(client, workload.endpoint)
    return result


def find_knee(levels: Sequence[LevelResult], marginal_gain_threshold: float) -> dict[str, Any]:
    """Return a confirmed knee: decline/backpressure immediately, plateau twice."""
    if not levels:
        return {"knee_offered_concurrency": None, "reason": "no_levels"}
    ordered = sorted(levels, key=lambda item: item.offered_concurrency)
    low_gain_start: LevelResult | None = None
    previous = ordered[0]
    for current in ordered[1:]:
        backpressure_rate = (current.status_429 + current.status_503) / max(current.attempted_requests, 1)
        if backpressure_rate >= SIGNIFICANT_BACKPRESSURE_RATE:
            return {
                "knee_offered_concurrency": current.offered_concurrency,
                "reason": "significant_backpressure",
                "backpressure_rate": backpressure_rate,
                "http_429_count": current.status_429,
                "http_503_count": current.status_503,
            }
        delta = current.img_s - previous.img_s
        marginal_gain = delta / max(previous.img_s, 1e-9)
        if delta < 0:
            return {"knee_offered_concurrency": current.offered_concurrency, "reason": "throughput_declined", "previous_img_s": previous.img_s, "current_img_s": current.img_s, "marginal_gain": marginal_gain}
        if marginal_gain < marginal_gain_threshold:
            if low_gain_start is not None:
                return {
                    "knee_offered_concurrency": low_gain_start.offered_concurrency,
                    "confirmation_offered_concurrency": current.offered_concurrency,
                    "reason": "consecutive_marginal_gain_below_threshold",
                    "previous_img_s": previous.img_s,
                    "current_img_s": current.img_s,
                    "marginal_gain": marginal_gain,
                }
            low_gain_start = current
        else:
            low_gain_start = None
        previous = current
    return {"knee_offered_concurrency": ordered[-1].offered_concurrency, "reason": "no_saturation_observed", "current_img_s": ordered[-1].img_s}


def allocate_card_budget(requested: Mapping[str, int], *, gpu1_cap: int, gpu3_cap: int) -> dict[str, int]:
    """Cap combined logical services on GPU1/GPU3 while preserving each card's total load."""
    assigned = dict(requested)
    for services, cap in ((("hands", "wilor"), gpu1_cap), (("hawor", "infiller"), gpu3_cap)):
        present = [service for service in services if service in requested]
        if len(present) < 2:
            continue
        total = sum(requested[service] for service in present)
        if cap < len(present):
            raise ValueError(f"shared-card budget {cap} cannot supply one in-flight request to each of {', '.join(present)}")
        if total <= cap:
            continue
        # Equal fractional allocation, then deterministic remainder assignment,
        # avoids one logical service silently consuming a shared physical card.
        shares = {service: requested[service] * cap / total for service in present}
        assigned.update({service: max(1, int(shares[service])) for service in present})
        remainder = cap - sum(assigned[service] for service in present)
        for service in sorted(present, key=lambda name: (shares[name] - int(shares[name]), name), reverse=True):
            if remainder <= 0:
                break
            assigned[service] += 1
            remainder -= 1
    return assigned


def parse_levels(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("levels must be positive comma-separated integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("levels must be strictly increasing and unique")
    return values


def parse_per_service_levels(raw: str | None, selected: Sequence[str], defaults: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    levels = {service: defaults for service in selected}
    if not raw:
        return levels
    for clause in raw.split(";"):
        if not clause.strip():
            continue
        service, separator, values = clause.partition("=")
        if not separator or service.strip() not in levels:
            raise ValueError("--service-levels uses service=1,2;service=1,2 and only selected services")
        levels[service.strip()] = parse_levels(values)
    if max(levels.get("droid", (0,))) > DROID_MAX_SESSIONS:
        raise ValueError(f"DROID offered sessions cannot exceed {DROID_MAX_SESSIONS}")
    return levels


def expected_gpus_for_services(services: Iterable[str]) -> frozenset[int]:
    selected = tuple(services)
    unknown = set(selected) - SERVICE_EXPECTED_GPUS.keys()
    if unknown:
        raise ValueError(f"unknown services have no GPU ownership declaration: {sorted(unknown)}")
    return frozenset().union(*(SERVICE_EXPECTED_GPUS[service] for service in selected))


def _cards_by_index(snapshot: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        card["index"]: card for card in snapshot.get("cards", [])
        if isinstance(card, Mapping) and isinstance(card.get("index"), int)
    }


def external_gpu_guard(
    before: Mapping[str, Any], after: Mapping[str, Any], selected_services: Iterable[str],
    *, memory_delta_mib: float = EXTERNAL_GPU_MEMORY_DELTA_MIB,
) -> dict[str, Any]:
    """Classify newly-started CUDA work by physical-GPU ownership.

    A service's expected cards are normal load destinations for this scan.  The
    guard therefore only aborts for a new CUDA PID plus a material card-memory
    increase on an unowned card; it records owned-card activity for diagnosis.
    """
    service_gpus = {service: sorted(SERVICE_EXPECTED_GPUS[service]) for service in selected_services}
    # A resident but unselected service is still our deployment, not external work.
    # Only a card without any declared service owner can trigger this guard.
    expected_gpus = expected_gpus_for_services(SERVICE_EXPECTED_GPUS)
    before_cards, after_cards = _cards_by_index(before), _cards_by_index(after)
    observations: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    ignored_expected_activity: list[dict[str, Any]] = []
    for index, current in sorted(after_cards.items()):
        previous = before_cards.get(index, {})
        before_pids = {
            int(process["pid"]) for process in previous.get("processes", [])
            if isinstance(process, Mapping) and isinstance(process.get("pid"), int)
        }
        new_processes = [
            dict(process) for process in current.get("processes", [])
            if isinstance(process, Mapping) and isinstance(process.get("pid"), int) and process["pid"] not in before_pids
        ]
        before_memory = previous.get("memory_used_mib")
        after_memory = current.get("memory_used_mib")
        delta = (
            float(after_memory) - float(before_memory)
            if isinstance(before_memory, (int, float)) and isinstance(after_memory, (int, float))
            else None
        )
        activity = {"gpu_index": index, "memory_delta_mib": delta, "new_cuda_processes": new_processes}
        if index in expected_gpus:
            if new_processes or (delta is not None and delta >= memory_delta_mib):
                ignored_expected_activity.append(activity)
            continue
        if new_processes:
            observations.append(activity)
            if delta is not None and delta >= memory_delta_mib:
                violations.append(activity)
    return {
        "expected_service_gpus": service_gpus,
        "expected_gpus": sorted(expected_gpus),
        "unowned_gpus": sorted(set(after_cards) - expected_gpus),
        "memory_delta_threshold_mib": memory_delta_mib,
        "ignored_expected_gpu_activity": ignored_expected_activity,
        "new_unowned_cuda_activity": observations,
        "abort": bool(violations),
        "violations": violations,
    }


def gpu_memory_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5.0)
        cards = []
        uuid_to_index: dict[str, int] = {}
        for line in completed.stdout.splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) == 5:
                index = int(cells[0])
                uuid_to_index[cells[1]] = index
                cards.append({"index": index, "uuid": cells[1], "memory_used_mib": float(cells[2]), "memory_total_mib": float(cells[3]), "utilization_pct": float(cells[4]), "processes": []})
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=5.0,
        )
        cards_by_index = {card["index"]: card for card in cards}
        for line in apps.stdout.splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) != 3 or cells[0] not in uuid_to_index:
                continue
            try:
                cards_by_index[uuid_to_index[cells[0]]]["processes"].append({"pid": int(cells[1]), "memory_used_mib": float(cells[2])})
            except ValueError:
                continue
        return {"observed_at": datetime.now(timezone.utc).isoformat(), "cards": cards}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"observed_at": datetime.now(timezone.utc).isoformat(), "unavailable": repr(exc)}


class NvmlSampler:
    """High-frequency NVML memory/utilization samples without a 1 Hz CLI blind spot."""

    def __init__(self, sample_hz: float) -> None:
        self.interval_s = 1.0 / sample_hz
        self.samples: list[dict[str, Any]] = []
        self.unavailable: str | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:
            self.unavailable = repr(exc)
            return
        self._task = asyncio.create_task(self._collect())

    async def _collect(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            cards = []
            for index, handle in enumerate(self._handles):
                try:
                    memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                    cards.append({"index": index, "memory_used_mib": memory.used / (1024 * 1024), "memory_total_mib": memory.total / (1024 * 1024), "utilization_pct": utilization.gpu})
                except Exception as exc:
                    cards.append({"index": index, "error": repr(exc)})
            self.samples.append({"monotonic_s": now, "cards": cards})
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._task is not None:
            await self._task
        try:
            if hasattr(self, "_pynvml"):
                self._pynvml.nvmlShutdown()
        except Exception as exc:
            self.unavailable = self.unavailable or repr(exc)
        return {"sample_hz": 1.0 / self.interval_s, "samples": self.samples, "unavailable": self.unavailable}


def memory_return_check(baseline: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], final: Mapping[str, Any]) -> dict[str, Any]:
    before = {card["index"]: card["memory_used_mib"] for card in baseline.get("cards", []) if isinstance(card, Mapping) and isinstance(card.get("index"), int) and isinstance(card.get("memory_used_mib"), (int, float))}
    after = {card["index"]: card["memory_used_mib"] for card in final.get("cards", []) if isinstance(card, Mapping) and isinstance(card.get("index"), int) and isinstance(card.get("memory_used_mib"), (int, float))}
    peaks: dict[int, float] = {}
    for sample in samples:
        for card in sample.get("cards", []) if isinstance(sample, Mapping) else ():
            if isinstance(card, Mapping) and isinstance(card.get("index"), int) and isinstance(card.get("memory_used_mib"), (int, float)):
                peaks[card["index"]] = max(peaks.get(card["index"], float(card["memory_used_mib"])), float(card["memory_used_mib"]))
    return {"cards": [{"index": index, "baseline_mib": used, "peak_mib": peaks.get(index), "final_mib": after.get(index), "final_minus_baseline_mib": None if index not in after else after[index] - used, "reclaimed_from_peak_mib": None if index not in after or index not in peaks else peaks[index] - after[index]} for index, used in sorted(before.items())], "note": "CUDA caching may retain a high-water reserve; this records return behavior without treating nonzero reserve as a leak."}


def droid_lease_snapshot(path: Path) -> dict[str, Any]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            affinities = connection.execute("SELECT replica_url, COUNT(*) FROM droid_session_affinity GROUP BY replica_url").fetchall()
            counters = connection.execute("SELECT replica_url, active_sessions FROM replica_inflight").fetchall()
        finally:
            connection.close()
        affinity_by_lane = {str(lane): int(count) for lane, count in affinities}
        counter_by_lane = {str(lane): int(count) for lane, count in counters}
        return {"path": str(path), "affinity_by_lane": affinity_by_lane, "counter_by_lane": counter_by_lane, "affinity_total": sum(affinity_by_lane.values()), "counter_total": sum(counter_by_lane.values())}
    except sqlite3.Error as exc:
        return {"path": str(path), "unavailable": repr(exc)}


def droid_leases_are_zero(snapshot: Mapping[str, Any]) -> bool:
    return snapshot.get("affinity_total") == 0 and snapshot.get("counter_total") == 0


def _fake_row(*, service: str, endpoint: str, status: int, payload: Mapping[str, Any], latency_ms: float) -> httpx.Response:
    return httpx.Response(status, json=payload, headers={"content-type": "application/json"}, request=httpx.Request("POST", endpoint))


class FakeSweepClient:
    """No-GPU in-process peer that records concurrent starts and DROID releases."""

    def __init__(self) -> None:
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}
        self.sessions: set[str] = set()
        self.finalized_sessions: list[str] = []
        self.calls: list[str] = []

    async def post(self, url: str, *, content: bytes, headers: Mapping[str, str]) -> httpx.Response:
        self.calls.append(url)
        parsed = urlsplit(url)
        endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        self.active[endpoint] = self.active.get(endpoint, 0) + 1
        self.max_active[endpoint] = max(self.max_active.get(endpoint, 0), self.active[endpoint])
        try:
            metadata = _metadata_from_request(content, headers.get("Content-Type", ""))
            trace = {"request_count": self.active[endpoint], "batch_id": f"fake-{len(self.calls)}"}
            if parsed.path.endswith("droid.create_session"):
                session_id = f"fake-session-{uuid.uuid4().hex[:10]}"
                self.sessions.add(session_id)
                payload = {"session_id": session_id, "trace": trace}
            elif parsed.path.endswith("droid.push_frame"):
                session_id = metadata.get("session_id")
                if session_id not in self.sessions:
                    return _fake_row(service="droid", endpoint=endpoint, status=404, payload={"error": {"code": "unknown_session"}}, latency_ms=0.0)
                payload = {"result": {"trace": trace}}
            elif parsed.path.endswith("droid.finalize"):
                session_id = metadata.get("session_id")
                if isinstance(session_id, str):
                    self.sessions.discard(session_id)
                    self.finalized_sessions.append(session_id)
                payload = {"result": {"trace": trace}}
            else:
                payload = {"result": {"trace": trace}}
            await asyncio.sleep(0)
            return _fake_row(service="fake", endpoint=endpoint, status=200, payload=payload, latency_ms=0.0)
        finally:
            self.active[endpoint] -= 1

    async def get(self, url: str) -> httpx.Response:
        max_inflight = max(self.max_active.values(), default=0)
        return httpx.Response(200, json={"batch_size": max_inflight, "active": dict(self.active)}, request=httpx.Request("GET", url))


def _metadata_from_request(body: bytes, content_type: str) -> Mapping[str, Any]:
    from ego_annotation.serving.transport import _iter_multipart
    try:
        for name, data, _params in _iter_multipart(body, content_type):
            if name == "metadata":
                decoded = json.loads(data.decode("utf-8"))
                return decoded if isinstance(decoded, Mapping) else {}
    except Exception:
        return {}
    return {}


def fake_workloads(selected: Sequence[str]) -> dict[str, SweepWorkload]:
    workloads: dict[str, SweepWorkload] = {}
    base = "http://fake-server"
    for service in selected:
        if service == "droid":
            frame = DroidFrame(b"\x00" * (320 * 568 * 3), (320, 568, 3), b"\x00" * (320 * 568 * 4), (320, 568), 0, 0.0)
            workloads[service] = DroidWorkload(base, "droid-v1", (frame,))
            continue
        endpoint = f"{base}/{service}"
        async def invoke(client: AsyncHttpClient, _run_id: str, *, name: str = service, url: str = endpoint) -> dict[str, Any]:
            return await post_raw(client, service=name, endpoint=url, body=b"fake", content_type="application/json")
        workloads[service] = StaticWorkload(PreparedOperation(service, endpoint, invoke, {"source": "fake"}))
    return workloads


def real_droid_workload(args: argparse.Namespace, router: ModelServiceRouter) -> DroidWorkload:
    manifest_path = resolve_existing(
        "DROID manifest", env_path("EGO_STRESS_DROID_MANIFEST", args.droid_manifest), (DEFAULT_DROID_MANIFEST,),
    )
    manifest = read_json(manifest_path)
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or len(payloads) < args.droid_frames_per_session:
        raise ValueError(f"DROID manifest requires {args.droid_frames_per_session} payloads: {manifest_path}")
    frames: list[DroidFrame] = []
    for payload in payloads[:args.droid_frames_per_session]:
        rgb_path, mask_path = Path(payload["rgb_path"]), Path(payload["mask_path"])
        rgb, mask = rgb_path.read_bytes(), mask_path.read_bytes()
        if len(rgb) != 320 * 568 * 3 or len(mask) != 320 * 568 * 4:
            raise ValueError(f"DROID payload has unexpected model-native shapes: {payload.get('payload_id')}")
        frames.append(DroidFrame(rgb, (320, 568, 3), mask, (320, 568), int(payload["source_frame_index"]), float(payload["timestamp_s"])))
    return DroidWorkload(router.base_url_for(ModelApiName.DROID_CREATE_SESSION), router.endpoint_for(ModelApiName.DROID_CREATE_SESSION).model_revision, tuple(frames))


def real_workloads(args: argparse.Namespace, router: ModelServiceRouter, selected: Sequence[str]) -> dict[str, SweepWorkload]:
    static_selected = tuple(service for service in selected if service != "droid")
    operations = prepare_operations(args, router, static_selected)
    workloads: dict[str, SweepWorkload] = {name: StaticWorkload(operation) for name, operation in operations.items()}
    if "droid" in selected:
        workloads["droid"] = real_droid_workload(args, router)
    return workloads


def write_summary_csv(path: Path, levels: Iterable[LevelResult]) -> None:
    rows = [level.to_dict() for level in levels]
    columns = [
        "api_name", "offered_intensity_per_s", "duration_s", "throughput_work_units_per_s",
        "response_latency_p50_ms", "response_latency_p95_ms", "response_latency_p99_ms", "service", "work_unit",
        "offered_concurrency", "requested_concurrency", "completed_requests", "attempted_requests", "failed_requests",
        "req_s", "work_units_completed", "img_s", "error_rate", "http_429_count", "http_503_count",
        "observed_batch_size_mean", "queue_depth_mean", "queue_depth_max", "cycles_completed", "cycles_failed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in columns}
            csv_row.update({
                "api_name": row["service"], "offered_intensity_per_s": row["offered_concurrency"],
                "throughput_work_units_per_s": row["img_s"], "response_latency_p50_ms": row["latency_p50_ms"],
                "response_latency_p95_ms": row["latency_p95_ms"], "response_latency_p99_ms": row["latency_p99_ms"],
            })
            writer.writerow(csv_row)


async def run_sweep(
    workloads: Mapping[str, SweepWorkload],
    levels_by_service: Mapping[str, tuple[int, ...]],
    *,
    client: AsyncHttpClient,
    duration_s: float | None,
    requests_per_worker: int | None,
    artifact_dir: Path,
    droid_db_path: Path | None,
    fake_server: bool,
    knee_threshold: float,
    gpu1_max_in_flight: int = 32,
    gpu3_max_in_flight: int = 32,
    nvml_sample_hz: float = 10.0,
) -> tuple[list[LevelResult], dict[str, Any]]:
    all_levels: list[LevelResult] = []
    memory_before = {"skipped": "fake_server"} if fake_server else gpu_memory_snapshot()
    sampler = NvmlSampler(nvml_sample_hz)
    if not fake_server:
        await sampler.start()
    droid_before = {"fake_sessions": 0} if fake_server and "droid" in workloads else droid_lease_snapshot(droid_db_path) if droid_db_path and "droid" in workloads else None
    max_depth = max((len(values) for values in levels_by_service.values()), default=0)
    levels_dir = artifact_dir / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)
    guard_checks: list[dict[str, Any]] = []
    stopped_services: dict[str, Mapping[str, Any]] = {}

    for index in range(max_depth):
        requested = {
            service: values[index] for service, values in levels_by_service.items()
            if index < len(values) and service not in stopped_services
        }
        if not requested:
            break
        memory_before_level = {"skipped": "fake_server"} if fake_server else gpu_memory_snapshot()
        assigned = allocate_card_budget(requested, gpu1_cap=gpu1_max_in_flight, gpu3_cap=gpu3_max_in_flight)
        release = asyncio.Event()
        ready_events = {service: asyncio.Event() for service in assigned}
        tasks = {
            service: asyncio.create_task(run_service_level(
                workloads[service], client, offered_concurrency=load, requested_concurrency=requested[service], duration_s=duration_s,
                requests_per_worker=requests_per_worker, run_id=f"{artifact_dir.name}-l{index}-{service}",
                ready=ready_events[service], release=release,
            ))
            for service, load in assigned.items()
        }
        await asyncio.gather(*(event.wait() for event in ready_events.values()))
        release.set()
        results = await asyncio.gather(*tasks.values())
        memory_after_level = {"skipped": "fake_server"} if fake_server else gpu_memory_snapshot()
        droid_after_level = ({"fake_sessions": len(client.sessions), "finalized_sessions": list(client.finalized_sessions)} if fake_server else droid_lease_snapshot(droid_db_path)) if "droid" in assigned else None
        if "droid" in assigned:
            zero = droid_after_level.get("fake_sessions") == 0 if fake_server else droid_leases_are_zero(droid_after_level or {})
            if not zero:
                raise RuntimeError(f"DROID lifecycle leaked dispatcher state after level {index}: {droid_after_level}")
        for result in results:
            result.memory_before_level = memory_before_level
            result.memory_after_level = memory_after_level
            result.droid_leases_after_level = droid_after_level if result.service == "droid" else None
            all_levels.append(result)
            path = levels_dir / f"{result.service}_load_{result.offered_concurrency}.json"
            path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not fake_server:
            guard = external_gpu_guard(memory_before_level, memory_after_level, workloads.keys())
            guard["level_index"] = index
            guard_checks.append(guard)
            if guard["abort"]:
                raise RuntimeError(f"external CUDA work appeared on unowned GPU(s): {guard['violations']}")
        for service in requested:
            knee = find_knee([level for level in all_levels if level.service == service], knee_threshold)
            if knee["reason"] != "no_saturation_observed":
                stopped_services[service] = knee

    memory_after = {"skipped": "fake_server"} if fake_server else gpu_memory_snapshot()
    nvml = {"sample_hz": nvml_sample_hz, "samples": [], "unavailable": "skipped: fake_server"} if fake_server else await sampler.stop()
    droid_after = {"fake_sessions": len(client.sessions), "finalized_sessions": list(client.finalized_sessions)} if fake_server and "droid" in workloads else droid_lease_snapshot(droid_db_path) if droid_db_path and "droid" in workloads else None
    if "droid" in workloads:
        zero = droid_after.get("fake_sessions") == 0 if fake_server else droid_leases_are_zero(droid_after or {})
        if not zero:
            raise RuntimeError(f"DROID lifecycle leaked dispatcher state: {droid_after}")

    knees = {service: find_knee([level for level in all_levels if level.service == service], knee_threshold) for service in workloads}
    report = {
        "schema": "ego.throughput-sweep.v1",
        "client_only": True,
        "fake_server": fake_server,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "nvml_samples": nvml,
        "memory_return_check": memory_return_check(memory_before, nvml["samples"], memory_after),
        "droid_leases_before": droid_before,
        "droid_leases_after": droid_after,
        "external_gpu_guard_checks": guard_checks,
        "stopped_services": stopped_services,
        "knees": knees,
        "level_count": len(all_levels),
    }
    return all_levels, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent multi-service throughput sweep and knee detector")
    parser.add_argument("--apis", help="comma-separated services; default all seven")
    parser.add_argument("--levels", default="1,2,4,8,16,32", help="default offered concurrency levels per service")
    parser.add_argument("--service-levels", help="per-service overrides, e.g. droid=1,2,4,8;unidepth=1,2,4,8,16")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--duration-s", type=float, default=60.0, help="persistent measurement duration per level (default: 60; use 60-300 live)")
    mode.add_argument("--requests-per-worker", type=int, help="fixed cycles per concurrent worker instead of a duration")
    parser.add_argument("--droid-frames-per-session", type=int, default=DEFAULT_DROID_FRAMES_PER_SESSION)
    parser.add_argument("--knee-marginal-gain", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=ARTIFACT_BASE)
    parser.add_argument("--run-id", help="artifact directory name; defaults to throughput_sweep_<UTC timestamp>")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu1-max-in-flight", type=int, default=32, help="combined Hands+WiLoR target-in-flight budget on shared GPU1")
    parser.add_argument("--gpu3-max-in-flight", type=int, default=32, help="combined HaWoR+Infiller target-in-flight budget on shared GPU3")
    parser.add_argument("--nvml-sample-hz", type=float, default=10.0, help="NVML sampler cadence during a scan (default: 10 Hz)")
    parser.add_argument("--droid-lease-db", type=Path, default=Path("/tmp/ego_lane_dispatcher_leases.sqlite3"))
    parser.add_argument("--fake-server", action="store_true", help="in-process no-GPU fake endpoints with DROID lifecycle accounting")
    parser.add_argument("--dry-run", action="store_true", help="write route/level plan only; send no network requests")
    # These preserve parallel_stress_test's real payload selection and fallbacks.
    parser.add_argument("--unidepth-frame", type=Path)
    parser.add_argument("--gpu1-root", type=Path)
    parser.add_argument("--gpu1-frames", type=Path)
    parser.add_argument("--wilor-root", type=Path)
    parser.add_argument("--wilor-config", type=Path)
    parser.add_argument("--droid-manifest", type=Path)
    parser.add_argument("--hawor-root", type=Path)
    parser.add_argument("--hawor-request", type=Path)
    parser.add_argument("--infiller-request", type=Path)
    parser.add_argument("--cosmos-request", type=Path)
    parser.add_argument("--cosmos-headers", type=Path)
    parser.add_argument("--hands-revision")
    parser.add_argument("--wilor-revision")
    parser.add_argument("--wilor-endpoint")
    args = parser.parse_args(argv)
    try:
        args.selected = normalize_apis(args.apis)
        args.default_levels = parse_levels(args.levels)
        args.levels_by_service = parse_per_service_levels(args.service_levels, args.selected, args.default_levels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.requests_per_worker is not None and args.requests_per_worker <= 0:
        parser.error("--requests-per-worker must be positive")
    if args.droid_frames_per_session <= 0:
        parser.error("--droid-frames-per-session must be positive")
    if args.gpu1_max_in_flight <= 0 or args.gpu3_max_in_flight <= 0:
        parser.error("shared-GPU in-flight budgets must be positive")
    if not 1 <= args.nvml_sample_hz <= 20:
        parser.error("--nvml-sample-hz must be between 1 and 20")
    if not 0 <= args.knee_marginal_gain <= 1:
        parser.error("--knee-marginal-gain must be between 0 and 1")
    args.run_id = args.run_id or datetime.now(timezone.utc).strftime("throughput_sweep_%Y%m%dT%H%M%SZ")
    return args


async def async_main(args: argparse.Namespace) -> int:
    artifact_dir = args.out / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    router = ModelServiceRouter.canonical()
    route_plan = {service: route_for_service(router, service) for service in args.selected}
    plan = {
        "schema": "ego.throughput-sweep-plan.v1", "client_only": True, "selected_services": list(args.selected),
        "levels_by_service": {name: list(values) for name, values in args.levels_by_service.items()},
        "routes": route_plan, "duration_s": args.duration_s, "requests_per_worker": args.requests_per_worker,
        "droid_frames_per_session": args.droid_frames_per_session, "gpu1_max_in_flight": args.gpu1_max_in_flight, "gpu3_max_in_flight": args.gpu3_max_in_flight, "nvml_sample_hz": args.nvml_sample_hz, "dry_run": args.dry_run, "fake_server": args.fake_server,
    }
    (artifact_dir / "run_manifest.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"artifact_dir={artifact_dir}")
        print("dry_run=true; no payload preparation, HTTP request, or GPU query was made")
        return 0

    workloads = fake_workloads(args.selected) if args.fake_server else real_workloads(args, router, args.selected)
    if args.fake_server:
        client: AsyncHttpClient = FakeSweepClient()
        levels, report = await run_sweep(workloads, args.levels_by_service, client=client, duration_s=args.duration_s, requests_per_worker=args.requests_per_worker, artifact_dir=artifact_dir, droid_db_path=None, fake_server=True, knee_threshold=args.knee_marginal_gain, gpu1_max_in_flight=args.gpu1_max_in_flight, gpu3_max_in_flight=args.gpu3_max_in_flight, nvml_sample_hz=args.nvml_sample_hz)
    else:
        timeout = httpx.Timeout(args.timeout_s)
        async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=512, max_keepalive_connections=256)) as client:
            levels, report = await run_sweep(workloads, args.levels_by_service, client=client, duration_s=args.duration_s, requests_per_worker=args.requests_per_worker, artifact_dir=artifact_dir, droid_db_path=args.droid_lease_db, fake_server=False, knee_threshold=args.knee_marginal_gain, gpu1_max_in_flight=args.gpu1_max_in_flight, gpu3_max_in_flight=args.gpu3_max_in_flight, nvml_sample_hz=args.nvml_sample_hz)

    write_summary_csv(artifact_dir / "summary.csv", levels)
    plot_paths = plot_throughput_latency(artifact_dir / "summary.csv", artifact_dir / "plots")
    (artifact_dir / "knee_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"artifact_dir={artifact_dir}")
    for service in args.selected:
        knee = report["knees"][service]
        print(f"service={service} knee_load={knee['knee_offered_concurrency']} reason={knee['reason']}")
    print(f"plots={len(plot_paths)}")
    return 0


def route_for_service(router: ModelServiceRouter, service: str) -> dict[str, str]:
    if service == "droid":
        return {api.value: router.url_for(api) for api in (ModelApiName.DROID_CREATE_SESSION, ModelApiName.DROID_PUSH_FRAME, ModelApiName.DROID_FINALIZE)}
    api = {
        "unidepth": ModelApiName.UNIDEPTH_INFER, "hands": ModelApiName.HANDS_DETECT,
        "wilor": ModelApiName.WILOR_RECONSTRUCT, "hawor": ModelApiName.HAWOR_INFER_TRACKS,
        "infiller": ModelApiName.HAWOR_INFILLER_FILL, "cosmos": ModelApiName.COSMOS3_REASON,
    }[service]
    return {api.value: router.url_for(api)}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
