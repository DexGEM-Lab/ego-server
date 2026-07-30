"""Open-loop offered-load generator: scheduled arrivals independent of completions.

An open-loop load generator fires requests at scheduled wall-clock times regardless
of how many are in flight. This is the only honest way to measure saturation: under
overload, in-flight count grows and latency rises even though offered rate is fixed.
A closed-loop generator (fire next on completion) *hides* overload by self-limiting
to the in-flight count, which is exactly what this benchmark must not do.

Design:

* **Scheduled arrivals.** Each level defines an offered intensity (native work
  units/s) and a target number of completed requests. The generator computes a
  schedule of item offer times from the manifest at the configured rate, sleeps
  until each offer time, and submits the item to the gateway. Submission is not
  delayed by in-flight count or by a completed response.
* **Independent of completions.** The generator submits at scheduled times even if
  earlier requests have not returned. A bounded in-flight semaphore is optional and
  defaults to a large value (or unbounded) so the generator does not self-limit;
  when it is set, hitting the bound records a ``rejected`` (backpressure) outcome —
  it never delays the next offer to wait for a completion.
* **Per-item records.** Each item records offered/submit/response times, outcome,
  server phase decomposition (from result.metadata), batch trace, payload hash, and
  model-load count. Records are appended to a JSONL artifact as they settle.
* **Distinct payloads.** Items come from a ``PayloadManifest`` whose hashes are all
  distinct; the run manifest records ``distinct_payload_hashes``.
* **Offered vs admitted vs completed vs rejected.** The four rates are computed by
  ``metrics.summarize`` from the per-item outcomes.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence

from ego_annotation.serving.benchmark.manifest import PayloadItem, PayloadManifest
from ego_annotation.serving.benchmark.metrics import (
    ItemRecord,
    OUTCOME_IN_FLIGHT,
    outcome_for_error_code,
)
from ego_annotation.serving.contracts import BatchTrace, ErrorCode
from ego_annotation.serving.gateway import GatewayRequest, GatewayResponse
from ego_annotation.serving.router import ModelApiName


class AsyncGatewayCaller(Protocol):
    """The minimal gateway boundary needed by the open-loop generator."""

    async def call(self, request: GatewayRequest) -> GatewayResponse: ...


@dataclass(frozen=True)
class OfferedLevel:
    """One offered-load level for one API.

    ``offered_intensity_per_s`` is in native work units/s. ``target_completed`` is
    the minimum number of completed requests to collect before stopping the level
    (used for percentile estimation); the level also stops at ``max_offered`` items
    offered. ``ramp_up_s`` spreads the first few arrivals to avoid a cold-start
    burst that is not representative of steady state.
    """

    api_name: ModelApiName
    offered_intensity_per_s: float
    target_completed: int = 100
    max_offered: int = 400
    ramp_up_s: float = 0.0
    # Optional in-flight bound. When set and reached, the next offer is recorded as
    # backpressure-rejected rather than delayed. Default None == unbounded (true
    # open-loop); the generator never waits for a completion.
    max_in_flight: int | None = None


def _phase_timings_from_metadata(meta: dict) -> dict[str, float | None]:
    """Extract admission/queue/dispatch/forward/encoding ms from server metadata.

    The serving replica emits ``phase_timing`` in the result metadata dict. All
    values are monotonic milliseconds from the replica clock. Missing keys -> None.
    """
    pt = meta.get("phase_timing") or {}
    diagnostics = meta.get("batch_diagnostics") or {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    # Adapter result construction is not ASGI multipart/HTTP serialization.
    # Only an explicitly external phase_timing encoding span may populate this.
    return {
        "admission_ms": pt.get("admission_ms"),
        "queue_ms": pt.get("queue_ms"),
        "dispatch_ms": pt.get("dispatch_ms"),
        "forward_ms": pt.get("forward_ms"),
        "encoding_ms": pt.get("encoding_ms"),
    }


def _unidepth_diagnostics_from_metadata(meta: dict) -> dict[str, float | int | str | None]:
    """Extract server-only UniDepth experimental spans without inventing zeros."""
    raw = meta.get("batch_diagnostics") or {}
    if not isinstance(raw, dict):
        raw = {}
    def _number(name: str) -> float | None:
        value = raw.get(name)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None
    availability = raw.get("availability")
    runtime_config = raw.get("runtime_config")
    batch_policy = runtime_config.get("batch_policy") if isinstance(runtime_config, dict) else None
    return {
        "cpu_collate_ms": _number("cpu_collate_ms"),
        "h2d_ms": _number("h2d_ms"),
        "cuda_model_ms": _number("cuda_model_ms"),
        "d2h_ms": _number("d2h_ms"),
        "validation_ms": _number("validation_ms"),
        "diagnostics_availability": availability if isinstance(availability, str) else None,
        "runtime_config_digest": raw.get("runtime_config_digest") if isinstance(raw.get("runtime_config_digest"), str) else None,
        "batch_policy_max_batch_size": batch_policy.get("max_batch_size") if isinstance(batch_policy, dict) and isinstance(batch_policy.get("max_batch_size"), int) else None,
        "batch_policy_wait_ms": _number_from(batch_policy, "batch_wait_timeout_ms"),
        "max_concurrent_forwards": runtime_config.get("max_concurrent_forwards") if isinstance(runtime_config, dict) and isinstance(runtime_config.get("max_concurrent_forwards"), int) else None,
        "peak_simultaneous_forwards": raw.get("peak_simultaneous_forwards") if isinstance(raw.get("peak_simultaneous_forwards"), int) else None,
        "adapter_forward_wait_ms": _number("adapter_forward_wait_ms"),
    }


def _number_from(raw: object, name: str) -> float | None:
    value = raw.get(name) if isinstance(raw, dict) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _allocator_memory_from_metadata(meta: dict) -> dict[str, int | None]:
    """Read an optional allocator snapshot already emitted by the server.

    The benchmark never infers allocator state from NVML or client timing.  Exact
    production revisions may expose either concise or PyTorch-style names; malformed
    values remain absent so an unavailable field is distinguishable from zero.
    """
    diagnostics = meta.get("batch_diagnostics")
    diagnostic_allocator = diagnostics.get("allocator_memory") if isinstance(diagnostics, dict) else None
    raw = meta.get("allocator_memory") or meta.get("allocator") or diagnostic_allocator or {}
    if not isinstance(raw, dict):
        raw = {}

    def _int(*names: str) -> int | None:
        for name in names:
            value = raw.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return {
        "allocator_allocated_bytes": _int("allocated_bytes", "memory_allocated_bytes"),
        "allocator_reserved_bytes": _int("reserved_bytes", "memory_reserved_bytes"),
        "allocator_max_allocated_bytes": _int("max_allocated_bytes", "max_memory_allocated_bytes"),
        "allocator_max_reserved_bytes": _int("max_reserved_bytes", "max_memory_reserved_bytes"),
    }


def _record_from_response(
    item: PayloadItem,
    response: GatewayResponse,
    offer_time_s: float,
    submit_time_s: float,
    response_time_s: float,
) -> ItemRecord:
    error_code: ErrorCode | None = None
    if response.error is not None:
        error_code = response.error.code
    outcome = outcome_for_error_code(error_code)
    response_latency_ms: float | None = None
    if outcome == "completed":
        response_latency_ms = (response_time_s - offer_time_s) * 1000.0
    transport_ms = response.transport_ms if response.transport_ms else None
    phases = {"admission_ms": None, "queue_ms": None, "dispatch_ms": None, "forward_ms": None, "encoding_ms": None}
    batch_id: str | None = None
    batch_size: int | None = None
    batch_work_units: int | None = None
    batch_wall_ms: float | None = None
    amortized_cost_ms: float | None = None
    model_load_count: int | None = None
    trace_replica_id: str | None = None
    allocator_memory = {
        "allocator_allocated_bytes": None,
        "allocator_reserved_bytes": None,
        "allocator_max_allocated_bytes": None,
        "allocator_max_reserved_bytes": None,
    }
    batch_diagnostics: dict[str, Any] = {
        "cpu_collate_ms": None, "h2d_ms": None, "cuda_model_ms": None,
        "d2h_ms": None, "validation_ms": None, "diagnostics_availability": None,
        "runtime_config_digest": None, "batch_policy_max_batch_size": None,
        "batch_policy_wait_ms": None, "max_concurrent_forwards": None,
        "peak_simultaneous_forwards": None, "adapter_forward_wait_ms": None,
    }
    if response.result is not None:
        response_metadata = dict(response.result.metadata)
        phases = _phase_timings_from_metadata(response_metadata)
        allocator_memory = _allocator_memory_from_metadata(response_metadata)
        batch_diagnostics = _unidepth_diagnostics_from_metadata(response_metadata)
        trace = response.result.trace
        if trace is not None:
            trace_replica_id = trace.replica_id
            batch_id = trace.batch_id
            batch_size = trace.request_count
            batch_work_units = trace.effective_work_units
            # batch wall = forward window of the batch (completed - forward_started).
            batch_wall_ms = (trace.completed_monotonic_s - trace.forward_started_monotonic_s) * 1000.0
            if batch_size and batch_size > 0:
                amortized_cost_ms = batch_wall_ms / batch_size
            model_load_count = trace.model_load_count
    return ItemRecord(
        item_id=item.item_id,
        api_name=item.api_name.value,
        request_id=item.ownership.request_id,
        job_id=item.ownership.job_id,
        work_units=item.work_units,
        payload_hash=item.payload_hash,
        source_timestamp_s=item.source_timestamp_s,
        offer_time_s=offer_time_s,
        submit_time_s=submit_time_s,
        response_time_s=response_time_s,
        offered_delay_s=submit_time_s - offer_time_s,
        outcome=outcome,
        http_status=response.last_status_code,
        attempts=response.attempts,
        error_code=error_code.value if error_code else None,
        error_message=response.error.message if response.error else None,
        response_latency_ms=response_latency_ms,
        transport_ms=transport_ms,
        admission_ms=phases["admission_ms"],
        queue_ms=phases["queue_ms"],
        dispatch_ms=phases["dispatch_ms"],
        forward_ms=phases["forward_ms"],
        encoding_ms=phases["encoding_ms"],
        batch_id=batch_id,
        batch_size=batch_size,
        batch_work_units=batch_work_units,
        batch_wall_ms=batch_wall_ms,
        amortized_cost_ms=amortized_cost_ms,
        model_load_count=model_load_count,
        replica_id=response.replica_id or trace_replica_id,
        **allocator_memory,
        **batch_diagnostics,
    )


@dataclass
class LevelRunResult:
    """One level's configured rate, actual offer window, and response drain.

    ``offered_intensity_per_s`` is the configured scheduler target.  The actual
    offered rate is derived only from submissions in ``run_start_s`` through
    ``offer_window_end_s``; ``run_end_s`` is retained separately because it also
    includes the response drain needed to observe completed throughput.
    """

    api_name: ModelApiName
    offered_intensity_per_s: float
    records: list[ItemRecord] = field(default_factory=list)
    run_start_s: float = 0.0
    offer_window_end_s: float = 0.0
    run_end_s: float = 0.0
    measurement_interval_id: str | None = None
    expected_replica_ids: tuple[str, ...] = ()

    @property
    def configured_offered_intensity_per_s(self) -> float:
        return self.offered_intensity_per_s

    @property
    def offer_window_duration_s(self) -> float:
        return max(self.offer_window_end_s - self.run_start_s, 0.0)

    @property
    def actual_submission_span_s(self) -> float:
        return self.offer_window_duration_s

    @property
    def drain_duration_s(self) -> float:
        return max(self.run_end_s - self.offer_window_end_s, 0.0)

    @property
    def observation_duration_s(self) -> float:
        return max(self.run_end_s - self.run_start_s, 0.0)

    @property
    def actual_offered_rate_per_s(self) -> float | None:
        if self.actual_submission_span_s <= 0:
            return None
        return sum(record.work_units for record in self.records) / self.actual_submission_span_s

    @property
    def duration_s(self) -> float:
        """Compatibility alias for the completed/drain observation interval."""
        return self.observation_duration_s

    @property
    def started_at_s(self) -> float:
        return self.run_start_s

    @property
    def ended_at_s(self) -> float:
        return self.run_end_s


class OpenLoopGenerator:
    """Schedules and submits offers at fixed wall-clock times, independent of completions.

    ``gateway`` is the typed gateway. ``clock`` and ``sleep`` are injected so tests
    can run deterministically without real time; production uses ``time.monotonic``
    and ``asyncio.sleep``.
    """

    def __init__(
        self,
        gateway: AsyncGatewayCaller,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        record_sink: Callable[[ItemRecord], Awaitable[None]] | None = None,
    ) -> None:
        self._gateway = gateway
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._record_sink = record_sink

    def _schedule_offers(
        self, manifest: PayloadManifest, level: OfferedLevel
    ) -> list[tuple[float, PayloadItem]]:
        """Compute offer times from offered intensity, independent of completions.

        Offer times are wall-clock offsets from the run start. With work-units>1, the
        offered rate is in work units/s, so the per-item interval is
        ``work_units / intensity``. This keeps ``offered_intensity_per_s`` comparable
        across APIs with different native units.
        """
        items = manifest.items[: level.max_offered]
        if not items:
            return []
        schedule: list[tuple[float, PayloadItem]] = []
        t = 0.0
        intensity = level.offered_intensity_per_s
        if intensity <= 0:
            raise ValueError("offered_intensity_per_s must be positive")
        for index, item in enumerate(items):
            if index > 0 and level.ramp_up_s > 0:
                # Spread the first few arrivals linearly over ramp_up_s.
                ramp_fraction = min(index / max(1, len(items)), 1.0)
                t = level.ramp_up_s * ramp_fraction + (index - 1) * (item.work_units / intensity)
            else:
                t = index * (item.work_units / intensity)
            schedule.append((t, item))
        return schedule

    async def run_level(
        self,
        manifest: PayloadManifest,
        level: OfferedLevel,
        *,
        measurement_interval_id: str | None = None,
        expected_replica_ids: Sequence[str] = (),
    ) -> LevelRunResult:
        """Run one offered level. Returns per-item records and wall duration."""
        schedule = self._schedule_offers(manifest, level)
        expected_ids = tuple(expected_replica_ids)
        if len(set(expected_ids)) != len(expected_ids) or any(not replica_id for replica_id in expected_ids):
            raise ValueError("expected_replica_ids must contain unique non-empty identities")
        if not schedule:
            now = self._clock()
            return LevelRunResult(
                api_name=level.api_name, offered_intensity_per_s=level.offered_intensity_per_s,
                run_start_s=now, offer_window_end_s=now, run_end_s=now,
                measurement_interval_id=measurement_interval_id, expected_replica_ids=expected_ids,
            )
        start_mono = self._clock()
        result = LevelRunResult(
            api_name=level.api_name,
            offered_intensity_per_s=level.offered_intensity_per_s,
            run_start_s=start_mono,
            measurement_interval_id=measurement_interval_id,
            expected_replica_ids=expected_ids,
        )
        in_flight = 0
        in_flight_lock = asyncio.Lock()

        async def _handle(item: PayloadItem, offer_offset: float) -> None:
            nonlocal in_flight
            offer_time = start_mono + offer_offset
            # Sleep until the offer time. This is open-loop: we never wait for a
            # completion, only for the scheduled offer time.
            now = self._clock()
            delay = offer_time - now
            if delay > 0:
                await self._sleep(delay)
            submit_time = self._clock()
            # Optional in-flight bound: reject (do not delay) when full.
            if level.max_in_flight is not None:
                async with in_flight_lock:
                    if in_flight >= level.max_in_flight:
                        response_time = self._clock()
                        rec = ItemRecord(
                            item_id=item.item_id,
                            api_name=item.api_name.value,
                            request_id=item.ownership.request_id,
                            job_id=item.ownership.job_id,
                            work_units=item.work_units,
                            payload_hash=item.payload_hash,
                            source_timestamp_s=item.source_timestamp_s,
                            offer_time_s=offer_time,
                            submit_time_s=submit_time,
                            response_time_s=response_time,
                            offered_delay_s=submit_time - offer_time,
                            outcome="backpressure",
                            http_status=None,
                            attempts=0,
                            error_code=ErrorCode.BACKPRESSURE.value,
                            error_message="in-flight bound reached at offer time",
                            response_latency_ms=None,
                            transport_ms=None,
                            admission_ms=None,
                            queue_ms=None,
                            dispatch_ms=None,
                            forward_ms=None,
                            encoding_ms=None,
                            batch_id=None,
                            batch_size=None,
                            batch_work_units=None,
                            batch_wall_ms=None,
                            amortized_cost_ms=None,
                            model_load_count=None,
                        )
                        result.records.append(rec)
                        if self._record_sink is not None:
                            await self._record_sink(rec)
                        return
                    in_flight += 1
            try:
                response = await self._gateway.call(item.to_gateway_request())
            finally:
                if level.max_in_flight is not None:
                    async with in_flight_lock:
                        in_flight = max(0, in_flight - 1)
            response_time = self._clock()
            rec = _record_from_response(item, response, offer_time, submit_time, response_time)
            result.records.append(rec)
            if self._record_sink is not None:
                await self._record_sink(rec)

        tasks = [asyncio.create_task(_handle(item, off)) for off, item in schedule]
        await asyncio.gather(*tasks)
        # The offer window ends at the final actual submit, not at final response.
        # Thus queueing or delayed responses can lengthen drain/observation without
        # rewriting the load that the generator actually offered.
        result.offer_window_end_s = max((record.submit_time_s for record in result.records), default=start_mono)
        result.run_end_s = self._clock()
        return result

    async def run_levels(
        self,
        manifest: PayloadManifest,
        levels: Sequence[OfferedLevel],
    ) -> list[LevelRunResult]:
        results: list[LevelRunResult] = []
        for level in levels:
            if level.api_name != manifest.api_name:
                raise ValueError(
                    f"level api {level.api_name} does not match manifest api {manifest.api_name}"
                )
            results.append(await self.run_level(manifest, level))
        return results
