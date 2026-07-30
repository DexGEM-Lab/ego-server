"""Per-item records and aggregate metrics for open-loop offered-load benchmarks.

Definitions (open-loop):

* **offered** rate — arrivals actually submitted per second, independent of
  completions. The generator fires at scheduled times regardless of in-flight count.
* **admitted** rate — requests admitted into serving (passed pre-batch admission),
  i.e. completed + processing failures (model/result-split errors). These entered a
  batch and consumed GPU work.
* **completed** rate — admitted requests that returned a successful typed result.
* **rejected** rate — requests rejected before/during admission (backpressure,
  transport exhaustion, validation) and never processed.
* **response latency** — client end-to-end (offer -> response), per item. Reported
  as p50/p95/p99. This is *not* amortized cost.
* **amortized cost** — server batch wall time / batch size, per item. Reported
  separately so request latency is never conflated with per-item compute cost.
* **phase decomposition** — admission/queue/dispatch/forward/encoding ms, sourced
  from the serving replica's emitted ``phase_timing`` in the result metadata.
* **throughput** — completed native work units / second (completed * work_units).
* **batch-size/weight distribution** — per-batch request_count and work_units,
  sourced from the server batch trace.
* **model-load count** — resident model load count from the batch trace; must stay
  constant across the run (one load at replica init, not per request).
* **payload hashes** — content hash per item, proving distinct payloads.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ego_annotation.serving.contracts import ErrorCode


# Outcome categories for the four-rate accounting.
OUTCOME_COMPLETED = "completed"
OUTCOME_MODEL_FAILURE = "model_failure"        # admitted, failed in processing
OUTCOME_RESULT_SPLIT = "result_split_failure"  # admitted, failed at split
OUTCOME_BACKPRESSURE = "backpressure"          # rejected at admission
OUTCOME_TRANSPORT = "transport"                # rejected (transport exhausted)
OUTCOME_VALIDATION = "validation"              # rejected (caller error)
OUTCOME_IN_FLIGHT = "in_flight"                # never settled before close

ADMITTED_OUTCOMES = frozenset({OUTCOME_COMPLETED, OUTCOME_MODEL_FAILURE, OUTCOME_RESULT_SPLIT})
REJECTED_OUTCOMES = frozenset({OUTCOME_BACKPRESSURE, OUTCOME_TRANSPORT, OUTCOME_VALIDATION})


def outcome_for_error_code(code: ErrorCode | None) -> str:
    if code is None:
        return OUTCOME_COMPLETED
    if code is ErrorCode.BACKPRESSURE:
        return OUTCOME_BACKPRESSURE
    if code is ErrorCode.TRANSPORT:
        return OUTCOME_TRANSPORT
    if code is ErrorCode.VALIDATION:
        return OUTCOME_VALIDATION
    if code is ErrorCode.MODEL_FAILURE:
        return OUTCOME_MODEL_FAILURE
    if code is ErrorCode.RESULT_SPLIT_FAILURE:
        return OUTCOME_RESULT_SPLIT
    return OUTCOME_TRANSPORT


@dataclass
class ItemRecord:
    """One item's full benchmark trace, written verbatim to JSONL."""

    item_id: str
    api_name: str
    request_id: str
    job_id: str
    work_units: int
    payload_hash: str
    source_timestamp_s: float

    # Timing (seconds, client wall clock unless noted).
    offer_time_s: float          # scheduled arrival wall time
    submit_time_s: float         # actual send wall time
    response_time_s: float       # response received wall time
    offered_delay_s: float       # submit - offer (scheduler jitter)

    # Outcome + server-side decomposition.
    outcome: str
    http_status: int | None
    attempts: int
    error_code: str | None
    error_message: str | None

    response_latency_ms: float | None        # client end-to-end
    transport_ms: float | None               # client transport total across attempts
    # Server-side phase decomposition (ms), from result.metadata.phase_timing.
    admission_ms: float | None
    queue_ms: float | None
    dispatch_ms: float | None
    forward_ms: float | None
    encoding_ms: float | None
    # Batch info from the server batch trace.
    batch_id: str | None
    batch_size: int | None
    batch_work_units: int | None
    batch_wall_ms: float | None              # amortized cost numerator
    amortized_cost_ms: float | None          # batch_wall_ms / batch_size (per-item)
    model_load_count: int | None
    # Explicit endpoint identity for isolated scaling experiments.  Canonical
    # single-endpoint benchmarks retain ``None`` and are not re-routed.
    replica_id: str | None = None
    # Optional fields emitted by a server allocator snapshot.  They are absent
    # unless the exact serving revision exposes them; no client estimate is used.
    allocator_allocated_bytes: int | None = None
    allocator_reserved_bytes: int | None = None
    allocator_max_allocated_bytes: int | None = None
    allocator_max_reserved_bytes: int | None = None
    # Experiment-only UniDepth batch diagnostics.  CUDA spans are null on CPU and
    # when telemetry is disabled; null is evidence of unavailability, never zero.
    cpu_collate_ms: float | None = None
    h2d_ms: float | None = None
    cuda_model_ms: float | None = None
    d2h_ms: float | None = None
    validation_ms: float | None = None
    diagnostics_availability: str | None = None
    # Worker-derived experiment policy and actual overlap state.  These are never
    # client-side estimates and remain null for non-instrumented production lanes.
    runtime_config_digest: str | None = None
    batch_policy_max_batch_size: int | None = None
    batch_policy_wait_ms: float | None = None
    max_concurrent_forwards: int | None = None
    peak_simultaneous_forwards: int | None = None
    adapter_forward_wait_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "api_name": self.api_name,
            "request_id": self.request_id,
            "job_id": self.job_id,
            "work_units": self.work_units,
            "payload_hash": self.payload_hash,
            "source_timestamp_s": self.source_timestamp_s,
            "offer_time_s": self.offer_time_s,
            "submit_time_s": self.submit_time_s,
            "response_time_s": self.response_time_s,
            "offered_delay_s": self.offered_delay_s,
            "outcome": self.outcome,
            "http_status": self.http_status,
            "attempts": self.attempts,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "response_latency_ms": self.response_latency_ms,
            "transport_ms": self.transport_ms,
            "admission_ms": self.admission_ms,
            "queue_ms": self.queue_ms,
            "dispatch_ms": self.dispatch_ms,
            "forward_ms": self.forward_ms,
            "encoding_ms": self.encoding_ms,
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "batch_work_units": self.batch_work_units,
            "batch_wall_ms": self.batch_wall_ms,
            "amortized_cost_ms": self.amortized_cost_ms,
            "model_load_count": self.model_load_count,
            "replica_id": self.replica_id,
            "allocator_allocated_bytes": self.allocator_allocated_bytes,
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "allocator_max_allocated_bytes": self.allocator_max_allocated_bytes,
            "allocator_max_reserved_bytes": self.allocator_max_reserved_bytes,
            "cpu_collate_ms": self.cpu_collate_ms,
            "h2d_ms": self.h2d_ms,
            "cuda_model_ms": self.cuda_model_ms,
            "d2h_ms": self.d2h_ms,
            "validation_ms": self.validation_ms,
            "diagnostics_availability": self.diagnostics_availability,
            "runtime_config_digest": self.runtime_config_digest,
            "batch_policy_max_batch_size": self.batch_policy_max_batch_size,
            "batch_policy_wait_ms": self.batch_policy_wait_ms,
            "max_concurrent_forwards": self.max_concurrent_forwards,
            "peak_simultaneous_forwards": self.peak_simultaneous_forwards,
            "adapter_forward_wait_ms": self.adapter_forward_wait_ms,
        }


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * pct
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


@dataclass
class LevelSummary:
    """Aggregate metrics for one (api, offered-intensity) level."""

    api_name: str
    offered_intensity_per_s: float            # configured offered rate
    duration_s: float                         # completed/drain observation interval
    offer_duration_s: float                   # actual submission window; never response drain
    offered_count: int
    admitted_count: int
    completed_count: int
    rejected_count: int
    in_flight_count: int
    offered_rate_per_s: float                 # offered_count / duration
    admitted_rate_per_s: float
    completed_rate_per_s: float
    rejected_rate_per_s: float
    throughput_work_units_per_s: float        # sum(work_units of completed) / duration
    # Response latency percentiles (ms) — client end-to-end, NOT amortized cost.
    response_latency_p50_ms: float | None
    response_latency_p95_ms: float | None
    response_latency_p99_ms: float | None
    response_latency_mean_ms: float | None
    # Amortized per-item compute cost (ms) — kept separate from latency.
    amortized_cost_p50_ms: float | None
    amortized_cost_p95_ms: float | None
    amortized_cost_mean_ms: float | None
    # Phase decomposition means (ms), completed items only.
    admission_ms_mean: float | None
    queue_ms_mean: float | None
    dispatch_ms_mean: float | None
    forward_ms_mean: float | None
    encoding_ms_mean: float | None
    transport_ms_mean: float | None
    # Batch-size / weight distribution.
    batch_size_p50: float | None
    batch_size_p95: float | None
    batch_size_mean: float | None
    batch_work_units_mean: float | None
    distinct_batch_count: int
    # Model load count (must be constant across the run).
    model_load_count_min: int | None
    model_load_count_max: int | None
    # Distinct payload hashes (proves distinct payloads).
    distinct_payload_hashes: int
    # Rejection breakdown.
    backpressure_count: int
    transport_reject_count: int
    validation_count: int
    model_failure_count: int
    result_split_count: int

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def csv_columns() -> tuple[str, ...]:
        return (
            "api_name", "offered_intensity_per_s", "duration_s", "offer_duration_s", "offered_count", "admitted_count",
            "completed_count", "rejected_count", "in_flight_count", "offered_rate_per_s",
            "admitted_rate_per_s", "completed_rate_per_s", "rejected_rate_per_s",
            "throughput_work_units_per_s", "response_latency_p50_ms", "response_latency_p95_ms",
            "response_latency_p99_ms", "response_latency_mean_ms", "amortized_cost_p50_ms",
            "amortized_cost_p95_ms", "amortized_cost_mean_ms", "admission_ms_mean", "queue_ms_mean",
            "dispatch_ms_mean", "forward_ms_mean", "encoding_ms_mean", "transport_ms_mean",
            "batch_size_p50", "batch_size_p95", "batch_size_mean", "batch_work_units_mean",
            "distinct_batch_count", "model_load_count_min", "model_load_count_max",
            "distinct_payload_hashes", "backpressure_count", "transport_reject_count",
            "validation_count", "model_failure_count", "result_split_count",
        )


def summarize(
    records: Sequence[ItemRecord],
    *,
    api_name: str,
    offered_intensity_per_s: float,
    duration_s: float,
    offer_duration_s: float | None = None,
) -> LevelSummary:
    offered = len(records)
    offered_work_units = sum(r.work_units for r in records)
    admitted = sum(1 for r in records if r.outcome in ADMITTED_OUTCOMES)
    completed = sum(1 for r in records if r.outcome == OUTCOME_COMPLETED)
    rejected = sum(1 for r in records if r.outcome in REJECTED_OUTCOMES)
    in_flight = sum(1 for r in records if r.outcome == OUTCOME_IN_FLIGHT)

    # Completion/admission outcomes settle over the observation interval.  Offers
    # are instead normalized by their own submission window so a slow response
    # drain cannot make open-loop offered load appear smaller.
    dur = duration_s if duration_s > 0 else 1.0
    offer_dur = offer_duration_s if offer_duration_s is not None and offer_duration_s > 0 else None
    latencies = [r.response_latency_ms for r in records if r.response_latency_ms is not None]
    amortized = [r.amortized_cost_ms for r in records if r.amortized_cost_ms is not None]

    def _mean(values: Iterable[float]) -> float | None:
        vals = list(values)
        return statistics.fmean(vals) if vals else None

    completed_records = [r for r in records if r.outcome == OUTCOME_COMPLETED]
    batch_sizes: list[float] = []
    batch_works: list[float] = []
    batch_ids: set[str] = set()
    for r in completed_records:
        if r.batch_size is not None:
            batch_sizes.append(float(r.batch_size))
        if r.batch_work_units is not None:
            batch_works.append(float(r.batch_work_units))
        if r.batch_id is not None:
            batch_ids.add(r.batch_id)

    model_loads = [r.model_load_count for r in completed_records if r.model_load_count is not None]
    throughput = sum(r.work_units for r in completed_records) / dur

    return LevelSummary(
        api_name=api_name,
        offered_intensity_per_s=offered_intensity_per_s,
        duration_s=duration_s,
        offer_duration_s=offer_duration_s if offer_duration_s is not None else duration_s,
        offered_count=offered,
        admitted_count=admitted,
        completed_count=completed,
        rejected_count=rejected,
        in_flight_count=in_flight,
        offered_rate_per_s=(offered_work_units / offer_dur) if offer_dur is not None else 0.0,
        admitted_rate_per_s=admitted / dur,
        completed_rate_per_s=completed / dur,
        rejected_rate_per_s=rejected / dur,
        throughput_work_units_per_s=throughput,
        response_latency_p50_ms=_percentile(latencies, 0.50),
        response_latency_p95_ms=_percentile(latencies, 0.95),
        response_latency_p99_ms=_percentile(latencies, 0.99),
        response_latency_mean_ms=_mean(latencies),
        amortized_cost_p50_ms=_percentile(amortized, 0.50),
        amortized_cost_p95_ms=_percentile(amortized, 0.95),
        amortized_cost_mean_ms=_mean(amortized),
        admission_ms_mean=_mean(r.admission_ms for r in completed_records if r.admission_ms is not None),
        queue_ms_mean=_mean(r.queue_ms for r in completed_records if r.queue_ms is not None),
        dispatch_ms_mean=_mean(r.dispatch_ms for r in completed_records if r.dispatch_ms is not None),
        forward_ms_mean=_mean(r.forward_ms for r in completed_records if r.forward_ms is not None),
        encoding_ms_mean=_mean(r.encoding_ms for r in completed_records if r.encoding_ms is not None),
        transport_ms_mean=_mean(r.transport_ms for r in records if r.transport_ms is not None),
        batch_size_p50=_percentile(batch_sizes, 0.50),
        batch_size_p95=_percentile(batch_sizes, 0.95),
        batch_size_mean=_mean(batch_sizes),
        batch_work_units_mean=_mean(batch_works),
        distinct_batch_count=len(batch_ids),
        model_load_count_min=min(model_loads) if model_loads else None,
        model_load_count_max=max(model_loads) if model_loads else None,
        distinct_payload_hashes=len({r.payload_hash for r in records}),
        backpressure_count=sum(1 for r in records if r.outcome == OUTCOME_BACKPRESSURE),
        transport_reject_count=sum(1 for r in records if r.outcome == OUTCOME_TRANSPORT),
        validation_count=sum(1 for r in records if r.outcome == OUTCOME_VALIDATION),
        model_failure_count=sum(1 for r in records if r.outcome == OUTCOME_MODEL_FAILURE),
        result_split_count=sum(1 for r in records if r.outcome == OUTCOME_RESULT_SPLIT),
    )
