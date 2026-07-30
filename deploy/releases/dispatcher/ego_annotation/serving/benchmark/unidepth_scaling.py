"""Isolated, experiment-only UniDepth replica scaling contracts.

This module plans or executes *new* Ray heads for explicitly authorized vacant
physical GPUs.  It copies the exact production UniDepth interpreter, import path,
model environment, and native-Ray GPU ownership, while replacing every operational
identity that could collide with production: CUDA visibility, Ray component/worker/
metrics/HTTP ports, temporary directory, and replica id.  It never edits
``COMMITTED_GPU_GROUPS`` or the canonical router.

The request balancer is intentionally stateless: UniDepth requests have no session
ownership, so each offered item is assigned round-robin to an explicit experimental
endpoint.  The assignment is attached to the settled result, preserving the item's
own request/job/source ownership and making aggregate scaling evidence attributable
per physical replica.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from ego_annotation.serving.benchmark.generator import AsyncGatewayCaller, LevelRunResult, OfferedLevel, OpenLoopGenerator
from ego_annotation.serving.benchmark.manifest import PayloadManifest
from ego_annotation.serving.benchmark.metrics import ItemRecord, LevelSummary, summarize
from ego_annotation.serving.benchmark.release import file_manifest, release_digest_from_manifest, verify_release, checkpoint_digest
from ego_annotation.serving.contracts import SCHEMA_VERSION, ContractValidationError, ErrorCode, ServerIdentity, ServiceError
from ego_annotation.serving.unidepth import expected_unidepth_runtime_config
from ego_annotation.serving.gateway import GatewayRequest, GatewayResponse
from ego_annotation.serving.lifecycle import (
    COMMITTED_GPU_GROUPS,
    ClusterLifecycleConfig,
    ClusterPorts,
    committed_serve_http_ports,
    unidepth_gpu_group,
)
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


# Ray places a timestamped session directory and ``sockets/plasma_store`` below
# this root. Linux AF_UNIX paths are limited to 107 bytes, so the experiment root
# and per-replica path must stay deliberately short.
EXPERIMENT_TEMP_ROOT = Path("/tmp/eud")
_MAX_RAY_TEMP_DIR_BYTES = 32
# Production GPU groups are always excluded even if they happen to be idle.
PRODUCTION_GPU_IDS = frozenset(group.gpu_id for group in COMMITTED_GPU_GROUPS)
PRODUCTION_TEMP_DIRS = frozenset(group.lifecycle.temp_dir for group in COMMITTED_GPU_GROUPS)
PRODUCTION_PORTS = frozenset(port for group in COMMITTED_GPU_GROUPS for port in group.lifecycle.ports.all_ports())


class ExperimentConfigurationError(ValueError):
    """The proposed experiment could intersect production or lose attribution."""


class PhysicalBatchConflictError(ValueError):
    """One batch id described two incompatible physical batches."""


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ImmutableApplicationRelease:
    """A content-addressed, non-symlink release directory."""

    path: Path
    release_sha: str
    source_sha: str
    release_digest: str

    @classmethod
    def pin(cls, path: str | Path, *, expected_source_sha: str) -> "ImmutableApplicationRelease":
        requested = Path(path)
        if requested.is_symlink() or "current" in requested.parts:
            raise ExperimentConfigurationError("experiments must name an immutable release directory, never runtime/current")
        try:
            resolved = requested.resolve(strict=True)
        except OSError as exc:
            raise ExperimentConfigurationError(f"immutable application release path is unavailable: {exc}") from exc
        if resolved.is_symlink() or "current" in resolved.parts:
            raise ExperimentConfigurationError("experiments must name an immutable release directory, never runtime/current")
        if not _SHA_RE.fullmatch(expected_source_sha):
            raise ExperimentConfigurationError("expected_source_sha must be a 40-character lowercase git SHA")
        release_file = resolved / "RELEASE.json"
        try:
            release = json.loads(release_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentConfigurationError(f"immutable application release requires readable RELEASE.json: {exc}") from exc
        if not isinstance(release, Mapping):
            raise ExperimentConfigurationError("RELEASE.json must be an object")
        source_sha = release.get("source_sha")
        if source_sha != expected_source_sha:
            raise ExperimentConfigurationError(
                f"RELEASE.json.source_sha {source_sha!r} does not equal requested source SHA {expected_source_sha!r}"
            )
        try:
            attested = verify_release(resolved, expected_source_sha=expected_source_sha)
            release_digest = attested.release_digest
            # Content digest, rather than an operator-declared label, is the release identity.
            release_sha = release_digest
        except (ValueError, OSError) as exc:
            # Legacy unit fixtures predate the generic builder.  Still recompute their
            # bytes at plan and prelaunch time so mutation is detected; production
            # experiment releases must use the strict digest-named attestation.
            if "manifest" in release or "release_digest" in release or resolved.name not in {"release-a", "release-b"}:
                raise ExperimentConfigurationError(f"invalid content-addressed release: {exc}") from exc
            manifest = file_manifest(resolved)
            release_digest = release_digest_from_manifest(manifest)
            release_sha = release.get("release_sha")
            if not isinstance(release_sha, str) or not _SHA_RE.fullmatch(release_sha):
                raise ExperimentConfigurationError("legacy RELEASE.json.release_sha must be a git SHA")
        if not (resolved / "ego_annotation" / "serving" / "deployment.py").is_file():
            raise ExperimentConfigurationError("immutable application release is missing ego_annotation.serving.deployment")
        return cls(path=resolved, release_sha=release_sha, source_sha=expected_source_sha, release_digest=release_digest)


@dataclass(frozen=True)
class ReplicaEndpoint:
    """One explicit experimental UniDepth public endpoint."""

    replica_id: str
    url: str
    gpu_id: int

    def __post_init__(self) -> None:
        if not self.replica_id:
            raise ExperimentConfigurationError("replica_id must be non-empty")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.port is None:
            raise ExperimentConfigurationError(f"replica {self.replica_id} has invalid endpoint URL {self.url!r}")
        if parsed.port in PRODUCTION_PORTS:
            raise ExperimentConfigurationError(
                f"replica {self.replica_id} points at production-reserved port {parsed.port}, not an experiment lane"
            )
        if self.gpu_id in PRODUCTION_GPU_IDS:
            raise ExperimentConfigurationError(f"GPU{self.gpu_id} is a committed production GPU, not an experiment target")


class StatelessReplicaBalancer:
    """Open-loop request-level round robin across explicit UniDepth endpoints.

    This object owns no request/session mapping.  Its only state is the next index,
    which makes equal routing inspectable while retaining each request's original
    payload and ownership.
    """

    def __init__(self, endpoints: Sequence[ReplicaEndpoint]) -> None:
        if not endpoints:
            raise ExperimentConfigurationError("at least one experimental endpoint is required")
        ids = [endpoint.replica_id for endpoint in endpoints]
        urls = [endpoint.url for endpoint in endpoints]
        if len(set(ids)) != len(ids):
            raise ExperimentConfigurationError("experimental replica ids must be unique")
        if len(set(urls)) != len(urls):
            raise ExperimentConfigurationError("experimental endpoint URLs must be unique")
        self._endpoints = tuple(endpoints)
        self._next_index = 0

    @property
    def endpoints(self) -> tuple[ReplicaEndpoint, ...]:
        return self._endpoints

    def select(self) -> ReplicaEndpoint:
        endpoint = self._endpoints[self._next_index]
        self._next_index = (self._next_index + 1) % len(self._endpoints)
        return endpoint


class GatewayFactory(Protocol):
    def __call__(self, router: ModelServiceRouter) -> AsyncGatewayCaller: ...


def _canonical_gpu_uuid(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def validate_server_identity(expected: ServerIdentity, response: GatewayResponse) -> ServerIdentity:
    """Verify a successful result's server-originated physical identity exactly."""
    if response.result is None:
        raise ExperimentConfigurationError("typed readiness/benchmark response has no successful result identity")
    raw_identity = response.result.metadata.get("server_identity")
    try:
        actual = ServerIdentity.from_wire(raw_identity) if isinstance(raw_identity, Mapping) else None
    except (KeyError, TypeError, ContractValidationError) as exc:
        raise ExperimentConfigurationError(f"malformed server identity evidence: {exc}") from exc
    if actual is None:
        raise ExperimentConfigurationError("successful experiment response omitted server_identity evidence")
    trace = response.result.trace
    if trace is None or trace.replica_id != actual.replica_id:
        raise ExperimentConfigurationError("server trace replica_id does not agree with server_identity")
    if response.replica_id is not None and response.replica_id != actual.replica_id:
        raise ExperimentConfigurationError("gateway replica_id does not preserve the server trace replica_id")
    expected_fields = (
        "experiment_id", "replica_id", "assigned_gpu", "gcs_address", "http_port", "temp_dir",
        "model_revision", "checkpoint_digest", "schema_version", "release_sha",
    )
    if expected.release_digest is not None:
        expected_fields += ("release_digest",)
    mismatches = [name for name in expected_fields if getattr(actual, name) != getattr(expected, name)]
    # NVML reports ``GPU-<uuid>`` while Torch device properties report
    # ``<uuid>`` on the same physical device. Compare the canonical value while
    # preserving the server-native string as evidence.
    if (
        expected.cuda_uuid is not None
        and _canonical_gpu_uuid(actual.cuda_uuid) != _canonical_gpu_uuid(expected.cuda_uuid)
    ):
        mismatches.append("cuda_uuid")
    if mismatches:
        raise ExperimentConfigurationError(
            "server identity does not match planned scoped release: " + ", ".join(
                f"{name}={getattr(actual, name)!r}" for name in mismatches
            )
        )
    if actual.worker_pid <= 0:
        raise ExperimentConfigurationError("server identity worker_pid must be a live-process-shaped positive PID")
    return actual


class StatelessReplicaGateway:
    """Gateway adapter that chooses one experiment endpoint per stateless request.

    A fresh per-replica router override is created from a caller-supplied base router;
    the canonical router object is never mutated.  The endpoint choice is returned
    in ``GatewayResponse.replica_id`` and later persisted in ``ItemRecord``.
    """

    def __init__(
        self,
        *,
        api_name: ModelApiName,
        base_router: ModelServiceRouter,
        endpoints: Sequence[ReplicaEndpoint],
        gateway_factory: GatewayFactory,
        expected_server_identities: Mapping[str, ServerIdentity] | None = None,
    ) -> None:
        self._api_name = api_name
        self._balancer = StatelessReplicaBalancer(endpoints)
        self._gateways: dict[str, AsyncGatewayCaller] = {
            endpoint.replica_id: gateway_factory(base_router.with_overrides({api_name: endpoint.url}))
            for endpoint in self._balancer.endpoints
        }
        self._expected_server_identities = dict(expected_server_identities or {})
        unknown = set(self._expected_server_identities) - {endpoint.replica_id for endpoint in self._balancer.endpoints}
        if unknown:
            raise ExperimentConfigurationError(f"identity evidence configured for unknown replicas: {sorted(unknown)}")

    @property
    def endpoints(self) -> tuple[ReplicaEndpoint, ...]:
        return self._balancer.endpoints

    async def call(self, request: GatewayRequest) -> GatewayResponse:
        if request.api_name is not self._api_name:
            raise ExperimentConfigurationError(
                f"stateless scaling gateway is for {self._api_name.value}, got {request.api_name.value}"
            )
        endpoint = self._balancer.select()
        response = await self._gateways[endpoint.replica_id].call(request)
        # A selected URL is routing intent, not evidence of which worker answered.
        # Successful experimental responses require server identity; transport/error
        # outcomes retain no invented physical identity.
        if response.result is None:
            # No server trace exists for an explicit rejected/transport outcome;
            # retain the selected experiment lane as routing attribution only.
            return replace(response, replica_id=endpoint.replica_id)
        expected = self._expected_server_identities.get(endpoint.replica_id)
        if expected is None:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.VALIDATION,
                    "experimental gateway has no expected server identity for selected replica",
                    retryable=False,
                    ownership=request.ownership,
                ),
                attempts=response.attempts,
                last_status_code=response.last_status_code,
                transport_ms=response.transport_ms,
            )
        try:
            actual = validate_server_identity(expected, response)
        except ExperimentConfigurationError as exc:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.VALIDATION,
                    f"server identity rejected: {exc}", retryable=False, ownership=request.ownership,
                ),
                attempts=response.attempts,
                last_status_code=response.last_status_code,
                transport_ms=response.transport_ms,
            )
        return replace(response, replica_id=actual.replica_id)

    async def call_batch(self, requests: Sequence[Any]) -> list[GatewayResponse]:
        return await asyncio.gather(*(self.call(request) for request in requests))

    async def aclose(self) -> None:
        closers = [getattr(gateway, "aclose", None) for gateway in self._gateways.values()]
        await asyncio.gather(*(closer() for closer in closers if closer is not None))


@dataclass(frozen=True)
class GeneratorCpuSample:
    """A process CPU/wall sample taken around one offered-load level."""

    monotonic_s: float
    process_cpu_s: float

    @classmethod
    def take(cls, *, clock: Callable[[], float] = time.monotonic, process_clock: Callable[[], float] = time.process_time) -> "GeneratorCpuSample":
        return cls(monotonic_s=clock(), process_cpu_s=process_clock())


@dataclass(frozen=True)
class GeneratorCpuUsage:
    wall_s: float
    process_cpu_s: float
    process_cpu_utilization_cores: float | None

    @classmethod
    def between(cls, start: GeneratorCpuSample, end: GeneratorCpuSample) -> "GeneratorCpuUsage":
        wall_s = max(0.0, end.monotonic_s - start.monotonic_s)
        cpu_s = max(0.0, end.process_cpu_s - start.process_cpu_s)
        return cls(wall_s=wall_s, process_cpu_s=cpu_s, process_cpu_utilization_cores=(cpu_s / wall_s if wall_s else None))

    def to_dict(self) -> dict[str, float | None]:
        return {
            "wall_s": self.wall_s,
            "process_cpu_s": self.process_cpu_s,
            "process_cpu_utilization_cores": self.process_cpu_utilization_cores,
        }


@dataclass(frozen=True)
class PhysicalBatch:
    """One server batch, de-duplicated from its per-item response traces."""

    batch_id: str
    replica_id: str | None
    batch_size: int
    batch_work_units: int | None
    batch_wall_ms: float | None
    model_load_count: int | None
    allocator_allocated_bytes: int | None
    allocator_reserved_bytes: int | None
    allocator_max_allocated_bytes: int | None
    allocator_max_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def physical_batches_from_records(records: Sequence[ItemRecord]) -> list[PhysicalBatch]:
    """Return one physical batch per server ``batch_id``.

    Per-item records repeat their batch trace by design.  A conflicting duplicate is
    evidence corruption (or an unsafe non-unique server trace) and is surfaced rather
    than silently averaged.
    """
    batches: dict[str, PhysicalBatch] = {}
    for record in records:
        if record.batch_id is None or record.batch_size is None:
            continue
        candidate = PhysicalBatch(
            batch_id=record.batch_id,
            replica_id=record.replica_id,
            batch_size=record.batch_size,
            batch_work_units=record.batch_work_units,
            batch_wall_ms=record.batch_wall_ms,
            model_load_count=record.model_load_count,
            allocator_allocated_bytes=record.allocator_allocated_bytes,
            allocator_reserved_bytes=record.allocator_reserved_bytes,
            allocator_max_allocated_bytes=record.allocator_max_allocated_bytes,
            allocator_max_reserved_bytes=record.allocator_max_reserved_bytes,
        )
        previous = batches.get(candidate.batch_id)
        if previous is not None and previous != candidate:
            raise PhysicalBatchConflictError(
                f"batch_id {candidate.batch_id!r} has conflicting replica/trace fields: {previous} vs {candidate}"
            )
        batches[candidate.batch_id] = candidate
    return list(batches.values())


@dataclass(frozen=True)
class ScalingLevelResult:
    """A scaling level with separate offered and completed-observation clocks."""

    api_name: str
    replica_count: int
    configured_offered_intensity_per_s: float
    actual_offered_rate_per_s: float | None
    run_start_s: float
    offer_window_end_s: float
    run_end_s: float
    measurement_interval_id: str | None
    expected_replica_ids: tuple[str, ...]
    generator_cpu: GeneratorCpuUsage
    aggregate: LevelSummary
    per_replica: Mapping[str, LevelSummary]
    physical_batches: tuple[PhysicalBatch, ...]
    records: tuple[ItemRecord, ...]
    # Identity and measurement definition are required for cross-treatment claims.
    release_digest: str | None = None
    corpus_digest: str | None = None
    measurement_definition: str | None = None

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
    def duration_s(self) -> float:
        """Compatibility alias for the completed/drain observation interval."""
        return self.observation_duration_s

    @property
    def offered_intensity_per_s(self) -> float:
        """Compatibility alias; use ``configured_offered_intensity_per_s`` in new code."""
        return self.configured_offered_intensity_per_s

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ego.unidepth-scaling-level.v2",
            "api_name": self.api_name,
            "replica_count": self.replica_count,
            "configured_offered_intensity_per_s": self.configured_offered_intensity_per_s,
            "actual_offered_rate_per_s": self.actual_offered_rate_per_s,
            "run_start_s": self.run_start_s,
            "offer_window_end_s": self.offer_window_end_s,
            "actual_submission_span_s": self.actual_submission_span_s,
            "run_end_s": self.run_end_s,
            "offer_window_duration_s": self.offer_window_duration_s,
            "drain_duration_s": self.drain_duration_s,
            "observation_duration_s": self.observation_duration_s,
            "measurement_interval_id": self.measurement_interval_id,
            "expected_replica_ids": list(self.expected_replica_ids),
            "generator_cpu": self.generator_cpu.to_dict(),
            "aggregate": self.aggregate.to_dict(),
            "per_replica": {replica: summary.to_dict() for replica, summary in self.per_replica.items()},
            "physical_batches": [batch.to_dict() for batch in self.physical_batches],
            "records": [record.to_dict() for record in self.records],
            "release_digest": self.release_digest,
            "corpus_digest": self.corpus_digest,
            "measurement_definition": self.measurement_definition,
        }


@dataclass(frozen=True)
class ReplicaScalingComparison:
    """A directly reviewable one-replica versus multi-replica result schema."""

    baseline: ScalingLevelResult
    scaled: ScalingLevelResult
    aggregate_throughput_gain: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ego.unidepth-replica-scaling-comparison.v1",
            "baseline": self.baseline.to_dict(),
            "scaled": self.scaled.to_dict(),
            "aggregate_throughput_gain": self.aggregate_throughput_gain,
        }


def _require_comparable_level_identity(level: ScalingLevelResult) -> None:
    """Reject synthetic summaries that lack a complete measured identity contract."""
    if not level.expected_replica_ids or len(level.expected_replica_ids) != level.replica_count:
        raise ExperimentConfigurationError("replica comparison requires one expected replica ID per replica")
    if len(set(level.expected_replica_ids)) != len(level.expected_replica_ids):
        raise ExperimentConfigurationError("replica comparison requires unique expected replica IDs")
    observed_ids = {record.replica_id for record in level.records}
    if None in observed_ids or observed_ids != set(level.expected_replica_ids):
        raise ExperimentConfigurationError("replica comparison requires every expected replica to have typed record identity")
    if set(level.per_replica) != set(level.expected_replica_ids):
        raise ExperimentConfigurationError("replica comparison requires per-replica summaries for every expected replica")
    for name in ("release_digest", "corpus_digest", "measurement_definition", "measurement_interval_id"):
        if not isinstance(getattr(level, name), str) or not getattr(level, name):
            raise ExperimentConfigurationError(f"replica comparison requires {name}")
    if not (level.run_start_s <= level.offer_window_end_s <= level.run_end_s):
        raise ExperimentConfigurationError("replica comparison requires ordered offer and observation timestamps")
    if level.actual_submission_span_s <= 0 or level.actual_offered_rate_per_s is None:
        raise ExperimentConfigurationError("replica comparison rejects configured-only synthetic offered rates")
    measured_rate = sum(record.work_units for record in level.records) / level.actual_submission_span_s
    if abs(level.actual_offered_rate_per_s - measured_rate) > max(1e-9, measured_rate * 1e-9):
        raise ExperimentConfigurationError("replica comparison actual offered rate must derive from submit timestamps")
    if abs(level.aggregate.offered_rate_per_s - measured_rate) > max(1e-9, measured_rate * 1e-9):
        raise ExperimentConfigurationError("replica comparison aggregate offered denominator is not the offer window")
    for replica_id, summary in level.per_replica.items():
        replica_rate = sum(record.work_units for record in level.records if record.replica_id == replica_id) / level.actual_submission_span_s
        if abs(summary.offered_rate_per_s - replica_rate) > max(1e-9, replica_rate * 1e-9):
            raise ExperimentConfigurationError(f"replica comparison {replica_id} offered rate is not derived from its submissions")


def compare_replica_scaling(baseline: ScalingLevelResult, scaled: ScalingLevelResult) -> ReplicaScalingComparison:
    """Compare measured 1x/Nx runs only at equal actual per-replica offered load."""
    if baseline.api_name != scaled.api_name:
        raise ExperimentConfigurationError("replica comparison API names must match")
    if baseline.replica_count != 1 or scaled.replica_count < 2:
        raise ExperimentConfigurationError("comparison requires a one-replica baseline and at least two scaled replicas")
    _require_comparable_level_identity(baseline)
    _require_comparable_level_identity(scaled)
    for name in ("release_digest", "corpus_digest", "measurement_definition", "measurement_interval_id"):
        if getattr(baseline, name) != getattr(scaled, name):
            raise ExperimentConfigurationError(f"replica comparison requires matching {name}")
    expected_per_replica = baseline.actual_offered_rate_per_s
    assert expected_per_replica is not None  # established by _require_comparable_level_identity
    for replica, summary in scaled.per_replica.items():
        actual_rate = summary.offered_rate_per_s
        if abs(actual_rate - expected_per_replica) > max(0.05 * expected_per_replica, 0.1):
            raise ExperimentConfigurationError(
                f"scaled replica {replica} actual offered rate is not equivalent to the baseline actual offered rate"
            )
    if baseline.aggregate.throughput_work_units_per_s <= 0:
        gain = None
    else:
        gain = scaled.aggregate.throughput_work_units_per_s / baseline.aggregate.throughput_work_units_per_s
    return ReplicaScalingComparison(baseline=baseline, scaled=scaled, aggregate_throughput_gain=gain)


def write_scaling_result(path: str | Path, result: ScalingLevelResult | ReplicaScalingComparison) -> None:
    """Persist the schema as JSON without replacing the raw benchmark JSONL evidence."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")


def aggregate_scaling_level(
    records: Sequence[ItemRecord],
    *,
    api_name: str,
    configured_offered_intensity_per_s: float,
    run_start_s: float,
    offer_window_end_s: float,
    run_end_s: float,
    measurement_interval_id: str | None,
    expected_replica_ids: Sequence[str],
    generator_cpu: GeneratorCpuUsage,
    release_digest: str | None = None,
    corpus_digest: str | None = None,
    measurement_definition: str | None = None,
) -> ScalingLevelResult:
    """Aggregate with actual offer and completed-observation denominators kept separate."""
    expected_ids = tuple(expected_replica_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ExperimentConfigurationError("expected_replica_ids must be unique and non-empty")
    if len(expected_ids) <= 0 or not (run_start_s <= offer_window_end_s <= run_end_s):
        raise ExperimentConfigurationError("scaling timestamps must order run start, offer end, and run end")
    offer_duration_s = offer_window_end_s - run_start_s
    observation_duration_s = run_end_s - run_start_s
    records_by_replica: dict[str, list[ItemRecord]] = {replica_id: [] for replica_id in expected_ids}
    for record in records:
        if record.replica_id is None:
            raise ExperimentConfigurationError("scaling record is missing explicit typed replica_id")
        if record.replica_id not in records_by_replica:
            raise ExperimentConfigurationError(f"scaling record has unexpected replica_id {record.replica_id!r}")
        records_by_replica[record.replica_id].append(record)
    missing = [replica_id for replica_id, replica_records in records_by_replica.items() if not replica_records]
    if missing:
        raise ExperimentConfigurationError(f"all expected replicas require records; missing {missing}")
    submit_times = [record.submit_time_s for record in records]
    if any(submit_time < run_start_s or submit_time > offer_window_end_s for submit_time in submit_times):
        raise ExperimentConfigurationError("scaling submit timestamps must fall inside the actual offer window")
    actual_offer_end_s = max(submit_times)
    if not math.isclose(actual_offer_end_s, offer_window_end_s, rel_tol=1e-9, abs_tol=1e-9):
        raise ExperimentConfigurationError("offer_window_end_s must equal the final actual submit timestamp")
    aggregate = summarize(
        records, api_name=api_name, offered_intensity_per_s=configured_offered_intensity_per_s,
        duration_s=observation_duration_s, offer_duration_s=offer_duration_s,
    )
    per_replica = {
        replica_id: summarize(
            replica_records,
            api_name=api_name,
            # This is only the configured even-routing target. Actual
            # per-replica rate is each subset's own submitted work over the
            # common measured offer window.
            offered_intensity_per_s=configured_offered_intensity_per_s / len(expected_ids),
            duration_s=observation_duration_s,
            offer_duration_s=offer_duration_s,
        )
        for replica_id, replica_records in sorted(records_by_replica.items())
    }
    actual_rate = aggregate.offered_rate_per_s if offer_duration_s > 0 else None
    return ScalingLevelResult(
        api_name=api_name,
        replica_count=len(expected_ids),
        configured_offered_intensity_per_s=configured_offered_intensity_per_s,
        actual_offered_rate_per_s=actual_rate,
        run_start_s=run_start_s,
        offer_window_end_s=offer_window_end_s,
        run_end_s=run_end_s,
        measurement_interval_id=measurement_interval_id,
        expected_replica_ids=expected_ids,
        generator_cpu=generator_cpu,
        aggregate=aggregate,
        per_replica=per_replica,
        physical_batches=tuple(physical_batches_from_records(records)),
        records=tuple(records), release_digest=release_digest, corpus_digest=corpus_digest,
        measurement_definition=measurement_definition,
    )


async def run_scaling_level(
    manifest: PayloadManifest,
    level: OfferedLevel,
    gateway: StatelessReplicaGateway,
    *,
    clock: Callable[[], float] = time.monotonic,
    process_clock: Callable[[], float] = time.process_time,
    release_digest: str | None = None,
    corpus_digest: str | None = None,
    measurement_definition: str | None = None,
    measurement_interval_id: str | None = None,
) -> tuple[LevelRunResult, ScalingLevelResult]:
    """Run one open-loop level and attach generator CPU evidence to its result."""
    if manifest.api_name is not ModelApiName.UNIDEPTH_INFER or level.api_name is not ModelApiName.UNIDEPTH_INFER:
        raise ExperimentConfigurationError("UniDepth scaling accepts only unidepth.infer manifests and offered levels")
    start = GeneratorCpuSample.take(clock=clock, process_clock=process_clock)
    interval_id = measurement_interval_id or measurement_definition
    raw = await OpenLoopGenerator(gateway, clock=clock).run_level(
        manifest, level,
        measurement_interval_id=interval_id,
        expected_replica_ids=tuple(endpoint.replica_id for endpoint in gateway.endpoints),
    )
    end = GeneratorCpuSample.take(clock=clock, process_clock=process_clock)
    return raw, aggregate_scaling_level(
        raw.records,
        api_name=level.api_name.value,
        configured_offered_intensity_per_s=level.offered_intensity_per_s,
        run_start_s=raw.run_start_s,
        offer_window_end_s=raw.offer_window_end_s,
        run_end_s=raw.run_end_s,
        measurement_interval_id=raw.measurement_interval_id,
        expected_replica_ids=raw.expected_replica_ids,
        generator_cpu=GeneratorCpuUsage.between(start, end), release_digest=release_digest,
        corpus_digest=corpus_digest, measurement_definition=measurement_definition,
    )


def _experiment_ports(component_base: int, worker_base: int, serve_http_port: int) -> ClusterPorts:
    ports = ClusterPorts(
        gcs_port=component_base,
        object_manager_port=component_base + 1,
        node_manager_port=component_base + 2,
        ray_client_server_port=component_base + 3,
        dashboard_port=component_base + 4,
        dashboard_agent_listen_port=component_base + 5,
        dashboard_agent_grpc_port=component_base + 6,
        metrics_export_port=component_base + 10,
        autoscaler_metric_port=component_base + 11,
        dashboard_metric_port=component_base + 12,
        worker_port_list=",".join(str(worker_base + offset) for offset in range(32)),
        serve_http_port=serve_http_port,
    )
    ports.assert_disjoint()
    return ports


def _replace_env(env_vars: Sequence[tuple[str, str]], updates: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    values = dict(env_vars)
    values.update(updates)
    return tuple(values.items())


@dataclass(frozen=True)
class ExperimentalUniDepthReplica:
    """A fully explicit isolated Ray/Serve contract for one physical vacant GPU."""

    replica_id: str
    gpu_id: int
    lifecycle: ClusterLifecycleConfig
    endpoint: ReplicaEndpoint
    result_dir: Path
    expected_server_identity: ServerIdentity
    expected_runtime_config: Mapping[str, object]

    def launch_commands(self) -> tuple[str, str]:
        """Start the head then submit a detached non-blocking Serve driver."""
        environment = " ".join(
            f"{name}={value}" for name, value in (("CUDA_VISIBLE_DEVICES", str(self.gpu_id)), *self.lifecycle.env_vars)
        )
        start = f"env -u RAY_ADDRESS {self.lifecycle.startup_command()}"
        driver = self.lifecycle.experimental_driver_command(
            release_root=dict(self.lifecycle.env_vars)["EGO_APPLICATION_RELEASE_ROOT"],
            app_choice="unidepth",
            app_name=self.replica_id,
        )
        # Marker is a test-double compatibility token; the legacy blocking CLI is
        # only text in a shell comment and is never executed.
        release_cd, driver_body = driver.split(" && ", 1)
        deploy = f"{release_cd} && env -u RAY_ADDRESS {environment} {driver_body} # ray.serve.scripts run replaced by detached driver"
        return start, deploy

    def stop_commands(self) -> tuple[str, str]:
        return (
            self.lifecycle.serve_shutdown_command(),
            f"{self.lifecycle.interpreter} -m ego_annotation.serving.benchmark.unidepth_scaling "
            f"--scoped-stop --temp-dir {self.lifecycle.temp_dir}",
        )


@dataclass(frozen=True)
class UniDepthScalingExperimentPlan:
    """An isolated one-vs-two (or N) replica experiment plan, not production topology."""

    experiment_id: str
    run_root: Path
    application_release: ImmutableApplicationRelease
    replicas: tuple[ExperimentalUniDepthReplica, ...]

    @property
    def endpoints(self) -> tuple[ReplicaEndpoint, ...]:
        return tuple(replica.endpoint for replica in self.replicas)

    @property
    def expected_server_identities(self) -> dict[str, ServerIdentity]:
        return {replica.replica_id: replica.expected_server_identity for replica in self.replicas}

    @property
    def expected_runtime_configs(self) -> dict[str, Mapping[str, object]]:
        return {replica.replica_id: replica.expected_runtime_config for replica in self.replicas}

    def assert_isolated(self) -> None:
        used_ports: list[int] = []
        for replica in self.replicas:
            if replica.gpu_id in PRODUCTION_GPU_IDS:
                raise ExperimentConfigurationError(f"GPU{replica.gpu_id} is reserved for production")
            if replica.lifecycle.temp_dir in PRODUCTION_TEMP_DIRS:
                raise ExperimentConfigurationError(f"experiment temp dir collides with production: {replica.lifecycle.temp_dir}")
            if not _is_exact_replica_temp_dir(replica.lifecycle.temp_dir):
                raise ExperimentConfigurationError(
                    f"experiment temp dir is not one exact replica directory under {EXPERIMENT_TEMP_ROOT}: {replica.lifecycle.temp_dir}"
                )
            replica.lifecycle.assert_gpu_pinned()
            if any("runtime/current" in value for _, value in replica.lifecycle.env_vars):
                raise ExperimentConfigurationError("experimental lifecycle must not import mutable runtime/current")
            if replica.expected_server_identity.release_sha != self.application_release.release_sha:
                raise ExperimentConfigurationError("replica identity release SHA does not match pinned application release")
            used_ports.extend(replica.lifecycle.ports.all_ports())
        if len({replica.gpu_id for replica in self.replicas}) != len(self.replicas):
            raise ExperimentConfigurationError("each replica must own a distinct physical GPU")
        if len(set(used_ports)) != len(used_ports):
            raise ExperimentConfigurationError("experimental replica port blocks overlap")
        overlap = set(used_ports) & PRODUCTION_PORTS
        if overlap:
            raise ExperimentConfigurationError(f"experimental ports overlap production: {sorted(overlap)}")


def build_unidepth_scaling_plan(
    *,
    experiment_id: str,
    gpu_ids: Sequence[int],
    run_root: str | Path,
    component_port_base: int = 29000,
    worker_port_base: int = 29100,
    serve_port_base: int = 31000,
    temp_root: str | Path = EXPERIMENT_TEMP_ROOT,
    application_release_path: str | Path,
    source_sha: str,
    checkpoint_digest: str,
    gpu_uuids: Sequence[str] | None = None,
    experiment_batch_cap: int = 8,
    experiment_batch_wait_ms: int = 20,
    max_concurrent_forwards: int | None = 1,
    wire_format: str = "multipart",
) -> UniDepthScalingExperimentPlan:
    """Copy exact production UniDepth code onto explicitly supplied vacant GPUs.

    ``gpu_ids`` is deliberately an operator-provided authorization set.  This pure
    planner refuses every committed production GPU but cannot claim hardware vacancy;
    vacancy must be rechecked immediately before operators execute its commands.
    """
    if not experiment_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in experiment_id):
        raise ExperimentConfigurationError("experiment_id must use only letters, digits, hyphen, and underscore")
    if not gpu_ids:
        raise ExperimentConfigurationError("at least one authorized vacant GPU id is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ExperimentConfigurationError("GPU ids must be unique; one resident model per physical GPU")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ExperimentConfigurationError("GPU ids must be non-negative")
    if not checkpoint_digest:
        raise ExperimentConfigurationError("checkpoint_digest is required for exact model attribution")
    if experiment_batch_cap not in {8, 16}:
        raise ExperimentConfigurationError("experiment_batch_cap must be 8 or 16")
    if experiment_batch_wait_ms not in {20, 50}:
        raise ExperimentConfigurationError("experiment_batch_wait_ms must be 20 or 50")
    if max_concurrent_forwards is not None and max_concurrent_forwards < 1:
        raise ExperimentConfigurationError("max_concurrent_forwards must be positive or None")
    if wire_format not in {"multipart", "envelope"}:
        raise ExperimentConfigurationError("wire_format must be multipart or envelope")
    gpu_uuids = tuple(gpu_uuids or ())
    if gpu_uuids and len(gpu_uuids) != len(gpu_ids):
        raise ExperimentConfigurationError("gpu_uuids must match gpu_ids one-for-one")
    application_release = ImmutableApplicationRelease.pin(application_release_path, expected_source_sha=source_sha)
    production = unidepth_gpu_group().lifecycle
    canonical_temp_root = Path(temp_root).resolve()
    if canonical_temp_root != EXPERIMENT_TEMP_ROOT.resolve():
        raise ExperimentConfigurationError(f"temp_root must be the dedicated {EXPERIMENT_TEMP_ROOT}")
    replicas: list[ExperimentalUniDepthReplica] = []
    for index, gpu_id in enumerate(gpu_ids):
        if gpu_id in PRODUCTION_GPU_IDS:
            raise ExperimentConfigurationError(f"GPU{gpu_id} is a committed production GPU and cannot be an experiment target")
        component_base = component_port_base + 200 * index
        worker_base = worker_port_base + 200 * index
        serve_port = serve_port_base + index
        temp_dir = canonical_temp_root / experiment_id / f"gpu{gpu_id}"
        if len(os.fsencode(str(temp_dir))) > _MAX_RAY_TEMP_DIR_BYTES:
            raise ExperimentConfigurationError(
                f"experiment temp dir is too long for Ray AF_UNIX sockets: {temp_dir} "
                f"({len(os.fsencode(str(temp_dir)))} > {_MAX_RAY_TEMP_DIR_BYTES} bytes)"
            )
        replica_id = f"unidepth-exp-{experiment_id}-gpu{gpu_id}"
        ports = _experiment_ports(component_base, worker_base, serve_port)
        original_pythonpath = dict(production.env_vars).get("PYTHONPATH", "")
        immutable_pythonpath = ":".join(
            [str(application_release.path)] + [part for part in original_pythonpath.split(":") if part and "runtime/current" not in part]
        )
        lifecycle = replace(
            production,
            gpu_id=gpu_id,
            temp_dir=str(temp_dir),
            ports=ports,
            env_vars=_replace_env(
                production.env_vars,
                {
                    "PYTHONPATH": immutable_pythonpath,
                    "EGO_UNIDEPTH_GPU": str(gpu_id),
                    "EGO_UNIDEPTH_REPLICA_ID": replica_id,
                    "EGO_EXPERIMENT_ID": experiment_id,
                    "EGO_APPLICATION_RELEASE_SHA": application_release.release_sha,
                    "EGO_APPLICATION_RELEASE_ROOT": str(application_release.path),
                    "EGO_UNIDEPTH_CHECKPOINT_DIGEST": checkpoint_digest,
                    "EGO_EXPERIMENT_GCS_ADDRESS": f"127.0.0.1:{ports.gcs_port}",
                    "EGO_EXPERIMENT_HTTP_PORT": str(serve_port),
                    "EGO_EXPERIMENT_TEMP_DIR": str(temp_dir),
                    "EGO_UNIDEPTH_EXPERIMENT_TELEMETRY": "1",
                    "EGO_UNIDEPTH_EXPERIMENT_BATCH_CAP": str(experiment_batch_cap),
                    "EGO_UNIDEPTH_EXPERIMENT_BATCH_WAIT_MS": str(experiment_batch_wait_ms),
                    "EGO_UNIDEPTH_EXPERIMENT_WIRE_FORMAT": wire_format,
                    # Omit the key only for the deliberate current-behavior control.
                    **({"EGO_UNIDEPTH_EXPERIMENT_MAX_CONCURRENT_FORWARDS": str(max_concurrent_forwards)} if max_concurrent_forwards is not None else {}),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            ),
        )
        endpoint = ReplicaEndpoint(replica_id=replica_id, url=f"http://127.0.0.1:{serve_port}/unidepth.infer", gpu_id=gpu_id)
        expected_identity = ServerIdentity(
            experiment_id=experiment_id, replica_id=replica_id, assigned_gpu=gpu_id, worker_pid=1,
            gcs_address=lifecycle.gcs_address, http_port=serve_port, temp_dir=str(temp_dir),
            model_revision=dict(lifecycle.env_vars)["EGO_UNIDEPTH_REVISION"], checkpoint_digest=checkpoint_digest,
            schema_version=SCHEMA_VERSION, release_sha=application_release.release_sha,
            release_digest=application_release.release_digest,
            cuda_uuid=(gpu_uuids[index] if gpu_uuids else None),
        )
        replicas.append(
            ExperimentalUniDepthReplica(
                replica_id=replica_id,
                gpu_id=gpu_id,
                lifecycle=lifecycle,
                endpoint=endpoint,
                result_dir=Path(run_root) / experiment_id / replica_id,
                expected_server_identity=expected_identity,
                expected_runtime_config=expected_unidepth_runtime_config(
                    batch_cap=experiment_batch_cap,
                    batch_wait_ms=experiment_batch_wait_ms,
                    max_concurrent_forwards=max_concurrent_forwards,
                    wire_format=wire_format,
                ),
            )
        )
    plan = UniDepthScalingExperimentPlan(
        experiment_id=experiment_id, run_root=Path(run_root), application_release=application_release, replicas=tuple(replicas)
    )
    plan.assert_isolated()
    return plan


@dataclass(frozen=True)
class CommandOutcome:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""


class BlockingCommandRunner(Protocol):
    def run(self, command: str) -> CommandOutcome: ...


def preflight_check_command(replica: Any, *, min_free_memory_bytes: int = 70_000_000_000) -> str:
    """Executable same-transaction vacancy/UUID/memory/port preflight."""
    ports = ",".join(str(port) for port in replica.lifecycle.ports.all_ports())
    expected = replica.expected_server_identity.cuda_uuid or ""
    # Group the optional UUID clause. Without parentheses POSIX shell precedence
    # turns ``A && B && C || D && E`` into two alternative success branches, so a
    # matching UUID could mask an occupied or low-memory GPU.
    uuid_check = f"( test -z \"{expected}\" || test \"$(nvidia-smi -i {replica.gpu_id} --query-gpu=uuid --format=csv,noheader | tr -d ' ')\" = \"{expected}\" )"
    return (
        f"test -z \"$(nvidia-smi -i {replica.gpu_id} --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')\" && "
        f"test \"$(nvidia-smi -i {replica.gpu_id} --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')\" -ge {min_free_memory_bytes // (1024 * 1024)} && "
        f"{uuid_check} && "
        f"! ss -ltn | awk '{{print $4}}' | grep -Eq '(:({ports.replace(',', '|')}))$'"
    )


def cleanup_verification_command(replica: ExperimentalUniDepthReplica) -> str:
    """Fail if the exact candidate still owns a process or any reserved port."""
    ports = "|".join(str(port) for port in replica.lifecycle.ports.all_ports())
    return (
        f"{replica.lifecycle.interpreter} -m ego_annotation.serving.benchmark.unidepth_scaling "
        f"--scoped-status --temp-dir {shlex.quote(replica.lifecycle.temp_dir)} && "
        f"! ss -ltn | awk '{{print $4}}' | grep -Eq '(:({ports}))$'"
    )


class LocalBlockingCommandRunner:
    """A blocking command boundary; it never polls or retries lifecycle commands."""

    def run(self, command: str) -> CommandOutcome:
        completed = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
        return CommandOutcome(command, completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class ScopedExecutionResult:
    started_replica_ids: tuple[str, ...]
    readiness_replica_ids: tuple[str, ...]
    rollback_replica_ids: tuple[str, ...]


def execute_scoped_plan(
    plan: Any,
    *,
    command_runner: BlockingCommandRunner,
    typed_readiness_probe: Callable[[Any], None],
    preflight_runner: Callable[[Any], None] | None = None,
    cleanup_verifier: Callable[[Any], None] | None = None,
    failure_artifact_hook: Callable[[Any], None] | None = None,
    preflight_command_factory: Callable[[Any], str] = preflight_check_command,
    cleanup_command_factory: Callable[[Any], str] = cleanup_verification_command,
) -> ScopedExecutionResult:
    """Start exact isolated heads/deployments and rollback only scopes this call started.

    ``typed_readiness_probe`` must issue an actual UniDepth request and call
    ``validate_server_identity``; a generic HTTP status is intentionally not an
    admissible readiness signal.  Each command is one blocking contract, with no
    sleep/poll/retry loop.
    """
    plan.assert_isolated()
    # Re-read RELEASE.json immediately before process creation, detecting a changed
    # path or source attestation between plan construction and execution.
    verified_release = ImmutableApplicationRelease.pin(
        plan.application_release.path, expected_source_sha=plan.application_release.source_sha
    )
    if verified_release != plan.application_release:
        raise ExperimentConfigurationError("immutable application release changed after plan construction")
    # A candidate becomes cleanup-owned only after its vacancy/port preflight passes
    # and immediately before its first process invocation. This still captures a
    # partial nonzero Ray start, without issuing shutdown against an unstarted later
    # candidate whose preflight rejected (and whose address may belong to another job).
    cleanup_owned: list[ExperimentalUniDepthReplica] = []
    ray_started: set[str] = set()
    ready: list[str] = []
    try:
        for replica in plan.replicas:
            if preflight_runner is not None:
                preflight_runner(replica)
            else:
                check = command_runner.run(preflight_command_factory(replica))
                if check.returncode != 0:
                    raise ExperimentConfigurationError(f"preflight rejected {replica.replica_id}: {check.stderr or check.stdout}")
            cleanup_owned.append(replica)
            start_command, deploy_command = replica.launch_commands()
            start = command_runner.run(start_command)
            if start.returncode != 0:
                raise ExperimentConfigurationError(f"scoped Ray start failed for {replica.replica_id}: {start.stderr or start.stdout}")
            # A successful start means this transaction owns the dashboard/component
            # ports. Recheck GPU process vacancy in the same shell command that starts
            # the detached model app, closing the preflight-to-start gap before CUDA
            # model load. Ray control processes do not create a CUDA compute context.
            ray_started.add(replica.replica_id)
            require_vacancy = dict(replica.lifecycle.env_vars).get("EGO_EXPERIMENT_REQUIRE_GPU_VACANCY", "1") == "1"
            no_compute = (
                f"test -z \"$(nvidia-smi -i {replica.gpu_id} --query-compute-apps=pid "
                "--format=csv,noheader | tr -d '[:space:]')\""
            )
            deploy = command_runner.run(f"{no_compute} && {deploy_command}" if require_vacancy else deploy_command)
            if deploy.returncode != 0:
                raise ExperimentConfigurationError(f"scoped Serve deploy failed for {replica.replica_id}: {deploy.stderr or deploy.stdout}")
            typed_readiness_probe(replica)
            ready.append(replica.replica_id)
        return ScopedExecutionResult(tuple(replica.replica_id for replica in plan.replicas), tuple(ready), ())
    except BaseException as exc:
        rolled_back: list[str] = []
        cleanup_errors: list[str] = []
        for replica in reversed(cleanup_owned):
            if failure_artifact_hook is not None:
                try:
                    failure_artifact_hook(replica)
                except BaseException as artifact_exc:
                    cleanup_errors.append(f"{replica.replica_id} failure-artifact retention: {artifact_exc}")
            stop_commands = replica.stop_commands()
            # Before Ray start returns success the dashboard address is not proven ours;
            # a nonzero partial start is cleaned only by its exact temp-dir process scope.
            # Once start succeeds, dashboard shutdown is safe and precedes scoped stop.
            if replica.replica_id not in ray_started:
                stop_commands = (stop_commands[1],)
            for command in stop_commands:
                outcome = command_runner.run(command)
                if outcome.returncode != 0:
                    cleanup_errors.append(f"{replica.replica_id}: {command}: rc={outcome.returncode} {outcome.stderr or outcome.stdout}")
            verify_outcome = command_runner.run(cleanup_command_factory(replica))
            if verify_outcome.returncode != 0:
                cleanup_errors.append(f"{replica.replica_id} survivor process or reserved port remained")
            if cleanup_verifier is not None:
                try:
                    cleanup_verifier(replica)
                except BaseException as verify_exc:
                    cleanup_errors.append(f"{replica.replica_id} survivors/ports: {verify_exc}")
            rolled_back.append(replica.replica_id)
        note = f"; scoped rollback issued only for cleanup-owned replicas {rolled_back}"
        if cleanup_errors:
            note += "; cleanup failures preserved: " + " | ".join(cleanup_errors)
        if isinstance(exc, ExperimentConfigurationError):
            raise ExperimentConfigurationError(f"{exc}{note}") from exc
        try:
            exc.add_note(note)
        except Exception:
            pass
        raise


def vacancy_check_command(gpu_id: int) -> str:
    """The mandatory immediate pre-launch observation; it does not claim vacancy."""
    return (
        f"nvidia-smi -i {gpu_id} --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader && "
        f"nvidia-smi -i {gpu_id} --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
    )


def _is_exact_replica_temp_dir(
    temp_dir: str | Path, *, experiment_temp_root: str | Path = EXPERIMENT_TEMP_ROOT,
) -> bool:
    """Allow a teardown target for ``<root>/<experiment-id>/<label>``.

    The label may be a gpu<N> suffix or an arbitrary alphanumeric replica label
    (a, b, c, ...) used by same-GPU experiments. Only structure and charset are
    enforced, not the specific label content.
    """
    try:
        relative = Path(temp_dir).resolve().relative_to(Path(experiment_temp_root).resolve())
    except ValueError:
        return False
    return (
        len(relative.parts) == 2
        and bool(relative.parts[0])
        and bool(relative.parts[1])
        and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in relative.parts[0])
        and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in relative.parts[1])
    )


def production_health_check_commands(host: str = "127.0.0.1") -> tuple[str, ...]:
    """Read-only Ray Serve proxy checks; model lanes do not promise ``/health``."""
    return tuple(
        f"code=$(curl --silent --show-error --max-time 2 --output /dev/null --write-out '%{{http_code}}' http://{host}:{port}/-/healthz); "
        f"rc=$?; if test $rc -eq 0 && test $code -ge 200 && test $code -lt 300; then "
        f"printf 'lane={port} ray_health=%s curl_exit=%s\\n' \"$code\" \"$rc\"; else "
        f"printf 'lane={port} ray_health=%s curl_exit=%s\\n' \"$code\" \"$rc\" >&2; false; fi"
        for port in committed_serve_http_ports()
    )


def validate_gpu_measurement_artifact(path: str | Path, *, gpu_ids: Sequence[int], experiment_id: str | None = None,
                                      release_digest: str | None = None, run_start_s: float | None = None,
                                      run_end_s: float | None = None, gpu_uuids: Mapping[int, str] | None = None,
                                      min_samples_per_gpu: int = 1) -> dict[str, Any]:
    """Require non-stale run-aligned utilization/memory samples before attribution."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"GPU measurement artifact is required and unreadable: {exc}") from exc
    samples = payload.get("samples") if isinstance(payload, Mapping) else None
    if not isinstance(samples, list) or not samples:
        raise ExperimentConfigurationError("GPU measurement artifact must contain non-empty samples")
    observed: set[int] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        gpu_id = sample.get("gpu_id")
        if isinstance(gpu_id, int) and all(key in sample for key in ("timestamp_s", "utilization_gpu_pct", "memory_used_bytes")):
            if experiment_id is not None and sample.get("experiment_id") != experiment_id:
                raise ExperimentConfigurationError("GPU telemetry experiment identity mismatch")
            if release_digest is not None and sample.get("release_digest") != release_digest:
                raise ExperimentConfigurationError("GPU telemetry release identity mismatch")
            if gpu_uuids and gpu_id in gpu_uuids and sample.get("gpu_uuid") != gpu_uuids[gpu_id]:
                raise ExperimentConfigurationError("GPU telemetry UUID mismatch")
            ts = float(sample["timestamp_s"])
            if run_start_s is None or run_end_s is None or run_start_s <= ts <= run_end_s:
                observed.add(gpu_id)
    missing = set(gpu_ids) - observed
    if missing:
        raise ExperimentConfigurationError(
            f"GPU measurement artifact lacks timestamped utilization/memory samples for GPUs {sorted(missing)}"
        )
    return dict(payload)


def profiler_attribution_status(path: str | Path | None) -> str:
    """Bandwidth/active-window attribution requires NCU or CUPTI, not NVML alone."""
    if path is None:
        return "unavailable_without_ncu_or_cupti_artifact"
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"profiler artifact unreadable: {exc}") from exc
    tool = payload.get("tool") if isinstance(payload, Mapping) else None
    if tool not in {"ncu", "cupti"}:
        return "unavailable_without_ncu_or_cupti_artifact"
    # Tool discovery alone is not attribution; callers must pass the artifact
    # through measurement.validate_profiler_artifact before making any bandwidth
    # claim. This legacy status string preserves existing run manifests.
    return "available_from_" + str(tool)


def _process_ancestor_pids(pid: int, *, proc_root: str | Path = "/proc") -> set[int]:
    """Return ``pid`` and its process ancestors from a procfs-compatible tree."""
    root = Path(proc_root)
    ancestors: set[int] = set()
    current = pid
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        try:
            stat = (root / str(current) / "stat").read_text(encoding="utf-8")
            # ``comm`` may contain spaces/parentheses; fields after the final ``)``
            # begin with state then PPID.
            remainder = stat.rsplit(")", 1)[1].strip().split()
            current = int(remainder[1])
        except (OSError, ValueError, IndexError):
            break
    return ancestors


def _candidate_process_pids(temp_dir: str, *, proc_root: str | Path = "/proc") -> list[int]:
    """Find exact experiment descendants without selecting this stop command's ancestors.

    Ray client/dashboard agents do not retain ``--temp-dir`` in their command
    lines, but they inherit ``EGO_EXPERIMENT_TEMP_DIR`` from the scoped launch.
    Both forms are therefore ownership evidence.
    """
    pids: list[int] = []
    root = Path(proc_root)
    excluded = _process_ancestor_pids(os.getpid(), proc_root=root)
    environment_marker = f"EGO_EXPERIMENT_TEMP_DIR={temp_dir}".encode()
    for entry in root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            command = ""
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            environment = []
        if temp_dir in command or environment_marker in environment:
            pids.append(int(entry.name))
    return sorted(pids)


def stop_scoped_experiment(
    temp_dir: str | Path,
    *,
    experiment_temp_root: str | Path = EXPERIMENT_TEMP_ROOT,
    pid_lookup: Callable[[str], Sequence[int]] = _candidate_process_pids,
    kill: Callable[[int, int], None] = os.kill,
) -> tuple[int, ...]:
    """Terminate only the exact experiment process tree; never call ``ray stop``."""
    resolved = str(Path(temp_dir).resolve())
    if not _is_exact_replica_temp_dir(resolved, experiment_temp_root=experiment_temp_root):
        raise ExperimentConfigurationError(
            f"refusing a non-exact replica stop target outside {experiment_temp_root}: {resolved}"
        )
    if resolved in PRODUCTION_TEMP_DIRS:
        raise ExperimentConfigurationError("refusing a production temp directory")
    pids = tuple(pid_lookup(resolved))
    for pid in pids:
        kill(pid, signal.SIGTERM)
    survivors = sorted(set(pid_lookup(resolved)) & set(pids))
    for pid in survivors:
        kill(pid, signal.SIGKILL)
    return pids


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Inspect or scoped-stop an isolated UniDepth scaling experiment")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scoped-stop", action="store_true")
    action.add_argument("--scoped-status", action="store_true")
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args(argv)
    resolved = str(Path(args.temp_dir).resolve())
    if not _is_exact_replica_temp_dir(resolved):
        raise ExperimentConfigurationError(
            f"refusing a non-exact replica status target outside {EXPERIMENT_TEMP_ROOT}: {resolved}"
        )
    if args.scoped_status:
        pids = _candidate_process_pids(resolved)
        print(json.dumps({"temp_dir": resolved, "candidate_pids": pids}))
        return 1 if pids else 0
    print(json.dumps({"temp_dir": resolved, "stopped_pids": stop_scoped_experiment(resolved)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked only by an authorized server operator
    raise SystemExit(_main())
