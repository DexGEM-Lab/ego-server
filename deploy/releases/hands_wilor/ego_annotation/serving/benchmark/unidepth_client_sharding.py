"""Independent-process UniDepth client sharding for the U4 scaling discriminator.

The ordinary U4 gateway drives every endpoint from one asyncio process. Multipart
body construction and response parsing are synchronous Python work, so that client
can become the common ceiling even when the serving replicas are independent. This
module runs one complete HTTP client per endpoint in a separate OS process. The
parent only assigns distinct corpus items, releases a common monotonic start,
collects child evidence, and owns one run-aligned NVML sampler.

It deliberately has no Ray lifecycle operations: endpoints must already have passed
the isolated launch/readiness contract in :mod:`unidepth_scaling`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ego_annotation.serving.benchmark.generator import OpenLoopGenerator, OfferedLevel
from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, PayloadManifest, load_payload_manifest
from ego_annotation.serving.benchmark.measurement import NvmlSampler, validate_gpu_samples
from ego_annotation.serving.benchmark.metrics import ItemRecord, OUTCOME_COMPLETED
from ego_annotation.serving.benchmark.unidepth_scaling import (
    ExperimentConfigurationError,
    ReplicaEndpoint,
    _canonical_gpu_uuid,
    validate_server_identity,
)
from ego_annotation.serving.contracts import ErrorCode, SCHEMA_VERSION, ServerIdentity, ServiceError
from ego_annotation.serving.unidepth import expected_unidepth_runtime_config
from ego_annotation.serving.gateway import GatewayRequest, GatewayResponse, ModelServiceGateway, RetryPolicy
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


CHILD_SCHEMA = "ego.unidepth-client-shard-child.v2"
RUN_SCHEMA = "ego.unidepth-client-shard-run.v2"
MIN_OFFERS_PER_ENDPOINT = 200
SUPPORTED_PER_ENDPOINT_RATES = frozenset({8.0, 12.0, 16.0, 20.0, 24.0, 32.0})
CLIENT_SATURATION_CORES = 0.90


class ClientShardRunInvalid(ExperimentConfigurationError):
    """A shard run cannot support a cross-replica scaling claim."""


def _float_field(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClientShardRunInvalid(f"{name} must be numeric")
    return float(value)


def _int_field(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClientShardRunInvalid(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class ClientShardSpec:
    """All data one independent client needs; no shared live client state."""

    experiment_id: str
    child_id: str
    endpoint_replica_id: str
    endpoint: str
    expected_identity: ServerIdentity
    expected_runtime_config: Mapping[str, object]
    payload_source: str
    corpus_digest: str
    item_indices: tuple[int, ...]
    scheduled_start_s: float
    per_endpoint_rate: float
    drivers_per_endpoint: int
    output_path: str
    wire_format: str = "multipart"

    def __post_init__(self) -> None:
        if not self.child_id or not self.endpoint_replica_id or not self.endpoint:
            raise ClientShardRunInvalid("child_id, endpoint_replica_id, and endpoint are required")
        if not self.item_indices or len(set(self.item_indices)) != len(self.item_indices):
            raise ClientShardRunInvalid("each child must receive a non-empty unique corpus shard")
        if any(not isinstance(index, int) or index < 0 for index in self.item_indices):
            raise ClientShardRunInvalid("child corpus indices must be non-negative integers")
        if self.per_endpoint_rate not in SUPPORTED_PER_ENDPOINT_RATES:
            raise ClientShardRunInvalid(f"per-endpoint rate must be one of {sorted(SUPPORTED_PER_ENDPOINT_RATES)}")
        if self.drivers_per_endpoint < 1:
            raise ClientShardRunInvalid("drivers_per_endpoint must be positive")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ClientShardRunInvalid("wire_format must be multipart or envelope")
        if not self.corpus_digest:
            raise ClientShardRunInvalid("corpus_digest is required")
        if self.expected_identity.experiment_id != self.experiment_id:
            raise ClientShardRunInvalid("expected server identity experiment_id must match child experiment_id")
        if self.expected_identity.replica_id != self.endpoint_replica_id:
            raise ClientShardRunInvalid("endpoint_replica_id must equal expected server replica_id")
        raw_config = self.expected_runtime_config.get("runtime_config")
        policy = raw_config.get("batch_policy") if isinstance(raw_config, Mapping) else None
        if not isinstance(policy, Mapping):
            raise ClientShardRunInvalid("expected runtime config lacks a batch policy")
        expected = expected_unidepth_runtime_config(
            batch_cap=int(policy.get("max_batch_size", 0)),
            batch_wait_ms=float(policy.get("batch_wait_timeout_ms", 0)),
            max_concurrent_forwards=raw_config.get("max_concurrent_forwards"),
            wire_format=self.wire_format,
        )
        if dict(self.expected_runtime_config) != expected:
            raise ClientShardRunInvalid("expected runtime config must be a canonical worker-derived policy identity")

    @property
    def per_driver_rate(self) -> float:
        return self.per_endpoint_rate / self.drivers_per_endpoint

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CHILD_SCHEMA,
            "experiment_id": self.experiment_id,
            "child_id": self.child_id,
            "endpoint_replica_id": self.endpoint_replica_id,
            "endpoint": self.endpoint,
            "expected_identity": self.expected_identity.to_wire(),
            "expected_runtime_config": dict(self.expected_runtime_config),
            "payload_source": self.payload_source,
            "corpus_digest": self.corpus_digest,
            "item_indices": list(self.item_indices),
            "scheduled_start_s": self.scheduled_start_s,
            "per_endpoint_rate": self.per_endpoint_rate,
            "drivers_per_endpoint": self.drivers_per_endpoint,
            "per_driver_rate": self.per_driver_rate,
            "output_path": self.output_path,
            "wire_format": self.wire_format,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ClientShardSpec":
        try:
            if raw.get("schema") != CHILD_SCHEMA:
                raise ClientShardRunInvalid(f"child spec schema must be {CHILD_SCHEMA}")
            identity_raw = raw["expected_identity"]
            if not isinstance(identity_raw, Mapping):
                raise ClientShardRunInvalid("child spec expected_identity must be an object")
            indices = raw["item_indices"]
            if not isinstance(indices, list):
                raise ClientShardRunInvalid("child spec item_indices must be a list")
            return cls(
                experiment_id=str(raw["experiment_id"]), child_id=str(raw["child_id"]),
                endpoint_replica_id=str(raw["endpoint_replica_id"]), endpoint=str(raw["endpoint"]),
                expected_identity=ServerIdentity.from_wire(identity_raw), expected_runtime_config=dict(raw["expected_runtime_config"]),
                payload_source=str(raw["payload_source"]), corpus_digest=str(raw["corpus_digest"]), item_indices=tuple(indices),
                scheduled_start_s=_float_field(raw["scheduled_start_s"], "scheduled_start_s"),
                per_endpoint_rate=_float_field(raw["per_endpoint_rate"], "per_endpoint_rate"),
                drivers_per_endpoint=_int_field(raw["drivers_per_endpoint"], "drivers_per_endpoint"),
                output_path=str(raw["output_path"]), wire_format=str(raw.get("wire_format", "multipart")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ClientShardRunInvalid):
                raise
            raise ClientShardRunInvalid(f"invalid child spec: {exc}") from exc


@dataclass(frozen=True)
class ClientShardAggregate:
    """Aggregate result retaining offer and completion/drain clocks separately."""

    experiment_id: str
    corpus_digest: str
    release_digest: str
    checkpoint_digest: str
    schema_version: str
    per_endpoint_rate: float
    drivers_per_endpoint: int
    wire_format: str
    runtime_config: Mapping[str, object]
    scheduled_start_s: float
    offer_start_s: float
    offer_end_s: float
    observation_end_s: float
    child_ids: tuple[str, ...]
    records: tuple[ItemRecord, ...]

    @property
    def offer_duration_s(self) -> float:
        return self.offer_end_s - self.offer_start_s

    @property
    def observation_duration_s(self) -> float:
        return self.observation_end_s - self.offer_start_s

    @property
    def drain_duration_s(self) -> float:
        return self.observation_end_s - self.offer_end_s

    @property
    def offered_rate_per_s(self) -> float:
        return sum(record.work_units for record in self.records) / self.offer_duration_s

    @property
    def completed_rate_per_s(self) -> float:
        completed = sum(record.work_units for record in self.records if record.outcome == OUTCOME_COMPLETED)
        return completed / self.observation_duration_s

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RUN_SCHEMA,
            "experiment_id": self.experiment_id,
            "corpus_digest": self.corpus_digest,
            "release_digest": self.release_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "schema_version": self.schema_version,
            "per_endpoint_rate": self.per_endpoint_rate,
            "drivers_per_endpoint": self.drivers_per_endpoint,
            "wire_format": self.wire_format,
            "runtime_config": dict(self.runtime_config),
            "scheduled_start_s": self.scheduled_start_s,
            "offer_start_s": self.offer_start_s,
            "offer_end_s": self.offer_end_s,
            "offer_duration_s": self.offer_duration_s,
            "observation_end_s": self.observation_end_s,
            "observation_duration_s": self.observation_duration_s,
            "drain_duration_s": self.drain_duration_s,
            "offered_count": len(self.records),
            "completed_count": sum(record.outcome == OUTCOME_COMPLETED for record in self.records),
            "offered_rate_per_s": self.offered_rate_per_s,
            "completed_rate_per_s": self.completed_rate_per_s,
            "child_ids": list(self.child_ids),
            "records": [record.to_dict() for record in self.records],
            "denominator_definition": {
                "offered_rate": "all submitted work / common actual offer interval",
                "completed_rate": "completed work / common observation interval including response drain",
            },
        }


def _write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _payload_source_items(source_path: str | Path) -> list[Mapping[str, object]]:
    """Read descriptor provenance without touching the large tensor part files."""
    try:
        raw = json.loads(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientShardRunInvalid(f"cannot read payload source descriptor: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != PAYLOAD_SOURCE_SCHEMA:
        raise ClientShardRunInvalid(f"payload source must use {PAYLOAD_SOURCE_SCHEMA}")
    items = raw.get("items")
    if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
        raise ClientShardRunInvalid("payload source items must be a list of objects")
    return list(items)


def payload_source_item_count(source_path: str | Path) -> int:
    """Read only the descriptor needed for deterministic shard assignment."""
    return len(_payload_source_items(source_path))


def _assigned_payload_keys(spec: ClientShardSpec) -> set[tuple[str, str]]:
    items = _payload_source_items(spec.payload_source)
    try:
        keys = {(str(items[index]["item_id"]), str(items[index]["payload_hash"])) for index in spec.item_indices}
    except (IndexError, KeyError, TypeError) as exc:
        raise ClientShardRunInvalid(f"child {spec.child_id} assigned corpus descriptor entries are invalid: {exc}") from exc
    if len(keys) != len(spec.item_indices):
        raise ClientShardRunInvalid(f"child {spec.child_id} assigned corpus contains duplicate item/hash evidence")
    return keys


def build_client_shard_specs(
    *,
    experiment_id: str,
    endpoints: Sequence[ReplicaEndpoint],
    expected_identities: Mapping[str, ServerIdentity],
    expected_runtime_configs: Mapping[str, Mapping[str, object]],
    payload_source: str | Path,
    corpus_digest: str,
    scheduled_start_s: float,
    per_endpoint_rate: float,
    drivers_per_endpoint: int,
    output_dir: str | Path,
    offers_per_endpoint: int = MIN_OFFERS_PER_ENDPOINT,
    corpus_start_index: int = 0,
    wire_format: str = "multipart",
) -> tuple[ClientShardSpec, ...]:
    """Assign disjoint payloads to N independent OS drivers for each endpoint."""
    if per_endpoint_rate not in SUPPORTED_PER_ENDPOINT_RATES:
        raise ClientShardRunInvalid(f"per-endpoint rate must be one of {sorted(SUPPORTED_PER_ENDPOINT_RATES)}")
    if not endpoints:
        raise ClientShardRunInvalid("at least one experimental endpoint is required")
    if drivers_per_endpoint < 1:
        raise ClientShardRunInvalid("drivers_per_endpoint must be positive")
    if wire_format not in {"multipart", "envelope"}:
        raise ClientShardRunInvalid("wire_format must be multipart or envelope")
    if offers_per_endpoint < MIN_OFFERS_PER_ENDPOINT or offers_per_endpoint % drivers_per_endpoint:
        raise ClientShardRunInvalid(f"offers_per_endpoint must be >= {MIN_OFFERS_PER_ENDPOINT} and divisible by drivers_per_endpoint")
    if isinstance(corpus_start_index, bool) or not isinstance(corpus_start_index, int) or corpus_start_index < 0:
        raise ClientShardRunInvalid("corpus_start_index must be a non-negative integer")
    source_count = payload_source_item_count(payload_source)
    required = len(endpoints) * offers_per_endpoint
    if source_count < corpus_start_index + required:
        raise ClientShardRunInvalid(f"payload corpus has {source_count} items; run requires {required} unique items from offset {corpus_start_index}")
    endpoint_ids = [endpoint.replica_id for endpoint in endpoints]
    if len(set(endpoint_ids)) != len(endpoint_ids) or set(endpoint_ids) != set(expected_identities) or set(endpoint_ids) != set(expected_runtime_configs):
        raise ClientShardRunInvalid("endpoint identities and runtime configs must name exactly the unique endpoints")
    for replica_id in endpoint_ids:
        if not _canonical_gpu_uuid(expected_identities[replica_id].cuda_uuid):
            raise ClientShardRunInvalid("every endpoint requires an expected hardware CUDA UUID")
    items_per_driver = offers_per_endpoint // drivers_per_endpoint
    output_root = Path(output_dir)
    specs: list[ClientShardSpec] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        endpoint_start = corpus_start_index + endpoint_index * offers_per_endpoint
        for driver_index in range(drivers_per_endpoint):
            child_id = f"{endpoint.replica_id}-driver{driver_index:02d}"
            start = endpoint_start + driver_index * items_per_driver
            specs.append(ClientShardSpec(
                experiment_id=experiment_id, child_id=child_id, endpoint_replica_id=endpoint.replica_id,
                endpoint=endpoint.url, expected_identity=expected_identities[endpoint.replica_id],
                expected_runtime_config=dict(expected_runtime_configs[endpoint.replica_id]),
                payload_source=str(Path(payload_source).resolve()), corpus_digest=corpus_digest,
                item_indices=tuple(range(start, start + items_per_driver)), scheduled_start_s=scheduled_start_s,
                per_endpoint_rate=per_endpoint_rate, drivers_per_endpoint=drivers_per_endpoint,
                output_path=str(output_root / "children" / f"{child_id}.json"), wire_format=wire_format,
            ))
    return tuple(specs)


def _identity_matches(
    expected: ServerIdentity,
    actual: ServerIdentity,
    *,
    require_worker_pid: bool = False,
) -> bool:
    fields = (
        "experiment_id", "replica_id", "assigned_gpu", "gcs_address", "http_port", "temp_dir",
        "model_revision", "checkpoint_digest", "schema_version", "release_sha", "release_digest",
    )
    return (
        actual.worker_pid > 0
        and (not require_worker_pid or expected.worker_pid == actual.worker_pid)
        and all(getattr(expected, name) == getattr(actual, name) for name in fields)
        and _canonical_gpu_uuid(expected.cuda_uuid) is not None
        and _canonical_gpu_uuid(expected.cuda_uuid) == _canonical_gpu_uuid(actual.cuda_uuid)
    )


class _PinnedIdentityGateway:
    """One endpoint/client per child, preserving one-attempt open-loop outcomes."""

    def __init__(self, gateway: ModelServiceGateway, expected_identity: ServerIdentity, expected_runtime_config: Mapping[str, object]) -> None:
        self._gateway = gateway
        self._expected_identity = expected_identity
        self._expected_runtime_config = dict(expected_runtime_config)
        self.observed_identity: ServerIdentity | None = None
        self.observed_runtime_config_digest: str | None = None

    async def call(self, request: GatewayRequest) -> GatewayResponse:
        response = await self._gateway.call(request)
        if response.result is None:
            return GatewayResponse(
                ownership=response.ownership, error=response.error, attempts=response.attempts,
                last_status_code=response.last_status_code, transport_ms=response.transport_ms,
                replica_id=self._expected_identity.replica_id,
            )
        try:
            actual = validate_server_identity(self._expected_identity, response)
        except ExperimentConfigurationError as exc:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, f"server identity rejected: {exc}", False, request.ownership),
                attempts=response.attempts, last_status_code=response.last_status_code, transport_ms=response.transport_ms,
            )
        diagnostics = response.result.metadata.get("batch_diagnostics")
        actual_runtime = {
            "runtime_config": diagnostics.get("runtime_config"),
            "runtime_config_digest": diagnostics.get("runtime_config_digest"),
        } if isinstance(diagnostics, Mapping) else None
        if actual_runtime != self._expected_runtime_config:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, "worker runtime config does not match planned treatment", False, request.ownership),
                attempts=response.attempts, last_status_code=response.last_status_code, transport_ms=response.transport_ms,
            )
        digest = actual_runtime["runtime_config_digest"]
        if self.observed_identity is None:
            self.observed_identity = actual
            self.observed_runtime_config_digest = str(digest)
        elif not _identity_matches(self.observed_identity, actual, require_worker_pid=True):
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, "server identity changed within one pinned child", False, request.ownership),
                attempts=response.attempts, last_status_code=response.last_status_code, transport_ms=response.transport_ms,
            )
        elif self.observed_runtime_config_digest != digest:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, "worker runtime config changed within one pinned child", False, request.ownership),
                attempts=response.attempts, last_status_code=response.last_status_code, transport_ms=response.transport_ms,
            )
        return GatewayResponse(
            ownership=response.ownership, result=response.result, attempts=response.attempts,
            last_status_code=response.last_status_code, transport_ms=response.transport_ms, replica_id=actual.replica_id,
        )

    async def aclose(self) -> None:
        await self._gateway.aclose()


async def _wait_for_common_start(scheduled_start_s: float) -> float:
    delay = scheduled_start_s - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    return time.monotonic()


async def run_child(spec: ClientShardSpec, *, request_timeout_s: float = 30.0) -> dict[str, object]:
    """Run one child from its independent Python process and return its artifact."""
    manifest: PayloadManifest = load_payload_manifest(
        spec.payload_source, expected_api=ModelApiName.UNIDEPTH_INFER, item_indices=spec.item_indices,
    )
    if len(manifest.items) != len(spec.item_indices):
        raise ClientShardRunInvalid(f"child {spec.child_id} did not load its complete payload assignment")
    # Cold httpx import/client construction takes longer than the allowed common-start
    # skew on this host. Prepare every process-local transport before releasing the
    # synchronized offer clock; otherwise a valid run deterministically invalidates
    # itself after sending all requests.
    router = ModelServiceRouter.canonical().with_overrides({ModelApiName.UNIDEPTH_INFER: spec.endpoint})
    raw_gateway = ModelServiceGateway.with_httpx(
        router, timeout_s=request_timeout_s, retry_policy=RetryPolicy(max_attempts=1, deadline_s=0.0),
        wire_format=spec.wire_format,
    )
    gateway = _PinnedIdentityGateway(raw_gateway, spec.expected_identity, spec.expected_runtime_config)
    actual_start_s = await _wait_for_common_start(spec.scheduled_start_s)
    cpu_started = time.process_time()
    try:
        level = OfferedLevel(
            api_name=ModelApiName.UNIDEPTH_INFER, offered_intensity_per_s=spec.per_driver_rate,
            target_completed=len(spec.item_indices), max_offered=len(spec.item_indices),
        )
        raw = await OpenLoopGenerator(gateway).run_level(
            manifest, level, measurement_interval_id=f"client-sharding:{spec.experiment_id}",
            expected_replica_ids=(spec.child_id,),
        )
    finally:
        await gateway.aclose()
    cpu_elapsed = time.process_time() - cpu_started
    if not math.isclose(raw.run_start_s, actual_start_s, rel_tol=0.0, abs_tol=0.1):
        raise ClientShardRunInvalid("child generator did not begin at its common-start clock")
    return {
        "schema": CHILD_SCHEMA,
        "experiment_id": spec.experiment_id,
        "child_id": spec.child_id,
        "client_pid": os.getpid(),
        "endpoint": spec.endpoint,
        "endpoint_replica_id": spec.endpoint_replica_id,
        "expected_identity": spec.expected_identity.to_wire(),
        "observed_identity": gateway.observed_identity.to_wire() if gateway.observed_identity else None,
        "expected_runtime_config": dict(spec.expected_runtime_config),
        "observed_runtime_config_digest": gateway.observed_runtime_config_digest,
        "release_digest": spec.expected_identity.release_digest,
        "checkpoint_digest": spec.expected_identity.checkpoint_digest,
        "schema_version": spec.expected_identity.schema_version,
        "corpus_digest": spec.corpus_digest,
        "payload_source": spec.payload_source,
        "item_indices": list(spec.item_indices),
        "per_endpoint_rate": spec.per_endpoint_rate,
        "drivers_per_endpoint": spec.drivers_per_endpoint,
        "wire_format": spec.wire_format,
        "per_driver_rate": spec.per_driver_rate,
        "scheduled_start_s": spec.scheduled_start_s,
        "actual_start_s": raw.run_start_s,
        "offer_window_end_s": raw.offer_window_end_s,
        "run_end_s": raw.run_end_s,
        "actual_offered_rate_per_s": raw.actual_offered_rate_per_s,
        "client_cpu": {
            "wall_s": raw.run_end_s - raw.run_start_s,
            "process_cpu_s": cpu_elapsed,
            "process_cpu_utilization_cores": cpu_elapsed / (raw.run_end_s - raw.run_start_s) if raw.run_end_s > raw.run_start_s else None,
        },
        "records": [record.to_dict() for record in raw.records],
    }


def _records_from_child(raw_records: object) -> tuple[ItemRecord, ...]:
    if not isinstance(raw_records, list) or any(not isinstance(record, Mapping) for record in raw_records):
        raise ClientShardRunInvalid("child records must be a list of objects")
    try:
        return tuple(ItemRecord(**record) for record in raw_records)
    except TypeError as exc:
        raise ClientShardRunInvalid(f"child record schema is invalid: {exc}") from exc


def _load_child_artifact(path: str | Path) -> Mapping[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientShardRunInvalid(f"child artifact is absent or unreadable: {path}: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != CHILD_SCHEMA:
        raise ClientShardRunInvalid(f"child artifact {path} does not use {CHILD_SCHEMA}")
    return raw


def aggregate_child_artifacts(
    specs: Sequence[ClientShardSpec],
    *,
    max_start_skew_s: float = 0.5,
) -> ClientShardAggregate:
    """Verify and aggregate every child; absence/failure of one invalidates the run."""
    if not specs:
        raise ClientShardRunInvalid("cannot aggregate zero child shards")
    if max_start_skew_s < 0:
        raise ClientShardRunInvalid("max_start_skew_s must be non-negative")
    expected_by_id = {spec.child_id: spec for spec in specs}
    if len(expected_by_id) != len(specs):
        raise ClientShardRunInvalid("child specs must have unique child IDs")
    scheduled_starts = {spec.scheduled_start_s for spec in specs}
    if len(scheduled_starts) != 1:
        raise ClientShardRunInvalid("all child specs must share one planned monotonic start")
    reference = specs[0].expected_identity
    for spec in specs[1:]:
        candidate = spec.expected_identity
        if (
            candidate.release_digest != reference.release_digest
            or candidate.checkpoint_digest != reference.checkpoint_digest
            or candidate.schema_version != reference.schema_version
            or spec.corpus_digest != specs[0].corpus_digest
            or spec.per_endpoint_rate != specs[0].per_endpoint_rate
            or spec.drivers_per_endpoint != specs[0].drivers_per_endpoint
            or spec.wire_format != specs[0].wire_format
            or dict(spec.expected_runtime_config) != dict(specs[0].expected_runtime_config)
        ):
            raise ClientShardRunInvalid("all child specs must share release, checkpoint, schema, corpus, and rate")
    artifacts = {_id: _load_child_artifact(spec.output_path) for _id, spec in expected_by_id.items()}
    records_by_child: dict[str, tuple[ItemRecord, ...]] = {}
    starts: list[float] = []
    offer_ends: list[float] = []
    run_ends: list[float] = []
    payload_hashes: set[str] = set()
    item_ids: set[str] = set()
    observed_hardware_uuids: set[str] = set()
    for child_id, spec in expected_by_id.items():
        raw = artifacts[child_id]
        if (raw.get("experiment_id") != spec.experiment_id or raw.get("child_id") != child_id
                or raw.get("endpoint_replica_id") != spec.endpoint_replica_id or raw.get("endpoint") != spec.endpoint):
            raise ClientShardRunInvalid(f"child {child_id} does not match its parent-issued spec")
        if raw.get("item_indices") != list(spec.item_indices):
            raise ClientShardRunInvalid(f"child {child_id} did not use its parent-assigned corpus indices")
        for name, expected in (
            ("release_digest", spec.expected_identity.release_digest),
            ("checkpoint_digest", spec.expected_identity.checkpoint_digest),
            ("schema_version", spec.expected_identity.schema_version),
            ("corpus_digest", spec.corpus_digest),
            ("per_endpoint_rate", spec.per_endpoint_rate),
            ("drivers_per_endpoint", spec.drivers_per_endpoint),
            ("wire_format", spec.wire_format),
            ("expected_runtime_config", dict(spec.expected_runtime_config)),
            ("scheduled_start_s", spec.scheduled_start_s),
        ):
            if raw.get(name) != expected:
                raise ClientShardRunInvalid(f"child {child_id} has unequal {name}")
        try:
            expected_identity_raw = raw.get("expected_identity")
            observed_identity_raw = raw.get("observed_identity")
            if not isinstance(expected_identity_raw, Mapping) or not isinstance(observed_identity_raw, Mapping):
                raise ClientShardRunInvalid(f"child {child_id} lacks expected or observed server identity")
            if ServerIdentity.from_wire(expected_identity_raw) != spec.expected_identity:
                raise ClientShardRunInvalid(f"child {child_id} expected server identity changed")
            observed_identity = ServerIdentity.from_wire(observed_identity_raw)
        except (TypeError, KeyError, ValueError) as exc:
            if isinstance(exc, ClientShardRunInvalid):
                raise
            raise ClientShardRunInvalid(f"child {child_id} identity evidence is malformed: {exc}") from exc
        if not _identity_matches(spec.expected_identity, observed_identity):
            raise ClientShardRunInvalid(f"child {child_id} observed server identity does not match the planned replica")
        observed_uuid = _canonical_gpu_uuid(observed_identity.cuda_uuid)
        if observed_uuid is None:
            raise ClientShardRunInvalid("child responder omitted its hardware CUDA UUID")
        known_uuid = next((
            _canonical_gpu_uuid(other.expected_identity.cuda_uuid)
            for other in specs if other.endpoint_replica_id == spec.endpoint_replica_id
        ), None)
        if observed_uuid != known_uuid:
            raise ClientShardRunInvalid("child responder GPU UUID changed from its endpoint identity")
        observed_hardware_uuids.add(observed_uuid)
        records = _records_from_child(raw.get("records"))
        if len(records) != len(spec.item_indices):
            raise ClientShardRunInvalid(f"child {child_id} did not offer its complete assigned interval")
        if any(record.replica_id != spec.endpoint_replica_id for record in records):
            raise ClientShardRunInvalid(f"child {child_id} has records attributed to another replica")
        if any(record.attempts != 1 for record in records):
            raise ClientShardRunInvalid(f"child {child_id} violated one-attempt open-loop semantics")
        record_keys = {(record.item_id, record.payload_hash) for record in records}
        if len(record_keys) != len(spec.item_indices):
            raise ClientShardRunInvalid(f"child {child_id} reused a payload or item")
        if record_keys != _assigned_payload_keys(spec):
            raise ClientShardRunInvalid(f"child {child_id} records do not match its assigned corpus payloads")
        if payload_hashes & {record.payload_hash for record in records}:
            raise ClientShardRunInvalid("payload hash appears in more than one child shard")
        if item_ids & {record.item_id for record in records}:
            raise ClientShardRunInvalid("corpus item appears in more than one child shard")
        payload_hashes.update(record.payload_hash for record in records)
        item_ids.update(record.item_id for record in records)
        try:
            actual_start = _float_field(raw["actual_start_s"], "actual_start_s")
            offer_end = _float_field(raw["offer_window_end_s"], "offer_window_end_s")
            run_end = _float_field(raw["run_end_s"], "run_end_s")
            declared_rate = _float_field(raw["actual_offered_rate_per_s"], "actual_offered_rate_per_s")
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientShardRunInvalid(f"child {child_id} has malformed clocks/rate: {exc}") from exc
        if actual_start < spec.scheduled_start_s - 1e-3 or actual_start > spec.scheduled_start_s + max_start_skew_s:
            raise ClientShardRunInvalid(f"child {child_id} did not honor the common planned monotonic start")
        if not (actual_start <= offer_end <= run_end):
            raise ClientShardRunInvalid(f"child {child_id} has unordered submit/offer/drain clocks")
        if not math.isclose(offer_end, max(record.submit_time_s for record in records), abs_tol=1e-6):
            raise ClientShardRunInvalid(f"child {child_id} offer end is not its final actual submission")
        if any(record.submit_time_s < actual_start or record.submit_time_s > offer_end for record in records):
            raise ClientShardRunInvalid(f"child {child_id} submissions fall outside its offer interval")
        calculated_rate = sum(record.work_units for record in records) / (offer_end - actual_start)
        if not math.isclose(declared_rate, calculated_rate, rel_tol=1e-6, abs_tol=1e-6):
            raise ClientShardRunInvalid(f"child {child_id} actual offer rate is not derived from submit clocks")
        if abs(calculated_rate - spec.per_driver_rate) > max(0.10 * spec.per_driver_rate, 0.15):
            raise ClientShardRunInvalid(f"child {child_id} actual offer rate {calculated_rate:.3f} differs from its divided target")
        cpu = raw.get("client_cpu")
        utilization = cpu.get("process_cpu_utilization_cores") if isinstance(cpu, Mapping) else None
        if not isinstance(utilization, (int, float)) or isinstance(utilization, bool):
            raise ClientShardRunInvalid(f"child {child_id} lacks client CPU saturation evidence")
        if utilization >= CLIENT_SATURATION_CORES:
            raise ClientShardRunInvalid(f"child {child_id} saturated one driver core ({utilization:.3f})")
        if any(record.outcome == "in_flight" or record.response_time_s < record.submit_time_s for record in records):
            raise ClientShardRunInvalid(f"child {child_id} has incomplete response interval")
        records_by_child[child_id] = records
        starts.append(actual_start)
        offer_ends.append(offer_end)
        run_ends.append(run_end)
    expected_endpoint_uuids = {
        _canonical_gpu_uuid(spec.expected_identity.cuda_uuid)
        for spec in specs
    }
    if len(expected_endpoint_uuids) != len({spec.endpoint_replica_id for spec in specs}):
        raise ClientShardRunInvalid("independent endpoints must occupy distinct hardware CUDA UUIDs")
    if max(starts) - min(starts) > max_start_skew_s:
        raise ClientShardRunInvalid("children did not start in a common synchronized interval")
    if max(starts) > min(offer_ends):
        raise ClientShardRunInvalid("child offer windows do not overlap")
    child_rates = [sum(record.work_units for record in records) / (offer_end - start)
                   for records, start, offer_end in zip(records_by_child.values(), starts, offer_ends)]
    if max(child_rates) - min(child_rates) > max(0.10 * specs[0].per_driver_rate, 0.15):
        raise ClientShardRunInvalid("actual per-driver offer rates are not equal")
    endpoint_rates: dict[str, float] = {}
    for endpoint_id in {spec.endpoint_replica_id for spec in specs}:
        endpoint_records = [record for spec in specs if spec.endpoint_replica_id == endpoint_id for record in records_by_child[spec.child_id]]
        endpoint_rates[endpoint_id] = sum(record.work_units for record in endpoint_records) / (max(offer_ends) - min(starts))
    if any(abs(rate - specs[0].per_endpoint_rate) > max(0.10 * specs[0].per_endpoint_rate, 0.25) for rate in endpoint_rates.values()):
        raise ClientShardRunInvalid("actual endpoint offers are unequal to the treatment rate")
    all_records = tuple(record for child_id in expected_by_id for record in records_by_child[child_id])
    common_offer_start = min(starts)
    common_offer_end = max(offer_ends)
    common_observation_end = max(run_ends)
    if common_offer_end <= common_offer_start or common_observation_end < common_offer_end:
        raise ClientShardRunInvalid("common offer/observation clocks are invalid")
    first = specs[0]
    return ClientShardAggregate(
        experiment_id=first.experiment_id, corpus_digest=first.corpus_digest,
        release_digest=first.expected_identity.release_digest or "", checkpoint_digest=first.expected_identity.checkpoint_digest,
        schema_version=first.expected_identity.schema_version, per_endpoint_rate=first.per_endpoint_rate,
        drivers_per_endpoint=first.drivers_per_endpoint, wire_format=first.wire_format,
        runtime_config=dict(first.expected_runtime_config),
        scheduled_start_s=first.scheduled_start_s, offer_start_s=common_offer_start,
        offer_end_s=common_offer_end, observation_end_s=common_observation_end,
        child_ids=tuple(expected_by_id), records=all_records,
    )


def compare_saturation_treatments(
    baseline: ClientShardAggregate,
    treatment: ClientShardAggregate,
) -> dict[str, object]:
    """Validate a cap/wait comparison before any throughput interpretation.

    Driver artifacts have already rejected missing drivers, replacement workers,
    response/config changes, payload reuse, CPU saturation, and incomplete windows.
    This cross-run boundary additionally requires one release/checkpoint/corpus and
    equal actual endpoint offers; it permits only a deliberate batch-policy change.
    """
    for field in ("release_digest", "checkpoint_digest", "schema_version", "corpus_digest", "drivers_per_endpoint", "wire_format"):
        if getattr(baseline, field) != getattr(treatment, field):
            raise ClientShardRunInvalid(f"treatment comparison requires matching {field}")
    if baseline.per_endpoint_rate != treatment.per_endpoint_rate:
        raise ClientShardRunInvalid("treatment comparison requires matching configured endpoint rate")
    if abs(baseline.offered_rate_per_s - treatment.offered_rate_per_s) > max(0.05 * baseline.offered_rate_per_s, 0.15):
        raise ClientShardRunInvalid("treatment comparison rejects unequal actual offers")
    left_config = baseline.runtime_config.get("runtime_config")
    right_config = treatment.runtime_config.get("runtime_config")
    if not isinstance(left_config, Mapping) or not isinstance(right_config, Mapping):
        raise ClientShardRunInvalid("treatment comparison requires worker-derived runtime config")
    if left_config.get("max_concurrent_forwards") != right_config.get("max_concurrent_forwards"):
        raise ClientShardRunInvalid("treatment comparison requires matching forward-concurrency treatment")
    return {
        "schema": "ego.unidepth-saturation-comparison.v1",
        "wire_format": baseline.wire_format,
        "baseline_runtime_config": dict(baseline.runtime_config),
        "treatment_runtime_config": dict(treatment.runtime_config),
        "baseline_offered_rate_per_s": baseline.offered_rate_per_s,
        "treatment_offered_rate_per_s": treatment.offered_rate_per_s,
        "baseline_completed_rate_per_s": baseline.completed_rate_per_s,
        "treatment_completed_rate_per_s": treatment.completed_rate_per_s,
    }


def _require_successful_child_processes(
    return_codes: Mapping[str, int], expected_child_ids: Sequence[str],
) -> None:
    """A child crash is invalid evidence, never a smaller aggregate sample."""
    if set(return_codes) != set(expected_child_ids):
        raise ClientShardRunInvalid("parent did not receive one exit status per expected child")
    failed = {child_id: code for child_id, code in return_codes.items() if code != 0}
    if failed:
        raise ClientShardRunInvalid(f"client child failure invalidates the aggregate: {failed}")


def _verify_client_process_identities(
    specs: Sequence[ClientShardSpec], processes: Mapping[str, subprocess.Popen[str]],
) -> None:
    """Bind each artifact to the independently spawned OS client process."""
    for spec in specs:
        raw = _load_child_artifact(spec.output_path)
        try:
            client_pid = _int_field(raw["client_pid"], "client_pid")
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientShardRunInvalid(f"child {spec.child_id} artifact has no valid client PID") from exc
        if client_pid != processes[spec.child_id].pid:
            raise ClientShardRunInvalid(f"child {spec.child_id} artifact PID does not match the spawned client process")


def run_parent(
    specs: Sequence[ClientShardSpec],
    *,
    gpu_ids: Sequence[int],
    gpu_uuids: Mapping[int, str],
    nvml_sampler: NvmlSampler,
    output_path: str | Path,
    python_executable: str = sys.executable,
) -> ClientShardAggregate:
    """Launch children, own the sole sampler, and reject any partial process set."""
    if tuple(nvml_sampler.gpu_ids) != tuple(gpu_ids):
        raise ClientShardRunInvalid("parent NVML sampler must cover exactly the planned experiment GPUs")
    expected_gpu_uuids = {
        spec.expected_identity.assigned_gpu: spec.expected_identity.cuda_uuid
        for spec in specs
    }
    if len(expected_gpu_uuids) != len({spec.endpoint_replica_id for spec in specs}):
        raise ClientShardRunInvalid("each endpoint must bind one physical GPU identity")
    if set(expected_gpu_uuids) != set(gpu_ids) or set(gpu_uuids) != set(gpu_ids):
        raise ClientShardRunInvalid("planned responders, GPU indices, and UUID map must match one-for-one")
    canonical_expected = {gpu: _canonical_gpu_uuid(uuid) for gpu, uuid in expected_gpu_uuids.items()}
    canonical_supplied = {gpu: _canonical_gpu_uuid(uuid) for gpu, uuid in gpu_uuids.items()}
    canonical_sampled = {gpu: _canonical_gpu_uuid(uuid) for gpu, uuid in nvml_sampler.gpu_uuids.items()}
    if (
        any(uuid is None or not uuid for uuid in canonical_supplied.values())
        or len(set(canonical_supplied.values())) != len(gpu_ids)
        or canonical_expected != canonical_supplied
        or canonical_sampled != canonical_supplied
    ):
        raise ClientShardRunInvalid("server identities and NVML must bind distinct expected physical GPU UUIDs")
    processes: dict[str, subprocess.Popen[str]] = {}
    return_codes: dict[str, int] = {}
    gpu_samples_path = Path(output_path).with_name("gpu_samples.json")
    nvml_sampler.start()
    try:
        for spec in specs:
            spec_path = Path(spec.output_path).with_suffix(".spec.json")
            _write_json_atomic(spec_path, spec.to_dict())
            processes[spec.child_id] = subprocess.Popen(
                [python_executable, "-m", "ego_annotation.serving.benchmark.unidepth_client_sharding", "--child-spec", str(spec_path)],
                text=True,
            )
        return_codes = {child_id: process.wait() for child_id, process in processes.items()}
    finally:
        nvml_sampler.stop()
        nvml_sampler.write(gpu_samples_path)
    _require_successful_child_processes(return_codes, tuple(processes))
    _verify_client_process_identities(specs, processes)
    aggregate = aggregate_child_artifacts(specs)
    validate_gpu_samples(
        Path(output_path).with_name("gpu_samples.json"), gpu_ids=gpu_ids,
        experiment_id=aggregate.experiment_id, release_digest=aggregate.release_digest,
        run_start_s=aggregate.offer_start_s, run_end_s=aggregate.observation_end_s,
        gpu_uuids=gpu_uuids,
    )
    _write_json_atomic(output_path, aggregate.to_dict())
    return aggregate


def _child_main(spec_path: str | Path) -> int:
    raw = _load_child_artifact(spec_path)
    spec = ClientShardSpec.from_dict(raw)
    artifact = asyncio.run(run_child(spec))
    _write_json_atomic(spec.output_path, artifact)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one independent UniDepth client shard")
    parser.add_argument("--child-spec", required=True, type=Path)
    parser.add_argument("--wire-format", choices=("multipart", "envelope"), help="must match the parent-issued child spec")
    args = parser.parse_args(argv)
    try:
        spec = ClientShardSpec.from_dict(_load_child_artifact(args.child_spec))
        if args.wire_format is not None and args.wire_format != spec.wire_format:
            raise ClientShardRunInvalid("--wire-format must match the parent-issued child spec")
        return _child_main(args.child_spec)
    except (ClientShardRunInvalid, ExperimentConfigurationError, OSError, ValueError) as exc:
        print(f"client shard failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
