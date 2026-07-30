"""CPU-only contract tests for the UniDepth saturation client drivers."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA
from ego_annotation.serving.benchmark.metrics import ItemRecord
from ego_annotation.serving.benchmark.unidepth_client_sharding import (
    CLIENT_SATURATION_CORES,
    ClientShardRunInvalid,
    SUPPORTED_PER_ENDPOINT_RATES,
    _PinnedIdentityGateway,
    aggregate_child_artifacts,
    compare_saturation_treatments,
    build_client_shard_specs,
)
from ego_annotation.serving.benchmark.unidepth_scaling import EXPERIMENT_TEMP_ROOT, ReplicaEndpoint
from ego_annotation.serving.contracts import BatchTrace, ErrorCode, SCHEMA_VERSION, ServerIdentity
from ego_annotation.serving.gateway import GatewayResponse, GatewayResult
from ego_annotation.serving.unidepth import expected_unidepth_runtime_config


RUNTIME = expected_unidepth_runtime_config(batch_cap=16, batch_wait_ms=50, max_concurrent_forwards=1)


def _identity(replica_id: str = "replica-gpu4", gpu_id: int = 4) -> ServerIdentity:
    return ServerIdentity(
        experiment_id="saturation", replica_id=replica_id, assigned_gpu=gpu_id, worker_pid=1234 + gpu_id,
        gcs_address=f"127.0.0.1:{29000 + gpu_id}", http_port=31000 + gpu_id,
        temp_dir=str(EXPERIMENT_TEMP_ROOT / "saturation" / f"gpu{gpu_id}"),
        model_revision="unidepth-v2-vitl14-corrected", checkpoint_digest="sha256:checkpoint",
        schema_version=SCHEMA_VERSION, release_sha="release", release_digest="release", cuda_uuid=f"GPU-{gpu_id}",
    )


def _source(path: Path, count: int = 400) -> Path:
    path.write_text(json.dumps({"schema": PAYLOAD_SOURCE_SCHEMA, "items": [
        {"item_id": f"item-{i}", "payload_hash": f"hash-{i}"} for i in range(count)
    ]}))
    return path


def _specs(tmp_path: Path):
    endpoint = ReplicaEndpoint("replica-gpu4", "http://127.0.0.1:31000/unidepth.infer", 4)
    identity = _identity()
    return build_client_shard_specs(
        experiment_id="saturation", endpoints=(endpoint,), expected_identities={endpoint.replica_id: identity},
        expected_runtime_configs={endpoint.replica_id: RUNTIME}, payload_source=_source(tmp_path / "payload.json"),
        corpus_digest="corpus", scheduled_start_s=100.0, per_endpoint_rate=32.0, drivers_per_endpoint=4,
        output_dir=tmp_path / "out",
    )


def _record(spec, index: int, start: float) -> ItemRecord:
    return ItemRecord(
        item_id=f"item-{spec.item_indices[index]}", api_name="unidepth.infer", request_id=f"request-{spec.child_id}-{index}",
        job_id="job", work_units=1, payload_hash=f"hash-{spec.item_indices[index]}", source_timestamp_s=float(index),
        offer_time_s=start + index / spec.per_driver_rate, submit_time_s=start + index / spec.per_driver_rate,
        response_time_s=start + index / spec.per_driver_rate + .01, offered_delay_s=0.0, outcome="completed", http_status=200,
        attempts=1, error_code=None, error_message=None, response_latency_ms=10., transport_ms=1., admission_ms=1., queue_ms=1.,
        dispatch_ms=1., forward_ms=1., encoding_ms=1., batch_id=f"batch-{spec.child_id}-{index}", batch_size=1,
        batch_work_units=1, batch_wall_ms=1., amortized_cost_ms=1., model_load_count=1, replica_id=spec.endpoint_replica_id,
        runtime_config_digest=str(RUNTIME["runtime_config_digest"]), batch_policy_max_batch_size=16,
        batch_policy_wait_ms=50., max_concurrent_forwards=1, peak_simultaneous_forwards=1,
    )


def _write_child(spec, start: float = 100.01, cpu: float = .2) -> None:
    records = [_record(spec, i, start) for i in range(len(spec.item_indices))]
    offer_end = records[-1].submit_time_s
    payload = {
        "schema": "ego.unidepth-client-shard-child.v2", "experiment_id": spec.experiment_id, "child_id": spec.child_id,
        "endpoint_replica_id": spec.endpoint_replica_id, "endpoint": spec.endpoint, "client_pid": 900 + int(spec.child_id[-2:]),
        "expected_identity": spec.expected_identity.to_wire(), "observed_identity": spec.expected_identity.to_wire(),
        "expected_runtime_config": RUNTIME, "observed_runtime_config_digest": RUNTIME["runtime_config_digest"],
        "release_digest": "release", "checkpoint_digest": "sha256:checkpoint", "schema_version": SCHEMA_VERSION,
        "corpus_digest": "corpus", "payload_source": spec.payload_source, "item_indices": list(spec.item_indices),
        "per_endpoint_rate": spec.per_endpoint_rate, "drivers_per_endpoint": spec.drivers_per_endpoint,
        "wire_format": spec.wire_format, "per_driver_rate": spec.per_driver_rate, "scheduled_start_s": spec.scheduled_start_s, "actual_start_s": start,
        "offer_window_end_s": offer_end, "run_end_s": offer_end + .1, "actual_offered_rate_per_s": len(records) / (offer_end - start),
        "client_cpu": {"wall_s": 1., "process_cpu_s": cpu, "process_cpu_utilization_cores": cpu},
        "records": [record.to_dict() for record in records],
    }
    target = Path(spec.output_path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload))


def test_four_os_drivers_divide_32_rate_and_assign_200_unique_payloads(tmp_path):
    specs = _specs(tmp_path)
    assert len(specs) == 4
    assert {spec.per_driver_rate for spec in specs} == {8.0}
    assert sum(len(spec.item_indices) for spec in specs) == 200
    assert len({index for spec in specs for index in spec.item_indices}) == 200
    assert len({spec.child_id for spec in specs}) == 4


def test_aggregate_rejects_missing_driver_saturation_reuse_and_incomplete_interval(tmp_path):
    specs = _specs(tmp_path)
    _write_child(specs[0])
    with pytest.raises(ClientShardRunInvalid, match="absent"):
        aggregate_child_artifacts(specs)
    for spec in specs[1:]: _write_child(spec)
    aggregate = aggregate_child_artifacts(specs)
    assert aggregate.per_endpoint_rate == 32.0 and len(aggregate.records) == 200
    raw = json.loads(Path(specs[1].output_path).read_text()); raw["client_cpu"]["process_cpu_utilization_cores"] = CLIENT_SATURATION_CORES
    Path(specs[1].output_path).write_text(json.dumps(raw))
    with pytest.raises(ClientShardRunInvalid, match="saturated"):
        aggregate_child_artifacts(specs)
    _write_child(specs[1]); raw = json.loads(Path(specs[1].output_path).read_text())
    raw["records"][0]["payload_hash"] = json.loads(Path(specs[0].output_path).read_text())["records"][0]["payload_hash"]
    Path(specs[1].output_path).write_text(json.dumps(raw))
    with pytest.raises(ClientShardRunInvalid, match="assigned corpus"):
        aggregate_child_artifacts(specs)


def test_conditional_knee_rate_20_is_accepted_and_nonmember_is_rejected(tmp_path):
    spec = _specs(tmp_path)[0]
    assert 20.0 in SUPPORTED_PER_ENDPOINT_RATES
    assert replace(spec, per_endpoint_rate=20.0).per_endpoint_rate == 20.0
    with pytest.raises(ClientShardRunInvalid, match="per-endpoint rate"):
        replace(spec, per_endpoint_rate=18.0)


def test_envelope_wire_format_is_bound_into_shard_spec_and_runtime_digest(tmp_path):
    endpoint = ReplicaEndpoint("replica-gpu4", "http://127.0.0.1:31000/unidepth.infer", 4)
    runtime = expected_unidepth_runtime_config(batch_cap=16, batch_wait_ms=50, max_concurrent_forwards=1, wire_format="envelope")
    specs = build_client_shard_specs(
        experiment_id="saturation", endpoints=(endpoint,), expected_identities={endpoint.replica_id: _identity()},
        expected_runtime_configs={endpoint.replica_id: runtime}, payload_source=_source(tmp_path / "payload.json"),
        corpus_digest="corpus", scheduled_start_s=100.0, per_endpoint_rate=32.0, drivers_per_endpoint=4,
        output_dir=tmp_path / "out", wire_format="envelope",
    )
    assert {spec.wire_format for spec in specs} == {"envelope"}
    assert all(spec.expected_runtime_config["runtime_config"]["wire_format"] == "envelope" for spec in specs)
    assert all(spec.to_dict()["wire_format"] == "envelope" for spec in specs)
    assert all(spec.expected_runtime_config["runtime_config_digest"] != RUNTIME["runtime_config_digest"] for spec in specs)


def test_wrapper_threads_envelope_to_plan_and_shard_specs(tmp_path, monkeypatch):
    from scripts import run_unidepth_client_sharding_benchmark as command

    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    (payload_dir / "unidepth.infer.json").write_text(json.dumps({"schema": PAYLOAD_SOURCE_SCHEMA, "items": []}))
    captured: dict[str, object] = {}
    plan = SimpleNamespace(
        experiment_id="u2e1", endpoints=(), expected_server_identities={}, expected_runtime_configs={},
        application_release=SimpleNamespace(release_digest="release"),
    )

    def fake_plan(**kwargs):
        captured["plan_wire_format"] = kwargs["wire_format"]
        return plan

    def fake_specs(**kwargs):
        captured["shard_wire_format"] = kwargs["wire_format"]
        return ("spec",)

    class FakeSampler:
        def __init__(self, **kwargs):
            captured["sampler_release_digest"] = kwargs["release_digest"]

    def fake_run_parent(specs, **_kwargs):
        assert specs == ("spec",)
        return SimpleNamespace(offered_rate_per_s=1.0, completed_rate_per_s=1.0)

    monkeypatch.setattr(command, "build_unidepth_scaling_plan", fake_plan)
    monkeypatch.setattr(command, "build_client_shard_specs", fake_specs)
    monkeypatch.setattr(command, "NvmlSampler", FakeSampler)
    monkeypatch.setattr(command, "run_parent", fake_run_parent)
    assert command.main([
        "--experiment-id", "u2e1", "--gpus", "4", "--gpu-uuids", "GPU-4", "--run-root", str(tmp_path / "out"),
        "--application-release", str(tmp_path / "release"), "--source-sha", "source", "--checkpoint-digest", "checkpoint",
        "--payload-dir", str(payload_dir), "--per-endpoint-rate", "12", "--wire-format", "envelope",
    ]) == 0
    assert captured == {
        "plan_wire_format": "envelope", "shard_wire_format": "envelope", "sampler_release_digest": "release",
    }


def test_comparison_requires_same_actual_offer_and_concurrency(tmp_path):
    specs = _specs(tmp_path)
    for spec in specs: _write_child(spec)
    baseline = aggregate_child_artifacts(specs)
    assert compare_saturation_treatments(baseline, baseline)["schema"] == "ego.unidepth-saturation-comparison.v1"
    changed = replace(baseline, per_endpoint_rate=24.0)
    with pytest.raises(ClientShardRunInvalid, match="configured endpoint rate"):
        compare_saturation_treatments(baseline, changed)


def test_pinned_gateway_rejects_runtime_config_mismatch_and_worker_replacement():
    request = type("Request", (), {"ownership": type("O", (), {"request_id": "request"})()})()
    identity = _identity()
    trace = BatchTrace("batch", identity.replica_id, 1., 1., 1., 1.1, 1, 1, 1, 1)
    def response(actual, runtime):
        return GatewayResponse(ownership=request.ownership, result=GatewayResult(request.ownership, {"server_identity": actual.to_wire(), "batch_diagnostics": runtime}, {}, trace), attempts=1, last_status_code=200, replica_id=actual.replica_id)
    class Raw:
        def __init__(self, replies): self.replies = replies
        async def call(self, _): return self.replies.pop(0)
        async def aclose(self): pass
    diagnostics = {"runtime_config": RUNTIME["runtime_config"], "runtime_config_digest": RUNTIME["runtime_config_digest"]}
    mismatch = {"runtime_config": expected_unidepth_runtime_config(batch_cap=8, batch_wait_ms=20, max_concurrent_forwards=1)["runtime_config"], "runtime_config_digest": "wrong"}
    assert asyncio.run(_PinnedIdentityGateway(Raw([response(identity, mismatch)]), identity, RUNTIME).call(request)).error.code is ErrorCode.VALIDATION
    changed = replace(identity, worker_pid=999)
    gate = _PinnedIdentityGateway(Raw([response(identity, diagnostics), response(changed, diagnostics)]), identity, RUNTIME)
    assert asyncio.run(gate.call(request)).result is not None
    assert asyncio.run(gate.call(request)).error.code is ErrorCode.VALIDATION
