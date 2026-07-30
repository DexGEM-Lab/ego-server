"""GPU-free tests for the isolated UniDepth scaling experiment contract."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
from pathlib import Path

import pytest

from ego_annotation.serving.benchmark.generator import OfferedLevel, OpenLoopGenerator
from ego_annotation.serving.benchmark.manifest import build_synthetic_unidepth_manifest
from ego_annotation.serving.benchmark.metrics import ItemRecord
from ego_annotation.serving.benchmark.unidepth_scaling import (
    EXPERIMENT_TEMP_ROOT,
    CommandOutcome,
    ExperimentConfigurationError,
    GeneratorCpuUsage,
    PhysicalBatchConflictError,
    ReplicaEndpoint,
    StatelessReplicaGateway,
    aggregate_scaling_level,
    build_unidepth_scaling_plan,
    compare_replica_scaling,
    execute_scoped_plan,
    physical_batches_from_records,
    stop_scoped_experiment,
    profiler_attribution_status,
    validate_gpu_measurement_artifact,
    validate_server_identity,
    _candidate_process_pids,
)
from ego_annotation.serving.contracts import BatchTrace, ErrorCode, SCHEMA_VERSION, ServerIdentity, ServiceError
from ego_annotation.serving.gateway import GatewayRequest, GatewayResponse, GatewayResult
from ego_annotation.serving.lifecycle import COMMITTED_GPU_GROUPS
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


SOURCE_SHA = "a" * 40
RELEASE_SHA = "b" * 40


def _release_kwargs(tmp_path: Path) -> dict[str, object]:
    release = tmp_path / "release-a"
    (release / "ego_annotation" / "serving").mkdir(parents=True, exist_ok=True)
    (release / "ego_annotation" / "serving" / "deployment.py").write_text("# pinned test release\n")
    (release / "RELEASE.json").write_text(
        '{"release_sha": "' + RELEASE_SHA + '", "source_sha": "' + SOURCE_SHA + '"}\n'
    )
    return {"application_release_path": release, "source_sha": SOURCE_SHA, "checkpoint_digest": "sha256:test-checkpoint"}


@dataclass
class RecordingGateway:
    name: str
    fail: bool = False
    calls: list[str] | None = None

    async def call(self, request: GatewayRequest) -> GatewayResponse:
        assert self.calls is not None
        self.calls.append(request.ownership.request_id)
        if self.fail:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.BACKPRESSURE, "intentional endpoint rejection", retryable=True, ownership=request.ownership
                ),
                attempts=1,
                last_status_code=429,
            )
        return GatewayResponse(
            ownership=request.ownership,
            error=ServiceError(ErrorCode.TRANSPORT, "intentional transport outcome", retryable=True, ownership=request.ownership),
            attempts=1,
        )

    async def aclose(self) -> None:
        return None


def _identity_for(replica_id: str, *, experiment_id: str = "scale-a", gpu_id: int = 4) -> ServerIdentity:
    return ServerIdentity(
        experiment_id=experiment_id, replica_id=replica_id, assigned_gpu=gpu_id, worker_pid=123,
        gcs_address="127.0.0.1:29000", http_port=31000,
        temp_dir=str(EXPERIMENT_TEMP_ROOT / experiment_id / f"gpu{gpu_id}"),
        model_revision="unidepth-v2-vitl14-corrected", checkpoint_digest="sha256:test-checkpoint",
        schema_version=SCHEMA_VERSION, release_sha=RELEASE_SHA,
    )


def _record(*, item_id: str, replica_id: str, batch_id: str | None, outcome: str = "completed", submit_time_s: float = 1.0) -> ItemRecord:
    return ItemRecord(
        item_id=item_id,
        api_name="unidepth.infer",
        request_id=f"request-{item_id}",
        job_id="job-a",
        work_units=1,
        payload_hash=f"hash-{item_id}",
        source_timestamp_s=0.0,
        offer_time_s=submit_time_s,
        submit_time_s=submit_time_s,
        response_time_s=1.0,
        offered_delay_s=0.0,
        outcome=outcome,
        http_status=200 if outcome == "completed" else 429,
        attempts=1,
        error_code=None if outcome == "completed" else "backpressure",
        error_message=None,
        response_latency_ms=10.0 if outcome == "completed" else None,
        transport_ms=1.0,
        admission_ms=1.0,
        queue_ms=2.0,
        dispatch_ms=1.0,
        forward_ms=5.0,
        encoding_ms=1.0,
        batch_id=batch_id,
        batch_size=2 if batch_id else None,
        batch_work_units=2 if batch_id else None,
        batch_wall_ms=10.0 if batch_id else None,
        amortized_cost_ms=5.0 if batch_id else None,
        model_load_count=1 if batch_id else None,
        replica_id=replica_id,
        allocator_allocated_bytes=100 if batch_id else None,
        allocator_reserved_bytes=200 if batch_id else None,
        allocator_max_allocated_bytes=120 if batch_id else None,
        allocator_max_reserved_bytes=240 if batch_id else None,
    )


def test_experiment_port_temp_cuda_contract_is_disjoint_and_does_not_mutate_production(tmp_path):
    before = tuple(
        (group.gpu_id, group.lifecycle.temp_dir, group.lifecycle.ports.all_ports(), group.lifecycle.env_vars)
        for group in COMMITTED_GPU_GROUPS
    )
    plan = build_unidepth_scaling_plan(
        experiment_id="scale-a",
        gpu_ids=(5, 7),
        run_root="/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks",
        **_release_kwargs(tmp_path),
    )
    after = tuple(
        (group.gpu_id, group.lifecycle.temp_dir, group.lifecycle.ports.all_ports(), group.lifecycle.env_vars)
        for group in COMMITTED_GPU_GROUPS
    )
    assert after == before, "experiment planning must not mutate canonical production topology"
    plan.assert_isolated()
    experiment_ports = {port for replica in plan.replicas for port in replica.lifecycle.ports.all_ports()}
    production_ports = {port for group in COMMITTED_GPU_GROUPS for port in group.lifecycle.ports.all_ports()}
    assert experiment_ports.isdisjoint(production_ports)
    assert all(Path(replica.lifecycle.temp_dir).is_relative_to(EXPERIMENT_TEMP_ROOT) for replica in plan.replicas)
    assert all(f"CUDA_VISIBLE_DEVICES={replica.gpu_id}" in replica.launch_commands()[0] for replica in plan.replicas)
    assert all("EGO_UNIDEPTH_REPLICA_ID=" in replica.launch_commands()[1] for replica in plan.replicas)
    assert all("EGO_UNIDEPTH_EXPERIMENT_TELEMETRY=1" in replica.launch_commands()[1] for replica in plan.replicas)
    assert all("EGO_UNIDEPTH_EXPERIMENT_WIRE_FORMAT=multipart" in replica.launch_commands()[1] for replica in plan.replicas)
    assert all(replica.expected_runtime_config["runtime_config"]["wire_format"] == "multipart" for replica in plan.replicas)
    assert all("runtime/current" not in " ".join(replica.launch_commands()) for replica in plan.replicas)
    assert all("ray stop" not in " ".join(replica.stop_commands()) for replica in plan.replicas)
    assert all("--app-choice unidepth" in replica.launch_commands()[1] for replica in plan.replicas)
    assert all("shutdown -a http://127.0.0.1:" in replica.stop_commands()[0] for replica in plan.replicas)


def test_plan_attests_envelope_wire_format_in_worker_environment_and_digest(tmp_path):
    multipart = build_unidepth_scaling_plan(
        experiment_id="wire-multipart", gpu_ids=(5,), run_root="/tmp/results", **_release_kwargs(tmp_path),
    )
    envelope = build_unidepth_scaling_plan(
        experiment_id="wire-envelope", gpu_ids=(5,), run_root="/tmp/results", wire_format="envelope", **_release_kwargs(tmp_path),
    )
    replica = envelope.replicas[0]
    assert "EGO_UNIDEPTH_EXPERIMENT_WIRE_FORMAT=envelope" in replica.launch_commands()[1]
    assert replica.expected_runtime_config["runtime_config"]["wire_format"] == "envelope"
    assert replica.expected_runtime_config["runtime_config_digest"] != multipart.replicas[0].expected_runtime_config["runtime_config_digest"]
    with pytest.raises(ExperimentConfigurationError, match="wire_format"):
        build_unidepth_scaling_plan(
            experiment_id="wire-invalid", gpu_ids=(5,), run_root="/tmp/results", wire_format="invalid", **_release_kwargs(tmp_path),
        )


def test_plan_refuses_temp_dir_too_long_for_ray_unix_socket(tmp_path):
    with pytest.raises(ExperimentConfigurationError, match="AF_UNIX"):
        build_unidepth_scaling_plan(
            experiment_id="x" * 25, gpu_ids=(5,), run_root="/tmp/results",
            **_release_kwargs(tmp_path),
        )


@pytest.mark.parametrize("gpu_ids", [(0,), (4,), (6,), (5, 5)])
def test_plan_refuses_production_or_duplicate_gpu_ownership(gpu_ids, tmp_path):
    with pytest.raises(ExperimentConfigurationError):
        build_unidepth_scaling_plan(experiment_id="bad", gpu_ids=gpu_ids, run_root="/tmp/results", **_release_kwargs(tmp_path))


def test_experiment_endpoint_refuses_a_production_lane_url(tmp_path):
    with pytest.raises(ExperimentConfigurationError, match="production-reserved port"):
        ReplicaEndpoint("bad-url", "http://127.0.0.1:28000/unidepth.infer", 5)
    with pytest.raises(ExperimentConfigurationError):
        build_unidepth_scaling_plan(
            experiment_id="bad-port", gpu_ids=(5,), run_root="/tmp/results", serve_port_base=28000,
            **_release_kwargs(tmp_path),
        )
    with pytest.raises(ExperimentConfigurationError):
        build_unidepth_scaling_plan(
            experiment_id="bad-component-port", gpu_ids=(5,), run_root="/tmp/results", component_port_base=26000,
            **_release_kwargs(tmp_path),
        )


def test_stateless_balancer_balances_requests_and_preserves_payload_ownership():
    manifest = build_synthetic_unidepth_manifest(manifest_id="balance", count=6)
    calls_a: list[str] = []
    calls_b: list[str] = []
    gateways = iter((RecordingGateway("a", calls=calls_a), RecordingGateway("b", calls=calls_b)))
    router = ModelServiceRouter.canonical()
    scaling_gateway = StatelessReplicaGateway(
        api_name=ModelApiName.UNIDEPTH_INFER,
        base_router=router,
        endpoints=(
            ReplicaEndpoint("replica-a", "http://127.0.0.1:31000/unidepth.infer", 5),
            ReplicaEndpoint("replica-b", "http://127.0.0.1:31001/unidepth.infer", 7),
        ),
        gateway_factory=lambda router: next(gateways),
    )

    async def scenario():
        return await scaling_gateway.call_batch([item.to_gateway_request() for item in manifest.items])

    responses = asyncio.run(scenario())
    assert [response.replica_id for response in responses] == ["replica-a", "replica-b"] * 3
    assert calls_a == [manifest.items[index].ownership.request_id for index in (0, 2, 4)]
    assert calls_b == [manifest.items[index].ownership.request_id for index in (1, 3, 5)]
    assert [response.ownership for response in responses] == [item.ownership for item in manifest.items]
    assert len({item.payload_hash for item in manifest.items}) == len(manifest.items)


def test_open_loop_scaling_gateway_surfaces_endpoint_rejection_with_replica_identity():
    manifest = build_synthetic_unidepth_manifest(manifest_id="reject", count=3)
    router = ModelServiceRouter.canonical()
    gateway = StatelessReplicaGateway(
        api_name=ModelApiName.UNIDEPTH_INFER,
        base_router=router,
        endpoints=(ReplicaEndpoint("replica-reject", "http://127.0.0.1:31000/unidepth.infer", 5),),
        gateway_factory=lambda router: RecordingGateway("reject", fail=True, calls=[]),
    )

    async def scenario():
        level = OfferedLevel(api_name=ModelApiName.UNIDEPTH_INFER, offered_intensity_per_s=1000.0, max_offered=3)
        return await OpenLoopGenerator(gateway).run_level(manifest, level)

    result = asyncio.run(scenario())
    assert [record.outcome for record in result.records] == ["backpressure"] * 3
    assert {record.replica_id for record in result.records} == {"replica-reject"}
    assert {record.request_id for record in result.records} == {item.ownership.request_id for item in manifest.items}


def test_result_aggregation_deduplicates_physical_batches_and_preserves_allocator_cpu_and_failures():
    records = [
        _record(item_id="a", replica_id="replica-a", batch_id="batch-a"),
        _record(item_id="b", replica_id="replica-a", batch_id="batch-a"),
        _record(item_id="c", replica_id="replica-b", batch_id="batch-b"),
        _record(item_id="d", replica_id="replica-b", batch_id=None, outcome="backpressure"),
    ]
    result = aggregate_scaling_level(
        records,
        api_name="unidepth.infer",
        configured_offered_intensity_per_s=4.0,
        run_start_s=0.0, offer_window_end_s=1.0, run_end_s=2.0,
        measurement_interval_id="open-loop-observation-v1", expected_replica_ids=("replica-a", "replica-b"),
        generator_cpu=GeneratorCpuUsage(wall_s=2.0, process_cpu_s=0.4, process_cpu_utilization_cores=0.2),
        release_digest="release", corpus_digest="corpus", measurement_definition="open-loop-v1",
    )
    assert result.aggregate.completed_count == 3
    assert result.aggregate.rejected_count == 1
    assert set(result.per_replica) == {"replica-a", "replica-b"}
    assert len(result.physical_batches) == 2, "two items in batch-a must not become two physical batches"
    assert result.physical_batches[0].allocator_reserved_bytes == 200
    assert result.generator_cpu.process_cpu_utilization_cores == 0.2
    rendered = result.to_dict()
    assert rendered["schema"] == "ego.unidepth-scaling-level.v2"
    assert rendered["offer_window_duration_s"] == 1.0
    assert rendered["drain_duration_s"] == 1.0
    baseline = aggregate_scaling_level(
        [
            _record(item_id="baseline-a", replica_id="replica-baseline", batch_id="baseline-batch"),
            _record(item_id="baseline-b", replica_id="replica-baseline", batch_id="baseline-batch"),
        ],
        api_name="unidepth.infer",
        configured_offered_intensity_per_s=2.0,
        run_start_s=0.0, offer_window_end_s=1.0, run_end_s=2.0,
        measurement_interval_id="open-loop-observation-v1", expected_replica_ids=("replica-baseline",),
        generator_cpu=GeneratorCpuUsage(wall_s=2.0, process_cpu_s=0.2, process_cpu_utilization_cores=0.1),
        release_digest="release", corpus_digest="corpus", measurement_definition="open-loop-v1",
    )
    comparison = compare_replica_scaling(baseline, result)
    assert comparison.aggregate_throughput_gain == 1.5
    assert comparison.to_dict()["schema"] == "ego.unidepth-replica-scaling-comparison.v1"


def test_conflicting_duplicate_batch_trace_is_surfaced_not_merged():
    first = _record(item_id="a", replica_id="replica-a", batch_id="same")
    second = _record(item_id="b", replica_id="replica-b", batch_id="same")
    with pytest.raises(PhysicalBatchConflictError):
        physical_batches_from_records([first, second])


def test_candidate_scan_excludes_scoped_stop_shell_ancestors(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    own_pid = os.getpid()
    parent_pid = 123
    target = str(EXPERIMENT_TEMP_ROOT / "scan" / "gpu4")

    def process(pid: int, ppid: int, command: bytes, environment: bytes = b"") -> None:
        root = proc / str(pid)
        root.mkdir()
        (root / "stat").write_text(f"{pid} (cmd) S {ppid} 0 0 0\n")
        (root / "cmdline").write_bytes(command)
        (root / "environ").write_bytes(environment)

    process(own_pid, parent_pid, b"python\0-m\0scoped-stop")
    process(parent_pid, 1, f"/bin/sh\0-c\0scoped-stop --temp-dir {target}".encode())
    process(456, 1, f"raylet --temp-dir={target}".encode())
    process(457, 1, b"dashboard-agent", f"EGO_EXPERIMENT_TEMP_DIR={target}\0".encode())
    assert _candidate_process_pids(target, proc_root=proc) == [456, 457]


def test_scoped_stop_matches_only_exact_experiment_temp_dir(tmp_path):
    plan = build_unidepth_scaling_plan(experiment_id="stop-test", gpu_ids=(5,), run_root="/tmp/results", **_release_kwargs(tmp_path))
    temp_dir = plan.replicas[0].lifecycle.temp_dir
    lookups: list[str] = []
    killed: list[tuple[int, int]] = []

    def lookup(value: str):
        lookups.append(value)
        return [101, 102] if len(lookups) == 1 else [102]

    stopped = stop_scoped_experiment(temp_dir, pid_lookup=lookup, kill=lambda pid, sig: killed.append((pid, sig)))
    assert stopped == (101, 102)
    assert all(value == str(Path(temp_dir).resolve()) for value in lookups)
    assert [pid for pid, _ in killed] == [101, 102, 102]
    with pytest.raises(ExperimentConfigurationError):
        stop_scoped_experiment("/tmp/ray-ego-serve-gpu0", pid_lookup=lambda _: [])
    with pytest.raises(ExperimentConfigurationError):
        stop_scoped_experiment(EXPERIMENT_TEMP_ROOT / "stop-test", pid_lookup=lambda _: [])


def test_immutable_release_rejects_mutable_current_and_wrong_source_sha(tmp_path):
    release = _release_kwargs(tmp_path)["application_release_path"]
    with pytest.raises(ExperimentConfigurationError, match="runtime/current"):
        build_unidepth_scaling_plan(
            experiment_id="bad-release", gpu_ids=(5,), run_root="/tmp/results",
            application_release_path=Path(release).parent / "current", source_sha=SOURCE_SHA,
            checkpoint_digest="sha256:test-checkpoint",
        )
    with pytest.raises(ExperimentConfigurationError, match="source_sha"):
        build_unidepth_scaling_plan(
            experiment_id="bad-source", gpu_ids=(5,), run_root="/tmp/results",
            application_release_path=release, source_sha="c" * 40, checkpoint_digest="sha256:test-checkpoint",
        )


def test_gateway_preserves_server_trace_replica_identity():
    from ego_annotation.serving.gateway import _gateway_response_from_multipart
    trace = BatchTrace("batch", "server-replica", 1.0, 1.0, 1.0, 1.1, 1, 1, 1, 1)
    ownership = build_synthetic_unidepth_manifest(manifest_id="trace", count=1).items[0].ownership
    response = _gateway_response_from_multipart({"result": {"trace": trace.to_wire()}}, {}, ownership, 1, 200, 1.0)
    assert response.replica_id == "server-replica"
    assert response.result.trace.replica_id == "server-replica"


def test_server_identity_accepts_equivalent_nvml_and_torch_uuid_formats():
    expected = replace(_identity_for("replica-a"), cuda_uuid="GPU-0b337756-c84d-f655-32f1-cd7a209d2c0e")
    actual = replace(expected, cuda_uuid="0b337756-c84d-f655-32f1-cd7a209d2c0e", worker_pid=456)
    trace = BatchTrace("batch", "replica-a", 1.0, 1.0, 1.0, 1.1, 1, 1, 1, 1)
    ownership = build_synthetic_unidepth_manifest(manifest_id="uuid", count=1).items[0].ownership
    response = GatewayResponse(
        ownership=ownership,
        result=GatewayResult(ownership, {"server_identity": actual.to_wire()}, {}, trace),
        replica_id="replica-a",
    )
    assert validate_server_identity(expected, response).worker_pid == 456


def test_spoofed_server_identity_is_rejected_and_gateway_preserves_trace_identity():
    manifest = build_synthetic_unidepth_manifest(manifest_id="identity", count=1)
    expected = _identity_for("replica-a")
    spoofed = _identity_for("replica-b")
    trace = BatchTrace("batch", "replica-b", 1.0, 1.0, 1.0, 1.1, 1, 1, 1, 1)

    @dataclass
    class SpoofGateway:
        async def call(self, request):
            return GatewayResponse(
                ownership=request.ownership,
                result=GatewayResult(request.ownership, {"server_identity": spoofed.to_wire()}, {}, trace),
                replica_id="replica-b",
            )

    gateway = StatelessReplicaGateway(
        api_name=ModelApiName.UNIDEPTH_INFER, base_router=ModelServiceRouter.canonical(),
        endpoints=(ReplicaEndpoint("replica-a", "http://127.0.0.1:31000/unidepth.infer", 5),),
        gateway_factory=lambda router: SpoofGateway(), expected_server_identities={"replica-a": expected},
    )
    response = asyncio.run(gateway.call(manifest.items[0].to_gateway_request()))
    assert response.error is not None and response.error.code is ErrorCode.VALIDATION
    assert response.replica_id is None, "a rejected identity must not be replaced by the selected endpoint id"


def test_scoped_execute_rolls_back_only_started_experiment_scopes(tmp_path):
    plan = build_unidepth_scaling_plan(experiment_id="rollback", gpu_ids=(5, 7), run_root="/tmp/results", **_release_kwargs(tmp_path))

    class Runner:
        def __init__(self):
            self.commands: list[str] = []
        def run(self, command):
            self.commands.append(command)
            deploys = [c for c in self.commands if "ray.serve.scripts run" in c]
            return CommandOutcome(command, 1 if "ray.serve.scripts run" in command and len(deploys) == 2 else 0)

    runner = Runner()
    with pytest.raises(ExperimentConfigurationError, match="deploy failed"):
        execute_scoped_plan(plan, command_runner=runner, typed_readiness_probe=lambda _: None)
    assert all("ray stop" not in command for command in runner.commands)
    assert all("ray-ego-serve-gpu0" not in command for command in runner.commands)
    # Both started heads are scoped cleanup targets; production never is.
    assert sum("--scoped-stop" in command for command in runner.commands) == 2


def test_experimental_cpu_diagnostics_have_explicit_unavailable_cuda_fields():
    import numpy as np
    from ego_annotation.serving.unidepth import UniDepthAdapter, build_unidepth_model_config

    manifest = build_synthetic_unidepth_manifest(manifest_id="telemetry", count=1)
    request = manifest.items[0].to_gateway_request()
    from ego_annotation.serving.contracts import UniDepthRequest, TensorPayload, SpatialMetadata
    parsed = UniDepthRequest(
        ownership=request.ownership, rgb=TensorPayload(request.parts[0].data, request.parts[0].shape, request.parts[0].dtype),
        spatial=request.spatial, model_revision=request.model_revision,
    )
    config = build_unidepth_model_config(
        checkpoint="test", model_revision=parsed.model_revision, device="cpu", performance_instrumentation=True,
        canonical_height=parsed.rgb.shape[0], canonical_width=parsed.rgb.shape[1],
    )
    class Backend:
        def infer(self, _):
            h, w = config.canonical_height, config.canonical_width
            return {"depth": np.ones((1, 1, h, w), np.float32), "intrinsics": np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
                    "confidence": np.ones((1, 1, h, w), np.float32)}
    response = asyncio.run(UniDepthAdapter(config, backend_factory=lambda _: Backend()).infer(parsed))
    diagnostics = response.result.batch_diagnostics
    assert diagnostics["availability"] == "unavailable_cpu"
    assert diagnostics["cpu_collate_ms"] is not None
    assert diagnostics["validation_ms"] is not None and diagnostics["encoding_ms"] is not None
    assert diagnostics["h2d_ms"] is diagnostics["cuda_model_ms"] is diagnostics["d2h_ms"] is None
    allocator = diagnostics["allocator_memory"]
    assert allocator == {"allocated_bytes": None, "reserved_bytes": None, "max_allocated_bytes": None, "max_reserved_bytes": None}
    assert diagnostics["runtime_config"]["batch_policy"]["max_batch_size"] == 8
    assert diagnostics["runtime_config"]["max_concurrent_forwards"] is None
    assert diagnostics["peak_simultaneous_forwards"] == 1


def test_forward_semaphore_bounds_physical_backend_overlap():
    """Two concurrent Serve callbacks may queue, but c1 permits one backend forward."""
    import threading
    import numpy as np
    from ego_annotation.serving.unidepth import UniDepthAdapter, build_unidepth_model_config
    from ego_annotation.serving.contracts import UniDepthRequest, TensorPayload

    manifest = build_synthetic_unidepth_manifest(manifest_id="c1", count=2)
    parsed = []
    for item in manifest.items:
        request = item.to_gateway_request()
        parsed.append(UniDepthRequest(ownership=request.ownership,
            rgb=TensorPayload(request.parts[0].data, request.parts[0].shape, request.parts[0].dtype),
            spatial=request.spatial, model_revision=request.model_revision))
    started, release, lock = threading.Event(), threading.Event(), threading.Lock()
    active = [0]; peak = [0]
    config = build_unidepth_model_config(checkpoint="test", model_revision=parsed[0].model_revision, device="cpu",
        performance_instrumentation=True, canonical_height=parsed[0].rgb.shape[0], canonical_width=parsed[0].rgb.shape[1],
        max_concurrent_forwards=1)
    class Backend:
        def infer(self, rgb):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0]); started.set()
            release.wait(timeout=1)
            with lock: active[0] -= 1
            b, _, h, w = rgb.shape
            return {"depth": np.ones((b, 1, h, w), np.float32), "intrinsics": np.tile(np.eye(3, dtype=np.float32), (b, 1, 1)),
                    "confidence": np.ones((b, 1, h, w), np.float32)}
    adapter = UniDepthAdapter(config, backend_factory=lambda _: Backend())
    async def run():
        first, second = asyncio.create_task(adapter.infer(parsed[0])), asyncio.create_task(adapter.infer(parsed[1]))
        assert await asyncio.to_thread(started.wait, 1)
        assert peak == [1]
        release.set()
        return await asyncio.gather(first, second)
    responses = asyncio.run(run())
    assert peak == [1] and all(response.result is not None for response in responses)
    assert responses[-1].result.batch_diagnostics["peak_simultaneous_forwards"] == 1


def test_gpu_measurement_artifact_is_required_and_bandwidth_needs_ncu_or_cupti(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ExperimentConfigurationError, match="required"):
        validate_gpu_measurement_artifact(missing, gpu_ids=(5,))
    nvml = tmp_path / "nvml.json"
    nvml.write_text('{"schema":"ego.gpu-samples.v1","samples":[{"gpu_id":5,"timestamp_s":1.0,"utilization_gpu_pct":30,"memory_used_bytes":100}]}')
    assert validate_gpu_measurement_artifact(nvml, gpu_ids=(5,))["schema"] == "ego.gpu-samples.v1"
    assert profiler_attribution_status(None) == "unavailable_without_ncu_or_cupti_artifact"
    ncu = tmp_path / "ncu.json"
    ncu.write_text('{"tool":"ncu","kernels":[]}')
    assert profiler_attribution_status(ncu) == "available_from_ncu"
