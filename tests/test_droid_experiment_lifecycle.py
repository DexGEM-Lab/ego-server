"""GPU-free adversarial tests for isolated DROID lifecycle and worker identity."""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import ego_annotation.serving.benchmark.droid_scaling as droid_scaling

from ego_annotation.serving.benchmark.droid_scaling import (
    DROID_EXPERIMENT_TEMP_ROOT,
    DroidTypedCall,
    ExperimentConfigurationError,
    build_droid_scaling_plan,
    droid_preflight_check_command,
    execute_droid_scoped_plan,
    stop_droid_scoped_experiment,
    typed_readiness_sequence,
    validate_droid_server_identity,
)
from ego_annotation.serving.benchmark.release import (
    WorkerRuntimeEvidence,
    artifact_digest,
    build_release,
    derive_worker_runtime_evidence,
)
from ego_annotation.serving.benchmark.unidepth_scaling import CommandOutcome
from ego_annotation.serving.contracts import (
    DroidBatchTrace,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    DroidImageShape,
    DroidSessionOptions,
    ErrorCode,
    FrameValidity,
    ImageSize,
    Ownership,
    PixelTransform,
    ServerIdentity,
    ServiceError,
    StepStatus,
    TensorPayload,
)
from ego_annotation.serving.lifecycle import droid_gpu_group
from ego_annotation.serving.droid_source import VerifiedDroidSourceRelease

SOURCE_SHA = "d" * 40
SOURCE_RELEASE = "/verified-droid-source"


@pytest.fixture(autouse=True)
def _verified_amended_source(monkeypatch, tmp_path):
    root = tmp_path / "verified-droid-source" / ("a" * 64)
    source_root = root / "droid_slam"
    source_root.mkdir(parents=True)
    verified = VerifiedDroidSourceRelease(
        root, source_root, "a" * 64, "recovered-hawor-droid-core-v1", "523bc9b92a8f11f3dfd061efd86d25c5b687d7983592cf5f5bc3f45b6ebaa9c6", (),
    )
    monkeypatch.setattr(droid_scaling, "verify_droid_source_release", lambda _path: verified)
    return verified


def _release(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    serving = source / "ego_annotation" / "serving"
    serving.mkdir(parents=True)
    (source / "ego_annotation" / "__init__.py").write_text("")
    (serving / "__init__.py").write_text("")
    (serving / "deployment.py").write_text("app = object()\n")
    (serving / "droid_deployment.py").write_text("app = object()\n")
    (serving / "droid.py").write_text("VALUE = 1\n")
    return build_release(source, tmp_path / "releases", source_sha=SOURCE_SHA)


def _plan(tmp_path: Path, gpus=(7,)):
    return build_droid_scaling_plan(
        experiment_id="droid-hardening",
        gpu_ids=gpus,
        gpu_uuids=tuple(f"GPU-{gpu}" for gpu in gpus),
        run_root=tmp_path / "runs",
        application_release_path=_release(tmp_path),
        droid_source_release_path=SOURCE_RELEASE,
        source_sha=SOURCE_SHA,
        checkpoint_digest="actual-checkpoint-digest",
        corpus_digest="corpus-digest", measurement_interval_id="interval-a",
    )


def test_gpu7_allowed_and_production_or_unapproved_gpus_rejected(tmp_path):
    plan = _plan(tmp_path)
    assert plan.replicas[0].gpu_id == 7
    for gpu in (0, 1, 2, 3, 6, 8):
        with pytest.raises(ExperimentConfigurationError, match="authorized"):
            build_droid_scaling_plan(
                experiment_id="bad", gpu_ids=(gpu,), gpu_uuids=(f"GPU-{gpu}",),
                run_root=tmp_path / "runs", application_release_path=plan.application_release.path,
                droid_source_release_path=SOURCE_RELEASE,
                source_sha=SOURCE_SHA, checkpoint_digest="digest",
                corpus_digest="corpus", measurement_interval_id="interval",
            )


def test_current_release_invalid_ports_and_temp_root_are_rejected(tmp_path):
    release = _release(tmp_path)
    current = tmp_path / "current"
    current.symlink_to(release, target_is_directory=True)
    with pytest.raises(ExperimentConfigurationError, match="immutable|current"):
        build_droid_scaling_plan(
            experiment_id="bad", gpu_ids=(7,), gpu_uuids=("GPU-7",), run_root=tmp_path,
            application_release_path=current, droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA, checkpoint_digest="digest",
            corpus_digest="corpus", measurement_interval_id="interval",
        )
    with pytest.raises(ExperimentConfigurationError, match="temp root"):
        build_droid_scaling_plan(
            experiment_id="bad", gpu_ids=(7,), gpu_uuids=("GPU-7",), run_root=tmp_path,
            application_release_path=release, droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA, checkpoint_digest="digest",
            corpus_digest="corpus", measurement_interval_id="interval", temp_root=tmp_path / "other",
        )
    with pytest.raises(ExperimentConfigurationError, match="1..65535"):
        build_droid_scaling_plan(
            experiment_id="bad", gpu_ids=(7,), gpu_uuids=("GPU-7",), run_root=tmp_path,
            application_release_path=release, droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA, checkpoint_digest="digest",
            corpus_digest="corpus", measurement_interval_id="interval", serve_port_base=70000,
        )
    with pytest.raises(ExperimentConfigurationError, match="ephemeral"):
        build_droid_scaling_plan(
            experiment_id="bad", gpu_ids=(7,), gpu_uuids=("GPU-7",), run_root=tmp_path,
            application_release_path=release, droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA, checkpoint_digest="digest",
            corpus_digest="corpus", measurement_interval_id="interval",
            component_port_base=34000, worker_port_base=34100, serve_port_base=36000,
        )


def test_droid_plan_uses_exact_abi_checkpoint_revision_and_explicit_droid_app(tmp_path):
    replica = _plan(tmp_path).replicas[0]
    production = droid_gpu_group().lifecycle
    env = dict(replica.lifecycle.env_vars)
    production_env = dict(production.env_vars)
    assert replica.lifecycle.interpreter == "/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python"
    for name in ("EGO_DROID_WEIGHTS", "EGO_DROID_REVISION", "EGO_DROID_DEVICE"):
        assert env[name] == production_env[name]
    assert env["EGO_DROID_REPO"].endswith("/droid_slam")
    assert env["EGO_DROID_SOURCE_DIGEST"] == "a" * 64
    assert env["EGO_DROID_SOURCE_AMENDMENT"] == "recovered-hawor-droid-core-v1"
    assert env["EGO_DROID_MAX_SESSIONS"] == "16"
    assert env["EGO_DROID_CPU_OFFLOAD"] == "0"
    assert env["EGO_DROID_EXPERIMENT_TELEMETRY"] == "1"
    start, deploy = replica.launch_commands()
    assert "CUDA_VISIBLE_DEVICES=7" in start
    assert "--app-choice droid" in deploy
    assert "ego_annotation.serving.deployment:app" not in deploy
    assert "runtime/current" not in deploy


def test_cpu_offload_and_max_sessions_are_explicit_in_worker_environment_and_launch_configuration(tmp_path):
    resident = _plan(tmp_path)
    offload = build_droid_scaling_plan(
        experiment_id="droid-offload",
        gpu_ids=(7,),
        gpu_uuids=("GPU-7",),
        run_root=tmp_path / "runs",
        application_release_path=resident.application_release.path,
        droid_source_release_path=SOURCE_RELEASE,
        source_sha=SOURCE_SHA,
        checkpoint_digest="actual-checkpoint-digest",
        corpus_digest="corpus-digest",
        measurement_interval_id="interval-a",
        cpu_offload=True,
        max_sessions=64,
    )
    assert dict(resident.replicas[0].lifecycle.env_vars)["EGO_DROID_CPU_OFFLOAD"] == "0"
    assert dict(offload.replicas[0].lifecycle.env_vars)["EGO_DROID_CPU_OFFLOAD"] == "1"
    assert dict(resident.replicas[0].lifecycle.env_vars)["EGO_DROID_MAX_SESSIONS"] == "16"
    assert dict(offload.replicas[0].lifecycle.env_vars)["EGO_DROID_MAX_SESSIONS"] == "64"
    assert resident.launch_configuration["cpu_offload"] is False
    assert offload.launch_configuration["cpu_offload"] is True
    assert resident.launch_configuration["max_sessions"] == 16
    assert offload.launch_configuration["max_sessions"] == 64
    assert resident.launch_configuration_digest != offload.launch_configuration_digest


def test_droid_plan_rejects_non_positive_max_sessions(tmp_path):
    resident = _plan(tmp_path)
    with pytest.raises(ExperimentConfigurationError, match="max_sessions"):
        build_droid_scaling_plan(
            experiment_id="droid-invalid-limit",
            gpu_ids=(7,), gpu_uuids=("GPU-7",), run_root=tmp_path / "runs",
            application_release_path=resident.application_release.path,
            droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA,
            checkpoint_digest="actual-checkpoint-digest", corpus_digest="corpus-digest",
            measurement_interval_id="interval-a", max_sessions=0,
        )


def test_droid_plan_retains_short_temp_pyc_exclusion_and_environment_owned_cleanup(tmp_path):
    from ego_annotation.serving.benchmark.droid_scaling import cleanup_verification_command

    replica = _plan(tmp_path).replicas[0]
    assert str(replica.lifecycle.temp_dir).startswith("/tmp/zjheds/")
    assert len(str(replica.lifecycle.temp_dir).encode()) <= 36
    assert dict(replica.lifecycle.env_vars)["PYTHONDONTWRITEBYTECODE"] == "1"
    assert max(replica.lifecycle.ports.all_ports()) < 32768
    preflight = droid_preflight_check_command(replica)
    assert "mkdir -p /tmp/zjheds" in preflight
    assert "test -w /tmp/zjheds" in preflight
    assert f"test ! -e {replica.lifecycle.temp_dir}" in preflight
    assert "curl" not in preflight
    assert "28000" not in preflight
    cleanup = cleanup_verification_command(replica)
    assert "droid_scaling --scoped-status" in cleanup
    assert "ray stop" not in cleanup


def test_isolation_only_preflight_has_no_production_interaction_and_static_disjointness(tmp_path):
    plan = _plan(tmp_path)
    replica = plan.replicas[0]
    command = droid_preflight_check_command(replica)
    assert "curl" not in command
    assert "health" not in command
    assert all(str(port) not in command for port in range(28000, 28007))
    assert replica.gpu_id == 7
    assert set(replica.lifecycle.ports.all_ports()).isdisjoint(range(28000, 28007))
    assert not str(replica.lifecycle.temp_dir).startswith("/tmp/ray-ego-serve-gpu")


def test_two_replicas_have_disjoint_gpu_ports_temp_and_http(tmp_path):
    plan = _plan(tmp_path, gpus=(7, 4))
    first, second = plan.replicas
    assert first.gpu_id != second.gpu_id
    assert set(first.lifecycle.ports.all_ports()).isdisjoint(second.lifecycle.ports.all_ports())
    assert first.lifecycle.temp_dir != second.lifecycle.temp_dir
    assert first.endpoint.base_url != second.endpoint.base_url


def test_same_gpu_replicas_require_labels_and_preserve_independent_scopes(tmp_path):
    base = _plan(tmp_path).application_release.path
    with pytest.raises(ExperimentConfigurationError, match="same-GPU.*labels"):
        build_droid_scaling_plan(
            experiment_id="mrg", gpu_ids=(7, 7), gpu_uuids=("GPU-7", "GPU-7"),
            run_root=tmp_path / "runs", application_release_path=base,
            droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA,
            checkpoint_digest="actual-checkpoint-digest", corpus_digest="corpus-digest", measurement_interval_id="mrg",
            cpu_offload=True, max_sessions=16,
            component_port_bases=(30000, 30400), worker_port_bases=(30100, 30500), serve_port_bases=(32000, 32100),
        )
    plan = build_droid_scaling_plan(
        experiment_id="mrg", gpu_ids=(7, 7), gpu_uuids=("GPU-7", "GPU-7"),
        run_root=tmp_path / "runs", application_release_path=base,
        droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA,
        checkpoint_digest="actual-checkpoint-digest", corpus_digest="corpus-digest", measurement_interval_id="mrg",
        cpu_offload=True, max_sessions=16, replica_labels=("a", "b"),
        component_port_bases=(30000, 30400), worker_port_bases=(30100, 30500), serve_port_bases=(32000, 32100),
    )
    first, second = plan.replicas
    assert [replica.replica_id for replica in plan.replicas] == ["droid-exp-mrg-a", "droid-exp-mrg-b"]
    assert [str(replica.lifecycle.temp_dir) for replica in plan.replicas] == ["/tmp/zjheds/mrg/a", "/tmp/zjheds/mrg/b"]
    assert first.gpu_id == second.gpu_id == 7
    assert set(first.lifecycle.ports.all_ports()).isdisjoint(second.lifecycle.ports.all_ports())
    assert dict(first.lifecycle.env_vars)["EGO_EXPERIMENT_REQUIRE_GPU_VACANCY"] == "1"
    assert dict(second.lifecycle.env_vars)["EGO_EXPERIMENT_REQUIRE_GPU_VACANCY"] == "0"
    assert "query-compute-apps" not in droid_preflight_check_command(second)
    runner = Runner()
    execution = execute_droid_scoped_plan(plan, command_runner=runner, typed_readiness_probe=lambda _: None)
    assert execution.readiness_replica_ids == ("droid-exp-mrg-a", "droid-exp-mrg-b")
    deploys = [command for command in runner.commands if "--app-choice droid" in command]
    assert len(deploys) == 2
    assert "query-compute-apps" in deploys[0]
    assert "query-compute-apps" not in deploys[1]


def test_three_replicas_accept_explicit_pairwise_disjoint_port_bases(tmp_path):
    base = _plan(tmp_path).application_release.path
    plan = build_droid_scaling_plan(
        experiment_id="hr3", gpu_ids=(4, 5, 7), gpu_uuids=("GPU-4", "GPU-5", "GPU-7"),
        run_root=tmp_path / "runs", application_release_path=base,
        droid_source_release_path=SOURCE_RELEASE, source_sha=SOURCE_SHA,
        checkpoint_digest="actual-checkpoint-digest", corpus_digest="corpus-digest", measurement_interval_id="hr3",
        cpu_offload=True, max_sessions=32,
        component_port_bases=(30000, 30400, 30800),
        worker_port_bases=(30100, 30500, 30900),
        serve_port_bases=(32000, 32100, 32200),
    )
    port_sets = [set(replica.lifecycle.ports.all_ports()) for replica in plan.replicas]
    assert all(max(ports) < 32768 for ports in port_sets)
    assert all(left.isdisjoint(right) for index, left in enumerate(port_sets) for right in port_sets[index + 1:])
    assert [replica.endpoint.base_url for replica in plan.replicas] == [
        "http://127.0.0.1:32000", "http://127.0.0.1:32100", "http://127.0.0.1:32200",
    ]


class Runner:
    def __init__(self, fail=lambda command: False):
        self.commands: list[str] = []
        self.fail = fail

    def run(self, command: str) -> CommandOutcome:
        self.commands.append(command)
        failed = self.fail(command)
        return CommandOutcome(command, 1 if failed else 0, stderr="forced" if failed else "")


def test_execute_uses_user_owned_fresh_temp_root_preflight_by_default(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner()
    result = execute_droid_scoped_plan(
        plan,
        command_runner=runner,
        typed_readiness_probe=lambda _: None,
    )
    assert result.readiness_replica_ids == ("droid-exp-droid-hardening-gpu7",)
    assert "mkdir -p /tmp/zjheds" in runner.commands[0]
    assert "test -w /tmp/zjheds" in runner.commands[0]
    assert f"test ! -e {plan.replicas[0].lifecycle.temp_dir}" in runner.commands[0]


def test_partial_start_and_keyboard_interrupt_cleanup_are_scoped_and_never_global(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner(lambda command: "ray.scripts.scripts start" in command)
    with pytest.raises(ExperimentConfigurationError, match="Ray start failed"):
        execute_droid_scoped_plan(
            plan, command_runner=runner, typed_readiness_probe=lambda _: None,
            preflight_runner=lambda _: None,
        )
    assert any("--scoped-stop" in command for command in runner.commands)
    assert all("ray stop" not in command for command in runner.commands)

    runner = Runner()
    with pytest.raises(KeyboardInterrupt):
        execute_droid_scoped_plan(
            plan, command_runner=runner,
            typed_readiness_probe=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
            preflight_runner=lambda _: None,
        )
    assert any("--scoped-stop" in command for command in runner.commands)
    assert all("ray stop" not in command for command in runner.commands)


def test_scoped_stop_rejects_broad_or_production_targets():
    for target in ("/tmp", "/tmp/ray-ego-serve-gpu2", str(DROID_EXPERIMENT_TEMP_ROOT / "exp")):
        with pytest.raises(ExperimentConfigurationError):
            stop_droid_scoped_experiment(target, pid_lookup=lambda _: ())
    assert stop_droid_scoped_experiment(
        DROID_EXPERIMENT_TEMP_ROOT / "exp" / "gpu7", pid_lookup=lambda _: (),
    ) == ()


def test_scoped_stop_removes_only_exact_empty_droid_temp_dir(tmp_path, monkeypatch):
    root = tmp_path / "zjheds"
    target = root / "d1h2" / "gpu7"
    target.mkdir(parents=True)
    (target / "ray-marker").write_text("candidate")
    sibling = root / "d1h2" / "gpu4"
    sibling.mkdir()
    (sibling / "keep").write_text("unrelated")
    monkeypatch.setattr(droid_scaling, "DROID_EXPERIMENT_TEMP_ROOT", root)

    assert stop_droid_scoped_experiment(target, pid_lookup=lambda _: ()) == ()
    assert not target.exists()
    assert (sibling / "keep").read_text() == "unrelated"


def test_failure_log_retention_copies_only_named_logs_from_exact_temp_scope(tmp_path, monkeypatch):
    root = tmp_path / "zjheds"
    scope = root / "d1h5" / "gpu7"
    logs = scope / "session_2026-07-21_20-41-41" / "logs"
    logs.mkdir(parents=True)
    (logs / "dashboard_MetricsHead.err").write_text("child traceback")
    (logs / "gcs_server.out").write_text("gcs evidence")
    (logs / "unlisted.log").write_text("must not copy")
    monkeypatch.setattr(droid_scaling, "DROID_EXPERIMENT_TEMP_ROOT", root)

    report = droid_scaling.retain_droid_failure_logs(scope, tmp_path / "evidence")

    retained = tmp_path / "evidence"
    assert report["temp_dir"] == str(scope.resolve())
    assert report["copied"] == ["dashboard_MetricsHead.err", "gcs_server.out"]
    assert "raylet.err" in report["missing"]
    assert (retained / "dashboard_MetricsHead.err").read_text() == "child traceback"
    assert (retained / "gcs_server.out").read_text() == "gcs evidence"
    assert not (retained / "unlisted.log").exists()
    with pytest.raises(ExperimentConfigurationError, match="outside one exact"):
        droid_scaling.retain_droid_failure_logs(root / "d1h5", retained)


def test_failure_log_hook_runs_before_scoped_droid_rollback(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    retained: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        droid_scaling,
        "retain_droid_failure_logs",
        lambda temp_dir, evidence_dir: retained.append((str(temp_dir), Path(evidence_dir))),
    )
    runner = Runner(lambda command: "ray.scripts.scripts start" in command)

    with pytest.raises(ExperimentConfigurationError, match="Ray start failed"):
        execute_droid_scoped_plan(
            plan,
            command_runner=runner,
            typed_readiness_probe=lambda _: None,
            preflight_runner=lambda _: None,
            failure_evidence_dir=tmp_path / "failure-evidence",
        )

    assert retained == [
        (str(plan.replicas[0].lifecycle.temp_dir), tmp_path / "failure-evidence" / plan.replicas[0].replica_id)
    ]
    assert any("--scoped-stop" in command for command in runner.commands)


def test_runtime_evidence_derives_release_module_checkpoint_pid_and_cuda(tmp_path, monkeypatch):
    release = _release(tmp_path)
    checkpoint = tmp_path / "droid.pth"
    checkpoint.write_bytes(b"actual droid checkpoint")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: SimpleNamespace(uuid="GPU-actual")),
    ))
    evidence = derive_worker_runtime_evidence(
        release_root=release,
        checkpoint_path=checkpoint,
        imported_module_file=release / "ego_annotation" / "serving" / "droid.py",
    )
    assert evidence.release_digest == release.name
    assert evidence.module_root == release
    assert evidence.checkpoint_digest == artifact_digest(checkpoint)
    assert evidence.worker_pid > 0
    assert evidence.cuda_uuid == "GPU-actual" and evidence.physical_gpu == 7
    outside = tmp_path / "shadow.py"
    outside.write_text("shadow")
    with pytest.raises(RuntimeError, match="outside verified release"):
        derive_worker_runtime_evidence(
            release_root=release, checkpoint_path=checkpoint, imported_module_file=outside,
        )


def _identity() -> ServerIdentity:
    return ServerIdentity(
        "exp", "replica", 7, 123, "127.0.0.1:34000", 36000,
        "/tmp/ego-droid-scaling/exp/gpu7", "droid-v1", "checkpoint", "ego.model-service.v1",
        "release", "release", "GPU-7",
        dependency_digest="source-digest", dependency_root="/sources/source-digest/droid_slam",
        source_amendment_id="recovered-hawor-droid-core-v1",
    )


def _requests():
    owner = Ownership("create", "job", "item", "droid.create_session", "source")
    create = DroidCreateSessionRequest(
        owner,
        DroidCamera((1.0, 1.0, 1.0, 1.0), ImageSize(8, 8), PixelTransform.identity()),
        DroidImageShape(8, 8),
        "droid-v1",
        DroidSessionOptions(buffer=8, warmup=2),
    )
    frame = DroidFrameRequest(
        Ownership("push", "job", "item", "droid.push_frame", "source", source_timestamp_s=0.0),
        "placeholder", "frame", 0.0, TensorPayload(bytes(8 * 8 * 3), (8, 8, 3), "uint8"),
        model_revision="droid-v1",
    )
    return create, frame


def _successful_push(request: DroidFrameRequest, identity: ServerIdentity) -> DroidFrameResponse:
    trace = DroidBatchTrace("batch", identity.replica_id, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, (request.session_id,))
    status = StepStatus(
        request.ownership, request.session_id, request.frame_id, request.source_timestamp_s,
        FrameValidity(request.frame_id, request.source_timestamp_s, True, True), 1, trace,
    )
    return DroidFrameResponse(request.ownership, status=status, server_identity=identity)


def test_typed_readiness_compares_create_push_and_terminal_identity(monkeypatch):
    identity = _identity()
    create_request, frame_request = _requests()
    calls = []

    async def fake_post(_client, _endpoint, operation, request):
        calls.append(operation)
        if operation == "create_session":
            response = DroidCreateSessionResponse(request.ownership, session_id="session", server_identity=identity)
        elif operation == "push_frame":
            response = _successful_push(request, identity)
        else:
            response = DroidFinalizeResponse(
                request.ownership,
                error=ServiceError(ErrorCode.UNRESOLVED, "one keyframe", False, request.ownership),
                server_identity=identity, terminal=True,
            )
        return DroidTypedCall(operation, 0, 1, 200, response)

    import ego_annotation.serving.benchmark.droid_scaling as module
    monkeypatch.setattr(module, "post_droid_typed", fake_post)
    endpoint = SimpleNamespace(base_url="http://127.0.0.1:36000")
    report = asyncio.run(typed_readiness_sequence(
        client=object(), endpoint=endpoint, expected_identity=identity,
        create_request=create_request, frame_request=frame_request,
    ))
    assert calls == ["create_session", "push_frame", "finalize"]
    assert report["terminal_outcome"] == "unresolved"


def test_identity_validation_rejects_jointly_missing_dependency_identity() -> None:
    expected = ServerIdentity(**{
        **_identity().__dict__, "dependency_digest": None, "dependency_root": None, "source_amendment_id": None,
    })
    response = DroidCreateSessionResponse(
        Ownership("r", "j", "i", "droid.create_session", "s"), session_id="session", server_identity=expected,
    )
    with pytest.raises(ExperimentConfigurationError, match="lacks dependency source identity"):
        validate_droid_server_identity(expected, response)


def test_wrong_identity_is_rejected_on_typed_response():
    expected = _identity()
    wrong = ServerIdentity(**{**expected.__dict__, "cuda_uuid": "GPU-wrong"})
    response = DroidCreateSessionResponse(
        Ownership("r", "j", "i", "droid.create_session", "s"), session_id="session", server_identity=wrong,
    )
    with pytest.raises(ExperimentConfigurationError, match="cuda_uuid"):
        validate_droid_server_identity(expected, response)


def test_readiness_identity_failure_still_terminally_finalizes_created_session(monkeypatch):
    expected = _identity()
    wrong = ServerIdentity(**{**expected.__dict__, "checkpoint_digest": "wrong"})
    create_request, frame_request = _requests()
    calls = []

    async def fake_post(_client, _endpoint, operation, request):
        calls.append(operation)
        if operation == "create_session":
            response = DroidCreateSessionResponse(request.ownership, session_id="session", server_identity=wrong)
        elif operation == "finalize":
            response = DroidFinalizeResponse(
                request.ownership,
                error=ServiceError(ErrorCode.UNRESOLVED, "terminal", False, request.ownership),
                server_identity=expected, terminal=True,
            )
        else:
            raise AssertionError("wrong create identity must prevent push")
        return DroidTypedCall(operation, 0, 1, 200, response)

    import ego_annotation.serving.benchmark.droid_scaling as module
    monkeypatch.setattr(module, "post_droid_typed", fake_post)
    with pytest.raises(ExperimentConfigurationError, match="checkpoint_digest"):
        asyncio.run(typed_readiness_sequence(
            client=object(), endpoint=SimpleNamespace(base_url="http://127.0.0.1:36000"),
            expected_identity=expected, create_request=create_request, frame_request=frame_request,
        ))
    assert calls == ["create_session", "finalize"]


def test_runtime_evidence_falls_back_to_nvidia_smi_without_pynvml(tmp_path, monkeypatch):
    import builtins
    import subprocess as real_subprocess
    import sys

    release = _release(tmp_path)
    checkpoint = tmp_path / "droid.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name="A800"),
    )))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("No module named 'pynvml'")
        return real_import(name, *args, **kwargs)

    class Completed:
        returncode = 0
        stdout = "GPU-smi-physical-7\n"
        stderr = ""

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(real_subprocess, "run", lambda *a, **k: Completed())
    evidence = derive_worker_runtime_evidence(
        release_root=release, checkpoint_path=checkpoint,
        imported_module_file=release / "ego_annotation" / "serving" / "droid.py",
    )
    assert evidence.physical_gpu == 7
    assert evidence.cuda_uuid == "GPU-smi-physical-7"


def test_runtime_evidence_uses_nvml_uuid_on_exact_torch_113_abi(tmp_path, monkeypatch):
    import sys

    release = _release(tmp_path)
    checkpoint = tmp_path / "droid.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(name="A800"),
    )))
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: b"GPU-nvml-physical-7" if handle == 7 else b"wrong",
    ))
    evidence = derive_worker_runtime_evidence(
        release_root=release, checkpoint_path=checkpoint,
        imported_module_file=release / "ego_annotation" / "serving" / "droid.py",
    )
    assert evidence.physical_gpu == 7
    assert evidence.cuda_uuid == "GPU-nvml-physical-7"
