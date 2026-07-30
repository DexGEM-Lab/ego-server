"""Focused V2 physical-GPU Hands/WiLoR control-plane tests.

These are CPU-only: deployment decorators and adapters are replaced at the module
boundary, so they prove composition and HTTP surface without loading either model.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import types
from typing import Any

import numpy as np
import pytest

from ego_annotation.serving import lifecycle
from ego_annotation.serving.contracts import (
    BatchTrace,
    HandDetection,
    HandsDetectResponse,
    HandsDetectResult,
    Ownership,
    PixelTransform,
    ImageSize,
    SpatialMetadata,
    TensorPayload,
)
from ego_annotation.serving.lifecycle import (
    COMMITTED_GPU_GROUPS,
    _candidate_cluster_pids,
    _cluster_membership_snapshot,
    _legacy_gpu1_process_pids,
    cluster_ownership,
    cosmos3_gpu_group,
    droid_gpu_group,
    hands_gpu_group,
    hawor_gpu_group,
    unidepth_gpu_group,
    hands_only_serve_config,
    scoped_stop_cluster,
    wilor_gpu_group,
    wilor_only_serve_config,
)
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import parse_multipart_response
from scripts import serve_group_driver
from scripts import start_ego_model_services as starter


def _deployment_module(monkeypatch: pytest.MonkeyPatch):
    class FakeServe:
        @staticmethod
        def deployment(**_kwargs: Any):
            def decorate(cls: type[Any]) -> type[Any]:
                cls.bind = classmethod(lambda bound_cls: bound_cls)  # type: ignore[attr-defined]
                return cls
            return decorate

        @staticmethod
        def batch(**_kwargs: Any):
            return lambda func: func

        @staticmethod
        def ingress(_app: Any):
            return lambda cls: cls

    fake_ray = types.ModuleType("ray")
    fake_ray.serve = FakeServe
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    sys.modules.pop("ego_annotation.serving.hands_deployment", None)
    return importlib.import_module("ego_annotation.serving.hands_deployment")


def _tensor(shape: tuple[int, ...], dtype: str = "float32") -> TensorPayload:
    return TensorPayload(np.zeros(shape, dtype=np.dtype(dtype)).tobytes(), shape, dtype)


def test_exclusive_deployments_construct_only_their_resident_adapter_and_publish_one_route(monkeypatch: pytest.MonkeyPatch):
    deployment = _deployment_module(monkeypatch)
    constructed: list[str] = []

    class Hands:
        def __init__(self, config: object) -> None:
            assert config == "hands-config"
            constructed.append("hands")

    class WiLoR:
        def __init__(self, config: object) -> None:
            assert config == "wilor-config"
            constructed.append("wilor")

    monkeypatch.setattr(deployment, "HandsAdapter", Hands)
    monkeypatch.setattr(deployment, "WiLoRAdapter", WiLoR)
    monkeypatch.setattr(deployment, "_hands_config_from_env", lambda: "hands-config")
    monkeypatch.setattr(deployment, "_wilor_config_from_env", lambda: "wilor-config")

    hands = deployment.HandsOnlyDeployment()
    assert constructed == ["hands"]
    assert hasattr(hands, "hands") and not hasattr(hands, "wilor")
    assert set(deployment.build_hands_only_api().openapi()["paths"]) == {"/hands.detect"}

    constructed.clear()
    wilor = deployment.WiLoROnlyDeployment()
    assert constructed == ["wilor"]
    assert hasattr(wilor, "wilor") and not hasattr(wilor, "hands")
    assert set(deployment.build_wilor_only_api().openapi()["paths"]) == {"/wilor.reconstruct"}
    assert deployment.hands_app is deployment.HandsWiLoRDeployment


def test_exclusive_hands_uses_the_existing_multipart_success_and_typed_error_encoders(monkeypatch: pytest.MonkeyPatch):
    deployment = _deployment_module(monkeypatch)
    ownership = Ownership("request", "job", "frame", "hands.detect", "source")
    detection = HandDetection(
        _tensor((1, 4)), _tensor((1,)), _tensor((1,), "uint8"), None,
        _tensor((1,)), _tensor((1,)), 1,
    )
    trace = BatchTrace("batch", "hands-gpu1", 1.0, 1.1, 1.2, 1.3, 1, 1, 1, 1)
    response = HandsDetectResponse(
        ownership,
        result=HandsDetectResult(
            ownership, detection, SpatialMetadata(
                ImageSize(width=12, height=8), ImageSize(width=12, height=8), "RGB", PixelTransform.identity(),
            ),
            "hands-yolo-v2", trace, 0,
        ),
    )
    success = deployment._hands_response_to_multipart_wire(response)
    metadata, arrays = parse_multipart_response(success.body, success.headers["content-type"])
    assert metadata["ownership"] == ownership.to_wire()
    assert set(arrays) == {"boxes", "scores", "sides", "visibility", "uncertainty"}
    typed_error = deployment._error_response("wrong crop")
    error_metadata, error_arrays = parse_multipart_response(typed_error.body, typed_error.headers["content-type"])
    assert error_metadata["error"]["code"] == "validation"
    assert error_arrays == {}


def test_split_lifecycle_has_one_gpu_per_group_disjoint_ports_and_rollback_config():
    hands, wilor = hands_gpu_group(), wilor_gpu_group()
    assert (hands.gpu_id, hands.logical_apis, hands.lifecycle.ports.gcs_port, hands.lifecycle.ports.serve_http_port) == (
        1, ("hands.detect",), 27000, 28001,
    )
    assert (wilor.gpu_id, wilor.logical_apis, wilor.lifecycle.ports.gcs_port, wilor.lifecycle.ports.serve_http_port) == (
        4, ("wilor.reconstruct",), 27200, 28004,
    )
    assert hands.lifecycle.num_gpus == wilor.lifecycle.num_gpus == 1
    all_ports = [port for group in COMMITTED_GPU_GROUPS for port in group.lifecycle.ports.all_ports()]
    assert len(all_ports) == len(set(all_ports))
    assert hands_only_serve_config()["applications"][0]["import_path"].endswith(":hands_only_app")
    assert wilor_only_serve_config()["applications"][0]["import_path"].endswith(":wilor_only_app")


def test_scoped_stop_requires_exact_allowlisted_ownership_and_shuts_serve_before_pids():
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    events: list[object] = []
    pid_sequences = iter(((101, 102), ()))

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        events.append(("shutdown", argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "shutdown-ok", "")

    def lookup(actual: object) -> tuple[int, ...]:
        assert actual == ownership
        return next(pid_sequences)

    result = scoped_stop_cluster(
        cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
        gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
        run=run, pid_lookup=lookup, kill=lambda pid, sig: events.append(("kill", pid, sig)),
    )
    assert result.stopped_pids == (101, 102)
    assert result.serve_shutdown_returncode == 0
    assert result.serve_shutdown_stdout == "shutdown-ok"
    assert result.serve_shutdown_timed_out is False
    assert events[0][0] == "shutdown"
    assert events[0][1][-3:] == ["-a", "http://127.0.0.1:27204", "-y"]
    assert events[0][2]["timeout"] == 20.0
    assert all("ray stop" not in str(event) for event in events)

    with pytest.raises(ValueError, match="mismatched"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir="/tmp/not-wilor",
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            run=run, pid_lookup=lookup,
        )

    legacy = cluster_ownership("ego-hands-gpu1-27000")
    legacy_lookups = iter(((303,), ()))
    legacy_result = scoped_stop_cluster(
        cluster_id=legacy.cluster_id, temp_dir=legacy.temp_dir,
        gcs_address=legacy.gcs_address, dashboard_address=legacy.dashboard_address,
        legacy_gpu1=True, run=run, legacy_pid_lookup=lambda actual: next(legacy_lookups),
        kill=lambda pid, sig: events.append(("legacy-kill", pid, sig)),
    )
    assert legacy_result.legacy_gpu1 is True and legacy_result.stopped_pids == (303,)
    with pytest.raises(ValueError, match="no corroborated"):
        scoped_stop_cluster(
            cluster_id=legacy.cluster_id, temp_dir=legacy.temp_dir,
            gcs_address=legacy.gcs_address, dashboard_address=legacy.dashboard_address,
            legacy_gpu1=True, run=run, legacy_pid_lookup=lambda actual: (),
        )
    with pytest.raises(ValueError, match="allowlisted"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            legacy_gpu1=True, run=run,
        )


def test_scoped_stop_filters_its_invoking_pid_before_any_signal(monkeypatch: pytest.MonkeyPatch):
    ownership = cluster_ownership("ego-hands-gpu1-27000")
    invoking_pid = 909
    lookups = iter(((invoking_pid, 303), (invoking_pid, 303)))
    signals: list[int] = []
    monkeypatch.setattr(lifecycle.os, "getpid", lambda: invoking_pid)

    result = scoped_stop_cluster(
        cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
        gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
        legacy_gpu1=True,
        run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
        legacy_pid_lookup=lambda _ownership: next(lookups),
        kill=lambda pid, _sig: signals.append(pid),
    )
    assert result.stopped_pids == (303,)
    assert signals == [303, 303]


def test_driver_selects_exclusive_apps_and_router_metadata_points_to_split_lanes(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, str, str]] = []
    serve = types.SimpleNamespace(
        start=lambda **_kwargs: None,
        run=lambda app, *, name, route_prefix: calls.append((app, name, route_prefix)),
    )
    ray = types.ModuleType("ray")
    ray.init = lambda **_kwargs: None
    ray.serve = serve
    monkeypatch.setitem(sys.modules, "ray", ray)
    hands_module = types.ModuleType("ego_annotation.serving.hands_deployment")
    hands_module.hands_app = "hands-wilor-rollback"
    hands_module.hands_only_app = "hands-only"
    hands_module.wilor_only_app = "wilor-only"
    monkeypatch.setitem(sys.modules, "ego_annotation.serving.hands_deployment", hands_module)

    serve_group_driver.deploy_group(1, address="127.0.0.1:27000", dashboard_address="http://127.0.0.1:27004", port=28001)
    serve_group_driver.deploy_group(4, address="127.0.0.1:27200", dashboard_address="http://127.0.0.1:27204", port=28004)
    serve_group_driver.deploy_group(
        1, address="127.0.0.1:27000", dashboard_address="http://127.0.0.1:27004", port=28001, combined=True,
    )
    assert calls == [
        ("hands-only", "ego-hands", "/"),
        ("wilor-only", "ego-wilor", "/"),
        ("hands-wilor-rollback", "ego-hands-wilor", "/"),
    ]
    with pytest.raises(ValueError, match="endpoint tuple"):
        serve_group_driver.deploy_group(1, address="127.0.0.1:27000", dashboard_address="http://127.0.0.1:27204", port=28001)

    router = ModelServiceRouter.canonical()
    hands = router.endpoint_for(ModelApiName.HANDS_DETECT)
    wilor = router.endpoint_for(ModelApiName.WILOR_RECONSTRUCT)
    assert (hands.gpu_id, hands.serve_http_port, hands.model_revision) == (1, 28001, "hands-yolo-v2")
    assert (wilor.gpu_id, wilor.serve_http_port, wilor.model_revision) == (4, 28004, "wilor-final-v1")


def test_starter_pins_split_groups_to_immutable_release_and_wilor_workdir(tmp_path):
    release = tmp_path / "immutable-release"
    treatment = starter.with_split_release(wilor_gpu_group(), release)
    env = dict(treatment.lifecycle.env_vars)
    assert env["EGO_APPLICATION_RELEASE_ROOT"] == str(release)
    assert env["PYTHONPATH"].split(":", 1)[0] == str(release)
    script = starter.serve_launch_script(treatment)
    assert f"cd {env['EGO_WILOR_REPO']} &&" in script
    assert f"{release}/scripts/serve_group_driver.py" in script
    assert "env -u RAY_ADDRESS" in script
    assert "--dashboard-address http://127.0.0.1:27204 --port 28004" in script


def test_marker_stop_requires_exact_temp_ray_command_in_addition_to_environment(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    required_env = (
        f"EGO_MODEL_CLUSTER_ID={ownership.cluster_id}\0"
        f"EGO_MODEL_GCS_ADDRESS={ownership.gcs_address}\0"
        f"EGO_MODEL_TEMP_DIR={ownership.temp_dir}\0"
    ).encode()
    proc, pair, session = tmp_path / "proc", (500, 500), f"{ownership.temp_dir}/session_1"

    def process(pid: int, parent: int, command: str, *, marked: bool = True, signature: tuple[int, int] = pair) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        fields = ["S", str(parent), str(signature[0]), str(signature[1])] + ["0"] * 15 + [str(pid * 10)]
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (ray) {' '.join(fields)}\n")
        (path / "environ").write_bytes(required_env if marked else b"")

    process(500, 1, "sleep infinity", marked=False)
    process(101, 500, f"gcs_server --gcs_server_port=27200 --session-dir={session}")
    process(102, 500, f"python -u /opt/ray/ray/dashboard/dashboard.py --session-dir={session}")
    process(103, 500, f"python -u /opt/ray/ray/_private/log_monitor.py --session-dir={session}")
    process(104, 500, f"raylet --session-dir={session}")
    process(105, 500, "raylet --session-dir=/tmp/ray-ego-serve-gpu4-wilor-old/session_1", signature=(600, 600))
    assert _candidate_cluster_pids(ownership, proc_root=proc) == (101, 102, 103, 104)


def _marker_generation_proc(tmp_path, ownership):
    proc, pair, session = tmp_path / "proc", (500, 500), f"{ownership.temp_dir}/session_20260723"
    marked = (
        f"EGO_MODEL_CLUSTER_ID={ownership.cluster_id}\0"
        f"EGO_MODEL_GCS_ADDRESS={ownership.gcs_address}\0"
        f"EGO_MODEL_TEMP_DIR={ownership.temp_dir}\0"
    ).encode()

    def process(pid: int, parent: int, command: str, *, environment: bytes = b"", signature: tuple[int, int] = pair):
        path = proc / str(pid)
        path.mkdir(parents=True)
        fields = ["S", str(parent), str(signature[0]), str(signature[1])] + ["0"] * 15 + [str(pid * 100)]
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (ray) {' '.join(fields)}\n")
        (path / "environ").write_bytes(environment)

    process(500, 1, "sleep infinity")
    process(501, 500, f"gcs_server --gcs_server_port={ownership.gcs_address.rsplit(':', 1)[1]} --session-dir={session}", environment=marked)
    process(502, 500, f"python -u /opt/ray/ray/dashboard/dashboard.py --session-dir={session}", environment=marked)
    process(503, 500, f"python -u /opt/ray/ray/_private/log_monitor.py --session-dir={session}", environment=marked)
    process(504, 500, f"raylet --session-dir={session}", environment=marked)
    process(505, 1, f"python -u /opt/ray/ray/autoscaler/_private/monitor.py --gcs-address={ownership.gcs_address} --session-dir={session}", environment=marked)
    process(506, 1, f"python -m ray.util.client.server --gcs-address={ownership.gcs_address} --port={ownership.ray_client_server_port} --session-dir={session}", environment=marked)
    # 17 generic descendants: resource tracker, nine dashboard helpers, two agents,
    # four Serve/IDLE actors, and one client helper.
    process(507, 502, "python resource_tracker.py")
    for pid in range(508, 517):
        process(pid, 502, f"ray-dashboard-helper-{pid}")
    process(517, 502, "ray::DashboardAgent")
    process(518, 504, "ray::RuntimeEnvAgent", environment=f"EGO_MODEL_TEMP_DIR={ownership.temp_dir}\0".encode())
    for pid, command in ((519, "ray::ServeController"), (520, "ray::ProxyActor"), (521, "ray::ServeReplica"), (522, "ray::IDLE")):
        process(pid, 504, command)
    process(523, 506, "python client_helper.py")
    return proc, process, pair


def test_marker_membership_matches_live_ray_session_topology(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, _process, pair = _marker_generation_proc(tmp_path, ownership)
    snapshot = _cluster_membership_snapshot(ownership, proc_root=proc)
    assert snapshot.signature == pair
    assert tuple(identity.pid for identity in snapshot.member_identities) == tuple(range(501, 524))
    assert {identity.pid for identity in snapshot.root_identities} == {501, 502, 503, 504, 505, 506}
    assert tuple(identity.pid for identity in snapshot.outsider_identities) == (500,)
    assert _candidate_cluster_pids(ownership, proc_root=proc) == tuple(range(501, 524))


def test_marker_membership_fails_closed_for_same_signature_outsider_or_split_seed(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, process, pair = _marker_generation_proc(tmp_path, ownership)
    process(600, 500, "python unrelated.py", signature=pair)
    with pytest.raises(ValueError, match="outsider"):
        _cluster_membership_snapshot(ownership, proc_root=proc)
    shutdowns: list[object] = []
    with pytest.raises(ValueError, match="outsider"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            run=lambda *args, **kwargs: shutdowns.append((args, kwargs)), proc_root=proc, wait=lambda _seconds: None,
        )
    assert shutdowns == []

    other = tmp_path / "other"
    proc, process, pair = _marker_generation_proc(other, ownership)
    fields = ["S", "500", "700", "700"] + ["0"] * 15 + ["50400"]
    (proc / "504" / "stat").write_text(f"504 (ray) {' '.join(fields)}\n")
    with pytest.raises(ValueError, match="generation-ambiguous"):
        _cluster_membership_snapshot(ownership, proc_root=proc)

    sibling = tmp_path / "sibling"
    proc, process, pair = _marker_generation_proc(sibling, ownership)
    sibling_env = b"EGO_MODEL_CLUSTER_ID=ego-hands-gpu1-27000\0EGO_MODEL_GCS_ADDRESS=127.0.0.1:27000\0EGO_MODEL_TEMP_DIR=/tmp/ray-ego-serve-gpu1\0"
    process(700, 1, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session_20260723", environment=sibling_env, signature=pair)
    with pytest.raises(ValueError, match="sibling"):
        _cluster_membership_snapshot(ownership, proc_root=proc)


def test_marker_scoped_stop_freezes_generation_reparents_children_and_signals_child_first(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, _process, _pair = _marker_generation_proc(tmp_path, ownership)
    events: list[tuple[str, int]] = []

    def run(argv, **_kwargs):
        events.append(("shutdown", 0))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def kill(pid: int, sig: int) -> None:
        events.append(("term" if sig == lifecycle.signal.SIGTERM else "kill", pid))
        if sig == lifecycle.signal.SIGTERM and pid == 507:
            stat = proc / "507" / "stat"
            stat.write_text(stat.read_text().replace(" S 502 500 500 ", " S 1 500 500 "))
        if sig == lifecycle.signal.SIGKILL:
            shutil.rmtree(proc / str(pid))

    result = scoped_stop_cluster(
        cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
        gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
        run=run, kill=kill, proc_root=proc, wait=lambda _seconds: None,
    )
    assert result.generation_signature == (500, 500)
    assert result.term_pids and result.kill_pids
    assert 500 not in result.term_pids and 500 not in result.kill_pids
    assert result.final_survivors == ()
    assert events[0] == ("shutdown", 0)
    assert max(index for index, event in enumerate(events) if event[0] == "term") < min(
        index for index, event in enumerate(events) if event[0] == "kill"
    )
    assert result.term_pids.index(507) < result.term_pids.index(502)


def test_marker_scoped_stop_rejects_new_unlinked_same_signature_process(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, process, _pair = _marker_generation_proc(tmp_path, ownership)

    def wait(_seconds: float) -> None:
        process(600, 1, "python unrelated.py")

    with pytest.raises(RuntimeError, match="unlinked"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
            kill=lambda _pid, _sig: None, proc_root=proc, wait=wait,
        )


def test_marker_scoped_stop_adds_only_causal_new_descendants_and_never_kills_pid_reuse(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, process, _pair = _marker_generation_proc(tmp_path, ownership)
    calls: list[tuple[int, int]] = []

    spawned = False

    def causal_wait(_seconds: float) -> None:
        nonlocal spawned
        if not spawned:
            process(524, 504, "python just_spawned_helper.py")
            spawned = True

    def causal_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        if sig == lifecycle.signal.SIGKILL:
            shutil.rmtree(proc / str(pid))

    result = scoped_stop_cluster(
        cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
        gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
        run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
        kill=causal_kill, proc_root=proc, wait=causal_wait,
    )
    assert 524 not in result.term_pids and 524 in result.kill_pids

    reused = tmp_path / "reused"
    proc, _process, _pair = _marker_generation_proc(reused, ownership)
    signals: list[tuple[int, int]] = []

    def reuse_wait(_seconds: float) -> None:
        stat = proc / "507" / "stat"
        stat.write_text(stat.read_text().replace(" 50700\n", " 99999\n"))

    with pytest.raises(RuntimeError, match="unlinked"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
            kill=lambda pid, sig: signals.append((pid, sig)), proc_root=proc, wait=reuse_wait,
        )
    assert (507, lifecycle.signal.SIGKILL) not in signals


def test_marker_scoped_stop_reports_final_survivors(tmp_path):
    ownership = cluster_ownership("ego-wilor-gpu4-27200")
    proc, _process, _pair = _marker_generation_proc(tmp_path, ownership)
    with pytest.raises(RuntimeError, match="incomplete"):
        scoped_stop_cluster(
            cluster_id=ownership.cluster_id, temp_dir=ownership.temp_dir,
            gcs_address=ownership.gcs_address, dashboard_address=ownership.dashboard_address,
            run=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
            kill=lambda _pid, _sig: None, proc_root=proc, wait=lambda _seconds: None,
        )


def test_legacy_gpu1_teardown_requires_correlated_ray_temp_and_gcs_process_tree(tmp_path):
    proc = tmp_path / "proc"

    def process(pid: int, parent: int, command: str) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (ray) S {parent} 0 0 0\n")
        (path / "environ").write_bytes(b"")

    # Raylet owns the exact legacy temp tree; the sibling GCS server corroborates
    # the same Ray parent and exact GCS port. A different Ray tree is excluded.
    process(100, 1, "python -m ray._private.monitor")
    process(101, 100, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session")
    process(102, 100, "gcs_server --gcs_server_port=27000")
    process(103, 101, "ray::ServeReplica")
    process(200, 1, "raylet --session-dir=/tmp/ray-ego-serve-gpu1-other/session")
    process(201, 200, "gcs_server --gcs_server_port=27000")

    ownership = cluster_ownership("ego-hands-gpu1-27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == (100, 101, 102, 103)
    with pytest.raises(ValueError, match="allowlisted"):
        _legacy_gpu1_process_pids(cluster_ownership("ego-wilor-gpu4-27200"), proc_root=proc)


def test_legacy_gpu1_lifecycle_cli_is_never_a_ray_candidate(tmp_path, monkeypatch: pytest.MonkeyPatch):
    proc = tmp_path / "proc"
    lifecycle_pid = 77

    def process(pid: int, parent: int, command: str) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (cmd) S {parent} 0 0 0\n")
        (path / "environ").write_bytes(b"")

    # This is the exact one-time lifecycle invocation. Its interpreter path and
    # target arguments contain `ray`, but it is neither a Ray role nor a target PID.
    process(
        lifecycle_pid, 10,
        "/home/zjh/miniconda3/envs/ray_serve_hands/bin/python -m "
        "ego_annotation.serving.lifecycle --scoped-stop --legacy-gpu1 "
        "--temp-dir /tmp/ray-ego-serve-gpu1 --gcs-address 127.0.0.1:27000",
    )
    process(100, 1, "python -m ray._private.monitor")
    process(101, 100, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session")
    process(102, 100, "gcs_server --gcs_server_port=27000")
    process(103, 101, "ray::ServeReplica")
    # Ray 2.55.1 agent command forms have generic basenames (`agent.py`, `main.py`)
    # but their package paths identify them as target Ray control-plane processes.
    process(104, 100, "python -u /opt/ray/python/ray/dashboard/agent.py --node-ip-address=127.0.0.1")
    process(105, 100, r"python -u C:\opt\ray\python\ray\_private\runtime_env\agent\main.py --listen-port=27009")
    monkeypatch.setattr(lifecycle.os, "getpid", lambda: lifecycle_pid)

    ownership = cluster_ownership("ego-hands-gpu1-27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == (100, 101, 102, 103, 104, 105)
    assert lifecycle._is_ray_process(
        b"/home/zjh/miniconda3/envs/ray_serve_hands/bin/python -m ego_annotation.serving.lifecycle"
    ) is False
    assert lifecycle._is_ray_process(b"PYTHONPATH=/srv/ray_serve_hands:/srv/app python") is False
    assert all(lifecycle._is_ray_process(command) for command in (
        b"gcs_server --gcs_server_port=27000",
        b"CUDA_VISIBLE_DEVICES=1 /opt/ray/bin/raylet --session-dir=/tmp/ray-ego-serve-gpu1/session",
        b"env -u RAY_ADDRESS CUDA_VISIBLE_DEVICES=1 python -m ray.dashboard.agent",
        b"raylet --session-dir=/tmp/ray-ego-serve-gpu1/session",
        b"python -u /opt/ray/python/ray/dashboard/dashboard.py --host=127.0.0.1",
        b"python -m ray._private.monitor",
        b"python -m ray.dashboard.agent",
        b"python -m ray._private.runtime_env.agent.main",
        b"python -m ray.autoscaler._private.monitor",
        b"python -m ray.util.client.server",
        b"python -u /opt/ray/python/ray/_private/log_monitor.py",
        b"python -u /opt/ray/python/ray/_private/runtime_env/agent/main.py",
        b"python -u /opt/ray/python/ray/dashboard/agent.py --node-ip-address=127.0.0.1",
        rb"python -u C:\opt\ray\python\ray\_private\runtime_env\agent\main.py --listen-port=27009",
        b"ray::ServeReplica",
    ))
    assert not lifecycle._is_ray_process(b"/tmp/operator/agent.py")
    assert not lifecycle._is_ray_process(b"/tmp/operator/main.py")
    assert not lifecycle._is_ray_process(b"python -m operator.agent")
    assert not lifecycle._is_ray_process(b"python -m operator.main")
    assert not lifecycle._is_ray_process(b"--module=ray.dashboard.agent")
    assert not lifecycle._is_ray_process(b"python -B -m ray.dashboard.agent")
    assert not lifecycle._is_ray_process(b"python worker.py --module ray.dashboard.agent")
    assert not lifecycle._is_ray_process(b"python worker.py --handler /opt/ray/python/ray/dashboard/agent.py")
    assert not lifecycle._is_ray_process(
        b"python worker.py --handler /opt/ray/python/ray/_private/runtime_env/agent/main.py"
    )


def test_legacy_gpu1_requires_exact_gcs_server_port_argument(tmp_path):
    proc = tmp_path / "proc"

    def process(pid: int, parent: int, command: str) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (ray) S {parent} 0 0 0\n")
        (path / "environ").write_bytes(b"")

    process(100, 1, "python -m ray._private.monitor")
    process(101, 100, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session")
    process(102, 100, "gcs_server --gcs_server_port=127000")
    process(103, 101, "ray::ServeReplica")
    ownership = cluster_ownership("ego-hands-gpu1-27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == ()

    # Ray also accepts the separated argument form; its exact target value
    # corroborates the temp-tree Ray root without accepting 127000 as 27000.
    (proc / "102" / "cmdline").write_bytes(b"gcs_server\0--gcs_server_port\0" + b"27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == (100, 101, 102, 103)


def test_legacy_gpu1_ignores_post_script_ray_role_values(tmp_path):
    proc = tmp_path / "proc"

    def process(pid: int, parent: int, command: str) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (cmd) S {parent} 0 0 0\n")
        (path / "environ").write_bytes(b"")

    # The worker's later option values resemble Ray roles, but its invoked script
    # is not Ray. It must not become the common Ray root for this temp/GCS tree.
    process(
        100, 1,
        "python worker.py --module ray.dashboard.agent "
        "--handler /opt/ray/python/ray/dashboard/agent.py "
        "--temp-dir /tmp/ray-ego-serve-gpu1 --gcs-address 127.0.0.1:27000",
    )
    process(101, 100, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session")
    process(102, 100, "gcs_server --gcs_server_port=27000")
    process(103, 101, "ray::ServeReplica")

    ownership = cluster_ownership("ego-hands-gpu1-27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == ()


def test_legacy_gpu1_privileged_proc_never_uses_pid1_or_sibling_ray_tree(tmp_path):
    proc = tmp_path / "proc"

    def process(pid: int, parent: int, command: str) -> None:
        path = proc / str(pid)
        path.mkdir(parents=True)
        (path / "cmdline").write_bytes(command.replace(" ", "\0").encode())
        (path / "stat").write_text(f"{pid} (cmd) S {parent} 0 0 0\n")
        (path / "environ").write_bytes(b"VISIBLE_TO_PRIVILEGED_CALLER=1\0")

    process(1, 0, "init")
    process(100, 1, "python -m ray._private.monitor")
    process(101, 100, "raylet --session-dir=/tmp/ray-ego-serve-gpu1/session")
    process(102, 100, "gcs_server --gcs_server_port=27000")
    process(103, 101, "ray::ServeReplica")
    # A sibling Ray cluster also orphaned below PID 1 must remain excluded.
    process(200, 1, "python -m ray._private.monitor")
    process(201, 200, "raylet --session-dir=/tmp/ray-ego-serve-gpu0/session")
    process(202, 200, "gcs_server --gcs_server_port=26000")
    process(203, 201, "ray::ServeReplica")

    ownership = cluster_ownership("ego-hands-gpu1-27000")
    assert _legacy_gpu1_process_pids(ownership, proc_root=proc) == (100, 101, 102, 103)


def test_starter_probe_scopes_openapi_to_ingress_and_accepts_rollback_state(monkeypatch: pytest.MonkeyPatch):
    class Response:
        def __init__(self, status: int, body: bytes = b"") -> None:
            self.status, self.body = status, body

        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return self.body

    urls: list[str] = []

    def good_urlopen(url: str, timeout: float):
        urls.append(url)
        if url.endswith("/openapi.json"):
            return Response(200, json.dumps({"paths": {"/wilor.reconstruct": {}}}).encode())
        return Response(200)

    monkeypatch.setattr(starter.urllib.request, "urlopen", good_urlopen)
    assert starter.probe_group_routes(wilor_gpu_group(), "127.0.0.1") is True
    assert urls == [
        "http://127.0.0.1:28004/-/healthz",
        "http://127.0.0.1:28004/-/routes",
        "http://127.0.0.1:28004/openapi.json",
    ]

    non_ingress_urls: list[str] = []
    def non_ingress_urlopen(url: str, timeout: float):
        non_ingress_urls.append(url)
        return Response(404) if url.endswith("/openapi.json") else Response(200)

    monkeypatch.setattr(starter.urllib.request, "urlopen", non_ingress_urlopen)
    for group in (unidepth_gpu_group(), droid_gpu_group(), hawor_gpu_group(), cosmos3_gpu_group()):
        assert starter.probe_group_state(group, "127.0.0.1") == "serve-healthy"
    assert all(not url.endswith("/openapi.json") for url in non_ingress_urls)
    assert {url.rsplit(":", 1)[1].split("/", 1)[0] for url in non_ingress_urls} == {"28000", "28002", "28003", "28006"}

    def rollback_urlopen(url: str, timeout: float):
        if url.endswith("/openapi.json"):
            return Response(200, json.dumps({"paths": {"/hands.detect": {}, "/wilor.reconstruct": {}}}).encode())
        return Response(200)

    monkeypatch.setattr(starter.urllib.request, "urlopen", rollback_urlopen)
    assert starter.probe_group_state(hands_gpu_group(), "127.0.0.1") == "hands-wilor-rollback"

    monkeypatch.setattr(starter.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(404))
    assert starter.probe_health("127.0.0.1", 28004) is False
