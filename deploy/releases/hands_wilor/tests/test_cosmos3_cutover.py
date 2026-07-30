"""CPU-only tests for the guarded Cosmos3 GPU6 cutover contract."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from ego_annotation.serving import cosmos3_cutover as cutover
from ego_annotation.serving.lifecycle import cosmos3_lifecycle


def _verified_reports(root: Path) -> None:
    logs = root / "logs"
    logs.mkdir(parents=True)
    venv = str(root / ".venv")
    interpreter = f"{venv}/bin/python"
    checks: dict[str, Any] = {
        "import:ray": {"ok": True},
        "import:ray.serve": {"ok": True},
        "import:vllm": {"ok": True},
        "import:torch": {"ok": True},
        "import:transformers_cosmos3": {"ok": True},
        "import:vllm_cosmos3": {"ok": True},
        "no_pth_overlay_into_ylang": {"ok": True, "offending": []},
        "ModelRegistry.Cosmos3ReasonerForConditionalGeneration": {"ok": True},
        "AsyncEngineArgs.construct_no_weights": {"ok": True, "model": "nvidia/Cosmos3-Nano"},
        "torch.cuda": {"available": True},
    }
    verify = {
        "python": "3.13.14",
        "venv": venv,
        "versions": {"ray": "2.55.1", "vllm": "0.19.1", "torch": "2.10.0"},
        "checks": checks,
        "errors": [],
        "ok": True,
    }
    verify_path = logs / cutover.VERIFY_REPORT_NAME
    # The actual verified report has this non-JSON warning preamble.
    verify_path.write_text("[transformers] deprecation warning\n" + json.dumps(verify))
    finalize = {
        "report_type": "standalone_finalize",
        "host": "dex-a800",
        "interpreter": {"path": interpreter, "venv": venv, "version": "3.13.14"},
        "versions": {"ray": "2.55.1", "vllm": "0.19.1", "torch": "2.10.0"},
        "production_port_8001": {
            "status": "unchanged",
            "model_endpoint": "http://127.0.0.1:8001/v1/models",
            "models": ["nvidia/Cosmos3-Nano"],
        },
        "pth_overlay_guard": {"status": "absent", "offending_files": []},
        "cosmos_plugins": {
            "transformers_cosmos3": {"import_ok": True},
            "vllm_cosmos3": {"class_accessible": True},
            "ModelRegistry": {"registration_confirmed": True},
            "AsyncEngineArgs": {"construct_ok": True},
        },
        "verification": {"imports_all_pass": True, "errors": [], "report": str(verify_path)},
        "cpu_diagnostic_cluster": {
            "ran": True,
            "gpu_advertised": False,
            "port_disjoint_from_gpu6": {"no_overlap": True},
            "residual_processes": [],
        },
    }
    (logs / cutover.FINALIZE_REPORT_NAME).write_text(json.dumps(finalize))


@pytest.fixture
def standalone_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "standalone"
    _verified_reports(root)
    # The lifecycle must name the exact interpreter proved by the reports.
    monkeypatch.setattr(cutover, "cosmos3_lifecycle", lambda: replace(cosmos3_lifecycle(), interpreter=str(root / ".venv/bin/python")))
    return root


def test_gate_consumes_warning_prefixed_verify_report_and_cross_checks_finalize(standalone_dir: Path) -> None:
    evidence = cutover.validate_standalone_artifacts(standalone_dir)
    assert evidence.interpreter == str(standalone_dir / ".venv/bin/python")
    assert evidence.verify_report.name == "verify_report.json"
    assert evidence.finalize_report.name == "finalize_20260717T000000Z.json"


def test_gate_rejects_failed_plugin_even_when_reports_exist(standalone_dir: Path) -> None:
    verify_path = standalone_dir / "logs" / cutover.VERIFY_REPORT_NAME
    text = verify_path.read_text()
    parsed = json.loads(text[text.index("{"):])
    parsed["checks"]["ModelRegistry.Cosmos3ReasonerForConditionalGeneration"]["ok"] = False
    verify_path.write_text("warning\n" + json.dumps(parsed))
    with pytest.raises(cutover.CutoverGateError, match="ModelRegistry"):
        cutover.validate_standalone_artifacts(standalone_dir)


def test_guarded_commands_pin_standalone_gcs_and_explicit_worker_ports() -> None:
    commands = cutover.guarded_cutover_commands("/home/zjh/cosmos3_ray_serve/run/cosmos3_serve.yaml")
    workers = ",".join(str(port) for port in range(26900, 26932))
    assert "standalone/.venv/bin/python" in commands.start_cluster
    assert f"--worker-port-list={workers}" in commands.start_cluster
    assert "26900-26931" not in commands.start_cluster
    assert "--port=26801" in commands.start_cluster
    assert commands.deploy.endswith("-a http://127.0.0.1:26800")
    assert commands.status.endswith("-a http://127.0.0.1:26800")


def test_execute_refuses_live_bare_listener_before_any_candidate_command(standalone_dir: Path) -> None:
    ran: list[Any] = []
    with pytest.raises(cutover.CutoverGateError, match="still listening"):
        cutover.execute_guarded_cutover(
            "/home/zjh/cosmos3_ray_serve/run/cosmos3_serve.yaml",
            artifacts_dir=standalone_dir,
            bare_listener=lambda _host, _port: True,
            run=lambda *args, **kwargs: ran.append((args, kwargs)),
        )
    assert ran == []


def test_execute_uses_explicit_commands_and_removes_stray_ray_address(standalone_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))

    cutover.execute_guarded_cutover(
        "/home/zjh/cosmos3_ray_serve/run/cosmos3_serve.yaml",
        artifacts_dir=standalone_dir,
        bare_listener=lambda _host, _port: False,
        run=fake_run,
    )
    assert len(calls) == 2
    assert calls[0][0][0].endswith("standalone/.venv/bin/python")
    assert "--port=26801" in calls[0][0]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert calls[1][0][-2:] == ["-a", "http://127.0.0.1:26800"]
    assert "RAY_ADDRESS" not in calls[0][1]["env"]


def test_scoped_rollback_never_uses_global_ray_stop_and_signals_only_matched_pids() -> None:
    commands = cutover.scoped_rollback_commands()
    assert all("ray.scripts.scripts stop" not in command for command in commands)
    assert commands[0].endswith("-a http://127.0.0.1:26800 -y")
    assert "--scoped-stop --temp-dir /tmp/ray-ego-serve-cosmos3" in commands[1]
    snapshots = iter(([11, 12], [12]))
    signals: list[tuple[int, int]] = []
    stopped = cutover.stop_scoped_candidate(
        "/tmp/ray-ego-serve-cosmos3",
        pid_lookup=lambda _temp_dir: next(snapshots),
        kill=lambda pid, sig: signals.append((pid, sig)),
    )
    assert stopped == (11, 12)
    assert signals == [(11, cutover.signal.SIGTERM), (12, cutover.signal.SIGTERM), (12, cutover.signal.SIGKILL)]


def test_scoped_rollback_rejects_any_non_candidate_temp_dir() -> None:
    with pytest.raises(cutover.CutoverGateError, match="only permits"):
        cutover.stop_scoped_candidate("/tmp")


def test_candidate_pid_scan_excludes_its_own_scoped_stop_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI itself contains --temp-dir, so it must never select and kill itself.
    for pid in (11, 12):
        proc = tmp_path / str(pid)
        proc.mkdir()
        (proc / "cmdline").write_bytes(f"python --temp-dir /tmp/ray-ego-serve-cosmos3 pid={pid}".encode())
    monkeypatch.setattr(cutover.os, "getpid", lambda: 11)
    assert cutover._candidate_process_pids("/tmp/ray-ego-serve-cosmos3", proc_root=tmp_path) == [12]


def test_repo_bundles_path_free_real_image_acceptance_seed() -> None:
    body = Path("assets/cosmos3/representative_request.multipart").read_bytes()
    headers = json.loads(Path("assets/cosmos3/representative_request_headers.json").read_text())
    assert len(body) > 100_000
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert headers["payload_sha256"] == "6842f3be595a1ddaace6777caf780c37ef568bede884f4d0c4d0ee86536a8ba4"
    assert b'"source_id":"cosmos3-acceptance-image-00"' in body
    assert b"/home/" not in body and b"/vePFS" not in body


def test_committed_serve_config_uses_lifecycle_pythonpath_not_invalid_local_runtime_env() -> None:
    config = Path("configs/cosmos3_serve.yaml").read_text()
    assert "import_path: ego_annotation.serving.cosmos3_deployment:app" in config
    assert "working_dir: /home/zjh/cosmos3_ray_serve/workspace" not in config
    assert "port: 28006" in config
    startup = cosmos3_lifecycle().startup_command()
    assert "PYTHONPATH=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_runtime/current" in startup
    assert "AUTOSCALER_METRIC_PORT=26811" in startup
    assert "DASHBOARD_METRIC_PORT=26812" in startup
    assert "--metrics-export-port=26810" in startup
    assert "HF_HOME=/home/ylang/.cache/huggingface" in startup


def test_worker_port_range_is_rejected_at_lifecycle_boundary() -> None:
    ports = replace(cosmos3_lifecycle().ports, worker_port_list="26900-26931")
    with pytest.raises(ValueError, match="explicit comma-separated"):
        ports.all_ports()


def test_durable_cutover_script_disarms_rollback_only_before_foreground_driver() -> None:
    script = Path("scripts/cosmos3_guarded_cutover.sh").read_text()
    assert "--in-tmux" in script
    assert "tmux new-window -d -t \"$SESSION\"" in script
    assert "rollback_armed=0" in script
    assert "trap - ERR INT TERM" in script
    assert "exec \"$PY\" -m ego_annotation.serving.cosmos3_resident_driver" in script
    assert "ray.scripts.scripts stop" not in script


@pytest.mark.parametrize(
    ("count", "returncode", "mode"),
    ((0, 0, "cold_start"), (1, 0, "cutover"), (2, 2, "")),
)
def test_durable_cutover_classifies_cold_start_and_live_bare_processes(
    count: int, returncode: int, mode: str
) -> None:
    result = subprocess.run(
        ["bash", "scripts/cosmos3_guarded_cutover.sh", "--classify-bare-count", str(count)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == returncode
    assert result.stdout.strip() == mode
    if count > 1:
        assert "expected zero or one bare Cosmos3 process" in result.stderr
