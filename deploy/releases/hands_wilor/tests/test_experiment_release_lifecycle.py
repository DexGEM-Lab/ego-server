"""GPU-free adversarial tests for content identity and scoped lifecycle."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess

import pytest

from ego_annotation.serving.benchmark.release import artifact_digest, build_release, verify_release
from ego_annotation.serving.benchmark.unidepth_driver import _load_application
from ego_annotation.serving.benchmark.unidepth_scaling import (
    CommandOutcome,
    ExperimentConfigurationError,
    build_unidepth_scaling_plan,
    execute_scoped_plan,
    preflight_check_command,
    production_health_check_commands,
)

SOURCE_SHA = "a" * 40


def _source(root: Path) -> Path:
    (root / "ego_annotation" / "serving").mkdir(parents=True)
    for package in (root / "ego_annotation" / "__init__.py", root / "ego_annotation" / "serving" / "__init__.py"):
        package.write_text("")
    (root / "ego_annotation" / "serving" / "deployment.py").write_text("app = 'release-app'\n")
    (root / "released_app.py").write_text("app = 'release-app'\n")
    return root


def _release(tmp_path: Path) -> Path:
    return build_release(_source(tmp_path / "src"), tmp_path / "releases", source_sha=SOURCE_SHA)


def _plan(tmp_path: Path, *, gpus=(5,)):
    release = _release(tmp_path)
    return build_unidepth_scaling_plan(
        experiment_id="hardening", gpu_ids=gpus, run_root=tmp_path / "runs", application_release_path=release,
        source_sha=SOURCE_SHA, checkpoint_digest="checkpoint-digest", gpu_uuids=tuple(f"GPU-{g}" for g in gpus),
    )


def test_release_is_digest_named_and_detects_code_mutation(tmp_path):
    release = _release(tmp_path)
    verified = verify_release(release, expected_source_sha=SOURCE_SHA)
    assert release.name == verified.release_digest
    (release / "released_app.py").write_text("app = 'mutated'\n")
    with pytest.raises(ValueError, match="manifest"):
        verify_release(release)


def test_release_import_ignores_shadow_cwd(tmp_path, monkeypatch):
    release = _release(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "released_app.py").write_text("app = 'shadow-app'\n")
    monkeypatch.chdir(shadow)
    monkeypatch.syspath_prepend(str(shadow))
    assert _load_application(release, "released_app:app") == "release-app"


def test_release_verification_ignores_derived_python_bytecode(tmp_path):
    release = _release(tmp_path)
    cache = release / "__pycache__"
    cache.mkdir()
    (cache / "released_app.cpython-311.pyc").write_bytes(b"derived-bytecode")
    assert verify_release(release, expected_source_sha=SOURCE_SHA).path == release


def test_preflight_cannot_mask_occupied_gpu_with_matching_uuid(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *query-compute-apps*) echo 4242 ;;\n"
        "  *memory.free*) echo 80000 ;;\n"
        "  *uuid*) echo GPU-5 ;;\n"
        "esac\n"
    )
    nvidia_smi.chmod(0o755)
    ss = fake_bin / "ss"
    ss.write_text("#!/bin/sh\nexit 0\n")
    ss.chmod(0o755)
    environment = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    outcome = subprocess.run(
        preflight_check_command(plan.replicas[0]), shell=True, env=environment,
        text=True, capture_output=True,
    )
    assert outcome.returncode != 0


def test_production_health_command_propagates_http_404(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text("#!/bin/sh\nprintf 404\n")
    curl.chmod(0o755)
    environment = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    outcome = subprocess.run(
        production_health_check_commands()[0], shell=True, env=environment,
        text=True, capture_output=True,
    )
    assert outcome.returncode != 0
    assert "ray_health=404" in outcome.stderr


def test_checkpoint_digest_is_derived_from_actual_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights-a")
    first = artifact_digest(checkpoint)
    (checkpoint / "weights.bin").write_bytes(b"weights-b")
    assert artifact_digest(checkpoint) != first


def test_ray_command_uses_detached_driver_not_invalid_port_cli(tmp_path):
    plan = _plan(tmp_path)
    _start, driver = plan.replicas[0].launch_commands()
    assert "experiment_driver" in driver
    assert "--http-port 31000" in driver
    assert "--port 31000" not in driver
    from ego_annotation.serving.benchmark import experiment_driver
    source = inspect.getsource(experiment_driver.run_driver)
    assert "ray.init(address=gcs_address" in source
    assert "detached=True" in source
    assert "blocking=False" in source
    assert "name=app_name" in source and "route_prefix=route_prefix" in source
    assert experiment_driver.EXPERIMENT_APPLICATIONS == {
        "unidepth": "ego_annotation.serving.deployment:app",
        "droid": "ego_annotation.serving.droid_deployment:app",
        "hawor": "ego_annotation.serving.hawor_deployment:app",
        "hands": "ego_annotation.serving.hands_deployment:hands_app",
    }


class Runner:
    def __init__(self, fail_predicate=lambda command, index: False):
        self.commands: list[str] = []
        self.fail_predicate = fail_predicate
    def run(self, command: str) -> CommandOutcome:
        self.commands.append(command)
        failed = self.fail_predicate(command, len(self.commands))
        return CommandOutcome(command, 1 if failed else 0, stderr="forced failure" if failed else "")


def test_partial_nonzero_start_is_cleanup_owned(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner(lambda command, _index: "ray.scripts.scripts start" in command)
    with pytest.raises(ExperimentConfigurationError, match="Ray start failed"):
        execute_scoped_plan(plan, command_runner=runner, typed_readiness_probe=lambda _: None, preflight_runner=lambda _: None)
    assert any("--scoped-stop" in command for command in runner.commands)
    assert not any("shutdown -a" in command for command in runner.commands)


def test_successful_head_rechecks_gpu_in_same_command_as_detached_deploy(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner()
    result = execute_scoped_plan(
        plan, command_runner=runner, typed_readiness_probe=lambda _: None,
        preflight_runner=lambda _: None,
    )
    assert result.readiness_replica_ids == ("unidepth-exp-hardening-gpu5",)
    deploy = next(
        command for command in runner.commands
        if "experiment_driver" in command and "--app-choice unidepth" in command
    )
    assert "query-compute-apps=pid" in deploy
    assert " && " in deploy


def test_later_preflight_failure_never_stops_unstarted_candidate(tmp_path):
    plan = _plan(tmp_path, gpus=(5, 7))
    runner = Runner()

    def preflight(replica):
        if replica.gpu_id == 7:
            raise ExperimentConfigurationError("GPU7 occupied")

    with pytest.raises(ExperimentConfigurationError, match="GPU7 occupied"):
        execute_scoped_plan(
            plan, command_runner=runner, typed_readiness_probe=lambda _: None,
            preflight_runner=preflight,
        )
    commands = "\n".join(runner.commands)
    assert "http://127.0.0.1:29004" in commands  # started GPU5 is cleanup-owned
    assert "http://127.0.0.1:29204" not in commands  # rejected GPU7 was never ours


def test_keyboard_interrupt_still_rolls_back(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner()
    with pytest.raises(KeyboardInterrupt):
        execute_scoped_plan(plan, command_runner=runner, typed_readiness_probe=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()), preflight_runner=lambda _: None)
    assert any("--scoped-stop" in command for command in runner.commands)


def test_stop_failure_and_survivor_are_preserved(tmp_path):
    plan = _plan(tmp_path)
    runner = Runner(lambda command, _index: "shutdown -a" in command)
    with pytest.raises(ExperimentConfigurationError) as raised:
        execute_scoped_plan(
            plan, command_runner=runner,
            typed_readiness_probe=lambda _: (_ for _ in ()).throw(ExperimentConfigurationError("readiness failed")),
            preflight_runner=lambda _: None,
            cleanup_verifier=lambda _: (_ for _ in ()).throw(RuntimeError("survivor pid 123; port 31000 open")),
        )
    message = str(raised.value)
    assert "cleanup failures preserved" in message and "survivor pid 123" in message
