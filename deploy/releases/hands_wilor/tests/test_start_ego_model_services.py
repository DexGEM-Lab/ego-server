"""GPU-free tests for the idempotent Ego Ray Serve group launcher.

These exercise command generation, health-gated skip/idempotency, stale/dead window
replacement, group selection, the corrected GPU3 interpreter, and Cosmos3 delegation.
Nothing here touches a GPU, a real tmux server, or a network socket: the health probe
and the tmux runner are injected.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from ego_annotation.serving.lifecycle import COMMITTED_GPU_GROUPS
from scripts import start_ego_model_services as starter


def _group(gpu_id: int):
    return next(g for g in COMMITTED_GPU_GROUPS if g.gpu_id == gpu_id)


class FakeTmux:
    """Records every tmux/bash argv and serves a fixed initial window list."""

    def __init__(self, existing_windows: set[str] | None = None, *, session_present: bool = True) -> None:
        self.existing = set(existing_windows or set())
        self.session_present = session_present
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        if argv[:2] == ["tmux", "list-windows"]:
            if not self.session_present:
                return subprocess.CompletedProcess(argv, 1, "", "no server")
            return subprocess.CompletedProcess(argv, 0, "\n".join(sorted(self.existing)) + "\n", "")
        if argv[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(argv, 0 if self.session_present else 1, "", "")
        if argv[:3] == ["setsid", "tmux", "new-session"]:
            self.session_present = True
        return subprocess.CompletedProcess(argv, 0, "", "")

    def new_window_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["tmux", "new-window"]]

    def kill_window_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["tmux", "kill-window"]]


def all_down(host: str, port: int) -> bool:
    return False


def all_up(host: str, port: int) -> bool:
    return True


# --- Generated tmux command content -----------------------------------------


def test_serve_window_argv_pins_gpu_ports_addresses_and_no_ambient_ray_address():
    argv = starter.serve_window_argv("ego_annotation", _group(0))
    assert argv[:7] == ["tmux", "new-window", "-d", "-t", "ego_annotation", "-n", "ego-serve-gpu0"]
    assert argv[7:9] == ["bash", "-lc"]
    script = argv[9]
    # physical CUDA pinning + native single GPU resource
    assert "CUDA_VISIBLE_DEVICES=0 " in script
    assert "--num-gpus=1" in script
    # disjoint explicit component + worker ports come straight from lifecycle
    assert "--port=26000" in script and "--dashboard-port=26004" in script
    assert "--worker-port-list=26100," in script
    # explicit GCS address on detached deployment driver + explicit HTTP lane port
    assert "--address 127.0.0.1:26000 --dashboard-address http://127.0.0.1:26004 --port 28000" in script
    assert "AUTOSCALER_METRIC_PORT=26011" in script
    assert "DASHBOARD_METRIC_PORT=26012" in script
    assert "--metrics-export-port=26010" in script
    assert script.count("env -u RAY_ADDRESS") == 2
    # Only the start/driver brace block writes to the log; the owner never tails it.
    assert "} >> /tmp/ego-serve-windows/gpu0.log 2>&1" in script
    assert "exec >> /tmp/ego-serve-windows/gpu0.log 2>&1" not in script
    assert "-m scripts.serve_group_driver --gpu-id 0" in script
    assert "tail -n" not in script
    assert "exec sleep infinity" in script


def test_owner_script_never_self_follows_or_uses_a_stdout_logger_pipe():
    for gpu_id in (0, 1, 2, 3):
        script = starter.serve_launch_script(_group(gpu_id))
        assert "ray stop" not in script
        assert "tail -n" not in script and "tail -F" not in script
        assert "process substitution" not in script
        assert "|" not in script
        assert "exec sleep infinity" in script
        assert "pip install" not in script and "conda " not in script


def test_guard_and_observer_are_distinct_tmux_process_groups_with_ptty_only_tail():
    group = _group(0)
    guard = starter.log_guard_window_argv("ego_annotation", group)
    observer = starter.log_observer_window_argv("ego_annotation", group)
    assert guard[0:7] == ["tmux", "new-window", "-d", "-t", "ego_annotation", "-n", "ego-serve-log-guard-gpu0"]
    assert "guard_ego_serve_log.py" in guard[-1]
    assert "--threshold-bytes 67108864" in guard[-1]
    assert "--max-retained-bytes 8388608" in guard[-1]
    assert "--max-lines 5000" in guard[-1]
    assert observer[0:7] == ["tmux", "new-window", "-d", "-t", "ego_annotation", "-n", "ego-serve-log-observer-gpu0"]
    assert observer[-1] == "exec tail -n 5000 -F /tmp/ego-serve-windows/gpu0.log"
    assert ">>" not in observer[-1] and "gpu0.log" not in guard[-1].split("--log-path", 1)[0]
    assert starter.kill_named_window_argv("ego_annotation", starter.log_observer_window_name(0))[-1] == "ego_annotation:ego-serve-log-observer-gpu0"
    assert starter.kill_named_window_argv("ego_annotation", starter.log_observer_window_name(0))[-1] != "ego_annotation:ego-serve-gpu0"


def test_gpu3_serve_uses_ray_bearing_hawor_env_not_bare_hawor():
    script = starter.serve_launch_script(_group(3))
    assert "/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" in script
    assert "/envs/hawor/bin/python" not in script  # bare model env has no Ray
    assert "CUDA_VISIBLE_DEVICES=3 " in script
    assert "--address 127.0.0.1:26600 --dashboard-address http://127.0.0.1:26604 --port 28003" in script


def test_detached_group_driver_selects_split_and_gpu3_apps():
    driver = Path("scripts/serve_group_driver.py").read_text()
    assert "from ego_annotation.serving.hands_deployment import hands_only_app" in driver
    assert "from ego_annotation.serving.hands_deployment import wilor_only_app" in driver
    assert "from ego_annotation.serving.hawor_deployment import app, infiller_app" in driver
    assert 'route_prefix="/hawor.infer_tracks"' in driver
    assert 'route_prefix="/hawor_infiller.fill"' in driver


# --- Skip / idempotency ------------------------------------------------------


def test_healthy_groups_are_skipped_when_owner_guard_and_observer_exist():
    tmux = FakeTmux(existing_windows={
        name for group in COMMITTED_GPU_GROUPS if group.gpu_id != 6
        for name in (starter.window_name(group.gpu_id), starter.log_guard_window_name(group.gpu_id), starter.log_observer_window_name(group.gpu_id))
    })
    plans = starter.run(
        COMMITTED_GPU_GROUPS, status_only=False, dry_run=False,
        health_probe=all_up, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert all(p.action == starter.SKIP_HEALTHY for p in plans)
    assert tmux.new_window_calls() == []
    assert tmux.kill_window_calls() == []


def test_healthy_group_adds_missing_independent_guard_and_observer_without_owner_replace():
    group = _group(0)
    tmux = FakeTmux(existing_windows={starter.window_name(0)})
    plans = starter.run(
        [group], status_only=False, dry_run=False,
        health_probe=all_up, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.SKIP_HEALTHY
    assert tmux.kill_window_calls() == []
    names = {call[6] for call in tmux.new_window_calls()}
    assert names == {starter.log_guard_window_name(0), starter.log_observer_window_name(0)}


def test_observer_replacement_targets_only_its_disposable_tmux_window():
    group = _group(0)
    tmux = FakeTmux(existing_windows={
        starter.window_name(0), starter.log_guard_window_name(0), starter.log_observer_window_name(0),
    })
    starter.run(
        [group], status_only=False, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    targets = {call[-1] for call in tmux.kill_window_calls()}
    assert targets == {
        "ego_annotation:ego-serve-gpu0",
        "ego_annotation:ego-serve-log-guard-gpu0",
        "ego_annotation:ego-serve-log-observer-gpu0",
    }
    observer_kill = next(call for call in tmux.kill_window_calls() if call[-1].endswith("observer-gpu0"))
    assert observer_kill[-1] != "ego_annotation:ego-serve-gpu0"


def test_dead_guard_window_is_replaced_without_touching_healthy_owner():
    group = _group(0)
    plan = starter.plan_group(
        group, healthy=True, window_exists=True, guard_exists=False, observer_exists=True,
        guard_window_exists=True, observer_window_exists=True, status_only=False,
        session="ego_annotation", cosmos_run_root="/tmp/cosmos",
    )
    assert plan.action == starter.SKIP_HEALTHY
    assert plan.commands[0] == starter.kill_named_window_argv("ego_annotation", starter.log_guard_window_name(0))
    assert all("ego-serve-gpu0" not in command[-1] for command in plan.commands if command[:2] == ["tmux", "kill-window"])
    assert any(command[6] == starter.log_guard_window_name(0) for command in plan.commands if command[:2] == ["tmux", "new-window"])


def test_down_group_with_no_window_gets_fresh_window():
    tmux = FakeTmux(existing_windows=set())
    plans = starter.run(
        [_group(0)], status_only=False, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.START
    assert len(tmux.new_window_calls()) == 3
    assert tmux.kill_window_calls() == []


def test_down_group_creates_persistent_tmux_session_when_absent():
    tmux = FakeTmux(existing_windows=set(), session_present=False)
    starter.run(
        [_group(0)], status_only=False, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    creates = [call for call in tmux.calls if call[:3] == ["setsid", "tmux", "new-session"]]
    assert len(creates) == 1
    assert "ego_annotation" in creates[0]
    assert len(tmux.new_window_calls()) == 3


def test_status_only_makes_no_tmux_changes():
    tmux = FakeTmux(existing_windows={"ego-serve-gpu0"})
    plans = starter.run(
        [_group(0)], status_only=True, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.STATUS_ONLY
    assert tmux.new_window_calls() == [] and tmux.kill_window_calls() == []


def test_status_does_not_require_split_release_root(monkeypatch: pytest.MonkeyPatch):
    captured = {}
    monkeypatch.setattr(starter, "run", lambda groups, **kwargs: captured.update(groups=groups, **kwargs))
    assert starter.main(["--status"]) == 0
    assert {group.gpu_id for group in captured["groups"]} >= {1, 4}
    assert captured["status_only"] is True


def test_dry_run_makes_no_tmux_changes():
    tmux = FakeTmux(existing_windows=set())
    starter.run(
        [_group(0)], status_only=False, dry_run=True,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert tmux.new_window_calls() == [] and tmux.kill_window_calls() == []


# --- Stale / dead window replacement ----------------------------------------


def test_down_group_with_stale_window_is_replaced():
    tmux = FakeTmux(existing_windows={"ego-serve-gpu0"})
    plans = starter.run(
        [_group(0)], status_only=False, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.REPLACE
    kills = tmux.kill_window_calls()
    assert len(kills) == 1 and kills[0][-1] == "ego_annotation:ego-serve-gpu0"
    # The owner is replaced; independent support windows are created separately.
    assert len(tmux.new_window_calls()) == 3
    assert tmux.calls.index(kills[0]) < tmux.calls.index(tmux.new_window_calls()[0])


# --- Group selection ---------------------------------------------------------


def test_select_groups_default_is_all_six():
    assert tuple(g.gpu_id for g in starter.select_groups(None)) == (0, 1, 4, 2, 3, 6)


def test_select_groups_accepts_gpu_prefixed_and_bare_ids_in_order():
    assert tuple(g.gpu_id for g in starter.select_groups(["gpu3", "0"])) == (3, 0)


def test_select_groups_dedupes():
    assert tuple(g.gpu_id for g in starter.select_groups(["gpu0", "0"])) == (0,)


def test_select_groups_accepts_wilor_gpu4_and_rejects_unknown_group():
    assert tuple(g.gpu_id for g in starter.select_groups(["gpu4"])) == (4,)
    with pytest.raises(SystemExit):
        starter.select_groups(["gpu9"])


# --- Cosmos3 delegation ------------------------------------------------------


def test_cosmos_down_delegates_to_guarded_launcher_not_inline_serve():
    tmux = FakeTmux(existing_windows=set())
    plans = starter.run(
        [_group(6)], status_only=False, dry_run=False,
        health_probe=all_down, tmux_runner=tmux, host="127.0.0.1",
        cosmos_run_root="/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/ego_start_TEST",
        out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.START
    # No inline serve window is created for cosmos; the guarded launcher is invoked.
    assert tmux.new_window_calls() == []
    launcher_calls = [c for c in tmux.calls if c and c[0] == "bash" and c[1].endswith("cosmos3_guarded_cutover.sh")]
    assert len(launcher_calls) == 1
    assert launcher_calls[0][2:] == [
        "--run-root",
        "/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/ego_start_TEST",
        "--skip-benchmark",
    ]


def test_cosmos_healthy_is_skipped_and_launcher_not_invoked():
    tmux = FakeTmux(existing_windows=set())
    plans = starter.run(
        [_group(6)], status_only=False, dry_run=False,
        health_probe=all_up, tmux_runner=tmux, host="127.0.0.1", out=open("/dev/null", "w"),
    )
    assert plans[0].action == starter.SKIP_HEALTHY
    assert all(not (c and c[0] == "bash") for c in tmux.calls)


def test_cosmos_launcher_run_root_must_be_under_benchmark_root():
    argv = starter.cosmos_launcher_argv("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/x")
    assert argv[0] == "bash" and argv[1].endswith("scripts/cosmos3_guarded_cutover.sh")
    assert starter.default_cosmos_run_root().startswith(starter.COSMOS3_BENCHMARK_ROOT + "/")


# --- Failure surfacing -------------------------------------------------------


def test_tmux_command_failure_is_surfaced_not_swallowed():
    def failing(argv):
        if argv[:2] == ["tmux", "new-window"]:
            return subprocess.CompletedProcess(argv, 1, "", "no session")
        if argv[:2] == ["tmux", "list-windows"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(SystemExit):
        starter.run(
            [_group(0)], status_only=False, dry_run=False,
            health_probe=all_down, tmux_runner=failing, host="127.0.0.1", out=open("/dev/null", "w"),
        )


def test_committed_groups_reserve_gpu4_for_wilor_only():
    assert 4 in {g.gpu_id for g in COMMITTED_GPU_GROUPS}
    assert {g.gpu_id for g in COMMITTED_GPU_GROUPS}.isdisjoint({5, 7})


def test_gpu1_combined_launch_plan_is_explicit_rollback_mode():
    script = starter.serve_launch_script(_group(1), gpu1_combined=True)
    assert "--gpu-id 1" in script and script.endswith("exec sleep infinity")
    assert "--combined" in script
    with pytest.raises(ValueError, match="GPU1"):
        starter.serve_launch_script(_group(4), gpu1_combined=True)


def test_gpu1_envelope_launch_plan_sets_matching_attested_env_for_both_logical_apis():
    treatment = starter.with_gpu1_wire_format(_group(1), "envelope")
    env = dict(treatment.lifecycle.env_vars)
    assert env["EGO_HANDS_EXPERIMENT_WIRE_FORMAT"] == "envelope"
    assert env["EGO_WILOR_EXPERIMENT_WIRE_FORMAT"] == "envelope"
    assert env["EGO_HANDS_EXPERIMENT_TELEMETRY"] == "1"
    assert env["EGO_WILOR_EXPERIMENT_TELEMETRY"] == "1"
    assert "EGO_HANDS_EXPERIMENT_WIRE_FORMAT=envelope" in starter.serve_launch_script(treatment)


def test_gpu3_envelope_launch_plan_sets_matching_attested_env_for_both_logical_apis():
    treatment = starter.with_gpu3_wire_format(_group(3), "envelope")
    env = dict(treatment.lifecycle.env_vars)
    assert env["EGO_HAWOR_EXPERIMENT_WIRE_FORMAT"] == "envelope"
    assert env["EGO_HAWOR_INFILLER_EXPERIMENT_WIRE_FORMAT"] == "envelope"
    assert env["EGO_HAWOR_EXPERIMENT_TELEMETRY"] == "1"
    assert env["EGO_HAWOR_INFILLER_EXPERIMENT_TELEMETRY"] == "1"
    assert "EGO_HAWOR_EXPERIMENT_WIRE_FORMAT=envelope" in starter.serve_launch_script(treatment)
