from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts" / "deploy"


def _read(name: str) -> str:
    return (DEPLOY / name).read_text()


def test_all_eighteen_service_scripts_are_versioned() -> None:
    scripts = sorted(path.name for path in DEPLOY.glob("*.sh"))
    assert scripts == [
        "deploy_all.sh",
        "deploy_cosmos.sh",
        "deploy_dispatcher.sh",
        "deploy_droid.sh",
        "deploy_hands.sh",
        "deploy_hawor.sh",
        "deploy_manager.sh",
        "deploy_unidepth.sh",
        "deploy_wilor.sh",
        "teardown_all.sh",
        "teardown_cosmos.sh",
        "teardown_dispatcher.sh",
        "teardown_droid.sh",
        "teardown_hands.sh",
        "teardown_hawor.sh",
        "teardown_manager.sh",
        "teardown_unidepth.sh",
        "teardown_wilor.sh",
    ]


def test_deploy_scripts_accept_only_2xx_health_and_return_after_empty_port_scan() -> None:
    for script in DEPLOY.glob("deploy_*.sh"):
        text = script.read_text()
        if "http_ready()" not in text:
            continue
        assert "^2[0-9][0-9]$" in text, script.name
        assert "^[234][0-9][0-9]$" not in text, script.name
        assert '  done\n  return 0\n}\nstart_window()' in text, script.name


def test_hawor_requires_both_serve_applications() -> None:
    text = _read("deploy_hawor.sh")
    assert "wait_for_hawor_apps" in text
    assert "hawor-infer-tracks:" in text
    assert "hawor-infiller-fill:" in text
    assert 'wait_for_hawor_apps "$PY" 26604 28003 900' in text


def test_teardowns_use_exact_ray_manifests_without_global_or_broad_kills() -> None:
    helper = (DEPLOY / "lifecycle_common.bash").read_text()
    assert "--gcs_server_port=${gcs_port}" in helper
    assert "exact manifest PIDs" in helper
    for script in DEPLOY.glob("teardown_*.sh"):
        text = script.read_text()
        assert "collect_matching_pids" not in text, script.name
        assert "pkill" not in text, script.name
        assert "fuser" not in text, script.name
        assert "ray stop" not in text, script.name
        assert "kill -TERM" not in text or "|| true" not in text, script.name


def test_droid_importers_stop_before_ipc_owners() -> None:
    text = _read("teardown_droid.sh")
    last_importer = text.index('DROID GPU2 importer r1')
    first_owner = text.index('DROID GPU7 owner')
    assert last_importer < first_owner
    assert "handles.pkl" in text


def test_dispatcher_and_manager_contracts_are_preserved() -> None:
    dispatcher = _read("deploy_dispatcher.sh")
    manager = _read("deploy_manager.sh")
    assert "ego_lane_dispatcher_leases.sqlite3" in dispatcher
    assert "lane_dispatcher" in dispatcher
    assert "serve_v22_annotation_api.py" in manager


def test_service_batch_timeouts_are_unified_to_two_seconds() -> None:
    serving = ROOT / "deploy" / "releases" / "runtime_current" / "ego_annotation" / "serving"
    unidepth = (serving / "deployment.py").read_text()
    hands = (serving / "hands_deployment.py").read_text()
    hawor = (serving / "hawor_deployment.py").read_text()
    assert 'EGO_UNIDEPTH_EXPERIMENT_BATCH_WAIT_MS", 2000' in unidepth
    assert unidepth.count("batch_wait_timeout_s=2.0") == 1
    assert hands.count("batch_wait_timeout_s=2.0") == 2
    assert hawor.count("batch_wait_timeout_s=2.0") == 2


def test_service_batch_tui_is_the_verified_artifact() -> None:
    tui = ROOT / "scripts" / "service_batch_tui.py"
    assert hashlib.sha256(tui.read_bytes()).hexdigest() == (
        "c7232f8652c155266c2c7738c12873a5f095ab2d04d2a36f5b74e4802be7b704"
    )
