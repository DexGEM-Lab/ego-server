"""CPU-only contracts for the DROID process-sharded horizontal benchmark."""
from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts/run_droid_client_sharding_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_droid_client_sharding_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
shards = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shards
SPEC.loader.exec_module(shards)


def _plan() -> dict[str, object]:
    replicas = []
    for index, gpu in enumerate((4, 5, 7)):
        replica_id = f"droid-exp-hr3-gpu{gpu}"
        replicas.append({
            "replica_id": replica_id,
            "gpu_id": gpu,
            "endpoint": f"http://127.0.0.1:{32000 + index * 100}",
            "expected_server_identity": {
                "replica_id": replica_id,
                "assigned_gpu": gpu,
                "gcs_address": f"127.0.0.1:{30000 + index * 400}",
            },
        })
    return {
        "schema": "ego.droid-scaling-launch-plan.v1", "experiment_id": "hr3",
        "application_release": {"path": "/release", "release_digest": "release"},
        "launch_configuration": {"cpu_offload": True, "max_sessions": 32, "max_concurrent_ba": 1},
        "replicas": replicas,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _child_output(child, *, first: float, final: float) -> None:
    replica = child.replica_id
    _write_json(child.run_root / "droid" / "summary.json", {
        "release_digest": "release",
        "rows": [
            {
                "operation": "push_frame", "sticky_replicas": {replica: 256},
                "planned_session_assignment": {replica: 32}, "completed_count": 160,
                "completed_rate_per_s": 46.08,
                "per_replica_actual_offer": {replica: {
                    "first_submit_s": first, "final_submit_s": final, "actual_offer_window_s": final - first,
                }},
            },
            {
                "operation": "finalize", "completed_count": 32, "semantic_valid_count": 32,
                "finite_pose_ratio": {"count": 32, "min": 1.0, "max": 1.0},
            },
        ],
    })
    samples = [
        {"gpu_id": child.gpu_id, "utilization_gpu_pct": 20.0, "memory_used_bytes": 1024},
        {"gpu_id": child.gpu_id, "utilization_gpu_pct": 80.0, "memory_used_bytes": 2048},
    ]
    gpu_path = child.run_root / "droid" / "raw" / "gpu_samples.json"
    _write_json(gpu_path, {"samples": samples})
    _write_json(child.run_root / "droid" / "run_manifest.json", {"nvml": {"path": str(gpu_path)}})


def test_hr3_child_specs_create_one_identity_view_per_endpoint(tmp_path: Path) -> None:
    plan_path = tmp_path / "launch_plan.json"
    _write_json(plan_path, _plan())
    plan, children = shards.build_child_specs(plan_path, tmp_path / "run")
    assert plan["experiment_id"] == "hr3"
    assert len(children) == 3
    assert len({child.endpoint for child in children}) == 3
    for child in children:
        identity_view = json.loads(child.identity_path.read_text(encoding="utf-8"))
        assert len(identity_view["replicas"]) == 1
        assert identity_view["replicas"][0]["replica_id"] == child.replica_id


def test_resource_wrapper_targets_benchmark_script_not_interpreter(tmp_path: Path) -> None:
    plan_path = tmp_path / "launch_plan.json"
    _write_json(plan_path, _plan())
    _, children = shards.build_child_specs(plan_path, tmp_path / "run")
    args = SimpleNamespace(
        python="/runtime/python", benchmark_script="/release/bench.py", preserved_payload_manifest=tmp_path / "payloads.json",
        payload_count=512, waves=8, session_buffer=256, sessions=32, wave_rate=4.0,
        start_delay_s=0.25, timeout_s=120.0, corpus_digest="corpus", measurement_interval_id="interval",
    )
    command = shards._resource_wrapped_child_command(args, children[0], "wrapper")
    assert command[:4] == ["/runtime/python", "-c", "wrapper", "/release/bench.py"]
    assert command.count("/runtime/python") == 1


def test_hr3_aggregate_requires_sticky_finite_terminals_and_uses_common_offer_window(tmp_path: Path) -> None:
    plan_path = tmp_path / "launch_plan.json"
    _write_json(plan_path, _plan())
    plan, children = shards.build_child_specs(plan_path, tmp_path / "run")
    for index, child in enumerate(children):
        _child_output(child, first=10.0 + index * 0.1, final=13.5 + index * 0.1)
    report = shards.aggregate(plan, children, cpu_seconds={child.replica_id: 1.0 for child in children},
                              wall_seconds={child.replica_id: 5.0 for child in children})
    measurement = report["measurement"]
    assert measurement["all_terminals_finite"] is True
    assert measurement["completed_pushes"] == 480
    assert measurement["aggregate_actual_offer_window_s"] == pytest.approx(3.7)
    assert report["process_sharding"]["httpx_stacks"] == 3
    assert all(item["client"]["one_core_cpu_fraction"] == 0.2 for item in report["per_replica"].values())
