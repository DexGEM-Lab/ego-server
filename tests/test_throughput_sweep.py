from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

import httpx

from ego_annotation.serving.router import ModelServiceRouter
from scripts import throughput_sweep as sweep


def run(coro):
    return asyncio.run(coro)


def test_default_mode_is_a_persistent_sixty_second_window() -> None:
    args = sweep.parse_args(["--apis", "unidepth"])
    assert args.duration_s == 60.0
    assert args.requests_per_worker is None


def test_canonical_routes_preserve_dispatchers_and_direct_service_lanes() -> None:
    routes = {name: sweep.route_for_service(ModelServiceRouter.canonical(), name) for name in (
        "unidepth", "hands", "wilor", "droid", "hawor", "infiller", "cosmos",
    )}
    assert routes["unidepth"] == {"unidepth.infer": "http://127.0.0.1:28000/unidepth.infer"}
    assert routes["droid"] == {
        "droid.create_session": "http://127.0.0.1:28002/droid.create_session",
        "droid.push_frame": "http://127.0.0.1:28002/droid.push_frame",
        "droid.finalize": "http://127.0.0.1:28002/droid.finalize",
    }
    assert routes["hands"] == {"hands.detect": "http://127.0.0.1:28001/hands.detect"}
    assert routes["wilor"] == {"wilor.reconstruct": "http://127.0.0.1:28004/wilor.reconstruct"}
    assert routes["hawor"] == {"hawor.infer_tracks": "http://127.0.0.1:28003/hawor.infer_tracks"}
    assert routes["infiller"] == {"hawor_infiller.fill": "http://127.0.0.1:28003/hawor_infiller.fill"}
    assert routes["cosmos"] == {"cosmos3.reason": "http://127.0.0.1:28006/cosmos3.reason"}


def test_level_aggregates_request_and_native_work_throughput() -> None:
    level = sweep.LevelResult(
        service="droid", offered_concurrency=4, duration_s=2.0,
        completed_requests=8, attempted_requests=10, failed_requests=2, work_units=12,
        latencies_ms=[10.0, 20.0, 30.0, 40.0], batch_sizes=[2.0, 4.0], status_429=1, status_503=1,
    )
    assert level.req_s == 4.0
    assert level.img_s == 6.0
    assert level.error_rate == 0.2
    assert level.latency_p50_ms == 25.0
    assert level.latency_p95_ms == 38.5
    assert level.batch_size_mean == 3.0
    assert level.to_dict()["work_unit"] == "frame/s"


def test_shared_gpu_budget_caps_combined_logical_service_concurrency() -> None:
    assigned = sweep.allocate_card_budget(
        {"hands": 32, "wilor": 32, "hawor": 16, "infiller": 16, "unidepth": 32},
        gpu1_cap=32, gpu3_cap=20,
    )
    assert assigned["hands"] + assigned["wilor"] == 32
    assert assigned["hawor"] + assigned["infiller"] == 20
    assert assigned["unidepth"] == 32


def test_knee_detection_confirms_two_low_gain_levels_or_stops_on_decline() -> None:
    def point(load: int, throughput: float) -> sweep.LevelResult:
        return sweep.LevelResult("unidepth", load, 1.0, int(throughput), int(throughput), 0, int(throughput))

    saturation = sweep.find_knee([point(1, 10), point(2, 19), point(4, 20), point(8, 20.2)], 0.10)
    decline = sweep.find_knee([point(1, 10), point(2, 15), point(4, 14)], 0.10)
    assert saturation["knee_offered_concurrency"] == 4
    assert saturation["confirmation_offered_concurrency"] == 8
    assert saturation["reason"] == "consecutive_marginal_gain_below_threshold"
    assert decline["knee_offered_concurrency"] == 4
    assert decline["reason"] == "throughput_declined"


def test_guard_declares_correct_unidepth_and_wilor_topology() -> None:
    assert sweep.SERVICE_EXPECTED_GPUS["unidepth"] == frozenset({0, 5})
    assert sweep.SERVICE_EXPECTED_GPUS["wilor"] == frozenset({4})
    assert sweep.expected_gpus_for_services(sweep.SERVICE_EXPECTED_GPUS) == frozenset(range(8))


def test_external_gpu_guard_ignores_droid_dispatch_to_gpu7() -> None:
    before = {"cards": [
        {"index": 2, "memory_used_mib": 1000.0, "processes": [{"pid": 200, "memory_used_mib": 1000.0}]},
        {"index": 7, "memory_used_mib": 2.0, "processes": []},
    ]}
    after = {"cards": [
        {"index": 2, "memory_used_mib": 1200.0, "processes": [{"pid": 200, "memory_used_mib": 1200.0}]},
        {"index": 7, "memory_used_mib": 1800.0, "processes": [{"pid": 701, "memory_used_mib": 1798.0}]},
    ]}

    guard = sweep.external_gpu_guard(before, after, ("droid",))

    assert guard["expected_service_gpus"] == {"droid": [2, 7]}
    assert guard["expected_gpus"] == list(range(8))
    assert guard["abort"] is False
    assert guard["violations"] == []
    assert guard["ignored_expected_gpu_activity"] == [{
        "gpu_index": 7,
        "memory_delta_mib": 1798.0,
        "new_cuda_processes": [{"pid": 701, "memory_used_mib": 1798.0}],
    }]


def test_external_gpu_guard_accepts_owned_gpu4_gpu5_baseline_residency() -> None:
    before = {"cards": [
        {"index": 4, "memory_used_mib": 3641.0, "processes": [{"pid": 401, "memory_used_mib": 3632.0}]},
        {"index": 5, "memory_used_mib": 1781.0, "processes": [{"pid": 501, "memory_used_mib": 1772.0}]},
    ]}
    after = {"cards": [
        {"index": 4, "memory_used_mib": 3641.0, "processes": [{"pid": 401, "memory_used_mib": 3632.0}]},
        {"index": 5, "memory_used_mib": 1781.0, "processes": [{"pid": 501, "memory_used_mib": 1772.0}]},
    ]}

    guard = sweep.external_gpu_guard(before, after, ("unidepth", "wilor"))

    assert guard["expected_service_gpus"] == {"unidepth": [0, 5], "wilor": [4]}
    assert guard["abort"] is False
    assert guard["new_unowned_cuda_activity"] == []
    assert guard["ignored_expected_gpu_activity"] == []


def test_external_gpu_guard_aborts_for_new_pid_on_service_free_gpu() -> None:
    before = {"cards": [{"index": 8, "memory_used_mib": 2.0, "processes": []}]}
    after = {"cards": [{"index": 8, "memory_used_mib": 1026.0, "processes": [{"pid": 801, "memory_used_mib": 1024.0}]}]}

    guard = sweep.external_gpu_guard(before, after, ("unidepth",))

    assert guard["abort"] is True
    assert guard["violations"] == [{
        "gpu_index": 8,
        "memory_delta_mib": 1024.0,
        "new_cuda_processes": [{"pid": 801, "memory_used_mib": 1024.0}],
    }]


class _BarrierWorkload(sweep.SweepWorkload):
    def __init__(self, service: str, started: set[str], all_started: asyncio.Event) -> None:
        self.service = service
        self.endpoint = f"http://fake/{service}"
        self._started = started
        self._all_started = all_started

    async def run_cycle(self, _client, _run_id: str, _worker_index: int) -> sweep.CycleResult:
        self._started.add(self.service)
        if len(self._started) == 2:
            self._all_started.set()
        await self._all_started.wait()
        return sweep.CycleResult(({"success": True, "latency_ms": 1.0, "http_status": 200, "trace": {"request_count": 2}},), 1)


class _SnapshotClient:
    async def get(self, url: str) -> httpx.Response:
        return httpx.Response(200, json={"endpoint": url})


def test_service_levels_release_all_services_together_before_any_cycle_completes() -> None:
    async def scenario() -> None:
        started: set[str] = set()
        all_started = asyncio.Event()
        release = asyncio.Event()
        first_ready, second_ready = asyncio.Event(), asyncio.Event()
        client = _SnapshotClient()
        first = _BarrierWorkload("unidepth", started, all_started)
        second = _BarrierWorkload("hands", started, all_started)
        tasks = [
            asyncio.create_task(sweep.run_service_level(first, client, offered_concurrency=1, requested_concurrency=1, duration_s=None, requests_per_worker=1, run_id="r", ready=first_ready, release=release)),
            asyncio.create_task(sweep.run_service_level(second, client, offered_concurrency=1, requested_concurrency=1, duration_s=None, requests_per_worker=1, run_id="r", ready=second_ready, release=release)),
        ]
        await asyncio.gather(first_ready.wait(), second_ready.wait())
        assert not started
        release.set()
        results = await asyncio.gather(*tasks)
        assert started == {"unidepth", "hands"}
        assert all(result.cycles_completed == 1 for result in results)

    run(scenario())


class _FailingPushClient(sweep.FakeSweepClient):
    async def post(self, url: str, *, content: bytes, headers: Mapping[str, str]) -> httpx.Response:
        if url.endswith("/droid.push_frame"):
            self.calls.append(url)
            return httpx.Response(503, json={"error": {"code": "backpressure"}}, headers={"content-type": "application/json"}, request=httpx.Request("POST", url))
        return await super().post(url, content=content, headers=headers)


def test_droid_failure_still_finalizes_and_fake_affinity_returns_to_zero(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> tuple[list[sweep.LevelResult], dict[str, Any], _FailingPushClient]:
        monkeypatch.setattr(sweep, "gpu_memory_snapshot", lambda: {"cards": []})
        client = _FailingPushClient()
        levels, report = await sweep.run_sweep(
            sweep.fake_workloads(("droid",)), {"droid": (1,)}, client=client,
            duration_s=None, requests_per_worker=1, artifact_dir=tmp_path,
            droid_db_path=None, fake_server=True, knee_threshold=0.1,
        )
        return levels, report, client

    levels, report, client = run(scenario())
    assert levels[0].cycles_failed == 1
    assert levels[0].status_503 == 1
    assert client.sessions == set()
    assert len(client.finalized_sessions) == 1
    assert report["droid_leases_after"]["fake_sessions"] == 0
    written = json.loads((tmp_path / "levels" / "droid_load_1.json").read_text())
    assert written["http_503_count"] == 1


def test_fake_sweep_writes_per_level_csv_plot_and_knee_report(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> tuple[list[sweep.LevelResult], dict[str, Any]]:
        monkeypatch.setattr(sweep, "gpu_memory_snapshot", lambda: {"cards": []})
        client = sweep.FakeSweepClient()
        return await sweep.run_sweep(
            sweep.fake_workloads(("unidepth", "hands")), {"unidepth": (1, 2), "hands": (1, 2)},
            client=client, duration_s=None, requests_per_worker=1, artifact_dir=tmp_path,
            droid_db_path=None, fake_server=True, knee_threshold=0.1,
        )

    levels, report = run(scenario())
    sweep.write_summary_csv(tmp_path / "summary.csv", levels)
    paths = sweep.plot_throughput_latency(tmp_path / "summary.csv", tmp_path / "plots")
    (tmp_path / "knee_report.json").write_text(json.dumps(report))
    assert len(levels) == 4
    assert (tmp_path / "levels" / "unidepth_load_2.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert {path.name for path in paths} == {"throughput_latency_unidepth.png", "throughput_latency_hands.png"}
    assert json.loads((tmp_path / "knee_report.json").read_text())["knees"]["hands"]["knee_offered_concurrency"] is not None
