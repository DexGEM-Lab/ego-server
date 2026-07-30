from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from ego_annotation.full_video_timeline import ModuleTimingCollector
from ego_annotation.scripted.contracts import ClientRequestTiming
from scripts.run_v22_api_egoscale30h_batch import build_service_performance_metrics


def test_concurrent_request_aggregation_and_stage_wall_are_distinct() -> None:
    collector = ModuleTimingCollector()
    timing = ClientRequestTiming(0.2, 0.5, 0.1, 0.8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: collector.request("hands.detect", timing), range(8)))
    breakdown, notes = collector.breakdown({"hands": 1.25})
    row = breakdown["hands"]
    assert row["request_count"] == 8
    assert row["client_prepare_s"] == pytest.approx(1.6)
    assert row["transport_wait_s"] == pytest.approx(4.0)
    assert row["client_decode_postprocess_s"] == pytest.approx(0.8)
    assert row["total_wall_s"] == 1.25
    assert "hands" not in notes


def test_service_observer_emits_capacity_and_unavailable_gpu_fields(tmp_path) -> None:
    metrics = build_service_performance_metrics(
        [{"service_lane_traces": {"unidepth": {"request_count": 4, "work_units": 40, "started_monotonic_s": 1.0, "completed_monotonic_s": 3.0}}}],
        output_root=tmp_path,
        manager_limit=128,
        algorithm_multiplier=2,
        outer_limit=32,
    )
    row = metrics["services"]["unidepth"]
    assert row["observed_req_per_s"] == 2.0
    assert row["effective_img_per_s"] == 20.0
    assert metrics["design_limits"]["outer_concurrency"] == 32
    assert metrics["nvidia_smi"]["status"] == "not_collected_by_client_only_task"
    assert (tmp_path / "service_performance_metrics.json").is_file()


def test_unavailable_timing_preserves_explicit_reason() -> None:
    collector = ModuleTimingCollector()
    collector.request("wilor.reconstruct", ClientRequestTiming(total_wall_s=0.25, available=False, unavailable_reason="mock backend has no timing boundary"))
    breakdown, notes = collector.breakdown({"wilor_service": 0.25})
    assert breakdown["wilor_service"]["request_count"] == 1
    assert breakdown["wilor_service"]["transport_wait_s"] == 0.0
    assert notes["wilor_service"] == "mock backend has no timing boundary"
