from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_v22_minimal_annotation_pipeline import run_parallel_lanes, validate_feishu_ray_stage_selection


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_feishu_ray_complete_non_cosmos_path_is_enabled() -> None:
    validate_feishu_ray_stage_selection(
        SimpleNamespace(
            model_execution="feishu_ray",
            camera_backend="droid",
            run_camera_trajectory=True,
            run_hawor_metric_hands=True,
            run_hybrid_hands=True,
            run_captioning=False,
            skip_cosmos=True,
        )
    )


def test_feishu_ray_path_fails_closed_on_missing_dependencies_or_cosmos() -> None:
    with pytest.raises(RuntimeError, match="requires_run_camera_trajectory"):
        validate_feishu_ray_stage_selection(
            SimpleNamespace(
                model_execution="feishu_ray",
                camera_backend="droid",
                run_camera_trajectory=False,
                run_hawor_metric_hands=True,
                run_hybrid_hands=False,
                run_captioning=False,
                skip_cosmos=True,
            )
        )
    with pytest.raises(RuntimeError, match="cosmos_disabled"):
        validate_feishu_ray_stage_selection(
            SimpleNamespace(
                model_execution="feishu_ray",
                camera_backend="droid",
                run_camera_trajectory=True,
                run_hawor_metric_hands=True,
                run_hybrid_hands=True,
                run_captioning=True,
                skip_cosmos=False,
            )
        )


def test_run_parallel_lanes_records_group_and_preserves_lane_order(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"

    def lane(name: str, delay_s: float) -> list[dict]:
        time.sleep(delay_s)
        return [{"step": name, "status": "ok"}]

    started = time.time()
    rows, group = run_parallel_lanes(
        "unit_parallel_group",
        [
            ("slow", lambda: lane("slow_step", 0.05)),
            ("fast", lambda: lane("fast_step", 0.01)),
        ],
        events=events,
    )
    elapsed = time.time() - started

    assert group["status"] == "ok"
    assert rows == [{"step": "slow_step", "status": "ok"}, {"step": "fast_step", "status": "ok"}]
    assert elapsed < 0.09
    events_payload = read_events(events)
    assert events_payload[0]["event"] == "parallel_group_started"
    assert events_payload[-1]["event"] == "parallel_group_finished"
    assert events_payload[-1]["parallel_group"] == "unit_parallel_group"


def test_run_parallel_lanes_waits_for_all_lanes_before_raising(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    completed = []

    def ok_lane() -> list[dict]:
        time.sleep(0.02)
        completed.append("ok")
        return [{"step": "ok", "status": "ok"}]

    def failed_lane() -> list[dict]:
        raise RuntimeError("boom")

    try:
        run_parallel_lanes("failed_group", [("failed", failed_lane), ("ok", ok_lane)], events=events)
    except RuntimeError as exc:
        assert "parallel_group_failed:failed_group" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert completed == ["ok"]
    assert read_events(events)[-1]["status"] == "failed"
