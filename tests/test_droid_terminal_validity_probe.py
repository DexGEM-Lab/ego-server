"""CPU tests for the direct DROID terminal-validity replay classifier."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "droid_terminal_validity_probe", Path(__file__).parents[1] / "scripts" / "run_droid_terminal_validity_probe.py"
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def stream():
    return tuple(
        probe.StreamPayload(
            payload_id=f"payload-{index:04d}",
            source_frame_index=60 + 3 * index,
            timestamp_s=2.0 + 0.1 * index,
            rgb_path=Path(f"/rgb/{index}"), mask_path=Path(f"/mask/{index}"),
            rgb_sha256="rgb", mask_sha256="mask",
        )
        for index in range(3)
    )


def record_all_finite(recorder):
    values = np.ones((3, 7), dtype=np.float64)
    recorder.check("frontend_pre_backend_poses", values, stream())
    recorder.check("frontend_pre_backend_disparities", values, stream())
    recorder.check("backend_7", values, stream())
    recorder.check("backend_12", values, stream())
    recorder.check("filler_trajectory", values, stream())
    recorder.check("pose_conversion", np.ones((3, 4, 4), dtype=np.float64), stream())
    recorder.check("camera_state_validation", np.ones((3, 4, 4), dtype=np.float64), stream())


def test_finite_path_is_success_and_retains_per_stage_counts():
    recorder = probe.ProbeRecorder()
    record_all_finite(recorder)
    result = probe._verdict(recorder)

    assert result["boundary"] == "none/finite-success"
    assert result["first_bad"] is None
    assert result["finite_counts"]["frontend_pre_backend_poses"] == {"total_rows": 3, "finite_rows": 3, "first_bad": None}
    assert result["finite_counts"]["backend_7"] == {"total_rows": 3, "finite_rows": 3, "first_bad": None}
    assert result["finite_counts"]["camera_state_validation"]["finite_rows"] == 3


@pytest.mark.parametrize(
    ("stage", "expected_boundary"),
    [
        ("frontend_pre_backend_poses", "frontend_pre_backend"),
        ("frontend_pre_backend_disparities", "frontend_pre_backend"),
        ("backend_7", "backend_poses"),
        ("backend_12", "backend_poses"),
        ("filler_trajectory", "filler_trajectory"),
        ("pose_conversion", "pose_conversion"),
        ("camera_state_validation", "camera_state_validation"),
    ],
)
def test_injected_nan_names_its_first_terminal_boundary(stage, expected_boundary):
    recorder = probe.ProbeRecorder()
    values = np.ones((3, 7), dtype=np.float64)
    for name in (
        "frontend_pre_backend_poses", "frontend_pre_backend_disparities",
        "backend_7", "backend_12", "filler_trajectory", "pose_conversion", "camera_state_validation",
    ):
        candidate = values.copy()
        if name == stage:
            candidate[1, 0] = np.nan
        recorder.check(name, candidate, stream())

    result = probe._verdict(recorder)
    assert result["boundary"] == expected_boundary
    assert result["first_bad"] == {
        "row_index": 1,
        "payload_id": "payload-0001",
        "source_frame_index": 63,
        "source_timestamp_s": pytest.approx(2.1),
    }
    assert result["finite_counts"][stage]["finite_rows"] == 2


def test_model_grid_camera_attests_the_v19_to_v22_droid_resize_contract():
    camera = probe.exact_camera()
    comparison = probe.model_grid_calibration_comparison(camera, probe.DroidImageShape(320, 568))

    expected = comparison["expected_model_intrinsics_px"]
    assert expected["fx"] == pytest.approx(497.3572877383715)
    assert expected["fy"] == pytest.approx(498.135623044721)
    assert expected["cx"] == 284.0 and expected["cy"] == 160.0
    assert comparison["passed_over_expected_ratio"] == {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0}
    assert comparison["classification"] == "model_pixel_calibration_agreement"


def test_option_profiles_change_only_requested_droid_options():
    curve = probe.exact_options("curve")
    defaults = probe.exact_options("serving-defaults")

    assert curve.to_wire() == {
        "buffer": 128, "filter_thresh": 1.0, "warmup": 2, "keyframe_thresh": 2.0,
        "frontend_thresh": 16.0, "frontend_window": 25, "frontend_radius": 2,
        "frontend_nms": 1, "backend_thresh": 22.0, "backend_radius": 2,
        "backend_nms": 3, "upsample": True, "beta": 0.3, "stereo": False,
    }
    assert defaults.to_wire() == {
        "buffer": 1024, "filter_thresh": 2.4, "warmup": 8, "keyframe_thresh": 4.0,
        "frontend_thresh": 16.0, "frontend_window": 25, "frontend_radius": 2,
        "frontend_nms": 1, "backend_thresh": 22.0, "backend_radius": 2,
        "backend_nms": 3, "upsample": True, "beta": 0.3, "stereo": False,
    }
    assert probe.journal_capacity(curve) == 129
    assert probe.journal_capacity(defaults) == 1025


def test_frontend_stats_reports_finite_nonmetric_gauge_and_actual_options():
    video = SimpleNamespace(
        counter=SimpleNamespace(value=2),
        poses=np.asarray([[0, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1]], dtype=np.float64),
        disps=np.ones((2, 2, 3), dtype=np.float32),
        intrinsics=np.asarray([[51.12, 51.12, 35.5, 20.0], [51.12, 51.12, 35.5, 20.0]]),
    )
    state = SimpleNamespace(video=video, options=probe.exact_options("curve"))
    calibration = {"passed_session_intrinsics_px": {"fx": 497.3572877383715, "fy": 498.135623044721, "cx": 284.0, "cy": 160.0, "units": "model_image_pixels"}}
    stats = probe.frontend_state_stats(state, calibration)

    assert stats["options"] == probe.exact_options("curve").to_wire()
    assert stats["poses"]["finite_count"] == 14
    assert stats["disparities"]["positive_ratio"] == 1.0
    assert stats["gauge_assessment"].startswith("numerically_plausible_monocular_gauge")


def test_ba_wrapper_records_the_first_post_iteration_nonfinite_pose():
    frames = stream()
    video = SimpleNamespace(
        counter=SimpleNamespace(value=3),
        poses=np.tile(np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float64), (3, 1)),
    )
    call_count = 0

    def ba(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            video.poses[0, 0] = np.nan

    video.ba = ba
    state = SimpleNamespace(
        video=video,
        keyframe_source=[(item.frame_id, item.timestamp_s) for item in frames],
        dense_source=[(item.frame_id, item.timestamp_s) for item in frames],
        filler=lambda *_args: SimpleNamespace(data=np.ones((3, 7))),
    )

    def backend(steps):
        for _ in range(steps):
            video.ba()

    state.backend = backend
    recorder = probe.ProbeRecorder()
    observations = []
    restore = probe._instrument_terminal_boundaries(object(), state, frames, recorder, observations)
    try:
        state.backend(3)
    finally:
        restore()

    assert [row["iteration"] for row in observations] == [1, 2, 3]
    assert probe.first_nonfinite_ba_iteration(observations)["iteration"] == 2


def test_first_nonfinite_ba_iteration_selects_earliest_post_ba_state():
    rows = [
        {"backend_call_steps": 7, "iteration": 1, "poses": {"finite_count": 84, "value_count": 84}},
        {"backend_call_steps": 7, "iteration": 2, "poses": {"finite_count": 0, "value_count": 84}},
        {"backend_call_steps": 7, "iteration": 3, "poses": {"finite_count": 0, "value_count": 84}},
    ]
    assert probe.first_nonfinite_ba_iteration(rows) == rows[1]


def test_identical_rerun_is_deterministic_but_boundary_mismatch_is_not():
    recorder = probe.ProbeRecorder()
    record_all_finite(recorder)
    finite = probe._verdict(recorder)

    same = probe.combine_reruns(finite, finite)
    assert same["deterministic"] is True
    assert same["boundary"] == "none/finite-success"

    numerical_trace_difference = {**finite, "pre_backend": {"different_float_stat": 1.0000001}}
    assert probe.combine_reruns(finite, numerical_trace_difference)["deterministic"] is True

    changed = {**finite, "boundary": "filler_trajectory"}
    nondeterministic = probe.combine_reruns(finite, changed)
    assert nondeterministic["deterministic"] is False
    assert nondeterministic["boundary"] == "nondeterministic"
