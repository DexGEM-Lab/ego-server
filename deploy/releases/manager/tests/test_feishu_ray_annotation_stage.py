from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import warnings
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import scripts.run_feishu_ray_annotation_stage as annotation_stage
from scripts.adapt_droid_to_hawor import load_shared_geometry
from scripts.build_v19_calibration_contract import load_unidepth_intrinsics
from scripts.call_feishu_ray_service import (
    ServiceCallerError,
    build_multipart_body,
    call_service_arrays,
    decode_service_response,
)
from scripts.run_feishu_ray_annotation_stage import (
    DROID_PINNED_RELEASE,
    FeishuRayAdapterError,
    build_wilor_candidate,
    call_typed,
    decode_hands_response,
    decode_unidepth_response,
    materialize_droid_finalize,
    project_full_image,
    run_droid,
    run_unidepth,
    run_wilor,
    validate_droid_create_response,
    validate_droid_finalize,
    validate_droid_push_response,
    validated_manifest_frames,
    write_unidepth_artifact,
)


def test_run_droid_holds_shared_maintenance_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "droid.lock"
    sentinel = object()

    def fake_run_droid_unlocked(args: object, *, caller: object) -> object:
        probe_fd = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd)
        return sentinel

    monkeypatch.setenv("EGO_DROID_MAINTENANCE_LOCK", str(lock_path))
    monkeypatch.setattr(annotation_stage, "_run_droid_unlocked", fake_run_droid_unlocked)

    assert run_droid(object(), caller=object()) is sentinel
    probe_fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe_fd)


def ownership(stage: str = "unidepth.infer") -> dict[str, Any]:
    return {
        "request_id": f"request-{stage}",
        "job_id": "job-1",
        "item_id": "item-1",
        "stage_id": stage,
        "source_id": "source-1",
        "schema_version": "ego.model-service.v1",
        "source_timestamp_s": 0.25,
        "submitted_at": "2026-07-17T00:00:00.000000Z",
    }


def array_row(name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "name": name,
        "data": array.tobytes(),
        "shape": tuple(array.shape),
        "dtype": array.dtype.name,
        "descr": array.dtype.str,
        "size_bytes": int(array.nbytes),
    }


def success_report(
    owner: dict[str, Any],
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    result_key: str = "result",
) -> dict[str, Any]:
    return {
        "status": "ok",
        "http_status": 200,
        "content_type": "multipart/form-data; boundary=test",
        "metadata": {"ownership": dict(owner), result_key: result},
        "arrays": [array_row(name, value) for name, value in arrays.items()],
    }


def test_call_typed_waits_for_explicit_retryable_response_then_reuses_request() -> None:
    owner = ownership("hands.detect")
    retry_report = {
        "status": "ok",
        "http_status": 200,
        "metadata": {
            "ownership": dict(owner),
            "error": {"code": "BACKPRESSURE", "message": "retry later", "retryable": True, "retry_after_s": 0},
        },
        "arrays": [],
    }
    calls: list[dict[str, Any]] = []
    responses = [retry_report, success_report(owner, {"ok": True}, {})]
    events: list[dict[str, Any]] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return responses.pop(0)

    result = call_typed(
        caller,
        base_url="http://127.0.0.1:28001",
        route="/hands.detect",
        metadata={"ownership": owner},
        arrays={"rgb": (b"abc", (1, 1, 3), "uint8")},
        timeout_s=3.0,
        retry_events=events,
        retry_initial_delay_s=0.0,
    )
    assert result["metadata"]["result"] == {"ok": True}
    assert len(calls) == 2
    assert events[0]["event"] == "retryable_response_waiting"
    assert events[0]["error_code"] == "BACKPRESSURE"


def test_call_typed_does_not_retry_received_droid_finalize_error() -> None:
    owner = ownership("droid.finalize")
    report = {
        "status": "ok",
        "http_status": 200,
        "metadata": {
            "ownership": dict(owner),
            "error": {"code": "BACKPRESSURE", "message": "retry later", "retryable": True},
        },
        "arrays": [],
    }
    calls = 0

    def caller(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return report

    result = call_typed(
        caller,
        base_url="http://127.0.0.1:28002",
        route="/droid.finalize",
        metadata={"ownership": owner},
        arrays={},
        timeout_s=3.0,
        retry_initial_delay_s=0.0,
        allow_retryable=False,
    )
    assert result is report
    assert calls == 1


def test_in_memory_decoder_accepts_zero_hand_dimension() -> None:
    body, content_type = build_multipart_body(
        {"ownership": ownership("hands.detect"), "result": {"detection": {"n_hands": 0}}},
        [
            {
                "name": "boxes",
                "data": b"",
                "shape": (0, 4),
                "dtype": "float32",
            }
        ],
        boundary="empty-hands",
    )
    decoded = decode_service_response(200, {"content-type": content_type}, body)
    assert decoded["arrays"][0]["shape"] == (0, 4)
    assert decoded["arrays"][0]["data"] == b""


def test_stage_manifest_requires_contiguous_order_and_monotonic_time() -> None:
    with pytest.raises(FeishuRayAdapterError) as noncontiguous:
        validated_manifest_frames(
            {"fps": 30.0, "frames": [{"frame_idx": 1, "time_s": 0.0, "rgb": "frame.jpg"}]}
        )
    assert noncontiguous.value.code == "raw_frame_manifest_order_invalid"
    with pytest.raises(FeishuRayAdapterError) as decreasing:
        validated_manifest_frames(
            {
                "fps": 30.0,
                "frames": [
                    {"frame_idx": 0, "time_s": 1.0, "rgb": "frame-0.jpg"},
                    {"frame_idx": 1, "time_s": 0.5, "rgb": "frame-1.jpg"},
                ],
            }
        )
    assert decreasing.value.code == "raw_frame_manifest_order_invalid"


def test_unidepth_response_and_legacy_artifact_preserve_timeline(tmp_path: Path) -> None:
    owner = ownership()
    depth = np.full((2, 3), 1.5, dtype=np.float32)
    confidence = np.full((2, 3), 0.75, dtype=np.float32)
    K_px = np.asarray([[500.0, 0.0, 1.0], [0.0, 510.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    report = success_report(
        owner,
        {
            "ownership": dict(owner),
            "model_revision": "unidepth-v2-vitl14-corrected",
            "spatial": {"model_size": {"width": 3, "height": 2}},
            "trace": {"batch_id": "batch-1"},
        },
        {"depth_m": depth, "confidence": confidence, "K_px": K_px},
    )
    decoded_depth, decoded_conf, decoded_K, _ = decode_unidepth_response(
        report,
        ownership=owner,
        height=2,
        width=3,
    )
    output_dir = tmp_path / "unidepth_v2"
    qc = write_unidepth_artifact(
        output_dir=output_dir,
        frame_idx=np.asarray([4, 7], dtype=np.int32),
        depth=np.stack([decoded_depth, decoded_depth + 0.5]),
        confidence=np.stack([decoded_conf, decoded_conf - 0.1]),
        intrinsics=np.asarray([[500.0, 510.0, 1.0, 0.5], [500.0, 510.0, 1.0, 0.5]], dtype=np.float32),
        provenance={
            "model_revision": "unidepth-v2-vitl14-corrected",
            "service_profile": "profile-test",
            "service_base_url": "http://127.0.0.1:28000",
            "trace_batch_ids": ["batch-1", "batch-2"],
        },
    )
    archive = np.load(output_dir / "unidepth_v2_depth.npz")
    assert set(archive.files) == {"depth", "confidence", "frame_idx", "intrinsics_fx_fy_cx_cy", "source_size"}
    assert archive["depth"].shape == (2, 2, 3)
    assert archive["depth"].dtype == np.float16
    assert archive["confidence"].shape == (2, 2, 3)
    assert archive["confidence"].dtype == np.float32
    assert archive["frame_idx"].tolist() == [4, 7]
    assert archive["intrinsics_fx_fy_cx_cy"].shape == (2, 4)
    assert archive["source_size"].tolist() == [3, 2]
    consumer_idx, consumer_intrinsics, source_size = load_unidepth_intrinsics(output_dir / "unidepth_v2_depth.npz")
    assert consumer_idx.tolist() == [4, 7]
    assert consumer_intrinsics.shape == (2, 4)
    assert source_size.tolist() == [3, 2]
    assert qc["frame_count"] == 2
    assert not list(output_dir.glob("*.staging-*.npz"))
    persisted_qc = json.loads((output_dir / "qc_unidepth_v2.json").read_text(encoding="utf-8"))
    assert persisted_qc["method"] == "feishu_ray_unidepth_adapter"


def test_unidepth_stage_connects_manifest_requests_and_legacy_output(tmp_path: Path) -> None:
    from PIL import Image

    run_root = tmp_path / "run"
    frame_dir = run_root / "input" / "raw_frame_manifest"
    frame_dir.mkdir(parents=True)
    image_path = tmp_path / "frame.png"
    source_rgb = np.empty((4, 8, 3), dtype=np.uint8)
    source_rgb[:] = [17, 33, 65]
    Image.fromarray(source_rgb, mode="RGB").save(image_path)
    (frame_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "fps": 30.0,
                "frames": [
                    {"frame_idx": 0, "time_s": 0.0, "rgb": str(image_path)},
                    {"frame_idx": 1, "time_s": 1.0 / 30.0, "rgb": str(image_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.feishu_ray_services_profile.v1",
                "profile": "test-profile",
                "services": {"unidepth": {"base_url": "http://127.0.0.1:28000"}},
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    confidence_boundary_values = [float(np.finfo(np.float16).max), 70000.0]

    def fake_caller(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        assert kwargs["route"] == "/unidepth.infer"
        assert kwargs["arrays"]["rgb"][1] == (540, 960, 3)
        assert kwargs["arrays"]["rgb"][2] == "uint8"
        assert len(kwargs["arrays"]["rgb"][0]) == 540 * 960 * 3
        request_rgb = np.frombuffer(kwargs["arrays"]["rgb"][0], dtype=np.uint8).reshape(540, 960, 3)
        assert request_rgb[0, 0].tolist() == [17, 33, 65]
        spatial = kwargs["metadata"]["spatial"]
        assert spatial["source_size"] == {"width": 8, "height": 4}
        assert spatial["model_size"] == {"width": 960, "height": 540}
        assert spatial["pixel_transform"]["source_to_model"] == [
            [120.0, 0.0, 0.0],
            [0.0, 135.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert spatial["pixel_transform"]["model_to_source"] == [
            [1.0 / 120.0, 0.0, 0.0],
            [0.0, 1.0 / 135.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        owner = kwargs["metadata"]["ownership"]
        return success_report(
            owner,
            {
                "ownership": owner,
                "model_revision": "unidepth-v2-vitl14-corrected",
                "spatial": {"model_size": {"width": 960, "height": 540}},
                "trace": {"batch_id": owner["source_id"]},
            },
            {
                "depth_m": np.full((540, 960), 1.0 + len(calls), dtype=np.float32),
                "confidence": np.full(
                    (540, 960),
                    confidence_boundary_values[len(calls) - 1],
                    dtype=np.float32,
                ),
                "K_px": np.asarray(
                    [[600.0, 0.0, 480.0], [0.0, 675.0, 270.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            },
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qc = run_unidepth(
            SimpleNamespace(
                run_root=run_root,
                repo_root=tmp_path,
                profile=profile_path,
                base_url=None,
                job_id=None,
                timeout_s=5.0,
            ),
            caller=fake_caller,
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert len(calls) == 2
    assert [call["metadata"]["ownership"]["source_id"] for call in calls] == ["frame-000000", "frame-000001"]
    archive = np.load(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz")
    assert archive["frame_idx"].tolist() == [0, 1]
    assert archive["depth"].shape == (2, 4, 8)
    assert archive["confidence"].shape == (2, 4, 8)
    assert archive["confidence"].dtype == np.float32
    assert archive["source_size"].tolist() == [8, 4]
    assert np.allclose(archive["depth"][:, 0, 0], [2.0, 3.0])
    assert np.isfinite(archive["confidence"]).all()
    assert np.array_equal(archive["confidence"][:, 0, 0], np.asarray(confidence_boundary_values, dtype=np.float32))
    assert np.allclose(archive["intrinsics_fx_fy_cx_cy"], [[5.0, 5.0, 4.0, 2.0]] * 2)
    assert qc["service_profile"] == "test-profile"
    assert [row["frame_idx"] for row in qc["service_frames"]] == [0, 1]
    assert qc["service_frames"][0]["ownership"]["stage_id"] == "unidepth.infer"
    assert qc["service_frames"][0]["K_model_px"] == [[600.0, 0.0, 480.0], [0.0, 675.0, 270.0], [0.0, 0.0, 1.0]]
    assert qc["service_frames"][0]["K_source_px"] == [[5.0, 0.0, 4.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]]
    assert "exp(logconfidence)" in qc["service_frames"][0]["confidence_semantics"]
    assert "not probability" in qc["service_frames"][0]["confidence_semantics"]
    assert qc["service_frames"][0]["confidence_transform"].endswith("no value normalization or calibration")
    assert not (run_root / "scratch").exists()


def test_unidepth_accepts_float16_boundary_and_larger_finite_confidence_scores(tmp_path: Path) -> None:
    owner = ownership()
    confidence = np.asarray([[np.finfo(np.float16).max, 70000.0]], dtype=np.float32)
    report = success_report(
        owner,
        {"ownership": owner, "spatial": {"model_size": {"width": 2, "height": 1}}},
        {
            "depth_m": np.ones((1, 2), dtype=np.float32),
            "confidence": confidence,
            "K_px": np.asarray([[500.0, 0.0, 1.0], [0.0, 500.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        },
    )
    depth, decoded_confidence, _, _ = decode_unidepth_response(
        report,
        ownership=owner,
        height=1,
        width=2,
    )
    assert np.array_equal(decoded_confidence, confidence)
    output_dir = tmp_path / "unidepth"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qc = write_unidepth_artifact(
            output_dir=output_dir,
            frame_idx=np.asarray([0], dtype=np.int32),
            depth=depth[None],
            confidence=decoded_confidence[None],
            intrinsics=np.asarray([[500.0, 500.0, 1.0, 0.5]], dtype=np.float32),
            provenance={},
        )
    persisted = np.load(output_dir / "unidepth_v2_depth.npz")
    assert persisted["confidence"].dtype == np.float32
    assert np.array_equal(persisted["confidence"][0, 0], confidence[0])
    assert np.isfinite(persisted["confidence"]).all()
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert "exp(logconfidence)" in qc["confidence"]["semantics"]
    assert "lower generally indicates better" in qc["confidence"]["semantics"]
    assert "not a calibrated probability" in qc["confidence"]["semantics"]
    assert qc["confidence"]["normalization"] == "none; source-grid resampling only, with finite service values preserved as float32"
    assert qc["confidence"]["downstream_use"] == "relative score/quantile reasoning only unless separately calibrated"


def test_unidepth_writer_preserves_even_float32_max_and_writes_finite_qc(tmp_path: Path) -> None:
    float32_max = np.finfo(np.float32).max
    confidence = np.full((1, 1, 2), float32_max, dtype=np.float32)
    output_dir = tmp_path / "unidepth_float32_max"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qc = write_unidepth_artifact(
            output_dir=output_dir,
            frame_idx=np.asarray([0], dtype=np.int32),
            depth=np.ones((1, 1, 2), dtype=np.float32),
            confidence=confidence,
            intrinsics=np.asarray([[500.0, 500.0, 1.0, 0.5]], dtype=np.float32),
            provenance={},
        )
    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]
    persisted = np.load(output_dir / "unidepth_v2_depth.npz", allow_pickle=False)
    assert persisted["confidence"].dtype == np.float32
    assert np.array_equal(persisted["confidence"], confidence)
    assert all(
        np.isfinite(qc["confidence"][field])
        for field in ("min", "median_of_frame_medians", "max")
    )
    assert qc["confidence"]["median_of_frame_medians"] == float(float32_max)
    persisted_qc = json.loads((output_dir / "qc_unidepth_v2.json").read_text(encoding="utf-8"))
    assert persisted_qc["confidence"]["median_of_frame_medians"] == float(float32_max)


def test_unidepth_rejects_nonfinite_confidence_scores() -> None:
    owner = ownership()
    report = success_report(
        owner,
        {"ownership": owner, "spatial": {"model_size": {"width": 2, "height": 1}}},
        {
            "depth_m": np.ones((1, 2), dtype=np.float32),
            "confidence": np.asarray([[1.0, np.inf]], dtype=np.float32),
            "K_px": np.asarray([[500.0, 0.0, 1.0], [0.0, 500.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        },
    )
    with pytest.raises(FeishuRayAdapterError) as raised:
        decode_unidepth_response(report, ownership=owner, height=1, width=2)
    assert raised.value.code == "feishu_ray_nonfinite_array"


def test_unidepth_rejects_ownership_mismatch() -> None:
    owner = ownership()
    wrong = {**owner, "request_id": "wrong"}
    report = success_report(
        wrong,
        {"ownership": wrong},
        {
            "depth_m": np.ones((2, 3), np.float32),
            "confidence": np.ones((2, 3), np.float32),
            "K_px": np.eye(3, dtype=np.float32),
        },
    )
    with pytest.raises(FeishuRayAdapterError) as raised:
        decode_unidepth_response(report, ownership=owner, height=2, width=3)
    assert raised.value.code == "feishu_ray_ownership_mismatch"


def test_empty_hands_response_is_a_valid_empty_frame() -> None:
    owner = ownership("hands.detect")
    report = success_report(
        owner,
        {"ownership": owner, "detection": {"n_hands": 0}},
        {
            "boxes": np.empty((0, 4), np.float32),
            "scores": np.empty((0,), np.float32),
            "sides": np.empty((0,), np.uint8),
            "masks": np.empty((0, 2, 3), np.uint8),
            "visibility": np.empty((0,), np.float32),
            "uncertainty": np.empty((0,), np.float32),
        },
    )
    decoded = decode_hands_response(report, ownership=owner, height=2, width=3)
    assert decoded["n_hands"] == 0
    assert decoded["boxes"].shape == (0, 4)
    assert decoded["masks"].shape == (0, 2, 3)


def test_wilor_stage_connects_detector_crop_reconstruction_and_raw_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    import scripts.run_feishu_ray_annotation_stage as stage_module

    run_root = tmp_path / "run"
    frame_dir = run_root / "input" / "raw_frame_manifest"
    frame_dir.mkdir(parents=True)
    image_path = tmp_path / "frame.png"
    source_rgb = np.empty((4, 8, 3), dtype=np.uint8)
    source_rgb[:] = [17, 33, 65]
    Image.fromarray(source_rgb, mode="RGB").save(image_path)
    (frame_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "fps": 30.0,
                "frames": [{"frame_idx": 0, "time_s": 0.0, "rgb": str(image_path)}],
            }
        ),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.feishu_ray_services_profile.v1",
                "profile": "test-profile",
                "services": {
                    "hands_wilor": {"base_url": "http://127.0.0.1:28001"},
                    "wilor": {"base_url": "http://127.0.0.1:28004"},
                },
            }
        ),
        encoding="utf-8",
    )
    wilor_root = tmp_path / "wilor"
    wilor_root.mkdir()
    wilor_config = wilor_root / "model_config.yaml"
    wilor_config.write_text("MODEL: {}\n", encoding="utf-8")

    dataset_inputs: list[dict[str, np.ndarray]] = []

    class FakeDataset:
        def __init__(self, _config: Any, frame_bgr: np.ndarray, boxes: np.ndarray, sides: np.ndarray, **_kwargs: Any) -> None:
            self.frame_bgr = frame_bgr
            self.boxes = boxes
            self.sides = sides
            dataset_inputs.append({"frame_bgr": frame_bgr.copy(), "boxes": boxes.copy(), "sides": sides.copy()})

        def __len__(self) -> int:
            return len(self.boxes)

        def __getitem__(self, index: int) -> dict[str, Any]:
            box = self.boxes[index]
            return {
                "img": np.zeros((3, 256, 256), dtype=np.float32),
                "box_center": np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32),
                "box_size": np.asarray(max(box[2] - box[0], box[3] - box[1]), dtype=np.float32),
                "img_size": np.asarray([self.frame_bgr.shape[1], self.frame_bgr.shape[0]], dtype=np.float32),
                "right": np.asarray(self.sides[index], dtype=np.float32),
            }

    monkeypatch.setattr(stage_module, "load_wilor_preprocessor", lambda *_args: (object(), FakeDataset))
    monkeypatch.setitem(
        sys.modules,
        "scripts.run_v21_wilor_hand_candidates",
        SimpleNamespace(
            diagnose_hand_candidates=lambda frames, depth, manifest: {
                "schema": "v21_hand_candidate_diagnosis.v0",
                "status": "ok",
                "total_frames": len(frames),
            }
        ),
    )
    calls: list[tuple[str, str]] = []

    def fake_caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        calls.append((route, str(kwargs["base_url"])))
        owner = kwargs["metadata"]["ownership"]
        if route == "/hands.detect":
            assert kwargs["arrays"]["rgb"][1:] == ((540, 960, 3), "uint8")
            request_rgb = np.frombuffer(kwargs["arrays"]["rgb"][0], dtype=np.uint8).reshape(540, 960, 3)
            assert request_rgb[0, 0].tolist() == [17, 33, 65]
            spatial = kwargs["metadata"]["spatial"]
            assert spatial["source_size"] == {"width": 8, "height": 4}
            assert spatial["model_size"] == {"width": 960, "height": 540}
            assert spatial["pixel_transform"]["source_to_model"] == [
                [120.0, 0.0, 0.0],
                [0.0, 135.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            model_mask = np.zeros((1, 540, 960), dtype=np.uint8)
            model_mask[:, 135:405, 120:840] = 1
            return success_report(
                owner,
                {
                    "ownership": owner,
                    "model_revision": "hands-yolo-sam2.1-hiera-l",
                    "trace": {"batch_id": "hands-1"},
                    "detection": {"n_hands": 1},
                },
                {
                    "boxes": np.asarray([[120.0, 135.0, 840.0, 405.0]], dtype=np.float32),
                    "scores": np.asarray([0.9], dtype=np.float32),
                    "sides": np.asarray([1], dtype=np.uint8),
                    "masks": model_mask,
                    "visibility": np.asarray([0.8], dtype=np.float32),
                    "uncertainty": np.asarray([0.1], dtype=np.float32),
                },
            )
        assert route == "/wilor.reconstruct"
        detector_path = run_root / "measurements" / "hand_detections" / "feishu_ray_hands" / "hands_detector_timeline.json"
        assert detector_path.is_file()
        persisted_detector = json.loads(detector_path.read_text(encoding="utf-8"))
        assert persisted_detector["frames"][0]["observations"][0]["bbox_xyxy_source"] == [1.0, 1.0, 7.0, 3.0]
        assert kwargs["arrays"]["crop"][1] == (3, 256, 256)
        assert kwargs["metadata"]["box_center"] == [4.0, 2.0]
        assert kwargs["metadata"]["box_size"] == 6.0
        assert kwargs["metadata"]["img_size"] == [8.0, 4.0]
        vertices = np.zeros((778, 3), dtype=np.float32)
        joints = np.zeros((21, 3), dtype=np.float32)
        projected_surface = np.tile(np.asarray([[4.0, 2.0]], dtype=np.float32), (778, 1))
        return success_report(
            owner,
            {
                "ownership": owner,
                "handedness": 1,
                "model_revision": "wilor-final-v1",
                "mano": {"focal_length": 100.0, "n_vertices": 778},
                "trace": {"batch_id": "wilor-1"},
            },
            {
                "global_orient": np.eye(3, dtype=np.float32)[None],
                "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None], 15, axis=0),
                "betas": np.zeros(10, dtype=np.float32),
                "vertices": vertices,
                "joints": joints,
                "cam_t_full": np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
                "pred_cam": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                "keypoints_2d": projected_surface,
                "confidence": np.asarray([1.0], dtype=np.float32),
                "uncertainty": np.asarray([0.0], dtype=np.float32),
            },
        )

    qc = run_wilor(
        SimpleNamespace(
            run_root=run_root,
            repo_root=tmp_path,
            profile=profile_path,
            base_url=None,
            job_id=None,
            timeout_s=5.0,
            wilor_root=wilor_root,
            wilor_config=wilor_config,
            rescale_factor=2.0,
            compute_target="fixture",
        ),
        caller=fake_caller,
    )
    assert calls == [
        ("/hands.detect", "http://127.0.0.1:28001"),
        ("/wilor.reconstruct", "http://127.0.0.1:28004"),
    ]
    assert len(dataset_inputs) == 1
    assert dataset_inputs[0]["frame_bgr"].shape == (4, 8, 3)
    assert dataset_inputs[0]["frame_bgr"][0, 0].tolist() == [65, 33, 17]
    assert np.allclose(dataset_inputs[0]["boxes"], [[1.0, 1.0, 7.0, 3.0]])
    raw = json.loads(
        (run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json").read_text(encoding="utf-8")
    )
    assert raw["frames"][0]["frame_idx"] == 0
    assert raw["frames"][0]["hands_detection"]["ownership"]["stage_id"] == "hands.detect"
    assert raw["frames"][0]["hands_detection"]["n_hands"] == 1
    hand = raw["frames"][0]["raw_hands"][0]
    assert hand["side"] == "right"
    assert hand["coordinate_space"] == "source_pixels"
    assert hand["bbox_xyxy"] == [1.0, 1.0, 7.0, 3.0]
    assert hand["crop_metadata"]["box_center"] == [4.0, 2.0]
    assert hand["crop_metadata"]["img_size"] == [8.0, 4.0]
    assert np.allclose(hand["joints2d"], [4.0, 2.0])
    assert np.allclose(hand["projected_surface_2d"], [4.0, 2.0])
    assert len(hand["vertices_camera"]) == 778
    detector = json.loads(Path(qc["hands_detector_path"]).read_text(encoding="utf-8"))
    assert detector["frame_count"] == 1
    assert detector["coordinate_semantics"]["artifact_boxes"] == "source_pixels_xyxy"
    observation = detector["frames"][0]["observations"][0]
    assert observation["bbox_xyxy_model"] == [120.0, 135.0, 840.0, 405.0]
    assert observation["bbox_xyxy_source"] == [1.0, 1.0, 7.0, 3.0]
    assert observation["mask_coordinate_space"] == "source_grid"
    side_states = {row["side"]: row for row in detector["frames"][0]["side_states"]}
    assert side_states["right"]["visibility_state"] == "visible"
    assert side_states["right"]["ownership"]["stage_id"] == "hands.detect"
    assert side_states["left"]["visibility_state"] == "unresolved"
    mask_archive = np.load(qc["hands_detector_masks_path"])
    assert mask_archive["source_size"].tolist() == [8, 4]
    assert mask_archive["model_size"].tolist() == [960, 540]
    source_mask = np.unpackbits(
        mask_archive["masks_packbits"][0],
        bitorder="little",
        count=4 * 8,
    ).reshape(4, 8)
    expected_mask = np.zeros((4, 8), dtype=np.uint8)
    expected_mask[1:3, 1:7] = 1
    assert np.array_equal(source_mask, expected_mask)
    assert qc["service_calls"] == {"hands.detect": 1, "wilor.reconstruct": 1}


def test_wilor_candidate_uses_real_joints_and_root_relative_mesh() -> None:
    owner = ownership("wilor.reconstruct")
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.05, 0.05, len(vertices), dtype=np.float32)
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[:, 0] = np.linspace(-0.02, 0.02, len(joints), dtype=np.float32)
    joints[:, 1] = np.linspace(-0.01, 0.01, len(joints), dtype=np.float32)
    cam_t = np.asarray([0.1, -0.05, 2.0], dtype=np.float32)
    projected_surface = project_full_image(
        vertices,
        cam_t,
        1000.0,
        np.asarray([320.0, 240.0], dtype=np.float32),
    )
    report = success_report(
        owner,
        {
            "ownership": owner,
            "handedness": 1,
            "model_revision": "wilor-final-v1",
            "mano": {"focal_length": 1000.0, "n_vertices": 778},
            "trace": {"batch_id": "wilor-batch"},
        },
        {
            "global_orient": np.eye(3, dtype=np.float32)[None],
            "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None], 15, axis=0),
            "betas": np.zeros(10, dtype=np.float32),
            "vertices": vertices,
            "joints": joints,
            "cam_t_full": cam_t,
            "pred_cam": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "keypoints_2d": projected_surface,
            "confidence": np.asarray([0.9], dtype=np.float32),
            "uncertainty": np.asarray([0.01], dtype=np.float32),
        },
    )
    candidate = build_wilor_candidate(
        report,
        ownership=owner,
        detector_score=0.8,
        bbox_xyxy=np.asarray([10.0, 20.0, 110.0, 220.0], dtype=np.float32),
        detector_visibility=0.7,
        detector_uncertainty=0.2,
        expected_side=1,
        img_size=np.asarray([320.0, 240.0], dtype=np.float32),
    )
    assert candidate["side"] == "right"
    assert np.allclose(np.asarray(candidate["vertices_camera"]), vertices)
    assert np.allclose(np.asarray(candidate["cam_t"]), cam_t)
    assert candidate["pred_cam"] == [1.0, 0.0, 0.0]
    assert candidate["source_image_size"] == {"width": 320, "height": 240}
    assert np.asarray(candidate["joints2d"]).shape == (21, 2)
    assert candidate["joints2d_coordinate_space"] == "source_pixels"
    assert np.allclose(candidate["projected_surface_2d"], projected_surface)
    assert candidate["projected_surface_2d_coordinate_space"] == "source_pixels"
    assert candidate["mano_params"]["representation"] == "rotation_matrices_pose2rot_false"
    assert np.asarray(candidate["mano_params"]["global_orient"]).shape == (1, 3, 3)
    assert np.asarray(candidate["mano_params"]["hand_pose"]).shape == (15, 3, 3)
    assert np.asarray(candidate["mano_params"]["betas"]).shape == (10,)
    assert candidate["projected_surface_2d_shape"] == [778, 2]
    assert candidate["filter_status"] == "measured_raw"

    mismatched_report = {**report, "arrays": [dict(row) for row in report["arrays"]]}
    keypoints_row = next(row for row in mismatched_report["arrays"] if row["name"] == "keypoints_2d")
    keypoints_row["data"] = np.zeros((778, 2), dtype=np.float32).tobytes()
    with pytest.raises(FeishuRayAdapterError) as raised:
        build_wilor_candidate(
            mismatched_report,
            ownership=owner,
            detector_score=0.8,
            bbox_xyxy=np.asarray([10.0, 20.0, 110.0, 220.0], dtype=np.float32),
            detector_visibility=0.7,
            detector_uncertainty=0.2,
            expected_side=1,
            img_size=np.asarray([320.0, 240.0], dtype=np.float32),
        )
    assert raised.value.code == "feishu_ray_wilor_projection_mismatch"


def droid_report(
    owner: dict[str, Any],
    *,
    T_world_camera: np.ndarray,
    T_camera_world: np.ndarray,
    dense_mapping: list[dict[str, Any]],
    finite_pose_ratio: float,
) -> dict[str, Any]:
    keyframes = [{"keyframe_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0}]
    state = {
        "ownership": owner,
        "session_id": "session-1",
        "model_revision": "droid-v1",
        "dense_mapping": dense_mapping,
        "keyframe_mapping": keyframes,
        "uncertainty": {
            "finite_pose_ratio": finite_pose_ratio,
            "valid_keyframe_ratio": 1.0,
            "scale_status": "up_to_scale",
            "reprojection_error": None,
        },
    }
    return success_report(
        owner,
        state,
        {
            "T_world_camera": T_world_camera,
            "T_camera_world": T_camera_world,
            "disparities": np.ones((1, 2, 3), dtype=np.float32),
            "intrinsics_px": np.asarray([[[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]]], dtype=np.float64),
        },
        result_key="camera_state",
    )


def expected_droid_timeline() -> list[dict[str, Any]]:
    return [
        {"frame_idx": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0},
        {"frame_idx": 1, "source_frame_id": "frame-1", "source_timestamp_s": 1.0 / 30.0},
    ]


def droid_json_report(owner: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "http_status": 200,
        "content_type": "application/json",
        "metadata": {"ownership": dict(owner), "error": None, **payload},
        "arrays": [],
    }


def make_droid_stage_fixture(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
    from PIL import Image

    run_root = tmp_path / "run"
    frame_dir = run_root / "input" / "raw_frame_manifest" / "rgb"
    frame_dir.mkdir(parents=True)
    source_width, source_height = 192, 108
    frames: list[dict[str, Any]] = []
    for frame_idx in range(2):
        x = np.arange(source_width, dtype=np.uint8)[None]
        y = np.arange(source_height, dtype=np.uint8)[:, None]
        rgb = np.empty((source_height, source_width, 3), dtype=np.uint8)
        rgb[:, :, 0] = (x + 20 + frame_idx) % 255
        rgb[:, :, 1] = (y + 60 + frame_idx) % 255
        rgb[:, :, 2] = ((x // 2 + y // 2) + 100 + frame_idx) % 255
        rgb_path = frame_dir / f"{frame_idx:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        frames.append(
            {
                "frame_idx": frame_idx,
                "time_s": frame_idx / 30.0,
                "rgb": str(rgb_path),
                "source_width": source_width,
                "source_height": source_height,
                "manifest_width": source_width,
                "manifest_height": source_height,
            }
        )
    clip_path = run_root / "input" / "clips" / "case-1.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"fixture clip identity")
    (run_root / "input" / "raw_frame_manifest" / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "clip": str(clip_path),
                "fps": 30.0,
                "frame_count": 2,
                "video": {"width": source_width, "height": source_height, "fps": 30.0, "frame_count": 2},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "input" / "input_manifest.json").write_text(
        json.dumps({"case_id": "case-1", "primary_video": str(clip_path)}),
        encoding="utf-8",
    )
    calibration_path = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "method": "fixture_calibration",
                "intrinsics_source": "fixture_source_K",
                "source_size": {"width": source_width, "height": source_height},
                "intrinsics_fx_fy_cx_cy": [96.0, 108.0, 96.0, 54.0],
            }
        ),
        encoding="utf-8",
    )

    masks: list[np.ndarray] = []
    mask_0_left = np.zeros((source_height, source_width), dtype=np.uint8)
    mask_0_left[10:38, 18:62] = 1
    mask_0_right = np.zeros_like(mask_0_left)
    mask_0_right[26:60, 48:92] = 1
    mask_1_right = np.zeros_like(mask_0_left)
    mask_1_right[54:92, 105:160] = 1
    masks.extend([mask_0_left, mask_0_right, mask_1_right])
    packed_masks = np.stack([np.packbits(mask.reshape(-1), bitorder="little") for mask in masks])
    detector_dir = run_root / "measurements" / "hand_detections" / "feishu_ray_hands"
    detector_dir.mkdir(parents=True)
    mask_archive = detector_dir / "hands_source_masks.npz"
    np.savez_compressed(
        mask_archive,
        masks_packbits=packed_masks,
        frame_idx=np.asarray([0, 0, 1], dtype=np.int32),
        detection_idx=np.asarray([0, 1, 0], dtype=np.int16),
        side=np.asarray([0, 1, 1], dtype=np.uint8),
        source_size=np.asarray([source_width, source_height], dtype=np.int32),
        model_size=np.asarray([960, 540], dtype=np.int32),
        mask_bit_count=np.asarray(source_width * source_height, dtype=np.int64),
        packbits_bitorder=np.asarray("little"),
    )
    mask_archive_hash = hashlib.sha256(mask_archive.read_bytes()).hexdigest()
    detector_frames: list[dict[str, Any]] = []
    archive_cursor = 0
    for frame_idx, frame_masks in ((0, masks[:2]), (1, masks[2:])):
        observations = []
        for detection_index, mask in enumerate(frame_masks):
            side_index = 0 if frame_idx == 0 and detection_index == 0 else 1
            observations.append(
                {
                    "detection_index": detection_index,
                    "side_index": side_index,
                    "mask_archive_index": archive_cursor,
                    "mask_source_pixel_count": int(np.count_nonzero(mask)),
                }
            )
            archive_cursor += 1
        detector_frames.append(
            {
                "frame_idx": frame_idx,
                "time_s": frame_idx / 30.0,
                "observations": observations,
            }
        )
    detector_path = detector_dir / "hands_detector_timeline.json"
    detector_path.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.hands_detector_timeline.v1",
                "status": "ok",
                "frame_count": 2,
                "mask_archive": {
                    "path": str(mask_archive),
                    "sha256": mask_archive_hash,
                    "array_key": "masks_packbits",
                },
                "frames": detector_frames,
            }
        ),
        encoding="utf-8",
    )
    return run_root, masks


def make_droid_stage_caller(
    *,
    invalid_finalize: bool = False,
    session_id: str = "session-1",
    finalize_session_id: Any | None = None,
    finalize_metadata_mutation: Callable[[dict[str, Any]], None] | None = None,
    invalid_push_position: int | None = None,
    finalize_transport_failure: bool = False,
    cleanup_transport_failure: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    source_K = np.asarray([[96.0, 0.0, 96.0], [0.0, 108.0, 54.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    source_to_model = np.asarray(
        [[591.0 / 192.0, 0.0, 0.0], [0.0, 332.0 / 108.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    model_K = source_to_model @ source_K

    def caller(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        route = str(kwargs["route"])
        metadata = kwargs["metadata"]
        owner = metadata["ownership"]
        if route == "/droid.create_session":
            assert len(calls) == 1
            assert kwargs["arrays"] == {}
            assert metadata["image_shape"] == {"height": 328, "width": 584}
            assert metadata["camera"]["source_size"] == {"width": 192, "height": 108}
            assert np.allclose(metadata["camera"]["K_px"], source_K)
            assert np.allclose(metadata["camera"]["intrinsics"], [model_K[0, 0], model_K[1, 1], model_K[0, 2], model_K[1, 2]])
            assert np.allclose(metadata["camera"]["pixel_transform"]["source_to_model"], source_to_model)
            assert metadata["options"]["filter_thresh"] == 8.0
            assert metadata["options"]["keyframe_thresh"] == 4.0
            assert metadata["options"]["buffer"] == 9
            return droid_json_report(owner, {"session_id": session_id})
        if route == "/droid.push_frame":
            push_index = sum(call["route"] == "/droid.push_frame" for call in calls) - 1
            assert metadata["session_id"] == session_id
            assert metadata["frame_id"] == f"frame-{push_index}"
            assert metadata["source_timestamp_s"] == pytest.approx(push_index / 30.0)
            rgb_spec = kwargs["arrays"]["rgb"]
            mask_spec = kwargs["arrays"]["static_confidence_mask"]
            assert rgb_spec[1:] == ((328, 584, 3), "uint8")
            assert mask_spec[1:] == ((328, 584), "float32")
            rgb = np.frombuffer(rgb_spec[0], dtype=np.uint8).reshape(rgb_spec[1])
            mask = np.frombuffer(mask_spec[0], dtype=np.float32).reshape(mask_spec[1])
            assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
            assert float(mask.max()) > 0.0
            assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
            assert np.all(rgb[mask > 0.0] == 0)
            status = {
                "ownership": dict(owner),
                "session_id": session_id,
                "frame_id": f"frame-{push_index}",
                "source_timestamp_s": push_index / 30.0,
                "validity": {
                    "frame_id": f"frame-{push_index}",
                    "source_timestamp_s": push_index / 30.0,
                    "admitted": True,
                    "keyframe_added": True,
                    "skip_reason": None,
                },
                "keyframe_count": push_index + 1,
                "trace": {"batch_id": f"push-{push_index}", "model_load_count": 1},
            }
            if invalid_push_position == push_index:
                status["frame_id"] = "mutated-frame"
            return droid_json_report(owner, {"status": status})
        assert route == "/droid.finalize"
        assert kwargs["arrays"] == {}
        assert metadata["session_id"] == session_id
        if str(owner["item_id"]).endswith("-cleanup"):
            if cleanup_transport_failure:
                raise ServiceCallerError(
                    "service_transport_failed",
                    "cleanup finalize transport failed",
                    response_received=False,
                )
            cleanup_state = {
                "ownership": dict(owner),
                "session_id": session_id,
                "model_revision": "droid-v1",
                "uncertainty": {"finite_pose_ratio": 0.0},
            }
            nonfinite = np.full((1, 4, 4), np.nan, dtype=np.float64)
            return success_report(
                owner,
                cleanup_state,
                {
                    "T_world_camera": nonfinite,
                    "T_camera_world": nonfinite,
                    "disparities": np.full((1, 1, 1), np.nan, dtype=np.float32),
                    "intrinsics_px": np.full((1, 3, 3), np.nan, dtype=np.float64),
                },
                result_key="camera_state",
            )
        if finalize_transport_failure:
            raise ServiceCallerError(
                "service_transport_failed",
                "normal finalize transport failed",
                response_received=False,
            )
        assert len(calls) == 4
        T_world_camera = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        T_world_camera[1, 0, 3] = 0.1
        T_camera_world = np.linalg.inv(T_world_camera)
        finite_pose_ratio = 1.0
        if invalid_finalize:
            T_world_camera[:] = np.nan
            T_camera_world[:] = np.nan
            finite_pose_ratio = 0.0
        state = {
            "ownership": dict(owner),
            "session_id": session_id if finalize_session_id is None else finalize_session_id,
            "model_revision": "droid-v1",
            "dense_mapping": [
                {"dense_index": index, "source_frame_id": f"frame-{index}", "source_timestamp_s": index / 30.0}
                for index in range(2)
            ],
            "keyframe_mapping": [
                {"keyframe_index": index, "source_frame_id": f"frame-{index}", "source_timestamp_s": index / 30.0}
                for index in range(2)
            ],
            "uncertainty": {
                "finite_pose_ratio": finite_pose_ratio,
                "valid_keyframe_ratio": 1.0,
                "scale_status": "up_to_scale",
                "reprojection_error": None,
            },
        }
        if finalize_metadata_mutation is not None:
            finalize_metadata_mutation(state)
        return success_report(
            owner,
            state,
            {
                "T_world_camera": T_world_camera,
                "T_camera_world": T_camera_world,
                "disparities": np.ones((2, 41, 73), dtype=np.float32),
                "intrinsics_px": np.repeat(model_K[None], 2, axis=0),
            },
            result_key="camera_state",
        )

    return caller, calls


def droid_stage_args(run_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=run_root,
        repo_root=Path(__file__).resolve().parents[1],
        profile=Path(__file__).resolve().parents[1] / "configs" / "feishu_ray_services.json",
        base_url="http://127.0.0.1:28002",
        job_id=None,
        timeout_s=5.0,
    )


def load_droid_cleanup_lifecycle(run_root: Path) -> dict[str, Any]:
    records = list(
        (run_root / "failures" / "feishu_ray_droid_session_cleanup").glob("*/cleanup_lifecycle.json")
    )
    assert len(records) == 1
    return json.loads(records[0].read_text(encoding="utf-8"))


def load_droid_create_failure(run_root: Path) -> dict[str, Any]:
    records = list(
        (run_root / "failures" / "feishu_ray_droid_create").glob("*/failed_create.json")
    )
    assert len(records) == 1
    return json.loads(records[0].read_text(encoding="utf-8"))


def test_droid_route_specific_envelopes_reject_identity_and_keyframe_mismatches() -> None:
    create_owner = ownership("droid.create_session")
    assert validate_droid_create_response(
        droid_json_report(create_owner, {"session_id": "session-1"}),
        ownership=create_owner,
    ) == "session-1"
    wrong_create_owner = {**create_owner, "request_id": "wrong"}
    with pytest.raises(FeishuRayAdapterError) as create_raised:
        validate_droid_create_response(
            droid_json_report(wrong_create_owner, {"session_id": "session-1"}),
            ownership=create_owner,
        )
    assert create_raised.value.code == "feishu_ray_ownership_mismatch"
    coerced_timestamp_owner = {**create_owner, "source_timestamp_s": "0.25"}
    with pytest.raises(FeishuRayAdapterError) as timestamp_raised:
        validate_droid_create_response(
            droid_json_report(coerced_timestamp_owner, {"session_id": "session-1"}),
            ownership=create_owner,
        )
    assert timestamp_raised.value.code == "feishu_ray_ownership_mismatch"

    push_owner = ownership("droid.push_frame")
    valid_status = {
        "ownership": dict(push_owner),
        "session_id": "session-1",
        "frame_id": "frame-0",
        "source_timestamp_s": 0.25,
        "validity": {
            "frame_id": "frame-0",
            "source_timestamp_s": 0.25,
            "admitted": True,
            "keyframe_added": True,
            "skip_reason": None,
        },
        "keyframe_count": 1,
    }
    report = droid_json_report(push_owner, {"status": valid_status})
    assert validate_droid_push_response(
        report,
        ownership=push_owner,
        session_id="session-1",
        frame_id="frame-0",
        source_timestamp_s=0.25,
        expected_keyframe_count=1,
    )["keyframe_count"] == 1
    bad_cases = [
        ("session_id", "wrong", "feishu_ray_droid_push_identity_mismatch"),
        ("frame_id", "wrong", "feishu_ray_droid_push_identity_mismatch"),
        ("source_timestamp_s", 0.5, "feishu_ray_droid_push_identity_mismatch"),
        ("source_timestamp_s", True, "feishu_ray_droid_push_identity_mismatch"),
        ("source_timestamp_s", "0.25", "feishu_ray_droid_push_identity_mismatch"),
        ("keyframe_count", 0, "feishu_ray_droid_keyframe_count_regressed"),
        ("keyframe_count", True, "feishu_ray_droid_all_keyframe_compensation_failed"),
        ("keyframe_count", "1", "feishu_ray_droid_all_keyframe_compensation_failed"),
        ("keyframe_count", 1.5, "feishu_ray_droid_all_keyframe_compensation_failed"),
    ]
    for field, value, code in bad_cases:
        bad = json.loads(json.dumps(valid_status))
        bad[field] = value
        with pytest.raises(FeishuRayAdapterError) as raised:
            validate_droid_push_response(
                droid_json_report(push_owner, {"status": bad}),
                ownership=push_owner,
                session_id="session-1",
                frame_id="frame-0",
                source_timestamp_s=0.25,
                expected_keyframe_count=1,
            )
        assert raised.value.code == code
    valid_non_keyframe = json.loads(json.dumps(valid_status))
    valid_non_keyframe["validity"]["keyframe_added"] = False
    valid_non_keyframe["validity"]["skip_reason"] = "insufficient_motion_or_first_frame"
    assert validate_droid_push_response(
        droid_json_report(push_owner, {"status": valid_non_keyframe}),
        ownership=push_owner,
        session_id="session-1",
        frame_id="frame-0",
        source_timestamp_s=0.25,
        expected_keyframe_count=1,
    )["keyframe_count"] == 1


def test_droid_response_identity_tokens_require_exact_strings() -> None:
    create_owner = ownership("droid.create_session")
    with pytest.raises(FeishuRayAdapterError) as create_error:
        validate_droid_create_response(
            droid_json_report(create_owner, {"session_id": 1}),
            ownership=create_owner,
        )
    assert create_error.value.code == "feishu_ray_droid_session_missing"

    push_owner = ownership("droid.push_frame")
    valid_status = {
        "ownership": dict(push_owner),
        "session_id": "1",
        "frame_id": "1",
        "source_timestamp_s": 0.25,
        "validity": {
            "frame_id": "1",
            "source_timestamp_s": 0.25,
            "admitted": True,
            "keyframe_added": True,
            "skip_reason": None,
        },
        "keyframe_count": 1,
    }
    for path in (("session_id",), ("frame_id",), ("validity", "frame_id")):
        malformed = json.loads(json.dumps(valid_status))
        target: Any = malformed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = 1
        with pytest.raises(FeishuRayAdapterError) as push_error:
            validate_droid_push_response(
                droid_json_report(push_owner, {"status": malformed}),
                ownership=push_owner,
                session_id="1",
                frame_id="1",
                source_timestamp_s=0.25,
                expected_keyframe_count=1,
            )
        assert push_error.value.code == "feishu_ray_droid_push_identity_mismatch"


def test_droid_post_create_local_frame_load_failure_attempts_one_cleanup(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    frame_path = run_root / "input" / "raw_frame_manifest" / "rgb" / "000000.png"
    expected_message = f"raw manifest RGB file not found: {frame_path}"

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.create_session":
            frame_path.unlink()
        return report

    with pytest.raises(FileNotFoundError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert str(raised.value) == expected_message
    assert routes == ["/droid.create_session", "/droid.finalize"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    assert cleanup["trigger"]["error"] == {
        "type": "FileNotFoundError",
        "code": "unexpected_session_failure",
        "message": expected_message,
    }


def test_droid_arbitrary_push_caller_failure_attempts_cleanup_and_preserves_error(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    push_failed = False

    def caller(**kwargs: Any) -> dict[str, Any]:
        nonlocal push_failed
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.push_frame" and not push_failed:
            push_failed = True
            raise RuntimeError("arbitrary caller failure")
        return base_caller(**kwargs)

    with pytest.raises(RuntimeError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert str(raised.value) == "arbitrary caller failure"
    assert routes == ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    assert cleanup["trigger"]["error"] == {
        "type": "RuntimeError",
        "code": "unexpected_session_failure",
        "message": "arbitrary caller failure",
    }


def test_droid_push_validation_failure_attempts_cleanup_and_retains_original_error(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(invalid_push_position=0)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_push_identity_mismatch"
    assert "mutated-frame" in str(raised.value)
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    failed_push_owner = calls[1]["metadata"]["ownership"]
    cleanup_owner = calls[2]["metadata"]["ownership"]
    assert cleanup_owner["stage_id"] == "droid.finalize"
    assert cleanup_owner["item_id"] == "case-1-cleanup"
    assert cleanup_owner["request_id"] != failed_push_owner["request_id"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["attempted"] is True
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    assert cleanup["successful_service_result_confirmed_retirement"] is True
    assert cleanup["trigger"]["error"]["code"] == "feishu_ray_droid_push_identity_mismatch"
    assert cleanup["cleanup"]["response"]["successful_service_result"] is True
    assert cleanup["cleanup"]["response"]["geometry_validated_or_published"] is False
    assert cleanup["successful_d4_artifacts_published"] is False
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_cleanup_failure_is_unresolved_without_masking_push_error(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(invalid_push_position=0, cleanup_transport_failure=True)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_push_identity_mismatch"
    assert "mutated-frame" in str(raised.value)
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["attempted"] is True
    assert cleanup["retirement_status"] == "unresolved"
    assert cleanup["successful_service_result_confirmed_retirement"] is False
    assert cleanup["cleanup"]["response"] is None
    assert cleanup["cleanup"]["error"]["code"] == "service_transport_failed"
    assert "may remain resident" in cleanup["service_contract_limitation"]
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


@pytest.mark.parametrize(
    ("mismatch", "expected_code"),
    [
        ("nested_ownership", "feishu_ray_ownership_mismatch"),
        ("session_id", "feishu_ray_droid_session_mismatch"),
        ("model_revision", "feishu_ray_droid_model_revision_mismatch"),
    ],
)
def test_droid_cleanup_identity_mismatch_is_unresolved_and_preserves_original_error(
    tmp_path: Path,
    mismatch: str,
    expected_code: str,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller(invalid_push_position=0)
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        owner = kwargs["metadata"]["ownership"]
        if kwargs["route"] == "/droid.finalize" and str(owner["item_id"]).endswith("-cleanup"):
            state = report["metadata"]["camera_state"]
            if mismatch == "nested_ownership":
                state["ownership"] = {**state["ownership"], "request_id": "wrong-request"}
            elif mismatch == "session_id":
                state["session_id"] = "wrong-session"
            else:
                state["model_revision"] = "wrong-revision"
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_push_identity_mismatch"
    assert str(raised.value) == "droid.push_frame returned session/frame session-1/mutated-frame"
    assert routes == ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["retirement_status"] == "unresolved"
    assert cleanup["successful_service_result_confirmed_retirement"] is False
    assert cleanup["cleanup"]["response"] is None
    assert cleanup["cleanup"]["error"]["code"] == expected_code


def test_droid_cleanup_missing_nested_ownership_is_unresolved_and_preserves_original_error(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller(invalid_push_position=0)
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        owner = kwargs["metadata"]["ownership"]
        if kwargs["route"] == "/droid.finalize" and str(owner["item_id"]).endswith("-cleanup"):
            report["metadata"]["camera_state"].pop("ownership")
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_push_identity_mismatch"
    assert str(raised.value) == "droid.push_frame returned session/frame session-1/mutated-frame"
    assert routes == ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["retirement_status"] == "unresolved"
    assert cleanup["successful_service_result_confirmed_retirement"] is False
    assert cleanup["trigger"]["error"] == {
        "type": "FeishuRayAdapterError",
        "code": "feishu_ray_droid_push_identity_mismatch",
        "message": "droid.push_frame returned session/frame session-1/mutated-frame",
    }
    assert cleanup["cleanup"]["response"] is None
    assert cleanup["cleanup"]["error"] == {
        "code": "feishu_ray_ownership_mismatch",
        "message": "/droid.finalize: cleanup nested result ownership is missing or does not match request",
    }
    assert cleanup["successful_d4_artifacts_published"] is False
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_create_validation_failure_with_trusted_handle_cleans_once_and_preserves_response(
    tmp_path: Path,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.create_session":
            report["metadata"]["status"] = {"unexpected": True}
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_envelope_invalid"
    assert str(raised.value) == "droid.create_session returned a non-create envelope"
    assert raised.value.response_received is True
    assert routes == ["/droid.create_session", "/droid.finalize"]
    failure = load_droid_create_failure(run_root)
    assert failure["status"] == "failed_create_validation"
    assert failure["validation_error"]["code"] == "feishu_ray_response_envelope_invalid"
    assert failure["session_handle_status"] == "trusted_exact_string_matching_success_json_envelope"
    assert failure["trusted_session_id"] == "session-1"
    assert failure["response_received"] is True
    assert failure["cleanup_attempt_required"] is True
    assert failure["cleanup_attempted"] is True
    assert failure["retirement_status"] == "confirmed_by_successful_finalize_result"
    response_metadata_path = Path(failure["response_metadata"]["path"])
    response_metadata = json.loads(response_metadata_path.read_text(encoding="utf-8"))
    assert response_metadata["session_id"] == "session-1"
    assert response_metadata["status"] == {"unexpected": True}
    assert failure["response_metadata"]["sha256"] == hashlib.sha256(
        response_metadata_path.read_bytes()
    ).hexdigest()
    cleanup_path = Path(failure["cleanup_lifecycle"]["path"])
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    assert cleanup["trigger"]["route"] == "/droid.create_session"
    assert cleanup["trigger"]["error"]["code"] == "feishu_ray_response_envelope_invalid"
    assert cleanup["session_id"] == "session-1"
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_create_trusted_handle_cleanup_failure_keeps_primary_and_marks_unresolved(
    tmp_path: Path,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller(cleanup_transport_failure=True)
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.create_session":
            report["metadata"]["camera_state"] = {"unexpected": True}
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_envelope_invalid"
    assert str(raised.value) == "droid.create_session returned a non-create envelope"
    assert routes == ["/droid.create_session", "/droid.finalize"]
    failure = load_droid_create_failure(run_root)
    assert failure["trusted_session_id"] == "session-1"
    assert failure["cleanup_attempt_required"] is True
    assert failure["cleanup_attempted"] is True
    assert failure["retirement_status"] == "unresolved"
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["trigger"]["error"]["code"] == "feishu_ray_response_envelope_invalid"
    assert cleanup["retirement_status"] == "unresolved"
    assert cleanup["cleanup"]["error"] == {
        "code": "service_transport_failed",
        "message": "/droid.finalize: cleanup finalize transport failed",
    }
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


@pytest.mark.parametrize(
    ("session_value", "expected_code", "expected_type"),
    [
        (None, "feishu_ray_droid_session_missing", "NoneType"),
        (1, "feishu_ray_droid_session_missing", "int"),
    ],
)
def test_droid_create_missing_or_numeric_handle_preserves_unresolved_evidence_without_cleanup(
    tmp_path: Path,
    session_value: Any,
    expected_code: str,
    expected_type: str,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.create_session":
            if session_value is None:
                report["metadata"].pop("session_id")
            else:
                report["metadata"]["session_id"] = session_value
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == expected_code
    assert raised.value.response_received is True
    assert routes == ["/droid.create_session"]
    failure = load_droid_create_failure(run_root)
    assert failure["status"] == "failed_create_validation"
    assert failure["response_received"] is True
    assert failure["trusted_session_id"] is None
    assert failure["observed_session_id_type"] == expected_type
    assert failure["session_handle_status"] == "unresolved_no_trusted_session_handle"
    assert failure["retirement_status"] == "unresolved_no_trusted_session_handle"
    assert failure["cleanup_attempt_required"] is False
    assert failure["cleanup_attempted"] is False
    assert failure["cleanup_not_attempted_reason"]
    response_metadata = json.loads(
        Path(failure["response_metadata"]["path"]).read_text(encoding="utf-8")
    )
    if session_value is None:
        assert "session_id" not in response_metadata
    else:
        assert response_metadata["session_id"] == session_value
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_create_cross_owned_handle_is_not_finalized_and_preserves_response(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.create_session":
            report["metadata"]["ownership"]["request_id"] = "cross-owned-request"
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_ownership_mismatch"
    assert raised.value.response_received is True
    assert routes == ["/droid.create_session"]
    failure = load_droid_create_failure(run_root)
    assert failure["observed_session_id_repr"] == repr("session-1")
    assert failure["trusted_session_id"] is None
    assert failure["retirement_status"] == "unresolved_no_trusted_session_handle"
    assert failure["cleanup_attempted"] is False
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()


def test_droid_create_pre_response_failure_preserves_transport_evidence_without_cleanup(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        raise ServiceCallerError(
            "service_transport_failed",
            "create route unreachable",
            response_received=False,
        )

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "service_transport_failed"
    assert raised.value.response_received is False
    assert routes == ["/droid.create_session"]
    failure = load_droid_create_failure(run_root)
    assert failure["status"] == "failed_create_call"
    assert failure["response_received"] is False
    assert failure["response_metadata"] is None
    assert failure["typed_arrays"] == []
    assert failure["retirement_status"] == "unresolved_no_trusted_session_handle"
    assert failure["cleanup_attempted"] is False
    assert set(failure["unavailable_response_evidence"]) == {
        "decoded_response_metadata",
        "typed_response_arrays",
        "http_status",
        "response_headers",
        "raw_response_bytes",
    }
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()


def test_droid_create_nonmapping_return_is_stable_and_preserved_without_cleanup(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        return None  # type: ignore[return-value]

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_envelope_invalid"
    assert str(raised.value) == "/droid.create_session: caller returned a non-mapping response report"
    assert raised.value.response_received is True
    assert routes == ["/droid.create_session"]
    failure = load_droid_create_failure(run_root)
    assert failure["status"] == "failed_create_call"
    assert failure["response_received"] is True
    assert failure["returned_report"] == {"python_type": "NoneType", "repr": "None"}
    assert failure["retirement_status"] == "unresolved_no_trusted_session_handle"
    assert failure["cleanup_attempted"] is False
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_finalize_transport_failure_attempts_one_cleanup_finalize(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(finalize_transport_failure=True)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "service_transport_failed"
    assert raised.value.response_received is False
    assert raised.value.response_status is None
    assert raised.value.response_headers is None
    assert raised.value.raw_response_bytes is None
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
        "/droid.finalize",
    ]
    assert calls[3]["metadata"]["ownership"]["stage_id"] == "droid.finalize"
    assert calls[4]["metadata"]["ownership"]["stage_id"] == "droid.finalize"
    assert calls[4]["metadata"]["ownership"]["item_id"] == "case-1-cleanup"
    assert calls[3]["metadata"]["ownership"]["request_id"] != calls[4]["metadata"]["ownership"]["request_id"]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["trigger"]["route"] == "/droid.finalize"
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    assert cleanup["successful_service_result_confirmed_retirement"] is True


def test_droid_finalize_pre_opener_failure_attempts_one_cleanup_and_preserves_primary_error(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    opener_called = False

    def unopened(_request: Any, timeout: float) -> Any:
        nonlocal opener_called
        opener_called = True
        raise AssertionError(f"opener must not be called with timeout={timeout}")

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        owner = kwargs["metadata"]["ownership"]
        if route == "/droid.finalize" and not str(owner["item_id"]).endswith("-cleanup"):
            return call_service_arrays(
                **{
                    **kwargs,
                    "metadata": {**kwargs["metadata"], "invalid_preflight": float("nan")},
                },
                opener=unopened,
            )
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "invalid_metadata_json"
    assert "metadata is not finite JSON data" in str(raised.value)
    assert raised.value.response_received is False
    assert opener_called is False
    assert routes == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
        "/droid.finalize",
    ]
    cleanup = load_droid_cleanup_lifecycle(run_root)
    assert cleanup["trigger"]["route"] == "/droid.finalize"
    assert cleanup["trigger"]["error"]["code"] == "invalid_metadata_json"
    assert cleanup["retirement_status"] == "confirmed_by_successful_finalize_result"
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "invalid_metadata_json"
    assert failure["response_received"] is False
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_finalize_rejects_malformed_receipt_provenance_without_retry(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            raise ServiceCallerError(
                "malformed_custom_receipt",
                "custom caller used integer receipt provenance",
                response_received=0,
                response_status=299,
                response_headers={"Content-Type": "text/plain"},
                raw_response_bytes=b"malformed receipt",
            )
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_receipt_invalid"
    assert str(raised.value) == (
        "/droid.finalize: caller returned invalid response_received provenance 0"
    )
    assert raised.value.response_received is None
    assert routes.count("/droid.finalize") == 1
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_response_receipt_invalid"
    assert failure["response_received"] is None
    assert failure["http_status"] == 299
    assert Path(failure["raw_response_bytes"]["path"]).read_bytes() == b"malformed receipt"


def test_droid_received_finalize_decode_failure_is_not_retried_and_preserves_raw_evidence(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    response_headers = {"Content-Type": "text/plain", "X-Service-Request": "finalize-17"}
    response_body = b"opaque finalize failure"

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            raise ServiceCallerError(
                "unsupported_response_content_type",
                "text/plain",
                response_received=True,
                response_status=502,
                response_headers=response_headers,
                raw_response_bytes=response_body,
            )
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "unsupported_response_content_type"
    assert str(raised.value) == "/droid.finalize: text/plain"
    assert raised.value.response_received is True
    assert raised.value.response_status == 502
    assert raised.value.response_headers == response_headers
    assert raised.value.raw_response_bytes == response_body
    assert routes == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    cleanup_root = run_root / "failures" / "feishu_ray_droid_session_cleanup"
    assert not cleanup_root.exists()
    failure_manifests = list((run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json"))
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["status"] == "failed_finalize_call"
    assert failure["response_received"] is True
    assert failure["http_status"] == 502
    assert failure["response_metadata"] is None
    assert failure["typed_arrays"] == []
    assert failure["unavailable_response_evidence"] == [
        "decoded_response_metadata",
        "typed_response_arrays",
    ]
    assert failure["validation_error"]["code"] == "unsupported_response_content_type"
    headers_artifact = failure["response_headers"]
    headers_path = Path(headers_artifact["path"])
    assert json.loads(headers_path.read_text(encoding="utf-8")) == response_headers
    assert headers_artifact["size_bytes"] == headers_path.stat().st_size
    assert headers_artifact["sha256"] == hashlib.sha256(headers_path.read_bytes()).hexdigest()
    raw_artifact = failure["raw_response_bytes"]
    raw_path = Path(raw_artifact["path"])
    assert raw_path.read_bytes() == response_body
    assert raw_artifact["size_bytes"] == len(response_body)
    assert raw_artifact["sha256"] == hashlib.sha256(response_body).hexdigest()


def test_droid_finalize_body_read_failure_is_received_and_not_retried(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    response_headers = {"Content-Type": "application/json", "X-Service-Request": "finalize-18"}

    class ReadFailureResponse:
        status = 503
        headers = response_headers

        def __enter__(self) -> "ReadFailureResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            raise OSError("response body read failed")

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            return call_service_arrays(
                **kwargs,
                opener=lambda _request, timeout: ReadFailureResponse(),
            )
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "service_transport_failed"
    assert str(raised.value) == (
        "/droid.finalize: http://127.0.0.1:28002/droid.finalize: response body read failed"
    )
    assert raised.value.response_received is True
    assert raised.value.response_status == 503
    assert raised.value.response_headers == response_headers
    assert raised.value.raw_response_bytes is None
    assert routes == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["response_received"] is True
    assert failure["http_status"] == 503
    assert failure["raw_response_bytes"] is None
    assert "raw_response_bytes" in failure["unavailable_response_evidence"]
    assert "response_headers" not in failure["unavailable_response_evidence"]
    assert json.loads(Path(failure["response_headers"]["path"]).read_text(encoding="utf-8")) == response_headers


def test_droid_finalize_incomplete_response_preserves_partial_body_without_retry(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    partial = b"partial droid finalize response"
    response_headers = {"Content-Type": "application/json", "X-Service-Request": "finalize-incomplete"}

    class IncompleteResponse:
        status = 206
        headers = response_headers

        def __enter__(self) -> "IncompleteResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            raise IncompleteRead(partial, 31)

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            return call_service_arrays(
                **kwargs,
                opener=lambda _request, timeout: IncompleteResponse(),
            )
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "service_response_incomplete"
    assert str(raised.value) == (
        "/droid.finalize: http://127.0.0.1:28002/droid.finalize: "
        f"IncompleteRead({len(partial)} bytes read, 31 more expected)"
    )
    assert raised.value.response_received is True
    assert raised.value.response_status == 206
    assert raised.value.response_headers == response_headers
    assert raised.value.raw_response_bytes == partial
    assert isinstance(raised.value.__cause__, ServiceCallerError)
    assert isinstance(raised.value.__cause__.__cause__, IncompleteRead)
    assert routes == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "service_response_incomplete"
    assert failure["response_received"] is True
    assert failure["http_status"] == 206
    assert failure["response_metadata"] is None
    assert failure["typed_arrays"] == []
    assert failure["unavailable_response_evidence"] == [
        "decoded_response_metadata",
        "typed_response_arrays",
    ]
    headers_artifact = failure["response_headers"]
    headers_path = Path(headers_artifact["path"])
    assert json.loads(headers_path.read_text(encoding="utf-8")) == response_headers
    assert headers_artifact["size_bytes"] == headers_path.stat().st_size
    assert headers_artifact["sha256"] == hashlib.sha256(headers_path.read_bytes()).hexdigest()
    raw_artifact = failure["raw_response_bytes"]
    raw_path = Path(raw_artifact["path"])
    assert raw_path.read_bytes() == partial
    assert raw_artifact["size_bytes"] == len(partial)
    assert raw_artifact["sha256"] == hashlib.sha256(partial).hexdigest()
    assert failure["successful_d4_artifacts_published"] is False


def test_droid_finalize_unknown_receipt_is_not_retried(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            raise ServiceCallerError("custom_finalize_failure", "receipt unknown")
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "custom_finalize_failure"
    assert str(raised.value) == "/droid.finalize: receipt unknown"
    assert raised.value.response_received is None
    assert routes.count("/droid.finalize") == 1
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["response_received"] is None
    assert failure["http_status"] is None
    assert failure["response_headers"] is None
    assert failure["raw_response_bytes"] is None
    assert set(failure["unavailable_response_evidence"]) == {
        "decoded_response_metadata",
        "typed_response_arrays",
        "http_status",
        "response_headers",
        "raw_response_bytes",
    }


def test_droid_returned_nonmapping_finalize_report_is_stable_and_not_retried(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route == "/droid.finalize":
            return None  # type: ignore[return-value]
        return base_caller(**kwargs)

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_envelope_invalid"
    assert str(raised.value) == "/droid.finalize: caller returned a non-mapping response report"
    assert routes.count("/droid.finalize") == 1
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_response_envelope_invalid"
    assert failure["response_received"] is True
    assert failure["response_metadata"] is None
    assert failure["typed_arrays"] == []
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()


def test_droid_normal_finalize_rejects_nonnull_nondict_error_without_retry(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []

    def caller(**kwargs: Any) -> dict[str, Any]:
        routes.append(str(kwargs["route"]))
        report = base_caller(**kwargs)
        if kwargs["route"] == "/droid.finalize":
            report["metadata"]["error"] = "malformed-service-error"
        return report

    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_response_envelope_invalid"
    assert str(raised.value) == "/droid.finalize: invalid error field 'malformed-service-error'"
    assert routes.count("/droid.finalize") == 1
    cleanup_root = run_root / "failures" / "feishu_ray_droid_session_cleanup"
    assert not cleanup_root.exists()
    failure_manifests = list((run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json"))
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_response_envelope_invalid"
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }


@pytest.mark.parametrize("add_note_mode", ["missing", "raises"])
def test_droid_received_model_failure_survives_broken_add_note_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_note_mode: str,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    base_caller, _ = make_droid_stage_caller()
    routes: list[str] = []
    service_message = (
        "CUDA out of memory. Tried to allocate 750.00 MiB "
        "(GPU 0; 79.33 GiB total capacity; 469.75 MiB free)"
    )

    def caller(**kwargs: Any) -> dict[str, Any]:
        route = str(kwargs["route"])
        routes.append(route)
        if route != "/droid.finalize":
            return base_caller(**kwargs)
        owner = kwargs["metadata"]["ownership"]
        return {
            "status": "ok",
            "http_status": 200,
            "content_type": "application/json",
            "metadata": {
                "ownership": dict(owner),
                "error": {
                    "code": "model_failure",
                    "message": service_message,
                    "retryable": False,
                    "ownership": dict(owner),
                    "batch_id": "oom-batch",
                },
            },
            "arrays": [],
        }

    if add_note_mode == "missing":
        monkeypatch.setattr(FeishuRayAdapterError, "add_note", None)
    else:
        def fail_add_note(_error: BaseException, _note: str) -> None:
            raise KeyboardInterrupt("diagnostic hook failed")

        monkeypatch.setattr(FeishuRayAdapterError, "add_note", fail_add_note)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)

    assert raised.value.code == "feishu_ray_model_failure"
    assert str(raised.value) == f"/droid.finalize: {service_message}"
    assert routes == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert any("DROID failed finalize evidence:" in note for note in raised.value.__notes__)
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()

    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed_finalize_validation"
    assert failure["route"] == "/droid.finalize"
    assert failure["http_status"] == 200
    assert failure["content_type"] == "application/json"
    assert failure["validation_error"] == {
        "type": "FeishuRayAdapterError",
        "code": "feishu_ray_model_failure",
        "message": f"/droid.finalize: {service_message}",
    }
    assert failure["typed_arrays"] == []
    assert failure["successful_d4_artifacts_published"] is False
    response_metadata = json.loads(
        Path(failure["response_metadata"]["path"]).read_text(encoding="utf-8")
    )
    assert response_metadata["error"] == {
        "code": "model_failure",
        "message": service_message,
        "retryable": False,
        "ownership": response_metadata["ownership"],
        "batch_id": "oom-batch",
    }


def test_droid_stage_prepares_compatibility_inputs_and_materializes_valid_geometry(tmp_path: Path) -> None:
    run_root, source_masks = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller()
    report = run_droid(droid_stage_args(run_root), caller=caller)

    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert report["status"] == "ok"
    assert report["processed_frames"] == 2
    assert report["pinned_service_release"] == DROID_PINNED_RELEASE
    input_dir = run_root / "measurements" / "camera_trajectory" / "droid_service_inputs"
    submitted_masks = np.load(input_dir / "droid_submitted_dynamic_ignore_masks.npy")
    assert submitted_masks.shape == (2, 328, 584)
    assert submitted_masks.dtype == np.float32
    assert float(submitted_masks.max()) > 0.0
    input_provenance = json.loads((input_dir / "droid_submitted_dynamic_ignore_masks.json").read_text(encoding="utf-8"))
    assert input_provenance["submitted_timeline"]["array_is_exact_submitted_tensor_timeline"] is True
    assert input_provenance["submitted_timeline"]["sha256"] == hashlib.sha256(
        (input_dir / "droid_submitted_dynamic_ignore_masks.npy").read_bytes()
    ).hexdigest()
    expected_union_0 = source_masks[0] | source_masks[1]
    assert input_provenance["frames"][0]["source_union_sha256"] == hashlib.sha256(
        np.ascontiguousarray(expected_union_0).tobytes()
    ).hexdigest()
    assert input_provenance["frames"][0]["detector_mask_archive_indices"] == [0, 1]
    assert input_provenance["frames"][1]["detector_mask_archive_indices"] == [2]

    output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    trajectory = np.load(output_dir / "droid_dense_trajectory.npz")
    assert trajectory["frame_idx"].tolist() == [0, 1]
    assert np.allclose(trajectory["intrinsics_source"], [96.0, 108.0, 96.0, 54.0])
    reconstruction = np.load(output_dir / "droid_keyframe_reconstruction.npz")
    assert reconstruction["tstamps"].tolist() == [0, 1]
    assert reconstruction["disps"].shape == (2, 41, 73)
    shared = json.loads((output_dir / "droid_shared_geometry.json").read_text(encoding="utf-8"))
    assert shared["droid_invocation"] == {
        "class": "feishu_ray.droid_session",
        "instance_count": 1,
        "track_call_count": 2,
        "terminate_call_count": 1,
        "session_id": "session-1",
    }
    assert shared["dynamic_mask"]["submitted_value_semantics"] == "positive=ignore,0=retain"
    assert shared["camera_provenance"]["model_size"] == {"width": 584, "height": 328}
    assert np.allclose(
        shared["camera_provenance"]["K_model_px"],
        [[295.5, 0.0, 295.5], [0.0, 332.0, 166.0], [0.0, 0.0, 1.0]],
    )
    assert shared["service_provenance"]["session_options"]["filter_thresh"] == 8.0
    assert shared["service_provenance"]["session_options"]["keyframe_thresh"] == 4.0
    assert shared["service_provenance"]["session_options"]["buffer"] == 9
    assert shared["service_provenance"]["compatibility_compensations"]["buffer_right_sizing"].startswith(
        "allocate only max(frame_count + 1, warmup + 1)"
    )
    assert shared["service_provenance"]["compatibility_compensations"]["pinned_service_release"] == DROID_PINNED_RELEASE
    stage = json.loads((output_dir / "v22_camera_trajectory_stage.json").read_text(encoding="utf-8"))
    assert stage["schema"] == "v22_camera_trajectory_stage.v0"
    assert stage["status"] == "ok"
    assert stage["camera_backend"] == "droid"
    assert stage["execution_backend"] == "feishu_ray"
    assert stage["outputs"]["dense_json"] == str(output_dir / "droid_dense_trajectory.json")
    assert stage["outputs"]["shared_geometry_manifest"] == str(output_dir / "droid_shared_geometry.json")
    assert stage["gauge_declaration"]["trajectory_frame"] == "DROID arbitrary world gauge"
    consumed = load_shared_geometry(output_dir / "droid_shared_geometry.json", expected_frames=2)
    assert consumed["frame_idx"].tolist() == [0, 1]
    assert consumed["tstamp"].tolist() == [0, 1]


def test_droid_atomic_publication_cleans_late_manifest_failure_and_allows_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_feishu_ray_annotation_stage as stage_module

    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller()
    original_write_json = stage_module.write_json

    def fail_final_manifest(path: Path, payload: Any) -> None:
        if Path(path).name == "droid_shared_geometry.json":
            raise RuntimeError("late final manifest write failed")
        original_write_json(path, payload)

    monkeypatch.setattr(stage_module, "write_json", fail_final_manifest)
    with pytest.raises(RuntimeError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert str(raised.value) == "late final manifest write failed"
    assert [call["route"] for call in calls].count("/droid.finalize") == 1
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    output_parent = run_root / "measurements" / "camera_trajectory"
    output_dir = output_parent / "droid_full_frame"
    assert not output_dir.exists()
    assert list(output_parent.glob(".droid_full_frame.staging-*")) == []
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"] == {
        "type": "RuntimeError",
        "code": "unexpected_session_failure",
        "message": "late final manifest write failed",
    }
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }

    monkeypatch.setattr(stage_module, "write_json", original_write_json)
    rerun_caller, rerun_calls = make_droid_stage_caller()
    rerun = run_droid(droid_stage_args(run_root), caller=rerun_caller)
    assert rerun["status"] == "ok"
    assert [call["route"] for call in rerun_calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert output_dir.is_dir()
    assert Path(rerun["shared_geometry_manifest"]) == output_dir / "droid_shared_geometry.json"
    assert rerun["shared_geometry_manifest_sha256"] == hashlib.sha256(
        Path(rerun["shared_geometry_manifest"]).read_bytes()
    ).hexdigest()
    assert list(output_parent.glob(".droid_full_frame.staging-*")) == []
    published_manifest = json.loads(Path(rerun["shared_geometry_manifest"]).read_text(encoding="utf-8"))
    for artifact_name in (
        "dense_trajectory",
        "dense_trajectory_json",
        "keyframe_reconstruction",
        "keyframes",
        "droid_qc",
    ):
        assert Path(published_manifest["artifacts"][artifact_name]["path"]).parent == output_dir


def test_droid_stage_rejects_finalize_session_mismatch_and_preserves_response(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(finalize_session_id="wrong-session")
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_session_mismatch"
    assert [call["route"] for call in calls][-1] == "/droid.finalize"
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()
    failure_manifests = list((run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json"))
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["session_id"] == "session-1"
    assert failure["validation_error"]["code"] == "feishu_ray_droid_session_mismatch"


def test_droid_stage_rejects_numeric_finalize_session_for_digit_string_session(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(session_id="1", finalize_session_id=1)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_session_mismatch"
    assert str(raised.value) == "droid.finalize returned a non-string session_id"
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_droid_session_mismatch"
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }
    response_metadata = json.loads(Path(failure["response_metadata"]["path"]).read_text(encoding="utf-8"))
    assert response_metadata["camera_state"]["session_id"] == 1


@pytest.mark.parametrize(
    ("mapping_name", "expected_code"),
    [
        ("dense_mapping", "feishu_ray_droid_incomplete_timeline"),
        ("keyframe_mapping", "feishu_ray_droid_keyframe_mapping_mismatch"),
    ],
)
def test_droid_stage_rejects_numeric_finalize_source_ids_and_preserves_evidence(
    tmp_path: Path,
    mapping_name: str,
    expected_code: str,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)

    def mutate_source_id(state: dict[str, Any]) -> None:
        state[mapping_name][0]["source_frame_id"] = 0

    caller, calls = make_droid_stage_caller(finalize_metadata_mutation=mutate_source_id)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == expected_code
    assert [call["route"] for call in calls].count("/droid.finalize") == 1
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()
    failure_path = next(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == expected_code
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }
    response_metadata = json.loads(Path(failure["response_metadata"]["path"]).read_text(encoding="utf-8"))
    assert response_metadata["camera_state"][mapping_name][0]["source_frame_id"] == 0


def test_droid_stage_missing_finalize_nested_ownership_preserves_response_without_publishing_d4(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)

    def remove_nested_ownership(state: dict[str, Any]) -> None:
        state.pop("ownership")

    caller, calls = make_droid_stage_caller(finalize_metadata_mutation=remove_nested_ownership)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_ownership_mismatch"
    assert str(raised.value) == (
        "/droid.finalize: nested result ownership is required and must match request ownership"
    )
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    assert not (run_root / "failures" / "feishu_ray_droid_session_cleanup").exists()
    output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    assert not output_dir.exists()
    failure_manifests = list(
        (run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json")
    )
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["validation_error"] == {
        "type": "FeishuRayAdapterError",
        "code": "feishu_ray_ownership_mismatch",
        "message": "/droid.finalize: nested result ownership is required and must match request ownership",
    }
    response_metadata_path = Path(failure["response_metadata"]["path"])
    response_metadata = json.loads(response_metadata_path.read_text(encoding="utf-8"))
    assert "ownership" not in response_metadata["camera_state"]
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }
    for artifact in failure["typed_arrays"]:
        assert Path(artifact["path"]).is_file()
        assert isinstance(np.load(artifact["path"], allow_pickle=False), np.ndarray)
    assert failure["successful_d4_artifacts_published"] is False


def test_droid_stage_preserves_invalid_finalize_without_publishing_d4(tmp_path: Path) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)
    caller, calls = make_droid_stage_caller(invalid_finalize=True)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_nonfinite_geometry"
    assert [call["route"] for call in calls] == [
        "/droid.create_session",
        "/droid.push_frame",
        "/droid.push_frame",
        "/droid.finalize",
    ]
    output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    assert not output_dir.exists()
    failure_manifests = list((run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json"))
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_droid_nonfinite_geometry"
    assert failure["successful_d4_artifacts_published"] is False
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }
    failed_world = next(row for row in failure["typed_arrays"] if row["name"] == "T_world_camera")
    preserved = np.load(failed_world["path"], allow_pickle=False)
    assert np.isnan(preserved).all()
    assert failed_world["finite"] is False


@pytest.mark.parametrize(
    ("metadata_path", "malformed_value"),
    [
        (("uncertainty", "finite_pose_ratio"), None),
        (("uncertainty", "finite_pose_ratio"), True),
        (("uncertainty", "finite_pose_ratio"), 10**1000),
        (("dense_mapping", 0, "dense_index"), "not-an-index"),
        (("dense_mapping", 0, "dense_index"), "0"),
        (("dense_mapping", 0, "dense_index"), 0.5),
        (("dense_mapping", 0, "source_timestamp_s"), []),
        (("dense_mapping", 0, "source_timestamp_s"), "0.0"),
        (("dense_mapping", 0, "source_timestamp_s"), -(10**1000)),
        (("keyframe_mapping", 0, "keyframe_index"), np.inf),
        (("keyframe_mapping", 0, "keyframe_index"), True),
        (("keyframe_mapping", 0, "source_timestamp_s"), {}),
        (("keyframe_mapping", 0, "source_timestamp_s"), "0.0"),
    ],
    ids=[
        "finite-pose-ratio-type-error",
        "finite-pose-ratio-bool",
        "finite-pose-ratio-huge-integer-overflow",
        "dense-index-value-error",
        "dense-index-string",
        "dense-index-fractional-float",
        "dense-timestamp-type-error",
        "dense-timestamp-string",
        "dense-timestamp-huge-negative-integer-overflow",
        "keyframe-index-overflow-error",
        "keyframe-index-bool",
        "keyframe-timestamp-type-error",
        "keyframe-timestamp-string",
    ],
)
def test_droid_stage_wraps_malformed_finalize_metadata_and_preserves_evidence(
    tmp_path: Path,
    metadata_path: tuple[Any, ...],
    malformed_value: Any,
) -> None:
    run_root, _ = make_droid_stage_fixture(tmp_path)

    def mutate(state: dict[str, Any]) -> None:
        target: Any = state
        for key in metadata_path[:-1]:
            target = target[key]
        target[metadata_path[-1]] = malformed_value

    caller, calls = make_droid_stage_caller(finalize_metadata_mutation=mutate)
    with pytest.raises(FeishuRayAdapterError) as raised:
        run_droid(droid_stage_args(run_root), caller=caller)
    assert raised.value.code == "feishu_ray_droid_finalize_metadata_invalid"
    assert [call["route"] for call in calls].count("/droid.finalize") == 1
    assert not (run_root / "measurements" / "camera_trajectory" / "droid_full_frame").exists()
    failure_manifests = list((run_root / "failures" / "feishu_ray_droid_finalize").glob("*/failed_finalize.json"))
    assert len(failure_manifests) == 1
    failure = json.loads(failure_manifests[0].read_text(encoding="utf-8"))
    assert failure["validation_error"]["code"] == "feishu_ray_droid_finalize_metadata_invalid"
    response_metadata_path = Path(failure["response_metadata"]["path"])
    assert response_metadata_path.is_file()
    assert json.loads(response_metadata_path.read_text(encoding="utf-8"))["camera_state"]
    assert {row["name"] for row in failure["typed_arrays"]} == {
        "T_world_camera",
        "T_camera_world",
        "disparities",
        "intrinsics_px",
    }
    for artifact in failure["typed_arrays"]:
        assert Path(artifact["path"]).is_file()
        assert isinstance(np.load(artifact["path"], allow_pickle=False), np.ndarray)


def test_droid_nonfinite_fixture_cannot_create_artifacts(tmp_path: Path) -> None:
    owner = ownership("droid.finalize")
    bad = np.full((2, 4, 4), np.nan, dtype=np.float64)
    report = droid_report(
        owner,
        T_world_camera=bad,
        T_camera_world=bad,
        dense_mapping=[
            {"dense_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0},
            {"dense_index": 1, "source_frame_id": "frame-1", "source_timestamp_s": 1.0 / 30.0},
        ],
        finite_pose_ratio=0.0,
    )
    output_dir = tmp_path / "camera"
    with pytest.raises(FeishuRayAdapterError) as raised:
        materialize_droid_finalize(
            report,
            ownership=owner,
            expected_timeline=expected_droid_timeline(),
            output_dir=output_dir,
            fps=30.0,
            clip=tmp_path / "clip.mp4",
            clip_sha256=None,
            dynamic_mask={"status": "applied"},
        )
    assert raised.value.code == "feishu_ray_droid_nonfinite_geometry"
    assert not output_dir.exists()


def test_droid_keyframe_only_mapping_is_rejected() -> None:
    owner = ownership("droid.finalize")
    identity = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    report = droid_report(
        owner,
        T_world_camera=identity,
        T_camera_world=identity,
        dense_mapping=[{"dense_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0}],
        finite_pose_ratio=1.0,
    )
    with pytest.raises(FeishuRayAdapterError) as raised:
        validate_droid_finalize(report, ownership=owner, expected_timeline=expected_droid_timeline())
    assert raised.value.code == "feishu_ray_droid_incomplete_timeline"


def test_droid_rejects_non_rigid_inverse_consistent_transforms() -> None:
    owner = ownership("droid.finalize")
    T_world_camera = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    T_world_camera[:, 0, 1] = 0.2
    T_camera_world = np.linalg.inv(T_world_camera)
    report = droid_report(
        owner,
        T_world_camera=T_world_camera,
        T_camera_world=T_camera_world,
        dense_mapping=[
            {"dense_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0},
            {"dense_index": 1, "source_frame_id": "frame-1", "source_timestamp_s": 1.0 / 30.0},
        ],
        finite_pose_ratio=1.0,
    )
    with pytest.raises(FeishuRayAdapterError) as raised:
        validate_droid_finalize(report, ownership=owner, expected_timeline=expected_droid_timeline())
    assert raised.value.code == "feishu_ray_droid_invalid_se3"


def test_droid_rejects_ambiguous_mask_semantics_before_writing(tmp_path: Path) -> None:
    owner = ownership("droid.finalize")
    identity = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    report = droid_report(
        owner,
        T_world_camera=identity,
        T_camera_world=identity,
        dense_mapping=[
            {"dense_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0},
            {"dense_index": 1, "source_frame_id": "frame-1", "source_timestamp_s": 1.0 / 30.0},
        ],
        finite_pose_ratio=1.0,
    )
    output_dir = tmp_path / "camera"
    with pytest.raises(FeishuRayAdapterError) as raised:
        materialize_droid_finalize(
            report,
            ownership=owner,
            expected_timeline=expected_droid_timeline(),
            output_dir=output_dir,
            fps=30.0,
            clip=tmp_path / "clip.mp4",
            clip_sha256=None,
            dynamic_mask={"status": "applied", "path": "/tmp/mask.npy", "sha256": "unknown"},
        )
    assert raised.value.code == "feishu_ray_droid_mask_semantics_unclear"
    assert not output_dir.exists()


def test_valid_droid_fixture_materializes_legacy_geometry(tmp_path: Path) -> None:
    owner = ownership("droid.finalize")
    T_world_camera = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    T_world_camera[1, 0, 3] = 0.1
    T_camera_world = np.linalg.inv(T_world_camera)
    report = droid_report(
        owner,
        T_world_camera=T_world_camera,
        T_camera_world=T_camera_world,
        dense_mapping=[
            {"dense_index": 0, "source_frame_id": "frame-0", "source_timestamp_s": 0.0},
            {"dense_index": 1, "source_frame_id": "frame-1", "source_timestamp_s": 1.0 / 30.0},
        ],
        finite_pose_ratio=1.0,
    )
    mask_path = tmp_path / "dynamic_mask.npy"
    np.save(mask_path, np.zeros((2, 2, 3), dtype=np.uint8))
    mask_sha256 = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "camera"
    manifest = materialize_droid_finalize(
        report,
        ownership=owner,
        expected_timeline=expected_droid_timeline(),
        output_dir=output_dir,
        fps=30.0,
        clip=tmp_path / "clip.mp4",
        clip_sha256="abc",
        dynamic_mask={
            "status": "applied_from_hawor_preparation",
            "path": str(mask_path),
            "sha256": mask_sha256,
            "source_value_semantics": "1=dynamic_ignore,0=static_keep",
            "submitted_value_semantics": "1=ignore,0=retain",
            "service_consumption_semantics": "1=ignore,0=retain",
            "source_to_submitted_conversion": "identity",
        },
    )
    trajectory = np.load(output_dir / "droid_dense_trajectory.npz")
    assert trajectory["frame_idx"].tolist() == [0, 1]
    assert trajectory["pose_world_camera_xyzw"].shape == (2, 7)
    assert np.allclose(trajectory["T_world_camera"], T_world_camera)
    reconstruction = np.load(output_dir / "droid_keyframe_reconstruction.npz")
    assert reconstruction["tstamps"].tolist() == [0]
    assert reconstruction["disps"].shape == (1, 2, 3)
    assert manifest["full_source_timeline"] is True
    assert manifest["droid_invocation"]["track_call_count"] == 2
    consumed = load_shared_geometry(output_dir / "droid_shared_geometry.json", expected_frames=2)
    assert consumed["frame_idx"].tolist() == [0, 1]
    assert consumed["tstamp"].tolist() == [0]
    assert consumed["disps"].shape == (1, 2, 3)
    assert consumed["dynamic_mask_path"] == mask_path.resolve()
    persisted = json.loads((output_dir / "droid_shared_geometry.json").read_text(encoding="utf-8"))
    assert persisted["execution_backend"] == "feishu_ray"
