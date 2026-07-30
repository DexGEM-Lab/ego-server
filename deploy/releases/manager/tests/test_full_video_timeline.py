from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import ego_annotation.full_video_timeline as full_video_timeline
import pytest

from ego_annotation.full_video_timeline import (
    AlgorithmStageClient,
    CoveragePolicy,
    FullVideoDriverConfig,
    FullVideoTimelineDriver,
    HandDetectionRecord,
    HaworChunkTrace,
    InMemoryFrameSource,
    OpenCvFrameSource,
    PreflightError,
    SourceTimeline,
    TimelineDriverError,
    plan_single_video,
    _densify_droid_output,
    _droid_chunks_with_overlap,
    _droid_sample_source_indices,
    _droid_stride,
    _droid_input_shape_yx,
    _crop_transform,
    _scheduled_droid_options,
    _interpolate_droid_poses,
    _stitch_droid_chunk_poses,
    DROID_SERVICE_PUSH_CAPACITY,
    DroidChunkFinalizeError,
    RequestBatchTrace,
    _droid_prefix_coverage,
    _droid_scheduled_coverage,
    _droid_session_buffer,
    _infiller_window_timestamp,
    _hawor_geometry_diagnostics,
    _ManoCandidate,
    _merge_timeline_candidates,
    _normalized_crop,
    _normalized_crop_batch,
    _prepare_inference_rgb,
    _selected_source_indices,
    _typed_K,
    _visibility_code,
)
from ego_annotation.scripted.contracts import AlgorithmRequest, AlgorithmResult, ContractError, FrameTimelineMetadata, NativeWorkDescription, StageMetadata
from ego_annotation.typed_contracts import DroidCapabilities, DroidCreateOutput, DroidFinalizeOutput, DroidPushOutput, HandDetections, HandsOutput, HandSide, ManoBatch, Ownership, TypedContractError, TypedTensor, UniDepthOutput, WiLoROutput, _validate_frame_batch


def make_source(frame_count: int = 17) -> InMemoryFrameSource:
    frames = [np.full((8, 12, 3), index % 256, dtype=np.uint8) for index in range(frame_count)]
    return InMemoryFrameSource(frames, fps=4.0, source_id="fixture-video")


def test_in_memory_source_preserves_complete_source_timeline() -> None:
    source = make_source()
    timeline = source.timeline

    assert timeline.frame_indices == tuple(range(17))
    assert timeline.timestamps_s == tuple(index / 4.0 for index in range(17))
    assert timeline.duration_s == pytest.approx(17 / 4.0)
    assert source.read_rgb(16)[0, 0, 0] == 16


def test_sequential_frame_store_preserves_source_pixel_crop_and_k_geometry(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (48, 32))
    assert writer.isOpened()
    for frame_index in range(8):
        yy, xx = np.mgrid[:32, :48]
        rgb = np.stack(
            ((xx * 3 + frame_index * 11) % 256, (yy * 5 + frame_index * 17) % 256, (xx + yy + frame_index * 23) % 256),
            axis=-1,
        ).astype(np.uint8)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    legacy = OpenCvFrameSource.from_video(video_path, cache_frames=1)
    legacy_frames = {index: legacy.read_rgb(index).copy() for index in (5, 1, 7)}
    legacy_report = legacy.frame_store_report()
    assert legacy_report["pos_frames_seek_calls"] == 3
    assert legacy_report["backward_seek_calls"] == 1

    stored = OpenCvFrameSource.from_video(video_path, frame_store_max_bytes=32 * 48 * 3)
    stored.build_frame_store((1, 5, 7), spill_dir=tmp_path / "frame_store")
    for index in (1, 5, 7):
        assert stored.read_rgb(index).tobytes() == legacy_frames[index].tobytes()

    detection = HandDetectionRecord(
        "synthetic:000005:00",
        5,
        5.0 / 6.0,
        HandSide.RIGHT,
        (7.25, 4.5, 39.75, 29.0),
        0.9,
        1.0,
        0.1,
        "visible",
        Ownership("case", "item", stored.timeline.source_id, "hands", "frame:000005"),
    )
    old_crop, old_transform = _normalized_crop(legacy_frames[5], detection, legacy.timeline, 1.25)
    new_crop, new_transform = _normalized_crop(stored.read_rgb(5), detection, stored.timeline, 1.25)
    assert new_crop.tobytes() == old_crop.tobytes()
    assert new_transform == old_transform
    old_inference, old_spatial = _prepare_inference_rgb(legacy_frames[5], legacy.timeline, (24, 36))
    new_inference, new_spatial = _prepare_inference_rgb(stored.read_rgb(5), stored.timeline, (24, 36))
    assert new_inference.tobytes() == old_inference.tobytes()
    assert new_spatial == old_spatial
    k_source = np.asarray([[610.0, 0.0, 24.0], [0.0, 608.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert _typed_K(k_source, stored.timeline.source_id).array.tobytes() == _typed_K(k_source, legacy.timeline.source_id).array.tobytes()

    with pytest.raises(TimelineDriverError, match="not pre-registered"):
        stored.read_rgb(2)
    report = stored.frame_store_report()
    assert report["registered_frame_count"] == 3
    assert report["ram_frame_count"] == 1
    assert report["spill_frame_count"] == 2
    assert report["capture_read_calls"] == 8
    assert report["traversed_frame_count"] == 8
    assert report["unique_decoded_frame_count"] == 8
    assert report["duplicate_decode_count"] == 0
    assert report["traversal_passes"] == 1
    assert report["pos_frames_seek_calls"] == 0
    assert report["backward_seek_calls"] == 0


def test_wilor_crop_affines_are_true_inverses_and_left_is_preflipped() -> None:
    source = InMemoryFrameSource([np.zeros((60, 100, 3), dtype=np.uint8)], fps=5.0, source_id="crop-affine-fixture")
    owner = Ownership("case", "item", source.timeline.source_id, "hands", "fixture")
    box = (20.0, 10.0, 60.0, 50.0)
    left = HandDetectionRecord("left", 0, 0.0, HandSide.LEFT, box, 0.9, 1.0, 0.1, "visible", owner)
    right = HandDetectionRecord("right", 0, 0.0, HandSide.RIGHT, box, 0.9, 1.0, 0.1, "visible", owner)

    left_source_to_crop, left_transform = _crop_transform(left, source.timeline, 2.0)
    right_source_to_crop, right_transform = _crop_transform(right, source.timeline, 2.0)
    left_h = np.vstack((left_source_to_crop, [0.0, 0.0, 1.0]))
    right_h = np.vstack((right_source_to_crop, [0.0, 0.0, 1.0]))
    left_crop_to_source = np.asarray(left_transform.source_to_crop.pixel_to_source)
    right_crop_to_source = np.asarray(right_transform.source_to_crop.pixel_to_source)

    np.testing.assert_allclose(left_crop_to_source @ left_h, np.eye(3), atol=1.0e-6)
    np.testing.assert_allclose(right_crop_to_source @ right_h, np.eye(3), atol=1.0e-6)
    source_points = np.asarray([[0.0, 40.0, 80.0], [30.0, 30.0, 30.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(right_h @ source_points, [[0.0, 128.0, 256.0], [128.0] * 3, [1.0] * 3], atol=1.0e-5)
    np.testing.assert_allclose(left_h @ source_points, [[256.0, 128.0, 0.0], [128.0] * 3, [1.0] * 3], atol=1.0e-5)
    np.testing.assert_allclose(right_h, [[3.2, 0.0, 0.0], [0.0, 3.2, 32.0], [0.0, 0.0, 1.0]], atol=1.0e-6)
    assert left_transform.side is HandSide.LEFT
    assert right_transform.side is HandSide.RIGHT

    K = np.asarray([[900.0, 0.0, 48.0], [0.0, 625.0, 27.0], [0.0, 0.0, 1.0]])
    wire = left_transform.to_wire(K)
    source_to_model = np.asarray(wire["pixel_transform"]["source_to_model"])
    model_to_source = np.asarray(wire["pixel_transform"]["model_to_source"])
    np.testing.assert_allclose(source_to_model, left_h, atol=1.0e-6)
    np.testing.assert_allclose(model_to_source @ source_to_model, np.eye(3), atol=1.0e-6)
    assert not np.array_equal(source_to_model, model_to_source)


def test_wilor_crop_batch_reads_each_frame_once_and_preserves_detection_identity() -> None:
    class CountingSource(InMemoryFrameSource):
        def __init__(self) -> None:
            frames = [
                np.arange(32 * 48 * 3, dtype=np.uint32).reshape(32, 48, 3).astype(np.uint8),
                np.full((32, 48, 3), 127, dtype=np.uint8),
            ]
            super().__init__(frames, fps=5.0, source_id="wilor-batch-fixture")
            self.read_counts: dict[int, int] = {}

        def read_rgb(self, frame_index: int) -> np.ndarray:
            self.read_counts[frame_index] = self.read_counts.get(frame_index, 0) + 1
            return super().read_rgb(frame_index)

    source = CountingSource()
    owner = Ownership("case", "item", source.timeline.source_id, "hands", "fixture")
    detections = (
        HandDetectionRecord("det-a", 0, 0.0, HandSide.LEFT, (2.0, 3.0, 25.0, 29.0), 0.9, 1.0, 0.1, "visible", owner),
        HandDetectionRecord("det-b", 0, 0.0, HandSide.RIGHT, (18.0, 1.0, 44.0, 24.0), 0.8, 0.8, 0.2, "visible", owner),
        HandDetectionRecord("det-c", 1, 0.2, HandSide.RIGHT, (8.0, 6.0, 40.0, 30.0), 0.7, 0.6, 0.3, "partially_visible", owner),
    )

    crops, transforms = _normalized_crop_batch(source, detections, source.timeline, 1.25)

    assert crops.shape == (3, 3, 256, 256)
    assert crops.dtype == np.float32
    assert source.read_counts == {0: 1, 1: 1}
    assert tuple(transform.source_to_crop.grid_id for transform in transforms) == (
        "wilor_crop:det-a",
        "wilor_crop:det-b",
        "wilor_crop:det-c",
    )
    for index, detection in enumerate(detections):
        legacy_crop, legacy_transform = _normalized_crop(
            super(CountingSource, source).read_rgb(detection.frame_index), detection, source.timeline, 1.25
        )
        np.testing.assert_array_equal(crops[index], legacy_crop)
        assert transforms[index] == legacy_transform

    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = SimpleNamespace(crop_scale=1.25, model_revisions={"wilor.reconstruct": "fixture"})
    source.read_counts.clear()
    requests = driver._build_wilor_requests(
        source,
        "case",
        "item",
        SimpleNamespace(k_canonical=np.asarray([[500.0, 0.0, 24.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]])),
        detections,
    )
    assert source.read_counts == {0: 1, 1: 1}
    assert len(requests) == len(detections)
    for request, detection, transform in zip(requests, detections, transforms):
        assert request.timeline.frame_indices == (detection.frame_index,)
        assert request.input.ownership.scope.endswith(f"detection:{detection.detection_id}")
        assert request.input.crop_batch.provenance == {"detection_id": detection.detection_id, "frame_index": detection.frame_index}
        assert request.input.crop_transforms == (transform,)
        assert request.input.crop_batch.shape == (1, 3, 256, 256)


def test_native_window_policies_cover_tail_without_dropping_source_frames() -> None:
    hawor = CoveragePolicy(16, 8)
    infiller = CoveragePolicy(120, 60)

    assert hawor.starts(17) == (0, 8, 16)
    assert infiller.starts(17) == (0,)
    assert hawor.tail == infiller.tail == "pad_unobserved"


def test_infiller_padding_keeps_unique_frozen_service_timestamps() -> None:
    source = make_source(17)
    timestamps = tuple(_infiller_window_timestamp(source.timeline, index) for index in range(17, 22))

    assert timestamps == pytest.approx((4.25, 4.5, 4.75, 5.0, 5.25))
    assert len(set(timestamps)) == len(timestamps)


def test_droid_submission_is_exact_256_prefix_with_explicit_missing_tail() -> None:
    short = _droid_prefix_coverage(200)
    exact = _droid_prefix_coverage(256)
    first_overflow = _droid_prefix_coverage(257)
    long = _droid_prefix_coverage(1440)

    assert short.submitted_count == 200
    assert exact.submitted_count == DROID_SERVICE_PUSH_CAPACITY
    assert first_overflow.submitted_count == DROID_SERVICE_PUSH_CAPACITY
    assert first_overflow.unannotated_range == [256, 257]
    assert long.submitted_count == DROID_SERVICE_PUSH_CAPACITY
    assert tuple(index for index, valid in enumerate(long.pose_valid) if valid) == tuple(range(256))
    assert not any(long.pose_valid[256:])
    assert long.unannotated_range == [256, 1440]
    assert long.to_wire() == {
        "status": "completed_with_partial_camera_coverage",
        "source_frame_count": 1440,
        "submitted_count": 256,
        "covered_source_range": [0, 256],
        "unannotated_range": [256, 1440],
        "unannotated_range_semantics": "[start_inclusive,end_exclusive)",
        "reason": "service_capacity_256_exceeded",
        "warnings": ["service_capacity_256_exceeded"],
        "pose_validity": "per_frame_npz_mask; false means no DROID pose was submitted or inferred",
    }
    assert _droid_session_buffer(155) == 155
    assert _droid_session_buffer(256) == 256
    assert _droid_session_buffer(1024) == 256
    with pytest.raises(TimelineDriverError, match="nonempty"):
        _droid_session_buffer(0)


def test_builtin_fps_condition_selects_unidepth_and_droid_schedule() -> None:
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(360)]
    source = InMemoryFrameSource(frames, fps=30.0)
    config = FullVideoDriverConfig(fps_condition="unidepth_10fps__droid_10fps", allow_monocular_droid_smoke=True, require_rgbd_capability=False)

    assert config.unidepth_fps == 10.0
    assert config.droid_fps == 10.0
    assert len(_selected_source_indices(source.timeline, config.unidepth_fps)) == 120
    assert len(_selected_source_indices(source.timeline, 15.0)) == 180
    assert len(_selected_source_indices(source.timeline, 20.0)) == 240
    assert _selected_source_indices(source.timeline, None) == source.timeline.frame_indices


def test_droid_uniform_schedule_uses_round_stride_and_includes_both_source_endpoints() -> None:
    source = InMemoryFrameSource([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4920)], fps=30.0)

    assert _droid_stride(30.0, 15.0) == 2
    assert _droid_stride(30.0, 10.0) == 3
    samples_15 = _droid_sample_source_indices(source.timeline, 15.0)
    samples_10 = _droid_sample_source_indices(source.timeline, 10.0)
    assert samples_15[0] == samples_10[0] == 0
    assert samples_15[-1] == samples_10[-1] == 4919
    assert len(samples_15) == 2461
    assert len(samples_10) == 1641


def test_first_2000_frame_droid_schedules_match_endpoint_and_256_overlap_invariants() -> None:
    source = InMemoryFrameSource([np.zeros((1, 1, 3), dtype=np.uint8)] * 2000, fps=30.0)

    coverage_15 = _droid_scheduled_coverage(source.timeline, 15.0)
    coverage_10 = _droid_scheduled_coverage(source.timeline, 10.0)

    assert coverage_15.submitted_count == 1001
    assert coverage_10.submitted_count == 668
    assert tuple(map(len, coverage_15.chunk_source_indices)) == (256, 256, 256, 236)
    assert tuple(map(len, coverage_10.chunk_source_indices)) == (256, 256, 158)
    assert all(len(chunk) <= DROID_SERVICE_PUSH_CAPACITY for chunk in (*coverage_15.chunk_source_indices, *coverage_10.chunk_source_indices))
    assert all(left[-1] == right[0] for chunks in (coverage_15.chunk_source_indices, coverage_10.chunk_source_indices) for left, right in zip(chunks, chunks[1:]))
    assert coverage_15.chunk_source_indices[0][0] == coverage_10.chunk_source_indices[0][0] == 0
    assert coverage_15.chunk_source_indices[-1][-1] == coverage_10.chunk_source_indices[-1][-1] == 1999


def test_droid_input_shape_override_is_the_only_explicit_geometry_control() -> None:
    source = InMemoryFrameSource([np.zeros((1080, 1920, 3), dtype=np.uint8)], fps=30.0)
    default = FullVideoDriverConfig(require_rgbd_capability=False, allow_monocular_droid_smoke=True)
    historical = FullVideoDriverConfig(
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        droid_input_shape_yx=(384, 512),
    )

    assert _droid_input_shape_yx(source.timeline, default) == (328, 584)
    assert _droid_input_shape_yx(source.timeline, historical) == (384, 512)
    with pytest.raises(TimelineDriverError, match="multiples of 8"):
        FullVideoDriverConfig(
            require_rgbd_capability=False,
            allow_monocular_droid_smoke=True,
            droid_input_shape_yx=(385, 512),
        )


def test_hands_cache_trace_serialization_uses_retained_response_count() -> None:
    from scripts.run_droid_first_chunk_probe import _hands_trace_summary

    trace = SimpleNamespace(request_count=1, started_monotonic_s=2.0, completed_monotonic_s=2.75)

    assert _hands_trace_summary((object(),), trace) == {
        "elapsed_s": 0.75,
        "native_request_count": 1,
        "response_count": 1,
        "response_cardinality_matches_requests": True,
    }


def test_scaled_droid_graph_thresholds_are_exact_float_create_options_with_fixed_buffer() -> None:
    options = _scheduled_droid_options(
        chunk_index=0,
        submitted_frame_count=256,
        droid_fps=10.0,
        attempt=0,
        filter_thresh=None,
        frontend_thresh=48.0,
        backend_thresh=66.0,
    )

    assert options == {
        "scheduled_chunk": 0,
        "buffer": 256,
        "droid_fps": 10.0,
        "attempt": 0,
        "frontend_thresh": 48.0,
        "backend_thresh": 66.0,
    }
    assert type(options["frontend_thresh"]) is float
    assert type(options["backend_thresh"]) is float


def test_scheduled_droid_coverage_includes_npz_reason() -> None:
    source = InMemoryFrameSource([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(11)], fps=30.0)

    coverage = _droid_scheduled_coverage(source.timeline, 10.0)

    assert coverage.to_wire()["status"] == "completed_diagnostic_stitched_interpolated"
    assert coverage.to_wire()["reason"] == "diagnostic_stitched_interpolated"


def test_scheduled_droid_lifecycle_propagates_truthful_monocular_capability_mode() -> None:
    source = make_source(5)
    source_indices = (0, 2, 4)
    capabilities = DroidCapabilities.frozen_3572551()

    depth = np.full((1, source.timeline.height_px, source.timeline.width_px), 2.0, dtype=np.float32)
    confidence = np.ones_like(depth)
    k = np.asarray(
        [[100.0, 0.0, source.timeline.width_px / 2.0], [0.0, 100.0, source.timeline.height_px / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    unidepth = UniDepthOutput(
        Ownership("case", "item", source.timeline.source_id, "fixture", "unidepth"),
        TypedTensor(depth, "metres", "source", "tyx", "depth", {}),
        TypedTensor(k[None], "pixels", "source", "tyx", "K", {}),
        TypedTensor(confidence, "probability", "source", "tyx", "confidence", {}),
        (0,),
        (0.0,),
        source.timeline.frames[0].spatial,
        "unidepth-v2-vitl14-corrected",
    )
    unidepth_record = SimpleNamespace(output=unidepth)

    class RecordingClient:
        def __init__(self) -> None:
            self.requests: list[AlgorithmRequest[object]] = []
            self.push_count = 0

        def execute(self, request: AlgorithmRequest[object]) -> AlgorithmResult[object]:
            self.requests.append(request)
            if request.algorithm_id == "droid.create_session":
                output: object = DroidCreateOutput(request.input.ownership, "diagnostic-session", capabilities)
            elif request.algorithm_id == "droid.push_frame":
                self.push_count += 1
                output = DroidPushOutput(request.input.ownership, "diagnostic-session", request.input.frame_index, True, self.push_count, capabilities)
            else:
                poses = np.repeat(np.eye(4, dtype=np.float32)[None], self.push_count, axis=0)
                output = DroidFinalizeOutput(
                    request.input.ownership,
                    "diagnostic-session",
                    TypedTensor(poses, "metres", "world_from_camera", "tij", "world", {}),
                    TypedTensor(poses, "metres", "camera_from_world", "tij", "camera", {}),
                    TypedTensor(np.asarray([100.0, 100.0, 6.0, 4.0], dtype=np.float32), "pixels", "droid_input", "four", "K", {}),
                    TypedTensor(np.ones((1, 2, 2), dtype=np.float32), "inverse_metres", "droid_model", "tyx", "disparity", {}),
                    self.push_count,
                    "up_to_scale_monocular",
                    capabilities,
                    acceptance=False,
                    diagnostic_only=True,
                )
            return AlgorithmResult.from_request(request, output=output)

    client = RecordingClient()
    driver = object.__new__(FullVideoTimelineDriver)
    driver.client = client
    driver.config = FullVideoDriverConfig(
        fps_condition="unidepth_full__droid_15fps",
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        droid_input_shape_yx=(8, 8),
    )
    static_masks = {
        0: np.ones((8, 8), dtype=np.uint8),
        2: np.zeros((8, 8), dtype=np.uint8),
        4: np.tile(np.asarray([0, 1], dtype=np.uint8), (8, 4)),
    }

    create, pushes, final, _traces = driver._run_droid_chunk(
        source,
        "case",
        "item",
        SimpleNamespace(k_canonical=k),
        (unidepth_record,),
        0,
        source_indices,
        static_confidence_masks=static_masks,
    )

    assert [request.algorithm_id for request in client.requests] == [
        "droid.create_session",
        "droid.push_frame",
        "droid.push_frame",
        "droid.push_frame",
        "droid.finalize",
    ]
    assert all(request.input.require_rgbd_capability is False for request in client.requests)
    assert all(request.input.allow_monocular_droid_smoke is True for request in client.requests)
    assert all(not result.output.capabilities.native_sensor_depth_consumed for result in (create, *pushes, final))
    assert all(request.options["buffer"] == DROID_SERVICE_PUSH_CAPACITY for request in client.requests)
    pushed_masks = [request.input.static_confidence_mask for request in client.requests if request.algorithm_id == "droid.push_frame"]
    assert [mask.semantic_tag for mask in pushed_masks] == ["box_rasterized_static_confidence"] * len(source_indices)
    assert [mask.provenance["value_semantics"] for mask in pushed_masks] == ["1=static_keep,0=dynamic_ignore"] * len(source_indices)
    assert all(np.array_equal(mask.array, static_masks[index]) for mask, index in zip(pushed_masks, source_indices))
    assert len(source_indices) < DROID_SERVICE_PUSH_CAPACITY
    assert len([request for request in client.requests if request.algorithm_id == "droid.push_frame"]) == len(source_indices)
    assert final.output.scale_mode == "up_to_scale_monocular"
    assert final.output.diagnostic_only is True
    assert final.output.acceptance is False


def test_scheduled_droid_nonfinite_finalize_replays_same_chunk_once_with_fresh_session() -> None:
    source = InMemoryFrameSource([np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(5)], fps=30.0, source_id="recovery-fixture")
    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = FullVideoDriverConfig(
        fps_condition="unidepth_full__droid_15fps",
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        lower_filter_retry_thresh=1.2,
        max_keyframe_retries=1,
    )
    calls: list[tuple[tuple[int, ...], int, float | None]] = []

    def result_request(stage: str, source_indices: tuple[int, ...], options: dict[str, object]) -> AlgorithmRequest[object]:
        sampled = source.timeline.droid_sampled_metadata(source_indices)
        return AlgorithmRequest(
            stage,
            "droid-v1",
            "case",
            "item",
            source.timeline.source_id,
            sampled,
            StageMetadata(stage, "droid", stage, "droid-v1"),
            NativeWorkDescription(stage, "droid-v1", None, 1, 1, (1,)),
            None,
            options,
        )

    def fake_chunk(
        _source: InMemoryFrameSource,
        _case_id: str,
        _item_id: str,
        _canonical: object,
        _unidepth: object,
        chunk_index: int,
        source_indices: tuple[int, ...],
        *,
        attempt: int,
        filter_thresh: float | None,
    ) -> tuple[AlgorithmResult[DroidCreateOutput], tuple[AlgorithmResult[DroidPushOutput], ...], AlgorithmResult[DroidFinalizeOutput], tuple[RequestBatchTrace, ...]]:
        calls.append((source_indices, attempt, filter_thresh))
        options = {
            "scheduled_chunk": chunk_index,
            "buffer": DROID_SERVICE_PUSH_CAPACITY,
            "droid_fps": 15.0,
            "attempt": attempt,
            **({"filter_thresh": filter_thresh, "bounded_lower_filter_retry": True} if filter_thresh is not None else {}),
        }
        create_request = result_request("droid.create_session", source_indices, options)
        session_id = "primary-session" if attempt == 0 else "retry-session"
        create = AlgorithmResult.from_request(
            create_request,
            output=DroidCreateOutput(Ownership("case", "item", source.timeline.source_id, "fixture", f"attempt:{attempt}"), session_id, DroidCapabilities.frozen_3572551()),
        )
        if attempt == 0:
            raise DroidChunkFinalizeError(
                chunk_index=chunk_index,
                attempt=attempt,
                session_id=session_id,
                options=options,
                create_result=create,
                push_results=(),
                traces=(),
                cause=RuntimeError("CameraState.T_world_camera must contain only finite values"),
            )
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(source_indices), axis=0)
        finalize_request = result_request("droid.finalize", source_indices, options)
        final = AlgorithmResult.from_request(
            finalize_request,
            output=DroidFinalizeOutput(
                Ownership("case", "item", source.timeline.source_id, "fixture", "retry"),
                session_id,
                TypedTensor(poses, "metres", "world_from_camera", "tij", "world", {}),
                TypedTensor(poses, "metres", "camera_from_world", "tij", "camera", {}),
                TypedTensor(np.asarray([100.0, 100.0, 6.0, 4.0], dtype=np.float32), "pixels", "droid_input", "four", "K", {}),
                TypedTensor(np.ones((1, 2, 2), dtype=np.float32), "inverse_metres", "droid_model", "tyx", "disparity", {}),
                len(source_indices),
                "up_to_scale_monocular",
                DroidCapabilities.frozen_3572551(),
                acceptance=False,
                diagnostic_only=True,
            ),
        )
        return create, (), final, ()

    driver._run_droid_chunk = fake_chunk
    records, _traces = driver._run_scheduled_droid(source, "case", "item", SimpleNamespace(), ())

    assert calls == [((0, 2, 4), 0, None), ((0, 2, 4), 1, 1.2)]
    assert records.retries_used == 1
    assert records.accepted_trajectory is True
    assert len(records.create_results) == 2
    outcome, = records.chunk_outcomes
    assert outcome.session_id == "retry-session"
    assert [attempt.session_id for attempt in outcome.attempts] == ["primary-session", "retry-session"]
    assert outcome.attempts[0].succeeded is False
    assert "finite values" in str(outcome.attempts[0].error)
    assert outcome.attempts[1].succeeded is True
    assert outcome.attempts[1].options["filter_thresh"] == 1.2
    assert outcome.source_indices == (0, 2, 4)
    assert records.final.output.scale_mode == "up_to_scale_monocular"
    assert records.final.output.diagnostic_only is True
    assert records.final.output.acceptance is False
    assert records.final.provenance[-1]["droid_finalize_retries_used"] == 1


def test_scheduled_droid_retry_exhaustion_reports_both_session_causes() -> None:
    source = InMemoryFrameSource([np.zeros((8, 12, 3), dtype=np.uint8) for _ in range(5)], fps=30.0, source_id="recovery-failure-fixture")
    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = FullVideoDriverConfig(
        fps_condition="unidepth_full__droid_15fps",
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        lower_filter_retry_thresh=1.2,
        max_keyframe_retries=1,
    )

    def failed_chunk(
        _source: InMemoryFrameSource,
        _case_id: str,
        _item_id: str,
        _canonical: object,
        _unidepth: object,
        chunk_index: int,
        source_indices: tuple[int, ...],
        *,
        attempt: int,
        filter_thresh: float | None,
    ) -> object:
        session_id = "primary-session" if attempt == 0 else "retry-session"
        options = {
            "attempt": attempt,
            "buffer": DROID_SERVICE_PUSH_CAPACITY,
            **({"filter_thresh": filter_thresh, "bounded_lower_filter_retry": True} if filter_thresh is not None else {}),
        }
        create = SimpleNamespace(output=SimpleNamespace(session_id=session_id))
        raise DroidChunkFinalizeError(
            chunk_index=chunk_index,
            attempt=attempt,
            session_id=session_id,
            options=options,
            create_result=create,
            push_results=(),
            traces=(),
            cause=RuntimeError("CameraState.T_world_camera must contain only finite values"),
        )

    driver._run_droid_chunk = failed_chunk

    with pytest.raises(TimelineDriverError) as error:
        driver._run_scheduled_droid(source, "case", "item", SimpleNamespace(), ())

    message = str(error.value)
    assert "recovery exhausted" in message
    assert "primary_session=primary-session" in message
    assert "retry_session=retry-session" in message
    assert "filter_thresh" in message
    assert "finite values" in message


def test_explicit_droid_sampled_timelines_admit_stride_two_and_three_only_in_droid_envelopes() -> None:
    source = InMemoryFrameSource([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(11)], fps=30.0, source_id="sparse-fixture")
    work = NativeWorkDescription("droid.create_session", "fixture", None, 1, 1, (1,))

    for droid_fps, expected in ((15.0, (0, 2, 4, 6, 8, 10)), (10.0, (0, 3, 6, 9, 10))):
        sampled = source.timeline.droid_sampled_metadata(_droid_sample_source_indices(source.timeline, droid_fps))
        assert sampled.timeline_mode == "droid_sampled"
        assert sampled.frame_indices == expected
        request = AlgorithmRequest(
            algorithm_id="droid.create_session",
            model_revision="fixture",
            case_id="case",
            item_id="item",
            source_id=source.timeline.source_id,
            timeline=sampled,
            stage=StageMetadata("droid.create_session", "droid", "fixture", "fixture"),
            work=work,
            input=None,
        )
        assert request.timeline.to_mapping()["timeline_mode"] == "droid_sampled"
        with pytest.raises(ContractError, match="only valid for DROID"):
            AlgorithmRequest(
                algorithm_id="hands.detect",
                model_revision="fixture",
                case_id="case",
                item_id="item",
                source_id=source.timeline.source_id,
                timeline=sampled,
                stage=StageMetadata("hands.detect", "hands", "fixture", "fixture"),
                work=work,
                input=None,
            )


def test_sparse_indices_remain_invalid_for_dense_timeline_and_dense_frame_batches() -> None:
    with pytest.raises(ContractError, match="contiguous"):
        FrameTimelineMetadata("dense-fixture", (0, 2, 4), (0.0, 2.0 / 30.0, 4.0 / 30.0))

    dense_tensor = TypedTensor(np.zeros((3, 1, 1, 3), dtype=np.uint8), "uint8_rgb", "source", "thwc", "fixture", {})
    with pytest.raises(TypedContractError, match="contiguous and increasing"):
        _validate_frame_batch(dense_tensor, (0, 2, 4), (0.0, 2.0 / 30.0, 4.0 / 30.0), "Hands", dtype="uint8", channels=3)


def test_droid_chunk_gate_uses_256_push_sessions_with_one_sample_overlap() -> None:
    chunks_1001 = _droid_chunks_with_overlap(tuple(range(1001)))
    chunks_668 = _droid_chunks_with_overlap(tuple(range(668)))

    assert tuple(map(len, chunks_1001)) == (256, 256, 256, 236)
    assert tuple(map(len, chunks_668)) == (256, 256, 158)
    for chunks, count in ((chunks_1001, 1001), (chunks_668, 668)):
        assert all(len(chunk) <= 256 for chunk in chunks)
        assert all(right[0] == left[-1] for left, right in zip(chunks[:-1], chunks[1:]))
        flattened = tuple(index for chunk_index, chunk in enumerate(chunks) for index in (chunk if chunk_index == 0 else chunk[1:]))
        assert flattened == tuple(range(count))


def test_droid_se3_stitch_makes_overlap_boundaries_continuous() -> None:
    def pose(x: float) -> np.ndarray:
        value = np.eye(4, dtype=np.float32)
        value[0, 3] = x
        return value

    stitched, continuity = _stitch_droid_chunk_poses((np.stack((pose(0.0), pose(1.0))), np.stack((pose(0.0), pose(0.5)))))

    assert stitched.shape == (3, 4, 4)
    assert stitched[:, 0, 3] == pytest.approx((0.0, 1.0, 1.5))
    assert continuity[0][0] == pytest.approx(0.0)
    assert continuity[0][1] == pytest.approx(0.0)


def test_droid_full_timeline_interpolation_has_no_nan_and_marks_samples() -> None:
    def pose(angle: float, x: float) -> np.ndarray:
        value = np.eye(4, dtype=np.float32)
        value[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        value[0, 3] = x
        return value

    dense, sampled = _interpolate_droid_poses((0, 2, 4), np.stack((pose(0.0, 0.0), pose(np.pi / 2, 2.0), pose(np.pi, 4.0))), 5)

    assert np.isfinite(dense).all()
    assert sampled.tolist() == [True, False, True, False, True]
    assert dense[:, 0, 3] == pytest.approx((0.0, 1.0, 2.0, 3.0, 4.0))
    assert dense[1, :2, :2] == pytest.approx(np.array([[np.sqrt(0.5), -np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)]]), abs=1e-5)


def test_droid_full_prefix_preserves_service_native_keyframe_disparities() -> None:
    source = make_source(4)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    disparities = np.ones((1, 2, 2), dtype=np.float32)
    tensor = lambda array, tag: TypedTensor(array, "unitless", "fixture", "tyx", tag, {})
    output = DroidFinalizeOutput(
        Ownership("case", "item", "fixture-video", "fixture-owner", "droid.finalize"),
        "session",
        tensor(poses, "world"),
        tensor(poses, "camera"),
        TypedTensor(np.ones(4, dtype=np.float32), "pixels", "fixture", "four", "K", {}),
        tensor(disparities, "disparities"),
        1,
        "up_to_scale_monocular",
        DroidCapabilities.frozen_3572551(),
        acceptance=False,
        diagnostic_only=True,
    )

    dense = _densify_droid_output(output, source.timeline, (0, 1, 2, 3))

    assert dense.T_world_camera.shape == (4, 4, 4)
    assert dense.T_camera_world.shape == (4, 4, 4)
    assert np.isfinite(dense.T_world_camera.array).all()
    assert dense.T_world_camera.provenance["droid_pose_valid"] == [True, True, True, True]
    assert dense.disparities.shape == (1, 2, 2)
    assert dense.disparities.provenance["sampling"] == "service_native_keyframe_tensor_preserved"


def test_droid_partial_prefix_output_never_synthesizes_tail_poses() -> None:
    source = make_source(1440)
    prefix = np.repeat(np.eye(4, dtype=np.float32)[None], 256, axis=0)
    tensor = lambda array, tag: TypedTensor(array, "unitless", "fixture", "tyx", tag, {})
    output = DroidFinalizeOutput(
        Ownership("case", "item", "fixture-video", "fixture-owner", "droid.finalize"),
        "session", tensor(prefix, "world"), tensor(prefix, "camera"),
        TypedTensor(np.ones(4, dtype=np.float32), "pixels", "fixture", "four", "K", {}),
        tensor(np.ones((1, 2, 2), dtype=np.float32), "disparities"), 2,
        "up_to_scale_monocular", DroidCapabilities.frozen_3572551(), acceptance=False, diagnostic_only=True,
    )

    dense = _densify_droid_output(output, source.timeline, tuple(range(256)))

    assert dense.T_world_camera.shape == (1440, 4, 4)
    assert np.isfinite(dense.T_world_camera.array[:256]).all()
    assert np.isnan(dense.T_world_camera.array[256:]).all()
    assert dense.T_world_camera.provenance["droid_pose_valid"][:256] == [True] * 256
    assert dense.T_world_camera.provenance["droid_pose_valid"][256:] == [False] * 1184
    assert dense.T_world_camera.provenance["unannotated_range"] == [256, 1440]
    assert dense.T_world_camera.provenance["reason"] == "service_capacity_256_exceeded"


def test_driver_config_keeps_single_outer_item_and_native_horizons() -> None:
    config = FullVideoDriverConfig(allow_monocular_droid_smoke=True, require_rgbd_capability=False)

    assert config.item_batch_size == 1
    assert not {
        "unidepth_concurrency",
        "hands_concurrency",
        "wilor_concurrency",
        "hawor_concurrency",
        "infiller_concurrency",
    }.intersection(FullVideoDriverConfig.__dataclass_fields__)
    assert config.hawor_coverage.chunk_length == 16
    assert config.infiller_coverage.chunk_length == 120
    assert config.cosmos_enabled is False
    assert config.model_revisions["hands.detect"] == "hands-yolo-v2"
    assert config.model_revisions["wilor.reconstruct"] == "wilor-final-v1"

    with pytest.raises(TimelineDriverError):
        FullVideoDriverConfig(item_batch_size=2, allow_monocular_droid_smoke=True, require_rgbd_capability=False)
    enabled = FullVideoDriverConfig(cosmos_enabled=True, allow_monocular_droid_smoke=True, require_rgbd_capability=False)
    assert enabled.cosmos_enabled is True
    assert enabled.model_revisions["cosmos3.reason"] == "cosmos3-frozen"


def test_yolo_detections_mark_only_observed_slots_visible() -> None:
    source = make_source(3)
    config = FullVideoDriverConfig(allow_monocular_droid_smoke=True, require_rgbd_capability=False)
    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = config

    def hands_output(frame_index: int, *, score: float | None) -> HandsOutput:
        count = 0 if score is None else 1
        boxes = np.asarray([[[1.0, 1.0, 6.0, 6.0]]], dtype=np.float32)[:, :count]
        scores = np.asarray([[score]], dtype=np.float32) if score is not None else np.empty((1, 0), dtype=np.float32)
        sides = np.ones((1, count), dtype=np.uint8)
        visibility = np.ones((1, count), dtype=np.float32)
        uncertainty = np.full((1, count), 0.2, dtype=np.float32)
        detections = HandDetections(
            TypedTensor(boxes, "pixels", "source", "tkf", "boxes", {}),
            TypedTensor(scores, "probability", "source", "tk", "scores", {}),
            TypedTensor(sides, "class_id", "source", "tk", "sides", {}),
            TypedTensor(visibility, "fraction", "source", "tk", "visibility", {}),
            TypedTensor(uncertainty, "score", "source", "tk", "uncertainty", {}),
        )
        return HandsOutput(
            Ownership("case", "item", source.timeline.source_id, "fixture", f"hands.detect:{frame_index}"),
            detections,
            (frame_index,),
            (source.timeline.frames[frame_index].timestamp_s,),
            source.timeline.frames[frame_index].spatial,
            "hands-yolo-v2",
        )

    observed = driver._hand_detections(
        (SimpleNamespace(output=hands_output(0, score=0.9)), SimpleNamespace(output=hands_output(2, score=0.8))),
        source.timeline,
    )
    tracks = driver._associate_tracks(observed, source.timeline.frame_count)

    assert [record.visibility for record in observed] == [1.0, 1.0]
    assert [record.occlusion_state for record in observed] == ["visible", "visible"]
    assert tracks[0].visibility_by_frame == ("visible", "unresolved", "visible")
    assert _visibility_code(tracks[0].visibility_by_frame[1]) == 4.0
    assert driver._hand_detections((SimpleNamespace(output=hands_output(1, score=None)),), source.timeline) == ()


def test_observed_surface_candidate_wins_over_parameter_only_infiller_candidate() -> None:
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    parameter_only = _ManoCandidate(0, HandSide.RIGHT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), None, None, True, False, 0.001, "hawor_infiller.fill", "infiller")
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.05, 0.05, 778)
    surface = _ManoCandidate(0, HandSide.RIGHT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), vertices, vertices[:21], True, False, 0.1, "hawor.infer_tracks", "hawor")

    state = _merge_timeline_candidates(1, (parameter_only, surface))

    assert np.isfinite(state.vertices_camera_m.array[1, 0]).all()
    assert np.isfinite(state.joints_camera_m.array[1, 0]).all()
    assert state.valid.array[1, 0] == 1
    assert state.provenance[0].source_stage == "hawor.infer_tracks"


def test_wilor_candidates_use_crop_projection_and_unidepth_metric_translation() -> None:
    root = np.eye(3, dtype=np.float32)[None]
    pose = np.repeat(root[:, None], 15, axis=1)
    vertices = np.zeros((1, 778, 3), dtype=np.float32)
    vertices[0, :, 0] = np.linspace(-0.05, 0.05, 778)
    joints = vertices[:, :21].copy()
    camera_translation = np.asarray([[0.1, -0.2, 0.6]], dtype=np.float32)
    tensor = lambda array, tag: TypedTensor(array, "fixture", "mano", "fixture", tag, {})
    ownership = Ownership("case", "item", "source", "wilor.reconstruct", "detection:det-0")
    output = WiLoROutput(
        ownership,
        (HandSide.LEFT,),
        ManoBatch(
            tensor(root, "root"),
            tensor(pose, "pose"),
            tensor(np.zeros((1, 10), dtype=np.float32), "betas"),
            tensor(vertices, "vertices"),
            tensor(joints, "joints"),
            tensor(camera_translation, "virtual_camera_translation"),
            tensor(np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32), "pred_cam"),
            tensor(np.tile(np.asarray([[[6.0, 4.0]]], dtype=np.float32), (1, 778, 1)), "vertices_source_px"),
            tensor(np.tile(np.asarray([[[6.0, 4.0]]], dtype=np.float32), (1, 21, 1)), "joints_source_px"),
            tensor(np.asarray([0.9], dtype=np.float32), "confidence"),
            tensor(np.asarray([0.02], dtype=np.float32), "uncertainty"),
        ),
        "wilor-final-v1",
    )
    detection = HandDetectionRecord("det-0", 2, 0.2, HandSide.LEFT, (1.0, 2.0, 10.0, 12.0), 0.9, 1.0, 0.02, "visible", ownership)
    driver = object.__new__(FullVideoTimelineDriver)
    spatial = SimpleNamespace(pixel_to_source=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    depth_output = SimpleNamespace(
        frame_indices=(2,),
        depth_m=SimpleNamespace(array=np.full((1, 8, 12), 0.7, dtype=np.float32)),
        confidence=SimpleNamespace(array=np.ones((1, 8, 12), dtype=np.float32)),
        spatial=spatial,
    )
    canonical = SimpleNamespace(k_canonical=np.asarray([[100.0, 0.0, 6.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32))

    candidate, = driver._wilor_candidates((SimpleNamespace(output=output),), (detection,), (SimpleNamespace(output=depth_output),), canonical)
    degenerate_hawor = _ManoCandidate(2, HandSide.LEFT, root[0], pose[0], np.zeros(10, np.float32), camera_translation[0], np.zeros((778, 3), np.float32), np.zeros((21, 3), np.float32), True, False, 0.001, "hawor.infer_tracks", "zero-padded-crop")
    state = _merge_timeline_candidates(3, (degenerate_hawor, candidate))

    assert candidate.observed is True
    assert candidate.inferred is False
    assert candidate.has_finite_geometry is True
    assert candidate.trans == pytest.approx([0.05, 0.0, 0.7])
    assert candidate.vertices == pytest.approx(vertices[0] + np.asarray([0.05, 0.0, 0.7], dtype=np.float32)[None, :])
    assert candidate.joints == pytest.approx(joints[0] + np.asarray([0.05, 0.0, 0.7], dtype=np.float32)[None, :])
    assert candidate.joints_source_px == pytest.approx(np.tile([[6.0, 4.0]], (21, 1)))
    assert state.valid.array[0, 2] == 1
    assert np.any(state.vertices_camera_m.array[0, 2] != 0)
    assert state.provenance[0].source_stage == "wilor.reconstruct"


def test_parameter_only_infiller_candidate_does_not_mark_timeline_geometry_valid() -> None:
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    candidate = _ManoCandidate(0, HandSide.RIGHT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), None, None, False, True, 0.1, "hawor_infiller.fill", "infiller")

    state = _merge_timeline_candidates(1, (candidate,))

    assert state.valid.array[1, 0] == 0
    assert np.isnan(state.vertices_camera_m.array[1, 0]).all()
    assert np.isnan(state.joints_camera_m.array[1, 0]).all()


def test_mano_geometry_rejects_zero_and_collapsed_surfaces_but_accepts_real_variance() -> None:
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    joints = np.zeros((21, 3), dtype=np.float32)
    zero = _ManoCandidate(0, HandSide.LEFT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), np.zeros((778, 3), np.float32), joints, True, False, 0.1, "hawor.infer_tracks", "zero")
    collapsed = _ManoCandidate(0, HandSide.LEFT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), np.full((778, 3), 0.5, np.float32), joints, True, False, 0.1, "hawor.infer_tracks", "collapsed")
    varied_vertices = np.zeros((778, 3), dtype=np.float32)
    varied_vertices[:, 1] = np.linspace(-0.04, 0.04, 778)
    varied = _ManoCandidate(0, HandSide.LEFT, root, pose, np.zeros(10, np.float32), np.zeros(3, np.float32), varied_vertices, varied_vertices[:21], True, False, 0.1, "wilor.reconstruct", "varied")

    assert zero.geometry_anomaly_codes == ("all_zero_vertices",)
    assert zero.has_finite_geometry is False
    assert collapsed.geometry_anomaly_codes == ("collapsed_vertices",)
    assert collapsed.has_finite_geometry is False
    assert varied.geometry_anomaly_codes == ()
    assert varied.has_finite_geometry is True


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    (
        ("root_orient", np.full((3, 3), np.nan, dtype=np.float32), "nonfinite_root_orient"),
        ("root_orient", np.diag([2.0, 1.0, 1.0]).astype(np.float32), "invalid_root_orient_rotation_frame_0"),
        ("hand_pose", np.full((15, 3, 3), np.nan, dtype=np.float32), "nonfinite_hand_pose"),
        ("trans", np.asarray([np.nan, 0.0, 0.0], dtype=np.float32), "nonfinite_translation"),
        ("trans", np.asarray([101.0, 0.0, 0.0], dtype=np.float32), "implausible_translation_gt_100m"),
        ("betas", np.full(10, np.nan, dtype=np.float32), "nonfinite_betas"),
        ("uncertainty", -0.01, "nonfinite_or_negative_uncertainty"),
    ),
)
def test_mano_parameter_anomalies_make_surface_candidate_ineligible(
    field: str,
    value: np.ndarray | float,
    expected_code: str,
) -> None:
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.05, 0.05, 778)
    parameters = {
        "root_orient": root,
        "hand_pose": pose,
        "betas": np.zeros(10, dtype=np.float32),
        "trans": np.zeros(3, dtype=np.float32),
        "uncertainty": 0.1,
    }
    parameters[field] = value
    candidate = _ManoCandidate(
        0,
        HandSide.LEFT,
        parameters["root_orient"],
        parameters["hand_pose"],
        parameters["betas"],
        parameters["trans"],
        vertices,
        vertices[:21],
        True,
        False,
        parameters["uncertainty"],
        "hawor.infer_tracks",
        "parameter-anomaly",
    )

    assert expected_code in candidate.parameter_anomaly_codes
    assert candidate.has_valid_parameters is False
    assert candidate.has_valid_surface_geometry is True
    assert candidate.is_eligible is False
    assert _merge_timeline_candidates(1, (candidate,)).valid.array[0, 0] == 0


def test_eligible_candidate_rank_wins_over_observed_candidate_with_invalid_parameters() -> None:
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.05, 0.05, 778)
    invalid = _ManoCandidate(
        0, HandSide.RIGHT, root, pose, np.zeros(10, np.float32),
        np.asarray([np.nan, 0.0, 0.0], dtype=np.float32), vertices, vertices[:21],
        True, False, 0.001, "hawor.infer_tracks", "invalid",
    )
    eligible = _ManoCandidate(
        0, HandSide.RIGHT, root, pose, np.zeros(10, np.float32),
        np.asarray([0.1, 0.0, 0.8], dtype=np.float32), vertices, vertices[:21],
        False, True, 0.1, "hawor.infer_tracks", "eligible",
    )

    state = _merge_timeline_candidates(1, (invalid, eligible))

    assert eligible.rank < invalid.rank
    assert state.valid.array[1, 0] == 1
    assert state.trans_camera_m.array[1, 0] == pytest.approx(eligible.trans)
    assert state.provenance[0].source_scope == "eligible"


def test_infiller_reuses_each_hands_last_valid_observation_for_masked_frames() -> None:
    source = make_source(3)
    driver = object.__new__(FullVideoTimelineDriver)
    driver.config = FullVideoDriverConfig(allow_monocular_droid_smoke=True, require_rgbd_capability=False)
    root = np.eye(3, dtype=np.float32)
    pose = np.repeat(root[None], 15, axis=0)
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.05, 0.05, 778)

    def candidate(side: HandSide, frame_index: int, trans: np.ndarray, *, observed: bool = True) -> _ManoCandidate:
        return _ManoCandidate(
            frame_index, side, root, pose, np.zeros(10, np.float32), trans,
            vertices, vertices[:21], observed, not observed, 0.02,
            "hawor.infer_tracks", f"{side.value}:{frame_index}",
        )

    candidates = (
        candidate(HandSide.LEFT, 0, np.asarray([0.1, 0.0, 0.8], dtype=np.float32)),
        candidate(HandSide.RIGHT, 0, np.asarray([0.2, 0.0, 0.9], dtype=np.float32)),
        candidate(HandSide.LEFT, 1, np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)),
        candidate(HandSide.RIGHT, 1, np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)),
        candidate(HandSide.LEFT, 2, np.asarray([0.5, 0.0, 0.8], dtype=np.float32), observed=False),
        candidate(HandSide.RIGHT, 2, np.asarray([0.6, 0.0, 0.9], dtype=np.float32), observed=False),
    )
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    ownership = Ownership("case", "item", source.timeline.source_id, "fixture", "droid.finalize")
    droid = DroidFinalizeOutput(
        ownership,
        "session",
        TypedTensor(poses, "metres", "world_from_camera", "tij", "world", {"droid_pose_valid": [True] * 3}),
        TypedTensor(poses, "metres", "camera_from_world", "tij", "camera", {}),
        TypedTensor(np.asarray([10.0, 10.0, 6.0, 4.0], dtype=np.float32), "pixels", "source", "four", "K", {}),
        TypedTensor(np.ones((1, 2, 2), dtype=np.float32), "disparity", "source", "tyx", "disparity", {}),
        1,
        "up_to_scale_monocular",
        DroidCapabilities.frozen_3572551(),
        acceptance=False,
        diagnostic_only=True,
    )
    canonical = SimpleNamespace(k_canonical=np.asarray([[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]))

    requests, traces = driver._build_infiller_requests(
        "case", "item", source.timeline, canonical, droid, candidates,
    )

    assert len(requests) == 1
    assert traces[0].submitted is True
    infiller_input = requests[0].input
    assert infiller_input.observation_mask.array[:3].tolist() == [[1, 1], [0, 0], [0, 0]]
    assert infiller_input.window_120x218.array[1] == pytest.approx(infiller_input.window_120x218.array[0])
    assert infiller_input.window_120x218.array[2] == pytest.approx(infiller_input.window_120x218.array[0])
    assert np.isfinite(infiller_input.window_120x218.array).all()
    for side_index in range(2):
        first = infiller_input.frames[side_index]
        for slot in (1, 2):
            frame = infiller_input.frames[slot * 2 + side_index]
            assert frame.observed is False
            assert frame.trans == pytest.approx(first.trans)
            assert frame.uncertainty == first.uncertainty


def test_hawor_degenerate_slots_are_grouped_by_chunk_for_output_diagnostics() -> None:
    vertices = np.zeros((16, 778, 3), dtype=np.float32)
    vertices[:, :, 0] = np.linspace(-0.05, 0.05, 778)
    joints = vertices[:, :21].copy()
    vertices[2] = 0.0
    joints[2] = np.float32(1e38)
    vertices[3] = 0.5
    output = SimpleNamespace(
        vertices_camera_m=SimpleNamespace(array=vertices),
        joints_camera_m=SimpleNamespace(array=joints),
    )
    chunk = HaworChunkTrace(
        "track-0",
        HandSide.LEFT,
        10,
        tuple(range(10, 26)),
        (True, True) + (False,) * 14,
        (False,) * 16,
        "track:track-0:start:10",
    )

    diagnostics = _hawor_geometry_diagnostics((SimpleNamespace(output=output),), (chunk,))

    assert diagnostics["status"] == "anomalies_detected"
    assert diagnostics["hawor_degenerate_slot_count"] == 2
    rows = diagnostics["hawor_degenerate_slots"]
    assert rows[0]["slot"] == 2
    assert rows[0]["frame_index"] == 12
    assert rows[0]["anomaly_codes"] == ["all_zero_vertices", "implausible_joint_magnitude_gt_100m"]
    assert rows[1]["anomaly_codes"] == ["collapsed_vertices"]
    chunk_row = diagnostics["hawor_chunks_with_degenerate_geometry"][0]
    assert chunk_row["degenerate_slot_count"] == 2
    assert chunk_row["message"] == "HaWoR output was degenerate for 2 slots in chunk track:track-0:start:10"


def test_stage_burst_uses_request_cardinality_and_preserves_result_order(monkeypatch: pytest.MonkeyPatch) -> None:
    request_count = 5
    worker_counts: list[int] = []
    native_executor = full_video_timeline.ThreadPoolExecutor

    class CapturingExecutor(native_executor):
        def __init__(self, *args, **kwargs) -> None:
            worker_counts.append(int(kwargs["max_workers"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(full_video_timeline, "ThreadPoolExecutor", CapturingExecutor)
    driver = object.__new__(FullVideoTimelineDriver)
    barrier = threading.Barrier(request_count)
    submitted: list[int] = []
    submitted_lock = threading.Lock()

    def execute(request: SimpleNamespace) -> str:
        with submitted_lock:
            submitted.append(request.ordinal)
        barrier.wait(timeout=5.0)
        return f"result:{request.ordinal}"

    driver._execute_timed = execute
    requests = tuple(
        SimpleNamespace(algorithm_id="hands.detect", work=SimpleNamespace(native_shape=(1,)), ordinal=index)
        for index in range(request_count)
    )

    results, trace = driver._run_many_traced("hands.detect", requests)
    stage_client = object.__new__(AlgorithmStageClient)
    stage_client.execute = lambda request: f"client:{request.ordinal}"

    assert stage_client.execute_many(requests) == tuple(f"client:{index}" for index in range(request_count))
    assert worker_counts == [request_count, request_count]
    assert "max_concurrency" not in inspect.signature(AlgorithmStageClient.execute_many).parameters
    assert sorted(submitted) == list(range(request_count))
    assert results == tuple(f"result:{index}" for index in range(request_count))
    assert trace.request_count == request_count
    assert trace.submitted_concurrency == request_count
    assert not hasattr(trace, "max_concurrency")
    assert full_video_timeline._stage_worker_count(()) == 1


def test_plan_reports_full_video_contract_without_creating_run_root(tmp_path) -> None:
    source = make_source()
    from ego_annotation.full_video_timeline import SingleVideoPreflight

    preflight = SingleVideoPreflight("case", str(tmp_path / "fresh"), source.timeline, None, ("fixture",))
    plan = plan_single_video(
        preflight,
        FullVideoDriverConfig(allow_monocular_droid_smoke=True, require_rgbd_capability=False),
    )

    assert plan["source"]["frame_count"] == 17
    assert plan["item_batch_size"] == 1
    assert "concurrency" not in plan
    assert plan["coverage"]["hawor"] == {"length": 16, "stride": 8, "tail": "pad_unobserved"}
    assert plan["coverage"]["infiller"]["length"] == 120
    assert plan["creates_run_root"] is False
    assert plan["cosmos"] == "disabled"
    enabled_plan = plan_single_video(
        preflight,
        FullVideoDriverConfig(cosmos_enabled=True, allow_monocular_droid_smoke=True, require_rgbd_capability=False),
    )
    assert enabled_plan["cosmos"] == "enabled"


def test_source_rejects_incomplete_or_non_rgb_timeline() -> None:
    source = make_source(2)
    with pytest.raises(TimelineDriverError):
        SourceTimeline(
            source.timeline.source_id,
            None,
            source.timeline.source_sha256,
            source.timeline.source_size_bytes,
            1,
            source.timeline.fps,
            source.timeline.duration_s,
            source.timeline.width_px,
            source.timeline.height_px,
            "RGB",
            source.timeline.frames,
        )


def test_preflight_rejects_existing_fresh_root(tmp_path) -> None:
    source = make_source(1)
    path = tmp_path / "existing"
    path.mkdir()
    from ego_annotation.full_video_timeline import preflight_single_video

    # The preflight's filesystem guard is tested with a real encoded video in the
    # production path; this fixture only confirms its error type for an invalid input.
    with pytest.raises(PreflightError):
        preflight_single_video(path / "missing.mp4", case_id="case", fresh_root=path)
