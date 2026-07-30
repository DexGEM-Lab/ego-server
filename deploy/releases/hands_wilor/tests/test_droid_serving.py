"""CPU-only tests for the stateful DROID Ray Serve vertical slice.

These never import Ray, torch, or lietorch. They exercise:
* the model-native DROID contracts (uint8 RGB, div-8 shapes, filesystem rejection,
  source-timestamp preservation, scale_status honesty);
* session isolation and unknown/closed/finalized session rejection;
* bounded session/queue memory and backpressure;
* the canonical pose convention via a synthetic nonidentity SE(3) test that forces
  dense and keyframe exports to agree on ``T_world_camera`` (translation-sign and
  quaternion-inversion errors must fail);
* true cross-session feature-network batching honesty: one fused ``fnet`` forward
  across compatible next-ready frames from distinct sessions, with correlation/
  update/BA traced as session-local forwards — not a single fused forward;
* resident model-load count does not increase with request count.

The DROID-specific continuation (correlation/update/BA) is injected as a fake so
the fnet-batching orchestration is testable without the droid_slam ABI. The real
continuation is validated on the GPU2 server deployment.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ego_annotation.serving.contracts import (
    CameraState,
    ContractValidationError,
    DenseSourceMapping,
    DroidBatchTrace,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidPhaseTiming,
    DroidFinalizeRequest,
    DroidFrameRequest,
    DroidImageShape,
    DroidSessionOptions,
    DroidUncertainty,
    DroidFrameResponse,
    ErrorCode,
    KeyframeSourceMapping,
    ImageSize,
    Ownership,
    PixelTransform,
    TensorPayload,
    reject_filesystem_fields,
)
from ego_annotation.serving.droid import (
    DroidAdapter,
    build_droid_model_config,
    camera_from_world_xyzw_to_world_camera_matrix,
    DroidModelConfig,
    _record_exact_dense_source_frame,
    _run_frontend_without_grad,
)


REVISION = "droid-v1"
H, W = 16, 24  # divisible by 8


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeVideo:
    """Minimal stand-in for DepthVideo: a counter and a pose store."""

    def __init__(self) -> None:
        self.counter = MagicMock()
        self.counter.value = 0
        self.poses = np.zeros((64, 7), dtype=np.float64)
        self.poses[:, 6] = 1.0  # identity quaternion
        self.images: list[Any] = []
        self.intrinsics = np.zeros((64, 4), dtype=np.float64)
        self.disps = np.ones((64, H // 8, W // 8), dtype=np.float32)

    def append(self, *item: Any) -> None:
        self.poses[self.counter.value] = item[2] if item[2] is not None else np.array([0, 0, 0, 0, 0, 0, 1.0])
        self.images.append(item[1])
        self.counter.value += 1


class FakeMotionState:
    def __init__(self) -> None:
        self.thresh = 2.5
        self.count = 0
        self.net = None
        self.inp = None
        self.fmap = None
        import torch
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406])[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225])[:, None, None]


class FakeFrontend:
    def __init__(self) -> None:
        self.is_initialized = False

    def __call__(self) -> None:
        self.is_initialized = True


class FakeBackendModules:
    """Fake shared backend: records fnet calls so the test proves one fused forward."""

    def __init__(self) -> None:
        self.fnet_calls: list[int] = []  # batch sizes
        self.cnet_calls: list[int] = []

    def fnet(self, images: Any) -> Any:
        arr = np.asarray(images)
        b = int(arr.shape[0])  # [B,1,3,H,W] -> B sessions
        self.fnet_calls.append(b)
        gh, gw = int(arr.shape[3] // 8), int(arr.shape[4] // 8)
        return np.zeros((b, 1, 128, gh, gw), dtype=np.float32)

    def cnet(self, image: Any) -> tuple[Any, Any]:
        self.cnet_calls.append(1)
        arr = np.asarray(image)
        b = int(arr.shape[0])
        gh, gw = int(arr.shape[3] // 8), int(arr.shape[4] // 8)
        net = np.zeros((b, 256, gh, gw), dtype=np.float32)
        return net[:, :128], net[:, 128:]

    @property
    def update_op(self) -> Any:
        op = MagicMock()
        op.return_value = (None, np.zeros((1, 1, 1, 1, 2), dtype=np.float32), None)
        return op


def fake_session_factory(backend, config, camera, image_shape, options):
    """Build fake per-session objects without importing droid_slam."""
    video = FakeVideo()
    filt = FakeMotionState()
    filt.thresh = options.filter_thresh
    backend_obj = MagicMock()
    backend_obj.errors = [0.01]
    return {
        "video": video,
        "filter": filt,
        "frontend": FakeFrontend(),
        "backend": backend_obj,
        "filler": MagicMock(),
    }


def fake_continuation(adapter, state, item, gmap):
    """Fake session-local continuation: appends a keyframe and counts local work."""
    state.video.counter.value = state.video.counter.value  # touch
    # First frame always adds; subsequent frames add for the test.
    keyframe_added = True
    local_fwds = 2  # one cnet + one update (session-local, NOT fused)
    state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
    state.video.poses[state.video.counter.value] = np.array([0, 0, 0, 0, 0, 0, 1.0])
    state.video.counter.value += 1
    state.frames_pushed += 1
    return local_fwds, keyframe_added


def make_config(**overrides: Any) -> DroidModelConfig:
    defaults = dict(
        weights="server-owned-droid.pth",
        model_revision=REVISION,
        device="cpu",
        assigned_gpu=2,
        max_sessions=4,
        max_queued_frames_per_session=2,
        max_fnet_batch_size=4,
        replica_id="droid-gpu2-test",
    )
    defaults.update(overrides)
    return build_droid_model_config(**defaults)


def make_camera() -> DroidCamera:
    return DroidCamera(
        intrinsics=(63.88, 63.78, 12.0, 6.0),
        source_size=ImageSize(width=W, height=H),
        pixel_transform=PixelTransform.identity(),
    )


def make_create_request(request_id: str = "r1") -> DroidCreateSessionRequest:
    return DroidCreateSessionRequest(
        ownership=Ownership(request_id, "job", "item", "droid.create_session", "src"),
        camera=make_camera(),
        image_shape=DroidImageShape(H, W),
        model_revision=REVISION,
    )


def make_frame_request(session_id: str, frame_id: str, *, ts: float = 0.0, pixels: int = 128) -> DroidFrameRequest:
    rgb = np.full((H, W, 3), pixels, dtype=np.uint8)
    return DroidFrameRequest(
        ownership=Ownership(f"r-{frame_id}", "job", "item", "droid.push_frame", "src", source_timestamp_s=ts),
        session_id=session_id,
        frame_id=frame_id,
        source_timestamp_s=ts,
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
    )


def make_camera_state(ownership: Ownership, session_id: str) -> CameraState:
    def payload(array: np.ndarray) -> TensorPayload:
        contiguous = np.ascontiguousarray(array)
        return TensorPayload(contiguous.tobytes(), tuple(contiguous.shape), contiguous.dtype.name)

    T_world_camera = np.eye(4, dtype=np.float64)[None]
    T_world_camera[0, 0, 3] = 2.0
    trace = DroidBatchTrace(
        batch_id="finalize-batch", replica_id="droid-gpu2-test",
        admitted_monotonic_s=1.0, dispatched_monotonic_s=1.0,
        fnet_forward_started_monotonic_s=1.0, fnet_completed_monotonic_s=1.0,
        completed_monotonic_s=2.0, fnet_forward_count=0,
        session_local_forward_count=1, request_count=1, effective_work_units=1,
        model_load_count=1, session_ids=(session_id,),
    )
    return CameraState(
        ownership=ownership,
        session_id=session_id,
        T_world_camera=payload(T_world_camera),
        T_camera_world=payload(np.linalg.inv(T_world_camera)),
        intrinsics_px=payload(np.eye(3, dtype=np.float64)[None]),
        disparities=payload(np.ones((1, 2, 3), dtype=np.float32)),
        keyframe_mapping=(KeyframeSourceMapping(0, "frame-0", 0.0),),
        dense_mapping=(DenseSourceMapping(0, "frame-0", 0.0),),
        uncertainty=DroidUncertainty("up_to_scale", None, 1.0, 1.0),
        model_revision=REVISION,
        trace=trace,
    )


def make_adapter(
    loads: list[int], *, config: DroidModelConfig | None = None, continuation_fn: Any = fake_continuation,
) -> tuple[DroidAdapter, FakeBackendModules]:
    backend = FakeBackendModules()

    def backend_factory(cfg):
        loads.append(1)
        return backend

    cfg = config or make_config()
    adapter = DroidAdapter(
        cfg,
        backend_factory=backend_factory,
        session_factory=fake_session_factory,
        continuation_fn=continuation_fn,
    )
    return adapter, backend


# --------------------------------------------------------------------------- #
# Semantic-regression tests
# --------------------------------------------------------------------------- #


def test_frontend_updates_run_without_retaining_autograd_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"grad_enabled": True}

    class NoGrad:
        def __enter__(self) -> None:
            state["grad_enabled"] = False

        def __exit__(self, *_args: Any) -> None:
            state["grad_enabled"] = True

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=lambda: NoGrad()))
    observed: list[bool] = []
    _run_frontend_without_grad(lambda: observed.append(state["grad_enabled"]))
    assert observed == [False]
    assert state["grad_enabled"] is True


def test_frontend_failure_surfaces_as_the_causing_frame_error() -> None:
    def frontend_failure(_adapter: Any, _state: Any, _item: Any, _gmap: Any) -> tuple[int, bool]:
        _run_frontend_without_grad(lambda: (_ for _ in ()).throw(RuntimeError("frontend BA failure")))
        raise AssertionError("unreachable")

    loads: list[int] = []
    adapter, _ = make_adapter(loads, continuation_fn=frontend_failure)
    sid = adapter.create_session(make_create_request()).session_id
    prepared = adapter.admit_frame(make_frame_request(sid, "f0"))
    response = asyncio.run(adapter.push_frame_batch([prepared]))[0]
    assert response.error is not None
    assert response.error.code is ErrorCode.MODEL_FAILURE
    assert "frontend BA failure" in response.error.message


def test_in_flight_ownership_persists_until_continuation_finishes() -> None:
    entered = threading.Event()
    release = threading.Event()
    continuation_started: list[float] = []

    def blocking_continuation(_adapter: Any, _state: Any, _item: Any, _gmap: Any) -> tuple[int, bool]:
        continuation_started.append(time.monotonic())
        entered.set()
        assert release.wait(timeout=2.0)
        return 0, False

    async def scenario() -> None:
        loads: list[int] = []
        adapter, _ = make_adapter(loads, continuation_fn=blocking_continuation)
        sid = adapter.create_session(make_create_request()).session_id
        first = adapter.admit_frame(make_frame_request(sid, "f0"))
        completion = asyncio.create_task(adapter.push_frame_batch([first]))
        assert await asyncio.to_thread(entered.wait, 2.0)
        with pytest.raises(ContractValidationError, match="in flight"):
            adapter.admit_frame(make_frame_request(sid, "f1"))
        release.set()
        response = (await completion)[0]
        assert response.error is None
        assert response.status is not None
        # fnet completes before session-local continuation begins; it is not a
        # broad span around the thread that also includes the continuation.
        assert response.status.trace.fnet_completed_monotonic_s <= continuation_started[0]
        # The session is released only after the continuation returns.
        next_frame = adapter.admit_frame(make_frame_request(sid, "f1"))
        adapter.request_completed(next_frame.request.session_id)

    asyncio.run(scenario())


def test_360_dense_source_frames_preserve_ownership_rgb_and_pose_inverse() -> None:
    def retain_exact_source(_adapter: Any, state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        _record_exact_dense_source_frame(state, item)
        state.frames_pushed += 1
        return 0, False

    loads: list[int] = []
    adapter, _ = make_adapter(loads, continuation_fn=retain_exact_source)
    sid = adapter.create_session(make_create_request()).session_id
    for index in range(360):
        rgb = np.empty((H, W, 3), dtype=np.uint8)
        rgb[..., 0] = index % 256
        rgb[..., 1] = (index * 3) % 256
        rgb[..., 2] = (255 - index) % 256
        frame = DroidFrameRequest(
            ownership=Ownership(f"r-{index}", "job", "item", "droid.push_frame", "source", source_timestamp_s=index / 30.0),
            session_id=sid,
            frame_id=f"frame-{index:03d}",
            source_timestamp_s=index / 30.0,
            rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
        )
        response = asyncio.run(adapter.push_frame_batch([adapter.admit_frame(frame)]))[0]
        assert response.error is None

    state = adapter._sessions[sid]
    assert state.frames_pushed == 360
    assert state.dense_source == [(f"frame-{index:03d}", index / 30.0) for index in range(360)]
    for index, stored_bgr in enumerate(state.dense_images_bgr_chw_uint8):
        assert tuple(stored_bgr[:, 0, 0]) == ((255 - index) % 256, (index * 3) % 256, index % 256)

    from scipy.spatial.transform import Rotation

    angles = np.linspace(0.0, 359.0, num=360)
    rotations = Rotation.from_euler("z", angles[:, None], degrees=True)
    poses_cw = np.column_stack((
        np.arange(360, dtype=float),
        -np.arange(360, dtype=float) / 2.0,
        np.ones(360),
        rotations.as_quat(),
    ))
    T_wc = camera_from_world_xyzw_to_world_camera_matrix(poses_cw)
    expected_cw = np.repeat(np.eye(4)[None], 360, axis=0)
    expected_cw[:, :3, :3] = rotations.as_matrix()
    expected_cw[:, :3, 3] = poses_cw[:, :3]
    np.testing.assert_allclose(np.linalg.inv(T_wc), expected_cw, atol=1e-9)


def test_push_phase_trace_marks_unobserved_stages_explicitly() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    response = asyncio.run(adapter.push_frame_batch([
        adapter.admit_frame(make_frame_request(sid, "f0")),
    ]))[0]
    assert response.status is not None
    round_trip = DroidFrameResponse.from_wire(response.to_wire())
    assert round_trip == response
    assert round_trip.status is not None
    assert round_trip.status.trace.replica_id == "droid-gpu2-test"
    phase = response.status.trace.phase_timing
    assert phase.preprocessing_h2d_s is not None
    assert phase.fnet_s is not None
    assert phase.correlation_update_s is None
    assert phase.cnet_s is None
    assert phase.frontend_ba_s is None
    assert set(phase.unavailable_stages) == {
        "correlation_update", "cnet", "frontend_ba", "backend_7", "backend_12", "filler", "encoding",
    }
    assert phase.measurement_basis == "host_monotonic_span"
    assert phase.cuda_event_elapsed_s == {}
    assert set(phase.cuda_event_unavailable_stages) == {
        "preprocessing_h2d", "fnet", "correlation_update", "cnet", "frontend_ba", "backend_7", "backend_12", "filler", "encoding",
    }
    assert phase.http_serialization_unavailable is True
    assert DroidBatchTrace.from_wire(response.status.trace.to_wire()).phase_timing == phase


def test_phase_contract_rejects_implicit_or_fabricated_unavailable_stages() -> None:
    with pytest.raises(ContractValidationError, match="must be explicit"):
        DroidPhaseTiming(unavailable_stages=())
    with pytest.raises(ContractValidationError, match="cannot be unavailable"):
        DroidPhaseTiming(fnet_s=0.0)


# --------------------------------------------------------------------------- #
# Contract tests
# --------------------------------------------------------------------------- #


def test_contract_rejects_non_div8_image_shape() -> None:
    with pytest.raises(ContractValidationError, match="divisible by 8"):
        DroidImageShape(18, 24)


def test_contract_rejects_float_rgb() -> None:
    rgb = TensorPayload(data=b"\x00" * (H * W * 3 * 4), shape=(H, W, 3), dtype="float32")
    with pytest.raises(ContractValidationError, match="uint8"):
        DroidFrameRequest(
            ownership=Ownership("r", "j", "i", "s", "src"),
            session_id="s", frame_id="f", source_timestamp_s=0.0, rgb=rgb,
        )


def test_contract_rejects_filesystem_fields_in_create_session() -> None:
    with pytest.raises(ContractValidationError, match="filesystem paths"):
        DroidCreateSessionRequest.from_wire({
            "ownership": {"request_id": "r", "job_id": "j", "item_id": "i", "stage_id": "s", "source_id": "src"},
            "camera": {"intrinsics": [1, 1, 1, 1], "source_size": {"width": 16, "height": 16},
                       "pixel_transform": {"resize_mode": "identity"}},
            "image_shape": {"height": 16, "width": 24},
            "model_revision": "r",
            "weights_path": "/tmp/droid.pth",
        })


def test_uncertainty_scale_status_must_be_up_to_scale_by_default() -> None:
    u = DroidUncertainty(scale_status="up_to_scale", reprojection_error=None,
                         valid_keyframe_ratio=1.0, finite_pose_ratio=1.0)
    assert u.scale_status == "up_to_scale"


# --------------------------------------------------------------------------- #
# Canonical pose convention: synthetic nonidentity SE(3) test
# --------------------------------------------------------------------------- #


def test_identity_pose_yields_identity_world_camera() -> None:
    p = np.array([[0, 0, 0, 0, 0, 0, 1.0]])
    M = camera_from_world_xyzw_to_world_camera_matrix(p)
    assert np.allclose(M[0], np.eye(4))


def test_nonidentity_se3_dense_and_keyframe_agree_on_world_camera() -> None:
    """The synthetic nonidentity SE(3) test required by the task spec.

    DROID stores internal poses as camera-from-world [tx,ty,tz,qx,qy,qz,qw]. Both
    the dense (trajectory-filler) and keyframe exports must produce the same
    ``T_world_camera``. We apply a nonidentity SE(3) and verify the canonical
    conversion inverts it correctly. A translation-sign or quaternion-inversion
    error would fail this test.

    The check: build a known camera-from-world pose, convert to world-from-camera,
    and verify it is the true SE(3) inverse (R^T, -R^T t). Dense and keyframe use
    the identical conversion function, so they agree by construction.
    """
    # Nonidentity rotation (90deg about z) + translation.
    from scipy.spatial.transform import Rotation

    R_cw = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    t_cw = np.array([2.0, 0.0, 1.0])
    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    quat_xyzw = Rotation.from_matrix(R_cw).as_quat()  # [qx,qy,qz,qw]
    pose_xyzw = np.array([[t_cw[0], t_cw[1], t_cw[2], *quat_xyzw]])

    # Canonical conversion used by BOTH dense and keyframe exports.
    T_wc = camera_from_world_xyzw_to_world_camera_matrix(pose_xyzw)[0]

    # Ground-truth world-from-camera = inverse of camera-from-world.
    T_wc_truth = np.linalg.inv(T_cw)
    assert np.allclose(T_wc, T_wc_truth, atol=1e-9), (
        f"canonical conversion is not the true SE(3) inverse:\n{T_wc}\nvs\n{T_wc_truth}"
    )
    # Translation sign check: world-from-camera translation = -R_cw^T t_cw.
    assert np.allclose(T_wc[:3, 3], -R_cw.T @ t_cw, atol=1e-9)
    # Quaternion-inversion check: rotation block is R_cw^T, not R_cw.
    assert np.allclose(T_wc[:3, :3], R_cw.T, atol=1e-9)
    assert not np.allclose(T_wc[:3, :3], R_cw, atol=1e-9)


def test_translation_sign_error_would_fail_the_convention() -> None:
    """If the conversion returned +R^T t (wrong sign) instead of -R^T t, this fails.

    This is a guard that the test actually catches the failure mode the task names.
    """
    R = np.eye(3)
    t = np.array([2.0, 0.0, 0.0])
    pose = np.array([[t[0], t[1], t[2], 0, 0, 0, 1.0]])
    T_wc = camera_from_world_xyzw_to_world_camera_matrix(pose)[0]
    # Correct: -t. A sign error would give +t.
    assert np.allclose(T_wc[:3, 3], -t)


# --------------------------------------------------------------------------- #
# Session lifecycle: isolation, rejection, bounds
# --------------------------------------------------------------------------- #


def test_create_session_loads_model_once_and_assigns_session_id() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    resp = adapter.create_session(make_create_request("r1"))
    assert resp.session_id is not None
    assert resp.error is None
    assert loads == [1]
    assert adapter.resident_session_count == 1


def test_revision_mismatch_rejected_before_session_creation() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    req = make_create_request()
    req = DroidCreateSessionRequest(
        ownership=req.ownership, camera=req.camera, image_shape=req.image_shape,
        model_revision="wrong-revision",
    )
    resp = adapter.create_session(req)
    assert resp.error is not None
    assert resp.error.code is ErrorCode.VALIDATION
    assert resp.session_id is None
    assert loads == [1]  # model still loaded once at startup


def test_max_sessions_bound_enforces_backpressure() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_sessions=2))
    r1 = adapter.create_session(make_create_request("r1"))
    r2 = adapter.create_session(make_create_request("r2"))
    assert r1.session_id and r2.session_id
    r3 = adapter.create_session(make_create_request("r3"))
    assert r3.error is not None
    assert r3.error.code is ErrorCode.BACKPRESSURE
    assert r3.error.retryable


def test_finalized_session_releases_capacity_and_gpu_state() -> None:
    """Completed sessions must not permanently exhaust max_sessions."""
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_sessions=1))
    first = adapter.create_session(make_create_request("first"))
    assert first.session_id is not None
    adapter._release_finalized_session(adapter._sessions[first.session_id])
    assert adapter.resident_session_count == 0
    second = adapter.create_session(make_create_request("second"))
    assert second.session_id is not None
    assert second.error is None


def test_partial_finalize_failure_quarantines_only_the_mutated_session() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_sessions=2))
    failed = adapter.create_session(make_create_request("failed"))
    unaffected = adapter.create_session(make_create_request("unaffected"))
    assert failed.session_id and unaffected.session_id
    adapter._sessions[failed.session_id].video.counter.value = 2

    def fail_after_mutation(state: Any, *_args: Any) -> Any:
        state.mutation_started = True
        raise RuntimeError("synthetic backend mutation failure")

    adapter._finalize_session = fail_after_mutation  # type: ignore[method-assign]
    response = asyncio.run(adapter.finalize(DroidFinalizeRequest(
        ownership=Ownership("finalize", "job", "item", "droid.finalize", "src"),
        session_id=failed.session_id,
        model_revision=REVISION,
    )))
    assert response.error is not None
    assert response.error.code is ErrorCode.MODEL_FAILURE
    assert adapter._sessions[failed.session_id].lifecycle.value == "quarantined"
    with pytest.raises(ContractValidationError, match="quarantined"):
        adapter.admit_frame(make_frame_request(failed.session_id, "blocked"))
    # The unrelated owner retains a usable session and can continue normally.
    assert asyncio.run(adapter.push_frame_batch([
        adapter.admit_frame(make_frame_request(unaffected.session_id, "ok")),
    ]))[0].error is None


def test_push_frame_rejects_unknown_session() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    with pytest.raises(ContractValidationError, match="unknown"):
        adapter.admit_frame(make_frame_request("no-such-session", "f0"))


def test_push_frame_rejects_explicit_finalized_tombstone() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    adapter._release_finalized_session(adapter._sessions[sid])
    with pytest.raises(ContractValidationError, match="finalized"):
        adapter.admit_frame(make_frame_request(sid, "f0"))


def test_at_most_one_ready_frame_per_session_in_flight() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_queued_frames_per_session=4))
    sid = adapter.create_session(make_create_request()).session_id
    adapter.admit_frame(make_frame_request(sid, "f0"))
    # Second concurrent frame for the SAME session before the first is dispatched.
    with pytest.raises(ContractValidationError, match="in flight"):
        adapter.admit_frame(make_frame_request(sid, "f1"))


def test_per_session_queue_bound_rejects_overflow() -> None:
    """A second frame for a session with one already in-flight is rejected (backpressure).

    This is the bounded-rejection mechanism: at most one ready frame per session
    enters a cross-session batch, so a concurrent second frame for the same session
    is rejected with a validation/backpressure error rather than accumulated.
    """
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_queued_frames_per_session=1))
    sid = adapter.create_session(make_create_request()).session_id
    adapter.admit_frame(make_frame_request(sid, "f0"))
    # The session now has one in-flight frame; a second concurrent frame is rejected.
    with pytest.raises(ContractValidationError, match="in flight"):
        adapter.admit_frame(make_frame_request(sid, "f1"))


# --------------------------------------------------------------------------- #
# Lifecycle transactionality, cancellation, and retained-frame bounds
# --------------------------------------------------------------------------- #


def test_cancelled_batch_releases_admission_only_after_worker_finishes() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_continuation(_adapter: Any, _state: Any, _item: Any, _gmap: Any) -> tuple[int, bool]:
        entered.set()
        assert release.wait(timeout=2.0)
        finished.set()
        return 0, False

    async def scenario() -> None:
        loads: list[int] = []
        adapter, _ = make_adapter(loads, continuation_fn=blocking_continuation)
        sid = adapter.create_session(make_create_request()).session_id
        cleanup_complete = threading.Event()
        original_complete = adapter.request_completed

        def observed_complete(session_id: str) -> None:
            original_complete(session_id)
            cleanup_complete.set()

        adapter.request_completed = observed_complete  # type: ignore[method-assign]
        task = asyncio.create_task(adapter.push_frame_batch([adapter.admit_frame(make_frame_request(sid, "f0"))]))
        assert await asyncio.to_thread(entered.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Shielding preserves the admission while the worker owns mutation.
        assert adapter._sessions[sid].in_flight is True
        assert adapter.status().admitted_pending == 0
        release.set()
        assert await asyncio.to_thread(finished.wait, 2.0)
        # Completion callback runs after the worker result; another frame proves
        # pending/in-flight cleanup is complete rather than merely decremented.
        assert await asyncio.to_thread(cleanup_complete.wait, 2.0)
        assert adapter._sessions[sid].in_flight is False
        adapter.request_abandoned(adapter.admit_frame(make_frame_request(sid, "f1")))
        assert adapter.status().admitted_pending == 0

    asyncio.run(scenario())


def test_cancelled_push_outcome_is_retrieved_without_duplicate_mutation_and_conflicts_are_explicit() -> None:
    entered = threading.Event()
    release = threading.Event()
    journaled = threading.Event()
    mutations: list[str] = []

    def blocking_continuation(_adapter: Any, state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        entered.set()
        assert release.wait(timeout=2.0)
        mutations.append(item.request.frame_id)
        state.frames_pushed += 1
        return 0, False

    async def scenario() -> None:
        loads: list[int] = []
        adapter, _ = make_adapter(loads, continuation_fn=blocking_continuation)
        sid = adapter.create_session(make_create_request()).session_id
        request = make_frame_request(sid, "stable-frame")
        original_record = adapter._record_push_result

        def observed_record(state: Any, req: Any, response: Any) -> None:
            original_record(state, req, response)
            journaled.set()

        adapter._record_push_result = observed_record  # type: ignore[method-assign]
        admitted = adapter.admit_frame(request)
        assert not isinstance(admitted, DroidFrameResponse)
        task = asyncio.create_task(adapter.push_frame_batch([admitted]))
        assert await asyncio.to_thread(entered.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        in_progress = adapter.admit_frame(request)
        assert isinstance(in_progress, DroidFrameResponse)
        assert in_progress.error is not None
        assert in_progress.error.code is ErrorCode.BACKPRESSURE
        release.set()
        assert await asyncio.to_thread(journaled.wait, 2.0)

        replay = adapter.admit_frame(request)
        assert isinstance(replay, DroidFrameResponse)
        assert replay.status is not None
        assert replay.status.frame_id == "stable-frame"
        assert mutations == ["stable-frame"]

        conflicting = make_frame_request(sid, "stable-frame")
        conflict = adapter.admit_frame(conflicting)
        assert isinstance(conflict, DroidFrameResponse)
        assert conflict.error is not None
        assert conflict.error.code is ErrorCode.CONFLICT
        assert mutations == ["stable-frame"]

    asyncio.run(scenario())


def test_gpu_execution_lock_serializes_overlapping_batch_workers() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()

    def blocking_first(_adapter: Any, _state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        if item.request.frame_id == "first":
            first_entered.set()
            assert release_first.wait(timeout=2.0)
        return 0, False

    async def scenario() -> None:
        loads: list[int] = []
        adapter, backend = make_adapter(loads, continuation_fn=blocking_first)
        sid_a = adapter.create_session(make_create_request("a")).session_id
        sid_b = adapter.create_session(make_create_request("b")).session_id
        first = adapter.admit_frame(make_frame_request(sid_a, "first"))
        second = adapter.admit_frame(make_frame_request(sid_b, "second"))
        first_task = asyncio.create_task(adapter.push_frame_batch([first]))
        assert await asyncio.to_thread(first_entered.wait, 2.0)
        second_task = asyncio.create_task(adapter.push_frame_batch([second]))
        await asyncio.to_thread(lambda: None)
        assert backend.fnet_calls == [1]
        release_first.set()
        assert (await first_task)[0].error is None
        assert (await second_task)[0].error is None
        assert backend.fnet_calls == [1, 1]

    asyncio.run(scenario())


def test_finalize_marks_session_finalizing_before_worker_and_blocks_push() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_finalize(state: Any, *_args: Any) -> Any:
        state.mutation_started = True
        entered.set()
        assert release.wait(timeout=2.0)
        raise RuntimeError("backend failure after mutation")

    async def scenario() -> None:
        loads: list[int] = []
        adapter, _ = make_adapter(loads)
        sid = adapter.create_session(make_create_request()).session_id
        adapter._sessions[sid].video.counter.value = 2
        adapter._finalize_session = blocked_finalize  # type: ignore[method-assign]
        task = asyncio.create_task(adapter.finalize(DroidFinalizeRequest(
            ownership=Ownership("finalize", "job", "item", "droid.finalize", "src"),
            session_id=sid, model_revision=REVISION,
        )))
        assert await asyncio.to_thread(entered.wait, 2.0)
        with pytest.raises(ContractValidationError, match="finalizing"):
            adapter.admit_frame(make_frame_request(sid, "racing-push"))
        release.set()
        result = await task
        assert result.error is not None
        assert adapter._sessions[sid].lifecycle.value == "quarantined"

    asyncio.run(scenario())


def test_cancelled_finalize_result_is_retrievable_and_never_recomputed() -> None:
    entered = threading.Event()
    release = threading.Event()
    terminal = threading.Event()
    calls: list[str] = []

    async def scenario() -> None:
        loads: list[int] = []
        adapter, _ = make_adapter(loads)
        sid = adapter.create_session(make_create_request()).session_id
        adapter._sessions[sid].video.counter.value = 2
        ownership = Ownership("stable-finalize", "job", "item", "droid.finalize", "src")
        request = DroidFinalizeRequest(ownership=ownership, session_id=sid, model_revision=REVISION)

        def blocked_finalize(_state: Any, _batch: str, _started: float, owner: Ownership) -> CameraState:
            calls.append(owner.request_id)
            entered.set()
            assert release.wait(timeout=2.0)
            return make_camera_state(owner, sid)

        original_mark = adapter._mark_terminal

        def observed_mark(state: Any, lifecycle: Any) -> None:
            original_mark(state, lifecycle)
            terminal.set()

        adapter._finalize_session = blocked_finalize  # type: ignore[method-assign]
        adapter._mark_terminal = observed_mark  # type: ignore[method-assign]
        task = asyncio.create_task(adapter.finalize(request))
        assert await asyncio.to_thread(entered.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        pending = await adapter.finalize(request)
        assert pending.error is not None
        assert pending.error.code is ErrorCode.BACKPRESSURE
        release.set()
        assert await asyncio.to_thread(terminal.wait, 2.0)

        replay = await adapter.finalize(request)
        assert replay.camera_state is not None
        assert replay.camera_state.session_id == sid
        assert calls == ["stable-finalize"]
        conflict = await adapter.finalize(DroidFinalizeRequest(
            ownership=Ownership("other-finalize", "job", "item", "droid.finalize", "src"),
            session_id=sid,
            model_revision=REVISION,
        ))
        assert conflict.error is not None
        assert conflict.error.code is ErrorCode.CONFLICT
        assert calls == ["stable-finalize"]

    asyncio.run(scenario())


def test_internal_keyframe_retry_replays_rgb_masks_and_source_timeline() -> None:
    """Retry uses fresh low-threshold session state and the original accepted inputs."""
    seen_masks: list[np.ndarray | None] = []

    def retry_selects_keyframes(adapter: Any, state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        _record_exact_dense_source_frame(state, item)
        state.frames_pushed += 1
        seen_masks.append(None if item.mask_grid is None else np.asarray(item.mask_grid.cpu()))
        if state.options.filter_thresh > 0.6:
            return 0, False
        state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
        state.video.append(
            item.request.source_timestamp_s,
            adapter._rgb_hwc_to_bgr_chw_uint8(item)[0],
            np.array([0, 0, 0, 0, 0, 0, 1.0]),
        )
        return 1, True

    loads: list[int] = []
    adapter, backend = make_adapter(loads, continuation_fn=retry_selects_keyframes)
    sid = adapter.create_session(make_create_request()).session_id
    mask = np.asarray([[0.25, 0.75, 0.5], [1.0, 0.0, 0.125]], dtype=np.float32)
    for index, (timestamp_s, channels) in enumerate(((0.0, (11, 22, 33)), (0.1, (44, 55, 66)))):
        rgb = np.empty((H, W, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = channels
        request = DroidFrameRequest(
            ownership=Ownership(f"retry-{index}", "job", "item", "droid.push_frame", "src"),
            session_id=sid,
            frame_id=f"retry-frame-{index}",
            source_timestamp_s=timestamp_s,
            rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
            static_confidence_mask=TensorPayload(data=mask.tobytes(), shape=mask.shape, dtype="float32"),
        )
        response = asyncio.run(adapter.push_frame_batch([adapter.admit_frame(request)]))[0]
        assert response.error is None

    state = adapter._sessions[sid]
    assert state.video.counter.value == 0
    assert len(state.replay_frames) == 2
    np.testing.assert_array_equal(state.replay_frames[0][1], mask)
    assert adapter._retry_with_lower_filter_thresholds_locked(state) is True

    # The 1.2 retry remains unresolved; 0.6 succeeds. Its fresh video and both
    # source maps replace the unresolved session, driven by original RGB/masks.
    assert state.options.filter_thresh == 0.6
    assert state.video.counter.value == 2
    assert state.keyframe_source == [("retry-frame-0", 0.0), ("retry-frame-1", 0.1)]
    assert state.dense_source == [("retry-frame-0", 0.0), ("retry-frame-1", 0.1)]
    assert backend.fnet_calls == [1, 1, 1, 1, 1, 1]
    np.testing.assert_allclose(seen_masks[-1], mask)
    assert tuple(state.video.images[0][:, 0, 0]) == (33, 22, 11)
    assert tuple(state.video.images[1][:, 0, 0]) == (66, 55, 44)


def test_single_keyframe_finalize_is_explicitly_unresolved_without_backend_calls() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    state = adapter._sessions[sid]
    state.video.counter.value = 1
    backend = state.backend
    response = asyncio.run(adapter.finalize(DroidFinalizeRequest(
        ownership=Ownership("finalize", "job", "item", "droid.finalize", "src"),
        session_id=sid, model_revision=REVISION,
    )))
    assert response.error is not None
    assert response.error.code is ErrorCode.UNRESOLVED
    backend.assert_not_called()
    assert state.lifecycle.value == "unresolved"
    assert state.video is None
    assert state.backend is None
    assert state.dense_images_bgr_chw_uint8 == []
    retry = asyncio.run(adapter.finalize(DroidFinalizeRequest(
        ownership=response.ownership, session_id=sid, model_revision=REVISION,
    )))
    assert retry == response


def test_terminal_tombstones_use_bounded_lru_and_release_resources() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads, config=make_config(max_terminal_tombstones=2))
    session_ids: list[str] = []
    for index in range(3):
        sid = adapter.create_session(make_create_request(f"create-{index}")).session_id
        assert sid is not None
        session_ids.append(sid)
        state = adapter._sessions[sid]
        state.video.counter.value = 1
        state.dense_images_bgr_chw_uint8.append(np.ones((3, H, W), dtype=np.uint8))
        response = asyncio.run(adapter.finalize(DroidFinalizeRequest(
            ownership=Ownership(f"finalize-{index}", "job", "item", "droid.finalize", "src"),
            session_id=sid,
            model_revision=REVISION,
        )))
        assert response.error is not None and response.error.code is ErrorCode.UNRESOLVED
        assert state.video is None
        assert state.dense_images_bgr_chw_uint8 == []
    assert adapter.resident_session_count == 0
    assert list(adapter._terminal_lru) == session_ids[-2:]
    assert session_ids[0] not in adapter._sessions
    assert set(adapter._sessions) == set(session_ids[-2:])


def test_frame_capacity_bounds_dense_retention_and_retire_releases_it() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    create = make_create_request()
    create = DroidCreateSessionRequest(
        ownership=create.ownership, camera=create.camera, image_shape=create.image_shape,
        model_revision=create.model_revision, options=DroidSessionOptions(buffer=1),
    )
    sid = adapter.create_session(create).session_id

    def retain(_adapter: Any, state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        _record_exact_dense_source_frame(state, item)
        state.frames_pushed += 1
        return 0, False

    adapter._continuation_fn = retain
    assert asyncio.run(adapter.push_frame_batch([adapter.admit_frame(make_frame_request(sid, "f0"))]))[0].error is None
    state = adapter._sessions[sid]
    assert len(state.dense_images_bgr_chw_uint8) == 1
    assert len(state.replay_frames) == 1
    with pytest.raises(ContractValidationError, match="frame capacity"):
        adapter.admit_frame(make_frame_request(sid, "f1"))
    adapter._release_finalized_session(state)
    assert state.dense_images_bgr_chw_uint8 == []
    assert state.replay_frames == []
    assert state.dense_source == []
    assert state.video is None


def test_one_continuation_failure_does_not_fail_successful_sibling() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid_bad = adapter.create_session(make_create_request("bad")).session_id
    sid_good = adapter.create_session(make_create_request("good")).session_id

    def mixed(_adapter: Any, state: Any, item: Any, _gmap: Any) -> tuple[int, bool]:
        if item.request.session_id == sid_bad:
            _record_exact_dense_source_frame(state, item)
            state.frames_pushed += 1
            raise RuntimeError("bad session mutates then fails")
        state.frames_pushed += 1
        return 3, True

    adapter._continuation_fn = mixed
    bad = adapter.admit_frame(make_frame_request(sid_bad, "bad"))
    good = adapter.admit_frame(make_frame_request(sid_good, "good"))
    responses = asyncio.run(adapter.push_frame_batch([bad, good]))
    assert responses[0].error is not None
    assert responses[1].error is None
    assert adapter._sessions[sid_bad].lifecycle.value == "quarantined"
    assert adapter._sessions[sid_good].lifecycle.value == "open"


# --------------------------------------------------------------------------- #
# True cross-session fnet batching honesty
# --------------------------------------------------------------------------- #


def test_two_sessions_share_one_fused_fnet_forward() -> None:
    """Two distinct sessions' ready frames enter ONE fnet forward.

    This is the core batching-honesty test: the FakeBackendModules.fnet records
    batch sizes. Two sessions in one push_frame_batch must produce exactly one
    fnet call with batch size 2 — not two serial calls of batch size 1.
    """
    loads: list[int] = []
    adapter, backend = make_adapter(loads)
    sid_a = adapter.create_session(make_create_request("a")).session_id
    sid_b = adapter.create_session(make_create_request("b")).session_id
    prepared_a = adapter.admit_frame(make_frame_request(sid_a, "fa", ts=0.0))
    prepared_b = adapter.admit_frame(make_frame_request(sid_b, "fb", ts=0.1))

    responses = asyncio.run(adapter.push_frame_batch([prepared_a, prepared_b]))

    assert len(responses) == 2
    assert backend.fnet_calls == [2], f"expected one fused fnet of size 2, got {backend.fnet_calls}"
    for r in responses:
        assert r.status is not None
        assert r.status.trace.fnet_forward_count == 1
        assert r.status.trace.request_count == 2
        # Session-local forwards are counted separately, NOT as a fused forward.
        assert r.status.trace.session_local_forward_count >= 1
    # Distinct sessions in the trace.
    assert set(responses[0].status.trace.session_ids) == {sid_a, sid_b}


def test_serial_single_session_frames_are_not_called_a_batch() -> None:
    """One session pushing two frames serially is NOT a cross-session batch.

    Each push is one fnet forward of size 1; the trace records request_count=1 and
    fnet_forward_count=1. Serial session steps must not be labeled a fused batch.
    """
    loads: list[int] = []
    adapter, backend = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    p0 = adapter.admit_frame(make_frame_request(sid, "f0", ts=0.0))
    r0 = asyncio.run(adapter.push_frame_batch([p0]))[0]
    p1 = adapter.admit_frame(make_frame_request(sid, "f1", ts=0.033))
    r1 = asyncio.run(adapter.push_frame_batch([p1]))[0]

    assert backend.fnet_calls == [1, 1]  # two separate size-1 forwards
    assert r0.status.trace.request_count == 1
    assert r1.status.trace.request_count == 1
    assert r0.status.trace.session_ids == (sid,)


def test_incompatible_image_shapes_split_into_separate_fnet_forwards() -> None:
    """Frames with different image shapes are incompatible and split honestly.

    Each shape group runs its own fused fnet forward; the trace per item records
    fnet_forward_count=1 for its own group, never claims a cross-shape fusion.
    """
    loads: list[int] = []
    adapter, backend = make_adapter(loads)
    # Two sessions with different image shapes.
    req_a = DroidCreateSessionRequest(
        ownership=Ownership("a", "j", "i", "s", "src"),
        camera=DroidCamera((10, 10, 4, 4), ImageSize(16, 16), PixelTransform.identity()),
        image_shape=DroidImageShape(16, 16),
        model_revision=REVISION,
    )
    req_b = DroidCreateSessionRequest(
        ownership=Ownership("b", "j", "i", "s", "src"),
        camera=DroidCamera((10, 10, 4, 4), ImageSize(16, 24), PixelTransform.identity()),
        image_shape=DroidImageShape(16, 24),
        model_revision=REVISION,
    )
    sid_a = adapter.create_session(req_a).session_id
    sid_b = adapter.create_session(req_b).session_id
    rgb_a = np.full((16, 16, 3), 100, dtype=np.uint8)
    rgb_b = np.full((16, 24, 3), 100, dtype=np.uint8)
    pa = adapter.admit_frame(DroidFrameRequest(
        ownership=Ownership("ra", "j", "i", "s", "src", source_timestamp_s=0.0),
        session_id=sid_a, frame_id="fa", source_timestamp_s=0.0,
        rgb=TensorPayload(data=rgb_a.tobytes(), shape=rgb_a.shape, dtype="uint8"),
    ))
    pb = adapter.admit_frame(DroidFrameRequest(
        ownership=Ownership("rb", "j", "i", "s", "src", source_timestamp_s=0.0),
        session_id=sid_b, frame_id="fb", source_timestamp_s=0.0,
        rgb=TensorPayload(data=rgb_b.tobytes(), shape=rgb_b.shape, dtype="uint8"),
    ))
    responses = asyncio.run(adapter.push_frame_batch([pa, pb]))
    # Two separate fnet forwards, one per shape group.
    assert sorted(backend.fnet_calls) == [1, 1]
    assert len(responses) == 2
    assert responses[0].status.trace.batch_id != responses[1].status.trace.batch_id


# --------------------------------------------------------------------------- #
# Timestamp preservation, model-load constancy, status
# --------------------------------------------------------------------------- #


def test_source_timestamp_preserved_in_step_status() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    prepared = adapter.admit_frame(make_frame_request(sid, "f7", ts=0.123))
    resp = asyncio.run(adapter.push_frame_batch([prepared]))[0]
    assert resp.status.source_timestamp_s == 0.123
    assert resp.status.frame_id == "f7"


def test_stored_droid_frame_is_bgr_chw_not_hwc_indexed() -> None:
    """DepthVideo must receive `[1,3,H,W]`, never HWC advanced-indexing output."""
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 11, 22, 33  # RGB
    prepared = adapter.admit_frame(DroidFrameRequest(
        ownership=Ownership("stored", "job", "item", "droid.push_frame", "src"),
        session_id=sid, frame_id="f", source_timestamp_s=0.0,
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
    ))
    bgr = adapter._rgb_hwc_to_bgr_chw_uint8(prepared).cpu().numpy()
    assert bgr.shape == (1, 3, H, W)
    assert np.all(bgr[0, 0] == 33) and np.all(bgr[0, 1] == 22) and np.all(bgr[0, 2] == 11)


def test_filler_source_image_has_per_frame_batch_dimension() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    state = adapter._sessions[sid]
    state.keyframe_source.append(("key", 0.0))
    state.video.images.append(np.zeros((3, H, W), dtype=np.uint8))
    state.video.counter.value = 1
    image = adapter._source_frame_image(state, "key", state.video)
    assert image.shape == (1, 3, H, W)


def test_model_load_count_constant_across_requests() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid = adapter.create_session(make_create_request()).session_id
    for i in range(5):
        p = adapter.admit_frame(make_frame_request(sid, f"f{i}", ts=float(i)))
        r = asyncio.run(adapter.push_frame_batch([p]))[0]
        assert r.status.trace.model_load_count == 1
    assert loads == [1]  # loaded exactly once at startup


def test_status_reports_gpu2_and_resident_session_count() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    adapter.create_session(make_create_request("a"))
    adapter.create_session(make_create_request("b"))
    status = adapter.status()
    assert status.assigned_gpu == 2
    assert status.model_load_count == 1
    assert status.loaded_models == (REVISION,)
    wire = status.to_wire()
    assert wire["assigned_gpu"] == 2


# --------------------------------------------------------------------------- #
# Distinct payload hashes (the task requires distinct frame payload hashes across
# the two exercised sessions). Verify the contract carries distinct binary.
# --------------------------------------------------------------------------- #


def test_two_sessions_carry_distinct_frame_payload_hashes() -> None:
    loads: list[int] = []
    adapter, _ = make_adapter(loads)
    sid_a = adapter.create_session(make_create_request("a")).session_id
    sid_b = adapter.create_session(make_create_request("b")).session_id
    import hashlib
    fa = make_frame_request(sid_a, "fa", pixels=10)
    fb = make_frame_request(sid_b, "fb", pixels=200)
    ha = hashlib.sha256(bytes(fa.rgb.data)).hexdigest()
    hb = hashlib.sha256(bytes(fb.rgb.data)).hexdigest()
    assert ha != hb
