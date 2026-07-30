"""Lifecycle regressions for the vendored single-request DROID endpoint.

The tests load the deployment file from ``deploy/releases/droid`` directly so a
passing result proves the code that the A800 launcher imports, not the similarly
named top-level development copy.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ego_annotation.serving.contracts import (
    CameraState,
    DenseSourceMapping,
    DroidBatchTrace,
    DroidCreateSessionResponse,
    DroidFinalizeResponse,
    DroidFrameResponse,
    DroidUncertainty,
    KeyframeSourceMapping,
    Ownership,
    TensorPayload,
)
from ego_annotation.serving.transport import build_multipart_request_fields, parse_droid_finalize_response


RELEASE_DEPLOYMENT = Path(__file__).parents[1] / "deploy/releases/droid/ego_annotation/serving/droid_deployment.py"
REVISION = "droid-v1"


def _module() -> Any:
    """Import the deployment through its vendored package root, never root source."""
    release_root = str(RELEASE_DEPLOYMENT.parents[2])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)
    for name in tuple(sys.modules):
        if name == "ego_annotation" or name.startswith("ego_annotation."):
            sys.modules.pop(name)

    class Serve:
        @staticmethod
        def deployment(**_kwargs: Any) -> Any:
            def decorate(cls: type) -> type:
                cls.bind = classmethod(lambda _cls: object())
                return cls
            return decorate

        @staticmethod
        def batch(**_kwargs: Any) -> Any:
            return lambda fn: fn

    ray = types.ModuleType("ray")
    ray.serve = Serve()
    prior_ray = sys.modules.get("ray")
    sys.modules["ray"] = ray
    try:
        return importlib.import_module("ego_annotation.serving.droid_deployment")
    finally:
        if prior_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = prior_ray


def _owner(request_id: str = "infer-request") -> Ownership:
    return Ownership(request_id, "job", "item", "droid.infer", "video")


def _payload(array: np.ndarray) -> TensorPayload:
    array = np.ascontiguousarray(array)
    return TensorPayload(array.tobytes(), tuple(array.shape), array.dtype.name)


def _camera_state(owner: Ownership, session_id: str) -> CameraState:
    world = np.eye(4, dtype=np.float64)[None]
    trace = DroidBatchTrace(
        batch_id="infer-finalize", replica_id="fake", admitted_monotonic_s=1.0,
        dispatched_monotonic_s=1.0, fnet_forward_started_monotonic_s=1.0,
        fnet_completed_monotonic_s=1.0, completed_monotonic_s=1.0,
        fnet_forward_count=0, session_local_forward_count=1, request_count=1,
        effective_work_units=1, model_load_count=1, session_ids=(session_id,),
    )
    return CameraState(
        ownership=owner, session_id=session_id,
        T_world_camera=_payload(world), T_camera_world=_payload(world),
        intrinsics_px=_payload(np.eye(3, dtype=np.float64)[None]),
        disparities=_payload(np.ones((1, 2, 3), dtype=np.float32)),
        keyframe_mapping=(KeyframeSourceMapping(0, "frame-0", 0.0),),
        dense_mapping=(DenseSourceMapping(0, "frame-0", 0.0),),
        uncertainty=DroidUncertainty("up_to_scale", None, 1.0, 1.0),
        model_revision=REVISION, trace=trace,
    )


def _request_body(owner: Ownership, count: int = 2) -> tuple[bytes, str]:
    metadata = {
        "ownership": owner.to_wire(),
        "camera": {
            "intrinsics": [400.0, 400.0, 12.0, 8.0],
            "source_size": {"width": 24, "height": 16},
            "pixel_transform": {
                "source_to_model": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "model_to_source": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "resize_mode": "identity",
            },
            "K_px": None,
        },
        "image_shape": {"height": 16, "width": 24},
        "options": {"buffer": 256, "warmup": 4},
        "model_revision": REVISION,
        "frames": [{"frame_id": f"frame-{index}", "source_timestamp_s": float(index)} for index in range(count)],
    }
    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    arrays = {f"rgb_{index}": (rgb.tobytes(), rgb.shape, "uint8") for index in range(count)}
    return build_multipart_request_fields(metadata, arrays)


class FakeAdapter:
    server_identity = None

    def __init__(self) -> None:
        self.released: list[str] = []

    async def release_session(self, session_id: str) -> bool:
        self.released.append(session_id)
        return True


def _deployment(module: Any, adapter: FakeAdapter) -> Any:
    deployment = object.__new__(module.DroidDeployment)
    deployment.adapter = adapter
    return deployment


def test_infer_returns_camera_state_and_releases_vendored_session() -> None:
    module = _module()
    adapter = FakeAdapter()
    deployment = _deployment(module, adapter)
    grouped: list[list[Any]] = []

    async def create(request: Any) -> Any:
        return DroidCreateSessionResponse(request.ownership, session_id="session-ok")

    async def infer_frames(frames: list[Any]) -> list[Any]:
        grouped.append(frames)
        return [SimpleNamespace(error=None) for _ in frames]

    async def finalize(request: Any) -> Any:
        return DroidFinalizeResponse(request.ownership, camera_state=_camera_state(request.ownership, request.session_id), terminal=True)

    deployment.create_session = create
    adapter.infer_frames = infer_frames
    deployment.finalize = finalize
    owner = _owner()
    body, content_type = _request_body(owner)
    response = asyncio.run(deployment._handle_infer(body, content_type))
    typed = parse_droid_finalize_response(response.body, response.headers["content-type"])

    assert response.status_code == 200
    assert typed.camera_state is not None
    assert typed.ownership.to_wire() == owner.to_wire()
    assert typed.camera_state.ownership.to_wire() == owner.to_wire()
    assert [request.ownership.request_id for request in grouped[0]] == [
        "infer-request:infer:push:0", "infer-request:infer:push:1",
    ]
    assert adapter.released == ["session-ok"]


def test_infer_mid_lifecycle_exception_still_releases_session() -> None:
    module = _module()
    adapter = FakeAdapter()
    deployment = _deployment(module, adapter)

    async def create(request: Any) -> Any:
        return DroidCreateSessionResponse(request.ownership, session_id="session-error")

    async def infer_frames(_frames: list[Any]) -> list[Any]:
        raise RuntimeError("synthetic continuation failure")

    deployment.create_session = create
    adapter.infer_frames = infer_frames
    body, content_type = _request_body(_owner())
    response = asyncio.run(deployment._handle_infer(body, content_type))
    wire = json.loads(response.body)
    assert response.status_code == 503
    assert wire["error"]["code"] == "model_failure"
    assert adapter.released == ["session-error"]


def test_cancelled_infer_still_releases_session() -> None:
    module = _module()
    adapter = FakeAdapter()
    deployment = _deployment(module, adapter)
    entered = asyncio.Event()

    async def create(request: Any) -> Any:
        return DroidCreateSessionResponse(request.ownership, session_id="session-cancel")

    async def infer_frames(_frames: list[Any]) -> list[Any]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    deployment.create_session = create
    adapter.infer_frames = infer_frames
    body, content_type = _request_body(_owner())

    async def scenario() -> None:
        task = asyncio.create_task(deployment._handle_infer(body, content_type))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert adapter.released == ["session-cancel"]


def test_single_push_runtime_configuration_defaults_to_one_session(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.delenv("EGO_DROID_MAX_SESSIONS", raising=False)
    config = module._config_from_env()
    assert config.max_sessions == 1
    assert config.max_fnet_batch_size == 256
    assert config.max_buffer_slots == 257  # public buffer plus predictive seed
    assert config.cpu_offload is False
    assert config.max_concurrent_ba == 1

    monkeypatch.setenv("EGO_DROID_MAX_SESSIONS", "2")
    assert module._config_from_env().max_sessions == 2


def test_concurrent_infers_have_independent_sessions_and_releases() -> None:
    module = _module()
    adapter = FakeAdapter()
    deployment = _deployment(module, adapter)
    created = 0

    async def create(request: Any) -> Any:
        nonlocal created
        created += 1
        return DroidCreateSessionResponse(request.ownership, session_id=f"session-{created}")

    async def infer_frames(frames: list[Any]) -> list[Any]:
        await asyncio.sleep(0)
        return [SimpleNamespace(error=None) for _ in frames]

    async def finalize(request: Any) -> Any:
        return DroidFinalizeResponse(request.ownership, camera_state=_camera_state(request.ownership, request.session_id), terminal=True)

    deployment.create_session = create
    adapter.infer_frames = infer_frames
    deployment.finalize = finalize
    first = _request_body(_owner("infer-a"))
    second = _request_body(_owner("infer-b"))

    async def scenario() -> list[Any]:
        return await asyncio.gather(
            deployment._handle_infer(*first),
            deployment._handle_infer(*second),
        )

    responses = asyncio.run(scenario())
    assert all(response.status_code == 200 for response in responses)
    assert set(adapter.released) == {"session-1", "session-2"}
