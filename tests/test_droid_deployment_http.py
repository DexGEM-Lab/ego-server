"""HTTP-boundary regressions for the deployment-only DROID Serve wrapper.

The module normally imports Ray.  A minimal decorator stub keeps these tests CPU-only
while exercising the actual ASGI response objects that Ray Serve passes through.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np

from ego_annotation.serving.contracts import (
    CameraState,
    DroidBatchTrace,
    ContractValidationError,
    DenseSourceMapping,
    DroidFinalizeResponse,
    DroidUncertainty,
    KeyframeSourceMapping,
    Ownership,
    ServerIdentity,
    TensorPayload,
)
from ego_annotation.serving.transport import build_multipart_request_fields, parse_droid_finalize_response, parse_multipart_response


def _deployment_module() -> Any:
    if "ego_annotation.serving.droid_deployment" in sys.modules:
        return sys.modules["ego_annotation.serving.droid_deployment"]

    class _Serve:
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
    setattr(ray, "serve", _Serve())
    sys.modules["ray"] = ray
    return importlib.import_module("ego_annotation.serving.droid_deployment")


def _ownership() -> Ownership:
    return Ownership("request", "job", "item", "droid.finalize", "source")


def _payload(array: Any) -> TensorPayload:
    contiguous = np.ascontiguousarray(array)
    return TensorPayload(contiguous.tobytes(), tuple(contiguous.shape), contiguous.dtype.name)


def _camera_state() -> CameraState:
    trace = DroidBatchTrace(
        batch_id="batch", replica_id="gpu2", admitted_monotonic_s=1.0,
        dispatched_monotonic_s=1.0, fnet_forward_started_monotonic_s=1.0,
        fnet_completed_monotonic_s=2.0, completed_monotonic_s=2.0,
        fnet_forward_count=0, session_local_forward_count=1, request_count=1,
        effective_work_units=1, model_load_count=1, session_ids=("session",),
    )
    T_world_camera = np.eye(4, dtype=np.float64)[None]
    T_world_camera[0, :3, 3] = (1.25, -0.5, 3.0)
    return CameraState(
        ownership=_ownership(), session_id="session",
        T_world_camera=_payload(T_world_camera),
        T_camera_world=_payload(np.linalg.inv(T_world_camera)),
        intrinsics_px=_payload(np.asarray([[[400.0, 0.0, 12.0], [0.0, 401.0, 8.0], [0.0, 0.0, 1.0]]])),
        disparities=_payload(np.ones((1, 2, 2), dtype=np.float32)),
        keyframe_mapping=(KeyframeSourceMapping(0, "frame-0", 1.5),),
        dense_mapping=(DenseSourceMapping(0, "frame-0", 1.5),),
        uncertainty=DroidUncertainty("up_to_scale", None, 1.0, 1.0),
        model_revision="droid-v1", trace=trace,
    )


def test_cpu_offload_env_is_explicit_opt_in(monkeypatch: Any) -> None:
    module = _deployment_module()
    monkeypatch.delenv("EGO_DROID_CPU_OFFLOAD", raising=False)
    assert module._config_from_env().cpu_offload is False
    monkeypatch.setenv("EGO_DROID_CPU_OFFLOAD", "1")
    assert module._config_from_env().cpu_offload is True


def test_invalid_create_returns_structured_json_error() -> None:
    module = _deployment_module()
    deployment = object.__new__(module.DroidDeployment)
    # The malformed pixel transform used to enter the except path and crash while
    # dereferencing `.ownership` on an Ownership object, yielding an HTTP 500.
    body = b'--x\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{}\r\n--x--\r\n'
    response = asyncio.run(deployment._handle_create(body, "multipart/form-data; boundary=x"))
    assert response.status_code == 400
    wire = json.loads(response.body)
    assert wire["error"]["code"] == "validation"


def test_push_parser_backpressure_error_carries_adapter_derived_identity() -> None:
    module = _deployment_module()
    deployment = object.__new__(module.DroidDeployment)
    identity = ServerIdentity(
        "exp", "replica", 7, 42, "127.0.0.1:34007", 36007, "/tmp/eds/exp/gpu7",
        "droid-v1", "checkpoint", "ego.model-service.v1", "source", "release", "GPU-7", "/release",
    )
    deployment.adapter = SimpleNamespace(server_identity=identity)
    owner = _ownership()
    body, content_type = build_multipart_request_fields({
        "ownership": owner.to_wire(), "session_id": "s", "frame_id": "f", "source_timestamp_s": 0.0,
        "model_revision": "droid-v1",
    }, {})
    response = asyncio.run(deployment._handle_push(body, content_type))
    wire = json.loads(response.body)
    assert response.status_code == 400
    assert wire["error"]["code"] == "validation"
    assert wire["server_identity"] == identity.to_wire()


def test_adapter_admission_and_finalize_validation_errors_keep_worker_identity() -> None:
    module = _deployment_module()
    identity = ServerIdentity(
        "exp", "replica", 7, 42, "127.0.0.1:34007", 36007, "/tmp/eds/exp/gpu7",
        "droid-v1", "checkpoint", "ego.model-service.v1", "source", "release", "GPU-7", "/release",
    )

    class Adapter:
        server_identity = identity
        def admit_frame(self, _request: Any) -> Any:
            raise ContractValidationError("admission rejected")
        async def finalize(self, _request: Any) -> Any:
            raise ContractValidationError("finalize rejected")

    deployment = object.__new__(module.DroidDeployment)
    deployment.adapter = Adapter()
    owner = _ownership()
    push = module.DroidFrameRequest(
        owner, "session", "frame", 0.0, TensorPayload(bytes(8 * 8 * 3), (8, 8, 3), "uint8"), "droid-v1",
    )
    push_response = asyncio.run(deployment.push_frame(push))
    finalize_response = asyncio.run(deployment.finalize(module.DroidFinalizeRequest(owner, "session", "droid-v1")))
    assert push_response.error.code.value == "validation" and push_response.server_identity == identity
    assert finalize_response.error.code.value == "validation" and finalize_response.server_identity == identity

    class BatchedAdapter:
        server_identity = identity
        abandoned = []
        def admit_frame(self, _request: Any) -> Any:
            return object()
        def request_abandoned(self, prepared: Any) -> None:
            self.abandoned.append(prepared)

    batched_deployment = object.__new__(module.DroidDeployment)
    batched_deployment.adapter = BatchedAdapter()
    async def invalid_batch(_prepared: Any) -> Any:
        raise ContractValidationError("batched validation rejected")
    batched_deployment._batched_push_frame = invalid_batch
    batched_response = asyncio.run(batched_deployment.push_frame(push))
    assert batched_response.error.code.value == "validation"
    assert batched_response.server_identity == identity
    assert len(batched_deployment.adapter.abandoned) == 1


def test_finalize_preserves_multipart_binary_response() -> None:
    module = _deployment_module()
    state = _camera_state()
    response = module._finalize_response_to_http_response(
        DroidFinalizeResponse(ownership=state.ownership, camera_state=state)
    )
    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    metadata, arrays = parse_multipart_response(response.body, content_type)
    assert metadata["ownership"] == state.ownership.to_wire()
    assert set(arrays) == {"T_world_camera", "T_camera_world", "intrinsics_px", "disparities"}
    assert arrays["disparities"][1:] == ((1, 2, 2), "float32")
    assert "data_b64" not in metadata["camera_state"]["T_world_camera"]

    typed = parse_droid_finalize_response(response.body, content_type)
    assert typed.ownership == state.ownership
    assert typed.camera_state is not None
    parsed = typed.camera_state
    T_wc = np.frombuffer(parsed.T_world_camera.data, dtype=parsed.T_world_camera.dtype).reshape(parsed.T_world_camera.shape)
    T_cw = np.frombuffer(parsed.T_camera_world.data, dtype=parsed.T_camera_world.dtype).reshape(parsed.T_camera_world.shape)
    np.testing.assert_allclose(T_wc @ T_cw, np.eye(4)[None], atol=1e-9)
    assert parsed.dense_mapping[0].source_timestamp_s == 1.5
    assert parsed.trace.replica_id == "gpu2"


def test_finalize_parser_rejects_descriptor_array_mismatch_and_invalid_inverse() -> None:
    import pytest
    from ego_annotation.serving.transport import build_multipart_response

    state = _camera_state()
    metadata = {"ownership": state.ownership.to_wire(), "camera_state": state.to_wire(), "error": None}
    arrays = {
        name: (bytes(getattr(state, name).data), getattr(state, name).shape, getattr(state, name).dtype)
        for name in ("T_world_camera", "T_camera_world", "intrinsics_px", "disparities")
    }
    for name, (_data, shape, dtype) in arrays.items():
        metadata["camera_state"][name] = {"part": name, "shape": list(shape), "dtype": dtype}
    metadata["camera_state"]["T_world_camera"]["shape"] = [2, 4, 4]
    body, content_type = build_multipart_response(metadata, arrays)
    with pytest.raises(ContractValidationError, match="descriptor shape"):
        parse_droid_finalize_response(body, content_type)

    metadata["camera_state"]["T_world_camera"]["shape"] = [1, 4, 4]
    bad_inverse = np.eye(4, dtype=np.float64)[None]
    arrays["T_camera_world"] = (bad_inverse.tobytes(), bad_inverse.shape, bad_inverse.dtype.name)
    body, content_type = build_multipart_response(metadata, arrays)
    with pytest.raises(ContractValidationError, match="mutual inverses"):
        parse_droid_finalize_response(body, content_type)
