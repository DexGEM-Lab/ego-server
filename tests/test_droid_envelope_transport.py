"""CPU-only dual-wire regressions for DROID's stateful session boundary."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from ego_annotation.serving.binary_envelope import (
    CONTENT_TYPE,
    BinaryEnvelopeError,
    binary_envelope_iovecs,
    build_binary_envelope,
    parse_binary_envelope_body,
)
from ego_annotation.serving.contracts import (
    CameraState,
    ContractValidationError,
    DenseSourceMapping,
    DroidBatchTrace,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    DroidImageShape,
    DroidUncertainty,
    FrameValidity,
    ImageSize,
    KeyframeSourceMapping,
    Ownership,
    PixelTransform,
    ServerIdentity,
    StepStatus,
    TensorPayload,
)
from ego_annotation.serving.droid import build_droid_model_config, expected_droid_runtime_config
from ego_annotation.serving.droid_transport import (
    droid_create_session_gateway_request,
    droid_finalize_gateway_request,
    droid_push_frame_gateway_request,
)
from ego_annotation.serving.gateway import ModelServiceGateway, _parse_generic_envelope
from ego_annotation.serving.router import ModelServiceRouter
from ego_annotation.serving.transport import (
    build_multipart_response,
    parse_droid_finalize_response,
    parse_multipart_response,
)


REVISION = "droid-v1"
SESSION = "session-envelope"


def _deployment_module(monkeypatch):
    ray = types.ModuleType("ray")
    serve = types.ModuleType("ray.serve")

    def deployment(**_kwargs):
        def decorate(cls):
            cls.bind = classmethod(lambda _cls: object())
            return cls
        return decorate

    def batch(**_kwargs):
        return lambda fn: fn

    serve.deployment = deployment
    serve.batch = batch
    ray.serve = serve
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.serve", serve)
    sys.modules.pop("ego_annotation.serving.droid_deployment", None)
    return importlib.import_module("ego_annotation.serving.droid_deployment")


def _ownership(stage: str, request_id: str) -> Ownership:
    return Ownership(request_id, "job-envelope", "item-envelope", stage, "source-envelope", source_timestamp_s=1.25)


def _camera() -> DroidCamera:
    return DroidCamera(
        intrinsics=(400.0, 401.0, 12.0, 8.0),
        source_size=ImageSize(24, 16),
        pixel_transform=PixelTransform.identity(),
        K_px=((400.0, 0.0, 12.0), (0.0, 401.0, 8.0), (0.0, 0.0, 1.0)),
    )


def _requests() -> tuple[DroidCreateSessionRequest, DroidFrameRequest, DroidFinalizeRequest]:
    create = DroidCreateSessionRequest(
        ownership=_ownership("droid.create_session", "create-envelope"),
        camera=_camera(), image_shape=DroidImageShape(16, 24), model_revision=REVISION,
    )
    rgb = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    mask = np.linspace(0.0, 1.0, 16 * 24, dtype=np.float32).reshape(16, 24)
    depth = np.full((16, 24), 1.5, dtype=np.float32)
    push = DroidFrameRequest(
        ownership=_ownership("droid.push_frame", "push-envelope"), session_id=SESSION,
        frame_id="frame-envelope", source_timestamp_s=1.25,
        rgb=TensorPayload(rgb.tobytes(), rgb.shape, "uint8"),
        static_confidence_mask=TensorPayload(mask.tobytes(), mask.shape, "float32"),
        depth_m=TensorPayload(depth.tobytes(), depth.shape, "float32"), model_revision=REVISION,
    )
    finalize = DroidFinalizeRequest(
        ownership=_ownership("droid.finalize", "finalize-envelope"), session_id=SESSION, model_revision=REVISION,
    )
    return create, push, finalize


def _identity() -> ServerIdentity:
    return ServerIdentity(
        "experiment-envelope", "droid-gpu2", 2, 1234, "127.0.0.1:10001", 28002,
        "/tmp/experiment-envelope", REVISION, "checkpoint", "ego.model-service.v1",
        "source", "release", "GPU-2", "/release",
    )


def _payload(array: np.ndarray) -> TensorPayload:
    array = np.ascontiguousarray(array)
    return TensorPayload(array.tobytes(), tuple(array.shape), array.dtype.name)


def _camera_state(request: DroidFinalizeRequest) -> CameraState:
    world_camera = np.eye(4, dtype=np.float64)[None]
    world_camera[0, :3, 3] = (1.0, -0.5, 3.0)
    trace = DroidBatchTrace(
        batch_id="finalize-envelope-batch", replica_id="droid-gpu2", admitted_monotonic_s=1.0,
        dispatched_monotonic_s=1.1, fnet_forward_started_monotonic_s=1.1,
        fnet_completed_monotonic_s=1.1, completed_monotonic_s=1.2, fnet_forward_count=0,
        session_local_forward_count=1, request_count=1, effective_work_units=1,
        model_load_count=1, session_ids=(SESSION,),
    )
    return CameraState(
        ownership=request.ownership, session_id=SESSION,
        T_world_camera=_payload(world_camera), T_camera_world=_payload(np.linalg.inv(world_camera)),
        intrinsics_px=_payload(np.array([[[400.0, 0.0, 12.0], [0.0, 401.0, 8.0], [0.0, 0.0, 1.0]]])),
        disparities=_payload(np.ones((1, 2, 3), dtype=np.float32)),
        keyframe_mapping=(KeyframeSourceMapping(0, "frame-envelope", 1.25),),
        dense_mapping=(DenseSourceMapping(0, "frame-envelope", 1.25),),
        uncertainty=DroidUncertainty("up_to_scale", None, 1.0, 1.0), model_revision=REVISION, trace=trace,
        batch_diagnostics={"runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:droid-wire"},
    )


def _push_response(request: DroidFrameRequest) -> DroidFrameResponse:
    trace = DroidBatchTrace(
        batch_id="push-envelope-batch", replica_id="droid-gpu2", admitted_monotonic_s=1.0,
        dispatched_monotonic_s=1.1, fnet_forward_started_monotonic_s=1.1,
        fnet_completed_monotonic_s=1.2, completed_monotonic_s=1.3, fnet_forward_count=1,
        session_local_forward_count=1, request_count=1, effective_work_units=1,
        model_load_count=1, session_ids=(SESSION,),
    )
    status = StepStatus(
        request.ownership, SESSION, request.frame_id, request.source_timestamp_s,
        FrameValidity(request.frame_id, request.source_timestamp_s, True, True), 1, trace,
    )
    return DroidFrameResponse(request.ownership, status=status, server_identity=_identity(), batch_diagnostics={
        "runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:droid-wire",
    })


async def _request_response(handler, path: str, body: bytes, content_type: str):
    from starlette.requests import Request

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": path, "query_string": b"",
         "headers": [(b"content-type", content_type.encode("ascii"))], "client": ("127.0.0.1", 1),
         "server": ("127.0.0.1", 2), "scheme": "http"}, receive,
    )
    return await handler(request)


async def _asgi_bytes(response):
    messages = []
    never_disconnect = asyncio.Event()

    async def receive():
        await never_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await response({"type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "path": "/"}, receive, send)
    return b"".join(bytes(message.get("body", b"")) for message in messages), messages


class _NoNetwork:
    async def post(self, *_args, **_kwargs):
        raise AssertionError("transport construction only")


def test_droid_gateway_envelope_maps_session_lifecycle_and_all_frame_parts():
    create, push, finalize = _requests()
    requests = (
        (droid_create_session_gateway_request(create), set()),
        (droid_push_frame_gateway_request(push), {"rgb", "static_confidence_mask", "depth_m"}),
        (droid_finalize_gateway_request(finalize), set()),
    )
    gateway = ModelServiceGateway(ModelServiceRouter.canonical(), _NoNetwork(), wire_format="envelope")
    for generic, expected_parts in requests:
        body, content_type = gateway._build_body(generic)
        assert content_type == CONTENT_TYPE
        metadata, parts = _parse_generic_envelope(b"".join(body.iovecs))
        assert set(parts) == expected_parts
        assert metadata["ownership"] == generic.ownership.to_wire()
        assert metadata["model_revision"] == REVISION
    push_generic = droid_push_frame_gateway_request(push)
    assert {part.name: (part.shape, part.dtype) for part in push_generic.parts} == {
        "rgb": ((16, 24, 3), "uint8"),
        "static_confidence_mask": ((16, 24), "float32"),
        "depth_m": ((16, 24), "float32"),
    }


def test_droid_dual_wire_equivalence_preserves_typed_metadata_identity_trace_and_array_hashes(monkeypatch):
    module = _deployment_module(monkeypatch)
    create, push, finalize = _requests()
    deployment = object.__new__(module.DroidDeployment)
    identity = _identity()
    expected_create = DroidCreateSessionResponse(create.ownership, session_id=SESSION, server_identity=identity, batch_diagnostics={
        "runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:droid-wire",
    })
    expected_push = _push_response(push)
    expected_finalize = DroidFinalizeResponse(finalize.ownership, camera_state=_camera_state(finalize), server_identity=identity, batch_diagnostics={
        "runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:droid-wire",
    })

    async def create_session(parsed):
        assert parsed == create
        return expected_create

    async def push_frame(parsed):
        assert parsed == push
        return expected_push

    async def finalize_session(parsed):
        assert parsed == finalize
        return expected_finalize

    deployment.create_session = create_session
    deployment.push_frame = push_frame
    deployment.finalize = finalize_session

    gateway_multipart = ModelServiceGateway(ModelServiceRouter.canonical(), _NoNetwork(), wire_format="multipart")
    gateway_envelope = ModelServiceGateway(ModelServiceRouter.canonical(), _NoNetwork(), wire_format="envelope")
    cases = (
        ("/droid.create_session", droid_create_session_gateway_request(create), DroidCreateSessionResponse, expected_create),
        ("/droid.push_frame", droid_push_frame_gateway_request(push), DroidFrameResponse, expected_push),
        ("/droid.finalize", droid_finalize_gateway_request(finalize), DroidFinalizeResponse, expected_finalize),
    )
    for path, generic, response_cls, expected in cases:
        multipart_body, multipart_type = gateway_multipart._build_body(generic)
        envelope_body, envelope_type = gateway_envelope._build_body(generic)
        multipart_response = asyncio.run(_request_response(deployment, path, multipart_body, multipart_type))
        envelope_response = asyncio.run(_request_response(deployment, path, b"".join(envelope_body.iovecs), envelope_type))
        envelope_bytes, asgi_messages = asyncio.run(_asgi_bytes(envelope_response))
        assert envelope_response.headers["content-type"] == CONTENT_TYPE
        assert all(isinstance(message.get("body", b""), bytes) for message in asgi_messages if message["type"] == "http.response.body")

        if path.endswith("finalize"):
            multipart_metadata, multipart_arrays = parse_multipart_response(multipart_response.body, multipart_response.headers["content-type"])
            envelope_metadata, envelope_arrays = _parse_generic_envelope(envelope_bytes)
            multipart_typed = parse_droid_finalize_response(multipart_response.body, multipart_response.headers["content-type"])
            envelope_typed = parse_droid_finalize_response(envelope_bytes, envelope_response.headers["content-type"])
            assert set(multipart_arrays) == set(envelope_arrays) == {"T_world_camera", "T_camera_world", "intrinsics_px", "disparities"}
            for name, (data, shape, dtype) in multipart_arrays.items():
                envelope_data, envelope_shape, envelope_dtype = envelope_arrays[name]
                assert (envelope_shape, envelope_dtype) == (shape, dtype)
                assert hashlib.sha256(envelope_data).digest() == hashlib.sha256(data).digest()
        else:
            multipart_metadata = json.loads(multipart_response.body)
            envelope_metadata, envelope_arrays = _parse_generic_envelope(envelope_bytes)
            assert not envelope_arrays
            multipart_typed = response_cls.from_wire(multipart_metadata)
            envelope_typed = response_cls.from_wire(envelope_metadata)

        assert envelope_metadata == multipart_metadata
        assert multipart_typed == envelope_typed == expected
        assert envelope_metadata["ownership"] == expected.ownership.to_wire()
        if path.endswith("push_frame"):
            assert envelope_metadata["status"]["trace"] == multipart_metadata["status"]["trace"]
        if path.endswith("finalize"):
            assert envelope_metadata["camera_state"]["trace"] == multipart_metadata["camera_state"]["trace"]
            assert envelope_metadata["server_identity"] == identity.to_wire()


@pytest.mark.parametrize(
    ("body", "message"),
    [(b"EGO", "truncated"), (b"\x00\x00\x00\x0cNOPE" + b"\x00" * 8, "magic")],
)
def test_droid_envelope_codec_rejects_truncated_or_wrong_magic(body, message):
    with pytest.raises(BinaryEnvelopeError, match=message):
        parse_binary_envelope_body(body)


def test_droid_envelope_codec_rejects_oversized_declared_part():
    envelope = build_binary_envelope({"metadata": (b"{}", (), "application/json")})
    body = bytearray(b"".join(binary_envelope_iovecs(envelope)))
    payload_length_offset = 4 + 12 + 6
    body[payload_length_offset:payload_length_offset + 8] = (2**31 + 1).to_bytes(8, "big")
    with pytest.raises(BinaryEnvelopeError, match="exceeds"):
        parse_binary_envelope_body(body)


def test_droid_finalize_parser_preserves_exact_four_part_finite_inverse_contract():
    _create, _push, finalize = _requests()
    state = _camera_state(finalize)
    metadata = {"ownership": finalize.ownership.to_wire(), "camera_state": state.to_wire(), "error": None}
    arrays = {
        name: (bytes(getattr(state, name).data), getattr(state, name).shape, getattr(state, name).dtype)
        for name in ("T_world_camera", "T_camera_world", "intrinsics_px", "disparities")
    }
    for name, (_data, shape, dtype) in arrays.items():
        metadata["camera_state"][name] = {"part": name, "shape": list(shape), "dtype": dtype}
    arrays["extra"] = (b"\x00", (1,), "uint8")
    body, content_type = build_multipart_response(metadata, arrays)
    with pytest.raises(ContractValidationError, match="exactly"):
        parse_droid_finalize_response(body, content_type)


@pytest.mark.parametrize("body", [b"EGO", b"\x00\x00\x00\x0cNOPE" + b"\x00" * 8])
def test_droid_deployment_returns_typed_envelope_error_for_malformed_request(monkeypatch, body):
    module = _deployment_module(monkeypatch)
    deployment = object.__new__(module.DroidDeployment)
    deployment.adapter = SimpleNamespace(server_identity=None)
    response = asyncio.run(_request_response(deployment, "/droid.create_session", body, CONTENT_TYPE))
    encoded, _messages = asyncio.run(_asgi_bytes(response))
    metadata, arrays = _parse_generic_envelope(encoded)
    assert response.status_code == 400
    assert arrays == {}
    assert metadata["error"]["code"] == "validation"


def test_droid_runtime_config_digest_binds_wire_format():
    multipart = build_droid_model_config(weights="weights", model_revision=REVISION, wire_format="multipart")
    envelope = build_droid_model_config(weights="weights", model_revision=REVISION, wire_format="envelope")
    assert envelope.runtime_config_wire()["wire_format"] == "envelope"
    assert multipart.runtime_config_digest() != envelope.runtime_config_digest()
    assert expected_droid_runtime_config(wire_format="envelope")["runtime_config_digest"] == envelope.runtime_config_digest()


def test_droid_instrumented_deployment_attests_resident_wire_config(monkeypatch):
    module = _deployment_module(monkeypatch)
    config = build_droid_model_config(
        weights="weights", model_revision=REVISION, wire_format="envelope", performance_instrumentation=True,
    )
    deployment = object.__new__(module.DroidDeployment)
    deployment.adapter = SimpleNamespace(config=config)
    diagnostics = deployment._runtime_diagnostics()
    expected = expected_droid_runtime_config(wire_format="envelope")
    assert diagnostics["runtime_config"] == expected["runtime_config"]
    assert diagnostics["runtime_config_digest"] == expected["runtime_config_digest"]
    # Instrumented treatments also carry worker allocator state for the sweep.
    allocator = diagnostics.get("allocator_memory")
    if allocator is not None:
        assert set(allocator) == {"allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes"}


def test_droid_status_uses_binary_envelope_when_requested(monkeypatch):
    module = _deployment_module(monkeypatch)
    deployment = object.__new__(module.DroidDeployment)
    deployment.adapter = SimpleNamespace(status=lambda: SimpleNamespace(to_wire=lambda: {"replica_id": "droid-gpu2"}))
    request = build_binary_envelope({"metadata": (b"{}", (), "application/json")})
    response = asyncio.run(_request_response(deployment, "/status", b"".join(binary_envelope_iovecs(request)), CONTENT_TYPE))
    body, _messages = asyncio.run(_asgi_bytes(response))
    metadata, arrays = _parse_generic_envelope(body)
    assert metadata == {"status": {"replica_id": "droid-gpu2"}}
    assert arrays == {}
