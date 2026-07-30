"""CPU-only binary-envelope regression tests for GPU1 Hands and WiLoR."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import sys
import types

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
    BatchTrace,
    HandDetection,
    HandSide,
    HandsDetectRequest,
    HandsDetectResponse,
    HandsDetectResult,
    ImageSize,
    ManoOutput,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
    WiLoRReconstructRequest,
    WiLoRReconstructResponse,
    WiLoRReconstructResult,
)
from ego_annotation.serving.gateway import ModelServiceGateway, _parse_generic_envelope
from ego_annotation.serving.hands import build_hands_model_config
from ego_annotation.serving.hands_transport import hands_detect_gateway_request, wilor_reconstruct_gateway_request
from ego_annotation.serving.router import ModelServiceRouter
from ego_annotation.serving.transport import parse_multipart_response
from ego_annotation.serving.wilor import build_wilor_model_config


HANDS_REV = "hands-yolo-sam2.1-hiera-l"
WILOR_REV = "wilor-final-v1"


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

    def ingress(_app):
        return lambda cls: cls

    serve.deployment = deployment
    serve.batch = batch
    serve.ingress = ingress
    ray.serve = serve
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.serve", serve)
    sys.modules.pop("ego_annotation.serving.hands_deployment", None)
    return importlib.import_module("ego_annotation.serving.hands_deployment")


def _hands_request() -> HandsDetectRequest:
    ownership = Ownership("hands-envelope", "job-hands", "frame-10", "hands.detect", "source-hands", source_timestamp_s=1.0)
    spatial = SpatialMetadata(
        ImageSize(12, 8), ImageSize(12, 8), "RGB", PixelTransform.identity(),
        ((500.0, 0.0, 6.0), (0.0, 500.0, 4.0), (0.0, 0.0, 1.0)),
    )
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    return HandsDetectRequest(ownership, TensorPayload(rgb.tobytes(), rgb.shape, "uint8"), spatial, HANDS_REV)


def _wilor_request() -> WiLoRReconstructRequest:
    ownership = Ownership("wilor-envelope", "job-wilor", "hand-10", "wilor.reconstruct", "source-wilor", source_timestamp_s=1.0)
    crop = np.arange(3 * 256 * 256, dtype=np.float32).reshape(3, 256, 256) / 255.0
    return WiLoRReconstructRequest(
        ownership, TensorPayload(crop.tobytes(), crop.shape, "float32"), HandSide.LEFT,
        (100.0, 200.0), 300.0, (1920.0, 1080.0), WILOR_REV,
        ((1200.0, 0.0, 960.0), (0.0, 1200.0, 540.0), (0.0, 0.0, 1.0)),
    )


def _tensor(shape: tuple[int, ...], dtype: str = "float32") -> TensorPayload:
    return TensorPayload(np.arange(np.prod(shape), dtype=np.dtype(dtype)).tobytes(), shape, dtype)


def _hands_response(request: HandsDetectRequest) -> HandsDetectResponse:
    detection = HandDetection(
        _tensor((1, 4)), _tensor((1,)), _tensor((1,), "uint8"), _tensor((1, 8, 12), "uint8"),
        _tensor((1,)), _tensor((1,)), 1,
    )
    trace = BatchTrace("hands-batch", "hands-gpu1", 1.0, 1.1, 1.2, 1.3, 1, 1, 1, 1)
    result = HandsDetectResult(
        request.ownership, detection, request.spatial, HANDS_REV, trace, 1,
        {"runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:hands-wire"},
    )
    return HandsDetectResponse(request.ownership, result=result)


def _wilor_response(request: WiLoRReconstructRequest) -> WiLoRReconstructResponse:
    mano = ManoOutput(
        _tensor((1, 3, 3)), _tensor((15, 3, 3)), _tensor((10,)), _tensor((778, 3)), _tensor((21, 3)),
        _tensor((3,)), _tensor((3,)), _tensor((21, 2)), 5000.0, _tensor((1,)), _tensor((1,)), 778,
    )
    trace = BatchTrace("wilor-batch", "hands-gpu1", 1.0, 1.1, 1.2, 1.3, 1, 1, 1, 1)
    result = WiLoRReconstructResult(
        request.ownership, mano, request.handedness, WILOR_REV, trace,
        {"runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:wilor-wire"},
    )
    return WiLoRReconstructResponse(request.ownership, result=result)


async def _request_response(handler, body, content_type):
    from starlette.requests import Request

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/", "query_string": b"",
         "headers": [(b"content-type", content_type.encode("ascii"))], "client": ("127.0.0.1", 1),
         "server": ("127.0.0.1", 2), "scheme": "http"}, receive,
    )
    return await handler(request)


async def _asgi_bytes(response):
    sent = []
    never_disconnect = asyncio.Event()

    async def receive():
        await never_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await response({"type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "path": "/"}, receive, send)
    return b"".join(bytes(message.get("body", b"")) for message in sent), sent


@pytest.mark.parametrize(
    ("request_factory", "response_factory", "gateway_factory", "handler_name", "parts"),
    [
        (_hands_request, _hands_response, hands_detect_gateway_request, "_handle_hands_detect_http", {"boxes", "scores", "sides", "masks", "visibility", "uncertainty"}),
        (_wilor_request, _wilor_response, wilor_reconstruct_gateway_request, "_handle_wilor_reconstruct_http", {"global_orient", "hand_pose", "betas", "vertices", "joints", "cam_t_full", "pred_cam", "keypoints_2d", "confidence", "uncertainty"}),
    ],
)
def test_gpu1_dual_wire_equivalence_preserves_metadata_ownership_trace_identity_and_array_hashes(
    monkeypatch, request_factory, response_factory, gateway_factory, handler_name, parts,
):
    module = _deployment_module(monkeypatch)
    request = request_factory()

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("transport construction only")

    multipart = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="multipart")
    envelope = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="envelope")
    multipart_body, multipart_type = multipart._build_body(gateway_factory(request))
    envelope_body, envelope_type = envelope._build_body(gateway_factory(request))
    assert isinstance(multipart_body, bytes)
    assert not isinstance(envelope_body, bytes)

    deployment = object.__new__(module.HandsWiLoRDeployment)
    response = response_factory(request)

    async def infer(parsed):
        assert parsed == request
        return response

    setattr(deployment, "detect" if handler_name.endswith("detect_http") else "reconstruct", infer)
    handler = getattr(deployment, handler_name)
    multipart_response = asyncio.run(_request_response(handler, multipart_body, multipart_type))
    envelope_response = asyncio.run(_request_response(handler, b"".join(envelope_body.iovecs), envelope_type))

    multipart_meta, multipart_arrays = parse_multipart_response(multipart_response.body, multipart_response.headers["content-type"])
    envelope_bytes, messages = asyncio.run(_asgi_bytes(envelope_response))
    envelope_meta, envelope_arrays = _parse_generic_envelope(envelope_bytes)

    assert envelope_response.headers["content-type"] == CONTENT_TYPE
    assert all(isinstance(message.get("body", b""), bytes) for message in messages if message["type"] == "http.response.body")
    assert envelope_meta == multipart_meta
    assert envelope_meta["ownership"] == request.ownership.to_wire()
    assert envelope_meta["result"]["trace"] == multipart_meta["result"]["trace"]
    assert envelope_meta["result"]["batch_diagnostics"] == multipart_meta["result"]["batch_diagnostics"]
    assert set(envelope_arrays) == set(multipart_arrays) == parts
    for name, (data, shape, dtype) in multipart_arrays.items():
        envelope_data, envelope_shape, envelope_dtype = envelope_arrays[name]
        assert (envelope_shape, envelope_dtype) == (shape, dtype)
        assert hashlib.sha256(envelope_data).digest() == hashlib.sha256(data).digest()


def test_gpu1_gateway_envelope_mapping_covers_real_hand_and_mano_input_shapes():
    hands = hands_detect_gateway_request(_hands_request())
    wilor = wilor_reconstruct_gateway_request(_wilor_request())
    assert {part.name: (part.shape, part.dtype) for part in hands.parts} == {"rgb": ((8, 12, 3), "uint8")}
    assert {part.name: (part.shape, part.dtype) for part in wilor.parts} == {"crop": ((3, 256, 256), "float32")}
    assert wilor.metadata["handedness"] == 0
    assert wilor.metadata["source_K_px"] == [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]]

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("transport construction only")

    gateway = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="envelope")
    for request, expected_parts in ((hands, {"rgb"}), (wilor, {"crop"})):
        body, _ = gateway._build_body(request)
        assert not isinstance(body, bytes)
        metadata, vectors = _parse_generic_envelope(b"".join(body.iovecs))
        assert set(vectors) == expected_parts
        assert metadata["ownership"] == request.ownership.to_wire()


def test_gpu1_runtime_config_digest_binds_wire_format():
    hands_multipart = build_hands_model_config(
        detector_checkpoint="detector", sam2_checkpoint="sam2", sam2_config="config", model_revision=HANDS_REV, wire_format="multipart",
    )
    hands_envelope = build_hands_model_config(
        detector_checkpoint="detector", sam2_checkpoint="sam2", sam2_config="config", model_revision=HANDS_REV, wire_format="envelope",
    )
    wilor_multipart = build_wilor_model_config(checkpoint="wilor", config_path="config", model_revision=WILOR_REV, wire_format="multipart")
    wilor_envelope = build_wilor_model_config(checkpoint="wilor", config_path="config", model_revision=WILOR_REV, wire_format="envelope")
    assert hands_envelope.runtime_config_wire()["wire_format"] == "envelope"
    assert wilor_envelope.runtime_config_wire()["wire_format"] == "envelope"
    assert hands_multipart.runtime_config_digest() != hands_envelope.runtime_config_digest()
    assert wilor_multipart.runtime_config_digest() != wilor_envelope.runtime_config_digest()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"EGO", "truncated"),
        (b"\x00\x00\x00\x0cNOPE" + b"\x00" * 8, "magic"),
    ],
)
def test_gpu1_envelope_codec_rejects_truncated_or_wrong_magic(body, message):
    with pytest.raises(BinaryEnvelopeError, match=message):
        parse_binary_envelope_body(body)


def test_gpu1_envelope_codec_rejects_oversized_declared_part():
    envelope = build_binary_envelope({"metadata": (b"{}", (), "application/json")})
    body = bytearray(b"".join(binary_envelope_iovecs(envelope)))
    payload_length_offset = 4 + 12 + 6
    body[payload_length_offset:payload_length_offset + 8] = (2**31 + 1).to_bytes(8, "big")
    with pytest.raises(BinaryEnvelopeError, match="exceeds"):
        parse_binary_envelope_body(body)
