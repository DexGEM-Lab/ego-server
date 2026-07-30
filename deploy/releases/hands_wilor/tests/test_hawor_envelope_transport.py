"""CPU-only dual-wire boundary tests for the GPU3 HaWoR track lane."""
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
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
)
from ego_annotation.serving.gateway import ModelServiceGateway
from ego_annotation.serving.hawor_contracts import (
    HAWOR_CHUNK_LEN,
    HAWOR_CROP_H,
    HAWOR_CROP_W,
    CameraSpaceManoResult,
    CropSourceTransform,
    DroidCameraEvidence,
    FrameObservation,
    HandSide,
    OcclusionState,
    TrackChunkRequest,
    UniDepthScaleK,
)
from ego_annotation.serving.hawor import expected_hawor_runtime_config
from ego_annotation.serving.infiller import expected_infiller_runtime_config
from ego_annotation.serving.hawor_transport import track_chunk_gateway_request
from ego_annotation.serving.router import ModelServiceRouter
from ego_annotation.serving.transport import parse_multipart_response


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
    sys.modules.pop("ego_annotation.serving.hawor_deployment", None)
    return importlib.import_module("ego_annotation.serving.hawor_deployment")


def _request() -> TrackChunkRequest:
    ownership = Ownership("hawor-envelope-request", "job-a", "track-a", "hawor.infer_tracks", "source-a", source_timestamp_s=1.0)
    source = ImageSize(width=1920, height=1080)
    transform = CropSourceTransform(
        center=(960.0, 540.0), scale=2.0, img_focal=1200.0, img_center=(960.0, 540.0), do_flip=False,
        source_size=source, pixel_transform=PixelTransform.identity(),
    )
    observations = tuple(
        FrameObservation(i, 1.0 + i / 30.0, OcclusionState.OCCLUDED if i == 3 else OcclusionState.VISIBLE, 0.8, HandSide.RIGHT)
        for i in range(HAWOR_CHUNK_LEN)
    )
    poses = np.tile(np.eye(4, dtype=np.float32), (HAWOR_CHUNK_LEN, 1, 1))
    timestamps = np.arange(HAWOR_CHUNK_LEN, dtype=np.float64) / 30.0 + 1.0
    droid = DroidCameraEvidence(
        TensorPayload(poses.tobytes(), poses.shape, "float32"), TensorPayload(timestamps.tobytes(), timestamps.shape, "float64"),
        1.0, 0.01, 0.9, "droid+unidepth_scale",
    )
    return TrackChunkRequest(
        ownership=ownership, track_id="track-a", side=HandSide.RIGHT,
        crop_batch=TensorPayload(np.arange(HAWOR_CHUNK_LEN * 3 * HAWOR_CROP_H * HAWOR_CROP_W, dtype=np.float32).tobytes(), (HAWOR_CHUNK_LEN, 3, HAWOR_CROP_H, HAWOR_CROP_W), "float32"),
        crop_transforms=(transform,) * HAWOR_CHUNK_LEN, observations=observations,
        unidepth=UniDepthScaleK(((1200.0, 0.0, 960.0), (0.0, 1200.0, 540.0), (0.0, 0.0, 1.0)), 1200.0, (960.0, 540.0), source, 1.0, "unidepth"),
        droid_evidence=droid, model_revision="hawor-v1",
    )


def _result(request: TrackChunkRequest) -> CameraSpaceManoResult:
    def tensor(shape, dtype="float32"):
        return TensorPayload(np.arange(np.prod(shape), dtype=np.dtype(dtype)).tobytes(), tuple(shape), dtype)

    spatial = SpatialMetadata(request.crop_transforms[0].source_size, ImageSize(width=256, height=256), "RGB", PixelTransform.identity(), request.unidepth.K_px)
    trace = BatchTrace("hawor-envelope-batch", "hawor-gpu3", 1.0, 1.1, 1.2, 1.3, 1, 1, 1, 1)
    return CameraSpaceManoResult(
        request.ownership, request.track_id, request.side,
        tensor((16, 3, 3)), tensor((16, 15, 3, 3)), tensor((16, 3)), tensor((16, 10)),
        tensor((16, 778, 3)), tensor((16, 16, 3)), TensorPayload(bytes([1] * 16), (16,), "bool"),
        tuple(o.occlusion_state for o in request.observations), tensor((16,)), tensor((16, 4, 4)),
        "resampled_droid_world_from_camera", spatial, "hawor-v1", trace,
        {"runtime_config": {"wire_format": "envelope"}, "runtime_config_digest": "sha256:hawor-wire"},
    )


async def _request_response(deployment, body, content_type):
    from starlette.requests import Request

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "path": "/hawor.infer_tracks", "query_string": b"",
         "headers": [(b"content-type", content_type.encode("ascii"))], "client": ("127.0.0.1", 1),
         "server": ("127.0.0.1", 2), "scheme": "http"}, receive,
    )
    return await deployment(request)


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


def test_gpu3_runtime_config_digest_attributes_wire_treatment():
    hawor_multipart = expected_hawor_runtime_config(wire_format="multipart")
    hawor_envelope = expected_hawor_runtime_config(wire_format="envelope")
    infiller_multipart = expected_infiller_runtime_config(wire_format="multipart")
    infiller_envelope = expected_infiller_runtime_config(wire_format="envelope")
    assert hawor_envelope["runtime_config"]["wire_format"] == "envelope"
    assert infiller_envelope["runtime_config"]["wire_format"] == "envelope"
    assert hawor_multipart["runtime_config_digest"] != hawor_envelope["runtime_config_digest"]
    assert infiller_multipart["runtime_config_digest"] != infiller_envelope["runtime_config_digest"]


def test_hawor_gateway_mapping_preserves_track_masks_and_droid_array_shapes():
    request = _request()
    generic = track_chunk_gateway_request(request)
    assert {part.name: (part.shape, part.dtype) for part in generic.parts} == {
        "crop_batch": ((16, 3, 256, 256), "float32"),
        "droid_poses": ((16, 4, 4), "float32"),
        "droid_timestamps": ((16,), "float64"),
    }
    assert generic.metadata["track_id"] == request.track_id
    assert generic.metadata["observations"][3]["occlusion_state"] == "occluded"


def test_hawor_dual_wire_typed_result_metadata_hashes_and_asgi_bytes(monkeypatch):
    module = _deployment_module(monkeypatch)
    request = _request()

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("body-build only")

    multipart_gateway = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="multipart")
    envelope_gateway = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="envelope")
    multipart_body, multipart_type = multipart_gateway._build_body(track_chunk_gateway_request(request))
    envelope_body, envelope_type = envelope_gateway._build_body(track_chunk_gateway_request(request))
    assert isinstance(multipart_body, bytes)

    deployment = object.__new__(module.HaWoRDeployment)
    result = _result(request)

    async def infer(parsed):
        assert parsed == request
        return result, None

    deployment.infer = infer
    multipart_response = asyncio.run(_request_response(deployment, multipart_body, multipart_type))
    envelope_response = asyncio.run(_request_response(deployment, b"".join(envelope_body.iovecs), envelope_type))
    multipart_meta, multipart_arrays = parse_multipart_response(multipart_response.body, multipart_response.headers["content-type"])
    envelope_bytes, asgi_messages = asyncio.run(_asgi_bytes(envelope_response))
    envelope = parse_binary_envelope_body(envelope_bytes)
    envelope_parts = {part.name: part for part in envelope.parts}
    envelope_meta = __import__("json").loads(envelope_parts.pop("metadata").data.tobytes())

    assert envelope_response.headers["content-type"] == CONTENT_TYPE
    body_messages = [message for message in asgi_messages if message["type"] == "http.response.body"]
    assert all(isinstance(message.get("body", b""), bytes) for message in body_messages)
    assert envelope_meta == multipart_meta
    assert envelope_meta["ownership"] == request.ownership.to_wire()
    assert envelope_meta["result"]["trace"] == multipart_meta["result"]["trace"]
    assert envelope_meta["result"]["batch_diagnostics"] == multipart_meta["result"]["batch_diagnostics"]
    assert set(envelope_parts) == set(multipart_arrays) == {"root_orient", "hand_pose", "trans", "betas", "vertices", "joints", "observed", "uncertainty", "world_lift"}
    for name, (data, shape, dtype) in multipart_arrays.items():
        part = envelope_parts[name]
        assert (part.shape, part.dtype) == (shape, dtype)
        assert hashlib.sha256(part.data).digest() == hashlib.sha256(data).digest()


@pytest.mark.parametrize("body, message", [
    (b"EGO", "truncated"),
    (b"\x00\x00\x00\x0cNOPE" + b"\x00" * 8, "magic"),
])
def test_hawor_envelope_codec_rejects_malformed_headers(body, message):
    with pytest.raises(BinaryEnvelopeError, match=message):
        parse_binary_envelope_body(body)


def test_hawor_envelope_codec_rejects_oversized_declared_part():
    envelope = build_binary_envelope({"metadata": (b"{}", (), "application/json")})
    body = bytearray(b"".join(binary_envelope_iovecs(envelope)))
    # HTTP framing: magic(4), version(1), header-length(4), header, then payloads.
    # Change the first part's declared payload length in the compact header to > MAX_PART_BYTES.
    header_len = int.from_bytes(body[:4], "big")
    assert header_len > 0
    header_start = 4
    # Header prelude is magic/count; first part is name_len,dtype_len,rank,payload_len.
    payload_length_offset = header_start + 12 + 6
    body[payload_length_offset:payload_length_offset + 8] = (2**31 + 1).to_bytes(8, "big")
    with pytest.raises(BinaryEnvelopeError, match="exceeds"):
        parse_binary_envelope_body(body)
