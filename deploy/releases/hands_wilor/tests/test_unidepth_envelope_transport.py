"""CPU-only dual-wire UniDepth deployment boundary tests."""
from __future__ import annotations

import asyncio
import importlib
import sys
import types

import numpy as np

from ego_annotation.serving.benchmark.manifest import build_synthetic_unidepth_manifest
from ego_annotation.serving.contracts import (
    BatchTrace,
    SCHEMA_VERSION,
    ServerIdentity,
    TensorPayload,
    UniDepthResponse,
    UniDepthResult,
)
from ego_annotation.serving.gateway import ModelServiceGateway, _parse_generic_envelope
from ego_annotation.serving.router import ModelServiceRouter
from ego_annotation.serving.transport import parse_multipart_response


def _deployment_module(monkeypatch):
    """Import the deployment boundary with a minimal CPU-only Ray decorator shim."""
    ray = types.ModuleType("ray")
    serve = types.ModuleType("ray.serve")
    exceptions = types.ModuleType("ray.serve.exceptions")

    class BackPressureError(Exception):
        pass

    def deployment(**_kwargs):
        def decorate(cls):
            cls.bind = classmethod(lambda _cls: object())
            return cls
        return decorate

    def batch(**_kwargs):
        return lambda fn: fn

    serve.deployment = deployment
    serve.batch = batch
    exceptions.BackPressureError = BackPressureError
    ray.serve = serve
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.serve", serve)
    monkeypatch.setitem(sys.modules, "ray.serve.exceptions", exceptions)
    sys.modules.pop("ego_annotation.serving.deployment", None)
    return importlib.import_module("ego_annotation.serving.deployment")


async def _request_response(deployment, body, content_type):
    from starlette.requests import Request

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({
        "type": "http", "method": "POST", "path": "/unidepth.infer", "query_string": b"",
        "headers": [(b"content-type", content_type.encode("ascii"))], "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 2), "scheme": "http",
    }, receive)
    return await deployment(request)


async def _response_bytes(response) -> bytes:
    if hasattr(response, "body"):
        return response.body
    chunks = [chunk async for chunk in response.body_iterator]
    return b"".join(chunks)


async def _response_bytes_through_asgi(response) -> tuple[bytes, list[dict[str, object]]]:
    """Drive Starlette's ASGI send loop rather than consuming body_iterator directly."""
    sent: list[dict[str, object]] = []
    never_disconnect = asyncio.Event()

    async def receive():
        await never_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await response(
        {"type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "path": "/"},
        receive,
        send,
    )
    return b"".join(bytes(message.get("body", b"")) for message in sent), sent


def test_server_accepts_both_wire_formats_with_identical_typed_result(monkeypatch):
    module = _deployment_module(monkeypatch)
    item = build_synthetic_unidepth_manifest(manifest_id="dual-wire", count=1, height=2, width=2).items[0]
    request = item.to_gateway_request()

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("build only")

    multipart_gateway = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="multipart")
    envelope_gateway = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="envelope")
    multipart_body, multipart_type = multipart_gateway._build_body(request)
    envelope_body, envelope_type = envelope_gateway._build_body(request)
    assert isinstance(multipart_body, bytes)

    ownership = item.ownership
    spatial = item.spatial
    assert spatial is not None
    trace = BatchTrace("batch-1", "unidepth-gpu0", 1.0, 1.1, 1.2, 1.3, 1, 1, 1, 1)
    identity = ServerIdentity(
        experiment_id="envelope-test", replica_id="unidepth-gpu0", assigned_gpu=0, worker_pid=99,
        gcs_address="127.0.0.1:29000", http_port=28000, temp_dir="/tmp/unidepth-envelope",
        model_revision=request.model_revision or "revision", checkpoint_digest="sha256:checkpoint",
        schema_version=SCHEMA_VERSION, release_sha="release", release_digest="release", cuda_uuid="GPU-0",
    )
    result = UniDepthResult(
        ownership,
        TensorPayload(np.arange(4, dtype=np.float32).tobytes(), (2, 2), "float32"),
        TensorPayload(np.eye(3, dtype=np.float32).tobytes(), (3, 3), "float32"),
        TensorPayload(np.ones((2, 2), dtype=np.float32).tobytes(), (2, 2), "float32"),
        spatial,
        request.model_revision or "revision",
        trace,
        batch_diagnostics={
            "allocator_memory": {"allocated_bytes": 1, "reserved_bytes": 2},
            "runtime_config": {"wire_format": "envelope"},
            "runtime_config_digest": "sha256:wire-format-attributed",
        },
        server_identity=identity,
    )
    response = UniDepthResponse(ownership, result=result)
    deployment = object.__new__(module.UniDepthDeployment)

    async def infer(parsed):
        # Both framing paths must reconstruct the exact typed request before inference.
        assert parsed.ownership == ownership
        assert parsed.rgb.shape == request.parts[0].shape
        assert parsed.rgb.dtype == request.parts[0].dtype
        assert bytes(parsed.rgb.data) == bytes(request.parts[0].data)
        return response

    deployment.infer = infer
    multipart_response = asyncio.run(_request_response(deployment, multipart_body, multipart_type))
    envelope_response = asyncio.run(_request_response(deployment, b"".join(envelope_body.iovecs), envelope_type))
    multipart_meta, multipart_arrays = parse_multipart_response(asyncio.run(_response_bytes(multipart_response)), multipart_response.headers["content-type"])
    envelope_bytes, asgi_messages = asyncio.run(_response_bytes_through_asgi(envelope_response))
    envelope_meta, envelope_arrays = _parse_generic_envelope(envelope_bytes)

    assert envelope_response.headers["content-type"] == "application/vnd.ego.binary-envelope"
    body_messages = [message for message in asgi_messages if message["type"] == "http.response.body"]
    assert len(body_messages) == 7  # HTTP prefix, envelope header, metadata, three tensors, then final empty body
    assert all(isinstance(message.get("body", b""), bytes) for message in body_messages)
    assert envelope_meta == multipart_meta
    assert set(envelope_arrays) == set(multipart_arrays) == {"depth_m", "K_px", "confidence"}
    for name, (data, shape, dtype) in multipart_arrays.items():
        envelope_data, envelope_shape, envelope_dtype = envelope_arrays[name]
        assert (envelope_shape, envelope_dtype) == (shape, dtype)
        assert envelope_data.tobytes() == data
    assert envelope_meta["result"]["trace"] == multipart_meta["result"]["trace"]
    assert envelope_meta["result"]["server_identity"] == multipart_meta["result"]["server_identity"]
    assert envelope_meta["result"]["batch_diagnostics"] == multipart_meta["result"]["batch_diagnostics"]
