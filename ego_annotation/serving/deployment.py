"""Ray Serve deployment-only module for the resident UniDepth model API.

This module imports Ray Serve at the top level and is therefore a **deployment-only
import path**: ordinary adapter/contract unit tests never import it. ``serve run`` and
``serve deploy`` resolve ``ego_annotation.serving.deployment:app`` against the GPU0
cluster's ``ray_serve_unidepth`` interpreter (Python 3.11, Torch 2.7.1+cu126, Ray 2.55.1).

Design invariants:

* One Ray cluster per physical GPU group (see ``lifecycle.py``). The GPU0 cluster
  starts with ``CUDA_VISIBLE_DEVICES=0`` and advertises one native Ray GPU
  (``--num-gpus=1``), so this deployment requests ``num_gpus=1`` and Ray owns the
  physical GPU.
* Native weighted Serve batching: ``@serve.batch(batch_size_fn=canonical_batch_size_fn)``
  with one canonical HxW compatibility bucket. Each admitted request is one normalized
  work unit, so one Serve callback executes exactly one upstream forward.
* Incompatible/overweight items are rejected at admission (before the callback) via
  ``UniDepthAdapter.admit``, so a callback never has to split into several forwards.
* Requests normally arrive as multipart binary HTTP (``transport.py``); the
  experimental envelope content type is dual-accepted. Both reconstruct the same
  contract request before admission, and each response uses its request's framing.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ray import serve
from ray.serve.exceptions import BackPressureError
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ego_annotation.serving.batching import BatchPolicy, canonical_batch_size_fn
from ego_annotation.serving.queue_budget import queued_request_capacity
from ego_annotation.serving.binary_envelope import (
    CONTENT_TYPE as BINARY_ENVELOPE_CONTENT_TYPE,
    BinaryEnvelope,
    binary_envelope_iovecs,
    build_binary_envelope,
    content_type_is_binary_envelope,
    parse_binary_envelope_body,
    read_binary_envelope_stream,
)
from ego_annotation.serving.contracts import (
    ContractValidationError,
    ErrorCode,
    Ownership,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
)
from ego_annotation.serving.transport import (
    lazy_resolve_object_ref,
    multipart_asgi_response,
    parse_multipart_request,
)
from ego_annotation.serving.unidepth import (
    UniDepthAdapter,
    UniDepthModelConfig,
    _default_tensor_resolver,
    build_unidepth_model_config,
)


def _experiment_int_env(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


def _config_from_env() -> UniDepthModelConfig:
    batch_cap = _experiment_int_env("EGO_UNIDEPTH_EXPERIMENT_BATCH_CAP", 8)
    batch_wait_ms = _experiment_int_env("EGO_UNIDEPTH_EXPERIMENT_BATCH_WAIT_MS", 20)
    # The experiment launcher supplies this key explicitly.  Its absence retains
    # production's existing unrestricted physical-forward behavior.
    concurrency_raw = os.environ.get("EGO_UNIDEPTH_EXPERIMENT_MAX_CONCURRENT_FORWARDS")
    max_concurrent_forwards = _experiment_int_env("EGO_UNIDEPTH_EXPERIMENT_MAX_CONCURRENT_FORWARDS") if concurrency_raw else None
    return build_unidepth_model_config(
        checkpoint=os.environ.get(
            "EGO_UNIDEPTH_CHECKPOINT",
            "/home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected",
        ),
        model_revision=os.environ.get("EGO_UNIDEPTH_REVISION", "unidepth-v2-vitl14-corrected"),
        replica_id=os.environ.get("EGO_UNIDEPTH_REPLICA_ID", "unidepth-gpu0"),
        canonical_height=int(os.environ.get("EGO_UNIDEPTH_CANONICAL_H", "540")),
        canonical_width=int(os.environ.get("EGO_UNIDEPTH_CANONICAL_W", "960")),
        assigned_gpu=int(os.environ.get("EGO_UNIDEPTH_GPU", "0")),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_sha=os.environ.get("EGO_APPLICATION_RELEASE_SHA"),
        checkpoint_digest=os.environ.get("EGO_UNIDEPTH_CHECKPOINT_DIGEST"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if os.environ.get("EGO_EXPERIMENT_HTTP_PORT") else None,
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
        performance_instrumentation=os.environ.get("EGO_UNIDEPTH_EXPERIMENT_TELEMETRY", "0") == "1",
        batch_policy=BatchPolicy(max_batch_size=batch_cap, batch_wait_timeout_s=batch_wait_ms / 1_000.0, max_queued_requests=64),
        max_concurrent_forwards=max_concurrent_forwards,
        wire_format=os.environ.get("EGO_UNIDEPTH_EXPERIMENT_WIRE_FORMAT", "multipart"),
    )


def _in_cluster_tensor_resolver(data: Any, shape: tuple[int, ...], dtype: str) -> Any:
    """Resolve nested Ray ``ObjectRef`` chains then decode the tensor.

    In-cluster callers may pass a ``rgb.data`` that is an ``ObjectRef`` (possibly
    pointing at another ``ObjectRef``) instead of raw bytes. This resolver unwraps
    such chains via ``ray.get`` (imported lazily so the module stays importable in
    unit tests without Ray) and then hands the resolved bytes/array to the standard
    byte-and-shape decoder. HTTP callers already supply raw bytes, which pass through
    ``lazy_resolve_object_ref`` unchanged.
    """
    import ray  # runs only inside a Ray Serve replica

    resolved = lazy_resolve_object_ref(data, ray.get)
    return _default_tensor_resolver(resolved, shape, dtype)


@serve.deployment(
    name="unidepth.infer",
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=16,
    max_queued_requests=queued_request_capacity("unidepth.infer"),
)
class UniDepthDeployment:
    def __init__(self) -> None:
        # The in-cluster resolver unwraps nested Ray ObjectRef chains in rgb.data
        # before byte decoding; HTTP-supplied raw bytes pass through unchanged.
        config = _config_from_env()
        self.adapter = UniDepthAdapter(config, tensor_resolver=_in_cluster_tensor_resolver)
        # Ray Serve owns the batch queue.  Set its live policy on the decorated
        # callable from the resident config so the response attestation describes
        # the actual cap/wait rather than merely launcher environment.
        self._batched_infer.set_max_batch_size(config.batch_policy.max_batch_size)
        self._batched_infer.set_batch_wait_timeout_s(config.batch_policy.batch_wait_timeout_s)

    @serve.batch(
        max_batch_size=8,
        batch_wait_timeout_s=0.02,
        batch_size_fn=canonical_batch_size_fn,
    )
    async def _batched_infer(self, requests: list[Any]) -> list[UniDepthResponse]:
        # Exactly one upstream forward per callback: every item was admitted (decoded
        # and validated) against the single canonical HxW bucket before reaching here.
        # ``requests`` are ``_PreparedRequest`` objects produced by ``adapter.admit``.
        return await self.adapter.infer_batch(requests)

    async def infer(self, request: UniDepthRequest) -> UniDepthResponse:
        try:
            prepared = self.adapter.admit(request)
        except ContractValidationError as exc:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
            )
        try:
            batched_infer: Any = self._batched_infer
            # Pass the prepared (decoded) request so the batch callback forwards the
            # already-decoded uint8 HWC array; admission happened exactly once here.
            return await batched_infer(prepared)
        except BackPressureError as exc:
            # Item-scoped backpressure: the replica's queued-request bound was hit.
            self.adapter.request_dispatched(prepared.request.ownership.request_id)
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.BACKPRESSURE,
                    f"Ray Serve rejected the request: {exc}",
                    retryable=True,
                    ownership=request.ownership,
                ),
            )

    async def __call__(self, request: Request) -> Response:
        """HTTP entry point: emit a real ASGI multipart response.

        Returning a ``starlette.responses.Response`` makes Ray Serve send the raw
        multipart bytes with the declared ``Content-Type``. Returning a dict would
        make Serve JSON-serialize it, destroying the binary multipart contract.
        """
        try:
            content_type = request.headers.get("Content-Type", "multipart/form-data")
            envelope_wire = content_type_is_binary_envelope(content_type)
            if envelope_wire:
                # Fill header and each declared tensor independently from ASGI chunks.
                metadata, rgb_bytes, rgb_shape, rgb_dtype = _parse_envelope_request(
                    await read_binary_envelope_stream(request.stream())
                )
            else:
                # Keep the established multipart branch byte-for-byte unchanged.
                body = await request.body()
                metadata, rgb_bytes, rgb_shape, rgb_dtype = parse_multipart_request(body, content_type)
            ownership = Ownership.from_mapping(metadata["ownership"])
            parsed = UniDepthRequest(
                ownership=ownership,
                rgb=TensorPayload(data=rgb_bytes, shape=rgb_shape, dtype=rgb_dtype),
                spatial=SpatialMetadata.from_mapping(metadata["spatial"]),
                model_revision=metadata["model_revision"],
                options=tuple(sorted((str(k), str(v)) for k, v in metadata.get("options", {}).items())),
            )
        except (ContractValidationError, ValueError, KeyError) as exc:
            return _error_response(_transport_error(exc), envelope_wire if "envelope_wire" in locals() else False)
        response = await self.infer(parsed)
        return _response_to_envelope_response(response) if envelope_wire else _response_to_multipart_response(response)


def _transport_error(exc: Exception) -> UniDepthResponse:
    ownership = Ownership(
        request_id="transport",
        job_id="unknown",
        item_id="unknown",
        stage_id="unidepth.infer",
        source_id="unknown",
    )
    return UniDepthResponse(
        ownership=ownership,
        error=ServiceError(ErrorCode.TRANSPORT, str(exc), retryable=False, ownership=ownership),
    )


def _parse_envelope_request(
    body: bytes | bytearray | memoryview | BinaryEnvelope,
) -> tuple[dict[str, Any], memoryview, tuple[int, ...], str]:
    """Decode the UniDepth envelope into the exact multipart request tuple."""
    envelope = body if isinstance(body, BinaryEnvelope) else parse_binary_envelope_body(body)
    parts = {part.name: part for part in envelope.parts}
    metadata_part = parts.get("metadata")
    rgb_part = parts.get("rgb")
    if metadata_part is None or metadata_part.dtype != "application/json" or metadata_part.shape:
        raise ValueError("binary envelope missing valid metadata part")
    if rgb_part is None:
        raise ValueError("binary envelope missing 'rgb' part")
    try:
        metadata = json.loads(metadata_part.data.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"binary envelope metadata is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("binary envelope metadata must be an object")
    return metadata, rgb_part.data, rgb_part.shape, rgb_part.dtype


def _envelope_asgi_response(
    metadata: dict[str, Any], arrays: dict[str, tuple[bytes | bytearray | memoryview, tuple[int, ...], str]],
) -> Response:
    """Stream header and tensor vectors as separate ASGI writes, never joining them."""
    parts: dict[str, tuple[bytes | bytearray | memoryview, tuple[int, ...], str]] = {
        "metadata": (json.dumps(metadata, separators=(",", ":")).encode("utf-8"), (), "application/json"),
    }
    parts.update(arrays)
    envelope = build_binary_envelope(parts)
    # Starlette's StreamingResponse only accepts ``bytes`` chunks in its ASGI send
    # loop; yielding our memoryview iovecs directly causes it to call ``.encode``
    # on a memoryview and close the HTTP body after headers.  Preserve framing as
    # independent chunks while converting each part at the ASGI boundary.
    return StreamingResponse((bytes(chunk) for chunk in binary_envelope_iovecs(envelope)), media_type=BINARY_ENVELOPE_CONTENT_TYPE)


def _error_response(response: UniDepthResponse, envelope_wire: bool = False) -> Response:
    metadata = {"error": response.error.to_wire(), "ownership": response.ownership.to_wire()}
    if envelope_wire:
        return _envelope_asgi_response(metadata, {})
    return multipart_asgi_response(metadata, {})


def _response_to_envelope_response(response: UniDepthResponse) -> Response:
    """Render the same response metadata and arrays through vectored envelope IO."""
    if response.error is not None:
        return _error_response(response, envelope_wire=True)
    result = response.result
    assert result is not None
    arrays = {
        "depth_m": (result.depth_m.data, result.depth_m.shape, result.depth_m.dtype),
        "K_px": (result.K_px.data, result.K_px.shape, result.K_px.dtype),
        "confidence": (result.confidence.data, result.confidence.shape, result.confidence.dtype),
    }
    metadata = {
        "result": result.to_wire(include_tensor_data=False),
        "ownership": response.ownership.to_wire(),
    }
    return _envelope_asgi_response(metadata, arrays)


def _response_to_multipart_response(response: UniDepthResponse) -> Response:
    """Render a response as an actual multipart ASGI ``Response``.

    On error the body carries only an error metadata part; on success it carries the
    result metadata plus the ``depth_m``/``K_px``/``confidence`` binary array parts.
    The client reconstructs either via ``parse_multipart_response``.
    """
    if response.error is not None:
        return _error_response(response)
    result = response.result
    assert result is not None
    arrays = {
        "depth_m": (bytes(result.depth_m.data), result.depth_m.shape, result.depth_m.dtype),
        "K_px": (bytes(result.K_px.data), result.K_px.shape, result.K_px.dtype),
        "confidence": (bytes(result.confidence.data), result.confidence.shape, result.confidence.dtype),
    }
    metadata = {
        # Dense arrays are already binary multipart fields.  Omitting data_b64 from
        # metadata prevents a second copy and keeps HTTP timing/accounting honest.
        "result": result.to_wire(include_tensor_data=False),
        "ownership": response.ownership.to_wire(),
    }
    return multipart_asgi_response(metadata, arrays)


# A bound Ray Serve Application. ``serve run ego_annotation.serving.deployment:app``
# deploys this against the GPU0 cluster.
app: Any = UniDepthDeployment.bind()
