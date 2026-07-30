"""Async clients for HTTP and in-cluster Ray Serve calls without an import-time Ray dependency.

The HTTP client uses the multipart binary transport (``transport.py``) as the primary
UniDepth transport: dense arrays travel as binary fields, never JSON/base64. The
in-cluster client accepts Ray object-backed tensors and resolves nested ``ObjectRef``
values lazily via an injected ``ray_get``.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Protocol

from ego_annotation.serving.contracts import (
    Cosmos3Request,
    Cosmos3Response,
    Cosmos3Result,
    ErrorCode,
    Ownership,
    ServiceError,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
    UniDepthResult,
    BatchTrace,
    SpatialMetadata,
)
from ego_annotation.serving.transport import (
    build_cosmos3_request,
    build_multipart_request,
    lazy_resolve_object_ref,
    parse_cosmos3_response,
    parse_multipart_response,
)


class AsyncHttpTransport(Protocol):
    async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> Any: ...


class RemoteMethod(Protocol):
    def remote(self, request: UniDepthRequest) -> Any: ...


class InClusterHandle(Protocol):
    infer: RemoteMethod


def _default_ray_get() -> Callable[[Any], Any]:
    import ray  # imported lazily so the module imports without Ray

    return ray.get


def _request_to_multipart(request: UniDepthRequest) -> tuple[bytes, str]:
    if not isinstance(request.rgb.data, (bytes, bytearray, memoryview)):
        raise ValueError("HTTP transport requires RGB data as binary bytes, not an in-cluster object reference")
    rgb_bytes = bytes(request.rgb.data)
    metadata = {
        "ownership": request.ownership.to_wire(),
        "spatial": request.spatial.to_wire(),
        "model_revision": request.model_revision,
        "options": dict(request.options),
        "rgb_shape": list(request.rgb.shape),
        "rgb_dtype": request.rgb.dtype,
    }
    return build_multipart_request(metadata, rgb=rgb_bytes, rgb_shape=request.rgb.shape, rgb_dtype=request.rgb.dtype)


def _result_from_multipart(metadata: dict[str, Any], arrays: dict[str, tuple[bytes, tuple[int, ...], str]],
                            ownership: Ownership) -> UniDepthResponse:
    if metadata.get("error"):
        return UniDepthResponse(
            ownership=ownership,
            error=ServiceError.from_wire(metadata["error"]),
        )
    result_meta = metadata["result"]
    depth_bytes, depth_shape, depth_dtype = arrays["depth_m"]
    k_bytes, k_shape, k_dtype = arrays["K_px"]
    conf_bytes, conf_shape, conf_dtype = arrays["confidence"]
    return UniDepthResponse(
        ownership=ownership,
        result=UniDepthResult(
            ownership=ownership,
            depth_m=TensorPayload(data=depth_bytes, shape=depth_shape, dtype=depth_dtype),
            K_px=TensorPayload(data=k_bytes, shape=k_shape, dtype=k_dtype),
            confidence=TensorPayload(data=conf_bytes, shape=conf_shape, dtype=conf_dtype),
            spatial=SpatialMetadata.from_mapping(result_meta["spatial"]),
            model_revision=result_meta["model_revision"],
            trace=BatchTrace.from_wire(result_meta["trace"]),
        ),
    )


class HttpModelServiceClient:
    """HTTP boundary client using multipart binary transport for dense arrays."""

    def __init__(self, base_url: str, transport: AsyncHttpTransport) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    @classmethod
    def with_httpx(cls, base_url: str, *, timeout_s: float = 30.0) -> "HttpModelServiceClient":
        """Create a convenient HTTP client while keeping test imports transport-free."""
        import httpx

        class HttpxAdapter:
            def __init__(self, client: Any) -> None:
                self._client = client

            async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> Any:
                return await self._client.post(url, content=content, headers=headers)

            async def aclose(self) -> None:
                await self._client.aclose()

        return cls(base_url, HttpxAdapter(httpx.AsyncClient(timeout=timeout_s)))

    async def aclose(self) -> None:
        close = getattr(self._transport, "aclose", None)
        if close is not None:
            value = close()
            if inspect.isawaitable(value):
                await value

    async def infer_unidepth(self, request: UniDepthRequest) -> UniDepthResponse:
        try:
            body, content_type = _request_to_multipart(request)
            response = await self._transport.post(
                f"{self._base_url}/unidepth.infer",
                content=body,
                headers={"Content-Type": content_type},
            )
        except Exception as exc:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, str(exc), retryable=True, ownership=request.ownership),
            )
        status_code = int(getattr(response, "status_code", 200))
        if status_code in {429, 503}:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.BACKPRESSURE,
                    "Ray Serve queue is full; retry with bounded backoff",
                    retryable=True,
                    ownership=request.ownership,
                ),
            )
        if status_code >= 400:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, f"HTTP {status_code}", retryable=status_code >= 500, ownership=request.ownership),
            )
        try:
            content = getattr(response, "content", None)
            if content is None:
                content = await response.read()
            resp_content_type = response.headers.get("Content-Type", "multipart/form-data")
            metadata, arrays = parse_multipart_response(content, resp_content_type)
            return _result_from_multipart(metadata, arrays, request.ownership)
        except Exception as exc:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, f"invalid service response: {exc}", retryable=True, ownership=request.ownership),
            )

    async def reason_cosmos3(self, request: Cosmos3Request) -> Cosmos3Response:
        """Send a model-native multimodal reasoning request with bounded binary media."""
        try:
            media_specs = [
                (bytes(m.data), m.kind, m.media_type, m.source_index)
                for m in request.media
                if isinstance(m.data, (bytes, bytearray, memoryview))
            ]
            metadata = {
                "ownership": request.ownership.to_wire(),
                "prompt": request.prompt,
                "messages": [{"role": role, "content": content} for role, content in request.messages],
                "generation": request.generation.to_wire(),
            }
            body, content_type = build_cosmos3_request(metadata, media_specs)
            response = await self._transport.post(
                f"{self._base_url}/cosmos3.reason",
                content=body,
                headers={"Content-Type": content_type},
            )
        except Exception as exc:
            return Cosmos3Response(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, str(exc), retryable=True, ownership=request.ownership),
            )
        status_code = int(getattr(response, "status_code", 200))
        if status_code in {429, 503}:
            return Cosmos3Response(
                ownership=request.ownership,
                error=ServiceError(
                    ErrorCode.BACKPRESSURE,
                    "Ray Serve queue is full; retry with bounded backoff",
                    retryable=True,
                    ownership=request.ownership,
                ),
            )
        if status_code >= 400:
            return Cosmos3Response(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, f"HTTP {status_code}", retryable=status_code >= 500, ownership=request.ownership),
            )
        try:
            content = getattr(response, "content", None)
            if content is None:
                content = await response.read()
            resp_content_type = response.headers.get("Content-Type", "multipart/form-data")
            wire = parse_cosmos3_response(content, resp_content_type)
            return Cosmos3Response.from_wire(wire)
        except Exception as exc:
            return Cosmos3Response(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, f"invalid service response: {exc}", retryable=True, ownership=request.ownership),
            )


class InClusterModelServiceClient:
    """Uses an injected deployment handle and accepts Ray object-backed tensors.

    Nested ``ObjectRef`` values in ``rgb.data`` are resolved lazily via an injected
    ``ray_get`` (default ``ray.get``) before the request is sent to the handle.
    """

    def __init__(self, handle: InClusterHandle, *, ray_get: Callable[[Any], Any] | None = None) -> None:
        self._handle = handle
        self._ray_get = ray_get

    def _resolve(self, value: Any) -> Any:
        ray_get = self._ray_get or _default_ray_get()
        return lazy_resolve_object_ref(value, ray_get)

    async def infer_unidepth(self, request: UniDepthRequest) -> UniDepthResponse:
        try:
            resolved_data = self._resolve(request.rgb.data)
            if isinstance(resolved_data, (bytes, bytearray, memoryview)):
                resolved_request = UniDepthRequest(
                    ownership=request.ownership,
                    rgb=TensorPayload(data=bytes(resolved_data), shape=request.rgb.shape, dtype=request.rgb.dtype),
                    spatial=request.spatial,
                    model_revision=request.model_revision,
                    options=request.options,
                )
            else:
                # Already a resolved in-memory array; keep it for the adapter resolver.
                resolved_request = UniDepthRequest(
                    ownership=request.ownership,
                    rgb=TensorPayload(data=resolved_data, shape=request.rgb.shape, dtype=request.rgb.dtype),
                    spatial=request.spatial,
                    model_revision=request.model_revision,
                    options=request.options,
                )
            response = await _resolve_remote(self._handle.infer.remote(resolved_request))
            if isinstance(response, UniDepthResponse):
                return response
            if isinstance(response, dict):
                return UniDepthResponse.from_wire(response)
            raise TypeError("Serve handle returned an unsupported response type")
        except Exception as exc:
            return UniDepthResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.TRANSPORT, str(exc), retryable=True, ownership=request.ownership),
            )


async def _resolve_remote(value: Any) -> Any:
    """Await a Serve DeploymentResponse without importing its Ray-specific type."""
    if inspect.isawaitable(value):
        return await value
    result = getattr(value, "result", None)
    if callable(result):
        resolved = result()
        if inspect.isawaitable(resolved):
            return await resolved
        return resolved
    return value
