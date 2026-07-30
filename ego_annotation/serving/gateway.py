"""Typed gateway client over the multi-cluster router.

This is the caller-facing boundary. It sits above ``client.py``'s per-API HTTP
helpers and above the router. Responsibilities:

* Accept a *generic* model-native request: a metadata dict (ownership, spatial
  transforms, source timestamps, model revision) plus named binary tensor/media
  parts carried as raw bytes or in-cluster Ray ``ObjectRef`` values.
* Resolve nested ``ObjectRef`` values lazily via an injected ``ray_get`` (default
  ``ray.get``) so in-cluster callers pass object references for dense tensors
  without a copy.
* Build either the production ``multipart/form-data`` body or the explicit
  experimental binary envelope (one JSON ``metadata`` part + named binary vectors),
  post it to the router-resolved Serve endpoint, and parse the matching response
  framing into per-item typed results or typed failures.
* Apply **bounded retries that do not hide overload**: a retry policy retries only
  transient transport errors and explicit backpressure, with a hard cap on attempts
  and a deadline. When the cap is exhausted the gateway returns a typed
  ``BACKPRESSURE``/``TRANSPORT`` failure for that item — it never silently retries
  forever and never converts overload into a slow success. Non-retryable errors
  (validation, model failure) are returned immediately as typed failures.
* Preserve per-item ownership: each request carries its own ``Ownership`` and each
  response is split back to that ownership, even when a batch contains mixed
  job/agent IDs.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Protocol, Sequence

from ego_annotation.serving.binary_envelope import (
    CONTENT_TYPE as BINARY_ENVELOPE_CONTENT_TYPE,
    BinaryEnvelope,
    binary_envelope_iovecs,
    build_binary_envelope,
    content_type_is_binary_envelope,
    parse_binary_envelope_body,
)
from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    ErrorCode,
    Ownership,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    reject_filesystem_fields,
)
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import (
    build_multipart_request,
    lazy_resolve_object_ref,
    parse_multipart_response,
)


class AsyncHttpTransport(Protocol):
    async def post(self, url: str, *, content: Any, headers: dict[str, str]) -> Any: ...


WIRE_FORMATS = frozenset({"multipart", "envelope"})


@dataclass(frozen=True)
class EnvelopeHttpBody:
    """Re-iterable async byte stream over an envelope's unflattened vectors.

    ``httpx`` accepts an async byte stream, so a retry can write the compact
    framing, JSON metadata, and each tensor as independent vectors rather than
    assembling a large aggregate request body. The object is intentionally
    re-iterable because ``RetryPolicy`` can submit the exact same request again.
    """

    envelope: BinaryEnvelope

    @property
    def iovecs(self) -> tuple[memoryview, ...]:
        return binary_envelope_iovecs(self.envelope)

    @property
    def content_length(self) -> int:
        return sum(vector.nbytes for vector in self.iovecs)

    def __aiter__(self) -> AsyncIterator[memoryview]:
        async def _chunks() -> AsyncIterator[memoryview]:
            for vector in self.iovecs:
                yield vector
        return _chunks()


def _validate_wire_format(wire_format: str) -> str:
    if wire_format not in WIRE_FORMATS:
        raise ContractValidationError(f"wire_format must be one of {sorted(WIRE_FORMATS)}")
    return wire_format


def _default_ray_get() -> Callable[[Any], Any]:
    import ray  # imported lazily so the module imports without Ray

    return ray.get


# --- request/response envelopes -----------------------------------------------------


@dataclass(frozen=True)
class GatewayBinaryPart:
    """A named binary tensor/media part of a gateway request.

    ``data`` is raw bytes for HTTP callers or an in-cluster Ray ``ObjectRef`` /
    resolved array for in-cluster callers. ``shape`` and ``dtype`` are required so the
    receiver can reconstruct the array without a separate schema; media parts may
    use dtype ``bytes``/``media`` with a shape of ``(len,)`` or ``()``.
    """

    name: str
    data: Any
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ContractValidationError("binary part name must be a non-empty string")
        if not self.shape:
            raise ContractValidationError(f"binary part {self.name!r} shape must be non-empty")
        if not self.dtype:
            raise ContractValidationError(f"binary part {self.name!r} dtype must be set")
        if isinstance(self.data, str):
            raise ContractValidationError("binary part data must be bytes or an object reference, never a path string")


@dataclass(frozen=True)
class GatewayRequest:
    """A typed model-native request carried as metadata + named binary parts.

    ``ownership`` preserves per-item provenance (request/job/item/stage/source IDs,
    schema version, source timestamp). ``spatial`` carries source image size, model
    image size, color convention, the full resize/crop/pad pixel transform, and
    ``K_px`` where applicable. ``metadata`` carries any additional model-native
    fields (e.g. handedness, crop-to-source transforms, track/hand identity,
    observation masks) that are JSON-serializable.
    """

    api_name: ModelApiName
    ownership: Ownership
    parts: tuple[GatewayBinaryPart, ...]
    spatial: SpatialMetadata | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    model_revision: str | None = None

    def __post_init__(self) -> None:
        # Session lifecycle calls (DROID create/finalize) are typed metadata-only
        # operations. Dense model inputs still use named binary parts when present.
        names = {p.name for p in self.parts}
        if len(names) != len(self.parts):
            raise ContractValidationError("binary part names must be unique within a request")
        reject_filesystem_fields(self.metadata, context="request.metadata")

    @property
    def work_units(self) -> int:
        """Native work weight for offered-load accounting.

        Default: one work unit per request. APIs that define a weighted batch unit
        (pixels, crops, track chunks, frames, temporal windows) override this by
        including an integer ``work_units`` in ``metadata``; the gateway trusts the
        caller-declared weight for load accounting and the serving replica re-validates
        it at admission against its own batch policy.
        """
        declared = self.metadata.get("work_units")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
            return 1
        return declared


@dataclass(frozen=True)
class GatewayResult:
    """A typed successful response for one item."""

    ownership: Ownership
    metadata: Mapping[str, Any]
    arrays: Mapping[str, TensorPayload]
    trace: BatchTrace | None = None


@dataclass(frozen=True)
class GatewayResponse:
    """Exactly one of result/error per item, preserving ownership."""

    ownership: Ownership
    result: GatewayResult | None = None
    error: ServiceError | None = None
    # Client-side observability: attempts made, last HTTP status, transport ms.
    attempts: int = 0
    last_status_code: int | None = None
    transport_ms: float = 0.0
    # Experiment-only multi-endpoint clients attach the selected replica here.
    # Production callers leave it unset; ownership still belongs to the request.
    replica_id: str | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("a gateway response must contain exactly one result or error")


# --- retry policy --------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries that do not hide overload.

    A retry is attempted only for retryable errors (transient transport failures and
    explicit ``BACKPRESSURE``). ``max_attempts`` caps total tries (1 == no retry).
    ``deadline_s`` bounds wall-clock time spent retrying one item. When the cap or
    deadline is hit, the *last* typed error is returned — overload is surfaced, not
    hidden behind infinite retries or a slow success.
    """

    max_attempts: int = 3
    deadline_s: float = 5.0
    initial_backoff_s: float = 0.01
    max_backoff_s: float = 0.2
    backoff_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ContractValidationError("max_attempts must be >= 1")
        if self.deadline_s < 0 or self.initial_backoff_s < 0 or self.max_backoff_s < 0:
            raise ContractValidationError("retry deadlines/backoffs must be non-negative")
        if self.backoff_factor < 1.0:
            raise ContractValidationError("backoff_factor must be >= 1.0")

    def next_backoff(self, attempt: int) -> float:
        """Backoff before attempt ``attempt`` (1-indexed, attempt>=2 only)."""
        if attempt <= 1:
            return 0.0
        delay = self.initial_backoff_s * (self.backoff_factor ** (attempt - 2))
        return min(delay, self.max_backoff_s)


def _is_retryable_status(status_code: int) -> bool:
    # 429 Too Many Requests, 503 Service Unavailable, and network-level 5xx are
    # retryable. 4xx (other than 429) are caller errors and not retried.
    return status_code == 429 or status_code >= 500


# --- gateway -------------------------------------------------------------------------


class ModelServiceGateway:
    """Typed gateway over the multi-cluster router.

    Construct with a router and an async HTTP transport. The HTTP transport is the
    only network boundary; tests inject a deterministic fake. ``ray_get`` is
    injected so in-cluster ``ObjectRef`` resolution is testable without Ray.
    """

    def __init__(
        self,
        router: ModelServiceRouter,
        transport: AsyncHttpTransport,
        *,
        retry_policy: RetryPolicy | None = None,
        ray_get: Callable[[Any], Any] | None = None,
        wire_format: str = "multipart",
    ) -> None:
        self._router = router
        self._transport = transport
        self._retry_policy = retry_policy or RetryPolicy()
        self._ray_get = ray_get
        self._wire_format = _validate_wire_format(wire_format)

    @property
    def wire_format(self) -> str:
        """The explicit wire format for this gateway instance; never auto-fallback."""
        return self._wire_format

    @classmethod
    def with_httpx(
        cls, router: ModelServiceRouter, *, timeout_s: float = 30.0,
        retry_policy: RetryPolicy | None = None, wire_format: str = "multipart",
    ) -> "ModelServiceGateway":
        import httpx

        class HttpxAdapter:
            def __init__(self, client: Any) -> None:
                self._client = client

            async def post(self, url: str, *, content: Any, headers: dict[str, str]) -> Any:
                return await self._client.post(url, content=content, headers=headers)

            async def aclose(self) -> None:
                await self._client.aclose()

        return cls(
            router, HttpxAdapter(httpx.AsyncClient(timeout=timeout_s)),
            retry_policy=retry_policy, wire_format=wire_format,
        )

    async def aclose(self) -> None:
        close = getattr(self._transport, "aclose", None)
        if close is not None:
            value = close()
            if inspect.isawaitable(value):
                await value

    def _resolve_part(self, part: GatewayBinaryPart) -> GatewayBinaryPart:
        """Resolve a possibly ObjectRef-backed part to bytes for HTTP transport.

        In-cluster callers pass ``ObjectRef`` values for dense tensors; HTTP callers
        pass bytes. Resolution repeats until the value is no longer reference-like.
        ``ray_get`` is imported lazily and only when a reference is actually present,
        so HTTP-only callers never import Ray.
        """
        if isinstance(part.data, (bytes, bytearray, memoryview)):
            return GatewayBinaryPart(name=part.name, data=bytes(part.data), shape=part.shape, dtype=part.dtype)
        ray_get = self._ray_get or _default_ray_get()
        resolved = lazy_resolve_object_ref(part.data, ray_get)
        if not isinstance(resolved, (bytes, bytearray, memoryview)):
            raise ContractValidationError(
                f"binary part {part.name!r} resolved to {type(resolved).__name__}; HTTP transport requires bytes"
            )
        return GatewayBinaryPart(name=part.name, data=bytes(resolved), shape=part.shape, dtype=part.dtype)

    def _build_body(self, request: GatewayRequest) -> tuple[bytes | EnvelopeHttpBody, str]:
        metadata: dict[str, Any] = {
            "ownership": request.ownership.to_wire(),
            "api_name": request.api_name.value,
        }
        if request.spatial is not None:
            metadata["spatial"] = request.spatial.to_wire()
        if request.model_revision is not None:
            metadata["model_revision"] = request.model_revision
        extra = dict(request.metadata)
        extra.pop("work_units", None)
        metadata["metadata"] = extra
        metadata["work_units"] = request.work_units
        # Resolve any ObjectRef-backed parts before building the selected body.
        resolved: list[tuple[str, bytes, tuple[int, ...], str]] = []
        for part in request.parts:
            resolved_part = self._resolve_part(part)
            resolved.append((part.name, bytes(resolved_part.data), part.shape, part.dtype))
        # Multipart remains byte-identical to the established production path.
        if self._wire_format == "multipart":
            return _build_generic_multipart(metadata, resolved)
        return _build_generic_envelope(metadata, resolved), BINARY_ENVELOPE_CONTENT_TYPE

    async def call(self, request: GatewayRequest) -> GatewayResponse:
        """Call one API for one item with bounded retries.

        Returns a typed ``GatewayResponse`` preserving the request's ownership. On
        overload exhaustion the response carries a ``BACKPRESSURE`` error with the
        attempt count; overload is never hidden.
        """
        url = self._router.url_for(request.api_name)
        policy = self._retry_policy
        deadline_mono = time.monotonic() + policy.deadline_s
        last_error: ServiceError | None = None
        last_status: int | None = None
        transport_total_ms = 0.0
        attempt = 0
        try:
            body, content_type = self._build_body(request)
        except ContractValidationError as exc:
            return GatewayResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
                attempts=0,
            )
        while attempt < policy.max_attempts:
            attempt += 1
            if attempt > 1:
                backoff = policy.next_backoff(attempt)
                if backoff > 0 and time.monotonic() + backoff > deadline_mono:
                    # No time left for another attempt; surface overload.
                    break
                if backoff > 0:
                    await asyncio.sleep(backoff)
            if time.monotonic() > deadline_mono and last_error is not None:
                break
            t0 = time.monotonic()
            try:
                response = await self._transport.post(
                    url, content=body, headers={"Content-Type": content_type}
                )
            except Exception as exc:
                transport_total_ms += (time.monotonic() - t0) * 1000.0
                last_status = None
                last_error = ServiceError(
                    ErrorCode.TRANSPORT, f"transport error: {exc}", retryable=True, ownership=request.ownership
                )
                # Transient transport errors are retryable up to the cap/deadline.
                if attempt >= policy.max_attempts:
                    break
                continue
            transport_total_ms += (time.monotonic() - t0) * 1000.0
            status_code = int(getattr(response, "status_code", 200))
            last_status = status_code
            # Backpressure / 5xx: retryable, but bounded.
            if status_code == 429 or status_code == 503:
                last_error = ServiceError(
                    ErrorCode.BACKPRESSURE,
                    f"server backpressure (HTTP {status_code})",
                    retryable=True,
                    ownership=request.ownership,
                )
                if attempt >= policy.max_attempts:
                    break
                continue
            if status_code >= 500:
                last_error = ServiceError(
                    ErrorCode.TRANSPORT, f"HTTP {status_code}", retryable=True, ownership=request.ownership
                )
                if attempt >= policy.max_attempts:
                    break
                continue
            if status_code >= 400:
                # Caller error (validation etc.): not retryable. Surface typed error.
                return GatewayResponse(
                    ownership=request.ownership,
                    error=ServiceError(
                        ErrorCode.TRANSPORT, f"HTTP {status_code}", retryable=False, ownership=request.ownership
                    ),
                    attempts=attempt,
                    last_status_code=status_code,
                    transport_ms=transport_total_ms,
                )
            # 2xx: parse the multipart response.
            try:
                content = getattr(response, "content", None)
                if content is None:
                    content = await response.read()
                resp_ct = response.headers.get("Content-Type", "multipart/form-data")
                if self._wire_format == "envelope":
                    if not content_type_is_binary_envelope(resp_ct):
                        raise ValueError(f"expected binary envelope response, got {resp_ct!r}")
                    meta, arrays = _parse_generic_envelope(content)
                else:
                    meta, arrays = parse_multipart_response(content, resp_ct)
            except Exception as exc:
                return GatewayResponse(
                    ownership=request.ownership,
                    error=ServiceError(
                        ErrorCode.TRANSPORT, f"invalid response body: {exc}", retryable=False, ownership=request.ownership
                    ),
                    attempts=attempt,
                    last_status_code=status_code,
                    transport_ms=transport_total_ms,
                )
            if meta.get("error"):
                err = ServiceError.from_wire(meta["error"])
                return GatewayResponse(
                    ownership=request.ownership,
                    error=err,
                    attempts=attempt,
                    last_status_code=status_code,
                    transport_ms=transport_total_ms,
                )
            return _gateway_response_from_multipart(meta, arrays, request.ownership, attempt, status_code, transport_total_ms)
        # Retry budget exhausted: surface the last typed error (overload or transport).
        return GatewayResponse(
            ownership=request.ownership,
            error=last_error or ServiceError(
                ErrorCode.TRANSPORT, "retry budget exhausted", retryable=False, ownership=request.ownership
            ),
            attempts=attempt,
            last_status_code=last_status,
            transport_ms=transport_total_ms,
        )

    async def call_batch(self, requests: Sequence[GatewayRequest]) -> list[GatewayResponse]:
        """Call multiple items concurrently; each is retried independently.

        Per-item ownership is preserved: responses are returned in input order and
        each carries its own ownership even when items belong to different jobs.
        """
        return await asyncio.gather(*(self.call(r) for r in requests))


# --- generic multipart/envelope builder/parser (multi-part, not rgb-only) ----------


def _build_generic_envelope(
    metadata: Mapping[str, Any], parts: Sequence[tuple[str, bytes, tuple[int, ...], str]],
) -> EnvelopeHttpBody:
    """Build a vectored envelope request: JSON metadata plus named binary parts."""
    encoded_metadata = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    envelope_parts: dict[str, tuple[bytes | memoryview, tuple[int, ...], str]] = {
        "metadata": (encoded_metadata, (), "application/json"),
    }
    for name, data, shape, dtype in parts:
        envelope_parts[name] = (data, shape, dtype)
    return EnvelopeHttpBody(build_binary_envelope(envelope_parts))


def _parse_generic_envelope(
    body: bytes | bytearray | memoryview,
) -> tuple[dict[str, Any], dict[str, tuple[memoryview, tuple[int, ...], str]]]:
    """Parse metadata and tensor vectors from the binary envelope codec."""
    envelope = parse_binary_envelope_body(body)
    by_name = {part.name: part for part in envelope.parts}
    metadata_part = by_name.pop("metadata", None)
    if metadata_part is None or metadata_part.dtype != "application/json" or metadata_part.shape:
        raise ValueError("binary envelope missing valid metadata part")
    try:
        metadata = json.loads(metadata_part.data.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"binary envelope metadata is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("binary envelope metadata must be an object")
    return metadata, {name: (part.data, part.shape, part.dtype) for name, part in by_name.items()}


def _build_generic_multipart(
    metadata: Mapping[str, Any], parts: Sequence[tuple[str, bytes, tuple[int, ...], str]]
) -> tuple[bytes, str]:
    """Build a multipart/form-data body with one JSON metadata part + N binary parts.

    Each binary part declares its tensor ``shape`` and ``dtype`` in
    ``Content-Disposition`` parameters so the receiver reconstructs the array without
    a separate schema. This is the generalized form of
    ``transport.build_multipart_request`` (which is rgb-only); both share the same
    wire format so existing Serve deployments that parse multipart requests keep
    working.
    """
    # Reuse the rgb-only builder's internals by constructing the metadata+parts in
    # the same shape, then delegating to a shared assembler. We inline the assembler
    # to avoid coupling to the rgb-specific helper signature.
    import uuid

    boundary = f"egogateway-{uuid.uuid4().hex}"
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    bdelim = b"--" + boundary.encode("ascii")

    def _part_headers(name: str, shape: tuple[int, ...] | None, dtype: str | None, content_type: str) -> bytes:
        params = f'name="{name}"'
        if shape is not None:
            params += f'; shape="{",".join(str(int(d)) for d in shape)}"'
        if dtype is not None:
            params += f'; dtype="{dtype}"'
        return f"Content-Disposition: form-data; {params}\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8")

    out += bdelim + b"\r\n"
    out += _part_headers("metadata", None, None, "application/json")
    out += meta_bytes + b"\r\n"
    for name, data, shape, dtype in parts:
        out += bdelim + b"\r\n"
        out += _part_headers(name, shape, dtype, "application/octet-stream")
        out += data + b"\r\n"
    out += bdelim + b"--\r\n"
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _gateway_response_from_multipart(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]],
    ownership: Ownership,
    attempts: int,
    status_code: int,
    transport_ms: float,
) -> GatewayResponse:
    result_meta = metadata.get("result", {})
    trace = None
    if isinstance(result_meta, Mapping) and result_meta.get("trace"):
        try:
            trace = BatchTrace.from_wire(result_meta["trace"])
        except Exception:
            trace = None
    arrays_typed: dict[str, TensorPayload] = {}
    for name, (data, shape, dtype) in arrays.items():
        arrays_typed[name] = TensorPayload(data=data, shape=shape, dtype=dtype)
    return GatewayResponse(
        ownership=ownership,
        result=GatewayResult(
            ownership=ownership,
            metadata=dict(result_meta),
            arrays=arrays_typed,
            trace=trace,
        ),
        attempts=attempts,
        last_status_code=status_code,
        transport_ms=transport_ms,
        # Preserve server-produced trace identity.  Higher-level experiment code
        # validates it against the selected endpoint; it must never overwrite it.
        replica_id=trace.replica_id if trace is not None else None,
    )
