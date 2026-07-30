"""Ray Serve deployment-only module for the resident stateful DROID model API on GPU2.

This module imports Ray Serve at the top level and is therefore a **deployment-only
import path**: ordinary adapter/contract/transport unit tests never import it.
``serve run``/``serve deploy`` resolve ``ego_annotation.serving.droid_deployment:app``
against the GPU2 cluster's ``ray_serve_hawor`` interpreter (Python 3.10,
Torch 1.13.0+cu117, Ray 2.55.1, lietorch + droid_backends).

Design invariants:

* One Ray cluster per physical GPU group (see ``lifecycle.py``). The GPU2 cluster
  starts with ``CUDA_VISIBLE_DEVICES=2`` and advertises one native Ray GPU
  (``--num-gpus=1``), so this deployment requests ``num_gpus=1`` and Ray owns the
  physical GPU2.
* ``/droid.infer`` is the primary API: one request owns an isolated DROID
  lifecycle. Its complete frame list shares one fnet forward, then runs stateful
  continuation serially in order, finalizes canonical ``T_world_camera``, and
  unconditionally releases its private session.
* The three-step create/push/finalize routes remain temporarily compatible. They
  preserve their existing sticky-session behavior, while infer has no batch window
  or idle reaper and is the deployment's normal orchestration path.
* Binary transport: the established multipart boundary remains the default, while
  ``application/vnd.ego.binary-envelope`` is dual-accepted for every DROID route.
  Both frames-in and CameraState arrays use identical typed metadata and named parts.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from typing import Any, Mapping

from ray import serve
from starlette.responses import StreamingResponse

from ego_annotation.serving.batch_snapshot import snapshot_collection
from ego_annotation.serving.binary_envelope import (
    CONTENT_TYPE as BINARY_ENVELOPE_CONTENT_TYPE,
    BinaryEnvelope,
    binary_envelope_iovecs,
    build_binary_envelope,
    content_type_is_binary_envelope,
    read_binary_envelope_stream,
)
from ego_annotation.serving.contracts import (
    ContractValidationError,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    ErrorCode,
    Ownership,
    ServiceError,
)
from ego_annotation.serving.droid import DroidAdapter, build_droid_model_config
from ego_annotation.serving.transport import build_multipart_response


def _config_from_env() -> Any:
    return build_droid_model_config(
        weights=os.environ.get(
            "EGO_DROID_WEIGHTS",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth",
        ),
        model_revision=os.environ.get("EGO_DROID_REVISION", "droid-v1"),
        device=os.environ.get("EGO_DROID_DEVICE", "cuda:0"),
        assigned_gpu=int(os.environ.get("EGO_DROID_GPU", "2")),
        replica_id=os.environ.get("EGO_DROID_REPLICA_ID", "droid-gpu2"),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if os.environ.get("EGO_EXPERIMENT_HTTP_PORT") else None,
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
        droid_source_release_path=os.environ.get("EGO_DROID_SOURCE_ROOT"),
        droid_source_digest=os.environ.get("EGO_DROID_SOURCE_DIGEST"),
        droid_source_amendment_id=os.environ.get("EGO_DROID_SOURCE_AMENDMENT"),
        # One self-contained infer lifecycle occupies one replica.  The public
        # contract fixes buffer=256; the extra slot is DROID's predictive seed.
        max_sessions=int(os.environ.get("EGO_DROID_MAX_SESSIONS", "1")),
        max_queued_frames_per_session=1,
        max_fnet_batch_size=256,
        fnet_batch_wait_timeout_s=0.5,
        idle_session_ttl_s=60.0,
        idle_reaper_interval_s=10.0,
        max_result_journal_entries_per_session=257,
        max_buffer_slots=257,
        max_terminal_tombstones=1,
        cpu_offload=False,
        max_concurrent_ba=1,
        performance_instrumentation=os.environ.get("EGO_DROID_EXPERIMENT_TELEMETRY", "0") == "1",
        wire_format=os.environ.get("EGO_DROID_EXPERIMENT_WIRE_FORMAT", "multipart"),
    )


@serve.deployment(
    name="droid",
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=32,
    max_queued_requests=64,
)
class DroidDeployment:
    def __init__(self) -> None:
        # Infer releases its complete lifecycle in _handle_infer's finally block.
        # No reaper runs on this deployment, so occupancy remains request-owned.
        self.adapter = DroidAdapter(_config_from_env())

    def _runtime_diagnostics(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Return only server-owned wire attestation for instrumented treatments."""
        config = getattr(self.adapter, "config", None)
        if config is None or not getattr(config, "performance_instrumentation", False):
            return None
        concurrency_diagnostics = getattr(self.adapter, "concurrency_diagnostics", None)
        diagnostics: dict[str, Any] = {
            "runtime_config": config.runtime_config_wire(),
            "runtime_config_digest": config.runtime_config_digest(),
            **(concurrency_diagnostics() if callable(concurrency_diagnostics) else {}),
        }
        # Allocator state is a worker fact, not an NVML inference.  It is returned
        # only on the explicitly instrumented experimental lane and makes the
        # buffer/concurrency sweep's per-treatment high-water reproducible.
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device(config.device)
                diagnostics["allocator_memory"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
        except Exception as exc:
            diagnostics["allocator_memory_error"] = repr(exc)
        if session_id is not None:
            diagnostics["session_allocator_memory"] = self.adapter.session_allocator_memory(session_id)
        return diagnostics

    async def create_session(self, request: DroidCreateSessionRequest) -> DroidCreateSessionResponse:
        response = self.adapter.create_session(request)
        if response.error is None and response.session_id is not None:
            self.adapter.response_sent(response.session_id)
        return replace(
            response,
            server_identity=self.adapter.server_identity,
            batch_diagnostics=self._runtime_diagnostics(response.session_id),
        )

    @serve.batch(
        max_batch_size=8,
        batch_wait_timeout_s=0.5,
        batch_size_fn=lambda items: len(items),
    )
    async def _batched_push_frame(self, requests: list[Any]) -> list[DroidFrameResponse]:
        # Each item is an _PreparedFrame produced by adapter.admit_frame. One fused
        # fnet forward runs across compatible ready frames from distinct sessions;
        # at most one ready frame per session is admitted (enforced at admission).
        return await self.adapter.push_frame_batch(requests)

    async def push_frame(self, request: DroidFrameRequest) -> DroidFrameResponse:
        try:
            prepared = self.adapter.admit_frame(request)
        except ContractValidationError as exc:
            return DroidFrameResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
                server_identity=self.adapter.server_identity,
                batch_diagnostics=self._runtime_diagnostics(),
            )
        if isinstance(prepared, DroidFrameResponse):
            session_id = prepared.status.session_id if prepared.status is not None else None
            return replace(
                prepared,
                server_identity=self.adapter.server_identity,
                batch_diagnostics=self._runtime_diagnostics(session_id),
            )
        try:
            batched: Any = self._batched_push_frame
            response = await batched(prepared)
            session_id = response.status.session_id if response.status is not None else None
            if response.error is None and session_id is not None:
                self.adapter.response_sent(session_id)
            return replace(
                response,
                server_identity=self.adapter.server_identity,
                batch_diagnostics=self._runtime_diagnostics(session_id),
            )
        except ContractValidationError as exc:
            self.adapter.request_abandoned(prepared)
            return DroidFrameResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
                server_identity=self.adapter.server_identity,
                batch_diagnostics=self._runtime_diagnostics(),
            )
        except BaseException:
            # If Serve cancels before this item enters the batch callback, undo its
            # admission. Once dispatched, adapter completion is shielded/joined by
            # the worker callback so state is never released while CUDA mutates it.
            self.adapter.request_abandoned(prepared)
            raise

    async def finalize(self, request: DroidFinalizeRequest) -> DroidFinalizeResponse:
        try:
            response = await self.adapter.finalize(request)
        except ContractValidationError as exc:
            return DroidFinalizeResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
                server_identity=self.adapter.server_identity,
            )
        return replace(
            response,
            server_identity=self.adapter.server_identity,
            batch_diagnostics=self._runtime_diagnostics(request.session_id),
        )

    def status(self) -> dict[str, Any]:
        return self.adapter.status().to_wire()

    async def __call__(self, request: Any) -> Any:
        path = request.url.path if hasattr(request, "url") else ""
        if getattr(request, "method", "").upper() == "GET" and path.rstrip("/").endswith("/-/batch-snapshot"):
            return _json_response(snapshot_collection(self.adapter.batch_snapshot()), headers={"Cache-Control": "no-store"})
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        envelope_wire = content_type_is_binary_envelope(content_type)
        body: bytes | BinaryEnvelope
        try:
            body = await read_binary_envelope_stream(request.stream()) if envelope_wire else await request.body()
        except (ValueError, UnicodeDecodeError) as exc:
            # Framing failures happen before a typed route parser can recover
            # ownership, but still use the selected wire for explicit transport
            # failure evidence rather than leaking an ASGI exception.
            return _metadata_response(
                _error_wire(None, exc, _deployment_identity(self)), status_code=400, envelope_wire=envelope_wire,
            )
        if path.endswith("/droid.create_session") or path.endswith("/create_session"):
            return await self._handle_create(body, content_type, envelope_wire=envelope_wire)
        if path.endswith("/droid.push_frame") or path.endswith("/push_frame"):
            return await self._handle_push(body, content_type, envelope_wire=envelope_wire)
        if path.endswith("/droid.finalize") or path.endswith("/finalize"):
            return await self._handle_finalize(body, content_type, envelope_wire=envelope_wire)
        if path.endswith("/droid.infer") or path.endswith("/infer"):
            return await self._handle_infer(body, content_type, envelope_wire=envelope_wire)
        if path.endswith("/status"):
            return _metadata_response({"status": self.status()}, envelope_wire=envelope_wire)
        return _metadata_response({"error": "unknown route", "path": path}, status_code=404, envelope_wire=envelope_wire)

    async def _handle_create(self, body: bytes | BinaryEnvelope, content_type: str, *, envelope_wire: bool = False) -> Any:
        metadata: dict[str, Any] = {}
        try:
            metadata, _arrays = _parse_wire_payloads(body, content_type)
            request = DroidCreateSessionRequest.from_wire(metadata)
        except (ContractValidationError, KeyError, ValueError, UnicodeDecodeError) as exc:
            return _metadata_response(_error_wire(_safe_ownership(metadata), exc, _deployment_identity(self)), status_code=400, envelope_wire=envelope_wire)
        response = await self.create_session(request)
        return _metadata_response(response.to_wire(), status_code=_droid_response_status(response), envelope_wire=envelope_wire)

    async def _handle_push(self, body: bytes | BinaryEnvelope, content_type: str, *, envelope_wire: bool = False) -> Any:
        metadata: dict[str, Any] = {}
        try:
            metadata, arrays = _parse_wire_payloads(body, content_type)
            ownership = Ownership.from_mapping(metadata.get("ownership", {}))
            rgb_bytes, rgb_shape, rgb_dtype = arrays["rgb"]
            mask = arrays.get("static_confidence_mask")
            depth = arrays.get("depth_m")
            request = DroidFrameRequest(
                ownership=ownership,
                session_id=metadata["session_id"],
                frame_id=metadata["frame_id"],
                source_timestamp_s=float(metadata["source_timestamp_s"]),
                rgb=_tensor(rgb_bytes, rgb_shape, rgb_dtype),
                static_confidence_mask=_tensor(*mask) if mask else None,
                depth_m=_tensor(*depth) if depth else None,
                model_revision=metadata.get("model_revision", ""),
            )
        except (ContractValidationError, KeyError, ValueError, UnicodeDecodeError) as exc:
            return _metadata_response(_error_wire(_safe_ownership(metadata), exc, _deployment_identity(self)), status_code=400, envelope_wire=envelope_wire)
        response = await self.push_frame(request)
        return _metadata_response(_frame_response_to_wire(response), status_code=_droid_response_status(response), envelope_wire=envelope_wire)

    async def _handle_finalize(self, body: bytes | BinaryEnvelope, content_type: str, *, envelope_wire: bool = False) -> Any:
        metadata: dict[str, Any] = {}
        try:
            metadata, _arrays = _parse_wire_payloads(body, content_type)
            ownership = Ownership.from_mapping(metadata.get("ownership", {}))
            request = DroidFinalizeRequest(
                ownership=ownership,
                session_id=metadata["session_id"],
                model_revision=metadata.get("model_revision", ""),
            )
        except (ContractValidationError, KeyError, ValueError, UnicodeDecodeError) as exc:
            return _metadata_response(_error_wire(_safe_ownership(metadata), exc, _deployment_identity(self)), status_code=400, envelope_wire=envelope_wire)
        response = await self.finalize(request)
        return _finalize_response_to_http_response(response, envelope_wire=envelope_wire)


    async def _handle_infer(self, body: bytes | BinaryEnvelope, content_type: str, *, envelope_wire: bool = False) -> Any:
        """Run one complete DROID lifecycle and retire its session unconditionally.

        The public request owns a sequence; internal lifecycle operations receive
        derived ownership IDs because push idempotency is keyed by request ID.
        The final CameraState is normalized back to the public infer ownership.
        """
        metadata: dict[str, Any] = {}
        session_id: str | None = None
        public_ownership: Ownership | None = None
        try:
            metadata, arrays = _parse_wire_payloads(body, content_type)
            create = DroidCreateSessionRequest.from_wire(metadata)
            public_ownership = create.ownership
            frames = _infer_frames(metadata, arrays, create)
            if create.options.buffer != 256:
                raise ContractValidationError("droid.infer requires options.buffer=256")
            create = replace(create, ownership=_infer_ownership(public_ownership, "create"))
            created = await self.create_session(create)
            if created.error is not None or created.session_id is None:
                return _metadata_response(created.to_wire(), status_code=_droid_response_status(created), envelope_wire=envelope_wire)
            session_id = created.session_id
            responses = await self.adapter.infer_frames([
                replace(frame, session_id=session_id) for frame in frames
            ])
            for response in responses:
                if response.error is not None:
                    return _metadata_response(response.to_wire(), status_code=_droid_response_status(response), envelope_wire=envelope_wire)
            finalize = await self.finalize(DroidFinalizeRequest(
                ownership=_infer_ownership(public_ownership, "finalize"),
                session_id=session_id,
                model_revision=create.model_revision,
            ))
            if finalize.error is not None:
                return _finalize_response_to_http_response(finalize, envelope_wire=envelope_wire)
            assert finalize.camera_state is not None
            public_state = replace(finalize.camera_state, ownership=public_ownership)
            public_response = replace(finalize, ownership=public_state.ownership, camera_state=public_state)
            return _finalize_response_to_http_response(public_response, envelope_wire=envelope_wire)
        except (ContractValidationError, KeyError, ValueError, UnicodeDecodeError) as exc:
            return _metadata_response(_error_wire(_safe_ownership(metadata), exc, _deployment_identity(self)), status_code=400, envelope_wire=envelope_wire)
        except Exception as exc:
            ownership = public_ownership or _safe_ownership(metadata)
            if ownership is None:
                return _metadata_response(_error_wire(None, exc, _deployment_identity(self)), status_code=503, envelope_wire=envelope_wire)
            failure = DroidFinalizeResponse(
                ownership=ownership,
                error=ServiceError(ErrorCode.MODEL_FAILURE, str(exc), retryable=False, ownership=ownership),
                terminal=True,
            )
            return _finalize_response_to_http_response(failure, envelope_wire=envelope_wire)
        finally:
            if session_id is not None:
                # release_session waits for shielded CUDA workers before it clears
                # state, so a disconnected caller cannot leave an OPEN session or
                # race a still-mutating DepthVideo.
                await self.adapter.release_session(session_id)


# --------------------------------------------------------------------------- #
# Wire helpers (multipart metadata + named binary array parts).
# --------------------------------------------------------------------------- #


def _tensor(data: bytes | memoryview, shape: tuple[int, ...], dtype: str) -> Any:
    from ego_annotation.serving.contracts import TensorPayload

    return TensorPayload(data=data, shape=shape, dtype=dtype)


def _deployment_identity(deployment: Any) -> Any | None:
    """Allow parser tests to construct a deployment shell without a resident adapter."""
    return getattr(getattr(deployment, "adapter", None), "server_identity", None)


def _safe_ownership(metadata: dict[str, Any]) -> Ownership | None:
    try:
        return Ownership.from_mapping(metadata.get("ownership", {}))
    except Exception:
        return None


def _error_wire(
    ownership: Ownership | None, exc: Exception, server_identity: Any | None,
) -> dict[str, Any]:
    """Serialize parser/admission errors with worker-derived identity evidence.

    The request may be invalid, but the serving actor that rejected it is still the
    measurement origin. Never reconstruct this from caller-selected endpoint/env.
    """
    err = ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=ownership)
    return {
        "ownership": ownership.to_wire() if ownership else {},
        "error": err.to_wire(),
        "server_identity": server_identity.to_wire() if server_identity else None,
    }


def _json_response(payload: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None) -> Any:
    """Return an actual ASGI JSON response instead of a Serve-serialized dict."""
    from starlette.responses import JSONResponse

    return JSONResponse(payload, status_code=status_code, headers=headers)


def _droid_response_status(
    response: DroidCreateSessionResponse | DroidFrameResponse | DroidFinalizeResponse,
) -> int:
    if response.error is None:
        return 200
    return {
        ErrorCode.VALIDATION: 400,
        ErrorCode.CONFLICT: 409,
        ErrorCode.BACKPRESSURE: 429,
        ErrorCode.UNRESOLVED: 422,
        ErrorCode.MODEL_FAILURE: 503,
        ErrorCode.RESULT_SPLIT_FAILURE: 503,
        ErrorCode.TRANSPORT: 503,
    }[response.error.code]


def _multipart_response(body: bytes, content_type: str) -> Any:
    """Preserve multipart bytes and boundary at the established HTTP boundary."""
    from starlette.responses import Response

    return Response(content=body, headers={"Content-Type": content_type})


ArrayPart = tuple[bytes | memoryview, tuple[int, ...], str]


def _unwrap_gateway_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize the generic gateway envelope without changing direct DROID callers."""
    nested = metadata.get("metadata")
    if not isinstance(nested, dict) or "ownership" not in metadata:
        return metadata
    payload = dict(nested)
    payload["ownership"] = metadata["ownership"]
    if "model_revision" in metadata:
        payload["model_revision"] = metadata["model_revision"]
    return payload


def _infer_ownership(ownership: Ownership, operation: str, *, frame_index: int | None = None, timestamp_s: float | None = None) -> Ownership:
    """Derive unique legacy-operation ownership from one public infer request."""
    suffix = operation if frame_index is None else f"push:{frame_index}"
    return replace(
        ownership,
        request_id=f"{ownership.request_id}:infer:{suffix}",
        stage_id=f"droid.{operation}" if frame_index is None else "droid.push_frame",
        source_timestamp_s=timestamp_s if timestamp_s is not None else ownership.source_timestamp_s,
    )


def _infer_frames(
    metadata: Mapping[str, Any], arrays: Mapping[str, ArrayPart], create: DroidCreateSessionRequest,
) -> tuple[DroidFrameRequest, ...]:
    """Decode indexed infer parts into the established per-frame request contract."""
    raw_frames = metadata.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ContractValidationError("droid.infer requires a non-empty frames list")
    if len(raw_frames) > 256:
        raise ContractValidationError("droid.infer accepts at most 256 frames")
    frames: list[DroidFrameRequest] = []
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise ContractValidationError(f"droid.infer frames[{index}] must be an object")
        frame_id = raw_frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ContractValidationError(f"droid.infer frames[{index}].frame_id is required")
        timestamp_s = float(raw_frame["source_timestamp_s"])
        try:
            rgb = arrays[f"rgb_{index}"]
        except KeyError as exc:
            raise ContractValidationError(f"droid.infer missing rgb_{index}") from exc
        mask = arrays.get(f"mask_{index}")
        depth = arrays.get(f"depth_{index}")
        frames.append(DroidFrameRequest(
            ownership=_infer_ownership(create.ownership, "push", frame_index=index, timestamp_s=timestamp_s),
            session_id="pending-infer-session",
            frame_id=frame_id,
            source_timestamp_s=timestamp_s,
            rgb=_tensor(*rgb),
            static_confidence_mask=_tensor(*mask) if mask is not None else None,
            depth_m=_tensor(*depth) if depth is not None else None,
            model_revision=create.model_revision,
        ))
    return tuple(frames)


def _parse_metadata_only(body: bytes, content_type: str) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
    """Compatibility parser for direct multipart create/finalize callers."""
    return _parse_wire_payloads(body, content_type)


def _parse_wire_payloads(body: bytes | BinaryEnvelope, content_type: str) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
    """Decode either framing into the same typed metadata and named binary parts."""
    if content_type_is_binary_envelope(content_type):
        envelope = body if isinstance(body, BinaryEnvelope) else None
        if envelope is None:
            raise ValueError("binary-envelope request must be read as an envelope stream")
        parts = {part.name: part for part in envelope.parts}
        metadata_part = parts.pop("metadata", None)
        if metadata_part is None or metadata_part.dtype != "application/json" or metadata_part.shape:
            raise ValueError("binary envelope missing valid metadata part")
        try:
            metadata = json.loads(metadata_part.data.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"binary envelope metadata is invalid JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("binary envelope metadata must be an object")
        return _unwrap_gateway_metadata(metadata), {
            name: (part.data, part.shape, part.dtype) for name, part in parts.items()
        }
    if not isinstance(body, bytes):
        raise ValueError("multipart request body must be bytes")
    return parse_multipart_request_droid(body, content_type)


def parse_multipart_request_droid(body: bytes, content_type: str) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
    """Parse the established multipart push request without changing its bytes."""
    parts = _iter_parts(body, content_type)
    metadata: dict[str, Any] = {}
    arrays: dict[str, ArrayPart] = {}
    for name, data, params in parts:
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif "shape" in params and "dtype" in params:
            arrays[name] = (data, _parse_shape(params["shape"]), params["dtype"])
    return _unwrap_gateway_metadata(metadata), arrays


def _iter_parts(body: bytes, content_type: str) -> list[tuple[str, bytes, dict[str, str]]]:
    from email.parser import BytesParser
    from email.policy import default as default_email_policy

    header_blob = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default_email_policy).parsebytes(header_blob + body)
    results: list[tuple[str, bytes, dict[str, str]]] = []
    if not message.is_multipart():
        return results
    for part in message.get_payload():
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        params: dict[str, str] = {}
        shape = part.get_param("shape", header="content-disposition")
        dtype = part.get_param("dtype", header="content-disposition")
        if shape is not None:
            params["shape"] = str(shape)
        if dtype is not None:
            params["dtype"] = str(dtype)
        payload = part.get_payload(decode=True)
        results.append((str(name), bytes(payload or b""), params))
    return results


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def _frame_response_to_wire(response: DroidFrameResponse) -> dict[str, Any]:
    return response.to_wire()


def _envelope_asgi_response(metadata: dict[str, Any], arrays: dict[str, ArrayPart], *, status_code: int = 200) -> Any:
    """Emit byte chunks at the ASGI edge; StreamingResponse cannot send memoryviews."""
    parts: dict[str, ArrayPart] = {
        "metadata": (json.dumps(metadata, separators=(",", ":")).encode("utf-8"), (), "application/json"),
    }
    parts.update(arrays)
    envelope = build_binary_envelope(parts)
    return StreamingResponse(
        (bytes(chunk) for chunk in binary_envelope_iovecs(envelope)),
        media_type=BINARY_ENVELOPE_CONTENT_TYPE,
        status_code=status_code,
    )


def _metadata_response(metadata: dict[str, Any], *, status_code: int = 200, envelope_wire: bool = False) -> Any:
    if envelope_wire:
        return _envelope_asgi_response(metadata, {}, status_code=status_code)
    return _json_response(metadata, status_code=status_code)


def _finalize_response_to_http_response(response: DroidFinalizeResponse, *, envelope_wire: bool = False) -> Any:
    if response.error is not None:
        return _metadata_response(response.to_wire(), status_code=_droid_response_status(response), envelope_wire=envelope_wire)
    state = response.camera_state
    assert state is not None
    # This descriptor and ordered part set are the established typed finalize contract.
    arrays: dict[str, ArrayPart] = {
        "T_world_camera": (bytes(state.T_world_camera.data), state.T_world_camera.shape, state.T_world_camera.dtype),
        "T_camera_world": (bytes(state.T_camera_world.data), state.T_camera_world.shape, state.T_camera_world.dtype),
        "intrinsics_px": (bytes(state.intrinsics_px.data), state.intrinsics_px.shape, state.intrinsics_px.dtype),
        "disparities": (bytes(state.disparities.data), state.disparities.shape, state.disparities.dtype),
    }
    state_metadata = state.to_wire()
    for name, (_data, shape, dtype) in arrays.items():
        state_metadata[name] = {"part": name, "shape": list(shape), "dtype": dtype}
    metadata = {
        "ownership": response.ownership.to_wire(),
        "camera_state": state_metadata,
        "error": None,
        "server_identity": response.server_identity.to_wire() if response.server_identity else None,
    }
    if response.batch_diagnostics is not None:
        metadata["batch_diagnostics"] = dict(response.batch_diagnostics)
    if envelope_wire:
        return _envelope_asgi_response(metadata, arrays, status_code=_droid_response_status(response))
    multipart_arrays = {
        name: (bytes(data), shape, dtype) for name, (data, shape, dtype) in arrays.items()
    }
    body, content_type = build_multipart_response(metadata, multipart_arrays)
    return _multipart_response(body, content_type)


# A bound Ray Serve Application. ``serve run ego_annotation.serving.droid_deployment:app``
# deploys this against the GPU2 cluster.
app: Any = DroidDeployment.bind()
