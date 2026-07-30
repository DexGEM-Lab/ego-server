"""Ray Serve deployment-only module for the GPU3 HaWoR + motion-infiller slice.

This module imports Ray Serve at the top level and is a **deployment-only import
path**: ordinary adapter/contract unit tests never import it. ``serve run`` and
``serve deploy`` resolve ``ego_annotation.serving.hawor_deployment:app`` against the
GPU3 cluster's ``ray_serve_hawor`` interpreter (Python 3.10, Torch 1.13.0+cu117,
Ray 2.55.1).

Design invariants (per the GPU3 vertical-slice task):

* One Ray cluster for GPU3, started with ``CUDA_VISIBLE_DEVICES=3`` and
  ``--num-gpus=1`` so Ray owns physical GPU3 natively. Component ports 26600-26606,
  worker ports 26700-26731, Serve HTTP 28003, CPU cap 4. The cluster advertises
  one native Ray GPU. Both colocated deployments request ``num_gpus=0.5`` in their
  ``ray_actor_options``: the physical GPU is already pinned at the cluster level
  via ``CUDA_VISIBLE_DEVICES=3``, so both replicas see exactly GPU3 and share it
  without Ray double-counting the resource (requesting ``num_gpus=1`` each would
  require two GPUs and deadlock on this one-GPU cluster). Each requests
  ``num_gpus=0.5`` so both fit on the one advertised GPU and Ray passes
  ``CUDA_VISIBLE_DEVICES=0`` (the cluster's one visible GPU) to both replicas.
  Ray still owns the cluster lifecycle and the GPU is exclusively GPU3.
* Responses are emitted as real ASGI ``starlette.responses.Response`` objects
  carrying the multipart body as raw bytes with the multipart content-type.
  Returning a plain ``dict`` (e.g. ``{"body": bytes, ...}``) is a contract
  violation: Ray Serve 2.55.1's ASGI path runs FastAPI ``jsonable_encoder`` on any
  non-Response return, whose ``bytes`` encoder calls ``bytes.decode()`` and raises
  ``UnicodeDecodeError`` on binary tensor payloads. Error paths return a JSON
  ``Response`` with an explicit 4xx/5xx status.
* Two colocated logical deployments on the one GPU3 replica group:
  ``hawor.infer_tracks`` and ``hawor_infiller.fill``. They have independent batch
  queues, batch limits, and metrics but share the physical GPU. Each loads its
  resident model once at replica init and never reloads.
* Native weighted Serve batching: ``@serve.batch(batch_size_fn=canonical_batch_size_fn)``.
  Each admitted track chunk / infiller window is one work unit, so one Serve callback
  is one model forward (HaWoR) or one-per-window (infiller, which has no cross-window
  fusion dimension).
* Requests arrive as multipart binary HTTP; the deployment parses the body,
  reconstructs the typed request, admits it, and forwards to the batched handler.
  Responses are multipart binary with metadata + dense arrays.
* Server-owned model revision: requests carrying a mismatched revision are rejected
  at admission; every result carries the resident revision only.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ray import serve
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse, StreamingResponse

from ego_annotation.serving.batching import canonical_batch_size_fn
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
    ErrorCode,
    ServiceError,
)
from ego_annotation.serving.hawor import HaWoRAdapter, build_hawor_model_config
from ego_annotation.serving.hawor_contracts import (
    HandSequenceRequest,
    TrackChunkRequest,
)
from ego_annotation.serving.infiller import InfillerAdapter, build_infiller_model_config
from ego_annotation.serving.transport import (
    _iter_multipart,
    _parse_shape,
    multipart_asgi_response,
)


def _hawor_config_from_env() -> Any:
    return build_hawor_model_config(
        checkpoint=os.environ.get(
            "EGO_HAWOR_CHECKPOINT",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt",
        ),
        model_revision=os.environ.get("EGO_HAWOR_REVISION", "hawor-v1"),
        device=os.environ.get("EGO_HAWOR_DEVICE", "cuda"),
        replica_id=os.environ.get("EGO_HAWOR_REPLICA_ID", "hawor-gpu3"),
        assigned_gpu=int(os.environ.get("EGO_HAWOR_GPU", "3")),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_sha=os.environ.get("EGO_APPLICATION_RELEASE_SHA"),
        checkpoint_digest=os.environ.get("EGO_HAWOR_CHECKPOINT_DIGEST"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if os.environ.get("EGO_EXPERIMENT_HTTP_PORT") else None,
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
        performance_instrumentation=os.environ.get("EGO_HAWOR_EXPERIMENT_TELEMETRY", "0") == "1",
        wire_format=os.environ.get("EGO_HAWOR_EXPERIMENT_WIRE_FORMAT", "multipart"),
    )


def _infiller_config_from_env() -> Any:
    return build_infiller_model_config(
        checkpoint=os.environ.get(
            "EGO_HAWOR_INFILLER_CHECKPOINT",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt",
        ),
        model_revision=os.environ.get("EGO_HAWOR_INFILLER_REVISION", "hawor-infiller-v1"),
        device=os.environ.get("EGO_HAWOR_DEVICE", "cuda"),
        replica_id=os.environ.get("EGO_HAWOR_INFILLER_REPLICA_ID", "hawor-infiller-gpu3"),
        assigned_gpu=int(os.environ.get("EGO_HAWOR_GPU", "3")),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_sha=os.environ.get("EGO_APPLICATION_RELEASE_SHA"),
        checkpoint_digest=os.environ.get("EGO_HAWOR_INFILLER_CHECKPOINT_DIGEST"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if os.environ.get("EGO_EXPERIMENT_HTTP_PORT") else None,
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
        performance_instrumentation=os.environ.get("EGO_HAWOR_INFILLER_EXPERIMENT_TELEMETRY", "0") == "1",
        wire_format=os.environ.get("EGO_HAWOR_INFILLER_EXPERIMENT_WIRE_FORMAT", "multipart"),
    )


ArrayPart = tuple[bytes | memoryview, tuple[int, ...], str]


def _unwrap_gateway_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Accept the generic gateway's typed metadata without changing legacy multipart.

    Direct GPU3 callers historically post the request wire object as ``metadata``.
    Gateway callers carry the same object below ``metadata`` and own the outer
    ownership/model revision fields.  Both are normalized before typed parsing.
    """
    nested = metadata.get("metadata")
    if not isinstance(nested, dict) or "ownership" not in metadata:
        return metadata
    payload = dict(nested)
    payload["ownership"] = metadata["ownership"]
    if "model_revision" in metadata:
        payload["model_revision"] = metadata["model_revision"]
    return payload


def _parse_multipart_payloads(body: bytes, content_type: str) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
    """Parse the established multipart request path without changing its bytes."""
    parts = _iter_multipart(body, content_type)
    metadata: dict[str, Any] = {}
    arrays: dict[str, ArrayPart] = {}
    for name, data, params in parts:
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif "shape" in params and "dtype" in params:
            arrays[name] = (data, _parse_shape(params["shape"]), params["dtype"])
    if not metadata:
        raise ValueError("multipart request missing 'metadata' part")
    return _unwrap_gateway_metadata(metadata), arrays


def _parse_envelope_payloads(envelope: BinaryEnvelope) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
    """Decode metadata and independently framed GPU3 binary vectors."""
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
    arrays: dict[str, ArrayPart] = {
        name: (part.data, part.shape, part.dtype) for name, part in parts.items()
    }
    return _unwrap_gateway_metadata(metadata), arrays


def _wire_request_from_multipart(metadata: dict[str, Any], arrays: dict[str, ArrayPart]) -> TrackChunkRequest:
    """Reconstruct a TrackChunkRequest from multipart parts, injecting binary tensors."""
    payload = dict(metadata)
    # Replace base64 tensor placeholders with the actual binary multipart arrays.
    if "crop_batch" in payload and "crop_batch" in arrays:
        b, s, d = arrays["crop_batch"]
        payload["crop_batch"] = {"data_b64": "", "_binary": True, "shape": list(s), "dtype": d}
        payload["_crop_batch_bytes"] = b
    return _track_chunk_from_wire_with_binary(payload, arrays)


def _track_chunk_from_wire_with_binary(payload: dict[str, Any], arrays: dict[str, ArrayPart]) -> TrackChunkRequest:
    from ego_annotation.serving.contracts import Ownership, TensorPayload
    from ego_annotation.serving.hawor_contracts import (
        CropSourceTransform, DroidCameraEvidence, FrameObservation, HandSide, UniDepthScaleK,
    )
    ownership = Ownership.from_mapping(payload["ownership"])
    crop_b = arrays["crop_batch"]
    crop_batch = TensorPayload(data=crop_b[0], shape=crop_b[1], dtype=crop_b[2])
    droid = None
    if payload.get("droid_evidence"):
        de = payload["droid_evidence"]
        poses = arrays.get("droid_poses")
        ts = arrays.get("droid_timestamps")
        if poses is None or ts is None:
            raise ContractValidationError("droid_evidence declared but poses/timestamps binary parts missing")
        droid = DroidCameraEvidence(
            poses_world_from_camera=TensorPayload(data=poses[0], shape=poses[1], dtype=poses[2]),
            timestamps_s=TensorPayload(data=ts[0], shape=ts[1], dtype=ts[2]),
            metric_scale=float(de["metric_scale"]),
            scale_residual=float(de["scale_residual"]),
            scale_confidence=float(de["scale_confidence"]),
            source=de["source"],
        )
    return TrackChunkRequest(
        ownership=ownership,
        track_id=payload["track_id"],
        side=HandSide(payload["side"]),
        crop_batch=crop_batch,
        crop_transforms=tuple(CropSourceTransform.from_mapping(c) for c in payload["crop_transforms"]),
        observations=tuple(FrameObservation.from_mapping(o) for o in payload["observations"]),
        unidepth=UniDepthScaleK.from_mapping(payload["unidepth"]),
        droid_evidence=droid,
        model_revision=payload["model_revision"],
        options=tuple(sorted((str(k), str(v)) for k, v in payload.get("options", {}).items())),
    )


def _json_error_response(error_wire: dict[str, Any], ownership_wire: dict[str, Any] | None = None,
                         status_code: int = 400) -> StarletteResponse:
    """Build a JSON error response carrying the typed ServiceError + ownership."""
    payload = {"metadata": {"error": error_wire}}
    if ownership_wire is not None:
        payload["metadata"]["ownership"] = ownership_wire
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return StarletteResponse(content=body, media_type="application/json", status_code=status_code)


def _envelope_asgi_response(metadata: dict[str, Any], arrays: dict[str, ArrayPart]) -> StarletteResponse:
    """Emit ASGI-valid bytes while preserving vectors until the send boundary."""
    parts: dict[str, ArrayPart] = {
        "metadata": (json.dumps(metadata, separators=(",", ":")).encode("utf-8"), (), "application/json"),
    }
    parts.update(arrays)
    envelope = build_binary_envelope(parts)
    # StreamingResponse rejects memoryview chunks in its ASGI send loop.  Keep all
    # envelope vectors separate, but materialize each individual vector as bytes.
    return StreamingResponse(
        (bytes(chunk) for chunk in binary_envelope_iovecs(envelope)),
        media_type=BINARY_ENVELOPE_CONTENT_TYPE,
    )


def _error_response(error_wire: dict[str, Any], ownership_wire: dict[str, Any] | None = None,
                    status_code: int = 400, *, envelope_wire: bool = False) -> StarletteResponse:
    if envelope_wire:
        metadata: dict[str, Any] = {"error": error_wire}
        if ownership_wire is not None:
            metadata["ownership"] = ownership_wire
        return _envelope_asgi_response(metadata, {})
    return _json_error_response(error_wire, ownership_wire, status_code)


def _hawor_result_to_envelope(result: Any, ownership: Any) -> StarletteResponse:
    arrays: dict[str, ArrayPart] = {
        "root_orient": (result.root_orient.data, result.root_orient.shape, result.root_orient.dtype),
        "hand_pose": (result.hand_pose.data, result.hand_pose.shape, result.hand_pose.dtype),
        "trans": (result.trans.data, result.trans.shape, result.trans.dtype),
        "betas": (result.betas.data, result.betas.shape, result.betas.dtype),
        "vertices": (result.vertices.data, result.vertices.shape, result.vertices.dtype),
        "joints": (result.joints.data, result.joints.shape, result.joints.dtype),
        "observed": (result.observed.data, result.observed.shape, result.observed.dtype),
        "uncertainty": (result.uncertainty.data, result.uncertainty.shape, result.uncertainty.dtype),
    }
    if result.world_lift is not None:
        arrays["world_lift"] = (result.world_lift.data, result.world_lift.shape, result.world_lift.dtype)
    return _envelope_asgi_response({"result": result.to_wire(), "ownership": ownership.to_wire()}, arrays)


def _infiller_result_to_envelope(result: Any, ownership: Any) -> StarletteResponse:
    arrays: dict[str, ArrayPart] = {
        "root_orient": (result.root_orient.data, result.root_orient.shape, result.root_orient.dtype),
        "hand_pose": (result.hand_pose.data, result.hand_pose.shape, result.hand_pose.dtype),
        "trans": (result.trans.data, result.trans.shape, result.trans.dtype),
        "betas": (result.betas.data, result.betas.shape, result.betas.dtype),
        "observed": (result.observed.data, result.observed.shape, result.observed.dtype),
        "inferred": (result.inferred.data, result.inferred.shape, result.inferred.dtype),
        "uncertainty": (result.uncertainty.data, result.uncertainty.shape, result.uncertainty.dtype),
        "timestamps_s": (result.timestamps_s.data, result.timestamps_s.shape, result.timestamps_s.dtype),
    }
    return _envelope_asgi_response({"result": result.to_wire(), "ownership": ownership.to_wire()}, arrays)


def _hawor_result_to_multipart(result: Any, ownership: Any) -> StarletteResponse:
    arrays = {
        "root_orient": (bytes(result.root_orient.data), result.root_orient.shape, result.root_orient.dtype),
        "hand_pose": (bytes(result.hand_pose.data), result.hand_pose.shape, result.hand_pose.dtype),
        "trans": (bytes(result.trans.data), result.trans.shape, result.trans.dtype),
        "betas": (bytes(result.betas.data), result.betas.shape, result.betas.dtype),
        "vertices": (bytes(result.vertices.data), result.vertices.shape, result.vertices.dtype),
        "joints": (bytes(result.joints.data), result.joints.shape, result.joints.dtype),
        "observed": (bytes(result.observed.data), result.observed.shape, result.observed.dtype),
        "uncertainty": (bytes(result.uncertainty.data), result.uncertainty.shape, result.uncertainty.dtype),
    }
    if result.world_lift is not None:
        arrays["world_lift"] = (bytes(result.world_lift.data), result.world_lift.shape, result.world_lift.dtype)
    metadata = {"result": result.to_wire(), "ownership": ownership.to_wire()}
    return multipart_asgi_response(metadata, arrays)


@serve.deployment(
    name="hawor.infer_tracks",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5},
    max_ongoing_requests=16,
    max_queued_requests=64,
)
class HaWoRDeployment:
    def __init__(self) -> None:
        self.adapter = HaWoRAdapter(_hawor_config_from_env())

    @serve.batch(
        max_batch_size=8,
        batch_wait_timeout_s=0.1,
        batch_size_fn=canonical_batch_size_fn,
    )
    async def _batched_infer(self, requests: list[Any]) -> list[Any]:
        return await self.adapter.infer_batch(requests)

    async def infer(self, request: TrackChunkRequest) -> Any:
        try:
            prepared = self.adapter.admit(request)
        except ContractValidationError as exc:
            return _json_error_response(
                ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership).to_wire(),
                request.ownership.to_wire(),
            )
        try:
            batched: Any = self._batched_infer
            return await batched(prepared)
        except BaseException:
            self.adapter.request_dispatched(prepared.request.ownership.request_id)
            raise

    async def __call__(self, request: Request) -> StarletteResponse:
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        envelope_wire = content_type_is_binary_envelope(content_type)
        try:
            metadata, arrays = (
                _parse_envelope_payloads(await read_binary_envelope_stream(request.stream()))
                if envelope_wire else _parse_multipart_payloads(await request.body(), content_type)
            )
            parsed = _wire_request_from_multipart(metadata, arrays)
        except (ContractValidationError, ValueError, KeyError) as exc:
            return _error_response(
                ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False).to_wire(), envelope_wire=envelope_wire,
            )
        outcome = await self.infer(parsed)
        if isinstance(outcome, StarletteResponse):
            return outcome
        result, err = outcome
        if err is not None:
            return _error_response(err.to_wire(), parsed.ownership.to_wire(), status_code=500, envelope_wire=envelope_wire)
        return _hawor_result_to_envelope(result, parsed.ownership) if envelope_wire else _hawor_result_to_multipart(result, parsed.ownership)


@serve.deployment(
    name="hawor_infiller.fill",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5},
    max_ongoing_requests=8,
    max_queued_requests=64,
)
class InfillerDeployment:
    def __init__(self) -> None:
        self.adapter = InfillerAdapter(_infiller_config_from_env())

    @serve.batch(
        max_batch_size=4,
        batch_wait_timeout_s=0.1,
        batch_size_fn=canonical_batch_size_fn,
    )
    async def _batched_fill(self, requests: list[Any]) -> list[Any]:
        return await self.adapter.fill_batch(requests)

    async def fill(self, request: HandSequenceRequest) -> Any:
        try:
            prepared = self.adapter.admit(request)
        except ContractValidationError as exc:
            return _json_error_response(
                ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership).to_wire(),
                request.ownership.to_wire(),
            )
        try:
            batched: Any = self._batched_fill
            return await batched(prepared)
        except BaseException:
            self.adapter.request_dispatched(prepared.request.ownership.request_id)
            raise

    async def __call__(self, request: Request) -> StarletteResponse:
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        envelope_wire = content_type_is_binary_envelope(content_type)
        try:
            metadata, arrays = (
                _parse_envelope_payloads(await read_binary_envelope_stream(request.stream()))
                if envelope_wire else _parse_multipart_payloads(await request.body(), content_type)
            )
            parsed = _infiller_request_from_multipart(metadata, arrays)
        except (ContractValidationError, ValueError, KeyError) as exc:
            return _error_response(
                ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False).to_wire(), envelope_wire=envelope_wire,
            )
        outcome = await self.fill(parsed)
        if isinstance(outcome, StarletteResponse):
            return outcome
        result, err = outcome
        if err is not None:
            return _error_response(err.to_wire(), parsed.ownership.to_wire(), status_code=500, envelope_wire=envelope_wire)
        return _infiller_result_to_envelope(result, parsed.ownership) if envelope_wire else _infiller_result_to_multipart(result, parsed.ownership)


def _infiller_request_from_multipart(metadata: dict[str, Any], arrays: dict[str, ArrayPart]) -> HandSequenceRequest:
    from ego_annotation.serving.contracts import Ownership, TensorPayload
    from ego_annotation.serving.hawor_contracts import DroidCameraEvidence, UniDepthScaleK, _hand_state_frame_from_wire
    payload = dict(metadata)
    ownership = Ownership.from_mapping(payload["ownership"])
    de = payload["droid_evidence"]
    poses = arrays.get("droid_poses")
    ts = arrays.get("droid_timestamps")
    if poses is None or ts is None:
        raise ContractValidationError("droid_evidence declared but poses/timestamps binary parts missing")
    droid = DroidCameraEvidence(
        poses_world_from_camera=TensorPayload(data=poses[0], shape=poses[1], dtype=poses[2]),
        timestamps_s=TensorPayload(data=ts[0], shape=ts[1], dtype=ts[2]),
        metric_scale=float(de["metric_scale"]),
        scale_residual=float(de["scale_residual"]),
        scale_confidence=float(de["scale_confidence"]),
        source=de["source"],
    )
    return HandSequenceRequest(
        ownership=ownership,
        window_id=payload["window_id"],
        frames=tuple(_hand_state_frame_from_wire(f) for f in payload["frames"]),
        droid_evidence=droid,
        unidepth=UniDepthScaleK.from_mapping(payload["unidepth"]),
        model_revision=payload["model_revision"],
        options=tuple(sorted((str(k), str(v)) for k, v in payload.get("options", {}).items())),
    )


def _infiller_result_to_multipart(result: Any, ownership: Any) -> StarletteResponse:
    arrays = {
        "root_orient": (bytes(result.root_orient.data), result.root_orient.shape, result.root_orient.dtype),
        "hand_pose": (bytes(result.hand_pose.data), result.hand_pose.shape, result.hand_pose.dtype),
        "trans": (bytes(result.trans.data), result.trans.shape, result.trans.dtype),
        "betas": (bytes(result.betas.data), result.betas.shape, result.betas.dtype),
        "observed": (bytes(result.observed.data), result.observed.shape, result.observed.dtype),
        "inferred": (bytes(result.inferred.data), result.inferred.shape, result.inferred.dtype),
        "uncertainty": (bytes(result.uncertainty.data), result.uncertainty.shape, result.uncertainty.dtype),
        "timestamps_s": (bytes(result.timestamps_s.data), result.timestamps_s.shape, result.timestamps_s.dtype),
    }
    metadata = {"result": result.to_wire(), "ownership": ownership.to_wire()}
    return multipart_asgi_response(metadata, arrays)


# A bound Ray Serve Application. ``serve run ego_annotation.serving.hawor_deployment:app``
# deploys this against the GPU3 cluster. Both colocated deployments live behind one
# Serve HTTP endpoint (port 28003); routing is by route prefix.
# Ray 2.55.1 ``serve.run`` accepts a single bound deployment, so we expose the HaWoR
# deployment as the primary ``app`` and the infiller as ``infiller_app``; the GPU3
# deploy script calls ``serve.run`` once per app with explicit route prefixes
# (``/hawor.infer_tracks`` and ``/hawor_infiller.fill``).
app: Any = HaWoRDeployment.bind()
infiller_app: Any = InfillerDeployment.bind()
