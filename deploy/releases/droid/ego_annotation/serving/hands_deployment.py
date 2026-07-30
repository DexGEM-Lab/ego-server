"""Ray Serve deployment-only module for GPU1 ``hands.detect`` and ``wilor.reconstruct``.

The deployment dual-accepts the established multipart body and the experimental
``application/vnd.ego.binary-envelope`` body.  Each branch reconstructs the same
typed request before admission.  Multipart remains the default and retains its
existing response assembler; envelope responses preserve the identical metadata and
named binary vectors while streaming ASGI-valid ``bytes`` chunks.
"""
from __future__ import annotations

import json
import os
from typing import Any, cast

from fastapi import FastAPI, Request
from ray import serve
from starlette.responses import Response, StreamingResponse

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
    HandSide,
    HandsDetectRequest,
    HandsDetectResponse,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    WiLoRReconstructRequest,
    WiLoRReconstructResponse,
)
from ego_annotation.serving.hands import HandsAdapter, build_hands_model_config
from ego_annotation.serving.transport import (
    multipart_asgi_response,
    parse_multipart_request_fields,
)
from ego_annotation.serving.wilor import WiLoRAdapter, build_wilor_model_config

ArrayPart = tuple[bytes | bytearray | memoryview, tuple[int, ...], str]

# Ray Serve workers do not inherit arbitrary driver environment variables.  The
# experiment lifecycle passes this exact allowlist into the replica runtime so the
# model-source imports and worker-derived identity have the same pinned contract as
# the detached driver.  Production retains its existing process environment.
_RUNTIME_ENV_KEYS = (
    "EGO_SAM2_REPO", "EGO_WILOR_REPO", "EGO_HANDS_GPU", "EGO_HANDS_REPLICA_ID", "EGO_WILOR_REPLICA_ID",
    "EGO_EXPERIMENT_ID", "EGO_APPLICATION_RELEASE_SHA", "EGO_APPLICATION_RELEASE_ROOT",
    "EGO_EXPERIMENT_GCS_ADDRESS", "EGO_EXPERIMENT_HTTP_PORT", "EGO_EXPERIMENT_TEMP_DIR",
    "EGO_HANDS_EXPERIMENT_TELEMETRY", "EGO_WILOR_EXPERIMENT_TELEMETRY",
    "EGO_HANDS_EXPERIMENT_WIRE_FORMAT", "EGO_WILOR_EXPERIMENT_WIRE_FORMAT",
)


def _replica_runtime_env() -> dict[str, dict[str, str]]:
    return {"env_vars": {key: os.environ[key] for key in _RUNTIME_ENV_KEYS if key in os.environ}}


def _hands_config_from_env() -> Any:
    return build_hands_model_config(
        detector_checkpoint=os.environ.get(
            "EGO_HANDS_DETECTOR_CKPT",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/detector.pt",
        ),
        sam2_checkpoint=os.environ.get(
            "EGO_SAM2_CKPT",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/sam2.1/sam2.1_hiera_large.pt",
        ),
        sam2_config=os.environ.get("EGO_SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        model_revision=os.environ.get("EGO_HANDS_REVISION", "hands-yolo-sam2.1-hiera-l"),
        canonical_height=int(os.environ.get("EGO_HANDS_CANONICAL_H", "540")),
        canonical_width=int(os.environ.get("EGO_HANDS_CANONICAL_W", "960")),
        assigned_gpu=int(os.environ.get("EGO_HANDS_GPU", "1")),
        performance_instrumentation=os.environ.get("EGO_HANDS_EXPERIMENT_TELEMETRY", "0") == "1",
        wire_format=os.environ.get("EGO_HANDS_EXPERIMENT_WIRE_FORMAT", "multipart"),
        replica_id=os.environ.get("EGO_HANDS_REPLICA_ID", "hands-wilor-gpu1"),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=(int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if "EGO_EXPERIMENT_HTTP_PORT" in os.environ else None),
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
    )


def _wilor_config_from_env() -> Any:
    return build_wilor_model_config(
        checkpoint=os.environ.get(
            "EGO_WILOR_CKPT",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/wilor_final.ckpt",
        ),
        config_path=os.environ.get(
            "EGO_WILOR_CFG",
            "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/model_config.yaml",
        ),
        model_revision=os.environ.get("EGO_WILOR_REVISION", "wilor-final-v1"),
        assigned_gpu=int(os.environ.get("EGO_HANDS_GPU", "1")),
        performance_instrumentation=os.environ.get("EGO_WILOR_EXPERIMENT_TELEMETRY", "0") == "1",
        wire_format=os.environ.get("EGO_WILOR_EXPERIMENT_WIRE_FORMAT", "multipart"),
        replica_id=os.environ.get("EGO_WILOR_REPLICA_ID", "hands-wilor-gpu1"),
        experiment_id=os.environ.get("EGO_EXPERIMENT_ID"),
        application_release_path=os.environ.get("EGO_APPLICATION_RELEASE_ROOT"),
        gcs_address=os.environ.get("EGO_EXPERIMENT_GCS_ADDRESS"),
        http_port=(int(os.environ["EGO_EXPERIMENT_HTTP_PORT"]) if "EGO_EXPERIMENT_HTTP_PORT" in os.environ else None),
        temp_dir=os.environ.get("EGO_EXPERIMENT_TEMP_DIR"),
    )


# A Serve ingress is mandatory for binary responses. A direct deployment return is
# JSON serialized by Serve and corrupts raw tensor bytes. Build FastAPI inside the
# replica because the deployed FastAPI object owns an unpicklable lock.
def build_hands_api() -> FastAPI:
    app = FastAPI()

    @app.post("/hands.detect")
    async def hands_detect_http(request: Request) -> Response:
        replica = serve.get_replica_context().servable_object
        return await replica._handle_hands_detect_http(request)

    @app.post("/wilor.reconstruct")
    async def wilor_reconstruct_http(request: Request) -> Response:
        replica = serve.get_replica_context().servable_object
        return await replica._handle_wilor_reconstruct_http(request)

    return app


@serve.deployment(
    name="hands_wilor",
    num_replicas=1,
    ray_actor_options=dict(num_gpus=1, runtime_env=_replica_runtime_env()),
    max_ongoing_requests=32,
    max_queued_requests=128,
)
@serve.ingress(build_hands_api)
class HandsWiLoRDeployment:
    """One GPU1 replica hosting both logical APIs with separate batch queues."""

    def __init__(self) -> None:
        self.hands = HandsAdapter(_hands_config_from_env())
        self.wilor = WiLoRAdapter(_wilor_config_from_env())

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.04, batch_size_fn=canonical_batch_size_fn)
    async def _batched_detect(self, requests: list[Any]) -> list[HandsDetectResponse]:
        return await self.hands.infer_batch(requests)

    async def detect(self, request: HandsDetectRequest) -> HandsDetectResponse:
        try:
            prepared = self.hands.admit(request)
        except ContractValidationError as exc:
            return HandsDetectResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
            )
        try:
            batched: Any = self._batched_detect
            return await batched(prepared)
        except BaseException:
            self.hands.request_dispatched(prepared.request.ownership.request_id)
            raise

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.04, batch_size_fn=canonical_batch_size_fn)
    async def _batched_reconstruct(self, requests: list[Any]) -> list[WiLoRReconstructResponse]:
        return await self.wilor.reconstruct_batch(requests)

    async def reconstruct(self, request: WiLoRReconstructRequest) -> WiLoRReconstructResponse:
        try:
            prepared = self.wilor.admit(request)
        except ContractValidationError as exc:
            return WiLoRReconstructResponse(
                ownership=request.ownership,
                error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=request.ownership),
            )
        try:
            batched: Any = self._batched_reconstruct
            return await batched(prepared)
        except BaseException:
            self.wilor.request_dispatched(prepared.request.ownership.request_id)
            raise

    async def _parse_http_request(
        self, request: Request,
    ) -> tuple[dict[str, Any], dict[str, ArrayPart], bool] | Response:
        content_type = request.headers.get("Content-Type", "multipart/form-data")
        envelope_wire = content_type_is_binary_envelope(content_type)
        try:
            if envelope_wire:
                metadata, fields = _parse_envelope_payloads(await read_binary_envelope_stream(request.stream()))
            else:
                # This established multipart parser is deliberately unchanged.
                metadata, fields = parse_multipart_request_fields(await request.body(), content_type)
                metadata = _unwrap_gateway_metadata(metadata)
            return metadata, cast(dict[str, ArrayPart], fields), envelope_wire
        except Exception as exc:
            return _error_response(str(exc), envelope_wire=envelope_wire)

    async def _handle_hands_detect_http(self, request: Request) -> Response:
        parsed = await self._parse_http_request(request)
        if isinstance(parsed, Response):
            return parsed
        metadata, fields, envelope_wire = parsed
        return await self._serve_detect(metadata, fields, envelope_wire)

    async def _handle_wilor_reconstruct_http(self, request: Request) -> Response:
        parsed = await self._parse_http_request(request)
        if isinstance(parsed, Response):
            return parsed
        metadata, fields, envelope_wire = parsed
        return await self._serve_reconstruct(metadata, fields, envelope_wire)

    async def _serve_detect(
        self, metadata: dict[str, Any], fields: dict[str, ArrayPart], envelope_wire: bool,
    ) -> Response:
        if "rgb" not in fields:
            return _error_response("hands.detect requires an 'rgb' binary field", envelope_wire=envelope_wire)
        rgb_bytes, rgb_shape, rgb_dtype = fields["rgb"]
        try:
            parsed = HandsDetectRequest(
                ownership=_ownership(metadata),
                rgb=TensorPayload(data=rgb_bytes, shape=rgb_shape, dtype=rgb_dtype),
                spatial=SpatialMetadata.from_mapping(metadata["spatial"]),
                model_revision=metadata["model_revision"],
                options=tuple(sorted((str(k), str(v)) for k, v in metadata.get("options", {}).items())),
            )
        except (ContractValidationError, KeyError) as exc:
            return _error_response(str(exc), _safe_ownership(metadata), envelope_wire=envelope_wire)
        response = await self.detect(parsed)
        return _hands_response_to_envelope_wire(response) if envelope_wire else _hands_response_to_multipart_wire(response)

    async def _serve_reconstruct(
        self, metadata: dict[str, Any], fields: dict[str, ArrayPart], envelope_wire: bool,
    ) -> Response:
        if "crop" not in fields:
            return _error_response("wilor.reconstruct requires a 'crop' binary field", envelope_wire=envelope_wire)
        crop_bytes, crop_shape, crop_dtype = fields["crop"]
        try:
            parsed = WiLoRReconstructRequest(
                ownership=_ownership(metadata),
                crop=TensorPayload(data=crop_bytes, shape=crop_shape, dtype=crop_dtype),
                handedness=HandSide.from_value(metadata.get("handedness")),
                box_center=cast(tuple[float, float], tuple(float(v) for v in metadata.get("box_center", (0.0, 0.0)))),
                box_size=float(metadata.get("box_size", 0.0)),
                img_size=cast(tuple[float, float], tuple(float(v) for v in metadata.get("img_size", (0.0, 0.0)))),
                source_K_px=metadata.get("source_K_px"),
                model_revision=metadata["model_revision"],
                options=tuple(sorted((str(k), str(v)) for k, v in metadata.get("options", {}).items())),
            )
        except (ContractValidationError, KeyError, ValueError) as exc:
            return _error_response(str(exc), _safe_ownership(metadata), envelope_wire=envelope_wire)
        response = await self.reconstruct(parsed)
        return _wilor_response_to_envelope_wire(response) if envelope_wire else _wilor_response_to_multipart_wire(response)


def _unwrap_gateway_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize generic gateway metadata while preserving legacy multipart input."""
    nested = metadata.get("metadata")
    if not isinstance(nested, dict) or "ownership" not in metadata:
        return metadata
    payload = dict(nested)
    for name in ("ownership", "model_revision", "spatial"):
        if name in metadata:
            payload[name] = metadata[name]
    return payload


def _parse_envelope_payloads(envelope: BinaryEnvelope) -> tuple[dict[str, Any], dict[str, ArrayPart]]:
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


def _ownership(metadata: dict[str, Any]):
    from ego_annotation.serving.contracts import Ownership

    own = metadata.get("ownership")
    if not own:
        raise ContractValidationError("ownership is required")
    return Ownership.from_mapping(own)


def _safe_ownership(metadata: dict[str, Any]) -> Any:
    try:
        return _ownership(metadata)
    except ContractValidationError:
        return None


def _envelope_asgi_response(metadata: dict[str, Any], arrays: dict[str, ArrayPart]) -> Response:
    parts: dict[str, ArrayPart] = {
        "metadata": (json.dumps(metadata, separators=(",", ":")).encode("utf-8"), (), "application/json"),
    }
    parts.update(arrays)
    envelope = build_binary_envelope(parts)
    # Starlette's send loop accepts bytes rather than memoryviews. Keep tensor
    # vectors independently framed and materialize only at the ASGI boundary.
    return StreamingResponse(
        (bytes(chunk) for chunk in binary_envelope_iovecs(envelope)),
        media_type=BINARY_ENVELOPE_CONTENT_TYPE,
    )


def _error_response(message: str, ownership: Any = None, *, envelope_wire: bool = False) -> Response:
    from ego_annotation.serving.contracts import Ownership

    wire = ownership.to_wire() if isinstance(ownership, Ownership) else None
    metadata = {"error": {"code": "validation", "message": message, "retryable": False}, "ownership": wire}
    return _envelope_asgi_response(metadata, {}) if envelope_wire else multipart_asgi_response(metadata, {})


def _hands_arrays(response: HandsDetectResponse) -> dict[str, ArrayPart]:
    result = response.result
    assert result is not None
    det = result.detection
    return {
        "boxes": (det.boxes.data, det.boxes.shape, det.boxes.dtype),
        "scores": (det.scores.data, det.scores.shape, det.scores.dtype),
        "sides": (det.sides.data, det.sides.shape, det.sides.dtype),
        "masks": (det.masks.data, det.masks.shape, det.masks.dtype),
        "visibility": (det.visibility.data, det.visibility.shape, det.visibility.dtype),
        "uncertainty": (det.uncertainty.data, det.uncertainty.shape, det.uncertainty.dtype),
    }


def _hands_response_to_multipart_wire(response: HandsDetectResponse) -> Response:
    if response.error is not None:
        metadata = {"error": response.error.to_wire(), "ownership": response.ownership.to_wire()}
        return multipart_asgi_response(metadata, {})
    result = response.result
    assert result is not None
    metadata = {"result": result.to_wire(), "ownership": response.ownership.to_wire()}
    arrays = {name: (bytes(data), shape, dtype) for name, (data, shape, dtype) in _hands_arrays(response).items()}
    return multipart_asgi_response(metadata, arrays)


def _hands_response_to_envelope_wire(response: HandsDetectResponse) -> Response:
    if response.error is not None:
        return _envelope_asgi_response({"error": response.error.to_wire(), "ownership": response.ownership.to_wire()}, {})
    result = response.result
    assert result is not None
    return _envelope_asgi_response(
        {"result": result.to_wire(), "ownership": response.ownership.to_wire()}, _hands_arrays(response)
    )


def _wilor_arrays(response: WiLoRReconstructResponse) -> dict[str, ArrayPart]:
    result = response.result
    assert result is not None
    mano = result.mano
    return {
        "global_orient": (mano.global_orient.data, mano.global_orient.shape, mano.global_orient.dtype),
        "hand_pose": (mano.hand_pose.data, mano.hand_pose.shape, mano.hand_pose.dtype),
        "betas": (mano.betas.data, mano.betas.shape, mano.betas.dtype),
        "vertices": (mano.vertices.data, mano.vertices.shape, mano.vertices.dtype),
        "joints": (mano.joints.data, mano.joints.shape, mano.joints.dtype),
        "cam_t_full": (mano.cam_t_full.data, mano.cam_t_full.shape, mano.cam_t_full.dtype),
        "pred_cam": (mano.pred_cam.data, mano.pred_cam.shape, mano.pred_cam.dtype),
        "keypoints_2d": (mano.keypoints_2d.data, mano.keypoints_2d.shape, mano.keypoints_2d.dtype),
        "confidence": (mano.confidence.data, mano.confidence.shape, mano.confidence.dtype),
        "uncertainty": (mano.uncertainty.data, mano.uncertainty.shape, mano.uncertainty.dtype),
    }


def _wilor_response_to_multipart_wire(response: WiLoRReconstructResponse) -> Response:
    if response.error is not None:
        metadata = {"error": response.error.to_wire(), "ownership": response.ownership.to_wire()}
        return multipart_asgi_response(metadata, {})
    result = response.result
    assert result is not None
    metadata = {"result": result.to_wire(), "ownership": response.ownership.to_wire()}
    arrays = {name: (bytes(data), shape, dtype) for name, (data, shape, dtype) in _wilor_arrays(response).items()}
    return multipart_asgi_response(metadata, arrays)


def _wilor_response_to_envelope_wire(response: WiLoRReconstructResponse) -> Response:
    if response.error is not None:
        return _envelope_asgi_response({"error": response.error.to_wire(), "ownership": response.ownership.to_wire()}, {})
    result = response.result
    assert result is not None
    return _envelope_asgi_response(
        {"result": result.to_wire(), "ownership": response.ownership.to_wire()}, _wilor_arrays(response)
    )


hands_app: Any = HandsWiLoRDeployment.bind()
