"""Canonical gateway mappings for stateful DROID session transport.

Both multipart and binary-envelope framings preserve these exact named frame input
parts: ``rgb`` ``[H,W,3] uint8`` plus optional ``static_confidence_mask`` and
``depth_m``. Create and finalize are metadata-only, but retain the same ownership,
model revision, session camera, image-grid, and option contracts. Finalize responses
always use exactly the CameraState parts ``T_world_camera``, ``T_camera_world``,
``intrinsics_px``, and ``disparities``; their shape, finite-value, and inverse checks
remain owned by :func:`parse_droid_finalize_response`.
"""
from __future__ import annotations

from typing import Any

from ego_annotation.serving.contracts import (
    DroidCreateSessionRequest,
    DroidFinalizeRequest,
    DroidFrameRequest,
)
from ego_annotation.serving.gateway import GatewayBinaryPart, GatewayRequest
from ego_annotation.serving.router import ModelApiName


def _tensor_part(name: str, tensor: Any) -> GatewayBinaryPart:
    return GatewayBinaryPart(name=name, data=tensor.data, shape=tensor.shape, dtype=tensor.dtype)


def _metadata_without_transport_fields(request: Any, *fields: str) -> dict[str, Any]:
    metadata = request.to_wire()
    for field in ("ownership", "model_revision", *fields):
        metadata.pop(field, None)
    return metadata


def droid_create_session_gateway_request(request: DroidCreateSessionRequest) -> GatewayRequest:
    """Map a typed session-creation request to metadata-only gateway transport."""
    return GatewayRequest(
        api_name=ModelApiName.DROID_CREATE_SESSION,
        ownership=request.ownership,
        parts=(),
        metadata=_metadata_without_transport_fields(request),
        model_revision=request.model_revision,
    )


def droid_push_frame_gateway_request(request: DroidFrameRequest) -> GatewayRequest:
    """Map one typed DROID frame and its optional measurement arrays to named parts."""
    parts = [_tensor_part("rgb", request.rgb)]
    if request.static_confidence_mask is not None:
        parts.append(_tensor_part("static_confidence_mask", request.static_confidence_mask))
    if request.depth_m is not None:
        parts.append(_tensor_part("depth_m", request.depth_m))
    return GatewayRequest(
        api_name=ModelApiName.DROID_PUSH_FRAME,
        ownership=request.ownership,
        parts=tuple(parts),
        metadata=_metadata_without_transport_fields(request, "rgb", "static_confidence_mask", "depth_m"),
        model_revision=request.model_revision,
    )


def droid_finalize_gateway_request(request: DroidFinalizeRequest) -> GatewayRequest:
    """Map a typed terminal request to metadata-only gateway transport."""
    return GatewayRequest(
        api_name=ModelApiName.DROID_FINALIZE,
        ownership=request.ownership,
        parts=(),
        metadata=_metadata_without_transport_fields(request),
        model_revision=request.model_revision,
    )


__all__ = [
    "droid_create_session_gateway_request",
    "droid_finalize_gateway_request",
    "droid_push_frame_gateway_request",
]
