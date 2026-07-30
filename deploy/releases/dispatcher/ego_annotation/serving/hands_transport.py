"""Canonical binary-part mapping for GPU1 Hands and WiLoR transports.

Both HTTP wire formats carry exactly these model-native input vectors:

``hands.detect``
    ``rgb`` ``[H,W,3] uint8``; source spatial/camera metadata remains typed JSON.

``wilor.reconstruct``
    ``crop`` ``[3,256,256] float32``; handedness and crop-to-source camera
    metadata remain typed JSON.

The response vectors are assembled by :mod:`hands_deployment` with the same part
names as the established multipart API: detector/SAM2 arrays for Hands and MANO
parameters, meshes, joints, camera state, projections, confidence, and uncertainty
for WiLoR.
"""
from __future__ import annotations

from typing import Any

from ego_annotation.serving.contracts import HandsDetectRequest, WiLoRReconstructRequest
from ego_annotation.serving.gateway import GatewayBinaryPart, GatewayRequest
from ego_annotation.serving.router import ModelApiName


def _tensor_part(name: str, tensor: Any) -> GatewayBinaryPart:
    return GatewayBinaryPart(name=name, data=tensor.data, shape=tensor.shape, dtype=tensor.dtype)


def hands_detect_gateway_request(request: HandsDetectRequest) -> GatewayRequest:
    """Expose one typed detector/mask request through the generic gateway."""
    return GatewayRequest(
        api_name=ModelApiName.HANDS_DETECT,
        ownership=request.ownership,
        parts=(_tensor_part("rgb", request.rgb),),
        spatial=request.spatial,
        metadata={"options": dict(request.options)},
        model_revision=request.model_revision,
    )


def wilor_reconstruct_gateway_request(request: WiLoRReconstructRequest) -> GatewayRequest:
    """Expose one typed normalized WiLoR crop through the generic gateway."""
    metadata = request.to_wire()
    metadata.pop("ownership", None)
    metadata.pop("model_revision", None)
    metadata.pop("crop", None)
    return GatewayRequest(
        api_name=ModelApiName.WILOR_RECONSTRUCT,
        ownership=request.ownership,
        parts=(_tensor_part("crop", request.crop),),
        metadata=metadata,
        model_revision=request.model_revision,
    )


__all__ = ["hands_detect_gateway_request", "wilor_reconstruct_gateway_request"]
