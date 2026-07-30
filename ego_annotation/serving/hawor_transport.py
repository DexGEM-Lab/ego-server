"""Canonical binary-part mapping for GPU3 HaWoR wire transports.

The GPU3 APIs have nested typed metadata, but their dense payloads stay in named
binary parts in both multipart and binary-envelope framing.  This module is the
single mapping shared by callers and deployment-boundary tests:

``hawor.infer_tracks``
    ``crop_batch`` `[16,3,256,256] float32`, plus optional
    ``droid_poses`` `[T,4,4] float32` and ``droid_timestamps`` `[T] float64`.
    Track ID, side, crop transforms, observations (including occlusion/mask
    support), UniDepth evidence, and ownership stay in JSON metadata.

``hawor_infiller.fill``
    ``droid_poses`` `[T,4,4] float32` and ``droid_timestamps`` `[T] float64`.
    The structurally nested two-hand sequence (frame IDs, side, MANO parameters,
    observation flags, and uncertainty) stays in JSON metadata because it is the
    typed sequence contract rather than an opaque tensor substitute.

Response array names deliberately mirror the established multipart response:
HaWoR returns MANO ``root_orient``, ``hand_pose``, ``trans``, ``betas``,
``vertices``, ``joints``, ``observed``, ``uncertainty``, and optional ``world_lift``;
the infiller returns the same state fields plus ``inferred`` and ``timestamps_s``.
"""
from __future__ import annotations

from typing import Any

from ego_annotation.serving.gateway import GatewayBinaryPart, GatewayRequest
from ego_annotation.serving.hawor_contracts import HandSequenceRequest, TrackChunkRequest
from ego_annotation.serving.router import ModelApiName


def _tensor_part(name: str, tensor: Any) -> GatewayBinaryPart:
    return GatewayBinaryPart(name=name, data=tensor.data, shape=tensor.shape, dtype=tensor.dtype)


def track_chunk_gateway_request(request: TrackChunkRequest) -> GatewayRequest:
    """Expose one typed HaWoR track request through the generic gateway boundary."""
    metadata = request.to_wire()
    metadata.pop("ownership", None)
    metadata.pop("model_revision", None)
    metadata.pop("crop_batch", None)
    parts = [_tensor_part("crop_batch", request.crop_batch)]
    if request.droid_evidence is not None:
        parts.extend((
            _tensor_part("droid_poses", request.droid_evidence.poses_world_from_camera),
            _tensor_part("droid_timestamps", request.droid_evidence.timestamps_s),
        ))
    return GatewayRequest(
        api_name=ModelApiName.HAWOR_INFER_TRACKS,
        ownership=request.ownership,
        parts=tuple(parts),
        metadata=metadata,
        model_revision=request.model_revision,
    )


def infiller_gateway_request(request: HandSequenceRequest) -> GatewayRequest:
    """Expose one typed infiller window through the generic gateway boundary."""
    metadata = request.to_wire()
    metadata.pop("ownership", None)
    metadata.pop("model_revision", None)
    return GatewayRequest(
        api_name=ModelApiName.HAWOR_INFILLER_FILL,
        ownership=request.ownership,
        parts=(
            _tensor_part("droid_poses", request.droid_evidence.poses_world_from_camera),
            _tensor_part("droid_timestamps", request.droid_evidence.timestamps_s),
        ),
        metadata=metadata,
        model_revision=request.model_revision,
    )


__all__ = ["infiller_gateway_request", "track_chunk_gateway_request"]
