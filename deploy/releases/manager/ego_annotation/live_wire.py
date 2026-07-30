"""Frozen-3572551 route wire adapters.

This is the live backend wire. It deliberately does not use the internal generic
``ego.annotation.api.v1`` envelope: each route emits the exact metadata keys and
binary part names consumed by the frozen service parser.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ego_annotation.multipart import MultipartMessage, decode_raw_multipart, encode_raw_multipart
from ego_annotation.scripted.contracts import AlgorithmRequest
from ego_annotation.typed_contracts import (
    BinaryAsset,
    CosmosInput,
    DroidCreateInput,
    DroidFinalizeInput,
    DroidPushInput,
    HandsInput,
    HaworTrackInput,
    InfillerInput,
    Ownership,
    TypedContractError,
    TypedTensor,
    UniDepthInput,
    WiLoRInput,
)


class LiveWireError(TypedContractError):
    """Frozen route request/response mismatch."""


@dataclass(frozen=True)
class LiveRouteResponse:
    ownership: Mapping[str, object]
    result: Mapping[str, object] | None
    route_metadata: Mapping[str, object]
    parts: Mapping[str, Any]


_GENERIC_LIVE_KEYS = {"protocol", "status", "request_id", "algorithm_id", "case_id", "item_id", "source_id", "input"}
_RESPONSE_PARTS: Mapping[str, frozenset[str]] = {
    "unidepth.infer": frozenset({"depth_m", "K_px", "confidence"}),
    "hands.detect": frozenset({"boxes", "scores", "sides", "visibility", "uncertainty"}),
    "wilor.reconstruct": frozenset({"global_orient", "hand_pose", "betas", "vertices", "joints", "cam_t_full", "pred_cam", "keypoints_2d", "confidence", "uncertainty"}),
    "droid.create_session": frozenset(),
    "droid.push_frame": frozenset(),
    "droid.finalize": frozenset({"T_world_camera", "T_camera_world", "intrinsics_px", "disparities"}),
    "hawor.infer_tracks": frozenset({"root_orient", "hand_pose", "trans", "betas", "vertices", "joints", "observed", "uncertainty"}),
    "hawor_infiller.fill": frozenset({"root_orient", "hand_pose", "trans", "betas", "observed", "inferred", "uncertainty", "timestamps_s"}),
    "cosmos3.reason": frozenset(),
}
_RESPONSE_OPTIONAL_PARTS: Mapping[str, frozenset[str]] = {"hawor.infer_tracks": frozenset({"world_lift"})}

# Mirrors the frozen DroidSessionOptions contract in the resident service.
# Only these model parameters may cross the pipeline boundary.  Lifecycle
# annotations (for example ``attempt``) stay local to the pipeline.
_DROID_SESSION_OPTION_TYPES: Mapping[str, type[object]] = {
    "buffer": int,
    "filter_thresh": float,
    "warmup": int,
    "keyframe_thresh": float,
    "frontend_thresh": float,
    "frontend_window": int,
    "frontend_radius": int,
    "frontend_nms": int,
    "backend_thresh": float,
    "backend_radius": int,
    "backend_nms": int,
    "upsample": bool,
    "beta": float,
    "stereo": bool,
}
_DROID_ORCHESTRATION_OPTION_KEYS = frozenset({"attempt", "bounded_lower_filter_retry"})


def _droid_create_options(options: Mapping[str, object]) -> dict[str, object]:
    """Select only frozen, exactly typed DROID session options for create."""

    forwarded: dict[str, object] = {}
    for key, value in options.items():
        if key in _DROID_ORCHESTRATION_OPTION_KEYS:
            continue
        expected_type = _DROID_SESSION_OPTION_TYPES.get(key)
        if expected_type is None:
            continue
        # ``bool`` is an ``int`` subclass, so isinstance would let an invalid
        # boolean capacity/threshold through the frozen service boundary.
        if type(value) is not expected_type:
            raise LiveWireError(
                f"droid.create_session option {key!r} must have frozen type "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
        forwarded[key] = value
    return forwarded


def _ownership_wire(value: Ownership) -> dict[str, object]:
    """Adapt internal ownership to the frozen Ownership.to_wire shape."""

    return {
        "request_id": f"{value.case_id}:{value.item_id}:{value.scope}",
        "job_id": value.case_id,
        "item_id": value.item_id,
        "stage_id": value.scope,
        "source_id": value.source_id,
        "schema_version": "ego.model-service.v1",
        "source_timestamp_s": None,
        "submitted_at": "1970-01-01T00:00:00Z",
    }


def _spatial_wire(value: Any, *, model_height: int, model_width: int, K_px: object | None = None) -> dict[str, object]:
    matrix = np.asarray(value.pixel_to_source, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    return {
        "source_size": {"width": int(value.source_width_px), "height": int(value.source_height_px)},
        "model_size": {"width": int(model_width), "height": int(model_height)},
        "color_space": "RGB",
        "pixel_transform": {
            "source_to_model": inverse.tolist(),
            "model_to_source": matrix.tolist(),
            "resize_mode": "declared_transform",
            "crop_xywh": None,
            "pad_ltrb": None,
        },
        "K_px": K_px,
    }


def _source_camera_wire(K_px: object, *, width: int, height: int) -> dict[str, object]:
    K = np.asarray(K_px, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise LiveWireError("HaWoR source calibration requires a finite positive-focal 3x3 K")
    if width <= 0 or height <= 0:
        raise LiveWireError("HaWoR source calibration requires positive source geometry")
    return {
        "K_px": K.tolist(),
        "img_focal": float(np.sqrt(K[0, 0] * K[1, 1])),
        "img_center": [float(K[0, 2]), float(K[1, 2])],
        "source_size": {"width": int(width), "height": int(height)},
        "metric_scale": 1.0,
        "source": "unidepth",
    }


def _part_descriptor(value: object) -> dict[str, object]:
    if isinstance(value, TypedTensor):
        return {"shape": list(value.shape), "dtype": value.dtype}
    if hasattr(value, "shape") and hasattr(value, "spec"):
        spec = value.spec
        return {"shape": list(value.shape), "dtype": "float32", "field_name": spec.get("field_name"), "semantic_tag": spec.get("semantic_tag")}
    raise LiveWireError(f"cannot make frozen tensor descriptor for {type(value).__name__}")


def _single_frame(value: TypedTensor, name: str) -> TypedTensor:
    if len(value.shape) != 4 or value.shape[0] != 1:
        raise LiveWireError(f"{name} live route requires one frame per request; batching stays in caller work metadata")
    return TypedTensor(value.array[0], units=value.units, coordinate_frame=value.coordinate_frame, tensor_index_order=value.tensor_index_order[1:], semantic_tag=value.semantic_tag, provenance=value.provenance, pixel_transform=value.pixel_transform)


def _single_crop(value: TypedTensor) -> TypedTensor:
    if value.shape != (1, 3, 256, 256):
        raise LiveWireError("WiLoR live route requires one [3,256,256] crop per request")
    return TypedTensor(value.array[0], units=value.units, coordinate_frame=value.coordinate_frame, tensor_index_order=value.tensor_index_order[1:], semantic_tag=value.semantic_tag, provenance=value.provenance, pixel_transform=value.pixel_transform)


def _request_parts_and_metadata(request: AlgorithmRequest[Any]) -> tuple[dict[str, object], dict[str, object]]:
    value = request.input
    ownership = getattr(value, "ownership", None)
    if not isinstance(ownership, Ownership):
        raise LiveWireError("live request has no typed ownership")
    owner_wire = _ownership_wire(ownership)
    if isinstance(value, UniDepthInput):
        rgb = _single_frame(value.rgb_batch, "UniDepth")
        metadata = {
            "ownership": owner_wire,
            "spatial": _spatial_wire(value.spatial, model_height=rgb.shape[0], model_width=rgb.shape[1]),
            "model_revision": request.model_revision,
            "options": {},
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": rgb.dtype,
        }
        return metadata, {"rgb": rgb}
    if isinstance(value, HandsInput):
        rgb = _single_frame(value.rgb_batch, "Hands")
        metadata = {
            "ownership": owner_wire,
            "spatial": _spatial_wire(value.spatial, model_height=rgb.shape[0], model_width=rgb.shape[1]),
            "model_revision": request.model_revision,
            "options": {},
        }
        return metadata, {"rgb": rgb}
    if isinstance(value, WiLoRInput):
        crop = _single_crop(value.crop_batch)
        transform = value.crop_transforms[0]
        metadata = {
            "ownership": owner_wire,
            "handedness": 0 if transform.side.value == "left" else 1,
            "box_center": list(transform.center_xy_px),
            "box_size": transform.size_px,
            "img_size": [transform.source_to_crop.source_width_px, transform.source_to_crop.source_height_px],
            "source_K_px": value.source_K_px.array.tolist(),
            "model_revision": request.model_revision,
            "options": {},
        }
        return metadata, {"crop": crop}
    if isinstance(value, DroidCreateInput):
        source_size = {
            "width": value.p_source_to_droid_input.source_width_px,
            "height": value.p_source_to_droid_input.source_height_px,
        }
        metadata = {
            "ownership": owner_wire,
            "camera": {
                "intrinsics": list(value.K_droid_input),
                "source_size": source_size,
                "pixel_transform": _spatial_wire(value.p_source_to_droid_input, model_height=source_size["height"], model_width=source_size["width"])["pixel_transform"],
                "K_px": [[value.K_droid_input[0], 0.0, value.K_droid_input[2]], [0.0, value.K_droid_input[1], value.K_droid_input[3]], [0.0, 0.0, 1.0]],
            },
            "image_shape": {"height": value.model_shape_yx[0] * 8, "width": value.model_shape_yx[1] * 8},
            "model_revision": request.model_revision,
            "options": _droid_create_options(request.options),
        }
        return metadata, {}
    if isinstance(value, DroidPushInput):
        metadata = {
            "ownership": owner_wire,
            "session_id": value.session_id,
            "frame_id": str(value.frame_index),
            "source_timestamp_s": value.timestamp_s,
            "rgb": _part_descriptor(value.rgb),
            "static_confidence_mask": _part_descriptor(value.static_confidence_mask) if value.static_confidence_mask is not None else None,
            "depth_m": _part_descriptor(value.native_sensor_depth_abi_payload_m),
            "model_revision": request.model_revision,
        }
        parts: dict[str, object] = {"rgb": value.rgb, "depth_m": value.native_sensor_depth_abi_payload_m}
        if value.static_confidence_mask is not None:
            parts["static_confidence_mask"] = value.static_confidence_mask
        return metadata, parts
    if isinstance(value, DroidFinalizeInput):
        return {"ownership": owner_wire, "session_id": value.session_id, "model_revision": request.model_revision}, {}
    if isinstance(value, HaworTrackInput):
        if value.crop_batch.shape[0] != 1:
            raise LiveWireError("HaWoR route receives one native 16-frame chunk per request; batch cap remains in work metadata")
        if len(value.crop_transforms) != 16 or len(value.observation_records) != 16:
            raise LiveWireError("HaWoR frozen parser requires 16 crop transforms and 16 typed observations per chunk")
        sides = {transform.side for transform in value.crop_transforms}
        observation_sides = {observation.side for observation in value.observation_records}
        if len(sides) != 1 or observation_sides != sides:
            raise LiveWireError("HaWoR track crops and observations must carry one uniform actual side")
        source_sizes = {(transform.source_to_crop.source_width_px, transform.source_to_crop.source_height_px) for transform in value.crop_transforms}
        if len(source_sizes) != 1:
            raise LiveWireError("HaWoR track crops must preserve one source image geometry")
        side = next(iter(sides))
        source_width, source_height = next(iter(source_sizes))
        source_camera = _source_camera_wire(value.unidepth_K_px.array, width=source_width, height=source_height)
        droid_metadata = {
            "metric_scale": 1.0,
            "scale_residual": 0.0,
            "scale_confidence": 1.0,
            "source": "droid_typed_evidence",
        }
        metadata = {
            "ownership": owner_wire,
            "track_id": "typed-track",
            "side": side.value,
            "crop_batch": _part_descriptor(value.crop_batch),
            "crop_transforms": [transform.to_wire(value.unidepth_K_px.array) for transform in value.crop_transforms],
            "observations": [observation.to_wire() for observation in value.observation_records],
            "unidepth": source_camera,
            "droid_evidence": droid_metadata,
            "model_revision": request.model_revision,
            "options": {},
        }
        return metadata, {
            "crop_batch": TypedTensor(value.crop_batch.array[0], units=value.crop_batch.units, coordinate_frame=value.crop_batch.coordinate_frame, tensor_index_order="tcyx", semantic_tag=value.crop_batch.semantic_tag, provenance=value.crop_batch.provenance),
            "droid_poses": value.droid_poses_world_camera,
            "droid_timestamps": value.droid_timestamps_s,
        }
    if isinstance(value, InfillerInput):
        frames = getattr(value, "frames", ())
        if not frames:
            raise LiveWireError("infiller live adapter requires typed frame records; it cannot fabricate frames from a model window")
        metadata = {
            "ownership": owner_wire,
            "window_id": "typed-window",
            "frames": [frame.to_wire() for frame in frames],
            "droid_evidence": {"metric_scale": 1.0, "scale_residual": 0.0, "scale_confidence": 1.0, "source": "droid_typed_evidence"},
            "unidepth": {"K_px": value.unidepth_scale_K.array.tolist(), "img_focal": float(value.unidepth_scale_K.array[0, 0]), "img_center": [0.0, 0.0], "source_size": {"width": 1, "height": 1}, "metric_scale": 1.0, "source": "unidepth"},
            "model_revision": request.model_revision,
            "options": {},
        }
        return metadata, {"droid_poses": value.droid_poses_world_camera, "droid_timestamps": value.timestamps_s}
    if isinstance(value, CosmosInput):
        metadata = {
            "ownership": owner_wire,
            "prompt": value.prompt,
            "messages": [{"role": message.role, "content": message.content} for message in value.messages],
            "generation": value.generation.to_wire(),
        }
        return metadata, {f"media_{index}": media for index, media in enumerate(value.media)}
    raise LiveWireError(f"no frozen route adapter for {type(value).__name__}")


def encode_live_request(request: AlgorithmRequest[Any]) -> tuple[bytes, str, dict[str, object], tuple[str, ...]]:
    metadata, parts = _request_parts_and_metadata(request)
    forbidden = _GENERIC_LIVE_KEYS.intersection(metadata)
    if forbidden:
        raise LiveWireError(f"generic live envelope keys are forbidden: {sorted(forbidden)}")
    body, content_type = encode_raw_multipart(metadata, parts)
    return body, content_type, metadata, tuple(parts)


def _ownership_matches(expected: Ownership, actual: object) -> bool:
    if not isinstance(actual, Mapping):
        return False
    expected_wire = _ownership_wire(expected)
    return dict(actual) == expected_wire


def _find_tensor_descriptor(value: object, name: str) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    candidate = value.get(name)
    if isinstance(candidate, Mapping) and "shape" in candidate and "dtype" in candidate:
        return candidate
    for child in value.values():
        found = _find_tensor_descriptor(child, name)
        if found is not None:
            return found
    return None


def _validate_response_parts(stage_id: str, result: Mapping[str, object], parts: Mapping[str, Any]) -> None:
    expected = _RESPONSE_PARTS[stage_id]
    optional = _RESPONSE_OPTIONAL_PARTS.get(stage_id, frozenset())
    actual = set(parts)
    if not set(expected).issubset(actual) or not actual.issubset(set(expected | optional)):
        raise LiveWireError(f"{stage_id} response parts {sorted(parts)} do not match frozen required/optional route parts")
    for name, part in parts.items():
        descriptor = part.descriptor
        metadata_tensor = _find_tensor_descriptor(result, name)
        if metadata_tensor is None:
            raise LiveWireError(f"{stage_id} result metadata lacks tensor descriptor {name}")
        if list(metadata_tensor.get("shape", ())) != list(descriptor.get("shape", ())) or str(metadata_tensor.get("dtype")) != str(descriptor.get("dtype")):
            raise LiveWireError(f"{stage_id} tensor {name} metadata/header shape or dtype mismatch")
        encoded = metadata_tensor.get("data_b64")
        if isinstance(encoded, str):
            try:
                metadata_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
            except ValueError as exc:
                raise LiveWireError(f"{stage_id} tensor {name} metadata base64 is invalid") from exc
            if hashlib.sha256(metadata_bytes).hexdigest() != descriptor.get("payload_sha256") or metadata_bytes != part.data:
                raise LiveWireError(f"{stage_id} tensor {name} digest/bytes mismatch")
            continue
        # Resident services transmit tensors as authenticated multipart binary
        # parts and keep result metadata compact (shape/dtype only).  The frozen
        # decoder has already checked the X-Part-Sha256 header; retain an
        # explicit descriptor check here before accepting that representation.
        digest = descriptor.get("payload_sha256")
        byte_length = descriptor.get("byte_length")
        if (
            not isinstance(digest, str)
            or not isinstance(byte_length, int)
            or byte_length != len(part.data)
            or hashlib.sha256(part.data).hexdigest() != digest
        ):
            raise LiveWireError(f"{stage_id} tensor {name} multipart payload digest/length mismatch")


def decode_live_response(stage_id: str, expected_ownership: Ownership, body: bytes, content_type: str) -> LiveRouteResponse:
    if content_type.lower().startswith("application/json"):
        try:
            metadata = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveWireError("route JSON response is invalid") from exc
        if not isinstance(metadata, Mapping):
            raise LiveWireError("route JSON response must be an object")
        if stage_id != "cosmos3.reason" and isinstance(metadata.get("metadata"), Mapping):
            metadata = metadata["metadata"]
        parts: Mapping[str, Any] = {}
    else:
        message = decode_raw_multipart(body, content_type)
        metadata = message.metadata
        parts = message.parts
    if stage_id == "cosmos3.reason":
        expected_top_level = {"ownership", "result", "error"}
        if type(metadata) is not dict or set(metadata) != expected_top_level:
            raise LiveWireError("cosmos3.reason response top-level fields do not match the frozen schema")
        if metadata["error"] is not None:
            raise LiveWireError(f"{stage_id} route returned an explicit error: {metadata['error']}")
        if type(metadata["result"]) is not dict:
            raise LiveWireError("cosmos3.reason response result must be an object")
    forbidden_response_keys = set(metadata).intersection(_GENERIC_LIVE_KEYS)
    if stage_id == "droid.push_frame":
        forbidden_response_keys.discard("status")
    if forbidden_response_keys:
        raise LiveWireError(f"live response contains generic envelope keys: {sorted(forbidden_response_keys)}")
    actual_ownership = metadata.get("ownership")
    if not _ownership_matches(expected_ownership, actual_ownership):
        raise LiveWireError("route response ownership mismatch")
    error = metadata.get("error")
    if error is not None:
        raise LiveWireError(f"{stage_id} route returned an explicit error: {error}")
    if stage_id == "droid.finalize":
        result = metadata.get("camera_state")
    elif stage_id in {"droid.create_session", "droid.push_frame"}:
        result = metadata
    else:
        result = metadata.get("result")
    if not isinstance(result, Mapping):
        raise LiveWireError(f"{stage_id} response has no route-specific result")
    _validate_response_parts(stage_id, result, parts)
    return LiveRouteResponse(ownership=dict(actual_ownership), result=dict(result), route_metadata=metadata, parts=parts)


__all__ = ["LiveRouteResponse", "LiveWireError", "decode_live_response", "encode_live_request"]
