"""Transport-neutral typed contracts for the single-item algorithm DAG.

The classes in this module are the only payloads accepted by the remote backend.
They deliberately carry physical tensor semantics (shape, dtype, units, frame,
index order, transforms, and provenance) instead of using untyped dictionaries.
"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, TypeVar, Union, get_args, get_origin, get_type_hints

import numpy as np

from ego_annotation.scripted.contracts import (
    AlgorithmRequest,
    AlgorithmResult,
    ContractError,
    FrameTimelineMetadata,
    NativeBatchTrace,
    NativeWorkDescription,
    StageMetadata,
)
from ego_annotation.scripted.droid_rgbd import DroidNativeSensorDepthAbiPayload


class TypedContractError(ContractError):
    """Raised when a typed stage contract is malformed."""


class Lane(str, Enum):
    PHYSICAL = "physical"
    SEMANTIC = "semantic"


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Ownership:
    """Logical ownership of a stage payload, independent of transport."""

    case_id: str
    item_id: str
    source_id: str
    owner: str
    scope: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.case_id, self.item_id, self.source_id, self.owner, self.scope)):
            raise TypedContractError("ownership fields are required")
        if "://" in self.source_id:
            raise TypedContractError("source_id must identify the source, not a URL")

    def to_wire(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "item_id": self.item_id,
            "source_id": self.source_id,
            "owner": self.owner,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class SpatialTransform:
    """A complete pixel-grid transform from a declared grid to source pixels."""

    grid_id: str
    source_width_px: int
    source_height_px: int
    pixel_to_source: tuple[tuple[float, float, float], ...]
    coordinate_frame: str

    def __post_init__(self) -> None:
        if not self.grid_id or self.source_width_px <= 0 or self.source_height_px <= 0:
            raise TypedContractError("spatial grid and source dimensions are required")
        matrix = np.asarray(self.pixel_to_source, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise TypedContractError("pixel_to_source must be an invertible finite 3x3 matrix")
        if not self.coordinate_frame:
            raise TypedContractError("coordinate_frame is required")

    def to_wire(self) -> dict[str, object]:
        return {
            "grid_id": self.grid_id,
            "source_width_px": self.source_width_px,
            "source_height_px": self.source_height_px,
            "pixel_to_source": [list(row) for row in self.pixel_to_source],
            "coordinate_frame": self.coordinate_frame,
        }


@dataclass(frozen=True)
class TypedTensor:
    """Immutable tensor with physical semantics and a content digest."""

    array: np.ndarray
    units: str
    coordinate_frame: str
    tensor_index_order: str
    semantic_tag: str
    provenance: Mapping[str, object]
    pixel_transform: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.array, np.ndarray) or self.array.ndim == 0:
            raise TypedContractError("tensor must be a non-scalar numpy array")
        if self.array.dtype.kind not in "biuf":
            raise TypedContractError(f"unsupported tensor dtype {self.array.dtype}")
        if not all(isinstance(value, str) and value for value in (self.units, self.coordinate_frame, self.tensor_index_order, self.semantic_tag)):
            raise TypedContractError("tensor units, frame, index order, and semantic tag are required")
        if any(int(dim) < 0 for dim in self.array.shape):
            raise TypedContractError("tensor dimensions must be non-negative")
        if self.pixel_transform is not None:
            matrix = np.asarray(self.pixel_transform, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise TypedContractError("tensor pixel_transform must be finite 3x3")
        copied = np.ascontiguousarray(self.array)
        copied.setflags(write=False)
        object.__setattr__(self, "array", copied)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def dtype(self) -> str:
        return self.array.dtype.name

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.array.shape)

    @property
    def wire_dtype(self) -> str:
        if self.array.dtype.name == "uint8":
            return "|u1"
        if self.array.dtype.name == "bool":
            return "|b1"
        if self.array.dtype.name == "float32":
            return "<f4"
        if self.array.dtype.name == "float64":
            return "<f8"
        if self.array.dtype.name == "int64":
            return "<i8"
        raise TypedContractError(f"no canonical wire dtype for {self.array.dtype.name}")

    @property
    def canonical_bytes(self) -> bytes:
        return np.ascontiguousarray(self.array.astype(np.dtype(self.wire_dtype), copy=False)).tobytes(order="C")

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def canonical_tensor_digest(self) -> str:
        spec = {
            "shape": list(self.shape), "dtype": self.dtype, "wire_dtype": self.wire_dtype,
            "units": self.units, "coordinate_frame": self.coordinate_frame,
            "tensor_index_order": self.tensor_index_order, "semantic_tag": self.semantic_tag,
            "provenance": dict(self.provenance),
            "pixel_transform": [list(row) for row in self.pixel_transform] if self.pixel_transform is not None else None,
        }
        encoded = json.dumps(spec, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded + b"\x00" + self.canonical_bytes).hexdigest()

    def descriptor(self, part_name: str) -> dict[str, object]:
        return {
            "name": part_name,
            "kind": "tensor",
            "shape": list(self.shape),
            "dtype": self.dtype,
            "wire_dtype": self.wire_dtype,
            "byte_length": len(self.canonical_bytes),
            "payload_sha256": self.payload_sha256,
            "canonical_tensor_digest": self.canonical_tensor_digest,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "tensor_index_order": self.tensor_index_order,
            "semantic_tag": self.semantic_tag,
            "provenance": dict(self.provenance),
            "pixel_transform": [list(row) for row in self.pixel_transform] if self.pixel_transform is not None else None,
        }


@dataclass(frozen=True)
class BinaryAsset:
    """Source-backed opaque media bytes (used by Cosmos)."""

    data: bytes
    media_type: str
    source_artifact_id: str
    source_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise TypedContractError("binary media must contain non-empty bytes")
        if not isinstance(self.media_type, str) or not self.media_type or not isinstance(self.source_artifact_id, str) or not self.source_artifact_id:
            raise TypedContractError("media type and source artifact must be non-empty text")
        if not isinstance(self.source_frame_indices, tuple) or not self.source_frame_indices:
            raise TypedContractError("media frame provenance must be a non-empty tuple")
        if any(type(index) is not int or index < 0 for index in self.source_frame_indices):
            raise TypedContractError("media frame indices must be non-negative integers")

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def descriptor(self, part_name: str) -> dict[str, object]:
        return {
            "name": part_name,
            "kind": "media",
            "byte_length": len(self.data),
            "payload_sha256": self.payload_sha256,
            "media_type": self.media_type,
            "source_artifact_id": self.source_artifact_id,
            "source_frame_indices": list(self.source_frame_indices),
        }


class TypedRecord(Protocol):
    def to_wire_record(self, prefix: str = "input") -> tuple[dict[str, object], dict[str, object]]:
        ...


T = TypeVar("T")


def _is_native_payload(value: object) -> bool:
    return isinstance(value, DroidNativeSensorDepthAbiPayload)


def _encode_value(value: object, prefix: str, parts: dict[str, object]) -> object:
    if isinstance(value, TypedTensor):
        parts[prefix] = value
        return {"__part__": prefix, "descriptor": value.descriptor(prefix)}
    if isinstance(value, BinaryAsset):
        parts[prefix] = value
        return {"__part__": prefix, "descriptor": value.descriptor(prefix)}
    if _is_native_payload(value):
        parts[prefix] = value
        descriptor = value.to_mapping()
        descriptor.update({"name": prefix, "kind": "sealed_native_depth", "dtype": "float32", "wire_dtype": "<f4"})
        return {"__part__": prefix, "descriptor": descriptor}
    if is_dataclass(value):
        return {
            field.name: _encode_value(getattr(value, field.name), f"{prefix}.{field.name}", parts)
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item, f"{prefix}.{key}", parts) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_encode_value(item, f"{prefix}.{index}", parts) for index, item in enumerate(value)]
    if isinstance(value, list):
        return [_encode_value(item, f"{prefix}.{index}", parts) for index, item in enumerate(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypedContractError(f"value {prefix} has unsupported type {type(value).__name__}")


def encode_typed_record(value: object, *, prefix: str = "input") -> tuple[dict[str, object], dict[str, object]]:
    """Encode a typed dataclass record while keeping tensors as named binary parts."""

    encoded = _encode_value(value, prefix, parts := {})
    if not isinstance(encoded, dict):
        raise TypedContractError("typed stage record must encode to an object")
    return encoded, parts


def _decode_value(value: object, expected: object, parts: Mapping[str, object]) -> object:
    if isinstance(value, Mapping) and "__part__" in value:
        name = str(value["__part__"])
        if name not in parts:
            raise TypedContractError(f"missing binary part {name!r}")
        return parts[name]
    origin = get_origin(expected)
    args = get_args(expected)
    if expected is Any or expected is object or expected is None:
        if isinstance(value, Mapping):
            return {key: _decode_value(item, Any, parts) for key, item in value.items()}
        if isinstance(value, list):
            return [_decode_value(item, Any, parts) for item in value]
        return value
    if origin in (list, Sequence):
        item_type = args[0] if args else Any
        return [_decode_value(item, item_type, parts) for item in value]  # type: ignore[arg-type]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode_value(item, args[0], parts) for item in value)  # type: ignore[arg-type]
        return tuple(_decode_value(item, item_type, parts) for item, item_type in zip(value, args))  # type: ignore[arg-type]
    if origin in (dict, Mapping):
        value_type = args[1] if len(args) > 1 else Any
        return {str(key): _decode_value(item, value_type, parts) for key, item in value.items()}  # type: ignore[union-attr]
    if origin in (Union, types.UnionType):
        non_none = [item for item in args if item is not type(None)]
        if value is None:
            return None
        return _decode_value(value, non_none[0] if non_none else Any, parts)
    if isinstance(expected, type) and issubclass(expected, Enum):
        return expected(value)
    if isinstance(expected, type) and is_dataclass(expected):
        hints = get_type_hints(expected)
        if not isinstance(value, Mapping):
            raise TypedContractError(f"expected object for {expected.__name__}")
        kwargs = {}
        for field in fields(expected):
            if field.name in value:
                kwargs[field.name] = _decode_value(value[field.name], hints.get(field.name, field.type), parts)
            elif field.default is not MISSING or field.default_factory is not MISSING:
                continue
            else:
                raise TypedContractError(f"decoded {expected.__name__} is missing field {field.name}")
        return expected(**kwargs)
    if expected in (int, float, str, bool):
        return expected(value)
    return value


def decode_typed_record(record_type: type[T], value: Mapping[str, object], parts: Mapping[str, object]) -> T:
    result = _decode_value(value, record_type, parts)
    if not isinstance(result, record_type):
        raise TypedContractError(f"decoded result is not {record_type.__name__}")
    return result


@dataclass(frozen=True)
class UniDepthInput:
    ownership: Ownership
    rgb_batch: TypedTensor
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    spatial: SpatialTransform

    def __post_init__(self) -> None:
        _validate_frame_batch(self.rgb_batch, self.frame_indices, self.timestamps_s, "unidepth RGB", dtype="uint8", channels=3)


@dataclass(frozen=True)
class UniDepthOutput:
    ownership: Ownership
    depth_m: TypedTensor
    K_px: TypedTensor
    confidence: TypedTensor
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    spatial: SpatialTransform
    model_revision: str

    def __post_init__(self) -> None:
        if self.depth_m.dtype != "float32" or self.confidence.dtype != "float32":
            raise TypedContractError("UniDepth depth/confidence must be float32")
        if self.K_px.shape != (len(self.frame_indices), 3, 3):
            raise TypedContractError("UniDepth K_px must be [T,3,3]")
        if self.depth_m.shape[0] != len(self.frame_indices) or self.confidence.shape != self.depth_m.shape:
            raise TypedContractError("UniDepth output timeline/tensor shapes disagree")


@dataclass(frozen=True)
class HandsInput:
    ownership: Ownership
    rgb_batch: TypedTensor
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    spatial: SpatialTransform

    def __post_init__(self) -> None:
        _validate_frame_batch(self.rgb_batch, self.frame_indices, self.timestamps_s, "hands RGB", dtype="uint8", channels=3)


@dataclass(frozen=True)
class HandDetections:
    """Locator-only hand observations from the ``hands-yolo-v2`` service contract.

    The online YOLO-only service intentionally has no segmentation-mask payload.
    Crops and HaWoR observation features are derived from these scalar detections.
    """

    boxes_xyxy: TypedTensor
    scores: TypedTensor
    sides: TypedTensor
    visibility: TypedTensor
    uncertainty: TypedTensor

    def __post_init__(self) -> None:
        if self.boxes_xyxy.dtype != "float32" or self.scores.dtype != "float32":
            raise TypedContractError("hand boxes/scores must be float32")
        if len(self.boxes_xyxy.shape) != 3 or self.boxes_xyxy.shape[-1] != 4:
            raise TypedContractError("hand boxes must be [T,K,4], not a box-only state")
        if self.scores.shape != self.boxes_xyxy.shape[:2] or self.sides.shape != self.scores.shape:
            raise TypedContractError("hand detection arrays disagree")
        for tensor in (self.visibility, self.uncertainty):
            if tensor.shape != self.scores.shape:
                raise TypedContractError("hand visibility/uncertainty must be [T,K]")


@dataclass(frozen=True)
class HandsOutput:
    ownership: Ownership
    detections: HandDetections
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    spatial: SpatialTransform
    model_revision: str

    def __post_init__(self) -> None:
        if self.detections.scores.shape[0] != len(self.frame_indices):
            raise TypedContractError("hands output timeline disagrees with detections")


@dataclass(frozen=True)
class CropTransform:
    center_xy_px: tuple[float, float]
    size_px: float
    source_to_crop: SpatialTransform
    side: HandSide

    def __post_init__(self) -> None:
        if len(self.center_xy_px) != 2 or self.size_px <= 0:
            raise TypedContractError("WiLoR crop transform requires center and positive size")

    def to_wire(self, source_K_px: np.ndarray) -> dict[str, object]:
        model_to_source = np.asarray(self.source_to_crop.pixel_to_source, dtype=np.float64)
        source_to_model = np.linalg.inv(model_to_source)
        K = np.asarray(source_K_px, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise TypedContractError("crop wire calibration requires a finite positive-focal source K")
        img_focal = float(np.sqrt(K[0, 0] * K[1, 1]))
        return {
            "center": list(self.center_xy_px),
            "scale": self.size_px / 200.0,
            "img_focal": img_focal,
            "img_center": [float(K[0, 2]), float(K[1, 2])],
            "do_flip": self.side is HandSide.LEFT,
            "source_size": {"width": self.source_to_crop.source_width_px, "height": self.source_to_crop.source_height_px},
            "pixel_transform": {
                "source_to_model": source_to_model.tolist(),
                "model_to_source": model_to_source.tolist(),
                "resize_mode": "hawor_crop_flip" if self.side is HandSide.LEFT else "hawor_crop",
                "crop_xywh": None,
                "pad_ltrb": None,
            },
        }


@dataclass(frozen=True)
class WiLoRInput:
    ownership: Ownership
    crop_batch: TypedTensor
    crop_transforms: tuple[CropTransform, ...]
    source_K_px: TypedTensor

    def __post_init__(self) -> None:
        if self.crop_batch.dtype != "float32" or self.crop_batch.shape[1:] != (3, 256, 256):
            raise TypedContractError("WiLoR crop batch must be float32 [B,3,256,256]")
        if self.crop_batch.shape[0] != len(self.crop_transforms):
            raise TypedContractError("WiLoR crop transforms must match batch")
        if self.source_K_px.shape != (3, 3):
            raise TypedContractError("WiLoR source_K_px must be full [3,3]")


@dataclass(frozen=True)
class ManoBatch:
    global_orient: TypedTensor
    hand_pose: TypedTensor
    betas: TypedTensor
    vertices: TypedTensor
    joints: TypedTensor
    cam_t_full: TypedTensor
    pred_cam: TypedTensor
    vertices_source_px: TypedTensor
    joints_source_px: TypedTensor
    confidence: TypedTensor
    uncertainty: TypedTensor

    def __post_init__(self) -> None:
        batch = self.global_orient.shape[0]
        if self.global_orient.shape[1:] != (3, 3) or self.hand_pose.shape[:2] != (batch, 15) or self.hand_pose.shape[-2:] != (3, 3):
            raise TypedContractError("MANO rotation output has invalid shape")
        if self.betas.shape != (batch, 10) or self.vertices.shape[:2] != (batch, 778) or self.vertices.shape[-1] != 3:
            raise TypedContractError("MANO surface output has invalid shape")
        if self.joints.shape[0] != batch or self.joints.shape[-1] != 3 or self.cam_t_full.shape != (batch, 3) or self.pred_cam.shape != (batch, 3):
            raise TypedContractError("MANO joints/camera output has invalid shape")
        if self.vertices_source_px.shape != self.vertices.shape[:-1] + (2,) or self.joints_source_px.shape != self.joints.shape[:-1] + (2,):
            raise TypedContractError("MANO crop projections must match vertex/joint axes in source pixels")
        if self.confidence.shape != (batch,) or self.uncertainty.shape != (batch,):
            raise TypedContractError("MANO confidence/uncertainty must be [B]")


@dataclass(frozen=True)
class WiLoROutput:
    ownership: Ownership
    handedness: tuple[HandSide, ...]
    mano: ManoBatch
    model_revision: str

    def __post_init__(self) -> None:
        if len(self.handedness) != self.mano.global_orient.shape[0]:
            raise TypedContractError("WiLoR handedness must match MANO batch")


@dataclass(frozen=True)
class DroidCreateInput:
    ownership: Ownership
    timeline: FrameTimelineMetadata
    K_droid_input: tuple[float, float, float, float]
    p_source_to_droid_input: SpatialTransform
    p_droid_input_to_model: SpatialTransform
    model_shape_yx: tuple[int, int]
    require_rgbd_capability: bool = True
    allow_monocular_droid_smoke: bool = False

    def __post_init__(self) -> None:
        _validate_k_four(self.K_droid_input)
        if len(self.model_shape_yx) != 2 or any(int(dim) <= 0 for dim in self.model_shape_yx):
            raise TypedContractError("DROID model shape must be positive H,W")
        if self.require_rgbd_capability and self.allow_monocular_droid_smoke:
            raise TypedContractError("DROID strict RGB-D and diagnostic monocular modes are mutually exclusive")
        if self.p_droid_input_to_model.pixel_to_source[0][0] <= 0 or self.p_droid_input_to_model.pixel_to_source[1][1] <= 0:
            raise TypedContractError("DROID pixel transform must preserve positive axes")


@dataclass(frozen=True)
class DroidCapabilities:
    full_K_consumed: bool
    native_sensor_depth_consumed: bool
    trace: tuple[str, ...]
    server_revision: str
    service_release: str = "unknown"
    frontend_no_grad: bool = False
    capability_source: str = "unknown"
    evidence: tuple[str, ...] = ()

    def require_rgbd(self) -> None:
        if not self.full_K_consumed or not self.native_sensor_depth_consumed:
            raise TypedContractError("remote DROID capability mismatch: full-K/native-depth consumption is unproven")
        required = {"session_create", "frame_push", "frontend", "bundle_adjustment"}
        if not required.issubset(set(self.trace)):
            raise TypedContractError("remote DROID capability trace does not prove frontend and BA consumption")
        if self.capability_source not in {"frozen_contract", "response_trace", "unknown"}:
            raise TypedContractError("invalid DROID capability provenance")
        if self.capability_source == "unknown":
            raise TypedContractError("remote DROID capability provenance is unknown")

    @classmethod
    def frozen_3572551(cls) -> "DroidCapabilities":
        return cls(
            full_K_consumed=True,
            native_sensor_depth_consumed=False,
            trace=("session_create", "frame_push", "frontend", "bundle_adjustment"),
            server_revision="droid-v1",
            service_release="3572551/observed-config",
            frontend_no_grad=False,
            capability_source="frozen_contract",
            evidence=("serving/droid.py:517-532", "serving/droid.py:843-874", "handler_depth_not_forwarded"),
        )

    def diagnostic_monocular(self) -> None:
        if self.native_sensor_depth_consumed:
            raise TypedContractError("diagnostic monocular mode cannot claim native sensor depth")


@dataclass(frozen=True)
class DroidCreateOutput:
    ownership: Ownership
    session_id: str
    capabilities: DroidCapabilities

    def __post_init__(self) -> None:
        if not self.session_id:
            raise TypedContractError("DROID create response requires session_id")


@dataclass(frozen=True)
class DroidPushInput:
    ownership: Ownership
    session_id: str
    frame_index: int
    timestamp_s: float
    rgb: TypedTensor
    native_sensor_depth_abi_payload_m: DroidNativeSensorDepthAbiPayload
    K_droid_input: tuple[float, float, float, float]
    static_confidence_mask: TypedTensor | None = None
    require_rgbd_capability: bool = True
    allow_monocular_droid_smoke: bool = False

    def __post_init__(self) -> None:
        if not self.session_id or self.frame_index < 0 or not np.isfinite(self.timestamp_s):
            raise TypedContractError("DROID push identity/timestamp is invalid")
        if self.rgb.dtype != "uint8" or len(self.rgb.shape) != 3 or self.rgb.shape[-1] != 3:
            raise TypedContractError("DROID RGB must be uint8 HWC")
        _validate_k_four(self.K_droid_input)
        if self.require_rgbd_capability and self.allow_monocular_droid_smoke:
            raise TypedContractError("DROID strict RGB-D and diagnostic monocular modes are mutually exclusive")
        spec = self.native_sensor_depth_abi_payload_m.spec
        if spec.get("field_name") != "native_sensor_depth_abi_payload_m":
            raise TypedContractError("DROID depth field must retain native_sensor_depth_abi_payload_m")
        if spec.get("semantic_tag") != "stock_depthvideo_gather_slots_v1":
            raise TypedContractError("DROID depth must be native stock-gather ABI, not aligned depth")
        if self.static_confidence_mask is not None and self.static_confidence_mask.dtype != "uint8":
            raise TypedContractError("DROID static mask must be uint8")


@dataclass(frozen=True)
class DroidPushOutput:
    ownership: Ownership
    session_id: str
    frame_index: int
    accepted: bool
    keyframe_count: int
    capabilities: DroidCapabilities

    def __post_init__(self) -> None:
        if not self.session_id or self.frame_index < 0 or self.keyframe_count < 0:
            raise TypedContractError("DROID push output is invalid")


@dataclass(frozen=True)
class DroidFinalizeInput:
    ownership: Ownership
    session_id: str
    require_rgbd_capability: bool = True
    allow_monocular_droid_smoke: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise TypedContractError("DROID finalize requires session_id")
        if self.require_rgbd_capability and self.allow_monocular_droid_smoke:
            raise TypedContractError("DROID strict RGB-D and diagnostic monocular modes are mutually exclusive")


@dataclass(frozen=True)
class DroidFinalizeOutput:
    ownership: Ownership
    session_id: str
    T_world_camera: TypedTensor
    T_camera_world: TypedTensor
    intrinsics_px: TypedTensor
    disparities: TypedTensor
    keyframe_count: int
    scale_mode: str
    capabilities: DroidCapabilities
    acceptance: bool = True
    diagnostic_only: bool = False
    scale_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.T_world_camera.shape[-2:] != (4, 4) or self.T_camera_world.shape != self.T_world_camera.shape or self.T_world_camera.array.ndim != 3:
            raise TypedContractError("DROID poses must be [T,4,4] and inverse-shaped")
        world = np.asarray(self.T_world_camera.array, dtype=np.float64)
        camera = np.asarray(self.T_camera_world.array, dtype=np.float64)
        valid_raw = self.T_world_camera.provenance.get("droid_pose_valid")
        valid = np.asarray(valid_raw, dtype=bool) if valid_raw is not None else np.ones(world.shape[0], dtype=bool)
        if valid.shape != (world.shape[0],):
            raise TypedContractError("DROID pose validity must be [T]")
        if np.any(~np.isfinite(world[valid])) or np.any(~np.isfinite(camera[valid])):
            raise TypedContractError("valid DROID poses must contain finite values")
        for index in np.flatnonzero(valid):
            w, c = world[index], camera[index]
            if not np.allclose(w[3], (0.0, 0.0, 0.0, 1.0), atol=2e-4) or not np.allclose(c[3], (0.0, 0.0, 0.0, 1.0), atol=2e-4):
                raise TypedContractError("DROID poses must be homogeneous SE(3) transforms")
            r = w[:3, :3]
            if np.max(np.abs(r.T @ r - np.eye(3))) > 2e-4 or np.linalg.det(r) <= 0.0:
                raise TypedContractError("DROID world pose rotation is not SO(3)")
            if not np.allclose(w @ c, np.eye(4), atol=3e-4) or not np.allclose(c @ w, np.eye(4), atol=3e-4):
                raise TypedContractError("DROID camera/world poses are not mutual inverses")
        provenance = dict(self.scale_provenance)
        if self.scale_mode == "metric_rgbd_unidepth" and not provenance.get("scale_source"):
            raise TypedContractError("metric DROID output requires explicit scale_source provenance")
        object.__setattr__(self, "scale_provenance", provenance)
        if self.intrinsics_px.shape != (4,):
            raise TypedContractError("DROID output must retain full [fx,fy,cx,cy]")
        if self.scale_mode not in {"metric_rgbd_unidepth", "up_to_scale_monocular"}:
            raise TypedContractError("unknown DROID scale mode")
        if self.diagnostic_only and self.acceptance:
            raise TypedContractError("diagnostic DROID output cannot be accepted")
        if self.diagnostic_only and self.scale_mode != "up_to_scale_monocular":
            raise TypedContractError("diagnostic DROID output must be monocular up-to-scale")
        if self.keyframe_count < 0:
            raise TypedContractError("keyframe_count must be non-negative")


@dataclass(frozen=True)
class HaworObservation:
    frame_index: int
    source_timestamp_s: float
    side: HandSide
    occlusion_state: str
    detection_confidence: float

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise TypedContractError("HaWoR observation frame index must be non-negative")
        if self.occlusion_state not in {"visible", "partially_visible", "occluded", "out_of_frame", "unresolved"}:
            raise TypedContractError(f"invalid HaWoR occlusion state {self.occlusion_state!r}")
        if not np.isfinite(self.source_timestamp_s):
            raise TypedContractError("HaWoR observation timestamp must be finite")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise TypedContractError("HaWoR observation confidence must be in [0, 1]")

    def to_wire(self) -> dict[str, object]:
        return {"frame_index": self.frame_index, "source_timestamp_s": self.source_timestamp_s, "side": self.side.value, "occlusion_state": self.occlusion_state, "detection_confidence": self.detection_confidence}


@dataclass(frozen=True)
class HaworTrackInput:
    ownership: Ownership
    crop_batch: TypedTensor
    frame_indices: tuple[tuple[int, ...], ...]
    crop_transforms: tuple[CropTransform, ...]
    observations: TypedTensor
    unidepth_depth_m: TypedTensor
    unidepth_K_px: TypedTensor
    droid_poses_world_camera: TypedTensor
    droid_timestamps_s: TypedTensor
    observation_records: tuple[HaworObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.crop_batch.dtype != "float32" or self.crop_batch.shape[2:] != (3, 256, 256):
            raise TypedContractError("HaWoR tracks require native [B,16,3,256,256] float32 crops")
        if self.crop_batch.shape[1] != 16 or self.crop_batch.shape[0] != len(self.frame_indices):
            raise TypedContractError("HaWoR frame chunks must be [B,16,...] and indexed per chunk")
        if len(self.crop_transforms) not in {self.crop_batch.shape[0], 16}:
            raise TypedContractError("HaWoR crop transforms must be per-chunk or all 16 frame transforms")
        if self.observations.shape[:2] != self.crop_batch.shape[:2]:
            raise TypedContractError("HaWoR observations must be [B,16,...]")
        if self.droid_poses_world_camera.shape[-2:] != (4, 4):
            raise TypedContractError("HaWoR must consume DROID world-from-camera poses")


@dataclass(frozen=True)
class HaworTrackOutput:
    ownership: Ownership
    root_orient: TypedTensor
    hand_pose: TypedTensor
    trans_camera_m: TypedTensor
    betas: TypedTensor
    vertices_camera_m: TypedTensor
    joints_camera_m: TypedTensor
    observed: TypedTensor
    occlusion_state: TypedTensor
    uncertainty_m: TypedTensor
    model_revision: str

    def __post_init__(self) -> None:
        if self.trans_camera_m.shape[-1] != 3 or self.betas.shape[-1] != 10 or self.vertices_camera_m.shape[-2:] != (778, 3):
            raise TypedContractError("HaWoR output must retain metric MANO surface state")
        if self.betas.shape[:-1] != self.trans_camera_m.shape[:-1]:
            raise TypedContractError("HaWoR MANO betas must follow the output timeline")
        if self.observed.shape != self.uncertainty_m.shape:
            raise TypedContractError("HaWoR observation and uncertainty shapes disagree")


@dataclass(frozen=True)
class InfillerFrame:
    frame_index: int
    source_timestamp_s: float
    side: HandSide
    root_orient: tuple[tuple[float, ...], ...]
    hand_pose: tuple[tuple[float, ...], ...]
    trans: tuple[float, float, float]
    betas: tuple[float, ...]
    observed: bool
    uncertainty: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.source_timestamp_s):
            raise TypedContractError("Infiller frame timestamp must be finite")
        if not np.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise TypedContractError("Infiller frame uncertainty must be finite and non-negative")
        if len(self.betas) != 10 or len(self.trans) != 3:
            raise TypedContractError("Infiller frame MANO shape/translation dimensions are invalid")

    def to_wire(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "source_timestamp_s": self.source_timestamp_s,
            "side": self.side.value,
            "root_orient": [list(row) for row in self.root_orient],
            "hand_pose": [list(row) for row in self.hand_pose],
            "trans": list(self.trans),
            "betas": list(self.betas),
            "observed": self.observed,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class InfillerInput:
    ownership: Ownership
    window_120x218: TypedTensor
    observation_mask: TypedTensor
    timestamps_s: TypedTensor
    droid_poses_world_camera: TypedTensor
    unidepth_scale_K: TypedTensor
    frames: tuple[InfillerFrame, ...] = ()

    def __post_init__(self) -> None:
        if self.window_120x218.dtype != "float32" or self.window_120x218.shape != (120, 218):
            raise TypedContractError("infiller must receive coupled [120,218] float32 windows")
        if self.observation_mask.shape != (120, 2):
            raise TypedContractError("infiller observation mask must be [120,2]")
        if self.timestamps_s.shape != (120,):
            raise TypedContractError("infiller timestamps must be [120]")
        if self.droid_poses_world_camera.shape[-2:] != (4, 4) or self.unidepth_scale_K.shape != (3, 3):
            raise TypedContractError("infiller must receive DROID poses and full UniDepth K")


@dataclass(frozen=True)
class InfillerOutput:
    ownership: Ownership
    state_2x120x109: TypedTensor
    observed: TypedTensor
    inferred: TypedTensor
    uncertainty_m: TypedTensor
    model_revision: str

    def __post_init__(self) -> None:
        if self.state_2x120x109.dtype != "float32" or self.state_2x120x109.shape != (2, 120, 109):
            raise TypedContractError("infiller output must retain two-hand [2,120,109] state")
        for tensor in (self.observed, self.inferred, self.uncertainty_m):
            if tensor.shape != (2, 120):
                raise TypedContractError("infiller state masks/uncertainty must be [2,120]")


COSMOS_MAX_MEDIA_ITEMS = 8
COSMOS_MAX_MEDIA_ITEM_BYTES = 16 * 1024 * 1024
COSMOS_MAX_MEDIA_REQUEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class CosmosMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"} or not isinstance(self.content, str) or not self.content.strip():
            raise TypedContractError("Cosmos messages require role and non-empty text content")


@dataclass(frozen=True)
class CosmosGeneration:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens <= 0 or self.max_tokens > 8192:
            raise TypedContractError("Cosmos max_tokens must be a positive integer <=8192")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not np.isfinite(self.temperature) or float(self.temperature) != 0.0:
            raise TypedContractError("production Cosmos generation must be deterministic (temperature=0)")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)) or not np.isfinite(self.top_p) or float(self.top_p) != 1.0:
            raise TypedContractError("production Cosmos generation must use top_p=1")

    def to_wire(self) -> dict[str, object]:
        return {"max_tokens": self.max_tokens, "temperature": 0.0, "top_p": 1.0}


@dataclass(frozen=True)
class CosmosInput:
    ownership: Ownership
    prompt: str | None
    messages: tuple[CosmosMessage, ...]
    generation: CosmosGeneration
    media: tuple[BinaryAsset, ...]
    source_frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        has_prompt = isinstance(self.prompt, str) and bool(self.prompt.strip())
        if has_prompt == bool(self.messages):
            raise TypedContractError("Cosmos requires direct prompt XOR messages")
        if self.prompt is not None and not has_prompt:
            raise TypedContractError("Cosmos prompt must be non-empty when supplied")
        if not isinstance(self.messages, tuple) or not all(isinstance(message, CosmosMessage) for message in self.messages):
            raise TypedContractError("Cosmos messages must be typed tuple records")
        if not isinstance(self.generation, CosmosGeneration):
            raise TypedContractError("Cosmos generation must use typed deterministic controls")
        if not isinstance(self.media, tuple) or not all(isinstance(media, BinaryAsset) for media in self.media):
            raise TypedContractError("Cosmos media must be typed binary assets")
        if not isinstance(self.source_frame_indices, tuple) or not self.media or not self.source_frame_indices:
            raise TypedContractError("Cosmos production requests require source-backed media and indices")
        if any(type(index) is not int or index < 0 for index in self.source_frame_indices):
            raise TypedContractError("Cosmos source indices must be non-negative integers")
        if len(self.media) > COSMOS_MAX_MEDIA_ITEMS:
            raise TypedContractError(f"Cosmos accepts at most {COSMOS_MAX_MEDIA_ITEMS} media items")
        if len(self.media) != len(self.source_frame_indices):
            raise TypedContractError("Cosmos media and source indices must have identical cardinality")
        if tuple(sorted(self.source_frame_indices)) != self.source_frame_indices:
            raise TypedContractError("Cosmos source indices must be nondecreasing")
        if len(set(self.source_frame_indices)) != len(self.source_frame_indices):
            pad_start = next(index for index in range(1, len(self.source_frame_indices)) if self.source_frame_indices[index] == self.source_frame_indices[index - 1])
            expected_padding = self.source_frame_indices[:pad_start] + (self.source_frame_indices[-1],) * (len(self.source_frame_indices) - pad_start)
            if len(self.source_frame_indices) != COSMOS_MAX_MEDIA_ITEMS or self.source_frame_indices != expected_padding:
                raise TypedContractError("Cosmos duplicate source indices must only pad final media to eight")
        total = 0
        for media, source_index in zip(self.media, self.source_frame_indices):
            if media.media_type != "image/jpeg":
                raise TypedContractError("Cosmos production media must be JPEG")
            if media.source_frame_indices != (source_index,):
                raise TypedContractError("Cosmos media provenance must exactly match ordered source indices")
            if len(media.data) > COSMOS_MAX_MEDIA_ITEM_BYTES:
                raise TypedContractError("Cosmos media item exceeds 16 MiB")
            total += len(media.data)
        if total > COSMOS_MAX_MEDIA_REQUEST_BYTES:
            raise TypedContractError("Cosmos request media exceeds 64 MiB")


@dataclass(frozen=True)
class CosmosOutput:
    ownership: Ownership
    text: str
    finish_reason: str
    stop_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timings: Mapping[str, float]
    trace: Mapping[str, object]
    media_provenance: tuple[Mapping[str, object], ...]
    model_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise TypedContractError("Cosmos output text must be non-empty")
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise TypedContractError("Cosmos finish reason is required")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypedContractError("Cosmos stop reason must be text or null")
        token_values = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if any(type(value) is not int or value < 0 for value in token_values):
            raise TypedContractError("Cosmos token counts must be non-negative integers")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise TypedContractError("Cosmos total_tokens must equal prompt_tokens + completion_tokens")
        expected_timings = {"queue_wait_s", "prefill_s", "time_to_first_token_s", "decode_s", "e2e_s"}
        if not isinstance(self.timings, Mapping) or set(self.timings) != expected_timings:
            raise TypedContractError("Cosmos timings must contain the exact frozen timing fields")
        if any(not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 for key, value in self.timings.items()):
            raise TypedContractError("Cosmos timings must be finite non-negative numeric fields")
        expected_trace = {
            "batch_id", "replica_id", "admitted_monotonic_s", "dispatched_monotonic_s",
            "forward_started_monotonic_s", "completed_monotonic_s", "effective_work_units",
            "request_count", "forward_count", "model_load_count",
        }
        if not isinstance(self.trace, Mapping) or set(self.trace) != expected_trace:
            raise TypedContractError("Cosmos service batch trace must contain the exact frozen fields")
        if not all(isinstance(self.trace[key], str) and self.trace[key] for key in ("batch_id", "replica_id")):
            raise TypedContractError("Cosmos batch/replica identity is required")
        trace_times = tuple(self.trace[key] for key in ("admitted_monotonic_s", "dispatched_monotonic_s", "forward_started_monotonic_s", "completed_monotonic_s"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in trace_times) or not all(left <= right for left, right in zip(trace_times, trace_times[1:])):
            raise TypedContractError("Cosmos trace timings must be finite and monotonic")
        if any(type(self.trace[key]) is not int or self.trace[key] < 0 for key in ("effective_work_units", "request_count", "forward_count", "model_load_count")):
            raise TypedContractError("Cosmos trace counts must be non-negative integers")
        if not isinstance(self.model_revision, str) or not self.model_revision:
            raise TypedContractError("Cosmos server model revision is required")
        if not isinstance(self.media_provenance, tuple) or not self.media_provenance:
            raise TypedContractError("Cosmos media provenance is required")
        for item in self.media_provenance:
            if type(item) is not dict or set(item) != {"kind", "media_type", "source_index", "bytes"}:
                raise TypedContractError("Cosmos media provenance must use the exact frozen fields")
            if item["kind"] != "image" or not isinstance(item["media_type"], str) or not item["media_type"]:
                raise TypedContractError("Cosmos media provenance kind/type is invalid")
            if type(item["source_index"]) is not int or item["source_index"] < 0 or type(item["bytes"]) is not int or item["bytes"] <= 0:
                raise TypedContractError("Cosmos media provenance source index/bytes must be positive exact integers")


STAGE_INPUT_TYPES: dict[str, type] = {
    "unidepth.infer": UniDepthInput,
    "hands.detect": HandsInput,
    "wilor.reconstruct": WiLoRInput,
    "droid.create_session": DroidCreateInput,
    "droid.push_frame": DroidPushInput,
    "droid.finalize": DroidFinalizeInput,
    "hawor.infer_tracks": HaworTrackInput,
    "hawor_infiller.fill": InfillerInput,
    "cosmos3.reason": CosmosInput,
}

STAGE_OUTPUT_TYPES: dict[str, type] = {
    "unidepth.infer": UniDepthOutput,
    "hands.detect": HandsOutput,
    "wilor.reconstruct": WiLoROutput,
    "droid.create_session": DroidCreateOutput,
    "droid.push_frame": DroidPushOutput,
    "droid.finalize": DroidFinalizeOutput,
    "hawor.infer_tracks": HaworTrackOutput,
    "hawor_infiller.fill": InfillerOutput,
    "cosmos3.reason": CosmosOutput,
}


def _validate_frame_batch(tensor: TypedTensor, indices: tuple[int, ...], timestamps: tuple[float, ...], name: str, *, dtype: str, channels: int) -> None:
    if tensor.dtype != dtype or len(tensor.shape) != 4 or tensor.shape[-1] != channels:
        raise TypedContractError(f"{name} must be {dtype} [T,H,W,{channels}]")
    if tensor.shape[0] != len(indices) or len(indices) != len(timestamps) or not indices:
        raise TypedContractError(f"{name} timeline does not match batch")
    if tuple(indices) != tuple(range(indices[0], indices[0] + len(indices))) or any(float(a) >= float(b) for a, b in zip(timestamps, timestamps[1:])):
        raise TypedContractError(f"{name} frame timeline must be contiguous and increasing")


def _validate_k_four(values: tuple[float, float, float, float]) -> None:
    if len(values) != 4 or not all(np.isfinite(value) for value in values) or values[0] <= 0 or values[1] <= 0:
        raise TypedContractError("full K must be finite [fx,fy,cx,cy] with positive focal values")


__all__ = [
    "AlgorithmRequest",
    "AlgorithmResult",
    "BinaryAsset",
    "COSMOS_MAX_MEDIA_ITEM_BYTES",
    "COSMOS_MAX_MEDIA_ITEMS",
    "COSMOS_MAX_MEDIA_REQUEST_BYTES",
    "CosmosGeneration",
    "CosmosInput",
    "CosmosMessage",
    "CosmosOutput",
    "CropTransform",
    "DroidCapabilities",
    "DroidCreateInput",
    "DroidCreateOutput",
    "DroidFinalizeInput",
    "DroidFinalizeOutput",
    "DroidPushInput",
    "DroidPushOutput",
    "HandDetections",
    "HandsInput",
    "HandsOutput",
    "HandSide",
    "HaworObservation",
    "HaworTrackInput",
    "HaworTrackOutput",
    "InfillerFrame",
    "InfillerInput",
    "InfillerOutput",
    "Lane",
    "ManoBatch",
    "Ownership",
    "SpatialTransform",
    "STAGE_INPUT_TYPES",
    "STAGE_OUTPUT_TYPES",
    "TypedContractError",
    "TypedTensor",
    "UniDepthInput",
    "UniDepthOutput",
    "WiLoRInput",
    "WiLoROutput",
    "decode_typed_record",
    "encode_typed_record",
]
