"""CPU-testable full-K and strict-ray RGB-D coupling for stock DROID."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class DroidContractError(ValueError):
    """Raised when a DROID pixel, depth, or payload contract is violated."""


ABI_VERSION = "stock_depthvideo_gather_slots_v1"
SEMANTIC_TAG = "stock_depthvideo_gather_slots_v1"
_BOUNDARY_ORDER = (
    "pack_output",
    "droid_worker_depth_argument",
    "motion_filter_depth_entry",
    "depthvideo_stock_gather_entry",
)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DroidContractError(f"value is not canonical-json encodable: {exc}") from exc


def _matrix(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise DroidContractError(f"{name} must be a finite 3x3 matrix")
    return result


def normalize_homogeneous(value: object, *, name: str = "matrix") -> np.ndarray:
    result = _matrix(value, name=name)
    if abs(float(result[2, 2])) <= 1e-12:
        raise DroidContractError(f"{name}[2,2] must be nonzero")
    if abs(float(np.linalg.det(result))) <= 1e-12:
        raise DroidContractError(f"{name} must be invertible")
    result = result / result[2, 2]
    if abs(float(result[2, 0])) > 1e-9 or abs(float(result[2, 1])) > 1e-9:
        raise DroidContractError(f"{name} is not an affine pixel transform")
    return result


def _validate_k(value: object, *, name: str = "K") -> np.ndarray:
    result = normalize_homogeneous(value, name=name)
    if abs(float(result[0, 1])) > 1e-8 or abs(float(result[1, 0])) > 1e-8:
        raise DroidContractError(f"{name} must use the supported zero-skew pinhole form")
    if not np.isfinite(result[0, 0]) or not np.isfinite(result[1, 1]) or result[0, 0] <= 0 or result[1, 1] <= 0:
        raise DroidContractError(f"{name} must have positive finite fx and fy")
    if not np.all(np.isfinite(result[:2, 2])):
        raise DroidContractError(f"{name} must have finite principal point")
    return result


@dataclass(frozen=True)
class IntrinsicsCandidate:
    frame_idx: int
    k_depth: np.ndarray
    p_depth_to_source: np.ndarray
    confidence: float = 1.0
    frame_quality: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_depth", _validate_k(self.k_depth, name="k_depth"))
        object.__setattr__(self, "p_depth_to_source", normalize_homogeneous(self.p_depth_to_source, name="p_depth_to_source"))
        if not np.isfinite(self.confidence) or self.confidence <= 0:
            raise DroidContractError("intrinsics confidence must be positive and finite")
        if not np.isfinite(self.frame_quality) or self.frame_quality <= 0:
            raise DroidContractError("frame quality must be positive and finite")

    @property
    def k_source(self) -> np.ndarray:
        return _validate_k(self.p_depth_to_source @ self.k_depth, name="k_source")


@dataclass(frozen=True)
class RobustParameterTrace:
    value: float
    retained_frame_indices: tuple[int, ...]
    outlier_frame_indices: tuple[int, ...]
    robust_scale: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "value": self.value,
            "retained_frame_indices": list(self.retained_frame_indices),
            "outlier_frame_indices": list(self.outlier_frame_indices),
            "robust_scale": self.robust_scale,
        }


@dataclass(frozen=True)
class CanonicalKAggregation:
    k_canonical: np.ndarray
    parameter_traces: Mapping[str, RobustParameterTrace]
    source_candidates: tuple[IntrinsicsCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_canonical", _validate_k(self.k_canonical, name="k_canonical"))
        if set(self.parameter_traces) != {"fx", "fy", "cx", "cy"}:
            raise DroidContractError("canonical K must trace fx, fy, cx, and cy separately")

    def to_mapping(self) -> dict[str, object]:
        return {
            "k_canonical": self.k_canonical.tolist(),
            "parameters": {key: trace.to_mapping() for key, trace in self.parameter_traces.items()},
            "source_frame_indices": [item.frame_idx for item in self.source_candidates],
        }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _robust_parameter(values: np.ndarray, weights: np.ndarray, frame_indices: tuple[int, ...]) -> RobustParameterTrace:
    location = _weighted_median(values, weights)
    deviations = np.abs(values - location)
    scale = 1.4826 * _weighted_median(deviations, weights)
    if scale <= 1e-9:
        weighted_mean = float(np.average(values, weights=weights))
        scale = max(1e-6, float(np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights))))
    for _ in range(12):
        residual = values - location
        normalized = np.abs(residual) / max(1e-9, 1.345 * scale)
        huber = np.minimum(1.0, 1.0 / np.maximum(normalized, 1.0))
        effective = weights * huber
        location = float(np.sum(effective * values) / np.sum(effective))
    residual = np.abs(values - location)
    threshold = max(3.0 * scale, 1e-5)
    outlier_mask = residual > threshold
    retained = tuple(frame for frame, is_outlier in zip(frame_indices, outlier_mask) if not is_outlier)
    outliers = tuple(frame for frame, is_outlier in zip(frame_indices, outlier_mask) if is_outlier)
    return RobustParameterTrace(value=location, retained_frame_indices=retained, outlier_frame_indices=outliers, robust_scale=scale)


def aggregate_canonical_k(candidates: Sequence[IntrinsicsCandidate]) -> CanonicalKAggregation:
    if not candidates:
        raise DroidContractError("at least one intrinsics candidate is required")
    source_candidates = tuple(candidates)
    values = np.asarray([[item.k_source[0, 0], item.k_source[1, 1], item.k_source[0, 2], item.k_source[1, 2]] for item in source_candidates], dtype=np.float64)
    weights = np.asarray([item.confidence * item.frame_quality for item in source_candidates], dtype=np.float64)
    frame_indices = tuple(item.frame_idx for item in source_candidates)
    traces = {
        key: _robust_parameter(values[:, index], weights, frame_indices)
        for index, key in enumerate(("fx", "fy", "cx", "cy"))
    }
    k_source = np.array(
        [[traces["fx"].value, 0.0, traces["cx"].value], [0.0, traces["fy"].value, traces["cy"].value], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CanonicalKAggregation(k_canonical=k_source, parameter_traces=traces, source_candidates=source_candidates)


@dataclass(frozen=True)
class DroidPixelGeometry:
    k_canonical: np.ndarray
    p_source_to_droid_input: np.ndarray
    p_droid_input_to_model: np.ndarray
    droid_input_shape: tuple[int, int]
    model_shape: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_canonical", _validate_k(self.k_canonical, name="k_canonical"))
        object.__setattr__(self, "p_source_to_droid_input", normalize_homogeneous(self.p_source_to_droid_input, name="p_source_to_droid_input"))
        object.__setattr__(self, "p_droid_input_to_model", normalize_homogeneous(self.p_droid_input_to_model, name="p_droid_input_to_model"))
        if len(self.droid_input_shape) != 2 or len(self.model_shape) != 2 or any(int(dim) <= 0 for dim in self.droid_input_shape + self.model_shape):
            raise DroidContractError("DROID input and model shapes must be positive H,W pairs")
        if self.droid_input_shape != (8 * self.model_shape[0], 8 * self.model_shape[1]):
            raise DroidContractError("DROID input shape must be exactly eight times model shape")
        if abs(float(self.p_droid_input_to_model[0, 1])) > 1e-9 or abs(float(self.p_droid_input_to_model[1, 0])) > 1e-9:
            raise DroidContractError("DROID input-to-model transform must preserve axis-aligned pixel grids")

    @property
    def p_source_to_model(self) -> np.ndarray:
        return normalize_homogeneous(self.p_droid_input_to_model @ self.p_source_to_droid_input, name="p_source_to_model")

    @property
    def k_droid_input(self) -> np.ndarray:
        return _validate_k(self.p_source_to_droid_input @ self.k_canonical, name="k_droid_input")

    @property
    def k_model(self) -> np.ndarray:
        return _validate_k(self.p_source_to_model @ self.k_canonical, name="k_model")

    @property
    def k_droid_input_four(self) -> tuple[float, float, float, float]:
        k = self.k_droid_input
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

    @property
    def k_model_four(self) -> tuple[float, float, float, float]:
        k = self.k_model
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

    def input_ray_for_model_cell(self, i: int, j: int) -> np.ndarray:
        if i < 0 or i >= self.model_shape[1] or j < 0 or j >= self.model_shape[0]:
            raise DroidContractError("model cell is out of bounds")
        return np.array([8.0 * i, 8.0 * j, 1.0], dtype=np.float64)

    def source_point_for_model_cell(self, i: int, j: int) -> np.ndarray:
        q_source = np.linalg.inv(self.p_source_to_droid_input) @ self.input_ray_for_model_cell(i, j)
        if abs(float(q_source[2])) <= 1e-12:
            raise DroidContractError("model ray maps to a point at infinity")
        return q_source / q_source[2]

    def verify_ray_equivalence(self, i: int, j: int, *, atol: float = 1e-9) -> bool:
        q_model = np.array([float(i), float(j), 1.0], dtype=np.float64)
        left = np.linalg.inv(self.k_model) @ q_model
        q_input = self.input_ray_for_model_cell(i, j)
        right = np.linalg.inv(self.k_droid_input) @ q_input
        left = left / left[2]
        right = right / right[2]
        return bool(np.allclose(left, right, atol=atol, rtol=0.0))

    def to_mapping(self) -> dict[str, object]:
        return {
            "k_canonical": self.k_canonical.tolist(),
            "p_source_to_droid_input": self.p_source_to_droid_input.tolist(),
            "p_droid_input_to_model": self.p_droid_input_to_model.tolist(),
            "p_source_to_model": self.p_source_to_model.tolist(),
            "k_droid_input": self.k_droid_input.tolist(),
            "k_model": self.k_model.tolist(),
            "droid_input_shape": list(self.droid_input_shape),
            "model_shape": list(self.model_shape),
        }


@dataclass(frozen=True)
class DepthEvidence:
    depth_m: np.ndarray
    p_depth_to_source: np.ndarray
    source_artifact_id: str
    frame_idx: int
    evidence_grid: str
    confidence: np.ndarray | None = None
    min_confidence: float = 0.0
    interpolation_policy: str = "validity_aware_bilinear"
    validity_policy: str = "finite_positive_metric_depth"

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
            raise DroidContractError("depth evidence must be a numeric H,W raster")
        confidence = None if self.confidence is None else np.asarray(self.confidence)
        if confidence is not None and confidence.shape != depth.shape:
            raise DroidContractError("confidence raster must match depth evidence shape")
        if not self.source_artifact_id or not self.evidence_grid:
            raise DroidContractError("depth evidence provenance is required")
        if self.min_confidence < 0 or not np.isfinite(self.min_confidence):
            raise DroidContractError("min_confidence must be finite and nonnegative")
        object.__setattr__(self, "depth_m", np.array(depth, copy=True))
        object.__setattr__(self, "p_depth_to_source", normalize_homogeneous(self.p_depth_to_source, name="p_depth_to_source"))
        if confidence is not None:
            object.__setattr__(self, "confidence", np.array(confidence, copy=True))


@dataclass(frozen=True)
class PayloadIdentity:
    payload_id: str
    spec_json: str
    shape: tuple[int, int]
    dtype: str
    byte_length: int
    payload_sha256: str
    canonical_tensor_digest: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "spec_json": self.spec_json,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_length": self.byte_length,
            "payload_sha256": self.payload_sha256,
            "canonical_tensor_digest": self.canonical_tensor_digest,
        }


@dataclass(frozen=True)
class PayloadBoundaryRecord:
    boundary: str
    identity: PayloadIdentity
    sealed_operation_trace: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "identity": self.identity.to_mapping(),
            "sealed_operation_trace": list(self.sealed_operation_trace),
        }


class _SealedPayloadArray(np.ndarray):
    """Read-only view carrying the seal identity into a native boundary."""

    payload_id: str


class DroidNativeSensorDepthAbiPayload:
    """Opaque stock-gather payload sealed after strict-ray packing."""

    def __init__(self, *, canonical_bytes: bytes, spec: Mapping[str, object], operation_trace: tuple[str, ...] = ("pack",)) -> None:
        if operation_trace not in (("pack",), ("pack", "serialize", "deserialize")):
            raise DroidContractError("invalid post-pack operation trace")
        spec_json = _canonical_json(dict(spec))
        shape = tuple(int(dim) for dim in spec["shape"])  # type: ignore[index]
        if len(shape) != 2 or any(dim <= 0 for dim in shape):
            raise DroidContractError("native payload shape must be positive H,W")
        if spec.get("abi_version") != ABI_VERSION or spec.get("semantic_tag") != SEMANTIC_TAG:
            raise DroidContractError("native payload ABI identity mismatch")
        if spec.get("wire_dtype") != "<f4" or spec.get("byte_order") != "little" or spec.get("memory_order") != "C":
            raise DroidContractError("native payload is not canonical little-endian C-order float32")
        if spec.get("dtype") != "float32" or spec.get("units") != "metres" or spec.get("tensor_index_order") != "yx":
            raise DroidContractError("native payload dtype, units, or indexing contract mismatch")
        expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
        if len(canonical_bytes) != expected_bytes:
            raise DroidContractError("native payload byte length does not match shape")
        self._canonical_bytes = bytes(canonical_bytes)
        self._spec = json.loads(spec_json)
        self._spec_json = spec_json
        self._operation_trace = operation_trace
        self._payload_sha256 = hashlib.sha256(self._canonical_bytes).hexdigest()
        self._canonical_tensor_digest = hashlib.sha256(self._spec_json + b"\x00" + self._canonical_bytes).hexdigest()
        self._payload_id = f"sha256:{self._canonical_tensor_digest}"

    @classmethod
    def seal(
        cls,
        payload: np.ndarray,
        *,
        model_shape: tuple[int, int],
        provenance: Mapping[str, object],
    ) -> "DroidNativeSensorDepthAbiPayload":
        array = np.asarray(payload)
        expected_shape = (8 * int(model_shape[0]), 8 * int(model_shape[1]))
        if array.shape != expected_shape:
            raise DroidContractError(f"native payload shape must be {expected_shape}, got {array.shape}")
        if array.dtype != np.dtype("float32") or not array.flags.c_contiguous:
            raise DroidContractError("native payload must be C-contiguous float32 before sealing")
        canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
        spec = {
            "abi_version": ABI_VERSION,
            "field_name": "native_sensor_depth_abi_payload_m",
            "shape": list(canonical.shape),
            "dtype": "float32",
            "wire_dtype": "<f4",
            "byte_order": "little",
            "memory_order": "C",
            "units": "metres",
            "tensor_index_order": "yx",
            "semantic_tag": SEMANTIC_TAG,
            "provenance": json.loads(_canonical_json(dict(provenance))),
        }
        return cls(canonical_bytes=canonical.tobytes(order="C"), spec=spec)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(dim) for dim in self._spec["shape"])  # type: ignore[return-value]

    @property
    def spec(self) -> dict[str, object]:
        return json.loads(self._spec_json)

    @property
    def operation_trace(self) -> tuple[str, ...]:
        return self._operation_trace

    @property
    def payload_id(self) -> str:
        return self._payload_id

    @property
    def payload_sha256(self) -> str:
        return self._payload_sha256

    @property
    def canonical_tensor_digest(self) -> str:
        return self._canonical_tensor_digest

    @property
    def canonical_bytes(self) -> bytes:
        return bytes(self._canonical_bytes)

    @property
    def array(self) -> np.ndarray:
        array = np.frombuffer(self._canonical_bytes, dtype=np.dtype("<f4")).reshape(self.shape).view(_SealedPayloadArray)
        array.payload_id = self.payload_id
        array.setflags(write=False)
        return array

    @property
    def identity(self) -> PayloadIdentity:
        actual = self.array
        payload = actual.tobytes(order="C")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        digest = hashlib.sha256(self._spec_json + b"\x00" + payload).hexdigest()
        if payload_sha256 != self._payload_sha256 or digest != self._canonical_tensor_digest:
            raise DroidContractError("sealed payload identity has been mutated")
        return PayloadIdentity(
            payload_id=self._payload_id,
            spec_json=self._spec_json.decode("utf-8"),
            shape=self.shape,
            dtype="float32",
            byte_length=len(payload),
            payload_sha256=payload_sha256,
            canonical_tensor_digest=digest,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "payload_id": self.payload_id,
            "spec": self.spec,
            "byte_length": len(self._canonical_bytes),
            "payload_sha256": self.payload_sha256,
            "canonical_tensor_digest": self.canonical_tensor_digest,
            "operation_trace": list(self.operation_trace),
        }

    def serialize(self) -> bytes:
        if self.operation_trace != ("pack",):
            raise DroidContractError("serialized payload cannot be serialized again")
        envelope = {
            "wire_schema": "ego.droid.native_depth.v1",
            "spec": self.spec,
            "canonical_bytes_b64": base64.b64encode(self._canonical_bytes).decode("ascii"),
            "byte_length": len(self._canonical_bytes),
            "payload_sha256": self.payload_sha256,
            "canonical_tensor_digest": self.canonical_tensor_digest,
            "payload_id": self.payload_id,
            "operation_trace": ["pack", "serialize"],
        }
        return _canonical_json(envelope)

    @classmethod
    def deserialize(cls, encoded: bytes) -> "DroidNativeSensorDepthAbiPayload":
        try:
            envelope = json.loads(encoded.decode("utf-8"))
            spec = envelope["spec"]
            payload = base64.b64decode(envelope["canonical_bytes_b64"], validate=True)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DroidContractError("invalid native payload serialization") from exc
        if envelope.get("wire_schema") != "ego.droid.native_depth.v1" or envelope.get("operation_trace") != ["pack", "serialize"]:
            raise DroidContractError("native payload serialization schema or operation trace mismatch")
        result = cls(canonical_bytes=payload, spec=spec, operation_trace=("pack", "serialize", "deserialize"))
        if envelope.get("byte_length") != len(payload) or envelope.get("payload_sha256") != result.payload_sha256 or envelope.get("canonical_tensor_digest") != result.canonical_tensor_digest or envelope.get("payload_id") != result.payload_id:
            raise DroidContractError("native payload serialization digest mismatch")
        return result

    def reject_post_pack_transform(self, operation: str) -> None:
        allowed = {"pass", "serialize", "deserialize"}
        if operation not in allowed:
            raise DroidContractError(f"post-pack operation {operation!r} is forbidden")


class PayloadIdentityTrace:
    """Recomputes payload identity at every native depth boundary."""

    def __init__(self, payload: DroidNativeSensorDepthAbiPayload) -> None:
        self.payload_id = payload.payload_id
        self._expected_identity = payload.identity
        self._operation_trace = payload.operation_trace
        self.records: list[PayloadBoundaryRecord] = []

    @property
    def sealed_operation_trace(self) -> tuple[str, ...]:
        return self._operation_trace + ("pass",)

    def record(self, boundary: str, actual: DroidNativeSensorDepthAbiPayload | np.ndarray) -> PayloadBoundaryRecord:
        expected_boundary = _BOUNDARY_ORDER[len(self.records)] if len(self.records) < len(_BOUNDARY_ORDER) else None
        if boundary not in _BOUNDARY_ORDER or boundary != expected_boundary:
            raise DroidContractError(f"unexpected payload boundary {boundary!r}; expected {expected_boundary!r}")
        if isinstance(actual, DroidNativeSensorDepthAbiPayload):
            identity = actual.identity
            operation_trace = actual.operation_trace
        elif isinstance(actual, np.ndarray):
            if not isinstance(actual, _SealedPayloadArray) or getattr(actual, "payload_id", None) != self.payload_id:
                raise DroidContractError("actual boundary tensor is not the sealed canonical view")
            if actual.shape != self._expected_identity.shape or actual.dtype != np.dtype("<f4") or not actual.flags.c_contiguous or actual.flags.writeable:
                raise DroidContractError("actual boundary tensor is not the sealed canonical representation")
            payload_bytes = actual.tobytes(order="C")
            identity = PayloadIdentity(
                payload_id=self.payload_id,
                spec_json=self._expected_identity.spec_json,
                shape=self._expected_identity.shape,
                dtype="float32",
                byte_length=len(payload_bytes),
                payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
                canonical_tensor_digest=hashlib.sha256(self._expected_identity.spec_json.encode("utf-8") + b"\x00" + payload_bytes).hexdigest(),
            )
            operation_trace = self._operation_trace
        else:
            raise DroidContractError("actual boundary value must be the sealed payload or its canonical array")
        if identity != self._expected_identity:
            raise DroidContractError(f"payload identity mismatch at {boundary}")
        if operation_trace != self._operation_trace:
            raise DroidContractError(f"payload operation trace changed at {boundary}")
        record = PayloadBoundaryRecord(boundary=boundary, identity=identity, sealed_operation_trace=self.sealed_operation_trace)
        self.records.append(record)
        return record

    def assert_complete(self) -> tuple[PayloadBoundaryRecord, ...]:
        if tuple(record.boundary for record in self.records) != _BOUNDARY_ORDER:
            raise DroidContractError("native payload trace is incomplete")
        if any(record.sealed_operation_trace not in (("pack", "pass"), ("pack", "serialize", "deserialize", "pass")) for record in self.records):
            raise DroidContractError("native payload trace contains a forbidden operation")
        if any(record.identity != self._expected_identity for record in self.records):
            raise DroidContractError("native payload identity changed across boundaries")
        return tuple(self.records)

    def to_mapping(self) -> dict[str, object]:
        self.assert_complete()
        return {
            "payload_id": self.payload_id,
            "sealed_operation_trace": list(self.sealed_operation_trace),
            "boundaries": [record.to_mapping() for record in self.records],
        }


def _axis_taps(coordinate: float, limit: int) -> tuple[tuple[int, float], ...] | None:
    if not np.isfinite(coordinate) or coordinate < 0.0 or coordinate > float(limit - 1):
        return None
    lower = int(np.floor(coordinate))
    upper = min(lower + 1, limit - 1)
    weight = float(coordinate - lower)
    if upper == lower or weight <= 1e-12:
        return ((lower, 1.0),)
    if weight >= 1.0 - 1e-12:
        return ((upper, 1.0),)
    return ((lower, 1.0 - weight), (upper, weight))


def _validity_aware_sample(evidence: DepthEvidence, q_depth: np.ndarray) -> float:
    height, width = evidence.depth_m.shape
    x_taps = _axis_taps(float(q_depth[0]), width)
    y_taps = _axis_taps(float(q_depth[1]), height)
    if x_taps is None or y_taps is None:
        return 0.0
    samples: list[tuple[float, float]] = []
    for y, wy in y_taps:
        for x, wx in x_taps:
            weight = wx * wy
            value = float(evidence.depth_m[y, x])
            confidence = 1.0 if evidence.confidence is None else float(evidence.confidence[y, x])
            if weight > 0.0:
                if not np.isfinite(value) or value <= 0.0 or not np.isfinite(confidence) or confidence < evidence.min_confidence:
                    return 0.0
                samples.append((value, weight))
    return float(sum(value * weight for value, weight in samples))


def pack_native_sensor_depth(evidence: DepthEvidence, geometry: DroidPixelGeometry) -> DroidNativeSensorDepthAbiPayload:
    payload = np.zeros(geometry.droid_input_shape, dtype=np.float32, order="C")
    inverse_source_to_input = np.linalg.inv(geometry.p_source_to_droid_input)
    inverse_depth_to_source = np.linalg.inv(evidence.p_depth_to_source)
    for j in range(geometry.model_shape[0]):
        for i in range(geometry.model_shape[1]):
            q_input = geometry.input_ray_for_model_cell(i, j)
            q_source = inverse_source_to_input @ q_input
            if abs(float(q_source[2])) <= 1e-12:
                continue
            q_source = q_source / q_source[2]
            q_depth = inverse_depth_to_source @ q_source
            if abs(float(q_depth[2])) <= 1e-12:
                continue
            q_depth = q_depth / q_depth[2]
            value = _validity_aware_sample(evidence, q_depth)
            if value > 0.0:
                payload[8 * j + 3, 8 * i + 3] = np.float32(value)
    provenance = {
        "source_artifact_id": evidence.source_artifact_id,
        "source_frame_idx": evidence.frame_idx,
        "evidence_grid": evidence.evidence_grid,
        "evidence_shape_yx": list(evidence.depth_m.shape),
        "p_depth_to_source": evidence.p_depth_to_source.tolist(),
        "p_source_to_droid_input": geometry.p_source_to_droid_input.tolist(),
        "interpolation_policy": evidence.interpolation_policy,
        "validity_policy": evidence.validity_policy,
        "confidence_policy": {"min_confidence": evidence.min_confidence, "source": "provided" if evidence.confidence is not None else "none"},
        "abi_version": ABI_VERSION,
    }
    return DroidNativeSensorDepthAbiPayload.seal(payload, model_shape=geometry.model_shape, provenance=provenance)


def stock_depthvideo_gather(payload: DroidNativeSensorDepthAbiPayload) -> np.ndarray:
    """The unchanged stock [3::8, 3::8] gather, restricted to the sealed ABI type."""
    if not isinstance(payload, DroidNativeSensorDepthAbiPayload):
        raise DroidContractError("stock gather requires DroidNativeSensorDepthAbiPayload, not an aligned raster")
    result = np.array(payload.array[3::8, 3::8], dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def sensor_disparity_from_stock_gather(depth_on_ray: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_on_ray)
    if depth.ndim != 2 or depth.dtype != np.dtype("float32") or not depth.flags.c_contiguous:
        raise DroidContractError("stock gathered depth must be C-contiguous float32 H,W")
    disparity = np.zeros(depth.shape, dtype=np.float32, order="C")
    valid = np.isfinite(depth) & (depth > 0.0)
    disparity[valid] = 1.0 / depth[valid]
    disparity.setflags(write=False)
    return disparity


def make_payload_identity_trace(payload: DroidNativeSensorDepthAbiPayload) -> PayloadIdentityTrace:
    return PayloadIdentityTrace(payload)


__all__ = [
    "ABI_VERSION",
    "SEMANTIC_TAG",
    "CanonicalKAggregation",
    "DepthEvidence",
    "DroidContractError",
    "DroidNativeSensorDepthAbiPayload",
    "DroidPixelGeometry",
    "IntrinsicsCandidate",
    "PayloadBoundaryRecord",
    "PayloadIdentity",
    "PayloadIdentityTrace",
    "RobustParameterTrace",
    "aggregate_canonical_k",
    "make_payload_identity_trace",
    "normalize_homogeneous",
    "pack_native_sensor_depth",
    "sensor_disparity_from_stock_gather",
    "stock_depthvideo_gather",
]
