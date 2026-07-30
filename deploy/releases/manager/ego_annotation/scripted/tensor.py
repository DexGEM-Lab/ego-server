"""Canonical tensor references and an immutable content-addressed store."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


class TensorContractError(ValueError):
    """Raised when tensor representation or identity is invalid."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TensorContractError(f"tensor spec is not canonical-json encodable: {exc}") from exc


def _wire_array(array: np.ndarray, wire_dtype: str) -> tuple[np.ndarray, bytes]:
    if not isinstance(array, np.ndarray):
        raise TensorContractError("tensor input must be a numpy.ndarray")
    if array.dtype.kind not in "biufc":
        raise TensorContractError(f"unsupported tensor dtype {array.dtype}")
    try:
        converted = array.astype(np.dtype(wire_dtype), copy=False)
    except (TypeError, ValueError) as exc:
        raise TensorContractError(f"cannot encode tensor as {wire_dtype}") from exc
    contiguous = np.ascontiguousarray(converted)
    if contiguous.dtype != np.dtype(wire_dtype) or not contiguous.flags.c_contiguous:
        raise TensorContractError("canonical tensor must be C-contiguous with declared wire dtype")
    return contiguous, contiguous.tobytes(order="C")


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    logical_dtype: str
    wire_dtype: str
    byte_order: str
    memory_order: str
    units: str
    coordinate_frame: str
    tensor_index_order: str
    semantic_tag: str
    pixel_transform: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.shape or any(int(dim) <= 0 for dim in self.shape):
            raise TensorContractError("tensor shape must contain positive dimensions")
        if self.byte_order != "little" or self.memory_order != "C":
            raise TensorContractError("canonical tensor wire representation must be little-endian C-order")
        if not self.units or not self.coordinate_frame or not self.tensor_index_order or not self.semantic_tag:
            raise TensorContractError("tensor units, frame, index order, and semantic tag are required")
        if self.pixel_transform is not None:
            matrix = np.asarray(self.pixel_transform, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise TensorContractError("pixel_transform must be a finite 3x3 matrix")

    def to_mapping(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "logical_dtype": self.logical_dtype,
            "wire_dtype": self.wire_dtype,
            "byte_order": self.byte_order,
            "memory_order": self.memory_order,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "tensor_index_order": self.tensor_index_order,
            "semantic_tag": self.semantic_tag,
            "pixel_transform": [list(row) for row in self.pixel_transform] if self.pixel_transform is not None else None,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "TensorSpec":
        matrix = mapping.get("pixel_transform")
        return cls(
            shape=tuple(int(dim) for dim in mapping["shape"]),  # type: ignore[index]
            logical_dtype=str(mapping["logical_dtype"]),
            wire_dtype=str(mapping["wire_dtype"]),
            byte_order=str(mapping["byte_order"]),
            memory_order=str(mapping["memory_order"]),
            units=str(mapping["units"]),
            coordinate_frame=str(mapping["coordinate_frame"]),
            tensor_index_order=str(mapping["tensor_index_order"]),
            semantic_tag=str(mapping["semantic_tag"]),
            pixel_transform=tuple(tuple(float(value) for value in row) for row in matrix) if matrix is not None else None,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CanonicalTensorRef:
    tensor_id: str
    spec: TensorSpec
    byte_length: int
    payload_sha256: str
    canonical_tensor_digest: str

    def __post_init__(self) -> None:
        if not self.tensor_id.startswith("sha256:"):
            raise TensorContractError("tensor_id must be content addressed")
        if self.byte_length <= 0:
            raise TensorContractError("tensor byte_length must be positive")
        if len(self.payload_sha256) != 64 or len(self.canonical_tensor_digest) != 64:
            raise TensorContractError("tensor digests must be SHA-256 hex strings")

    def to_mapping(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "spec": self.spec.to_mapping(),
            "byte_length": self.byte_length,
            "payload_sha256": self.payload_sha256,
            "canonical_tensor_digest": self.canonical_tensor_digest,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "CanonicalTensorRef":
        return cls(
            tensor_id=str(mapping["tensor_id"]),
            spec=TensorSpec.from_mapping(mapping["spec"]),  # type: ignore[arg-type]
            byte_length=int(mapping["byte_length"]),
            payload_sha256=str(mapping["payload_sha256"]),
            canonical_tensor_digest=str(mapping["canonical_tensor_digest"]),
        )


class CanonicalTensorStore:
    """Stores canonical bytes and returns references, never live mutable arrays."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        array: np.ndarray,
        *,
        units: str,
        coordinate_frame: str,
        tensor_index_order: str,
        semantic_tag: str,
        pixel_transform: np.ndarray | None = None,
        wire_dtype: str | None = None,
    ) -> CanonicalTensorRef:
        if wire_dtype is None:
            wire_dtype = {"float32": "<f4", "float64": "<f8", "uint8": "|u1", "int64": "<i8"}.get(array.dtype.name)
        if wire_dtype is None:
            raise TensorContractError(f"no canonical wire dtype for {array.dtype}")
        canonical, payload = _wire_array(array, wire_dtype)
        logical_dtype = np.dtype(array.dtype).name
        spec = TensorSpec(
            shape=tuple(int(dim) for dim in canonical.shape),
            logical_dtype=logical_dtype,
            wire_dtype=wire_dtype,
            byte_order="little",
            memory_order="C",
            units=units,
            coordinate_frame=coordinate_frame,
            tensor_index_order=tensor_index_order,
            semantic_tag=semantic_tag,
            pixel_transform=tuple(tuple(float(value) for value in row) for row in pixel_transform) if pixel_transform is not None else None,
        )
        spec_json = _canonical_json(spec.to_mapping())
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        digest = hashlib.sha256(spec_json + b"\x00" + payload).hexdigest()
        ref = CanonicalTensorRef(
            tensor_id=f"sha256:{digest}",
            spec=spec,
            byte_length=len(payload),
            payload_sha256=payload_sha256,
            canonical_tensor_digest=digest,
        )
        data_path = self.root / f"{digest}.bin"
        spec_path = self.root / f"{digest}.json"
        if data_path.exists() or spec_path.exists():
            if not data_path.exists() or not spec_path.exists():
                raise TensorContractError("incomplete tensor store entry")
            if data_path.read_bytes() != payload or json.loads(spec_path.read_text(encoding="utf-8")) != ref.to_mapping():
                raise TensorContractError("content-address collision or mutation detected")
            return ref
        data_path.write_bytes(payload)
        spec_path.write_bytes(_canonical_json(ref.to_mapping()))
        return ref

    def get(self, ref: CanonicalTensorRef) -> np.ndarray:
        data_path = self.root / f"{ref.canonical_tensor_digest}.bin"
        spec_path = self.root / f"{ref.canonical_tensor_digest}.json"
        if not data_path.is_file() or not spec_path.is_file():
            raise TensorContractError(f"missing tensor store entry {ref.tensor_id}")
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        if stored != ref.to_mapping():
            raise TensorContractError("tensor reference metadata does not match stored metadata")
        payload = data_path.read_bytes()
        if len(payload) != ref.byte_length or hashlib.sha256(payload).hexdigest() != ref.payload_sha256:
            raise TensorContractError("tensor bytes are truncated or mutated")
        spec_json = _canonical_json(ref.spec.to_mapping())
        digest = hashlib.sha256(spec_json + b"\x00" + payload).hexdigest()
        if digest != ref.canonical_tensor_digest:
            raise TensorContractError("tensor canonical digest mismatch")
        array = np.frombuffer(payload, dtype=np.dtype(ref.spec.wire_dtype)).reshape(ref.spec.shape)
        array.setflags(write=False)
        return array


__all__ = ["CanonicalTensorRef", "CanonicalTensorStore", "TensorContractError", "TensorSpec"]
