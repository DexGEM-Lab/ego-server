"""Strict metadata-plus-binary multipart codec for algorithm requests/results.

The codec is independent of HTTP clients and model callers. JSON carries only
schema/identity/descriptors; tensor and media bytes remain binary parts and are
verified by shape, dtype, byte length, and SHA-256 on both encode and decode.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as default_email_policy
from typing import Mapping, Sequence

import numpy as np

from ego_annotation.typed_contracts import BinaryAsset, TypedContractError, TypedTensor
from ego_annotation.scripted.droid_rgbd import DroidNativeSensorDepthAbiPayload


class MultipartCodecError(TypedContractError):
    """Raised when a multipart message violates the binary wire contract."""


@dataclass(frozen=True)
class DecodedPart:
    name: str
    data: bytes
    descriptor: Mapping[str, object]


@dataclass(frozen=True)
class MultipartMessage:
    metadata: Mapping[str, object]
    parts: Mapping[str, DecodedPart]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MultipartCodecError(f"metadata is not canonical JSON: {exc}") from exc


def _wire_descriptor(name: str, value: object) -> tuple[dict[str, object], bytes]:
    if isinstance(value, TypedTensor):
        return value.descriptor(name), value.canonical_bytes
    if isinstance(value, BinaryAsset):
        return value.descriptor(name), value.data
    if isinstance(value, DroidNativeSensorDepthAbiPayload):
        descriptor = value.to_mapping()
        descriptor.update({
            "name": name,
            "kind": "sealed_native_depth",
            "shape": list(value.shape),
            "dtype": "float32",
            "wire_dtype": "<f4",
            "byte_length": len(value.canonical_bytes),
            "payload_sha256": value.payload_sha256,
        })
        return descriptor, value.canonical_bytes
    raise MultipartCodecError(f"unsupported binary part type {type(value).__name__}")


def encode_future_multipart(metadata: Mapping[str, object], parts: Mapping[str, object], *, boundary: str | None = None) -> tuple[bytes, str]:
    """Encode one JSON metadata part and exact binary parts."""

    if boundary is None:
        boundary = f"ego-api-{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", boundary):
        raise MultipartCodecError("multipart boundary contains invalid characters")
    descriptors: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes, str]] = []
    for name, value in parts.items():
        if not name or name in {"metadata", "parts"}:
            raise MultipartCodecError(f"invalid or reserved binary part name {name!r}")
        descriptor, payload = _wire_descriptor(name, value)
        if descriptor["name"] != name:
            raise MultipartCodecError(f"binary descriptor name mismatch for {name}")
        expected_digest = hashlib.sha256(payload).hexdigest()
        if descriptor.get("payload_sha256") != expected_digest:
            raise MultipartCodecError(f"binary part {name} digest does not match bytes")
        if int(descriptor.get("byte_length", -1)) != len(payload):
            raise MultipartCodecError(f"binary part {name} byte length does not match bytes")
        descriptors.append(descriptor)
        payloads.append((name, payload, str(descriptor.get("kind", "binary"))))
    envelope = dict(metadata)
    if envelope.get("wire_schema") not in (None, "ego.annotation.multipart.v1"):
        raise MultipartCodecError("unsupported multipart wire schema")
    envelope["wire_schema"] = "ego.annotation.multipart.v1"
    envelope["parts"] = descriptors
    body = bytearray()
    delimiter = b"--" + boundary.encode("ascii")
    body.extend(delimiter + b"\r\n")
    body.extend(b"Content-Disposition: form-data; name=\"metadata\"\r\n")
    body.extend(b"Content-Type: application/json\r\n\r\n")
    body.extend(_canonical_json(envelope) + b"\r\n")
    for name, payload, kind in payloads:
        descriptor = next(item for item in descriptors if item["name"] == name)
        body.extend(delimiter + b"\r\n")
        body.extend(f"Content-Disposition: form-data; name=\"{name}\"; kind=\"{kind}\"\r\n".encode("ascii"))
        body.extend(f"Content-Type: application/octet-stream\r\nX-Part-Sha256: {descriptor['payload_sha256']}\r\n\r\n".encode("ascii"))
        body.extend(payload + b"\r\n")
    body.extend(delimiter + b"--\r\n")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _content_type_boundary(content_type: str) -> str:
    if not content_type.lower().startswith("multipart/form-data"):
        raise MultipartCodecError("response/request must use multipart/form-data")
    header = f"Content-Type: {content_type}\r\n\r\n".encode("ascii")
    message = BytesParser(policy=default_email_policy).parsebytes(header)
    boundary = message.get_param("boundary", header="content-type")
    if not boundary:
        raise MultipartCodecError("multipart Content-Type has no boundary")
    return str(boundary)


def encode_raw_multipart(metadata: Mapping[str, object], parts: Mapping[str, object], *, boundary: str | None = None) -> tuple[bytes, str]:
    """Encode the frozen service wire: metadata plus original named parts.

    Unlike :func:`encode_future_multipart`, this function does not add a schema, parts
    manifest, protocol, request id, or any generic envelope keys. The live route
    adapters use it with the exact field names consumed by the frozen parsers.
    """

    if boundary is None:
        boundary = f"ego-route-{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", boundary):
        raise MultipartCodecError("multipart boundary contains invalid characters")
    descriptor_payloads: list[tuple[str, dict[str, object], bytes]] = []
    for name, value in parts.items():
        descriptor, payload = _wire_descriptor(name, value)
        descriptor_payloads.append((name, descriptor, payload))
    body = bytearray()
    delimiter = b"--" + boundary.encode("ascii")
    body.extend(delimiter + b"\r\n")
    body.extend(b"Content-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n")
    body.extend(_canonical_json(dict(metadata)) + b"\r\n")
    for name, descriptor, payload in descriptor_payloads:
        body.extend(delimiter + b"\r\n")
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype") or descriptor.get("wire_dtype")
        disposition = f"Content-Disposition: form-data; name=\"{name}\""
        if shape is not None:
            disposition += "; shape=\"" + ",".join(str(int(dim)) for dim in shape) + "\""
        if dtype is not None:
            disposition += f"; dtype=\"{dtype}\""
        if descriptor.get("kind") == "media":
            disposition += f"; kind=\"image\"; media_type=\"{descriptor['media_type']}\"; source_index=\"{descriptor['source_frame_indices'][0]}\""
        body.extend((disposition + "\r\nContent-Type: application/octet-stream\r\nX-Part-Sha256: " + str(descriptor["payload_sha256"]) + "\r\n\r\n").encode("ascii"))
        body.extend(payload + b"\r\n")
    body.extend(delimiter + b"--\r\n")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _parse_multipart_message(body: bytes, content_type: str) -> tuple[dict[str, object], dict[str, tuple[bytes, dict[str, str]]]]:
    _content_type_boundary(content_type)
    header = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default_email_policy).parsebytes(header + body)
    if not message.is_multipart():
        raise MultipartCodecError("multipart body is not multipart")
    metadata: dict[str, object] | None = None
    parsed: dict[str, tuple[bytes, dict[str, str]]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            raise MultipartCodecError("multipart part has no name")
        if name in parsed or name == "metadata" and metadata is not None:
            raise MultipartCodecError(f"duplicate multipart part {name!r}")
        payload = part.get_payload(decode=True)
        if payload is None:
            raise MultipartCodecError(f"multipart part {name!r} has no binary payload")
        if name == "metadata":
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MultipartCodecError("metadata part is invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise MultipartCodecError("metadata part must be a JSON object")
            metadata = decoded
            continue
        params: dict[str, str] = {}
        shape = part.get_param("shape", header="content-disposition")
        dtype = part.get_param("dtype", header="content-disposition")
        if shape is not None:
            params["shape"] = str(shape)
        if dtype is not None:
            params["dtype"] = str(dtype)
        digest = part.get("X-Part-Sha256")
        if digest:
            params["header_digest"] = str(digest)
        parsed[str(name)] = (bytes(payload), params)
    if metadata is None:
        raise MultipartCodecError("missing metadata part")
    return metadata, parsed


def decode_raw_multipart(body: bytes, content_type: str) -> MultipartMessage:
    """Parse frozen route multipart without requiring a generic manifest."""

    metadata, parsed = _parse_multipart_message(body, content_type)
    result: dict[str, DecodedPart] = {}
    for name, (payload, params) in parsed.items():
        digest = hashlib.sha256(payload).hexdigest()
        if params.get("header_digest") not in {None, "", digest}:
            raise MultipartCodecError(f"binary part {name} digest mismatch")
        descriptor: dict[str, object] = {
            "name": name,
            "kind": "tensor" if "shape" in params and "dtype" in params else "binary",
            "byte_length": len(payload),
            "payload_sha256": digest,
        }
        if "shape" in params:
            shape = [int(item) for item in params["shape"].split(",") if item]
            try:
                expected_bytes = int(np.prod(shape)) * np.dtype(params["dtype"]).itemsize
            except (TypeError, ValueError) as exc:
                raise MultipartCodecError(f"binary part {name} has invalid frozen shape/dtype") from exc
            if expected_bytes != len(payload):
                raise MultipartCodecError(f"binary part {name} frozen shape/dtype byte length mismatch")
            descriptor["shape"] = shape
            descriptor["dtype"] = params["dtype"]
            descriptor["wire_dtype"] = params["dtype"]
        result[name] = DecodedPart(name=name, data=payload, descriptor=descriptor)
    return MultipartMessage(metadata=metadata, parts=result)


def decode_future_multipart(body: bytes, content_type: str) -> MultipartMessage:
    """Parse and validate metadata descriptors and every binary part."""

    metadata, parsed = _parse_multipart_message(body, content_type)
    if metadata.get("wire_schema") != "ego.annotation.multipart.v1":
        raise MultipartCodecError("missing or unsupported multipart wire schema")
    descriptors_value = metadata.get("parts")
    if not isinstance(descriptors_value, list):
        raise MultipartCodecError("metadata must declare a parts list")
    descriptors: dict[str, Mapping[str, object]] = {}
    for descriptor in descriptors_value:
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("name"), str):
            raise MultipartCodecError("invalid binary part descriptor")
        name = str(descriptor["name"])
        if name in descriptors:
            raise MultipartCodecError(f"duplicate descriptor {name!r}")
        descriptors[name] = descriptor
    if set(descriptors) != set(parsed):
        raise MultipartCodecError(f"declared parts {sorted(descriptors)} differ from wire parts {sorted(parsed)}")
    result: dict[str, DecodedPart] = {}
    for name, descriptor in descriptors.items():
        payload, headers = parsed[name]
        digest = hashlib.sha256(payload).hexdigest()
        if descriptor.get("payload_sha256") != digest or headers.get("header_digest") not in {"", digest}:
            raise MultipartCodecError(f"binary part {name} digest mismatch")
        if int(descriptor.get("byte_length", -1)) != len(payload):
            raise MultipartCodecError(f"binary part {name} byte length mismatch")
        kind = str(descriptor.get("kind", ""))
        if kind == "tensor":
            shape = descriptor.get("shape")
            wire_dtype = descriptor.get("wire_dtype")
            if not isinstance(shape, list) or not isinstance(wire_dtype, str):
                raise MultipartCodecError(f"tensor part {name} lacks shape/wire_dtype")
            try:
                expected_bytes = int(np.prod(tuple(int(dim) for dim in shape))) * np.dtype(wire_dtype).itemsize
            except (TypeError, ValueError) as exc:
                raise MultipartCodecError(f"tensor part {name} has invalid shape/dtype") from exc
            if expected_bytes != len(payload):
                raise MultipartCodecError(f"tensor part {name} shape/dtype byte length mismatch")
            digest_spec = {key: descriptor.get(key) for key in ("shape", "dtype", "wire_dtype", "units", "coordinate_frame", "tensor_index_order", "semantic_tag", "provenance", "pixel_transform")}
            tensor_digest = hashlib.sha256(_canonical_json(digest_spec) + b"\x00" + payload).hexdigest()
            if descriptor.get("canonical_tensor_digest") != tensor_digest:
                raise MultipartCodecError(f"tensor part {name} canonical tensor digest mismatch")
        elif kind == "sealed_native_depth":
            if descriptor.get("field_name") != "native_sensor_depth_abi_payload_m" or descriptor.get("semantic_tag") != "stock_depthvideo_gather_slots_v1":
                raise MultipartCodecError("sealed native depth part lost its ABI identity")
        elif kind != "media":
            raise MultipartCodecError(f"unsupported binary part kind {kind!r}")
        result[name] = DecodedPart(name=name, data=payload, descriptor=descriptor)
    return MultipartMessage(metadata=metadata, parts=result)


def materialize_part(part: DecodedPart) -> object:
    """Materialize a verified binary part without changing its bytes."""

    descriptor = part.descriptor
    kind = descriptor.get("kind")
    if kind == "tensor":
        dtype = np.dtype(str(descriptor["wire_dtype"]))
        array = np.frombuffer(part.data, dtype=dtype).reshape(tuple(int(dim) for dim in descriptor["shape"]))
        array = np.array(array, copy=True, order="C")
        logical_dtype = str(descriptor["dtype"])
        if array.dtype.name != logical_dtype:
            array = array.astype(np.dtype(logical_dtype), copy=False)
        array.setflags(write=False)
        return TypedTensor(
            array=array,
            units=str(descriptor["units"]),
            coordinate_frame=str(descriptor["coordinate_frame"]),
            tensor_index_order=str(descriptor["tensor_index_order"]),
            semantic_tag=str(descriptor["semantic_tag"]),
            provenance=dict(descriptor.get("provenance", {})),
            pixel_transform=tuple(tuple(float(item) for item in row) for row in descriptor["pixel_transform"]) if descriptor.get("pixel_transform") is not None else None,
        )
    if kind == "media":
        return BinaryAsset(
            data=part.data,
            media_type=str(descriptor["media_type"]),
            source_artifact_id=str(descriptor["source_artifact_id"]),
            source_frame_indices=tuple(int(item) for item in descriptor["source_frame_indices"]),
        )
    if kind == "sealed_native_depth":
        return DroidNativeSensorDepthAbiPayload(
            canonical_bytes=part.data,
            spec=dict(descriptor["spec"]),
            operation_trace=("pack", "serialize", "deserialize"),
        )
    raise MultipartCodecError(f"cannot materialize part kind {kind!r}")


__all__ = ["DecodedPart", "MultipartCodecError", "MultipartMessage", "decode_future_multipart", "decode_raw_multipart", "encode_future_multipart", "encode_raw_multipart", "materialize_part"]
