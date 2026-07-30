"""Zero-copy-capable binary envelope codec shared by the UniDepth experiment.

The codec represents a message as one compact header plus one memoryview per
payload. ``write_binary_envelope`` and ``read_binary_envelope`` use ``os.writev``
and ``os.readv`` without concatenating tensors. HTTP adds only a bounded
header-length prefix, then streams the same header and payload vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import struct
from typing import AsyncIterable, Mapping, Sequence


MAGIC = b"EGOZC001"
CONTENT_TYPE = "application/vnd.ego.binary-envelope"
# HTTP bodies begin with this bounded header-length prefix, followed by the
# existing vector header and its unflattened payload vectors.  The envelope header
# itself deliberately remains unchanged so the host-path prototype stays valid.
_HTTP_PREFIX = struct.Struct("!I")
_PREFIX = struct.Struct("!8sI")
_PART_PREFIX = struct.Struct("!HHHQ")
_DIMENSION = struct.Struct("!Q")
MAX_HEADER_BYTES = 1 << 20
MAX_PART_BYTES = 1 << 31
MAX_PARTS = 4096


class BinaryEnvelopeError(ValueError):
    """Raised when a binary envelope is malformed or cannot remain zero-copy."""


@dataclass(frozen=True)
class BinaryPart:
    """One named payload and the tensor metadata carried by its header."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    data: memoryview


@dataclass(frozen=True)
class BinaryEnvelope:
    """Header plus payload vectors suitable for vectored output."""

    header: bytes
    parts: tuple[BinaryPart, ...]

    @property
    def iovecs(self) -> tuple[memoryview, ...]:
        """Return header and unflattened payloads for ``writev``/``sendmsg``."""
        return (memoryview(self.header), *(part.data for part in self.parts))


@dataclass(frozen=True)
class _HeaderPart:
    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_length: int


def _bytes_view(data: bytes | bytearray | memoryview) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise BinaryEnvelopeError("part data must support the buffer protocol") from exc
    if not view.c_contiguous:
        raise BinaryEnvelopeError("part data must be C-contiguous for a zero-copy envelope")
    try:
        return view.cast("B")
    except TypeError as exc:
        raise BinaryEnvelopeError("part data must be byte-addressable") from exc


def _encoded_text(value: str, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise BinaryEnvelopeError(f"part {label} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise BinaryEnvelopeError(f"part {label} must be valid UTF-8") from exc
    if len(encoded) > 0xFFFF:
        raise BinaryEnvelopeError(f"part {label} is too long")
    return encoded


def _validated_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(shape)
    if len(result) > 0xFFFF:
        raise BinaryEnvelopeError("part rank is too large")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in result):
        raise BinaryEnvelopeError("part shape dimensions must be non-negative integers")
    return result


def _parse_header(header: bytes | bytearray | memoryview) -> tuple[bytes, tuple[_HeaderPart, ...]]:
    header_view = _bytes_view(header)
    if header_view.nbytes < _PREFIX.size:
        raise BinaryEnvelopeError("binary envelope header is truncated")
    magic, part_count = _PREFIX.unpack_from(header_view, 0)
    if magic != MAGIC:
        raise BinaryEnvelopeError("binary envelope magic/version is unsupported")
    if part_count > MAX_PARTS:
        raise BinaryEnvelopeError("binary envelope has too many parts")

    offset = _PREFIX.size
    seen_names: set[str] = set()
    parsed: list[_HeaderPart] = []
    for _ in range(part_count):
        if offset + _PART_PREFIX.size > header_view.nbytes:
            raise BinaryEnvelopeError("binary envelope part header is truncated")
        name_size, dtype_size, rank, byte_length = _PART_PREFIX.unpack_from(header_view, offset)
        offset += _PART_PREFIX.size
        dimensions_size = rank * _DIMENSION.size
        text_end = offset + dimensions_size + name_size + dtype_size
        if text_end > header_view.nbytes:
            raise BinaryEnvelopeError("binary envelope part metadata is truncated")
        shape = tuple(_DIMENSION.unpack_from(header_view, offset + item * _DIMENSION.size)[0] for item in range(rank))
        offset += dimensions_size
        try:
            name = bytes(header_view[offset:offset + name_size]).decode("utf-8")
            offset += name_size
            dtype = bytes(header_view[offset:offset + dtype_size]).decode("utf-8")
            offset += dtype_size
        except UnicodeDecodeError as exc:
            raise BinaryEnvelopeError("binary envelope name/dtype is not UTF-8") from exc
        if not name or not dtype or name in seen_names:
            raise BinaryEnvelopeError("binary envelope has empty or duplicate part names")
        if byte_length > MAX_PART_BYTES:
            raise BinaryEnvelopeError("binary envelope declared part exceeds maximum size")
        seen_names.add(name)
        parsed.append(_HeaderPart(name, shape, dtype, byte_length))
    if offset != header_view.nbytes:
        raise BinaryEnvelopeError("binary envelope header has trailing bytes")
    return bytes(header_view), tuple(parsed)


def build_binary_envelope(
    parts: Mapping[str, tuple[bytes | bytearray | memoryview, Sequence[int], str]],
) -> BinaryEnvelope:
    """Build an envelope header without copying or concatenating payload bytes.

    ``parts`` preserves mapping order.  Each value is ``(data, shape, dtype)``;
    callers that need JSON metadata should include it as a normal e.g. ``json``
    part with an empty shape.  Payloads remain memoryviews over the supplied
    buffers, while only their small headers are materialized.
    """
    if not parts:
        raise BinaryEnvelopeError("binary envelope must contain at least one part")
    if len(parts) > MAX_PARTS:
        raise BinaryEnvelopeError("binary envelope has too many parts")

    header = bytearray(_PREFIX.pack(MAGIC, len(parts)))
    built_parts: list[BinaryPart] = []
    seen_names: set[str] = set()
    for name, (data, shape, dtype) in parts.items():
        name_bytes = _encoded_text(name, "name")
        dtype_bytes = _encoded_text(dtype, "dtype")
        if name in seen_names:
            raise BinaryEnvelopeError(f"duplicate part {name!r}")
        seen_names.add(name)
        validated_shape = _validated_shape(shape)
        view = _bytes_view(data)
        if view.nbytes > MAX_PART_BYTES:
            raise BinaryEnvelopeError("binary envelope declared part exceeds maximum size")
        header.extend(_PART_PREFIX.pack(len(name_bytes), len(dtype_bytes), len(validated_shape), view.nbytes))
        for dimension in validated_shape:
            header.extend(_DIMENSION.pack(dimension))
        header.extend(name_bytes)
        header.extend(dtype_bytes)
        built_parts.append(BinaryPart(name, validated_shape, dtype, view))
    return BinaryEnvelope(bytes(header), tuple(built_parts))


def parse_binary_envelope(header: bytes | bytearray | memoryview, payloads: Sequence[bytes | bytearray | memoryview]) -> BinaryEnvelope:
    """Parse a header and vectored payloads without concatenating either stream."""
    parsed_header, header_parts = _parse_header(header)
    if len(payloads) != len(header_parts):
        raise BinaryEnvelopeError("binary envelope payload count does not match header")
    parts: list[BinaryPart] = []
    for index, (definition, payload) in enumerate(zip(header_parts, payloads)):
        view = _bytes_view(payload)
        if view.nbytes != definition.byte_length:
            raise BinaryEnvelopeError(
                f"binary envelope payload {index} length {view.nbytes} does not match declared {definition.byte_length}"
            )
        parts.append(BinaryPart(definition.name, definition.shape, definition.dtype, view))
    return BinaryEnvelope(parsed_header, tuple(parts))


def content_type_is_binary_envelope(content_type: str | None) -> bool:
    """Return whether an HTTP Content-Type selects the binary envelope codec."""
    return bool(content_type and content_type.split(";", 1)[0].strip().lower() == CONTENT_TYPE)


def binary_envelope_iovecs(envelope: BinaryEnvelope) -> tuple[memoryview, ...]:
    """Return HTTP framing and envelope vectors without joining tensor payloads."""
    if len(envelope.header) > MAX_HEADER_BYTES:
        raise BinaryEnvelopeError("binary envelope header exceeds maximum size")
    return (memoryview(_HTTP_PREFIX.pack(len(envelope.header))), *envelope.iovecs)


def parse_binary_envelope_body(body: bytes | bytearray | memoryview) -> BinaryEnvelope:
    """Parse one HTTP envelope body while keeping payloads as slices of ``body``.

    ASGI hands an application a received body buffer. This function never makes a
    second aggregate body or joins individual tensor payloads: parsed parts are
    memoryviews into that buffer. The bounded length prefix makes hostile declared
    headers/parts reject before an allocator can be driven by them.
    """
    view = _bytes_view(body)
    if view.nbytes < _HTTP_PREFIX.size:
        raise BinaryEnvelopeError("binary envelope HTTP header is truncated")
    (header_size,) = _HTTP_PREFIX.unpack_from(view, 0)
    if header_size > MAX_HEADER_BYTES:
        raise BinaryEnvelopeError("binary envelope declared header exceeds maximum size")
    header_start = _HTTP_PREFIX.size
    header_end = header_start + header_size
    if header_end > view.nbytes:
        raise BinaryEnvelopeError("binary envelope HTTP header is truncated")
    header = view[header_start:header_end]
    _parsed_header, definitions = _parse_header(header)
    offset = header_end
    payloads: list[memoryview] = []
    for definition in definitions:
        payload_end = offset + definition.byte_length
        if payload_end > view.nbytes:
            raise BinaryEnvelopeError("binary envelope payload is truncated")
        payloads.append(view[offset:payload_end])
        offset = payload_end
    if offset != view.nbytes:
        raise BinaryEnvelopeError("binary envelope body has trailing bytes")
    return parse_binary_envelope(header, payloads)


async def read_binary_envelope_stream(chunks: AsyncIterable[bytes | bytearray | memoryview]) -> BinaryEnvelope:
    """Read an HTTP byte stream into per-part buffers without an aggregate body.

    HTTP/ASGI can split any framing boundary. This incremental counterpart to
    :func:`read_binary_envelope` first bounds and fills the compact header, then
    allocates one buffer per declared payload and copies each received slice directly
    into its destination. No large temporary request body is assembled.
    """
    prefix = bytearray(_HTTP_PREFIX.size)
    prefix_offset = 0
    header: bytearray | None = None
    header_offset = 0
    definitions: tuple[_HeaderPart, ...] | None = None
    buffers: list[bytearray] = []
    part_index = 0
    part_offset = 0

    async for chunk in chunks:
        incoming = _bytes_view(chunk)
        offset = 0
        while offset < incoming.nbytes:
            if prefix_offset < _HTTP_PREFIX.size:
                count = min(_HTTP_PREFIX.size - prefix_offset, incoming.nbytes - offset)
                prefix[prefix_offset:prefix_offset + count] = incoming[offset:offset + count]
                prefix_offset += count
                offset += count
                if prefix_offset < _HTTP_PREFIX.size:
                    continue
                (header_size,) = _HTTP_PREFIX.unpack(prefix)
                if header_size > MAX_HEADER_BYTES:
                    raise BinaryEnvelopeError("binary envelope declared header exceeds maximum size")
                header = bytearray(header_size)
                if not header:
                    _parsed_header, definitions = _parse_header(header)
                    buffers = [bytearray(definition.byte_length) for definition in definitions]
                continue
            assert header is not None
            if header_offset < len(header):
                count = min(len(header) - header_offset, incoming.nbytes - offset)
                header[header_offset:header_offset + count] = incoming[offset:offset + count]
                header_offset += count
                offset += count
                if header_offset < len(header):
                    continue
                _parsed_header, definitions = _parse_header(header)
                buffers = [bytearray(definition.byte_length) for definition in definitions]
                continue
            assert definitions is not None
            while part_index < len(buffers) and part_offset == len(buffers[part_index]):
                part_index += 1
                part_offset = 0
            if part_index == len(buffers):
                raise BinaryEnvelopeError("binary envelope body has trailing bytes")
            target = buffers[part_index]
            count = min(len(target) - part_offset, incoming.nbytes - offset)
            target[part_offset:part_offset + count] = incoming[offset:offset + count]
            part_offset += count
            offset += count

    if prefix_offset < _HTTP_PREFIX.size or header is None or header_offset < len(header):
        raise BinaryEnvelopeError("binary envelope HTTP header is truncated")
    assert definitions is not None
    while part_index < len(buffers) and part_offset == len(buffers[part_index]):
        part_index += 1
        part_offset = 0
    if part_index != len(buffers):
        raise BinaryEnvelopeError("binary envelope payload is truncated")
    return parse_binary_envelope(header, buffers)


def _advance_iovecs(vectors: list[memoryview], consumed: int) -> list[memoryview]:
    """Drop a partial ``readv``/``writev`` prefix while retaining buffer views."""
    while vectors and consumed >= vectors[0].nbytes:
        consumed -= vectors[0].nbytes
        vectors.pop(0)
    if consumed and vectors:
        vectors[0] = vectors[0][consumed:]
    return vectors


def write_binary_envelope(fd: int, envelope: BinaryEnvelope) -> int:
    """Write header and all payloads through ``os.writev`` without flattening.

    Handles short writes by slicing memoryviews; callers retain ownership of the
    payload buffers for the duration of this call.
    """
    pending = list(envelope.iovecs)
    written_total = 0
    while pending:
        written = os.writev(fd, pending)
        if written <= 0:
            raise BinaryEnvelopeError("writev made no progress")
        written_total += written
        _advance_iovecs(pending, written)
    return written_total


def read_binary_envelope(fd: int, header: bytes | bytearray | memoryview) -> BinaryEnvelope:
    """Read declared payload vectors with ``os.readv`` and parse the envelope.

    The header is normally obtained by a transport-specific bounded-header read.
    Each payload gets its own bytearray because incoming bytes require ownership;
    no aggregate body or per-tensor concatenation buffer is allocated.
    """
    _parsed_header, definitions = _parse_header(header)
    buffers = [bytearray(definition.byte_length) for definition in definitions]
    pending = [memoryview(buffer) for buffer in buffers if buffer]
    while pending:
        read = os.readv(fd, pending)
        if read <= 0:
            raise BinaryEnvelopeError("readv ended before all declared payload bytes arrived")
        _advance_iovecs(pending, read)
    return parse_binary_envelope(header, buffers)


__all__ = [
    "MAGIC",
    "CONTENT_TYPE",
    "MAX_HEADER_BYTES",
    "MAX_PART_BYTES",
    "BinaryEnvelope",
    "BinaryEnvelopeError",
    "BinaryPart",
    "binary_envelope_iovecs",
    "build_binary_envelope",
    "content_type_is_binary_envelope",
    "parse_binary_envelope",
    "parse_binary_envelope_body",
    "read_binary_envelope",
    "read_binary_envelope_stream",
    "write_binary_envelope",
]
