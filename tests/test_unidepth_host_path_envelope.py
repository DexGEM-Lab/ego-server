"""Tests for the measurement-only vectored binary-envelope prototype."""
from __future__ import annotations

import asyncio
import os
import struct

import numpy as np
import pytest

from ego_annotation.serving.binary_envelope import (
    MAGIC,
    MAX_PART_BYTES,
    BinaryEnvelopeError,
    binary_envelope_iovecs,
    build_binary_envelope,
    parse_binary_envelope,
    parse_binary_envelope_body,
    read_binary_envelope,
    read_binary_envelope_stream,
    write_binary_envelope,
)


def test_binary_envelope_round_trips_numpy_bytes_without_payload_flattening():
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    confidence = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    envelope = build_binary_envelope({
        "metadata": (b'{"request_id":"round-trip"}', (), "application/json"),
        "depth_m": (depth, depth.shape, "float32"),
        "confidence": (confidence, confidence.shape, "float32"),
    })

    assert envelope.header.startswith(MAGIC)
    assert len(envelope.iovecs) == 4
    parsed = parse_binary_envelope(envelope.header, envelope.iovecs[1:])

    assert [(part.name, part.shape, part.dtype) for part in parsed.parts] == [
        ("metadata", (), "application/json"),
        ("depth_m", (3, 4), "float32"),
        ("confidence", (3, 4), "float32"),
    ]
    np.testing.assert_array_equal(np.frombuffer(parsed.parts[1].data, dtype=np.float32).reshape(3, 4), depth)
    np.testing.assert_array_equal(np.frombuffer(parsed.parts[2].data, dtype=np.float32).reshape(3, 4), confidence)
    # The parser retains a view of the original ndarray buffer, rather than a
    # concatenated copy.  The exact memoryview parent may be an ndarray view.
    assert parsed.parts[1].data.nbytes == depth.nbytes
    assert parsed.parts[1].data.readonly is False


def test_binary_envelope_uses_os_writev_and_readv_without_body_flattening():
    source = np.arange(8, dtype=np.float32)
    envelope = build_binary_envelope({
        "metadata": (b'{"kind":"test"}', (), "application/json"),
        "depth_m": (source, source.shape, "float32"),
    })
    read_fd, write_fd = os.pipe()
    try:
        assert write_binary_envelope(write_fd, envelope) == sum(vector.nbytes for vector in envelope.iovecs)
        received_header = os.read(read_fd, len(envelope.header))
        assert received_header == envelope.header
        parsed = read_binary_envelope(read_fd, received_header)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert bytes(parsed.parts[0].data) == b'{"kind":"test"}'
    np.testing.assert_array_equal(np.frombuffer(parsed.parts[1].data, dtype=np.float32), source)


def test_binary_envelope_rejects_truncated_header_wrong_payload_size_and_noncontiguous_input():
    envelope = build_binary_envelope({"rgb": (b"abcdef", (2, 3), "uint8")})
    with pytest.raises(BinaryEnvelopeError, match="truncated"):
        parse_binary_envelope(envelope.header[:-1], envelope.iovecs[1:])
    with pytest.raises(BinaryEnvelopeError, match="length"):
        parse_binary_envelope(envelope.header, (b"abc",))

    noncontiguous = np.ones((4, 4), dtype=np.float32)[:, ::-1]
    with pytest.raises(BinaryEnvelopeError, match="C-contiguous"):
        build_binary_envelope({"depth": (noncontiguous, noncontiguous.shape, "float32")})


def test_binary_envelope_http_body_rejects_truncated_magic_oversized_and_duplicate_parts():
    envelope = build_binary_envelope({"alpha": (b"one", (3,), "uint8"), "bravo": (b"two", (3,), "uint8")})
    body = b"".join(binary_envelope_iovecs(envelope))
    with pytest.raises(BinaryEnvelopeError, match="truncated"):
        parse_binary_envelope_body(body[:3])

    wrong_magic = bytearray(body)
    wrong_magic[4] ^= 0xFF
    with pytest.raises(BinaryEnvelopeError, match="magic"):
        parse_binary_envelope_body(wrong_magic)

    oversized = bytearray(body)
    # HTTP length prefix (4), envelope prefix (12), then the part prefix's Q.
    struct.pack_into("!Q", oversized, 4 + 12 + 6, MAX_PART_BYTES + 1)
    with pytest.raises(BinaryEnvelopeError, match="exceeds"):
        parse_binary_envelope_body(oversized)

    duplicate = bytearray(body[:4] + bytes(body[4:]).replace(b"bravo", b"alpha"))
    with pytest.raises(BinaryEnvelopeError, match="duplicate"):
        parse_binary_envelope_body(duplicate)


def test_binary_envelope_stream_reader_handles_split_frames_without_aggregate_body():
    envelope = build_binary_envelope({"metadata": (b"{}", (), "application/json"), "rgb": (b"abcdef", (2, 3), "uint8")})
    body = b"".join(binary_envelope_iovecs(envelope))

    async def chunks():
        for offset in range(0, len(body), 3):
            yield body[offset:offset + 3]

    parsed = asyncio.run(read_binary_envelope_stream(chunks()))
    assert [part.name for part in parsed.parts] == ["metadata", "rgb"]
    assert parsed.parts[1].data.tobytes() == b"abcdef"


def test_binary_envelope_http_iovecs_keep_large_parts_separate():
    depth = np.zeros((1024, 1024), dtype=np.float32)
    envelope = build_binary_envelope({"metadata": (b"{}", (), "application/json"), "depth_m": (depth, depth.shape, "float32")})
    vectors = binary_envelope_iovecs(envelope)
    assert len(vectors) == 4  # HTTP prefix, envelope header, metadata, depth
    assert vectors[-1].nbytes == depth.nbytes
    assert vectors[-1].obj is depth
    assert sum(vector.nbytes for vector in vectors) == 4 + len(envelope.header) + 2 + depth.nbytes


def test_binary_envelope_rejects_extra_payload_vectors():
    envelope = build_binary_envelope({"rgb": (b"abc", (3,), "uint8")})
    with pytest.raises(BinaryEnvelopeError, match="payload count"):
        parse_binary_envelope(envelope.header, (b"abc", b"ignored"))
