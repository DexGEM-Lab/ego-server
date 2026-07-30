#!/usr/bin/env python3
"""Call one deployed Feishu Ray model route with typed multipart arrays."""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import secrets
import struct
import sys
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_ARRAY_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 2 * 1024 * 1024 * 1024
ARRAY_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
FILESYSTEM_METADATA_KEYS = {
    "path",
    "file",
    "filename",
    "directory",
    "dir",
    "root",
    "uri",
    "url",
    "input_video",
    "video_uri",
    "output_dir",
    "output_root",
    "source_path",
}
DTYPES: dict[str, tuple[str, int]] = {
    "bool": ("|b1", 1),
    "bool_": ("|b1", 1),
    "uint8": ("|u1", 1),
    "int8": ("|i1", 1),
    "uint16": ("<u2", 2),
    "int16": ("<i2", 2),
    "uint32": ("<u4", 4),
    "int32": ("<i4", 4),
    "uint64": ("<u8", 8),
    "int64": ("<i8", 8),
    "float16": ("<f2", 2),
    "float32": ("<f4", 4),
    "float64": ("<f8", 8),
    "|b1": ("|b1", 1),
    "|u1": ("|u1", 1),
    "|i1": ("|i1", 1),
    "<u2": ("<u2", 2),
    "<i2": ("<i2", 2),
    "<u4": ("<u4", 4),
    "<i4": ("<i4", 4),
    "<u8": ("<u8", 8),
    "<i8": ("<i8", 8),
    "<f2": ("<f2", 2),
    "<f4": ("<f4", 4),
    "<f8": ("<f8", 8),
    "=u2": ("<u2", 2),
    "=i2": ("<i2", 2),
    "=u4": ("<u4", 4),
    "=i4": ("<i4", 4),
    "=u8": ("<u8", 8),
    "=i8": ("<i8", 8),
    "=f2": ("<f2", 2),
    "=f4": ("<f4", 4),
    "=f8": ("<f8", 8),
}
DESCR_TO_NAME = {
    "|b1": "bool",
    "|u1": "uint8",
    "|i1": "int8",
    "<u2": "uint16",
    "<i2": "int16",
    "<u4": "uint32",
    "<i4": "int32",
    "<u8": "uint64",
    "<i8": "int64",
    "<f2": "float16",
    "<f4": "float32",
    "<f8": "float64",
}


class ServiceCallerError(RuntimeError):
    """A concrete request, transport, or response contract error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        response_received: bool | None = None,
        response_status: int | None = None,
        response_headers: dict[str, str] | None = None,
        raw_response_bytes: bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.response_received = response_received
        self.response_status = response_status
        self.response_headers = dict(response_headers) if response_headers is not None else None
        self.raw_response_bytes = bytes(raw_response_bytes) if raw_response_bytes is not None else None


RETRYABLE_SERVICE_CODES = {
    "BACKPRESSURE",
    "RESOURCE_EXHAUSTED",
    "RATE_LIMITED",
    "TOO_MANY_REQUESTS",
    "UNAVAILABLE",
    "SERVICE_UNAVAILABLE",
    "TRY_AGAIN",
}


def _retry_after_seconds(headers: dict[str, str] | None, error: Any) -> float | None:
    """Read an explicit retry delay without treating malformed hints as truth."""
    candidates: list[Any] = []
    if isinstance(error, dict):
        candidates.extend([error.get("retry_after_s"), error.get("retry_after")])
    if isinstance(headers, dict):
        candidates.append(headers.get("Retry-After"))
    for raw in candidates:
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value >= 0.0:
            return min(value, 300.0)
    return None


def retryable_response_info(report: Any) -> tuple[bool, float | None, str | None]:
    """Return retryability only for an explicit, not-yet-accepted response."""
    if not isinstance(report, dict):
        return False, None, None
    metadata = report.get("metadata")
    error = metadata.get("error") if isinstance(metadata, dict) else None
    status = report.get("http_status")
    explicit = isinstance(error, dict) and error.get("retryable") is True
    code = str(error.get("code") or "") if isinstance(error, dict) else ""
    try:
        status_retryable = int(status) in {408, 425, 429, 502, 503, 504}
    except (TypeError, ValueError, OverflowError):
        status_retryable = False
    retryable = bool(explicit or code.upper() in RETRYABLE_SERVICE_CODES or status_retryable)
    if not retryable:
        return False, None, code or None
    return True, _retry_after_seconds(report.get("response_headers"), error), code or None


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceCallerError("invalid_request_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ServiceCallerError("invalid_request_object", "request JSON must be an object")
    return payload


def validate_base_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceCallerError("invalid_base_url", "base_url must be a non-empty HTTP(S) URL")
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ServiceCallerError("invalid_base_url", "base_url must include http:// or https:// and a host")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ServiceCallerError("invalid_base_url", "base_url must not include credentials, a path, query, or fragment")
    return url


def validate_route(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ServiceCallerError("invalid_route", "route must start with exactly one slash")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not parsed.path or parsed.path.startswith("//"):
        raise ServiceCallerError("invalid_route", "route must be an absolute path without host, query, or fragment")
    if any(segment in {"", ".", ".."} for segment in parsed.path.split("/")[1:]):
        raise ServiceCallerError("invalid_route", "route contains an empty or traversal segment")
    return parsed.path


def filesystem_metadata_locations(value: Any, location: str = "metadata") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            child_location = f"{location}.{key}"
            if lowered in FILESYSTEM_METADATA_KEYS or lowered.endswith(("_path", "_dir", "_directory", "_root", "_uri")):
                found.append(child_location)
            found.extend(filesystem_metadata_locations(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(filesystem_metadata_locations(child, f"{location}[{index}]"))
    return found


def normalize_dtype(value: Any) -> tuple[str, str, int]:
    if not isinstance(value, str) or value not in DTYPES:
        raise ServiceCallerError("invalid_array_dtype", f"unsupported dtype {value!r}; use a fixed-width numeric dtype")
    descr, itemsize = DTYPES[value]
    return DESCR_TO_NAME[descr], descr, itemsize


def normalize_shape(value: Any, *, allow_zero: bool = False) -> tuple[int, ...]:
    qualifier = "non-negative" if allow_zero else "positive"
    if not isinstance(value, (list, tuple)) or not value:
        raise ServiceCallerError("invalid_array_shape", f"shape must be a non-empty array of {qualifier} integers")
    minimum = 0 if allow_zero else 1
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < minimum for dim in value):
        raise ServiceCallerError("invalid_array_shape", f"shape dimensions must be {qualifier} integers")
    shape = tuple(value)
    if len(shape) > 16:
        raise ServiceCallerError("invalid_array_shape", "shape has more than 16 dimensions")
    return shape


def expected_array_bytes(shape: tuple[int, ...], itemsize: int) -> int:
    count = math.prod(shape)
    if count > MAX_ARRAY_BYTES // itemsize:
        raise ServiceCallerError("array_too_large", f"array requires {count * itemsize} bytes; limit is {MAX_ARRAY_BYTES}")
    return count * itemsize


def read_npy(path: Path) -> tuple[bytes, tuple[int, ...], str, str]:
    try:
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise ServiceCallerError("invalid_npy", f"not an NPY file: {path}")
            major, minor = struct.unpack("BB", stream.read(2))
            if (major, minor) == (1, 0):
                header_len = struct.unpack("<H", stream.read(2))[0]
            elif major in {2, 3}:
                header_len = struct.unpack("<I", stream.read(4))[0]
            else:
                raise ServiceCallerError("invalid_npy", f"unsupported NPY version {major}.{minor}")
            header = ast.literal_eval(stream.read(header_len).decode("latin1"))
            if not isinstance(header, dict) or header.get("fortran_order") is not False:
                raise ServiceCallerError("invalid_npy", "only C-contiguous numeric NPY arrays are supported")
            shape = normalize_shape(header.get("shape"), allow_zero=True)
            dtype_name, descr, itemsize = normalize_dtype(header.get("descr"))
            expected = expected_array_bytes(shape, itemsize)
            data = stream.read(MAX_ARRAY_BYTES + 1)
    except ServiceCallerError:
        raise
    except (OSError, EOFError, SyntaxError, ValueError, struct.error) as exc:
        raise ServiceCallerError("invalid_npy", f"could not read {path}: {exc}") from exc
    if len(data) != expected:
        raise ServiceCallerError("array_size_mismatch", f"{path} has {len(data)} data bytes; shape/dtype require {expected}")
    return data, shape, dtype_name, descr


def read_array_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ServiceCallerError("invalid_array_spec", "every arrays entry must be an object")
    unknown = sorted(set(spec) - {"name", "path", "shape", "dtype"})
    if unknown:
        raise ServiceCallerError("unknown_array_fields", f"unknown array fields: {', '.join(unknown)}")
    name = spec.get("name")
    if not isinstance(name, str) or not ARRAY_NAME_RE.fullmatch(name):
        raise ServiceCallerError("invalid_array_name", f"invalid multipart array field name: {name!r}")
    raw_path = spec.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ServiceCallerError("invalid_array_path", f"array {name} requires a local path")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ServiceCallerError("array_path_not_found", f"array {name} path is not a file: {path}")
    if path.suffix.lower() == ".npy":
        if "shape" in spec or "dtype" in spec:
            raise ServiceCallerError("ambiguous_npy_spec", f"array {name}: omit shape/dtype for .npy input")
        data, shape, dtype_name, descr = read_npy(path)
    else:
        if "shape" not in spec or "dtype" not in spec:
            raise ServiceCallerError("raw_array_schema_required", f"array {name}: raw input requires shape and dtype")
        shape = normalize_shape(spec["shape"])
        dtype_name, descr, itemsize = normalize_dtype(spec["dtype"])
        expected = expected_array_bytes(shape, itemsize)
        try:
            size = path.stat().st_size
            if size != expected:
                raise ServiceCallerError("array_size_mismatch", f"{path} has {size} bytes; shape/dtype require {expected}")
            data = path.read_bytes()
        except ServiceCallerError:
            raise
        except OSError as exc:
            raise ServiceCallerError("array_read_failed", f"could not read {path}: {exc}") from exc
    return {"name": name, "path": str(path), "data": data, "shape": shape, "dtype": dtype_name, "descr": descr}


def validate_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(payload) - {"base_url", "route", "metadata", "arrays", "timeout_s", "output_dir"})
    if unknown:
        raise ServiceCallerError("unknown_request_fields", f"unknown request fields: {', '.join(unknown)}")
    missing = sorted({"base_url", "route", "metadata", "arrays", "timeout_s", "output_dir"} - set(payload))
    if missing:
        raise ServiceCallerError("missing_request_fields", f"missing request fields: {', '.join(missing)}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ServiceCallerError("invalid_metadata", "metadata must be a JSON object")
    filesystem_fields = filesystem_metadata_locations(metadata)
    if filesystem_fields:
        raise ServiceCallerError(
            "filesystem_metadata_forbidden",
            "deployed services accept bytes plus provenance, not service-side filesystem references: " + ", ".join(filesystem_fields),
        )
    arrays = payload.get("arrays")
    if not isinstance(arrays, list):
        raise ServiceCallerError("invalid_arrays", "arrays must be a JSON array")
    timeout = payload.get("timeout_s")
    try:
        valid_timeout = (
            not isinstance(timeout, bool)
            and isinstance(timeout, (int, float))
            and math.isfinite(float(timeout))
            and float(timeout) > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ServiceCallerError("invalid_timeout", "timeout_s must be a finite positive number")
    output_dir = payload.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ServiceCallerError("invalid_output_dir", "output_dir must be a non-empty local path")
    resolved_arrays = [read_array_spec(spec) for spec in arrays]
    names = [row["name"] for row in resolved_arrays]
    if len(names) != len(set(names)):
        raise ServiceCallerError("duplicate_array_name", "multipart array field names must be unique")
    total = sum(len(row["data"]) for row in resolved_arrays)
    if total > MAX_TOTAL_INPUT_BYTES:
        raise ServiceCallerError("request_too_large", f"array payload totals {total} bytes; limit is {MAX_TOTAL_INPUT_BYTES}")
    return {
        "base_url": validate_base_url(payload["base_url"]),
        "route": validate_route(payload["route"]),
        "metadata": metadata,
        "arrays": resolved_arrays,
        "timeout_s": float(timeout),
        "output_dir": Path(output_dir).expanduser().resolve(),
    }


def quote_disposition(value: str) -> str:
    if any(char in value for char in {'"', "\r", "\n", ";"}):
        raise ServiceCallerError("invalid_multipart_parameter", f"unsafe multipart parameter: {value!r}")
    return f'"{value}"'


def build_multipart_body(metadata: dict[str, Any], arrays: list[dict[str, Any]], *, boundary: str | None = None) -> tuple[bytes, str]:
    token = boundary or f"ego-ray-{secrets.token_hex(16)}"
    if not ARRAY_NAME_RE.fullmatch(token):
        raise ServiceCallerError("invalid_multipart_boundary", "multipart boundary contains unsupported characters")
    chunks: list[bytes] = []

    def append_part(headers: list[str], data: bytes) -> None:
        chunks.append(f"--{token}\r\n".encode("ascii"))
        chunks.append(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
        chunks.append(data)
        chunks.append(b"\r\n")

    try:
        metadata_bytes = json.dumps(
            metadata,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ServiceCallerError("invalid_metadata_json", f"metadata is not finite JSON data: {exc}") from exc
    append_part(
        ['Content-Disposition: form-data; name="metadata"', "Content-Type: application/json; charset=utf-8"],
        metadata_bytes,
    )
    for row in arrays:
        shape = ",".join(str(dim) for dim in row["shape"])
        disposition = (
            "Content-Disposition: form-data; "
            f"name={quote_disposition(row['name'])}; "
            f"shape={quote_disposition(shape)}; "
            f"dtype={quote_disposition(row['dtype'])}"
        )
        append_part([disposition, "Content-Type: application/octet-stream"], row["data"])
    chunks.append(f"--{token}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={token}"


def parse_shape_parameter(value: str | None) -> tuple[int, ...]:
    if value is None:
        raise ServiceCallerError("response_array_shape_missing", "binary response part has no shape parameter")
    text = value.strip()
    try:
        if text.startswith("[") or text.startswith("("):
            parsed = ast.literal_eval(text)
            return normalize_shape(parsed, allow_zero=True)
        separator = "x" if "x" in text.lower() and "," not in text else ","
        return normalize_shape(
            [int(part.strip()) for part in re.split(separator, text, flags=re.IGNORECASE) if part.strip()],
            allow_zero=True,
        )
    except (SyntaxError, ValueError) as exc:
        raise ServiceCallerError("invalid_response_array_shape", f"invalid response shape {value!r}") from exc


def npy_bytes(data: bytes, shape: tuple[int, ...], descr: str) -> bytes:
    shape_repr = repr(shape)
    header = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': {shape_repr}, }}".encode("latin1")
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    header += b" " * padding + b"\n"
    if len(header) > 65535:
        raise ServiceCallerError("response_array_header_too_large", "NPY header exceeds version-1 limit")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data


def parse_json_bytes(data: bytes, *, code: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceCallerError(code, str(exc)) from exc


def decode_service_response(status: int, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    """Decode one service response without writing array payloads to disk."""
    content_type = headers.get("content-type", headers.get("Content-Type", "")).strip()
    report: dict[str, Any] = {
        "schema": "ego.annotation.feishu_ray_direct_call_report.v1",
        "status": "ok" if 200 <= status < 300 else "http_error",
        "http_status": status,
        "content_type": content_type,
        "metadata": None,
        "arrays": [],
    }
    lowered = content_type.lower()
    if lowered.startswith("application/json") or (not lowered and body.lstrip().startswith((b"{", b"["))):
        report["metadata"] = parse_json_bytes(body, code="invalid_json_response")
        return report
    if not lowered.startswith(("multipart/form-data", "multipart/mixed")):
        raise ServiceCallerError("unsupported_response_content_type", content_type or "missing Content-Type")
    message = BytesParser(policy=email_policy).parsebytes(
        f"MIME-Version: 1.0\r\nContent-Type: {content_type}\r\n\r\n".encode("latin1") + body
    )
    if not message.is_multipart():
        raise ServiceCallerError("invalid_multipart_response", "response Content-Type declared multipart but no parts were parsed")
    seen: set[str] = set()
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not ARRAY_NAME_RE.fullmatch(name):
            raise ServiceCallerError("invalid_response_part_name", f"invalid or missing response part name: {name!r}")
        if name in seen:
            raise ServiceCallerError("duplicate_response_part", f"duplicate response part: {name}")
        seen.add(name)
        data = part.get_payload(decode=True) or b""
        part_type = part.get_content_type().lower()
        if name == "metadata" or part_type == "application/json":
            parsed = parse_json_bytes(data, code="invalid_response_metadata")
            if name == "metadata":
                report["metadata"] = parsed
            else:
                report.setdefault("json_parts", {})[name] = parsed
            continue
        shape = parse_shape_parameter(part.get_param("shape", header="content-disposition"))
        dtype_name, descr, itemsize = normalize_dtype(part.get_param("dtype", header="content-disposition"))
        expected = expected_array_bytes(shape, itemsize)
        if len(data) != expected:
            raise ServiceCallerError("response_array_size_mismatch", f"part {name} has {len(data)} bytes; shape/dtype require {expected}")
        report["arrays"].append(
            {
                "name": name,
                "data": data,
                "shape": shape,
                "dtype": dtype_name,
                "descr": descr,
                "size_bytes": len(data),
            }
        )
    if report["metadata"] is None:
        raise ServiceCallerError("response_metadata_missing", "multipart response has no metadata JSON part")
    return report


def parse_service_response(status: int, headers: dict[str, str], body: bytes, output_dir: Path) -> dict[str, Any]:
    """Decode a service response and materialize binary parts as NPY files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded = decode_service_response(status, headers, body)
    report = {key: value for key, value in decoded.items() if key != "arrays"}
    report["output_dir"] = str(output_dir)
    report["arrays"] = []
    for row in decoded["arrays"]:
        path = output_dir / f"{row['name']}.npy"
        path.write_bytes(npy_bytes(row["data"], row["shape"], row["descr"]))
        report["arrays"].append(
            {
                "name": row["name"],
                "path": str(path),
                "shape": list(row["shape"]),
                "dtype": row["dtype"],
                "size_bytes": row["size_bytes"],
            }
        )
    return report


def normalize_memory_arrays(arrays: dict[str, tuple[bytes, tuple[int, ...], str]]) -> list[dict[str, Any]]:
    """Validate named in-memory arrays for a model-native request."""
    rows: list[dict[str, Any]] = []
    for name, raw in arrays.items():
        if not isinstance(name, str) or not ARRAY_NAME_RE.fullmatch(name):
            raise ServiceCallerError("invalid_array_name", f"invalid multipart array field name: {name!r}")
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise ServiceCallerError("invalid_array_spec", f"array {name} must be (bytes, shape, dtype)")
        data, raw_shape, raw_dtype = raw
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ServiceCallerError("invalid_array_data", f"array {name} data must be bytes-like")
        shape = normalize_shape(raw_shape, allow_zero=True)
        dtype_name, descr, itemsize = normalize_dtype(raw_dtype)
        expected = expected_array_bytes(shape, itemsize)
        data_bytes = bytes(data)
        if len(data_bytes) != expected:
            raise ServiceCallerError(
                "array_size_mismatch",
                f"array {name} has {len(data_bytes)} bytes; shape/dtype require {expected}",
            )
        rows.append(
            {
                "name": name,
                "data": data_bytes,
                "shape": shape,
                "dtype": dtype_name,
                "descr": descr,
            }
        )
    return rows


def incomplete_response_error(
    error: IncompleteRead,
    *,
    url: str,
    status: int | None,
    headers: dict[str, str] | None,
) -> ServiceCallerError:
    partial = error.partial
    partial_bytes = bytes(partial) if isinstance(partial, (bytes, bytearray, memoryview)) else None
    return ServiceCallerError(
        "service_response_incomplete",
        f"{url}: {error}",
        response_received=True,
        response_status=status,
        response_headers=headers,
        raw_response_bytes=partial_bytes,
    )


def read_service_response(
    http_request: Request,
    *,
    timeout_s: float,
    opener: Callable[..., Any],
    url: str,
) -> tuple[int, dict[str, str], bytes]:
    """Read one HTTP response while retaining the exact receipt boundary."""
    response_received = False
    status: int | None = None
    response_headers: dict[str, str] | None = None
    response_body: bytes | None = None
    try:
        response_context = opener(http_request, timeout=timeout_s)
    except HTTPError as exc:
        response_received = True
        status = int(exc.code)
        response_headers = dict(exc.headers.items()) if exc.headers is not None else {}
        try:
            response_body = exc.read()
        except IncompleteRead as read_exc:
            raise incomplete_response_error(
                read_exc,
                url=url,
                status=status,
                headers=response_headers,
            ) from read_exc
        except (URLError, TimeoutError, OSError) as read_exc:
            raise ServiceCallerError(
                "service_transport_failed",
                f"{url}: {read_exc}",
                response_received=response_received,
                response_status=status,
                response_headers=response_headers,
            ) from read_exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ServiceCallerError(
            "service_transport_failed",
            f"{url}: {exc}",
            response_received=response_received,
        ) from exc
    else:
        response_received = True
        try:
            with response_context as response:
                status = int(response.status)
                response_headers = dict(response.headers.items())
                response_body = response.read()
        except IncompleteRead as exc:
            raise incomplete_response_error(
                exc,
                url=url,
                status=status,
                headers=response_headers,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ServiceCallerError(
                "service_transport_failed",
                f"{url}: {exc}",
                response_received=response_received,
                response_status=status,
                response_headers=response_headers,
                raw_response_bytes=response_body,
            ) from exc
    if status is None or response_headers is None or response_body is None:
        raise AssertionError("HTTP response evidence was not initialized")
    return status, response_headers, response_body


def received_response_error(
    error: ServiceCallerError,
    *,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> ServiceCallerError:
    """Bind a decoder/parser error to the response bytes that caused it."""
    return ServiceCallerError(
        error.code,
        str(error),
        response_received=True,
        response_status=status,
        response_headers=headers,
        raw_response_bytes=body,
    )


def prepare_memory_service_request(
    *,
    base_url: str,
    route: str,
    metadata: dict[str, Any],
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]],
    timeout_s: float,
    boundary: str | None,
) -> tuple[Request, str, list[dict[str, Any]]]:
    if not isinstance(metadata, dict):
        raise ServiceCallerError("invalid_metadata", "metadata must be a JSON object")
    if not isinstance(arrays, dict):
        raise ServiceCallerError("invalid_arrays", "arrays must be an object of in-memory array specifications")
    filesystem_fields = filesystem_metadata_locations(metadata)
    if filesystem_fields:
        raise ServiceCallerError(
            "filesystem_metadata_forbidden",
            "deployed services accept bytes plus provenance, not service-side filesystem references: "
            + ", ".join(filesystem_fields),
        )
    try:
        valid_timeout = (
            not isinstance(timeout_s, bool)
            and isinstance(timeout_s, (int, float))
            and math.isfinite(float(timeout_s))
            and float(timeout_s) > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ServiceCallerError("invalid_timeout", "timeout_s must be a finite positive number")
    request_arrays = normalize_memory_arrays(arrays)
    total = sum(len(row["data"]) for row in request_arrays)
    if total > MAX_TOTAL_INPUT_BYTES:
        raise ServiceCallerError("request_too_large", f"array payload totals {total} bytes; limit is {MAX_TOTAL_INPUT_BYTES}")
    body, content_type = build_multipart_body(metadata, request_arrays, boundary=boundary)
    url = validate_base_url(base_url) + validate_route(route)
    http_request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "multipart/form-data, application/json"},
    )
    return http_request, url, request_arrays


def call_service_arrays(
    *,
    base_url: str,
    route: str,
    metadata: dict[str, Any],
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]],
    timeout_s: float,
    opener: Callable[..., Any] = urlopen,
    boundary: str | None = None,
) -> dict[str, Any]:
    """Call one route with in-memory arrays and return decoded response bytes."""
    try:
        http_request, url, request_arrays = prepare_memory_service_request(
            base_url=base_url,
            route=route,
            metadata=metadata,
            arrays=arrays,
            timeout_s=timeout_s,
            boundary=boundary,
        )
    except ServiceCallerError as exc:
        exc.response_received = False
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceCallerError(
            "service_request_construction_failed",
            str(exc),
            response_received=False,
        ) from exc
    except Exception as exc:
        raise ServiceCallerError(
            "service_request_preflight_failed",
            str(exc),
            response_received=False,
        ) from exc
    status, response_headers, response_body = read_service_response(
        http_request,
        timeout_s=float(timeout_s),
        opener=opener,
        url=url,
    )
    try:
        report = decode_service_response(status, response_headers, response_body)
    except ServiceCallerError as exc:
        raise received_response_error(
            exc,
            status=status,
            headers=response_headers,
            body=response_body,
        ) from exc
    report["response_headers"] = dict(response_headers)
    report["request_url"] = url
    report["request_arrays"] = [
        {
            "name": row["name"],
            "shape": list(row["shape"]),
            "dtype": row["dtype"],
            "size_bytes": len(row["data"]),
        }
        for row in request_arrays
    ]
    return report


def prepare_disk_service_request(
    payload: dict[str, Any],
    *,
    boundary: str | None,
) -> tuple[dict[str, Any], Request, str]:
    request = validate_request_payload(payload)
    body, content_type = build_multipart_body(request["metadata"], request["arrays"], boundary=boundary)
    url = request["base_url"] + request["route"]
    http_request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "Accept": "multipart/form-data, application/json"},
    )
    return request, http_request, url


def call_service(
    payload: dict[str, Any],
    *,
    opener: Callable[..., Any] = urlopen,
    boundary: str | None = None,
) -> dict[str, Any]:
    try:
        request, http_request, url = prepare_disk_service_request(payload, boundary=boundary)
    except ServiceCallerError as exc:
        exc.response_received = False
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceCallerError(
            "service_request_construction_failed",
            str(exc),
            response_received=False,
        ) from exc
    except Exception as exc:
        raise ServiceCallerError(
            "service_request_preflight_failed",
            str(exc),
            response_received=False,
        ) from exc
    status, headers, response_body = read_service_response(
        http_request,
        timeout_s=request["timeout_s"],
        opener=opener,
        url=url,
    )
    try:
        report = parse_service_response(status, headers, response_body, request["output_dir"])
    except ServiceCallerError as exc:
        raise received_response_error(
            exc,
            status=status,
            headers=headers,
            body=response_body,
        ) from exc
    report.update(
        {
            "response_headers": dict(headers),
            "request_url": url,
            "request_arrays": [
                {"name": row["name"], "path": row["path"], "shape": list(row["shape"]), "dtype": row["dtype"], "size_bytes": len(row["data"])}
                for row in request["arrays"]
            ],
        }
    )
    report_path = request["output_dir"] / "response_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = call_service(load_json_object(args.request_json))
    except ServiceCallerError as exc:
        print(json.dumps({"status": "failed", "code": exc.code, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
