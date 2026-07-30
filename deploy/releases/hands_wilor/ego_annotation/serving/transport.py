"""Multipart binary HTTP transport and lazy Ray ObjectRef resolution.

This module is the primary transport for the UniDepth endpoint: dense arrays travel
as binary fields inside ``multipart/form-data`` rather than JSON/base64. It depends
only on the Python standard library (``email``, ``uuid``, ``json``) so it is fully
unit-testable without Ray, httpx, or a live server.

Two responsibilities:

1. Build and parse ``multipart/form-data`` bodies carrying a JSON metadata part plus
   one or more binary array parts (``rgb``, ``depth_m``, ``K_px``, ``confidence``).
   Each binary part declares its tensor ``shape`` and ``dtype`` in ``Content-Disposition``
   parameters so the receiver can reconstruct the array without a separate schema.
2. Resolve nested Ray ``ObjectRef`` values lazily. The in-cluster resolver calls an
   injected ``ray_get`` (default: ``ray.get``) repeatedly until the value is no longer
   reference-like, so a caller may pass an ``ObjectRef`` pointing at another
   ``ObjectRef`` pointing at the actual bytes.
"""
from __future__ import annotations

import json
import uuid
from email.parser import BytesParser
from email.policy import default as default_email_policy
from typing import Any, Callable, Iterable, Mapping, Sequence


# A value is treated as a Ray ObjectRef if it looks like one without importing Ray.
# Real ``ray.ObjectRef`` is a Cython type with a ``binary()`` method and type name
# ``ObjectRef``. This duck-type covers both; tests inject their own fake refs that
# satisfy the same shape (see ``FakeObjectRef`` in the tests).
def _is_object_ref(value: Any) -> bool:
    if type(value).__name__ == "ObjectRef":
        return True
    return hasattr(value, "binary") and callable(getattr(value, "binary")) and hasattr(value, "is_nil")


def lazy_resolve_object_ref(value: Any, ray_get: Callable[[Any], Any]) -> Any:
    """Resolve a (possibly nested) Ray ObjectRef to its underlying value.

    ``ray_get`` is injected so tests can simulate nested refs without Ray installed.
    In-cluster, the default is ``ray.get``. Resolution repeats until the result is no
    longer reference-like, handling ``ObjectRef -> ObjectRef -> bytes`` chains.
    """
    current = value
    hops = 0
    while _is_object_ref(current):
        if hops > 64:  # defense against a pathological reference cycle
            raise RuntimeError("object reference resolution exceeded 64 hops; possible reference cycle")
        current = ray_get(current)
        hops += 1
    return current


def _boundary() -> str:
    return f"egounidepth-{uuid.uuid4().hex}"


def _part_headers(name: str, *, shape: Sequence[int] | None = None, dtype: str | None = None,
                   content_type: str = "application/octet-stream",
                   extra: Mapping[str, Any] | None = None) -> str:
    params = f'name="{name}"'
    if shape is not None:
        params += f'; shape="{_shape_to_str(shape)}"'
    if dtype is not None:
        params += f'; dtype="{dtype}"'
    # Extra Content-Disposition params (e.g. kind/media_type/source_index for Cosmos3).
    for key, value in (extra or {}).items():
        params += f'; {key}="{value}"'
    return f"Content-Disposition: form-data; {params}\r\nContent-Type: {content_type}\r\n\r\n"


def _shape_to_str(shape: Sequence[int]) -> str:
    return ",".join(str(int(dim)) for dim in shape)


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip() != "")


def build_multipart_request(metadata: Mapping[str, Any], *, rgb: bytes, rgb_shape: Sequence[int],
                             rgb_dtype: str) -> tuple[bytes, str]:
    """Build a multipart/form-data UniDepth request body.

    Returns ``(body, content_type_header)``. The body has a ``metadata`` JSON part
    and an ``rgb`` binary part. The receiver reconstructs the request via
    ``parse_multipart_request``.
    """
    boundary = _boundary()
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    body = _assemble_multipart(
        boundary,
        [
            ("metadata", meta_bytes, {"content_type": "application/json"}),
            ("rgb", rgb, {"shape": tuple(rgb_shape), "dtype": rgb_dtype}),
        ],
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _assemble_multipart(boundary: str,
                        parts: Sequence[tuple[str, bytes, Mapping[str, Any]]]) -> bytes:
    out = bytearray()
    bdelim = b"--" + boundary.encode("ascii")
    for name, data, params in parts:
        out += bdelim + b"\r\n"
        # Distinguish known tensor params (shape/dtype) from extra Content-Disposition params.
        shape = params.get("shape")
        dtype = params.get("dtype")
        extra = {k: v for k, v in params.items() if k not in {"shape", "dtype", "content_type"}}
        out += _part_headers(
            name,
            shape=tuple(shape) if shape is not None else None,
            dtype=dtype,
            content_type=params.get("content_type", "application/octet-stream"),
            extra=extra or None,
        ).encode("utf-8")
        out += data
        out += b"\r\n"
    out += bdelim + b"--\r\n"
    return bytes(out)


def _iter_multipart(body: bytes, content_type: str) -> list[tuple[str, bytes, dict[str, str]]]:
    """Parse a multipart body into (name, data, params) tuples using stdlib email."""
    # The email parser needs a full message with headers; synthesize one.
    header_blob = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default_email_policy).parsebytes(header_blob + body)
    results: list[tuple[str, bytes, dict[str, str]]] = []
    if not message.is_multipart():
        return results
    for part in message.get_payload():
        disposition = part.get("Content-Disposition", "")
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        params: dict[str, str] = {}
        shape = part.get_param("shape", header="content-disposition")
        dtype = part.get_param("dtype", header="content-disposition")
        if shape is not None:
            params["shape"] = str(shape)
        if dtype is not None:
            params["dtype"] = str(dtype)
        # Cosmos3 media parts declare kind/media_type/source_index in Content-Disposition.
        for extra in ("kind", "media_type", "source_index"):
            value = part.get_param(extra, header="content-disposition")
            if value is not None:
                params[extra] = str(value)
        # get_payload(decode=True) returns the raw bytes for the part.
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = part.get_payload()
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
        results.append((str(name), bytes(payload), params))
    return results


def parse_multipart_request(body: bytes, content_type: str) -> tuple[dict[str, Any], bytes, tuple[int, ...], str]:
    """Parse a multipart UniDepth request into (metadata, rgb_bytes, rgb_shape, rgb_dtype)."""
    parts = {name: (data, params) for name, data, params in _iter_multipart(body, content_type)}
    if "metadata" not in parts:
        raise ValueError("multipart request missing 'metadata' part")
    if "rgb" not in parts:
        raise ValueError("multipart request missing 'rgb' part")
    metadata = json.loads(parts["metadata"][0].decode("utf-8"))
    rgb_bytes, rgb_params = parts["rgb"]
    if "shape" not in rgb_params or "dtype" not in rgb_params:
        raise ValueError("multipart 'rgb' part must declare shape and dtype")
    return metadata, rgb_bytes, _parse_shape(rgb_params["shape"]), rgb_params["dtype"]


def build_multipart_request_fields(
    metadata: Mapping[str, Any], fields: Mapping[str, tuple[bytes, Sequence[int], str]]
) -> tuple[bytes, str]:
    """Build a multipart/form-data request with one JSON metadata part plus named binary fields.

    ``fields`` maps field name -> ``(bytes, shape, dtype)``. Used by the GPU1 hands/wilor
    APIs where the single binary field is ``rgb`` (hands) or ``crop`` (wilor). The
    UniDepth builder above is a specialization of this for the ``rgb`` field.
    """
    boundary = _boundary()
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    part_specs: list[tuple[str, bytes, Mapping[str, Any]]] = [
        ("metadata", meta_bytes, {"content_type": "application/json"}),
    ]
    for name, (data, shape, dtype) in fields.items():
        part_specs.append((name, data, {"shape": tuple(shape), "dtype": dtype}))
    return _assemble_multipart(boundary, part_specs), f"multipart/form-data; boundary={boundary}"


def parse_multipart_request_fields(
    body: bytes, content_type: str
) -> tuple[dict[str, Any], dict[str, tuple[bytes, tuple[int, ...], str]]]:
    """Parse a multipart request into (metadata, {field: (bytes, shape, dtype)}).

    Generic counterpart to ``parse_multipart_request``: returns the JSON metadata part
    and every binary part that declares ``shape``/``dtype``. Used by the GPU1 deployments
    to reconstruct hands/wilor requests carrying a ``rgb`` or ``crop`` binary field.
    """
    parts = _iter_multipart(body, content_type)
    metadata: dict[str, Any] = {}
    fields: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
    for name, data, params in parts:
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif "shape" in params and "dtype" in params:
            fields[name] = (data, _parse_shape(params["shape"]), params["dtype"])
    if not metadata:
        raise ValueError("multipart request missing 'metadata' part")
    return metadata, fields


def build_multipart_response(metadata: Mapping[str, Any], arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]]) -> tuple[bytes, str]:
    """Build a model-service multipart response body.

    ``arrays`` maps field name to ``(bytes, shape, dtype)``. Metadata-only responses
    use an empty mapping. Returns ``(body, content_type_header)``.
    """
    boundary = _boundary()
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    part_specs: list[tuple[str, bytes, Mapping[str, Any]]] = [
        ("metadata", meta_bytes, {"content_type": "application/json"}),
    ]
    for name, (data, shape, dtype) in arrays.items():
        part_specs.append((name, data, {"shape": shape, "dtype": dtype}))
    return _assemble_multipart(boundary, part_specs), f"multipart/form-data; boundary={boundary}"


def multipart_asgi_response(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]],
    *,
    status_code: int = 200,
) -> Any:
    """Return the canonical binary-safe ASGI response for every model endpoint.

    Ray Serve JSON-encodes ordinary return values, which corrupts arbitrary tensor
    bytes. A Starlette ``Response`` bypasses that encoder. The import stays lazy so
    contract/adapter code can import the transport module without an ASGI dependency.
    """
    from starlette.responses import Response

    body, content_type = build_multipart_response(metadata, arrays)
    return Response(content=body, media_type=content_type, status_code=status_code)


def parse_multipart_response(body: bytes, content_type: str) -> tuple[dict[str, Any], dict[str, tuple[bytes, tuple[int, ...], str]]]:
    """Parse a multipart response into metadata and declared binary tensor parts."""
    parts = _iter_multipart(body, content_type)
    metadata: dict[str, Any] = {}
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
    seen_names: set[str] = set()
    for name, data, params in parts:
        if name in seen_names:
            raise ValueError(f"multipart response contains duplicate part {name!r}")
        seen_names.add(name)
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif "shape" in params and "dtype" in params:
            arrays[name] = (data, _parse_shape(params["shape"]), params["dtype"])
        else:
            raise ValueError(f"multipart response part {name!r} lacks tensor shape/dtype")
    if not metadata:
        raise ValueError("multipart response missing 'metadata' part")
    return metadata, arrays


def parse_droid_finalize_response(body: bytes, content_type: str) -> Any:
    """Parse and semantically validate either typed DROID finalize response framing.

    Both multipart and binary-envelope success responses must contain exactly the
    four real CameraState arrays. ``DroidFinalizeResponse.from_wire`` then preserves
    descriptor, shape, finite-value, and mutual-inverse validation identically.
    """
    from ego_annotation.serving.binary_envelope import content_type_is_binary_envelope
    from ego_annotation.serving.contracts import DroidFinalizeResponse
    from ego_annotation.serving.gateway import _parse_generic_envelope

    if content_type_is_binary_envelope(content_type):
        metadata, envelope_arrays = _parse_generic_envelope(body)
        # Typed CameraState parsing owns binary payloads after HTTP ingress; make
        # the envelope's read-only vectors ordinary bytes to share the established
        # multipart validator unchanged.
        arrays = {name: (bytes(data), shape, dtype) for name, (data, shape, dtype) in envelope_arrays.items()}
    elif "multipart/form-data" in content_type.lower():
        metadata, arrays = parse_multipart_response(body, content_type)
    else:
        raise ValueError("successful DROID finalize response must be multipart/form-data or binary-envelope")
    return DroidFinalizeResponse.from_wire(metadata, arrays)


# --- Cosmos3 transport: multipart media-in, JSON metadata-out -----------------------
#
# The Cosmos3 endpoint carries bounded binary image/video media as multipart parts
# (``media_0``, ``media_1``, ...) alongside a JSON ``metadata`` part. The response is
# scalar (text/token/timing/provenance), so it is returned as a JSON metadata part
# with no binary arrays.


def build_cosmos3_request(metadata: Mapping[str, Any], media: Sequence[tuple[bytes, str, str, int]]) -> tuple[bytes, str]:
    """Build a multipart/form-data Cosmos3 request body.

    ``media`` is a sequence of ``(data_bytes, kind, media_type, source_index)`` tuples.
    Returns ``(body, content_type_header)``. The body has a ``metadata`` JSON part plus
    one ``media_<i>`` binary part per item declaring ``kind``, ``media_type`` and
    ``source_index`` in Content-Disposition parameters.
    """
    boundary = f"egocosmos3-{uuid.uuid4().hex}"
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    parts: list[tuple[str, bytes, Mapping[str, Any]]] = [
        ("metadata", meta_bytes, {"content_type": "application/json"}),
    ]
    for index, (data, kind, media_type, source_index) in enumerate(media):
        parts.append((
            f"media_{index}",
            data,
            {"kind": kind, "media_type": media_type, "source_index": str(source_index)},
        ))
    return _assemble_multipart(boundary, parts), f"multipart/form-data; boundary={boundary}"


def parse_cosmos3_request(body: bytes, content_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a multipart Cosmos3 request into (metadata, [media_item_dict, ...]).

    Each media item dict carries ``data`` (bytes), ``kind``, ``media_type`` and
    ``source_index``. The deployment reconstructs ``Cosmos3MediaItem`` from these.
    """
    parts = _iter_multipart(body, content_type)
    metadata: dict[str, Any] = {}
    media_items: list[dict[str, Any]] = []
    for name, data, params in parts:
        if name == "metadata":
            metadata = json.loads(data.decode("utf-8"))
        elif name.startswith("media_"):
            if "kind" not in params or "media_type" not in params:
                raise ValueError(f"multipart part {name!r} must declare kind and media_type")
            media_items.append({
                "data": data,
                "kind": params["kind"],
                "media_type": params["media_type"],
                "source_index": int(params.get("source_index", "0")),
            })
    if not metadata:
        raise ValueError("multipart Cosmos3 request missing 'metadata' part")
    return metadata, media_items


def build_cosmos3_response(metadata: Mapping[str, Any]) -> tuple[bytes, str]:
    """Build a multipart Cosmos3 response body (metadata-only, no binary arrays).

    The response is scalar (text/token/timing/provenance), so it is a single JSON
    ``metadata`` part. Multipart framing keeps the transport uniform with the request
    and lets the client reuse ``parse_cosmos3_response``.
    """
    boundary = f"egocosmos3-{uuid.uuid4().hex}"
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return _assemble_multipart(boundary, [("metadata", meta_bytes, {"content_type": "application/json"})]), \
        f"multipart/form-data; boundary={boundary}"


def parse_cosmos3_response(body: bytes, content_type: str) -> dict[str, Any]:
    """Parse a multipart Cosmos3 response into its metadata dict."""
    parts = _iter_multipart(body, content_type)
    for name, data, _params in parts:
        if name == "metadata":
            return json.loads(data.decode("utf-8"))
    # Some Serve proxies may return the JSON metadata directly without multipart framing.
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Cosmos3 response missing 'metadata' part: {exc}") from exc
