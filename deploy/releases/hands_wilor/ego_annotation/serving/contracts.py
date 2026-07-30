"""Path-free, model-native contracts shared by Ray Serve deployments and clients.

These contracts are deliberately Python 3.10-compatible: model-facing serving
environments (GPU1) run on Python 3.10, so this module avoids ``datetime.UTC``
(3.11+) and other 3.11-only runtime syntax. The project-wide ``requires-python``
constraint in ``pyproject.toml`` is intentionally unchanged.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, cast


SCHEMA_VERSION = "ego.model-service.v1"
_PROHIBITED_PATH_TOKENS = ("path", "directory", "output_dir", "input_video", "run_root", "artifact_root")


class ContractValidationError(ValueError):
    """A request or response violates the public model-serving contract."""


def reject_filesystem_fields(value: Any, *, context: str = "payload") -> None:
    """Reject filesystem-shaped caller fields recursively before service admission."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in _PROHIBITED_PATH_TOKENS):
                raise ContractValidationError(f"{context}.{key} is not allowed: model APIs do not accept filesystem paths")
            reject_filesystem_fields(child, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_filesystem_fields(child, context=f"{context}[{index}]")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{name} must be a non-negative integer")
    return value


def _matrix3(value: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ContractValidationError(f"{name} must be a 3x3 matrix")
    try:
        rows = tuple(tuple(float(cell) for cell in row) for row in value)
        return cast(tuple[tuple[float, float, float], ...], rows)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{name} must contain numeric values") from exc


def _timestamp(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    text = _required_text(value, name)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(f"{name} must be ISO-8601") from exc
    return text


def utc_now() -> str:
    # timezone.utc exists in Python 3.10; datetime.UTC does not (3.11+ only).
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Ownership:
    request_id: str
    job_id: str
    item_id: str
    stage_id: str
    source_id: str
    schema_version: str = SCHEMA_VERSION
    source_timestamp_s: float | None = None
    submitted_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("request_id", "job_id", "item_id", "stage_id", "source_id", "schema_version"):
            _required_text(getattr(self, name), name)
        if self.source_timestamp_s is not None:
            try:
                float(self.source_timestamp_s)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError("source_timestamp_s must be numeric") from exc
        _timestamp(self.submitted_at, "submitted_at")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Ownership":
        reject_filesystem_fields(payload, context="ownership")
        return cls(
            request_id=_required_text(payload.get("request_id"), "request_id"),
            job_id=_required_text(payload.get("job_id"), "job_id"),
            item_id=_required_text(payload.get("item_id"), "item_id"),
            stage_id=_required_text(payload.get("stage_id"), "stage_id"),
            source_id=_required_text(payload.get("source_id"), "source_id"),
            schema_version=_required_text(payload.get("schema_version", SCHEMA_VERSION), "schema_version"),
            source_timestamp_s=payload.get("source_timestamp_s"),
            submitted_at=payload.get("submitted_at", utc_now()),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "job_id": self.job_id,
            "item_id": self.item_id,
            "stage_id": self.stage_id,
            "source_id": self.source_id,
            "schema_version": self.schema_version,
            "source_timestamp_s": self.source_timestamp_s,
            "submitted_at": self.submitted_at,
        }


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        _positive_int(self.width, "width")
        _positive_int(self.height, "height")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ImageSize":
        return cls(width=_positive_int(payload.get("width"), "width"), height=_positive_int(payload.get("height"), "height"))

    def to_wire(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True)
class PixelTransform:
    """Full source/model pixel mapping, including resize/crop/pad provenance."""

    source_to_model: tuple[tuple[float, float, float], ...]
    model_to_source: tuple[tuple[float, float, float], ...]
    resize_mode: str
    crop_xywh: tuple[float, float, float, float] | None = None
    pad_ltrb: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_to_model", _matrix3(self.source_to_model, "source_to_model"))
        object.__setattr__(self, "model_to_source", _matrix3(self.model_to_source, "model_to_source"))
        _required_text(self.resize_mode, "resize_mode")
        for name in ("crop_xywh", "pad_ltrb"):
            value = getattr(self, name)
            if value is not None and len(value) != 4:
                raise ContractValidationError(f"{name} must contain four values")

    @classmethod
    def identity(cls) -> "PixelTransform":
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        return cls(identity, identity, "identity")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PixelTransform":
        return cls(
            source_to_model=cast(tuple[tuple[float, float, float], ...], payload.get("source_to_model", ())),
            model_to_source=cast(tuple[tuple[float, float, float], ...], payload.get("model_to_source", ())),
            resize_mode=_required_text(payload.get("resize_mode"), "resize_mode"),
            crop_xywh=tuple(payload["crop_xywh"]) if payload.get("crop_xywh") is not None else None,
            pad_ltrb=tuple(payload["pad_ltrb"]) if payload.get("pad_ltrb") is not None else None,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "source_to_model": [list(row) for row in self.source_to_model],
            "model_to_source": [list(row) for row in self.model_to_source],
            "resize_mode": self.resize_mode,
            "crop_xywh": list(self.crop_xywh) if self.crop_xywh is not None else None,
            "pad_ltrb": list(self.pad_ltrb) if self.pad_ltrb is not None else None,
        }


@dataclass(frozen=True)
class SpatialMetadata:
    source_size: ImageSize
    model_size: ImageSize
    color_space: str
    pixel_transform: PixelTransform
    K_px: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.color_space not in {"RGB"}:
            raise ContractValidationError("color_space must be RGB")
        if self.K_px is not None:
            object.__setattr__(self, "K_px", _matrix3(self.K_px, "K_px"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SpatialMetadata":
        return cls(
            source_size=ImageSize.from_mapping(payload.get("source_size", {})),
            model_size=ImageSize.from_mapping(payload.get("model_size", {})),
            color_space=_required_text(payload.get("color_space"), "color_space"),
            pixel_transform=PixelTransform.from_mapping(payload.get("pixel_transform", {})),
            K_px=payload.get("K_px"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "source_size": self.source_size.to_wire(),
            "model_size": self.model_size.to_wire(),
            "color_space": self.color_space,
            "pixel_transform": self.pixel_transform.to_wire(),
            "K_px": [list(row) for row in self.K_px] if self.K_px is not None else None,
        }


@dataclass(frozen=True)
class TensorPayload:
    """A dense tensor carried as binary bytes over HTTP or an in-cluster object.

    ``data`` deliberately has no path-like representation. An in-cluster caller may
    provide an object-reference-like value; the adapter resolves it only in the
    injected tensor decoder, so importing this module never imports Ray.

    The HTTP transport carries dense arrays as binary fields inside a multipart
    body (see ``transport.py``); ``to_wire``/``from_wire`` with ``data_b64`` remain
    only for JSON fallback/debugging and for metadata-only round trips.
    """

    data: bytes | bytearray | memoryview | Any
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ContractValidationError("tensor shape must contain non-negative integer dimensions")
        _required_text(self.dtype, "dtype")
        _required_text(self.dtype, "dtype")
        if isinstance(self.data, str):
            raise ContractValidationError("tensor data must be binary or an in-cluster object, never a path string")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "TensorPayload":
        reject_filesystem_fields(payload, context="tensor")
        encoded = payload.get("data_b64")
        if not isinstance(encoded, str):
            raise ContractValidationError("tensor.data_b64 must be a base64 string")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except ValueError as exc:
            raise ContractValidationError("tensor.data_b64 is invalid base64") from exc
        shape = payload.get("shape")
        if not isinstance(shape, list):
            raise ContractValidationError("tensor.shape must be a list")
        return cls(data=data, shape=tuple(shape), dtype=_required_text(payload.get("dtype"), "dtype"))

    def to_wire(self) -> dict[str, Any]:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise ContractValidationError("in-cluster tensor references cannot be serialized over HTTP")
        return {
            "data_b64": base64.b64encode(bytes(self.data)).decode("ascii"),
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


# The initial UniDepth API accepts uint8 HWC RGB only. The real backend transposes
# contiguous [B,H,W,C] to [B,C,H,W] before upstream ``model.infer`` and does NOT
# divide by 255 (UniDepth's own ``infer`` does). Float RGB is range-ambiguous and
# therefore rejected at the contract boundary.
UNIDEPTH_RGB_DTYPE = "uint8"


@dataclass(frozen=True)
class UniDepthRequest:
    ownership: Ownership
    rgb: TensorPayload
    spatial: SpatialMetadata
    model_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.model_revision, "model_revision")
        if self.rgb.dtype != UNIDEPTH_RGB_DTYPE:
            raise ContractValidationError(
                f"UniDepth RGB dtype must be {UNIDEPTH_RGB_DTYPE} only; float RGB is range-ambiguous and rejected"
            )
        expected_shape = (self.spatial.model_size.height, self.spatial.model_size.width, 3)
        if self.rgb.shape != expected_shape:
            raise ContractValidationError(f"UniDepth RGB shape must be HxWx3 matching model_size {expected_shape}")

    @property
    def work_units(self) -> int:
        """Normalized canonical-image work: one request is exactly one work unit.

        The deployment advertises one canonical HxW compatibility bucket, so every
        admitted request carries identical weight and ``serve.batch``'s
        ``batch_size_fn`` returns ``len(items)``.
        """
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (self.rgb.dtype, self.rgb.shape, self.model_revision, self.options)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "UniDepthRequest":
        reject_filesystem_fields(payload)
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            raise ContractValidationError("options must be an object")
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            rgb=TensorPayload.from_wire(payload.get("rgb", {})),
            spatial=SpatialMetadata.from_mapping(payload.get("spatial", {})),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            options=tuple(sorted((str(key), str(value)) for key, value in options.items())),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "rgb": self.rgb.to_wire(),
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


# --- Cosmos3 (GPU6) model-native multimodal reasoning contract ----------------------
#
# ``cosmos3.reason`` accepts model-native prompt/messages plus bounded binary
# image/video media and generation controls. A caller NEVER supplies a server
# filesystem path: media travels as inline binary parts inside the multipart HTTP
# body. The resident config owns the model revision; the request carries none, and
# every result carries only the server-owned revision. vLLM's engine owns
# continuous batching across concurrent requests, so (unlike UniDepth) the Serve
# layer does NOT fuse requests with ``@serve.batch``; each request is one engine
# ``generate`` call and ``work_units`` is informational.

COSMOS3_MEDIA_KINDS = ("image", "video")
COSMOS3_IMAGE_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
COSMOS3_VIDEO_MEDIA_TYPES = ("video/mp4", "video/quicktime", "video/webm")


@dataclass(frozen=True)
class Cosmos3MediaItem:
    """One bounded binary media item carried inline (never a server path).

    ``data`` is raw encoded media bytes (PNG/JPEG/MP4/...). The serving replica
    decodes it to the vLLM-native PIL image / numpy video array at admission so no
    caller-controlled path ever reaches the model.
    """

    kind: str
    data: bytes | bytearray | memoryview | Any
    media_type: str
    source_index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in COSMOS3_MEDIA_KINDS:
            raise ContractValidationError(f"media kind must be one of {COSMOS3_MEDIA_KINDS}, got {self.kind!r}")
        if isinstance(self.data, str):
            raise ContractValidationError("media data must be binary bytes, never a path string")
        if self.kind == "image" and self.media_type not in COSMOS3_IMAGE_MEDIA_TYPES:
            raise ContractValidationError(f"image media_type must be one of {COSMOS3_IMAGE_MEDIA_TYPES}, got {self.media_type!r}")
        if self.kind == "video" and self.media_type not in COSMOS3_VIDEO_MEDIA_TYPES:
            raise ContractValidationError(f"video media_type must be one of {COSMOS3_VIDEO_MEDIA_TYPES}, got {self.media_type!r}")
        if not isinstance(self.source_index, int) or self.source_index < 0:
            raise ContractValidationError("source_index must be a non-negative integer")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Cosmos3MediaItem":
        reject_filesystem_fields(payload, context="media")
        encoded = payload.get("data_b64")
        if not isinstance(encoded, str):
            raise ContractValidationError("media.data_b64 must be a base64 string (binary travels in multipart parts)")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except ValueError as exc:
            raise ContractValidationError("media.data_b64 is invalid base64") from exc
        return cls(
            kind=_required_text(payload.get("kind"), "media.kind"),
            data=data,
            media_type=_required_text(payload.get("media_type"), "media.media_type"),
            source_index=int(payload.get("source_index", 0)),
        )

    def to_wire(self) -> dict[str, Any]:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise ContractValidationError("in-cluster media references cannot be serialized over HTTP")
        return {
            "kind": self.kind,
            "data_b64": base64.b64encode(bytes(self.data)).decode("ascii"),
            "media_type": self.media_type,
            "source_index": self.source_index,
        }


@dataclass(frozen=True)
class GenerationControls:
    """Bounded sampling/generation limits. All fields are clamped at admission."""

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0 or self.max_tokens > 8192:
            raise ContractValidationError("max_tokens must be a positive integer <= 8192")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or self.temperature < 0.0 or self.temperature > 2.0:
            raise ContractValidationError("temperature must be in [0.0, 2.0]")
        if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool) or self.top_p < 0.0 or self.top_p > 1.0:
            raise ContractValidationError("top_p must be in [0.0, 1.0]")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ContractValidationError("top_k must be a non-negative integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ContractValidationError("seed must be an integer or null")
        if not isinstance(self.frequency_penalty, (int, float)) or isinstance(self.frequency_penalty, bool) or self.frequency_penalty < -2.0 or self.frequency_penalty > 2.0:
            raise ContractValidationError("frequency_penalty must be in [-2.0, 2.0]")
        if not isinstance(self.presence_penalty, (int, float)) or isinstance(self.presence_penalty, bool) or self.presence_penalty < -2.0 or self.presence_penalty > 2.0:
            raise ContractValidationError("presence_penalty must be in [-2.0, 2.0]")
        for token in self.stop:
            if not isinstance(token, str):
                raise ContractValidationError("stop tokens must be strings")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "GenerationControls":
        if payload is None:
            return cls()
        reject_filesystem_fields(payload, context="generation")
        stop = payload.get("stop", [])
        if stop is None:
            stop = []
        if not isinstance(stop, (list, tuple)):
            raise ContractValidationError("stop must be a list of strings")
        return cls(
            max_tokens=int(payload.get("max_tokens", 512)),
            temperature=float(payload.get("temperature", 0.0)),
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", 0)),
            seed=payload.get("seed"),
            stop=tuple(str(t) for t in stop),
            frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
            presence_penalty=float(payload.get("presence_penalty", 0.0)),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "stop": list(self.stop),
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


@dataclass(frozen=True)
class Cosmos3Request:
    """Model-native multimodal reasoning request. No caller server paths.

    Either ``prompt`` (a plain string) or ``messages`` (OpenAI-style chat messages
    whose media is referenced by ``source_index`` into ``media``) must be supplied.
    The request carries NO ``model_revision``: the resident config owns it.
    """

    ownership: Ownership
    prompt: str | None = None
    messages: tuple[tuple[str, Any], ...] = ()
    media: tuple[Cosmos3MediaItem, ...] = ()
    generation: GenerationControls = field(default_factory=GenerationControls)

    def __post_init__(self) -> None:
        if self.prompt is None and not self.messages:
            raise ContractValidationError("cosmos3 request must supply prompt or messages")
        if self.prompt is not None and self.messages:
            raise ContractValidationError("cosmos3 request must supply prompt OR messages, not both")
        if self.prompt is not None:
            _required_text(self.prompt, "prompt")
        for role, _content in self.messages:
            if role not in {"system", "user", "assistant"}:
                raise ContractValidationError(f"message role must be system/user/assistant, got {role!r}")

    @property
    def work_units(self) -> int:
        # vLLM batches internally; one request is one engine generate call.
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        # One resident model => all requests compatible. vLLM's engine owns batching.
        return ("cosmos3",)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "Cosmos3Request":
        reject_filesystem_fields(payload)
        messages_raw = payload.get("messages")
        messages: tuple[tuple[str, Any], ...] = ()
        if messages_raw is not None:
            if not isinstance(messages_raw, list):
                raise ContractValidationError("messages must be a list")
            messages = tuple((str(m.get("role")), m.get("content")) for m in messages_raw if isinstance(m, Mapping))
        media_raw = payload.get("media", [])
        if media_raw is None:
            media_raw = []
        if not isinstance(media_raw, list):
            raise ContractValidationError("media must be a list")
        media = tuple(Cosmos3MediaItem.from_mapping(m) for m in media_raw if isinstance(m, Mapping))
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            prompt=payload.get("prompt"),
            messages=messages,
            media=media,
            generation=GenerationControls.from_mapping(payload.get("generation")),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "prompt": self.prompt,
            "messages": [{"role": role, "content": content} for role, content in self.messages],
            "media": [m.to_wire() for m in self.media],
            "generation": self.generation.to_wire(),
        }


class ErrorCode(str, Enum):
    VALIDATION = "validation"
    CONFLICT = "conflict"
    BACKPRESSURE = "backpressure"
    MODEL_FAILURE = "model_failure"
    RESULT_SPLIT_FAILURE = "result_split_failure"
    TRANSPORT = "transport"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ServiceError:
    code: ErrorCode
    message: str
    retryable: bool
    ownership: Ownership | None = None
    batch_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "ownership": self.ownership.to_wire() if self.ownership else None,
            "batch_id": self.batch_id,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ServiceError":
        return cls(
            code=ErrorCode(payload["code"]),
            message=_required_text(payload.get("message"), "error.message"),
            retryable=bool(payload.get("retryable")),
            ownership=Ownership.from_mapping(payload["ownership"]) if payload.get("ownership") else None,
            batch_id=payload.get("batch_id"),
        )


@dataclass(frozen=True)
class BatchTrace:
    """Truthful monotonic timing for one Serve batch callback == one model forward.

    All monotonic seconds use ``time.monotonic()`` from the serving replica's clock;
    they are comparable within a replica but not wall-clock timestamps. Ordering is
    enforced: admitted <= dispatched <= forward_started <= completed.
    """

    batch_id: str
    replica_id: str
    admitted_monotonic_s: float
    dispatched_monotonic_s: float
    forward_started_monotonic_s: float
    completed_monotonic_s: float
    effective_work_units: int
    request_count: int
    forward_count: int
    model_load_count: int

    def __post_init__(self) -> None:
        for name in ("batch_id", "replica_id"):
            _required_text(getattr(self, name), name)
        for name in ("admitted_monotonic_s", "dispatched_monotonic_s", "forward_started_monotonic_s", "completed_monotonic_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractValidationError(f"{name} must be a number")
        if not (self.admitted_monotonic_s <= self.dispatched_monotonic_s <= self.forward_started_monotonic_s <= self.completed_monotonic_s):
            raise ContractValidationError("trace timings must be monotonic: admitted<=dispatched<=forward_started<=completed")
        for name in ("effective_work_units", "request_count", "forward_count", "model_load_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ContractValidationError(f"{name} must be a non-negative integer")

    def to_wire(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "replica_id": self.replica_id,
            "admitted_monotonic_s": self.admitted_monotonic_s,
            "dispatched_monotonic_s": self.dispatched_monotonic_s,
            "forward_started_monotonic_s": self.forward_started_monotonic_s,
            "completed_monotonic_s": self.completed_monotonic_s,
            "effective_work_units": self.effective_work_units,
            "request_count": self.request_count,
            "forward_count": self.forward_count,
            "model_load_count": self.model_load_count,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "BatchTrace":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DeploymentStatus:
    deployment_name: str
    replica_id: str
    assigned_gpu: int
    loaded_models: tuple[str, ...]
    # Replica-local count of requests admitted (passed pre-batch validation) but not
    # yet dispatched to a Serve batch callback. This is NOT an authoritative
    # Serve-internal queue depth, which the replica cannot observe truthfully.
    admitted_pending: int
    running_batches: int
    model_load_count: int

    def __post_init__(self) -> None:
        _required_text(self.deployment_name, "deployment_name")
        _required_text(self.replica_id, "replica_id")
        if self.assigned_gpu < 0 or self.admitted_pending < 0 or self.running_batches < 0 or self.model_load_count < 0:
            raise ContractValidationError("deployment status counts must be non-negative")

    def to_wire(self) -> dict[str, Any]:
        return {
            "deployment_name": self.deployment_name,
            "replica_id": self.replica_id,
            "assigned_gpu": self.assigned_gpu,
            "loaded_models": list(self.loaded_models),
            "admitted_pending": self.admitted_pending,
            "running_batches": self.running_batches,
            "model_load_count": self.model_load_count,
        }


@dataclass(frozen=True)
class ServerIdentity:
    """Server-produced evidence binding an experimental response to its replica.

    The caller may choose a URL but cannot manufacture this value.  Scaling accepts
    a successful response only when this identity, its batch trace, and the pinned
    application release agree with the experiment plan.
    """

    experiment_id: str
    replica_id: str
    assigned_gpu: int
    worker_pid: int
    gcs_address: str
    http_port: int
    temp_dir: str
    model_revision: str
    checkpoint_digest: str
    schema_version: str
    release_sha: str
    # Content-addressed release and physical CUDA identity are optional for
    # production responses, but required by hardened experiment readiness.
    release_digest: str | None = None
    cuda_uuid: str | None = None
    module_root: str | None = None
    # Optional dependency identity keeps existing production responses wire-compatible.
    # Experimental DROID uses these values only after deriving them from its verified
    # immutable source manifest and loaded module location.
    dependency_digest: str | None = None
    dependency_root: str | None = None
    source_amendment_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "experiment_id", "replica_id", "gcs_address", "temp_dir", "model_revision",
            "checkpoint_digest", "schema_version", "release_sha",
        ):
            _required_text(getattr(self, name), name)
        for name in ("assigned_gpu", "worker_pid", "http_port"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractValidationError(f"{name} must be a non-negative integer")

    def to_wire(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "replica_id": self.replica_id,
            "assigned_gpu": self.assigned_gpu,
            "worker_pid": self.worker_pid,
            "gcs_address": self.gcs_address,
            "http_port": self.http_port,
            "temp_dir": self.temp_dir,
            "model_revision": self.model_revision,
            "checkpoint_digest": self.checkpoint_digest,
            "schema_version": self.schema_version,
            "release_sha": self.release_sha,
            "release_digest": self.release_digest,
            "cuda_uuid": self.cuda_uuid,
            "module_root": self.module_root,
            "dependency_digest": self.dependency_digest,
            "dependency_root": self.dependency_root,
            "source_amendment_id": self.source_amendment_id,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ServerIdentity":
        return cls(
            experiment_id=payload["experiment_id"],
            replica_id=payload["replica_id"],
            assigned_gpu=payload["assigned_gpu"],
            worker_pid=payload["worker_pid"],
            gcs_address=payload["gcs_address"],
            http_port=payload["http_port"],
            temp_dir=payload["temp_dir"],
            model_revision=payload["model_revision"],
            checkpoint_digest=payload["checkpoint_digest"],
            schema_version=payload["schema_version"],
            release_sha=payload["release_sha"],
            release_digest=payload.get("release_digest"),
            cuda_uuid=payload.get("cuda_uuid"),
            module_root=payload.get("module_root"),
            dependency_digest=payload.get("dependency_digest"),
            dependency_root=payload.get("dependency_root"),
            source_amendment_id=payload.get("source_amendment_id"),
        )


@dataclass(frozen=True)
class UniDepthResult:
    ownership: Ownership
    depth_m: TensorPayload
    K_px: TensorPayload
    confidence: TensorPayload
    spatial: SpatialMetadata
    model_revision: str
    trace: BatchTrace
    # Present for an instrumented experimental release.  It is optional so the
    # public UniDepth contract remains backward-compatible for production callers.
    batch_diagnostics: Mapping[str, Any] | None = None
    server_identity: ServerIdentity | None = None

    def to_wire(self, *, include_tensor_data: bool = True) -> dict[str, Any]:
        payload = {
            "ownership": self.ownership.to_wire(),
            "depth_m": self.depth_m.to_wire() if include_tensor_data else {"shape": list(self.depth_m.shape), "dtype": self.depth_m.dtype},
            "K_px": self.K_px.to_wire() if include_tensor_data else {"shape": list(self.K_px.shape), "dtype": self.K_px.dtype},
            "confidence": self.confidence.to_wire() if include_tensor_data else {"shape": list(self.confidence.shape), "dtype": self.confidence.dtype},
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            "batch_diagnostics": dict(self.batch_diagnostics) if self.batch_diagnostics is not None else None,
            "server_identity": self.server_identity.to_wire() if self.server_identity is not None else None,
        }
        return payload

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "UniDepthResult":
        diagnostics = payload.get("batch_diagnostics")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise ContractValidationError("batch_diagnostics must be an object or null")
        identity = payload.get("server_identity")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            depth_m=TensorPayload.from_wire(payload["depth_m"]),
            K_px=TensorPayload.from_wire(payload["K_px"]),
            confidence=TensorPayload.from_wire(payload["confidence"]),
            spatial=SpatialMetadata.from_mapping(payload["spatial"]),
            model_revision=payload["model_revision"],
            trace=BatchTrace.from_wire(payload["trace"]),
            batch_diagnostics=dict(diagnostics) if diagnostics is not None else None,
            server_identity=ServerIdentity.from_wire(identity) if identity is not None else None,
        )


@dataclass(frozen=True)
class UniDepthResponse:
    ownership: Ownership
    result: UniDepthResult | None = None
    error: ServiceError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("a response must contain exactly one result or error")
        if self.result is not None and self.result.ownership != self.ownership:
            raise ContractValidationError("result ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "result": self.result.to_wire() if self.result else None,
            "error": self.error.to_wire() if self.error else None,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "UniDepthResponse":
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            result=UniDepthResult.from_wire(payload["result"]) if payload.get("result") else None,
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
        )


def validation_error(ownership: Ownership, exc: Exception) -> UniDepthResponse:
    return UniDepthResponse(
        ownership=ownership,
        error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=ownership),
    )


# forward per Serve batch callback, truthful monotonic tracing, and no caller
# filesystem paths. The two logical APIs are physically colocated on one GPU1
# replica (detector + SAM2 + WiLoR) but retain separate batch queues, policies,
# and methods.

HANDS_RGB_DTYPE = "uint8"
WILOR_CROP_DTYPE = "float32"
WILOR_CROP_SIZE = 256
WILOR_CROP_CHANNELS = 3


class HandSide(int, Enum):
    """Detector class semantics for the WiLoR hand detector.pt."""

    LEFT = 0
    RIGHT = 1

    @classmethod
    def from_value(cls, value: Any) -> "HandSide":
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"handedness must be 0 (left) or 1 (right), got {value!r}") from exc

    def to_wire(self) -> int:
        return int(self.value)


@dataclass(frozen=True)
class HandsDetectRequest:
    """One RGB image for hand detection + mask.

    RGB is uint8 HWC matching the resident canonical bucket (like UniDepth).
    The detector does not divide by 255; Ultralytics owns normalization.
    """

    ownership: Ownership
    rgb: TensorPayload
    spatial: SpatialMetadata
    model_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.model_revision, "model_revision")
        if self.rgb.dtype != HANDS_RGB_DTYPE:
            raise ContractValidationError(
                f"hands RGB dtype must be {HANDS_RGB_DTYPE} only; float RGB is range-ambiguous and rejected"
            )
        expected_shape = (self.spatial.model_size.height, self.spatial.model_size.width, 3)
        if self.rgb.shape != expected_shape:
            raise ContractValidationError(f"hands RGB shape must be HxWx3 matching model_size {expected_shape}")

    @property
    def work_units(self) -> int:
        # One canonical-image work unit per request; the deployment advertises one
        # canonical HxW compatibility bucket so one Serve callback is one detector
        # forward.
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (self.rgb.dtype, self.rgb.shape, self.model_revision, self.options)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "HandsDetectRequest":
        reject_filesystem_fields(payload)
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            raise ContractValidationError("options must be an object")
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            rgb=TensorPayload.from_wire(payload.get("rgb", {})),
            spatial=SpatialMetadata.from_mapping(payload.get("spatial", {})),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            options=tuple(sorted((str(k), str(v)) for k, v in options.items())),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "rgb": self.rgb.to_wire(),
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class HandDetection:
    """Per-image hand detections in source-image pixel coordinates.

    ``masks`` exists only for revisions that run SAM2; it is a ``[K,H,W]``
    uint8 array produced by SAM2 prompted with detector boxes. YOLO-only
    revisions leave it absent rather than manufacturing a proxy mask.
    """

    boxes: TensorPayload          # [K,4] float32 xyxy in source-image px
    scores: TensorPayload         # [K] float32 detector confidence
    sides: TensorPayload          # [K] uint8 (0=left, 1=right)
    masks: TensorPayload | None   # [K,H,W] uint8 {0,1} REAL SAM2 masks when present
    visibility: TensorPayload     # [K] float32 in [0,1]
    uncertainty: TensorPayload    # [K] float32 >= 0 (1 - score + edge penalty)
    n_hands: int

    def __post_init__(self) -> None:
        if self.n_hands < 0:
            raise ContractValidationError("n_hands must be non-negative")
        if self.n_hands == 0:
            return
        k = self.n_hands
        for name, shape in (
            ("boxes", (k, 4)),
            ("scores", (k,)),
            ("sides", (k,)),
            ("visibility", (k,)),
            ("uncertainty", (k,)),
        ):
            field = getattr(self, name)
            if tuple(field.shape) != shape:
                raise ContractValidationError(f"{name} must have shape {shape}, got {tuple(field.shape)}")
        # When present, masks have shape [K,H,W] and are validated against the
        # source height/width at assembly.

    def to_wire(self) -> dict[str, Any]:
        wire = {
            "boxes": self.boxes.to_wire(),
            "scores": self.scores.to_wire(),
            "sides": self.sides.to_wire(),
            "visibility": self.visibility.to_wire(),
            "uncertainty": self.uncertainty.to_wire(),
            "n_hands": self.n_hands,
        }
        if self.masks is not None:
            wire["masks"] = self.masks.to_wire()
        return wire

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "HandDetection":
        return cls(
            boxes=TensorPayload.from_wire(payload["boxes"]),
            scores=TensorPayload.from_wire(payload["scores"]),
            sides=TensorPayload.from_wire(payload["sides"]),
            masks=(TensorPayload.from_wire(payload["masks"]) if "masks" in payload else None),
            visibility=TensorPayload.from_wire(payload["visibility"]),
            uncertainty=TensorPayload.from_wire(payload["uncertainty"]),
            n_hands=int(payload["n_hands"]),
        )

    @classmethod
    def empty(cls, height: int, width: int, *, include_masks: bool = True) -> "HandDetection":
        import numpy as np

        zero = lambda shape, dtype: TensorPayload(data=np.zeros(shape, dtype=np.dtype(dtype)).tobytes(), shape=shape, dtype=dtype)
        return cls(
            boxes=zero((0, 4), "float32"),
            scores=zero((0,), "float32"),
            sides=zero((0,), "uint8"),
            masks=zero((0, height, width), "uint8") if include_masks else None,
            visibility=zero((0,), "float32"),
            uncertainty=zero((0,), "float32"),
            n_hands=0,
        )


@dataclass(frozen=True)
class HandsDetectResult:
    ownership: Ownership
    detection: HandDetection
    spatial: SpatialMetadata
    model_revision: str
    trace: BatchTrace
    # Number of SAM2 image-embedding forwards (one per image with >=1 box) for
    # honest accounting; the detector forward_count is in the trace.
    sam2_mask_calls: int
    # Explicit benchmark treatments attest their live wire configuration here.
    # Omit it from normal multipart metadata to preserve the established wire bytes.
    batch_diagnostics: Mapping[str, Any] | None = None
    # Isolated workers derive this only after their resident models load.
    server_identity: ServerIdentity | None = None

    def to_wire(self) -> dict[str, Any]:
        wire = {
            "ownership": self.ownership.to_wire(),
            "detection": self.detection.to_wire(),
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            "sam2_mask_calls": self.sam2_mask_calls,
        }
        if self.batch_diagnostics is not None:
            wire["batch_diagnostics"] = dict(self.batch_diagnostics)
        if self.server_identity is not None:
            wire["server_identity"] = self.server_identity.to_wire()
        return wire


@dataclass(frozen=True)
class HandsDetectResponse:
    ownership: Ownership
    result: HandsDetectResult | None = None
    error: ServiceError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("a response must contain exactly one result or error")
        if self.result is not None and self.result.ownership != self.ownership:
            raise ContractValidationError("result ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "result": self.result.to_wire() if self.result else None,
            "error": self.error.to_wire() if self.error else None,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "HandsDetectResponse":
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            result=HandsDetectResult.from_wire(payload["result"]) if payload.get("result") else None,
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
        )


@classmethod  # type: ignore[no-redef]
def _hands_result_from_wire(cls: type, payload: Mapping[str, Any]) -> "HandsDetectResult":
    return cls(
        ownership=Ownership.from_mapping(payload["ownership"]),
        detection=HandDetection.from_wire(payload["detection"]) if payload.get("detection") else HandDetection.empty(0, 0),
        spatial=SpatialMetadata.from_mapping(payload["spatial"]),
        model_revision=payload["model_revision"],
        trace=BatchTrace.from_wire(payload["trace"]),
        sam2_mask_calls=int(payload.get("sam2_mask_calls", 0)),
        batch_diagnostics=dict(payload["batch_diagnostics"]) if payload.get("batch_diagnostics") is not None else None,
        server_identity=(ServerIdentity.from_wire(payload["server_identity"]) if isinstance(payload.get("server_identity"), Mapping) else None),
    )


HandsDetectResult.from_wire = classmethod(_hands_result_from_wire)  # type: ignore[assignment]


@dataclass(frozen=True)
class WiLoRReconstructRequest:
    """One model-native hand crop for WiLoR reconstruction.

    The crop is a normalized ``[3,256,256]`` float32 tensor exactly as produced by
    WiLoR's ViTDetDataset (ImageNet mean/std scaled by 255). Detection and crop
    construction are upstream CPU work; this contract is the WiLoR forward only.
    ``box_center``/``box_size``/``img_size`` carry the crop-to-source transform so
    the adapter can lift the weak-perspective camera translation into the source
    image frame (``cam_crop_to_full``).
    """

    ownership: Ownership
    crop: TensorPayload
    handedness: HandSide
    box_center: tuple[float, float]      # crop center in source-image px
    box_size: float                       # expanded square bbox size (source px)
    img_size: tuple[float, float]         # source (width, height) in px
    model_revision: str
    source_K_px: tuple[tuple[float, float, float], ...] | None = None
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.model_revision, "model_revision")
        expected_crop = (WILOR_CROP_CHANNELS, WILOR_CROP_SIZE, WILOR_CROP_SIZE)
        if self.crop.shape != expected_crop:
            raise ContractValidationError(f"wilor crop must be {expected_crop}, got {tuple(self.crop.shape)}")
        if self.crop.dtype != WILOR_CROP_DTYPE:
            raise ContractValidationError(f"wilor crop dtype must be {WILOR_CROP_DTYPE}")
        if len(self.box_center) != 2:
            raise ContractValidationError("box_center must be (cx, cy)")
        if not isinstance(self.box_size, (int, float)) or isinstance(self.box_size, bool) or self.box_size <= 0:
            raise ContractValidationError("box_size must be a positive number")
        if len(self.img_size) != 2 or any(v <= 0 for v in self.img_size):
            raise ContractValidationError("img_size must be (width, height) with positive values")
        if self.source_K_px is not None:
            object.__setattr__(self, "source_K_px", _matrix3(self.source_K_px, "source_K_px"))

    @property
    def work_units(self) -> int:
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (self.crop.dtype, self.crop.shape, self.model_revision, self.options)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "WiLoRReconstructRequest":
        reject_filesystem_fields(payload)
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            raise ContractValidationError("options must be an object")
        kpx = payload.get("source_K_px")
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            crop=TensorPayload.from_wire(payload.get("crop", {})),
            handedness=HandSide.from_value(payload.get("handedness")),
            box_center=tuple(float(v) for v in payload.get("box_center", (0.0, 0.0))),
            box_size=float(payload.get("box_size", 0.0)),
            img_size=tuple(float(v) for v in payload.get("img_size", (0.0, 0.0))),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            source_K_px=kpx,
            options=tuple(sorted((str(k), str(v)) for k, v in options.items())),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "crop": self.crop.to_wire(),
            "handedness": int(self.handedness),
            "box_center": list(self.box_center),
            "box_size": self.box_size,
            "img_size": list(self.img_size),
            "source_K_px": [list(row) for row in self.source_K_px] if self.source_K_px is not None else None,
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class ManoOutput:
    """Reproducible MANO state + surface for one hand crop.

    Rotation matrices (pose2rot=False convention) are returned as-is from the
    resident WiLoR forward. ``vertices`` are the 778 MANO surface vertices in the
    hand-root frame; ``cam_t_full`` lifts them into the source-image camera frame
    via ``cam_crop_to_full`` with the resident focal-length semantics.
    """

    global_orient: TensorPayload   # [1,3,3] float32
    hand_pose: TensorPayload       # [15,3,3] float32
    betas: TensorPayload           # [10] float32
    vertices: TensorPayload        # [778,3] float32
    joints: TensorPayload          # [J,3] float32 pred_keypoints_3d
    cam_t_full: TensorPayload      # [3] float32 source-image camera translation
    pred_cam: TensorPayload        # [3] float32 weak-perspective crop cam
    keypoints_2d: TensorPayload    # [J,2] float32 projected into source image
    focal_length: float            # px used for projection
    confidence: TensorPayload      # [1] float32 (model-supplied where available)
    uncertainty: TensorPayload     # [1] float32 >= 0
    n_vertices: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "global_orient": self.global_orient.to_wire(),
            "hand_pose": self.hand_pose.to_wire(),
            "betas": self.betas.to_wire(),
            "vertices": self.vertices.to_wire(),
            "joints": self.joints.to_wire(),
            "cam_t_full": self.cam_t_full.to_wire(),
            "pred_cam": self.pred_cam.to_wire(),
            "keypoints_2d": self.keypoints_2d.to_wire(),
            "focal_length": self.focal_length,
            "confidence": self.confidence.to_wire(),
            "uncertainty": self.uncertainty.to_wire(),
            "n_vertices": self.n_vertices,
        }


@dataclass(frozen=True)
class WiLoRReconstructResult:
    ownership: Ownership
    mano: ManoOutput
    handedness: HandSide
    model_revision: str
    trace: BatchTrace
    # Explicit benchmark treatments attest their live wire configuration here.
    # Omit it from normal multipart metadata to preserve the established wire bytes.
    batch_diagnostics: Mapping[str, Any] | None = None
    # Isolated workers derive this only after their resident models load.
    server_identity: ServerIdentity | None = None

    def to_wire(self) -> dict[str, Any]:
        wire = {
            "ownership": self.ownership.to_wire(),
            "mano": self.mano.to_wire(),
            "handedness": int(self.handedness),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
        }
        if self.batch_diagnostics is not None:
            wire["batch_diagnostics"] = dict(self.batch_diagnostics)
        if self.server_identity is not None:
            wire["server_identity"] = self.server_identity.to_wire()
        return wire


@dataclass(frozen=True)
class WiLoRReconstructResponse:
    ownership: Ownership
    result: WiLoRReconstructResult | None = None
    error: ServiceError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("a response must contain exactly one result or error")
        if self.result is not None and self.result.ownership != self.ownership:
            raise ContractValidationError("result ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "result": self.result.to_wire() if self.result else None,
            "error": self.error.to_wire() if self.error else None,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "WiLoRReconstructResponse":
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            result=WiLoRReconstructResult.from_wire(payload["result"]) if payload.get("result") else None,
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
        )


@classmethod  # type: ignore[no-redef]
def _wilor_result_from_wire(cls: type, payload: Mapping[str, Any]) -> "WiLoRReconstructResult":
    return cls(
        ownership=Ownership.from_mapping(payload["ownership"]),
        mano=ManoOutput.from_wire(payload["mano"]),
        handedness=HandSide.from_value(payload["handedness"]),
        model_revision=payload["model_revision"],
        trace=BatchTrace.from_wire(payload["trace"]),
        batch_diagnostics=dict(payload["batch_diagnostics"]) if payload.get("batch_diagnostics") is not None else None,
        server_identity=(ServerIdentity.from_wire(payload["server_identity"]) if isinstance(payload.get("server_identity"), Mapping) else None),
    )


WiLoRReconstructResult.from_wire = classmethod(_wilor_result_from_wire)  # type: ignore[assignment]


@classmethod  # type: ignore[no-redef]
def _mano_from_wire(cls: type, payload: Mapping[str, Any]) -> "ManoOutput":
    return cls(
        global_orient=TensorPayload.from_wire(payload["global_orient"]),
        hand_pose=TensorPayload.from_wire(payload["hand_pose"]),
        betas=TensorPayload.from_wire(payload["betas"]),
        vertices=TensorPayload.from_wire(payload["vertices"]),
        joints=TensorPayload.from_wire(payload["joints"]),
        cam_t_full=TensorPayload.from_wire(payload["cam_t_full"]),
        pred_cam=TensorPayload.from_wire(payload["pred_cam"]),
        keypoints_2d=TensorPayload.from_wire(payload["keypoints_2d"]),
        focal_length=float(payload["focal_length"]),
        confidence=TensorPayload.from_wire(payload["confidence"]),
        uncertainty=TensorPayload.from_wire(payload["uncertainty"]),
        n_vertices=int(payload["n_vertices"]),
    )


ManoOutput.from_wire = classmethod(_mano_from_wire)  # type: ignore[assignment]


@dataclass(frozen=True)
class Cosmos3Result:
    """Structured semantic result for one ``cosmos3.reason`` request.

    Carries generated text, finish/stop reason, token counts, truthful timings, the
    server-owned model revision, and batch/replica provenance. All timings are
    ``time.monotonic()`` seconds from the serving replica's clock; ``e2e_s`` is the
    end-to-end request latency observed at the replica (admission to result).
    """

    ownership: Ownership
    text: str
    finish_reason: str
    stop_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    timings: Mapping[str, float]
    model_revision: str
    trace: BatchTrace
    media_provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.text, "text") if self.text is not None else None
        _required_text(self.finish_reason, "finish_reason")
        _required_text(self.model_revision, "model_revision")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractValidationError(f"{name} must be a non-negative integer")
        for key, value in self.timings.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractValidationError(f"timings.{key} must be a number")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "text": self.text,
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "timings": dict(self.timings),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            "media_provenance": [dict(m) for m in self.media_provenance],
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "Cosmos3Result":
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            text=payload["text"],
            finish_reason=payload["finish_reason"],
            stop_reason=payload.get("stop_reason"),
            prompt_tokens=int(payload["prompt_tokens"]),
            completion_tokens=int(payload["completion_tokens"]),
            total_tokens=int(payload["total_tokens"]),
            timings=dict(payload.get("timings", {})),
            model_revision=payload["model_revision"],
            trace=BatchTrace.from_wire(payload["trace"]),
            media_provenance=tuple(dict(m) for m in payload.get("media_provenance", [])),
        )


@dataclass(frozen=True)
class Cosmos3Response:
    ownership: Ownership
    result: Cosmos3Result | None = None
    error: ServiceError | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("a response must contain exactly one result or error")
        if self.result is not None and self.result.ownership != self.ownership:
            raise ContractValidationError("result ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "result": self.result.to_wire() if self.result else None,
            "error": self.error.to_wire() if self.error else None,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "Cosmos3Response":
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            result=Cosmos3Result.from_wire(payload["result"]) if payload.get("result") else None,
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
        )


def cosmos3_validation_error(ownership: Ownership, exc: Exception) -> Cosmos3Response:
    return Cosmos3Response(
        ownership=ownership,
        error=ServiceError(ErrorCode.VALIDATION, str(exc), retryable=False, ownership=ownership),
    )


# ---------------------------------------------------------------------------
# Stateful DROID session API contracts.
#
# These model the three logical methods that share one resident DroidNet/CUDA
# backend on GPU2: ``droid.create_session``, ``droid.push_frame``, and
# ``droid.finalize``. The contract is deliberately Python 3.10-compatible and
# importable without Ray.
#
# Pose convention invariants (validated against the real DROID source in
# ``droid_slam/droid.py``):
#
# * DROID stores keyframe poses internally as **camera-from-world** (``lietorch
#   SE3`` quaternions in ``DepthVideo.poses``). The first keyframe is the identity.
# * ``Droid.terminate()`` returns ``camera_trajectory.inv()``: that inverse is
#   **world-from-camera** (``T_world_camera``).
# * The dense (trajectory-filler) and keyframe exports both derive from the same
#   canonical internal ``DepthVideo.poses`` and both invert to ``T_world_camera``.
#   They must never be independently interpreted. The synthetic nonidentity SE(3)
#   test below forces this: a sign/quaternion-inversion error makes dense and
#   keyframe disagree under any nonidentity transform.
# * DROID's monocular translation is up to scale; every finalize result declares
#   ``scale_status="up_to_scale"`` so downstream camera-scale reuse (UniDepth
#   pairing) is explicit, never silent.
#
# Cross-session batching honesty: only the feature network (``fnet``) forward is
# batchable across sessions — compatible next-ready frames from distinct sessions
# stack into one ``[B,C,H,W]`` ``fnet`` forward. Correlation, recurrent update,
# context-network (``cnet``) for newly added keyframes, factor-graph mutation, and
# bundle adjustment are session-local and are traced as session-local forwards, not
# as a fused batch.
# ---------------------------------------------------------------------------

DROID_RGB_DTYPE = "uint8"
# DROID's internal feature/pose grids operate at 1/8 resolution, so the model
# image dimensions must be divisible by 8. The representative EgoScale prior
# (1080x1920 -> target_area 196608) yields 328x584.
_DROID_MODEL_DIM_MULTIPLE = 8


@dataclass(frozen=True)
class DroidCamera:
    """Canonical camera/pixel contract recorded at session creation.

    ``intrinsics`` is the pinhole prior in source-image pixels as the 4-vector
    ``[fx, fy, cx, cy]`` DROID consumes (matching ``DepthVideo.intrinsics``).
    ``K_px`` is the optional full 3x3 source intrinsics for downstream reuse.
    No field here may select a caller filesystem path.
    """

    intrinsics: tuple[float, float, float, float]
    source_size: ImageSize
    pixel_transform: PixelTransform
    K_px: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.intrinsics) != 4:
            raise ContractValidationError("DroidCamera.intrinsics must be [fx, fy, cx, cy]")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.intrinsics):
            raise ContractValidationError("DroidCamera.intrinsics must be numeric")
        if any(v <= 0.0 for v in self.intrinsics[:2]):
            raise ContractValidationError("DroidCamera fx/fy must be positive")
        if self.K_px is not None:
            object.__setattr__(self, "K_px", _matrix3(self.K_px, "K_px"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DroidCamera":
        return cls(
            intrinsics=tuple(float(x) for x in payload["intrinsics"]),
            source_size=ImageSize.from_mapping(payload.get("source_size", {})),
            pixel_transform=PixelTransform.from_mapping(payload.get("pixel_transform", {})),
            K_px=payload.get("K_px"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "intrinsics": list(self.intrinsics),
            "source_size": self.source_size.to_wire(),
            "pixel_transform": self.pixel_transform.to_wire(),
            "K_px": [list(row) for row in self.K_px] if self.K_px is not None else None,
        }


@dataclass(frozen=True)
class DroidImageShape:
    """Model image dimensions for a session. Must be divisible by 8."""

    height: int
    width: int

    def __post_init__(self) -> None:
        _positive_int(self.height, "height")
        _positive_int(self.width, "width")
        if self.height % _DROID_MODEL_DIM_MULTIPLE != 0 or self.width % _DROID_MODEL_DIM_MULTIPLE != 0:
            raise ContractValidationError(
                f"DROID model image dimensions must be divisible by {_DROID_MODEL_DIM_MULTIPLE}; "
                f"got {self.height}x{self.width}"
            )

    @property
    def shape_hwc(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DroidImageShape":
        return cls(height=_positive_int(payload.get("height"), "height"), width=_positive_int(payload.get("width"), "width"))

    def to_wire(self) -> dict[str, int]:
        return {"height": self.height, "width": self.width}


@dataclass(frozen=True)
class DroidSessionOptions:
    """Server-owned DROID session tuning. No request field selects a server path.

    These mirror the real DROID args (``buffer``, ``filter_thresh``, ``warmup``,
    ``keyframe_thresh``, frontend/backend thresholds, ``upsample``). Defaults match
    the representative EgoScale run validated in task memory. A request may only
    override documented numeric options; checkpoint/weights are never request-set.
    """

    buffer: int = 1024
    filter_thresh: float = 2.4
    warmup: int = 8
    keyframe_thresh: float = 4.0
    frontend_thresh: float = 16.0
    frontend_window: int = 25
    frontend_radius: int = 2
    frontend_nms: int = 1
    backend_thresh: float = 22.0
    backend_radius: int = 2
    backend_nms: int = 3
    upsample: bool = True
    beta: float = 0.3
    stereo: bool = False

    def __post_init__(self) -> None:
        if self.buffer <= 0 or self.warmup <= 0:
            raise ContractValidationError("buffer and warmup must be positive")
        if self.frontend_window <= 0 or self.frontend_radius <= 0 or self.backend_radius <= 0:
            raise ContractValidationError("DROID window/radius options must be positive")
        if self.backend_nms < 0 or self.frontend_nms < 0:
            raise ContractValidationError("DROID nms options must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DroidSessionOptions":
        known = {f.name for f in fields(cls)}
        unknown = set(payload.keys()) - known
        if unknown:
            raise ContractValidationError(f"unknown DROID session options: {sorted(unknown)}")
        coerced: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in payload:
                coerced[f.name] = payload[f.name]
        return cls(**coerced)

    def to_wire(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class DroidCreateSessionRequest:
    ownership: Ownership
    camera: DroidCamera
    image_shape: DroidImageShape
    model_revision: str
    options: DroidSessionOptions = field(default_factory=DroidSessionOptions)

    def __post_init__(self) -> None:
        _required_text(self.model_revision, "model_revision")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidCreateSessionRequest":
        reject_filesystem_fields(payload)
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            camera=DroidCamera.from_mapping(payload.get("camera", {})),
            image_shape=DroidImageShape.from_mapping(payload.get("image_shape", {})),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            options=DroidSessionOptions.from_mapping(payload.get("options", {})),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "camera": self.camera.to_wire(),
            "image_shape": self.image_shape.to_wire(),
            "model_revision": self.model_revision,
            "options": self.options.to_wire(),
        }


@dataclass(frozen=True)
class DroidFrameRequest:
    """One timestamped model-size frame for an existing session.

    ``rgb`` is uint8 HWC in the model image grid declared at session creation.
    DROID consumes BGR internally; the adapter performs the RGB->BGR channel swap
    exactly as ``MotionFilter.track`` does ``image[...,[2,1,0]]``, so callers send
    RGB. ``static_confidence_mask`` is the optional real hand/object static mask
    evidence HaWoR/masked-DROID requires; when present it is downsampled to the
    1/8 feature grid by the adapter. No caller filesystem paths.
    """

    ownership: Ownership
    session_id: str
    frame_id: str
    source_timestamp_s: float
    rgb: TensorPayload
    static_confidence_mask: TensorPayload | None = None
    depth_m: TensorPayload | None = None
    model_revision: str = ""

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")
        _required_text(self.frame_id, "frame_id")
        if not isinstance(self.source_timestamp_s, (int, float)) or isinstance(self.source_timestamp_s, bool):
            raise ContractValidationError("source_timestamp_s must be numeric")
        if self.rgb.dtype != DROID_RGB_DTYPE:
            raise ContractValidationError(f"DROID RGB dtype must be {DROID_RGB_DTYPE} only")
        if len(self.rgb.shape) != 3 or self.rgb.shape[2] != 3:
            raise ContractValidationError("DROID RGB shape must be HxWx3")
        if self.rgb.shape[0] % _DROID_MODEL_DIM_MULTIPLE != 0 or self.rgb.shape[1] % _DROID_MODEL_DIM_MULTIPLE != 0:
            raise ContractValidationError("DROID RGB H/W must be divisible by 8")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidFrameRequest":
        reject_filesystem_fields(payload)
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            session_id=_required_text(payload.get("session_id"), "session_id"),
            frame_id=_required_text(payload.get("frame_id"), "frame_id"),
            source_timestamp_s=float(payload["source_timestamp_s"]),
            rgb=TensorPayload.from_wire(payload.get("rgb", {})),
            static_confidence_mask=TensorPayload.from_wire(payload["static_confidence_mask"]) if payload.get("static_confidence_mask") else None,
            depth_m=TensorPayload.from_wire(payload["depth_m"]) if payload.get("depth_m") else None,
            model_revision=_required_text(payload.get("model_revision", ""), "model_revision"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "source_timestamp_s": self.source_timestamp_s,
            "rgb": self.rgb.to_wire(),
            "static_confidence_mask": self.static_confidence_mask.to_wire() if self.static_confidence_mask else None,
            "depth_m": self.depth_m.to_wire() if self.depth_m else None,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class DroidFinalizeRequest:
    ownership: Ownership
    session_id: str
    model_revision: str = ""

    def __post_init__(self) -> None:
        _required_text(self.session_id, "session_id")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidFinalizeRequest":
        reject_filesystem_fields(payload)
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            session_id=_required_text(payload.get("session_id"), "session_id"),
            model_revision=_required_text(payload.get("model_revision", ""), "model_revision"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "session_id": self.session_id,
            "model_revision": self.model_revision,
        }


_DROID_PHASE_NAMES = (
    "preprocessing_h2d",
    "fnet",
    "correlation_update",
    "cnet",
    "frontend_ba",
    "backend_7",
    "backend_12",
    "filler",
    "encoding",
)


@dataclass(frozen=True)
class DroidPhaseTiming:
    """Per-request host spans for DROID's non-interchangeable execution stages.

    Values are host wall-clock durations around the named stage, not inferred GPU
    kernel time. A stage that did not run or could not be observed is ``None`` and
    must appear in ``unavailable_stages``; callers must not replace it with zero.
    """

    preprocessing_h2d_s: float | None = None
    fnet_s: float | None = None
    correlation_update_s: float | None = None
    cnet_s: float | None = None
    frontend_ba_s: float | None = None
    backend_7_s: float | None = None
    backend_12_s: float | None = None
    filler_s: float | None = None
    encoding_s: float | None = None
    unavailable_stages: tuple[str, ...] = _DROID_PHASE_NAMES
    # Adapter measurements are host monotonic spans. They are deliberately not
    # presented as kernel timing. CUDA event timing and HTTP serialization happen
    # outside or beyond this adapter unless explicitly instrumented.
    measurement_basis: str = "host_monotonic_span"
    cuda_event_elapsed_s: Mapping[str, float] = field(default_factory=dict)
    cuda_event_unavailable_stages: tuple[str, ...] = _DROID_PHASE_NAMES
    http_serialization_unavailable: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.unavailable_stages) - set(_DROID_PHASE_NAMES)
        if unknown:
            raise ContractValidationError(f"unknown DROID timing stages: {sorted(unknown)}")
        if len(set(self.unavailable_stages)) != len(self.unavailable_stages):
            raise ContractValidationError("DROID unavailable timing stages must be unique")
        if self.measurement_basis != "host_monotonic_span":
            raise ContractValidationError("DROID phase timing measurement_basis must be host_monotonic_span")
        cuda_unknown = set(self.cuda_event_elapsed_s) | (set(self.cuda_event_unavailable_stages) - set(_DROID_PHASE_NAMES))
        if cuda_unknown - set(_DROID_PHASE_NAMES):
            raise ContractValidationError(f"unknown DROID CUDA timing stages: {sorted(cuda_unknown - set(_DROID_PHASE_NAMES))}")
        if len(set(self.cuda_event_unavailable_stages)) != len(self.cuda_event_unavailable_stages):
            raise ContractValidationError("DROID CUDA unavailable timing stages must be unique")
        for stage, value in self.cuda_event_elapsed_s.items():
            if stage in self.cuda_event_unavailable_stages:
                raise ContractValidationError(f"measured CUDA DROID stage {stage} cannot be unavailable")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0.0:
                raise ContractValidationError(f"DROID CUDA stage {stage} duration must be a non-negative number")
        if set(self.cuda_event_elapsed_s) | set(self.cuda_event_unavailable_stages) != set(_DROID_PHASE_NAMES):
            raise ContractValidationError("every DROID CUDA timing stage must be measured or explicitly unavailable")
        if self.http_serialization_unavailable is not True:
            raise ContractValidationError("DROID adapter cannot claim HTTP serialization timing")
        unavailable = set(self.unavailable_stages)
        for stage in _DROID_PHASE_NAMES:
            value = getattr(self, f"{stage}_s")
            if value is None:
                if stage not in unavailable:
                    raise ContractValidationError(f"unavailable DROID stage {stage} must be explicit")
            else:
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0.0:
                    raise ContractValidationError(f"DROID stage {stage} duration must be a non-negative number")
                if stage in unavailable:
                    raise ContractValidationError(f"measured DROID stage {stage} cannot be unavailable")

    def to_wire(self) -> dict[str, Any]:
        return {
            **{f"{stage}_s": getattr(self, f"{stage}_s") for stage in _DROID_PHASE_NAMES},
            "unavailable_stages": list(self.unavailable_stages),
            "measurement_basis": self.measurement_basis,
            "cuda_event_elapsed_s": dict(self.cuda_event_elapsed_s),
            "cuda_event_unavailable_stages": list(self.cuda_event_unavailable_stages),
            "http_serialization_unavailable": self.http_serialization_unavailable,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidPhaseTiming":
        return cls(
            **{f"{stage}_s": payload.get(f"{stage}_s") for stage in _DROID_PHASE_NAMES},
            unavailable_stages=tuple(str(value) for value in payload.get("unavailable_stages", ())),
            measurement_basis=str(payload.get("measurement_basis", "host_monotonic_span")),
            cuda_event_elapsed_s={str(k): float(v) for k, v in payload.get("cuda_event_elapsed_s", {}).items()},
            cuda_event_unavailable_stages=tuple(str(value) for value in payload.get("cuda_event_unavailable_stages", _DROID_PHASE_NAMES)),
            http_serialization_unavailable=bool(payload.get("http_serialization_unavailable", True)),
        )


@dataclass(frozen=True)
class DroidBatchTrace:
    """Truthful trace separating fused feature-network work from session-local work.

    One Serve ``push_frame`` batch callback executes exactly ONE fused ``fnet``
    forward over compatible next-ready frames from distinct sessions
    (``fnet_forward_count == 1``). Correlation, recurrent update, context-network
    for newly added keyframes, factor-graph mutation, and bundle adjustment are
    session-local; each item records ``session_local_forward_count`` honestly. A
    callback is never reported as a single fused forward for the local stages.
    """

    batch_id: str
    replica_id: str
    admitted_monotonic_s: float
    dispatched_monotonic_s: float
    fnet_forward_started_monotonic_s: float
    fnet_completed_monotonic_s: float
    completed_monotonic_s: float
    fnet_forward_count: int
    session_local_forward_count: int
    request_count: int
    effective_work_units: int
    model_load_count: int
    # Per-item session identity so a batch trace can be split back to callers.
    session_ids: tuple[str, ...]
    phase_timing: DroidPhaseTiming = field(default_factory=DroidPhaseTiming)

    def __post_init__(self) -> None:
        for name in ("batch_id", "replica_id"):
            _required_text(getattr(self, name), name)
        for name in ("admitted_monotonic_s", "dispatched_monotonic_s", "fnet_forward_started_monotonic_s",
                     "fnet_completed_monotonic_s", "completed_monotonic_s"):
            if not isinstance(getattr(self, name), (int, float)) or isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be a number")
        if not (self.admitted_monotonic_s <= self.dispatched_monotonic_s
                <= self.fnet_forward_started_monotonic_s <= self.fnet_completed_monotonic_s
                <= self.completed_monotonic_s):
            raise ContractValidationError(
                "DROID trace timings must be monotonic: admitted<=dispatched<=fnet_started<=fnet_done<=completed"
            )
        for name in ("fnet_forward_count", "session_local_forward_count", "request_count", "effective_work_units", "model_load_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ContractValidationError(f"{name} must be a non-negative integer")
        if len(self.session_ids) != self.request_count:
            raise ContractValidationError("session_ids length must equal request_count")
        if not isinstance(self.phase_timing, DroidPhaseTiming):
            raise ContractValidationError("phase_timing must be a DroidPhaseTiming")

    def to_wire(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "replica_id": self.replica_id,
            "admitted_monotonic_s": self.admitted_monotonic_s,
            "dispatched_monotonic_s": self.dispatched_monotonic_s,
            "fnet_forward_started_monotonic_s": self.fnet_forward_started_monotonic_s,
            "fnet_completed_monotonic_s": self.fnet_completed_monotonic_s,
            "completed_monotonic_s": self.completed_monotonic_s,
            "fnet_forward_count": self.fnet_forward_count,
            "session_local_forward_count": self.session_local_forward_count,
            "request_count": self.request_count,
            "effective_work_units": self.effective_work_units,
            "model_load_count": self.model_load_count,
            "session_ids": list(self.session_ids),
            "phase_timing": self.phase_timing.to_wire(),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidBatchTrace":
        values = {name: payload[name] for name in cls.__dataclass_fields__ if name != "phase_timing"}
        values["session_ids"] = tuple(str(value) for value in payload.get("session_ids", ()))
        phase_payload = payload.get("phase_timing")
        values["phase_timing"] = (
            DroidPhaseTiming.from_wire(phase_payload)
            if isinstance(phase_payload, Mapping)
            else DroidPhaseTiming()
        )
        return cls(**values)


@dataclass(frozen=True)
class FrameValidity:
    """Per-frame validity/uncertainty for a pushed frame."""

    frame_id: str
    source_timestamp_s: float
    admitted: bool
    keyframe_added: bool
    # Why a frame was skipped (e.g. insufficient motion per DROID's motion filter).
    skip_reason: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "source_timestamp_s": self.source_timestamp_s,
            "admitted": self.admitted,
            "keyframe_added": self.keyframe_added,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "FrameValidity":
        return cls(
            frame_id=_required_text(payload.get("frame_id"), "frame_id"),
            source_timestamp_s=float(payload["source_timestamp_s"]),
            admitted=bool(payload.get("admitted")),
            keyframe_added=bool(payload.get("keyframe_added")),
            skip_reason=payload.get("skip_reason"),
        )


@dataclass(frozen=True)
class StepStatus:
    """Result of ``droid.push_frame``: per-frame step status with honest trace."""

    ownership: Ownership
    session_id: str
    frame_id: str
    source_timestamp_s: float
    validity: FrameValidity
    keyframe_count: int
    trace: DroidBatchTrace

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "source_timestamp_s": self.source_timestamp_s,
            "validity": self.validity.to_wire(),
            "keyframe_count": self.keyframe_count,
            "trace": self.trace.to_wire(),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "StepStatus":
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            session_id=_required_text(payload.get("session_id"), "session_id"),
            frame_id=_required_text(payload.get("frame_id"), "frame_id"),
            source_timestamp_s=float(payload["source_timestamp_s"]),
            validity=FrameValidity.from_wire(payload.get("validity", {})),
            keyframe_count=_non_negative_int(payload.get("keyframe_count"), "keyframe_count"),
            trace=DroidBatchTrace.from_wire(payload.get("trace", {})),
        )


@dataclass(frozen=True)
class KeyframeSourceMapping:
    """Mapping between DROID keyframe index and source frame/timestamp.

    Timeline joins use source timestamps, never assumed frame-index agreement
    across decoded streams. ``source_timestamp_s`` is the caller-declared timestamp
    preserved from ``push_frame``; ``source_frame_id`` is the caller frame id.
    """

    keyframe_index: int
    source_frame_id: str
    source_timestamp_s: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "keyframe_index": self.keyframe_index,
            "source_frame_id": self.source_frame_id,
            "source_timestamp_s": self.source_timestamp_s,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "KeyframeSourceMapping":
        return cls(
            keyframe_index=_non_negative_int(payload.get("keyframe_index"), "keyframe_index"),
            source_frame_id=_required_text(payload.get("source_frame_id"), "source_frame_id"),
            source_timestamp_s=float(payload["source_timestamp_s"]),
        )


@dataclass(frozen=True)
class DenseSourceMapping:
    """Mapping for one dense (non-keyframe) pose to its source frame/timestamp."""

    dense_index: int
    source_frame_id: str
    source_timestamp_s: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "dense_index": self.dense_index,
            "source_frame_id": self.source_frame_id,
            "source_timestamp_s": self.source_timestamp_s,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DenseSourceMapping":
        return cls(
            dense_index=_non_negative_int(payload.get("dense_index"), "dense_index"),
            source_frame_id=_required_text(payload.get("source_frame_id"), "source_frame_id"),
            source_timestamp_s=float(payload["source_timestamp_s"]),
        )


@dataclass(frozen=True)
class DroidUncertainty:
    """Honest uncertainty/QC for a finalized camera state.

    DROID's monocular translation is up to scale; ``scale_status`` is always
    ``up_to_scale`` until downstream UniDepth metric-scale pairing is applied.
    ``reprojection_error`` is DROID's own backend residual; ``valid_keyframe_ratio``
    and ``finite_pose_ratio`` are QC fractions.
    """

    scale_status: str
    reprojection_error: float | None
    valid_keyframe_ratio: float
    finite_pose_ratio: float
    note: str | None = None

    def __post_init__(self) -> None:
        if self.scale_status not in {"up_to_scale", "metric_paired"}:
            raise ContractValidationError("scale_status must be 'up_to_scale' or 'metric_paired'")
        for name in ("valid_keyframe_ratio", "finite_pose_ratio"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
                raise ContractValidationError(f"{name} must be a fraction in [0, 1]")

    def to_wire(self) -> dict[str, Any]:
        return {
            "scale_status": self.scale_status,
            "reprojection_error": self.reprojection_error,
            "valid_keyframe_ratio": self.valid_keyframe_ratio,
            "finite_pose_ratio": self.finite_pose_ratio,
            "note": self.note,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidUncertainty":
        return cls(
            scale_status=_required_text(payload.get("scale_status"), "scale_status"),
            reprojection_error=(
                float(payload["reprojection_error"])
                if payload.get("reprojection_error") is not None else None
            ),
            valid_keyframe_ratio=float(payload["valid_keyframe_ratio"]),
            finite_pose_ratio=float(payload["finite_pose_ratio"]),
            note=payload.get("note"),
        )


@dataclass(frozen=True)
class CameraState:
    """``droid.finalize`` result: explicit metric-free camera state.

    ``T_world_camera`` and ``T_camera_world`` are both derived from the same
    canonical source (DROID's internal camera-from-world poses, inverted) so the
    two directions can never disagree. ``keyframe_mapping`` and ``dense_mapping``
    preserve source timestamps for cross-module timeline joins. ``disparities`` is
    the DROID depth gauge (``1/disparity``; up-to-scale metric depth requires the
    shared camera-scale step). ``intrinsics_px`` carries the model-grid intrinsics
    actually used; ``intrinsics_provenance`` records the pixel-transform chain.
    """

    ownership: Ownership
    session_id: str
    T_world_camera: TensorPayload
    T_camera_world: TensorPayload
    intrinsics_px: TensorPayload
    disparities: TensorPayload
    keyframe_mapping: tuple[KeyframeSourceMapping, ...]
    dense_mapping: tuple[DenseSourceMapping, ...]
    uncertainty: DroidUncertainty
    model_revision: str
    trace: DroidBatchTrace
    # Optional server-attested runtime policy. Omitted from the default multipart
    # contract so legacy response metadata remains byte-for-byte stable.
    batch_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        import numpy as np

        tensors = {
            "T_world_camera": self.T_world_camera,
            "T_camera_world": self.T_camera_world,
            "intrinsics_px": self.intrinsics_px,
            "disparities": self.disparities,
        }
        arrays: dict[str, Any] = {}
        for name, tensor in tensors.items():
            if not isinstance(tensor.data, (bytes, bytearray, memoryview)):
                raise ContractValidationError(f"CameraState.{name} must contain materialized binary data")
            try:
                dtype = np.dtype(tensor.dtype)
            except TypeError as exc:
                raise ContractValidationError(f"CameraState.{name} has invalid dtype") from exc
            expected_bytes = int(np.prod(tensor.shape)) * dtype.itemsize
            if len(bytes(tensor.data)) != expected_bytes:
                raise ContractValidationError(f"CameraState.{name} byte length does not match shape and dtype")
            arrays[name] = np.frombuffer(bytes(tensor.data), dtype=dtype).reshape(tensor.shape)
        dense_count = len(self.dense_mapping)
        keyframe_count = len(self.keyframe_mapping)
        if self.T_world_camera.shape != (dense_count, 4, 4):
            raise ContractValidationError("T_world_camera shape must be [dense_mapping,4,4]")
        if self.T_camera_world.shape != self.T_world_camera.shape:
            raise ContractValidationError("T_camera_world shape must match T_world_camera")
        if self.intrinsics_px.shape != (keyframe_count, 3, 3):
            raise ContractValidationError("intrinsics_px shape must be [keyframe_mapping,3,3]")
        if len(self.disparities.shape) != 3 or self.disparities.shape[0] != keyframe_count:
            raise ContractValidationError("disparities shape must be [keyframe_mapping,H,W]")
        if dense_count == 0 or keyframe_count == 0:
            raise ContractValidationError("CameraState mappings must be non-empty")
        if [m.dense_index for m in self.dense_mapping] != list(range(dense_count)):
            raise ContractValidationError("dense mapping indices must be contiguous")
        if [m.keyframe_index for m in self.keyframe_mapping] != list(range(keyframe_count)):
            raise ContractValidationError("keyframe mapping indices must be contiguous")
        for name, mapping in (("dense", self.dense_mapping), ("keyframe", self.keyframe_mapping)):
            timestamps = [m.source_timestamp_s for m in mapping]
            if not all(np.isfinite(value) for value in timestamps):
                raise ContractValidationError(f"{name} mapping timestamps must be finite")
            if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
                raise ContractValidationError(f"{name} mapping timestamps must be nondecreasing")
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise ContractValidationError(f"CameraState.{name} must contain only finite values")
        identity = np.eye(4, dtype=np.float64)
        if not np.allclose(
            arrays["T_world_camera"] @ arrays["T_camera_world"], identity,
            rtol=1e-5, atol=1e-6,
        ):
            raise ContractValidationError("T_world_camera and T_camera_world must be mutual inverses")
        if self.trace.session_ids != (self.session_id,):
            raise ContractValidationError("finalize trace must name exactly the CameraState session")
        _required_text(self.model_revision, "model_revision")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "session_id": self.session_id,
            "T_world_camera": self.T_world_camera.to_wire(),
            "T_camera_world": self.T_camera_world.to_wire(),
            "intrinsics_px": self.intrinsics_px.to_wire(),
            "disparities": self.disparities.to_wire(),
            "keyframe_mapping": [m.to_wire() for m in self.keyframe_mapping],
            "dense_mapping": [m.to_wire() for m in self.dense_mapping],
            "uncertainty": self.uncertainty.to_wire(),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
        }

    @classmethod
    def from_wire(
        cls,
        payload: Mapping[str, Any],
        arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]] | None = None,
    ) -> "CameraState":
        def tensor(name: str) -> TensorPayload:
            descriptor = payload.get(name)
            if not isinstance(descriptor, Mapping):
                raise ContractValidationError(f"camera_state.{name} descriptor is required")
            if arrays is None:
                return TensorPayload.from_wire(descriptor)
            part_name = str(descriptor.get("part", name))
            if part_name != name or name not in arrays:
                raise ContractValidationError(f"camera_state.{name} binary part is missing or misnamed")
            data, shape, dtype = arrays[name]
            declared_shape = descriptor.get("shape")
            declared_dtype = descriptor.get("dtype")
            if declared_shape is None or tuple(declared_shape) != tuple(shape):
                raise ContractValidationError(f"camera_state.{name} descriptor shape does not match binary part")
            if declared_dtype is None or str(declared_dtype) != dtype:
                raise ContractValidationError(f"camera_state.{name} descriptor dtype does not match binary part")
            return TensorPayload(data=data, shape=tuple(shape), dtype=dtype)

        expected_parts = {"T_world_camera", "T_camera_world", "intrinsics_px", "disparities"}
        if arrays is not None and set(arrays) != expected_parts:
            raise ContractValidationError(
                f"camera-state multipart parts must be exactly {sorted(expected_parts)}"
            )
        return cls(
            ownership=Ownership.from_mapping(payload.get("ownership", {})),
            session_id=_required_text(payload.get("session_id"), "session_id"),
            T_world_camera=tensor("T_world_camera"),
            T_camera_world=tensor("T_camera_world"),
            intrinsics_px=tensor("intrinsics_px"),
            disparities=tensor("disparities"),
            keyframe_mapping=tuple(
                KeyframeSourceMapping.from_wire(item) for item in payload.get("keyframe_mapping", ())
            ),
            dense_mapping=tuple(
                DenseSourceMapping.from_wire(item) for item in payload.get("dense_mapping", ())
            ),
            uncertainty=DroidUncertainty.from_wire(payload.get("uncertainty", {})),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            trace=DroidBatchTrace.from_wire(payload.get("trace", {})),
            batch_diagnostics=(dict(payload["batch_diagnostics"]) if isinstance(payload.get("batch_diagnostics"), Mapping) else None),
        )


@dataclass(frozen=True)
class DroidCreateSessionResponse:
    ownership: Ownership
    session_id: str | None = None
    error: ServiceError | None = None
    server_identity: ServerIdentity | None = None
    batch_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.session_id is None) == (self.error is None):
            raise ContractValidationError("a create-session response must contain exactly one session_id or error")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "session_id": self.session_id,
            "error": self.error.to_wire() if self.error else None,
            "server_identity": self.server_identity.to_wire() if self.server_identity else None,
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidCreateSessionResponse":
        identity = payload.get("server_identity")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            session_id=payload.get("session_id"),
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
            server_identity=ServerIdentity.from_wire(identity) if isinstance(identity, Mapping) else None,
            batch_diagnostics=(dict(payload["batch_diagnostics"]) if isinstance(payload.get("batch_diagnostics"), Mapping) else None),
        )


@dataclass(frozen=True)
class DroidFrameResponse:
    ownership: Ownership
    status: StepStatus | None = None
    error: ServiceError | None = None
    server_identity: ServerIdentity | None = None
    batch_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.status is None) == (self.error is None):
            raise ContractValidationError("a push-frame response must contain exactly one status or error")
        if self.status is not None and self.status.ownership != self.ownership:
            raise ContractValidationError("status ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "status": self.status.to_wire() if self.status else None,
            "error": self.error.to_wire() if self.error else None,
            "server_identity": self.server_identity.to_wire() if self.server_identity else None,
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DroidFrameResponse":
        identity = payload.get("server_identity")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            status=StepStatus.from_wire(payload["status"]) if payload.get("status") else None,
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
            server_identity=ServerIdentity.from_wire(identity) if isinstance(identity, Mapping) else None,
            batch_diagnostics=(dict(payload["batch_diagnostics"]) if isinstance(payload.get("batch_diagnostics"), Mapping) else None),
        )


@dataclass(frozen=True)
class DroidFinalizeResponse:
    ownership: Ownership
    camera_state: CameraState | None = None
    error: ServiceError | None = None
    server_identity: ServerIdentity | None = None
    # This is server lifecycle evidence, not a client inference from an error code.
    # A sticky client may retire its endpoint affinity only after this field is true.
    terminal: bool = False
    batch_diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.camera_state is None) == (self.error is None):
            raise ContractValidationError("a finalize response must contain exactly one camera_state or error")
        if self.camera_state is not None and self.camera_state.ownership != self.ownership:
            raise ContractValidationError("camera_state ownership must equal response ownership")
        if self.error is not None and self.error.ownership not in {None, self.ownership}:
            raise ContractValidationError("error ownership must equal response ownership")
        # A camera state is itself terminal evidence. Normalize direct in-process
        # construction so every serialized successful response carries it too.
        if self.camera_state is not None and not self.terminal:
            object.__setattr__(self, "terminal", True)
        if self.terminal and self.error is not None and self.error.retryable:
            raise ContractValidationError("a terminal finalize error cannot be retryable")

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "camera_state": self.camera_state.to_wire() if self.camera_state else None,
            "error": self.error.to_wire() if self.error else None,
            "server_identity": self.server_identity.to_wire() if self.server_identity else None,
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
            "terminal": self.terminal,
        }

    @classmethod
    def from_wire(
        cls,
        payload: Mapping[str, Any],
        arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]] | None = None,
    ) -> "DroidFinalizeResponse":
        identity = payload.get("server_identity")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            camera_state=(
                CameraState.from_wire(payload["camera_state"], arrays)
                if payload.get("camera_state") else None
            ),
            error=ServiceError.from_wire(payload["error"]) if payload.get("error") else None,
            server_identity=ServerIdentity.from_wire(identity) if isinstance(identity, Mapping) else None,
            batch_diagnostics=(dict(payload["batch_diagnostics"]) if isinstance(payload.get("batch_diagnostics"), Mapping) else None),
            terminal=bool(payload.get("terminal", False)),
        )
