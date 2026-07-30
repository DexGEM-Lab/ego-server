"""Transport-neutral contracts for the scripted algorithm seam."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Generic, Mapping, Protocol, Sequence, TypeVar


class ContractError(ValueError):
    """Raised when an algorithm envelope violates its typed boundary."""


def _json_ready(value: object) -> object:
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _json_ready(value.to_mapping())
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ContractError(f"value of type {type(value).__name__} is not envelope-serializable")


@dataclass(frozen=True)
class FrameTimelineMetadata:
    source_id: str
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    source_sha256: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    fps: float | None = None
    timeline_mode: str = "dense"

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ContractError("source_id is required")
        if "://" in self.source_id:
            raise ContractError("source_id must identify a local source, not a remote location")
        if not self.frame_indices:
            raise ContractError("frame timeline cannot be empty")
        if self.timeline_mode not in {"dense", "droid_sampled"}:
            raise ContractError("timeline_mode must be dense or droid_sampled")
        if self.timeline_mode == "dense" and tuple(self.frame_indices) != tuple(range(self.frame_indices[0], self.frame_indices[0] + len(self.frame_indices))):
            raise ContractError("frame_indices must be contiguous")
        if self.timeline_mode == "droid_sampled" and any(left >= right for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise ContractError("DROID sampled frame_indices must be strictly increasing")
        if len(self.frame_indices) != len(self.timestamps_s):
            raise ContractError("frame_indices and timestamps_s must have equal length")
        if any(float(a) >= float(b) for a, b in zip(self.timestamps_s, self.timestamps_s[1:])):
            raise ContractError("timestamps_s must be strictly increasing")
        if self.width_px is not None and self.width_px <= 0:
            raise ContractError("width_px must be positive")
        if self.height_px is not None and self.height_px <= 0:
            raise ContractError("height_px must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ContractError("fps must be positive")

    def to_mapping(self) -> dict[str, object]:
        value = asdict(self)
        if self.timeline_mode == "dense":
            value.pop("timeline_mode")
        return _json_ready(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class StageMetadata:
    stage_id: str
    owner: str
    ownership_scope: str
    model_revision: str

    def __post_init__(self) -> None:
        if not all((self.stage_id, self.owner, self.ownership_scope, self.model_revision)):
            raise ContractError("stage, ownership, and model revision metadata are required")

    def to_mapping(self) -> dict[str, object]:
        return _json_ready(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class NativeWorkDescription:
    work_unit_type: str
    compatibility_key: str
    native_batch_axis: int | None
    native_batch_size: int
    native_batch_cap: int
    native_shape: tuple[int, ...]
    chunk_length: int | None = None
    temporal_window: int | None = None
    outer_item_batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.work_unit_type or not self.compatibility_key:
            raise ContractError("native work unit and compatibility key are required")
        if self.native_batch_size <= 0 or self.native_batch_cap <= 0:
            raise ContractError("native batch size and cap must be positive")
        if self.native_batch_size > self.native_batch_cap:
            raise ContractError("native batch size exceeds native batch cap")
        if not self.native_shape or any(int(dim) <= 0 for dim in self.native_shape):
            raise ContractError("native_shape must contain positive dimensions")
        if self.native_batch_axis is not None:
            if self.native_batch_axis < 0 or self.native_batch_axis >= len(self.native_shape):
                raise ContractError("native_batch_axis is outside native_shape")
            if self.native_shape[self.native_batch_axis] != self.native_batch_size:
                raise ContractError("native_shape batch axis disagrees with native_batch_size")
        if self.chunk_length is not None and self.chunk_length <= 0:
            raise ContractError("chunk_length must be positive")
        if self.temporal_window is not None and self.temporal_window <= 0:
            raise ContractError("temporal_window must be positive")
        if self.outer_item_batch_size != 1:
            raise ContractError("this entrypoint accepts exactly one outer item")

    def to_mapping(self) -> dict[str, object]:
        return _json_ready(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class NativeBatchTrace:
    work_unit_type: str
    compatibility_key: str
    native_shape: tuple[int, ...]
    native_batch_axis: int | None
    native_batch_size: int
    native_batch_cap: int
    execution_units: int

    def __post_init__(self) -> None:
        if self.execution_units != 1:
            raise ContractError("one request must remain one native work unit")
        if self.native_batch_size <= 0 or self.native_batch_size > self.native_batch_cap:
            raise ContractError("invalid native batch trace")
        if self.native_batch_axis is not None and self.native_shape[self.native_batch_axis] != self.native_batch_size:
            raise ContractError("native batch trace shape mismatch")

    @classmethod
    def from_work(cls, work: NativeWorkDescription) -> "NativeBatchTrace":
        return cls(
            work_unit_type=work.work_unit_type,
            compatibility_key=work.compatibility_key,
            native_shape=work.native_shape,
            native_batch_axis=work.native_batch_axis,
            native_batch_size=work.native_batch_size,
            native_batch_cap=work.native_batch_cap,
            execution_units=1,
        )

    def to_mapping(self) -> dict[str, object]:
        return _json_ready(asdict(self))  # type: ignore[return-value]


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class AlgorithmRequest(Generic[TInput]):
    algorithm_id: str
    model_revision: str
    case_id: str
    item_id: str
    source_id: str
    timeline: FrameTimelineMetadata
    stage: StageMetadata
    work: NativeWorkDescription
    input: TInput
    options: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.algorithm_id, self.model_revision, self.case_id, self.item_id, self.source_id)):
            raise ContractError("algorithm identity fields are required")
        if "://" in self.source_id:
            raise ContractError("source_id must identify a local source, not a remote location")
        if self.source_id != self.timeline.source_id:
            raise ContractError("source_id must match timeline source_id")
        if self.timeline.timeline_mode == "droid_sampled" and self.algorithm_id not in {"droid.create_session", "droid.finalize"}:
            raise ContractError("DROID sampled timelines are only valid for DROID create/finalize requests")
        if self.stage.model_revision != self.model_revision:
            raise ContractError("stage model revision must match request model revision")
        if not isinstance(self.options, Mapping):
            raise ContractError("options must be a mapping")
        forbidden = ("url", "endpoint", "route", "transport", "host", "port")
        for key, value in self.options.items():
            if any(token in str(key).lower() for token in forbidden):
                raise ContractError(f"transport detail is not allowed in algorithm options: {key}")
            if not isinstance(value, (str, int, float, bool)):
                raise ContractError(f"option {key!r} is not scalar")

    def to_mapping(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "model_revision": self.model_revision,
            "case_id": self.case_id,
            "item_id": self.item_id,
            "source_id": self.source_id,
            "timeline": self.timeline.to_mapping(),
            "stage": self.stage.to_mapping(),
            "work": self.work.to_mapping(),
            "input": _json_ready(self.input),
            "options": _json_ready(dict(self.options)),
        }


@dataclass(frozen=True)
class ClientRequestTiming:
    """Client-observed timing for one request; transport includes all HTTP wait."""

    client_prepare_s: float = 0.0
    transport_wait_s: float = 0.0
    client_decode_postprocess_s: float = 0.0
    total_wall_s: float = 0.0
    available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("client_prepare_s", "transport_wait_s", "client_decode_postprocess_s", "total_wall_s"):
            value = float(getattr(self, name))
            if value < 0:
                raise ContractError(f"{name} must be non-negative")
        if self.available and self.unavailable_reason is not None:
            raise ContractError("available timing cannot have an unavailable reason")
        if not self.available and not self.unavailable_reason:
            raise ContractError("unavailable timing requires an explicit reason")

    def to_mapping(self) -> dict[str, object]:
        return _json_ready(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True)
class AlgorithmResult(Generic[TOutput]):
    algorithm_id: str
    model_revision: str
    case_id: str
    item_id: str
    source_id: str
    timeline: FrameTimelineMetadata
    output: TOutput
    uncertainty: Mapping[str, object]
    visibility: Mapping[str, object]
    native_batch_trace: NativeBatchTrace
    provenance: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not all((self.algorithm_id, self.model_revision, self.case_id, self.item_id, self.source_id)):
            raise ContractError("result identity fields are required")
        if self.source_id != self.timeline.source_id:
            raise ContractError("result source_id must match timeline source_id")

    @classmethod
    def from_request(
        cls,
        request: AlgorithmRequest[TInput],
        *,
        output: TOutput,
        uncertainty: Mapping[str, object] | None = None,
        visibility: Mapping[str, object] | None = None,
        native_batch_trace: NativeBatchTrace | None = None,
        provenance: Sequence[Mapping[str, object]] = (),
    ) -> "AlgorithmResult[TOutput]":
        return cls(
            algorithm_id=request.algorithm_id,
            model_revision=request.model_revision,
            case_id=request.case_id,
            item_id=request.item_id,
            source_id=request.source_id,
            timeline=request.timeline,
            output=output,
            uncertainty=dict(uncertainty or {}),
            visibility=dict(visibility or {}),
            native_batch_trace=native_batch_trace or NativeBatchTrace.from_work(request.work),
            provenance=tuple(dict(row) for row in provenance),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "model_revision": self.model_revision,
            "case_id": self.case_id,
            "item_id": self.item_id,
            "source_id": self.source_id,
            "timeline": self.timeline.to_mapping(),
            "output": _json_ready(self.output),
            "uncertainty": _json_ready(dict(self.uncertainty)),
            "visibility": _json_ready(dict(self.visibility)),
            "native_batch_trace": self.native_batch_trace.to_mapping(),
            "provenance": _json_ready(list(self.provenance)),
        }


class ScriptBackend(Protocol):
    """Stable execution seam implemented by a scripted backend."""

    def execute(self, request: AlgorithmRequest[TInput]) -> AlgorithmResult[TOutput]:
        ...


class ApiBackend(ScriptBackend, Protocol):
    """Future transport implementation with the same typed method."""

    pass


__all__ = [
    "AlgorithmRequest",
    "AlgorithmResult",
    "ApiBackend",
    "ContractError",
    "FrameTimelineMetadata",
    "NativeBatchTrace",
    "NativeWorkDescription",
    "ScriptBackend",
    "StageMetadata",
]
