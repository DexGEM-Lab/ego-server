"""Dependency-light request/result models for annotation jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


def _as_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _as_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return list(value)


@dataclass(frozen=True)
class MediaInfo:
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    sha256: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "MediaInfo":
        data = _as_dict(payload, "media")
        frame_count = data.get("frame_count")
        fps = data.get("fps")
        width = data.get("width")
        height = data.get("height")
        duration_s = data.get("duration_s")
        return cls(
            frame_count=int(frame_count) if frame_count is not None else None,
            fps=float(fps) if fps is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            duration_s=float(duration_s) if duration_s is not None else None,
            sha256=str(data["sha256"]) if data.get("sha256") else None,
        )

    def completed_duration_s(self) -> float | None:
        if self.duration_s is not None and self.duration_s >= 0:
            return float(self.duration_s)
        if self.frame_count is not None and self.fps is not None and self.fps > 0:
            return float(self.frame_count) / float(self.fps)
        return None


@dataclass(frozen=True)
class AnnotationJobRequest:
    video_uri: str
    output_root: Path
    job_id: str = field(default_factory=lambda: uuid4().hex)
    media: MediaInfo = field(default_factory=MediaInfo)
    calibration: dict[str, Any] = field(default_factory=dict)
    state_inputs: dict[str, Any] = field(default_factory=dict)
    semantic_sources: list[dict[str, Any]] = field(default_factory=list)
    metric_observations: dict[str, Any] = field(default_factory=dict)
    throughput_observations: list[dict[str, Any]] = field(default_factory=list)
    render_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_estimated_calibration: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AnnotationJobRequest":
        if not isinstance(payload, dict):
            raise ValueError("annotation job request must be an object")
        video_uri = payload.get("video_uri") or payload.get("video")
        if not video_uri:
            raise ValueError("video_uri is required")
        output_root_raw = payload.get("output_root") or payload.get("artifact_root")
        if not output_root_raw:
            raise ValueError("output_root is required for the alpha runner")
        semantic_sources = payload.get("semantic_sources")
        if semantic_sources is None:
            semantic_sources = payload.get("captions")
        return cls(
            video_uri=str(video_uri),
            output_root=Path(str(output_root_raw)),
            job_id=str(payload.get("job_id") or uuid4().hex),
            media=MediaInfo.from_mapping(payload.get("media")),
            calibration=_as_dict(payload.get("calibration"), "calibration"),
            state_inputs=_as_dict(payload.get("state_inputs"), "state_inputs"),
            semantic_sources=[dict(x) for x in _as_list(semantic_sources, "semantic_sources") if isinstance(x, dict)],
            metric_observations=_as_dict(payload.get("metric_observations"), "metric_observations"),
            throughput_observations=[dict(x) for x in _as_list(payload.get("throughput_observations"), "throughput_observations") if isinstance(x, dict)],
            render_options=_as_dict(payload.get("render_options"), "render_options"),
            metadata=_as_dict(payload.get("metadata"), "metadata"),
            allow_estimated_calibration=bool(payload.get("allow_estimated_calibration", True)),
        )


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: str
    artifact_root: Path
    manifest_path: Path
    errors: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    throughput_forecast: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "artifact_root": str(self.artifact_root),
            "manifest_path": str(self.manifest_path),
            "error_count": len(self.errors),
            "errors": self.errors,
            "metrics_measured": sum(1 for row in self.metrics if row.get("status") == "measured"),
            "metrics_total": len(self.metrics),
            "throughput_forecast": self.throughput_forecast,
        }
