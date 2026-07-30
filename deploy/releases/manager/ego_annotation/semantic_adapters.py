"""Render validated Cosmos semantic rows over the complete source timeline."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ego_annotation.cosmos_semantics import validate_semantic_coverage


class SemanticAdapterError(RuntimeError):
    """Semantic rows cannot be rendered as a truthful full-duration artifact."""


class SourceLike(Protocol):
    timeline: object

    def read_rgb(self, frame_index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticArtifacts:
    subtitle_video: str
    report_json: str
    frame_count: int
    duration_s: float


def caption_for_frame(rows: Sequence[Mapping[str, object]], frame_index: int, cursor: int) -> tuple[str, int]:
    while cursor + 1 < len(rows) and frame_index >= int(rows[cursor]["end_frame"]):
        cursor += 1
    row = rows[cursor]
    if not int(row["start_frame"]) <= frame_index < int(row["end_frame"]):
        raise SemanticAdapterError(f"frame {frame_index} has no semantic row")
    return str(row["caption"]), cursor


def semantic_row_anomalies(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = row.get("semantic_anomalies")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _wrapped_lines(text: str, max_chars: int = 72) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def draw_semantic_caption(
    bgr: np.ndarray,
    caption: str,
    anomalies: Sequence[Mapping[str, object]] = (),
) -> None:
    import cv2

    height, width = bgr.shape[:2]
    lines = _wrapped_lines(caption)
    if anomalies:
        raw = ", ".join(f"{item.get('raw_field')}={item.get('raw_value')}" for item in anomalies)
        lines.append(f"SEMANTIC ENUM ANOMALY -> unknown ({raw})")
    panel_height = max(48, 22 + len(lines) * 28)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, bgr, 0.28, 0.0, bgr)
    for line_index, line in enumerate(lines):
        color = (0, 215, 255) if anomalies and line_index == len(lines) - 1 else (255, 255, 255)
        cv2.putText(bgr, line, (18, height - panel_height + 34 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class SemanticArtifactAdapter:
    def render(self, rows: Sequence[Mapping[str, object]], source: SourceLike, run_root: Path) -> SemanticArtifacts:
        import cv2

        timeline = source.timeline
        frame_count = int(getattr(timeline, "frame_count"))
        fps = float(getattr(timeline, "fps"))
        width = int(getattr(timeline, "width_px"))
        height = int(getattr(timeline, "height_px"))
        validate_semantic_coverage(rows, frame_count)
        renders = run_root / "renders"
        renders.mkdir(parents=True, exist_ok=True)
        output = renders / "v22_semantic_subtitle.mp4"
        temporary = renders / f".{output.name}.{os.getpid()}.tmp.mp4"
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SemanticAdapterError("cannot open semantic subtitle video writer")
        cursor = 0
        rendered_anomaly_frames = 0
        anomaly_count = sum(len(semantic_row_anomalies(row)) for row in rows)
        try:
            try:
                for frame_index in range(frame_count):
                    rgb = np.asarray(source.read_rgb(frame_index))
                    if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
                        raise SemanticAdapterError("source frame changed shape/type during semantic render")
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    caption, cursor = caption_for_frame(rows, frame_index, cursor)
                    anomalies = semantic_row_anomalies(rows[cursor])
                    if anomalies:
                        rendered_anomaly_frames += 1
                    draw_semantic_caption(bgr, caption, anomalies)
                    writer.write(bgr)
            finally:
                writer.release()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise SemanticAdapterError("semantic subtitle writer produced no video")
        os.replace(temporary, output)
        report = {
            "schema": "v22_semantic_subtitle_report.v1",
            "status": "completed_with_anomalies" if anomaly_count else "ok",
            "video": str(output),
            "frame_count": frame_count,
            "fps": fps,
            "duration_s": frame_count / fps,
            "semantic_row_count": len(rows),
            "anomaly_count": anomaly_count,
            "anomaly_annotated_frame_count": rendered_anomaly_frames,
            "coverage": {"start_frame": 0, "end_frame": frame_count, "fraction": 1.0},
            "claim_scope": "semantic_only_not_physical_evidence",
            "render_source": "validated Cosmos semantic rows over immutable source frames",
        }
        report_path = renders / "v22_semantic_subtitle_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        return SemanticArtifacts(str(output), str(report_path), frame_count, frame_count / fps)


__all__ = [
    "SemanticAdapterError",
    "SemanticArtifactAdapter",
    "SemanticArtifacts",
    "caption_for_frame",
    "draw_semantic_caption",
    "semantic_row_anomalies",
]
