"""Annotation job runner for ego.annotation.output alpha bundles."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ego_annotation.artifacts import ArtifactBundle, status_from_errors
from ego_annotation.calibration import resolve_calibration
from ego_annotation.metrics import build_metric_rows, throughput_forecast
from ego_annotation.models import AnnotationJobRequest, JobResult
from ego_annotation.schema import SCHEMA_NAME, SCHEMA_VERSION


def _path_sha256(uri: str) -> str | None:
    if "://" in uri:
        return None
    path = Path(uri).expanduser()
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_rows(request: AnnotationJobRequest) -> list[dict[str, Any]]:
    frame_count = request.media.frame_count
    fps = request.media.fps
    rows: list[dict[str, Any]] = []
    if frame_count is None or frame_count <= 0:
        duration = request.media.completed_duration_s()
        return [
            {
                "frame_idx": None,
                "time_s": None,
                "video_uri": request.video_uri,
                "status": "frame_timeline_unresolved",
                "frame_count": frame_count,
                "fps": fps,
                "duration_s": duration,
            }
        ]
    for idx in range(frame_count):
        rows.append(
            {
                "frame_idx": idx,
                "time_s": float(idx / fps) if fps and fps > 0 else None,
                "video_uri": request.video_uri,
                "width": request.media.width,
                "height": request.media.height,
                "status": "declared_from_request_metadata",
            }
        )
    return rows


def _state_rows(request: AnnotationJobRequest, key: str, expected_table: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    value = request.state_inputs.get(key)
    errors: list[dict[str, Any]] = []
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)], errors
    if value is None:
        errors.append(
            {
                "code": f"{expected_table}_unavailable",
                "severity": "error",
                "message": f"No {expected_table} numeric state was supplied or produced by the alpha runner.",
                "mechanism": "The product bundle preserves the missing physical estimate as an explicit error; downstream consumers must not infer it from renders or schema presence.",
            }
        )
        return [], errors
    errors.append(
        {
            "code": f"{expected_table}_invalid_input",
            "severity": "error",
            "message": f"state_inputs.{key} must be a list of row objects.",
            "mechanism": "Numeric state tables are frame-aligned rows, not opaque blobs.",
        }
    )
    return [], errors


def _semantic_rows(request: AnnotationJobRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    caption_events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    duration = request.media.completed_duration_s()
    for i, src in enumerate(request.semantic_sources):
        start = src.get("start_s", src.get("start"))
        end = src.get("end_s", src.get("end"))
        caption = src.get("caption") or src.get("text") or src.get("action")
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            errors.append(
                {
                    "code": "semantic_clip_invalid_time",
                    "severity": "error",
                    "message": f"semantic source {i} lacks numeric start_s/end_s",
                    "mechanism": "Semantic clips must cover a physical time interval and cannot be anonymous captions.",
                }
            )
            continue
        if not caption or end_f <= start_f:
            errors.append(
                {
                    "code": "semantic_clip_invalid_caption_or_duration",
                    "severity": "error",
                    "message": f"semantic source {i} has empty caption or non-positive duration",
                    "mechanism": "Caption lane needs grounded time segments, not detached text.",
                }
            )
            continue
        evidence_frames = src.get("evidence_frames") if isinstance(src.get("evidence_frames"), list) else []
        row = {
            "clip_id": str(src.get("clip_id") or f"semantic_{i:05d}"),
            "start_s": start_f,
            "end_s": end_f,
            "duration_s": end_f - start_f,
            "caption": str(caption),
            "confidence": float(src.get("confidence", 0.0)),
            "source": str(src.get("source") or "request_semantic_source"),
            "grounding_status": str(src.get("grounding_status") or ("grounded" if evidence_frames else "unverified")),
            "evidence_frames": evidence_frames,
        }
        rows.append(row)
        caption_events.append({"event": "semantic_clip_ingested", **row})
    if duration and duration > 0 and rows:
        covered = sum(max(0.0, min(duration, r["end_s"]) - max(0.0, r["start_s"])) for r in rows)
        if covered < duration * 0.95:
            errors.append(
                {
                    "code": "semantic_timeline_incomplete",
                    "severity": "degraded",
                    "message": f"Semantic clips cover approximately {covered:.3f}s of {duration:.3f}s.",
                    "mechanism": "Caption alpha requires full-timeline coverage; partial source captions are preserved but flagged.",
                }
            )
    elif not rows:
        errors.append(
            {
                "code": "semantic_clips_unavailable",
                "severity": "error",
                "message": "No semantic clips or captions were supplied or produced by the alpha runner.",
                "mechanism": "Captioning lane is required for annotation output; missing captions remain an explicit error.",
            }
        )
    return rows, caption_events, errors


def _overlay_events(head_rows: list[dict[str, Any]], hand_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if head_rows:
        events.append({"event": "head_camera_numeric_state_available", "rows": len(head_rows)})
    if hand_rows:
        events.append({"event": "hand_numeric_state_available", "rows": len(hand_rows)})
    if semantic_rows:
        events.append({"event": "semantic_clips_available", "rows": len(semantic_rows)})
    return events


class AnnotationJobRunner:
    """Runs the annotation alpha without invoking local heavy model inference."""

    def run(self, request: AnnotationJobRequest | dict[str, Any]) -> JobResult:
        if isinstance(request, dict):
            request = AnnotationJobRequest.from_mapping(request)
        bundle = ArtifactBundle(request.output_root, request.job_id)
        bundle.add_provenance(
            "annotation_job_runner",
            "job_started",
            schema=SCHEMA_NAME,
            schema_version=SCHEMA_VERSION,
            video_uri=request.video_uri,
        )

        digest = request.media.sha256 or _path_sha256(request.video_uri)
        if digest is None:
            bundle.add_error(
                "input_hash_unavailable",
                "degraded",
                "Input SHA256 was not supplied and could not be computed from a local file path.",
                "Remote/object-store URIs need caller-provided content hash or an ingress downloader stage for reproducible provenance.",
            )
        else:
            bundle.add_provenance("ingestion", "input_hash_resolved", sha256=digest)

        calib = resolve_calibration(
            request.calibration,
            media_width=request.media.width,
            media_height=request.media.height,
            allow_estimated=request.allow_estimated_calibration,
        )
        bundle.errors.extend(calib.errors)
        bundle.provenance.extend(calib.provenance)

        frames = _frame_rows(request)
        if frames and frames[0].get("status") == "frame_timeline_unresolved":
            bundle.add_error(
                "frame_timeline_unresolved",
                "error",
                "Frame count/FPS metadata was not supplied; alpha runner did not decode video locally.",
                "The API contract needs frame-aligned outputs; production ingress should run raw-frame manifest extraction on approved compute.",
            )

        head_rows, head_errors = _state_rows(request, "head_camera", "head_camera")
        hand_rows, hand_errors = _state_rows(request, "hand_states", "hand_states")
        for err in head_errors + hand_errors:
            bundle.errors.append(err)

        semantic_rows, caption_events, semantic_errors = _semantic_rows(request)
        bundle.errors.extend(semantic_errors)

        metric_rows = build_metric_rows(
            request.metric_observations,
            request.throughput_observations,
            calibration_status=calib.status,
        )
        forecast = throughput_forecast(request.throughput_observations)
        if forecast["status"] != "measured":
            bundle.add_error(
                "throughput_measurements_unavailable",
                "error",
                "No module-speed benchmark rows were supplied or produced.",
                "The 10,000 video-hours/week forecast requires module_speed_x and GPU-hours/video-hour measurements.",
            )

        initial_status = status_from_errors(bundle.errors)
        manifest_path = bundle.write(
            request={
                "job_id": request.job_id,
                "video_uri": request.video_uri,
                "media": request.media.__dict__,
                "public_endpoint": "/v1/annotation-jobs",
            },
            calibration_contract=calib.to_contract(),
            tables={
                "frames": frames,
                "head_camera": head_rows,
                "hand_states": hand_rows,
                "semantic_clips": semantic_rows,
                "validation_metrics": metric_rows,
            },
            events={
                "overlay_events": _overlay_events(head_rows, hand_rows, semantic_rows),
                "caption_events": caption_events,
            },
            throughput_forecast=forecast,
            status=initial_status,
        )
        final_status = status_from_errors(bundle.errors)
        return JobResult(
            job_id=request.job_id,
            status=final_status,
            artifact_root=manifest_path.parent,
            manifest_path=manifest_path,
            errors=list(bundle.errors),
            metrics=metric_rows,
            throughput_forecast=forecast,
        )
