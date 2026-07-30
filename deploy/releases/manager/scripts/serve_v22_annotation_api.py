#!/usr/bin/env python3
"""FastAPI service for the minimal V22 annotation pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation_api_uploads import UploadError, clean_upload_filename, upload_destination, write_upload_stream
from ego_annotation.fps_config import DEFAULT_FPS_CONDITION, FPS_CONDITION_BY_NAME, get_fps_condition
from scripts.annotation_remote_runner import RemoteExecutionError, config_from_env, run_remote_annotation_job, run_remote_annotation_job_set
from scripts.package_v22_annotation_result import PackageError, create_result_package, resolve_download_package

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ANNOTATION_OUTPUT_ROOT", "/home/zjh/data/v22_api_jobs"))
DEFAULT_PACKAGE_ROOT = Path(os.environ.get("ANNOTATION_PACKAGE_ROOT", "/home/zjh/data/v22_api_downloads"))
PUBLIC_BASE_URL = os.environ.get("ANNOTATION_PUBLIC_BASE_URL", "").rstrip("/")
REMOTE_CONFIG = config_from_env(os.environ)
REMOTE_TIMEOUT_S = int(os.environ.get("ANNOTATION_REMOTE_TIMEOUT_S", "7200"))
ANNOTATION_RUNTIME_MODE = os.environ.get("ANNOTATION_RUNTIME_MODE", "direct_script").strip().lower()


def positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {value}")
    return value


TOTAL_REQUEST_LIMIT = positive_int_env("ANNOTATION_TOTAL_REQUEST_LIMIT", 128)
ALGORITHM_INFLIGHT_MULTIPLIER = positive_int_env("ANNOTATION_ALGORITHM_INFLIGHT_MULTIPLIER", 2)
API_REQUEST_SEMAPHORE = asyncio.Semaphore(TOTAL_REQUEST_LIMIT)
ANNOTATION_METADATA_PYTHON = os.environ.get("ANNOTATION_METADATA_PYTHON", "/home/zjh/miniconda3/bin/python")
MODEL_BACKENDS = {"script", "feishu_ray", "api_ify"}
SERVICE_ENDPOINT_KEYS = {"unidepth", "hands_wilor", "wilor", "droid", "hawor", "cosmos3"}
KNOWN_SERVICE_PROFILES = {"feishu_ray_a800_server_local"}
ANNOTATION_QUERY_FIELDS: set[str] = set()


@asynccontextmanager
async def acquire_api_request_slot() -> AsyncIterator[None]:
    await API_REQUEST_SEMAPHORE.acquire()
    try:
        yield
    finally:
        API_REQUEST_SEMAPHORE.release()


def validate_request_admission(total_request_limit: int, algorithm_inflight_multiplier: int) -> None:
    if total_request_limit != TOTAL_REQUEST_LIMIT:
        raise contract_error(
            "total_request_limit_must_match_server_capacity",
            requested=total_request_limit,
            server_capacity=TOTAL_REQUEST_LIMIT,
            reason="set the API server capacity with --total-request-limit or ANNOTATION_TOTAL_REQUEST_LIMIT",
        )
    if algorithm_inflight_multiplier <= 0:
        raise contract_error("invalid_algorithm_inflight_multiplier", value=algorithm_inflight_multiplier)


def configure_admission_limits(*, total_request_limit: int, algorithm_inflight_multiplier: int) -> None:
    global TOTAL_REQUEST_LIMIT, ALGORITHM_INFLIGHT_MULTIPLIER, API_REQUEST_SEMAPHORE
    if total_request_limit <= 0 or algorithm_inflight_multiplier <= 0:
        raise ValueError("admission limits must be positive")
    TOTAL_REQUEST_LIMIT = int(total_request_limit)
    ALGORITHM_INFLIGHT_MULTIPLIER = int(algorithm_inflight_multiplier)
    API_REQUEST_SEMAPHORE = asyncio.Semaphore(TOTAL_REQUEST_LIMIT)


def admission_summary(*, total_request_limit: int | None = None, algorithm_inflight_multiplier: int | None = None) -> dict[str, Any]:
    return {
        "total_request_limit": int(total_request_limit if total_request_limit is not None else TOTAL_REQUEST_LIMIT),
        "algorithm_inflight_multiplier": int(algorithm_inflight_multiplier if algorithm_inflight_multiplier is not None else ALGORITHM_INFLIGHT_MULTIPLIER),
        "algorithm_inflight_multiplier_policy": "compatibility_ignored_by_client_batch_scheduler",
        "admission_boundary": "api_request_semaphore_plus_client_batch_scheduler",
    }


async def run_job_without_early_slot_release(req: "AnnotationJobRequest") -> "AnnotationJobResponse":
    job_task = asyncio.ensure_future(run_in_threadpool(run_annotation_job, req))
    try:
        return await asyncio.shield(job_task)
    except asyncio.CancelledError:
        # A disconnected client must not release the slot while the worker
        # thread continues mutating the run root and consuming model capacity.
        await job_task
        raise


class AnnotationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_uri: str
    job_id: str | None = None
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    start_s: float | None = None
    end_s: float | None = None
    render_width: int | None = None
    gpu_ids: str | None = None
    run_preflight: bool = False
    run_camera_trajectory: bool = True
    run_hawor_metric_hands: bool = True
    run_hybrid_hands: bool = True
    run_gt_free_drift_self_calibration: bool = True
    run_captioning: bool = True
    run_self_consistency_qc: bool = True
    run_evaluator: bool = True
    actions_json: str | None = None
    captions_json: str | None = None
    semantic_review_json: str | None = None
    run_semantic_annotation_agent: bool = False
    head_gt: str | None = None
    hand_gt: str | None = None
    write_product_bundle: bool = True
    total_request_limit: int = Field(default=TOTAL_REQUEST_LIMIT, ge=1)
    algorithm_inflight_multiplier: int = Field(default=ALGORITHM_INFLIGHT_MULTIPLIER, ge=1)
    model_backend: str = "script"
    diagnostic_monocular: bool = False
    service_profile: str | None = None
    service_endpoints: dict[str, str] | None = None
    fps_condition: str = Field(default=DEFAULT_FPS_CONDITION, exclude=True)
    upload_info: dict[str, Any] | None = Field(default=None, exclude=True)


class AnnotationJobResponse(BaseModel):
    job_id: str
    status: str
    run_root: str
    manifest_path: str
    overlay_path: str | None = None
    depth_overlay_path: str | None = None
    product_manifest_path: str | None = None
    package_path: str | None = None
    download_url: str | None = None
    elapsed_s: float
    summary: dict[str, Any] = Field(default_factory=dict)


class AnnotationJobSetItem(BaseModel):
    video_uri: str
    item_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationJobSetRequest(BaseModel):
    items: list[AnnotationJobSetItem] | None = None
    video_uris: list[str] | None = None
    data_root: str | None = None
    job_id: str | None = None
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    max_items: int | None = None
    item_agents: int = 16
    gpu_count: int = 8
    gpu_ids: str | None = None
    prepare_workers: int = 16
    calibration_workers: int = 16
    run_camera_trajectory: bool = True
    run_metric_hands: bool = True


class AnnotationJobSetResponse(BaseModel):
    job_id: str
    status: str
    run_root: str
    manifest_path: str
    report_dir: str
    delivery_index_path: str | None = None
    download_url: str | None = None
    elapsed_s: float
    summary: dict[str, Any] = Field(default_factory=dict)
    processing_summary: dict[str, Any] = Field(default_factory=dict)


def wants_json_response(request: Request, response_format: str | None = None) -> bool:
    requested = (response_format or request.query_params.get("response_format") or "").strip().lower()
    if requested in {"json", "full_json", "debug"}:
        return True
    if requested in {"text", "plain", "short"}:
        return False
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in {"", "application/json"}:
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "*/*" not in accept and "text/plain" not in accept


def compact_error_text(detail: Any) -> str:
    if isinstance(detail, dict):
        lines = ["status=failed"]
        for key in ("code", "job_id", "run_root", "log", "error"):
            value = detail.get(key)
            if value is not None:
                text = str(value).replace("\n", " ")
                if key == "error" and len(text) > 1200:
                    text = text[-1200:]
                lines.append(f"{key}={text}")
        return "\n".join(lines) + "\n"
    return f"status=failed\nerror={str(detail)}\n"


def summarize_step_progress(summary: dict[str, Any]) -> str | None:
    steps = summary.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    total = len(steps)
    finished = sum(1 for row in steps if isinstance(row, dict) and row.get("status") == "ok")
    failed = [row.get("step") for row in steps if isinstance(row, dict) and row.get("status") not in {"ok", None}]
    if failed:
        return f"{finished}/{total} stages completed; failed_at={failed[-1]}"
    return f"{finished}/{total} stages completed"


def summarize_overlay_video(summary: dict[str, Any]) -> str | None:
    ffprobe = summary.get("ffprobe_overlay")
    if not isinstance(ffprobe, dict):
        return None
    payload = ffprobe.get("ffprobe")
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    if not isinstance(stream, dict):
        return None
    width = stream.get("width")
    height = stream.get("height")
    frames = stream.get("nb_read_frames")
    duration = stream.get("duration")
    parts = []
    if frames is not None:
        parts.append(f"{frames} frames")
    if duration is not None:
        parts.append(f"{duration}s")
    if width is not None and height is not None:
        parts.append(f"{width}x{height}")
    return ", ".join(parts) if parts else None


def format_elapsed_value(value: Any) -> str | None:
    try:
        elapsed = float(value)
    except (TypeError, ValueError):
        return None
    return f"{elapsed:.3f}s"


def summarize_parallel_group_timings(summary: dict[str, Any]) -> list[str]:
    groups = summary.get("parallel_groups")
    if not isinstance(groups, list):
        return []
    lines: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = group.get("group")
        elapsed = format_elapsed_value(group.get("elapsed_s"))
        if not name or elapsed is None:
            continue
        lanes = group.get("lanes") if isinstance(group.get("lanes"), list) else []
        lanes_text = ",".join(str(lane) for lane in lanes)
        status = str(group.get("status") or "unknown")
        suffix = f" lanes={lanes_text}" if lanes_text else ""
        lines.append(f"parallel_group_timing.{name}={elapsed} status={status}{suffix}")
    return lines


def summarize_module_timings(summary: dict[str, Any]) -> list[str]:
    steps = summary.get("steps")
    if not isinstance(steps, list):
        return []
    lines: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("step")
        elapsed = format_elapsed_value(step.get("elapsed_s"))
        if not name or elapsed is None:
            continue
        status = str(step.get("status") or "unknown")
        lines.append(f"module_timing.{name}={elapsed} status={status}")
    return lines


def format_annotation_job_text(response: AnnotationJobResponse) -> str:
    lines = [
        f"status={response.status}",
        f"job_id={response.job_id}",
    ]
    progress = summarize_step_progress(response.summary)
    if progress:
        lines.append(f"progress={progress}")
    overlay = summarize_overlay_video(response.summary)
    if overlay:
        lines.append(f"overlay={overlay}")
    lines.append(f"elapsed_s={response.elapsed_s:.1f}")
    lines.extend(summarize_parallel_group_timings(response.summary))
    lines.extend(summarize_module_timings(response.summary))
    if response.download_url:
        lines.append(f"download_command=curl --noproxy '*' -O {response.download_url}")
        lines.append(f"download_url={response.download_url}")
    else:
        lines.append("download_url=")
    return "\n".join(lines) + "\n"


def clean_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return cleaned[:96] or f"job_{uuid4().hex[:12]}"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return payload


def package_download_url(filename: str) -> str:
    path = f"/v1/downloads/{filename}"
    return f"{PUBLIC_BASE_URL}{path}" if PUBLIC_BASE_URL else path


def safe_download_path(filename: str, package_root: Path = DEFAULT_PACKAGE_ROOT) -> Path:
    path = resolve_download_package(filename, package_root)
    if path is None:
        raise HTTPException(status_code=404, detail={"code": "download_not_found"})
    return path


def contract_error(code: str, **detail: Any) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": code, **detail})


def validate_service_endpoints(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise contract_error("invalid_service_endpoints", reason="expected an object mapping service names to base URLs")
    endpoints: dict[str, str] = {}
    for raw_name, raw_url in value.items():
        if not isinstance(raw_name, str) or raw_name not in SERVICE_ENDPOINT_KEYS:
            raise contract_error(
                "invalid_service_endpoint_key",
                field=str(raw_name),
                allowed=sorted(SERVICE_ENDPOINT_KEYS),
            )
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise contract_error("invalid_service_endpoint_url", field=raw_name, reason="expected a non-empty HTTP(S) URL")
        url = raw_url.strip().rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise contract_error(
                "invalid_service_endpoint_url",
                field=raw_name,
                value=raw_url,
                reason="expected an HTTP(S) base URL without path, query, or fragment",
            )
        endpoints[raw_name] = url
    return endpoints


def validate_backend_configuration(
    *,
    model_backend: Any,
    service_profile: Any,
    service_endpoints: dict[str, str] | None,
) -> None:
    if not isinstance(model_backend, str) or model_backend not in MODEL_BACKENDS:
        raise contract_error("invalid_model_backend", value=model_backend, allowed=sorted(MODEL_BACKENDS))
    if model_backend == "script":
        if service_profile is not None or service_endpoints is not None:
            raise contract_error(
                "service_configuration_requires_feishu_ray",
                reason="service_profile/service_endpoints are ignored by the script backend; select model_backend=feishu_ray explicitly",
            )
        return
    if model_backend == "api_ify":
        if service_profile is not None or service_endpoints is not None:
            raise contract_error(
                "service_configuration_not_supported_for_api_ify",
                reason="api_ify uses the frozen fixed localhost service routes and does not accept legacy Feishu caller overrides",
            )
        return
    if service_profile is None:
        raise contract_error(
            "service_profile_required_for_feishu_ray",
            reason="select a named Feishu service profile before requesting the Feishu backend",
            allowed=sorted(KNOWN_SERVICE_PROFILES),
        )
    if not isinstance(service_profile, str) or service_profile not in KNOWN_SERVICE_PROFILES:
        raise contract_error(
            "unknown_service_profile",
            value=service_profile,
            allowed=sorted(KNOWN_SERVICE_PROFILES),
        )


def annotation_request_from_payload(
    payload: dict[str, Any],
    *,
    multipart: bool,
    authoritative_video_uri: str | None = None,
) -> AnnotationJobRequest:
    if multipart and "video_uri" in payload:
        raise contract_error(
            "multipart_video_uri_forbidden",
            field="video_uri",
            reason="the multipart file field is authoritative",
        )
    if multipart:
        if authoritative_video_uri is None:
            raise RuntimeError("multipart request validation requires the uploaded destination")
        payload = {**payload, "video_uri": authoritative_video_uri}
    unknown = sorted(set(payload) - set(AnnotationJobRequest.model_fields))
    if unknown:
        raise contract_error("unknown_annotation_request_fields", fields=unknown)
    backend = payload.get("model_backend", "script")
    endpoints = validate_service_endpoints(payload.get("service_endpoints"))
    if "service_profile" in payload and payload["service_profile"] is not None:
        profile = payload["service_profile"]
        if not isinstance(profile, str) or not profile.strip():
            raise contract_error("invalid_service_profile", reason="expected a non-empty string or null")
        payload = {**payload, "service_profile": profile.strip()}
    if endpoints is not None:
        payload = {**payload, "service_endpoints": endpoints}
    validate_backend_configuration(
        model_backend=backend,
        service_profile=payload.get("service_profile"),
        service_endpoints=endpoints,
    )
    try:
        return AnnotationJobRequest(**payload)
    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in row.get("loc", ())),
                "message": row.get("msg", "invalid value"),
                "type": row.get("type", "value_error"),
            }
            for row in exc.errors(include_url=False)
        ]
        raise contract_error("invalid_annotation_request", errors=errors) from exc


def parse_query_service_endpoints(value: str | None) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise contract_error(
            "invalid_service_endpoints_json",
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        ) from exc
    if not isinstance(payload, dict):
        raise contract_error("invalid_service_endpoints", reason="query service_endpoints must encode a JSON object")
    return payload


def validate_annotation_query_fields(request: Request) -> None:
    unknown = sorted(set(request.query_params) - ANNOTATION_QUERY_FIELDS)
    if unknown:
        raise contract_error("unknown_annotation_query_fields", fields=unknown)


def parse_multipart_request_json(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        raise contract_error("invalid_multipart_request_json", reason="request must be a JSON string form field")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise contract_error(
            "invalid_multipart_request_json",
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        ) from exc
    if not isinstance(payload, dict):
        raise contract_error("invalid_multipart_request_object", reason="request JSON must be an object")
    return payload


async def request_to_annotation_job(
    request: Request,
    *,
    job_id: str | None,
    output_root: str,
    start_s: float | None,
    end_s: float | None,
    render_width: int | None,
    gpu_ids: str | None,
    run_preflight: bool,
    run_camera_trajectory: bool,
    run_hawor_metric_hands: bool,
    run_hybrid_hands: bool,
    run_gt_free_drift_self_calibration: bool,
    run_captioning: bool,
    run_self_consistency_qc: bool,
    run_evaluator: bool,
    actions_json: str | None,
    captions_json: str | None,
    semantic_review_json: str | None,
    run_semantic_annotation_agent: bool,
    head_gt: str | None,
    hand_gt: str | None,
    write_product_bundle: bool,
    total_request_limit: int,
    algorithm_inflight_multiplier: int,
    model_backend: str,
    diagnostic_monocular: bool,
    service_profile: str | None,
    service_endpoints: str | None,
    filename: str | None,
) -> AnnotationJobRequest:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    fps_condition = request.headers.get("x-ego-api-ify-fps-condition", DEFAULT_FPS_CONDITION).strip()
    try:
        get_fps_condition(fps_condition)
    except ValueError as exc:
        raise contract_error("invalid_internal_fps_condition", value=fps_condition, allowed=sorted(FPS_CONDITION_BY_NAME)) from exc
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        unknown_form_fields = sorted(set(form) - {"file"})
        if unknown_form_fields:
            raise contract_error("unknown_multipart_form_fields", fields=unknown_form_fields)
        file_values = form.getlist("file")
        if not file_values:
            raise HTTPException(status_code=422, detail={"code": "missing_upload_file_field", "field": "file"})
        if len(file_values) != 1:
            raise contract_error("duplicate_upload_file_field", field="file")
        file = file_values[0]
        if not isinstance(file, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=422, detail={"code": "invalid_upload_file_field", "field": "file"})
        multipart_payload: dict[str, Any] = {}
        query_payload: dict[str, Any] = {
            "job_id": job_id,
            "output_root": output_root,
            "start_s": start_s,
            "end_s": end_s,
            "render_width": render_width,
            "gpu_ids": gpu_ids,
            "run_preflight": run_preflight,
            "run_camera_trajectory": run_camera_trajectory,
            "run_hawor_metric_hands": run_hawor_metric_hands,
            "run_hybrid_hands": run_hybrid_hands,
            "run_gt_free_drift_self_calibration": run_gt_free_drift_self_calibration,
            "run_captioning": run_captioning,
            "run_self_consistency_qc": run_self_consistency_qc,
            "run_evaluator": run_evaluator,
            "actions_json": actions_json,
            "captions_json": captions_json,
            "semantic_review_json": semantic_review_json,
            "run_semantic_annotation_agent": run_semantic_annotation_agent,
            "head_gt": head_gt,
            "hand_gt": hand_gt,
            "write_product_bundle": write_product_bundle,
            "total_request_limit": total_request_limit,
            "algorithm_inflight_multiplier": algorithm_inflight_multiplier,
            "model_backend": model_backend,
            "diagnostic_monocular": diagnostic_monocular,
            "service_profile": service_profile,
            "service_endpoints": (
                multipart_payload.get("service_endpoints")
                if "service_endpoints" in multipart_payload
                else parse_query_service_endpoints(service_endpoints)
            ),
            "fps_condition": fps_condition,
        }
        merged_payload = {**query_payload, **multipart_payload}
        provisional = annotation_request_from_payload(
            merged_payload,
            multipart=True,
            authoritative_video_uri="/pending/upload",
        )
        ensure_total_backend_supported(provisional)
        resolved_job_id = clean_id(provisional.job_id or f"annotation_{uuid4().hex[:12]}")
        resolved_output_root = Path(provisional.output_root).expanduser().resolve()
        upload_name = clean_upload_filename(filename or file.filename or request.headers.get("x-filename"), default=f"{resolved_job_id}.mp4")
        destination = upload_destination(resolved_output_root, resolved_job_id, upload_name)
        req = provisional.model_copy(
            update={
                "video_uri": str(destination),
                "job_id": resolved_job_id,
                "output_root": str(resolved_output_root),
            }
        )
        try:
            upload_info = await write_upload_stream(file_chunks(file), destination)
        except UploadError as exc:
            raise HTTPException(status_code=400, detail={"code": str(exc)}) from exc
        req.upload_info = {
            "upload": upload_info,
            "content_type": content_type,
            "filename": upload_name,
            "request_field_present": False,
        }
        return req
    if content_type in {"", "application/json"}:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise contract_error("invalid_json_body", reason=str(exc)) from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail={"code": "invalid_json_body"})
        return annotation_request_from_payload(payload, multipart=False)
    if content_type not in {"application/octet-stream", "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"}:
        raise HTTPException(status_code=415, detail={"code": "unsupported_upload_content_type", "content_type": content_type, "supported": ["application/json", "application/octet-stream", "video/mp4"]})
    resolved_job_id = clean_id(job_id or f"annotation_{uuid4().hex[:12]}")
    resolved_output_root = Path(output_root).expanduser().resolve()
    upload_name = clean_upload_filename(filename or request.headers.get("x-filename"), default=f"{resolved_job_id}.mp4")
    destination = upload_destination(resolved_output_root, resolved_job_id, upload_name)
    try:
        upload_info = await write_upload_stream(request.stream(), destination)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail={"code": str(exc)}) from exc
    parsed_endpoints = validate_service_endpoints(parse_query_service_endpoints(service_endpoints))
    validate_backend_configuration(
        model_backend=model_backend,
        service_profile=service_profile,
        service_endpoints=parsed_endpoints,
    )
    req = AnnotationJobRequest(
        video_uri=str(destination),
        job_id=resolved_job_id,
        output_root=str(resolved_output_root),
        start_s=start_s,
        end_s=end_s,
        render_width=render_width,
        gpu_ids=gpu_ids,
        run_preflight=run_preflight,
        run_camera_trajectory=run_camera_trajectory,
        run_hawor_metric_hands=run_hawor_metric_hands,
        run_hybrid_hands=run_hybrid_hands,
        run_gt_free_drift_self_calibration=run_gt_free_drift_self_calibration,
        run_captioning=run_captioning,
        run_self_consistency_qc=run_self_consistency_qc,
        run_evaluator=run_evaluator,
        actions_json=actions_json,
        captions_json=captions_json,
        semantic_review_json=semantic_review_json,
        run_semantic_annotation_agent=run_semantic_annotation_agent,
        head_gt=head_gt,
        hand_gt=hand_gt,
        write_product_bundle=write_product_bundle,
        total_request_limit=total_request_limit,
        algorithm_inflight_multiplier=algorithm_inflight_multiplier,
        model_backend=model_backend,
        diagnostic_monocular=diagnostic_monocular,
        service_profile=service_profile,
        service_endpoints=parsed_endpoints,
        fps_condition=fps_condition,
    )
    req.upload_info = {"upload": upload_info, "content_type": content_type}
    return req


def ffprobe_video_metadata(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration", "-of", "json", str(path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ffprobe_failed_rc={proc.returncode}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("ffprobe returned no video stream")
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate") or "0/1")
    num, den = (rate.split("/", 1) + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    return {
        "path": str(path),
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_count": int(frames or 0),
        "duration_s": float(stream.get("duration") or 0.0),
        "metadata_source": "ffprobe",
    }


def opencv_video_metadata(path: Path) -> dict[str, Any]:
    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could_not_open_video: {path}")
    meta = {
        "path": str(path),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "metadata_source": "opencv",
    }
    cap.release()
    meta["duration_s"] = float(meta["frame_count"] / meta["fps"]) if meta["fps"] else 0.0
    return meta


def helper_python_video_metadata(path: Path, python_bin: str) -> dict[str, Any]:
    helper = Path(python_bin).expanduser()
    if not helper.exists():
        raise RuntimeError(f"metadata_python_missing: {helper}")
    code = """
import json
import sys
import cv2
path = sys.argv[1]
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f'could_not_open_video: {path}')
fps = float(cap.get(cv2.CAP_PROP_FPS))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
payload = {
    'path': path,
    'fps': fps,
    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    'frame_count': frames,
    'duration_s': float(frames / fps) if fps else 0.0,
    'metadata_source': 'helper_python_opencv',
    'metadata_python': sys.executable,
}
cap.release()
print(json.dumps(payload))
"""
    proc = subprocess.run([str(helper), "-c", code, str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"metadata_python_failed_rc={proc.returncode}")
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"metadata_python_returned_non_object: {proc.stdout[:200]}")
    return payload


def local_video_metadata(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    for source in ("local_opencv", "helper_python", "ffprobe"):
        try:
            if source == "local_opencv":
                meta = opencv_video_metadata(path)
            elif source == "helper_python":
                meta = helper_python_video_metadata(path, ANNOTATION_METADATA_PYTHON)
            else:
                meta = ffprobe_video_metadata(path)
            if meta["fps"] <= 0 or meta["width"] <= 0 or meta["height"] <= 0 or meta["frame_count"] <= 0:
                raise RuntimeError(f"invalid_video_metadata: {meta}")
            return meta
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError("; ".join(errors))


def extract_request_video_metadata(video_uri: str, *, require_local: bool) -> dict[str, Any]:
    path = Path(video_uri).expanduser()
    if path.exists() and path.is_file():
        try:
            return {"status": "ok", **local_video_metadata(path)}
        except Exception as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_uploaded_video_metadata", "video_uri": video_uri, "error": str(exc)}) from exc
    if require_local:
        raise HTTPException(status_code=422, detail={"code": "video_metadata_requires_local_file", "video_uri": video_uri})
    return {"status": "not_extracted_service_side", "reason": "video path is not local to the API process", "video_uri": video_uri}


def pipeline_flags_from_request(req: AnnotationJobRequest) -> dict[str, Any]:
    return {
        "start_s": req.start_s,
        "end_s": req.end_s,
        "render_width": req.render_width,
        "gpu_ids": req.gpu_ids,
        "run_preflight": bool(req.run_preflight),
        "run_camera_trajectory": bool(req.run_camera_trajectory),
        "run_hawor_metric_hands": bool(req.run_hawor_metric_hands),
        "run_hybrid_hands": bool(req.run_hybrid_hands),
        "run_gt_free_drift_self_calibration": bool(req.run_gt_free_drift_self_calibration),
        "run_captioning": bool(req.run_captioning),
        "run_self_consistency_qc": bool(req.run_self_consistency_qc),
        "run_evaluator": bool(req.run_evaluator),
        "actions_json": req.actions_json,
        "captions_json": req.captions_json,
        "semantic_review_json": req.semantic_review_json,
        "run_semantic_annotation_agent": bool(req.run_semantic_annotation_agent),
        "write_product_bundle": bool(req.write_product_bundle),
        "model_backend": req.model_backend,
        "diagnostic_monocular": bool(req.diagnostic_monocular),
        "service_profile": req.service_profile,
        "service_endpoints": dict(req.service_endpoints) if req.service_endpoints is not None else None,
    }


def ensure_total_backend_supported(req: AnnotationJobRequest) -> None:
    if req.model_backend == "api_ify":
        if not req.diagnostic_monocular:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "api_ify_strict_rgbd_capability_unproven",
                    "reason": "the frozen DROID service has not provided full-K/native-depth consumption evidence; set diagnostic_monocular=true to execute the complete API flow without metric acceptance",
                },
            )
        return
    if req.model_backend == "feishu_ray":
        raise HTTPException(
            status_code=501,
            detail={
                "code": "feishu_ray_pipeline_adapter_not_implemented",
                "model_backend": req.model_backend,
                "service_profile": req.service_profile,
                "reason": "the required D4/D5/D9b service responses are not yet materialized into a complete annotation pipeline",
            },
        )
    if req.model_backend != "script":
        raise HTTPException(status_code=422, detail={"code": "invalid_model_backend", "value": req.model_backend, "allowed": sorted(MODEL_BACKENDS)})


def run_annotation_job(req: AnnotationJobRequest) -> AnnotationJobResponse:
    started = time.time()
    job_id = clean_id(req.job_id or f"annotation_{uuid4().hex[:12]}")
    output_root = Path(req.output_root).expanduser().resolve()
    package_root = DEFAULT_PACKAGE_ROOT.expanduser().resolve()
    run_root = output_root / job_id
    ensure_total_backend_supported(req)
    if req.run_semantic_annotation_agent:
        raise HTTPException(status_code=422, detail={"code": "semantic_annotation_agent_disabled", "reason": "current D9b captioning must be fully scripted through Cosmos or explicit source sidecars"})
    if ANNOTATION_RUNTIME_MODE == "pi_agent":
        raise HTTPException(status_code=500, detail={"code": "agent_runtime_disabled", "mode": ANNOTATION_RUNTIME_MODE, "reason": "current pipeline run must be fully scripted and must not launch annotation agents"})
    if ANNOTATION_RUNTIME_MODE not in {"direct_script", "direct"}:
        raise HTTPException(status_code=500, detail={"code": "unsupported_annotation_runtime_mode", "mode": ANNOTATION_RUNTIME_MODE, "supported": ["direct_script"]})
    if REMOTE_CONFIG is not None:
        return run_remote_job(req, job_id=job_id, output_root=output_root, package_root=package_root, started=started)
    if req.model_backend == "api_ify":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_single_video_api.py"),
            "--case-id", job_id,
            "--input", req.video_uri,
            "--run-root", str(run_root),
            "--diagnostic-monocular",
            "--fps-condition", req.fps_condition,
        ]
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_v22_minimal_annotation_pipeline.py"),
            "--case-id", job_id,
            "--input-video", req.video_uri,
            "--run-root", str(run_root),
            "--repo-root", str(REPO_ROOT),
            *( ["--render-width", str(int(req.render_width))] if req.render_width is not None else [] ),
        ]
    if req.model_backend != "api_ify":
        if req.start_s is not None:
            cmd.extend(["--start-s", str(float(req.start_s))])
        if req.end_s is not None:
            cmd.extend(["--end-s", str(float(req.end_s))])
        if req.gpu_ids:
            cmd.extend(["--gpu-ids", req.gpu_ids])
        if req.run_preflight:
            cmd.append("--run-preflight")
        if req.run_camera_trajectory:
            cmd.append("--run-camera-trajectory")
        if req.run_hawor_metric_hands:
            cmd.append("--run-hawor-metric-hands")
        if req.run_hybrid_hands:
            cmd.append("--run-hybrid-hands")
        if req.run_gt_free_drift_self_calibration:
            cmd.append("--run-gt-free-drift-self-calibration")
        if req.run_captioning:
            cmd.append("--run-captioning")
        if req.run_self_consistency_qc:
            cmd.append("--run-self-consistency-qc")
        if req.run_evaluator:
            cmd.append("--run-evaluator")
        if req.actions_json:
            cmd.extend(["--actions-json", req.actions_json])
        if req.captions_json:
            cmd.extend(["--captions-json", req.captions_json])
        if req.semantic_review_json:
            cmd.extend(["--semantic-review-json", req.semantic_review_json])
        if req.head_gt:
            cmd.extend(["--head-gt", req.head_gt])
        if req.hand_gt:
            cmd.extend(["--hand-gt", req.hand_gt])
        if req.write_product_bundle:
            cmd.append("--write-product-bundle")
    if req.model_backend == "api_ify":
        # The wrapper is the single-item manager's control plane. Its route
        # proxy owns stage-scoped admission, including DROID create→finalize;
        # the API-Ify DAG itself remains unchanged.
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_v22_api_job_with_admission.py"),
            "--job-id", job_id,
            "--repo-root", str(REPO_ROOT),
            "--profile", str(REPO_ROOT / "configs" / "feishu_ray_services.json"),
            "--lock-root", str(output_root / "_algorithm_admission"),
            "--events-path", str(output_root / "_algorithm_admission_events.jsonl"),
            "--algorithm-inflight-multiplier", str(req.algorithm_inflight_multiplier),
            "--upstream-endpoints-json", json.dumps(req.service_endpoints or {}, sort_keys=True),
            "--api-ify",
            "--",
            *cmd,
        ]
    log_dir = output_root / "_api_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    child_env = os.environ.copy()
    child_pythonpath = [str(REPO_ROOT)]
    if child_env.get("PYTHONPATH"):
        child_pythonpath.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(child_pythonpath)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=child_env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise HTTPException(status_code=500, detail={"code": "annotation_pipeline_failed", "run_root": str(run_root), "log": str(log_path), "tail": tail})
    manifest_path = run_root / "annotation_pipeline_manifest.json"
    summary = {
        **load_json(manifest_path),
        "admission": admission_summary(
            total_request_limit=req.total_request_limit,
            algorithm_inflight_multiplier=req.algorithm_inflight_multiplier,
        ),
    }
    renders = summary.get("renders") if isinstance(summary.get("renders"), dict) else {}
    try:
        package = create_result_package(run_root, package_root, package_name=f"{job_id}_annotation_result")
    except PackageError as exc:
        raise HTTPException(status_code=500, detail={"code": "annotation_package_failed", "run_root": str(run_root), "error": str(exc)}) from exc
    package_path = Path(str(package["package_path"]))
    upload_info = req.upload_info
    if upload_info is not None:
        summary = {**summary, "request_upload": upload_info}
    return AnnotationJobResponse(
        job_id=job_id,
        status=str(summary.get("status", "ok")),
        run_root=str(run_root),
        manifest_path=str(manifest_path),
        overlay_path=renders.get("v22_combined") or renders.get("v22_overlay"),
        depth_overlay_path=renders.get("depth_overlay"),
        product_manifest_path=summary.get("product_manifest_path"),
        package_path=str(package_path),
        download_url=package_download_url(package_path.name),
        elapsed_s=float(time.time() - started),
        summary=summary,
    )


def run_annotation_job_set(req: AnnotationJobSetRequest) -> AnnotationJobSetResponse:
    raise HTTPException(status_code=410, detail={"code": "annotation_job_sets_disabled", "reason": "current MVP service accepts exactly one uploaded video per request; callers should loop over /v1/annotation-jobs outside the service if they need multiple videos"})


def run_remote_job(req: AnnotationJobRequest, *, job_id: str, output_root: Path, package_root: Path, started: float) -> AnnotationJobResponse:
    source_path = Path(req.video_uri).expanduser()
    has_local_upload = req.upload_info is not None or source_path.exists()
    local_video = source_path.resolve() if has_local_upload else None
    remote_video = None if has_local_upload else Path(req.video_uri)
    filename = Path(req.video_uri).name or f"{job_id}.mp4"
    try:
        remote = run_remote_annotation_job(
            config=REMOTE_CONFIG,  # type: ignore[arg-type]
            job_id=job_id,
            local_video=local_video,
            remote_video=remote_video,
            local_package_root=package_root,
            filename=filename,
            start_s=req.start_s,
            end_s=req.end_s,
            render_width=int(req.render_width) if req.render_width is not None else None,
            gpu_ids=req.gpu_ids,
            run_preflight=bool(req.run_preflight),
            run_camera_trajectory=bool(req.run_camera_trajectory),
            run_hawor_metric_hands=bool(req.run_hawor_metric_hands),
            run_hybrid_hands=bool(req.run_hybrid_hands),
            run_gt_free_drift_self_calibration=bool(req.run_gt_free_drift_self_calibration),
            run_captioning=bool(req.run_captioning),
            run_self_consistency_qc=bool(req.run_self_consistency_qc),
            run_evaluator=bool(req.run_evaluator),
            actions_json=Path(req.actions_json) if req.actions_json else None,
            captions_json=Path(req.captions_json) if req.captions_json else None,
            semantic_review_json=Path(req.semantic_review_json) if req.semantic_review_json else None,
            head_gt=Path(req.head_gt) if req.head_gt else None,
            hand_gt=Path(req.hand_gt) if req.hand_gt else None,
            write_product_bundle=bool(req.write_product_bundle),
            algorithm_inflight_multiplier=req.algorithm_inflight_multiplier,
            model_backend=req.model_backend,
            service_profile=req.service_profile,
            service_endpoints=req.service_endpoints,
            diagnostic_monocular=bool(req.diagnostic_monocular),
            timeout=REMOTE_TIMEOUT_S,
        )
    except RemoteExecutionError as exc:
        raise HTTPException(status_code=500, detail={"code": "remote_annotation_failed", "job_id": job_id, "error": str(exc)[-5000:]}) from exc
    summary = remote["summary"]
    renders = summary.get("renders") if isinstance(summary.get("renders"), dict) else {}
    package_path = Path(str(remote["local_package_path"]))
    response_summary = {
        **summary,
        "admission": admission_summary(
            total_request_limit=req.total_request_limit,
            algorithm_inflight_multiplier=req.algorithm_inflight_multiplier,
        ),
        "remote_execution": {k: v for k, v in remote.items() if k != "summary"},
    }
    if req.upload_info is not None:
        response_summary["request_upload"] = req.upload_info
    return AnnotationJobResponse(
        job_id=job_id,
        status=str(summary.get("status", "ok")),
        run_root=str(remote["remote_run_root"]),
        manifest_path=str(remote["remote_manifest_path"]),
        overlay_path=renders.get("v22_combined") or renders.get("v22_overlay"),
        depth_overlay_path=renders.get("depth_overlay"),
        product_manifest_path=summary.get("product_manifest_path"),
        package_path=str(package_path),
        download_url=package_download_url(package_path.name),
        elapsed_s=float(time.time() - started),
        summary=response_summary,
    )


async def file_chunks(upload: UploadFile, chunk_size: int = 1024 * 1024):
    try:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        await upload.close()


app = FastAPI(title="Ego Annotation API", version="0.1.0")

ANNOTATION_JOB_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "The authoritative input video upload.",
                        },
                    },
                    "additionalProperties": False,
                }
            },
        },
    }
}


@app.post(
    "/v1/annotation-jobs",
    response_model=AnnotationJobResponse,
    openapi_extra=ANNOTATION_JOB_OPENAPI_EXTRA,
)
async def create_annotation_job(request: Request) -> AnnotationJobResponse:
    validate_annotation_query_fields(request)
    async with acquire_api_request_slot():
        req = await request_to_annotation_job(
            request,
            job_id=None,
            output_root=str(DEFAULT_OUTPUT_ROOT),
            start_s=None,
            end_s=None,
            render_width=None,
            gpu_ids=None,
            run_preflight=False,
            run_camera_trajectory=True,
            run_hawor_metric_hands=True,
            run_hybrid_hands=True,
            run_gt_free_drift_self_calibration=True,
            run_captioning=True,
            run_self_consistency_qc=True,
            run_evaluator=False,
            actions_json=None,
            captions_json=None,
            semantic_review_json=None,
            run_semantic_annotation_agent=False,
            head_gt=None,
            hand_gt=None,
            write_product_bundle=True,
            total_request_limit=TOTAL_REQUEST_LIMIT,
            algorithm_inflight_multiplier=ALGORITHM_INFLIGHT_MULTIPLIER,
            model_backend="api_ify",
            diagnostic_monocular=True,
            service_profile=None,
            service_endpoints=None,
            filename=None,
        )
        validate_request_admission(req.total_request_limit, req.algorithm_inflight_multiplier)
        return await run_job_without_early_slot_release(req)


@app.post("/v1/annotation-job-sets", response_model=AnnotationJobSetResponse)
def create_annotation_job_set(payload: AnnotationJobSetRequest) -> AnnotationJobSetResponse:
    return run_annotation_job_set(payload)


@app.get("/v1/downloads/{filename}")
def download_annotation_result(filename: str) -> FileResponse:
    path = safe_download_path(filename)
    return FileResponse(path, media_type="application/zip", filename=path.name)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--total-request-limit", type=int, default=TOTAL_REQUEST_LIMIT)
    parser.add_argument("--algorithm-inflight-multiplier", type=int, default=ALGORITHM_INFLIGHT_MULTIPLIER)
    args = parser.parse_args()
    configure_admission_limits(
        total_request_limit=args.total_request_limit,
        algorithm_inflight_multiplier=args.algorithm_inflight_multiplier,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
