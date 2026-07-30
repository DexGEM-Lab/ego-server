#!/usr/bin/env python3
"""FastAPI service for the minimal V22 annotation pipeline."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation_api_uploads import UploadError, clean_upload_filename, upload_destination, write_upload_stream
from scripts.annotation_remote_runner import RemoteExecutionError, config_from_env, run_remote_annotation_job
from scripts.package_v22_annotation_result import PackageError, create_result_package, resolve_download_package

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ANNOTATION_OUTPUT_ROOT", "/home/zjh/data/v22_api_jobs"))
DEFAULT_PACKAGE_ROOT = Path(os.environ.get("ANNOTATION_PACKAGE_ROOT", "/home/zjh/data/v22_api_downloads"))
PUBLIC_BASE_URL = os.environ.get("ANNOTATION_PUBLIC_BASE_URL", "").rstrip("/")
REMOTE_CONFIG = config_from_env(os.environ)
REMOTE_TIMEOUT_S = int(os.environ.get("ANNOTATION_REMOTE_TIMEOUT_S", "7200"))


class AnnotationJobRequest(BaseModel):
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
    head_gt: str | None = None
    hand_gt: str | None = None
    write_product_bundle: bool = True
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
    head_gt: str | None,
    hand_gt: str | None,
    write_product_bundle: bool,
    filename: str | None,
) -> AnnotationJobRequest:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if file is None or not isinstance(file, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=422, detail={"code": "missing_upload_file_field", "field": "file"})
        resolved_job_id = clean_id(job_id or f"annotation_{uuid4().hex[:12]}")
        resolved_output_root = Path(output_root).expanduser().resolve()
        upload_name = clean_upload_filename(filename or file.filename or request.headers.get("x-filename"), default=f"{resolved_job_id}.mp4")
        destination = upload_destination(resolved_output_root, resolved_job_id, upload_name)
        try:
            upload_info = await write_upload_stream(file_chunks(file), destination)
        except UploadError as exc:
            raise HTTPException(status_code=400, detail={"code": str(exc)}) from exc
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
            head_gt=head_gt,
            hand_gt=hand_gt,
            write_product_bundle=write_product_bundle,
        )
        req.upload_info = {"upload": upload_info, "content_type": content_type, "filename": upload_name}
        return req
    if content_type in {"", "application/json"}:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail={"code": "invalid_json_body"})
        return AnnotationJobRequest(**payload)
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
        head_gt=head_gt,
        hand_gt=hand_gt,
        write_product_bundle=write_product_bundle,
    )
    req.upload_info = {"upload": upload_info, "content_type": content_type}
    return req


def run_annotation_job(req: AnnotationJobRequest) -> AnnotationJobResponse:
    started = time.time()
    job_id = clean_id(req.job_id or f"annotation_{uuid4().hex[:12]}")
    output_root = Path(req.output_root).expanduser().resolve()
    package_root = DEFAULT_PACKAGE_ROOT.expanduser().resolve()
    run_root = output_root / job_id
    if REMOTE_CONFIG is not None:
        return run_remote_job(req, job_id=job_id, output_root=output_root, package_root=package_root, started=started)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_v22_minimal_annotation_pipeline.py"),
        "--case-id", job_id,
        "--input-video", req.video_uri,
        "--run-root", str(run_root),
        "--repo-root", str(REPO_ROOT),
        *( ["--render-width", str(int(req.render_width))] if req.render_width is not None else [] ),
    ]
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
    if req.head_gt:
        cmd.extend(["--head-gt", req.head_gt])
    if req.hand_gt:
        cmd.extend(["--hand-gt", req.hand_gt])
    if req.write_product_bundle:
        cmd.append("--write-product-bundle")
    log_dir = output_root / "_api_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise HTTPException(status_code=500, detail={"code": "annotation_pipeline_failed", "run_root": str(run_root), "log": str(log_path), "tail": tail})
    manifest_path = run_root / "annotation_pipeline_manifest.json"
    summary = load_json(manifest_path)
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
        overlay_path=renders.get("v22_overlay"),
        depth_overlay_path=renders.get("depth_overlay"),
        product_manifest_path=summary.get("product_manifest_path"),
        package_path=str(package_path),
        download_url=package_download_url(package_path.name),
        elapsed_s=float(time.time() - started),
        summary=summary,
    )


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
            head_gt=Path(req.head_gt) if req.head_gt else None,
            hand_gt=Path(req.hand_gt) if req.hand_gt else None,
            write_product_bundle=bool(req.write_product_bundle),
            timeout=REMOTE_TIMEOUT_S,
        )
    except RemoteExecutionError as exc:
        raise HTTPException(status_code=500, detail={"code": "remote_annotation_failed", "job_id": job_id, "error": str(exc)[-5000:]}) from exc
    summary = remote["summary"]
    renders = summary.get("renders") if isinstance(summary.get("renders"), dict) else {}
    package_path = Path(str(remote["local_package_path"]))
    response_summary = {**summary, "remote_execution": {k: v for k, v in remote.items() if k != "summary"}}
    if req.upload_info is not None:
        response_summary["request_upload"] = req.upload_info
    return AnnotationJobResponse(
        job_id=job_id,
        status=str(summary.get("status", "ok")),
        run_root=str(remote["remote_run_root"]),
        manifest_path=str(remote["remote_manifest_path"]),
        overlay_path=renders.get("v22_overlay"),
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


@app.post("/v1/annotation-jobs", response_model=AnnotationJobResponse)
async def create_annotation_job(
    request: Request,
    job_id: str | None = None,
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
    start_s: float | None = None,
    end_s: float | None = None,
    render_width: int | None = None,
    gpu_ids: str | None = None,
    run_preflight: bool = False,
    run_camera_trajectory: bool = True,
    run_hawor_metric_hands: bool = True,
    run_hybrid_hands: bool = True,
    run_gt_free_drift_self_calibration: bool = True,
    run_captioning: bool = True,
    run_self_consistency_qc: bool = True,
    run_evaluator: bool = True,
    actions_json: str | None = None,
    captions_json: str | None = None,
    head_gt: str | None = None,
    hand_gt: str | None = None,
    write_product_bundle: bool = True,
    filename: str | None = None,
    response_format: str | None = None,
):
    json_response = wants_json_response(request, response_format=response_format)
    try:
        req = await request_to_annotation_job(
            request,
            job_id=job_id,
            output_root=output_root,
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
            head_gt=head_gt,
            hand_gt=hand_gt,
            write_product_bundle=write_product_bundle,
            filename=filename,
        )
        response = run_annotation_job(req)
    except HTTPException as exc:
        if json_response:
            raise
        return PlainTextResponse(compact_error_text(exc.detail), status_code=exc.status_code)
    if json_response:
        return response
    return PlainTextResponse(format_annotation_job_text(response))


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
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
