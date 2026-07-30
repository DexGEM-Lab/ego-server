from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")

from scripts import serve_v22_annotation_api as api
from scripts.annotation_runtime_agent import build_runtime_request


async def asgi_request(path: str, body: bytes, content_type: str) -> tuple[int, bytes, dict[str, str]]:
    parsed = urlsplit(path)
    messages: list[dict] = []
    received = False

    async def receive() -> dict:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode(),
        "query_string": parsed.query.encode(),
        "headers": [(b"host", b"testserver"), (b"content-type", content_type.encode()), (b"accept", b"*/*")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await api.app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    headers = {key.decode().lower(): value.decode() for key, value in start.get("headers", [])}
    return int(start["status"]), response_body, headers


def multipart(parts: list[tuple[str, bytes, str | None, str | None]]) -> tuple[bytes, str]:
    boundary = "endpoint-contract-boundary"
    chunks: list[bytes] = []
    for name, data, filename, part_type in parts:
        chunks.append(f"--{boundary}\r\n".encode())
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        chunks.append((disposition + "\r\n").encode())
        if part_type is not None:
            chunks.append(f"Content-Type: {part_type}\r\n".encode())
        chunks.append(b"\r\n" + data + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def fake_response(job_id: str) -> api.AnnotationJobResponse:
    return api.AnnotationJobResponse(
        job_id=job_id,
        status="ok",
        run_root=f"/runs/{job_id}",
        manifest_path=f"/runs/{job_id}/annotation_pipeline_manifest.json",
        elapsed_s=0.01,
        summary={},
    )


def test_exact_short_curl_file_only_reaches_canonical_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[api.AnnotationJobRequest] = []
    destination = tmp_path / "uploads" / "video.mp4"
    monkeypatch.setattr(api, "upload_destination", lambda *_args: destination)

    def fake_run(req: api.AnnotationJobRequest) -> api.AnnotationJobResponse:
        captured.append(req)
        return fake_response(req.job_id or "missing")

    monkeypatch.setattr(api, "run_annotation_job", fake_run)
    body, content_type = multipart([("file", b"local-video-bytes", "video.mp4", "video/mp4")])
    status, response_body, headers = asyncio.run(asgi_request("/v1/annotation-jobs", body, content_type))

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert json.loads(response_body)["status"] == "ok"
    assert len(captured) == 1
    req = captured[0]
    assert req.model_backend == "api_ify"
    assert req.diagnostic_monocular is True
    assert req.run_captioning is True
    assert req.total_request_limit == api.TOTAL_REQUEST_LIMIT
    assert req.algorithm_inflight_multiplier == api.ALGORITHM_INFLIGHT_MULTIPLIER
    assert req.service_profile is None
    assert req.video_uri == str(destination)
    assert destination.read_bytes() == b"local-video-bytes"


def test_external_query_options_are_rejected() -> None:
    body, content_type = multipart([("file", b"video", "uploaded.mp4", "video/mp4")])
    status, response_body, _ = asyncio.run(asgi_request("/v1/annotation-jobs?total_request_limit=1", body, content_type))
    assert status == 422
    assert json.loads(response_body)["detail"]["code"] == "unknown_annotation_query_fields"


def test_multipart_request_options_are_rejected() -> None:
    body, content_type = multipart(
        [
            ("request", b"{}", None, "application/json"),
            ("file", b"video", "uploaded.mp4", "video/mp4"),
        ]
    )
    status, response_body, _ = asyncio.run(asgi_request("/v1/annotation-jobs", body, content_type))
    assert status == 422
    assert json.loads(response_body)["detail"] == {
        "code": "unknown_multipart_form_fields",
        "fields": ["request"],
    }


def test_openapi_describes_file_only_multipart_contract() -> None:
    operation = api.app.openapi()["paths"]["/v1/annotation-jobs"]["post"]
    multipart_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert multipart_schema["required"] == ["file"]
    assert multipart_schema["properties"] == {
        "file": {
            "type": "string",
            "format": "binary",
            "description": "The authoritative input video upload.",
        }
    }
    assert set(operation["requestBody"]["content"]) == {"multipart/form-data"}
    assert operation.get("parameters", []) == []


def test_backend_configuration_is_preserved_in_runtime_pipeline_flags(tmp_path: Path) -> None:
    req = api.AnnotationJobRequest(
        video_uri="/video.mp4",
        model_backend="feishu_ray",
        service_profile="profile-a",
        service_endpoints={"droid": "http://127.0.0.1:28002", "wilor": "http://127.0.0.1:28004"},
    )
    flags = api.pipeline_flags_from_request(req)
    runtime_request = build_runtime_request(
        repo_root=tmp_path / "repo",
        job_id="job-1",
        video_uri=req.video_uri,
        run_root=tmp_path / "run",
        package_root=tmp_path / "packages",
        local_video=True,
        remote_config=None,
        timeout_s=30,
        metadata=None,
        pipeline_flags=flags,
    )
    assert runtime_request["pipeline_flags"]["model_backend"] == "feishu_ray"
    assert runtime_request["pipeline_flags"]["service_profile"] == "profile-a"
    assert runtime_request["pipeline_flags"]["service_endpoints"] == {"droid": "http://127.0.0.1:28002", "wilor": "http://127.0.0.1:28004"}
