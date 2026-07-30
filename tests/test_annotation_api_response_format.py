from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from scripts.serve_v22_annotation_api import AnnotationJobResponse, format_annotation_job_text, wants_json_response


class DummyRequest:
    def __init__(self, accept: str = "*/*", content_type: str = "multipart/form-data", query_params: dict[str, str] | None = None) -> None:
        self.headers = {"accept": accept, "content-type": content_type}
        self.query_params = query_params or {}


def test_default_multipart_curl_accept_gets_plain_text() -> None:
    assert wants_json_response(DummyRequest("*/*")) is False


def test_json_body_keeps_json_default() -> None:
    assert wants_json_response(DummyRequest("*/*", content_type="application/json")) is True


def test_explicit_json_accept_gets_json() -> None:
    assert wants_json_response(DummyRequest("application/json")) is True
    assert wants_json_response(DummyRequest("*/*"), response_format="json") is True


def test_explicit_text_overrides_json_accept() -> None:
    assert wants_json_response(DummyRequest("application/json"), response_format="text") is False


def test_annotation_job_text_is_compact_and_downloadable() -> None:
    response = AnnotationJobResponse(
        job_id="annotation_123",
        status="ok",
        run_root="/remote/run",
        manifest_path="/remote/run/annotation_pipeline_manifest.json",
        package_path="/downloads/annotation_123_annotation_result.zip",
        download_url="http://192.168.9.220:8091/v1/downloads/annotation_123_annotation_result.zip",
        elapsed_s=12.345,
        summary={
            "steps": [
                {"step": "prepare_single_video", "status": "ok"},
                {"step": "unidepth", "status": "ok"},
            ],
            "ffprobe_overlay": {
                "ffprobe": {
                    "streams": [
                        {
                            "width": 1920,
                            "height": 1080,
                            "duration": "24.000000",
                            "nb_read_frames": "720",
                        }
                    ]
                }
            },
            "remote_execution": {"internal": "not shown"},
        },
    )
    text = format_annotation_job_text(response)
    assert "status=ok" in text
    assert "job_id=annotation_123" in text
    assert "progress=2/2 stages completed" in text
    assert "overlay=720 frames, 24.000000s, 1920x1080" in text
    assert "download_command=curl --noproxy '*' -O http://192.168.9.220:8091/v1/downloads/annotation_123_annotation_result.zip" in text
    assert "remote_execution" not in text
    assert "summary" not in text
