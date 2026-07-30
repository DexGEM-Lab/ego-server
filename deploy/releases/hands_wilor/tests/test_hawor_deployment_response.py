"""Deployment-boundary tests for GPU3 multipart HTTP responses.

The serving module is intentionally Ray-only; skip this test in lightweight
contract environments that do not install Ray/Starlette.
"""
from __future__ import annotations

import pytest

pytest.importorskip("ray")
from starlette.responses import Response

from ego_annotation.serving.hawor_deployment import _json_error_response, _multipart_response


def test_binary_multipart_response_is_an_asgi_response_not_json_data() -> None:
    body = b"--boundary\r\n\xff\x00\x80\r\n--boundary--\r\n"
    response = _multipart_response(body, "multipart/form-data; boundary=boundary")

    assert isinstance(response, Response)
    assert response.body == body
    assert response.headers["content-type"] == "multipart/form-data; boundary=boundary"


def test_error_response_is_explicit_json_with_status() -> None:
    response = _json_error_response({"code": "validation", "message": "bad request"}, status_code=400)

    assert isinstance(response, Response)
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert b'"code":"validation"' in response.body
