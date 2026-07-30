from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default as email_policy
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from scripts.call_feishu_ray_service import (
    ServiceCallerError,
    build_multipart_body,
    call_service,
    call_service_arrays,
    decode_service_response,
    parse_service_response,
    read_npy,
    validate_request_payload,
)


class FakeResponse:
    def __init__(self, content_type: str, body: bytes, status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def parse_multipart(content_type: str, body: bytes):
    return BytesParser(policy=email_policy).parsebytes(
        f"MIME-Version: 1.0\r\nContent-Type: {content_type}\r\n\r\n".encode() + body
    )


def test_direct_caller_encodes_and_decodes_deployed_multipart(tmp_path: Path) -> None:
    raw = tmp_path / "rgb.raw"
    raw.write_bytes(bytes(range(12)))
    output_dir = tmp_path / "result"
    response_data = b"\x00\x00\x80?\x00\x00\x00@"
    response_body, response_type = build_multipart_body(
        {"batch_id": "batch-1"},
        [{"name": "depth_m", "data": response_data, "shape": (2,), "dtype": "float32"}],
        boundary="response-boundary",
    )
    observed: dict[str, Any] = {}

    def opener(request, timeout: float):
        observed.update(request=request, timeout=timeout)
        return FakeResponse(response_type, response_body)

    report = call_service(
        {
            "base_url": "http://127.0.0.1:28000",
            "route": "/unidepth.infer",
            "metadata": {
                "ownership": {"request_id": "req-1", "job_id": "job-1", "stage_id": "unidepth.infer"},
                "model_revision": "unidepth-v2-vitl14-corrected",
            },
            "arrays": [{"name": "rgb", "path": str(raw), "shape": [2, 2, 3], "dtype": "uint8"}],
            "timeout_s": 5,
            "output_dir": str(output_dir),
        },
        opener=opener,
        boundary="request-boundary",
    )

    request = observed["request"]
    message = parse_multipart(request.headers["Content-type"], request.data)
    parts = list(message.iter_parts())
    assert [part.get_param("name", header="content-disposition") for part in parts] == ["metadata", "rgb"]
    assert json.loads(parts[0].get_payload(decode=True))["model_revision"] == "unidepth-v2-vitl14-corrected"
    assert parts[1].get_param("shape", header="content-disposition") == "2,2,3"
    assert parts[1].get_param("dtype", header="content-disposition") == "uint8"
    assert parts[1].get_payload(decode=True) == bytes(range(12))
    assert observed["timeout"] == 5.0

    assert report["metadata"] == {"batch_id": "batch-1"}
    data, shape, dtype, _ = read_npy(output_dir / "depth_m.npy")
    assert data == response_data
    assert shape == (2,)
    assert dtype == "float32"
    assert (output_dir / "response_report.json").is_file()


def test_in_memory_caller_returns_decoded_arrays_without_filesystem_output() -> None:
    response_data = b"\x00\x00\x80?\x00\x00\x00@"
    response_body, response_type = build_multipart_body(
        {"ownership": {"request_id": "req-1"}, "result": {"ok": True}},
        [{"name": "depth_m", "data": response_data, "shape": (2,), "dtype": "float32"}],
        boundary="response-boundary",
    )
    observed: dict[str, Any] = {}

    def opener(request, timeout: float):
        observed.update(request=request, timeout=timeout)
        return FakeResponse(response_type, response_body)

    report = call_service_arrays(
        base_url="http://127.0.0.1:28000",
        route="/unidepth.infer",
        metadata={"ownership": {"request_id": "req-1"}},
        arrays={"rgb": (bytes(range(12)), (2, 2, 3), "uint8")},
        timeout_s=5.0,
        opener=opener,
        boundary="request-boundary",
    )

    assert report["metadata"]["result"] == {"ok": True}
    assert report["arrays"][0]["name"] == "depth_m"
    assert report["arrays"][0]["shape"] == (2,)
    assert report["arrays"][0]["dtype"] == "float32"
    assert report["arrays"][0]["data"] == response_data
    assert "path" not in report["arrays"][0]
    request_parts = list(parse_multipart(observed["request"].headers["Content-type"], observed["request"].data).iter_parts())
    assert [part.get_param("name", header="content-disposition") for part in request_parts] == ["metadata", "rgb"]


def test_in_memory_caller_rejects_nonfinite_metadata_and_total_size(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.call_feishu_ray_service as caller_module

    opener_called = False

    def unopened(_request: Any, timeout: float) -> Any:
        nonlocal opener_called
        opener_called = True
        raise AssertionError(f"opener must not be called with timeout={timeout}")

    with pytest.raises(ServiceCallerError) as nonfinite:
        call_service_arrays(
            base_url="http://127.0.0.1:28000",
            route="/unidepth.infer",
            metadata={"temperature": float("nan")},
            arrays={},
            timeout_s=5.0,
            opener=unopened,
        )
    assert nonfinite.value.code == "invalid_metadata_json"
    assert nonfinite.value.response_received is False
    assert nonfinite.value.response_status is None
    assert nonfinite.value.response_headers is None
    assert nonfinite.value.raw_response_bytes is None
    assert opener_called is False

    monkeypatch.setattr(caller_module, "MAX_TOTAL_INPUT_BYTES", 3)
    with pytest.raises(ServiceCallerError) as oversized:
        call_service_arrays(
            base_url="http://127.0.0.1:28000",
            route="/unidepth.infer",
            metadata={},
            arrays={"rgb": (b"1234", (4,), "uint8")},
            timeout_s=5.0,
            opener=unopened,
        )
    assert oversized.value.code == "request_too_large"
    assert oversized.value.response_received is False
    assert opener_called is False


def test_disk_caller_marks_serialization_failure_not_received_before_opener(tmp_path: Path) -> None:
    opener_called = False

    def opener(_request: Any, timeout: float) -> Any:
        nonlocal opener_called
        opener_called = True
        raise AssertionError(f"opener must not be called with timeout={timeout}")

    with pytest.raises(ServiceCallerError) as raised:
        call_service(
            {
                "base_url": "http://127.0.0.1:28002",
                "route": "/droid.finalize",
                "metadata": {"invalid": float("nan")},
                "arrays": [],
                "timeout_s": 5,
                "output_dir": str(tmp_path / "response"),
            },
            opener=opener,
        )
    assert raised.value.code == "invalid_metadata_json"
    assert "metadata is not finite JSON data" in str(raised.value)
    assert raised.value.response_received is False
    assert raised.value.response_status is None
    assert raised.value.response_headers is None
    assert raised.value.raw_response_bytes is None
    assert opener_called is False


def test_callers_mark_request_construction_failure_not_received(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.call_feishu_ray_service as caller_module

    opener_called = False

    def broken_request(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("request construction failed")

    def opener(_request: Any, timeout: float) -> Any:
        nonlocal opener_called
        opener_called = True
        raise AssertionError(f"opener must not be called with timeout={timeout}")

    monkeypatch.setattr(caller_module, "Request", broken_request)
    with pytest.raises(ServiceCallerError) as memory_error:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    with pytest.raises(ServiceCallerError) as disk_error:
        call_service(
            {
                "base_url": "http://127.0.0.1:28002",
                "route": "/droid.finalize",
                "metadata": {},
                "arrays": [],
                "timeout_s": 5,
                "output_dir": str(tmp_path / "response"),
            },
            opener=opener,
        )
    for error in (memory_error.value, disk_error.value):
        assert error.code == "service_request_construction_failed"
        assert str(error) == "request construction failed"
        assert error.response_received is False
    assert opener_called is False


def test_in_memory_caller_marks_pre_response_opener_failure_not_received() -> None:
    def opener(_request: Any, timeout: float) -> Any:
        assert timeout == 5.0
        raise URLError("connection refused")

    with pytest.raises(ServiceCallerError) as raised:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    assert raised.value.code == "service_transport_failed"
    assert str(raised.value) == "http://127.0.0.1:28002/droid.finalize: <urlopen error connection refused>"
    assert raised.value.response_received is False
    assert raised.value.response_status is None
    assert raised.value.response_headers is None
    assert raised.value.raw_response_bytes is None


@pytest.mark.parametrize(
    ("content_type", "body", "expected_code"),
    [
        ("text/plain", b"opaque finalize failure", "unsupported_response_content_type"),
        ("application/json", b'{"truncated":', "invalid_json_response"),
    ],
)
def test_in_memory_caller_preserves_received_decode_failure_evidence(
    content_type: str,
    body: bytes,
    expected_code: str,
) -> None:
    headers = {"Content-Type": content_type, "X-Service-Request": "response-17"}

    def opener(_request: Any, timeout: float) -> FakeResponse:
        assert timeout == 5.0
        response = FakeResponse(content_type, body, status=422)
        response.headers = dict(headers)
        return response

    with pytest.raises(ServiceCallerError) as direct_decode:
        decode_service_response(422, headers, body)
    with pytest.raises(ServiceCallerError) as raised:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    assert raised.value.code == expected_code
    assert str(raised.value) == str(direct_decode.value)
    assert raised.value.response_received is True
    assert raised.value.response_status == 422
    assert raised.value.response_headers == headers
    assert raised.value.raw_response_bytes == body


def test_in_memory_caller_preserves_incomplete_ordinary_response() -> None:
    partial = b"partial ordinary response"
    headers = {"Content-Type": "application/json", "X-Service-Request": "response-incomplete-1"}

    class IncompleteResponse(FakeResponse):
        def read(self) -> bytes:
            raise IncompleteRead(partial, 19)

    def opener(_request: Any, timeout: float) -> IncompleteResponse:
        assert timeout == 5.0
        response = IncompleteResponse("application/json", b"", status=206)
        response.headers = dict(headers)
        return response

    with pytest.raises(ServiceCallerError) as raised:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    assert raised.value.code == "service_response_incomplete"
    assert str(raised.value) == (
        "http://127.0.0.1:28002/droid.finalize: "
        f"IncompleteRead({len(partial)} bytes read, 19 more expected)"
    )
    assert raised.value.response_received is True
    assert raised.value.response_status == 206
    assert raised.value.response_headers == headers
    assert raised.value.raw_response_bytes == partial
    assert isinstance(raised.value.__cause__, IncompleteRead)


def test_in_memory_caller_preserves_incomplete_http_error_response() -> None:
    partial = b"partial HTTP error response"
    headers = {"Content-Type": "application/json", "X-Service-Request": "response-incomplete-2"}

    class IncompleteReader:
        def read(self) -> bytes:
            raise IncompleteRead(partial, 23)

        def close(self) -> None:
            return None

    def opener(request: Any, timeout: float) -> Any:
        assert timeout == 5.0
        raise HTTPError(request.full_url, 502, "Bad Gateway", headers, IncompleteReader())

    with pytest.raises(ServiceCallerError) as raised:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    assert raised.value.code == "service_response_incomplete"
    assert str(raised.value) == (
        "http://127.0.0.1:28002/droid.finalize: "
        f"IncompleteRead({len(partial)} bytes read, 23 more expected)"
    )
    assert raised.value.response_received is True
    assert raised.value.response_status == 502
    assert raised.value.response_headers == headers
    assert raised.value.raw_response_bytes == partial
    assert isinstance(raised.value.__cause__, IncompleteRead)


def test_in_memory_caller_marks_body_read_failure_received() -> None:
    class ReadFailureResponse(FakeResponse):
        def read(self) -> bytes:
            raise OSError("response body read failed")

    headers = {"Content-Type": "application/json", "X-Service-Request": "response-18"}

    def opener(_request: Any, timeout: float) -> ReadFailureResponse:
        assert timeout == 5.0
        response = ReadFailureResponse("application/json", b"", status=503)
        response.headers = dict(headers)
        return response

    with pytest.raises(ServiceCallerError) as raised:
        call_service_arrays(
            base_url="http://127.0.0.1:28002",
            route="/droid.finalize",
            metadata={},
            arrays={},
            timeout_s=5.0,
            opener=opener,
        )
    assert raised.value.code == "service_transport_failed"
    assert str(raised.value) == "http://127.0.0.1:28002/droid.finalize: response body read failed"
    assert raised.value.response_received is True
    assert raised.value.response_status == 503
    assert raised.value.response_headers == headers
    assert raised.value.raw_response_bytes is None


def test_disk_caller_preserves_received_invalid_json_evidence(tmp_path: Path) -> None:
    body = b"not-json"
    headers = {"Content-Type": "application/json", "X-Service-Request": "response-19"}

    def opener(_request: Any, timeout: float) -> FakeResponse:
        assert timeout == 5.0
        response = FakeResponse("application/json", body, status=200)
        response.headers = dict(headers)
        return response

    with pytest.raises(ServiceCallerError) as raised:
        call_service(
            {
                "base_url": "http://127.0.0.1:28002",
                "route": "/droid.finalize",
                "metadata": {},
                "arrays": [],
                "timeout_s": 5,
                "output_dir": str(tmp_path / "response"),
            },
            opener=opener,
        )
    assert raised.value.code == "invalid_json_response"
    assert raised.value.response_received is True
    assert raised.value.response_status == 200
    assert raised.value.response_headers == headers
    assert raised.value.raw_response_bytes == body


def test_json_response_is_reported(tmp_path: Path) -> None:
    report = parse_service_response(
        200,
        {"content-type": "application/json"},
        b'{"result":{"session_id":"abc"}}',
        tmp_path,
    )
    assert report["status"] == "ok"
    assert report["metadata"]["result"]["session_id"] == "abc"
    assert report["arrays"] == []


def test_npy_input_infers_shape_dtype_and_bytes(tmp_path: Path) -> None:
    raw = b"\x01\x00\x02\x00"
    body, content_type = build_multipart_body(
        {"ok": True},
        [{"name": "values", "data": raw, "shape": (2,), "dtype": "int16"}],
        boundary="source-boundary",
    )
    parsed = parse_service_response(200, {"content-type": content_type}, body, tmp_path / "source")
    payload = validate_request_payload(
        {
            "base_url": "http://127.0.0.1:28003",
            "route": "/hawor_infiller.fill",
            "metadata": {"work_units": 1},
            "arrays": [{"name": "mano_state", "path": parsed["arrays"][0]["path"]}],
            "timeout_s": 30,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert payload["arrays"][0]["shape"] == (2,)
    assert payload["arrays"][0]["dtype"] == "int16"
    assert payload["arrays"][0]["data"] == raw


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"route": "http://evil.invalid/infer"}, "invalid_route"),
        ({"base_url": "file:///tmp/service"}, "invalid_base_url"),
        ({"metadata": {"input_path": "/tmp/input.npy"}}, "filesystem_metadata_forbidden"),
        ({"arrays": [{"name": "rgb", "path": "/missing", "shape": [1], "dtype": "uint8"}]}, "array_path_not_found"),
        ({"typo_timeout": 1}, "unknown_request_fields"),
    ],
)
def test_direct_caller_rejects_invalid_contract(tmp_path: Path, patch: dict[str, Any], code: str) -> None:
    payload: dict[str, Any] = {
        "base_url": "http://127.0.0.1:28000",
        "route": "/unidepth.infer",
        "metadata": {},
        "arrays": [],
        "timeout_s": 5,
        "output_dir": str(tmp_path),
    }
    payload.update(patch)
    with pytest.raises(ServiceCallerError) as raised:
        validate_request_payload(payload)
    assert raised.value.code == code


def test_raw_array_size_must_match_shape_and_dtype(tmp_path: Path) -> None:
    raw = tmp_path / "short.raw"
    raw.write_bytes(b"\x00\x01")
    with pytest.raises(ServiceCallerError) as raised:
        validate_request_payload(
            {
                "base_url": "http://127.0.0.1:28001",
                "route": "/hands.detect",
                "metadata": {},
                "arrays": [{"name": "rgb", "path": str(raw), "shape": [2, 2, 3], "dtype": "uint8"}],
                "timeout_s": 5,
                "output_dir": str(tmp_path / "out"),
            }
        )
    assert raised.value.code == "array_size_mismatch"


def test_profile_records_routes_without_health_claim() -> None:
    profile = json.loads(Path("configs/feishu_ray_services.json").read_text(encoding="utf-8"))
    assert profile["state"] == {"network_scope": "A800 server-local", "health_claim": False}
    assert profile["health_route"] == "/-/healthz"
    assert profile["services"]["hands_wilor"]["routes"] == ["/hands.detect"]
    assert profile["services"]["wilor"]["routes"] == ["/wilor.reconstruct"]
    assert profile["services"]["droid"]["routes"] == ["/droid.create_session", "/droid.push_frame", "/droid.finalize"]
    assert profile["services"]["hawor"]["routes"] == ["/hawor.infer_tracks", "/hawor_infiller.fill"]
    assert profile["services"]["cosmos3"]["routes"] == ["/cosmos3.reason"]
    ports = [profile["services"][name]["base_url"].rsplit(":", 1)[1] for name in ["unidepth", "hands_wilor", "wilor", "droid", "hawor", "cosmos3"]]
    assert ports == ["28000", "28001", "28004", "28002", "28003", "28006"]
