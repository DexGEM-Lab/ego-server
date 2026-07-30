"""Tests for the Cosmos3 (vLLM) Ray Serve model API slice (GPU6).

CPU-only: never imports Ray and never loads vLLM. They cover the model-native
contract (bounded binary media, no caller paths, server-owned revision), the
vLLM-native multimodal boundary (PIL image / numpy video passed to the engine),
truthful monotonic timings, bounded media admission/rejection, the model-load-once
invariant, multipart media transport, the HTTP client, the corrected native-GPU6
lifecycle (components 26800-26806, workers 26900-26931, Serve HTTP 28006, CUDA 6,
Python 3.13 interpreter), and the deployment import-path shape.
"""
from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from ego_annotation.serving.client import HttpModelServiceClient
from ego_annotation.serving.contracts import (
    ContractValidationError,
    Cosmos3MediaItem,
    Cosmos3Request,
    Cosmos3Response,
    ErrorCode,
    GenerationControls,
    Ownership,
    ServiceError,
)
from ego_annotation.serving.cosmos3 import (
    Cosmos3Adapter,
    Cosmos3ModelConfig,
    _apply_chat_template,
    build_cosmos3_model_config,
    decode_media,
    _fake_sampling_params,
)
from ego_annotation.serving.lifecycle import (
    COSMOS3_SERVE_HTTP_PORT,
    COMMITTED_GPU_GROUPS,
    RAY_VERSION,
    cosmos3_gpu_group,
    cosmos3_serve_config,
)
from ego_annotation.serving.transport import (
    build_cosmos3_request,
    build_cosmos3_response,
    parse_cosmos3_request,
    parse_cosmos3_response,
)


REVISION = "cosmos3-nano:vllm-0.19.1"


def _png_bytes(color: tuple[int, int, int] = (220, 30, 30), size: int = 8) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mp4_bytes(num_frames: int = 4, size: int = 8) -> bytes:
    """Synthesize a tiny valid mp4 via imageio if available; else skip-dependent path."""
    import imageio.v2 as imageio

    frames = [np.full((size, size, 3), (10 + i * 20) % 255, dtype=np.uint8) for i in range(num_frames)]
    buf = io.BytesIO()
    writer = imageio.get_writer(buf, format="mp4", fps=4, codec="libx264")
    for f in frames:
        writer.append_data(f)
    writer.close()
    return buf.getvalue()


def _has_video_decoder() -> bool:
    try:
        import imageio  # noqa: F401
        return True
    except Exception:
        try:
            import decord  # noqa: F401
            return True
        except Exception:
            return False


def make_request(
    request_id: str,
    *,
    prompt: str | None = "Describe this image in one short sentence.",
    messages: tuple[tuple[str, Any], ...] = (),
    media: tuple[Cosmos3MediaItem, ...] = (),
    generation: GenerationControls | None = None,
    job_id: str = "job-a",
) -> Cosmos3Request:
    return Cosmos3Request(
        ownership=Ownership(
            request_id=request_id,
            job_id=job_id,
            item_id=f"item-{request_id}",
            stage_id="cosmos3.reason",
            source_id=f"source-{request_id}",
            source_timestamp_s=1.25,
        ),
        prompt=prompt,
        messages=messages,
        media=media,
        generation=generation or GenerationControls(max_tokens=32, temperature=0.0),
    )


def make_media(kind: str = "image", *, media_type: str | None = None, source_index: int = 0) -> Cosmos3MediaItem:
    if kind == "image":
        return Cosmos3MediaItem(kind="image", data=_png_bytes(), media_type=media_type or "image/png", source_index=source_index)
    return Cosmos3MediaItem(kind="video", data=_mp4_bytes(), media_type=media_type or "video/mp4", source_index=source_index)


def make_config(**overrides: Any) -> Cosmos3ModelConfig:
    return build_cosmos3_model_config(
        model_source="nvidia/Cosmos3-Nano",
        model_revision=REVISION,
        max_media_per_request=4,
        max_media_bytes_per_item=4 * 1024 * 1024,
        max_media_bytes_per_request=8 * 1024 * 1024,
        **overrides,
    )


@dataclass
class FakeCompletion:
    text: str
    finish_reason: str = "stop"
    stop_reason: str | None = None
    token_ids: list[int] | None = None


@dataclass
class FakeMetrics:
    num_prompt_tokens: int = 12
    num_generation_tokens: int = 6


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletion]
    finished: bool = True
    prompt_token_ids: list[int] | None = None
    metrics: FakeMetrics | None = None


class FakeCosmos3Backend:
    """Fake vLLM AsyncLLMEngine asserting the multimodal boundary and resident revision."""

    def __init__(self, config: Cosmos3ModelConfig, loads: list[int]) -> None:
        self.config = config
        self.loads = loads
        self.loads.append(1)  # constructed once at replica startup
        self.generate_calls: list[dict[str, Any]] = []

        class Tokenizer:
            @staticmethod
            def apply_chat_template(_messages: Any, **_kwargs: Any) -> dict[str, list[int]]:
                return {"input_ids": [11, 12, 13]}

        self._tokenizer = Tokenizer()

    @property
    def model_revision(self) -> str:
        return self.config.model_revision

    async def generate(self, prompt: Any, sampling_params: Any, request_id: str, *, multi_modal_data=None):
        # Record the boundary: prompt is text str or token-id list; media is PIL/np.
        call = {"prompt": prompt, "multi_modal_data": multi_modal_data, "request_id": request_id}
        self.generate_calls.append(call)
        # Assert the model-native multimodal boundary when media is present.
        if multi_modal_data is not None:
            if "image" in multi_modal_data:
                from PIL import Image

                imgs = multi_modal_data["image"]
                if not isinstance(imgs, list):
                    imgs = [imgs]
                for img in imgs:
                    assert isinstance(img, Image.Image), f"image media must be PIL.Image, got {type(img)}"
            if "video" in multi_modal_data:
                vids = multi_modal_data["video"]
                if not isinstance(vids, list):
                    vids = [vids]
                for v in vids:
                    arr, meta = v
                    assert isinstance(arr, np.ndarray), f"video must be np.ndarray, got {type(arr)}"
                    assert arr.ndim == 4 and arr.shape[-1] == 3
        # Yield incremental then final output (mirror vLLM's async generator).
        yield FakeRequestOutput(
            outputs=[FakeCompletion(text="", token_ids=[])],
            finished=False,
            metrics=FakeMetrics(num_prompt_tokens=12, num_generation_tokens=0),
        )
        yield FakeRequestOutput(
            outputs=[FakeCompletion(text="A solid red background.", token_ids=[1, 2, 3, 4, 5, 6])],
            finished=True,
            prompt_token_ids=list(range(12)),
            metrics=FakeMetrics(num_prompt_tokens=12, num_generation_tokens=6),
        )

    def engine_running_requests(self) -> int:
        return 1


def make_adapter(loads: list[int], *, config: Cosmos3ModelConfig | None = None) -> Cosmos3Adapter:
    cfg = config or make_config()

    def factory(c: Cosmos3ModelConfig) -> FakeCosmos3Backend:
        return FakeCosmos3Backend(c, loads)

    # Use the vLLM-free sampling params factory so tests never import vllm.
    return Cosmos3Adapter(cfg, backend_factory=factory, sampling_params_factory=_fake_sampling_params)


# --- REQ1: model-native contract, no caller paths, server-owned revision ------------

def test_request_rejects_filesystem_fields_recursively() -> None:
    with pytest.raises(ContractValidationError, match="filesystem paths"):
        Cosmos3Request.from_wire(
            {"ownership": {"request_id": "r", "job_id": "j", "item_id": "i", "stage_id": "s", "source_id": "src"},
             "prompt": "x", "media": [{"kind": "image", "data_b64": "AA==", "media_type": "image/png", "rgb_path": "/tmp/x.png"}]}
        )


def test_request_requires_prompt_or_messages() -> None:
    o = Ownership("r", "j", "i", "s", "src")
    with pytest.raises(ContractValidationError, match="prompt or messages"):
        Cosmos3Request(ownership=o)


def test_request_rejects_both_prompt_and_messages() -> None:
    o = Ownership("r", "j", "i", "s", "src")
    with pytest.raises(ContractValidationError, match="not both"):
        Cosmos3Request(ownership=o, prompt="hi", messages=(("user", "hi"),))


def test_request_carries_no_model_revision_field() -> None:
    # The contract is server-owned: there is no model_revision field on the request.
    fields = Cosmos3Request.__dataclass_fields__
    assert "model_revision" not in fields, "cosmos3 request must not carry a caller model_revision"


def test_generation_controls_clamp_bounds() -> None:
    with pytest.raises(ContractValidationError, match="max_tokens"):
        GenerationControls(max_tokens=99999)
    with pytest.raises(ContractValidationError, match="temperature"):
        GenerationControls(temperature=5.0)
    with pytest.raises(ContractValidationError, match="top_p"):
        GenerationControls(top_p=2.0)


# --- REQ2: bounded binary media admission -------------------------------------------

def test_admission_rejects_too_many_media_items() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    media = tuple(make_media() for _ in range(5))  # bound is 4
    resp = asyncio.run(adapter.infer(make_request("r", media=media)))
    assert resp.error is not None and resp.error.code is ErrorCode.VALIDATION
    assert "media items" in resp.error.message
    # No forward ran; model loaded once.
    assert loads == [1]
    assert adapter._backend.generate_calls == []


def test_admission_rejects_oversized_media_item() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    big = Cosmos3MediaItem(kind="image", data=b"\x00" * (5 * 1024 * 1024), media_type="image/png")
    resp = asyncio.run(adapter.infer(make_request("r", media=(big,))))
    assert resp.error is not None and resp.error.code is ErrorCode.VALIDATION
    assert "per-item bound" in resp.error.message


def test_admission_rejects_oversized_request_total() -> None:
    loads: list[int] = []
    # 3 items of 3 MiB each = 9 MiB > 8 MiB per-request bound.
    cfg = build_cosmos3_model_config(model_revision=REVISION, max_media_per_request=4, max_media_bytes_per_item=4 * 1024 * 1024, max_media_bytes_per_request=8 * 1024 * 1024)
    adapter = make_adapter(loads, config=cfg)
    media = tuple(Cosmos3MediaItem(kind="image", data=b"\x00" * (3 * 1024 * 1024), media_type="image/png") for _ in range(3))
    resp = asyncio.run(adapter.infer(make_request("r", media=media)))
    assert resp.error is not None and resp.error.code is ErrorCode.VALIDATION
    assert "per-request bound" in resp.error.message


def test_media_kind_and_media_type_validated() -> None:
    with pytest.raises(ContractValidationError, match="kind"):
        Cosmos3MediaItem(kind="audio", data=b"x", media_type="audio/wav")
    with pytest.raises(ContractValidationError, match="image media_type"):
        Cosmos3MediaItem(kind="image", data=b"x", media_type="audio/wav")


# --- REQ3: vLLM-native multimodal boundary (PIL image / numpy video) ----------------

def test_decode_media_image_yields_pil_image() -> None:
    media = (make_media("image"),)
    mmd = decode_media(media, lambda v: v)
    assert mmd is not None
    from PIL import Image
    assert isinstance(mmd["image"], Image.Image)


def test_decode_media_video_yields_numpy_array_and_metadata() -> None:
    if not _has_video_decoder():
        pytest.skip("no imageio/decord available locally for video decode")
    media = (make_media("video"),)
    mmd = decode_media(media, lambda v: v)
    assert mmd is not None
    arr, meta = mmd["video"]
    assert isinstance(arr, np.ndarray) and arr.ndim == 4
    assert "num_frames" in meta


def test_adapter_passes_pil_image_to_engine() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("img", media=(make_media("image"),))))
    assert resp.result is not None
    from PIL import Image
    mmd = adapter._backend.generate_calls[0]["multi_modal_data"]
    assert isinstance(mmd["image"], Image.Image)
    assert adapter._backend.generate_calls[0]["prompt"] == [11, 12, 13]


def test_prompt_media_template_inserts_kind_placeholders_and_extracts_batch_encoding() -> None:
    captured: dict[str, Any] = {}

    class Tokenizer:
        def apply_chat_template(self, messages: Any, **_kwargs: Any) -> dict[str, list[int]]:
            captured["messages"] = messages
            return {"input_ids": [7, 8]}

    request = make_request("template", media=(make_media("image"),))
    text, token_ids = _apply_chat_template(request, Tokenizer())
    assert text == "" and token_ids == [7, 8]
    assert captured["messages"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": request.prompt},
    ]


def test_adapter_passes_numpy_video_to_engine() -> None:
    if not _has_video_decoder():
        pytest.skip("no imageio/decord available locally for video decode")
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("vid", media=(make_media("video"),))))
    assert resp.result is not None
    mmd = adapter._backend.generate_calls[0]["multi_modal_data"]
    arr, meta = mmd["video"]
    assert isinstance(arr, np.ndarray)


# --- REQ4: server-owned revision, model-load-once, truthful timings -----------------

def test_results_carry_only_server_owned_revision() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("r", media=(make_media(),))))
    assert resp.result.model_revision == REVISION


def test_model_loads_once_and_not_per_request() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    asyncio.run(adapter.infer(make_request("a")))
    asyncio.run(adapter.infer(make_request("b")))
    asyncio.run(adapter.infer(make_request("c")))
    assert loads == [1]  # constructed once
    assert adapter.status().model_load_count == 1
    assert len(adapter._backend.generate_calls) == 3


def test_empty_backend_exception_becomes_parseable_model_error() -> None:
    class EmptyErrorBackend(FakeCosmos3Backend):
        async def generate(self, *args: Any, **kwargs: Any):
            raise TimeoutError()
            yield  # pragma: no cover - preserves async-generator shape

    loads: list[int] = []
    cfg = make_config()
    adapter = Cosmos3Adapter(
        cfg,
        backend_factory=lambda c: EmptyErrorBackend(c, loads),
        sampling_params_factory=_fake_sampling_params,
    )
    response = asyncio.run(adapter.infer(make_request("empty-error")))
    assert response.error is not None
    assert response.error.code is ErrorCode.MODEL_FAILURE
    assert response.error.message == "TimeoutError: TimeoutError()"


def test_trace_records_truthful_monotonic_timings() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("r")))
    trace = resp.result.trace
    assert trace.admitted_monotonic_s <= trace.dispatched_monotonic_s
    assert trace.dispatched_monotonic_s <= trace.forward_started_monotonic_s
    assert trace.forward_started_monotonic_s <= trace.completed_monotonic_s
    assert trace.model_load_count == 1
    assert trace.request_count == 1


def test_result_token_counts_and_timings_present() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("r")))
    r = resp.result
    assert r.text == "A solid red background."
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 12 and r.completion_tokens == 6 and r.total_tokens == 18
    for key in ("queue_wait_s", "prefill_s", "time_to_first_token_s", "decode_s", "e2e_s"):
        assert key in r.timings


def test_deployment_status_reports_cosmos3_identity() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    status = adapter.status()
    assert status.deployment_name == "cosmos3.reason"
    assert status.assigned_gpu == 6
    assert status.loaded_models == (REVISION,)
    wire = status.to_wire()
    assert "admitted_pending" in wire and "queue_depth" not in wire


# --- REQ5: multipart media transport ------------------------------------------------

def test_cosmos3_request_round_trip_preserves_metadata_and_media_binary() -> None:
    png = _png_bytes()
    metadata = {"ownership": {"request_id": "r1"}, "prompt": "hi", "generation": {"max_tokens": 8}}
    body, content_type = build_cosmos3_request(metadata, [(png, "image", "image/png", 0)])
    assert content_type.startswith("multipart/form-data; boundary=egocosmos3-")
    parsed_meta, media_items = parse_cosmos3_request(body, content_type)
    assert parsed_meta["prompt"] == "hi"
    assert len(media_items) == 1
    assert media_items[0]["kind"] == "image" and media_items[0]["media_type"] == "image/png"
    assert media_items[0]["data"] == png


def test_cosmos3_response_round_trip_preserves_result_metadata() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("r")))
    body, content_type = build_cosmos3_response(resp.to_wire())
    wire = parse_cosmos3_response(body, content_type)
    rebuilt = Cosmos3Response.from_wire(wire)
    assert rebuilt.result.text == "A solid red background."
    assert rebuilt.result.model_revision == REVISION


def test_cosmos3_response_parses_plain_json_fallback() -> None:
    import json
    loads: list[int] = []
    adapter = make_adapter(loads)
    resp = asyncio.run(adapter.infer(make_request("r")))
    wire = parse_cosmos3_response(json.dumps(resp.to_wire()).encode(), "application/json")
    assert wire["result"]["text"] == "A solid red background."


# --- REQ6: HTTP client sends multipart media and surfaces backpressure --------------

@dataclass
class FakeCosmos3HttpResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]


class FakeCosmos3HttpTransport:
    def __init__(self, response: FakeCosmos3HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> FakeCosmos3HttpResponse:
        self.calls.append((url, content, headers))
        return self.response


def test_http_client_sends_multipart_media_binary() -> None:
    req = make_request("http", media=(make_media(),))
    transport = FakeCosmos3HttpTransport(FakeCosmos3HttpResponse(503, b"", {}))
    resp = asyncio.run(HttpModelServiceClient("http://serve", transport).reason_cosmos3(req))
    assert transport.calls[0][0] == "http://serve/cosmos3.reason"
    assert transport.calls[0][2]["Content-Type"].startswith("multipart/form-data; boundary=egocosmos3-")
    # The media binary travels as a multipart part, not base64 JSON.
    assert b"media_0" in transport.calls[0][1]
    assert b'"data_b64"' not in transport.calls[0][1]
    assert resp.error is not None and resp.error.code is ErrorCode.BACKPRESSURE and resp.error.retryable


def test_http_client_parses_cosmos3_result_response() -> None:
    loads: list[int] = []
    adapter = make_adapter(loads)
    server_resp = asyncio.run(adapter.infer(make_request("r")))
    body, content_type = build_cosmos3_response(server_resp.to_wire())
    req = make_request("http")
    transport = FakeCosmos3HttpTransport(FakeCosmos3HttpResponse(200, body, {"Content-Type": content_type}))
    resp = asyncio.run(HttpModelServiceClient("http://serve", transport).reason_cosmos3(req))
    assert resp.result is not None
    assert resp.result.text == "A solid red background."
    assert resp.result.completion_tokens == 6


# --- REQ7: GPU6 lifecycle, ports, interpreter, serve config -------------------------

def test_cosmos3_gpu6_uses_native_num_gpus_and_python313_interpreter() -> None:
    g = cosmos3_gpu_group()
    assert g.gpu_id == 6
    assert g.adapter_implemented is True
    assert g.ray_actor_options == {"num_gpus": 1}
    assert g.interpreter == "/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python"
    assert g.logical_apis == ("cosmos3.reason",)


def test_cosmos3_lifecycle_ports_and_cpu_cap_match_committed_topology() -> None:
    g = cosmos3_gpu_group()
    lc = g.lifecycle
    assert lc.gpu_id == 6
    assert lc.num_gpus == 1
    assert lc.num_cpus == 8  # CPU cap appropriate to vLLM async scheduling
    assert lc.ray_version == RAY_VERSION == "2.55.1"
    assert lc.temp_dir == "/tmp/ray-ego-serve-cosmos3"
    cmd = lc.startup_command("ego-cosmos3")
    assert "CUDA_VISIBLE_DEVICES=6" in cmd
    assert "--num-gpus=1" in cmd
    assert "--num-cpus=8" in cmd
    assert "--port=26801" in cmd  # gcs port (dashboard 26800 + 1)
    assert "--dashboard-port=26800" in cmd
    assert "--worker-port-list=26900,26901,26902,26903,26904,26905,26906,26907,26908,26909,26910,26911,26912,26913,26914,26915,26916,26917,26918,26919,26920,26921,26922,26923,26924,26925,26926,26927,26928,26929,26930,26931" in cmd
    assert "--worker-port-list=26900-26931" not in cmd
    assert "cosmos3_ray_serve/standalone/.venv/bin/python" in cmd


def test_cosmos3_component_and_worker_ports_disjoint_and_in_committed_ranges() -> None:
    g = cosmos3_gpu_group()
    ports = g.lifecycle.ports.all_ports()
    assert len(set(ports)) == len(ports)
    components = (
        g.lifecycle.ports.dashboard_port,
        g.lifecycle.ports.gcs_port,
        g.lifecycle.ports.object_manager_port,
        g.lifecycle.ports.node_manager_port,
        g.lifecycle.ports.ray_client_server_port,
        g.lifecycle.ports.dashboard_agent_listen_port,
        g.lifecycle.ports.dashboard_agent_grpc_port,
    )
    assert sorted(components) == list(range(26800, 26807))
    workers = [int(port) for port in g.lifecycle.ports.worker_port_list.split(",")]
    assert workers == list(range(26900, 26932))
    assert g.lifecycle.ports.serve_http_port == 28006


def test_cosmos3_ports_disjoint_across_all_clusters_and_serve_http() -> None:
    all_ports: list[int] = []
    for group in COMMITTED_GPU_GROUPS:
        all_ports.extend(group.lifecycle.ports.all_ports())
    assert len(set(all_ports)) == len(all_ports)
    # HTTP is part of the canonical port inventory and appears exactly once.
    assert all_ports.count(COSMOS3_SERVE_HTTP_PORT) == 1


def test_cosmos3_serve_config_points_at_deployment_module_with_http_port_28006() -> None:
    cfg = cosmos3_serve_config()
    assert cfg["http_options"]["port"] == 28006
    app_spec = cfg["applications"][0]
    assert app_spec["import_path"] == "ego_annotation.serving.cosmos3_deployment:app"
    assert "runtime_env" not in app_spec  # local paths are invalid in `serve deploy` configs
    startup = cosmos3_gpu_group().lifecycle.startup_command()
    assert "PYTHONPATH=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_runtime/current" in startup
    assert "AUTOSCALER_METRIC_PORT=26811" in startup
    assert "DASHBOARD_METRIC_PORT=26812" in startup
    assert "--metrics-export-port=26810" in startup
    deployment = app_spec["deployments"][0]
    assert deployment["name"] == "cosmos3.reason"
    assert deployment["ray_actor_options"] == {"num_gpus": 1}
    assert deployment["num_replicas"] == 1


# --- REQ8: deployment import-path shape (Ray-free inspection) -----------------------

def test_cosmos3_deployment_module_exposes_bound_application_and_no_serve_batch() -> None:
    # Ray is not installed locally; verify the module path/symbol via source inspection.
    import importlib.util
    import os

    spec = importlib.util.find_spec("ego_annotation.serving.cosmos3_deployment")
    assert spec is not None
    with open(spec.origin) as handle:
        source = handle.read()
    assert "from ray import serve" in source
    assert "@serve.deployment(" in source
    assert "name=\"cosmos3.reason\"" in source
    assert "num_gpus=1" in source
    # vLLM owns batching: the cosmos3 deployment must NOT use @serve.batch as a decorator.
    # (the substring may appear in docstrings; check it is not used as a decorator.)
    import re
    assert not re.search(r"^\s*@serve\.batch\b", source, re.MULTILINE), \
        "cosmos3 must rely on vLLM continuous batching, not @serve.batch"
    assert "app: Any = Cosmos3Deployment.bind()" in source
    # Every deployment uses the shared binary-safe multipart ASGI response path.
    assert "from starlette.responses import Response" in source
    assert "multipart_asgi_response" in source


def test_cosmos3_adapter_imports_do_not_require_ray() -> None:
    from ego_annotation.serving.cosmos3 import Cosmos3Adapter  # noqa: F401
    from ego_annotation.serving.client import HttpModelServiceClient  # noqa: F401
    from ego_annotation.serving.transport import build_cosmos3_request  # noqa: F401
