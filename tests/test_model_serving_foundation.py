"""Tests for the corrected Ray Serve UniDepth model API foundation.

These are CPU-only tests: they never import Ray and never load a model. They cover
the corrected contract (uint8-only RGB), the BCHW backend boundary, real-output
squeeze shapes, resident-revision ownership, one-forward batch tracing, multipart
binary transport, nested ObjectRef lazy resolution, the corrected native-GPU
lifecycle, and the deployment import-path shape.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from ego_annotation.serving.batching import BatchPolicy, assert_one_forward, canonical_batch_size_fn
from ego_annotation.serving.client import HttpModelServiceClient, InClusterModelServiceClient
from ego_annotation.serving.contracts import (
    ContractValidationError,
    ErrorCode,
    ImageSize,
    Ownership,
    PixelTransform,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
    reject_filesystem_fields,
)
from ego_annotation.serving.lifecycle import COMMITTED_GPU_GROUPS, RAY_VERSION, unidepth_serve_config
from ego_annotation.serving.router import ModelServiceRouter
from ego_annotation.serving.transport import (
    build_multipart_request,
    build_multipart_response,
    lazy_resolve_object_ref,
    multipart_asgi_response,
    parse_multipart_request,
    parse_multipart_response,
)
from ego_annotation.serving.unidepth import (
    UniDepthAdapter,
    UniDepthModelConfig,
    build_unidepth_model_config,
    squeeze_unidepth_outputs,
)


REVISION = "unidepth-v2-vitl14-corrected"
H, W = 4, 6


def make_request(
    request_id: str,
    *,
    job_id: str = "job-a",
    pixels: int = 255,
    shape: tuple[int, int] = (H, W),
    binary: bytes | None = None,
    dtype: str = "uint8",
    model_revision: str = REVISION,
    options: dict[str, str] | None = None,
) -> UniDepthRequest:
    height, width = shape
    if dtype == "uint8":
        rgb = np.full((height, width, 3), pixels, dtype=np.uint8)
    else:
        rgb = np.full((height, width, 3), pixels, dtype=np.float32)
    return UniDepthRequest(
        ownership=Ownership(
            request_id=request_id,
            job_id=job_id,
            item_id=f"frame-{request_id}",
            stage_id="unidepth.infer",
            source_id=f"source-{request_id}",
            source_timestamp_s=1.25,
        ),
        rgb=TensorPayload(data=rgb.tobytes() if binary is None else binary, shape=rgb.shape, dtype=dtype),
        spatial=SpatialMetadata(
            source_size=ImageSize(width=width, height=height),
            model_size=ImageSize(width=width, height=height),
            color_space="RGB",
            pixel_transform=PixelTransform.identity(),
            K_px=((12.0, 0.0, 1.0), (0.0, 13.0, 1.0), (0.0, 0.0, 1.0)),
        ),
        model_revision=model_revision,
        options=tuple(sorted(options.items())) if options else (),
    )


def make_config(**overrides: Any) -> UniDepthModelConfig:
    return build_unidepth_model_config(
        checkpoint="server-owned-checkpoint",
        model_revision=REVISION,
        canonical_height=H,
        canonical_width=W,
        batch_policy=BatchPolicy(max_batch_size=8, batch_wait_timeout_s=0.01, max_queued_requests=2),
        **overrides,
    )


class FakeUniDepth:
    """Fake backend that asserts the BCHW contract and emits real-shaped outputs."""

    def __init__(self, loads: list[int]) -> None:
        loads.append(1)
        self.batches: list[np.ndarray] = []

    def infer(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        # The real backend receives contiguous [B,C,H,W] uint8; assert that boundary.
        assert rgb.ndim == 4, f"backend must receive 4D BCHW, got {rgb.ndim}D"
        assert rgb.shape[1] == 3, f"backend channel dim must be 3 (BCHW), got shape {rgb.shape}"
        assert rgb.dtype == np.uint8, f"backend must receive uint8, got {rgb.dtype}"
        self.batches.append(rgb.copy())
        count = rgb.shape[0]
        height, width = rgb.shape[2], rgb.shape[3]
        # Real UniDepth shapes: depth [B,1,H,W], intrinsics [B,3,3], confidence [B,1,H,W].
        return {
            "depth": np.full((count, 1, height, width), 0.5, dtype=np.float32),
            "intrinsics": np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy(),
            "confidence": np.full((count, 1, height, width), 0.75, dtype=np.float32),
        }


def make_adapter(fake_holder: list[FakeUniDepth], loads: list[int], *, config: UniDepthModelConfig | None = None) -> UniDepthAdapter:
    def factory(_: UniDepthModelConfig) -> FakeUniDepth:
        fake = FakeUniDepth(loads)
        fake_holder.append(fake)
        return fake

    return UniDepthAdapter(config or make_config(), backend_factory=factory)


# --- REQ1: uint8-only contract + float rejection -------------------------------------

@pytest.mark.parametrize("bad_dtype", ["float32", "float64"])
def test_contract_rejects_float_rgb_dtype(bad_dtype: str) -> None:
    # Float RGB is range-ambiguous and rejected at the contract boundary.
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    with pytest.raises(ContractValidationError, match="uint8"):
        UniDepthRequest(
            ownership=Ownership("r", "j", "i", "s", "src"),
            rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype=bad_dtype),
            spatial=SpatialMetadata(
                source_size=ImageSize(width=W, height=H),
                model_size=ImageSize(width=W, height=H),
                color_space="RGB",
                pixel_transform=PixelTransform.identity(),
            ),
            model_revision=REVISION,
        )


# --- REQ1: BCHW backend boundary + output shapes --------------------------------------

def test_adapter_passes_contiguous_bchw_to_backend_without_dividing_by_255() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)

    responses = asyncio.run(adapter.infer_batch([
        adapter.admit(make_request("one", pixels=255)),
        adapter.admit(make_request("two", pixels=128)),
    ]))

    assert loads == [1]
    assert len(fakes[0].batches) == 1
    batch = fakes[0].batches[0]
    # BCHW shape and no /255: max value preserved as uint8.
    assert batch.shape == (2, 3, H, W)
    assert batch.dtype == np.uint8
    assert int(batch[0].max()) == 255
    assert int(batch[1].max()) == 128
    # One forward only.
    assert all(r.result is not None for r in responses)
    assert responses[0].result.trace.forward_count == 1
    assert responses[0].result.trace.request_count == 2


def test_real_unidepth_output_shapes_are_squeezed_to_per_image() -> None:
    # Directly exercise the squeeze helper with real [B,1,H,W] / [B,3,3] outputs.
    count, height, width = 3, 5, 7
    outputs = {
        "depth": np.full((count, 1, height, width), 1.0, dtype=np.float32),
        "intrinsics": np.broadcast_to(np.eye(3, dtype=np.float32), (count, 3, 3)).copy(),
        "confidence": np.full((count, 1, height, width), 0.8, dtype=np.float32),
    }
    depth, intrinsics, confidence = squeeze_unidepth_outputs(outputs, count, height, width)
    assert depth.shape == (count, height, width)        # channel-1 squeezed
    assert confidence.shape == (count, height, width)   # channel-1 squeezed
    assert intrinsics.shape == (count, 3, 3)


def test_squeeze_rejects_wrong_backend_output_shapes() -> None:
    with pytest.raises(ContractValidationError, match="depth"):
        squeeze_unidepth_outputs(
            {"depth": np.zeros((2, 5, 7)), "intrinsics": np.zeros((2, 3, 3)),
             "confidence": np.zeros((2, 1, 5, 7))}, 2, 5, 7,
        )
    with pytest.raises(ContractValidationError, match="intrinsics"):
        squeeze_unidepth_outputs(
            {"depth": np.zeros((2, 1, 5, 7)), "intrinsics": np.zeros((2, 3, 4)),
             "confidence": np.zeros((2, 1, 5, 7))}, 2, 5, 7,
        )


def test_squeeze_rejects_non_finite_or_non_positive_depth() -> None:
    # Finite-positive depth/K/validity semantics: NaN/Inf depth and non-positive
    # depth or focal length are model/ABI failures, not usable measurements.
    base = {"intrinsics": np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
            "confidence": np.full((2, 1, 5, 7), 0.5, dtype=np.float32)}
    nan_depth = np.full((2, 1, 5, 7), 1.0, dtype=np.float32)
    nan_depth[0, 0, 0, 0] = np.nan
    with pytest.raises(ContractValidationError, match="depth must be finite"):
        squeeze_unidepth_outputs({**base, "depth": nan_depth}, 2, 5, 7)
    zero_depth = np.zeros((2, 1, 5, 7), dtype=np.float32)
    with pytest.raises(ContractValidationError, match="depth must be strictly positive"):
        squeeze_unidepth_outputs({**base, "depth": zero_depth}, 2, 5, 7)


def test_squeeze_rejects_non_positive_focal_length() -> None:
    bad_k = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
    bad_k[0, 0, 0] = -1.0
    with pytest.raises(ContractValidationError, match="positive fx/fy"):
        squeeze_unidepth_outputs(
            {"depth": np.full((2, 1, 5, 7), 1.0, dtype=np.float32),
             "intrinsics": bad_k,
             "confidence": np.full((2, 1, 5, 7), 0.5, dtype=np.float32)}, 2, 5, 7,
        )


def test_adapter_result_depth_k_and_confidence_shapes_are_per_image() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_request("one"))]))
    result = responses[0].result
    assert result is not None
    assert result.depth_m.shape == (H, W)        # squeezed from [B,1,H,W]
    assert result.confidence.shape == (H, W)     # squeezed from [B,1,H,W]
    assert result.K_px.shape == (3, 3)           # from [B,3,3]


# --- REQ2: resident config owns model revision ----------------------------------------

def test_admission_rejects_revision_mismatch_before_batching() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    wrong = make_request("wrong-rev", model_revision="some-other-revision")
    with pytest.raises(ContractValidationError, match="model_revision"):
        adapter.admit(wrong)
    # The model loads once at replica startup (one model_load_count); no forward ran.
    assert loads == [1]
    assert fakes[0].batches == []


def test_results_carry_only_the_configured_revision() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_request("one"))]))
    assert responses[0].result.model_revision == REVISION


# --- REQ3: one-forward batch trace + weighted batching helpers ------------------------

def test_canonical_batch_size_fn_counts_normalized_work_units() -> None:
    items = [object(), object(), object()]
    assert canonical_batch_size_fn(items) == 3
    assert canonical_batch_size_fn([]) == 0


def test_assert_one_forward_rejects_oversized_callback() -> None:
    policy = BatchPolicy(max_batch_size=4, batch_wait_timeout_s=0.01, max_queued_requests=2)
    assert_one_forward([1, 2, 3, 4], policy=policy)
    with pytest.raises(ContractValidationError, match="multiple forwards"):
        assert_one_forward([1, 2, 3, 4, 5], policy=policy)


def test_batch_trace_records_one_forward_and_truthful_monotonic_timings() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    responses = asyncio.run(adapter.infer_batch([
        adapter.admit(make_request("a")),
        adapter.admit(make_request("b")),
    ]))
    trace = responses[0].result.trace
    assert trace.forward_count == 1
    assert trace.request_count == 2
    assert trace.effective_work_units == 2
    # Monotonic ordering enforced by the contract.
    assert trace.admitted_monotonic_s <= trace.dispatched_monotonic_s
    assert trace.dispatched_monotonic_s <= trace.forward_started_monotonic_s
    assert trace.forward_started_monotonic_s <= trace.completed_monotonic_s


def test_invalid_binary_is_isolated_at_admission_before_any_forward() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    # A request with binary too short for the declared shape fails admission decode.
    with pytest.raises(ContractValidationError):
        adapter.admit(make_request("broken", binary=b"short"))
    # The healthy request still gets exactly one forward.
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_request("healthy"))]))
    assert responses[0].result is not None
    assert len(fakes[0].batches) == 1


# --- REQ4: deployment status exposes qualified admitted_pending, not fake queue depth -

def test_deployment_status_uses_admitted_pending_not_authoritative_queue_depth() -> None:
    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = make_adapter(fakes, loads)
    status = adapter.status()
    assert status.admitted_pending == 0
    wire = status.to_wire()
    assert "admitted_pending" in wire
    assert "queue_depth" not in wire


# --- REQ5: multipart binary transport -------------------------------------------------

def test_multipart_request_round_trip_preserves_metadata_and_rgb_binary() -> None:
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    metadata = {"ownership": {"request_id": "r1"}, "model_revision": REVISION, "rgb_shape": [H, W, 3], "rgb_dtype": "uint8"}
    body, content_type = build_multipart_request(metadata, rgb=rgb.tobytes(), rgb_shape=rgb.shape, rgb_dtype="uint8")
    assert content_type.startswith("multipart/form-data; boundary=egounidepth-")
    parsed_meta, parsed_rgb, parsed_shape, parsed_dtype = parse_multipart_request(body, content_type)
    assert parsed_meta["model_revision"] == REVISION
    assert parsed_shape == (H, W, 3)
    assert parsed_dtype == "uint8"
    assert np.frombuffer(parsed_rgb, dtype=np.uint8).tolist() == rgb.flatten().tolist()


def test_multipart_response_round_trip_preserves_all_array_fields() -> None:
    depth = np.full((H, W), 1.5, dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    conf = np.full((H, W), 0.9, dtype=np.float32)
    body, content_type = build_multipart_response(
        {"ok": True, "trace": {"batch_id": "b1"}},
        {
            "depth_m": (depth.tobytes(), depth.shape, "float32"),
            "K_px": (K.tobytes(), K.shape, "float32"),
            "confidence": (conf.tobytes(), conf.shape, "float32"),
        },
    )
    meta, arrays = parse_multipart_response(body, content_type)
    assert meta["ok"] is True
    assert set(arrays.keys()) == {"depth_m", "K_px", "confidence"}
    assert arrays["depth_m"][1] == (H, W)
    assert arrays["K_px"][1] == (3, 3)
    assert arrays["confidence"][1] == (H, W)


def test_canonical_multipart_asgi_response_preserves_binary_body() -> None:
    payload = bytes([0, 255, 128, 1])
    response = multipart_asgi_response(
        {"ok": True}, {"tensor": (payload, (4,), "uint8")}, status_code=201
    )
    assert response.status_code == 201
    assert response.headers["content-type"].startswith("multipart/form-data; boundary=")
    metadata, arrays = parse_multipart_response(response.body, response.headers["content-type"])
    assert metadata == {"ok": True}
    assert arrays["tensor"] == (payload, (4,), "uint8")


def test_multipart_request_missing_fields_raise() -> None:
    # Build a response body and try to parse it as a request (no rgb part).
    body, content_type = build_multipart_response({"ok": True}, {})
    with pytest.raises(ValueError, match="rgb"):
        parse_multipart_request(body, content_type)


# --- REQ5: nested ObjectRef lazy resolution via injected fake -------------------------

class FakeObjectRef:
    """Duck-typed fake matching real ray.ObjectRef (binary() + is_nil())."""

    def __init__(self, key: str) -> None:
        self.key = key

    def binary(self) -> bytes:
        return b""

    def is_nil(self) -> bool:
        return False


def test_lazy_resolve_handles_nested_object_refs_via_injected_fake() -> None:
    store: dict[str, Any] = {
        "a": FakeObjectRef("b"),
        "b": FakeObjectRef("c"),
        "c": b"final-bytes",
    }

    def fake_get(ref: Any) -> Any:
        return store[ref.key]

    resolved = lazy_resolve_object_ref(FakeObjectRef("a"), fake_get)
    assert resolved == b"final-bytes"


def test_lazy_resolve_passes_through_plain_bytes() -> None:
    def fake_get(_: Any) -> Any:
        raise AssertionError("ray_get must not be called for non-ref values")

    assert lazy_resolve_object_ref(b"plain-bytes", fake_get) == b"plain-bytes"


def test_in_cluster_client_resolves_nested_object_ref_before_dispatch() -> None:
    store: dict[str, Any] = {
        "outer": FakeObjectRef("inner"),
        "inner": b"\x00" * (H * W * 3),
    }

    def fake_get(ref: Any) -> Any:
        return store[ref.key]

    captured: list[UniDepthRequest] = []

    class RemoteInfer:
        def remote(self, request: UniDepthRequest) -> Any:
            async def response() -> UniDepthResponse:
                captured.append(request)
                return UniDepthResponse(
                    ownership=request.ownership,
                    error=ServiceError(ErrorCode.VALIDATION, "resolved", retryable=False, ownership=request.ownership),
                )
            return response()

    class Handle:
        infer = RemoteInfer()

    request = make_request("cluster-req")
    # Replace RGB data with a nested ObjectRef.
    request_with_ref = UniDepthRequest(
        ownership=request.ownership,
        rgb=TensorPayload(data=FakeObjectRef("outer"), shape=request.rgb.shape, dtype=request.rgb.dtype),
        spatial=request.spatial,
        model_revision=request.model_revision,
        options=request.options,
    )
    actual = asyncio.run(
        InClusterModelServiceClient(cast(Any, Handle()), ray_get=fake_get).infer_unidepth(request_with_ref)
    )
    assert actual.error is not None and actual.error.message == "resolved"
    # The handle received resolved binary bytes, not the ObjectRef.
    assert isinstance(captured[0].rgb.data, bytes)


# --- REQ5: HTTP client uses multipart binary transport -------------------------------

@dataclass
class FakeMultipartResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]


class FakeMultipartTransport:
    def __init__(self, response: FakeMultipartResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> FakeMultipartResponse:
        self.calls.append((url, content, headers))
        return self.response


def test_http_client_sends_multipart_binary_and_surfaces_backpressure() -> None:
    request = make_request("http-request")
    transport = FakeMultipartTransport(FakeMultipartResponse(503, b"", {}))
    response = asyncio.run(HttpModelServiceClient("http://serve", transport).infer_unidepth(request))
    assert transport.calls[0][0] == "http://serve/unidepth.infer"
    assert transport.calls[0][2]["Content-Type"].startswith("multipart/form-data")
    # The body is multipart containing the rgb binary, not a JSON/base64 payload.
    assert b"rgb" in transport.calls[0][1]
    assert b'"data_b64"' not in transport.calls[0][1]
    assert response.error is not None
    assert response.error.code is ErrorCode.BACKPRESSURE
    assert response.error.retryable


def test_http_client_parses_multipart_response_arrays() -> None:
    request = make_request("http-request")
    depth = np.full((H, W), 2.0, dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    conf = np.full((H, W), 0.6, dtype=np.float32)
    expected = make_adapter([], []).admit(request)
    trace = expected.request.spatial  # reuse a valid spatial; build a real result below
    from ego_annotation.serving.contracts import BatchTrace, UniDepthResult
    result = UniDepthResult(
        ownership=request.ownership,
        depth_m=TensorPayload(data=depth.tobytes(), shape=depth.shape, dtype="float32"),
        K_px=TensorPayload(data=K.tobytes(), shape=K.shape, dtype="float32"),
        confidence=TensorPayload(data=conf.tobytes(), shape=conf.shape, dtype="float32"),
        spatial=request.spatial,
        model_revision=REVISION,
        trace=BatchTrace(
            batch_id="b1", replica_id="rep", admitted_monotonic_s=1.0, dispatched_monotonic_s=1.0,
            forward_started_monotonic_s=1.0, completed_monotonic_s=2.0,
            effective_work_units=1, request_count=1, forward_count=1, model_load_count=1,
        ),
    )
    body, content_type = build_multipart_response(
        {"result": result.to_wire(), "ownership": request.ownership.to_wire()},
        {
            "depth_m": (depth.tobytes(), depth.shape, "float32"),
            "K_px": (K.tobytes(), K.shape, "float32"),
            "confidence": (conf.tobytes(), conf.shape, "float32"),
        },
    )
    transport = FakeMultipartTransport(FakeMultipartResponse(200, body, {"Content-Type": content_type}))
    response = asyncio.run(HttpModelServiceClient("http://serve", transport).infer_unidepth(request))
    assert response.result is not None
    assert response.result.depth_m.shape == (H, W)
    assert response.result.K_px.shape == (3, 3)
    assert response.result.confidence.shape == (H, W)


# --- REQ6/contract: filesystem field rejection still holds ---------------------------

@pytest.mark.parametrize("payload", [{"rgb_path": "/tmp/frame.png"}, {"nested": {"output_dir": "/tmp/out"}}])
def test_contract_rejects_all_caller_filesystem_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ContractValidationError, match="filesystem paths"):
        reject_filesystem_fields(payload)


# --- REQ7: corrected native-GPU lifecycle ---------------------------------------------

def test_gpu_groups_use_native_num_gpus_not_custom_resource_pinning() -> None:
    for group in COMMITTED_GPU_GROUPS:
        opts = group.ray_actor_options
        assert opts == {"num_gpus": 1}, f"GPU{group.gpu_id} must use native num_gpus=1, got {opts}"
        assert "resources" not in opts, "custom-resource-only pinning must be removed"
        assert "runtime_env" not in opts or "CUDA_VISIBLE_DEVICES" not in opts.get("runtime_env", {}).get("env_vars", {})


def test_unidepth_gpu0_uses_ray_serve_unidepth_interpreter_not_hamer_310() -> None:
    gpu0 = COMMITTED_GPU_GROUPS[0]
    assert gpu0.gpu_id == 0
    assert gpu0.interpreter.endswith("ray_serve_unidepth/bin/python"), gpu0.interpreter
    assert COMMITTED_GPU_GROUPS[1].interpreter.endswith("ray_serve_hands/bin/python")


def test_cluster_lifecycle_records_gpu_ports_temp_dir_ray_version_and_startup() -> None:
    gpu0 = COMMITTED_GPU_GROUPS[0]
    lifecycle = gpu0.lifecycle
    assert lifecycle.gpu_id == 0
    assert lifecycle.num_gpus == 1
    assert lifecycle.num_cpus == 4
    assert lifecycle.ray_version == RAY_VERSION == "2.55.1"
    assert lifecycle.temp_dir.startswith("/tmp/ray-ego-serve")
    cmd = lifecycle.startup_command("ego-unidepth")
    assert "CUDA_VISIBLE_DEVICES=0" in cmd
    assert "--num-gpus=1" in cmd
    assert "--num-cpus=4" in cmd
    assert "--port=" in cmd
    assert "--dashboard-port=" in cmd
    assert "--object-manager-port=" in cmd
    assert "--worker-port-list=" in cmd
    assert "--temp-dir=" in cmd
    assert "ray_serve_unidepth/bin/python" in cmd
    # The corrected startup command emits only Ray 2.55-supported flags.
    assert "--cluster-name=" not in cmd  # not a `ray start` flag
    assert "--ray-client-server-port=" in cmd
    assert "--dashboard-agent-listen-port=" in cmd
    assert "--dashboard-agent-grpc-port=" in cmd
    assert "--node-manager-port=" in cmd


def test_gpu0_port_allocation_matches_committed_layout() -> None:
    gpu0 = COMMITTED_GPU_GROUPS[0]
    ports = gpu0.lifecycle.ports
    # GCS 26000, object manager 26001, node manager 26002, ray client 26003,
    # dashboard 26004, dashboard agents 26005/26006, workers 26100-26131, Serve HTTP 28000.
    assert ports.gcs_port == 26000
    assert ports.object_manager_port == 26001
    assert ports.node_manager_port == 26002
    assert ports.ray_client_server_port == 26003
    assert ports.dashboard_port == 26004
    assert ports.dashboard_agent_listen_port == 26005
    assert ports.dashboard_agent_grpc_port == 26006
    assert ports.serve_http_port == 28000
    worker_ports = [int(p) for p in ports.worker_port_list.split(",")]
    assert worker_ports == list(range(26100, 26132))
    cmd = gpu0.lifecycle.startup_command("ego-unidepth")
    assert "--port=26000" in cmd
    assert "--object-manager-port=26001" in cmd
    assert "--node-manager-port=26002" in cmd
    assert "--ray-client-server-port=26003" in cmd
    assert "--dashboard-port=26004" in cmd
    assert "--dashboard-agent-listen-port=26005" in cmd
    assert "--dashboard-agent-grpc-port=26006" in cmd
    assert "--worker-port-list=26100,26101" in cmd


def test_cluster_component_and_worker_ports_are_disjoint() -> None:
    for group in COMMITTED_GPU_GROUPS:
        ports = group.lifecycle.ports.all_ports()
        assert len(set(ports)) == len(ports), f"GPU{group.gpu_id} ports collide: {ports}"
        # Worker ports must be outside the component block.
        components = (group.lifecycle.ports.dashboard_port, group.lifecycle.ports.gcs_port,
                      group.lifecycle.ports.object_manager_port, group.lifecycle.ports.node_manager_port)
        workers = [p for p in ports if p not in components]
        assert workers, f"GPU{group.gpu_id} must have explicit worker ports"


def test_clusters_use_disjoint_port_allocations() -> None:
    all_ports: list[int] = []
    for group in COMMITTED_GPU_GROUPS:
        all_ports.extend(group.lifecycle.ports.all_ports())
    assert len(set(all_ports)) == len(all_ports), "ports must be disjoint across all clusters"


def test_ray_component_and_worker_ports_do_not_collide_with_public_lane_ports() -> None:
    public_ports = {endpoint.serve_http_port for endpoint in ModelServiceRouter.canonical().endpoints}
    ray_internal_ports = {
        port
        for group in COMMITTED_GPU_GROUPS
        for port in group.lifecycle.ports.all_ports()
        if port != group.lifecycle.ports.serve_http_port
    }
    assert public_ports == {28000, 28001, 28002, 28003, 28004, 28006}
    assert {group.lifecycle.ports.serve_http_port for group in COMMITTED_GPU_GROUPS} == public_ports
    assert public_ports.isdisjoint(ray_internal_ports)


# --- REQ8: deployment import-path shape (Ray-free) ------------------------------------

def test_unidepth_serve_config_points_at_deployment_only_module_with_bound_app() -> None:
    config = unidepth_serve_config()
    app_spec = config["applications"][0]
    assert app_spec["import_path"] == "ego_annotation.serving.deployment:app"
    deployment = app_spec["deployments"][0]
    assert deployment["name"] == "unidepth.infer"
    assert deployment["ray_actor_options"] == {"num_gpus": 1}


def test_deployment_module_exposes_bound_application_import_path() -> None:
    # The import path `module:app` must resolve to a real attribute. Ray is not
    # installed locally, so importing the deployment module is expected to fail at
    # `from ray import serve`; verify the module path and symbol name are correct
    # by inspecting the source without importing Ray.
    import importlib.util
    import os

    spec = importlib.util.find_spec("ego_annotation.serving.deployment")
    assert spec is not None, "deployment module must exist on the import path"
    source_path = os.path.join(os.path.dirname(spec.origin), "deployment.py")
    with open(source_path) as handle:
        source = handle.read()
    # The module imports Ray Serve at top level (deployment-only) and binds an app.
    assert "from ray import serve" in source
    assert "@serve.deployment(" in source
    assert "@serve.batch(" in source
    assert "batch_size_fn=canonical_batch_size_fn" in source
    assert "num_gpus=1" in source
    assert "app: Any = UniDepthDeployment.bind()" in source


def test_ordinary_adapter_imports_do_not_require_ray() -> None:
    # These imports must succeed without Ray installed (the local environment has no Ray).
    from ego_annotation.serving.unidepth import UniDepthAdapter  # noqa: F401
    from ego_annotation.serving.client import HttpModelServiceClient  # noqa: F401
    from ego_annotation.serving.transport import build_multipart_request  # noqa: F401


# --- REQ9: known-working manual checkpoint loader (not from_pretrained) ---------------

def test_load_unidepth_backend_uses_manual_config_safetensors_path() -> None:
    # The loader must read config.json -> UniDepthV2(config) -> safetensors ->
    # load_state_dict(strict=False), not from_pretrained. Verify by inspecting the
    # source (the function imports torch/unidepth only at call time, so it is not
    # executable locally without the GPU env). This guards against regressing to
    # the unverified from_pretrained fallback.
    import importlib.util
    import os

    spec = importlib.util.find_spec("ego_annotation.serving.unidepth")
    source_path = os.path.join(os.path.dirname(spec.origin), "unidepth.py")
    with open(source_path) as handle:
        source = handle.read()
    # Locate the loader function body.
    marker = "def _load_unidepth_backend"
    start = source.index(marker)
    # Slice to the next top-level def/class.
    rest = source[start + len(marker):]
    next_def = rest.find("\ndef ")
    loader_src = rest[:next_def] if next_def != -1 else rest
    assert "config.json" in loader_src
    assert "UniDepthV2(model_config)" in loader_src or "UniDepthV2(config" in loader_src
    assert "safe_open" in loader_src
    assert "load_state_dict" in loader_src and "strict=False" in loader_src
    assert "from_pretrained(" not in loader_src, "loader must not call from_pretrained"


# --- REQ10: in-cluster nested ObjectRef resolution wired into the adapter -------------

def test_adapter_in_cluster_resolver_unwraps_nested_object_ref() -> None:
    # The deployment injects an in-cluster tensor resolver that resolves nested
    # ObjectRef chains via ray.get before byte decoding. Verify the resolver unwraps
    # a nested ref and decodes the resolved bytes to the declared uint8 HWC array.
    from ego_annotation.serving.transport import lazy_resolve_object_ref

    store = {"outer": FakeObjectRef("inner"), "inner": b"\x01" * (H * W * 3)}

    def fake_get(ref: Any) -> Any:
        return store[ref.key]

    def in_cluster_resolver(data: Any, shape: tuple[int, ...], dtype: str) -> Any:
        from ego_annotation.serving.unidepth import _default_tensor_resolver
        resolved = lazy_resolve_object_ref(data, fake_get)
        return _default_tensor_resolver(resolved, shape, dtype)

    fakes: list[FakeUniDepth] = []
    loads: list[int] = []
    adapter = UniDepthAdapter(make_config(), backend_factory=lambda c: (fakes.append(FakeUniDepth(loads)) or fakes[-1]),
                              tensor_resolver=in_cluster_resolver)
    request = make_request("ref-req")
    request_with_ref = UniDepthRequest(
        ownership=request.ownership,
        rgb=TensorPayload(data=FakeObjectRef("outer"), shape=request.rgb.shape, dtype=request.rgb.dtype),
        spatial=request.spatial,
        model_revision=request.model_revision,
        options=request.options,
    )
    # admit must resolve the nested ref and decode bytes without raising.
    prepared = adapter.admit(request_with_ref)
    assert prepared.rgb.shape == (H, W, 3)
    assert prepared.rgb.dtype == np.uint8


# --- REQ11: deployment emits a real ASGI multipart Response, not a dict --------------

def test_deployment_source_emits_starlette_response_not_dict() -> None:
    # __call__ must return a starlette.responses.Response so Ray Serve sends raw
    # multipart bytes with the declared Content-Type, rather than JSON-serializing
    # a dict. Inspect the source because importing the module requires Ray.
    import importlib.util
    import os

    spec = importlib.util.find_spec("ego_annotation.serving.deployment")
    source_path = os.path.join(os.path.dirname(spec.origin), "deployment.py")
    with open(source_path) as handle:
        source = handle.read()
    assert "from starlette.responses import Response" in source
    assert "async def __call__(self, request" in source
    assert "-> Response:" in source
    assert "multipart_asgi_response" in source
    # The old dict-returning helper must be gone.
    assert "_response_to_multipart_wire" not in source
    # Duplicate UniDepthAdapter import must be removed.
    assert source.count("UniDepthAdapter") <= 3
    # The in-cluster ObjectRef resolver is wired into the replica adapter.
    assert "_in_cluster_tensor_resolver" in source
    assert "lazy_resolve_object_ref" in source
    assert "tensor_resolver=_in_cluster_tensor_resolver" in source
