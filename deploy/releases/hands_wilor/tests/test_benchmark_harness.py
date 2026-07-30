"""Open-loop benchmark harness tests against a deterministic fake HTTP server.

These tests exercise the *real* multipart bytes end-to-end:
* the gateway builds real ``multipart/form-data`` request bodies,
* the fake aiohttp server parses those bodies and emits real multipart responses,
* the gateway parses the responses into typed results with batch traces and phase
  decomposition,
* the open-loop generator schedules arrivals at wall-clock times independent of
  completions,
* metrics record offered/admitted/completed/rejected rates, response latency
  percentiles (separate from amortized cost), phase decomposition, batch
  distribution, model-load count, and distinct payload hashes,
* the endpoint runner probes live endpoints once without polling,
* bounded retries surface overload as typed BACKPRESSURE rather than hiding it.

No Ray is imported. No live model is loaded. The fake server is deterministic.
Tests use ``asyncio.run`` (no pytest-asyncio dependency), matching the existing
``test_model_serving_foundation.py`` pattern.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest

from ego_annotation.serving.benchmark.artifacts import (
    batch_rows_from_records,
    write_batches_csv,
    write_items_jsonl,
    write_levels_csv,
    write_manifest_json,
    write_run_manifest,
)
from ego_annotation.serving.benchmark.endpoints import (
    EndpointProbeConfig,
    build_run_manifest,
    probe_endpoints_once,
)
from ego_annotation.serving.benchmark.fakeserver import (
    FakeHttpGatewayTransport,
    FakeHttpProbeTransport,
    FakeServerConfig,
    start_fake_server,
)
from ego_annotation.serving.benchmark.generator import OfferedLevel, OpenLoopGenerator
from ego_annotation.serving.benchmark.manifest import (
    PAYLOAD_SOURCE_SCHEMA,
    build_synthetic_droid_session_plan,
    build_synthetic_hand_crops_manifest,
    build_synthetic_hand_images_manifest,
    build_synthetic_hawor_infiller_manifest,
    build_synthetic_hawor_tracks_manifest,
    build_synthetic_media_manifest,
    build_synthetic_unidepth_manifest,
    load_payload_manifest,
)
from ego_annotation.serving.benchmark.metrics import (
    ADMITTED_OUTCOMES,
    REJECTED_OUTCOMES,
    summarize,
)
from ego_annotation.serving.benchmark.runner import ApiBenchmarkPlan, BenchmarkRunner
from ego_annotation.serving.benchmark.plotting import plot_batch_distribution, plot_throughput_latency
from ego_annotation.serving.contracts import ErrorCode
from ego_annotation.serving.gateway import ModelServiceGateway, RetryPolicy
from ego_annotation.serving.router import (
    MODEL_SERVICES,
    COSMOS3_BASELINE_URL,
    ModelApiName,
    ModelServiceRouter,
    cosmos3_baseline_override,
    service_for_api,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _router_to(server, api: ModelApiName) -> ModelServiceRouter:
    return ModelServiceRouter.canonical().with_overrides({api: f"{server.base_url}/{api.value}"})


def run(coro):
    """Run a coroutine to completion (single asyncio.run per test, like existing tests)."""
    return asyncio.run(coro)


async def _start_server(config=None):
    return await start_fake_server(host="127.0.0.1", port=0, config=config or FakeServerConfig(forward_latency_s=0.005))


# --- manifest: distinct payloads -----------------------------------------------------

def test_manifest_items_have_distinct_payload_hashes_and_provenance():
    manifest = build_synthetic_unidepth_manifest(manifest_id="m1", count=20)
    hashes = {item.payload_hash for item in manifest.items}
    assert len(hashes) == 20, "every manifest item must have a distinct content hash"
    job_ids = {item.ownership.job_id for item in manifest.items}
    assert len(job_ids) >= 2, "manifest must span multiple job/agent ownerships"
    ts = [item.source_timestamp_s for item in manifest.items]
    assert ts == sorted(ts)
    for item in manifest.items:
        assert item.spatial is not None
        assert item.spatial.pixel_transform.resize_mode == "identity"
        assert item.spatial.source_size == item.spatial.model_size


def test_manifest_rejects_duplicate_payload_hashes():
    from ego_annotation.serving.benchmark.manifest import PayloadItem, PayloadManifest
    from ego_annotation.serving.contracts import Ownership

    base = build_synthetic_unidepth_manifest(manifest_id="m", count=2).items
    dup = PayloadItem(
        item_id="dup", api_name=base[0].api_name, ownership=Ownership("r", "j", "i", "s", "src"),
        parts=base[0].parts, spatial=base[0].spatial, model_revision=base[0].model_revision,
        work_units=1, source_timestamp_s=0.0, payload_hash=base[0].payload_hash,
    )
    with pytest.raises(ValueError, match="distinct"):
        PayloadManifest(manifest_id="dup", api_name=base[0].api_name, items=(base[0], dup))


# --- gateway: real multipart bytes round-trip through fake server --------------------

def test_gateway_round_trips_real_multipart_bytes_through_fake_server():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="rt", count=3)
            responses = await gateway.call_batch([item.to_gateway_request() for item in manifest.items])
            await transport.aclose()
            return responses
        finally:
            await server.stop()

    responses = run(scenario())
    assert len(responses) == 3
    for resp in responses:
        assert resp.result is not None, f"expected result, got error {resp.error}"
        assert resp.error is None
        assert "out_rgb" in resp.result.arrays
        assert resp.result.trace is not None
        assert resp.result.trace.model_load_count == 1
        assert resp.result.trace.forward_count == 1
        pt = resp.result.metadata["phase_timing"]
        for key in ("admission_ms", "queue_ms", "dispatch_ms", "forward_ms", "encoding_ms"):
            assert key in pt and pt[key] >= 0


# --- gateway: bounded retries surface overload as typed BACKPRESSURE ----------------

def test_gateway_envelope_round_trips_typed_result_without_multipart_fallback():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(
                router, transport, retry_policy=RetryPolicy(max_attempts=1), wire_format="envelope",
            )
            item = build_synthetic_unidepth_manifest(manifest_id="envelope", count=1).items[0]
            response = await gateway.call(item.to_gateway_request())
            await transport.aclose()
            return gateway, response
        finally:
            await server.stop()

    gateway, response = run(scenario())
    assert gateway.wire_format == "envelope"
    assert response.result is not None, response.error
    assert response.result.trace is not None
    assert response.result.trace.forward_count == 1
    assert set(response.result.arrays) == {"out_rgb"}


def test_httpx_gateway_streams_envelope_vectors_end_to_end():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            gateway = ModelServiceGateway.with_httpx(
                router, retry_policy=RetryPolicy(max_attempts=1), wire_format="envelope",
            )
            item = build_synthetic_unidepth_manifest(manifest_id="httpx-envelope", count=1).items[0]
            response = await gateway.call(item.to_gateway_request())
            await gateway.aclose()
            return response
        finally:
            await server.stop()

    response = run(scenario())
    assert response.result is not None, response.error
    assert response.result.trace is not None


def test_gateway_rejects_response_framed_as_multipart_when_envelope_was_requested():
    class MultipartOnlyTransport:
        async def post(self, url, *, content, headers):
            from ego_annotation.serving.transport import build_multipart_response

            body, content_type = build_multipart_response({"result": {}}, {})
            return type("Response", (), {"status_code": 200, "content": body, "headers": {"Content-Type": content_type}})()

    item = build_synthetic_unidepth_manifest(manifest_id="no-fallback", count=1).items[0]
    gateway = ModelServiceGateway(
        ModelServiceRouter.canonical(), MultipartOnlyTransport(), retry_policy=RetryPolicy(max_attempts=1), wire_format="envelope",
    )
    response = run(gateway.call(item.to_gateway_request()))
    assert response.error is not None
    assert response.error.code is ErrorCode.TRANSPORT
    assert "expected binary envelope response" in response.error.message


def test_gateway_retries_surface_overload_as_typed_backpressure_not_hidden():
    async def scenario():
        server = await start_fake_server(
            host="127.0.0.1", port=0, config=FakeServerConfig(forward_latency_s=0.2, overload_in_flight=1)
        )
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=2, deadline_s=0.5, initial_backoff_s=0.001))
            manifest = build_synthetic_unidepth_manifest(manifest_id="ovl", count=8)
            responses = await gateway.call_batch([item.to_gateway_request() for item in manifest.items])
            await transport.aclose()
            return responses
        finally:
            await server.stop()

    responses = run(scenario())
    backpressure = [r for r in responses if r.error is not None and r.error.code is ErrorCode.BACKPRESSURE]
    assert backpressure, "overload must surface as typed BACKPRESSURE, not be hidden behind retries"
    for r in responses:
        assert r.attempts <= 2


# --- open-loop generator: arrivals independent of completions -----------------------

def test_open_loop_generator_schedules_arrivals_independent_of_completions():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="ol", count=10)
            # Offer 200/s with 0.005s forward latency. Open-loop: submits at the
            # scheduled rate regardless of in-flight count.
            level = OfferedLevel(api_name=api, offered_intensity_per_s=200.0, target_completed=10, max_offered=10)
            gen = OpenLoopGenerator(gateway)
            result = await gen.run_level(manifest, level)
            await transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    assert len(result.records) == 10
    submits = [r.submit_time_s for r in result.records]
    deltas = [submits[i + 1] - submits[i] for i in range(len(submits) - 1)]
    mean_delta = sum(deltas) / len(deltas)
    # Mean delta should be near 1/200 = 0.005s. A closed-loop generator would space
    # by completion latency; an open-loop one does not.
    assert mean_delta < 0.05, f"open-loop submit spacing {mean_delta} too large; generator is self-limiting"


def test_open_loop_generator_under_overload_still_offers_at_scheduled_rate():
    """The defining property of open-loop: under overload, offered rate stays fixed
    and rejected (backpressure) count rises — the generator does not self-limit."""
    async def scenario():
        # Server that rejects everything above 1 in-flight with 503.
        server = await start_fake_server(
            host="127.0.0.1", port=0, config=FakeServerConfig(forward_latency_s=0.1, overload_in_flight=1)
        )
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1, deadline_s=0.2))
            manifest = build_synthetic_unidepth_manifest(manifest_id="olo", count=12)
            level = OfferedLevel(api_name=api, offered_intensity_per_s=200.0, target_completed=12, max_offered=12)
            gen = OpenLoopGenerator(gateway)
            result = await gen.run_level(manifest, level)
            await transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    # Offered count is fixed at 12 regardless of overload.
    assert len(result.records) == 12
    # Under overload, rejections (backpressure/transport) dominate.
    rejected = [r for r in result.records if r.outcome in REJECTED_OUTCOMES]
    assert len(rejected) > 0, "under overload the open-loop generator must record rejections, not self-limit"
    # Offered rate is recorded even when most are rejected.
    summary = summarize(result.records, api_name="unidepth.infer", offered_intensity_per_s=200.0, duration_s=result.duration_s)
    assert summary.offered_count == 12
    assert summary.rejected_count == len(rejected)


# --- metrics: four rates + latency vs amortized cost separation ---------------------

def test_metrics_record_four_rates_and_separate_latency_from_amortized_cost():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="m4", count=30)
            level = OfferedLevel(api_name=api, offered_intensity_per_s=100.0, target_completed=30, max_offered=30)
            gen = OpenLoopGenerator(gateway)
            result = await gen.run_level(manifest, level)
            await transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    summary = summarize(result.records, api_name="unidepth.infer", offered_intensity_per_s=100.0, duration_s=result.duration_s)
    for attr in ("offered_rate_per_s", "admitted_rate_per_s", "completed_rate_per_s", "rejected_rate_per_s"):
        assert getattr(summary, attr) is not None
    assert summary.offered_count == 30
    assert summary.completed_count + summary.rejected_count + summary.in_flight_count == 30
    assert summary.response_latency_p50_ms is not None
    assert summary.response_latency_mean_ms is not None
    completed = [r for r in result.records if r.outcome == "completed" and r.amortized_cost_ms is not None]
    if completed:
        # Response latency (end-to-end) >= amortized per-item compute cost.
        assert summary.response_latency_mean_ms >= summary.amortized_cost_mean_ms - 1e-6
    # Model load count constant (resident model).
    assert summary.model_load_count_min == summary.model_load_count_max == 1
    # Distinct payload hashes.
    assert summary.distinct_payload_hashes == 30


# --- metrics: phase decomposition + batch distribution ------------------------------

def test_metrics_phase_decomposition_and_batch_distribution():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="ph", count=15)
            level = OfferedLevel(api_name=api, offered_intensity_per_s=50.0, target_completed=15, max_offered=15)
            gen = OpenLoopGenerator(gateway)
            result = await gen.run_level(manifest, level)
            await transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    summary = summarize(result.records, api_name="unidepth.infer", offered_intensity_per_s=50.0, duration_s=result.duration_s)
    for attr in ("admission_ms_mean", "queue_ms_mean", "dispatch_ms_mean", "forward_ms_mean", "encoding_ms_mean"):
        assert getattr(summary, attr) is not None and getattr(summary, attr) >= 0
    completed = [r for r in result.records if r.outcome == "completed"]
    assert all(r.batch_size is not None for r in completed)
    assert all(r.batch_id is not None for r in completed)
    rows = batch_rows_from_records(result.records)
    assert len(rows) == len({r.batch_id for r in completed})


# --- endpoint runner: probe once, no polling, run manifest preserved ---------------

def test_endpoint_runner_probes_once_and_preserves_run_manifest():
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            dead_port = 1
            router = ModelServiceRouter.canonical().with_overrides({
                api: f"{server.base_url}/{api.value}",
                ModelApiName.COSMOS3_REASON: f"http://127.0.0.1:{dead_port}/cosmos3.reason",
            })
            probe_transport = FakeHttpProbeTransport(server.runner)
            observations = await probe_endpoints_once(
                router, probe_transport, apis=[api, ModelApiName.COSMOS3_REASON],
                config=EndpointProbeConfig(health_path="/health", timeout_s=0.5),
            )
            await probe_transport.aclose()
            return observations
        finally:
            await server.stop()

    observations = run(scenario())
    live = {o.api_name: o.live for o in observations}
    assert live[ModelApiName.UNIDEPTH_INFER] is True
    assert live[ModelApiName.COSMOS3_REASON] is False
    manifest = build_run_manifest(
        run_id="test-run", observations=observations,
        probe_config=EndpointProbeConfig(health_path="/health", timeout_s=0.5),
    )
    assert "unidepth.infer" in manifest.live_apis
    assert "cosmos3.reason" in manifest.down_apis


# --- full runner: probe, sweep, write raw artifacts ---------------------------------

def test_benchmark_runner_writes_raw_artifacts_and_skips_down_endpoints(tmp_path):
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            dead_port = 1
            router = ModelServiceRouter.canonical().with_overrides({
                api: f"{server.base_url}/{api.value}",
                ModelApiName.COSMOS3_REASON: f"http://127.0.0.1:{dead_port}/cosmos3.reason",
            })
            transport = FakeHttpGatewayTransport(server.runner)
            probe_transport = FakeHttpProbeTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="run", count=40)
            levels = (
                OfferedLevel(api_name=api, offered_intensity_per_s=50.0, target_completed=20, max_offered=20),
                OfferedLevel(api_name=api, offered_intensity_per_s=200.0, target_completed=20, max_offered=20),
            )
            cosmos_manifest = build_synthetic_media_manifest(manifest_id="run-cosmos", count=5, api_name=ModelApiName.COSMOS3_REASON)
            cosmos_levels = (OfferedLevel(api_name=ModelApiName.COSMOS3_REASON, offered_intensity_per_s=5.0, target_completed=5, max_offered=5),)
            plans = [
                ApiBenchmarkPlan(api_name=api, manifest=manifest, offered_levels=levels),
                ApiBenchmarkPlan(api_name=ModelApiName.COSMOS3_REASON, manifest=cosmos_manifest, offered_levels=cosmos_levels),
            ]
            runner = BenchmarkRunner(
                router=router, gateway=gateway, probe_transport=probe_transport, base_dir=tmp_path,
                probe_config=EndpointProbeConfig(health_path="/health", timeout_s=0.5),
            )
            result = await runner.run(plans, run_id="test-full-run")
            await transport.aclose()
            await probe_transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    assert (result.run_dir / "levels.csv").exists()
    assert (result.run_dir / "items_unidepth.infer.jsonl").exists()
    assert (result.run_dir / "batches_unidepth.infer.csv").exists()
    assert (result.run_dir / "manifest_unidepth.infer.json").exists()
    assert (result.run_dir / "run_manifest.json").exists()
    assert not (result.run_dir / "items_cosmos3.reason.jsonl").exists()
    run_man = json.loads((result.run_dir / "run_manifest.json").read_text())
    assert "cosmos3.reason" in run_man["down_apis"]
    assert any("cosmos3.reason" in n for n in run_man["notes"])
    import csv as csvmod

    with (result.run_dir / "levels.csv").open() as h:
        rows = list(csvmod.DictReader(h))
    assert len(rows) == 2
    assert all(r["api_name"] == "unidepth.infer" for r in rows)
    item_lines = (result.run_dir / "items_unidepth.infer.jsonl").read_text().strip().split("\n")
    assert len(item_lines) == 40


# --- plotting: generates plots from raw artifacts -----------------------------------

def test_plotting_generates_plots_from_raw_artifacts(tmp_path):
    async def scenario():
        server = await _start_server()
        try:
            api = ModelApiName.UNIDEPTH_INFER
            router = _router_to(server, api)
            transport = FakeHttpGatewayTransport(server.runner)
            probe_transport = FakeHttpProbeTransport(server.runner)
            gateway = ModelServiceGateway(router, transport, retry_policy=RetryPolicy(max_attempts=1))
            manifest = build_synthetic_unidepth_manifest(manifest_id="plt", count=30)
            levels = (
                OfferedLevel(api_name=api, offered_intensity_per_s=50.0, target_completed=15, max_offered=15),
                OfferedLevel(api_name=api, offered_intensity_per_s=200.0, target_completed=15, max_offered=15),
            )
            plan = ApiBenchmarkPlan(api_name=api, manifest=manifest, offered_levels=levels)
            runner = BenchmarkRunner(
                router=router, gateway=gateway, probe_transport=probe_transport, base_dir=tmp_path,
                probe_config=EndpointProbeConfig(health_path="/health", timeout_s=0.5),
            )
            result = await runner.run([plan], run_id="test-plot-run")
            await transport.aclose()
            await probe_transport.aclose()
            return result
        finally:
            await server.stop()

    result = run(scenario())
    out_dir = tmp_path / "plots"
    tl = plot_throughput_latency(result.run_dir / "levels.csv", out_dir)
    bd = plot_batch_distribution(result.run_dir / "items_unidepth.infer.jsonl", out_dir)
    assert len(tl) >= 1 and all(p.suffix == ".png" for p in tl)
    assert len(bd) >= 1 and all(p.suffix == ".png" for p in bd)
    for p in tl + bd:
        assert p.stat().st_size > 0


# --- router: stable public API names + per-GPU Serve HTTP ports ---------------------

def test_router_maps_each_api_to_its_committed_gpu_serve_port():
    router = ModelServiceRouter.canonical()
    assert router.url_for("unidepth.infer").endswith(":28000/unidepth.infer")
    assert router.url_for("hands.detect").endswith(":28001/hands.detect")
    assert router.url_for("wilor.reconstruct").endswith(":28004/wilor.reconstruct")
    assert router.url_for("droid.create_session").endswith(":28002/droid.create_session")
    assert router.url_for("droid.push_frame").endswith(":28002/droid.push_frame")
    assert router.url_for("droid.finalize").endswith(":28002/droid.finalize")
    assert router.url_for("hawor.infer_tracks").endswith(":28003/hawor.infer_tracks")
    assert router.url_for("hawor_infiller.fill").endswith(":28003/hawor_infiller.fill")
    assert router.url_for("cosmos3.reason").endswith(":28006/cosmos3.reason")
    assert router.endpoint_for("hands.detect").model_revision == "hands-yolo-v2"


def test_router_overrides_do_not_mutate_canonical_config():
    router = ModelServiceRouter.canonical()
    overridden = router.with_overrides({"unidepth.infer": "http://fake:9999/unidepth.infer"})
    assert overridden.url_for("unidepth.infer") == "http://fake:9999/unidepth.infer"
    assert router.url_for("unidepth.infer").endswith(":28000/unidepth.infer")


def test_router_unknown_api_raises():
    router = ModelServiceRouter.canonical()
    with pytest.raises(KeyError):
        router.url_for("nonexistent.api")


def test_router_endpoint_manifest_records_gpu_and_work_unit():
    router = ModelServiceRouter.canonical()
    ep = router.endpoint_for("unidepth.infer")
    m = ep.to_manifest()
    assert m["gpu_id"] == "0"
    assert m["work_unit"] == "images"
    assert m["url"].endswith(":28000/unidepth.infer")


def test_seven_services_group_nine_methods_and_baseline_is_explicit():
    assert len(MODEL_SERVICES) == 7
    assert {service.name for service in MODEL_SERVICES} == {
        "unidepth", "hands", "wilor", "droid", "hawor_tracks", "hawor_infiller", "cosmos3",
    }
    droid = service_for_api(ModelApiName.DROID_PUSH_FRAME)
    assert droid.name == "droid" and droid.is_stateful_session
    assert set(droid.api_names) == {
        ModelApiName.DROID_CREATE_SESSION,
        ModelApiName.DROID_PUSH_FRAME,
        ModelApiName.DROID_FINALIZE,
    }
    router = ModelServiceRouter.canonical()
    assert router.url_for(ModelApiName.COSMOS3_REASON).endswith(":28006/cosmos3.reason")
    baseline_router = router.with_overrides(cosmos3_baseline_override())
    assert baseline_router.url_for(ModelApiName.COSMOS3_REASON) == COSMOS3_BASELINE_URL


@pytest.mark.parametrize(
    ("manifest", "required_parts"),
    [
        (build_synthetic_hand_images_manifest(manifest_id="hands", count=2), {"rgb"}),
        (build_synthetic_hand_crops_manifest(manifest_id="wilor", count=2), {"crop"}),
        (build_synthetic_hawor_tracks_manifest(manifest_id="hawor", count=2), {"track_chunk", "source_timestamps", "observation_mask"}),
        (build_synthetic_hawor_infiller_manifest(manifest_id="infill", count=2), {"mano_state", "source_timestamps", "observation_mask", "uncertainty"}),
    ],
)
def test_model_native_fixture_manifests_preserve_required_parts(manifest, required_parts):
    if manifest.api_name == ModelApiName.HANDS_DETECT:
        assert {item.model_revision for item in manifest.items} == {"hands-yolo-sam2.1-hiera-l"}
    assert len({item.payload_hash for item in manifest.items}) == len(manifest.items)
    for item in manifest.items:
        assert {part.name for part in item.parts} == required_parts
        assert item.ownership.stage_id == item.api_name.value
        assert item.ownership.source_timestamp_s == item.source_timestamp_s


def test_droid_session_plan_preserves_real_lifecycle_identifiers():
    (plan,) = build_synthetic_droid_session_plan(plan_id="droid", session_count=1, frames_per_session=3)
    assert plan.create_manifest.items[0].metadata["session_id"] == plan.session_id
    assert {item.metadata["session_id"] for item in plan.push_frame_manifest.items} == {plan.session_id}
    assert [item.metadata["frame_id"] for item in plan.push_frame_manifest.items] == [0, 1, 2]
    assert plan.finalize_manifest.items[0].metadata["session_id"] == plan.session_id
    assert {part.name for part in plan.create_manifest.items[0].parts} == {"K_px", "options"}
    assert {part.name for part in plan.push_frame_manifest.items[0].parts} == {"rgb", "static_mask"}


def test_real_payload_source_loader_recomputes_hash_and_strips_file_names(tmp_path):
    source_manifest = build_synthetic_hand_crops_manifest(manifest_id="source", count=2)
    source_items = []
    for item_index, item in enumerate(source_manifest.items):
        parts = []
        for part_index, part in enumerate(item.parts):
            file_name = f"payload-{item_index}-{part_index}.bin"
            (tmp_path / file_name).write_bytes(part.data)
            parts.append({"name": part.name, "file": file_name, "shape": list(part.shape), "dtype": part.dtype})
        source_items.append({
            "item_id": item.item_id,
            "ownership": item.ownership.to_wire(),
            "parts": parts,
            "spatial": item.spatial.to_wire() if item.spatial else None,
            "model_revision": item.model_revision,
            "work_units": item.work_units,
            "source_timestamp_s": item.source_timestamp_s,
            "metadata": dict(item.metadata),
        })
    source_path = tmp_path / "wilor.reconstruct.json"
    source_path.write_text(json.dumps({
        "schema": PAYLOAD_SOURCE_SCHEMA,
        "manifest_id": "captured-wilor",
        "api_name": "wilor.reconstruct",
        "items": source_items,
    }))
    loaded = load_payload_manifest(source_path, expected_api=ModelApiName.WILOR_RECONSTRUCT, limit=2)
    assert loaded.manifest_id == "captured-wilor"
    assert [item.payload_hash for item in loaded.items] == [item.payload_hash for item in source_manifest.items]
    assert "payload-0-0.bin" not in json.dumps(loaded.to_manifest())


def test_unified_command_fake_server_probes_once_and_writes_combined_artifacts(tmp_path):
    from scripts.ray_serve_benchmark_all import main

    result = main([
        "--out", str(tmp_path), "--run-id", "unified-smoke", "--fake-server",
        "--apis", "unidepth.infer,hands.detect", "--levels", "20",
        "--target-completed", "2", "--max-offered", "2", "--manifest-count", "2",
    ])
    assert result == 0
    run_dir = tmp_path / "unified-smoke"
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["live_apis"] == ["unidepth.infer", "hands.detect"]
    assert set(run_manifest["artifact_paths"]) >= {
        "unidepth.infer.manifest", "hands.detect.manifest", "levels_csv",
    }
    assert (run_dir / "manifest_hands.detect.json").exists()
    assert (run_dir / "items_hands.detect.jsonl").exists()
