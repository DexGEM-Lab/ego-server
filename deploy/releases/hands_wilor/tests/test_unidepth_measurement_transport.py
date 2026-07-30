"""GPU-free adversarial tests for U4 measurement and UniDepth transport."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ego_annotation.serving.benchmark.endpoints import EndpointProbeConfig, ProbeResult, probe_endpoints_once
from ego_annotation.serving.benchmark.measurement import NvmlSampler, validate_gpu_samples, validate_profiler_artifact
from ego_annotation.serving.benchmark.manifest import build_synthetic_unidepth_manifest
from ego_annotation.serving.benchmark.release import artifact_digest, build_release
from ego_annotation.serving.benchmark.metrics import ItemRecord
from ego_annotation.serving.benchmark.unidepth_scaling import (
    EXPERIMENT_TEMP_ROOT,
    ExperimentConfigurationError,
    GeneratorCpuUsage,
    aggregate_scaling_level,
    compare_replica_scaling,
    profiler_attribution_status,
)
from ego_annotation.serving.contracts import (
    BatchTrace, ErrorCode, ImageSize, Ownership, PixelTransform, ServiceError, SpatialMetadata, TensorPayload, UniDepthResult,
)
from ego_annotation.serving.gateway import GatewayResponse
from ego_annotation.serving.benchmark.generator import OfferedLevel, OpenLoopGenerator
from ego_annotation.serving.benchmark.manifest import build_synthetic_unidepth_manifest
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import build_multipart_response, parse_multipart_response


def _record(item: str, replica: str, submit_time_s: float) -> ItemRecord:
    return ItemRecord(item, "unidepth.infer", f"r-{item}", "job", 1, f"hash-{item}", 0.0,
        submit_time_s, submit_time_s, 2.0, 0.0, "completed", 200, 1, None, None, 1.0, 1.0,
        0.0, 0.0, 0.0, 1.0, None, f"batch-{item}", 1, 1, 1.0, 1.0, 1, replica_id=replica)


def _level(*, replicas: int, rate: float, release: str | None = "release", corpus: str | None = "corpus", definition: str | None = "open-loop-v1", interval: str | None = "offer-v1"):
    # Every replica submits ``rate / replicas`` work units in the same measured
    # one-second offer window.  The later response timestamp deliberately makes
    # the completed-observation interval longer without changing offered load.
    assert rate.is_integer() and int(rate) % replicas == 0
    records_per_replica = int(rate) // replicas
    records = [
        _record(f"{replica}-{index}", f"replica-{replica}", (index + 1) / records_per_replica)
        for replica in range(replicas) for index in range(records_per_replica)
    ]
    expected = tuple(f"replica-{replica}" for replica in range(replicas))
    return aggregate_scaling_level(
        records, api_name="unidepth.infer", configured_offered_intensity_per_s=rate,
        run_start_s=0.0, offer_window_end_s=1.0, run_end_s=2.0,
        measurement_interval_id=interval, expected_replica_ids=expected,
        generator_cpu=GeneratorCpuUsage(2.0, 0.1, 0.05), release_digest=release,
        corpus_digest=corpus, measurement_definition=definition,
    )


@pytest.mark.parametrize("field,value", [("release", "other"), ("corpus", "other"), ("definition", "other"), ("interval", "other")])
def test_u4_rejects_mismatched_identity(field, value):
    baseline = _level(replicas=1, rate=4)
    kwargs = {field: value}
    scaled = _level(replicas=2, rate=8, **kwargs)
    with pytest.raises(ExperimentConfigurationError):
        compare_replica_scaling(baseline, scaled)


def test_u4_uses_equivalent_per_replica_actual_offered_rate():
    assert compare_replica_scaling(_level(replicas=1, rate=4), _level(replicas=2, rate=8)).aggregate_throughput_gain == 2
    with pytest.raises(ExperimentConfigurationError, match="actual offered rate"):
        compare_replica_scaling(_level(replicas=1, rate=4), _level(replicas=2, rate=12))


def test_u4_rejects_configured_rate_when_submissions_measured_at_point_zero_four():
    baseline = _level(replicas=1, rate=4)
    slow_records = [
        _record("slow-a", "replica-0", 50.0),
        _record("slow-b", "replica-1", 100.0),
        _record("slow-c", "replica-0", 75.0),
        _record("slow-d", "replica-1", 25.0),
    ]
    slow = aggregate_scaling_level(
        slow_records, api_name="unidepth.infer", configured_offered_intensity_per_s=4.0,
        run_start_s=0.0, offer_window_end_s=100.0, run_end_s=110.0,
        measurement_interval_id="offer-v1", expected_replica_ids=("replica-0", "replica-1"),
        generator_cpu=GeneratorCpuUsage(110.0, 0.1, 0.001), release_digest="release",
        corpus_digest="corpus", measurement_definition="open-loop-v1",
    )
    assert slow.configured_offered_intensity_per_s == 4.0
    assert slow.actual_offered_rate_per_s == pytest.approx(0.04)
    with pytest.raises(ExperimentConfigurationError, match="actual offered rate"):
        compare_replica_scaling(baseline, slow)


def test_u4_rejects_scaled_all_on_one_replica_and_missing_identity_definition():
    with pytest.raises(ExperimentConfigurationError, match="missing"):
        aggregate_scaling_level(
            [_record(f"all-on-zero-{index}", "replica-0", (index + 1) / 4) for index in range(4)], api_name="unidepth.infer",
            configured_offered_intensity_per_s=2.0, run_start_s=0.0, offer_window_end_s=1.0, run_end_s=2.0,
            measurement_interval_id="offer-v1", expected_replica_ids=("replica-0", "replica-1"),
            generator_cpu=GeneratorCpuUsage(2.0, 0.1, 0.05), release_digest="release", corpus_digest="corpus",
            measurement_definition="open-loop-v1",
        )
    with pytest.raises(ExperimentConfigurationError, match="release_digest"):
        compare_replica_scaling(_level(replicas=1, rate=4, release=None), _level(replicas=2, rate=8))


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _DelayedGateway:
    def __init__(self, clock: _Clock, drain_per_response_s: float):
        self.clock = clock
        self.drain_per_response_s = drain_per_response_s
        self.release = asyncio.Event()
        self.calls = 0

    async def call(self, request):
        self.calls += 1
        await self.release.wait()
        self.clock.now += self.drain_per_response_s
        return GatewayResponse(
            ownership=request.ownership,
            error=ServiceError(ErrorCode.TRANSPORT, "delayed test response", retryable=True, ownership=request.ownership),
        )


async def _run_delayed_generator(drain_per_response_s: float):
    clock = _Clock()
    gateway = _DelayedGateway(clock, drain_per_response_s)

    async def advance_offer_clock(delay_s: float):
        clock.now += delay_s

    manifest = build_synthetic_unidepth_manifest(manifest_id="delayed-offers", count=3)
    level = OfferedLevel(api_name=ModelApiName.UNIDEPTH_INFER, offered_intensity_per_s=1.0, max_offered=3)
    task = asyncio.create_task(OpenLoopGenerator(gateway, clock=clock, sleep=advance_offer_clock).run_level(manifest, level))
    while gateway.calls != 3:
        await asyncio.sleep(0)
    gateway.release.set()
    return await task


def test_generator_actual_offer_rate_is_invariant_to_delayed_response_drain():
    immediate = asyncio.run(_run_delayed_generator(0.0))
    delayed = asyncio.run(_run_delayed_generator(10.0))
    assert immediate.actual_submission_span_s == delayed.actual_submission_span_s == 2.0
    assert immediate.actual_offered_rate_per_s == delayed.actual_offered_rate_per_s == 1.5
    assert delayed.drain_duration_s > immediate.drain_duration_s
    assert delayed.observation_duration_s > immediate.observation_duration_s


def test_nvml_sample_falls_back_to_nvidia_smi_without_pynvml(monkeypatch):
    import builtins
    import subprocess as real_subprocess

    from ego_annotation.serving.benchmark.measurement import NvmlSampler

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("No module named 'pynvml'")
        return real_import(name, *args, **kwargs)

    class Completed:
        returncode = 0
        stdout = "GPU-smi-7, 42, 1024\n"
        stderr = ""

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(real_subprocess, "run", lambda *a, **k: Completed())
    sample = NvmlSampler._nvml_sample(7)
    assert sample == {"gpu_uuid": "GPU-smi-7", "utilization_gpu_pct": 42.0, "memory_used_bytes": 1024 * 1024 * 1024}


def test_run_aligned_sampler_and_validator_reject_stale_empty_or_wrong_identity(tmp_path):
    now = iter([0.0, 0.1, 0.9, 1.0])
    sampler = NvmlSampler(gpu_ids=(4,), gpu_uuids={4: "GPU-4"}, experiment_id="exp", release_digest="rel",
        sample_fn=lambda _: {"gpu_uuid": "GPU-4", "utilization_gpu_pct": 30, "memory_used_bytes": 100},
        clock=lambda: next(now), interval_s=1000)
    sampler.start()
    sampler.set_level("rate-4")
    sampler.stop()
    path = sampler.write(tmp_path / "samples.json")
    assert validate_gpu_samples(path, gpu_ids=(4,), experiment_id="exp", release_digest="rel", run_start_s=0.1, run_end_s=0.9)
    payload = json.loads(path.read_text())
    payload["samples"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="empty"):
        validate_gpu_samples(path, gpu_ids=(4,), experiment_id="exp", release_digest="rel", run_start_s=0.1, run_end_s=0.9)


def test_profiler_requires_nonempty_kernels_counters_and_overlap(tmp_path):
    path = tmp_path / "ncu.json"
    base = {"tool": "ncu", "experiment_id": "exp", "release_digest": "rel", "start_s": 1.0, "end_s": 2.0,
            "kernels": [], "counters": {}}
    path.write_text(json.dumps(base))
    assert profiler_attribution_status(path) == "available_from_ncu"  # discovery only, never attribution
    with pytest.raises(ValueError):
        validate_profiler_artifact(path, experiment_id="exp", release_digest="rel", run_start_s=1.2, run_end_s=1.8)
    base.update({"kernels": [{"name": "kernel"}], "counters": {"dram_bytes": 10}})
    path.write_text(json.dumps(base))
    assert validate_profiler_artifact(path, experiment_id="exp", release_digest="rel", run_start_s=1.2, run_end_s=1.8)
    with pytest.raises(ValueError, match="overlap"):
        validate_profiler_artifact(path, experiment_id="exp", release_digest="rel", run_start_s=3.0, run_end_s=4.0)


class Probe:
    async def get(self, url: str, *, timeout_s: float):
        return ProbeResult(live=True, status_code=404, latency_ms=1.0)


def test_generic_health_never_accepts_404():
    router = ModelServiceRouter.canonical().with_overrides({ModelApiName.UNIDEPTH_INFER: "http://127.0.0.1:31000/unidepth.infer"})
    observation = asyncio.run(probe_endpoints_once(router, Probe(), apis=[ModelApiName.UNIDEPTH_INFER]))[0]
    assert observation.url.endswith("/-/healthz") and observation.live is False


def test_worker_identity_derives_release_checkpoint_and_pid_instead_of_echoing_env(tmp_path, monkeypatch):
    from ego_annotation.serving.unidepth import UniDepthAdapter, build_unidepth_model_config
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("actual release bytes\n")
    release = build_release(source, tmp_path / "releases", source_sha="a" * 40)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"actual checkpoint bytes")
    item = build_synthetic_unidepth_manifest(manifest_id="identity", count=1).items[0]
    request = item.to_gateway_request()
    from ego_annotation.serving.contracts import UniDepthRequest
    typed = UniDepthRequest(ownership=request.ownership, rgb=TensorPayload(request.parts[0].data, request.parts[0].shape, request.parts[0].dtype),
                            spatial=request.spatial, model_revision=request.model_revision)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, get_device_properties=lambda _: SimpleNamespace(uuid="GPU-actual"))))
    config = build_unidepth_model_config(
        checkpoint=str(checkpoint), model_revision=typed.model_revision, device="cpu", canonical_height=typed.rgb.shape[0],
        canonical_width=typed.rgb.shape[1], replica_id="replica", assigned_gpu=4, experiment_id="exp",
        application_release_sha="spoofed-release", application_release_path=str(release), checkpoint_digest="spoofed-checkpoint",
        gcs_address="127.0.0.1:29000", http_port=31000,
        temp_dir=str(EXPERIMENT_TEMP_ROOT / "exp" / "gpu4"),
    )
    class Backend:
        def infer(self, _):
            h, w = config.canonical_height, config.canonical_width
            return {"depth": np.ones((1, 1, h, w), np.float32), "intrinsics": np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
                    "confidence": np.ones((1, 1, h, w), np.float32)}
    result = asyncio.run(UniDepthAdapter(config, backend_factory=lambda _: Backend()).infer(typed)).result
    identity = result.server_identity
    assert identity.release_digest == release.name and identity.release_sha == release.name
    assert identity.checkpoint_digest == artifact_digest(checkpoint)
    assert identity.cuda_uuid == "GPU-actual" and identity.worker_pid > 0


def test_unidepth_multipart_metadata_has_no_duplicate_base64_round_trip():
    ownership = Ownership("request", "job", "item", "unidepth.infer", "source")
    spatial = SpatialMetadata(ImageSize(2, 2), ImageSize(2, 2), "RGB", PixelTransform.identity())
    trace = BatchTrace("batch", "replica", 0, 0, 0, 1, 1, 1, 1, 1)
    depth = TensorPayload(np.ones((2, 2), np.float32).tobytes(), (2, 2), "float32")
    intrinsics = TensorPayload(np.eye(3, dtype=np.float32).tobytes(), (3, 3), "float32")
    confidence = TensorPayload(np.ones((2, 2), np.float32).tobytes(), (2, 2), "float32")
    result = UniDepthResult(ownership, depth, intrinsics, confidence, spatial, "rev", trace)
    metadata = {"result": result.to_wire(include_tensor_data=False), "ownership": ownership.to_wire()}
    arrays = {"depth_m": (bytes(depth.data), depth.shape, depth.dtype), "K_px": (bytes(intrinsics.data), intrinsics.shape, intrinsics.dtype),
              "confidence": (bytes(confidence.data), confidence.shape, confidence.dtype)}
    body, content_type = build_multipart_response(metadata, arrays)
    assert b"data_b64" not in body
    parsed_meta, parsed_arrays = parse_multipart_response(body, content_type)
    assert set(parsed_arrays) == set(arrays)
    assert parsed_arrays["depth_m"][0] == bytes(depth.data)
    assert parsed_meta["result"]["depth_m"] == {"shape": [2, 2], "dtype": "float32"}
