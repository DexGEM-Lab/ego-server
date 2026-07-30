"""Run an already-launched isolated UniDepth replica scaling trial.

This command never starts/stops Ray, changes the canonical router, or targets a
production lane.  It re-derives only experiment endpoints from the same isolated
plan used for launch, probes every requested experimental endpoint once, then drives
real distinct payloads through the stateless multi-endpoint gateway.  It is intended
to run after the separately printed launch contract has completed its fresh vacancy
check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.artifacts import write_batches_csv, write_items_jsonl, write_manifest_json
from ego_annotation.serving.benchmark.generator import OfferedLevel
from ego_annotation.serving.benchmark.manifest import PayloadManifest, load_payload_manifest
from ego_annotation.serving.benchmark.measurement import NvmlSampler, validate_profiler_artifact
from ego_annotation.serving.benchmark.unidepth_scaling import (
    ExperimentConfigurationError,
    StatelessReplicaGateway,
    build_unidepth_scaling_plan,
    profiler_attribution_status,
    run_scaling_level,
    validate_gpu_measurement_artifact,
    validate_server_identity,
    write_scaling_result,
)
from ego_annotation.serving.gateway import ModelServiceGateway, RetryPolicy
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


def _level_name(rate: float) -> str:
    return str(rate).replace(".", "_")


async def _typed_probe_once(endpoint, item, expected_identity, *, timeout_s: float) -> dict[str, object]:
    """One real UniDepth request plus exact server identity validation; no /health."""
    router = ModelServiceRouter.canonical().with_overrides({ModelApiName.UNIDEPTH_INFER: endpoint.url})
    gateway = ModelServiceGateway.with_httpx(router, timeout_s=timeout_s, retry_policy=RetryPolicy(max_attempts=1))
    try:
        response = await gateway.call(item.to_gateway_request())
        identity = validate_server_identity(expected_identity, response)
        return {"endpoint": endpoint.url, "typed_ready": True, "server_identity": identity.to_wire()}
    finally:
        await gateway.aclose()


async def _run(args: argparse.Namespace) -> int:
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    plan = build_unidepth_scaling_plan(
        experiment_id=args.experiment_id,
        gpu_ids=gpu_ids,
        run_root=args.run_root,
        component_port_base=args.component_port_base,
        worker_port_base=args.worker_port_base,
        serve_port_base=args.serve_port_base,
        application_release_path=args.application_release,
        source_sha=args.source_sha,
        checkpoint_digest=args.checkpoint_digest,
    )
    payload_manifest_path = Path(args.payload_dir) / "unidepth.infer.json"
    if not payload_manifest_path.is_file():
        raise ExperimentConfigurationError(f"explicit UniDepth payload manifest is absent: {payload_manifest_path}")
    manifest = load_payload_manifest(payload_manifest_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=args.manifest_count)
    required_items = args.max_offered * len(args.levels) + 1
    if len(manifest.items) < required_items:
        raise ExperimentConfigurationError(
            f"distinct evidence requires {required_items} payloads (one typed readiness payload plus {args.max_offered} per level), "
            f"but source supplied {len(manifest.items)}"
        )
    output_dir = Path(args.run_root) / args.experiment_id / "scaling"
    if output_dir.exists():
        raise ExperimentConfigurationError(f"refusing to mix evidence into existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    gpu_measurement_path = args.gpu_measurement_artifact or (output_dir / "gpu_samples.json")
    sampler = None
    if args.gpu_measurement_artifact is None:
        sampler = NvmlSampler(gpu_ids=gpu_ids, gpu_uuids={}, experiment_id=args.experiment_id,
                              release_digest=plan.application_release.release_digest)
        sampler.start()
    profiler_status = profiler_attribution_status(args.active_window_profiler_artifact)
    run_start_s = time.monotonic()
    observations = []
    for endpoint in plan.endpoints:
        observations.append(await _typed_probe_once(
            endpoint, manifest.items[0], plan.expected_server_identities[endpoint.replica_id], timeout_s=args.health_timeout_s
        ))
    write_manifest_json(output_dir / "manifest_unidepth.infer.json", manifest)
    router = ModelServiceRouter.canonical()
    gateway = StatelessReplicaGateway(
        api_name=ModelApiName.UNIDEPTH_INFER,
        base_router=router,
        endpoints=plan.endpoints,
        gateway_factory=lambda router: ModelServiceGateway.with_httpx(
            router,
            timeout_s=args.request_timeout_s,
            retry_policy=RetryPolicy(max_attempts=args.max_attempts, deadline_s=args.deadline_s),
        ),
        expected_server_identities=plan.expected_server_identities,
    )
    try:
        level_paths: list[str] = []
        for level_index, rate in enumerate(args.levels):
            if sampler is not None:
                sampler.set_level(str(rate))
            level_manifest = PayloadManifest(
                manifest_id=f"{manifest.manifest_id}-level-{level_index}",
                api_name=manifest.api_name,
                items=manifest.items[1 + level_index * args.max_offered:1 + (level_index + 1) * args.max_offered],
            )
            level = OfferedLevel(
                api_name=ModelApiName.UNIDEPTH_INFER,
                offered_intensity_per_s=rate,
                target_completed=args.target_completed,
                max_offered=args.max_offered,
            )
            raw, result = await run_scaling_level(
                level_manifest, level, gateway,
                release_digest=plan.application_release.release_digest,
                corpus_digest=manifest.manifest_hash,
                measurement_definition="unidepth-open-loop-v2-client-e2e-with-server-phase-traces",
            )
            suffix = _level_name(rate)
            write_items_jsonl(output_dir / f"items_unidepth.infer_{suffix}.jsonl", raw.records)
            write_batches_csv(output_dir / f"batches_unidepth.infer_{suffix}.csv", raw.records)
            result_path = output_dir / f"scaling_level_{suffix}.json"
            write_scaling_result(result_path, result)
            level_paths.append(str(result_path))
    finally:
        await gateway.aclose()
        run_end_s = time.monotonic()
        if sampler is not None:
            sampler.stop()
            sampler.write(gpu_measurement_path)
    gpu_measurement = validate_gpu_measurement_artifact(
        gpu_measurement_path, gpu_ids=gpu_ids, experiment_id=args.experiment_id,
        release_digest=plan.application_release.release_digest, run_start_s=run_start_s, run_end_s=run_end_s,
        min_samples_per_gpu=2,
    )
    if args.active_window_profiler_artifact is not None:
        from ego_annotation.serving.benchmark.measurement import validate_profiler_artifact
        validate_profiler_artifact(args.active_window_profiler_artifact, experiment_id=args.experiment_id,
                                   release_digest=plan.application_release.release_digest, run_start_s=run_start_s, run_end_s=run_end_s)

    (output_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "ego.unidepth-scaling-run.v1",
        "experiment_id": plan.experiment_id,
        "replica_count": len(plan.replicas),
        "endpoints": [endpoint.__dict__ for endpoint in plan.endpoints],
        "endpoint_observations": observations,
        "application_release": {
            "path": str(plan.application_release.path), "release_sha": plan.application_release.release_sha,
            "source_sha": plan.application_release.source_sha,
        },
        "gpu_measurement_artifact": {"path": str(args.gpu_measurement_artifact), "schema": gpu_measurement.get("schema")},
        "active_window_profiler_attribution": profiler_status,
        "payload_manifest": str(output_dir / "manifest_unidepth.infer.json"),
        "distinct_payload_hashes": len({item.payload_hash for item in manifest.items}),
        "payloads_consumed_once": args.max_offered * len(args.levels),
        "scaling_levels": level_paths,
        "generator_cpu_source": "process_time and monotonic wall samples attached to each scaling_level JSON",
        "measurement_definition": "unidepth-open-loop-v2-client-e2e-with-server-phase-traces",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"run_dir={output_dir}")
    print(f"scaling_levels={len(level_paths)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark already-launched isolated UniDepth replicas")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpus", required=True, help="same authorized vacant GPU ids used for the isolated launch plan")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--application-release", required=True, type=Path, help="immutable release directory, never runtime/current")
    parser.add_argument("--source-sha", required=True, help="source SHA attested by RELEASE.json")
    parser.add_argument("--checkpoint-digest", required=True, help="immutable UniDepth checkpoint digest")
    parser.add_argument("--payload-dir", required=True, type=Path, help="explicit real unidepth.infer payload corpus directory")
    parser.add_argument("--gpu-measurement-artifact", type=Path, help="pre-recorded run-aligned NVML artifact; otherwise sample around this run")
    parser.add_argument("--active-window-profiler-artifact", type=Path, help="NCU/CUPTI artifact; absent means bandwidth attribution is unavailable")
    parser.add_argument("--levels", required=True, help="comma-separated offered images/s, e.g. 2,4,6,8,10")
    parser.add_argument("--target-completed", type=int, default=100)
    parser.add_argument("--max-offered", type=int, default=400)
    parser.add_argument("--manifest-count", type=int, default=400, help="must be at least max-offered × level count; payloads are never reused across levels")
    parser.add_argument("--max-attempts", type=int, default=1, help="1 surfaces overload without retry")
    parser.add_argument("--deadline-s", type=float, default=5.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--health-timeout-s", type=float, default=2.0)
    parser.add_argument("--component-port-base", type=int, default=29000)
    parser.add_argument("--worker-port-base", type=int, default=29100)
    parser.add_argument("--serve-port-base", type=int, default=31000)
    args = parser.parse_args(argv)
    args.levels = tuple(float(value) for value in args.levels.split(",") if value.strip())
    if not args.levels:
        parser.error("--levels must contain at least one positive offered rate")
    if any(rate <= 0 for rate in args.levels):
        parser.error("--levels values must be positive")
    try:
        return asyncio.run(_run(args))
    except ExperimentConfigurationError as exc:
        print(f"experiment refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
