"""Run one U4 client-sharding rate after isolated UniDepth endpoints are live.

This is intentionally benchmark-only: it neither starts nor stops Ray/Serve and
never targets the canonical production router. Invoke it once per per-replica rate
(e.g. 6 then 8); every invocation assigns exactly 100 fresh corpus items to every
independent client process.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.measurement import NvmlSampler
from ego_annotation.serving.benchmark.unidepth_client_sharding import (
    MIN_OFFERS_PER_ENDPOINT,
    SUPPORTED_PER_ENDPOINT_RATES,
    ClientShardRunInvalid,
    build_client_shard_specs,
    run_parent,
)
from ego_annotation.serving.benchmark.unidepth_scaling import build_unidepth_scaling_plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent-process UniDepth U4 client-sharding benchmark")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated already-launched isolated UniDepth GPU IDs")
    parser.add_argument("--gpu-uuids", required=True, help="comma-separated preflight NVML UUIDs, one per --gpus entry")
    parser.add_argument("--run-root", required=True, type=Path, help="fresh benchmark run root")
    parser.add_argument("--application-release", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--per-endpoint-rate", required=True, type=float, choices=sorted(SUPPORTED_PER_ENDPOINT_RATES))
    parser.add_argument("--drivers-per-endpoint", type=int, default=4)
    parser.add_argument("--offers-per-endpoint", type=int, default=MIN_OFFERS_PER_ENDPOINT)
    parser.add_argument("--experiment-batch-cap", type=int, choices=(8, 16), default=8)
    parser.add_argument("--experiment-batch-wait-ms", type=int, choices=(20, 50), default=20)
    parser.add_argument("--wire-format", choices=("multipart", "envelope"), default="multipart")
    parser.add_argument("--max-concurrent-forwards", type=int, default=1)
    parser.add_argument("--common-start-delay-s", type=float, default=2.0)
    parser.add_argument("--nvml-interval-s", type=float, default=0.2)
    parser.add_argument("--component-port-base", type=int, default=29000)
    parser.add_argument("--worker-port-base", type=int, default=29100)
    parser.add_argument("--serve-port-base", type=int, default=31000)
    args = parser.parse_args(argv)
    if args.common_start_delay_s <= 0:
        parser.error("--common-start-delay-s must be positive so every child can reach the common monotonic start")
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    gpu_uuid_values = tuple(value.strip() for value in args.gpu_uuids.split(",") if value.strip())
    if not gpu_ids:
        parser.error("--gpus must contain at least one GPU")
    if len(gpu_uuid_values) != len(gpu_ids):
        parser.error("--gpu-uuids must supply exactly one UUID per --gpus entry")
    gpu_uuids = dict(zip(gpu_ids, gpu_uuid_values))
    output_dir = args.run_root / args.experiment_id / (
        f"cap{args.experiment_batch_cap}_wait{args.experiment_batch_wait_ms}_c{args.max_concurrent_forwards}"
        f"_rate_{str(args.per_endpoint_rate).replace('.', '_')}"
    )
    if output_dir.exists():
        parser.error(f"refusing to mix evidence into existing run directory: {output_dir}")
    source = args.payload_dir / "unidepth.infer.json"
    if not source.is_file():
        parser.error(f"explicit UniDepth corpus descriptor is absent: {source}")
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
        gpu_uuids=gpu_uuid_values,
        experiment_batch_cap=args.experiment_batch_cap,
        experiment_batch_wait_ms=args.experiment_batch_wait_ms,
        max_concurrent_forwards=args.max_concurrent_forwards,
        wire_format=args.wire_format,
    )
    scheduled_start_s = time.monotonic() + args.common_start_delay_s
    specs = build_client_shard_specs(
        experiment_id=plan.experiment_id,
        endpoints=plan.endpoints,
        expected_identities=plan.expected_server_identities,
        expected_runtime_configs=plan.expected_runtime_configs,
        payload_source=source,
        corpus_digest=_sha256(source),
        scheduled_start_s=scheduled_start_s,
        per_endpoint_rate=args.per_endpoint_rate,
        drivers_per_endpoint=args.drivers_per_endpoint,
        offers_per_endpoint=args.offers_per_endpoint,
        output_dir=output_dir,
        # Each rate receives a deterministic non-overlapping corpus slot while cap
        # comparisons share the same descriptor/rate population.
        corpus_start_index=sorted(SUPPORTED_PER_ENDPOINT_RATES).index(args.per_endpoint_rate) * len(plan.endpoints) * args.offers_per_endpoint,
        wire_format=args.wire_format,
    )
    sampler = NvmlSampler(
        gpu_ids=gpu_ids,
        gpu_uuids=gpu_uuids,
        experiment_id=plan.experiment_id,
        release_digest=plan.application_release.release_digest,
        interval_s=args.nvml_interval_s,
    )
    try:
        result = run_parent(
            specs,
            gpu_ids=gpu_ids,
            gpu_uuids=gpu_uuids,
            nvml_sampler=sampler,
            output_path=output_dir / "client_sharding_summary.json",
        )
    except ClientShardRunInvalid as exc:
        print(f"client-sharding aggregate invalid: {exc}", file=sys.stderr)
        return 2
    print(f"run_dir={output_dir}")
    print(f"offered_rate_per_s={result.offered_rate_per_s:.6f}")
    print(f"completed_rate_per_s={result.completed_rate_per_s:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
