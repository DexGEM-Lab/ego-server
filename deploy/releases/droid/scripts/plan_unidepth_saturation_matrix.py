#!/usr/bin/env python3
"""Emit the exact one-GPU UniDepth saturation treatment matrix; no GPU work occurs."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class Treatment:
    name: str
    batch_cap: int
    batch_wait_ms: int
    max_concurrent_forwards: int
    per_endpoint_rate: int


REQUIRED_TREATMENTS = (
    *(Treatment(f"cap8-wait20-c1-rate{rate}", 8, 20, 1, rate) for rate in (8, 12, 16, 24)),
    *(Treatment(f"cap16-wait20-c1-rate{rate}", 16, 20, 1, rate) for rate in (16, 24, 32)),
    *(Treatment(f"cap16-wait50-c1-rate{rate}", 16, 50, 1, rate) for rate in (16, 24)),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan, but never launch, the required UniDepth saturation matrix")
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--application-release", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--payload-dir", required=True)
    parser.add_argument("--drivers-per-endpoint", type=int, default=4)
    args = parser.parse_args(argv)
    if args.drivers_per_endpoint < 1 or 200 % args.drivers_per_endpoint:
        parser.error("--drivers-per-endpoint must divide 200 so every driver receives an equal independent payload shard")
    rows = []
    for treatment in REQUIRED_TREATMENTS:
        experiment_id = f"{args.experiment_prefix}-{treatment.name}"
        shared = (
            f"--experiment-id {experiment_id} --gpus {args.gpu} --gpu-uuids {args.gpu_uuid} "
            f"--run-root {args.run_root} --application-release {args.application_release} "
            f"--source-sha {args.source_sha} --checkpoint-digest {args.checkpoint_digest} "
            f"--experiment-batch-cap {treatment.batch_cap} --experiment-batch-wait-ms {treatment.batch_wait_ms} "
            f"--max-concurrent-forwards {treatment.max_concurrent_forwards}"
        )
        rows.append({
            **asdict(treatment), "experiment_id": experiment_id,
            "fresh_output_root": f"{args.run_root}/{experiment_id}",
            "fixed_physical_batch_control": (
                f"python scripts/run_unidepth_fixed_batch_benchmark.py --experiment-id {experiment_id}-fixed-batch "
                f"--gpu {args.gpu} --gpu-uuid {args.gpu_uuid} --release-digest {args.release_digest} "
                f"--checkpoint {args.checkpoint} --checkpoint-digest {args.checkpoint_digest} "
                f"--payload-dir {args.payload_dir} --out {args.run_root}/{experiment_id}/fixed_batch"
            ),
            "launch_plan": f"python scripts/plan_unidepth_scaling_experiment.py {shared}",
            "service_burst_control": (
                f"python scripts/run_unidepth_service_burst.py --endpoint http://127.0.0.1:PORT/unidepth.infer "
                f"--payload-dir {args.payload_dir} --wave-size {treatment.batch_cap} --waves 25 "
                f"--out {args.run_root}/{experiment_id}/service_burst_b{treatment.batch_cap}.json"
            ),
            "benchmark": (
                f"python scripts/run_unidepth_client_sharding_benchmark.py {shared} --payload-dir {args.payload_dir} "
                f"--per-endpoint-rate {treatment.per_endpoint_rate} --drivers-per-endpoint {args.drivers_per_endpoint} "
                "--offers-per-endpoint 200"
            ),
        })
    print(json.dumps({
        "schema": "ego.unidepth-saturation-matrix.v1",
        "prediction": "cap alone cannot enlarge batches without compatible queue backlog; multi-driver overload must precede cap attribution. c1 bounds peak simultaneous forwards to one.",
        "treatments": rows,
        "optional_after_headroom": "Matched current-forward-behavior (no semaphore) is allowed only after allocator/NVML headroom from all required c1 rows is reviewed.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
