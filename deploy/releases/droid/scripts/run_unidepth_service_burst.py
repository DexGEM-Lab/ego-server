#!/usr/bin/env python3
"""Issue fixed B=8/B=16 process waves and preserve physical Serve batch evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.manifest import load_payload_manifest
from ego_annotation.serving.benchmark.measurement import NvmlSampler
from ego_annotation.serving.benchmark.unidepth_service_burst import run_fixed_process_service_waves
from ego_annotation.serving.router import ModelApiName


def _run(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_payload_manifest(
        args.payload_dir / "unidepth.infer.json", expected_api=ModelApiName.UNIDEPTH_INFER,
        limit=args.wave_size * args.waves,
    )
    if len(manifest.items) != args.wave_size * args.waves:
        raise ValueError("corpus does not contain enough distinct items")
    sampler = NvmlSampler(
        gpu_ids=(args.gpu_id,), gpu_uuids={}, experiment_id=args.experiment_id,
        release_digest=args.release_digest, interval_s=args.nvml_interval_s,
    )
    run_started_s = sampler.start()
    try:
        run = run_fixed_process_service_waves(
            args.endpoint, manifest, wave_size=args.wave_size, wave_count=args.waves,
            release_lead_s=args.release_lead_s, synchronization_window_s=args.synchronization_window_s,
            timeout_s=args.timeout_s, clock=time.monotonic,
        )
    finally:
        run_ended_s = sampler.stop()
    return {
        "schema": "ego.unidepth-service-burst.v2",
        "endpoint": args.endpoint,
        "wave_size": args.wave_size,
        "wave_count": args.waves,
        "run_started_s": run_started_s,
        "run_ended_s": run_ended_s,
        **run.to_dict(),
        "physical_batch_sizes": sorted({record.batch_size for record in run.records if record.batch_size is not None}),
        "nvml": {
            "gpu_ids": [args.gpu_id], "experiment_id": args.experiment_id,
            "release_digest": args.release_digest, "sample_interval_s": args.nvml_interval_s,
            "samples": [sample.to_dict() for sample in sampler.samples],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--wave-size", required=True, type=int, choices=(8, 16))
    parser.add_argument("--waves", type=int, default=25)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--release-lead-s", type=float, default=0.05)
    parser.add_argument("--synchronization-window-s", type=float, default=0.020)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nvml-interval-s", type=float, default=0.2)
    parser.add_argument("--experiment-id", default="unidepth-service-burst")
    parser.add_argument("--release-digest", default="unknown")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing existing output: {args.out}")
    report = _run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
