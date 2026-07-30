#!/usr/bin/env python3
"""Run the direct one-GPU UniDepth physical-batch control on an authorized server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ego_annotation.serving.benchmark.manifest import load_payload_manifest
from ego_annotation.serving.benchmark.measurement import NvmlSampler
from ego_annotation.serving.benchmark.unidepth_fixed_batch import REQUIRED_FIXED_BATCHES, run_fixed_batch_sweep
from ego_annotation.serving.router import ModelApiName
from ego_annotation.serving.unidepth import _load_unidepth_backend, build_unidepth_model_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direct UniDepth B=1/2/4/8/16 physical-forward benchmark")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--payload-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup-forwards", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--include-batch32-after-headroom-review", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to mix evidence into existing output: {args.out}")
    manifest = load_payload_manifest(args.payload_dir / "unidepth.infer.json", expected_api=ModelApiName.UNIDEPTH_INFER)
    batch_sizes = (*REQUIRED_FIXED_BATCHES, 32) if args.include_batch32_after_headroom_review else REQUIRED_FIXED_BATCHES
    selected = manifest.items[:max(batch_sizes)]
    if len(selected) < max(batch_sizes):
        parser.error("corpus lacks the distinct tensors required by the requested maximum physical batch")
    tensors, hashes = [], []
    for item in selected:
        rgb = next((part for part in item.parts if part.name == "rgb"), None)
        if rgb is None or rgb.dtype != "uint8" or rgb.shape != (540, 960, 3):
            parser.error(f"payload {item.item_id} is not a canonical 540x960 uint8 RGB tensor")
        tensors.append(np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.shape))
        hashes.append(item.payload_hash)
    args.out.mkdir(parents=True)
    sampler = NvmlSampler(gpu_ids=(args.gpu,), gpu_uuids={args.gpu: args.gpu_uuid}, experiment_id=args.experiment_id,
                          release_digest=args.release_digest, interval_s=.2)
    config = build_unidepth_model_config(checkpoint=args.checkpoint, model_revision=selected[0].model_revision,
        assigned_gpu=args.gpu, performance_instrumentation=True, device="cuda")
    sampler.start()
    try:
        # The factory constructs one exact resident model; run_fixed_batch_sweep only invokes infer.
        report = run_fixed_batch_sweep(_load_unidepth_backend(config), tensors=tensors, payload_hashes=hashes,
                                       batch_sizes=batch_sizes, warmup_forwards=args.warmup_forwards, repeats=args.repeats)
    finally:
        sampler.stop(); sampler.write(args.out / "gpu_samples.json")
    report.update({"experiment_id": args.experiment_id, "gpu_id": args.gpu, "gpu_uuid": args.gpu_uuid,
                   "release_digest": args.release_digest, "checkpoint_digest": args.checkpoint_digest,
                   "corpus_digest": manifest.manifest_hash, "model_load_count": 1,
                   "execution_boundary": "direct resident backend; no HTTP, Ray request batching, or client load generator"})
    (args.out / "fixed_batch_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
