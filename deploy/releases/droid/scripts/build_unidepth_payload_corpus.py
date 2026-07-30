#!/usr/bin/env python3
"""Materialize a deterministic V22 multi-video UniDepth tensor payload corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.unidepth_payload_corpus import (
    CorpusBuildError,
    MODEL_REVISION_DEFAULT,
    build_unidepth_payload_corpus,
    parse_source_specs,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode V22 JPEG frames to a fresh, path-free UniDepth payload corpus."
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="V22 input/raw_frame_manifest/manifest.json; repeat once per source")
    parser.add_argument("--source-id", action="append", required=True,
                        help="public path-free source identifier paired with --manifest")
    parser.add_argument("--take-count", action="append", required=True, type=int,
                        help="exact selected frame count paired with --manifest")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="fresh corpus directory; the command refuses an existing path")
    parser.add_argument("--selection-policy", action="append", choices=("source-order", "uniform"),
                        help="optional per-source policy; omit to use --default-selection-policy for every source")
    parser.add_argument("--default-selection-policy", choices=("source-order", "uniform"), default="source-order")
    parser.add_argument("--manifest-id", default="unidepth-v22-multivideo-corpus")
    parser.add_argument("--job-id", default="unidepth-scaling-corpus")
    parser.add_argument("--model-revision", default=MODEL_REVISION_DEFAULT)
    args = parser.parse_args(argv)
    try:
        result = build_unidepth_payload_corpus(
            sources=parse_source_specs(args.manifest, args.source_id, args.take_count, args.selection_policy),
            output_root=args.output_root,
            selection_policy=args.default_selection_policy,
            manifest_id=args.manifest_id,
            job_id=args.job_id,
            model_revision=args.model_revision,
        )
    except CorpusBuildError as exc:
        print(f"corpus build refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_root": str(result.output_root),
        "descriptor": str(result.descriptor_path),
        "item_count": result.item_count,
        "descriptor_sha256": result.descriptor_sha256,
        "distinct_payload_hashes": len(set(result.payload_hashes)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
