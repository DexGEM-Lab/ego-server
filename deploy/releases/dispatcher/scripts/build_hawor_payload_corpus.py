#!/usr/bin/env python3
"""Materialize typed real-frame HaWoR track chunks for envelope benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.hawor_payload_corpus import (
    CorpusBuildError,
    build_hawor_payload_corpus,
    parse_source_specs,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Reconstruct fresh 16-frame HaWoR crop chunks from V22 JPEGs and "
                     "preserved typed HaWoR geometry/evidence requests.")
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="V22 input/raw_frame_manifest/manifest.json; repeat once per source")
    parser.add_argument("--source-id", action="append", required=True,
                        help="path-free source identifier paired with --manifest")
    parser.add_argument("--historic-payload-manifest", type=Path,
                        help="optional preserved ego.benchmark-payload-source.v1 hawor.infer_tracks descriptor")
    parser.add_argument("--evidence-root", action="append", type=Path,
                        help="V22 job root with preserved WiLoR/DROID/UniDepth inputs; pair with every --manifest when no historic descriptor exists")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="fresh corpus directory; existing paths are refused")
    parser.add_argument("--count", type=int,
                        help="exact number of distinct historical typed chunks to materialize")
    parser.add_argument("--manifest-id", default="hawor-v22-track-corpus")
    parser.add_argument("--job-id", default="hawor-envelope-soak")
    args = parser.parse_args(argv)
    try:
        result = build_hawor_payload_corpus(
            sources=parse_source_specs(args.manifest, args.source_id, args.evidence_root),
            historic_payload_manifest=args.historic_payload_manifest,
            output_root=args.output_root,
            count=args.count,
            manifest_id=args.manifest_id,
            job_id=args.job_id,
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
