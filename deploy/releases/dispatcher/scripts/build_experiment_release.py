#!/usr/bin/env python3
"""Build and verify a content-addressed experiment application release."""
from __future__ import annotations

import argparse
from pathlib import Path

from ego_annotation.serving.benchmark.release import build_release, verify_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()
    path = build_release(args.source_root, args.output_root, source_sha=args.source_sha, include=args.include or None)
    verified = verify_release(path)
    print(f"release_root={verified.path}")
    print(f"release_digest={verified.release_digest}")
    print(f"source_sha={verified.source_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
