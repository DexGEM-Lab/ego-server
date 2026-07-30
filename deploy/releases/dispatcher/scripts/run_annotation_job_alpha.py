#!/usr/bin/env python3
"""Run one ego.annotation.output alpha job from a JSON request."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego_annotation import AnnotationJobRequest, AnnotationJobRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON request payload for POST /v1/annotation-jobs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    request = AnnotationJobRequest.from_mapping(payload)
    result = AnnotationJobRunner().run(request)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
