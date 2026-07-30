#!/usr/bin/env python3
"""Deprecated V21 GroundingDINO bbox runner.

GroundingDINO bbox detection is disabled for V21 by user request because its
object recognition is currently unreliable for this harness. Use
`scripts/run_v21_owlv2_bbox_proposals.py` or explicit agent object-plan bbox /
point prompts instead.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root")
    parser.add_argument("--text-prompt")
    parser.add_argument("--num-keyframes", type=int)
    parser.add_argument("--box-threshold", type=float)
    parser.add_argument("--text-threshold", type=float)
    _args = parser.parse_args()
    print(
        json.dumps(
            {
                "status": "disabled",
                "reason": "GroundingDINO bbox detection is disabled for V21 by user request.",
                "replacement": "scripts/run_v21_owlv2_bbox_proposals.py or explicit agent object-plan bbox/point prompts",
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
