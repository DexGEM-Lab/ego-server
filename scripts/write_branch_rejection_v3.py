#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--metric", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = {}
    for item in args.metric:
        if "=" not in item:
            raise RuntimeError(f"metric must be key=value: {item}")
        key, value = item.split("=", 1)
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    payload = {
        "branch": args.branch,
        "status": args.status,
        "accepted": False,
        "reason": args.reason,
        "evidence": [str(path) for path in args.evidence],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
