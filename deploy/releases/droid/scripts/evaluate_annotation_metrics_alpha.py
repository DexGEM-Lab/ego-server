#!/usr/bin/env python3
"""Compute annotation metric observations from prediction and GT row files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego_annotation.evaluators import evaluate_hands, evaluate_head_camera, load_rows, merge_observations
from ego_annotation.metrics import build_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-pred", type=Path)
    parser.add_argument("--head-gt", type=Path)
    parser.add_argument("--hand-pred", type=Path)
    parser.add_argument("--hand-gt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-status", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = []
    if args.head_pred and args.head_gt:
        parts.append(evaluate_head_camera(load_rows(args.head_pred), load_rows(args.head_gt)))
    if args.hand_pred:
        parts.append(evaluate_hands(load_rows(args.hand_pred), load_rows(args.hand_gt) if args.hand_gt else None))
    observations = merge_observations(*parts)
    rows = build_metric_rows(observations, [], calibration_status=args.calibration_status)
    payload = {"metric_observations": observations, "validation_metrics": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    measured = sum(1 for row in rows if row["status"] == "measured")
    print(json.dumps({"output": str(args.output), "metrics_measured": measured, "metrics_total": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
