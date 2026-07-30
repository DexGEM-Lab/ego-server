#!/usr/bin/env python3
"""Compare two V19 HOT3D MANO3D evaluator reports.

This is an evaluator-phase utility. It consumes already-frozen prediction
reports and never reads or mutates runtime prediction state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "wrist_error_m",
    "joint_mpjpe_m",
    "joint_median_error_m",
    "root_aligned_mpjpe_m",
    "root_aligned_median_error_m",
    "vertex_centroid_error_m",
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    if payload.get("status") != "ok":
        raise RuntimeError(f"{path} status is not ok: {payload.get('status')}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def metric_summary(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    summary = report.get("summary", {})
    value = summary.get(key) if isinstance(summary, dict) else None
    return value if isinstance(value, dict) else None


def compare_metric(baseline: dict[str, Any], candidate: dict[str, Any], key: str) -> dict[str, Any] | None:
    b = metric_summary(baseline, key)
    c = metric_summary(candidate, key)
    if not b or not c:
        return None
    out: dict[str, Any] = {}
    for stat in ("mean", "median", "p10", "p90", "p95", "min", "max"):
        bv = b.get(stat)
        cv = c.get(stat)
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            out[f"baseline_{stat}_m"] = float(bv)
            out[f"candidate_{stat}_m"] = float(cv)
            out[f"delta_{stat}_vs_baseline_m"] = float(cv) - float(bv)
    if "count" in b or "count" in c:
        out["baseline_count"] = b.get("count")
        out["candidate_count"] = c.get("count")
    return out


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_json(args.baseline_report)
    candidate = load_json(args.candidate_report)
    b_summary = baseline.get("summary", {})
    c_summary = candidate.get("summary", {})
    metrics = {k: v for k in METRIC_KEYS if (v := compare_metric(baseline, candidate, k)) is not None}
    rows = {
        "baseline_matched_rows": b_summary.get("matched_rows") if isinstance(b_summary, dict) else None,
        "candidate_matched_rows": c_summary.get("matched_rows") if isinstance(c_summary, dict) else None,
        "baseline_measurable_rows": b_summary.get("measurable_rows") if isinstance(b_summary, dict) else None,
        "candidate_measurable_rows": c_summary.get("measurable_rows") if isinstance(c_summary, dict) else None,
    }
    payload = {
        "status": "ok",
        "method": "compare_v19_hot3d_mano3d_reports",
        "claim_scope": args.claim_scope,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "reports": {
            "baseline": str(args.baseline_report),
            "candidate": str(args.candidate_report),
        },
        "rows": rows,
        "metrics": metrics,
        "interpretation_notes": [
            "Negative delta means candidate has lower error than baseline for that statistic.",
            "Root-aligned metrics subtract wrist translation only; they are not rotation/scale Procrustes metrics.",
            "This utility compares hand MANO evaluator reports only; it does not score contact, occlusion, nonpenetration, or object pose.",
        ],
    }
    write_json(args.output_report, payload)
    print(json.dumps(payload, indent=2)[:20000])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--claim-scope",
        default=(
            "HOT3D camera-coordinate 21-joint MANO report comparison; not contact, "
            "occlusion, nonpenetration, object-pose, or full-vertex scoring"
        ),
    )
    return parser.parse_args()


def main() -> None:
    compare(parse_args())


if __name__ == "__main__":
    main()
