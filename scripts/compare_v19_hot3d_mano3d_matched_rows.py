#!/usr/bin/env python3
"""Compare HOT3D MANO3D evaluator reports on identical frame/side rows.

This avoids the denominator artifact where a full HaWoR baseline report has more
matched rows than an interval-state candidate.  It consumes evaluator outputs
only and does not read or modify prediction state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRIC_KEYS = [
    "wrist_error_m",
    "joint_mpjpe_m",
    "joint_median_error_m",
    "root_aligned_mpjpe_m",
    "root_aligned_median_error_m",
    "root_aligned_p95_error_m",
    "vertex_centroid_error_m",
]


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"{path} is not an ok evaluator report")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} lacks rows[]")
    return payload


def row_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["frame_idx"]), str(row["side"])


def usable(row: dict[str, Any]) -> bool:
    return bool(row.get("measurable")) and bool(row.get("matched"))


def summarize(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_report(args.baseline_report)
    candidate = load_report(args.candidate_report)
    b_rows = {row_key(r): r for r in baseline["rows"] if isinstance(r, dict) and usable(r)}
    c_rows = {row_key(r): r for r in candidate["rows"] if isinstance(r, dict) and usable(r)}
    if args.match_on == "candidate":
        keys = sorted(k for k in c_rows if k in b_rows)
    elif args.match_on == "intersection":
        keys = sorted(set(b_rows) & set(c_rows))
    else:
        raise RuntimeError(f"unsupported match mode {args.match_on}")
    if not keys:
        raise RuntimeError("no common matched frame/side rows")
    metrics: dict[str, Any] = {}
    max_abs_delta_by_metric: dict[str, float] = {}
    for key in METRIC_KEYS:
        pairs: list[tuple[float, float]] = []
        for k in keys:
            bv = b_rows[k].get(key)
            cv = c_rows[k].get(key)
            if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
                pairs.append((float(bv), float(cv)))
        if not pairs:
            continue
        baseline_vals = [b for b, _ in pairs]
        candidate_vals = [c for _, c in pairs]
        deltas = [c - b for b, c in pairs]
        metrics[key] = {
            "baseline": summarize(baseline_vals),
            "candidate": summarize(candidate_vals),
            "delta_candidate_minus_baseline": summarize(deltas),
            "max_abs_per_row_delta_m": float(max(abs(d) for d in deltas)),
        }
        max_abs_delta_by_metric[key] = float(max(abs(d) for d in deltas))
    by_side: dict[str, Any] = {}
    for side in sorted({s for _, s in keys}):
        side_keys = [k for k in keys if k[1] == side]
        by_side[side] = {"row_count": len(side_keys), "frame_min": min(k[0] for k in side_keys), "frame_max": max(k[0] for k in side_keys)}
    payload = {
        "status": "ok",
        "method": "compare_v19_hot3d_mano3d_matched_rows",
        "claim_scope": args.claim_scope,
        "baseline_report": str(args.baseline_report),
        "candidate_report": str(args.candidate_report),
        "match_on": args.match_on,
        "row_count": len(keys),
        "baseline_usable_rows": len(b_rows),
        "candidate_usable_rows": len(c_rows),
        "frame_side_keys_preview": [{"frame_idx": f, "side": s} for f, s in keys[:20]],
        "by_side": by_side,
        "metrics": metrics,
        "max_abs_delta_by_metric_m": max_abs_delta_by_metric,
        "interpretation_notes": [
            "Rows are matched by exact frame_idx and side and require measurable=true, matched=true in both reports.",
            "Negative delta means candidate lower error; zero delta means the metric prediction is identical on common rows.",
            "This compares only HOT3D camera-coordinate MANO localization, not contact, occlusion, nonpenetration, or object pose.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:20000])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--match-on", choices=("candidate", "intersection"), default="candidate")
    parser.add_argument(
        "--claim-scope",
        default="HOT3D MANO3D evaluator comparison on identical frame/side rows; not contact, occlusion, nonpenetration, or object-pose scoring",
    )
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
