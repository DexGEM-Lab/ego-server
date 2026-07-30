#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_cotracker_pairwise_rigid_factors_v6 import summarize


def load_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    if data.get("status") != "ok":
        raise RuntimeError(f"pair report is not ok: {path}")
    return data


def row_score(row: dict) -> tuple[int, float, int, float]:
    ready = 1 if row.get("rigid_factor_ready") else 0
    p95 = float(row.get("inlier_residual_m", {}).get("p95", np.inf))
    inliers = int(row.get("inlier_count", 0))
    median = float(row.get("inlier_residual_m", {}).get("median", np.inf))
    return ready, -p95, inliers, -median


def factor_key(row: dict) -> tuple[int, int]:
    return int(row["source_frame"]), int(row["target_frame"])


def compact_pair_row(row: dict) -> dict:
    residual = row.get("inlier_residual_m", {})
    return {
        "source_frame": int(row["source_frame"]),
        "target_frame": int(row["target_frame"]),
        "source_anchor": str(row.get("source_anchor", "")),
        "track_count": int(row.get("track_count", 0)),
        "inlier_count": int(row.get("inlier_count", 0)),
        "rigid_factor_ready": bool(row.get("rigid_factor_ready")),
        "inlier_median_m": residual.get("median"),
        "inlier_p95_m": residual.get("p95"),
        "candidate_count": int(row.get("candidate_count", 0)),
        "ready_candidate_count": int(row.get("ready_candidate_count", 0)),
    }


def run(args: argparse.Namespace) -> dict:
    candidates: dict[tuple[int, int], list[dict]] = {}
    for report_path in args.pair_report:
        report = load_report(report_path)
        anchor = report_path.parent.name
        for row in report.get("pair_rows", []):
            item = dict(row)
            item["source_report"] = str(report_path)
            item["source_anchor"] = anchor
            candidates.setdefault(factor_key(item), []).append(item)

    selected = []
    rejected = []
    for key in sorted(candidates):
        rows = candidates[key]
        best = max(rows, key=row_score)
        out = dict(best)
        out["candidate_count"] = int(len(rows))
        out["ready_candidate_count"] = int(sum(1 for row in rows if row.get("rigid_factor_ready")))
        out["candidate_sources"] = [str(row["source_report"]) for row in rows]
        if best.get("rigid_factor_ready"):
            selected.append(out)
        else:
            rejected.append(out)

    pair_rows = sorted([*selected, *rejected], key=factor_key)
    ready_p95 = [
        float(row.get("inlier_residual_m", {}).get("p95", np.nan))
        for row in selected
    ]
    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "method": "merge_cotracker_pair_factors_v6",
        "claim_tested": "multiple learned CoTracker anchors can provide local graph-ready pair factors over different parts of the object sequence",
        "pair_reports": [str(path) for path in args.pair_report],
        "pair_count": int(len(candidates)),
        "ready_pair_count": int(len(selected)),
        "rejected_pair_count": int(len(rejected)),
        "ready_pair_inlier_p95_m": summarize(np.asarray(ready_p95, dtype=np.float64)),
        "pair_summary_rows": [compact_pair_row(row) for row in pair_rows],
        "pair_rows": pair_rows,
        "selected_pair_rows": selected,
        "rejected_pair_rows": rejected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    omitted = {"pair_rows", "selected_pair_rows", "rejected_pair_rows"}
    print(json.dumps({k: v for k, v in report.items() if k not in omitted}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-report", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
