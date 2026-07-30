#!/usr/bin/env python3
"""Aggregate V19 HOT3D 2D hand-box evaluation reports.

The aggregate is deliberately scoped to the evaluator's claim family: 2D hand-box
localization. It does not synthesize MANO/contact/object-pose metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    clip_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for report_path in args.reports:
        report = load_json(report_path)
        if report.get("method") != "evaluate_v19_hot3d_hawor_boxes":
            raise RuntimeError(f"unexpected method in {report_path}: {report.get('method')}")
        clip_id = report_path.parent.parent.name if report_path.parent.name == "evaluation" else report_path.stem
        report_rows = report.get("rows")
        if not isinstance(report_rows, list):
            raise RuntimeError(f"{report_path} has no rows list; rerun evaluator with full report output")
        for row in report_rows:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched["clip_id"] = clip_id
                rows.append(enriched)
        clip_summaries.append(
            {
                "clip_id": clip_id,
                "report": str(report_path),
                "frame_count_gt": report.get("frame_count_gt"),
                "summary": report.get("summary"),
            }
        )

    measurable = [r for r in rows if r.get("measurable")]
    matched = [r for r in measurable if r.get("matched")]
    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sr = [r for r in measurable if r.get("side") == side]
        sm = [r for r in matched if r.get("side") == side]
        by_side[side] = {
            "measurable_rows": len(sr),
            "matched_rows": len(sm),
            "match_rate": float(len(sm) / max(1, len(sr))),
            "iou": summarize([float(r["iou"]) for r in sm if r.get("iou") is not None]),
            "center_distance_px": summarize([float(r["center_distance_px"]) for r in sm if r.get("center_distance_px") is not None]),
            "pred_score": summarize([float(r["pred_score"]) for r in sm if r.get("pred_score") is not None]),
        }
    payload = {
        "status": "ok",
        "method": "aggregate_v19_hot3d_box_evals",
        "claim_scope": "aggregate 2D HOT3D hand-box localization only; not 3D MANO/contact/object-pose scoring",
        "reports": [str(p) for p in args.reports],
        "clip_summaries": clip_summaries,
        "summary": {
            "clip_count": len(args.reports),
            "measurable_rows": len(measurable),
            "matched_rows": len(matched),
            "match_rate": float(len(matched) / max(1, len(measurable))),
            "iou": summarize([float(r["iou"]) for r in matched if r.get("iou") is not None]),
            "center_distance_px": summarize([float(r["center_distance_px"]) for r in matched if r.get("center_distance_px") is not None]),
            "by_side": by_side,
        },
    }
    write_json(args.output_report, payload)
    print(json.dumps(payload["summary"], indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    aggregate(parse_args())


if __name__ == "__main__":
    main()
