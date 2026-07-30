#!/usr/bin/env python3
"""Aggregate fixed-slice HOT3D 3D MANO evaluation reports."""
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
    "vertex_centroid_error_m",
]


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
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    clip_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for report_path in args.reports:
        report = load_json(report_path)
        if report.get("status") != "ok":
            raise RuntimeError(f"report {report_path} is not ok: {report.get('status')}")
        clip_id = next((part for part in reversed(report_path.parts) if part.startswith("clip-")), report_path.parent.name)
        summary = report.get("summary", {})
        clip_reports.append(
            {
                "clip_id": clip_id,
                "report": str(report_path),
                "matched_rows": summary.get("matched_rows"),
                "measurable_rows": summary.get("measurable_rows"),
                "match_rate": summary.get("match_rate"),
                "wrist_error_m_median": summary.get("wrist_error_m", {}).get("median") if isinstance(summary.get("wrist_error_m"), dict) else None,
                "joint_mpjpe_m_median": summary.get("joint_mpjpe_m", {}).get("median") if isinstance(summary.get("joint_mpjpe_m"), dict) else None,
                "root_aligned_mpjpe_m_median": summary.get("root_aligned_mpjpe_m", {}).get("median") if isinstance(summary.get("root_aligned_mpjpe_m"), dict) else None,
            }
        )
        for row in report.get("rows", []):
            if isinstance(row, dict):
                row = dict(row)
                row["clip_id"] = clip_id
                rows.append(row)
    measurable = [r for r in rows if r.get("measurable")]
    matched = [r for r in rows if r.get("matched")]
    by_side: dict[str, Any] = {}
    for side in ("left", "right"):
        sm = [r for r in matched if r.get("side") == side]
        sr = [r for r in measurable if r.get("side") == side]
        by_side[side] = {
            "measurable_rows": len(sr),
            "matched_rows": len(sm),
            "match_rate": float(len(sm) / max(1, len(sr))),
            **{key: summarize([float(r[key]) for r in sm if key in r and r[key] is not None]) for key in METRIC_KEYS},
        }
    detected = [r for r in matched if r.get("pred_detected_same_frame") is True]
    infilled = [r for r in matched if r.get("pred_detected_same_frame") is False]
    payload = {
        "status": "ok",
        "method": "aggregate_v19_hot3d_mano3d_evals",
        "claim_scope": "Fixed-slice aggregate of 3D MANO hand localization/articulation against HOT3D MANO in camera coordinates; root_aligned metrics are wrist-subtracted translation-aligned only; not contact, occlusion, nonpenetration, or object-pose scoring",
        "reports": [str(p) for p in args.reports],
        "clip_count": len(args.reports),
        "clip_summaries": clip_reports,
        "summary": {
            "measurable_rows": len(measurable),
            "matched_rows": len(matched),
            "match_rate": float(len(matched) / max(1, len(measurable))),
            "same_frame_detector_supported_rows": len(detected),
            "infilled_or_not_same_frame_rows": len(infilled),
            "same_frame_detector_supported_rate_among_matched": float(len(detected) / max(1, len(matched))),
            "root_aligned_metric_definition": "Subtract each hand's wrist joint translation before computing joint errors; no rotation or scale alignment is applied.",
            **{key: summarize([float(r[key]) for r in matched if key in r and r[key] is not None]) for key in METRIC_KEYS},
            "by_detection_support": {
                "same_frame_detector_supported": {
                    "rows": len(detected),
                    **{key: summarize([float(r[key]) for r in detected if key in r and r[key] is not None]) for key in METRIC_KEYS},
                },
                "infilled_or_not_same_frame": {
                    "rows": len(infilled),
                    **{key: summarize([float(r[key]) for r in infilled if key in r and r[key] is not None]) for key in METRIC_KEYS},
                },
            },
            "by_side": by_side,
        },
    }
    write_json(args.output_report, payload)
    print(json.dumps(payload, indent=2)[:20000])
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
