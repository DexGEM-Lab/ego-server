#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict, key: str) -> float | None:
    value = summary.get(key, {}).get("median")
    return None if value is None else float(value)


def surface_row(job: dict) -> dict:
    path = Path(job["contact_reliability_json"])
    if not path.exists():
        return {
            "track_id": job["track_id"],
            "status": "missing_reliability_json",
            "contact_reliability_json": str(path),
        }
    report = load_json(path)
    measured = report.get("summary_measured_high_score", {})
    row = {
        "track_id": job["track_id"],
        "status": str(report.get("status")),
        "rows": int(report.get("rows", 0)),
        "measured_high_score_rows": int(report.get("measured_high_score_rows", 0)),
        "reliable_contact_rows": int(report.get("reliable_contact_rows", 0)),
        "median_reprojection_px": metric(measured, "joint_reprojection_px"),
        "median_mano_minus_metric_depth_m": metric(measured, "mano_minus_metric_depth_m"),
        "median_contact_gap_m": metric(measured, "contact_gap_m"),
        "condition_counts_measured_high_score": report.get("condition_counts_measured_high_score", {}),
        "contact_reliability_json": str(path),
    }
    row["passes_surface_contact"] = bool(row["reliable_contact_rows"] > 0)
    return row


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.postprocess_manifest)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"postprocess manifest has no jobs: {args.postprocess_manifest}")
    rows = [surface_row(job) for job in jobs]
    passing = [row for row in rows if row.get("passes_surface_contact")]
    out = {
        "status": "ok",
        "postprocess_manifest": str(args.postprocess_manifest),
        "surface_tracks": len(rows),
        "passing_surface_tracks": len(passing),
        "passes_v3_surface_branch": bool(passing),
        "interpretation": (
            "A surface track supports contact only when its reliability diagnostic reports at least one reliable row. "
            "Missing reliability JSONs mean SAM2/postprocess has not run for that surface yet."
        ),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
