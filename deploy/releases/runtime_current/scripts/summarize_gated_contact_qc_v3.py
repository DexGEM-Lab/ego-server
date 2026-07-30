#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_volume_sdf_contact_v3 import summarize


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enabled_frames(deform_report: dict) -> set[int]:
    frames = set()
    for row in deform_report.get("rows", []):
        if any(bool(contact.get("contact_constraint_enabled", False)) for contact in row.get("contact_rows", [])):
            frames.add(int(row["frame_idx"]))
    return frames


def summarize_rows(contact_report: dict, frames: set[int]) -> dict:
    rows = [row for row in contact_report.get("rows", []) if int(row["frame_idx"]) in frames]
    values = []
    abs_values = []
    penetration_values = []
    near_values = []
    for row in rows:
        vals = []
        summary = row.get("sdf_m", {})
        if "median" not in summary:
            raise RuntimeError(f"row {row.get('frame_idx')} missing sdf summary")
        values.append(float(summary["median"]))
        abs_values.append(float(row.get("abs_sdf_m", {}).get("median", abs(float(summary["median"])))))
        penetration_values.append(float(row.get("penetration_fraction", 0.0)))
        near_values.append(float(row.get("near_surface_fraction", 0.0)))
    if not rows:
        return {"frames": [], "rows": 0}
    return {
        "frames": sorted(int(v) for v in frames),
        "rows": int(len(rows)),
        "row_median_sdf_m": summarize(values),
        "row_median_abs_sdf_m": summarize(abs_values),
        "row_penetration_fraction": summarize(penetration_values),
        "row_near_surface_fraction": summarize(near_values),
        "rows_detail": rows,
    }


def run(args: argparse.Namespace) -> dict:
    deform_report = json.loads(args.deform_report.read_text(encoding="utf-8"))
    contact_report = json.loads(args.contact_qc.read_text(encoding="utf-8"))
    frames = enabled_frames(deform_report)
    report = {
        "status": "ok",
        "method": "summarize_gated_contact_qc_v3",
        "deform_report": str(args.deform_report),
        "contact_qc": str(args.contact_qc),
        "enabled_contact_frames": sorted(int(v) for v in frames),
        "summary_enabled_frames": summarize_rows(contact_report, frames),
    }
    save_json(args.output_json, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deform-report", type=Path, required=True)
    parser.add_argument("--contact-qc", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
