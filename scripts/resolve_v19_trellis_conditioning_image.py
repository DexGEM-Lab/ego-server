#!/usr/bin/env python3
"""Resolve the object-isolated image that conditions TRELLIS in V19 rigid branches.

The P11 evidence-bundle report contains several image paths: raw frame, mask, and
an object crop. TRELLIS must be conditioned on the object crop, not the raw scene.
This helper intentionally has no fallback to raw_frame_path because that silently
changes the physical mechanism from per-instance mesh reconstruction to scene
prior generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_path(value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"missing or invalid {key}: {value!r}")
    return Path(value)


def resolve_crop(report_path: Path) -> Path:
    if not report_path.exists() or report_path.stat().st_size <= 0:
        raise SystemExit(f"evidence report is missing or empty: {report_path}")
    data = json.loads(report_path.read_text())
    if data.get("status") != "ok":
        raise SystemExit(f"evidence report status is {data.get('status')!r}, expected 'ok': {report_path}")

    selected = data.get("selected")
    if not isinstance(selected, dict):
        raise SystemExit(f"evidence report lacks selected object evidence: {report_path}")
    crop_record = selected.get("trellis_conditioning_crop")
    if not isinstance(crop_record, dict):
        raise SystemExit(
            "evidence report lacks selected.trellis_conditioning_crop; "
            "P12 must not fall back to the raw frame"
        )

    crop = _as_path(crop_record.get("crop_rgba"), key="selected.trellis_conditioning_crop.crop_rgba")
    raw = selected.get("raw_frame_path")
    mask = selected.get("mask_path")
    if isinstance(raw, str) and Path(raw) == crop:
        raise SystemExit(f"TRELLIS conditioning image resolves to raw frame, not object crop: {crop}")
    if isinstance(mask, str) and Path(mask) == crop:
        raise SystemExit(f"TRELLIS conditioning image resolves to mask path, not RGBA object crop: {crop}")
    if crop.suffix.lower() not in {".png", ".webp"}:
        raise SystemExit(f"TRELLIS conditioning crop must be an object RGBA image, got {crop}")
    if "/crops/" not in crop.as_posix():
        raise SystemExit(f"TRELLIS conditioning image is not under an evidence crops directory: {crop}")
    if not crop.exists() or crop.stat().st_size <= 0:
        raise SystemExit(f"TRELLIS conditioning crop is missing or empty: {crop}")
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-report", required=True, type=Path)
    args = parser.parse_args()
    print(resolve_crop(args.evidence_report))


if __name__ == "__main__":
    main()
