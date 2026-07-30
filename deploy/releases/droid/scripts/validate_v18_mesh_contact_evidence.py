#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = report.get("case")
    rows = report.get("rows")
    sources_raw = report.get("sources")
    sources: dict[str, Any] = sources_raw if isinstance(sources_raw, dict) else {}
    snapshot = sources.get("v18_full_annotations_snapshot")
    sha = sources.get("v18_full_annotations_sha256")
    require(isinstance(snapshot, str), f"{case}: missing mesh-contact source snapshot")
    snapshot_path = Path(str(snapshot))
    require(snapshot_path.exists(), f"{case}: missing mesh-contact source snapshot")
    require(isinstance(sha, str) and hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == sha, f"{case}: mesh-contact source snapshot hash mismatch")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: contact rows missing")
    require(report.get("contact_ownership_complete") is False, f"{case}: overclaims complete contact ownership")
    require(int(report.get("contact_ownership_accepted_rows", -1)) == 0, f"{case}: should not accept contact ownership")
    finite_rows = 0
    support_values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        require(row.get("contact_owner_claim") == "not_accepted_contact_owner_v16_mesh_distance_evidence_only", f"{case}: row overclaims ownership")
        dist = row.get("min_hand_surface_to_v16_object_mesh_m")
        if isinstance(dist, (float, int)) and math.isfinite(float(dist)):
            finite_rows += 1
            support = row.get("mesh_contact_support_score")
            require(isinstance(support, (float, int)), f"{case}: invalid support score")
            support_float = float(support)  # type: ignore[arg-type]
            require(0.0 <= support_float <= 1.0, f"{case}: invalid support score")
            support_values.append(support_float)
    require(finite_rows > 0, f"{case}: no finite mesh distance rows")
    require(finite_rows == int(report.get("finite_mesh_distance_rows", -1)), f"{case}: finite row count mismatch")
    return {
        "case": case,
        "contact_evidence_rows": len(rows),
        "finite_mesh_distance_rows": finite_rows,
        "max_support": max(support_values) if support_values else None,
        "mean_support": sum(support_values) / len(support_values) if support_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_mesh_contact_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_mesh_contact_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
