#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: occlusion rows missing")
    require(report.get("occlusion_ownership_complete") is False, f"{case}: overclaims occlusion ownership completion")
    accepted = 0
    with_support = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        if row.get("accepted_occlusion_owner") is True:
            accepted += 1
        else:
            require(row.get("occlusion_owner_claim") == "not_accepted_owner_without_depth_order_acceptance", f"{case}: unsupported owner claim")
        support = row.get("mesh_contact_temporal_support")
        require(isinstance(support, dict), f"{case}: missing mesh temporal support")
        if support.get("max_support") is not None:
            with_support += 1
    require(accepted == int(report.get("accepted_occlusion_owner_rows", -1)), f"{case}: accepted count mismatch")
    require(with_support == int(report.get("candidate_rows_with_mesh_support", -1)), f"{case}: support count mismatch")
    return {"case": case, "candidate_rows": len(rows), "candidate_rows_with_mesh_support": with_support, "accepted_occlusion_owner_rows": accepted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_mesh_owner_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_occlusion_mesh_owner_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
