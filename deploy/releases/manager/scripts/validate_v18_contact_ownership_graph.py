#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = str(report.get("case"))
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: contact ownership rows missing")
    params = report.get("parameters") if isinstance(report.get("parameters"), dict) else {}
    contact_tol = finite_float(params.get("contact_tolerance_m"), 0.01)
    margin_min = finite_float(params.get("accept_energy_margin"), 0.25)
    assoc_min = finite_float(params.get("association_accept_iou"), 0.15)
    require(report.get("contact_ownership_complete") is False, f"{case}: overclaims complete contact ownership")
    require(report.get("signed_nonpenetration_solved") is False, f"{case}: overclaims signed nonpenetration")
    accepted = 0
    selected = 0
    for raw in rows:
        require(isinstance(raw, dict), f"{case}: malformed row")
        claim = raw.get("contact_owner_claim")
        assignment = raw.get("graph_assignment")
        if raw.get("selected_by_contact_graph") is True:
            selected += 1
            require(isinstance(assignment, dict), f"{case}: selected row lacks assignment")
        if raw.get("accepted_contact_owner") is True:
            accepted += 1
            require(claim == "accepted_contact_owner_by_temporal_mesh_distance_graph", f"{case}: accepted row has wrong claim")
            require(isinstance(assignment, dict), f"{case}: accepted row lacks assignment")
            dist = raw.get("min_hand_surface_to_v16_object_mesh_m")
            require(isinstance(dist, (int, float)) and math.isfinite(float(dist)), f"{case}: accepted row lacks finite distance")
            require(float(dist) <= contact_tol + 1e-12, f"{case}: accepted row exceeds contact tolerance")
            require(finite_float(assignment.get("v16_to_v18_mesh_association_iou"), 0.0) >= assoc_min, f"{case}: accepted row has weak V16/V18 association")
            require(finite_float(assignment.get("unary_energy_margin"), 0.0) >= margin_min, f"{case}: accepted row lacks energy margin")
            require(assignment.get("source_row_blockers") == [], f"{case}: accepted row has blockers")
            require(assignment.get("nonpenetration_status") == "unsigned_surface_distance_only_signed_nonpenetration_unresolved", f"{case}: accepted row overclaims nonpenetration")
        else:
            require(claim in {"temporal_graph_selected_not_accepted", "not_selected_by_contact_graph"}, f"{case}: unsupported nonaccepted claim {claim}")
    require(selected == int(report.get("contact_graph_selected_rows", -1)), f"{case}: selected count mismatch")
    require(accepted == int(report.get("contact_ownership_accepted_rows", -1)), f"{case}: accepted count mismatch")
    require(accepted > 0, f"{case}: no accepted contact owner rows")
    return {"case": case, "contact_graph_selected_rows": selected, "contact_ownership_accepted_rows": accepted, "signed_nonpenetration_solved": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    cases = [validate_case(args.root / case / "v18_contact_ownership_graph_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
