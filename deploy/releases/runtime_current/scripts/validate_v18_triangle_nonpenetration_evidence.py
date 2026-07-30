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


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = str(report.get("case"))
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: rows missing")
    require(report.get("triangle_nonpenetration_complete") is False, f"{case}: overclaims complete triangle nonpenetration")
    require("support-gated HaWoR MANO" in str(report.get("claim")) or "support-gated" in str(report.get("claim")), f"{case}: report claim does not name support-gated HaWoR mechanism")
    require(report.get("depth_fused_mesh_object_count", 0) > 0, f"{case}: no depth-fused mesh objects available")
    evaluated = 0
    penetration = 0
    watertight = 0
    support_blocked = 0
    physical_ineligible = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        require(row.get("triangle_nonpenetration_complete") is False, f"{case}: row overclaims complete nonpenetration")
        claim = row.get("triangle_nonpenetration_claim")
        if claim in {"depth_fused_mesh_triangle_penetration_evidence", "depth_fused_mesh_triangle_no_penetration_beyond_tolerance_evidence"}:
            evaluated += 1
            require("depth_fused_completion_mesh" in str(row.get("local_triangle_signed_distance_semantics")), f"{case}: signed semantics do not name depth-fused completion mesh candidate")
            require("not_complete_object_ground_truth_sdf" in str(row.get("triangle_nonpenetration_scope")), f"{case}: row scope overstates nonpenetration proof")
            require(row.get("hand_geometry_source") == "HaWoR_metric_MANO_full_surface_reference_current_V18_world", f"{case}: row does not consume final HaWoR MANO surface")
            require(row.get("hand_support_state") == "observed_same_frame_detection", f"{case}: evaluated nonpenetration row is not observed HaWoR support")
            require(row.get("object_mesh_backend") in {"depth_fused_poisson_visible_completion_candidate", "depth_fused_convex_hull_visible_completion_candidate"}, f"{case}: row does not consume depth-fused object completion mesh")
            require(isinstance(row.get("mesh_watertight_by_edges"), bool), f"{case}: missing watertight diagnostic")
            if row.get("mesh_watertight_by_edges") is True:
                watertight += 1
            for key in ["min_triangle_unsigned_distance_m", "min_local_triangle_signed_distance_m", "negative_triangle_signed_distance_fraction"]:
                val = row.get(key)
                require(isinstance(val, (float, int)) and math.isfinite(float(val)), f"{case}: invalid {key}")
            frac = float(row.get("negative_triangle_signed_distance_fraction"))
            require(0.0 <= frac <= 1.0, f"{case}: invalid negative fraction")
            if row.get("local_triangle_penetration_detected") is True:
                penetration += 1
        else:
            require(claim in {"blocked", "not_evaluated_object_not_strict_rigid_nonpenetration_eligible"}, f"{case}: unsupported claim {claim}")
            if row.get("blocker") == "hand_not_observed_hawor_support_for_nonpenetration_claim":
                support_blocked += 1
            if row.get("blocker") == "object_not_strict_rigid_nonpenetration_eligible":
                physical_ineligible += 1
                require(row.get("strict_nonpenetration_eligibility") == "strict_rigid_nonpenetration_not_eligible", f"{case}: physical-ineligible row missing eligibility state")
                require(isinstance(row.get("strict_nonpenetration_eligibility_blockers"), list) and row.get("strict_nonpenetration_eligibility_blockers"), f"{case}: physical-ineligible row missing blockers")
    require(evaluated == int(report.get("evaluated_triangle_rows", -1)), f"{case}: evaluated count mismatch")
    require(penetration == int(report.get("local_triangle_penetration_detected_rows", -1)), f"{case}: penetration count mismatch")
    require(watertight == int(report.get("mesh_watertight_rows", -1)), f"{case}: watertight count mismatch")
    require(support_blocked == int(report.get("support_blocked_rows", -1)), f"{case}: support-blocked count mismatch")
    require(watertight > 0 or physical_ineligible > 0, f"{case}: neither watertight evaluation nor physical-eligibility blocker rows are present")
    return {"case": case, "triangle_rows": len(rows), "evaluated_triangle_rows": evaluated, "support_blocked_rows": support_blocked, "physical_ineligible_rows": physical_ineligible, "local_triangle_penetration_detected_rows": penetration, "mesh_watertight_rows": watertight, "triangle_nonpenetration_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_triangle_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_triangle_nonpenetration_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
