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
    case = str(report.get("case"))
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: signed rows missing")
    require(report.get("signed_nonpenetration_complete") is False, f"{case}: overclaims complete signed nonpenetration")
    source_rows = report.get("source_contact_rows")
    if isinstance(source_rows, int):
        require(source_rows >= len(rows), f"{case}: more signed rows than source contact rows")
    require("support-gated HaWoR" in str(report.get("claim")) or "support-gated" in str(report.get("claim")), f"{case}: report claim does not name support-gated HaWoR mechanism")
    evaluated = 0
    penetration = 0
    watertight = 0
    support_blocked = 0
    physical_ineligible = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        require(row.get("signed_nonpenetration_complete") is False, f"{case}: row overclaims complete nonpenetration")
        claim = row.get("signed_nonpenetration_claim")
        if claim in {"depth_fused_mesh_normal_penetration_evidence", "depth_fused_mesh_normal_no_penetration_beyond_tolerance_evidence"}:
            evaluated += 1
            require(row.get("local_signed_distance_semantics") == "nearest_face_centroid_normal_projection_on_depth_fused_completion_mesh_not_ground_truth_sdf", f"{case}: signed semantics missing")
            require(row.get("hand_geometry_source") == "HaWoR_metric_MANO_full_surface_reference_current_V18_world", f"{case}: row does not consume HaWoR MANO surface")
            require(row.get("hand_support_state") == "observed_same_frame_detection", f"{case}: evaluated signed row is not observed HaWoR support")
            require(row.get("object_mesh_backend") in {"depth_fused_poisson_visible_completion_candidate", "depth_fused_convex_hull_visible_completion_candidate"}, f"{case}: row does not consume depth-fused completion mesh")
            require(isinstance(row.get("min_local_signed_distance_m"), (int, float)), f"{case}: missing signed distance")
            if row.get("mesh_watertight_by_edges") is True:
                watertight += 1
            if row.get("local_penetration_detected") is True:
                penetration += 1
        else:
            require(claim in {"blocked", "not_evaluated_object_not_strict_rigid_nonpenetration_eligible"}, f"{case}: unsupported signed claim {claim}")
            if row.get("blocker") == "hand_not_observed_hawor_support_for_nonpenetration_claim":
                support_blocked += 1
            if row.get("blocker") == "object_not_strict_rigid_nonpenetration_eligible":
                physical_ineligible += 1
                require(row.get("strict_nonpenetration_eligibility") == "strict_rigid_nonpenetration_not_eligible", f"{case}: physical-ineligible row missing eligibility state")
                require(isinstance(row.get("strict_nonpenetration_eligibility_blockers"), list) and row.get("strict_nonpenetration_eligibility_blockers"), f"{case}: physical-ineligible row missing blockers")
    require(evaluated == int(report.get("evaluated_signed_rows", -1)), f"{case}: evaluated count mismatch")
    require(penetration == int(report.get("local_penetration_detected_rows", -1)), f"{case}: penetration count mismatch")
    require(watertight == int(report.get("mesh_watertight_rows", -1)), f"{case}: watertight count mismatch")
    require(watertight > 0 or physical_ineligible > 0, f"{case}: neither watertight evaluation nor physical-eligibility blocker rows are present")
    require(support_blocked == int(report.get("support_blocked_rows", -1)), f"{case}: support-blocked count mismatch")
    return {"case": case, "signed_rows": len(rows), "evaluated_signed_rows": evaluated, "support_blocked_rows": support_blocked, "physical_ineligible_rows": physical_ineligible, "local_penetration_detected_rows": penetration, "mesh_watertight_rows": watertight, "signed_nonpenetration_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_signed_nonpenetration_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_signed_nonpenetration_evidence_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
