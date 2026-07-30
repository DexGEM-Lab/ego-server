#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    summary_path = args.root / "hawor_bridge_state" / "v18_hawor_strict_contact_probe_summary.json"
    require(summary_path.exists(), f"missing summary {summary_path}", failures)
    summary = load_json(summary_path) if summary_path.exists() else {}
    require(summary.get("status") == "candidate_strict_contact_probe_not_acceptance", f"summary status unexpected {summary.get('status')}", failures)
    require(summary.get("claim_scope") == "strict_HaWoR_bridge_contact_proximity_probe_only_no_contact_acceptance_no_nonpenetration_proof", f"summary claim scope unexpected {summary.get('claim_scope')}", failures)
    require(summary.get("contact_acceptance_from_probe") is False, "summary contact acceptance must be false", failures)
    require(summary.get("nonpenetration_acceptance_from_probe") is False, "summary nonpenetration acceptance must be false", failures)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    require(len(cases) == 1, f"expected one case, got {len(cases)}", failures)
    case = cases[0] if cases and isinstance(cases[0], dict) else {}
    require(case.get("case") == "trash_1050", f"case unexpected {case.get('case')}", failures)
    require(case.get("status") == "candidate_strict_contact_probe_not_acceptance", f"case status unexpected {case.get('status')}", failures)
    require(case.get("strict_policy_hand_rows") == 1297, f"strict policy rows expected 1297 got {case.get('strict_policy_hand_rows')}", failures)
    require(case.get("strict_contact_rows_evaluated") == 223, f"strict contact rows expected 223 got {case.get('strict_contact_rows_evaluated')}", failures)
    require(case.get("strict_contact_rows_missing_hand_vertices") == 0, f"missing hand vertices expected 0 got {case.get('strict_contact_rows_missing_hand_vertices')}", failures)
    require(case.get("strict_contact_rows_missing_visible_surface") == 0, f"missing visible surface expected 0 got {case.get('strict_contact_rows_missing_visible_surface')}", failures)
    require(case.get("contact_acceptance_from_probe") is False, "case contact acceptance must be false", failures)
    require(case.get("nonpenetration_acceptance_from_probe") is False, "case nonpenetration acceptance must be false", failures)
    require(case.get("downstream_physics_recomputed_or_accepted") is False, "case downstream physics must not be accepted", failures)
    cats = case.get("source_contact_rows_by_category") if isinstance(case.get("source_contact_rows_by_category"), dict) else {}
    expected_cats = {
        "graph_selected_not_contact_accepted": 60,
        "source_graph_candidate_local_no_penetration_open_mesh_not_strict": 1,
        "source_graph_candidate_local_penetration_veto": 162,
    }
    require(cats == expected_cats, f"source categories unexpected {cats}", failures)
    thresholds = case.get("distance_threshold_counts") if isinstance(case.get("distance_threshold_counts"), dict) else {}
    expected_thresholds = {"distance_le_1cm": 4, "distance_le_3cm": 11, "distance_le_5cm": 12, "distance_le_10cm": 19}
    require(thresholds == expected_thresholds, f"distance thresholds unexpected {thresholds}", failures)
    dist = case.get("hawor_hand_to_visible_object_surface_min_m") if isinstance(case.get("hawor_hand_to_visible_object_surface_min_m"), dict) else {}
    require(dist.get("count") == 223, f"distance count expected 223 got {dist.get('count')}", failures)
    require(float(dist.get("median", 0.0)) > 0.40, f"distance median should preserve large 3D mismatch anomaly: {dist}", failures)
    require(float(dist.get("p05", 0.0)) > 0.04, f"distance p05 should remain nontrivial: {dist}", failures)
    delta = case.get("distance_delta_hawor_minus_source_m") if isinstance(case.get("distance_delta_hawor_minus_source_m"), dict) else {}
    require(delta.get("count") == 223, f"delta count expected 223 got {delta.get('count')}", failures)
    require(float(delta.get("median", 0.0)) > 0.40, f"delta median should preserve mismatch vs source graph distance: {delta}", failures)
    depth_gap = case.get("camera_depth_gap_hawor_minus_object_m") if isinstance(case.get("camera_depth_gap_hawor_minus_object_m"), dict) else {}
    require(depth_gap.get("count") == 223, f"depth gap count expected 223 got {depth_gap.get('count')}", failures)
    require(float(depth_gap.get("median", 0.0)) > 0.50, f"depth gap median should preserve HaWoR-behind-object anomaly: {depth_gap}", failures)
    require(float(depth_gap.get("p05", 0.0)) > 0.15, f"depth gap p05 should preserve broad mismatch: {depth_gap}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in [
        "probe_uses_visible_open_object_surfaces_only_not_complete_mesh_or_sdf",
        "task5_hawor_absent_blocks_all_cases_requirement",
        "contact_not_accepted_without_complete_nonpenetration_and_foundation_state",
        "probe_restricted_to_trash_strict_candidate_queue_only",
    ]:
        require(blocker in blockers, f"missing blocker {blocker}", failures)
    rows = case.get("rows") if isinstance(case.get("rows"), list) else []
    require(len(rows) == 223, f"expected 223 rows got {len(rows)}", failures)
    for row in rows:
        if not isinstance(row, dict):
            failures.append("non-dict row")
            continue
        require(row.get("policy_tier") == "strict_candidate_recompute_only", f"row has wrong policy tier {row.get('policy_tier')}", failures)
        require(row.get("contact_acceptance_from_probe") is False, "row contact acceptance must be false", failures)
        require(row.get("nonpenetration_acceptance_from_probe") is False, "row nonpenetration acceptance must be false", failures)
        require(row.get("state_role") == "HaWoR_strict_bridge_visible_surface_proximity_probe_not_contact_or_nonpenetration_acceptance", f"row state_role unexpected {row.get('state_role')}", failures)
    return {
        "method": "validate_v18_hawor_strict_contact_probe",
        "status": "ok" if not failures else "failed",
        "root": str(args.root),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
