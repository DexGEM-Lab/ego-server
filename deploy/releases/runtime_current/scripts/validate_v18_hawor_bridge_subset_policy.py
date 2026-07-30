#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STRICT_TIER = "strict_candidate_recompute_only"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_case_common(case: dict[str, Any], failures: list[str]) -> None:
    require(case.get("foundation_acceptance_from_policy") is False, f"{case.get('case')}: policy must not accept foundation", failures)
    require(case.get("metric_hand_state_acceptance_from_policy") is False, f"{case.get('case')}: policy must not accept metric hand state", failures)
    require(case.get("downstream_acceptance_from_policy") is False, f"{case.get('case')}: policy must not accept downstream state", failures)
    require(case.get("contact_occlusion_nonpenetration_recomputed") is False, f"{case.get('case')}: policy must not recompute downstream physics", failures)
    require(case.get("policy_rows_with_true_acceptance_flags") == 0, f"{case.get('case')}: policy rows contain true acceptance flags", failures)
    require(case.get("annotation_true_acceptance_flags") == 0, f"{case.get('case')}: annotation bridge quality rows contain true acceptance flags", failures)
    for row in case.get("policy_rows", []) if isinstance(case.get("policy_rows"), list) else []:
        if not isinstance(row, dict):
            failures.append(f"{case.get('case')}: non-dict policy row")
            continue
        require(row.get("foundation_acceptance_from_policy") is False, f"{case.get('case')}: row foundation acceptance true/missing", failures)
        require(row.get("metric_hand_state_acceptance_from_policy") is False, f"{case.get('case')}: row metric acceptance true/missing", failures)
        require(row.get("downstream_acceptance_from_policy") is False, f"{case.get('case')}: row downstream acceptance true/missing", failures)


def validate_trash(case: dict[str, Any], failures: list[str]) -> None:
    validate_case_common(case, failures)
    require(case.get("status") == "candidate_subset_policy_built_not_foundation_acceptance", f"trash status unexpected {case.get('status')}", failures)
    require(case.get("quality_state_status") == "hawor_bridge_quality_candidate_state_built_not_accepted", f"trash quality status unexpected {case.get('quality_state_status')}", failures)
    require(case.get("bridge_candidate_rows") == 2098, f"trash bridge rows expected 2098 got {case.get('bridge_candidate_rows')}", failures)
    rows = case.get("policy_rows") if isinstance(case.get("policy_rows"), list) else []
    require(len(rows) == 2098, f"trash policy rows expected 2098 got {len(rows)}", failures)
    counts = case.get("policy_counts") if isinstance(case.get("policy_counts"), dict) else {}
    expected = {
        "strict_candidate_recompute_only": 1297,
        "projection_supported_review_only": 75,
        "moderate_residual_review_only": 152,
        "no_current_reference_blocked": 189,
        "tail_or_visibility_conflict_blocked": 59,
        "in_frame_conflict_blocked": 157,
        "unsupported_or_large_residual_blocked": 169,
    }
    for key, value in expected.items():
        require(counts.get(key) == value, f"trash policy count {key} expected {value} got {counts.get(key)}", failures)
    require(sum(int(v) for v in counts.values()) == 2098, f"trash policy counts must sum to 2098 got {sum(int(v) for v in counts.values())}", failures)
    thresholds = case.get("strict_gate_thresholds") if isinstance(case.get("strict_gate_thresholds"), dict) else {}
    require(thresholds.get("median_residual_px_max") == 50.0, f"trash strict median threshold unexpected {thresholds}", failures)
    require(thresholds.get("p95_residual_px_max") == 100.0, f"trash strict p95 threshold unexpected {thresholds}", failures)
    require(thresholds.get("hawor_inside_current_bbox_fraction_min") == 0.95, f"trash strict bbox threshold unexpected {thresholds}", failures)
    require(case.get("existing_contact_acceptance_audit_rows_in_strict_candidate_queue") == 223, f"trash strict contact rows expected 223 got {case.get('existing_contact_acceptance_audit_rows_in_strict_candidate_queue')}", failures)
    require(case.get("existing_contact_acceptance_audit_rows_total") == 371, f"trash contact total expected 371 got {case.get('existing_contact_acceptance_audit_rows_total')}", failures)
    require(case.get("existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue") == 0, f"trash strict occlusion rows expected 0 got {case.get('existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue')}", failures)
    require(case.get("existing_occlusion_acceptance_audit_rows_total") == 165, f"trash occlusion total expected 165 got {case.get('existing_occlusion_acceptance_audit_rows_total')}", failures)
    require(case.get("existing_contact_nonpenetration_hands_in_strict_candidate_queue") == 223, f"trash strict nonpenetration hands expected 223 got {case.get('existing_contact_nonpenetration_hands_in_strict_candidate_queue')}", failures)
    require(case.get("existing_contact_nonpenetration_hands_total") == 371, f"trash nonpenetration total expected 371 got {case.get('existing_contact_nonpenetration_hands_total')}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in [
        "policy_is_candidate_queue_only_not_foundation_acceptance",
        "task5_hawor_absent_blocks_all_cases_requirement",
        "downstream_physics_not_recomputed_or_accepted",
        "tail_or_visibility_conflict_rows_blocked_from_policy_use",
        "in_frame_conflict_rows_blocked_from_policy_use",
    ]:
        require(blocker in blockers, f"trash missing blocker {blocker}", failures)
    # Strict rows must really satisfy the gate and must not include occlusion examples.
    for row in rows:
        if row.get("policy_tier") == STRICT_TIER:
            require(row.get("quality_state") == "projection_supported_visible_hawor_bridge_candidate", f"trash strict row has quality {row.get('quality_state')}", failures)
            require(float(row.get("projection_residual_px_median") or 9999) <= 50.0, "trash strict row median residual above gate", failures)
            require(float(row.get("projection_residual_px_p95") or 9999) <= 100.0, "trash strict row p95 residual above gate", failures)
            require(float(row.get("hawor_projected_inside_current_bbox_fraction") or -1) >= 0.95, "trash strict row bbox fraction below gate", failures)


def validate_task5(case: dict[str, Any], failures: list[str]) -> None:
    validate_case_common(case, failures)
    require(case.get("quality_state_status") == "blocked_no_hawor_bridge_candidates_for_case", f"task5 quality status unexpected {case.get('quality_state_status')}", failures)
    require(case.get("expected_frame_side_rows") == 1920, f"task5 expected rows expected 1920 got {case.get('expected_frame_side_rows')}", failures)
    require(case.get("policy_rows") == [], f"task5 policy rows should be empty got {len(case.get('policy_rows', [])) if isinstance(case.get('policy_rows'), list) else 'non-list'}", failures)
    require(case.get("policy_counts") == {}, f"task5 policy counts should be empty got {case.get('policy_counts')}", failures)
    require(case.get("existing_contact_acceptance_audit_rows_in_strict_candidate_queue") == 0, f"task5 strict contact rows expected 0 got {case.get('existing_contact_acceptance_audit_rows_in_strict_candidate_queue')}", failures)
    require(case.get("existing_contact_acceptance_audit_rows_total") == 808, f"task5 contact total expected 808 got {case.get('existing_contact_acceptance_audit_rows_total')}", failures)
    require(case.get("existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue") == 0, f"task5 strict occlusion expected 0 got {case.get('existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue')}", failures)
    require(case.get("existing_occlusion_acceptance_audit_rows_total") == 1, f"task5 occlusion total expected 1 got {case.get('existing_occlusion_acceptance_audit_rows_total')}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    require("quality_state_blocked_no_hawor_bridge_candidates_for_case" in blockers, "task5 missing no-bridge blocker", failures)


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = args.root / "hawor_bridge_state" / "v18_hawor_bridge_subset_policy_summary.json"
    require(path.exists(), f"missing subset policy summary {path}", failures)
    summary = load_json(path) if path.exists() else {}
    require(summary.get("status") == "candidate_subset_policy_built_not_foundation_acceptance", f"summary status unexpected {summary.get('status')}", failures)
    require(summary.get("claim_scope") == "subset_policy_only_for_future_candidate_recompute_no_foundation_or_downstream_acceptance", f"summary claim scope unexpected {summary.get('claim_scope')}", failures)
    require(summary.get("all_cases_policy_foundation_accepted") is False, "summary policy foundation accepted must be false", failures)
    require(summary.get("all_cases_metric_hand_state_accepted_from_policy") is False, "summary metric hand accepted must be false", failures)
    require(summary.get("all_cases_downstream_accepted_from_policy") is False, "summary downstream accepted must be false", failures)
    by_case = {case.get("case"): case for case in summary.get("cases", []) if isinstance(case, dict)}
    require(set(by_case) == {"trash_1050", "task5_tomato_960"}, f"unexpected cases {sorted(by_case)}", failures)
    if "trash_1050" in by_case:
        validate_trash(by_case["trash_1050"], failures)
    if "task5_tomato_960" in by_case:
        validate_task5(by_case["task5_tomato_960"], failures)
    return {
        "method": "validate_v18_hawor_bridge_subset_policy",
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
