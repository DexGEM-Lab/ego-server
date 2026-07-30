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


def validate_case(case: dict[str, Any], failures: list[str]) -> None:
    name = str(case.get("case"))
    require(case.get("status") == "candidate_coverage_audit_not_downstream_recompute_or_acceptance", f"{name}: bad status {case.get('status')}", failures)
    require(case.get("accepted_contact_or_occlusion_input_flags") == 0, f"{name}: accepted contact/occlusion flags should be zero", failures)
    if name == "trash_1050":
        require(case.get("hawor_bridge_quality_candidate_hand_rows") == 2098, f"trash expected 2098 quality rows, got {case.get('hawor_bridge_quality_candidate_hand_rows')}", failures)
        require(case.get("hawor_bridge_projection_supported_hand_rows") == 1372, f"trash expected 1372 supported quality rows, got {case.get('hawor_bridge_projection_supported_hand_rows')}", failures)
        require(case.get("existing_contact_acceptance_audit_rows") == 371, f"trash contact rows expected 371, got {case.get('existing_contact_acceptance_audit_rows')}", failures)
        require(case.get("existing_contact_rows_with_projection_supported_hawor_bridge") == 237, f"trash supported contact rows expected 237, got {case.get('existing_contact_rows_with_projection_supported_hawor_bridge')}", failures)
        require(case.get("existing_occlusion_acceptance_audit_rows") == 165, f"trash occlusion rows expected 165, got {case.get('existing_occlusion_acceptance_audit_rows')}", failures)
        require(case.get("existing_occlusion_rows_with_projection_supported_hawor_bridge") == 11, f"trash supported occlusion rows expected 11, got {case.get('existing_occlusion_rows_with_projection_supported_hawor_bridge')}", failures)
        require(case.get("existing_contact_nonpenetration_hands") == 371, f"trash contact/nonpenetration hands expected 371, got {case.get('existing_contact_nonpenetration_hands')}", failures)
        require(case.get("existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge") == 237, f"trash supported contact/nonpenetration hands expected 237, got {case.get('existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge')}", failures)
        blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
        for blocker in ["coverage_audit_counts_existing_hypotheses_only_not_recompute", "HaWoR_bridge_quality_candidates_not_accepted_foundation", "task5_hawor_absent_blocks_all_cases_requirement"]:
            require(blocker in blockers, f"trash missing blocker {blocker}", failures)
    elif name == "task5_tomato_960":
        require(case.get("hawor_bridge_quality_candidate_hand_rows") == 0, f"task5 expected 0 quality rows, got {case.get('hawor_bridge_quality_candidate_hand_rows')}", failures)
        require(case.get("existing_contact_acceptance_audit_rows") == 808, f"task5 contact rows expected 808, got {case.get('existing_contact_acceptance_audit_rows')}", failures)
        require(case.get("existing_contact_rows_with_projection_supported_hawor_bridge") == 0, f"task5 supported contact rows should be 0", failures)
        require(case.get("existing_occlusion_acceptance_audit_rows") == 1, f"task5 occlusion rows expected 1, got {case.get('existing_occlusion_acceptance_audit_rows')}", failures)
        require(case.get("existing_occlusion_rows_with_projection_supported_hawor_bridge") == 0, f"task5 supported occlusion rows should be 0", failures)
        blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
        require("case_has_no_hawor_bridge_candidates" in blockers, "task5 missing no-bridge blocker", failures)
    else:
        failures.append(f"unexpected case {name}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = args.root / "hawor_bridge_state" / "v18_hawor_bridge_downstream_coverage_summary.json"
    require(path.exists(), f"missing {path}", failures)
    summary = load_json(path) if path.exists() else {}
    require(summary.get("status") == "candidate_coverage_audit_not_downstream_recompute_or_acceptance", f"summary bad status {summary.get('status')}", failures)
    require(summary.get("all_cases_downstream_accepted") is False, "downstream accepted flag must be false", failures)
    require(summary.get("claim_scope") == "coverage_only_no_contact_occlusion_nonpenetration_acceptance_no_full_V18_closure", f"bad claim scope {summary.get('claim_scope')}", failures)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    require(len(cases) == 2, f"expected 2 cases, got {len(cases)}", failures)
    for case in cases:
        if isinstance(case, dict):
            validate_case(case, failures)
        else:
            failures.append("non-dict case in summary")
    return {"method": "validate_v18_hawor_bridge_downstream_coverage", "status": "ok" if not failures else "failed", "root": str(args.root), "failures": failures}


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
