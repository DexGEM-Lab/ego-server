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
    summary_path = args.root / "hawor_bridge_state" / "v18_hawor_temporal_offset_probe_summary.json"
    require(summary_path.exists(), f"missing summary {summary_path}", failures)
    summary = load_json(summary_path) if summary_path.exists() else {}
    require(summary.get("method") == "build_v18_hawor_temporal_offset_probe", f"method unexpected {summary.get('method')}", failures)
    require(summary.get("status") == "candidate_temporal_offset_probe_not_acceptance", f"status unexpected {summary.get('status')}", failures)
    require(summary.get("claim_scope") == "temporal_offset_mechanism_probe_only_no_contact_acceptance_no_foundation_acceptance", f"claim scope unexpected {summary.get('claim_scope')}", failures)
    require(summary.get("contact_acceptance_from_probe") is False, f"contact acceptance must remain false: {summary.get('contact_acceptance_from_probe')}", failures)
    require(summary.get("nonpenetration_acceptance_from_probe") is False, f"nonpenetration acceptance must remain false: {summary.get('nonpenetration_acceptance_from_probe')}", failures)
    require(summary.get("foundation_acceptance_from_probe") is False, f"foundation acceptance must remain false: {summary.get('foundation_acceptance_from_probe')}", failures)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    require(len(cases) == 1, f"expected one case report, got {len(cases)}", failures)
    case = cases[0] if cases and isinstance(cases[0], dict) else {}
    require(case.get("case") == "trash_1050", f"case unexpected {case.get('case')}", failures)
    require(case.get("max_offset") == 5, f"max_offset unexpected {case.get('max_offset')}", failures)
    require(case.get("rows_evaluated") == 223, f"rows evaluated expected 223 got {case.get('rows_evaluated')}", failures)
    require(case.get("temporal_offset_supports_contact_mismatch_explanation") is False, "temporal offset must not be marked explanatory/supportive", failures)
    require(case.get("interpretation") == "no_consistent_temporal_offset_explains_strict_contact_mismatch", f"interpretation unexpected {case.get('interpretation')}", failures)
    require(case.get("dominant_best_distance_fraction", 1.0) < 0.6, f"dominant distance fraction too high for no-consistent-offset claim: {case.get('dominant_best_distance_fraction')}", failures)
    require(case.get("dominant_best_abs_depth_gap_fraction", 1.0) < 0.6, f"dominant abs-depth fraction too high for no-consistent-offset claim: {case.get('dominant_best_abs_depth_gap_fraction')}", failures)
    baseline = case.get("baseline_offset0_distance_m") if isinstance(case.get("baseline_offset0_distance_m"), dict) else {}
    best = case.get("best_any_offset_distance_m") if isinstance(case.get("best_any_offset_distance_m"), dict) else {}
    baseline_depth = case.get("baseline_offset0_abs_depth_gap_m") if isinstance(case.get("baseline_offset0_abs_depth_gap_m"), dict) else {}
    best_depth = case.get("best_any_offset_abs_depth_gap_m") if isinstance(case.get("best_any_offset_abs_depth_gap_m"), dict) else {}
    require(baseline.get("median") is not None and baseline.get("median") > 0.4, f"baseline distance median should expose mismatch: {baseline}", failures)
    require(best.get("median") is not None and best.get("median") > 0.3, f"best distance median should remain large: {best}", failures)
    require(baseline_depth.get("median") is not None and baseline_depth.get("median") > 0.5, f"baseline depth gap should expose mismatch: {baseline_depth}", failures)
    require(best_depth.get("median") is not None and best_depth.get("median") > 0.5, f"best depth gap should remain large: {best_depth}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in [
        "probe_reuses_candidate_only_trash_HaWoR_bridge_rows",
        "task5_hawor_absent_blocks_all_cases_requirement",
        "visible_surface_open_geometry_cannot_prove_contact_or_nonpenetration",
        "temporal_offset_probe_is_mechanism_diagnostic_not_downstream_recompute",
    ]:
        require(blocker in blockers, f"missing blocker {blocker}", failures)
    report_path = args.root / "hawor_bridge_state" / "trash_1050" / "v18_hawor_temporal_offset_probe_report.json"
    require(report_path.exists(), f"missing case report {report_path}", failures)
    markdown = args.root / "hawor_bridge_state" / "V18_HAWOR_TEMPORAL_OFFSET_PROBE.md"
    require(markdown.exists(), f"missing markdown {markdown}", failures)
    return {
        "method": "validate_v18_hawor_temporal_offset_probe",
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
