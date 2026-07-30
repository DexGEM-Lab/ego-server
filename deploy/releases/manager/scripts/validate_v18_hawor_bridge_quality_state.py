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


def validate_trash(case: dict[str, Any], failures: list[str]) -> None:
    require(case.get("status") == "hawor_bridge_quality_candidate_state_built_not_accepted", f"trash status unexpected: {case.get('status')}", failures)
    require(case.get("accepted_v18_hawor_foundation") is False, "trash quality state must not accept foundation", failures)
    require(case.get("v18_physical_hand_state_valid_from_quality") is False, "trash quality state must not validate physical hand state", failures)
    require(case.get("contact_occlusion_nonpenetration_recomputed") is False, "trash downstream recomputation must remain false", failures)
    require(case.get("bridge_candidate_rows") == 2098, f"trash bridge rows expected 2098, got {case.get('bridge_candidate_rows')}", failures)
    require(case.get("missing_bridge_rows") == 2, f"trash missing bridge rows expected 2, got {case.get('missing_bridge_rows')}", failures)
    counts = case.get("quality_counts") if isinstance(case.get("quality_counts"), dict) else {}
    expected_counts = {
        "projection_supported_visible_hawor_bridge_candidate": 1365,
        "projection_supported_nonvisible_hawor_bridge_candidate": 7,
        "moderate_residual_uncertain_hawor_bridge_candidate": 152,
        "unsupported_residual_uncertain_hawor_bridge_candidate": 114,
        "large_residual_in_frame_conflict_candidate": 157,
        "large_residual_uncertain_bridge_candidate": 55,
        "residual_tail_hawor_out_of_frame_or_visibility_conflict": 59,
        "no_current_projection_reference_candidate": 189,
    }
    for key, value in expected_counts.items():
        require(counts.get(key) == value, f"trash quality count {key} expected {value}, got {counts.get(key)}", failures)
    require(sum(int(v) for v in counts.values()) == 2098, f"trash quality counts must sum to 2098, got {sum(int(v) for v in counts.values())}", failures)
    require(case.get("accepted_like_quality_state_names") == [], f"trash quality names must not contain accepted-like states: {case.get('accepted_like_quality_state_names')}", failures)
    residual = case.get("projection_residual_px_median_per_row") if isinstance(case.get("projection_residual_px_median_per_row"), dict) else {}
    supported = case.get("supported_candidate_projection_residual_px_median_per_row") if isinstance(case.get("supported_candidate_projection_residual_px_median_per_row"), dict) else {}
    require(residual.get("count") == 1909, f"trash reference rows expected 1909, got {residual.get('count')}", failures)
    require(30.0 <= float(residual.get("median", -1.0)) <= 36.0, f"trash residual median unexpected: {residual}", failures)
    require(float(residual.get("p95", -1.0)) > 600.0, f"trash residual p95 should preserve tail blocker: {residual}", failures)
    require(supported.get("count") == 1372, f"trash supported candidate rows expected 1372, got {supported.get('count')}", failures)
    require(float(supported.get("p95", 999.0)) < 50.0, f"trash supported p95 should be below 50 px: {supported}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in [
        "quality_state_is_candidate_only_not_foundation_acceptance",
        "task5_hawor_absent_blocks_all_cases_requirement",
        "contact_occlusion_nonpenetration_not_recomputed_from_quality_state",
        "residual_tail_requires_visibility_reference_localization_before_downstream_use",
    ]:
        require(blocker in blockers, f"trash missing blocker {blocker}", failures)
    rows = case.get("quality_rows") if isinstance(case.get("quality_rows"), list) else []
    require(len(rows) == 2098, f"trash expected 2098 quality rows, got {len(rows)}", failures)
    if rows:
        require(all("accepted" not in str(row.get("quality_state", "")) for row in rows), "trash row quality_state contains accepted", failures)
    # Overlay is a QC artifact produced by a separate renderer. If present in the standard location, validate it.
    root = Path(str(case.get("bridge_report", ""))).parents[2] if case.get("bridge_report") else None
    if root is not None:
        overlay_path = root / "hawor_bridge_state" / "trash_1050" / "v18_hawor_bridge_quality_overlay_report.json"
        if overlay_path.exists():
            overlay = load_json(overlay_path)
            require(overlay.get("claim_scope") == "full_timeline_QC_overlay_for_HaWoR_bridge_candidates_not_foundation_acceptance", f"trash overlay claim scope unexpected: {overlay.get('claim_scope')}", failures)
            require(overlay.get("frame_count") == 1050, f"trash overlay frame_count expected 1050, got {overlay.get('frame_count')}", failures)
            frame_counts = overlay.get("frame_counts") if isinstance(overlay.get("frame_counts"), dict) else {}
            require(frame_counts.get("video") == 1050, f"trash overlay video frame count expected 1050, got {frame_counts.get('video')}", failures)
            draw_counts = overlay.get("draw_counts_by_quality_state") if isinstance(overlay.get("draw_counts_by_quality_state"), dict) else {}
            require(draw_counts == counts, f"trash overlay draw counts must match quality counts, got {draw_counts} vs {counts}", failures)


def validate_task5(case: dict[str, Any], failures: list[str]) -> None:
    require(case.get("status") == "blocked_no_hawor_bridge_candidates_for_case", f"task5 status unexpected: {case.get('status')}", failures)
    require(case.get("accepted_v18_hawor_foundation") is False, "task5 quality state must not accept foundation", failures)
    require(case.get("v18_physical_hand_state_valid_from_quality") is False, "task5 physical hand state must be false", failures)
    require(case.get("expected_frame_side_rows") == 1920, f"task5 expected 1920 frame-side rows, got {case.get('expected_frame_side_rows')}", failures)
    require(case.get("quality_counts") == {}, f"task5 quality counts should be empty, got {case.get('quality_counts')}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    for blocker in ["case_hawor_world_hands_npz_missing", "HaWoR_repo_weights_or_MANO_assets_missing_locally"]:
        require(blocker in blockers, f"task5 missing blocker {blocker}", failures)


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = args.root / "hawor_bridge_state" / "v18_hawor_bridge_quality_state_summary.json"
    require(path.exists(), f"missing quality summary {path}", failures)
    summary = load_json(path) if path.exists() else {}
    require(summary.get("status") == "candidate_quality_state_built_not_foundation_accepted", f"summary status unexpected: {summary.get('status')}", failures)
    require(summary.get("all_cases_quality_foundation_accepted") is False, "summary all_cases_quality_foundation_accepted must be false", failures)
    require(summary.get("v18_physical_hand_state_valid_from_quality") is False, "summary physical hand state from quality must be false", failures)
    require(summary.get("claim_scope") == "HaWoR_bridge_quality_state_no_model_substitution_no_full_V18_closure", f"summary claim scope unexpected: {summary.get('claim_scope')}", failures)
    by_case = {case.get("case"): case for case in summary.get("cases", []) if isinstance(case, dict)}
    require(set(by_case) == {"trash_1050", "task5_tomato_960"}, f"unexpected cases {sorted(by_case)}", failures)
    if "trash_1050" in by_case:
        validate_trash(by_case["trash_1050"], failures)
    if "task5_tomato_960" in by_case:
        validate_task5(by_case["task5_tomato_960"], failures)
    return {
        "method": "validate_v18_hawor_bridge_quality_state",
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
