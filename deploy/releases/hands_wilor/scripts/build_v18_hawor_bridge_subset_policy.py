#!/usr/bin/env python3
"""Build a candidate-only HaWoR bridge subset policy for V18.

This policy is a guardrail for future experiments. It does not accept HaWoR as a
V18 foundation, does not accept metric hand state, and does not recompute or
accept contact/occlusion/nonpenetration.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

STRICT_MEDIAN_RESIDUAL_PX = 50.0
STRICT_P95_RESIDUAL_PX = 100.0
STRICT_IMAGE_INSIDE_FRACTION = 0.8
STRICT_BBOX_INSIDE_FRACTION = 0.95

STRICT_TIER = "strict_candidate_recompute_only"
REVIEW_SUPPORTED_TIER = "projection_supported_review_only"
MODERATE_TIER = "moderate_residual_review_only"
NO_REFERENCE_TIER = "no_current_reference_blocked"
TAIL_TIER = "tail_or_visibility_conflict_blocked"
IN_FRAME_CONFLICT_TIER = "in_frame_conflict_blocked"
UNSUPPORTED_TIER = "unsupported_or_large_residual_blocked"
NO_BRIDGE_TIER = "no_hawor_bridge_candidate"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def f(value: Any) -> float | None:
    return float(value) if finite_number(value) else None


def strict_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row.get("quality_state") != "projection_supported_visible_hawor_bridge_candidate":
        reasons.append("quality_state_not_visible_projection_supported")
    if row.get("quality_blockers"):
        reasons.append("quality_blockers_present")
    med = f(row.get("projection_residual_px_median"))
    p95 = f(row.get("projection_residual_px_p95"))
    h_inside = f(row.get("hawor_projected_inside_image_fraction"))
    r_inside = f(row.get("reference_projected_inside_image_fraction"))
    bbox_inside = f(row.get("hawor_projected_inside_current_bbox_fraction"))
    if med is None or med > STRICT_MEDIAN_RESIDUAL_PX:
        reasons.append("median_residual_not_within_strict_gate")
    if p95 is None or p95 > STRICT_P95_RESIDUAL_PX:
        reasons.append("p95_residual_not_within_strict_gate")
    if h_inside is None or h_inside < STRICT_IMAGE_INSIDE_FRACTION:
        reasons.append("hawor_projection_not_sufficiently_in_image")
    if r_inside is None or r_inside < STRICT_IMAGE_INSIDE_FRACTION:
        reasons.append("reference_projection_not_sufficiently_in_image")
    if bbox_inside is None or bbox_inside < STRICT_BBOX_INSIDE_FRACTION:
        reasons.append("hawor_projection_not_sufficiently_inside_current_bbox")
    return len(reasons) == 0, reasons


def policy_tier(row: dict[str, Any] | None) -> tuple[str, list[str]]:
    if row is None:
        return NO_BRIDGE_TIER, ["no_hawor_bridge_quality_row"]
    ok, strict_reject_reasons = strict_gate(row)
    if ok:
        return STRICT_TIER, ["candidate_queue_only_not_foundation_or_physics_acceptance"]
    quality = str(row.get("quality_state") or "")
    if quality.startswith("projection_supported"):
        return REVIEW_SUPPORTED_TIER, ["projection_supported_but_failed_strict_gate", *strict_reject_reasons]
    if quality == "moderate_residual_uncertain_hawor_bridge_candidate":
        return MODERATE_TIER, ["moderate_projection_residual_requires_review", *strict_reject_reasons]
    if quality == "no_current_projection_reference_candidate":
        return NO_REFERENCE_TIER, ["missing_current_projection_reference", *strict_reject_reasons]
    if quality == "residual_tail_hawor_out_of_frame_or_visibility_conflict":
        return TAIL_TIER, ["tail_or_visibility_conflict_requires_localization", *strict_reject_reasons]
    if quality == "large_residual_in_frame_conflict_candidate":
        return IN_FRAME_CONFLICT_TIER, ["in_frame_projection_conflict_blocks_use", *strict_reject_reasons]
    return UNSUPPORTED_TIER, ["unsupported_or_large_residual_blocks_use", *strict_reject_reasons]


def index_policy_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        side = str(row.get("side"))
        if side not in {"left", "right"}:
            continue
        out[(int(row.get("frame_idx", -1)), side)] = row
    return out


def annotation_quality(hand: dict[str, Any]) -> dict[str, Any] | None:
    q = hand.get("hawor_bridge_quality_candidate")
    return q if isinstance(q, dict) else None


def annotation_true_acceptance_flag(q: dict[str, Any] | None) -> bool:
    if q is None:
        return False
    return any(q.get(key) is True for key in ["accepted_v18_hawor_foundation", "accepted_metric_hand_state", "accepted_contact_or_occlusion_input"])


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    q_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_quality_state.json"
    ann_path = args.output_root / case / "annotations_v18_corrective_state.json"
    out_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_subset_policy_report.json"
    base = {
        "method": "build_v18_hawor_bridge_subset_policy",
        "case": case,
        "status": "candidate_subset_policy_built_not_foundation_acceptance",
        "claim_scope": "candidate_subset_policy_only_no_HaWoR_foundation_acceptance_no_downstream_physics_acceptance",
        "source_quality_state": str(q_path),
        "source_annotation": str(ann_path),
        "foundation_acceptance_from_policy": False,
        "metric_hand_state_acceptance_from_policy": False,
        "contact_occlusion_nonpenetration_recomputed": False,
        "downstream_acceptance_from_policy": False,
    }
    if not q_path.exists():
        report = {
            **base,
            "status": "blocked_missing_quality_state",
            "policy_rows": [],
            "policy_counts": {},
            "blocking_reasons": ["hawor_bridge_quality_state_missing"],
            "elapsed_s": time.perf_counter() - start,
        }
        write_json(out_path, report)
        return report
    q_report = load_json(q_path)
    rows = q_report.get("quality_rows") if isinstance(q_report.get("quality_rows"), list) else []
    policy_rows: list[dict[str, Any]] = []
    policy_counts: Counter[str] = Counter()
    true_acceptance_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        tier, reasons = policy_tier(row)
        policy_counts[tier] += 1
        if any(row.get(key) is True for key in ["accepted_v18_hawor_foundation", "accepted_metric_hand_state", "accepted_contact_or_occlusion_input"]):
            true_acceptance_rows += 1
        policy_rows.append({
            "frame_idx": int(row.get("frame_idx", -1)),
            "side": str(row.get("side")),
            "quality_state": row.get("quality_state"),
            "policy_tier": tier,
            "policy_reasons": reasons,
            "projection_residual_px_median": row.get("projection_residual_px_median"),
            "projection_residual_px_p95": row.get("projection_residual_px_p95"),
            "hawor_projected_inside_image_fraction": row.get("hawor_projected_inside_image_fraction"),
            "reference_projected_inside_image_fraction": row.get("reference_projected_inside_image_fraction"),
            "hawor_projected_inside_current_bbox_fraction": row.get("hawor_projected_inside_current_bbox_fraction"),
            "current_visibility_state": row.get("current_visibility_state"),
            "candidate_source": "HaWoR_current_V18_camera_local_bridge",
            "foundation_acceptance_from_policy": False,
            "metric_hand_state_acceptance_from_policy": False,
            "downstream_acceptance_from_policy": False,
        })
    policy_index = index_policy_rows(policy_rows)
    contact_by_policy: Counter[str] = Counter()
    occlusion_by_policy: Counter[str] = Counter()
    nonpenetration_by_policy: Counter[str] = Counter()
    annotation_hand_rows_by_policy: Counter[str] = Counter()
    annotation_true_acceptance_flags = 0
    examples: list[dict[str, Any]] = []
    if ann_path.exists():
        ann = load_json(ann_path)
        frames = ann.get("frames") if isinstance(ann.get("frames"), list) else []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_idx = int(frame.get("frame_idx", -1))
            for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
                if not isinstance(hand, dict):
                    continue
                side = str(hand.get("hand_side") or hand.get("side"))
                prow = policy_index.get((frame_idx, side))
                tier = str(prow.get("policy_tier")) if prow else NO_BRIDGE_TIER
                q = annotation_quality(hand)
                if q is not None:
                    annotation_hand_rows_by_policy[tier] += 1
                if annotation_true_acceptance_flag(q):
                    annotation_true_acceptance_flags += 1
                contact_rows = hand.get("contact_acceptance_audit") if isinstance(hand.get("contact_acceptance_audit"), list) else []
                if contact_rows:
                    contact_by_policy[tier] += len(contact_rows)
                    if tier == STRICT_TIER and len(examples) < 12:
                        examples.append({"frame_idx": frame_idx, "side": side, "kind": "contact", "policy_tier": tier, "rows": len(contact_rows)})
                occlusion_rows = hand.get("occlusion_owner_acceptance_audit") if isinstance(hand.get("occlusion_owner_acceptance_audit"), list) else []
                if occlusion_rows:
                    occlusion_by_policy[tier] += len(occlusion_rows)
                    if tier == STRICT_TIER and len(examples) < 12:
                        examples.append({"frame_idx": frame_idx, "side": side, "kind": "occlusion", "policy_tier": tier, "rows": len(occlusion_rows)})
                if isinstance(hand.get("contact_nonpenetration_state"), dict):
                    nonpenetration_by_policy[tier] += 1
    blocking_reasons = [
        "policy_is_candidate_queue_only_not_foundation_acceptance",
        "task5_hawor_absent_blocks_all_cases_requirement" if case == "trash_1050" else "case_has_no_hawor_bridge_candidates",
        "downstream_physics_not_recomputed_or_accepted",
    ]
    if policy_counts.get(TAIL_TIER, 0):
        blocking_reasons.append("tail_or_visibility_conflict_rows_blocked_from_policy_use")
    if policy_counts.get(IN_FRAME_CONFLICT_TIER, 0):
        blocking_reasons.append("in_frame_conflict_rows_blocked_from_policy_use")
    if q_report.get("status") == "blocked_no_hawor_bridge_candidates_for_case":
        blocking_reasons.append("quality_state_blocked_no_hawor_bridge_candidates_for_case")
    report = {
        **base,
        "quality_state_status": q_report.get("status"),
        "expected_frame_side_rows": q_report.get("expected_frame_side_rows"),
        "bridge_candidate_rows": q_report.get("bridge_candidate_rows", len(rows)),
        "policy_rows": policy_rows,
        "policy_counts": dict(sorted(policy_counts.items())),
        "strict_gate_thresholds": {
            "quality_state": "projection_supported_visible_hawor_bridge_candidate",
            "median_residual_px_max": STRICT_MEDIAN_RESIDUAL_PX,
            "p95_residual_px_max": STRICT_P95_RESIDUAL_PX,
            "hawor_and_reference_inside_image_fraction_min": STRICT_IMAGE_INSIDE_FRACTION,
            "hawor_inside_current_bbox_fraction_min": STRICT_BBOX_INSIDE_FRACTION,
        },
        "policy_rows_with_true_acceptance_flags": true_acceptance_rows,
        "annotation_hand_rows_by_policy": dict(sorted(annotation_hand_rows_by_policy.items())),
        "annotation_true_acceptance_flags": annotation_true_acceptance_flags,
        "existing_contact_acceptance_audit_rows_by_policy": dict(sorted(contact_by_policy.items())),
        "existing_contact_acceptance_audit_rows_in_strict_candidate_queue": int(contact_by_policy.get(STRICT_TIER, 0)),
        "existing_contact_acceptance_audit_rows_total": int(sum(contact_by_policy.values())),
        "existing_occlusion_acceptance_audit_rows_by_policy": dict(sorted(occlusion_by_policy.items())),
        "existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue": int(occlusion_by_policy.get(STRICT_TIER, 0)),
        "existing_occlusion_acceptance_audit_rows_total": int(sum(occlusion_by_policy.values())),
        "existing_contact_nonpenetration_hands_by_policy": dict(sorted(nonpenetration_by_policy.items())),
        "existing_contact_nonpenetration_hands_in_strict_candidate_queue": int(nonpenetration_by_policy.get(STRICT_TIER, 0)),
        "existing_contact_nonpenetration_hands_total": int(sum(nonpenetration_by_policy.values())),
        "examples": examples,
        "blocking_reasons": blocking_reasons,
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(out_path, report)
    return report


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V18 HaWoR bridge subset policy",
        "",
        "This is a candidate-only guardrail for future HaWoR bridge experiments. It does not accept HaWoR as the V18 foundation, does not accept metric hand state, and does not recompute or accept contact, occlusion ownership, pose fill, nonpenetration, or full V18 closure.",
        "",
        f"Status: `{summary['status']}`",
        f"All cases policy foundation accepted: `{summary['all_cases_policy_foundation_accepted']}`",
        f"All cases downstream accepted from policy: `{summary['all_cases_downstream_accepted_from_policy']}`",
        "",
        "Strict candidate queue gate:",
        f"- quality state must be `projection_supported_visible_hawor_bridge_candidate`",
        f"- median residual <= `{STRICT_MEDIAN_RESIDUAL_PX}` px",
        f"- p95 residual <= `{STRICT_P95_RESIDUAL_PX}` px",
        f"- HaWoR/reference projections inside image fraction >= `{STRICT_IMAGE_INSIDE_FRACTION}`",
        f"- HaWoR joints inside current bbox fraction >= `{STRICT_BBOX_INSIDE_FRACTION}`",
        "",
    ]
    for case in summary["cases"]:
        lines += [
            f"## {case['case']}",
            "",
            f"Quality status: `{case['quality_state_status']}`",
            f"Policy counts: `{case['policy_counts']}`",
            f"Existing contact rows in strict candidate queue: `{case['existing_contact_acceptance_audit_rows_in_strict_candidate_queue']}/{case['existing_contact_acceptance_audit_rows_total']}`",
            f"Existing occlusion rows in strict candidate queue: `{case['existing_occlusion_acceptance_audit_rows_in_strict_candidate_queue']}/{case['existing_occlusion_acceptance_audit_rows_total']}`",
            f"Existing contact/nonpenetration hands in strict candidate queue: `{case['existing_contact_nonpenetration_hands_in_strict_candidate_queue']}/{case['existing_contact_nonpenetration_hands_total']}`",
            f"Blocking reasons: `{case['blocking_reasons']}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_hawor_bridge_subset_policy",
        "status": "candidate_subset_policy_built_not_foundation_acceptance",
        "claim_scope": "subset_policy_only_for_future_candidate_recompute_no_foundation_or_downstream_acceptance",
        "output_root": str(args.output_root),
        "all_cases_policy_foundation_accepted": False,
        "all_cases_metric_hand_state_accepted_from_policy": False,
        "all_cases_downstream_accepted_from_policy": False,
        "cases": cases,
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_bridge_state"
    write_json(out_dir / "v18_hawor_bridge_subset_policy_summary.json", summary)
    write_markdown(out_dir / "V18_HAWOR_BRIDGE_SUBSET_POLICY.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
