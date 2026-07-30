#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

SUPPORTED_PREFIX = "projection_supported"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def quality_state(hand: dict[str, Any]) -> str | None:
    q = hand.get("hawor_bridge_quality_candidate") if isinstance(hand.get("hawor_bridge_quality_candidate"), dict) else None
    if q is None:
        return None
    return str(q.get("status")) if q.get("status") is not None else None


def is_supported(state: str | None) -> bool:
    return bool(state and state.startswith(SUPPORTED_PREFIX))


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    ann_path = args.output_root / case / "annotations_v18_corrective_state.json"
    ann = load_json(ann_path)
    frames = ann.get("frames") if isinstance(ann.get("frames"), list) else []
    q_counts: Counter[str] = Counter()
    hand_rows = 0
    supported_hand_rows = 0
    contact_hands = 0
    contact_rows = 0
    contact_rows_supported = 0
    contact_rows_by_quality: Counter[str] = Counter()
    occlusion_hands = 0
    occlusion_rows = 0
    occlusion_rows_supported = 0
    occlusion_rows_by_quality: Counter[str] = Counter()
    nonpenetration_hands = 0
    nonpenetration_supported = 0
    accepted_contact_or_occlusion_flags = 0
    examples: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        for hand in frame.get("hands", []) if isinstance(frame.get("hands"), list) else []:
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            q = hand.get("hawor_bridge_quality_candidate") if isinstance(hand.get("hawor_bridge_quality_candidate"), dict) else None
            qs = quality_state(hand)
            if qs is not None:
                hand_rows += 1
                q_counts[qs] += 1
                if is_supported(qs):
                    supported_hand_rows += 1
                if q.get("accepted_v18_hawor_foundation") is True or q.get("accepted_metric_hand_state") is True or q.get("accepted_contact_or_occlusion_input") is True:
                    accepted_contact_or_occlusion_flags += 1
            contact_audit = hand.get("contact_acceptance_audit") if isinstance(hand.get("contact_acceptance_audit"), list) else []
            if contact_audit:
                contact_hands += 1
                contact_rows += len(contact_audit)
                contact_rows_by_quality[str(qs)] += len(contact_audit)
                if is_supported(qs):
                    contact_rows_supported += len(contact_audit)
                    if len(examples) < 12:
                        examples.append({"frame_idx": frame_idx, "side": side, "kind": "contact", "quality_state": qs, "rows": len(contact_audit)})
            occlusion_audit = hand.get("occlusion_owner_acceptance_audit") if isinstance(hand.get("occlusion_owner_acceptance_audit"), list) else []
            if occlusion_audit:
                occlusion_hands += 1
                occlusion_rows += len(occlusion_audit)
                occlusion_rows_by_quality[str(qs)] += len(occlusion_audit)
                if is_supported(qs):
                    occlusion_rows_supported += len(occlusion_audit)
                    if len(examples) < 12:
                        examples.append({"frame_idx": frame_idx, "side": side, "kind": "occlusion", "quality_state": qs, "rows": len(occlusion_audit)})
            if isinstance(hand.get("contact_nonpenetration_state"), dict):
                nonpenetration_hands += 1
                if is_supported(qs):
                    nonpenetration_supported += 1
    report = {
        "method": "build_v18_hawor_bridge_downstream_coverage",
        "case": case,
        "status": "candidate_coverage_audit_not_downstream_recompute_or_acceptance",
        "claim_scope": "counts_existing_hypotheses_by_HaWoR_bridge_quality_candidate_only_no_contact_occlusion_nonpenetration_acceptance",
        "source_annotation": str(ann_path),
        "frame_count": len(frames),
        "accepted_contact_or_occlusion_input_flags": accepted_contact_or_occlusion_flags,
        "hawor_bridge_quality_candidate_hand_rows": hand_rows,
        "hawor_bridge_projection_supported_hand_rows": supported_hand_rows,
        "quality_counts": dict(sorted(q_counts.items())),
        "existing_contact_acceptance_audit_rows": contact_rows,
        "existing_contact_acceptance_audit_hands": contact_hands,
        "existing_contact_rows_with_projection_supported_hawor_bridge": contact_rows_supported,
        "existing_contact_rows_by_hawor_bridge_quality": dict(sorted(contact_rows_by_quality.items())),
        "existing_occlusion_acceptance_audit_rows": occlusion_rows,
        "existing_occlusion_acceptance_audit_hands": occlusion_hands,
        "existing_occlusion_rows_with_projection_supported_hawor_bridge": occlusion_rows_supported,
        "existing_occlusion_rows_by_hawor_bridge_quality": dict(sorted(occlusion_rows_by_quality.items())),
        "existing_contact_nonpenetration_hands": nonpenetration_hands,
        "existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge": nonpenetration_supported,
        "examples": examples,
        "blocking_reasons": [
            "coverage_audit_counts_existing_hypotheses_only_not_recompute",
            "HaWoR_bridge_quality_candidates_not_accepted_foundation",
            "task5_hawor_absent_blocks_all_cases_requirement" if case == "trash_1050" else "case_has_no_hawor_bridge_candidates",
        ],
        "elapsed_s": time.perf_counter() - start,
    }
    out_path = args.output_root / "hawor_bridge_state" / case / "v18_hawor_bridge_downstream_coverage_report.json"
    write_json(out_path, report)
    return report


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V18 HaWoR bridge downstream coverage audit",
        "",
        "This audit only counts existing contact/occlusion/nonpenetration hypotheses by HaWoR bridge quality state. It does not recompute or accept contact, occlusion ownership, pose fill, nonpenetration, or V18 closure.",
        "",
        f"Status: `{summary['status']}`",
        f"All cases downstream accepted: `{summary['all_cases_downstream_accepted']}`",
        "",
    ]
    for case in summary["cases"]:
        lines += [
            f"## {case['case']}",
            "",
            f"Quality hand rows: `{case['hawor_bridge_quality_candidate_hand_rows']}`; projection-supported: `{case['hawor_bridge_projection_supported_hand_rows']}`",
            f"Contact audit rows with projection-supported HaWoR bridge: `{case['existing_contact_rows_with_projection_supported_hawor_bridge']}/{case['existing_contact_acceptance_audit_rows']}`",
            f"Occlusion audit rows with projection-supported HaWoR bridge: `{case['existing_occlusion_rows_with_projection_supported_hawor_bridge']}/{case['existing_occlusion_acceptance_audit_rows']}`",
            f"Contact/nonpenetration hand rows with projection-supported HaWoR bridge: `{case['existing_contact_nonpenetration_hands_with_projection_supported_hawor_bridge']}/{case['existing_contact_nonpenetration_hands']}`",
            f"Blocking reasons: `{case['blocking_reasons']}`",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_hawor_bridge_downstream_coverage",
        "status": "candidate_coverage_audit_not_downstream_recompute_or_acceptance",
        "claim_scope": "coverage_only_no_contact_occlusion_nonpenetration_acceptance_no_full_V18_closure",
        "output_root": str(args.output_root),
        "all_cases_downstream_accepted": False,
        "cases": cases,
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_bridge_state"
    write_json(out_dir / "v18_hawor_bridge_downstream_coverage_summary.json", summary)
    write_markdown(out_dir / "V18_HAWOR_BRIDGE_DOWNSTREAM_COVERAGE.md", summary)
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
