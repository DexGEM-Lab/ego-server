#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


FALSE_READY: dict[str, bool] = {
    "annotation_ready": False,
    "deliverable_ready": False,
    "accuracy_target_met": False,
    "object_geometry_complete": False,
    "object_pose_requirement_met": False,
    "rigid_pose_requirement_met": False,
    "v3_solver_complete": False,
}

STATUS = "v18_part_mask_promotion_gate"
CLAIM = (
    "This artifact gates whether promptable SAM proposal masks can be promoted into V18 part-mask evidence. "
    "Current promptable proposals lack referring/open-vocabulary part semantics and temporal track association, so no "
    "proposal is promoted to an accepted part track, geometry, pose, or contact-ready factor."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def blocker_by_object(blocker_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(blocker_report.get("object_rows"), "blocker object rows"):
        row = require_dict(raw, "blocker row")
        out[require_str(row.get("object_id"), "blocker object_id")] = row
    return out


def proposal_by_object(proposal_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in require_list(proposal_report.get("object_records"), "proposal object records"):
        row = require_dict(raw, "proposal object row")
        out[require_str(row.get("object_id"), "proposal object_id")] = row
    return out


def promotion_state(saved_count: int, promptable_candidates: int, open_vocab_ready: bool, part_prompt_plan_ready: bool) -> tuple[str, list[str]]:
    blockers = [
        "promptable_sam_proposals_are_not_referring_part_tracks",
        "no_temporal_part_track_association",
        "no_semantic_part_label_for_proposals",
    ]
    if not open_vocab_ready:
        blockers.append("open_vocab_or_referring_prompt_backend_not_ready")
    if not part_prompt_plan_ready:
        blockers.append("model_produced_part_prompt_plan_not_ready")
    if saved_count <= 0 and promptable_candidates <= 0:
        return "blocked_no_promptable_proposals", sorted(blockers + ["no_saved_promptable_proposals"])
    return "blocked_promptable_proposals_need_semantic_temporal_validation", sorted(blockers)


def case_report(case: str, args: argparse.Namespace) -> dict[str, Any]:
    blocker_path = args.part_object_blockers_root / case / "v18_part_object_blocker_manifest_report.json"
    proposal_path = args.sam_promptable_proposals_root / case / "v18_sam_promptable_part_proposals_report.json"
    acquisition_path = args.part_mask_acquisition_root / case / "v18_part_mask_acquisition_plan_report.json"
    blocker_report = require_dict(load_json(blocker_path), f"{case} blocker report")
    proposal_report = require_dict(load_json(proposal_path), f"{case} promptable proposals")
    acquisition_report = require_dict(load_json(acquisition_path), f"{case} acquisition report")
    acquisition_env = require_dict(acquisition_report.get("environment"), "acquisition environment")
    open_vocab_ready = bool(acquisition_env.get("open_vocab_or_referring_prompt_backend_available"))
    part_prompt_plan_ready = bool(acquisition_env.get("model_produced_part_prompt_plan_ready"))
    blockers = blocker_by_object(blocker_report)
    proposals = proposal_by_object(proposal_report)
    object_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for object_id in sorted(blockers):
        blocker = blockers[object_id]
        proposal = proposals.get(object_id, {})
        saved_count = require_int(proposal.get("saved_promptable_proposal_mask_count", 0), "saved proposal mask count")
        raw_count = require_int(proposal.get("raw_sam_mask_candidate_count", 0), "raw SAM candidate count")
        proposal_counts = require_dict(proposal_report.get("proposal_state_counts"), "proposal state counts")
        # Use per-object saved/raw counts for readiness, and per-report state counts only for reporting totals.
        object_promptable_candidates = saved_count
        state, state_blockers = promotion_state(saved_count, object_promptable_candidates, open_vocab_ready, part_prompt_plan_ready)
        state_counts[state] += 1
        object_rows.append(
            {
                "case": case,
                "object_id": object_id,
                "source_blocker_state": blocker.get("part_object_blocker_state"),
                "saved_promptable_proposal_mask_count": saved_count,
                "raw_sam_mask_candidate_count": raw_count,
                "promotion_gate_state": state,
                "promotion_blockers": state_blockers,
                "proposal_state_counts_report_scope": proposal_counts,
                "open_vocab_or_referring_prompt_backend_available": open_vocab_ready,
                "model_produced_part_prompt_plan_ready": part_prompt_plan_ready,
                "promoted_part_track_count": 0,
                "mask_evidence_created": False,
                "part_geometry_extraction_ready": False,
                "part_pose_ready": False,
                "object_pose_requirement_met": False,
                "contact_ownership_ready": False,
            }
        )
    report = {
        "method": "build_v18_part_mask_promotion_gate",
        "status": STATUS,
        "claim": CLAIM,
        "case": case,
        "sources": {
            "part_object_blockers": str(blocker_path),
            "sam_promptable_part_proposals": str(proposal_path),
            "part_mask_acquisition_plan": str(acquisition_path),
        },
        "object_count": len(object_rows),
        "promotion_gate_state_counts": dict(sorted(state_counts.items())),
        "saved_promptable_proposal_mask_count": sum(require_int(row.get("saved_promptable_proposal_mask_count"), "saved proposals") for row in object_rows),
        "objects_with_saved_promptable_proposals_count": sum(1 for row in object_rows if require_int(row.get("saved_promptable_proposal_mask_count"), "saved proposals") > 0),
        "promoted_part_track_count": 0,
        "mask_evidence_created_count": 0,
        "part_geometry_extraction_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "contact_ownership_ready_count": 0,
        "object_rows": object_rows,
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / case / "v18_part_mask_promotion_gate_report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    reports = [case_report(case, args) for case in args.cases]
    elapsed = time.perf_counter() - start
    state_counts: Counter[str] = Counter()
    for report in reports:
        state_counts.update(require_dict(report.get("promotion_gate_state_counts"), "promotion state counts"))
    summary = {
        "method": "build_v18_part_mask_promotion_gate",
        "status": STATUS,
        "claim": CLAIM,
        "case_count": len(reports),
        "build_elapsed_s": elapsed,
        "object_count": sum(require_int(report.get("object_count"), "object count") for report in reports),
        "promotion_gate_state_counts": dict(sorted(state_counts.items())),
        "saved_promptable_proposal_mask_count": sum(require_int(report.get("saved_promptable_proposal_mask_count"), "saved proposal count") for report in reports),
        "objects_with_saved_promptable_proposals_count": sum(require_int(report.get("objects_with_saved_promptable_proposals_count"), "objects with proposals") for report in reports),
        "promoted_part_track_count": 0,
        "mask_evidence_created_count": 0,
        "part_geometry_extraction_ready_count": 0,
        "part_pose_ready_count": 0,
        "object_pose_requirement_met_count": 0,
        "contact_ownership_ready_count": 0,
        "cases": [
            {
                "case": report["case"],
                "report_path": str(args.output_root / str(report["case"]) / "v18_part_mask_promotion_gate_report.json"),
                "object_count": report["object_count"],
                "saved_promptable_proposal_mask_count": report["saved_promptable_proposal_mask_count"],
                "promoted_part_track_count": 0,
                "mask_evidence_created_count": 0,
                **FALSE_READY,
            }
            for report in reports
        ],
        "default_path_uses_bundlesdf_or_nerf": False,
        **FALSE_READY,
    }
    write_json(args.output_root / "v18_part_mask_promotion_gate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-object-blockers-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_object_blocker_manifest"))
    parser.add_argument("--sam-promptable-proposals-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals"))
    parser.add_argument("--part-mask-acquisition-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_part_mask_promotion_gate"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
