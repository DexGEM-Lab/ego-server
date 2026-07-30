#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEPTH_SCALE_SUPPORT_STATUS = "depth_scaled_from_projected_hawor_vertices_to_unidepth"
ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE = "scene_depth_supports_accepted_foreground_occluder_owner"
RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE = "scene_depth_supports_foreground_occluder_candidate_owner_unaccepted"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_accepted_row(case: str, row: dict[str, Any]) -> str:
    require(row.get("accepted_occlusion_owner") is True, f"{case}: pose fill accepted without accepted owner")
    require(row.get("owner_depth_order_supported") is True, f"{case}: pose fill accepted without owner depth-order support")
    acceptance_type = str(row.get("pose_fill_acceptance_type") or "")
    if acceptance_type == "observed_depth_scaled_mano_behind_accepted_occluder":
        require(row.get("observed_mano_pose_through_occlusion_accepted") is True, f"{case}: observed pose-fill acceptance flag missing")
        require(row.get("temporal_pose_fill_accepted") is not True, f"{case}: observed pose fill also marked temporal")
        require(row.get("final_hawor_support_state") == "observed_same_frame_detection", f"{case}: observed pose fill lacks same-frame HaWoR support")
        require(row.get("final_hawor_same_frame_detection") is True, f"{case}: observed pose fill lacks same-frame detector flag")
        require(row.get("final_hawor_observed_depth_scaled_mano_supported") is True, f"{case}: observed pose fill lacks depth-scaled MANO support")
        require(row.get("hawor_to_v18_depth_scale_status") == DEPTH_SCALE_SUPPORT_STATUS, f"{case}: observed pose fill has wrong depth-scale status")
        min_samples = int(row.get("min_hawor_to_v18_depth_scale_sample_count") or 0)
        sample_count = int(row.get("hawor_to_v18_depth_scale_sample_count") or 0)
        require(min_samples > 0 and sample_count >= min_samples, f"{case}: observed pose fill has too few depth-scale samples")
        require(not row.get("observed_pose_acceptance_blockers"), f"{case}: observed pose fill has fatal blockers")
        owner_support = row.get("source_occlusion_owner_depth_support") if isinstance(row.get("source_occlusion_owner_depth_support"), dict) else {}
        require(owner_support.get("source_depth_order_resolved") is True, f"{case}: observed pose fill source owner depth order unresolved")
        require(owner_support.get("source_occluder_owner_accepted") is True, f"{case}: observed pose fill source owner not accepted")
        require(owner_support.get("graph_occlusion_owner_accepted") is True, f"{case}: observed pose fill source graph owner not accepted")
        require(owner_support.get("hawor_depth_order_accepted") is True, f"{case}: observed pose fill lacks HaWoR depth-order support")
        require(owner_support.get("depth_pair_evidence_state") == ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE, f"{case}: accepted pose fill carries non-accepted depth support state")
        raw_depth_state = owner_support.get("raw_depth_pair_evidence_state_before_graph_acceptance")
        require(raw_depth_state is None or raw_depth_state == RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE, f"{case}: accepted pose fill raw depth provenance has unexpected state")
        bridge = row.get("source_hawor_bridge_row") if isinstance(row.get("source_hawor_bridge_row"), dict) else {}
        require(bridge.get("observed_depth_scaled_mano_supported") is True, f"{case}: observed pose fill source bridge lacks support flag")
        return acceptance_type
    if acceptance_type == "temporal_occlusion_pose_baseline":
        require(row.get("temporal_pose_fill_accepted") is True, f"{case}: temporal pose fill acceptance flag missing")
        require(row.get("hand_baseline_temporal_occlusion_pose_accepted") is True, f"{case}: temporal pose fill accepted without accepted hand baseline")
        require(not row.get("blockers"), f"{case}: temporal pose fill accepted despite blockers")
        return acceptance_type
    raise RuntimeError(f"{case}: unsupported pose-fill acceptance type {acceptance_type!r}")


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = str(report.get("case"))
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: rows missing")
    require(report.get("pose_fill_through_occlusion_complete") is False, f"{case}: overclaims complete pose fill")
    accepted = 0
    observed_accepted = 0
    temporal_accepted = 0
    candidate = 0
    owner_blocker_rows = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        if row.get("hawor_candidate_present") is True or row.get("hawor_measurement_available") is True:
            candidate += 1
        owner_candidate_rows = row.get("source_occlusion_owner_candidate_rows")
        owner_blockers = row.get("occlusion_owner_acceptance_blockers")
        if isinstance(owner_candidate_rows, list) and len(owner_candidate_rows) > 0 and row.get("accepted_occlusion_owner") is not True:
            require(isinstance(owner_blockers, list) and len(owner_blockers) > 0, f"{case}: owner candidate rows lack propagated owner blockers")
            owner_blocker_rows += 1
        if row.get("pose_fill_through_occlusion_accepted") is True:
            accepted += 1
            acceptance_type = validate_accepted_row(case, row)
            if acceptance_type == "observed_depth_scaled_mano_behind_accepted_occluder":
                observed_accepted += 1
            elif acceptance_type == "temporal_occlusion_pose_baseline":
                temporal_accepted += 1
        else:
            require(row.get("pose_fill_gate_claim") == "pose_fill_blocked_not_accepted", f"{case}: unsupported pose fill claim")
            require(isinstance(row.get("blockers"), list) and len(row.get("blockers")) > 0, f"{case}: blocked pose fill lacks blockers")
    require(accepted == int(report.get("pose_fill_through_occlusion_accepted_rows", -1)), f"{case}: accepted count mismatch")
    require(observed_accepted == int(report.get("observed_mano_pose_through_occlusion_accepted_rows", -1)), f"{case}: observed accepted count mismatch")
    require(temporal_accepted == int(report.get("temporal_pose_fill_through_occlusion_accepted_rows", -1)), f"{case}: temporal accepted count mismatch")
    require(candidate == int(report.get("pose_fill_candidate_rows", -1)), f"{case}: candidate count mismatch")
    return {
        "case": case,
        "row_count": len(rows),
        "pose_fill_candidate_rows": candidate,
        "pose_fill_through_occlusion_accepted_rows": accepted,
        "observed_mano_pose_through_occlusion_accepted_rows": observed_accepted,
        "temporal_pose_fill_through_occlusion_accepted_rows": temporal_accepted,
        "owner_blocker_rows": owner_blocker_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_pose_fill_gate_complete_depth_hawor"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_occlusion_pose_fill_gate_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
