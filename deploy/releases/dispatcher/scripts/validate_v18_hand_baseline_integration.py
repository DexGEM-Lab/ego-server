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
    ann = load_json(path)
    case = str(ann.get("case"))
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: missing frames")
    modules_raw = ann.get("modules")
    modules: dict[str, Any] = modules_raw if isinstance(modules_raw, dict) else {}
    require("hand_baseline_evidence" in str(modules.get("hand_branch")), f"{case}: hand module does not report baseline integration")
    hand_rows = 0
    baseline_rows = 0
    hawor_rows = 0
    wilor_rows = 0
    blocker_rows = 0
    accepted_occlusion_pose = 0
    missing_rows = 0
    metric_depth_component_rows = 0
    temporal_acceleration_component_rows = 0
    bone_scale_component_rows = 0
    pose_fill_accepted_rows = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            hand_rows += 1
            baseline_raw = hand.get("hand_baseline_branch")
            baseline: dict[str, Any] = baseline_raw if isinstance(baseline_raw, dict) else {}
            if baseline.get("hand_baseline_state"):
                baseline_rows += 1
            if baseline.get("state") == "missing_hand_baseline_branch_row":
                missing_rows += 1
            if baseline.get("hawor_candidate_present") is True:
                hawor_rows += 1
            if baseline.get("wilor_measurement_available") is True:
                wilor_rows += 1
            blockers = baseline.get("acceptance_blockers")
            blocker_set = {str(v) for v in blockers} if isinstance(blockers, list) else set()
            if isinstance(blockers, list) and len(blockers) > 0:
                blocker_rows += 1
            components_raw = baseline.get("baseline_score_components")
            components: dict[str, Any] = components_raw if isinstance(components_raw, dict) else {}
            missing_components_raw = components.get("score_contract_missing_components")
            missing_component_set = {str(v) for v in missing_components_raw} if isinstance(missing_components_raw, list) else set()
            metric_depth = components.get("median_metric_depth_abs_residual_m")
            temporal_acceleration = components.get("temporal_acceleration_m_per_frame2")
            bone_scale = components.get("hand_bone_scale_median_abs_error_m")
            if metric_depth is None:
                require("median_metric_depth_abs_m/0.05" in missing_component_set, f"{case}: missing metric-depth component not recorded in score contract")
                require("median_metric_depth_abs_residual_component_missing" in blocker_set, f"{case}: missing metric-depth component lacks blocker")
            else:
                metric_depth_component_rows += 1
                require("median_metric_depth_abs_m/0.05" not in missing_component_set, f"{case}: present metric-depth component still marked missing")
            if temporal_acceleration is None:
                require("temporal_acceleration_m_per_frame2/0.05" in missing_component_set, f"{case}: missing temporal acceleration not recorded in score contract")
                require("temporal_acceleration_component_missing" in blocker_set, f"{case}: missing temporal acceleration lacks blocker")
            else:
                temporal_acceleration_component_rows += 1
                require("temporal_acceleration_m_per_frame2/0.05" not in missing_component_set, f"{case}: present temporal acceleration still marked missing")
            if bone_scale is None:
                require("hand_bone_scale_error_m/0.025" in missing_component_set, f"{case}: missing bone-scale component not recorded in score contract")
                require("hand_bone_scale_component_missing" in blocker_set, f"{case}: missing bone-scale component lacks blocker")
            else:
                bone_scale_component_rows += 1
                require("hand_bone_scale_error_m/0.025" not in missing_component_set, f"{case}: present bone-scale component still marked missing")
            if baseline.get("hand_baseline_state") == "hawor_visible_measurement_score_components_supported_no_occluded_pose_acceptance":
                require(len(blocker_set) == 0, f"{case}: supported hand-baseline state has active blockers")
            pose_fill_raw = hand.get("occlusion_pose_fill_gate")
            pose_fill: dict[str, Any] = pose_fill_raw if isinstance(pose_fill_raw, dict) else {}
            if pose_fill.get("pose_fill_through_occlusion_accepted") is True:
                pose_fill_accepted_rows += 1
                acceptance_type = str(pose_fill.get("pose_fill_acceptance_type") or "")
                require(acceptance_type == "observed_depth_scaled_mano_behind_accepted_occluder", f"{case}: unsupported pose-fill acceptance type {acceptance_type!r}")
                require(pose_fill.get("accepted_occlusion_owner") is True and pose_fill.get("owner_depth_order_supported") is True, f"{case}: observed pose fill lacks accepted owner depth support")
                require(pose_fill.get("final_hawor_support_state") == "observed_same_frame_detection", f"{case}: observed pose fill lacks same-frame final HaWoR support")
                require(pose_fill.get("final_hawor_observed_depth_scaled_mano_supported") is True, f"{case}: observed pose fill lacks depth-scaled MANO support")
            if baseline.get("temporal_occlusion_pose_accepted") is True:
                accepted_occlusion_pose += 1
            require(baseline.get("pose_claim") in {"no_occluded_pose_accepted_from_current_hand_baseline", "no_occluded_pose_supported_from_current_hand_baseline", None}, f"{case}: unsupported hand pose claim")
    require(hand_rows > 0, f"{case}: no hands")
    require(baseline_rows == hand_rows, f"{case}: not every hand has baseline row")
    require(missing_rows == 0, f"{case}: missing baseline integration rows")
    require(blocker_rows > 0, f"{case}: no blockers preserved")
    require(accepted_occlusion_pose == 0, f"{case}: temporal hand-baseline occlusion pose unexpectedly accepted")
    require(metric_depth_component_rows > 0, f"{case}: no metric-depth evidence component rows")
    if hawor_rows > 0:
        require(temporal_acceleration_component_rows > 0, f"{case}: HaWoR rows exist but no temporal-acceleration component rows")
        require(bone_scale_component_rows > 0, f"{case}: HaWoR rows exist but no bone-scale component rows")
    return {
        "case": case,
        "hand_rows": hand_rows,
        "baseline_rows": baseline_rows,
        "hawor_rows": hawor_rows,
        "wilor_rows": wilor_rows,
        "blocker_rows": blocker_rows,
        "accepted_occlusion_pose_rows": accepted_occlusion_pose,
        "pose_fill_accepted_rows": pose_fill_accepted_rows,
        "metric_depth_component_rows": metric_depth_component_rows,
        "temporal_acceleration_component_rows": temporal_acceleration_component_rows,
        "bone_scale_component_rows": bone_scale_component_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "annotations_v18_full.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
