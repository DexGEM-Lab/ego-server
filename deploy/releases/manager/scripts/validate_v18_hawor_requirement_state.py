#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "trash_1050": {
        "status": "hawor_support_qualified_metric_mano_available_not_full_physical_closure",
        "expected_frame_side_rows": 2100,
        "available_hawor_frame_side_rows": 2100,
        "same_frame_detection_frame_side_rows": 1562,
        "valid_without_same_frame_detection_frame_side_rows": 538,
        "temporal_boundary_filled_frame_side_rows": 0,
        "full_timeline_hawor_npz_shape_valid": True,
    },
    "task5_tomato_960": {
        "status": "hawor_support_qualified_metric_mano_available_not_full_physical_closure",
        "expected_frame_side_rows": 1920,
        "available_hawor_frame_side_rows": 1920,
        "same_frame_detection_frame_side_rows": 1715,
        "valid_without_same_frame_detection_frame_side_rows": 205,
        "temporal_boundary_filled_frame_side_rows": 0,
        "full_timeline_hawor_npz_shape_valid": True,
    },
}

FORBIDDEN_SUBSTITUTES = ["WiLoR", "HaMeR", "MANO2D", "depth_probe", "metric_alignment_probe"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_case(case: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    name = str(case.get("case"))
    exp = EXPECTED.get(name)
    require(exp is not None, f"unexpected case {name}", failures)
    if exp is None:
        return {"case": name, "status": "unexpected"}
    for key, value in exp.items():
        require(case.get(key) == value, f"{name}: expected {key}={value}, got {case.get(key)}", failures)
    require(case.get("hard_requirement") == "HaWoR_full_timeline_metric_MANO_required_for_V18_physical_hand_state", f"{name}: missing hard requirement string", failures)
    require(case.get("accepted_v18_hawor_requirement_met") is True, f"{name}: support-qualified HaWoR requirement should be met", failures)
    require(case.get("accepted_metric_hand_state_from_hawor") is True, f"{name}: support-qualified metric hand state should be accepted", failures)
    require(case.get("support_qualified_full_timeline_metric_mano_available") is True, f"{name}: full-timeline support-qualified MANO should be available", failures)
    require(case.get("observed_same_frame_physical_support_complete") is False, f"{name}: observed same-frame support should remain incomplete", failures)
    require(case.get("physical_claim_policy") == "observed contact occlusion and nonpenetration claims require observed_same_frame_detection hand support; inferred and boundary-filled rows are renderable continuity only", f"{name}: physical claim policy missing or changed", failures)
    require(case.get("claim_scope") == "HaWoR_requirement_state_only_no_WiLoR_or_other_backend_substitution", f"{name}: incorrect claim scope", failures)
    text = json.dumps(case)
    # The no-substitution artifact may mention WiLoR only inside the literal no-substitution claim scope; nowhere else.
    sanitized = text.replace("HaWoR_requirement_state_only_no_WiLoR_or_other_backend_substitution", "")
    for forbidden in FORBIDDEN_SUBSTITUTES:
        require(forbidden not in sanitized, f"{name}: substitute backend/probe string appears outside no-substitution scope: {forbidden}", failures)
    blockers = case.get("blocking_reasons") if isinstance(case.get("blocking_reasons"), list) else []
    if name == "trash_1050":
        bridge = case.get("current_v18_bridge_candidate") if isinstance(case.get("current_v18_bridge_candidate"), dict) else {}
        if bridge.get("exists") is True:
            require(bridge.get("status") == "trash_hawor_bridge_candidate_built_not_accepted", f"{name}: unexpected bridge status {bridge.get('status')}", failures)
            require(bridge.get("bridge_candidate_rows") == 2100, f"{name}: unexpected bridge candidate rows {bridge.get('bridge_candidate_rows')}", failures)
            require(bridge.get("accepted_v18_hawor_foundation") is False, f"{name}: bridge report itself remains candidate-only", failures)
        else:
            require(False, f"{name}: bridge report should exist after support-qualified HaWoR bridge build", failures)
        limitations = case.get("support_limitations") if isinstance(case.get("support_limitations"), list) else []
        for limitation in [
            "hawor_valid_rows_include_inferred_without_same_frame_detection_support",
            "single_global_HaWoR_to_V18_world_sim3_alignment_too_loose_for_global_world_physical_claims",
            "contact_occlusion_nonpenetration_require_support_gated_recompute_before_observed_physical_claims",
        ]:
            require(limitation in limitations, f"{name}: missing support limitation {limitation}", failures)
        npz_info = case.get("hawor_output") if isinstance(case.get("hawor_output"), dict) else {}
        require(npz_info.get("exists") is True, f"{name}: expected existing HaWoR NPZ", failures)
        npz_path = Path(str(npz_info.get("path")))
        if npz_path.exists():
            z = np.load(npz_path)
            require(z["left_vertices_world_m"].shape == (1050, 778, 3), f"{name}: left vertices shape mismatch", failures)
            require(z["right_vertices_world_m"].shape == (1050, 778, 3), f"{name}: right vertices shape mismatch", failures)
            require(int(np.count_nonzero(z["left_valid"])) == 1050, f"{name}: left valid count mismatch", failures)
            require(int(np.count_nonzero(z["right_valid"])) == 1050, f"{name}: right valid count mismatch", failures)
            if "left_temporal_boundary_filled" in z.files:
                require(int(np.count_nonzero(z["left_temporal_boundary_filled"])) == 0, f"{name}: left boundary fill count mismatch", failures)
            if "right_temporal_boundary_filled" in z.files:
                require(int(np.count_nonzero(z["right_temporal_boundary_filled"])) == 0, f"{name}: right boundary fill count mismatch", failures)
    if name == "task5_tomato_960":
        npz_info = case.get("hawor_output") if isinstance(case.get("hawor_output"), dict) else {}
        expected_contract_path = "/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands_with_track_support.npz"
        require(npz_info.get("path") == expected_contract_path, f"{name}: expected contract HaWoR path {expected_contract_path}, got {npz_info.get('path')}", failures)
        if npz_info.get("exists") is False:
            require(case.get("status") == "blocked_no_case_hawor_output", f"{name}: absent NPZ should keep blocked status, got {case.get('status')}", failures)
            require(case.get("available_hawor_frame_side_rows") == 0, f"{name}: absent NPZ should have 0 rows, got {case.get('available_hawor_frame_side_rows')}", failures)
            require(case.get("full_timeline_hawor_npz_shape_valid") is False, f"{name}: absent NPZ shape valid should be false", failures)
            for blocker in ["case_hawor_world_hands_npz_missing", "HaWoR_repo_weights_or_MANO_assets_missing_locally"]:
                require(blocker in blockers, f"{name}: missing blocker {blocker}", failures)
        elif npz_info.get("exists") is True:
            require(case.get("status") == "hawor_support_qualified_metric_mano_available_not_full_physical_closure", f"{name}: present NPZ status unexpected {case.get('status')}", failures)
            require(case.get("accepted_v18_hawor_requirement_met") is True, f"{name}: present support-qualified NPZ should satisfy HaWoR MANO requirement", failures)
            require(case.get("accepted_metric_hand_state_from_hawor") is True, f"{name}: present support-qualified NPZ should satisfy metric hand state availability", failures)
            limitations = case.get("support_limitations") if isinstance(case.get("support_limitations"), list) else []
            require("hawor_valid_rows_include_inferred_without_same_frame_detection_support" in limitations, f"{name}: missing inferred support limitation", failures)
            require("contact_occlusion_nonpenetration_require_support_gated_recompute_before_observed_physical_claims" in limitations, f"{name}: missing downstream physical limitation", failures)
            npz_path = Path(str(npz_info.get("path")))
            if npz_path.exists():
                z = np.load(npz_path)
                require(z["left_vertices_world_m"].shape == (960, 778, 3), f"{name}: left vertices shape mismatch", failures)
                require(z["right_vertices_world_m"].shape == (960, 778, 3), f"{name}: right vertices shape mismatch", failures)
                require(z["left_joints_world_m"].shape == (960, 21, 3), f"{name}: left joints shape mismatch", failures)
                require(z["right_joints_world_m"].shape == (960, 21, 3), f"{name}: right joints shape mismatch", failures)
                require(z["frame_idx"].shape == (960,), f"{name}: frame_idx shape mismatch", failures)
                require(int(z["frame_idx"][0]) == 0 and int(z["frame_idx"][-1]) == 959, f"{name}: frame_idx range mismatch", failures)
        else:
            require(False, f"{name}: hawor_output.exists must be boolean false/true", failures)
    return {"case": name, "status": case.get("status"), "available_hawor_frame_side_rows": case.get("available_hawor_frame_side_rows")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = args.root / "hawor_requirement_state" / "v18_hawor_requirement_state.json"
    require(path.exists(), f"missing {path}", failures)
    payload = load_json(path) if path.exists() else {}
    require(payload.get("status") == "hawor_hard_requirement_met_not_downstream_validated", f"summary status unexpected: {payload.get('status')}", failures)
    require(payload.get("claim_scope") == "HaWoR_hard_requirement_state_no_model_substitution_no_full_V18_closure", "summary claim scope incorrect", failures)
    require(payload.get("all_cases_hawor_requirement_met") is True, "all_cases_hawor_requirement_met should be true for support-qualified HaWoR MANO availability", failures)
    require(payload.get("v18_physical_hand_state_valid_from_hawor") is False, "v18 physical hand state must be false", failures)
    require(payload.get("provisioning_status") == "blocked_missing_required_hawor_assets", "provisioning status must be blocked", failures)
    missing = payload.get("missing_required") if isinstance(payload.get("missing_required"), list) else []
    for req in ["configured_hawor_repo", "configured_hawor_checkpoint", "configured_infiller_weight", "configured_model_config", "configured_mano_left"]:
        require(req in missing, f"summary missing required {req}", failures)
    cases = payload.get("cases") if isinstance(payload.get("cases"), list) else []
    require(len(cases) == len(EXPECTED), f"expected {len(EXPECTED)} cases, got {len(cases)}", failures)
    case_reports = [validate_case(case, failures) for case in cases if isinstance(case, dict)]
    return {
        "method": "validate_v18_hawor_requirement_state",
        "status": "ok" if not failures else "failed",
        "root": str(args.root),
        "cases": case_reports,
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
