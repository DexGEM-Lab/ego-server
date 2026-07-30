#!/usr/bin/env python3
"""Validate the V18 MANO foundation audit/state.

This validator is intentionally strict about scope: recovered MANO candidates are
evidence, but the current V18 physical pipeline remains invalid unless a full
metric MANO foundation is present and explicitly marked valid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "trash_1050": {"expected_rows": 2100, "wilor_rows": 1617, "wilor_unique_frame_side_rows": 1601, "hawor_rows": 182},
    "task5_tomato_960": {"expected_rows": 1920, "wilor_rows": 1744, "wilor_unique_frame_side_rows": 1733, "hawor_rows": 0},
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def finite(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def validate_npz(case: str, report: dict[str, Any], failures: list[str]) -> None:
    wilor = report.get("recovered_wilor_virtual_camera_mano_candidates", {}) if isinstance(report.get("recovered_wilor_virtual_camera_mano_candidates"), dict) else {}
    rows = int(wilor.get("complete_virtual_camera_candidate_rows", 0))
    npz_path_raw = wilor.get("npz_path")
    require(isinstance(npz_path_raw, str) and bool(npz_path_raw), f"{case}: missing WiLoR MANO NPZ path", failures)
    if not isinstance(npz_path_raw, str):
        return
    npz_path = Path(npz_path_raw)
    require(npz_path.exists(), f"{case}: WiLoR MANO NPZ does not exist: {npz_path}", failures)
    if not npz_path.exists():
        return
    arrays = np.load(npz_path)
    expected_shapes = {
        "frame_idx": (rows,),
        "hand_side_code": (rows,),
        "detector_score": (rows,),
        "bbox_xyxy": (rows, 4),
        "wilor_virtual_camera_intrinsics": (rows, 4),
        "T_world_camera_metric": (rows, 4, 4),
        "cam_t": (rows, 3),
        "joints_v18_pose_transformed_from_wilor_virtual_camera": (rows, 21, 3),
        "vertices_v18_pose_transformed_from_wilor_virtual_camera": (rows, 778, 3),
        "mano_global_orient": (rows, 9),
        "mano_hand_pose": (rows, 135),
        "mano_betas": (rows, 10),
    }
    for key, shape in expected_shapes.items():
        require(key in arrays.files, f"{case}: NPZ missing array {key}", failures)
        if key in arrays.files:
            require(tuple(arrays[key].shape) == shape, f"{case}: NPZ array {key} shape {arrays[key].shape} != {shape}", failures)
            require(bool(np.all(np.isfinite(arrays[key]))), f"{case}: NPZ array {key} contains non-finite values", failures)


def validate_case(root: Path, case: str, failures: list[str]) -> dict[str, Any] | None:
    report_path = root / case / "v18_mano_foundation_state_report.json"
    require(report_path.exists(), f"{case}: report missing", failures)
    if not report_path.exists():
        return None
    report = load_json(report_path)
    exp = EXPECTED[case]
    require(report.get("claim_scope") == "MANO_first_foundation_state_and_validity_gate_not_contact_occlusion_or_object_pose_closure", f"{case}: wrong claim scope", failures)
    require(report.get("foundational_mano_state_valid") is True, f"{case}: support-qualified foundational MANO should be valid", failures)
    require(report.get("support_qualified_mano_foundation_valid") is True, f"{case}: support-qualified MANO field should be true", failures)
    require(report.get("observed_same_frame_physical_support_complete") is False, f"{case}: observed same-frame support should remain incomplete", failures)
    require(report.get("v18_physical_pipeline_valid_without_further_hand_work") is False, f"{case}: physical pipeline should remain invalid without downstream object/nonpenetration closure", failures)
    require(report.get("expected_two_hand_rows") == exp["expected_rows"], f"{case}: expected row count changed", failures)
    current = report.get("current_v18_full_mano_storage", {}).get("counts", {}) if isinstance(report.get("current_v18_full_mano_storage"), dict) else {}
    surface_contract_rows = max(int(current.get("surface_candidates_stored_in_v18_full", 0)), int(current.get("surface_reference_rows_stored_in_v18_full", 0)))
    param_contract_rows = max(int(current.get("mano_params_stored_in_v18_full", 0)), int(current.get("reproducible_mano_param_rows_stored_in_v18_full", 0)))
    require(surface_contract_rows == exp["expected_rows"], f"{case}: V18 full MANO surface contract rows {surface_contract_rows} != expected {exp['expected_rows']}", failures)
    require(param_contract_rows == exp["expected_rows"], f"{case}: V18 full MANO parameter contract rows {param_contract_rows} != expected {exp['expected_rows']}", failures)
    wilor = report.get("recovered_wilor_virtual_camera_mano_candidates", {}) if isinstance(report.get("recovered_wilor_virtual_camera_mano_candidates"), dict) else {}
    hawor = report.get("hawor_world_mano_candidates", {}) if isinstance(report.get("hawor_world_mano_candidates"), dict) else {}
    require(int(wilor.get("complete_virtual_camera_candidate_rows", -1)) == exp["wilor_rows"], f"{case}: recovered WiLoR raw virtual-camera candidate rows mismatch", failures)
    require(int(wilor.get("unique_virtual_camera_frame_side_rows", -1)) == exp["wilor_unique_frame_side_rows"], f"{case}: recovered WiLoR unique frame-side coverage mismatch", failures)
    require(int(hawor.get("current_v18_hawor_surface_param_contract_rows", -1)) == exp["expected_rows"], f"{case}: current V18 HaWoR surface/param contract rows mismatch", failures)
    require(finite(wilor.get("wilor_internal_projection_residual_px_median")) and float(wilor["wilor_internal_projection_residual_px_median"]) < 0.01, f"{case}: WiLoR internal projection residual does not validate raw virtual-camera consistency", failures)
    require(wilor.get("metric_world_alignment_valid") is False, f"{case}: WiLoR virtual-camera candidates should not be marked metric-world aligned", failures)
    require(wilor.get("coordinate_status") == "wilor_virtual_camera_surface_transformed_by_v18_camera_pose_not_metric_depth_aligned", f"{case}: missing virtual-camera coordinate status", failures)
    if wilor.get("source_sha256") is not None:
        require(isinstance(wilor.get("source_sha256"), str) and len(wilor.get("source_sha256")) == 64, f"{case}: malformed hashed WiLoR source provenance", failures)
    require(finite(wilor.get("wilor_virtual_camera_cam_t_z_median")) and float(wilor["wilor_virtual_camera_cam_t_z_median"]) > 5.0, f"{case}: virtual camera depth sanity check did not expose non-metric scale", failures)
    blockers = report.get("blocking_reasons") if isinstance(report.get("blocking_reasons"), list) else []
    require("current_v18_full_annotations_drop_mano_vertices" not in blockers, f"{case}: stale dropped MANO vertices blocker remains after surface contract", failures)
    require("current_v18_full_annotations_drop_mano_parameters" not in blockers, f"{case}: stale dropped MANO params blocker remains after parameter contract", failures)
    require("hawor_valid_rows_include_inferred_without_same_frame_detection_support" not in blockers, f"{case}: inferred rows should be limitations, not storage blockers", failures)
    limitations = report.get("support_limitations") if isinstance(report.get("support_limitations"), list) else []
    require("hawor_valid_rows_include_inferred_without_same_frame_detection_support" in limitations, f"{case}: missing limitation for inferred/no-same-frame HaWoR support rows", failures)
    if case == "trash_1050":
        require("hawor_timeline_contains_explicit_temporal_boundary_fill_rows" not in limitations, f"{case}: trash boundary-fill limitation should be gone after padded HaWoR tail repair", failures)
    validate_npz(case, report, failures)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/mano_foundation_audit"))
    args = parser.parse_args()
    failures: list[str] = []
    summary_path = args.root / "v18_mano_foundation_audit_summary.json"
    require(summary_path.exists(), f"summary missing: {summary_path}", failures)
    summary = load_json(summary_path) if summary_path.exists() else {}
    require(summary.get("all_cases_foundational_mano_valid") is True, "summary should mark support-qualified foundation valid", failures)
    require(summary.get("v18_physical_pipeline_valid_without_further_hand_work") is False, "summary should mark physical pipeline invalid", failures)
    reports = []
    for case in EXPECTED:
        report = validate_case(args.root, case, failures)
        if report is not None:
            reports.append({"case": case, "foundational_mano_state_valid": report.get("foundational_mano_state_valid"), "blocking_reasons": report.get("blocking_reasons")})
    out = {"method": "validate_v18_mano_foundation_state", "status": "ok" if not failures else "failed", "root": str(args.root), "cases": reports, "failures": failures}
    print(json.dumps(out, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
