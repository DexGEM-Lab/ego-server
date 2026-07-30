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
    path = args.root / "hawor_task5_export_contract" / "v18_hawor_task5_export_contract.json"
    require(path.exists(), f"missing contract {path}", failures)
    contract = load_json(path) if path.exists() else {}
    require(contract.get("method") == "build_v18_hawor_task5_export_contract", f"method unexpected {contract.get('method')}", failures)
    require(contract.get("claim_scope") == "task5_HaWoR_export_contract_only_no_execution_no_model_substitution_no_physical_acceptance", f"claim scope unexpected {contract.get('claim_scope')}", failures)
    require(contract.get("case") == "task5_tomato_960", f"case unexpected {contract.get('case')}", failures)
    require(contract.get("expected_frame_count") == 960, f"expected_frame_count unexpected {contract.get('expected_frame_count')}", failures)
    require(contract.get("expected_frame_side_rows") == 1920, f"expected_frame_side_rows unexpected {contract.get('expected_frame_side_rows')}", failures)
    local_clip = contract.get("local_raw_clip") if isinstance(contract.get("local_raw_clip"), dict) else {}
    expected_sha256 = "66791eaa646aac2e8cb24bb00fe30b2801436302327b1c46fea650446c41c4ac"
    require(local_clip.get("exists") is True, f"task5 local clip should exist for contract: {local_clip}", failures)
    require(str(local_clip.get("path", "")).endswith("20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4"), f"contract local clip is not task5: {local_clip}", failures)
    require(local_clip.get("sha256") == expected_sha256, f"task5 local clip sha256 mismatch: {local_clip}", failures)
    identity = contract.get("task5_clip_identity") if isinstance(contract.get("task5_clip_identity"), dict) else {}
    require(identity.get("expected_local_clip_sha256") == expected_sha256, f"identity expected sha unexpected: {identity}", failures)
    require(identity.get("local_clip_sha256") == expected_sha256, f"identity local sha unexpected: {identity}", failures)
    require(identity.get("local_clip_matches_expected_sha256") is True, f"identity should verify local clip: {identity}", failures)
    require(identity.get("remote_clip_sha256_env") == expected_sha256, f"remote sha env unexpected: {identity}", failures)
    metadata = identity.get("expected_video_metadata") if isinstance(identity.get("expected_video_metadata"), dict) else {}
    require(metadata.get("frame_count") == 960 and metadata.get("width") == 1920 and metadata.get("height") == 1080 and metadata.get("fps") == 30.0, f"expected video metadata unexpected: {metadata}", failures)
    expected_npz = contract.get("expected_local_output_npz") if isinstance(contract.get("expected_local_output_npz"), dict) else {}
    require(expected_npz.get("path") == "/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands.npz", f"expected task5 NPZ path unexpected {expected_npz}", failures)
    layout = contract.get("post_copy_expected_local_layout") if isinstance(contract.get("post_copy_expected_local_layout"), dict) else {}
    require(layout.get("expected_qc_video_sha256") == expected_sha256, f"post-copy expected qc sha unexpected: {layout}", failures)
    require(layout.get("expected_npz_video_sha256") == expected_sha256, f"post-copy expected npz sha unexpected: {layout}", failures)
    qc_provenance_fields = layout.get("expected_qc_export_provenance_hash_fields") if isinstance(layout.get("expected_qc_export_provenance_hash_fields"), list) else []
    for field in ["checkpoint", "infiller_weight", "model_config"]:
        require(field in qc_provenance_fields, f"post-copy expected qc provenance hash field missing {field}: {layout}", failures)
    npz_provenance_fields = layout.get("expected_npz_export_provenance_hash_fields") if isinstance(layout.get("expected_npz_export_provenance_hash_fields"), list) else []
    for field in ["checkpoint_sha256", "infiller_weight_sha256", "model_config_sha256"]:
        require(field in npz_provenance_fields, f"post-copy expected npz provenance hash field missing {field}: {layout}", failures)
    flags = contract.get("acceptance_flags") if isinstance(contract.get("acceptance_flags"), dict) else {}
    require(flags.get("accepted_v18_hawor_requirement_met") is False, f"contract must not accept HaWoR requirement: {flags}", failures)
    require(flags.get("accepted_metric_hand_state_from_hawor") is False, f"contract must not accept metric hand state: {flags}", failures)
    require(flags.get("accepted_contact_occlusion_nonpenetration") is False, f"contract must not accept downstream physics: {flags}", failures)
    require(flags.get("task5_clip_identity_verified_locally") is True, f"contract should verify local task5 clip identity: {flags}", failures)
    remote_cmd = str(contract.get("remote_export_command"))
    remote_root = "/mnt/user-home/yiwen/ego_annotation_remote/hawor_work"
    expected_remote_output = f"{remote_root}/outputs/task5_tomato_960_hawor_world"
    require(f"EGO_HAWOR_ROOT={remote_root}" in remote_cmd, f"remote command must set absolute HaWoR root: {remote_cmd}", failures)
    require("EGO_HAWOR_CASE=task5_tomato_960" in remote_cmd, f"remote command must select task5: {remote_cmd}", failures)
    require("20260118_1257_Rec3db6_P0_Sc6ab88_task_5" in remote_cmd, f"remote command must reference task5 clip: {remote_cmd}", failures)
    require(f"EGO_HAWOR_CLIP_SHA256={expected_sha256}" in remote_cmd, f"remote command must enforce task5 clip sha256: {remote_cmd}", failures)
    require(f"EGO_HAWOR_OUTPUT_DIR={expected_remote_output}" in remote_cmd, f"remote command must set absolute task5 output dir: {remote_cmd}", failures)
    require("EGO_HAWOR_OUTPUT_DIR=$EGO_HAWOR_ROOT" not in remote_cmd, f"remote command must not depend on caller shell EGO_HAWOR_ROOT expansion: {remote_cmd}", failures)
    require("20260108_1057_Recf94e_P0_S994da4_task_9" not in remote_cmd, f"remote command should not default to trash clip: {remote_cmd}", failures)
    blockers = contract.get("blocking_reasons") if isinstance(contract.get("blocking_reasons"), list) else []
    for blocker in [
        "external_HaWoR_assets_or_output_required_for_task5",
        "do_not_substitute_WiLoR_HaMeR_MANO2D_or_depth_probe",
        "post_ingest_bridge_and_downstream_validation_required_before_any_physical_claim",
        "remote_task5_clip_sha256_must_match_contract_before_export",
    ]:
        require(blocker in blockers, f"missing blocker {blocker}", failures)
    if expected_npz.get("exists") is False:
        require(contract.get("status") == "blocked_task5_hawor_export_contract_written_waiting_for_external_assets_or_output", f"absent NPZ should keep blocked status, got {contract.get('status')}", failures)
        require(flags.get("task5_hawor_output_present") is False, f"absent NPZ should set output_present false: {flags}", failures)
        require("task5_expected_hawor_npz_absent_at_contract_path" in blockers, "missing absent-NPZ blocker", failures)
    elif expected_npz.get("exists") is True:
        require(contract.get("status") == "task5_hawor_output_present_needs_requirement_rebuild", f"present NPZ status unexpected {contract.get('status')}", failures)
        require(flags.get("task5_hawor_output_present") is True, f"present NPZ should set output_present true: {flags}", failures)
    else:
        require(False, f"expected NPZ exists field not boolean: {expected_npz}", failures)
    commands = contract.get("post_ingest_validation_commands") if isinstance(contract.get("post_ingest_validation_commands"), list) else []
    for needle in [
        "build_v18_hawor_requirement_state.py",
        "validate_v18_hawor_requirement_state.py",
        "build_v18_hawor_bridge_state.py",
        "validate_v18_hawor_bridge_state.py",
        "build_v18_hawor_bridge_quality_state.py",
        "validate_v18_hawor_bridge_quality_state.py",
        "build_v18_hawor_bridge_subset_policy.py",
        "validate_v18_hawor_bridge_subset_policy.py",
    ]:
        require(any(needle in str(cmd) for cmd in commands), f"missing post-ingest command {needle}", failures)
    return {
        "method": "validate_v18_hawor_task5_export_contract",
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
