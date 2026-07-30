#!/usr/bin/env python3
"""Write the V18 task5 HaWoR export contract.

This does not run HaWoR. It records the exact task5 video, required assets,
expected output path, remote command, and post-ingest validators needed to turn
the current external blocker into a reproducible HaWoR export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

TASK5_CASE = "task5_tomato_960"
TASK5_LOCAL_CLIP = Path("/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4")
TASK5_REMOTE_ROOT = "/mnt/user-home/yiwen/ego_annotation_remote/hawor_work"
TASK5_REMOTE_CLIP = "/mnt/user-home/yiwen/ego_annotation_remote/data/clip/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4"
EXPECTED_LOCAL_OUTPUT_DIR = Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960")
EXPECTED_REMOTE_OUTPUT_DIR = f"{TASK5_REMOTE_ROOT}/outputs/task5_tomato_960_hawor_world"
EXPECTED_FRAME_COUNT = 960
EXPECTED_FRAME_SIDE_ROWS = 1920
EXPECTED_LOCAL_CLIP_SHA256 = "66791eaa646aac2e8cb24bb00fe30b2801436302327b1c46fea650446c41c4ac"
EXPECTED_VIDEO_METADATA = {
    "frame_count": EXPECTED_FRAME_COUNT,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_s": 32.0,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    info = {"path": str(path), "exists": path.exists(), "is_file": path.is_file() if path.exists() else False, "bytes": path.stat().st_size if path.exists() and path.is_file() else None}
    if hash_file and path.exists() and path.is_file():
        info["sha256"] = sha256(path)
    return info


def task5_requirement_case(requirement: dict[str, Any]) -> dict[str, Any]:
    for case in requirement.get("cases", []) if isinstance(requirement.get("cases"), list) else []:
        if isinstance(case, dict) and case.get("case") == TASK5_CASE:
            return case
    return {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    requirement_path = args.output_root / "hawor_requirement_state" / "v18_hawor_requirement_state.json"
    provisioning_path = args.output_root / "hawor_provisioning_audit" / "v18_hawor_provisioning_audit_report.json"
    requirement = load_json(requirement_path) if requirement_path.exists() else {}
    provisioning = load_json(provisioning_path) if provisioning_path.exists() else {}
    task5_req = task5_requirement_case(requirement)
    expected_npz = args.expected_local_output_dir / "hawor_world_hands.npz"
    expected_qc = args.expected_local_output_dir / "qc_hawor_world_hands.json"
    missing_required = provisioning.get("missing_required") if isinstance(provisioning.get("missing_required"), list) else []
    output_exists = expected_npz.exists()
    local_clip_info = file_info(TASK5_LOCAL_CLIP, hash_file=True)
    local_clip_sha256 = local_clip_info.get("sha256")
    local_clip_matches_expected = local_clip_sha256 == EXPECTED_LOCAL_CLIP_SHA256
    remote_clip_sha256_env = local_clip_sha256 or EXPECTED_LOCAL_CLIP_SHA256
    status = "task5_hawor_output_present_needs_requirement_rebuild" if output_exists else "blocked_task5_hawor_export_contract_written_waiting_for_external_assets_or_output"
    report = {
        "method": "build_v18_hawor_task5_export_contract",
        "status": status,
        "claim_scope": "task5_HaWoR_export_contract_only_no_execution_no_model_substitution_no_physical_acceptance",
        "case": TASK5_CASE,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "expected_frame_side_rows": EXPECTED_FRAME_SIDE_ROWS,
        "local_raw_clip": local_clip_info,
        "task5_clip_identity": {
            "expected_local_clip_sha256": EXPECTED_LOCAL_CLIP_SHA256,
            "local_clip_sha256": local_clip_sha256,
            "local_clip_matches_expected_sha256": local_clip_matches_expected,
            "expected_video_metadata": EXPECTED_VIDEO_METADATA,
            "remote_clip_sha256_env": remote_clip_sha256_env,
            "remote_preflight_enforced_by": "scripts/remote_run_hawor_export.sh EGO_HAWOR_CLIP_SHA256",
            "identity_claim_scope": "clip_identity_guard_only_no_HaWoR_or_physical_acceptance",
        },
        "remote_clip_expected_path": args.remote_clip,
        "expected_local_output_npz": file_info(expected_npz),
        "expected_local_qc_json": file_info(expected_qc),
        "expected_remote_output_dir": args.remote_output_dir,
        "required_external_assets": [
            "HaWoR repo with .git and submodules",
            "weights/hawor/checkpoints/hawor.ckpt",
            "weights/hawor/checkpoints/infiller.pt",
            "weights/hawor/model_config.yaml",
            "licensed MANO_LEFT.pkl",
            "licensed MANO_RIGHT.pkl",
            "DROID/Metric3D/HaWoR dependency weights from remote_setup_hawor.sh",
            "CUDA-capable environment validated by remote_setup_hawor.sh",
        ],
        "current_missing_required_from_provisioning_audit": missing_required,
        "setup_command": f"EGO_HAWOR_ROOT={TASK5_REMOTE_ROOT} EGO_MANO_ROOT={TASK5_REMOTE_ROOT}/assets/mano scripts/remote_setup_hawor.sh",
        "remote_export_command": f"EGO_HAWOR_ROOT={TASK5_REMOTE_ROOT} EGO_HAWOR_CASE=task5_tomato_960 EGO_HAWOR_CLIP={args.remote_clip} EGO_HAWOR_CLIP_SHA256={remote_clip_sha256_env} EGO_HAWOR_OUTPUT_DIR={args.remote_output_dir} scripts/remote_run_hawor_export.sh",
        "post_copy_expected_local_layout": {
            "npz": str(expected_npz),
            "qc_json": str(expected_qc),
            "expected_qc_video_sha256": EXPECTED_LOCAL_CLIP_SHA256,
            "expected_npz_video_sha256": EXPECTED_LOCAL_CLIP_SHA256,
            "expected_qc_export_provenance_hash_fields": ["checkpoint", "infiller_weight", "model_config"],
            "expected_npz_export_provenance_hash_fields": ["checkpoint_sha256", "infiller_weight_sha256", "model_config_sha256"],
            "copy_note": "Copy hawor_world_hands.npz and qc_hawor_world_hands.json from the remote output directory into this local directory before rebuilding V18 HaWoR requirement state; both files should record the same video_sha256 as this contract and matching SHA256 hashes for checkpoint, infiller_weight, and model_config.",
        },
        "post_ingest_validation_commands": [
            f".venv/bin/python scripts/build_v18_hawor_requirement_state.py --output-root {args.output_root} --hash-sources",
            f".venv/bin/python scripts/validate_v18_hawor_requirement_state.py --root {args.output_root}",
            f".venv/bin/python scripts/build_v18_hawor_bridge_state.py --output-root {args.output_root}",
            f".venv/bin/python scripts/validate_v18_hawor_bridge_state.py --root {args.output_root}",
            f".venv/bin/python scripts/build_v18_hawor_bridge_quality_state.py --output-root {args.output_root}",
            f".venv/bin/python scripts/validate_v18_hawor_bridge_quality_state.py --root {args.output_root}",
            f".venv/bin/python scripts/build_v18_hawor_bridge_subset_policy.py --output-root {args.output_root}",
            f".venv/bin/python scripts/validate_v18_hawor_bridge_subset_policy.py --root {args.output_root}",
        ],
        "acceptance_flags": {
            "task5_hawor_output_present": output_exists,
            "task5_clip_identity_verified_locally": local_clip_matches_expected,
            "accepted_v18_hawor_requirement_met": False,
            "accepted_metric_hand_state_from_hawor": False,
            "accepted_contact_occlusion_nonpenetration": False,
        },
        "requirement_state_task5_snapshot": {
            "status": task5_req.get("status"),
            "hawor_output": task5_req.get("hawor_output"),
            "available_hawor_frame_side_rows": task5_req.get("available_hawor_frame_side_rows"),
            "blocking_reasons": task5_req.get("blocking_reasons"),
        },
        "blocking_reasons": [
            "external_HaWoR_assets_or_output_required_for_task5",
            "do_not_substitute_WiLoR_HaMeR_MANO2D_or_depth_probe",
            "post_ingest_bridge_and_downstream_validation_required_before_any_physical_claim",
            "remote_task5_clip_sha256_must_match_contract_before_export",
        ] + (["task5_local_clip_sha256_mismatch_or_missing"] if not local_clip_matches_expected else []) + (["task5_expected_hawor_npz_absent_at_contract_path"] if not output_exists else ["task5_hawor_npz_present_but_not_validated_or_accepted_by_this_contract"]),
        "elapsed_s": time.perf_counter() - start,
    }
    out_dir = args.output_root / "hawor_task5_export_contract"
    write_json(out_dir / "v18_hawor_task5_export_contract.json", report)
    lines = [
        "# V18 task5 HaWoR export contract",
        "",
        "This is a HaWoR-only execution contract. It does not run HaWoR, does not substitute another hand model, and does not accept physical V18 state.",
        "",
        f"Status: `{status}`",
        f"Task5 local clip: `{TASK5_LOCAL_CLIP}` exists=`{TASK5_LOCAL_CLIP.exists()}`",
        f"Task5 local clip SHA256: `{local_clip_sha256}` expected=`{EXPECTED_LOCAL_CLIP_SHA256}` match=`{local_clip_matches_expected}`",
        f"Remote export command includes `EGO_HAWOR_CLIP_SHA256={remote_clip_sha256_env}` and `remote_run_hawor_export.sh` refuses mismatches.",
        f"Expected local output NPZ: `{expected_npz}` exists=`{expected_npz.exists()}`",
        f"Expected frame-side rows: `{EXPECTED_FRAME_SIDE_ROWS}`",
        "",
        "## Required command sequence",
        "",
        f"1. Setup: `{report['setup_command']}`",
        f"2. Export: `{report['remote_export_command']}`",
        f"3. Copy remote `hawor_world_hands.npz` and `qc_hawor_world_hands.json` into `{args.expected_local_output_dir}`.",
        "4. Run the post-ingest validation commands recorded in the JSON report.",
        "",
        "## Current blockers",
        "",
    ]
    for blocker in report["blocking_reasons"]:
        lines.append(f"- {blocker}")
    (out_dir / "V18_HAWOR_TASK5_EXPORT_CONTRACT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--expected-local-output-dir", type=Path, default=EXPECTED_LOCAL_OUTPUT_DIR)
    parser.add_argument("--remote-clip", default=TASK5_REMOTE_CLIP)
    parser.add_argument("--remote-output-dir", default=EXPECTED_REMOTE_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
