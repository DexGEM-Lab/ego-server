#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def run_stage(name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    py = args.python
    stages = [
        ("graph_corrective_render", [py, "scripts/render_v18_corrective_state.py", "--output-root", str(args.output_root)]),
        ("temporal_hand_pose_smoothing", [py, "scripts/render_v18_temporal_hand_pose_smoothing.py", "--output-root", str(args.output_root)]),
        ("generic_rigid_se3_attempt", [py, "scripts/render_v18_rigid_se3_attempt.py", "--output-root", str(args.output_root), "--max-points-per-object", "900"]),
        ("hawor_ghost_or_failure", [py, "scripts/render_v18_hawor_ghost_attempt.py", "--output-root", str(args.output_root)]),
        ("hawor_provisioning_audit", [py, "scripts/audit_v18_hawor_provisioning.py", "--output-root", str(args.output_root)]),
        ("hawor_bridge_state", [py, "scripts/build_v18_hawor_bridge_state.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_bridge_state", [py, "scripts/validate_v18_hawor_bridge_state.py", "--root", str(args.output_root)]),
        ("hawor_bridge_quality_state", [py, "scripts/build_v18_hawor_bridge_quality_state.py", "--output-root", str(args.output_root)]),
        ("hawor_bridge_quality_overlay", [py, "scripts/render_v18_hawor_bridge_quality_overlay.py", "--output-root", str(args.output_root), "--case", "trash_1050"]),
        ("validate_hawor_bridge_quality_state", [py, "scripts/validate_v18_hawor_bridge_quality_state.py", "--root", str(args.output_root)]),
        ("hawor_hard_requirement_state", [py, "scripts/build_v18_hawor_requirement_state.py", "--output-root", str(args.output_root), "--hash-sources"]),
        ("validate_hawor_hard_requirement_state", [py, "scripts/validate_v18_hawor_requirement_state.py", "--root", str(args.output_root)]),
        ("hawor_task5_export_contract", [py, "scripts/build_v18_hawor_task5_export_contract.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_task5_export_contract", [py, "scripts/validate_v18_hawor_task5_export_contract.py", "--root", str(args.output_root)]),
        ("mano_foundation_state", [py, "scripts/build_v18_mano_foundation_state.py", "--output-root", str(args.output_root / "mano_foundation_audit"), "--hash-sources"]),
        ("validate_mano_foundation_state", [py, "scripts/validate_v18_mano_foundation_state.py", "--root", str(args.output_root / "mano_foundation_audit")]),
        ("mano_foundation_overlay", [py, "scripts/render_v18_mano_foundation_overlay.py", "--output-root", str(args.output_root)]),
        ("visible_surface_state", [py, "scripts/render_v18_visible_surface_state.py", "--output-root", str(args.output_root)]),
        ("geometry_coverage_audit", [py, "scripts/render_v18_geometry_coverage_audit.py", "--output-root", str(args.output_root)]),
        ("tentative_occlusion_owner", [py, "scripts/render_v18_occlusion_owner_best_effort.py", "--output-root", str(args.output_root)]),
        ("occlusion_owner_acceptance_audit", [py, "scripts/render_v18_occlusion_owner_acceptance_audit.py", "--output-root", str(args.output_root)]),
        ("contact_nonpenetration_state", [py, "scripts/render_v18_contact_nonpenetration_state.py", "--output-root", str(args.output_root)]),
        ("contact_acceptance_audit", [py, "scripts/render_v18_contact_acceptance_audit.py", "--output-root", str(args.output_root)]),
        ("nonpenetration_repair_proposal", [py, "scripts/render_v18_nonpenetration_repair_proposal.py", "--output-root", str(args.output_root)]),
        ("rigid_se3_residual_check", [py, "scripts/render_v18_rigid_se3_residual_check.py", "--output-root", str(args.output_root)]),
        ("corrective_annotation_state", [py, "scripts/build_v18_corrective_annotation_state.py", "--output-root", str(args.output_root), "--corrective-root", str(args.output_root)]),
        ("validate_corrective_annotation_state", [py, "scripts/validate_v18_corrective_annotation_state.py", "--root", str(args.output_root)]),
        ("hawor_bridge_downstream_coverage", [py, "scripts/build_v18_hawor_bridge_downstream_coverage.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_bridge_downstream_coverage", [py, "scripts/validate_v18_hawor_bridge_downstream_coverage.py", "--root", str(args.output_root)]),
        ("hawor_bridge_subset_policy", [py, "scripts/build_v18_hawor_bridge_subset_policy.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_bridge_subset_policy", [py, "scripts/validate_v18_hawor_bridge_subset_policy.py", "--root", str(args.output_root)]),
        ("hawor_strict_contact_probe", [py, "scripts/build_v18_hawor_strict_contact_probe.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_strict_contact_probe", [py, "scripts/validate_v18_hawor_strict_contact_probe.py", "--root", str(args.output_root)]),
        ("hawor_temporal_offset_probe", [py, "scripts/build_v18_hawor_temporal_offset_probe.py", "--output-root", str(args.output_root)]),
        ("validate_hawor_temporal_offset_probe", [py, "scripts/validate_v18_hawor_temporal_offset_probe.py", "--root", str(args.output_root)]),
        ("corrective_review_sheets", [py, "scripts/build_v18_corrective_review_sheets.py", "--corrective-root", str(args.output_root)]),
        ("corrective_montage", [py, "scripts/render_v18_corrective_montage.py", "--output-root", str(args.output_root)]),
        ("post_bridge_targeted_validation_report", [py, "scripts/build_v18_post_bridge_targeted_validation_report.py", "--output-root", str(args.output_root)]),
        ("corrective_bundle_manifest", [py, "scripts/build_v18_corrective_bundle_manifest.py", "--output-root", str(args.output_root)]),
    ]
    results = []
    partial_path = args.output_root / "v18_corrective_1600_pipeline_report.partial.json"
    for name, cmd in stages:
        result = run_stage(name, cmd, args.repo_root)
        results.append(result)
        write_json(partial_path, {
            "method": "run_v18_corrective_1600_pipeline",
            "status": "running" if result["returncode"] == 0 else "failed",
            "completed_stage_count": len(results),
            "stage_count": len(stages),
            "stages": results,
        })
        if result["returncode"] != 0:
            break
    report = {
        "method": "run_v18_corrective_1600_pipeline",
        "status": "ok" if all(r["returncode"] == 0 for r in results) and len(results) == len(stages) else "failed",
        "claim_scope": "cached_v18_evidence_to_corrective_bundle_runtime; not_raw_video_to_full_v18_closure",
        "output_root": str(args.output_root),
        "stage_count": len(stages),
        "completed_stage_count": len(results),
        "total_elapsed_s": time.perf_counter() - start,
        "stages": results,
    }
    write_json(args.output_root / "v18_corrective_1600_pipeline_report.json", report)
    if partial_path.exists():
        partial_path.unlink()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--python", default=".venv/bin/python")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
