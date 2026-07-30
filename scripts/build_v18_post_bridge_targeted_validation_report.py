#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "elapsed_s": time.perf_counter() - start,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    py = args.python
    commands = [
        [py, "scripts/validate_v18_hawor_bridge_state.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_bridge_quality_state.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_bridge_downstream_coverage.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_bridge_subset_policy.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_strict_contact_probe.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_temporal_offset_probe.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_requirement_state.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_hawor_task5_export_contract.py", "--root", str(args.output_root)],
        [py, "scripts/validate_v18_corrective_annotation_state.py", "--root", str(args.output_root)],
    ]
    validations = [run_command(command, args.repo_root) for command in commands]
    old_pipeline = args.output_root / "v18_corrective_1600_pipeline_report.json"
    old_payload = load_json(old_pipeline) if old_pipeline.exists() else None
    active_partial = args.output_root / "v18_corrective_1600_pipeline_report.partial.json"
    stale_partial_archives = sorted(str(p) for p in (args.output_root / "stale_pipeline_partials").glob("*.json")) if (args.output_root / "stale_pipeline_partials").exists() else []
    report = {
        "method": "build_v18_post_bridge_targeted_validation_report",
        "status": "ok" if all(v["status"] == "ok" for v in validations) else "failed",
        "claim_scope": "post_bridge_targeted_validation_only_long_corrective_pipeline_not_rerun_after_HaWoR_bridge_stage_additions",
        "output_root": str(args.output_root),
        "pre_bridge_pipeline_report": {
            "path": str(old_pipeline),
            "exists": old_pipeline.exists(),
            "status": old_payload.get("status") if isinstance(old_payload, dict) else None,
            "completed_stage_count": old_payload.get("completed_stage_count") if isinstance(old_payload, dict) else None,
            "stage_count": old_payload.get("stage_count") if isinstance(old_payload, dict) else None,
            "scope_note": "pre_HaWoR_bridge_quality_downstream_coverage_pipeline_report_not_current_committed_runner_validation",
        },
        "long_pipeline_rerun_after_bridge_changes": False,
        "why_not_rerun_long_pipeline": "task5_HaWoR_provisioning_blocker_remains_unresolved; targeted validators are the honest current evidence for bridge/quality/coverage additions",
        "active_partial_pipeline_report": {
            "path": str(active_partial),
            "exists": active_partial.exists(),
            "expected_current_state": "absent_after_success_or_explicit_stale_archive",
        },
        "stale_partial_pipeline_report_archives": stale_partial_archives,
        "validations": validations,
        "artifacts_validated": [
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_state_summary.json"),
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_quality_state_summary.json"),
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_downstream_coverage_summary.json"),
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_bridge_subset_policy_summary.json"),
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_strict_contact_probe_summary.json"),
            str(args.output_root / "hawor_bridge_state" / "v18_hawor_temporal_offset_probe_summary.json"),
            str(args.output_root / "hawor_requirement_state" / "v18_hawor_requirement_state.json"),
            str(args.output_root / "hawor_task5_export_contract" / "v18_hawor_task5_export_contract.json"),
            str(args.output_root / "trash_1050" / "annotations_v18_corrective_state.json"),
            str(args.output_root / "task5_tomato_960" / "annotations_v18_corrective_state.json"),
        ],
        "elapsed_s": time.perf_counter() - start,
    }
    write_json(args.output_root / "v18_post_bridge_targeted_validation_report.json", report)
    status_md = args.output_root / "V18_PIPELINE_REPORT_SCOPE_NOTE.md"
    status_md.write_text(
        "# V18 pipeline report scope note\n\n"
        "`v18_corrective_1600_pipeline_report.json` is the pre-HaWoR-bridge synchronized corrective pipeline report. "
        "It records the earlier 21-stage diagnostic corrective bundle and was not rerun after HaWoR bridge, bridge-quality, task5-export-contract, annotation integration, downstream-coverage, subset-policy, strict-contact-probe, and temporal-offset-probe stages were added.\n\n"
        "Current post-bridge evidence is the targeted validation report:\n\n"
        f"- `{args.output_root / 'v18_post_bridge_targeted_validation_report.json'}`\n\n"
        f"Active partial report exists: `{active_partial.exists()}`. "
        "Any interrupted stale partial evidence is preserved under `stale_pipeline_partials/` rather than the active `.partial.json` path.\n\n"
        "This note exists to prevent the old 21/21 report or interrupted partial reports from being misread as current validation of the committed runner.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "ok":
        raise SystemExit(1)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600"))
    parser.add_argument("--python", default=".venv/bin/python")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
