#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - start,
        "output": proc.stdout,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def require_ok(step: dict[str, Any]) -> None:
    if step["returncode"] != 0:
        raise RuntimeError(f"command failed ({step['returncode']}): {' '.join(step['cmd'])}\n{step['output'][-4000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--task5-constraint-report", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/task5_tomato_960/object_obj_tomato/surface806_sign929_full_bridge_all_signed/iter1_remeasure/v18_mano_object_constraint_state_full_bridge.json"))
    parser.add_argument("--trash-constraint-report", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_bridge_all_signed_rebuild_v1/trash_1050/object_pink_lid_trash_can_second/frame872_full_bridge_all_signed/iter1_remeasure/v18_mano_object_constraint_state_full_bridge.json"))
    parser.add_argument("--verified-annotation", action="append", default=[], help="CASE=/path/to/verified annotations override passed to merge and verifier")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    merge_cmd = [
        args.python,
        "scripts/build_v18_verified_hprime_final_annotations.py",
        "--base-root",
        str(args.base_root),
        "--output-root",
        str(args.output_root),
    ]
    for item in args.verified_annotation:
        merge_cmd.extend(["--verified-annotation", item])
    step = run(merge_cmd)
    steps.append(step)
    require_ok(step)

    for case in ["task5_tomato_960", "trash_1050"]:
        render_cmd = [
            args.python,
            "scripts/render_v18_full_pipeline_from_annotations.py",
            "--case",
            case,
            "--annotations",
            str(args.output_root / case / "annotations_v18_full.json"),
            "--output-root",
            str(args.output_root),
        ]
        step = run(render_cmd)
        steps.append(step)
        require_ok(step)

    review_cmds = [
        [
            args.python,
            "scripts/render_v18_mano_constraint_final_artifact_review.py",
            "--render-root",
            str(args.output_root),
            "--case",
            "task5_tomato_960",
            "--annotations",
            str(args.output_root / "task5_tomato_960" / "annotations_v18_full.json"),
            "--constraint-report",
            str(args.task5_constraint_report),
            "--output",
            str(args.output_root / "task5_tomato_960" / "verified_hprime_final_artifact_review.jpg"),
        ],
        [
            args.python,
            "scripts/render_v18_mano_constraint_final_artifact_review.py",
            "--render-root",
            str(args.output_root),
            "--case",
            "trash_1050",
            "--annotations",
            str(args.output_root / "trash_1050" / "annotations_v18_full.json"),
            "--constraint-report",
            str(args.trash_constraint_report),
            "--output",
            str(args.output_root / "trash_1050" / "verified_hprime_final_artifact_review.jpg"),
        ],
    ]
    for cmd in review_cmds:
        step = run(cmd)
        steps.append(step)
        require_ok(step)

    verify_cmd = [
        args.python,
        "scripts/verify_v18_verified_hprime_final_artifact.py",
        "--root",
        str(args.output_root),
        "--summary",
        str(args.output_root / "verify_v18_verified_hprime_final_artifact.json"),
        "--task5-remeasure-report",
        str(args.task5_constraint_report),
        "--trash-remeasure-report",
        str(args.trash_constraint_report),
    ]
    for item in args.verified_annotation:
        verify_cmd.extend(["--verified-annotation", item])
    step = run(verify_cmd)
    steps.append(step)
    require_ok(step)

    manifest = {
        "method": "run_v18_verified_hprime_final_artifact",
        "status": "ok",
        "base_root": str(args.base_root),
        "output_root": str(args.output_root),
        "verified_annotation_overrides": list(args.verified_annotation),
        "steps": [
            {
                "cmd": s["cmd"],
                "returncode": s["returncode"],
                "elapsed_s": s["elapsed_s"],
                "output_tail": s["output"][-2000:],
            }
            for s in steps
        ],
        "final_verification": str(args.output_root / "verify_v18_verified_hprime_final_artifact.json"),
        "claim_scope": "Builds final two-case V18 artifact from sanitized base annotations plus verified compact-rigid H-prime hand-state consequences; then renders and verifies the consumed metric MANO state.",
    }
    write_json(args.output_root / "v18_verified_hprime_final_artifact_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
