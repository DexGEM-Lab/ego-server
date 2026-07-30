#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def repo_exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def command_string(parts: list[str]) -> str:
    return " ".join(parts)


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.input_manifest)
    raw_manifest = Path(str(manifest["raw_frame_manifest"]))
    run_root = Path(str(manifest["run_root"]))
    frame_count = int(manifest["raw_frame_manifest_summary"]["frame_count"])
    frame_start = 0
    frame_end = frame_count - 1
    primary_video = Path(str(manifest["primary_video"]))
    primary_meta = manifest["primary_video_metadata"]
    jobs: list[dict[str, Any]] = []

    depthpro_repo = args.depthpro_repo
    if repo_exists(depthpro_repo):
        jobs.append(
            {
                "job_id": "depthpro_full_timeline_monocular_baseline",
                "stage": "depth_camera",
                "compute_target": args.compute_target,
                "status": "ready_to_run",
                "command": command_string(
                    [
                        args.python_bin,
                        "scripts/run_v21_depthpro_full_frame_candidate.py",
                        "--raw-frame-manifest",
                        str(raw_manifest),
                        "--output-dir",
                        str(run_root / "measurements" / "depth_candidates" / "depthpro_full_frame"),
                        "--frame-start",
                        str(frame_start),
                        "--frame-end",
                        str(frame_end),
                        "--depthpro-repo",
                        str(depthpro_repo),
                        "--source-width",
                        str(primary_meta["width"]),
                        "--source-height",
                        str(primary_meta["height"]),
                    ]
                ),
            }
        )
    else:
        jobs.append(
            {
                "job_id": "depthpro_full_timeline_monocular_baseline",
                "stage": "depth_camera",
                "compute_target": args.compute_target,
                "status": "blocked_missing_repo",
                "missing_repo": str(depthpro_repo) if depthpro_repo else None,
            }
        )

    unidepth_repo = args.unidepth_repo
    if repo_exists(unidepth_repo):
        jobs.append(
            {
                "job_id": "unidepth_full_timeline_monocular_baseline",
                "stage": "depth_camera",
                "compute_target": args.compute_target,
                "status": "ready_to_run",
                "command": command_string(
                    [
                        args.python_bin,
                        "scripts/run_unidepth_full_frame_v3.py",
                        "--manifest",
                        str(raw_manifest),
                        "--output-dir",
                        str(run_root / "measurements" / "depth_candidates" / "unidepth_full_frame"),
                        "--frame-start",
                        str(frame_start),
                        "--frame-end",
                        str(frame_end),
                        "--unidepth-repo",
                        str(unidepth_repo),
                        "--source-width",
                        str(primary_meta["width"]),
                        "--source-height",
                        str(primary_meta["height"]),
                    ]
                ),
            }
        )
    else:
        jobs.append(
            {
                "job_id": "unidepth_full_timeline_monocular_baseline",
                "stage": "depth_camera",
                "compute_target": args.compute_target,
                "status": "blocked_missing_repo",
                "missing_repo": str(unidepth_repo) if unidepth_repo else None,
            }
        )

    stereo_right = manifest.get("stereo_right_video")
    if stereo_right:
        jobs.append(
            {
                "job_id": "stereo_sgbm_relative_inverse_depth_smoke",
                "stage": "depth_camera",
                "compute_target": "local_cpu_allowed",
                "status": "ready_to_run",
                "command": command_string(
                    [
                        "python",
                        "scripts/run_v21_stereo_sgbm_candidate.py",
                        "--raw-frame-manifest",
                        str(raw_manifest),
                        "--left-video",
                        str(primary_video),
                        "--right-video",
                        str(stereo_right),
                        "--output-npz",
                        str(run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "relative_inverse_depth.npz"),
                        "--output-report",
                        str(run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "report.json"),
                        "--preview-dir",
                        str(run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "preview"),
                    ]
                ),
            }
        )

    jobs.append(
        {
            "job_id": "depth_candidate_selection_after_monocular_baselines",
            "stage": "depth_camera",
            "compute_target": "local_cpu_allowed",
            "status": "pending_upstream_depth_candidates",
            "command": command_string(
                [
                    "python",
                    "scripts/select_v21_depth_camera_bundle.py",
                    "--run-root",
                    str(run_root),
                    "--input-manifest",
                    str(args.input_manifest),
                    "--modality-report",
                    str(run_root / "measurements" / "camera_depth" / "depth_modality_report.json"),
                    "--output-report",
                    str(run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json"),
                ]
            ),
        }
    )

    payload = {
        "schema": "v21_model_job_plan.v0",
        "status": "ok",
        "method": "plan_v21_model_jobs",
        "case_id": manifest.get("case_id"),
        "run_root": str(run_root),
        "input_manifest": str(args.input_manifest),
        "compute_target_policy": "Commands that import torch/model repositories must run on the declared compute_target; local workstation heavy GPU is forbidden unless compute_target explicitly names it.",
        "compute_target": args.compute_target,
        "jobs": jobs,
    }
    write_json(args.output_plan, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write V21 model job commands bound to a run root and compute target.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--compute-target", default="ylang-U22_local_4090_explicitly_authorized_for_this_run")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--depthpro-repo", type=Path, default=Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/depthpro_work/ml-depth-pro"))
    parser.add_argument("--unidepth-repo", type=Path, default=Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/unidepth_work/UniDepth"))
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
