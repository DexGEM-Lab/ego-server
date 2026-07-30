#!/usr/bin/env python3
"""One-line single-video annotation entry.

Usage:
    python ego_annotate.py <video_path> [--unidepth-fps-drop 0.5] [--droid-fps-drop 0.5] [--no-report]

This is a thin wrapper around run_single_video_api.py that:
  - auto-generates case_id and run_root under /home/zjh/data/
  - defaults to the fixed worktree as repo-root
  - accepts human-friendly FPS-drop fractions (0.5 = half source FPS)
  - enables the integrated timing/batch report by default (--no-report to skip)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation_worktree")
PY = "/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="path to the input video file")
    parser.add_argument("--unidepth-fps-drop", type=float, default=None, help="UniDepth sampling fraction of source FPS (e.g. 0.5 = half). Default: no downsampling.")
    parser.add_argument("--droid-fps-drop", type=float, default=None, help="DROID sampling fraction of source FPS (e.g. 0.5 = half). Default: no downsampling.")
    parser.add_argument("--no-report", action="store_true", help="skip the integrated timing/batch report and package")
    parser.add_argument("--repo-root", type=Path, default=WORKTREE, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_id = f"ego_annotate_{stamp}"
    base = Path("/home/zjh/data") / case_id
    run_root = base / "run_client"
    base.mkdir(parents=True, exist_ok=True)

    profile = args.repo_root / "configs" / "feishu_ray_services.json"
    events_path = base / "client_admission_events.jsonl"
    package_root = base / "package"
    package_name = f"{case_id}_annotation_result"

    cmd = [
        PY,
        str(args.repo_root / "scripts" / "run_v22_api_job_with_admission.py"),
        "--job-id", case_id,
        "--repo-root", str(args.repo_root),
        "--profile", str(profile),
        "--lock-root", "/home/zjh/data/v22_api_jobs/_algorithm_admission",
        "--events-path", str(events_path),
        "--algorithm-inflight-multiplier", "1",
        "--api-ify",
        "--package-root", str(package_root),
        "--package-name", package_name,
        "--",
        PY,
        str(args.repo_root / "scripts" / "run_single_video_api.py"),
        "--case-id", case_id,
        "--input", str(args.input),
        "--run-root", str(run_root),
        "--diagnostic-monocular",
    ]
    if args.unidepth_fps_drop is not None:
        cmd += ["--unidepth-fps-drop", str(args.unidepth_fps_drop)]
    if args.droid_fps_drop is not None:
        cmd += ["--droid-fps-drop", str(args.droid_fps_drop)]
    if args.no_report:
        cmd += ["--no-report"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo_root)
    proc = subprocess.run(cmd, env=env, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
