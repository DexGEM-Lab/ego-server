#!/usr/bin/env python3
"""Run the paired API-Ify full/20/15/10 FPS stable-window conditions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego_annotation.fps_config import get_fps_condition


MANAGER = REPO_ROOT / "scripts" / "run_v22_api_egoscale30h_batch.py"
SUMMARIZER = REPO_ROOT / "scripts" / "summarize_fps_production_condition.py"
DEFAULT_OUTPUT_PARENT = Path("/home/zjh/data/api_ify_fps_autoresearch")
DEFAULT_STABILITY_VIDEO_LIMIT = 30
OUTPUT_ROOT_ENV = "EGO_API_IFY_OUTPUT_ROOT"
STABILITY_VIDEO_LIMIT_ENV = "EGO_API_IFY_STABILITY_VIDEO_LIMIT"
PAIRED_CONDITIONS = (
    "unidepth_full__droid_full",
    "unidepth_20fps__droid_20fps",
    "unidepth_15fps__droid_15fps",
    "unidepth_10fps__droid_10fps",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def configured_output_root(environ: Mapping[str, str] = os.environ) -> Path:
    configured = environ.get(OUTPUT_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_PARENT / stamp


def configured_stability_video_limit(environ: Mapping[str, str] = os.environ) -> int:
    value = int(environ.get(STABILITY_VIDEO_LIMIT_ENV, str(DEFAULT_STABILITY_VIDEO_LIMIT)))
    if value < 0:
        raise ValueError(f"{STABILITY_VIDEO_LIMIT_ENV} must be non-negative")
    return value


def main() -> int:
    research_root = configured_output_root()
    stability_video_limit = configured_stability_video_limit()
    manifest_path = research_root / "autoresearch_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "ego.annotation.api_ify_fps_autoresearch.v1",
        "status": "running",
        "research_root": str(research_root),
        "manager": str(MANAGER),
        "public_contract": "file-only POST /v1/annotation-jobs; FPS is manager-owned",
        "dataset_root": "/home/zjh/data/egoscale_demo_30h",
        "stability_video_limit": stability_video_limit,
        "rolling_control": {
            "api_client_concurrency": int(os.environ.get("EGO_API_IFY_API_CLIENT_CONCURRENCY", "8")),
            "warmup_count": int(os.environ.get("EGO_API_IFY_STABILITY_WARMUP_COUNT", "4")),
            "window_size": int(os.environ.get("EGO_API_IFY_STABILITY_WINDOW_SIZE", "4")),
            "tolerance": float(os.environ.get("EGO_API_IFY_STABILITY_TOLERANCE", "0.1")),
        },
        "conditions": [],
    }
    write_json(manifest_path, manifest)
    for condition_name in PAIRED_CONDITIONS:
        condition = get_fps_condition(condition_name)
        condition_root = research_root / condition.name
        env = os.environ.copy()
        env.update(
            {
                "EGO_API_IFY_FPS_CONDITION": condition.name,
                "EGO_API_IFY_STABILITY_VIDEO_LIMIT": str(stability_video_limit),
                "EGO_API_IFY_OUTPUT_ROOT": str(condition_root),
            }
        )
        row: dict[str, Any] = {
            "condition": condition.name,
            "unidepth_fps": condition.unidepth_fps,
            "droid_fps": condition.droid_fps,
            "condition_root": str(condition_root),
            "status": "running",
        }
        manifest["conditions"].append(row)
        write_json(manifest_path, manifest)
        manager_log = research_root / f"{condition.name}.manager.log"
        with manager_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, str(MANAGER)], cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        row["manager_returncode"] = int(completed.returncode)
        report_path = condition_root / "fps_production_condition_metrics.json"
        summary = subprocess.run([sys.executable, str(SUMMARIZER), "--condition-root", str(condition_root), "--output", str(report_path)], cwd=str(REPO_ROOT), check=False)
        row["summary_returncode"] = int(summary.returncode)
        row["report"] = str(report_path)
        row["status"] = "completed" if completed.returncode == 0 and summary.returncode == 0 else "failed"
        write_json(manifest_path, manifest)
        if row["status"] != "completed":
            manifest["status"] = "stopped_on_condition_failure"
            write_json(manifest_path, manifest)
            return int(completed.returncode or summary.returncode or 1)
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
