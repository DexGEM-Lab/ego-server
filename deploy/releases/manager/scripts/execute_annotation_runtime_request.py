#!/usr/bin/env python3
"""Execute one annotation runtime request produced for a Pi runtime agent."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation_remote_runner import RemoteConfig, run_remote_annotation_job
from scripts.package_v22_annotation_result import create_result_package


class RuntimeRequestError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeRequestError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def path_or_none(value: Any) -> Path | None:
    return Path(str(value)) if value not in {None, ""} else None


def build_pipeline_cmd(request: dict[str, Any]) -> list[str]:
    flags = request.get("pipeline_flags") if isinstance(request.get("pipeline_flags"), dict) else {}
    job_id = str(request["job_id"])
    run_root = Path(str(request["run_root"]))
    repo_root = Path(str(request.get("repo_root") or REPO_ROOT))
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_v22_minimal_annotation_pipeline.py"),
        "--case-id",
        job_id,
        "--input-video",
        str(request["video_uri"]),
        "--run-root",
        str(run_root),
        "--repo-root",
        str(repo_root),
    ]
    for key, flag in [
        ("render_width", "--render-width"),
        ("start_s", "--start-s"),
        ("end_s", "--end-s"),
        ("gpu_ids", "--gpu-ids"),
        ("camera_backend", "--camera-backend"),
    ]:
        value = flags.get(key)
        if value not in {None, ""}:
            cmd.extend([flag, str(value)])
    for key, flag in [
        ("run_preflight", "--run-preflight"),
        ("run_camera_trajectory", "--run-camera-trajectory"),
        ("run_hawor_metric_hands", "--run-hawor-metric-hands"),
        ("run_hybrid_hands", "--run-hybrid-hands"),
        ("run_gt_free_drift_self_calibration", "--run-gt-free-drift-self-calibration"),
        ("run_captioning", "--run-captioning"),
        ("run_self_consistency_qc", "--run-self-consistency-qc"),
        ("run_evaluator", "--run-evaluator"),
        ("write_product_bundle", "--write-product-bundle"),
    ]:
        if bool(flags.get(key)):
            cmd.append(flag)
    for key, flag in [("actions_json", "--actions-json"), ("captions_json", "--captions-json"), ("semantic_review_json", "--semantic-review-json")]:
        value = flags.get(key)
        if value:
            cmd.extend([flag, str(value)])
    return cmd


def existing_source(flags: dict[str, Any]) -> bool:
    return bool(flags.get("actions_json") or flags.get("captions_json") or flags.get("semantic_review_json"))


def maybe_run_semantic_agent(request: dict[str, Any]) -> dict[str, Any] | None:
    flags = request.get("pipeline_flags") if isinstance(request.get("pipeline_flags"), dict) else {}
    if bool(flags.get("run_semantic_annotation_agent")):
        raise RuntimeRequestError("run_semantic_annotation_agent is disabled; D9b captions must come from scripted Cosmos or an explicit source sidecar")
    return None


def execute_local(request: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_root = Path(str(request["run_root"]))
    package_root = Path(str(request["package_root"]))
    repo_root = Path(str(request.get("repo_root") or REPO_ROOT))
    log_path = run_root / "logs" / "runtime_executor_pipeline.log"
    cmd = build_pipeline_cmd(request)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeRequestError(f"local_annotation_pipeline_failed rc={proc.returncode}; log={log_path}\n{tail}")
    package = create_result_package(run_root, package_root, package_name=f"{request['job_id']}_annotation_result")
    summary = load_json(run_root / "annotation_pipeline_manifest.json")
    return {
        "status": "ok",
        "backend": "local_script",
        "job_id": request["job_id"],
        "run_root": str(run_root),
        "manifest_path": str(run_root / "annotation_pipeline_manifest.json"),
        "local_package_path": str(package["package_path"]),
        "summary": summary,
        "elapsed_s": float(time.time() - started),
    }


def execute_remote(request: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    remote = request.get("remote") if isinstance(request.get("remote"), dict) else None
    if remote is None:
        raise RuntimeRequestError("remote backend requested without remote config")
    flags = request.get("pipeline_flags") if isinstance(request.get("pipeline_flags"), dict) else {}
    config = RemoteConfig(
        host=str(remote["host"]),
        repo_root=Path(str(remote["repo_root"])),
        output_root=Path(str(remote["output_root"])),
        upload_root=Path(str(remote["upload_root"])),
        package_root=Path(str(remote["package_root"])),
        python=Path(str(remote.get("python") or "/home/zjh/ego-annotation-api-venv/bin/python")),
    )
    local_video = Path(str(request["video_uri"])) if bool(request.get("local_video")) else None
    remote_video = None if local_video is not None else Path(str(request["video_uri"]))
    result = run_remote_annotation_job(
        config=config,
        job_id=str(request["job_id"]),
        local_video=local_video,
        remote_video=remote_video,
        local_package_root=Path(str(request["package_root"])),
        filename=Path(str(request["video_uri"])).name,
        start_s=flags.get("start_s"),
        end_s=flags.get("end_s"),
        render_width=int(flags["render_width"]) if flags.get("render_width") is not None else None,
        gpu_ids=flags.get("gpu_ids"),
        run_preflight=bool(flags.get("run_preflight")),
        run_camera_trajectory=bool(flags.get("run_camera_trajectory")),
        run_hawor_metric_hands=bool(flags.get("run_hawor_metric_hands")),
        run_hybrid_hands=bool(flags.get("run_hybrid_hands")),
        run_gt_free_drift_self_calibration=bool(flags.get("run_gt_free_drift_self_calibration")),
        run_captioning=bool(flags.get("run_captioning")),
        run_self_consistency_qc=bool(flags.get("run_self_consistency_qc")),
        run_evaluator=bool(flags.get("run_evaluator")),
        actions_json=path_or_none(flags.get("actions_json")),
        captions_json=path_or_none(flags.get("captions_json")),
        semantic_review_json=path_or_none(flags.get("semantic_review_json")),
        head_gt=None,
        hand_gt=None,
        write_product_bundle=bool(flags.get("write_product_bundle")),
        timeout=int(request.get("timeout_s") or 7200),
    )
    return {"status": "ok", "backend": "remote_ssh_script", "elapsed_s": float(time.time() - started), **result}


def execute(request_path: Path, result_path: Path | None = None) -> dict[str, Any]:
    request = load_json(request_path)
    if request.get("schema") != "ego.annotation.runtime_request.v1":
        raise RuntimeRequestError(f"unsupported runtime request schema: {request.get('schema')}")
    if request.get("head_gt") or request.get("hand_gt"):
        raise RuntimeRequestError("GT sidecars are not allowed in prediction runtime requests")
    flags = request.get("pipeline_flags") if isinstance(request.get("pipeline_flags"), dict) else {}
    model_backend = str(flags.get("model_backend") or "script")
    if model_backend == "feishu_ray":
        raise RuntimeRequestError("feishu_ray_pipeline_adapter_not_implemented")
    if model_backend != "script":
        raise RuntimeRequestError(f"unsupported model_backend: {model_backend}")
    if flags.get("service_profile") is not None or flags.get("service_endpoints") is not None:
        raise RuntimeRequestError("service_configuration_requires_feishu_ray")
    semantic_agent = maybe_run_semantic_agent(request)
    backend = str(request.get("execution_backend") or "local_script")
    if backend == "remote_ssh_script":
        result = execute_remote(request)
    elif backend == "local_script":
        result = execute_local(request)
    else:
        raise RuntimeRequestError(f"unsupported execution_backend: {backend}")
    out = result_path or Path(str(request.get("result_path") or Path(str(request["run_root"])) / "logs" / "runtime_agent_result.json"))
    write_json(out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    execute(args.request, args.result)
