#!/usr/bin/env python3
"""SSH/SCP remote execution backend for the annotation API.

This lets the HTTP API run on the local workstation while heavy model execution
runs on the A800 server using the same remote V22 pipeline scripts.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    repo_root: Path
    output_root: Path
    upload_root: Path
    package_root: Path
    python: Path = Path("/home/zjh/ego-annotation-api-venv/bin/python")


class RemoteExecutionError(RuntimeError):
    pass


def run_checked(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
    if proc.returncode != 0:
        raise RemoteExecutionError(f"command_failed rc={proc.returncode}: {cmd}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
    return proc


def ssh(config: RemoteConfig, command: str, *, timeout: int | None = None) -> str:
    return run_checked(["ssh", config.host, command], timeout=timeout).stdout


def scp_to(config: RemoteConfig, local: Path, remote: str, *, timeout: int | None = None) -> None:
    run_checked(["scp", str(local), f"{config.host}:{remote}"], timeout=timeout)


def scp_from(config: RemoteConfig, remote: str, local: Path, *, timeout: int | None = None) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["scp", f"{config.host}:{remote}", str(local)], timeout=timeout)


def q(value: object) -> str:
    return shlex.quote(str(value))


def remote_json(config: RemoteConfig, path: Path, *, timeout: int | None = None) -> dict[str, Any]:
    text = ssh(config, f"cat {q(path)}", timeout=timeout)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RemoteExecutionError(f"expected JSON object at remote path {path}")
    return payload


def build_pipeline_command(config: RemoteConfig, *, job_id: str, remote_video: Path, start_s: float | None, end_s: float | None, render_width: int | None, gpu_ids: str | None, run_preflight: bool, run_camera_trajectory: bool, run_hawor_metric_hands: bool, run_hybrid_hands: bool, run_gt_free_drift_self_calibration: bool, run_captioning: bool, run_self_consistency_qc: bool, run_evaluator: bool, actions_json: Path | None, captions_json: Path | None, head_gt: Path | None, hand_gt: Path | None, write_product_bundle: bool) -> str:
    run_root = config.output_root / job_id
    parts = [
        q(config.python),
        "scripts/run_v22_minimal_annotation_pipeline.py",
        "--case-id", q(job_id),
        "--input-video", q(remote_video),
        "--run-root", q(run_root),
        "--repo-root", q(config.repo_root),
    ]
    if render_width is not None:
        parts.extend(["--render-width", q(render_width)])
    if start_s is not None:
        parts.extend(["--start-s", q(start_s)])
    if end_s is not None:
        parts.extend(["--end-s", q(end_s)])
    if gpu_ids:
        parts.extend(["--gpu-ids", q(gpu_ids)])
    if run_preflight:
        parts.append("--run-preflight")
    if run_camera_trajectory:
        parts.append("--run-camera-trajectory")
    if run_hawor_metric_hands:
        parts.append("--run-hawor-metric-hands")
    if run_hybrid_hands:
        parts.append("--run-hybrid-hands")
    if run_gt_free_drift_self_calibration:
        parts.append("--run-gt-free-drift-self-calibration")
    if run_captioning:
        parts.append("--run-captioning")
    if run_self_consistency_qc:
        parts.append("--run-self-consistency-qc")
    if run_evaluator:
        parts.append("--run-evaluator")
    if actions_json is not None:
        parts.extend(["--actions-json", q(actions_json)])
    if captions_json is not None:
        parts.extend(["--captions-json", q(captions_json)])
    if head_gt is not None:
        parts.extend(["--head-gt", q(head_gt)])
    if hand_gt is not None:
        parts.extend(["--hand-gt", q(hand_gt)])
    if write_product_bundle:
        parts.append("--write-product-bundle")
    return f"cd {q(config.repo_root)} && " + " ".join(parts)


def run_remote_annotation_job(
    *,
    config: RemoteConfig,
    job_id: str,
    local_video: Path | None,
    remote_video: Path | None,
    local_package_root: Path,
    filename: str,
    start_s: float | None,
    end_s: float | None,
    render_width: int | None,
    gpu_ids: str | None,
    run_preflight: bool,
    run_camera_trajectory: bool,
    run_hawor_metric_hands: bool,
    run_hybrid_hands: bool,
    run_gt_free_drift_self_calibration: bool,
    run_captioning: bool,
    run_self_consistency_qc: bool,
    run_evaluator: bool,
    actions_json: Path | None,
    captions_json: Path | None,
    head_gt: Path | None,
    hand_gt: Path | None,
    write_product_bundle: bool,
    timeout: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    if local_video is None and remote_video is None:
        raise RemoteExecutionError("either local_video or remote_video must be provided")
    remote_upload_dir = config.upload_root / job_id
    if remote_video is None:
        remote_video = remote_upload_dir / Path(filename).name
    if local_video is not None and (not local_video.exists() or not local_video.is_file()):
        raise RemoteExecutionError(f"missing local video for remote execution: {local_video}")
    remote_run_root = config.output_root / job_id
    remote_package_name = f"{job_id}_annotation_result"
    remote_package_path = config.package_root / f"{remote_package_name}.zip"
    local_package_path = local_package_root.expanduser().resolve() / f"{remote_package_name}.zip"

    ssh(config, f"mkdir -p {q(remote_upload_dir)} {q(config.output_root)} {q(config.package_root)}", timeout=60)
    if local_video is not None:
        scp_to(config, local_video, str(remote_video), timeout=timeout)
    sidecars = {"actions_json": actions_json, "captions_json": captions_json, "head_gt": head_gt, "hand_gt": hand_gt}
    remote_sidecars: dict[str, Path | None] = {}
    for key, sidecar in sidecars.items():
        if sidecar is None:
            remote_sidecars[key] = None
            continue
        sidecar_path = sidecar.expanduser()
        if sidecar_path.exists() and sidecar_path.is_file():
            remote_path = remote_upload_dir / sidecar_path.name
            scp_to(config, sidecar_path.resolve(), str(remote_path), timeout=timeout)
            remote_sidecars[key] = remote_path
        else:
            remote_sidecars[key] = sidecar
    pipeline_cmd = build_pipeline_command(
        config,
        job_id=job_id,
        remote_video=remote_video,
        start_s=start_s,
        end_s=end_s,
        render_width=render_width,
        gpu_ids=gpu_ids,
        run_preflight=run_preflight,
        run_camera_trajectory=run_camera_trajectory,
        run_hawor_metric_hands=run_hawor_metric_hands,
        run_hybrid_hands=run_hybrid_hands,
        run_gt_free_drift_self_calibration=run_gt_free_drift_self_calibration,
        run_captioning=run_captioning,
        run_self_consistency_qc=run_self_consistency_qc,
        run_evaluator=run_evaluator,
        actions_json=remote_sidecars["actions_json"],
        captions_json=remote_sidecars["captions_json"],
        head_gt=remote_sidecars["head_gt"],
        hand_gt=remote_sidecars["hand_gt"],
        write_product_bundle=write_product_bundle,
    )
    ssh(config, pipeline_cmd, timeout=timeout)
    package_cmd = f"cd {q(config.repo_root)} && {q(config.python)} scripts/package_v22_annotation_result.py --run-root {q(remote_run_root)} --package-root {q(config.package_root)} --package-name {q(remote_package_name)}"
    package_json_text = ssh(config, package_cmd, timeout=timeout)
    package_json = json.loads(package_json_text)
    if not isinstance(package_json, dict) or package_json.get("status") != "ok":
        raise RemoteExecutionError(f"remote package failed: {package_json_text[-2000:]}")
    scp_from(config, str(remote_package_path), local_package_path, timeout=timeout)
    summary = remote_json(config, remote_run_root / "annotation_pipeline_manifest.json", timeout=timeout)
    return {
        "status": "ok",
        "job_id": job_id,
        "remote_run_root": str(remote_run_root),
        "remote_manifest_path": str(remote_run_root / "annotation_pipeline_manifest.json"),
        "remote_package_path": str(remote_package_path),
        "local_package_path": str(local_package_path),
        "summary": summary,
        "elapsed_s": float(time.time() - started),
    }


def config_from_env(env: dict[str, str]) -> RemoteConfig | None:
    host = env.get("ANNOTATION_REMOTE_HOST", "").strip()
    if not host:
        return None
    return RemoteConfig(
        host=host,
        repo_root=Path(env.get("ANNOTATION_REMOTE_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annotaion-jiahong-dev")),
        output_root=Path(env.get("ANNOTATION_REMOTE_OUTPUT_ROOT", "/home/zjh/data/v22_api_jobs")),
        upload_root=Path(env.get("ANNOTATION_REMOTE_UPLOAD_ROOT", "/home/zjh/data/v22_api_uploads")),
        package_root=Path(env.get("ANNOTATION_REMOTE_PACKAGE_ROOT", "/home/zjh/data/v22_api_downloads")),
        python=Path(env.get("ANNOTATION_REMOTE_PYTHON", "/home/zjh/ego-annotation-api-venv/bin/python")),
    )
