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


def build_pipeline_command(config: RemoteConfig, *, job_id: str, remote_video: Path, start_s: float | None, end_s: float | None, render_width: int | None, gpu_ids: str | None, run_preflight: bool, run_camera_trajectory: bool, run_hawor_metric_hands: bool, run_hybrid_hands: bool, run_gt_free_drift_self_calibration: bool, run_captioning: bool, run_self_consistency_qc: bool, run_evaluator: bool, actions_json: Path | None, captions_json: Path | None, semantic_review_json: Path | None, head_gt: Path | None, hand_gt: Path | None, write_product_bundle: bool, algorithm_inflight_multiplier: int = 2, model_backend: str = "script", service_profile: str | None = None, service_endpoints: dict[str, str] | None = None, diagnostic_monocular: bool = False) -> str:
    run_root = config.output_root / job_id
    if model_backend == "api_ify":
        parts = [
            q(config.python),
            "scripts/run_single_video_api.py",
            "--input", q(remote_video),
            "--case-id", q(job_id),
            "--run-root", q(run_root),
        ]
        if diagnostic_monocular:
            parts.append("--diagnostic-monocular")
        wrapper_parts = [
            q(config.python),
            "scripts/run_v22_api_job_with_admission.py",
            "--job-id", q(job_id),
            "--repo-root", q(config.repo_root),
            "--profile", q(config.repo_root / "configs" / "feishu_ray_services.json"),
            "--lock-root", q(config.output_root / "_algorithm_admission"),
            "--events-path", q(config.output_root / "_algorithm_admission_events.jsonl"),
            "--algorithm-inflight-multiplier", q(algorithm_inflight_multiplier),
            "--upstream-endpoints-json", q(json.dumps(service_endpoints or {}, sort_keys=True)),
            "--api-ify",
            "--",
            *parts,
        ]
        return f"cd {q(config.repo_root)} && PYTHONPATH={q(config.repo_root)}${{PYTHONPATH:+:$PYTHONPATH}} " + " ".join(wrapper_parts)
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
    if semantic_review_json is not None:
        parts.extend(["--semantic-review-json", q(semantic_review_json)])
    if head_gt is not None:
        parts.extend(["--head-gt", q(head_gt)])
    if hand_gt is not None:
        parts.extend(["--hand-gt", q(hand_gt)])
    if write_product_bundle:
        parts.append("--write-product-bundle")
    if model_backend == "feishu_ray":
        parts.extend(["--model-execution", "feishu_ray", "--feishu-service-profile", q(config.repo_root / "configs" / "feishu_ray_services.json")])
        endpoint_flags = {
            "unidepth": "--feishu-unidepth-base-url",
            "hands_wilor": "--feishu-hands-wilor-base-url",
            "droid": "--feishu-droid-base-url",
            "hawor": "--feishu-hawor-base-url",
        }
        for service, flag in endpoint_flags.items():
            if service_endpoints and service in service_endpoints:
                parts.extend([flag, q(service_endpoints[service])])
    wrapper_parts = [
        q(config.python),
        "scripts/run_v22_api_job_with_admission.py",
        "--job-id", q(job_id),
        "--repo-root", q(config.repo_root),
        "--profile", q(config.repo_root / "configs" / "feishu_ray_services.json"),
        "--lock-root", q(config.output_root / "_algorithm_admission"),
        "--events-path", q(config.output_root / "_algorithm_admission_events.jsonl"),
        "--algorithm-inflight-multiplier", q(algorithm_inflight_multiplier),
        "--upstream-endpoints-json", q(json.dumps(service_endpoints or {}, sort_keys=True)),
        "--",
        *parts,
    ]
    return f"cd {q(config.repo_root)} && " + " ".join(wrapper_parts)


def build_stage_batch_command(
    config: RemoteConfig,
    *,
    job_id: str,
    data_root: Path,
    batch_root: Path,
    max_items: int | None,
    video_list: Path | None,
    item_agents: int,
    gpu_count: int,
    gpu_ids: str | None,
    prepare_workers: int,
    calibration_workers: int,
    run_resident_droid: bool,
    run_resident_hawor: bool,
    report_only: bool = False,
    wave_size: int = 16,
    resident_worker_mode: str = "stdin-server",
) -> str:
    parts = [
        q(config.python),
        "scripts/run_v22_stage_batch_job.py",
        "--data-root", q(data_root),
        "--batch-root", q(batch_root),
        "--repo-root", q(config.repo_root),
        "--job-id", q(job_id),
        "--item-agents", q(item_agents),
        "--wave-size", q(wave_size),
        "--execution-topology", "wave-major",
        "--resident-worker-mode", q(resident_worker_mode),
        "--prepare-workers", q(prepare_workers),
        "--calibration-workers", q(calibration_workers),
        "--gpu-count", q(gpu_count),
        "--tmux-session", "ego_annotation",
        "--tmux-window-prefix", q(job_id[:24]),
    ]
    if max_items is not None:
        parts.extend(["--max-items", q(max_items)])
    if video_list is not None:
        parts.extend(["--video-list", q(video_list)])
    if gpu_ids:
        parts.extend(["--gpu-ids", q(gpu_ids)])
    if run_resident_droid:
        parts.append("--run-resident-droid")
    if run_resident_hawor:
        parts.append("--run-resident-hawor")
    if report_only:
        parts.append("--report-only")
    return f"cd {q(config.repo_root)} && " + " ".join(str(part) for part in parts)


def int_field(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def derive_job_set_completion(summary: dict[str, Any], resident: dict[str, Any], *, run_resident_droid: bool, run_resident_hawor: bool) -> dict[str, Any]:
    counts = summary.get("manifest_counts") if isinstance(summary.get("manifest_counts"), dict) else {}
    artifacts = summary.get("artifact_counts") if isinstance(summary.get("artifact_counts"), dict) else {}
    stage_satisfaction = resident.get("stage_satisfaction") if isinstance(resident.get("stage_satisfaction"), dict) else summary.get("stage_satisfaction")
    if not isinstance(stage_satisfaction, dict):
        stage_satisfaction = {}

    entry_count = int_field(summary, "entry_count")
    completed = int_field(counts, "completed")
    failed = int_field(counts, "failed")
    packages = int_field(artifacts, "packages_for_manifest_completed")
    required_artifacts = int_field(artifacts, "completed_with_required_artifacts")
    required_stage_statuses = {
        "unidepth_v2_depth_resident": "satisfied_true_resident_tensor_batch",
        "wilor_v21_hand_candidates_resident": "satisfied_true_resident_frame_and_crop_batch",
    }
    if run_resident_droid:
        required_stage_statuses["droid_camera_trajectory"] = "satisfied_resident_sequence_batch"
    if run_resident_hawor:
        required_stage_statuses["hawor_metric_hands"] = "partial_resident_submodels_complete_item_coverage"
    stage_results = {stage: stage_satisfaction.get(stage) for stage in required_stage_statuses}
    stages_satisfied = all(stage_results[stage] == expected for stage, expected in required_stage_statuses.items())
    items_complete = entry_count > 0 and completed == entry_count and packages == entry_count and required_artifacts == entry_count
    has_failed_entries = bool(summary.get("failed_entries")) or failed > 0
    status = "failed" if has_failed_entries else ("completed" if items_complete and stages_satisfied else "incomplete")
    return {
        "schema": "v22.job_set_completion_criteria.v0",
        "status": status,
        "entry_count": entry_count,
        "manifest_completed": completed,
        "manifest_failed": failed,
        "packages_for_manifest_completed": packages,
        "completed_with_required_artifacts": required_artifacts,
        "items_complete": items_complete,
        "required_stage_statuses": required_stage_statuses,
        "observed_stage_statuses": stage_results,
        "stages_satisfied": stages_satisfied,
        "status_note": "completed requires all items, required artifacts/packages, and requested resident stage evidence; command success alone is not completion evidence.",
    }


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
    semantic_review_json: Path | None,
    head_gt: Path | None,
    hand_gt: Path | None,
    write_product_bundle: bool,
    algorithm_inflight_multiplier: int = 2,
    model_backend: str = "script",
    service_profile: str | None = None,
    service_endpoints: dict[str, str] | None = None,
    diagnostic_monocular: bool = False,
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
    sidecars = {"actions_json": actions_json, "captions_json": captions_json, "semantic_review_json": semantic_review_json, "head_gt": head_gt, "hand_gt": hand_gt}
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
        semantic_review_json=remote_sidecars["semantic_review_json"],
        head_gt=remote_sidecars["head_gt"],
        hand_gt=remote_sidecars["hand_gt"],
        write_product_bundle=write_product_bundle,
        algorithm_inflight_multiplier=algorithm_inflight_multiplier,
        model_backend=model_backend,
        service_profile=service_profile,
        service_endpoints=service_endpoints,
        diagnostic_monocular=diagnostic_monocular,
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
        "final_report_path": package_json.get("final_report_path"),
        "final_video_path": package_json.get("final_video_path"),
        "summary": summary,
        "elapsed_s": float(time.time() - started),
    }


def run_remote_annotation_job_set(
    *,
    config: RemoteConfig,
    job_id: str,
    data_root: Path,
    local_package_root: Path,
    max_items: int | None,
    video_uris: list[str] | None = None,
    video_items: list[dict[str, Any]] | None = None,
    item_agents: int = 16,
    gpu_count: int,
    gpu_ids: str | None,
    prepare_workers: int,
    calibration_workers: int,
    run_resident_droid: bool,
    run_resident_hawor: bool,
    timeout: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    remote_run_root = config.output_root / job_id
    remote_upload_dir = config.upload_root / job_id
    ssh(config, f"mkdir -p {q(config.output_root)} {q(config.package_root)} {q(remote_upload_dir)}", timeout=60)
    video_list_path = None
    if video_items:
        video_list_path = remote_upload_dir / "selected_video_items.json"
        payload = json.dumps({"items": video_items}, ensure_ascii=False, indent=2) + "\n"
        ssh(config, f"printf %s {q(payload)} > {q(video_list_path)}", timeout=60)
    elif video_uris:
        video_list_path = remote_upload_dir / "selected_video_paths.txt"
        payload = "".join(f"{uri}\n" for uri in video_uris)
        ssh(config, f"printf %s {q(payload)} > {q(video_list_path)}", timeout=60)
    cmd = build_stage_batch_command(
        config,
        job_id=job_id,
        data_root=data_root,
        batch_root=remote_run_root,
        max_items=max_items,
        video_list=video_list_path,
        item_agents=item_agents,
        gpu_count=gpu_count,
        gpu_ids=gpu_ids,
        prepare_workers=prepare_workers,
        calibration_workers=calibration_workers,
        run_resident_droid=run_resident_droid,
        run_resident_hawor=run_resident_hawor,
    )
    ssh(config, cmd, timeout=timeout)
    refresh_cmd = build_stage_batch_command(
        config,
        job_id=job_id,
        data_root=data_root,
        batch_root=remote_run_root,
        max_items=max_items,
        video_list=video_list_path,
        item_agents=item_agents,
        gpu_count=gpu_count,
        gpu_ids=gpu_ids,
        prepare_workers=prepare_workers,
        calibration_workers=calibration_workers,
        run_resident_droid=run_resident_droid,
        run_resident_hawor=run_resident_hawor,
        report_only=True,
    )
    ssh(config, refresh_cmd, timeout=timeout)
    reports = remote_run_root / "reports"
    summary = remote_json(config, reports / "batch_summary.json", timeout=timeout)
    resident = remote_json(config, reports / "resident_model_summary.json", timeout=timeout)
    completion = derive_job_set_completion(summary, resident, run_resident_droid=run_resident_droid, run_resident_hawor=run_resident_hawor)
    summary = {**summary, "status": completion["status"], "completion_criteria": completion}
    delivery_zip = reports / "v22_stage_batch_delivery_index.zip"
    local_delivery_zip = local_package_root.expanduser().resolve() / f"{job_id}_delivery_index.zip"
    scp_from(config, str(delivery_zip), local_delivery_zip, timeout=timeout)
    return {
        "status": completion["status"],
        "job_id": job_id,
        "remote_run_root": str(remote_run_root),
        "remote_manifest_path": str(remote_run_root / "batch_manifest.json"),
        "remote_report_dir": str(reports),
        "remote_delivery_index_zip": str(delivery_zip),
        "local_delivery_index_zip": str(local_delivery_zip),
        "summary": summary,
        "resident_model_summary": resident,
        "completion_criteria": completion,
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
