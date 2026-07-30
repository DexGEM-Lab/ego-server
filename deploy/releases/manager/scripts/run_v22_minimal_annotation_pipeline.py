#!/usr/bin/env python3
"""Run the minimal V22 annotation chain for one video and publish overlay outputs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hawor_peripheral_bundle import write_bundle_plan
from scripts.v22_gpu_usage import record_gpu_snapshot
from scripts.v22_model_request_helpers import write_droid_request, write_hawor_request, write_unidepth_request, write_wilor_request

HAMER_PYTHON = Path("/home/zjh/miniconda3/envs/hamer/bin/python")
EGO_PYTHON = Path("/home/zjh/miniconda3/envs/ego_foundation/bin/python")
EVENT_WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def run_step(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    events: Path,
    env: dict[str, str] | None = None,
    run_root: Path | None = None,
    model_request: Path | None = None,
    gpu_heavy: bool = False,
    parallel_group: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{name}.log"
    started_event = {"event": "step_started", "step": name, "at": utc_now(), "cmd": cmd}
    if parallel_group is not None:
        started_event["parallel_group"] = parallel_group
    if model_request is not None:
        started_event["model_request"] = str(model_request)
    append_event(events, started_event)
    gpu_before = record_gpu_snapshot(run_root=run_root, stage=name, phase="before", model_request=model_request) if gpu_heavy and run_root is not None else None
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    gpu_after = record_gpu_snapshot(run_root=run_root, stage=name, phase="after", model_request=model_request) if gpu_heavy and run_root is not None else None
    elapsed = time.time() - started
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    row = {"step": name, "status": "ok" if proc.returncode == 0 else "failed", "returncode": int(proc.returncode), "elapsed_s": elapsed, "log": str(log_path), "stdout_tail": tail}
    if parallel_group is not None:
        row["parallel_group"] = parallel_group
    if model_request is not None:
        row["model_request"] = str(model_request)
    if gpu_before is not None or gpu_after is not None:
        row["gpu_usage"] = {"before": gpu_before, "after": gpu_after, "snapshot_log": str((run_root or cwd) / "logs" / "gpu_usage_snapshots.jsonl")}
    append_event(events, {"event": "step_finished", "step": name, "at": utc_now(), **row})
    if proc.returncode != 0:
        raise RuntimeError(f"step_failed:{name}; log={log_path}\n{tail}")
    return row


def run_parallel_lanes(
    group_name: str,
    lanes: list[tuple[str, Callable[[], list[dict[str, Any]]]]],
    *,
    events: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not lanes:
        return [], {"group": group_name, "status": "skipped", "lanes": [], "elapsed_s": 0.0}
    started = time.time()
    append_event(events, {"event": "parallel_group_started", "parallel_group": group_name, "at": utc_now(), "lanes": [name for name, _ in lanes]})
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(lanes), thread_name_prefix=f"v22_{group_name}") as executor:
        future_to_lane = {executor.submit(fn): (idx, name) for idx, (name, fn) in enumerate(lanes)}
        for future in as_completed(future_to_lane):
            idx, name = future_to_lane[future]
            try:
                rows_by_index[idx] = future.result()
            except Exception as exc:
                errors.append({"lane": name, "error": str(exc)})
    elapsed = time.time() - started
    group = {"group": group_name, "status": "failed" if errors else "ok", "lanes": [name for name, _ in lanes], "elapsed_s": elapsed, "errors": errors}
    append_event(events, {"event": "parallel_group_finished", "parallel_group": group_name, "at": utc_now(), **group})
    if errors:
        detail = "; ".join(f"{row['lane']}: {row['error']}" for row in errors)
        raise RuntimeError(f"parallel_group_failed:{group_name}; {detail}")
    rows: list[dict[str, Any]] = []
    for idx in range(len(lanes)):
        rows.extend(rows_by_index.get(idx, []))
    return rows, group


def collect_future_rows(
    future: Future[list[dict[str, Any]]] | None,
    *,
    name: str,
    collected: set[Future[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    if future is None:
        return []
    if future in collected:
        return []
    rows = future.result()
    collected.add(future)
    return rows


def publish_overlay(run_root: Path) -> dict[str, Any]:
    renders = run_root / "renders"
    wilor_overlay = renders / "v22_wilor_hand_overlay.mp4"
    hybrid_overlay = renders / "v22_hybrid_hand_overlay.mp4"
    world_head_hand_3d = renders / "v22_world_head_hand_3d.mp4"
    semantic_subtitle = renders / "v22_semantic_subtitle.mp4"
    final_overlay = renders / "v22_overlay.mp4"
    source_overlay = hybrid_overlay if hybrid_overlay.exists() else wilor_overlay
    if not source_overlay.exists():
        raise RuntimeError(f"missing hand overlay: {source_overlay}")
    overlay_source = "hybrid_hand_state" if source_overlay == hybrid_overlay else "wilor_raw_candidates"
    if world_head_hand_3d.exists():
        proc = subprocess.run(
            [str(HAMER_PYTHON), "scripts/render_v22_primary_overlay.py", "--run-root", str(run_root), "--hand-overlay", str(source_overlay), "--world-video", str(world_head_hand_3d), "--output", str(final_overlay)],
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"primary overlay composition failed rc={proc.returncode}\n{proc.stdout[-4000:]}")
        overlay_source = f"two_pane_{overlay_source}_plus_3d_world"
    else:
        shutil.copy2(source_overlay, final_overlay)
    return {
        "v22_overlay": str(final_overlay),
        "overlay_source": overlay_source,
        "primary_overlay_report": str(renders / "v22_primary_overlay_report.json") if (renders / "v22_primary_overlay_report.json").exists() else None,
        "hand_overlay": str(wilor_overlay) if wilor_overlay.exists() else None,
        "hybrid_hand_overlay": str(hybrid_overlay) if hybrid_overlay.exists() else None,
        "world_head_hand_3d": str(world_head_hand_3d) if world_head_hand_3d.exists() else None,
        "semantic_subtitle": str(semantic_subtitle) if semantic_subtitle.exists() else None,
    }


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,duration,width,height", "-of", "json", str(path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return {"status": "failed", "stderr": proc.stderr[-2000:]}
    return {"status": "ok", "ffprobe": json.loads(proc.stdout)}


def validate_feishu_ray_stage_selection(args: argparse.Namespace) -> None:
    if args.model_execution != "feishu_ray":
        return
    if bool(getattr(args, "run_hawor_metric_hands", False)) and not bool(getattr(args, "run_camera_trajectory", False)):
        raise RuntimeError("feishu_ray_hawor_requires_run_camera_trajectory")
    if bool(getattr(args, "run_hybrid_hands", False)) and not bool(getattr(args, "run_hawor_metric_hands", False)):
        raise RuntimeError("feishu_ray_hybrid_requires_run_hawor_metric_hands")
    if getattr(args, "camera_backend", "droid") != "droid" and (
        bool(getattr(args, "run_camera_trajectory", False))
        or bool(getattr(args, "run_hawor_metric_hands", False))
    ):
        raise RuntimeError("feishu_ray_complete_path_requires_camera_backend_droid")
    if bool(getattr(args, "run_captioning", False)) and not bool(getattr(args, "skip_cosmos", False)):
        raise RuntimeError("feishu_ray_cosmos_disabled_use_skip_cosmos_or_omit_run_captioning")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_feishu_ray_stage_selection(args)
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    log_dir = run_root / "logs" / "minimal_pipeline"
    events = run_root / "logs" / "minimal_pipeline_events.jsonl"
    if run_root.exists() and not args.skip_prepare:
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    if args.skip_prepare:
        required_prepared = [run_root / "input" / "input_manifest.json", run_root / "input" / "raw_frame_manifest" / "manifest.json"]
        missing_prepared = [str(path) for path in required_prepared if not path.is_file()]
        if missing_prepared:
            raise FileNotFoundError(f"--skip-prepare requires prepared input artifacts: {missing_prepared}")
    log_dir.mkdir(parents=True, exist_ok=True)
    hawor_peripheral_profile = str(getattr(args, "hawor_peripheral_profile", "api_adapter"))
    hawor_peripheral_bundle = write_bundle_plan(run_root, hawor_peripheral_profile)
    env = os.environ.copy()
    env["V22_RUNTIME_CASE_ID"] = args.case_id
    env["V22_RUNTIME_RUN_ROOT"] = str(run_root)
    env.setdefault("V22_RUNTIME_AGENT_ID", os.environ.get("V22_RUNTIME_AGENT_ID", "single_item_pipeline"))
    if args.gpu_ids:
        env["V22_GPU_IDS"] = args.gpu_ids
    steps: list[dict[str, Any]] = []
    model_requests: dict[str, str] = {}
    parallel_groups: list[dict[str, Any]] = []

    if args.run_preflight:
        steps.append(run_step("preflight", [sys.executable, "scripts/preflight_v22_environment.py", "--output", str(run_root / "logs" / "preflight_v22_environment.json")], cwd=repo_root, log_dir=log_dir, events=events, env=env))

    if args.skip_prepare:
        steps.append({"step": "prepare_single_video", "status": "skipped_prepared_input", "run_root": str(run_root)})
    else:
        steps.append(run_step("prepare_single_video", [str(HAMER_PYTHON), "scripts/prepare_v22_single_video_run.py", "--case-id", args.case_id, "--input-video", str(args.input_video), "--run-root", str(run_root), *( ["--render-width", str(args.render_width)] if args.render_width is not None else [] ), *( ["--start-s", str(args.start_s)] if args.start_s is not None else [] ), *( ["--end-s", str(args.end_s)] if args.end_s is not None else [] )], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("depth_modality_report", [str(HAMER_PYTHON), "scripts/build_v21_depth_modality_report.py", "--input-manifest", str(run_root / "input" / "input_manifest.json"), "--output-report", str(run_root / "measurements" / "camera_depth" / "depth_modality_report.json")], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    unidepth_request = run_root / "requests" / "unidepth.json"
    write_unidepth_request(run_root)
    model_requests["unidepth"] = str(unidepth_request)
    wilor_request = run_root / "requests" / "wilor.json"
    write_wilor_request(run_root)
    model_requests["wilor"] = str(wilor_request)

    service_endpoints = {
        "unidepth": str(args.unidepth_service_url),
        "wilor": str(args.wilor_service_url),
        "hawor": str(args.hawor_service_url),
        "vggt": str(args.vggt_service_url),
    }

    def api_model_command(request_path: Path, model: str) -> list[str]:
        return [sys.executable, "-m", "services.model_client", "--request-json", str(request_path), "--endpoint", service_endpoints[model], "--timeout-s", str(float(args.service_timeout_s))]

    def feishu_stage_command(stage: str) -> list[str]:
        script = "scripts/run_feishu_ray_hawor_stage.py" if stage == "hawor" else "scripts/run_feishu_ray_annotation_stage.py"
        cmd = [
            str(HAMER_PYTHON),
            script,
            *([] if stage == "hawor" else [stage]),
            "--run-root",
            str(run_root),
            "--repo-root",
            str(repo_root),
            "--profile",
            str(args.feishu_service_profile),
            "--job-id",
            args.case_id,
            "--timeout-s",
            str(float(args.service_timeout_s)),
            "--retry-max-wait-s",
            str(float(getattr(args, "retry_max_wait_s", 0.0))),
            "--retry-initial-delay-s",
            str(float(getattr(args, "retry_initial_delay_s", 1.0))),
        ]
        overrides = {
            "unidepth": getattr(args, "feishu_unidepth_base_url", None),
            "wilor": getattr(args, "feishu_hands_wilor_base_url", None),
            "droid": getattr(args, "feishu_droid_base_url", None),
            "hawor": getattr(args, "feishu_hawor_base_url", None),
        }
        override = overrides.get(stage)
        if override:
            cmd.extend(["--base-url", str(override)])
        if stage == "hawor":
            cmd.extend(["--hawor-root", str(args.hawor_root)])
        return cmd

    initial_group_name = "initial_independent_lanes"
    post_calibration_group_name = "post_calibration_model_lanes"
    if args.model_execution == "api":
        execution_topology = "single_item_resident_api_unidepth_wilor_vggt_hawor_no_droid_fusion_scripted_cosmos_lanes"
    elif args.model_execution == "feishu_ray":
        execution_topology = "complete_non_cosmos_feishu_ray_D3_D6_then_D4_then_D5_D7_render_D8_QC_evaluator_bundle"
    else:
        execution_topology = "single_item_unidepth_gated_hawor_mask_preparation_then_one_mask_aware_shared_droid_then_hawor_adapter_fusion_scripted_cosmos_lanes"

    def unidepth_calibration_lane() -> list[dict[str, Any]]:
        if args.model_execution == "api":
            unidepth_cmd = api_model_command(unidepth_request, "unidepth")
        elif args.model_execution == "feishu_ray":
            unidepth_cmd = feishu_stage_command("unidepth")
        else:
            unidepth_cmd = [sys.executable, "scripts/v22_gpu_wrapper.py", "--module-id", "M02_S04_unidepth_v22_candidate", "--request-mb", "4096", "--log-jsonl", str(run_root / "logs" / "gpu_wrapper_events.jsonl"), "--", str(EGO_PYTHON), "scripts/run_v21_unidepth.py", "--run-root", str(run_root)]
        rows = [
            run_step(
                "unidepth",
                unidepth_cmd,
                cwd=repo_root,
                log_dir=log_dir,
                events=events,
                env=env,
                run_root=run_root,
                model_request=unidepth_request,
                gpu_heavy=args.model_execution == "script",
                parallel_group=initial_group_name,
            )
        ]
        rows.append(run_step("calibration_contract", [str(HAMER_PYTHON), "scripts/build_v19_calibration_contract.py", "--case", args.case_id, "--raw-frame-manifest", str(run_root / "input" / "raw_frame_manifest" / "manifest.json"), "--unidepth-npz", str(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"), "--output-dir", str(run_root / "state" / "calibration"), "--aggregation", "median", "--square-focal", "--center-principal-point"], cwd=repo_root, log_dir=log_dir, events=events, env=env))
        return rows

    def wilor_product_lane() -> list[dict[str, Any]]:
        if args.model_execution == "api":
            wilor_cmd = api_model_command(wilor_request, "wilor")
        elif args.model_execution == "feishu_ray":
            wilor_cmd = feishu_stage_command("wilor")
        else:
            wilor_cmd = [sys.executable, "scripts/v22_gpu_wrapper.py", "--module-id", "M04_S02_wilor_v22_hand_candidates", "--request-mb", "4096", "--log-jsonl", str(run_root / "logs" / "gpu_wrapper_events.jsonl"), "--", str(HAMER_PYTHON), "scripts/run_v21_wilor_hand_candidates.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--compute-target", "zjh@115.190.235.210:A800"]
        rows = [
            run_step(
                "wilor_hands",
                wilor_cmd,
                cwd=repo_root,
                log_dir=log_dir,
                events=events,
                env=env,
                run_root=run_root,
                model_request=wilor_request,
                gpu_heavy=args.model_execution == "script",
                parallel_group=initial_group_name,
            )
        ]
        rows.append(run_step("render_hand_overlay", [str(HAMER_PYTHON), "scripts/render_v22_wilor_hand_overlay.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--raw-hands", str(run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"), "--output", str(run_root / "renders" / "v22_wilor_hand_overlay.mp4"), "--review-dir", str(run_root / "renders" / "review_frames" / "wilor"), "--review-frames", "0", "90", "180", "270"], cwd=repo_root, log_dir=log_dir, events=events, env=env, parallel_group="wilor_product_lane"))
        return rows

    def caption_lane() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        caption_cmd = [str(HAMER_PYTHON), "scripts/run_v22_captioning_stage.py", "--run-root", str(run_root)]
        if args.actions_json is not None:
            caption_cmd.extend(["--actions-json", str(args.actions_json)])
        if args.captions_json is not None:
            caption_cmd.extend(["--captions-json", str(args.captions_json)])
        if args.semantic_review_json is not None:
            caption_cmd.extend(["--semantic-review-json", str(args.semantic_review_json)])
        elif args.actions_json is None and args.captions_json is None:
            cosmos_review = run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"
            rows.append(
                run_step(
                    "cosmos_caption_source",
                    [
                        str(HAMER_PYTHON),
                        "scripts/run_v22_cosmos_captioning_source.py",
                        "--video",
                        str(args.input_video),
                        "--run-root",
                        str(run_root),
                        "--case-id",
                        args.case_id,
                        "--output",
                        str(cosmos_review),
                    ],
                    cwd=repo_root,
                    log_dir=log_dir,
                    events=events,
                    env=env,
                    parallel_group=initial_group_name,
                )
            )
            caption_cmd.extend(["--semantic-review-json", str(cosmos_review)])
        rows.append(run_step("captioning", caption_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env, parallel_group=initial_group_name))
        rows.append(run_step("render_semantic_subtitle", [str(HAMER_PYTHON), "scripts/render_v22_semantic_subtitle_video.py", "--run-root", str(run_root), "--output", str(run_root / "renders" / "v22_semantic_subtitle.mp4")], cwd=repo_root, log_dir=log_dir, events=events, env=env, parallel_group=initial_group_name))
        return rows

    def build_camera_command(hawor_preparation_report: Path | None = None) -> tuple[Path, list[str]]:
        droid_request = run_root / "requests" / "droid.json"
        camera_output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
        calibration_contract = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
        if args.camera_backend == "droid":
            write_droid_request(run_root, output_dir=camera_output_dir, calibration_contract=calibration_contract)
        else:
            from scripts.v22_model_request_helpers import write_vggt_camera_request
            write_vggt_camera_request(run_root, output_dir=camera_output_dir, request_path=droid_request, calibration_contract=calibration_contract, backend=args.camera_backend)
        model_requests["droid"] = str(droid_request)
        if args.model_execution == "feishu_ray":
            return droid_request, feishu_stage_command("droid")
        camera_cmd = [
            str(HAMER_PYTHON),
            "scripts/run_v22_camera_trajectory_stage.py",
            "--run-root",
            str(run_root),
            "--repo-root",
            str(repo_root),
            "--camera-backend",
            args.camera_backend,
            "--runner-python",
            str(args.droid_python),
            "--droid-root",
            str(args.droid_root),
            "--droid-weights",
            str(args.droid_weights),
            "--vggt-python",
            str(args.vggt_python),
            "--vggt-device",
            str(args.vggt_device),
            "--vggt-target-size",
            str(args.vggt_target_size),
            "--vggt-patch-multiple",
            str(args.vggt_patch_multiple),
            "--vggt-batch-size",
            str(args.vggt_batch_size),
        ]
        if args.vggt_sequence_length is not None:
            camera_cmd.extend(["--vggt-sequence-length", str(args.vggt_sequence_length)])
        if args.vggt_checkpoint is not None:
            camera_cmd.extend(["--vggt-checkpoint", str(args.vggt_checkpoint.expanduser().resolve())])
        if args.vggt_allow_remote_model_download:
            camera_cmd.append("--vggt-allow-remote-model-download")
        if args.vggt_model_id:
            camera_cmd.extend(["--vggt-model-id", args.vggt_model_id])
        if args.vggt_model_file:
            camera_cmd.extend(["--vggt-model-file", args.vggt_model_file])
        if hawor_preparation_report is not None:
            camera_cmd.extend(["--hawor-preparation-report", str(hawor_preparation_report)])
        return droid_request, camera_cmd

    def build_hawor_command(*, prepare_only: bool = False) -> Path:
        hawor_request = run_root / "requests" / ("hawor_prepare.json" if prepare_only else "hawor.json")
        payload = write_hawor_request(run_root, calibration_contract=run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
        if prepare_only:
            payload["stage"] = "D5a_hawor_motion_preparation"
        write_json(hawor_request, payload)
        model_requests["hawor_prepare" if prepare_only else "hawor"] = str(hawor_request)
        return hawor_request

    caption_lane_enabled = bool(args.run_captioning and not getattr(args, "skip_cosmos", False))
    initial_lanes = ["D3_unidepth_to_D2_calibration", "D6_wilor_to_overlay"]
    if caption_lane_enabled:
        initial_lanes.append("D9b_caption_lane")
    initial_dependency_note = (
        "D3 calibration and D6 Feishu detector/WiLoR lanes run in parallel; the one Feishu DROID stage waits for both, then D5 consumes shared geometry."
        if args.model_execution == "feishu_ray"
        else "WiLoR and caption lanes remain independent; D2 calibration releases HaWoR mask preparation, which precedes the one canonical shared DROID when HaWoR is enabled."
    )
    append_event(events, {"event": "parallel_group_started", "parallel_group": initial_group_name, "at": utc_now(), "lanes": initial_lanes, "dependency_note": initial_dependency_note})
    initial_started = time.time()
    post_calibration_started: float | None = None
    post_calibration_lanes: list[str] = []
    collected_futures: set[Future[list[dict[str, Any]]]] = set()
    lane_finished_at: dict[str, float] = {}

    def tracked_lane(name: str, fn: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            return fn()
        finally:
            lane_finished_at[name] = time.time()

    if args.model_execution == "api":
        if args.camera_backend != "vggt":
            raise RuntimeError("API execution requires camera-backend=vggt; DROID is not a deployed service")
        if args.run_hawor_metric_hands and not args.run_camera_trajectory:
            raise RuntimeError("API HaWoR execution requires an upstream VGGT camera service stage")
        shared_droid_needed = False
    elif args.model_execution == "feishu_ray":
        shared_droid_needed = bool(args.run_camera_trajectory or args.run_hawor_metric_hands)
    else:
        shared_droid_needed = bool(args.run_camera_trajectory or args.run_hawor_metric_hands)
        if args.run_hawor_metric_hands and args.camera_backend != "droid":
            raise RuntimeError("legacy script HaWoR adapter requires --camera-backend droid")
    max_workers = max(1, len(initial_lanes) + int(shared_droid_needed) + int(args.run_hawor_metric_hands))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="v22_dependency_dag") as executor:
        unidepth_future = executor.submit(lambda: tracked_lane("D3_unidepth_to_D2_calibration", unidepth_calibration_lane))
        wilor_future = executor.submit(lambda: tracked_lane("D6_wilor_to_overlay", wilor_product_lane))
        caption_future = executor.submit(lambda: tracked_lane("D9b_caption_lane", caption_lane)) if caption_lane_enabled else None

        steps.extend(collect_future_rows(unidepth_future, name="D3_unidepth_to_D2_calibration", collected=collected_futures))
        if args.model_execution == "feishu_ray" and shared_droid_needed:
            steps.extend(collect_future_rows(wilor_future, name="D6_wilor_to_overlay", collected=collected_futures))

        droid_future: Future[list[dict[str, Any]]] | None = None
        hawor_future: Future[list[dict[str, Any]]] | None = None
        droid_lane_name = f"D4_{args.camera_backend}" if args.run_camera_trajectory else f"shared_{args.camera_backend}_geometry_for_hawor"
        if args.model_execution == "script" and args.run_hawor_metric_hands:
            post_calibration_lanes.append("D5a_hawor_mask_preparation")
        if shared_droid_needed:
            post_calibration_lanes.append(droid_lane_name)
        if args.run_hawor_metric_hands:
            post_calibration_lanes.append("D5b_hawor_adapter")
        if post_calibration_lanes:
            post_calibration_started = time.time()
            post_dependency_note = (
                "Feishu DROID starts only after D3 calibration and D6 complete; Feishu D5 adapts the shared manifest without DROID; D7 waits for D5 and D6."
                if args.model_execution == "feishu_ray"
                else "D2 releases HaWoR mask preparation; one mask-aware canonical DROID consumes that report; final HaWoR adapts the shared manifest without DROID; D7 waits for HaWoR and WiLoR."
            )
            append_event(events, {"event": "parallel_group_started", "parallel_group": post_calibration_group_name, "at": utc_now(), "lanes": post_calibration_lanes, "dependency_note": post_dependency_note})

        hawor_preparation_report: Path | None = None
        if args.model_execution == "script" and args.run_hawor_metric_hands:
            hawor_prepare_request = build_hawor_command(prepare_only=True)
            hawor_prepare_cmd = [str(HAMER_PYTHON), "scripts/run_v22_hawor_metric_hand_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--direct-export", "--prepare-motion-only", "--force-focal-cache-refresh", "--runner-python", str(args.hawor_python), "--hawor-root", str(args.hawor_root), "--checkpoint", str(args.hawor_checkpoint), "--infiller-weight", str(args.hawor_infiller), "--model-config", str(args.hawor_model_config)]
            steps.extend(tracked_lane("D5a_hawor_mask_preparation", lambda: [run_step("hawor_motion_preparation", hawor_prepare_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env, run_root=run_root, model_request=hawor_prepare_request, gpu_heavy=True, parallel_group=post_calibration_group_name)]))
            hawor_preparation_report = run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_motion_preparation.json"
            if not hawor_preparation_report.is_file():
                raise FileNotFoundError(f"HaWoR dynamic-mask preparation report missing before canonical DROID: {hawor_preparation_report}")
            preparation_payload = json.loads(hawor_preparation_report.read_text(encoding="utf-8"))
            preparation_mask = ((preparation_payload.get("artifacts") or {}).get("dynamic_mask") if isinstance(preparation_payload.get("artifacts"), dict) else None)
            if not isinstance(preparation_mask, dict) or not preparation_mask.get("path") or not preparation_mask.get("sha256"):
                raise RuntimeError(f"HaWoR preparation report lacks hash-bound dynamic mask: {hawor_preparation_report}")

        if args.model_execution == "api":
            droid_request = run_root / "requests" / "vggt_camera.json"
            camera_output_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
            calibration_contract = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
            from scripts.v22_model_request_helpers import write_vggt_camera_request
            write_vggt_camera_request(
                run_root,
                output_dir=camera_output_dir,
                request_path=droid_request,
                calibration_contract=calibration_contract,
                backend="vggt",
                parameters={"sequence_length": int(args.vggt_sequence_length), "batch_contract": {"sequence_length": int(args.vggt_sequence_length)}},
            )
            model_requests["vggt"] = str(droid_request)
            steps.append(run_step("camera_trajectory_vggt_api", api_model_command(droid_request, "vggt"), cwd=repo_root, log_dir=log_dir, events=events, env=env, run_root=run_root, model_request=droid_request, gpu_heavy=False, parallel_group=post_calibration_group_name))
            if args.run_hawor_metric_hands:
                camera_artifact = camera_output_dir / "droid_dense_trajectory.npz"
                if not camera_artifact.is_file():
                    raise FileNotFoundError(f"VGGT camera service did not produce the upstream camera artifact: {camera_artifact}")
                hawor_request = write_hawor_request(run_root, calibration_contract=calibration_contract, camera_artifact=camera_artifact)
                model_requests["hawor"] = str(run_root / "requests" / "hawor.json")
                steps.append(run_step("hawor_metric_hands_api", api_model_command(Path(str(hawor_request.get("request_path") or run_root / "requests" / "hawor.json")), "hawor"), cwd=repo_root, log_dir=log_dir, events=events, env=env, run_root=run_root, model_request=run_root / "requests" / "hawor.json", gpu_heavy=False, parallel_group=post_calibration_group_name))
        if shared_droid_needed:
            droid_request, camera_cmd = build_camera_command(hawor_preparation_report)
            droid_future = executor.submit(lambda: tracked_lane(droid_lane_name, lambda: [run_step(f"camera_trajectory_{args.camera_backend}", camera_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env, run_root=run_root, model_request=droid_request, gpu_heavy=args.model_execution == "script" and args.camera_backend != "contract", parallel_group=post_calibration_group_name)]))
        if args.model_execution != "api" and args.run_hawor_metric_hands:
            steps.extend(collect_future_rows(droid_future, name=droid_lane_name, collected=collected_futures))
            shared_manifest = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json"
            if not shared_manifest.is_file():
                raise FileNotFoundError(f"shared DROID manifest missing before HaWoR launch: {shared_manifest}")
            hawor_request = build_hawor_command()
            if args.model_execution == "feishu_ray":
                hawor_cmd = feishu_stage_command("hawor")
                hawor_gpu_heavy = False
            else:
                hawor_cmd = [str(HAMER_PYTHON), "scripts/run_v22_hawor_metric_hand_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--direct-export", "--runner-python", str(args.hawor_python), "--hawor-root", str(args.hawor_root), "--checkpoint", str(args.hawor_checkpoint), "--infiller-weight", str(args.hawor_infiller), "--model-config", str(args.hawor_model_config), "--droid-shared-manifest", str(shared_manifest)]
                hawor_gpu_heavy = True
            hawor_future = executor.submit(lambda: tracked_lane("D5b_hawor_adapter", lambda: [run_step("hawor_metric_hands", hawor_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env, run_root=run_root, model_request=hawor_request, gpu_heavy=hawor_gpu_heavy, parallel_group=post_calibration_group_name)]))

        if args.run_hybrid_hands:
            steps.extend(collect_future_rows(wilor_future, name="D6_wilor_to_overlay", collected=collected_futures))
            steps.extend(collect_future_rows(hawor_future, name="D5b_hawor_adapter", collected=collected_futures))
            steps.append(run_step("hybrid_hand_fusion", [str(HAMER_PYTHON), "scripts/run_v22_hybrid_hand_fusion_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
        elif args.run_gt_free_drift_self_calibration:
            steps.extend(collect_future_rows(wilor_future, name="D6_wilor_to_overlay", collected=collected_futures))
        if args.run_gt_free_drift_self_calibration:
            steps.append(run_step("gt_free_drift_self_calibration", [str(HAMER_PYTHON), "scripts/run_v22_gt_free_drift_self_calibration_stage.py", "--run-root", str(run_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
        if args.run_hybrid_hands:
            steps.append(run_step("render_hybrid_hand_overlay", [str(HAMER_PYTHON), "scripts/render_v22_hybrid_hand_overlay.py", "--run-root", str(run_root), "--output", str(run_root / "renders" / "v22_hybrid_hand_overlay.mp4")], cwd=repo_root, log_dir=log_dir, events=events, env=env))
            steps.append(run_step("render_world_head_hand_3d", [str(HAMER_PYTHON), "scripts/render_v22_world_head_hand_3d.py", "--run-root", str(run_root), "--output", str(run_root / "renders" / "v22_world_head_hand_3d.mp4")], cwd=repo_root, log_dir=log_dir, events=events, env=env))

        steps.extend(collect_future_rows(droid_future, name=droid_lane_name, collected=collected_futures))
        steps.extend(collect_future_rows(hawor_future, name="D5b_hawor_adapter", collected=collected_futures))
        steps.extend(collect_future_rows(wilor_future, name="D6_wilor_to_overlay", collected=collected_futures))
        steps.extend(collect_future_rows(caption_future, name="D9b_caption_lane", collected=collected_futures))

    initial_finished = max((lane_finished_at.get(name, time.time()) for name in initial_lanes), default=time.time())
    initial_group = {"group": initial_group_name, "status": "ok", "lanes": initial_lanes, "elapsed_s": float(initial_finished - initial_started), "dependency_note": initial_dependency_note}
    parallel_groups.append(initial_group)
    append_event(events, {"event": "parallel_group_finished", "parallel_group": initial_group_name, "at": utc_now(), **initial_group})
    if post_calibration_started is not None:
        post_calibration_finished = max((lane_finished_at.get(name, time.time()) for name in post_calibration_lanes), default=time.time())
        post_calibration_group = {"group": post_calibration_group_name, "status": "ok", "lanes": post_calibration_lanes, "elapsed_s": float(post_calibration_finished - post_calibration_started), "dependency_note": ("Feishu DROID waits for D3/D6; Feishu D5 adapts shared geometry without DROID; D7 waits for D5 and D6." if args.model_execution == "feishu_ray" else "D5a publishes hash-bound masks; one DROID invocation consumes them and produces shared geometry; D5b adapts that geometry without DROID; D7 fusion waits for D5b and WiLoR.")}
        parallel_groups.append(post_calibration_group)
        append_event(events, {"event": "parallel_group_finished", "parallel_group": post_calibration_group_name, "at": utc_now(), **post_calibration_group})

    overlays = publish_overlay(run_root)
    overlay_probe = ffprobe(Path(overlays["v22_overlay"]))
    stage_artifacts = {
        "gt_free_drift_self_calibration": str(run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json") if args.run_gt_free_drift_self_calibration else None,
        "captioning": str(run_root / "state" / "semantic_clips" / "v22_captioning_stage.json") if caption_lane_enabled else None,
        "self_consistency_qc": str(run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json") if args.run_self_consistency_qc else None,
        "evaluator": str(run_root / "evaluation" / "v22_evaluator_stage.json") if args.run_evaluator else None,
    }
    enabled_stages = {
        "camera_trajectory": bool(args.run_camera_trajectory),
        "hawor_metric_hands": bool(args.run_hawor_metric_hands),
        "hybrid_hands": bool(args.run_hybrid_hands),
        "gt_free_drift_self_calibration": bool(args.run_gt_free_drift_self_calibration),
        "captioning": caption_lane_enabled,
        "self_consistency_qc": bool(args.run_self_consistency_qc),
        "evaluator": bool(args.run_evaluator),
        "product_bundle": bool(args.write_product_bundle),
    }
    preliminary_summary = {"status": "ok", "method": "run_v22_minimal_annotation_pipeline", "case_id": args.case_id, "run_root": str(run_root), "steps": steps, "renders": overlays, "product_manifest_path": None, "enabled_stages": enabled_stages, "stage_artifacts": stage_artifacts, "model_requests": model_requests, "gpu_usage_snapshots": str(run_root / "logs" / "gpu_usage_snapshots.jsonl"), "parallel_groups": parallel_groups, "execution_topology": execution_topology, "ffprobe_overlay": overlay_probe, "hawor_peripheral_profile": hawor_peripheral_profile, "hawor_peripheral_bundle": str(hawor_peripheral_bundle)}
    write_json(run_root / "annotation_pipeline_manifest.json", preliminary_summary)
    if args.run_evaluator:
        evaluator_cmd = [str(HAMER_PYTHON), "scripts/run_v22_evaluator_stage.py", "--run-root", str(run_root)]
        if args.head_gt is not None:
            evaluator_cmd.extend(["--head-gt", str(args.head_gt)])
        if args.hand_gt is not None:
            evaluator_cmd.extend(["--hand-gt", str(args.hand_gt)])
        steps.append(run_step("evaluator", evaluator_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env))
    if args.run_self_consistency_qc:
        steps.append(run_step("self_consistency_qc", [str(HAMER_PYTHON), "scripts/run_v22_self_consistency_qc_stage.py", "--run-root", str(run_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    product_manifest_path = None
    if args.write_product_bundle:
        write_json(run_root / "annotation_pipeline_manifest.json", {**preliminary_summary, "steps": steps})
        product_job_id = f"{args.case_id}_product"
        steps.append(run_step("product_annotation_bundle", [str(HAMER_PYTHON), "scripts/adapt_v22_minimal_run_to_annotation_bundle.py", "--run-root", str(run_root), "--output-root", str(run_root / "product_bundle"), "--job-id", product_job_id], cwd=repo_root, log_dir=log_dir, events=events, env=env))
        product_manifest_path = str(run_root / "product_bundle" / product_job_id / "manifest.json")
    state = {
        "schema": "v22_minimal_annotation_state.v0",
        "status": "ok",
        "case_id": args.case_id,
        "input_video": str(args.input_video),
        "run_root": str(run_root),
        "render_boundary": "v22_overlay.mp4 is the primary two-pane display when D7/world render exist: left=image-space hand skeleton overlay, right=3D side-rear world head/hand render, with D9b subtitles. Without world render it falls back to the hand overlay only.",
        "measurements": {
            "raw_frame_manifest": str(run_root / "input" / "raw_frame_manifest" / "manifest.json"),
            "unidepth_npz": str(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"),
            "calibration_contract": str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"),
            "droid_shared_geometry": str(run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json"),
            "hawor_motion_preparation": str(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_motion_preparation.json") if args.model_execution == "script" and args.run_hawor_metric_hands else None,
            "hawor_world_hands": str(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz") if args.run_hawor_metric_hands else None,
            "hawor_service_adapter": str(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_slam_adapter_report.json") if args.model_execution == "feishu_ray" and args.run_hawor_metric_hands else None,
            "hawor_peripheral_bundle": str(hawor_peripheral_bundle),
            "wilor_raw_hands": str(run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"),
            "gt_free_drift_self_calibration": stage_artifacts["gt_free_drift_self_calibration"],
            "captioning": stage_artifacts["captioning"],
            "self_consistency_qc": stage_artifacts["self_consistency_qc"],
            "evaluator": stage_artifacts["evaluator"],
        },
        "renders": overlays,
        "limitations": [
            "WiLoR rows are raw MANO candidate evidence, not accepted optimized metric MANO state.",
            "D8 drift self-calibration is a GT-free correction hypothesis and does not certify fixed-gauge 3D accuracy.",
            "Object pose, contact, occlusion ownership, nonpenetration, and mesh reconstruction are intentionally unresolved in this minimal service path.",
        ],
    }
    write_json(run_root / "state" / "annotations_v22_renderable.json", state)
    summary = {"status": "ok", "method": "run_v22_minimal_annotation_pipeline", "case_id": args.case_id, "run_root": str(run_root), "steps": steps, "renders": overlays, "product_manifest_path": product_manifest_path, "enabled_stages": enabled_stages, "stage_artifacts": stage_artifacts, "model_requests": model_requests, "gpu_usage_snapshots": str(run_root / "logs" / "gpu_usage_snapshots.jsonl"), "parallel_groups": parallel_groups, "execution_topology": execution_topology, "ffprobe_overlay": overlay_probe, "hawor_peripheral_profile": hawor_peripheral_profile, "hawor_peripheral_bundle": str(hawor_peripheral_bundle)}
    write_json(run_root / "annotation_pipeline_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true", help="Use a run root already prepared by the full-dataset launcher.")
    parser.add_argument("--model-execution", choices=("script", "api", "feishu_ray"), default="script")
    parser.add_argument("--service-timeout-s", type=float, default=86400.0)
    parser.add_argument("--retry-max-wait-s", type=float, default=0.0, help="Maximum cumulative wait for explicit Feishu retryable responses; 0 means wait indefinitely")
    parser.add_argument("--retry-initial-delay-s", type=float, default=1.0)
    parser.add_argument(
        "--feishu-service-profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "feishu_ray_services.json",
    )
    parser.add_argument("--feishu-unidepth-base-url", default=None)
    parser.add_argument("--feishu-hands-wilor-base-url", default=None)
    parser.add_argument("--feishu-droid-base-url", default=None)
    parser.add_argument("--feishu-hawor-base-url", default=None)
    parser.add_argument("--unidepth-service-url", default="http://127.0.0.1:18101")
    parser.add_argument("--wilor-service-url", default="http://127.0.0.1:18102")
    parser.add_argument("--hawor-service-url", default="http://127.0.0.1:18103")
    parser.add_argument("--vggt-service-url", default="http://127.0.0.1:18104")
    parser.add_argument("--run-camera-trajectory", action="store_true")
    parser.add_argument("--camera-backend", choices=("droid", "vggt_omega", "vggt", "contract"), default="droid")
    parser.add_argument("--run-hawor-metric-hands", action="store_true")
    parser.add_argument("--hawor-peripheral-profile", choices=("api_adapter", "legacy_equivalent_reserved"), default="api_adapter", help="Planning profile only; legacy_equivalent_reserved never changes execution in this release.")
    parser.add_argument("--run-hybrid-hands", action="store_true")
    parser.add_argument("--run-gt-free-drift-self-calibration", action="store_true")
    parser.add_argument("--run-captioning", action="store_true")
    parser.add_argument("--skip-cosmos", action="store_true", help="Disable the Cosmos/caption/subtitle lane explicitly. Required for non-Cosmos Feishu Ray runs when --run-captioning is otherwise requested.")
    parser.add_argument("--run-self-consistency-qc", action="store_true")
    parser.add_argument("--run-evaluator", action="store_true")
    parser.add_argument("--actions-json", type=Path, default=None)
    parser.add_argument("--captions-json", type=Path, default=None)
    parser.add_argument("--semantic-review-json", type=Path, default=None)
    parser.add_argument("--head-gt", type=Path, default=None)
    parser.add_argument("--hand-gt", type=Path, default=None)
    parser.add_argument("--write-product-bundle", action="store_true")
    parser.add_argument("--droid-python", type=Path, default=Path("/home/zjh/miniconda3/envs/hawor/bin/python"))
    parser.add_argument("--droid-root", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM"))
    parser.add_argument("--droid-weights", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth"))
    parser.add_argument("--vggt-python", type=Path, default=Path("/home/zjh/ego_annotation_checkpoint/vggt/env/bin/python"))
    parser.add_argument("--vggt-device", default="cuda")
    parser.add_argument("--vggt-target-size", type=int, default=518)
    parser.add_argument("--vggt-patch-multiple", type=int, default=14)
    parser.add_argument("--vggt-batch-size", type=int, default=1)
    parser.add_argument("--vggt-sequence-length", type=int, default=32)
    parser.add_argument("--vggt-checkpoint", type=Path, default=Path("/home/zjh/ego_annotation_checkpoint/vggt/model.pt"))
    parser.add_argument("--vggt-allow-remote-model-download", action="store_true")
    parser.add_argument("--vggt-model-id", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-model-file", default="model.pt")
    parser.add_argument("--hawor-python", type=Path, default=Path("/home/zjh/miniconda3/envs/hawor/bin/python"))
    parser.add_argument("--hawor-root", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel/third_party/algorithms/hawor"))
    parser.add_argument("--hawor-checkpoint", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt"))
    parser.add_argument("--hawor-infiller", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt"))
    parser.add_argument("--hawor-model-config", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/model_config.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
