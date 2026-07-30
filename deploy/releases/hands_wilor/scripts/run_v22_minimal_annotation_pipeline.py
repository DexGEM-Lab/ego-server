#!/usr/bin/env python3
"""Run the minimal V22 annotation chain for one video and publish overlay outputs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HAMER_PYTHON = Path("/home/zjh/miniconda3/envs/hamer/bin/python")
EGO_PYTHON = Path("/home/zjh/miniconda3/envs/ego_foundation/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def run_step(name: str, cmd: list[str], *, cwd: Path, log_dir: Path, events: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{name}.log"
    append_event(events, {"event": "step_started", "step": name, "at": utc_now(), "cmd": cmd})
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    row = {"step": name, "status": "ok" if proc.returncode == 0 else "failed", "returncode": int(proc.returncode), "elapsed_s": elapsed, "log": str(log_path), "stdout_tail": tail}
    append_event(events, {"event": "step_finished", "step": name, "at": utc_now(), **row})
    if proc.returncode != 0:
        raise RuntimeError(f"step_failed:{name}; log={log_path}\n{tail}")
    return row


def publish_overlay(run_root: Path) -> dict[str, Any]:
    renders = run_root / "renders"
    wilor_overlay = renders / "v22_wilor_hand_overlay.mp4"
    hybrid_overlay = renders / "v22_hybrid_hand_overlay.mp4"
    depth_overlay = renders / "v22_depth_overlay.mp4"
    final_overlay = renders / "v22_overlay.mp4"
    source_overlay = hybrid_overlay if hybrid_overlay.exists() else wilor_overlay
    if not source_overlay.exists():
        raise RuntimeError(f"missing hand overlay: {source_overlay}")
    shutil.copy2(source_overlay, final_overlay)
    return {"v22_overlay": str(final_overlay), "overlay_source": "hybrid_hand_state" if source_overlay == hybrid_overlay else "wilor_raw_candidates", "hand_overlay": str(wilor_overlay) if wilor_overlay.exists() else None, "hybrid_hand_overlay": str(hybrid_overlay) if hybrid_overlay.exists() else None, "depth_overlay": str(depth_overlay) if depth_overlay.exists() else None}


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,duration,width,height", "-of", "json", str(path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return {"status": "failed", "stderr": proc.stderr[-2000:]}
    return {"status": "ok", "ffprobe": json.loads(proc.stdout)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    log_dir = run_root / "logs" / "minimal_pipeline"
    events = run_root / "logs" / "minimal_pipeline_events.jsonl"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.gpu_ids:
        env["V22_GPU_IDS"] = args.gpu_ids
    steps: list[dict[str, Any]] = []

    if args.run_preflight:
        steps.append(run_step("preflight", [sys.executable, "scripts/preflight_v22_environment.py", "--output", str(run_root / "logs" / "preflight_v22_environment.json")], cwd=repo_root, log_dir=log_dir, events=events, env=env))

    steps.append(run_step("prepare_single_video", [str(HAMER_PYTHON), "scripts/prepare_v22_single_video_run.py", "--case-id", args.case_id, "--input-video", str(args.input_video), "--run-root", str(run_root), *( ["--render-width", str(args.render_width)] if args.render_width is not None else [] ), *( ["--start-s", str(args.start_s)] if args.start_s is not None else [] ), *( ["--end-s", str(args.end_s)] if args.end_s is not None else [] )], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("depth_modality_report", [str(HAMER_PYTHON), "scripts/build_v21_depth_modality_report.py", "--input-manifest", str(run_root / "input" / "input_manifest.json"), "--output-report", str(run_root / "measurements" / "camera_depth" / "depth_modality_report.json")], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("unidepth", [sys.executable, "scripts/v22_gpu_wrapper.py", "--module-id", "M02_S04_unidepth_v22_candidate", "--request-mb", "4096", "--log-jsonl", str(run_root / "logs" / "gpu_wrapper_events.jsonl"), "--", str(EGO_PYTHON), "scripts/run_v21_unidepth.py", "--run-root", str(run_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("calibration_contract", [str(HAMER_PYTHON), "scripts/build_v19_calibration_contract.py", "--case", args.case_id, "--raw-frame-manifest", str(run_root / "input" / "raw_frame_manifest" / "manifest.json"), "--unidepth-npz", str(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"), "--output-dir", str(run_root / "state" / "calibration"), "--aggregation", "median", "--square-focal", "--center-principal-point"], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("wilor_hands", [sys.executable, "scripts/v22_gpu_wrapper.py", "--module-id", "M04_S02_wilor_v22_hand_candidates", "--request-mb", "4096", "--log-jsonl", str(run_root / "logs" / "gpu_wrapper_events.jsonl"), "--", str(HAMER_PYTHON), "scripts/run_v21_wilor_hand_candidates.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--compute-target", "zjh@115.190.235.210:A800"], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("render_hand_overlay", [str(HAMER_PYTHON), "scripts/render_v22_wilor_hand_overlay.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--raw-hands", str(run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"), "--output", str(run_root / "renders" / "v22_wilor_hand_overlay.mp4"), "--review-dir", str(run_root / "renders" / "review_frames" / "wilor"), "--review-frames", "0", "90", "180", "270"], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    steps.append(run_step("render_depth_overlay", [str(HAMER_PYTHON), "scripts/render_v22_depth_overlay.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--depth-npz", str(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"), "--output", str(run_root / "renders" / "v22_depth_overlay.mp4")], cwd=repo_root, log_dir=log_dir, events=events, env=env))

    if args.run_camera_trajectory:
        steps.append(run_step("camera_trajectory_droid", [str(HAMER_PYTHON), "scripts/run_v22_camera_trajectory_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--runner-python", str(args.droid_python), "--droid-root", str(args.droid_root), "--droid-weights", str(args.droid_weights)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    if args.run_hawor_metric_hands:
        steps.append(run_step("hawor_metric_hands", [str(HAMER_PYTHON), "scripts/run_v22_hawor_metric_hand_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root), "--direct-export", "--force-focal-cache-refresh", "--runner-python", str(args.hawor_python), "--hawor-root", str(args.hawor_root), "--checkpoint", str(args.hawor_checkpoint), "--infiller-weight", str(args.hawor_infiller), "--model-config", str(args.hawor_model_config)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    if args.run_hybrid_hands:
        steps.append(run_step("hybrid_hand_fusion", [str(HAMER_PYTHON), "scripts/run_v22_hybrid_hand_fusion_stage.py", "--run-root", str(run_root), "--repo-root", str(repo_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    if args.run_gt_free_drift_self_calibration:
        steps.append(run_step("gt_free_drift_self_calibration", [str(HAMER_PYTHON), "scripts/run_v22_gt_free_drift_self_calibration_stage.py", "--run-root", str(run_root)], cwd=repo_root, log_dir=log_dir, events=events, env=env))
    if args.run_hybrid_hands:
        steps.append(run_step("render_hybrid_hand_overlay", [str(HAMER_PYTHON), "scripts/render_v22_hybrid_hand_overlay.py", "--run-root", str(run_root), "--output", str(run_root / "renders" / "v22_hybrid_hand_overlay.mp4")], cwd=repo_root, log_dir=log_dir, events=events, env=env))

    overlays = publish_overlay(run_root)
    overlay_probe = ffprobe(Path(overlays["v22_overlay"]))
    stage_artifacts = {
        "gt_free_drift_self_calibration": str(run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json") if args.run_gt_free_drift_self_calibration else None,
        "captioning": str(run_root / "state" / "semantic_clips" / "v22_captioning_stage.json") if args.run_captioning else None,
        "self_consistency_qc": str(run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json") if args.run_self_consistency_qc else None,
        "evaluator": str(run_root / "evaluation" / "v22_evaluator_stage.json") if args.run_evaluator else None,
    }
    enabled_stages = {
        "camera_trajectory": bool(args.run_camera_trajectory),
        "hawor_metric_hands": bool(args.run_hawor_metric_hands),
        "hybrid_hands": bool(args.run_hybrid_hands),
        "gt_free_drift_self_calibration": bool(args.run_gt_free_drift_self_calibration),
        "captioning": bool(args.run_captioning),
        "self_consistency_qc": bool(args.run_self_consistency_qc),
        "evaluator": bool(args.run_evaluator),
        "product_bundle": bool(args.write_product_bundle),
    }
    preliminary_summary = {"status": "ok", "method": "run_v22_minimal_annotation_pipeline", "case_id": args.case_id, "run_root": str(run_root), "steps": steps, "renders": overlays, "product_manifest_path": None, "enabled_stages": enabled_stages, "stage_artifacts": stage_artifacts, "ffprobe_overlay": overlay_probe}
    write_json(run_root / "annotation_pipeline_manifest.json", preliminary_summary)
    if args.run_captioning:
        caption_cmd = [str(HAMER_PYTHON), "scripts/run_v22_captioning_stage.py", "--run-root", str(run_root)]
        if args.actions_json is not None:
            caption_cmd.extend(["--actions-json", str(args.actions_json)])
        if args.captions_json is not None:
            caption_cmd.extend(["--captions-json", str(args.captions_json)])
        steps.append(run_step("captioning", caption_cmd, cwd=repo_root, log_dir=log_dir, events=events, env=env))
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
        "render_boundary": "v22_overlay.mp4 is published from the hybrid hand-state overlay when D7 runs, otherwise from WiLoR raw candidate overlay; depth and calibration are supporting measurements.",
        "measurements": {
            "raw_frame_manifest": str(run_root / "input" / "raw_frame_manifest" / "manifest.json"),
            "unidepth_npz": str(run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"),
            "calibration_contract": str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"),
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
    summary = {"status": "ok", "method": "run_v22_minimal_annotation_pipeline", "case_id": args.case_id, "run_root": str(run_root), "steps": steps, "renders": overlays, "product_manifest_path": product_manifest_path, "enabled_stages": enabled_stages, "stage_artifacts": stage_artifacts, "ffprobe_overlay": overlay_probe}
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
    parser.add_argument("--run-camera-trajectory", action="store_true")
    parser.add_argument("--run-hawor-metric-hands", action="store_true")
    parser.add_argument("--run-hybrid-hands", action="store_true")
    parser.add_argument("--run-gt-free-drift-self-calibration", action="store_true")
    parser.add_argument("--run-captioning", action="store_true")
    parser.add_argument("--run-self-consistency-qc", action="store_true")
    parser.add_argument("--run-evaluator", action="store_true")
    parser.add_argument("--actions-json", type=Path, default=None)
    parser.add_argument("--captions-json", type=Path, default=None)
    parser.add_argument("--head-gt", type=Path, default=None)
    parser.add_argument("--hand-gt", type=Path, default=None)
    parser.add_argument("--write-product-bundle", action="store_true")
    parser.add_argument("--droid-python", type=Path, default=Path("/home/zjh/miniconda3/envs/hawor/bin/python"))
    parser.add_argument("--droid-root", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM"))
    parser.add_argument("--droid-weights", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth"))
    parser.add_argument("--hawor-python", type=Path, default=Path("/home/zjh/miniconda3/envs/hawor/bin/python"))
    parser.add_argument("--hawor-root", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR"))
    parser.add_argument("--hawor-checkpoint", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt"))
    parser.add_argument("--hawor-infiller", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt"))
    parser.add_argument("--hawor-model-config", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/model_config.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
