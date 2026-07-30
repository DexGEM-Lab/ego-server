#!/usr/bin/env python3
"""Build a timing ledger for one V22 single-item run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in {float("inf"), float("-inf")} else None


def iso_delta_s(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (b - a).total_seconds()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def video_duration(raw_manifest: dict[str, Any]) -> float | None:
    video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
    duration = finite(video.get("duration_s")) or finite(raw_manifest.get("duration_s"))
    if duration is not None and duration > 0:
        return duration
    fps = finite(raw_manifest.get("fps")) or finite(video.get("fps"))
    frame_count = finite(raw_manifest.get("frame_count")) or finite(video.get("frame_count"))
    if fps and fps > 0 and frame_count:
        return frame_count / fps
    return None


def step_by_name(pipeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = pipeline.get("steps") if isinstance(pipeline.get("steps"), list) else []
    return {str(row.get("step")): row for row in rows if isinstance(row, dict)}


def stage_elapsed(path: Path, *keys: str) -> float | None:
    payload = load_json_if_exists(path)
    if payload is None:
        return None
    for key in keys or ("elapsed_s",):
        value = finite(payload.get(key))
        if value is not None:
            return value
    return None


def build(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    pipeline = load_json(run_root / "annotation_pipeline_manifest.json")
    raw_manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    duration_s = video_duration(raw_manifest)
    steps = pipeline.get("steps") if isinstance(pipeline.get("steps"), list) else []
    steps_by_name = step_by_name(pipeline)

    intermediate_scripts = []
    for row in steps:
        if not isinstance(row, dict):
            continue
        elapsed = finite(row.get("elapsed_s"))
        item = {
            "segment_type": "intermediate_script",
            "name": row.get("step"),
            "status": row.get("status"),
            "elapsed_s": elapsed,
            "returncode": row.get("returncode"),
            "log": row.get("log"),
        }
        if elapsed and duration_s:
            item["input_video_realtime_x"] = duration_s / elapsed
        intermediate_scripts.append(item)

    atomic_specs = [
        ("D1_prepare_single_video", "prepare_single_video", None),
        ("D2_D3_unidepth_depth_intrinsics", "unidepth", run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "qc_unidepth_v2.json"),
        ("D2_calibration_contract", "calibration_contract", None),
        ("D6_wilor_visible_hand_evidence", "wilor_hands", run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_qc.json"),
        ("D4_droid_head_camera_trajectory", "camera_trajectory_droid", run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "v22_camera_trajectory_stage.json"),
        ("D5_hawor_metric_mano", "hawor_metric_hands", run_root / "measurements" / "hand_candidates" / "hawor_world" / "v22_hawor_metric_hand_stage.json"),
        ("D7_hybrid_temporal_hand_fusion", "hybrid_hand_fusion", run_root / "state" / "hands_metric" / "v22_hybrid_hand_fusion_stage.json"),
        ("D8_gt_free_drift_self_calibration", "gt_free_drift_self_calibration", run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json"),
        ("D9_hybrid_hand_overlay_render", "render_hybrid_hand_overlay", run_root / "renders" / "v22_hybrid_hand_overlay_report.json"),
        ("D9_world_head_hand_3d_render", "render_world_head_hand_3d", run_root / "renders" / "v22_world_head_hand_3d_report.json"),
        ("D9b_cosmos_caption_source", "cosmos_caption_source", run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"),
        ("D9b_caption_semantic_alignment", "captioning", run_root / "state" / "semantic_clips" / "v22_captioning_stage.json"),
        ("D9b_semantic_subtitle_render", "render_semantic_subtitle", run_root / "renders" / "v22_semantic_subtitle_report.json"),
        ("D11_evaluator_diagnostics", "evaluator", run_root / "evaluation" / "v22_evaluator_stage.json"),
        ("D10_self_consistency_qc", "self_consistency_qc", run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json"),
        ("product_annotation_bundle", "product_annotation_bundle", None),
    ]
    atomic_algorithms = []
    for name, step_name, artifact in atomic_specs:
        step = steps_by_name.get(step_name, {})
        artifact_elapsed = stage_elapsed(artifact) if artifact is not None else None
        elapsed = artifact_elapsed if artifact_elapsed is not None else finite(step.get("elapsed_s"))
        item = {
            "segment_type": "atomic_algorithm_or_required_stage",
            "name": name,
            "pipeline_step": step_name,
            "status": step.get("status"),
            "elapsed_s": elapsed,
            "elapsed_source": "stage_artifact" if artifact_elapsed is not None else "pipeline_step",
            "artifact": str(artifact) if artifact is not None else None,
        }
        if elapsed and duration_s:
            item["input_video_realtime_x"] = duration_s / elapsed
        atomic_algorithms.append(item)

    gpu_events = read_jsonl(run_root / "logs" / "gpu_wrapper_events.jsonl")
    gpu_usage_snapshots = read_jsonl(run_root / "logs" / "gpu_usage_snapshots.jsonl")
    launches: dict[str, list[dict[str, Any]]] = {}
    exits: dict[str, list[dict[str, Any]]] = {}
    gpu_checks = []
    for event in gpu_events:
        module = str(event.get("module_id") or "unknown")
        if event.get("event") == "launch":
            launches.setdefault(module, []).append(event)
        elif event.get("event") == "exit":
            exits.setdefault(module, []).append(event)
        elif event.get("event") == "gpu_check":
            gpu_checks.append(event)
    gpu_wrapper_segments = []
    for module, rows in launches.items():
        for i, launch in enumerate(rows):
            exit_event = exits.get(module, [None] * (i + 1))[i] if i < len(exits.get(module, [])) else None
            elapsed = iso_delta_s(launch.get("at"), exit_event.get("at") if isinstance(exit_event, dict) else None)
            gpu_wrapper_segments.append(
                {
                    "segment_type": "gpu_wrapper_child",
                    "module_id": module,
                    "gpu": launch.get("gpu"),
                    "request_mb": launch.get("request_mb"),
                    "elapsed_s": elapsed,
                    "returncode": exit_event.get("returncode") if isinstance(exit_event, dict) else None,
                    "launch_at": launch.get("at"),
                    "exit_at": exit_event.get("at") if isinstance(exit_event, dict) else None,
                }
            )

    review = load_json_if_exists(run_root / "state" / "agent_evidence_review.json")
    agent_reasoning_segments = []
    if review is not None:
        agent_reasoning_segments.append(
            {
                "segment_type": "agent_reasoning_or_review",
                "name": "single_item_delivery_evidence_review",
                "agent_id": review.get("agent_id"),
                "status": review.get("status"),
                "elapsed_s": finite(review.get("elapsed_s")),
                "source": str(run_root / "state" / "agent_evidence_review.json"),
                "claim_scope": review.get("claim_scope"),
            }
        )
    else:
        agent_reasoning_segments.append(
            {
                "segment_type": "agent_reasoning_or_review",
                "name": "single_item_delivery_evidence_review",
                "status": "not_run_at_ledger_build_time",
                "elapsed_s": None,
                "source": None,
            }
        )
    orchestration_elapsed = iso_delta_s(args.agent_session_start, args.agent_session_end)
    if orchestration_elapsed is not None:
        agent_reasoning_segments.append(
            {
                "segment_type": "agent_reasoning_or_review",
                "name": args.agent_session_label,
                "status": "observed_wall_clock_window",
                "elapsed_s": orchestration_elapsed,
                "start_at": args.agent_session_start,
                "end_at": args.agent_session_end,
                "source": "parent_agent_git_commit_window_and_task_ops",
                "claim_scope": "Upper-bound parent-agent orchestration wall-clock. Includes code edits, remote launch/monitoring, failure diagnosis, verification, packaging, and evidence recording; it is not a pure model-thinking timer.",
            }
        )

    def total(rows: list[dict[str, Any]]) -> float:
        return float(sum(v for row in rows for v in [finite(row.get("elapsed_s"))] if v is not None))

    ledger = {
        "schema": "v22.single_item_timing_ledger.v0",
        "status": "ok",
        "run_root": str(run_root),
        "input_video_duration_s": duration_s,
        "frame_count": int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or [])),
        "fps": finite(raw_manifest.get("fps")) or finite((raw_manifest.get("video") or {}).get("fps") if isinstance(raw_manifest.get("video"), dict) else None),
        "intermediate_scripts": intermediate_scripts,
        "atomic_algorithms": atomic_algorithms,
        "gpu_wrapper_segments": gpu_wrapper_segments,
        "gpu_check_events": gpu_checks,
        "gpu_usage_snapshots": gpu_usage_snapshots,
        "parallel_groups": pipeline.get("parallel_groups") if isinstance(pipeline.get("parallel_groups"), list) else [],
        "execution_topology": pipeline.get("execution_topology"),
        "agent_reasoning_segments": agent_reasoning_segments,
        "totals": {
            "intermediate_script_elapsed_s_sum": total(intermediate_scripts),
            "atomic_algorithm_elapsed_s_sum": total(atomic_algorithms),
            "gpu_wrapper_child_elapsed_s_sum": total(gpu_wrapper_segments),
            "agent_reasoning_elapsed_s_sum": total(agent_reasoning_segments),
        },
        "claim_scope": "Timing ledger over observed execution artifacts. Pipeline step elapsed_s is the authoritative wall-clock for scripts; atomic entries prefer stage-reported elapsed_s when available and otherwise use the wrapping script duration. Agent/review timing includes deterministic delivery review, plus an optional parent-agent orchestration wall-clock when start/end timestamps are supplied. No separate runtime LLM item-agent was launched in the single-item MVP path.",
    }
    write_json(args.output or (run_root / "state" / "timing_ledger.json"), ledger)
    print(json.dumps(ledger, indent=2, ensure_ascii=False))
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--agent-session-start", default=None)
    parser.add_argument("--agent-session-end", default=None)
    parser.add_argument("--agent-session-label", default="parent_agent_orchestration_wall_clock")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
