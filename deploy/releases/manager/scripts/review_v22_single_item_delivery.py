#!/usr/bin/env python3
"""Review one completed V22 single-item run for required delivery artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ffprobe_frame_count(path: Path) -> int | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def status_okish(value: Any, accepted: set[str]) -> bool:
    return str(value) in accepted


def review(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    raw_manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    frame_count = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    pipeline = load_json(run_root / "annotation_pipeline_manifest.json")
    renders = pipeline.get("renders") if isinstance(pipeline.get("renders"), dict) else {}
    d8 = load_json(run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json")
    d9b = load_json(run_root / "state" / "semantic_clips" / "v22_captioning_stage.json")
    d10 = load_json(run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json")
    d11 = load_json(run_root / "evaluation" / "v22_evaluator_stage.json")
    world_report = load_json(run_root / "renders" / "v22_world_head_hand_3d_report.json")
    subtitle_report = load_json(run_root / "renders" / "v22_semantic_subtitle_report.json")
    hybrid_report = load_json(run_root / "renders" / "v22_hybrid_hand_overlay_report.json")
    product_manifest_path = Path(str(pipeline.get("product_manifest_path") or "")) if pipeline.get("product_manifest_path") else None
    product_manifest = load_json(product_manifest_path) if product_manifest_path is not None and product_manifest_path.exists() else {}
    steps = pipeline.get("steps") if isinstance(pipeline.get("steps"), list) else []
    bad_steps = [row for row in steps if isinstance(row, dict) and str(row.get("status")) != "ok"]
    required_videos = {
        "hand_overlay": run_root / "renders" / "v22_overlay.mp4",
        "world_head_hand_3d": run_root / "renders" / "v22_world_head_hand_3d.mp4",
        "semantic_subtitle": run_root / "renders" / "v22_semantic_subtitle.mp4",
    }
    video_counts = {name: ffprobe_frame_count(path) for name, path in required_videos.items()}
    semantic_rows = d9b.get("semantic_rows") if isinstance(d9b.get("semantic_rows"), list) else []
    world_counts = world_report.get("draw_counts") if isinstance(world_report.get("draw_counts"), dict) else {}

    failures: list[str] = []
    if frame_count <= 0:
        failures.append("raw_frame_count_missing")
    if pipeline.get("status") != "ok":
        failures.append(f"pipeline_status_not_ok:{pipeline.get('status')}")
    if bad_steps:
        failures.append("pipeline_contains_failed_steps")
    if renders.get("overlay_source") != "hybrid_hand_state":
        failures.append(f"final_overlay_not_hybrid_state:{renders.get('overlay_source')}")
    if not status_okish(d8.get("status"), {"ok"}):
        failures.append(f"d8_not_ok:{d8.get('status')}")
    if not semantic_rows:
        failures.append(f"d9b_no_source_backed_semantic_rows:{d9b.get('status')}")
    if not status_okish(d10.get("status"), {"ok"}):
        failures.append(f"d10_not_ok:{d10.get('status')}")
    if not status_okish(d11.get("status"), {"ok", "no_gt_unmeasured"}):
        failures.append(f"d11_unexpected_status:{d11.get('status')}")
    if not product_manifest:
        failures.append("product_bundle_manifest_missing")
    elif not status_okish(product_manifest.get("status"), {"ok", "completed", "completed_with_degraded_outputs"}):
        failures.append(f"product_bundle_status_not_accepted:{product_manifest.get('status')}")
    for label, count in video_counts.items():
        if count != frame_count:
            failures.append(f"{label}_frame_count_mismatch:{count}!={frame_count}")
    if world_report.get("video_frame_count") != frame_count:
        failures.append(f"world_report_frame_count_mismatch:{world_report.get('video_frame_count')}!={frame_count}")
    if subtitle_report.get("video_frame_count") != frame_count:
        failures.append(f"subtitle_report_frame_count_mismatch:{subtitle_report.get('video_frame_count')}!={frame_count}")
    surface_frames = int(world_counts.get("left_surface_frames") or 0) + int(world_counts.get("right_surface_frames") or 0)
    if surface_frames <= 0:
        failures.append("world_render_no_mano_surface_frames")
    if subtitle_report.get("status") != "ok":
        failures.append(f"subtitle_not_source_backed:{subtitle_report.get('status')}")
    if args.package_path is not None and not args.package_path.exists():
        failures.append(f"package_missing:{args.package_path}")

    review = {
        "schema": "v22.single_item_agent_evidence_review.v0",
        "status": "clean_complete" if not failures else "failed_required_evidence",
        "agent_id": args.agent_id,
        "run_root": str(run_root),
        "claim_scope": "Deterministic agent/reviewer evidence pass over required D1-D11 artifacts and three full-length videos. It verifies delivery mechanics and source-backed semantics; it does not certify metric accuracy without GT.",
        "elapsed_s": float(time.time() - started),
        "observations": {
            "frame_count": frame_count,
            "pipeline_status": pipeline.get("status"),
            "bad_step_count": len(bad_steps),
            "overlay_source": renders.get("overlay_source"),
            "d8_status": d8.get("status"),
            "d9b_status": d9b.get("status"),
            "semantic_row_count": len(semantic_rows),
            "d10_status": d10.get("status"),
            "d11_status": d11.get("status"),
            "product_bundle_status": product_manifest.get("status"),
            "product_bundle_errors_count": product_manifest.get("errors_count"),
            "world_status": world_report.get("status"),
            "world_surface_frames": surface_frames,
            "subtitle_status": subtitle_report.get("status"),
            "subtitle_active_frame_count": subtitle_report.get("active_subtitle_frame_count"),
            "hybrid_overlay_status": hybrid_report.get("status"),
            "video_ffprobe_counts": video_counts,
            "package_path": str(args.package_path) if args.package_path is not None else None,
        },
        "failures": failures,
    }
    write_json(args.output or (run_root / "state" / "agent_evidence_review.json"), review)
    print(json.dumps(review, indent=2, ensure_ascii=False))
    if failures and args.fail_on_error:
        raise SystemExit(1)
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--agent-id", default="single_item_delivery_reviewer")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    review(parse_args())
