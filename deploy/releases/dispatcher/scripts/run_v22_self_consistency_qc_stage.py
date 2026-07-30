#!/usr/bin/env python3
"""Recompute D10 self-consistency QC from frozen V22 run artifacts."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def ffprobe_video(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,duration,width,height", "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError as exc:
        return {"status": "ffprobe_unavailable", "path": str(path), "error": str(exc)}
    if proc.returncode != 0:
        return {"status": "failed", "path": str(path), "stderr": proc.stderr[-1000:]}
    return {"status": "ok", "path": str(path), "ffprobe": json.loads(proc.stdout)}


def stream0_frames(probe: dict[str, Any]) -> tuple[int | None, float | None, int | None, int | None]:
    streams = ((probe.get("ffprobe") or {}).get("streams") if isinstance(probe.get("ffprobe"), dict) else None)
    if not isinstance(streams, list) or not streams:
        return None, None, None, None
    stream = streams[0]
    if not isinstance(stream, dict):
        return None, None, None, None
    frames = finite_float(stream.get("nb_read_frames"))
    duration = finite_float(stream.get("duration"))
    width = finite_float(stream.get("width"))
    height = finite_float(stream.get("height"))
    return (int(frames) if frames is not None else None, duration, int(width) if width is not None else None, int(height) if height is not None else None)


def npz_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        blob = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "frame_idx" in blob.files:
        return int(np.asarray(blob["frame_idx"]).shape[0])
    return None


def hand_residuals(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    blob = np.load(path, allow_pickle=True)
    vals: list[float] = []
    for side in ("left", "right"):
        key = f"{side}_wilor_fit_reprojection_median_px"
        if key in blob.files:
            arr = np.asarray(blob[key], dtype=np.float64).reshape(-1)
            vals.extend(float(x) for x in arr if math.isfinite(float(x)))
    return {"status": "ok", "median_residual_px": summarize(vals)}


def row_count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    output = args.output or (run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json")
    raw_manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    frame_count = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    pipeline_manifest = load_json(run_root / "annotation_pipeline_manifest.json") if (run_root / "annotation_pipeline_manifest.json").exists() else {}
    renders = pipeline_manifest.get("renders") if isinstance(pipeline_manifest.get("renders"), dict) else {}
    overlay_path = Path(str(renders.get("v22_overlay") or run_root / "renders" / "v22_overlay.mp4"))
    overlay_probe = ffprobe_video(overlay_path)
    overlay_frames, overlay_duration, overlay_width, overlay_height = stream0_frames(overlay_probe)

    unidepth_npz = run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"
    wilor_raw_path = run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"
    wilor_frames = None
    wilor_hands = None
    if wilor_raw_path.exists():
        wilor = load_json(wilor_raw_path)
        frames = wilor.get("frames") if isinstance(wilor.get("frames"), list) else []
        wilor_frames = len(frames)
        wilor_hands = sum(len(row.get("raw_hands") or []) for row in frames if isinstance(row, dict))
    camera_json = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.json"
    camera_rows = None
    if camera_json.exists():
        camera_payload = load_json(camera_json)
        camera_rows = len(camera_payload.get("frames") or []) if isinstance(camera_payload.get("frames"), list) else None
    hawor_count = npz_frame_count(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz")
    hybrid_npz = run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz"
    hybrid_count = npz_frame_count(hybrid_npz)
    d8_path = run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json"
    d9b_path = run_root / "state" / "semantic_clips" / "v22_captioning_stage.json"
    d11_path = run_root / "evaluation" / "v22_evaluator_stage.json"
    d8 = load_json(d8_path) if d8_path.exists() else {"status": "missing"}
    d9b = load_json(d9b_path) if d9b_path.exists() else {"status": "missing"}
    d11 = load_json(d11_path) if d11_path.exists() else {"status": "missing"}

    checks = []
    def add(name: str, ok: bool | None, **extra: Any) -> None:
        checks.append({"check": name, "ok": ok, **extra})

    add("overlay_frame_count_matches_raw_manifest", overlay_frames == frame_count if overlay_frames is not None else False, expected=frame_count, actual=overlay_frames)
    add("wilor_frame_count_matches_raw_manifest", wilor_frames == frame_count if wilor_frames is not None else False, expected=frame_count, actual=wilor_frames)
    add("camera_rows_match_raw_manifest", camera_rows == frame_count if camera_rows is not None else False, expected=frame_count, actual=camera_rows)
    add("hawor_frame_count_matches_raw_manifest", hawor_count == frame_count if hawor_count is not None else False, expected=frame_count, actual=hawor_count)
    add("hybrid_frame_count_matches_raw_manifest", hybrid_count == frame_count if hybrid_count is not None else False, expected=frame_count, actual=hybrid_count)
    add("d8_stage_present", d8.get("status") != "missing", status=d8.get("status"))
    add("d9b_stage_present", d9b.get("status") != "missing", status=d9b.get("status"))
    add("d11_stage_present", d11.get("status") != "missing", status=d11.get("status"))

    hard_failures = [row for row in checks if row.get("ok") is False and row["check"] in {"overlay_frame_count_matches_raw_manifest", "wilor_frame_count_matches_raw_manifest"}]
    degraded = [row for row in checks if row.get("ok") is False and row not in hard_failures]
    status = "failed" if hard_failures else "degraded" if degraded else "ok"
    payload = {
        "schema": "v22_full_self_consistency_qc.v0",
        "status": status,
        "method": "recompute_consistency_from_frozen_v22_artifacts",
        "run_root": str(run_root),
        "frame_count_raw_manifest": frame_count,
        "overlay": {
            "path": str(overlay_path),
            "frames": overlay_frames,
            "duration_s": overlay_duration,
            "width": overlay_width,
            "height": overlay_height,
        },
        "streams": {
            "unidepth_npz_exists": unidepth_npz.exists(),
            "wilor_frames": wilor_frames,
            "wilor_hands": wilor_hands,
            "camera_rows": camera_rows,
            "hawor_frames": hawor_count,
            "hybrid_frames": hybrid_count,
        },
        "hand_projection_residuals": hand_residuals(hybrid_npz),
        "d8_gt_free_drift_self_calibration": {"path": str(d8_path), "status": d8.get("status"), "summary": d8.get("summary")},
        "d9b_captioning": {"path": str(d9b_path), "status": d9b.get("status"), "summary": d9b.get("summary")},
        "d11_evaluator": {"path": str(d11_path), "status": d11.get("status"), "summary": d11.get("summary")},
        "checks": checks,
        "claim_scope": "D10 self-consistency QC over frozen prediction artifacts. This checks internal agreement and does not replace fixed-gauge evaluator metrics.",
        "elapsed_s": float(time.time() - started),
    }
    write_json(output, payload)
    print(json.dumps({"status": status, "checks": len(checks), "output": str(output)}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
