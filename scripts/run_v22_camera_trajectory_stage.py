#!/usr/bin/env python3
"""Run the V22 video-derived head/camera trajectory stage with DROID-SLAM.

This stage produces a first-class camera trajectory artifact for D4. It uses the
canonical V22 calibration contract to set DROID's focal prior scale, but the
result remains a video-derived trajectory with explicit gauge/scale uncertainty
unless external metric VIO/SLAM/IMU/fiducial anchors are supplied later.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def resolve_clip(run_root: Path, input_manifest: dict[str, Any]) -> Path:
    raw = input_manifest.get("primary_video") or input_manifest.get("clip_video")
    if not raw:
        raise RuntimeError("V22 input manifest lacks primary_video")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (run_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"prepared V22 clip missing: {path}")
    return path


def load_calibration_focal_scale(calibration_contract: Path, raw_manifest: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    contract = load_json(calibration_contract)
    values = contract.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise RuntimeError(f"canonical calibration lacks intrinsics_fx_fy_cx_cy: {calibration_contract}")
    fx = finite_positive(values[0])
    fy = finite_positive(values[1])
    if fx is None or fy is None:
        raise RuntimeError(f"invalid canonical focal values: {values}")
    video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
    width = finite_positive(video.get("width"))
    height = finite_positive(video.get("height"))
    if width is None or height is None:
        frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
        if frames and isinstance(frames[0], dict):
            width = finite_positive(frames[0].get("source_width") or frames[0].get("manifest_width"))
            height = finite_positive(frames[0].get("source_height") or frames[0].get("manifest_height"))
    if width is None or height is None:
        raise RuntimeError("raw frame manifest lacks image dimensions for DROID focal_scale derivation")
    focal = math.sqrt(fx * fy)
    return float(focal / max(width, height)), contract


def build_command(args: argparse.Namespace, clip: Path, output_dir: Path, focal_scale: float) -> list[str]:
    cmd = [
        str(args.runner_python),
        str(args.repo_root / "scripts" / "run_droid_full_frame.py"),
        "--clip",
        str(clip),
        "--output-dir",
        str(output_dir),
        "--droid-root",
        str(args.droid_root),
        "--weights",
        str(args.droid_weights),
        "--focal-scale",
        f"{focal_scale:.10f}",
        "--droid-area",
        str(int(args.droid_area)),
    ]
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(int(args.max_frames))])
    return cmd


def summarize_droid_qc(qc_path: Path, raw_manifest: dict[str, Any]) -> dict[str, Any]:
    if not qc_path.exists():
        raise FileNotFoundError(f"missing DROID QC: {qc_path}")
    qc = load_json(qc_path)
    expected = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    processed = int(qc.get("processed_frames") or 0)
    dense = int(qc.get("dense_trajectory_frames") or 0)
    status = "ok" if processed == dense and (processed == expected or not qc.get("full_source_timeline")) else "failed_frame_count_mismatch"
    return {
        "status": status,
        "processed_frames": processed,
        "dense_trajectory_frames": dense,
        "expected_raw_manifest_frames": expected,
        "full_source_timeline": qc.get("full_source_timeline"),
        "outputs": qc.get("outputs"),
        "trajectory_path_length": qc.get("trajectory_path_length"),
        "median_step": qc.get("median_step"),
        "p95_step": qc.get("p95_step"),
        "source_qc": qc,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    input_manifest = load_json(run_root / "input" / "input_manifest.json")
    raw_manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    raw_manifest = load_json(raw_manifest_path)
    calibration_contract = args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    focal_scale, calibration_payload = load_calibration_focal_scale(calibration_contract, raw_manifest)
    clip = resolve_clip(run_root, input_manifest)
    output_dir = (args.output_dir or (run_root / "measurements" / "camera_trajectory" / "droid_full_frame")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_v22_camera_trajectory_stage.log"
    cmd = build_command(args, clip, output_dir, focal_scale)
    stage: dict[str, Any] = {
        "schema": "v22_camera_trajectory_stage.v0",
        "status": "dry_run" if args.dry_run else "running",
        "run_root": str(run_root),
        "clip": str(clip),
        "calibration_contract": str(calibration_contract),
        "focal_scale_from_canonical_k": focal_scale,
        "calibration_source": calibration_payload.get("intrinsics_source") or calibration_payload.get("method"),
        "outputs": {
            "output_dir": str(output_dir),
            "dense_npz": str(output_dir / "droid_dense_trajectory.npz"),
            "dense_json": str(output_dir / "droid_dense_trajectory.json"),
            "qc_json": str(output_dir / "droid_qc.json"),
        },
        "command": cmd,
        "claim_scope": "D4 video-derived camera trajectory with explicit gauge/scale uncertainty; not fixed-gauge metric evaluation and not a substitute for external VIO/SLAM/IMU anchors.",
        "gauge_declaration": {
            "trajectory_frame": "DROID arbitrary world gauge",
            "scale_status": "video_derived_uncertain_without_external_metric_anchor",
            "metric_anchor_needed": "device VIO/SLAM/IMU, fiducial/mocap, or fixed-gauge benchmark GT",
        },
        "log": str(log_path),
    }
    stage_path = output_dir / "v22_camera_trajectory_stage.json"
    if args.dry_run:
        write_json(stage_path, stage)
        print(json.dumps(stage, indent=2))
        return stage
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    stage["returncode"] = int(proc.returncode)
    stage["elapsed_s"] = float(time.time() - started)
    stage["status"] = "ok" if proc.returncode == 0 else "failed"
    stage["stdout_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    if proc.returncode == 0:
        stage["droid_qc"] = summarize_droid_qc(output_dir / "droid_qc.json", raw_manifest)
        if stage["droid_qc"]["status"] != "ok":
            stage["status"] = stage["droid_qc"]["status"]
    write_json(stage_path, stage)
    print(json.dumps(stage, indent=2)[:12000])
    if stage["status"] != "ok":
        raise RuntimeError(f"camera trajectory stage failed; see {stage_path}")
    return stage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--droid-root", type=Path, default=Path("third_party/DROID-SLAM"))
    parser.add_argument("--droid-weights", type=Path, default=Path("third_party/DROID-SLAM/droid.pth"))
    parser.add_argument("--droid-area", type=int, default=384 * 512)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--runner-python", type=Path, default=Path(sys.executable), help="Python environment with DROID-SLAM runtime dependencies such as lietorch.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
