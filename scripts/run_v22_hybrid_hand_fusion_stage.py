#!/usr/bin/env python3
"""Run the V22 HaWoR+WiLoR hybrid hand fusion stage.

This wrapper requires three concrete inputs from the same V22 run root: raw WiLoR
candidate JSON, HaWoR metric MANO NPZ, and the canonical calibration contract.
It calls the existing V19 hybrid builder without allowing heuristic intrinsics.
"""
from __future__ import annotations

import argparse
import json
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


def resolve_measurement_path(run_root: Path, state: dict[str, Any], key: str, fallback: Path) -> Path:
    measurements = state.get("measurements") if isinstance(state.get("measurements"), dict) else {}
    value = measurements.get(key)
    if value:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (run_root / path).resolve()
    return fallback.resolve()


def existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def summarize_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"missing hybrid report: {report_path}")
    report = load_json(report_path)
    output_npz = Path(str((report.get("outputs") or {}).get("hybrid_npz") or ""))
    status = "ok" if output_npz.exists() else "failed_missing_hybrid_npz"
    return {
        "status": status,
        "report": report,
        "output_npz": str(output_npz) if str(output_npz) else None,
        "frame_count": report.get("frame_count"),
        "counts": report.get("counts"),
        "translation_policy": report.get("translation_policy"),
    }


def build_command(args: argparse.Namespace, wilor_raw: Path, hawor_npz: Path, calibration_contract: Path, output_npz: Path, report_json: Path) -> list[str]:
    return [
        sys.executable,
        str(args.repo_root / "scripts" / "build_v19_wilor_hawor_hybrid_hand_npz.py"),
        "--wilor-raw",
        str(wilor_raw),
        "--hawor-npz",
        str(hawor_npz),
        "--calibration-contract",
        str(calibration_contract),
        "--output-npz",
        str(output_npz),
        "--report-json",
        str(report_json),
        "--translation-policy",
        str(args.translation_policy),
        "--min-wilor-score",
        str(float(args.min_wilor_score)),
        "--max-fit-median-px",
        str(float(args.max_fit_median_px)),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    state_path = run_root / "state" / "annotations_v22_renderable.json"
    state = load_json(state_path) if state_path.exists() else {}
    wilor_raw = existing(args.wilor_raw or resolve_measurement_path(run_root, state, "wilor_raw_hands", run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"), "WiLoR raw candidates")
    hawor_npz = existing(args.hawor_npz or (run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz"), "HaWoR metric MANO NPZ")
    calibration_contract = existing(args.calibration_contract or resolve_measurement_path(run_root, state, "calibration_contract", run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"), "canonical calibration contract")
    output_dir = (args.output_dir or (run_root / "state" / "hands_metric")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "v22_hybrid_hands_metric.npz"
    report_json = output_dir / "v22_hybrid_hands_metric_report.json"
    log_path = output_dir / "run_v22_hybrid_hand_fusion_stage.log"
    cmd = build_command(args, wilor_raw, hawor_npz, calibration_contract, output_npz, report_json)
    stage: dict[str, Any] = {
        "schema": "v22_hybrid_hand_fusion_stage.v0",
        "status": "dry_run" if args.dry_run else "running",
        "run_root": str(run_root),
        "inputs": {
            "wilor_raw": str(wilor_raw),
            "hawor_npz": str(hawor_npz),
            "calibration_contract": str(calibration_contract),
        },
        "outputs": {"hybrid_npz": str(output_npz), "report_json": str(report_json)},
        "translation_policy": str(args.translation_policy),
        "command": cmd,
        "claim_scope": "D7 candidate hand fusion: HaWoR metric translation plus WiLoR visible geometry; not contact, occlusion ownership, nonpenetration, or GT-free self-calibration.",
        "log": str(log_path),
    }
    stage_path = output_dir / "v22_hybrid_hand_fusion_stage.json"
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
        stage["hybrid_report"] = summarize_report(report_json)
        if stage["hybrid_report"]["status"] != "ok":
            stage["status"] = stage["hybrid_report"]["status"]
    write_json(stage_path, stage)
    print(json.dumps(stage, indent=2)[:12000])
    if stage["status"] != "ok":
        raise RuntimeError(f"hybrid hand fusion stage failed; see {stage_path}")
    return stage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--wilor-raw", type=Path, default=None)
    parser.add_argument("--hawor-npz", type=Path, default=None)
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--translation-policy", choices=["wilor_metricfit", "hawor_wrist_aligned"], default="hawor_wrist_aligned")
    parser.add_argument("--min-wilor-score", type=float, default=0.30)
    parser.add_argument("--max-fit-median-px", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
