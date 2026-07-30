#!/usr/bin/env python3
"""Run the V22 HaWoR metric MANO hand stage for a prepared V22 run root.

This wrapper is an execution adapter around the existing HaWoR export contract.
It derives the required HaWoR focal from the canonical V22 calibration contract,
uses the prepared V22 clip as the only input video, and writes HaWoR outputs
under the V22 run root. It does not substitute WiLoR or any non-HaWoR source for
D5 metric MANO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
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


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_positive(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out) and out > 0.0:
        return out
    return None


def load_canonical_focal(calibration_contract: Path) -> tuple[float, dict[str, Any]]:
    contract = load_json(calibration_contract)
    values = contract.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise RuntimeError(f"calibration contract lacks intrinsics_fx_fy_cx_cy: {calibration_contract}")
    fx = finite_positive(values[0])
    fy = finite_positive(values[1])
    if fx is None or fy is None:
        raise RuntimeError(f"calibration contract has invalid focal values: {values}")
    return float(math.sqrt(fx * fy)), contract


def resolve_clip(input_manifest: dict[str, Any], run_root: Path) -> Path:
    raw = input_manifest.get("primary_video") or input_manifest.get("clip_video")
    if not raw:
        raise RuntimeError("V22 input manifest lacks primary_video")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (run_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"prepared V22 clip is missing: {path}")
    return path


def expected_frame_count(run_root: Path) -> int | None:
    raw_manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    if not raw_manifest_path.exists():
        return None
    raw_manifest = load_json(raw_manifest_path)
    count = raw_manifest.get("frame_count")
    if count is not None:
        return int(count)
    frames = raw_manifest.get("frames")
    return len(frames) if isinstance(frames, list) else None


def summarize_hawor_qc(qc_path: Path, expected_frames: int | None) -> dict[str, Any]:
    if not qc_path.exists():
        raise FileNotFoundError(f"HaWoR QC JSON missing: {qc_path}")
    qc = load_json(qc_path)
    frames = qc.get("frames")
    status = "ok"
    checks: list[dict[str, Any]] = []
    if expected_frames is not None:
        matches = int(frames) == int(expected_frames) if frames is not None else False
        checks.append({"check": "frame_count_matches_raw_manifest", "ok": matches, "expected": expected_frames, "actual": frames})
        if not matches:
            status = "failed_frame_count_mismatch"
    output_npz = Path(str(qc.get("output_npz") or qc_path.parent / "hawor_world_hands.npz"))
    checks.append({"check": "hawor_npz_exists", "ok": output_npz.exists(), "path": str(output_npz)})
    if not output_npz.exists():
        status = "failed_missing_hawor_npz"
    return {"status": status, "qc": qc, "checks": checks, "output_npz": str(output_npz)}


def build_command(args: argparse.Namespace, run_root: Path, clip: Path, focal: float, clip_hash: str, output_dir: Path) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update(
        {
            "EGO_HAWOR_ROOT": str(args.hawor_work_root),
            "EGO_HAWOR_CASE": str(args.case_id),
            "EGO_HAWOR_CLIP": str(clip),
            "EGO_HAWOR_CLIP_SHA256": clip_hash,
            "EGO_HAWOR_OUTPUT_DIR": str(output_dir),
            "EGO_HAWOR_IMG_FOCAL": f"{focal:.8f}",
            "EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH": "1" if args.force_focal_cache_refresh else "0",
        }
    )
    if args.direct_export:
        cmd = [
            str(args.runner_python),
            str(args.repo_root / "scripts" / "export_hawor_world.py"),
            "--hawor-root",
            str(args.hawor_root),
            "--video_path",
            str(clip),
            "--checkpoint",
            str(args.checkpoint),
            "--infiller_weight",
            str(args.infiller_weight),
            "--model_config",
            str(args.model_config),
            "--img_focal",
            f"{focal:.8f}",
            "--output-dir",
            str(output_dir),
        ]
        if args.force_focal_cache_refresh:
            cmd.append("--force-focal-cache-refresh")
    else:
        cmd = ["bash", str(args.repo_root / "scripts" / "remote_run_hawor_export.sh")]
    return cmd, env


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    input_manifest = load_json(run_root / "input" / "input_manifest.json")
    case_id = args.case_id or str(input_manifest.get("case_id") or run_root.name)
    args.case_id = case_id
    args.repo_root = repo_root
    clip = resolve_clip(input_manifest, run_root)
    calibration_contract = args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    focal, calibration_payload = load_canonical_focal(calibration_contract)
    clip_hash = sha256(clip)
    output_dir = args.output_dir or (run_root / "measurements" / "hand_candidates" / "hawor_world")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_hawor_metric_hand_stage.log"
    cmd, env = build_command(args, run_root, clip, focal, clip_hash, output_dir)
    env_view = {key: env[key] for key in sorted(env) if key.startswith("EGO_HAWOR_")}
    stage: dict[str, Any] = {
        "schema": "v22_hawor_metric_hand_stage.v0",
        "status": "dry_run" if args.dry_run else "running",
        "case_id": case_id,
        "run_root": str(run_root),
        "clip": str(clip),
        "clip_sha256": clip_hash,
        "calibration_contract": str(calibration_contract),
        "canonical_focal_px": focal,
        "calibration_source": calibration_payload.get("intrinsics_source") or calibration_payload.get("method"),
        "hawor_output_dir": str(output_dir),
        "command": cmd,
        "environment": env_view,
        "claim_scope": "D5 HaWoR metric MANO execution stage; no WiLoR substitution and no hybrid fusion acceptance.",
        "log": str(log_path),
    }
    stage_path = output_dir / "v22_hawor_metric_hand_stage.json"
    if args.dry_run:
        write_json(stage_path, stage)
        print(json.dumps(stage, indent=2))
        return stage
    if args.direct_export:
        for label, path in [("HaWoR root", args.hawor_root), ("HaWoR checkpoint", args.checkpoint), ("HaWoR infiller", args.infiller_weight), ("HaWoR model config", args.model_config), ("runner python", args.runner_python)]:
            if not Path(path).exists():
                raise FileNotFoundError(f"missing {label}: {path}")
    else:
        script_path = repo_root / "scripts" / "remote_run_hawor_export.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"missing HaWoR runner: {script_path}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    stage["returncode"] = int(proc.returncode)
    stage["elapsed_s"] = float(time.time() - started)
    stage["status"] = "ok" if proc.returncode == 0 else "failed"
    stage["stdout_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    if proc.returncode == 0:
        stage["hawor_qc"] = summarize_hawor_qc(output_dir / "qc_hawor_world_hands.json", expected_frame_count(run_root))
        if stage["hawor_qc"]["status"] != "ok":
            stage["status"] = stage["hawor_qc"]["status"]
    write_json(stage_path, stage)
    print(json.dumps(stage, indent=2))
    if stage["status"] != "ok":
        raise RuntimeError(f"HaWoR metric hand stage failed; see {stage_path}")
    return stage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--calibration-contract", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hawor-work-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/hawor_work"))
    parser.add_argument("--direct-export", action="store_true", help="Call export_hawor_world.py directly instead of remote_run_hawor_export.sh.")
    parser.add_argument("--runner-python", type=Path, default=Path("/home/zjh/miniconda3/envs/hawor/bin/python"))
    parser.add_argument("--hawor-root", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt"))
    parser.add_argument("--infiller-weight", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt"))
    parser.add_argument("--model-config", type=Path, default=Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/model_config.yaml"))
    parser.add_argument("--force-focal-cache-refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
