#!/usr/bin/env python3
"""Run the V22 video-derived head/camera trajectory stage.

DROID remains the default backend. VGGT/Omega-compatible backends route through
the resident tensor-batch camera worker and write the same D4-compatible output
paths so downstream V22 consumers can keep reading camera trajectory artifacts.
All backends remain video-derived trajectories with explicit gauge/scale
uncertainty unless external metric VIO/SLAM/IMU/fiducial anchors are supplied.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.v22_model_request_helpers import write_droid_request


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


def load_hawor_preparation(path: Path, *, expected_frames: int, expected_clip: Path) -> tuple[Path, str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("schema") != "v22_hawor_motion_preparation.v0" or payload.get("status") != "ok":
        raise RuntimeError(f"invalid HaWoR motion preparation report: {path}")
    timeline = payload.get("timeline") if isinstance(payload.get("timeline"), dict) else {}
    if int(timeline.get("frame_count") or -1) != int(expected_frames):
        raise RuntimeError(f"HaWoR preparation frame count does not match raw timeline: {timeline.get('frame_count')} != {expected_frames}")
    video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
    if Path(str(video.get("path", ""))).resolve() != expected_clip.resolve():
        raise RuntimeError(f"HaWoR preparation clip does not match camera input: {video.get('path')} != {expected_clip}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    dynamic_mask = artifacts.get("dynamic_mask") if isinstance(artifacts.get("dynamic_mask"), dict) else None
    if dynamic_mask is None or not dynamic_mask.get("path") or not dynamic_mask.get("sha256"):
        raise RuntimeError(f"HaWoR preparation lacks hash-bound dynamic mask: {path}")
    mask_path = Path(str(dynamic_mask["path"])).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"HaWoR preparation dynamic mask is missing: {mask_path}")
    return mask_path, str(dynamic_mask["sha256"]), payload


def build_command(args: argparse.Namespace, clip: Path, output_dir: Path, focal_scale: float, dynamic_mask_path: Path | None = None, dynamic_mask_sha256: str | None = None) -> list[str]:
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
    if dynamic_mask_path is not None:
        cmd.extend(["--dynamic-mask-npy", str(dynamic_mask_path)])
        if dynamic_mask_sha256 is None:
            raise RuntimeError("dynamic mask path requires a preparation-report hash")
        cmd.extend(["--dynamic-mask-sha256", str(dynamic_mask_sha256)])
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
        "shared_geometry": qc.get("shared_geometry"),
        "droid_invocation": qc.get("droid_invocation"),
        "trajectory_path_length": qc.get("trajectory_path_length"),
        "median_step": qc.get("median_step"),
        "p95_step": qc.get("p95_step"),
        "source_qc": qc,
    }


def raw_frame_count(raw_manifest: dict[str, Any]) -> int:
    return int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))


def sampled_timeline_frame_indices(raw_manifest: dict[str, Any], sequence_length: int) -> list[int] | None:
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    total = len(frames)
    if sequence_length <= 0 or total <= 0 or sequence_length >= total:
        return None
    if sequence_length == 1:
        row = frames[0]
        return [int(row.get("frame_idx", 0))] if isinstance(row, dict) else [0]
    positions = [int(round(i * float(total - 1) / float(sequence_length - 1))) for i in range(sequence_length)]
    frame_indices: list[int] = []
    for pos in positions:
        row = frames[min(max(pos, 0), total - 1)]
        frame_indices.append(int(row.get("frame_idx", pos)) if isinstance(row, dict) else int(pos))
    if len(set(frame_indices)) != len(frame_indices):
        raise RuntimeError("VGGT timeline sampling produced duplicate frame indices")
    return frame_indices


def build_vggt_worker_request(
    args: argparse.Namespace,
    *,
    run_root: Path,
    output_dir: Path,
    calibration_contract: Path,
    raw_manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    sequence_length = int(args.vggt_sequence_length or raw_frame_count(raw_manifest))
    if sequence_length <= 0:
        raise RuntimeError("VGGT/Omega camera backend requires a positive sequence length")
    request_path = run_root / "requests" / "vggt_camera_batch.json"
    input_manifest = load_json(run_root / "input" / "input_manifest.json")
    item_id = str(input_manifest.get("case_id") or run_root.name)
    item_payload: dict[str, Any] = {
        "item_id": item_id,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
    }
    frame_indices = sampled_timeline_frame_indices(raw_manifest, sequence_length)
    if frame_indices is not None:
        item_payload["frame_indices"] = frame_indices
        item_payload["sampling_policy"] = "linspace_full_timeline"
    payload: dict[str, Any] = {
        "job_id": item_id,
        "backend": args.camera_backend,
        "worker_id": f"{args.camera_backend}_camera_stage_worker_000",
        "stage_id": f"{args.camera_backend}_camera_geometry_stage",
        "batch_size": int(args.vggt_batch_size),
        "sequence_length": sequence_length,
        "target_size": int(args.vggt_target_size),
        "patch_multiple": int(args.vggt_patch_multiple),
        "device": str(args.vggt_device),
        "output_root": str(output_dir),
        "compat_request_name": "droid",
        "calibration_contract": str(calibration_contract),
        "items": [item_payload],
    }
    if args.vggt_checkpoint is not None:
        payload["checkpoint"] = str(args.vggt_checkpoint.expanduser().resolve())
    if args.vggt_allow_remote_model_download:
        payload["allow_remote_model_download"] = True
    if args.vggt_model_id:
        payload["model_id"] = args.vggt_model_id
    if args.vggt_model_file:
        payload["model_file"] = args.vggt_model_file
    write_json(request_path, payload)
    return request_path, payload


def build_vggt_command(args: argparse.Namespace, request_path: Path) -> list[str]:
    cmd = [
        str(args.vggt_python),
        str(args.repo_root / "scripts" / "run_v22_resident_vggt_camera_batch.py"),
        "--request",
        str(request_path),
        "--backend",
        args.camera_backend,
        "--device",
        str(args.vggt_device),
        "--batch-size",
        str(int(args.vggt_batch_size)),
        "--target-size",
        str(int(args.vggt_target_size)),
        "--patch-multiple",
        str(int(args.vggt_patch_multiple)),
    ]
    if args.vggt_checkpoint is not None:
        cmd.extend(["--checkpoint", str(args.vggt_checkpoint.expanduser().resolve())])
    if args.vggt_allow_remote_model_download:
        cmd.append("--allow-remote-model-download")
    if args.vggt_model_id:
        cmd.extend(["--model-id", args.vggt_model_id])
    if args.vggt_model_file:
        cmd.extend(["--model-file", args.vggt_model_file])
    return cmd


def run_vggt_camera_stage(
    args: argparse.Namespace,
    *,
    started: float,
    run_root: Path,
    repo_root: Path,
    clip: Path,
    output_dir: Path,
    calibration_contract: Path,
    raw_manifest: dict[str, Any],
) -> dict[str, Any]:
    from scripts.v22_model_request_helpers import write_vggt_camera_request

    model_request = write_vggt_camera_request(
        run_root,
        output_dir=output_dir,
        request_path=run_root / "requests" / "droid.json",
        calibration_contract=calibration_contract,
        backend=args.camera_backend,
    )
    worker_request_path, worker_request = build_vggt_worker_request(
        args,
        run_root=run_root,
        output_dir=output_dir,
        calibration_contract=calibration_contract,
        raw_manifest=raw_manifest,
    )
    cmd = build_vggt_command(args, worker_request_path)
    log_path = output_dir / "run_v22_camera_trajectory_stage.log"
    stage_path = output_dir / "v22_camera_trajectory_stage.json"
    dry_stage: dict[str, Any] = {
        "schema": "v22_camera_trajectory_stage.v0",
        "status": "dry_run" if args.dry_run else "running",
        "run_root": str(run_root),
        "clip": str(clip),
        "calibration_contract": str(calibration_contract),
        "camera_backend": args.camera_backend,
        "replacement_for": "D4_droid_head_camera_trajectory",
        "outputs": {
            "output_dir": str(output_dir),
            "dense_npz": str(output_dir / "droid_dense_trajectory.npz"),
            "dense_json": str(output_dir / "droid_dense_trajectory.json"),
            "qc_json": str(output_dir / "droid_qc.json"),
        },
        "command": cmd,
        "model_request": str(run_root / "requests" / "droid.json"),
        "model_request_payload": model_request,
        "worker_request": str(worker_request_path),
        "worker_request_payload": worker_request,
        "claim_scope": "D4 camera trajectory candidate from a VGGT/Omega-style resident tensor-batch worker; video-derived uncertain gauge, not fixed-gauge metric accuracy.",
        "gauge_declaration": {
            "trajectory_frame": "VGGT/Omega local anchor world gauge",
            "scale_status": "video_derived_uncertain_without_external_metric_anchor",
            "metric_anchor_needed": "device VIO/SLAM/IMU, fiducial/mocap, known-size scene/object measurement, or benchmark GT",
        },
        "log": str(log_path),
    }
    if args.dry_run:
        write_json(stage_path, dry_stage)
        print(json.dumps(dry_stage, indent=2))
        return dry_stage
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        failed = {
            **dry_stage,
            "status": "failed",
            "returncode": int(proc.returncode),
            "elapsed_s": float(time.time() - started),
            "stdout_tail": log_path.read_text(encoding="utf-8", errors="replace")[-6000:],
        }
        write_json(stage_path, failed)
        raise RuntimeError(f"VGGT/Omega camera trajectory stage failed; see {stage_path}")
    stage = load_json(stage_path)
    stage["wrapper_command"] = cmd
    stage["wrapper_returncode"] = int(proc.returncode)
    stage["wrapper_elapsed_s"] = float(time.time() - started)
    stage["wrapper_log"] = str(log_path)
    write_json(stage_path, stage)
    print(json.dumps(stage, indent=2)[:12000])
    return stage


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    run_root = args.run_root.resolve()
    repo_root = args.repo_root.resolve()
    input_manifest = load_json(run_root / "input" / "input_manifest.json")
    raw_manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
    raw_manifest = load_json(raw_manifest_path)
    calibration_contract = args.calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    clip = resolve_clip(run_root, input_manifest)
    output_dir = (args.output_dir or (run_root / "measurements" / "camera_trajectory" / "droid_full_frame")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.camera_backend != "droid":
        return run_vggt_camera_stage(
            args,
            started=started,
            run_root=run_root,
            repo_root=repo_root,
            clip=clip,
            output_dir=output_dir,
            calibration_contract=calibration_contract,
            raw_manifest=raw_manifest,
        )
    focal_scale, calibration_payload = load_calibration_focal_scale(calibration_contract, raw_manifest)
    dynamic_mask_path: Path | None = None
    dynamic_mask_sha256: str | None = None
    preparation_payload: dict[str, Any] | None = None
    if args.hawor_preparation_report is not None:
        dynamic_mask_path, dynamic_mask_sha256, preparation_payload = load_hawor_preparation(
            args.hawor_preparation_report.expanduser().resolve(),
            expected_frames=raw_frame_count(raw_manifest),
            expected_clip=clip,
        )
    if args.camera_backend != "droid" and dynamic_mask_path is not None:
        raise RuntimeError("HaWoR dynamic masks can only be consumed by the DROID camera backend")
    model_request = write_droid_request(run_root, output_dir=output_dir, calibration_contract=calibration_contract)
    if dynamic_mask_path is not None:
        model_request["dynamic_mask"] = {
            "preparation_report": str(args.hawor_preparation_report.expanduser().resolve()),
            "path": str(dynamic_mask_path),
            "sha256": dynamic_mask_sha256,
            "required": True,
        }
        write_json(run_root / "requests" / "droid.json", model_request)
    log_path = output_dir / "run_v22_camera_trajectory_stage.log"
    cmd = build_command(args, clip, output_dir, focal_scale, dynamic_mask_path, dynamic_mask_sha256)
    stage: dict[str, Any] = {
        "schema": "v22_camera_trajectory_stage.v0",
        "status": "dry_run" if args.dry_run else "running",
        "run_root": str(run_root),
        "clip": str(clip),
        "calibration_contract": str(calibration_contract),
        "focal_scale_from_canonical_k": focal_scale,
        "calibration_source": calibration_payload.get("intrinsics_source") or calibration_payload.get("method"),
        "dynamic_mask": {
            "status": "applied_from_hawor_preparation" if dynamic_mask_path is not None else "not_provided",
            "preparation_report": str(args.hawor_preparation_report.expanduser().resolve()) if args.hawor_preparation_report is not None else None,
            "path": str(dynamic_mask_path) if dynamic_mask_path is not None else None,
            "sha256": dynamic_mask_sha256,
        },
        "outputs": {
            "output_dir": str(output_dir),
            "dense_npz": str(output_dir / "droid_dense_trajectory.npz"),
            "dense_json": str(output_dir / "droid_dense_trajectory.json"),
            "qc_json": str(output_dir / "droid_qc.json"),
            "shared_geometry_manifest": str(output_dir / "droid_shared_geometry.json"),
        },
        "command": cmd,
        "model_request": str(run_root / "requests" / "droid.json"),
        "model_request_payload": model_request,
        "hawor_preparation_payload": preparation_payload,
        "claim_scope": "D4 video-derived camera trajectory with explicit gauge/scale uncertainty; the same DROID artifacts are consumed by the HaWoR adapter without a second DROID invocation.",
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
    parser.add_argument("--camera-backend", choices=("droid", "vggt_omega", "vggt", "contract"), default="droid")
    parser.add_argument("--droid-root", type=Path, default=Path("third_party/DROID-SLAM"))
    parser.add_argument("--droid-weights", type=Path, default=Path("third_party/DROID-SLAM/droid.pth"))
    parser.add_argument("--droid-area", type=int, default=384 * 512)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--hawor-preparation-report", type=Path, default=None, help="Hash-bound HaWoR motion preparation report supplying the dynamic mask to the one canonical DROID run.")
    parser.add_argument("--runner-python", type=Path, default=Path(sys.executable), help="Python environment with DROID-SLAM runtime dependencies such as lietorch.")
    parser.add_argument("--vggt-python", type=Path, default=Path(sys.executable), help="Python environment for the VGGT/Omega resident camera worker.")
    parser.add_argument("--vggt-device", default="cuda")
    parser.add_argument("--vggt-target-size", type=int, default=518)
    parser.add_argument("--vggt-patch-multiple", type=int, default=14)
    parser.add_argument("--vggt-batch-size", type=int, default=1)
    parser.add_argument("--vggt-sequence-length", type=int, default=None)
    parser.add_argument("--vggt-checkpoint", type=Path, default=None)
    parser.add_argument("--vggt-allow-remote-model-download", action="store_true")
    parser.add_argument("--vggt-model-id", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-model-file", default="model.pt")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
