#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prompt_frames(path: Path) -> list[int]:
    payload = load_json(path)
    rows = payload.get("point_prompts")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"prompt_file_has_no_point_prompts: {path}")
    out = []
    for row in rows:
        if isinstance(row, dict) and row.get("target_visible", True) and row.get("positive_points"):
            out.append(int(row["frame_idx"]))
    if not out:
        raise ContractError(f"prompt_file_has_no_visible_positive_frames: {path}")
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    prompt_summary = load_json(args.prompt_summary)
    raw_manifest_path = Path(str(input_manifest["raw_frame_manifest"]))
    raw_manifest = load_json(raw_manifest_path)
    frame_count = int(input_manifest["raw_frame_manifest_summary"]["frame_count"])
    frame_start = 0
    frame_end = frame_count - 1
    primary_video = Path(str(input_manifest["primary_video"]))
    targets = prompt_summary.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ContractError("prompt_summary_has_no_targets")
    outputs: list[dict[str, Any]] = []
    for target in targets:
        prompt_path = Path(str(target["prompt_path"]))
        track_id = str(target["track_id"])
        out_dir = args.output_root / track_id / "sam2_rgb_baseline"
        frames = prompt_frames(prompt_path)
        command = [
            str(args.python_bin),
            "scripts/run_sam2_vlm_points_track.py",
            "--clip",
            str(primary_video),
            "--point-prompts",
            str(prompt_path),
            "--output-dir",
            str(out_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--sam2-repo",
            str(args.sam2_repo),
            "--model-cfg",
            str(args.model_cfg),
            "--frame-start",
            str(frame_start),
            "--frame-end",
            str(frame_end),
            "--prompt-frames",
            ",".join(str(v) for v in frames),
            "--sam2-image-width",
            str(args.sam2_image_width),
            "--render-width",
            str(args.render_width),
        ]
        env = os.environ.copy()
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        env["PYTHONPATH"] = f"scripts:{args.sam2_repo}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("compute_target=" + args.compute_target + "\n")
            log.write("command=" + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if proc.returncode != 0:
            raise ContractError(f"sam2_rgb_baseline_failed: track={track_id} code={proc.returncode} log={log_path}")
        qc_path = out_dir / "qc_sam2_vlm_points_track.json"
        if not qc_path.exists():
            raise ContractError(f"sam2_qc_missing: {qc_path}")
        qc = load_json(qc_path)
        outputs.append(
            {
                "track_id": track_id,
                "target_object_id": target.get("object_id"),
                "prompt_path": str(prompt_path),
                "sam2_output_dir": str(out_dir),
                "sam2_track": str(out_dir / "sam2_track.json"),
                "sam2_qc": str(qc_path),
                "overlay": qc.get("outputs", {}).get("overlay"),
                "visible_frames": int(qc.get("visible_frames", 0)),
                "frame_count": int(qc.get("frames", 0)),
                "prompt_frames": frames,
            }
        )
    summary = {
        "schema": "v21_sam2_rgb_baseline_summary.v0",
        "status": "ok",
        "method": "run_v21_sam2_rgb_baseline",
        "case_id": input_manifest.get("case_id"),
        "compute_target": args.compute_target,
        "input_manifest": str(args.input_manifest),
        "raw_frame_manifest": str(raw_manifest_path),
        "primary_video": str(primary_video),
        "frame_count": frame_count,
        "tracks": outputs,
        "claim_scope": "RGB-only SAM2 mask baseline for selected target prompts. Masks are measured segmentation evidence; they require contamination review before geometry or pose use.",
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V21 RGB-only SAM2 baseline masks for object prompts.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--prompt-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--sam2-image-width", type=int, default=960)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--compute-target", default="declared_gpu_compute_target")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
