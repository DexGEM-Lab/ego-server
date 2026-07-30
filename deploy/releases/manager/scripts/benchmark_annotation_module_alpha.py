#!/usr/bin/env python3
"""Benchmark one annotation module command and emit throughput observation rows."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--input-duration-s", type=float, required=True)
    parser.add_argument("--gpu-hours-per-video-hour", type=float, default=None)
    parser.add_argument("--queue-wait-s", type=float, default=0.0)
    parser.add_argument("--batch-fill-ratio", type=float, default=None)
    parser.add_argument("--worker-residency-ratio", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to benchmark; prefix with -- before the command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command to benchmark")
    started = time.perf_counter()
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - started
    row = {
        "module": args.module,
        "command": command,
        "input_duration_s": float(args.input_duration_s),
        "elapsed_s": float(elapsed),
        "module_speed_x": float(args.input_duration_s / elapsed) if elapsed > 0 else None,
        "gpu_hours_per_video_hour": args.gpu_hours_per_video_hour,
        "queue_wait_s": float(args.queue_wait_s),
        "batch_fill_ratio": args.batch_fill_ratio,
        "worker_residency_ratio": args.worker_residency_ratio,
        "status": "ok" if proc.returncode == 0 else "failed",
        "failed": proc.returncode != 0,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"throughput_observations": [row]}, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
