#!/usr/bin/env python3
"""Sweep batch size for the four GPU-heavy ego annotation model stages.

The benchmark treats batch_size as the number of same-format model requests in
one benchmark batch. A model can be driven either by a true batch command or by
launching one command per sample concurrently. The request handed to the model
stays minimal: input_video, output_dir, and optionally camera for stages that
need it.

Run this on the remote A800 host, not on the local workstation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ALGORITHMS = ("unidepth", "wilor", "droid", "hawor")
DEFAULT_BATCH_SIZES = (1, 4, 16, 64, 128, 256, 512, 1024)
OOM_RE = re.compile(
    r"out of memory|cuda.*oom|cublas.*alloc|cudnn.*alloc|cannot allocate memory|oom", re.IGNORECASE
)


@dataclass(frozen=True)
class CommandSpec:
    sample_command: list[str] | None
    batch_command: list[str] | None


@dataclass
class GpuSample:
    t_s: float
    util_gpu_pct: float | None
    memory_used_mb: float | None


class GpuMonitor:
    def __init__(self, gpu_index: int, interval_s: float, active_threshold_pct: float) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self.active_threshold_pct = float(active_threshold_pct)
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4.0))

    def _run(self) -> None:
        query = "utilization.gpu,memory.used"
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            t_s = time.perf_counter()
            try:
                proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if proc.returncode != 0:
                    self.error = proc.stderr.strip()[-500:] or f"nvidia-smi exited {proc.returncode}"
                    self.samples.append(GpuSample(t_s=t_s, util_gpu_pct=None, memory_used_mb=None))
                else:
                    first = proc.stdout.strip().splitlines()[0]
                    parts = [part.strip() for part in first.split(",")]
                    util = float(parts[0]) if parts and parts[0] else None
                    mem = float(parts[1]) if len(parts) > 1 and parts[1] else None
                    self.samples.append(GpuSample(t_s=t_s, util_gpu_pct=util, memory_used_mb=mem))
            except Exception as exc:  # noqa: BLE001 - monitor must not kill benchmark process
                self.error = repr(exc)
                self.samples.append(GpuSample(t_s=t_s, util_gpu_pct=None, memory_used_mb=None))
            self._stop.wait(self.interval_s)

    def summarize(self, baseline_memory_mb: float | None = None) -> dict[str, Any]:
        utils = [s.util_gpu_pct for s in self.samples if s.util_gpu_pct is not None]
        mems = [s.memory_used_mb for s in self.samples if s.memory_used_mb is not None]
        active_count = sum(1 for value in utils if value is not None and value >= self.active_threshold_pct)
        active_ms = active_count * self.interval_s * 1000.0
        peak_mem = max(mems) if mems else None
        return {
            "gpu_latency_ms": active_ms,
            "gpu_latency_source": "nvidia_smi_active_sample_count",
            "gpu_util_avg_pct": sum(utils) / len(utils) if utils else None,
            "gpu_util_max_pct": max(utils) if utils else None,
            "gpu_active_threshold_pct": self.active_threshold_pct,
            "peak_gpu_memory_mb": peak_mem,
            "baseline_gpu_memory_mb": baseline_memory_mb,
            "peak_gpu_memory_delta_mb": (peak_mem - baseline_memory_mb) if peak_mem is not None and baseline_memory_mb is not None else None,
            "sample_count": len(self.samples),
            "monitor_error": self.error,
        }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv_ints(raw: str | None, default: tuple[int, ...]) -> list[int]:
    if raw is None or not raw.strip():
        return list(default)
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise RuntimeError(f"batch sizes must be positive: {raw}")
        values.append(value)
    if not values:
        raise RuntimeError("empty batch size list")
    return values


def command_config_from_payload(payload: dict[str, Any]) -> dict[str, CommandSpec]:
    raw_algorithms = payload.get("algorithms", payload)
    if not isinstance(raw_algorithms, dict):
        raise RuntimeError("command config must be an object or contain an algorithms object")
    out: dict[str, CommandSpec] = {}
    for name, raw_spec in raw_algorithms.items():
        if not isinstance(raw_spec, dict):
            raise RuntimeError(f"command spec for {name} must be an object")
        sample_command = raw_spec.get("sample_command")
        batch_command = raw_spec.get("batch_command")
        if sample_command is not None and not isinstance(sample_command, list):
            raise RuntimeError(f"{name}.sample_command must be a list of argv tokens")
        if batch_command is not None and not isinstance(batch_command, list):
            raise RuntimeError(f"{name}.batch_command must be a list of argv tokens")
        out[str(name)] = CommandSpec(
            sample_command=[str(x) for x in sample_command] if sample_command is not None else None,
            batch_command=[str(x) for x in batch_command] if batch_command is not None else None,
        )
    return out


def load_command_config(path: Path | None) -> dict[str, CommandSpec]:
    if path is None:
        raise RuntimeError(
            "--command-config is required. It maps each algorithm to either a sample_command "
            "that accepts {request_json}, or a batch_command that accepts {batch_request_json}."
        )
    return command_config_from_payload(load_json(path))


def check_gpu_available(gpu_index: int) -> None:
    proc = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("nvidia-smi is unavailable or the selected GPU is invalid; run this on the A800 host")


def current_gpu_memory_mb(gpu_index: int) -> float | None:
    proc = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def materialize_input(source: Path, target: Path, mode: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "same_path":
        return source
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise RuntimeError(f"unknown materialize mode: {mode}")
    return target


def make_sample_request(
    *,
    algorithm: str,
    input_video: Path,
    output_dir: Path,
    camera: dict[str, Any] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "input_video": str(input_video),
        "output_dir": str(output_dir),
    }
    if camera is not None and algorithm in {"droid", "hawor"}:
        request["camera"] = camera
    return request


def expand_command(template: list[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in template]


def classify_failure(text: str, returncodes: list[int]) -> tuple[str, bool]:
    if any(code != 0 for code in returncodes):
        if OOM_RE.search(text):
            return "oom", True
        return "failed", False
    return "ok", False


def launch_many(commands: list[list[str]], cwd: Path | None, env: dict[str, str]) -> tuple[list[int], str, str]:
    procs: list[subprocess.Popen[str]] = []
    for cmd in commands:
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    returncodes: list[int] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for proc in procs:
        stdout, stderr = proc.communicate()
        returncodes.append(int(proc.returncode or 0))
        stdout_parts.append(stdout or "")
        stderr_parts.append(stderr or "")
    return returncodes, "\n".join(stdout_parts), "\n".join(stderr_parts)


def run_one_batch(
    *,
    algorithm: str,
    batch_size: int,
    spec: CommandSpec,
    input_video: Path,
    output_root: Path,
    camera: dict[str, Any] | None,
    materialize_mode: str,
    cwd: Path | None,
    env: dict[str, str],
    gpu_index: int,
    monitor_interval_s: float,
    active_threshold_pct: float,
    dry_run: bool,
) -> dict[str, Any]:
    batch_dir = output_root / algorithm / f"batch_size_{batch_size:06d}"
    requests_dir = batch_dir / "requests"
    samples_dir = batch_dir / "samples"
    batch_dir.mkdir(parents=True, exist_ok=True)

    sample_requests: list[dict[str, Any]] = []
    sample_commands: list[list[str]] = []
    for sample_idx in range(batch_size):
        sample_dir = samples_dir / f"sample_{sample_idx:06d}"
        sample_input = materialize_input(input_video, sample_dir / "input.mp4", materialize_mode)
        sample_output = sample_dir / "output"
        request = make_sample_request(algorithm=algorithm, input_video=sample_input, output_dir=sample_output, camera=camera)
        request_json = requests_dir / f"sample_{sample_idx:06d}.json"
        write_json(request_json, request)
        sample_requests.append({"request_json": str(request_json), **request})
        if spec.sample_command is not None:
            sample_commands.append(
                expand_command(
                    spec.sample_command,
                    {
                        "request_json": str(request_json),
                        "input_video": str(sample_input),
                        "output_dir": str(sample_output),
                        "batch_dir": str(batch_dir),
                        "algorithm": algorithm,
                        "batch_size": str(batch_size),
                    },
                )
            )

    batch_request = {
        "algorithm": algorithm,
        "batch_size": batch_size,
        "samples": sample_requests,
    }
    batch_request_json = batch_dir / "batch_request.json"
    write_json(batch_request_json, batch_request)

    if spec.batch_command is not None:
        commands = [
            expand_command(
                spec.batch_command,
                {
                    "batch_request_json": str(batch_request_json),
                    "batch_dir": str(batch_dir),
                    "algorithm": algorithm,
                    "batch_size": str(batch_size),
                },
            )
        ]
        command_mode = "batch_command"
    elif sample_commands:
        commands = sample_commands
        command_mode = "concurrent_sample_commands"
    else:
        raise RuntimeError(f"{algorithm} has neither batch_command nor sample_command")

    write_json(batch_dir / "commands.json", {"mode": command_mode, "commands": commands})
    if dry_run:
        return {
            "algorithm": algorithm,
            "batch_size": batch_size,
            "status": "dry_run",
            "command_mode": command_mode,
            "batch_dir": str(batch_dir),
            "wall_latency_ms": None,
            "gpu_latency_ms": None,
            "latency_per_sample_ms": None,
            "throughput_samples_s": None,
            "peak_gpu_memory_mb": None,
        }

    baseline_memory = current_gpu_memory_mb(gpu_index)
    monitor = GpuMonitor(gpu_index=gpu_index, interval_s=monitor_interval_s, active_threshold_pct=active_threshold_pct)
    started = time.perf_counter()
    monitor.start()
    try:
        returncodes, stdout, stderr = launch_many(commands, cwd=cwd, env=env)
    finally:
        monitor.stop()
    finished = time.perf_counter()
    wall_latency_ms = (finished - started) * 1000.0
    monitor_summary = monitor.summarize(baseline_memory_mb=baseline_memory)
    combined_text = "\n".join([stdout[-8000:], stderr[-8000:]])
    status, is_oom = classify_failure(combined_text, returncodes)
    gpu_latency_ms = monitor_summary["gpu_latency_ms"]
    row = {
        "algorithm": algorithm,
        "batch_size": int(batch_size),
        "status": status,
        "is_oom": bool(is_oom),
        "command_mode": command_mode,
        "batch_dir": str(batch_dir),
        "wall_latency_ms": float(wall_latency_ms),
        "gpu_latency_ms": float(gpu_latency_ms) if gpu_latency_ms is not None else None,
        "latency_per_sample_ms": float(gpu_latency_ms / batch_size) if gpu_latency_ms is not None else None,
        "throughput_samples_s": float(batch_size / (wall_latency_ms / 1000.0)) if wall_latency_ms > 0 else None,
        "peak_gpu_memory_mb": monitor_summary["peak_gpu_memory_mb"],
        "returncodes": returncodes,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        **monitor_summary,
    }
    write_json(batch_dir / "benchmark_result.json", row)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "algorithm",
        "batch_size",
        "status",
        "wall_latency_ms",
        "gpu_latency_ms",
        "latency_per_sample_ms",
        "throughput_samples_s",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_delta_mb",
        "gpu_util_avg_pct",
        "gpu_util_max_pct",
        "command_mode",
        "batch_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for algorithm in sorted({str(row.get("algorithm")) for row in rows}):
        alg_rows = [row for row in rows if str(row.get("algorithm")) == algorithm]
        ok_rows = [row for row in alg_rows if row.get("status") == "ok"]
        oom_rows = [row for row in alg_rows if row.get("status") == "oom" or row.get("is_oom")]
        best = None
        if ok_rows:
            best = max(ok_rows, key=lambda row: float(row.get("throughput_samples_s") or 0.0))
        summary[algorithm] = {
            "tested_batch_sizes": [int(row["batch_size"]) for row in alg_rows if row.get("batch_size") is not None],
            "max_successful_batch_size": max((int(row["batch_size"]) for row in ok_rows), default=None),
            "first_oom_batch_size": min((int(row["batch_size"]) for row in oom_rows), default=None),
            "best_throughput_batch_size": int(best["batch_size"]) if best is not None else None,
            "best_throughput_samples_s": best.get("throughput_samples_s") if best is not None else None,
            "best_latency_per_sample_ms": best.get("latency_per_sample_ms") if best is not None else None,
            "best_peak_gpu_memory_mb": best.get("peak_gpu_memory_mb") if best is not None else None,
            "last_status": alg_rows[-1].get("status") if alg_rows else None,
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True, help="EgoScale video visible from the A800 host.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command-config", type=Path, required=True, help="JSON mapping algorithms to batch_command or sample_command argv templates.")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--batch-sizes", default=",".join(str(x) for x in DEFAULT_BATCH_SIZES))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--camera-intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--camera-image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--materialize-input", choices=("same_path", "symlink", "copy"), default="symlink")
    parser.add_argument("--monitor-interval-s", type=float, default=0.1)
    parser.add_argument("--gpu-active-threshold-pct", type=float, default=10.0)
    parser.add_argument("--cwd", type=Path, default=None, help="Working directory for algorithm commands.")
    parser.add_argument("--stop-on-any-failure", action="store_true", help="Stop an algorithm sweep on any failure, not only OOM.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_video = args.input_video.expanduser().resolve()
    if not input_video.exists():
        raise SystemExit(f"missing input video: {input_video}")
    if not args.dry_run:
        check_gpu_available(args.gpu_index)
    algorithms = [item.strip() for item in str(args.algorithms).split(",") if item.strip()]
    batch_sizes = parse_csv_ints(args.batch_sizes, DEFAULT_BATCH_SIZES)
    command_config = load_command_config(args.command_config)
    missing = [name for name in algorithms if name not in command_config]
    if missing:
        raise SystemExit(f"command config lacks algorithms: {', '.join(missing)}")
    camera = None
    if args.camera_intrinsics is not None or args.camera_image_size is not None:
        if args.camera_intrinsics is None or args.camera_image_size is None:
            raise SystemExit("camera requires both --camera-intrinsics and --camera-image-size")
        camera = {
            "model": "pinhole",
            "intrinsics_px": [float(x) for x in args.camera_intrinsics],
            "image_size": [int(x) for x in args.camera_image_size],
            "distortion": None,
        }

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    rows: list[dict[str, Any]] = []
    run_config = {
        "input_video": str(input_video),
        "algorithms": algorithms,
        "batch_sizes": batch_sizes,
        "gpu_index": int(args.gpu_index),
        "camera": camera,
        "materialize_input": args.materialize_input,
        "monitor_interval_s": float(args.monitor_interval_s),
        "gpu_active_threshold_pct": float(args.gpu_active_threshold_pct),
        "cwd": str(args.cwd.expanduser().resolve()) if args.cwd is not None else None,
        "dry_run": bool(args.dry_run),
    }
    write_json(output_root / "benchmark_config.json", run_config)

    for algorithm in algorithms:
        for batch_size in batch_sizes:
            row = run_one_batch(
                algorithm=algorithm,
                batch_size=batch_size,
                spec=command_config[algorithm],
                input_video=input_video,
                output_root=output_root,
                camera=camera,
                materialize_mode=args.materialize_input,
                cwd=args.cwd.expanduser().resolve() if args.cwd is not None else None,
                env=env,
                gpu_index=args.gpu_index,
                monitor_interval_s=args.monitor_interval_s,
                active_threshold_pct=args.gpu_active_threshold_pct,
                dry_run=args.dry_run,
            )
            rows.append(row)
            summary = summarize_rows(rows)
            write_json(output_root / "benchmark_results.json", {"config": run_config, "rows": rows, "summary": summary})
            write_json(output_root / "benchmark_summary.json", summary)
            write_csv(output_root / "benchmark_results.csv", rows)
            print(json.dumps({k: row.get(k) for k in ["algorithm", "batch_size", "status", "wall_latency_ms", "gpu_latency_ms", "throughput_samples_s", "peak_gpu_memory_mb"]}, ensure_ascii=False))
            if row.get("is_oom") or (args.stop_on_any_failure and row.get("status") not in {"ok", "dry_run"}):
                break

    summary = summarize_rows(rows)
    write_json(output_root / "benchmark_results.json", {"config": run_config, "rows": rows, "summary": summary})
    write_json(output_root / "benchmark_summary.json", summary)
    write_csv(output_root / "benchmark_results.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
