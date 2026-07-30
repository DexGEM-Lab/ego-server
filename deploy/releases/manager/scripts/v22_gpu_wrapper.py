#!/usr/bin/env python3
"""Run one V22 GPU-heavy model call after selecting a physical GPU.

This is a single-video helper. It does not claim dataset rows, launch parallel
workers, or control Pi. It selects one physical GPU for one child command,
records the decision, exposes the selected GPU through CUDA_VISIBLE_DEVICES, and
releases its reservation when the child exits.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class GpuState:
    index: str
    total_mb: int
    used_mb: int
    free_mb: int
    util_percent: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_gpu_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def query_gpus(nvidia_smi: str) -> list[GpuState]:
    cmd = [
        nvidia_smi,
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr.strip() or proc.stdout.strip()}")
    states: list[GpuState] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        states.append(
            GpuState(
                index=parts[0],
                total_mb=int(float(parts[1])),
                used_mb=int(float(parts[2])),
                free_mb=int(float(parts[3])),
                util_percent=float(parts[4]),
            )
        )
    if not states:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return states


def append_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def global_lock(lock_dir: Path) -> Iterator[None]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "select.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_active_reservations(lock_dir: Path) -> dict[str, int]:
    reservation_dir = lock_dir / "reservations"
    reservation_dir.mkdir(parents=True, exist_ok=True)
    reserved: dict[str, int] = {}
    for path in reservation_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            pid = int(row.get("wrapper_pid", -1))
            if pid <= 0 or not pid_alive(pid):
                path.unlink(missing_ok=True)
                continue
            gpu = str(row.get("gpu_index"))
            reserved[gpu] = reserved.get(gpu, 0) + int(row.get("request_mb", 0))
        except Exception:
            path.unlink(missing_ok=True)
    return reserved


def runtime_identity() -> dict[str, str]:
    return {
        "runtime_agent_id": os.environ.get("V22_RUNTIME_AGENT_ID") or os.environ.get("V22_PARALLEL_AGENT_ID", "unknown"),
        "case_id": os.environ.get("V22_RUNTIME_CASE_ID") or os.environ.get("V22_PARALLEL_CASE_ID", "unknown"),
        "run_root": os.environ.get("V22_RUNTIME_RUN_ROOT") or os.environ.get("V22_PARALLEL_RUN_ROOT", "unknown"),
    }


def select_gpu(args: argparse.Namespace, allowed: set[str] | None, log_jsonl: Path | None) -> GpuState:
    identity = runtime_identity()
    with global_lock(args.lock_dir.expanduser()):
        try:
            states = [gpu for gpu in query_gpus(args.nvidia_smi) if allowed is None or gpu.index in allowed]
        except RuntimeError as exc:
            append_event(log_jsonl, {"event": "gpu_unavailable", "at": utc_now(), "schema": "v22_gpu_wrapper_event.v1", "module_id": args.module_id, **identity, "request_mb": int(args.request_mb), "error": str(exc)})
            raise
        if not states:
            raise RuntimeError(f"no GPUs match --gpu-ids={args.gpu_ids!r}")
        reserved = load_active_reservations(args.lock_dir.expanduser())
        candidates: list[tuple[int, float, GpuState, int]] = []
        for gpu in states:
            reservation_used = reserved.get(gpu.index, 0)
            effective_free = min(gpu.free_mb, gpu.total_mb - reservation_used) - int(args.reserve_mb)
            candidates.append((effective_free, -gpu.util_percent, gpu, reservation_used))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].free_mb), reverse=True)
        best_free, _neg_util, best_gpu, best_reserved = candidates[0]
        append_event(
            log_jsonl,
            {
                "event": "gpu_check",
                "at": utc_now(),
                "schema": "v22_gpu_wrapper_event.v1",
                "module_id": args.module_id,
                **identity,
                "request_mb": int(args.request_mb),
                "reserve_mb": int(args.reserve_mb),
                "best_gpu": best_gpu.index,
                "best_effective_free_mb": int(best_free),
                "best_reserved_mb": int(best_reserved),
                "states": [gpu.__dict__ for gpu in states],
                "reservations": reserved,
            },
        )
        if best_free < int(args.request_mb):
            raise RuntimeError(
                f"gpu_capacity_unavailable: module={args.module_id} request_mb={args.request_mb} "
                f"best_gpu={best_gpu.index} best_effective_free_mb={best_free}"
            )
        reservation_dir = args.lock_dir.expanduser() / "reservations"
        reservation_dir.mkdir(parents=True, exist_ok=True)
        reservation_path = reservation_dir / f"{os.getpid()}.json"
        reservation_path.write_text(
            json.dumps(
                {
                    "schema": "v22_gpu_wrapper_reservation.v0",
                    "created_at": utc_now(),
                    "wrapper_pid": os.getpid(),
                    "gpu_index": best_gpu.index,
                    "request_mb": int(args.request_mb),
                    "module_id": args.module_id,
                    "command": args.command,
                    **identity,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return best_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-mb", type=int, required=True, help="Estimated peak GPU memory required by this V22 submodule.")
    parser.add_argument("--module-id", required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("V22_GPU_IDS", ""), help="Comma-separated physical GPU ids allowed for scheduling.")
    parser.add_argument("--reserve-mb", type=int, default=int(os.environ.get("V22_GPU_RESERVE_MB", "2048")))
    parser.add_argument("--lock-dir", type=Path, default=Path(os.environ.get("V22_GPU_WRAPPER_LOCK_DIR", "/tmp/v22_gpu_wrapper_locks")))
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--nvidia-smi", default=os.environ.get("NVIDIA_SMI", "nvidia-smi"))
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run. Use -- before the command.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing command after --")
    if args.request_mb <= 0:
        raise SystemExit("--request-mb must be positive")
    return args


def main() -> int:
    args = parse_args()
    log_jsonl = args.log_jsonl.expanduser() if args.log_jsonl else None
    reservation_path: Path | None = None
    child: subprocess.Popen[str] | None = None
    try:
        gpu = select_gpu(args, parse_gpu_ids(args.gpu_ids), log_jsonl)
        reservation_path = args.lock_dir.expanduser() / "reservations" / f"{os.getpid()}.json"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu.index
        env["V22_MODULE_ID"] = args.module_id
        env["V22_GPU_WRAPPER_MODULE_ID"] = args.module_id
        env["V22_GPU_WRAPPER_REQUEST_MB"] = str(int(args.request_mb))
        env["V22_GPU_WRAPPER_PHYSICAL_GPU"] = gpu.index
        identity = runtime_identity()
        env.setdefault("V22_RUNTIME_AGENT_ID", identity["runtime_agent_id"])
        env.setdefault("V22_RUNTIME_CASE_ID", identity["case_id"])
        env.setdefault("V22_RUNTIME_RUN_ROOT", identity["run_root"])
        env.setdefault("V21_COMPUTE_TARGET", "zjh@115.190.235.210:A800")
        append_event(log_jsonl, {"event": "launch", "at": utc_now(), "schema": "v22_gpu_wrapper_event.v1", "module_id": args.module_id, **identity, "gpu": gpu.index, "request_mb": int(args.request_mb), "command": args.command})
        child = subprocess.Popen(args.command, env=env, text=True)
        return_code = child.wait()
        append_event(log_jsonl, {"event": "exit", "at": utc_now(), "schema": "v22_gpu_wrapper_event.v1", "module_id": args.module_id, **identity, "gpu": gpu.index, "returncode": int(return_code)})
        return int(return_code)
    except KeyboardInterrupt:
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
        raise
    finally:
        if reservation_path is not None:
            reservation_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
