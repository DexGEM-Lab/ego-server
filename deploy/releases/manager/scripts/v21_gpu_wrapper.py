#!/usr/bin/env python3
"""Run one GPU submodule only after a GPU has enough estimated memory."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


def query_gpus() -> list[GpuState]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr.strip() or proc.stdout.strip()}")
    out: list[GpuState] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        out.append(GpuState(index=parts[0], total_mb=int(parts[1]), used_mb=int(parts[2]), free_mb=int(parts[3])))
    if not out:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return out


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


def append_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


@contextmanager
def global_lock(lock_dir: Path) -> Iterator[None]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "select.lock"
    with lock_path.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def select_gpu(args: argparse.Namespace, allowed: set[str] | None, log_jsonl: Path | None) -> tuple[GpuState, int]:
    lock_dir = args.lock_dir.expanduser()
    waited = 0.0
    while True:
        with global_lock(lock_dir):
            states = [gpu for gpu in query_gpus() if allowed is None or gpu.index in allowed]
            if not states:
                raise RuntimeError(f"no GPUs match --gpu-ids={args.gpu_ids!r}")
            reserved = load_active_reservations(lock_dir)
            candidates: list[tuple[int, GpuState, int]] = []
            for gpu in states:
                reservation_used = reserved.get(gpu.index, 0)
                effective_free = min(gpu.free_mb, gpu.total_mb - reservation_used) - args.reserve_mb
                candidates.append((effective_free, gpu, reservation_used))
            candidates.sort(key=lambda item: (item[0], item[1].free_mb), reverse=True)
            best_free, best_gpu, best_reserved = candidates[0]
            append_event(
                log_jsonl,
                {
                    "event": "gpu_check",
                    "at": utc_now(),
                    "module_id": args.module_id,
                    "request_mb": args.request_mb,
                    "best_gpu": best_gpu.index,
                    "best_effective_free_mb": best_free,
                    "best_reserved_mb": best_reserved,
                    "states": [gpu.__dict__ for gpu in states],
                    "reservations": reserved,
                },
            )
            if best_free >= args.request_mb:
                reservation_dir = lock_dir / "reservations"
                reservation_dir.mkdir(parents=True, exist_ok=True)
                reservation_path = reservation_dir / f"{os.getpid()}.json"
                reservation_path.write_text(
                    json.dumps(
                        {
                            "schema": "v21_gpu_wrapper_reservation.v1",
                            "created_at": utc_now(),
                            "wrapper_pid": os.getpid(),
                            "gpu_index": best_gpu.index,
                            "request_mb": args.request_mb,
                            "module_id": args.module_id,
                            "command": args.command,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return best_gpu, args.request_mb
        if args.max_wait_s > 0 and waited >= args.max_wait_s:
            raise TimeoutError(f"timed out waiting for {args.request_mb} MB GPU memory for {args.module_id}")
        threading.Event().wait(args.poll_s)
        waited += float(args.poll_s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request-mb", type=int, required=True, help="Estimated peak GPU memory required by this submodule.")
    p.add_argument("--module-id", required=True)
    p.add_argument("--gpu-ids", default=os.environ.get("V21_GPU_IDS", ""), help="Comma-separated physical GPU ids allowed for scheduling.")
    p.add_argument("--reserve-mb", type=int, default=int(os.environ.get("V21_GPU_RESERVE_MB", "1024")))
    p.add_argument("--poll-s", type=float, default=float(os.environ.get("V21_GPU_WRAPPER_POLL_S", "15")))
    p.add_argument("--max-wait-s", type=float, default=float(os.environ.get("V21_GPU_WRAPPER_MAX_WAIT_S", "0")), help="0 means wait indefinitely.")
    p.add_argument("--lock-dir", type=Path, default=Path(os.environ.get("V21_GPU_WRAPPER_LOCK_DIR", "/tmp/v21_gpu_wrapper_locks")))
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run. Use -- before the command.")
    args = p.parse_args()
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
    allowed = parse_gpu_ids(args.gpu_ids)
    gpu: GpuState | None = None
    reservation_path: Path | None = None
    child: subprocess.Popen[str] | None = None
    try:
        gpu, _ = select_gpu(args, allowed, log_jsonl)
        reservation_path = args.lock_dir.expanduser() / "reservations" / f"{os.getpid()}.json"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu.index
        env["V21_GPU_WRAPPER_MODULE_ID"] = args.module_id
        env["V21_GPU_WRAPPER_REQUEST_MB"] = str(args.request_mb)
        env["V21_GPU_WRAPPER_PHYSICAL_GPU"] = gpu.index
        append_event(log_jsonl, {"event": "launch", "at": utc_now(), "module_id": args.module_id, "gpu": gpu.index, "request_mb": args.request_mb, "command": args.command})
        child = subprocess.Popen(args.command, env=env, text=True)
        return_code = child.wait()
        append_event(log_jsonl, {"event": "exit", "at": utc_now(), "module_id": args.module_id, "gpu": gpu.index, "returncode": return_code})
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
