#!/usr/bin/env python3
"""Lightweight GPU usage snapshots for V22 single-video runs."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def query_nvidia_smi(nvidia_smi: str = "nvidia-smi") -> tuple[str, list[dict[str, Any]], str | None]:
    cmd = [
        nvidia_smi,
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except Exception as exc:
        return "unavailable", [], str(exc)
    if proc.returncode != 0:
        return "unavailable", [], (proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}")
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            return "unavailable", [], f"unexpected nvidia-smi row: {line!r}"
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mib": _float(parts[2]),
                "memory_used_mib": _float(parts[3]),
                "memory_free_mib": _float(parts[4]),
                "gpu_util_percent": _float(parts[5]),
                "memory_util_percent": _float(parts[6]),
                "power_draw_w": None if parts[7] in {"[Not Supported]", "N/A"} else _float(parts[7]),
            }
        )
    return "ok", rows, None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_gpu_snapshot(
    *,
    run_root: Path,
    stage: str,
    phase: str,
    model_request: Path | None = None,
    nvidia_smi: str | None = None,
) -> dict[str, Any]:
    binary = nvidia_smi or os.environ.get("NVIDIA_SMI", "nvidia-smi")
    status, gpus, error = query_nvidia_smi(binary)
    payload: dict[str, Any] = {
        "schema": "v22.gpu_usage_snapshot.v0",
        "event": "gpu_usage_snapshot",
        "status": status,
        "at": utc_now(),
        "stage": stage,
        "phase": phase,
        "nvidia_smi": binary,
        "gpus": gpus,
        "model_request": str(model_request) if model_request is not None else None,
    }
    if error is not None:
        payload["error"] = error
    append_jsonl(run_root / "logs" / "gpu_usage_snapshots.jsonl", payload)
    return payload
