#!/usr/bin/env python3
"""Record the batch producer, resident-service, and GPU condition every 30 seconds.

The observer is deliberately read-only.  It derives request throughput and p50/p95
from manager-owned admission events appended after the observer begins, samples
all local service health endpoints on ports 28000--28006, and captures a single
``nvidia-smi`` GPU-utilization snapshot per observation.  It never submits,
cancels, or changes annotation requests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ADMISSION_EVENTS = Path("/home/zjh/data/v22_api_release_da17415/jobs/_algorithm_admission_events.jsonl")
SERVICE_PORTS = tuple(range(28000, 28007))
ROUTE_PORTS = {
    "/unidepth.infer": 28000,
    "/hands.detect": 28001,
    "/wilor.reconstruct": 28004,
    "/droid.create_session": 28002,
    "/droid.push_frame": 28002,
    "/droid.finalize": 28002,
    "/hawor.infer_tracks": 28003,
    "/hawor_infiller.fill": 28003,
    "/cosmos3.reason": 28006,
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def service_health(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/-/healthz"
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read(256).decode("utf-8", errors="replace").strip()
        return {"status": "reachable", "http_status": int(response.status), "body": body, "latency_s": time.monotonic() - started}
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "http_status": int(exc.code), "latency_s": time.monotonic() - started}
    except Exception as exc:
        return {"status": "unreachable", "error": repr(exc), "latency_s": time.monotonic() - started}


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return [{"status": "nvidia_smi_failed", "error": proc.stderr.strip() or proc.stdout.strip()}]
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        rows.append(
            {
                "index": fields[0],
                "name": fields[1],
                "memory_used_mib": fields[2],
                "memory_total_mib": fields[3],
                "utilization_gpu_pct": fields[4],
                "utilization_memory_pct": fields[5],
                "temperature_c": fields[6],
                "power_w": fields[7],
            }
        )
    return rows


def producer_state(condition_root: Path) -> dict[str, Any]:
    summary = read_json(condition_root / "dataset_batch_summary.json")
    return {
        "summary_present": bool(summary),
        "eligible_video_count": summary.get("eligible_video_count"),
        "submitted_count": summary.get("submitted_count"),
        "terminal_count": summary.get("terminal_count"),
        "status_counts": summary.get("status_counts", {}),
        "elapsed_s": summary.get("elapsed_s"),
    }


def service_statistics(admission_events: Path, *, started_at_unix: float, interval_started_at_unix: float) -> dict[str, dict[str, Any]]:
    by_port: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(admission_events):
        route = row.get("route")
        finished = row.get("finished_at_unix", row.get("finished_at"))
        if route not in ROUTE_PORTS or not isinstance(finished, (int, float)) or float(finished) < started_at_unix:
            continue
        by_port[ROUTE_PORTS[str(route)]].append(row)

    result: dict[str, dict[str, Any]] = {}
    interval_s = max(0.001, time.time() - interval_started_at_unix)
    for port in SERVICE_PORTS:
        rows = by_port.get(port, [])
        completed_since_last = [
            row for row in rows if isinstance(row.get("finished_at_unix", row.get("finished_at")), (int, float)) and float(row.get("finished_at_unix", row.get("finished_at"))) >= interval_started_at_unix
        ]
        latencies = [float(row["total_wall_s"]) for row in rows if isinstance(row.get("total_wall_s"), (int, float))]
        waits = [float(row["wait_s"]) for row in rows if isinstance(row.get("wait_s"), (int, float))]
        result[str(port)] = {
            "admission_events_since_start": len(rows),
            "completed_events_this_interval": len(completed_since_last),
            "throughput_requests_per_s": len(completed_since_last) / interval_s,
            "p50_wall_s": percentile(latencies, 0.50),
            "p95_wall_s": percentile(latencies, 0.95),
            "p50_admission_wait_s": percentile(waits, 0.50),
            "p95_admission_wait_s": percentile(waits, 0.95),
            "status_counts_since_start": {
                str(status): sum(1 for row in rows if str(row.get("status")) == str(status))
                for status in sorted({row.get("status") for row in rows}, key=str)
            },
        }
    return result


class AdmissionEventTail:
    """Incrementally consume only manager rows appended after observer startup."""

    def __init__(self, path: Path, *, started_at_unix: float) -> None:
        self.path = path
        self.started_at_unix = started_at_unix
        try:
            self.offset = path.stat().st_size
        except OSError:
            self.offset = 0
        self.by_port: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def update(self, *, interval_started_at_unix: float) -> dict[str, dict[str, Any]]:
        new_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                while True:
                    line_start = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        handle.seek(line_start)
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    route = row.get("route") if isinstance(row, dict) else None
                    finished = row.get("finished_at_unix", row.get("finished_at")) if isinstance(row, dict) else None
                    if route in ROUTE_PORTS and isinstance(finished, (int, float)) and float(finished) >= self.started_at_unix:
                        port = ROUTE_PORTS[str(route)]
                        self.by_port[port].append(row)
                        new_rows[port].append(row)
                self.offset = handle.tell()
        except OSError:
            pass

        interval_s = max(0.001, time.time() - interval_started_at_unix)
        result: dict[str, dict[str, Any]] = {}
        for port in SERVICE_PORTS:
            rows = self.by_port.get(port, [])
            latencies = [float(row["total_wall_s"]) for row in rows if isinstance(row.get("total_wall_s"), (int, float))]
            waits = [float(row["wait_s"]) for row in rows if isinstance(row.get("wait_s"), (int, float))]
            result[str(port)] = {
                "admission_events_since_start": len(rows),
                "completed_events_this_interval": len(new_rows.get(port, [])),
                "throughput_requests_per_s": len(new_rows.get(port, [])) / interval_s,
                "p50_wall_s": percentile(latencies, 0.50),
                "p95_wall_s": percentile(latencies, 0.95),
                "p50_admission_wait_s": percentile(waits, 0.50),
                "p95_admission_wait_s": percentile(waits, 0.95),
                "status_counts_since_start": {
                    str(status): sum(1 for row in rows if str(row.get("status")) == str(status))
                    for status in sorted({row.get("status") for row in rows}, key=str)
                },
            }
        return result


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def observation(*, condition_root: Path, admission_events: Path, service_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "ego.annotation.two_batch_condition_observation.v1",
        "observed_at_unix": time.time(),
        "condition_root": str(condition_root),
        "admission_events": str(admission_events),
        "admission_scope": "manager events with finished_at_unix at or after observer start; observer is read-only",
        "producer": producer_state(condition_root),
        "services": {
            str(port): {"health": service_health(port), "performance": service_stats[str(port)]}
            for port in SERVICE_PORTS
        },
        "gpus": gpu_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-events", type=Path, default=DEFAULT_ADMISSION_EVENTS)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--producer-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_s <= 0:
        raise ValueError("--interval-s must be positive")

    condition_root = args.condition_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    admission_events = args.admission_events.expanduser().resolve()
    started_at_unix = time.time()
    interval_started_at_unix = started_at_unix
    admission_tail = AdmissionEventTail(admission_events, started_at_unix=started_at_unix)
    stop = threading.Event()
    while True:
        row = observation(
            condition_root=condition_root,
            admission_events=admission_events,
            service_stats=admission_tail.update(interval_started_at_unix=interval_started_at_unix),
        )
        append_jsonl(output, row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if args.once:
            return 0
        if args.producer_pid is not None and not process_alive(args.producer_pid):
            return 0
        interval_started_at_unix = time.time()
        stop.wait(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
