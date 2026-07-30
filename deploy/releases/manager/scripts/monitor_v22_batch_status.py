#!/usr/bin/env python3
"""Event-driven durable telemetry for a complete V22 batch run.

The monitor records every watched filesystem event and writes rich snapshots when
pipeline/global state changes. It exits only after every admitted item has an
item_result.json, or on an explicit signal. It does not issue model requests.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

IN_ACCESS = 0x00000001
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
WATCH_MASK = IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE | IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE | IN_DELETE | IN_DELETE_SELF | IN_MOVE_SELF
EVENT_STRUCT = struct.Struct("iIII")
SKIP_DIRS = {
    "rgb",
    "review_frames",
    "hybrid_overlay_frames",
    "wilor_overlay_frames",
    "input",
    "measurements",
    "renders",
    "requests",
    "failures",
    "product_bundle",
    ".rapid_complete_pipeline_slots",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def append_jsonl_batch(path: Path, payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    append_jsonl_batch(path, [payload])


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def tail_jsonl(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def result_inventory(root: Path) -> dict[str, Any]:
    result_files = list(root.glob("items/item_*/item_result.json"))
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    finished_values: list[float] = []
    for path in result_files:
        value = load_json(path)
        if value is None:
            counts["invalid_json"] += 1
            continue
        status = str(value.get("status") or "unknown")
        counts[status] += 1
        try:
            elapsed = float(value.get("elapsed_s"))
            if elapsed >= 0.0 and math.isfinite(elapsed):
                elapsed_values.append(elapsed)
        except (TypeError, ValueError):
            pass
        finished = value.get("finished_at")
        if isinstance(finished, str):
            try:
                finished_values.append(datetime.fromisoformat(finished.replace("Z", "+00:00")).timestamp())
            except ValueError:
                pass
        if status.startswith("failed"):
            errors.append({"item": path.parent.name, "status": status, "error": value.get("error"), "run_root": value.get("run_root")})
    timing: dict[str, Any] = {"count": len(elapsed_values)}
    if elapsed_values:
        timing.update({"mean_elapsed_s": statistics.fmean(elapsed_values), "p50_elapsed_s": statistics.median(elapsed_values), "max_elapsed_s": max(elapsed_values)})
        if len(elapsed_values) >= 2:
            timing["p95_elapsed_s"] = statistics.quantiles(elapsed_values, n=20, method="inclusive")[18]
    if len(finished_values) >= 2:
        span = max(finished_values) - min(finished_values)
        timing["terminal_rate_per_hour"] = len(finished_values) * 3600.0 / span if span > 0.0 else None
        timing["finished_at_span_s"] = span
    return {"count": len(result_files), "status_counts": dict(counts), "failed_examples": errors[-20:], "timing": timing}


def process_inventory(root: Path) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    slot_waiters = 0
    monitor_pid = os.getpid()
    root_texts = {str(root), root.name}
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "status", "cpu_percent", "create_time"]):
        try:
            info = process.info
            if info["pid"] == monitor_pid:
                continue
            cmdline = info.get("cmdline") or []
            command = " ".join(cmdline)
            name = str(info.get("name") or "")
            if name.startswith("python") and any(root_text in command for root_text in root_texts) and "single-item-index" in command:
                active.append({"pid": info["pid"], "ppid": info["ppid"], "name": name, "status": info["status"], "cpu_percent": info["cpu_percent"], "command": command})
            elif name == "flock":
                try:
                    parent_command = " ".join(psutil.Process(int(info["ppid"])).cmdline())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    parent_command = ""
                if any(root_text in parent_command for root_text in root_texts):
                    slot_waiters += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"active_pipeline_processes": active, "active_pipeline_count": len(active), "slot_waiter_process_count": slot_waiters}


def service_snapshot() -> dict[str, Any]:
    services: dict[str, Any] = {}
    for port in (28000, 28001, 28002, 28003, 28004):
        row: dict[str, Any] = {"port": port, "health_http_status": None, "health_body": None}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/-/healthz", timeout=3.0) as response:
                row["health_http_status"] = int(response.status)
                row["health_body"] = response.read(4096).decode("utf-8", errors="replace")
        except Exception as exc:
            row["health_error"] = repr(exc)
        if port == 28002:
            try:
                with urllib.request.urlopen("http://127.0.0.1:28002/status", timeout=3.0) as response:
                    row["status_http_status"] = int(response.status)
                    row["status_body"] = response.read(16384).decode("utf-8", errors="replace")
            except Exception as exc:
                row["status_error"] = repr(exc)
        services[str(port)] = row
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        gpu = {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        gpu = {"error": repr(exc)}
    return {"services": services, "gpu": gpu}


def host_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "hostname": socket.gethostname(),
        "load_average": list(os.getloadavg()),
        "cpu_count": psutil.cpu_count(),
        "memory": {"total": vm.total, "available": vm.available, "used": vm.used, "percent": vm.percent},
        "root_disk": {"total": disk.total, "free": disk.free, "used": disk.used, "percent": disk.percent},
        "boot_time": psutil.boot_time(),
    }


def service_log_inventory() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pattern in ("/tmp/ray-ego-serve-v2-gpu*/session_latest/logs/serve/*.log", "/home/zjh/ego-service-redeploy/runtime/droid_fix_redeploy_20260718.log"):
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                stat = path.stat()
                with path.open("rb") as handle:
                    handle.seek(max(0, stat.st_size - 4096))
                    tail = handle.read().decode("utf-8", errors="replace")
                output[str(path)] = {"size_bytes": stat.st_size, "mtime": stat.st_mtime, "tail": tail[-2000:]}
            except OSError:
                continue
    return output


class Inotify:
    def __init__(self) -> None:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        self.libc = ctypes.CDLL(libc_name, use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]
        self.libc.inotify_init1.restype = ctypes.c_int
        self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.libc.inotify_add_watch.restype = ctypes.c_int
        self.fd = self.libc.inotify_init1(os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        self.paths: dict[int, Path] = {}

    def add(self, path: Path) -> None:
        if not path.is_dir():
            return
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), WATCH_MASK)
        if wd >= 0:
            self.paths[wd] = path

    def add_tree(self, root: Path) -> None:
        if not root.is_dir():
            return
        for current, dirs, _files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".feishu_")]
            self.add(Path(current))

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def event_names(mask: int) -> list[str]:
    names = []
    for bit, name in ((IN_MODIFY,"modify"),(IN_ATTRIB,"attrib"),(IN_CLOSE_WRITE,"close_write"),(IN_MOVED_FROM,"moved_from"),(IN_MOVED_TO,"moved_to"),(IN_CREATE,"create"),(IN_DELETE,"delete"),(IN_DELETE_SELF,"delete_self"),(IN_MOVE_SELF,"move_self"),(IN_ISDIR,"is_dir"),(IN_IGNORED,"ignored")):
        if mask & bit:
            names.append(name)
    return names


def is_meaningful(path: Path) -> bool:
    name = path.name
    text = str(path)
    return name in {"dataset_admission.jsonl", "dataset_request_events.jsonl", "dataset_batch_summary.json", "item_result.json", "minimal_pipeline_events.jsonl"} or "/state/" in text and name.endswith(".json") or "/evaluation/" in text and name.endswith(".json") or "/product_bundle/" in text and name == "manifest.json"


def terminal_target_count(status_snapshot: dict[str, Any]) -> int:
    batch = status_snapshot.get("batch")
    if not isinstance(batch, dict):
        return 0
    summary = batch.get("summary")
    if isinstance(summary, dict):
        video_count = summary.get("video_count")
        try:
            if video_count is not None and int(video_count) > 0:
                return int(video_count)
        except (TypeError, ValueError):
            pass
    try:
        return int(batch.get("admission_rows") or 0)
    except (TypeError, ValueError):
        return 0


def snapshot(root: Path, out_dir: Path, reason: str, changed_paths: list[str], *, include_services: bool) -> dict[str, Any]:
    admission = jsonl_rows(root / "dataset_admission.jsonl")
    events = jsonl_rows(root / "dataset_request_events.jsonl")
    summary = load_json(root / "dataset_batch_summary.json") or {}
    value: dict[str, Any] = {
        "schema": "v22_batch_status_snapshot.v1",
        "captured_at": utc_now(),
        "reason": reason,
        "changed_paths": changed_paths[-100:],
        "batch": {
            "summary": {key: summary.get(key) for key in ("status", "video_count", "admitted_count", "terminal_count", "status_counts", "complete_pipeline_active_limit", "updated_at")},
            "admission_rows": len(admission),
            "admission_status_counts": dict(Counter(str(row.get("status") or "unknown") for row in admission)),
            "request_event_rows": len(events),
            "request_event_status_counts": dict(Counter(str(row.get("status") or "unknown") for row in events)),
            "results": result_inventory(root),
            "attempt_count": sum(1 for _ in root.glob("items/item_*/attempt_*")),
        },
        "processes": process_inventory(root),
        "host": host_snapshot(),
        "service_log_inventory": service_log_inventory(),
    }
    if include_services:
        value["services"] = service_snapshot()
    write_path = out_dir / "status_snapshots.jsonl"
    append_jsonl(write_path, value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service-log-root", action="append", type=Path, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_meta = {"schema": "v22_batch_status_monitor.v1", "started_at": utc_now(), "run_root": str(root), "output_dir": str(out_dir), "pid": os.getpid(), "hostname": socket.gethostname(), "argv": sys.argv}
    watcher = Inotify()
    watcher.add_tree(root)
    for service_root in args.service_log_root:
        watcher.add_tree(service_root.expanduser().resolve())
    monitor_meta["watch_directory_count"] = len(watcher.paths)
    start_path = out_dir / ("monitor_start.json" if not (out_dir / "monitor_start.json").exists() else f"monitor_start_{os.getpid()}.json")
    start_path.write_text(json.dumps(monitor_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_jsonl(out_dir / "monitor_events.jsonl", {"captured_at": utc_now(), "event": "monitor_started", "metadata": monitor_meta})
    stop = {"value": False}

    def handle_signal(signum: int, _frame: Any) -> None:
        stop["value"] = True
        append_jsonl(out_dir / "monitor_events.jsonl", {"captured_at": utc_now(), "event": "monitor_signal", "signal": signum})

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    snapshot(root, out_dir, "startup", [], include_services=True)
    meaningful_count = 0
    while not stop["value"]:
        data = os.read(watcher.fd, 262144)
        if not data:
            break
        meaningful_paths: list[str] = []
        event_payloads: list[dict[str, Any]] = []
        offset = 0
        while offset + EVENT_STRUCT.size <= len(data):
            wd, mask, cookie, name_len = EVENT_STRUCT.unpack_from(data, offset)
            offset += EVENT_STRUCT.size
            raw_name = data[offset:offset + name_len]
            offset += name_len
            name = raw_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            parent = watcher.paths.get(wd)
            path = parent / name if parent is not None and name else parent
            path_text = str(path) if path is not None else f"<wd:{wd}>"
            event_payloads.append({"captured_at": utc_now(), "event": "filesystem", "watch_descriptor": wd, "path": path_text, "mask": mask, "mask_names": event_names(mask), "cookie": cookie})
            if path is not None and is_meaningful(path):
                meaningful_paths.append(path_text)
            if parent is not None and mask & IN_ISDIR and mask & (IN_CREATE | IN_MOVED_TO) and path is not None and path.name not in SKIP_DIRS and not path.name.startswith(".feishu_"):
                watcher.add_tree(path)
        append_jsonl_batch(out_dir / "monitor_events.jsonl", event_payloads)
        if meaningful_paths:
            meaningful_count += 1
            include_services = meaningful_count == 1 or meaningful_count % 32 == 0 or any(path.endswith("item_result.json") or path.endswith("dataset_request_events.jsonl") for path in meaningful_paths)
            current = snapshot(root, out_dir, "filesystem_change", meaningful_paths, include_services=include_services)
            target_count = terminal_target_count(current)
            completed = int(current["batch"]["results"]["count"])
            if target_count > 0 and completed >= target_count:
                append_jsonl(out_dir / "monitor_events.jsonl", {"captured_at": utc_now(), "event": "monitor_terminal", "target_count": target_count, "completed": completed})
                snapshot(root, out_dir, "terminal", meaningful_paths, include_services=True)
                break
    watcher.close()
    final = snapshot(root, out_dir, "monitor_stopped", [], include_services=True)
    monitor_meta.update({"stopped_at": utc_now(), "final_snapshot": final})
    (out_dir / f"monitor_final_{os.getpid()}.json").write_text(json.dumps(monitor_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
