#!/usr/bin/env python3
"""Read-only live batch dashboard for the Ego model services on A800."""
from __future__ import annotations

import argparse
import concurrent.futures
import curses
import http.client
import json
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping


DEFAULT_SNAPSHOT_URLS = {
    "unidepth-gpu0": "http://127.0.0.1:29000/-/batch-snapshot",
    "unidepth-gpu5": "http://127.0.0.1:28005/-/batch-snapshot",
    "hands": "http://127.0.0.1:28001/-/batch-snapshot",
    "wilor": "http://127.0.0.1:28004/-/batch-snapshot",
    "hawor": "http://127.0.0.1:28003/hawor.infer_tracks/-/batch-snapshot",
    "infiller": "http://127.0.0.1:28003/hawor_infiller.fill/-/batch-snapshot",
    "droid-29002": "http://127.0.0.1:29002/-/batch-snapshot",
    "droid-28012": "http://127.0.0.1:28012/-/batch-snapshot",
    "droid-28022": "http://127.0.0.1:28022/-/batch-snapshot",
    "droid-28007": "http://127.0.0.1:28007/-/batch-snapshot",
    "droid-28017": "http://127.0.0.1:28017/-/batch-snapshot",
    "droid-28027": "http://127.0.0.1:28027/-/batch-snapshot",
    "cosmos3": "http://127.0.0.1:28006/-/batch-snapshot",
}
DEFAULT_RAY_METRICS = {
    "unidepth-gpu0": "http://127.0.0.1:26007/metrics",
    "hands": "http://127.0.0.1:27007/metrics",
    "wilor": "http://127.0.0.1:27207/metrics",
    "hawor": "http://127.0.0.1:26607/metrics",
    "infiller": "http://127.0.0.1:26607/metrics",
    "droid-29002": "http://127.0.0.1:26407/metrics",
    "droid-28012": "http://127.0.0.1:26427/metrics",
    "droid-28022": "http://127.0.0.1:26447/metrics",
    "droid-28007": "http://127.0.0.1:30007/metrics",
    "droid-28017": "http://127.0.0.1:30027/metrics",
    "droid-28027": "http://127.0.0.1:30047/metrics",
    "cosmos3": "http://127.0.0.1:26810/metrics",
}
RAY_QUEUE_FAMILY = "ray_serve_deployment_queued_queries"
SNAPSHOT_NOT_DEPLOYED = "N/A snapshot not deployed"
DROID_CAPACITY = 48
DROID_LANE_CAPACITY = 8


@dataclass(frozen=True)
class SourceResult:
    value: Any | None
    error: str | None = None
    sampled_at: float | None = None
    snapshot_unavailable: bool = False

    @property
    def observed_at(self) -> float | None:
        """Wall-clock time this individual source was observed."""
        return self.sampled_at


@dataclass(frozen=True)
class RaySource:
    url: str
    family: str | None = None
    labels: Mapping[str, str] | None = None


DEFAULT_RAY_FILTERS = {
    "unidepth-gpu0": {"application": "ego-unidepth", "deployment": "unidepth.infer"},
    "hands": {"application": "ego-hands", "deployment": "hands"},
    "wilor": {"application": "ego-wilor", "deployment": "wilor"},
    "hawor": {"application": "hawor-infer-tracks", "deployment": "hawor.infer_tracks"},
    "infiller": {"application": "hawor-infiller-fill", "deployment": "hawor_infiller.fill"},
    "droid-29002": {"application": "ego-droid-service", "deployment": "droid"},
    "droid-28012": {"application": "ego-droid-service", "deployment": "droid"},
    "droid-28022": {"application": "ego-droid-service", "deployment": "droid"},
    "droid-28007": {"application": "ego-droid-service", "deployment": "droid"},
    "droid-28017": {"application": "ego-droid-service", "deployment": "droid"},
    "droid-28027": {"application": "ego-droid-service", "deployment": "droid"},
    "cosmos3": {"application": "cosmos3", "deployment": "cosmos3.reason"},
}


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def get_json(url: str, timeout_s: float) -> SourceResult:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return SourceResult(json.loads(response.read().decode("utf-8")), sampled_at=time.time())
    except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError, ValueError, UnicodeDecodeError) as exc:
        return SourceResult(None, _error(exc), time.time())


def _http_error(exc: urllib.error.HTTPError) -> str:
    return f"HTTP {exc.code} {exc.reason}".rstrip()


def _is_snapshot_collection(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("snapshots"), list)


def get_snapshot(url: str, timeout_s: float) -> SourceResult:
    """Fetch a snapshot without mistaking an old router for a dead service."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            try:
                value = json.loads(response.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return SourceResult(None, SNAPSHOT_NOT_DEPLOYED, time.time(), snapshot_unavailable=True)
            if not _is_snapshot_collection(value):
                return SourceResult(None, SNAPSHOT_NOT_DEPLOYED, time.time(), snapshot_unavailable=True)
            return SourceResult(value, sampled_at=time.time())
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404, 405):
            return SourceResult(None, SNAPSHOT_NOT_DEPLOYED, time.time(), snapshot_unavailable=True)
        return SourceResult(None, _http_error(exc), time.time())
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
        return SourceResult(None, _error(exc), time.time())


def _get_text(url: str, timeout_s: float) -> SourceResult:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return SourceResult(response.read().decode("utf-8"), sampled_at=time.time())
    except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, OSError, TimeoutError, UnicodeDecodeError) as exc:
        return SourceResult(None, _error(exc), time.time())


def _collect_futures(futures: Mapping[concurrent.futures.Future[SourceResult], str]) -> dict[str, SourceResult]:
    collected: dict[str, SourceResult] = {}
    for future, name in futures.items():
        try:
            collected[name] = future.result()
        except BaseException as exc:
            collected[name] = SourceResult(None, _error(exc), time.time())
    return collected


def fetch_snapshots(urls: Mapping[str, str], timeout_s: float = 0.6) -> dict[str, SourceResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        return _collect_futures({pool.submit(get_snapshot, url, timeout_s): name for name, url in urls.items()})


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            metric_and_labels, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
            if "{" not in metric_and_labels:
                samples.append((metric_and_labels, {}, value))
                continue
            metric, raw_labels = metric_and_labels.split("{", 1)
            labels: dict[str, str] = {}
            for entry in _split_labels(raw_labels.rsplit("}", 1)[0]):
                key, raw = entry.split("=", 1)
                labels[key] = json.loads(raw)
            samples.append((metric, labels, value))
        except (ValueError, json.JSONDecodeError):
            continue
    return samples


def _split_labels(text: str) -> list[str]:
    entries, start, escaped, quoted = [], 0, False, False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char == "," and not quoted:
            entries.append(text[start:index])
            start = index + 1
    entries.append(text[start:])
    return [entry for entry in entries if entry]


def queue_metric(samples: Iterable[tuple[str, Mapping[str, str], float]], family: str | None, labels: Mapping[str, str]) -> SourceResult:
    if not family:
        return SourceResult(None, "N/A (metric family unset)")
    family_samples = [(sample_labels, value) for metric, sample_labels, value in samples if metric == family]
    if not family_samples:
        return SourceResult(None, "N/A (metric family/series absent)")
    if not labels:
        return SourceResult(None, "N/A (service label unset)")
    matched = [value for sample_labels, value in family_samples if all(sample_labels.get(k) == v for k, v in labels.items())]
    if not matched:
        return SourceResult(None, "N/A (metric family/series absent)")
    return SourceResult(sum(matched), sampled_at=time.time())


def queue_metric_source(source: SourceResult, config: RaySource) -> SourceResult:
    if source.error:
        return SourceResult(None, source.error, source.sampled_at)
    if not isinstance(source.value, str):
        return SourceResult(None, "N/A (metrics response unavailable)", source.sampled_at)
    metric = queue_metric(parse_prometheus(source.value), config.family, dict(config.labels or {}))
    return SourceResult(metric.value, metric.error, source.sampled_at)


def ray_source_config(urls: Mapping[str, str], declarations: Iterable[str]) -> dict[str, RaySource]:
    """Parse per-source ``NAME=FAMILY:label=value,label=value`` declarations."""
    sources = {
        name: RaySource(url, RAY_QUEUE_FAMILY, DEFAULT_RAY_FILTERS[name])
        if name in DEFAULT_RAY_FILTERS else RaySource(url)
        for name, url in urls.items()
    }
    for declaration in declarations:
        name, separator, spec = declaration.partition("=")
        family, colon, raw_labels = spec.partition(":")
        if not separator or not name or not family or name not in sources:
            raise argparse.ArgumentTypeError("ray queue source must be NAME=FAMILY:label=value[,label=value]")
        labels = _label_options(raw_labels.split(",")) if colon and raw_labels else {}
        sources[name] = RaySource(sources[name].url, family, labels)
    return sources


def source_age(source: SourceResult, *, now: float | None = None, stale_after_s: float = 3.0) -> str:
    if source.sampled_at is None:
        return "N/A"
    age = max(0.0, (time.time() if now is None else now) - source.sampled_at)
    return "STALE" if age > stale_after_s else f"fresh {age:.1f}s"


def freshness(snapshot: Mapping[str, Any], *, now_ns: int | None = None, stale_after_s: float = 3.0) -> str:
    sampled = snapshot.get("sampled_at_unix_ns")
    if not isinstance(sampled, int):
        return "N/A"
    age_s = ((now_ns if now_ns is not None else time.time_ns()) - sampled) / 1_000_000_000.0
    return "STALE" if age_s > stale_after_s else f"fresh {max(0.0, age_s):.1f}s"


def _rates(record: Mapping[str, Any] | None) -> tuple[str, str]:
    if not isinstance(record, Mapping) or not isinstance(record.get("duration_s"), (int, float)) or record["duration_s"] <= 0:
        return "N/A", "N/A"
    duration = float(record["duration_s"])
    req = record.get("request_count")
    image = record.get("image_item_count")
    return (
        f"{float(req) / duration:.2f}" if isinstance(req, (int, float)) else "N/A",
        f"{float(image) / duration:.2f}" if isinstance(image, (int, float)) else "N/A",
    )


def throughput(snapshot: Mapping[str, Any]) -> tuple[str, str]:
    terminal, success = snapshot.get("last_terminal"), snapshot.get("last_success")
    if not isinstance(terminal, Mapping) or not terminal.get("success") or not isinstance(success, Mapping):
        return "N/A", "N/A"
    if terminal.get("completed_monotonic_ns", terminal.get("completed_at_unix_ns")) != success.get("completed_monotonic_ns", success.get("completed_at_unix_ns")):
        return "N/A", "N/A"
    return _rates(success)


def _event_age(record: Mapping[str, Any] | None, now_ns: int) -> str:
    if not isinstance(record, Mapping) or not isinstance(record.get("completed_at_unix_ns"), int):
        return "N/A"
    age = max(0.0, (now_ns - record["completed_at_unix_ns"]) / 1_000_000_000.0)
    return "STALE" if age > 3.0 else f"{age:.1f}s"


def _current_elapsed(active: Mapping[str, Any], now_ns: int) -> str:
    started = active.get("oldest_started_unix_ns")
    if not isinstance(started, int):
        return "N/A"
    return f"{max(0.0, (now_ns - started) / 1_000_000_000.0):.2f}s"


def droid_sqlite_snapshot(path: str, timeout_s: float = 0.6) -> SourceResult:
    """Read lease state through SQLite ``mode=ro`` and normalize zero-session lanes."""
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=max(0.01, timeout_s))
        try:
            connection.execute("BEGIN")
            inflight = {str(url): int(count) for url, count in connection.execute("SELECT replica_url, active_sessions FROM replica_inflight")}
            affinity = {str(url): int(count) for url, count in connection.execute("SELECT replica_url, COUNT(*) FROM droid_session_affinity GROUP BY replica_url")}
        finally:
            connection.close()
        normalized_affinity = {lane: affinity.get(lane, 0) for lane in inflight}
        unknown = sorted(set(affinity) - set(inflight))
        mismatch = {lane: {"inflight": count, "affinity": normalized_affinity[lane]} for lane, count in inflight.items() if count != normalized_affinity[lane]}
        value = {"lanes": inflight, "affinity": normalized_affinity, "total": sum(inflight.values()), "unknown_lanes": unknown, "mismatch": mismatch}
        if unknown:
            return SourceResult(value, f"sqlite-unknown-lane: {', '.join(unknown)}", time.time())
        if mismatch:
            return SourceResult(value, "sqlite-inconsistent", time.time())
        return SourceResult(value, sampled_at=time.time())
    except sqlite3.Error as exc:
        return SourceResult(None, f"sqlite-{type(exc).__name__}: {exc}", time.time())


def gpu_snapshot(timeout_s: float = 0.6) -> SourceResult:
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout_s)
        gpus = []
        for line in result.stdout.splitlines():
            columns = [cell.strip() for cell in line.split(",")]
            if len(columns) == 5:
                gpus.append({"index": int(columns[0]), "name": columns[1], "util": float(columns[2]), "used": float(columns[3]), "total": float(columns[4])})
        return SourceResult(gpus, sampled_at=time.time())
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return SourceResult(None, _error(exc), time.time())


def bar(numerator: int | float | None, denominator: int | float | None, width: int = 20) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "N/A"
    filled = max(0, min(width, round(width * float(numerator) / float(denominator))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def reduce_scroll(offset: int, key: int, content_rows: int, viewport_rows: int) -> int:
    maximum = max(0, content_rows - max(1, viewport_rows))
    page = max(1, viewport_rows - 1)
    if key == curses.KEY_UP:
        offset -= 1
    elif key == curses.KEY_DOWN:
        offset += 1
    elif key == curses.KEY_PPAGE:
        offset -= page
    elif key == curses.KEY_NPAGE:
        offset += page
    elif key == curses.KEY_HOME:
        offset = 0
    elif key == curses.KEY_END:
        offset = maximum
    return max(0, min(maximum, offset))


def content_viewport_rows(*, height: int, gpu_rows: int) -> int:
    """One source of truth for draw and navigation: title + gpu rows + footer."""
    return max(1, height - (1 + gpu_rows + 1))


def _snapshots(sources: Mapping[str, SourceResult]) -> list[tuple[str, SourceResult, Mapping[str, Any]]]:
    rows = []
    for name, source in sources.items():
        if isinstance(source.value, Mapping):
            for snap in source.value.get("snapshots", []):
                if isinstance(snap, Mapping):
                    rows.append((name, source, snap))
    return rows


def _service_line(label: str, source: SourceResult, snap: Mapping[str, Any], now_ns: int, *, record: Mapping[str, Any] | None = None, capacity_key: str = "max_batch_size", image_rate_available: bool = True, ray_queue: SourceResult | None = None) -> str:
    capacity, active, status = snap.get("capacity", {}), snap.get("active", {}), snap.get("adapter_status", {})
    current = active.get("request_count") if isinstance(active, Mapping) else None
    cap = capacity.get(capacity_key) if isinstance(capacity, Mapping) else None
    record = record if record is not None else snap.get("last_success")
    terminal = snap.get("last_terminal")
    req_s, img_s = _rates(record) if not isinstance(terminal, Mapping) or terminal.get("success") else ("N/A", "N/A")
    if not image_rate_available:
        img_s = "N/A"
    q = ray_queue.error if ray_queue and ray_queue.error else str(ray_queue.value) if ray_queue and ray_queue.value is not None else "N/A"
    admitted = status.get("admitted_pending", "N/A") if isinstance(status, Mapping) else "N/A"
    return (f"{label:<22} endpoint {source_age(source):<10} {bar(current if isinstance(current, (int, float)) else None, cap if isinstance(cap, (int, float)) else None)} "
            f"{current if current is not None else 'N/A'}/{cap if cap is not None else 'N/A'} ADM {admitted} "
            f"RUN {_current_elapsed(active, now_ns) if isinstance(active, Mapping) else 'N/A'} last {_event_age(record, now_ns)} req/s {req_s} img/s {img_s} Qserve {q}")


def snapshot_lines(sources: Mapping[str, SourceResult], droid_db: SourceResult, ray_queues: Mapping[str, SourceResult], *, now_ns: int | None = None) -> list[str]:
    now_ns = time.time_ns() if now_ns is None else now_ns
    lines = ["SERVICE CAPACITY / FORWARD (snapshot routes are read-only)"]
    if droid_db.error and not isinstance(droid_db.value, Mapping):
        lines.append(f"DROID sessions N/A/{DROID_CAPACITY} {droid_db.error}")
    else:
        db = droid_db.value if isinstance(droid_db.value, Mapping) else {}
        total = db.get("total")
        lines.append(f"DROID sessions {bar(total if isinstance(total, (int, float)) else None, DROID_CAPACITY)} {total if total is not None else 'N/A'}/{DROID_CAPACITY} {droid_db.error or 'OK'}")
        for lane, count in sorted((db.get("lanes") or {}).items()):
            lines.append(f"  DROID lane {lane:<28} {bar(count, DROID_LANE_CAPACITY)} {count}/{DROID_LANE_CAPACITY}")
    rows = _snapshots(sources)
    unidepth = [(name, source, snap) for name, source, snap in rows if snap.get("service") == "unidepth.infer"]
    if unidepth:
        active = sum(int(snap.get("active", {}).get("request_count", 0)) for _name, _source, snap in unidepth)
        cap = sum(int(snap.get("capacity", {}).get("max_batch_size", 0) or 0) for _name, _source, snap in unidepth)
        lines.append(f"UniDepth total          {bar(active, cap)} {active}/{cap}")
        for name, source, snap in unidepth:
            lines.append(_service_line(f"  UniDepth {name}", source, snap, now_ns, ray_queue=ray_queues.get(name)))
    for name, source, snap in rows:
        service = str(snap.get("service", name))
        if service == "unidepth.infer":
            continue
        if service == "droid":
            active = snap.get("active", {})
            capacity = snap.get("capacity", {})
            current = active.get("request_count") if isinstance(active, Mapping) else None
            cap = capacity.get("max_batch_size") if isinstance(capacity, Mapping) else None
            terminal = snap.get("last_terminal")
            terminal_text = (
                f"terminal ERR {terminal.get('error_type', 'UnknownError')}"
                if isinstance(terminal, Mapping) and not terminal.get("success")
                else f"terminal {_event_age(terminal, now_ns)}"
            )
            lines.append(
                f"DROID RUN {current if current is not None else 'N/A'}/{cap if cap is not None else 'N/A'} "
                f"FWD {_current_elapsed(active, now_ns) if isinstance(active, Mapping) else 'N/A'} {terminal_text}"
            )
            fnet = snap.get("last_success_by_operation", {}).get("push_frame.fnet") if isinstance(snap.get("last_success_by_operation"), Mapping) else None
            finalize = snap.get("last_success_by_operation", {}).get("finalize") if isinstance(snap.get("last_success_by_operation"), Mapping) else None
            req_s, img_s = _rates(fnet)
            lines.append(f"DROID FNet {name:<10} endpoint {source_age(source):<10} last {_event_age(fnet, now_ns)} req/s {req_s} img/s {img_s}")
            if isinstance(finalize, Mapping):
                lines.append(f"  DROID finalize {name:<8} last {_event_age(finalize, now_ns)} duration {finalize.get('duration_s', 'N/A')}")
        elif service == "cosmos3.reason":
            lines.append(_service_line("Cosmos", source, snap, now_ns, capacity_key="max_ongoing_requests", image_rate_available=False, ray_queue=ray_queues.get(name)))
        else:
            lines.append(_service_line(service, source, snap, now_ns, ray_queue=ray_queues.get(name)))
    for name, source in sorted(sources.items()):
        if source.error:
            if source.snapshot_unavailable:
                lines.append(f"{name:<22} {SNAPSHOT_NOT_DEPLOYED}")
            else:
                lines.append(f"{name:<22} DOWN {source.error}")
    for name, queue in sorted(ray_queues.items()):
        if queue.error:
            lines.append(f"Qserve {name:<16} {queue.error} ({source_age(queue)})")
        else:
            lines.append(f"Qserve {name:<16} {queue.value} ({source_age(queue)})")
    return lines


@dataclass(frozen=True)
class DashboardState:
    """A complete collection round published as one immutable reference."""

    gpus: SourceResult
    sources: Mapping[str, SourceResult]
    droid: SourceResult
    ray_queues: Mapping[str, SourceResult]
    started_monotonic: float | None = None
    completed_monotonic: float | None = None
    cycle: int = 0
    collector_error: str | None = None


def _freeze(value: Any) -> Any:
    """Recursively remove mutability from values shared across UI/collector threads."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set) or isinstance(value, frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


def _freeze_source(source: SourceResult) -> SourceResult:
    return SourceResult(_freeze(source.value), source.error, source.sampled_at, source.snapshot_unavailable)


def _empty_state(error: str = "not sampled") -> DashboardState:
    return DashboardState(
        SourceResult(None, error), MappingProxyType({}), SourceResult(None, error), MappingProxyType({}),
    )


def _poll_ms(args: argparse.Namespace) -> int:
    return int(getattr(args, "poll_ms", getattr(args, "refresh_ms", 1000)))


def _ui_ms(args: argparse.Namespace) -> int:
    return int(getattr(args, "ui_ms", 100))


def next_cadence_start(*, previous_scheduled: float, interval_s: float, completed: float) -> float:
    """Return the next fixed-rate start, dropping missed slots after an overrun."""
    return max(previous_scheduled + interval_s, completed)


class Collector:
    """The sole writer of a completed dashboard state.

    One round submits all external observations to one bounded executor.  The
    curses thread only reads the single state reference published afterwards.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        collect_round: Callable[[float], DashboardState] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.args = args
        self._clock = clock
        self._collect_round_override = collect_round
        self._state = _empty_state()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycle = 0

    def latest(self) -> DashboardState:
        with self._state_lock:
            return self._state

    def publish(self, state: DashboardState) -> DashboardState:
        """Freeze a full round, then replace the one UI-visible reference."""
        frozen = DashboardState(
            _freeze_source(state.gpus),
            MappingProxyType({name: _freeze_source(source) for name, source in state.sources.items()}),
            _freeze_source(state.droid),
            MappingProxyType({name: _freeze_source(source) for name, source in state.ray_queues.items()}),
            state.started_monotonic,
            state.completed_monotonic,
            state.cycle,
            state.collector_error,
        )
        with self._state_lock:
            self._state = frozen
        return frozen

    def collect_once(self, started_monotonic: float | None = None) -> DashboardState:
        started = self._clock() if started_monotonic is None else started_monotonic
        if self._collect_round_override is not None:
            state = self._collect_round_override(started)
        else:
            state = self._collect_round(started)
        return self.publish(state)

    def _collect_round(self, started: float) -> DashboardState:
        timeout_s = float(self.args.timeout)
        snapshots = self.args.snapshot_urls
        ray_sources = self.args.ray_sources
        # GPU, SQLite, every snapshot, and every metrics endpoint enter this
        # one executor together. Parsing metrics is local work after the I/O.
        futures: dict[concurrent.futures.Future[SourceResult], tuple[str, str | None]] = {}
        worker_count = max(1, 2 + len(snapshots) + len(ray_sources))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="service-batch-sample") as pool:
            futures[pool.submit(gpu_snapshot, timeout_s)] = ("gpu", None)
            futures[pool.submit(droid_sqlite_snapshot, self.args.droid_db, timeout_s)] = ("droid", None)
            for name, url in snapshots.items():
                futures[pool.submit(get_snapshot, url, timeout_s)] = ("snapshot", name)
            for name, spec in ray_sources.items():
                futures[pool.submit(_get_text, spec.url, timeout_s)] = ("ray", name)
            completed: dict[tuple[str, str | None], SourceResult] = {}
            for future, identity in futures.items():
                try:
                    completed[identity] = future.result()
                except BaseException as exc:
                    completed[identity] = SourceResult(None, _error(exc), time.time())
        raw_ray = {name: completed[("ray", name)] for name in ray_sources}
        return DashboardState(
            completed[("gpu", None)],
            {name: completed[("snapshot", name)] for name in snapshots},
            completed[("droid", None)],
            {name: queue_metric_source(raw_ray[name], spec) for name, spec in ray_sources.items()},
            started,
            self._clock(),
            self._cycle,
        )

    def request_sample(self) -> None:
        self._sample_requested.set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="service-batch-collector", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        interval_s = _poll_ms(self.args) / 1000.0
        next_start = self._clock()
        while not self._stop_event.is_set():
            # A requested post-pause sample wakes promptly. It does not move
            # the established periodic schedule unless the request overruns it.
            requested = False
            while True:
                remaining = next_start - self._clock()
                if remaining <= 0:
                    self._sample_requested.clear()
                    break
                if self._sample_requested.is_set():
                    self._sample_requested.clear()
                    requested = True
                    break
                if self._stop_event.wait(min(remaining, 0.05)):
                    return
            if self._stop_event.is_set():
                return
            started = self._clock()
            try:
                self._cycle += 1
                state = self.collect_once(started)
            except BaseException as exc:
                # Do not let an unexpected collector failure turn into a quiet
                # dead thread. The error is a normal, renderable state.
                state = self.publish(DashboardState(
                    SourceResult(None, _error(exc), time.time()), {}, SourceResult(None, _error(exc), time.time()), {},
                    started, self._clock(), self._cycle, _error(exc),
                ))
            # Skip missed cadence slots on an overrun instead of adding the
            # work duration to the polling interval.
            completed = self._clock()
            next_start = max(next_start, completed) if requested else next_cadence_start(
                previous_scheduled=next_start, interval_s=interval_s, completed=completed,
            )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        # urllib's socket timeout bounds a blocked read but a locally streaming
        # response can finish a little later. Keep a finite join budget while
        # still joining non-daemon executor workers rather than leaking them.
        thread.join(timeout=max(float(self.args.timeout), 0.01) + 3.0)
        if thread.is_alive():
            raise RuntimeError("collector did not stop within source timeout bound")
        self._thread = None


class Dashboard:
    def __init__(self, args: argparse.Namespace, *, collector: Collector | None = None) -> None:
        self.args, self.offset = args, 0
        self.collector = collector or Collector(args)
        self.displayed_state = self.collector.latest()
        self._viewport = 1
        self.mouse_capture = False
        self.paused = False

    def _sync_displayed_state(self) -> DashboardState:
        if not self.paused:
            self.displayed_state = self.collector.latest()
        return self.displayed_state

    def _raw_gpu_rows(self, state: DashboardState) -> int:
        return 1 if state.gpus.error else len(state.gpus.value or [])

    def _gpu_rows(self, height: int, state: DashboardState) -> int:
        return min(self._raw_gpu_rows(state), max(0, height - 3))

    def _layout(self, height: int, state: DashboardState) -> int:
        self._viewport = content_viewport_rows(height=height, gpu_rows=self._gpu_rows(height, state))
        return self._viewport

    @staticmethod
    def _write(screen: Any, height: int, width: int, row: int, text: str, attrs: int = 0) -> None:
        if not (0 <= row < height) or width <= 0:
            return
        try:
            screen.addnstr(row, 0, text, max(0, width - 1), attrs)
        except curses.error:
            return

    def draw(self, screen: Any) -> None:
        state = self._sync_displayed_state()
        height, width = screen.getmaxyx()
        screen.erase()
        if height <= 0:
            return
        footer_row = height - 1
        if height > 1:
            self._write(screen, height, width, 0, "Ego Service Batch TUI | q quit | p/Space freeze | m mouse | arrows/PgUp/PgDn/Home/End scroll", curses.A_BOLD)
        row = 1 if height > 1 else 0
        gpu_rows = self._gpu_rows(height, state)
        if state.gpus.error:
            if gpu_rows:
                self._write(screen, height, width, row, f"GPU DOWN: {state.gpus.error}")
                row += 1
        else:
            for gpu in (state.gpus.value or [])[:gpu_rows]:
                self._write(screen, height, width, row, f"GPU{gpu['index']} {gpu['name'][:18]:18} util {bar(gpu['util'], 100, 18)} {gpu['util']:5.1f}% mem {bar(gpu['used'], gpu['total'], 18)} {gpu['used']:.0f}/{gpu['total']:.0f} MiB")
                row += 1
        lines = snapshot_lines(state.sources, state.droid, state.ray_queues)
        viewport = self._layout(height, state)
        self.offset = reduce_scroll(self.offset, curses.KEY_RESIZE, len(lines), viewport)
        for line in lines[self.offset:self.offset + viewport]:
            if row >= footer_row:
                break
            self._write(screen, height, width, row, line)
            row += 1
        age = "N/A" if state.completed_monotonic is None else f"{max(0.0, time.monotonic() - state.completed_monotonic):.1f}s"
        pause_state = "PAUSED | " if self.paused else ""
        mouse_state = "mouse ON" if self.mouse_capture else "mouse OFF"
        collector_error = f" | collector {state.collector_error}" if state.collector_error else ""
        self._write(screen, height, width, footer_row, f"{pause_state}{mouse_state} | render {_ui_ms(self.args)}ms | poll {_poll_ms(self.args)}ms | sample age {age} | scroll {self.offset}/{max(0, len(lines) - viewport)}{collector_error}", curses.A_REVERSE)
        screen.refresh()

    def set_mouse_capture(self, enabled: bool) -> None:
        self.mouse_capture = enabled
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS if enabled else 0)
        except curses.error:
            pass

    def handle_key(self, key: int, height: int, *, mouse_buttons: int | None = None) -> None:
        if key in (ord("m"), ord("M")):
            self.set_mouse_capture(not self.mouse_capture)
            return
        if key in (ord("p"), ord("P"), ord(" ")):
            if not self.paused:
                self._sync_displayed_state()
                self.paused = True
            else:
                self.paused = False
                self.displayed_state = self.collector.latest()
                self.collector.request_sample()
            return
        if key == curses.KEY_MOUSE and mouse_buttons is not None:
            if mouse_buttons & curses.BUTTON4_PRESSED:
                key = curses.KEY_UP
            elif mouse_buttons & curses.BUTTON5_PRESSED:
                key = curses.KEY_DOWN
        state = self.displayed_state
        lines = snapshot_lines(state.sources, state.droid, state.ray_queues)
        self.offset = reduce_scroll(self.offset, key, len(lines), self._layout(height, state))

    def run(self, screen: Any) -> None:
        curses.curs_set(0)
        screen.timeout(_ui_ms(self.args))
        # Mouse capture is OFF by default so the terminal handles mouse natively
        # and text selection/copy works out of the box. Press 'm' to opt into
        # scroll-wheel mode (use Shift+drag to select while it is on).
        self.set_mouse_capture(False)
        self.collector.start()
        try:
            while True:
                self.draw(screen)
                key = screen.getch()
                if key in (ord("q"), ord("Q")):
                    return
                if key == curses.KEY_MOUSE:
                    try:
                        _id, _x, _y, _z, buttons = curses.getmouse()
                    except curses.error:
                        continue
                    self.handle_key(key, screen.getmaxyx()[0], mouse_buttons=buttons)
                else:
                    self.handle_key(key, screen.getmaxyx()[0])
        finally:
            self.collector.stop()

def _mapping_options(values: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for value in values:
        key, separator, target = value.partition("=")
        if not separator or not key or not target:
            raise argparse.ArgumentTypeError("mapping must use NAME=VALUE")
        mapped[key] = target
    return mapped


def _label_options(values: Iterable[str]) -> dict[str, str]:
    return _mapping_options([value for value in values if value])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-ms", "--refresh-ms", dest="poll_ms", type=int, default=1000,
                        help="collector start-to-start cadence (legacy --refresh-ms alias)")
    parser.add_argument("--ui-ms", type=int, default=100, help="curses input/render cadence")
    parser.add_argument("--timeout", type=float, default=0.6)
    parser.add_argument("--droid-db", default="/tmp/ego_lane_dispatcher_leases.sqlite3")
    parser.add_argument("--snapshot-url", action="append", default=[], metavar="NAME=URL")
    parser.add_argument("--ray-metrics-url", action="append", default=[], metavar="NAME=URL")
    parser.add_argument("--ray-queue-source", action="append", default=[], metavar="NAME=FAMILY:label=value[,label=value]")
    args = parser.parse_args()
    args.snapshot_urls = {**DEFAULT_SNAPSHOT_URLS, **_mapping_options(args.snapshot_url)}
    args.ray_sources = ray_source_config({**DEFAULT_RAY_METRICS, **_mapping_options(args.ray_metrics_url)}, args.ray_queue_source)
    if args.poll_ms < 100:
        parser.error("--poll-ms/--refresh-ms must be at least 100")
    if args.ui_ms < 25:
        parser.error("--ui-ms must be at least 25")
    args.refresh_ms = args.poll_ms  # compatibility for external callers
    curses.wrapper(Dashboard(args).run)


if __name__ == "__main__":
    main()
