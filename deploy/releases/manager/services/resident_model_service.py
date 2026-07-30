#!/usr/bin/env python3
"""One-process resident HTTP service and compatibility-window coalescer.

The server deliberately uses the Python standard library. HTTP request threads
only enqueue work; exactly one scheduler thread invokes the resident model
adapter. Model adapters own native tensor construction and output splitting.

The upload endpoint is content-addressed. An inference request can only consume
files that were PUT to the service and named by SHA256 in its artifact envelope.
Original caller paths are never used as server input paths.
"""
from __future__ import annotations

import copy
import hashlib
import http.server
import json
import os
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

DEFAULT_WAIT_S = 20.0
DEFAULT_PENDING_LIMIT = 1024


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _replace_paths(value: Any, mapping: dict[str, str], *, base: Path | None = None) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(item, mapping, base=base) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_paths(item, mapping, base=base) for item in value]
    if not isinstance(value, str):
        return value
    if value in mapping:
        return mapping[value]
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        candidate = str((base / path).resolve())
        if candidate in mapping:
            return mapping[candidate]
    return value


class ArtifactStore:
    """Content-addressed input store plus request-local materialization."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.cas = self.root / "cas"
        self.jobs = self.root / "jobs"
        self.cas.mkdir(parents=True, exist_ok=True)
        self.jobs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put_stream(self, artifact_id: str, stream: Any, expected_bytes: int | None = None) -> dict[str, Any]:
        artifact_id = str(artifact_id).lower()
        if len(artifact_id) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_id):
            raise ValueError("artifact id must be a SHA256 hex digest")
        target = self.cas / artifact_id
        temporary = self.cas / f".{artifact_id}.{os.getpid()}.{threading.get_ident()}.upload"
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as handle:
            remaining = expected_bytes
            while remaining is None or remaining > 0:
                chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
        actual = digest.hexdigest()
        if actual != artifact_id:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"artifact sha256 mismatch: declared={artifact_id} actual={actual}")
        with self._lock:
            if not target.exists():
                temporary.replace(target)
            else:
                temporary.unlink(missing_ok=True)
        return {"artifact_id": artifact_id, "sha256": actual, "bytes": size, "path": str(target), "deduplicated": target.exists()}

    def materialize(self, request_id: str, artifacts: list[dict[str, Any]]) -> tuple[Path, dict[str, str], list[dict[str, Any]]]:
        request_root = self.jobs / request_id
        input_root = request_root / "input"
        input_root.mkdir(parents=True, exist_ok=True)
        source_to_staged: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ValueError(f"artifacts[{index}] must be an object")
            source = str(artifact.get("source_path") or artifact.get("path") or "")
            artifact_id = str(artifact.get("artifact_id") or artifact.get("sha256") or "").lower()
            role = str(artifact.get("role") or "input")
            if not source or len(artifact_id) != 64:
                raise ValueError(f"artifact[{index}] requires source_path and sha256 artifact_id")
            stored = self.cas / artifact_id
            if not stored.is_file():
                raise FileNotFoundError(f"artifact was not uploaded: {artifact_id}")
            suffix = Path(source).suffix
            staged = input_root / f"{index:05d}_{artifact_id[:16]}{suffix}"
            if not staged.exists():
                atomic_copy(stored, staged)
            source_abs = str(Path(source).expanduser().resolve())
            source_to_staged[source_abs] = str(staged)
            rows.append({"role": role, "source_path": source, "source_path_resolved": source_abs, "artifact_id": artifact_id, "staged_path": str(staged), "bytes": stored.stat().st_size})
        # Rewrite uploaded JSON manifests so their frame paths resolve only in the staged tree.
        for row in rows:
            if Path(row["source_path"]).suffix.lower() != ".json":
                continue
            staged_path = Path(row["staged_path"])
            try:
                payload = json.loads(staged_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rewritten = _replace_paths(payload, source_to_staged, base=Path(row["source_path"]).expanduser().resolve().parent)
            staged_path.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False), encoding="utf-8")
        return request_root, source_to_staged, rows


class ModelAdapter(Protocol):
    model_name: str
    native_batch_cap: int

    def load(self) -> None: ...

    def compatibility_key(self, payload: dict[str, Any]) -> str: ...

    def process_batch(self, entries: list["PendingRequest"]) -> list[dict[str, Any]]: ...


@dataclass
class PendingRequest:
    request_id: str
    envelope: dict[str, Any]
    payload: dict[str, Any]
    request_root: Path
    artifacts: list[dict[str, Any]]
    enqueued_monotonic: float
    done: threading.Event
    result: dict[str, Any] | None = None


class ResidentServiceState:
    def __init__(self, *, adapter: ModelAdapter, store: ArtifactStore, wait_s: float, pending_limit: int) -> None:
        self.adapter = adapter
        self.store = store
        self.wait_s = float(wait_s)
        self.pending_limit = int(pending_limit)
        self.ready = False
        self.ready_error: str | None = None
        self.started_utc = utc_now()
        self.model_load_count = 0
        self.request_count = 0
        self.completed_request_count = 0
        self.rejected_request_count = 0
        self.batch_count = 0
        self.native_forward_count = 0
        self.total_rows = 0
        self.last_batch: dict[str, Any] | None = None
        self._condition = threading.Condition()
        self._queues: dict[str, list[PendingRequest]] = {}
        self._first_queued: dict[str, float] = {}
        self._scheduler = threading.Thread(target=self._scheduler_loop, name=f"{adapter.model_name}-coalescer", daemon=True)

    def start(self) -> None:
        try:
            self.adapter.load()
            self.model_load_count = 1
            self.ready = True
        except Exception as exc:  # readiness must surface the real failure
            self.ready_error = f"{type(exc).__name__}: {exc}"
            raise
        self._scheduler.start()

    def submit(self, envelope: dict[str, Any], payload: dict[str, Any], request_root: Path, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        request_id = str(envelope.get("request_id") or payload.get("request_id") or f"req_{os.getpid()}_{time.time_ns()}")
        key = self.adapter.compatibility_key(payload)
        pending = PendingRequest(request_id=request_id, envelope=envelope, payload=payload, request_root=request_root, artifacts=artifacts, enqueued_monotonic=time.monotonic(), done=threading.Event())
        with self._condition:
            self.request_count += 1
            current = sum(len(items) for items in self._queues.values())
            if current >= self.pending_limit:
                self.rejected_request_count += 1
                return {"status": "rejected", "request_id": request_id, "error": {"code": "pending_request_limit", "limit": self.pending_limit}}
            self._queues.setdefault(key, []).append(pending)
            self._first_queued.setdefault(key, pending.enqueued_monotonic)
            self._condition.notify_all()
        # The service can spend hours on a full video; the HTTP connection remains a request/response operation.
        pending.done.wait()
        return pending.result or {"status": "failed", "request_id": request_id, "error": {"code": "missing_scheduler_result"}}

    def _scheduler_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queues:
                    self._condition.wait()
                now = time.monotonic()
                selected_key = None
                selected_due = float("inf")
                for key, items in self._queues.items():
                    if not items:
                        continue
                    due = self._first_queued[key] + self.wait_s
                    if len(items) >= self.pending_limit or due <= now:
                        selected_key = key
                        break
                    selected_due = min(selected_due, due)
                if selected_key is None:
                    self._condition.wait(timeout=max(0.01, selected_due - now))
                    continue
                items = self._queues.pop(selected_key)
                self._first_queued.pop(selected_key, None)
            self._run_batch(selected_key, items)

    def _run_batch(self, key: str, entries: list[PendingRequest]) -> None:
        started = time.monotonic()
        queue_wait = max(0.0, started - min(item.enqueued_monotonic for item in entries))
        batch_id = f"{self.adapter.model_name}_batch_{self.batch_count:08d}"
        try:
            results = self.adapter.process_batch(entries)
            if len(results) != len(entries):
                raise RuntimeError(f"adapter returned {len(results)} results for {len(entries)} requests")
            forward_elapsed = time.monotonic() - started
            self.batch_count += 1
            for entry, result in zip(entries, results):
                result.setdefault("request_id", entry.request_id)
                result.setdefault("status", "ok")
                result["batch_id"] = batch_id
                result["worker_id"] = f"{self.adapter.model_name}_resident_pid_{os.getpid()}"
                result["model_load_count"] = self.model_load_count
                result["queue_wait_s"] = queue_wait
                result["batch_elapsed_s"] = forward_elapsed
                entry.result = result
                entry.done.set()
                self.completed_request_count += 1
            self.last_batch = {"batch_id": batch_id, "compatibility_key": key, "request_count": len(entries), "queue_wait_s": queue_wait, "elapsed_s": forward_elapsed, "native_forward_count": int(getattr(self.adapter, "last_native_forward_count", 0)), "native_batch_shapes": list(getattr(self.adapter, "last_native_batch_shapes", []))}
            self.native_forward_count += int(getattr(self.adapter, "last_native_forward_count", 0))
            self.total_rows += int(getattr(self.adapter, "last_rows_processed", 0))
        except Exception as exc:
            error = {"code": "resident_batch_failed", "message": str(exc), "traceback": traceback.format_exc()[-8000:], "batch_id": batch_id}
            for entry in entries:
                entry.result = {"status": "failed", "request_id": entry.request_id, "batch_id": batch_id, "worker_id": f"{self.adapter.model_name}_resident_pid_{os.getpid()}", "model_load_count": self.model_load_count, "queue_wait_s": queue_wait, "error": error}
                entry.done.set()
                self.completed_request_count += 1

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.adapter.model_name, "pid": os.getpid(), "started_utc": self.started_utc, "ready": self.ready, "ready_error": self.ready_error}

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            queue_depth = {key: len(items) for key, items in self._queues.items()}
        return {"schema": "ego.annotation.resident_service_metrics.v1", "model": self.adapter.model_name, "pid": os.getpid(), "model_load_count": self.model_load_count, "request_count": self.request_count, "completed_request_count": self.completed_request_count, "rejected_request_count": self.rejected_request_count, "batch_count": self.batch_count, "native_forward_count": self.native_forward_count, "total_rows": self.total_rows, "queue_depth": queue_depth, "last_batch": self.last_batch, "wait_window_s": self.wait_s, "pending_request_limit": self.pending_limit, "native_batch_cap": int(getattr(self.adapter, "native_batch_cap", 0))}


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ego-resident-service/1.0"

    @property
    def state(self) -> ResidentServiceState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/healthz":
            self._send_json(200, self.state.health())
        elif path == "/readyz":
            self._send_json(200 if self.state.ready else 503, self.state.health())
        elif path == "/metrics":
            self._send_json(200, self.state.metrics())
        else:
            self._send_json(404, {"status": "not_found"})

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        prefix = "/v1/artifacts/"
        if path.startswith(prefix):
            artifact_id = unquote(path[len(prefix):]).lower()
            exists = (self.state.store.cas / artifact_id).is_file()
            self.send_response(200 if exists else 404)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        prefix = "/v1/artifacts/"
        if not path.startswith(prefix):
            self._send_json(404, {"status": "not_found"})
            return
        artifact_id = unquote(path[len(prefix) :])
        try:
            expected = int(self.headers.get("Content-Length", "0"))
            result = self.state.store.put_stream(artifact_id, self.rfile, expected)
            self._send_json(200, {"status": "ok", **result})
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": {"code": "artifact_upload_failed", "message": str(exc)}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.endswith("/infer") or not path.startswith("/v1/"):
            self._send_json(404, {"status": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            request_payload = payload.get("request") if isinstance(payload.get("request"), dict) else payload
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("inference envelope requires non-empty artifacts")
            request_id = str(payload.get("request_id") or request_payload.get("request_id") or f"req_{time.time_ns()}")
            request_root, source_to_staged, materialized = self.state.store.materialize(request_id, artifacts)
            rewritten = _replace_paths(copy.deepcopy(request_payload), source_to_staged)
            rewritten["request_id"] = request_id
            rewritten["transport"] = {"materialized": True, "request_root": str(request_root), "artifacts": materialized}
            result = self.state.submit(payload, rewritten, request_root, materialized)
            self._send_json(200 if result.get("status") != "rejected" else 429, result)
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": {"code": "invalid_inference_request", "message": str(exc), "traceback": traceback.format_exc()[-4000:]}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{utc_now()}] {self.address_string()} {fmt % args}", flush=True)


class ResidentHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: ResidentServiceState) -> None:
        super().__init__(address, _Handler)
        self.state = state


def serve(adapter: ModelAdapter, *, host: str, port: int, artifact_root: Path, wait_s: float = DEFAULT_WAIT_S, pending_limit: int = DEFAULT_PENDING_LIMIT) -> None:
    store = ArtifactStore(artifact_root)
    state = ResidentServiceState(adapter=adapter, store=store, wait_s=wait_s, pending_limit=pending_limit)
    state.start()
    server = ResidentHTTPServer((host, int(port)), state)
    print(json.dumps({"status": "listening", "model": adapter.model_name, "pid": os.getpid(), "host": host, "port": int(port), "ready": state.ready, "wait_window_s": wait_s, "pending_request_limit": pending_limit}, indent=2), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
