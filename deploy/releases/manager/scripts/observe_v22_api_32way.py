#!/usr/bin/env python3
"""Read-only 4090 observer for the manager-backed V22 32-way dataset run.

The observer reads the local batch event ledger, reads returned A800 run-result
and manifest JSON when paths are available, samples manager/service status, and
runs only ``nvidia-smi`` on the A800 through SSH.  It never submits, cancels, or
changes a manager or model-service lifecycle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

SCHEMA = "ego.annotation.v22_api_32way_observation.v1"
SUMMARY_SCHEMA = "ego.annotation.v22_api_32way_observation_summary.v1"
EVENT_LEDGER_NAME = "dataset_request_events.jsonl"
BATCH_SUMMARY_NAME = "dataset_batch_summary.json"

BREAKDOWN_FIELDS = (
    "client_prepare",
    "transport_wait",
    "client_decode_postprocess",
    "local_assembly_write",
    "total_wall",
    "request_count",
)
OWN_COMPONENTS = ("client_prepare", "client_decode_postprocess", "local_assembly_write")

DESIGN_CAPACITIES = {
    "outer": 32,
    "manager": 128,
    "unidepth": 32,
    "hands": 32,
    "wilor": 64,
    "hawor": 16,
    "infiller": 8,
    "cosmos": 32,
    "droid_absolute": 8,
}

# These are the fixed A800-local origins used by the distributed API backend.
# Reaching them is best-effort: older deployments may expose only a subset.
SERVICE_ENDPOINTS: tuple[dict[str, Any], ...] = (
    {"name": "unidepth_0", "family": "unidepth", "origin": "http://127.0.0.1:28000"},
    {"name": "unidepth_1", "family": "unidepth", "origin": "http://127.0.0.1:28005"},
    {"name": "hands", "family": "hands", "origin": "http://127.0.0.1:28001"},
    {"name": "wilor", "family": "wilor", "origin": "http://127.0.0.1:28004"},
    {"name": "hawor", "family": "hawor", "origin": "http://127.0.0.1:28003"},
    {"name": "infiller", "family": "infiller", "origin": "http://127.0.0.1:28003"},
    {"name": "cosmos", "family": "cosmos", "origin": "http://127.0.0.1:28006"},
    {"name": "droid_gpu2_0", "family": "droid", "gpu_id_design": 2, "origin": "http://127.0.0.1:28002"},
    {"name": "droid_gpu2_1", "family": "droid", "gpu_id_design": 2, "origin": "http://127.0.0.1:28012"},
    {"name": "droid_gpu2_2", "family": "droid", "gpu_id_design": 2, "origin": "http://127.0.0.1:28022"},
    {"name": "droid_gpu7_0", "family": "droid", "gpu_id_design": 7, "origin": "http://127.0.0.1:28007"},
    {"name": "droid_gpu7_1", "family": "droid", "gpu_id_design": 7, "origin": "http://127.0.0.1:28017"},
    {"name": "droid_gpu7_2", "family": "droid", "gpu_id_design": 7, "origin": "http://127.0.0.1:28027"},
)

TERMINAL_EVENTS = {"terminal", "completed", "failed", "request_finished", "request_terminal"}
SUCCESS_STATUSES = {"completed", "complete", "ok", "success", "succeeded"}
NONTERMINAL_STATUSES = {"queued", "request_started", "started", "submitted", "running", "active", "admitted", "pending"}
CONCURRENCY_FIELDS = (
    "inflight",
    "in_flight",
    "active",
    "active_requests",
    "ongoing_requests",
    "admitted_pending",
    "running_batches",
)
STATUS_IDENTITY_FIELDS = (
    "gpu_id",
    "gpu",
    "device",
    "loaded_model",
    "model",
    "model_name",
    "model_revision",
    "revision",
    "model_load_count",
    "admitted_pending",
    "running_batches",
    "queue",
    "queue_size",
    "queued",
    "inflight",
    "in_flight",
    "active_requests",
    "ongoing_requests",
)


def utc_iso(unix_s: float | None = None) -> str:
    value = time.time() if unix_s is None else float(unix_s)
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear percentile, identical for a given input on every platform."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    index = (len(finite) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return finite[lower]
    return finite[lower] + (finite[upper] - finite[lower]) * (index - lower)


def percentile_summary(values: Iterable[float], *, unit: str = "s") -> dict[str, Any]:
    observed = [float(value) for value in values if math.isfinite(float(value))]
    return {
        f"p50_{unit}": percentile(observed, 0.50),
        f"p95_{unit}": percentile(observed, 0.95),
        "denominator_observation_count": len(observed),
    }


def compare_capacity(capacity: int, observations: Iterable[Any]) -> dict[str, Any]:
    """Compare numeric concurrency evidence to one design capacity."""
    values = [float(value) for value in observations if numeric(value) is not None]
    if not values:
        return {
            "status": "unknown",
            "design_capacity": int(capacity),
            "observed_max": None,
            "denominator_observation_count": 0,
        }
    observed_max = max(values)
    return {
        "status": "reached" if observed_max >= capacity else "not_reached",
        "design_capacity": int(capacity),
        "observed_max": observed_max,
        "denominator_observation_count": len(values),
    }


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    trailing_partial_ignored = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            line_number = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.endswith("\n"):
                    trailing_partial_ignored = True
                    break
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines.append(line_number)
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                else:
                    malformed_lines.append(line_number)
    except OSError as exc:
        return [], {
            "path": str(path),
            "status": "unreadable",
            "error": repr(exc),
            "complete_row_count": 0,
            "malformed_line_numbers": [],
            "trailing_partial_ignored": False,
        }
    return rows, {
        "path": str(path),
        "status": "ok",
        "complete_row_count": len(rows),
        "malformed_line_numbers": malformed_lines,
        "trailing_partial_ignored": trailing_partial_ignored,
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def exact_values(payload: Any, names: set[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if key in names:
                    found.append((path, child))
                visit(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(payload, "")
    return found


def first_exact(payloads: Iterable[Any], names: Sequence[str]) -> tuple[Any, str | None]:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for name in names:
            if name in payload and payload[name] is not None:
                return payload[name], name
        found = exact_values(payload, set(names))
        for name in names:
            for path, value in found:
                if path.rsplit(".", 1)[-1] == name and value is not None:
                    return value, path
    return None, None


def _breakdown_value(row: Mapping[str, Any], field: str) -> float | int | None:
    value = row.get(field)
    if value is None:
        value = row.get(f"{field}_s")
    if field == "request_count":
        return integer(value)
    return numeric(value)


def normalize_module_breakdown(value: Any) -> dict[str, dict[str, Any]]:
    """Preserve declared timing components and expose every missing component."""
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_module, raw_row in sorted(value.items(), key=lambda pair: str(pair[0])):
        module = str(raw_module)
        if not isinstance(raw_row, dict):
            normalized[module] = {
                **{field: None for field in BREAKDOWN_FIELDS},
                "own": None,
                "wait": None,
                "total": None,
                "missing_fields": list(BREAKDOWN_FIELDS),
                "source_type": type(raw_row).__name__,
            }
            continue
        row = {field: _breakdown_value(raw_row, field) for field in BREAKDOWN_FIELDS}
        own_values = [row[field] for field in OWN_COMPONENTS]
        own = sum(float(item) for item in own_values) if all(item is not None for item in own_values) else None
        missing = [field for field in BREAKDOWN_FIELDS if row[field] is None]
        normalized[module] = {
            **row,
            # Preserve the concrete producer spellings as well as the compact
            # contract names used by the observer's aggregate statistics.
            **{f"{field}_s": row[field] for field in BREAKDOWN_FIELDS if field != "request_count"},
            "own": own,
            "wait": row["transport_wait"],
            "total": row["total_wall"],
            "missing_fields": missing,
            "source_type": "mapping",
        }
    return normalized


def normalize_module_timings(value: Any) -> dict[str, float | None]:
    if not isinstance(value, dict):
        return {}
    return {str(key): numeric(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _record_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for field in ("item_id", "item_index", "request_id", "request_token", "job_id"):
        value = row.get(field)
        if value is not None and str(value) != "":
            aliases.append(f"{field}:{value}")
    return aliases


def _is_terminal_row(row: Mapping[str, Any]) -> bool:
    event = str(row.get("event") or "").lower()
    status = str(row.get("status") or "").lower()
    return event in TERMINAL_EVENTS or (
        status not in NONTERMINAL_STATUSES
        and ("finished_at" in row or "finished_at_unix" in row)
    )


def _set_observed(record: dict[str, Any], field: str, value: Any, source: str | None) -> None:
    if value is not None:
        record[field] = value
        record.setdefault("field_sources", {})[field] = source


def parse_batch_events(rows: Sequence[Mapping[str, Any]], *, batch_root: Path | None = None) -> list[dict[str, Any]]:
    """Merge queued/start/terminal ledger rows into one explicit record per job."""
    records: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    sequence = 0
    for event_row in rows:
        if not isinstance(event_row, Mapping):
            continue
        aliases = _record_aliases(event_row)
        key = next((alias_to_key[alias] for alias in aliases if alias in alias_to_key), None)
        if key is None:
            key = aliases[0] if aliases else f"ledger_sequence:{sequence}"
            sequence += 1
            records[key] = {
                "request_id": None,
                "job_id": None,
                "item_id": None,
                "item_index": None,
                "source_path": None,
                "source_size_bytes": None,
                "submitted_at": None,
                "submitted_at_unix": None,
                "upload_prepare_s": None,
                "manager_http_wait_s": None,
                "response_decode_s": None,
                "total_submit_wall_s": None,
                "terminal": False,
                "terminal_status": None,
                "error": None,
                "result_path": None,
                "package_path": None,
                "manifest_path": None,
                "run_root": None,
                "frame_count": None,
                "module_timings_s": {},
                "module_timing_breakdown_s": {},
                "field_sources": {},
                "ledger_event_count": 0,
            }
        record = records[key]
        for alias in aliases:
            alias_to_key[alias] = key
        record["ledger_event_count"] += 1

        request_id, source = first_exact((event_row,), ("request_id", "request_token"))
        _set_observed(record, "request_id", request_id, source)
        job_id, source = first_exact((event_row,), ("job_id",))
        _set_observed(record, "job_id", job_id, source)
        item_id, source = first_exact((event_row,), ("item_id",))
        _set_observed(record, "item_id", item_id, source)
        item_index, source = first_exact((event_row,), ("item_index",))
        _set_observed(record, "item_index", integer(item_index), source)
        source_path, source = first_exact((event_row,), ("source_path", "video", "video_uri", "input_video"))
        _set_observed(record, "source_path", str(source_path) if source_path is not None else None, source)
        source_size, source = first_exact((event_row,), ("source_size_bytes", "size_bytes", "video_size_bytes"))
        _set_observed(record, "source_size_bytes", integer(source_size), source)
        submitted, source = first_exact((event_row,), ("submitted_at", "request_started_at"))
        submitted_unix, unix_source = first_exact((event_row,), ("submitted_at_unix", "request_started_at_unix"))
        _set_observed(record, "submitted_at", submitted, source)
        parsed_submitted = numeric(submitted_unix) if numeric(submitted_unix) is not None else parse_utc(submitted)
        _set_observed(record, "submitted_at_unix", parsed_submitted, unix_source or source)

        for target, aliases_for_field in (
            ("upload_prepare_s", ("upload_prepare_s", "client_upload_prepare_s")),
            ("manager_http_wait_s", ("manager_http_wait_s", "http_wait_s")),
            ("response_decode_s", ("response_decode_s", "submitter_response_decode_s", "client_response_decode_s")),
            ("total_submit_wall_s", ("total_submit_wall_s", "submit_wall_s", "elapsed_s")),
        ):
            value, value_source = first_exact((event_row,), aliases_for_field)
            _set_observed(record, target, numeric(value), value_source)

        for target, aliases_for_field in (
            ("result_path", ("result_path", "run_result_path")),
            ("package_path", ("package_path",)),
            ("manifest_path", ("manifest_path", "remote_manifest_path")),
            ("run_root", ("remote_run_root", "api_run_root", "run_root")),
        ):
            value, value_source = first_exact((event_row,), aliases_for_field)
            _set_observed(record, target, str(value) if value is not None else None, value_source)

        module_timings, module_source = first_exact((event_row,), ("module_timings_s",))
        if module_timings is not None:
            record["module_timings_s"] = normalize_module_timings(module_timings)
            record["field_sources"]["module_timings_s"] = module_source
        module_breakdown, breakdown_source = first_exact((event_row,), ("module_timing_breakdown_s",))
        if module_breakdown is not None:
            record["module_timing_breakdown_s"] = normalize_module_breakdown(module_breakdown)
            record["field_sources"]["module_timing_breakdown_s"] = breakdown_source
        frame_count, frame_source = first_exact((event_row,), ("frame_count", "completed_frame_count", "image_count"))
        _set_observed(record, "frame_count", integer(frame_count), frame_source)

        if _is_terminal_row(event_row):
            record["terminal"] = True
            record["terminal_status"] = str(event_row.get("status") or event_row.get("event") or "unknown")
            error, error_source = first_exact((event_row,), ("error", "error_message", "detail"))
            if error is None and not _successful_status(event_row.get("status")):
                error = event_row.get("response")
                error_source = "response" if error is not None else error_source
            _set_observed(record, "error", error, error_source)
            finished, finished_source = first_exact((event_row,), ("finished_at",))
            finished_unix, finished_unix_source = first_exact((event_row,), ("finished_at_unix",))
            _set_observed(record, "finished_at", finished, finished_source)
            parsed_finished = numeric(finished_unix) if numeric(finished_unix) is not None else parse_utc(finished)
            _set_observed(record, "finished_at_unix", parsed_finished, finished_unix_source or finished_source)

        # Preserve returned mappings so artifact hydration does not need remote I/O.
        for name in ("run_result", "manifest"):
            value = event_row.get(name)
            if isinstance(value, dict):
                record[f"_{name}_payload"] = value

    output = sorted(
        records.values(),
        key=lambda row: (
            row.get("item_index") is None,
            row.get("item_index") if row.get("item_index") is not None else 0,
            str(row.get("item_id") or row.get("request_id") or row.get("job_id") or ""),
        ),
    )
    if batch_root is not None:
        for record in output:
            source_path = record.get("source_path")
            if record["source_size_bytes"] is None and isinstance(source_path, str):
                path = Path(source_path)
                try:
                    record["source_size_bytes"] = path.stat().st_size
                    record["field_sources"]["source_size_bytes"] = "local_source_stat"
                except OSError:
                    pass
    return output


class SSHReader:
    def __init__(self, host: str, *, port: int | None, timeout_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def run(self, remote_args: Sequence[str]) -> dict[str, Any]:
        command = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={max(1, int(self.timeout_s))}",
        ]
        if self.port is not None:
            command.extend(["-p", str(self.port)])
        command.extend([self.host, shlex.join(list(remote_args))])
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "ssh_error", "error": repr(exc), "elapsed_s": time.monotonic() - started}
        return {
            "status": "ok" if proc.returncode == 0 else "ssh_error",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr.strip(),
            "elapsed_s": time.monotonic() - started,
        }

    def read_json(self, path: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        result = self.run(("cat", "--", path))
        evidence = {key: value for key, value in result.items() if key != "stdout"}
        evidence["path"] = path
        if result.get("status") != "ok":
            return None, evidence
        try:
            payload = json.loads(str(result.get("stdout") or ""))
        except json.JSONDecodeError as exc:
            evidence.update({"status": "invalid_json", "error": repr(exc)})
            return None, evidence
        if not isinstance(payload, dict):
            evidence.update({"status": "invalid_json_root", "root_type": type(payload).__name__})
            return None, evidence
        evidence["status"] = "ok"
        return payload, evidence

    def http_json(self, url: str) -> dict[str, Any]:
        result = self.run((
            "curl", "--silent", "--show-error", "--max-time", str(max(1, int(self.timeout_s))),
            "--write-out", "\n%{http_code}", url,
        ))
        if result.get("status") != "ok":
            return {key: value for key, value in result.items() if key != "stdout"}
        lines = str(result.get("stdout") or "").splitlines()
        if not lines:
            return {"status": "invalid_response", "error": "curl returned no status line"}
        try:
            http_status = int(lines[-1])
        except ValueError:
            return {"status": "invalid_response", "error": "curl status line was not numeric"}
        body = "\n".join(lines[:-1])
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = body[-4000:]
        return {
            "status": "reachable" if 200 <= http_status < 300 else "http_error",
            "http_status": http_status,
            "payload": payload,
            "elapsed_s": result.get("elapsed_s"),
        }


class ArtifactLoader:
    def __init__(self, ssh: SSHReader) -> None:
        self.ssh = ssh
        self.cache: dict[str, dict[str, Any]] = {}
        self.missing: set[str] = set()
        self.evidence: dict[str, dict[str, Any]] = {}

    def load(self, path: str) -> dict[str, Any] | None:
        if path in self.cache:
            return self.cache[path]
        if path in self.missing:
            return None
        local = read_json(Path(path))
        if local is not None:
            self.cache[path] = local
            self.evidence[path] = {"path": path, "status": "ok", "transport": "local_file"}
            return local
        remote, evidence = self.ssh.read_json(path)
        self.evidence[path] = {**evidence, "transport": "ssh_cat"}
        if remote is not None:
            self.cache[path] = remote
        else:
            # Terminal job artifacts are immutable. Remembering a miss avoids
            # an O(completed_jobs) SSH retry storm on every later poll.
            self.missing.add(path)
        return remote


def _candidate_paths(record: Mapping[str, Any], job_root: Path) -> tuple[list[str], list[str]]:
    result_paths: list[str] = []
    manifest_paths: list[str] = []
    if isinstance(record.get("result_path"), str):
        result_paths.append(str(record["result_path"]))
    if isinstance(record.get("manifest_path"), str):
        manifest_paths.append(str(record["manifest_path"]))
    run_root = record.get("run_root")
    if isinstance(run_root, str) and run_root:
        result_paths.append(str(Path(run_root) / "run_result.json"))
        manifest_paths.append(str(Path(run_root) / "annotation_pipeline_manifest.json"))
    job_id = record.get("job_id")
    if isinstance(job_id, str) and job_id and all(char.isalnum() or char in "._-" for char in job_id):
        result_paths.append(str(job_root / job_id / "run_result.json"))
        manifest_paths.append(str(job_root / job_id / "annotation_pipeline_manifest.json"))
    return list(dict.fromkeys(result_paths)), list(dict.fromkeys(manifest_paths))


def hydrate_job_artifacts(record: dict[str, Any], *, job_root: Path, loader: ArtifactLoader) -> None:
    run_result = record.pop("_run_result_payload", None)
    manifest = record.pop("_manifest_payload", None)
    result_paths, manifest_paths = _candidate_paths(record, job_root)
    if not isinstance(run_result, dict):
        for path in result_paths:
            run_result = loader.load(path)
            if run_result is not None:
                record["result_path"] = path
                break
    if not isinstance(manifest, dict):
        for path in manifest_paths:
            manifest = loader.load(path)
            if manifest is not None:
                record["manifest_path"] = path
                break
    payloads = tuple(payload for payload in (run_result, manifest) if isinstance(payload, dict))
    module_timings, module_source = first_exact(payloads, ("module_timings_s",))
    if module_timings is not None:
        record["module_timings_s"] = normalize_module_timings(module_timings)
        record["field_sources"]["module_timings_s"] = f"artifact:{module_source}"
    module_breakdown, breakdown_source = first_exact(payloads, ("module_timing_breakdown_s",))
    if module_breakdown is not None:
        record["module_timing_breakdown_s"] = normalize_module_breakdown(module_breakdown)
        record["field_sources"]["module_timing_breakdown_s"] = f"artifact:{breakdown_source}"
    frame_count, frame_source = first_exact(payloads, ("frame_count", "completed_frame_count", "image_count"))
    if integer(frame_count) is not None:
        record["frame_count"] = integer(frame_count)
        record["field_sources"]["frame_count"] = f"artifact:{frame_source}"
    if record.get("run_root") is None:
        run_root, root_source = first_exact(payloads, ("run_root",))
        _set_observed(record, "run_root", str(run_root) if run_root is not None else None, f"artifact:{root_source}")
    if record.get("package_path") is None:
        package, package_source = first_exact(payloads, ("package_path",))
        _set_observed(record, "package_path", str(package) if package is not None else None, f"artifact:{package_source}")
    required = (
        "request_id", "job_id", "item_id", "source_path", "source_size_bytes", "submitted_at",
        "upload_prepare_s", "manager_http_wait_s", "response_decode_s", "total_submit_wall_s",
        "terminal_status", "error", "result_path", "package_path", "manifest_path", "frame_count",
    )
    record["missing_fields"] = [field for field in required if record.get(field) is None]
    record["artifact_payloads"] = {
        "run_result": "present" if isinstance(run_result, dict) else "missing",
        "manifest": "present" if isinstance(manifest, dict) else "missing",
    }


def _successful_status(status: Any) -> bool:
    value = str(status or "").lower()
    if value in SUCCESS_STATUSES:
        return True
    return value.startswith("completed_with_") and value not in {
        "completed_with_failures", "completed_with_errors",
    }


def is_success(record: Mapping[str, Any]) -> bool:
    return bool(record.get("terminal")) and _successful_status(record.get("terminal_status"))


def is_failure(record: Mapping[str, Any]) -> bool:
    if not record.get("terminal"):
        return False
    return not _successful_status(record.get("terminal_status"))


def max_concurrent_jobs(jobs: Sequence[Mapping[str, Any]], *, observed_at_unix: float) -> int | None:
    """Reconstruct the outer HTTP maximum from half-open job intervals."""
    points: list[tuple[float, int]] = []
    for job in jobs:
        started = numeric(job.get("submitted_at_unix"))
        if started is None:
            continue
        if job.get("terminal"):
            finished = numeric(job.get("finished_at_unix"))
            if finished is None:
                wall = numeric(job.get("total_submit_wall_s"))
                finished = started + wall if wall is not None else None
        else:
            finished = observed_at_unix
        if finished is None or finished < started:
            continue
        points.extend(((started, 1), (finished, -1)))
    if not points:
        return None
    # A completion and a new start at the same timestamp are non-overlapping.
    points.sort(key=lambda item: (item[0], item[1]))
    active = 0
    observed_max = 0
    for _timestamp, delta in points:
        active += delta
        observed_max = max(observed_max, active)
    return observed_max


def throughput_summary(jobs: Sequence[Mapping[str, Any]], *, observer_started_unix: float, observed_at_unix: float) -> dict[str, Any]:
    submitted_times = [numeric(job.get("submitted_at_unix")) for job in jobs]
    finite_submitted = [value for value in submitted_times if value is not None]
    window_start = min(finite_submitted) if finite_submitted else observer_started_unix
    window_s = max(0.001, observed_at_unix - window_start)
    completed = [job for job in jobs if is_success(job)]
    failed = [job for job in jobs if is_failure(job)]
    terminal = [job for job in jobs if job.get("terminal")]
    active = [job for job in jobs if job.get("submitted_at_unix") is not None and not job.get("terminal")]
    completed_frames = [integer(job.get("frame_count")) for job in completed]
    known_frames = [value for value in completed_frames if value is not None and value >= 0]
    job_walls = [numeric(job.get("total_submit_wall_s")) for job in terminal]
    job_wall_values = [value for value in job_walls if value is not None]

    module_names = sorted({
        module
        for job in jobs
        for module in (job.get("module_timing_breakdown_s") or {})
    })
    module_stats: dict[str, Any] = {}
    for module in module_names:
        rows = [
            job["module_timing_breakdown_s"][module]
            for job in jobs
            if isinstance(job.get("module_timing_breakdown_s"), dict)
            and isinstance(job["module_timing_breakdown_s"].get(module), dict)
        ]
        request_counts = [integer(row.get("request_count")) for row in rows]
        module_stats[module] = {
            metric: percentile_summary(
                [float(row[metric]) for row in rows if numeric(row.get(metric)) is not None],
                unit="s",
            )
            for metric in ("own", "wait", "total")
        }
        total_requests = sum(value for value in request_counts if value is not None)
        module_stats[module]["request_count"] = {
            "numerator_requests": total_requests,
            "denominator_job_count_with_request_count": sum(value is not None for value in request_counts),
        }
        module_stats[module]["wall_requests_per_s"] = {
            "value": total_requests / window_s,
            "numerator_requests": total_requests,
            "denominator_window_s": window_s,
        }

    return {
        "window_started_at_utc": utc_iso(window_start),
        "window_started_at_unix": window_start,
        "window_ended_at_utc": utc_iso(observed_at_unix),
        "window_ended_at_unix": observed_at_unix,
        "window_wall_s": window_s,
        "counts": {
            "known_jobs": len(jobs),
            "active": len(active),
            "completed": len(completed),
            "failed": len(failed),
            "terminal": len(terminal),
            "outer_observed_max_inflight": max_concurrent_jobs(jobs, observed_at_unix=observed_at_unix),
        },
        "videos_per_hour": {
            "value": len(completed) / (window_s / 3600.0),
            "numerator_completed_videos": len(completed),
            "denominator_window_hours": window_s / 3600.0,
        },
        "wall_requests_per_s": {
            "value": len(terminal) / window_s,
            "numerator_terminal_requests": len(terminal),
            "denominator_window_s": window_s,
        },
        "completed_images_per_s": {
            "value": sum(known_frames) / window_s if known_frames else None,
            "numerator_completed_images": sum(known_frames),
            "denominator_window_s": window_s,
            "denominator_completed_jobs_with_frame_count": len(known_frames),
            "completed_jobs_missing_frame_count": len(completed) - len(known_frames),
        },
        "job_submit_wall": percentile_summary(job_wall_values, unit="s"),
        "job_submit_wall_source": "batch ledger total_submit_wall_s; legacy elapsed_s is accepted only as total submit wall, never as server compute",
        "module_timing_breakdown": module_stats,
        "module_own_definition": "client_prepare + client_decode_postprocess + local_assembly_write; missing when any component is absent",
        "module_wait_definition": "transport_wait as reported; HTTP duration is not treated as server-internal compute",
        "module_total_definition": "total_wall as reported",
    }


def http_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "v22-4090-observer/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read(4 * 1024 * 1024)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read(1024 * 1024)
        status = int(exc.code)
    except Exception as exc:
        return {"status": "unreachable", "error": repr(exc), "elapsed_s": time.monotonic() - started}
    try:
        payload: Any = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = body.decode("utf-8", errors="replace")[-4000:]
    return {
        "status": "reachable" if 200 <= status < 300 else "http_error",
        "http_status": status,
        "payload": payload,
        "elapsed_s": time.monotonic() - started,
    }


def identity_fields(payload: Any) -> dict[str, Any]:
    found = exact_values(payload, set(STATUS_IDENTITY_FIELDS))
    values: dict[str, list[dict[str, Any]]] = {name: [] for name in STATUS_IDENTITY_FIELDS}
    for path, value in found:
        name = path.rsplit(".", 1)[-1]
        if name in values:
            values[name].append({"path": path, "value": value})
    return {
        "observed": {name: rows for name, rows in values.items() if rows},
        "missing": [name for name, rows in values.items() if not rows],
    }


def manager_snapshot(manager_url: str, *, timeout_s: float) -> dict[str, Any]:
    base = manager_url.rstrip("/")
    status = http_json(f"{base}/status", timeout_s=timeout_s)
    openapi = http_json(f"{base}/openapi.json", timeout_s=timeout_s)
    payload = openapi.get("payload")
    identity: dict[str, Any]
    if isinstance(payload, dict):
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
        identity = {
            "openapi": payload.get("openapi"),
            "title": info.get("title"),
            "version": info.get("version"),
            "paths": sorted(str(path) for path in paths),
        }
    else:
        identity = {"openapi": None, "title": None, "version": None, "paths": None}
    return {
        "manager_url": manager_url,
        "status": {**status, "identity_fields": identity_fields(status.get("payload"))},
        "openapi": {key: value for key, value in openapi.items() if key != "payload"},
        "openapi_identity": identity,
    }


def service_snapshots(ssh: SSHReader) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    origin_cache: dict[str, dict[str, Any]] = {}
    for spec in SERVICE_ENDPOINTS:
        origin = str(spec["origin"])
        if origin not in origin_cache:
            status = ssh.http_json(f"{origin}/status")
            if status.get("status") != "reachable":
                health = ssh.http_json(f"{origin}/-/healthz")
            else:
                health = {"status": "not_queried_status_reachable"}
            origin_cache[origin] = {"status_probe": status, "health_probe": health}
        probes = origin_cache[origin]
        status_payload = probes["status_probe"].get("payload")
        rows[str(spec["name"])] = {
            **spec,
            **probes,
            "identity_fields": identity_fields(status_payload),
        }
    return rows


def parse_nvidia_smi(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 9:
            continue
        names = (
            "gpu_id", "uuid", "name", "utilization_gpu_pct", "utilization_memory_pct",
            "memory_used_mib", "memory_total_mib", "power_draw_w", "power_limit_w",
        )
        row: dict[str, Any] = {}
        for name, value in zip(names, fields):
            if name in {"gpu_id", "uuid", "name"}:
                row[name] = integer(value) if name == "gpu_id" else value
            else:
                try:
                    row[name] = float(value)
                except ValueError:
                    row[name] = None
        rows.append(row)
    return rows


def gpu_snapshot(ssh: SSHReader) -> dict[str, Any]:
    polled_at_unix = time.time()
    result = ssh.run((
        "nvidia-smi",
        "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ))
    stdout = str(result.pop("stdout", ""))
    return {
        "poll_started_at_utc": utc_iso(polled_at_unix),
        "poll_started_at_unix": polled_at_unix,
        "poll_finished_at_utc": utc_iso(),
        "command": "ssh <a800-host> nvidia-smi --query-gpu=... --format=csv,noheader,nounits",
        "transport": result,
        "gpus": parse_nvidia_smi(stdout) if result.get("status") == "ok" else [],
    }


def concurrency_evidence(payload: Any, *, observed_at_utc: str, source: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path, value in exact_values(payload, set(CONCURRENCY_FIELDS)):
        number = numeric(value)
        if number is not None:
            evidence.append({
                "observed_at_utc": observed_at_utc,
                "source": source,
                "field": path,
                "value": number,
            })
    return evidence


def current_capacity_evidence(
    throughput: Mapping[str, Any],
    manager: Mapping[str, Any],
    services: Mapping[str, Any],
    *,
    observed_at_utc: str,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in DESIGN_CAPACITIES}
    outer_max = numeric((throughput.get("counts") or {}).get("outer_observed_max_inflight"))
    if outer_max is not None:
        result["outer"].append({
            "observed_at_utc": observed_at_utc,
            "source": "batch_event_ledger_interval_reconstruction",
            "field": "outer_observed_max_inflight",
            "value": outer_max,
        })
    manager_payload = ((manager.get("status") or {}).get("payload"))
    result["manager"].extend(concurrency_evidence(manager_payload, observed_at_utc=observed_at_utc, source="manager:/status"))
    for name, row in services.items():
        family = str(row.get("family") or "")
        capacity_name = "droid_absolute" if family == "droid" else family
        if capacity_name not in result:
            continue
        payload = ((row.get("status_probe") or {}).get("payload"))
        result[capacity_name].extend(concurrency_evidence(payload, observed_at_utc=observed_at_utc, source=f"service:{name}:/status"))
    return result


def merge_capacity_evidence(
    retained: dict[str, list[dict[str, Any]]],
    current: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for name, rows in current.items():
        target = retained.setdefault(name, [])
        for row in rows:
            target.append(dict(row))
        target.sort(key=lambda row: float(row.get("value") or 0.0), reverse=True)
        del target[64:]


def capacity_report(evidence: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, capacity in DESIGN_CAPACITIES.items():
        rows = list(evidence.get(name, ()))
        comparison = compare_capacity(capacity, (row.get("value") for row in rows))
        report[name] = {
            **comparison,
            "evidence": rows[:16],
            "evidence_scope": "DROID absolute capacity is compared per exposed endpoint field; service families retain endpoint-qualified evidence",
        }
    return report


def gpu_maxima(existing: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    maxima = {str(key): dict(value) for key, value in existing.items() if isinstance(value, dict)}
    timestamp = snapshot.get("poll_finished_at_utc")
    for row in snapshot.get("gpus", []):
        gpu_id = str(row.get("gpu_id"))
        target = maxima.setdefault(gpu_id, {"gpu_id": row.get("gpu_id"), "uuid": row.get("uuid"), "name": row.get("name")})
        for field in ("utilization_gpu_pct", "utilization_memory_pct", "memory_used_mib", "power_draw_w"):
            value = numeric(row.get(field))
            old = numeric(target.get(f"max_{field}"))
            if value is not None and (old is None or value > old):
                target[f"max_{field}"] = value
                target[f"max_{field}_observed_at_utc"] = timestamp
    return maxima


def batch_terminal(batch_summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(batch_summary, Mapping):
        return False
    status = str(batch_summary.get("status") or "").lower()
    if status not in {"completed", "completed_with_failures", "completed_with_errors", "failed"}:
        return False
    submitted = integer(batch_summary.get("submitted_count"))
    terminal = integer(batch_summary.get("terminal_count"))
    return submitted is None or terminal is None or terminal >= submitted


def load_prior_state(output_json: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    prior = read_json(output_json)
    evidence = {name: [] for name in DESIGN_CAPACITIES}
    gpu = {}
    if prior is None:
        return evidence, gpu
    capacities = prior.get("capacity_comparisons")
    if isinstance(capacities, dict):
        for name in evidence:
            row = capacities.get(name)
            if isinstance(row, dict) and isinstance(row.get("evidence"), list):
                evidence[name] = [item for item in row["evidence"] if isinstance(item, dict)]
    if isinstance(prior.get("gpu_maxima"), dict):
        gpu = dict(prior["gpu_maxima"])
    return evidence, gpu


def observe_once(
    *,
    batch_root: Path,
    job_root: Path,
    manager_url: str,
    ssh: SSHReader,
    artifact_loader: ArtifactLoader,
    observer_started_unix: float,
    capacity_evidence: dict[str, list[dict[str, Any]]],
    prior_gpu_maxima: Mapping[str, Any],
    http_timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observed_at_unix = time.time()
    observed_at_utc = utc_iso(observed_at_unix)
    event_rows, ledger = read_complete_jsonl(batch_root / EVENT_LEDGER_NAME)
    jobs = parse_batch_events(event_rows, batch_root=batch_root)
    for job in jobs:
        if job.get("terminal"):
            hydrate_job_artifacts(job, job_root=job_root, loader=artifact_loader)
        else:
            for private in ("_run_result_payload", "_manifest_payload"):
                job.pop(private, None)
            job["missing_fields"] = [
                field for field in (
                    "request_id", "job_id", "item_id", "source_path", "source_size_bytes", "submitted_at",
                    "upload_prepare_s", "manager_http_wait_s", "response_decode_s", "total_submit_wall_s",
                    "terminal_status", "error", "result_path", "package_path", "manifest_path", "frame_count",
                ) if job.get(field) is None
            ]
    throughput = throughput_summary(jobs, observer_started_unix=observer_started_unix, observed_at_unix=observed_at_unix)
    manager = manager_snapshot(manager_url, timeout_s=http_timeout_s)
    services = service_snapshots(ssh)
    gpu = gpu_snapshot(ssh)
    current = current_capacity_evidence(throughput, manager, services, observed_at_utc=observed_at_utc)
    merge_capacity_evidence(capacity_evidence, current)
    capacities = capacity_report(capacity_evidence)
    updated_gpu_maxima = gpu_maxima(prior_gpu_maxima, gpu)
    batch_summary = read_json(batch_root / BATCH_SUMMARY_NAME)
    snapshot = {
        "schema": SCHEMA,
        "observed_at_utc": observed_at_utc,
        "observed_at_unix": observed_at_unix,
        "observer_started_at_utc": utc_iso(observer_started_unix),
        "observer_started_at_unix": observer_started_unix,
        "scope": {
            "batch_root": str(batch_root),
            "job_root": str(job_root),
            "manager_url": manager_url,
            "a800_host": ssh.host,
            "read_only": True,
        },
        "ledger": ledger,
        "batch_summary": batch_summary,
        "jobs": jobs,
        "throughput": throughput,
        "manager": manager,
        "services": services,
        "a800_gpus": gpu,
        "capacity_observations_this_poll": current,
        "capacity_comparisons": capacities,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "batch_terminal" if batch_terminal(batch_summary) else "observing",
        "updated_at_utc": utc_iso(),
        "observer_started_at_utc": utc_iso(observer_started_unix),
        "scope": snapshot["scope"],
        "ledger": ledger,
        "batch_summary": batch_summary,
        "jobs": jobs,
        "throughput": throughput,
        "manager_identity": manager.get("openapi_identity"),
        "latest_manager_status": manager.get("status"),
        "latest_services": services,
        "latest_a800_gpus": gpu,
        "gpu_maxima": updated_gpu_maxima,
        "capacity_comparisons": capacities,
        "artifact_reads": dict(sorted(artifact_loader.evidence.items())),
        "interpretation_boundary": "HTTP/client wall durations are transport/client observations, not server-internal compute time.",
    }
    return snapshot, summary, updated_gpu_maxima


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True, help="4090 batch output root containing dataset_request_events.jsonl")
    parser.add_argument("--job-root", type=Path, required=True, help="A800 manager job root used to resolve returned job paths")
    parser.add_argument("--manager-url", required=True, help="Existing configured/tunneled manager origin, e.g. http://127.0.0.1:8092")
    parser.add_argument("--a800-host", required=True, help="Existing OpenSSH host/alias; the observer does not create a tunnel")
    parser.add_argument("--a800-ssh-port", type=int, help="Optional SSH port when it is not in ssh_config")
    parser.add_argument("--poll-s", type=float, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--ssh-timeout-s", type=float, default=15.0)
    parser.add_argument("--once", action="store_true", help="Write exactly one snapshot and summary")
    parser.add_argument("--keep-running-after-terminal", action="store_true")
    args = parser.parse_args(argv)
    if args.poll_s <= 0 or args.http_timeout_s <= 0 or args.ssh_timeout_s <= 0:
        parser.error("--poll-s and timeout values must be positive")
    if args.a800_ssh_port is not None and not 1 <= args.a800_ssh_port <= 65535:
        parser.error("--a800-ssh-port must be in [1, 65535]")
    manager = urlsplit(args.manager_url)
    if manager.scheme not in {"http", "https"} or not manager.netloc or manager.path not in {"", "/"}:
        parser.error("--manager-url must be an HTTP(S) origin without a path")
    args.batch_root = args.batch_root.expanduser().resolve()
    args.job_root = args.job_root.expanduser()
    args.output_jsonl = args.output_jsonl.expanduser().resolve()
    args.output_json = args.output_json.expanduser().resolve()
    if args.output_jsonl == args.output_json:
        parser.error("--output-jsonl and --output-json must be different paths")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    observer_started_unix = time.time()
    ssh = SSHReader(args.a800_host, port=args.a800_ssh_port, timeout_s=args.ssh_timeout_s)
    artifact_loader = ArtifactLoader(ssh)
    capacity_evidence, maxima = load_prior_state(args.output_json)
    waiter = threading.Event()
    stop_reason = "once" if args.once else "running"
    try:
        while True:
            snapshot, summary, maxima = observe_once(
                batch_root=args.batch_root,
                job_root=args.job_root,
                manager_url=args.manager_url,
                ssh=ssh,
                artifact_loader=artifact_loader,
                observer_started_unix=observer_started_unix,
                capacity_evidence=capacity_evidence,
                prior_gpu_maxima=maxima,
                http_timeout_s=args.http_timeout_s,
            )
            append_jsonl(args.output_jsonl, snapshot)
            write_json_atomic(args.output_json, summary)
            print(json.dumps({
                "observed_at_utc": snapshot["observed_at_utc"],
                "counts": snapshot["throughput"]["counts"],
                "videos_per_hour": snapshot["throughput"]["videos_per_hour"]["value"],
                "summary": str(args.output_json),
            }, ensure_ascii=False, sort_keys=True), flush=True)
            if args.once:
                break
            if summary["status"] == "batch_terminal" and not args.keep_running_after_terminal:
                stop_reason = "batch_terminal"
                break
            waiter.wait(args.poll_s)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    final = read_json(args.output_json) or {}
    final["observer_exit"] = {"reason": stop_reason, "at_utc": utc_iso()}
    write_json_atomic(args.output_json, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
