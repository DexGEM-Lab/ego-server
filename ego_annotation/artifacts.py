"""Artifact bundle writer for ego.annotation.output v1."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ego_annotation.schema import COORDINATE_FRAMES, NDJSON_STREAMS, PARQUET_TABLES, SCHEMA_NAME, SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default, sort_keys=True))
            f.write("\n")


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, sort_keys=True, default=_json_default)
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [_flatten_for_csv(row) for row in rows]
    fieldnames: list[str] = []
    for row in flat:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)


def status_from_errors(errors: list[dict[str, Any]]) -> str:
    severities = {str(row.get("severity")) for row in errors}
    if "error" in severities:
        return "completed_with_errors"
    if severities:
        return "completed_with_degraded_outputs"
    return "ok"


def write_table(path_base: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl_path = path_base.with_suffix(".ndjson")
    csv_path = path_base.with_suffix(".csv")
    write_ndjson(jsonl_path, rows)
    write_csv(csv_path, rows)
    artifact = {
        "rows": len(rows),
        "ndjson": str(jsonl_path),
        "csv": str(csv_path),
        "parquet": None,
        "format_status": "jsonl_csv_written",
    }
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:
        errors.append(
            {
                "code": "parquet_writer_unavailable",
                "severity": "degraded",
                "message": f"Parquet table {path_base.name} was not written because optional dependency is unavailable: {exc.name}.",
                "mechanism": "alpha artifact remains inspectable as NDJSON/CSV; production deployment must install pyarrow for ego.annotation.output parquet tables.",
            }
        )
        return artifact
    parquet_path = path_base.with_suffix(".parquet")
    table = pa.Table.from_pylist(rows if rows else [{}])
    pq.write_table(table, parquet_path)
    artifact["parquet"] = str(parquet_path)
    artifact["format_status"] = "parquet_ndjson_csv_written"
    return artifact


class ArtifactBundle:
    def __init__(self, root: Path, job_id: str) -> None:
        self.root = root.resolve() / job_id
        self.job_id = job_id
        self.tables_dir = self.root / "tables"
        self.events_dir = self.root / "events"
        self.renders_dir = self.root / "renders"
        self.state_dir = self.root / "state"
        self.errors: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []

    def add_error(self, code: str, severity: str, message: str, mechanism: str, **extra: Any) -> None:
        row = {"code": code, "severity": severity, "message": message, "mechanism": mechanism, **extra}
        self.errors.append(row)

    def add_provenance(self, stage: str, event: str, **extra: Any) -> None:
        self.provenance.append({"stage": stage, "event": event, "time_utc": utc_now(), **extra})

    def write(
        self,
        *,
        request: dict[str, Any],
        calibration_contract: dict[str, Any],
        tables: dict[str, list[dict[str, Any]]],
        events: dict[str, list[dict[str, Any]]],
        throughput_forecast: dict[str, Any],
        status: str,
        render_artifacts: dict[str, Any] | None = None,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        write_json(self.state_dir / "calibration_contract.json", calibration_contract)
        table_artifacts: dict[str, Any] = {}
        for name in PARQUET_TABLES:
            rows = tables.get(name, [])
            table_artifacts[name] = write_table(self.tables_dir / name, rows, self.errors)

        merged_events = dict(events)
        merged_events["errors"] = self.errors
        merged_events["provenance"] = self.provenance
        stream_artifacts: dict[str, Any] = {}
        for name in NDJSON_STREAMS:
            rows = merged_events.get(name, [])
            path = self.events_dir / f"{name}.ndjson"
            write_ndjson(path, rows)
            stream_artifacts[name] = {"rows": len(rows), "ndjson": str(path)}

        final_status = status_from_errors(self.errors)
        renders = {"optional_qc_demo": [], "note": "renders are projections of numeric state and cannot change numeric results"}
        if render_artifacts:
            renders.update(render_artifacts)
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "status": final_status,
            "created_utc": utc_now(),
            "request": request,
            "coordinate_frames": COORDINATE_FRAMES,
            "calibration_contract": str(self.state_dir / "calibration_contract.json"),
            "tables": table_artifacts,
            "events": stream_artifacts,
            "renders": renders,
            "throughput_forecast": throughput_forecast,
            "errors_count": len(self.errors),
            "provenance_count": len(self.provenance),
        }
        manifest_path = self.root / "manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path
