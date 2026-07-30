"""Resident batch worker protocol and dependency-light implementation.

The production A800 workers should implement the same request/response shape
with real model loaders such as UniDepth, WiLoR, or HaWoR. The local worker here
is intentionally light: it proves load-once residency, true batch request
consumption, per-item output mapping, and ownership fields without local heavy
model inference.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ego_annotation.artifacts import write_json
from ego_annotation.batch import utc_now_s


class ResidentModel(Protocol):
    model_name: str
    model_version: str

    def infer(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one true batch through an already-loaded model instance."""


@dataclass
class ResidentWorkerStats:
    model_load_count: int = 0
    batch_inference_count: int = 0
    rows_inferred: int = 0
    first_load_utc: str | None = None
    last_inference_utc: str | None = None


@dataclass
class BatchRow:
    row_id: str
    job_id: str
    item_id: str
    batch_id: str
    stage_id: str
    input_artifact: str
    output_artifact: str
    run_root: str
    agent_id: str
    attempt_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, agent_id: str, attempt_id: str) -> "BatchRow":
        required = ["row_id", "job_id", "item_id", "batch_id", "stage_id", "input_artifact", "output_artifact", "run_root"]
        missing = [key for key in required if not payload.get(key)]
        if missing:
            raise ValueError(f"batch row missing required fields: {missing}")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return cls(
            row_id=str(payload["row_id"]),
            job_id=str(payload["job_id"]),
            item_id=str(payload["item_id"]),
            batch_id=str(payload["batch_id"]),
            stage_id=str(payload["stage_id"]),
            input_artifact=str(payload["input_artifact"]),
            output_artifact=str(payload["output_artifact"]),
            run_root=str(payload["run_root"]),
            agent_id=agent_id,
            attempt_id=attempt_id,
            metadata=dict(metadata),
        )

    def to_model_input(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "job_id": self.job_id,
            "item_id": self.item_id,
            "batch_id": self.batch_id,
            "stage_id": self.stage_id,
            "input_artifact": self.input_artifact,
            "output_artifact": self.output_artifact,
            "run_root": self.run_root,
            "agent_id": self.agent_id,
            "attempt_id": self.attempt_id,
            "metadata": self.metadata,
        }


class FileFingerprintModel:
    """Tiny resident model used by local tests and smoke runs.

    It consumes a real batch of rows and reads each input artifact if it exists.
    It is a contract model, not a substitute for UniDepth/WiLoR/HaWoR.
    """

    model_name = "file_fingerprint_contract_model"
    model_version = "v0"

    def infer(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for row in inputs:
            path = Path(str(row["input_artifact"])).expanduser()
            exists = path.exists() and path.is_file()
            digest = None
            size = None
            if exists:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                size = len(data)
            outputs.append(
                {
                    "row_id": row["row_id"],
                    "status": "ok" if exists else "failed",
                    "prediction": {
                        "input_exists": exists,
                        "input_sha256": digest,
                        "input_size_bytes": size,
                        "contract_note": "dependency-light resident batch contract output; not a physical annotation model",
                    },
                    "error": None if exists else {"code": "input_artifact_missing", "message": str(path)},
                }
            )
        return outputs


class ResidentBatchWorker:
    """One loaded model instance that can consume multiple stage batches."""

    def __init__(self, *, worker_id: str, stage_id: str, model: ResidentModel, device: str = "cpu", model_config: dict[str, Any] | None = None) -> None:
        self.worker_id = worker_id
        self.stage_id = stage_id
        self.model = model
        self.device = device
        self.model_config = dict(model_config or {})
        self.stats = ResidentWorkerStats(model_load_count=1, first_load_utc=utc_now_s())

    @property
    def model_identity(self) -> dict[str, Any]:
        return {
            "model_name": self.model.model_name,
            "model_version": self.model.model_version,
            "stage_id": self.stage_id,
            "worker_id": self.worker_id,
            "device": self.device,
            "model_config": self.model_config,
        }

    def infer_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if batch.get("stage_id") != self.stage_id:
            raise ValueError(f"worker stage {self.stage_id} cannot handle {batch.get('stage_id')}")
        agent_id = str(batch.get("agent_id") or "unknown_agent")
        attempt_id = str(batch.get("attempt_id") or "attempt_0001")
        raw_rows = batch.get("rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("batch rows must be a non-empty list")
        rows = [BatchRow.from_mapping(row, agent_id=agent_id, attempt_id=attempt_id) for row in raw_rows if isinstance(row, dict)]
        if len(rows) != len(raw_rows):
            raise ValueError("batch rows must be objects")
        started = time.time()
        model_outputs = self.model.infer([row.to_model_input() for row in rows])
        output_by_row = {str(row.get("row_id")): row for row in model_outputs}
        row_results: list[dict[str, Any]] = []
        for row in rows:
            raw = output_by_row.get(row.row_id)
            output_artifact = Path(row.output_artifact)
            if raw is None:
                result = {
                    "row_id": row.row_id,
                    "job_id": row.job_id,
                    "item_id": row.item_id,
                    "batch_id": row.batch_id,
                    "stage_id": row.stage_id,
                    "agent_id": row.agent_id,
                    "worker_id": self.worker_id,
                    "attempt_id": row.attempt_id,
                    "status": "failed",
                    "output_artifact": str(output_artifact),
                    "error": {"code": "missing_model_output", "message": "model did not return this row"},
                }
            else:
                result = {
                    "schema": "ego.annotation.resident_batch_row.v0",
                    "row_id": row.row_id,
                    "job_id": row.job_id,
                    "item_id": row.item_id,
                    "batch_id": row.batch_id,
                    "stage_id": row.stage_id,
                    "agent_id": row.agent_id,
                    "worker_id": self.worker_id,
                    "attempt_id": row.attempt_id,
                    "status": str(raw.get("status") or "ok"),
                    "input_artifact": row.input_artifact,
                    "output_artifact": str(output_artifact),
                    "model_identity": self.model_identity,
                    "prediction": raw.get("prediction"),
                    "error": raw.get("error"),
                    "created_utc": utc_now_s(),
                }
                write_json(output_artifact, result)
            row_results.append(result)
        self.stats.batch_inference_count += 1
        self.stats.rows_inferred += len(rows)
        self.stats.last_inference_utc = utc_now_s()
        errors = [row for row in row_results if row.get("status") != "ok"]
        return {
            "schema": "ego.annotation.resident_batch_result.v0",
            "status": "ok" if not errors else "completed_with_errors",
            "job_id": str(batch.get("job_id")),
            "batch_id": str(batch.get("batch_id")),
            "stage_id": self.stage_id,
            "agent_id": agent_id,
            "worker_id": self.worker_id,
            "attempt_id": attempt_id,
            "model_identity": self.model_identity,
            "model_load_count": self.stats.model_load_count,
            "batch_inference_count": self.stats.batch_inference_count,
            "batch_size": len(rows),
            "rows_inferred": self.stats.rows_inferred,
            "elapsed_s": float(time.time() - started),
            "row_results": row_results,
            "gpu_residency": {
                "device": self.device,
                "resident_process": True,
                "model_loaded_once_for_worker_lifetime": self.stats.model_load_count == 1,
                "first_load_utc": self.stats.first_load_utc,
                "last_inference_utc": self.stats.last_inference_utc,
            },
            "errors": errors,
        }

    def write_worker_report(self, path: Path) -> Path:
        report = {
            "schema": "ego.annotation.resident_worker_report.v0",
            "worker_id": self.worker_id,
            "stage_id": self.stage_id,
            "model_identity": self.model_identity,
            "model_load_count": self.stats.model_load_count,
            "batch_inference_count": self.stats.batch_inference_count,
            "rows_inferred": self.stats.rows_inferred,
            "first_load_utc": self.stats.first_load_utc,
            "last_inference_utc": self.stats.last_inference_utc,
        }
        write_json(path, report)
        return path
