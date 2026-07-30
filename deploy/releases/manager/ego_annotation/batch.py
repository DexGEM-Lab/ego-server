"""Batch job ownership primitives for multi-item annotation jobs."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_STAGE_ID = "resident_batch_stage"


def utc_now_s() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_id(raw: str, *, fallback: str = "id") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._-")
    return cleaned[:96] or f"{fallback}_{uuid4().hex[:12]}"


@dataclass(frozen=True)
class BatchItemRequest:
    """One user datum inside a multi-input annotation job."""

    source_uri: str
    item_id: str | None = None
    media_kind: str = "rgb_video"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], index: int) -> "BatchItemRequest":
        if not isinstance(payload, dict):
            raise ValueError(f"items[{index}] must be an object")
        source_uri = payload.get("source_uri") or payload.get("video_uri") or payload.get("video")
        if not source_uri:
            raise ValueError(f"items[{index}].source_uri is required")
        metadata = payload.get("metadata") if payload.get("metadata") is not None else {}
        if not isinstance(metadata, dict):
            raise ValueError(f"items[{index}].metadata must be an object")
        return cls(
            source_uri=str(source_uri),
            item_id=str(payload["item_id"]) if payload.get("item_id") else None,
            media_kind=str(payload.get("media_kind") or "rgb_video"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class AnnotationBatchJobRequest:
    """A product annotation job containing multiple input items."""

    output_root: Path
    items: list[BatchItemRequest]
    job_id: str = field(default_factory=lambda: f"annotation_{uuid4().hex[:12]}")
    stage_id: str = DEFAULT_STAGE_ID
    batch_size: int = 2
    parallelism: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AnnotationBatchJobRequest":
        if not isinstance(payload, dict):
            raise ValueError("annotation batch job request must be an object")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        output_root = payload.get("output_root") or payload.get("artifact_root")
        if not output_root:
            raise ValueError("output_root is required")
        batch_size = int(payload.get("batch_size", 2))
        parallelism = int(payload.get("parallelism", 1))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if parallelism <= 0:
            raise ValueError("parallelism must be positive")
        metadata = payload.get("metadata") if payload.get("metadata") is not None else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return cls(
            output_root=Path(str(output_root)),
            items=[BatchItemRequest.from_mapping(item, index) for index, item in enumerate(raw_items)],
            job_id=clean_id(str(payload.get("job_id") or f"annotation_{uuid4().hex[:12]}"), fallback="job"),
            stage_id=clean_id(str(payload.get("stage_id") or DEFAULT_STAGE_ID), fallback="stage"),
            batch_size=batch_size,
            parallelism=parallelism,
            metadata=dict(metadata),
        )


class BatchJobManifest:
    """JSON-backed ownership manifest for job/item/batch attribution.

    The manifest is not an execution substitute. Its job is to preserve the
    identities that a resident model worker must consume and write back.
    """

    schema = "ego.annotation.batch_job.v0"

    def __init__(self, payload: dict[str, Any], path: Path | None = None) -> None:
        self.payload = payload
        self.path = path

    @classmethod
    def from_request(cls, request: AnnotationBatchJobRequest) -> "BatchJobManifest":
        job_root = (request.output_root / request.job_id).resolve()
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        now = utc_now_s()
        for index, item in enumerate(request.items):
            item_id = clean_id(item.item_id or f"item_{index:06d}", fallback="item")
            if item_id in seen:
                raise ValueError(f"duplicate item_id: {item_id}")
            seen.add(item_id)
            item_root = job_root / "items" / item_id
            items.append(
                {
                    "job_id": request.job_id,
                    "item_id": item_id,
                    "source_uri": item.source_uri,
                    "media_kind": item.media_kind,
                    "run_root": str(item_root),
                    "output_prefix": str(item_root),
                    "metadata": dict(item.metadata),
                    "status": "queued",
                    "created_utc": now,
                    "updated_utc": now,
                    "errors": [],
                }
            )
        batches: list[dict[str, Any]] = []
        for start in range(0, len(items), request.batch_size):
            batch_index = start // request.batch_size
            batch_items = items[start : start + request.batch_size]
            batch_id = clean_id(f"{request.job_id}_{request.stage_id}_batch_{batch_index:05d}", fallback="batch")
            rows = []
            for row_index, item in enumerate(batch_items):
                output_artifact = Path(str(item["run_root"])) / "stages" / request.stage_id / f"{batch_id}_{item['item_id']}.json"
                rows.append(
                    {
                        "row_id": f"{batch_id}_row_{row_index:05d}",
                        "job_id": request.job_id,
                        "item_id": item["item_id"],
                        "batch_id": batch_id,
                        "stage_id": request.stage_id,
                        "input_artifact": item["source_uri"],
                        "output_artifact": str(output_artifact),
                        "run_root": item["run_root"],
                        "status": "queued",
                    }
                )
            batches.append(
                {
                    "job_id": request.job_id,
                    "batch_id": batch_id,
                    "stage_id": request.stage_id,
                    "agent_id": None,
                    "worker_id": None,
                    "attempt_id": None,
                    "status": "queued",
                    "items": [row["item_id"] for row in rows],
                    "rows": rows,
                    "created_utc": now,
                    "updated_utc": now,
                    "errors": [],
                }
            )
        return cls(
            {
                "schema": cls.schema,
                "job_id": request.job_id,
                "status": "queued",
                "stage_id": request.stage_id,
                "job_root": str(job_root),
                "batch_size": request.batch_size,
                "parallelism": request.parallelism,
                "item_count": len(items),
                "batch_count": len(batches),
                "metadata": dict(request.metadata),
                "created_utc": now,
                "updated_utc": now,
                "items": items,
                "batches": batches,
                "agents": [],
                "errors": [],
            }
        )

    @classmethod
    def load(cls, path: Path) -> "BatchJobManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != cls.schema:
            raise ValueError(f"not a batch job manifest: {path}")
        return cls(payload, path=path)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        if target is None:
            raise ValueError("manifest path is required")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(target)
        self.path = target
        return target

    @property
    def items(self) -> list[dict[str, Any]]:
        items = self.payload.get("items")
        if not isinstance(items, list):
            raise ValueError("manifest items must be a list")
        return items

    @property
    def batches(self) -> list[dict[str, Any]]:
        batches = self.payload.get("batches")
        if not isinstance(batches, list):
            raise ValueError("manifest batches must be a list")
        return batches

    def counts(self, field: str = "batches") -> dict[str, int]:
        rows = self.batches if field == "batches" else self.items
        counts: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return dict(sorted(counts.items()))

    def claim_next_batch(self, agent_id: str) -> dict[str, Any] | None:
        for batch in self.batches:
            if batch.get("status") != "queued":
                continue
            attempt = int(str(batch.get("attempt_id") or "attempt_0000").rsplit("_", 1)[-1]) + 1
            batch["status"] = "running"
            batch["agent_id"] = agent_id
            batch["attempt_id"] = f"attempt_{attempt:04d}"
            batch["claimed_utc"] = utc_now_s()
            batch["updated_utc"] = utc_now_s()
            self._record_agent(agent_id)
            for row in batch.get("rows", []):
                if isinstance(row, dict):
                    row["status"] = "running"
                    row["agent_id"] = agent_id
                    row["attempt_id"] = batch["attempt_id"]
            self._refresh_status()
            return json.loads(json.dumps(batch))
        return None

    def complete_batch(self, batch_id: str, agent_id: str, worker_id: str, row_results: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._owned_batch(batch_id, agent_id)
        result_by_row = {str(row.get("row_id")): row for row in row_results}
        errors: list[dict[str, Any]] = []
        for row in batch.get("rows", []):
            if not isinstance(row, dict):
                continue
            result = result_by_row.get(str(row.get("row_id")))
            if result is None:
                row["status"] = "failed"
                row["error"] = {"code": "missing_row_result", "message": "resident worker did not return this row"}
            else:
                row["status"] = str(result.get("status") or "ok")
                row["output_artifact"] = result.get("output_artifact", row.get("output_artifact"))
                row["worker_id"] = worker_id
                if result.get("error"):
                    row["error"] = result["error"]
            if row.get("status") != "ok":
                errors.append({"row_id": row.get("row_id"), "item_id": row.get("item_id"), "error": row.get("error")})
        batch["worker_id"] = worker_id
        batch["status"] = "completed" if not errors else "completed_with_errors"
        batch["errors"] = errors
        batch["completed_utc"] = utc_now_s()
        batch["updated_utc"] = utc_now_s()
        self._refresh_items_from_batches()
        self._refresh_status()
        return json.loads(json.dumps(batch))

    def fail_batch(self, batch_id: str, agent_id: str, reason: str, *, retry: bool = True, max_attempts: int = 3) -> dict[str, Any]:
        batch = self._owned_batch(batch_id, agent_id)
        attempt_no = int(str(batch.get("attempt_id") or "attempt_0001").rsplit("_", 1)[-1])
        error = {"code": "batch_failed", "message": reason, "agent_id": agent_id, "at_utc": utc_now_s()}
        batch.setdefault("errors", []).append(error)
        if retry and attempt_no < max_attempts:
            batch["status"] = "queued"
            batch["agent_id"] = None
            for row in batch.get("rows", []):
                if isinstance(row, dict):
                    row["status"] = "queued"
                    row.pop("agent_id", None)
        else:
            batch["status"] = "failed"
            for row in batch.get("rows", []):
                if isinstance(row, dict):
                    row["status"] = "failed"
                    row["error"] = error
        batch["updated_utc"] = utc_now_s()
        self._refresh_items_from_batches()
        self._refresh_status()
        return json.loads(json.dumps(batch))

    def _owned_batch(self, batch_id: str, agent_id: str) -> dict[str, Any]:
        for batch in self.batches:
            if batch.get("batch_id") == batch_id:
                if batch.get("agent_id") != agent_id:
                    raise ValueError(f"batch {batch_id} is owned by {batch.get('agent_id')}, not {agent_id}")
                if batch.get("status") != "running":
                    raise ValueError(f"batch {batch_id} is not running")
                return batch
        raise ValueError(f"unknown batch_id: {batch_id}")

    def _record_agent(self, agent_id: str) -> None:
        agents = self.payload.setdefault("agents", [])
        if not isinstance(agents, list):
            self.payload["agents"] = agents = []
        for row in agents:
            if isinstance(row, dict) and row.get("agent_id") == agent_id:
                row["last_claim_utc"] = utc_now_s()
                return
        agents.append({"agent_id": agent_id, "first_claim_utc": utc_now_s(), "last_claim_utc": utc_now_s()})

    def _refresh_items_from_batches(self) -> None:
        rows_by_item: dict[str, list[dict[str, Any]]] = {}
        for batch in self.batches:
            for row in batch.get("rows", []):
                if isinstance(row, dict):
                    rows_by_item.setdefault(str(row.get("item_id")), []).append(row)
        for item in self.items:
            rows = rows_by_item.get(str(item.get("item_id")), [])
            statuses = {str(row.get("status")) for row in rows}
            if not rows:
                continue
            if statuses <= {"ok"}:
                item["status"] = "completed"
            elif "failed" in statuses:
                item["status"] = "failed"
            elif "running" in statuses:
                item["status"] = "running"
            elif "queued" in statuses:
                item["status"] = "queued"
            else:
                item["status"] = "completed_with_errors"
            item["updated_utc"] = utc_now_s()

    def _refresh_status(self) -> None:
        statuses = {str(row.get("status")) for row in self.batches}
        if not self.batches:
            status = "empty"
        elif statuses <= {"completed"}:
            status = "completed"
        elif "failed" in statuses:
            status = "completed_with_errors" if any(s == "completed" for s in statuses) else "failed"
        elif "completed_with_errors" in statuses:
            status = "completed_with_errors"
        elif "running" in statuses:
            status = "running"
        else:
            status = "queued"
        self.payload["status"] = status
        self.payload["updated_utc"] = utc_now_s()


class AnnotationBatchPlanner:
    """Creates a durable job manifest. It does not perform inference."""

    def create(self, request: AnnotationBatchJobRequest | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, dict):
            request = AnnotationBatchJobRequest.from_mapping(request)
        manifest = BatchJobManifest.from_request(request)
        manifest_path = request.output_root / request.job_id / "job_manifest.json"
        manifest.save(manifest_path)
        return {
            "job_id": request.job_id,
            "status": manifest.payload["status"],
            "manifest_path": str(manifest_path),
            "job_root": manifest.payload["job_root"],
            "item_count": manifest.payload["item_count"],
            "batch_count": manifest.payload["batch_count"],
            "parallelism": request.parallelism,
            "batch_counts": manifest.counts("batches"),
        }
