"""Raw JSONL/CSV artifact writers for open-loop benchmarks.

Artifacts are the evidence of a run. Every plotted point must trace to a raw result
file. We write:

* ``items.jsonl``  — one JSON line per settled item (the full per-item record).
* ``levels.csv``   — one row per (api, offered-intensity) level summary.
* ``batches.csv``  — one row per distinct batch (batch_size/work_units/wall/model_load).
* ``manifest.json`` — the payload manifest (distinct payload hashes + provenance).
* ``run_manifest.json`` — run config, endpoint observations, and result file paths.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ego_annotation.serving.benchmark.manifest import PayloadManifest
from ego_annotation.serving.benchmark.metrics import ItemRecord, LevelSummary


def write_items_jsonl(path: Path, records: Iterable[ItemRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec.to_dict(), separators=(",", ":")) + "\n")
            count += 1
    return count


def write_levels_csv(path: Path, summaries: Sequence[LevelSummary]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = LevelSummary.csv_columns()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for s in summaries:
            row = [s.to_dict().get(col) for col in columns]
            writer.writerow(["" if v is None else v for v in row])
    return len(summaries)


@dataclass
class BatchRow:
    batch_id: str
    api_name: str
    replica_id: str | None
    batch_size: int
    batch_work_units: int
    batch_wall_ms: float | None
    model_load_count: int | None
    member_item_ids: tuple[str, ...] = ()


def batch_rows_from_records(records: Sequence[ItemRecord]) -> list[BatchRow]:
    """Collapse per-item records into one row per distinct batch.

    Items that share a ``batch_id`` are members of the same server batch; their
    batch_size/work_units/wall/model_load must agree (the server assigned them
    together). We keep the first seen values and collect member item ids.
    """
    rows: dict[str, BatchRow] = {}
    for r in records:
        if r.batch_id is None or r.batch_size is None:
            continue
        if r.batch_id in rows:
            rows[r.batch_id].member_item_ids += (r.item_id,)
            continue
        rows[r.batch_id] = BatchRow(
            batch_id=r.batch_id,
            api_name=r.api_name,
            replica_id=None,
            batch_size=r.batch_size,
            batch_work_units=r.batch_work_units or 0,
            batch_wall_ms=r.batch_wall_ms,
            model_load_count=r.model_load_count,
            member_item_ids=(r.item_id,),
        )
    return list(rows.values())


def write_batches_csv(path: Path, records: Sequence[ItemRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = batch_rows_from_records(records)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["batch_id", "api_name", "batch_size", "batch_work_units", "batch_wall_ms",
             "model_load_count", "member_count"]
        )
        for row in rows:
            writer.writerow([
                row.batch_id, row.api_name, row.batch_size, row.batch_work_units,
                "" if row.batch_wall_ms is None else row.batch_wall_ms,
                "" if row.model_load_count is None else row.model_load_count,
                len(row.member_item_ids),
            ])
    return len(rows)


def write_manifest_json(path: Path, manifest: PayloadManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_manifest(), handle, indent=2)


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
