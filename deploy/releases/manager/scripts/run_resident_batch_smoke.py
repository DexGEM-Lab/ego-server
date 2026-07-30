#!/usr/bin/env python3
"""Run a dependency-light resident batch worker over a multi-item manifest.

This smoke proves the scheduling contract: one worker loads once, consumes real
batch requests, writes per-item row artifacts, and preserves ownership fields.
It does not run heavy perception models locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ego_annotation.batch import AnnotationBatchJobRequest, AnnotationBatchPlanner, BatchJobManifest
from ego_annotation.resident_batch import FileFingerprintModel, ResidentBatchWorker
from ego_annotation.artifacts import write_json


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.request is not None:
        request = AnnotationBatchJobRequest.from_mapping(load_json(args.request))
        plan = AnnotationBatchPlanner().create(request)
        manifest_path = Path(plan["manifest_path"])
    else:
        if args.manifest is None:
            raise RuntimeError("either --request or --manifest is required")
        manifest_path = args.manifest
    manifest = BatchJobManifest.load(manifest_path)
    worker = ResidentBatchWorker(
        worker_id=args.worker_id,
        stage_id=str(manifest.payload["stage_id"]),
        model=FileFingerprintModel(),
        device=args.device,
        model_config={"implementation": "dependency_light_contract_smoke"},
    )
    batch_results: list[dict[str, Any]] = []
    while True:
        claimed = manifest.claim_next_batch(args.agent_id)
        if claimed is None:
            break
        result = worker.infer_batch(claimed)
        manifest.complete_batch(str(claimed["batch_id"]), args.agent_id, worker.worker_id, result["row_results"])
        manifest.save(manifest_path)
        batch_results.append(result)
    report_path = Path(str(manifest.payload["job_root"])) / "reports" / "resident_worker_report.json"
    worker.write_worker_report(report_path)
    summary = {
        "schema": "ego.annotation.resident_batch_smoke.v0",
        "status": manifest.payload["status"],
        "job_id": manifest.payload["job_id"],
        "manifest_path": str(manifest_path),
        "worker_report": str(report_path),
        "model_load_count": worker.stats.model_load_count,
        "batch_inference_count": worker.stats.batch_inference_count,
        "rows_inferred": worker.stats.rows_inferred,
        "batch_result_count": len(batch_results),
        "batch_counts": manifest.counts("batches"),
        "item_counts": manifest.counts("items"),
        "claim_scope": "Dependency-light resident worker contract smoke; not a physical annotation model.",
    }
    summary_path = Path(str(manifest.payload["job_root"])) / "reports" / "resident_batch_smoke_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=None, help="Batch job request JSON. Creates a new job manifest.")
    parser.add_argument("--manifest", type=Path, default=None, help="Existing job manifest to consume.")
    parser.add_argument("--agent-id", default="agent_000")
    parser.add_argument("--worker-id", default="resident_contract_worker_000")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
