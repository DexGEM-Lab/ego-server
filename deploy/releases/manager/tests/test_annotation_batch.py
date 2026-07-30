from __future__ import annotations

import json
from pathlib import Path

from ego_annotation.batch import AnnotationBatchJobRequest, AnnotationBatchPlanner, BatchJobManifest
from ego_annotation.resident_batch import FileFingerprintModel, ResidentBatchWorker


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_inputs(tmp_path: Path, count: int = 3) -> list[Path]:
    inputs = []
    for idx in range(count):
        path = tmp_path / "inputs" / f"clip_{idx}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video-{idx}".encode("utf-8"))
        inputs.append(path)
    return inputs


def test_batch_manifest_tracks_job_item_batch_stage_and_agent(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, 3)
    request = AnnotationBatchJobRequest.from_mapping(
        {
            "job_id": "job_parallel",
            "output_root": str(tmp_path / "out"),
            "stage_id": "contract_stage",
            "batch_size": 2,
            "parallelism": 4,
            "items": [
                {"item_id": "clip_a", "source_uri": str(inputs[0])},
                {"item_id": "clip_b", "source_uri": str(inputs[1])},
                {"item_id": "clip_c", "source_uri": str(inputs[2])},
            ],
        }
    )
    manifest = BatchJobManifest.from_request(request)
    assert manifest.counts("items") == {"queued": 3}
    assert manifest.counts("batches") == {"queued": 2}
    assert manifest.payload["item_count"] == 3
    assert manifest.payload["batch_count"] == 2
    assert [row["items"] for row in manifest.batches] == [["clip_a", "clip_b"], ["clip_c"]]

    claimed = manifest.claim_next_batch("agent_0")
    assert claimed is not None
    assert claimed["job_id"] == "job_parallel"
    assert claimed["stage_id"] == "contract_stage"
    assert claimed["agent_id"] == "agent_0"
    assert claimed["attempt_id"] == "attempt_0001"
    assert {row["agent_id"] for row in claimed["rows"]} == {"agent_0"}

    failed = manifest.fail_batch(claimed["batch_id"], "agent_0", "transient_capacity", retry=True, max_attempts=2)
    assert failed["status"] == "queued"
    reclaimed = manifest.claim_next_batch("agent_1")
    assert reclaimed is not None
    assert reclaimed["batch_id"] == claimed["batch_id"]
    assert reclaimed["attempt_id"] == "attempt_0002"


def test_batch_planner_persists_manifest(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, 5)
    result = AnnotationBatchPlanner().create(
        {
            "job_id": "batch_api_job",
            "output_root": str(tmp_path / "out"),
            "batch_size": 2,
            "parallelism": 8,
            "items": [{"source_uri": str(path)} for path in inputs],
        }
    )
    assert result["status"] == "queued"
    assert result["item_count"] == 5
    assert result["batch_count"] == 3
    assert result["parallelism"] == 8
    manifest_path = Path(result["manifest_path"])
    payload = read_json(manifest_path)
    assert payload["schema"] == "ego.annotation.batch_job.v0"
    assert payload["batches"][2]["items"] == ["item_000004"]


def test_batch_manifest_rejects_duplicate_item_ids(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, 2)
    request = AnnotationBatchJobRequest.from_mapping(
        {
            "job_id": "bad_batch",
            "output_root": str(tmp_path / "out"),
            "items": [
                {"item_id": "same", "source_uri": str(inputs[0])},
                {"item_id": "same", "source_uri": str(inputs[1])},
            ],
        }
    )
    try:
        BatchJobManifest.from_request(request)
    except ValueError as exc:
        assert "duplicate item_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate item_id should fail")


def test_resident_worker_loads_once_and_processes_multiple_true_batches(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, 3)
    request = AnnotationBatchJobRequest.from_mapping(
        {
            "job_id": "resident_job",
            "output_root": str(tmp_path / "out"),
            "stage_id": "contract_stage",
            "batch_size": 2,
            "items": [{"source_uri": str(path)} for path in inputs],
        }
    )
    manifest = BatchJobManifest.from_request(request)
    worker = ResidentBatchWorker(
        worker_id="worker_gpu0_contract",
        stage_id="contract_stage",
        model=FileFingerprintModel(),
        device="cpu",
    )

    results = []
    while True:
        claimed = manifest.claim_next_batch("agent_0")
        if claimed is None:
            break
        result = worker.infer_batch(claimed)
        manifest.complete_batch(claimed["batch_id"], "agent_0", worker.worker_id, result["row_results"])
        results.append(result)

    assert worker.stats.model_load_count == 1
    assert worker.stats.batch_inference_count == 2
    assert worker.stats.rows_inferred == 3
    assert manifest.payload["status"] == "completed"
    assert manifest.counts("items") == {"completed": 3}
    assert [result["batch_size"] for result in results] == [2, 1]
    assert all(result["model_load_count"] == 1 for result in results)

    for batch in manifest.batches:
        assert batch["worker_id"] == "worker_gpu0_contract"
        for row in batch["rows"]:
            output = read_json(Path(row["output_artifact"]))
            assert output["job_id"] == "resident_job"
            assert output["item_id"] == row["item_id"]
            assert output["batch_id"] == batch["batch_id"]
            assert output["stage_id"] == "contract_stage"
            assert output["agent_id"] == "agent_0"
            assert output["worker_id"] == "worker_gpu0_contract"
            assert output["prediction"]["input_exists"] is True
