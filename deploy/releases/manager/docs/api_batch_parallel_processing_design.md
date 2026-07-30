# Algorithm Batch Parallel Processing Design

This design extends the current MVP annotation API from one input video per job to a multi-input annotation job. It preserves the existing single-video API and remote A800 pipeline as the item compatibility path, then adds a wave-major resident batch layer for model stages. The production topology is 16 input items per end-to-end wave: wave 0 runs prepare -> resident model stages -> item agents -> packages, closes with package/failure evidence, and only then wave 1 enters. A stage-major sweep over all 256 items is diagnostic only and is not a valid final delivery for this requirement. The resident worker layer is a hard requirement: manifest scheduling and 16 cold-start item workers are not enough.

## Current MVP Path To Preserve

The user-facing path today is synchronous and single-video:

1. `scripts/serve_v22_annotation_api.py` exposes `POST /v1/annotation-jobs` and `GET /v1/downloads/{filename}`.
2. Upload handling writes one uploaded video through `scripts/annotation_api_uploads.py`.
3. `run_annotation_job()` creates one `job_id` and one `run_root`.
4. If `ANNOTATION_REMOTE_HOST` is set, `run_remote_job()` calls `scripts/annotation_remote_runner.py`.
5. The remote runner copies the input when needed, then runs `scripts/run_v22_minimal_annotation_pipeline.py` on the A800-side repo.
6. The single-video pipeline writes `annotation_pipeline_manifest.json`, stage logs, state files, render videos, optional product bundle, and packageable outputs.
7. `scripts/package_v22_annotation_result.py` creates the downloadable zip and the API returns the download URL.

This stays as the compatibility path for one-video jobs and as a fallback item adapter while stages are migrated to resident workers. It is not the resident batch mechanism because each model stage is still a fresh subprocess.

The frozen single-item `api-ify@3572551` path is the authoritative physical item adapter for a one-video API-Ify request. The total entry selects it with `model_backend=api_ify` and explicit `diagnostic_monocular=true` until DROID full-K/native-depth consumption is proven. Its contract and package boundary are documented in `docs/single_item_api_ify_total_entry_integration.md`; the multi-item wave scheduler must preserve this item adapter rather than route the item through the legacy Feishu caller.

## Added Job And Ownership Model

A multi-input annotation job introduces stable internal identities:

| id | Meaning |
|---|---|
| `job_id` | One user-submitted annotation job. |
| `item_id` | One input datum inside the job, usually one video. |
| `batch_id` | One scheduler unit for a stage, frame window, or crop group. |
| `stage_id` | The algorithm stage producing an artifact family. |
| `agent_id` | The child agent or runner that owns a claimed batch. |
| `worker_id` | The resident worker process/actor that executes inference. |
| `attempt_id` | One execution attempt for retry and failure isolation. |

The implementation adds `ego_annotation/batch.py` with `AnnotationBatchJobRequest`, `BatchJobManifest`, and `AnnotationBatchPlanner`. The manifest records job/items/batches/agents/errors and keeps per-row `job_id/item_id/batch_id/stage_id/agent_id/worker_id/attempt_id/input_artifact/output_artifact/status` fields.

The manifest is a provenance and coordination artifact. It does not by itself satisfy batch inference.

## Input Preprocessing Rules

For every uploaded or remote video, preprocessing must:

1. Validate readability, suffix, byte size, hash, FPS, frame count, duration, dimensions, and requested time window.
2. Create a stable `item_id` and an item-scoped run root under the parent job root.
3. Preserve original source path, upload filename, hash, user metadata, sidecars, and parameter snapshot.
4. Normalize timing into the existing raw-frame manifest contract.
5. Emit explicit item errors for unreadable media or invalid windows without blocking unrelated items.

The minimum implementation can call `prepare_v22_single_video_run.py` per item because that is the current single-video bootstrap. Later optimization can batch decoding, but item frame identity must remain explicit.

## Batch Construction

Batch size and boundaries are stage-specific:

| Batch type | Stages | Boundary rule |
|---|---|---|
| Item batch | legacy single-video V22 compatibility path | One item per claimed runner. |
| Frame-window batch | UniDepth, DROID, SAM2, frame render windows | Rows carry `item_id`, frame range, frame manifest path, output prefix. |
| Crop batch | WiLoR/HaWoR hand crops, open-vocabulary detectors | Rows carry `item_id`, frame index, crop id, bbox, source image path. |
| Semantic-window batch | caption/evaluator/action segmentation | Rows carry `item_id`, time span, sidecar source, output path. |

All stage batches preserve item boundaries. Video A frame 0 and video B frame 0 may be in one model batch, but their row outputs must write to different item roots and carry different `item_id`s.

## Scheduler And Child Agent Ownership

The parent process owns global lifecycle:

1. Accept multi-input job.
2. Build one master manifest, then partition it into deterministic 16-item waves.
3. Start resident model worker pools once for the job when GPU stages are enabled.
4. For each wave, run prepare, resident model stages, post-resident item agents, package generation, and wave closure.
5. Refuse to enter the next wave until the current wave has either package evidence or explicit failure/unresolved evidence for every item.
6. Aggregate item outputs, wave summaries, resident worker reports, GPU/token summaries, and final job-level package/report indexes.

Child agents own claimed units within the current wave. The resident workers own stage batches for that wave while preserving item boundaries. A child agent must claim atomically, write heartbeat/status/artifacts/errors, and mark only its own item or batch completed or failed.

This follows the jiahong-dev parallel pattern: parent schedules and summarizes; child agents own item/batch execution state. It adds a hard wave barrier because package timing is part of the user-visible contract.

## Resident Model Worker Contract

The hard architecture requirement is implemented by `ego_annotation/resident_batch.py` as a contract and local smoke implementation. Production A800 workers should implement the same shape with real model loaders.

A resident worker lifecycle is:

1. Start one remote worker/actor for a model family and GPU for the job, not for one item or one wave.
2. Load model weights once at worker startup.
3. Register `worker_id`, `stage_id`, `model_identity`, `gpu_id`, load timestamp, and `worker_lifetime_scope`.
4. Accept repeated wave request batches containing rows with item boundaries.
5. Write per-row outputs under each item output prefix.
6. Record `model_load_count`, `server_model_load_count`, wave/request IDs, batch/sequence counts, batch size, rows inferred, GPU residency, and per-item output mapping.

Minimum response evidence:

```json
{
  "worker_id": "unidepth_actor_gpu0_v1",
  "stage_id": "unidepth_v2_depth",
  "model_load_count": 1,
  "batch_inference_count": 3,
  "batch_size": 8,
  "gpu_residency": {
    "device": "cuda:0",
    "resident_process": true,
    "model_loaded_once_for_worker_lifetime": true
  },
  "row_results": [
    {
      "job_id": "annotation_...",
      "item_id": "item_000001",
      "batch_id": "...",
      "stage_id": "unidepth_v2_depth",
      "agent_id": "agent_00",
      "worker_id": "unidepth_actor_gpu0_v1",
      "output_artifact": ".../unidepth_v2_depth.npz",
      "status": "ok"
    }
  ]
}
```

The local `FileFingerprintModel` smoke worker is intentionally not a physical annotation model. It proves the resident worker protocol: one worker instance loads once, consumes multiple real batch requests, writes per-item outputs, and preserves traceability. It is not counted as UniDepth/WiLoR/HaWoR completion.

## First Real A800 Resident Stage

The first real resident stage is UniDepth.

Mechanism: the remote `scripts/run_v21_unidepth.py` already loaded UniDepth once, then looped through frames from one run root. `scripts/run_v22_resident_unidepth_batch.py` moves that model-loading block into worker initialization and exposes a multi-item batch request: rows carry `job_id/item_id/batch_id/stage_id/agent_id/frame_idx/rgb_path/output_prefix`, tensors are stacked per batch, and outputs are assembled back into separate item depth archives.

Verified A800 smoke:

- Run root: `/home/zjh/data/resident_unidepth_smoke_20260709`.
- Worker report: `/home/zjh/data/resident_unidepth_smoke_20260709/reports/resident_unidepth_worker_report.json`.
- Result: `model_load_count=1`, `batch_inference_count=2`, `batch_sizes=[2,2]`, `rows_inferred=4`.
- GPU residency evidence: `memory_allocated_mb=1363.919921875`, `memory_reserved_mb=3350.0` on CUDA.
- Item outputs: `item_a` and `item_b` each wrote independent `unidepth_v2_depth_resident.npz` and `qc_unidepth_v2_resident.json` under their own item run roots.

This smoke satisfies the minimum real-model resident batch mechanism for one stage. It is not yet wired into the 256-item production batch runner; that integration is the next implementation step.

WiLoR and HaWoR should follow after UniDepth because their current wrappers are more tightly coupled to per-clip scripts and detector/MANO environment state.

## Stage Migration Plan

M0: single-item boundary record

- Deliverable: documented path `serve_v22_annotation_api.py -> annotation_remote_runner.py -> run_v22_minimal_annotation_pipeline.py -> package_v22_annotation_result.py`.
- Status: done by inspection reports under `.pi-subagents/artifacts/outputs/.../reports/`.

M1: multi-input job manifest

- Deliverable: `ego_annotation/batch.py` creates job/item/batch manifests and supports batch claim/complete/fail ownership.
- Validation: unit tests prove stable `job_id/item_id/batch_id/stage_id/agent_id/attempt_id` and duplicate item rejection.

M2: resident batch protocol smoke

- Deliverable: `ego_annotation/resident_batch.py` and `scripts/run_resident_batch_smoke.py` run one worker over multiple batches with `model_load_count=1` and `batch_inference_count>1`.
- Validation: unit tests inspect written row artifacts and worker counters.

M3: API batch coordinator

- Deliverable: the existing `/v1/annotation-jobs` accepts multiple items without exposing internal version names; single-item behavior remains unchanged.
- Validation: two uploaded/remote videos create one parent job, two item roots, batch manifest, and job-level package.

M4: A800 resident UniDepth

- Deliverable: a remote resident UniDepth worker consumes frame batches from multiple items and writes per-item depth outputs.
- Validation: worker report proves load-once residency and multiple true batch inferences.

M5: 16-agent production run with resident stages

- Deliverable: 16 child agents manage batches or item groups while model workers stay resident. Parent summarizes item status, batch status, worker status, GPU usage, token usage, progress, errors, and packages.
- Validation: 256 selected items complete without uncaught errors; report includes resident worker counters, not only GPU wrapper launches.

## Output And Error Aggregation

Final job package should contain:

1. `job_manifest.json`: all items, batches, agents, workers, attempts, statuses, output paths, and errors.
2. `items/<item_id>/...`: item pipeline outputs and product manifests.
3. `tables/item_index.csv`: item status/output/error index.
4. `tables/batch_index.csv`: stage batch status and ownership index.
5. `tables/stage_attempts.csv`: attempts, failures, retries, model identity.
6. `reports/visual_report.html`: progress, GPU use, token use, worker residency, item sample, and artifact links.
7. `reports/gpu_summary.json`, `reports/worker_summary.json`, `reports/batch_summary.json`.

Failure handling is scoped:

| Failure level | Effect |
|---|---|
| Item preprocessing failure | That item fails before scheduling; other items continue. |
| Stage row failure | Affected item/stage row fails; batch can complete with row errors. |
| Agent failure | Claimed batch is requeued until max attempts, then failed. |
| Resident worker failure | Parent restarts or marks worker failed, then requeues affected batches with new attempts. |
| Shared environment failure | Parent stops scheduling and writes a job-level blocker. |

## Existing 256-Item Run Evidence

A previous A800 run exists at:

`/home/zjh/data/v22_parallel_runs/v22_parallel_egoscale30h_short256_20260708T152300Z`

Evidence observed on the A800 host:

- Dataset mirror: `/home/zjh/data/egoscale_demo_30h`, 1757 videos.
- Manifest: `batch_manifest.json`, 256 entries.
- Report: `reports/batch_summary.json` shows `entry_count=256`, `completed=256`, `overlays=256`, `packages_for_manifest_completed=256`.
- Report: `reports/worker_summary.json` shows 16 `api_short_*` deterministic workers and per-worker completed counts.
- Report: `reports/gpu_summary.json` shows all 8 A800 GPUs used through `gpu_wrapper_events.jsonl` fallback, with UniDepth and WiLoR launches attributed to agents.
- Report: `reports/visual_report.html` explicitly states the run validates scheduling/ownership/resource reporting over the existing MVP API path and does not validate resident model actor reuse.
- Package: `reports/v22_api_batch_delivery_index.zip` contains report/index artifacts.

Conclusion: this run is valid evidence for 16-agent parallel item scheduling, item output isolation, full 256 item completion, and monitoring/report packaging. It is not evidence for model residency or true algorithm-stage batch inference because each item still used the existing single-video pipeline and model stage subprocesses.

## Verification From User Perspective

A finished batch implementation must prove all three layers:

1. User layer: one job can contain multiple videos; each item has visible status, errors, output paths, and downloadable artifacts.
2. Algorithm layer: at least one stage batch is consumed by a resident model worker, with model load count lower than inference batch count and item boundaries preserved inside the batch.
3. Operations layer: 16 child agents/batch owners can process 256 items while reporting GPU usage, token usage, batch progress, worker residency, failures, and artifact indexes.

Any delivery missing layer 2 remains incomplete even if layer 1 and layer 3 pass.
