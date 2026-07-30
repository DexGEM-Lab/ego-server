# Single-Item API-Ify Total Entry

## Purpose

The total entry remains `POST /v1/annotation-jobs`. A one-video request can select the frozen original `api-ify@3572551` service decomposition through `model_backend=api_ify`.

This path is separate from the legacy `script` pipeline and from the older `feishu_ray` caller. It invokes the transport-neutral typed DAG directly:

```text
POST /v1/annotation-jobs
  -> serve_v22_annotation_api.py
  -> annotation_remote_runner.py / run_v22_api_job_with_admission.py
  -> annotation_admission_proxy.py (manager-owned algorithm boundary)
  -> run_single_video_api.py
  -> FullVideoTimelineDriver
  -> ApiBackend / fixed localhost service routes
  -> PhysicalArtifactAdapter
  -> annotation_pipeline_manifest.json
  -> package_v22_annotation_result.py
```

The same entry preserves `item_batch_size=1`, native UniDepth/Hands/WiLoR batching, one ordered DROID session, 16-frame HaWoR chunks, 120-step coupled Infiller windows, full source duration, and fresh run-root ownership.

## Request Contract

The API-Ify route requires explicit diagnostic mode until the frozen DROID service proves full-K/native-depth consumption.

The deployed user-facing outer entry runs on `192.168.9.220:8091` and delegates heavy execution to the A800 release runtime through `ANNOTATION_REMOTE_*`. The A800 also exposes a runtime-local maintenance entry on port `8092`.

The user-facing server is launched from the local release checkout with the authoritative remote bundle configured explicitly:

```bash
ANNOTATION_REMOTE_HOST=zjh@115.190.235.210 \
ANNOTATION_REMOTE_REPO=/home/zjh/ego-api-runtime-release-da17415 \
ANNOTATION_REMOTE_OUTPUT_ROOT=/home/zjh/data/v22_api_release_da17415_remote_jobs \
ANNOTATION_REMOTE_UPLOAD_ROOT=/home/zjh/data/v22_api_release_da17415_remote_uploads \
ANNOTATION_REMOTE_PACKAGE_ROOT=/home/zjh/data/v22_api_release_da17415_remote_packages \
ANNOTATION_REMOTE_PYTHON=/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python \
/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python \
  scripts/serve_v22_annotation_api.py --host 0.0.0.0 --port 8091
```

The public API is fixed. The caller supplies only the video file; the manager internally selects API-Ify diagnostic execution and owns all admission values.

```bash
curl --noproxy '*' -sS --fail-with-body \
  -X POST 'http://192.168.9.220:8091/v1/annotation-jobs' \
  -H 'Accept: application/json' \
  -F 'file=@/path/to/video.mp4'
```

The manager generates a fresh job ID and output root. Internally it uses `model_backend=api_ify`, `diagnostic_monocular=true`, and the outer `total_request_limit=128`. The legacy `algorithm_inflight_multiplier=2` field remains accepted for launcher compatibility but no longer creates per-algorithm B semaphores or lifecycle slots. The client admission proxy fully buffers each request, groups route-compatible requests at the native batch cap, releases each group as concurrent independent HTTP POSTs, and retries only service-rejected 429 items. DROID retains local create→push→finalize ownership ordering; its six-replica/48-resident service capacity is surfaced by typed service backpressure rather than a multiplied client slot pool. Native UniDepth/Hands/WiLoR batches, HaWoR 16-frame chunks, Infiller 120-step windows, and `item_batch_size=1` remain unchanged.

A successful response includes `run_root`, `manifest_path`, `overlay_path`, `package_path`, `download_url`, and `summary`.

## Full-Dataset Traversal

Run the fixed zero-argument command on A800:

```bash
cd /home/zjh/ego-api-runtime-release-da17415 && \
/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python scripts/run_v22_api_egoscale30h_batch.py
```

The script uses the fixed dataset `/home/zjh/data/egoscale_demo_30h`, fixed manager `http://127.0.0.1:8092/v1/annotation-jobs`, and an automatically generated fresh timestamped output root. It recursively uploads every `.mp4` using the same file-only API contract.

## Artifacts

The API-Ify runner writes:

- `annotation_pipeline_manifest.json`;
- `run_result.json`;
- `state/v22_physical_state.npz`;
- `renders/v22_overlay.mp4`;
- `renders/v22_world_head_hand_3d.mp4`;
- `renders/v22_side_by_side.mp4`;
- `renders/physical_adapter_report.json`;
- `stage_captures/` with raw Infiller request/response exchanges.

The package adapter recognizes the `single_video_api_ify` manifest and packages only those artifacts that this path actually produces. It does not create legacy files by renaming or fabricating them.

## Admission Verification Boundary

The manager's admission evidence must come from its own active/queued/request-terminal records and from the algorithm service request traces it owns. The outer traversal's `queued` and `terminal` rows only prove that it called the manager. A response containing configured values `128` and `2` is not runtime evidence that either limit was enforced. For API-Ify, the client proxy records per-attempt and per-logical-request forwarding rows in `_algorithm_admission_events.jsonl`, including `batch_id`, `batch_size`, `attempt`, `retry_count`, and terminal status. Nonterminal 429 attempts are excluded from logical request-rate summaries and retained as explicit retry telemetry.

## Acceptance Boundary

For this operational integration checkpoint, acceptance concerns non-quality failures: request admission, stage transport, typed contracts, ordered DROID lifecycle, timeline completion, artifact writes, renderer execution, and package creation. Visual quality, sparse hand coverage, camera scale quality, and projection quality are recorded diagnostics and do not fail the no-error entrypoint check unless they trigger a downstream runtime or contract failure. A downstream failure caused by insufficient upstream quality may be retried with an explicitly relaxed model parameter or threshold, preserving the original failed root.

A successful command and decodable full-length videos prove that the total entry reached the API-Ify stages and renderer. They do not prove metric acceptance. The run result must preserve:

```text
accepted=false
diagnostic_only=true
scale_mode=up_to_scale_monocular
```

until DROID capability evidence proves native sensor-depth and full-K consumption. Physical render QC must still inspect projected hand content, finite MANO geometry, and full-frame coverage; a manifest or ZIP alone is not completion evidence.

## Related Plans

- `docs/single_item_api_ify_rewrite_design.md`: frozen single-item typed API DAG and service contracts.
- `docs/api_batch_parallel_processing_design.md`: total-entry ownership, multi-input extension, resident worker boundary, and package model.
- `docs/autoresearch_unidepth_droid_fps_plan.md`: controlled UniDepth/DROID sampling research; it is an offline comparison plan and does not change the default full-frame API-Ify path.
