# Typed model-service gateway & open-loop benchmark harness

This package adds a caller-facing typed gateway and an open-loop offered-load
benchmark harness on top of the existing Ray Serve model-serving foundation. Ray
remains internal; callers see stable model-native API names.

## Components

- `ego_annotation/serving/router.py` — typed multi-cluster router. Maps stable
  public API names (`unidepth.infer`, `hands.detect`, `wilor.reconstruct`,
  `droid.create_session`, `droid.push_frame`, `droid.finalize`,
  `hawor.infer_tracks`, `hawor_infiller.fill`, `cosmos3.reason`) to the Serve HTTP
  endpoint of their committed GPU group (GPU0/1/2/3/6). Canonical public lane ports
  are GPU0 `28000`, GPU1 `28001`, GPU2 `28002`, GPU3 `28003`, and GPU6 `28006`
  (overridable via `EGO_SERVE_HTTP_PORT_GPU<id>` / `EGO_SERVE_HOST`). The bare
  Cosmos3 process on `8001` is reachable only through an explicit baseline override.
  Per-API URL overrides support runtime lane discovery and tests.
- `ego_annotation/serving/gateway.py` — typed gateway client. Carries dense
  tensors/media as real multipart binary parts or the explicit
  `application/vnd.ego.binary-envelope` treatment, resolves in-cluster `ObjectRef`
  values lazily, preserves source timestamps + pixel/K transforms + per-item
  ownership, returns typed failures, and applies **bounded retries that do not hide
  overload** (overload surfaces as typed `BACKPRESSURE`, never an infinite retry).
- `ego_annotation/serving/hands_transport.py` — GPU1 typed gateway mapping. Hands
  sends `rgb [H,W,3] uint8`; WiLoR sends normalized `crop [3,256,256] float32`
  while handedness and crop/source-camera metadata remain JSON. Envelope responses
  retain the multipart detector/SAM2 arrays and all MANO parameter/mesh arrays.
- `ego_annotation/serving/hawor_transport.py` — GPU3 typed gateway mapping. HaWoR
  places `crop_batch [16,3,256,256] float32` and optional DROID pose/timestamp arrays
  in named binary parts while keeping track ID, transforms, observations and occlusion
  support in metadata. The infiller places DROID arrays in binary parts and preserves
  its nested two-hand MANO/observation sequence in metadata. Envelope responses keep
  the established multipart part names: HaWoR MANO arrays plus optional `world_lift`,
  and infiller MANO arrays plus `inferred` and `timestamps_s`.
- `ego_annotation/serving/benchmark/` — open-loop offered-load generator, payload
  manifests, metrics, artifacts, endpoint runner, plotting, runner, and a
  deterministic fake HTTP server.

## Open-loop method

Arrivals are scheduled at wall-clock times **independent of completions**. Under
overload, in-flight count grows and rejections rise while offered rate stays fixed —
the generator never self-limits (the defining property of open-loop vs closed-loop).

Per item the harness records: offered/submit/response times, outcome
(completed/admitted/rejected/in-flight), response latency (p50/p95/p99, separate
from amortized cost), server phase decomposition (admission/queue/dispatch/forward/
encoding), batch id/size/work units, batch wall time, amortized per-item cost,
model-load count, and payload hash.

Artifacts (raw, one point per plotted value): `items_<api>.jsonl`, `levels.csv`,
`batches_<api>.csv`, `manifest_<api>.json`, `run_manifest.json`.

Live runs require a real per-model payload corpus. Each
`<payload-dir>/<api>.json` source descriptor uses schema
`ego.benchmark-payload-source.v1`: `api_name`, `items[]`, and for every item,
`item_id`, `ownership`, binary `parts[]` (`name`, relative `file`, `shape`,
`dtype`), optional `spatial`, `model_revision`, `work_units`,
`source_timestamp_s`, and path-free metadata. The runner reads the referenced bytes,
recomputes hashes, and emits a path-free `manifest_<api>.json`; it never repeats a
source payload to manufacture load. DROID descriptors preserve the real
create/push/finalize session identifiers used by that benchmark lifecycle.

## Invocation

Smoke run against a deterministic in-process fake HTTP server (exercises real
multipart bytes; no Ray, no GPU):

```bash
python -m scripts.ray_serve_benchmark \
    --api unidepth.infer --levels 20,80,200 \
    --target-completed 15 --max-offered 15 \
    --out /tmp/ego_bench_smoke --fake-server
python -m scripts.ray_serve_benchmark_all \
    --apis unidepth.infer,hands.detect --levels 20 \
    --target-completed 4 --max-offered 4 --manifest-count 4 \
    --out /tmp/ego_bench_all_smoke --fake-server
```

Use `--wire-format envelope` to benchmark the envelope transport. GPU1 and GPU3
restart treatments are explicit and remain opt-in:
`python -m scripts.start_ego_model_services --groups gpu1 --gpu1-wire-format envelope`
or
`python -m scripts.start_ego_model_services --groups gpu3 --gpu3-wire-format envelope`.
Each enables matching dual-API telemetry so every result's
`batch_diagnostics.runtime_config_digest` attributes the wire treatment; omitting
these options preserves multipart and its existing production behavior.

Live run (probes each endpoint once, skips down lanes, writes a run manifest so the
harness can be re-invoked as lanes become available):

```bash
EGO_SERVE_HOST=dex-a800 python -m scripts.ray_serve_benchmark_all \
    --payload-dir /vePFS-Mindverse/.../ray_serve_payloads \
    --target-completed 100 --out /vePFS-Mindverse/.../ray_serve_benchmarks

The unified command probes each selected endpoint exactly once, then runs its
open-loop offered-load sweep only for lanes observed live. It reads public endpoints
only and never starts, stops, or edits a model deployment.
```

## Isolated UniDepth vacant-GPU scaling

`python -m scripts.plan_unidepth_scaling_experiment` builds an **experiment-only**
contract for one or more operator-authorized vacant physical GPUs. It copies the
production UniDepth interpreter, checkpoint environment, and bound deployment import
path while allocating a new CUDA-visible physical GPU, independent Ray component /
worker / metrics / HTTP port blocks, a short AF_UNIX-safe isolated `/tmp/eud/...`
temp directory, and a unique replica id. It never edits `COMMITTED_GPU_GROUPS` or the
canonical router.

```bash
python -m scripts.plan_unidepth_scaling_experiment \
  --experiment-id unidepth-scale-YYYYMMDDTHHMMSSZ \
  --gpus 4,5 \
  --application-release /vePFS-Mindverse/.../releases/<immutable-release> \
  --source-sha <40-char-source-sha> --checkpoint-digest <checkpoint-digest> \
  --run-root /vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks
```

The plan requires a non-symlink immutable release directory whose `RELEASE.json`
attests both `release_sha` and the requested `source_sha`; it rejects
`runtime/current`. It can print commands for review, or `--execute` starts the exact
scoped heads/deployments as blocking commands and performs one real typed UniDepth
readiness request per replica (`--readiness-payload-dir`). Readiness validates the
server-produced experiment/replica/GPU/GCS/HTTP/temp/PID/release/model/checkpoint
identity rather than a generic HTTP status. Any failure shuts down only heads started
by that invocation, using the explicit dashboard address then exact-temp-dir PID
scope; it never uses global `ray stop`. Production proxy checks use Ray `/-/healthz`,
not a model `/health` route.

After all explicit experimental lanes pass their one-shot health probes, run the
already-launched endpoints with real payloads only:

```bash
python -m scripts.run_unidepth_scaling_benchmark \
  --experiment-id unidepth-scale-YYYYMMDDTHHMMSSZ \
  --gpus 4,5 \
  --run-root /vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks \
  --application-release /vePFS-Mindverse/.../releases/<immutable-release> \
  --source-sha <40-char-source-sha> --checkpoint-digest <checkpoint-digest> \
  --payload-dir /vePFS-Mindverse/.../exact-unidepth-payloads \
  --gpu-measurement-artifact /vePFS-Mindverse/.../gpu_nvml_samples.json \
  --active-window-profiler-artifact /vePFS-Mindverse/.../ncu_or_cupti.json \
  --levels 2,4,6,8,10 --max-offered 100 --manifest-count 501
```

It fails early if the explicit payload manifest/hash corpus is absent, reserves one
distinct payload for typed readiness, and rejects a run without timestamped GPU
utilization/memory samples. It assigns each independent UniDepth request round-robin,
preserves the server trace replica identity, records generator CPU, CPU-collate/H2D/
CUDA-model/D2H/validation/encoding spans, allocator allocated/reserved/peaks, and
de-duplicated physical batches. CUDA timing synchronizes only under the experimental
telemetry environment; CPU emits explicit unavailable CUDA fields. NVML/dmon samples
do not establish bandwidth: the manifest labels bandwidth attribution unavailable
unless an NCU/CUPTI artifact is supplied. The fresh result root contains
`items_*.jsonl`, `batches_*.csv`, `scaling_level_*.json`, and `run_manifest.json`;
generator CPU is in every scaling level JSON. Persist one-replica and multi-replica `ScalingLevelResult`s with
`write_scaling_result`, then use `compare_replica_scaling` only for the equivalent
one-versus-two-or-more comparison. A gain is interpretable only when the raw
per-replica records retain distinct payload hashes, success/rejection outcomes, and
latency distributions.

## Isolated DROID vacant-GPU scaling

`python -m scripts.manage_droid_scaling_experiment` exposes explicit `plan`,
`execute`, and `stop` subcommands. The planner accepts only GPU4/GPU5/GPU7,
requires a fresh physical UUID for every GPU, reuses the exact production
`ray_serve_hawor` interpreter, checkpoint, revision, and device, and allocates
disjoint component/worker/metrics/HTTP ports and `/tmp/zjheds/<experiment>/gpuN`
roots. The deleted production DROID checkout is never used for a cold start: build
one immutable recovered source bundle first. Its manifest records every source file,
the six-core fingerprint, provenance, and amendment
`recovered-hawor-droid-core-v1`; the source directory is digest-named and the
planner sets `EGO_DROID_REPO` only to its `droid_slam` child. The checkpoint argument
must be the exact production path; its digest is derived from bytes.

```bash
python -m scripts.build_droid_source_release build \
  --candidate-root /vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel/third_party/HaWoR/thirdparty/DROID-SLAM \
  --output-root /vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/droid-source-releases \
  --origin-evidence-json /vePFS-Mindverse/user/yiwen/user-home/zjh/droid_candidate_b_evidence.json
# Re-run this byte-level check before planning or after transport:
python -m scripts.build_droid_source_release verify \
  --release /vePFS-Mindverse/.../droid-source-releases/<source-digest>
```

```bash
python -m scripts.manage_droid_scaling_experiment plan \
  --experiment-id droid-gpu7-YYYYMMDDTHHMMSSZ --gpus 7 --gpu-uuids GPU-... \
  --application-release /vePFS-Mindverse/.../releases/<content-digest> \
  --droid-source-release /vePFS-Mindverse/.../droid-source-releases/<source-digest> \
  --source-sha <40-char-source-sha> \
  --checkpoint /vePFS-Mindverse/.../droid/droid.pth \
  --run-root /vePFS-Mindverse/.../ray_serve_benchmarks > droid-plan.json

python -m scripts.manage_droid_scaling_experiment execute \
  <same plan arguments> --readiness-rgb <real-320x568-rgb.bin> \
  --readiness-mask <real-320x568-float32-static-mask.bin> \
  --fx 408.96 --fy 408.96 --cx 284 --cy 160
```

Execution performs the fresh NVML/port preflight in the same scoped transaction as
Ray start. Readiness creates a typed session, pushes one real frame, and finalizes
the one-keyframe session to a typed terminal `unresolved` tombstone. Before importing
`droid_net`, the worker re-verifies the source release; after import it proves that
`droid_net.__file__` lies under the bundle. Create, push, and terminal finalize must
all return the same worker-derived application release digest/module root, source
dependency digest/root/amendment, checkpoint digest, PID, CUDA UUID, physical GPU,
endpoint, and model revision. Multi-replica benchmarks require the same non-empty
source dependency digest. Failure and `KeyboardInterrupt` roll back only the exact
replicas owned by that invocation. `stop --temp-dir ...` accepts only exact DROID
experiment roots and never runs global `ray stop`.

Run D1/D2/D4 against the plan's explicit endpoints without editing the canonical
router:

```bash
python -m scripts.run_droid_scaling_benchmark \
  --endpoint http://127.0.0.1:32000 \
  --runtime-identities droid-plan.json --gcs-address 127.0.0.1:30000 \
  --run-root /vePFS-Mindverse/.../fresh-droid-run \
  --sessions 1,2,4 --wave-rates 0.5,1,2,4,8
```

Session creation is round-robin and each resulting session remains sticky to one
verified runtime identity through every push and terminal finalize. Run-aligned
NVML starts immediately before the first D1/D2/D4 level, labels each sample with the
level, release, experiment, physical GPU and UUID, and stops after the final level.
The validator rejects missing interval overlap, stale identity, or wrong UUID.

## Endpoint liveness

The endpoint runner probes each configured endpoint **once** (single GET to
`/health`); it never polls and never waits for model lanes. Down endpoints are
recorded in `run_manifest.json` and skipped for that run. Re-running the harness
re-probes once, so it can be invoked again as more lanes come up.

## Tests

`tests/test_benchmark_harness.py` uses the deterministic fake aiohttp HTTP server
to exercise actual multipart bytes and open-loop scheduling end-to-end:

```bash
python -m pytest tests/test_benchmark_harness.py tests/test_model_serving_foundation.py -q
```
