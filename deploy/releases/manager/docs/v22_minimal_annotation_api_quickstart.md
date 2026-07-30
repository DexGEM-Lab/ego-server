# V22 Annotation API Quickstart

The canonical total annotation API is:

```text
POST http://192.168.9.220:8091/v1/annotation-jobs
```

It accepts one video and returns after the current synchronous annotation pipeline finishes. The default `script` backend preserves the existing V22 behavior and returns terminal-oriented text unless JSON is explicitly requested.

## Shortest request

This exact command is the normal user contract:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/video.mp4' \
  'http://192.168.9.220:8091/v1/annotation-jobs?total_request_limit=128&algorithm_inflight_multiplier=2'
```

The response includes `status`, `job_id`, elapsed time, per-stage timings when available, and a result download URL/command.

The server keeps a rolling admission window for complete annotation jobs. The overall API call can declare `total_request_limit=128` and `algorithm_inflight_multiplier=2` directly in its query string:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/mnt/nas-222-download/egoscale_demo_30h/egoscale_tasks/20251121_1019_Recdf53_P1_S237208_task_3/20251121_1019_Recdf53_P1_S237208_task_3.mp4' \
  'http://192.168.9.220:8091/v1/annotation-jobs?total_request_limit=128&algorithm_inflight_multiplier=2'
```

The same outer capacity can be set with `ANNOTATION_TOTAL_REQUEST_LIMIT` and `--total-request-limit`. The legacy `ANNOTATION_ALGORITHM_INFLIGHT_MULTIPLIER` and `--algorithm-inflight-multiplier` fields remain accepted so older launchers keep working, but they no longer create per-algorithm request semaphores. The request's `total_request_limit` must match the running API process capacity; this prevents a request from claiming a larger or smaller global window than the semaphore actually enforces. A request holds one top-level slot from upload through terminal response. When the remote A800 backend is enabled, its client-side route proxy groups fully buffered requests at native batch caps and retries only service-rejected 429 items.

## One multipart request with API configuration

Add an optional multipart form field named `request`. Its value is a JSON object containing `job_id` and existing pipeline options:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/video.mp4' \
  -F 'request={"job_id":"demo-001","render_width":960,"run_captioning":true,"model_backend":"script"};type=application/json' \
  'http://192.168.9.220:8091/v1/annotation-jobs'
```

The service-selection fields are:

- `model_backend`: `script` or `feishu_ray`.
- `service_profile`: required with `feishu_ray`; the current known value is `feishu_ray_a800_server_local`.
- `service_endpoints`: optional with `feishu_ray`; maps `unidepth`, `hands_wilor`, `wilor`, `droid`, `hawor`, or `cosmos3` to an HTTP(S) base URL.

Supplying `service_profile` or `service_endpoints` with `model_backend=script` returns `422 service_configuration_requires_feishu_ray`; service configuration is never silently ignored.

Multipart `request` values override query/default option values. The uploaded `file` is always authoritative: including `video_uri` inside multipart request JSON returns `422 multipart_video_uri_forbidden`. Invalid JSON, a non-object request, unknown option names, unsupported backends, or invalid endpoint mappings return 422 with a concrete error code instead of being ignored.

### Current backend boundary

`model_backend=script` runs the existing direct scripted pipeline.

`model_backend=feishu_ray` currently returns HTTP 501 with code `feishu_ray_pipeline_adapter_not_implemented` before any script is launched. D3 UniDepth and D6 hands/WiLoR now have stage adapters that materialize the existing depth NPZ/QC and raw MANO candidate JSON/render inputs. The total backend remains closed because the deployed DROID finalize response has non-finite geometry and lacks the full pushed timeline, while D5 HaWoR and D9b Cosmos3 still lack complete artifact assembly. There is no script fallback for this selection.

The canonical deployment profile is [`configs/feishu_ray_services.json`](../configs/feishu_ray_services.json). It records server-local addresses/routes and `/-/healthz`; it is configuration, not an availability claim. Inside the A800 service network, the implemented partial slice can be run directly with `--model-execution feishu_ray`; requesting D4, D5, or dependent D7-D11 flags fails before the run root is changed and names every blocked stage.

## JSON compatibility

A JSON body with `video_uri` remains supported for a path visible to the service/runtime:

```bash
curl --noproxy '*' -X POST \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"path-job","video_uri":"/server/path/video.mp4","model_backend":"script"}' \
  'http://192.168.9.220:8091/v1/annotation-jobs'
```

Request JSON rejects unknown fields. Use `?response_format=json` to request the full machine-readable response for multipart uploads.

## Direct low-level Feishu-Ray model call

`scripts/call_feishu_ray_service.py` sends one model-native request. It is dependency-light and accepts one JSON file:

```bash
python scripts/call_feishu_ray_service.py --request-json /path/unidepth_request.json
```

Example `/path/unidepth_request.json`:

```json
{
  "base_url": "http://127.0.0.1:28000",
  "route": "/unidepth.infer",
  "metadata": {
    "ownership": {
      "request_id": "req-001",
      "job_id": "job-001",
      "item_id": "frame-000001",
      "stage_id": "unidepth.infer",
      "source_id": "frame-000001"
    },
    "model_revision": "unidepth-v2-vitl14-corrected",
    "spatial": {
      "source_size": {"width": 1920, "height": 1080},
      "model_size": {"width": 960, "height": 540},
      "color_space": "RGB"
    }
  },
  "arrays": [
    {
      "name": "rgb",
      "path": "/path/rgb_uint8_hwc.raw",
      "shape": [540, 960, 3],
      "dtype": "uint8"
    }
  ],
  "timeout_s": 120,
  "output_dir": "/path/direct_result"
}
```

Each array entry is either:

- an `.npy` path, with shape and dtype read from the file; or
- a raw binary path plus explicit positive `shape` dimensions and fixed-width numeric `dtype`; the byte count must match exactly. Response decoding supports zero-sized dimensions for valid empty model outputs.

The caller builds the deployed multipart form: JSON part `metadata`, followed by named `application/octet-stream` parts whose `Content-Disposition` carries `shape` and `dtype`. It validates URL/route, rejects unknown request/array fields and size mismatches, parses JSON or multipart responses, saves response arrays as `.npy`, and writes `response_report.json` under `output_dir`. Stage adapters use the same decoder through `call_service_arrays(...)` and validate ownership, tensors, geometry, and timeline in memory before writing legacy artifacts.

Direct calls have a narrower contract than the total API:

- They work only inside the **A800 server-local** network scope (`127.0.0.1`/the internal server network); workstation access to those URLs is not implied.
- They require model-native inputs such as decoded RGB HWC bytes, normalized WiLoR crops, DROID session frames, or HaWoR temporal tensors. Passing a video path does not make the service decode or orchestrate the video.
- Filesystem references belong only in the local caller's top-level `arrays[*].path` and `output_dir`. The deployed service metadata must describe ownership/model/spatial state, not ask the service to read caller-local paths.
- A successful low-level model call does not produce a complete annotation package or rendered video.

Canonical routes are:

```text
28000 /unidepth.infer
28001 /hands.detect
28004 /wilor.reconstruct
28002 /droid.create_session
28002 /droid.push_frame
28002 /droid.finalize
28003 /hawor.infer_tracks
28003 /hawor_infiller.fill
28006 /cosmos3.reason
```

## Result package and disabled batch ingress

A successful total API response provides a download URL for a zip containing the root `v22_overlay.mp4`, annotation manifest, render artifacts, and product bundle. `/v1/annotation-job-sets` remains disabled with HTTP 410; multi-video orchestration is a caller concern.
