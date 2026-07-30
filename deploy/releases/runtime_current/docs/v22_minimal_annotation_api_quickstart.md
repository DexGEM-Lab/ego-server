# V22 Annotation API Quickstart

Current user-facing service:

```text
http://192.168.9.220:8091
```

Architecture:

```text
user machine -> local/workstation HTTP API -> remote A800 V22 pipeline -> local download URL
```

The local API accepts uploads and serves downloads. Heavy inference still runs on the remote A800 server through SSH/SCP.

## Start The Local API

Local repo:

```text
/home/zjh/ego-annotation-mvp
```

Local API venv:

```text
/home/zjh/ego-annotation-local-api-venv
```

Start command:

```bash
cd /home/zjh/ego-annotation-mvp

tmux new-window -t ego_annotation -n local_annotation_api -c "$PWD" \
  'ANNOTATION_PUBLIC_BASE_URL=http://192.168.9.220:8091 \
   ANNOTATION_OUTPUT_ROOT=/home/zjh/data/local_v22_api_jobs \
   ANNOTATION_PACKAGE_ROOT=/home/zjh/data/local_v22_api_downloads \
   ANNOTATION_REMOTE_HOST=115.190.235.210 \
   ANNOTATION_REMOTE_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annotaion-jiahong-dev \
   ANNOTATION_REMOTE_OUTPUT_ROOT=/home/zjh/data/v22_api_jobs \
   ANNOTATION_REMOTE_UPLOAD_ROOT=/home/zjh/data/v22_api_uploads \
   ANNOTATION_REMOTE_PACKAGE_ROOT=/home/zjh/data/v22_api_downloads \
   /home/zjh/ego-annotation-local-api-venv/bin/python scripts/serve_v22_annotation_api.py --host 0.0.0.0 --port 8091'
```

Endpoints:

```text
POST http://192.168.9.220:8091/v1/annotation-jobs
GET  http://192.168.9.220:8091/v1/downloads/<package>.zip
```

The API is synchronous: the POST returns after the remote job finishes.

## Simplest User Command

Users only need to upload a file. The service now defaults to:

```text
- uploaded filename as saved filename
- full video length
- raw video width
- camera trajectory on
- HaWoR metric hands on
- hybrid hands on
- D8 GT-free drift self-calibration hypothesis on
- D9b captioning stage on
- D10 self-consistency QC on
- D11 evaluator stage on
- product bundle on
```

Command:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/to/local/video.mp4' \
  http://192.168.9.220:8091/v1/annotation-jobs
```

## Optional Job Name

If you want to control the output package name, add `job_id` as a query parameter:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/to/local/video.mp4' \
  'http://192.168.9.220:8091/v1/annotation-jobs?job_id=my_upload_job'
```

## Response

The default upload response is terminal-oriented text:

```text
status=ok
job_id=<job_id>
progress=16/16 stages completed
overlay=720 frames, 24.000000s, 1920x1080
elapsed_s=...
download_command=curl --noproxy '*' -O http://192.168.9.220:8091/v1/downloads/<job_id>_annotation_result.zip
download_url=http://192.168.9.220:8091/v1/downloads/<job_id>_annotation_result.zip
```

Download the result by running the printed `download_command`.

For full machine-readable details, request JSON explicitly:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/to/local/video.mp4' \
  'http://192.168.9.220:8091/v1/annotation-jobs?response_format=json'
```

The zip contains final overlay at the root:

```text
v22_overlay.mp4
```

It also contains:

```text
annotation_pipeline_manifest.json
package_manifest.json
renders/*.mp4
product_bundle/...
```

## Optional Sidecars

D9b captions and D11 GT-backed evaluator metrics require explicit sidecar files. For multipart uploads, pass local sidecar paths as query parameters; the local proxy uploads those files to the remote run directory before execution:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/to/local/video.mp4' \
  'http://192.168.9.220:8091/v1/annotation-jobs?actions_json=/path/to/actions.json&head_gt=/path/to/head_gt.json&hand_gt=/path/to/hand_gt.json'
```

If `actions_json` or `captions_json` is omitted, D9b writes `source_absent_no_caption_rows`. If `head_gt` and `hand_gt` are omitted, D11 writes evaluator readiness and prediction diagnostics only; no-GT reprojection/jitter rows are marked `prediction_diagnostic`, not GT-backed `measured` metrics.

## Compatibility Mode: Raw Body Upload

The old raw-body upload still works, but it is no longer the simplest user path:

```bash
curl --noproxy '*' -X POST \
  'http://192.168.9.220:8091/v1/annotation-jobs?job_id=my_upload_job&filename=input.mp4&start_s=0&end_s=1&render_width=960&run_camera_trajectory=true&run_hawor_metric_hands=true&run_hybrid_hands=true&write_product_bundle=true' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary '@/path/to/local/video.mp4'
```

## Compatibility Mode: Remote Server Path

If a video is already on the remote server, the same endpoint still accepts JSON:

```bash
curl --noproxy '*' -X POST http://192.168.9.220:8091/v1/annotation-jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "my_path_job",
    "video_uri": "/home/zjh/data/egoscale_demo_30h/egoscale_tasks/20260115_1931_Rec7ad6_P0_S2790c9_task_8/20260115_1931_Rec7ad6_P0_S2790c9_task_8.mp4"
  }'
```

Caveat: in proxy mode, the JSON `video_uri` fallback uses local path existence to decide whether a path is already remote, so a mirrored server path can be misclassified as a local upload. The primary upload path above is unaffected.

## Verified Smoke

Verified local-proxy upload job:

```text
local_proxy_upload_smoke_20260708_a
```

Verified local download URL:

```text
http://192.168.9.220:8091/v1/downloads/local_proxy_upload_smoke_20260708_a_annotation_result.zip
```

This smoke proved:

```text
local upload -> local API -> remote A800 execution -> local download package
```

The returned/downloaded zip root `v22_overlay.mp4` matches the final remote run overlay hash, and package metadata records `render_source=hybrid_hand_state`.

## Scope Boundary

This is a synchronous alpha proxy service. It proves local HTTP upload, remote execution, local result packaging, and local HTTP download for smoke runs. The default pipeline now runs D8/D9b/D10/D11 stages as frozen-artifact stages: D8 writes a GT-free image-plane drift/bias hypothesis, D9b writes source-backed caption rows when an action/caption sidecar is supplied and a no-source artifact otherwise, D10 recomputes self-consistency over frozen outputs, and D11 records evaluator readiness/metrics with GT inputs when supplied.

It does not prove auth, async queue/status, external internet exposure, concurrency, production security, 5mm accuracy, accepted final MANO quality, or GT-backed evaluator accuracy when no GT sidecar is supplied.
