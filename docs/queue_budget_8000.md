# 8000-image internal queue budget

The annotation model fleet uses Ray Serve's `max_queued_requests` as the
service-side pending-request queue. The queue target is expressed in the work
unit accepted by each endpoint, not as one shared numeric request count.

| Deployment | Endpoint work unit | Queue cap | Equivalent pending images/frames |
|---|---|---:|---:|
| UniDepth | one RGB frame/request | 8000 | 8000 |
| Hands | one RGB frame/request | 8000 | 8000 |
| WiLoR | one RGB frame/request | 8000 | 8000 |
| HaWoR tracks | fixed 16-frame chunk/request | 500 | 8000 |
| HaWoR infiller | fixed 120-frame horizon/request | 67 | 8040 |
| Cosmos3 | bounded group of up to 8 media items/request | 1000 | 8000 |
| DROID | atomic request containing up to 256 frames | 32 | 8192 |

Ceiling conversion is intentional: flooring would leave a queue below 8000
images. `max_ongoing_requests` and native model batch caps remain compute
controls and are unchanged.

## Admission behavior

`deploy/releases/manager/scripts/annotation_admission_proxy.py` reserves the
conservative route weight before reading a request body, then streams the body to
a private spool and waits on a route-local weighted queue with an 8000-image
budget. Only bounded internal forwarding workers open model-service connections.
A 256-thread internal handler bound plus an 8192-listener backlog prevents
offered load from allocating one resident thread and spool body per blocked
request. External submissions are not assigned a route semaphore; the internal
queue and listener provide backpressure. Explicit HTTP 429 admission rejection
is retried while retaining the same queue weight. Transport failures, HTTP 5xx,
and payload rejection are terminal ambiguous/error outcomes and are never
replayed automatically; operators reconcile them before retrying.

## DROID deployment rule

DROID replicas share CUDA IPC weights. A queue-cap change restarts a Ray Serve
replica under the current Ray version. For a DROID group, deploy the owner lane
with the canonical IPC handle path absent so it recreates the handles, verify it
is healthy, and only then deploy the two importer lanes. Never redeploy all six
DROID lanes in parallel and never use global `ray stop`.
