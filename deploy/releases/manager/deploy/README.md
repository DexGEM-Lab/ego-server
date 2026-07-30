# V22 Resident Deployment Archive

## Status and withdrawal

The custom V22 resident stack is archived and withdrawn. `configs/v22_resident_services.json` is deliberately marked `archive_only=true`, `deployment_state=withdrawn`, and `enabled=false`. The start script refuses by default; an operator must set `V22_ARCHIVED_DEPLOYMENT_OPT_IN=1` to acknowledge that this is a historical deployment. The stop script is dry-run by default and requires `V22_ARCHIVED_DEPLOYMENT_STOP=1` before it can send a signal.

The custom stack is not the current authority because a later Feishu Ray Serve deployment claims the same A800 GPUs with a different service protocol and DAG. Starting both stacks would create an external GPU ownership conflict even though their listener ports differ. The custom HaWoR service is the no-DROID variant; it must not be revived by adding DROID fields or a DROID fallback.

## Historical real-run evidence

This archive preserves a real side-loaded run, not only configuration files:

- On 2026-07-16, custom listeners on ports `18101` through `18104` reached `/readyz` with stable PIDs and `model_load_count=1` after checkpoint load and warmup.
- UniDepth processed two requests in resident batches with shapes such as `[32,3,1080,1920]` and `[24,3,1080,1920]`.
- WiLoR processed distinct-request hand crops through native crop tensors and preserved crop provenance.
- VGGT produced a distinct-request native sequence batch `[2,32,3,518,518]`.
- HaWoR-without-DROID processed distinct items through native temporal chunks such as `[8,16,3,256,256]` and `[6,16,3,256,256]`; its output recorded sparse-interpolated upstream camera provenance rather than claiming dense camera measurement.
- A full launcher was started, but the observed wave ended `completed_with_errors`: only 3/32 pipelines completed and 29 failed with connection resets; stage request emission also exceeded the five-second target. This is retained as failure evidence, not a completion claim.

Historical process and artifact evidence lives in `.memory/tasks/2026-07-16-resident-model-service-deployment/OPS.md`. It does not establish current residency.

## Current blocker and conflict

The last deployment audit on 2026-07-17 could not refresh server-local state because SSH to `115.190.235.210:57938` timed out from two independent local sessions. The custom process state, listener ownership, and GPU occupancy are therefore unknown until an operator runs the inventory on the A800 host.

Feishu revision 1370 describes the authoritative Ray Serve layout on `dex-a800` / `192.168.42.193`:

| Resource | Feishu Ray Serve | Withdrawn custom stack |
| --- | --- | --- |
| GPU 0 | UniDepth, port 28000 | UniDepth, port 18101 |
| GPU 1 | hands, port 28001 | WiLoR, port 18102 |
| GPU 2 | DROID, port 28002 | HaWoR without DROID, port 18103 |
| GPU 3 | HaWoR + infiller, port 28003 | VGGT, port 18104 |
| GPU 4 | WiLoR, port 28004 | unassigned |
| GPU 6 | Cosmos3, port 28006 | unassigned |

Ports are disjoint, but GPU 0 through GPU 3 ownership overlaps. Ray's resource accounting does not include unrelated external PIDs. The systems also disagree on protocol and DAG: the custom stack uses JSON plus CAS `/v1/{model}/infer` and no-DROID HaWoR, while Feishu uses multipart model-native endpoints and a DROID-inclusive path. A URL or port change is not an integration. Do not start the custom stack alongside Ray.

## Recovery inventory before any restore

Run these commands on the A800 host, after SSH access is restored, and save the raw output with the deployment incident record:

```bash
ss -ltnp | rg ':(18101|18102|18103|18104|28000|28001|28002|28003|28004|28006)\\b'
ps -eo pid,ppid,user,stat,lstart,args | rg 'serve_v22_resident_model|ray|uvicorn|1810[1-4]|2800[0-6]'
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
ray status
for port in 18101 18102 18103 18104 28000 28001 28002 28003 28004 28006; do
  curl --max-time 2 -fsS "http://127.0.0.1:${port}/healthz" || true
  printf '\\n'
done
```

Interpret the inventory jointly: identify every listener PID, verify its full command line, map each PID to a GPU process, and determine which stack owns the GPU before changing anything. The custom start script performs the port and visible-GPU checks itself after explicit opt-in and refuses on any conflict. It never stops another process. The custom stop script sends SIGTERM only when the PID file, exact model/port command-line arguments, and `ss` listener ownership all match; it remains a dry-run unless `V22_ARCHIVED_DEPLOYMENT_STOP=1` is set.

If restoration is approved after the inventory, also verify the exact gitlinks without downloading or running models:

```bash
git submodule status --cached
git config -f .gitmodules --get-regexp 'path|url'
python3 -m json.tool configs/v22_resident_services.json >/dev/null
python3 -m json.tool contracts/v22_resident_transport.schema.json >/dev/null
```

Restore only one authoritative stack, preserve the no-DROID HaWoR contract, and record the selected protocol, port map, GPU allocation, checkpoint hashes, readiness evidence, and rollback owner before enabling traffic.
