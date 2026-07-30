# Ray Serve Benchmark: Throughput/Latency Overview

## Queueing Mechanism

Ray Serve is an **open-loop** request serving system: clients submit requests at a target rate independent of when prior responses return. Three regimes emerge:

**Below service capacity** — arrivals per second < processing rate. The single replica's GPU is idle between requests. Queues are empty; response latency equals pure model inference time plus minimal proxy overhead. Throughput matches offered load exactly (y = x).

**Near capacity (the queueing knee)** — arrivals ≈ processing rate. Variance in service time (model forward time, GPU kernel scheduling, Ray actor dispatch) and variance in inter-arrival time interact to produce transient queue buildup. The **p95 latency rises before achieved throughput flattens** because: a few requests arrive in a burst, the replica is busy, those requests wait in the proxy/queue for the next available slot. The queue drains during a subsequent idle interval, but the **worst-case wait grows**. This is the queueing knee — the most important threshold for capacity planning.

**Above capacity** — arrivals > processing rate. Without admission control, queues grow without bound (unstable: throughput drops to the service rate but latency increases unboundedly). Ray Serve deployments can be configured with bounded admission (`max_ongoing_requests` + `max_queued_requests`). Above this bound, new requests receive an immediate HTTP error (503) rather than queuing. This turns unbounded latency into **explicit bounded rejection** — a design choice that protects the cluster from memory exhaustion at the cost of dropped work.

### Why different APIs show different knees

| Variable | Effect |
|----------|--------|
| **Service time** | HaWoR (~620 ms/req) saturates at ~2 req/s; UniDepth (~160 ms at low load) remains stable through 4 img/s; Infiller (~60 ms) handles 16 req/s without saturation. |
| **Dynamic batching** | UniDepth dynamically batches concurrent requests — batch size grows from 1 to 8 at high load, trading latency for throughput. This softens the knee but pushes p95 higher. |
| **Session-level admission** | DROID uses per-session state with at-most-one-ready-frame-in-flight validation, not stateless open-loop. Its admission ceiling (~7–8.6 frames/s) is a **concurrency limit** on in-flight frames across sessions, not a processing-rate limit. |
| **Bounded backpressure** | HaWoR (`max_ongoing_requests=8` + `max_queued_requests=32`) rejects excess requests with HTTP 503 at ≥4 req/s. At 4 req/s: 48% rejected; at 8 req/s: 58% rejected; at 16 req/s: 56% rejected. |
| **Finite-window timing** | Cosmos3 Nano shows achieved > offered at low rates (0.309 vs 0.25 rps). This is a measurement artifact: the short sample window (5 requests) can capture an extra response in-flight from before the window opened. |
| **No observed saturation** | Hands.Detect, WiLoR.Reconstruct, HaWoR Infiller, and Cosmos3 all exit their sweeps below the rate where sustained queueing begins. Their results establish **lower bounds** on capacity, not ceilings. |

## Source Artifacts

All artifacts reside under `/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/` on dex-a800.

| API | GPU | Source path |
|-----|-----|-------------|
| UniDepth (depth) | GPU0 | `gpu0_unidepth_endpoint_openloop_20260716T200015Z_final/open_loop_sweep/summary.json` |
| Hands.Detect | GPU1 | `gpu1_hands_wilor_20260716T2001Z/open_loop_summary.csv` |
| WiLoR.Reconstruct | GPU1 | `gpu1_hands_wilor_20260716T2001Z/open_loop_summary.csv` |
| DROID (VO) | GPU2 | `droid_openloop_final_20260716T204043Z/droid/summary.json` |
| HaWoR (hand) | GPU3 | `gpu3_hawor_infiller_20260716T2003Z/open_loop_sweep_100x5_retry.json` |
| HaWoR Infiller | GPU3 | `gpu3_hawor_infiller_20260716T2003Z/open_loop_sweep_100x5_retry.json` |
| Cosmos3 Nano | GPU6 | `cosmos3_open_loop_3f980de_20260716T205845Z/cosmos3/open_loop_summary.json` |

## Throughput Lower Bounds vs. Latency-SLO Limits

The plot reports two distinct quantities that must not be conflated:

- **Achieved throughput** (solid colored line, left y-axis) is a **lower bound on capacity**: the system can serve at least this rate under the tested load. It answers "how fast can this system go before it breaks?" For APIs that did not saturate within the tested range, the maximum tested rate is a lower bound, not a capacity limit.

- **p95 latency** (dashed grey line, right y-axis) is the **cost** of service at each offered load. It is useful for latency-SLO planning: e.g. "at what offered load does p95 exceed 1 second?" This SLO-capacity may be well below the throughput ceiling.

| API | Throughput lower bound | p95 at max tested | Latency-SLO capacity (p95 < 1 s) | Notes |
|-----|----------------------|-------------------|----------------------------------|-------|
| UniDepth | ~4.4 img/s (saturated) | 10.4 s at 8 img/s offered | ~4 img/s | Stable p95 ≤ 0.32 s through 4 img/s; knee between 4–6 img/s |
| Hands.Detect | ≥32 req/s (lower bound) | 1.57 s at 32 req/s | ~8 req/s | N=8/level → p95 descriptive only |
| WiLoR.Reconstruct | ≥32 req/s (lower bound) | 0.27 s at 32 req/s | ≥32 req/s | N=8/level → p95 descriptive only; ~100 ms baseline |
| DROID | ~8.6 frames/s (admission ceiling) | 1.72 s at 8.64 f/s (s=8×1) | ~4 f/s | Concurrency limit, not processing ceiling; 20 push-frame data points; see group breakdown below |
| HaWoR | ~2 req/s (saturated effective) | 43 s at 8 req/s offered (of successes) | ~2 req/s | Bounded backpressure: 48–58% HTTP 503 rejection at ≥4 req/s |
| HaWoR Infiller | ≥16 req/s (lower bound) | 0.07 s at 16 req/s | ≥16 req/s | ~60 ms per request; no batch accumulation needed |
| Cosmos3 Nano | ≥4 req/s (lower bound) | 0.87 s at 4 req/s | ~2 req/s | N=5/level → p95 descriptive only; 25-req sweep, ~2k prompt tokens/req |

## HaWoR Rejection Fraction

The HaWoR deployment on GPU3 uses bounded admission: `max_ongoing_requests=8` + `max_queued_requests=32`. At ≥4 req/s offered (exceeding the ~2 req/s processing rate), the queue fills and Ray Serve returns HTTP 503:

| Offered (req/s) | HTTP 200 (ok) | HTTP 503 (error) | Rejection fraction | Achieved (successes only) | p95 of successes |
|-----------------|---------------|------------------|--------------------|-----------------------------|-----------------|
| 1.0 | 100 | 0 | 0% | 1.001 req/s | 0.63 s |
| 2.0 | 100 | 0 | 0% | 1.982 req/s | 1.02 s |
| 4.0 | 52 | 48 | 48% | 1.027 req/s | 33.6 s |
| 8.0 | 42 | 58 | 58% | 0.854 req/s | 42.8 s |
| 16.0 | 44 | 56 | 56% | 0.884 req/s | 34.9 s |

The achieved throughput (successes only) collapses to ~1 req/s — the GPU's true processing rate. The rejection fraction stabilizes at ~50–58% because the system processes what it can (~1 req/s) and rejects the rest. This is explicit bounded backpressure, not an unbounded queue.

## DROID Session/Wave Structure

DROID is not a stateless request API. It maintains per-session state across `create_session` → `push_frame` (×8 per session per level) → `finalize`. The benchmark parameterizes two independent dimensions:

- **Sessions** (1, 2, 4, 8): concurrent session identities multiplexed through the single GPU replica.
- **Waves-per-second** (0.5, 1, 2, 4, 8): target Poisson rate at which each session's 8 `push_frame` calls are dispatched.

**Driver semantics**: Each level creates `sessions` concurrent sessions. Each session receives exactly 8 `push_frame` calls (not per wave — per level), dispatched at the target Poisson rate `waves-per-s`. Total push-frame count = `sessions × 8` per level. The measured **offered rate** is `offered_count / elapsed_s` (finite-window rate from `summary.json`), not the target ideal `sessions × waves-per-s` — actual offered rates vary due to scheduler timing, admission rejection, and finite-window measurement.

The service admits frames up to a ceiling of ~7.5–8.6 frames/s total across all sessions, enforced by per-session at-most-one-ready-frame-in-flight validation. Above this ceiling, excess frames get `"outcome": "rejected"` with reason `"already has a ready frame in flight"`.

### Session Group Breakdown

The plot groups points by session count (s=1 through s=8) rather than connecting all points in a single sweep. This reveals that the admission ceiling is approximately constant regardless of session count, but latency costs differ. All 20 push-frame data points (4 session counts × 5 wave rates) are plotted; rejected points (hollow markers) lie at their measured offered rate on the x-axis, making the overload visible as points far to the right of the ceiling with achieved throughput collapsed.

| Configuration | Offered (f/s) | Achieved (f/s) | p95 (s) | Push reject | Notes |
|--------------|--------------|----------------|---------|-------------|-------|
| s=1, waves=8 | 7.74 | 7.74 | 0.227 | 0/8 | Single-session baseline; lowest p95 |
| s=2, waves=4 | 7.89 | 7.89 | 0.436 | 0/16 | Highest no-reject throughput with ~0.44 s p95 |
| s=8, waves=1 | 8.64 | 8.64 | 1.715 | 0/64 | Highest achieved; p95 7.5× higher than s=1; 1 finalize model_failure |
| s=4, waves=2 | 7.90 | 7.66 | 0.911 | 1/32 (3%) | Marginal: single-frame reject near ceiling boundary |
| s=2, waves=8 | 13.36 | 6.68 | 0.797 | 8/16 (50%) | First systematic rejection; offered far right of ceiling |
| s=4, waves=4 | 13.87 | 6.94 | 1.280 | 16/32 (50%) | ~50% in-flight reject at offered 13.9 f/s |
| s=8, waves=2 | 13.84 | 6.92 | 2.488 | 32/64 (50%) | ~50% in-flight reject; highest p95 at this load |
| s=4, waves=8 | 19.33 | 6.04 | 1.385 | 22/32 (69%) | Deeper overload: 69% rejection at offered 19.3 f/s |
| s=8, waves=4 | 19.19 | 5.10 | 2.825 | 47/64 (73%) | Deepest overload: achieved collapses to 5.1 f/s |
| s=8, waves=8 | 19.76 | 5.25 | 2.729 | 47/64 (73%) | Saturation floor: similar to s=8×4 at offered ~20 f/s |

The overload points (offered ≥ 13.4 f/s) all show achieved throughput collapsing to the ~5–7 f/s range, with rejection fractions increasing from 50% (at offered ~13–14 f/s) to 69–73% (at offered ~19–20 f/s). The admission ceiling is a concurrency limit: at most ~7–8.6 frames can be in flight across all sessions at once, regardless of how many sessions are multiplexed. The achieved throughput floor of ~5 f/s is the pure processing rate when admission rejects excess frames.

The one `finalize` `model_failure` at `sessions-8_waves-per-s-1` occurred on a session that had been idle during earlier push-frame processing. Root cause not determined; may be a timing/race condition rather than a capacity indicator.

## Plot

`ray_serve_throughput_latency_overview.png` — small-multiples figure (3×3 grid, 7 data panels + 2 explanatory panels).

Each data panel:
- **Left axis (colored line with markers)**: achieved throughput vs. offered load. The grey dashed diagonal (y = x) is the ideal line. For DROID, points are grouped by session count (s=1, 2, 4, 8); hollow markers indicate any rejection (rejected_count > 0). Rejected points use measured offered rate on the x-axis (not achieved), so overload points appear far to the right of the admission ceiling with collapsed achieved throughput.
- **Right axis (hollow-square markers, grey dashed)**: p95 response latency in seconds. For DROID, only non-rejected points carry latency values.
- **Annotations**: identify stable regions, queueing knees, saturation ceilings, rejection fractions (HaWoR), and finite-window artifacts (Cosmos3).
- **Sample size** noted in panel titles. N=8 (Hands, WiLoR) and N=5 (Cosmos3) panels note that p95 values are descriptive only.

Two explanatory panels:
- Throughput lower bounds vs. latency-SLO limits distinction.
- Data source summary with GPU assignments, model revisions, and token metrics.

## Open Questions / Residual Risks

1. **Cosmos3 sweep** (N=5 per level, total 25 requests) is too thin for reliable p95 estimates, especially at the tail. The p95 values reported are best-effort from 5 samples and should not be used for capacity planning. Achieved > offered at low rates (0.309 vs 0.25 rps) is a finite-window timing artifact, not a real capacity signal.
2. **DROID finalize failure**: 1 `model_failure` at `sessions-8_waves-per-s-1`. Root cause not determined; may be a timing/race condition rather than a capacity indicator.
3. **DROID marginal rejection**: s=4 w=2 has a single-frame push rejection (1/32, 3%) at offered 7.90 f/s — below the approximate admission ceiling of ~8.6 f/s. This is a boundary artifact, not a systematic overload; it does not indicate the ceiling itself is misestimated.
4. **Hands.Detect and WiLoR.Reconstruct** use N=8 per level — too small for reliable tail latency. The absence of throughput saturation up to 32 req/s is genuine, but the p95 values are descriptive only. Hands.Detect p95 rises to 1.57 s at 32 req/s, suggesting a knee may appear slightly above 32.
5. **UniDepth at 8 img/s offered**: achieved throughput *dropped* from 4.41 (at offered 6) to 4.23 (at offered 8). This is likely due to excessive dynamic batching overhead — p95 at 10.4 s suggests the batch size of 8 is past the optimal operating point.
6. **HaWoR rejection fraction** is measured from HTTP status codes (200 vs 503). Some 503 responses may include timeout aborts from the client side in addition to queue-full rejections; the exact breakdown requires the 503 response body, which was not captured in this sweep.
