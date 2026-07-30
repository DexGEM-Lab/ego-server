# Autoresearch Plan: UniDepth and DROID Frame-Rate Sampling

## 1. Research Question

Measure the throughput and output differences caused by reducing the source-frame schedule used by UniDepth and DROID from the full source timeline to target rates of 20 FPS, 15 FPS, and 10 FPS.

The throughput target is operational: determine whether UniDepth and DROID approach the separately measured throughput of HaWoR, WiLoR, and `hands.detect` under the same service deployment and load policy. The output comparison is descriptive: quantify how sparse outputs differ from the full-frame reference. This plan does not classify any condition as more accurate or less accurate.

The raw video, service revision, model weights, camera preprocessing, coordinate transforms, and output decoder remain fixed. Only the selected source-frame schedule for UniDepth and/or DROID changes.

## 2. Conditions

Use one immutable full-frame reference and three target schedules:

```text
full: every source frame
20: deterministic source-frame schedule at 20 target FPS
15: deterministic source-frame schedule at 15 target FPS
10: deterministic source-frame schedule at 10 target FPS
```

Run the following minimum matrix:

| Condition | UniDepth | DROID | Purpose |
|---|---:|---:|---|
| `full_full` | full | full | paired reference |
| `20_full`, `15_full`, `10_full` | sparse | full | isolate UniDepth sampling |
| `full_20`, `full_15`, `full_10` | full | sparse | isolate DROID sampling |
| `20_20`, `15_15`, `10_10` | sparse | sparse | paired operating points |

If runtime allows, complete the 4 x 4 factorial matrix. The paired rows alone cannot identify whether a difference comes from UniDepth, DROID, or their interaction through calibration and camera evidence.

Each condition uses a fresh run root and a fresh DROID session. No DROID session, cached pose, depth archive, keyframe set, or serialized model output is reused across conditions.

## 3. Frame Schedule Contract

Do not create a re-encoded low-FPS proxy video. Keep the original video byte-identical and create a schedule manifest consumed by the stage.

For each target rate `r`, choose monotonically increasing source indices by target timestamps:

```text
source_timestamp[k] = k / r
source_index[k] = nearest source frame to source_timestamp[k]
```

Deduplicate indices, retain their original timestamps, and record the exact rounding rule. The schedule must include:

```json
{
  "source_video_sha256": "...",
  "source_fps": 30.0,
  "source_frame_count": 900,
  "source_duration_s": 30.0,
  "target_fps": 15.0,
  "selected_source_frame_indices": [0, 2, 4],
  "selected_source_timestamps_s": [0.0, 0.066667, 0.133333],
  "selection_rule": "nearest_monotonic_unique_source_index"
}
```

Skipped source frames are explicitly `unprocessed`. They are not interpolated, copied from a neighboring prediction, or presented as measured depth/camera state. Full-video render or downstream consumers must preserve this distinction if a sparse condition is sent beyond the two measured stages.

DROID receives selected RGB frames in increasing source-time order with the original timestamps. It must retain its ordered single-flight state mutation and one terminal finalize. A 15 FPS run has larger temporal gaps; those gaps are part of the experiment and must not be hidden by synthetic intermediate frames.

UniDepth receives only selected frames for the sparse condition, while preserving its native request batch axis and service-side batching policy. The request manifest records selected frame ownership so the sparse result can be joined back to common source indices without assuming complete coverage.

## 4. Throughput Measurements

Measure all conditions on the A800 with the same service revision, GPU assignment, process policy, warmup policy, request timeout, and admission settings. Run at least three timed repetitions per condition after one discarded warmup. Record the raw repetitions, not only an average.

Use stage timing boundaries that separate:

```text
decode and frame selection
request construction and serialization
HTTP wait and upstream service time
model/postprocess response decode
archive/materialization
complete stage wall time
```

Report both native work-unit rate and source-time rate:

```text
selected_frame_fps = selected_frame_count / model_stage_seconds
source_time_fps = source_frame_count / complete_stage_wall_seconds
realtime_factor = source_duration_seconds / complete_stage_wall_seconds
```

`selected_frame_fps` answers how quickly the stage processes the work it was given. `source_time_fps` and `realtime_factor` prevent a 10 FPS run from appearing superior solely because it processes fewer frames.

For DROID also record:

```text
track_call_count
keyframe_count
keyframe_added_count
frontend/backend update counts
finite dense-output count
finalize wall time
peak allocated and reserved memory
session create/push/finalize timestamps
```

For UniDepth also record:

```text
request count
native request batch sizes
selected frame count per request
service response latency
valid depth/confidence count
```

## 5. Throughput Parity Reference

Measure or recover a fresh matching-load reference for:

```text
hands.detect
WiLoR
HaWoR track
HaWoR infiller
```

Use the same A800 service deployment and admission configuration. Keep the original Feishu standalone pressure results as a separate reference column; do not merge standalone endpoint request/s with full-video stage FPS.

Compare matching units:

| Stage | Primary unit |
|---|---|
| UniDepth | selected input frames |
| DROID | ordered `track` frames and complete session wall time |
| `hands.detect` | detector frames and complete request wall time |
| WiLoR | hand crops and crop-model forwards |
| HaWoR track | native 16-frame crop chunks and complete track wall time |
| HaWoR infiller | temporal windows |

For each condition, report the gap to the median reference rate using the same denominator. The result is a throughput comparison, not an acceptance gate and not an accuracy conclusion.

## 6. Output Difference Measurements

Treat `full_full` as a paired reference generated from the same raw video. It is not ground truth.

### UniDepth

At common selected source indices, compare sparse depth to the full-frame depth:

```text
valid coverage difference
absolute depth delta: median, p95, max
relative depth delta on finite positive reference pixels
confidence delta
per-frame valid-pixel fraction
spatially aggregated depth maps and residual maps
```

Also report a robust scale-normalized comparison as a separate diagnostic. This prevents a global scale offset from being confused with local spatial disagreement. Preserve both raw metric deltas and normalized deltas; neither is a judgment label.

### DROID

Align each sparse result to the full-frame result using the first common camera pose as the fixed gauge. Do not align each frame independently.

Compare at common selected source indices:

```text
world-camera translation delta
rotation geodesic delta
relative pose delta from the first common frame
cumulative trajectory length difference
per-frame translation and rotation time series
keyframe index set and keyframe-added differences
dense-output coverage
finite/invalid state counts
disparity delta when both outputs provide it
```

Record both the raw gauge-aligned differences and the coverage differences. A missing sparse frame is not a zero delta.

### Cross-stage effects

For paired conditions, compare the downstream evidence that consumes UniDepth or DROID:

```text
calibration K and scale artifacts
DROID metric-scale inputs derived from UniDepth
HaWoR camera-evidence join coverage
shared source-frame ownership and timestamp joins
```

These are observations of changed inputs, not an accuracy verdict.

## 7. Representative Videos

Select a fixed, recorded set before the first run:

1. low-motion clip with at least one long static interval;
2. high camera-motion clip with several DROID keyframes;
3. hand-occlusion or fast hand-motion clip with changing detector visibility.

For every selected video record path, SHA-256, source FPS, frame count, duration, resolution, and selection rationale. Use the same videos for every condition. Add more clips only as a declared second cohort; do not silently change the cohort after seeing results.

## 8. Autoresearch Loop

Each iteration changes one mechanism or one experiment parameter and produces a comparable artifact set.

### Baseline iteration

1. Freeze the raw-video manifest and service revisions.
2. Run `full_full` three times.
3. Verify full source-frame coverage, one DROID finalize, finite response accounting, and timing fields.
4. Store the median baseline and all raw repetitions.

### Sampling iterations

1. Generate the 20/15/10 schedule manifests.
2. Run the one-factor UniDepth rows with DROID full.
3. Run the one-factor DROID rows with UniDepth full.
4. Run paired rows at 20/15/10.
5. Compare each result to the paired full reference on common indices.
6. Inspect throughput and difference time series before choosing the next experiment.

### Mechanism predictions

- If per-frame model compute dominates, selected-frame processing time should decrease roughly with selected-frame count and source-time throughput should rise.
- If request setup, decode, serialization, or fixed initialization dominates, selected-frame FPS may rise while complete stage wall time plateaus.
- If DROID keyframe graph and dense filling dominate, DROID wall time may not scale linearly with selected frames; keyframe spacing and pose deltas will change with temporal gaps.
- If UniDepth service batching dominates, reducing frame count may reduce request count but leave per-request latency nearly unchanged.
- If HaWoR/WiLoR/hands.detect remain the bottleneck, reducing UniDepth or DROID time will not bring the complete pipeline to their standalone throughput.
- If differences concentrate near skipped intervals, high motion, or changing visibility, preserve that localization in the comparison rather than collapsing it to one global number.

The next iteration must be selected from these observations. Do not tune a sampling rate based only on a proxy metric or a single short probe.

## 9. Planned Harness and Artifacts

Suggested implementation units:

```text
scripts/build_fps_schedule.py
scripts/run_unidepth_droid_fps_sweep.py
scripts/compare_unidepth_droid_fps_outputs.py
```

The harness should write:

```text
fps_sweep_manifest.json
fps_sweep_results.csv
stage_timing.jsonl
service_request_trace.jsonl
unidepth_full_vs_sparse_diff.json
droid_full_vs_sparse_diff.json
throughput_parity_report.json
full_vs_20_15_10_side_by_side.mp4
```

Each condition directory contains the immutable schedule, raw request metadata, full service response provenance, timing records, output arrays, comparison JSON, and visual QC frames. Do not use a row count, manifest presence, or a passing schema validator as evidence that the sampled physical state is equivalent.

## 10. Execution Boundary

The plan is offline until the remote A800 execution target and service versions are explicitly recorded. Heavy decoding/model inference/service calls run on the A800 or another authorized non-local target. Local work is limited to harness edits, manifest generation, contract tests, and read-only comparison of copied artifacts.

No experiment is considered complete until it has:

```text
an immutable source/schedule manifest
fresh service/session provenance
timing for every declared boundary
full-frame paired comparison on common indices
coverage and invalid-state accounting
raw per-frame difference series
throughput parity table
visual descriptive QC where available
```

The final report should state the measured throughput frontier and the observed differences for 20/15/10 FPS. It should not call any rate accurate, inaccurate, better, or worse without an external ground-truth study.
