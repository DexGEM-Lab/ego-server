# Design amendment: single-item automatic scripted algorithm pipeline

## Decision

The current deliverable is a fully automatic, script-run pipeline for one video. It does not call HTTP APIs. The fixed original service release, `api-ify@3572551`, defines algorithm boundaries, transport-neutral typed inputs and outputs, native batch semantics, and model-environment ownership; it is not the current execution backend.

The pipeline presents one stable algorithm DAG to either backend:

- `ScriptBackend` is the current backend. It invokes the real algorithm scripts/models in their owned environments and converts their native artifacts to typed algorithm results.
- `ApiBackend` is a future backend. It may encode the same requests as multipart HTTP and decode the same results, without changing the DAG, physical-state adapters, or render/package consumers.

A multipart caller is therefore outside the current deliverable. HTTP routes, retries, timeouts, and response parsing belong only inside a future `ApiBackend`.

The outer item batch is fixed to one uploaded video. `item_batch_size=1` fixes dataset cardinality only; it does not change image batches, hand-crop batches, 16-frame HaWoR chunks, 120-step infiller windows, DROID frames/sessions, or Cosmos/vLLM request batching.

## Evidence boundary

Implementation may use only:

1. this design and the current task memory;
2. the fixed original serving source at `api-ify@3572551` for algorithm contracts, batching, adapters, deployments, and environment ownership;
3. the original scripted sources through the allowed committed snapshots (`df45b6e`, `32a01fa`, `326399d`, and the `c772fef` render helpers);
4. the saved Feishu content in `/tmp/ego_parallel_doc.md` and `/tmp/ego_mvp_doc.md`;
5. the upstream Princeton-VL DROID-SLAM RGB-D mechanism named in this amendment.

The implementation must not read or copy the current files in `/mnt/user-home/zjh/ego-annotation-mvp`, `feat/parallel`, existing APIized callers/tests, `call_feishu_ray_service.py`, `run_feishu_ray_*`, or post-`api-ify` service commits.

## Raw-video input contract

Required input:

- one local MP4 path;
- one case ID;
- one fresh run root;
- optional explicit camera intrinsics;
- `item_batch_size`, which must equal `1` for this entrypoint.

Preparation expands the complete video into one immutable source timeline:

- contiguous `frame_idx` from `0` to `N-1`;
- strictly increasing `time_s` derived from source FPS;
- exact RGB frame paths and source SHA-256;
- source width, height, FPS, duration, and color convention;
- bounded source-backed media samples for Cosmos, mapped to source frame indices.

Every final video contains exactly `N` frames and the original source duration.

## Algorithm graph

```text
PHYSICAL LANE

UniDepth -- K[t] ---------------------------------------------------> Robust Canonical-K Aggregation
Robust Canonical-K Aggregation -- K_canonical ----------------------> DROID
UniDepth -- depth_m[t] ---------------------------------------------> DROID
Hand Detect -- dynamic masks when enabled --------------------------> DROID
Hand Detect --------------------------------------------------------> WiLoR

UniDepth -- depth/scale/K[t] evidence ------------------------------+
Robust Canonical-K Aggregation -- K_canonical ----------------------+
Hand Detect -- observations/crops ----------------------------------+--> HaWoR Track Inference
DROID -- poses/timestamps/depth evidence ---------------------------+

HaWoR Track Inference ----------------------------------------------+
UniDepth -- scale/K[t] evidence ------------------------------------+
Robust Canonical-K Aggregation -- K_canonical ----------------------+--> Infiller
DROID -- camera/optimization evidence ------------------------------+

Infiller -----------------------------------------------------------+--> MANO Replay / World Lift
DROID -- T_world_camera --------------------------------------------+

MANO Replay / World Lift -------------------------------------------+
WiLoR -- visible MANO/geometry -------------------------------------+--> Fusion

Fusion -------------------------------------------------------------+--> Physical Hand/World Render
DROID -- camera state ----------------------------------------------+

Fusion -------------------------------------------------------------+
DROID -- camera/depth state ----------------------------------------+
Physical Hand/World Render -----------------------------------------+--> QC

Fusion -------------------------------------------------------------+
DROID -- camera/depth state ----------------------------------------+
QC -----------------------------------------------------------------+--> Evaluator

SEMANTIC LANE

Cosmos --> Semantic Alignment --> Subtitle Render
```

Only algorithms appear as nodes. Source-timeline preparation remains the input adapter described in the raw-video contract: it supplies source-indexed RGB, timestamps, grid transforms, and media without becoming an algorithm node. After the graph completes, a package adapter assembles the physical render, semantic subtitle render, QC, evaluator, numeric state, provenance, and manifests into the downloadable annotation artifact; packaging is likewise outside the algorithm graph.

Wave and dependency semantics:

1. **Wave 1:** UniDepth, Hand Detect, and Cosmos run concurrently after timeline preparation.
2. **Wave 2:** DROID waits for UniDepth `depth_m[t]` and for `K_canonical` from Robust Canonical-K Aggregation; WiLoR waits for Hand Detect. DROID and WiLoR may then run concurrently. If dynamic hand masks are enabled, DROID also waits for Hand Detect.
3. **HaWoR Track Inference:** waits for UniDepth, Robust Canonical-K Aggregation, Hand Detect, and DROID. Its input construction creates side-specific normalized 16-frame crop chunks, crop/source transforms, observation evidence, camera timestamps/poses, and metric depth/K evidence before the native track model runs.
4. **Infiller:** has direct inputs from HaWoR Track Inference, DROID camera/optimization evidence, UniDepth scale/K evidence, and `K_canonical`; it preserves the coupled two-hand 120-step, 218-D checkpoint semantics.
5. **MANO Replay / World Lift:** deterministically replays MANO parameters with real side-specific MANO assets and applies the metric DROID world-from-camera trajectory.
6. **Fusion:** combines the replayed HaWoR temporal/metric state with WiLoR visible geometry and carries uncertainty/visibility forward.
7. **Physical Hand/World Render:** consumes only physical Fusion and DROID camera state, plus the source timeline supplied outside the graph.
8. **QC and Evaluator:** are distinct physical-lane algorithms over the fused hand state, DROID camera/depth state, and physical render. Neither consumes Cosmos or semantic output.
9. **Semantic lane:** Cosmos feeds only Semantic Alignment, which feeds Subtitle Render. Cosmos has no edge to MANO replay, Fusion, Physical Hand/World Render, QC, or Evaluator.

## Transport-neutral algorithm seam

### Common request and result envelopes

`AlgorithmRequest[TInput]` contains:

- `algorithm_id` and model revision;
- case, item, source, frame/track/window identity, and source timestamps;
- typed `input: TInput` whose tensors declare dtype, shape, coordinate frame, units, and pixel transform;
- an explicit native-work description: compatibility key, work-unit type, native batch axis/cap, chunk/window length where applicable;
- backend-neutral options that affect the algorithm, never a URL or server filesystem path.

`AlgorithmResult[TOutput]` contains:

- the matching identity/provenance fields;
- typed `output: TOutput` with the same unit/frame discipline;
- uncertainty, visibility/occlusion state, model revision, and native batch trace;
- reproducible algorithm metadata needed by downstream physical-state adapters.

The DAG depends only on:

```text
AlgorithmBackend.execute(AlgorithmRequest[TInput]) -> AlgorithmResult[TOutput]
```

### Current `ScriptBackend`

For each algorithm, `ScriptBackend`:

1. validates the typed request and its native compatibility bucket;
2. serializes it to a fresh stage workspace owned by the run;
3. launches the algorithm script with the fixed interpreter/environment and model assets owned by that algorithm;
4. preserves the algorithm's native batching rather than looping over singleton model calls;
5. decodes the script output into the typed result;
6. leaves all HTTP concepts absent from the current execution path.

Each algorithm adapter is narrow: typed request → native script input, and native script output → typed result. The adapter does not implement a second version of the model or replace geometry with a proxy.

### Future `ApiBackend`

`ApiBackend` implements the same backend protocol and the same typed request/result pairs. Its internal transport may use the original multipart metadata plus binary tensor parts, but transport details do not enter the DAG or stage contracts. Switching `ScriptBackend` to `ApiBackend` changes only backend construction/configuration.

## Algorithm contracts, environments, and native batching

| Algorithm | Typed work unit | Native execution semantics retained by `ScriptBackend` | Owned environment/model boundary |
|---|---|---|---|
| UniDepth | canonical `uint8 RGB [540,960,3]` image | compatible images stack to one model image batch, cap 8 | UniDepth Python/model environment corresponding to original GPU0 ownership |
| Hand Detect | canonical `uint8 RGB [540,960,3]` image | compatible images batch, cap 8; detector boxes drive SAM2 masks | detector+SAM2+WiLoR environment corresponding to original GPU1 ownership |
| WiLoR | normalized `float32 [3,256,256]` crop plus crop/source transform and side | crops stack as `[B,3,256,256]`, cap 16 | same GPU1 model environment; shared Hand Detect output is reused |
| DROID | one video session with ordered RGB frames, typed `DroidNativeSensorDepthAbiPayload` frames, K, timestamps, optional static mask | one stateful session for this item; ordered frames; native session/window state; cross-session feature batching machinery remains representable although realized session batch is 1 | DROID ABI/environment corresponding to original GPU2 ownership |
| HaWoR tracks | one side-specific normalized `float32 [16,3,256,256]` chunk plus observations/K/camera evidence | chunks stack to `[B,16,3,256,256]`, cap 4 | HaWoR environment corresponding to original GPU3 ownership |
| HaWoR infiller | one coupled two-hand temporal window | 120-step, 218-D checkpoint input; window queue cap 2; checkpoint itself retains its native window execution semantics | HaWoR infiller environment corresponding to original GPU3 ownership |
| Cosmos | bounded source-backed media plus prompt/messages | concurrent requests enter the resident vLLM engine's continuous batch; no synthetic Serve-level tensor batch | Cosmos/vLLM Python 3.13 environment corresponding to original GPU6 ownership |

The logical service names (`unidepth.infer`, `hands.detect`, `wilor.reconstruct`, DROID session lifecycle, `hawor.infer_tracks`, `hawor_infiller.fill`, and `cosmos3.reason`) identify algorithms and typed contracts. They do not imply HTTP in `ScriptBackend`.

## DROID depth and canonical-intrinsics amendment

### Two depth-coupling levels in the evidence

1. **Old scripted D4 had a real but incomplete UniDepth-to-DROID calibration dependency.** It robustly aggregated UniDepth intrinsics before DROID, but then reduced that calibration to a scalar `focal_scale = sqrt(fx*fy)/max(width,height)`. That scalar loses focal anisotropy and both principal-point coordinates, so it is evidence of dependency but is not an acceptable implementation of the amended full-intrinsics contract.
2. **The original API contract advertised more depth coupling than its adapter executed.** In `api-ify@3572551`, `DroidFrameRequest` declares optional `depth_m`, and the saved Feishu table also lists it. The HTTP push handler constructs `DroidFrameRequest` without `depth_m`; the resident adapter never reads `request.depth_m`; and both `DepthVideo.append` calls pass depth as `None`. A present-but-unconsumed field is not algorithmic equivalence.

### Robust Canonical-K Aggregation

Each accepted UniDepth frame supplies `K_depth[t]` on a declared depth-prediction pixel grid and a complete homogeneous transform `P_depth[t]_to_source` that maps depth-grid pixel coordinates to source-grid pixel coordinates. Normalize the transformed matrix after left multiplication:

```text
K_source[t] = normalize_h(P_depth[t]_to_source · K_depth[t])
normalize_h(A) = A / A[2,2]
```

The transform is a pixel-coordinate transform, not a camera pose. Its composition must encode every resize, crop offset, pad offset, and pixel-center convention between grids. A valid source-grid pinhole candidate has the form

```text
K_source[t] = [[fx[t], 0,     cx[t]],
               [0,     fy[t], cy[t]],
               [0,     0,     1   ]]
```

with finite positive `fx[t]` and `fy[t]`. Robust Canonical-K Aggregation estimates `fx`, `fy`, `cx`, and `cy` **separately** in source-pixel units using confidence/frame-quality-weighted robust location (weighted median initialization followed by bounded Huber IRLS), and emits each parameter's retained-frame support, robust scale, and outlier set. The resulting matrix is

```text
K_canonical = [[robust(fx[t]), 0,                robust(cx[t])],
               [0,               robust(fy[t]), robust(cy[t])],
               [0,               0,              1             ]]
```

No square-pixel constraint (`fx = fy`) and no centered-principal-point constraint (`cx = width/2`, `cy = height/2`) is imposed unless an explicit calibrated-camera input supplies and justifies that constraint. A scalar focal statistic, including `sqrt(fx*fy)`, cannot stand in for `K_canonical`.

### Exact source-to-DROID model-grid intrinsics and rays

Let `P_source_to_droid_input` map source pixel coordinates to the ordinary RGB input grid submitted to DROID. Let `P_droid_input_to_model` map that input grid to the projection/BA grid on which `DepthVideo` stores intrinsics. The complete pixel transform and model-grid calibration are

```text
P_source_to_model = P_droid_input_to_model · P_source_to_droid_input
K_droid_input     = normalize_h(P_source_to_droid_input · K_canonical)
K_model           = normalize_h(P_source_to_model · K_canonical)
```

For the common axis-aligned transform

```text
P_source_to_model = [[sx, 0,  tx],
                     [0,  sy, ty],
                     [0,  0,  1 ]],
```

the equality is verifiable component by component:

```text
fx_model = sx·fx_canonical       cx_model = sx·cx_canonical + tx
fy_model = sy·fy_canonical       cy_model = sy·cy_canonical + ty
```

Here `tx` and `ty` include crop/pad offsets and any half-pixel offset required by the selected RGB resampler. The implementation verifies homogeneous pixel-ray consistency on fixture points: for every source pixel `p_source` and ray `r` used by the fixture, `p_model ~ P_source_to_model·p_source ~ K_model·r`.

For the named upstream DROID-SLAM implementation, define

```text
S_8 = P_droid_input_to_model = diag(1/8, 1/8, 1)
```

because `MotionFilter.track` divides the four-vector intrinsics componentwise by 8 before `DepthVideo` storage. Therefore

```text
K_model = normalize_h(S_8 · K_droid_input).
```

Use homogeneous pixel coordinates `q = [x,y,1]^T`, with tensor indexing `[y,x]`. Projection-model cell `(i,j)` has

```text
q_model(i,j) = [i, j, 1]^T
q_input(i,j) = S_8^-1 · q_model(i,j) = [8i, 8j, 1]^T.
```

The corresponding back-projected rays are exactly equal:

```text
K_model^-1 · q_model(i,j) = K_droid_input^-1 · q_input(i,j).
```

This `(8i,8j)` input ray is the only ray whose metric depth may become `disps_sens[j,i]`. The stock `DepthVideo` storage gather, however, reads the incoming depth tensor at

```text
gather(payload)[j,i] = payload[8j+3, 8i+3].
```

On an ordinary RGB-aligned depth raster, that storage location describes input pixel `(8i+3,8j+3)`, not `(8i,8j)`. The `+3` is a stock ABI storage gather offset; it is not a principal-point shift and does not make an ordinary aligned raster ray-equivalent to `K_model`.

### Native sensor-depth ABI payload

The DROID request therefore uses a distinct tensor type, `DroidNativeSensorDepthAbiPayload`, with field name `native_sensor_depth_abi_payload_m`. Its contract is:

- dtype `float32`, units metres, shape `[H_droid_input,W_droid_input]`, tensor index order `[y,x]`;
- semantic tag `stock_depthvideo_gather_slots_v1`, which identifies sparse native ABI packing rather than an image or an RGB-aligned depth raster;
- `H_droid_input = 8·H_model` and `W_droid_input = 8·W_model`; input preparation must crop/pad/resize to these divisible-by-eight dimensions and include that operation in `P_source_to_droid_input`;
- provenance containing the source depth artifact/frame, its declared evidence grid, `P_depth_to_source` when the evidence is not already on the source grid, `P_source_to_droid_input`, metric-depth interpolation/validity policy, confidence source, and the packing ABI version.

Packing is explicit. Initialize every payload element to zero. For every model cell `0 <= i < W_model`, `0 <= j < H_model`:

```text
q_input  = [8i, 8j, 1]^T
q_source = normalize_h(inv(P_source_to_droid_input) · q_input)
q_depth  = q_source                                      # depth evidence already on source grid
         or normalize_h(inv(P_depth_to_source) · q_source) # native depth-evidence grid
d_ray    = valid_metric_sample(D_t, q_depth)
payload[8j+3, 8i+3] = d_ray if valid else 0
```

`valid_metric_sample` is validity-aware bilinear sampling on the exact continuous `q_depth` coordinate. Every source tap with nonzero interpolation weight must lie inside the evidence grid, contain finite positive metric depth, and pass the declared confidence-validity policy; otherwise the sample is invalid. An exact integer coordinate needs only its one nonzero-weight tap. This makes out-of-view coordinates, insufficient boundary support, invalid/low-confidence support, `NaN`, infinity, and nonpositive depth invalid. Those cases write zero; the adapter never clamps to an image edge, reflects, extrapolates, or invents support. Padding created to satisfy divisible-by-eight boundary dimensions therefore remains zero wherever its `(8i,8j)` ray maps outside the source/depth evidence grid. All non-gathered payload locations remain zero, including unused boundary locations. No consumer may visualize, resample, or describe this payload as an aligned depth image.

#### Sealed post-pack representation and identity

Packing is the last geometric or numeric operation permitted on this payload. The packer emits a sealed `DroidNativeSensorDepthAbiPayload` in the canonical representation `logical_dtype=float32`, `wire_dtype=<f4`, `byte_order=little`, `memory_order=C-contiguous`, exact shape `[H_droid_input,W_droid_input]`, tensor index order `[y,x]`, units `metres`, and semantic tag `stock_depthvideo_gather_slots_v1`. Endianness and contiguity canonicalization happen inside packing, before the payload is sealed; a host or tensor runtime that cannot consume the decoded canonical representation without another conversion must fail explicitly.

Let `spec_json` be the RFC 8785 canonical JSON encoding of those fields plus the ABI version, and let `canonical_bytes` be the exact little-endian C-order float32 payload bytes. The packer records both:

```text
payload_sha256          = SHA256(canonical_bytes)
canonical_tensor_digest = SHA256(spec_json || 0x00 || canonical_bytes)
payload_id              = "sha256:" + hex(canonical_tensor_digest)
```

After sealing, the only permitted boundary operation is content-addressed serialization/deserialization of this opaque canonical byte sequence. Serialization may frame or transport `spec_json`, shape, byte length, and `canonical_bytes`; deserialization must verify the declared byte length and hashes and restore exactly the same float32 shape and bytes. An in-process handoff passes the sealed value unchanged. A cross-process decoder may materialize the canonical tensor container required by the DROID worker, but the decoded canonical bytes, spec, shape, `payload_sha256`, and `canonical_tensor_digest` must be identical before the tensor is exposed as the `depth` argument. Serialization/deserialization is representation-preserving transport; it does not authorize a tensor transform.

From pack output through the call `MotionFilter.track(tstamp, image, depth, intrinsics)`, and onward to the unchanged `DepthVideo` stock-gather entrance, the payload is immutable. No component may resize, warp, interpolate, resample, crop, pad, transpose, permute or transform channels, normalize as an image, cast, reinterpret dtype, change endianness or layout, repack, relabel it as ordinary/aligned depth, or reconstruct it from numeric values. This prohibition applies even when an operation appears numerically lossless or produces a matching digest. The typed handoff exposes no generic image/depth preprocessing API; every handoff carries a sealed-operation trace whose only valid sequence is `pack -> pass` or `pack -> serialize -> deserialize -> pass`. Any other event invalidates the seal before `MotionFilter.track`, including a transform followed by reconstruction of the original bytes.

The payload trace has four mandatory checkpoints: `pack_output`, `droid_worker_depth_argument`, `motion_filter_depth_entry`, and `depthvideo_stock_gather_entry`. At every checkpoint, the observer reads the actual tensor presented at that boundary, canonical-serializes it without mutation, and recomputes the full spec, shape, byte length, `payload_sha256`, and `canonical_tensor_digest`; merely copying digest metadata from the request is invalid. Each checkpoint records those observed values plus the same `payload_id` and sealed-operation trace. Equality of all four observed identity records and validity of the operation trace are required; a matching shape alone, matching floating-point values alone, or a later re-created hash is insufficient.

With this packing, unmodified stock `DepthVideo` executes

```text
depth_on_ray[j,i] = payload[8j+3, 8i+3]
disps_sens[j,i]   = 1 / depth_on_ray[j,i]  if depth_on_ray[j,i] is finite and > 0
                    0                        otherwise.
```

Thus `disps_sens[j,i]` is the reciprocal metric depth on input ray `(8i,8j)`, exactly the ray represented by `K_model` cell `(i,j)`, without changing stock gather code or retaining a `+3` ray approximation.

The session trace records `P_source_to_droid_input`, `P_droid_input_to_model`, `K_canonical`, `K_droid_input`, `K_model`, and the native payload type/name/provenance. If the native DROID call accepts a four-vector, the adapter passes `[fx, fy, cx, cy]` extracted from the appropriate full matrix only after verifying that the chosen resize/crop/pad transform preserves the supported zero-skew form. A transform that introduces unsupported skew/rotation must fail explicitly rather than being collapsed to one focal value.

### Required `ScriptBackend` RGB-D mechanism

The new DROID script adapter restores upstream DROID-SLAM's native RGB-D hook:

- sample metric UniDepth evidence on every projection ray `(8i,8j)`, pack it into the canonical little-endian C-contiguous float32 `DroidNativeSensorDepthAbiPayload` at stock gather slot `(8i+3,8j+3)`, leave every unconsumed or invalid slot zero, then seal its spec/shape/bytes under `payload_sha256` and `canonical_tensor_digest`;
- submit `K_droid_input` with all four values `fx`, `fy`, `cx`, and `cy` to the DROID session, and verify that the native componentwise `/8` produces exactly `K_model`;
- if the script and DROID worker cross a process boundary, serialize only the sealed spec plus opaque canonical bytes and require content-addressed deserialization to recover the identical float32 shape/bytes; in-process execution passes the sealed payload unchanged;
- reject any post-pack resize, warp, interpolation, resample, crop, pad, transpose/channel transform, image normalization, dtype cast or reinterpretation, endian/layout conversion, repack, ordinary-depth relabel, or non-allow-listed operation before native tracking;
- record identical spec, shape, byte length, `payload_sha256`, and `canonical_tensor_digest` at pack output, the DROID worker's received `depth` argument, `MotionFilter.track` entry, and the `DepthVideo` stock-gather entry; fail before gather on any mismatch;
- pass the verified `native_sensor_depth_abi_payload_m`—never an ordinary aligned depth raster—as the unchanged `depth` argument to `MotionFilter.track(tstamp, image, depth, intrinsics)`;
- let the unchanged `DepthVideo.append` gather `[3::8,3::8]`, write `disps_sens = 1 / depth_on_ray` on valid support, and store the full model-grid intrinsics;
- let `DroidFrontend._update` use valid `disps_sens` and the stored `[fx_model, fy_model, cx_model, cy_model]` in its projection/update path;
- let `DepthVideo.ba` pass `disps_sens` and the same full model-grid intrinsics into `droid_backends.ba` so metric depth and the calibrated projection both participate in bundle adjustment;
- preserve `DroidBackend`'s distinction: monocular disparity normalization is used only when sensor disparity is absent throughout.

This makes per-frame UniDepth depth influence the optimization itself on the same rays used by projection. A finalize-only depth/disparity scale fit may remain a diagnostic or uncertainty estimate, but it cannot substitute for the native RGB-D coupling.

The DROID result declares one of two scale modes:

- `metric_rgbd_unidepth`: valid UniDepth sensor disparity entered frontend initialization and BA;
- `up_to_scale_monocular`: no valid sensor disparity entered the session and the native monocular normalization path was used.

The result also records depth support fraction, rejected-depth reasons, canonical K, sensor-disparity statistics, and the branch actually executed. It never labels an RGB-only run metric merely because a `depth_m` field existed upstream.

If dynamic hand masking is enabled, the source-grid Hand Detect/SAM2 masks are transformed to the DROID grid and combined with depth validity as confidence support. This adds the Hand Detect → DROID dependency without changing the RGB-D mechanism.

## Full-timeline physical state

For source frame `t`:

- `K_depth[t]`: UniDepth per-frame intrinsics candidate;
- `K_canonical`: robust clip-level K whose `fx`, `fy`, `cx`, and `cy` are aggregated separately before DROID;
- `P_source_to_droid_input`, `P_droid_input_to_model`, and `P_source_to_model`: complete homogeneous pixel transforms;
- `K_droid_input` and `K_model`: full four-parameter DROID input-grid and projection/BA-grid intrinsics;
- `D_t`: UniDepth metric depth and confidence on a declared source/depth evidence grid;
- `D_model_ray[t,j,i]`: metric sample on DROID input projection ray `(8i,8j)`, or zero when invalid/out of view;
- `native_sensor_depth_abi_payload_m[t]`: typed `DroidNativeSensorDepthAbiPayload` whose stock gather slots carry `D_model_ray` and whose other locations are zero; this is ABI packing, not an aligned depth raster;
- `disps_sens[t,j,i]`: zero for invalid support and otherwise exactly `1 / D_model_ray[t,j,i]`;
- `T_world_camera[t]`: DROID world-from-camera pose, metric only under `metric_rgbd_unidepth`;
- `Q_camera[t]`: DROID disparity/depth state and uncertainty;
- `H_t`: shared hand detections, masks, side, score, visibility, occlusion, and uncertainty;
- `W_t`: WiLoR visible MANO/geometry candidate per detector crop;
- `A_t`: observed HaWoR camera-space MANO state;
- `I_t`: infiller observed/inferred two-hand MANO state and uncertainty;
- `M_world[t]`: replayed metric MANO surface/joints in world coordinates;
- `F_t`: fused hand state;
- `S_t`: source-backed Cosmos semantic row when enabled.

All transforms use one documented convention, `T_world_camera`. Image grids always carry the pixel transform that relates them to source pixels.

## Physical reconciliation and outputs

The downstream chain preserves the original scripted MVP mechanisms:

1. separate robust aggregation of UniDepth `fx`, `fy`, `cx`, and `cy`, followed by complete source-to-model pixel transformation before DROID;
2. RGB-D DROID optimization with full model-grid intrinsics, depth validity/confidence, and optional dynamic masks;
3. one camera pose convention across DROID, HaWoR evidence, MANO world lift, and world render;
4. deterministic MANO replay from returned parameters and real side-specific assets;
5. hybrid fusion using HaWoR metric/temporal state plus WiLoR visible geometry;
6. reprojection, depth, temporal, RGB-D BA, and cross-stream residuals with uncertainty propagation;
7. explicit visibility/occlusion states rather than certain filling through missing evidence;
8. no contact, object-pose, or nonpenetration claim without those mechanisms.

The pipeline reproduces the complete original MVP consumer surface:

- raw-frame/input manifests;
- UniDepth depth/K/confidence artifact and canonical calibration;
- shared hand-detection timeline and WiLoR raw candidates;
- DROID dense trajectory, keyframes, disparity/depth state, scale mode, and camera-stage contract;
- HaWoR track and infiller outputs;
- replayed world MANO and hybrid hand archives;
- GT-free residual/drift state;
- source-backed Cosmos semantics when enabled;
- full-length hand overlay, metric world head+hand render, semantic subtitle, and primary side-by-side video;
- self-consistency QC, evaluator diagnostics, product manifest, and downloadable annotation package.

## Acceptance checks

### Backend and dependency checks

- The current run instantiates `ScriptBackend` only and makes zero HTTP/API calls.
- The orchestrator imports no HTTP/multipart client and contains no endpoint URLs.
- Replacing backend construction with a future `ApiBackend` requires no DAG or physical/render consumer change.
- Trace order proves Wave 1 = UniDepth + Hand Detect + Cosmos; Wave 2 = DROID after both UniDepth depth and Robust Canonical-K Aggregation, and WiLoR after Hand Detect.
- HaWoR input creation and Infiller execution each start only after their direct UniDepth, canonical-K, Hand Detect/HaWoR-track, and DROID dependencies exist.
- If dynamic masks are enabled, DROID also starts only after Hand Detect.
- Dataflow tracing proves Cosmos reaches only Semantic Alignment and Subtitle Render; Physical Hand/World Render, QC, and Evaluator have no Cosmos or semantic-result input.

### Canonical-K and DROID RGB-D discriminating tests

- **Four-parameter aggregation fixture:** asymmetric, off-center per-frame candidates with outliers produce independently robust `fx_canonical`, `fy_canonical`, `cx_canonical`, and `cy_canonical`, along with separate support/scale/outlier traces. The fixture fails if `fx_canonical == fy_canonical` is forced, if the principal point is forced to the image center, or if any output is reconstructed from one scalar focal statistic.
- **Complete pixel-transform fixture:** a nonuniform scale plus nonzero crop/pad/half-pixel offsets constructs an expected `P_source_to_model`. Matrix multiplication must satisfy `K_model = normalize_h(P_source_to_model · K_canonical)` and preserve the four expected values `sx·fx`, `sy·fy`, `sx·cx+tx`, and `sy·cy+ty`. Projected fixture rays must agree before and after the transform to numerical tolerance.
- **Model-cell native-gather fixture:** use nonconstant synthetic metric depth and a nonidentity `P_source_to_droid_input`. For every model cell `(i,j)`, independently compute the source/depth evidence coordinate of input ray `(8i,8j)` and its expected metric sample. After ABI packing, assert cell by cell that `native_sensor_depth_abi_payload_m[8j+3,8i+3]` and the actual stock `[3::8,3::8]` gather both equal that expected source-ray sample. Assert every unconsumed payload location is zero.
- **Sensor-disparity reciprocal fixture:** after the real stock gather, each valid cell must satisfy `disps_sens[j,i] == 1 / expected_depth_on_ray(8i,8j)` to numerical tolerance, while every invalid cell remains exactly zero.
- **Model-K back-projection fixture:** for every exercised cell, assert `K_model^-1·[i,j,1]^T ~ K_droid_input^-1·[8i,8j,1]^T`. Couple this assertion to the gathered depth value so the fixture proves the disparity and projection at `(i,j)` refer to the same ray, not merely that both arrays have the same shape.
- **Uncompensated `+3` negative control:** pass a nonconstant ordinary DROID-input-aligned metric depth raster directly to the stock gather, without ABI packing. The strict-ray cell fixture must fail because it reads the depth of `(8i+3,8j+3)` where `(8i,8j)` is expected. A fixture that passes this negative control, or uses constant depth that hides the offset, is invalid.
- **Payload type/provenance fixture:** request validation and serialized trace must retain type `DroidNativeSensorDepthAbiPayload`, field `native_sensor_depth_abi_payload_m`, `float32` metres, `[H_droid_input,W_droid_input]`, semantic tag `stock_depthvideo_gather_slots_v1`, source depth/frame identity, evidence-grid transform, `P_source_to_droid_input`, confidence/validity/interpolation policy, and ABI version. Relabeling an ordinary aligned raster with this type fails.
- **Post-pack identity-trace fixture:** at `pack_output`, `droid_worker_depth_argument`, `motion_filter_depth_entry`, and `depthvideo_stock_gather_entry`, recompute from the actual boundary tensor and record its full spec, shape, byte length, `payload_sha256`, and `canonical_tensor_digest`; copied request metadata does not count. The four observed records must be exactly equal and the sealed-operation trace must contain only `pack -> pass` or `pack -> serialize -> deserialize -> pass`. The worker-received `depth` parameter and both stock entry traces must therefore identify the same sealed payload that packing produced, rather than a shape-compatible reconstruction.
- **Serialization/deserialization fixture:** round-trip the sealed payload across the actual process boundary as canonical little-endian C-contiguous float32. Decode must recover the exact shape and bytes and the same `payload_sha256`/`canonical_tensor_digest`; truncated/corrupt bytes or mismatched shape, dtype, endianness, memory order, semantic tag, ABI version, or digest must fail before `MotionFilter.track`. The same fixture must show that this exact content-addressed round trip is accepted as tensor serialization, without classifying it as a geometric or numeric transform.
- **Post-pack transform negative controls:** independently insert post-pack resize/resample, warp/interpolation, crop, pad, transpose, channel permutation/transform, image normalization, dtype cast/reinterpretation, endian/layout conversion, repack, and ordinary-depth relabeling. Every control must fail before `MotionFilter.track` even if it preserves shape, appears lossless, or later recreates the original digest. These controls distinguish forbidden geometry/numeric/semantic transforms from the one allowed opaque serialization/deserialization path.
- **Boundary/invalid-depth fixture:** exercise divisible-by-eight preparation padding, boundary model cells, coordinates mapped outside the source/depth evidence grid, insufficient interpolation support, low-confidence support, `NaN`, infinity, zero, and negative depth. Each corresponding gather slot and `disps_sens` cell must be zero; no clamp, reflection, extrapolation, or partial boundary cell is allowed.
- **End-to-end intrinsics trace:** the exact asymmetric/off-center `[fx_model, fy_model, cx_model, cy_model]` is evidenced at DROID session creation/push, `MotionFilter.track`, `DepthVideo` storage, `DroidFrontend._update`, and the `DepthVideo.ba -> droid_backends.ba` call. A trace containing only `focal_scale`, `sqrt(fx*fy)`, one repeated focal value, or a reconstructed centered principal point fails even if a K perturbation changes the final trajectory.
- **Independent K interventions:** perturbing each of `fx`, `fy`, `cx`, and `cy` separately changes the corresponding projection coordinates and reaches frontend and BA numerically. Merely recording another K, or changing only a scalar focal proxy, fails.
- **Depth-present case:** valid native sensor-depth ABI payload produces nonzero finite `disps_sens`; frontend and BA receive it; the all-empty sensor-disparity/monocular-normalization branch is not taken; result scale mode is `metric_rgbd_unidepth`.
- **Depth-absent case:** `disps_sens` remains empty/zero; native monocular normalization executes; result scale mode is `up_to_scale_monocular`.
- **Depth-scaling intervention:** multiplying valid source depth evidence by a known factor changes packed gathered depth by that factor, changes sensor disparity inversely, and changes optimized disparity/trajectory scale consistently. An unchanged result fails the coupling test.
- HTTP/service parity is not claimed from the `depth_m` dataclass field alone; the `api-ify@3572551` adapter gap remains explicit until a future `ApiBackend` endpoint truly consumes the field.

### Native batch checks

- `item_batch_size == 1` is recorded only at the outer item layer.
- UniDepth and Hand Detect retain image batches; WiLoR retains crop batches; HaWoR retains `[B,16,3,256,256]`; infiller retains 120-step/218-D two-hand windows; Cosmos retains vLLM continuous batching; DROID retains session/window state and its cross-session batch-capable boundary.
- No algorithm adapter replaces its native batch forward with a per-frame/per-crop singleton loop merely because one video was submitted.

### Timeline and rendered checks

- Every timeline artifact uses the same `N` source frames and timestamps.
- All final videos contain exactly `N` frames and source duration.
- MANO surfaces are finite, nondegenerate, and project near observed hands.
- World and image renders consume the same camera/hand state.
- Uncertainty and visibility change where detector, depth, DROID, HaWoR, or infiller evidence is weak.
- Render comparison preserves every valid capability of the original scripted MVP.

## Representative validation run

Use one existing full EgoScale video under a fresh A800 run root. The run uses the scripted backend in the algorithm-owned environments, does not call or mutate active services, does not reuse APIized prediction artifacts, and does not touch the existing full-dataset batch.

Expected implementation failures are pixel-grid/depth alignment, invalid-depth support, K transformation, accidental monocular normalization in an RGB-D run, pose convention, timeline ownership, shared detector reuse, HaWoR chunk layout, infiller two-hand ordering, MANO replay, and legacy artifact adaptation. Mechanical failures stop publication; noisy measurements continue with uncertainty.
