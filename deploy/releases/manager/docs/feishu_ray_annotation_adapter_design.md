# Feishu Ray Annotation Adapter Design

## Objective

`feat/parallel` owns the video annotation product. The resident Ray services own
only model inference on typed work units. The service-backed path must preserve
the public upload contract:

```bash
curl --noproxy '*' -X POST \
  -F 'file=@/path/to/video.mp4' \
  -F 'request={"model_backend":"feishu_ray","service_profile":"feishu_ray_a800_server_local"}' \
  http://192.168.9.220:8091/v1/annotation-jobs
```

Each adapter converts between one service's model-native contract and the
existing annotation DAG's native artifacts. Calibration, cross-model fusion,
physical-state construction, rendering, packaging, and evaluation remain owned
by the existing pipeline.

The representative acceptance video is the complete 720-frame, 30 fps,
1920x1080, 24 s clip with source SHA-256
`0234472d803d9c7c625ac8478ce34f3b9ed9b7de15c4c1a998df454115f8f525`.
Its server path is:

```text
/home/zjh/data/v22_api_jobs/annotation_shared_droid_mask_20260716_151500/
input/clips/annotation_shared_droid_mask_20260716_151500.mp4
```

## Ownership Boundary

Services own:

- resident model weights and model forward execution;
- request admission and model-native batching;
- typed model outputs and service trace metadata.

Adapters own:

- source decoding and source-to-model image transforms;
- exact pixel, intrinsic, timestamp, handedness, and frame conventions;
- service request construction and stateful lifecycle orchestration;
- ownership, dtype, shape, finite-value, and physical-invariant validation;
- cross-service joins and uncertainty propagation;
- materialization of the existing NPZ/JSON artifacts.

Adapters must not alter or restart services, execute local substitutes for the
resident models, repair NaNs, invent poses, or silently downgrade a requested
service stage to the script backend.

## Common Adapter Contract

Every logical adapter implements the same four responsibilities:

1. `prepare`: convert pipeline artifacts into typed service work units and
   record all source/model transforms.
2. `invoke`: call the configured route, including session/chunk/media lifecycle.
3. `validate`: verify ownership, response schema, model revision, arrays, and
   stage-specific physical invariants before publishing output.
4. `materialize`: write the exact artifact consumed by the existing next stage,
   with service request IDs, transforms, uncertainty, and hashes.

Uniformity is behavioral. A frame adapter, a stateful SLAM adapter, and a
temporal infiller do not share a false stateless function signature.

## DAG

```text
prepare video
  |-- D3 UniDepth adapter --------------------------> D2 calibration
  |-- D6a hands.detect adapter --> source detections/masks
  |                                  |-- D6b WiLoR adapter --> visible overlay
  |                                  `-- D4 DROID adapter
  |                                        ^
  |                                        `-- D2 calibration
  |-- D9b Cosmos adapter ---------------------------> captions/subtitles
  |
  `-- D5 HaWoR adapter <--- D2 + D4 + D6a
           |-- track chunks
           |-- infiller windows
           `-- metric world MANO artifact
                         |
                  D7 hybrid fusion
                         |
              overlay + metric world render
```

`hands.detect` runs once per source frame. Its output is shared by WiLoR and the
DROID dynamic-mask path. DROID runs one create/push/finalize session per video.
HaWoR never starts another DROID or UniDepth instance.

## D3 UniDepth Adapter

Input:

- ordered raw-frame manifest;
- arbitrary source resolution, normally 1920x1080 RGB.

Service request:

- resize RGB to the fixed 960x540 bucket;
- submit `uint8[540,960,3]` without value normalization;
- record diagonal `source_to_model` and inverse `model_to_source` matrices.

Response adaptation:

- validate positive finite depth, confidence in `[0,1]`, and pinhole `K_px`;
- lift intrinsics with `K_source = model_to_source @ K_model`;
- resample depth/confidence to the source grid required by legacy consumers;
- write `measurements/depth_candidates/unidepth_v2/unidepth_v2_depth.npz` and
  `qc_unidepth_v2.json`.

## D6a Hands Adapter

Service request uses the same 960x540 RGB transform as D3. Returned boxes and
masks are treated as model-grid tensors because the deployed implementation does
not lift them despite its source-coordinate prose.

The adapter:

- transforms box corners to source pixels;
- lifts masks with nearest-neighbor sampling;
- records visible, partially visible, occluded, out-of-frame, or unresolved
  evidence with detector uncertainty;
- writes one full-timeline detector artifact and compressed source-grid masks.

This artifact is the only detector source for D6b and D4.

## D6b WiLoR Adapter

For every accepted source-grid hand box, the adapter runs the established
`ViTDetDataset` crop transform on the original source frame and sends normalized
`float32[3,256,256]` to `wilor.reconstruct`.

The response must contain proper MANO rotations, 778 vertices, 21 joints,
translation, focal, confidence, and uncertainty. The adapter verifies
reprojection in source pixels and writes the existing
`measurements/hand_candidates/wilor_v21/wilor_raw_hands.json`.

## D4 DROID Adapter

The deployed DROID behavior, rather than stale prose, defines the compatibility
transform. The adapter must make that behavior explicit:

- resize source frames to the DROID target-area grid with dimensions divisible
  by 8 and pass model-grid intrinsics in `camera.intrinsics`;
- preserve source `K_px` and the complete source/model transform separately;
- send channel-symmetric grayscale RGB so the deployed extra R/B reversal cannot
  make fnet and trajectory-filler color conventions disagree;
- convert source dynamic masks (`1=dynamic`) to the deployed BA ignore semantics
  (`positive=ignore`) and hash both source and submitted tensors;
- set negative motion-filter and keyframe-removal thresholds so every submitted
  source frame is retained, compensating for the deployment's absent
  `dense_source` bookkeeping without interpolating poses in the client;
- call create once, push every frame in timestamp order, and finalize once.

The adapter accepts only a finite proper-SE(3), inverse-consistent, complete
source timeline with positive finite disparity and valid model-grid intrinsics.
It writes the existing `droid_dense_trajectory.npz`, keyframe reconstruction,
and `droid_shared_geometry.json`. A service response that remains non-finite is
preserved as an explicit failed service artifact and cannot enter D5.

## DROID Metric Scale Fusion

Monocular DROID translation remains up to scale. The adapter estimates one
video scale from service outputs only:

1. join DROID keyframes to UniDepth frames by source timestamp;
2. transform UniDepth depth/confidence and the dynamic mask to the disparity grid;
3. compute static confident samples of `depth_metric / (1 / disparity)`;
4. use a robust median as scale and MAD/coverage as residual/confidence;
5. multiply DROID translations by this scale before HaWoR world lifting.

No local Metric3D inference is allowed in the service-backed path.

## D5 HaWoR And Infiller Adapter

The adapter builds one left and one right timeline from D6a detections. A
16-frame request contains source-image crops, exact crop/source transforms,
per-frame observation/occlusion states, source K, UniDepth scale, and metric
DROID poses/timestamps. Missing rows are padded with the nearest observed crop
only as model input and remain `observed=false`.

The `scripts/hawor_peripheral_bundle.py` module reserves one switch point between
D5b and D7. The default `api_adapter` profile is a pass-through plan and does
not change service calls, crop selection, infiller scheduling, or fusion. The
`legacy_equivalent_reserved` profile records the complete original peripheral
stage order: track identity and bbox interpolation, native 16-frame chunks,
wrist-aware camera/world lift, filling preprocess with lerp/SLERP and
canonicalization, common-anchor interval scheduling, infiller windows, ordered
`pred_valid` updates, and MANO world materialization. It remains planning-only
until every stage is implemented and validated; changing the profile currently
cannot alter the active pipeline.

Chunk results are joined by source timestamp. Duplicate predictions select the
lower-uncertainty row. The adapter sends at most 120-frame, two-hand windows to
`hawor_infiller.fill`, preserving observed/inferred flags and using overlapping
windows only to remove boundary discontinuities.

The current HaWoR service returns reproducible MANO parameters but zero-filled
surface placeholders. The adapter therefore evaluates the existing MANO layer
from returned root orientation, hand pose, betas, and translation on CPU. This
is deterministic geometry materialization, not a second hand-pose model.
Camera-space surfaces are transformed by the metric `T_world_camera` and written
to the existing `hawor_world_hands.npz` and QC schema, including valid,
detected-same-frame, occlusion, uncertainty, and track-support arrays.

## D9b Cosmos3 Adapter

Cosmos uses media multipart parts, not tensor parts. The adapter preserves the
existing gallery/boundary semantic algorithm:

- extract timestamped gallery images;
- call `cosmos3.reason` with bounded groups of images and structured JSON prompts;
- parse and validate returned JSON rows;
- detect semantic changes and refine them with denser boundary image groups;
- materialize `v22_cosmos_semantic_review.v1`.

The representative 25 MB source video exceeds the 16 MiB per-media limit, so
the default path never submits the whole video as one media part. Existing
caption normalization and subtitle rendering remain unchanged.

## Public API And Runtime

`model_backend=script` and `model_backend=feishu_ray` are separate execution
paths. Service profiles or endpoint overrides are rejected for `script`.
`feishu_ray` passes `--model-execution feishu_ray`, the profile, and explicit
overrides into the pipeline. It never invokes local model scripts.

The A800 run uses an isolated runtime workspace containing the committed adapter
code, pipeline scripts, service profile, and required deterministic assets. It
does not contain task memory, evaluator targets, or development instructions.
Prediction artifacts go to a fresh run root. Evaluation begins only after
prediction state and renders are frozen.

## Expected Failure Modes

- wrong source/model grid: request admission failure or shifted overlays;
- wrong DROID mask polarity: low visual evidence, non-finite BA, or trajectory
  discontinuity;
- DROID workaround insufficient: finite checks fail before D5, preserving the
  raw typed response and request hashes;
- missing hand observations: explicit unresolved/occluded rows, never certain
  synthetic MANO state;
- HaWoR zero surface used directly: prohibited; reconstructed MANO surface must
  be nonzero and reproduce returned parameters;
- Cosmos response is prose rather than JSON: typed parse failure and bounded
  retry with a corrective JSON-only prompt, never filename-derived captions.

## Verification

First run a 32-frame interface experiment. Predicted observations are admission
of D3/D6 requests, 32 finite retained DROID poses, one finite HaWoR chunk with a
nonzero reconstructed MANO surface, and parseable Cosmos output. Each failure
identifies one adapter boundary before full-video runtime is spent.

The complete acceptance run uses all 720 frames. Required evidence:

- every native artifact has exactly 720 source rows or an explicit per-side
  visibility state for every row;
- calibration consumes lifted source-grid K;
- DROID trajectory is finite, full-timeline, and visually continuous;
- D5 contains metric world MANO parameters and nonzero 778-vertex surfaces;
- D7 renders use D5/D6 service-backed state;
- semantic rows cover the full 24 s timeline;
- overlay, world, subtitle, and published side-by-side videos all contain 720
  frames and 24 s duration;
- representative rendered frames are visually inspected for hand alignment,
  world-motion continuity, and subtitle timing before backend activation.
