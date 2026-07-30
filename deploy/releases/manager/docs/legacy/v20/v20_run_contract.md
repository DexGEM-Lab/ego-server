# V20 Pi Run Contract

This contract defines how to start and govern V20 annotation and benchmark runs. It is not a wrapper design. Pi is the harness; scripts and dataset adapters are callable tools. The concrete orchestration over existing components is `docs/v20_english_orchestration.md`; the runtime prompt must follow that runbook rather than fake numbered pipeline scripts.

V20 is derived from the V19 harness and remains monotonic with V19 physical-state semantics. It adds explicit modes and a benchmark feedback loop; it does not replace render-consumed physical state with metrics, reports, or JSON containers.

## Launch model

### V20 infer

Use Pi directly with the V20 system prompt:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v20_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v20_infer.md \
  "/v20_infer <input-video> <run-root> [case-id]"
```

### V20 benchmark

Use Pi directly with the same V20 system prompt and the benchmark prompt:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v20_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v20_benchmark.md \
  "/v20_benchmark <dataset-name> <dataset-root> <sample-id-or-list> <run-root> [max-iterations]"
```

Project-local prompt aliases also exist for the user-facing forms:

```text
/V20_benchmark:ycb <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
/V20_benchmark:ho3d <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
```

These aliases are backed by `.pi/prompts/V20_benchmark:ycb.md` and `.pi/prompts/V20_benchmark:ho3d.md`.

For an isolated model-route smoke check only, do not give tools, context, skills, extensions, or a persistent session:

```bash
pi -p \
  --no-tools \
  --no-session \
  --no-context-files \
  --no-skills \
  --no-extensions \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v20_agent_system_prompt.md)" \
  "Smoke test only: report the active V20 model route and do not perform annotation or benchmark work."
```

The smoke command checks prompt/model routing. It is not a V20 annotation run and must not be reported as physical progress.

## Modes

### `v20_infer`

Input contract:

- one raw egocentric video path;
- run root;
- optional case id;
- optional calibration, depth, camera, known-object, or prior artifact hints recorded in `input/input_manifest.json`.

The infer mode produces full-duration physical annotation and uncertainty artifacts. If no reliable calibration/depth/camera evidence exists, metric claims must be weaker and uncertainty must widen.

### `v20_benchmark`

Input contract:

- dataset name: initially `ycb`/`dexycb`/`dex-ycb` or `ho3d` only;
- dataset root matching one of the fail-fast layouts below;
- sample id or a file containing sample ids;
- run root;
- optional max iterations, default `3`.

The benchmark mode runs the same physical annotation flow as `v20_infer`, but obtains RGB/depth/calibration through dataset-specific loaders and keeps GT sealed until evaluation. GT paths are written only to `evaluation/reference_manifest.json`; they must not enter prediction manifests, candidates, observation bundles, state, renders, or algorithm choices. After each completed prediction-side prediction iteration, `scripts/evaluate_v20_benchmark_gt.py` compares the prediction against GT under documented coordinate semantics.

## Local benchmark dataset registry

These entries are based on paths observed under `/mnt/nas/` on 2026-06-24. A V20 benchmark run must validate these contracts before inference. If a field is missing, stop with `v20_benchmark_dataset_contract_failed` and name the missing path or key.

### `dexycb`

Observed root:

```text
/mnt/nas/dex-ycb
```

Observed representative sample:

```text
dataset-name: dexycb
dataset-root: /mnt/nas/dex-ycb
sample-id: 20200813-subject-02/20200813_151041/932122062010
sequence-root: /mnt/nas/dex-ycb/20200813-subject-02/20200813_151041
camera-root: /mnt/nas/dex-ycb/20200813-subject-02/20200813_151041/932122062010
```

Observed fail-fast layout:

- `<sequence-root>/meta.yml` exists and contains `serials`, `num_frames`, `ycb_ids`, `mano_sides`, and calibration/grasp metadata.
- `<sequence-root>/pose.npz` exists and contains `pose_m` and `pose_y`.
- `<camera-root>/color_%06d.jpg` exists for all required frames.
- `<camera-root>/aligned_depth_to_color_%06d.png` exists for all required frames.
- `<camera-root>/labels_%06d.npz` exists for all required frames and contains at least `seg`, `pose_y`, `pose_m`, `joint_3d`, and `joint_2d`.
- The first observed local sample has 72 color frames, 72 aligned-depth frames, and 72 label files.

Loader semantics:

- Treat `color_*.jpg` as RGB.
- Treat `aligned_depth_to_color_*.png` as depth already aligned to color, but preserve dataset unit/provenance in the manifest until validated.
- Treat `labels_*.npz` and `pose.npz` as GT measurements with dataset camera/object semantics, not as unquestioned pipeline state.
- Fail fast if frame counts across color/depth/labels disagree with `meta.yml:num_frames`.

### `ho3d`

Observed root:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D
```

Observed representative samples:

```text
dataset-name: ho3d
dataset-root: /mnt/nas/ho3d/HO3D_dataset/HO3D
sample-id: train/MC1
sample-id: train/BB10
```

Observed fail-fast layout for `train/<sequence>`:

- `<dataset-root>/train/<sequence>/rgb/%04d.jpg` exists for required frames.
- `<dataset-root>/train/<sequence>/depth/%04d.png` exists for required frames.
- `<dataset-root>/train/<sequence>/meta/%04d.pkl` exists for required frames.
- `meta/%04d.pkl` is a dictionary containing at least `camMat`, `handPose`, `handTrans`, `handBeta`, `handJoints3D`, `objRot`, `objTrans`, `objCorners3D`, `objCorners3DRest`, `objName`, and `objLabel` when the frame is annotated.
- The local `MC1` folder has 897 RGB/depth/meta files, but the official `train.txt` annotated subset contains 814 `MC1` frames; benchmark mode must select the official annotated subset unless an explicit subspan is requested. The local `BB10` folder has 1606 RGB/depth/meta files; use `train.txt` for its annotated subset.
- Use `train.txt` to avoid unannotated frames when a full training/evaluation subset is requested.

Loader semantics:

- Treat `rgb/*.jpg` as RGB.
- Decode HO3D 16-bit depth according to the dataset README and local utilities; do not assume PNG intensity is metric depth without decoding.
- Treat `meta/*.pkl` as GT hand/object/camera annotations in the dataset's OpenGL camera convention; record coordinate semantics before evaluation.
- Fail fast if any required GT key is absent or `None` for a requested benchmark frame.

## Implemented benchmark and V20 algorithm tools

The repository includes a GT-isolated benchmark preparer:

```text
scripts/prepare_v20_benchmark_dataset.py
```

It supports `--dataset ycb|dexycb|dex-ycb|ho3d`, fail-fast validates the local DexYCB/HO3D contracts, writes RGB/depth/calibration/object-model prediction inputs to `input/dataset_manifest.json`, and writes GT references only to `evaluation/reference_manifest.json` with evaluation-only policy.

After prediction-side prediction state/renders exist, evaluate with:

```text
scripts/evaluate_v20_benchmark_gt.py
```

It rejects prediction states containing GT/oracle markers before reading GT, then writes `gt_metrics.json`, `gt_alignment.json`, `failure_clusters.json`, and `evaluation_agent_report.md`.

V20 algorithm-side harness tools now include:

```text
scripts/build_v20_infer_point_prompts_from_object_plan.py
scripts/filter_v20_sam2_masks_by_prompt_components.py
scripts/build_v20_infer_base_annotations.py
scripts/build_v20_depth_candidate_registry.py
scripts/select_v20_depth_observation_bundle.py
scripts/build_v20_uncalibrated_stereo_disparity.py
scripts/build_v20_geometry_candidate_registry.py
scripts/validate_v20_geometry_candidates.py
scripts/adapt_hawor_camspace_to_v20_mano_npz.py
scripts/solve_v20_hand_shape_track.py
scripts/align_v20_hawor_overlay_to_detector_boxes.py
scripts/build_v20_contact_point_render_rows.py
scripts/build_v20_observation_bundle.py
scripts/solve_v20_infer_temporal_observation_graph.py
scripts/render_v20_benchmark_annotations.py
scripts/assemble_v20_state_from_v19_annotations.py
```

These tools consume prediction-side measurement/model outputs, compute residuals or posteriors, and write sidecars consumed by optimization/rendering. Missing upstream model outputs are contract failures, not placeholder success.

## Benchmark evaluation semantics

The GT evaluation agent compares supported physical quantities under the same physical meaning:

- hand joints/vertices: compare in camera frame or the dataset-provided hand frame after documented convention conversion;
- object pose: compare SE(3) or Sim(3) only after confirming object frame, camera frame, units, and symmetry treatment;
- world/camera trajectory: compare relative frame-to-frame motion or aligned trajectory when absolute world origins differ;
- contact/nonpenetration: compare only when the dataset provides compatible surfaces, distances, contacts, or enough geometry to derive them;
- visibility/occlusion: stratify errors by visible, partially visible, occluded, out-of-frame, and unresolved when labels or reliable evidence exist.

Absolute coordinate equality is not a default benchmark claim. If a coordinate frame differs, use a justified alignment and report what the alignment preserves and what it invalidates.

## Run directory contract

Every V20 run writes one immutable run root:

```text
v20_runs/<run_id>/
  input/
    input_manifest.json
    dataset_manifest.json          # benchmark mode only
    frames/                        # optional extracted frames or links
  measurements/
    hand_candidates/
    object_candidates/
    depth_slam/
    depth_candidates/
    masks_tracks/
    geometry_completion/
    pose_fits/
    hand_shape/
    contact_visualization/
  state/
    v20_physical_state.json
    v20_uncertainty_state.json
    v20_agent_evidence.md
    v20_observation_bundle.json
    annotations_v20_renderable.json
  renders/
    v20_overlay.mp4
    v20_world.mp4
    v20_side_by_side.mp4
    review_frames/
  evaluation/
    gt_metrics.json                # benchmark mode only
    gt_alignment.json              # benchmark mode only
    benchmark_iterations.jsonl     # benchmark mode only
    algorithm_parameter_changes.jsonl
    benchmark_export/
  logs/
    harness_events.jsonl
```

`state/` is the renderer boundary. Final visible annotations must be driven by `state/`, not by private measurement files, logs, metrics, or status labels.

## Minimal initial state

Before measurement tools run, Pi should create unresolved physical state rather than empty success containers:

```json
{
  "schema": "v20_physical_state.v0",
  "run_id": "<run_id>",
  "mode": "v20_infer | v20_benchmark",
  "input_video": "<path-or-dataset-derived-video>",
  "dataset": null,
  "timeline": {
    "frame_count": null,
    "fps": null,
    "duration_s": null,
    "resolution": null
  },
  "camera": {
    "state": "unmeasured",
    "required_for_metric_claims": true
  },
  "hands": [],
  "objects": [],
  "contacts": [],
  "occlusions": [],
  "nonpenetration": [],
  "benchmark": {
    "enabled": false,
    "evaluation_reference_loaded": false,
    "iteration_index": 0
  },
  "renderer_boundary": "renders consume this state directory only"
}
```

Unresolved state is not progress by itself. It is the starting point that prevents silent success when physical variables have not been measured.

## Physical state requirements

The current executable command truth is recorded in `docs/v20_english_orchestration.md`. If that runbook marks a required component as missing, a V20 run must stop with the named missing implementation and blocked physical variable rather than authoring placeholder outputs during annotation.

A V20 state claim must specify the physical variable, evidence, uncertainty, renderer consumption path, and benchmark comparison semantics when applicable.

Required variable families:

- MANO hand state over time, with metric camera/world transforms;
- camera/head pose and metric scale provenance;
- target object roster selected from visual/task evidence, geometry state, and pose/posterior where branch-selected; dataset/public model rosters are candidate libraries only;
- visibility and occlusion ownership;
- contact/near-contact/non-contact with patch and distance evidence;
- nonpenetration residuals/uncertainty;
- residuals and uncertainty consumed by visualization and evaluation;
- benchmark GT alignment, metric scope, and iteration feedback when `v20_benchmark` is active.

## Rigid-object branch contract

Once the agent classifies an object as rigid over a span, the following branch is mandatory:

```text
rigid decision
  -> TRELLIS or equivalent per-instance mesh completion
  -> alignment/adaptation to observed depth/mask evidence
  -> visible-frame SE(3)/Sim(3) pose fitting
  -> temporal factor-graph correction with camera/depth/MANO/contact/occlusion/nonpenetration terms
  -> renderer consumes corrected rigid mesh pose
```

Visible surfaces and V20 sidecars are metric measurements and anchors. They cannot replace rigid pose after rigid classification, cannot select target objects from a dataset/public roster, and cannot bypass branch optimization/factor correction before final render. Residuals widen uncertainty or trigger repair; they do not demote the state to a point cloud.

## GT evaluation agent prompt

The benchmark evaluation role is governed by:

```text
configs/v20_gt_evaluator_system_prompt.md
```

This prompt is a role contract for the evaluation agent. It is not an outer wrapper and does not edit annotation state directly. Its reports are fed back to the V20 controller agent as evidence.

## Benchmark loop contract

For `v20_benchmark`, each iteration must produce:

1. physical annotation state and full-duration/selected-span renders;
2. GT evaluation report with frame, unit, and coordinate semantics;
3. failure clusters with physical mechanism interpretation;
4. proposed algorithm/factor/parameter changes;
5. controller decision: apply, reject, or defer each proposed change;
6. next iteration run or final stop state.

The final benchmark deliverable is not a single best metric. It is the full iteration ledger plus final selected state and renders.

## Yield gate

Before yielding from a substantial V20 implementation turn, run a clean-room adversarial review with:

- user intent;
- changed artifacts;
- V20 harness files copied or modified;
- physical deliverables owed;
- completed physical facts;
- benchmark dataset contracts and unresolved loader gaps;
- completed/remaining iteration loop facts;
- parked user decisions;
- constraints;
- risks of proxy progress, wrapper regression, or metric-over-physical overfitting.

Apply the findings before reporting, unless the only remaining issue is a true user decision or high-risk irreversible step.
