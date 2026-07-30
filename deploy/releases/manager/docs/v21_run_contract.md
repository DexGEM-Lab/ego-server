# V21 Pi Run Contract

This contract defines how to start and govern V21 annotation and benchmark runs. Pi is the harness. Scripts, model runners, dataset loaders, optimizers, renderers, and evaluators are tools only. This contract is intended to be read before any V21 implementation or run.

The authoritative orchestration is `docs/v21_english_orchestration.md`. The design rationale is `docs/pipeline_v21_design.md`. The component map is `docs/v21_component_extraction.md`.

## Launch model

### V21 infer

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v21_infer.md \
  "/v21_infer <input-video> <run-root> [case-id] [optional-side-inputs-json]"
```

### V21 benchmark

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v21_benchmark.md \
  "/v21_benchmark <dataset-name> <dataset-root> <sample-id-or-list> <run-root> [max-iterations] [optional-frame-count]"
```

### V21 parallel manager

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v21_parallel.md \
  "/v21-parallel [data-root=~/data/<EgoScale 30h>] [batch-root=~/data/v21_parallel_runs/<batch-id>] [parallelism=64]"
```

The manager builds a batch manifest and launches runner Pi agents. It is not allowed to make per-entry physical annotation decisions.

### V21 parallel runner

Runners are normally launched by `scripts/launch_v21_parallel_agents.py`, but the direct form is:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v21_parallel_runner.md \
  "/v21_parallel_runner <batch-manifest> <runner-id>"
```

Convenience aliases are defined for the initial local benchmark datasets:

```text
/V21_benchmark:ycb <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
/V21_benchmark:ho3d <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
```

For an isolated model-route smoke check only:

```bash
pi -p \
  --no-tools \
  --no-session \
  --no-context-files \
  --no-skills \
  --no-extensions \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  "Smoke test only: report the active V21 model route and do not perform annotation or benchmark work."
```

The smoke command is not physical progress.

## Modes

### `v21_infer`

Input contract:

- one raw egocentric video path;
- one immutable run root;
- optional case id;
- optional side-input JSON containing calibration, native depth, stereo pair paths, camera trajectory, known-object model paths, or prior artifact hints.

Output contract:

- full-duration `v21_overlay.mp4`, `v21_world.mp4`, and `v21_side_by_side.mp4`;
- render-consumed `state/v21_physical_state.json`;
- uncertainty and evidence state;
- tuning records for weak bottleneck observations;
- logs with compute target and parameter changes.

### `v21_benchmark`

Input contract:

- dataset name, initially `ycb`, `dexycb`, `dex-ycb`, or `ho3d`;
- dataset root matching the fail-fast layout below;
- sample id or sample-list file;
- immutable run root;
- max iterations, default `3`;
- optional benchmark frame count for smoke spans.

Output contract:

- all prediction-side artifacts from `v21_infer`;
- `evaluation/reference_manifest.json` containing GT paths and evaluation-only policy;
- per-iteration render/state/evaluation artifacts;
- `evaluation/benchmark_iterations.jsonl`;
- `evaluation/algorithm_parameter_changes.jsonl`;
- `evaluation/final_selection_report.md`.

GT must be sealed until after prediction-side state and renders exist. Dataset public object/CAD rosters are model libraries only, not annotation targets.

### `v21_parallel_manager`

Input contract:

- EgoScale 30h data root on the default remote server, defaulting to the matching dataset directory under `~/data`;
- one batch root, defaulting to `~/data/v21_parallel_runs/<batch-id>`;
- parallelism `n`, default `64`;
- optional explicit manifest input when a subset is desired.

Output contract:

- `batch_manifest.json` with one row per discovered video/data entry;
- runner launch commands or tmux windows in the canonical `ego_annotation` session;
- queue events in `logs/runner_events.jsonl`;
- no physical annotation completion claim from the manager.

### `v21_parallel_runner`

Input contract:

- one `batch_manifest.json` path;
- one `runner_id`;
- access to `pipeline.md` and `scripts/v21_gpu_wrapper.py`.

Output contract:

- for each claimed entry, the normal `v21_infer` run-root artifacts;
- manifest status update through `scripts/v21_parallel_claim_next.py`;
- completion only after render review, or failure with a concrete missing/failed physical mechanism.

## Compute Targets

Default A800/server target for V21:

```text
SSH command: ssh -p 57938 zjh@115.190.235.210
SSH host: 115.190.235.210
SSH port: 57938
SSH user: zjh
Compute target label: zjh@115.190.235.210
Remote working root: ~
Remote repo root: ~/ego-annotaion-jiahong-dev
Remote checkpoint root: ~/ego-annation-checkpoints
Large output/data root: ~/data
Default parallel dataset: EgoScale 30h under ~/data
Default parallelism: 64
Canonical tmux session: ego_annotation
```

Local machine responsibilities:

- Pi session control.
- Git and task memory.
- Light file inspection and metadata extraction.
- Small state transformations.
- Review of rendered artifacts copied back from remote outputs.

Server/A800 responsibilities after non-mutating probes:

- Hand model inference and MANO optimization.
- Depth/SLAM, metric depth, and multiview candidates.
- Open-vocabulary detection and SAM2 video segmentation.
- Mesh completion and generated shape candidates.
- GPU-accelerated rendering batches.
- Benchmark prediction/evaluation runs when heavy.

Probe before heavy work. Record hostname, GPU memory/utilization, storage capacity, repo presence, environment roots, selected GPU, and output root in `logs/harness_events.jsonl`.

In `/v21-parallel`, GPU submodules must be admitted by `scripts/v21_gpu_wrapper.py` using the estimates in `pipeline.md`. The wrapper owns `CUDA_VISIBLE_DEVICES` for the child process and writes admission events to the run-root GPU wrapper log. Direct GPU launches are allowed only outside parallel mode or after the pipeline marks a step `cpu_direct`.

## Parallel batch directory contract

A V21 parallel batch writes one batch root while preserving per-entry V21 run roots:

```text
v21_parallel_runs/<batch_id>/
  batch_manifest.json
  batch_manifest.json.lock
  logs/
    runner_events.jsonl
  entries/
    <case_id>/                 # normal V21 run root for one claimed entry
      input/
      measurements/
      tuning/
      state/
      renders/
      evaluation/
      logs/
```

The batch manifest is scheduling state only. Per-entry `state/` and `renders/` remain the annotation source of truth.

## Run directory contract

Every V21 run writes one run root:

```text
v21_runs/<run_id>/
  input/
    input_manifest.json
    side_inputs.json                         # optional, infer mode
    dataset_manifest.json                    # benchmark mode
    raw_frame_manifest/manifest.json         # when generated or linked
    frames/                                  # optional extracted frames or links
  measurements/
    monocular_baselines/
    camera_depth/
    hand_candidates/
    object_candidates/
    masks_tracks/
    visible_surfaces/
    geometry_completion/
    pose_fits/
    contact_occlusion_nonpenetration/
  tuning/
    depth_camera/
    segmentation/
    hand_mano/
  state/
    v21_physical_state.json
    v21_uncertainty_state.json
    v21_agent_evidence.md
    v21_observation_bundle.json
    annotations_v21_renderable.json
  renders/
    v21_overlay.mp4
    v21_world.mp4
    v21_side_by_side.mp4
    review_frames/
    review_sheets/
  evaluation/
    reference_manifest.json                  # benchmark mode only
    iteration_<k>/
    gt_metrics.json                          # final/best benchmark result
    gt_alignment.json
    failure_clusters.json
    benchmark_iterations.jsonl
    algorithm_parameter_changes.jsonl
    final_selection_report.md
  logs/
    harness_events.jsonl
```

`state/` is the renderer boundary. Final visible annotations must be driven by `state/`, not private measurement files, logs, metrics, or status labels.

## Minimal initial state

Before measurement tools run, create unresolved state:

```json
{
  "schema": "v21_physical_state.v0",
  "run_id": "<run_id>",
  "mode": "v21_infer|v21_benchmark",
  "input": {
    "video": "<path>",
    "case_id": "<case-id>",
    "side_inputs": []
  },
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
  "depth": {
    "state": "unmeasured",
    "monocular_baseline_required": true
  },
  "hands": [],
  "objects": [],
  "contacts": [],
  "occlusions": [],
  "nonpenetration": [],
  "tuning": {
    "bottleneck_observation_policy": "strong_tune_before_downweight"
  },
  "renderer_boundary": "renders consume state/ only"
}
```

Unresolved state prevents silent success. It is not progress by itself.

## Required physical state families

A V21 state claim must specify physical variable, evidence, tuning status when applicable, uncertainty, and renderer consumption path.

Required families:

- camera intrinsics/extrinsics, camera trajectory, and metric scale provenance;
- depth evidence by modality and scope;
- MANO hand state over time with metric camera/world semantics, shape/scale/betas, visibility, provenance, and uncertainty;
- target object roster selected from visual/task evidence;
- object masks/tracks with contamination review;
- object geometry and pose/posterior for rigid/articulated branches;
- contact/near-contact/non-contact with patch/distance evidence or explicit unresolved state;
- visibility and occlusion ownership;
- nonpenetration/free-space residuals;
- residuals and uncertainty consumed by render/evaluation.

## Strong-tuning gate for bottleneck observations

Depth/camera, segmentation, and hand/MANO observations are bottleneck observations.

If a bottleneck observation is weak, V21 must not immediately downweight it and continue. The harness agent must first run or record a strong-tuning cycle that changes algorithm-internal parameters, input preparation, model branch, prompts, ROI/keyframes, calibration, or fitting objective in a way that targets the failure mechanism.

Every strong-tuning attempt writes:

```text
$RUN_ROOT/tuning/<depth_camera|segmentation|hand_mano>/<target>/attempt_<k>.json
```

It must include:

- sample/run/input hash;
- algorithm backend and version;
- parameter set before and after;
- physical failure mechanism;
- prediction before running;
- artifacts/residuals after running;
- visual review outcome;
- keep/reject/continue decision.

A tuned parameter set is sample-bound. It must not be reused on another sample unless the run records explicit transfer rationale and validation evidence.

Downweighting is allowed only after this gate or a named missing implementation.

## Monocular baseline gate

Whenever V21 uses native depth, RGB-D, stereo, or multiview depth/camera evidence, it must also run a monocular depth/camera baseline on the same frames or representative keyframes.

Any depth-assisted segmentation candidate must be compared to the active OWLv2 bbox-prompt SAM2 track. If the assisted path is worse, the agent must tune calibration, registration, rectification, depth scale, OWLv2 prompts, approved boxes, keyframes, model branch, or component filtering before selecting it as primary evidence.

Uncalibrated stereo may be recorded only as weak relative evidence until calibration/rectification is available or estimated and validated.

## Rigid-object branch contract

Once the agent classifies an object as rigid over a span, this branch is mandatory:

```text
rigid decision
  -> accepted mask track and selected depth/camera evidence
  -> visible surface samples
  -> mesh completion/adaptation or public-CAD/observed-mesh candidate when benchmark supports it
  -> mesh-to-visible evidence alignment
  -> visible-frame SE(3)/Sim(3) pose fitting
  -> temporal factor graph correction with camera/depth/MANO/contact/occlusion/nonpenetration terms
  -> renderer consumes corrected mesh pose
```

Centroids, boxes, spheres, point clouds, visible-surface-only rows, object labels, and generated mesh previews cannot satisfy object pose.

## Benchmark dataset registry

### DexYCB / YCB

Supported aliases:

```text
ycb
dexycb
dex-ycb
```

Observed local root:

```text
/mnt/nas/dex-ycb
```

Representative sample:

```text
20200813-subject-02/20200813_151041/932122062010
```

Fail-fast prediction inputs:

- `<sequence-root>/meta.yml` with `serials`, `num_frames`, `ycb_ids`, `mano_sides`;
- `<sequence-root>/pose.npz` with `pose_m`, `pose_y` recorded as evaluation reference, not prediction state;
- `<camera-root>/color_%06d.jpg`;
- `<camera-root>/aligned_depth_to_color_%06d.png`;
- `<camera-root>/labels_%06d.npz` with `seg`, `pose_y`, `pose_m`, `joint_3d`, `joint_2d` recorded as evaluation reference only.

Prediction path may use RGB, aligned depth, calibration, and public object model paths. GT labels and poses live only under `evaluation/reference_manifest.json`.

### HO3D

Supported alias:

```text
ho3d
```

Observed local root:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D
```

Representative samples:

```text
train/MC1
train/BB10
```

Fail-fast inputs:

- `<dataset-root>/train/<sequence>/rgb/%04d.jpg`;
- `<dataset-root>/train/<sequence>/depth/%04d.png`;
- `<dataset-root>/train/<sequence>/meta/%04d.pkl`.

Required meta keys for selected benchmark frames:

```text
camMat
handPose
handTrans
handBeta
handJoints3D
objRot
objTrans
objCorners3D
objCorners3DRest
objName
objLabel
```

Use `train.txt` when an official annotated subset is requested. Decode HO3D depth according to dataset semantics; do not treat raw PNG intensity as metric depth without decoding.

## Benchmark evaluation semantics

The GT evaluator compares only compatible physical quantities:

- camera-frame hand joints/vertices when dataset and prediction share documented camera semantics;
- object pose after object frame, units, symmetry, and camera/world convention are aligned;
- relative or aligned camera/object trajectories when absolute world frames differ;
- contact, occlusion, visibility, and nonpenetration only when GT or derivable geometry supports them;
- visibility-stratified metrics when labels or reliable evidence exist.

Absolute coordinate equality is not a default claim.

## Evidence-cycle budget

Each run declares budgets before measurement work:

- `max_evidence_cycles`: default `6` for project representatives;
- `max_bottleneck_tuning_attempts_per_family`: default `3` per depth/camera, segmentation, and hand/MANO blocker;
- `max_benchmark_iterations`: default `3`;
- `runtime_budget_multiplier`: default target within the same order of magnitude as input duration for normal infer, excluding explicitly declared benchmark iterations or offline research branches.

If a budget is exhausted, render the best current uncertain state, write unresolved blockers, and stop with scoped claims. Do not create placeholder success.

## Yield gate

Before yielding from V21 implementation or run work, perform an adversarial review:

- user intent and V21 requirements;
- changed artifacts;
- physical deliverables owed;
- completed physical facts;
- unresolved physical variables;
- bottleneck tuning attempts and outcomes;
- monocular baseline comparisons;
- benchmark GT isolation status;
- risks of proxy progress, wrapper regression, or cross-sample parameter leakage.

Apply fixable findings before reporting.
