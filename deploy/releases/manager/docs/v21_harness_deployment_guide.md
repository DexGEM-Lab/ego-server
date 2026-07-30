# V21 Harness Deployment Guide

This guide lists the files and launch procedures for the V21 harness package. V21 remains Pi-native: Pi is the harness, and repository scripts are tools.

## 1. V21 files

Core V21 files:

```text
docs/pipeline_v21_design.md
docs/v21_run_contract.md
docs/v21_component_extraction.md
docs/v21_english_orchestration.md
docs/v21_harness_deployment_guide.md
configs/v21_agent_system_prompt.md
configs/v21_gt_evaluator_system_prompt.md
configs/v21_parallel_runtime_profile.json
.pi/prompts/v21_infer.md
.pi/prompts/v21_benchmark.md
.pi/prompts/v21_parallel.md
.pi/prompts/v21_parallel_runner.md
.pi/prompts/V21_benchmark:ycb.md
.pi/prompts/V21_benchmark:ho3d.md
pipeline.md
```

V21 should preserve V19 context and legacy V20 context:

```text
docs/v19_run_contract.md
docs/v19_english_orchestration.md
docs/pipeline_v19_design_proposal.md
docs/legacy/v20/v20_run_contract.md
docs/legacy/v20/v20_component_extraction.md
docs/legacy/v20/v20_english_orchestration.md
docs/legacy/v20/pipeline_v20.md
```

## 2. Launch commands

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
  "/v21_benchmark <dataset-name> <dataset-root> <sample-id-or-list> <run-root> [max-iterations] [frame-count]"
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

The manager should run on the default remote target `ssh -p 57938 zjh@115.190.235.210`, resolve the EgoScale 30h dataset under `~/data`, build the manifest with `scripts/build_v21_parallel_manifest.py`, and launch runner agents with `scripts/launch_v21_parallel_agents.py`.

### V21 parallel runner

Direct runner launch, normally emitted by the manager:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v21_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v21_parallel_runner.md \
  "/v21_parallel_runner <batch-manifest> <runner-id>"
```

Aliases:

```text
/V21_benchmark:ycb <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
/V21_benchmark:ho3d <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
```

## 3. Harness file verification

Before any V21 work, verify:

1. all files in Section 1 exist;
2. `.pi/prompts/v21_infer.md`, `.pi/prompts/v21_benchmark.md`, `.pi/prompts/v21_parallel.md`, and `.pi/prompts/v21_parallel_runner.md` reference `configs/v21_agent_system_prompt.md`;
3. `configs/v21_agent_system_prompt.md` requires reading the V21 design, run contract, component map, orchestration, deployment guide, and in parallel mode `pipeline.md`;
4. `docs/v21_run_contract.md` names `docs/v21_english_orchestration.md` as authoritative orchestration;
5. V21 prompts do not point to V19/V20 prompts as active entry points;
6. V21 docs name V20 tools as reusable/adaptable only where appropriate;
7. `pipeline.md` marks every parallel submodule as `gpu_wrapper`, `cpu_direct`, or `agent_judgment`;
8. missing V21 implementations are named explicitly.

## 4. Initial supported datasets

V21 benchmark initially supports the same local benchmark contracts as V20:

```text
ycb
dexycb
dex-ycb
ho3d
```

Any other dataset must stop before inference with:

```text
v21_benchmark_dataset_contract_failed: unsupported_dataset
```

### DexYCB/YCB observed local contract

Observed root:

```text
/mnt/nas/dex-ycb
```

Representative sample:

```text
20200813-subject-02/20200813_151041/932122062010
```

Required structure:

```text
<dataset-root>/<subject>/<sequence>/meta.yml
<dataset-root>/<subject>/<sequence>/pose.npz
<dataset-root>/<subject>/<sequence>/<camera>/color_%06d.jpg
<dataset-root>/<subject>/<sequence>/<camera>/aligned_depth_to_color_%06d.png
<dataset-root>/<subject>/<sequence>/<camera>/labels_%06d.npz
```

Required keys:

- `meta.yml`: `serials`, `num_frames`, `ycb_ids`, `mano_sides`.
- `pose.npz`: `pose_m`, `pose_y` as evaluation references.
- `labels_%06d.npz`: `seg`, `pose_y`, `pose_m`, `joint_3d`, `joint_2d` as evaluation references.

### HO3D observed local contract

Observed root:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D
```

Representative samples:

```text
train/MC1
train/BB10
```

Required structure:

```text
<dataset-root>/train/<sequence>/rgb/%04d.jpg
<dataset-root>/train/<sequence>/depth/%04d.png
<dataset-root>/train/<sequence>/meta/%04d.pkl
```

Required meta keys:

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

Use `train.txt` for official annotated subsets when requested.

## 5. Deployment phases

### Phase A — Read and initialize

- Load V21 system prompt.
- Read V21 docs.
- Inspect git status and task memory.
- Verify input/dataset.
- Create unresolved V21 state.
- Declare budgets and the default A800 target `ssh -p 57938 zjh@115.190.235.210` unless an alternate authorized compute target was explicitly declared.

### Phase A-parallel — Build batch and launch runners

- Read `configs/v21_parallel_runtime_profile.json` and `pipeline.md`.
- Resolve EgoScale 30h under remote `~/data` unless the user supplies an explicit data root.
- Build `batch_manifest.json` with `scripts/build_v21_parallel_manifest.py --parallelism 64` by default.
- Launch runner agents with `scripts/launch_v21_parallel_agents.py --parallelism 64` in tmux session `ego_annotation`.
- Runners claim entries with `scripts/v21_parallel_claim_next.py` and run one entry at a time.
- Runners must wrap every `gpu_wrapper` step from `pipeline.md` with `scripts/v21_gpu_wrapper.py`.

### Phase B — Establish timeline and modality

- Create or import raw-frame manifest.
- Write `input/input_manifest.json`.
- Write `measurements/camera_depth/depth_modality_report.json`.
- Record native depth, RGB-D, stereo, multiview, calibration, or rgb-only status.

### Phase C — Run monocular baseline

- Run a monocular/RGB-only depth/camera baseline on the same frames or representative keyframes.
- Register it under `measurements/monocular_baselines/`.
- If no backend exists, stop with `missing_v21_monocular_depth_baseline_backend` unless the user explicitly requested a non-comparative diagnostic.

### Phase D — Run assisted depth/camera paths and compare

- Run native-depth/RGB-D/stereo/multiview candidates when available.
- Compare them against the monocular baseline.
- If assisted result is worse, tune calibration/registration/rectification/depth scale/model parameters before selecting it.
- Record attempts under `tuning/depth_camera/`.

### Phase E — Hand candidates and strong tuning

- Run/import 2D and MANO candidates.
- Diagnose side/crop/camera/scale/depth errors.
- Tune before downweighting.
- Record attempts under `tuning/hand_mano/`.

### Phase F — Object plan and segmentation

- Agent writes target object plan from visual/task evidence.
- Public dataset object roster remains only a model library.
- Select OWLv2 detector keyframes from the object plan.
- Run OWLv2 bbox proposals on those keyframes.
- Approve the target OWLv2 boxes and write bbox prompts for SAM2.
- Run SAM2 proper full-video propagation from approved bbox prompts.
- Create contamination review sheets from `sam2_proper_summary.json`.
- Tune keyframes/OWLv2 text/threshold/SAM2 box propagation before accepting weak masks.
- Record attempts under `tuning/segmentation/`.

### Phase G — Geometry and pose

- Lift accepted masks to visible surfaces.
- Decide object branch.
- For rigid objects, run mesh completion/adaptation and pose fitting.
- Validate mesh candidates against visible depth, silhouette, free space, scale, temporal stability.
- Run the strongest applicable factor graph.

### Phase H — Contact, occlusion, nonpenetration

- Run only when metric MANO and object mesh pose support these variables.
- Otherwise write explicit unresolved state.

### Phase I — State, render, review

- Assemble V21 physical state and renderable annotations.
- Render full-duration or selected benchmark span.
- Agent visually reviews overlay/world/side-by-side and records whether physical annotation improved or which mechanism failed.

### Phase J — Benchmark loop

- After prediction state/render, run GT evaluation.
- Convert failure clusters into one atomic next intervention.
- Record parameter/model/branch changes with sample binding.
- Rerun affected stages.
- Stop at plateau, budget exhaustion, missing implementation, or overfitting risk.

## 6. Deliverables

### `v21_infer`

```text
$RUN_ROOT/input/input_manifest.json
$RUN_ROOT/measurements/camera_depth/depth_modality_report.json
$RUN_ROOT/measurements/monocular_baselines/
$RUN_ROOT/tuning/
$RUN_ROOT/state/v21_physical_state.json
$RUN_ROOT/state/v21_uncertainty_state.json
$RUN_ROOT/state/v21_agent_evidence.md
$RUN_ROOT/renders/v21_overlay.mp4
$RUN_ROOT/renders/v21_world.mp4
$RUN_ROOT/renders/v21_side_by_side.mp4
$RUN_ROOT/logs/harness_events.jsonl
```

### `v21_benchmark`

Everything from `v21_infer`, plus:

```text
$RUN_ROOT/input/dataset_manifest.json
$RUN_ROOT/evaluation/reference_manifest.json
$RUN_ROOT/evaluation/iteration_<k>/gt_metrics.json
$RUN_ROOT/evaluation/iteration_<k>/gt_alignment.json
$RUN_ROOT/evaluation/iteration_<k>/failure_clusters.json
$RUN_ROOT/evaluation/benchmark_iterations.jsonl
$RUN_ROOT/evaluation/algorithm_parameter_changes.jsonl
$RUN_ROOT/evaluation/final_selection_report.md
```

## 7. Acceptance criteria for the harness package

The V21 harness package is internally consistent when:

- V21 launch prompts exist and point to the V21 system prompt;
- V21 run contract defines both modes and run layout;
- V21 orchestration names existing reusable scripts or explicit missing implementations;
- bottleneck strong-tuning gate is required before downweighting;
- monocular baseline comparison is required for native depth/RGB-D/stereo/depth-assisted segmentation;
- sample-bound parameter records are required;
- rigid object branch cannot close with visible-surface-only geometry;
- benchmark GT isolation and iteration records are required;
- final claims depend on render-consumed state and visual review.

This consistency check is support-only. It does not mean V21 physical annotation is runnable end-to-end. The V21 benchmark bootstrap writes policy/state placeholders that are not consumed by a V21 physical mechanism until the downstream measurement, optimization, and render adapters are implemented.
