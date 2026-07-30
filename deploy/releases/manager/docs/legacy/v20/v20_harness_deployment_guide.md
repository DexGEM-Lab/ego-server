# V20 Harness Deployment Guide

This guide turns the V20 harness draft into a deployment plan. It assumes V20 remains Pi-native: Pi is the harness, and scripts/adapters are tools.

## 1. Files Created For V20

V20 harness files copied or derived from V19:

```text
configs/v20_agent_system_prompt.md
configs/v20_gt_evaluator_system_prompt.md
docs/v20_run_contract.md
docs/v20_component_extraction.md
docs/v20_english_orchestration.md
.pi/prompts/v20_infer.md
.pi/prompts/v20_benchmark.md
.pi/prompts/V20_benchmark:ycb.md
.pi/prompts/V20_benchmark:ho3d.md
```

Supporting V20 documents:

```text
docs/v20_harness_draft.md
docs/v20_harness_deployment_guide.md
```

Existing V20 observation-enhancement draft, preserved rather than overwritten:

```text
docs/v20_draft.md
```

## 2. Launch Commands

### V20 inference on an arbitrary video

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v20_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v20_infer.md \
  "/v20_infer <input-video> <run-root> [case-id]"
```

### V20 benchmark on a GT dataset sample

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v20_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v20_benchmark.md \
  "/v20_benchmark <dataset-name> <dataset-root> <sample-id-or-list> <run-root> [max-iterations]"
```

User-facing aliases are also available as project prompt templates:

```text
/V20_benchmark:ycb <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
/V20_benchmark:ho3d <dataset-root> <sample-id> <run-root> [max-iterations] [frame-count]
```

The implemented benchmark preparation path behind these aliases is `scripts/prepare_v20_benchmark_dataset.py`. It validates the dataset contract while keeping GT out of prediction manifests, state, renders, and candidate generation. Any public object/CAD roster in the prediction manifest is only a model library; target objects must still be selected by the object plan from visual/task evidence. GT is consumed only later by `scripts/evaluate_v20_benchmark_gt.py` after prediction-side prediction state and renders exist.

Examples based on observed local dataset roots:

```bash
/v20_benchmark dexycb /mnt/nas/dex-ycb 20200813-subject-02/20200813_151041/932122062010 /data2/ego_annotation_outputs/v20_benchmark_dexycb_subject02_151041_cam932 3
```

```bash
/v20_benchmark ho3d /mnt/nas/ho3d/HO3D_dataset/HO3D train/MC1 /data2/ego_annotation_outputs/v20_benchmark_ho3d_mc1 3
```

```bash
/v20_benchmark ho3d /mnt/nas/ho3d/HO3D_dataset/HO3D train/BB10 /data2/ego_annotation_outputs/v20_benchmark_ho3d_bb10 3
```

## 3. Deployment Phases

### Phase A — Harness file verification

Before any run:

1. Confirm all V20 files listed in Section 1 exist.
2. Confirm `.pi/prompts/v20_infer.md` and `.pi/prompts/v20_benchmark.md` reference `configs/v20_agent_system_prompt.md`.
3. Confirm `docs/v20_run_contract.md` references `docs/v20_english_orchestration.md`.
4. Confirm `configs/v20_agent_system_prompt.md` requires reading the V20 contract, component extraction, orchestration, and this deployment guide.
5. Confirm `configs/v20_gt_evaluator_system_prompt.md` exists and is referenced by `docs/v20_run_contract.md`.
6. Confirm no V20 prompt points to `v19-run.md` as the active entry.

### Phase B — Dataset registry verification

V20 benchmark mode initially supports exactly:

```text
ycb    # alias for dexycb
dexycb
dex-ycb
ho3d
```

Any other dataset name must fail before inference with:

```text
v20_benchmark_dataset_contract_failed: unsupported_dataset
```

### Phase C — DexYCB fail-fast loader contract

For `dexycb`, validate before inference:

```text
<dataset-root>/<subject>/<sequence>/meta.yml
<dataset-root>/<subject>/<sequence>/pose.npz
<dataset-root>/<subject>/<sequence>/<camera>/color_%06d.jpg
<dataset-root>/<subject>/<sequence>/<camera>/aligned_depth_to_color_%06d.png
<dataset-root>/<subject>/<sequence>/<camera>/labels_%06d.npz
```

Required keys:

- `meta.yml`: `serials`, `num_frames`, `ycb_ids`, `mano_sides`.
- `pose.npz`: `pose_m`, `pose_y`.
- `labels_%06d.npz`: `seg`, `pose_y`, `pose_m`, `joint_3d`, `joint_2d`.

Required count checks:

- color frame count equals selected frame count;
- depth frame count equals selected frame count;
- label frame count equals selected frame count;
- full-sample count equals `meta.yml:num_frames` unless a selected subspan is explicitly requested.

If any field fails, stop before V20 measurement work.

### Phase D — HO3D fail-fast loader contract

For `ho3d`, validate before inference:

```text
<dataset-root>/train/<sequence>/rgb/%04d.jpg
<dataset-root>/train/<sequence>/depth/%04d.png
<dataset-root>/train/<sequence>/meta/%04d.pkl
```

Required meta keys for benchmark frames:

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

Required semantics:

- decode HO3D depth according to the dataset README/local utilities;
- record OpenGL camera convention from the HO3D README;
- use `train.txt` when an official train subset is requested;
- fail fast on `None` annotations for selected benchmark frames.

### Phase E — Input manifest generation

Each V20 run writes:

```text
$RUN_ROOT/input/input_manifest.json
```

Benchmark mode additionally writes:

```text
$RUN_ROOT/input/dataset_manifest.json
```

The dataset manifest must include:

- dataset name and root;
- sample id;
- selected frame indices;
- RGB paths;
- depth paths;
- GT paths;
- calibration/intrinsics path or inline values;
- units;
- coordinate-frame convention;
- GT fields supported for evaluation;
- unsupported physical variables.

### Phase F — Physical annotation flow

Run the same physical flow for `v20_infer` and `v20_benchmark`:

1. timeline/frame manifest;
2. camera/depth/scale sources;
3. MANO hand candidate streams;
4. agent object roster and point prompts;
5. SAM2 masks/tracks;
6. visible metric surfaces;
7. object branch decision;
8. rigid/articulated/deformable/support/unresolved branch handling;
9. object pose, MANO interval, contact, occlusion, nonpenetration factors;
10. render-consumed state assembly;
11. full-duration or selected-span rendering;
12. visual consumption and mechanism repair.

Benchmark mode may use dataset RGB/depth/calibration as measurements, but GT must not be used to secretly correct predictions before evaluation unless explicitly labeled as oracle/ablation.

### Phase G — GT evaluation agent

After each benchmark iteration, run the GT evaluation role:

Inputs:

```text
$RUN_ROOT/state/v20_physical_state.json
$RUN_ROOT/state/v20_uncertainty_state.json
$RUN_ROOT/state/annotations_v20_renderable.json
$RUN_ROOT/renders/<iteration render files>
$RUN_ROOT/input/dataset_manifest.json
<GT paths from dataset_manifest.json>
```

Outputs:

```text
$RUN_ROOT/evaluation/iteration_<k>/gt_metrics.json
$RUN_ROOT/evaluation/iteration_<k>/gt_alignment.json
$RUN_ROOT/evaluation/iteration_<k>/failure_clusters.json
$RUN_ROOT/evaluation/iteration_<k>/evaluation_agent_report.md
```

Evaluation rules:

- compare hand joints only after coordinate-frame semantics match;
- compare object pose only after object frame, unit, camera/world frame, and symmetry semantics are defined;
- compare world/camera pose as relative motion or aligned trajectories when absolute origins differ;
- label contact/occlusion metrics as unsupported when the dataset lacks compatible GT or derivable geometry;
- record unsupported claims rather than fabricating metrics.

### Phase H — Benchmark controller loop

For each iteration:

1. Read the evaluation agent report.
2. Identify failure clusters and map them to mechanisms: prompt/mask identity, depth scale, camera pose, MANO state, object completion, object pose fit, contact factor, occlusion factor, nonpenetration factor, renderer adapter, or coordinate/evaluation mismatch.
3. Propose one atomic intervention.
4. Record the predicted effect and falsifier.
5. Apply the intervention only if it affects a physical mechanism or supported metric.
6. Rerun affected measurement/optimization/render stages.
7. Append algorithm and parameter changes to:

```text
$RUN_ROOT/evaluation/algorithm_parameter_changes.jsonl
```

8. Append iteration summary to:

```text
$RUN_ROOT/evaluation/benchmark_iterations.jsonl
```

Stop when max iterations are exhausted, no supported intervention remains, or the next intervention would require a missing implementation.

## 4. Deliverables

### `v20_infer`

```text
$RUN_ROOT/input/input_manifest.json
$RUN_ROOT/state/v20_physical_state.json
$RUN_ROOT/state/v20_uncertainty_state.json
$RUN_ROOT/state/v20_agent_evidence.md
$RUN_ROOT/renders/v20_overlay.mp4
$RUN_ROOT/renders/v20_world.mp4
$RUN_ROOT/renders/v20_side_by_side.mp4
$RUN_ROOT/logs/harness_events.jsonl
```

### `v20_benchmark`

Everything from `v20_infer`, plus:

```text
$RUN_ROOT/input/dataset_manifest.json
$RUN_ROOT/evaluation/gt_metrics.json
$RUN_ROOT/evaluation/gt_alignment.json
$RUN_ROOT/evaluation/benchmark_iterations.jsonl
$RUN_ROOT/evaluation/algorithm_parameter_changes.jsonl
$RUN_ROOT/evaluation/final_selection_report.md
$RUN_ROOT/evaluation/iteration_<k>/
```

The final benchmark report must name:

- selected dataset and sample IDs;
- GT physical variables actually evaluated;
- unsupported physical claims;
- coordinate-frame and alignment semantics;
- best iteration and why;
- every algorithm/factor/parameter change;
- visible artifact sanity outcome.

## 5. Current Missing Implementations

The V20 docs define the harness and fail-fast contracts. The repository now implements GT-isolated DexYCB/YCB and HO3D benchmark preparation/evaluation plus V20 sidecar tools for depth selection, geometry validation, hand-shape posterior, contact render rows, observation bundle, and state assembly. The following remain implementation tasks:

- standalone raw-frame manifest command;
- fresh full hand/object/camera base annotation builder;
- native V20 renderer for arbitrary inferred states; current path still adapts V18/V19-compatible renderable annotations;
- remote/server execution wiring that produces the required learned hand/object/mask/depth and conditioned geometry outputs for `/v20_infer`;
- model-driven controller loop around `configs/v20_gt_evaluator_system_prompt.md` for iterative non-oracle refinement;
- conditioned geometry generation backend output on server/A800.

A V20 run that needs one of these must stop with a named blocker unless the current task explicitly implements it.

## 6. Acceptance Criteria For The Harness Itself

The V20 harness setup is acceptable when:

- `v20_infer`, `v20_benchmark`, `/V20_benchmark:ycb`, and `/V20_benchmark:ho3d` prompts exist and point to the V20 system/run contract;
- V20 contract and orchestration define both modes;
- local DexYCB and HO3D representative samples are recorded with observed paths;
- dataset fail-fast rules are explicit;
- GT evaluation semantics distinguish absolute pose from relative/aligned physical quantities;
- iteration outputs and algorithm/parameter change records are required deliverables;
- missing adapter implementations are named rather than hidden.
