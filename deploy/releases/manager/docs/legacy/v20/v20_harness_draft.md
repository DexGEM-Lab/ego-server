# V20 Harness Draft

## 0. User Requirement Capture

V20 should be created on top of the existing V19 Pi-agent harness. The first step is to copy the current V19 harness files into V20 versions, excluding `Pi Agent Runtime` because that is a running Pi session rather than a repository file.

Copied V20 harness files:

- `docs/v20_run_contract.md` from `docs/v19_run_contract.md`
- `configs/v20_agent_system_prompt.md` from `configs/v19_agent_system_prompt.md`
- `configs/v20_gt_evaluator_system_prompt.md` as a new GT evaluation role prompt
- `docs/v20_component_extraction.md` from `docs/v19_component_extraction.md`
- `docs/v20_english_orchestration.md` from `docs/v19_english_orchestration.md`
- `.pi/prompts/v20_infer.md` from `.pi/prompts/v19-run.md`
- `.pi/prompts/v20_benchmark.md` as a new benchmark entry prompt derived from the same V19 prompt family

## 1. Required V20 Modes

### `v20_infer`

Rename the V19 runtime command concept from `v19_run` / `v19-run` to `v20_infer`.

Purpose:

- accept an arbitrary input video;
- run the same physical annotation flow as V19;
- preserve V19 render-consumed physical state semantics;
- produce full-duration overlay/world/side-by-side annotation videos and V20 state sidecars.

### `v20_benchmark`

Add a new benchmark command:

```text
/v20_benchmark <dataset-name> <dataset-root> <sample-id-or-list> <run-root> [max-iterations]
```

Purpose:

- run the same physical annotation flow as `v20_infer` on public GT datasets such as DexYCB and HO3D;
- load dataset RGB/depth/calibration/GT using dataset-specific hard-coded adapters;
- fail fast if the expected dataset format, files, frame counts, keys, units, or coordinate conventions are missing;
- after producing visual artifacts and annotation state, run an additional harness evaluation agent that reads dataset GT;
- evaluate precision/trustworthiness of annotation state under correct physical semantics;
- feed evaluation results back to the controller agent so algorithm choices and parameters can be adjusted;
- loop a bounded number of iterations;
- deliver every iteration's result plus algorithm/parameter modification records.

## 2. Benchmark Dataset Scope

Initial local dataset roots observed under `/mnt/nas/`:

### DexYCB

Observed root:

```text
/mnt/nas/dex-ycb
```

Representative sample observed locally:

```text
/mnt/nas/dex-ycb/20200813-subject-02/20200813_151041/932122062010
```

Observed structure:

- sequence-level `meta.yml`;
- sequence-level `pose.npz` with `pose_m` and `pose_y`;
- camera-level `color_%06d.jpg`;
- camera-level `aligned_depth_to_color_%06d.png`;
- camera-level `labels_%06d.npz` with `seg`, `pose_y`, `pose_m`, `joint_3d`, and `joint_2d`.

The observed representative camera folder has 72 RGB frames, 72 aligned depth frames, and 72 label files.

### HO3D

Observed root:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D
```

Representative samples observed locally:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D/train/MC1
/mnt/nas/ho3d/HO3D_dataset/HO3D/train/BB10
```

Observed structure:

- `rgb/%04d.jpg`;
- `depth/%04d.png`;
- `meta/%04d.pkl`;
- train-level `train.txt`;
- HO3D README describes the meta keys and OpenGL camera convention.

Observed counts:

- `train/MC1`: 897 RGB/depth/meta files, 814 official annotated frames in `train.txt`;
- `train/BB10`: 1606 RGB/depth/meta files; use `train.txt` to derive the official annotated frame subset.

Observed HO3D meta keys include `camMat`, `handPose`, `handTrans`, `handBeta`, `handJoints3D`, `objRot`, `objTrans`, `objCorners3D`, `objCorners3DRest`, `objName`, and `objLabel`.

## 3. Benchmark Evaluation Semantics

The evaluation agent must compare annotations to GT under the same physical meaning:

- camera-frame hand joints should be compared to camera-frame GT, not to an arbitrary world frame;
- object pose should be compared only after object coordinate frame, symmetry, unit, and camera/world convention are defined;
- world-frame poses often matter through temporal change rather than absolute coordinates, so relative frame-to-frame deltas or aligned trajectories may be the correct metric;
- contact/nonpenetration should be evaluated only if GT or dataset geometry supports the claim;
- visibility/occlusion errors should be stratified where possible.

Evaluation output is evidence for the controller. It does not directly overwrite annotation state.

## 4. Feedback Loop Requirement

The V20 benchmark loop should run for a bounded number of iterations, default 3:

1. run annotation flow;
2. render physical state;
3. run GT evaluation agent;
4. identify failure clusters and physical mechanisms;
5. propose one atomic algorithm/factor/parameter intervention;
6. record the prediction before applying the intervention;
7. rerun affected stages;
8. compare new state and metrics;
9. deliver all iterations and parameter/algorithm change records.

The final deliverable is not just the best render or the best metric. It is:

- all iteration outputs;
- final selected render/state;
- GT metrics and alignment semantics;
- failure cluster reports;
- algorithm/factor/parameter change ledger;
- explanation of why the final iteration was selected.

## 5. Fail-Fast Principle

Dataset loading must be hard-coded and fail fast for V20 benchmark mode. It must not guess at unknown dataset structure or silently skip GT fields. If a loader cannot prove RGB/depth/GT/calibration semantics, it stops before inference with a named missing path/key/semantic field.

## 6. Non-Goals

V20 must not:

- create an outer wrapper around Pi;
- replace physical annotation with benchmark metrics;
- use GT to secretly tune the prediction path unless the run is explicitly labeled as oracle/ablation;
- compare unrelated coordinate-frame quantities as if they had the same physical meaning;
- hide failed iterations;
- claim dataset benchmark closure before adapters and GT semantics are implemented.
