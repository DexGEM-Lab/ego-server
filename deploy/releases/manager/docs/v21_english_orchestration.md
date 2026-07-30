# V21 English Orchestration Over Real Components

Status: V21 implementation runbook. This document is the authoritative V21 orchestration for runtime agents. It names existing runnable commands where they exist, V21 prompt/contract files where they define harness entry, and explicit missing implementations where physical mechanisms are not yet implemented. It does not claim that V21 physical annotation has run, and it does not authorize fake numbered scripts, registry-only progress, validator loops, or V20 approximate closure as final physical annotation.

## 0. Physical objective and modes

V21 has four modes:

- `v21_infer`: arbitrary input video -> full-duration renderable physical annotation state.
- `v21_benchmark`: supported GT dataset sample/batch -> the same prediction-side physical annotation flow, followed by sealed GT evaluation and iterative controller improvements.
- `v21_parallel_manager`: `/v21-parallel` batch manager -> build a batch manifest, launch runner Pi agents, and report queue/resource state without physical annotation decisions.
- `v21_parallel_runner`: one worker Pi agent -> claim one data entry at a time, run the normal V21 physical annotation flow, review renders, mark completion/failure, and claim the next entry.

The physical objective is to produce visible annotations driven by real physical variables:

- metric camera/depth state;
- metric MANO hands with shape/scale/visibility/uncertainty;
- target object masks/tracks from visual/task evidence;
- completed/adapted object geometry and pose for rigid manipulated objects;
- contact, occlusion, and nonpenetration states or explicit unresolved uncertainty;
- full-duration overlay/world/side-by-side renders consumed from `state/`.

A command is progress only when it measures, tunes, optimizes, renders, evaluates, or falsifies one of these variables. Reports and schemas are evidence, not deliverables.

## 1. Runtime start

1. Inspect `git status --short`; preserve unrelated work.
2. Read `docs/pipeline_v21_design.md`, `docs/v21_run_contract.md`, `docs/v21_component_extraction.md`, this file, and `docs/v21_harness_deployment_guide.md`. In parallel mode, also read `pipeline.md` and `configs/v21_parallel_runtime_profile.json`.
3. Read task memory if a V21 task directory exists.
4. Verify that `configs/v21_agent_system_prompt.md` was loaded. If not, stop and report the launch command from `docs/v21_run_contract.md`.
5. Verify input video or benchmark dataset contract.
6. Choose a run root only after confirming it will not overwrite an existing run.
7. Declare budgets: evidence cycles, benchmark iterations, bottleneck tuning attempts, and runtime target.
8. Probe the default A800 target `ssh -p 57938 zjh@115.190.235.210` before heavy inference or rendering unless an alternate authorized compute target was explicitly declared.
9. In `v21_parallel_runner`, launch every GPU step marked `gpu_wrapper` in `pipeline.md` through `scripts/v21_gpu_wrapper.py` with the listed estimated VRAM. The wrapper owns `CUDA_VISIBLE_DEVICES`.
10. Write unresolved initial state and logs before model runs.

### 1.1 Parallel manager and runner start

`v21_parallel_manager` should run only control-plane work:

```bash
python scripts/build_v21_parallel_manifest.py \
  --data-root "$DATA_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --parallelism 64

python scripts/launch_v21_parallel_agents.py \
  --batch-manifest "$BATCH_ROOT/batch_manifest.json" \
  --parallelism 64
```

If no data root is supplied, resolve the EgoScale 30h dataset under remote `~/data` using `configs/v21_parallel_runtime_profile.json`. The manager does not inspect object identity, contact, occlusion, pose, or render acceptance.

`v21_parallel_runner` claims one entry at a time:

```bash
python scripts/v21_parallel_claim_next.py \
  --manifest "$BATCH_MANIFEST" \
  --runner-id "$RUNNER_ID" \
  --claim
```

For a claimed row, `input_video`, `run_root`, and `case_id` become the normal `v21_infer` contract. After render review, the runner marks the row completed or failed through the same claim tool. A queue status is not annotation progress.

## 2. Input and timeline

### 2.1 `v21_infer`

Verify:

- input video exists and is readable;
- frame count, FPS, resolution, duration;
- optional side-input JSON paths if supplied;
- side-input modalities: calibration, native depth/RGB-D, stereo, multiview, camera trajectory, known object models, or prior measurements.

Current executable helper:

```bash
python scripts/run_v19_v20_infer_bootstrap.py \
  --mode v20 \
  --input-video "$INPUT_VIDEO" \
  --run-root "$RUN_ROOT" \
  --case-id "$CASE_ID"
```

Use this only as a raw-frame/run-root bootstrap if no V21-specific bootstrap exists. Afterward, write V21 `state/` files and V21 logs. If this helper cannot represent the input modality, stop with `missing_v21_raw_frame_manifest_adapter` and the blocked timeline variable.

### 2.2 `v21_benchmark`

Validate dataset before any prediction work. Current V21 adapter:

```bash
python scripts/prepare_v21_benchmark_dataset.py \
  --dataset "$DATASET_NAME" \
  --dataset-root "$DATASET_ROOT" \
  --sample-id "$SAMPLE_ID" \
  --run-root "$RUN_ROOT"
```

This adapter reuses the V20 dataset parser internally but writes V21 manifest schemas, unresolved V21 state, uncertainty state, observation bundle, evidence note, and harness log. If it cannot represent a dataset/sample, stop with `v21_benchmark_dataset_contract_failed` or the specific missing adapter before running inference.

GT paths must live only in `evaluation/reference_manifest.json` and must not enter prediction state, candidates, object plans, masks, depth selection, renders, or tuning decisions before evaluation.

## 3. Modality report and monocular baseline

Before selecting depth/camera evidence, write a modality report:

```text
$RUN_ROOT/measurements/camera_depth/depth_modality_report.json
```

Required fields:

- has_native_depth;
- has_rgbd_stream;
- has_stereo_pair;
- has_calibration;
- has_multiview;
- rgb_only;
- intrinsics_source;
- extrinsics_source;
- depth_unit_source;
- expected failure modes.

If no script exists, write the report directly from inspected manifests and file metadata; this is bookkeeping that enables the next mechanism, not progress by itself.

### 3.1 Monocular baseline requirement

For every input, run or import a monocular/RGB-only depth/camera baseline. Existing candidate tools include:

```bash
python scripts/run_unidepth_full_frame_v3.py \
  --manifest "$RAW_FRAME_MANIFEST" \
  --output-dir "$RUN_ROOT/measurements/monocular_baselines/unidepth_full_frame" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --unidepth-repo "$UNIDEPTH_REPO"
```

Alternative existing tools when available:

```bash
python scripts/run_depthpro_metric_source_v3.py ...
python scripts/run_droid_full_frame.py ...
python scripts/run_vggt_native_camera_v3.py ...
python scripts/run_vggt_scene_geometry_v3.py ...
```

If the chosen monocular backend is unavailable, record `missing_v21_monocular_depth_baseline_backend` and the blocked comparison. Do not select a native-depth/stereo path as superior without a baseline comparison unless the user explicitly narrows the run to a non-comparative diagnostic.

## 4. Depth/camera strong-tuning path

### 4.1 Register candidates

Use existing V20 registry tooling as a starting point:

```bash
python scripts/build_v20_depth_candidate_registry.py \
  --input-video "$INPUT_VIDEO" \
  --candidate "<id>|<path>|<method>|<source_modality>|<initial_status>|<scale_hint>" \
  --output-dir "$RUN_ROOT/measurements/camera_depth/depth_candidates" \
  --require-candidate
```

Register:

- monocular baseline depth/camera;
- native depth/RGB-D candidates;
- calibrated stereo candidates when calibration/rectification exists;
- uncalibrated stereo only as weak relative evidence;
- dataset depth/calibration in benchmark mode as prediction-side measurements;
- DROID/VGGT/multiview candidates when available.

### 4.2 Compare assisted paths against monocular

Required comparison report:

```text
$RUN_ROOT/measurements/camera_depth/monocular_vs_assisted_depth_report.json
```

Compare:

- focal/intrinsics plausibility;
- RGB-depth registration;
- mask depth continuity;
- hand-depth residual;
- object visible-surface residual;
- depth-order/contact-gap residual when available;
- temporal smoothness;
- downstream geometry pose fit residual;
- visual review of scale/alignment failures.

If native depth/RGB-D/stereo/multiview is worse than monocular, do not select it as primary. Enter strong tuning.

### 4.3 Tune before downweighting

Depth/camera tuning examples:

- depth scale/unit correction;
- RGB-depth registration and resize alignment;
- intrinsics/focal estimation;
- stereo rectification;
- disparity search range and confidence thresholds;
- left-right consistency settings;
- monocular input resolution or focal prior;
- VGGT/DROID frame sampling and scale alignment;
- temporal depth smoothing only after metric alignment is understood.

Each attempt writes:

```text
$RUN_ROOT/tuning/depth_camera/<candidate_id>/attempt_<k>.json
```

Predict before running. If no existing script exposes the needed parameter, stop with the missing implementation, e.g. `missing_v21_calibrated_stereo_depth_adapter` or `missing_v21_depth_tuning_parameter_hook`.

Only after tuning can `select_v20_depth_observation_bundle.py` or a V21 selector be used to produce selected depth evidence:

```bash
python scripts/select_v20_depth_observation_bundle.py \
  --registry "$RUN_ROOT/measurements/camera_depth/depth_candidates/depth_candidate_registry.json" \
  --annotations "$BASE_ANNOTATIONS" \
  --output-report "$RUN_ROOT/measurements/camera_depth/depth_selection_report.json" \
  --output-bundle "$RUN_ROOT/state/v21_depth_observation_bundle.json" \
  --require-selected
```

If this selector cannot enforce V21 monocular comparison, write `missing_v21_depth_selector_comparison_gate` rather than claiming V21 selection.

## 5. Hand / MANO measurement and tuning

Run or import hand evidence streams:

```bash
python scripts/run_rtmlib_hand2d_v3.py ...
python scripts/run_wilor_full_frame.py ...
python scripts/export_hawor_world.py ...
python scripts/run_hamer_rtmlib_hand_stream_v3.py ...
python scripts/merge_hand_candidate_streams_v7.py ...
python scripts/refit_mano_metric_depth_v3.py ...
```

Before accepting MANO, diagnose systematic errors:

- detector boxes;
- side mapping;
- crop/resize convention;
- camera convention;
- MANO global transform;
- scale;
- depth alignment;
- temporal offset;
- occlusion visibility state.

If hands are offset or shape/pose is poor, enter strong tuning before downweighting:

```text
$RUN_ROOT/tuning/hand_mano/<hand_track_id>/attempt_<k>.json
```

Hand tuning examples:

- choose WiLoR vs HaWoR vs HaMeR backend for the sample;
- adjust detector boxes or mask-conditioned boxes;
- fix side mapping;
- correct crop resize/intrinsics coupling;
- refit translation/scale to selected metric depth;
- adjust MANO pose/shape priors;
- use hand silhouette/mask residual when available;
- tune temporal smoothness after projection/depth semantics are correct.

`scripts/solve_v20_hand_shape_track.py` can provide a posterior prior from candidates, but V21 accepted hand shape requires active betas/scale optimization or explicit uncertainty. If no active betas/scale optimization exists, stop with `missing_v21_active_mano_shape_scale_optimizer` for exact hand-shape claims.

## 6. Target object discovery and segmentation

### 6.1 Object plan

The agent writes:

```text
$RUN_ROOT/measurements/object_candidates/object_plan_agent.json
```

It must include:

- selected target object instances from visual/task evidence;
- active intervals;
- physical branch hypotheses;
- open-vocabulary prompts for OWLv2;
- selected OWLv2 detector keyframes;
- approved bbox prompt review criteria;
- rejected alternatives and why;
- public dataset/model roster use, if any, explicitly labeled as model library only.

VLM schema scripts may guide structure, but the runtime default is agent visual judgment unless the user authorizes API-based VLM assistance.

### 6.2 OWLv2 Bbox-Prompt SAM2 Segmentation

Run the active mask/track path from agent keyframes through approved OWLv2 boxes:

```bash
python scripts/select_v21_agent_keyframes_from_plan.py ...
python scripts/run_v21_owlv2_bbox_proposals.py ...
python scripts/approve_v21_owlv2_bbox_prompts.py ...
python scripts/run_v21_sam2_proper_segmentation.py ...
```

Expected artifacts:

```text
measurements/object_candidates/segmentation_stable_keyframes.json
measurements/object_candidates/owlv2_bbox_proposals.json
measurements/object_candidates/owlv2_bbox_approved_prompts.json
measurements/object_tracks/sam2_proper/<object_id>/sam2_track.json
measurements/object_tracks/sam2_proper/<object_id>/sam2_masks/*.png
measurements/object_tracks/sam2_proper_summary.json
```

### 6.3 Assisted segmentation candidate

If depth/stereo/multiview assistance is available, produce it only as an additional candidate after the active bbox-prompt SAM2 track exists. If the assisted candidate needs a new depth-aware SAM2/ROI/registration adapter, stop with `missing_v21_depth_assisted_segmentation_adapter` rather than silently treating another mask source as active segmentation.

### 6.4 Segmentation comparison and contamination review

Write:

```text
$RUN_ROOT/review/segmentation_sam2_proper/segmentation_contamination_review.json
$RUN_ROOT/review/segmentation_sam2_proper/<object_id>/segmentation_contamination_review.jpg
```

Review the active SAM2 proper track, and compare any later assisted candidate against it:

- correct target identity;
- boundary completeness;
- missing visible object pixels;
- background/table contamination;
- temporal drift;
- mask-depth edge agreement;
- downstream visible-surface residual;
- rendered overlay readability.

If segmentation is weak, tune before downweighting:

```text
$RUN_ROOT/tuning/segmentation/<object_id>/attempt_<k>.json
```

Tuning examples:

- OWLv2 text prompts and score/area thresholds;
- keyframe seed choice;
- approved bbox selection;
- SAM2 box-prompt propagation controls;
- component filtering thresholds;
- registration between depth and RGB for review;
- rejected target alternatives.

Wrong-object masks and table/background capture are systematic errors. They must not enter geometry fitting as weak observations.

## 7. Visible surfaces and object branch decision

After accepted masks and selected depth/camera evidence exist, lift masks to visible metric surfaces:

```bash
python scripts/build_v19_visible_geometry_from_sam2_depth.py ...
```

or use an equivalent visible-surface builder if the current annotation format requires a V20/V21 adapter.

The agent then commits each object branch:

- rigid;
- articulated;
- deformable;
- support/occluder;
- unresolved.

The decision must include falsifiers and consequences. If rigid, visible surface is only the measurement anchor and the rigid branch is mandatory.

## 8. Rigid object geometry and pose

For rigid objects, build a condition packet:

```text
$RUN_ROOT/measurements/geometry_completion/condition_packets/<object_id>.json
```

Then run real mesh candidate generation. Existing options when inputs and environments exist:

```bash
python scripts/reconstruct_object_mesh_v2.py ...
python scripts/reconstruct_scaled_observed_object_mesh_v3.py ...
python scripts/reconstruct_object_visual_hull_depth_carve_v3.py ...
python scripts/complete_object_heightfield_from_mask_depth_v3.py ...
python scripts/remote_run_trellis_shape_v3.py ...
python scripts/build_v18_compact_rigid_trellis_completion.py ...
python scripts/build_v18_scale_sane_compact_rigid_completion.py ...
python scripts/fit_v20_cad_mesh_to_visible_depth.py ...   # benchmark public CAD only, target-plan tied
```

Register and validate candidates:

```bash
python scripts/build_v20_geometry_candidate_registry.py ...
python scripts/validate_v20_geometry_candidates.py ...
```

If no real mesh output exists, stop with `missing_v21_conditioned_geometry_generation_output` or the more specific missing backend. Do not register fake candidates.

Fit pose and optimize:

```bash
python scripts/fit_v18_compact_rigid_object_pose.py ...
python scripts/solve_v19_rigid_object_pose_graph.py ...
python scripts/optimize_object_factor_graph_v3.py ...
python scripts/optimize_mesh_prior_pose_graph_v3.py ...
python scripts/optimize_joint_camera_object_graph_v3.py ...
python scripts/optimize_joint_mano_object_graph_v3.py ...
python scripts/optimize_contact_patch_object_pose_graph_v3.py ...
```

Use the strongest applicable existing graph after its inputs exist. If no wrapper connects these outputs into V21 state, stop with `missing_v21_rigid_branch_state_adapter`.

## 9. Contact, occlusion, and nonpenetration

Run these only when geometry and MANO support them:

```bash
python scripts/build_v18_contact_ownership_graph.py ...
python scripts/build_v18_occlusion_owner_graph.py ...
python scripts/build_v18_signed_nonpenetration_evidence.py ...
python scripts/build_v18_triangle_nonpenetration_evidence.py ...
python scripts/build_v18_mano_object_constraint_state.py ...
python scripts/apply_v18_mano_object_constraint_state.py ...
```

If object mesh pose or metric MANO is weak, write explicit unresolved states. Do not use render-only contact points as evidence.

## 10. State assembly and rendering

Write V21 state:

```text
$RUN_ROOT/state/v21_physical_state.json
$RUN_ROOT/state/v21_uncertainty_state.json
$RUN_ROOT/state/v21_observation_bundle.json
$RUN_ROOT/state/annotations_v21_renderable.json
$RUN_ROOT/state/v21_agent_evidence.md
```

If no native V21 renderer exists, use V18 renderer only through an explicit adapter that maps V21 state to the renderer's expected annotation/mesh/pose files:

```bash
python scripts/render_v18_full_pipeline_from_annotations.py ...
```

The output paths must be V21 deliverables:

```text
$RUN_ROOT/renders/v21_overlay.mp4
$RUN_ROOT/renders/v21_world.mp4
$RUN_ROOT/renders/v21_side_by_side.mp4
```

If the only available renderer would produce scatter/skeleton diagnostic views rather than metric MANO + object mesh pose for accepted rigid claims, stop with `missing_v21_metric_mesh_pose_renderer` or label the render as diagnostic, not final V21.

## 11. Visual review

The agent must consume rendered artifacts before claiming progress. Review must answer:

- Do masks cover the intended object at pixel level?
- Is native-depth/stereo better than or at least not worse than monocular baseline?
- Is MANO metric or only overlay-aligned?
- Does the rigid object render as fitted/adapted mesh pose?
- Are contact/occlusion/nonpenetration visible or explicitly unresolved?
- Did any tuning improve the physical variable it targeted?
- Did any V19 capability regress?

Record findings in `state/v21_agent_evidence.md`.

## 12. Benchmark loop

After prediction state/render exists:

```bash
python scripts/evaluate_v20_benchmark_gt.py \
  --state "$RUN_ROOT/state/v21_physical_state.json" \
  --uncertainty-state "$RUN_ROOT/state/v21_uncertainty_state.json" \
  --dataset-manifest "$RUN_ROOT/input/dataset_manifest.json" \
  --reference-manifest "$RUN_ROOT/evaluation/reference_manifest.json" \
  --output-dir "$RUN_ROOT/evaluation/iteration_${ITERATION}"
```

Use this evaluator only if it accepts the V21 state fields produced by the prediction run or after an explicit compatibility adapter. Otherwise stop with `missing_v21_gt_evaluator_adapter`.

For each iteration, append:

```text
$RUN_ROOT/evaluation/benchmark_iterations.jsonl
$RUN_ROOT/evaluation/algorithm_parameter_changes.jsonl
```

Each next intervention must be one atomic mechanism change based on failure clusters. Examples:

- tune stereo rectification/disparity because stereo depth underperforms monocular;
- change SAM2 prompt/ROI/keyframe because mask contamination drives geometry rejection;
- change MANO crop/intrinsics/scale because hand projection/depth residual is systematic;
- switch mesh candidate or condition packet because silhouette/free-space residual rejects geometry;
- adjust graph residual weights only after observation strength has been tuned.

Stop when metrics and visual quality plateau, iteration budget is exhausted, no supported mechanism remains, or the next action would require a missing implementation.

## 13. Yield gate

Before reporting, verify:

- all final claims are driven by `state/`;
- bottleneck downweighting, if any, followed strong tuning or named missing implementation;
- depth/stereo/native-depth claims were compared against monocular baseline;
- segmentation has contamination review;
- hand state distinguishes metric from overlay-only;
- rigid objects are not represented by centroids/scatter;
- benchmark GT did not leak into prediction;
- parameter sets are sample-bound;
- unresolved physical variables are explicit.
