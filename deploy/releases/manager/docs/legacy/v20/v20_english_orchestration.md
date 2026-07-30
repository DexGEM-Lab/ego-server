# V20 English Orchestration Over Real Components

Status: V20 harness draft-to-deployment artifact. This runbook is derived from `docs/v19_english_orchestration.md` and keeps V19 physical-state semantics. V20 adds mode selection and benchmark-driven refinement. Every executable action must name an existing repository script, a copied V20 harness prompt/contract file, or an explicit missing implementation. It does not authorize fake numbered scripts, JSON registries, validator loops, or V19/V18 artifact repackaging as progress.

## 0. Modes and physical objective

V20 has two modes:

- `v20_infer`: arbitrary input video -> renderable physical annotation state and full-duration overlay/world/side-by-side videos.
- `v20_benchmark`: public GT dataset sample or batch -> the same physical annotation flow, plus dataset fail-fast loading, GT-aligned evaluation, controller feedback, and multiple documented iterations.

The physical variables remain:

- `K_t`, `T_world_camera,t`, and depth/scale evidence;
- `H_t`: metric MANO hand state with side, camera/world semantics, provenance, visibility, and uncertainty;
- `O_i`: object instances with masks/tracks and physical branch: rigid, articulated, deformable, support/occluder, or unresolved;
- `G_i`: object geometry, with completed/adapted instance mesh required for rigid branches;
- `T_world_object,i,t`: object or part pose trajectory/posterior for rigid/articulated branches;
- `C_t`, `V_t`, `N_t`: contact/near-contact, visibility/occlusion ownership, and nonpenetration residual/uncertainty;
- benchmark GT alignment and evaluation state when `v20_benchmark` is active;
- rendered overlay/world/side-by-side videos whose visible marks are caused by the variables above.

A command is progress only when it measures, optimizes, renders, evaluates, or falsifies one of these variables. Reports, schemas, row counts, and validation outputs are evidence only.

## 1. Scope of current command truth

### Directly reusable existing scripts

V20 reuses the V19/V18 component map unless a V20 adapter explicitly supersedes a component:

- Timeline/frame extraction: `scripts/run_v19_v20_infer_bootstrap.py` writes a standalone raw-frame manifest for `v19_infer`/`v20_infer`; older V16 manifest functions remain reusable when prior V16/V17/V18 artifacts exist.
- Minimal V20 infer closure: `scripts/build_v20_infer_point_prompts_from_object_plan.py`, `scripts/filter_v20_sam2_masks_by_prompt_components.py`, `scripts/build_v20_infer_base_annotations.py`, `scripts/solve_v20_infer_temporal_observation_graph.py`, `scripts/render_v20_benchmark_annotations.py`, and `scripts/assemble_v20_state_from_v19_annotations.py` produce renderable approximate state from selected target SAM2 tracks, optional depth, optional HaWoR MANO, and temporal observation smoothing.
- Camera/depth/SLAM: `scripts/run_droid_full_frame.py`, `scripts/run_unidepth_full_frame_v3.py`, `scripts/run_unidepth_metric_source_v3.py`, plus `scripts/build_v20_depth_candidate_registry.py`, `scripts/select_v20_depth_observation_bundle.py`, and `scripts/build_v20_uncalibrated_stereo_disparity.py` for V20 depth evidence. Uncalibrated stereo is weak non-metric evidence only.
- Hand measurements: `scripts/run_rtmlib_hand2d_v3.py`, `scripts/run_wilor_full_frame.py`, `scripts/export_hawor_world.py`, `scripts/run_hamer_rtmlib_hand_stream_v3.py`, `scripts/merge_hand_candidate_streams_v7.py`, `scripts/refit_mano_metric_depth_v3.py`, `scripts/adapt_hawor_camspace_to_v20_mano_npz.py`, `scripts/solve_v20_hand_shape_track.py`, and `scripts/align_v20_hawor_overlay_to_detector_boxes.py`. Detector-box overlay alignment is visualization-only and must not repair metric MANO/contact claims.
- Agent-replaced VLM structures: `scripts/build_object_plan_vlm.py` and `scripts/build_object_point_prompts_vlm.py` define schemas; the runtime agent writes equivalent files instead of calling the API.
- Masks/tracks: `scripts/run_sam2_vlm_points_multiobject.py`, with `scripts/filter_v20_sam2_masks_by_prompt_components.py` for object-plan component filtering.
- SAM2/depth visible-surface bridge for rigid branches: `scripts/build_v19_visible_geometry_from_sam2_depth.py` or `scripts/build_v20_infer_base_annotations.py` for the current V20 visible-surface annotation backbone.
- Observed object geometry and optimization candidates: `scripts/reconstruct_object_mesh_v2.py`, `scripts/reconstruct_scaled_observed_object_mesh_v3.py`, `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`, `scripts/complete_object_heightfield_from_mask_depth_v3.py`, `scripts/optimize_object_factor_graph_v3.py`, `scripts/optimize_joint_mano_object_graph_v3.py`, `scripts/optimize_joint_camera_object_graph_v3.py`, `scripts/optimize_contact_patch_object_pose_graph_v3.py`, `scripts/build_v20_geometry_candidate_registry.py`, and `scripts/validate_v20_geometry_candidates.py`.
- V18 rigid branch components: `scripts/build_v18_compact_rigid_evidence_bundle.py`, `scripts/remote_run_trellis_shape_v3.py`, `scripts/build_v18_compact_rigid_trellis_completion.py`, `scripts/build_v18_scale_sane_compact_rigid_completion.py`, `scripts/fit_v18_compact_rigid_object_pose.py`, and `scripts/fit_v20_cad_mesh_to_visible_depth.py` for public-CAD benchmark candidates tied to selected object-plan targets.
- Contact/render/state: `scripts/build_v20_contact_point_render_rows.py`, `scripts/build_v20_observation_bundle.py`, `scripts/render_v20_benchmark_annotations.py`, plus MANO/object correction and interval rendering scripts from V18 when a stronger branch state exists.

### Explicit missing implementation, not to be faked

The current repository can close a minimal approximate V20 infer run when an object plan, real masks, available depth/weak depth evidence, and optional hand candidates exist. These gaps still block full-strength physical claims:

1. **Automatic target-object discovery.** A public dataset/model roster or background inventory is not a target list. If no object plan or point prompts can be selected from visual/task evidence, stop with `missing_v20_target_object_plan`.
2. **Complete object mesh reconstruction for arbitrary objects.** Visible surfaces are renderable approximate geometry, not completed object pose. If a rigid/object-pose claim is required and no completed/adapted mesh candidate exists, stop with `missing_v20_conditioned_geometry_generation_output` or an equivalent completed-geometry failure.
3. **Metric stereo/multiview depth without calibration.** `scripts/build_v20_uncalibrated_stereo_disparity.py` may produce weak relative disparity evidence, but metric depth/object/contact claims require calibration/rectification or native depth. Without it, primary metric depth remains unset.
4. **High-quality metric MANO/contact.** HaWoR/WiLoR/HaMeR rows may be rendered with uncertainty, but visibly offset or detector-aligned overlay-only MANO cannot support contact ownership, nonpenetration, or metric hand-object claims.
5. **Benchmark GT isolation and evaluation.** `scripts/prepare_v20_benchmark_dataset.py` writes prediction inputs and `evaluation/reference_manifest.json`; `scripts/evaluate_v20_benchmark_gt.py` reads GT only after completed prediction-side state/renders exist.

If a run reaches one of these gaps, the correct outcome is a named missing implementation or scoped uncertainty state with the physical variable blocked. Do not invent a script name or write placeholder success.

## 2. Runtime variables

Use variables explicitly so commands remain general:

```bash
REPO_ROOT=/home/yiwen/ego_annotation
PYTHON="$REPO_ROOT/.venv/bin/python"
MODE="v20_infer | v20_benchmark"
INPUT_VIDEO="<input video>"
RUN_ROOT="<run root>"
CASE_ID="<case id>"
DATASET_NAME="ycb | dexycb | dex-ycb | ho3d"
DATASET_ROOT="<dataset root>"
SAMPLE_ID="<sample id>"
FRAME_START=0
FRAME_END="<last source frame index>"
RAW_FRAME_MANIFEST="<raw_frame_manifest/manifest.json>"
BASE_ANNOTATIONS="<one-frame-per-source-frame annotations json>"
OBJECT_ID="object:<track_id>"
TRACK_ID="<track_id>"
GPU_ID="<selected GPU>"
SAM2_CHECKPOINT="<sam2.1 checkpoint path>"
TRELLIS_REPO="<TRELLIS repo root>"
UNIDEPTH_REPO="<UniDepth repo root if not importable>"
DROID_ROOT="<DROID root>"
WILOR_ROOT="<WiLoR root>"
HAMER_ROOT="<HaMeR root>"
HAMER_CHECKPOINT="<HaMeR checkpoint>"
HAWOR_ROOT="<HaWoR root>"
```

Heavy commands run on the declared A800/server target after a non-mutating probe. Use one tmux session for long-running local/remote jobs; do not use `sleep` or polling loops.

## 3. V20 input handling

### 3.1 `v20_infer` raw video input

Use the V19 timeline and coordinate-frame procedure:

- verify the raw video path, frame count, FPS, resolution, and duration;
- write `input/input_manifest.json` before measurement work;
- use `scripts/run_v19_v20_infer_bootstrap.py --mode v20` to create `input/raw_frame_manifest/manifest.json` for fresh videos;
- use existing V16/V17/V18 raw-frame manifests only as development inputs with explicit provenance;
- if the raw video is absent, write an input-contract failure rather than fabricating annotations.

### 3.2 `v20_benchmark` fail-fast dataset loading

Benchmark mode must validate dataset structure before any inference, optimization, rendering, or metric computation. The implemented path is `scripts/prepare_v20_benchmark_dataset.py`; it should be used for `/V20_benchmark:ycb`, `/V20_benchmark:ho3d`, and `/v20_benchmark` setup. It writes RGB/depth/calibration/object-model prediction inputs under `input/` and writes GT references only under `evaluation/reference_manifest.json` for post-render evaluation.

#### DexYCB local contract

Supported root and observed representative sample:

```text
DATASET_NAME=dexycb
DATASET_ROOT=/mnt/nas/dex-ycb
SAMPLE_ID=20200813-subject-02/20200813_151041/932122062010
SEQUENCE_ROOT=/mnt/nas/dex-ycb/20200813-subject-02/20200813_151041
CAMERA_ROOT=/mnt/nas/dex-ycb/20200813-subject-02/20200813_151041/932122062010
```

Fail-fast checks:

- `DATASET_NAME` must equal `dexycb`.
- `DATASET_ROOT` must exist.
- `SEQUENCE_ROOT/meta.yml` must exist and include `serials`, `num_frames`, `ycb_ids`, and `mano_sides`.
- `SEQUENCE_ROOT/pose.npz` must exist and include `pose_m` and `pose_y`.
- `CAMERA_ROOT/color_%06d.jpg`, `CAMERA_ROOT/aligned_depth_to_color_%06d.png`, and `CAMERA_ROOT/labels_%06d.npz` must exist for all selected frames.
- Each label NPZ must include `seg`, `pose_y`, `pose_m`, `joint_3d`, and `joint_2d`.
- Frame counts must match `meta.yml:num_frames`; the observed representative has 72 frames.

Dataset semantics:

- RGB source is `color_*.jpg`.
- Depth source is `aligned_depth_to_color_*.png`; unit/scale must be recorded and validated before metric comparison.
- GT object and hand sources are `labels_*.npz` and `pose.npz`; they are GT measurements with dataset camera/object semantics, not pipeline state.

#### HO3D local contract

Supported root and observed representative samples:

```text
DATASET_NAME=ho3d
DATASET_ROOT=/mnt/nas/ho3d/HO3D_dataset/HO3D
SAMPLE_ID=train/MC1
SAMPLE_ID=train/BB10
```

Fail-fast checks for `train/<sequence>`:

- `DATASET_ROOT/train/<sequence>/rgb/%04d.jpg` must exist for selected frames.
- `DATASET_ROOT/train/<sequence>/depth/%04d.png` must exist for selected frames.
- `DATASET_ROOT/train/<sequence>/meta/%04d.pkl` must exist for selected frames.
- Meta pickle must contain `camMat`, `handPose`, `handTrans`, `handBeta`, `handJoints3D`, `objRot`, `objTrans`, `objCorners3D`, `objCorners3DRest`, `objName`, and `objLabel` for selected benchmark frames.
- `train.txt` should be used for official training/evaluation frame lists because the HO3D README states some images do not contain annotations.
- The observed local `MC1` folder has 897 RGB/depth/meta files, while `train.txt` contains 814 annotated `MC1` benchmark frames. The observed local `BB10` folder has 1606 RGB/depth/meta files; use `train.txt` to derive its annotated benchmark frames.

Dataset semantics:

- RGB source is `rgb/*.jpg`.
- Depth source is `depth/*.png`, decoded according to the HO3D README and local utilities; do not treat raw PNG intensity as metric depth without decoding.
- GT hand/object annotations are in `meta/*.pkl` and assume OpenGL camera coordinates according to the HO3D README.

## 4. Build physical measurements

After `v20_infer` input verification or `v20_benchmark` prediction dataset manifest creation, follow the V19 harness sequence. The dataset/public object roster is only a candidate model library; it is never the annotation target list.

1. Establish the timeline and coordinate frame.
2. Run or import camera/depth/scale sources.
3. Build metric MANO measurements.
4. Agent writes the target object plan and point prompts from visual/task evidence.
5. Segment and track only target object surfaces with SAM2.
6. Lift target masks to visible metric surfaces.
7. Agent selects each target object's physical branch with explicit falsifiers.
8. Reconstruct/complete/fuse observed object geometry when required by the selected branch.
9. Run branch optimization/factor correction before final state assembly and render.

Weak local measurements continue downstream with uncertainty. Dataset GT must not be used to correct, initialize, select, render, or otherwise influence the prediction path. GT is consumed only by `scripts/evaluate_v20_benchmark_gt.py` after a completed prediction-side prediction state and render exist.

## 5. V20 observation-enhancement stages

These stages are mandatory in `v20_infer` and `v20_benchmark` prediction when their required prediction-side inputs exist, but they are observation-enhancement stages only. They must be keyed to the selected object plan and consumed by branch optimization/factor correction before final state/render. If the needed model output is absent, stop with the named missing mechanism; do not write placeholder success.

### 5.1 Depth modality and candidate registry

After raw/dataset RGB/depth inputs and any remote depth outputs exist, register all prediction-side depth candidates:

```bash
python "$REPO_ROOT/scripts/build_v20_depth_candidate_registry.py" \
  --input-video "$INPUT_VIDEO" \
  --candidate "unidepth_metric|npz|$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz|unidepth_metric_depth|monocular_video|retained_uncertain|1.0" \
  --output-dir "$RUN_ROOT/measurements/depth_candidates" \
  --require-candidate
```

For dataset RGB-D/native depth, register an image/NPZ candidate with source modality `native_depth` or `rgbd`. For stereo, use calibrated stereo depth when calibration/rectification exists. If calibration is absent, `scripts/build_v20_uncalibrated_stereo_disparity.py` may produce weak relative inverse-depth evidence; record it with `--weak-depth-report` in the observation bundle and do not select it as primary metric depth.

### 5.2 Depth selector/evaluator

Select depth by physical residuals rather than visual preference:

```bash
python "$REPO_ROOT/scripts/select_v20_depth_observation_bundle.py" \
  --registry "$RUN_ROOT/measurements/depth_candidates/depth_candidate_registry.json" \
  --annotations "$BASE_ANNOTATIONS" \
  --contact-report "<optional_contact_report.json>" \
  --output-report "$RUN_ROOT/measurements/depth_candidates/depth_selection_report.json" \
  --output-bundle "$RUN_ROOT/state/v20_depth_observation_bundle.json" \
  --require-selected
```

The selector computes mask continuity, visible-surface depth residual, hand-depth residual, temporal smoothness, and contact-gap residual when those inputs exist. Weak but usable candidates remain in the report; they are not deleted to make the artifact look certain.

### 5.3 Geometry candidate registry and conditioning packets

After TRELLIS/equivalent or conditioned generation has produced real mesh files on the declared server/A800 target, standardize them:

```bash
python "$REPO_ROOT/scripts/build_v20_geometry_candidate_registry.py" \
  --object-plan "$RUN_ROOT/measurements/object_candidates/object_plan_agent.json" \
  --depth-candidate-id "<selected_depth_candidate_id>" \
  --candidate "$OBJECT_ID|trellis_candidate|trellis_or_equivalent_image_to_3d_prior|<trellis_mesh.ply>|<trellis_report.json>|$TRACK_ID" \
  --candidate "$OBJECT_ID|conditioned_generation_candidate|agent_evidence_conditioned_geometry_generation|<conditioned_mesh.ply>|<conditioned_report.json>|$TRACK_ID" \
  --output-dir "$RUN_ROOT/measurements/geometry_completion" \
  --require-candidate
```

If no conditioned generator output exists, stop with `missing_v20_conditioned_geometry_generation_output`; do not create a missing-implementation candidate as success.

### 5.4 Geometry validation and promotion

Promote generated/reconstructed geometry only after prediction-side visible evidence supports it:

```bash
python "$REPO_ROOT/scripts/validate_v20_geometry_candidates.py" \
  --registry "$RUN_ROOT/measurements/geometry_completion/geometry_candidate_registry.json" \
  --annotations "$BASE_ANNOTATIONS" \
  --hidden-volume-validation "<optional_hidden_volume_validation.json>" \
  --nonpenetration-report "<optional_nonpenetration_report.json>" \
  --contact-report "<optional_contact_report.json>" \
  --output-report "$RUN_ROOT/measurements/geometry_completion/geometry_validation_report.json" \
  --output-bundle "$RUN_ROOT/state/v20_geometry_observation_bundle.json" \
  --output-dir "$RUN_ROOT/measurements/geometry_completion/validated"
```

Only `promoted_geometry_observation` candidates may become strong object pose/contact/nonpenetration constraints. Other candidates remain weak priors or uncertainty render layers.

### 5.5 Per-hand-track MANO betas/scale solve

After prediction-side MANO candidates and depth refit exist, solve track-level shape priors:

```bash
python "$REPO_ROOT/scripts/solve_v20_hand_shape_track.py" \
  --annotations "$BASE_ANNOTATIONS" \
  --depth-refit-report "$RUN_ROOT/measurements/hand_candidates/refit_mano_metric_depth.json" \
  --output-report "$RUN_ROOT/measurements/hand_shape/hand_shape_solve_report.json" \
  --output-npz "$RUN_ROOT/measurements/hand_shape/mano_betas_posterior.npz"
```

The posterior is a prior/initialization for later MANO interval solve, not a claim of exact hand shape.

### 5.6 Render-only contact points

After hand/object surfaces and contact or near-contact state exist, generate render-only markers:

```bash
python "$REPO_ROOT/scripts/build_v20_contact_point_render_rows.py" \
  --annotations "$BASE_ANNOTATIONS" \
  --output "$RUN_ROOT/measurements/contact_visualization/contact_point_render_rows.json" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END"
```

These rows must have `evidence_created=false` and must not enter contact evidence or contact ownership solvers.

### 5.7 V20 observation bundle

Merge the selected V20 sidecars. This bundle is optimization evidence, not final physical state:

```bash
python "$REPO_ROOT/scripts/build_v20_observation_bundle.py" \
  --depth-registry "$RUN_ROOT/measurements/depth_candidates/depth_candidate_registry.json" \
  --depth-selection "$RUN_ROOT/measurements/depth_candidates/depth_selection_report.json" \
  --geometry-registry "$RUN_ROOT/measurements/geometry_completion/geometry_candidate_registry.json" \
  --geometry-validation "$RUN_ROOT/measurements/geometry_completion/geometry_validation_report.json" \
  --hand-shape-report "$RUN_ROOT/measurements/hand_shape/hand_shape_solve_report.json" \
  --contact-render-rows "$RUN_ROOT/measurements/contact_visualization/contact_point_render_rows.json" \
  --weak-depth-report "$RUN_ROOT/measurements/depth_candidates/uncalibrated_stereo_disparity_report.json" \
  --output "$RUN_ROOT/state/v20_observation_bundle.json"
```

Pass `--weak-depth-report` only for reports that declare `metric_depth_available=false`; it records evidence and uncertainty but does not create a primary metric depth candidate.

The bundle rejects GT/oracle markers in prediction artifacts. It must not be rendered as final state until Section 6 branch optimization/factor correction has produced optimized annotation/pose reports.

## 6. Branch-specific optimization and rendering

### 6.0 Minimal V20 infer temporal observation graph

When complete rigid mesh/pose branches are unavailable but selected target masks, visible surfaces, optional depth, and optional hand observations exist, run the minimal renderable V20 infer graph. This is an artifact-producing approximation, not a substitute for full mesh reconstruction:

```bash
python "$REPO_ROOT/scripts/solve_v20_infer_temporal_observation_graph.py" \
  --annotations "$BASE_ANNOTATIONS" \
  --output-annotations "$RUN_ROOT/state/annotations_v20_temporal_optimized.json" \
  --output-report "$RUN_ROOT/state/v20_temporal_observation_graph_report.json"
```

The graph smooths per-frame visible-surface object centers and MANO observations over time. Its report is acceptable branch evidence for visible-surface-only renderable state, but any rigid object pose, mesh completion, contact ownership, or nonpenetration claim still requires the stronger branches below.

### 6.1 Rigid branch

After a rigid decision, visible surfaces are measurements only. The branch must execute:

```text
evidence crop/bundle
  -> TRELLIS or equivalent mesh completion
  -> metric completion/adaptation to observed surfaces
  -> visible-frame pose fitting
  -> V20/V19 temporal rigid-pose correction
  -> object/MANO/contact/nonpenetration/occlusion factor correction
  -> hidden-volume/depth validation
  -> corrected mesh-pose rendering
```

Use existing V18/V19 tools listed in Section 1 until V20-specific replacements exist.

### 6.2 Articulated, deformable, support, unresolved branches

- Articulated: require part masks/tracks and part pose/articulation evidence before pose claims.
- Deformable: render uncertainty/deformation; do not force a rigid pose.
- Support/occluder: include in contact/occlusion reasoning without claiming manipulated-object pose.
- Unresolved: render uncertainty and request the next discriminating measurement.

### 6.3 Renderable state assembly

Current practical state backbone remains a V18/V19-compatible optimized annotation JSON plus V20 sidecars. Assemble only after branch optimization/factor correction has produced reports for the selected targets:

```text
state/v20_physical_state.json
state/v20_uncertainty_state.json
state/v20_agent_evidence.md
state/annotations_v20_renderable.json
```

```bash
python "$REPO_ROOT/scripts/assemble_v20_state_from_v19_annotations.py" \
  --annotations "$OPTIMIZED_ANNOTATIONS" \
  --observation-bundle "$RUN_ROOT/state/v20_observation_bundle.json" \
  --render-summary "$RUN_ROOT/renders/render_summary.json" \
  --branch-optimization-report "$RUN_ROOT/measurements/object_pose_graph/object_pose_graph_report.json" \
  --require-branch-optimization \
  --dataset-manifest "$RUN_ROOT/input/dataset_manifest.json" \
  --run-root "$RUN_ROOT" \
  --mode "$MODE" \
  --case-id "$CASE_ID" \
  --output-state "$RUN_ROOT/state/v20_physical_state.json" \
  --output-uncertainty "$RUN_ROOT/state/v20_uncertainty_state.json" \
  --output-annotations "$RUN_ROOT/state/annotations_v20_renderable.json"
```

These sidecars must name which annotation/interval/mesh/pose files drive each visual layer. They are not completion evidence by themselves, and the assembler must fail with `missing_v20_branch_optimization_report` when final assembly is attempted without optimized branch evidence.

### 6.4 Full-duration rendering

Use the native V20 renderable-annotation renderer for the minimal V20 annotation backbone:

```bash
python "$REPO_ROOT/scripts/render_v20_benchmark_annotations.py" \
  --annotations "$RUN_ROOT/state/annotations_v20_temporal_optimized.json" \
  --output-dir "$RUN_ROOT/renders" \
  --output-summary "$RUN_ROOT/renders/render_summary.json" \
  --fps 30
```

When a stronger V18/V19 interval or mesh-pose state is available, the V18 renderers may still be used and copied into the same standardized output names:

```text
$RUN_ROOT/renders/v20_overlay.mp4
$RUN_ROOT/renders/v20_world.mp4
$RUN_ROOT/renders/v20_side_by_side.mp4
```

## 7. V20 benchmark GT evaluation agent

The benchmark evaluation agent is a separate role in the Pi harness. It consumes final state/renders for an iteration plus dataset GT, then writes scoped evaluation evidence. It does not directly edit pipeline state.

### Inputs

```text
$RUN_ROOT/state/v20_physical_state.json
$RUN_ROOT/state/v20_uncertainty_state.json
$RUN_ROOT/state/annotations_v20_renderable.json
$RUN_ROOT/renders/<iteration render files>
$RUN_ROOT/input/dataset_manifest.json
<dataset GT files>
```

### Outputs

```text
$RUN_ROOT/evaluation/iteration_<k>/gt_metrics.json
$RUN_ROOT/evaluation/iteration_<k>/gt_alignment.json
$RUN_ROOT/evaluation/iteration_<k>/failure_clusters.json
$RUN_ROOT/evaluation/iteration_<k>/evaluation_agent_report.md
```

### Metric semantics

The evaluation agent must compare the same physical meaning:

- MANO/joints: camera-frame or explicitly converted frame errors.
- Object pose: SE(3)/Sim(3) errors after unit, object-frame, camera-frame, and symmetry semantics are defined.
- World/camera motion: relative frame-to-frame deltas or aligned trajectories when absolute origins differ.
- Contact/nonpenetration: compare only when dataset labels or derivable geometry support the claim.
- Occlusion/visibility: stratify metrics when dataset labels or reliable visual evidence support the state.

If only a proxy metric is available, label it as diagnostic and do not use it to claim physical success.

## 8. Benchmark controller loop

For `v20_benchmark`, repeat up to `max_benchmark_iterations`:

1. Run the physical annotation flow for the selected sample or batch.
2. Render the current state.
3. Run the GT evaluation agent.
4. Interpret failure clusters as mechanisms, not just metric deltas.
5. Propose one atomic intervention: measurement change, factor enable/disable, weight/sigma/threshold adjustment, branch correction, prompt/mask repair, camera/depth correction, or renderer adapter fix.
6. Record the predicted effect and falsifier before applying the intervention.
7. Apply the intervention only if it can improve the physical artifact or metric under supported semantics.
8. Rerun the affected stages and render again.

The final deliverable is:

```text
$RUN_ROOT/evaluation/benchmark_iterations.jsonl
$RUN_ROOT/evaluation/algorithm_parameter_changes.jsonl
$RUN_ROOT/evaluation/final_selection_report.md
$RUN_ROOT/renders/v20_overlay.mp4
$RUN_ROOT/renders/v20_world.mp4
$RUN_ROOT/renders/v20_side_by_side.mp4
```

A metric improvement that makes the rendered annotation physically worse is a failed iteration requiring explanation.

## 9. Visual consumption and repair loop

The agent must consume final videos as physical annotations before claiming progress. The review asks:

- Do MANO joints/surfaces align with visible hands where visible?
- Are occluded hands shown as uncertain rather than hallucinated certainty?
- Does the completed rigid mesh project onto the object, not onto the hand/support/background?
- Does the world view agree with overlay depth/order?
- Does contact/near-contact/non-contact follow the rendered geometry and timing?
- Are nonpenetration corrections bounded, local, and compatible with visible 2D evidence?
- For benchmark mode, do metric failures correspond to visible/geometric failures, or do they expose coordinate/semantic mismatch?

Repair is allowed only when it targets a named mechanism. Repeating validators or writing new status fields is not repair.

## 10. Ordered execution summary

### `v20_infer`

1. Verify raw input video and run root.
2. Establish timeline/manifest and camera pose; stop at missing fresh-video components when required.
3. Run camera/depth/SLAM measurements.
4. Run hand candidate measurements and metric/depth refit.
5. Agent writes object plan and point prompts from visual evidence.
6. Run SAM2 multi-object masks/tracks.
7. Lift masks to visible metric surfaces and reconstruct observed geometry as needed.
8. Agent chooses physical branch with falsifiers.
9. Build V20 depth registry and run depth selector/evaluator.
10. For rigid objects: evidence crop -> TRELLIS/conditioned geometry output -> V20 geometry registry -> validation/promotion -> metric completion/adaptation -> visible-frame pose -> object/MANO correction/factors -> hidden-volume validation.
11. Solve per-hand-track MANO betas/scale posterior and feed it as a prior into interval MANO when applicable.
12. Solve interval MANO over selected physical intervals with object/camera/depth/contact/visibility factors actually available.
13. Build render-only contact point rows from existing contact/near-contact state.
14. Build V20 observation bundle and assemble V20 prediction state.
15. Render full-duration overlay/world/side-by-side.
16. Visually consume the rendered artifact, identify mechanism failures, repair, and rerender within the evidence-cycle budget.

### `v20_benchmark`

1. Validate dataset name/root/sample against a hard-coded local contract with `scripts/prepare_v20_benchmark_dataset.py`.
2. Treat `input/dataset_manifest.json` as the only dataset input for prediction; `evaluation/reference_manifest.json` remains sealed until after prediction state/renders exist.
3. Run the same prediction-side physical annotation flow as `v20_infer`, including V20 depth registry/selector, geometry registry/validation, hand-shape solve, contact render rows, observation bundle, state assembly, and full render.
4. Render current state.
5. Run `scripts/evaluate_v20_benchmark_gt.py` and write iteration metrics/alignment/failure clusters.
6. Feed evaluation interpretation to the controller agent.
7. Apply one atomic algorithm/factor/parameter intervention if justified; never apply GT directly as prediction state.
8. Iterate until the declared budget is exhausted or no supported intervention remains.
9. Deliver all iterations, metrics, renders, and algorithm/parameter change records for prediction-without-reference-labels runs.
