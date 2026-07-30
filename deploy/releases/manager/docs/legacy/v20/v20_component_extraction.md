# V20 Component Extraction From V19/V18

Status: V20 harness component map. This file is a V20 copy and extension of the V19 component extraction. It separates reusable V18/V19 components from missing V20 benchmark adapters so `docs/v20_english_orchestration.md` can stay grounded in real commands and explicit fail-fast gaps.

## Key finding

V20 should not introduce an outer wrapper around Pi. Pi remains the harness. Existing V18/V19 scripts are reusable measurement, tracking, geometry, optimization, rendering, and diagnostic tools after their inputs exist. V20 adds two entry modes:

- `v20_infer`: arbitrary input video physical annotation;
- `v20_benchmark`: public GT dataset sample/batch, fail-fast dataset loading, GT evaluation agent, controller feedback loop.

V20 must not treat generated masks, completed geometry, metrics, dataset loaders, or public benchmark object rosters as final deliverables or target selections. The final artifact remains renderable physical annotation state plus overlay/world/side-by-side videos; benchmark mode additionally delivers iteration metrics and algorithm/parameter change records.

## Not reusable as V20 harness entry points

### `scripts/run_v18_full_pipeline.py`

Reusable pieces:

- dependency order and stage families;
- factor-graph/contact/occlusion/nonpenetration logic after inputs exist;
- final state/render assembly patterns.

Not reusable as V20 entry:

- it consumes many upstream roots and is not a raw-video or dataset-sample harness;
- it is too broad to be the agent-controlled loop;
- it must not replace `v20_infer` or `v20_benchmark` prompts.

### VLM object-plan scripts

- `scripts/build_object_plan_vlm.py`
- `scripts/build_object_point_prompts_vlm.py`

V20 policy: these define schemas only unless the user explicitly chooses API-based VLM assistance. Runtime default is agent visual judgment writing equivalent files.

## Reusable measurement and model-runner components

### Input/frame/depth/camera

- `scripts/run_unidepth_full_frame_v3.py` — UniDepth full-frame depth over frame spans.
- `scripts/run_unidepth_metric_source_v3.py` — metric depth source over spans.
- `scripts/run_droid_full_frame.py` — camera/head trajectory candidate.
- `scripts/run_v16_full_pipeline.py` — can regenerate a V16 manifest only when annotations/WiLoR QC/depth inputs already exist; not a clean V20 fresh-video first step.

Gap: no standalone V20 `input_video -> raw_frame_manifest/manifest.json` command exists yet.

### Hand / MANO

- `scripts/run_rtmlib_hand2d_v3.py`
- `scripts/run_wilor_full_frame.py`
- `scripts/export_hawor_world.py`
- `scripts/run_hamer_rtmlib_hand_stream_v3.py`
- `scripts/merge_hand_candidate_streams_v7.py`
- `scripts/refit_mano_metric_depth_v3.py`
- `scripts/solve_v18_joint_mano_interval_trajectory.py`

Policy: 2D detectors are prompts/measurements only. A valid hand state must come through MANO candidate/export/refit/interval solver and carry camera/world semantics and uncertainty.

### Object discovery, masks, tracking

- `scripts/run_sam2_vlm_points_multiobject.py`
- `scripts/run_sam2_vlm_points_track.py`
- `scripts/run_sam2_vlm_points_image.py`
- `scripts/run_sam2_object_track.py`
- prompt/mask visual review tools where available.

Policy: object discovery remains category-agnostic. Differences between object categories belong in agent-selected prompts/masks/state, not Python branches.

### Geometry / object pose / rigid branch

- `scripts/build_v19_visible_geometry_from_sam2_depth.py`
- `scripts/build_v18_depth_fused_reconstruction.py`
- `scripts/reconstruct_object_mesh_v2.py`
- `scripts/reconstruct_scaled_observed_object_mesh_v3.py`
- `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`
- `scripts/complete_object_heightfield_from_mask_depth_v3.py`
- `scripts/build_v18_compact_rigid_evidence_bundle.py`
- `scripts/remote_run_trellis_shape_v3.py`
- `scripts/build_v18_compact_rigid_trellis_completion.py`
- `scripts/build_v18_scale_sane_compact_rigid_completion.py`
- `scripts/fit_v18_compact_rigid_object_pose.py`
- `scripts/solve_v19_rigid_object_pose_graph.py`
- `scripts/optimize_object_factor_graph_v3.py`
- `scripts/optimize_contact_patch_object_pose_graph_v3.py`
- `scripts/optimize_joint_mano_object_graph_v3.py`
- `scripts/optimize_joint_camera_object_graph_v3.py`

Policy: if the agent decides an object is rigid, V20 must force completion/adaptation -> visible-frame pose -> factor graph correction -> corrected mesh-pose render. Visible surfaces cannot be final object pose.

### Contact / occlusion / nonpenetration / temporal graph

- `scripts/build_v18_contact_ownership_graph.py`
- `scripts/build_v18_occlusion_owner_graph.py`
- `scripts/build_v18_signed_nonpenetration_evidence.py`
- `scripts/build_v18_triangle_nonpenetration_evidence.py`
- `scripts/build_v18_mano_object_constraint_state.py`
- `scripts/build_v18_full_bridge_mano_object_constraint_state.py`
- `scripts/apply_v18_mano_object_constraint_state.py`
- `scripts/build_v18_compact_rigid_hidden_volume_depth_validation.py`
- render/review diagnostics for contact, occlusion, and nonpenetration where available.

Policy: contact/occlusion/nonpenetration can be uncertain, but they must remain explicit variables or unresolved state consumed by render. Diagnostics are not deliverables by themselves.

### Final state assembly and rendering

- `scripts/render_v18_full_pipeline_from_annotations.py`
- `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`
- `scripts/render_v18_joint_mano_interval_correction.py`

Policy: V20 may reuse V18 render code only if the arguments pass the current V20 state/annotation/mesh/pose files. Do not copy V18 videos as V20 output.

## V20 benchmark adapter components

### Required adapter families

The repository now includes GT-isolated benchmark and V20 sidecar tools:

```text
scripts/prepare_v20_benchmark_dataset.py
scripts/evaluate_v20_benchmark_gt.py
scripts/build_v20_depth_candidate_registry.py
scripts/select_v20_depth_observation_bundle.py
scripts/build_v20_geometry_candidate_registry.py
scripts/validate_v20_geometry_candidates.py
scripts/solve_v20_hand_shape_track.py
scripts/build_v20_contact_point_render_rows.py
scripts/build_v20_observation_bundle.py
scripts/assemble_v20_state_from_v19_annotations.py
```

Implemented in these tools:

1. `ycb`/`dexycb`/`dex-ycb -> V20 prediction dataset_manifest` adapter with GT excluded from prediction inputs.
2. `ho3d -> V20 prediction dataset_manifest` adapter with GT excluded from prediction inputs.
3. dataset RGB/depth/calibration/public object model path validation and manifest recording under `input/`; public object model rosters are candidate libraries only, never annotation targets.
4. dataset GT reference recording only under `evaluation/reference_manifest.json` for post-prediction evaluation.
5. prediction-side depth modality registry and residual-driven selector/evaluator.
6. prediction-side geometry candidate registry, agent conditioning packets, validation, and promotion.
7. prediction-side per-hand-track MANO betas/scale posterior report.
8. render-only contact point rows with `evidence_created=false`.
9. V20 observation bundle and state assembly that reject GT/oracle markers; final assembly can require branch optimization reports so sidecars cannot bypass V19-style optimization.
10. GT evaluator that refuses to evaluate prediction states containing GT/oracle markers.

The deprecated `scripts/run_v20_benchmark_oracle_bootstrap.py` is disabled and is not part of the normal V20 benchmark or infer path.

### Observed local DexYCB structure

Observed root:

```text
/mnt/nas/dex-ycb
```

Observed representative sample:

```text
/mnt/nas/dex-ycb/20200813-subject-02/20200813_151041/932122062010
```

Observed files:

- sequence-level `meta.yml`;
- sequence-level `pose.npz` with `pose_m`, `pose_y`;
- camera-level `color_%06d.jpg`;
- camera-level `aligned_depth_to_color_%06d.png`;
- camera-level `labels_%06d.npz` with `seg`, `pose_y`, `pose_m`, `joint_3d`, `joint_2d`.

### Observed local HO3D structure

Observed root:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D
```

Observed representative samples:

```text
/mnt/nas/ho3d/HO3D_dataset/HO3D/train/MC1
/mnt/nas/ho3d/HO3D_dataset/HO3D/train/BB10
```

Observed files:

- `rgb/%04d.jpg`;
- `depth/%04d.png`;
- `meta/%04d.pkl`;
- metadata keys including `camMat`, `handPose`, `handTrans`, `handBeta`, `handJoints3D`, `objRot`, `objTrans`, `objCorners3D`, `objCorners3DRest`, `objName`, `objLabel`.

## Benchmark evaluation semantics

The GT evaluation adapter must compare equivalent physical quantities:

- camera-frame hand joints/vertices against camera-frame GT;
- object pose after object-frame/unit/symmetry semantics are aligned;
- relative motion or aligned trajectory when absolute world coordinates differ;
- contact/nonpenetration only when GT or geometry supports the claim;
- visibility-stratified metrics when visibility/occlusion labels exist.

A diagnostic proxy must be labeled diagnostic and cannot ground an acceptance claim by itself.

## Immediate V20 gaps

- Automatic target-object discovery and point-prompt selection for arbitrary videos; current V20 infer closure requires an object plan grounded in visual/task evidence.
- Complete object mesh reconstruction/pose for arbitrary objects; the minimal V20 infer path renders visible-surface-only geometry, while rigid pose/contact/nonpenetration still require completed/adapted meshes and stronger branch optimization.
- Metric stereo/multiview depth without calibration; `scripts/build_v20_uncalibrated_stereo_disparity.py` produces weak non-metric evidence only.
- End-to-end remote execution wiring for learned hand/object/mask/depth models inside `/v20_infer`; the V20 tools consume real outputs and the agent may run models on the declared server target, but the slash command is not a single turnkey runner.
- Model-driven controller loop around `configs/v20_gt_evaluator_system_prompt.md` for multi-iteration benchmark refinement.
- Conditioned geometry generation backend output on server/A800; the registry now requires a real mesh output and fails rather than faking a missing implementation.

These gaps are implementation targets, not blockers that justify fake output.
