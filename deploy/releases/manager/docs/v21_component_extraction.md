# V21 Component Extraction From V19 With Legacy V20 Adapters

Status: V21 component map. This document separates reusable existing repository tools from V21-required missing or modified mechanisms. A V21 runtime agent must not invent script names or treat design-only mechanisms as implemented.

## 1. Key finding

V21 reuses V19 as the physical annotation spine and treats V20 as legacy adapter history. Individual V20-era scripts may be inspected or explicitly adapted, but V20 closure is insufficient for the user's target physical quality where it closes with visible-surface-only objects, overlay-aligned hands, weak stereo disparity, or centroid/joint temporal smoothing.

V21 adds harness obligations rather than merely adding registries:

- strong-tune bottleneck observations before downweighting;
- compare any later native-depth/RGB-D/stereo/depth-assisted segmentation against the active OWLv2 bbox-prompt SAM2 track;
- bind tuned algorithm parameters to sample/run/input hash;
- force rigid manipulated objects through mesh completion/adaptation, pose fitting, factor correction, and mesh-pose render.

## 2. Not reusable as V21 entry points

### Top-level pipeline wrappers

Do not use broad legacy wrappers as V21 harness entry points:

- `scripts/run_v18_full_pipeline.py` consumes many prebuilt roots and is not a Pi-native V21 run.
- `scripts/run_v19_v20_infer_bootstrap.py` may help create run structure and raw-frame manifests, but it is not V21 inference closure.
- `scripts/run_v20_benchmark_oracle_bootstrap.py` is historical/oracle diagnostic only and must not be used in V21 prediction.

### V20 minimal closure as final V21 state

The following can be inspected or reused as interim adapters only. They do not satisfy V21 physical closure by themselves:

- `scripts/build_v20_infer_base_annotations.py`: approximate base annotations from masks/depth/hands.
- `scripts/solve_v20_infer_temporal_observation_graph.py`: visible-surface centroid and MANO temporal smoothing, not a full physical graph.
- `scripts/render_v20_benchmark_annotations.py`: diagnostic renderer that may label infer outputs as benchmark and renders scatter/skeleton rather than V21 metric mesh-pose world view.
- `scripts/align_v20_hawor_overlay_to_detector_boxes.py`: visualization-only overlay alignment, not metric MANO repair.
- `scripts/build_v20_uncalibrated_stereo_disparity.py`: weak nonmetric disparity evidence only.
- `scripts/build_v20_contact_point_render_rows.py`: render-only contact points with `evidence_created=false`.

## 3. Reusable components

### 3.1 Input, frame, and benchmark adapters

Existing useful tools:

- `scripts/run_v19_v20_infer_bootstrap.py` — may be reused to create raw-frame manifests and initial run roots, but V21 must write V21 state/contract files after it.
- `scripts/prepare_v20_benchmark_dataset.py` — reusable for V21 DexYCB/YCB and HO3D benchmark manifest preparation if outputs are written or adapted under V21 run roots with GT sealed.
- `scripts/build_v20_benchmark_raw_frame_manifest.py` — reusable where benchmark frame manifests are needed.
- `scripts/evaluate_v20_benchmark_gt.py` — reusable as the initial V21 GT evaluator if invoked after V21 prediction state/render exists and output names/policies are recorded as V21-compatible.

V21-specific adapter now present:

- `scripts/prepare_v21_benchmark_dataset.py` — wraps the V20 DexYCB/HO3D parser, preserves GT isolation, and writes V21 manifest schemas, unresolved V21 state, uncertainty state, observation bundle, evidence note, and harness log. It is benchmark input/state bootstrap only; it does not run prediction measurements or rendering.

V21 missing or needed adapter:

- Standalone `input_video -> raw_frame_manifest/manifest.json` command if `run_v19_v20_infer_bootstrap.py` remains too V19/V20-specific.

### 3.2 Camera, depth, and monocular baselines

Existing tools:

- `scripts/run_unidepth_full_frame_v3.py` — metric monocular depth candidate.
- `scripts/run_unidepth_metric_source_v3.py` — metric depth source over selected/masked spans.
- `scripts/run_depthpro_metric_source_v3.py` — DepthPro-style metric source where environment supports it.
- `scripts/run_droid_full_frame.py` — camera/head trajectory and geometry candidate.
- `scripts/run_vggt_native_camera_v3.py` — VGGT native camera candidate.
- `scripts/run_vggt_scene_geometry_v3.py` — scene geometry/camera evidence.
- `scripts/run_vggt_object_geometry_v3.py` — object geometry evidence.
- `scripts/run_vggt_object_predictions_v3.py` — object predictions where applicable.
- `scripts/build_v20_depth_candidate_registry.py` — reusable schema/checking tool for candidate registration.
- `scripts/select_v20_depth_observation_bundle.py` — reusable as a starting selector if extended with V21 monocular-baseline comparison and strong-tuning records.
- `scripts/build_v20_uncalibrated_stereo_disparity.py` — weak nonmetric stereo evidence only.

V21 missing or needed mechanisms:

- `depth_modality_report` builder that records native depth/RGB-D/calibrated stereo/uncalibrated stereo/multiview/monocular modes.
- Calibrated stereo adapter with calibration/rectification, disparity confidence, left-right consistency, and metric depth output.
- Intrinsics-estimation adapter when no intrinsics are provided.
- V21 depth selector that explicitly requires monocular baseline comparison before selecting native-depth/RGB-D/stereo/multiview evidence as primary.
- Tuning-record writer for depth/camera algorithm-internal parameter changes.

### 3.3 Hand / MANO

Existing tools:

- `scripts/run_rtmlib_hand2d_v3.py` — 2D hand evidence.
- `scripts/run_wilor_full_frame.py` — WiLoR MANO candidate stream.
- `scripts/run_wilor_maskbox_hand_stream_v7.py` — mask/box-conditioned WiLoR stream.
- `scripts/export_hawor_world.py` — HaWoR world/camera candidate where environment exists.
- `scripts/run_hamer_rtmlib_hand_stream_v3.py` — HaMeR from RTMLib boxes.
- `scripts/merge_hand_candidate_streams_v7.py` — merge candidate streams.
- `scripts/refit_mano_metric_depth_v3.py` — depth/scale refit for MANO candidates.
- `scripts/solve_v18_joint_mano_interval_trajectory.py` — interval MANO trajectory solve.
- `scripts/adapt_hawor_camspace_to_v20_mano_npz.py` — adapter for HaWoR cam-space outputs.
- `scripts/solve_v20_hand_shape_track.py` — track-level posterior from candidate betas/scales; useful as a prior but not accepted V21 hand shape closure.
- `scripts/optimize_contact_aware_mano_graph_v8.py` — contact-aware MANO graph if inputs are available.

V21 missing or needed mechanisms:

- Metric hand diagnosis stage for detector boxes, side mapping, crop/resize convention, camera convention, MANO global transform, and depth alignment.
- Joint optimization where track-level betas/scale are active variables together with per-frame pose/root/translation.
- Tuning-record writer for hand/MANO backend/parameter changes.
- Renderer semantics that visually distinguish metric MANO from overlay-only hand visualization.

### 3.4 Object discovery, masks, and segmentation

Existing tools:

- `scripts/select_v21_agent_keyframes_from_plan.py` — agent/object-plan keyframe selection for OWLv2 detector frames.
- `scripts/run_v21_owlv2_bbox_proposals.py` — OWLv2 keyframe bbox proposal runner.
- `scripts/approve_v21_owlv2_bbox_prompts.py` — approved target bbox prompt writer for SAM2.
- `scripts/run_v21_sam2_proper_segmentation.py` — SAM2 video propagation from approved OWLv2 bbox prompts.
- `scripts/review_v21_segmentation_contamination.py` — contamination/drift review over the active SAM2 proper summary.
- `scripts/assemble_v21_segmentation_state.py` — renderable segmentation-state assembly from accepted SAM2 proper review.

V21 missing or needed mechanisms:

- Agent-driven target discovery harness step: keyframes -> OWLv2 proposals -> approved bbox prompts -> rejected alternatives.
- Authorized heavy rerun of OWLv2 and SAM2 proper on the selected videos.
- Visual contamination review with target identity, missing-object, table/background contamination, and temporal drift flags.
- Segmentation tuning-record writer for keyframe seeds, OWLv2 thresholds, SAM2 memory/propagation, component filtering, and registration.

### 3.5 Visible surfaces, geometry completion, and object pose

Existing tools:

- `scripts/build_v19_visible_geometry_from_sam2_depth.py` — SAM2/depth visible-surface bridge.
- `scripts/build_v18_depth_fused_reconstruction.py` — depth-fused reconstruction where inputs exist.
- `scripts/reconstruct_object_mesh_v2.py` — object mesh reconstruction.
- `scripts/reconstruct_scaled_observed_object_mesh_v3.py` — scaled observed mesh reconstruction.
- `scripts/reconstruct_object_visual_hull_depth_carve_v3.py` — visual hull/depth carve.
- `scripts/complete_object_heightfield_from_mask_depth_v3.py` — heightfield completion.
- `scripts/build_v18_compact_rigid_evidence_bundle.py` — rigid evidence bundle.
- `scripts/remote_run_trellis_shape_v3.py` — TRELLIS shape generation.
- `scripts/build_v18_compact_rigid_trellis_completion.py` — TRELLIS completion adapter.
- `scripts/build_v18_scale_sane_compact_rigid_completion.py` — scale-sanity adapter.
- `scripts/fit_v18_compact_rigid_object_pose.py` — rigid object pose fit.
- `scripts/solve_v19_rigid_object_pose_graph.py` — rigid pose graph.
- `scripts/build_v20_geometry_candidate_registry.py` — geometry candidate registry with GT/oracle guards.
- `scripts/validate_v20_geometry_candidates.py` — geometry validation/promotion.
- `scripts/fit_v20_cad_mesh_to_visible_depth.py` — public CAD fit for benchmark prediction-side candidates.
- `scripts/optimize_object_factor_graph_v3.py` — object factor graph.
- `scripts/optimize_mesh_prior_pose_graph_v3.py` — mesh-prior pose graph.
- `scripts/optimize_anchor_surface_pose_graph_v3.py` — anchor-surface pose graph.
- `scripts/optimize_joint_camera_object_graph_v3.py` — joint camera/object graph.
- `scripts/optimize_joint_mano_object_graph_v3.py` — joint MANO/object graph.
- `scripts/optimize_contact_patch_object_pose_graph_v3.py` — contact-patch object pose graph.

V21 missing or needed mechanisms:

- Evidence-conditioned geometry packet builder keyed to accepted target masks and selected depth/camera candidates.
- Conditioned geometry generation adapter if TRELLIS cannot consume structured evidence; must output real mesh files and source reports.
- V21 geometry selector that promotes only candidates that improve visible depth/silhouette/free-space/scale/temporal pose residuals.
- Object pose graph wrapper that consumes completed/adapted mesh pose, camera/depth, MANO, contact, occlusion, and nonpenetration evidence and writes V21 state.

### 3.6 Contact, occlusion, and nonpenetration

Existing tools:

- `scripts/build_v18_contact_ownership_graph.py`
- `scripts/build_v18_occlusion_owner_graph.py`
- `scripts/build_v18_signed_nonpenetration_evidence.py`
- `scripts/build_v18_triangle_nonpenetration_evidence.py`
- `scripts/build_v18_mano_object_constraint_state.py`
- `scripts/build_v18_full_bridge_mano_object_constraint_state.py`
- `scripts/apply_v18_mano_object_constraint_state.py`
- `scripts/build_v18_compact_rigid_hidden_volume_depth_validation.py`
- `scripts/render_v18_contact_nonpenetration_state.py`
- `scripts/render_v18_occlusion_owner_acceptance_audit.py`

V21 policy:

- Contact/occlusion/nonpenetration should run only after object mesh pose and metric MANO are good enough to make the variables meaningful.
- If geometry or MANO is weak, these variables remain explicit unresolved states rather than render-only points or labels.

### 3.7 Final state and rendering

Existing tools:

- `scripts/render_v18_full_pipeline_from_annotations.py`
- `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`
- `scripts/render_v18_joint_mano_interval_correction.py`
- `scripts/render_v18_side_by_side.py`
- `scripts/render_v18_world_status.py`
- `scripts/render_v18_mano_object_constraint_review.py`
- `scripts/assemble_v20_state_from_v19_annotations.py` — can inform state assembly guards, but V21 should not emit V20 schemas as final state.

V21 missing or needed mechanisms:

- `state/annotations_v21_renderable.json` adapter from optimized V21 state.
- V21 renderer or V18 renderer invocation contract that consumes V21 state and produces `v21_overlay.mp4`, `v21_world.mp4`, `v21_side_by_side.mp4`.
- Renderer semantics for uncertainty, monocular-vs-assisted depth comparison, metric MANO vs overlay-only hand, and unresolved contact/occlusion/nonpenetration.

## 4. Reusable benchmark semantics

V21 can reuse the V20 benchmark loader/evaluator semantics initially:

- DexYCB/YCB and HO3D fail-fast dataset contracts.
- GT references only under `evaluation/reference_manifest.json`.
- Prediction state rejected if GT/oracle markers are present.
- Metrics reported only under compatible physical semantics.

V21 must add benchmark iteration controller records:

- failure cluster;
- mechanism hypothesis;
- single atomic intervention;
- tuned parameters and sample binding;
- prediction before rerun;
- metric/render outcome;
- stop/continue decision.

## 5. V21 implementation priorities

1. Harness docs/prompts and run directory/state contract, including `scripts/prepare_v21_benchmark_dataset.py` for GT-isolated benchmark bootstrap.
2. Depth/camera modality report, monocular baseline runner/registry, and comparison gate.
3. Segmentation object-plan, OWLv2 bbox prompts, SAM2 proper propagation, contamination review, and tuning records.
4. Metric MANO diagnosis and active betas/scale/pose optimization.
5. Rigid object geometry condition packet, mesh completion/adaptation, pose fit, and graph wrapper.
6. V21 state adapter and renderer semantics.
7. Benchmark controller loop and final selection report.

These priorities are not acceptance gates that block rendering uncertain state. They identify the mechanisms that must be implemented to claim full V21 physical quality.
