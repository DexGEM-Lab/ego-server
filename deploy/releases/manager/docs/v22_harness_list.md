# V22 Harness List

Status: V22 active harness module list. This file is the V22 list source used by `configs/v22_agent_system_prompt.md`, `.pi/prompts/v22-run.md`, `docs/v22_run_contract.md`, and `docs/v22_english_orchestration.md`.

Scope:

- V22 is the active target.
- V19/V18 remain the physical annotation spine.
- V20 documentation is legacy-only under `docs/legacy/v20/`.
- V20-era scripts may be used only as explicit legacy adapters, evaluators, or registries. They are not current main-chain entry points.
- Existing V21-named scripts are implementation tools for V22 until V22-named wrappers exist. Their outputs must be adapted into V22 state and render names.
- The active object segmentation chain is agent object plan -> agent keyframes -> OWLv2 proposals -> approved bbox prompts -> SAM2 proper masks -> contamination review. Do not substitute an older point-prompt, local-mask, or disabled bbox path as the V22 main chain.

## M00 V22 Entry, Harness Contract, And Run Root

- `M00_S01_verify_v22_harness_files`: CPU.
- `M00_S02_prepare_v22_infer_run`: CPU. Use `scripts/prepare_v21_infer_run.py` until a V22 wrapper exists.
- `M00_S03_prepare_v22_benchmark_dataset`: CPU. Use `scripts/prepare_v21_benchmark_dataset.py` until a V22 wrapper exists.
- `M00_S04_legacy_bootstrap_if_needed`: CPU. `scripts/run_v19_v20_infer_bootstrap.py`; only for raw-frame/run-root bootstrap when V22/V21 setup cannot represent the input. It is not V22 closure.
- `M00_S05_create_unresolved_initial_state`: CPU. Create or verify `state/v22_physical_state.json`, `state/v22_uncertainty_state.json`, `state/v22_observation_bundle.json`, and `logs/harness_events.jsonl`.
- `M00_S06_compute_target_and_budget_record`: CPU. Record compute target, runtime budget, benchmark iteration budget, and bottleneck tuning budget before heavy work. Default A800 target: `ssh -p 57938 zjh@115.190.235.210` unless an alternate authorized compute target was explicitly declared.
- `agent_runtime_contract_judgment`: agent/CPU. Decide whether the requested work is `v22_infer`, `v22_benchmark`, or a narrower diagnostic, and reject run roots that would overwrite existing state.

M00 establishes the harness contract. It has no physical annotation claim.

## M01 Input, Timeline, And Frame Backbone

- `M01_S01_input_manifest_contract_check`: CPU.
- `M01_S02_rebuild_v22_raw_frame_manifest_from_input`: CPU. Use `scripts/rebuild_v21_raw_frame_manifest_from_input.py` until a V22 wrapper exists.
- `M01_S03_build_v22_source_frame_manifest`: CPU. Use `scripts/build_v21_source_frame_manifest.py` until a V22 wrapper exists.
- `M01_S04_build_v19_raw_frame_manifest`: CPU. `scripts/build_v19_raw_frame_manifest.py` when V19-compatible state requires it.
- `M01_S05_build_v20_benchmark_raw_frame_manifest`: CPU. Legacy benchmark adapter only when explicitly required.
- `M01_S06_video_metadata_probe`: CPU.
- `M01_S07_timeline_contract_check`: CPU.

Core outputs:

```text
input/input_manifest.json
input/raw_frame_manifest/manifest.json
input/source_frame_manifest/manifest.json
input/source_frame_manifest/rgb/*.jpg
```

Frame mismatch is a contract error, not a noisy measurement.

## M02 Depth And Camera Candidate Generation

- `M02_S01_depth_modality_report`: CPU. `scripts/build_v21_depth_modality_report.py`.
- `M02_S02_v19_calibration_contract`: CPU. `scripts/build_v19_calibration_contract.py`.
- `M02_S03_depthpro_full_frame_candidate`: GPU/heavy. `scripts/run_v21_depthpro_full_frame_candidate.py`.
- `M02_S04_unidepth_v22_candidate`: GPU/heavy. Use `scripts/run_v21_unidepth.py` until a V22 wrapper exists.
- `M02_S05_unidepth_legacy_full_frame`: GPU/heavy. `scripts/run_unidepth_full_frame_v3.py`.
- `M02_S06_metric_depth_source_commands`: GPU/heavy. `scripts/run_depthpro_metric_source_v3.py` and `scripts/run_unidepth_metric_source_v3.py`.
- `M02_S07_stereo_sgbm_candidate`: CPU/CPU-heavy. `scripts/run_v21_stereo_sgbm_candidate.py`; metric only when calibration, rectification, and baseline support it.
- `M02_S08_uncalibrated_stereo_disparity`: CPU. Legacy weak relative evidence only.
- `M02_S09_droid_camera_trajectory_candidate`: GPU/heavy. `scripts/run_droid_full_frame.py`.
- `M02_S10_vggt_camera_scene_candidates`: GPU/heavy, only when explicitly enabled and declared as a depth/camera candidate source.
- `M02_S11_reserved_depth_candidate_extensions`: missing/design-only unless implemented or imported through explicit side inputs.
- `agent_depth_camera_sanity_judgment`: agent/CPU. Inspect depth/camera previews, residuals, and scale behavior; distinguish weak measurements from frame/camera/unit implementation errors.

M02 produces depth/camera candidates. It does not decide object pose, contact, or nonpenetration.

## M03 Depth Registry, Comparison, Selection, And Tuning

- `M03_S01_v20_depth_candidate_registry`: CPU. Legacy registry adapter, `scripts/build_v20_depth_candidate_registry.py`.
- `M03_S02_v20_depth_observation_bundle`: CPU. Legacy adapter, `scripts/select_v20_depth_observation_bundle.py`.
- `M03_S03_v22_depth_camera_bundle`: CPU. Use `scripts/select_v21_depth_camera_bundle.py` until a V22 wrapper exists.
- `M03_S04_stereo_relative_depth_comparison`: CPU. `scripts/compare_v21_depth_against_stereo_relative.py`.
- `M03_S05_monocular_vs_assisted_depth_comparison`: required when native depth, RGB-D, stereo, multiview, or assisted segmentation is used. Missing/design-only if no current report exists for the selected sample.
- `M03_S06_depth_camera_tuning_record`: agent/CPU. Write `tuning/depth_camera/<candidate_id>/attempt_<k>.json`.
- `M03_S07_depth_overlay_and_qc`: CPU.
- `agent_depth_tuning_decision`: agent/CPU. Predict before tuning, compare residuals, and select/downweight candidates only after strong tuning or a named missing implementation.

Depth selection is evidence with uncertainty. It is not a user-facing physical annotation deliverable by itself.

## M04 Hand Candidate Streams

- `M04_S01_rtmlib_hand2d`: GPU/heavy. `scripts/run_rtmlib_hand2d_v3.py`.
- `M04_S02_wilor_v22_hand_candidates`: GPU/heavy. Use `scripts/run_v21_wilor_hand_candidates.py` until a V22 wrapper exists.
- `M04_S03_wilor_legacy_full_frame`: GPU/heavy. `scripts/run_wilor_full_frame.py` and `scripts/run_wilor_maskbox_hand_stream_v7.py`.
- `M04_S04_hamer_from_rtmlib_boxes`: GPU/heavy. `scripts/run_hamer_rtmlib_hand_stream_v3.py`.
- `M04_S05_hawor_world_candidate`: GPU/heavy. `scripts/export_hawor_world.py`.
- `M04_S06_hand_mask_prompt_support`: CPU/GPU depending backend.
- `M04_S07_merge_hand_candidate_streams`: CPU. `scripts/merge_hand_candidate_streams_v7.py`.
- `M04_S08_compare_hand_streams`: CPU.
- `M04_S09_hand_candidate_overlay`: CPU/GPU.
- `agent_hand_candidate_sanity_judgment`: agent/CPU. Inspect side labels, drift, occlusion, candidate provenance, and whether a single stream is only uncertainty rather than closure.

A 2D track or single unvalidated MANO stream is not accepted V22 hand state.

## M05 Metric MANO Diagnosis, Refit, And Optimization

- `M05_S01_v22_hand_candidate_diagnosis`: CPU. Use `scripts/diagnose_v21_hand_candidate_inputs.py` until a V22 wrapper exists.
- `M05_S02_v22_mano_metric_refit`: CPU/GPU. Use `scripts/run_v21_mano_metric_refit.py` until a V22 wrapper exists.
- `M05_S03_legacy_mano_metric_depth_refit`: CPU/GPU. `scripts/refit_mano_metric_depth_v3.py`, `scripts/build_v19_mano_mask_depth_refit_inputs.py`, and `scripts/apply_v19_mano_mask_depth_refit.py`.
- `M05_S04_v20_hand_shape_track_prior`: CPU. Legacy prior only, `scripts/solve_v20_hand_shape_track.py`.
- `M05_S05_v22_active_mano`: GPU/heavy. Use `scripts/solve_v21_active_mano.py` until a V22 wrapper exists.
- `M05_S06_contact_aware_mano_graph`: CPU/GPU. `scripts/optimize_contact_aware_mano_graph_v8.py`.
- `M05_S07_v18_interval_mano_trajectory`: GPU/heavy. `scripts/solve_v18_joint_mano_interval_trajectory.py`.
- `M05_S08_mano_object_constraint_inputs`: CPU.
- `M05_S09_mano_tuning_record`: agent/CPU. Write `tuning/hand_mano/<hand_track_id>/attempt_<k>.json`.
- `agent_metric_mano_acceptance_judgment`: agent/CPU. Decide whether MANO is metric camera/world state or only overlay-aligned; record unresolved shape, scale, pose, and visibility uncertainty.

V22 hand claims require metric MANO semantics, provenance, visibility, uncertainty, and renderer consumption.

## M06 Target Object Plan, Keyframes, And Bbox Evidence

- `agent_raw_video_object_review`: agent/CPU. Inspect raw frames, task context, hand proximity, side inputs, and rejected alternatives.
- `agent_write_target_object_plan`: agent/CPU. Write `measurements/object_candidates/object_plan_agent.json` and `measurements/object_candidates/object_plan_current.json`.
- `M06_S01_agent_keyframe_selection`: CPU. `scripts/select_v21_agent_keyframes_from_plan.py`; output `measurements/object_candidates/segmentation_stable_keyframes.json` with `selected_keyframes[].frame_idx`.
- `M06_S02_owlv2_bbox_proposals`: GPU/heavy. `scripts/run_v21_owlv2_bbox_proposals.py`; inputs `object_plan_current.json` and `segmentation_stable_keyframes.json`; output `measurements/object_candidates/owlv2_bbox_proposals.json`.
- `M06_S03_owlv2_bbox_approved_prompts`: CPU/agent. `scripts/approve_v21_owlv2_bbox_prompts.py`; inputs `owlv2_bbox_proposals.json` and `segmentation_stable_keyframes.json`; output `measurements/object_candidates/owlv2_bbox_approved_prompts.json`.
- `M06_S04_object_plan_validation`: CPU.
- `agent_bbox_sanity_judgment`: agent/CPU. Confirm approved boxes point to the intended physical object rather than public roster names or background artifacts.

Public dataset rosters are model libraries, not target selections.

## M07 Segmentation, Mask Tracking, And Contamination Review

- `M07_S01_sam2_proper_segmentation`: GPU/heavy. `scripts/run_v21_sam2_proper_segmentation.py`; input `measurements/object_candidates/owlv2_bbox_approved_prompts.json`; outputs `sam2_track.json`, `sam2_masks/*.png`, `segmentation_report.json`, `qc_sam2_proper.json`, `sam2_proper_overlay.mp4`, and `measurements/object_tracks/sam2_proper_summary.json` under the current run root.
- `M07_S02_v22_segmentation_contamination_review`: CPU. Use `scripts/review_v21_segmentation_contamination.py` until a V22 wrapper exists; input `sam2_proper_summary.json`; outputs `review/segmentation_sam2_proper/segmentation_contamination_review.json` and per-object review images.
- `M07_S03_v22_segmentation_state_assembly`: CPU. Use `scripts/assemble_v21_segmentation_state.py` and `scripts/assemble_v21_state.py` only as adapters into V22 state.
- `M07_S04_segmentation_overlay`: CPU/GPU.
- `M07_S05_segmentation_tuning_record`: agent/CPU. Write `tuning/segmentation/<object_id>/attempt_<k>.json`.
- `agent_mask_identity_judgment`: agent/CPU. Inspect target identity, missing visible pixels, table/background contamination, temporal drift, and depth-edge agreement.

Only accepted SAM2 proper masks may feed V22 visible geometry. Wrong-object or contaminated masks are implementation failures, not weak observations to fuse away.

## M08 Visible Metric Surfaces And Branch Decision

- `M08_S01_v22_visible_surface_from_depth`: CPU/CPU-heavy. Use `scripts/run_v21_visible_surface_from_depth.py` until a V22 wrapper exists.
- `M08_S02_v19_visible_geometry_from_sam2_depth`: CPU/CPU-heavy. `scripts/build_v19_visible_geometry_from_sam2_depth.py`.
- `M08_S03_v18_depth_fused_reconstruction`: CPU/GPU. `scripts/build_v18_depth_fused_reconstruction.py`.
- `M08_S04_visible_surface_overlay`: CPU/GPU. `scripts/render_v21_visible_surface_overlay.py`.
- `M08_S05_heightfield_dataset_export`: CPU. `scripts/export_v21_heightfield_dataset.py`.
- `M08_S06_geometry_condition_packet`: CPU. `scripts/build_v21_geometry_condition_packet.py`.
- `M08_S07_visible_surface_qc`: CPU.
- `agent_visible_surface_sanity_judgment`: agent/CPU.
- `agent_object_branch_decision`: agent/CPU. Decide rigid, articulated, deformable, support/occluder, or unresolved with falsifiers and downstream consequences.
- `M08_S08_branch_decision_validation`: CPU.

M08 consumes accepted SAM2 proper masks plus selected depth/camera. Visible surfaces are measurements, not object pose.

## M09 Object Geometry Completion And Mesh Candidate Registry

- `M09_S01_v22_observed_mesh_candidate`: CPU/CPU-heavy. Use `scripts/build_v21_mesh_candidate_from_observed.py` until a V22 wrapper exists.
- `M09_S02_reconstruct_object_mesh`: CPU/GPU. `scripts/reconstruct_object_mesh_v2.py`.
- `M09_S03_reconstruct_scaled_observed_mesh`: CPU/GPU. `scripts/reconstruct_scaled_observed_object_mesh_v3.py`.
- `M09_S04_visual_hull_depth_carve`: CPU/GPU. `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`.
- `M09_S05_heightfield_completion`: CPU. `scripts/complete_object_heightfield_from_mask_depth_v3.py`.
- `M09_S06_rigid_evidence_bundle`: CPU. `scripts/build_v18_compact_rigid_evidence_bundle.py`.
- `M09_S07_trellis_mesh_prior`: GPU/heavy remote. `scripts/remote_run_trellis_shape_v3.py`.
- `M09_S08_trellis_completion_adapters`: CPU.
- `M09_S09_public_cad_fit`: CPU/GPU. Legacy benchmark input only.
- `M09_S10_optional_shape_candidate_runners`: GPU/heavy remote. Optional only when explicitly enabled for mesh-candidate evidence and reviewed against mask/depth/free-space residuals.
- `M09_S11_geometry_candidate_registry`: CPU. Legacy adapter, `scripts/build_v20_geometry_candidate_registry.py`.
- `M09_S12_geometry_candidate_validation`: CPU. Legacy adapter, `scripts/validate_v20_geometry_candidates.py`.
- `M09_S13_geometry_candidate_review_renders`: CPU/GPU.
- `agent_mesh_candidate_selection_judgment`: agent/CPU. Inspect mesh identity, scale, silhouette/depth/free-space residuals, and decide which candidate may enter pose fitting.

Generated, retrieved, or observed meshes are candidates until fitted, validated, optimized, and rendered.

## M10 Rigid Object Pose Fit And Factor Graphs

- `M10_S01_v22_rigid_pose_estimate`: CPU. Use `scripts/solve_v21_rigid_pose_estimate.py` until a V22 wrapper exists; not closure by itself.
- `M10_S02_v22_rigid_pose_fit`: CPU. Use `scripts/solve_v21_rigid_pose_fit.py` until a V22 wrapper exists.
- `M10_S03_v22_robust_pose`: CPU. Use `scripts/solve_v21_robust_pose.py` until a V22 wrapper exists.
- `M10_S04_v22_temporal_pose_graph`: CPU. Use `scripts/solve_v21_temporal_pose_graph.py` until a V22 wrapper exists.
- `M10_S05_v18_compact_rigid_object_pose_fit`: CPU. `scripts/fit_v18_compact_rigid_object_pose.py`.
- `M10_S06_v19_rigid_object_pose_graph`: CPU. `scripts/solve_v19_rigid_object_pose_graph.py`.
- `M10_S07_object_factor_graph`: CPU. `scripts/optimize_object_factor_graph_v3.py`.
- `M10_S08_mesh_prior_pose_graph`: CPU. `scripts/optimize_mesh_prior_pose_graph_v3.py`.
- `M10_S09_joint_camera_object_graph`: CPU. `scripts/optimize_joint_camera_object_graph_v3.py`.
- `M10_S10_joint_mano_object_graph`: CPU. `scripts/optimize_joint_mano_object_graph_v3.py`.
- `M10_S11_contact_patch_object_pose_graph`: CPU. `scripts/optimize_contact_patch_object_pose_graph_v3.py`.
- `M10_S12_additional_pose_optimizers`: CPU, only when their required inputs exist.
- `M10_S13_v20_temporal_observation_graph`: CPU. Legacy smoother only; not a full physical factor graph.
- `M10_S14_pose_graph_qc`: CPU.
- `agent_rigid_pose_acceptance_judgment`: agent/CPU. Decide whether object pose is completed/adapted mesh pose or explicit unresolved state.

A centroid or visible-surface transform is not V22 object pose closure.

## M11 Contact, Occlusion, Visibility, And Nonpenetration

- `M11_S01_v22_contact_report_with_mano`: CPU. Use `scripts/build_v21_contact_report_with_mano.py` until a V22 wrapper exists.
- `M11_S02_v22_contact_occlusion_nonpenetration`: CPU. Use `scripts/build_v21_contact_occlusion_nonpenetration.py` until a V22 wrapper exists.
- `M11_S03_v19_visible_contact_ownership_factor`: CPU. `scripts/build_v19_visible_contact_ownership_factor.py`.
- `M11_S04_v18_mesh_contact_evidence`: CPU. `scripts/build_v18_mesh_contact_evidence.py`.
- `M11_S05_v18_contact_ownership_graph`: CPU. `scripts/build_v18_contact_ownership_graph.py`.
- `M11_S06_v18_occlusion_owner_graph`: CPU. `scripts/build_v18_occlusion_owner_graph.py`.
- `M11_S07_v18_occlusion_depth_and_mesh_owner_evidence`: CPU.
- `M11_S08_v18_visibility_state`: CPU.
- `M11_S09_v18_signed_nonpenetration`: CPU. `scripts/build_v18_signed_nonpenetration_evidence.py`.
- `M11_S10_v18_triangle_nonpenetration`: CPU. `scripts/build_v18_triangle_nonpenetration_evidence.py`.
- `M11_S11_hidden_volume_depth_validation`: CPU.
- `M11_S12_mano_object_constraint_state`: CPU.
- `M11_S13_v20_contact_point_rows`: CPU. Legacy render-only rows, not contact evidence.
- `M11_S14_interaction_review_renders`: CPU/GPU.
- `agent_contact_occlusion_nonpenetration_judgment`: agent/CPU. Inspect metric distances, depth order, penetration, visibility, and uncertainty; localize failures to hand, object, camera/depth, or mask identity.

Contact, occlusion, and nonpenetration require metric MANO and object mesh pose support. If either is weak, write unresolved states rather than render-only labels.

## M12 State Assembly, V18/V19 Compatibility, And Renderer Boundary

- `M12_S01_v19_base_annotations`: CPU. `scripts/build_v19_base_annotations.py`.
- `M12_S02_v20_base_annotation_adapters`: CPU. Legacy adapters only.
- `M12_S03_v20_observation_bundle`: CPU. Legacy adapter.
- `M12_S04_v20_state_from_v19_annotations`: CPU. Legacy adapter.
- `M12_S05_v22_to_v18_annotations`: CPU. Use `scripts/assemble_v21_to_v18_annotations.py` until a V22 wrapper exists.
- `M12_S06_v22_to_v18_layout_bridge`: CPU. Use `scripts/bridge_v21_to_v18_layout.py` until a V22 wrapper exists.
- `M12_S07_v22_state_assembly`: CPU. Use `scripts/assemble_v21_state.py` only as an adapter into V22 state names.
- `M12_S08_v22_uncertainty_and_evidence_state`: CPU/agent.
- `M12_S09_state_contract_check`: CPU.
- `agent_write_state_provenance_and_uncertainty`: agent/CPU.

Core state outputs:

```text
state/annotations_v22_renderable.json
state/v22_physical_state.json
state/v22_uncertainty_state.json
state/v22_agent_evidence.md
```

State is the renderer boundary. Private measurement files do not define final visible annotations unless represented in `state/`.

## M13 Full-Duration Rendering, Publish, And Visual Review

- `M13_S01_v22_segmentation_overlay`: CPU/GPU. Use `scripts/render_v21_segmentation_overlay.py` until a V22 wrapper exists.
- `M13_S02_v22_visible_surface_overlay`: CPU/GPU. Use `scripts/render_v21_visible_surface_overlay.py` until a V22 wrapper exists.
- `M13_S03_v22_hand_overlay`: CPU/GPU. Use `scripts/render_v21_hand_overlay.py` until a V22 wrapper exists.
- `M13_S04_v22_integrated_overlay`: CPU/GPU. Use `scripts/render_v21_integrated_overlay.py` until a V22 wrapper exists.
- `M13_S05_v22_full_annotation_render`: CPU/GPU. Use `scripts/render_v21_full_annotation.py` until a V22 wrapper exists. Required outputs: `renders/v22_overlay.mp4`, `renders/v22_world.mp4`, `renders/v22_side_by_side.mp4`.
- `M13_S06_v22_v18_compositor`: CPU/GPU. Use `scripts/render_v21_v18_compositor.py` until a V22 wrapper exists.
- `M13_S07_v22_dual_pane`: CPU/GPU. Use `scripts/render_v21_dual_pane.py` until a V22 wrapper exists.
- `M13_S08_v18_full_pipeline_renderer`: CPU/GPU. Reusable only after explicit V22 state adaptation.
- `M13_S09_v18_world_and_side_by_side_renderers`: CPU/GPU.
- `M13_S10_v20_benchmark_annotation_renderer`: CPU/GPU. Legacy diagnostic only.
- `M13_S11_v19_review_renderers`: CPU/GPU.
- `M13_S12_v19_publish_artifact`: CPU. V19-compatible publication only.
- `M13_S13_render_file_contract_check`: CPU.
- `agent_consume_rendered_videos`: agent/CPU.
- `agent_submit_repair_requests`: agent/CPU. Route failures back to the earliest causal module.

A video file is success only when its visible marks are driven by the named physical state.

## M14 Benchmark Evaluation And Iteration Controller

- `M14_S01_v22_benchmark_dataset_preparation`: CPU. Use `scripts/prepare_v21_benchmark_dataset.py` until a V22 wrapper exists.
- `M14_S02_v20_benchmark_dataset_preparation`: CPU. Legacy adapter only if explicitly adapted.
- `M14_S03_v20_benchmark_video_and_prompts`: CPU. Legacy adapter only if explicitly adapted.
- `M14_S04_v20_benchmark_raw_frame_manifest`: CPU. Legacy adapter.
- `M14_S05_gt_evaluation`: CPU. `scripts/evaluate_v20_benchmark_gt.py`, only after prediction-side state and renders exist and only if V22 fields are compatible or adapted. GT must not leak into prediction state.
- `M14_S06_hot3d_legacy_adapters_and_evaluation`: CPU.
- `M14_S07_v22_benchmark_recalibration`: CPU/GPU. Use `scripts/run_v21_benchmark_recalibrate.py` until a V22 wrapper exists.
- `M14_S08_v22_benchmark_mano_injection`: CPU/GPU. Use `scripts/run_v21_benchmark_mano_injection.py` until a V22 wrapper exists; must not leak GT into prediction state.
- `M14_S09_benchmark_iteration_records`: agent/CPU.
- `M14_S10_gt_leakage_guard`: CPU/agent.
- `agent_failure_cluster_and_atomic_intervention_judgment`: agent/CPU.

Benchmark evaluation is post-prediction evidence. It must not select targets, masks, depth, pose, or tuning parameters before prediction-side render exists.

## M15 Atomic Algorithm Overlay, QC, And Materialization

- `M15_S01_v22_atomic_algorithm_suite`: CPU orchestrator with optional heavy stages. Use `scripts/run_v21_atomic_algorithm_suite.py` until a V22 wrapper exists.
- `M15_S02_v22_selected_batch_helper`: shell helper. Use `scripts/run_v21_complete_pipeline.sh` until a V22 wrapper exists. It is a tool, not the Pi harness.
- `M15_S03_model_job_planning`: CPU. `scripts/plan_v21_model_jobs.py`.
- `M15_S04_generic_algorithm_overlay`: CPU/GPU. `scripts/generate_algorithm_overlay.py`.
- `M15_S05_atomic_overlay_audit`: CPU. `scripts/audit_v21_atomic_algorithm_overlays.py`.
- `M15_S06_atomic_overlay_qc_writer`: CPU. `scripts/write_v21_atomic_overlay_qc.py`.
- `M15_S07_materialize_atomic_results`: CPU. `scripts/materialize_v21_atomic_algorithm_results.py`.
- `M15_S08_atomic_data_overlay_qc_contract_check`: CPU.
- `agent_atomic_overlay_visual_review`: agent/CPU.

Atomic overlays preserve evidence. They do not replace integrated V22 state or final render.

## M16 Batch And Parallel Control Surface

- `M16_S01_batch_manifest_builder`: missing/design-only.
- `M16_S02_parallel_submit_request`: missing/design-only.
- `M16_S03_parallel_scheduler`: missing/design-only.
- `M16_S04_parallel_resource_report`: missing/design-only.
- `M16_S05_parallel_job_status`: missing/design-only.
- `M16_S06_parallel_batch_index`: missing/design-only.
- `M16_S07_single_tmux_session_policy`: agent/CPU.
- `agent_parallel_mode_go_no_go_judgment`: agent/CPU.

Batch/parallel control may schedule work. It may not decide object identity, pose acceptance, contact, occlusion, or annotation completion.

## One-Line V22 Flow

```text
entry/run-root
-> raw/source timeline
-> depth/camera candidates + selected depth/camera
-> hand candidates + metric MANO path
-> agent object plan
-> agent OWLv2 keyframes
-> OWLv2 bbox proposals
-> approved bbox prompts
-> SAM2 proper masks
-> contamination review
-> visible metric surfaces
-> branch decision
-> mesh candidates
-> rigid pose fit/graph
-> contact/occlusion/nonpenetration
-> V22 state assembly
-> full-duration overlay/world/side-by-side render
-> visual review / benchmark iteration / repair routing
```
