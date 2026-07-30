# Parallel Annotation Draft For Current V21 Repository

Status: module/submodule inventory. This document mirrors the module/submodule list format used by the sibling feat-parallel repository, but it is adapted to this repository's current V21 codebase. It is not a delivery claim and it is not a replacement for `docs/v21_english_orchestration.md`.

Source format: `../ego-annation-feat-parallel/docs/parallel_annotation_draft.md`, section `Current V19 Module And Submodule Cut Points`.

Authoritative current repo sources:

- `docs/pipeline_v21_design.md`
- `docs/v21_run_contract.md`
- `docs/v21_component_extraction.md`
- `docs/v21_english_orchestration.md`
- `docs/v21_algorithm_research_and_implementation_plan.md`
- `docs/v21_atomic_tuning_plan.md`
- `docs/v21_completeness_audit.md`
- `scripts/run_v21_atomic_algorithm_suite.py`
- `scripts/run_v21_complete_pipeline.sh`
- current `scripts/` inventory

## Inventory Rules

1. The active target is V21. V19 remains the physical-state spine. V20 documents live under `docs/legacy/v20/` and are historical reference only; individual V20 scripts may be explicitly adapted, but V20 closure is not a current inventory entry point.
2. A module is a functional stage such as harness setup, timeline, depth, MANO, segmentation, geometry, pose graph, contact, rendering, review, or benchmark evaluation.
3. A numbered `Mxx_Sxx` submodule is a `runner_agent`-scheduled operation with a named purpose, command contract or checkpoint action, inputs, outputs, and resource profile.
4. Review and tuning checkpoints are also runner submodules. If no shell script exists, the command contract is `runner_agent_checkpoint:<checkpoint_name>` and the output is a named review, tuning, or evidence artifact written after visual/geometric/log inspection.
5. Every heavy model run must execute on a declared server/A800/authorized target. Local scripts in this list are command contracts, not permission to run local GPU inference.
6. If a required V21 mechanism has no current script, it is listed as `missing/design-only` rather than omitted or renamed as a weaker proxy.
7. JSON files, registries, validators, overlays, and reports are evidence containers. They count only when the underlying physical mechanism exists and the output is consumed by `state/` and rendered annotations.
8. GroundingDINO is disabled as the V21 default bbox source. Historical GroundingDINO artifacts are deprecated evidence only.
9. Visible surfaces, mask centroids, boxes, scatter plots, and render-only contact points are not object pose, metric MANO, contact, occlusion, or nonpenetration closure.

## Current V21 Module And Submodule Cut Points

The following decomposition covers the current repository chain. Names are stable planning identifiers, not final filenames.

Execution model: every numbered submodule is scheduled and executed by `runner_agent`. Module family names such as depth, hand, segmentation, geometry, or render describe the physical function; they are not execution roles. `Current command` names the shell command the runner executes. `runner_agent_checkpoint:<name>` names a runner-executed review/tuning action whose artifact is written by inspection rather than by a dedicated script.

### M00 V21 Entry, Harness Contract, And Run Root

Scheduled runner: `runner_agent`.

- `M00_S01_verify_v21_harness_files`: CPU. Check V21 docs, prompts, and configs named in `docs/v21_harness_deployment_guide.md`.
- `M00_S02_prepare_v21_infer_run`: CPU. Current command: `scripts/prepare_v21_infer_run.py` for arbitrary input-video run-root setup.
- `M00_S03_prepare_v21_benchmark_dataset`: CPU. Current command: `scripts/prepare_v21_benchmark_dataset.py` for GT-isolated V21 benchmark setup.
- `M00_S04_legacy_bootstrap_if_needed`: CPU. Reusable fallback command: `scripts/run_v19_v20_infer_bootstrap.py`, only as raw-frame/run-root bootstrap when V21-specific setup cannot represent the input.
- `M00_S05_create_unresolved_initial_state`: CPU. Create or verify unresolved `state/v21_physical_state.json`, `state/v21_uncertainty_state.json`, `state/v21_observation_bundle.json`, and `logs/harness_events.jsonl`.
- `M00_S06_compute_target_and_budget_record`: CPU. Record compute target, runtime budget, benchmark iterations, and strong-tuning budget before heavy inference.
- `M00_S07_runtime_contract_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:runtime_contract`. Decide whether the requested mode is `v21_infer`, `v21_benchmark`, or a narrower diagnostic, and reject any run root that would overwrite existing state.

This module has no physical annotation claim. It establishes the harness contract and prevents silent run-root or GT-policy errors.

### M01 Input, Timeline, And Frame Backbone

Scheduled runner: `runner_agent`.

- `M01_S01_input_manifest_contract_check`: CPU. Verify raw video, optional side inputs, dataset manifest, frame metadata, and path readability.
- `M01_S02_rebuild_v21_raw_frame_manifest_from_input`: CPU. Current command: `scripts/rebuild_v21_raw_frame_manifest_from_input.py`.
- `M01_S03_build_v21_source_frame_manifest`: CPU. Current command: `scripts/build_v21_source_frame_manifest.py`.
- `M01_S04_build_v19_raw_frame_manifest`: CPU. Reusable V19 command: `scripts/build_v19_raw_frame_manifest.py` when V19-compatible state requires it.
- `M01_S05_build_v20_benchmark_raw_frame_manifest`: CPU. Reusable benchmark command: `scripts/build_v20_benchmark_raw_frame_manifest.py`.
- `M01_S06_video_metadata_probe`: CPU. Verify frame count, FPS, duration, resolution, render width, and frame-index convention.
- `M01_S07_timeline_contract_check`: CPU. Check one source row per expected frame and confirm downstream manifests reference the same frame ids.

This module establishes the source timeline. Frame mismatch is a contract error, not noisy measurement.

### M02 Depth And Camera Candidate Generation

Scheduled runner: `runner_agent`.

- `M02_S01_depth_modality_report`: CPU. Current command: `scripts/build_v21_depth_modality_report.py`.
- `M02_S02_v19_calibration_contract`: CPU. Reusable V19 command: `scripts/build_v19_calibration_contract.py`.
- `M02_S03_depthpro_full_frame_candidate`: GPU/heavy. Current command: `scripts/run_v21_depthpro_full_frame_candidate.py`.
- `M02_S04_unidepth_v21_candidate`: GPU/heavy. Current command: `scripts/run_v21_unidepth.py`.
- `M02_S05_unidepth_legacy_full_frame`: GPU/heavy. Reusable command: `scripts/run_unidepth_full_frame_v3.py`.
- `M02_S06_metric_depth_source_commands`: GPU/heavy. Reusable commands: `scripts/run_depthpro_metric_source_v3.py` and `scripts/run_unidepth_metric_source_v3.py`.
- `M02_S07_stereo_sgbm_candidate`: CPU or CPU-heavy. Current command: `scripts/run_v21_stereo_sgbm_candidate.py`; metric only when calibration/rectification/baseline support it.
- `M02_S08_uncalibrated_stereo_disparity`: CPU. Reusable weak-evidence command: `scripts/build_v20_uncalibrated_stereo_disparity.py`.
- `M02_S09_droid_camera_trajectory_candidate`: GPU/heavy. Reusable command: `scripts/run_droid_full_frame.py`.
- `M02_S10_vggt_camera_scene_candidates`: GPU/heavy. Reusable commands: `scripts/run_vggt_native_camera_v3.py`, `scripts/run_vggt_scene_geometry_v3.py`, `scripts/run_vggt_object_geometry_v3.py`, and `scripts/run_vggt_object_predictions_v3.py`.
- `M02_S11_depth_anything_v2_candidate`: missing/design-only. `scripts/run_v21_atomic_algorithm_suite.py` records that no current `run_v21_depth_anything_v2` entrypoint exists.
- `M02_S12_metric3d_mast3r_dustr_multiview_candidates`: missing/design-only unless imported as external artifacts through side inputs.
- `M02_S13_depth_camera_sanity_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:depth_camera_sanity`. Inspect depth/camera previews, residuals, and scale behavior; distinguish weak measurements from frame/camera/unit implementation errors.

This module produces depth/camera measurements and candidates. It does not decide object pose, contact, or nonpenetration.

### M03 Depth Registry, Comparison, Selection, And Tuning

Scheduled runner: `runner_agent`.

- `M03_S01_v20_depth_candidate_registry`: CPU. Current reusable command: `scripts/build_v20_depth_candidate_registry.py`.
- `M03_S02_v20_depth_observation_bundle`: CPU. Current reusable command: `scripts/select_v20_depth_observation_bundle.py`.
- `M03_S03_v21_depth_camera_bundle`: CPU. Current command: `scripts/select_v21_depth_camera_bundle.py`.
- `M03_S04_stereo_relative_depth_comparison`: CPU. Current command: `scripts/compare_v21_depth_against_stereo_relative.py`.
- `M03_S05_monocular_vs_assisted_depth_comparison`: missing/design-only under planned name `scripts/compare_v21_depth_against_monocular.py`; current V21 selection must explicitly state whether the available reports satisfy this requirement.
- `M03_S06_depth_camera_tuning_record`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:depth_camera_tuning_record`; write `tuning/depth_camera/<candidate_id>/attempt_<k>.json`; no dedicated writer script exists.
- `M03_S07_depth_overlay_and_qc`: CPU. Use `scripts/generate_algorithm_overlay.py`, native previews, and `scripts/write_v21_atomic_overlay_qc.py` when producing per-atom audit artifacts.
- `M03_S08_depth_tuning_decision_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:depth_tuning_decision`. Predict before tuning, compare against residuals, and select/downweight candidates only after strong tuning or named missing implementation.

This module selects depth/camera evidence with uncertainty. Selection success is not a physical annotation deliverable by itself.

### M04 Hand Candidate Streams

Scheduled runner: `runner_agent`.

- `M04_S01_rtmlib_hand2d`: GPU/heavy. Current reusable command: `scripts/run_rtmlib_hand2d_v3.py`.
- `M04_S02_wilor_v21_hand_candidates`: GPU/heavy. Current command: `scripts/run_v21_wilor_hand_candidates.py`.
- `M04_S03_wilor_legacy_full_frame`: GPU/heavy. Reusable commands: `scripts/run_wilor_full_frame.py` and `scripts/run_wilor_maskbox_hand_stream_v7.py`.
- `M04_S04_hamer_from_rtmlib_boxes`: GPU/heavy. Reusable command: `scripts/run_hamer_rtmlib_hand_stream_v3.py`.
- `M04_S05_hawor_world_candidate`: GPU/heavy. Reusable command: `scripts/export_hawor_world.py`; adapters include `scripts/adapt_hawor_camspace_to_v20_mano_npz.py`, `scripts/adapt_hawor_camera_local_v3.py`, and `scripts/adapt_hawor_to_annotations_v3.py`.
- `M04_S06_hand_mask_prompt_support`: CPU/GPU depending backend. Reusable commands include `scripts/build_rtmlib_sam2_hand_prompts_v7.py`, `scripts/build_v19_hawor_hand_sam2_prompts.py`, `scripts/build_hand_mask_box_evidence_v3.py`, and `scripts/remap_sam2_hand_track_to_source_frames_v7.py`.
- `M04_S07_merge_hand_candidate_streams`: CPU. Reusable command: `scripts/merge_hand_candidate_streams_v7.py`.
- `M04_S08_compare_hand_streams`: CPU. Reusable command: `scripts/compare_hand_streams_scale055_v3.py` and related hand diagnostics.
- `M04_S09_hand_candidate_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_hand_overlay.py` or per-atom `scripts/generate_algorithm_overlay.py`.
- `M04_S10_hand_candidate_sanity_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:hand_candidate_sanity`. Inspect side labels, drift, occlusion, candidate provenance, and whether single-stream evidence is enough only as uncertainty, not closure.

This module produces candidate hand evidence. A 2D track or single unvalidated MANO stream is not accepted V21 hand state.

### M05 Metric MANO Diagnosis, Refit, And Optimization

Scheduled runner: `runner_agent`.

- `M05_S01_v21_hand_candidate_diagnosis`: CPU. Current command: `scripts/diagnose_v21_hand_candidate_inputs.py`.
- `M05_S02_v21_mano_metric_refit`: CPU/GPU depending implementation. Current command: `scripts/run_v21_mano_metric_refit.py`.
- `M05_S03_legacy_mano_metric_depth_refit`: CPU/GPU depending implementation. Reusable commands: `scripts/refit_mano_metric_depth_v3.py`, `scripts/build_v19_mano_mask_depth_refit_inputs.py`, `scripts/apply_v19_mano_mask_depth_refit.py`, `scripts/apply_mano_depth_refit_v3.py`, `scripts/refit_mano_articulation_mask_depth_v3.py`, and `scripts/refit_mano_pose_contact_v3.py`.
- `M05_S04_v20_hand_shape_track_prior`: CPU. Reusable command: `scripts/solve_v20_hand_shape_track.py`; this is a prior/posterior report, not accepted V21 shape closure.
- `M05_S05_v21_active_mano`: GPU/heavy. Current command: `scripts/solve_v21_active_mano.py`.
- `M05_S06_contact_aware_mano_graph`: CPU/GPU depending implementation. Reusable command: `scripts/optimize_contact_aware_mano_graph_v8.py`.
- `M05_S07_v18_interval_mano_trajectory`: GPU/heavy in historical runner. Reusable command: `scripts/solve_v18_joint_mano_interval_trajectory.py`.
- `M05_S08_mano_object_constraint_inputs`: CPU. Reusable commands include `scripts/build_v18_mano_foundation_state.py`, `scripts/build_v18_mano_object_constraint_state.py`, and `scripts/build_v18_full_bridge_mano_object_constraint_state.py` after object geometry exists.
- `M05_S09_mano_tuning_record`: missing dedicated writer. runner_agent command contract: `runner_agent_checkpoint:mano_tuning_record`; write `tuning/hand_mano/<hand_track_id>/attempt_<k>.json`; planned name `scripts/write_v21_mano_tuning_record.py` is not present.
- `M05_S10_metric_mano_acceptance_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:metric_mano_acceptance`. Decide whether MANO is metric camera/world state or only overlay-aligned, and record unresolved shape/scale/pose uncertainty.

This module is the hand-state boundary. V21 hand claims require metric MANO semantics, provenance, visibility, uncertainty, and renderer consumption.

### M06 Target Object Plan, Keyframes, And Bbox Evidence

Scheduled runner: `runner_agent`.

- `M06_S01_raw_video_object_review_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:raw_video_object_review`; inspect raw frames, task context, hand proximity, side inputs, and rejected alternatives.
- `M06_S02_write_target_object_plan_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:write_target_object_plan`; write `measurements/object_candidates/object_plan_agent.json` or current equivalent from visual/task evidence.
- `M06_S03_keyframe_selection`: CPU. Current command: `scripts/select_v21_agent_keyframes_from_plan.py`; output `measurements/object_candidates/segmentation_stable_keyframes.json` with `selected_keyframes[].frame_idx`.
- `M06_S04_owlv2_bbox_proposals`: GPU/heavy. Current default bbox command: `scripts/run_v21_owlv2_bbox_proposals.py`; output `measurements/object_candidates/owlv2_bbox_proposals.json`.
- `M06_S05_owlv2_bbox_approved_prompts`: runner_agent/CPU. Current command: `scripts/approve_v21_owlv2_bbox_prompts.py`; output `measurements/object_candidates/owlv2_bbox_approved_prompts.json` with selected `bbox_xyxy` per keyframe.
- `M06_S06_groundingdino_disabled_default`: CPU. Current command `scripts/run_v21_groundingdino.py` exists only to record disabled/deprecated status; it must not feed default V21 SAM2 or geometry.
- `M06_S07_object_plan_validation`: CPU. Check object ids, active intervals, rejected alternatives, OWLv2 text prompts, selected keyframes, approved boxes, and public model-library use.
- `M06_S08_bbox_sanity_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:bbox_sanity`; confirm approved OWLv2 boxes point to the intended physical object rather than public roster names or background artifacts.

This module decides target identity, detector keyframes, and bbox prompt evidence. Public dataset rosters are model libraries, not target selections.

### M07 Segmentation, Mask Tracking, And Contamination Review

Scheduled runner: `runner_agent`.

- `M07_S01_sam2_proper_segmentation`: GPU/heavy. Current command: `scripts/run_v21_sam2_proper_segmentation.py`; input `owlv2_bbox_approved_prompts.json`; outputs `measurements/object_tracks/sam2_proper/<object_id>/sam2_track.json`, `sam2_masks/*.png`, `segmentation_report.json`, `qc_sam2_proper.json`, `sam2_proper_overlay.mp4`, and `measurements/object_tracks/sam2_proper_summary.json`.
- `M07_S02_v21_segmentation_contamination_review`: CPU. Current command: `scripts/review_v21_segmentation_contamination.py`; input `sam2_proper_summary.json`; output `review/segmentation_sam2_proper/segmentation_contamination_review.json`.
- `M07_S03_v21_segmentation_state_assembly`: CPU. Current command: `scripts/assemble_v21_segmentation_state.py`; input `segmentation_sam2_proper` review; outputs renderable annotations/state.
- `M07_S04_segmentation_overlay`: CPU/GPU depending renderer. Current source is `measurements/object_tracks/sam2_proper/<object_id>/sam2_masks` or the native `sam2_proper_overlay.mp4`.
- `M07_S05_segmentation_tuning_record`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:segmentation_tuning_record`; write `tuning/segmentation/<object_id>/attempt_<k>.json` when mask output has large deviation.
- `M07_S06_mask_identity_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:mask_identity`. Inspect target identity, missing visible pixels, table/background contamination, temporal drift, and depth-edge agreement.

Masks are image support measurements. Wrong-object or contaminated masks must not enter geometry as weak observations.

### M08 Visible Metric Surfaces And Branch Decision

Scheduled runner: `runner_agent`.

- `M08_S01_v21_visible_surface_from_depth`: CPU or CPU-heavy geometry. Current command: `scripts/run_v21_visible_surface_from_depth.py`.
- `M08_S02_v19_visible_geometry_from_sam2_depth`: CPU or CPU-heavy geometry. Reusable command: `scripts/build_v19_visible_geometry_from_sam2_depth.py`.
- `M08_S03_v18_depth_fused_reconstruction`: CPU/GPU depending inputs. Reusable command: `scripts/build_v18_depth_fused_reconstruction.py`.
- `M08_S04_visible_surface_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_visible_surface_overlay.py`.
- `M08_S05_heightfield_dataset_export`: CPU. Current command: `scripts/export_v21_heightfield_dataset.py`.
- `M08_S06_geometry_condition_packet`: CPU. Current command: `scripts/build_v21_geometry_condition_packet.py`.
- `M08_S07_visible_surface_qc`: CPU. Check mask-depth alignment, sample counts, metric units, object id, and camera/world provenance.
- `M08_S08_visible_surface_sanity_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:visible_surface_sanity`. Judge whether visible surfaces are sane measurements or reveal mask/depth/camera implementation errors.
- `M08_S09_object_branch_decision_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:object_branch_decision`. Decide rigid, articulated, deformable, support/occluder, or unresolved with falsifiers and downstream consequences.
- `M08_S10_branch_decision_validation`: CPU. Confirm rigid objects are routed to mesh completion and pose graph rather than visible-surface-only closure.

This module lifts accepted masks and selected depth into metric visible evidence. Visible surfaces remain measurements, not final object pose.

### M09 Object Geometry Completion And Mesh Candidate Registry

Scheduled runner: `runner_agent`.

- `M09_S01_v21_observed_mesh_candidate`: CPU or CPU-heavy geometry. Current command: `scripts/build_v21_mesh_candidate_from_observed.py`.
- `M09_S02_reconstruct_object_mesh`: CPU/GPU depending backend. Reusable command: `scripts/reconstruct_object_mesh_v2.py`.
- `M09_S03_reconstruct_scaled_observed_mesh`: CPU/GPU depending backend. Reusable command: `scripts/reconstruct_scaled_observed_object_mesh_v3.py`.
- `M09_S04_visual_hull_depth_carve`: CPU/GPU depending backend. Reusable command: `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`.
- `M09_S05_heightfield_completion`: CPU. Reusable command: `scripts/complete_object_heightfield_from_mask_depth_v3.py`.
- `M09_S06_rigid_evidence_bundle`: CPU. Reusable command: `scripts/build_v18_compact_rigid_evidence_bundle.py`.
- `M09_S07_trellis_mesh_prior`: GPU/heavy remote. Reusable command: `scripts/remote_run_trellis_shape_v3.py`.
- `M09_S08_trellis_completion_adapters`: CPU. Reusable commands: `scripts/build_v18_compact_rigid_trellis_completion.py` and `scripts/build_v18_scale_sane_compact_rigid_completion.py`.
- `M09_S09_public_cad_fit`: CPU/GPU depending fitting. Reusable command: `scripts/fit_v20_cad_mesh_to_visible_depth.py`; benchmark prediction input only, target-plan tied.
- `M09_S10_additional_shape_candidate_runners`: GPU/heavy remote. Optional commands include `scripts/remote_run_spar3d_shape_v7.py`, `scripts/remote_run_triposg_shape_v7.py`, `scripts/remote_run_hunyuan21_shape_v7.py`, `scripts/remote_run_hunyuan3d_shape_v3.py`, `scripts/remote_run_partcrafter_shape_v7.py`, `scripts/remote_run_pixal3d_shape_v7.py`, and `scripts/remote_run_sam3d_objects_mesh_v7.py`.
- `M09_S11_geometry_candidate_registry`: CPU. Reusable command: `scripts/build_v20_geometry_candidate_registry.py`.
- `M09_S12_geometry_candidate_validation`: CPU. Reusable command: `scripts/validate_v20_geometry_candidates.py`.
- `M09_S13_geometry_candidate_review_renders`: CPU/GPU depending renderer. Reusable commands include `scripts/render_v18_compact_rigid_evidence_candidates.py`, `scripts/render_vggt_object_mesh_review_v3.py`, `scripts/render_mesh_alignment_v3.py`, and `scripts/render_mesh_zbuffer_qc_v3.py`.
- `M09_S14_mesh_candidate_selection_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:mesh_candidate_selection`. Inspect mesh identity, scale, silhouette/depth/free-space residuals, and decide which candidate may enter pose fitting.

Generated or retrieved meshes are candidates. They are not accepted object geometry until fitted, validated, optimized, and rendered.

### M10 Rigid Object Pose Fit And Factor Graphs

Scheduled runner: `runner_agent`.

This module runs for rigid or rigid-part objects. It may emit unresolved state if the visible evidence cannot support pose.

- `M10_S01_v21_rigid_pose_estimate`: CPU. Current command: `scripts/solve_v21_rigid_pose_estimate.py`; current audit notes this can be mask-centroid based and is not by itself V21 pose closure.
- `M10_S02_v21_rigid_pose_fit`: CPU. Current command: `scripts/solve_v21_rigid_pose_fit.py`.
- `M10_S03_v21_robust_pose`: CPU. Current command: `scripts/solve_v21_robust_pose.py` when applicable.
- `M10_S04_v21_temporal_pose_graph`: CPU. Current command: `scripts/solve_v21_temporal_pose_graph.py`.
- `M10_S05_v18_compact_rigid_object_pose_fit`: CPU. Reusable command: `scripts/fit_v18_compact_rigid_object_pose.py`; dense archive variant: `scripts/fit_v18_compact_rigid_object_pose_dense_archive.py`.
- `M10_S06_v19_rigid_object_pose_graph`: CPU. Reusable command: `scripts/solve_v19_rigid_object_pose_graph.py`.
- `M10_S07_object_factor_graph`: CPU. Reusable command: `scripts/optimize_object_factor_graph_v3.py`.
- `M10_S08_mesh_prior_pose_graph`: CPU. Reusable command: `scripts/optimize_mesh_prior_pose_graph_v3.py`.
- `M10_S09_joint_camera_object_graph`: CPU. Reusable command: `scripts/optimize_joint_camera_object_graph_v3.py`.
- `M10_S10_joint_mano_object_graph`: CPU. Reusable command: `scripts/optimize_joint_mano_object_graph_v3.py`.
- `M10_S11_contact_patch_object_pose_graph`: CPU. Reusable command: `scripts/optimize_contact_patch_object_pose_graph_v3.py`.
- `M10_S12_additional_pose_optimizers`: CPU. Reusable commands include `scripts/optimize_anchor_surface_pose_graph_v3.py`, `scripts/optimize_single_frame_mesh_mask_depth_v3.py`, `scripts/optimize_shared_surface_depth_v3.py`, `scripts/optimize_joint_depth_contact_v3.py`, and `scripts/optimize_vggt_mano_depth_contact_v3.py` when their inputs exist.
- `M10_S13_v20_temporal_observation_graph`: CPU. Reusable command: `scripts/solve_v20_infer_temporal_observation_graph.py`; this is not a full V19-style physical factor graph.
- `M10_S14_pose_graph_qc`: CPU. Check pose rows, residuals, temporal smoothness, surface-fit terms, and whether adopted pose comes from a graph or weak estimate.
- `M10_S15_rigid_pose_acceptance_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:rigid_pose_acceptance`. Decide whether object pose is completed/adapted mesh pose or explicit unresolved state.

This module is the rigid object pose boundary. A centroid or visible-surface transform is not V21 object pose closure.

### M11 Contact, Occlusion, Visibility, And Nonpenetration

Scheduled runner: `runner_agent`.

- `M11_S01_v21_contact_report_with_mano`: CPU. Current command: `scripts/build_v21_contact_report_with_mano.py`.
- `M11_S02_v21_contact_occlusion_nonpenetration`: CPU. Current command: `scripts/build_v21_contact_occlusion_nonpenetration.py`.
- `M11_S03_v19_visible_contact_ownership_factor`: CPU. Reusable command: `scripts/build_v19_visible_contact_ownership_factor.py`.
- `M11_S04_v18_mesh_contact_evidence`: CPU. Reusable command: `scripts/build_v18_mesh_contact_evidence.py`.
- `M11_S05_v18_contact_ownership_graph`: CPU. Reusable command: `scripts/build_v18_contact_ownership_graph.py`.
- `M11_S06_v18_occlusion_owner_graph`: CPU. Reusable command: `scripts/build_v18_occlusion_owner_graph.py`.
- `M11_S07_v18_occlusion_depth_and_mesh_owner_evidence`: CPU. Reusable commands: `scripts/build_v18_occlusion_depth_order_evidence.py`, `scripts/build_v18_occlusion_mesh_owner_evidence.py`, and `scripts/build_v18_occlusion_owner_candidates.py`.
- `M11_S08_v18_visibility_state`: CPU. Reusable commands: `scripts/build_v18_visibility_occlusion_state.py`, `scripts/build_v18_visible_ownership_factor.py`, and `scripts/build_v18_hand_observation_visibility_factor.py`.
- `M11_S09_v18_signed_nonpenetration`: CPU. Reusable command: `scripts/build_v18_signed_nonpenetration_evidence.py`.
- `M11_S10_v18_triangle_nonpenetration`: CPU. Reusable command: `scripts/build_v18_triangle_nonpenetration_evidence.py`.
- `M11_S11_hidden_volume_depth_validation`: CPU. Reusable command: `scripts/build_v18_compact_rigid_hidden_volume_depth_validation.py`.
- `M11_S12_mano_object_constraint_state`: CPU. Reusable commands: `scripts/build_v18_mano_object_constraint_state.py`, `scripts/build_v18_full_bridge_mano_object_constraint_state.py`, and `scripts/apply_v18_mano_object_constraint_state.py`.
- `M11_S13_v20_contact_point_rows`: CPU. Reusable command: `scripts/build_v20_contact_point_render_rows.py`; render-only rows do not create contact evidence.
- `M11_S14_interaction_review_renders`: CPU/GPU depending renderer. Reusable commands include `scripts/render_v18_contact_nonpenetration_state.py`, `scripts/render_v18_mano_object_constraint_review.py`, `scripts/render_v18_occlusion_owner_acceptance_audit.py`, and `scripts/render_v18_nonpenetration_repair_proposal.py`.
- `M11_S15_contact_occlusion_nonpenetration_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:contact_occlusion_nonpenetration`. Inspect metric distances, depth order, penetration, visibility, uncertainty, and whether failures originate from hand, object, camera/depth, or mask identity.

This module requires metric MANO and object mesh pose support. If either is weak, it must write unresolved states rather than render-only contact labels.

### M12 State Assembly, V18/V19 Compatibility, And Renderer Boundary

Scheduled runner: `runner_agent`.

- `M12_S01_v19_base_annotations`: CPU. Reusable command: `scripts/build_v19_base_annotations.py`.
- `M12_S02_v20_base_annotation_adapters`: CPU. Reusable commands: `scripts/build_v20_infer_base_annotations.py`, `scripts/build_v20_benchmark_base_annotations.py`, and `scripts/build_v20_dexycb_base_annotations.py`.
- `M12_S03_v20_observation_bundle`: CPU. Reusable command: `scripts/build_v20_observation_bundle.py`.
- `M12_S04_v20_state_from_v19_annotations`: CPU. Reusable command: `scripts/assemble_v20_state_from_v19_annotations.py`.
- `M12_S05_v21_to_v18_annotations`: CPU. Current command: `scripts/assemble_v21_to_v18_annotations.py`.
- `M12_S06_v21_to_v18_layout_bridge`: CPU. Current command: `scripts/bridge_v21_to_v18_layout.py`.
- `M12_S07_v21_state_assembly`: CPU. Current command: `scripts/assemble_v21_state.py`.
- `M12_S08_v21_uncertainty_and_evidence_state`: runner_agent/CPU. Ensure `state/v21_uncertainty_state.json` and `state/v21_agent_evidence.md` reflect unresolved variables, provenance, and renderer consumption.
- `M12_S09_state_contract_check`: CPU. Verify that `state/` is driven by current run-root measurements and that GT/oracle markers are absent from prediction state.
- `M12_S10_state_provenance_uncertainty_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:state_provenance_uncertainty`; record observation, interpretation, commitment, tuning attempts, and unresolved physical variables.

This module is the renderer boundary. Private measurement files, registries, and reports do not define final visible annotations unless represented in `state/`.

### M13 Full-Duration Rendering, Publish, And Visual Review

Scheduled runner: `runner_agent`.

- `M13_S01_v21_segmentation_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_segmentation_overlay.py`.
- `M13_S02_v21_visible_surface_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_visible_surface_overlay.py`.
- `M13_S03_v21_hand_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_hand_overlay.py`.
- `M13_S04_v21_integrated_overlay`: CPU/GPU depending renderer. Current command: `scripts/render_v21_integrated_overlay.py`.
- `M13_S05_v21_full_annotation_render`: CPU/GPU depending renderer. Current command: `scripts/render_v21_full_annotation.py`; required outputs are `renders/v21_overlay.mp4`, `renders/v21_world.mp4`, and `renders/v21_side_by_side.mp4`.
- `M13_S06_v21_v18_compositor`: CPU/GPU depending renderer. Current command: `scripts/render_v21_v18_compositor.py`.
- `M13_S07_v21_dual_pane`: CPU/GPU depending renderer. Current command: `scripts/render_v21_dual_pane.py`.
- `M13_S08_v18_full_pipeline_renderer`: CPU/GPU depending renderer. Reusable command: `scripts/render_v18_full_pipeline_from_annotations.py` when V21 state has been explicitly adapted.
- `M13_S09_v18_world_and_side_by_side_renderers`: CPU/GPU depending renderer. Reusable commands include `scripts/render_v18_world_status.py`, `scripts/render_v18_side_by_side.py`, `scripts/render_v18_joint_mano_interval_correction.py`, and `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`.
- `M13_S10_v20_benchmark_annotation_renderer`: CPU/GPU depending renderer. Reusable command: `scripts/render_v20_benchmark_annotations.py`; diagnostic unless it renders V21-equivalent state semantics.
- `M13_S11_v19_review_renderers`: CPU/GPU depending renderer. Reusable commands include `scripts/render_v19_hot3d_hawor_box_review.py` and `scripts/render_v19_interval_branch_comparison.py`.
- `M13_S12_v19_publish_artifact`: CPU. Reusable command: `scripts/publish_v19_render_artifact.py` only for V19-compatible publication, not V21 closure by copying.
- `M13_S13_render_file_contract_check`: CPU. Verify full duration or declared benchmark span, non-empty video, frame count, naming, and visible state consumption.
- `M13_S14_rendered_video_review_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:consume_rendered_videos`; inspect overlay/world/side-by-side and record whether masks, MANO, object mesh pose, contact, occlusion, nonpenetration, and uncertainty are visible and physically sane.
- `M13_S15_repair_request_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:submit_repair_requests`; route failures back to the earliest causal module instead of adding status overlays or manifests.

This module produces user-facing annotation videos. A video container is not success unless its visible marks are driven by the named physical mechanisms.

### M14 Benchmark Evaluation And Iteration Controller

Scheduled runner: `runner_agent`.

- `M14_S01_v21_benchmark_dataset_preparation`: CPU. Current command: `scripts/prepare_v21_benchmark_dataset.py`.
- `M14_S02_v20_benchmark_dataset_preparation`: CPU. Reusable command: `scripts/prepare_v20_benchmark_dataset.py`.
- `M14_S03_v20_benchmark_video_and_prompts`: CPU. Reusable command: `scripts/build_v20_benchmark_video_and_object_prompts.py`.
- `M14_S04_v20_benchmark_raw_frame_manifest`: CPU. Reusable command: `scripts/build_v20_benchmark_raw_frame_manifest.py`.
- `M14_S05_gt_evaluation`: CPU. Reusable command: `scripts/evaluate_v20_benchmark_gt.py`, only after prediction-side state and renders exist and only if V21 fields are compatible or adapted.
- `M14_S06_hot3d_legacy_adapters_and_evaluation`: CPU. Reusable commands: `scripts/build_v19_hot3d_clip_adapter.py`, `scripts/build_v19_hot3d_pinhole_adapter.py`, `scripts/evaluate_v19_hot3d_hawor_boxes.py`, `scripts/evaluate_v19_hot3d_hawor_mano3d.py`, `scripts/aggregate_v19_hot3d_box_evals.py`, and `scripts/aggregate_v19_hot3d_mano3d_evals.py`.
- `M14_S07_v20_oracle_bootstrap_disabled`: CPU. Current disabled historical command: `scripts/run_v20_benchmark_oracle_bootstrap.py`; it is not part of normal V21 prediction.
- `M14_S08_v21_benchmark_recalibration`: CPU/GPU depending affected stage. Current command: `scripts/run_v21_benchmark_recalibrate.py`.
- `M14_S09_v21_benchmark_mano_injection`: CPU/GPU depending affected stage. Current command: `scripts/run_v21_benchmark_mano_injection.py`; must not leak GT into prediction state.
- `M14_S10_benchmark_iteration_records`: runner_agent/CPU. Append `evaluation/benchmark_iterations.jsonl`, `evaluation/algorithm_parameter_changes.jsonl`, and `evaluation/final_selection_report.md`.
- `M14_S11_gt_leakage_guard`: runner_agent/CPU. Confirm reference paths remain under `evaluation/reference_manifest.json` and prediction artifacts contain no GT/oracle markers.
- `M14_S12_failure_cluster_atomic_intervention_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:failure_cluster_atomic_intervention`; convert metrics and visual failures into one atomic next intervention with a prediction before rerun.

Benchmark evaluation is post-prediction evidence. It must not select targets, masks, depth, pose, or tuning parameters before a prediction-side render exists.

### M15 Atomic Algorithm Overlay, QC, And Materialization

Scheduled runner: `runner_agent`.

- `M15_S01_v21_atomic_algorithm_suite`: CPU orchestrator with optional heavy stages. Current command: `scripts/run_v21_atomic_algorithm_suite.py`.
- `M15_S02_v21_selected_batch_helper`: shell helper. Current command: `scripts/run_v21_complete_pipeline.sh`; this is a tool, not the Pi harness or proof of closure.
- `M15_S03_model_job_planning`: CPU. Current command: `scripts/plan_v21_model_jobs.py`.
- `M15_S04_generic_algorithm_overlay`: CPU/GPU depending overlay. Current command: `scripts/generate_algorithm_overlay.py`.
- `M15_S05_atomic_overlay_audit`: CPU. Current command: `scripts/audit_v21_atomic_algorithm_overlays.py`.
- `M15_S06_atomic_overlay_qc_writer`: CPU. Current command: `scripts/write_v21_atomic_overlay_qc.py`.
- `M15_S07_materialize_atomic_results`: CPU. Current command: `scripts/materialize_v21_atomic_algorithm_results.py`.
- `M15_S08_atomic_data_overlay_qc_contract_check`: CPU. Check every active atom has data path, overlay path, QC record, unresolved reason, and tuning record where large deviation was observed.
- `M15_S09_atomic_overlay_visual_review_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:atomic_overlay_visual_review`; inspect algorithm overlays and decide whether each atom's output is usable evidence, weak evidence, implementation error, missing implementation, or deprecated history.

This module preserves per-algorithm evidence. It does not replace the integrated render or physical state.

### M16 Batch And Parallel Control Surface

Scheduled runner: `runner_agent`.

The sibling feat-parallel repository defines the intended parallel control plane. This current repository does not yet contain the parallel scheduler scripts from that plan, so the missing control-plane entries are listed explicitly.

- `M16_S01_batch_manifest_builder`: missing/design-only. Planned sibling name: `scripts/build_parallel_dataset_manifest.py`.
- `M16_S02_parallel_submit_request`: missing/design-only. Planned sibling name: `scripts/parallel_submit_request.py`.
- `M16_S03_parallel_scheduler`: missing/design-only. Planned sibling name: `scripts/parallel_scheduler.py`.
- `M16_S04_parallel_resource_report`: missing/design-only. Planned sibling name: `scripts/parallel_resource_report.py`.
- `M16_S05_parallel_job_status`: missing/design-only. Planned sibling name: `scripts/parallel_job_status.py`.
- `M16_S06_parallel_batch_index`: missing/design-only. Planned sibling name: `scripts/parallel_collect_batch_index.py`.
- `M16_S07_single_tmux_session_policy`: runner_agent/CPU. Use the project tmux discipline from `AGENTS.md` if long-running scheduler or heavy jobs are launched.
- `M16_S08_parallel_mode_go_no_go_checkpoint`: runner_agent/CPU. Command contract: `runner_agent_checkpoint:parallel_mode_go_no_go`; decide whether a task is single-entry V21, selected batch helper, or blocked on missing scheduler infrastructure.

This module is execution control only. It may schedule work, but it may not decide object identity, contact, occlusion, pose acceptance, or annotation completion.

## Coverage Notes

- V21-specific current scripts are covered in M00 through M15, including setup, depth, segmentation, MANO, visible surface, mesh, pose, contact, state, render, benchmark, and atomic audit scripts.
- V19/V18 physical-spine scripts are covered where they remain required for monotonic capability: metric MANO, object mesh pose graph, contact ownership, occlusion, nonpenetration, and full render.
- V20-era scripts appear only as explicitly adapted legacy tools, registries, benchmark loaders/evaluators, render-only contact rows, diagnostic renderers, disabled oracle history, or weak evidence paths. They do not define final V21 closure or raw-to-segmentation behavior.
- Deprecated/default-disabled paths are named rather than hidden: GroundingDINO, V20 oracle bootstrap, visible-surface-only closure, render-only contact points, uncalibrated stereo as metric depth, and centroid pose estimates.
- Planned but absent V21 scripts are named as missing/design-only: monocular-vs-assisted comparison under the planned name, Depth Anything V2 entrypoint, Metric3D/MASt3R/DUSt3R adapters, dedicated MANO tuning writer, and parallel scheduler scripts.

## Minimal Acceptance For This Inventory

A future run can claim it followed this inventory only if:

1. Every module M00-M15 has either executed applicable submodules or recorded explicit unresolved/missing states.
2. Heavy submodules record the declared compute target.
3. Bottleneck failures in depth, segmentation, and MANO have strong-tuning attempts or named missing implementations before downweighting.
4. Rigid object claims pass through mesh candidate, pose fit, pose graph or explicit unresolved state.
5. Contact, occlusion, and nonpenetration are computed from metric MANO and object mesh pose or rendered as unresolved uncertainty.
6. Final claims are consumed by `state/` and visible in `renders/v21_overlay.mp4`, `renders/v21_world.mp4`, and `renders/v21_side_by_side.mp4`.
