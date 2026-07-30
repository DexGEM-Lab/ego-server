# V19 Component Extraction From V18

Status: Workbench item 1 extraction artifact, revised after the fresh-video standard was clarified. This document now distinguishes two things: (1) V18 mechanisms/contracts mined for V19 and (2) V19-owned executable components that must run from a fresh input video/run root without cached V16/V17/V18 artifact dependencies. Inventory alone is not completion.

## Key finding

V18 is not a clean fresh-video pipeline. The apparent outer file `scripts/run_v18_full_pipeline.py` is a 536 KB / 8166-line god-file that mixes loaders, reducers, factor logic, object/contact/occlusion state construction, and rendering. Its CLI consumes many already-produced roots, including annotation state, V16 renders, bounded state, camera/depth correction, HaWoR bridge, visible geometry, physical schema, part surfaces, depth-fused reconstruction, contact graphs, nonpenetration evidence, occlusion graphs, and articulation candidates. It is therefore reusable as an assembler/corrector/renderer component only after upstream measurements exist; it is not the fresh raw-video entry point.

The usable V19 path is to extract component scripts and orchestrate them in English. The agent replaces VLM/API judgment only at declared visual-choice points; scripts still perform measurement, tracking, geometry, optimization, and rendering.

## Not reusable as-is for fresh-video orchestration

### `scripts/run_v18_full_pipeline.py`

Role in V18: final artifact assembler/factor/render god-file over cached inputs.

Evidence:
- 536165 bytes, 8166 lines, 168 top-level functions.
- Longest functions include `solve_v18_factor_graph` (663 lines), `build_case_annotations` (580), `attach_contact_physical_modes` (388), `contact_switch_energy` (373), `load_hawor_bridge_index` (244), and render functions.
- CLI roots are all upstream artifact roots: `--annotation-state-root`, `--v16-root`, `--bounded-root`, `--camera-depth-correction-root`, `--hawor-bridge-root`, `--visible-geometry-root`, `--physical-state-schema-root`, `--part-surfaces-root`, `--depth-fused-reconstruction-root`, `--mesh-contact-evidence-root`, `--contact-ownership-graph-root`, `--signed-nonpenetration-root`, `--triangle-nonpenetration-root`, `--occlusion-owner-graph-root`, `--articulation-root`.

Reusable pieces:
- data model conventions for final `annotations_v18_full.json`;
- loaders for measurement roots;
- factor-graph/contact/occlusion/nonpenetration logic after inputs exist;
- overlay/world/side-by-side rendering code paths;
- sanitization and monotonicity ideas.

Do not translate the whole file linearly into the V19 runbook. Extract only its dependency order and reusable functions/stage types.

### `scripts/run_v18_measured_status_pipeline.py`

Role in V18: explicit stage runner for cached-evidence-to-status artifacts.

Reusable pieces:
- stage-runner pattern: named stage, exact script command, stdout/stderr capture, partial report on failure;
- stage list as a map of V18 reducer dependencies.

Not enough for V19:
- its own `CLAIM` says upstream raw hand/object/depth model outputs remain cached and it is not fresh raw-video-to-final-pose runtime;
- status renderers are support/debug artifacts, not V19 physical deliverables.

### `scripts/run_v18_corrective_1600_pipeline.py`

Role in V18: explicit stage runner for corrective analyses and review renders.

Reusable pieces:
- simple, readable stage orchestration pattern;
- corrective modules that visualize or diagnose hand foundation, visible surfaces, rigid SE(3), occlusion ownership, contact/nonpenetration, and review sheets.

Not enough for V19:
- it is a cached V18 evidence-to-corrective-bundle runner, not raw video ingestion;
- many stages are audits/validators or review renders, not primary measurement production.

### `scripts/run_v18_interval_mano_canonical_artifact.py`

Role in V18: interval-MANO artifact assembler and renderer for two known cases.

Reusable pieces:
- how to attach interval MANO state, hidden-volume validation, observed-surface constraints, and rigid object pose/mesh into an annotation JSON;
- how to call `render_v18_compact_rigid_tomato_temporal_mano_attempt.py` with annotations, pose report, completed mesh, constraint report, temporal MANO state, hidden-volume validation.

Not enough for V19:
- hard-coded two cases and paths;
- assembles existing artifacts, does not produce measurements from a fresh video;
- preserves V18 interval uncertainty, not a general runtime branch.

### `scripts/run_v18_verified_hprime_final_artifact.py`

Role in V18: final V18 artifact merge/render/verify runner.

Reusable pieces:
- concise artifact assembly pattern: merge annotations, render from annotations, create review sheet, verify.

Not enough for V19:
- operates on existing verified annotations and constraint reports;
- verification/reporting cannot substitute for physical measurement.

## Reusable measurement and model-runner components

These are candidates for the English V19 orchestration. They should be called directly or adapted lightly only if their CLI already supports fresh-video inputs.

### Input/frame/depth/camera

- `scripts/run_unidepth_full_frame_v3.py` — UniDepth full-frame depth over frame spans. Inputs: manifest, output dir, frame start/end, optional repo/remote/local roots.
- `scripts/run_unidepth_metric_source_v3.py` — metric depth source over spans; same role family.
- `scripts/run_depthpro_metric_source_v3.py` — DepthPro metric source over spans; alternative/ablation depth source.
- `scripts/build_mask_unidepth_metric_manifest_v3.py`, `scripts/build_mask_vggt_depth_metric_manifest_v3.py` — manifests aligning masks with metric depth.
- `scripts/diagnose_metric_depth_alignment_v3.py`, `scripts/diagnose_depth_source_hand_scale_v3.py` — diagnostic, not primary progress, but useful for mechanism failures.

Extracted V19 component: `scripts/build_v19_raw_frame_manifest.py` now provides the clean `input_video -> raw_frame_manifest/manifest.json + rgb/` step under the current run root. It writes one row per source frame with `frame_idx`, `time_s`, `rgb`, `raw_frame_path`, and source/manifest dimensions. It has no V16/V17/V18 cached artifact dependency.

### Hand / MANO

- `scripts/run_rtmlib_hand2d_v3.py` — 2D hand detector stream; measurement only, not hand state.
- `scripts/run_hamer_rtmlib_hand_stream_v3.py` — HaMeR hand stream using RTMLib JSON, target annotations, frame manifest, model root/checkpoint, output annotations/QC.
- `scripts/run_wilor_full_frame.py` — WiLoR full-frame hand recovery for a clip.
- `scripts/run_wilor_maskbox_hand_stream_v7.py` — WiLoR from mask/box prompts into target annotations and QC.
- `scripts/run_handdgp_export_v3.py`, `scripts/run_omnihands_export_v3.py` — candidate hand model exports if environment/resources are present.
- `scripts/merge_hand_candidate_streams_v7.py` — combines hand candidate streams.
- `scripts/convert_handdgp_to_mano_candidates_v7.py` — converts hand model output to MANO candidates.
- `scripts/refit_mano_metric_depth_v3.py`, `scripts/refit_mano_articulation_mask_depth_v3.py`, `scripts/fit_mano_to_hand_mask_depth_v3.py` — metric/depth/mask refit tools.
- `scripts/solve_v18_joint_mano_interval_trajectory.py` — strong interval MANO solver with object/visual constraints; reusable after annotations, depth, object pose/mesh, and interval spans exist.
- `scripts/render_v18_joint_mano_interval_correction.py` — renders continuous joint MANO interval correction and object mesh context.

Policy for V19: 2D detectors are prompts/measurements only. A valid hand state must come through MANO candidate/export/refit/interval solver and carry camera/world semantics and uncertainty.

### Object discovery, masks, tracking

- `scripts/build_object_plan_vlm.py` — VLM object plan generator. For V19 runtime, replace API call with agent visual judgment over supplied frames: the agent writes the object plan in the same structure instead of calling VLM.
- `scripts/build_object_point_prompts_vlm.py` — VLM point prompt generator. For V19, replace API call with agent-chosen points from visual evidence.
- `scripts/run_sam2_object_track.py` — SAM2 object tracking from annotations/clip/checkpoint.
- `scripts/run_sam2_vlm_points_multiobject.py`, `scripts/run_sam2_vlm_points_track.py`, `scripts/run_sam2_vlm_points_image.py` — SAM2 point-prompt tracking/image mask tools. Keep scripts; replace VLM prompt source with agent-chosen prompt points.
- `scripts/remap_sam2_track_to_source_frames_v7.py` — remaps SAM2 track JSON back to source frame indexing.
- `scripts/run_cotracker_object_tracks_v5.py`, `scripts/run_v17_object_material_tracks.py` — tracking/material track tools from manifest/annotations/depth.
- `scripts/build_v18_owlv2_sam2_part_tracks.py`, `scripts/build_v18_part_track_source_manifest.py`, `scripts/build_v18_part_split_evidence.py`, `scripts/build_v18_part_visible_surfaces.py` — V18 part/track reducers. Useful after masks/tracks exist, but not fresh discovery by themselves.
- `scripts/render_point_prompt_review_v3.py`, `scripts/render_sam2_surface_mask_grid_v3.py` — visual review for prompt/track quality.

Policy for V19: object discovery must be category-agnostic. Differences between object categories belong in agent-selected prompts/masks/state, not Python branches.

### Geometry / object pose / rigid branch

- `scripts/remote_run_trellis_shape_v3.py` — remote TRELLIS mesh completion from an image crop. Reusable as the rigid hidden-surface prior stage.
- `scripts/build_v18_compact_rigid_trellis_completion.py` — V18 compact-rigid completion builder; needs inspection/use as completion+alignment stage.
- `scripts/build_v18_compact_rigid_evidence_bundle.py` — evidence bundle/crop preparation for rigid completion.
- `scripts/build_v19_raw_frame_manifest.py` — V19-owned fresh-video timeline/image extraction into `input/raw_frame_manifest/manifest.json` and `rgb/`.
- `scripts/build_v19_base_annotations.py` — V19-owned base annotation/state assembly from fresh raw manifest, camera trajectory, depth intrinsics, HaWoR MANO export, agent object plan, and SAM2 tracks. It also writes `v19_mano_bridge_from_hawor_world.npz` with the compatibility arrays required by existing MANO solvers.
- `scripts/build_v19_visible_geometry_from_sam2_depth.py` — V19 bridge from SAM2 masks plus metric depth/camera poses into V18-compatible `visible_geometry_candidate` and centroid-initialized `reconstructed_geometry_pose` annotation rows for rigid branch input. It is a measurement adapter, not final pose.
- `scripts/fit_v18_compact_rigid_object_pose.py` — visible-frame object pose fitting against depth/mask samples; core rigid pose measurement component.
- `scripts/solve_v19_rigid_object_pose_graph.py` — V19 temporal correction graph over visible-frame rigid pose observations with bounded nonpenetration pressure. It outputs corrected pose rows consumed by the rigid renderer/constraint tools; if corrections stay tiny while penetration remains broad, the next mechanism is interval MANO/contact/occlusion, not object-pose smoothing.
- `scripts/fit_v18_compact_rigid_object_pose_dense_archive.py` — dense archive pose fit variant.
- `scripts/build_v18_scale_sane_compact_rigid_completion.py` — scale-sane rigid completion/alignment; core tomato branch component.
- `scripts/reconstruct_object_mesh_v2.py`, `scripts/reconstruct_scaled_observed_object_mesh_v3.py`, `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`, `scripts/complete_object_heightfield_from_mask_depth_v3.py` — observed-geometry reconstruction/visual hull/heightfield components.
- `scripts/optimize_object_factor_graph_v3.py`, `scripts/optimize_contact_patch_object_pose_graph_v3.py`, `scripts/optimize_joint_mano_object_graph_v3.py`, `scripts/optimize_joint_camera_object_graph_v3.py` — optimization components for object/hand/camera coupling.
- `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py`, `scripts/render_mesh_alignment_v3.py`, `scripts/render_mesh_zbuffer_qc_v3.py`, `scripts/render_v18_rigid_se3_attempt.py`, `scripts/render_v18_rigid_se3_residual_check.py` — visualizers/QC for geometry/pose.

Policy for V19 rigid enforcement: if the agent decides an object is rigid, the English pipeline must force completion/adaptation -> visible-frame pose -> factor graph correction -> corrected mesh-pose render. It must not allow visible surfaces as final object pose.

### Contact / occlusion / nonpenetration / temporal graph

- `scripts/solve_v18_joint_mano_interval_trajectory.py` — interval hand/object solver with contact/visual constraints.
- `scripts/build_v18_mano_object_constraint_state.py`, `scripts/build_v18_full_bridge_mano_object_constraint_state.py` — MANO-object constraint states.
- `scripts/build_v19_visible_contact_ownership_factor.py` — V19 factor bridge from Pi-agent interval contact/occlusion judgment plus projected MANO/object masks into solver-consumed `visible_ownership` and `contact_patch` rows. It replaces hidden VLM/API contact-judgment use with an explicit run-root judgment artifact and graph numeric priors/uncertainties.
- `scripts/build_v18_contact_ownership_graph.py`, `scripts/build_v18_contact_patch_factor.py`, `scripts/build_v18_local_contact_patch_support_factor.py` — contact ownership/patch factors.
- `scripts/build_v18_mesh_contact_evidence.py` — mesh contact evidence.
- `scripts/build_v18_occlusion_owner_candidates.py`, `scripts/build_v18_occlusion_depth_order_evidence.py`, `scripts/build_v18_occlusion_mesh_owner_evidence.py`, `scripts/build_v18_occlusion_owner_graph.py`, `scripts/build_v18_occlusion_pose_fill_gate.py` — occlusion candidate/evidence/graph components.
- `scripts/build_v18_signed_nonpenetration_evidence.py`, `scripts/build_v18_triangle_nonpenetration_evidence.py` — nonpenetration evidence.
- `scripts/render_v18_contact_nonpenetration_state.py`, `scripts/render_v18_contact_acceptance_audit.py`, `scripts/render_v18_occlusion_owner_best_effort.py`, `scripts/render_v18_occlusion_owner_acceptance_audit.py`, `scripts/render_v18_nonpenetration_repair_proposal.py` — visual diagnostics/review, not final acceptance by themselves.

Policy for V19: contact/occlusion/nonpenetration can be uncertain, but they must remain explicit variables or explicit unresolved state consumed by render. Do not treat diagnostics as deliverables.

### Final state assembly and rendering

- `scripts/run_v18_full_pipeline.py` — use only after all expected measurement roots exist; contains final annotation assembly, factor graph, and render functions.
- `scripts/render_v18_full_pipeline_from_annotations.py` — renders overlay/world/side-by-side from an annotation JSON.
- `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py` — renders rigid object plus interval MANO uncertainty from annotations/pose/mesh/constraint/temporal state.
- `scripts/render_v18_joint_mano_interval_correction.py` — renders interval correction from annotations, pose report, completed mesh, joint MANO state.
- `scripts/render_world_reconstruction_v3.py`, `scripts/render_mesh_surface_contact_3d_v3.py`, `scripts/render_mesh_surface_contact_review_v3.py` — richer world/contact visualizers.

Policy for V19: renderer must be fed from V19 state built by the English pipeline. It may reuse V18 render code, but not V18 output videos as the final artifact.

## English orchestration artifact for Workbench item 2

The Workbench item 2 artifact now lives at `docs/v19_english_orchestration.md`. It is the executable English pipeline control logic, not a decorative plan: the default path starts with `build_v19_raw_frame_manifest.py`, runs fresh measurement components under the run root, assembles `annotations_v19_base.json` with `build_v19_base_annotations.py`, and only then invokes rigid/contact/MANO/render components on V19-generated inputs.

The runbook follows this structure:

1. Create or locate a frame manifest for the fresh video.
2. Run depth/camera source(s): UniDepth/DepthPro/VGGT as available; record scale uncertainty.
3. Run hand measurement stack: RTMLib -> HaMeR/WiLoR/HaWoR/HandDGP candidates -> merge/refit -> MANO state.
4. Agent inspects sampled frames and writes object plan / point prompts in the same structures expected by existing object-plan/prompt scripts, replacing VLM calls.
5. Run SAM2/object tracking from agent prompts; remap tracks; review masks.
6. Build object visible surfaces/depth archives from masks and depth.
7. Agent chooses object branch(es): rigid/articulated/deformable/support/unresolved.
8. For rigid objects: build evidence crop/bundle -> TRELLIS or equivalent mesh completion -> scale/adapt to observed surfaces -> visible-frame pose fit -> V19 temporal rigid-pose correction -> object/MANO/contact factor correction.
9. Run contact/occlusion/nonpenetration evidence modules and factor graph; preserve uncertainty.
10. Assemble state/annotations using extracted final assembler pieces.
11. Render overlay/world/side-by-side; visually inspect and repair concrete contradictions.
12. Only after representative pipeline works, freeze benchmark clips and run ablation/comparison.

## Immediate gaps exposed by extraction

- Need a clean fresh-video frame-manifest step or identify the existing script that already does it.
- `scripts/build_v19_base_annotations.py` now covers the base hand/object/camera annotation backbone. Remaining extraction work is to wrap/rename older `v18_*` rigid/contact/render scripts as V19-owned components and to ensure each consumes only V19 run-root outputs.
- Need to decide whether V19 state should become a new thin schema or continue using the V18 annotation shape as the renderable state backbone until a V19 renderer exists.
- Need to map exact command templates for the selected hand stack on A800, including env/checkpoint paths.
- Representative task5 rigid branch now runs through visible-geometry adaptation, scale-sane completion, visible pose fit, temporal rigid-pose graph, MANO/object constraints, interval MANO uncertainty render, and 690-725 contact-factor ablations. `scripts/build_v19_visible_contact_ownership_factor.py` now covers the V19 agent-judged visible contact/ownership prior path for interval MANO. Remaining extraction gaps are richer hand-owned surface/MANO refit mechanisms, cleaner V19 wrappers for older `v18_*` rigid/contact/render components, state-to-render cleanup, and benchmark adapters.
- Need to implement HOT3D/H2O-or-DexYCB benchmark adapters before quantitative external claims.
