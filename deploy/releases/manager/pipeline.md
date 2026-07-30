# V21 Parallel Pipeline

This file is the `pipeline.md` required by the Feishu `Ego parallel pipeline` plan. It converts the single-entry V21 runbook into runner-visible submodule execution rules for `/v21-parallel`.

The rule is mechanical: if `execution` is `gpu_wrapper`, the runner must call `scripts/v21_gpu_wrapper.py` with the listed `estimated_vram_mb` before the submodule command. If `execution` is `cpu_direct` or `agent_judgment`, the runner does not use the GPU wrapper.

The estimates are scheduling contracts, not quality claims. They should be revised from observed A800 usage logs after real runs.

| id | stage | execution | estimated_vram_mb | command or owner | output / decision |
| --- | --- | --- | ---: | --- | --- |
| P00_S01 | claim_next_entry | cpu_direct | 0 | `scripts/v21_parallel_claim_next.py --claim` | one runner owns one data entry |
| P01_S01 | input_manifest_and_timeline | cpu_direct | 0 | `scripts/prepare_v21_infer_run.py` or `scripts/rebuild_v21_raw_frame_manifest_from_input.py` | run root and raw-frame manifest |
| P02_S01 | depth_modality_report | cpu_direct | 0 | `scripts/build_v21_depth_modality_report.py` or direct manifest inspection | modality report |
| P03_S01 | monocular_unidepth_baseline | gpu_wrapper | 22000 | `scripts/run_v21_unidepth.py` or `scripts/run_unidepth_full_frame_v3.py` | RGB-only depth baseline |
| P03_S02 | droid_or_vggt_camera_candidate | gpu_wrapper | 18000 | existing DROID/VGGT camera tool when selected | camera trajectory candidate |
| P04_S01 | depth_candidate_registry | cpu_direct | 0 | `scripts/build_v20_depth_candidate_registry.py` where available | depth candidate registry |
| P04_S02 | assisted_depth_or_stereo_candidate | gpu_wrapper | 16000 | `scripts/run_v21_depthpro_full_frame_candidate.py`, `scripts/run_v21_stereo_sgbm_candidate.py`, or selected backend | assisted depth candidate |
| P04_S03 | depth_selection_and_tuning_record | cpu_direct | 0 | `scripts/select_v21_depth_camera_bundle.py` or V21 tuning record | selected depth/camera bundle or missing selector |
| P05_S01 | rtmlib_hand_2d | gpu_wrapper | 6000 | `scripts/run_rtmlib_hand2d_v3.py` | 2D hand evidence |
| P05_S02 | wilor_mano_candidates | gpu_wrapper | 14000 | `scripts/run_v21_wilor_hand_candidates.py` or `scripts/run_wilor_full_frame.py` | MANO candidate stream |
| P05_S03 | hamer_or_hawor_candidates | gpu_wrapper | 18000 | `scripts/run_hamer_rtmlib_hand_stream_v3.py` or `scripts/export_hawor_world.py` | alternate MANO stream |
| P05_S04 | hand_merge_refit_tuning | cpu_direct | 0 | `scripts/merge_hand_candidate_streams_v7.py`, `scripts/refit_mano_metric_depth_v3.py`, `scripts/solve_v21_active_mano.py` when CPU; wrap if configured GPU | metric MANO candidate/uncertainty |
| P06_S01 | object_plan | agent_judgment | 0 | runner agent writes `object_plan_agent.json` | target object roster and branch hypotheses |
| P06_S02 | keyframe_selection | cpu_direct | 0 | `scripts/select_v21_segmentation_stable_keyframes.py` or `scripts/select_v21_agent_keyframes_from_plan.py` | segmentation keyframes |
| P06_S03 | owlv2_bbox_proposals | gpu_wrapper | 10000 | `scripts/run_v21_owlv2_bbox_proposals.py` | open-vocabulary boxes |
| P06_S04 | bbox_approval | agent_judgment | 0 | runner agent / `scripts/approve_v21_owlv2_bbox_prompts.py` | accepted bbox prompts |
| P06_S05 | sam2_video_segmentation | gpu_wrapper | 18000 | `scripts/run_v21_sam2_proper_segmentation.py` or `scripts/run_sam2_vlm_points_track.py` | mask tracks |
| P06_S06 | mask_contamination_review | agent_judgment | 0 | `scripts/review_v21_segmentation_contamination.py` plus visual review | accepted masks or repair request |
| P07_S01 | visible_surface_from_depth | gpu_wrapper | 12000 | `scripts/run_v21_visible_surface_from_depth.py` | visible metric surface |
| P07_S02 | geometry_condition_packet | cpu_direct | 0 | `scripts/build_v21_geometry_condition_packet.py` | geometry conditioning packet |
| P07_S03 | mesh_candidate_or_completion | gpu_wrapper | 24000 | `scripts/build_v21_mesh_candidate_from_observed.py`, TRELLIS, or selected mesh backend | object mesh candidate |
| P07_S04 | rigid_pose_fit | cpu_direct | 0 | `scripts/solve_v21_rigid_pose_fit.py`, `scripts/solve_v21_robust_pose.py`, `scripts/solve_v21_temporal_pose_graph.py` | object pose posterior |
| P08_S01 | contact_occlusion_nonpenetration | cpu_direct | 0 | `scripts/build_v21_contact_occlusion_nonpenetration.py` and V18 reducers when supported | explicit interaction state or unresolved state |
| P09_S01 | state_assembly | cpu_direct | 0 | `scripts/assemble_v21_state.py`, `scripts/assemble_v21_to_v18_annotations.py` | renderable V21 state |
| P09_S02 | full_annotation_render | gpu_wrapper | 12000 | `scripts/render_v21_full_annotation.py`, `scripts/render_v21_v18_compositor.py`, or selected renderer | overlay/world/side-by-side videos |
| P10_S01 | render_review_and_manifest_update | agent_judgment | 0 | runner visual review, then `scripts/v21_parallel_claim_next.py --complete/--fail` | completion or concrete mechanism failure |

## Wrapper command template

```bash
python scripts/v21_gpu_wrapper.py \
  --module-id P06_S05_sam2_video_segmentation \
  --request-mb 18000 \
  --log-jsonl "$RUN_ROOT/logs/gpu_wrapper_events.jsonl" \
  -- python scripts/run_v21_sam2_proper_segmentation.py ...
```

## CPU direct command rule

CPU direct commands may run normally, but they still need `case_id`, `run_root`, expected outputs, and a physical blocker in the runner's internal state. A CPU command finishing successfully is execution evidence only.

## Agent judgment rule

Agent judgment steps are part of the pipeline. They are unnumbered physical decisions in the V21 runbook, but in parallel mode they are represented here so runners do not skip object planning, prompt approval, contamination review, render review, or failure classification.
