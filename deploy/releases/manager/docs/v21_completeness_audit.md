# V21 Pipeline Completeness Audit

## Purpose
Verify that V21's algorithm flow does not reduce V19's capabilities,
includes all additional algorithms discussed, maintains V21 harness
framework (validation, fusion, weighting), and only differs in having
more diverse observation/evidence sources and modality-dependent depth.

## 1. V21 Orchestration Stages vs Actually-Executed Stages

The V21 orchestration (`docs/v21_english_orchestration.md`) defines the
complete pipeline. Here is each stage with execution status:

### Stage 1: Input and Timeline (Sections 1-2)
| Step | Script | Status | Notes |
|---|---|---|---|
| Bootstrap run root | `prepare_v21_infer_run.py` | ✅ DONE | Both datasets |
| Raw frame manifest | `build_v21_source_frame_manifest.py` | ✅ DONE | Both datasets |
| Benchmark dataset | `prepare_v21_benchmark_dataset.py` | ✅ EXISTS | Not run (infer mode only) |

### Stage 2: Depth/Camera (Sections 3-4)
| Step | Script | Status | Notes |
|---|---|---|---|
| Modality report | `build_v21_depth_modality_report.py` | ✅ DONE | |
| DepthPro baseline | `run_v21_depthpro_full_frame_candidate.py` | ✅ DONE | 500+1266 frames |
| UniDepth comparator | `run_unidepth_full_frame_v3.py` | ❌ NOT RUN | Network failure (git clone timeout) |
| DROID comparator | `run_droid_full_frame.py` | ❌ NOT RUN | Not attempted |
| VGGT camera | `run_vggt_native_camera_v3.py` | ❌ NOT RUN | Not attempted |
| VGGT scene geometry | `run_vggt_scene_geometry_v3.py` | ❌ NOT RUN | Not attempted |
| Stereo SGBM candidate | `run_v21_stereo_sgbm_candidate.py` | ✅ DONE | Pico stereo ran, rejected |
| Depth candidate registry | `build_v20_depth_candidate_registry.py` | ❌ NOT RUN | V20 script skipped |
| Depth selection bundle | `select_v21_depth_camera_bundle.py` | ✅ DONE | DepthPro selected provisional |
| Depth comparison | `compare_v21_depth_against_stereo_relative.py` | ✅ DONE | Stereo vs DepthPro compared |

**GAP**: No independent monocular depth comparator (UniDepth/DROID/VGGT).
DepthPro is the sole depth source, selected "provisional."

### Stage 3: Hand/MANO Candidates (Section 5)
| Step | Script | Status | Notes |
|---|---|---|---|
| RTMLib 2D keypoints | `run_rtmlib_hand2d_v3.py` | ❌ NOT RUN | RTMLib not available in env |
| WiLoR full-frame | `run_v21_wilor_hand_candidates.py` | ✅ DONE | New V21 script, both datasets |
| HaWoR world | `export_hawor_world.py` | ❌ NOT RUN | HaWoR not installed |
| HaMeR from RTMLib | `run_hamer_rtmlib_hand_stream_v3.py` | ❌ NOT RUN | HaMeR checkpoint missing |
| Merge candidate streams | `merge_hand_candidate_streams_v7.py` | ❌ NOT RUN | Only WiLoR stream exists |
| Metric MANO refit | `run_v21_mano_metric_refit.py` | ✅ DONE | New V21 script |
| Hand diagnosis | `diagnose_v21_hand_candidate_inputs.py` | ⚠️ PARTIAL | Diagnosis in runner, not standalone |
| Active MANO optimizer | `optimize_contact_aware_mano_graph_v8.py` | ❌ NOT RUN | V8 optimizer not invoked |
| Hand shape posterior | `solve_v20_hand_shape_track.py` | ❌ NOT RUN | |
| MANO-object constraint | `apply_v18_mano_object_constraint_state.py` | ❌ NOT RUN | |

**GAP**: Only WiLoR candidate stream. No RTMLib 2D, HaMeR, or HaWoR.
No active optimization. No merged multi-candidate stream.

### Stage 4: Segmentation (Section 6)
| Step | Script | Status | Notes |
|---|---|---|---|
| Agent keyframes from object plan | `select_v21_agent_keyframes_from_plan.py` | ✅ WIRED | Outputs `segmentation_stable_keyframes.json` for OWLv2 detector frames |
| OWLv2 bbox proposals | `run_v21_owlv2_bbox_proposals.py` | ✅ WIRED | Runs on selected keyframes |
| Approved OWLv2 bbox prompts | `approve_v21_owlv2_bbox_prompts.py` | ✅ WIRED | Outputs `owlv2_bbox_approved_prompts.json` |
| SAM2 proper bbox propagation | `run_v21_sam2_proper_segmentation.py` | ✅ WIRED | Active output is `sam2_proper_summary.json` and `sam2_proper/<object_id>/sam2_masks` |
| Visible geometry from SAM2+depth | `build_v19_visible_geometry_from_sam2_depth.py` | ⚠️ ADAPTED | Via V21 visible surface script |
| Segmentation contamination review | `review_v21_segmentation_contamination.py` | ✅ WIRED | Reads `sam2_proper_summary.json` |

**GAP**: Heavy OWLv2/SAM2 rerun still requires authorized non-local compute and visual review of produced masks.

### Stage 5: Object Geometry and Pose (Sections 7-8)
| Step | Script | Status | Notes |
|---|---|---|---|
| Visible surface from depth | `run_v21_visible_surface_from_depth.py` | ✅ DONE | |
| Heightfield dataset export | `export_v21_heightfield_dataset.py` | ✅ DONE | |
| Heightfield completion | `complete_object_heightfield_from_mask_depth_v3.py` | ❌ NOT RUN | V18 script not invoked |
| Scaled observed mesh | `reconstruct_scaled_observed_object_mesh_v3.py` | ❌ NOT RUN | |
| Visual hull depth carve | `reconstruct_object_visual_hull_depth_carve_v3.py` | ❌ NOT RUN | |
| TRELLIS shape | `remote_run_trellis_shape_v3.py` | ❌ NOT RUN | TRELLIS setup incomplete |
| Compact rigid completion | `build_v18_compact_rigid_trellis_completion.py` | ❌ NOT RUN | |
| CAD fit | `fit_v20_cad_mesh_to_visible_depth.py` | ❌ NOT RUN | No CAD |
| Geometry candidate registry | `build_v20_geometry_candidate_registry.py` | ❌ NOT RUN | |
| Geometry validation | `validate_v20_geometry_candidates.py` | ❌ NOT RUN | |
| ICP pose fit | `fit_v18_compact_rigid_object_pose.py` | ❌ NOT RUN | ICP diverged, used mask-centroid |
| Rigid pose graph | `solve_v19_rigid_object_pose_graph.py` | ❌ NOT RUN | |
| Object factor graph | `optimize_object_factor_graph_v3.py` | ❌ NOT RUN | |
| Mesh prior pose graph | `optimize_mesh_prior_pose_graph_v3.py` | ❌ NOT RUN | |
| Joint camera-object graph | `optimize_joint_camera_object_graph_v3.py` | ❌ NOT RUN | |
| Joint MANO-object graph | `optimize_joint_mano_object_graph_v3.py` | ❌ NOT RUN | |
| Contact patch pose graph | `optimize_contact_patch_object_pose_graph_v3.py` | ❌ NOT RUN | |
| **V21 mesh candidate** | `build_v21_mesh_candidate_from_observed.py` | ✅ DONE | New V21 script |
| **V21 pose estimate** | `solve_v21_rigid_pose_estimate.py` | ✅ DONE | New V21 script (mask-centroid) |

**CRITICAL GAP**: NO factor graph optimization of any kind.
V19's rigid pose graph (`solve_v19_rigid_object_pose_graph.py`) not invoked.
None of the 5 pose graph optimizers from V18/V19 run.
My `solve_v21_rigid_pose_estimate.py` uses mask-centroid, not a factor graph.

### Stage 6: Contact/Occlusion/Nonpenetration (Section 9)
| Step | Script | Status | Notes |
|---|---|---|---|
| Contact ownership graph | `build_v18_contact_ownership_graph.py` | ❌ NOT RUN | |
| Occlusion owner graph | `build_v18_occlusion_owner_graph.py` | ❌ NOT RUN | |
| Signed nonpenetration | `build_v18_signed_nonpenetration_evidence.py` | ❌ NOT RUN | |
| Triangle nonpenetration | `build_v18_triangle_nonpenetration_evidence.py` | ❌ NOT RUN | |
| MANO-object constraint | `build_v18_mano_object_constraint_state.py` | ❌ NOT RUN | |
| Apply MANO-object constraint | `apply_v18_mano_object_constraint_state.py` | ❌ NOT RUN | |

**CRITICAL GAP**: NO contact, occlusion, or nonpenetration state at all.

### Stage 7: Integrated Render (Section 10)
| Step | Script | Status | Notes |
|---|---|---|---|
| V21 overlay render | `render_v21_integrated_overlay.py` | ✅ DONE | Hand+object overlay |
| V18 overlay/world/side-by-side | `render_v18_full_pipeline_from_annotations.py` | ❌ NOT RUN | |
| V18 full pipeline renderer | `run_v18_full_pipeline.py` | ❌ NOT RUN | |

**GAP**: Only simple overlay. No world metric view, no side-by-side.
V18 full pipeline renderer (8166 lines with contact/occlusion/nonpenetration
visualization) not invoked.

### Stage 8: Benchmark Evaluation
| Step | Script | Status | Notes |
|---|---|---|---|
| GT evaluation | `evaluate_v20_benchmark_gt.py` | ❌ NOT RUN | Infer mode only |

## 2. Summary of Missing V19 Capabilities

### A. Factor Graph Optimization (V19 spine)
V19's physical-state spine includes multiple factor graph optimizers that
perform temporal smoothing, multi-measurement fusion, and joint optimization.
**NONE of these run in current V21:**

- `optimize_contact_aware_mano_graph_v8.py` — joint MANO + contact optimization
- `solve_v19_rigid_object_pose_graph.py` — temporal rigid pose graph with nonpenetration
- `optimize_object_factor_graph_v3.py` — object factor graph
- `optimize_mesh_prior_pose_graph_v3.py` — mesh prior + pose graph
- `optimize_joint_camera_object_graph_v3.py` — joint camera + object
- `optimize_joint_mano_object_graph_v3.py` — joint MANO + object
- `optimize_contact_patch_object_pose_graph_v3.py` — contact patch + object pose

### B. Contact/Occlusion/Nonpenetration (V18 modules)
V18 has dedicated evidence builders and graph solvers for:
- Contact ownership (which hand touches which object)
- Occlusion depth order (what is in front of what)
- Nonpenetration residuals (hands don't pass through objects)
**NONE of these run in current V21.**

### C. Multi-Candidate Hand Streams
V19 design uses RTMLib 2D + WiLoR + HaMeR + HaWoR as multiple candidate
streams that get merged. Current V21 only runs WiLoR.

### D. Full Pipeline Render
V18's `run_v18_full_pipeline.py` renders overlay + world + side-by-side
with all physical state layers (MANO mesh, object mesh pose, contact,
occlusion, nonpenetration, visibility). Current V21 renders a simple
overlay with skeleton lines and bounding boxes only.

## 3. Missing V21-Specified Algorithms

### A. Additional Depth Algorithms
V21 design Section 8 specifies multiple depth modalities:
- **UniDepth**: metric depth + intrinsics (not run, network failure)
- **Metric3D V2**: metric depth + surface normals (not attempted)
- **MASt3R/DUSt3R**: multiview camera/depth (not attempted)
- **RAFT-Stereo/IGEV-Stereo**: stereo disparity (SGBM ran only)

### B. Segmentation Runtime And Review
- **OWLv2 bbox proposals**: active detector source; requires authorized heavy rerun on selected keyframes
- **Approved bbox prompts**: required machine-consumable artifact before SAM2 proper
- **SAM2 proper video predictor**: active full-video propagation from approved bbox prompts
- **Segmentation_sam2_proper review**: required visual/programmatic contamination review before masks feed geometry

### C. Object Geometry Generation
- **TRELLIS**: image-to-3D mesh generation (code cloned, not run)
- **SPAR3D**: fast mesh candidate (not attempted)
- **TripoSG/Hunyuan3D**: high-quality candidates (not attempted)

### D. MANO Active Optimization
V21 Section 5 requires active betas/scale/pose optimization with:
- 2D keypoint reprojection loss
- Hand silhouette distance
- Visible hand depth loss
- Bone/span prior
- MANO shape prior
- Temporal smoothness
- Contact compatibility
- Nonpenetration
**Not implemented.** Only bone-length scaling + depth-z refit.

## 4. What Must Be Fixed (Priority Order)

### P0: Factor Graph (V19 spine)
The single most critical gap. Without any factor graph optimization,
V21 is a feed-forward pipeline, not the physical-state spine V19 defined.
Must invoke at minimum:
- `solve_v19_rigid_object_pose_graph.py` for object pose
- `optimize_contact_aware_mano_graph_v8.py` for MANO

### P1: Contact/Occlusion/Nonpenetration
These are explicit V21 physical variables. Without them, contact claims
and depth-order are absent from the final artifact.

### P2: V18 Full Pipeline Render
Must produce overlay + world + side-by-side with all layers.
The V18 renderer already supports mesh-pose rendering, MANO mesh rendering,
contact visualization, occlusion state, and nonpenetration.

### P3: Additional Depth Comparators
UniDepth at minimum (for monocular baseline comparison).

### P4: TRELLIS Mesh Generation
For higher-quality object geometry candidate.

### P5: Multi-Candidate Hand Streams
RTMLib 2D + HaMeR alongside WiLoR.
