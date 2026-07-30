# Pipeline V18: Real-Time-Scale Occlusion-Aware HOI Annotation

## Status

V18 is open as a redesign after formal V17 failure. The V18 contract is artifact-first but not schema-first: every run must produce full-video renderable annotations, and those annotations must be driven by the named mechanisms. The artifact is the rendered video annotation set—overlay, world, and side-by-side videos—plus backing data only insofar as it reproduces or explains those rendered annotations. All object, hand, pose, contact, occlusion, and geometry estimates may be approximate and uncertain by default. Residuals, audits, and thresholds may score, explain, or debug the artifact, but they may not prevent artifact production or replace the artifact. A module that cannot produce a confident estimate may emit a typed candidate or an explicit unresolved/unknown state in the backing data to keep the videos honest, but that emission is **not completion** of the named requirement; strict completion requires the real mechanism to run, be sanity-checked on video/geometric evidence, affect downstream reasoning where applicable, and be rendered in the final videos.

V17 failed as a pipeline design, not merely as an unfinished run:

1. Runtime violated the product constraint. A path that can spend tens of hours on a roughly one-minute clip is unacceptable; pipeline wall time must be the same order of magnitude as input duration.
2. BundleSDF was the wrong default object mechanism. It performs per-instance/test-time neural optimization for a rigid canonical mesh plus per-frame SE(3). Using GPU-hours to discover whether a manipulated object is rigid is bad methodology; physical state type must be inferred cheaply from model-produced semantics and fast residual checks.
3. Occlusion was not a first-class state. V17 mostly marked rows as unobserved, repaired selected hand-depth measurements, or used temporal candidates; it did not maintain explicit hand/object visibility, occluder ownership, depth ordering, uncertainty, and bounded infill through occlusion.
4. V17 over-optimized evidence ledgers and component diagnostics instead of delivering an efficient full-video annotation path.
5. V17 produced useful evidence, especially depth-edge ownership for MANO/depth measurement validity, but it did not deliver an acceptable V17 annotation result.

All V17 readiness flags remain false: `v3_solver_complete=false`, `annotation_ready=false`, `deliverable_ready=false`, `accuracy_target_met=false`, `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `rigid_pose_requirement_met=false`.

## Binding Baseline Pipeline

This section is the V18 contract. Cached artifacts may be used as memoized stage outputs, but the logical pipeline is self-contained from raw video and named model/config inputs. A component named here cannot be silently replaced; replacement requires an evidence-backed design amendment.

**Execution rule.** V18 implements every named module as an executable stage that drives the final rendered annotation videos, with backing JSON only where needed to reproduce or explain those rendered marks. There are no arbitrary acceptance gates that stop the pipeline. Numeric residuals, thresholds, and audits are diagnostic evidence and confidence features only. The pipeline distinguishes implementation mistakes from quality limitations:

- Implementation mistakes must be fixed before delivery: missing rendered-annotation writer, missing full-video output, frame/FPS mismatch, broken backing schema needed to reproduce the videos, failed model execution, invalid coordinate/depth/frame transform, or disconnected module output that does not affect the rendered annotations.
- Quality limitations do not block artifact production: weak masks, incomplete geometry, approximate poses, uncertain contact, unresolved occlusion owner, noisy hand estimates, or low confidence. These must be represented honestly in the artifact and render, not hidden in diagnostics. They also must not be counted as strict requirement completion unless the named mechanism itself is implemented and driving the annotation.

All outputs are approximate and uncertain. The user-facing judgment is whether the delivered V18 videos and explanations are plausibly no worse than V16 and show concrete improvements, not whether an arbitrary internal gate declares a field ready.

1. **Camera/depth backbone.** Run DROID-SLAM-style camera tracking and the project metric-depth backend on every raw frame. These produce the camera/depth coordinate backbone and measurement factors, not unquestioned physical truth. The graph may use bounded camera/depth correction variables.
2. **Hand branch.** Run HaWoR, WiLoR, and RTMLib on the full video. HaWoR is the required temporal/occlusion hand baseline. WiLoR is the visible-frame MANO candidate stream. RTMLib is the independent 2D keypoint anchor. For visible frames, compare candidates using 2D reprojection, metric-depth consistency, temporal smoothness, and hand bone-scale plausibility as confidence features. HaWoR temporal continuation through occlusion is an approximate candidate stream, not a proof. Low confidence changes the uncertainty label; it does not remove the hand state from the artifact.
3. **Object/part perception.** A VLM planner produces object roster, physical-state proposals, and object/part prompts. OWLv2 produces text-conditioned keyframe boxes. SAM2 is the baseline video segmentation/tracking model that converts prompts into temporal object/part masks. SAM v1 is not the baseline tracking path.
4. **Physical-state decision.** VLM labels are hypotheses, not branch logic. Rigid, articulated, deformable, and unresolved states are accepted by residual tests over masks, depth, and temporal geometry; not by object name, category, color, material, or action phrase.
5. **Geometry/reconstruction.** Accepted masks plus metric depth produce visible point clouds/surfaces. Rigid objects or rigid parts use multi-frame depth fusion with SE(3) registration and report residuals, silhouette agreement, and uncertainty. Articulated objects reconstruct parts separately and compare relative-transform, hinge/circle, single-rigid, and unresolved models by residuals and visual plausibility. Deformable or under-observed objects still emit visible-surface/deformable candidates. Weak reconstruction lowers confidence; it does not delete the module output.
6. **Factor graph.** Variables are bounded camera/depth correction, hand state, object/part SE(3), articulation parameter, contact switch, and occlusion owner. Factors are hand observation residuals, object mask/depth/registration residuals, temporal/rigid/articulation consistency, occlusion depth ordering, and contact/nonpenetration. Contact factors are always represented as approximate hypotheses with confidence/uncertainty. Geometry, depth ordering, nonpenetration, and temporal cues affect confidence and ownership; weak evidence yields uncertain contact rather than omitted contact.
7. **Outputs.** The deliverable is the full-duration renderable annotation set: raw-video overlay, metric/world render, and side-by-side video. Runtime reports, validation artifacts, and `annotations_v18_full.json` are backing evidence/reproducibility data only; they are not the artifact the user needs and cannot satisfy requirements by themselves. Status videos and audits validate implementation; they do not replace the rendered annotation pipeline. A short explanation may accompany the videos, but completion is judged from the rendered annotations and the mechanisms that drive them.

Current implementation status: the final artifact-producing path is implemented as `scripts/run_v18_full_pipeline.py` and produces full-video overlay/world/side-by-side renders plus `annotations_v18_full.json` for `trash_1050` and `task5_tomato_960`. V18 is still not closed as a strict physical solution: the current solver remains approximate, some contact anchors are intentionally not pose-coupled, deformable object state lacks a nonrigid solver, and global occlusion/object-geometry closure remains incomplete.

## Implementation Checkpoint 1: Runtime And Visibility Scaffold

V18 now has the first bounded scaffold artifacts, generated without heavy perception or reconstruction:

```text
/data2/ego_annotation_outputs/v18_runtime_manifest/
/data2/ego_annotation_outputs/v18_visibility_occlusion_state/
```

`build_v18_runtime_manifest.py` writes the default V18 DAG and hard budget contract before any heavy stage runs. The planned default critical path is 7.75x real time, under the initial 10x hard ceiling: 271.25 seconds for `trash_1050` (35.0 s raw video) and 248.0 seconds for `task5_tomato_960` (32.0 s raw video). The default DAG contains no BundleSDF, NeRF, neural-field training, or all-face CPU raster stage. This is a plan/budget gate, not proof that later implementation will meet the budget.

`build_v18_visibility_occlusion_state.py` writes full-timeline hand and object visibility rows from existing fast evidence. It does not infer certain poses through occlusion. Unobserved hands are marked unresolved, with possible short occlusion recorded only as an unowned hypothesis when bounded detector gaps overlap visible active objects. Object geometry is scoped as visible depth-backed surface, visible mask with rejected surface, or no visible geometry; every object row keeps `object_geometry_complete=false` and `object_pose_requirement_met=false`. Physical state types are recovered from a structured schema over model-produced VLM physical notes rather than object-name branches. Current counts:

- `trash_1050`: 2,100 hand-state rows with 1,593 visible, 8 partially visible, and 499 unresolved; 4,200 object-state rows with 1,604 visible, 29 unresolved active-mask gaps, and 2,567 out-of-frame/inactive rows; object geometry scopes are 1,417 visible depth-backed surfaces, 187 visible masks with rejected surfaces, and 2,596 no-visible-geometry rows. Model physical states: 2 deformable, 1 articulated, 1 rigid.
- `task5_tomato_960`: 1,920 hand-state rows with 1,722 visible, 11 partially visible, and 187 unresolved; 8,640 object-state rows with 1,109 visible, 40 unresolved active-mask gaps, and 7,491 out-of-frame/inactive rows; object geometry scopes are 694 visible depth-backed surfaces, 415 visible masks with rejected surfaces, and 7,531 no-visible-geometry rows. Model physical states: 6 rigid, 1 deformable, 1 articulated, 1 unknown.

Readiness remains false. The next V18 step is fast object-surface/motion state and a bounded consistency graph; the scaffold only makes runtime and occlusion state explicit.

## Implementation Checkpoint 2: Fast Object Motion State

V18 now writes a cheap object surface/motion reducer:

```text
/data2/ego_annotation_outputs/v18_fast_motion_state/
```

`build_v18_fast_motion_state.py` consumes the V18 visibility/occlusion state plus existing V17 visible-surface/material-track/surface-replay evidence. It runs in under a second on the representative cases and does not run BundleSDF, NeRF, or any new reconstruction backend. The reducer preserves the distinction between visible-surface motion evidence and complete object pose.

Current fast motion-state counts across both cases: 1 partial rigid visible-surface motion support, 1 local rigid-motion-only-not-pose, 3 deformable visible-surface/surface-motion states, 2 articulated visible-surface unresolved states, 1 visible-surface-only motion unresolved state, and 5 motion-unresolved/no-surface states. Per-case observations:

- `trash_1050`: black trash bag and white trash bag are deformable visible-surface states; off-white can is articulated/visible-surface unresolved; pink-lid can is the only partial rigid visible-surface motion-supported object, with 44 rigid-ready material pairs and 2 visible-surface replay-ready partial segments.
- `task5_tomato_960`: faucet handle is articulated/visible-surface unresolved; tomato has local rigid motion only, not pose; tomato peel is deformable visible-surface/motion; several context objects have no usable surface/motion evidence and stay unresolved.

This confirms the V18 design premise: fast model-produced physical state plus cheap residual summaries can prevent wasting GPU-hours on rigid reconstruction for deformable or unresolved objects. It still does not close object geometry, contact ownership, full consistency optimization, or rendering.

## Implementation Checkpoint 3: Consistency/Contact Scaffold

V18 now writes a bounded consistency/contact reducer:

```text
/data2/ego_annotation_outputs/v18_consistency_graph/
```

`build_v18_consistency_graph.py` joins the V18 visibility/occlusion state, V18 fast motion state, V17 pairwise image contact, and V17 pairwise metric depth-gap evidence. It does not run a nonlinear optimizer and does not fill occluded poses. It exposes blocker classes that the future bounded optimizer must address.

Across the two representative cases, the reducer materializes 5,564 hand-object pair rows: 619 image-contact candidates are rejected by metric-depth contradiction, 1,490 are image-overlap-only, 3,107 have no contact image evidence, and 348 are unobserved pairs. Contact-factor-ready rows remain zero. Blockers include 619 metric-depth contradictions, 619 incomplete-object-geometry rows, 679 hand-visibility-unresolved rows, 138 object-visibility-unresolved rows, and 1,490 image-overlap-is-not-contact rows.

Per-case counts:

- `trash_1050`: 97 image-contact candidates rejected by metric depth, 977 image-overlap-only rows, 1,940 no-contact-image rows, and 252 unobserved pairs.
- `task5_tomato_960`: 522 image-contact candidates rejected by metric depth, 513 image-overlap-only rows, 1,167 no-contact-image rows, and 96 unobserved pairs.

This preserves the V17 lesson in V18 form: projected hand/object overlap is not physical contact. The next implementation step is a bounded optimizer or state update that uses interior-owned hand depth, visible object surfaces, occlusion ownership, and fast motion state together; it must still keep unresolved states explicit when metric depth or geometry do not support contact.

## Implementation Checkpoint 4: Renderable Annotation State And Status Overlay

V18 now writes a full-timeline renderable annotation state and full-duration status overlay videos:

```text
/data2/ego_annotation_outputs/v18_annotation_state/
/data2/ego_annotation_outputs/v18_renders/
```

`build_v18_annotation_state.py` joins V18 visibility/occlusion rows, fast motion state, consistency/contact rows, V17 timeline boxes/masks, and V16 raw-frame manifests into one per-frame state. Both representative cases match the raw frame count: `trash_1050` has 1,050 state frames and `task5_tomato_960` has 960 state frames. The joined state has 3,334 renderable hand-box rows and 2,713 renderable object-mask rows across both cases. It keeps `annotation_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`.

`render_v18_status_overlay.py` renders full-duration MP4 status overlays from raw frames and verified object masks. The renderer completed both cases in 61.55 seconds total and the QC manifests report exact frame-count matches:

- `trash_1050`: `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_status_overlay.mp4`, 1,050/1,050 frames, 1,601 hand boxes, 1,604 object masks, 499 unresolved-hand labels, and 29 unresolved-object labels.
- `task5_tomato_960`: `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_status_overlay.mp4`, 960/960 frames, 1,733 hand boxes, 1,109 object masks, 187 unresolved-hand labels, and 40 unresolved-object labels.

These videos are a real V18 status output, not a final pose-complete annotation. They satisfy the full-duration/same-frame-count render constraint for the 2D status overlay, and they make the missing geometry/contact state visible instead of hiding it. The remaining V18 gap is the actual bounded optimizer/geometry path and full 3D/side-by-side outputs.

## Implementation Checkpoint 5: Bounded State Solution, World/Status, And Side-By-Side Status Deliverable

V18 now writes a bounded fixed-pass state solution and a status-deliverable manifest:

```text
/data2/ego_annotation_outputs/v18_bounded_state_solution/
/data2/ego_annotation_outputs/v18_status_deliverable_manifest/
```

`build_v18_bounded_state_solution.py` classifies hand observation gaps, object geometry scope, and contact modes without filling poses through occlusion or promoting image overlap to physical contact. Across both representative cases it reports:

- Hand states: 1,257 observed depth-consistent, 2,058 observed depth-unchecked, 19 partial observations, 113 short-gap possible occlusion candidates left unfilled, 25 short gaps with no visible occluder evidence, and 548 long/open unresolved gaps.
- Object states: 2,111 visible-surface-only rows with hidden geometry unresolved, 602 visible-mask-only/surface-rejected rows, 69 active-object visibility-unresolved rows, and 10,058 inactive/out-of-frame rows.
- Contact states: 619 rejected under current metric-depth evidence, 1,490 image-overlap-only/near rows, 3,107 no-contact-image rows, and 348 unobserved pairs. Contact-factor-ready rows remain zero. Pose-filled-through-occlusion rows remain zero.

V18 also now renders full-duration abstract world/status and side-by-side status videos:

- `trash_1050`: `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_world_status.mp4` and `/data2/ego_annotation_outputs/v18_renders/trash_1050/v18_status_side_by_side.mp4`, both 1,050/1,050 frames.
- `task5_tomato_960`: `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_world_status.mp4` and `/data2/ego_annotation_outputs/v18_renders/task5_tomato_960/v18_status_side_by_side.mp4`, both 960/960 frames.

The world/status render is deliberately image-normalized abstract status geometry, not a metric 3D reconstruction. The side-by-side videos place the raw-frame status overlay next to that abstract status view. Visual sheets were extracted for both cases for non-corruption checks.

`build_v18_status_deliverable_manifest.py` writes `/data2/ego_annotation_outputs/v18_status_deliverable_manifest/v18_status_deliverable_manifest.json`. The manifest marks `status_deliverable_ready=true` and `final_pose_complete_deliverable_ready=false`. Measured render time for the status outputs is 126.15 seconds for 67 seconds of source video, or 1.88x real time, under the 10x V18 status-render budget. This closes a V18 status deliverable, not the final geometry/pose/contact deliverable.

Remaining gap after this checkpoint: implement or integrate a bounded object geometry/pose path that can actually reconstruct manipulated-object geometry where evidence supports it, validate contact ownership with metric depth and complete/appropriate geometry, and only then upgrade final annotation readiness.

## Implementation Checkpoint 6: Visible-Surface Geometry Archive

V18 now stages actual depth-backed visible-surface geometry evidence:

```text
/data2/ego_annotation_outputs/v18_visible_geometry_archive/
```

`build_v18_visible_geometry_archive.py` validates and copies the compact visible-surface NPZ archives into V18, then writes per-case reports with per-frame surface offsets, vertex/face counts, bounded object state, and explicit geometry claims. The current full-pipeline default consumes the complete-depth visible-surface archive at `/data2/ego_annotation_outputs/v18_unidepth_extension/v18_visible_geometry_archive_complete_depth`, which fills earlier metric-depth-missing visible masks with memoized UniDepth-derived metric surfaces. It preserves visible-surface measurement evidence as V18 geometry evidence, but it does not reconstruct hidden geometry, canonical meshes, or complete object pose.

Across both representative cases the archive contains 2,111 accepted depth-backed visible-surface frame rows, 602 rejected visible-mask rows, 1,212,570 vertices, and 2,107,754 faces. Object-level status counts are: 1 partial rigid visible-surface archive ready but not complete pose, 7 visible-surface archives with hidden geometry unresolved, and 5 visible masks without accepted surface. Geometry claims are 8 depth-backed visible-surface-only objects and 5 objects with no accepted visible surface.

The status manifest now points to these archives and reports `visible_geometry_archive_ready=true`, while keeping `hidden_geometry_reconstructed=false`, `canonical_mesh_ready=false`, `complete_object_pose_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`. This is the first V18 object-geometry artifact, scoped to observed visible surfaces only.

Remaining gap after this checkpoint: a bounded method that completes manipulated-object geometry where warranted, estimates object pose only under that geometry support, and validates contact ownership.

## Implementation Checkpoint 7: Object Completion Eligibility Gate

V18 now writes an object completion/pose eligibility gate:

```text
/data2/ego_annotation_outputs/v18_object_completion_gate/
```

`build_v18_object_completion_gate.py` uses V18 visible geometry and fast motion state to decide which objects may enter a future bounded completion path and which objects must remain blocked or visible-surface-only. It does not run completion and does not mark any object pose complete.

Across 13 objects, the reviewed gate finds zero single-rigid completion candidates. It identifies 3 part/relative-motion candidates that require an object/part split before any completion path: `object:off_white_trash_can_first`, `object:pink_lid_trash_can_second`, and `object:obj_faucet_handle`, each with action `candidate_requires_part_model_not_run`. It blocks or defers the rest: 3 deformable objects remain visible-surface-only/no rigid pose, 5 objects have no accepted visible surface, 1 object has only local motion not pose (`object:obj_tomato`), and 1 rigid-prior object has visible surface but lacks persistent motion/completion evidence.

The updated status manifest reports `object_completion_candidate_count=0`, `object_part_split_candidate_count=3`, `object_completion_run_count=0`, and `object_completion_pose_ready_count=0`. This gate is a methodological guardrail: the next geometry step may only proceed after part-level object splitting or stronger geometry/motion evidence, and must keep all blocked states explicit.

Remaining gap after this checkpoint: implement part-level splitting/geometry evidence for the part-motion candidates, or integrate a bounded feed-forward/observed multi-view geometry prior, then validate object pose and contact ownership.

## Implementation Checkpoint 8: Part-Split Evidence And Part Visible Surfaces

V18 now audits generated model-produced part/segment tracks and extracts bounded part visible-surface evidence:

```text
/data2/ego_annotation_outputs/v18_part_split_evidence/
/data2/ego_annotation_outputs/v18_part_visible_surfaces/
```

`build_v18_part_split_evidence.py` uses the generated-only part-track source manifest by default. Legacy cached roots are opt-in debug inputs and make the source pool non-uniform. Within the selected candidate pool, assignments are made only by mask overlap/containment with the whole-object mask, not by object name. Across the three objects requiring part/articulation handling (`object:off_white_trash_can_first`, `object:pink_lid_trash_can_second`, and `object:obj_faucet_handle`), it finds 5 accepted generated part-track assignments: two for off-white can, one for pink-lid can, and two for faucet handle. Pink-lid is explicitly `single_part_mask_evidence_insufficient_for_split`.

`build_v18_part_visible_surfaces.py` then extracts metric-depth-backed visible surfaces for the accepted generated part masks, without OpenCV and without BundleSDF/NeRF. It writes a compact NPZ archive in depth-camera coordinates. Current totals: 809 accepted part visible-surface frame rows, 302,902 vertices, and 525,741 faces. Rejections remain scoped to missing metric depth, part/object containment failure, or too few connected sampled vertices/faces; target-support sampling does not relax minimal acceptance thresholds.

The updated status manifest reports `part_required_object_count=3`, `accepted_part_track_assignment_count=5`, `part_visible_surface_frame_rows=809`, `part_visible_surface_vertices=302902`, `part_visible_surface_faces=525741`, `part_pose_ready_count=0`, and `object_pose_requirement_met=false`. This advances V18 from whole-object visible surfaces to generated part-level visible surface evidence, but it still does not reconstruct hidden part geometry, estimate part pose, or validate contact ownership.

Remaining gap after this checkpoint: convert generated part visible surfaces into bounded part/articulation model candidates only where supported by residuals, get more pink-lid part evidence if a split model is required, and then test pose/contact ownership against metric depth.

## Implementation Checkpoint 9: Part-Motion State Reducer

V18 now writes bounded part-motion diagnostics:

```text
/data2/ego_annotation_outputs/v18_part_motion_state/
```

`build_v18_part_motion_state.py` reduces part visible-surface centers into pairwise relative-distance summaries. It does not estimate part pose, articulation parameters, hidden geometry, or object pose. Under the current generated-only source pool, it sees three objects with part surfaces: off-white can has a variable two-part relationship, pink-lid has only one part surface, and faucet has a variable two-part relationship after adaptive surface sampling recovers shared frames.

The updated status manifest reports `part_motion_object_count=3`, `articulation_model_ready_count=0`, and `part_pose_ready_count=0`. This means V18 has part-level visible-surface evidence to test candidate mechanisms, but not enough to accept part pose or contact ownership.

Remaining gap after this checkpoint: resolve whether variable/underconstrained part distances are true articulation, mask drift, sparse-surface artifacts, or depth/visibility noise; formulate a bounded part model only if discriminating residual evidence supports it.

## Implementation Checkpoint 10: Part-Motion Confound QC

V18 now audits the part-motion diagnostic for quality confounds:

```text
/data2/ego_annotation_outputs/v18_part_motion_qc/
```

`build_v18_part_motion_qc.py` checks whether variable part-pair distances are supported by robust part surfaces or are confounded by sparse/unstable part tracks. Current generated-only QC states are: off-white can and faucet handle each have robust variable part-pair evidence classified as articulation hypotheses not yet fitted; pink-lid is underconstrained because it has one generated part track.

The updated status manifest reports `part_motion_qc_object_count=3`, `articulation_model_ready_count=0`, and `part_pose_ready_count=0`. This prevents V18 from overinterpreting noisy or underconstrained pair distances as an articulation model.

Remaining gap after this checkpoint: fit and validate bounded articulation parameters for robust variable pairs; for pink-lid, obtain a second semantic part track if a split/articulation model is required. Do not accept articulation from underconstrained residuals.

## Implementation Checkpoint 11: Bounded Visible Part-Model Candidates

V18 now records bounded visible part-model candidates:

```text
/data2/ego_annotation_outputs/v18_part_model_candidates/
```

`build_v18_part_model_candidates.py` now records accepted candidates, rejected residual probes, and surface-level ICP diagnostics over the actual depth-backed part vertices. Under the current generated-only evidence it accepts no part model candidates. It writes two not-yet-fitted articulation-hypothesis probes for robust variable pairs (off-white can and faucet handle) and one rejected single-part probe for pink-lid because a split/articulation model requires at least two semantic part tracks.

The updated status manifest reports `part_model_candidate_count=0`, `part_model_rejected_candidate_count=3`, `part_surface_icp_probe_count=5`, `part_surface_icp_probe_state_counts={surface_icp_residual_supported_visible_only_not_pose: 5}`, `part_articulation_hypothesis_pair_count=2`, `visible_subset_model_candidate_count=0`, `articulation_model_ready_count=0`, `part_pose_ready_count=0`, and `object_pose_requirement_met=false`. Surface ICP is diagnostic visible-shape evidence, not part pose.

Remaining gap after this checkpoint: fit bounded articulation parameters/joint axes for the two robust variable pairs and obtain additional pink-lid part evidence if a split model is required; no part model may be promoted without passing residual tests.

## Implementation Checkpoint 12: Materialized Visible Part-Subset Archive

V18 now materializes robust stable visible part-subset candidates as mesh archives:

```text
/data2/ego_annotation_outputs/v18_visible_part_subset_archive/
```

`build_v18_visible_part_subset_archive.py` copies only observed depth-backed surfaces from accepted robust stable part-subset candidates. Under the current generated-only evidence there are no accepted visible subset candidates, so the archive has no evidence-ready subset.

The current manifest reports `visible_part_subset_archive_ready_count=0` and `all_cases_visible_part_subset_archive_ready=false`. Hidden geometry, part pose, articulation readiness, contact readiness, and object pose remain false.

Remaining gap after this checkpoint: create an archive only after a residual-tested visible subset candidate exists; do not promote rejected probes or single-part surfaces beyond visible evidence.

## Implementation Checkpoint 13: Part-Object Blocker Manifest

V18 now writes explicit blockers for part/relative-motion objects:

```text
/data2/ego_annotation_outputs/v18_part_object_blocker_manifest/
```

`build_v18_part_object_blocker_manifest.py` joins part-split evidence, completion gating, part-motion QC, rejected residual probes, visible subset candidates, and the visible part-subset archive. It records 3 required part/articulation objects: `object:off_white_trash_can_first`, `object:pink_lid_trash_can_second`, and `object:obj_faucet_handle`. One is currently `blocked_part_se3_surface_residual_rejected` (faucet center fit supported, but handle surface SE(3) residual outliers fail validation), one is `blocked_articulation_fit_residual_rejected` (off-white fit rejected by radial/plane residuals), and one is `blocked_part_model_residual_probes_rejected` (pink-lid single-part evidence).

The updated manifest reports `required_part_object_blocker_count=3`, `part_object_blocker_rejected_candidate_count=3`, `visible_part_subset_archive_ready_count=0`, and `contact_ownership_ready_count=0`. This is an explicit stop against treating generated masks, visible surfaces, articulation hypotheses, or rejected residual probes as hidden geometry, part pose, contact ownership, or final object pose.

Remaining gap after this checkpoint: repair off-white articulation residuals, promote faucet only after full part SE(3)/silhouette/depth validation, and obtain additional pink-lid part evidence if needed, then rerun geometry/motion/contact checks.

## Implementation Checkpoint 14: Part-Mask Acquisition Status

V18 now records the status of acquiring missing or improved part masks:

```text
/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan/
```

`build_v18_part_mask_acquisition_plan.py` turns the blocker manifest into object-level acquisition requirements and probes local runner prerequisites. It covers the same 3 part/relative-motion objects. The current state is no longer missing generated masks: all three have generated OWLv2→SAM2 mask evidence; off-white is blocked by a rejected articulation fit, faucet is blocked by shared-frame part-surface SE(3) residual outliers, and pink-lid remains blocked by a rejected single-part residual probe.

The `.venv` environment has Python cv2, torch, CUDA, local SAM2, and local OWLv2 available, but no SAMWISE repo or checkpoint was found in the checked paths. The current status manifest records `local_new_mask_generation_ready_count=3` and `mask_evidence_created_count=5`, while keeping `part_pose_ready_count=0`.

Remaining gap after this checkpoint: repair off-white articulation residuals, repair faucet handle SE(3) surface residual outliers, and acquire a second pink-lid part track if that object still requires a split model.

## Implementation Checkpoint 15: Review-Driven Readiness And Source-Scope Corrections

A clean-room adversarial review found one false readiness issue and one source-scope caveat. V18 separates visible part-subset archive file creation from evidence readiness. After the generated-only source correction, no visible part-subset archive is evidence-ready: `visible_part_subset_archive_ready_count=0` and `all_cases_visible_part_subset_archive_ready=false`.

The later generated-only source correction records `part_track_candidate_source_scope=v18_owlv2_sam2_generated_only` and `uniform_part_track_generation_ready=true`. This preserves the true scoped claim: source selection is uniform and auditable, while downstream assignment remains geometric overlap/containment against whole-object masks.

Remaining gap after this checkpoint: continue from generated masks to residual-tested part models, reconstruction, contact/depth ownership, and graph factors.

## Implementation Checkpoint 16: Part-Track Source Manifest

V18 now writes an explicit source-of-truth for part-track candidate inputs:

```text
/data2/ego_annotation_outputs/v18_part_track_source_manifest/
```

`build_v18_part_track_source_manifest.py` records generated OWLv2→SAM2 part-track roots, discovered track counts, usable-track counts, and source-scope readiness. Legacy cached roots are no longer default inputs; they require explicit debug CLI arguments and make the pool non-uniform.

`build_v18_part_split_evidence.py` consumes this manifest instead of carrying case-specific roots internally. This turns the prior review caveat into an executable contract: source selection is explicit and auditable, while downstream assignment remains geometric overlap/containment against whole-object masks. The current status manifest reports `part_track_source_manifest_ready_all_cases=true`, `part_track_source_root_count=2`, `part_track_source_usable_track_count=5`, and `uniform_part_track_generation_ready=true`.

Remaining gap after this checkpoint: use the uniform generated tracks to fit residual-tested part models and reject/accept reconstruction/contact claims.

## Implementation Checkpoint 17: Structured Physical-State Schema

V18 now centralizes physical-state interpretation:

```text
/data2/ego_annotation_outputs/v18_physical_state_schema/
```

`build_v18_physical_state_schema.py` converts model-produced physical notes into structured fields: primary whole-object physical state, part/relative-motion requirement, secondary deformable/surface-component evidence, optical difficulty, and unresolved surface-change flags. Visibility now consumes this schema, fast-motion rows propagate its fields, and the object-completion gate consumes the structured `requires_part_or_relative_motion_model` flag instead of reparsing notes locally.

The latest structured schema covers 13 objects and reports physical-state counts of 7 rigid, 3 deformable, 3 articulated, and 0 unknown. It identifies 3 part/relative-motion-required objects: `object:off_white_trash_can_first`, `object:pink_lid_trash_can_second`, and `object:obj_faucet_handle`. The schema source is structured VLM physical-model fields; free-text `physical_notes` are retained as context/provenance but are not keyword-parsed as the authority for physical type.

The updated status manifest reports `physical_state_schema_object_count=13`, `structured_part_or_relative_motion_required_count=3`, `structured_secondary_deformable_or_surface_component_count=1`, `physical_state_changed_from_legacy_keyword_count=1`, and `object_part_split_candidate_count=3`. Final hidden geometry, part pose, object pose, articulation model, and contact ownership remain false/zero.

Remaining gap after this checkpoint: replace the deterministic schema adapter with direct structured model output when the perception backend is available; until then, downstream gates consume this single auditable schema rather than local ad-hoc text parsing.

## Implementation Checkpoint 18: Measured Cached-Evidence-To-Status Runtime

V18 now has a measured runtime artifact for the implemented status pipeline:

```text
/data2/ego_annotation_outputs/v18_measured_status_pipeline_runtime/
```

`run_v18_measured_status_pipeline.py` runs 30 current V18 stages in dependency order, including HaWoR/WiLoR/RTMLib hand-baseline reduction, OWLv2→SAM2 part-track generation, and status overlay/world/side-by-side rendering, and writes per-stage stdout/stderr logs plus a runtime report. The measured generated-only run succeeded in 473.12 seconds over 67.0 seconds of representative video, or 7.06x video duration. The slowest stages were OWLv2→SAM2 part tracks (257.60 s), world/status render (61.54 s), status overlay render (59.16 s), part visible-surface extraction (26.03 s), part-SE(3) surface residuals (19.60 s), side-by-side render (19.13 s), articulation-fit candidates (10.45 s), and SAM promptable proposal probe (5.16 s).

This is explicitly `cached_evidence_to_status_runtime_measured=true`, not fresh raw-video runtime. The report keeps `fresh_raw_video_to_status_runtime_measured=false` and `fresh_raw_video_to_final_pose_runtime_measured=false` because upstream raw hand/object/depth model outputs are cached from V16/V17/V18 artifacts, while the HaWoR/WiLoR/RTMLib hand-baseline reducer and OWLv2→SAM2 part-track stage are regenerated inside this measured pipeline. The status manifest now links this report and records `cached_evidence_to_status_elapsed_to_video_ratio=7.0615`, while final pose/contact readiness remains false.

Remaining gap after this checkpoint: measure true fresh raw-video-to-status runtime only after the perception backend is provisioned; measure final runtime only after final geometry/pose/contact stages exist.

## Implementation Checkpoint 19: Occlusion Owner-Candidate Evidence

V18 now writes bounded occlusion owner-candidate evidence:

```text
/data2/ego_annotation_outputs/v18_occlusion_owner_candidates/
```

`build_v18_occlusion_owner_candidates.py` examines unresolved hand rows. For short detector gaps, it interpolates the neighboring observed hand boxes and tests overlap with current visible object boxes. This produces possible occluder-owner candidates but does not accept owner identity, depth ordering, contact, or hand pose through occlusion.

Across both representative cases there are 686 unresolved hand rows. The reducer finds 116 short-gap rows with visible-object overlap owner candidates, 51 short-gap rows with no visible-object overlap candidate, and 519 unbounded unresolved rows without temporal-gap evidence. Candidate objects are mostly trash-case objects: black trash bag (78 candidate rows), white trash bag (38), pink-lid trash can second (26), off-white trash can first (23), and one tomato row. The status manifest reports `occlusion_candidate_owner_row_count=116`, `occluder_owner_accepted_count=0`, `occlusion_depth_order_resolved_count=0`, and `pose_filled_through_occlusion_rows=0`.

Remaining gap after this checkpoint: resolve candidate ownership only with depth ordering and visibility evidence; do not fill poses or contact from box overlap alone.

## Implementation Checkpoint 20: Bounded-State Occlusion Candidate Projection

The bounded state solution now consumes `/data2/ego_annotation_outputs/v18_occlusion_owner_candidates/` and projects owner-candidate evidence into hand occlusion solution rows. This keeps candidate evidence available to renderers and future optimizers while preserving the invariant that ownership and depth ordering are not accepted.

The status manifest reports `bounded_occlusion_owner_candidate_rows=116`, `bounded_occluder_owner_accepted_rows=0`, `bounded_occlusion_depth_order_resolved_rows=0`, and `pose_filled_through_occlusion_rows=0`. The measured status pipeline was rerun after this integration and after adding the invariant audit stages; it remains a cached-evidence-to-status measurement, not fresh raw-video-to-final runtime.

Remaining gap after this checkpoint: add metric depth ordering or explicit visibility reasoning before any candidate can become an accepted occluder owner.

## Implementation Checkpoint 21: Status Invariant Audit

V18 now writes a machine-readable invariant audit:

```text
/data2/ego_annotation_outputs/v18_status_invariant_audit/
```

`audit_v18_status_invariants.py` checks the generated status manifest, runtime report, HaWoR/WiLoR/RTMLib hand-baseline summary, visibility/occlusion summary, visible part-subset reports, occlusion candidate reports, bounded state summary, physical-state schema, generated-only part-track source manifest, part-split evidence, part visible surfaces, surface-ICP diagnostics, part-object blockers, classified part-mask acquisition blockers, and part-mask acquisition plan. The current audit passes 66 required checks with zero failures.

The audit enforces the main scoped claims: status deliverable ready, final pose-complete deliverable not ready, full-duration/frame-count/FPS checks true, BundleSDF/NeRF absent from the default path, WiLoR measurement count, HaWoR partial coverage recorded without full-video readiness, RTMLib `loaded` status normalized for trash, no HaWoR occlusion pose accepted, object/part pose readiness false, contact and occlusion ownership readiness zero, occlusion depth-order evidence not promoted to ownership, visible part-subset archives not evidence-ready under generated-only evidence, promptable SAM assets and saved proposal masks blocked by the promotion gate rather than treated as accepted referring/open-vocabulary part-mask tracks, generated-only OWLv2→SAM2 source scope/counts, 5 accepted generated part assignments, 809 part-surface rows, 3 non-promoted part-model/articulation probes, 5 surface-ICP diagnostic probes with no pose promotion, 2 world-frame articulation-fit diagnostics with preserved frame-level residual outlier counts, 1 rejected faucet part-SE(3) surface residual, 1 rejected off-white articulation fit, 1 rejected single-part pink-lid probe, fresh raw-video runtime not measured, and cached-evidence-to-status runtime under 10x. The status manifest links the latest audit and reports `status_invariant_audit_passed=true`. The measured runtime orchestrator writes the runtime report, refreshes the manifest, runs the post-report audit, and refreshes the manifest again so the audit observes the same runtime ratio that the final manifest reports.

Remaining gap after this checkpoint: keep this audit in the validation path for future V18 changes; add new required checks when new geometry, pose, contact, or perception-backend stages are introduced.

## Implementation Checkpoint 22: Occlusion Depth-Order Triage Evidence

V18 now writes bounded scene-depth triage for occlusion-owner candidates:

```text
/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence/
```

`build_v18_occlusion_depth_order_evidence.py` consumes the occlusion owner-candidate rows, V18 hand visibility/depth states, and V17 same-frame object visible-surface depth summaries. It classifies each candidate object pair as scene-depth support for a foreground-occluder candidate, scene-depth contradiction, metric-compatible/no foreground signal, insufficient object surface depth, or insufficient/untrusted hand depth. This is candidate triage only: it does not assign an occluder owner, does not resolve object-specific depth ordering, does not fill pose, and does not validate contact.

Current evidence covers 116 owner-candidate hand rows and 166 candidate object pairs. 160 pairs have same-frame object visible-surface depth. The depth triage finds 1 pair where scene depth supports a foreground-occluder candidate but ownership remains unaccepted, 27 pairs where scene depth contradicts a foreground-occluder explanation for the current hand state, 13 pairs with metric-compatible/no foreground occlusion signal, and 125 pairs with insufficient object-surface or hand-depth evidence.

The bounded state solution now projects these counts while keeping `occluder_owner_accepted_rows=0`, `occlusion_depth_order_resolved_rows=0`, and `pose_filled_through_occlusion_rows=0`. The abstract world/status render labels this triage as candidate-only: the current videos draw 1 support label, 6 contradiction labels, 8 metric-compatible/no-foreground-signal labels, and 72 insufficient-evidence labels across the two cases. The status manifest reports `occlusion_depth_evidence_candidate_pair_rows=166`, `occlusion_depth_evidence_foreground_support_pair_rows=1`, `occlusion_depth_evidence_foreground_contradiction_pair_rows=27`, `occlusion_depth_evidence_owner_accepted_count=0`, and `occlusion_depth_evidence_depth_order_resolved_count=0`.

Remaining gap after this checkpoint: convert depth triage into accepted ownership only with object-specific visibility/depth reasoning and a valid temporal hand model; current evidence mostly rules out or fails to evaluate candidate rows rather than solving occlusion.

## Implementation Checkpoint 23: Part-Mask Backend Capability Probe

V18 now distinguishes promptable segmentation assets from a complete referring/open-vocabulary part-mask generation backend in the acquisition status:

```text
/data2/ego_annotation_outputs/v18_part_mask_acquisition_plan/
```

`build_v18_part_mask_acquisition_plan.py` still finds no runnable SAMWISE repo/checkpoint. A broader probe finds promptable segmentation assets: `third_party/sam2`, `/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt`, the `segment_anything` Python package, and `/home/yiwen/ego_annotation/checkpoints/sam_vit_b_01ec64.pth`. CUDA, torch, and cv2 are available, so `promptable_segmentation_backend_available=true`.

This checkpoint originally identified the missing piece as the absence of a model-produced part prompt plan. Checkpoint 26 implements that plan with OWLv2 keyframe detections and SAM2 video tracking. The current manifest now records `open_vocab_detector_backend_cached_available=true`, `open_vocab_or_referring_prompt_backend_available=true`, `model_produced_part_prompt_plan_ready=true`, `local_new_mask_generation_ready=true`, `local_new_mask_generation_ready_count=3`, and `mask_evidence_created_count=5`.

Remaining gap after Checkpoint 26: use the generated OWLv2→SAM2 tracks to build part-model candidates, reconstruction residuals, contact/depth ownership, and final graph factors; do not stop at mask evidence.

## Implementation Checkpoint 24: Promptable SAM Proposal Probe

V18 now has a bounded promptable-SAM proposal probe for the part-mask blocker objects:

```text
/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals/
```

`build_v18_sam_promptable_part_proposals.py` uses the local SAM ViT-B checkpoint through the `segment_anything` package. For each part/relative-motion object with whole-object masks, it selects up to three depth-backed visible frames, samples generic prompt points inside the whole-object mask, and asks SAM for mask proposals. This is a uniform promptable segmentation probe, not a referring/open-vocabulary part tracker.

Current probe counts: 3 objects, 9 selected frames, 39 prompt points, 117 raw SAM masks, 49 within-object proposal candidates, and 40 saved proposal-mask files for QC. The output sheets are written per case under `/data2/ego_annotation_outputs/v18_sam_promptable_part_proposals/`. All promotion counts remain zero: `accepted_part_track_count=0`, `semantic_part_label_ready_count=0`, `mask_evidence_created_count=0`, `part_pose_ready_count=0`, and `object_pose_requirement_met_count=0`.

The status manifest records `sam_promptable_saved_proposal_mask_count=40`, `sam_promptable_not_referring_part_track_count=49`, `sam_promptable_accepted_part_track_count=0`, and `sam_promptable_mask_evidence_created_count=0`. The invariant audit checks that saved SAM proposal masks are not treated as accepted part tracks or created mask evidence.

Remaining gap after this checkpoint: turn proposals into usable part tracks only with semantic/referring prompt evidence and temporal association/QC; promptable proposals alone are not enough for part split, pose, or contact.

## Implementation Checkpoint 25: Promptable Proposal Promotion Gate

V18 now gates promptable SAM proposal masks before they can affect part evidence:

```text
/data2/ego_annotation_outputs/v18_part_mask_promotion_gate/
```

`build_v18_part_mask_promotion_gate.py` joins the part-object blockers, promptable SAM proposal probe, and acquisition/backend status. All 3 part/relative-motion objects have saved promptable proposal masks, but all 3 are classified as `blocked_promptable_proposals_need_semantic_temporal_validation`. The gate blockers are generic for the promptable proposal probe: promptable SAM proposals are not referring part tracks, no temporal part-track association exists, and no semantic part label is attached. The separate OWLv2→SAM2 path now provides the model-produced temporal part tracks; the promptable SAM proposals remain unpromoted.

The gate records `saved_promptable_proposal_mask_count=40` and `objects_with_saved_promptable_proposals_count=3`, while preserving `promoted_part_track_count=0`, `mask_evidence_created_count=0`, `part_geometry_extraction_ready_count=0`, `part_pose_ready_count=0`, `object_pose_requirement_met_count=0`, and `contact_ownership_ready_count=0`. The status manifest and invariant audit now carry these fields.

Remaining gap after this checkpoint: add semantic/referring prompt evidence and temporal association/QC before any promptable proposal can become an accepted part track consumed by part split, geometry, motion, or contact stages.

## Design Goal

Build a full raw-video hand-object interaction annotation pipeline whose default path is fast, occlusion-aware, and honest about unresolved geometry. The direction remains hand detection + object detection + consistency optimization, but every required stage must have bounded cost and every inferred state must carry visibility and uncertainty.

The pipeline must output full-duration videos with the same frame count and duration as the original raw video:

1. Annotated raw-video overlay with hands, object instances, depth-backed object surfaces or unresolved-object markers, contact/occlusion status, uncertainty, and semantic captions.
2. Full-duration 3D world animation with head camera, MANO hands, object visible/deformable surfaces, contact hypotheses, occlusion ghosts, and uncertainty.
3. Full-duration side-by-side presentation of annotated video and 3D reconstruction with captions and explicit status flags.

A V18 result is not allowed to call object pose complete unless manipulated-object geometry is actually reconstructed. When only visible depth-backed surface exists, the artifact must say `visible_surface_only`, `hidden_geometry_unresolved`, and keep `object_geometry_complete=false`. This is an honest pipeline output, not closure of the object-pose requirement.

## Runtime Contract

Runtime is a design invariant, not an afterthought.

- Target end-to-end wall time: same order of magnitude as video duration.
- Initial V18 hard ceiling for representative clips: no more than 10x real time end-to-end on available workstation/A800 resources. A 60-second clip should finish in minutes, not hours.
- Stretch target after the first working prototype: 1-3x real time for a 60-second clip.
- Any required stage whose expected runtime exceeds the hard ceiling is not in the default pipeline.
- Per-instance neural-field optimization, NeRF/SDF training loops, BundleSDF-style optimization, and all-face CPU raster sweeps are disallowed in the default path. They may exist only as offline research branches that cannot block delivery or be used to discover obvious physical state types.
- No foreground sleep/poll loops are allowed for progress waiting. Long work must run in tmux or a job system with sentinel files/logs while the agent does useful parallel work or returns control.

Every run writes a runtime manifest with per-stage wall time, device, frame count, object-frame count, and whether each stage stayed within budget.

## Raw-Video Input Contract

For each case the pipeline consumes:

- Raw RGB video or source-resolution frame manifest.
- Frame count, fps, timestamps, and source image dimensions.
- Camera intrinsics if known; otherwise an explicit estimated-intrinsics state with uncertainty.
- Optional existing head/camera trajectory; if absent, a fast visual-inertial/visual-only trajectory estimate with uncertainty.
- Optional metric depth archive; if absent, a fast monocular metric-depth pass with cached per-frame intrinsics/scale confidence.
- Representative V18 cases: `trash_1050` and `task5_tomato_960` from `/data2/egoscale_demo_30h/` and the existing V16/V17 artifacts only as reusable evidence, not as accepted annotations.

## Full-Timeline State Variables

V18 state is per frame unless otherwise noted.

### Camera and Depth

- `T_world_camera[t]` with uncertainty.
- `K[t]` or selected camera-intrinsics model with uncertainty.
- Metric depth map `D[t]` with confidence, depth-edge bands, and invalid/low-confidence masks.

### Hands

For each hand side/track:

- Existence state: `visible`, `partially_visible`, `occluded`, `out_of_frame`, or `unresolved`.
- 2D keypoints, boxes, and hand masks with source/confidence.
- MANO pose/shape/translation when observed or inferred.
- Camera-depth alignment variables and uncertainty.
- Temporal track identity and infill source.
- Occluder owner when hidden by an object or another hand.

### Objects

For each model-produced object id:

- Semantic label/caption and physical state type proposed by VLM/LLM/perception: `rigid`, `deformable`, `articulated`, `container/context`, `fluid/loose`, or `unknown`.
- Existence/visibility state: `visible`, `partially_visible`, `occluded`, `out_of_frame`, or `unresolved`.
- Open-vocabulary detection boxes, SAM2 masks/tracks, mask confidence, and identity continuity.
- Depth-backed visible surface mesh/surfels for observed pixels.
- Material/feature tracks on visible surfaces.
- Rigid SE(3) only when cheap motion residuals support a rigid state.
- Deformation field or per-frame surfel motion for non-rigid states.
- Hidden/completed geometry state only if produced by a bounded feed-forward prior or directly supported by observations; otherwise explicitly unresolved.

### Contact and Occlusion

- Pairwise hand-object contact candidates and owner variables.
- Contact mode per hand-object pair: `contact`, `near`, `not_contact`, `occluded_contact_possible`, or `unresolved`.
- Occlusion ownership graph with depth-order evidence: what hides what, and for how long.
- Uncertainty/covariance for every inferred-through-occlusion hand/object state.

## Perception Sources

Differences between cases must come from model outputs, not hand-written visual case branches.

Default sources:

- VLM/LLM object-plan pass for object roster, active intervals, physical state type, and likely occlusion/contact semantics.
- Open-vocabulary detector or grounding model for object boxes.
- SAM2 or equivalent video segmentation/tracking for object masks and identity continuity.
- RTMLib-style 2D hand keypoints and boxes.
- HaMeR/WiLoR-style per-frame MANO measurements where visible.
- HaWoR or another temporal hand-motion model only after a validation contract proves infill behavior; until then it is a candidate temporal measurement, not an accepted occlusion solution.
- Fast metric depth and depth-edge ownership masks.
- Optical-flow/material point trackers for object surface motion and occlusion consistency.

Disallowed default sources:

- BundleSDF/NeRF/test-time neural reconstruction loops.
- Category-specific if/else code for bags, cans, tomatoes, peels, faucets, colors, or action phrases.
- Silent fallback from missing geometry to centroids, spheres, boxes, patches, or masks as if object pose were solved.

## Fast Object Geometry Path

V18 object geometry is state-aware and bounded-cost.

1. Always build a per-frame visible surface from mask + metric depth when available.
2. Track surface points/features across adjacent frames with bounded optical flow/material tracking.
3. Use VLM physical state type as a prior, then revise it with cheap residuals:
   - rigid hypothesis: adjacent-frame Procrustes/ICP residuals, surface-track consistency, scale stability;
   - deformable hypothesis: local surface tracks explain motion but no single SE(3) explains the object;
   - occluded/unresolved hypothesis: insufficient visible surface or depth.
4. For rigid-supported objects, build a lightweight surfel/TSDF-style canonical visible geometry and per-frame pose under a strict iteration/time cap.
5. For deformable objects, maintain per-frame visible geometry plus surface motion/deformation variables. Do not force a canonical rigid pose.
6. For hidden geometry, use only bounded feed-forward priors or observed multi-view fusion. If hidden geometry is not supported, mark it unresolved and render uncertainty.

This path is allowed to produce useful partial geometry quickly. It is not allowed to claim complete object pose when the object geometry is only visible surface.

## Occlusion Model

Occlusion is a first-class inference target.

For each hand/object track, V18 estimates a visibility state and occluder owner. The model uses:

- Detector confidence drop or mask disappearance.
- Depth ordering: foreground object/hand depth relative to the predicted hidden track.
- Boundary/depth-edge ownership masks.
- Before/after temporal continuity.
- Contact persistence when supported by pre/post evidence.
- Nonpenetration constraints.
- Maximum occlusion duration thresholds; long occlusions become unresolved, not hallucinated exact pose.

Short occlusion infill is permitted only with uncertainty. Rendered hidden states must be visually distinct from observed states, e.g. translucent/ghosted MANO or object surfaces with covariance bands and labels.

## Optimization Objective

The V18 optimizer is a bounded robust factor graph or equivalent fixed-iteration smoother. It must run under the runtime contract.

Variables:

- Camera/depth scale/intrinsics adjustments within small priors.
- Hand pose/translation/depth alignment and visibility state.
- Object visible-surface state, rigid pose or deformation state, and visibility state.
- Contact ownership/mode and occlusion ownership.

Factors:

- 2D hand keypoint/box/mask reprojection.
- MANO depth against interior-owned metric depth, excluding depth-edge bands.
- Object mask/depth visible-surface residuals.
- Surface-track temporal consistency.
- Rigid/deformable state residuals from cheap motion tests.
- Occlusion depth ordering and visibility transition costs.
- Contact proximity, contact persistence, and nonpenetration.
- Temporal smoothness with robust losses and switch variables.

The optimizer must expose unresolved rows and rejected factors. It must not hide broken contracts with silent fallbacks.

## Implementation Checkpoint 26: OWLv2→SAM2 Semantic Part Tracks

V18 now has the first executable baseline object/part perception stage required by the binding pipeline:

```text
/data2/ego_annotation_outputs/v18_owlv2_sam2_part_tracks/
```

`build_v18_owlv2_sam2_part_tracks.py` uses the V18 physical-state schema notes as the model-produced source for part terms, runs cached OWLv2 on selected object keyframes, prompts SAM2 video tracking with accepted OWLv2 boxes, and writes accepted tracks in the downstream `sam2_track.json` format. SAM v1 is not used in this stage.

The full two-case run produced 109 OWLv2 candidate part boxes and 5 accepted semantic temporal SAM2 part tracks, with 1 rejected track. Accepted tracks are:

- `object:off_white_trash_can_first`: `owlv2_sam2_off_white_trash_can_first_hinge` and `owlv2_sam2_off_white_trash_can_first_lid`.
- `object:pink_lid_trash_can_second`: `owlv2_sam2_pink_lid_trash_can_second_lid`.
- `object:obj_faucet_handle`: `owlv2_sam2_obj_faucet_handle_handle` and `owlv2_sam2_obj_faucet_handle_lever`.

`build_v18_part_track_source_manifest.py` now defaults to these generated accepted-track roots only. Legacy cached roots are opt-in debug inputs and make the source pool non-uniform. After rerunning the downstream part chain with generated-only evidence, V18 has 5 usable generated tracks and 5 accepted part-track assignments. Off-white can and faucet handle each have two accepted generated tracks; pink-lid can has one generated track and is explicitly `single_part_mask_evidence_insufficient_for_split`. Part visible-surface extraction now produces 809 depth-backed part surface rows, 302,902 vertices, and 525,741 faces after target-support small-mask depth sampling.

This is real part-mask and visible-geometry progress, not pose completion. The current downstream blocker has moved: off-white can and faucet handle are no longer blocked by missing accepted part masks; faucet has a residual-supported visible-center articulation fit but fails part-surface SE(3) residual validation, while off-white has a rejected articulation fit. Pink-lid can has generated mask evidence but only one part track, so its single-part surface probe is rejected as insufficient for a split/articulation model. Hidden geometry, part pose, contact ownership, occlusion ownership, and final pose-complete readiness remain false.


## Implementation Checkpoint 27: HaWoR/WiLoR/RTMLib Hand Baseline Branch

V18 now writes an explicit hand-baseline branch artifact:

```text
/data2/ego_annotation_outputs/v18_hand_baseline_branch/
```

`build_v18_hand_baseline_branch.py` joins full-timeline WiLoR visible-frame measurements, HaWoR temporal hand measurements, RTMLib 2D keypoint anchors, and the interior-owned hand/depth graph. `build_v18_visibility_occlusion_state.py` now consumes this branch and normalizes RTMLib measurement-manifest status `loaded` as a valid source status. This fixes the previous silent omission where the trash RTMLib file existed but `rtmlib_hand2d` was reported as `None`.

Observed hand-branch evidence across the two representative cases: 4,020 hand rows, 3,361 WiLoR measurement rows, 182 HaWoR rows, 132 available HaWoR visible measurements, 50 HaWoR motion-infill candidates, 1 RTMLib-loaded case, 1,041 RTMLib frames with hands, and 1,551 RTMLib/WiLoR comparison rows. HaWoR coverage is trash-only and limited to frames 840--930; task5 has no HaWoR rows. Strict full-video readiness requires an available HaWoR measurement for every frame/hand-side, and the current run has only 132 available HaWoR frame-side measurements out of 4,020 required frame-side slots.

At this hand-baseline checkpoint no temporal occluded hand pose was accepted: `hawor_full_video_baseline_ready_all_cases=false`, `hawor_temporal_occlusion_pose_accepted_count=0`, and `pose_filled_through_occlusion_rows=0`. The explicit blockers are missing full-video HaWoR coverage and missing score components for metric-depth absolute residual, temporal acceleration, and hand bone-scale error. This statement is superseded only for the later observed-MANO path: the latest final artifact accepts three trash pose-fill rows from final depth-scaled same-frame HaWoR MANO plus accepted occluder depth order, while temporal hand-baseline pose remains zero.

This checkpoint restores HaWoR as an explicit V18 baseline source, but it does not satisfy the binding full-video HaWoR hand contract. The next hand step for temporal fill is to generate or recover full-video HaWoR-equivalent temporal measurements and validate them against WiLoR, RTMLib, and metric depth before temporal occlusion pose fill is allowed.


## Implementation Checkpoint 28: Rejected Part-Model Residual Probes

V18 now preserves non-promoted part-model evidence instead of collapsing all generated part surfaces to a generic missing-candidate blocker. `build_v18_part_model_candidates.py` writes robust articulation-hypothesis probes for off-white/faucet and a rejected single-part residual probe for pink-lid when the generated surfaces are insufficient for an accepted model:

- `object:off_white_trash_can_first`: a robust variable two-part pair becomes an articulation hypothesis, but its world-frame circle fit is rejected by radial/plane residuals.
- `object:pink_lid_trash_can_second`: a single-part visible-surface probe is rejected because a part/articulation model requires at least two semantic part tracks.
- `object:obj_faucet_handle`: a world-frame center-circle fit is supported, but the shared-frame handle surface SE(3) residual fails the p95 threshold.

The current summary records `part_model_candidate_count=0`, `part_model_rejected_candidate_count=3`, `part_surface_icp_probe_count=5`, `part_surface_icp_probe_state_counts={surface_icp_residual_supported_visible_only_not_pose: 5}`, `part_articulation_hypothesis_pair_count=2`, `part_pose_ready_count=0`, and `object_pose_requirement_met_count=0`. `build_v18_part_object_blocker_manifest.py` now reports faucet as `blocked_part_se3_surface_residual_rejected`, off-white as `blocked_articulation_fit_residual_rejected`, and pink-lid as `blocked_part_model_residual_probes_rejected`, not as missing mask evidence or accepted part models.

This is real residual-testing evidence, but it is not pose evidence. It narrows the next object/part interventions: repair or reject the off-white articulation relation, repair faucet handle SE(3) residual outliers before silhouette/depth validation, and obtain additional pink-lid semantic part evidence before attempting part pose, hidden geometry, contact ownership, or factor-graph promotion.


## Implementation Checkpoint 29: Adaptive Small-Mask Part Surface Sampling

`build_v18_part_visible_surfaces.py` now uses explicit adaptive sampling for small part masks. The previous fixed 8-pixel stride undersampled small hinge/lever masks even when valid metric depth existed. The new extractor tries smaller strides within the valid mask bounding box, records `mask_stride_requested`, `mask_stride_used`, and `sampled_vertex_count_before_connectivity`, and still rejects rows that fail vertex/face connectivity thresholds.

This increases accepted depth-backed part surface rows from 753 to 809 and removes all task5 faucet surface-extraction rejections: faucet handle has 59 surface rows and faucet lever has 41. Trash off-white hinge increases from 248 to 259 rows; the only remaining trash rejections are missing metric depth frames.

The adaptive sampler improved visible part geometry evidence but did not by itself make any part model valid. A later target-support sampling policy supersedes the sparse-small-part diagnosis by continuing to smaller strides until robust surface support is reached or stride 1 is exhausted.


## Implementation Checkpoint 30: Surface-Level ICP Diagnostics for Rejected Part Models

`build_v18_part_model_candidates.py` now attaches bounded surface-level ICP probes to non-promoted part-model/articulation probes. Each probe aligns sampled depth-backed part-surface vertices from selected frames to a reference part surface and reports median/p95 residuals, support counts, and explicit blockers. The coordinate frame is per-frame metric-depth camera; ICP estimates only visible-shape consistency, not world pose or articulation.

Current generated-only evidence produces 5 ICP probes, all residual-supported visible-only after target-support surface sampling. A later world-frame articulation diagnostic fits the two robust variable pairs: faucet is residual-supported as a visible-center circle fit, while off-white is rejected by radial/plane residuals. Pink-lid still has only one semantic part track.

No readiness is promoted. The manifest reports `part_surface_icp_probe_count=5`, `part_surface_icp_probe_state_counts={surface_icp_residual_supported_visible_only_not_pose: 5}`, `part_articulation_hypothesis_pair_count=2`, `articulation_fit_probe_count=2`, `articulation_fit_supported_count=1`, `articulation_fit_rejected_count=1`, `part_se3_pair_count=2`, `part_se3_pair_rejected_count=1`, `part_model_candidate_count=0`, `part_model_rejected_candidate_count=3`, `part_pose_ready_count=0`, and `object_pose_requirement_met=false`. The invariant audit checks these fields and requires zero unclassified part-mask acquisition blockers.


## Implementation Checkpoint 31: Target-Support Part Surface Sampling

`build_v18_part_visible_surfaces.py` no longer stops as soon as a part surface satisfies the minimal 8-vertex/6-face acceptance threshold. It now continues to smaller mask strides until the surface reaches a robust target of 100 vertices and 100 faces or stride 1 is exhausted. This preserves the minimal rejection gate while avoiding an extractor-induced sparse-small-part artifact.

The current generated-only part surface archive still has 809 accepted frame rows and 5 rejected rows, but the geometry support increases to 302,902 vertices and 525,741 faces. All five semantic part tracks now meet the surface-ICP visible-only residual support state. Off-white can and faucet handle move from sparse/confounded variable-pair blockers to robust variable-pair articulation hypotheses. Pink-lid remains blocked by single-part evidence.

This is not pose completion. V18 still reports `part_model_candidate_count=0`, `part_articulation_hypothesis_pair_count=2`, `articulation_model_ready_count=0`, `part_pose_ready_count=0`, `contact_ownership_ready_count=0`, and `object_pose_requirement_met=false`. The next causal step is split: repair/rethink the off-white articulation model or evidence because its center-circle fit is rejected; for faucet, repair the handle surface SE(3) residual outliers before silhouette/depth residual validation and hidden-geometry reasoning. The acquisition plan classifies these as articulation/pose-validation blockers rather than unclassified mask-acquisition failures.


## Implementation Checkpoint 32: World-Frame Articulation Fit Diagnostics

`build_v18_articulation_fit_candidates.py` fits bounded world-frame circle/hinge residual diagnostics for robust variable part-pair hypotheses. It transforms part visible-surface centers from metric-depth camera coordinates into the V16 `T_world_camera_metric` world frame, fits a 3D circle to the relative part-center vectors, and records radial/plane residuals, radius, angle span, and blockers. The fit scope is visible part centers only; it is not full part SE(3), hidden geometry, contact, or object pose.

The current two articulation probes split the next work. `object:obj_faucet_handle` is `articulation_fit_residual_supported_visible_center_only_not_pose` with 39 shared frames, radius about 0.136 m, radial residual median about 1.0 cm, radial p95 about 2.5 cm, and plane p95 about 1.4 cm. `object:off_white_trash_can_first` is `articulation_fit_residual_rejected`: it has 259 shared frames, radial p95 about 8.1 cm and plane p95 about 5.1 cm, above the thresholds. The articulation report now preserves frame-level residual rows: off-white has 37 radial-threshold outlier frames, 73 plane-threshold outlier frames, and 95 combined outlier frames; the worst normalized residual is frame 449 with plane residual about 19.6 cm. This makes the off-white failure a broad residual problem, not a single-frame artifact.

No readiness is promoted. After part-surface SE(3) validation, blockers are `blocked_part_se3_surface_residual_rejected` for faucet, `blocked_articulation_fit_residual_rejected` for off-white, and `blocked_part_model_residual_probes_rejected` for pink-lid. The manifest reports `articulation_fit_probe_count=2`, `articulation_fit_supported_count=1`, `articulation_fit_rejected_count=1`, `part_se3_pair_count=2`, `part_se3_pair_rejected_count=1`, `articulation_model_ready_count=0`, `part_pose_ready_count=0`, `contact_ownership_ready_count=0`, and `object_pose_requirement_met=false`.


## Implementation Checkpoint 33: Part-Surface SE(3) Residual Diagnostics

`build_v18_part_se3_surface_residuals.py` adds the next validation layer after visible-center articulation fitting. It transforms depth-backed part vertices to the V16 `T_world_camera_metric` world frame, registers selected shared-frame observations of each part surface to a reference surface by rigid ICP, and evaluates whether the part surfaces themselves support a visible-only SE(3) track. This is still diagnostic evidence only: no per-frame part pose is accepted, and no contact or object pose is promoted.

The current result blocks the only center-supported articulation before pose. Faucet has two part surfaces evaluated on the 39 shared frames used by the articulation fit: the lever passes the SE(3) surface residual diagnostic, but the handle fails because the p95 over per-frame p95 residuals is about 9.55 cm, above the 6 cm threshold. The report exhaustively evaluates all 39 shared frames for this pair. It preserves eight threshold-exceeding handle frames (919, 921, 922, 924, 925, 927, 928, 933), worst p95≈14.2 cm at frame 921, and their residual/depth/containment metadata; the audit checks that this exhaustive shared-frame outlier evidence remains present. The lever passes the p95-of-p95 criterion but records one per-frame p95 exceedance at frame 926, so it is not a pose-ready part either. Off-white is not evaluated as a pair because its world-frame articulation fit is already rejected, although both individual surfaces can register to references. Pink-lid remains a single-part rejected-probe case.

The manifest reports `part_se3_pair_count=2`, `part_se3_pair_rejected_count=1`, `part_se3_surface_supported_count=3`, `part_se3_surface_rejected_count=1`, `part_pose_ready_count=0`, `contact_ownership_ready_count=0`, and `object_pose_requirement_met=false`. The next faucet intervention is to repair/validate handle surface SE(3) outliers before silhouette, depth, hidden-geometry, or contact reasoning.

## Implementation Checkpoint 34: Artifact-First Reset

V18 no longer treats arbitrary gates as artifact-production blockers. Every named stage may write a best-effort candidate or explicit unresolved state into the backing annotation data so the full video remains inspectable. That is an honesty mechanism, not completion, and it matters only if it is visible in, or needed to reproduce, the rendered annotations. A stage is strict-checklist DONE only when the actual named mechanism runs, is sanity-checked on video/geometric evidence, drives downstream state where applicable, and is rendered in the final videos. Residuals remain useful for explanation and subjective judgment, but approximate outputs do not permit approximate implementation of the spec.

The next required implementation artifact is the rendered video annotation set:

```text
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_overlay.mp4
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_world.mp4
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_side_by_side.mp4
```

Backing data and evidence live beside those videos:

```text
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/annotations_v18_full.json
/data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json
```

The report must compare delivered V18 videos against V16 by subjective judgment: what is no worse, what visibly improves, what remains approximate/uncertain, and what implementation mistakes were fixed. Diagnostics are allowed only when they directly repair the rendered annotation path or explain the delivered videos.

## Implementation Checkpoint 35: Full Approximate Pipeline Artifact

V18 now has an executable full-pipeline rendered-annotation writer:

```text
scripts/run_v18_full_pipeline.py
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/annotations_v18_full.json
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_overlay.mp4
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_world.mp4
/data2/ego_annotation_outputs/v18_full_pipeline/<case>/v18_side_by_side.mp4
/data2/ego_annotation_outputs/v18_full_pipeline/v18_full_pipeline_report.json
```

This is the artifact-first V18 baseline, not strict V18 physical closure. It assembles the existing V16 camera/depth backbone, V18 hand evidence, generated OWLv2→SAM2 object/part masks, visible geometry archives, part surfaces, bounded occlusion candidates, approximate contact hypotheses, approximate hidden-geometry candidates, approximate object/part pose candidates, and a single-pass bounded factor-graph baseline into full-video rendered annotations with backing JSON. Every output is approximate and uncertain by design. No arbitrary threshold suppresses artifact production, but JSON presence, labels, or unresolved/candidate rows are not evidence that the substantive mechanisms are implemented.

The delivered run completed both representative cases in about 108 seconds total after rendering. Frame counts match raw videos: `trash_1050` has 1050 frames and `task5_tomato_960` has 960 frames in the overlay video, world video, side-by-side video, and backing JSON. The backing JSON contains named module families, but those fields are not the artifact and do not prove the modules are substantively implemented; the rendered annotations and their causal mechanisms are what matter.

Visual inspection against V16 side-by-side frames shows the current tradeoff. V16 still has denser mesh-style hand/object rendering for selected manipulation objects. V18 is now plausibly no worse for frame coverage and 2D localization, and it adds explicit multi-object state, part candidates, uncertainty labels, contact/occlusion hypotheses, hidden-geometry candidates, and factor-graph state in the delivered JSON/video instead of hiding them in diagnostics. The next improvement should be render fidelity and solver quality, not another readiness gate: use the delivered videos to decide which approximate modules are visibly wrong, then patch those modules.

## Implementation Checkpoint 36: Monotonic V16 Render Integration

V18 full-pipeline rendering now preserves V16 visual capabilities instead of replacing them. `scripts/run_v18_full_pipeline.py` extracts V16 `overlay_mano_object.mp4` and `reconstruction_3d_world.mp4` frames and uses them as the base layers for V18 overlay/world videos. V18 masks, part candidates, contact/occlusion labels, uncertainty, and graph overlays are drawn on top.

This fixes the previous degradation where V18 replaced V16's MANO/object mesh and metric-world render with boxes and an abstract world panel. The artifact report and per-case QC now record the base V16 overlay/world paths. The full representative run passed frame-count equality: trash 1050/1050 and task5 960/960 for annotation JSON, overlay, world, and side-by-side videos. Visual inspection of trash frame 330 and task5 frame 780 confirms V16 render content remains visible while V18 layers are additive. This is a render monotonicity repair, not evidence that the missing V18 solver modules are complete.

## Implementation Checkpoint 37: Numerical Observable-State Factor Graph WIP

`run_v18_full_pipeline.py` now materializes an explicit numerical factor graph instead of the earlier string/count placeholder. The graph contains named variables for camera/depth scale, hands, object SE(3), part visible centers, articulation coordinates, contact switches, and occlusion-owner choices. Continuous observed tracks are solved with SciPy sparse linear systems over observation and adjacent-frame temporal factors. Discrete contact and occlusion-owner variables are assigned by explicit per-candidate energies. The artifact records variable counts, factor counts, per-series energy before/after inference, and per-frame local objectives.

This is meaningful progress only for the factor-graph mechanism, and its scope is bounded. Object SE(3) observations now use visible-surface translation plus PCA-derived rotation vectors when the metric visible point cloud supports them. Part rotations are still not solved because the current part-surface artifact records center/extent/counts rather than part point coordinates. Camera/depth correction is still a prior-only identity variable. Contact and occlusion energies still consume current overlap/depth-candidate evidence and do not yet implement full geometry nonpenetration or accepted depth-order ownership. Therefore this checkpoint is not the completed V18 physical graph; it is the first explicit numerical graph that future modules can replace/refine.

## Implementation Checkpoint 38: Depth-Fused Visible Geometry Reconstruction

V18 now builds a real geometry reconstruction artifact in `scripts/build_v18_depth_fused_reconstruction.py`. It fuses metric visible-surface vertices into graph-object coordinates using the V18 object SE(3) estimates, writes per-object fused point clouds, and reconstructs Poisson and convex-hull mesh candidates. The current full-pipeline default consumes the complete-depth pass-2 reconstruction at `/data2/ego_annotation_outputs/v18_unidepth_extension/v18_depth_fused_reconstruction_complete_depth_pass2`; this is still visible-surface fusion, not hidden-geometry completion. The default path does not use BundleSDF, NeRF, or test-time neural reconstruction. `scripts/validate_v18_depth_fused_reconstruction.py` checks that point clouds/meshes exist while also requiring the artifact not to claim complete hidden geometry. `scripts/render_v18_depth_fused_reconstruction_sheet.py` renders QC sheets for subjective inspection.

The reconstruction is now integrated back into `run_v18_full_pipeline.py`: hidden-geometry candidates prefer the depth-fused reconstruction report and fall back to the older PCA mirror only when the fused artifact is absent. This is a stronger geometry artifact than the previous PCA mirror, but it still does not satisfy completed hidden/full object geometry. The current mesh candidates are visible-depth fusion/completion candidates with outliers and incomplete hidden surfaces; object_geometry_complete and hidden_geometry_reconstructed remain false.

## Implementation Checkpoint 39: V16 Mesh-Distance Contact Evidence

V18 now has a mesh-distance contact evidence artifact in `scripts/build_v18_mesh_contact_evidence.py`. The initial attempt to compare V18 depth-fused object clouds with MANO world hands exposed a coordinate mismatch: V18 visible geometry was hundreds of millimeters from V16 MANO hands even in known contact frames. The module was revised to use the monotonic V16 metric source: V16 MANO hand surface vertices and the V16 measured object mesh archive. For each frame, the V16 single-object mesh is associated to the V18 object with maximum bbox overlap, and existing V18 hand-object contact hypotheses receive metric hand-surface-to-object-mesh distances and support scores.

This is contact evidence, not accepted contact ownership. The artifact intentionally keeps `contact_ownership_accepted_rows = 0`; it does not solve full hand-object nonpenetration or multi-object ownership. It does, however, expose useful contradictions: some image/depth contact rows previously rejected by the older depth heuristic have sub-millimeter V16 mesh distances. `run_v18_full_pipeline.py` now includes this mesh-contact evidence under each contact hypothesis when available.

## Implementation Checkpoint 40: Occlusion Owner Evidence From Mesh Contact

V18 now has an occlusion-owner evidence artifact in `scripts/build_v18_occlusion_mesh_owner_evidence.py`. It combines bounded occlusion-owner candidates with nearby V16 mesh-contact support over a temporal window. This does not accept new owners without depth-order acceptance; it records `not_accepted_owner_without_depth_order_acceptance` when mesh support exists but depth order remains unresolved.

The artifact is integrated into `run_v18_full_pipeline.py` under each hand's `occlusion_owner_hypothesis.mesh_owner_evidence`. The current full-pipeline default uses the complete-depth/Hawor occlusion-owner graph, so strict accepted owner assignments can become final graph variables when the source graph has depth order resolved by HaWoR MANO/object depth evidence, temporal graph selection, sufficient mesh temporal support, and observed same-frame HaWoR support. This is solved ownership for those frames, not pose fill-through-occlusion.

## Implementation Checkpoint 41: Temporal Mesh-Distance Contact Ownership Graph

V18 has a discrete hand-level contact-ownership graph in `scripts/build_v18_contact_ownership_graph.py`. For each hand and frame it chooses one state from `{none} ∪ candidate V18 object ids` using V16 MANO-to-object mesh distances, image/depth evidence penalties, V16-to-V18 mesh association evidence, and temporal continuity. Earlier graph rows were named as accepted by the contact graph, but under the current HaWoR hard-requirement reset they are preserved only as source graph contact candidates/hypotheses.

The historical representative graph produced trash 295 and task5 721 source graph contact-candidate rows. These are not current accepted physical contact ownership: task5 has no HaWoR/metric MANO foundation, local nonpenetration evidence is incomplete/conflicted, and the corrective annotation/audit path keeps strict-promotable contact at zero. Low-association false positives and conflicted rows remain selected/candidate evidence rather than promoted. `signed_nonpenetration_solved=false`, `contact_ownership_complete=false`, and accepted physical contact remain false/zero under the current V18 corrective state.

## Implementation Checkpoint 42: Visible-Surface Part SE(3) From PCA

V18 now consumes the part visible-surface archive `v18_part_visible_surfaces_camera.npz` in `run_v18_full_pipeline.py` instead of using only per-row center/count summaries. For each part surface row with archived vertices, the full pipeline computes a PCA pose observation: camera-frame translation, rotation vector, rotation matrix, anisotropy, singular values, and source archive row. These appear in each part's `pose_candidate` as `approximate_part_visible_surface_pca_se3_candidate`.

The numerical factor graph now uses 6D part SE(3) observations (`translation_xyz_m_and_rotation_vector_xyz_rad`) when these PCA rotations exist. This advances the part pose mechanism beyond center-only translation. The scope remains limited: PCA orientation is visible-surface-only, sign-canonicalized, and not a canonical full part pose through occlusion or hidden geometry.

## Implementation Checkpoint 43: Integrated Hand Baseline Evidence

V18 now carries the hand-baseline branch rows into the final full annotations. `run_v18_full_pipeline.py` reads `v18_hand_baseline_branch.json` and attaches each hand's HaWoR/WiLoR/RTMLib/interior-depth state, score components, blockers, and pose claim under `hand_baseline_branch`. The full artifact therefore preserves why HaWoR is present or absent for every frame-side rather than reducing the branch to a boolean.

This is evidence integration, not accepted occluded-hand pose fill. Current outputs still keep `temporal_occlusion_pose_accepted=false` and `pose_claim=no_occluded_pose_accepted_from_current_hand_baseline`; trash has HaWoR candidate rows, while task5 has explicit HaWoR-missing blockers.

## Implementation Checkpoint 44: Observed Camera/Depth Scale Correction

V18 now has a camera/depth correction artifact in `scripts/build_v18_camera_depth_correction.py`. It samples the reused depth backend at V16 object centers and estimates a per-frame depth-scale correction from `V16 object depth / backend depth patch median`, then smooths log-scale observations temporally. `scripts/validate_v18_camera_depth_correction.py` checks full-timeline rows, direct observation counts, positive scale estimates, and lower energy than the identity prior.

`run_v18_full_pipeline.py` now loads these correction rows into the factor graph as `camera_depth_correction` variables with observation/interpolation state. This replaces the previous identity-only prior with observed scale evidence. The scope remains limited: this is a scale correction for the reused backend, not a new SLAM solve, dense depth refit, or proof that all depth/camera geometry is accurate.

## Implementation Checkpoint 45: Local Signed Nonpenetration Evidence

V18 has local signed-distance evidence in `scripts/build_v18_signed_nonpenetration_evidence.py`. For historical source graph contact-candidate rows, it loads the V16 object mesh faces and V16 hand vertices, orients face normals outward from the mesh centroid, and records nearest-face normal-projection signed distances. `scripts/validate_v18_signed_nonpenetration_evidence.py` checks that the evidence is present and that it does not claim complete nonpenetration.

The current evidence often flags local penetration, especially where meshes are thin, deformable, open, or normal projection is only a local approximation. Therefore the artifact is integrated into contact hypotheses as evidence only. It is not a watertight signed-distance field and not a complete nonpenetration solver. Physical contact acceptance is vetoed when this local evidence reports penetration, so the contact graph remains preserved as conflicted evidence instead of being promoted to accepted ownership.

## Implementation Checkpoint 46: Temporal Occlusion Owner Graph

V18 now has a temporal occlusion-owner graph in `scripts/build_v18_occlusion_owner_graph.py`. It solves an object-or-none sequence over bounded occlusion candidates using box coverage, candidate overlap, nearby mesh-contact support, and temporal continuity. The graph selects likely owners as evidence but only accepts ownership when source depth-order evidence already accepted it.

Current representative outputs use the complete-depth/Hawor strict graph for final occlusion-owner variables. It accepts 3 trash ownership states and 0 task5 ownership states; larger gaps still reset to unary candidate evidence. `run_v18_full_pipeline.py` integrates each hand/frame assignment under `occlusion_owner_hypothesis.temporal_owner_graph` and exposes accepted states as `accepted_occlusion_owner=true` in final factor-graph variables.

## Implementation Checkpoint 47: Contact Signed-Conflict Veto and Mesh Evidence Provenance

The contact graph still stores its temporal mesh-distance selections, but `run_v18_full_pipeline.py` no longer promotes a graph-selected/source contact candidate to final physical contact ownership when local nonpenetration evidence reports penetration. Checkpoint 47 used signed-normal evidence only; Checkpoint 49 extends the veto with nearest-triangle local signed evidence. Those rows are emitted as `contact_owner_graph_conflicted_by_local_nonpenetration_evidence_not_accepted`, and factor-graph contact switches are forced inactive under the same conflict condition. Historical build-only rows that survived local conflict checks are now superseded by the HaWoR hard-requirement reset and corrective acceptance audit: they are not accepted physical contact while task5 HaWoR is absent and complete nonpenetration is unsolved.

`build_v18_mesh_contact_evidence.py` now snapshots and hashes the full-annotation source used for mesh-distance evidence under each evidence output directory. This does not remove the bootstrap dependency on a full-annotation source, but it prevents the report from depending only on a mutable path after later full reruns overwrite `/data2/ego_annotation_outputs/v18_full_pipeline/<case>/annotations_v18_full.json`.

## Implementation Checkpoint 48: Occlusion Pose Fill Gate

V18 now has an explicit pose fill-through-occlusion gate in `scripts/build_v18_occlusion_pose_fill_gate.py`. The gate joins the HaWoR/WiLoR hand-baseline branch with the temporal occlusion-owner graph. It can accept a filled occluded-hand pose only when occlusion ownership is accepted and the hand baseline has accepted temporal occlusion pose evidence; otherwise it writes blockers.

This checkpoint's historical zero-accepted-fill statement is superseded by the 2026-06-16 pose-fill repair: the latest final artifact accepts three trash observed-MANO pose-fill rows through accepted occluders and still accepts zero temporal fills. The original gate remains relevant as the baseline/temporal-fill blocker ledger; it is no longer the sole acceptance source after the repaired HaWoR bridge.

## Implementation Checkpoint 49: Nearest-Triangle Nonpenetration Evidence

V18 has `scripts/build_v18_triangle_nonpenetration_evidence.py`, a stronger local nonpenetration artifact than the earlier face-centroid normal projection. It samples V16 hand surface points for source graph contact-candidate rows, queries nearest triangle centroids, computes closest points on candidate triangles, and records unsigned closest-triangle distances plus local oriented-normal signed distances. It also records edge-based mesh closure diagnostics.

Representative V16 object meshes are open: triangle evidence reports zero watertight rows for both videos. Historical graph-candidate counts were trash 295 evaluated rows / 293 local triangle penetration rows / 0 watertight rows and task5 721 evaluated rows / 703 local triangle penetration rows / 0 watertight rows. The current corrective state preserves this as contact/nonpenetration evidence only and vetoes physical contact acceptance when signed-normal or nearest-triangle evidence reports local penetration. This is still not a complete SDF or nonpenetration solver, and it cannot override the missing task5 HaWoR/metric-MANO foundation.

## Implementation Checkpoint 50: Temporal Contact Switch Factors

V18 factor-graph contact switches are no longer solved as independent per-frame choices. `solve_v18_factor_graph()` now groups each hand/object contact variable across time and runs a gap-aware binary Viterbi inference with explicit temporal switch penalties. Temporal continuity is applied only across gaps of at most 30 frames, and the on-state remains hard-vetoed when local nonpenetration evidence reports conflict.

Build-only checks after this change produced gap-valid temporal contact factors for adjacent rows within the 30-frame gap window (trash 3256, task5 2276), with no active contact switch carrying a nonpenetration conflict. This is still a candidate contact-switch graph, not proof of physical contact ownership or complete nonpenetration, but it adds a real temporal factor family to the V18 factor graph.

## Implementation Checkpoint 51: Occlusion Owner Factor Evidence Integration

V18 factor-graph occlusion-owner variables now consume the temporal occlusion-owner graph assignment, mesh-contact temporal support, and depth-evidence state already present in final hand annotations. Candidate energies include box overlap, mesh support, temporal selection, foreground-support/contradiction depth states, and an explicit unowned competitor.

This changes factor-graph evidence integration only; accepted occlusion ownership still requires source depth/temporal acceptance, observed same-frame HaWoR support, and an explicit unowned competitor. Current final artifacts have 3 accepted trash occlusion-owner variables and 0 task5 accepted variables.

## Implementation Checkpoint 52: Hand Baseline Score Components

`build_v18_hand_baseline_branch.py` now computes the three previously-missing hand-baseline score components when supporting evidence exists. Metric depth uses the absolute `interior_median_gap_m` from the V17 interior hand graph. Temporal acceleration uses an adjacent-frame second difference of the HaWoR 3D joint centroid. Bone-scale consistency uses per-side median HaWoR 3D bone lengths within the source annotation. Source HaWoR annotation paths are hashed in the report so these evidence terms do not depend only on mutable paths.

The representative evidence in this checkpoint remains partial. Trash has 843 metric-depth component rows, 178 temporal-acceleration rows, and 182 bone-scale rows; task5 has 1083 metric-depth rows but no HaWoR geometry for temporal/bone components. A review caught an over-broad `supported` hand-state label; the final taxonomy only uses `hawor_visible_measurement_score_components_supported_no_occluded_pose_acceptance` when no active blockers remain. This checkpoint's zero pose-fill statement is superseded by the latest observed-MANO pose-fill repair: temporal hand-baseline pose remains zero, but three trash rows now use final depth-scaled HaWoR MANO plus accepted owner depth order as observed pose-fill evidence.

## Implementation Checkpoint 53: Strict Occlusion Owner Acceptance Gate

`build_v18_occlusion_owner_graph.py` now consumes exact pair-level depth-order evidence from `v18_occlusion_depth_order_evidence` instead of relying only on row-level source status. Candidate energies penalize exact foreground-depth contradictions and reward exact foreground-depth support for selection, while each candidate row receives a strict acceptance gate with explicit blockers. The gate requires temporal graph selection, exact foreground-depth support, no same-frame foreground contradiction, sufficient mesh temporal support, temporal margin, resolved source depth order, and source owner acceptance.

This checkpoint's zero-owner/zero-fill counts are superseded by the complete-depth/HaWoR owner graph and latest pose-fill repair. Current final artifacts have three accepted trash occlusion-owner variables and three matching observed-MANO pose-fill rows; task5 remains selected 0 / accepted 0 for owner and pose fill. The strict acceptance-gate design remains: accepted ownership still requires temporal graph selection, exact foreground-depth support, no same-frame foreground contradiction, sufficient mesh temporal support, temporal margin, resolved source depth order, and source owner acceptance.

## Implementation Checkpoint 54: Pose-Fill Owner-Blocker Propagation

`build_v18_occlusion_pose_fill_gate.py` now carries occlusion-owner graph candidate blockers into each pose-fill row. When a hand has owner candidates but no accepted owner, the pose-fill blockers include `occlusion_owner_*` prefixed reasons and the raw owner candidate gate rows are preserved. The final full annotations preserve these fields through `run_v18_full_pipeline.py`.

This blocker-propagation checkpoint is superseded for trash frame 850 by the complete-depth/HaWoR owner graph and observed-MANO pose-fill repair: frame 850 right hand now has accepted owner support for `object:white_trash_bag` and an accepted observed-MANO pose-fill row. The blocker propagation still applies to non-accepted owner candidates and to task5, where pose fill remains zero.

## Implementation Checkpoint 55: Explicit Local Nonpenetration Factor Family

`run_v18_full_pipeline.py` now represents signed-normal and nearest-triangle local nonpenetration evidence as an explicit `contact_local_nonpenetration` factor family in the V18 factor graph. This separates local nonpenetration evidence from the contact-switch unary factor while preserving the existing hard veto: no active contact switch may carry signed or triangle local-penetration conflict.

Representative final factor counts are trash `contact_local_nonpenetration=295` and task5 `contact_local_nonpenetration=721`, matching the available signed/triangle evidence rows. Each factor is explicitly scoped as `signed_normal_and_nearest_triangle_local_evidence_not_watertight_sdf` and `local_nonpenetration_factor_complete=false`. This advances the physical graph evidence ledger without claiming complete nonpenetration or a watertight SDF.

## Implementation Checkpoint 56: HaWoR Bridge Subset Policy Is Candidate-Only

The corrective V18 evidence bundle now separates four HaWoR states:

1. the hard HaWoR requirement state,
2. the trash-only current-V18 camera-local bridge candidate,
3. bridge quality strata, and
4. a conservative subset policy for future candidate-only recomputation.

Artifacts are under:

```text
/data2/ego_annotation_outputs/v18_corrective_1600/hawor_bridge_state/
```

The subset policy keeps only visible projection-supported trash bridge rows with no quality blockers, median residual `<=50 px`, p95 residual `<=100 px`, HaWoR/reference image-inside fraction `>=0.8`, and HaWoR joints inside the current hand bbox fraction `>=0.95`. Current counts: trash strict candidate queue `1297/2098`, existing contact rows in strict queue `223/371`, existing occlusion rows in strict queue `0/165`, and task5 strict queue `0` because task5 has no HaWoR bridge.

This policy is a guardrail, not a physical result: it does not accept HaWoR as the V18 foundation, does not accept metric hand state, and does not recompute or accept contact, occlusion ownership, pose fill, nonpenetration, or V18 closure. It implies that future local downstream work, if attempted before task5 provisioning, should start with candidate-only trash contact recomputation rather than occlusion ownership. Full V18 physical validity still requires real time-indexed HaWoR/metric MANO for all cases, especially task5.

## Implementation Checkpoint 57: Task5 HaWoR Export Contract

V18 now records an explicit task5 HaWoR export contract under:

```text
/data2/ego_annotation_outputs/v18_corrective_1600/hawor_task5_export_contract/
```

The contract names the task5 raw clip, the expected local HaWoR output path, the remote task5 export command, required HaWoR/MANO assets, and the post-ingest validators. It also records the task5 source clip SHA256 `66791eaa646aac2e8cb24bb00fe30b2801436302327b1c46fea650446c41c4ac` plus expected 960-frame/30fps/1920x1080 identity metadata. `remote_run_hawor_export.sh` now supports `EGO_HAWOR_CASE=task5_tomato_960` and refuses export if `EGO_HAWOR_CLIP_SHA256` is provided and the remote clip hash differs, so task5 export is not hidden behind the previous trash-only default or a wrong remote video. `export_hawor_world.py` records the input `video_sha256` plus HaWoR checkpoint, infiller, and model-config SHA256 provenance in both HaWoR QC and NPZ outputs. The HaWoR requirement state treats missing/mismatched task5 QC or NPZ video hashes, missing QC/NPZ export asset hashes, or QC-vs-NPZ hash disagreement as blockers. The expected local task5 output path is:

```text
/data2/ego_annotation_outputs/v18_corrective_1600/hawor_exports/task5_tomato_960/hawor_world_hands.npz
```

Current status remains blocked because that NPZ is absent and the local HaWoR repo/checkpoints/config/`MANO_LEFT.pkl` are still missing. The contract is not execution evidence and does not accept the HaWoR requirement, metric hand state, contact, occlusion, nonpenetration, or V18 closure.

## Implementation Checkpoint 58: Strict HaWoR Contact Probe Rejects Immediate Contact Recompute

A candidate-only strict contact proximity probe now evaluates existing trash contact rows whose hand side passes the strict HaWoR bridge subset policy. It uses HaWoR bridge hand vertices in current V18 world coordinates and depth-backed visible object surface vertices only.

Current evidence: `223` strict trash contact rows evaluated, visible-surface minimum distance median `0.436 m`, p05 `0.051 m`, p95 `0.701 m`; only `19/223` rows are within `10 cm`, `12/223` within `5 cm`, and `4/223` within `1 cm`. The median delta relative to the source graph hand/object distance is `+0.433 m`. In current camera coordinates, HaWoR hand median depth is behind the visible object surface by median `0.571 m` (p05 `0.184 m`).

This is negative mechanism evidence: strict image-space bridge support does not imply 3D contact support. The leading mechanism is HaWoR bridge depth/scale mismatch; coordinate-frame mismatch with the visible-surface archive or a source-contact graph using a different hand/object geometry basis remain possible. The probe uses open visible object surfaces, so it cannot prove non-contact or nonpenetration; it only blocks immediate contact recomputation/acceptance from the trash bridge. Contact, nonpenetration, occlusion ownership, and V18 closure remain unaccepted.

## Implementation Checkpoint 59: Temporal Offset Probe Rejects Simple Frame-Shift Explanation

A candidate-only temporal-offset mechanism probe now evaluates whether the strict trash contact mismatch is explained by a fixed frame-index offset. For each strict contact row at frame `f`, it compares the visible object surface and camera at `f` against same-side HaWoR bridge hands from nearby frames `f + offset`, for offsets `[-5, 5]`.

Current evidence: `223` rows evaluated. The dominant best-distance offset is `-5`, but it explains only `0.202` of rows; the dominant best-absolute-depth-gap offset is also `-5`, explaining only `0.193` of rows. Offset `0` distance median is `0.436 m`; best-any-offset distance median remains `0.393 m`. Offset `0` absolute camera-depth-gap median is `0.571 m`; best-any-offset absolute depth gap remains `0.541 m`.

Interpretation: no consistent temporal offset explains the strict-contact mismatch. This makes a simple frame-index shift unlikely as the primary mechanism; scale/depth/coordinate-basis mismatch remains more plausible. The probe is diagnostic only and does not accept contact, nonpenetration, HaWoR foundation, or V18 closure.

## Pipeline DAG and Parallelism

V18 is parallel by construction:

1. Decode/cache frames.
2. In parallel: hand detection, object planning/detection, depth estimation, camera trajectory, captions.
3. In parallel per object: SAM2 tracking, visible surface extraction, material/flow tracking, cheap rigidity/deformation test.
4. In parallel per hand track: MANO measurement assembly and visibility/occlusion evidence.
5. Single-writer reducers assemble measurement tables.
6. Bounded optimizer runs once per case or in small temporal chunks with explicit overlap/reducer semantics.
7. Full-duration render runs by frame chunks and concatenates with frame-count verification.

Every parallel worker writes isolated artifacts. Reducers are the only writers to summary JSONs.

## Acceptance Checks

A V18 run is acceptable as a pipeline artifact only if:

1. Full-duration outputs match the raw video frame count, fps, and duration.
2. Runtime manifest satisfies the hard runtime ceiling, or the run is explicitly marked failed for runtime.
3. No default stage uses BundleSDF/NeRF/test-time neural reconstruction.
4. Hand/object visibility states are present for every frame/track.
5. Occluded intervals are labeled with occluder owner or unresolved status; inferred poses through occlusion carry uncertainty.
6. Object geometry claims are scoped: visible-surface, rigid visible canonical, deformable surface, feed-forward completed, or unresolved.
7. Contact labels are tied to a hand side, object id, geometry source, and visibility/depth evidence.
8. Renders distinguish observed from inferred/occluded/unresolved states.
9. Readiness flags remain false unless object geometry, contact ownership, and visual QC actually support them.
10. Representative visual review covers named V16/V17 failure frames and any new occlusion intervals.

## Expected Failure Modes

- Long or complete hand occlusion without reliable before/after state: output unresolved, not hallucinated.
- Deformable or transparent objects with poor depth: visible-surface uncertainty grows; no rigid pose claim.
- SAM2 identity switches under occlusion: object identity marked ambiguous until re-associated.
- Metric depth failure near hand/object edges: interior/depth-edge ownership masks prevent false residual claims.
- Camera/world scale drift: report trajectory uncertainty and avoid metric-contact claims that depend on unvalidated scale.
- Fast model failures: if feed-forward hand/object detectors miss a visible state, V18 may fail the case quickly rather than launch expensive reconstruction.

## Deliverable Layout

Default output roots:

```text
/data2/ego_annotation_outputs/v18_runtime_manifest/
/data2/ego_annotation_outputs/v18_measurements/
/data2/ego_annotation_outputs/v18_visibility_occlusion_state/
/data2/ego_annotation_outputs/v18_object_surfaces/
/data2/ego_annotation_outputs/v18_fast_motion_state/
/data2/ego_annotation_outputs/v18_consistency_graph/
/data2/ego_annotation_outputs/v18_full_state/
/data2/ego_annotation_outputs/v18_renders/
```

Required videos per case:

```text
v18_overlay_mano_object_occlusion.mp4
v18_reconstruction_3d_world_occlusion.mp4
v18_side_by_side_occlusion.mp4
```

Every summary JSON must include runtime, frame-count equality, readiness flags, unresolved-state counts, and evidence-source counts.

## Immediate Implementation Order

1. Provision and run real HaWoR/metric MANO for task5 using the task5 HaWoR export contract; without task5 HaWoR, V18 physical hand/contact/occlusion claims remain invalid. Trash bridge rows remain candidate-only until accepted by explicit bridge/foundation criteria.
2. After copying a real task5 `hawor_world_hands.npz` into the contract path, rebuild/validate HaWoR requirement and bridge state before any downstream claim.
3. If doing local pre-provisioning work, use the HaWoR bridge subset policy only as a candidate queue. The strict contact probe currently argues against immediate contact recomputation/acceptance from that queue because image-supported rows are usually not 3D contact-aligned.
4. Continue the object/part path from the accepted OWLv2→SAM2 tracks: part visible surfaces -> part-model candidate residuals -> rigid/articulated/deformable decision.
5. Implement depth-fused rigid/part reconstruction only where the residual acceptance tests can be evaluated.
6. Implement the bounded factor graph over camera/depth correction, hand state, object/part SE(3), articulation, contact switch, and occlusion owner.
7. Render full-duration outputs with uncertainty/occlusion status and run runtime/manifest/audit checks only as validation of these implementation artifacts.
8. Evaluate on `trash_1050` and `task5_tomato_960`; if runtime, HaWoR validation, SAM2 tracking, reconstruction, or graph optimization fails, preserve the concrete failed residuals and causal evidence.

## Checkpoint 2026-06-15 — final-pipeline contact support modes

`run_v18_full_pipeline.py` now distinguishes contact support modes instead of treating every active contact as a rigid-object pose claim.

- Rigid visible-mesh support remains strict-rigid only. Weak mask-depth point clouds and surface-changing/deformable objects are explicitly blocked from `rigid_pose_supported_visible_mesh`.
- Compact surface-changing support is a separate mode for objects whose schema has `surface_change_without_pose_state` without part/deformable blockers. It can support visible pose/contact for the main tomato when source depth intrinsics, mask projection, visible mesh distance, HaWoR MANO, and near MANO/object geometry agree. It keeps `object_geometry_complete=false` and `object_pose_requirement_met=false`.
- Weak visible-depth pose rows are recovered from rejected visible-surface rows using the source metric-depth NPZ, SAM2 mask, source intrinsics, and `T_world_camera_metric`. They are downweighted graph observations and must remain explicitly weak; they may not support strict rigid pose.
- Validated-part contact support now records the nearest validated part separately from the nearest part, so a rejected closer part cannot hide a supported farther part. Active validated-part contact still requires the supported part to be within the near-geometry band.
- Deformable visible-surface contact is a local contact mode for deformable/secondary-deformable objects. It requires same-frame visible depth surface, observed HaWoR hand support, same-frame raw MANO-to-visible-surface distance no greater than 5 cm, and strong mesh/metric support. Graph-shifted/coupled object distances may be recorded as diagnostics, but they must not activate deformable contact. This mode does not claim object pose, hidden geometry, nonpenetration, or rigid SE(3).
- Surface-changing visible-pose support requires nontrivial observed and predicted mask projection agreement; one or two projected vertices inside a mask are not enough to support pose/contact.
- Active contacts are physical frame-pair states, not rendered line counts or isolated visible-surface hits. A contact switch is active only when the final state has direct frame-local visible/validated physical contact support. The episode path is explicit graph evidence (`contact_episode` variables and `contact_episode_persistence` factors): it requires observed HaWoR support, no local nonpenetration conflict, continuous hand-object manipulation evidence, and local anchors, but episode continuity alone is now a non-active `contact_episode_hypothesis_nonactive` state. Episode-supported rows preserve visible-surface distance as evidence, but that distance is not interpreted as the contact-patch gap when the contact patch is occluded or unmodeled. This supports uncertainty/evidence only unless a direct frame-local contact support path is present; it does not complete hidden geometry, object pose, or global occlusion ownership.
- Active contacts with depth contradiction are blocked unless the contradiction is explicitly resolved by either a bounded visual-contact prior or a supported manipulation episode. The bounded visual-contact prior still requires image contact evidence, high mesh contact support, close MANO/object geometry (<=7 cm), observed HaWoR, no nonpenetration conflict, and a final post-graph object/part support path. For partially visible surface-changing objects, a local observed-surface contact support (`surface_changing_local_visible_contact_surface`) may serve as a direct anchor when observed surface projection remains inside the current mask and residuals are bounded. Without direct or episode support, a depth-contradicted row may render only as `depth_occluded_contact_possible` when final support and near geometry exist, or remain raw/contradicted evidence. Depth-occluded rows are non-active uncertainty states, not contact claims, and populate local hand/object `contact_depth_order_occlusion` evidence without becoming global occlusion-owner claims.
- `supported_near_noncontact` is a second non-active mode for final validated object/part/deformable support within the near band when the contact switch remains off and depth ordering is not contradictory. For part-required objects, this now requires same-frame part depth/mask validation rather than aggregate track-level support, and its reported distance/endpoints come from the supported part path rather than any rejected nearer part. For deformable visible surfaces, the same-frame visible-surface distance can support a non-active near state in the 5-12 cm band; this does not satisfy the active contact predicate. It records a physically close state without claiming contact.
- Unsupported contact proposals stay in JSON as raw/gated evidence and must not be drawn as solved or possible contact. Overlay/world videos may render active contacts and the two scoped non-active modes with different labels/styles, but the artifact now reports contact in three separate layers: active frame-pair states, consecutive temporal contact episodes, and contact geometry/evidence paths. Render draw counts are explicitly excluded from contact semantics. Final `physical_contact_claim_supported` and family-specific `*_contact_claim_supported` flags mean solved active physical contact only; non-active near rows keep `*_contact_evidence_supported` fields instead.
- Part SE(3) contact coupling is strict: a `contact_part_pose_anchor` factor may be emitted only for an active, raw-on, or accepted-owner part-contact proposal with observed HaWoR support and near depth-fused part geometry. Near hand/part proximity or weak image support alone must not pull a part pose under a contact label; those rows remain non-contact evidence unless the contact switch has physical support. Part pose readiness is frame-local: graph-phase depth/silhouette validation sets `part_pose_ready=true` only for the current supported part frame, while rejected/unvalidated part meshes remain candidates. Part-required object state can now be frame-locally ready when a visible base/reference surface and at least one accepted global moving-part track has a supported current-frame pose; accepted global part tracks without a current ready pose remain residual uncertainty and the state still does not complete hidden geometry or whole-object pose.
- World-render direct active and non-active contact endpoints are driven by metric MANO/world and metric object/part/mesh/world state, not 2D bbox centers. Episode-only active contact states are rendered as episode-state edges between hand/object anchors rather than fake nearest-visible-surface metric gaps.
- Compact multiview object-geometry completion is now a separate strict clean-rigid assessment from visible-pose/contact validation. It can set `object_geometry_complete=true` and `object_pose_requirement_met=true` only for structured rigid, pose-allowed compact objects with current-frame visible depth/silhouette pose support, at least 25 source frames, at least 5,000 fused depth points, and non-sparse hull/Poisson mesh topology. Minor surface-layer or texture changes are compatible with compact completion only when the structured physical model still allows object pose; `surface_change_without_pose_state`, deformable, secondary-deformable, part/relative-motion, sparse, and no-current-visible-support cases remain blocked. Rendered object mesh marks are validation-aware: completed clean-rigid poses, supported visible poses, and rejected mesh candidates are visually separated.

This 2026-06-15 checkpoint is superseded by the 2026-06-16 metric/schema/contact-anchor checkpoint below. Use the later checkpoint for current counts and the latest validated run path.

## Checkpoint 2026-06-16 — metric alignment, structured schema, stable anchors, and deformable patch state

V18 now treats metric alignment and physical-state schema as upstream requirements for contact/object coupling.

- HaWoR MANO vertices are depth-scaled in the V18 bridge from projected HaWoR vertices to complete-depth UniDepth before entering the final metric world state. The bridge records `hawor_to_v18_depth_scale`, scale status, sample count, and depth source. Active contact requires observed same-frame HaWoR plus valid depth-scale provenance.
- `audit_v18_metric_alignment.py` is the frame-local check for shared metric geometry. It projects raw RGB/masks, object visible-depth surfaces, MANO vertices, and camera/world transforms into the same frame and reports reprojection support plus MANO-object nearest distances. It is intended to localize transform/scale failures before contact or occlusion logic is trusted.
- Physical-state schema no longer derives physical model type from free-text `physical_notes` keywords. `build_v18_structured_physical_model.py` emits structured VLM fields, and `build_v18_physical_state_schema.py` consumes those fields. The main tomato is modeled as rigid/near-rigid with surface-appearance uncertainty and pose-model allowance; detached peel can remain separate/deformable.
- Contact/object pose anchors are not generated from raw contact proposals or nonpenetration conflicts. The final graph first estimates geometry/contact without contact-pose anchors, proposes anchors only from solved active direct contact rows, and repeats bounded passes. If the anchor set does not converge, only the stable intersection of bounded-pass anchors with stable support-path signatures may affect object/part pose; unstable direct active contacts remain contact states but are explicitly not pose-coupled.
- Coupled object/part distances are diagnostics after this checkpoint. They do not create same-row contact evidence. Active contact support must come from direct same-frame MANO-to-visible/validated geometry and final support gates.
- Final contact semantics distinguish active contact, pose-coupled active contact, stable-contact-without-emitted-factor, unstable active contact, deformable active contact coupled only to a local visible-surface patch state, and non-active evidence rows. Rendered contact lines do not redefine these states.
- Latest validated final counts: trash has 348 active contact states, 8 emitted stable `contact_part_pose_anchor` factors, 340 `deformable_surface_patch` variables, 34 supported-near noncontact rows, 3 accepted occlusion-owner variables, 3 observed MANO pose-fill-through-occlusion rows with matching `hand_occlusion_pose_fill` factors, 332 visible-base-reference-plus-lid structured part-object rows for the off-white trash can, and 0 object-geometry/object-pose-complete rows. Task5 has 32 active contact states, 19 emitted stable pose anchors (11 object, 8 part), 5 `deformable_surface_patch` variables, 8 direct active contacts marked not pose-coupled because the anchor fixed point oscillated, 15 supported-near noncontact rows, 0 pose-fill-through-occlusion rows, 101 compact rigid object-geometry/object-pose-complete rows, and 19 visible-base-reference-plus-moving-part structured faucet rows.
- Tomato compact completion is allowed only under the structured rigid/minor-surface-change contract: `model_physical_state_type=rigid`, `pose_model_allowed_by_structured_vlm=true`, `geometry_changes=minor_surface_layer_or_texture_change`, no `surface_change_without_pose_state`, no secondary deformable component, and no part/relative-motion requirement.

- Pose fill-through-occlusion is now accepted only for observed same-frame depth-scaled MANO behind an accepted occluder with HaWoR MANO/object depth-order support, or for a future explicitly accepted temporal baseline. Current outputs use only the observed-MANO path: trash frames 262 left, 268 left, and 850 right. Temporal occlusion pose fill remains zero for both cases, and accepted-owner absence keeps task5 unresolved.
- Accepted pose-fill backing state uses graph-accepted owner labels consistently: hand `occlusion_owner_hypothesis.state`, `source_occlusion_owner_depth_support.depth_pair_evidence_state`, embedded owner-evidence rows, and `hand_occlusion_pose_fill` factors all use accepted-owner states. Pre-graph `*_owner_unaccepted` labels are retained only under explicit raw/provenance fields.
- The final artifact sanitizer no longer rewrites legitimate `accepted_*` physical-state labels into `supported_*`; validators now preserve and audit scoped acceptance semantics rather than banning the word globally.
- Visible-base-reference-plus-moving-part structured object state is not whole-object completion: off-white trash-can rows retain accepted global missing-hinge uncertainty where the hinge has no current ready pose, and faucet rows retain accepted global missing-handle/lever uncertainty where only one generated part is ready. The validator now cross-checks `accepted_global_part_track_labels` against the recorded `part_object_blocker_manifest` source and then checks `ready_part_track_labels`, `unready_part_track_labels`, `missing_current_frame_part_track_labels`, and residual-uncertainty entries so absent tracks cannot disappear from the accepted partial state.

Latest validated run for this checkpoint: `/data2/ego_annotation_outputs/v18_full_pipeline/final_run_20260616_part_structured_global_required_labels.log`.
