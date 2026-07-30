# Version Delivery Audit

This audit uses the delivery standard from the user: a deliverable video must have the same duration and frame coverage as the original raw video. The rendered outputs must show the annotated video, the 3D world reconstruction, MANO hands, head camera, object mesh, and semantic caption over the full source clip. Any shorter render counts as QC evidence for individual mechanisms.

## Current Delivery Status

v1 produced full-clip videos. Its object representation is physically weak and uses proxy object state in key cases, so it stands as a baseline artifact.

```text
/data2/ego_annotation_outputs/fullmesh_task7/fused/side_by_side.mp4
/data2/ego_annotation_outputs/fullmesh_task5/fused/side_by_side.mp4
/data2/ego_annotation_outputs/representative_trash/fused_bagprompt_full_final/side_by_side.mp4
```

v2 produced a 91-frame contact-window video with observed object surface geometry. It is a QC demo for observed-surface meshing.

```text
/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/side_by_side.mp4
```

v3 produced investigation clips. It failed as a clean pipeline version because scope expanded during implementation around segmentation, mesh completion, depth scale, hand refit, and contact. The later `V3 Closure State` wording in the V4 document referred only to two bounded evidence windows, not to completion of the V3 solver design.

V3 also created the long-running factor-graph obligation. Its design and diagnostics identified the hand-object metric contradiction as a joint state-estimation problem: MANO hands, object geometry, camera/depth scale, and contact could not be accepted independently. Later versions implemented real component graphs. The original full-video hand-object interaction solver remained open.

v4 through v6 produced 31-frame wild-rice videos. These support dynamic surface and sparse correspondence inspection.

```text
/data2/ego_annotation_outputs/representative_wild_rice/v4_world_reconstruction_completed_measurement_plus_sam2seed_finalvis_2520_2550/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4
```

v7 through v15 produced short QC/evidence clips, mostly three to seven frames. Those clips support inspection of mechanisms such as mesh replay, MANO repair, hidden-surface fusion, and contact dynamics.

Representative short evidence clips:

```text
/data2/ego_annotation_outputs/representative_box_books/v7_box_books_similarity_refit_box_with_books_616_618/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_box_books/v8_probe_box_books_612_617/deliverables_tail_v1/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v10_mesh4d_consecutive_outputs/fused_hidden/wild_rice_mesh4d_hidden_compact_2538_2543/deliverables/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

## Versioning Correction

Future pipeline versions must begin with a design document before implementation. The design document must define:

- the action interval length and representative samples;
- state variables for head pose, MANO hands, object mesh, contact, and caption;
- model-produced perception inputs;
- optimization objective;
- physical consistency terms;
- acceptance predicates;
- render deliverables;
- expected failure modes.

Bug fixes, threshold corrections, renderer improvements, and short diagnostic studies belong inside the current version as patches or sub-experiments. A new top-level version number requires a new upfront pipeline definition.

## Immediate Consequence

The current repository contains zero v2-through-v15 deliverables under the full-raw-video standard. The current repository contains validated components and short evidence clips. The missing artifact is a v15-quality full raw-video deliverable. The next work item is to define the next real pipeline version upfront, then run it over complete source videos on representative samples.

## Runnable Pipeline Status

v2 through v15 also contain zero full runnable pipelines under the full-raw-video standard.

The closest executable paths are component chains:

- v2 has object-plan, segmentation, metric-depth, observed-surface mesh, and v1 renderer hooks. It relies on precomputed v1 hand/camera annotations and produces observed-surface mesh evidence, with no full-video mesh-completion, contact physics, or acceptance driver.
- v3 through v6 scripts generally require explicit `--frame-start` and `--frame-end` inputs and operate on selected windows.
- v7 batch wrappers require prebuilt target contracts: observed mesh archive, manifest, annotations, metric depth, baseline z-buffer report, and configured frame ranges.
- v8 through v15 consume accepted short-window artifacts from earlier stages and solve local hand/contact/physics graphs.

The only current source path that records a full source timeline check is the v1 fusion path. Running any v2-through-v15 script over a full raw video today would create another partial component output or crash on missing precomputed full-video contracts, and the result would fail the full-video deliverable standard.

The graph status follows the same distinction. V3, V6, V8, and V12 through V15 contain real factor-graph or graph-ready solvers over object pose, sparse object correspondence, MANO contact, contact dynamics, handoff, and contact switching. Those solvers are local in time or local in state variables. V17 must treat them as evidence modules for the integrated full-timeline factor graph required for annotation-quality closure.
