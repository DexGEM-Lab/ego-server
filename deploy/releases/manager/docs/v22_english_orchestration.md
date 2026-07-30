# V22 English Orchestration Over Real Components

Status: V22 runtime runbook. This document is the authoritative V22 orchestration for runtime agents. It is based on `docs/v22_harness_list.md` and must be followed with `docs/v22_run_contract.md`. It does not authorize fake numbered scripts, registry-only progress, validator loops, or legacy closure as final physical annotation.

## 0. Physical Objective And State Variables

The runbook exists to build and render physical variables, not containers:

- `K_t`, `T_world_camera,t`, depth, and scale provenance: the metric coordinate frame.
- `H_t`: per-frame metric MANO hand state with side, camera/world semantics, provenance, visibility, and uncertainty.
- `O_i`: target object instances selected from visual/task evidence.
- `M_i,t`: accepted object masks/tracks from the active OWLv2 bbox-prompt SAM2 proper chain.
- `G_i`: object geometry. For rigid branches this means completed/adapted instance mesh, not centroid, primitive, mask, or point cloud.
- `T_world_object,i,t`: object pose trajectory/posterior for rigid or rigid-part branches.
- `C_t`, `V_t`, `N_t`: contact/near-contact, visibility/occlusion ownership, and nonpenetration residual/uncertainty.
- `R_t`: rendered overlay/world/side-by-side videos whose visible marks are caused by the variables above.

A command is progress only when it measures, optimizes, renders, or falsifies one of these variables. Reports, schemas, row counts, and validation outputs are evidence only.

## 1. Runtime Start

1. Inspect `git status --short`; preserve unrelated dirty files.
2. Read `docs/v22_run_contract.md`, `docs/v22_harness_list.md`, and this runbook.
3. Resolve mode: `v22_infer`, `v22_benchmark`, or narrower diagnostic.
4. Verify the input video or dataset root and side inputs without mutating them.
5. Verify the run root does not overwrite an existing V22 run.
6. Probe the default A800 target `ssh -p 57938 zjh@115.190.235.210` before heavy work unless an alternate authorized compute target was explicitly declared.
7. Record compute target and budgets in `logs/harness_events.jsonl`.
8. Create unresolved initial state under `state/`.

If the prompt was not loaded from `configs/v22_agent_system_prompt.md`, stop and report the launch command from `docs/v22_run_contract.md`.

## 2. Input And Timeline

Run or verify the V22 input setup from M00-M01:

```text
input/input_manifest.json
input/raw_frame_manifest/manifest.json
input/source_frame_manifest/manifest.json
input/source_frame_manifest/rgb/*.jpg
```

Use the V21 timeline scripts listed in `docs/v22_harness_list.md` until V22 wrappers exist. Confirm frame count, FPS, duration, source dimensions, render dimensions, and frame-index convention. A frame mismatch is a contract error and must be fixed before downstream physical claims.

## 3. Depth And Camera

Build depth/camera candidates from M02, then select and tune through M03:

1. Write the depth modality report.
2. Build or verify the V19 calibration contract when V19/V18 spine components consume the state.
3. Run monocular depth/camera candidates and any declared assisted candidates on the default A800 target `ssh -p 57938 zjh@115.190.235.210` unless an alternate authorized compute target was explicitly declared.
4. Register candidates through the explicit legacy registry adapter if needed.
5. Compare assisted/native/stereo paths against monocular evidence whenever those paths are used.
6. Write `tuning/depth_camera/<candidate_id>/attempt_<k>.json` before downweighting a weak bottleneck candidate.
7. Select the depth/camera bundle with uncertainty.

Depth/camera selection does not establish object pose or contact. It provides the metric backbone consumed by later modules.

## 4. Hand Candidates And Metric MANO

Run hand candidate streams from M04, then diagnose/refit/optimize through M05:

1. Run RTMLib 2D and MANO candidate streams as available on the default A800 target `ssh -p 57938 zjh@115.190.235.210` unless an alternate authorized compute target was explicitly declared.
2. Merge and compare candidates where the scripts exist.
3. Review overlays for side swaps, drift, crop failures, occlusion, and scale inconsistency.
4. Run metric MANO diagnosis/refit/active optimization when candidate evidence supports it.
5. Write `tuning/hand_mano/<hand_track_id>/attempt_<k>.json` before downweighting a weak hand/MANO bottleneck.
6. Accept metric MANO only when camera/world semantics, provenance, visibility, and uncertainty are represented in state and can drive render.

2D hand detections and overlay-aligned hands must remain visually distinguished from metric MANO and cannot support contact ownership or nonpenetration.

## 5. Target Object Plan, Keyframes, And Bbox Evidence

Use M06 as the only V22 active target-selection and bbox-prompt chain:

1. Inspect raw/source frames, task context, hand proximity, and side inputs.
2. Write `measurements/object_candidates/object_plan_agent.json` from visual/task evidence.
3. Sync or derive `measurements/object_candidates/object_plan_current.json` as the current target plan.
4. Run `scripts/select_v21_agent_keyframes_from_plan.py` to produce `segmentation_stable_keyframes.json`.
5. Run `scripts/run_v21_owlv2_bbox_proposals.py` on those keyframes.
6. Run `scripts/approve_v21_owlv2_bbox_prompts.py` to produce approved SAM2 bbox prompts.
7. Validate target identity, active intervals, rejected alternatives, text prompts, selected keyframes, and approved boxes.

Dataset/public object rosters are model libraries only. They cannot select targets before visual/task evidence.

## 6. Segmentation, Mask Tracking, And Contamination Review

Run M07 from approved bbox prompts:

1. Run SAM2 proper video propagation from `measurements/object_candidates/owlv2_bbox_approved_prompts.json`.
2. Produce the per-object SAM2 track, masks, report, QC, overlay, and summary files listed in `docs/v22_harness_list.md`.
3. Run contamination review on `measurements/object_tracks/sam2_proper_summary.json`.
4. Assemble segmentation state only after review marks masks as accepted, uncertain, or rejected.
5. Write `tuning/segmentation/<object_id>/attempt_<k>.json` before downweighting a weak segmentation bottleneck.

Wrong-object masks, table/background capture, major missing visible object parts, and coordinate/prompt mismatch are implementation errors. They must not enter geometry as weak observations.

## 7. Visible Surfaces And Branch Decision

Run M08 after accepted masks and selected depth/camera exist:

1. Lift accepted SAM2 proper masks into visible metric surfaces.
2. Render or review visible surface overlays.
3. Check mask-depth alignment, units, sample counts, object id, and camera/world provenance.
4. Decide object branch: rigid, articulated, deformable, support/occluder, or unresolved.
5. Validate that rigid objects are routed to mesh completion and pose graph, not visible-surface-only closure.

Visible surfaces are metric observations. They are not completed geometry or object pose.

## 8. Object Geometry Completion And Mesh Candidate Registry

Run M09 for rigid or rigid-part branches:

1. Build observed mesh candidates and/or visible-depth completions.
2. Run TRELLIS or another explicitly enabled remote mesh candidate path only on the default A800 target `ssh -p 57938 zjh@115.190.235.210` unless an alternate authorized compute target was explicitly declared.
3. Fit or validate public CAD only as benchmark prediction input tied to the target plan.
4. Register and validate candidates through explicit adapters where needed.
5. Render/review candidate identity, scale, silhouette, depth, and free-space residuals.
6. Select the candidate that may enter pose fitting, or write explicit unresolved geometry state.

Generated or retrieved meshes remain candidates until fitted, validated, optimized, and rendered.

## 9. Rigid Object Pose Fit And Factor Graphs

Run M10 for each rigid object candidate:

1. Estimate and fit visible-frame object pose against accepted mesh/depth/mask evidence.
2. Run robust pose and temporal pose graph when inputs exist.
3. Use V19/V18 factor-graph components where needed for the physical spine.
4. Check pose rows, residuals, temporal smoothness, surface-fit terms, and adopted-pose provenance.
5. Accept only completed/adapted mesh pose or write explicit unresolved state.

Centroid estimates, visible-surface transforms, and smoothers are not object pose closure.

## 10. Contact, Occlusion, Visibility, And Nonpenetration

Run M11 only after enough metric MANO and object mesh pose evidence exists:

1. Build MANO/object contact reports and contact/occlusion/nonpenetration rows.
2. Use V19/V18 ownership, occlusion, mesh-contact, and nonpenetration components as the physical spine.
3. Review interaction renders and residuals.
4. Record whether failures originate from hand, object, camera/depth, mask identity, or missing mechanism.
5. Write explicit unresolved contact/occlusion/nonpenetration state when evidence is weak.

Render-only contact marks are not contact evidence. Contact, occlusion, and nonpenetration must be explicit state variables or explicit unresolved states consumed by render.

## 11. State Assembly And Renderer Boundary

Run M12 after measurement and branch modules have produced current outputs:

1. Build V19-compatible base annotations if needed by physical-spine components.
2. Use V20 adapters only when explicitly required as adapters, never as final V22 closure.
3. Adapt V21/V19/V18 outputs into V22 state names.
4. Write:
   - `state/annotations_v22_renderable.json`
   - `state/v22_physical_state.json`
   - `state/v22_uncertainty_state.json`
   - `state/v22_agent_evidence.md`
5. Verify no GT/oracle markers enter prediction state.
6. Record provenance, uncertainty, tuning attempts, renderer consumption, and unresolved variables.

Private measurement files, registries, QC reports, and overlays are not final annotations unless represented in V22 state.

## 12. Full-Duration Rendering And Visual Review

Run M13 after V22 state exists:

1. Render segmentation, visible surface, hand, integrated, and full annotation overlays as needed.
2. Produce required final outputs:
   - `renders/v22_overlay.mp4`
   - `renders/v22_world.mp4`
   - `renders/v22_side_by_side.mp4`
3. Check duration, frame count, non-empty video, naming, and state consumption.
4. Consume rendered videos as physical annotations before claiming progress.
5. Route failures back to the earliest causal module.

The final artifact is the rendered physical annotation. A video container is not success unless the visible marks are driven by the named mechanisms.

## 13. Benchmark Evaluation And Iteration Controller

Run M14 only after prediction-side V22 state and renders exist:

1. Prepare benchmark inputs with strict GT isolation.
2. Keep GT paths only under `evaluation/reference_manifest.json`.
3. Evaluate V22-compatible prediction state/renders against GT only after prediction render exists.
4. Write iteration records, parameter changes, GT leakage checks, and final selection reports.
5. Convert metrics and visual failures into one atomic next intervention, with prediction before rerun and overfitting risk.

Benchmark metrics are evidence, not authority. Visual contradictions remain evidence even when a metric improves.

## 14. Atomic Algorithm Overlays And Materialization

Run M15 as support evidence:

1. Produce per-atom data/overlay/QC only to inspect algorithm behavior.
2. Materialize atomic results when they correspond to current run-root measurements.
3. Review whether each atom is usable evidence, weak evidence, implementation error, missing implementation, or deprecated history.

Atomic overlays do not replace V22 integrated state or final render.

## 15. Batch And Parallel Control

Run M16 only as execution control:

1. Use a single project tmux session for long-running work.
2. Treat missing parallel scheduler pieces as missing/design-only.
3. Decide whether a task is single-entry V22, selected batch helper, or blocked on missing scheduler infrastructure.

Batch control may schedule work. It may not decide object identity, pose acceptance, contact, occlusion, or annotation completion.

## 16. Ordered V22 Flow

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
