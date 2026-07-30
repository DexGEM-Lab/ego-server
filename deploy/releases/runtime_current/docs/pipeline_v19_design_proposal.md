# V19 Design Proposal: Agent-Harness Physical Annotation and Evaluation

**Status:** design only. This document does not start V19 implementation and does not supersede the current V18 v5 scoped bounded-MANO deliverable.

## 1. Thesis

V19 should be a pipeline/evaluation release whose primary change is **where annotation authority lives**. V18 proved that a script stack can produce a bounded interval-MANO artifact, but it also exposed three structural weaknesses: the entry point is not agent-native, the object and visualization story is not audience-ready, and the result is not benchmarked strongly enough to guide future work.

V19 therefore makes the Pi agent harness the annotation engine. Scripts may still extract frames, run hand models, run depth/SLAM, track masks, fit object hypotheses, solve factors, render videos, and compute metrics. They are instruments. The agent owns the loop that turns those measurements into a physical-state claim: it inspects evidence, keeps competing mechanisms alive, requests targeted measurements, rejects invalid measurements, assigns uncertainty, and writes the final renderable state.

The release goal is not “more V18 patches.” The release goal is a self-contained system that can take an egocentric video, produce visible physical annotations, and report quantitative evidence against open benchmarks. The V18 evidence base referenced here is the current v5 artifact at `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`, with its manifest and uncertainty classification as the scoped bounded-MANO baseline.

## 2. Non-negotiable V19 outputs

A completed V19 run must produce a directory with these artifacts for the full input video duration:

1. **Audience-facing videos**
   - `v19_overlay.mp4`: raw video with hand meshes, manipulated object render, contact/occlusion cues, and minimal explanatory labels.
   - `v19_world.mp4`: 3D scene view with camera/head trajectory, MANO hand meshes, manipulated object geometry/hypotheses, and uncertainty.
   - `v19_side_by_side.mp4`: raw, overlay, and world views synchronized.
   - Optional `v19_story.mp4`: a concise demo video with the key physical message of the sequence.
2. **Renderable state**
   - Per-frame MANO state with camera/world-frame semantics, visibility, provenance, and uncertainty.
   - Per-object state-type decisions, visible-surface measurements, branch-specific geometry, corrected pose trajectories where the branch is rigid/articulated, visibility, contact/occlusion state, and uncertainty.
   - Camera/head trajectory and intrinsics/metric-scale provenance.
3. **Evaluation bundle**
   - Dataset-normalized metrics for any benchmark run.
   - Ablation table and configuration manifest.
   - Failure clusters with frame ranges and causal interpretation.
4. **Agent evidence ledger**
   - The harness records what the agent observed, what explanation it accepted or rejected, and which uncertainty remains.

A JSON row count, schema pass, or internal status overlay is not a V19 deliverable unless it drives the rendered annotation and is consumed by the viewer/evaluator.

## 3. Harness entry point

### 3.1 Agent first, not script first

The V19 entry point should be a Pi command/session with the V19 system prompt loaded, not a top-level `run_v19_full_pipeline.py` script and not an external program that creates or controls a Pi session. The Pi session receives the V19 task contract and a narrow set of measurement/render/evaluation tools, then owns the evidence loop that produces a physical annotation artifact.

Scripts remain reusable tool implementations. The difference is ownership:

| Layer | V18 pattern | V19 pattern |
|---|---|---|
| Outer control | Python script decides stages and gates | Pi agent decides evidence needs and state claims |
| Visual judgment | Separate VLM calls produce JSON fields | Agent inspects visual/geometric evidence directly and records scoped claims |
| Failure handling | Validators/status manifests often dominate | Failed measurements revise the causal model and next measurement |
| Output authority | Script artifacts imply completion | Rendered physical annotation plus benchmark evidence imply completion |

### 3.2 Practical Pi route

The implementation route is now Pi-native. Pi itself is the V19 harness: a Pi command/session with the V19 system prompt loaded owns the evidence loop, physical-state decisions, uncertainty, and renderer handoff. Do not implement an outer Python, TypeScript, shell, or SDK wrapper that creates or controls a Pi session. Scripts are callable measurement, optimization, rendering, export, and evaluation tools only.

Installed Pi documentation supports the route needed by the user request:

- `~/.pi/agent/models.json` can define custom providers and models. The current config already defines provider `occ` with model `gpt-5.5`, image input, reasoning, and `openai-responses`.
- Pi CLI supports selecting a provider/model via `--provider <name>` and `--model <pattern>`, and replacing the default prompt via `--system-prompt <text>`.
- Pi project prompt templates under `.pi/prompts/*.md` provide native slash-command entry prompts for repeatable runs.

The governing implementation route is defined by `docs/v19_run_contract.md` and supersedes any earlier SDK-created-session route. The route is:

```text
load configs/v19_agent_system_prompt.md through `pi --system-prompt`
  -> start Pi directly with provider `occ`, model `gpt-5.5:xhigh`, and the V19 project prompt template
  -> Pi verifies input, compute target, worktree ownership, and run root
  -> Pi calls measurement/render/evaluation scripts as tools when they reduce a named physical blocker
  -> Pi writes accepted physical claims into render-consumed state
  -> Pi renders/inspects full-duration artifacts and bounded benchmark evidence
  -> Pi stops when deliverables are produced or an explicit uncertainty state is the honest artifact
```

For smoke testing the model route, use the isolated command in `docs/v19_run_contract.md`. The smoke form is intentionally non-interactive, tool-disabled, session-disabled, and isolated from discovered context/skills/extensions so it checks model/prompt routing without starting implementation or allowing file/system mutation.

### 3.3 Harness responsibilities

The harness should enforce what a normal chat session cannot reliably enforce:

1. **Input contract:** one video path plus optional calibration/depth/known-object hints.
2. **Artifact root:** every output is under one immutable run directory.
3. **Prompt contract:** the V19 system prompt defines physical annotation standards, no-proxy rules, benchmark discipline, and stop conditions.
4. **Tool contract:** the agent can call measurement/render/evaluation tools, not arbitrary ad hoc scripts as final authority.
5. **State contract:** every accepted physical claim is written to a state file consumed by the renderer.
6. **Evidence contract:** every uncertain or rejected mechanism is recorded with frame ranges and artifacts.
7. **Benchmark contract:** if a benchmark dataset is selected, output conversion and metric computation are part of the run.
8. **Execution stop discipline:** a run proceeds through concrete pipeline components and physical decisions. If available measurements cannot support a stronger claim, the harness must render the current uncertain state and report the unresolved blocker rather than inventing a confident state or looping on proxies.

### 3.4 Agent-native visual judgment

Replacing VLM calls does not mean removing visual evidence. It means the harness exposes frames, crops, render sheets, depth overlays, 3D snapshots, and benchmark comparison images directly to the Pi session, and the same agent that owns the physical claim inspects them. The output of that judgment is not a standalone `vlm_contact=true` field. It is a scoped state update with frame ranges, visual observations, competing explanations, and uncertainty.

This design keeps model judgment auditable: the agent must cite the visual/geometric artifact that changed its belief, and any downstream state must still be rendered and evaluated. If the agent cannot see enough evidence to decide whether a rigid object, contact, or occlusion claim is supported, V19 carries an uncertainty hypothesis rather than inventing a confident state.

### 3.5 Runtime and compute placement

V19 must keep runtime as a design invariant, not a post-hoc metric. The default path should run in the same order of magnitude as the input video duration; for planning, the target is a first complete full-video artifact within roughly `10x` input duration on the designated server-class compute target, excluding optional external-evaluation runs. Any per-instance training, NeRF/BundleSDF-style optimization, exhaustive mesh search, or long autoresearch sweep is an offline research branch and cannot be the default path.

Heavy inference must not run on the local workstation by accident. The harness should declare a compute target for every expensive tool invocation:

- local machine: light orchestration, file inspection, prompt/session control, small metadata transforms, and final commit work;
- server/A800 or other explicitly configured remote target: HaWoR/WiLoR/HaMeR inference, depth/SLAM, SAM/video-mask tracking, open-vocabulary detection, mesh reconstruction, rendering batches, and benchmark runs;
- offline research branch: long per-instance reconstruction, training, large sweeps, or methods whose expected runtime is more than the default budget.

Every run manifest must record wall-clock time by phase, GPU target, model versions, and any budget overrun. If the default runtime budget is exceeded, V19 should render the best current uncertain state and mark the overrun as a design failure to be addressed, not continue until the artifact merely looks finished.

## 4. Self-contained video input/output

### 4.1 Input modes

V19 should accept these input modes, in order of evidential strength:

1. **Raw video only**
   - Required minimum contract.
   - V19 estimates intrinsics, camera motion, metric depth, hand/object tracks, and uncertainty.
   - Metric claims are explicitly weaker when scale/camera evidence is inferred.
2. **Raw video + calibration**
   - Known intrinsics/extrinsics reduce projection ambiguity.
3. **Raw video + depth/SLAM/camera trajectory**
   - External geometry is treated as a measurement with provenance, not as unquestioned truth.
4. **Benchmark dataset sample**
   - Dataset loader supplies ground truth, masks, poses, and official evaluation format.

Self-contained does not mean pretending all videos contain enough information for solved 3D physics. It means the pipeline can run from the video, produce the best supported annotation, and expose uncertainty instead of omitting modules or silently substituting proxies.

### 4.2 Output layout

A V19 run should have a stable layout:

```text
v19_runs/<run_id>/
  input/
    input_manifest.json
    frames/
  measurements/
    hand_candidates/
    object_candidates/
    depth_slam/
    masks_tracks/
  state/
    v19_physical_state.json
    v19_uncertainty_state.json
    v19_agent_evidence.md
  renders/
    v19_overlay.mp4
    v19_world.mp4
    v19_side_by_side.mp4
    review_frames/
  evaluation/
    metrics.json
    ablations.json
    benchmark_export/
  logs/
    harness_events.jsonl
    agent_session_link_or_export.html
```

The renderer consumes `state/`, not private measurement files. This prevents a familiar V18 failure mode where a source file changed but the final artifact did not.

## 5. Full-timeline state variables and optimization objective

V19 must define physical annotation as a solved state, not as agent prose. The agent can choose evidence, hypotheses, and interventions, but the renderable artifact should be produced from explicit state variables and an explicit robust objective.

### 5.1 State variables

For each frame `t`, hand side `s`, and object/part instance `i`, V19 should maintain:

- `H_{s,t}`: MANO pose, shape, global transform, camera-frame transform, world-frame transform, visibility state, and uncertainty.
- `K_t`, `T_world_cam,t`: intrinsics, camera/head pose, metric scale provenance, and camera-pose uncertainty.
- `G_i`: per-instance object geometry state selected by the physical-state branch: visible-surface measurements as metric evidence, video/depth-adapted rigid mesh when the object is rigid, dataset-provided instance mesh when available, articulated part graph, or deformation/uncertainty volume. A generic retrieved/category/TRELLIS mesh is only a prior until it is fitted and adapted to the observed instance.
- `O_{i,t}`: object or part pose trajectory/posterior in camera and world coordinates for rigid/articulated branches; residual uncertainty remains attached to the trajectory rather than replacing it with a surface-only proxy.
- `M_{i,t}`, `D_t`: image masks/tracks, depth observations, visible-surface samples, and their provenance.
- `V_{s,t}`, `V_{i,t}`: hand/object visibility and occlusion ownership states.
- `C_{s,i,t}`: contact/near-contact/non-contact hypothesis with contact-patch support and uncertainty.
- `U_t`: residual and uncertainty summary consumed by the renderer and evaluator.

These variables are full-timeline variables. A single-frame correction can be a measurement, but it cannot be the final V19 state unless it is propagated through the timeline state and rendered in the full-video artifact.

### 5.2 Robust objective

The default V19 solver should minimize a robust objective over the full timeline:

```text
min over H, O, G, C, V, T_world_cam
  E_hand_image(H)              # 2D/mesh/keypoint hand evidence
+ E_hand_metric_depth(H, D)    # metric depth/order consistency
+ E_hand_prior(H)              # MANO pose/shape and motion priors
+ E_camera(T_world_cam)        # SLAM/head trajectory and scale evidence
+ E_object_mask_depth(O, G, M, D)
+ E_object_rigidity_or_articulation(O, G)
+ E_contact(C, H, O, G)
+ E_occlusion(V, H, O, G, D)
+ E_nonpenetration(H, O, G)
+ E_temporal(H, O, C, V)
+ E_uncertainty_calibration(U)
```

All terms must use robust losses and preserve residuals. A failed or weak term should downweight or widen uncertainty; it should not silently delete the variable family or replace it with a proxy. Weights may be benchmark-tuned inside the autoresearch loop, but the tuned configuration and split must be recorded.

### 5.3 Agent authority versus solver authority

The agent owns hypothesis management and acceptance judgment; the solver owns numerical consistency. The agent may decide that a measurement is invalid, that an object hypothesis should be tested, or that a rendered contradiction requires another intervention. It may not replace the objective with an unsupported statement. A state claim becomes accepted only when it is written into the state variables, consumed by the renderer, and either evaluated quantitatively or reviewed visually with residual uncertainty preserved.

## 6. Object discovery and object state

### 6.1 Why V18 under-detected objects

V18 frequently focused on the object named by the immediate task, e.g. tomato in `task5_tomato_960`. That made sense for a rescue effort centered on interval MANO, but it is not sufficient for V19. In task5 the visible manipulation context includes tomato, bowl, pot, and supporting surfaces. If V19 only detects tomato, it cannot explain occlusion, support, contact alternatives, or audience-visible scene semantics.

### 6.2 Category-agnostic object roster

V19 should build an object roster before solving physical state:

1. Sample frames across the full video and around high hand motion/contact likelihood.
2. Agent proposes a candidate roster from visual evidence: manipulated objects, containers, tools, supporting surfaces, and likely occluders.
3. Open-vocabulary detection and segmentation gather masks/tracks for the roster.
4. Video object segmentation maintains temporal tracks.
5. Depth/SLAM builds visible surfaces and candidate 3D supports.
6. Agent reviews coverage failures and requests more prompts/tracks only where they can change the physical state.

The downstream path must be uniform. A tomato, bowl, pot, lid, tool, or support surface should all enter the same object-instance schema. Object-specific Python branches for color/category/action phrases are not a V19 perception strategy.

### 6.3 Object state-type decision and posterior

Each object instance should maintain candidate state types until the physical-state decision is made:

- **Visible surface measurement:** observed depth/mask-derived point or surfel support used as evidence and metric anchor.
- **Rigid object branch:** reconstructed or instance-adapted mesh with SE(3)/Sim(3) trajectory. Retrieved or TRELLIS/category/dataset meshes are priors until aligned to the observed instance.
- **Articulated/part branch:** linked rigid parts when motion evidence supports part motion.
- **Deformable/uncertain branch:** uncertainty volume or deformation model when the object is not judged rigid.
- **Out-of-frame/occluded state:** temporal carry with explicit visibility and uncertainty.

The important control-flow rule is that the VLM/agent physical-state decision selects the branch. Once an object is classified as rigid, visible surfaces are no longer an alternate deliverable replacing pose; they are measurements used to complete, fit, and correct the rigid mesh trajectory. Residuals can widen uncertainty, trigger another measurement, or mark a frame for repair, but they must not silently demote a rigid-object annotation to a point cloud or status label.

## 7. Rigid-object pipeline for tomato/task5

The tomato logic should be stated as a mechanism, not as a claim-retreat. When the V18 VLM physical-state path, or the V19 agent harness, judges an object to be rigid, the pipeline must execute the rigid-object branch and render the resulting object state. The branch is:

1. **Rigidity decision.** The VLM/agent-harness declares that a specific object instance is rigid over a frame span. This decision selects the rigid-object pipeline; it is not a category/name-specific Python branch.
2. **TRELLIS mesh completion.** Use TRELLIS — the local V18 code names this stage through `remote_run_trellis_shape_v3.py`, `build_v18_compact_rigid_trellis_completion.py`, and `microsoft/TRELLIS-image-large` — to complete a per-instance mesh from object crops/masks. Align the TRELLIS prior to observed depth/mask evidence; observed depth-fused visible surfaces remain the metric anchor and TRELLIS supplies hidden-surface completion.
3. **Visible-frame pose estimation.** Fit the completed mesh SE(3)/Sim(3) pose for every frame in which the object is visible and has mask/depth support. V18 reports such as `v18_compact_rigid_object_pose_fit_report.json` are the right stage type: they initialize from the graph/object state and refit against current visible depth samples.
4. **Factor-graph correction.** Put the object pose trajectory into the temporal factor graph with camera/depth corrections, MANO hand state, contact, occlusion, and nonpenetration terms. The graph corrects the object pose and hand/object relation jointly; residuals and uncertainty are carried in the state, not used to demote a rigid object back to a point cloud.
5. **Canonical render consumption.** The overlay/world/side-by-side renderer consumes the corrected rigid mesh pose as the object annotation. If a frame is occluded or out of view, the state should show the temporal/occlusion uncertainty for that rigid object, not replace the object with an unrelated visible-surface-only story.

For task5, this means the correct tomato story is: rigid decision -> TRELLIS completion -> pose on visible frames -> factor-graph pose correction -> rendered rigid tomato state. The harness must treat this chain as the object-pose mechanism and evaluate whether it ran, rendered, and produced acceptable pose/contact quality on the selected clips.

This gives the audience a real manipulated-object visualization when the object is rigid, without violating the project rule that object pose means reconstructed/adapted object geometry, not a centroid, primitive, or label.

## 8. Hand state and physical consistency

V19 should preserve V18’s strongest contribution: interval-level metric MANO rather than isolated-frame keypoints. The agent harness should treat WiLoR/HaMeR/HaWoR-style hand reconstructions as measurements, not final answers.

The V19 hand state should include:

- MANO pose/shape/global transform per hand and frame.
- Camera-frame and world-frame semantics.
- Visibility state: visible, partially visible, occluded, out-of-frame, unresolved.
- Observation provenance: model prediction, depth support, temporal infill, contact support, benchmark ground truth if applicable.
- Uncertainty intervals or samples over translation/pose where evidence is ambiguous.
- Temporal consistency terms that smooth without hiding occlusion or missing observations.

Physical consistency terms should remain approximate but explicit:

- hand depth/order consistency;
- object surface support;
- contact likelihood;
- occlusion ownership;
- nonpenetration as a soft/uncertain constraint;
- temporal motion priors;
- camera/head trajectory consistency.

A weak local measurement should continue downstream with uncertainty. It should not block the pipeline from producing an approximate artifact.

## 9. Visualization and audience presentation

V19 visualization should be designed for an external viewer, not for a developer reading residual fields.

### 9.1 Visual style

The reference style should be closer to the HaWoR website/demo world-view aesthetic: clean 3D hand meshes moving in a world coordinate frame, visible camera/head motion, and minimal text. V19 extends that style to manipulated objects and physical uncertainty.

The default world view should show:

- camera/head trajectory as a smooth path with current camera frustum;
- left/right MANO meshes with stable colors;
- manipulated object mesh or uncertainty volume;
- supporting surfaces/containers only when relevant;
- contact glyphs at likely contact patches;
- occlusion as translucent depth-order bands or hidden-hand ghost meshes;
- uncertainty as opacity/halo/interval envelope, not as dense text.

### 9.2 Key message panel

The overlay should use short audience-facing labels. These examples are task5-style illustrations, not category templates or object-specific branches:

```text
right hand grasping tomato (uncertain contact)
left hand partially occluded by bowl
object pose: rigid tomato mesh, corrected pose, residual uncertainty
```

Internal details such as factor names, residual thresholds, schema states, and validator names belong in the evidence report, not in the main video.

### 9.3 Deliverable views

V19 should ship three default views:

1. **Overlay view** for frame-local correctness.
2. **World view** for metric motion, head/camera path, and object/hand relationships.
3. **Narrative side-by-side** for demos and reviews: raw, overlay, world, and a small timeline strip.

An optional HTML/GLB/USD viewer should allow rotating the scene and inspecting object/hand uncertainty interactively, but MP4 videos remain the required artifact.

## 10. Quantitative evaluation plan

V19 needs an executable benchmark slice, not a survey. The active evaluation budget is one primary dataset and at most one secondary dataset, with a few clips each. Any additional dataset belongs in future-work notes or a later design amendment; adding it to the V19 acceptance gate without running it would dilute the evaluation to zero.

### 10.1 Active benchmark scope

| Slot | Dataset | Clip budget | Why it is in scope | What it tests |
|---|---|---:|---|---|
| Primary | HOT3D | 3-5 curated clips | Best single anchor for V19's joint egocentric hand/object/head/world claim; also matches the HaWoR evaluation ecosystem | MANO/UmeTrack hands, rigid object meshes/poses, headset/camera pose, visibility-aware render/metric checks |
| Secondary, optional | H2O | 2-3 clips | Egocentric RGB-D hand-object sequences with two-hand interaction and 6D object poses | RGB-D hand-object pose, object trajectory, two-hand manipulation, contact-adjacent sanity |

If H2O access/tooling blocks implementation, V19 may substitute DexYCB for the secondary slot because it has RGB-D, MANO, object pose, segmentation, and official evaluation tooling. H2O and DexYCB must not both be active in the initial V19 gate. No additional hand-only, 2D occlusion, state-change, or HOI dataset is part of V19 acceptance.

The selected primary and optional secondary clips must be named before tuning. Clip selection should cover only a few phenomena: visible rigid-object manipulation, partial occlusion, two-hand interaction when available, and a failure-prone object/contact span. A large external evaluation is explicitly out of scope for V19.

### 10.2 Baselines

V19 should compare against only baselines that will actually run on the selected clips:

- **V18 v5** on `task5_tomato_960` and `trash_1050`, for project-video monotonic comparison.
- **HaWoR** on HOT3D clips, for world-space egocentric hand motion and camera/head-aware hand reconstruction.
- **Dataset official/reference tracks** for the active HOT3D and optional H2O/DexYCB clips where the provided evaluator supports object pose or hand metrics.

WiLoR and HaMeR can remain measurement components or later ablation candidates. V19 comparison scope is limited to the selected clips unless a later release allocates a separate benchmark budget.

### 10.3 Metrics

V19 should report metrics only where the active clips provide ground truth or official evaluation support:

**Hand pose / MANO**
- MPJPE and PA-MPJPE in mm.
- MANO vertex error where ground-truth mesh/parameters exist.
- Wrist/global translation error where camera-frame/world-frame pose is evaluated.
- Temporal jitter/acceleration error on video sequences.
- Visibility-stratified errors: visible, partially visible, occluded, out-of-frame.

**World/camera/head motion**
- Absolute and relative trajectory error where headset/camera pose is available.
- Reprojection consistency against camera intrinsics.
- Scale error when metric depth/SLAM estimates scale.

**Object pose/geometry**
- BOP metrics: VSD, MSSD, MSPD where official object meshes and poses exist.
- Mask IoU/silhouette error as diagnostic evidence unless paired with geometry/depth support.
- Depth residual distribution on visible object surfaces.
- Rigid-vs-nonrigid residual evidence for physical-state selection before branch commitment; after rigid commitment, residuals measure pose quality and repair needs.

**Contact/occlusion**
- Contact patch precision/recall where contact/contact-region annotations exist.
- Signed hand-object distance distribution, stratified by claimed contact/no-contact, as diagnostic evidence unless paired with geometry/depth/contact support.
- Occlusion ownership accuracy where ground truth exists; proxy agreement with visibility masks is diagnostic only and cannot ground an acceptance claim by itself.
- Calibration of uncertainty: do high-uncertainty spans contain more errors?

**Object roster and state change**
- Manipulated-object recall on the selected project videos and active HOT3D/H2O clips.
- Track continuity and identity switches on those clips.
- State-change AP is not a V19 metric unless a later design amendment replaces the secondary slot with a state-change task.

**Runtime and usability**
- Wall-clock runtime per video minute.
- GPU memory peak.
- Fraction of run time spent in agent reasoning, model inference, rendering, and evaluation.
- Failure rate by input mode.

### 10.4 Ablations

Required V19 ablations:

1. Agent harness vs fixed script ordering.
2. Agent-native visual judgment vs isolated VLM JSON calls.
3. Full object roster vs primary-object-only detection.
4. Rigid-object branch enabled vs visible-surface-only ablation after a rigid physical-state decision.
5. Occlusion-aware MANO intervals vs direct framewise hand predictions.
6. World/camera trajectory enabled vs camera-frame-only hand reconstruction.
7. Contact/nonpenetration terms enabled vs hand-only temporal smoothing.
8. Audience renderer enabled vs developer/status renderer.

Each ablation should specify the claim it can falsify before it is run. A result that can be explained by several mechanisms without changing the next intervention is bookkeeping, not research.

## 11. Autoresearch loop

The user’s proposed autoresearch direction is appropriate only if benchmark discipline is built in first. V19 should define an autoresearch harness with these rules:

1. **Fixed benchmark registry.** The small active clip list, metrics, and any held-out clips are declared before tuning. V19 starts with HOT3D plus at most one secondary dataset; expanding the dataset list requires a new design amendment.
2. **Prediction before run.** The agent writes what a proposed change should improve and what failure would falsify the mechanism.
3. **Atomic interventions.** One mechanism change per run: e.g. object roster expansion, occlusion factor, depth scale correction, renderer clarity.
4. **Automatic evaluation.** Every run produces metrics, rendered samples, and failure clusters.
5. **Adversarial review.** A clean-room or separate agent reviews whether the metric improvement corresponds to visible physical improvement.
6. **No overfitting path.** The harness keeps a lockbox set and reports validation/test separately.
7. **No category hacks.** Improvements cannot be hand-written for tomato/bowl/pot or benchmark-specific object IDs unless the split explicitly studies a domain-specific model.
8. **Commit and provenance.** Each accepted improvement is tied to code/config commits and benchmark artifacts.

The autoresearch loop should optimize benchmark performance, but it should not replace physical judgment. If a benchmark metric improves while the rendered annotation becomes physically less coherent, the run is a failure requiring explanation.

## 12. Representative videos, expected failure modes, and acceptance checks

### 12.1 Representative video set

The project-regression representatives are:

1. `task5_tomato_960`: tomato/bowl/pot/support-surface manipulation; stresses object roster completeness, tomato rigid-vs-uncertain presentation, two-hand MANO continuity, support/contact uncertainty, and audience-facing visualization. Current verified raw clip from the V16 raw-frame manifest: `/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4` (`960` frames, `30fps`, `1920x1080`).
2. `trash_1050`: trash-can/lid/occlusion sequence; stresses occluded-hand intervals, lid/object occlusion ownership, late hand visibility transitions, world/camera rendering, and uncertainty communication. Current verified raw clip from the V16 raw-frame manifest: `/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4` (`1050` frames, `30fps`, `1920x1080`).

The future V19 implementation must record the exact raw video file path, decoded frame count, FPS, duration, and any calibration/depth side inputs in each run's `input/input_manifest.json`. If only an existing raw-frame manifest is available, V19 may use it as a development input, but a release claim must still identify the raw-video source or explicitly mark raw-video provenance as unresolved.

Benchmark representatives are not a broad matrix. V19 uses 3-5 HOT3D clips as the primary external benchmark and, only if budget permits, 2-3 H2O clips as the secondary hand-object benchmark. DexYCB is a fallback for the secondary slot if H2O access/tooling blocks implementation. No other dataset is part of V19 closure.

### 12.2 Expected failure modes

V19 should expect and render these failure modes instead of hiding them:

- **Metric-scale ambiguity:** raw-video-only inputs may not support strong metric scale without depth/camera evidence.
- **SLAM/head-pose failure:** fast ego motion, blur, low texture, or moving foreground can break camera trajectory estimates.
- **Hand measurement failure:** severe occlusion, out-of-frame hands, motion blur, left/right swaps, and hand-object overlap can corrupt framewise hand models.
- **Object roster misses:** open-vocabulary detection can miss transparent, reflective, small, deformable, or partially occluded manipulated objects.
- **Object geometry underconstraint:** visible surfaces may weakly constrain a rigid mesh/pose; if the physical-state decision is rigid, V19 keeps the rigid branch active, renders residual/pose uncertainty, and marks repair targets rather than falling back to a point-cloud-only deliverable.
- **Contact ambiguity:** mask/depth overlap can look like contact without metric surface support; signed distances can look separated when depth is biased.
- **Occlusion ownership ambiguity:** hidden hands or objects may be plausible under several depth-order hypotheses.
- **Runtime overrun:** a branch may exceed the default runtime budget and must be demoted to offline research.
- **Benchmark/domain mismatch:** a metric may improve on a narrow object set while project-video visual coherence degrades.
- **Agent overconfidence:** the harness may propose a plausible physical narrative not supported by state variables or rendered evidence; clean-room review must catch this.

### 12.3 Acceptance checks

A future V19 implementation should not close until these checks pass for both representative project videos and for the selected small benchmark slice:

1. **Full-duration render check:** overlay/world/side-by-side videos match input frame count, FPS, and duration.
2. **State-to-render check:** every visible annotation layer is driven by `state/` variables, not private measurement files or status text.
3. **Object roster check:** manipulated objects, containers, tools, supports, and occluders visible in representative review frames are either tracked or explicitly marked unresolved.
4. **Geometry identity check:** every object-pose claim points to per-instance reconstructed/adapted/dataset-instance geometry evidence; category priors alone cannot satisfy object pose.
5. **MANO provenance check:** every hand state has camera/world semantics, visibility, provenance, and uncertainty.
6. **Contact/occlusion check:** contact and occlusion claims carry supporting depth/geometry/temporal evidence or visible uncertainty.
7. **Runtime/compute check:** heavy tools ran on the declared server-class target, local execution stayed light, and the default path stayed within its runtime budget.
8. **Benchmark check:** each reported metric is tied to a dataset that actually annotates the corresponding claim family.
9. **Ablation check:** every accepted improvement has an ablation that changes a physical artifact or a claim-specific metric, not only a schema/report.
10. **Clean-room review check:** an independent review consumes the rendered videos and benchmark bundle as a user would and records any first-glance physical contradiction.

## 13. Acceptance criteria for a future V19 implementation

V19 can be called implemented only when all of the following are true:

1. A Pi agent-harness entry point accepts an arbitrary input video path, produces a full-duration artifact directory, and exposes uncertainty when the video cannot support strong metric hand/object/contact claims.
2. The harness uses GPT-5.5 through the `occ` provider or a documented equivalent route, with a V19 system prompt replacing the generic Pi prompt for the annotation session.
3. The pipeline builds an object roster that includes all visibly manipulated or physically relevant objects in the representative project videos, not only the named target object.
4. The renderer shows MANO hand meshes, camera/head trajectory, manipulated object hypotheses, contact/occlusion cues, and uncertainty in audience-readable form.
5. The artifact runs on the bounded benchmark slice: 3-5 HOT3D clips, plus optionally 2-3 H2O clips or a DexYCB fallback occupying the same secondary slot. No extra dataset is required for V19 closure.
6. V19 reports ablations against V18 v5 and HaWoR on the selected clips. WiLoR/HaMeR may appear only as internal component ablations if they are actually used in the V19 run, not as separate external-evaluation claims.
7. Tomato/task5 presentation follows the rigid-object branch when the VLM/agent-harness classifies tomato as rigid: TRELLIS completion, visible-frame pose estimation, factor-graph pose correction, and canonical overlay/world/side-by-side rendering of the corrected rigid object state.
8. Every final report ties claims to project-video and selected-clip evidence; pose/contact/nonpenetration failures are rendered as residuals, uncertainty, and repair targets in the artifact and evaluation report.

## 14. Proposed implementation phases for later work

No phase below is implemented by this design document.

### Phase A — Pi-native harness entry

- Create V19 system prompt.
- Create the Pi-native project entry prompt and run contract using provider `occ` and model `gpt-5.5:xhigh` through the Pi CLI/session.
- Define tool allowlist and artifact-root contract.
- Run through the Pi entry prompt on a tiny or representative input only to prove input/state/render plumbing; do not create an outer SDK/script wrapper around Pi.

### Phase B — Self-contained measurement instruments

- Frame extraction, video manifest, optional calibration ingest.
- Hand model adapters for WiLoR/HaMeR/HaWoR-style measurements.
- Depth/SLAM/camera trajectory measurement path.
- Open-vocabulary object roster and SAM/video-mask tracking path.

### Phase C — Physical state and branch commitment

- Interval MANO state writer.
- Object state-type decision writer and branch-specific state writers: rigid mesh/pose trajectory, articulated/part state, deformable/uncertain state, and visible-surface measurements as evidence.
- Contact/occlusion/nonpenetration uncertainty state.

### Phase D — Audience renderer

- Overlay/world/side-by-side videos.
- Camera/head trajectory visualization.
- Mesh/uncertainty styling.
- Narrative review sheets.

### Phase E — Bounded benchmark adapters

- HOT3D adapter for the primary 3-5 clip benchmark slice.
- H2O adapter for the optional 2-3 clip secondary slice, or DexYCB adapter only as a replacement if H2O access/tooling blocks the secondary slot.
- No adapters for additional hand-only, 2D occlusion, state-change, or HOI datasets in V19.

### Phase F — Autoresearch

- Fixed benchmark registry.
- Atomic intervention queue.
- Prediction/evaluation/review ledger.
- Held-out lockbox and clean-room review.

## 15. Source anchors

The design relies on these current source observations:

- V18 baseline evidence: `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`, especially `v18_current_frontier_interval_mano_artifact_manifest.json` and `v18_frontier_uncertainty_classification.json`, is the scoped bounded-MANO baseline that V19 should compare against.
- Pi docs: installed package files `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/models.md`, `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/usage.md`, and `/home/yiwen/.npm-global/lib/node_modules/@earendil-works/pi-coding-agent/docs/prompt-templates.md` document custom models in `~/.pi/agent/models.json`, CLI `--provider`/`--model`/`--system-prompt`, and project prompt templates under `.pi/prompts/*.md`. The earlier SDK-created-session idea is not the V19 implementation route.
- [HaWoR](https://arxiv.org/abs/2501.02973), [official repo](https://github.com/ThunderVVV/HaWoR): world-space egocentric hand reconstruction baseline/component for HOT3D-style clips.
- [WiLoR](https://arxiv.org/abs/2409.12259), [official repo](https://github.com/rolpotamias/WiLoR): multi-hand localization/reconstruction component candidate for selected-clip ablations.
- [HaMeR](https://arxiv.org/abs/2312.05251), [official repo](https://github.com/geopavlakos/hamer): transformer hand mesh recovery component candidate for selected-clip ablations.
- [HOT3D](https://facebookresearch.github.io/hot3d/), [toolkit](https://github.com/facebookresearch/hot3d): primary V19 external benchmark slice, limited to 3-5 selected clips.
- [H2O](https://taeinkwon.com/projects/h2o/), [arXiv](https://arxiv.org/abs/2104.11181): optional secondary benchmark slice, limited to 2-3 selected clips if access/tooling permits.
- [DexYCB toolkit](https://github.com/NVlabs/dex-ycb-toolkit): fallback for the secondary benchmark slot only if H2O is blocked; not an additional V19 dataset.
