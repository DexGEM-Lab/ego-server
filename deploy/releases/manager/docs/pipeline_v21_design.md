# Pipeline V21 Design: V19 Spine Plus Depth And Mesh Extensions

Status: upfront V21 design document. This document defines the V21 harness before implementation. It is meant to be read directly by Pi/runtime agents and by implementers. It does not claim that V21 scripts already exist unless the named scripts are already present in this repository.

## 1. Grounding

V21 starts from three observations.

1. V19 defines the correct physical-state spine: Pi is the harness, `state/` is the renderer boundary, scripts are tools, and rigid manipulated objects require completed/adapted geometry, pose fitting, factor correction, and mesh-pose rendering.
2. V20 implemented real components and produced full-duration approximate outputs, but the current normal outputs remain too weak for the user's target: object pose quality is poor, masks can select the wrong object or table/background, metric hand state remains uncertain, and arbitrary infer often renders visible-surface scatter rather than fitted object mesh pose.
3. V20's two-mode idea is still correct: `infer` should annotate arbitrary inputs, and `benchmark` should run the same prediction path on GT datasets with GT sealed until after prediction-side state and renders exist.

V21 therefore is not a rewrite wrapper and not a continuation of V20's weakest closure path. It is:

```text
V19 physical-state spine
  + V20 infer / benchmark mode split
  + V20 GT isolation and candidate-evidence discipline
  + depth/camera modality support for no-depth, monocular, native depth/RGB-D, stereo, and multiview inputs
  + additional point-cloud/mesh-completion candidates inside the geometry cluster
  + sample-bound internal tuning only when an atomic algorithm output shows large deviation
  + rigid object mesh-pose closure as the default manipulated-object target
```

V21 does not replace the V19 bbox or segmentation spine. OWLv2/SAM2/object-plan prompt components may receive interface adapters when upstream or downstream schemas change, but their algorithm semantics must not be retuned or replaced merely because V21 adds depth or mesh-completion algorithms.

## 2. Non-negotiable outputs

A completed V21 run must produce a single run root containing full-duration or explicitly requested benchmark-span artifacts:

```text
v21_runs/<run_id>/
  input/
  measurements/
  tuning/
  state/
  renders/
  evaluation/
  logs/
```

The required user-facing artifacts are:

- `renders/v21_overlay.mp4`: raw video with physically grounded hand/object annotations.
- `renders/v21_world.mp4`: metric world/camera view with MANO surfaces and fitted/adapted object mesh poses when those claims are made.
- `renders/v21_side_by_side.mp4`: synchronized raw/overlay/world views.
- `state/v21_physical_state.json`: renderer-consumed physical state.
- `state/v21_uncertainty_state.json`: unresolved or weak physical variables with causes.
- `state/v21_agent_evidence.md`: observations, interpretations, commitments, tuning attempts, and visual review conclusions.
- `logs/harness_events.jsonl`: run events, compute target, commands, parameter changes, and artifact paths.

Benchmark mode additionally produces per-iteration metrics, failure clusters, parameter changes, and final selection evidence under `evaluation/`.

A JSON registry, validation pass, report field, frame count, or rendered container is not a V21 deliverable unless it changes or explains the visible physical annotation.

## 3. Physical variables

V21 keeps V19's physical variables and makes V20 additions subordinate to them:

- `K_t`: camera intrinsics and distortion, observed or inferred, with provenance and uncertainty.
- `T_world_camera,t`: camera/head trajectory and metric scale provenance.
- `D_t`: depth candidates and selected depth evidence by scope, including native depth, stereo depth, monocular/foundation depth, and weak/nonmetric evidence when applicable.
- `H_s,t`: MANO hand pose, shape, scale, global transform, camera/world semantics, visibility, provenance, and uncertainty per side/track.
- `M_i,t`: target object masks/tracks and observed/inferred mask regions with provenance.
- `G_i`: object geometry branch state. For a rigid manipulated object this means a completed/adapted instance mesh, not a centroid, primitive, box, category prior, or visual patch.
- `T_world_object,i,t`: object pose/posterior for rigid/articulated branches.
- `C_s,i,t`: contact/near-contact/non-contact state with patch/distance evidence and uncertainty.
- `V_s,t`, `V_i,t`: hand/object visibility and occlusion ownership.
- `N_s,i,t`: nonpenetration residuals and uncertainty.
- `U_t`: residual summaries and uncertainty consumed by render and benchmark evaluation.

## 4. V21 modes

### 4.1 `v21_infer`

Input: arbitrary egocentric video plus optional calibration/depth/camera/prior hints.

Output: full-duration physical annotation render/state. When information is insufficient, V21 must still render the best supported uncertain state, but must not substitute proxies for named physical variables.

### 4.2 `v21_benchmark`

Input: a supported GT dataset sample or sample list.

Output: the same prediction-side physical annotation flow as `v21_infer`, then sealed GT evaluation after prediction state and renders exist. Each benchmark iteration must record a mechanism diagnosis and one atomic parameter/model/branch intervention before rerunning affected stages.

GT may guide the next prediction-side intervention only after a completed prediction-side iteration. GT must never enter prediction manifests, candidates, masks, depth selection, state, render, or algorithm choices before evaluation.

## 5. V20 lessons carried into V21

### Keep

- Pi remains the harness; scripts are tools.
- Target objects must come from visual/task evidence, not from public dataset object rosters.
- Benchmark GT isolation and post-prediction evaluation are required.
- Depth, geometry, and hand-shape candidates should carry provenance, uncertainty, residuals, and promotion status.
- Generated geometry is a candidate until fitted, validated, optimized, and rendered through state.

### Do not keep as final mechanisms

- Visible-surface-only centroids or scatter plots as object pose.
- 2D detector-box overlay alignment as metric MANO repair.
- Uncalibrated stereo disparity as metric depth.
- Render-only contact points as contact evidence.
- A temporal smoother over centroids/joints as a full V19-style physical factor graph.
- `run_summary.json` or status fields as source of truth when final state/render disagree.

## 6. Atomic Large-Deviation Tuning Contract

V21 introduces a harness-level rule for atomic algorithm outputs with large deviation. It applies only after an output has been measured or visually reviewed and the evidence shows a large deviation that would poison downstream fusion.

When such a deviation is observed, the agent must not immediately lower that atom's weight and continue. The agent must first attempt a sample-bound internal parameter change inside the same atomic algorithm, unless a documented missing implementation prevents it. This is not permission to change inherited V19 bbox or segmentation semantics; those components can be tuned only when their own rendered output shows a large deviation, and only through their established prompt/threshold/keyframe interfaces.

### 6.1 Required strong-tuning cycle

For every large-deviation observation failure, the agent writes a tuning record under:

```text
$RUN_ROOT/tuning/<variable_family>/<sample_or_track_id>/attempt_<k>.json
```

Each attempt records:

- physical variable blocked;
- raw observation that shows weakness;
- algorithm family and concrete backend;
- algorithm principle brief, including the parameter that controls the suspected failure mechanism;
- current parameter set;
- proposed internal parameter change;
- prediction before running the change;
- output artifacts and residuals after the change;
- visual review result;
- decision: keep, reject, or continue tuning;
- sample/run/input hash binding for the parameter set.

The associated ledger entry goes in `state/v21_agent_evidence.md`.

### 6.2 No cross-sample silent reuse

An algorithm-internal parameter set tuned for one sample may not silently become a global default. It can be reused only if the run explicitly records:

- original source sample/input hash;
- target sample/input hash;
- reason the failure mechanism is expected to match;
- validation evidence on the target sample.

### 6.3 When downweighting is allowed

Downweighting an atomic observation is allowed only after:

1. at least one causally targeted strong-tuning cycle has run or a documented missing implementation prevents it;
2. the residual/visual evidence still shows the atom's output is weak;
3. the state records why further tuning is unlikely to change the downstream physical claim within the run budget;
4. the renderer exposes the resulting uncertainty.

## 7. Monocular baseline requirement for depth, stereo, RGB-D, and assisted segmentation

Whenever V21 enters a native-depth, RGB-D, stereo, multiview, or depth-assisted path, it must also run a monocular baseline on the same RGB frames or keyframes.

The monocular baseline has two roles:

1. It is a sanity floor: additional sensor modalities must not make the result worse without diagnosis.
2. It is a fallback evidence source when calibration, registration, stereo rectification, or depth scale fails.

### 7.1 Depth/camera comparison

For each depth/camera path, compare against the monocular baseline using available residuals:

- intrinsics/focal consistency;
- RGB-depth registration residual;
- object-mask depth continuity;
- visible-surface reprojection residual;
- hand-depth residual;
- hand/object depth-order residual;
- temporal depth/camera smoothness;
- scale plausibility from object/hand priors;
- downstream geometry pose fit residual;
- visual review of obvious scale or alignment failures.

A native-depth or stereo result that is worse than monocular must trigger calibration/registration/rectification/depth-scale tuning before it can be selected as primary metric depth.

### 7.2 Segmentation comparison

V21's active target mask source is SAM2 proper propagated from approved OWLv2 bbox prompts. If depth/stereo/multiview assistance is added later, it must be compared against this active track for the same target/keyframes. Compare:

- target identity correctness;
- boundary quality;
- missing visible object pixels;
- background/table contamination;
- temporal propagation stability;
- mask-depth edge agreement;
- downstream visible-surface and geometry-fit residuals;
- rendered overlay review.

If an assisted path is worse than the active bbox-prompt SAM2 track, the agent must first diagnose whether the deviation comes from the new assisted modality/registration path or from the OWLv2 bbox/SAM2 propagation interface. Tune only the component that produced the deviation, using its existing interface parameters; do not introduce a new bbox-refinement loop as the default path.

## 8. Depth and camera design

V21 supports these input modalities:

1. **Native depth / RGB-D:** treat sensor depth as privileged but not unquestioned. Validate unit, scale, intrinsics, extrinsics, RGB-depth registration, valid mask, and flying-pixel/edge risk.
2. **Calibrated stereo:** use rectification and stereo-specific disparity/depth. Validate left-right consistency, disparity confidence, occlusion/disocclusion masks, and metric scale.
3. **Uncalibrated stereo:** may produce weak relative evidence only. It cannot support metric depth, object pose, contact, or nonpenetration claims until calibration/rectification is obtained or estimated.
4. **Monocular RGB:** use metric/foundation depth and camera estimation candidates such as UniDepth, DepthPro, DROID, VGGT, DUSt3R/MASt3R-style methods when available.
5. **Benchmark datasets:** use dataset-provided RGB/depth/calibration as prediction-side measurements while keeping GT sealed for evaluation.

If intrinsics are provided, V21 must use and validate them. If intrinsics are absent, V21 must estimate them and record focal/scale uncertainty. A guessed default focal may be an initialization, not an accepted camera state.

## 9. Segmentation design

V21 treats target segmentation as a first-class physical bottleneck.

Default flow:

```text
raw/source frame manifests
  -> agent object plan and rejected alternatives
  -> agent-selected OWLv2 detector keyframes
  -> OWLv2 open-vocabulary bbox proposals on keyframes
  -> agent/approval-selected target bbox prompts
  -> SAM2 proper full-video propagation from approved boxes
  -> contamination review sheet under segmentation_sam2_proper
  -> accepted target mask track or explicit unresolved state
```

A mask may enter visible-surface reconstruction only if the agent has reviewed target identity and contamination evidence. GroundingDINO bbox proposals are disabled as the V21 default source; historical GroundingDINO outputs are deprecated evidence only and must not feed default SAM2 or geometry. Wrong-object masks, table/background capture, major missing object parts, or coordinate/prompt miscoupling are systematic errors, not weak observations to be fused away. V21 must not add a SAM-mask-feedback bbox algorithm unless a later design amendment explicitly changes the segmentation cluster; bbox/segmentation changes here are interface-only.

## 10. Object geometry and pose design

Once an object is classified as rigid over a span, V21 must run the rigid branch:

```text
rigid decision
  -> accepted target mask and selected depth/camera evidence
  -> visible surface samples
  -> evidence-conditioned mesh completion/adaptation
  -> mesh-to-visible evidence alignment
  -> per-frame SE(3)/Sim(3) pose fitting
  -> temporal/object factor graph correction
  -> corrected mesh-pose render
```

For rigid manipulated objects, visible surfaces are metric measurements and anchors; they are not the final object pose. A generated mesh, public CAD mesh, or retrieved mesh is a prior/candidate until adapted to the observed instance and fitted to the visible evidence.

The geometry condition packet must include:

- target object id and branch;
- keyframes and crops;
- accepted masks and visible surfaces;
- selected depth/camera candidate ids;
- metric extent hints and uncertainty;
- occlusion/contact notes;
- negative constraints such as observed free space and forbidden background fill.

## 11. Hand / MANO design

V21 must repair metric MANO, not just visual overlay.

Hand state starts from candidate streams such as RTMLib 2D, WiLoR, HaWoR, HaMeR, or equivalent hand models. Before accepting a hand state, the agent must diagnose:

- detector box quality;
- left/right side mapping;
- crop/resize convention;
- camera intrinsics and distortion;
- model coordinate frame;
- MANO global transform and scale;
- depth alignment and visible hand depth support;
- temporal consistency;
- occlusion visibility state.

V21 optimizes track-level hand shape/scale/betas together with per-frame pose/root/translation. Shape is not accepted by taking the median of upstream betas. The objective should include available 2D keypoints, hand masks/silhouettes, visible hand depth, bone-length priors, MANO pose/shape priors, temporal smoothness, contact compatibility, and nonpenetration residuals.

If metric camera/depth support is absent, the renderer must distinguish overlay-only or uncertain hand visualization from metric MANO state.

## 12. Factor graph and state acceptance

V21's final physical state must come from an integrated optimization or an explicitly scoped unresolved state. The graph should include robust terms for:

- hand image evidence;
- hand metric depth;
- MANO pose/shape/scale priors;
- camera/depth/scale evidence;
- object mask/depth/silhouette evidence;
- object geometry and rigidity/articulation;
- contact and near-contact;
- occlusion/depth order;
- nonpenetration/free space;
- temporal consistency;
- uncertainty calibration.

Weak measurements continue downstream only after any explicitly triggered large-deviation tuning obligations have been satisfied and uncertainty is visible in state/render. Contract errors must be fixed or represented as separate hypotheses, not downweighted.

## 13. Benchmark loop

`v21_benchmark` runs the same prediction-side flow as `v21_infer`. After prediction render/state exists:

1. GT evaluator reads sealed references and writes metrics, alignment semantics, and failure clusters.
2. Controller maps each failure cluster to a physical mechanism.
3. Controller selects one atomic intervention: model branch, algorithm-internal parameter, prompt/ROI/keyframe, calibration/registration, graph weight, or branch decision.
4. Controller predicts the expected metric/render change and falsifier.
5. Controller reruns only affected stages and then rerenders/evaluates.
6. Iteration stops at max budget, plateau, no supported intervention, missing implementation, or overfitting risk.

The benchmark goal is not a single metric table. It is documented iterative improvement until supported metrics and visual sanity checks plateau.

## 14. Runtime and compute

V21 preserves the project compute invariant: heavy inference and rendering run on the declared server/A800/authorized compute target, not accidentally on the local workstation. Every heavy stage records compute target, environment, GPU, wall-clock duration, and output path.

The default pipeline must remain within a bounded runtime budget. Large sweeps are allowed only inside benchmark iterations or offline research branches and must not become the default `v21_infer` path.

## 15. Acceptance criteria

A V21 run is acceptable only if:

- full-duration or explicitly requested benchmark-span render frame counts match source frames;
- `state/` drives the renderer;
- target masks are visually reviewed and contamination status is recorded;
- depth/stereo/RGB-D paths have monocular baselines and comparison evidence;
- atomic algorithm outputs with measured large deviation have sample-bound tuning records before downweighting;
- rigid manipulated objects render fitted/adapted mesh poses, not centroids or scatter points;
- hand state distinguishes metric MANO from overlay-only visualization;
- contact/occlusion/nonpenetration are explicit variables or explicit unresolved states;
- benchmark GT is sealed until post-prediction evaluation;
- every tuned parameter set is bound to sample/run/input hash;
- unresolved variables are visible in `v21_uncertainty_state.json` and render/evidence.
