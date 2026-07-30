# V21 Algorithm Research And Implementation Plan

Status: algorithm research synthesis and implementation plan. This document does not claim V21 physical annotation has run. It selects candidate algorithms and specifies how to integrate them into the V21 harness so the next implementation step can change rendered physical annotations.

## 1. Research Scope And Evidence

The user requirement is to move from V21 harness policy to concrete algorithms for:

1. depth/camera across no-depth/V19 behavior, monocular, dataset camera parameters, native depth/RGB-D, stereo, and multiview inputs;
2. V19/master bbox and segmentation spine preservation with interface adapters only where depth/camera or geometry inputs change;
3. metric MANO hand pose/shape/scale optimization through the inherited hand-state cluster;
4. additional point-cloud/mesh-completion candidates inside the geometry cluster, followed by rigid object pose;
5. benchmark-driven, sample-bound internal tuning only when an atomic algorithm output shows large deviation.

Evidence used:

- Repo inventory: `docs/v21_component_extraction.md`, `docs/pipeline_v21_design.md`, V18/V19/V20 scripts under `scripts/`.
- Official READMEs fetched under `/tmp/v21_research_sources/`: Depth Pro, UniDepth, Metric3D, SAM2, OWLv2/GroundingDINO background material, DUSt3R, MASt3R, RAFT-Stereo, CREStereo, IGEV-Stereo, HaMeR, WiLoR, TRELLIS, Hunyuan3D-2.1, TripoSG, SPAR3D.
- Source fetch limitations: FoundationStereo README fetch returned `404`; OmniHands/EgoForce README URLs returned `404`. These are not selected as primary V21 defaults from external evidence, though repo runners may remain optional experimental branches.

## 2. Key Research Conclusions

### 2.1 Depth / Camera

Observations:

- Depth Pro claims zero-shot metric monocular depth with absolute scale and focal length prediction from a single image, producing 2.25MP depth in about 0.3s on a standard GPU. It is strong for the V21 monocular baseline because it predicts both depth and focal length.
- UniDepth supports metric depth and intrinsics prediction from RGB only, and can also consume known camera intrinsics. The repo already has `scripts/run_unidepth_full_frame_v3.py`.
- Metric3D V2 is a zero-shot metric depth and surface-normal model, but its README emphasizes focal length sensitivity; wrong focal causes distorted point clouds. This makes it useful as a candidate, but only with explicit focal tuning.
- DUSt3R/MASt3R provide dense 3D reconstruction/camera evidence from image pairs or multi-view images. MASt3R adds matching and scalable alignment. They are appropriate for multiview/camera hypotheses, not as the only metric truth without scale/camera validation.
- RAFT-Stereo, CREStereo, and IGEV-Stereo are stereo disparity candidates. RAFT-Stereo README explicitly states that known intrinsics and baseline are required to convert disparity to depth. Therefore stereo is metric only after calibration/rectification/baseline are known or estimated and validated.

Commitment:

- V21 default monocular baseline: run Depth Pro and UniDepth on selected keyframes/full frames where runtime permits. Depth Pro gives fast depth+focal; UniDepth gives depth+intrinsics and can consume known intrinsics.
- V21 optional monocular candidate: Metric3D V2 when focal tuning is needed or surface normals help object geometry.
- V21 multiview/camera candidate: MASt3R first, DUSt3R second, VGGT existing repo path when already integrated.
- V21 stereo candidate: RAFT-Stereo default, IGEV-Stereo as second candidate for benchmark/tuning, CREStereo as fallback where its environment is easier. All stereo candidates remain nonmetric unless calibration/rectification/baseline are valid.

### 2.2 Segmentation

Observations:

- SAM2 is designed for promptable image and video segmentation and has video predictor APIs with prompts and propagation. The README notes SAM2.1 checkpoints and multi-object video support.
- OWLv2 is the V21 default open-vocabulary bbox source because the latest master/V18 path already uses OWLv2-to-SAM2 for model-produced boxes and tracks. GroundingDINO is disabled for the V21 default bbox path by user request; historical GroundingDINO outputs may be kept only as deprecated evidence.
- Repo already has `scripts/run_v21_owlv2_bbox_proposals.py` for OWLv2 keyframe bbox proposals and `scripts/run_v21_sam2_proper_segmentation.py` for bbox-prompt SAM2 propagation. Under the clarified V21 scope, the active raw-to-segmentation spine is agent keyframes -> OWLv2 proposals -> approved bbox prompts -> SAM2 proper.

Commitment:

- V21 default target discovery uses agent-selected keyframes and OWLv2 proposals, not public object rosters. If OWLv2 is unavailable, the agent records the missing backend; GroundingDINO is not a fallback.
- V21 default segmentation uses SAM2 proper seeded only by approved OWLv2 bbox prompts.
- V21 does not add bbox continuity, SAM-mask validated bbox tightening, or local-mask replacement as default algorithms. Prompt/keyframe/threshold changes are allowed only as sample-bound internal tuning when that atom's own output shows large deviation.
- V21 hard rule: no mask enters geometry until a contamination review sheet confirms target identity, boundary coverage, missing visible pixels, background/table contamination, and temporal drift.

### 2.3 Hand / MANO

Observations:

- HaMeR reconstructs hands in 3D with transformers and requires MANO assets; repo already has HaMeR/RTMLib integration.
- WiLoR is an end-to-end 3D hand localization and reconstruction model in the wild and has repo integration. Its README notes detector and MANO model requirements.
- Repo already contains strong hand optimization pieces: RTMLib 2D evidence, WiLoR/HaMeR/HaWoR candidates, metric-depth refit, mask-depth MANO fit, contact-aware MANO graph, and interval solvers.
- Current V20 hand-shape posterior is only a candidate/prior. V21 requires active optimization over track-level betas/scale plus per-frame pose/root/translation.

Commitment:

- V21 default hand candidate stack: RTMLib 2D keypoints + WiLoR full-frame + HaMeR from RTMLib boxes. HaWoR remains optional when environment is available but must pass camera/crop convention checks.
- V21 accepted hand state comes from a new active optimizer, not direct upstream output. It must optimize track-level betas/scale and per-frame pose/root/translation against keypoints, hand masks, depth, temporal priors, and optional object/contact/nonpenetration terms.
- Existing `optimize_contact_aware_mano_graph_v8.py` is the closest starting point but currently optimizes per-frame pose/orient/translation/log-scale while using betas as input. V21 must extend or wrap it so betas become track-level variables.

### 2.4 Object Geometry / Pose

Observations:

- TRELLIS is an image/text-conditioned large 3D asset generation model that can output meshes. Its README explicitly recommends image-conditioned models for better performance and supports mesh outputs.
- Hunyuan3D-2.1 is an open-source image-to-3D asset generation system with shape and PBR texture models, with VRAM around 10GB for shape and 29GB for shape+texture. It is suitable as a high-quality candidate but not default for fast infer.
- TripoSG is an image-to-3D mesh generation foundation model with a direct image-to-GLB inference command. It is a strong candidate for mesh priors.
- SPAR3D is a fast feedforward single-image 3D mesh reconstruction model using point-cloud conditioning; it is promising for fast candidate generation and backside repair, with default VRAM around 6GB and low-VRAM mode.
- Repo already has TRELLIS/Hunyuan/TripoSG/SPAR3D remote runners, geometry registries, validators, and rigid pose graph scripts. Missing V21 pieces are condition packet standardization, mesh candidate selection strategy, and state/render adapters.

Commitment:

- V21 default geometry candidate order for manipulated rigid objects:
  1. If public CAD exists in benchmark prediction inputs: use CAD as candidate, fit to visible depth/mask, but only for selected target objects.
  2. If no CAD: run TRELLIS image-to-3D from best target crop/mask keyframes.
  3. Run SPAR3D as fast alternative candidate when installation/model access is available.
  4. Run TripoSG or Hunyuan3D as high-quality candidate in benchmark/strong-tuning iterations when default candidates fail.
- Every generated mesh is a candidate, never accepted geometry. It must pass visible-surface alignment, silhouette projection, free-space/nonpenetration, scale, and temporal pose stability checks before pose graph use.

## 3. Selected V21 Algorithm Stack

### 3.1 Default `v21_infer` Stack

```text
input video + optional side inputs
  -> raw frame manifest
  -> depth modality report
  -> Depth Pro monocular baseline on keyframes/full timeline
  -> UniDepth monocular candidate on same frames
  -> native depth/RGB-D/stereo/multiview candidates when present
  -> monocular-vs-assisted comparison and tuning
  -> agent-selected OWLv2 detector keyframes
  -> OWLv2 keyframe bbox proposals
  -> approved OWLv2 bbox prompts and rejected keyframes
  -> SAM2 proper full-video mask propagation from approved boxes
  -> segmentation_sam2_proper contamination review
  -> RTMLib + WiLoR + HaMeR hand candidates
  -> metric MANO diagnosis and active shape/pose/scale optimization
  -> visible surface from accepted masks + selected depth/camera
  -> object branch decision
  -> rigid branch: CAD/TRELLIS/SPAR3D/TripoSG/Hunyuan candidates
  -> mesh validation and per-frame pose fitting
  -> V19-style rigid pose graph + MANO/object/contact terms when supported
  -> V21 state adapter and full render
```

### 3.2 Default `v21_benchmark` Stack

Same as `v21_infer`, except:

- dataset loader provides RGB/depth/calibration/model library under prediction inputs;
- GT remains sealed until post-render evaluation;
- after each iteration, failure clusters select one atomic intervention;
- tuned parameters are sample-bound unless cross-sample validation later promotes them.

## 4. Implementation Plan By Module

### 4.1 Depth / Camera Module

#### New scripts

1. `scripts/build_v21_depth_modality_report.py`
   - Inputs: `input_manifest.json`, optional side inputs, dataset manifest.
   - Outputs: `measurements/camera_depth/depth_modality_report.json`.
   - Classifies: native depth, RGB-D, calibrated stereo, uncalibrated stereo, multiview, monocular-only.

2. `scripts/run_v21_monocular_depth_baselines.py`
   - Wraps existing Depth Pro and UniDepth runners.
   - Optional Metric3D runner when implemented.
   - Outputs:
     - `measurements/monocular_baselines/depthpro/depth_candidate.npz`
     - `measurements/monocular_baselines/unidepth/depth_candidate.npz`
     - `measurements/monocular_baselines/monocular_baseline_report.json`
   - Required tunable parameters:
     - backend list;
     - frame stride/keyframes/full timeline;
     - input resolution;
     - known focal/intrinsics override;
     - min valid pixels;
     - crop/mask vs full-frame mode.

3. `scripts/run_v21_stereo_depth_candidate.py`
   - Backends: RAFT-Stereo default, IGEV optional, CREStereo fallback.
   - Inputs: left/right frames, calibration if available.
   - Outputs: disparity, confidence, left-right residual, occlusion/disocclusion masks, metric depth only when `K`, baseline, and rectification are valid.
   - Required tunable parameters:
     - rectification source;
     - disparity range/search settings when backend exposes them;
     - iteration count (`valid_iters` for RAFT/IGEV);
     - correlation implementation / memory mode;
     - confidence thresholds;
     - LR consistency threshold;
     - resize scale.

4. `scripts/compare_v21_depth_against_monocular.py`
   - Compares native/RGB-D/stereo/multiview candidates against monocular baseline.
   - Residuals:
     - focal/intrinsics plausibility;
     - RGB-depth registration;
     - mask depth continuity;
     - hand-depth residual;
     - visible-surface residual;
     - temporal smoothness;
     - downstream pose-fit residual when available.
   - Outputs: `measurements/camera_depth/monocular_vs_assisted_depth_report.json`.

5. `scripts/select_v21_depth_camera_bundle.py`
   - Supersedes V20 selector for V21.
   - Enforces monocular comparison before selecting assisted primary depth.
   - Writes selected scope-specific depth state and tuning requirements.

#### Harness integration

- The agent predicts expected residual changes before each tuning attempt.
- If native depth/stereo is worse than Depth Pro/UniDepth baseline, tune registration/calibration/rectification/focal/scale before downweighting.
- Uncalibrated stereo can be retained only as relative evidence.

### 4.2 Segmentation Module

#### Inherited V19/master scripts

1. `scripts/run_v21_owlv2_bbox_proposals.py`
   - Inputs: sampled keyframes and text prompts from task/agent.
   - Outputs: sparse OWLv2 keyframe boxes, phrases, scores, thresholds, and rejected proposals.
   - Scope: interface-compatible replacement for deprecated GroundingDINO, not a new continuous bbox tracker or SAM-mask bbox refiner.

2. `scripts/approve_v21_owlv2_bbox_prompts.py`
   - Combines OWLv2 boxes, agent-selected keyframes, and target identity context.
   - Outputs selected target bbox prompts and rejected keyframes.

3. `scripts/run_v21_sam2_proper_segmentation.py`
   - Runs SAM2 from approved OWLv2 bbox prompts only.
   - Outputs the active V21 mask track, mask directory, QC, overlay, and summary.

4. Contamination review sheet
   - Checks target identity, missing visible pixels, boundary, background/table contamination, temporal drift, and depth-edge agreement.
   - Outputs accept/reject/large-deviation decision for each mask track.

#### Harness integration

- The active SAM2 proper track from approved OWLv2 bbox prompts is required before any segmentation enters geometry.
- Assisted masks, if added later, must be compared against the active SAM2 proper track and cannot replace it silently.
- Wrong object/table contamination is a systematic error, not a weak observation.
- Tuning records are written only when measured or reviewed large deviation is present; they must tune the existing OWLv2 bbox/SAM2 box-prompt interface rather than introduce a new bbox/segmentation algorithm.

### 4.3 Hand / MANO Module

#### New scripts or modifications

1. `scripts/diagnose_v21_metric_mano_inputs.py`
   - Inputs: RTMLib, WiLoR, HaMeR, HaWoR candidates, selected depth/camera, optional hand masks.
   - Diagnoses detector boxes, side mapping, crop/resize convention, camera convention, global transform, temporal offsets, and depth alignment.

2. `scripts/solve_v21_active_mano_shape_pose.py`
   - Extends `optimize_contact_aware_mano_graph_v8.py` or wraps it.
   - Variables:
     - track-level `betas_h`;
     - track-level base scale or per-track log scale;
     - per-frame global orient, hand pose, translation;
     - optional contact logits only when object mesh pose exists.
   - Losses:
     - 2D keypoint reprojection;
     - RTMLib keypoint reprojection;
     - hand silhouette distance;
     - visible hand depth;
     - bone/span prior;
     - MANO shape prior;
     - pose/orient/translation priors;
     - temporal smoothness;
     - contact compatibility;
     - nonpenetration.
   - Tunable parameters:
     - backend candidate weights;
     - keypoint confidence thresholds;
     - silhouette weight;
     - depth weight and depth valid thresholds;
     - shape prior sigma;
     - min/max hand span;
     - LR/iters;
     - temporal weights.

3. `scripts/write_v21_mano_tuning_record.py`
   - Records backend/box/side/camera/shape parameter attempts with sample binding.

#### Harness integration

- WiLoR and HaMeR are default candidate generators; RTMLib gives 2D constraints.
- HaWoR is allowed only after projection/camera/crop convention diagnosis.
- V20 `solve_v20_hand_shape_track.py` can initialize betas but cannot be accepted hand shape.
- Contact claims are disabled unless this module outputs metric MANO state with visible uncertainty.

### 4.4 Object Geometry And Pose Module

#### New scripts or modifications

1. `scripts/build_v21_geometry_condition_packet.py`
   - Inputs: accepted object masks, keyframes/crops, visible surfaces, depth/camera candidate ids, object plan, occlusion/contact notes.
   - Outputs condition packet with negative constraints.

2. `scripts/run_v21_geometry_candidates.py`
   - Orchestrates candidate generation:
     - public CAD fit when available and target-plan tied;
     - TRELLIS image-to-3D default;
     - SPAR3D fast mesh candidate;
     - TripoSG high-fidelity candidate;
     - Hunyuan3D high-quality candidate in benchmark/strong tuning.
   - Tunable parameters:
     - crop/mask keyframe selection;
     - background removal/crop padding;
     - generation seed;
     - inference steps;
     - CFG strength where available;
     - target face count/remesh options;
     - image vs multi-image conditioning.

3. `scripts/select_v21_geometry_candidate.py`
   - Builds on V20 registry/validator.
   - Required residuals:
     - visible surface alignment;
     - silhouette projection;
     - free-space violation;
     - depth residual;
     - scale plausibility;
     - temporal pose stability;
     - contact/nonpenetration compatibility when available.

4. `scripts/solve_v21_rigid_mesh_pose_graph.py`
   - Wraps `fit_v18_compact_rigid_object_pose.py`, `solve_v19_rigid_object_pose_graph.py`, and joint object/MANO/camera graph scripts.
   - Writes V21 state-compatible object pose posterior.

#### Harness integration

- Generated meshes are never final without validation and pose graph.
- Visible surface scatter is a measurement only.
- Benchmark iteration can switch candidate generator if validation rejects the current mesh.

### 4.5 Renderer / State Adapter

#### New scripts

1. `scripts/assemble_v21_state_from_optimized_branches.py`
   - Consumes selected depth/camera bundle, accepted masks, MANO posterior, object mesh pose graph, contact/occlusion/nonpenetration states.
   - Writes `state/v21_physical_state.json`, `state/annotations_v21_renderable.json`, and uncertainty.

2. `scripts/render_v21_physical_annotations.py`
   - May initially adapt V18 renderers but must output V21 paths.
   - Must distinguish:
     - metric MANO vs overlay-only hand;
     - monocular vs assisted depth support;
     - accepted mesh pose vs visible-surface measurement;
     - unresolved contact/occlusion/nonpenetration.

## 5. Benchmark Optimization Plan

### 5.1 Initial benchmark samples

Start with one DexYCB sample and one HO3D sample already represented by the V21 benchmark bootstrap. Use short smoke spans first for mechanism development, then full sample spans.

### 5.2 Iteration policy

Each iteration changes one mechanism only:

- depth/camera backend or internal parameter;
- segmentation keyframe/OWLv2 threshold/SAM2 box-prompt interface parameter only when that atom shows large deviation;
- MANO backend/crop/side/shape/weight parameter;
- geometry candidate backend or condition packet;
- pose graph residual weight only after observation quality has been tuned.

Each iteration records:

- failure cluster;
- hypothesis;
- parameter change;
- prediction;
- render/metric result;
- keep/reject decision;
- sample-bound parameter file.

### 5.3 Metrics

Use GT only after prediction render/state exists.

Primary benchmark metrics:

- hand joint camera error;
- hand mesh/vertex error when available;
- object translation/rotation error after coordinate/symmetry alignment;
- mask IoU/boundary F where GT masks exist;
- depth residual against native depth for prediction-side depth, not as oracle correction;
- visibility-stratified errors;
- contact/nonpenetration only when GT/geometry supports it.

## 6. Recommended Implementation Order

### Phase 1 — Depth/Camera physical blocker

Goal: produce V21 depth/camera candidate bundle with monocular baseline and comparison report on one benchmark sample.

Implement:

1. `build_v21_depth_modality_report.py`
2. `run_v21_monocular_depth_baselines.py`
3. `compare_v21_depth_against_monocular.py`
4. `select_v21_depth_camera_bundle.py`

Why first: segmentation, MANO refit, visible surface, object pose, and contact all depend on depth/camera semantics.

### Phase 2 — Segmentation physical blocker

Goal: accepted target mask track with contamination review and RGB-only vs assisted comparison when an assisted modality exists.

Implement or preserve:

1. Agent keyframe selector (`scripts/select_v21_agent_keyframes_from_plan.py`).
2. OWLv2 keyframe proposal runner (`scripts/run_v21_owlv2_bbox_proposals.py`); GroundingDINO remains disabled as a V21 default bbox source.
3. Approved bbox prompt writer (`scripts/approve_v21_owlv2_bbox_prompts.py`).
4. SAM2 proper bbox-prompt runner (`scripts/run_v21_sam2_proper_segmentation.py`).
5. Contamination review sheet over `sam2_proper_summary.json`.
6. Segmentation tuning records only for measured large-deviation cases.

### Phase 3 — Metric MANO blocker

Goal: metric MANO state with active betas/scale/pose optimization and review render.

Implement:

1. MANO input diagnosis.
2. Active shape/pose optimizer.
3. Hand metric/overlay renderer semantics.

### Phase 4 — Rigid object geometry/pose blocker

Goal: one target rigid object rendered as fitted/adapted mesh pose, not scatter.

Implement:

1. Geometry condition packet.
2. Mesh candidate orchestration.
3. Candidate selection.
4. Rigid mesh pose graph.

### Phase 5 — Integrated render and benchmark loop

Goal: full V21 render and at least one benchmark iteration with failure-cluster-driven improvement.

Implement:

1. V21 state assembler.
2. V21 renderer adapter.
3. Benchmark iteration controller.

## 7. Algorithm Defaults And Strong-Tuning Parameters

| Module | Default | Secondary | Strong tuning knobs |
|---|---|---|---|
| Monocular depth | Depth Pro + UniDepth | Metric3D | input resolution, focal/intrinsics prior, full-frame vs crop, valid-pixel threshold, temporal smoothing after alignment |
| Multiview camera/depth | MASt3R | DUSt3R, VGGT existing runner | frame/keyframe spacing, pair graph, global alignment iterations, scale anchors, confidence thresholds |
| Stereo | RAFT-Stereo | IGEV, CREStereo | rectification, baseline/K, valid_iters, resize, confidence/LR threshold, memory/correlation mode |
| Target detection | OWLv2 keyframe proposals plus approved target boxes | Missing-backend failure record | prompt text, score threshold, keyframes, phrase grouping, model path/download authorization only when large deviation is observed |
| Video segmentation | SAM2.1 propagation from approved OWLv2 bbox prompts | optional assisted candidate only after active bbox-prompt SAM2 exists | box prompts, seed frames, propagation direction, memory reset only when large deviation is observed |
| Hand candidates | WiLoR + HaMeR + RTMLib | HaWoR, OmniHands/EgoForce experimental | detector box, side mapping, crop scale, backend choice, keypoint confidence |
| MANO optimization | V21 active betas/scale/pose | V8 contact-aware graph extended | shape prior sigma, depth/silhouette/keypoint weights, span bounds, temporal weights, LR/iters |
| Mesh generation | CAD if public benchmark model; otherwise TRELLIS | SPAR3D, TripoSG, Hunyuan3D | keyframe crop, mask/background removal, seed, steps, CFG, face count/remesh, multi-image conditioning |
| Pose graph | V19 rigid pose graph | joint camera/object/MANO graph | visible residual weights, pose sigma, temporal sigma, nonpenetration/contact weights |

## 8. Residual Uncertainty

- FoundationStereo may be valuable, but source fetch failed; do not select it as default until repository/source access is verified.
- OmniHands/EgoForce may be useful for egocentric hand/arm cues; current source fetch failed, but repo runners exist. Treat as experimental branches until validated on representative samples.
- TRELLIS/Hunyuan/TripoSG/SPAR3D generate object priors; none guarantees instance-accurate geometry without visible evidence fitting.
- Monocular metric depth can fail on unusual egocentric close-hand scenes. This is why native/stereo/depth evidence must be compared against monocular, not replaced by it.
- Sample-bound tuning is required; benchmark-improved parameters are not global defaults until cross-sample validation.
