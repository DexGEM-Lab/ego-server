# Pipeline V16: Full Raw-Video Annotation Pipeline

## Purpose

V16 is the first full-raw-video packaging version after the component exploration in v2 through v15. It consumes one original EgoScale raw video and produces full-length render artifacts with no frame clipping:

- annotated video with MANO hand overlay and object mesh annotation;
- 3D world animation with head camera, MANO hands, object mesh, object motion, and contact state;
- side-by-side annotated video plus 3D reconstruction with semantic captions.

The output videos must have the same source-frame count, duration, and timeline as the raw input video. This packaging rule did not satisfy the original V3 joint graph requirement: V16 did not jointly solve camera trajectory, MANO articulation, object geometry/topology, object pose, depth, contact labels, and physical contact consistency.

## Inputs

Required input for one run:

```text
raw_video.mp4
action_json
output_root
```

Optional inputs:

```text
calibration or recovered intrinsics
MANO model files
server profile for heavy perception jobs
```

The pipeline may use GPU servers for heavy stages, but server execution is an implementation detail. The v16 source-of-truth output is the full-timeline annotation package under one output root.

## Full-Timeline State

For each source frame `t`, v16 stores:

```text
T_wc_t        head camera pose in a clip world coordinate system
K_t           camera intrinsics or recovered intrinsics
H_t           MANO hand state for every visible or predicted hand
J_t           hand keypoints in image, camera, and world coordinates
O_t           manipulated object mesh state
M_t           object mask and visibility support
C_t           contact candidates and contact state
A_t           semantic action caption
Q_t           per-frame QC flags and confidence
```

Object state `O_t` is mesh-backed. A centroid, primitive, bounding box, or 2D patch can appear as a diagnostic field, but it cannot satisfy object pose annotation.

## Perception Sources

V16 uses model-produced perception outputs and routes them through one uniform reconstruction path:

- head pose from DROID-SLAM, VGGT, or a fused pose track;
- hand pose from WiLoR as the selected MANO backbone, with RTMLib 2D keypoints, hand masks, metric depth, and object-contact geometry as independent measurement sources;
- object plan from VLM action/video review;
- object detection and segmentation from open-vocabulary detector plus SAM/SAM2/SAMWISE/SAM3-family video segmentation when available;
- metric depth from UniDepth, Depth Anything metric, VGGT, or another depth source with source labels;
- sparse object tracks from CoTracker or an equivalent learned point tracker;
- generated or completed object mesh priors from video-conditioned or multi-view 3D models only after replay acceptance.

The geometry, filtering, contact reasoning, and rendering stages consume masks, tracks, depths, meshes, poses, confidences, and captions through a category-agnostic schema.

## Hand Model Decision

V16 uses WiLoR as the delivered MANO backbone. The v1-v15 evidence supports this choice:

- WiLoR already produced the full-video MANO contract needed by v16: handedness, joints, vertices, pose/shape parameters, and projection provenance.
- RTMLib, hand masks, metric depth, and object SDF residuals exposed WiLoR failures under occlusion and contact, so v16 can correct or reject WiLoR states while preserving one hand model contract.
- HaWoR produced plausible camera-local hand scale in probes, but the tested world alignment created severe reprojection and scale failures. HaWoR remains a research comparison until its coordinate contract is solved.
- HaMeR helped the selected-right noncontact mop check, but prior tests did not establish a stronger full-video MANO contract than WiLoR.
- HandDGP produced useful diagnostic geometry, but it did not satisfy the MANO pose-parameter contract required for v16 deliverables.

The current evidence selects WiLoR for V16. A later version can use an LLM/agent to choose among hand backbones at runtime when the version design defines:

- the allowed hand-model contracts;
- the evidence each model must emit on every source frame;
- the residuals used for comparison against image, depth, mask, temporal, and object-contact evidence;
- the optimization objective that decides whether to switch, refit, or reject;
- the audit trace that records the agent's observation and decision.

V16's runtime agent judges WiLoR state quality, refit actions, occlusion prediction, and failure escalation inside one MANO contract. Hand-backbone selection stays a research item until comparative full-video evidence exists.

## Pipeline Stages

### Stage 0: Raw Video Inventory

Read the raw video frame count, fps, resolution, action JSON, and available metadata. The output manifest records the required full-frame count.

Acceptance for this stage: output manifest frame count equals raw video frame count.

### Stage 1: Head Camera Pose

Run full-frame camera localization over the complete raw video. The first v16 implementation should use the strongest locally available pose source from previous work, with full-frame DROID/VGGT comparison where available.

Outputs:

```text
camera_trajectory.npz
camera_qc.json
```

Acceptance:

- one pose per source frame;
- no missing frame in the delivered timeline;
- pose scale source recorded;
- large jumps flagged and smoothed only through an explicit prediction/update model.

### Stage 2: Full-Timeline Hand Evidence

Run WiLoR over every source frame as the hand reconstructor. Fit and filter one full-timeline MANO stream using:

- WiLoR MANO pose, shape, joints, vertices, handedness, and projection metadata;
- RTMLib 2D keypoints;
- hand masks;
- metric depth;
- temporal velocity and acceleration priors;
- object SDF contact and nonpenetration residuals;
- prediction/update smoothing for occlusion and lost frames.

Outputs:

```text
hands_full_timeline.json
hand_qc.json
```

Acceptance:

- every source frame has a WiLoR-measured or explicitly predicted MANO hand state;
- predicted states preserve uncertainty and reason codes;
- RTMLib, mask, depth, temporal, and contact residuals are stored for every active hand state;
- physically unsupported WiLoR states are corrected when the residual graph has enough evidence and rejected when it does not;
- hand overlays remain visually coherent on full-video inspection samples;
- 2D keypoints and masks appear only as measurement evidence and QC signals, never as the delivered hand annotation.

### Stage 3: Full-Timeline Object Plan And Segmentation

Use a VLM to identify manipulated object tracks over the whole raw video and produce prompts, active spans, and physical notes. Run open-vocabulary detection and video segmentation over the full timeline for the selected manipulated object tracks.

Outputs:

```text
object_plan.json
object_masks_full_timeline/
object_mask_qc.json
```

Acceptance:

- one explicit object state for every source frame: measured, predicted, outside active manipulation, occluded, or rejected;
- no hand-written visual category branching;
- identity drift is rejected by model/video review and temporal mask evidence;
- full-frame coverage is reported against the raw frame count.

### Stage 4: Object Mesh Reconstruction

Build a full-timeline object mesh stream. The first v16 implementation combines:

- observed visible surface from mask plus metric depth;
- temporal fusion of hidden surfaces using learned tracks and motion factors;
- accepted generated or video-conditioned mesh priors only where visible replay and temporal QC support them.

Outputs:

```text
object_meshes_full_timeline.npz
object_mesh_qc.json
```

Acceptance:

- every frame has a mesh state or an explicit inactive/occluded state;
- active manipulated-object frames must be mesh-backed;
- visible replay against mask/depth passes on the full active timeline;
- hidden/completed geometry passes temporal support checks before entering the delivered mesh;
- mesh state remains tied to model-produced object identity and depth evidence.

### Stage 5: Full-Timeline Factor Graph

Optimize a full-timeline graph with frame nodes and auxiliary contact nodes.

Nodes:

```text
T_wc_t        camera pose
H_t           MANO hand state
O_t           object mesh pose/deformation state
S_k           temporally supported object surface elements
C_th          hand-object contact variables
G_th          contact gap vectors
```

Edges:

```text
camera temporal motion
hand reprojection and MANO prior
hand mask and depth support
object mask and depth replay
CoTracker object correspondence
object temporal motion and hidden-surface support
hand-object nonpenetration
contact equality when contact is supported
contact switch and sliding dynamics when evidence exists
caption/action consistency over active intervals
```

Objective:

```text
min_X
  image_replay_residuals
+ hand_reprojection_residuals
+ hand_depth_mask_residuals
+ object_depth_mask_residuals
+ object_track_residuals
+ temporal_motion_residuals
+ nonpenetration_residuals
+ supported_contact_residuals
+ contact_dynamics_residuals
```

All residuals are robust and confidence-weighted. Prediction fills missing frames only through explicit process-model propagation with uncertainty.

Acceptance:

- full source timeline remains represented;
- contact claims require geometry, temporal, and dynamics support;
- noncontact frames use nonpenetration and separation evidence;
- solver outputs preserve visible replay and hand overlay quality.

### Stage 6: Full-Length Rendering

Render three full-length videos:

```text
overlay_mano_object.mp4
world_reconstruction_3d.mp4
side_by_side.mp4
```

Each output must match raw video frame count and duration. The world view must show head camera, head trajectory, MANO hands, object mesh, object motion/contact state, scale, axes, and caption in stakeholder-facing presentation form.

### Stage 7: QC Package

Write one manifest:

```text
v16_full_pipeline_manifest.json
```

Required manifest fields:

```text
raw_video
raw_frame_count
raw_fps
output_frame_count
output_fps
frame_count_match
overlay_video
world_video
side_by_side_video
camera_qc
hand_qc
object_mask_qc
object_mesh_qc
timeline_residual_qc
render_qc
visual_inspection_sheet
failure_rows
```

Acceptance:

- `frame_count_match` must be true;
- each required video exists and has the raw frame count;
- QC flags list unresolved weak spots explicitly;
- visual inspection covers the whole clip through sampled sheets and targeted failure segments.

## First Representative Runs

V16 should start with one full raw video, then add broader representative samples:

```text
/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4
/data2/egoscale_demo_30h/egoscale_tasks/20251224_1141_Rec3a3b_P0_S3a8b63_task_4/20251224_1141_Rec3a3b_P0_S3a8b63_task_4.mp4
/data2/egoscale_demo_30h/egoscale_tasks/20251210_0002_Rec4afc_P0_S296a7f_task_4/20251210_0002_Rec4afc_P0_S296a7f_task_4.mp4
```

The first clip reuses the most prior evidence from the trash-bag/lid work. The second stresses clutter and thin objects. The third stresses long tool geometry and large camera motion.

## Implementation Contract

V16 implementation should create one orchestrator:

```text
scripts/run_v16_full_pipeline.py
```

Required CLI:

```text
--clip
--actions-json
--output-dir
--server-profile
--resume
--dry-run
```

The orchestrator owns:

- full-video stage scheduling;
- server job launching through tmux for heavy stages;
- artifact path provenance;
- frame-count checks after every stage;
- final render and manifest creation;
- fail-fast behavior for missing required full-timeline artifacts.

## Closure Rule

V16 closure meant full-duration packaging closure only. It required at least one original raw video to have a complete full-length overlay video, full-length world reconstruction, full-length side-by-side render, full-timeline annotation JSON, and passing manifest frame-count checks.

All later versions inherit the full-duration packaging rule, but full-duration packaging is not sufficient for annotation closure. A later version must also satisfy its own scientific solver and quality predicates before it can count as a complete annotation pipeline.
