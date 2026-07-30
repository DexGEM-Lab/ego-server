# Pipeline V8: Contact-Aware MANO and Mesh-Prior Graph

## Starting Point

V7 now has three accepted representative samples with delivered videos:

- wild rice frames 2538 to 2540: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/wild_rice/video_mesh_v6_repaired_measured_2538_2540`
- trash frames 865 to 870: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/trash/video_mesh_v3_solidified_unidepth_vggt_865_870`
- residential cardboard box with blue books frames 616 to 618: `/data2/ego_annotation_outputs/representative_box_books/v7_box_books_similarity_refit_box_with_books_616_618`

The accepted V7 path is still mostly a measured visible-geometry pipeline. It succeeds when SAM/VLM object evidence, metric depth, VGGT camera geometry, MANO hands, CoTracker temporal factors, and contact SDF agree on a short window. It does not yet solve the broader failure that dominated the candidate search: MANO topology often collapses or slides onto palm/wrist when fingers are occluded by a manipulated object, even when object masks, hand masks, and WiLoR/HaMeR candidates are visually plausible.

V8 therefore targets the hand-object alignment mechanism directly. The goal is not another candidate search loop. The goal is a graph where hand topology, object mesh, contact state, and temporal motion constrain each other before selection and delivery checks.

## Source Refresh

The current open-source and paper frontier supports this direction:

- HOLD jointly reconstructs articulated hands and novel objects from monocular interaction videos without a pre-scanned object template. Repository: https://github.com/zc-alexfan/hold
- SAM 3D Objects reconstructs full object geometry, texture, pose, and layout from masked objects in natural images, including occlusion and clutter cases. Repository: https://github.com/facebookresearch/sam-3d-objects
- VGGT predicts camera parameters, depth maps, point maps, and 3D point tracks from one or more views; V7 already uses VGGT as a camera/geometry source. Repository: https://github.com/facebookresearch/vggt
- MoGe and MoGe-2 provide open-domain monocular geometry and metric point maps that can serve as an independent metric-depth source. Repository: https://github.com/microsoft/MoGe
- GHOST is a 2026 category-agnostic hand-object interaction reconstruction method using Gaussian splatting, geometric-prior retrieval, and grasp-aware alignment. Paper: https://arxiv.org/abs/2603.18912
- ArtHOI is a 2026 monocular 4D hand-articulated-object reconstruction system that integrates foundation priors, CoTracker3, metric-scale object alignment, and MLLM-guided contact reasoning. Project: https://arthoi-reconstruction.github.io/

The shared lesson is precise: foundation models can provide masks, meshes, contact labels, hand estimates, and tracks, but the final state still needs metric alignment and physical hand-object consistency. V8 should treat these systems as evidence generators and priors, not as final annotations.

## Representation

For each frame `t`, V8 keeps the V7 fixed observations and adds optimizable hand-object state:

```text
T_wc_t       head camera pose, initialized from VGGT and fixed in V8.0
H_t          MANO hand state: global orientation, articulated pose, translation, scale
O_t          observed object mesh from mask + metric depth, fixed as visible evidence
P            optional complete object prior mesh or Gaussian surface from SAM3D/HOLD/GHOST/etc.
A_t          object pose or Sim3 transform mapping P into world frame t
C_th         contact logit for hand patch h at frame t
Q_th         hand patch surface samples for anatomical/contact patch h
```

The state is category-agnostic. A tomato, box, bag, mop, rice stem, cup, or phone enters through the same data fields: masks, depths, tracks, meshes, contact evidence, and captions.

VLM or MLLM contact reasoning enters as data, not control flow. A model may report that the right index/middle fingertips press a blue book stack; the graph consumes that as a contact prior over MANO patch vertices. The downstream solver has no object-family branches.

## Factor Graph

The V8 graph has these measurement and prior factors.

Hand factors:

- 2D anatomical keypoint factors from WiLoR, RTMLib, HaMeR, HandDGP, or future hand models;
- hand-mask silhouette factor using SAM/VLM hand mask distance fields;
- hand metric-depth factor using UniDepth, VGGT depth, and optional MoGe/MoGe-2 point maps;
- MANO pose, shape, bone scale, and joint-limit priors;
- temporal hand motion factor on wrist translation/orientation and articulated pose velocity.

Object factors:

- object mask and z-buffer depth replay from the V7 observed mesh contract;
- CoTracker sparse surface factors, already validated in V6/V7;
- optional complete-prior visible-surface alignment factor for SAM3D/HOLD/GHOST/ArtHOI-style generated meshes;
- object pose smoothness or low-dimensional deformation factors where sparse tracks support them.

Contact factors:

- nonpenetration hinge for all non-contact hand vertices against the object SDF;
- contact equality factor only for hand patches with visual, geometric, and temporal support;
- no-slip factor for sustained contact patches, requiring the contact patch motion to agree with object surface motion where CoTracker/object factors make that surface observable;
- contact-logit prior from mesh-surface diagnostics, VLM/MLLM contact labels, and temporal support;
- contact sparsity prior so the graph cannot attract every nearby hand vertex to the object.

Dynamics factors:

- object acceleration consistency is optional and evidence-dependent. Without mass, friction, and external support estimates, V8 should use only weak directional consistency: when a contact patch is active and object acceleration is observable, the contact patch displacement should not contradict the measured object motion. Exact force magnitude is not observable from these videos.

## Objective

V8 minimizes a robust objective over `H_t`, `A_t`, and contact logits:

```text
min_X
  sum_t rho_keypoint(project(H_t) - hand_keypoints_t)
+ sum_t rho_mask(distance_to_hand_mask(project(H_t)))
+ sum_t rho_depth(depth(H_t) - metric_depth_t)
+ sum_t rho_pose(MANO_pose_prior(H_t))
+ sum_t rho_hand_motion(H_t - predict(H_{t-1}, H_{t-2}))
+ sum_t rho_object_replay(render(O_t or A_t(P)) - mask_depth_t)
+ sum_edges rho_track(A_t(surface_k) - lifted_track_tk)
+ sum_t,h (1 - C_th) rho_nonpen(max(0, -SDF_object(Q_th)))
+ sum_t,h C_th rho_contact(abs(SDF_object(Q_th)))
+ sum_t,h rho_contact_prior(logit(C_th) - evidence_logit_t,h)
+ sum_sustained rho_no_slip(contact_patch_motion - object_surface_motion)
```

The contact equality term is gated by `C_th`; non-contact vertices stay under the nonpenetration term. This matters because V7 showed both failure modes: missing real contact leaves hand/object alignment unconstrained, while unconditional attraction would falsely drag visible fingers into an object surface.

## Solver

V8.0 should use a staged PyTorch/SciPy solver on short windows:

1. freeze `T_wc_t` and the observed mesh archive `O_t`;
2. initialize `H_t` from the best V7 accepted or diagnostic MANO source;
3. build contact evidence from V7 mesh-surface diagnostics, VLM/MLLM contact labels when available, and temporal support;
4. optimize MANO translation, global orientation, limited pose deltas, scale, and contact logits;
5. optionally optimize complete-prior object pose `A_t` only after the observed mesh and hand fit pass local checks;
6. replay the solved annotations through the unchanged V7 replay, track-surface, physics, and render checks.

The first implementation should extend the existing hand-contact optimizer rather than start from a neural-field baseline. HOLD/GHOST/ArtHOI remain important baselines and possible measurement sources, but V8.0 can already test the causal question with existing annotations: can contact-aware MANO optimization repair previously rejected windows without degrading accepted V7 samples?

## First Artifact

Implement `scripts/optimize_contact_aware_mano_graph_v8.py`.

Inputs:

- V7 annotations with MANO candidates;
- object mesh archive;
- object mask/depth manifest;
- metric depth archive;
- mesh-surface contact report;
- optional RTMLib/WiLoR/HandDGP keypoint evidence;
- optional VLM/MLLM contact-label JSON.

Outputs:

- optimized annotations JSON;
- QC report with residual summaries for keypoints, hand mask, hand depth, temporal motion, SDF contact, nonpenetration, and contact logits;
- before/after review video for the hand/object window.

First target windows:

- negative-control accepted windows: wild rice 2538 to 2540, trash 865 to 870, box-books 616 to 618. V8 must not degrade their accepted replay/physics/render evidence.
- repair target: box-books 612 to 618. V7 rejects this full window because early selected contact is 18.6 mm from the repaired object surface while the physical contact subsegment 616 to 618 passes at 0.76 mm. V8 can test whether contact-aware MANO refinement repairs the early contact frames without inventing false contact.
- hand-topology target: detergent 672 to 678 or remote-control 540 to 546. These had good object/hand masks but rejected MANO topology. V8 succeeds only if it produces visually credible fingers and passes the downstream physical checks, not merely if scalar residuals improve.

## Acceptance

V8.0 is accepted only if all of these are true:

- no regression on all three V7 accepted samples under replay, track-surface, physics, and visual render inspection;
- at least one previously rejected contact/topology window is converted into an accepted delivered sample by the new hand-object graph;
- the improvement is visible in before/after MANO overlays, not only in scalar metrics;
- generated complete object priors, if used, pass the same visible replay, temporal track, and physics checks before appearing in stakeholder videos;
- the pipeline still has no object-category if/else logic.

## Fundamental Limit

A monocular egocentric RGB video cannot guarantee 5 mm absolute 3D accuracy for all hidden surfaces, hidden fingers, and unseen object backsides. V8 can improve correctness only where it adds independent evidence: multi-frame tracks, metric depth, hand keypoints, contact priors, object mesh priors, and physical consistency. The acceptance claim should therefore stay evidence-local: the delivered window is annotation-ready when the observable geometry, temporal tracks, MANO evidence, and contact physics agree under the stated thresholds.
