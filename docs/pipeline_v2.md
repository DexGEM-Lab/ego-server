# Pipeline V2: VLM-Planned Object Mesh Reconstruction

## Logic

V1 established full-frame plumbing with DROID camera poses, WiLoR MANO hands, captions, object masks, and side-by-side rendering. Its object stage still encoded visual variation in hand-written category logic and rendered object geometry as a proxy. That failed the central requirement: manipulated objects must be reconstructed as geometry, and visual object selection must be model-produced.

V2 replaces that object path:

1. A VLM reads sampled video frames and action metadata, then writes an object plan with track IDs, descriptions, open-vocabulary prompts, active intervals, and physical notes.
2. OWLv2 proposes boxes from the VLM prompts; SAM produces masks from those boxes.
3. Visual QC and optional VLM mask verification reject masks that cover a different planned object or background.
4. Depth Anything V2 metric indoor estimates dense per-frame depth for verified mask frames.
5. The mesh stage back-projects verified mask pixels through the camera intrinsics and head-camera pose to build a dynamic observed-surface mesh in world coordinates. Each mesh archive now uses one declared depth source so DROID-relative depth and monocular metric depth are not mixed inside the same object track.
6. Contact-depth correction is an explicit ablation with reported shift values. The default V2 mesh archive leaves monocular depth unchanged, then reports the resulting hand-mesh distances.
7. The renderer draws MANO hands, head camera frustum, trajectory, and the object mesh in the world reconstruction panel.

V2 scope: reconstruct the observed surface for a manipulated object and show contact-frame object geometry. V3 scope starts at complete watertight geometry and a single object-centric mesh state across the whole clip.

## V2 Milestone Closure Criteria

The V2 observed-surface milestone is closed against this plan when it has:

- a VLM-produced object plan for a manipulated object track in a representative clip;
- accepted masks for the delivered interval, with visual QC rejecting identity drift;
- one declared depth source for the mesh archive;
- per-frame observed-surface object meshes in world coordinates for the delivered interval;
- overlay, 3D world reconstruction, and side-by-side videos with MANO hands, head camera, object mesh, and semantic caption;
- QC that reports hand-object distance residuals and inspected visual failures.

## Implemented Components

- `scripts/build_object_plan_vlm.py`: calls the OpenAI Responses API with sampled frames and action metadata, returning a structured object plan.
- `scripts/segment_object_plan_v2.py`: runs plan-driven OWLv2 plus SAM and writes full-timeline annotations with object masks.
- `scripts/verify_plan_masks_vlm.py`: verifies proposed masks against the target object description using a VLM review sheet.
- `scripts/estimate_metric_depth_v2.py`: runs Depth Anything V2 metric indoor on measured object-mask frames and stores dense metric depth maps.
- `scripts/reconstruct_object_mesh_v2.py`: builds a per-frame dynamic mesh from masks, one selected depth source, and head-camera pose; optional contact-depth correction is disabled by default and reported when enabled.
- `scripts/fuse_v1_full_fidelity.py`: renders `--object-mesh-npz` archives in the 3D world panel.

## Representative Trash Clip

Clip:

`/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4`

VLM plan:

`/data2/ego_annotation_outputs/representative_trash/v2_object_plan/object_plan_vlm.json`

The VLM identified four tracks:

- `black_trash_bag`
- `white_trash_bag`
- `off_white_trash_can_first`
- `pink_lid_trash_can_second`

### Failed White-Bag Track

The white-bag segmentation run processed 471 planned frames and wrote masks for all 471 frames:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_white_bag_masks/`

Visual QC rejected this result. The masks selected the pink lid in frames 880 to 918 and a wall/door-like surface around frame 534. The detector score and detection rate were live signals for prompt match; the review sheet showed identity drift. This mask set is quarantined, and mesh reconstruction consumes accepted object masks only.

Review sheet:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_white_bag_masks/review_sheets/white_bag_plan_masks_000_918.jpg`

### White-Liner Track Status

The white liner is the harder manipulated deformable object in the second half of this clip. Its V2 OWLv2-box mask run failed visual QC, so V2 quarantines that track and delivers the accepted pink-lid observed-surface mesh. Follow-on white-liner recovery attempts belong in `docs/pipeline_v3.md`.

### Accepted Pink-Lid Track

The pink-lid/trash-can track is visibly manipulated and the mask sheet matches the target object.

Mask output:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/`

Mask QC:

- planned frames processed: 372
- detected frames: 372
- target object: `pink_lid_trash_can_second`
- active frame span: 678 to 1049

Review sheet:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/review_sheets/pink_lid_plan_masks_678_1049.jpg`

The accepted sheet shows a consistent mask on the pink lid/trash-can assembly from approach through close-up handling.

## Mesh Reconstruction

DROID-only mesh reconstruction produced a real observed-surface mesh, but it missed the most important contact interval because DROID keyframes jumped from frame 818 to frame 980. That left frames 840 to 930 without nearby DROID depth.

DROID-only mesh QC:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh/qc_object_mesh_v2.json`

- mesh frames: 186
- valid vertices: 74,638
- valid triangles: 126,956
- missing near-DROID-keyframe frames: 153
- hand-mesh distance median: 0.284 m

Depth Anything V2 metric indoor filled that observability gap:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_metric_depth/qc_metric_depth_v2.json`

- selected mask frames: 372
- dense depth resolution: 960 by 540
- depth median: 1.109 m
- depth p05/p95: 0.372 m / 2.234 m

An adversarial review exposed that the earlier metric-depth archive mixed 186 DROID-depth frames with 153 metric-depth frames. The corrected archive uses only `metric_depth`.

Strict metric-depth mesh QC:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict/qc_object_mesh_v2.json`

- depth source: `metric_depth`
- mesh frames: 372
- valid vertices: 550,893
- valid triangles: 1,047,857
- frames outside the object interval: 678
- hand-mesh distance frame count: 281
- hand-mesh distance median over frames with hands: 0.202 m
- hand-mesh distance p05/p95 over frames with hands: 0.0049 m / 0.572 m
- worst hand-mesh distance: 0.832 m at frame 817

Bounded contact-depth ablation:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_contact03/qc_object_mesh_v2.json`

- contact-depth correction cap: 0.03 m
- corrected frames: 159
- median reported shift: 0.03 m
- hand-mesh distance median over frames with hands: 0.201 m
- hand-mesh distance p95 over frames with hands: 0.566 m

The ablation saturated the 30 mm cap on most corrected frames and barely changed the distribution. That falsifies a simple global contact-depth shift as the scale/contact fix.

For the contact window 840 to 930:

- mesh frames: 91
- hand-mesh distance median: 0.092 m
- hand-mesh distance p05/p95: 0.0015 m / 0.544 m
- worst distance in this window: 0.597 m at frame 875
- frames 848 and 885 have near-contact distances below 10 mm, while frames 840 to 847 and 875 show large hand-object depth disagreement

V3 rechecked this contact window by merging the V2 pink-lid masks with the same WiLoR hand stream used in the earlier full annotation:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_masks_wilor_hands_merged.json`

Corrected contact reliability against the V2 pink-lid masks:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_wilor_contact_reliability_840_930.json`

- rows: 182
- measured high-score rows: 124
- reliable contact rows: 0
- measured high-score median joint reprojection: 10.9 px
- measured high-score median MANO-minus-metric-depth residual: 173 mm
- measured high-score median hand-lid contact gap: 269 mm
- measured high-score contact-ok rows: 1

This diagnostic uses the V2 pink-lid mask source, not the older `remote_supported_area_full_strict2` white-bag track. It confirms that the pink-lid observed surface is real, while the hand/depth/contact state remains physically inconsistent.

## Current V2 Milestone Delivery

Contact-window side-by-side render:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/side_by_side.mp4`

Frame count and size:

- side-by-side: 91 frames, 30 fps, 1920 by 540
- overlay: 91 frames, 30 fps, 960 by 540
- 3D reconstruction: 91 frames, 30 fps, 960 by 540

Inspected stills:

`/data2/ego_annotation_outputs/representative_trash/v2_pink_lid_mesh_metric_strict_render_840_930/review_stills/`

Visual inspection:

- frame 840: the 2D mask is correct, but the metric-depth surface is visibly separated from the hands in the world panel;
- frame 858: the object mask and observed surface are plausible, while the left hand remains separated in 3D;
- frame 875: the 2D mask covers the lid, but the observed surface is far from both hands;
- frame 880: the surface stays present and detailed, with remaining depth mismatch against the MANO hands.

This closes the V2 observed-surface mesh milestone for the accepted pink-lid contact-window slice. It does not close the overall annotation pipeline.

## Evidence Status

The current v2 result supports these mechanisms on one representative non-kitchen clip:

- VLM object planning can identify manipulated object tracks.
- Open-vocabulary detection plus SAM can produce correct object masks when the target is visually unambiguous.
- The same path can fail when the prompt is ambiguous, as shown by the rejected white-bag masks.
- Dense metric monocular depth supplies full mask-interval coverage when DROID keyframes skip the contact interval.
- Dynamic observed-surface mesh reconstruction gives a real object mesh in the world panel.
- The corrected strict run exposes the open scale/contact problem instead of hiding it: the hand-mesh distance distribution remains far above the 5 mm target.

Evidence required by later pipeline versions for the overall task:

- complete mesh reconstruction for the full object, including the unseen backside;
- a single temporally consistent object-centric mesh identity;
- deformable white-bag reconstruction;
- external scale and ground-truth-style validation before any absolute 5 mm claim;
- joint optimization of depth scale, hand pose, camera pose, object pose, and contact state;
- physical force consistency with explicit force, mass, inertia, and object acceleration estimates.

## After V2

`docs/pipeline_v3.md` owns the complete-mesh, persistent object-state, referring-segmentation, joint scale/depth/contact, and force-consistency work. Those items are required by the overall annotation constitution, while the V2 milestone closes at observed-surface mesh reconstruction plus an inspected contact-window render.
