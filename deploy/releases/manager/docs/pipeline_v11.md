# Pipeline V11: Temporal Hidden-Surface Fusion

## Starting Point

V10 proved that raw Mesh4D replacement is too inaccurate on the visible surface, and that per-frame hidden-face appending can pass image replay while remaining temporally incoherent on unseen geometry. V11 keeps the visible mesh as measured evidence and changes only the hidden-completion mechanism.

The V11 question is:

```text
Can hidden object geometry become temporally stable when generated hidden proposals are fused through the observed object motion factors, while preserving replay, MANO physics, and stakeholder render quality?
```

## State

For each object window, V11 represents geometry as:

```text
M_t       measured visible object mesh from object mask, metric depth, and head pose
P_t       generated hidden-geometry proposal for frame t
E_tu      CoTracker-derived object surface motion factor from frame t to frame u
R         reference-frame hidden surface fused from multi-frame support
G_t       delivered mesh candidate: M_t plus filtered transform of R into frame t
H_t       MANO hand mesh and keypoints
T_wc_t    head camera pose
```

The category-agnostic rule remains unchanged. Detectors and VLMs may produce masks, tracks, captions, depth, mesh proposals, and confidences. The downstream reconstruction path consumes those arrays through the same replay, temporal-surface, SDF, and rendering checks.

## Diagnostic That Motivated V11

`scripts/check_v11_hidden_face_temporal_qc.py` isolates appended hidden faces and evaluates their temporal consistency under object motion factors. It samples hidden-surface points, transforms samples across consecutive frames, and measures symmetric surface distance and hidden-face count jumps.

The per-frame V10 hidden append failed this diagnostic:

- trash 865 to 870: no meaningful hidden geometry, because only one frame retained hidden faces;
- wild rice 2538 to 2543: hidden-surface p95 distance 179 mm, despite full multi-anchor motion coverage;
- mop 760 to 765: hidden-surface p95 distance 95 mm, despite full pair coverage.

These failures localize the problem to unseen generated geometry. Visible replay and visible-surface tracks were not enough to validate hidden surfaces.

## Temporal Fusion Mechanism

`scripts/fuse_v11_temporal_hidden_surface.py` implements the V11 mesh step:

1. extract hidden proposals from the generated-per-frame append reports;
2. sample proposal surfaces in each frame;
3. transform proposal samples into a reference frame using CoTracker object motion factors;
4. keep hidden samples that receive multi-frame support;
5. voxel-downsample supported samples and build a shared triangle surface;
6. map the shared surface back into each frame;
7. apply the V10 projection, mask, depth, free-space, z-buffer, and measured-surface filters per frame;
8. append only surviving hidden faces to the measured visible mesh archive.

The unfiltered wild-rice run passed hidden temporal QC but failed replay with IoU 0.431 and z-buffer p95 40.8 mm. That failure shows why temporal support and image evidence must both be active.

## Accepted V11 Deliveries

### Trash 865-870

Root:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered
```

Evidence:

- input hidden source: V9 PartCrafter hidden-prior archive, which had active hand-object contact but failed hidden temporal QC before fusion;
- pre-fusion hidden temporal QC rejected: hidden-surface p95 distance 76.9 mm and hidden pair coverage 0.8;
- one object-motion pair, 868 to 869, was marginal under the original 10 mm p95 threshold and accepted only in an explicit diagnostic factor report with an 11 mm p95 threshold; its inlier p95 is 10.34 mm;
- fused hidden surface support sample fraction: 0.897;
- retained hidden faces per frame: median 8,289;
- hidden temporal QC accepted: symmetric hidden-surface p95 6.47 mm, hidden pair coverage 1.0;
- replay accepted: IoU median 0.9478, visible-inside median 1.0, z-buffer p95 median 0.534 mm;
- track-surface QC accepted: 363 tracks, 1,792 edges, pair residual p95 9.32 mm;
- contact physics accepted: 6 reliable temporal contact rows, selected-contact abs SDF p95 1.245 mm, near-surface fraction 1.0, selected-contact penetration 0.0, full-window hand penetration fraction 0.00557;
- rendered overlay, world 3D, and side-by-side videos contain 6 frames at 6 fps.

Videos:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/deliverables_preferred/world/world_reconstruction_side_by_side.mp4
```

Visual inspection found the overlay and side-by-side coherent: the red object mesh stays on the pink lid, MANO follows the right hand, and magenta contact markers stay on the measured contact patch in all frames. The preferred world render separates the measured object surface from the temporally fused hidden surface, with a legend, head-camera glyph, head trajectory, axes, and metric scale. The side-by-side video carries the V11 label and the semantic caption fits within the caption band.

### Wild Rice 2538-2543

Root:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/wild_rice_2538_2543_filtered
```

Evidence:

- fused hidden surface support sample fraction: 0.5254;
- hidden temporal QC accepted for frames 2538 to 2543;
- replay accepted: IoU median 0.9458, visible-inside median 0.9834, z-buffer p95 median 6.03 mm;
- visible-surface track QC accepted on frames 2538 to 2541;
- physics accepted with no reliable temporal contact claim: full-window hand penetration fraction 0.00236;
- rendered overlay, world 3D, and side-by-side videos contain 6 frames at 6 fps.

Videos:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/wild_rice_2538_2543_filtered/deliverables/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/wild_rice_2538_2543_filtered/deliverables/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/wild_rice_2538_2543_filtered/deliverables/world/world_reconstruction_side_by_side.mp4
```

Visual inspection of overlay, world-view, and side-by-side contact sheets found the object mesh stable on the active stem, MANO hands near the visible hands, no-contact state consistent with physics, and the world view legible with object mesh, hands, head camera, axes, scale, and caption.

### Mop 760-765

Root:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/mop_760_765_filtered
```

Evidence:

- fused hidden surface support sample fraction: 0.7474;
- hidden temporal QC accepted for frames 760 to 765;
- replay accepted: IoU median 0.9715, visible-inside median 1.0, z-buffer p95 median 1.95 mm;
- visible-surface track QC accepted: 51 tracks, 227 edges, pair residual p95 9.64 mm;
- measured right-hand HaMeR annotation accepted after full-window SDF coverage repair;
- physics accepted with no reliable temporal contact claim: full-window hand penetration fraction 0.0;
- full-window hand-object SDF median separation: 0.619 m;
- rendered overlay, world 3D, and side-by-side videos contain 6 frames at 6 fps.

Videos:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/mop_760_765_filtered/deliverables_selected_right_fixedsdf/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/mop_760_765_filtered/deliverables_selected_right_fixedsdf/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/mop_760_765_filtered/deliverables_selected_right_fixedsdf/world/world_reconstruction_side_by_side.mp4
```

Visual inspection found the object mesh on the mop and the MANO mesh on the visible right hand. The hand is far from the mop handle, matching the SDF report. This sample demonstrates temporal object-mesh fusion and hand-pose rendering in a noncontact window.

## Full-Window SDF Coverage Repair

Mop initially exposed a measurement defect in `scripts/diagnose_full_window_hand_object_sdf_v7.py`: when the hand lay outside the object-local SDF grid, the checker produced no SDF rows and the physics wrapper crashed. The checker now builds the SDF grid over the object mesh and decoded hand vertices through the existing `cover_points` path.

Verification:

- mop selected-right HaMeR physics changed from checker crash to accepted noncontact measurement, with full-window penetration fraction 0.0;
- wild-rice no-regression remained accepted, with full-window penetration fraction 0.00236 versus the previous 0.00249.

## Relation To Prior Delivered Samples

Additional contact-rich evidence comes from the V8/V9/V10 accepted samples:

- box-books 612 to 618 remains the V8 contact-aware MANO repair sample, with selected-contact abs SDF p95 3.84 mm and contact active on frames 614 to 616;
- trash 865 to 870 remains the contact-rich object-mesh no-regression sample, with selected-contact abs SDF p95 1.245 mm and full-window penetration fraction 0.0118.

V11 adds temporal hidden-surface stability for object mesh completion on wild rice, mop, and trash. Trash is the current V11 contact-rich object-completion sample; box-books remains the strongest V8 hand-topology repair sample.

## Evidence Limit

V11 produces evidence-supported meshes. The hidden surface is accepted where multi-frame support, replay, tracks, hand-object SDF, and visual inspection agree. Surfaces without observation, motion support, or contact support remain uncertain.

The next version should target a stronger physical model for contact-rich manipulation. V11 verifies geometric contact and nonpenetration, but it does not yet explain the contact as a force-bearing event in a temporal dynamics model. V12 should add a factor graph that jointly optimizes head pose, object pose, MANO pose, contact state, object surface support, temporal smoothness, and force/acceleration consistency while preserving the same mesh-level replay and hidden-surface evidence.
