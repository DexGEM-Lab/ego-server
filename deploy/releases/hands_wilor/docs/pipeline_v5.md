# Pipeline V5: Visibility-Aware Dynamic Object Mesh

## Starting Evidence

V4 has an accepted bounded result for wild-rice frames 2520 to 2550:

- per-frame measured/completed active-stem mesh archive with frame 2550 completed by SAM2 mask-seed propagation;
- overlay, 3D world animation, and side-by-side videos;
- z-buffer median silhouette IoU 0.964 and median z-buffer absolute p95 3.42 mm over 31 frames;
- 15 reliable temporal hand-object contact rows;
- selected-contact penetration 0 percent and full-hand penetration 0 percent under 1 mm SDF QC.

V4 also falsified a rigid canonical object map on the same frames. ICP/BPA fusion of the completed per-frame meshes produced an open canonical surface spanning 0.239 x 0.449 x 0.309 m. Replaying that single map through per-frame poses gave median silhouette IoU 0.373 and median z-buffer p95 80.1 mm. Visual QC showed a broad merged surface covering multiple stem/sheath regions. The cause is object-state change under peeling plus partial visibility near parallel plant material.

## Research Readout

Target-aligned hand-object reconstruction work exists, but direct drop-in code is limited:

- ForeHOI matches the daily hand-object video problem and combines 2D mask inpainting with 3D object completion. Its public repository currently lists inference and training code as unreleased.
- AGILE targets watertight object reconstruction with contact-aware tracking, but its repository currently says full code will be available by August 2026.
- HOLD is released and reconstructs category-agnostic hand/object surfaces from monocular video without object templates. It is a viable baseline/integration candidate, though its neural-field workflow is heavier than the accepted V4 measured-sheet evidence path.
- FoundationPose and RGBTrack are useful after a real mesh prior exists; they are pose trackers, not object mesh reconstructors.
- Mesh4D and ViSER support the representation direction of dynamic/deformable surfaces, but they are not immediate replacements for this project's full annotation contract.

V5 should therefore implement a project-owned dynamic object-surface graph while testing released baselines where they can be made to run.

## Representation

For each object track, V5 stores:

- per-frame measured visible surface mesh from mask, metric depth, camera pose, and intrinsics;
- per-frame source label: measured, model-propagated completion, rejected, or ambiguous;
- canonical dynamic surface coordinates for object material points when tracking is reliable;
- per-frame deformation field from canonical surface to observed frame;
- per-region visibility and measurement confidence;
- optional full-mesh prior from an image-to-3D or HOI reconstruction model;
- MANO vertices, keypoints, hand confidence, and contact patches;
- full-hand nonpenetration state and active-contact equality state.

The core state is a mesh or neural surface with visibility weights, deformable surface support, and explicit full-mesh uncertainty.

## Objective

The V5 graph minimizes robust residuals:

- measured visible-surface residual: observed per-frame mesh vertices must lie on the deformed object surface where visibility confidence is high;
- mask and z-buffer residual: rendered deformed surface must match the accepted mask and metric depth;
- temporal deformation residual: neighboring frames prefer low acceleration and local as-rigid-as-possible deformation;
- correspondence residual: model/track evidence links surface regions across frames when confidence is high;
- full-hand nonpenetration residual: non-contact MANO surface samples stay outside the object SDF;
- active-contact residual: contact-supported MANO patches lie on the object surface with near-zero signed distance;
- full-shape prior residual: generated or learned object mesh priors constrain unobserved regions with lower weight than measured depth/mask evidence.

The solver must expose residuals by frame and by source. A generated full mesh that fails z-buffer, contact, or nonpenetration QC is evidence of a bad prior, not a substitute object annotation.

## Implementation Order

1. Build a V5 dynamic-surface diagnostic on top of V4 accepted meshes.
   - Input: completed V4 mesh archive, manifest, annotations, contact rows, and SDF reports.
   - Output: per-frame local surface descriptors, visibility masks, temporal correspondence candidates, and a report explaining where a dynamic map is observable.
   - Acceptance: the diagnostic must identify stable surface regions and reject ambiguous merged surfaces such as frame 2539.

2. Add a deformable surface fit on a short high-quality subwindow.
   - Start with frames 2531 to 2537, where V3/V4 contact and z-buffer evidence are strongest.
   - Optimize per-frame deformations with ARAP and z-buffer residuals.
   - Acceptance: replayed mesh must preserve per-frame z-buffer p95 near the measured-sheet baseline and keep full-hand penetration at 0 percent.

3. Integrate a released full-object prior path.
   - First baseline: test HOLD on a cropped representative sequence if dependencies are runnable.
   - Second baseline: use an image-to-3D mesh prior from selected clean keyframes, then fit it to V4 measured surfaces and contact residuals.
   - Acceptance: generated prior must improve unobserved/full-shape plausibility without degrading measured-frame z-buffer/contact/SDF QC.

4. Upgrade the presentation renderer.
   - Replace diagnostic line-plot styling with a stakeholder 3D scene: shaded MANO surfaces, shaded object mesh, contact markers, V5 object-state badges, readable captions, and a separate head-camera trajectory inset.
   - Acceptance: side-by-side and standalone 3D must immediately communicate head pose, two hands, manipulated object mesh, and contact state.

## Current V5 Binary State

V5 is open. The rigid canonical-map branch has been falsified. The next concrete implementation target is the dynamic-surface observability diagnostic, followed by a deformable short-window fit only if the diagnostic finds stable surface regions.

## Dynamic-Surface Observability Result

The first V5 diagnostic measured which frames in the completed V4 archive are observable enough for dynamic surface fitting. A frame must pass three tests: accepted z-buffer residuals, PCA extent consistency against the track median, and stable nearest-neighbor surface overlap with at least one adjacent frame after local ICP.

The report marks 13 observable frames: 2522, 2523, 2525 to 2529, 2531, 2532, 2542, 2545, 2549, and 2550. The only contiguous window of length at least four is 2525 to 2529. This window has median z-buffer p95 1.71 mm, stable neighbor support for every frame, and one reliable contact row.

The contact-rich 2533 to 2537 interval is rejected for dynamic-map anchoring despite good per-frame z-buffer residuals. Frames 2534 to 2537 are extent outliers, and frame 2533 lacks stable temporal overlap. This matches the visual failure mode: the selected measured masks include parallel stem/sheath surfaces, so the frames are contact evidence with ambiguity, not a stable object-map anchor.

Diagnostic artifact:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_observability_completed_2520_2550/qc_dynamic_surface_observability_v5.json`

The next V5 implementation should fit the first dynamic surface on frames 2525 to 2529 as a geometry-first window. Contact-rich ambiguous frames should enter later with uncertainty weights or segmentation repair.

## Observable-Window Rigid Map Falsification

The 2525 to 2529 observable window was tested with the same ICP/BPA rigid-map replay. This smaller window still fails the delivered-geometry contract. The reconstructed canonical map is open, spans 0.139 x 0.317 x 0.214 m, and independent z-buffer replay gives median silhouette IoU 0.563 and median z-buffer p95 24.1 mm over five frames. The measured-sheet baseline for the same window has median z-buffer p95 1.71 mm.

The failure frame at 2527 shows a broad reconstructed surface over the held stem. This rejects another round of rigid-map tuning. V5 should preserve per-frame measured meshes as the delivered object geometry, then add temporal regularization, segmentation repair, and deformable/local surface correspondences around that geometry.

Artifacts:

- Rigid map report and archive: `/data2/ego_annotation_outputs/representative_wild_rice/v5_observable_window_rigid_map_icp_bpa_2525_2529/`
- Independent z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_observable_window_rigid_map_icp_bpa_zbuffer_qc_2525_2529/qc_mesh_zbuffer_projection_v3.json`
- Visual failure frame: `/data2/ego_annotation_outputs/representative_wild_rice/v5_observable_window_rigid_map_icp_bpa_zbuffer_qc_2525_2529/frame_2527_qc.jpg`

## Dynamic-Surface State Package

The current V5 object state preserves the completed V4 per-frame mesh archive as accepted geometry and adds explicit per-frame map-observability labels. The package classifies frames as:

- 12 map-observable measured geometry frames;
- 12 ambiguous measured geometry frames;
- 6 ambiguous contact geometry frames;
- 1 completed geometry frame, 2550 from `sam2_mask_seed_from_2549`.

The ambiguous contact frames are 2533, 2534, 2535, 2536, 2546, and 2547. They contain reliable contact rows, while their object geometry lacks stable dynamic-map support. Downstream optimization should consume their contact evidence with uncertainty.

State package:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_state_completed_2520_2550/v5_dynamic_surface_state.json`

## Presentation Upgrade

The V5 renderer now consumes the state package directly. If a V5 state file is supplied, every rendered frame must have a state row. The world panel renders a large manipulation close-up in metric world coordinates, with the head-camera trajectory in a separately scaled inset. MANO surfaces use the real MANO face topology from `/data/dex_home/yiwen/hand_trajectory_loader/assets/mano/models/MANO_RIGHT.pkl`; the annotation stream supplies the per-frame MANO vertices. The object mesh remains the accepted completed V4 per-frame mesh archive.

Rendered artifacts:

- side-by-side video: `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/world_reconstruction_side_by_side.mp4`
- standalone 3D video: `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/world_reconstruction_3d.mp4`
- render manifest: `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/render_manifest.json`
- inspected stills: `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/stills/frame_002520.jpg`, `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/stills/frame_002535.jpg`, and `/data2/ego_annotation_outputs/representative_wild_rice/v5_world_reconstruction_state_presentation_2520_2550/stills/frame_002550.jpg`

The videos contain 31 frames at 6 fps. The side-by-side render is 1920 x 778 and the standalone 3D render is 960 x 720. Visual inspection of measured, contact-ambiguous, and completed frames shows that the object mesh, MANO surfaces, contact patch, V5 state, semantic caption, and head trajectory are readable in the same frame.

## First Dynamic-Topology Fit Falsification

A first category-agnostic dynamic-surface graph was tested on the observable window 2525 to 2529. The graph used frame 2527 as an anchor, simplified its measured mesh to one 3960-vertex, 8000-face topology, initialized each frame by rigid ICP, then solved vertex positions with measured-surface nearest-neighbor residuals, Laplacian preservation, and temporal smoothness. The solved archive is:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_graph_observable_2525_2529/dynamic_surface_meshes_world.npz`
- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_graph_observable_2525_2529/qc_dynamic_surface_graph_v5.json`

Independent z-buffer replay rejects this topology-sharing fit as delivered geometry. On the same 2525 to 2529 window and the same UniDepth/VGGT projection contract, the measured per-frame mesh baseline has median silhouette IoU 0.975 and median z-buffer p95 1.71 mm. The dynamic-topology fit has median silhouette IoU 0.840 and median z-buffer p95 8.55 mm. Frame 2528 drops to IoU 0.779 and z-buffer p95 16.2 mm. The visual failure is a smoothed/shrunken thin-stem surface: the shared topology and nearest-neighbor correspondences preserve median depth but lose silhouette support.

Artifacts:

- dynamic-fit z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_graph_observable_zbuffer_qc_2525_2529/qc_mesh_zbuffer_projection_v3.json`
- measured baseline z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_measured_baseline_zbuffer_qc_2525_2529/qc_mesh_zbuffer_projection_v3.json`
- failure still: `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_graph_observable_zbuffer_qc_2525_2529/stills/frame_002527.png`

This falsifies the first shared-topology dynamic fit. V5 should keep the per-frame measured/completed meshes as delivered geometry and place temporal reasoning in correspondence or transport edges between stable surface samples, then use those edges for smoothing or repair only when replay QC does not degrade.

## Strict Transport-Edge Diagnostic

A second V5 diagnostic kept the accepted per-frame measured meshes unchanged and attempted to add only temporal transport edges. For each neighboring pair in 2525 to 2529, it aligned sampled surfaces by local ICP and kept mutual nearest-neighbor edges below 6 mm. This avoids changing the delivered object geometry.

The strict transport diagnostic also rejects dense dynamic-state closure on this window. It stores 48,000 proximity edges, capped at 12,000 per pair, but the accepted overlap fraction is only about 0.18 median and no neighboring pair reaches the 0.45 stable-pair threshold. Source-to-target p95 is 8.16 mm median and target-to-source p95 is 12.76 mm median. The thin peeled-stem surfaces support local proximity edges, not a dense material correspondence map.

Artifact:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_dynamic_surface_transport_edges_2525_2529/dynamic_surface_transport_edges_v5.json`

Current V5 conclusion: the deliverable geometry should remain the per-frame measured/completed mesh archive with V5 state labels. Dynamic regularization is still open. The next valid mechanism is segmentation repair plus visibility-aware local patch tracking, or a stronger released hand-object reconstruction baseline such as HOLD, evaluated by the same z-buffer/contact/SDF replay. A canonical rigid mesh, a shared simplified topology, or dense transport edges should not be presented as solved object pose for this wild-rice window.

## Segmentation-Repair Contact Window

The first V5 repair branch targets the contact-rich but object-ambiguous interval 2532 to 2537. The input seed is the clean active-stem mask at frame 2532. SAM2 video propagation tracks that instance through frames 2532 to 2537, and visual inspection of frames 2534 and 2536 shows that the repaired mask follows the held stem while removing much of the adjacent parallel stem/sheath support that made the V4 state contact-ambiguous.

The repaired mask stream was converted through the same category-agnostic geometry path as the accepted V4 meshes: RGB/mask manifest, UniDepth metric depth, annotation VGGT intrinsics, observed mask-depth surface, 1 mm watertight sheet solidification, z-buffer replay, mesh-surface contact, selected-contact SDF, and full-hand SDF. The repair-specific state package marks the six frames as `segmentation_repaired_geometry`; the renderer displays this as `repaired mesh`.

Repair artifacts:

- SAM2 mask track: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_2531_2537/sam2_mask_seed_track_local.json`
- mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_solidified_thick001_2532_2537/solidified_sheet_object_meshes_world.npz`
- z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_zbuffer_qc_2532_2537/qc_mesh_zbuffer_projection_v3.json`
- contact QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_solidified_thick001_2532_2537/mesh_surface_contact_recomputed_det015.json`
- selected-contact SDF QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_solidified_thick001_2532_2537/volume_sdf_contact_recomputed_det015_pitch001_qc.json`
- full-hand SDF QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_solidified_thick001_2532_2537/full_hand_sdf_penetration_recomputed_det015_pitch001_qc.json`
- repair state package: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_seed2532_state_2532_2537/v5_segmentation_repair_state.json`
- side-by-side video: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_world_reconstruction_2532_2537/world_reconstruction_side_by_side.mp4`
- standalone 3D video: `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_world_reconstruction_2532_2537/world_reconstruction_3d.mp4`

The repaired stream improves the exact 2532 to 2537 image/depth contract relative to the V4 completed stream. V4 completed gives median silhouette IoU 0.979 and median z-buffer p95 2.56 mm on this window. The repaired stream gives median silhouette IoU 0.991 and median z-buffer p95 1.46 mm. Mesh-surface contact remains accepted for both hands on every frame: 12 of 12 geometry-backed rows are reliable temporal contact rows. Selected-contact SDF has 0 percent penetration, median absolute SDF 1.19 mm, and p95 absolute SDF 8.54 mm over 64 selected contact samples. Full-hand SDF has 0 percent penetration over 4,611 sampled MANO vertices, with median signed distance 17.1 mm outside the object.

Structural render QC confirms both repair videos contain six frames at 6 fps. The side-by-side render is 1920 x 778, and the standalone 3D render is 960 x 720. Visual inspection of frames 2534 and 2536 shows the repaired object mesh, two MANO surfaces, contact patch marker, semantic caption, repaired-mesh badge, and head-camera trajectory inset in the same frame.

This branch closes a bounded V5 segmentation-repair result for frames 2532 to 2537. It does not close dynamic material correspondence, full-sequence object completion, or a global deformable object model. Those remain open V5 mechanisms after the rigid map, shared-topology dynamic fit, and dense transport-edge branches were falsified.

## Repaired-Window Transport Falsification

The repaired 2532 to 2537 mesh archive was tested with the same transport-edge diagnostic, with `segmentation_repaired_geometry` supplied as an explicit allowed state. This keeps the accepted repaired per-frame meshes unchanged and asks only whether local temporal correspondences are stable enough for a dynamic material map.

The result still rejects dense material transport. The diagnostic found 54,617 proximity edges over five neighboring frame pairs, but no pair reached the 0.45 stable-overlap threshold. Median accepted overlap is 0.15. Median source-to-target p95 distance is 6.64 mm, and median target-to-source p95 distance is 7.71 mm. The repaired segmentation improves per-frame instance identity and physical contact evidence, but the visible thin-stem surface still lacks enough repeated material support for dense correspondence.

Artifact:

- `/data2/ego_annotation_outputs/representative_wild_rice/v5_segmentation_repair_transport_edges_2532_2537/dynamic_surface_transport_edges_v5.json`

## Learned Point-Track Diagnostic

A first learned correspondence branch runs CoTracker3 on the repaired 2532 to 2537 RGB/mask sequence. Query points are sampled inside the repaired frame-2532 mask, tracked by the official `facebookresearch/co-tracker` PyTorch Hub model `cotracker3_offline`, then filtered by CoTracker visibility, repaired mask support, and UniDepth validity. Accepted tracks are lifted to metric world coordinates with the same annotation VGGT intrinsics and camera poses used by the mesh pipeline.

CoTracker produces sparse material-candidate tracks, not dense correspondence closure. A frame-2532 seed run with a 28 px query grid tracks 58 seed points and keeps 18 tracks accepted through all six frames. A stronger middle-query run samples frame 2535 with a 24 px grid and backward tracking. It tracks 86 seed points, keeps 29 tracks accepted through all six frames, and raises accepted track counts to 37, 50, 72, 86, 81, and 77 across frames 2532 to 2537. The middle-query median valid-frames-per-track count is 5.0, and the p95 is 6.0. Its accepted consecutive world-step median is 10.5 mm, with p95 31.2 mm. Visual inspection of frames 2532, 2536, and 2537 shows the retained tracks stay on the active stem; lost/rejected tracks occur near occlusion, mask edges, and the small bottom fragment.

This is useful evidence for a future sparse correspondence factor, because it supplies model-produced temporal point hypotheses where geometry-only nearest neighbors had no stable dense overlap. It is not enough to declare a dynamic material map: the accepted track set is sparse, and p95 world motion is too large to use as an unqualified rigid/deformable regularizer. The next V5 graph should consume these tracks as confidence-weighted sparse factors alongside mask/depth/SDF residuals, with explicit rejection of tracks that leave the repaired mask or jump in world space.

The sparse-track edge diagnostic converts the stronger middle-query CoTracker run into mesh-anchored correspondence edges without changing any delivered mesh. Each accepted world point is attached to the nearest repaired mesh vertex in each frame. Tracks must have at least four accepted frames, stay within 4 mm of the repaired mesh, and form consecutive-frame edges with world step below 40 mm. This yields 69 usable tracks, including 29 tracks visible across all six frames, and 272 consecutive correspondence edges. Edge world-step median is 10.4 mm and p95 is 27.3 mm. Edge surface-distance median is 0.17 mm and p95 is 0.30 mm.

These edges are the first V5 signal that can support a sparse dynamic factor graph. They do not replace the per-frame measured meshes and do not prove dense material correspondence. The next graph should keep the per-frame repaired mesh archive as the delivered geometry and use these sparse edges only as a regularizer or diagnostic factor, with z-buffer/contact/SDF replay required after any deformation.

Local rigidity QC is mixed. Across consecutive frame pairs, the sparse tracks have median rigid-fit residual 5.0 mm and median pairwise length error 3.39 mm, but p95 pairwise length error is 15.3 mm. The strongest pair is 2535 to 2536, with rigid RMSD 3.78 mm and pairwise length-error p95 9.71 mm. The weakest early pair, 2532 to 2533, has rigid RMSD 16.2 mm and pairwise length-error p95 49.4 mm. The learned tracks can therefore be robust sparse factors after residual clipping or confidence weighting. They should not be used as hard material constraints across the whole window.

Artifacts:

- frame-2532 seed QC report: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_2532_2537/qc_cotracker_object_tracks_v5.json`
- middle-query QC report: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/qc_cotracker_object_tracks_v5.json`
- middle-query track archive: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_object_tracks_v5.npz`
- middle-query overlay video: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_tracks_overlay.mp4`
- sparse correspondence edges: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/cotracker_sparse_correspondence_edges_v5.json`
- local rigidity QC: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/qc_cotracker_local_rigidity_v5.json`
- inspected stills: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/stills/frame_002532.jpg`, `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/stills/frame_002536.jpg`, and `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/stills/frame_002537.jpg`
