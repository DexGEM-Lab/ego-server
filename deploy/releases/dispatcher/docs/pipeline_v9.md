# Pipeline V9: Evidence-Gated Object Mesh Completion

## Starting Point

V8 closed the hand-object contact failure on the box-with-books sample by optimizing MANO against RTMLib, SAM hand masks, metric depth, object SDF, temporal priors, and a selected-contact tail loss. The remaining object-pose problem is different: V7 and V8 delivered measured visible object meshes, while generated complete-object priors usually failed visible replay.

V9 keeps the object-mesh requirement intact. A generated mesh becomes delivered object geometry only when its visible contribution survives the same mask, depth, temporal-track, and hand-object physics checks used for measured meshes.

## Failure Mechanism From Prior Runs

The strongest available generated prior for the trash sample was PartCrafter on frame 880. Its original V7 replay alignment failed before z-buffer replay:

- observed target replay was valid: silhouette IoU median 0.9489 and z-buffer p95 median 0.53 mm;
- generated prior visible-surface p95 was 17.7 mm, above the 10 mm delivery threshold;
- hidden-surface conflict p95 was 70.7 mm.

Naive V9 fusion through Poisson reconstruction improved nearest-surface metrics but failed image replay. The generated front shell remained visible:

- visible-surface p95 improved to 3.37 mm median;
- full-fidelity replay failed with IoU median 0.837 and z-buffer p95 median 13.85 mm;
- visual inspection showed a cyan/fuzzy generated shell drawn over the observed lid.

Appending generated faces to the accepted measured mesh without iterative raster filtering also failed. Conservative centroid and vertex filters still left a visible front shell with z-buffer p95 around 108 mm. The cause was triangle visibility: after front faces were removed, deeper faces became the new rendered front surface.

## Representation

V9 represents object geometry as:

```text
M_t        accepted measured visible mesh for frame t
P         generated complete-object prior mesh
A_t       existing Sim3 from prior coordinates to frame-t world coordinates
F_t       accepted hidden-prior face set for frame t
G_t        delivered object mesh = M_t plus A_t(P[F_t])
```

The measured mesh is the authority for visible pixels. The generated prior can add only hidden geometry. Object class, color, material, and action enter through model-produced masks, depths, tracks, camera poses, MANO hands, captions, and contact reports. The downstream geometry path stays category-agnostic.

## Hidden-Face Filter

`scripts/append_v9_hidden_prior_faces_to_observed_mesh.py` implements the accepted V9 path.

For each frame:

1. decimate the generated prior to a bounded face count while preserving overall geometry;
2. transform the prior with the existing Sim3 row from V7 prior alignment;
3. remove generated faces whose projected centroids or vertices contradict the object mask, metric depth, free space, or proximity to the measured mesh;
4. iteratively render the remaining generated faces into an image-space z-buffer;
5. remove any generated face that becomes the rendered front surface outside the object mask, in front of the accepted measured object surface, or in unsupported visible mask area;
6. append the surviving generated faces to the measured mesh archive;
7. run unchanged replay, track-surface, physics, and render wrappers.

The iterative raster step is the key mechanism. One-shot filtering failed because removing one visible shell exposed the next one. V9 removes visible generated shells until the surviving prior faces are hidden from the observed camera or the iteration cap is reached.

## Accepted Probe

Target:

- sample: representative trash, frames 865 to 870;
- generated prior: `/data2/ego_annotation_outputs/v7_partcrafter_prior_outputs/generated_meshes/trash_0880/partcrafter_mesh.ply`;
- measured mesh: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_perframe_thick001_865_870/solidified_sheet_object_meshes_world.npz`;
- output root: `/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k`.

The prior was decimated from 3,191,236 faces to 60,000 faces. The accepted hidden-prior filter kept a median of 635 generated faces per frame. This small hidden component augments the measured visible lid.

Acceptance evidence:

- replay accepted: IoU median 0.9453, visible-inside median 0.9937, z-buffer p95 median 0.61 mm;
- CoTracker surface QC accepted: 363 tracks, 1,437 accepted edges, pair residual p95 7.80 mm, zero surface correction displacement;
- physics accepted: six reliable temporal contact rows, selected-contact abs SDF p95 1.25 mm, near-surface fraction 1.0, selected-contact penetration 0.0, full-window hand penetration 0.0117;
- deliverables rendered with six frames at 6 fps for overlay, standalone world 3D, and side-by-side video.

Deliverables:

- overlay video: `/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/overlay/mesh_surface_contact_review.mp4`
- world 3D video: `/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/world/world_reconstruction_3d.mp4`
- side-by-side video: `/data2/ego_annotation_outputs/representative_trash/v9_partcrafter_fused_prior_trash_865_870/observed_plus_hidden_prior_iterativeraster_60k/deliverables/world/world_reconstruction_side_by_side.mp4`

Visual inspection of frame 869 confirmed that the generated shell visible in rejected attempts is gone. The overlay shows the measured object mask, MANO hand, and contact markers coherently. The world view shows the object mesh, MANO hand, contact marker, head-camera frustum, view ray, axes, and scale.

## Interpretation

V9 is a stricter object-mesh completion step with visible replay as authority. It accepted a small generated hidden component because most PartCrafter geometry contradicted observed video. Object pose remains tied to measured video evidence.

The accepted V9 trash result improves the pipeline by adding a real generated-mesh pathway with falsification. The result also exposes the current limit: single-image complete priors usually compete with visible surfaces instead of supplying reliable hidden geometry. V10 should use video-conditioned reconstruction or multi-view object priors before attempting larger hidden completion.

## Next Version Direction

V10 should move the expensive iterative raster filtering to a GPU/vectorized rasterizer and test video-conditioned object priors:

- Mesh4D or another video-conditioned source where available;
- SAM 3D Objects once checkpoint access is available;
- VGGT or similar multi-view point/mesh completion constrained by masks and camera poses;
- the same replay, track-surface, physics, and visual-deliverable acceptance wrappers.

The acceptance rule should stay unchanged: generated geometry enters the delivered mesh only where measured video and physical hand-object evidence support it.
