# Pipeline V10: Video-Conditioned Object Mesh Priors

## Starting Point

V9 added a real generated-mesh path, then accepted a small hidden component on the trash sample. The rejection mechanism was visible-shell competition: single-image complete priors usually produced surfaces that rendered in front of the measured object mask or depth. V10 therefore changes the object-prior source, while keeping the acceptance evidence unchanged.

The question V10 tests is:

```text
Can a video-conditioned generated mesh stream explain the same observed object across time better than a single-image complete prior while preserving image replay, surface tracks, MANO contact, and hand-object nonpenetration?
```

## State

For each object window, V10 keeps the measured-visible mesh archive as the visible evidence:

```text
M_t       measured visible object mesh from mask, depth, camera pose
P_t       video-conditioned generated object mesh for frame t
A_t       metric Sim3 or SE3 alignment from generated coordinates to world coordinates
G_t       delivered object mesh candidate after evidence filtering
H_t       MANO hand mesh and contact patches
T_wc_t    head camera pose
```

The generated stream `P_t` can come from Mesh4D, another video-conditioned reconstruction model, or a multi-view point/mesh completion model. The downstream path stays the same for every object: align, replay, track, physics, render.

## Evidence Contract

V10 uses the same observable agreement used by V7 through V9. Raw replacement and hidden completion are separate candidate modes:

1. measured target replay must pass before generated geometry is evaluated;
2. raw generated replacement candidates are a diagnostic path and must cover the measured visible mesh under the 10 mm p95 threshold before they can enter delivery;
3. hidden-completion candidates must preserve the measured visible mesh and reject generated faces that become visible or contradict mask, depth, free space, z-buffer, or measured-surface evidence;
4. full-fidelity z-buffer replay must pass silhouette, visible-inside, and depth p95 thresholds;
5. CoTracker surface factors must remain accepted on the candidate visible surface;
6. mesh-surface contact and selected-contact SDF must agree where contact evidence exists;
7. full-window hand-object SDF must keep hand penetration within threshold;
8. stakeholder render must show head camera, MANO hand, object mesh, contact marker, axes, scale, and caption clearly.

Single-image priors remain useful as proposals and negative controls. They become delivered geometry through the same checks.

## Implementation

The first V10 executable path used Mesh4D as the accessible video-conditioned generator:

```text
local RGBA sequence input
  -> A800 Mesh4D generation in tmux
  -> sync generated qc_mesh4d_sequence_v7.json and meshes
  -> archive_mesh4d_sequence_prior_v7.py
  -> run_v7_generated_prior_replay_qc.py for raw replacement diagnosis
  -> append_v10_mesh4d_hidden_faces_to_observed_mesh.py for observed-surface-preserving completion
  -> run_v7_video_mesh_replay_qc.py with triangle surface replay
  -> check_v7_candidate_track_surface_qc.py
  -> run_v7_candidate_physics_qc.py
  -> render_v7_candidate_deliverables.py
```

The evaluated V10 windows are:

- wild rice frames 2538 to 2543;
- trash frames 865 to 870;
- mop frames 760 to 765.

The local acceptance wrapper for raw generated streams is `scripts/run_v7_mesh4d_sequence_batch.py`. It discovers remote Mesh4D reports after sync, aligns each generated six-frame stream to the measured target, and runs full-fidelity replay. Hidden-face completion then preserves the measured visible mesh archive and appends only Mesh4D faces that pass the mask, depth, free-space, z-buffer, and measured-surface filters.

## Remote Execution Result

A800 host `192.168.11.220` completed the Mesh4D jobs under tmux. The 4090 host `192.168.9.220` timed out on SSH during V10 troubleshooting, so heavy Mesh4D generation ran on A800.

SAM 3D Objects currently lacks checkpoint access: Hugging Face returned HTTP 403 for `facebook/sam-3d-objects`. That route needs checkpoint authorization before it can generate object meshes.

Mesh4D setup required these environment and runner repairs:

- install `torch-cluster` for `torch_cluster.fps` in Mesh4D autoencoder blocks;
- install `plyfile` for `im2mesh.utils.io`;
- replace the bundled `im2mesh` pykdtree C extension with a SciPy `cKDTree` wrapper, because the bundled extension fails under Python 3.10 while Mesh4D uses KDTree query semantics.
- install `timm` for Hunyuan3D shape denoiser imports;
- pass the generated runtime `pipeline_cfg` into `Mesh4DPipeline`, because the source default latent shape produced `canonical_points.shape[1] != N`.

The durable launch path is `scripts/write_v10_mesh4d_consecutive_remote_job.sh`, which prepares consecutive six-frame Mesh4D runs without local heavy compute.

## Raw Mesh4D Replacement

Raw Mesh4D mesh replacement failed the visible-surface contract on every representative sample:

- trash visible-surface p95 median: 22.6 mm;
- wild rice visible-surface p95 median: 35.2 mm;
- mop visible-surface p95 median: 66.5 mm.

The measured visible-mesh target replay still passed on these samples, so the rejection localizes to generated mesh geometry mismatch rather than to the depth/mask/camera target.

## Hidden-Face Completion

V10 therefore uses Mesh4D as a hidden-geometry proposal while preserving the measured visible mesh. Two implementation issues were fixed before interpreting the completion results:

- sparse Mesh4D frames were evaluated only on their generated frame IDs, with no dense-frame interpolation;
- appended Mesh4D vertices are compacted to vertices referenced by retained faces, and completed-mesh replay uses triangle surfaces so orphan vertices cannot create a false visible silhouette.

Consecutive six-frame Mesh4D runs were needed for wild rice and mop because temporal track factors need consecutive evidence.

The hidden-face yield is itself evidence:

- trash kept a median of 0 Mesh4D faces per frame, with maximum 577, so its accepted delivery is mainly an observed-surface no-regression control after Mesh4D filtering;
- wild rice kept a median of 28,500 Mesh4D faces per frame;
- mop kept a median of 15,418 Mesh4D faces per frame.

The current track-surface QC constrains the candidate surface where model-produced tracks land on the observed visible object. Hidden faces that remain unseen throughout the window still need temporal stability, shape correctness, and physical relevance checks.

## Accepted Rendered Deliveries

### Trash 865-870

Root:

```text
/data2/ego_annotation_outputs/v10_mesh4d_outputs/fused_hidden/trash_mesh4d_hidden_865_870
```

Evidence:

- hidden completion kept a median of 0 Mesh4D faces per frame and maximum 577;
- replay accepted: IoU median 0.9478, visible-inside median 1.0, z-buffer p95 median 0.534 mm;
- track-surface accepted: 363 tracks, 1437 edges, pair residual p95 7.80 mm, zero correction displacement;
- contact physics accepted: 6 reliable contact rows, selected-contact abs SDF p95 1.245 mm, selected penetration 0, full-window hand penetration fraction 0.0118.

Videos:

```text
deliverables/overlay/mesh_surface_contact_review.mp4
deliverables/world/world_reconstruction_3d.mp4
deliverables/world/world_reconstruction_side_by_side.mp4
```

All six rendered frames were inspected as overlay, world-view, and side-by-side contact sheets. The mesh stays on the visible manipulated object, the MANO hand stays at the contact region, and the world view shows object mesh, hand, head-camera cue, scale, and caption.

### Wild Rice 2538-2543

Root:

```text
/data2/ego_annotation_outputs/v10_mesh4d_consecutive_outputs/fused_hidden/wild_rice_mesh4d_hidden_compact_2538_2543
```

Evidence:

- hidden completion kept a median of 28,500 Mesh4D faces per frame;
- replay accepted: IoU median 0.9460, visible-inside median 0.9866, z-buffer p95 median 6.03 mm;
- track-surface accepted on repaired single-archive factors for frames 2538 to 2541: 32 tracks, 67 edges, pair residual p95 8.02 mm, zero correction displacement;
- nonpenetration physics accepted with no reliable temporal contact claim: full-window hand penetration fraction 0.00305.

Videos:

```text
deliverables/overlay/mesh_surface_contact_review.mp4
deliverables/world/world_reconstruction_3d.mp4
deliverables/world/world_reconstruction_side_by_side.mp4
```

All six rendered frames were inspected as overlay, world-view, and side-by-side contact sheets. The overlay mesh stays on the active stem, MANO hands stay near the visible hands, the no-contact label is consistent with physics, and the standalone world view contains object mesh, hands, head-camera cue, scale, and caption.

## Accepted Object Branch With Missing Hand Evidence

Mop frames 760 to 765 produced an accepted object mesh replay and accepted temporal surface track:

```text
/data2/ego_annotation_outputs/v10_mesh4d_consecutive_outputs/fused_hidden/mop_mesh4d_hidden_compact_760_765
```

Evidence:

- hidden completion kept a median of 15,418 Mesh4D faces per frame;
- replay accepted: IoU median 0.9715, visible-inside median 1.0, z-buffer p95 median 1.95 mm;
- track-surface accepted: 51 tracks, 227 edges, pair residual p95 9.64 mm, zero correction displacement;
- physics rejected before SDF evaluation because the annotation table has zero MANO hand rows for frames 760 to 765.

Mop supports object-only replay and visible-surface tracking on a third representative object. Full annotation for this window still requires the hand evidence needed for hand-object physics.

## V11 Direction

V10 answers a narrower object-mesh question: video-conditioned meshes can provide evidence-consistent hidden completion proposals after filtering, while direct generated mesh replacement remains too inaccurate on the visible surface. Trash mainly tests no-regression after filtering; wild rice and mop test nonzero hidden-face proposals.

V11 should keep the observed-surface-preserving completion contract and improve the two causes exposed by V10:

1. repair missing MANO evidence for windows like mop 760-765, using WiLoR/HaMeR/RTMLib/SAM hand evidence in the contact-aware V8 factor graph;
2. add hidden-face temporal continuity and component-stability checks, because current track factors constrain observed visible surfaces;
3. improve generated object geometry by fusing multi-frame point evidence and learned completion under the same replay, track, and SDF constraints, instead of trusting raw generated visible surfaces;
4. add explicit head-pose and caption QC, because V10 uses those streams in rendering while independent accuracy evidence remains pending.

The downstream optimizer remains category-agnostic: masks, depths, tracks, mesh proposals, hand evidence, captions, and confidences enter as data; replay, temporal surface factors, contact SDF, nonpenetration, and rendering use one reconstruction path.

## V11 First Diagnostic

`scripts/check_v11_hidden_face_temporal_qc.py` measures appended hidden-face stability under the observed-surface motion factors. The diagnostic isolates appended Mesh4D faces from the archive using the append report, samples hidden surfaces, transforms consecutive-frame samples with CoTracker-derived object motion, and measures symmetric hidden-surface distance and hidden-face count jumps. Visible replay validates visible-image consistency; this diagnostic targets unseen geometry.

Results:

- trash 865-870: status `no_hidden_geometry`; only one frame retained hidden faces, so this sample remains an observed-surface no-regression delivery;
- wild rice 2538-2543: rejected; hidden-surface p95 distance 179 mm under full multi-anchor motion coverage, hidden-face count jump p95 above threshold, with hundreds to thousands of components per frame;
- mop 760-765: rejected; hidden-surface p95 distance 95 mm despite full pair coverage, with thousands of components per frame.

The hidden-face diagnostic changes the V11 priority: object mesh completion needs temporally fused geometry and component continuity in addition to visible replay and visible-surface tracks.

## V11 Temporal Fusion Result

`scripts/fuse_v11_temporal_hidden_surface.py` implements the next mesh step for wild rice. It transforms per-frame hidden Mesh4D proposals into a reference frame using CoTracker-derived object motion, keeps hidden points with multi-frame support, builds one shared hidden surface, maps that surface back to each frame, then applies the same mask, depth, free-space, z-buffer, and measured-surface filter used by V10.

Wild rice 2538-2543 accepted after temporal fusion and projection filtering:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/wild_rice_2538_2543_filtered
```

Evidence:

- fused hidden surface: 8,908 vertices and 55,465 faces before per-frame projection filtering;
- retained hidden faces per frame: median 37,704.5;
- hidden temporal QC accepted: symmetric hidden-surface p95 6.34 mm and hidden-face count log-step p95 0.059;
- replay accepted: IoU median 0.9458, visible-inside median 0.9834, z-buffer p95 median 6.03 mm;
- visible-surface track QC accepted for frames 2538 to 2541: 32 tracks, 67 edges, pair residual p95 8.02 mm;
- nonpenetration physics accepted with no reliable temporal contact claim: full-window hand penetration fraction 0.00249;
- rendered overlay, world 3D, and side-by-side videos each contain 6 frames at 6 fps, and all frames were inspected as contact sheets.

This V11 result fixes the hidden-geometry temporal failure exposed by the diagnostic on wild rice while preserving visible replay and hand-object nonpenetration. Mop still needs MANO hand evidence before full annotation physics can be evaluated.

Mop 760-765 also accepts on the object-only V11 mesh branch:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/mop_760_765_filtered
```

Evidence:

- fused hidden surface: 5,403 vertices and 33,821 faces before per-frame projection filtering;
- retained hidden faces per frame: median 3,656.5;
- hidden temporal QC accepted: symmetric hidden-surface p95 7.75 mm and hidden-face count log-step p95 0.138;
- replay accepted: IoU median 0.9715, visible-inside median 1.0, z-buffer p95 median 1.95 mm;
- visible-surface track QC accepted: 51 tracks, 227 edges, pair residual p95 9.64 mm;
- physics still rejects before SDF with `hand_rows=0`.

This result shows that V11 temporal fusion improves the object mesh branch on a second representative object. The full annotation result for mop remains blocked by missing MANO hand evidence in frames 760 to 765.
