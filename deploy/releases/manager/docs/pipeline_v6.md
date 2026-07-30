# Pipeline V6: Robust Sparse Correspondence Factors

## Starting State

V5 has two accepted wild-rice geometry streams:

- completed 31-frame active-stem archive, frames 2520 to 2550, with measured/completed state labels and stakeholder-readable render videos;
- repaired six-frame active-stem archive, frames 2532 to 2537, created from a SAM2 seed track and verified by z-buffer, mesh-surface contact, selected-contact SDF, full-hand SDF, and visual render QC.

V5 also falsified three dense temporal object-map hypotheses:

- a rigid canonical mesh over the completed 31-frame sequence;
- a shared-topology dynamic mesh over the observable 2525 to 2529 window;
- dense transport edges over both the observable window and the repaired 2532 to 2537 window.

The repaired sequence has strong per-frame object geometry and contact evidence, while the dense material map remains unobservable from nearest-neighbor geometry alone. V6 therefore moves temporal object reasoning from dense correspondence to sparse learned correspondence factors.

## Representation

V6 keeps the V5 repaired per-frame mesh archive as delivered geometry. CoTracker points become auxiliary graph observations:

- each track has frame-local image coordinates, visibility, mask support, and lifted world coordinates;
- each accepted world point is attached to the nearest repaired mesh surface in each visible frame;
- each neighboring frame pair gets a candidate SE3 factor only from tracks visible in both frames;
- each factor stores its source tracks, robust fit residuals, inlier set, and acceptance state.

This representation keeps visual case variation out of hand-written logic. The point tracks come from a learned video tracker, masks come from the repaired SAM2 object track, and the graph consumes masks, depths, mesh attachments, and residuals through one category-agnostic path.

## Objective

For a neighboring frame pair with accepted track positions `x_i` and `y_i`, V6 estimates a robust rigid transform:

```text
min_R,t sum_i rho(||x_i R + t - y_i||_2)
```

where `R` is a proper rotation, `t` is translation in meters, and `rho` is the Huber loss implemented by iterative reweighted least squares. Each IRLS step solves a weighted Kabsch problem, recomputes point residuals, and updates weights by:

```text
w_i = min(1, delta / max(residual_i, eps))
```

The factor is ready only when enough inlier tracks remain and the inlier p95 residual stays under the chosen tolerance. The current strict settings are:

- at least 12 pair tracks;
- at least 12 inlier tracks;
- Huber delta 10 mm;
- inlier residual threshold 12 mm;
- accepted inlier p95 threshold 10 mm.

The graph can later use ready factors as sparse temporal priors on object pose or local deformation. Rejected factors stay in the report as evidence about occlusion, object bending, bad tracks, or changing visible support.

## First Diagnostic

Input artifacts:

- CoTracker archive: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_object_tracks_v5.npz`
- mesh-anchored sparse edges: `/data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/cotracker_sparse_correspondence_edges_v5.json`

Command:

```bash
.venv/bin/python scripts/fit_cotracker_pairwise_rigid_factors_v6.py \
  --cotracker-npz /data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_repaired_object_tracks_midquery2535_2532_2537/cotracker_object_tracks_v5.npz \
  --sparse-edges-json /data2/ego_annotation_outputs/representative_wild_rice/v5_cotracker_sparse_edges_midquery2535_2532_2537/cotracker_sparse_correspondence_edges_v5.json \
  --output-json /data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_pairwise_rigid_factors_2532_2537/qc_cotracker_pairwise_rigid_factors_v6.json \
  --min-pair-tracks 12 \
  --min-inlier-tracks 12 \
  --huber-delta-m 0.010 \
  --max-inlier-residual-m 0.012 \
  --accept-inlier-p95-m 0.010 \
  --irls-iterations 8
```

Report:

- `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_pairwise_rigid_factors_2532_2537/qc_cotracker_pairwise_rigid_factors_v6.json`

Result summary:

- usable tracks: 69
- neighboring frame pairs: 5
- rigid-factor-ready pairs: 2
- ready-pair inlier residual median: 3.36 mm
- ready-pair inlier residual p95: 8.27 mm
- all clipped inlier residual p95 across every pair: 10.06 mm

Pair results:

| Pair | Tracks | Inliers | Ready | Median residual | P95 residual | Inlier p95 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 2532 to 2533 | 31 | 25 | no | 8.56 mm | 27.77 mm | 11.53 mm |
| 2533 to 2534 | 46 | 39 | no | 6.91 mm | 19.36 mm | 11.65 mm |
| 2534 to 2535 | 66 | 61 | no | 5.38 mm | 13.33 mm | 10.06 mm |
| 2535 to 2536 | 69 | 69 | yes | 2.91 mm | 6.53 mm | 6.53 mm |
| 2536 to 2537 | 69 | 67 | yes | 3.98 mm | 11.42 mm | 8.50 mm |

## Interpretation

The learned point tracks produce useful sparse graph factors on the late repaired interval, especially 2535 to 2537. Earlier pairs have enough visible tracks, but their tail residuals exceed the 10 mm inlier-p95 tolerance. Those early pairs should enter the next graph as weak or rejected evidence, because a hard temporal material constraint would distort the accepted per-frame mesh geometry.

V6 has therefore established the first pair-local temporal correspondence factors with millimeter-scale residuals on part of the repaired window. The delivered object mesh remains the V5 repaired per-frame mesh archive until a graph that uses these factors is solved and then replayed through z-buffer, contact, selected-contact SDF, full-hand SDF, and visual QC.

## Next Implementation

The next V6 graph should add these pair-local factors without changing the accepted geometry by default:

1. load V5 repaired meshes, MANO vertices, contact rows, CoTracker tracks, and robust pair factors;
2. optimize per-frame object pose and optional low-dimensional local deformation, weighted by ready-pair factors only where the factor report marks them ready;
3. preserve the per-frame z-buffer residual as a hard delivery check after solve;
4. recompute mesh-surface contact, selected-contact SDF, and full-hand SDF on the solved archive;
5. render only graph states that pass the same visual and metric QC used by V4/V5.

The first solve should target frames 2535 to 2537, because both neighboring factors pass the strict residual criterion. Frames 2532 to 2535 need weaker correspondence handling or additional perception evidence before they can support temporal smoothing.

## Transport Replay Test

The first graph-use test transports the accepted source-frame mesh through each ready CoTracker SE3 factor and indexes the transported mesh by the target frame. It then compares the transported surface to the accepted target-frame mesh before any image replay.

Artifacts:

- transported mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_2535_2537/transported_ready_pair_meshes_world.npz`
- transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_2535_2537/qc_transport_ready_pair_meshes_v6.json`
- z-buffer replay report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_transport_ready_pairs_zbuffer_qc_2536_2537/qc_mesh_zbuffer_projection_v3.json`

Surface transport succeeds on the common visible surface:

| Pair | Bidirectional median | Bidirectional p95 |
| --- | ---: | ---: |
| 2535 to 2536 | 0.85 mm | 3.14 mm |
| 2536 to 2537 | 1.69 mm | 5.34 mm |

Target-frame z-buffer replay gives low depth residual but poor full-silhouette agreement:

| Target frame | Silhouette IoU | Visible inside mask | Z-buffer p95 |
| --- | ---: | ---: | ---: |
| 2536 | 0.710 | 0.939 | 6.45 mm |
| 2537 | 0.629 | 0.912 | 9.54 mm |

This is the expected distinction between material-patch tracking and full object-mask propagation. The ready CoTracker factors are valid sparse temporal factors for a stable common surface region. The transported source mesh misses or overdraws target-frame silhouette regions when the visible support changes. V6 should use these factors for local pose/deformation regularization and missing-patch support, then keep measured target masks/depth as the authority for delivered mesh coverage.

## Wider Ambiguity-Bridge Test

The wider CoTracker run uses the completed V4 sequence from 2532 to 2550 and queries the same clean source frame 2535. This tests whether learned sparse correspondence can carry object evidence from the repaired contact interval into later ambiguous or completed frames.

Artifacts:

- wide CoTracker run: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_midquery2535_2532_2550/`
- wide mesh-anchored sparse edges: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_sparse_edges_midquery2535_2532_2550/cotracker_sparse_correspondence_edges_v6.json`
- wide pairwise factor report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_pairwise_rigid_factors_meshanchored_2532_2550/qc_cotracker_pairwise_rigid_factors_v6.json`
- wide transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_transport_ready_pairs_2534_2538/qc_transport_ready_pair_meshes_v6.json`
- wide transported-mesh z-buffer replay: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_wide_transport_ready_pairs_zbuffer_qc_2535_2538/qc_mesh_zbuffer_projection_v3.json`

The tracker does not produce long-range all-frame tracks. Accepted track count is strong from 2533 to 2537, drops to 39 at 2538, and falls below the 12-track factor threshold after 2541. After mesh-surface anchoring, 88 tracks remain usable and 421 neighboring-frame edges survive.

Mesh-anchored pairwise factor readiness:

| Pair | Tracks | Inliers | Ready | Inlier p95 |
| --- | ---: | ---: | --- | ---: |
| 2532 to 2533 | 37 | 24 | no | 11.00 mm |
| 2533 to 2534 | 72 | 53 | no | 10.48 mm |
| 2534 to 2535 | 85 | 79 | yes | 9.59 mm |
| 2535 to 2536 | 87 | 83 | yes | 6.90 mm |
| 2536 to 2537 | 74 | 70 | yes | 7.64 mm |
| 2537 to 2538 | 32 | 29 | yes | 9.67 mm |

The new useful fact is the 2537 to 2538 bridge: visual inspection of the 2538 overlay shows retained tracks on the active stem mask, and the pair passes the strict residual criterion. This extends sparse temporal evidence into the first later ambiguous measured frame.

Transport replay still rejects the transported source meshes as full target annotations. The four ready-pair transported surfaces have bidirectional surface medians between 1.32 and 1.99 mm, and p95 residuals between 4.89 and 8.29 mm. However, full z-buffer replay over target frames 2535 to 2538 gives median silhouette IoU 0.681 and median z-buffer p95 20.46 mm. Frame 2538 has IoU 0.701, visible-inside-mask 0.856, and z-buffer p95 21.55 mm.

V6 conclusion after the wide run: learned sparse factors bridge stable local surface patches from the repaired contact interval into frame 2538. Track support collapses through the later interval, and per-frame mask/depth geometry remains the source of delivered mesh coverage even where a pair factor is ready. The next valid solver should use these factors as local temporal priors with measured target masks/depth as hard replay checks.

## Multi-Anchor Factor Coverage

The single wide query establishes a useful bridge through frame 2538, then loses support. V6 therefore tested additional CoTracker anchors at frames 2542, 2545, and 2549. Each anchor uses the same category-agnostic path:

1. sample learned object tracks from the model-produced active-stem mask;
2. attach retained tracks to the measured mesh surface;
3. fit pair-local robust SE3 factors with the same IRLS Kabsch objective;
4. merge candidates by the strict ready flag, lower inlier p95, higher inlier count, and lower inlier median.

Artifacts:

- anchor 2542 tracks: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2542_2532_2550/`
- anchor 2545 tracks: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2545_2532_2550/`
- anchor 2549 tracks: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2549_2532_2550/`
- merged factor report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor_merged_pair_factors_2532_2550/qc_merged_pair_factors_v6.json`

The merged report has 18 neighboring pairs, 14 ready pairs, and 4 rejected pairs. The ready-pair inlier p95 median is 6.57 mm, with p95 of the ready-pair p95 values at 9.62 mm.

| Pair | Anchor | Tracks | Inliers | Inlier p95 |
| --- | --- | ---: | ---: | ---: |
| 2532 to 2533 | wide 2535 | 37 | 24 | 11.00 mm |
| 2533 to 2534 | wide 2535 | 72 | 53 | 10.48 mm |
| 2534 to 2535 | wide 2535 | 85 | 79 | 9.59 mm |
| 2535 to 2536 | wide 2535 | 87 | 83 | 6.90 mm |
| 2536 to 2537 | wide 2535 | 74 | 70 | 7.64 mm |
| 2537 to 2538 | wide 2535 | 32 | 29 | 9.67 mm |
| 2538 to 2539 | wide 2535 | 5 | 0 | rejected |
| 2539 to 2540 | wide 2535 | 5 | 0 | rejected |
| 2540 to 2541 | anchor 2542 | 25 | 23 | 8.58 mm |
| 2541 to 2542 | anchor 2542 | 28 | 27 | 6.24 mm |
| 2542 to 2543 | anchor 2542 | 28 | 28 | 3.46 mm |
| 2543 to 2544 | anchor 2542 | 25 | 25 | 5.63 mm |
| 2544 to 2545 | anchor 2542 | 20 | 17 | 8.62 mm |
| 2545 to 2546 | anchor 2545 | 26 | 18 | 9.48 mm |
| 2546 to 2547 | anchor 2545 | 25 | 24 | 4.08 mm |
| 2547 to 2548 | anchor 2542 | 17 | 17 | 3.29 mm |
| 2548 to 2549 | anchor 2542 | 17 | 17 | 2.81 mm |
| 2549 to 2550 | anchor 2542 | 19 | 19 | 2.54 mm |

The four-anchor report left 2538 to 2539 and 2539 to 2540 unlinked because all four tested anchors had too little support there. The next test therefore queried CoTracker inside that interval instead of treating the missing factors as a final perception limit.

## Gap-Query Anchor

The rejected 2538 to 2540 interval was then tested directly by querying CoTracker inside the gap at frame 2539. The first A800 attempt failed before tracking because the TripoSR virtualenv now raises a native bus error while importing PyTorch. The run was repeated in the TRELLIS virtualenv, which imports PyTorch 2.4.0+cu121 and loads the cached CoTracker model.

Artifacts:

- frame-2539 CoTracker run: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2539_2532_2550/`
- frame-2539 sparse edges: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2539_sparse_edges_2532_2550/cotracker_sparse_correspondence_edges_v6.json`
- frame-2539 pair factors: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_anchor2539_pairwise_rigid_factors_meshanchored_2532_2550/qc_cotracker_pairwise_rigid_factors_v6.json`
- five-anchor merged factor report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_merged_pair_factors_2532_2550/qc_merged_pair_factors_v6.json`

The frame-2539 query produced 51 query points and 17 mesh-anchored usable tracks. It added two strict ready factors:

| Pair | Tracks | Inliers | Inlier median | Inlier p95 |
| --- | ---: | ---: | ---: | ---: |
| 2538 to 2539 | 13 | 12 | 5.31 mm | 9.01 mm |
| 2539 to 2540 | 15 | 15 | 2.90 mm | 5.89 mm |

The five-anchor merge now has 18 neighboring pairs, 16 ready pairs, and 2 rejected early pairs. The ready-pair inlier p95 median stays 6.57 mm, with p95 of the ready-pair p95 values at 9.61 mm. The ready chain is continuous from 2534 to 2550:

| Pair | Anchor | Inliers | Inlier p95 |
| --- | --- | ---: | ---: |
| 2532 to 2533 | wide 2535 | 24 | 11.00 mm |
| 2533 to 2534 | wide 2535 | 53 | 10.48 mm |
| 2534 to 2535 | wide 2535 | 79 | 9.59 mm |
| 2535 to 2536 | wide 2535 | 83 | 6.90 mm |
| 2536 to 2537 | wide 2535 | 70 | 7.64 mm |
| 2537 to 2538 | wide 2535 | 29 | 9.67 mm |
| 2538 to 2539 | anchor 2539 | 12 | 9.01 mm |
| 2539 to 2540 | anchor 2539 | 15 | 5.89 mm |
| 2540 to 2541 | anchor 2542 | 23 | 8.58 mm |
| 2541 to 2542 | anchor 2542 | 27 | 6.24 mm |
| 2542 to 2543 | anchor 2542 | 28 | 3.46 mm |
| 2543 to 2544 | anchor 2542 | 25 | 5.63 mm |
| 2544 to 2545 | anchor 2542 | 17 | 8.62 mm |
| 2545 to 2546 | anchor 2545 | 18 | 9.48 mm |
| 2546 to 2547 | anchor 2545 | 24 | 4.08 mm |
| 2547 to 2548 | anchor 2542 | 17 | 3.29 mm |
| 2548 to 2549 | anchor 2542 | 17 | 2.81 mm |
| 2549 to 2550 | anchor 2542 | 19 | 2.54 mm |

Frame 2539 remains visually and geometrically ambiguous. The gap-query anchor supplies sparse temporal constraints through it, while the delivered mesh quality for that frame remains controlled by the measured segmentation and depth evidence.

## Multi-Anchor Transport Replay

The merged ready factors were replayed as a falsification test by transporting the accepted source-frame mesh into each target frame. The four-anchor replay tested 14 factors, and the five-anchor replay tested 16 factors including the 2538 to 2540 gap.

Artifacts:

- four-anchor transported mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor_transport_ready_pairs_2534_2550/transported_ready_pair_meshes_world.npz`
- four-anchor transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor_transport_ready_pairs_2534_2550/qc_transport_ready_pair_meshes_v6.json`
- four-anchor selected-frame z-buffer replay: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor_transport_zbuffer_qc_selected_2535_2550/qc_mesh_zbuffer_projection_v3.json`
- five-anchor transported mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_transport_ready_pairs_2534_2550/transported_ready_pair_meshes_world.npz`
- five-anchor transport residual report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_transport_ready_pairs_2534_2550/qc_transport_ready_pair_meshes_v6.json`
- five-anchor all-frame z-buffer replay: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_transport_zbuffer_splat0_qc_2535_2550/qc_mesh_zbuffer_projection_v3.json`

Four-anchor common-surface transport looks strong in nearest-surface space:

- transported pair count: 14
- bidirectional median, across pairs: 1.36 mm
- bidirectional p95, across pairs: median 7.42 mm
- bidirectional p95, across pairs: p95 30.18 mm

Five-anchor common-surface transport keeps a low median but exposes large tail error on the two newly linked gap pairs:

- transported pair count: 16
- bidirectional median, across pairs: 1.47 mm
- bidirectional p95, across pairs: median 7.41 mm
- bidirectional p95, across pairs: p95 118.59 mm
- 2538 to 2539 bidirectional p95: 142.29 mm
- 2539 to 2540 bidirectional p95: 110.69 mm

Full target-frame replay rejects the transported meshes as deliverable annotations.

Four-anchor selected replay:

- target frames replayed: 14
- median silhouette IoU: 0.655
- median visible silhouette inside target mask: 0.889
- median z-buffer depth median: 2.00 mm
- median z-buffer depth p95: 21.68 mm
- p95 of z-buffer depth p95 values: 94.94 mm

Five-anchor all-frame replay:

- target frames replayed: 16
- median silhouette IoU: 0.811
- median visible silhouette inside target mask: 0.891
- median z-buffer depth median: 2.13 mm
- median z-buffer depth p95: 23.32 mm
- p95 of z-buffer depth p95 values: 90.22 mm

Selected visual stills confirm the mechanism. Frame 2535 overlays a doubled and shifted stem surface. Frame 2539 aligns a local patch but overdraws the target active-stem support, giving IoU 0.277 in the transported replay. Frame 2544 covers a narrow stem strip while the visible manipulated object includes a different surface. Frame 2550 has a millimeter median depth on overlap, yet its full silhouette still misses target-frame coverage. These results show that the pair factors track stable local material patches and do not determine the full visible object mesh.

## Current V6 State

V6 has added a real temporal smoothing signal: 16 graph-ready pair factors forming a continuous chain from 2534 to 2550, with strict inlier p95 below 10 mm. Those factors are suitable as sparse motion/deformation priors for a factor graph. V6 has also falsified direct mesh transport as an object annotation path, because full image replay fails on silhouette and tail depth even when nearest-surface residuals pass.

The graph solve keeps measured per-frame object meshes as the observation source and uses ready CoTracker factors as auxiliary constraints. A solved archive can only enter deliverables after the same replay suite passes: all-face z-buffer, mesh-surface contact, selected-contact SDF, full-hand SDF, and visual render inspection.

## Conservative Factor-Graph Solve

V6 then implemented a small correction graph over the multi-anchor ready-factor frames. The graph estimates state corrections for existing measured object meshes. It keeps the completed V4 measured mesh archive as the observation source and estimates one small SE3 correction per mesh-backed frame.

Nodes:

- one six-parameter correction for each selected object frame: rotation vector plus translation;
- selected frames are the endpoints of five-anchor ready CoTracker pairs: 2534 to 2550.

Edges:

- measurement edges keep sampled vertices close to their original measured positions;
- CoTracker factor edges bind source and target mesh vertices from sparse correspondence edge files, using the merged pair SE3 as the motion observation;
- smoothness edges penalize adjacent-frame correction jumps where frame indices are consecutive.

Objective:

```text
min_delta  rho(
  sum_f ||T_f(v_f) - v_f||^2 / sigma_obs^2
  + sum_(i,j,k) ||T_i(v_i,k) R_ij + t_ij - T_j(v_j,k)||^2 / sigma_factor^2
  + sum_(i,i+1) ||delta_i+1 - delta_i||^2 / sigma_smooth^2
)
```

`rho` is the soft-L1 loss used by `scipy.optimize.least_squares`. The measurement edge is intentionally strong because the measured mesh already passes image, contact, and penetration QC; the CoTracker factors regularize temporal state under the observed object geometry.

Artifacts:

- graph script: `scripts/fit_cotracker_factor_graph_v6.py`
- four-anchor graph report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_factor_graph_multianchor_2534_2550/qc_cotracker_factor_graph_v6.json`
- four-anchor graph mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_factor_graph_multianchor_2534_2550/cotracker_factor_graph_meshes_world.npz`
- five-anchor graph report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_factor_graph_multianchor5_2534_2550/qc_cotracker_factor_graph_v6.json`
- five-anchor graph mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_factor_graph_multianchor5_2534_2550/cotracker_factor_graph_meshes_world.npz`

Four-anchor result:

- status: `diagnostic_factor_compatible_no_material_correction`
- accepted graph pairs: 14
- accepted sparse correspondence edges: 435
- edge p95 median before solve: 6.64 mm
- edge p95 median after solve: 6.64 mm
- correction displacement p95 median: 0.003 mm
- maximum frame correction displacement p95: 0.009 mm

Five-anchor result:

- status: `diagnostic_factor_compatible_no_material_correction`
- selected frames: 2534 to 2550
- accepted graph pairs: 16
- accepted sparse correspondence edges: 462
- edge p95 median before solve: 6.64 mm
- edge p95 median after solve: 6.64 mm
- correction displacement p95 median: 0.003 mm
- maximum frame correction displacement p95: 0.011 mm

The graph result proves the current CoTracker factors are compatible with the measured geometry under a strong observation prior. The measured meshes already sit at the factor-compatible optimum within micron-scale corrections.

## Z-Buffer QC Hardening

The first replay of the graph archive exposed a QC bug. The graph and baseline mesh archives differed by only microns, yet the original triangle-only z-buffer reported silhouette collapse on frames 2538, 2543, and 2550. A single-frame reproduction showed the mechanism: dense sheet meshes contain projection-thin triangles, and tiny perturbations can make the triangle fill path skip large regions. Vertex-depth splatting over the same mesh vertices restored graph and baseline agreement on frame 2538.

`scripts/render_mesh_zbuffer_qc_v3.py` now combines triangle depth with a configurable vertex z-buffer. The default `--vertex-splat-radius-px 0` adds only the projected vertex pixels, preserving a tight silhouette measurement.

Patched graph-vs-baseline replay on the same 16 frames:

| Archive | Median IoU | Median visible inside mask | Median depth p95 |
| --- | ---: | ---: | ---: |
| completed V4 measured mesh | 0.965 | 0.977 | 4.22 mm |
| V6 graph-corrected mesh | 0.965 | 0.977 | 4.21 mm |

Per-frame deltas confirm that the graph archive preserves the measured mesh replay within measurement noise. The V6 graph archive is a factor-compatible diagnostic copy of the measured archive. The delivered geometry remains the completed V4/V5 measured mesh stream, with V6 factors attached as auxiliary temporal priors for future missing-frame or local-deformation solves.

The five-anchor graph archive was replayed over frames 2534 to 2550 with the same hardened all-face z-buffer QC:

- replay frames: 17
- median silhouette IoU: 0.965
- median visible silhouette inside mask: 0.976
- median z-buffer depth p95: 3.96 mm
- p95 of z-buffer depth p95 values: 12.09 mm

Visual spot checks of the graph replay show frame 2540 has a clean long active-stem surface, while the pre-repair frame 2539 measured surface remains narrow and ambiguous with IoU 0.536. The graph preserves the measured evidence. Frame-2539 repair therefore belongs to the perception and mesh-reconstruction layer, while the factor layer contributes sparse temporal priors that are continuous and metric-compatible.

## Frame-2539 Perception Repair Attempt

V6 then tested whether the remaining ambiguous frame 2539 could be repaired by stronger model-produced mask evidence. The repair target was the one continuous active stem held between the hands. The rejected alternatives were not used as replacement geometry.

Artifacts:

- strict VLM repair points: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_points_strict_vlm/visual_track_point_prompts_vlm.json`
- strict box-conditioned SAM2 candidates: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_strict/qc_sam2_image_points.json`
- strict no-box SAM2 candidates: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_strict_nobox/qc_sam2_image_points.json`
- propagated seed branch: `/data2/ego_annotation_outputs/representative_wild_rice/v6_sam2_gap_seed2538_2538_2540/qc_sam2_mask_seed_track_v4.json`

Results:

- SAM2 seed propagation from frame 2538 visually merged two long stem surfaces. Its observed surface extent was 0.161 x 0.454 x 0.250 m, and 1 mm sheet solidification failed the 0.120 sheet PCA threshold with ratio 0.150.
- The existing VLM-selected candidate 0 for frame 2539 was semantically closer but still depth-mixed: observed extent 0.118 x 0.321 x 0.239 m, sheet PCA ratio 0.168 under the stricter repair contract.
- The stricter frame-local VLM prompt placed positive points along the held active stem and negative points on the adjacent parallel strip, both hands, background stems, countertop glare, and basket/background. Box-conditioned SAM2 produced no candidate satisfying all positives and zero negatives.
- Among box-conditioned raw candidates, candidate 1 had strong image-depth replay after forced geometry testing, with silhouette IoU 0.883 and z-buffer p95 1.71 mm, but it hit the semantic negative point and visually merged the adjacent strip. Candidates 0 and 2 avoided negatives but were partial fragments with IoU about 0.32.
- No-box SAM2 selected a mask that satisfied the sparse point contract, but geometry falsified it: observed extent 0.277 x 0.402 x 0.243 m and sheet PCA ratio 0.326. The point contract missed a horizontal unrelated stem leak.

The first VLM verifier iteration rejected the no-box mask with 0.92 confidence. Its correction points were initially appended to the stale positives, and that produced no accepted SAM2 candidate. The failure mechanism was contradictory positive evidence: the verifier had changed the target support, while the old positives still forced SAM2 toward merged same-category surfaces.

The accepted iteration used the verifier positives as replacement positives and retained the semantic negatives. Box-conditioned SAM2 then selected candidate 0 for frame 2539. The candidate passed the point contract and the independent image-depth replay:

- replacement prompt artifact: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_mask_verifier_iter1_replacement_points/visual_track_point_prompts_vlm.json`
- accepted SAM2 image run: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_replace_iter1_box/qc_sam2_image_points.json`
- selected one-frame dataset: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_replace_iter1_box_selected_dataset/manifest.json`
- selected one-frame solidified mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_replace_iter1_box_selected_solidified_thick001/solidified_sheet_object_meshes_world.npz`
- selected one-frame z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v6_frame2539_repair_sam2_replace_iter1_box_selected_zbuffer_qc/qc_mesh_zbuffer_projection_v3.json`

Frame-2539 accepted metrics:

- silhouette IoU: 0.948
- visible silhouette inside mask: 0.960
- z-buffer absolute median: 0.24 mm
- z-buffer absolute p95: 4.47 mm
- mask area: 23,712 px
- mesh silhouette area: 24,355 px

## V6 Repaired 31-Frame Archive

The accepted frame-2539 mesh was assembled into the completed wild-rice track while preserving every other frame from the completed V4 measured stream.

Artifacts:

- assembled manifest: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/manifest.json`
- assembled mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz`
- all-frame z-buffer QC: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_zbuffer_qc_2520_2550/qc_mesh_zbuffer_projection_v3.json`
- exact-threshold contact QC: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/mesh_surface_contact_recomputed_det015_exact_v4thresholds.json`
- contact-frame mesh identity proof: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/qc_contact_frame_mesh_identity_v6.json`
- selected-contact SDF report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/volume_sdf_contact_recomputed_det015_exact_v4thresholds_pitch001_qc_composed.json`
- full-hand SDF report: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/full_hand_sdf_penetration_recomputed_det015_exact_v4thresholds_pitch001_qc_composed.json`
- V6 state package: `/data2/ego_annotation_outputs/representative_wild_rice/v6_repaired_track_state_2520_2550/v6_repaired_track_state.json`

State counts:

- measured mesh geometry: 29 frames
- verified VLM/SAM2 repaired geometry: 1 frame, frame 2539
- completed tracked geometry: 1 frame, frame 2550

All-frame image-depth replay over frames 2520 to 2550:

- frames: 31
- median silhouette IoU: 0.967
- p05 silhouette IoU: 0.919
- median visible silhouette inside mask: 0.979
- median z-buffer absolute median: 0.49 mm
- median z-buffer absolute p95: 4.47 mm
- p95 of z-buffer absolute p95 values: 12.16 mm
- max z-buffer absolute p95: 13.16 mm

The physical-contact report was recomputed under the same thresholds as the accepted V4 evidence path: detector score at least 0.15, median reprojection at most 18 px, absolute hand-depth bias at most 20 mm, contact patch distance p95 at most 25 mm, signed gap p95 at most 25 mm, and temporal patch gap at most 6 frames. The repaired archive preserves the V4 contact evidence:

- reliable temporal contact rows: 15
- reliable geometry contact rows: 17
- geometry-backed observation rows: 22
- reliable-row median contact patch p95: 0.70 mm
- reliable-row p95 contact patch p95: 4.66 mm
- reliable-row median signed-gap p95 absolute: 0.58 mm
- reliable-row p95 signed-gap p95 absolute: 3.91 mm
- reliable-row penetration fraction: 0 percent

Frame 2539 has no reliable contact row. The SDF reports therefore use a machine-checked composition proof over unchanged contact-frame meshes. `qc_contact_frame_mesh_identity_v6.json` proves that the assembled archive changes only frame 2539 and changes zero reliable-contact frames. The selected-contact and full-hand SDF values are consequently unchanged from the source V4 measured SDF reports:

- selected-contact penetration fraction: 0 percent
- selected-contact absolute SDF median: 0.96 mm
- selected-contact absolute SDF p95: 3.45 mm
- full-hand penetration fraction: 0 percent
- full-hand signed SDF median: 14.30 mm
- full-hand signed SDF p05: 1.48 mm

## V6 Deliverables

V6 renders the repaired 31-frame archive through the stakeholder-grade world renderer introduced in V5. The right panel is a shaded metric manipulation view with the object mesh, MANO surfaces, contact points, metric scale, and a separate head-camera trajectory inset.

Artifacts:

- overlay video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_mesh_surface_contact_review_repaired2539_2520_2550/mesh_surface_contact_review.mp4`
- side-by-side annotated video plus 3D reconstruction: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4`
- standalone 3D world animation: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_3d.mp4`
- render manifest: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/render_manifest.json`
- visual QC contact sheet for frames 2538 to 2540: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/qc_contact_sheet_2538_2540.jpg`

Structural QC:

- overlay video: 1280 x 720, 31 frames, 6 fps
- side-by-side video: 1920 x 778, 31 frames, 6 fps
- standalone 3D video: 960 x 720, 31 frames, 6 fps

Visual inspection of the contact sheet accepts the repaired frame 2539 presentation. The overlay shows a narrow active-stem mask after removing the previous broad merged two-stem surface. The 3D panel shows the object mesh, MANO hand surfaces, contact points, metric scale, and head-camera inset as readable geometry.

## V6 Status

V6 closes two concrete gaps in the previous state. First, the sparse CoTracker factor chain now spans frames 2534 to 2550 and is compatible with the measured geometry under the conservative graph objective. Second, frame 2539 has a verified VLM/SAM2 perception repair and a reconstructed object mesh that passes one-frame z-buffer replay and full-archive replay.

The delivered V6 object geometry remains a per-frame reconstructed mesh stream. The current evidence still rejects direct full-mesh transport and rigid canonical-map replacement, because those hypotheses fail image replay. The sparse factors are temporal priors for future missing-frame and local-deformation solves; the delivered mesh surfaces continue to come from model-produced masks, UniDepth, camera pose, and 1 mm mesh reconstruction.
