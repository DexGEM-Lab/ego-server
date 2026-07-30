# Pipeline V12: Contact Dynamics Factor Graph

## Starting Point

V11 closed the mesh-completion problem for a contact-rich trash window: measured visible geometry, temporally fused hidden geometry, MANO contact, nonpenetration, replay, track QC, and stakeholder renders agree on frames 865 to 870. V11 still treats contact as geometry. It verifies that a hand patch lies on the object surface without penetrating it, but it does not explain the contact as a temporal physical relation.

V12 tests this question:

```text
Can the accepted V11 contact rows, object motion factors, MANO patch centers, and object surface points admit a contact-dynamics explanation while preserving measured contact anchors?
```

## Graph

`scripts/solve_contact_dynamics_factor_graph_v12.py` builds one short-window factor graph from existing model evidence. It consumes only data arrays already produced by perception and geometry modules:

```text
C_t       object contact point in world coordinates
G_t       vector from object contact point to MANO contact patch center
E_tu      object motion factor from frame t to u
H_t       observed MANO contact patch center
O_t       observed nearest object surface point
```

Nodes:

- one object contact point `C_t` per selected contact frame;
- one hand contact gap `G_t` per selected contact frame.

Edges:

- object-anchor factor: `C_t` must stay near measured object surface point `O_t`;
- hand-anchor factor: `C_t + G_t` must stay near measured MANO contact patch center `H_t`;
- contact-gap factor: `G_t` should stay near zero under active contact;
- object-motion factor: transported `C_t` through `E_tu` explains `C_u` up to slip;
- relative-contact factor: the hand patch and contact point maintain the same local relative motion;
- slip-prior factor: slip is allowed but measured;
- acceleration-consistency factor: hand patch acceleration and object contact-point acceleration should agree when the patch remains in contact.

The graph has no object-category branches. Contact rows are selected from `geometry_backed_temporal_contact == true`; the unreliable predicted left-hand rows stay outside the graph.

## Objective

The solver minimizes a robust least-squares objective:

```text
min_X
  sum_t w_o ||C_t - O_t||
+ sum_t w_h ||C_t + G_t - H_t||
+ sum_t w_c ||G_t||
+ sum_(t,u) w_m ||C_u - E_tu(C_t)||
+ sum_(t,u) w_r ||((C_u + G_u) - E_tu(C_t + G_t)) - (C_u - E_tu(C_t))||
+ sum_(t,u) w_s ||C_u - E_tu(C_t)||
+ sum_(t,u,v) w_a ||acc_hand - acc_object||
```

The object-motion residual is interpreted as surface slip, because a human hand can slide across the manipulated surface while remaining in contact. Sticking is a special case where the slip p95 is below the sticking threshold. V12 therefore reports the contact motion regime instead of forcing all valid contact to be no-slip.

The report contract requires at least three contact frames and at least one acceleration-consistency row. A two-frame crop can still be useful as a contact-distance diagnostic, but it cannot accept a dynamics claim.

## Trash 865-870 Result

Output:

```text
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/qc_contact_dynamics_factor_graph_v12.json
```

Inputs:

```text
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/physics_qc/annotations_with_manifest_object_masks.json
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/observed_plus_temporal_fused_hidden_meshes_world.npz
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_865_870_filtered/physics_qc/mesh_surface_contact_qc.json
/data2/ego_annotation_outputs/v11_temporal_fused_hidden/trash_partcrafter_pair_factors_p95_011/qc_cotracker_pairwise_rigid_factors_v6.json
```

Evidence:

- status: accepted;
- contact motion regime: sliding;
- evidence predicates: 6 contact frames and 4 acceleration-consistency rows;
- graph nodes: 6 object contact points and 6 hand contact gaps;
- graph edges: 6 object anchors, 6 hand anchors, 6 contact gaps, 5 object-motion edges, 5 relative-contact edges, 5 slip priors, and 4 acceleration-consistency edges;
- contact gap p95: 0.77 mm;
- relative-contact residual p95: 1.04 mm;
- slip speed p95: 0.256 m/s;
- object factor inlier p95 summary: p95 10.03 mm, max 10.34 mm;
- acceleration-consistency residual p95: 0.0557 m/s^2;
- object anchor shift p95: 0.46 mm;
- hand anchor shift p95: 0.18 mm;
- contact mode segments: middle-finger contact on frames 865 to 869, then palm contact on frame 870.

The accepted interpretation is sliding contact. The hand remains geometrically and dynamically tied to the object surface, while the active patch moves across the surface. The graph does not claim a single material point is stuck to the hand across the whole window.

Rendered deliverables:

```text
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/deliverables_dynamics_final/world/world_reconstruction_side_by_side.mp4
```

Visual inspection sheets:

```text
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/visual_inspection_dynamics_final/overlay_sheet.jpg
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/visual_inspection_dynamics_final/world_sheet.jpg
/data2/ego_annotation_outputs/v12_contact_dynamics/trash_865_870/visual_inspection_dynamics_final/side_by_side_sheet.jpg
```

The V12 world panel overlays the contact dynamics state without replacing the geometry: contact regime, active patch, gap, slip speed, and dynamics residual appear beside the object and hand reconstruction. The final side-by-side video keeps the V12 caption label and the semantic caption fits within the caption band.

## Relation To V11

V12 adds a physical consistency layer on top of the V11 accepted geometry. It keeps the V11 mesh, MANO, contact rows, and object motion factors as evidence. It does not replace object mesh reconstruction, object pose, hand pose, replay, hidden-surface QC, SDF contact, or rendered deliverables.

V13 should test the dynamics graph on additional contact-rich windows and use the graph output to refine MANO/object state, then rerun replay, hidden-surface QC, SDF contact, nonpenetration, dynamics QC, and rendered deliverables.
