# Pipeline V14: Contact Handoff Dynamics

## Purpose

V13 made contact dynamics mode-aware: it evaluates temporal factors only inside a stable contact mode and records transfers between modes. That protected the solver from treating a finger change as one continuous contact, but it left short transfers unresolved. V14 adds an explicit handoff factor graph for windows where the contact patch changes and the adjacent stable segment is too short for acceleration evidence.

The graph consumes model-produced evidence only: MANO patch rows, mesh surface distances, and open-vocabulary object tracks encoded as pairwise rigid factors. Object class, color, material, and action phrase stay outside the solver state.

## Graph

`scripts/solve_contact_handoff_factor_graph_v14.py` uses the same state variables as V12 and V13:

```text
C_t       object contact point in world coordinates
G_t       vector from object contact point to MANO contact patch center
E_tu      object motion factor from frame t to u
H_t       observed MANO contact patch center
O_t       observed nearest object surface point
M_t       contact mode key from the contact evidence row
```

Nodes:

- one object contact point `C_t` for each surface-supported contact row;
- one hand contact gap `G_t` for each surface-supported contact row.

Edges:

- object-anchor factor: `C_t` stays near `O_t`;
- hand-anchor factor: `C_t + G_t` stays near `H_t`;
- contact-gap factor: `G_t` stays near zero;
- continuous object-motion, relative-contact, and slip-prior factors when adjacent rows share the same contact mode `M_t`;
- handoff object-motion and handoff surface-continuity factors when adjacent rows have different contact modes.

The handoff edge tests whether the object point at the target contact can be explained by transporting the source contact point through the measured object motion factor. Acceleration consistency is reserved for segments with three or more frames, because a two-frame transfer lacks an acceleration signal.

## Objective

V14 solves a robust least-squares problem:

```text
min_X
  sum_t w_o ||C_t - O_t||
+ sum_t w_h ||C_t + G_t - H_t||
+ sum_t w_c ||G_t||
+ sum_(t,u: M_t=M_u) w_m ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t=M_u) w_r ||((C_u + G_u) - E_tu(C_t + G_t)) - (C_u - E_tu(C_t))||
+ sum_(t,u: M_t=M_u) w_s ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t!=M_u) w_ho ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t!=M_u) w_hg ||G_u - G_t||
```

Acceptance requires:

- at least three surface-supported contact rows;
- at least one handoff edge;
- at least one continuous edge in a neighboring contact-mode segment;
- each handoff has a neighboring contact-mode segment with at least two frames;
- contact gap p95 at most 6 mm;
- continuous relative-contact p95 at most 2 mm;
- continuous slip p95 at most 0.45 m/s;
- handoff object-motion speed p95 at most 0.45 m/s;
- handoff gap-delta p95 at most 6 mm;
- pair-factor p95 at most 12 mm;
- object and hand anchor p95 at most 6 mm;
- solver convergence.

The adjacent-segment predicate is deliberate. A handoff can occur at the end of a clip, so requiring a two-frame post-handoff segment would falsely reject a measured transition whose source segment and handoff edge are both observed.

## Accepted Control: Trash 865-870

Report:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/qc_contact_handoff_factor_graph_v14.json
```

Status: accepted.

Evidence:

- graph nodes: 6 object contact points and 6 hand contact gaps;
- continuous edges: 4, all within middle-finger contact on frames 865 to 869;
- handoff edge: middle-finger frame 869 to palm frame 870;
- adjacent segment support: 5 frames;
- contact gap p95: 0.77 mm;
- relative-contact residual p95: 1.06 mm;
- continuous slip p95: 0.265 m/s;
- handoff object-motion residual: 5.06 mm;
- handoff object-motion speed: 0.030 m/s;
- handoff gap delta: 0.285 mm;
- object factor inlier p95 summary: p95 10.03 mm, max 10.34 mm.

Rendered deliverables:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/deliverables_handoff_final/world/world_reconstruction_side_by_side.mp4
```

Structural QC:

- overlay: 1280 by 720, 6 frames, 6 fps;
- world: 960 by 720, 6 frames, 6 fps;
- side-by-side: 1920 by 778, 6 frames, 6 fps.

Visual inspection sheets:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/visual_inspection_handoff_final/overlay_sheet.jpg
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/visual_inspection_handoff_final/world_sheet.jpg
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/visual_inspection_handoff_final/side_by_side_sheet.jpg
```

Inspection result: the object mesh sits on the trash-lid surface, MANO tracks the visible right hand, the 3D view shows object mesh, MANO, head camera, head trajectory, axes, scale, semantic caption, and observed/completed surface legend. The dynamics badge labels frames 865 to 868 as sliding, frame 869 as handoff, and frame 870 as contact.

## Rejected Diagnostic: Box-Books 614-616

Report:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/box_books_probe_612_618_fixed_contact_identity/qc_contact_handoff_factor_graph_v14.json
```

Status: rejected.

Evidence:

- kept rows after surface-support filtering: 614 pinky, 615 middle, 616 middle;
- rejected rows: frame 613 at 17.4 mm p95 and frame 617 at 35.2 mm p95 surface distance;
- handoff edge: pinky frame 614 to middle frame 615;
- adjacent segment support: 2 frames;
- contact gap p95: 1.63 mm;
- relative-contact residual: 0.87 mm;
- continuous middle-edge slip: 0.325 m/s;
- handoff gap delta: 1.51 mm;
- object factor p95 summary: p95 11.56 mm, max 11.57 mm;
- handoff object-motion residual: 92.7 mm;
- handoff object-motion speed: 0.556 m/s, above the 0.45 m/s threshold.

The rejection localizes the failure of the same-material-point handoff claim. Contact identity, surface gap, hand gap continuity, and the following 615 to 616 middle-contact edge pass their current tests. V15 tests the separate physical claim that the hand changed contact point on one coherently moving object surface.
