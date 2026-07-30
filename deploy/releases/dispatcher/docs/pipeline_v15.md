# Pipeline V15: Contact-Switch Surface Dynamics

## Purpose

V14 tested a same-material-point handoff: the object point touched before a finger change was transported through the object motion factor and compared with the object point touched after the finger change. Box-books exposed the missing physical distinction. A dexterous finger switch can move from one object surface point to another while the object itself moves coherently.

V15 tests that contact-switch claim directly. The solver keeps the contact-gap and continuous sliding factors from V12 to V14, then evaluates contact-mode transitions with local object-surface transport across the two contact neighborhoods.

## Graph

`scripts/solve_contact_switch_surface_factor_graph_v15.py` uses:

```text
C_t       object contact point in world coordinates
G_t       vector from object contact point to MANO contact patch center
E_tu      object motion factor from frame t to u
S_t(C)    local object surface neighborhood around contact point C
H_t       observed MANO contact patch center
O_t       observed nearest object surface point
M_t       contact mode key from the contact evidence row
```

Nodes:

- one object contact point `C_t` for each surface-supported contact row;
- one hand contact gap `G_t` for each surface-supported contact row.

Optimized edges:

- object-anchor factor: `C_t` stays near `O_t`;
- hand-anchor factor: `C_t + G_t` stays near `H_t`;
- contact-gap factor: `G_t` stays near zero;
- continuous object-motion, relative-contact, and slip-prior factors when adjacent rows share `M_t`;
- switch gap-continuity factor when adjacent rows change contact mode.

Surface-switch QC:

- source neighborhood: transport `S_t(C_t)` through `E_tu` and measure nearest distance to the target-frame object surface;
- target preimage neighborhood: inverse-transport `S_u(C_u)` through `E_tu` and measure nearest distance to the source-frame object surface;
- source contact center and target contact center receive the same surface-distance checks.

The switch evidence tests one moving object surface while allowing the contact point to change.

## Objective

V15 solves:

```text
min_X
  sum_t w_o ||C_t - O_t||
+ sum_t w_h ||C_t + G_t - H_t||
+ sum_t w_c ||G_t||
+ sum_(t,u: M_t=M_u) w_m ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t=M_u) w_r ||((C_u + G_u) - E_tu(C_t + G_t)) - (C_u - E_tu(C_t))||
+ sum_(t,u: M_t=M_u) w_s ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t!=M_u) w_g ||G_u - G_t||
```

Acceptance requires:

- at least three surface-supported contact rows;
- at least one switch edge;
- at least one continuous edge in a neighboring contact-mode segment;
- each switch has a neighboring contact-mode segment with at least two frames;
- contact gap p95 at most 6 mm;
- continuous relative-contact p95 at most 2 mm;
- continuous slip p95 at most 0.45 m/s;
- switch gap-delta p95 at most 6 mm;
- source-neighborhood and target-preimage surface transport p95 at most 12 mm;
- source and target contact-center surface distance at most 8 mm;
- pair-factor p95 at most 12 mm;
- object and hand anchor p95 at most 6 mm;
- solver convergence.

## Accepted Box-Books 614-616

Report:

```text
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/qc_contact_switch_surface_factor_graph_v15.json
```

Status: accepted.

Evidence:

- kept rows: 614 pinky, 615 middle, 616 middle;
- switch edge: pinky frame 614 to middle frame 615;
- continuous edge: middle frame 615 to middle frame 616;
- contact gap p95: 1.83 mm;
- relative-contact residual: 0.87 mm;
- continuous slip: 0.326 m/s;
- switch gap delta: 1.91 mm;
- switch source-neighborhood transport p95: 8.36 mm;
- switch target-preimage transport p95: 4.42 mm;
- transported source contact-center surface distance: 6.46 mm;
- target contact-center preimage surface distance: 0.50 mm;
- object factor p95 summary: p95 11.56 mm, max 11.57 mm.

Rendered deliverables:

```text
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

Structural QC:

- overlay: 1280 by 720, 7 frames, 6 fps;
- world: 960 by 720, 7 frames, 6 fps;
- side-by-side: 1920 by 778, 7 frames, 6 fps.

Visual inspection sheets:

```text
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/visual_inspection_switch_final/overlay_sheet.jpg
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/visual_inspection_switch_final/world_sheet.jpg
/data2/ego_annotation_outputs/v15_contact_switch/box_books_probe_614_616/visual_inspection_switch_final/side_by_side_sheet.jpg
```

Inspection result: the overlay places MANO on the visible right hand and the object mesh on the box/books object. The world and side-by-side views show a switch label at frame 614 and sliding labels at frames 615 and 616. The caption fits, and the 3D panel shows object mesh, MANO surface, head camera, head trajectory, axes, and scale.

## Accepted Trash Control

Report:

```text
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/qc_contact_switch_surface_factor_graph_v15.json
```

Status: accepted.

Evidence:

- continuous middle-contact segment: frames 865 to 869;
- switch edge: frame 869 middle to frame 870 palm;
- contact gap p95: 0.77 mm;
- relative-contact residual p95: 1.06 mm;
- continuous slip p95: 0.265 m/s;
- switch gap delta: 0.304 mm;
- switch source-neighborhood transport p95: 6.17 mm;
- switch target-preimage transport p95: 5.90 mm;
- source and target contact-center surface distances: 3.11 mm and 2.99 mm;
- object factor p95 summary: p95 10.03 mm, max 10.34 mm.

Rendered deliverables:

```text
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/deliverables_switch_final/world/world_reconstruction_side_by_side.mp4
```

Visual inspection sheets:

```text
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/visual_inspection_switch_final/overlay_sheet.jpg
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/visual_inspection_switch_final/world_sheet.jpg
/data2/ego_annotation_outputs/v15_contact_switch/trash_865_870/visual_inspection_switch_final/side_by_side_sheet.jpg
```

Inspection result: the world and side-by-side views show sliding labels through frame 868, a switch label at frame 869, and contact at frame 870. The 3D panel remains readable with MANO, object mesh, observed/completed surface legend, head camera, head trajectory, axes, and scale.

## Interpretation

V15 tests a different physical claim from V14. V14 tests same-material-point handoff. V15 tests contact switching, where the hand changes to another surface point while object motion stays coherent. The box-books failure in V14 becomes an accepted V15 contact-switch case because both local surface neighborhoods survive the learned object motion factor and both contact centers remain near the corresponding object surfaces.
