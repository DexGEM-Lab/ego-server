# Pipeline V13: Contact-Mode Dynamics

## Starting Point

V12 models every selected contact row as one temporal contact trajectory. That is valid for the V11 trash window because frames 865 to 869 share middle-finger contact and frame 870 is a final palm transition. The box-books branch exposed the missing distinction: a window can have good MANO/object surface distance while the active hand patch changes from one finger to another. Treating that transfer as one continuous contact creates a false slip measurement.

V13 tests contact dynamics inside contact-mode-continuous segments. A contact mode is defined by hand index, side, selected patch source, selected patch region, and selected patch anchor joint. Adjacent rows with different contact modes become transition records. They do not create object-motion, relative-contact, slip, or acceleration factors.

## Graph

`scripts/solve_contact_mode_dynamics_factor_graph_v13.py` reuses the V12 observation model:

```text
C_t       object contact point in world coordinates
G_t       vector from object contact point to MANO contact patch center
E_tu      object motion factor from frame t to u
H_t       observed MANO contact patch center
O_t       observed nearest object surface point
M_t       contact mode key from the contact evidence row
```

Nodes:

- one object contact point `C_t` per selected contact frame;
- one hand contact gap `G_t` per selected contact frame.

Edges:

- object-anchor factor: `C_t` stays near `O_t`;
- hand-anchor factor: `C_t + G_t` stays near `H_t`;
- contact-gap factor: `G_t` stays near zero;
- object-motion, relative-contact, slip-prior, and acceleration-consistency factors only when adjacent frames have the same `M_t`;
- transition rows when adjacent frames have different `M_t`.

This keeps the downstream graph category-agnostic. Model-produced contact evidence represents the discontinuity; object class and action labels stay outside the dynamics state.

## Objective

For continuous contact edges, V13 minimizes the same robust objective as V12:

```text
min_X
  sum_t w_o ||C_t - O_t||
+ sum_t w_h ||C_t + G_t - H_t||
+ sum_t w_c ||G_t||
+ sum_(t,u: M_t=M_u) w_m ||C_u - E_tu(C_t)||
+ sum_(t,u: M_t=M_u) w_r ||((C_u + G_u) - E_tu(C_t + G_t)) - (C_u - E_tu(C_t))||
+ sum_(t,u: M_t=M_u) w_s ||C_u - E_tu(C_t)||
+ sum_(t,u,v: M_t=M_u=M_v) w_a ||acc_hand - acc_object||
```

V13 accepts when the window has at least three contact rows, at least one continuous contact segment with three frames, at least one temporal edge, and at least one acceleration row. This evidence contract ties acceptance to a measured temporal dynamics signal.

The solver also filters selected contact observations by measured surface support before graph construction. The default predicate keeps rows whose selected MANO patch has object-surface distance p95 at most 6 mm and records rejected rows in `input_rejected_observations`.

## Trash 865-870 Control

Report:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/qc_contact_mode_dynamics_factor_graph_v13.json
```

After adding the input surface-support predicate, the no-regression report is:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/trash_865_870/qc_contact_mode_dynamics_factor_graph_v13_surface_filtered.json
```

Status: accepted.

Evidence:

- graph nodes: 6 object contact points and 6 hand contact gaps;
- graph edges: 4 object-motion edges, 4 relative-contact edges, 4 slip priors, 3 acceleration-consistency edges, and 1 reported contact-mode transition;
- continuous segment: middle-finger contact on frames 865 to 869;
- transition: middle-finger contact on frame 869 to palm contact on frame 870;
- contact gap p95: 0.77 mm;
- relative-contact residual p95: 1.06 mm;
- slip speed p95: 0.265 m/s;
- object factor inlier p95 summary: p95 9.63 mm, max 10.34 mm;
- acceleration-consistency residual p95: 0.0569 m/s^2;
- object anchor shift p95: 0.46 mm;
- hand anchor shift p95: 0.18 mm.

Rendered deliverables:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/overlay/mesh_surface_contact_review.mp4
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/world/world_reconstruction_3d.mp4
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/deliverables_mode_dynamics_final/world/world_reconstruction_side_by_side.mp4
```

Structural QC:

- overlay: 1280 by 720, 6 frames, 6 fps;
- world: 960 by 720, 6 frames, 6 fps;
- side-by-side: 1920 by 778, 6 frames, 6 fps.

Visual inspection sheets:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/visual_inspection_mode_dynamics_final/overlay_sheet.jpg
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/visual_inspection_mode_dynamics_final/world_sheet.jpg
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/trash_865_870/visual_inspection_mode_dynamics_final/side_by_side_sheet.jpg
```

The sheets show coherent MANO/object overlay, readable 3D world layout, observed/completed surface separation, head camera cue, scale, semantic caption, and contact-mode dynamics badge.

## Box-Books Generalization Diagnostic

The V7 accepted crop 616 to 618 remains a negative dynamics diagnostic:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/box_books_616_618/qc_contact_dynamics_factor_graph_v12.json
```

It has two contact rows, zero acceleration rows, and slip speed p95 0.761 m/s across the single 616 to 617 edge. Contact gap and relative residual pass; temporal support and slip speed fail the dynamics claim.

The first V8 repaired-branch diagnostic used a stale contact identity from the source contact report:

```text
/data2/ego_annotation_outputs/v13_contact_dynamics_generalization/box_books_v8_614_616/qc_contact_mode_dynamics_factor_graph_v13_p95_012.json
```

That report labeled frame 615 as pinky contact while the selected anatomical patch vertices and anchor were middle-finger evidence. V14 fixed the source contact diagnostic so geometry-backed sliding support updates the selected patch identity from the candidate that earned support:

```text
/data2/ego_annotation_outputs/v14_contact_transfer/box_books_probe_612_618_fixed_contact_identity/mesh_surface_contact_qc.json
/data2/ego_annotation_outputs/v14_contact_transfer/box_books_probe_612_618_fixed_contact_identity/qc_contact_mode_dynamics_factor_graph_v13_surface_filtered_p95_012.json
```

The corrected branch rejects under V13:

- kept rows: 614 pinky, 615 middle, 616 middle;
- rejected rows by input surface support: frame 613 middle at 17.4 mm p95 and frame 617 pinky at 35.2 mm p95;
- longest continuous contact mode: 2 frames, middle contact on 615 to 616;
- acceleration rows: 0 after contact-mode segmentation;
- transition: pinky anchor joint 20 on frame 614 to middle anchor joint 12 on frame 615;
- continuous middle edge slip speed: 0.327 m/s;
- contact gap p95: 1.86 mm;
- relative-contact residual: 0.70 mm;
- object and hand anchor shifts: under 1 mm;
- object motion factors required a marginal 12 mm p95 diagnostic report.

This rejection matches the corrected evidence. The branch has a plausible transfer from pinky to middle contact, followed by only two middle-contact frames. V14 should model contact transfer explicitly and decide whether a handoff plus short post-transfer segment is sufficient evidence for annotation, while keeping acceleration consistency reserved for segments with at least three frames.
