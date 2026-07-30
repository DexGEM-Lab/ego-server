# V19 HOT3D fixed-slice v1 results

## Scope

This records the first bounded open-source V19 result for Workbench item 6. The fixed slice was selected before tuning beyond the initial HOT3D gate: `clip-001849`, `clip-001850`, and `clip-001851`. Each clip was run through the Pi-runtime prediction path, frozen before evaluator access, rendered as full-duration overlay/world/side-by-side videos, and scored only after freeze.

The metric scope is HOT3D 21-joint MANO in camera coordinates. It does not score object pose, contact, occlusion, nonpenetration, or full MANO vertices.

## Physical claim

The support-gated V19 interval mechanism prevents unsupported global wrist/root translation. On all three clips, selected visible-surface depth-order support count is zero for every hand/frame row, and the output translation gate preserves the runtime HaWoR wrist/root state.

This is not a general MANO-accuracy win. The remaining wrist-relative articulation changes are small and mixed across clips.

## Runtime-HaWoR baseline vs V19 support-gated interval

Negative deltas mean V19 has lower error than the runtime HaWoR baseline.

| clip | rows | wrist median delta (m) | joint MPJPE median delta (m) | root-aligned MPJPE median delta (m) | mechanism interpretation |
|---|---:|---:|---:|---:|---|
| `clip-001850` | 300 | `0.000000000` | `-0.001136926` | `+0.000304392` | Wrist/root safety held; articulation slightly helped absolute MPJPE but slightly hurt root-aligned MPJPE. |
| `clip-001849` | 300 | `0.000000000` | `+0.000851875` | `-0.001110242` | Wrist/root safety held; articulation hurt absolute MPJPE but helped root-aligned MPJPE. |
| `clip-001851` | 300 | `0.000000000` | `-0.000999961` | `+0.000852540` | Wrist/root safety held; articulation slightly helped absolute MPJPE but hurt root-aligned MPJPE. |

Artifact roots:

- `clip-001850`: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260626_hot3d_clip001850_pinhole_a800_native_v5_focalfix_coordrigid_v1`
- `clip-001849`: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001849_pinhole_a800_native_v1_supportgate`
- `clip-001851`: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001851_pinhole_a800_native_v1_supportgate`

Freeze manifest hashes:

- `clip-001850`: `06c8ff45a955421b341ceafb0548b27d2d791e9ea6495f1fc6d17ee4db67c594`
- `clip-001849`: `b7a82b2add62ca268f136982d76b5cbdc227cc7c52d4157b42d314242a3b3bea`
- `clip-001851`: `faacbf86eb890ebac022e97bfdf34b98175c7acd48487cab3ff472d12f7e605f`

Comparison report hashes:

- `clip-001850`: `evaluation/hot3d_mano3d_owlv2_support_gated_runtime/hot3d_baseline_vs_support_gated_runtime_comparison.json`, SHA256 `c35c49dc966ffb24a77b78b0d54c926ad21ba46bd90199b83c1af16367df45b5`.
- `clip-001849`: `evaluation/hot3d_mano3d_support_gated_runtime/hot3d_runtime_baseline_vs_support_gated_runtime_comparison.json`, SHA256 `15578bdb36e5dc8c0109f5fa7ca119ab18dc7173a7fa1cfc32352e2622988066`.
- `clip-001851`: `evaluation/hot3d_mano3d_support_gated_runtime/hot3d_runtime_baseline_vs_support_gated_runtime_comparison.json`, SHA256 `2663a1f863f10313b231ad2be930bb518d27eebcaedcbc081f4ad25cc8f7be69`.

## Failure clusters

1. **No positive visible-surface support.** All 900 hand/frame rows have zero selected visible-surface depth-order support and zero finite MANO-vertex/visible-object-depth overlap. The gate is therefore operating only as a safety boundary, not as a positive correction mechanism.
2. **Occluded-contact support is not visible under the MANO projection.** The exact visible-object-mask overlap criterion is physically too narrow for hands over a keyboard: the hand-owned pixels occlude the touched keys, so useful rigid surface support is adjacent or latent, not visible directly below projected MANO vertices.
3. **Broad contact priors still create raw unsupported shifts.** The raw optimizer attempts global wrist/root shifts up to centimeters even without selected depth-order support. The output gate suppresses those shifts, but wrist-relative articulation can still change.
4. **Contact/nonpenetration remains uncertain.** The current completed/sign meshes are non-watertight and the rendered world views show visible hand-keyboard gaps, so contact closure is not supported.
5. **Runtime is too slow for the default design target.** P18 takes tens of minutes on 5-second clips, so runtime remains a design defect even though the physics claim is bounded.

## Workbench item 7 starting hypothesis

The next controlled autoresearch question is whether unsupported latent global translation contaminates articulation before the output wrist/root gate is applied.

Prediction:

- If in-solver translation freeze materially changes root-aligned/articulation metrics while wrist remains unchanged, the current output gate is only a partial physical repair because latent unsupported translation affects the root/pose solution.
- If in-solver translation freeze leaves the metrics essentially unchanged, the mixed articulation effects come from root/pose/contact terms rather than latent translation, and the next mechanism should be nearby/latent visible-surface support from adjacent rigid object geometry.

The first item-7 ablation tested this on `clip-001851` under:

`/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001851_pinhole_a800_native_v1_supportgate/evaluation/autoresearch/in_solver_translation_freeze_v1/`

It used the same frozen prediction inputs, wrote outside the frozen prediction branch, and evaluated only after the ablation interval state existed. The result is recorded below.

## Workbench item 7 ablation: in-solver zero-support translation freeze

First controlled autoresearch branch: `clip-001851` was rerun from the frozen prediction inputs with `--freeze-translation-without-visible-surface-support` and the existing output gate. The branch wrote outside frozen prediction state under:

`/mnt/truenas-user-home/yiwen/ego_annotation_outputs/v19_runs/20260627_hot3d_clip001851_pinhole_a800_native_v1_supportgate/evaluation/autoresearch/in_solver_translation_freeze_v1/`

State SHA256: `31004d0367fedf05a3f17c9b3f2bdf051f258a8b0d7ca4d92bf4770541ffeaa3`.

Result: all 300 rows froze in-solver global translation and all 300 rows still preserved wrist/root. The mechanism was therefore tested as intended. It failed as an improvement: compared with the accepted output-gated candidate, translation-freeze worsened median joint MPJPE by `+0.006670432 m` and median root-aligned MPJPE by `+0.006031273 m` while wrist error stayed unchanged. Compared with runtime HaWoR, it worsened median joint MPJPE by `+0.005670470 m` and median root-aligned MPJPE by `+0.006883813 m`.

Interpretation: freezing zero-support translation inside the optimizer overconstrains the hand configuration, especially for the right hand, and degrades wrist-relative articulation. The output gate is the better current safety mechanism because it lets the optimizer use translation as an internal slack variable but projects unsupported global wrist/root motion back to HaWoR before emitting state. This ablation is rejected as a V19 improvement.

The next item-7 mechanism should not be stricter zero-support translation gating. It should construct a physically justified nearby/latent rigid-surface support posterior for occluded keyboard contact, because exact visible-mask overlap has no support rows and current contact-patch anchors are broad/sliding rather than stable.
