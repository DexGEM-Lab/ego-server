# V19 Post-Run Quantitative Comparison Orchestration

This document is for the evaluator phase after prediction outputs are frozen. It is not part of the runtime prediction prompt or runtime runbook.

## 14. Bounded quantitative comparison

Current command truth: `scripts/build_v19_hot3d_clip_adapter.py` adapts public HOT3D-Clips WebDataset tars into a V19 input video/frame manifest and an evaluation-only HOT3D GT sidecar. H2O and DexYCB adapters are not implemented. Workbench item 6 must run real adapters before any external quantitative claim.

Minimal HOT3D-Clips adaptation command:

```bash
python "$REPO_ROOT/scripts/build_v19_hot3d_clip_adapter.py" \
  --tar "$BENCH_ROOT/hot3d_clips/raw/train_aria/clip-001849.tar" \
  --output-root "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849" \
  --clip-id clip-001849 \
  --split train_aria \
  --image-field image_214-1.jpg
```

The adapter's HOT3D hand/object/MANO/object-pose annotations under `evaluation/hot3d_gt/` are scoring-only and must not feed object prompts, hand state, contact, occlusion, or physical-state selection. HOT3D `cameras.json` calibration is sensor metadata, not a perception label; it may feed the camera adapter below.

Current fixed HOT3D slice v1 for the initial gate is the first three `train_aria` clip tars in HuggingFace repository path order, selected before scoring clips beyond `clip-001849`: `clip-001849`, `clip-001850`, and `clip-001851`. The run artifact records this at `$BENCH_ROOT/hot3d_clips/evaluation/v19_hot3d_fixed_slice_v1.json`.

Minimal fisheye-to-pinhole camera adaptation command:

```bash
python "$REPO_ROOT/scripts/build_v19_hot3d_pinhole_adapter.py" \
  --input-root "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849" \
  --output-root "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849" \
  --stream-id 214-1
```

This camera adapter writes a new V19 input video/manifest plus `state/calibration/v19_hot3d_pinhole_camera_calibration_contract.json`. It is not a prediction and does not score 3D state; it only makes later V19/HaWoR runs consume a pinhole camera instead of raw fisheye frames.

Initial HOT3D hand-box comparison command, valid only for 2D localization claims:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_boxes.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/measurements/hawor_world_f609/hawor_world_hands.npz" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_hawor_box_eval.json" \
  --stream-id 214-1

python "$REPO_ROOT/scripts/render_v19_hot3d_hawor_box_review.py" \
  --manifest "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/input/raw_frame_manifest/manifest.json" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/measurements/hawor_world_f609/hawor_world_hands.npz" \
  --output "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_hawor_box_review.jpg"

python "$REPO_ROOT/scripts/aggregate_v19_hot3d_box_evals.py" \
  --reports "$BENCH_ROOT"/hot3d_clips/v19_inputs/clip-*/evaluation/hot3d_hawor_box_eval.json \
  --output-report "$BENCH_ROOT/hot3d_clips/evaluation/hot3d_hawor_box_eval_aggregate.json"
```

After HaWoR has been run on the pinhole input with the pinhole focal from the calibration contract, score 3D MANO hand state in camera coordinates:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001849/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/measurements/hawor_world_pinhole_f610/hawor_world_hands.npz" \
  --image-manifest "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/input/raw_frame_manifest/manifest.json" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/evaluation/hot3d_hawor_mano3d_eval.json" \
  --review-output "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001849/evaluation/hot3d_hawor_mano3d_review.jpg"

python "$REPO_ROOT/scripts/aggregate_v19_hot3d_mano3d_evals.py" \
  --reports "$BENCH_ROOT"/hot3d_clips/v19_inputs_pinhole/clip-*/evaluation/hot3d_hawor_mano3d_eval.json \
  --output-report "$BENCH_ROOT/hot3d_clips/evaluation/hot3d_hawor_mano3d_eval_aggregate.json"
```

The 3D evaluator must run in an environment with `smplx` and side-specific MANO assets; the current fixed-slice run used the remote HaWoR environment and explicit `--mano-left/--mano-right` paths. It replays HOT3D GT MANO with HaWoR's 21-joint ordering, then compares HOT3D GT and HaWoR predictions in camera coordinates. It reports absolute wrist/joint errors separately from wrist-subtracted translation-aligned errors; no rotation or scale Procrustes alignment is applied, so this is not a pure articulation metric. The aggregate also splits same-frame detector-supported rows from infilled rows. It still does not score contact, occlusion, nonpenetration, or object pose.

After a V19 runtime prediction boundary is frozen, the same evaluator may score a P18 interval-state JSON directly:

```bash
python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001850/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --interval-state "$RUN_ROOT/measurements/mano_interval_correction/keyboard_0_149/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-report "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_v19_interval_mano3d_eval.json" \
  --image-manifest "$RUN_ROOT/evaluation/hot3d_mano3d_interval/review_frames_1408/manifest.json" \
  --review-output "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_v19_interval_mano3d_review.jpg" \
  --mano-left "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_LEFT.pkl" \
  --mano-right "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_RIGHT.pkl"
```

Interval-state mode evaluates `optimized_joints_world_m` only. It obtains camera trajectory from the row `source_hawor_npz`, and it must not report full-vertex MANO metrics unless a future interval state stores full predicted vertices. Compare the HaWoR baseline report and frozen interval-state report with the reusable evaluator-phase comparator rather than an ad hoc JSON wrapper:

```bash
python "$REPO_ROOT/scripts/compare_v19_hot3d_mano3d_reports.py" \
  --baseline-report "$RUN_ROOT/evaluation/hot3d_mano3d_support_gated/hot3d_hawor_runtime_baseline_mano3d_eval.json" \
  --candidate-report "$RUN_ROOT/evaluation/hot3d_mano3d_support_gated/hot3d_v19_support_gated_runtime_mano3d_eval.json" \
  --baseline-label hawor_runtime_baseline \
  --candidate-label support_gated_runtime \
  --output-report "$RUN_ROOT/evaluation/hot3d_mano3d_support_gated/hot3d_baseline_vs_support_gated_runtime_comparison.json"
```

The comparison to the HaWoR NPZ baseline must be interpreted mechanistically: if MPJPE worsens while wrist-subtracted MPJPE is unchanged, the correction changed global wrist/root placement more than articulation, so the next intervention should gate translation/contact constraints rather than treat the interval correction as a hand-accuracy improvement.

Workbench item 7 autoresearch compares MANO correction mechanisms against that baseline, not camera-adapter variants. The first supported correction target is low-support/occluded intervals where HaWoR keeps plausible boxes but hallucinates MANO articulation. Build a prediction-side repaired NPZ or interval state, then score it with the same evaluator:

```bash
python "$REPO_ROOT/scripts/repair_v19_mano_support_temporal.py" \
  --input-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610/hawor_world_hands.npz" \
  --output-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/hawor_world_hands.npz" \
  --report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/v19_mano_support_temporal_repair_report.json" \
  --sides left --repair-mode wrist_relative \
  --min-score 0.45 --min-area-ratio 0.35 --pose-norm-z 3.0 --pose-norm-mad-floor 0.25 \
  --fill-gap-frames 3 --pre-dilate-frames 4 --post-dilate-frames 4 \
  --long-run-min-frames 5 --long-run-post-dilate-frames 22 \
  --min-interval-frames 3 --max-anchor-gap-frames 55 --min-raw-bad-frames-per-interval 12

python "$REPO_ROOT/scripts/evaluate_v19_hot3d_hawor_mano3d.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001850/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/measurements/hawor_world_pinhole_f610_left_support_temporal_repair_v5_wristrel/hawor_world_hands.npz" \
  --output-report "$BENCH_ROOT/hot3d_clips/v19_inputs_pinhole/clip-001850/evaluation/hot3d_hawor_mano3d_left_support_temporal_repair_v5_eval.json"
```

This repair consumes no HOT3D hand labels. It uses prediction-side support signals only: same-frame support, detector score/area, hand-pose magnitude, and temporal continuity. Its necessary comparison is HaWoR pinhole baseline versus repaired MANO on the fixed slice, with target-interval metrics and review sheets preserved. If it improves only wrist-subtracted error but not MPJPE, the mechanism supports an articulation-prior component but does not close MANO state; the next intervention must use stronger hand-owned visible-surface/mask-depth evidence rather than more temporal smoothing.

For interval-state autoresearch, first decompose the error before changing the solver:

```bash
python "$REPO_ROOT/scripts/analyze_v19_hot3d_interval_mano_delta.py" \
  --hot3d-gt "$BENCH_ROOT/hot3d_clips/v19_inputs/clip-001850/evaluation/hot3d_gt/hot3d_clip_gt_sidecar.json" \
  --hawor-npz "$RUN_ROOT/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --interval-state "$RUN_ROOT/measurements/mano_interval_correction/keyboard_0_149/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-json "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_interval_mano_error_decomposition.json" \
  --output-review "$RUN_ROOT/evaluation/hot3d_mano3d_interval/hot3d_interval_mano_error_decomposition_timeline.jpg" \
  --mano-left "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_LEFT.pkl" \
  --mano-right "$BUNDLE_ROOT/third_party/WiLoR/mano_data/MANO_RIGHT.pkl"
```

If MPJPE regresses while root-aligned MPJPE stays nearly fixed, inspect wrist correction direction. A high fraction of wrist corrections moving along the baseline error direction supports a systematic root-translation failure, not normal joint noise. In that case the prediction-side ablation is support-gated translation:

```bash
python "$REPO_ROOT/scripts/build_v19_interval_mano_translation_gate.py" \
  --interval-state "$RUN_ROOT/measurements/mano_interval_correction/keyboard_0_149/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-state "$RUN_ROOT/evaluation/hot3d_mano3d_interval/support_gated/v19_interval_mano_support_gated_state.json" \
  --output-report "$RUN_ROOT/evaluation/hot3d_mano3d_interval/support_gated/v19_interval_mano_support_gate_report.json" \
  --min-support-vertices 0 \
  --require-support-count
```

This gate uses no GT: when selected visible-surface support is absent, it preserves the source HaWoR wrist/root translation and keeps interval wrist-relative articulation. If that restores absolute MPJPE while leaving root-aligned MPJPE unchanged, the next runtime solver must enable `--gate-translation-with-visible-surface-support` rather than relying on ungated contact/temporal translation.

Integrated `clip-001850` result: after the runtime solver/spec enabled `--gate-translation-with-visible-surface-support`, P18-P21 regenerated prediction artifacts and froze manifest SHA256 `06c8ff45a955421b341ceafb0548b27d2d791e9ea6495f1fc6d17ee4db67c594` before any GT scoring. The frozen support-gated runtime state scored 300 HOT3D hand/frame rows with HaWoR baseline median wrist error `0.033586475 m` and support-gated wrist error `0.033586475 m`; baseline median joint MPJPE `0.047187152 m` and support-gated `0.046050225 m`; baseline median root-aligned MPJPE `0.022665785 m` and support-gated `0.022970178 m`. Comparison artifact: `evaluation/hot3d_mano3d_owlv2_support_gated_runtime/hot3d_baseline_vs_support_gated_runtime_comparison.json`, SHA256 `c35c49dc966ffb24a77b78b0d54c926ad21ba46bd90199b83c1af16367df45b5`. The evaluator review sheet projects GT and predicted joints onto visible hands, so the metric result is not a gross camera/review artifact. Scope: this is one HOT3D clip, 21-joint MANO only; it does not score object pose, contact, occlusion, nonpenetration, or full MANO vertices.

The next Workbench item-6 test is a new fixed HOT3D clip, not more threshold tuning on `clip-001850`. A falsifying result would be: zero-support rows on another clip still show absolute MPJPE regression after gating, or positive-support rows require translation corrections that the current gate wrongly blocks.

The runbook still fixes the evaluation discipline now:

- Primary benchmark: 3-5 HOT3D clips.
- Optional secondary: 2-3 H2O clips, or DexYCB only if H2O is blocked. Do not run both H2O and DexYCB in the initial V19 gate.
- Baselines: V18 v5 on project representatives; HaWoR on selected HOT3D clips; official/reference metrics only where the dataset annotates the claim.
- Required ablations: MANO candidate source/refit, depth/camera source, contact/occlusion/nonpenetration factors, rigid branch enabled versus visible-surface-only after a rigid decision.

If benchmark evaluation is requested before adapters exist, stop with `missing_benchmark_adapter` and name the blocked physical metric family. Do not fabricate `metrics.json`.

## Fixed HOT3D slice v1 outcome

The first Workbench item-6 fixed slice is now complete for HOT3D `clip-001849`, `clip-001850`, and `clip-001851`. The supported result is documented in `docs/v19_hot3d_fixed_slice_v1_results.md`.

Mechanism-level conclusion: support-gated V19 interval MANO preserves runtime HaWoR wrist/root translation on all three clips when selected visible-surface depth-order support is zero. It does not yet establish a general MANO-accuracy improvement. The next Workbench item-7 research target is physically supported correction beyond this safety gate: either in-solver translation freezing if latent unsupported translation contaminates articulation, or a nearby/latent visible-surface support mechanism if exact visible-mask overlap is too narrow for occluded keyboard contact.


Translation-freeze item-7 result: the `clip-001851` in-solver zero-support translation-freeze ablation preserved wrist/root but worsened median joint MPJPE by `+0.006670432 m` versus the accepted output-gated candidate. Do not integrate stricter zero-support translation freeze as the next V19 mechanism; pursue nearby/latent rigid-surface support evidence instead.
