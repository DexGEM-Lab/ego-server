# V19 English Orchestration Over Real Components

Status: Runtime prediction runbook. This is the runbook the Pi runtime agent follows to produce prediction-side physical annotation artifacts from an input video and fresh run root. It is intentionally written in English, but every executable action names an existing repository script or states an explicit missing implementation. It does not authorize fake numbered scripts, JSON registries, validator loops, or prior-version artifact repackaging as progress.

## 0. Physical objective and state variables

The runbook exists to build and render physical variables, not containers:

- `K_t`, `T_world_camera,t`, and depth/scale evidence: the metric coordinate frame that lets image observations become 3D claims.
- `H_t`: per-frame metric MANO hand state with side, camera/world semantics, provenance, visibility, and uncertainty.
- `O_i`: object instances with masks/tracks and a physical branch: rigid, articulated, deformable, support/occluder, or unresolved.
- `G_i`: object geometry. For a rigid branch this means a completed/adapted instance mesh, not a centroid, primitive, mask, or point cloud.
- `T_world_object,i,t`: object pose trajectory/posterior for rigid/articulated branches.
- `C_t`, `V_t`, `N_t`: contact/near-contact, visibility/occlusion ownership, and nonpenetration residual/uncertainty.
- Rendered overlay/world/side-by-side videos whose visible marks are caused by the variables above.

A command is progress only when it measures, optimizes, renders, or falsifies one of these variables. Reports, schemas, row counts, and validation outputs are evidence only.

## 1. Scope of current command truth

### Directly reusable existing scripts

These scripts exist and have traced CLIs:

- Fresh timeline/frame extraction: `scripts/build_v19_raw_frame_manifest.py`.
- Camera/depth/SLAM: `scripts/run_droid_full_frame.py`, `scripts/run_unidepth_full_frame_v3.py`, `scripts/run_unidepth_metric_source_v3.py`.
- Hand measurements: `scripts/run_rtmlib_hand2d_v3.py`, `scripts/run_wilor_full_frame.py`, `scripts/export_hawor_world.py`, `scripts/run_hamer_rtmlib_hand_stream_v3.py`, `scripts/merge_hand_candidate_streams_v7.py`, `scripts/refit_mano_metric_depth_v3.py`.
- Agent-replaced VLM structures: `scripts/build_object_plan_vlm.py` and `scripts/build_object_point_prompts_vlm.py` define the schemas; the runtime agent writes equivalent files instead of calling the API.
- Masks/tracks: `scripts/run_sam2_vlm_points_multiobject.py`.
- Fresh base annotation/state assembly: `scripts/build_v19_base_annotations.py`.
- SAM2/depth visible-surface bridge for rigid branches: `scripts/build_v19_visible_geometry_from_sam2_depth.py`.
- Observed object geometry and optimization candidates: `scripts/reconstruct_object_mesh_v2.py`, `scripts/reconstruct_scaled_observed_object_mesh_v3.py`, `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`, `scripts/complete_object_heightfield_from_mask_depth_v3.py`, `scripts/optimize_object_factor_graph_v3.py`, `scripts/optimize_joint_mano_object_graph_v3.py`, `scripts/optimize_joint_camera_object_graph_v3.py`, `scripts/optimize_contact_patch_object_pose_graph_v3.py`.
- V18 rigid branch components: `scripts/build_v18_compact_rigid_evidence_bundle.py`, `scripts/remote_run_trellis_shape_v3.py`, `scripts/build_v18_compact_rigid_trellis_completion.py`, `scripts/build_v18_scale_sane_compact_rigid_completion.py`, `scripts/fit_v18_compact_rigid_object_pose.py`.
- MANO/object correction and rendering: `scripts/build_v18_mano_object_constraint_state.py`, `scripts/build_v18_full_bridge_mano_object_constraint_state.py`, `scripts/apply_v18_mano_object_constraint_state.py`, `scripts/solve_v18_joint_mano_interval_trajectory.py`, `scripts/build_v18_compact_rigid_hidden_volume_depth_validation.py`, `scripts/build_v19_rigid_render_state.py`, `scripts/render_v19_rigid_state_artifact.py`, `scripts/render_v18_joint_mano_interval_correction.py`, `scripts/render_v18_full_pipeline_from_annotations.py`. `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py` is historical/diagnostic only because it renders sampled vertices rather than a rigid body.

### Remaining implementation that is not allowed to be faked

The default V19 path now has V19-owned commands for raw-frame extraction and base annotation/state assembly. The remaining non-fake gaps are narrower:

1. **Renderer naming/state cleanup.** Some reliable renderers still have `v18_*` filenames and consume V18-compatible annotation shapes. V19 may use them only as extracted executable components fed by V19-generated inputs, and final outputs must be copied/symlinked to canonical V19 MP4 names.
2. **V19 wrappers/generalized names for extracted rigid/contact/render components.** Existing `v18_*` scripts can be executed only when all inputs come from the V19 run root; future cleanup should rename/wrap them, but cached V18 roots are not valid pipeline inputs.
3. **Post-run comparison adapters.** External scoring adapters are outside this runtime prediction runbook.

If a run reaches one of these gaps, the correct outcome is a named missing implementation with the physical variable blocked. Do not invent a script name, use a cached prior-version root, or write a placeholder output.

## 2. Runtime variables

Use variables explicitly so commands remain general rather than tomato/trash-specific:

```bash
REPO_ROOT=/home/yiwen/ego_annotation
PYTHON="$REPO_ROOT/.venv/bin/python"            # replace with the active env for the chosen component
INPUT_VIDEO="<input video>"
RUN_ROOT="<run root>"
CASE_ID="<case id>"
FRAME_START=0
FRAME_END="<last source frame index>"
RAW_FRAME_MANIFEST="$RUN_ROOT/input/raw_frame_manifest/manifest.json"
BASE_ANNOTATIONS="$RUN_ROOT/state/base_annotations/annotations_v19_base.json"
OBJECT_ID="object:<track_id>"
TRACK_ID="<track_id>"
GPU_ID="<selected GPU>"
SAM2_CHECKPOINT="<sam2.1 checkpoint path>"
TRELLIS_REPO="<TRELLIS repo root>"
UNIDEPTH_REPO="<UniDepth repo root if not importable>"
DROID_ROOT="<DROID root>"
WILOR_ROOT="<WiLoR root>"
HAMER_ROOT="<HaMeR root>"
HAMER_CHECKPOINT="<HaMeR checkpoint>"
HAWOR_ROOT="<HaWoR root>"
```

Heavy commands run on the declared A800/server target after a non-mutating probe. Use one tmux session for long-running local/remote jobs; do not use `sleep` or polling loops.

## 3. Establish the timeline and coordinate frame

### 3.1 Raw frame manifest

Physical mechanism: every later measurement must refer to the same source frame index and image coordinate convention. A one-frame-per-source-frame manifest prevents time-base drift and hidden frame subsampling.

Default executable path:

```bash
python "$REPO_ROOT/scripts/build_v19_raw_frame_manifest.py" \
  --video "$INPUT_VIDEO" \
  --output-dir "$RUN_ROOT/input/raw_frame_manifest" \
  --render-width 960

RAW_FRAME_MANIFEST="$RUN_ROOT/input/raw_frame_manifest/manifest.json"
FRAME_END=$(python - "$RAW_FRAME_MANIFEST" <<'PY'
import json, sys
frames=json.load(open(sys.argv[1]))['frames']
print(max(int(f['frame_idx']) for f in frames))
PY
)
```

The output manifest is a V19-owned timeline artifact with `frame_idx`, `time_s`, `rgb`, `raw_frame_path`, source dimensions, and manifest dimensions. Do not use V16/V17/V18 raw-frame roots as pipeline inputs.

Systematic errors to rule out: frame index offset, resized-frame coordinate confusion, dropped video frames. Normal measurement error does not apply here; timeline mismatch is a contract error.

### 3.2 Camera/depth/scale sources

Physical mechanism: depth and camera pose convert 2D masks/keypoints into metric 3D rays/surfaces. These sources can be noisy, but a missing or inconsistent scale is a systematic error that contaminates all metric claims.

Full-frame metric depth/intrinsics candidate:

```bash
python "$REPO_ROOT/scripts/run_unidepth_full_frame_v3.py" \
  --manifest "$RAW_FRAME_MANIFEST" \
  --output-dir "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --unidepth-repo "$UNIDEPTH_REPO"
```

Outputs include `unidepth_full_frame_depth_v3.npz` and `qc_unidepth_full_frame_v3.json`. Treat focal/scale disagreement as uncertainty or a calibration bug, not as a reason to omit 3D state.

Build the run-level calibration contract before metric hand/object lifting. A physical camera normally has constant intrinsics; UniDepth's per-frame intrinsics are measurement hypotheses, not hardware truth. The default V19 contract robustly aggregates them into one video-level pinhole `K`:

```bash
python "$REPO_ROOT/scripts/build_v19_calibration_contract.py" \
  --case "$CASE_ID" \
  --raw-frame-manifest "$RAW_FRAME_MANIFEST" \
  --unidepth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "$RUN_ROOT/state/calibration" \
  --aggregation median

CALIBRATION_CONTRACT="$RUN_ROOT/state/calibration/v19_camera_calibration_contract.json"
CALIBRATION_INTRINSICS_NPZ="$RUN_ROOT/state/calibration/v19_camera_calibration_intrinsics.npz"
HAWOR_IMG_FOCAL="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["focal_geom_px"])' "$CALIBRATION_CONTRACT")"
```

If the dataset later provides real calibration, replace this contract source with that dataset calibration and keep the same downstream contract path. Do not mix HaWoR, DROID, UniDepth, and object surfels under different unaligned intrinsics.

Camera/head trajectory candidate:

```bash
python "$REPO_ROOT/scripts/run_droid_full_frame.py" \
  --clip "$INPUT_VIDEO" \
  --droid-root "$DROID_ROOT" \
  --weights "$DROID_ROOT/droid.pth" \
  --output-dir "$RUN_ROOT/measurements/depth_slam/droid"
```

Outputs include `droid_dense_trajectory.npz`, `droid_dense_trajectory.json`, and `droid_qc.json`. DROID provides a trajectory with calibration prior; it does not by itself prove metric scale.

Masked object-depth source after masks exist:

```bash
python "$REPO_ROOT/scripts/run_unidepth_metric_source_v3.py" \
  --manifest "$RUN_ROOT/measurements/masks_tracks/<track_id>/masked_depth_manifest.json" \
  --output-dir "$RUN_ROOT/measurements/depth_slam/unidepth_masked_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --unidepth-repo "$UNIDEPTH_REPO"
```

The masked source is an object-scale/depth observation. It is not an object pose or mesh.

## 4. Build metric MANO measurements

Physical mechanism: `H_t` starts as multiple noisy MANO hypotheses from hand detectors/reconstructors. 2D detections, boxes, and keypoints only localize image evidence; they become a physical hand state only after MANO surface/joints are metric-lifted, camera/world semantics are known, and uncertainty is carried into interval correction.

2D hand evidence:

```bash
python "$REPO_ROOT/scripts/run_rtmlib_hand2d_v3.py" \
  --clip "$INPUT_VIDEO" \
  --output-dir "$RUN_ROOT/measurements/hand_candidates/rtmlib" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --device cuda
```

WiLoR full-frame MANO candidate stream:

```bash
python "$REPO_ROOT/scripts/run_wilor_full_frame.py" \
  --clip "$INPUT_VIDEO" \
  --output-dir "$RUN_ROOT/measurements/hand_candidates/wilor_full_frame" \
  --wilor-root "$WILOR_ROOT" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END"
```

HaWoR world-space candidate stream when the environment is available:

```bash
python "$REPO_ROOT/scripts/export_hawor_world.py" \
  --hawor-root "$HAWOR_ROOT" \
  --video_path "$INPUT_VIDEO" \
  --input_type file \
  --img_focal "$HAWOR_IMG_FOCAL" \
  --force-focal-cache-refresh \
  --output-dir "$RUN_ROOT/measurements/hand_candidates/hawor_world"
```

The `--img_focal` value must come from `$CALIBRATION_CONTRACT` unless a recorded design amendment chooses a different calibrated hypothesis. Omitting it lets HaWoR choose an internal/default focal and breaks the shared metric backbone. HaWoR's sequence folder caches motion chunks, rendered masks, and SLAM under the video pathname, so the V19 wrapper refuses to reuse focal-dependent cache artifacts when their recorded focal differs from `--img_focal`; use a fresh focal-specific video path or the explicit `--force-focal-cache-refresh` flag when rerunning the same sequence under a new calibration.

Hand-owned surface evidence for MANO refit is acquired as a mask/depth measurement, not as a MANO label. When RTMLib is unavailable or when HaWoR is the calibrated hand source, generate SAM2 prompt files from calibrated HaWoR projections and run the same SAM2 point-prompt tracker on the interval:

```bash
python "$REPO_ROOT/scripts/build_v19_hawor_hand_sam2_prompts.py" \
  --hawor-npz "$RUN_ROOT/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --raw-frame-manifest "$RAW_FRAME_MANIFEST" \
  --calibration-contract "$CALIBRATION_CONTRACT" \
  --output-root "$RUN_ROOT/measurements/hand_candidates/hawor_hand_sam2_prompts" \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END" \
  --prompt-frames "$INTERVAL_START" "$INTERVAL_MID" "$INTERVAL_END"

python "$REPO_ROOT/scripts/run_sam2_vlm_points_multiobject.py" \
  --clip "$INPUT_VIDEO" \
  --point-root "$RUN_ROOT/measurements/hand_candidates/hawor_hand_sam2_prompts" \
  --output-root "$RUN_ROOT/measurements/hand_candidates/hawor_hand_sam2_masks" \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END" \
  --checkpoint "$SAM2_CHECKPOINT" \
  --model-cfg "$SAM2_MODEL_CFG" \
  --sam2-image-width 960
```

The HaWoR projections are prompt seeds only; the resulting SAM2 masks plus metric depth are candidate hand-owned surface observations. A SAM2 hand mask is usable for MANO refit only where it actually covers visible hand/palm/fingers after object-owned pixels and occluders are excluded. Forearm, sleeve, sink, object, or occluder pixels are not hand-owned surface; using them as MANO targets is a false physical mechanism.

When the calibrated HaWoR state still leaves a rendered MANO/object failure, prepare a scale-preserving mask/depth refit branch instead of tuning contact labels. This branch is an uncertain measurement path unless the rendered state and quantitative residuals support promotion:

```bash
python "$REPO_ROOT/scripts/build_v19_mano_mask_depth_refit_inputs.py" \
  --annotations "$RUN_ROOT/measurements/object_geometry/<visible_geometry_branch>/annotations_v19_visible_geometry.json" \
  --left-hand-track "$RUN_ROOT/measurements/hand_candidates/hawor_hand_sam2_masks/<left_track>/sam2/sam2_track.json" \
  --right-hand-track "$RUN_ROOT/measurements/hand_candidates/hawor_hand_sam2_masks/<right_track>/sam2/sam2_track.json" \
  --object-track "$RUN_ROOT/measurements/object_tracks/<object_track>/sam2/sam2_track.json" \
  --output-dir "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/prep" \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END" \
  --remote-root "$REMOTE_RUN_ROOT" \
  --local-root "$RUN_ROOT"

python "$REPO_ROOT/scripts/refit_mano_articulation_mask_depth_v3.py" \
  --annotations "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/prep/legacy_mano_refit_input_annotations.json" \
  --mask-track "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/prep/mask_tracks/<side>/sam2_track.json" \
  --metric-depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-annotations "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/<side>_refit_annotations.json" \
  --output-qc "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/<side>_refit_qc.json" \
  --video "$INPUT_VIDEO" \
  --review-dir "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/review_<side>" \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END" \
  --track-id "v19_<side>_hand_sam2_mano_filtered" \
  --side "<side>" \
  --source-width "$SOURCE_WIDTH" \
  --source-height "$SOURCE_HEIGHT" \
  --device cuda \
  --min-scale 1.0 \
  --max-scale 1.0

python "$REPO_ROOT/scripts/apply_v19_mano_mask_depth_refit.py" \
  --v19-annotations "$RUN_ROOT/measurements/object_geometry/<visible_geometry_branch>/annotations_v19_visible_geometry.json" \
  --source-hawor-npz "$RUN_ROOT/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --left-refit-annotations "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/left_refit_annotations.json" \
  --left-refit-qc "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/left_refit_qc.json" \
  --right-refit-annotations "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/right_refit_annotations.json" \
  --right-refit-qc "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/right_refit_qc.json" \
  --output-annotations "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/annotations_v19_visible_geometry_mask_depth_refit.json" \
  --output-bridge-npz "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/v19_mano_bridge_mask_depth_refit.npz" \
  --output-source-npz "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/hawor_world_hands_mask_depth_refit.npz" \
  --output-report "$RUN_ROOT/measurements/hand_candidates/mano_mask_depth_refit/apply_v19_mano_mask_depth_refit_report.json" \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END"
```

`build_v19_mano_mask_depth_refit_inputs.py` converts V19 world MANO state to a camera-frame refit contract and filters each SAM2 hand mask by calibrated MANO projection minus object mask. `apply_v19_mano_mask_depth_refit.py` promotes only fits representable as MANO pose plus one translation with no scale change. If a side's mask is forearm/occluder rather than visible hand, do not pass that side to `apply_v19_mano_mask_depth_refit.py`; leave it unresolved/occluded and continue with uncertainty.

HaMeR from RTMLib boxes requires a base annotation stream and a frame manifest:

```bash
python "$REPO_ROOT/scripts/run_hamer_rtmlib_hand_stream_v3.py" \
  --target-annotations "$BASE_ANNOTATIONS" \
  --frame-manifest "$RAW_FRAME_MANIFEST" \
  --rtmlib-json "$RUN_ROOT/measurements/hand_candidates/rtmlib/rtmlib_hand2d.json" \
  --output-annotations "$RUN_ROOT/measurements/hand_candidates/hamer_rtmlib/annotations_hamer.json" \
  --output-qc "$RUN_ROOT/measurements/hand_candidates/hamer_rtmlib/qc_hamer.json" \
  --hamer-root "$HAMER_ROOT" \
  --checkpoint "$HAMER_CHECKPOINT" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --device cuda:0
```

Merge candidate streams when they share the same frame convention:

```bash
python "$REPO_ROOT/scripts/merge_hand_candidate_streams_v7.py" \
  --base-annotations "$BASE_ANNOTATIONS" \
  --hand-streams \
    "$RUN_ROOT/measurements/hand_candidates/hamer_rtmlib/annotations_hamer.json" \
    "<other_hand_stream_annotations.json>" \
  --output-annotations "$RUN_ROOT/measurements/hand_candidates/merged/annotations_hand_candidates.json" \
  --output-qc "$RUN_ROOT/measurements/hand_candidates/merged/qc_merge_hand_candidates.json" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END"
```

Refit MANO depth/scale against metric depth:

```bash
python "$REPO_ROOT/scripts/refit_mano_metric_depth_v3.py" \
  --annotations "$RUN_ROOT/measurements/hand_candidates/merged/annotations_hand_candidates.json" \
  --metric-depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-json "$RUN_ROOT/measurements/hand_candidates/refit_mano_metric_depth.json" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END"
```

Systematic hand errors: left/right side swaps, camera-frame/world-frame mismatch, invalid MANO shape convention, scale bias, or temporal frame offset. Normal measurement errors: detector jitter, partial occlusion, local reprojection residuals, and noisy depth. Systematic errors must be fixed or represented as a separate hypothesis; normal errors continue downstream with uncertainty.

## 5. Replace VLM object discovery with agent visual judgment

Physical mechanism: object identity and physical branch choice are semantic/visual judgments. The agent should inspect sampled source frames, masks, depth overlays, and hand proximity, then write the same structures the old VLM scripts produced. This replaces the API call, not the downstream segmentation or geometry mechanisms.

### 5.1 Object plan file

Do **not** call `scripts/build_object_plan_vlm.py` during V19 runtime unless the user explicitly chooses API-based VLM assistance. Use the schema in that script and write:

```text
$RUN_ROOT/measurements/object_candidates/object_plan_agent.json
```

Required top-level shape:

```json
{
  "status": "ok",
  "backend": "Pi agent visual judgment replacing build_object_plan_vlm.py API call",
  "model": "pi-agent",
  "clip": "<input video>",
  "video": {"fps": 30.0, "width": 1920, "height": 1080, "frame_count": 960},
  "sampled_frames": [0, 120, 240],
  "plan": {
    "task_summary": "...",
    "objects": [
      {
        "track_id": "obj_instance_name",
        "description": "physical object instance, not a category-only label",
        "open_vocabulary_prompts": ["prompt strings for detection/segmentation"],
        "active_intervals": [{"start_frame": 0, "end_frame": 10, "evidence": "visible/source-frame evidence"}],
        "physical_notes": "...",
        "physical_model": {
          "primary_physical_model": "rigid | deformable | articulated | unknown | unknown_optically_difficult",
          "pose_model_allowed": true,
          "surface_appearance_changes": false,
          "geometry_changes": "none | minor_surface_layer_or_texture_change | nonrigid_deformation | articulation_or_part_motion | unknown",
          "requires_part_or_relative_motion_model": false,
          "secondary_deformable_or_surface_component": false,
          "optical_difficulty": false,
          "confidence": 0.0,
          "evidence": "what visible/physical evidence supports this branch",
          "uncertainty": "what would change the branch decision"
        },
        "confidence": 0.0
      }
    ],
    "uncertainties": []
  }
}
```

Rigid branch prediction before measuring: if the object is rigid, pairwise distances on visible surface points should be stable up to depth noise, a single SE(3)/Sim(3) should explain visible masks/surfaces over frames, and contact should move the object without deforming it. If those predictions fail systematically, switch to articulated/deformable/unresolved with evidence.

### 5.2 Point-prompt files

Do **not** call `scripts/build_object_point_prompts_vlm.py` during V19 runtime unless explicit API assistance is requested. The agent writes one file per track under a point-root directory that `run_sam2_vlm_points_multiobject.py` already expects:

```text
$RUN_ROOT/measurements/object_candidates/object_point_prompts_agent/<track_id>/object_point_prompts_vlm.json
```

Required fields consumed by SAM2:

```json
{
  "status": "ok",
  "backend": "Pi agent visual point prompts replacing build_object_point_prompts_vlm.py API call",
  "model": "pi-agent",
  "clip": "<input video>",
  "object_plan": "<object_plan_agent.json>",
  "track_id": "obj_instance_name",
  "description": "same object description",
  "object_plan_record": {"track_id": "obj_instance_name", "active_intervals": []},
  "prompt_image_width": 960,
  "point_prompts": [
    {
      "frame_idx": 123,
      "target_visible": true,
      "positive_points": [{"x": 100.0, "y": 120.0, "evidence": "inside visible target surface"}],
      "negative_points": [{"x": 50.0, "y": 80.0, "evidence": "nearby hand/support/not target"}],
      "bbox_xyxy": [80.0, 90.0, 160.0, 180.0],
      "visual_evidence": "why these pixels are the target instance",
      "confidence": 0.8
    }
  ],
  "batches": [],
  "elapsed_s": 0.0
}
```

Positive points must lie on visible pixels of the same physical instance. Negative points should target likely leakage: hands, supports, neighboring objects, and confusing parts. A prompt on a hand or support is a systematic identity error, not boundary noise.

## 6. Segment and track object surfaces with SAM2

Physical mechanism: SAM2 turns semantic point evidence into time-indexed visible object masks. Masks are measurements of visible image support. They are not geometry or pose until lifted by depth/camera and fitted to a model.

```bash
python "$REPO_ROOT/scripts/run_sam2_vlm_points_multiobject.py" \
  --clip "$INPUT_VIDEO" \
  --point-root "$RUN_ROOT/measurements/object_candidates/object_point_prompts_agent" \
  --output-root "$RUN_ROOT/measurements/object_tracks/sam2_agent_points" \
  --checkpoint "$SAM2_CHECKPOINT" \
  --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --sam2-image-width 960 \
  --render-width 960
```

Outputs include per-track `sam2/sam2_track.json`, `sam2/sam2_masks/`, `qc_sam2_vlm_points_track.json`, `qc_sam2_multiobject_points.json`, and `sam2_multiobject_overlay.mp4`.

Systematic mask errors: wrong object identity, persistent leakage onto the hand/support, track switch, active interval excluding real manipulation, or non-overlap conflict deleting the target. Normal mask errors: boundary jitter, holes, small missed occluded regions. Systematic errors require prompt/object-plan correction before geometry; normal errors become uncertainty in surface fitting.

## 6.5 Assemble fresh base annotations/state

Physical mechanism: downstream rigid/contact/MANO/render components need a single one-frame-per-source-frame backbone containing raw frame paths, camera/world pose, metric MANO candidates, object roster rows, and mask references. This is now generated from V19 run-root measurement outputs, not copied from V18 annotations.

```bash
python "$REPO_ROOT/scripts/build_v19_base_annotations.py" \
  --case "$CASE_ID" \
  --raw-frame-manifest "$RAW_FRAME_MANIFEST" \
  --depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --calibration-contract "$CALIBRATION_CONTRACT" \
  --hawor-npz "$RUN_ROOT/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --object-plan "$RUN_ROOT/measurements/object_candidates/object_plan_agent.json" \
  --sam2-output-root "$RUN_ROOT/measurements/object_tracks/sam2_agent_points" \
  --remote-root "<remote RUN_ROOT prefix if SAM2 ran on the server>" \
  --local-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/state/base_annotations"

BASE_ANNOTATIONS="$RUN_ROOT/state/base_annotations/annotations_v19_base.json"
```

The script also writes `v19_mano_bridge_from_hawor_world.npz`, `v19_base_physical_state.json`, and `v19_base_annotations_report.json`. It fails if no real camera/world pose source is supplied. When HaWoR world MANO is the hand source, the default camera pose should be HaWoR's `R_c2w/t_c2w` so camera and MANO vertices stay in the same world frame. Intrinsics should come from `$CALIBRATION_CONTRACT`; per-frame UniDepth intrinsics are measurement evidence used to build the contract, not the default rendered-state calibration. A DROID camera NPZ is a separate fresh camera candidate; pass it with `--prefer-camera-npz` only after estimating an explicit DROID↔HaWoR world alignment. If SAM2 ran on A800 and its track JSON contains server-absolute `mask_path` values, `--remote-root/--local-root` localizes those paths after the SAM2 output directory is copied into the local run root; this preserves the same mask measurement and avoids a local-only path contract error. The compatibility field names used by existing MANO solvers are present, but their data source is the fresh V19 HaWoR/camera run.

## 7. Lift masks to visible metric surfaces

Physical mechanism: visible surface points are produced by back-projecting mask pixels with depth through intrinsics into camera/world coordinates. They are metric measurements and anchors. They are not hidden geometry and do not satisfy rigid object pose by themselves.

The default V19 bridge from SAM2 masks to rigid-branch visible surfaces is a candidate-review flow, not a one-shot automatic anchor. First propose anchor candidates:

```bash
python "$REPO_ROOT/scripts/build_v19_visible_geometry_from_sam2_depth.py" \
  --case "$CASE_ID" \
  --track-id "$TRACK_ID" \
  --object-id "$OBJECT_ID" \
  --raw-frame-manifest "$RAW_FRAME_MANIFEST" \
  --base-annotations "$BASE_ANNOTATIONS" \
  --sam2-track-json "$RUN_ROOT/measurements/object_tracks/sam2_agent_points/$TRACK_ID/sam2/sam2_track.json" \
  --depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --calibration-contract "$CALIBRATION_CONTRACT" \
  --object-plan "$RUN_ROOT/measurements/object_candidates/object_plan_agent.json" \
  --output-dir "$RUN_ROOT/measurements/object_geometry/anchor_candidates_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --propose-anchor-candidates-only \
  --anchor-candidate-count 12
```

This writes `anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`. The proposal score is an ordering aid only. The agent must inspect the review image and write `$RUN_ROOT/state/anchor_decisions/<object_id>.json` with the selected frame, visual rationale, rejected candidates, and remaining uncertainty. The chosen frame should be the cleanest full-object evidence: broad visible object support, low hand overlap, non-border mask, coherent object outline/texture, stable depth, and metric extent consistent with plausible neighboring frames. Do not choose an anchor merely because it has the largest mask, most sampled points, or lies near a manipulation/contact moment.

Then run the canonical visible-geometry adapter using that explicit decision:

```bash
python "$REPO_ROOT/scripts/build_v19_visible_geometry_from_sam2_depth.py" \
  --case "$CASE_ID" \
  --track-id "$TRACK_ID" \
  --object-id "$OBJECT_ID" \
  --raw-frame-manifest "$RAW_FRAME_MANIFEST" \
  --base-annotations "$BASE_ANNOTATIONS" \
  --sam2-track-json "$RUN_ROOT/measurements/object_tracks/sam2_agent_points/$TRACK_ID/sam2/sam2_track.json" \
  --depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --calibration-contract "$CALIBRATION_CONTRACT" \
  --object-plan "$RUN_ROOT/measurements/object_candidates/object_plan_agent.json" \
  --output-dir "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<agent_selected_full_object_frame>" \
  --require-anchor-frame
```

The `--base-annotations` input must be the V19-generated base annotation file above. When HaWoR MANO is the active hand state, use its camera poses from the base annotations unless a camera NPZ has been explicitly aligned into the same HaWoR/MANO world frame; do not mix DROID and HaWoR worlds by default. Mask/depth backprojection must use `$CALIBRATION_CONTRACT` or the same intrinsics already recorded in base annotations; do not silently fall back to per-frame UniDepth `K` after a contract exists. The script writes:

```text
annotations_v19_visible_geometry.json
v19_visible_geometry_adapter_report.json
v19_visible_geometry_depth_fused_report.json
anchor_visible_surface_mesh/<object>/frame_<anchor>_*_canonical.ply
```

The annotation output contains one frame per selected source frame, `raw_frame_path`, camera pose provenance, object `mask_path`, `visible_geometry_candidate.world_vertices_sample_m`, and a centroid-only `reconstructed_geometry_pose` initialization. With an anchor frame, the adapter also exports selected-frame visible surfels in object-canonical anchor-centroid coordinates as a point cloud/Poisson/hull mesh for TRELLIS metric alignment. The adapter marks visible surfaces whose metric extent is inconsistent with the selected rigid-object anchor as ineligible for rigid pose fitting; those rows remain mask/depth measurements with systematic leakage uncertainty. This is a rigid-branch measurement adapter, not final pose. It fails unless a real camera/world pose is provided through base annotations or an explicitly aligned `--camera-npz`; `--allow-camera-frame-world` must be explicit and is not valid for temporal metric world claims.

Additional runnable options depend on available annotation shape:

### 7.1 Annotation + DROID/depth route

When the current annotation stream contains masks and camera fields, use the existing object mesh reconstruction command:

```bash
python "$REPO_ROOT/scripts/reconstruct_object_mesh_v2.py" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/annotations_v19_visible_geometry.json" \
  --droid-npz "$RUN_ROOT/measurements/depth_slam/droid/droid_dense_trajectory.npz" \
  --droid-reconstruction "$RUN_ROOT/measurements/depth_slam/droid/droid_keyframe_reconstruction.pth" \
  --metric-depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "$RUN_ROOT/measurements/object_geometry/reconstruct_object_mesh_v2" \
  --droid-to-meters "<estimated_or_calibrated_scale>"
```

### 7.3 Observed-mesh sequence refinement

When an observed mesh archive exists, use:

```bash
python "$REPO_ROOT/scripts/reconstruct_scaled_observed_object_mesh_v3.py" \
  --observed-mesh-npz "<observed_mesh_archive.npz>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<anchor_frame>" \
  --output-dir "$RUN_ROOT/measurements/object_geometry/scaled_observed_<track_id>" \
  --annotations "<annotations.json>"
```

This tests whether visible surfaces can be brought into a coherent object coordinate frame. A stable rigid fit supports a rigid branch; unstable residuals can indicate depth/camera bias, mask identity error, articulation, or deformation.

## 8. Agent branch decision

The agent must choose the physical branch from evidence, not object names:

- **Rigid:** visible shape is one body; a single pose explains the visible surface trajectory; no relative part motion is needed. This forces the rigid branch in Section 9.
- **Articulated:** parts maintain local rigidity but move relative to each other; pose state must include part graph/joints.
- **Deformable:** material shape changes enough that a rigid pose would be false; render uncertainty/deformation rather than rigid pose.
- **Support/occluder:** object affects contact/occlusion but is not the manipulated object state.
- **Unresolved:** evidence cannot distinguish the mechanisms; render uncertainty and choose the next discriminating measurement.

Branch decision falsifiers must be explicit. Example: if rigid is chosen, a later large surface residual under a well-calibrated camera/depth model falsifies either the rigid model, the mask identity, or the pose solver, and the next action must distinguish those mechanisms.

## 9. Mandatory rigid branch

After a rigid decision, visible surfaces are measurements only. The branch must execute completion/adaptation, visible-frame pose, correction/factor reasoning, and corrected mesh-pose rendering.

### 9.1 Evidence crop/bundle for completion

```bash
python "$REPO_ROOT/scripts/build_v18_compact_rigid_evidence_bundle.py" \
  --case "$CASE_ID" \
  --object-id "$OBJECT_ID" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/annotations_v19_visible_geometry.json" \
  --depth-fused-report "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/v19_visible_geometry_depth_fused_report.json" \
  --output-root "$RUN_ROOT/measurements/geometry_completion/rigid_evidence" \
  --selected-frame-idx "<agent_selected_full_object_frame_from_anchor_decision>" \
  --selection-note "anchor decision: <why this crop best represents the rigid object>"
```

P11 must reuse the frame recorded in `$RUN_ROOT/state/anchor_decisions/<object_id>.json`; it must not silently rerank by support count or mask area after the agent has selected an anchor. The conditioning crop is evidence for a mesh prior. It is not itself geometry.

### 9.2 TRELLIS mesh prior

Run on the A800/server with the selected GPU:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$REPO_ROOT/scripts/remote_run_trellis_shape_v3.py" \
  --repo "$TRELLIS_REPO" \
  --image "<evidence_bundle/crops/frame_xxxxxx_<object>_rgba.png>" \
  --output-dir "$RUN_ROOT/measurements/geometry_completion/trellis_<track_id>" \
  --mesh-name "${TRACK_ID}_trellis_seed42.ply" \
  --seed 42
```

TRELLIS produces a prior mesh in model units. It becomes a physical mesh only after metric adaptation to observed evidence.

### 9.3 Adapt/complete the mesh to observed metric surfaces

If the evidence bundle contains fused observed points and a Poisson visible mesh, use compact-rigid completion:

```bash
python "$REPO_ROOT/scripts/build_v18_compact_rigid_trellis_completion.py" \
  --evidence-report "<evidence_bundle_report.json>" \
  --trellis-report "$RUN_ROOT/measurements/geometry_completion/trellis_<track_id>/qc_trellis_shape_v3.json" \
  --output-dir "$RUN_ROOT/measurements/geometry_completion/completed_<track_id>"
```

If the stronger available evidence is a selected-frame visible-depth sample in annotations, use the scale-sane adaptation path:

```bash
python "$REPO_ROOT/scripts/build_v18_scale_sane_compact_rigid_completion.py" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/annotations_v19_visible_geometry.json" \
  --trellis-mesh "$RUN_ROOT/measurements/geometry_completion/trellis_<track_id>/${TRACK_ID}_trellis_seed42.ply" \
  --trellis-report "$RUN_ROOT/measurements/geometry_completion/trellis_<track_id>/qc_trellis_shape_v3.json" \
  --object-id "$OBJECT_ID" \
  --scale-frame-idx "<agent_selected_scale_frame>" \
  --output-dir "$RUN_ROOT/measurements/geometry_completion/scale_sane_<track_id>"
```

Systematic completion errors: wrong mesh instance, wrong scale, alignment to hand/support instead of object, mirrored/flipped pose, hidden prior overriding observed surface, or uncertainty labels that do not change downstream body semantics. Normal errors: incomplete hidden side, small surface residuals, partial depth noise. Observed depth must overwrite visible regions, but unsupported observed Poisson fill is diagnostic uncertainty, not accepted object body; `outputs.completed_mesh_labeled` must exclude `unsupported_uncertain` faces from the accepted mesh consumed by pose/contact/render.

### 9.4 Fit per-frame visible object pose

```bash
python "$REPO_ROOT/scripts/fit_v18_compact_rigid_object_pose.py" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/annotations_v19_visible_geometry.json" \
  --completion-report "<completion_report.json>" \
  --object-id "$OBJECT_ID" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>"
```

This creates `v18_compact_rigid_object_pose_fit_report.json`. The physical claim is: the completed mesh has per-frame `T_world_object` initialized/refit from current visible depth. A high residual is a mechanism signal, not a reason to demote rigid pose to a point cloud.

### 9.5 Correct the rigid pose trajectory

The independent visible-frame ICP rows are pose measurements, not a corrected trajectory. After a rigid decision, run the temporal pose graph so normal framewise measurement noise is smoothed and bounded hand/object nonpenetration pressure can be tested without letting it override visible-depth evidence:

```bash
python "$REPO_ROOT/scripts/solve_v19_rigid_object_pose_graph.py" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry_<track_id>/annotations_v19_visible_geometry.json" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completion-report "<completion_report.json>" \
  --constraint-report "$RUN_ROOT/measurements/contact_nonpenetration/mano_object_<track_id>/v18_mano_object_constraint_state.json" \
  --object-id "$OBJECT_ID" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>_corrected_graph"
```

If the MANO/object constraint state does not exist yet, run this graph without `--constraint-report`, then build the constraint state and rerun the graph if the first constraint measurement exposes object-pose-sized conflicts. The output is still a pose report with `pose_rows`, but corrected rows carry `status: corrected_temporal_rigid_pose_graph`. Existing render/constraint consumers accept this status.

Interpretation rule: if the corrected graph preserves visible-surface residuals and only moves the object by sub-millimetre or millimetre-scale deltas while MANO/object penetration remains broad, the dominant mechanism is not object-pose jitter. Continue to interval MANO/contact/occlusion reasoning. Do not keep increasing object-pose graph weights to hide a hand/camera/contact conflict.

### 9.6 Optional generic object/camera/hand pose graphs

Use these only when their required inputs exist. They are real scripts, but not yet wired to a general V19 state adapter.

Object pose graph:

```bash
python "$REPO_ROOT/scripts/optimize_object_factor_graph_v3.py" \
  --annotations "<annotations.json>" \
  --droid-npz "$RUN_ROOT/measurements/depth_slam/droid/droid_dense_trajectory.npz" \
  --observed-mesh-npz "<observed_mesh_archive.npz>" \
  --mesh-prior "<completed_or_prior_mesh.ply>" \
  --initial-alignment-qc "<initial_alignment_qc.json>" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/object_factor_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<anchor_frame>"
```

Joint MANO/object graph:

```bash
python "$REPO_ROOT/scripts/optimize_joint_mano_object_graph_v3.py" \
  --annotations "<annotations.json>" \
  --droid-npz "$RUN_ROOT/measurements/depth_slam/droid/droid_dense_trajectory.npz" \
  --observed-mesh-npz "<observed_mesh_archive.npz>" \
  --mesh-prior "<completed_or_prior_mesh.ply>" \
  --initial-alignment-qc "<initial_alignment_qc.json>" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/joint_mano_object_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<anchor_frame>"
```

Camera/object graph:

```bash
python "$REPO_ROOT/scripts/optimize_joint_camera_object_graph_v3.py" \
  --mesh-prior-camera "<mesh_prior_camera_frame.ply>" \
  --observed-mesh-npz "<observed_mesh_archive.npz>" \
  --dataset "<mask/depth dataset dir>" \
  --manifest "<mask/depth manifest.json>" \
  --annotations "<annotations.json>" \
  --initial-object-pose-qc "<initial_object_pose_qc.json>" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/joint_camera_object_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<anchor_frame>"
```

Contact-patch object pose graph:

```bash
python "$REPO_ROOT/scripts/optimize_contact_patch_object_pose_graph_v3.py" \
  --annotations "<annotations.json>" \
  --manifest "<mask/depth manifest.json>" \
  --metric-depth-npz "<metric_depth.npz>" \
  --contact-report "<contact_report.json>" \
  --mesh-prior-camera "<mesh_prior_camera_frame.ply>" \
  --output-dir "$RUN_ROOT/measurements/pose_fits/contact_patch_object_<track_id>" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --anchor-frame "<anchor_frame>"
```

### 9.7 MANO/object nonpenetration and interval MANO correction

Build a MANO/object constraint state:

```bash
python "$REPO_ROOT/scripts/build_v18_mano_object_constraint_state.py" \
  --annotations "<annotations_with_hands_and_pose.json>" \
  --hawor-npz "<hawor_bridge_or_export_npz>" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completion-report "<completion_report.json>" \
  --object-id "$OBJECT_ID" \
  --output-dir "$RUN_ROOT/measurements/contact_nonpenetration/mano_object_<track_id>"
```

If a watertight sign mesh exists and the full bridge semantics are required, use:

```bash
python "$REPO_ROOT/scripts/build_v18_full_bridge_mano_object_constraint_state.py" \
  --annotations "<annotations_with_hands_and_pose.json>" \
  --hawor-npz "<hawor_bridge_or_export_npz>" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completion-report "<completion_report.json>" \
  --sign-mesh "<watertight_sign_mesh.ply>" \
  --sign-mesh-source-report "<sign_mesh_source_report.json>" \
  --object-id "$OBJECT_ID" \
  --output-dir "$RUN_ROOT/measurements/contact_nonpenetration/full_bridge_<track_id>"
```

### 9.7.1 Agent-native contact/occlusion judgment for interval MANO

V19 replaces V18-style VLM/API contact/occlusion judgment with an explicit Pi-agent-authored interval artifact. The agent does **not** label pixels. It inspects raw frames, overlays, world views, mask/depth reviews, and prior branch failures, then writes semantic interaction priors that the factor builder must convert into numeric graph rows.

Write one judgment file under the run root:

```text
$RUN_ROOT/state/agent_interaction_judgments/<track_id>_<start>_<end>_v1.json
```

Required segment fields consumed by `scripts/build_v19_visible_contact_ownership_factor.py`:

```json
{
  "status": "ok",
  "method": "v19_pi_agent_interaction_judgment_v1",
  "backend": "Pi agent visual judgment replacing VLM/API contact-occlusion judgment",
  "case": "<case_id>",
  "target_entity_id": "<object_id>",
  "interaction_judgments": [
    {
      "judgment_id": "right_700_725_visible_grasp_contact_prior",
      "frame_start": 700,
      "frame_end": 725,
      "hand_side": "right",
      "contact_state": "likely_contact | possible_contact | no_contact | unresolved",
      "contact_prior_probability": 0.82,
      "contact_support_uncertainty_m": 0.055,
      "contact_weight_multiplier": 1.0,
      "occlusion_relation": "hand_in_front_of_object | object_in_front_of_hand | object_partially_occluded_by_hand | no_visible_occlusion | unresolved",
      "depth_reliability": "hand_depth_unreliable | object_depth_reliable | mixed_or_unresolved | not_evaluated",
      "ownership_quarantine": "hand_projected | none | unresolved | measurement_default",
      "evidence": "what visible evidence supports the semantic relation",
      "uncertainty": "what would revise the relation or requires soft graph treatment"
    }
  ]
}
```

`contact_prior_probability` and `contact_support_uncertainty_m` are required because the factor graph consumes numbers, not prose. Missing values are a broken contract, not defaults. `ownership_quarantine=hand_projected` means projected MANO support may quarantine object hard-surface constraints where the agent judged the hand to be in front or unresolved; `none` keeps the object mask eligible when the agent judged no occlusion/object-in-front. The rigid-extent mask filter still wins: known leaky object masks must stay skipped even when the agent judges contact in that interval.

Build solver-consumed factor rows from the agent judgment plus projected MANO/object support:

```bash
python "$REPO_ROOT/scripts/build_v19_visible_contact_ownership_factor.py" \
  --annotations "<annotations_with_rigid_extent_eligibility.json>" \
  --case "$CASE_ID" \
  --target-entity-id "$OBJECT_ID" \
  --frame-span "<interval_start>" "<interval_end>" \
  --sides left right \
  --agent-interaction-judgment "$RUN_ROOT/state/agent_interaction_judgments/<track_id>_<start>_<end>_v1.json" \
  --output-root "$RUN_ROOT/measurements/contact_visibility_factors/<track_id>_<start>_<end>_agent_judgment_v1/visible_contact_ownership_agent_v1"
```

The generated report contains `visible_ownership` rows and `contact_patch` rows. Agent judgment changes solver-consumed `contact_state_prior_probability`, `weight`, `contact_patch_support_uncertainty_m`, `object_support_uncertainty_m`, and ownership quarantine masks. It does not create a persistent object-frame contact anchor or accepted metric contact by itself.

Regenerate dependent visibility factors from the same ownership report before solving; do not mix old ownership reports with new agent priors.

Solve interval MANO with the rigid object and optional factor reports:

```bash
python "$REPO_ROOT/scripts/solve_v18_joint_mano_interval_trajectory.py" \
  --case "$CASE_ID" \
  --object-id "$OBJECT_ID" \
  --annotations "<sanitized_annotations_with_hands_and_pose.json>" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completed-mesh "<completed_mesh_labeled.ply>" \
  --depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "$RUN_ROOT/measurements/interval_mano" \
  --start-frame "<interval_start>" \
  --end-frame "<interval_end>" \
  --sides left right \
  --optimize-object-translation \
  --hand-owned-object-depth-quarantine \
  --visibility-weighted-hand-observation
```

Add `--factor-report <path>` for visible ownership, visible surface track, surface eligibility, hand-observation visibility, hand-depth shift, or contact-patch factors when those reports are actually produced. Do not add empty factor paths to look rigorous.

V19 interval repair lessons from the fresh task5 run:

- Raw UniDepth sampled at projected MANO joints is not automatically hand-depth evidence. If an optical-axis hand-depth prior reduces object residuals only by producing large image drift, and a root-ray prior preserves image evidence but cannot move, treat the projected-joint depth residual as hand/occlusion ownership uncertainty rather than as a direct hand-depth target.
- Use `scripts/build_v19_visible_contact_ownership_factor.py` to create the first generic V19 contact/ownership source when annotations lack `contact_hypotheses`. In V19 it should normally consume the Pi-agent interaction judgment JSON above, so true contact/occlusion semantics come from explicit agent visual judgment and projected MANO/object adjacency supplies local support. It skips object masks already marked ineligible by the rigid-extent filter. Its rows are latent/sliding contact and hand-owned visibility quarantine, not persistent contact anchors.
- If contact rows become active but the rendered MANO remains incoherent, use `scripts/refit_v19_mano_contact_similarity_interval.py` only as a bounded diagnostic: it tests whether a camera-space Sim(3) of the current MANO can satisfy image projection plus rigid-object contact. A solution that saturates the hand-scale bound is negative evidence for acceptance and points to a missing full MANO pose/shape/depth refit, not a successful correction.

The solver's falsifiable claim: if root translation, root orientation, articulation, and bounded object translation cannot make MANO compatible with visible/depth/object constraints under the stated uncertainty, then the remaining failure is a real conflict among hand observation, object pose/geometry, camera/depth alignment, occlusion, or contact evidence. The next action must distinguish those mechanisms.

### 9.8 Hidden-volume/depth validation

```bash
python "$REPO_ROOT/scripts/build_v18_compact_rigid_hidden_volume_depth_validation.py" \
  --case "$CASE_ID" \
  --object-id "$OBJECT_ID" \
  --annotations "<annotations_with_hands_and_pose.json>" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completed-mesh "<completed_mesh_labeled.ply>" \
  --depth-npz "$RUN_ROOT/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --temporal-mano-state "$RUN_ROOT/measurements/interval_mano/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-dir "$RUN_ROOT/measurements/contact_nonpenetration/hidden_volume_<track_id>"
```

This tests whether completed hidden surfaces contradict observed depth/free space. It does not replace rendering.

## 10. Contact, occlusion, and nonpenetration factors

The V18 reducers exist but are mostly root-based. Use them when the required roots are produced; otherwise create the missing factor adapter rather than inventing reports.

Root-based contact ownership:

```bash
python "$REPO_ROOT/scripts/build_v18_contact_ownership_graph.py" \
  --mesh-contact-root "<mesh_contact_root>" \
  --output-root "$RUN_ROOT/measurements/contact_nonpenetration/contact_ownership" \
  --cases "$CASE_ID"
```

Root-based occlusion ownership:

```bash
python "$REPO_ROOT/scripts/build_v18_occlusion_owner_graph.py" \
  --occlusion-mesh-root "<occlusion_mesh_root>" \
  --occlusion-depth-root "<occlusion_depth_root>" \
  --output-root "$RUN_ROOT/measurements/occlusion/occlusion_owner" \
  --cases "$CASE_ID"
```

Signed nonpenetration evidence:

```bash
python "$REPO_ROOT/scripts/build_v18_signed_nonpenetration_evidence.py" \
  --contact-ownership-root "<contact_ownership_root>" \
  --full-pipeline-root "<full_pipeline_root>" \
  --depth-fused-root "<depth_fused_root>" \
  --physical-state-schema-root "<physical_state_schema_root>" \
  --output-root "$RUN_ROOT/measurements/contact_nonpenetration/signed_nonpenetration" \
  --cases "$CASE_ID"
```

Triangle nonpenetration evidence:

```bash
python "$REPO_ROOT/scripts/build_v18_triangle_nonpenetration_evidence.py" \
  --contact-ownership-root "<contact_ownership_root>" \
  --full-pipeline-root "<full_pipeline_root>" \
  --depth-fused-root "<depth_fused_root>" \
  --physical-state-schema-root "<physical_state_schema_root>" \
  --output-root "$RUN_ROOT/measurements/contact_nonpenetration/triangle_nonpenetration" \
  --cases "$CASE_ID"
```

Interpretation rule: weak or conflicting factors must widen uncertainty, downweight residuals, or trigger a discriminating measurement. They must not delete the contact/occlusion/nonpenetration variable family.

## 11. Renderable state assembly

The renderer boundary is an explicit V19 render-state JSON under `$RUN_ROOT/state/render_state/`. The annotation JSON remains the frame/backbone source, but the final rigid object layer is not allowed to read pose/mesh measurement reports privately. Before rendering, materialize the completed mesh path, accepted full-timeline rigid pose rows, MANO/object uncertainty rows, temporal MANO state, and projection contract into render-consumed state.

Apply coordinate-level constraint candidates into annotations only when the constraint report marks them as accepted or explicitly uncertain:

```bash
python "$REPO_ROOT/scripts/apply_v18_mano_object_constraint_state.py" \
  --annotations "<annotations_before_constraint.json>" \
  --constraint-report "$RUN_ROOT/measurements/contact_nonpenetration/mano_object_<track_id>/v18_mano_object_constraint_state.json" \
  --output-annotations "$RUN_ROOT/state/annotations_with_mano_object_constraint.json" \
  --summary "$RUN_ROOT/state/apply_mano_object_constraint_summary.json"
```

Write `state/v19_physical_state.json`, `state/v19_uncertainty_state.json`, and `state/v19_agent_evidence.md` as renderer-facing sidecars. They must name which annotation/interval/mesh/pose files drive each visual layer. They are not completion evidence by themselves.

For every rigid branch, build the concrete render state before P19 rendering:

```bash
python "$REPO_ROOT/scripts/build_v19_rigid_render_state.py" \
  --case "$CASE_ID" \
  --object-id "$TRACK_ID" \
  --object-label "$TRACK_ID" \
  --annotations "$RUN_ROOT/measurements/object_geometry/visible_geometry/$TRACK_ID/annotations_v19_visible_geometry.json" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_$TRACK_ID/v19_rigid_object_pose_graph_report.json" \
  --completed-mesh "<completed_mesh_labeled.ply>" \
  --completion-report "<completion_report.json>" \
  --constraint-report "$RUN_ROOT/measurements/contact_nonpenetration/mano_object_$TRACK_ID/v18_mano_object_constraint_state.json" \
  --temporal-mano-state "$RUN_ROOT/measurements/interval_mano/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output "$RUN_ROOT/state/render_state/${TRACK_ID}_rigid_render_state.json"
```

This state builder fails by default when a rigid branch lacks full-timeline pose rows. Missing local mask/depth observations should already have become uncertain rigid poses in the P15 graph, not omitted render frames.

## 12. Full-duration rendering

### 12.1 Interval MANO + rigid mesh render

```bash
python "$REPO_ROOT/scripts/render_v18_joint_mano_interval_correction.py" \
  --case "$CASE_ID" \
  --annotations "$RUN_ROOT/state/annotations_with_mano_object_constraint.json" \
  --pose-report "$RUN_ROOT/measurements/pose_fits/rigid_<track_id>/v18_compact_rigid_object_pose_fit_report.json" \
  --completed-mesh "<completed_mesh_labeled.ply>" \
  --joint-mano-state "$RUN_ROOT/measurements/interval_mano/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --full-video \
  --output-root "$RUN_ROOT/renders/interval_mano_rigid_<track_id>"
```

Outputs include full-video overlay/world/side-by-side files and `v18_joint_mano_interval_correction_render_manifest.json`.

### 12.2 Rigid object body + MANO uncertainty render

The default rigid branch renderer consumes the render state from Section 11 and rasterizes mesh faces as a body. It also scales source-coordinate intrinsics to the decoded render frame size before projection, so a 960x960 render of 1408x1408 source intrinsics does not use unscaled source K:

```bash
python "$REPO_ROOT/scripts/render_v19_rigid_state_artifact.py" \
  --render-state "$RUN_ROOT/state/render_state/${TRACK_ID}_rigid_render_state.json" \
  --output-root "$RUN_ROOT/renders/${TRACK_ID}_rigid_state_runtime" \
  --world-view local
```

Outputs include full-video overlay/world/side-by-side files and `v19_rigid_state_render_manifest.json`. The manifest must show `rigid_object_body_rasterized_from_mesh_faces: true`, nonzero body pixels for frames with object pose, and projection examples with `scaled_intrinsics_fx_fy_cx_cy`. The old `render_v18_compact_rigid_tomato_temporal_mano_attempt.py` may be used only for historical diagnostics; it draws sampled vertices and cannot close the V19 final rigid-body render requirement.

### 12.3 Branch comparison render

When item-4 iteration produces multiple physical branches, render a comparison video/contact sheet from the produced overlay/world frames and interval metrics. This is the preferred way to present quantitative branch tradeoffs without treating a JSON report as the deliverable:

```bash
python "$REPO_ROOT/scripts/render_v19_interval_branch_comparison.py" \
  --branch calibrated="$RUN_ROOT/renders/<calibrated_branch>/$CASE_ID:$RUN_ROOT/measurements/interval_mano/<calibrated_branch>/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --branch candidate="$RUN_ROOT/renders/<candidate_branch>/$CASE_ID:$RUN_ROOT/measurements/interval_mano/<candidate_branch>/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --side right \
  --frame-start "$INTERVAL_START" \
  --frame-end "$INTERVAL_END" \
  --output-dir "$RUN_ROOT/renders/review_frames/interval_branch_comparison" \
  --still-frames "$INTERVAL_START" "$INTERVAL_MID" "$INTERVAL_END"
```

The comparison renderer consumes state-driven render frames; it must not be used to replace canonical full-duration overlay/world/side-by-side videos.

### 12.4 Canonical V19 render names

Do not use render commands that require prior-version raw-frame roots as pipeline inputs. The default V19 render path is the render-state rigid-body renderer above, fed by V19-generated annotations whose `raw_frame_path` fields point into `$RUN_ROOT/input/raw_frame_manifest/rgb`.

A standardized V19 run publishes the chosen rendered videos to canonical names only after the agent has selected the current physical branch and stated the claim scope. Prefer the publication helper because it adds a stable legend/metric banner and writes the source branch into a report:

```bash
python "$REPO_ROOT/scripts/publish_v19_render_artifact.py" \
  --overlay "$RUN_ROOT/renders/<chosen_branch>/$CASE_ID/<overlay_video>.mp4" \
  --world "$RUN_ROOT/renders/<chosen_branch>/$CASE_ID/<world_video>.mp4" \
  --side-by-side "$RUN_ROOT/renders/<chosen_branch>/$CASE_ID/<side_by_side_video>.mp4" \
  --interval-state "$RUN_ROOT/measurements/interval_mano/<chosen_branch>/$CASE_ID/v18_joint_mano_interval_trajectory_state.json" \
  --output-dir "$RUN_ROOT/renders/<published_branch>" \
  --canonical-dir "$RUN_ROOT/renders" \
  --replace-canonical \
  --title "V19 <claim-scope>"
```

The canonical names are:

```text
$RUN_ROOT/renders/v19_overlay.mp4
$RUN_ROOT/renders/v19_world.mp4
$RUN_ROOT/renders/v19_side_by_side.mp4
```

Publication is not physics progress; the visible physical content and the selected branch's render-consumed state are what matter. The report and banner must not relabel an uncertain MANO interval as accepted closure.

## 13. Visual consumption and repair loop

The agent must consume the final videos as a physical annotation before claiming progress. The review asks falsifiable questions:

- Do MANO joints/surfaces align with visible hands where visible?
- Are occluded hands shown as uncertain rather than hallucinated certainty?
- Does the completed rigid mesh project onto the object, not onto the hand/support/background?
- Does the world view agree with overlay depth/order?
- Does contact/near-contact/non-contact follow the rendered geometry and timing?
- Are nonpenetration corrections bounded, local, and compatible with visible 2D evidence?
- Where the render is wrong, which mechanism is most likely: mask identity, depth/scale, camera pose, mesh completion, pose fit, hand model, contact factor, occlusion factor, or renderer bug?

Repair is allowed only when it targets a named mechanism. Repeating validators or writing new status fields is not repair.

## 14. Ordered execution summary

1. Build the V19 raw-frame manifest from the input video with `scripts/build_v19_raw_frame_manifest.py`.
2. Run camera/depth/SLAM measurements.
3. Run hand candidate measurements and metric/depth refit.
4. Agent writes object plan and point prompts from visual evidence.
5. Run SAM2 multi-object masks/tracks.
6. Build the V19 base annotation/state backbone with `scripts/build_v19_base_annotations.py` from the fresh run-root measurements.
7. Lift masks to visible metric surfaces with `scripts/build_v19_visible_geometry_from_sam2_depth.py`; treat its centroid pose as initialization only.
8. Agent chooses physical branch with falsifiers.
9. For every rigid object: evidence crop -> TRELLIS -> metric completion/adaptation -> visible-frame pose -> object/MANO correction/factors -> hidden-volume validation.
10. Solve interval MANO over selected physical intervals with object/camera/depth/contact/visibility factors actually available.
11. Assemble renderable state from corrected physical variables.
12. Render full-duration overlay/world/side-by-side.
13. Visually consume the rendered artifact, identify mechanism failures, repair the causal mechanism, and rerender.
