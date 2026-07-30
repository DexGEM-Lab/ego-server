# Egocentric Locomanipulation Annotation Pipeline v1

## Scope

Pipeline v1 is the first end-to-end RGB-only annotation pipeline for EgoScale manipulation clips. The initial full deliveries were tomato kitchen clips because they were the first clips with all backends completed. Representative non-kitchen clips are tracked in `docs/representative_samples.md`; the trash-bag sample is the current full non-kitchen validation of the action-segment object front-end.

- `overlay_mano_object.mp4`: 960x540 source video render with MANO hand overlays, object mask/extent, and semantic caption.
- `reconstruction_3d_world.mp4`: clip-local 3D animation of DROID-SLAM head camera path, MANO hand joints/surface samples, and object centroid/extent.
- `side_by_side.mp4`: synchronized 1920x540 overlay and 3D reconstruction with the caption timeline.
- `annotations_v1_full.json`: per-source-frame hand, camera, object, caption, status, and QC-bearing fields.
- `qc_v1_full.json`: backend coverage, scale evidence, smoothing/prediction counts, and output paths.

The inspected task folders contain RGB video and action JSON only; ground-truth poses, depth stream, IMU, and camera calibration files are absent. v1 therefore reports a metric-like clip-local reconstruction anchored by hand anthropometry and monocular DROID depth. Absolute 5 mm certification requires explicit calibration, depth, fiducials, or measured object/scene priors.

## Delivered Examples

The final v1 deliveries are the full-MANO reruns under `/data2/ego_annotation_outputs`, produced after the earlier repo-local sampled-vertex validation folders.

Task7 tomato chopping/preparation:

`/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_7/20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4`

This is a 2040-frame, 1920x1080, 30 fps tomato preparation clip. The manipulated object interval is frame 312 through 1911 from the JSON action captions.

Final v1 outputs are under:

`/data2/ego_annotation_outputs/fullmesh_task7/fused/`

The final full run processed all 2040 source frames and produced 2040-frame videos at 30 fps.

Task5 tomato washing/peeling:

`/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4`

This is a 960-frame, 1920x1080, 30 fps tomato washing/peeling clip. The manipulated object interval is frame 270 through 939 from the JSON action captions.

Final v1 outputs are under:

`/data2/ego_annotation_outputs/fullmesh_task5/fused/`

The final full run processed all 960 source frames and produced 960-frame videos at 30 fps.

## Coordinate Contract

All 3D data is expressed in one DROID-derived clip-local world coordinate system:

- `T_world_camera`: camera-to-world transform per source frame.
- Hand joints and MANO vertices are first solved in source camera meters, then transformed by `T_world_camera`. The final `/data2/ego_annotation_outputs/fullmesh_task*` deliveries carry full 778-vertex MANO meshes per detected hand.
- Tomato objects are represented as deformable centroids with spherical extent because chopping changes topology and visible shape.
- Deformable objects can be rendered as visible surface patches from mask/bbox rays plus optimized depth/contact anchors. The JSON still stores centroid/depth/radius fields for continuity.
- Image coordinates remain in original 1920x1080 pixels inside JSON; rendered videos are 960x540 overlay and 1920x540 side-by-side.

World scale is estimated by aligning DROID relative depth to source-camera hand depths from WiLoR/MANO geometry. The QC stores the scale sample count, ratio IQR, and residual IQR because this scale anchor is approximate.

## Modules

### 1. Metadata And Captions

`run_v1_wilor_colmap.load_actions` reads the EgoScale JSON with `utf-8-sig` because some files contain a BOM. v1 uses the action segment text as the semantic caption source. The caption for each frame is copied from the active segment; no off-screen events are invented.

### 2. Head Camera Localization

`scripts/run_droid_full_frame.py` runs DROID-SLAM on every source frame at stride 1. The output contains:

- `droid_dense_trajectory.npz`
- `droid_dense_trajectory.json`
- `droid_keyframe_reconstruction.pth`
- `droid_keyframes.json`
- `droid_qc.json`

The final tomato run has 2040/2040 dense camera poses and 51 DROID keyframes. DROID depth is stored at network stride 8 and remains monocular relative depth until the fusion stage estimates a global scale.

### 3. MANO Hand Reconstruction

`scripts/run_wilor_full_frame.py` runs WiLoR on every source frame. The output contains full-frame hand detections, 2D joints, MANO camera-space joints, MANO parameters, vertices, side labels, and detector scores.

`scripts/fuse_v1_full_fidelity.py` converts WiLoR output into source-camera meters by:

1. Scaling WiLoR local hand geometry so median wrist-to-middle-tip distance matches 0.175 m.
2. Solving a least-squares source-camera translation from MANO 3D joints, 2D joints, and source intrinsics.
3. Rejecting physically invalid hand solves.
4. Running a constant-velocity RTS Kalman smoother per hand side.

Measured hand frames stay marked as measured; short occlusion/truncation gaps are marked as Kalman predictions. On the final tomato run, left/right hand median source-frame reprojection residuals were about 7.28 px and 9.91 px.

### 4. Object Segmentation And Tracking

The object front-end has two operating regimes. Tomato clips use color-refined SAM proposals because red material gives a strong object cue. Representative non-kitchen clips use action-segment object profiles, OWLv2 prompts, SAM masks, hand-contact scoring, temporal continuity, and deformable-size rejection.

For each semantic object frame, `scripts/fuse_v1_full_fidelity.py` builds object candidates from:

- caption-relevant hand geometry;
- compact tomato-red connected components near active hands or prior object state;
- OWLv2 tomato proposals;
- SAM ViT-B masks from candidate boxes;
- temporal prior from the previous accepted object state.

SAM candidates are scored by mask confidence, red content, hand-object contact, distance to action-relevant hands, and temporal consistency. The chosen mask is then refined:

- `chop_red_component_union` joins multiple high-confidence tomato-red components when chopping splits the visible object.
- `scrape_optical_flow_temporal_union` warps the previous accepted object mask with dense DIS optical flow during scraping so the chopped pile does not collapse to one small red patch when color saturation changes.
- Degenerate edge-clamped states are rejected instead of rendered.

The object track is smoothed with an RTS Kalman smoother over center, bbox, and area. Predicted frames remain marked as predictions. The final tomato run measured 1587 object frames, predicted 3, and rejected 8 invalid measurements.

### 5. Object World Pose Proxy

Tomato chopping makes a rigid 6D pose invalid. v1 therefore estimates a deformable object centroid and extent:

- The 2D object centroid defines a source-camera ray.
- DROID depth samples provide absolute depth rows when a nearby keyframe depth is available.
- Fingertip/contact anchors provide 3D contact constraints when hands are close to the object.
- Temporal smoothness regularizes the per-frame object depth.

The optimizer solves one depth variable per active object frame with sparse L-BFGS-B bounds. The final annotation stores:

- `center_source_camera_m`
- `center_world_m`
- `depth_m`
- `radius_m`
- `pose_type = deformable_object_centroid_with_spherical_extent`
- per-frame depth evidence flags

### 6. Rendering

The overlay renderer draws:

- MANO hand keypoints, skeletons, decimated MANO mesh edges, and projected vertex samples for left and right hands;
- object masks, extent boxes, and centroids;
- semantic caption text.

The 3D renderer draws:

- camera trajectory in DROID world coordinates;
- the current head camera as a frustum with camera-forward/up/right axes;
- current hand joints and MANO surface samples in world coordinates;
- object centroid/extent for compact objects or an object surface patch for deformable objects.

The renderer uses a head-local view for each frame so the egocentric hand-object interaction remains legible while the points stay in DROID world coordinates. The underlying JSON remains in DROID world coordinates; the rendered vertical axis is a display convention and should not be read as a calibrated gravity estimate.

The side-by-side video concatenates the overlay and 3D render at the same frame index, so one source frame corresponds to one output frame.

## Verification On Task7

Final video checks:

- `overlay_mano_object.mp4`: 960x540, 2040 frames, 30 fps, 68 s.
- `reconstruction_3d_world.mp4`: 960x540, 2040 frames, 30 fps, 68 s.
- `side_by_side.mp4`: 1920x540, 2040 frames, 30 fps, 68 s.

Representative visual inspections after the final full run:

- Frame 312: object mask is on the tomato in the container; the sink/edge artifact is absent.
- Frames 334-343: the tracker marks ten degenerate edge/occlusion states as unobserved and suppresses object rendering.
- Frame 600: object extent covers the intact tomato slice and the piece under hand/knife contact.
- Frame 1020: object extent covers the chopped tomato material on the board.
- Frame 1860: scrape phase tracks the chopped tomato pile.
- Frame 1910: optical-flow temporal union prevents collapse to a tiny contact patch and keeps the chopped pile extent.
- Frame 1980: object annotation is absent after the tomato semantic interval; only hand annotations remain.

## Second-Sample Validation

Task5 tests a different scene, sink reflections, water, and pre-contact object visibility.

Checks after the corrective rerun:

- DROID-SLAM produced 960/960 dense camera poses.
- WiLoR detected hands in 939/960 frames.
- The object module processed the semantic tomato interval frame 270 through 939, measured 666 frames, and marked the first four active frames as Kalman predictions from the first measured tomato state while keeping the visible tomato represented. The object world proxy covers all 670 semantic frames.
- Final videos are 960 frames at 30 fps: `overlay_mano_object.mp4` is 960x540, `reconstruction_3d_world.mp4` is 960x540, and `side_by_side.mp4` is 1920x540.
- Visual frames 270, 274, 480, 690, and 900 show object association on the tomato and hand overlays on the active hands.

## Representative Non-Kitchen Validation

The trash-bag sample under `/data2/ego_annotation_outputs/representative_trash/fused_bagprompt_full_final/` exercises large deformable object tracking outside the tomato/kitchen setting.

Checks after the final corrective rerun:

- DROID-SLAM produced 1050/1050 dense camera poses.
- WiLoR detected hands in 898/1050 frames.
- The object module tracked the action-segment `trash_bag` object with 807 measured frames, 3 predicted frames, and 13 rejected invalid/degenerated measurements.
- The world object stage produced 810 active surface/centroid states using 585 DROID-depth frames and 321 contact-anchor frames.
- Hand/object contact correction accepted 819 contact-depth measurements and updated 775 hand frames.
- Final videos are 1050 frames at 30 fps: `overlay_mano_object.mp4` is 960x540, `reconstruction_3d_world.mp4` is 960x540, and `side_by_side.mp4` is 1920x540.
- Visual frames 90, 269, 678, 900, 910, and 917 were inspected. Frames 910 and 917 correctly suppress object rendering after the visible target is gone.

## Failure Limits

v1 is a real pipeline, but it is still limited by RGB-only monocular evidence:

- DROID scale is inferred from hand geometry and relative depth; calibrated metric reconstruction requires an external metric source.
- Hand anthropometry introduces scale error because the actual subject hand size is unknown.
- The object pose is a centroid/extent proxy for a deformable object.
- The original tomato segmentation path uses object-specific color/semantic cues plus SAM and temporal motion. The generalized object front-end uses action-segment labels, OWLv2 prompts, SAM masks, hand-contact scoring, and temporal continuity; representative clips are needed to validate each object family.
- Contact constraints help stabilize object depth. Physical non-penetration proof requires a scene/object SDF.

These limits are recorded in QC and drive v2.

## Commands

Run DROID on the example:

```bash
PYTHONPATH=third_party/DROID-SLAM uv run --project . python scripts/run_droid_full_frame.py \
  --clip /data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_7/20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4 \
  --output-dir /data2/ego_annotation_outputs/fullmesh_task7/droid \
  --droid-area 98304
```

Run WiLoR on the example:

```bash
uv run --project . python scripts/run_wilor_full_frame.py \
  --clip /data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_7/20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4 \
  --output-dir /data2/ego_annotation_outputs/fullmesh_task7/wilor
```

Fuse and render:

```bash
PYTHONPATH=scripts uv run --project . python scripts/fuse_v1_full_fidelity.py \
  --clip /data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_7/20260118_1257_Rec3db6_P0_Sc6ab88_task_7.mp4 \
  --wilor-raw /data2/ego_annotation_outputs/fullmesh_task7/wilor/wilor_raw.json \
  --droid-npz /data2/ego_annotation_outputs/fullmesh_task7/droid/droid_dense_trajectory.npz \
  --droid-reconstruction /data2/ego_annotation_outputs/fullmesh_task7/droid/droid_keyframe_reconstruction.pth \
  --output-dir /data2/ego_annotation_outputs/fullmesh_task7/fused \
  --object-stride 1 \
  --render-width 960
```
