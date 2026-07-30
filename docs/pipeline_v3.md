# Pipeline V3: Referring Segmentation and Joint Metric Contact

## Why V3 Exists

V2 replaced object proxies with model-planned masks and observed-surface mesh evidence. It proved that the pipeline can acquire clean object masks and measured visible surfaces, but it also exposed a larger 3D inconsistency:

- The accepted pink-lid mask is visually stable.
- Metric-depth meshing can fit the visible surface in image space.
- MANO vertices often project near the object mask in 2D.
- The same vertices are hundreds of millimeters away from the object surface in camera depth.

The compact contact-depth report for frames 840 to 930 has 84 frames where hand vertices project near the object mask:

- median hand-minus-object depth gap: 0.385 m
- p95 hand-minus-object depth gap: 0.667 m
- median object-over-hand depth ratio: 0.741
- p05/p95 object-over-hand depth ratio: 0.472 / 0.937

This magnitude cannot be fixed by a Kalman smoother or an object-pose-only optimizer. V3 must jointly reason about object identity, object mesh, MANO metric scale/depth, camera pose scale, camera intrinsics, metric depth reliability, and contact state. A visible surface patch or a clean 2D mask is evidence, not a complete manipulated-object annotation.

A lightweight 1D contact-depth diagnostic then solved only the depth gap for frames with at least 80 near-mask hand vertices. This diagnostic is intentionally underdetermined: each frame has one contact-depth equation and four correction variables before priors. It reduced median corrected depth gap to 0.16 mm, but the inferred correction was not small:

- median hand-depth scale: 0.939
- median object-depth scale: 1.217
- p95 object-depth scale: 1.581
- median object-depth shift: 43 mm

This is useful as a causal diagnostic, not as an annotation result. It shows that contact can be made numerically true when scale and shift corrections are explicit variables, while the true faulty subsystem remains unidentified. A hidden correction would destroy the evidence about which subsystem is wrong.

## Perception Branch

The white-liner track remains the hardest object-perception case in the representative trash clip. The rejected evidence is explicit:

- OWLv2 plus SAM selected the pink lid or background for the white-bag track.
- VLM point prompts plus SAM improved selected frames but spilled onto floor glare, clothing, and the pink lid.
- Metric-depth component cleanup failed because the wrong lid/background components can have plausible monocular depth.
- SAM2 propagation from weak point-prompt seeds tracked floor glare or the pink lid.

The next valid mask model must be language-conditioned across video, not another hand-written cleanup rule.

### SAM3

SAM3/SAM3.1 is the preferred direct interface because the official repository supports image and video segmentation from text prompts. SAM3.1 adds multiplex video tracking for multiple objects. Current environment access blocks this path:

- `facebook/sam3` metadata is visible, but `sam3.pt` download returns Hugging Face gated-access 403.
- `facebook/sam3.1` metadata is visible, but `sam3.1_multiplex.pt` download returns gated-access 403.

Implemented preparation:

- `scripts/run_sam3_referring_track.py`

This runner should be used after checkpoint access is granted. It consumes a video window, text prompt, optional VLM point prompts, and writes mask PNGs, review stills, overlay video, and QC JSON.

### SAMWISE

SAMWISE is the current executable language-conditioned video segmentation path. It is a CVPR 2025 text-driven video segmentation model built on SAM2 and provides arbitrary-video/frame-folder inference from natural-language prompts.

Implemented preparation:

- `scripts/run_samwise_referring_masks.py`
- `scripts/remote_setup_samwise.sh`

Executed run:

- input: frames 840 to 930 of the representative trash clip, with the mesh branch currently using the clean 858 to 880 subwindow
- accepted prompt result: clean object-level masks for the central pink lid over frames 858 to 880
- rejected point-prompt surface result: SAM2 point-prompt variants for rim, flange, and liner repeatedly collapsed into broad lid/background regions and must not be meshed
- rejected text-only contact-surface result: SAMWISE prompts for the white liner draped edge, pink lid perimeter rim, and can opening rim were run on frames 840 to 930. Visual inspection rejects all three as mesh inputs. The liner and can-rim prompts selected the same magenta background or occluded region in frames 840 to 858 and then disappeared. The pink-rim prompt selected small off-rim/background patches.
- current target: image-conditioned VLM point prompts for the actual contact objects, especially the translucent white liner draped edge and the can/lid perimeter surfaces

The pink-lid SAMWISE result is useful context geometry, but the contact-surface plan says the broad central lid panel is mostly a visible separator/support surface. Hand contact in this window belongs to perimeter/rim/liner pixels. The next SAMWISE run must therefore target those surfaces directly instead of refining the central lid mask again.

After the failed contact-surface SAMWISE run, the next segmentation branch should not be another text-prompt wording change. The valid source of difference is model-produced per-frame visual evidence: VLM-selected positive and negative points, SAM masks constrained by those points on the same image, and VLM/visual review of the resulting masks. Long propagation from weak sparse seeds is rejected for this clip because prior SAM2 propagation drifted to floor, wall, and lid pixels.

Image-conditioned SAM2 was then run per prompted frame. The strict selector accepted only 3 white-liner frames, 4 can-rim frames, and zero annular-rim/flange frames. Candidate-mask review explained the failure: narrow rim/liner prompts deliberately place positive points on multiple disconnected visible fragments, while SAM2 returns either one local fragment that misses other positives or one large whole-lid mask that includes negative points. The strict all-positive contract correctly rejects whole-lid masks, but it is the wrong representation for local contact fragments.

A fragment selector was added with explicit thresholds: at least half of positive points, zero negative hits, and a maximum 8 percent image-area cap. This yields local observed fragments for the liner, can rim, flange, and occasional annular rim. Visual review accepts some fragments as local surface evidence, but rejects others as full-object geometry: for example, frame 886 white-liner selection expands onto the pink lid. The fragment branch is therefore contact evidence only, not a complete mesh branch.

### SOLA

SOLA is a secondary text-to-track candidate. It generates SAM2 tracks and selects tracks by language alignment. Its public instructions are organized around MeViS and Ref-Youtube-VOS dataset-format track generation, so it is less direct than SAMWISE for immediate custom-video inference.

## Mesh Branch

V2 observed-surface meshes are necessary but incomplete: they reconstruct only the visible surface from mask and depth. V3 needs an object-centric complete mesh state.

Current evidence:

- TripoSR produced a complete mesh prior for frame 858 from the accepted pink-lid crop.
- Strict frame-858 alignment to the observed metric-depth surface reached median prior-to-observed and observed-to-prior distances near 10 mm.
- The 858-930 window optimizer reduced object surface residuals but failed contact consistency.
- VGGT masked multiview geometry over frames 858 to 880 produced noncollapsed raw object points, but camera-motion Sim3 scaling collapsed the object to millimeters. Rescaling from the observed surface extent produced a 0.57 m by 0.66 m by 0.42 m rounded mesh with 71 mm median point-to-observed distance and a visual blob around the lid, so it is rejected as an annotation mesh.
- Hunyuan3D-2mv multiview generation from four depth-grown object crops ran successfully on A800 and produced a 369k-vertex mesh. Frame 858/878 alignment again reached low nearest-surface medians, but visual QC rejects the result as a generic oval shell with fuzzy side growth rather than scene-faithful lid/can geometry.
- SAMWISE text-conditioned segmentation produced clean object-level lid masks for frames 858 to 880. A dense mask-depth height-field mesh from those masks reached median silhouette-mask IoU 0.9965 and median vertex-depth error 0.24 mm across 23 frames. This establishes a valid visible-surface mesh observation for the lid.
- The same height-field archive fails temporal object consistency. Its robust camera-frame Z extent changes up to 2.50x relative to the median, and the world center speed reaches 8.79 m/s between adjacent video frames. This is a metric-depth/camera-scale inconsistency, because the per-frame projection evidence is already strong.
- A one-variable-per-frame depth-scale graph regularized object XY extent to within about 6 percent of the median, but projection-depth error rose to 36.7 mm median and 293 mm on frame 880. The tradeoff exposes the missing constraint: V3 needs independent metric camera/depth scale evidence before rigid object pose can be trusted.
- Full-scene VGGT was run on frames 858 to 880 using the original scene frames and SAMWISE masks only for object-point selection. This is stronger evidence than the rejected object-crop VGGT branch because the model sees the room and camera motion. VGGT camera centers align to the DROID trajectory with 10.5 mm median error after a Sim3, but the required VGGT-to-DROID scale is 0.129 and the VGGT-predicted focal maps back to about 1200 px in the 1920-wide source frame, far from the DROID prior of 2304 px.
- VGGT object points project entirely inside the SAMWISE mask after correcting the aspect-ratio-preserving resize and padding. They expose a concrete Depth Anything failure: in frames 878 to 880, VGGT places the lid surface at about 0.69 to 0.75 m while Depth Anything places the same mask at about 1.01 to 1.12 m. This explains the late-frame heightfield expansion without blaming the mask.
- A VGGT observed-surface mesh built from the selected points is temporally coherent: robust camera extents are 0.050 x 0.034 x 0.044 m, extent-ratio max-log median is 0.049, and pairwise center speed median is 0.056 m/s. This is still only a compact visible-surface patch, not a full lid mesh or watertight object.
- Re-solving MANO translations under VGGT intrinsics reduces the high-score contact gap to 86 mm median, but does not close it. A bounded temporal Z-shift/contact optimizer can drive median contact gap near zero only by hitting the 35 cm shift bound and producing frame-880 median keypoint reprojection error of 80 px with an implausibly small hand scale. VGGT reduces the object-depth error, but current MANO measurements remain inconsistent with 5 mm contact annotation.
- A source-focal sweep over the VGGT object surface shows that focal length is an active variable, not a harmless visualization parameter. VGGT predicts about 1191 px source focal, where the high-score hand rows still sit about 100 mm in front of the object surface. Around 1400 px the median hand/object depth gap crosses near zero with 7.0 px median keypoint reprojection, but the p95 absolute surface gap remains 68.5 mm and about 32 percent of near-mask vertices violate the surface by more than 10 mm. The DROID prior focal of 2304 px is incompatible with this contact interpretation: the median gap becomes about 425 mm behind the surface.
- The focal/hand/contact graph optimizes one source focal, one hand scale, per-observation depth shifts/velocities, and per-observation contact probability. It keeps keypoint reprojection reasonable at 7.8 px median and removes positive surface violation with small 15 mm median shifts, but it does so by driving contact probability to 0.029 median and leaving only 37 percent of near-mask vertices within 30 mm. The graph therefore rejects broad pink-lid contact rather than accepting a false physical annotation.
- TRELLIS produced the strongest single-image complete-mesh prior so far. The raw mesh has lid-like shape and side tabs. Aligning it to the Depth Anything late-thin frame-880 surface gave median prior-to-observed and observed-to-prior distances of 19.9 mm and 12.9 mm, but late-window pose graphs still required 1.7 to 2.0 m/s object center speed or implausible camera corrections. The failure mechanism was traced to mixed camera/depth evidence: the prior was anchored to a Depth Anything surface around 1.12 m camera depth, while VGGT places the same visible patch around 0.69 to 0.75 m before Sim3 scaling.
- A full-SE3 VGGT camera adapter was added for frames 878 to 880. It changes camera centers by only 12 to 27 mm, but replaces the DROID rotation with a smooth VGGT trajectory: center speed becomes 0.052 m/s and angular speed becomes 0.128 rad/s. Running the TRELLIS object graph with this camera pose while keeping Depth Anything surfaces still leaves about 2.1 m/s object speed, so camera pose alone does not close the branch.
- A VGGT observed-surface exporter exposed the current metric-scale ambiguity. VGGT camera-coordinate depths are internally consistent with the predicted focal and image size, but they are in VGGT units. Multiplying by the VGGT-to-DROID Sim3 scale puts the frame-878 to 880 visible patch into the existing DROID-aligned world with robust camera extents about 47 x 51 x 11 mm and median camera depth about 93 mm. That scale is useful for comparing against the existing annotation world, but it should not be accepted as ground-truth metric scale.
- Aligning the raw TRELLIS mesh to the corrected VGGT frame-880 patch reaches millimeter surface residuals: 3.4 mm median prior-to-observed, 1.6 mm median observed-to-prior, and 8.5 mm p95 prior-to-observed. The required Sim3 scale is 0.086 of the TRELLIS model. A late-window graph with fixed anchor-visible-patch matching then tracks the VGGT patch with 1.9 mm median and 4.5 mm p95 observed-to-prior residual, and object center speed about 0.098 m/s. This is accepted only as visible-surface tracking evidence. It is rejected as complete object annotation because the resulting mesh extent is patch-sized, about 4 to 6 cm across, and cannot represent the manipulated trash-can lid/can object.
- A static complete-mesh scale sweep then anchored one TRELLIS mesh to the VGGT visible patch at frame 880 and projected the same world mesh through frames 878 to 880 while varying the VGGT metric scale from 0.129 to 1.6. This directly tests whether a single global scale can reconcile complete mesh, visible patch, mask silhouette, and depth. It cannot. The current reproducible run keeps median silhouette IoU nearly flat at about 0.74 to 0.75, and projected vertex-in-mask fraction stays around 0.89 across the scale range, so the silhouette cannot identify metric scale in this anchored setup. The visible-patch optimum is the Sim3 scale 0.129, which gives 1.7 mm median observed-to-prior residual but shrinks the mesh to a 49 x 52 x 17 mm robust extent and leaves 0.91 m median absolute Depth Anything residual. Depth Anything selects scale 1.4, giving a 526 x 565 x 182 mm robust extent, 17.5 mm patch residual, and 69 mm median absolute depth residual. The report is `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale_static_mesh_sweep_depthpro_878_880/qc_scale_static_mesh_sweep_v3.json`.
- Depth Pro was tested as an independent RGB-only metric depth and focal source on the same frames. It predicts source focal around 1500 px and mask depth around 0.40 to 0.50 m, giving apparent visible-lid widths around 0.21 to 0.26 m. Adding Depth Pro to the same static complete-mesh sweep selects scale 0.55, with a 208 x 222 x 72 mm robust mesh extent, 7.0 mm visible-patch residual, and 25 mm median absolute Depth Pro residual. That scale still leaves about 596 mm median absolute Depth Anything residual. The Depth Pro report is `/data2/ego_annotation_outputs/representative_trash/v3_depthpro_metric_source_878_880/qc_depthpro_metric_source_v3.json`.
- UniDepth was then tested as a second independent RGB metric-depth/focal source on the same frame-878 to 880 window. It predicts source focal around 1236 px, mask depth around 0.398 m, and apparent visible-lid width around 0.236 m. The corrected archive stores UniDepth's full `fx, fy, cx, cy` per frame. In the same static complete-mesh sweep, UniDepth also selects scale 0.55, with 19 mm median absolute UniDepth depth residual. This agreement with Depth Pro makes the 0.55 scale a real hypothesis, not a single-model artifact. It still contradicts the manifest Depth Anything optimum at scale 1.4 and the visible-patch optimum at scale 0.129, so it is scale evidence rather than V3 closure. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_unidepth_metric_source_878_880/qc_unidepth_metric_source_v3.json` and `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale_static_mesh_sweep_depthsources_878_880/qc_scale_static_mesh_sweep_v3.json`.
- The scale-0.55 static complete mesh was then packaged as a world mesh archive using the VGGT custom-anchor camera poses. Projection QC must use the same camera/depth contract that produced the hypothesis. Under old manifest intrinsics and Depth Anything depth, the mesh looks like a large translucent sheet and fails with median IoU 0.377 and about 0.57 m median depth error. Under VGGT source intrinsics and UniDepth depth, the same archive reaches median IoU 0.736, projected-vertex mask fraction 0.890, and 24 mm median depth error. The object hypothesis is therefore plausible under the RGB metric-depth contract, but it still gives zero reliable contact rows and about 0.82 m hand/object contact gap under the current MANO stream. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale055_static_mesh_projection_qc_vggtK_unidepth_878_880/qc_bundlesdf_projection_v3.json` and `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale055_static_mesh_contact_unidepth_878_880.json`.
- UniDepth was extended to the full 858 to 880 V3 object window. The full-window source predicts median focal 1410 px and median lid-mask depth 0.408 m. A hand-depth/contact graph using the scale-0.55 world mesh, VGGT annotation intrinsics, UniDepth depth, and RTMLib/WiLoR association found zero valid hand observations. The skipped rows are 31 predicted hands, 11 missing RTMLib/WiLoR matches, 3 low WiLoR scores, and 1 row with too few initial 2D keypoints. This localizes the immediate V3 blocker to the hand observation stage: the scale-0.55 object hypothesis cannot be accepted or rejected by contact until a measured, temporally supported MANO hand stream exists for the same frames. The diagnostic is `/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_scale055_unidepth_858_880/vggt_intrinsics_emptydiag.json`.
- Existing alternative hand streams were then evaluated under the same target geometry contract instead of their older V2 depth reports: scale-0.55 VGGT object poses, VGGT source intrinsics, and UniDepth depth on frames 858 to 880. Direct V3 acceptance fails for WiLoR, HaWoR, EgoForce detector, EgoForce posehead, and HandDGP. Measured-high-score target reprojection medians are about 184 px, 178 px, 152 px, 255 px, and 184 px, respectively, with zero reliable contact rows across 134 rows. HandDGP has the smallest median broad-mask contact gap, about 101 mm after target projection, although several rows fail hand-scale plausibility. The comparison report is `/data2/ego_annotation_outputs/representative_trash/v3_hand_stream_compare_scale055_unidepth_858_880.json`.
- A target-camera MANO similarity refit was added to test a stronger hand observation mechanism with contact kept as a checked residual. It treats each measured posed MANO cloud as a template and solves a camera-space similarity transform under VGGT intrinsics, 2D keypoints, UniDepth keypoint depths, hand-size priors, and temporal smoothness. WiLoR is the strongest source after refit: measured rows reach 7.1 px median reprojection, 7.5 mm median UniDepth residual, 149 mm median hand-bone scale, and four reliable rows against the broad scale-0.55 object mask. Visual review accepts the refit projection on the visible hand for frames 868, 869, and 879; frame 859 remains broad-mask proximity. The refit report is `/data2/ego_annotation_outputs/representative_trash/v3_wilor_target_similarity_refit_scale055_unidepth_858_880/qc_refit.json`; broad-mask contact QC is `/data2/ego_annotation_outputs/representative_trash/v3_wilor_target_similarity_refit_scale055_unidepth_858_880/contact_qc.json`; review stills are under `/data2/ego_annotation_outputs/representative_trash/v3_wilor_target_similarity_refit_scale055_unidepth_858_880/review_stills/`.
- Measured 2D hand-keypoint continuity was then used to associate the WiLoR rows into temporal tracks before contact reasoning. This correctly links the late visible interaction across frames 877 to 880 and exposes a side-label flip: the same geometric track appears as right, left, left, then right. Track-aware contact prevents side labels from creating artificial temporal breaks, but it still leaves late frames without compact stable surface support.
- The same WiLoR-refit hand stream produces zero reliable rows against the VLM/SAM2 fragments and full-scene VGGT surface points. Over 858 to 880, the best focal hypothesis, 1800 px, gives best-patch surface-distance p95 median about 62 mm and lacks temporal surface support. The refit hand stream is therefore a usable measured hand observation branch, while physical contact remains limited by missing or wrong contact-surface masks/mesh regions. The surface report is `/data2/ego_annotation_outputs/representative_trash/v3_wilor_refit_vggt_fragment_contact_858_880.json`.
- The refit-supported frames 868 and 869 were rechecked with the per-frame SAM2 image predictor using the previously working HaWoR virtualenv on A800, through tmux. The outer-flange masks pass the point-hit contract but fail visual QC because they cover the lid face, hand/skin, floor, and leg texture. The annular-rim candidates look closer to a rim fragment but fail the strict point contract and include off-rim components. These masks are rejected as mesh inputs and recorded under `/data2/ego_annotation_outputs/representative_trash/v3_refit_contact_surface_sam2_image_868_869/`.
- A mesh-surface contact diagnostic then tested the WiLoR-refit hands against the actual scale-0.55 object mesh instead of SAM2 surface-name fragments. It queries nearest object-mesh surface vertices in the current camera frame, estimates signed separation from mesh normals oriented toward the camera, and requires 2D reprojection, UniDepth hand depth, bone scale, patch spread, penetration fraction, and MANO-local temporal continuity to agree. The first run finds a three-frame right-hand contact episode on frames 866 to 868. The report is `/data2/ego_annotation_outputs/representative_trash/v3_wilor_refit_scale055_mesh_surface_contact_858_880.json`.
- A first local MANO articulation refit was rejected because it used a generic SMPL-X axis-angle MANO convention that did not reproduce WiLoR's saved vertices. Zero-state reconstruction errors were tens to hundreds of millimeters, the optimizer saturated pose deltas at 0.8 rad, median reprojection worsened to 26.6 px, and strict mesh-surface contact dropped to zero reliable temporal rows.
- The corrected MANO articulation refit uses WiLoR's MANO wrapper, rotation matrices with `pose2rot=False`, the saved right-hand MANO convention, explicit side x-sign mirroring, and zero-state vertex/joint checks before optimization. On the track-aware WiLoR refit stream it reaches 3.7 px median measured-hand reprojection, -6.2 mm median MANO-minus-UniDepth depth residual, and 3.9 mm median contact gap. The first corrected mesh-surface diagnostic finds five reliable temporal rows on frames 862, 864, 865, 866, and 867. The corrected reports are `/data2/ego_annotation_outputs/representative_trash/v3_wilor_track_pose_refit_wilorconv_scale055_unidepth_858_880/qc_pose_refit.json` and `/data2/ego_annotation_outputs/representative_trash/v3_wilor_track_pose_refit_wilorconv_scale055_mesh_surface_contact_858_880.json`.
- The mesh-surface contact diagnostic was then extended from one global nearest-vertex patch to two category-agnostic patch hypotheses: a compact global hand patch and a compact MANO anatomical patch. The anatomical patch partitions MANO vertices by nearest local hand joint, then requires the same track and same anatomical region to have temporal support. This is a dexterous-hand prior, not an object-family rule. The raw MANO-local anatomical diagnostic finds six reliable temporal contact rows on frames 862, 864, 865, 866, 867, and 868. Over reliable rows, median reprojection is 3.27 px, median MANO-minus-UniDepth depth is -11.5 mm, median best-patch p95 surface distance is 0.87 mm, median signed p95 absolute distance is 0.81 mm, and penetration fraction remains 0. The report is `/data2/ego_annotation_outputs/representative_trash/v3_wilor_track_pose_refit_wilorconv_scale055_mesh_surface_contact_anatomical_858_880.json`.
- A second anatomical diagnostic measures temporal drift relative to each finger region's distal MANO joint. This is more appropriate under side/canonical changes because the contact patch should move with the finger segment, not with a frame-specific global hand coordinate origin. With strict detector score threshold 0.50, the contact-aware refit found ten reliable temporal contact rows on frames 862, 864, 865, 866, 867, 868, 869, 870, 879, and 880. That result is now treated as an over-claim because the same stream had a -28.8 mm median MANO-minus-UniDepth depth bias on frame 880, and a contact factor can pull a hand toward the mesh while preserving plausible 2D reprojection. The report remains useful as a falsified branch: `/data2/ego_annotation_outputs/representative_trash/v3_wilor_track_pose_refit_wilorconv_scale055_mesh_surface_contact_anatomical_anchorrel_858_880.json`.
- A side-hypothesis diagnostic now evaluates left and right MANO mirror hypotheses from WiLoR's saved rotation matrices under the same 2D keypoints and UniDepth samples. The opposite-side hypotheses fail badly: on frame 880 the right hypothesis gives 5.8 px median reprojection and -4.0 mm median depth bias, while the forced-left hypothesis gives 73.2 px median reprojection. Across all measured rows in frames 858 to 880, the selected side always equals the stored WiLoR side. This falsifies the simple "wrong label" explanation. The report is `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/qc_side_metric_refit.json`.
- The selected side-hypothesis fit was materialized as a metric-side MANO stream and rechecked against the scale-0.55 object mesh. It gives 7 reliable temporal contact rows on frames 864 to 870. Over those rows, median reprojection is 3.12 px, median MANO-minus-UniDepth depth is -0.85 mm, median best-patch p95 surface distance is 4.77 mm, median signed p95 absolute distance is 4.68 mm, and penetration fraction remains 0. Frames 879 and 880 remain visually plausible proximity rows, but they no longer have stable patch temporal support, so they are not contact annotations. The report is `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/mesh_surface_contact_anatomical_anchorrel.json`.
- A stricter temporal setting now requires consecutive support frames. It keeps six contact rows on frames 865 to 870 and improves the reliable median MANO-minus-UniDepth depth residual to 0.54 mm, median best-patch p95 distance to 3.02 mm, and median signed p95 absolute distance to 2.92 mm. A contact-kinematics diagnostic then recomputes nearest object surface patches and compares hand-patch and object-patch world motion. The strongest physically coherent contact is the middle-finger sub-episode on frames 868 to 869, where the relative patch step is 0.29 mm. Other accepted rows remain geometric contact/proximity evidence because the selected patch source or anatomical region changes across time. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/mesh_surface_contact_anatomical_anchorrel_consecutive.json` and `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/contact_kinematics_consecutive_qc.json`.
- A contact-patch object pose graph now tests the strict contact rows as factors on a complete watertight TRELLIS mesh state instead of the earlier per-frame static camera mesh archive. The graph uses the scale-0.55 complete mesh prior, SAMWISE mask, UniDepth depth, annotation VGGT intrinsics, object temporal smoothness, watertight signed-distance penetration QC, and the six strict MANO contact-patch rows on frames 865 to 870. With dense complete-mesh surface sampling, the graph reaches good silhouette/contact/world-motion numbers: median projection-inside-mask fraction 0.949, median world center speed 0.208 m/s, median contact p95 4.71 mm, and median penetration fraction 0. But it remains diagnostic, not annotation-ready, because median observed-to-prior p95 is 118 mm and front-depth p95 is 50 mm. Weakening only the front-depth factor makes silhouette perfect and keeps contact near threshold, but observed-to-prior p95 stays above 115 mm. This isolates the current V3 blocker: the complete mesh prior can explain silhouette, temporal motion, and local contact, but its hidden/backside geometry is inconsistent with the visible UniDepth-derived surface. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_vggtK_dense_865_870/qc_contact_patch_object_pose_graph_v3.json` and `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_vggtK_dense_frontweak_865_870/qc_contact_patch_object_pose_graph_v3.json`.
- Projection QC and surface-conflict diagnosis make the failure visible. The graph mesh projects mostly inside the object mask, but rendered IoU is only about 0.55 because the mesh covers the central grey/green panel and tab while missing much of the orange rim/flange inside the SAMWISE object mask. Pixel-depth QC gives median absolute depth error 37 mm and p95 124 mm. A sampled conflict diagnostic gives median observed-to-graph distance 28 mm and p95 124 mm over 96k visible mask-depth samples. The failed assumption is now specific: the frame-880 TRELLIS prior is a plausible central-surface/contact prior, but it is not a complete mesh for the full manipulated object region. Projection QC is `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_vggtK_dense_865_870_projection_qc/qc_bundlesdf_projection_v3.json`; conflict QC is `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_vggtK_dense_865_870/complete_mesh_surface_conflict_qc.json`.
- A current-contract observed-surface fusion branch now uses the already solved contact-patch graph poses to pull SAMWISE plus UniDepth mask-depth samples into the complete-prior coordinate frame, then reconstructs a mesh from those observed samples plus the TRELLIS prior. This is category-agnostic: the input is model-produced mask/depth/pose evidence, not an object-family primitive. On frames 865 to 870, the fused mesh graph reduces median observed-to-mesh p95 from 118 mm to 12.3 mm, keeps median contact p95 at 5.13 mm, and keeps median object center speed at 0.256 m/s. Projection QC improves from 0.55 to 0.764 median IoU and from 37 mm to 13.5 mm median vertex-depth absolute error. The sampled visible-surface conflict improves from 124 mm p95 to 14.7 mm p95. This branch repairs the missing rim/flange surface mechanism, but it still is not V3 closure because the Poisson fused mesh is not watertight, signed penetration is unsupported, and vertex-depth p95 remains about 49 to 55 mm. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_observed_prior_fused_mesh_smoke_865_870/qc_observed_prior_fused_mesh_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_865_870/qc_contact_patch_object_pose_graph_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_865_870_projection_qc/qc_bundlesdf_projection_v3.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_865_870/complete_mesh_surface_conflict_qc.json`.
- A multiview mask-support filter was then applied to the fused mesh. Requiring five of six graph frames to support a mesh face removes most of the rendered halo and improves projection QC to 0.823 median IoU, 0.981 median projected-vertex mask fraction, and 13.6 mm median vertex-depth absolute error. The sampled observed-to-mesh surface p95 stays low at 15.1 mm. The graph keeps plausible contact and motion with median contact p95 4.52 mm and median object center speed 0.225 m/s. The mesh remains open, so a conservative voxel SDF diagnostic was added for contact penetration. Nearest-voxel SDF at 3 mm pitch overreported shallow penetration. After switching the diagnostic and graph to trilinear SDF interpolation and checking at 1.5 mm pitch, the strict filtered graph reports zero penetration beyond 2 mm, 1.43 mm median signed clearance, and 96.9 percent of contact-patch points within 6 mm. An SDF-aware graph keeps zero penetration beyond 2 mm, 1.44 mm median signed clearance, and 100 percent of contact-patch points within 6 mm, while maintaining 0.824 median projection IoU, 0.991 median projected-vertex mask fraction, 14.8 mm median vertex-depth absolute error, 5.39 mm median contact p95, and 0.223 m/s median object center speed. This supports nonpenetrating contact under a conservative volume-SDF tolerance, but it still is not a watertight mesh. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_observed_prior_fused_mesh_mask_supported_strict_865_870/qc_mask_supported_mesh_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_strict_865_870_projection_qc/qc_bundlesdf_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_strict_865_870/complete_mesh_surface_conflict_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_strict_865_870/volume_sdf_contact_trilinear_pitch0015_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_sdf_trilinear_865_870/qc_contact_patch_object_pose_graph_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_sdf_trilinear_865_870_projection_qc/qc_bundlesdf_projection_v3.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_sdf_trilinear_865_870/volume_sdf_contact_pitch0015_qc.json`.
- The SDF-aware fused-mesh contact interval was rendered as an inspected deliverable slice. The overlay video covers exactly the six solved object-pose/contact frames, 865 to 870, at 2 fps so each annotation state is readable: `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_review_sdf_fused_slow_865_870/mesh_surface_contact_review.mp4`. The world-coordinate side-by-side video renders the same frames with the current head-camera frustum, MANO hand, fused object mesh, and semantic caption: `/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_sdf_fused_slow_865_870/world_reconstruction_side_by_side.mp4`. The local contact side-by-side uses a decimated display copy of the same fused mesh so the 3D object appears as a shaded surface instead of a sparse triangle plot: `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_3d_sdf_fused_decimated_865_870/mesh_surface_contact_side_by_side.mp4`. Structural QC confirms six frames and three-second duration for each video. Visual QC of frame 868 accepts the object overlay, MANO projection, contact patch, head-camera frustum, and shaded local 3D object as readable. The render intentionally does not extend beyond frames 865 to 870 because the SDF-aware object-pose graph currently owns only that interval.
- The open fused mesh was then converted into an explicit closed mesh archive by exporting the same 1.5 mm filled-volume representation used by the contact SDF diagnostic. This measured-geometry closure consumes the existing mesh archive, per-frame camera pose, and voxel fill; it records discarded disconnected voxel artifacts; and it fails if the delivered mesh remains open. The accepted sigma-0.25 SDF level-set archive is watertight in all six frames, with zero boundary and non-manifold edges, median 562k vertices, median 1.13M faces, and maximum discarded component area fraction below 0.008 percent. The closed archive preserves contact: 32 contact-patch samples have 1.63 mm median absolute SDF distance, 3.48 mm p95 absolute SDF distance, zero penetration beyond 2 mm, and 100 percent near-surface support within 6 mm. All-face corrected z-buffer QC shows that closure improves topology and preserves near-surface contact while keeping strong rendered mask support: median rendered silhouette IoU is 0.940, median visible-silhouette-inside-mask fraction is 0.978, sampled visible-surface depth error is 19.9 mm median and 94.1 mm p95, and z-buffer depth error is 17.0 mm median and 124 mm p95. The closed archive is therefore a stronger topology/contact evidence slice, while the remaining visible-depth spread is still a V3 blocker. Topology and QC reports are `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_865_870/qc_voxel_closed_mesh_archive_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_865_870/complete_mesh_surface_conflict_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_865_870/volume_sdf_contact_pitch0015_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_zbuffer_interp_allfaces_qc_865_870/qc_mesh_zbuffer_projection_v3.json`.
- The closed-mesh evidence was rendered through manifest-backed RGB frames so the presentation source matches the mask/depth/mesh evidence path. The overlay video is `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_review_closed_sdf_levelset_865_870/mesh_surface_contact_review.mp4`; the world-coordinate side-by-side video is `/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_closed_sdf_levelset_865_870/world_reconstruction_side_by_side.mp4`; the higher-fidelity local 3D contact video is `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_3d_closed_sdf_levelset_display20k_865_870/mesh_surface_contact_3d.mp4`; and the local contact side-by-side is `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_3d_closed_sdf_levelset_display20k_865_870/mesh_surface_contact_side_by_side.mp4`. Structural QC confirms six frames, 2 fps, and three-second duration for each video. Visual QC of frame 868 accepts the world side-by-side as a readable stakeholder presentation of the current solved slice: MANO hand, contact patch, closed lid mesh, head-camera frustum, metric scale, and semantic caption are visible. The local 3D panel remains an auxiliary contact view because display decimation can expose small holes that are absent from the watertight source archive.
- Signed z-buffer QC localizes the remaining visible-depth failure. The dense closed mesh does not fail only at the mask rim: even pixels more than 40 px inside the object mask are mostly closer to the camera than UniDepth on frames 865, 866, 867, 868, and 869. The all-face signed-band report is `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_zbuffer_signed_bands_allfaces_qc_865_870/qc_mesh_zbuffer_projection_v3.json`. Switching the object graph to UniDepth's own metric-depth intrinsics was also falsified: sampled contact improves, but projection-inside-mask falls to about 0.68 and silhouette distance saturates. That report is `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_observed_prior_fused_masksupport_sdf_trilinear_metricK_865_870/qc_contact_patch_object_pose_graph_v3.json`. The camera contract for stakeholder renders remains annotation-VGGT.
- A sampled visible-depth residual was added to the contact-patch object pose graph. It approximates rendered visible depth by projecting object-surface samples, keeping the nearest sample per rounded image pixel, and penalizing signed depth error inside the model-produced mask. On frames 865 to 870, the sampled visible-depth graph keeps silhouette and contact plausible, reduces sampled visible-depth median absolute error from 22.3 mm to 15.2 mm, and reduces sampled p95 from 94.8 mm to 82.6 mm. After converting that pose archive to the same closed SDF-level-set mesh, full all-face z-buffer QC still reports 18.7 mm median and 121.7 mm p95 absolute depth error, and contact SDF shows one shallow penetration sample beyond 2 mm. The graph, closure, contact, and z-buffer reports are `/data2/ego_annotation_outputs/representative_trash/v3_contact_patch_object_pose_graph_visible_depth_865_870/qc_contact_patch_object_pose_graph_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_visible_depth_sigma025_865_870/qc_voxel_closed_mesh_archive_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_visible_depth_sigma025_865_870/volume_sdf_contact_pitch0015_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_visible_depth_sigma025_zbuffer_signed_bands_allfaces_qc_865_870/qc_mesh_zbuffer_projection_v3.json`. The result falsifies pose-only repair for the current mesh.
- A category-agnostic visual-hull plus metric-depth carving branch was then tested. It reconstructs a closed object-centric occupancy volume from model-produced masks, UniDepth depth, annotation-VGGT intrinsics, and solved object poses, then exports a mesh archive under the same frame poses. The coarse 4 mm carve is watertight with zero boundary/non-manifold edges and improves rendered depth p95 from about 124 mm to 52.5 mm while keeping median silhouette IoU about 0.940. It fails physical contact: 87.5 percent of contact samples penetrate the carved volume and only 12.5 percent are near-surface. Loosening the depth carve worsens contact and median depth, and sparse contact-point protection fills volume around the contact patch and makes penetration worse. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_pitch004_865_870/qc_depth_carved_mesh_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_pitch004_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_pitch004_865_870/volume_sdf_contact_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_front025_back045_pitch004_865_870/volume_sdf_contact_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_contactprotect006_pitch004_865_870/volume_sdf_contact_pitch003_qc.json`. The mechanism is now explicit: depth carving improves visible surface accuracy by changing object volume near the MANO contact region, while contact SDF requires a surface there. V3 needs a joint shape-contact formulation, not looser carve tolerances or sparse contact-point filling.
- Full-hand SDF penetration QC further tightens the physical criterion. The prior closed SDF-level-set mesh has good selected contact-patch SDF, but over the entire contacting hand it still places 13.1 percent of MANO vertices more than 2 mm inside the object. The depth-carved mesh increases that full-hand penetration to 37.8 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_voxel_closed_mesh_sdf_levelset_sigma025_865_870/full_hand_sdf_penetration_pitch003_qc.json` and `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_pitch004_865_870/full_hand_sdf_penetration_pitch003_qc.json`. The accepted V3 criterion must therefore include full-hand nonpenetration, not only selected contact-patch proximity.
- The depth-carve diagnostic was extended with sampled MANO-surface exclusion and an SDF boundary-repair probe. Full-hand exclusion without contact filling produces a watertight visible-depth mesh and 0.72 percent full-hand penetration beyond 2 mm. A first SDF-contact check reused contact rows selected against the previous object mesh and therefore falsely reported a 12.8 mm selected-contact separation. Recomputing the mesh-surface contact rows against the current carved mesh changes the result: the current geometry has a consistent index-finger anatomical patch track over frames 865 to 870, selected-contact abs median 2.84 mm, p95 3.90 mm, and 100 percent of selected samples within 6 mm. It still has shallow selected-patch penetration on 20.8 percent of samples. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/mesh_surface_contact_recomputed.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/volume_sdf_contact_recomputed_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`. Adding a local SDF pull toward stale active contact vertices drives the internal pre-extraction contact SDF below 1 mm, but the delivered mesh still leaves those stale selected vertices about 11.5 mm away while full-hand penetration remains zero. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_sdfboundary_hclear004_csigma020_targetneg001_pitch004_865_870/qc_depth_carved_mesh_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_sdfboundary_hclear004_csigma020_targetneg001_pitch004_865_870/volume_sdf_contact_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_sdfboundary_hclear004_csigma020_targetneg001_pitch004_865_870/full_hand_sdf_penetration_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_sdfboundary_hclear004_csigma020_targetneg001_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`. The current causal result is sharper: contact rows must be selected against the same delivered object mesh that QC evaluates. Stale contact evidence can make a valid mesh look physically wrong.
- A visible/contact SDF fusion diagnostic then tested a concrete alternative to parameter sweeping: use the hand-clearance carved mesh as the visible-depth source and union only the contact-neighborhood volume from the contact-preserving closed mesh. The result is watertight and keeps visible-depth in the carved range, with 19.5 mm z-buffer median and 54.1 mm p95, but it restores the original physical failure: selected-contact penetration is 96.9 percent and full-hand penetration is 12.6 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_fused_visible_contact_sdf_hclear004_crad026_pitch004_865_870/qc_fused_visible_contact_sdf_mesh_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_fused_visible_contact_sdf_hclear004_crad026_pitch004_865_870/volume_sdf_contact_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_fused_visible_contact_sdf_hclear004_crad026_pitch004_865_870/full_hand_sdf_penetration_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_fused_visible_contact_sdf_hclear004_crad026_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`. This falsifies contact-source volume union. The active contact condition is a surface-placement constraint, not permission to add object volume around the MANO patch.
- Direct delivered-surface deformation then tested the missing surface-placement mechanism, but these runs used stale contact rows from the previous mesh. They are retained as a diagnostic of objective behavior, not as evidence about the current carved mesh. Starting from the hand-clearance carved mesh, a local vertex displacement toward stale selected contact patches preserved visible-depth quality, but exposed the contact/non-contact hand tradeoff. A conservative direct deformation reached selected-contact median SDF 4.47 mm and 84.4 percent near-surface, but full-MANO penetration rose to 5.47 percent. A constrained least-squares variant with non-contact hand clearance lowered full-MANO penetration to 2.44 percent, but selected contact separated to 6.48 mm median SDF and only 43.8 percent near-surface. Tightening contact and widening the active-contact exclusion improved selected contact to 3.50 mm median and 81.2 percent near-surface, but full-MANO penetration returned to 5.02 percent. Signed-normal clearance reduced full-MANO penetration to 1.05 percent, but pushed selected contact to 10.38 mm median SDF. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_deformed_visible_contact_surface_clear004_sigma007_radius020_865_870/volume_sdf_contact_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_constrained_visible_contact_surface_clear004_sigma007_radius020_865_870/full_hand_sdf_penetration_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_constrained_visible_contact_surface_clear0025_excl020_sigma007_radius020_865_870/volume_sdf_contact_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_signedclear_visible_contact_surface_clear0025_excl012_sigma007_radius020_865_870/full_hand_sdf_penetration_pitch003_qc.json`.
- A local SDF least-squares edit was also tested to avoid mesh-normal artifacts. It drove internal pre-extraction contact SDF samples to about 1.7 to 2.8 mm and preserved watertight topology, but the delivered marched mesh failed external QC: selected-contact median SDF was 10.18 mm, full-MANO penetration was 3.94 percent, and z-buffer p95 depth error was 55.98 mm. The report is `/data2/ego_annotation_outputs/representative_trash/v3_sdf_contact_surface_target0015_hclear004_er018_pitch003_865_870/qc_optimized_sdf_contact_surface_v3.json`. This falsifies the current 3 mm local-field formulation because optimized internal samples do not control the delivered surface under the same voxel-SDF QC used for annotation.
- Contact-state gating was then added to the deformation diagnostic. Frames 865 and 867 are now treated as proximity evidence rather than hard contact because their selected source patches already exceed 5 mm p95 distance in the upstream contact report. The gated solve records enabled hard-contact frames 866 and 868 to 870, but those enabled rows still have 6.56 mm median row SDF and only 37.5 percent median near-surface fraction after delivery. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_gated_contact_surface_sourcep95_005_clear004_865_870/qc_deformed_contact_surface_v3.json` and `/data2/ego_annotation_outputs/representative_trash/v3_gated_contact_surface_sourcep95_005_clear004_865_870/gated_contact_summary_qc.json`.
- The current hand-clearance carved mesh was then rechecked with diagnostics that separate pose, occlusion, and z-buffer artifacts from real geometry error. A per-frame camera-z shift centers visible-depth medians below 5 mm, but leaves z-buffer p95 at 62.7 mm, raises selected-contact penetration to 29.2 percent, and raises full-hand penetration to 5.0 percent, so global z translation is not an acceptable repair. Rendering MANO depth into the same camera shows hand-occluded pixels are only about 1.1 percent of object-depth samples; the unoccluded p95 stays near 52.3 mm on the current mesh and near 62.0 mm after the z shift, so hand occlusion is not the main depth-failure source. Signed residual maps show coherent high-error surface regions: early frames contain broad near-depth interiors, while frames 867 to 870 contain far-depth patches and diagonal streaks across the lid. A small local z-buffer minimum reduces the positive far tail, but the p95 remains above 25 mm and near-side bias increases, so rasterization cracks are not the sole cause. Depth Pro was rerun on A800 for the full 858 to 880 trash window and predicts 1504 px median focal and 0.399 m median object-mask depth, while the manifest depth median is 0.65 m. Evaluating the same carved mesh against Depth Pro worsens visible-depth QC to 21.4 mm median and 84.7 mm p95 z-buffer error on frames 865 to 870. The old scale-0.55 static complete mesh also fails this interval: all-face z-buffer IoU is 0.623, with 87.4 mm p95 under Depth Pro and 142.3 mm p95 under UniDepth. These reports are `/data2/ego_annotation_outputs/representative_trash/v3_zbuffer_depth_shift_carved_current_865_870/qc_zbuffer_depth_shift_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_zbuffer_depth_shift_carved_current_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/object_depth_hand_occlusion_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_zbuffer_depth_shift_carved_current_865_870/object_depth_hand_occlusion_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_residual_map_865_870/qc_zbuffer_residual_map_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/zbuffer_local_min_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depthpro_metric_source_858_880/qc_depthpro_metric_source_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_depthpro_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale055_static_mesh_zbuffer_depthpro_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_vggt_scale055_static_mesh_zbuffer_unidepth_qc_865_870/qc_mesh_zbuffer_projection_v3.json`. The remaining visible-depth error is therefore a depth-source/object-surface/camera-model disagreement. The active V3 path is a joint depth-scale and surface model, not a return to the old static scale-0.55 mesh.
- Depth-source hand-scale comparison then tested whether the visible-depth failure should be handled by switching metric-depth providers. On frames 865 to 870, UniDepth best matches the measured MANO z coordinate, with median depth-minus-MANO z -0.54 mm. Depth Pro preserves three reliable temporal contact rows when recomputing mesh-surface contact against the current carved mesh, but its measured-row p05/p95 MANO-depth residual spans about -185 mm to +65 mm. Manifest depth and VGGT depth are hundreds of millimeters from MANO z in this interval. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_depth_source_hand_scale_compare_865_870/qc_depth_source_hand_scale_v3.json` and `/data2/ego_annotation_outputs/representative_trash/v3_depth_carved_visual_hull_handclear004_noprotect_pitch004_865_870/mesh_surface_contact_depthpro_recomputed.json`. This supports UniDepth as the stronger hand-depth evidence source for this clip, while keeping the object visible-depth residual as a surface/geometry problem.
- A shared object-frame surface-depth solver was then added to test whether the remaining error can be repaired by one category-agnostic mesh-state update. The solver keeps the current object poses fixed, assigns one scalar normal displacement to each prior-frame mesh vertex, and uses visible-depth residuals, strict MANO contact rows, silhouette-boundary preservation, edge smoothness, and displacement regularization. The first version reduced sampled visible-depth residual and kept six reliable contact rows, but z-buffer silhouette IoU fell to 0.735 and mesh area inflated from 0.35 m2 to 2.26 m2. Adding boundary preservation and limiting displacement improves z-buffer p95 to about 30 mm and keeps silhouette IoU near 0.91, but full-hand penetration rises to 13.3 percent. Adding a linearized non-contact hand-clearance residual initially used the wrong normal-displacement sign; after correction it improves z-buffer p95 to 26.6 mm median and keeps six reliable recomputed mesh-surface contact rows with 1.39 mm median signed-gap p95, but delivered-volume QC rejects it. The mesh area is 1.23 m2 with negative signed volume, selected volume-SDF contact penetration is 87.5 percent, and full-hand penetration is 24.7 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_unidepth_contact_865_870/qc_shared_surface_depth_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_unidepth_contact_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_boundary_unidepth_contact_865_870/qc_shared_surface_depth_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_boundary_unidepth_contact_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_boundary_unidepth_contact_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_unidepth_contact_865_870/qc_shared_surface_depth_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_unidepth_contact_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_unidepth_contact_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_fixedsign_unidepth_contact_865_870/qc_shared_surface_depth_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_fixedsign_unidepth_contact_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_fixedsign_unidepth_contact_865_870/mesh_surface_contact_recomputed.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_fixedsign_unidepth_contact_865_870/volume_sdf_contact_recomputed_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_shared_surface_depth_handclear_fixedsign_unidepth_contact_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`. This falsifies the current single-pass normal-displacement formulation: it can trade visible-depth error for surface-distance contact, but the delivered volume still penetrates the hand.
- A shared object-frame implicit-SDF solver then tested the same hypothesis at the delivered zero-surface level. The first version used visible-depth zero-surface targets, selected MANO contact targets, full-hand clearance targets, SDF smoothness, and SDF preservation. It produced a watertight mesh and kept six reliable recomputed contact rows, but independent QC rejected it: z-buffer p95 rose to 60.1 mm, silhouette IoU dropped to 0.901, selected volume-SDF contact penetration reached 66.7 percent, and full-hand penetration reached 4.14 percent. Adding active-set free-space constraints along object-depth rays moved the internal visible SDF from -11.6 mm median to +2.3 mm and made most free-space samples positive, but the delivered mesh still failed: z-buffer p95 was 54.8 mm, silhouette IoU was 0.873, selected volume-SDF contact penetration was 58.3 percent, and full-hand penetration was 1.53 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_contact_unidepth_865_870/qc_shared_sdf_visible_contact_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_contact_unidepth_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_contact_unidepth_865_870/volume_sdf_contact_recomputed_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_contact_unidepth_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_free_contact_unidepth_865_870/qc_shared_sdf_visible_contact_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_free_contact_unidepth_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_free_contact_unidepth_865_870/volume_sdf_contact_recomputed_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_shared_sdf_visible_free_contact_unidepth_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`. This falsifies fixed-pose surface-only SDF repair: free-space and contact constraints carve the volume, but they do not create a tangent contact surface with high-fidelity visible depth and silhouette.
- The fixed-shape contact-patch object pose graph was also repaired and rerun on the carved mesh. Its previous visible-depth residual had variable length because the set of projected pixels changed as the pose moved; the residual is now sampled from a deterministic subset so SciPy cannot accept or reject a graph with changing dimensionality. The fixed-shape graph kept six reliable recomputed contact rows, but independent QC rejected the archive: z-buffer p95 was 55.3 mm, silhouette IoU was 0.894, selected volume-SDF contact penetration was 50 percent, and full-hand penetration was 6.54 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_pose_graph_carved_visible_unidepth_865_870/qc_contact_patch_object_pose_graph_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_pose_graph_carved_visible_unidepth_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_pose_graph_carved_visible_unidepth_865_870/volume_sdf_contact_recomputed_pitch003_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_pose_graph_carved_visible_unidepth_865_870/full_hand_sdf_penetration_recomputed_pitch003_qc.json`. This falsifies fixed-shape pose-only repair and also exposes that the graph's old "consistent" label was weaker than the delivered-mesh QC. The graph acceptance now includes visible-depth p95 and volume-SDF contact penetration, so the same failure mode cannot be labeled consistent.
- A per-frame measured-surface branch then corrected the depth and intrinsics contract: the observed mesh exporter now consumes UniDepth NPZ depth and annotation-VGGT intrinsics instead of the stale BundleSDF PNG depth and 2304 px dataset focal. The first stride-4 archive gave sub-millimeter depth error, but its coverage result was confounded by coarse surface sampling and z-buffer face subsampling. Re-exporting at stride 2 and rendering all faces gives a dense open measured surface with median silhouette IoU 0.916 and z-buffer p95 0.53 mm. Per-frame solidification closes each measured surface as a 1 mm watertight sheet with zero boundary and non-manifold edges. The dense solid sheet reaches median silhouette IoU 0.983, visible-silhouette-inside-mask 0.9999, z-buffer p95 1.07 mm, six reliable recomputed mesh-surface contact rows, selected-contact volume-SDF p95 1.25 mm at 1 mm SDF pitch, selected-contact penetration 0 percent, and full-hand penetration 0 percent. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_mask_mesh_world_stride2_q000_100_edge040_865_870/qc_mask_depth_observed_mesh_archive_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_mask_mesh_stride2_q000_100_edge040_allfaces_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/qc_solidify_sheet_mesh_archive_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_thick001_allfaces_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/mesh_surface_contact_recomputed.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/volume_sdf_contact_recomputed_pitch001_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/full_hand_sdf_penetration_recomputed_pitch001_qc.json`. This branch proves that the measured object sheet can satisfy depth, topology, object-mask coverage, and local hand-object physics on frames 865 to 870 when the reconstruction and QC resolutions match the shell thickness.
- The dense measured-sheet branch was then extended to the full 858 to 880 trash-lid window at stride 3. The 23-frame 1 mm watertight sheet archive reaches median silhouette IoU 0.992, visible-silhouette-inside-mask 0.9990, z-buffer median depth error 1.32 mm, and z-buffer p95 depth error 2.40 mm. Recomputed mesh-surface contact has seven reliable temporal contact rows, with median contact-patch p95 0.88 mm and median signed-gap p95 0.41 mm. The 1 mm selected-contact SDF check has zero penetration and 1.24 mm abs p95; the full-hand SDF check also has zero penetration. Reports are `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride3_thick001_allfaces_zbuffer_qc_858_880/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride3_perframe_thick001_858_880/mesh_surface_contact_recomputed.json`, `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride3_perframe_thick001_858_880/volume_sdf_contact_recomputed_pitch001_qc.json`, and `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride3_perframe_thick001_858_880/full_hand_sdf_penetration_recomputed_pitch001_qc.json`.
- The 23-frame trash run has inspected final presentation renders. The overlay video is `/data2/ego_annotation_outputs/representative_trash/v3_mesh_surface_contact_review_dense_measured_sheet_finalvis2_858_880/mesh_surface_contact_review.mp4`. The side-by-side annotated video plus world reconstruction is `/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_dense_measured_sheet_finalvis2_858_880/world_reconstruction_side_by_side.mp4`. The standalone world-coordinate 3D animation is `/data2/ego_annotation_outputs/representative_trash/v3_world_reconstruction_dense_measured_sheet_finalvis2_858_880/world_reconstruction_3d.mp4`. The videos have 23 frames at 6 fps. The final render pass uses high-contrast object mesh edges, magenta contact markers with black/white outlines, a metric world view with head-camera frustum/path, and the caption prefix `23-frame mesh-backed V3 run`.
- A second representative branch was run on wild-rice stem manipulation. Dense per-frame mask/depth meshing over the model-supported frames 2522 to 2549, followed by 1 mm sheet solidification, produces a 21-frame watertight object-mesh archive. The all-face z-buffer QC reaches median silhouette IoU 0.968, visible-silhouette-inside-mask 0.989, median depth error 0.25 mm, and p95 depth error 3.42 mm. Recomputed contact has 15 reliable temporal rows under the low-score hand-evidence threshold required for this clip; their median contact-patch p95 is 0.70 mm and median signed-gap p95 is 0.58 mm. Selected-contact SDF has zero penetration and 3.45 mm abs p95; full-hand SDF has zero penetration. Reports are `/data2/ego_annotation_outputs/representative_wild_rice/v3_active_stem_pruned_dense_solidified_stride1_thick001_allfaces_zbuffer_qc_2522_2549/qc_mesh_zbuffer_projection_v3.json`, `/data2/ego_annotation_outputs/representative_wild_rice/v3_active_stem_pruned_dense_solidified_stride1_thick001_2522_2549/mesh_surface_contact_recomputed_det015.json`, `/data2/ego_annotation_outputs/representative_wild_rice/v3_active_stem_pruned_dense_solidified_stride1_thick001_2522_2549/volume_sdf_contact_recomputed_det015_pitch001_qc.json`, and `/data2/ego_annotation_outputs/representative_wild_rice/v3_active_stem_pruned_dense_solidified_stride1_thick001_2522_2549/full_hand_sdf_penetration_recomputed_det015_pitch001_qc.json`. This is a mesh-backed evidence window, because the object stream is pruned to model-supported frames rather than dense over every frame in 2520 to 2550.
- The wild-rice evidence window has inspected presentation renders over the continuous mesh-backed interval 2531 to 2537. The overlay video is `/data2/ego_annotation_outputs/representative_wild_rice/v3_mesh_surface_contact_review_dense_solidified_det015_finalvis2_2531_2537/mesh_surface_contact_review.mp4`. The side-by-side annotated video plus world reconstruction is `/data2/ego_annotation_outputs/representative_wild_rice/v3_world_reconstruction_dense_solidified_det015_finalvis2_2531_2537/world_reconstruction_side_by_side.mp4`. The standalone world-coordinate 3D animation is `/data2/ego_annotation_outputs/representative_wild_rice/v3_world_reconstruction_dense_solidified_det015_finalvis2_2531_2537/world_reconstruction_3d.mp4`. The videos have seven frames at 2 fps and carry the caption prefix `7-frame mesh-backed evidence window`.
- The final evidence manifest is `/data2/ego_annotation_outputs/v3_final_evidence_manifest_20260606.json`.

Interpretation:

Object-complete asset priors are feasible, but the current object crops and the central VGGT visible patch underconstrain hidden geometry and absolute scale. Mesh alignment alone can fit the partial observed surface while inventing or shrinking the object. A single global VGGT scale is also insufficient because the anchored projection can preserve silhouette while metric size and depth move by orders of magnitude. Depth Pro and UniDepth independently favor the 0.55 scale, and the metric-side MANO branch supplies verified mesh-surface contact evidence at that scale. The dense per-frame UniDepth/VGGT measured-sheet result now satisfies object-mesh, visible-depth, MANO-contact, full-hand nonpenetration, and deliverable-render checks on a 23-frame trash run and on a seven-frame continuous wild-rice evidence window.

These are bounded evidence windows. They validate measured-sheet reconstruction and local contact checks, while the V3 joint solver obligation remains open. The required state still couples camera scale, MANO metric state, object mesh pose or deformation, metric-depth reliability, and contact state across the clip. V3 also leaves temporal completeness unresolved: the wild-rice object stream is pruned to frames where the VLM/SAM/depth evidence supports the object mesh, and no object mesh is fabricated for rejected frames. V4 should make temporal object completion a learned/model-based perception problem, then check the completed dense stream with the same z-buffer, contact, SDF, and visual-review contract. V17 must inherit the full joint state-estimation obligation instead of treating the V3 evidence windows as solver closure.

Implemented current mesh tools:

- `scripts/complete_object_heightfield_from_mask_depth_v3.py`
- `scripts/export_mask_depth_observed_mesh_archive_v3.py`
- `scripts/diagnose_object_mesh_temporal_consistency_v3.py`
- `scripts/diagnose_vggt_focal_sweep_v3.py`
- `scripts/diagnose_depth_source_hand_scale_v3.py`
- `scripts/diagnose_mesh_surface_contact_v3.py`
- `scripts/diagnose_mano_side_hypotheses_v3.py`
- `scripts/diagnose_contact_kinematics_v3.py`
- `scripts/optimize_contact_patch_object_pose_graph_v3.py`
- `scripts/diagnose_complete_mesh_surface_conflict_v3.py`
- `scripts/fuse_observed_surface_with_complete_prior_v3.py`
- `scripts/filter_mesh_by_multiview_mask_support_v3.py`
- `scripts/close_mesh_archive_with_voxel_fill_v3.py`
- `scripts/diagnose_volume_sdf_contact_v3.py`
- `scripts/diagnose_hand_object_sdf_penetration_v3.py`
- `scripts/apply_zbuffer_depth_shift_v3.py`
- `scripts/diagnose_object_depth_with_hand_occlusion_v3.py`
- `scripts/render_zbuffer_residual_map_v3.py`
- `scripts/diagnose_zbuffer_local_min_v3.py`
- `scripts/optimize_shared_surface_depth_v3.py`
- `scripts/optimize_shared_sdf_visible_contact_v3.py`
- `scripts/reconstruct_object_visual_hull_depth_carve_v3.py`
- `scripts/fuse_visible_contact_sdf_mesh_v3.py`
- `scripts/deform_visible_mesh_contact_surface_v3.py`
- `scripts/optimize_object_sdf_contact_surface_v3.py`
- `scripts/summarize_gated_contact_qc_v3.py`
- `scripts/associate_measured_hand_tracks_v3.py`
- `scripts/export_vggt_scene_observed_mesh_camera_v3.py`
- `scripts/optimize_vggt_focal_hand_contact_graph_v3.py`
- `scripts/patch_annotations_with_vggt_poses_v3.py`
- `scripts/package_static_camera_mesh_window_v3.py`
- `scripts/refit_hand_similarity_target_camera_v3.py`
- `scripts/refit_mano_pose_contact_v3.py`
- `scripts/render_mesh_surface_contact_3d_v3.py`
- `scripts/render_mesh_surface_contact_review_v3.py`
- `scripts/render_world_reconstruction_v3.py`
- `scripts/remote_setup_depthpro.sh`
- `scripts/remote_setup_unidepth.sh`
- `scripts/reconstruct_scaled_observed_object_mesh_v3.py`
- `scripts/regularize_heightfield_depth_scale_v3.py`
- `scripts/run_depthpro_metric_source_v3.py`
- `scripts/run_unidepth_metric_source_v3.py`
- `scripts/sweep_vggt_scale_mesh_silhouette_v3.py`

## Required V3 Solver

V3 should model one joint state over the clip:

- camera trajectory scale and per-frame pose corrections;
- MANO hand pose, translation, and metric scale correction;
- object mesh pose and, for deformable objects, low-dimensional deformation;
- metric-depth scale/bias and per-frame reliability;
- binary or probabilistic contact state per hand/object region.

Factors:

- 2D MANO reprojection to WiLoR keypoints;
- verified mask silhouette agreement;
- object mesh to metric-depth surface residuals;
- MANO/object non-penetration signed-distance residuals;
- contact attraction only when contact is inferred from image evidence and temporal continuity;
- contact attraction downweighted or disabled for predicted or low-confidence hands whose reprojection residual is large;
- temporal smoothness on camera, hands, object pose, and deformation;
- prior terms on physically plausible hand size, object rigidity/deformation, and depth scale.

The graph must expose residual conflicts. A low object-depth residual with a 0.4 m hand/object depth gap is a failed joint annotation, not a success.

After the full-scene VGGT branch, the next graph should treat VGGT scene geometry as an independent depth/camera factor rather than replacing all other sources with it. The current factor graph needs:

- camera intrinsics and Sim3 scale variables constrained by DROID camera motion, VGGT camera motion, and any real calibration if it becomes available;
- object visible-surface factors from SAMWISE plus VGGT, with Depth Anything downweighted or rejected on frames where it contradicts VGGT by hundreds of millimeters;
- MANO depth variables constrained by 2D reprojection, temporal velocity/acceleration, hand-size priors, and detector confidence;
- contact variables that can turn off or mark a hand observation unreliable when satisfying contact would require large reprojection error, hand-scale collapse, or bound-saturated depth shifts.

The focal/hand/contact graph tests this principle on frames 858 to 880. It rejects the broad-lid contact hypothesis by lowering contact probability instead of forcing geometry into contact. That is the correct failure signal for this branch: the central pink lid is a measured support/context surface, while the actual manipulated contact object is likely the liner or perimeter/rim material identified by the VLM surface plan.

The failure to close contact after VGGT is a useful V3 result because it separates three mechanisms: Depth Anything creates large object-depth outliers in late frames, the DROID focal prior is inconsistent with VGGT/contact geometry, and WiLoR/MANO still places some measured hands at incompatible depths even when focal length is allowed to move. V3 cannot close until the object surface being contacted is reconstructed and a stronger temporal hand/depth model passes the same residual checks.

A MANO hand-size depth-source diagnostic was added for frames 878 to 880. It backprojects measured 2D hand keypoints through the manifest depth maps, Depth Pro, UniDepth, and VGGT depth, then compares the resulting 3D hand bone scale with the stored MANO hand geometry. Only one hand row passes the strict measured-hand and reprojection-valid filters, so the result is weak evidence rather than a global scale owner. In that row, the manifest depth backprojects the hand to 1.15x the MANO reference scale, Depth Pro to 1.52x, VGGT to 1.44x, and UniDepth to 0.63x. The right-hand row in frame 880 has zero reprojection-valid joints under the same test. This rejects MANO hand size as a current closure mechanism for object scale in the late window, while preserving it as a residual for future graph optimization after hand tracking improves. The report is `/data2/ego_annotation_outputs/representative_trash/v3_depth_source_hand_scale_878_880/qc_depth_source_hand_scale_v3.json`.

## Surface-Specific Contact Branch

The corrected V2-mask reliability diagnostic used the real pink-lid mask source and still found zero reliable contact rows. The residual comes from more than mask identity: a single lid mask collapses several physical surfaces that can have different depth, visibility, and contact state.

Implemented current branch:

- `scripts/build_contact_surface_plan_v3.py`
- `scripts/render_point_prompt_review_v3.py`
- `scripts/adapt_sam2_track_to_annotations_v3.py`

Model-produced surface plan:

`/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_plan_840_930/contact_surface_plan_vlm.json`

The plan splits the 840 to 930 contact window into five visible surface tracks:

- `pink_lid_top_dished_panel_visible`
- `pink_lid_raised_annular_rim`
- `pink_lid_outer_vertical_flange_edge`
- `second_can_exposed_opening_rim`
- `white_liner_draped_edge_second_can`

Point prompts and review stills:

`/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_points_840_930/`

Inspection status:

- the central lid panel, annular rim, outer flange, and early liner prompts are visually credible;
- the exposed can rim is lower confidence and correctly marks frame 858 invisible;
- later liner prompts are plausible but boundary-sensitive because liner, rim, hand, and lid pixels overlap near the perimeter.
- text-only SAMWISE did not convert these surface names into valid masks. The failed masks are recorded at `/data2/ego_annotation_outputs/representative_trash/v3_samwise_contact_surfaces_rejection_840_930.json`. V3 should use image-conditioned point prompts and per-frame mask selection for these surfaces before attempting mesh reconstruction.
- image-conditioned SAM2 fragment masks are recorded under `/data2/ego_annotation_outputs/representative_trash/v3_contact_surface_sam2_image_fragments_840_930`. They improve local surface evidence but do not supply complete object geometry. A metric-depth fragment-contact diagnostic found 15 hand-surface rows and zero reliable contact rows. The closest liner row at frame 848 still has 62 mm p95 absolute gap and 71 percent penetration deeper than 10 mm. Later plausible-looking fragments remain 60 to 250 mm from MANO in depth or have large hand reprojection residuals.
- full-scene VGGT was rerun on the whole 840 to 930 contact window with full-image masks, not object crops. The 91-frame run completed on A800 and produced `/data2/ego_annotation_outputs/representative_trash/v3_vggt_full_scene_geometry_840_930/vggt_scene_object_points_v3.npz`. Camera-center Sim3 alignment to the existing trajectory has 15.7 mm median and 38.6 mm p95 error. VGGT predicts a median source focal of about 1063 px over this longer window, again far from the 2304 px DROID prior. A nearest-surface fragment-contact diagnostic over all SAM2 fragments still reports zero reliable contact rows. The best global focal in the grid is about 1400 px by median gap, but the supported rows still have 114 mm median p95 surface distance. Some individual rows are near-contact candidates, for example frame 880 can-rim at focal 1800 has 33 mm p95 surface distance, but these rows have one-sided violation fractions around 0.9 to 1.0. They are local proximity evidence, not valid force/contact annotation.
- a stricter patch-contact diagnostic then tested fingertip-sized MANO subsets against the same full-window VGGT fragment surfaces. This found close local geometric patches, but only as isolated observations or anatomical jumps. At source focal 1400 px, the median best-patch p95 distance is about 24 mm and the median patch depth gap is about -1.6 mm, but reliable contact remains zero after requiring temporal support and MANO-local anatomical continuity. The failure mechanism is concrete: frames 880 and 886 on the white-liner track become close only by switching to different MANO vertex regions. The graph must therefore treat these rows as proximity candidates, not contact constraints.
- a temporal MANO patch graph then tested whether bounded hand-depth shifts can turn the VGGT/SAM2 patch candidates into reliable contact while preserving keypoint reprojection, hand scale, temporal smoothness, and anatomical continuity. A smoke run at source focals 1400 px and 1800 px still reports zero reliable fragment-patch contact rows. At 1400 px, hand shifts stay small and keypoint reprojection remains good, but only two singleton geometry rows survive: frame 858 annular rim and frame 903 outer flange. At 1800 px, no patch row passes geometry. This rules out another local MANO depth smoother as the V3 closure mechanism for this clip.

This branch supplies surface-specific mask targets so SAM2 or a referring video segmentation model can produce measured masks, metric-depth observed-surface meshes, and per-surface contact reliability. Contact factors should activate only after the surface mask, metric depth, MANO projection, and temporal support agree. V3 closes only when the resulting hand/object state passes per-surface contact reliability; zero reliable rows would strengthen the falsification of the current hand/camera/depth state.

Current diagnostics narrow the remaining missing evidence. The next solver cannot be another local smoother over the same variables. It must add at least one nonlocal source of metric information:

- a calibrated camera/head scale source, such as measured camera intrinsics plus a known-size object or scene measurement;
- a stronger 3D hand-depth model that predicts metric hand state directly from egocentric images and is checked against 2D RTMLib/WiLoR agreement;
- a temporal hand reconstruction stage that optimizes MANO pose and depth over many frames with metric-depth observations, then uses contact as a checked residual rather than a mandatory attraction;
- object support and force checks only after the hand/object trajectory is geometrically credible.

Until one of these sources is added, V3 should report the contradiction instead of rendering corrected contact. The current evidence rejects these approaches as closure mechanisms:

- object-only mesh pose refinement;
- translation-only, rigid-only, and convention-mismatched local MANO pose refits;
- similarity refits and WiLoR-convention local MANO pose refits as V3 closure on their own;
- HaWoR direct replacement or camera-local adaptation;
- Kalman smoothing over the current hand state;
- forcing contact when image-supported contact rows are sparse or inconsistent.

Operationally, the version sequence is:

- V1: prove the full annotation plumbing with dense head trajectory, WiLoR MANO, captions, and an initial 3D presentation.
- V2: remove object proxies by using VLM object plans, open-vocabulary detection, SAM masks, metric depth, and observed-surface object meshes.
- V3: solve or expose the metric contradiction between the observed object mesh and MANO hands through a joint factor graph. V3 cannot close by producing a nicer render while contact stays hundreds of millimeters wrong.

QC in this project means falsification of the annotation, not only file checks. Structural QC checks that videos, frame counts, JSON records, and mesh archives exist. Geometric QC checks surface fit, silhouette fit, MANO reprojection, hand/object contact distance, penetration support, temporal smoothness, and whether any residual improves only by pushing hidden variables to implausible values. Visual QC checks that the overlay and 3D presentation show the intended object and hand state rather than a diagnostic-looking plot or a wrong object.

## Current V3 Evidence Package

The current dense measured-sheet evidence slice is:

`/data2/ego_annotation_outputs/v3_dense_measured_sheet_evidence_manifest_20260606.json`

The broader V3 evidence history also includes these representative branches:

- trash-lid frames 865 to 870: strongest deliverable-shaped V3 evidence slice. It has MANO/object overlay, standalone 3D world animation, world side-by-side with caption, and a dense watertight per-frame measured object mesh. All-face z-buffer QC gives median silhouette IoU 0.983 and median p95 depth residual 1.07 mm. Recomputed mesh-surface contact gives six reliable temporal contact rows. The 1 mm volume-SDF contact check gives selected-contact penetration 0 percent and p95 absolute SDF 1.25 mm. The full-hand SDF check gives full-hand penetration 0 percent. It remains a V3 evidence slice because the acceptance window is six frames of one clip; V3 still needs the same contract over longer intervals and more representative objects.
- wild-rice frames 2546 to 2549: short complete-mesh evidence slice for a thin manipulated stem. TRELLIS complete mesh plus mask-depth pose graph gives observed-surface median distance 5.10 mm and inside-mask median depth error 11.1 mm. Contact rows are geometry-backed and temporally supported, but detector-backed contact rows remain zero, so this slice is evidence for the mechanism and leaves closure open.
- keyboard frames 60 to 75: rigid-object mesh success and hand-stream failure. The solidified keyboard sheet mesh is watertight with 10.6 mm thickness, 0.414 x 0.209 x 0.056 m extent, median silhouette IoU 0.832, and median absolute depth error 3.65 mm. The hand stream is still rejected: left-hand mask-depth MANO fits reach mask overlap only by saturating pose deltas and minimum scale, while right-hand SAM2 masks still merge glove with grey cloth under visual QC.

The newest hand diagnostics are implemented in:

- `scripts/refit_mano_articulation_mask_depth_v3.py`
- `scripts/select_sam2_visual_track_candidates_vlm_v3.py`

These scripts intentionally keep annotation readiness false unless the hand fit passes mask, depth, pose-bound, scale-bound, and visual checks. On keyboard, that contract produces zero accepted hand rows, which is the correct failure signal.

### MANO Reprojection Diagnostic

The contact window also shows that the MANO source-camera placement has nontrivial 2D disagreement with the hand detector boxes:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_reprojection_depth_840_930.json`

Result:

- hand rows: 182;
- MANO median depth: 1.326 m;
- projected MANO bbox versus detector bbox median L2 residual: 103 px;
- projected MANO bbox versus detector bbox p95 L2 residual: 231 px;
- projected MANO bbox versus detector bbox median max-axis residual: 75 px.
- per-side median L2 residual: left 119 px, right 67 px;
- worst residuals occur on the right hand around frames 875 to 884, where several rows are predicted or low-confidence hand states.

Interpretation: the hand side of the geometry is not a fixed metric oracle. V3 needs explicit MANO translation/depth variables constrained by 2D reprojection, temporal motion, hand-size priors, and hand observation confidence before contact factors can be trusted.

### Joint Depth Contact Probe

The first low-dimensional joint probe uses the same 840 to 930 contact-depth rows, but constrains the correction to one shared MANO depth scale, one shared object depth scale, and smooth per-frame depth shifts:

`/data2/ego_annotation_outputs/representative_trash/v3_joint_depth_contact_840_930.json`

Result:

- rows: 79 frames with at least 80 near-mask hand vertices;
- raw hand-object depth gap median/p95: 0.390 m / 0.667 m;
- corrected absolute gap median/p95: 0.009 m / 0.059 m;
- contact solved threshold: 0.010 m p95;
- hand depth scale: 0.750, exactly the lower bound;
- object depth scale: 1.052;
- hand and object depth shifts both hit their absolute bounds.

Status: `diagnostic_contact_depth_conflict_remains`.

Interpretation: shared depth scale plus smooth shifts cannot explain the contact conflict within the current bounds. The next solver must add image reprojection and silhouette terms for MANO/camera/object state, because the depth-only contact rows alone force implausible corrections and still leave 59 mm p95 contact-depth error.

### Confidence-Gated Object Factor Graph

The object-pose factor graph was rerun on frames 858 to 880 with contact residuals weighted by hand detector score and MANO reprojection fit:

`/data2/ego_annotation_outputs/representative_trash/v3_factor_graph_858_880_confidence/qc_object_factor_graph_v3.json`

Result:

- used frames: 23;
- contact weight median/p95: 0.360 / 0.609;
- observed-to-prior median surface distance: 56.4 mm to 18.0 mm;
- prior-to-observed median surface distance: 60.6 mm to 16.7 mm;
- contact median distance: 608 mm to 560 mm;
- contact p95 distance: 665 mm to 659 mm;
- depth-axis offset median/p95: 23 mm / 281 mm;
- optimizer status: hit `max_nfev=45`.

Interpretation: confidence gating reduces the influence of weak hand states, but object-pose optimization still cannot repair contact. V3 must add MANO/camera depth variables with reprojection and temporal constraints rather than treating the hand mesh as fixed.

### MANO Contact-Reprojection Tradeoff

The next diagnostic asks what would happen if MANO were moved far enough to satisfy the object contact depth:

`/data2/ego_annotation_outputs/representative_trash/v3_mano_contact_reprojection_tradeoff_840_930.json`

Result:

- per-hand rows: 106;
- median absolute hand/object contact-depth gap: 0.423 m;
- measured hand rows: 65, with median gap 0.381 m;
- measured rows with detector score at least 0.50: 58, with median gap 0.393 m;
- median center-ray hand translation required to match object depth: 0.427 m;
- current MANO bbox residual median: 122 px L2;
- median center-ray translation changes bbox residual by 18 px, with p95 change 151 px;
- projection-preserving camera-origin scale median: 0.717;
- median implied metric hand-extent change under that scale: 78.6 mm.

Interpretation: the large gap remains when predicted and low-score hands are excluded. There is no small harmless MANO correction that makes the current object surface contact-consistent. Rigidly moving the hand to the object requires roughly 0.4 m translation. Preserving the 2D projection while changing depth implies a hand-size change around 80 mm at the median. The conflict belongs to joint MANO/camera/depth scale estimation, not to object-pose refinement alone and not mainly to predicted-hand artifacts.

### Joint MANO-Object Graph Probe

Implemented:

- `scripts/optimize_joint_mano_object_graph_v3.py`

The first probe uses frames 858 to 862 with the strict pink-lid observed mesh and the frame-858 TripoSR mesh prior:

`/data2/ego_annotation_outputs/representative_trash/v3_joint_mano_object_858_862/qc_joint_mano_object_graph_v3.json`

State:

- per-frame object rotation, translation, and camera-axis depth offset;
- one global hand metric scale;
- one per-hand center-ray depth shift.

Factors:

- observed object surface to complete-mesh prior;
- complete-mesh prior to observed object surface;
- object silhouette inside the accepted mask;
- hand-object contact proximity from near-mask MANO vertices;
- MANO bbox reprojection to detector boxes;
- temporal smoothness and anchor priors.

Result:

- used frames: 5;
- hand factors: 3;
- optimizer hit `max_nfev=25`, so this is still a diagnostic probe;
- observed-to-prior median surface distance: 93.4 mm to 24.2 mm;
- prior-to-observed median surface distance: 84.1 mm to 20.2 mm;
- contact median distance: 679 mm to 533 mm;
- contact p95 distance: 698 mm to 572 mm;
- MANO bbox residual median: 129 px to 128 px;
- hand scale: 0.99994;
- median hand ray shift: -8.0 mm;
- median object depth-axis offset: 65 mm.

Status: `diagnostic_joint_surface_improved_contact_remains_large`.

Interpretation: adding explicit MANO scale and ray-shift variables does not solve contact under reprojection and hand-size priors. The graph improves object surface fit but leaves more than 0.5 m median hand/object contact distance. This is the correct failure signal: V3 needs a stronger hand/camera/depth estimation stage, not looser contact weights or a hidden smoothing correction.

### Metric-Depth Alignment Diagnostic

Implemented:

- `scripts/diagnose_metric_depth_alignment_v3.py`

The diagnostic compares the current MANO source-camera depth and object mesh source-camera depth against the independent Depth Anything V2 metric-depth map used for observed-surface meshing:

`/data2/ego_annotation_outputs/representative_trash/v3_metric_depth_alignment_840_930.json`

Result:

- rows: 182;
- high-confidence measured hand rows: 124;
- object mesh depth minus metric-depth median: -0.98 mm;
- high-confidence measured MANO depth minus metric-depth median: 165 mm;
- high-confidence measured MANO over metric-depth median ratio: 1.136;
- high-confidence measured MANO over metric-depth p95 ratio: 1.784.

Interpretation: the object mesh sits at the metric-depth surface because V2 meshed that surface, while MANO is systematically deeper than metric depth at the hand joint projections. This does not prove Depth Anything is metrically exact, but it localizes the current pink-lid contact conflict to MANO/camera-depth alignment more strongly than object-mesh depth. The next v3 component must refit measured MANO depth against metric depth and 2D keypoints before contact can become a physically meaningful factor.

### Independent 2D Hand-Keypoint Evidence

Implemented:

- `scripts/run_rtmlib_hand2d_v3.py`
- `scripts/diagnose_rtmlib_wilor_hand2d_v3.py`

RTMLib was run on the A800 host for frames 840 to 930:

`/data2/ego_annotation_outputs/representative_trash/v3_rtmlib_hand2d_840_930/`

Runtime note: the job was launched under tmux on the GPU server, but ONNXRuntime could not load the CUDA provider because `libcudnn.so.9` was missing from the runtime library path. RTMLib still completed with the ONNXRuntime backend and wrote 91-frame outputs. This affects speed, not the semantics of the 2D keypoint model output.

RTMLib output:

- processed frames: 91;
- frames with hands: 91;
- median hands per frame: 2;
- median RTMLib hand mean score: 0.403;
- overlay video: `rtmlib_hand2d_overlay.mp4`.

The first all-pairs comparison to WiLoR was misleading because correct and crossed hand pairings were mixed. The one-to-one association diagnostic gives the actual 2D agreement:

`/data2/ego_annotation_outputs/representative_trash/v3_rtmlib_hand2d_840_930/qc_rtmlib_wilor_association.json`

Result:

- frames: 91;
- frames with at least one RTMLib/WiLoR match: 65;
- frames with at least one match below 30 px median keypoint error: 54;
- matched hand instances: 113;
- good matched hand instances: 101;
- matched median keypoint delta: 16.3 px;
- good-match p95 keypoint delta: 21.8 px.

Visual review:

- frame 840 rejects RTMLib as a direct replacement because it marks the lower body/leg region as a low-confidence hand while the visible hand is near the can;
- frame 880 gives one useful left-hand match at 12.3 px median keypoint delta but misses the right hand;
- frames 903 and 930 give visually plausible two-hand RTMLib detections and one-to-one matches to WiLoR at roughly 9 to 22 px median keypoint delta.

Interpretation: RTMLib provides an independent live 2D hand-keypoint observation for many contact-window frames. It does not explain the hundreds-of-millimeters hand/object depth conflict, and it cannot be converted into contact constraints when detections are unmatched, low-confidence, or visually false. The next graph should use RTMLib and WiLoR as associated 2D evidence, while metric depth and contact terms own the depth disagreement.

### Hand Depth, Keypoint, and Contact Graph

Implemented:

- `scripts/optimize_hand_depth_keypoint_contact_v3.py`

The graph uses only hand observations with associated RTMLib/WiLoR 2D landmarks. Its state is deliberately small:

- one global hand scale;
- per-hand camera-ray depth shift;
- per-row object depth shift;
- no hidden fallback to predicted hands.

Factors:

- 2D keypoint reprojection to the WiLoR keypoints when RTMLib and WiLoR agree below 30 px;
- MANO joint depth against the metric-depth map;
- hand/object contact depth for hand vertices that project near the object mask;
- hand bone-scale prior;
- temporal smoothness on hand and object depth shifts.

Strict contact-supported run:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hand_depth_keypoint_contact_loose_near.json`

Result:

- observations: 3, all left-hand frames 886, 888, and 889;
- before fitting: keypoint reprojection median 14.5 px, MANO-minus-metric-depth median -62 mm, hand-object depth median 85 mm;
- depth-only fit: MANO-minus-metric-depth median 13 mm, but hand-object depth median worsens to 155 mm;
- contact fit: hand-object depth median becomes 3.2 mm, but object shift saturates the 80 mm bound and MANO-minus-metric-depth median remains -54 mm;
- status: `diagnostic_metric_depth_residual_remains`.

Matched-2D run with no required near-mask contact:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hand_depth_keypoint_contact_no_near_minrows1.json`

Result:

- observations: 6;
- before fitting: keypoint reprojection median 19.7 px, MANO-minus-metric-depth median 49 mm;
- depth-only fit: MANO-minus-metric-depth median 0.45 mm, but hand shifts reach the 150 mm bound;
- contact fit: contact-supported rows reach 2.4 mm median hand-object depth, but hand scale drops to 0.883, hand shifts reach the 150 mm bound, object shift reaches the 80 mm bound, and keypoint reprojection median rises to 20.6 px;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

Interpretation: independent 2D keypoints make the contradiction sharper. In rows where 2D hands are live, metric depth and contact can be made individually plausible, but the joint fit requires bounded depth shifts or loses keypoint quality. This rejects a Kalman-only or smoothing-only fix. The next v3 mechanism must estimate hand depth from stronger 3D evidence, external scale, or a richer MANO/depth/camera state before contact can serve as a physical regularizer.

HaWoR was also tested in the same diagnostic instead of assuming it was unusable from the earlier adapter failure.

Camera-local HaWoR contact-supported run:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hawor_camera_local_hand_depth_keypoint_contact.json`

Result:

- observations: 3;
- before fitting: keypoint reprojection median 26.2 px, MANO-minus-metric-depth median -385 mm, hand-object depth median -220 mm;
- contact fit: hand-object depth median becomes 3.9 mm, but hand scale hits the 1.15 upper bound, hand shift hits the 150 mm bound, and keypoint reprojection median rises to 28.1 px;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

Camera-local HaWoR without required near-mask contact:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_keypoint_contact_840_930/qc_hawor_camera_local_hand_depth_keypoint_contact_no_near.json`

Result:

- observations: 10;
- before fitting: keypoint reprojection median 29.0 px, MANO-minus-metric-depth median -284 mm;
- contact fit: MANO-minus-metric-depth median becomes -10.8 mm, but hand scale hits the 1.15 upper bound and hand shift reaches the 150 mm bound;
- status: `diagnostic_keypoint_reprojection_residual_too_large`.

The HaWoR translation-refit annotation produced zero contact-supported observations under the same matched-2D/contact criteria. HaWoR therefore does not close the v3 hand state on this slice. It is still useful evidence that a different hand backend changes the error direction: HaWoR is too shallow relative to metric depth, while WiLoR’s accepted contact rows have a smaller but still inconsistent depth/contact tradeoff.

### EgoForce Pose-Head Branch

EgoForce was tested because it directly targets egocentric camera-space hand pose, the missing quantity in the current failure. The diagnostic used annotation-derived hand boxes only as crops and disabled Kalman filtering:

`/data2/ego_annotation_outputs/representative_trash/v3_egoforce_posehead_840_930/`

Runtime repairs on the A800 host:

- downloaded and size-verified `_DATA/model_weights.pth` from the public EgoForce Hugging Face repository;
- copied MANO left/right pickle files from `/data/dex_home/yiwen/mano_assets/mano/models`;
- installed `chumpy` with no build isolation and restored legacy NumPy aliases before MANO pickle loading;
- bypassed unused EgoForce package imports that pulled in PyTorch3D and depth-model wrappers before the pose head could run;
- replaced EgoForce's PyTorch3D-dependent camera-space solve with the same pinhole ray-translation least-squares equation over the pose-head 2D/3D outputs and crop metadata.

Output:

- requested frames: 91;
- hand rows: 131;
- skipped rows/frames: 43;
- overlay video: `egoforce_posehead_overlay.mp4`, 83 frames at 1920 x 1080;
- median joint reprojection to the source observed keypoints: 127 px;
- p95 joint reprojection: 463 px.

The shared contact-reliability diagnostic gives:

`/data2/ego_annotation_outputs/representative_trash/v3_egoforce_posehead_840_930/qc_contact_reliability_bonescale.json`

Result:

- rows with object masks and meshes: 31;
- high-score measured rows: 27;
- reliable contact rows: 0;
- high-score median joint reprojection: 238 px;
- depth-consistent rows: 0;
- contact-consistent rows: 2, but these rows fail projection and depth checks;
- median hand bone scale: 163 mm.

Visual review of `overlay_probe_001.jpg`, `overlay_probe_002.jpg`, and `overlay_probe_003.jpg` confirms the metric failure: EgoForce points are displaced onto the lid, floor, or arm rather than the visible hand. This branch therefore does not supply usable contact evidence on the representative slice. Full EgoForce detector mode remains a separate test after the full detector setup finishes, because the pose-head diagnostic used annotation boxes and pseudo-arm boxes as crop evidence.

### MANO Metric-Depth Refit Probe

Implemented:

- `scripts/refit_mano_metric_depth_v3.py`

This probe uses measured hands with detector score at least 0.50. It scans one global MANO local-geometry scale and computes a bounded per-row center-ray shift to match metric-depth samples at WiLoR 2D joints:

`/data2/ego_annotation_outputs/representative_trash/v3_refit_mano_metric_depth_840_930.json`

Result:

- rows: 124;
- global hand scale: 0.9975;
- median MANO-minus-metric-depth residual: 154 mm to 0 mm;
- p95 absolute ray shift hits the 220 mm bound;
- median absolute ray shift: 170 mm;
- median 2D joint reprojection residual: 10.9 px to 13.3 px;
- median hand span changes by -0.4 mm.
- good-keypoint subset, defined by initial median reprojection at most 20 px: 90 rows;
- good-keypoint subset median depth residual: 144 mm to 0 mm;
- good-keypoint subset median 2D reprojection: 9.1 px to 11.5 px;
- good-keypoint subset p95 depth residual after refit: 140 mm.

Status: `diagnostic_mano_reprojection_residual_too_large`.

Interpretation: metric-depth evidence can pull MANO to the depth surface without shrinking the hand. On rows with good initial 2D keypoints, the median reprojection remains below the 12 px threshold after refit. The required shifts are still large, many rows hit the shift bound, and the good-keypoint p95 depth residual remains 140 mm. V3 needs a hand-depth estimator that models metric-depth reliability, hand occlusion, and temporal consistency before applying contact factors to the final annotation.

### Hand-Depth Reliability Diagnostic

Implemented:

- `scripts/diagnose_hand_depth_reliability_v3.py`

This diagnostic samples Depth Anything metric depth at measured hand joints and records local depth-patch stability, keypoint reprojection quality, and proximity to the accepted object mask:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_depth_reliability_840_930.json`

Result:

- joint rows: 2604;
- good-keypoint rows: 1853;
- stable good-keypoint rows: 1740;
- stable good-keypoint MANO-minus-metric-depth median: 161 mm;
- stable good-keypoint depth-patch IQR ratio median/p95: 0.0035 / 0.0187;
- stable good points near the object mask: 198;
- stable good near-object MANO-minus-metric-depth median: 230 mm.

Interpretation: local metric-depth instability does not explain the main MANO-depth excess. Even stable depth patches with good 2D keypoints place current MANO substantially deeper than the depth surface. The near-object subset is worse, so contact/occlusion regions need special treatment, but the broad mechanism is a MANO/camera-depth alignment error rather than only edge noise in metric depth.

### Hand-Contact Reliability Diagnostic

Implemented:

- `scripts/diagnose_hand_contact_reliability_v3.py`

This diagnostic asks whether current WiLoR hands can be used as physical contact observations for the accepted pink-lid object mesh. A hand row is reliable only when the hand is measured, detector score is high, raw 2D keypoint reprojection is good, metric-depth samples at hand joints agree with MANO depth, local depth patches are stable, MANO bone scale is plausible, and near-mask hand vertices are already close to the object surface:

`/data2/ego_annotation_outputs/representative_trash/v3_hand_contact_reliability_840_930.json`

Result:

- hand rows: 50;
- measured high-score hand rows: 27;
- reliable contact rows: 0;
- measured high-score median 2D keypoint reprojection: 23.6 px;
- measured high-score median MANO-minus-metric-depth residual: 257 mm;
- measured high-score median hand/object contact-depth gap: 115 mm, computed only on the 5 rows with near-mask contact samples;
- measured high-score median bone scale: 213 mm;
- measured high-score median fingertip spread: 101 mm;
- condition counts for measured high-score rows after the corrected bone-scale QC: 5 pass projection, 0 pass depth, 7 pass stable-depth, 27 pass bone scale, 0 pass contact.

Interpretation: the current WiLoR hand stream cannot provide valid contact factors in this slice. Detector score alone is misleading under occlusion. The original tip-spread size check was wrong because a grasping hand can be closed; the corrected bone-scale check shows that hand scale is not the active bottleneck. The failures are depth agreement and hand/object contact support.

### MANO Depth-Refit Candidate Render

Implemented:

- `scripts/apply_mano_depth_refit_v3.py`

This script creates a candidate annotation JSON by shifting only measured hands with good initial 2D reprojection toward metric-depth samples, then recomputes source-camera and world MANO vertices:

`/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/annotations_mano_depth_refit.json`

Rendered videos:

- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/overlay_mano_object.mp4`
- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/reconstruction_3d_world.mp4`
- `/data2/ego_annotation_outputs/representative_trash/v3_mano_depth_refit_candidate_840_930/render/side_by_side.mp4`

Result:

- applied hand corrections: 86;
- median applied shift: -155 mm;
- structural video check: 91 frames, overlay and reconstruction 960 by 540, side-by-side 1920 by 540;
- contact-depth median: 385 mm to 249 mm;
- high-confidence measured contact-depth median: 393 mm to 234 mm;
- high-confidence measured contact-depth p95 remains 598 mm.

Visual review:

- frame 858: image overlay remains plausible, but the 3D hands are still separated from the object mesh;
- frame 880: right-hand overlay collapses into a narrow vertical strip on the lid;
- frame 903: hands remain below or beside the lid in 3D.

Interpretation: depth translation alone is not an acceptable v3 annotation stage. It improves the median depth residual but does not produce contact-consistent MANO geometry and degrades some visible hand overlays. The next hand branch must refit MANO pose/translation jointly, or replace the hand backend with a model whose metric hand placement is better conditioned under egocentric occlusion.

### Hand Translation and MANO-Layer Refit Diagnostics

Implemented:

- `scripts/optimize_hand_translation_contact_v3.py`
- `scripts/refit_mano_pose_contact_v3.py`
- `scripts/optimize_hand_similarity_contact_v3.py`

The translation-only diagnostic uses the fused WiLoR local geometry as the source of truth and optimizes per-hand source-camera translation against 2D keypoints, metric depth, temporal smoothness, hand-size limits, and object contact depth.

Smoke result on frames 880 to 889:

- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after translation refit: 0;
- measured high-score median 2D reprojection: 45.4 px;
- measured high-score median MANO-minus-metric-depth residual: 13.8 mm;
- measured high-score median bone scale after corrected QC: 164 mm;
- measured high-score median fingertip spread: 32.5 mm.

Interpretation: translation can reduce some depth residuals, but it does not create reliable contact evidence. The small fingertip spread is a closed-hand pose descriptor, not a size failure. The refit is rejected because projection support remains weak and contact rows still fail.

The MANO-layer pose/contact refit attempted to optimize MANO pose, global orientation, translation, and local scale from the saved `mano_params`.

Smoke result on frames 880 to 889:

- fit rows: 12;
- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after pose refit: 0;
- measured high-score median 2D reprojection: 41.2 px;
- measured high-score median MANO-minus-metric-depth residual: -68.4 mm;
- measured high-score median contact gap on available near-mask rows: 160 mm;
- measured high-score median bone scale after corrected QC: 191 mm;
- measured high-score median fingertip spread: 112 mm.

This refit changes the visible hand pose but still fails contact and projection reliability. The cause is partly a representation mismatch: the fused annotations store WiLoR local geometry after a global scale and a source-camera translation solve, while `mano_params` remain in the raw WiLoR MANO frame. Reconstructing fused geometry from a plain SMPLX MANO layer and the saved params gives tens to hundreds of millimeters of geometry error. Using WiLoR's own MANO wrapper and the recovered raw-to-metric scale of 1.341 improves reproduction, but the right hand in frames 886 to 889 still has median joint errors from 14 mm to 61 mm.

Interpretation: saved `mano_params` are a useful pose prior, not the current source of truth for metric hand geometry. The next v3 hand solver must operate on the fused local vertex/joint stream or a stronger hand backend, and must treat tip spread as pose state rather than hand-size evidence.

The fused-geometry similarity refit then tested whether a local isotropic scale and source-camera translation can resolve the contact conflict.

Smoke result on frames 880 to 889:

- fit observations: 12;
- independent contact-reliability rows: 18;
- measured high-score rows: 12;
- reliable contact rows after similarity refit: 0;
- measured high-score median 2D reprojection: 44.5 px;
- measured high-score median MANO-minus-metric-depth residual: 13.8 mm;
- measured high-score median bone scale after corrected QC: 189 mm;
- measured high-score median fingertip spread: 38.4 mm;
- measured high-score contact gap on the one available near-mask row: 27.7 mm.

Interpretation: similarity refit improves depth and one contact-depth row, but still yields zero reliable contact rows because projection support and contact support remain weak. The limiting variable is the coupled image/depth/contact hand state, not scalar hand size.

## Implemented Diagnostics

The implemented v3 code is diagnostic, not the required solver above:

- `scripts/summarize_contact_depth_scale_v3.py`
- `scripts/optimize_contact_depth_scale_v3.py`
- `scripts/diagnose_hand_reprojection_depth_v3.py`
- `scripts/diagnose_mano_contact_reprojection_tradeoff_v3.py`
- `scripts/diagnose_metric_depth_alignment_v3.py`
- `scripts/refit_mano_metric_depth_v3.py`
- `scripts/diagnose_hand_depth_reliability_v3.py`
- `scripts/diagnose_hand_contact_reliability_v3.py`
- `scripts/apply_mano_depth_refit_v3.py`
- `scripts/optimize_hand_translation_contact_v3.py`
- `scripts/refit_mano_pose_contact_v3.py`
- `scripts/optimize_hand_similarity_contact_v3.py`
- `scripts/optimize_temporal_hand_contact_v3.py`
- `scripts/optimize_joint_depth_contact_v3.py`
- `scripts/optimize_object_factor_graph_v3.py`
- `scripts/optimize_joint_mano_object_graph_v3.py`
- `scripts/remote_setup_hawor.sh`
- `scripts/export_hawor_world.py`
- `scripts/adapt_hawor_to_annotations_v3.py`
- `scripts/adapt_hawor_camera_local_v3.py`
- `scripts/optimize_hand_rigid_contact_v3.py`
- `scripts/remote_setup_handdgp.sh`
- `scripts/run_handdgp_export_v3.py`

### HaWoR World-Hand Branch

HaWoR was run on the representative trash clip on the A800 GPU server through tmux:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_world/hawor_world_hands.npz`

Export QC:

- frames: 1050;
- image focal used by HaWoR: 2304;
- valid hand frames: 1049 left and 1049 right;
- HaWoR metric scale estimate: 0.872534;
- SLAM DBA errors reported by HaWoR: 1.098 and 0.797.

The raw HaWoR hand output has plausible surface extent before alignment:

- left-hand fingertip-spread median: 86 mm;
- right-hand fingertip-spread median: 79 mm;
- vertex bounding-box diagonal median: about 221 to 226 mm.

The camera-path Sim(3) alignment to the existing DROID-derived annotation world for frames 840 to 930 reached:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_world_adapted_840_930/qc_adapt_hawor_to_annotations_v3.json`

- alignment frames: 19;
- camera-position error median/p95/max: 14.9 mm / 26.4 mm / 30.5 mm;
- Sim(3) scale: 0.350569;
- adapted hands: 182;
- skipped hands: 0.

That camera alignment is not physically valid for the hands. It shrinks the HaWoR hand span to about 26 mm and gives hundreds of pixels of 2D error against the original observed keypoints. The shared contact-reliability diagnostic reports:

`/data2/ego_annotation_outputs/representative_trash/v3_hawor_contact_reliability_840_930.json`

- rows: 50;
- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median joint reprojection error: 481.5 px;
- measured high-score median fingertip spread after Sim(3): 26.2 mm;
- near-mask contact-gap median on available rows: -847.8 mm.

Alternative bridge checks do not rescue the branch. Raw HaWoR projection under HaWoR's own camera convention has a better median reprojection error of 25.8 px, but p95 remains 715.9 px. Whole-clip and near-window camera alignment variants still have hundreds of pixels of median reprojection error after adaptation. A reprojection-aware Sim(3) compromise improves median reprojection to about 39 px, but preserves a tiny 29 mm hand span and increases camera error to roughly 35 to 54 mm. Tightening the hand-size prior only raises the hand span to about 43 mm while pushing camera error toward 58 to 93 mm.

Interpretation: HaWoR is useful evidence, but it is not a direct v3 replacement for WiLoR in this clip. HaWoR's own hand-camera projection is partly plausible, while a single world-frame Sim(3) that aligns HaWoR cameras to the existing DROID trajectory makes the hands physically impossible. The next v3 implementation must optimize hand state in the target metric frame with explicit hand-size, 2D keypoint, metric-depth, temporal, and contact factors. It must not accept HaWoR contact factors through a hidden scale correction.

### HaWoR Camera-Local Branch

Implemented:

- `scripts/adapt_hawor_camera_local_v3.py`

The camera-local adapter preserves HaWoR's per-frame hand geometry in HaWoR camera coordinates, then uses the existing annotation camera pose only to place that camera-local hand in the DROID/object world. This avoids the global camera-path Sim(3) that shrank HaWoR hands.

Result on frames 840 to 930:

- adapted hands: 182;
- median hand bone scale: 165 mm;
- median fingertip spread: 76 mm;
- median camera depth: 1.01 m;
- median joint reprojection: 22.6 px over all adapted hands.

Corrected contact reliability:

- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median reprojection: 30.7 px;
- measured high-score median MANO-minus-metric-depth residual: -147 mm;
- measured high-score median contact gap on available near-mask rows: -137 mm;
- all measured high-score rows pass bone scale.

A translation-only refit of HaWoR camera-local hands improves metric-depth residual but still fails contact reliability:

- measured high-score median reprojection: 31.4 px;
- measured high-score median MANO-minus-metric-depth residual: -13 mm;
- measured high-score contact rows: only 3 available, with median gap -19 mm;
- reliable contact rows: 0.

Interpretation: HaWoR camera-local geometry is more plausible than the Sim(3) world bridge, but translation alone cannot satisfy projection, depth, and contact. The next solver needs at least per-frame rotation/depth/contact-state variables, or stronger 2D hand keypoints, before rendering a candidate.

### HaWoR Rigid Hand-State Probe

Implemented:

- `scripts/optimize_hand_rigid_contact_v3.py`

This probe adds per-frame hand rotation to the HaWoR camera-local translation variables. It uses reprojection, metric-depth, temporal, bone-scale, and object contact-depth factors.

Result on frames 840 to 930:

- observations: 27;
- median reprojection: 30.7 px to 20.2 px;
- median MANO-minus-metric-depth residual under the reliability rows: -147 mm to -24 mm;
- median contact gap on available measured high-score rows: -137 mm to -386 mm;
- median rotation correction: 0.36 rad;
- median translation correction: 167 mm;
- reliable contact rows after corrected reliability QC: 0.

Interpretation: rigid per-frame freedom improves some reprojection and depth residuals, but it worsens contact and uses large hidden corrections. This is a failed diagnostic, not a candidate for rendering. The next hand stage needs contact-state inference and robust keypoint selection, or a stronger hand keypoint backend, rather than looser rigid optimization.

### HandDGP Camera-Space Branch

Implemented:

- `scripts/remote_setup_handdgp.sh`
- `scripts/run_handdgp_export_v3.py`

HandDGP was tested as an independent camera-space hand mesh source because its DGP module solves global camera translation from cropped hand images and crop intrinsics. The adapter runs on measured WiLoR hand crops, transforms the source camera intrinsics into crop coordinates, runs the official FreiHAND checkpoint, writes camera-space vertices and joints into the annotation schema as diagnostic full-vertex hand geometry, and reuses the same contact reliability diagnostic.

Result on frames 840 to 930:

- exported hands: 124;
- skipped hands: 0;
- export median reprojection: 26.5 px;
- export p95 reprojection: 234.9 px;
- export median hand depth: 0.785 m.

Corrected contact reliability:

`/data2/ego_annotation_outputs/representative_trash/v3_handdgp_840_930/qc_contact_reliability_bonescale.json`

- reliability rows: 27;
- measured high-score rows: 27;
- reliable contact rows: 0;
- measured high-score median reprojection: 36.5 px;
- measured high-score median MANO-minus-metric-depth residual: -244 mm;
- measured high-score median contact gap on available near-mask rows: -330 mm;
- measured high-score median bone scale: 126 mm.

Interpretation: HandDGP does not solve the current v3 hand/object metric contradiction on this egocentric contact window. It places the hand shallower than metric depth and the object mesh, while the projection error remains above the contact-reliability threshold for nearly all rows. This branch strengthens the current diagnosis: replacing WiLoR with a generic camera-space hand mesh model is insufficient. HandDGP also does not provide MANO pose parameters, so it cannot satisfy the final MANO deliverable by itself. The missing mechanism is a clip-specific joint hand/depth/contact estimation stage or a more egocentric metric hand backend whose output passes the same residual checks.

### Temporal Hand Contact Graph Probe

Implemented:

- `scripts/optimize_temporal_hand_contact_v3.py`

This probe keeps the fused WiLoR local hand geometry and optimizes a temporal source-camera translation, velocity, and continuous contact probability for measured hands. It uses separate WiLoR and RTMLib 2D keypoint residuals, metric-depth residuals, object-depth contact residuals only for near-mask vertices, non-penetration residuals, bone-scale priors, and temporal motion/contact smoothness. It also checks the annotation representation contract before fitting: `joints3d_camera + cam_t` and `vertices_camera + cam_t` must match the fused source-camera geometry within 25 mm, or the row is rejected.

Smoke result on frames 880 to 889:

`/data2/ego_annotation_outputs/representative_trash/v3_temporal_hand_contact_split2d_880_889/qc_temporal_hand_contact.json`

- observations: 12;
- variables: 84;
- solver RMS: 7.60 to 5.59;
- median translation shift: 158 mm;
- median contact probability: 0.050 to 0.039;
- median WiLoR keypoint reprojection: 305 px to 46 px;
- median RTMLib keypoint reprojection on matched rows: 167 px to 45 px;
- median MANO-minus-metric-depth: -58 mm to 3.4 mm;
- median contact gap: 120 mm to 143 mm.

External contact reliability on the candidate:

`/data2/ego_annotation_outputs/representative_trash/v3_temporal_hand_contact_split2d_880_889/qc_contact_reliability_bonescale_after.json`

- rows: 18;
- measured high-score rows: 12;
- reliable contact rows: 0;
- measured high-score median reprojection: 46.4 px;
- measured high-score median MANO-minus-metric-depth: 14 mm;
- measured high-score contact gap: 13.7 mm on only one near-mask row;
- contact-ok rows: 0 because the near-mask support is below the reliability threshold.

Interpretation: the graph improves the depth residual without forcing contact. This is the right failure mode, because it exposes that the available 2D and near-mask contact support cannot justify a physically reliable contact factor. Temporal translation, velocity, and contact-state inference over the existing fused hand stream are therefore insufficient to close V3.

### Corrected V2-Mask Contact And Focal Sweep

Implemented:

- `scripts/merge_v2_object_masks_with_hands_v3.py`
- `scripts/diagnose_intrinsics_focal_sweep_v3.py`

The earlier contact diagnostics used the older full-annotation object track. For the V2 pink-lid mesh, the object source of truth is:

`/data2/ego_annotation_outputs/representative_trash/v2_plan_pink_lid_masks/annotations_plan_masks.local.json`

The merge script combines that V2 object-mask source with the WiLoR hand stream:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_masks_wilor_hands_merged.json`

Corrected contact reliability for frames 840 to 930:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_wilor_contact_reliability_840_930.json`

- rows: 182;
- measured high-score rows: 124;
- reliable contact rows: 0;
- measured high-score median joint reprojection: 10.9 px;
- measured high-score median MANO-minus-metric-depth residual: 173 mm;
- measured high-score median hand-lid contact gap: 269 mm;
- measured high-score contact-ok rows: 1.

This corrects a source-mixing error in the diagnostic workflow. The bad mask seen in the old contact probe came from the stale white-bag track, not from the V2 pink-lid mask source. Visual probes with the V2 masks show that the pink-lid observed-surface mask is semantically correct in frames 857 and 886. Those frames still do not give reliable physical contact evidence: the right hand interacts with liner or rim context, and the left-hand MANO fit near the lid edge is not reliable enough to serve as a contact factor.

Focal-length sweep:

`/data2/ego_annotation_outputs/representative_trash/v3_v2pink_intrinsics_focal_sweep_fine_840_930.json`

- tested focal range: 1800 to 2400 px with principal point 960, 540;
- best median contact gap focal: 1800 px, with median contact gap -1.4 mm;
- at 1800 px, reliable contact rows remain 0, depth-ok rows are 5 out of 124, and contact-ok rows are 1 out of 124;
- at 2304 px, reliable contact rows remain 0, measured high-score median contact gap is 269 mm, and measured high-score median depth residual is 173 mm.

Interpretation: focal length is a real sensitivity, but focal-only correction does not close V3. Lowering focal can align the median hand-lid depth gap, yet it fails the per-row reliability tests. The missing mechanism is a joint object-context and hand-state model that represents which surface is being contacted: lid, rim, liner, or no contact.

## Immediate Execution Plan

### Wild-Rice Representative Branch

The active wild-rice clip tests a different representative failure mode from the trash-lid branch: many visually similar thin stems, hands occluding the manipulated object, and a fast egocentric camera above a reflective preparation table.

Current evidence for frames 2520 to 2550:

- full-scene VGGT native camera was run on the A800 host and scaled by object-mask UniDepth/VGGT depth ratio. The 31-frame camera archive is `/data2/ego_annotation_outputs/representative_wild_rice/v3_vggt_native_camera_2520_2550/vggt_native_camera_v3.npz`.
- HaWoR needed a 121-frame context clip because a 31-frame clip produced no DROID-SLAM proximity factors. The context run produced 62 hands for frames 2520 to 2550, but visual projection QC rejected raw HaWoR as final MANO because both hands were shifted lower/outward.
- RTMLib was run on A800 for the same source frames and merged as independent 2D hand-keypoint evidence. The target-camera similarity refit improved median hand reprojection from 26.4 px to 16.5 px and median MANO-minus-UniDepth from -13.2 mm to -3.3 mm. The p95 reprojection remains high, so the refit stream is a diagnostic MANO observation, not final physical contact evidence.
- dense VLM point prompts plus per-frame SAM2 candidates were run for every frame. A one-sheet VLM selector accepted masks that visual QC rejected: frames 2534 to 2539 merge nearby stems, and frames 2548 to 2550 initially selected prompt/image artifacts.
- the selector was changed to batch smaller candidate sheets at higher tile resolution. The rerun fixed late frames 2548 to 2549 by selecting a clean single-stem candidate, while the VLM verifier rejected frame 2550 as a detached peel.
- `scripts/prune_verified_mask_track_v3.py` adds a generic mask-quality filter using VLM verdicts, connected components, secondary-component ratio, and bbox fill. With relaxed thin-object fill, it keeps 21 of 31 frames and rejects disconnected or wrong-object masks. The accepted mask stream is `/data2/ego_annotation_outputs/representative_wild_rice/v3_active_stem_sam2_vlm_selected_dense_batched_pruned_relaxed_2520_2550/sam2_vlm_selected_track_pruned.json`.

Interpretation: this branch now has a defensible observed-surface mask stream over most of the 31-frame manipulation window, but it does not yet have a complete object mesh or closed contact reasoning. The next valid object step is to build observed surfaces from the pruned mask stream and then test mesh completion/fusion against projection, depth, temporal rigidity, and hand contact. Frames rejected by the pruner must remain unobserved; forcing them into a continuous object mesh would reintroduce the same wrong-object simplification.

1. Recover full object scale and pose for the central pink-lid object with explicit scale ownership. Use the VGGT patch as a visible-surface factor, but add full-object evidence from the SAMWISE silhouette, the TRELLIS complete-prior silhouette, MANO hand-size scale, Depth Pro/UniDepth/Depth Anything/VGGT depth-source reliability, and any accepted scene support or calibration source. The optimizer must keep the complete mesh at a physically plausible object scale; shrinking the complete prior to the 4 to 6 cm DROID-aligned VGGT patch is rejected. The current strongest scale hypothesis is 0.55 because Depth Pro and UniDepth both select it, but it is still an unresolved hypothesis until it also passes hand/object contact and calibrated scene-scale checks.
2. Run language-conditioned or image-conditioned segmentation for the actual contact materials in the 840 to 930 window, starting with `white_liner_draped_edge_second_can` and perimeter/rim prompts from the VLM surface plan. Visual or VLM review must accept the masks before meshing.
3. Reconstruct accepted contact-surface masks through the same category-agnostic geometry path: SAMWISE or image-conditioned SAM mask, VGGT metric surface where available, temporal consistency QC, and projection residuals. Depth Anything should be used only when it agrees with VGGT or another metric source.
4. Run the contact reliability diagnostic per reconstructed surface. Contact factors may activate only for rows that pass 2D projection, surface depth, bone-scale, temporal support, and near-surface support on the same physical surface.
5. Extend the focal/hand/contact graph from the central lid diagnostic to the accepted liner/rim surface. The graph must keep focal, hand scale, hand depth shifts, and contact probability explicit, and it must report contact-off solutions as failed contact evidence.
6. Render a candidate only after the reconstructed object, reconstructed contact surface, and hand state pass geometric QC and visual QC together. A candidate must not be rendered from a graph that satisfies contact by suppressing contact probability, hitting bounds, shrinking hand scale, shrinking the complete object to a visible patch, or sacrificing keypoint reprojection.
