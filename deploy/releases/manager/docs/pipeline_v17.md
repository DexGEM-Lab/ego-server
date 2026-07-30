# Pipeline V17: Measurement-Grounded Multi-Object HOI Annotation

## Status

**Final status: V17 FAILED (2026-06-12).** V17 is closed as a failed design, not as an accepted annotation pipeline. Its evidence remains useful, but its default path violated the required runtime scale, failed to implement first-class occlusion handling, used BundleSDF/test-time neural optimization as a central object path, and did not deliver an acceptable full-video result. All readiness flags remain false. Active V17 automation was stopped; further work moves to V18.

V16 closed only as full-duration packaging. It produced full-length videos for two raw clips, but it did not satisfy the original V3 joint graph requirement and its annotations do not meet the quality requirement. V17 treats every detector output as a measurement with residuals, confidence, and source evidence before the solver can accept an annotation state.

V17 implementation has produced the measurement store, full-state integration, a latent contact-mode graph, and sparse full-timeline evidence-consistency graphs. The measurement store reads V16 full-video outputs, prior HaWoR/WiLoR artifacts, HaMeR repairs, SAM2 object masks, contact-state rows, local deformable contact patches, and tomato persistent-shape state as measurements, then emits anchor QC before any graph solver can accept or repair them. The full-state integration writes full-length V17 evidence/QC JSONs, and the current render step verifies duration only. The contact-mode graph estimates per-frame per-hand contact/no-contact/unobserved modes from graph-corrected hand-object gaps, object-mask proximity, selected anchors, and temporal switch cost. The sparse geometry graph optimizes per-active-frame object translation corrections, per-active-frame small-angle object rotation corrections, and per-valid-hand camera-ray depth corrections against either selected anchor contacts or contact-mode factor-ready rows, object priors, pose smoothness, and hand-ray smoothness. Its contact equality terms use a local nearest MANO surface patch and report broader hand-object distances separately. It keeps camera trajectory, MANO articulation and shape, object mesh topology, and contact mode labels fixed, so it is an integrated consistency solver for the current V17 evidence layer. The complete nonlinear V3 joint solver remains open.

The current generated graph annotations still carry one manipulated object stream per frame. V17 now materializes the object roster as a separate multi-object mask-evidence timeline, but it is not an object-pose solution: the timeline carries SAM2 mask evidence for simultaneous objects and marks geometry and pose unresolved. Trash has four active VLM objects over 1,633 object-frame rows, with 1,604 visible-mask rows and 29 active rows without visible masks. Tomato has nine active VLM objects over 1,149 object-frame rows, with 1,109 visible-mask rows and 40 active rows without visible masks. The contact-mode-factor graph still outputs one legacy object stream, with 816 object-variable frames for trash and 670 for tomato. Any artifact whose schema exposes only `object` rather than simultaneous `objects` is a single-manipulated-object QC artifact.

V17 now also materializes multi-object visible-surface evidence under `/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces`. This layer fuses the multi-object SAM2 masks with the V16 UniDepth metric archives and the graph annotation camera poses, then writes object-aware world-surface mesh archives. The trash clip has 1,417 depth-backed visible-surface rows and 187 rejected visible-mask rows; rejections are 156 missing metric-depth frames, 15 rows with too few sampled vertices, and 16 rows with too little surface connectivity. The tomato clip has 694 depth-backed visible-surface rows and 415 rejected visible-mask rows; rejections are 172 missing metric-depth frames, 230 rows with too few sampled vertices, one row with too few valid masked depth pixels, and 12 rows with too little surface connectivity. Every row is marked `visible_surface_only_not_canonical_mesh`, `no_object_pose_variable`, `object_geometry_complete=false`, and `object_pose_requirement_met=false`. These surfaces are measurement evidence for the future solver, not reconstructed object poses.

V17 also writes a multi-object geometry-state diagnostic under `/data2/ego_annotation_outputs/v17_multi_object_geometry_state`. This diagnostic measures center-normalized visible-surface envelope repeatability from the object-aware RGBD surfaces. Trash has four envelope candidates and tomato has two, but both clips have zero persistent visible-surface candidates and zero rigid-pose candidates. The distinction matters: a center-normalized envelope can repeat when a deformable bag, a symmetric object, or a depth artifact produces similar silhouettes, so it cannot supply material correspondence, canonical object geometry, or SE(3) pose.

V17 now exports object-track datasets under `/data2/ego_annotation_outputs/v17_object_track_datasets`. These datasets contain per-object RGB, mask, metric-depth PNGs, and intrinsics at the 960 by 540 depth resolution. They are inputs for material-correspondence tracking, not tracking results. Trash exports four object datasets with 1,448 total frames and 185 rejected frames. Tomato exports four usable object datasets with 936 total frames and 213 rejected frames; five tomato context objects have no exported tracking dataset because visible masks and metric depth do not overlap enough. The material-correspondence tracker and motion-state diagnostic now consume these datasets.

V17 now also has a first material-correspondence measurement layer under `/data2/ego_annotation_outputs/v17_object_material_tracks`. The tracker consumes the V17 object-track dataset contract directly: RGB, SAM2 object mask, depth PNG, per-frame depth intrinsics, and graph annotation camera pose. It does not read legacy single-object intrinsics fields. The material-track summary now contains ten attempted windows, one for every exported object dataset in both representative clips, and 203 rigid-ready adjacent pairs. Trash has five windows across four tracked objects and 135 rigid-ready adjacent pairs: black trash bag 0160-0219 has 26, off-white trash can 0182-0241 has 25, pink-lid trash can 0800-0859 has 44, white bag 0720-0779 has 30, and white bag 0800-0859 has 10. Tomato has five windows across four tracked objects and 68 rigid-ready adjacent pairs: early tomato 0274-0333 has zero from only four query points, later tomato 0690-0749 has 9, tomato peel 0720-0779 has 59, faucet handle 0906-0939 has zero, and the two-frame plastic-container window 0938-0939 has zero. The rigid diagnostic requires enough inliers, local spatial support, plausible centroid displacement, and plausible rotation angle. Several windows have local adjacent-pair factors but zero tracks surviving the whole window, so they provide short-term surface-motion evidence rather than a persistent object pose. These tracks are local correspondence evidence; they are not a full-timeline object pose state, do not reconstruct complete object geometry, and do not satisfy the V3 solver requirement.

V17 now writes material-motion state diagnostics under `/data2/ego_annotation_outputs/v17_object_material_motion_state`. This diagnostic composes adjacent material-track rigid factors and measures whether the chained transform still explains tracked material points at later frames. The 203 local rigid-ready adjacent pairs produce one persistent window-motion candidate: the pink-lid trash can has three short candidate segments inside frames 0811-0844, with 8 to 13 composed adjacent pairs and chained p95 residuals below 15 mm. The other four trash windows have only local adjacent material motion. Tomato has zero persistent window-motion candidates: the tomato and peel windows have local adjacent factors that do not satisfy the persistent-track and chained-residual predicates, while the early tomato, faucet-handle, and two-frame plastic-container windows have no ready material motion. The diagnostic converts local correspondence evidence into a stricter object-motion question, but it still does not create canonical object geometry, full-timeline SE(3), or deformation variables.

V17 also writes partial material-pose candidate diagnostics under `/data2/ego_annotation_outputs/v17_object_material_pose_candidates`. This layer fits a canonical material-point cloud and per-frame SE(3) transforms only for persistent material-motion segments. Trash has three ready partial pose segments, all on the pink-lid trash can: 0811-0820 with 160 persistent tracks and p95 residual 9.80 mm, 0822-0830 with 161 tracks and p95 residual 9.57 mm, and 0831-0844 with 152 tracks and p95 residual 9.27 mm. Tomato has zero partial pose candidates. These candidates are object-motion evidence over tracked surface points, not reconstructed object geometry: they do not include hidden surfaces, complete topology, full active intervals, contact-coupled physical state, or a delivered per-object pose timeline.

V17 now writes visible-surface replay diagnostics under `/data2/ego_annotation_outputs/v17_object_material_surface_replay`. This layer takes each partial material-pose candidate, transforms the visible surface from the segment's first frame, and compares it against observed object-aware RGBD visible surfaces at later frames. Two of the three trash-can partial pose segments pass this stricter surface replay test: 0822-0830 and 0831-0844 have symmetric visible-surface p95 residuals about 10.8 mm and 10.5 mm. The 0811-0820 segment fails because frames 0814, 0819, and 0820 have replay p95 residuals above 30 mm, despite low median residuals. Tomato still has zero replay candidates. Surface replay is observed-surface evidence only; it does not fill hidden geometry or establish complete object topology.

V17 now writes full multi-object contact evidence under `/data2/ego_annotation_outputs/v17_multi_object_contact_evidence`. This table contains two hand-side rows for every active multi-object timeline row: 5,564 rows total, with 3,266 rows for trash and 2,298 for tomato. It measures MANO-to-visible-surface distances where both hand geometry and object visible-surface geometry exist; 4,213 rows are measured and 1,351 rows are unobserved because hand geometry or visible-surface geometry is absent. This layer reports zero factor-ready contact rows. The zero is not a no-contact conclusion: accepted sparse-graph contacts at trash frames 0182 and 0856 use local contact-patch meshes, and other sparse contacts use legacy corrected object surfaces, while the new multi-object table measures raw object-mask RGBD visible surfaces. The table exposes the unmerged geometry sources that a real hand-object contact solver must reconcile.

V17 now writes pairwise contact state under `/data2/ego_annotation_outputs/v17_pairwise_contact_state`. This layer creates `contact_pair[frame_idx, hand_side, object_id]` rows for every active multi-object timeline object and both hands, then measures projected MANO vertices against the corresponding object mask. The image-plane signal is live: 5,564 pairwise variables produce 5,216 measured image-pair rows, 2,109 image-overlap candidates, and 619 hand-object image-contact candidates. These image candidates support 460 contact-owner candidates across the current contact-mode-ready rows, with 444 single-candidate owner variables and eight ambiguous owner variables. This layer still emits zero physical contact factors. Projected hand/mask overlap can propose which object owns a hand-side contact state; it does not resolve the metric contradiction between MANO depth, visible RGBD surfaces, accepted short-window reconstructions, and object pose.

V17 now writes pairwise contact depth-gap diagnostics under `/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap`. This layer takes the 619 image-contact candidates from the pairwise contact state, samples the projected MANO vertices near each object mask, and compares their camera depth with UniDepth at the same 960 by 540 object pixels. Every candidate has usable metric depth samples. Every candidate is classified as `hand_behind_object_depth`, with zero metric-depth-compatible candidates and zero physical contact factors. Trash has 97 evaluated pair-depth rows with median hand-minus-object depth about 0.94 m; tomato has 522 evaluated pair-depth rows with median hand-minus-object depth about 0.98 m. The signal falsifies the interpretation that image overlap alone can become a contact factor in the current hand state: the projected hand surface overlaps the object mask while sitting roughly one meter behind the object visible depth.

V17 now writes hand metric-depth state diagnostics under `/data2/ego_annotation_outputs/v17_hand_metric_depth_state`. This layer tests the current graph-corrected MANO state against UniDepth before contact ownership enters: it projects the front-most MANO surface into the 960 by 540 depth raster, compares hand camera depth with UniDepth at the same pixels, and splits the samples by distance to active object masks. Across both clips it materializes 3,767 hand-depth variables and measures 2,928 rows. Every measured row is classified as `hand_behind_metric_depth`, with zero metric-depth-compatible hands. The far-from-object partition carries the decisive causal evidence: trash has 841 measured far-object hand rows with median hand-minus-UniDepth depth about 1.10 m, and tomato has 973 measured far-object hand rows with median gap about 0.94 m. The same contradiction therefore exists away from object masks, including rows with acceptable 2D projection residuals. V17 must repair the MANO/camera/depth state before pairwise image overlap or contact-owner variables can become physical hand-object factors.

V17 now writes a hand-depth factor problem under `/data2/ego_annotation_outputs/v17_hand_depth_factor_problem`. This layer converts the hand metric-depth diagnostic into solver variables and factor blocks. It records inherited source-camera translation as a measurement prior, not an accepted metric hand state: 3,758 hand rows come from least-squares translation of MANO local geometry against 2D keypoints, and nine trash repair rows come from HaMeR local geometry against RTMLib 2D keypoints. Across both clips the factor problem has 3,767 hand-depth variables, 2,928 metric-depth factor rows, 3,213 projection-ready rows, 2,619 depth-repair factor candidates, and zero accepted metric hand states. The source solve depths are much larger than UniDepth hand support while the sparse graph hand-ray shifts are millimeter-scale: trash median source-solve hand depth is about 1.71 m with hand-ray-shift p95 about 0.05 mm, and tomato median source-solve hand depth is about 1.52 m with hand-ray-shift p95 about 0.94 mm. The mechanism is therefore not the sparse contact graph. The current source-camera hand translation was solved from monocular hand geometry and 2D projection without UniDepth, so the next V17 optimizer must promote hand translation, local MANO pose or surface state, and depth observation switches into the joint graph before physical contact factors can activate.

V17 now writes a hand intrinsics-depth counterfactual under `/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual`. This layer keeps the original local MANO-family geometry and raw 2D keypoints, recomputes each source-camera hand translation with UniDepth intrinsics scaled to the hand image size, and reruns the same front-surface MANO versus UniDepth acceptance test. The counterfactual improves median depth gaps for 2,887 of 3,767 hand rows, which identifies the hand-source intrinsics mismatch as a real mechanism. It accepts only six metric hand states and leaves 2,679 counterfactual depth-repair candidates. Trash median counterfactual hand-minus-UniDepth gap is about 0.15 m with five accepted states; tomato median gap is about 0.17 m with one accepted state. The intrinsics repair therefore becomes a solver variable and calibration factor, while local MANO pose or surface state and depth observation switches remain necessary before hand-object contact can become a physical factor.

V17 now writes a hand scale-depth counterfactual under `/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual`. This layer tests the remaining monocular hand-scale/depth degeneracy after the intrinsics counterfactual. With fixed 2D keypoints and UniDepth-aligned intrinsics, a positive scale applied to local MANO geometry and source-camera translation preserves 2D reprojection while changing metric hand depth. Case-global scale accepts 423 hand states and leaves 2,262 depth-repair candidates; per-side scale accepts 417 and leaves 2,268; a per-row scale oracle accepts 951 and leaves 1,734. The case-global scale would shrink the median wrist-to-middle-tip length to about 12.3 cm for trash and 13.1 cm for tomato, below the current hand-size prior. Hand scale is therefore part of the latent depth problem, while a physically stable scale variable cannot close the depth tails by itself.

V17 now writes a hand-depth repair graph under `/data2/ego_annotation_outputs/v17_hand_depth_repair_graph`. This layer promotes the V3 hand-depth mechanism into the current full-timeline evidence stack: one case-global hand scale and one camera-ray depth shift per base-available hand row, constrained by UniDepth front-surface depth samples, 2D projection factors on projection-trusted rows, temporal smoothness, hand-size bounds, and positive camera-depth bounds. The graph then reprojects and resamples the corrected MANO surface before counting accepted rows. Across both clips it tracks the same 3,767 hand-depth rows as the metric-depth state, has 2,927 base-available repair variables, uses 2,685 projection-trusted depth-data rows, accepts 831 metric hand states, leaves 1,926 depth-repair candidates, and hits the ray-shift bound in 17 rows. Trash improves to 612 accepted rows and 843 repair candidates; tomato has 219 accepted rows and 1,083 repair candidates. The residual is still mostly far from active object masks: the repaired state has 1,524 `hand_behind_metric_depth` rows and 356 `depth_tail_incompatible` rows. This graph identifies bounded scale and ray-depth translation as a real repair mechanism, while the remaining residual forces local MANO surface/projection repair and depth-observation variables before physical contact factors can activate.

V17 now writes hand-depth repair residual-owner state under `/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state`. This layer consumes post-solve repair graph rows and selects residual pixels from the same owner sample partition that produced each graph residual, preserving all-projected, near-object, and far-object masks at sample level. Across both clips, the 1,926 repair residual candidates split into 1,840 independently supported rows and 86 unsupported projection-owner rows. The supported rows split into 407 rows with nearby compatible UniDepth, 688 rows with partial nearby compatible depth, and 745 rows lacking nearby compatible depth. Trash has 843 residual rows: 757 supported and 86 unsupported, with supported rows split 142 compatible, 281 partial, and 334 lacking. Tomato has 1,083 residual rows, all independently supported, split 265 compatible, 407 partial, and 411 lacking. Nearby compatible depth points to local MANO surface or projection repair; missing nearby compatible depth points to depth-observation or occlusion state; partial rows require mixed ownership. The joint solver problem now consumes this residual-owner state and asserts that its candidate count matches the hand-depth repair graph residual count.

V17 now writes a hand local-projection repair problem under `/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem`. This layer converts the residual-owner split into factor candidates by searching from owner residual pixels to nearby same-hand pixels whose repaired MANO depth already matches UniDepth within 3 cm. Across both clips, the 1,926 repair residual candidates split into 407 local projection or surface repair factor candidates, 688 mixed projection/depth owner rows, 745 depth-observation or occlusion owner rows, and 86 projection-support unresolved rows. Trash has 142 local candidates, 281 mixed rows, 334 depth-observation rows, and 86 projection-support rows. Tomato has 265 local candidates, 407 mixed rows, 411 depth-observation rows, and zero projection-support rows. The assignment search covers 313,128 residual samples, assigns 125,304 of them to compatible nearby same-hand depth seeds, and finds 398,637 compatible seed samples. This is the first explicit factor-problem partition for the post-repair hand-depth residual: local candidates can become local MANO surface or projection factors, while mixed and depth-observation rows require depth-state or occlusion variables before contact factors can become physical evidence.

V17 now writes a MANO parameter ownership state under `/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state`. This layer tests whether the saved `mano_params` reproduce the V17 local hand vertices used by the depth residuals through the WiLoR MANO wrapper, after applying the side convention and per-row similarity alignment. Across both clips, 1,892 of 1,926 repair residual rows have a valid MANO parameter owner under 10 mm median and 30 mm p95 vertex and joint alignment thresholds. The owned rows have median vertex alignment error 2.23 mm and p95 over row medians 6.15 mm. All 407 local projection or surface repair factor candidates are parameter-owned, so they can attach to future MANO articulation variables. The remaining 34 ownership mismatches are nonlocal residual rows: 30 in trash and four in tomato. This artifact establishes a solver precondition; it does not update MANO articulation or depth-observation switches.

V17 now writes a MANO articulation factor-input state under `/data2/ego_annotation_outputs/v17_mano_articulation_factor_input`. This layer reconstructs the repaired front-most MANO surface from the same local vertices, UniDepth-scaled intrinsics, case-global scale, and row ray shift used by the hand-depth repair graph, then asserts that the reconstructed depth pixels match the stored graph samples. It materializes residual-to-seed factor pairs only for the 407 parameter-owned local projection candidates. All 407 candidate rows have surface correspondences: 142 in trash and 265 in tomato. The factor input contains 60,958 residual samples, 53,893 assigned residual-to-compatible-seed pairs, and 138,033 compatible seed samples. Each assigned pair stores residual and seed MANO vertex ids, depth pixels, pixel shift, MANO depth, and UniDepth depth. This artifact supplies the local articulation factor inputs; it still does not optimize MANO pose or update the accepted hand state.

V17 now writes a local MANO articulation solve diagnostic under `/data2/ego_annotation_outputs/v17_mano_articulation_local_solve`. This layer consumes the 407 materialized factor-input rows, fixes the hand-depth graph scale and ray shift, and optimizes only per-row MANO hand-pose deltas against the residual-to-seed surface pairs, joint reprojection, hand-span bounds, and pose-delta prior. After correcting the surface-factor projection coordinates, the solve improves median local depth residual by at least 5 mm in 144 rows, reaches the local depth threshold in 139 rows, and keeps projection trusted in 405 rows. The median depth residual across all rows changes from 49.1 mm to 42.8 mm, and the median per-row improvement is 0.91 mm. Pose deltas still hit the 0.35 rad bound in 344 of 407 rows. This falsifies articulation-only closure under the current local objective: MANO articulation is a live contributor, but the remaining residual still requires depth-observation/occlusion switches, stronger temporal coupling, or a broader joint hand-depth formulation before any hand state can be accepted.

V17 now writes a hand residual-switch problem under `/data2/ego_annotation_outputs/v17_hand_residual_switch_problem`. This layer converts the 1,926 post-repair residual rows into explicit discrete owner variables for the eventual joint solver. It attaches the local-projection assignment and the local MANO articulation solve evidence to each residual row. Rows with only partial compatible depth become mixed projection-depth switch rows; rows without nearby compatible depth become depth-observation or occlusion switch rows; unsupported rows become projection-support switch rows. Local projection rows can become articulation-ready only when local MANO articulation causally reduces the residual, satisfies depth and projection predicates, and stays inside the pose-delta bound. Under the corrected evidence, one row is articulation-ready, 344 local rows still require a pose-bound switch, and 62 local rows require a no-gain switch. The broader switch split remains 407 local surface/articulation switch rows, 688 mixed projection-depth rows, 745 depth-observation or occlusion rows, and 86 projection-support rows. This artifact makes the next solver variable explicit: residual ownership must be optimized jointly with MANO pose and depth-observation state instead of forcing every residual into an articulation-only objective.

V17 now writes a hand depth-observation switch problem under `/data2/ego_annotation_outputs/v17_hand_depth_observation_switch_problem`. This layer consumes the residual switches and reuses the hand-depth repair graph's near-object and far-object sample partitions to ask which depth-side residuals can plausibly be owned by object/occluder observations. Across both clips, 1,433 residual rows require depth-observation decisions: 1,075 are far-field hand-depth switches, 151 are object/occluder switches, and 207 are mixed object/far-field switches. The candidate residual samples contain 122,469 near-object pixels and 57,363 far-object pixels, but the row-level owner split is dominated by far-field rows because many residual rows were selected from the far-from-active-object partition. This rejects object occlusion as the sole explanation for the remaining hand-depth residual. The complete hand solver still needs depth-observation variables, but those variables must be coupled to the broader hand-depth state, not only to hand-object contact.

V17 now writes a hand far-field depth temporal problem under `/data2/ego_annotation_outputs/v17_hand_far_field_depth_temporal_problem`. This layer groups the 1,075 far-field hand-depth switch rows by case, hand side, hand index, signed depth state, and consecutive frame index. Across both clips, those rows form 316 temporal segments. Thirty-five segments last at least eight frames and cover 545 residual rows; the longest segment lasts 68 frames. Trash contributes 433 far-field rows, 142 temporal segments, 13 candidate segments, and 213 candidate rows. Tomato contributes 642 far-field rows, 174 temporal segments, 22 candidate segments, and 332 candidate rows. Segment signs are 225 hand-behind-metric-depth segments, 90 hand-in-front-of-metric-depth segments, and one near-zero median segment. The residual is therefore temporally structured hand-depth evidence. A solver that treats these rows as isolated per-frame observation switches will miss the mechanism; the next hand solver needs full-timeline hand-depth variables coupled to local MANO pose, projection, and depth-observation state.

V17 now writes a hand far-field temporal refit diagnostic under `/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit`. This layer keeps the post-repair residual sample definition fixed, then fits bounded incremental camera-ray depth shifts for the 35 long far-field temporal segments with same-hand temporal smoothness. It tests nonlinear relinearization after the existing repair graph rather than a larger shift bound. Across both clips, 525 of the 545 long-run rows have enough residual samples for the refit; all 525 improve, 522 meet the depth thresholds, and zero hit the ray-shift bound. Trash has 213 refit rows and all 213 meet the depth thresholds, with the median selected residual dropping from about 79 mm to about 14 mm. Tomato has 332 refit rows: 312 have enough samples, 309 meet the depth thresholds, and 20 rows are too sparse for the refit, with the median selected residual dropping from about 68 mm to about 12 mm on solved rows. This is a weak observation because it measures the original residual samples after delta fitting. It identifies temporal ray-depth state as causal; the reprojected measurement path below decides whether that update closes the hand state.

V17 now writes a hand far-field temporal reprojection diagnostic under `/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection`. This layer applies the temporal refit deltas to the repaired hand state, reprojects the MANO surface, and resamples UniDepth on the resulting pixels. That remeasurement falsifies the fixed-sample temporal refit as a closure mechanism. Across both clips, 525 temporal deltas are applied, 504 rows improve after reprojection, and only two rows become metric-depth compatible. The accepted hand rows increase from 831 to 833, and residual repair candidates fall from 1,926 to 1,898. Trash applies 213 deltas, produces 200 improved rows, and accepts one reprojected temporal row. Tomato applies 312 deltas, produces 304 improved rows, and accepts one reprojected temporal row. Temporal ray-depth update is therefore real causal evidence, yet it is insufficient without local MANO surface, projection, and depth-observation relinearization in the same loop.

V17 now writes a hand temporal reprojection residual-owner state under `/data2/ego_annotation_outputs/v17_hand_temporal_reprojection_residual_owner_state`. This layer takes the temporally shifted and reprojected MANO state above, then repeats the nearby-compatible-depth owner test on the post-reprojection residual pixels. Across both clips, the 525 applied temporal rows split into two compatible rows, 26 projection-untrusted rows, and 497 residual-owner rows. Those 497 residual-owner rows contain only 14 clean local MANO-surface factor candidates, plus 190 mixed surface/depth owner rows and 293 depth-observation owner rows. Trash has seven local candidates, 50 mixed rows, 129 depth-observation rows, 26 projection-untrusted rows, and one compatible row. Tomato has seven local candidates, 140 mixed rows, 164 depth-observation rows, and one compatible row. The owner split changes the next solver obligation: a local-MANO-only temporal pass would target 14 rows while ignoring the dominant depth-observation and mixed residual owners. The next V17 hand graph must couple temporal hand-depth variables with residual-owner switches, depth-observation state, and the small local MANO surface/projection factor set.

V17 now writes a hand temporal owner-weighted refit under `/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit`. This layer performs one coupled test after the post-temporal owner split: it keeps the 525 applied temporal ray-depth variables, gives geometry depth factors only to residual pixels matched to nearby compatible same-hand depth, and leaves depth-observation and projection-untrusted rows as explicit prior/smoothness variables. The graph has 182 geometry-factor rows, 1,679 geometry depth sample factors, 293 depth-observation prior/smooth rows, 26 projection-untrusted prior/smooth rows, and zero ray-shift bound hits. The fixed-factor objective again looks stronger than the real state: 205 rows meet the fixed factor depth thresholds, but after reprojecting MANO and resampling UniDepth only nine temporal rows are metric-depth compatible. Accepted hand rows rise from 833 after temporal reprojection to 840, and residual repair candidates fall from 1,898 to 1,891. Trash accounts for eight of the nine compatible temporal rows; tomato stays at one. This identifies owner-weighted temporal relinearization as a real but small repair mechanism. The remaining rows still split into 62 local surface candidates, 178 mixed surface/depth owners, 250 depth-observation owners, 26 projection-untrusted rows, and 20 unapplied temporal rows, so V17 still needs local MANO pose/surface variables and explicit depth-observation state in the same nonlinear loop.

V17 now writes a post-temporal MANO factor input under `/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input`. This layer consumes the owner-weighted reprojected rows above and materializes MANO residual-to-compatible-depth vertex-pair factors for every current local or mixed surface owner. Across both clips it has 240 candidate rows and materializes all 240: 62 local surface rows and 178 mixed surface/depth rows, with 2,267 assigned residual-to-seed MANO vertex pairs, 4,082 residual samples, and 100,863 compatible seed samples. Trash contributes 59 materialized rows and 374 assigned pairs; tomato contributes 181 materialized rows and 1,893 assigned pairs. This artifact is a current-state MANO correspondence input for the next local pose/surface solve. It keeps the 250 depth-observation owner rows and 26 projection-untrusted rows outside geometry factors, so it still records the unresolved observation-state problem instead of converting every residual into a surface-repair claim.

V17 now writes a post-temporal MANO articulation local solve under `/data2/ego_annotation_outputs/v17_post_temporal_mano_articulation_local_solve`. This layer consumes the 240 current correspondence rows above, keeps the owner-weighted total camera-ray hand shift fixed, and optimizes only per-row MANO hand-pose deltas. The result rejects post-temporal articulation-only closure. All 240 rows remain projection-trusted under the keypoint predicate and 210 rows meet the local depth thresholds, but only one tomato row improves median local depth by at least 5 mm. The median depth improvement across all rows is under 0.001 mm, and 151 rows hit the 0.35 rad pose-delta clamp. Trash has 59 solve rows, zero depth-improved rows, 54 local threshold rows, and 57 clamp hits. Tomato has 181 solve rows, one depth-improved row, 156 local threshold rows, and 94 clamp hits. The current local correspondence factors are already small after owner-weighted temporal refit: median before-solve absolute depth residual is about 11.3 mm and after-solve residual is about 11.2 mm. The remaining hand problem therefore requires a coupled state with depth-observation variables and full reprojected hand-depth acceptance, not an isolated MANO articulation pass.

V17 now writes a post-temporal depth-observation state under `/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_state`. This layer measures the 250 depth-observation owner rows left after owner-weighted temporal refit and after the isolated MANO articulation test. The rows match the owner-weighted refit exactly: 120 in trash and 130 in tomato. All 250 use the far-from-active-object owner partition, so the remaining hand-depth residual is not an object-near occlusion explanation. The selected residual samples have 109,540 compatible same-hand depth seeds somewhere in the reprojected hand masks, but only 816 of 9,170 selected residual samples attach to a compatible seed within the 8-pixel local search radius, and zero selected residual samples are directly depth compatible. The row split is 64 zero-local-assignment rows, 85 sparse-local-assignment rows, and 101 weak-local-assignment rows. The residual sign is also asymmetric: 244 rows have MANO in front of UniDepth on the selected tail samples and only six rows have MANO behind UniDepth. This measurement sharpens the next solver obligation: depth-observation variables must be coupled to temporal hand-depth and MANO surface state, because the unsolved rows are far-field hand-depth tails with weak local compatible-depth support, not local articulation residuals and not object-near contact occlusions.

V17 now writes a post-temporal depth-observation support state under `/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state`. This layer tests the 250 post-temporal depth-observation rows against same-side RTMLib, WiLoR, HaMeR, and VLM hand evidence. The V17 measurement store preserves WiLoR `joints2d` as keypoints, so the support state can separate box support from anatomical keypoint support. Independent same-side box support covers 234 rows and leaves 16 unsupported rows, all in the trash clip. Tomato has 130 of 130 box-supported rows; trash has 104 of 120. Same-side independent keypoint support is narrower: 85 rows are partial or strong, and 32 of those are strong under the current 32-pixel, 50-percent residual-sample threshold. The split rejects a broad projected-hand-spillover explanation while preventing the next solver from treating every same-side box hit as equally reliable hand-depth evidence. The next solver therefore needs graded hand-depth observation variables for supported rows and a smaller projection/support ownership variable for unsupported rows.

V17 now writes a post-temporal depth-observation weighted refit under `/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit`. This layer consumes the owner-weighted temporal refit and corrected support state, keeps the same 525 temporal variables, assigns UniDepth depth-observation factors only to same-side box-supported rows with partial or strong independent keypoint support, and leaves sparse, absent, unsupported, and projection-untrusted rows as prior/smoothness variables. Across both clips, 78 depth-observation rows receive observation factors, split into 46 partial-keypoint rows and 32 strong-keypoint rows; 172 depth-observation rows remain prior/smooth variables. The fixed-factor objective improves 134 rows and meets depth thresholds on 250 rows, but the full reprojected measurement path still accepts only 13 temporal rows, raises accepted hand states only from 840 to 844, leaves 1,895 depth-repair candidates, and preserves all 250 depth-observation owners after MANO reprojection and UniDepth resampling. This falsifies scalar camera-ray observation factors as the closure mechanism. The next hand graph must couple MANO surface/pose state, temporal hand-depth variables, and depth-observation ownership in one reprojected objective.

V17 now writes a coupled hand-depth MANO observation graph under `/data2/ego_annotation_outputs/v17_coupled_hand_depth_mano_observation_graph`. This layer places the 525 temporal scalar hand-depth variables, 240 current local/mixed MANO pose variables, and 78 keypoint-supported UniDepth observation rows in one objective, then reprojects MANO and resamples UniDepth through the same owner measurement path. The fixed factors remain locally plausible: 241 scalar rows meet fixed-factor thresholds, 69 geometry rows improve local depth, and 228 geometry rows meet local depth thresholds. The corrected reprojected metric state improves compatibility from 13 to 15 and accepted hand rows from 844 to 846, but residual repair candidates rise from 1,895 to 1,899. Depth-observation owners fall from 250 to 245, and pose deltas hit the 0.35 rad clamp in 124 of 240 geometry rows. This falsifies the current fixed-correspondence coupled objective as the closure mechanism: coupling helps part of the state, but fixed assignments still leave the full residual population open. The next hand graph needs relinearized surface ownership and observation assignment inside the solve, in addition to scalar depth and per-row pose deltas.

V17 now writes a relinearized hand surface-observation graph under `/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph`. This layer starts from the scalar weighted-refit state, then runs three outer passes that replay the current MANO surface, rebuild owner partitions, refresh residual-to-compatible-surface vertex pairs, refresh supported depth-observation factors, and optimize scalar hand-depth plus per-row MANO pose deltas. The corrected relinearized graph activates 235 surface-factor rows, 71 depth-observation factor rows, and 18 compatible-anchor rows. It improves the residual distribution but still does not close the metric state: compatible temporal rows rise from 13 to 18, accepted hand rows rise from 844 to 849, residual candidates fall from 1,895 to 1,882, and depth-observation owners fall from 250 to 224. Trash ends at 12 compatible temporal rows, 624 accepted hand rows, 805 residual candidates, and 105 depth-observation owners. Tomato ends at six compatible temporal rows, 225 accepted hand rows, 1,077 residual candidates, and 119 depth-observation owners. This shows stale assignment explains part of the residual-owner split, but not the full hand-depth contradiction. The next hand state must add a deeper owner such as MANO shape, camera/depth state, or object/contact-coupled depth ownership rather than only refreshing correspondences.

V17 now writes a relinearized hand capacity diagnostic under `/data2/ego_annotation_outputs/v17_relinearized_hand_capacity_diagnostic`. This layer joins the relinearized full-reprojection rows with MANO parameter replay ownership and local surface-factor geometry to test whether the remaining residuals are primarily a missing MANO identity-shape or hand-scale variable. Across the 525 relinearized variables, the diagnostic preserves the same 18 compatible rows as the relinearized graph and studies the 481 residual-owner rows inside those applied variables; the full hand state still has 1,882 depth-repair candidates after relinearized reprojection. The studied residual-owner rows are dominated by mixed surface/depth and depth-observation owners, all 481 have MANO parameter replay geometry, the failed-row wrist-to-middle-tip span range overlaps the compatible-row span range, and the corrected surface projection-to-seed median is 5.6 px. The diagnostic therefore records `shape_only_closure_supported=false` and `capacity_conclusion_state=hand_shape_only_not_supported_by_current_measurements`. This does not make MANO shape irrelevant to a final solver; it prevents V17 from treating hand shape alone as the next closure mechanism before adding joint depth-owner, contact, camera/depth, and object-state variables.

V17 now writes a relinearized residual object-contact diagnostic under `/data2/ego_annotation_outputs/v17_relinearized_residual_object_contact_state`. This layer joins every post-relinearization hand-depth residual row to active-object mask proximity, visible-surface contact evidence, pairwise image/depth contact evidence, and contact-owner variables. It covers all 1,882 residual rows and verifies that 481 of them are the applied residual-owner rows studied by the capacity diagnostic. Current object/contact evidence cannot own any of those residuals: zero rows have metric-depth-compatible pairwise contact, zero rows have a contact-owner factor ready, and zero rows support object-contact closure. The residual population splits into 498 image-contact rows with metric-depth contradiction, 1,198 rows near active object masks without metric contact, and 186 far-field rows that are not object-contact residuals. The sample audit has 1,325,884 valid object-distance samples and 6,388 invalid object-distance samples across nine tomato rows, so missing object-distance values are a small measured absence rather than a closure mechanism. The missing state is therefore not a simple object-contact switch over the current object evidence. V17 still needs a unified camera, hand, dense-depth, and object-depth owner state before contact factors can repair the remaining hand-depth residuals.

V17 now writes a relinearized residual factor-coverage diagnostic under `/data2/ego_annotation_outputs/v17_relinearized_residual_factor_coverage`. This layer evaluates every post-relinearization hand-depth residual row against the same residual selection, compatible-depth assignment, owner partition, and independent hand-support predicates used by the current relinearized graph. All 1,882 residual rows are scalar hand-depth variable candidates, but the current relinearized graph applied variables to only 481 residual-owner rows and skipped 1,401 residual rows. The full residual population contains 1,528 direct factor rows: 1,189 surface-factor rows and 339 supported depth-observation rows. The remaining 354 rows are prior/smooth-only under current measurement predicates. Among the 1,401 skipped rows, 1,241 already have direct factor evidence, split into 956 surface-factor rows and 285 supported depth-observation rows; 160 skipped rows remain prior/smooth-only. This changes the next hand-graph target from searching for an object-contact switch to broadening solver coverage across the full residual population, while keeping explicit prior/smooth or new measurement-source state for rows without direct factors. The diagnostic measures factor availability; closure still requires an optimizer that reprojects MANO, resamples depth, and preserves the false readiness flags until the full metric hand state improves.

V17 now writes full-residual relinearized hand surface-observation graphs under `/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph` and `/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose`, plus row-wise pose-transition and surface-tail diagnostics under `/data2/ego_annotation_outputs/v17_full_residual_pose_transition_diagnostic` and `/data2/ego_annotation_outputs/v17_full_residual_surface_tail_diagnostic`. These layers start from the sparse relinearized graph state, promote the 1,401 previously nonapplied residual rows, and optimize 1,926 scalar hand-depth variables. The scalar-only graph keeps sparse MANO pose deltas fixed to isolate the coverage mechanism: accepted hand rows rise from 849 to 1,057 and residual candidates fall from 1,882 to 1,635 after full MANO reprojection and UniDepth resampling. The pose-enabled graph adds active per-row MANO pose deltas on the same variable population, improving accepted rows to 1,072 and residual candidates to 1,614, while 713 pose rows hit the clamp. The transition diagnostic shows why this is not solver closure: pose creates 94 compatible rows but loses 79 previously compatible rows, leaves 1,535 residual owners persistent, and has nearly zero median absolute gap change across the 1,926 variables. The surface-tail diagnostic localizes the dominant persistent residual mechanism: 921 of 969 surface factors pass local geometry-depth thresholds, but 772 persistent surface-depth-tail rows remain; 551 of those rows both pass local surface geometry and reject the source depth in favor of nearby compatible seed pixels. The remaining pose-enabled residual owners are 376 local surface-factor rows, 676 mixed surface/depth rows, and 562 depth-observation rows, with 71 projection-untrusted rows. Coverage and pose coupling are real mechanisms, but V17 now needs a richer full-residual hand graph with depth-observation owner state and camera/depth consistency rather than more scalar, isolated pose adjustment, or local surface fitting.

V17 now writes a full-residual depth-owner diagnostic under `/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic`. This layer tests the depth owner of the persistent surface-tail rows after the pose-enabled full-residual graph by recomputing residual-to-compatible-seed assignment from the final solved state and splitting unassigned tail pixels by depth-gap sign and active-object proximity. It identifies 714 studied depth-owner rows; 602 have unassigned residual pixels, 533 are far-field hand-in-front majorities and only 33 are near-object majorities, with unassigned pixel gaps near -0.31 m and nearest compatible depth pixels typically 10-11 px away, just outside the 8 px assignment radius. This localizes the persistent residual to UniDepth pixels at hand silhouette boundaries rather than object occlusion or global calibration.

V17 now writes a depth-edge ownership counterfactual under `/data2/ego_annotation_outputs/v17_depth_edge_ownership_counterfactual`. This layer re-evaluates the unchanged pose-enabled solution after excluding UniDepth pixels inside hand-independent depth-discontinuity bands (7 px window, 0.10 m local range, 6 px dilation). On the 1,926 applied variables, 525 rows flip to interior-compatible, 47 previously legacy-compatible rows lose acceptance because edge pixels were balancing their medians, and 1,072 rows remain interior-incompatible with a small +24 mm median interior bias instead of the former meter-scale contradiction. This classifies the all-pixel depth acceptance and factor targets as a measurement-ownership defect: they violate the depth-edge visibility invariant of monocular metric depth at hand boundaries.

V17 now writes an interior-owned full-residual hand graph under `/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph`. This solver keeps the pose-enabled MANO pose deltas fixed and re-solves the 1,926 per-row camera-ray depth shifts against interior-owned UniDepth factors that exclude the depth-discontinuity bands, with temporal smoothness and bounded deltas, then measures the result through full MANO reprojection and UniDepth resampling under both the legacy all-pixel predicate and the interior-owned predicate. Interior-compatible variable rows reach 1,270 of 1,926 with an interior median gap near zero and only 77 bound hits; total interior-predicate accepted hand rows are 1,996 (trash 1,056, tomato 940) versus 1,072 under the prior legacy state, while the legacy predicate on the same solution reads 1,022 because edge pixels no longer drive the objective. The remaining interior-incompatible classes are 504 depth-tail rows, 74 hand-in-front rows, 49 hand-behind rows, and 29 projection-untrusted rows. Scope limits: the interior predicate's validity rests on the edge-band parameters as a visibility model and has not been validated against an independent hand-silhouette source; object geometry, object pose, contact ownership, and all readiness flags remain false, and the highest-leverage V17 blocker moves back to full-interval object geometry and contact-compatible ownership.

V17 now writes full-active-interval object reconstruction jobs under `/data2/ego_annotation_outputs/v17_full_interval_geometry_reconstruction_jobs`. This layer applies the same constant-intrinsics RGBD rectification contract as the seed-window jobs to every exported object-track dataset, one job per object over its full exported interval, with per-frame rectification gating: frames whose rectified depth fails the ray-preservation contract are dropped explicitly into `dropped_frames` evidence, and a job is solver-ready only if at least 75 percent of exported frames survive. All seven jobs are solver-ready: trash black bag 348/437 frames kept, off-white can 336/347, pink-lid can 211/232, white bag 356/432; tomato faucet handle 62/62, tomato 649/655, tomato peel 217/217. The plastic container is skipped with two exported frames. The dropped frames trace to per-frame UniDepth intrinsics outliers (per-case fx spread up to about 300 pixels against the median target camera), not to a systematic rectification failure. These jobs are BundleSDF-compatible solver inputs for full-interval unknown-object pose and geometry; they are not object geometry, and readiness flags remain false until evaluated reconstruction results pass projection/depth QC over full intervals. After the first backend pass the job layer also splits objects at exported-frame gaps larger than 30 frames, because the merged faucet-handle job (two active segments separated by 537 frames) produced silhouette IoU medians near 0.27 while the contiguous tomato-peel job was accepted with IoU median 0.84; the current job set is nine contiguous-segment jobs, all solver-ready.

V17 now writes the deliverable assembly chain: full-interval reconstruction results under `/data2/ego_annotation_outputs/v17_geometry_reconstruction_results_full_interval`, a multi-object world mesh archive under `/data2/ego_annotation_outputs/v17_multi_object_world_mesh_archive` that poses every evaluator-accepted canonical reconstruction by `T_world_camera @ ob_in_cam` and concatenates objects per frame, interior-owned baked hand annotations under `/data2/ego_annotation_outputs/v17_interior_owned_hand_annotations` (trash 843 baked plus 1,065 kept-prior hand rows, tomato 1,083 plus 776; wrist-to-middle-tip medians 11.6 cm and 13.5 cm), deliverable-state manifests under `/data2/ego_annotation_outputs/v17_deliverable_state` with raw-frame-count contracts, and a full-duration render driver writing overlay, 3D world, and side-by-side videos under `/data2/ego_annotation_outputs/v17_deliverable_renders` with an explicit partial-state banner. Captions come from the V16 annotations, which already carry per-frame action descriptions from the egoscale task JSON. The chain refuses to assemble when no accepted object stream exists, and every layer preserves false readiness flags: closure requires accepted full-interval geometry for the active objects, contact-compatible ownership, and a visual quality review that is explicitly out of scope for the duration QC.

V17 now writes a hand surface-depth tail state under `/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state`. This layer consumes the per-row scale oracle and asks what remains after the best scalar hand-depth explanation for each row. Rows accepted by the oracle become scalar-depth-compatible rows; rows whose median is corrected but whose p95 depth tail remains above threshold become local surface/depth tail candidates. Across both clips, the report materializes 3,767 tail variables, 951 scalar-depth-compatible rows, 1,734 scalar-tail candidates, 242 projection-untrusted rows, and 840 unobserved rows. Most tail candidates use the far-from-object owner partition: 569 of 715 in trash and 940 of 1,019 in tomato. That distribution separates the residual from a pure hand-object mask occlusion artifact; the dominant mechanism is broader local MANO surface or depth-observation mismatch. A V17 hand solver therefore needs local MANO pose/surface and depth-observation state in addition to intrinsics, translation, and hand scale.

V17 now writes hand tail-support diagnostics under `/data2/ego_annotation_outputs/v17_hand_tail_support_state`. This layer recomputes the per-row scale-oracle MANO front-surface pixels, selects only residual tail pixels above the 8 cm p95 threshold, and tests their 2D support against model-produced hand boxes and points. The artifact separates selected annotation support from independent model support because the selected hand box is part of the state being evaluated. Across both clips, 1,734 tail candidates contain 53,073 residual tail pixels. Selected annotation support puts 1,394 candidates inside the selected hand box, 135 near it, and 205 outside it. Independent model support puts 1,469 candidates inside a same-side independent hand box, 107 near one, four inside or near another independent hand box, and 154 unsupported. The residual tail is therefore a supported hand-surface/depth mismatch: the next hand solver needs local MANO surface or depth-observation variables; 2D support reassignment accounts for only 154 of 1,734 candidates.

V17 now writes hand tail depth-observation diagnostics under `/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state`. This layer searches a local 8-pixel radius around each residual tail pixel for UniDepth samples within 3 cm of the scaled MANO surface. Across both clips, the same 1,734 tail candidates split into 154 unsupported projection tails and 1,580 independently supported tails. Within the supported tails, 413 have nearby compatible depth for at least 75 percent of residual pixels, 706 have partial nearby compatible depth, and 461 lack nearby compatible depth. Trash has 99 compatible, 235 partial, and 228 lacking among supported tails; tomato has 314 compatible, 471 partial, and 233 lacking. The residual therefore has two live owners: local MANO surface or projection repair for rows with nearby compatible depth, and depth-observation or occlusion variables for rows without it.

V17 now writes a geometry-source audit under `/data2/ego_annotation_outputs/v17_geometry_source_audit`. This audit names which source owns each current geometry claim and checks whether contact factors can be interpreted against the multi-object visible-surface state. The result is incompatible with a closed object-pose solver: the current contact-mode graph has 463 factor-ready rows across the two clips, while the multi-object visible-surface contact table has zero factor-ready rows and zero same-frame visible-surface contact candidates for those ready contact rows. Trash has two accepted local contact-patch states, and both conflict with the multi-object visible-surface distances: frame 0182 has a local patch minimum distance of 2.47 mm while the black-bag visible-surface distance is 342 mm; frame 0856 has a local patch minimum distance of 0.97 mm while the white-bag visible-surface distance is 120 mm. The audit also records that the two passing material-surface replay segments are short observed-surface checks, not complete object geometry. The audit now consumes depth-contact consistency rows for accepted reconstruction meshes. The accepted short-segment pink-lid reconstructions pass rasterized front-surface depth QC against their rectified RGBD observations, while the current hand/contact graph uses a different depth state: 23 evaluated reconstruction frames and 46 hand rows produce zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, zero shared-depth ready frames, and 23 depth-owner incompatibilities. This localizes the current blocker to depth/state ownership after mesh existence has been established.

V17 now writes a contact-ownership problem under `/data2/ego_annotation_outputs/v17_contact_ownership_problem`. This layer turns every contact-mode-ready hand-side row into a discrete owner variable whose domain is the active multi-object rows in that frame. Across the two representative clips, 463 contact-state rows produce 747 candidate object-owner rows. Only seven contact-owner variables have a selected measurement at all; 456 are temporal or legacy contact-mode rows without a selected measurement. Pairwise image evidence supports 460 candidate owner rows and gives at least one supported candidate for 452 owner variables. Pairwise metric-depth evidence supports zero candidate owner rows: all 619 evaluated image-contact candidates place the hand behind the object depth. Zero owner variables are geometry-supported against the current multi-object visible-surface or accepted-reconstruction state, and zero owner factors are ready for the object solver. This is the object-ownership form of the same scientific problem: a hand-side contact state becomes a hand-object factor only when the solver knows which object owns the contact and that object shares metric geometry, depth, and hand state with the contact.

V17 now writes a per-object geometry-hypothesis state under `/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state`. This state groups every active object by object id and compares its mask evidence, visible RGBD surface rows, persistent visible-surface shape measurements, object-depth repair candidates, local contact patches, material-motion windows, material-pose segments, accepted short-segment reconstructions, and source-compatibility audit rows. Across the 13 active objects in the two clips, zero objects have complete geometry, zero can own contact factors, and zero are pose-factor-ready. Trash has two local-contact-patch conflict objects, one partial short-segment hidden-topology reconstruction object, and one visible-surface-only object. Tomato has five mask-only objects, one partial persistent visible-surface object, and three visible-surface-only objects. This table is the current object-geometry state owner for V17: a future solver must promote one of these hypotheses by adding the missing geometry/topology/pose variables, not by treating a contact patch or visible mask as object pose.

V17 now writes an object-geometry factor problem under `/data2/ego_annotation_outputs/v17_object_geometry_factor_problem`. This artifact converts every active object into explicit variable blocks and factor blocks for a future object solver: canonical mesh or SDF geometry, per-active-frame SE(3) pose, topology or deformation deltas, contact attachment state, visible-surface residuals, material-correspondence rigidity residuals, material-motion segments, partial material-pose priors, visible-surface replay checks, unknown-object RGBD reconstruction jobs, evaluated reconstruction results, depth-contact consistency, multi-object hand-contact distances, contact-owner variables, pairwise metric-depth checks, object-depth repair validation, and geometry-source compatibility. Across the two clips it materializes 13 object rows, 2,111 visible-surface factor rows, 203 material-rigidity pair factors, three partial material-pose segments, two visible-surface replay-ready segments, two solver-ready RGBD reconstruction jobs, two detected BundleSDF outputs, two scale-plausible recovered metric meshes, two projection/depth-passing short-segment reconstructions, two accepted short-segment hidden-topology reconstructions, 23 depth-contact evaluated frames, 46 evaluated hand rows, zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, zero shared-depth ready frames, 23 depth-owner incompatibilities, zero contact-ready rows against multi-object geometry, 463 contact-owner variables, 747 candidate owner rows, 460 image-supported owner rows, 619 evaluated pair-depth rows, zero metric-depth-compatible pair-depth rows, zero geometry-supported owner variables, zero contact-owner factor-ready rows, and zero solve-activation-ready objects. This matches the object-level structure used by unknown-object RGBD tracking and reconstruction methods such as BundleSDF: object pose and object geometry must be solved together from RGBD/mask observations, while contact factors can enter only after they attach to the same mesh or SDF state.

V17 now writes observed-surface geometry seeds under `/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed`. This builder takes replay-passing material-pose segments, maps each segment's visible RGBD surfaces back into the material-pose canonical frame, and stores canonical observed-surface mesh archives. The current output has two pink-lid trash-can seed segments: frames 0822-0830 with 11,478 vertices and 21,009 faces, and frames 0831-0844 with 17,618 vertices and 32,390 faces. Their canonicalized surface centroids stay within about 3-4 mm median of the segment source surface, so the material-pose transforms are geometrically coherent for the observed surface. The seeds are still short-window observed-surface geometry: they do not reconstruct hidden topology, do not cover the full active interval, and cannot own contact factors. The object-hypothesis state therefore changes one trash object from partial pose segments to a partial observed-surface geometry seed while preserving `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `v3_solver_complete=false`.

V17 now writes unknown-object RGBD reconstruction jobs under `/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs`. This layer takes the observed-surface seed windows, rectifies their per-frame metric RGBD observations into a constant-intrinsics 1,920 by 1,080 camera contract, and measures whether the rectified depth preserves the original object rays before any hidden-topology solver is allowed to consume the folder. The current output has two solver-ready pink-lid trash-can jobs, both from the two observed-surface seed segments. The rectification p95 residual over 23 job frames is about 0.00027 m, and the projected-inside fraction is at least 0.9967. These jobs are prepared solver inputs; reconstructed hidden surfaces, full active-interval meshes, and contact-compatible object geometry enter only through evaluated backend outputs and later object-geometry factors.

V17 now writes evaluated reconstruction-result reports under `/data2/ego_annotation_outputs/v17_geometry_reconstruction_results`. This layer consumes the prepared RGBD jobs and a BundleSDF output root, then checks whether each solver output has a mesh, a complete per-frame object-in-camera pose sequence, a mesh scale compatible with the rectified RGBD observations, plausible mesh topology, and projection/depth agreement against the job masks and depth maps. Missing backend output is recorded as `pending_solver_output`; a bad mesh is recorded as rejected evidence. The current local report has two detected BundleSDF outputs and two accepted short-segment reconstructions. Both backend runs produced complete pose sequences and `mesh_cleaned.obj`; source inspection shows `mesh_cleaned.obj` is exported in normalized NeRF coordinates before BundleSDF's `mesh_to_real_world` step. The texture path is nonessential for geometry QC and can hang or crash after the mesh exists, so the remote runner now preserves stable `mesh_cleaned.obj` evidence and terminates that path explicitly. V17 recovers a metric QC mesh through BundleSDF's persisted `sc_factor` and translation, records that coordinate contract, and runs scale, topology, projection, and rasterized front-surface depth tests over all projected mesh faces. The all-face condition matters: an earlier 120k-face cap skipped nearer triangles and created a false centimeter-scale front-surface depth tail. Under all-face QC, the recovered metric meshes pass scale, topology, mask projection, and front-surface depth checks. The 0822-0830 segment has median front-surface depth error 2.7 mm and median frame p95 13.7 mm; the 0831-0844 segment has median front-surface depth error 1.6 mm and median frame p95 6.8 mm. All-frame BundleSDF under `/data2/ego_annotation_outputs/v17_bundlesdf_outputs_allframes` also passes both segments and is slightly better on the first segment, with median frame p95 12.0 mm and 6.5 mm. The depth-weighted all-frame run under `/data2/ego_annotation_outputs/v17_bundlesdf_outputs_allframes_depth100` also passes; its second-segment median frame p95 is 8.3 mm, higher than the all-frame depth0 value of 6.5 mm. These accepted meshes cover only short pink-lid trash-can windows. Full active-interval geometry, simultaneous multi-object pose, and contact-factor ownership in one unified object-geometry state remain unresolved.

V17 now writes a depth-contact consistency audit under `/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit`. This layer places accepted BundleSDF meshes into graph world coordinates with their `ob_in_cam` poses, compares them to graph-corrected MANO vertices, and compares visible UniDepth, reconstructed mesh camera depth, accepted front-surface depth QC, legacy object-center depth, hand source-camera depth, contact-mode gaps, legacy single-object ownership, and multi-object contact evidence for the reconstructed object id. The accepted trash windows have visible object depth median 0.386 m, reconstructed full-mesh camera-depth median 0.462 m, and rasterized front-surface depth p95 values already accepted by reconstruction QC, while the current MANO source-camera depth and legacy object depth sit near 1.3-1.8 m. The report records 23/23 depth-owner incompatibility frames, zero near reconstructed-mesh hand rows, zero reconstructed-mesh contact candidates, and zero shared-depth ready frames. It also records four legacy contact-ready hand rows in the same windows, but zero multi-object contact candidates for `object:pink_lid_trash_can_second` and 23 legacy-owner mismatch frames because the legacy single-object stream is labeled `trash_bag`. The accepted mesh is geometry evidence; contact factors still require a joint camera, hand, object, and depth state with explicit object ownership.

V17 also now writes a joint solver problem contract under `/data2/ego_annotation_outputs/v17_joint_solver_problem`. That contract is a generated gap audit. It binds the V3 requirement to concrete variable families and reports the still-missing families: camera trajectory, MANO articulation and shape, multi-object identity timeline, object geometry topology, object pose or deformation state, hand-object contact modes for every object pair, contact patch identity, dense depth and visible-surface state, and physical consistency terms. The contract now requires the multi-object visible-surface, geometry-state, object-track-dataset, object-material-track, object-material-motion-state, object-material-pose-candidate, object-material-surface-replay, multi-object-contact-evidence, pairwise-contact-state, pairwise-contact-depth-gap, hand-metric-depth-state, hand-depth-factor-problem, hand-intrinsics counterfactual, hand-scale counterfactual, hand-depth repair graph, hand-depth repair residual-owner state, hand local-projection repair problem, MANO parameter ownership state, MANO articulation factor-input state, local MANO articulation solve state, hand residual-switch problem, hand depth-observation switch problem, hand far-field depth temporal problem, hand far-field temporal refit, hand far-field temporal reprojection, hand temporal reprojection residual-owner state, hand temporal owner-weighted refit, post-temporal MANO factor input, post-temporal MANO articulation solve, post-temporal depth-observation state, post-temporal depth-observation support state, post-temporal depth-observation weighted refit, coupled hand-depth MANO observation graph, relinearized hand surface-observation graph, relinearized hand capacity diagnostic, relinearized residual object-contact state, relinearized residual factor coverage, hand surface-depth tail state, hand tail-support state, hand tail depth-observation state, contact-ownership-problem, geometry-source-audit, object-geometry-hypothesis-state, object-geometry-factor-problem, geometry-reconstruction-job, geometry-reconstruction-result, and depth-contact consistency reports and records their measured, rejected, envelope-candidate, rigid-candidate, exported tracking-input, material-track-window, rigid-factor, local-motion, persistent-motion-candidate, partial-pose, surface-replay, hand-object distance, image-pair contact, pair-depth contradiction, hand-depth contradiction, hand-depth repair-factor, scale counterfactual, bounded graph residual-owner split, local projection assignment, MANO parameter ownership, MANO surface correspondence, local articulation residual reduction, residual-switch ownership, depth-observation switch ownership, far-field temporal hand-depth segments, temporal relinearization depth-repair evidence, post-update temporal reprojection evidence, post-temporal residual-owner split, post-temporal depth-observation support split, post-temporal depth-observation weighted-refit nonclosure, coupled fixed-correspondence nonclosure, relinearized ownership nonclosure, hand-shape capacity nonclosure, residual object-contact nonclosure, full residual factor-coverage gap, supported hand-surface depth tail, local compatible-depth tail split, contact-owner, source-incompatibility, object-hypothesis, object-factor, solver-job, evaluated solver-output, depth-owner incompatibility, and shared-depth readiness rows inside the object-geometry, object-pose, contact, dense-depth, and physical-consistency variable families. The contract makes the liability explicit: the sparse graph optimizes a small consistency layer over fixed central state variables, and the current object/contact/hand-depth sources still lack one unified metric state.

V17 also corrects a version-accounting problem. V3 already identified the core requirement: solve or expose the metric contradiction between MANO hands and object geometry through a joint factor graph. Later versions implemented real component graphs, including object-pose, sparse object-track, and contact-dynamics graphs. Their scope stayed at selected windows or selected state variables. V16 then closed as a full-length delivery artifact with QC flags while the original joint graph requirement remained open. V17 therefore treats prior graph outputs as evidence modules and reports the current sparse full-timeline graph separately from the still-unimplemented complete nonlinear V3 solver.

The current measurement-store implementation is the evidence layer for V17 state estimation. Model outputs remain traceable measurements with confidence, residual, source, and failure fields; missing hands, missing objects, missing contact states, and incomplete HaWoR/WiLoR coverage become explicit QC failures.

The current HaWoR evidence path uses a compact full-video adapter input generated from V16 annotations. The compact file preserves frame indices, timestamps, source camera transforms, source intrinsics, measured V16 hand 2D keypoints, detector scores, and hand boxes, then reruns the HaWoR camera-local adapter against the full 0-1049 HaWoR NPZ. The adapter input therefore contains only the fields read by the HaWoR residual calculation.

HaWoR rows without current-frame observed hand support are stored as `hawor_motion_infill_candidate` measurements. They can support an occluded or detector-miss state after temporal, projection-contradiction, contact, and nonpenetration checks. Current-frame 2D evidence is required for observed-visible measurements, not for motion-infill candidates.

The trash hand-evidence path now includes VLM-localized visible hand boxes for anchor frames where RTMLib or full-frame HaMeR crops were broad, missing, or attached to the wrong region. The VLM boxes enter the measurement store as image-localization evidence and HaMeR crop inputs. Synthetic keypoints derived from those boxes have no metric meaning; the usable 3D evidence is the HaMeR MANO output and its source-camera reprojection residual.

The trash anchor repair path now materializes selected VLM-box HaMeR hypotheses as V17 repair hand states for frames 0182, 0260, 0764, 0856, 0949, and 0970. The measurement store also ingests contact evidence recomputed from those repaired states. This clears the hand-state blockers at the named trash anchors.

Frames 0182 and 0856 exposed the wrong object variable. Whole-object depth re-anchoring can force a hand-bag metric contact, but temporal validation rejects that variable because it shifts the entire deformable bag surface. V17 now records separate local deformable contact-patch meshes for those frames. Frame 0182 uses a 206-vertex, 302-face black-bag patch centered at the repaired right-hand contact support; the patch occupies 1.43 percent of the SAM2 bag mask and has a 2.47 mm nearest hand-surface distance. Frame 0856 uses a 638-vertex, 1,086-face white-bag patch; the patch occupies 12.77 percent of the SAM2 bag mask and has a 0.97 mm nearest hand-surface distance. The anchor contact graph v4 selects those local patch rows for 0182 and 0856, accepts contact at 0182, 0260, 0764, and 0856, and accepts no-contact at 0949 and 0970. The rejected whole-object depth candidates remain in the measurement store as failed evidence, while the accepted state is the local deformable surface geometry.

The tomato measurement path now has a persistent visible-surface mesh state for `object:obj_tomato`. The solver fuses SAM2 object masks with V16 metric-depth surface extraction over the full active mask interval, rejects 20 temporal surface-scale outliers, and writes a canonical mesh with 84,318 vertices and 136,902 faces. The robust 1-99 percent canonical extents are 9.85 cm, 9.67 cm, and 9.56 cm; raw min-max extent is reported separately because sparse tails can overstate object scale. Tomato anchors 0480, 0720, and 0760 pass persistent-shape QC with surface-to-canonical p95 residuals of 1.7 mm, 5.6 mm, and 1.3 mm. The full-state builder translates the canonical-local visible surfaces by their per-frame `object_center_world_m` before saving the V17 mesh archive, so the archive carries world-coordinate meshes. The tomato contact graph v1 selects left-hand contact rows at 0480, 0720, and 0760.

The current measurement store passes the named trash and tomato anchors. The evidence-layer full-state integration also renders full raw-video outputs for both representative clips: trash has 1,050 raw frames and 1,050 frames in overlay, world, and side-by-side renders; tomato has 960 raw frames and 960 frames in overlay, world, and side-by-side renders. The anchor-only sparse evidence-consistency graph remains a limited QC branch. Trash uses four selected contact factors at 0182, 0260, 0764, and 0856; after optimization its local contact p95-of-p95 is 4.56 mm, broader 80-nearest-point p95-of-p95 is 11.94 mm, max object translation correction is 6.89 mm, max object rotation correction is 0.0259 rad, and max hand camera-ray correction is 3.18 mm. Tomato uses three selected contact factors at 0480, 0720, and 0760; after optimization its local contact p95-of-p95 is 6.08 mm, broader 80-nearest-point p95-of-p95 is 22.44 mm, max object translation correction is 7.62 mm, max object rotation correction is 0.0105 rad, and max hand camera-ray correction is 10.69 mm. The tomato anchor-only branch therefore remains partial even under the sparse evidence-consistency metric. The graph uses local nearest MANO patches for the current contact correspondence selection; a previous 160-point stochastic hand subsample made tomato frame 0480 fail at 36.1 mm, which exposed a sampling artifact in the sparse contact linearization.

The contact-mode graph is implemented as a V17 contact-state layer; the full V3 solver remains open. Its default manifests read the anchor-only sparse graph outputs, so it estimates contact modes from fixed graph-corrected geometry while hand geometry, object geometry, object pose, and contact labels remain fixed inputs. It accepts both representative clips with zero anchor contradictions and explicit `v3_solver_complete=false` metadata. Trash has 2,100 hand-side rows, 1,625 active geometry observations, 473 unobserved rows, 172 contact-mode rows, and 81 contact-factor-ready rows after requiring positive contact evidence, explicit mask-distance evidence, and sparse-graph hand residual compatibility. Tomato has 1,920 hand-side rows, 1,340 active observations, 580 unobserved rows, 527 contact-mode rows, and 382 contact-factor-ready rows. Contact-mode reports now store row-level `contact_factor_readiness_checks`, including hand residual values, mask-distance availability, and threshold predicates, so readiness can be audited from the JSON. Contact-mode interval sheets are regenerated from solved rows with visible QC banners and report per-side timelines, counts, interval previews, and readiness state; they are contact-state QC summaries, and visual-quality rendering remains open. The long tomato left-hand interval from frame 0714 to 0939 matches hand-held washing/rinsing behavior in the graph-corrected side-by-side video, while the trash contact bursts match bag/trash-can manipulation. Rows without hand/object geometry are marked `unobserved` and cannot inherit temporal contact. Rows whose unary evidence opposes contact can remain temporally labeled as contact, but they cannot become geometry factors.

The contact-mode-factor sparse graph consumes only accepted contact-mode `contact_factor_ready` rows through an explicit `--contact-mode-graph-root` input. It structurally accepts both representative clips with converged local contact-patch correspondences. The modeled contact patch now uses 16 nearest MANO surface vertices. A support-size sweep showed why the broader metric cannot be the local contact predicate: trash/tomato p95-of-p95 stays below 5 mm through 18 nearest vertices, then rises to 12.84/13.22 mm at 80 vertices because non-contact hand surface is being included. The 80-nearest-point report remains a support-size sensitivity diagnostic, not a requirement that the whole nearby hand surface touch the object. Trash uses all 81 corrected contact-mode factors, with 1,296 linearized local-patch correspondences, contact-patch p95-of-p95 4.75 mm, broader 80-nearest-point p95-of-p95 12.84 mm, max object translation correction 6.32 mm, max object rotation correction 0.0295 rad, and max hand ray correction 3.36 mm. Tomato uses all 382 corrected contact-mode factors, with 6,112 local-patch correspondences, contact-patch p95-of-p95 5.00 mm, broader 80-nearest-point p95-of-p95 13.22 mm, max object translation correction 7.15 mm, max object rotation correction 0.0294 rad, and max hand ray correction 12.21 mm. The 5 mm numbers are local evidence-consistency diagnostics under the sparse graph's fixed camera, fixed MANO articulation, fixed object topology, fixed contact-label, and nearest-vertex support assumptions. They do not prove a physically valid contact patch until patch identity, image/depth support, and anatomical/contact-area stability are also estimated. The generated reports set `accuracy_target_met=false`, `annotation_ready=false`, and `deliverable_ready=false` because the complete V3-class nonlinear solver remains open.

Two contact-support experiments were rejected before this interpretation was adopted. A single contiguous MANO patch made the graph nonconvergent and raised p95-of-p95 to 15.8 mm for trash and 10.3 mm for tomato. A small multi-site anatomical support model remained nonconvergent at 8.77 mm and 8.07 mm while increasing runtime. These negative results show that support selection alone is not the remaining mechanism; full MANO articulation/surface fitting and object geometry remain fixed in the current sparse graph.

The graph output materializes the optimized state in the annotation files. Object translation and small-angle rotation corrections move `center_world_m`, nested V17 surface centers, local-patch world vertex arrays when present, and the corrected mesh archive around the solved object center. Hand camera-ray corrections move world-space MANO vertices and joints along the solved camera optical axis. Source-camera measurements remain unchanged as evidence. Corrected annotation JSONs now expose `annotation_ready=false`, `deliverable_ready=false`, `v3_solver_complete=false`, `multi_object_timeline_ready=false`, `object_geometry_complete=false`, and `object_pose_requirement_met=false` at the root, in the graph metadata, and on every legacy `object` record. Corrected mesh NPZ archives also embed `v17_archive_metadata_json` and write an adjacent `.metadata.json` sidecar with the same object-geometry and multi-object limitations, so the mesh arrays do not stand alone as complete object-pose evidence.

Graph-corrected full-length renders pass duration QC only. Trash graph renders contain 1,050 frames for overlay, world, and side-by-side outputs; tomato graph renders contain 960 frames for all three outputs. The visual inspection sheets are sampled from the final graph-corrected side-by-side videos and recorded in the render summary. The current trash sheet shows selected contact frames 0182, 0260, 0764, and 0856 with displayed nearest gaps of 0.5 mm, 1.7 mm, 2.0 mm, and 0.9 mm. The current tomato sheet shows selected contact frames 0480, 0720, and 0760 with displayed nearest gaps of 2.4 mm, 0.5 mm, and 2.3 mm. The regenerated contact-mode-factor renders also pass duration QC: trash contains 1,050 frames for overlay, world, and side-by-side outputs; tomato contains 960 frames for all three outputs. New QC renders use `qc_` filenames and their render summary carries `render_qc_scope=duration_only_not_visual_quality`, `visual_quality_qc_pass=false`, `stage9_visual_deliverable_ready=false`, `annotation_ready=false`, `deliverable_ready=false`, and `accuracy_target_met=false`. These videos are duration evidence/QC renders rather than V17 closure deliverables. The current world videos and sheets still use the diagnostic V16/V17 renderer; the Stage 9 audience renderer with shaded MANO/object surfaces, image-plane context, close-up manipulation views, and uncertainty/rejection status remains open.

Current V17 evidence outputs:

```text
/data2/ego_annotation_outputs/v17_full_state/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph/
/data2/ego_annotation_outputs/v17_full_timeline_factor_graph_renders/
/data2/ego_annotation_outputs/v17_multi_object_timeline/v17_multi_object_timeline_summary.json
/data2/ego_annotation_outputs/v17_multi_object_timeline/{trash_1050,task5_tomato_960}/v17_multi_object_timeline.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/v17_multi_object_visible_surface_summary.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/{trash_1050,task5_tomato_960}/v17_multi_object_visible_surface_report.json
/data2/ego_annotation_outputs/v17_multi_object_visible_surfaces/{trash_1050,task5_tomato_960}/multi_object_visible_surfaces_world.npz
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/v17_multi_object_geometry_state_summary.json
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/{trash_1050,task5_tomato_960}/v17_multi_object_geometry_state_report.json
/data2/ego_annotation_outputs/v17_multi_object_geometry_state/{trash_1050,task5_tomato_960}/multi_object_center_normalized_visible_surface_points.npz
/data2/ego_annotation_outputs/v17_object_track_datasets/v17_object_track_dataset_summary.json
/data2/ego_annotation_outputs/v17_object_track_datasets/{trash_1050,task5_tomato_960}/v17_object_track_dataset_summary.json
/data2/ego_annotation_outputs/v17_object_track_datasets/{trash_1050,task5_tomato_960}/{track_id}/manifest.json
/data2/ego_annotation_outputs/v17_object_material_tracks/v17_object_material_track_summary.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/v17_object_material_track_summary.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/{window_id}/v17_object_material_track_report.json
/data2/ego_annotation_outputs/v17_object_material_tracks/{trash_1050,task5_tomato_960}/{window_id}/rigid_pair_factors.json
/data2/ego_annotation_outputs/v17_object_material_motion_state/v17_object_material_motion_state_summary.json
/data2/ego_annotation_outputs/v17_object_material_motion_state/{trash_1050,task5_tomato_960}/v17_object_material_motion_state_report.json
/data2/ego_annotation_outputs/v17_object_material_pose_candidates/v17_object_material_pose_candidate_summary.json
/data2/ego_annotation_outputs/v17_object_material_pose_candidates/{trash_1050,task5_tomato_960}/v17_object_material_pose_candidate_report.json
/data2/ego_annotation_outputs/v17_object_material_surface_replay/v17_object_material_surface_replay_summary.json
/data2/ego_annotation_outputs/v17_object_material_surface_replay/{trash_1050,task5_tomato_960}/v17_object_material_surface_replay_report.json
/data2/ego_annotation_outputs/v17_multi_object_contact_evidence/v17_multi_object_contact_evidence_summary.json
/data2/ego_annotation_outputs/v17_multi_object_contact_evidence/{trash_1050,task5_tomato_960}/v17_multi_object_contact_evidence_report.json
/data2/ego_annotation_outputs/v17_pairwise_contact_state/v17_pairwise_contact_state_summary.json
/data2/ego_annotation_outputs/v17_pairwise_contact_state/{trash_1050,task5_tomato_960}/v17_pairwise_contact_state.json
/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap/v17_pairwise_contact_depth_gap_summary.json
/data2/ego_annotation_outputs/v17_pairwise_contact_depth_gap/{trash_1050,task5_tomato_960}/v17_pairwise_contact_depth_gap.json
/data2/ego_annotation_outputs/v17_hand_metric_depth_state/v17_hand_metric_depth_state_summary.json
/data2/ego_annotation_outputs/v17_hand_metric_depth_state/{trash_1050,task5_tomato_960}/v17_hand_metric_depth_state.json
/data2/ego_annotation_outputs/v17_hand_depth_factor_problem/v17_hand_depth_factor_problem_summary.json
/data2/ego_annotation_outputs/v17_hand_depth_factor_problem/{trash_1050,task5_tomato_960}/v17_hand_depth_factor_problem.json
/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual/v17_hand_intrinsics_depth_counterfactual_summary.json
/data2/ego_annotation_outputs/v17_hand_intrinsics_depth_counterfactual/{trash_1050,task5_tomato_960}/v17_hand_intrinsics_depth_counterfactual.json
/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual/v17_hand_scale_depth_counterfactual_summary.json
/data2/ego_annotation_outputs/v17_hand_scale_depth_counterfactual/{trash_1050,task5_tomato_960}/v17_hand_scale_depth_counterfactual.json
/data2/ego_annotation_outputs/v17_hand_depth_repair_graph/v17_hand_depth_repair_graph_summary.json
/data2/ego_annotation_outputs/v17_hand_depth_repair_graph/{trash_1050,task5_tomato_960}/v17_hand_depth_repair_graph.json
/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state/v17_hand_depth_repair_residual_owner_state_summary.json
/data2/ego_annotation_outputs/v17_hand_depth_repair_residual_owner_state/{trash_1050,task5_tomato_960}/v17_hand_depth_repair_residual_owner_state.json
/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem/v17_hand_local_projection_repair_problem_summary.json
/data2/ego_annotation_outputs/v17_hand_local_projection_repair_problem/{trash_1050,task5_tomato_960}/v17_hand_local_projection_repair_problem.json
/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state/v17_mano_parameter_ownership_state_summary.json
/data2/ego_annotation_outputs/v17_mano_parameter_ownership_state/{trash_1050,task5_tomato_960}/v17_mano_parameter_ownership_state.json
/data2/ego_annotation_outputs/v17_mano_articulation_factor_input/v17_mano_articulation_factor_input_summary.json
/data2/ego_annotation_outputs/v17_mano_articulation_factor_input/{trash_1050,task5_tomato_960}/v17_mano_articulation_factor_input.json
/data2/ego_annotation_outputs/v17_mano_articulation_local_solve/v17_mano_articulation_local_solve_summary.json
/data2/ego_annotation_outputs/v17_mano_articulation_local_solve/{trash_1050,task5_tomato_960}/v17_mano_articulation_local_solve.json
/data2/ego_annotation_outputs/v17_hand_residual_switch_problem/v17_hand_residual_switch_problem_summary.json
/data2/ego_annotation_outputs/v17_hand_residual_switch_problem/{trash_1050,task5_tomato_960}/v17_hand_residual_switch_problem.json
/data2/ego_annotation_outputs/v17_hand_depth_observation_switch_problem/v17_hand_depth_observation_switch_problem_summary.json
/data2/ego_annotation_outputs/v17_hand_depth_observation_switch_problem/{trash_1050,task5_tomato_960}/v17_hand_depth_observation_switch_problem.json
/data2/ego_annotation_outputs/v17_hand_far_field_depth_temporal_problem/v17_hand_far_field_depth_temporal_problem_summary.json
/data2/ego_annotation_outputs/v17_hand_far_field_depth_temporal_problem/{trash_1050,task5_tomato_960}/v17_hand_far_field_depth_temporal_problem.json
/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit/v17_hand_far_field_temporal_refit_summary.json
/data2/ego_annotation_outputs/v17_hand_far_field_temporal_refit/{trash_1050,task5_tomato_960}/v17_hand_far_field_temporal_refit.json
/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection/v17_hand_far_field_temporal_reprojection_summary.json
/data2/ego_annotation_outputs/v17_hand_far_field_temporal_reprojection/{trash_1050,task5_tomato_960}/v17_hand_far_field_temporal_reprojection.json
/data2/ego_annotation_outputs/v17_hand_temporal_reprojection_residual_owner_state/v17_hand_temporal_reprojection_residual_owner_state_summary.json
/data2/ego_annotation_outputs/v17_hand_temporal_reprojection_residual_owner_state/{trash_1050,task5_tomato_960}/v17_hand_temporal_reprojection_residual_owner_state.json
/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit/v17_hand_temporal_owner_weighted_refit_summary.json
/data2/ego_annotation_outputs/v17_hand_temporal_owner_weighted_refit/{trash_1050,task5_tomato_960}/v17_hand_temporal_owner_weighted_refit.json
/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input/v17_post_temporal_mano_factor_input_summary.json
/data2/ego_annotation_outputs/v17_post_temporal_mano_factor_input/{trash_1050,task5_tomato_960}/v17_post_temporal_mano_factor_input.json
/data2/ego_annotation_outputs/v17_post_temporal_mano_articulation_local_solve/v17_post_temporal_mano_articulation_local_solve_summary.json
/data2/ego_annotation_outputs/v17_post_temporal_mano_articulation_local_solve/{trash_1050,task5_tomato_960}/v17_post_temporal_mano_articulation_local_solve.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_state/v17_post_temporal_depth_observation_state_summary.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_state/{trash_1050,task5_tomato_960}/v17_post_temporal_depth_observation_state.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state/v17_post_temporal_depth_observation_support_state_summary.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_support_state/{trash_1050,task5_tomato_960}/v17_post_temporal_depth_observation_support_state.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit/v17_post_temporal_depth_observation_weighted_refit_summary.json
/data2/ego_annotation_outputs/v17_post_temporal_depth_observation_weighted_refit/{trash_1050,task5_tomato_960}/v17_post_temporal_depth_observation_weighted_refit.json
/data2/ego_annotation_outputs/v17_coupled_hand_depth_mano_observation_graph/v17_coupled_hand_depth_mano_observation_graph_summary.json
/data2/ego_annotation_outputs/v17_coupled_hand_depth_mano_observation_graph/{trash_1050,task5_tomato_960}/v17_coupled_hand_depth_mano_observation_graph.json
/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph/v17_relinearized_hand_surface_observation_graph_summary.json
/data2/ego_annotation_outputs/v17_relinearized_hand_surface_observation_graph/{trash_1050,task5_tomato_960}/v17_relinearized_hand_surface_observation_graph.json
/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph/v17_full_residual_relinearized_hand_surface_observation_graph_summary.json
/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph/{trash_1050,task5_tomato_960}/v17_full_residual_relinearized_hand_surface_observation_graph.json
/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose/v17_full_residual_relinearized_hand_surface_observation_graph_summary.json
/data2/ego_annotation_outputs/v17_full_residual_relinearized_hand_surface_observation_graph_pose/{trash_1050,task5_tomato_960}/v17_full_residual_relinearized_hand_surface_observation_graph.json
/data2/ego_annotation_outputs/v17_full_residual_pose_transition_diagnostic/v17_full_residual_pose_transition_diagnostic_summary.json
/data2/ego_annotation_outputs/v17_full_residual_pose_transition_diagnostic/{trash_1050,task5_tomato_960}/v17_full_residual_pose_transition_diagnostic.json
/data2/ego_annotation_outputs/v17_full_residual_surface_tail_diagnostic/v17_full_residual_surface_tail_diagnostic_summary.json
/data2/ego_annotation_outputs/v17_full_residual_surface_tail_diagnostic/{trash_1050,task5_tomato_960}/v17_full_residual_surface_tail_diagnostic.json
/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic/v17_full_residual_depth_owner_diagnostic_summary.json
/data2/ego_annotation_outputs/v17_full_residual_depth_owner_diagnostic/{trash_1050,task5_tomato_960}/v17_full_residual_depth_owner_diagnostic.json
/data2/ego_annotation_outputs/v17_depth_edge_ownership_counterfactual/v17_depth_edge_ownership_counterfactual_summary.json
/data2/ego_annotation_outputs/v17_depth_edge_ownership_counterfactual/{trash_1050,task5_tomato_960}/v17_depth_edge_ownership_counterfactual.json
/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph/v17_interior_owned_full_residual_hand_graph_summary.json
/data2/ego_annotation_outputs/v17_interior_owned_full_residual_hand_graph/{trash_1050,task5_tomato_960}/v17_interior_owned_full_residual_hand_graph.json
/data2/ego_annotation_outputs/v17_full_interval_geometry_reconstruction_jobs/v17_full_interval_geometry_reconstruction_jobs_summary.json
/data2/ego_annotation_outputs/v17_full_interval_geometry_reconstruction_jobs/{trash_1050,task5_tomato_960}/v17_full_interval_geometry_reconstruction_jobs_report.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results_full_interval/v17_geometry_reconstruction_results_summary.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results_full_interval/{trash_1050,task5_tomato_960}/v17_geometry_reconstruction_results_report.json
/data2/ego_annotation_outputs/v17_multi_object_world_mesh_archive/v17_multi_object_world_mesh_archive_summary.json
/data2/ego_annotation_outputs/v17_multi_object_world_mesh_archive/{trash_1050,task5_tomato_960}/v17_multi_object_world_mesh_archive_report.json
/data2/ego_annotation_outputs/v17_interior_owned_hand_annotations/v17_interior_owned_hand_annotations_summary.json
/data2/ego_annotation_outputs/v17_interior_owned_hand_annotations/{trash_1050,task5_tomato_960}/v17_interior_owned_hand_annotations_report.json
/data2/ego_annotation_outputs/v17_deliverable_state/v17_deliverable_state_summary.json
/data2/ego_annotation_outputs/v17_deliverable_state/{trash_1050,task5_tomato_960}/v17_deliverable_state_manifest.json
/data2/ego_annotation_outputs/v17_deliverable_renders/v17_deliverable_render_summary.json
/data2/ego_annotation_outputs/v17_deliverable_renders/{trash_1050,task5_tomato_960}/renders/{v17_overlay_mano_object,v17_reconstruction_3d_world,v17_side_by_side}.mp4
/data2/ego_annotation_outputs/v17_relinearized_hand_capacity_diagnostic/v17_relinearized_hand_capacity_diagnostic_summary.json
/data2/ego_annotation_outputs/v17_relinearized_hand_capacity_diagnostic/{trash_1050,task5_tomato_960}/v17_relinearized_hand_capacity_diagnostic.json
/data2/ego_annotation_outputs/v17_relinearized_residual_object_contact_state/v17_relinearized_residual_object_contact_state_summary.json
/data2/ego_annotation_outputs/v17_relinearized_residual_object_contact_state/{trash_1050,task5_tomato_960}/v17_relinearized_residual_object_contact_state.json
/data2/ego_annotation_outputs/v17_relinearized_residual_factor_coverage/v17_relinearized_residual_factor_coverage_summary.json
/data2/ego_annotation_outputs/v17_relinearized_residual_factor_coverage/{trash_1050,task5_tomato_960}/v17_relinearized_residual_factor_coverage.json
/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state/v17_hand_surface_depth_tail_state_summary.json
/data2/ego_annotation_outputs/v17_hand_surface_depth_tail_state/{trash_1050,task5_tomato_960}/v17_hand_surface_depth_tail_state.json
/data2/ego_annotation_outputs/v17_hand_tail_support_state/v17_hand_tail_support_state_summary.json
/data2/ego_annotation_outputs/v17_hand_tail_support_state/{trash_1050,task5_tomato_960}/v17_hand_tail_support_state.json
/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state/v17_hand_tail_depth_observation_state_summary.json
/data2/ego_annotation_outputs/v17_hand_tail_depth_observation_state/{trash_1050,task5_tomato_960}/v17_hand_tail_depth_observation_state.json
/data2/ego_annotation_outputs/v17_contact_ownership_problem/v17_contact_ownership_problem_summary.json
/data2/ego_annotation_outputs/v17_contact_ownership_problem/{trash_1050,task5_tomato_960}/v17_contact_ownership_problem.json
/data2/ego_annotation_outputs/v17_geometry_source_audit/v17_geometry_source_audit_summary.json
/data2/ego_annotation_outputs/v17_geometry_source_audit/{trash_1050,task5_tomato_960}/v17_geometry_source_audit_report.json
/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state/v17_object_geometry_hypothesis_state_summary.json
/data2/ego_annotation_outputs/v17_object_geometry_hypothesis_state/{trash_1050,task5_tomato_960}/v17_object_geometry_hypothesis_state_report.json
/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed/v17_observed_surface_geometry_seed_summary.json
/data2/ego_annotation_outputs/v17_observed_surface_geometry_seed/{trash_1050,task5_tomato_960}/v17_observed_surface_geometry_seed_report.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/v17_geometry_reconstruction_jobs_summary.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/{trash_1050,task5_tomato_960}/v17_geometry_reconstruction_jobs_report.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/v17_geometry_reconstruction_job.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/{rgb,depth,masks}/
/data2/ego_annotation_outputs/v17_geometry_reconstruction_jobs/trash_1050/{job_id}/cam_K.txt
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results/v17_geometry_reconstruction_results_summary.json
/data2/ego_annotation_outputs/v17_geometry_reconstruction_results/{trash_1050,task5_tomato_960}/v17_geometry_reconstruction_results_report.json
/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit/v17_depth_contact_consistency_audit_summary.json
/data2/ego_annotation_outputs/v17_depth_contact_consistency_audit/{trash_1050,task5_tomato_960}/v17_depth_contact_consistency_audit_report.json
/data2/ego_annotation_outputs/v17_object_geometry_factor_problem/v17_object_geometry_factor_problem_summary.json
/data2/ego_annotation_outputs/v17_object_geometry_factor_problem/{trash_1050,task5_tomato_960}/v17_object_geometry_factor_problem.json
/data2/ego_annotation_outputs/v17_joint_solver_problem/v17_joint_solver_problem_summary.json
/data2/ego_annotation_outputs/v17_joint_solver_problem/{trash_1050,task5_tomato_960}/v17_joint_solver_problem.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/trash_1050/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/v17_contact_mode_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_graph/task5_tomato_960/contact_mode_interval_review_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_graph/v17_contact_mode_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/v17_full_timeline_factor_graph_summary.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/v17_full_timeline_factor_graph_report.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/annotations_v17_full_timeline_graph.json
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph/{trash_1050,task5_tomato_960}/object_meshes_v17_full_timeline_graph.npz
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/trash_1050/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_overlay_mano_object_multi.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_world_reconstruction_3d_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/renders/qc_side_by_side_v17.mp4
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/task5_tomato_960/v17_contact_mode_factor_side_by_side_sheet.jpg
/data2/ego_annotation_outputs/v17_contact_mode_factor_graph_renders/v17_render_summary.json
```

The first three roots above are legacy V17 evidence/QC outputs whose filenames predate the QC naming correction. The current contact-mode-factor render manifests use `qc_` render filenames.

## V16 Failure Analysis

The failures are not isolated rendering bugs.

Trash frame 0182:
- The right hand is visible in the raw image, but the delivered MANO state drifts away from it.
- The object mesh covers disconnected or incomplete bag/can regions.
- The final annotation record contains hand boxes but no retained confidence, source, or residual fields, so a bad hand measurement looks like an accepted hand state.

Trash frame 0260:
- Both rendered hands are attached to the wrong image region near the frame edge.
- This is a single-frame hand hypothesis failure under truncation and occlusion.
- The temporal state did not reject the hypothesis against mask, keypoint, depth, or motion evidence.

Trash frame 0764:
- The visible hand and bag are in contact, but the V16 manipulation view reports them as separated.
- Nearest hand-mesh distance is an insufficient contact model. Contact must be a latent state supported by image adjacency, depth ordering, 3D gap, surface motion, and temporal consistency.

Trash frame 0856:
- The object mask/mesh and both MANO hands are visibly wrong.
- The contact label becomes numerically small even though the underlying hand/object states are bad. A contact residual cannot rescue invalid state estimates.

Trash frame 0949:
- The hand annotation is wrong even without requiring object reasoning.
- V16 lacks an independent hand-state rejection path.

Trash frame 0970:
- A hand region is visible in the raw frame, but the delivered hand list is empty.
- V16 drops the state when measurements disappear; it does not maintain a prediction/update filter with uncertainty through occlusion or detector failure.

Tomato frames around 0480 to 0760:
- The tomato mesh changes shape frame to frame because V16 reconstructs a visible depth patch each frame.
- The tomato requires a persistent near-rigid canonical shape with per-frame pose and small allowed deformation.
- Bowls, plates, tray, sink, and containers can participate in manipulation context, but V16 only carries one object stream.

World reconstruction:
- Skeleton-only hands do not visually read as hands.
- The camera frustum and trajectory do not explain the camera-hand-object relation.
- The 3D panel looks like a diagnostic plot. V17 must render shaded MANO surfaces, shaded object meshes, the current camera image plane/frustum, a local manipulation close-up, and a stable world/camera relationship so the view reads as a reconstruction.

## Root Causes

1. V16 promoted measurements into annotations.

WiLoR hands, object masks, and depth-derived meshes enter the delivered timeline without a strong measurement residual contract. The final JSON often lacks score/source/residual fields for active hand states. The renderer cannot distinguish accepted state, low-confidence measurement, prediction, or rejection.

2. V16 is single-object.

The user-facing task is object pose annotation for manipulated objects. A clip can involve several manipulated or contact-relevant objects. V17 must keep a multi-object timeline where each object has identity, active interval, role, geometry state, and evidence status.

3. V16 has no persistent object shape state.

A tomato, bowl, plate, lid, and trash can are not independent depth patches per frame. Near-rigid objects need a canonical mesh plus per-frame pose. Deformable objects need a canonical or reference surface plus deformation state and temporal regularization.

4. V16 does not implement occlusion smoothing.

Missing hands or objects become predicted states with uncertainty. Bad measurements are rejected or downweighted by image, mask, keypoint, depth, and motion residuals.

5. V16 contact is a diagnostic distance, not contact reasoning.

Contact requires a state variable. A small nearest distance can be false when hand/object states are bad; a large distance can be false when depth or pose is wrong. V17 must estimate contact jointly with hand and object state.

6. V16's 3D view is not designed as a visual explanation.

The 3D panel must show what the annotation means. A line plot with labels cannot satisfy the V17 rendering contract.

## Research Conclusions

The current literature supports a measurement-and-state design over a single replacement model.

HaWoR is directly relevant because it targets egocentric world-space hand motion, combines camera trajectory estimation with hand reconstruction, and includes a motion infiller for missing frames. Its official repository also depends on masked DROID-SLAM and Metric3D, which matches the failure mode of moving egocentric cameras and missing hands. Source: https://github.com/ThunderVVV/HaWoR and https://arxiv.org/abs/2501.02973.

The previous HaWoR branch did not prove HaWoR wrong. It proved the integration was incomplete: the raw camera-local HaWoR hands were partly plausible, while the tested bridge into the existing DROID/object world used a global Sim(3) alignment that produced severe reprojection and hand-scale errors. V17 therefore treats HaWoR as a primary measurement source for world-space hand motion and missing-frame infilling, while making the coordinate bridge itself a residual-checked graph variable.

SAM 2 supports promptable video segmentation and mask propagation, including multi-object video tracking support in the official repository. V17 uses it as one segmentation measurement source; the graph estimates the final object state from masks, depth, tracks, geometry, and contact evidence. Source: https://github.com/facebookresearch/sam2 and https://arxiv.org/abs/2408.00714.

VGGT predicts camera parameters, depth maps, point maps, and 3D point tracks from multiple views. V17 uses it as a geometry and track source for camera/object consistency checks, especially when DROID or monocular depth is unstable. Source: https://github.com/facebookresearch/vggt and https://arxiv.org/abs/2503.11651.

FoundationPose and BundleSDF are relevant for rigid or near-rigid object pose and reconstruction. FoundationPose handles model-based and model-free 6D pose estimation/tracking for novel objects; BundleSDF reconstructs and tracks unknown rigid objects from RGBD sequences with pose-graph optimization. These methods are not suitable as a universal solution for deformable bags, but they are the right class of method for tomato, bowl, plate, lid, can, and container-like objects after masks and depth are available. Sources: https://github.com/NVlabs/FoundationPose, https://arxiv.org/abs/2312.08344, https://github.com/NVlabs/BundleSDF, and https://arxiv.org/abs/2303.14158.

TRELLIS, Hunyuan3D, and PartCrafter can propose complete meshes or structured object priors from images. V17 may use them as prior proposal sources, but any generated mesh must pass visible replay, temporal track support, and physical consistency before entering the delivered annotation. Sources: https://github.com/microsoft/TRELLIS, https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1, and https://github.com/wgsxm/PartCrafter.

WHOLE and EgoGrasp point toward the correct formulation: world-space hand-object interaction reconstruction must model hands and objects jointly over time, especially under occlusion and object entries/exits. V17 uses this formulation as the missing target. Current V17 artifacts provide measurement layers, contact-mode QC, and a sparse evidence-consistency graph, while the full joint hand-object-camera-depth-contact solver remains open. Sources: https://arxiv.org/abs/2602.22209 and https://arxiv.org/abs/2601.01050.

## V17 Pipeline Definition

### Stage 0: Representative Inputs

V17 will run on at least the same two full raw clips as V16:

- task5 tomato, 960 frames;
- trash, 1050 frames.

The named failure frames become required QC anchors for trash: 0182, 0260, 0764, 0856, 0949, 0970. Tomato must include frames around 0480, 0720, and 0760, plus frames where bowls/plates/trays are visible.

### Stage 1: Measurement Store

Every model output enters as a measurement record. The graph solver creates annotation states from residual-checked measurements.

Each measurement stores:

```text
frame_idx
entity_id
measurement_type
source_model
source_checkpoint
coordinate_frame
value
confidence
covariance_or_scale
residuals_against_other_sources
visibility_state
failure_reason
```

No hand, object, camera, or contact state can enter the delivered annotation unless its source measurements and residuals remain traceable.

### Stage 2: Camera And Metric Geometry

V17 will estimate camera/geometry with at least two independent sources:

- masked DROID/HaWoR-style egocentric SLAM for full trajectory continuity;
- VGGT for camera, depth, point maps, and tracks on selected windows;
- UniDepth or Metric3D as dense metric depth measurement.

Dynamic hand/object masks are excluded from static-background SLAM where the method supports masking. The camera state stores uncertainty and residual spikes. The renderer must project accepted hand/object states back through the same camera model, so an impossible camera-hand relation becomes visible in QC.

### Stage 3: Multi-Object Plan

V17 replaces the single object stream with an object roster.

A VLM/video-review agent proposes:

```text
object_id
natural_language_name
active_intervals
role_distribution
prompt_points_or_boxes
expected_physical_behavior
```

Roles are data, not code branches. Examples of roles are manipulated object, support object, container, tool, target surface, and distractor. The downstream pipeline treats all object records uniformly.

Open-vocabulary detection plus SAM 2/Cutie-style video segmentation produces masks for every active object. Object identities are checked by temporal mask propagation, point tracks, depth, and VLM review. A clip can deliver multiple object meshes and poses.

The current V17 multi-object timeline is the first concrete state owner for this stage. It consumes the measurement-store object roster and SAM2 per-object track JSONs, verifies local mask paths, and emits one full-frame `objects` array per raw frame. Each object state contains mask evidence, active interval, role notes, and explicit unresolved geometry/pose fields. It intentionally sets `object_geometry_complete=false`, `object_pose_requirement_met=false`, and `annotation_ready=false` until mesh reconstruction and pose/deformation variables are added.

### Stage 4: Hand Measurement And State Estimation

V17 uses a fixed measurement set:

- HaWoR world-space hand motion and infilled hand trajectory measurements as the primary temporal hand-motion source;
- WiLoR per-frame MANO measurements as an independent image-conditioned hand source;
- HaMeR per-frame MANO measurements from RTMLib crop evidence, with explicit source-coordinate intrinsics and metric-translation residuals;
- VLM-localized visible hand boxes for detector-miss or bad-crop anchors, used as HaMeR crop localization and as image-level hand evidence;
- RTMLib 2D keypoints;
- SAM 2 hand masks or another hand-mask source;
- metric depth over visible hand regions;
- learned hand-motion and hand-object interaction priors.

The delivered hand state is a fixed-lag smoothed MANO trajectory. The solver can choose among measurements because the objective defines residuals, not because code branches on visual cases.

HaWoR contributes motion continuity, world-space trajectory, and missing-frame infill. WiLoR and HaMeR contribute per-frame MANO image evidence through separate model families and crop contracts. The accepted hand state is the graph solution that best satisfies projection, mask, depth, temporal, and contact residuals. When hand sources disagree, V17 records the disagreement and either repairs the state or marks it unresolved.

For each hand and frame, the state can be:

```text
measured_accepted
measured_repaired
predicted_occluded
predicted_detector_miss
rejected_unresolved
outside_view
```

A visible hand cannot silently disappear. If all hand reconstructor measurements fail but the raw image/mask/keypoint evidence shows a hand, V17 emits a predicted state with high uncertainty and a QC flag, then tries local repair from neighboring frames.

### Stage 5: Object Geometry State

Each object has a canonical geometry state and a per-frame state.

Canonical geometry can come from:

- visible RGBD/depth fusion over accepted masks;
- BundleSDF-style rigid object reconstruction when the rigidity residual supports it;
- FoundationPose tracking when a reconstructed/generated mesh is accepted;
- TRELLIS/Hunyuan3D/PartCrafter mesh priors after replay acceptance.

Per-frame object state contains:

```text
T_wo_t        pose for rigid or near-rigid component
D_t           deformation field or surface offsets when needed
M_t           mask support
P_t           point-track support
Q_t           geometry and replay residuals
```

Rigidity is inferred from evidence. If a surface track set can be explained by one SE(3) transform with small residual, the object is near-rigid. If it cannot, the object uses a deformable surface state with smoothness and local area regularization. This is a physical distinction, not category branching.

The current visible-surface builder is the first multi-object RGBD measurement layer for this stage. It treats every visible object uniformly: resize the verified SAM2 mask to the metric-depth raster, keep finite metric depth inside the mask, backproject with the depth archive intrinsics, transform through the graph annotation camera pose, and save an object-frame surface mesh with object id and frame index. It rejects rows with missing depth, too few valid depth pixels, too few sampled vertices, or disconnected surface support. This produces world-coordinate visible surface evidence, but it does not estimate `G_o`, `T_wo,t`, or `D_o,t`; the V3 object-geometry and object-pose families remain unmet.

The current geometry-state diagnostic splits each object's center-normalized surfaces into even and odd frame partitions and measures cross-partition nearest-surface residuals. It is a measurement of envelope repeatability after subtracting the per-frame surface center. It is not a rigidity proof because it ignores material correspondences and removes translation by construction. Current reports therefore keep `persistent_visible_surface_candidate_count=0`, `rigid_pose_candidate_count=0`, `object_geometry_complete=false`, and `object_pose_requirement_met=false` even when an object has a repeatable envelope.

The current object-track dataset exporter prepares the next evidence source for this stage. It writes per-object frame datasets only where the multi-object timeline has a visible SAM2 mask and the UniDepth metric archive has depth for the same raw frame. The exported rows remain tracking inputs; they become object-pose evidence only after a material point tracker produces correspondences and a rigid/deformable motion test accepts them.

For tomato, V17 enforces a near-rigid shape prior: one canonical tomato mesh with pose and small deformation, plus visible-surface updates only when they do not contradict the canonical shape.

For trash bag, V17 uses deformable surface state and avoids pretending there is a rigid 6D pose.

### Stage 6: Contact State And Physical Consistency

Contact is a latent state per hand patch and object surface region:

```text
no_contact
candidate_contact
sticking
sliding
supporting
occluded_contact
rejected_unresolved
```

The graph estimates contact from:

- 2D hand/object mask adjacency;
- depth ordering and depth gap;
- 3D hand-object SDF gap;
- nonpenetration;
- relative motion of hand patches and object surface tracks;
- object acceleration consistency when object mass/rigidity evidence makes the test meaningful;
- deformation consistency for nonrigid surfaces.

Contact cannot be asserted from nearest distance alone. Contact cannot be rejected when image/depth/track evidence supports contact but the current 3D state is inconsistent; that case becomes a state repair target.

### Stage 7: Full Nonlinear Graph Target

This section defines the still-open full nonlinear solver target. The implemented V17 graphs are the contact-mode graph and sparse evidence-consistency graphs described in the Status section: the contact-mode graph estimates binary contact/no-contact/unobserved modes from fixed V17 geometry and image evidence, and the sparse geometry graph optimizes object translation, small-angle object rotation, local contact-patch correspondences, and hand camera-ray depth corrections. Camera trajectory, MANO articulation and shape, object mesh topology, full object geometry, and nonlinear contact physics stay fixed. The current sparse graph only shows consistency of the accepted evidence layer under that limited variable set; it is not full annotation closure.

The generated joint-problem contract is the current machine-readable definition of this target. It compares the current sparse graph against the required variable families and fails closed by setting `v3_solver_complete=false`, `annotation_ready=false`, and `deliverable_ready=false`. The next optimizer must either create graph variables for those families or record a source-backed reason for fixing one family. A graph over one legacy object stream cannot satisfy this stage even if its local contact residuals are small.

The full V3-class solver should implement the prediction/update idea as a fixed-lag nonlinear factor graph. A simple constant-velocity or constant-acceleration prior is not the process model for hand-object manipulation. It can appear only as a weak local smoothness regularizer. The actual process terms are learned priors and physically grounded residuals.

State variables:

```text
T_wc_t                    camera pose
H_h,t                     MANO hand pose/shape/global transform
V_h,t                     hand velocity latent
Z_h,t                     learned hand-motion latent
G_o                       object canonical geometry
T_wo,t                    object pose for near-rigid components
D_o,t                     object deformation state
C_h,o,t                   contact mode variables
U_*                       uncertainty/covariance variables
```

Residuals:

```text
camera motion and reprojection
hand MANO prior
hand 2D keypoint reprojection
hand mask silhouette
hand metric-depth support
hand temporal velocity/acceleration
object mask replay
object depth replay
object point-track consistency
rigidity or deformation energy
object shape-prior consistency
hand-object nonpenetration
contact equality / sliding / support residuals
measurement confidence calibration
HaWoR motion-infill prior
hand-object correspondence prior
hand-object generative plausibility prior
```

The learned process model predicts through missing measurements with growing uncertainty. For hands, HaWoR-style motion infilling provides the primary learned temporal proposal. For contact, TOCH-style spatio-temporal object-to-hand correspondence provides a learned contact refinement prior. For broader hand-object plausibility, a G-HOP-style diffusion prior can propose or score physically plausible hand-object states. These learned priors enter as factors in the objective; they do not override image, mask, depth, object-track, or nonpenetration evidence.

Measurements update the state only when their residuals are plausible. Outlier measurements remain in the measurement store but do not become accepted states.

### Stage 8: QC

V17 QC is frame-local and timeline-local.

Required frame anchors:

- trash 0182: right hand must stay on the visible hand or be flagged unresolved;
- trash 0260: wrong edge hand hypotheses must be rejected or repaired;
- trash 0764: hand-bag contact must be represented as contact or unresolved repair target, not isolated separation;
- trash 0856: bad hand/object state must not produce a confident contact label;
- trash 0949: visible hands must pass hand residual checks;
- trash 0970: visible hand evidence requires an accepted, predicted, or unresolved hand state;
- tomato 0480/0720/0760: tomato mesh must preserve persistent near-rigid shape;
- tomato context: bowls/plates/trays must appear in the multi-object roster when visible and relevant to the manipulation.

Aggregate QC is insufficient. Every deliverable manifest must include:

```text
named_anchor_status
measurement_source_coverage
accepted_state_coverage
rejected_unresolved_frames
hand_residual_summary
object_residual_summary
contact_residual_summary
render_frame_count_qc
```

### Stage 9: Rendering

The V17 world view must be rebuilt.

The 3D panel must show:

- shaded MANO hand meshes with skeleton overlays as secondary cues;
- shaded object meshes with separate colors per object;
- current camera frustum with textured image plane or raw-frame thumbnail plane;
- a line of sight from camera to hand-object region;
- stable world axes and scale;
- head trajectory as a subtle path;
- local manipulation close-up with contact patches;
- uncertainty/rejection status when state is predicted or unresolved.

The side-by-side video remains full-length and synchronized with the raw video. Debug plots can exist as QC artifacts, but the deliverable must read as a reconstruction.

### Stage 10: Deliverables And Closure

For every V17 sample:

```text
overlay_mano_object_multi.mp4
world_reconstruction_3d_v17.mp4
side_by_side_v17.mp4
annotations_v17_full.json
measurements_v17_full/
object_meshes_v17_full/
v17_manifest.json
v17_anchor_qc.json
```

Closure requires:

- output frame count and fps equal raw video;
- all named V16 failure frames pass or are explicitly marked unresolved with correct evidence;
- no visible hand disappears silently;
- no per-frame object depth patch is delivered as a persistent near-rigid object;
- multiple manipulated/context objects are represented where visible and relevant;
- contact labels are state estimates with supporting residuals;
- world reconstruction is visually legible as a 3D scene.
