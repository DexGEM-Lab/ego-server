# Pipeline V7: Replayable Mesh Priors and Visibility-Aware Surface State

## Starting Point

V6 has a repaired 31-frame wild-rice active-stem archive:

- object mesh archive: `/data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz`
- overlay video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_mesh_surface_contact_review_repaired2539_2520_2550/mesh_surface_contact_review.mp4`
- side-by-side video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_side_by_side.mp4`
- 3D video: `/data2/ego_annotation_outputs/representative_wild_rice/v6_world_reconstruction_repaired2539_2520_2550/world_reconstruction_3d.mp4`

The V6 object geometry is still a per-frame measured or repaired mesh stream. V6 also supplies sparse CoTracker factors with a continuous ready chain from frame 2534 to 2550, but transporting full meshes through those factors fails image replay. V7 therefore treats sparse factors as temporal evidence and tests explicit object priors through the same replay checks used for delivered meshes.

## Research Grounding

The relevant open-source frontier is split across four mechanism classes:

- SAM 3 provides open-vocabulary concept and exemplar segmentation in images and video, including text prompts, points, boxes, masks, and a detector-tracker design. Source: https://github.com/facebookresearch/sam3
- SAM 3D Objects reconstructs full 3D shape, texture, pose, and layout from masked objects in natural images. Source: https://github.com/facebookresearch/sam-3d-objects
- CoTracker3 tracks arbitrary or quasi-dense points through video. V6 already validated that its mask- and mesh-attached tracks can become sparse temporal factors. Source: https://github.com/facebookresearch/co-tracker
- 4DTAM and Shape of Motion support the representation choice: dynamic surfaces should be driven by depth, long-range tracks, low-dimensional motion, and visibility. Sources: https://github.com/muskie82/4dtam and https://shape-of-motion.github.io/

The research implication is precise: generated 3D models are candidate priors, and dynamic reconstruction papers inform temporal state variables. A generated or dynamic prior becomes delivered object annotation only after it replays against the measured video evidence.

## Representation

V7 keeps the V6 per-frame object meshes as measured observations. It adds two candidate state layers.

### Mesh-Prior Candidate

For each generated prior mesh `P`, V7 estimates a per-frame metric similarity transform to the measured object surface:

```text
X_t = s_t R_t P + p_t
```

where `s_t` is scale, `R_t` is a proper rotation, and `p_t` is translation in world coordinates. The first harness uses robust PCA and nearest-neighbor distances to align the prior to each measured frame. It reports both visible-surface coverage and hidden-surface conflict. Visible coverage asks whether the measured camera-visible sheet lies on the prior. Hidden conflict asks how much generated prior surface is unsupported by the measured sheet. Hidden conflict is diagnostic for completion quality, while delivery readiness comes from visible coverage plus image-depth, contact, SDF, and visual replay.

### Visibility-Aware Surface State

The temporal state is a set of object surfels or vertices with frame-local visibility and uncertainty:

```text
S_k = canonical surfel position and normal
M_tk = low-dimensional motion for surfel k at frame t
V_tk = model-produced visibility/support confidence
C_th = contact likelihood for hand region h at frame t
```

The surfel state is category-agnostic. SAM3/SAM2 masks, VLM verifier points, CoTracker tracks, UniDepth depth, VGGT cameras, MANO vertices, and contact rows all enter as data. Model evidence, geometry, and physics decide object behavior.

## Factor Graph

The graph has these node classes:

- `T_wc_t`: head camera pose at frame `t`, initialized from the accepted VGGT world trajectory and held fixed unless a later version tests camera refinement.
- `H_t`: MANO hand pose and vertices, initialized from the accepted hand stream and depth/contact refit.
- `O_t`: delivered per-frame observed object mesh, fixed as measurement.
- `P`: optional generated prior mesh, such as SAM 3D Objects or Hunyuan3D output.
- `A_t`: similarity or low-dimensional deformation variables mapping `P` or canonical surfels into frame `t`.
- `S_k`: canonical surfel variables for temporally supported surface patches.
- `z_tk`: continuous contact likelihood variables between hand regions and object surface patches.

The graph has these edge classes:

- image-depth observation edge: projected object surface must match the model-produced mask silhouette and metric depth under `T_wc_t`;
- prior alignment edge: `A_t(P)` or `A_t(S_k)` should stay near the measured object surface where that surface is visible;
- CoTracker correspondence edge: a tracked 2D point lifted by depth and attached to the mesh should remain near the same moving surfel across neighboring frames;
- local surface regularity edge: neighboring surfels should preserve short-range shape where both are visible, using robust penalties so occlusion and changing support can fail locally;
- temporal motion edge: `A_t` should change smoothly under a constant-velocity or low-dimensional SE(3) motion model;
- nonpenetration edge: hand vertices outside active contact patches should stay outside the object signed-distance field;
- contact equality edge: only hand patches with visual, geometric, and temporal support receive near-zero signed-distance residuals;
- contact dynamics edge: when object acceleration, contact patch, and hand motion are all observable, the contact impulse direction must be consistent with the object's measured acceleration up to unknown mass/friction bounds.

The contact variable `z_tk` controls attraction. A frame lacking contact evidence contributes nonpenetration and observation edges, while a contact-supported patch contributes a near-surface equality edge.

## Objective

V7 minimizes a robust weighted least-squares objective:

```text
min_X
  sum_t rho_mask_depth(E_replay(O_t or A_t(P), I_t, D_t, K_t, T_wc_t))
+ sum_t rho_align(d(A_t(P), O_t_visible))
+ sum_(t,k) rho_track(||A_t(S_k) - lift(track_tk)||)
+ sum_(t,k,j) rho_arap(||(S_k - S_j)_t - (S_k - S_j)_(t-1)||)
+ sum_t rho_motion(||A_t - predict(A_t | A_(t-1), A_(t-2))||)
+ sum_(t,h) rho_nonpen(max(0, -sdf_object(H_t,h)))
+ sum_(t,h) z_th rho_contact(|sdf_object(H_t,h)|)
```

`rho` is robust, because tracks, masks, monocular depth, and hand fits all fail locally. Replay terms are delivery checks as well as optimization residuals: a solution that improves the objective but fails silhouette/depth/contact/SDF replay is rejected.

## Solver

V7 uses two solver tiers:

1. Mesh-prior replay harness: align each candidate mesh to measured surfaces, archive it, then run existing z-buffer/contact/SDF QC. Complete priors are judged by visible-surface coverage and image-depth replay; hidden-surface conflict remains in the report because a full mesh can contain unseen geometry that the video cannot refute in one view.
2. Dynamic surfel graph: optimize low-dimensional per-frame motion and surfel offsets with SciPy or PyTorch least squares on bounded frame windows. The first target is the V6 wild-rice 2534 to 2550 interval, because it has continuous sparse track factors and verified measured geometry.

The solver keeps measurements fixed while testing prior completion only where measured evidence and visibility allow it.

## Acceptance Checks

For any V7 candidate archive:

- z-buffer replay must preserve silhouette and metric depth against the object mask and UniDepth depth;
- mesh-surface contact must be recomputed on the candidate archive;
- selected-contact SDF must show near-surface contact and object exterior consistency;
- full-hand SDF must show hand/object exterior consistency;
- sparse track residuals must remain within the V6 factor tolerance on ready pairs when the target has model-produced temporal factors;
- stakeholder render must show head trajectory, MANO hands, object mesh, contact markers, and captions clearly.

## First Implemented Artifact

### Generated-Prior Replay Harness

`scripts/archive_aligned_mesh_prior_v7.py` creates a replayable mesh-prior archive:

```bash
.venv/bin/python scripts/archive_aligned_mesh_prior_v7.py \
  --mesh-prior <candidate_mesh.obj-or-ply> \
  --observed-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --frame-start 2534 \
  --frame-end 2550 \
  --output-mesh-archive <out>/aligned_prior_meshes_world.npz \
  --output-json <out>/qc_aligned_mesh_prior_v7.json
```

The output mesh archive can be passed directly to `render_mesh_zbuffer_qc_v3.py`, `diagnose_mesh_surface_contact_v3.py`, `diagnose_volume_sdf_contact_v3.py`, and `diagnose_hand_object_sdf_penetration_v3.py`.

This first V7 artifact is a falsification harness. It decides whether a real generated mesh prior deserves to enter the dynamic graph.

Negative-control run:

```bash
.venv/bin/python scripts/archive_aligned_mesh_prior_v7.py \
  --mesh-prior /data2/ego_annotation_outputs/representative_wild_rice/v3_trellis_frame2548_raw/trellis_mesh.ply \
  --observed-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --frame-start 2538 \
  --frame-end 2540 \
  --output-mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/aligned_prior_meshes_world.npz \
  --output-json /data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/qc_aligned_mesh_prior_v7.json
```

The TRELLIS prior is rejected before contact checks:

- alignment bidirectional p95 median: 0.842 m;
- visible-surface coverage p95 median: 0.022 m;
- alignment threshold: 0.010 m;
- z-buffer replay median silhouette IoU: 0.152;
- z-buffer replay median depth absolute error: 0.083 m;
- z-buffer replay p95 depth absolute error median: 0.124 m.

Artifacts:

- alignment report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/qc_aligned_mesh_prior_v7.json`
- z-buffer report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/zbuffer_qc/qc_mesh_zbuffer_projection_v3.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_trellis_prior_replay_negative_control_2538_2540/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`

The rejected result proves the harness is live: a visually plausible generated mesh becomes object pose only after visible-surface coverage and image-depth replay agree with the observed video.

`scripts/run_v7_generated_prior_replay_qc.py` wraps the same acceptance logic for future SAM 3D Objects, Hunyuan3D, TRELLIS, or Mesh4D outputs. It first replays the observed target archive against the supplied mask, metric depth, camera pose, and intrinsics contract. If that measured target fails silhouette IoU, visible-inside, or depth thresholds, the wrapper returns `invalid_observed_target` before prior alignment, because prior acceptance would otherwise mix target-contract failure with generated-mesh failure. After the observed target passes, the wrapper runs prior alignment, then runs image-depth z-buffer replay only when visible-surface coverage remains eligible for delivery. The TRELLIS negative-control wrapper report rejects the prior because visible-surface coverage and every replay check fail:

The wrapper now fails fast after prior alignment when visible-surface coverage already exceeds the delivery threshold. That state is a measured rejection: the report contains observed-target replay evidence, the aligned archive, alignment metrics, `rejection_stage: visible_surface_alignment`, and `not_evaluated_delivery_keys` for image-depth checks skipped because they cannot repair failed visible-surface coverage. This preserves the acceptance contract while avoiding long CPU z-buffer videos for dense priors that have already failed a delivery-critical geometry factor.

- alignment bidirectional p95: 0.906 m, threshold 0.010 m;
- visible-surface coverage p95: 0.023 m, threshold 0.010 m;
- median silhouette IoU: 0.152, threshold 0.900;
- median visible-inside fraction: 0.152, threshold 0.900;
- median z-buffer p95 depth error: 0.124 m, threshold 0.010 m.

Wrapper artifact:

- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_2538_2540/qc_v7_generated_prior_replay.json`

Corrected visible-surface replay artifact:

- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_visible_semantics_2538_2540/qc_v7_generated_prior_replay.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_generated_prior_replay_trellis_negative_control_visible_semantics_2538_2540/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`

The corrected report keeps strict full-surface alignment as a diagnostic and uses these delivery checks: visible-surface coverage p95, silhouette IoU, visible-inside fraction, and z-buffer p95 depth. The TRELLIS prior remains rejected:

- visible-surface coverage p95: 0.025 m, threshold 0.010 m;
- median silhouette IoU: 0.158, threshold 0.900;
- median visible-inside fraction: 0.159, threshold 0.900;
- median z-buffer p95 depth error: 0.123 m, threshold 0.010 m.

Guarded replay matrix:

- matrix JSON: `/data2/ego_annotation_outputs/v7_generated_prior_replay_matrix_guarded_20260607.json`
- matrix Markdown: `/data2/ego_annotation_outputs/v7_generated_prior_replay_matrix_guarded_20260607.md`

The guarded matrix separates target validity from generated-prior validity. The observed target archives pass their own replay checks on wild-rice, trash, and mop. TRELLIS and Hunyuan priors are still rejected because the generated visible surface misses measured geometry and image-depth replay fails. This means V7 has a working acceptance harness and valid measured targets, but it has not accepted a generated complete object mesh prior.

`configs/v7_prior_replay_targets.json` stores the measured target contracts for the current representative samples, including the intrinsics source used for physics checks. `scripts/run_v7_prior_candidate_batch.py` consumes generated mesh candidates as `target_id|candidate_name|mesh_path|note`, either through repeated `--candidate` arguments or a discovery-produced `--candidate-file`. It runs the same guarded replay wrapper and writes a compact matrix with `summarize_v7_prior_replay_matrix.py`. The batch report records `samples`, `max_faces`, `vertex_splat_radius_px`, and whether z-buffer replay was full-fidelity. A bounded face-count run is diagnostic only; delivery evidence requires a full-fidelity replay report. With `--run-physics --render-deliverables`, the batch proceeds through physics QC and renders videos only for candidates whose replay and physics reports are both accepted.

Candidate rows must resolve to unique output directories before any replay runs. A duplicate sanitized candidate name for the same target is a provenance error because it would mix reports from different meshes.

`scripts/run_v7_candidate_physics_qc.py` is the second-stage acceptance wrapper. It refuses replay reports whose status is not `accepted` and also refuses bounded-face diagnostic replay. Physics QC runs only after full-fidelity z-buffer replay. When a target has a `track_qc` block, the batch driver also requires accepted topology-aware track QC before physics. The wrapper requires mesh-surface contact diagnostics to read the object-mask and hand evidence, uses the V6-validated contact threshold contract by default, runs selected-contact local-crop SDF at 1 mm pitch when contact evidence exists, and always runs full-window hand/object SDF nonpenetration at 3 mm pitch on the aligned mesh archive. The batch driver can call this stage with `--run-physics`; rejected replay reports and diagnostic replay reports receive explicit skipped-physics records instead of running contact checks against failed or non-delivery evidence. A generated mesh can become an object-pose candidate for delivery only after visible replay, temporal track QC, and this physics wrapper pass.

The physics wrapper treats contact as an evidence-dependent claim. It runs selected-contact SDF only when mesh-surface contact diagnosis finds reliable or geometry-backed temporal contact rows. A window whose hand/object rows support no contact can pass nonpenetration without claiming contact; a contact-supported window must also pass near-surface selected-contact checks.

### MANO Measurement Repair

`scripts/refit_mano_articulation_mask_depth_v3.py` now supports an explicit RTMLib 2D keypoint factor through `--rtmlib-json` and `--rtmlib-prompts`. The factor constrains MANO joints to the model-detected anatomical keypoints selected for the same hand track that drives SAM2. The refit report keeps both the old base-joint reprojection metric and the new RTMLib reprojection metric, so a candidate cannot pass by fitting only a mask silhouette while ignoring the measured finger skeleton.

The same refit report also separates all-projected-vertex depth residuals from hand-mask-interior depth residuals. The all-vertex p95 remains diagnostic. Acceptance can use the interior metric only when enough projected vertices lie well inside the SAM hand mask, because boundary pixels mix hand, object, and background depth at contact edges.

This repair was tested on detergent-bottle frames 672 to 678. RTMLib reduced median anatomical reprojection to about 24 px and the median pose delta to about 0.67 rad. The run still selected only 3 of 7 hands under V7 selection: later frames exceeded the 60 mm reliable-depth threshold or pose/RTMLib acceptance. The result is a useful hand-measurement improvement, not a deliverable acceptance for detergent.

`scripts/check_v7_candidate_track_surface_qc.py` is the topology-aware track-consistency wrapper. Sparse CoTracker edge reports store vertex indices from the observed measured mesh archive, while generated candidate meshes have unrelated topology. The wrapper therefore uses those vertex indices only to recover the model-produced 3D track observations on the observed archive, then queries the candidate archive by nearest surface in each frame. It reports candidate-surface correction and cross-frame residuals under the same rigid pair factors used by the V6/V7 track graph. The raw report now carries the target-level decision directly: `status: accepted` and `annotation_ready: true` require enough tracks, enough accepted edges, p95 pair residual below the target threshold, and p95 surface-correction displacement below the target threshold. All three representative targets require this check before physics. Wild-rice uses the repaired-archive factors over frames 2538 to 2540; the accepted measured archive gives 42 accepted edges, p95 pair residual 8.29 mm, and p95 surface correction 0 mm. Trash and mop use A800-generated CoTracker factors localized to the local output tree. Trash frames 865 to 870 have 4 of 5 ready neighboring pairs, 1437 accepted measured-track edges, and p95 pair residual 7.80 mm. Mop frames 759 to 765 have 6 of 6 ready neighboring pairs, 277 accepted measured-track edges, and p95 pair residual 9.20 mm. A generated mesh cannot enter physics on any representative target unless its surface remains compatible with these model-produced temporal factors.

`scripts/write_v7_cotracker_factor_remote_job.sh` packages the same temporal-factor path for trash and mop. The remote job runs CoTracker on the existing model-produced object masks, lifts tracks through the accepted metric depth and camera annotations, attaches them to the measured target mesh archive, and fits pairwise rigid factors. If those reports produce enough ready factors, the target JSON can add the same `track_qc` block used by wild-rice. This keeps temporal consistency evidence model-driven and target-data driven rather than encoded as object-category logic.

`scripts/localize_v7_cotracker_factor_report.py` rewrites synced remote CoTracker report paths to local paths before they enter target config. Pair-factor reports reference sparse-edge JSON files, and sparse-edge reports reference the measured mesh archive used for vertex attachment. The localizer updates those provenance paths together so the topology-aware track checker consumes the local synced evidence rather than stale A800 absolute paths.

`scripts/render_v7_candidate_deliverables.py` is the final V7 delivery wrapper for an accepted candidate. It refuses any replay or physics report whose status is not `accepted` and `annotation_ready`, and it also refuses diagnostic replay controls. The wrapper then calls the existing overlay and world-coordinate renderers. The wrapper writes:

- MANO/object overlay video: `overlay/mesh_surface_contact_review.mp4`
- side-by-side annotated video plus 3D reconstruction: `world/world_reconstruction_side_by_side.mp4`
- standalone 3D world animation: `world/world_reconstruction_3d.mp4`
- delivery manifest with structural video QC: `v7_candidate_deliverables_manifest.json`

This command is intentionally downstream of replay and physics acceptance, so a rejected generated mesh cannot become a stakeholder render by accident.

`scripts/fuse_v7_sim3_prior_observed_surfaces.py` is the repair path for a generated prior that is close enough to align but still misses visible surface detail. It maps model-produced mask/depth observations back into the prior's canonical coordinates using the same per-frame Sim3 rows from the replay alignment report, fuses those observations with sampled prior surface points, and rearchives the fused mesh through the original Sim3 rows. It fails before meshing when the canonical observed extent or Sim3 scale drift is physically implausible. Existing trash TRELLIS replay hits this failure: the observed depth points spread to a 10.9 m canonical extent, so fusion would only hide the bad prior alignment.

Full generated-candidate batch:

- batch root: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_20260607_180135/replay_batch`
- matrix: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_20260607_180135/replay_batch/qc_v7_prior_candidate_batch_matrix.md`
- visual QC sheet: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_20260607_180135/replay_batch/qc_v7_prior_candidate_batch_visual_sheet.png`
- candidates: 12 complete-mesh priors from Hunyuan3D, Hunyuan3D 2.1, and TripoSG across mop, trash, and wild-rice;
- outcome: 12 rejected, 0 accepted;
- observed targets: all replayed successfully before candidate evaluation;
- physics and deliverables: skipped for every candidate because replay did not accept any generated prior.

`scripts/render_v7_prior_batch_visual_qc.py` renders the observed-target replay and generated-prior replay side by side for every candidate in a batch report. The sheet ties visual inspection to the same metrics used for rejection, so the rejection reason stays anchored to the video evidence. `scripts/run_v7_prior_candidate_batch.py` writes this sheet after real batch runs unless `--skip-visual-qc` is supplied.

The batch falsifies single-image complete priors as V7 closure for the current representative set. Mop priors miss long thin tool geometry, trash priors overlap the lid silhouette but are 56 to 111 mm wrong in depth, and wild-rice priors cover only a small fraction of the active stem. V7 therefore moves to video-conditioned geometry sources before considering final delivery.

PartCrafter generated-candidate batch:

- batch root: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_partcrafter_20260608_004553/replay_batch`
- matrix: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_partcrafter_20260608_004553/replay_batch/qc_v7_prior_candidate_batch_matrix.md`
- visual QC sheet: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_partcrafter_20260608_004553/replay_batch/qc_v7_prior_candidate_batch_visual_sheet.png`
- candidates: four complete-mesh priors from PartCrafter across mop, trash, and wild-rice;
- outcome: four rejected, zero accepted;
- observed targets: all replayed successfully before candidate evaluation;
- best generated visible-surface p95: trash at 17.7 mm, still above the 10 mm delivery threshold;
- physics and deliverables: skipped for every candidate because replay did not accept any generated prior.

The PartCrafter batch expands the falsification beyond Hunyuan3D, Hunyuan3D 2.1, and TripoSG. It also shows that producing a structured part mesh is not enough: the visible surface still has to land on the measured video geometry within the V7 tolerance before temporal factors or contact physics can be meaningful.

InstantMesh generated-candidate batch:

- batch root: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_instantmesh_clean_20260608_011616/replay_batch`
- matrix: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_instantmesh_clean_20260608_011616/replay_batch/qc_v7_prior_candidate_batch_matrix.md`
- visual QC sheet: `/data2/ego_annotation_outputs/v7_generated_candidate_batch_instantmesh_clean_20260608_011616/replay_batch/qc_v7_prior_candidate_batch_visual_sheet.png`
- candidates: four watertight InstantMesh OBJ priors for mop, trash, and wild-rice;
- outcome: four rejected, zero accepted;
- observed targets: all replayed successfully before candidate evaluation;
- best generated visible-surface p95: wild-rice frame 2539 at 33.3 mm, still above the 10 mm delivery threshold;
- physics and deliverables: skipped for every candidate because replay did not accept any generated prior.

The clean InstantMesh run fixes the earlier stale-input contamination: the log reports four input images, matching the four intended current crops. Its rejection strengthens the same conclusion as the prior batches: independent single-image mesh priors do not provide the measured visible surface needed by the current representative clips.

BundleSDF was rechecked as the direct RGB-D object-reconstruction path because its input contract matches RGB frames, depth PNGs, masks, and one `cam_K.txt`. `scripts/export_bundlesdf_dataset_v3.py` now supports the newer frame-indexed depth NPZ schema and can read annotation-VGGT intrinsics, but it refuses export when those intrinsics vary across the sequence. The mop V7 target frames 759 to 765 have annotation-VGGT focal spread of about 29 px in fx and 27 px in fy, so a single-`cam_K.txt` BundleSDF dataset would silently change the accepted replay camera model. BundleSDF remains available for targets with constant intrinsics or a justified constant-K source; it is not a valid V7 completion route for this mop contract.

### SAM 3D Objects Candidate Source

SAM 3D Objects is the next complete-mesh source tested by V7. Its official setup requires a Linux NVIDIA GPU with at least 32 GB VRAM, Hugging Face checkpoint access for `facebook/sam-3d-objects`, and the `hf` checkpoint directory containing `pipeline.yaml`. The official single-object API accepts an RGB image plus a mask and the underlying pipeline decodes both `mesh` and `gaussian` representations. V7 exports the decoded triangle mesh directly and keeps the GLB and Gaussian only as secondary visual evidence.

Remote A800 job package:

- runner: `scripts/remote_run_sam3d_objects_mesh_v7.py`
- job writer: `scripts/write_v7_sam3d_objects_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_sam3d_objects_outputs`
- setup tmux session: `ego_v7_sam3d_objects_setup`

The job evaluates the same representative classes already used by the guarded replay matrix:

- wild-rice active stem frames 2539 and 2545;
- trash lid frame 880;
- mop frame 759 from the current V7 replay target.

The SAM 3D Objects output is a generated complete-object mesh candidate, not an accepted annotation. Each exported mesh must pass `run_v7_generated_prior_replay_qc.py` against the corresponding observed target archive before it can enter contact or final-render checks.

Current A800 setup evidence: `ego_v7_sam3d_objects_setup` reached the Hugging Face checkpoint download and returned `GatedRepoError` 403 for `facebook/sam-3d-objects`. The remote token exists, but Hugging Face reported that the account lacks access to the gated model. The waiter was stopped so an access-blocked setup cannot consume the next free GPU. SAM 3D Objects can resume from the same job package after checkpoint access is granted or the checkpoint directory is supplied.

### TripoSG Candidate Source

TripoSG is the next accessible complete-mesh source. Its official inference script downloads `VAST-AI/TripoSG` and `briaai/RMBG-1.4`, runs image-conditioned shape synthesis, and exports a triangle mesh as GLB. The V7 wrapper uses the already model-produced object alpha crops rather than category-specific preprocessing, then writes both GLB and PLY for replay.

Remote A800 job package:

- runner: `scripts/remote_run_triposg_shape_v7.py`
- job writer: `scripts/write_v7_triposg_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_triposg_prior_outputs`
- setup tmux session: `ego_v7_triposg_setup`
- inference waiter tmux session: `ego_v7_triposg_wait`

The output has the same status as every generated prior: it is a complete-object mesh hypothesis until `run_v7_generated_prior_replay_qc.py` accepts it against measured target replay.

Current evidence: the repository requirements leave `transformers` unpinned, which first installed `transformers 5.10.2`. That version imports `torch.float8_e8m0fnu`, a dtype absent from the A800 venv's `torch 2.5.1+cu121`, so the real failure was a Hugging Face stack incompatibility before model loading. The official TripoSG Hugging Face Space pins `transformers==4.49.0`; V7 pins `transformers==4.49.0`, `trimesh==4.5.3`, `scipy==1.11.4`, and `huggingface_hub<1.0`. After that repair, TripoSG produced four PLY/GLB mesh candidates for mop, trash, and wild-rice. The full-fidelity V7 candidate batch rejected all four by replay before physics or deliverable rendering.

### PartCrafter Candidate Source

PartCrafter is a public structured mesh generator whose Hugging Face model metadata reports `gated=False` for `wgsxm/PartCrafter`; the weights total about 4.0 GB. The model produces part-level meshes from a single image. V7 uses it as another generated complete-mesh prior source, because its compositional latent representation differs from Hunyuan3D, TripoSG, Pixal3D, and InstantMesh.

Remote A800 job package:

- case-plan builder: `scripts/build_partcrafter_case_plan_v7.py`
- runner: `scripts/remote_run_partcrafter_shape_v7.py`
- job writer: `scripts/write_v7_partcrafter_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_partcrafter_prior_outputs`

`scripts/build_partcrafter_case_plan_v7.py` asks a VLM to choose the PartCrafter `num_parts` conditioning variable from the masked RGBA crops, then writes a case-argument file consumed by the A800 job. The current OCC Responses plan selected these counts from visible crop geometry: wild-rice frame 2539 has 4 parts, wild-rice frame 2545 has 5 parts, trash frame 880 has 3 parts, and mop frame 759 has 3 parts. These counts are model-produced data, not object-family branch logic.

The V7 runner calls the PartCrafter pipeline directly and rejects `None`, tiny, degenerate, or non-finite part meshes. This is stricter than the official script, which substitutes a dummy triangle mesh on decode failure. Each merged part composition is exported as `partcrafter_mesh.ply` and must pass the same generated-prior replay, track, physics, and deliverable checks as every other candidate source.

Current A800 evidence: the first PartCrafter setup installed `transformers 5.10.2`, which imports `torch.float8_e8m0fnu` and failed against `torch 2.5.1+cu121` before pipeline construction. V7 pins `transformers==4.49.0` and `huggingface_hub<1.0` after upstream requirements. The repaired setup log `setup_partcrafter_transformers449.log` ended with `EXIT:0`, wrote `setup_complete.marker`, confirmed the `wgsxm/PartCrafter` snapshot was cached, and imported `PartCrafterPipeline` with CUDA-visible torch. The completed run wrote four merged `partcrafter_mesh.ply` candidates. The full-fidelity V7 batch rejected all four by visible-surface alignment before track QC, physics QC, or deliverable rendering.

### InstantMesh Candidate Source

InstantMesh is another complete-mesh source queued for V7. Its official command-line path accepts image inputs, can skip background removal with `--no_rembg`, and exports OBJ meshes. V7 feeds it the same model-produced RGBA object crops used by TripoSG, so the downstream replay contract remains unchanged.

Remote A800 job package:

- job writer: `scripts/write_v7_instantmesh_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_instantmesh_prior_outputs`
- setup tmux session: `ego_v7_instantmesh_setup`
- inference waiter tmux session: `ego_v7_instantmesh_wait`

The InstantMesh setup is allowed to run while A800 GPUs are occupied because it prepares the repo and venv. The inference waiter uses the same per-GPU lock as Hunyuan and TripoSG, so the first free GPU is reserved by one V7 job without preventing a second V7 job from using a different free GPU.

Current runtime evidence: InstantMesh imports `rembg` at module load even when `--no_rembg` is passed. The previous setup checked diffusion and reconstruction imports but missed `rembg -> onnxruntime`, so inference failed after a free GPU was selected. The setup now installs `onnxruntime==1.16.3` and imports `rembg` during setup and again at runtime before invoking `run.py`. A later setup run failed while caching `sudo-ai/zero123plus-v1.2` because Hugging Face reported a partial `model.safetensors` blob: expected 1,264,217,240 bytes, got 398,785,453 bytes. The setup deletes the exact InstantMesh and zero123plus cache directories and force-downloads those files before writing `setup_complete.marker`. The next inference failure came from the reconstruction model loading `facebook/dino-vitb16` in offline mode without a verified local PyTorch snapshot. The setup now also force-caches `facebook/dino-vitb16` and import-tests `ViTModel.from_pretrained("facebook/dino-vitb16", add_pooling_layer=False)` before writing the marker. The run script clears its own `input_images` and `generated` staging directories before copying current crops, so stale cases cannot contaminate inference. The clean A800 run produced four OBJ meshes and the full-fidelity V7 batch rejected all four at visible-surface alignment.

### Hunyuan3D 2.1 Candidate Source

Hunyuan3D 2.1 is queued as a stronger successor to the existing Hunyuan3D-2mini candidate source. The official repository describes Hunyuan3D-Shape-v2-1 as an image-to-shape model and reports about 10 GB VRAM for shape generation. V7 uses the shape-only path and feeds the same representative RGBA object crops as every other generated prior source.

Remote A800 job package:

- runner: `scripts/remote_run_hunyuan21_shape_v7.py`
- job writer: `scripts/write_v7_hunyuan21_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_hunyuan21_prior_outputs`
- setup tmux session: `ego_v7_hunyuan21_setup`
- inference waiter tmux session: `ego_v7_hunyuan21_wait`

Current evidence: Hunyuan3D 2.1 produced four PLY/GLB mesh candidates for mop, trash, and wild-rice. The full-fidelity V7 candidate batch rejected all four by replay before physics or deliverable rendering. These outputs remain rejected complete-object hypotheses, not object-pose annotations.

### Mesh4D Candidate Source

Mesh4D is the next V7 mechanism after the single-image prior batch. Its public inference contract accepts an in-the-wild segmented RGBA image sequence with a white background and predicts a complete animated mesh over a six-frame window. This matches the failure mode seen in the generated-prior matrix: independent single-image priors miss thin long geometry, visible surface coverage, or metric depth even when the measured target replay passes.

`scripts/export_mesh4d_rgba_sequence_v7.py` packages model-produced object masks into Mesh4D's `DATA/<group>/<sequence>/<frame>.png` layout without category-specific visual logic. The exporter writes RGBA frames and a review sheet, and preserves source frame indices for replay mapping.

Current exported inputs:

- wild-rice active stem: `/data2/ego_annotation_outputs/representative_wild_rice/v7_mesh4d_rgba_input_2538_2548`
- trash lid: `/data2/ego_annotation_outputs/representative_trash/v7_mesh4d_rgba_input_865_870`
- mop: `/data2/ego_annotation_outputs/representative_mop/v7_mesh4d_rgba_input_759_765`

Visual review accepts these three input sequences as model-produced evidence for Mesh4D. Mesh4D outputs will still enter the same guarded replay, physics QC, and render inspection path as every generated prior.

Remote A800 job package:

- runner: `scripts/remote_run_mesh4d_sequence_v7.py`
- job writer: `scripts/write_v7_mesh4d_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_mesh4d_outputs`
- setup tmux session: `ego_v7_mesh4d_setup`
- inference waiter tmux session: `ego_v7_mesh4d_wait`

The runner follows Mesh4D's public inference contract: it loads one six-frame RGBA sequence from `DATA/<group>/<sequence>/`, generates an initial Hunyuan3D-2.1 mesh from the first RGBA frame, runs Mesh4D's video deformation model, and writes six generated OBJ meshes with source-frame mapping in `qc_mesh4d_sequence_v7.json`. The Mesh4D result is an animated complete-mesh hypothesis. V7 still must align each generated frame to the measured target, replay image-depth evidence, recompute physics/contact checks, and inspect stakeholder renders before any Mesh4D output can be used as object-pose annotation.

`scripts/archive_mesh4d_sequence_prior_v7.py` converts a Mesh4D six-mesh report into the same per-frame mesh archive schema used by the existing replay tools. It aligns each generated frame to the corresponding measured visible surface and writes both the archive and alignment rows. `scripts/run_v7_generated_prior_replay_qc.py` accepts this prealigned archive through `--prealigned-mesh-archive` plus `--prealigned-report`, so Mesh4D follows the same observed-target replay, z-buffer replay, thresholding, physics QC, and render path as static generated priors.

`scripts/run_v7_mesh4d_sequence_batch.py` discovers completed Mesh4D reports under a local sync root, maps each report to the same representative target contracts through `configs/v7_mesh4d_case_targets.json`, archives the six generated meshes, and runs guarded replay. With `--run-physics --render-deliverables`, the same batch continues into the V7 physics wrapper and delivery renderer only after an accepted full-fidelity replay report.

### SPAR3D Candidate Source

SPAR3D is an official Stability AI single-image object mesh source that writes GLB meshes and point clouds from image inputs. It is queued as another complete-mesh prior, using the same model-produced RGBA object crops and the same downstream replay contract as TripoSG and InstantMesh.

Remote A800 job package:

- runner: `scripts/remote_run_spar3d_shape_v7.py`
- job writer: `scripts/write_v7_spar3d_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_spar3d_prior_outputs`
- setup tmux session: `ego_v7_spar3d_setup`
- inference waiter tmux session: `ego_v7_spar3d_wait`

SPAR3D's official model `stabilityai/stable-point-aware-3d` is gated on Hugging Face. The remote setup now imports SPAR3D successfully after making the background-remover import lazy for pre-masked RGBA inputs. The checkpoint access check still returns `GatedRepoError` 403 for `config.yaml`, so SPAR3D is queued but access-blocked until the token/account is authorized. The V7 job keeps that as an explicit outcome: if access is missing, first inference fails visibly instead of substituting a weaker mesh source.

### Pixal3D Candidate Source

Pixal3D is a public TencentARC single-image mesh source. The current public repository runs image-to-GLB inference through `inference.py`, accepts RGBA alpha masks directly during preprocessing, supports low-VRAM mode, and exposes attention backend selection through environment variables. V7 feeds Pixal3D the same model-produced RGBA object crops as the other complete-mesh sources and exports the resulting GLB as a triangle PLY for guarded replay.

Remote A800 job package:

- runner: `scripts/remote_run_pixal3d_shape_v7.py`
- job writer: `scripts/write_v7_pixal3d_remote_job.sh`
- remote output root: `/mnt/user-home/yiwen/ego_annotation_remote/v7_pixal3d_prior_outputs`
- setup tmux session: `ego_v7_pixal3d_setup`
- inference waiter tmux session: `ego_v7_pixal3d_wait`

The Pixal3D model repository is public and about 22.4 GiB. The A800 host has Python 3.10 and a CUDA-capable driver compatible with the repository's public Hugging Face demo wheels. The setup path uses the official demo wheel stack plus `ATTN_BACKEND=flash_attn_3` and `SPARSE_ATTN_BACKEND=flash_attn_3`, writes a setup marker only after Pixal3D, `flash_attn_3`, `o_voxel`, torch, torchvision, and trimesh import, then queues inference behind the same per-GPU lock used by the other V7 model sources. Pixal3D output remains a complete-object mesh hypothesis until guarded replay, physics QC, and render inspection accept it.

Current setup repair: Pixal3D's public pipeline constructs a background-removal model during `from_pretrained`, even though its preprocessing uses non-opaque RGBA alpha directly and V7 supplies pre-masked RGBA crops. The public `briaai/RMBG-2.0` dependency is gated and returned 401 on A800. V7 patches the cloned Pixal3D source under `PIXAL3D_REQUIRE_PREMASKED_RGBA=1` so `rembg_model` is disabled and non-alpha inputs fail loudly instead of invoking a separate segmentation path. The first inference run selected SDPA in Pixal3D's own dense and sparse config logs but still failed inside the official `natten 0.21.0+torch2.6cu124` wheel with `no kernel image is available for execution on the device`. The A800 venv also imports `flash_attn_3`, and Pixal3D's config accepts that backend. V7 selected `flash_attn_3` for both dense and sparse attention and verified the selected sparse backend before writing the setup marker. The repaired inference still failed after sparse-structure sampling with the same NATTEN native-kernel error. This means the exposed Pixal3D attention backend is not the owner of the failing NATTEN call. Further Pixal3D work must isolate the dependency-level native callsite before spending more GPU time.

### CRM Candidate Source

CRM is a public single-image mesh source from `thu-ml/CRM`. The official README states that it generates a textured 3D mesh from one image, and the command-line path writes an `output3d.zip` containing OBJ, MTL, and texture files. The official Hugging Face model `Zhengyi/CRM` is public and not gated; metadata reports about 12.8 GB of weights across `CRM.pth`, `pixel-diffusion.pth`, and `ccm-diffusion.pth`.

CRM is not queued before Pixal3D, InstantMesh, and PartCrafter finish or fail. Its official environment contract is Python 3.9 with `torch==1.13.0+cu117`, `kaolin==0.14.0`, `nvdiffrast`, and `xformers`; creating that environment would add another large venv and checkpoint set while `/mnt/user-home` is already 99 percent used. CRM remains the next public source to package if the already setup-complete waiters fail or produce rejected priors.

### Representative Trash Prior Replay

V7 also tests the generated-prior replay contract on the non-kitchen trash-lid representative. The measured input is the existing SAMWISE/UniDepth/VGGT-K solidified sheet archive for frames 865 to 870:

- measured archive: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_perframe_thick001_865_870/solidified_sheet_object_meshes_world.npz`
- manifest: `/data2/ego_annotation_outputs/representative_trash/v3_samwise_pink_lid_bundlesdf_dataset_858_880/manifest.json`
- hand/camera annotations: `/data2/ego_annotation_outputs/representative_trash/v3_mano_side_metric_refit_858_880/annotations_side_metric_refit.json`
- metric depth: `/data2/ego_annotation_outputs/representative_trash/v3_unidepth_metric_source_858_880/unidepth_metric_depth_v3.npz`

This measured sheet archive is visible-surface evidence only. It is not treated as closed object pose.

TRELLIS frame-880 prior replay:

- report: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_trellis_frame880_865_870/qc_v7_generated_prior_replay.json`
- visual check: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_trellis_frame880_865_870/visual_check/contact_sheet.png`
- visible-surface coverage p95: 0.0248 m;
- median silhouette IoU: 0.404;
- median visible-inside fraction: 0.918;
- median z-buffer p95 depth error: 0.0531 m.

Hunyuan3D-mv prior replay:

- report: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_hunyuan_mv_depthgrown_865_870/qc_v7_generated_prior_replay.json`
- visual check: `/data2/ego_annotation_outputs/representative_trash/v7_generated_prior_replay_hunyuan_mv_depthgrown_865_870/visual_check/contact_sheet.png`
- visible-surface coverage p95: 0.0549 m;
- median silhouette IoU: 0.315;
- median visible-inside fraction: 0.885;
- median z-buffer p95 depth error: 0.0940 m.

Visual inspection shows both generated priors become smooth cap or bowl-like surfaces over the lid and miss the rim/interior depth structure. V7 therefore rejects both as object-pose annotations.

### Representative Mop Prior Replay

V7 tests a third representative class: a long-handled mop. This stresses long, thin tool geometry and large perspective changes.

Measured evidence:

- measured archive: `/data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_759_765/observed_mask_depth_meshes_world.npz`
- stride-1 measured archive: `/data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_stride1_759_765/observed_mask_depth_meshes_world.npz`
- manifest: `/data2/ego_annotation_outputs/representative_mop/v3_mop_depth_manifold_dataset_735_765/manifest.json`
- hand/camera annotations: `/data2/ego_annotation_outputs/representative_mop/v3_vggt_object_skeleton_735_765/annotations_v3_vggt_object_skeleton.json`
- metric depth: `/data2/ego_annotation_outputs/representative_mop/v3_unidepth_dense_735_765/unidepth_full_frame_depth_v3.npz`
- measured replay baseline: `/data2/ego_annotation_outputs/representative_mop/v7_mop_observed_surface_contract_unidepth_vggt_stride1_zbuffer_759_765/qc_mesh_zbuffer_projection_v3.json`

The earlier mop baseline mixed a per-row depth PNG and fixed intrinsics at export time with full-frame UniDepth and annotation-VGGT intrinsics at replay time, producing a false failure. Re-exporting the same model-produced masks with the same UniDepth and VGGT contract used by replay fixed the metric-depth mismatch, but stride-5 pixel sampling still underfilled the thin mop silhouette: median silhouette IoU was 0.808 while every rendered pixel lay inside the mask and median z-buffer p95 depth error was 0.0016 m. Exporting the same mask-depth evidence at stride 1 removes that sampling artifact: median silhouette IoU is 0.9995, median visible-inside fraction is 1.000, and median z-buffer p95 depth error is 0.0010 m. This remains measured visible-surface evidence, not closed object-pose delivery.

The guarded wrapper now separates three mechanisms. Running the stale archive through the corrected replay contract returns `invalid_observed_target` with median visible-inside fraction 0.352 and median z-buffer p95 depth error 0.048 m. Running the stride-5 UniDepth/VGGT archive returns `invalid_observed_target` because it is a downsampled surface that underfills the silhouette. Running the stride-1 archive passes observed-target replay and makes generated-prior rejection interpretable.

TRELLIS frame-750 prior replay:

- report: `/data2/ego_annotation_outputs/representative_mop/v7_generated_prior_replay_trellis_frame750_contract_unidepth_vggt_759_765/qc_v7_generated_prior_replay.json`
- z-buffer video: `/data2/ego_annotation_outputs/representative_mop/v7_generated_prior_replay_trellis_frame750_contract_unidepth_vggt_759_765/zbuffer_qc/mesh_zbuffer_projection_qc.mp4`
- visible-surface coverage p95: 0.137 m;
- hidden-surface conflict p95: 0.744 m;
- median silhouette IoU: 0.126;
- median visible-inside fraction: 0.142;
- median z-buffer p95 depth error: 0.0730 m.

Visual inspection shows the generated prior becomes a long diagonal plank crossing the room instead of the mop head and handle. V7 rejects it as object-pose annotation.

### Measured Shell Evidence

The trash representative has a watertight measured shell archive:

- archive: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/solidified_sheet_object_meshes_world.npz`
- z-buffer report: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_thick001_allfaces_zbuffer_qc_865_870/qc_mesh_zbuffer_projection_v3.json`
- contact SDF report: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/volume_sdf_contact_recomputed_pitch001_qc.json`
- full-hand SDF report: `/data2/ego_annotation_outputs/representative_trash/v3_observed_unidepth_vggtK_solidified_stride2_perframe_thick001_865_870/full_hand_sdf_penetration_recomputed_pitch001_qc.json`

It is watertight, one connected component per frame, and replays well: median silhouette IoU is 0.983, median visible-inside fraction is 1.000, and median z-buffer p95 depth error is 0.0011 m. The selected-contact SDF report has 0 percent penetration and near-surface contact. This archive is still a measured thin shell, with median thickness 0.001 m. It is valid measured object geometry for a flat lid surface, but it is not evidence that V7 solved generic complete object mesh reconstruction.

### Visibility-Aware Surfel Graph

`scripts/fit_visibility_surfel_graph_v7.py` builds temporal surfel nodes from mesh-attached learned point tracks. It consumes the V6 merged pair-factor report and the accepted V6 mesh archive, rejects duplicate observation conflicts explicitly, and solves:

```text
min_X
  ||X - X_measured|| / sigma_obs
+ ||R_t X_i + p_t - X_j|| / sigma_track
+ ||X_{t+1} - 2 X_t + X_{t-1}|| / sigma_smooth
```

The solve changes only the surfel state package and leaves the delivered mesh archive fixed.

Command:

```bash
.venv/bin/python scripts/fit_visibility_surfel_graph_v7.py \
  --mesh-archive /data2/ego_annotation_outputs/representative_wild_rice/v6_completed_plus_verified_repair2539_2520_2550/solidified_sheet_object_meshes_world.npz \
  --pair-factors-json /data2/ego_annotation_outputs/representative_wild_rice/v6_cotracker_multianchor5_merged_pair_factors_2532_2550/qc_merged_pair_factors_v6.json \
  --frame-start 2534 \
  --frame-end 2550 \
  --output-dir /data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_multianchor5_2534_2550
```

Result:

- surfel nodes: 695;
- accepted CoTracker edges: 509;
- smooth triples: 325;
- rejected pairs: 0;
- pair residual p95 before solve: 8.36 mm;
- pair residual p95 after solve: 6.93 mm;
- correction displacement p95: 0.91 mm;
- solver evaluations: 5;
- status: `annotation_ready: false`.

Artifacts:

- repaired-archive sparse edges for frame-2539 anchor: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_anchor2539_sparse_edges_repaired_archive_2532_2550/cotracker_sparse_correspondence_edges_v7.json`
- repaired-archive pair factors: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_anchor2539_pairwise_repaired_archive_2532_2550/qc_cotracker_pairwise_rigid_factors_v7.json`
- merged pair factors: `/data2/ego_annotation_outputs/representative_wild_rice/v7_cotracker_multianchor5_repaired_archive_merged_pair_factors_2532_2550/qc_merged_pair_factors_v7.json`
- report: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_repaired_archive_multianchor5_2534_2550/qc_visibility_surfel_graph_v7.json`
- solved surfel positions: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_graph_repaired_archive_multianchor5_2534_2550/visibility_surfel_positions_v7.npz`
- visual review video: `/data2/ego_annotation_outputs/representative_wild_rice/v7_visibility_surfel_review_repaired_archive_multianchor5_2534_2550/visibility_surfel_review_v7.mp4`

The first surfel run exposed a provenance bug: frame 2539 disappeared because gap sparse-edge vertex indices came from the pre-repair factor-graph mesh archive, while V7 consumed the repaired V6 mesh archive. V7 now fails fast when sparse-edge provenance points to a different mesh archive. Reattaching the frame-2539 CoTracker world points to the repaired archive restores continuous surfel support through 2539.

This closes the first V7 temporal-state evidence: the repaired-archive CoTracker factors support a visibility-aware surfel graph over observed patches with sub-millimeter correction magnitude. The graph represents observed temporal patches and serves as a state prior for later mesh completion.

## Current V7 Full-Chain Status

The current V7 delivery batch tests video-derived object mesh archives through the same guarded chain used for generated priors:

```bash
.venv/bin/python scripts/run_v7_prior_candidate_batch.py \
  --candidate-kind video_mesh \
  --candidate-file configs/v7_video_mesh_candidates.tsv \
  --observed-cache-file configs/v7_observed_zbuffer_cache_full_fidelity.tsv \
  --output-root /data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238 \
  --run-physics \
  --render-deliverables
```

Artifacts:

- batch report: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/qc_v7_prior_candidate_batch.json`
- full-chain summary: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/qc_v7_prior_candidate_batch_summary.md`
- replay matrix: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/qc_v7_prior_candidate_batch_matrix.md`
- visual replay sheet: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/qc_v7_prior_candidate_batch_visual_sheet.png`

Results:

| Target | Full delivery | Replay IoU | Replay depth p95 | Track p95 | Contact rows | Contact SDF p95 | Full-hand penetration | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| wild-rice | yes | 0.934 | 4.47 mm | 8.29 mm | 0 | n/a | 0.67 percent | Delivered without a contact claim; full-window hand/object nonpenetration passes. |
| trash | yes | 0.949 | 0.53 mm | 7.80 mm | 6 | 1.25 mm | 1.18 percent | Delivered with geometry-backed temporal contact. |
| mop | no | 0.999 | 1.46 mm | 9.20 mm | n/a | n/a | n/a | Rejected at physics because the selected annotations contain zero MANO hand rows in frames 759 to 765. |

Delivered videos:

- wild-rice overlay: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/wild_rice/video_mesh_v6_repaired_measured_2538_2540/deliverables/overlay/mesh_surface_contact_review.mp4`
- wild-rice side-by-side: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/wild_rice/video_mesh_v6_repaired_measured_2538_2540/deliverables/world/world_reconstruction_side_by_side.mp4`
- wild-rice 3D: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/wild_rice/video_mesh_v6_repaired_measured_2538_2540/deliverables/world/world_reconstruction_3d.mp4`
- trash overlay: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/trash/video_mesh_v3_solidified_unidepth_vggt_865_870/deliverables/overlay/mesh_surface_contact_review.mp4`
- trash side-by-side: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/trash/video_mesh_v3_solidified_unidepth_vggt_865_870/deliverables/world/world_reconstruction_side_by_side.mp4`
- trash 3D: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/trash/video_mesh_v3_solidified_unidepth_vggt_865_870/deliverables/world/world_reconstruction_3d.mp4`

The render pass was rerun after the presentation patches through commit `bf0342f`. The world panel now fits the current head-camera frustum and full head trajectory in the same metric 3D scene as the object mesh and MANO hands. Labels moved to a compact legend so they do not cover the object or hand. The side-by-side caption prefix now says `V7 mesh-backed reconstruction`, which matches measured video-derived archives instead of implying a generated prior. The head-frustum visual scale is reduced so the camera-pose cue stays visible without dominating the hand-object geometry.

Visual inspection artifacts:

- wild-rice side-by-side still: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/visual_inspection_stills_after_render/wild_rice/side_mid.jpg`
- wild-rice 3D still: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/visual_inspection_stills_after_render/wild_rice/world_mid.jpg`
- trash side-by-side still: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/visual_inspection_stills_after_render/trash/side_mid.jpg`
- trash 3D still: `/data2/ego_annotation_outputs/v7_video_mesh_candidate_batch_20260608_0238/visual_inspection_stills_after_render/trash/world_mid.jpg`

The rendered head frustum is a metric camera-pose cue, not a physical object. It can overlap the object in image space when the camera lies above the manipulated surface in the selected 3D view. The latest renderer uses a smaller world-frustum visual scale so the hand-object geometry remains the primary visual signal while the head pose stays visible.

Interpretation:

V7 has a full local delivery chain for video-derived measured mesh archives on two representative clips. Single-image complete generated-prior acceptance remains zero across the current representative set. The original mop window 759 to 765 remains rejected: replay and CoTracker track QC passed, but the frame window lacked a physically compatible measured MANO stream. V7 therefore moved the mop representative to frames 702 to 708 from source clip `/data2/egoscale_demo_30h/egoscale_tasks/20251210_0002_Rec4afc_P0_S296a7f_task_4/20251210_0002_Rec4afc_P0_S296a7f_task_4.mp4`.

The measured remote job can use RTMLib in two distinct roles. `USE_RTMLIB_HAND_EVIDENCE=1` runs RTMLib and passes selected 2D hand keypoints into the MANO articulation refit. `USE_RTMLIB_SAM2_HAND_PROMPTS=1` additionally replaces the original VLM hand prompts with RTMLib-derived SAM2 prompts. Keep the second switch disabled when RTMLib detects the correct hand but its skeleton collapses fingers or produces a box that makes SAM2 reject the hand mask; in that case RTMLib is valid anatomy evidence for refit but invalid segmentation evidence.

The measured remote job can also test HandDGP as a third hand-source hypothesis with `USE_HANDDGP_HAND_EVIDENCE=1`. HandDGP output is diagnostic until `scripts/convert_handdgp_to_mano_candidates_v7.py` fits MANO rotation matrices and betas to the HandDGP local hand geometry, solves source-camera translation from the same 2D measurement, and emits the normal V7 MANO annotation contract. The existing mask-depth-articulation refit and hand selector then accept or reject the converted candidates under the same silhouette, depth, RTMLib, pose-delta, and scale checks used for HaMeR and WiLoR.
