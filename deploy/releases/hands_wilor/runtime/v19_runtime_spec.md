# V19 Runtime Spec

This is the only runtime instruction document. It defines state ontology, execution policy, exact phase order, scripts, command templates, required outputs, and stop conditions. Do not inspect or mention any file that is not named by this spec.

## Runtime inputs

The launch provides:

- `{INPUT_VIDEO}`: egocentric input video;
- `{RUN_ROOT}`: fresh output run root;
- `{CASE_ID}`: case id;
- this runtime workspace;
- prediction-side sensor metadata, if present next to the input.

## Runtime outputs

The runtime output is a prediction run root containing `input/`, `measurements/`, `state/`, `renders/`, and `logs/`. The renderer consumes `state/`. Logs and measurements are provenance, not final annotations. Treat `{RUN_ROOT}/logs/harness_events.jsonl` as append-only JSONL: append one complete JSON object per event and never overwrite prior events.

## State ontology

- `camera`: intrinsics, camera/head pose, depth/scale provenance, frame/time semantics, uncertainty.
- `hands`: metric 3D MANO state over time, side, camera/world transforms, visibility, provenance, uncertainty.
- `objects`: object instances, masks/tracks, physical branch, reconstructed or adapted geometry, pose/posterior, provenance, uncertainty.
- `visibility_occlusion`: visible, partially visible, occluded, out-of-frame, or unresolved state for hands and objects, with occluder ownership when inferable.
- `contact`: contact, near-contact, non-contact, or unresolved state with patch/distance evidence and uncertainty.
- `nonpenetration`: hand/object geometry residuals and uncertainty; absence of a valid signed volume is unresolved, not success.
- `renders`: visible overlay/world/side-by-side annotations caused by state variables.

## Evidence rules

- A detector box, keypoint track, mask, depth map, point cloud, centroid, label, or JSON row is a measurement, not physical state by itself.
- Object pose requires object geometry adapted or fitted to observed instance evidence and a pose trajectory/posterior.
- Hand state requires metric MANO surface or reproducible MANO parameters with camera/world semantics.
- Contact and occlusion require geometric, depth-order, temporal, or explicitly uncertain evidence. Do not make them certain from a semantic label alone.
- Weak measurements continue downstream with uncertainty. Broken contracts, wrong frame alignment, wrong coordinate frame, wrong object mask, side swap, missing geometry, or invalid units must be fixed or represented as unresolved.

## Execution policy

1. Execute phases in order.
2. Do not discover or substitute scripts. Each script phase names the script to run.
3. For an agent-write phase, write only the specified JSON/Markdown artifact and preserve uncertainty.
4. Bind placeholders from launch arguments, phase outputs, or this spec. If a placeholder cannot be bound without searching outside the bundle, record the unresolved placeholder as a blocker.
5. The runtime itself runs on the A800 compute host. Heavy model phases and light metadata/state phases both read and write the same A800/truenas run root; do not copy phase outputs to any other machine during prediction.
6. Infrastructure is out of scope for runtime. Launch preflight is complete before start. Execute prediction phases only; if a named phase command fails, record that phase blocker and stop.
7. Do not run scoring or comparisons inside this runtime run.
8. Do not use sleep, polling loops, or idle waits. Long-running jobs need durable command logs/status files and inspectable job handles.

## Declared compute and asset targets

- Runtime host: A800 compute host `yiwen@192.168.11.220`; Pi is launched inside a tmux session on this host.
- Runtime workspace: `/mnt/user-home/yiwen/ego_annotation_runtime/v19_bundle_a800`.
- Run roots and runtime inputs are A800-local/truenas paths under `/mnt/truenas-user-home/yiwen/ego_annotation_outputs`.
- HaWoR work root: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work`
- HaWoR Python: `/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/.venv_hawor/bin/python`
- SAM2 checkpoint: `/mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt`
- OWLv2 Python: `/mnt/user-home/yiwen/ego_annotation_remote/hunyuan3d_v3_env/bin/python`; this interpreter must import `transformers`, `torch`, `PIL`, and `cv2` before P06.
- OWLv2 model cache: `/home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble`; this is a parent-preflighted local cache, not a runtime download.
- UniDepth checkout: `/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth`
- Model Python for UniDepth/SAM2 on the A800 host: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`; this is a launch-preflighted contract.

## Stop condition

If a phase cannot run because an input, script, model asset, or environment is missing, write:

`{RUN_ROOT}/state/runtime_blockers/<PHASE_ID>.json`

with phase id, missing component, blocked state variable, evidence, and next required repair. Stop that branch rather than inventing substitute outputs.

## Placeholders

- `{FRAME_END}`: last frame index from P01 manifest.
- `{SOURCE_WIDTH}`, `{SOURCE_HEIGHT}`: source video resolution from P01 manifest.
- `{GPU_ID}`: selected A800 GPU from P02.
- `{REMOTE_MODEL_PYTHON}`: `/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2/bin/python`, a launch-preflighted A800 model interpreter used for UniDepth/SAM2 Python phases.
- `{OWLV2_PYTHON}`: `/mnt/user-home/yiwen/ego_annotation_remote/hunyuan3d_v3_env/bin/python`, a launch-preflighted A800 interpreter used only for OWLv2 detector-box prompting.
- `{OBJECT_ID}`: object id chosen in P05.
- `{TRACK_ID}`: SAM2 track id for `{OBJECT_ID}`.
- `{ANCHOR_FRAME}`: selected clean object evidence frame.
- `{INTERVAL_START}`, `{INTERVAL_END}`: selected physical interval for MANO/object correction.
- `<calibration_contract>`: chosen calibration contract JSON filename under `{RUN_ROOT}/state/calibration/`.
- `<completed_mesh_ply>`: completed mesh path from P13.
- `<visible_contact_ownership_factor_report>`: factor report from P17.
- `<render_branch_overlay_mp4>`, `<render_branch_world_mp4>`, `<render_branch_side_by_side_mp4>`: P19 render outputs.

# Phase graph

## P00 startup records

Type: agent writes JSON/Markdown.

Outputs:

- `{RUN_ROOT}/input/runtime_input_contract.json`
- `{RUN_ROOT}/logs/harness_events.jsonl`
- `{RUN_ROOT}/state/v19_physical_state.json`
- `{RUN_ROOT}/state/v19_uncertainty_state.json`
- `{RUN_ROOT}/state/v19_agent_evidence.md`

State after phase: unresolved camera, hands, objects, contact, occlusion, and nonpenetration.

## P01 raw frame manifest

Script: `scripts/build_v19_raw_frame_manifest.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_raw_frame_manifest.py \
  --video "{INPUT_VIDEO}" \
  --output-dir "{RUN_ROOT}/input/raw_frame_manifest" \
  --render-width 960
```

Required output: `{RUN_ROOT}/input/raw_frame_manifest/manifest.json`.

## P02 A800 host probe and GPU selection

Type: bash command.

```bash
set -euo pipefail
hostname
df -h "{RUN_ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Required output: append a log event to `{RUN_ROOT}/logs/harness_events.jsonl` with selected `{GPU_ID}` and successful A800 host probe. There is no per-phase bundle sync or raw-frame-manifest staging because Pi, the input video, and the run root are already on the A800 host.

## P03 depth and intrinsics measurement

Script: `scripts/run_unidepth_full_frame_v3.py`

```bash
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{REMOTE_MODEL_PYTHON}' scripts/run_unidepth_full_frame_v3.py \
  --manifest '{RUN_ROOT}/input/raw_frame_manifest/manifest.json' \
  --output-dir '{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame' \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --unidepth-repo /mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work/UniDepth \
  --remote-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --local-root '{RUN_ROOT}/input/raw_frame_manifest' \
  --source-width {SOURCE_WIDTH} \
  --source-height {SOURCE_HEIGHT}
```

Required output: `{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz` and `qc_unidepth_full_frame_v3.json` written directly under the A800/truenas run root.

## P03b calibration contract

If prediction-side calibration metadata is present next to the input, copy it to `{RUN_ROOT}/state/calibration/` and record the source. Otherwise run:

Script: `scripts/build_v19_calibration_contract.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_calibration_contract.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --unidepth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/state/calibration" \
  --aggregation median \
  --square-focal
```

Required output: one calibration contract JSON under `{RUN_ROOT}/state/calibration/`.

The canonical runtime-generated filename is `{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json`. If a copied prediction-side contract uses another filename, record that path and use that same copied contract for P04/P08/P09.

## P04 MANO hand measurement

Script: `scripts/remote_run_hawor_export.sh` (calls `scripts/export_hawor_world.py`)

Extract the HaWoR focal only from the chosen calibration value, never from diagnostics, outlier tables, review statistics, or `largest_selected_focal_deviations`. Use this exact extraction priority: top-level `focal_px`, top-level `focal_geom_px`, top-level `intrinsics_fx_fy_cx_cy[0]`, then `intrinsics.fx`. If none exists, stop with a P04 blocker before running HaWoR.

```bash
CONTRACT='{RUN_ROOT}/state/calibration/v19_camera_calibration_contract.json'
FOCAL=$("{REMOTE_MODEL_PYTHON}" - "$CONTRACT" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
value = None
for key in ("focal_px", "focal_geom_px"):
    if isinstance(data.get(key), (int, float)):
        value = float(data[key])
        break
if value is None:
    intr = data.get("intrinsics_fx_fy_cx_cy")
    if isinstance(intr, list) and intr and isinstance(intr[0], (int, float)):
        value = float(intr[0])
if value is None:
    intr = data.get("intrinsics")
    if isinstance(intr, dict) and isinstance(intr.get("fx"), (int, float)):
        value = float(intr["fx"])
if value is None:
    raise SystemExit(f"missing canonical focal in {path}; do not scan diagnostics")
print(value)
PY
)
EGO_HAWOR_ROOT=/mnt/user-home/yiwen/ego_annotation_remote/hawor_work \
EGO_HAWOR_CASE='{CASE_ID}' \
EGO_HAWOR_CLIP='{INPUT_VIDEO}' \
EGO_HAWOR_OUTPUT_DIR='{RUN_ROOT}/measurements/hand_candidates/hawor_world' \
EGO_HAWOR_IMG_FOCAL="$FOCAL" \
EGO_HAWOR_FORCE_FOCAL_CACHE_REFRESH=1 \
bash scripts/remote_run_hawor_export.sh
```

Required output: `{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz` and `qc_hawor_world_hands.json` written directly under the A800/truenas run root.

## P05 object plan

Type: agent writes JSON from visual evidence.

Output: `{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json`.

Minimum fields per object: `object_id`, `description`, `physical_branch_hypotheses`, `evidence_frames`, `expected_visible_intervals`, `uncertainty_notes`, `detector_text_prompts`, `detector_prompt_frames`, and `detector_active_interval`. `detector_text_prompts` are open-vocabulary object queries such as `keyboard.` / `computer keyboard.`; they are not pixel coordinates. `detector_prompt_frames` must be representative visible frames selected from the raw video, and `detector_active_interval` must be a frame span over which the object is expected to be physically present or uncertain.

## P06 object grounded detector box prompts

Script: `scripts/build_v19_owlv2_object_box_prompts.py`

The default P06 source is OWLv2 text-conditioned detection, not VLM/agent pixel clicks. A VLM/agent may name the target object, write open-vocabulary text prompts, and choose representative frames in P05, but it must not be treated as the source of pixel-accurate click coordinates. For each rigid/manipulated object selected in P05, run OWLv2 with the object's `detector_text_prompts` on `detector_prompt_frames`, then write SAM2 prompt JSON containing `box_xyxy` in the detector image coordinate frame. Do not hard-code a category-specific object id, prompt string, or active interval in the runtime command; keyboard prompts are an example P05 object-plan value, not the P06 contract.

```bash
mkdir -p "{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2" "{RUN_ROOT}/renders/review_frames/P06_owlv2_object_boxes"
TEXT_PROMPT_ARGS=()
for PROMPT in "${DETECTOR_TEXT_PROMPTS[@]}"; do
  TEXT_PROMPT_ARGS+=(--text-prompt "$PROMPT")
done
CUDA_VISIBLE_DEVICES="{GPU_ID}" "{OWLV2_PYTHON}" scripts/build_v19_owlv2_object_box_prompts.py \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --output-root "{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2" \
  --review-dir "{RUN_ROOT}/renders/review_frames/P06_owlv2_object_boxes" \
  --case-id "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --track-id "{TRACK_ID}" \
  --description "{OBJECT_DESCRIPTION}" \
  "${TEXT_PROMPT_ARGS[@]}" \
  --prompt-frames "{DETECTOR_PROMPT_FRAMES_CSV}" \
  --active-start "{DETECTOR_ACTIVE_START}" \
  --active-end "{DETECTOR_ACTIVE_END}" \
  --owlv2-model /home/yiwen/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble \
  --box-threshold 0.03 \
  --device cuda \
  --box-only
```

Required output for each object: `{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2/{OBJECT_ID}/object_point_prompts_vlm.json` and `v19_owlv2_object_box_prompt_report.json`. The prompt JSON must contain `prompt_source=owlv2_text_grounded_detector_boxes`, `box_xyxy` on visible prompt frames, and a coordinate declaration matching the frame images used by the detector. If OWLv2 produces no usable box for the target object, write a P06 blocker and stop; do not replace it with VLM/agent click coordinates as the default path. For the current fixed HOT3D keyboard clips, a valid P05 object plan may set `{OBJECT_ID}=keyboard`, `{TRACK_ID}=keyboard`, `{DETECTOR_TEXT_PROMPTS}=["keyboard.", "computer keyboard."]`, and prompt frames such as `30,32,45,60,75,90,105,120,135,149` only after the runtime visually confirms those frames contain the keyboard.

## P07 object masks/tracks

Script: `scripts/run_sam2_vlm_points_multiobject.py`

```bash
mkdir -p '{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points'
CUDA_VISIBLE_DEVICES='{GPU_ID}' '{REMOTE_MODEL_PYTHON}' scripts/run_sam2_vlm_points_multiobject.py \
  --clip '{INPUT_VIDEO}' \
  --point-root '{RUN_ROOT}/measurements/object_candidates/object_box_prompts_owlv2' \
  --output-root '{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points' \
  --checkpoint /mnt/user-home/yiwen/ego_annotation_remote/data/sam2.1_hiera_small.pt \
  --frame-start 0 \
  --frame-end {FRAME_END} \
  --sam2-image-width 960 \
  --render-width 960 \
  --use-positive-prompt-box \
  --prompt-box-pad-ratio 0.18 \
  --prompt-box-min-pad-px 24
```

Required output for each object: `{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points/{TRACK_ID}/sam2/sam2_track.json` written directly under the A800/truenas run root.

Required P07 self-check before P08: inspect `qc_sam2_multiobject_points.json`, P06 OWLv2 box review frames, prompt contract reports, and the SAM2 overlay/mask review for representative prompted frames, visible gaps inside expected active intervals, and any frames later used for rigid fitting/evidence. A mask that tracks a hand/sleeve/table edge while the object is visible is a hard P07 failure, not noisy-but-usable evidence. A mask that contains the target object but also broad hand/table/arm support is still a wrong object-support mask and must not become a rigid pose or geometry-completion observation. If OWLv2 boxes are loose or wrong, repair the grounded detector query/frame set/threshold or use another text-grounded detector; do not return to VLM/agent pixel-click prompting as the default. After at least one grounded detector rerun, remaining local mask gaps or low-confidence frames are not by themselves a stop condition when the accepted masks preserve the object identity on usable evidence frames; record those gaps as missing/uncertain mask observations and continue so the rigid branch can complete the full timeline in P15. The pipeline must not let an obvious wrong object track become a rigid pose observation, and it also must not prevent a rigid object from reaching P15 merely because local SAM2 evidence is missing in some visible frames.

## P08 base annotations

Script: `scripts/build_v19_base_annotations.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_base_annotations.py \
  --case "{CASE_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --sam2-output-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --calibration-contract "{RUN_ROOT}/state/calibration/<calibration_contract>.json" \
  --output-dir "{RUN_ROOT}/state/base_annotations"
```

Required output: `{RUN_ROOT}/state/base_annotations/annotations_v19_base.json`, `v19_base_physical_state.json`, and `v19_mano_bridge_from_hawor_world.npz`.

## P09 visible metric geometry and anchor proposal

Script: `scripts/build_v19_visible_geometry_from_sam2_depth.py`

P09 is intentionally two-step. First propose anchor candidates from the same SAM2/depth/camera evidence without committing to a canonical object frame:

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case "{CASE_ID}" \
  --track-id "{TRACK_ID}" \
  --object-id "{OBJECT_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --sam2-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/measurements/object_geometry/anchor_candidates/{OBJECT_ID}" \
  --base-annotations "{RUN_ROOT}/state/base_annotations/annotations_v19_base.json" \
  --calibration-contract "{RUN_ROOT}/state/calibration/<calibration_contract>.json" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --preserve-source-index \
  --exclude-hand-bboxes \
  --hand-bbox-exclusion-pad-px 12 \
  --propose-anchor-candidates-only \
  --anchor-candidate-count 12
```

Required proposal outputs: `anchor_candidate_proposals.json` and `anchor_candidate_review.jpg`. The numeric proposal score is not an acceptance gate. The runtime agent must inspect the review sheet as an image, compare raw appearance/mask/depth/hand-removal summaries, and write an explicit anchor decision:

```text
{RUN_ROOT}/state/anchor_decisions/{OBJECT_ID}.json
```

Minimum decision fields: `object_id`, `selected_anchor_frame_idx`, `selected_candidate_rank`, `visual_rationale`, `rejected_candidate_observations`, `uncertainties`, and `candidate_report_path`. Favor the cleanest full-object evidence: large visible support, low hand overlap, non-border mask, coherent object outline/key-grid or texture, reliable depth, and metric extent consistent with neighboring plausible frames. Do not choose a frame solely because it has the largest mask or most sampled points.

Then run canonical visible geometry with the selected anchor. This second command must use `--require-anchor-frame` so the pipeline cannot silently fall back to the max-point frame:

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_visible_geometry_from_sam2_depth.py \
  --case "{CASE_ID}" \
  --track-id "{TRACK_ID}" \
  --object-id "{OBJECT_ID}" \
  --raw-frame-manifest "{RUN_ROOT}/input/raw_frame_manifest/manifest.json" \
  --sam2-root "{RUN_ROOT}/measurements/object_tracks/sam2_owlv2_box_points" \
  --depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --output-dir "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}" \
  --base-annotations "{RUN_ROOT}/state/base_annotations/annotations_v19_base.json" \
  --calibration-contract "{RUN_ROOT}/state/calibration/<calibration_contract>.json" \
  --object-plan "{RUN_ROOT}/measurements/object_candidates/object_plan_agent.json" \
  --anchor-frame "{ANCHOR_FRAME}" \
  --require-anchor-frame \
  --preserve-source-index \
  --exclude-hand-bboxes \
  --hand-bbox-exclusion-pad-px 12
```

Required output: `v19_visible_geometry_depth_fused_report.json` and visible-geometry annotations. P09 must treat hand-owned pixels as occlusion/uncertainty, not visible object surface; if same-frame hand boxes are available, subtract them before depth lifting and record the removed support in `object_surface_ownership_filter`. The per-object `mask_path` written to annotations must be the object-owned mask after this subtraction, because P11/P12 evidence crops and TRELLIS conditioning must not consume hand-owned pixels as object appearance.

## P10 branch decision

Type: agent writes JSON/Markdown.

Output: `{RUN_ROOT}/state/physical_branch_decisions/{OBJECT_ID}.json` and evidence text appended to `state/v19_agent_evidence.md`.

If branch is not rigid, stop rigid path and render uncertainty from available state. If branch is rigid, continue.

## P11 rigid evidence bundle

Script: `scripts/build_v18_compact_rigid_evidence_bundle.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v18_compact_rigid_evidence_bundle.py \
  --case "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --depth-fused-report "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/v19_visible_geometry_depth_fused_report.json" \
  --output-root "{RUN_ROOT}/measurements/geometry_completion/rigid_evidence" \
  --selected-frame-idx "{ANCHOR_FRAME}" \
  --selection-note "agent-selected anchor from state/anchor_decisions/{OBJECT_ID}.json"
```

Required output: evidence bundle report and object crop image path. P11 must use the same `selected_anchor_frame_idx` recorded in P09's anchor-decision JSON; it must not independently rerank evidence frames by mask area or point count. The TRELLIS conditioning image is `selected.trellis_conditioning_crop.crop_rgba` in the evidence-bundle report. Do not use `selected.raw_frame_path`, the full raw frame, or the binary mask as TRELLIS input; those substitute a scene/mask prior for per-instance object mesh reconstruction.

## P12 mesh prior

Scripts: `scripts/resolve_v19_trellis_conditioning_image.py`, `scripts/remote_run_trellis_shape_v3.py`

Resolve the object-isolated conditioning crop from P11 before running TRELLIS:

```bash
EVIDENCE_REPORT="{RUN_ROOT}/measurements/geometry_completion/rigid_evidence/{CASE_ID}/{OBJECT_ID}/evidence_bundle/evidence_bundle_report.json"
EVIDENCE_CROP_RGBA=$("{REMOTE_MODEL_PYTHON}" scripts/resolve_v19_trellis_conditioning_image.py \
  --evidence-report "$EVIDENCE_REPORT")
```

```bash
PYTHONPATH="/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/trellis_work/.venv_trellis/lib/python3.10/site-packages:${PYTHONPATH:-}" \
"{REMOTE_MODEL_PYTHON}" scripts/remote_run_trellis_shape_v3.py \
  --repo /mnt/user-home/yiwen/ego_annotation_remote/trellis_work/TRELLIS \
  --image "$EVIDENCE_CROP_RGBA" \
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42" \
  --seed 42
```

Required output: `{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42/qc_trellis_shape_v3.json` with `status: ok` and `mesh` equal to `{RUN_ROOT}/measurements/geometry_completion/trellis_{OBJECT_ID}_seed42/trellis_mesh.ply`. Do not search for `*report*.json`, do not use `trellis_gaussian.ply` as the mesh input for P13, and do not continue if `EVIDENCE_CROP_RGBA` is missing or resolves to the raw frame.

## P13 mesh adaptation/completion

Script: `scripts/build_v18_compact_rigid_trellis_completion.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v18_compact_rigid_trellis_completion.py \
  --evidence-report "<evidence_bundle_report>" \
  --trellis-report "<trellis_report>" \
  --output-dir "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42" \
  --silhouette-free-space-filter \
  --silhouette-dilate-px 16 \
  --planar-slab-support-filter \
  --planar-slab-eigenvalue-ratio-max 0.04 \
  --planar-slab-min-band-m 0.018 \
  --planar-slab-max-band-m 0.055
```

Required output: completion report and completed mesh. The completed mesh used by all downstream pose/contact/render stages is exactly `outputs.completed_mesh_labeled` in `{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json`. Do not substitute the P12 raw TRELLIS mesh (`trellis_mesh.ply`) for `<completed_mesh_ply>`; P14/P15 poses are in the P13 completed-canonical frame, not the raw TRELLIS model frame. P13 must not promote `unsupported_uncertain` observed Poisson fill into `outputs.completed_mesh_labeled`; unsupported observed fill may remain in labeled diagnostics, but the downstream accepted body must contain only `observed_depth_surface` faces plus accepted hidden-prior faces. P13 must not promote TRELLIS hidden faces that project outside the evidence-frame object-owned silhouette; those faces are free-space-inconsistent hidden prior, not object body. If the observed support surfels are planar, P13 must also reject hidden-prior faces far outside the observed support slab; this is a conditional physical support constraint, not a category-specific keyboard rule.

Resolve the downstream mesh path with the completion report as source of truth:

```bash
COMPLETION_REPORT="{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json"
COMPLETED_MESH_PLY=$("{REMOTE_MODEL_PYTHON}" - "$COMPLETION_REPORT" <<'PY'
import json, sys
from pathlib import Path
report = Path(sys.argv[1])
data = json.loads(report.read_text())
mesh = (data.get("outputs") or {}).get("completed_mesh_labeled")
if not mesh:
    raise SystemExit(f"missing outputs.completed_mesh_labeled in {report}")
path = Path(mesh)
if not path.exists() or path.stat().st_size <= 0:
    raise SystemExit(f"completed mesh from {report} is missing or empty: {path}")
print(path)
PY
)
```

## P14 visible-frame pose fit

Script: `scripts/fit_v18_compact_rigid_object_pose.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/fit_v18_compact_rigid_object_pose.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --object-id "{OBJECT_ID}" \
  --output-dir "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_visible_pose_fit"
```

Required output: object pose fit report.

## P15 temporal rigid pose graph

Script: `scripts/solve_v19_rigid_object_pose_graph.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/solve_v19_rigid_object_pose_graph.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_visible_pose_fit/v18_compact_rigid_object_pose_fit_report.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --object-id "{OBJECT_ID}" \
  --complete-full-timeline-rigid-pose \
  --output-dir "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph"
```

Required output: rigid pose graph report with `full_timeline_rigid_pose_completion.enabled: true`. For a rigid branch, all frames in the raw video must have either a direct corrected pose (`corrected_temporal_rigid_pose_graph`) or an explicit uncertain rigid trajectory completion (`completed_temporal_rigid_pose_uncertain`). A rigid object must not disappear from frames merely because the local mask/depth observation is missing; missing local observations become uncertainty/provenance, not omitted object pose.

## P16 MANO/object constraint measurement

Script: `scripts/build_v18_mano_object_constraint_state.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v18_mano_object_constraint_state.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completion-report "{RUN_ROOT}/measurements/geometry_completion/compact_{OBJECT_ID}_seed42/v18_compact_rigid_trellis_completion_report.json" \
  --output-dir "{RUN_ROOT}/measurements/contact_nonpenetration/{OBJECT_ID}_mano_object_constraint" \
  --object-id "{OBJECT_ID}"
```

Required output: MANO/object constraint state.

## P17 contact/occlusion prior rows

Type: agent writes interval judgment JSON, then script consumes it.

Agent output: `{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}.json`.

The JSON must be a single object with `status: "ok"` (or no status), `case: "{CASE_ID}"` (or no case), and a non-empty `interaction_judgments` list. Each segment in that list must target the object and a single hand side, with fields:

- `judgment_id`: stable segment id.
- `target_entity_id`: `object:{OBJECT_ID}`.
- `hand_side`: `left` or `right`; use separate segments for both hands.
- `frame_start`, `frame_end`: inclusive frame interval.
- `contact_state`: one of `likely_contact`, `possible_contact`, `no_contact`, `unresolved`.
- `occlusion_relation`: one of `hand_in_front_of_object`, `object_in_front_of_hand`, `object_partially_occluded_by_hand`, `no_visible_occlusion`, `unresolved`.
- `depth_reliability`: one of `hand_depth_unreliable`, `object_depth_reliable`, `mixed_or_unresolved`, `not_evaluated`.
- `contact_prior_probability`: numeric in `[0, 1]`.
- `contact_support_uncertainty_m`: non-negative numeric, usually `0.03`–`0.08` for uncertain hand/keyboard contact.
- `contact_weight_multiplier`: non-negative numeric, usually `1.0` unless downweighting uncertain intervals.
- Optional `evidence`, `uncertainty`, `ownership_quarantine` / `object_surface_policy`; use `hand_projected` when hands occlude the object.

Do not write only narrative `contact_intervals` or `occlusion_ownership`; those are not consumed by the factor builder unless converted into `interaction_judgments`.

Script: `scripts/build_v19_visible_contact_ownership_factor.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_visible_contact_ownership_factor.py \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --case "{CASE_ID}" \
  --target-entity-id "object:{OBJECT_ID}" \
  --frame-span "{INTERVAL_START}" "{INTERVAL_END}" \
  --output-root "{RUN_ROOT}/measurements/contact_visibility_factors/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}" \
  --agent-interaction-judgment "{RUN_ROOT}/state/agent_interaction_judgments/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}.json"
```

Required output: visible contact/ownership factor report at `{RUN_ROOT}/measurements/contact_visibility_factors/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}/{CASE_ID}/v19_visible_contact_ownership_factor_report.json`. Because the script nests outputs under `--case`, bind P18 `--factor-report` to this concrete path; do not guess `{output-root}/v19_visible_contact_ownership_factor_report.json`.

## P18 interval MANO correction

Script: `scripts/solve_v18_joint_mano_interval_trajectory.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/solve_v18_joint_mano_interval_trajectory.py \
  --case "{CASE_ID}" \
  --object-id "object:{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completed-mesh "$COMPLETED_MESH_PLY" \
  --completion-report "$COMPLETION_REPORT" \
  --depth-npz "{RUN_ROOT}/measurements/depth_slam/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
  --wilor-root third_party/WiLoR \
  --wilor-mano-left third_party/WiLoR/mano_data/MANO_LEFT.pkl \
  --output-dir "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}" \
  --start-frame "{INTERVAL_START}" \
  --end-frame "{INTERVAL_END}" \
  --sides left right \
  --factor-report "{RUN_ROOT}/measurements/contact_visibility_factors/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}/{CASE_ID}/v19_visible_contact_ownership_factor_report.json" \
  --optimize-contact-state \
  --visible-surface-depth-order-term \
  --gate-translation-with-visible-surface-support \
  --translation-gate-min-visible-surface-depth-vertices 0
```

Required output: raw interval MANO/contact trajectory state. The translation gate preserves source HaWoR wrist/root translation when no selected visible-surface support vertices exist, while keeping optimized wrist-relative articulation; this prevents contact/temporal terms from moving global hand pose without direct support evidence.

## P18b metric-MANO/contact-surface state split

Script: `scripts/build_v19_mano_surface_hypothesis_state.py`

P18 may generate contact-like surface hypotheses from object geometry, depth order, and MANO surfaces. Those hypotheses must not automatically overwrite the metric MANO joint/root state used for evaluation. A surface-normal contact factor is evidence about a local uncertain contact surface, not proof that the whole hand root and 21 joints should move. Therefore the default runtime path preserves source metric MANO joints from P04 and carries P18 surface samples as a separate uncertain contact-surface hypothesis for rendering.

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_mano_surface_hypothesis_state.py \
  --contact-state "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json" \
  --joint-source hawor_npz \
  --hawor-npz "{RUN_ROOT}/measurements/hand_candidates/hawor_world/hawor_world_hands.npz" \
  --case "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --output "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}_surface_hypothesis_metric_mano/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json"
```

Required output: `{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}_surface_hypothesis_metric_mano/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json`. Its per-frame states must set `joint_state_policy` to a metric-MANO-preserved policy, keep `optimized_joints_world_m` equal to the selected metric source, carry contact-surface samples under `optimized_vertices_world_sample_m` / `contact_surface_vertices_world_sample_m`, and label contact as unresolved/uncertain. This is the default P19/P20 interval state. The raw P18 state remains provenance and may be evaluated separately, but it must not be the canonical rendered/evaluated hand state unless a later evidence record proves it improves metric MANO without visual regression.

## P19 full-duration render

P19 has two required substeps. First materialize the render-consumed state under `state/`; then render from that state. Do not use `scripts/render_v18_compact_rigid_tomato_temporal_mano_attempt.py` as the final P19 renderer. That script is a legacy diagnostic point/vertex renderer and cannot close the rigid-body artifact requirement.

### P19a build rigid render state

Script: `scripts/build_v19_rigid_render_state.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/build_v19_rigid_render_state.py \
  --case "{CASE_ID}" \
  --object-id "{OBJECT_ID}" \
  --object-label "{OBJECT_ID}" \
  --annotations "{RUN_ROOT}/measurements/object_geometry/visible_geometry/{OBJECT_ID}/annotations_v19_visible_geometry.json" \
  --pose-report "{RUN_ROOT}/measurements/pose_fits/{OBJECT_ID}_rigid_pose_graph/v19_rigid_object_pose_graph_report.json" \
  --completed-mesh "$COMPLETED_MESH_PLY" \
  --completion-report "$COMPLETION_REPORT" \
  --constraint-report "{RUN_ROOT}/measurements/contact_nonpenetration/{OBJECT_ID}_mano_object_constraint/v18_mano_object_constraint_state.json" \
  --temporal-mano-state "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}_surface_hypothesis_metric_mano/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json" \
  --output "{RUN_ROOT}/state/render_state/{OBJECT_ID}_rigid_render_state.json"
```

Required output: `{RUN_ROOT}/state/render_state/{OBJECT_ID}_rigid_render_state.json`. This file is the P19 renderer boundary: it must explicitly contain the completed mesh path, accepted full-timeline rigid pose rows, MANO/object constraint rows, temporal MANO state when present, and the projection contract. For a rigid branch, missing pose frames are a P19a failure unless the state explicitly records them as missing-pose uncertainty via `--allow-missing-poses`; the default runtime path must not omit rigid object poses for unobserved frames.

### P19b render rigid body from state

Script: `scripts/render_v19_rigid_state_artifact.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/render_v19_rigid_state_artifact.py \
  --render-state "{RUN_ROOT}/state/render_state/{OBJECT_ID}_rigid_render_state.json" \
  --output-root "{RUN_ROOT}/renders/{OBJECT_ID}_rigid_state_runtime" \
  --world-view local
```

Required output: full-duration overlay/world/side-by-side render branch listed in `{RUN_ROOT}/renders/{OBJECT_ID}_rigid_state_runtime/{CASE_ID}/v19_rigid_state_render_manifest.json` as `outputs.overlay`, `outputs.world`, and `outputs.side_by_side`. The renderer must rasterize mesh faces as a visible rigid body, not draw sampled vertices as a point cloud. The manifest must record the projection rule that scales source-coordinate intrinsics to the decoded render frame size; a 960x960 render of 1408x1408 source intrinsics must show `scale_xy` near `[960/1408, 960/1408]`. A missing or empty manifest value is a P19 failure. For `{OBJECT_ID}=keyboard`, the expected branch videos are `v19_overlay_keyboard.mp4`, `v19_world_keyboard.mp4`, and `v19_side_by_side_keyboard.mp4` under that case directory.

### P19c presentation rerender for Workbench item 4

Only after P19b has passed physical visual sanity, rerender the same P19a render state with presentation styling for audience readability. This is a render-only branch: it must not modify prediction state, rerun physical inference, publish canonical videos, or run metrics.

```bash
"{REMOTE_MODEL_PYTHON}" scripts/render_v19_rigid_state_artifact.py \
  --render-state "{RUN_ROOT}/state/render_state/{OBJECT_ID}_rigid_render_state.json" \
  --output-root "{RUN_ROOT}/renders/{OBJECT_ID}_rigid_state_presentation_runtime" \
  --world-view local \
  --render-style presentation
```

Required output: full-duration overlay/world/side-by-side presentation videos under `{RUN_ROOT}/renders/{OBJECT_ID}_rigid_state_presentation_runtime/{CASE_ID}/`. The manifest must record `rendered_state.render_style=presentation` and the alpha/wireframe settings. Visual review must verify that the physical body from P19b is preserved, object overpaint/text clutter are reduced, and unresolved MANO/contact remains explicitly labeled as uncertainty rather than accepted contact.

## P20 publish canonical render names

Script: `scripts/publish_v19_render_artifact.py`

```bash
"{REMOTE_MODEL_PYTHON}" scripts/publish_v19_render_artifact.py \
  --overlay "<render_manifest.outputs.overlay>" \
  --world "<render_manifest.outputs.world>" \
  --side-by-side "<render_manifest.outputs.side_by_side>" \
  --interval-state "{RUN_ROOT}/measurements/mano_interval_correction/{OBJECT_ID}_{INTERVAL_START}_{INTERVAL_END}_surface_hypothesis_metric_mano/{CASE_ID}/v18_joint_mano_interval_trajectory_state.json" \
  --output-dir "{RUN_ROOT}/renders/v19_published_runtime" \
  --canonical-dir "{RUN_ROOT}/renders" \
  --replace-canonical \
  --title "V19 runtime prediction {CASE_ID}"
```

Required output:

- non-empty `{RUN_ROOT}/renders/v19_overlay.mp4`
- non-empty `{RUN_ROOT}/renders/v19_world.mp4`
- non-empty `{RUN_ROOT}/renders/v19_side_by_side.mp4`

On filesystems that do not preserve POSIX symlinks, `publish_v19_render_artifact.py` must publish real canonical copies rather than zero-byte placeholder files. Treat a zero-byte canonical render as a P20 failure even if the published-runtime copy is valid. Treat an empty P19 manifest field as a P19/P20 contract failure before publication, not as a directory path.

## P21 visual consumption

Type: agent inspects the rendered videos and state rows.

Required output: append to `{RUN_ROOT}/state/v19_agent_evidence.md` a concrete statement of which physical mechanisms worked, which failed, and which state variables remain unresolved.
