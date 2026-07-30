# V22 Pi Run Contract

This contract defines how to start and govern a V22 annotation or benchmark run. It is not a wrapper design. Pi is the harness; scripts are callable tools. The V22 module list is `docs/v22_harness_list.md`; the concrete English orchestration is `docs/v22_english_orchestration.md`.

## Launch Model

Use Pi directly with the V22 system prompt:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v22_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v22-run.md \
  "/v22-run <input-video-or-dataset-root> <run-root> [case-id-or-sample-id] [mode] [side-inputs-json]"
```

For an isolated model-route smoke check only, do not give tools, context, skills, extensions, or a persistent session:

```bash
pi -p \
  --no-tools \
  --no-session \
  --no-context-files \
  --no-skills \
  --no-extensions \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v22_agent_system_prompt.md)" \
  "Smoke test only: report the active model route and do not perform annotation work."
```

The smoke command checks prompt/model routing. It is not a V22 annotation run and must not be reported as physical progress.

## Modes

`v22_infer` annotates an arbitrary egocentric video. It writes V22 prediction state and full-duration render artifacts.

`v22_benchmark` prepares a supported benchmark sample with GT isolation, runs the same prediction-side V22 annotation flow, renders prediction artifacts, and evaluates only after prediction state/renders exist.

## Hard Runtime Document Set

The V22 single-run runtime document set is intentionally small:

```text
configs/v22_agent_system_prompt.md
.pi/prompts/v22-run.md
docs/v22_run_contract.md
docs/v22_harness_list.md
docs/v22_english_orchestration.md
```

V22 parallel batch runs add these manager/runner artifacts:

```text
.pi/prompts/v22-parallel.md
.pi/prompts/v22_parallel_runner.md
configs/v22_parallel_runtime_profile.json
scripts/build_v22_parallel_manifest.py
scripts/v22_parallel_claim_next.py
scripts/launch_v22_parallel_agents.py
```

Design/background documents may inform future amendments, but they are not required runtime authority unless this contract or the system prompt names them.

## Active Chain Contract

V22 target segmentation starts from agent target selection and uses the current OWLv2 bbox-prompt SAM2 proper chain:

```text
agent object plan
  -> agent-selected OWLv2 keyframes
  -> OWLv2 bbox proposals
  -> approved bbox prompts
  -> SAM2 proper masks
  -> contamination review
  -> accepted masks for visible metric surfaces
```

Do not substitute older object-prompt, local-mask, or disabled bbox paths as the main chain. If the active chain fails, record the concrete failed mechanism, repair that mechanism, or write an explicit unresolved state.

## Compute Targets

Local machine responsibilities:

- Pi session control.
- Git and task memory.
- Light file inspection and metadata extraction.
- Small state transformations.
- Review of rendered artifacts copied back from remote outputs.

Default A800/server target for V22:

```text
SSH command: ssh -p 57938 zjh@115.190.235.210
SSH host: 115.190.235.210
SSH port: 57938
SSH user: zjh
Compute target label: zjh@115.190.235.210
Remote working root: ~
Remote repo root: ~/ego-annotaion-jiahong-dev
Remote checkpoint root: ~/ego-annation-checkpoints
Large output/data root: ~/data
Canonical tmux session: ego_annotation
```

Server/A800 responsibilities after non-mutating probes:

- Hand model inference and MANO optimization.
- Depth/SLAM, metric depth, and multiview candidates.
- Open-vocabulary detection and SAM2 video segmentation.
- Mesh completion and generated shape candidates.
- GPU-accelerated rendering batches.
- Benchmark prediction/evaluation runs when heavy.

Probe before heavy work. Record hostname, GPU memory/utilization, storage capacity, repo presence, environment roots, selected GPU, and output root in `logs/harness_events.jsonl`.

Before any single-data or parallel V22 measurement module runs, execute the hard remote environment gate from the remote repo:

```bash
python scripts/preflight_v22_environment.py --output <preflight-output.json>
```

The report must have `status: ok`, `required_failed: 0`, and `optional_missing: 0`. A failed preflight is a shared environment/code-contract blocker, not a per-video failure. The gate checks fixed algorithm script availability/syntax, module Python routing from `configs/v22_gpu_runtime_profile.json`, required model repos/checkpoints/configs, CUDA availability, and model-level smoke initialization for the fixed depth/detection/segmentation stack.

For single-data V22 runs, GPU-heavy submodules use the same GPU selection semantics as the feat/parallel runner without entering parallel mode: read `configs/v22_gpu_runtime_profile.json`, estimate the module's VRAM from `estimated_vram_mb`, then launch the submodule through `scripts/v22_gpu_wrapper.py` with `--module-id`, `--request-mb`, and `--log-jsonl <run_root>/logs/gpu_wrapper_events.jsonl`. The wrapper is a child-command launcher only. It does not claim dataset entries, launch multiple agents, or replace Pi as the harness.

For `/v22-parallel`, the manager uses `configs/v22_parallel_runtime_profile.json` and `scripts/build_v22_parallel_manifest.py` to create a shared manifest, runs `scripts/preflight_v22_environment.py` on the remote repo, then launches runner Pi agents through `scripts/launch_v22_parallel_agents.py` only when the preflight report has `status: ok`, `required_failed: 0`, and `optional_missing: 0`. Default parallelism is 32. Each runner claims one entry at a time with `scripts/v22_parallel_claim_next.py`; every GPU-heavy child command still goes through `scripts/v22_run_module.sh gpu ...` or `scripts/v22_gpu_wrapper.py`. Parallel runners must preserve `V22_PARALLEL_AGENT_ID` in remote child process environments so `scripts/run_v22_resource_monitor.py` can attribute CPU/GPU samples by agent. `configs/v22_gpu_runtime_profile.json` is the source of module-specific Python env routing; DepthPro is routed through its dedicated remote venv when enabled, while UniDepth remains the required default metric-depth path for the EgoScale batch.

## Run Directory Contract

Every V22 run writes one immutable run root:

```text
v22_runs/<run_id>/
  input/
    input_manifest.json
    raw_frame_manifest/manifest.json
    source_frame_manifest/manifest.json
    source_frame_manifest/rgb/*.jpg
  measurements/
    camera_depth/
    depth_candidates/
    hand_candidates/
    object_candidates/
    object_tracks/
    object_visible_surfaces/
    object_geometry/
    object_geometry_mesh_pose/
    contact_occlusion_nonpenetration/
  review/
    segmentation_sam2_proper/
  tuning/
    depth_camera/
    hand_mano/
    segmentation/
  state/
    annotations_v22_renderable.json
    v22_physical_state.json
    v22_uncertainty_state.json
    v22_observation_bundle.json
    v22_agent_evidence.md
  renders/
    v22_overlay.mp4
    v22_world.mp4
    v22_side_by_side.mp4
    review_frames/
  evaluation/
    reference_manifest.json
    benchmark_iterations.jsonl
    algorithm_parameter_changes.jsonl
    final_selection_report.md
  logs/
    harness_events.jsonl
```

`state/` is the renderer boundary. Final visible annotations must be driven by V22 state, not private measurement files, logs, run summaries, or status labels.

## Minimal Initial State

Before measurement tools run, create unresolved physical state rather than empty success containers:

```json
{
  "schema": "v22_physical_state.v0",
  "mode": "v22_infer_or_benchmark",
  "run_id": "<run_id>",
  "input": "<input-video-or-dataset-sample>",
  "timeline": {
    "frame_count": null,
    "fps": null,
    "duration_s": null,
    "resolution": null
  },
  "camera_depth": {
    "state": "unmeasured",
    "required_for_metric_claims": true
  },
  "hands": [],
  "objects": [],
  "contacts": [],
  "occlusions": [],
  "nonpenetration": [],
  "renderer_boundary": "renders consume V22 state files only"
}
```

Unresolved state is not progress by itself. It prevents silent success when physical variables have not been measured.

## Physical State Requirements

A V22 state claim must specify the physical variable, evidence, uncertainty, and renderer consumption path.

Required variable families:

- Camera/head pose, intrinsics, depth, and metric scale provenance.
- Metric MANO hand state over time with visibility and uncertainty.
- Target object roster from visual/task evidence.
- Accepted object masks/tracks from SAM2 proper plus contamination review.
- Object visible surfaces, completed/adapted geometry, and pose/posterior where branch-selected.
- Contact/near-contact/non-contact with patch, distance, ownership, and uncertainty evidence.
- Occlusion/visibility ownership.
- Nonpenetration residuals/uncertainty.
- Residuals and uncertainty consumed by visualization and evaluation.

## Rigid-Object Branch Contract

Once the agent classifies an object as rigid over a span, this branch is mandatory:

```text
rigid decision
  -> accepted SAM2 proper mask and selected depth/camera evidence
  -> metric visible surface
  -> mesh completion/adaptation
  -> visible-frame SE(3)/Sim(3) pose fitting
  -> temporal factor-graph correction with camera/depth/MANO/contact/occlusion/nonpenetration terms
  -> renderer consumes corrected rigid mesh pose
```

Visible surfaces are metric measurements and anchors. They cannot replace rigid pose after rigid classification.

## Bottleneck Tuning Contract

Depth/camera, segmentation, and hand/MANO are bottleneck observations. If a bottleneck output is weak, the V22 agent must first run or record a sample-bound strong-tuning attempt before downweighting it.

Tuning records live under:

```text
tuning/depth_camera/<candidate_id>/attempt_<k>.json
tuning/hand_mano/<hand_track_id>/attempt_<k>.json
tuning/segmentation/<object_id>/attempt_<k>.json
```

Each tuning record must include predicted effect, changed parameters or missing implementation, observed residuals, visual review, and keep/reject/continue decision.

## Benchmark Contract

In `v22_benchmark`, GT paths must live only under `evaluation/reference_manifest.json` and may be read only after prediction-side state and renders exist. GT must not enter prediction manifests, candidates, masks, depth selection, tuning, state, renders, or algorithm choices.

Benchmark evaluation is evidence, not authority. Controller interventions after evaluation must be atomic, mechanism-based, predicted before rerun, and recorded with parameter changes and overfitting risk.

## Yield Gate

Before yielding from a substantial V22 implementation or run, consume the rendered overlay/world/side-by-side videos when they exist and report:

- what physical state changed;
- what mechanism explains the change;
- what rendered or geometric evidence supports it;
- what remains uncertain;
- which next action is required if the artifact is not yet physically sane.
