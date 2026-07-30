# V19 Pi Run Contract

This contract defines how to start and govern a V19 annotation run. It is not a wrapper design. Pi is the harness; scripts are callable tools. The concrete English orchestration over existing components is `docs/v19_english_orchestration.md`; the runtime prompt must follow that runbook rather than fake numbered pipeline scripts.

## Launch model

Use Pi directly with the V19 system prompt:

```bash
pi \
  --provider occ \
  --model gpt-5.5:xhigh \
  --system-prompt "$(cat configs/v19_agent_system_prompt.md)" \
  --tools read,bash,edit,write \
  --prompt-template .pi/prompts/v19-run.md \
  "/v19-run <input-video> <run-root> [case-id]"
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
  --system-prompt "$(cat configs/v19_agent_system_prompt.md)" \
  "Smoke test only: report the active model route and do not perform annotation work."
```

The smoke command checks prompt/model routing. It is not a V19 annotation run and must not be reported as physical progress.

## Project representatives

Initial project representatives:

- `task5_tomato_960`: `/data2/egoscale_demo_30h/egoscale_tasks/20260118_1257_Rec3db6_P0_Sc6ab88_task_5/20260118_1257_Rec3db6_P0_Sc6ab88_task_5.mp4`
- `trash_1050`: `/data2/egoscale_demo_30h/egoscale_tasks/20260108_1057_Recf94e_P0_S994da4_task_9/20260108_1057_Recf94e_P0_S994da4_task_9.mp4`

The V18 baseline for monotonic comparison is `/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5/`.

## Compute targets

Local machine responsibilities:

- Pi session control.
- Git and task memory.
- Light file inspection and metadata extraction.
- Small state transformations.
- Review of rendered artifacts copied back from remote outputs.

A800/server responsibilities after non-mutating probes:

- HaWoR/WiLoR/HaMeR-style hand inference.
- Depth/SLAM or metric depth estimation.
- Open-vocabulary detection and video segmentation.
- TRELLIS or other mesh completion.
- Rendering batches when GPU-accelerated rendering is needed.
- HOT3D/H2O/DexYCB benchmark runs.

Current A800 target from the V19 task memory probe:

- SSH target: `yiwen@192.168.11.220`
- Remote repo root: `/mnt/user-home/yiwen/ego_annotation_remote/repo`
- Remote working root: `/mnt/user-home/yiwen/ego_annotation_remote/`
- Large output/data root: `/mnt/truenas-user-home/yiwen/ego_annotation_outputs`

Probe before heavy work. Record hostname, GPU memory/utilization, storage capacity, repo presence, environment roots, selected `GPU_ID`, and output root in `logs/harness_events.jsonl` and `.memory/tasks/2026-06-23-pipeline-v19/OPS.md`.

## Execution policy

A V19 run is a concrete execution of the English runbook over the input video and run root. It is not organized as an artificial evidence-cycle loop. At each step, Pi either runs the next named component, makes the required physical branch/judgment from rendered or geometric evidence, repairs the named mechanism that failed, or writes an explicit uncertainty state when the available measurements cannot support a stronger claim.

Operational settings such as compute target, selected GPU, model/provider route, and benchmark clip list are recorded for reproducibility, but they do not define pipeline progress.

## Run directory contract

Every V19 run writes one immutable run root:

```text
v19_runs/<run_id>/
  input/
    input_manifest.json
    frames/                    # optional extracted frames or links
  measurements/
    hand_candidates/
    object_candidates/
    depth_slam/
    masks_tracks/
    geometry_completion/
    pose_fits/
  state/
    v19_physical_state.json
    v19_uncertainty_state.json
    v19_agent_evidence.md
  renders/
    v19_overlay.mp4
    v19_world.mp4
    v19_side_by_side.mp4
    review_frames/
  evaluation/
    metrics.json
    ablations.json
    benchmark_export/
  logs/
    harness_events.jsonl
```

`state/` is the renderer boundary. Final visible annotations must be driven by `state/`, not by private measurement files, logs, or status labels.

## Minimal initial state

Before measurement tools run, Pi should create unresolved physical state rather than empty success containers:

```json
{
  "schema": "v19_physical_state.v0",
  "run_id": "<run_id>",
  "input_video": "<path>",
  "timeline": {
    "frame_count": null,
    "fps": null,
    "duration_s": null,
    "resolution": null
  },
  "camera": {
    "state": "unmeasured",
    "required_for_metric_claims": true
  },
  "hands": [],
  "objects": [],
  "contacts": [],
  "occlusions": [],
  "nonpenetration": [],
  "renderer_boundary": "renders consume this state directory only"
}
```

Unresolved state is not progress by itself. It is the starting point that prevents silent success when physical variables have not been measured.

## Physical state requirements

The current executable command truth is recorded in `docs/v19_english_orchestration.md`. If that runbook marks a required component as missing, a V19 run must stop with the named missing implementation and blocked physical variable rather than authoring placeholder outputs during annotation.


A V19 state claim must specify the physical variable, evidence, uncertainty, and renderer consumption path.

Required variable families:

- MANO hand state over time, with metric camera/world transforms.
- Camera/head pose and metric scale provenance.
- Object roster, geometry state, and pose/posterior where branch-selected.
- Visibility and occlusion ownership.
- Contact/near-contact/non-contact with patch and distance evidence.
- Nonpenetration residuals/uncertainty.
- Residuals and uncertainty consumed by visualization and evaluation.

## Rigid-object branch contract

Once the agent classifies an object as rigid over a span, the following branch is mandatory:

```text
rigid decision
  -> TRELLIS or equivalent per-instance mesh completion
  -> alignment/adaptation to observed depth/mask evidence
  -> visible-frame SE(3)/Sim(3) pose fitting
  -> temporal factor-graph correction with camera/depth/MANO/contact/occlusion/nonpenetration terms
  -> renderer consumes corrected rigid mesh pose
```

Visible surfaces are metric measurements and anchors. They cannot replace rigid pose after rigid classification. Residuals widen uncertainty or trigger repair; they do not demote the state to a point cloud.

## Benchmark contract

The initial V19 benchmark scope is fixed and small:

- HOT3D primary: 3-5 clips.
- Optional H2O secondary: 2-3 clips.
- DexYCB fallback only if H2O blocks the secondary slot.

No other dataset is an acceptance target without a design amendment. Metrics must be reported only when the selected dataset annotates the corresponding physical claim family.

## Yield gate

Before yielding from a substantial V19 implementation turn, run a clean-room adversarial review with:

- user intent;
- changed artifacts;
- physical deliverables owed;
- completed physical facts;
- remaining next actions;
- parked user decisions;
- constraints;
- risks of proxy progress or wrapper regression.

Apply the findings before reporting, unless the only remaining issue is a true user decision or high-risk irreversible step.
