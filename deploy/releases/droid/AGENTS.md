# Ego Annotation Operating Rules

## Work Ethics

Do not satisfy a requirement with an obviously false simplification. Difficulty cannot justify an incorrect substitute. If a required component is hard, keep the requirement intact, expose the unsolved part as an unsolved part, and work on the real mechanism.

Object pose in this project means reconstructed object geometry when the object is manipulated. A centroid, sphere, bounding box, category-specific primitive, or visual patch is not an acceptable replacement for object mesh reconstruction.

Hand state in this project means a time-indexed metric 3D MANO state with provenance, camera/world-frame semantics, surface or reproducible MANO parameters, and uncertainty. A 2D detector track, 2D keypoint skeleton, projected joint overlay, smoothed image-space trajectory, or hand box is not a physical hand state and cannot ground contact, occlusion ownership, object pose, or nonpenetration claims.

Do not claim deadline success, version closure, or meaningful progress from validators, clean git state, full-frame bookkeeping, JSON fields, schema rows, or a finer unresolved/accepted ledger. Those are evidence-preservation tools only. The artifact the user needs is renderable video annotation: overlay/world/side-by-side videos whose visible marks are driven by real mechanisms. JSON is backing data only; it is useful only to the extent that it drives, explains, or reproduces the rendered annotations. Do not ship containers: a JSON file, report, video file, overlay label, row count, module field, or rendered box is not the artifact unless the visual annotation content inside it is produced by the named mechanism and sanity-checked against the actual video/geometric evidence. Do not put available data into a container and claim the required object, hand, contact, occlusion, or geometry annotation exists. A checkpoint is progress only when it changes the delivered annotation/render in the direction of the named requirement, or when a real attempted mechanism fails and the failure mechanism is preserved with commands, artifacts, visual evidence, and the next causal implication.

When the user points out non-progress, stop the current loop immediately. Do not answer by adding more gates, readiness flags, blockers, manifests, or status overlays. Re-ground the task in the final artifact, choose the smallest integrated mechanism that could improve it, run it, render it, and report either the visible improvement or the concrete mechanism failure.

For time-boxed recovery work, maintain an hour-by-hour budget that separates code-edit time, GPU/model runtime, full-pipeline/render runtime, visual review time, and iteration count. If the time box cannot close the full requirement, say so before implementation and commit only to artifact-changing attempts or failure analyses that can complete inside the time box.

## Methodology

Do not encode visual case variation with hand-written if/else logic or object-family state machines. Python branches for categories such as color, object class, material, or action phrase are a failed perception strategy.

Use open-source vision models, open-vocabulary detectors, segmentation/tracking models, or LLM/VLM calls to produce object plans and segmentation evidence. The geometry pipeline should consume model outputs such as masks, tracks, depths, poses, confidences, and captions through one uniform reconstruction path.

When different cases require different treatment, put the difference in model-produced data or learned/open-vocabulary perception outputs. Keep the downstream reconstruction, filtering, contact reasoning, and rendering code category-agnostic unless a domain discontinuity is physically real and represented explicitly in the data.

## Pipeline Reasoning Standard

This project exists because the measurements are noisy. If DROID/depth/HaWoR/WiLoR/OWLv2/SAM2 were locally reliable enough to decide the annotation alone, the consistency pipeline would be unnecessary. Do not gate the pipeline on local certainty from any component.

Reasoning and judgment are the primary standards. Rigor comes from understanding the intended transformation, inspecting real outputs, distinguishing implementation mistakes from noisy-but-usable measurements, and choosing the next constructive intervention. Caution is only a support tool; it must not become refusal to build.

For each module, the objective is sanity, not perfection and not passing an arbitrary programmatic gate. Build the module, run it, inspect representative visual/geometric/timeline outputs, and judge whether the result is sane enough to feed downstream as an approximate uncertain measurement. Programmatic checks catch mechanical failures such as missing files, frame-count mismatch, empty outputs, invalid transforms, impossible units, or broken schemas; they do not replace visual and causal judgment.

A failed or weak local measurement is normally an input to filtering, not a reason to stop. The factor graph/consistency layer should reconcile, smooth, downweight, expose, or carry uncertainty from noisy measurements. It is not a final readiness gate that waits for upstream certainty.

Distinguish these states explicitly:

- Implementation mistakes: missing rendered-annotation writer, missing module output that drives the rendered annotation, crash, invalid frame/depth/camera alignment, broken backing schema needed to reproduce the videos, disconnected rendered dataflow, or nonsensical visualization. These must be fixed.
- Noisy but usable measurements: weak masks, drifting hand boxes, partial surfaces, approximate poses, ambiguous contact, uncertain occlusion, or low confidence. These must continue downstream with uncertainty.
- Fundamental information limits: facts not inferable from the available sensors. These should be represented as uncertainty in the artifact, not used to omit the module.

Every complete pipeline version should be run end-to-end, rendered, inspected, and compared to the previous version using subjective judgment when ground truth is absent. Intensive reasoning/research belongs after seeing the delivered pipeline output and should inform the next pipeline version or patch.

## Research Discipline

Do not run sequential guesses. Before spending runtime on an experiment, write the causal account that makes the intervention necessary: the concrete defect in the final artifact, the physical variable that is wrong, the mechanism that could have produced it, why the proposed observation/intervention couples to that variable, and what each possible outcome would imply for the next implementation step.

Do not frame work as “try a branch” or “kill a branch.” A negative result is not a stopping state and not a reason to hop to an unrelated mechanism. It must either revise the causal model, expose that the experiment did not actually test the mechanism, reveal a missing coupling between measurement and solver/render, or identify the next stronger intervention. If no outcome of an experiment would change the model or force a next action, the experiment is not worth running.

An experiment must be designed so that every result is informative. “It works” and “it does not work” are not interpretations. State the discriminating predictions before execution: what would be observed if the suspected mechanism is dominant, what would be observed if the measurement is weak or miscoupled, what would be observed if another uncertainty dominates, and how each case changes the artifact-building plan.

Never abandon a mechanism because one implementation was inert. First decide whether the causal idea was wrong, the measurement was too weak, the solver ignored it, the render failed to expose it, the intervention attacked a proxy, or another variable dominated. Continue by correcting the model or experiment until the related facts are understood well enough to justify the next artifact-changing mechanism.

## Integrity And Monotonicity

Do not cheat by substituting a convenient proxy for a named pipeline mechanism. Approximate measurements are expected; approximate implementation of the spec is not allowed. If the design names a variable family, factor family, reconstruction stage, render layer, or model branch, the code and final artifact must contain that mechanism explicitly. A centroid is not object pose; a smoother is not a factor graph; mask overlap is not contact ownership; an abstract panel is not a metric 3D render; a status field is not an implemented module.

V18 and later versions must be monotonic relative to the previous working version unless a design amendment explicitly rejects a previous capability as invalid. Monotonicity means preserving every valid prior capability and adding new capability on top. Do not replace V16 MANO rendering, object mesh rendering, camera/depth backbone, or metric world visualization with weaker boxes, labels, abstract panels, or schema fields. If a new module is weaker than the previous version for a field, keep the previous version's output as the base layer and add the new module as an additional uncertain hypothesis.

If visual inspection reveals degradation relative to the previous version, report and fix it immediately. Do not call it a tradeoff. There is no compensation for losing an existing capability; monotonic progress requires preservation plus extension.

When a module is hard, implement the specified mechanism directly at the best available fidelity before the deadline. Do not rename a partial helper as the module. If a mechanism is approximate, expose the approximation in estimates and uncertainty, not in the existence or identity of the mechanism.

## Versioning And Delivery

Starting at v16, every pipeline version is a complete pipeline version. A version cannot close with component evidence, a short window, or a partial render. A full version may be approximate and uncertain, but it must execute the named modules and produce full-video artifacts.

Do not silently substitute pipeline components after an upfront design exists. If a design names a baseline component, that component remains required until a design amendment records the evidence that rejects it or replaces it. Cached outputs may be reused as memoized results of the same logical stage, but the pipeline must remain self-contained from raw video and named model/config inputs.

Each v16-or-later version must begin with an upfront design document before implementation. The design must define the raw-video input contract, full-timeline state variables, perception sources, optimization objective, physical consistency terms, acceptance checks, representative raw videos, expected failure modes, and complete render outputs.

Every v16-or-later deliverable must have the same frame count and duration as the original raw video. Short windows, contact slices, debug clips, contact sheets, and selected-frame renders are QC artifacts only.

Bug fixes, threshold changes, renderer fixes, server setup, and local mechanism studies belong inside the current version as patches or experiments. They do not create a new top-level version number.

## Work Progress Standard

Understanding is required before risky intervention, but understanding must produce action. Once the missing mechanism is identified, progress means implementing the mechanism, running the pipeline, inspecting outputs, improving the artifact, or preserving a concrete failure that changes the next implementation decision. Repeating readiness summaries, blocker manifests, audits, or render/status updates without a new artifact-producing attempt is not progress.

Maintain ledgers and audits at full standard, but they should be thin records of actual work. They must not become a separate workstream or an excuse for not advancing the causal pipeline. A ledger entry that only says the same thing is still not done is evidence of task steering failure.

Thresholds, residual checks, and audits are diagnostic instruments. They may guide confidence labels, prioritization, and debugging, but they must not become arbitrary acceptance gates that prevent the pipeline from producing approximate uncertain outputs. Use subjective judgment from rendered videos, overlays, meshes, trajectories, and timelines to decide whether a module output is sane enough to continue.

## Anti-Avoidance Invariant

Before every substantive action, name the strict blocker or mechanism uncertainty that the action will reduce. If the action cannot be tied to a named blocker, do not do it. When several blockers are open, choose the hardest essential root blocker whose resolution would unlock downstream physical claims; do not choose an easier support task merely because it is locally verifiable.

The following are support actions, not primary progress: validators, audits, reports, ledgers, manifests, schema fields, render styling, status overlays, provenance-only patches, refactors, documentation updates, subagent reviews, and commits. They are allowed only when they verify a just-implemented mechanism, expose a concrete failure mechanism that determines the next intervention, or are strictly necessary to keep an artifact reproducible after a mechanism change. They must not be used as a substitute for hand/MANO foundation, object geometry/pose, contact, occlusion, nonpenetration, or runtime mechanisms.

Do not close a task, declare a checklist improved, or summarize success immediately after a support action. First show what physical annotation mechanism changed, what rendered or geometric evidence changed because of that mechanism, and which strict blocker was actually reduced. If no strict blocker was reduced, say the action was support-only and continue to mechanism work.

When the user identifies avoidance or false progress, freeze new feature work. Inspect and revert any unvalidated partial edits unless they directly implement a named blocker and have already passed mechanism-level validation. Restore contaminated generated artifacts from committed code when necessary. Then resume from one explicit strict blocker; do not answer with more readiness framing.


## Runtime And Occlusion Discipline

Starting at v18, runtime is a design invariant. The default pipeline for a raw video must run in the same order of magnitude as the input duration. A method that takes hours for a roughly one-minute clip is a failed default design, even if its intermediate evidence is interesting. Per-instance neural reconstruction or training loops such as BundleSDF/NeRF-style optimization may be used only as offline research branches, never as the default path or as a way to discover obvious physical state types.

Heavy compute placement is a project invariant. Do not run local GPU jobs, local model inference, or heavy local CPU inference on the user workstation unless the user explicitly authorizes that exact local run. SAM2, OWLv2, HaWoR, UniDepth, VLM/LLM batch inference, training, reconstruction, or any other fan/noise/thermal-heavy workload must run on the server/A800 or another non-local compute target. The local machine is for light repo edits, inspection, orchestration, and small non-heavy scripts only. Before launching any model or heavy job, make the compute target explicit in the command/session notes; accidental framework defaults such as `cuda` on the local machine are forbidden.

Occlusion must be represented explicitly. Hands and objects need per-frame visibility states such as visible, partially visible, occluded, out-of-frame, and unresolved, with occluder ownership and uncertainty for inferred states. Do not treat occlusion as only missing data, do not silently fill occluded hands/objects with certain poses, and do not claim contact or object pose through occlusion without depth-order and temporal evidence.

During agent command execution, do not use `sleep` for any reason. Do not run `sleep`, polling loops, timed waits, or equivalent idle waits as a progress substitute. Long-running work must run in tmux or a job system with durable sentinels/logs; after launch, either work on an independent blocker, perform an immediate non-blocking status/log check, or yield the job handle. Interactive tools must be driven without `sleep`; use explicit prompts, captured output, or tool/session state instead.

Agent roles must remain separated. The parent/development agent edits code, docs, prompts, runbooks, task memory, launch contracts, and infrastructure provisioning; it may launch or monitor runtime work, but it must not manually perform the runtime agent's pipeline stages or assemble prediction artifacts and then call them runtime output. A runtime agent is a prediction executor only: it receives the input video, fresh run root, case id, repository code/runbook, parent-preflighted model/runtime environments, and prediction-side sensor metadata. It must not create virtualenvs, install packages, repair symlinks, run setup scripts, spend a runtime phase on infrastructure, or otherwise provision infrastructure inside a prediction run. If a named prediction command fails because infrastructure is invalid, it records that command's phase blocker and stops. It must not receive GT sidecars, evaluator targets, ablation objectives, Workbench/task-memory instructions, parent-session concerns, or prompts about what the parent must not do. Evaluation/ablation is a third phase after runtime prediction output is frozen; only then may GT/baselines/evaluator scripts be consumed, and evaluation must not mutate prediction state.

Runtime agents must not share the parent agent's active cwd, dirty working tree, or full project worktree. Launch a runtime agent from a curated runtime workspace containing only the runtime system prompt, prompt template, one authoritative runtime spec, and scripts/config/assets required by that spec. Do not expose `.memory/`, `AGENTS.md`, parent/development docs, evaluator/benchmark docs, historical task reports, or unrelated project documentation to the runtime agent. Generated prediction artifacts must go under a fresh run root. If any runtime-visible prompt/spec includes Workbench, `.memory/tasks`, OPS/EPISTEMIC, parent-session language, GT/evaluation/evaluator language, ablation objectives, or development checklists as runtime instructions, stop and repair the runtime bundle before launch.

Use a single tmux session per project for ordinary project commands. For this project, the canonical parent/development agent session is `ego_annotation`. Create multiple windows inside that session for separate parent-owned commands or long-running jobs; do not create one tmux session per task, branch, validator, probe, or script. A clean runtime-agent run is the exception: it must use a dedicated visible tmux session/window tied to its isolated cwd/worktree so the user can inspect the runtime agent directly. Tmux is execution containment, not evidence of progress.
