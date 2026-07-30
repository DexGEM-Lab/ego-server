# Cosmos3 GPU6 guarded candidate cutover

This is the only lifecycle for the Ray-managed Cosmos3 candidate. It leaves the
bare production process on GPU6/port 8001 untouched until an authorized operator
stops it. The candidate serves only on port 28006; it does not bind 8001.

## Verified standalone evidence

The cutover gate consumes these remote files in place and never copies them into
Git:

- `/home/zjh/cosmos3_ray_serve/standalone/logs/verify_report.json`
- `/home/zjh/cosmos3_ray_serve/standalone/logs/finalize_20260717T000000Z.json`

The first file has one transformers warning line before its JSON object. The gate
parses the object after that preamble, then requires successful Ray/Serve/vLLM/Torch
and Cosmos3 plugin imports, the registered
`Cosmos3ReasonerForConditionalGeneration` architecture, CUDA availability, and no
`.pth` reference into `/home/ylang`. It cross-checks the finalize report's standalone
interpreter, ABI versions, unchanged bare 8001 evidence, no-overlay report, CPU-only
Ray diagnostic, and report linkage.

The required interpreter is:

```text
/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python
```

The adjacent `/home/zjh/cosmos3_ray_serve/.venv` is not a candidate interpreter.
The standalone venv intentionally does not install `ego_annotation`. Ray Serve's
declarative config rejects a local `runtime_env.working_dir`, so the GPU6 Ray head
inherits `PYTHONPATH=/home/zjh/cosmos3_ray_serve/workspace`; its local workers import
the curated workspace through that inherited path. It also inherits
`HF_HOME=/home/ylang/.cache/huggingface`, the read-only validated bare-model snapshot,
so model loading does not redownload weights into a root-owned cache.

## Fixed topology

| purpose | address / port |
|---|---|
| bare production baseline | `http://127.0.0.1:8001` |
| Ray GCS | `127.0.0.1:26801` |
| Ray dashboard | `http://127.0.0.1:26800` |
| Ray object manager / node manager | `26802` / `26803` |
| Ray worker ports | `26900,26901,26902,26903,26904,26905,26906,26907,26908,26909,26910,26911,26912,26913,26914,26915,26916,26917,26918,26919,26920,26921,26922,26923,26924,26925,26926,26927,26928,26929,26930,26931` |
| Ray Serve candidate | `http://127.0.0.1:28006` |

Every Serve command carries either the explicit GCS address or the explicit
Dashboard address. `RAY_ADDRESS` is removed from the candidate command environment.
Worker-port ranges are forbidden: Ray receives the literal comma-separated list
above.

## Preflight (no GPU or service mutation)

After this repository revision is present in the remote workspace, validate the
reports without starting Ray or touching port 8001:

```bash
cd /home/zjh/cosmos3_ray_serve/workspace
/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python -m ego_annotation.serving.cosmos3_cutover \
  --standalone-artifacts-dir /home/zjh/cosmos3_ray_serve/standalone
```

Expected result: JSON with `status` equal to `preflight_passed`.

## Guarded cutover command

Run the committed launcher during an authorized maintenance window. It creates a
new `ego_annotation` tmux window before it stops bare GPU6, so the successful
candidate is never owned by a disposable benchmark/client shell:

```bash
COSMOS3_WORKSPACE=/home/zjh/cosmos3_ray_serve/workspace \
COSMOS3_STANDALONE=/home/zjh/cosmos3_ray_serve/standalone \
bash scripts/cosmos3_guarded_cutover.sh \
  --run-root /vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/<new-run-id>
```

The tmux-owned transaction validates the standalone reports, stops only the bare
Cosmos3 port-8001 process, launches the GPU6 Ray head, deploys Serve on `28006`,
checks two typed real-image responses (the second after the first client exits), and
runs the manifest-defined open-loop sweep. Before acceptance, any failure performs
only the scoped candidate stop and creates a bare-GPU6 restore window. After these
checks succeed, rollback is disarmed and the shell `exec`s
`cosmos3_resident_driver`, which holds an explicit Ray connection in the tmux
foreground. The persistent window is the live ownership boundary; ending a client
or benchmark process does not end the service.

## Scoped rollback

Rollback has three explicit operations:

```bash
# Remove only the Cosmos3 Serve application on the explicit GPU6 dashboard.
/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python -m ray.serve.scripts shutdown -a http://127.0.0.1:26800 -y

# Stop only Ray PIDs whose command line names this candidate's temp directory.
/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python -m ego_annotation.serving.cosmos3_cutover \
  --scoped-stop --temp-dir /tmp/ray-ego-serve-cosmos3

# Restore the bare production service only when it was intentionally stopped.
su - ylang -c 'bash /home/zjh/cosmos3_ray_serve/RESTORE_BARE_COSMOS3.sh'
```

Do not use `ray stop`: it is node-wide and can stop unrelated Ray clusters. The
candidate stop command enumerates and signals only process command lines containing
`/tmp/ray-ego-serve-cosmos3`.

## Remaining prerequisite

An authorized maintenance window must stop the existing bare `ylang` Cosmos3 vLLM
process on GPU6/8001 so the candidate can own GPU6 memory. The reports verify the
standalone ABI and plugin surface, not a loaded Ray-managed model; after the guarded
command, candidate health and equivalent inference on port 28006 remain the live
acceptance evidence before callers can be moved.
