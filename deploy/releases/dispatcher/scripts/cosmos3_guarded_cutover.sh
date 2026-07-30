#!/usr/bin/env bash
# Transactional GPU6 Cosmos3 cutover.  The accepted candidate remains owned by a
# foreground Ray driver in an ego_annotation tmux window; only pre-acceptance failures
# invoke the scoped teardown and bare-GPU6 restoration path.
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKSPACE=${COSMOS3_WORKSPACE:-$SCRIPT_ROOT}
STANDALONE=${COSMOS3_STANDALONE:-/home/zjh/cosmos3_ray_serve/standalone}
PY="$STANDALONE/.venv/bin/python"
SESSION=${COSMOS3_TMUX_SESSION:-ego_annotation}
BENCHMARK_ROOT=/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks
ACCEPTANCE_SEED=${COSMOS3_ACCEPTANCE_SEED:-$(cd "$(dirname "$0")/.." && pwd)/assets/cosmos3}

usage() {
  echo "usage: $0 --run-root <new benchmark root> [--in-tmux] [--skip-benchmark] | --classify-bare-count <N>" >&2
  exit 64
}

classify_bare_count() {
  case "$1" in
    0) printf '%s\n' cold_start ;;
    1) printf '%s\n' cutover ;;
    *) printf 'expected zero or one bare Cosmos3 process, found %s\n' "$1" >&2; return 2 ;;
  esac
}

run_root=""
in_tmux=0
skip_benchmark=0
classify_count=""
while (($#)); do
  case "$1" in
    --run-root) run_root=${2:-}; shift 2 ;;
    --in-tmux) in_tmux=1; shift ;;
    --skip-benchmark) skip_benchmark=1; shift ;;
    --classify-bare-count) classify_count=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
if [[ -n "$classify_count" ]]; then
  [[ "$classify_count" =~ ^[0-9]+$ ]] || usage
  classify_bare_count "$classify_count"
  exit $?
fi
[[ -n "$run_root" && "$run_root" == "$BENCHMARK_ROOT"/* ]] || usage

if (( ! in_tmux )); then
  tmux has-session -t "$SESSION"
  window="cosmos3-ray6-$(date -u +%Y%m%dT%H%M%SZ)"
  script=$(readlink -f "$0")
  extra=()
  (( skip_benchmark )) && extra+=(--skip-benchmark)
  tmux new-window -d -t "$SESSION" -n "$window" \
    "exec bash $(printf '%q' "$script") --in-tmux --run-root $(printf '%q' "$run_root") ${extra[*]}"
  printf '{"event":"cutover_launched","tmux_window":"%s:%s","run_root":"%s"}\n' "$SESSION" "$window" "$run_root"
  exit 0
fi

ROOT="$run_root/cosmos3"
mkdir -p "$ROOT/raw"
for seed_file in representative_request.multipart representative_request_headers.json; do
  if [[ ! -s "$ROOT/$seed_file" ]]; then
    [[ -s "$ACCEPTANCE_SEED/$seed_file" ]] || { echo "missing Cosmos3 acceptance seed: $ACCEPTANCE_SEED/$seed_file" >&2; exit 66; }
    cp "$ACCEPTANCE_SEED/$seed_file" "$ROOT/$seed_file"
  fi
done
exec > >(tee -a "$ROOT/raw/cutover.log") 2>&1
rollback_armed=1

rollback() {
  local rc=$?
  if (( ! rollback_armed )); then
    exit "$rc"
  fi
  printf '{"event":"candidate_failure","exit_code":%s}\n' "$rc" > "$ROOT/raw/rollback_trigger.json"
  "$PY" -m ray.serve.scripts shutdown -a http://127.0.0.1:26800 -y || true
  "$PY" -m ego_annotation.serving.cosmos3_cutover --scoped-stop --temp-dir /tmp/ray-ego-serve-cosmos3 || true
  local rollback_window="cosmos3-rollback-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-window -d -t "$SESSION" -n "$rollback_window" \
    "exec su - ylang -c 'CUDA_VISIBLE_DEVICES=6 bash /home/zjh/cosmos3_ray_serve/RESTORE_BARE_COSMOS3.sh'"
  printf '{"event":"rollback_command_started","tmux_window":"%s:%s"}\n' "$SESSION" "$rollback_window" > "$ROOT/raw/rollback_command.json"
  exit "$rc"
}
trap rollback ERR INT TERM

cd "$WORKSPACE"
"$PY" -m ego_annotation.serving.cosmos3_cutover --standalone-artifacts-dir "$STANDALONE" > "$ROOT/raw/preflight_at_cutover.json"

mapfile -t bare_pids < <("$PY" - <<'PY'
from pathlib import Path
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd = (proc / 'cmdline').read_bytes()
    except OSError:
        continue
    if b'vllm.entrypoints.cli.main' in cmd and b'--port\x008001' in cmd and b'Cosmos3-Nano' in cmd:
        print(proc.name)
PY
)
bare_mode=$(classify_bare_count "${#bare_pids[@]}")
if [[ "$bare_mode" == cutover ]]; then
  printf '{"mode":"cutover","bare_api_pid":%s}\n' "${bare_pids[0]}" > "$ROOT/raw/bare_process_selected.json"
  pkill -TERM -P "${bare_pids[0]}" || true
  kill -TERM "${bare_pids[0]}" || true
  if ss -ltn '( sport = :8001 )' | grep -q LISTEN; then
    pkill -KILL -P "${bare_pids[0]}" || true
    kill -KILL "${bare_pids[0]}" || true
  fi
  fuser -k -KILL 8001/tcp || true
else
  printf '{"mode":"cold_start","bare_api_pid":null}\n' > "$ROOT/raw/bare_process_selected.json"
fi
! ss -ltn '( sport = :8001 )' | grep -q LISTEN

env -u RAY_ADDRESS "$PY" -m ego_annotation.serving.cosmos3_cutover \
  --standalone-artifacts-dir "$STANDALONE" \
  --serve-config "$WORKSPACE/configs/cosmos3_serve.yaml" \
  --bare-cosmos3-stopped --execute > "$ROOT/raw/gate_result.json"
"$PY" -m ray.serve.scripts status -a http://127.0.0.1:26800 > "$ROOT/raw/serve_status_after_deploy.txt"

CTYPE=$("$PY" -c 'import json; print(json.load(open("'"$ROOT"'/representative_request_headers.json"))["Content-Type"])')
for attempt in initial post_client_exit; do
  curl -fsS --max-time 900 -X POST http://127.0.0.1:28006/cosmos3.reason \
    -H "Content-Type: $CTYPE" --data-binary "@$ROOT/representative_request.multipart" \
    -D "$ROOT/raw/${attempt}_response_headers.txt" -o "$ROOT/raw/${attempt}_response.multipart"
  ROOT="$ROOT" ATTEMPT="$attempt" "$PY" - <<'PY' > "$ROOT/raw/${attempt}_response.json"
import json, os
from pathlib import Path
from ego_annotation.serving.contracts import Cosmos3Response
from ego_annotation.serving.transport import parse_cosmos3_response
root = Path(os.environ['ROOT']); attempt = os.environ['ATTEMPT']
headers = (root / 'raw' / f'{attempt}_response_headers.txt').read_text().splitlines()
content_type = next(line.split(':', 1)[1].strip() for line in headers if line.lower().startswith('content-type:'))
response = Cosmos3Response.from_wire(parse_cosmos3_response((root / 'raw' / f'{attempt}_response.multipart').read_bytes(), content_type))
assert response.error is None, response.error
assert response.result is not None and response.result.text.strip(), response
assert response.result.trace.model_load_count == 1, response.result.trace.model_load_count
print(json.dumps(response.to_wire(), indent=2))
PY
done

if (( skip_benchmark )); then
  printf '{"status":"skipped_for_service_start"}\n' > "$ROOT/raw/open_loop_validation.json"
else
  [[ -s "$ROOT/benchmark_manifest.json" ]] || { echo "missing benchmark manifest: $ROOT/benchmark_manifest.json" >&2; false; }
  PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    "$WORKSPACE/benchmarks/ray_serve/cosmos3_open_loop.py" \
    --root "$ROOT" --manifest "$ROOT/benchmark_manifest.json" > "$ROOT/raw/open_loop_console.log"
  ROOT="$ROOT" "$PY" - <<'PY' > "$ROOT/raw/open_loop_validation.json"
import json, os
from pathlib import Path
summary = json.loads((Path(os.environ['ROOT']) / 'open_loop_summary.json').read_text())
assert summary and all(point['errors'] == 0 for point in summary), summary
assert all(point['payload_hashes'] == point['requests'] for point in summary), summary
print(json.dumps({'levels': len(summary), 'successes': sum(point['successes'] for point in summary)}))
PY
fi
"$PY" -m ray.serve.scripts status -a http://127.0.0.1:26800 > "$ROOT/raw/serve_status_after_benchmark.txt"
"$PY" -m ego_annotation.serving.cosmos3_resident_driver \
  --address 127.0.0.1:26801 --application cosmos3 \
  --state-file "$ROOT/raw/resident_driver.json" --check-only

printf '{"status":"candidate_accepted","endpoint":"http://127.0.0.1:28006/cosmos3.reason"}\n' > "$ROOT/raw/cutover_success.json"
rollback_armed=0
trap - ERR INT TERM
exec "$PY" -m ego_annotation.serving.cosmos3_resident_driver \
  --address 127.0.0.1:26801 --application cosmos3 \
  --state-file "$ROOT/raw/resident_driver.json"
