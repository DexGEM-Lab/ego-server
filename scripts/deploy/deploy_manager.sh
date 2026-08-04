#!/usr/bin/env bash
# ==============================================================================
# A800 ego annotation service topology (run this script on A800)
#
# Python environments:
#   HaWoR + DROID : /home/zjh/miniconda3/envs/ray_serve_hawor/bin/python
#   Hands + WiLoR : /home/zjh/miniconda3/envs/ray_serve_hands/bin/python
#   UniDepth      : /home/zjh/miniconda3/envs/ray_serve_unidepth/bin/python
#   Dispatcher/API: /home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python
#   Cosmos3       : /home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python
#
# Weights/checkpoints:
#   DROID     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
#   UniDepth  : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected
#   Hands YOLO: /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/detector.pt
#   Hands SAM2: /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/sam2.1/sam2.1_hiera_large.pt
#   WiLoR     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/wilor_final.ckpt
#   HaWoR     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt
#   Infiller  : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt
#   Cosmos3   : nvidia/Cosmos3-Nano, cached under /home/ylang/.cache/huggingface
#
# Vendored service code (resolved from this script):
#   deploy/releases/runtime_current -> UniDepth, HaWoR, Infiller, Cosmos3
#   deploy/releases/hands_wilor    -> Hands, WiLoR
#   deploy/releases/droid          -> DROID retry-fix
#   deploy/releases/dispatcher     -> lane_dispatcher
#   deploy/releases/manager        -> API manager
# Third-party model repositories, environments, and checkpoints remain external
# A800 dependencies; see deploy/DEPENDENCIES.md for their exact identities.
#
# GPU / Ray head / Serve HTTP mapping:
#   GPU0 UniDepth       26000 / 29000 (dispatcher publishes it on 28000)
#   GPU1 Hands          27000 / 28001
#   GPU2 DROID owner    26400 / 29002
#        DROID importer 26420 / 28012
#        DROID importer 26440 / 28022
#   GPU3 HaWoR+Infiller 26600 / 28003
#   GPU4 WiLoR          27200 / 28004
#   GPU5 UniDepth       30800 / 28005 (moved from 26800 to avoid Cosmos3)
#   GPU6 Cosmos3        26801 / 28006 (Ray dashboard 26800)
#   GPU7 DROID owner    30000 / 28007
#        DROID importer 30020 / 28017
#        DROID importer 30040 / 28027
#   CPU  lane_dispatcher public ports 28002 (DROID), 28000 (UniDepth)
#   CPU  API manager    8092
#
# All Ray heads are independent. DROID BA concurrency is fixed at 1. DROID IPC
# owners are r0 (29002 and 28007); owners must start before and stop after importers.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

SESSION="ego_annotation"
EXPECTED_RELEASE_COMMIT="c8fc9bd203be8770f32c4f76ec65cd96cda02f82"
RUNTIME="$REPO_ROOT/deploy/releases/runtime_current"
export RUNTIME

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
require_executable() { [[ -x "$1" ]] || die "missing executable: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
assert_release() {
  [[ -s "$RUNTIME/RELEASE.json" ]] || die "missing $RUNTIME/RELEASE.json"
  grep -Fq "\"commit\":\"${EXPECTED_RELEASE_COMMIT}\"" "$RUNTIME/RELEASE.json" \
    || die "vendored release identity mismatch in $RUNTIME/RELEASE.json"
}
ensure_session() {
  if ! /usr/bin/tmux has-session -t "$SESSION" 2>/dev/null; then
    log "Creating /usr/bin/tmux session $SESSION"
    /usr/bin/tmux new-session -d -s "$SESSION" -n control -c "$RUNTIME" "exec tail -f /dev/null"
  fi
}
window_exists() {
  /usr/bin/tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -Fxq "$1"
}
port_is_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
require_free_ports() {
  local port
  for port in "$@"; do
    port_is_listening "$port" && die "TCP port $port is already listening; run the matching teardown first"
  done
  return 0
}
start_window() {
  local name=$1 cwd=$2 command=$3 bootstrap
  window_exists "$name" && die "tmux window $SESSION:$name already exists"
  # tmux windows inherit the server environment, not this launcher shell.  Pass
  # repository-relative paths explicitly so quoted command templates resolve them.
  printf -v bootstrap 'export RUNTIME=%q CODE=%q; ' "$RUNTIME" "${CODE:-}"
  log "Starting $SESSION:$name"
  /usr/bin/tmux new-window -d -t "$SESSION:" -n "$name" -c "$cwd" "exec bash -lc $(printf '%q' "${bootstrap}${command}")"
}
http_ready() {
  local port=$1 code
  code=$(curl -sS -o /dev/null --max-time 3 -w '%{http_code}' "http://127.0.0.1:${port}/-/healthz" 2>/dev/null || true)
  [[ $code =~ ^2[0-9][0-9]$ ]]
}
wait_for_serve() {
  local py=$1 dashboard=$2 http_port=$3 label=$4 timeout=${5:-600}
  local deadline=$((SECONDS + timeout)) status=""
  log "Waiting for $label on HTTP $http_port"
  while (( SECONDS < deadline )); do
    status=$($py -m ray.serve.scripts status -a "http://127.0.0.1:${dashboard}" 2>&1 || true)
    if grep -q 'RUNNING' <<<"$status" && http_ready "$http_port"; then
      log "$label is healthy"
      return 0
    fi
    sleep 3
  done
  printf '%s\n' "$status" >&2
  die "$label did not become healthy within ${timeout}s"
}

PY="/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python"
CODE="$REPO_ROOT/deploy/releases/manager"
export CODE
assert_release
require_executable "$PY"
require_dir "$CODE"
[[ -f $CODE/scripts/serve_v22_annotation_api.py ]] || die "missing API manager script"
ensure_session
manager_ready() { local code; code=$(curl -sS -o /dev/null --max-time 3 -w "%{http_code}" "http://127.0.0.1:8092/docs" 2>/dev/null || true); [[ $code =~ ^2[0-9][0-9]$ ]]; }
if window_exists manager; then manager_ready && { log "API manager already healthy"; exit 0; }; die "manager window exists but is unhealthy"; fi
http_ready 28000 || die "UniDepth dispatcher port 28000 is not healthy"
http_ready 28002 || die "DROID dispatcher port 28002 is not healthy"
require_free_ports 8092
cmd=$(cat <<'CMD'
set -euo pipefail
cd "$CODE"
export PYTHONPATH="${CODE}"
exec /home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python scripts/serve_v22_annotation_api.py \
  --host 127.0.0.1 --port 8092 --total-request-limit 128 --algorithm-inflight-multiplier 2
CMD
)
start_window manager "$CODE" "$cmd"
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do manager_ready && { log "API manager is healthy"; exit 0; }; sleep 1; done
die "API manager did not become healthy on 8092"
