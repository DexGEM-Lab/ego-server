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
# Permit a deployment to pin an immutable DROID runtime while retaining the
# repository vendored release as the normal default.
RUNTIME="${RUNTIME:-$REPO_ROOT/deploy/releases/droid}"
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
  if /usr/bin/tmux has-session -t "$SESSION" 2>/dev/null; then
    return 0
  fi
  log "Creating /usr/bin/tmux session $SESSION"
  /usr/bin/tmux new-session -d -s "$SESSION" -n control -c "$RUNTIME" "exec tail -f /dev/null"
}
window_exists() {
  local windows
  windows=$(/usr/bin/tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null || true)
  grep -Fxq "$1" <<<"$windows"
}
port_is_listening() { ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .; }
require_free_ports() {
  local port
  for port in "$@"; do
    if port_is_listening "$port"; then
      die "TCP port $port is already listening; run the matching teardown first"
    fi
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

PY="/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python"
wait_for_file() {
  local path=$1 label=$2 timeout=$3
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if [[ -s $path ]]; then
      log "$label ready"
      return 0
    fi
    sleep 2
  done
  die "$label not created within ${timeout}s"
}
assert_release
require_executable "$PY"
ensure_session
if window_exists droid-gpu2-r0 || window_exists droid-gpu7-r0; then
  wait_for_serve "$PY" 26404 29002 "DROID GPU2 owner" 30
  wait_for_serve "$PY" 26424 28012 "DROID GPU2 importer r1" 30
  wait_for_serve "$PY" 26444 28022 "DROID GPU2 importer r2" 30
  wait_for_serve "$PY" 30004 28007 "DROID GPU7 owner" 30
  wait_for_serve "$PY" 30024 28017 "DROID GPU7 importer r1" 30
  wait_for_serve "$PY" 30044 28027 "DROID GPU7 importer r2" 30
  log "all DROID replicas already healthy"
  exit 0
fi
require_free_ports 26400 26401 26402 26403 26404 26405 26406 26407 29002 26500 26501 26502 26503 26504 26505 26506 26507 26508 26509 26510 26511 26512 26513 26514 26515 26516 26517 26518 26519 26420 26421 26422 26423 26424 26425 26426 26427 28012 26520 26521 26522 26523 26524 26525 26526 26527 26528 26529 26530 26531 26532 26533 26534 26535 26536 26537 26538 26539 26440 26441 26442 26443 26444 26445 26446 26447 28022 26540 26541 26542 26543 26544 26545 26546 26547 26548 26549 26550 26551 26552 26553 26554 26555 26556 26557 26558 26559 30000 30001 30002 30003 30004 30005 30006 30007 28007 30100 30101 30102 30103 30104 30105 30106 30107 30108 30109 30110 30111 30112 30113 30114 30115 30116 30117 30118 30119 30020 30021 30022 30023 30024 30025 30026 30027 28017 30120 30121 30122 30123 30124 30125 30126 30127 30128 30129 30130 30131 30132 30133 30134 30135 30136 30137 30138 30139 30040 30041 30042 30043 30044 30045 30046 30047 28027 30140 30141 30142 30143 30144 30145 30146 30147 30148 30149 30150 30151 30152 30153 30154 30155 30156 30157 30158 30159
cmd_2_0=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=2 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu2/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu2-r0 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=26400 --dashboard-port=26404 \
  --object-manager-port=26401 --node-manager-port=26402 \
  --ray-client-server-port=26403 --dashboard-agent-listen-port=26405 \
  --dashboard-agent-grpc-port=26406 --metrics-export-port=26407 \
  --worker-port-list=26500,26501,26502,26503,26504,26505,26506,26507,26508,26509,26510,26511,26512,26513,26514,26515,26516,26517,26518,26519 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu2-0 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:26400 --port 29002
exec sleep infinity
CMD
)
cmd_2_1=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=2 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu2/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu2-r1 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=26420 --dashboard-port=26424 \
  --object-manager-port=26421 --node-manager-port=26422 \
  --ray-client-server-port=26423 --dashboard-agent-listen-port=26425 \
  --dashboard-agent-grpc-port=26426 --metrics-export-port=26427 \
  --worker-port-list=26520,26521,26522,26523,26524,26525,26526,26527,26528,26529,26530,26531,26532,26533,26534,26535,26536,26537,26538,26539 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu2-1 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:26420 --port 28012
exec sleep infinity
CMD
)
cmd_2_2=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=2 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu2/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu2-r2 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=26440 --dashboard-port=26444 \
  --object-manager-port=26441 --node-manager-port=26442 \
  --ray-client-server-port=26443 --dashboard-agent-listen-port=26445 \
  --dashboard-agent-grpc-port=26446 --metrics-export-port=26447 \
  --worker-port-list=26540,26541,26542,26543,26544,26545,26546,26547,26548,26549,26550,26551,26552,26553,26554,26555,26556,26557,26558,26559 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu2-2 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:26440 --port 28022
exec sleep infinity
CMD
)
cmd_7_0=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=7 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu7/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu7-r0 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=30000 --dashboard-port=30004 \
  --object-manager-port=30001 --node-manager-port=30002 \
  --ray-client-server-port=30003 --dashboard-agent-listen-port=30005 \
  --dashboard-agent-grpc-port=30006 --metrics-export-port=30007 \
  --worker-port-list=30100,30101,30102,30103,30104,30105,30106,30107,30108,30109,30110,30111,30112,30113,30114,30115,30116,30117,30118,30119 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu7-0 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:30000 --port 28007
exec sleep infinity
CMD
)
cmd_7_1=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=7 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu7/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu7-r1 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=30020 --dashboard-port=30024 \
  --object-manager-port=30021 --node-manager-port=30022 \
  --ray-client-server-port=30023 --dashboard-agent-listen-port=30025 \
  --dashboard-agent-grpc-port=30026 --metrics-export-port=30027 \
  --worker-port-list=30120,30121,30122,30123,30124,30125,30126,30127,30128,30129,30130,30131,30132,30133,30134,30135,30136,30137,30138,30139 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu7-1 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:30020 --port 28017
exec sleep infinity
CMD
)
cmd_7_2=$(cat <<'CMD'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7 PYTHONPATH="${RUNTIME}"
export EGO_DROID_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam
export EGO_DROID_WEIGHTS=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
export EGO_DROID_REVISION=droid-v1 EGO_DROID_GPU=7 EGO_DROID_DEVICE=cuda:0
export EGO_DROID_MAX_SESSIONS="${EGO_DROID_MAX_SESSIONS:-1}" EGO_DROID_MAX_BUFFER_SLOTS=256 EGO_DROID_CPU_OFFLOAD=0
export EGO_DROID_MAX_CONCURRENT_BA=1
export EGO_DROID_IPC_HANDLE_FILE=/tmp/droid-ipc-gpu7/handles.pkl
export EGO_DROID_REPLICA_ID=droid-gpu7-r2 RAY_DEDUP_LOGS=0
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m ray.scripts.scripts start --head \
  --port=30040 --dashboard-port=30044 \
  --object-manager-port=30041 --node-manager-port=30042 \
  --ray-client-server-port=30043 --dashboard-agent-listen-port=30045 \
  --dashboard-agent-grpc-port=30046 --metrics-export-port=30047 \
  --worker-port-list=30140,30141,30142,30143,30144,30145,30146,30147,30148,30149,30150,30151,30152,30153,30154,30155,30156,30157,30158,30159 --num-gpus=1 --num-cpus=4 \
  --temp-dir=/tmp/ray-ego-serve-gpu7-2 --include-dashboard=true
"/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python" -m scripts.serve_group_driver --gpu-id 2 --address 127.0.0.1:30040 --port 28027
exec sleep infinity
CMD
)
rm -f /tmp/droid-ipc-gpu2/handles.pkl
mkdir -p /tmp/droid-ipc-gpu2
start_window droid-gpu2-r0 "$RUNTIME" "$cmd_2_0"
wait_for_serve "$PY" 26404 29002 "DROID GPU2 owner" 900
wait_for_file /tmp/droid-ipc-gpu2/handles.pkl "DROID GPU2 IPC handles" 180
rm -f /tmp/droid-ipc-gpu7/handles.pkl
mkdir -p /tmp/droid-ipc-gpu7
start_window droid-gpu7-r0 "$RUNTIME" "$cmd_7_0"
wait_for_serve "$PY" 30004 28007 "DROID GPU7 owner" 900
wait_for_file /tmp/droid-ipc-gpu7/handles.pkl "DROID GPU7 IPC handles" 180
start_window droid-gpu2-r1 "$RUNTIME" "$cmd_2_1"
wait_for_serve "$PY" 26424 28012 "DROID GPU2 importer r1" 900
start_window droid-gpu2-r2 "$RUNTIME" "$cmd_2_2"
wait_for_serve "$PY" 26444 28022 "DROID GPU2 importer r2" 900
start_window droid-gpu7-r1 "$RUNTIME" "$cmd_7_1"
wait_for_serve "$PY" 30024 28017 "DROID GPU7 importer r1" 900
start_window droid-gpu7-r2 "$RUNTIME" "$cmd_7_2"
wait_for_serve "$PY" 30044 28027 "DROID GPU7 importer r2" 900
log "DROID deployment complete; BA=1 on all six replicas"
