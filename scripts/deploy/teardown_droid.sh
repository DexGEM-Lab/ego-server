#!/usr/bin/env bash
# DROID importers consume IPC handles exported by their r0 owner. Stop every
# importer before either owner so an owner never disappears beneath an importer.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

PY=/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python
SESSION=ego_annotation

if [[ ${1:-} == "--dry-run" ]]; then
  lifecycle_log "DROID teardown dry run: would stop importers before owners and verify every target reaches its terminal state"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  lifecycle_die "usage: $0 [--dry-run]"
fi

stop_droid_group() {
  local temp_dir=$1 gcs_port=$2 dashboard_port=$3 http_port=$4 window=$5 label=$6
  stop_ray_head "$PY" "$temp_dir" "$gcs_port" "$dashboard_port" "$label"
  stop_tmux_window "$SESSION" "$window"
  # Terminal success is defined by the service no longer owning its ports or
  # process group. A vanished PID/window during teardown is therefore success.
  require_ports_closed "$gcs_port" "$dashboard_port" "$http_port"
  require_droid_group_gone "$temp_dir" "$gcs_port" "$http_port"
}

# Importers, newest first within each owner group.
stop_droid_group /tmp/ray-ego-serve-gpu7-2 30040 30044 28027 droid-gpu7-r2 "DROID GPU7 importer r2"
stop_droid_group /tmp/ray-ego-serve-gpu7-1 30020 30024 28017 droid-gpu7-r1 "DROID GPU7 importer r1"
stop_droid_group /tmp/ray-ego-serve-gpu2-2 26440 26444 28022 droid-gpu2-r2 "DROID GPU2 importer r2"
stop_droid_group /tmp/ray-ego-serve-gpu2-1 26420 26424 28012 droid-gpu2-r1 "DROID GPU2 importer r1"

# Owners last, after no importer can retain their CUDA IPC handles.
stop_droid_group /tmp/ray-ego-serve-gpu7-0 30000 30004 28007 droid-gpu7-r0 "DROID GPU7 owner"
stop_droid_group /tmp/ray-ego-serve-gpu2-0 26400 26404 29002 droid-gpu2-r0 "DROID GPU2 owner"
rm -f /tmp/droid-ipc-gpu2/handles.pkl /tmp/droid-ipc-gpu7/handles.pkl
require_gpu_compute_released 2
require_gpu_compute_released 7
lifecycle_log "DROID teardown complete: all ports/process groups are absent and GPU compute allocations are released"
