#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

SESSION=ego_annotation
PY=/home/zjh/miniconda3/envs/ray_serve_unidepth/bin/python

stop_ray_head "$PY" /tmp/ray-ego-serve-gpu5 30800 30804 "UniDepth GPU5"
stop_tmux_window "$SESSION" unidepth-gpu5
stop_ray_head "$PY" /tmp/ray-ego-serve-gpu0 26000 26004 "UniDepth GPU0"
stop_tmux_window "$SESSION" unidepth-gpu0
lifecycle_log "UniDepth teardown complete"
