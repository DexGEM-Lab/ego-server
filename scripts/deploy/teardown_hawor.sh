#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

stop_ray_head /home/zjh/miniconda3/envs/ray_serve_hawor/bin/python /tmp/ray-ego-serve-gpu3 26600 26604 "HaWoR GPU3"
stop_tmux_window ego_annotation hawor-gpu3
lifecycle_log "HaWoR teardown complete"
