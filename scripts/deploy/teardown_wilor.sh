#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

stop_ray_head /home/zjh/miniconda3/envs/ray_serve_hands/bin/python /tmp/ray-ego-serve-gpu4 27200 27204 "WiLoR GPU4"
stop_tmux_window ego_annotation wilor-gpu4
lifecycle_log "WiLoR teardown complete"
