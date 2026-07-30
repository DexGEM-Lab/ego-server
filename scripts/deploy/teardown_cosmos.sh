#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

# Cosmos owns one fixed Ray manifest.  The named launcher windows are the only
# non-Ray processes it starts; no model-name or generic vLLM command matching is
# used because that could stop an unrelated inference server.
stop_ray_head /home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python /tmp/ray-ego-serve-cosmos3 26801 26800 "Cosmos3 GPU6"
stop_tmux_windows_with_prefix ego_annotation cosmos3-ray6-
require_ports_closed 28006
lifecycle_log "Cosmos3 teardown complete"
