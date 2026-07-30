#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

# The API manager has no Ray head.  Its exact launcher window is its manifest.
stop_tmux_window ego_annotation manager
require_ports_closed 8092
lifecycle_log "API manager teardown complete"
