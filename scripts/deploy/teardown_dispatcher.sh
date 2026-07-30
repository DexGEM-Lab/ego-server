#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lifecycle_common.bash
source "${SCRIPT_DIR}/lifecycle_common.bash"

# The dispatcher is owned by this exact tmux window; do not search arbitrary
# command lines for a Python module name.
stop_tmux_window ego_annotation dispatcher
require_ports_closed 28000 28002
lifecycle_log "dispatcher teardown complete"
