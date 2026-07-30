#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

"${DEPLOY_DIR}/teardown_droid.sh" --dry-run
bash --noprofile --norc -ceu '
  source "$1/lifecycle_common.bash"
  stop_ray_head /does/not/exist /tmp/droid-teardown-absent 59990 59991 absent-ray-head
  stop_tmux_window droid-teardown-absent absent-window
  require_ports_closed 59990 59991 59992
  require_droid_group_gone /tmp/droid-teardown-absent 59990 59992
' bash "${DEPLOY_DIR}"

# A TERM-ignoring child forces the SIGKILL branch.  terminate_manifest_pids
# must wait for its /proc identity to disappear instead of reporting a false
# survivor while the kernel completes SIGKILL teardown.
bash --noprofile --norc -ceu '
  source "$1/lifecycle_common.bash"
  (trap "" TERM; exec sleep 60) &
  pid=$!
  terminate_manifest_pids droid-teardown-sigkill-grace 0 "$pid"
  test ! -e "/proc/${pid}"
' bash "${DEPLOY_DIR}"
