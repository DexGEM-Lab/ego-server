#!/usr/bin/env bash
# Shared, deliberately narrow lifecycle primitives for the independent Ray heads.
# A Ray head is identified by the pair (fixed GCS port, fixed temp root).  This is
# the process manifest: a service can never tear down another head merely because
# it happens to share an interpreter or a substring in its command line.

lifecycle_log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
lifecycle_die() { lifecycle_log "ERROR: $*" >&2; exit 1; }

read_process_cmdline() {
  local pid=$1
  [[ -r /proc/${pid}/cmdline ]] || return 0
  tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || :
}

process_is_live() {
  local pid=$1 state
  [[ -r /proc/${pid}/status ]] || return 1
  state=$(awk '/^State:/{print $2}' "/proc/${pid}/status" 2>/dev/null || true)
  [[ $state != Z && -n $state ]]
}

is_current_process_ancestor() {
  local candidate=$1 current=$$ parent
  while [[ $current =~ ^[0-9]+$ && $current -gt 1 ]]; do
    [[ $current == "$candidate" ]] && return 0
    parent=$(awk '{print $4}' "/proc/${current}/stat" 2>/dev/null || true)
    [[ $parent =~ ^[0-9]+$ && $parent != "$current" ]] || break
    current=$parent
  done
  return 1
}

port_is_listening() {
  ss -H -ltn "sport = :$1" 2>/dev/null | grep -q .
}

require_free_ports() {
  local port
  for port in "$@"; do
    if port_is_listening "$port"; then
      lifecycle_die "TCP port ${port} is already listening; run the matching teardown first"
    fi
  done
  return 0
}

healthz_2xx() {
  local port=$1 code
  code=$(curl -sS -o /dev/null --max-time 3 -w '%{http_code}' "http://127.0.0.1:${port}/-/healthz" 2>/dev/null || :)
  [[ $code =~ ^2[0-9][0-9]$ ]]
}

ray_head_fingerprint_present() {
  local temp_dir=$1 gcs_port=$2 proc pid cmd count=0
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    # An invoking shell can contain target literals in its command text. An
    # ancestor of this teardown cannot be a residual service process.
    if is_current_process_ancestor "$pid"; then
      continue
    fi
    cmd=$(read_process_cmdline "$pid")
    if [[ $cmd == *"--gcs_server_port=${gcs_port}"* && $cmd == *"${temp_dir}/"* ]]; then
      count=$((count + 1))
    fi
  done
  if (( count == 1 )); then
    return 0
  fi
  if (( count == 0 )); then
    return 1
  fi
  lifecycle_die "Ray manifest ${temp_dir} / GCS ${gcs_port} matched ${count} heads"
}

ray_head_manifest_pids() {
  local temp_dir=$1 proc pid cmd
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    cmd=$(read_process_cmdline "$pid")
    if [[ $cmd == *"${temp_dir}/"* ]]; then
      printf '%s\n' "$pid"
    fi
  done
}

process_starttime() {
  local pid=$1
  [[ -r /proc/${pid}/stat ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null
}

# A numeric PID is not a durable identity: it can be reused after a target exits.
# Keep the start time captured from the verified manifest and never signal or
# report a replacement process as a DROID residual.
process_matches_starttime() {
  local pid=$1 expected_starttime=$2 actual_starttime
  [[ -n $expected_starttime ]] || return 1
  process_is_live "$pid" || return 1
  actual_starttime=$(process_starttime "$pid" 2>/dev/null || true)
  [[ $actual_starttime == "$expected_starttime" ]]
}

signal_verified_pid() {
  local signal=$1 pid=$2 label=$3 expected_starttime=${4:-}
  if [[ -n $expected_starttime ]] && ! process_matches_starttime "$pid" "$expected_starttime"; then
    lifecycle_log "${label}: PID ${pid} exited or was reused before SIG${signal} delivery"
    return 0
  fi
  if kill "-${signal}" "$pid" 2>/dev/null; then
    return 0
  fi
  if [[ -n $expected_starttime ]] && ! process_matches_starttime "$pid" "$expected_starttime"; then
    lifecycle_log "${label}: PID ${pid} exited or was reused during SIG${signal} delivery"
    return 0
  fi
  if process_is_live "$pid"; then
    lifecycle_die "${label}: failed to deliver SIG${signal} to manifest PID ${pid}"
  fi
  lifecycle_log "${label}: PID ${pid} exited before SIG${signal} delivery"
  return 0
}

terminate_manifest_pids() {
  local label=$1 timeout_s=$2
  shift 2
  local -a requested=("$@") pids=() alive=()
  local pid deadline expected_starttime
  declare -A starttimes=()
  if (( ${#requested[@]} == 0 )); then
    lifecycle_log "${label}: no manifest processes remain"
    return 0
  fi
  for pid in "${requested[@]}"; do
    expected_starttime=$(process_starttime "$pid" 2>/dev/null || true)
    if [[ -z $expected_starttime ]]; then
      # The exact manifest was sampled before this process exited. Its absence
      # before identity capture is already the requested terminal state, not a
      # teardown error or a license to signal a reused PID.
      lifecycle_log "${label}: manifest PID ${pid} exited before identity capture"
      continue
    fi
    starttimes["$pid"]=$expected_starttime
    pids+=("$pid")
  done
  if (( ${#pids[@]} == 0 )); then
    lifecycle_log "${label}: all sampled manifest processes exited before signaling"
    return 0
  fi
  lifecycle_log "${label}: SIGTERM exact manifest PIDs ${pids[*]}"
  for pid in "${pids[@]}"; do signal_verified_pid TERM "$pid" "$label" "${starttimes[$pid]}"; done
  deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    alive=()
    for pid in "${pids[@]}"; do
      if process_matches_starttime "$pid" "${starttimes[$pid]}"; then
        alive+=("$pid")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      return 0
    fi
    sleep 1
  done
  alive=()
  for pid in "${pids[@]}"; do
    if process_matches_starttime "$pid" "${starttimes[$pid]}"; then
      alive+=("$pid")
    fi
  done
  if (( ${#alive[@]} == 0 )); then
    return 0
  fi
  lifecycle_log "${label}: SIGKILL exact manifest PIDs ${alive[*]}"
  for pid in "${alive[@]}"; do signal_verified_pid KILL "$pid" "$label" "${starttimes[$pid]}"; done

  # SIGKILL is asynchronous.  A direct liveness assertion here races the
  # kernel's final process teardown and turns a successful scoped shutdown into
  # a false failure.  Poll the identities captured above for a short bounded
  # grace period; a vanished /proc entry (or PID reuse) is terminal success.
  deadline=$((SECONDS + 5))
  while ((SECONDS < deadline)); do
    alive=()
    for pid in "${pids[@]}"; do
      if process_matches_starttime "$pid" "${starttimes[$pid]}"; then
        alive+=("$pid")
      fi
    done
    if (( ${#alive[@]} == 0 )); then
      lifecycle_log "${label}: SIGKILL targets exited within grace window"
      return 0
    fi
    sleep 0.25
  done
  alive=()
  for pid in "${pids[@]}"; do
    if process_matches_starttime "$pid" "${starttimes[$pid]}"; then
      alive+=("$pid")
    fi
  done
  if (( ${#alive[@]} > 0 )); then
    lifecycle_die "${label}: manifest PIDs survived SIGKILL grace window: ${alive[*]}"
  fi
}

stop_ray_head() {
  local py=$1 temp_dir=$2 gcs_port=$3 dashboard_port=$4 label=$5
  local -a pids=()
  if ! ray_head_fingerprint_present "$temp_dir" "$gcs_port"; then
    lifecycle_log "${label}: exact Ray manifest absent"
    return 0
  fi
  mapfile -t pids < <(ray_head_manifest_pids "$temp_dir")
  if (( ${#pids[@]} == 0 )); then
    lifecycle_die "${label}: fingerprint exists but manifest has no processes"
  fi
  lifecycle_log "${label}: verified (temp=${temp_dir}, gcs=${gcs_port}, dashboard=${dashboard_port})"
  if [[ -x $py ]]; then
    if ! timeout 30 "$py" -m ray.serve.scripts shutdown -a "http://127.0.0.1:${dashboard_port}" -y >/dev/null; then
      lifecycle_log "${label}: Serve shutdown endpoint unavailable; stopping its verified head"
    fi
  else
    lifecycle_log "${label}: interpreter unavailable; stopping its verified head"
  fi
  # terminate_manifest_pids waits through the bounded SIGKILL grace window and
  # verifies the original PID identities, including the PID-reuse case.
  terminate_manifest_pids "${label}" 25 "${pids[@]}"
}

stop_tmux_window() {
  local session=$1 name=$2 windows
  if ! /usr/bin/tmux has-session -t "$session" 2>/dev/null; then
    lifecycle_log "${session}:${name}: session absent"
    return 0
  fi
  windows=$(/usr/bin/tmux list-windows -t "$session" -F '#{window_name}' 2>/dev/null || true)
  if ! grep -Fxq "$name" <<<"$windows"; then
    lifecycle_log "${session}:${name}: window absent"
    return 0
  fi
  lifecycle_log "closing exact launcher window ${session}:${name}"
  if /usr/bin/tmux kill-window -t "${session}:${name}" 2>/dev/null; then
    return 0
  fi
  # The window can exit between the inventory and kill. Its terminal absence is
  # success; only a still-present launcher is a teardown failure.
  windows=$(/usr/bin/tmux list-windows -t "$session" -F '#{window_name}' 2>/dev/null || true)
  if grep -Fxq "$name" <<<"$windows"; then
    lifecycle_die "${session}:${name}: failed to close launcher window"
  fi
  lifecycle_log "${session}:${name}: window exited before close"
  return 0
}

stop_tmux_windows_with_prefix() {
  local session=$1 prefix=$2 window
  if ! /usr/bin/tmux has-session -t "$session" 2>/dev/null; then
    return 0
  fi
  while IFS= read -r window; do
    if [[ -n $window ]]; then
      stop_tmux_window "$session" "$window"
    fi
  done < <(/usr/bin/tmux list-windows -t "$session" -F '#{window_name}' | grep -E "^${prefix}[0-9]+$" || :)
}

require_ports_closed() {
  local deadline=$((SECONDS + 5)) port
  local -a open=()
  while :; do
    open=()
    for port in "$@"; do
      port_is_listening "$port" && open+=("$port")
    done
    if (( ${#open[@]} == 0 )); then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      lifecycle_die "expected port(s) ${open[*]} to close after scoped teardown grace"
    fi
    sleep 0.25
  done
}

require_gpu_compute_released() {
  local gpu=$1 output
  output=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n ${output//[[:space:]]/} ]]; then
    lifecycle_die "GPU ${gpu} still has compute allocation(s): ${output}"
  fi
  lifecycle_log "GPU ${gpu}: no compute allocation remains"
}

require_droid_group_gone() {
  local temp_dir=$1 gcs_port=$2 http_port=$3 proc pid cmd
  local -a survivors=()
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    # Shell wrappers, pipes, and logging commands legitimately contain the
    # target path. A residual must be a real Ray process rooted in this temp
    # directory, or the exact DROID driver attached to this head/HTTP port.
    if is_current_process_ancestor "$pid"; then
      continue
    fi
    cmd=$(read_process_cmdline "$pid")
    if [[ $cmd == *"${temp_dir}"* ]] && [[ $cmd == *"/ray/"* || $cmd == *"raylet"* || $cmd == *"gcs_server"* ]]; then
      survivors+=("${pid}:${cmd}")
    elif [[ $cmd == *"serve_group_driver"* ]] \
      && [[ $cmd == *"--address 127.0.0.1:${gcs_port}"* ]] \
      && [[ $cmd == *"--port ${http_port}"* ]]; then
      survivors+=("${pid}:${cmd}")
    fi
  done
  if (( ${#survivors[@]} > 0 )); then
    lifecycle_die "DROID group temp=${temp_dir} gcs=${gcs_port} http=${http_port} has residual processes: ${survivors[*]}"
  fi
}
