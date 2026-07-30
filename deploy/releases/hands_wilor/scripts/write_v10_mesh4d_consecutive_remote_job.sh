#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/mesh4d_work}
REPO=${REPO:-$WORK_ROOT/Mesh4D}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v10_mesh4d_consecutive_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_mesh4d_sequence_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/mesh4d_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
SETUP_COMPLETE=${SETUP_COMPLETE:-/mnt/user-home/yiwen/ego_annotation_remote/v7_mesh4d_outputs/setup_complete.marker}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v10_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v10_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/run_mesh4d_consecutive_v10.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  echo "Mesh4D setup marker missing: $SETUP_COMPLETE" >&2
  exit 1
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --sequence-json "$REMOTE_ROOT/v10_mesh4d_inputs/wild_rice_2538_2543/qc_mesh4d_rgba_sequence_v7.json" \\
  --sequence-dir "$REMOTE_ROOT/v10_mesh4d_inputs/wild_rice_2538_2543/DATA/ego_v10/wild_rice_2538_2543" \\
  --output-dir "$OUT_ROOT/generated/wild_rice_2538_2543" \\
  --denoiser-ckpt "$REPO/ckpt/denoiser.ckpt" \\
  --mesh4d-steps 50 \\
  --guidance-scale 5.0
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --sequence-json "$REMOTE_ROOT/v10_mesh4d_inputs/mop_0760_0765/qc_mesh4d_rgba_sequence_v7.json" \\
  --sequence-dir "$REMOTE_ROOT/v10_mesh4d_inputs/mop_0760_0765/DATA/ego_v10/mop_0760_0765" \\
  --output-dir "$OUT_ROOT/generated/mop_0760_0765" \\
  --denoiser-ckpt "$REPO/ckpt/denoiser.ckpt" \\
  --mesh4d-steps 50 \\
  --guidance-scale 5.0
EOF
chmod +x "$OUT_ROOT/run_mesh4d_consecutive_v10.sh"

cat > "$OUT_ROOT/wait_and_run_mesh4d_consecutive_v10.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_mesh4d_consecutive_v10.sh"
MAX_USED_MB="\${MAX_USED_MB:-$MAX_USED_MB}"
POLL_SECONDS="\${POLL_SECONDS:-$POLL_SECONDS}"
GPU_SELECT_LOCK="\${GPU_SELECT_LOCK:-$GPU_SELECT_LOCK}"
GPU_LOCK_DIR="\${GPU_LOCK_DIR:-$GPU_LOCK_DIR}"
mkdir -p "\$GPU_LOCK_DIR"
while true; do
  GPU_ID=""
  exec 9>"\$GPU_SELECT_LOCK"
  flock -x 9
  while IFS=, read -r gpu_idx used_mb; do
    gpu_idx="\${gpu_idx//[[:space:]]/}"
    used_mb="\${used_mb//[[:space:]]/}"
    if [[ -n "\$gpu_idx" && -n "\$used_mb" && "\$used_mb" -le "\$MAX_USED_MB" ]]; then
      exec 8>"\$GPU_LOCK_DIR/gpu_\${gpu_idx}.lock"
      if flock -n 8; then
        GPU_ID="\$gpu_idx"
        break
      fi
      exec 8>&-
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [[ -n "\$GPU_ID" ]]; then
    export GPU_ID
    flock -u 9
    exec 9>&-
    date '+%Y-%m-%d %H:%M:%S selected GPU '"\$GPU_ID"
    exec bash "\$RUN_SCRIPT"
  fi
  flock -u 9
  exec 9>&-
  date '+%Y-%m-%d %H:%M:%S no GPU below memory threshold; sleeping'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "\$POLL_SECONDS"
done
EOF
chmod +x "$OUT_ROOT/wait_and_run_mesh4d_consecutive_v10.sh"

printf '%s\n%s\n' \
  "$OUT_ROOT/run_mesh4d_consecutive_v10.sh" \
  "$OUT_ROOT/wait_and_run_mesh4d_consecutive_v10.sh"
