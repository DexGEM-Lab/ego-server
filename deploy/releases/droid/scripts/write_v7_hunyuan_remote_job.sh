#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_hunyuan_prior_outputs}
ENV_PY=${ENV_PY:-$REMOTE_ROOT/hunyuan3d_v3_env/bin/python}
REPO=${REPO:-$REMOTE_ROOT/Hunyuan3D-2}
RUNNER=${RUNNER:-$REMOTE_ROOT/remote_run_hunyuan3d_shape_v3.py}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/run_hunyuan_v7_representatives.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
cd "$REMOTE_ROOT"
cases=(
  "wild_rice_2539|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_crop_rgba.png|2539"
  "wild_rice_2545|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_crop_rgba.png|2545"
  "trash_0880|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_crop_rgba.png|880"
  "mop_0759|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_crop_rgba.png|759"
)
for raw_case in "\${cases[@]}"; do
  IFS='|' read -r case_name image_path seed <<<"\$raw_case"
  "$ENV_PY" "$RUNNER" \\
    --repo "$REPO" \\
    --mode single \\
    --image "\$image_path" \\
    --output-dir "$OUT_ROOT/\${case_name}_single" \\
    --model tencent/Hunyuan3D-2mini \\
    --subfolder hunyuan3d-dit-v2-mini-fast \\
    --steps 5 \\
    --octree-resolution 256 \\
    --num-chunks 12000 \\
    --seed "\$seed" \\
    --mesh-name mesh.glb
done
EOF
chmod +x "$OUT_ROOT/run_hunyuan_v7_representatives.sh"
cat > "$OUT_ROOT/wait_and_run_hunyuan_v7_representatives.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_hunyuan_v7_representatives.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_hunyuan_v7_representatives.sh"
printf '%s\n%s\n' "$OUT_ROOT/run_hunyuan_v7_representatives.sh" "$OUT_ROOT/wait_and_run_hunyuan_v7_representatives.sh"
