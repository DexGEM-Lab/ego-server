#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/sam3d_objects_work}
REPO=${REPO:-$WORK_ROOT/sam-3d-objects}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_sam3d_objects_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_sam3d_objects_mesh_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/sam3d_objects_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_sam3d_objects_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/sam3d_objects_git_head.txt"
python3 -m pip install --user virtualenv
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools wheel
"$ENV_PY" -m pip install 'huggingface-hub[cli]<1.0'
TAG=hf
mkdir -p checkpoints
if [[ ! -f checkpoints/\$TAG/pipeline.yaml ]]; then
  rm -rf checkpoints/\${TAG}-download
  "$ENV_DIR/bin/hf" download \\
    --repo-type model \\
    --local-dir checkpoints/\${TAG}-download \\
    --max-workers 1 \\
    facebook/sam-3d-objects
  rm -rf checkpoints/\$TAG
  mv checkpoints/\${TAG}-download/checkpoints checkpoints/\$TAG
  rm -rf checkpoints/\${TAG}-download
fi
"$ENV_PY" -m pip install -e '.[p3d]'
"$ENV_PY" -m pip install -e '.[inference]'
./patching/hydra
"$ENV_PY" - <<'PY'
import torch
import sam3d_objects
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
PY
test -f checkpoints/\$TAG/pipeline.yaml
find checkpoints/\$TAG -maxdepth 2 -type f | sort > "$OUT_ROOT/sam3d_objects_checkpoint_files.txt"
EOF
chmod +x "$OUT_ROOT/setup_sam3d_objects_v7.sh"

cat > "$OUT_ROOT/run_sam3d_objects_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
export LIDRA_SKIP_INIT=true
cd "$REPO"
if [[ ! -x "$ENV_PY" || ! -f checkpoints/hf/pipeline.yaml ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_sam3d_objects_v7.sh"
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --config "$REPO/checkpoints/hf/pipeline.yaml" \\
  --case "wild_rice_2539|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_image.png|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_mask.png|2539" \\
  --case "wild_rice_2545|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_image.png|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_mask.png|2545" \\
  --case "trash_0880|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_image.png|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_mask.png|880" \\
  --case "mop_0759|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_image.png|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_mask.png|759" \\
  --output-dir "$OUT_ROOT/generated_meshes"
EOF
chmod +x "$OUT_ROOT/run_sam3d_objects_v7.sh"

cat > "$OUT_ROOT/wait_and_run_sam3d_objects_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_sam3d_objects_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_sam3d_objects_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_sam3d_objects_v7.sh" \
  "$OUT_ROOT/run_sam3d_objects_v7.sh" \
  "$OUT_ROOT/wait_and_run_sam3d_objects_v7.sh"
