#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/hunyuan21_work}
REPO=${REPO:-$WORK_ROOT/Hunyuan3D-2.1}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_hunyuan21_prior_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_hunyuan21_shape_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/hunyuan21_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
SETUP_COMPLETE=${SETUP_COMPLETE:-$OUT_ROOT/setup_complete.marker}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_hunyuan21_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/hunyuan21_git_head.txt"
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
rm -f "$SETUP_COMPLETE"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel
"$ENV_PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install \
  accelerate==1.1.1 \
  diffusers==0.30.0 \
  einops==0.8.0 \
  huggingface-hub==0.30.2 \
  numpy==1.24.4 \
  omegaconf==2.3.0 \
  opencv-python==4.10.0.84 \
  pillow \
  pymeshlab==2022.2.post3 \
  safetensors==0.4.4 \
  scikit-image==0.24.0 \
  scipy==1.14.1 \
  timm \
  tqdm==4.66.5 \
  transformers==4.46.0 \
  trimesh==4.4.7
"$ENV_PY" - <<'PY'
import sys
from pathlib import Path
import torch, trimesh
repo = Path("$REPO")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "hy3dshape"))
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("Hunyuan21", Hunyuan3DDiTFlowMatchingPipeline.__name__, "trimesh", trimesh.__version__)
PY
date '+%Y-%m-%d %H:%M:%S setup complete' > "$SETUP_COMPLETE"
EOF
chmod +x "$OUT_ROOT/setup_hunyuan21_v7.sh"

cat > "$OUT_ROOT/run_hunyuan21_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
cd "$REPO"
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_hunyuan21_v7.sh"
fi
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  echo "Hunyuan3D 2.1 setup did not produce $SETUP_COMPLETE" >&2
  exit 1
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --case "wild_rice_2539|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_crop_rgba.png|2539" \\
  --case "wild_rice_2545|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_crop_rgba.png|2545" \\
  --case "trash_0880|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_crop_rgba.png|880" \\
  --case "mop_0759|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_crop_rgba.png|759" \\
  --output-dir "$OUT_ROOT/generated_meshes" \\
  --steps 30 \\
  --octree-resolution 256 \\
  --num-chunks 200000
EOF
chmod +x "$OUT_ROOT/run_hunyuan21_v7.sh"

cat > "$OUT_ROOT/wait_and_run_hunyuan21_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_hunyuan21_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_hunyuan21_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_hunyuan21_v7.sh" \
  "$OUT_ROOT/run_hunyuan21_v7.sh" \
  "$OUT_ROOT/wait_and_run_hunyuan21_v7.sh"
