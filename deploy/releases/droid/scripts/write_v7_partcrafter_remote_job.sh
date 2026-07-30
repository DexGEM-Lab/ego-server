#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/partcrafter_work}
REPO=${REPO:-$WORK_ROOT/PartCrafter}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_partcrafter_prior_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_partcrafter_shape_v7.py}
CASE_PLAN=${CASE_PLAN:-$OUT_ROOT/partcrafter_case_plan_v7.args}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/partcrafter_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
SETUP_COMPLETE=${SETUP_COMPLETE:-$OUT_ROOT/setup_complete.marker}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_partcrafter_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/wgsxm/PartCrafter.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/partcrafter_git_head.txt"
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
rm -f "$SETUP_COMPLETE"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel ninja
"$ENV_PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
"$ENV_PY" -m pip install -r settings/requirements.txt
"$ENV_PY" -m pip install "transformers==4.49.0" "huggingface_hub<1.0"
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
"$ENV_PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="wgsxm/PartCrafter", local_dir="pretrained_weights/PartCrafter", max_workers=1)
print("partcrafter_model_cache_ready")
PY
"$ENV_PY" - <<'PY'
import sys
from pathlib import Path
import torch, trimesh
repo = Path("$REPO")
sys.path.insert(0, str(repo))
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("PartCrafter", PartCrafterPipeline.__name__, "trimesh", trimesh.__version__)
PY
date '+%Y-%m-%d %H:%M:%S setup complete' > "$SETUP_COMPLETE"
EOF
chmod +x "$OUT_ROOT/setup_partcrafter_v7.sh"

cat > "$OUT_ROOT/run_partcrafter_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=120
cd "$REPO"
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_partcrafter_v7.sh"
fi
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  echo "PartCrafter setup did not produce $SETUP_COMPLETE" >&2
  exit 1
fi
if [[ ! -s "$CASE_PLAN" ]]; then
  echo "PartCrafter case plan is missing: $CASE_PLAN" >&2
  exit 1
fi
case_args=()
while IFS= read -r raw_case; do
  [[ -n "\$raw_case" ]] && case_args+=(--case "\$raw_case")
done < "$CASE_PLAN"
if [[ "\${#case_args[@]}" -eq 0 ]]; then
  echo "PartCrafter case plan contains no cases: $CASE_PLAN" >&2
  exit 1
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  "\${case_args[@]}" \\
  --output-dir "$OUT_ROOT/generated_meshes" \\
  --num-inference-steps 50 \\
  --guidance-scale 7.0
EOF
chmod +x "$OUT_ROOT/run_partcrafter_v7.sh"

cat > "$OUT_ROOT/wait_and_run_partcrafter_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_partcrafter_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_partcrafter_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_partcrafter_v7.sh" \
  "$OUT_ROOT/run_partcrafter_v7.sh" \
  "$OUT_ROOT/wait_and_run_partcrafter_v7.sh"
