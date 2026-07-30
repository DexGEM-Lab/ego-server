#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/spar3d_work}
REPO=${REPO:-$WORK_ROOT/stable-point-aware-3d}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_spar3d_prior_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_spar3d_shape_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/spar3d_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_spar3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/Stability-AI/stable-point-aware-3d.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/spar3d_git_head.txt"
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel
"$ENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install --no-build-isolation -r requirements.txt
"$ENV_PY" -m pip install trimesh pillow pygltflib
"$ENV_PY" - <<'PY'
from pathlib import Path

utils_path = Path("spar3d/utils.py")
text = utils_path.read_text(encoding="utf-8")
import_line = "from transparent_background import Remover\n\n"
annotation = "bg_remover: Remover = None,"
eager_call = (
    '    if do_remove:\n        image = bg_remover.process(\n'
    '            image.convert("RGB"), **transparent_background_kwargs\n        )\n'
)
lazy_call = (
    '    if do_remove:\n        if bg_remover is None:\n'
    '            from transparent_background import Remover\n\n'
    '            bg_remover = Remover(device=get_device())\n'
    '        image = bg_remover.process(\n'
    '            image.convert("RGB"), **transparent_background_kwargs\n        )\n'
)
missing = [
    name
    for name, source, patched in (
        ("transparent_background import", import_line, "from typing import Any\n\n"),
        ("Remover annotation", annotation, "bg_remover: Any = None,"),
        ("background removal call", eager_call, lazy_call),
    )
    if source not in text and patched not in text
]
if missing:
    raise RuntimeError(f"SPAR3D utils patch source mismatch: {missing}")
if import_line in text:
    text = text.replace(import_line, "from typing import Any\n\n", 1)
if annotation in text:
    text = text.replace(annotation, "bg_remover: Any = None,", 1)
if eager_call in text:
    text = text.replace(eager_call, lazy_call, 1)
utils_path.write_text(text, encoding="utf-8")
PY
"$ENV_PY" - <<'PY'
import torch, trimesh
from spar3d.system import SPAR3D
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("SPAR3D", SPAR3D.__name__, "trimesh", trimesh.__version__)
PY
EOF
chmod +x "$OUT_ROOT/setup_spar3d_v7.sh"

cat > "$OUT_ROOT/run_spar3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
cd "$REPO"
if [[ ! -x "$ENV_PY" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_spar3d_v7.sh"
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --case "wild_rice_2539|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_crop_rgba.png" \\
  --case "wild_rice_2545|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_crop_rgba.png" \\
  --case "trash_0880|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_crop_rgba.png" \\
  --case "mop_0759|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_crop_rgba.png" \\
  --output-dir "$OUT_ROOT/generated_meshes" \\
  --low-vram-mode
EOF
chmod +x "$OUT_ROOT/run_spar3d_v7.sh"

cat > "$OUT_ROOT/wait_and_run_spar3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_spar3d_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_spar3d_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_spar3d_v7.sh" \
  "$OUT_ROOT/run_spar3d_v7.sh" \
  "$OUT_ROOT/wait_and_run_spar3d_v7.sh"
