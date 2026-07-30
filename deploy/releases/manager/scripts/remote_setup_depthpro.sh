#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_DEPTHPRO_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/depthpro_work}"
REPO="$ROOT/ml-depth-pro"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$ROOT"/{logs,outputs,data}
cd "$ROOT"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/apple/ml-depth-pro.git "$REPO"
else
  git -C "$REPO" pull --ff-only
fi

cd "$REPO"
if [ ! -x .venv/bin/python ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" .venv
fi

"$UV_BIN" pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python .venv/bin/python -e .
"$UV_BIN" pip install --python .venv/bin/python opencv-python pillow numpy scipy huggingface_hub

mkdir -p checkpoints
if [ ! -s checkpoints/depth_pro.pt ] || [ "$(stat -c%s checkpoints/depth_pro.pt)" -lt 1000000000 ]; then
  rm -f checkpoints/depth_pro.pt
  wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
fi

.venv/bin/python - <<'PY'
from pathlib import Path
import torch
import depth_pro

checkpoint = Path("checkpoints/depth_pro.pt")
assert checkpoint.exists() and checkpoint.stat().st_size > 1024 * 1024, checkpoint
model, transform = depth_pro.create_model_and_transforms()
model.eval().cuda()
print("depthpro_setup_ok", "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
