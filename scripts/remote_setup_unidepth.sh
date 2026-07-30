#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_UNIDEPTH_ROOT:-/mnt/truenas-user-home/yiwen/a800_migrated_home/ego_annotation_remote/unidepth_work}"
REPO="$ROOT/UniDepth"
MODEL_ENV="${EGO_MODEL_ENV:-/mnt/user-home/yiwen/ego_annotation_remote/model_envs/unidepth_sam2}"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"

mkdir -p "$ROOT"/{logs,outputs,data}
cd "$ROOT"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [ -d "$REPO/.git" ]; then
  git config --global --add safe.directory "$REPO" || true
  REPO_REALPATH="$(readlink -f "$REPO" 2>/dev/null || printf '%s' "$REPO")"
  git config --global --add safe.directory "$REPO_REALPATH" || true
fi

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/lpiccinelli-eth/UniDepth.git "$REPO"
else
  git -C "$REPO" pull --ff-only
fi

cd "$REPO"
mkdir -p "$(dirname "$MODEL_ENV")"
if [ -d "$MODEL_ENV" ] && [ ! -x "$MODEL_ENV/bin/python" ]; then
  rm -rf "$MODEL_ENV"
fi
if [ ! -x "$MODEL_ENV/bin/python" ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$MODEL_ENV"
fi
if [ ! -x "$MODEL_ENV/bin/python" ]; then
  echo "UniDepth/SAM2 model env python is not executable after venv creation: $MODEL_ENV/bin/python" >&2
  exit 1
fi

"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" --upgrade pip setuptools wheel
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python "$MODEL_ENV/bin/python" opencv-python pillow numpy scipy tqdm
"$MODEL_ENV/bin/python" -m pip install hydra-core omegaconf iopath

"$MODEL_ENV/bin/python" - <<'PY'
import importlib
import torch

for name in ["torch", "cv2", "PIL", "numpy", "unidepth", "hydra", "omegaconf", "iopath", "tqdm"]:
    importlib.import_module(name)
if not torch.cuda.is_available():
    raise RuntimeError("UniDepth setup requires CUDA")
print("unidepth_setup_ok", "torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
