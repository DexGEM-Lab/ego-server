#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/user-home/yiwen/ego_annotation_remote/sam3_work
mkdir -p "$ROOT/repo/scripts" "$ROOT/data" "$ROOT/checkpoints" "$ROOT/third_party" "$ROOT/outputs"
cd "$ROOT"

if [ -n "${SAMWISE_SOURCE_ARCHIVE:-}" ]; then
  rm -rf third_party/SAMWISE
  mkdir -p third_party
  tar -xzf "$SAMWISE_SOURCE_ARCHIVE" -C third_party
  if [ -d third_party/SAMWISE-main ]; then
    mv third_party/SAMWISE-main third_party/SAMWISE
  fi
  if [ ! -f third_party/SAMWISE/inference_demo.py ]; then
    echo "extracted SAMWISE archive lacks inference_demo.py" >&2
    exit 1
  fi
elif [ ! -d third_party/SAMWISE/.git ]; then
  git clone https://github.com/ClaudiaCuttano/SAMWISE.git third_party/SAMWISE
else
  git -C third_party/SAMWISE pull --ff-only
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x .venv_samwise/bin/python ]; then
  uv venv --python 3.10 .venv_samwise
fi
source .venv_samwise/bin/activate
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
grep -v '^pyav$' third_party/SAMWISE/requirements.txt > "$ROOT/samwise_requirements_runtime.txt"
uv pip install -r "$ROOT/samwise_requirements_runtime.txt"
uv pip install gdown

cd "$ROOT/checkpoints"
if [ ! -f final_model_mevis.pth ]; then
  gdown '1Molt2up2bP41ekeczXWQU-LWTskKJOV2' -O final_model_mevis.pth
fi

python - <<'PY'
from pathlib import Path
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable in SAMWISE environment"
assert Path("final_model_mevis.pth").exists(), "SAMWISE checkpoint missing"
import sys
sys.path.insert(0, "/mnt/user-home/yiwen/ego_annotation_remote/sam3_work/third_party/SAMWISE")
sys.path.insert(0, "/mnt/user-home/yiwen/ego_annotation_remote/sam3_work/third_party/SAMWISE/models/sam2")
from models.samwise import build_samwise
from datasets.transform_utils import VideoEvalDataset
print("samwise_import_ok", build_samwise is not None, VideoEvalDataset is not None)
PY
