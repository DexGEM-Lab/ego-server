#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/handdgp_work}"
HANDDGP_ROOT="$ROOT/HandDGP"
HANDMESH_ROOT="$HANDDGP_ROOT/third_party/HandMesh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"

mkdir -p "$ROOT"/{logs,outputs,data}
cd "$ROOT"

if [ ! -d "$HANDDGP_ROOT/.git" ]; then
  if [ -f "$ROOT/handdgp_src.tgz" ]; then
    tar -xzf "$ROOT/handdgp_src.tgz" -C "$ROOT"
  else
    git clone https://github.com/nianticlabs/HandDGP.git "$HANDDGP_ROOT"
  fi
fi

if [ "${EGO_SKIP_GIT_UPDATE:-0}" != "1" ] && [ ! -f "$ROOT/handdgp_src.tgz" ]; then
  git -C "$HANDDGP_ROOT" pull --ff-only
fi

mkdir -p "$HANDDGP_ROOT/third_party"
if [ ! -d "$HANDMESH_ROOT/.git" ]; then
  if [ -d "$ROOT/HandMesh" ]; then
    mv "$ROOT/HandMesh" "$HANDMESH_ROOT"
  else
    git clone https://github.com/SeanChenxy/HandMesh.git "$HANDMESH_ROOT"
  fi
fi

cd "$HANDDGP_ROOT"
if [ ! -x .venv/bin/python ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" .venv
fi
source .venv/bin/activate
"$UV_BIN" pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$UV_BIN" pip install --python .venv/bin/python -e .
"$UV_BIN" pip install --python .venv/bin/python numpy==1.26.4 loguru==0.6.0 gin-config==0.5.0 kornia==0.6.11 timm==0.6.13 scipy==1.10.1 opencv-python==4.7.0.72
"$UV_BIN" pip install --python .venv/bin/python cmake==3.27.9
export PATH="$HANDDGP_ROOT/.venv/bin:$PATH"
"$UV_BIN" pip install --python .venv/bin/python openmesh==1.2.1
"$UV_BIN" pip install --python .venv/bin/python torch-geometric==2.3.0
rm -rf "$HANDMESH_ROOT/openmesh"

mkdir -p weights
if [ ! -s weights/handdgp_freihand.ckpt ]; then
  curl -L --fail --retry 4 --retry-delay 5 -o weights/handdgp_freihand.ckpt \
    https://storage.googleapis.com/niantic-lon-static/research/handdgp/handdgp_freihand.ckpt
fi

.venv/bin/python - <<'PY'
from pathlib import Path
import torch
from src.models.handdgp import HandDGP
ckpt = Path("weights/handdgp_freihand.ckpt")
assert ckpt.exists() and ckpt.stat().st_size > 1024 * 1024, ckpt
model = HandDGP(
    batch_size=1,
    latent_size=256,
    spiral_len=(9, 9, 9, 9),
    spiral_dilation=(1, 1, 1, 1),
    spiral_out_channels=(32, 64, 128, 256),
    variant="resnet50",
    imagenet_pretrain=False,
    input_size=224,
)
state = torch.load(ckpt, map_location="cpu")
state = state.get("state_dict", state)
clean = {}
for key, value in state.items():
    key = str(key)
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    clean[key] = value
missing, unexpected = model.load_state_dict(clean, strict=False)
loaded = set(model.state_dict()).intersection(clean)
assert len(loaded) >= int(0.8 * len(model.state_dict())), (len(loaded), len(model.state_dict()), missing[:20], unexpected[:20])
print("handdgp_setup_ok")
PY
