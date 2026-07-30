#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_HAWOR_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/hawor_work}"
MANO_ROOT="${EGO_MANO_ROOT:-$ROOT/assets/mano}"
MANO_RIGHT="${EGO_MANO_RIGHT:-$MANO_ROOT/MANO_RIGHT.pkl}"
MANO_LEFT="${EGO_MANO_LEFT:-$MANO_ROOT/MANO_LEFT.pkl}"
ENV_MARKER="torch2.6.0-cu126"

mkdir -p "$ROOT/third_party" "$ROOT/weights/external" "$ROOT/weights/hawor/checkpoints" "$ROOT/data" "$ROOT/outputs"
cd "$ROOT"

for asset in "$MANO_RIGHT" "$MANO_LEFT"; do
  if [ ! -s "$asset" ]; then
    echo "missing required MANO asset: $asset" >&2
    exit 1
  fi
done

export PATH="$HOME/.local/bin:$PATH"
if [ ! -d third_party/HaWoR/.git ]; then
  git clone --recursive https://github.com/ThunderVVV/HaWoR.git third_party/HaWoR
else
  git -C third_party/HaWoR pull --ff-only
  git -C third_party/HaWoR submodule update --init --recursive
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

if [ -f .venv_hawor/.ego_env_marker ] && [ "$(cat .venv_hawor/.ego_env_marker)" != "$ENV_MARKER" ]; then
  rm -rf .venv_hawor
fi
if [ ! -d .venv_hawor ]; then
  uv venv --python 3.10 .venv_hawor
fi
source .venv_hawor/bin/activate
printf '%s\n' "$ENV_MARKER" > .venv_hawor/.ego_env_marker
uv pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install pip "setuptools<70" wheel ninja packaging
uv pip install torch-scatter==2.1.2 --find-links https://data.pyg.org/whl/torch-2.6.0+cu126.html
uv pip install --no-build-isolation -r third_party/HaWoR/requirements.txt
uv pip install pytorch-lightning==2.2.4 --no-deps
uv pip install lightning-utilities torchmetrics==1.4.0 gdown

cd "$ROOT/third_party/HaWoR"

mkdir -p weights/external weights/hawor/checkpoints weights/hawor _DATA/data/mano _DATA/data_left/mano_left
if [ ! -f weights/external/detector.pt ]; then
  wget -O weights/external/detector.pt https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
fi
if [ ! -f weights/hawor/checkpoints/hawor.ckpt ]; then
  wget -O weights/hawor/checkpoints/hawor.ckpt https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt
fi
if [ ! -f weights/hawor/checkpoints/infiller.pt ]; then
  wget -O weights/hawor/checkpoints/infiller.pt https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt
fi
if [ ! -f weights/hawor/model_config.yaml ]; then
  wget -O weights/hawor/model_config.yaml https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml
fi
if [ ! -f weights/external/droid.pth ]; then
  gdown 'https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh' -O weights/external/droid.pth
fi
if [ ! -f thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth ]; then
  mkdir -p thirdparty/Metric3D/weights
  gdown 'https://drive.google.com/uc?id=1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr' -O thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
fi

cp "$MANO_RIGHT" _DATA/data/mano/MANO_RIGHT.pkl
cp "$MANO_LEFT" _DATA/data_left/mano_left/MANO_LEFT.pkl

cd thirdparty/DROID-SLAM
sed -i -E \
  "/-gencode=arch=compute_(60|61|70|75|80|86),code=(sm_|compute_)(60|61|70|75|80|86)/d" \
  setup.py \
  thirdparty/lietorch/setup.py
sed -i \
  -e 's/volume\.type()/volume.scalar_type()/g' \
  -e 's/fmap1\.type()/fmap1.scalar_type()/g' \
  src/correlation_kernels.cu \
  src/altcorr_kernel.cu \
  thirdparty/lietorch/lietorch/extras/corr_index_kernel.cu
sed -i \
  -e 's/::detail::scalar_type(the_type)/the_type.scalarType()/g' \
  thirdparty/lietorch/lietorch/include/dispatch.h
rm -rf build droid_backends.egg-info thirdparty/lietorch/build thirdparty/lietorch/lietorch.egg-info
python setup.py install
cd "$ROOT/third_party/HaWoR"

python - <<'PY'
from pathlib import Path
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable in HaWoR environment"
required = [
    "weights/external/detector.pt",
    "weights/external/droid.pth",
    "weights/hawor/checkpoints/hawor.ckpt",
    "weights/hawor/checkpoints/infiller.pt",
    "weights/hawor/model_config.yaml",
    "thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth",
    "_DATA/data/mano/MANO_RIGHT.pkl",
    "_DATA/data_left/mano_left/MANO_LEFT.pkl",
]
missing = [path for path in required if not Path(path).exists()]
assert not missing, missing
print("hawor_setup_ok")
PY
