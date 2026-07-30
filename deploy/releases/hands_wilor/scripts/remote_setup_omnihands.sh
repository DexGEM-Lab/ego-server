#!/usr/bin/env bash
set -euo pipefail

ROOT="${OMNI_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/omnihands_work}"
REPO="${ROOT}/OmniHands"
LOG_DIR="${ROOT}/logs"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"

mkdir -p "${ROOT}" "${LOG_DIR}"
cd "${ROOT}"

if [ ! -x "${UV_BIN}" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="/mnt/user-home/yiwen/.local/bin/uv"
fi

if [ ! -d "${REPO}/.git" ]; then
  git clone --recursive https://github.com/LinDixuan/OmniHands.git "${REPO}"
elif [ "${OMNI_SKIP_GIT_UPDATE:-0}" != "1" ]; then
  git -C "${REPO}" pull --ff-only
  git -C "${REPO}" submodule update --init --recursive
fi

cd "${REPO}"
"${UV_BIN}" venv --clear --python 3.10 .venv
source .venv/bin/activate

"${UV_BIN}" pip install --python .venv/bin/python --upgrade pip setuptools==81.0.0 wheel
"${UV_BIN}" pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
"${UV_BIN}" pip install --python .venv/bin/python git+https://github.com/mattloper/chumpy.git --no-build-isolation
"${UV_BIN}" pip install --python .venv/bin/python \
  gdown numpy opencv-python pyrender pytorch-lightning scikit-image smplx==0.1.28 yacs \
  timm einops pandas plyfile hydra-core \
  hydra-submitit-launcher hydra-colorlog pyrootutils rich webdataset xtcocotools
if [ "${OMNI_INSTALL_DETECTORS:-0}" = "1" ]; then
  "${UV_BIN}" pip install --python .venv/bin/python git+https://github.com/facebookresearch/detectron2 --no-build-isolation
  "${UV_BIN}" pip install --python .venv/bin/python -v -e third-party/ViTPose
fi

python - <<'PY'
import importlib
for name in ["torch", "cv2", "einops", "hands_4d"]:
    importlib.import_module(name)
print("omnihands_setup_imports_ok")
PY

echo "omnihands_setup_ok"
