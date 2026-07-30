#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/egoforce_work}"
REPO="${ROOT}/EgoForce"
UV_BIN="${UV_BIN:-/mnt/user-home/yiwen/.local/bin/uv}"
PY="${REPO}/.venv/bin/python"

if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
elif [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

for path in "${REPO}/.git" "${PY}" "${REPO}/thirdparty/datapipes" "${REPO}/thirdparty/mmdetection"; do
  if [ ! -e "${path}" ]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

cd "${REPO}"
"${UV_BIN}" pip install --python "${PY}" thirdparty/datapipes
"${UV_BIN}" pip install --python "${PY}" thirdparty/mmdetection --no-build-isolation
"${UV_BIN}" pip install --python "${PY}" numpy==1.26.4 projectaria_client_sdk==1.1.0 --no-cache-dir --prerelease=allow

"${PY}" - <<'PY'
import importlib

mods = ["torch", "torch_tensorrt", "cv2", "mmdet", "ultralytics", "pytorch3d", "anycalib"]
for name in mods:
    importlib.import_module(name)
print("egoforce_setup_imports_ok")
PY

echo "egoforce_setup_ok"
