#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote/egoforce_work}"
REPO="${ROOT}/EgoForce"
PY="${REPO}/.venv/bin/python"
GPU="${EGO_GPU:-6}"
OUT="${ROOT}/outputs/v3_egoforce_detector_840_930"
SCRIPT="${ROOT}/local_scripts/run_egoforce_export_v3.py"

VIDEO="${EGO_VIDEO:-/mnt/user-home/yiwen/ego_annotation_remote/data/clip/20260108_1057_Recf94e_P0_S994da4_task_9.mp4}"
ANNOTATIONS="${EGO_ANNOTATIONS:-${ROOT}/data/annotations/annotations_v1_full.json}"

for path in "${PY}" "${SCRIPT}" "${VIDEO}" "${ANNOTATIONS}" "${REPO}/_DATA/model_weights.pth" "${REPO}/_DATA/epoch_460.pth" "${REPO}/_DATA/detector.torchscript" "${REPO}/_DATA/mano/MANO_LEFT.pkl" "${REPO}/_DATA/mano/MANO_RIGHT.pkl"; do
  if [ ! -e "${path}" ]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUT}"
cd "${REPO}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO}/demo:${REPO}:${PYTHONPATH:-}"
if [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

"${PY}" "${SCRIPT}" \
  --video "${VIDEO}" \
  --annotations "${ANNOTATIONS}" \
  --egoforce-root "${REPO}" \
  --output-annotations "${OUT}/annotations_egoforce_detector.json" \
  --output-npz "${OUT}/egoforce_detector_raw.npz" \
  --output-qc "${OUT}/qc_egoforce_detector.json" \
  --overlay-video "${OUT}/egoforce_detector_overlay.mp4" \
  --frame-start 840 \
  --frame-end 930 \
  --frame-stride 1 \
  --intrinsics 2304 2304 960 540 \
  --disable-kalman \
  --crop-source egoforce_detector
