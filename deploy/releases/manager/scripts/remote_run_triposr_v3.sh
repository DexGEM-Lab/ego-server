#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 REMOTE_ROOT INPUT_IMAGE OUTPUT_DIR GPU_ID MC_RESOLUTION MODEL_FORMAT" >&2
  exit 2
fi

ROOT="$1"
INPUT_IMAGE="$2"
OUTPUT_DIR="$3"
GPU_ID="$4"
MC_RESOLUTION="$5"
MODEL_FORMAT="$6"

WORK="$ROOT/triposr"
PY="$WORK/.venv/bin/python"

if [ ! -d "$WORK/.git" ]; then
  echo "missing TripoSR checkout: $WORK" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "missing TripoSR Python env: $PY" >&2
  exit 1
fi
if [ ! -f "$INPUT_IMAGE" ]; then
  echo "missing input image: $INPUT_IMAGE" >&2
  exit 1
fi

case "$MODEL_FORMAT" in
  obj|ply)
    ;;
  *)
    echo "unsupported model format: $MODEL_FORMAT" >&2
    exit 2
    ;;
esac

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/0"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-/mnt/user-home/yiwen/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/user-home/yiwen/.cache/pip}"
export CMAKE_PREFIX_PATH="$WORK/.venv/lib/python3.10/site-packages/torch/share/cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

cd "$WORK"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA unavailable")
PY

"$PY" run.py "$INPUT_IMAGE" \
  --no-remove-bg \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR" \
  --model-save-format "$MODEL_FORMAT" \
  --mc-resolution "$MC_RESOLUTION"

if [ ! -f "$OUTPUT_DIR/0/mesh.$MODEL_FORMAT" ]; then
  echo "missing TripoSR output mesh: $OUTPUT_DIR/0/mesh.$MODEL_FORMAT" >&2
  exit 1
fi

echo TRIPOSR_RUN_V3_OK
