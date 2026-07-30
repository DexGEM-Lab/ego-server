#!/usr/bin/env bash
set -euo pipefail

ROOT="${EGO_REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}"
FRAME_START="${EGO_FRAME_START:-840}"
FRAME_END="${EGO_FRAME_END:-930}"
GPU="${EGO_GPU:-0}"
DATASET="${EGO_DATASET:-$ROOT/outputs/v3_full_scene_vggt_dataset_${FRAME_START}_${FRAME_END}/manifest.json}"
ANNOTATIONS="${EGO_ANNOTATIONS:-$ROOT/outputs/v3_camera_annotations_${FRAME_START}_${FRAME_END}_for_vggt.json}"
OUT="${EGO_OUTPUT_DIR:-$ROOT/outputs/v3_vggt_full_scene_geometry_${FRAME_START}_${FRAME_END}}"
LOG="$OUT/tmux.log"
PY="${EGO_PYTHON:-$ROOT/triposr/.venv/bin/python}"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT/repo"

echo "python $PY" | tee "$LOG"
"$PY" - <<'PY' | tee -a "$LOG"
import torch
import cv2
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("cv2", cv2.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable for VGGT run")
PY

"$PY" scripts/run_vggt_scene_geometry_v3.py \
  --dataset-manifest "$DATASET" \
  --annotations "$ANNOTATIONS" \
  --output-dir "$OUT" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --gpu 0 \
  --remote-output-root "$ROOT/outputs" \
  --local-output-root /data2/ego_annotation_outputs/representative_trash \
  --max-points-per-frame "${EGO_MAX_POINTS_PER_FRAME:-12000}" \
  --min-points-per-frame "${EGO_MIN_POINTS_PER_FRAME:-900}" \
  --conf-quantile "${EGO_CONF_QUANTILE:-0.30}" 2>&1 | tee -a "$LOG"
echo "EXIT:${PIPESTATUS[0]}" | tee -a "$LOG"
