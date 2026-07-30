#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 REPO_DIR CLIP POINT_PROMPTS OUTPUT_DIR CHECKPOINT FRAME_START FRAME_END" >&2
  exit 2
fi

REPO_DIR="$1"
CLIP="$2"
POINT_PROMPTS="$3"
OUTPUT_DIR="$4"
CHECKPOINT="$5"
FRAME_START="$6"
FRAME_END="$7"

cd "$REPO_DIR"
PYTHON_BIN="${EGO_PYTHON_BIN:-python}"

if [ ! -f "$CLIP" ]; then
  echo "missing clip: $CLIP" >&2
  exit 1
fi
if [ ! -f "$POINT_PROMPTS" ]; then
  echo "missing point prompts: $POINT_PROMPTS" >&2
  exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
  echo "missing SAM2 checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -d third_party/sam2/sam2 ]; then
  echo "missing third_party/sam2 checkout under $REPO_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ "${EGO_USE_UV:-0}" = "1" ]; then
  RUNNER=(uv run python)
else
  RUNNER=("$PYTHON_BIN")
fi

EXTRA_ARGS=()
if [ "${EGO_SAVE_CANDIDATE_MASKS:-0}" = "1" ]; then
  EXTRA_ARGS+=(--save-candidate-masks)
fi
if [ -n "${EGO_MIN_POSITIVE_HIT_FRACTION:-}" ]; then
  EXTRA_ARGS+=(--min-positive-hit-fraction "$EGO_MIN_POSITIVE_HIT_FRACTION")
fi
if [ -n "${EGO_MAX_NEGATIVE_HITS:-}" ]; then
  EXTRA_ARGS+=(--max-negative-hits "$EGO_MAX_NEGATIVE_HITS")
fi
if [ -n "${EGO_MAX_AREA_FRACTION:-}" ]; then
  EXTRA_ARGS+=(--max-area-fraction "$EGO_MAX_AREA_FRACTION")
fi
if [ -n "${EGO_MAX_PROMPT_AREA_RATIO:-}" ]; then
  EXTRA_ARGS+=(--max-prompt-area-ratio "$EGO_MAX_PROMPT_AREA_RATIO")
fi

PYTHONPATH="scripts:third_party/sam2${PYTHONPATH:+:$PYTHONPATH}" "${RUNNER[@]}" scripts/run_sam2_vlm_points_image.py \
  --clip "$CLIP" \
  --point-prompts "$POINT_PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --sam2-image-width 960 \
  --render-width 960 \
  --use-box \
  "${EXTRA_ARGS[@]}"
