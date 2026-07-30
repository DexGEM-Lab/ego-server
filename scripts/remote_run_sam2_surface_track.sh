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

PROMPT_FRAMES="$(
  POINT_PROMPTS="$POINT_PROMPTS" "$PYTHON_BIN" - <<'PY'
import json
import os
payload = json.load(open(os.environ["POINT_PROMPTS"]))
frames = [
    int(row["frame_idx"])
    for row in payload["point_prompts"]
    if row.get("target_visible") and row.get("positive_points")
]
if not frames:
    raise SystemExit("no visible prompt frames with positive points")
print(",".join(str(x) for x in frames))
PY
)"

mkdir -p "$OUTPUT_DIR"

if [ "${EGO_USE_UV:-0}" = "1" ]; then
  RUNNER=(uv run python)
else
  RUNNER=("$PYTHON_BIN")
fi

PYTHONPATH="scripts:third_party/sam2${PYTHONPATH:+:$PYTHONPATH}" "${RUNNER[@]}" scripts/run_sam2_vlm_points_track.py \
  --clip "$CLIP" \
  --point-prompts "$POINT_PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --prompt-frames "$PROMPT_FRAMES" \
  --sam2-image-width 960 \
  --render-width 960
