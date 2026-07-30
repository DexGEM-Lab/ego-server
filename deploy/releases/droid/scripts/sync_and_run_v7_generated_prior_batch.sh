#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-192.168.11.220}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
LOCAL_ROOT=${LOCAL_ROOT:-/data2/ego_annotation_outputs}
OUTPUT_ROOT=${OUTPUT_ROOT:-$LOCAL_ROOT/v7_generated_candidate_batch_$(date +%Y%m%d_%H%M%S)}
PY=${PY:-.venv/bin/python}
SSH_CMD=${SSH_CMD:-ssh -o IPQoS=none -o ConnectTimeout=10}
DISCOVERY_SOURCES=${DISCOVERY_SOURCES:-}

mkdir -p \
  "$LOCAL_ROOT/v7_triposg_prior_outputs" \
  "$LOCAL_ROOT/v7_hunyuan_prior_outputs" \
  "$LOCAL_ROOT/v7_hunyuan21_prior_outputs" \
  "$LOCAL_ROOT/v7_instantmesh_prior_outputs" \
  "$LOCAL_ROOT/v7_spar3d_prior_outputs" \
  "$LOCAL_ROOT/v7_pixal3d_prior_outputs" \
  "$LOCAL_ROOT/v7_sam3d_objects_outputs" \
  "$LOCAL_ROOT/v7_partcrafter_prior_outputs" \
  "$OUTPUT_ROOT"

rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_triposg_prior_outputs/" "$LOCAL_ROOT/v7_triposg_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_hunyuan_prior_outputs/" "$LOCAL_ROOT/v7_hunyuan_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_hunyuan21_prior_outputs/" "$LOCAL_ROOT/v7_hunyuan21_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_instantmesh_prior_outputs/" "$LOCAL_ROOT/v7_instantmesh_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_spar3d_prior_outputs/" "$LOCAL_ROOT/v7_spar3d_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_pixal3d_prior_outputs/" "$LOCAL_ROOT/v7_pixal3d_prior_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_sam3d_objects_outputs/" "$LOCAL_ROOT/v7_sam3d_objects_outputs/"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_ROOT/v7_partcrafter_prior_outputs/" "$LOCAL_ROOT/v7_partcrafter_prior_outputs/"

DISCOVERY_DIR="$OUTPUT_ROOT/discovery"
source_args=()
if [ -n "$DISCOVERY_SOURCES" ]; then
  for source_name in $DISCOVERY_SOURCES; do
    source_args+=(--source "$source_name")
  done
fi
"$PY" scripts/discover_v7_generated_prior_candidates.py \
  "${source_args[@]}" \
  --output-json "$DISCOVERY_DIR/qc_discovered_candidates.json" \
  --output-args "$DISCOVERY_DIR/candidate_args.txt" \
  --require-candidates

"$PY" scripts/run_v7_prior_candidate_batch.py \
  --candidate-file "$DISCOVERY_DIR/candidate_args.txt" \
  --output-root "$OUTPUT_ROOT/replay_batch" \
  --run-physics \
  --render-deliverables
