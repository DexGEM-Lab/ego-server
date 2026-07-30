#!/usr/bin/env bash
set -euo pipefail

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable is missing: $name" >&2
    exit 1
  fi
}

for name in \
  CASE_NAME \
  OUT_ROOT \
  SOURCE_CLIP \
  FULL_SCENE_MANIFEST \
  OBJECT_PROMPTS \
  HAND_PROMPTS \
  FRAME_START \
  FRAME_END \
  ANCHOR_FRAME \
  QUERY_FRAME_INDEX \
  HAND_TRACK_ID \
  HAND_SIDE \
  OBJECT_LABEL \
  OBJECT_TRACK_ID \
  DEFAULT_CAPTION; do
  require_var "$name"
done

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
REPO_DIR=${REPO_DIR:-$REMOTE_ROOT/repo}
DATA_REMOTE_ROOT=${DATA_REMOTE_ROOT:-$REMOTE_ROOT/data2}
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-/data2}
SAM2_CHECKPOINT=${SAM2_CHECKPOINT:-$REMOTE_ROOT/data/sam2.1_hiera_small.pt}
SAM2_MODEL_CFG=${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}
HAWOR_PY=${HAWOR_PY:-$REMOTE_ROOT/hawor_work/.venv_hawor/bin/python}
HANDDGP_ROOT=${HANDDGP_ROOT:-$REMOTE_ROOT/handdgp_work/HandDGP}
HANDDGP_PY=${HANDDGP_PY:-$HANDDGP_ROOT/.venv/bin/python}
HANDDGP_CHECKPOINT=${HANDDGP_CHECKPOINT:-$HANDDGP_ROOT/weights/handdgp_freihand.ckpt}
UNIDEPTH_PY=${UNIDEPTH_PY:-$REMOTE_ROOT/unidepth_work/UniDepth/.venv/bin/python}
UNIDEPTH_REPO=${UNIDEPTH_REPO:-$REMOTE_ROOT/unidepth_work/UniDepth}
VGGT_PY=${VGGT_PY:-$REMOTE_ROOT/hunyuan3d_v3_env/bin/python}
COTRACKER_PY=${COTRACKER_PY:-$REMOTE_ROOT/cotracker_env/bin/python}
COTRACKER_REPO=${COTRACKER_REPO:-$REMOTE_ROOT/cotracker_work/co-tracker}
HAMER_ROOT=${HAMER_ROOT:-/dev/shm/ego_annotation_hamer_keyboard/hamer}
HAMER_CHECKPOINT=${HAMER_CHECKPOINT:-$HAMER_ROOT/_DATA/hamer_ckpts/checkpoints/hamer.ckpt}
WILOR_ROOT=${WILOR_ROOT:-$REMOTE_ROOT/third_party/WiLoR}
MANO_MODEL_ROOT=${MANO_MODEL_ROOT:-/mnt/user-home/yiwen/data/dex_home/yiwen/arctic/data/body_models/mano}
MIN_OBJECT_MASK_PIXELS=${MIN_OBJECT_MASK_PIXELS:-500}
MIN_HAND_MASK_AREA_PX=${MIN_HAND_MASK_AREA_PX:-200}
MIN_OBJECT_FRAMES=${MIN_OBJECT_FRAMES:-$((FRAME_END - FRAME_START + 1))}
MIN_HAND_FRAMES=${MIN_HAND_FRAMES:-$((FRAME_END - FRAME_START + 1))}
SAM2_OBJECT_MAX_NEGATIVE_HITS=${SAM2_OBJECT_MAX_NEGATIVE_HITS:-0}
SAM2_OBJECT_MIN_POSITIVE_HIT_FRACTION=${SAM2_OBJECT_MIN_POSITIVE_HIT_FRACTION:-0.333}
SAM2_OBJECT_SELECTION_MODE=${SAM2_OBJECT_SELECTION_MODE:-prompt_hits}
SAM2_HAND_MAX_NEGATIVE_HITS=${SAM2_HAND_MAX_NEGATIVE_HITS:-0}
SAM2_HAND_MIN_POSITIVE_HIT_FRACTION=${SAM2_HAND_MIN_POSITIVE_HIT_FRACTION:-0.75}
SAM2_HAND_SELECTION_MODE=${SAM2_HAND_SELECTION_MODE:-prompt_hits}
USE_RTMLIB_HAND_EVIDENCE=${USE_RTMLIB_HAND_EVIDENCE:-0}
USE_RTMLIB_SAM2_HAND_PROMPTS=${USE_RTMLIB_SAM2_HAND_PROMPTS:-$USE_RTMLIB_HAND_EVIDENCE}
USE_HANDDGP_HAND_EVIDENCE=${USE_HANDDGP_HAND_EVIDENCE:-0}
HANDDGP_INVERSE_DEVICE=${HANDDGP_INVERSE_DEVICE:-cuda:0}
MANO_ARTICULATION_DEVICE=${MANO_ARTICULATION_DEVICE:-cuda:0}
RTMLIB_PY=${RTMLIB_PY:-$REMOTE_ROOT/rtmlib_work/venv/bin/python}
RTMLIB_DEVICE=${RTMLIB_DEVICE:-cuda}
RTMLIB_BACKEND=${RTMLIB_BACKEND:-onnxruntime}
VGGT_MIN_DEPTH_PIXELS=${VGGT_MIN_DEPTH_PIXELS:-500}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

case "$HAND_SIDE" in
  left|right|any) ;;
  *)
    echo "HAND_SIDE must be left, right, or any: $HAND_SIDE" >&2
    exit 1
    ;;
esac

RUN_SCRIPT="$OUT_ROOT/run_${CASE_NAME}_v7_measured_video_mesh.sh"
WAIT_SCRIPT="$OUT_ROOT/wait_and_run_${CASE_NAME}_v7_measured_video_mesh.sh"

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:?GPU_ID is required}"
export PYTHONUNBUFFERED=1
cd "$REPO_DIR"

for required_path in \
  "$HAWOR_PY" \
  "$UNIDEPTH_PY" \
  "$VGGT_PY" \
  "$COTRACKER_PY" \
  "$SOURCE_CLIP" \
  "$FULL_SCENE_MANIFEST" \
  "$OBJECT_PROMPTS" \
  "$HAND_PROMPTS" \
  "$SAM2_CHECKPOINT" \
  "$HAMER_ROOT" \
  "$HAMER_CHECKPOINT" \
  "$WILOR_ROOT/wilor/models/mano_wrapper.py" \
  "$MANO_MODEL_ROOT/MANO_RIGHT.pkl" \
  "$COTRACKER_REPO"; do
  if [[ ! -e "\$required_path" ]]; then
    echo "required input is missing: \$required_path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"/{sam2_object,sam2_hand,object_rgb_dataset,unidepth_full_frame,object_metric_manifest,vggt_native,annotations,hand_maskbox,rtmlib_hand2d,rtmlib_hand_prompts,hamer,wilor,hand_candidates,handdgp,handdgp_mano,mano_refit,mano_mask_depth_fit,hand_selection,observed_mesh,cotracker_tracks,cotracker_edges,cotracker_pair_factors}

run_stage() {
  local name="\$1"
  shift
  date '+%Y-%m-%d %H:%M:%S start '"\$name"
  "\$@"
  date '+%Y-%m-%d %H:%M:%S done '"\$name"
}

run_stage sam2_object \
  "$HAWOR_PY" scripts/run_sam2_vlm_points_image.py \
    --clip "$SOURCE_CLIP" \
    --point-prompts "$OBJECT_PROMPTS" \
    --output-dir "$OUT_ROOT/sam2_object" \
    --checkpoint "$SAM2_CHECKPOINT" \
    --model-cfg "$SAM2_MODEL_CFG" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --sam2-image-width 960 \
    --render-width 960 \
    --use-box \
    --save-candidate-masks \
    --min-area-px 80 \
    --max-prompt-area-ratio 3.5 \
    --max-area-fraction 0.45 \
    --min-positive-hit-fraction "$SAM2_OBJECT_MIN_POSITIVE_HIT_FRACTION" \
    --max-negative-hits "$SAM2_OBJECT_MAX_NEGATIVE_HITS" \
    --selection-mode "$SAM2_OBJECT_SELECTION_MODE"

HAND_PROMPTS_FOR_SAM2="$HAND_PROMPTS"
HAND_PROPOSAL_JSON="$OUT_ROOT/hand_maskbox/hand_mask_box_evidence.json"
if [[ "$USE_RTMLIB_HAND_EVIDENCE" == "1" ]]; then
  if [[ ! -x "$RTMLIB_PY" ]]; then
    echo "RTMLib Python is missing or not executable: $RTMLIB_PY" >&2
    exit 1
  fi
  run_stage rtmlib_hand2d \
    "$RTMLIB_PY" scripts/run_rtmlib_hand2d_v3.py \
      --clip "$SOURCE_CLIP" \
      --output-dir "$OUT_ROOT/rtmlib_hand2d" \
      --frame-start "$FRAME_START" \
      --frame-end "$FRAME_END" \
      --review-frames "$FRAME_START" "$ANCHOR_FRAME" "$FRAME_END" \
      --device "$RTMLIB_DEVICE" \
      --backend "$RTMLIB_BACKEND"

  run_stage rtmlib_hand_prompts \
    "$HAWOR_PY" scripts/build_rtmlib_sam2_hand_prompts_v7.py \
      --rtmlib-json "$OUT_ROOT/rtmlib_hand2d/rtmlib_hand2d.json" \
      --reference-prompts "$HAND_PROMPTS" \
      --output-json "$OUT_ROOT/rtmlib_hand_prompts/visual_track_point_prompts_rtmlib_v7.json" \
      --frame-start "$FRAME_START" \
      --frame-end "$FRAME_END" \
      --track-id "$HAND_TRACK_ID"
  if [[ "$USE_RTMLIB_SAM2_HAND_PROMPTS" == "1" ]]; then
    HAND_PROMPTS_FOR_SAM2="$OUT_ROOT/rtmlib_hand_prompts/visual_track_point_prompts_rtmlib_v7.json"
  fi
  HAND_PROPOSAL_JSON="$OUT_ROOT/rtmlib_hand2d/rtmlib_hand2d.json"
fi

run_stage sam2_hand \
  "$HAWOR_PY" scripts/run_sam2_vlm_points_image.py \
    --clip "$SOURCE_CLIP" \
    --point-prompts "\$HAND_PROMPTS_FOR_SAM2" \
    --output-dir "$OUT_ROOT/sam2_hand" \
    --checkpoint "$SAM2_CHECKPOINT" \
    --model-cfg "$SAM2_MODEL_CFG" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --sam2-image-width 960 \
    --render-width 960 \
    --use-box \
    --save-candidate-masks \
    --min-area-px 80 \
    --max-prompt-area-ratio 3.5 \
    --max-area-fraction 0.10 \
    --min-positive-hit-fraction "$SAM2_HAND_MIN_POSITIVE_HIT_FRACTION" \
    --max-negative-hits "$SAM2_HAND_MAX_NEGATIVE_HITS" \
    --selection-mode "$SAM2_HAND_SELECTION_MODE"

run_stage object_rgb_dataset \
  "$HAWOR_PY" scripts/export_mask_track_rgb_dataset_v3.py \
    --clip "$SOURCE_CLIP" \
    --mask-track "$OUT_ROOT/sam2_object/sam2_track.json" \
    --output-dir "$OUT_ROOT/object_rgb_dataset" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --min-mask-pixels "$MIN_OBJECT_MASK_PIXELS" \
    --min-frames "$MIN_OBJECT_FRAMES"

run_stage unidepth_full_frame \
  "$UNIDEPTH_PY" scripts/run_unidepth_full_frame_v3.py \
    --manifest "$FULL_SCENE_MANIFEST" \
    --output-dir "$OUT_ROOT/unidepth_full_frame" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --unidepth-repo "$UNIDEPTH_REPO" \
    --remote-root "$DATA_REMOTE_ROOT" \
    --local-root "$LOCAL_DATA_ROOT" \
    --source-width 1920 \
    --source-height 1080

run_stage object_metric_manifest \
  "$HAWOR_PY" scripts/build_mask_unidepth_metric_manifest_v3.py \
    --mask-manifest "$OUT_ROOT/object_rgb_dataset/manifest.json" \
    --unidepth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-dir "$OUT_ROOT/object_metric_manifest" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --min-mask-depth-pixels "$MIN_OBJECT_MASK_PIXELS" \
    --min-frames "$MIN_OBJECT_FRAMES"

run_stage vggt_native \
  "$VGGT_PY" scripts/run_vggt_native_camera_v3.py \
    --dataset-manifest "$FULL_SCENE_MANIFEST" \
    --output-dir "$OUT_ROOT/vggt_native" \
    --repo-root "$REPO_DIR" \
    --remote-output-root "$DATA_REMOTE_ROOT" \
    --local-output-root "$LOCAL_DATA_ROOT" \
    --metric-depth-manifest "$OUT_ROOT/object_metric_manifest/manifest.json" \
    --metric-remote-root "$DATA_REMOTE_ROOT" \
    --metric-local-root "$LOCAL_DATA_ROOT" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --anchor-frame "$ANCHOR_FRAME" \
    --source-width 1920 \
    --source-height 1080 \
    --target-size 518 \
    --gpu 0 \
    --min-depth-pixels "$VGGT_MIN_DEPTH_PIXELS"

run_stage annotations \
  "$HAWOR_PY" scripts/build_vggt_object_skeleton_generic_v3.py \
    --video "$SOURCE_CLIP" \
    --vggt-archive "$OUT_ROOT/vggt_native/vggt_native_camera_v3.npz" \
    --mask-track "$OUT_ROOT/sam2_object/sam2_track.json" \
    --output-dir "$OUT_ROOT/annotations" \
    --object-label "$OBJECT_LABEL" \
    --track-id "$OBJECT_TRACK_ID" \
    --default-caption "$DEFAULT_CAPTION" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END"

run_stage hand_maskbox \
  "$HAWOR_PY" scripts/build_hand_mask_box_evidence_v3.py \
    --"$HAND_SIDE"-track "$OUT_ROOT/sam2_hand/sam2_track.json" \
    --"$HAND_SIDE"-track-id "$HAND_TRACK_ID" \
    --output-json "$OUT_ROOT/hand_maskbox/hand_mask_box_evidence.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --source-width 1920 \
    --source-height 1080 \
    --mask-width 960 \
    --mask-height 540 \
    --min-mask-area-px "$MIN_HAND_MASK_AREA_PX" \
    --remote-output-root "$OUT_ROOT/sam2_hand" \
    --local-output-root "$OUT_ROOT/sam2_hand"

run_stage hamer_maskbox \
  "$HAWOR_PY" scripts/run_hamer_rtmlib_hand_stream_v3.py \
    --target-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --frame-manifest "$FULL_SCENE_MANIFEST" \
    --rtmlib-json "\$HAND_PROPOSAL_JSON" \
    --output-annotations "$OUT_ROOT/hamer/annotations_hamer_maskbox.json" \
    --output-qc "$OUT_ROOT/hamer/qc_hamer_maskbox.json" \
    --hamer-root "$HAMER_ROOT" \
    --checkpoint "$HAMER_CHECKPOINT" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --local-root "$LOCAL_DATA_ROOT" \
    --remote-root "$DATA_REMOTE_ROOT" \
    --device cuda:0 \
    --measurement-source hamer-full-projection \
    --min-keypoint-score 0.2 \
    --min-valid-keypoints 12 \
    --min-mean-score 0.30 \
    --min-measured-hands "$MIN_HAND_FRAMES" \
    --allow-insufficient-measured-hands

run_stage wilor_maskbox \
  "$HAWOR_PY" scripts/run_wilor_maskbox_hand_stream_v7.py \
    --target-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --frame-manifest "$FULL_SCENE_MANIFEST" \
    --maskbox-json "$OUT_ROOT/hand_maskbox/hand_mask_box_evidence.json" \
    --output-annotations "$OUT_ROOT/wilor/annotations_wilor_maskbox.json" \
    --output-qc "$OUT_ROOT/wilor/qc_wilor_maskbox.json" \
    --wilor-root "$WILOR_ROOT" \
    --mano-right "$MANO_MODEL_ROOT/MANO_RIGHT.pkl" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --local-root "$LOCAL_DATA_ROOT" \
    --remote-root "$DATA_REMOTE_ROOT" \
    --device cuda:0 \
    --track-id "$HAND_TRACK_ID" \
    --side "$HAND_SIDE" \
    --min-measured-hands "$MIN_HAND_FRAMES" \
    --allow-insufficient-measured-hands

run_stage merge_hamer_wilor_candidates \
  "$HAWOR_PY" scripts/merge_hand_candidate_streams_v7.py \
    --base-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --hand-streams "$OUT_ROOT/hamer/annotations_hamer_maskbox.json" "$OUT_ROOT/wilor/annotations_wilor_maskbox.json" \
    --output-annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_initial.json" \
    --output-qc "$OUT_ROOT/hand_candidates/qc_hamer_wilor_maskbox_initial.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END"

if [[ "$USE_HANDDGP_HAND_EVIDENCE" == "1" ]]; then
  run_stage handdgp_export \
    "$HANDDGP_PY" scripts/run_handdgp_export_v3.py \
      --video "$SOURCE_CLIP" \
      --annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_initial.json" \
      --handdgp-root "$HANDDGP_ROOT" \
      --checkpoint "$HANDDGP_CHECKPOINT" \
      --output-annotations "$OUT_ROOT/handdgp/annotations_handdgp.json" \
      --output-raw-npz "$OUT_ROOT/handdgp/handdgp_raw.npz" \
      --output-qc "$OUT_ROOT/handdgp/qc_handdgp.json" \
      --frame-start "$FRAME_START" \
      --frame-end "$FRAME_END" \
      --batch-size 16 \
      --min-score 0.10
  run_stage handdgp_inverse_mano \
    "$HAWOR_PY" scripts/convert_handdgp_to_mano_candidates_v7.py \
      --annotations "$OUT_ROOT/handdgp/annotations_handdgp.json" \
      --output-annotations "$OUT_ROOT/handdgp_mano/annotations_handdgp_mano.json" \
      --output-qc "$OUT_ROOT/handdgp_mano/qc_handdgp_inverse_mano.json" \
      --wilor-root "$WILOR_ROOT" \
      --mano-right "$MANO_MODEL_ROOT/MANO_RIGHT.pkl" \
      --frame-start "$FRAME_START" \
      --frame-end "$FRAME_END" \
      --track-id "$HAND_TRACK_ID" \
      --side "$HAND_SIDE" \
      --device "$HANDDGP_INVERSE_DEVICE" \
      --min-measured-hands "$MIN_HAND_FRAMES"
  run_stage merge_hand_candidates \
    "$HAWOR_PY" scripts/merge_hand_candidate_streams_v7.py \
      --base-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
      --hand-streams "$OUT_ROOT/hamer/annotations_hamer_maskbox.json" "$OUT_ROOT/wilor/annotations_wilor_maskbox.json" "$OUT_ROOT/handdgp_mano/annotations_handdgp_mano.json" \
      --output-annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox.json" \
      --output-qc "$OUT_ROOT/hand_candidates/qc_hamer_wilor_maskbox.json" \
      --frame-start "$FRAME_START" \
      --frame-end "$FRAME_END"
else
  cp "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_initial.json" "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox.json"
  cp "$OUT_ROOT/hand_candidates/qc_hamer_wilor_maskbox_initial.json" "$OUT_ROOT/hand_candidates/qc_hamer_wilor_maskbox.json"
fi

run_stage mano_metric_refit \
  "$HAWOR_PY" scripts/refit_mano_metric_depth_v3.py \
    --annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-json "$OUT_ROOT/mano_refit/annotations_hamer_wilor_maskbox_metric_refit.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --min-detector-score 0.10 \
    --min-depth-joints 12 \
    --min-rows "$MIN_HAND_FRAMES"

articulation_refit_args=(
  --annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox.json"
  --mask-track "$OUT_ROOT/sam2_hand/sam2_track.json"
  --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
  --output-annotations "$OUT_ROOT/mano_mask_depth_fit/annotations_articulation_mask_depth_refit.json"
  --output-qc "$OUT_ROOT/mano_mask_depth_fit/qc_articulation_mask_depth_refit.json"
  --video "$SOURCE_CLIP"
  --review-dir "$OUT_ROOT/mano_mask_depth_fit/review"
  --mano-wrapper-root "$WILOR_ROOT"
  --mano-model-root "$MANO_MODEL_ROOT"
  --frame-start "$FRAME_START"
  --frame-end "$FRAME_END"
  --track-id "$HAND_TRACK_ID"
  --side "$HAND_SIDE"
  --source-width 1920
  --source-height 1080
  --remote-output-root "$OUT_ROOT/sam2_hand"
  --local-output-root "$OUT_ROOT/sam2_hand"
  --min-observations "$MIN_HAND_FRAMES"
  --still-frames "$FRAME_START" "$ANCHOR_FRAME" "$FRAME_END"
  --device "$MANO_ARTICULATION_DEVICE"
)
if [[ "$USE_RTMLIB_HAND_EVIDENCE" == "1" ]]; then
  articulation_refit_args+=(
    --rtmlib-json "$OUT_ROOT/rtmlib_hand2d/rtmlib_hand2d.json"
    --rtmlib-prompts "$OUT_ROOT/rtmlib_hand_prompts/visual_track_point_prompts_rtmlib_v7.json"
    --w-rtmlib-keypoints 1.0
    --sigma-rtmlib-keypoint-px 18.0
    --rtmlib-min-score 0.30
    --rtmlib-min-keypoints 12
  )
fi

run_stage mano_articulation_mask_depth_refit \
  "$HAWOR_PY" scripts/refit_mano_articulation_mask_depth_v3.py \
    "\${articulation_refit_args[@]}"

hand_selection_args=(
  --annotations "$OUT_ROOT/mano_mask_depth_fit/annotations_articulation_mask_depth_refit.json"
  --output-annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit.json"
  --output-qc "$OUT_ROOT/hand_selection/qc_selected_hand_metric_refit.json"
  --frame-start "$FRAME_START"
  --frame-end "$FRAME_END"
)
if [[ "$HAND_SIDE" != "any" ]]; then
  hand_selection_args+=(--required-side "$HAND_SIDE")
fi

run_stage hand_selection \
  "$HAWOR_PY" scripts/select_hand_hypotheses_by_residual_v7.py \
    "\${hand_selection_args[@]}"

"$HAWOR_PY" - <<'PY'
import json
from pathlib import Path

qc_path = Path("$OUT_ROOT/hand_selection/qc_selected_hand_metric_refit.json")
qc = json.loads(qc_path.read_text(encoding="utf-8"))
expected = int("$FRAME_END") - int("$FRAME_START") + 1
selected = int(qc.get("selected_frames", -1))
if qc.get("status") != "ok" or selected != expected:
    raise RuntimeError(f"hand selection did not produce one measured hand per frame: status={qc.get('status')} selected={selected} expected={expected}")
PY

run_stage observed_mesh \
  "$HAWOR_PY" scripts/export_mask_depth_observed_mesh_archive_v3.py \
    --dataset "$OUT_ROOT/object_metric_manifest" \
    --manifest "$OUT_ROOT/object_metric_manifest/manifest.json" \
    --annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-dir "$OUT_ROOT/observed_mesh" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --coordinate world \
    --intrinsics-source annotation-vggt \
    --mask-stride 1 \
    --mask-erode-px 0 \
    --min-depth-pixels "$MIN_OBJECT_MASK_PIXELS" \
    --min-vertices 250 \
    --min-faces 300 \
    --min-frames "$MIN_OBJECT_FRAMES" \
    --max-triangle-edge-m 0.050

run_stage cotracker_tracks \
  "$COTRACKER_PY" scripts/run_cotracker_object_tracks_v5.py \
    --manifest "$OUT_ROOT/object_metric_manifest/manifest.json" \
    --annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-dir "$OUT_ROOT/cotracker_tracks" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --query-frame-index "$QUERY_FRAME_INDEX" \
    --grid-step-px 24 \
    --max-points 384 \
    --output-fps 6 \
    --still-frames "$FRAME_START" "$ANCHOR_FRAME" "$FRAME_END" \
    --torchhub-repo "$COTRACKER_REPO" \
    --torchhub-model cotracker3_offline \
    --torchhub-source local \
    --backward-tracking \
    --require-cuda

run_stage cotracker_edges \
  "$COTRACKER_PY" scripts/build_cotracker_sparse_correspondence_edges_v5.py \
    --cotracker-npz "$OUT_ROOT/cotracker_tracks/cotracker_object_tracks_v5.npz" \
    --mesh-archive "$OUT_ROOT/observed_mesh/observed_mask_depth_meshes_world.npz" \
    --output-json "$OUT_ROOT/cotracker_edges/cotracker_sparse_correspondence_edges_v6.json" \
    --min-track-frames 4 \
    --max-surface-distance-m 0.004 \
    --max-world-step-m 0.040 \
    --max-frame-gap 1

run_stage cotracker_pair_factors \
  "$COTRACKER_PY" scripts/fit_cotracker_pairwise_rigid_factors_v6.py \
    --cotracker-npz "$OUT_ROOT/cotracker_tracks/cotracker_object_tracks_v5.npz" \
    --sparse-edges-json "$OUT_ROOT/cotracker_edges/cotracker_sparse_correspondence_edges_v6.json" \
    --output-json "$OUT_ROOT/cotracker_pair_factors/qc_cotracker_pairwise_rigid_factors_v6.json" \
    --min-pair-tracks 12 \
    --min-inlier-tracks 12 \
    --huber-delta-m 0.010 \
    --max-inlier-residual-m 0.012 \
    --accept-inlier-p95-m 0.010

"$HAWOR_PY" - <<'PY'
import json
from pathlib import Path

root = Path("$OUT_ROOT")
report = {
    "status": "ok",
    "method": "v7_measured_video_mesh_remote_job",
    "case_name": "$CASE_NAME",
    "frame_start": int("$FRAME_START"),
    "frame_end": int("$FRAME_END"),
    "outputs": {
        "object_sam2_track": str(root / "sam2_object" / "sam2_track.json"),
        "hand_sam2_track": str(root / "sam2_hand" / "sam2_track.json"),
        "object_manifest": str(root / "object_metric_manifest" / "manifest.json"),
        "metric_depth_npz": str(root / "unidepth_full_frame" / "unidepth_full_frame_depth_v3.npz"),
        "vggt_archive": str(root / "vggt_native" / "vggt_native_camera_v3.npz"),
        "annotations": str(root / "hand_selection" / "annotations_selected_hand_metric_refit.json"),
        "observed_mesh_archive": str(root / "observed_mesh" / "observed_mask_depth_meshes_world.npz"),
        "cotracker_npz": str(root / "cotracker_tracks" / "cotracker_object_tracks_v5.npz"),
        "sparse_edges_json": str(root / "cotracker_edges" / "cotracker_sparse_correspondence_edges_v6.json"),
        "pair_factors_json": str(root / "cotracker_pair_factors" / "qc_cotracker_pairwise_rigid_factors_v6.json"),
    },
}
(root / "qc_v7_measured_video_mesh_remote_job.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
EOF
chmod +x "$RUN_SCRIPT"

cat > "$WAIT_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$RUN_SCRIPT"
MAX_USED_MB="\${MAX_USED_MB:-$MAX_USED_MB}"
POLL_SECONDS="\${POLL_SECONDS:-$POLL_SECONDS}"
GPU_SELECT_LOCK="\${GPU_SELECT_LOCK:-$GPU_SELECT_LOCK}"
GPU_LOCK_DIR="\${GPU_LOCK_DIR:-$GPU_LOCK_DIR}"
mkdir -p "\$GPU_LOCK_DIR"
while true; do
  GPU_ID=""
  exec 9>"\$GPU_SELECT_LOCK"
  flock -x 9
  while IFS=, read -r gpu_idx used_mb; do
    gpu_idx="\${gpu_idx//[[:space:]]/}"
    used_mb="\${used_mb//[[:space:]]/}"
    if [[ -n "\$gpu_idx" && -n "\$used_mb" && "\$used_mb" -le "\$MAX_USED_MB" ]]; then
      exec 8>"\$GPU_LOCK_DIR/gpu_\${gpu_idx}.lock"
      if flock -n 8; then
        GPU_ID="\$gpu_idx"
        break
      fi
      exec 8>&-
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [[ -n "\$GPU_ID" ]]; then
    export GPU_ID
    flock -u 9
    exec 9>&-
    date '+%Y-%m-%d %H:%M:%S selected GPU '"\$GPU_ID"
    exec bash "\$RUN_SCRIPT"
  fi
  flock -u 9
  exec 9>&-
  date '+%Y-%m-%d %H:%M:%S no GPU below memory threshold; sleeping'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep "\$POLL_SECONDS"
done
EOF
chmod +x "$WAIT_SCRIPT"

printf '%s\n%s\n' "$RUN_SCRIPT" "$WAIT_SCRIPT"
