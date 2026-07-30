#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
REPO_DIR=${REPO_DIR:-$REMOTE_ROOT/repo}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_mop_702_708_outputs}
INPUT_ROOT=${INPUT_ROOT:-$REMOTE_ROOT/v7_mop_702_708_inputs}
DATA_REMOTE_ROOT=${DATA_REMOTE_ROOT:-$REMOTE_ROOT/data2}
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-/data2}
SOURCE_CLIP=${SOURCE_CLIP:-$DATA_REMOTE_ROOT/egoscale_demo_30h/egoscale_tasks/20251210_0002_Rec4afc_P0_S296a7f_task_4/20251210_0002_Rec4afc_P0_S296a7f_task_4.mp4}
FULL_SCENE_MANIFEST=${FULL_SCENE_MANIFEST:-$DATA_REMOTE_ROOT/ego_annotation_outputs/representative_mop/v7_full_scene_vggt_dataset_702_708/manifest.json}
OBJECT_PROMPTS=${OBJECT_PROMPTS:-$DATA_REMOTE_ROOT/ego_annotation_outputs/representative_mop/v7_mop_point_prompts_702_708/object_point_prompts_vlm.json}
HAND_PROMPTS=${HAND_PROMPTS:-$DATA_REMOTE_ROOT/ego_annotation_outputs/representative_mop/v7_mop_hand_point_prompts_702_708/visual_track_point_prompts_vlm.json}
SAM2_CHECKPOINT=${SAM2_CHECKPOINT:-$REMOTE_ROOT/data/sam2.1_hiera_small.pt}
SAM2_MODEL_CFG=${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_s.yaml}
HAWOR_PY=${HAWOR_PY:-$REMOTE_ROOT/hawor_work/.venv_hawor/bin/python}
UNIDEPTH_PY=${UNIDEPTH_PY:-$REMOTE_ROOT/unidepth_work/UniDepth/.venv/bin/python}
UNIDEPTH_REPO=${UNIDEPTH_REPO:-$REMOTE_ROOT/unidepth_work/UniDepth}
VGGT_PY=${VGGT_PY:-$REMOTE_ROOT/hunyuan3d_v3_env/bin/python}
COTRACKER_PY=${COTRACKER_PY:-$REMOTE_ROOT/cotracker_env/bin/python}
COTRACKER_REPO=${COTRACKER_REPO:-$REMOTE_ROOT/cotracker_work/co-tracker}
HAMER_ROOT=${HAMER_ROOT:-/dev/shm/ego_annotation_hamer_keyboard/hamer}
HAMER_CHECKPOINT=${HAMER_CHECKPOINT:-$HAMER_ROOT/_DATA/hamer_ckpts/checkpoints/hamer.ckpt}
MANO_MODEL_ROOT=${MANO_MODEL_ROOT:-/mnt/user-home/yiwen/data/dex_home/yiwen/arctic/data/body_models/mano}
FRAME_START=${FRAME_START:-702}
FRAME_END=${FRAME_END:-708}
ANCHOR_FRAME=${ANCHOR_FRAME:-705}
QUERY_FRAME_INDEX=${QUERY_FRAME_INDEX:-3}
HAND_TRACK_ID=${HAND_TRACK_ID:-visible_gloved_hand_gripping_mop}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/run_mop_702_708_v7.sh" <<EOF
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
  "$REPO_DIR/third_party/WiLoR/wilor/models/mano_wrapper.py" \
  "$MANO_MODEL_ROOT/MANO_RIGHT.pkl" \
  "$COTRACKER_REPO"; do
  if [[ ! -e "\$required_path" ]]; then
    echo "required input is missing: \$required_path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"/{sam2_object,sam2_hand,object_rgb_dataset,unidepth_full_frame,object_metric_manifest,vggt_native,annotations,hand_maskbox,hamer,wilor,hand_candidates,mano_refit,mano_mask_depth_fit,hand_selection,observed_mesh,cotracker_tracks,cotracker_edges,cotracker_pair_factors}

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
    --min-positive-hit-fraction 0.333

run_stage sam2_hand \
  "$HAWOR_PY" scripts/run_sam2_vlm_points_image.py \
    --clip "$SOURCE_CLIP" \
    --point-prompts "$HAND_PROMPTS" \
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
    --max-area-fraction 0.08 \
    --min-positive-hit-fraction 0.75

run_stage object_rgb_dataset \
  "$HAWOR_PY" scripts/export_mask_track_rgb_dataset_v3.py \
    --clip "$SOURCE_CLIP" \
    --mask-track "$OUT_ROOT/sam2_object/sam2_track.json" \
    --output-dir "$OUT_ROOT/object_rgb_dataset" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --min-mask-pixels 500 \
    --min-frames 7

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
    --min-mask-depth-pixels 500 \
    --min-frames 7

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
    --gpu 0

run_stage annotations \
  "$HAWOR_PY" scripts/build_vggt_object_skeleton_generic_v3.py \
    --video "$SOURCE_CLIP" \
    --vggt-archive "$OUT_ROOT/vggt_native/vggt_native_camera_v3.npz" \
    --mask-track "$OUT_ROOT/sam2_object/sam2_track.json" \
    --output-dir "$OUT_ROOT/annotations" \
    --object-label "long-handled floor mop" \
    --track-id "mop_01" \
    --default-caption "A gloved hand grips a long-handled floor mop while moving through an indoor room." \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END"

run_stage hand_maskbox \
  "$HAWOR_PY" scripts/build_hand_mask_box_evidence_v3.py \
    --right-track "$OUT_ROOT/sam2_hand/sam2_track.json" \
    --right-track-id "$HAND_TRACK_ID" \
    --output-json "$OUT_ROOT/hand_maskbox/hand_mask_box_evidence_702_708.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --source-width 1920 \
    --source-height 1080 \
    --mask-width 960 \
    --mask-height 540 \
    --min-mask-area-px 200 \
    --remote-output-root "$OUT_ROOT/sam2_hand" \
    --local-output-root "$OUT_ROOT/sam2_hand"

run_stage hamer_maskbox \
  "$HAWOR_PY" scripts/run_hamer_rtmlib_hand_stream_v3.py \
    --target-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --frame-manifest "$FULL_SCENE_MANIFEST" \
    --rtmlib-json "$OUT_ROOT/hand_maskbox/hand_mask_box_evidence_702_708.json" \
    --output-annotations "$OUT_ROOT/hamer/annotations_hamer_maskbox_702_708.json" \
    --output-qc "$OUT_ROOT/hamer/qc_hamer_maskbox_702_708.json" \
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
    --min-measured-hands 7

run_stage wilor_maskbox \
  "$HAWOR_PY" scripts/run_wilor_maskbox_hand_stream_v7.py \
    --target-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --frame-manifest "$FULL_SCENE_MANIFEST" \
    --maskbox-json "$OUT_ROOT/hand_maskbox/hand_mask_box_evidence_702_708.json" \
    --output-annotations "$OUT_ROOT/wilor/annotations_wilor_maskbox_702_708.json" \
    --output-qc "$OUT_ROOT/wilor/qc_wilor_maskbox_702_708.json" \
    --wilor-root "$REMOTE_ROOT/third_party/WiLoR" \
    --mano-right "$MANO_MODEL_ROOT/MANO_RIGHT.pkl" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --local-root "$LOCAL_DATA_ROOT" \
    --remote-root "$DATA_REMOTE_ROOT" \
    --device cuda:0 \
    --track-id "$HAND_TRACK_ID" \
    --side any \
    --min-measured-hands 7

run_stage merge_hand_candidates \
  "$HAWOR_PY" scripts/merge_hand_candidate_streams_v7.py \
    --base-annotations "$OUT_ROOT/annotations/annotations_v3_vggt_object_skeleton.json" \
    --hand-streams "$OUT_ROOT/hamer/annotations_hamer_maskbox_702_708.json" "$OUT_ROOT/wilor/annotations_wilor_maskbox_702_708.json" \
    --output-annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_702_708.json" \
    --output-qc "$OUT_ROOT/hand_candidates/qc_hamer_wilor_maskbox_702_708.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END"

run_stage mano_metric_refit \
  "$HAWOR_PY" scripts/refit_mano_metric_depth_v3.py \
    --annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_702_708.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-json "$OUT_ROOT/mano_refit/annotations_hamer_maskbox_metric_refit_702_708.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --min-detector-score 0.10 \
    --min-depth-joints 12 \
    --min-rows 7

run_stage mano_articulation_mask_depth_refit \
  "$HAWOR_PY" scripts/refit_mano_articulation_mask_depth_v3.py \
    --annotations "$OUT_ROOT/hand_candidates/annotations_hamer_wilor_maskbox_702_708.json" \
    --mask-track "$OUT_ROOT/sam2_hand/sam2_track.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-annotations "$OUT_ROOT/mano_mask_depth_fit/annotations_articulation_mask_depth_refit_702_708.json" \
    --output-qc "$OUT_ROOT/mano_mask_depth_fit/qc_articulation_mask_depth_refit_702_708.json" \
    --video "$SOURCE_CLIP" \
    --review-dir "$OUT_ROOT/mano_mask_depth_fit/review" \
    --mano-wrapper-root "$REPO_DIR/third_party/WiLoR" \
    --mano-model-root "$MANO_MODEL_ROOT" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --track-id "$HAND_TRACK_ID" \
    --side any \
    --source-width 1920 \
    --source-height 1080 \
    --remote-output-root "$OUT_ROOT/sam2_hand" \
    --local-output-root "$OUT_ROOT/sam2_hand" \
    --min-observations 7 \
    --still-frames "$FRAME_START" "$ANCHOR_FRAME" "$FRAME_END"

run_stage hand_selection \
  "$HAWOR_PY" scripts/select_hand_hypotheses_by_residual_v7.py \
    --annotations "$OUT_ROOT/mano_mask_depth_fit/annotations_articulation_mask_depth_refit_702_708.json" \
    --output-annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit_702_708.json" \
    --output-qc "$OUT_ROOT/hand_selection/qc_selected_hand_metric_refit_702_708.json" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END"

"$HAWOR_PY" - <<'PY'
import json
from pathlib import Path

qc_path = Path("$OUT_ROOT/hand_selection/qc_selected_hand_metric_refit_702_708.json")
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
    --annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit_702_708.json" \
    --metric-depth-npz "$OUT_ROOT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz" \
    --output-dir "$OUT_ROOT/observed_mesh" \
    --frame-start "$FRAME_START" \
    --frame-end "$FRAME_END" \
    --coordinate world \
    --intrinsics-source annotation-vggt \
    --mask-stride 1 \
    --mask-erode-px 0 \
    --min-depth-pixels 500 \
    --min-vertices 250 \
    --min-faces 300 \
    --min-frames 7 \
    --max-triangle-edge-m 0.050

run_stage cotracker_tracks \
  "$COTRACKER_PY" scripts/run_cotracker_object_tracks_v5.py \
    --manifest "$OUT_ROOT/object_metric_manifest/manifest.json" \
    --annotations "$OUT_ROOT/hand_selection/annotations_selected_hand_metric_refit_702_708.json" \
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
    "method": "v7_mop_702_708_remote_measurement_job",
    "frame_start": int("$FRAME_START"),
    "frame_end": int("$FRAME_END"),
    "outputs": {
        "object_sam2_track": str(root / "sam2_object" / "sam2_track.json"),
        "hand_sam2_track": str(root / "sam2_hand" / "sam2_track.json"),
        "object_manifest": str(root / "object_metric_manifest" / "manifest.json"),
        "metric_depth_npz": str(root / "unidepth_full_frame" / "unidepth_full_frame_depth_v3.npz"),
        "vggt_archive": str(root / "vggt_native" / "vggt_native_camera_v3.npz"),
        "annotations": str(root / "hand_selection" / "annotations_selected_hand_metric_refit_702_708.json"),
        "observed_mesh_archive": str(root / "observed_mesh" / "observed_mask_depth_meshes_world.npz"),
        "cotracker_npz": str(root / "cotracker_tracks" / "cotracker_object_tracks_v5.npz"),
        "sparse_edges_json": str(root / "cotracker_edges" / "cotracker_sparse_correspondence_edges_v6.json"),
        "pair_factors_json": str(root / "cotracker_pair_factors" / "qc_cotracker_pairwise_rigid_factors_v6.json"),
    },
}
(root / "qc_v7_mop_702_708_remote_measurement_job.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
EOF
chmod +x "$OUT_ROOT/run_mop_702_708_v7.sh"

cat > "$OUT_ROOT/wait_and_run_mop_702_708_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_mop_702_708_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_mop_702_708_v7.sh"

printf '%s\n%s\n' \
  "$OUT_ROOT/run_mop_702_708_v7.sh" \
  "$OUT_ROOT/wait_and_run_mop_702_708_v7.sh"
