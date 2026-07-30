#!/usr/bin/env bash
set -euo pipefail

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable is missing: $name" >&2
    exit 1
  fi
}

for name in CASE_NAME REMOTE_OUT LOCAL_OUT FRAME_START FRAME_END MANO_RIGHT; do
  require_var "$name"
done

REMOTE_HOST=${REMOTE_HOST:-192.168.11.220}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
RUN_ROOT=${RUN_ROOT:-$LOCAL_OUT/local_acceptance}
PY=${PY:-.venv/bin/python}
SSH_CMD=${SSH_CMD:-ssh -o IPQoS=none -o ConnectTimeout=10}
MIN_TRACKS=${MIN_TRACKS:-40}
MIN_EDGES=${MIN_EDGES:-200}
MAX_PAIR_RESIDUAL_P95_M=${MAX_PAIR_RESIDUAL_P95_M:-0.010}
MAX_CORRECTION_DISPLACEMENT_P95_M=${MAX_CORRECTION_DISPLACEMENT_P95_M:-0.002}
CAPTION_PREFIX=${CAPTION_PREFIX:-V7 mesh-backed reconstruction}

mkdir -p "$LOCAL_OUT" "$RUN_ROOT"
rsync -a -e "$SSH_CMD" "$REMOTE_HOST:$REMOTE_OUT/" "$LOCAL_OUT/"

REMOTE_PREFIX="$REMOTE_OUT"
LOCAL_PREFIX="$LOCAL_OUT"
REMOTE_DATA_PREFIX="$REMOTE_ROOT/data2"
LOCAL_DATA_PREFIX="/data2"

"$PY" - "$LOCAL_OUT" "$REMOTE_PREFIX" "$LOCAL_PREFIX" "$REMOTE_DATA_PREFIX" "$LOCAL_DATA_PREFIX" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
prefix_pairs = ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5]))

def rewrite(obj):
    if isinstance(obj, dict):
        return {key: rewrite(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rewrite(value) for value in obj]
    if isinstance(obj, str):
        for remote_prefix, local_prefix in prefix_pairs:
            if obj.startswith(remote_prefix):
                return local_prefix + obj[len(remote_prefix):]
    return obj

for path in root.rglob("*.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(rewrite(payload), indent=2), encoding="utf-8")
PY

ANNOTATIONS="$LOCAL_OUT/hand_selection/annotations_selected_hand_metric_refit.json"
MANIFEST="$LOCAL_OUT/object_metric_manifest/manifest.json"
DEPTH_NPZ="$LOCAL_OUT/unidepth_full_frame/unidepth_full_frame_depth_v3.npz"
MESH_ARCHIVE="$LOCAL_OUT/observed_mesh/observed_mask_depth_meshes_world.npz"
PAIR_FACTORS="$LOCAL_OUT/cotracker_pair_factors/qc_cotracker_pairwise_rigid_factors_v6.json"
REMOTE_MEASUREMENT_REPORT="$LOCAL_OUT/qc_v7_measured_video_mesh_remote_job.json"

for required_path in "$REMOTE_MEASUREMENT_REPORT" "$ANNOTATIONS" "$MANIFEST" "$DEPTH_NPZ" "$MESH_ARCHIVE" "$PAIR_FACTORS" "$MANO_RIGHT"; do
  if [[ ! -f "$required_path" ]]; then
    echo "required acceptance input is missing: $required_path" >&2
    exit 1
  fi
done

"$PY" - "$REMOTE_MEASUREMENT_REPORT" "$CASE_NAME" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
case_name = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok":
    raise RuntimeError(f"remote measurement report is not ok: {path}: {payload.get('status')}")
if payload.get("case_name") != case_name:
    raise RuntimeError(f"remote measurement report case mismatch: {payload.get('case_name')} != {case_name}")
PY

REPLAY_DIR="$RUN_ROOT/replay"
"$PY" scripts/run_v7_video_mesh_replay_qc.py \
  --video-mesh-archive "$MESH_ARCHIVE" \
  --manifest "$MANIFEST" \
  --annotations "$ANNOTATIONS" \
  --metric-depth-npz "$DEPTH_NPZ" \
  --output-dir "$REPLAY_DIR" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --intrinsics-source annotation-vggt \
  --max-faces 0 \
  --vertex-splat-radius-px 0

TRACK_QC_DIR="$RUN_ROOT/track_surface_qc"
"$PY" scripts/check_v7_candidate_track_surface_qc.py \
  --candidate-mesh-archive "$MESH_ARCHIVE" \
  --pair-factors-json "$PAIR_FACTORS" \
  --output-dir "$TRACK_QC_DIR" \
  --frame-start "$FRAME_START" \
  --frame-end "$FRAME_END" \
  --max-track-surface-distance-m 0.012 \
  --max-pair-factor-residual-m 0.012 \
  --min-edges "$MIN_EDGES" \
  --min-tracks "$MIN_TRACKS" \
  --max-pair-residual-p95-m "$MAX_PAIR_RESIDUAL_P95_M" \
  --max-correction-displacement-p95-m "$MAX_CORRECTION_DISPLACEMENT_P95_M" \
  --fail-on-rejected

PHYSICS_DIR="$RUN_ROOT/physics"
"$PY" scripts/run_v7_candidate_physics_qc.py \
  --replay-report "$REPLAY_DIR/qc_v7_video_mesh_replay.json" \
  --annotations "$ANNOTATIONS" \
  --manifest "$MANIFEST" \
  --metric-depth-npz "$DEPTH_NPZ" \
  --output-dir "$PHYSICS_DIR" \
  --output-json "$PHYSICS_DIR/qc_v7_candidate_physics.json" \
  --intrinsics-source annotation-vggt

DELIVERABLES_DIR="$RUN_ROOT/deliverables"
"$PY" scripts/render_v7_candidate_deliverables.py \
  --replay-report "$REPLAY_DIR/qc_v7_video_mesh_replay.json" \
  --physics-report "$PHYSICS_DIR/qc_v7_candidate_physics.json" \
  --manifest "$MANIFEST" \
  --annotations "$ANNOTATIONS" \
  --mano-model "$MANO_RIGHT" \
  --output-dir "$DELIVERABLES_DIR" \
  --caption-prefix "$CAPTION_PREFIX"

"$PY" - "$RUN_ROOT" "$CASE_NAME" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
case_name = sys.argv[2]
paths = {
    "replay": (root / "replay" / "qc_v7_video_mesh_replay.json", "accepted"),
    "track_surface": (root / "track_surface_qc" / "qc_v7_candidate_track_surface.json", "accepted"),
    "physics": (root / "physics" / "qc_v7_candidate_physics.json", "accepted"),
    "deliverables": (root / "deliverables" / "v7_candidate_deliverables_manifest.json", "ok"),
}
report = {"status": "ok", "method": "sync_and_run_v7_measured_video_mesh_acceptance", "case_name": case_name, "reports": {}}
failed = {}
for name, (path, expected_status) in paths.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    report["reports"][name] = {"path": str(path), "status": status, "expected_status": expected_status}
    if status != expected_status:
        failed[name] = {"path": str(path), "status": status, "expected_status": expected_status}
if failed:
    report["status"] = "rejected"
    report["failed_reports"] = failed
out = root / "qc_v7_measured_video_mesh_acceptance_summary.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if failed:
    raise SystemExit(1)
PY
