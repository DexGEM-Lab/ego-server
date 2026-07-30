#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/pixal3d_work}
REPO=${REPO:-$WORK_ROOT/Pixal3D}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_pixal3d_prior_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_pixal3d_shape_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/pixal3d_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
SETUP_COMPLETE=${SETUP_COMPLETE:-$OUT_ROOT/setup_complete.marker}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_pixal3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/TencentARC/Pixal3D.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin master
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/pixal3d_git_head.txt"
python3 - <<'PY'
from pathlib import Path

path = Path("pixal3d/pipelines/pixal3d_image_to_3d.py")
text = path.read_text(encoding="utf-8")
needle = "import torch\n"
if "import os\nimport torch\n" not in text:
    if needle not in text:
        raise RuntimeError(f"unexpected Pixal3D import layout in {path}")
    text = text.replace(needle, "import os\nimport torch\n", 1)
old = "        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
new = (
    "        if os.environ.get('PIXAL3D_REQUIRE_PREMASKED_RGBA') == '1':\n"
    "            pipeline.rembg_model = None\n"
    "        else:\n"
    "            pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
)
if new not in text:
    if old not in text:
        raise RuntimeError(f"unexpected Pixal3D rembg construction layout in {path}")
    text = text.replace(old, new, 1)
old = (
    "        else:\n"
    "            input = input.convert('RGB')\n"
    "            if self.low_vram:\n"
    "                self.rembg_model.to(self.device)\n"
)
new = (
    "        else:\n"
    "            if self.rembg_model is None:\n"
    "                raise RuntimeError('Pixal3D V7 requires pre-masked RGBA input with non-opaque alpha; background removal is disabled')\n"
    "            input = input.convert('RGB')\n"
    "            if self.low_vram:\n"
    "                self.rembg_model.to(self.device)\n"
)
if new not in text:
    if old not in text:
        raise RuntimeError(f"unexpected Pixal3D preprocessing layout in {path}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("pixal3d_rgba_only_patch_applied")
PY
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
rm -f "$SETUP_COMPLETE"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel
"$ENV_PY" -m pip install -r requirements-hfdemo.txt
"$ENV_PY" -m pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
export HF_HOME="$WORK_ROOT/hf_cache"
export TORCH_HOME="$WORK_ROOT/torch_cache"
export ATTN_BACKEND=flash_attn_3
export SPARSE_ATTN_BACKEND=flash_attn_3
export PIXAL3D_REQUIRE_PREMASKED_RGBA=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120
"$ENV_PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="TencentARC/Pixal3D", repo_type="model", max_workers=1)
snapshot_download(repo_id="camenduru/dinov3-vitl16-pretrain-lvd1689m", repo_type="model", max_workers=1)
print("pixal3d_model_cache_ready")
PY
"$ENV_PY" - <<'PY'
import torch.hub
import flash_attn_3
import o_voxel, torch, torchvision, trimesh
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.modules.sparse import config as sparse_config
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
from PIL import Image
import numpy as np

if sparse_config.ATTN != "flash_attn_3":
    raise RuntimeError(f"Pixal3D sparse attention backend mismatch: {sparse_config.ATTN}")

probe = DinoV3ProjFeatureExtractor(
    model_name="camenduru/dinov3-vitl16-pretrain-lvd1689m",
    image_size=512,
    grid_resolution=16,
    use_naf_upsample=False,
)
probe.eval()
torch.hub.load("valeoai/NAF", "naf", pretrained=True, device="cpu", trust_repo=True)
rgba = Image.fromarray(np.dstack([np.zeros((8, 8, 3), dtype=np.uint8), np.eye(8, dtype=np.uint8) * 255]), mode="RGBA")
pipeline = Pixal3DImageTo3DPipeline.__new__(Pixal3DImageTo3DPipeline)
pipeline.low_vram = True
pipeline.rembg_model = None
pipeline.preprocess_image(rgba)
loaded = Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")
if loaded.rembg_model is not None:
    raise RuntimeError("Pixal3D RGBA-only patch did not disable rembg_model")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("torchvision", torchvision.__version__, "trimesh", trimesh.__version__)
print("flash_attn_3", flash_attn_3.__name__)
print("pixal3d_pipeline", Pixal3DImageTo3DPipeline.__name__)
print("o_voxel", o_voxel.__name__)
PY
date '+%Y-%m-%d %H:%M:%S setup complete' > "$SETUP_COMPLETE"
EOF
chmod +x "$OUT_ROOT/setup_pixal3d_v7.sh"

cat > "$OUT_ROOT/run_pixal3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
export HF_HOME="$WORK_ROOT/hf_cache"
export TORCH_HOME="$WORK_ROOT/torch_cache"
export ATTN_BACKEND=flash_attn_3
export SPARSE_ATTN_BACKEND=flash_attn_3
export PIXAL3D_REQUIRE_PREMASKED_RGBA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_pixal3d_v7.sh"
fi
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  echo "Pixal3D setup did not produce $SETUP_COMPLETE" >&2
  exit 1
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --python "$ENV_PY" \\
  --case "wild_rice_2539|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2539/frame_002539_crop_rgba.png|2539" \\
  --case "wild_rice_2545|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_frame2545/frame_002545_crop_rgba.png|2545" \\
  --case "trash_0880|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_trash_frame880/frame_000880_crop_rgba.png|880" \\
  --case "mop_0759|$REMOTE_ROOT/v7_sam3d_object_prior_inputs_mop_frame759/frame_000759_crop_rgba.png|759" \\
  --output-dir "$OUT_ROOT/generated_meshes" \\
  --resolution 1024
EOF
chmod +x "$OUT_ROOT/run_pixal3d_v7.sh"

cat > "$OUT_ROOT/wait_and_run_pixal3d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_pixal3d_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_pixal3d_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_pixal3d_v7.sh" \
  "$OUT_ROOT/run_pixal3d_v7.sh" \
  "$OUT_ROOT/wait_and_run_pixal3d_v7.sh"
