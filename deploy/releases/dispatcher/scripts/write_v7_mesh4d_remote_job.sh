#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=${REMOTE_ROOT:-/mnt/user-home/yiwen/ego_annotation_remote}
WORK_ROOT=${WORK_ROOT:-$REMOTE_ROOT/mesh4d_work}
REPO=${REPO:-$WORK_ROOT/Mesh4D}
OUT_ROOT=${OUT_ROOT:-$REMOTE_ROOT/v7_mesh4d_outputs}
RUNNER=${RUNNER:-$REMOTE_ROOT/scripts/remote_run_mesh4d_sequence_v7.py}
ENV_DIR=${ENV_DIR:-$WORK_ROOT/mesh4d_env}
ENV_PY=${ENV_PY:-$ENV_DIR/bin/python}
SETUP_COMPLETE=${SETUP_COMPLETE:-$OUT_ROOT/setup_complete.marker}
GPU_ID=${GPU_ID:-0}
MAX_USED_MB=${MAX_USED_MB:-2000}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_SELECT_LOCK=${GPU_SELECT_LOCK:-$REMOTE_ROOT/v7_gpu_wait_select.lock}
GPU_LOCK_DIR=${GPU_LOCK_DIR:-$REMOTE_ROOT/v7_gpu_locks}

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/setup_mesh4d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$WORK_ROOT" "$OUT_ROOT"
cd "$WORK_ROOT"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/jzr99/Mesh4D.git "$REPO"
fi
cd "$REPO"
git fetch --depth 1 origin main
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee "$OUT_ROOT/mesh4d_git_head.txt"
if ! grep -Fq "cythonize(ext_modules, force=True" "$REPO/hy3dshape/setup_im2mesh.py"; then
  perl -0pi -e "s/ext_modules=cythonize\\(ext_modules\\),/ext_modules=cythonize(ext_modules, force=True, compiler_directives={'language_level': '3'}),/" "$REPO/hy3dshape/setup_im2mesh.py"
fi
perl -0pi -e "s#im2mesh/utils/libkdtree/pykdtree/kdtree\\.c#im2mesh/utils/libkdtree/pykdtree/kdtree.pyx#g" "$REPO/hy3dshape/setup_im2mesh.py"
perl -0pi -e "s/^\\s*pykdtree,\\n//m" "$REPO/hy3dshape/setup_im2mesh.py"
perl -0pi -e "s/^from tkinter import S\\n//m" "$REPO/hy3dshape/dataset/custom_dataloader.py"
cat > "$REPO/hy3dshape/im2mesh/utils/libkdtree/__init__.py" <<'PY'
import numpy as np
from scipy.spatial import cKDTree


class KDTree:
    def __init__(self, data_pts, leafsize=16):
        self.data_pts = np.asarray(data_pts)
        self.n = int(len(self.data_pts))
        self._tree = cKDTree(self.data_pts, leafsize=int(leafsize))

    def query(self, query_pts, k=1, eps=0, distance_upper_bound=None, sqr_dists=False):
        upper = np.inf if distance_upper_bound is None else float(distance_upper_bound)
        dist, idx = self._tree.query(query_pts, k=int(k), eps=float(eps), distance_upper_bound=upper)
        if sqr_dists:
            dist = np.square(dist)
        return dist, idx


__all__ = ["KDTree"]
PY
grep -F "im2mesh/utils/libkdtree/pykdtree/kdtree.pyx" "$REPO/hy3dshape/setup_im2mesh.py"
grep -F "cythonize(ext_modules, force=True" "$REPO/hy3dshape/setup_im2mesh.py"
grep -F "pykdtree," "$REPO/hy3dshape/setup_im2mesh.py" && exit 1 || true
grep -F "from tkinter import S" "$REPO/hy3dshape/dataset/custom_dataloader.py" && exit 1 || true
python3 -m pip install --user virtualenv
rm -rf "$ENV_DIR"
rm -f "$SETUP_COMPLETE"
python3 -m virtualenv "$ENV_DIR"
"$ENV_PY" -m pip install --upgrade pip setuptools==69.5.1 wheel ninja gdown
"$ENV_PY" -m pip install Cython==3.0.12
"$ENV_PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install --no-build-isolation -r requirements.txt
"$ENV_PY" -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
"$ENV_PY" -m pip install \
  "huggingface_hub<1.0" \
  transformers==4.49.0 \
  trimesh==4.5.3 \
  scipy==1.11.4 \
  pymeshlab==2022.2.post3 \
  pycpd==2.0.0 \
  omegaconf==2.3.0 \
  munch==4.0.0 \
  plyfile==1.1.3 \
  timm==1.0.22
cd "$REPO/hy3dshape"
"$ENV_PY" ./setup_im2mesh.py build_ext --inplace
cd "$REPO"
mkdir -p "$REPO/ckpt"
if [[ ! -s "$REPO/ckpt/deform_vae.ckpt" ]]; then
  "$ENV_PY" -m gdown 1e9YGQAuFr5BDN2--srDEMnoULb4ZP5sx -O "$REPO/ckpt/deform_vae.ckpt"
fi
if [[ ! -s "$REPO/ckpt/denoiser.ckpt" ]]; then
  "$ENV_PY" -m gdown 1jeNwiP9-B1uyKvk3_yBN7Y9-D7YQxBc5 -O "$REPO/ckpt/denoiser.ckpt"
fi
"$ENV_PY" - <<'PY'
import sys
from pathlib import Path
import torch, trimesh, yaml, pymeshlab
from pycpd import RigidRegistration
repo = Path("$REPO")
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "hy3dshape"))
sys.path.insert(0, str(repo / "hy3dpaint"))
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines_video_newvae_all_nonalign_infer import Hunyuan3DDiTFlowMatchingPipeline as Mesh4DPipeline
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("Mesh4D imports", Hunyuan3DDiTFlowMatchingPipeline.__name__, Mesh4DPipeline.__name__, "trimesh", trimesh.__version__)
PY
test -s "$REPO/ckpt/deform_vae.ckpt"
test -s "$REPO/ckpt/denoiser.ckpt"
date '+%Y-%m-%d %H:%M:%S setup complete' > "$SETUP_COMPLETE"
EOF
chmod +x "$OUT_ROOT/setup_mesh4d_v7.sh"

cat > "$OUT_ROOT/run_mesh4d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="\${GPU_ID:-$GPU_ID}"
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  flock "$OUT_ROOT/setup.lock" bash "$OUT_ROOT/setup_mesh4d_v7.sh"
fi
if [[ ! -f "$SETUP_COMPLETE" ]]; then
  echo "Mesh4D setup did not produce $SETUP_COMPLETE" >&2
  exit 1
fi
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --sequence-json "$REMOTE_ROOT/v7_mesh4d_inputs/wild_rice_2538_2548/qc_mesh4d_rgba_sequence_v7.json" \\
  --sequence-dir "$REMOTE_ROOT/v7_mesh4d_inputs/wild_rice_2538_2548/DATA/ego_v7/wild_rice_2538_2548" \\
  --output-dir "$OUT_ROOT/generated/wild_rice_2538_2548" \\
  --denoiser-ckpt "$REPO/ckpt/denoiser.ckpt" \\
  --mesh4d-steps 50 \\
  --guidance-scale 5.0
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --sequence-json "$REMOTE_ROOT/v7_mesh4d_inputs/trash_0865_0870/qc_mesh4d_rgba_sequence_v7.json" \\
  --sequence-dir "$REMOTE_ROOT/v7_mesh4d_inputs/trash_0865_0870/DATA/ego_v7/trash_0865_0870" \\
  --output-dir "$OUT_ROOT/generated/trash_0865_0870" \\
  --denoiser-ckpt "$REPO/ckpt/denoiser.ckpt" \\
  --mesh4d-steps 50 \\
  --guidance-scale 5.0
"$ENV_PY" "$RUNNER" \\
  --repo "$REPO" \\
  --sequence-json "$REMOTE_ROOT/v7_mesh4d_inputs/mop_0759_0765/qc_mesh4d_rgba_sequence_v7.json" \\
  --sequence-dir "$REMOTE_ROOT/v7_mesh4d_inputs/mop_0759_0765/DATA/ego_v7/mop_0759_0765" \\
  --output-dir "$OUT_ROOT/generated/mop_0759_0765" \\
  --denoiser-ckpt "$REPO/ckpt/denoiser.ckpt" \\
  --mesh4d-steps 50 \\
  --guidance-scale 5.0
EOF
chmod +x "$OUT_ROOT/run_mesh4d_v7.sh"

cat > "$OUT_ROOT/wait_and_run_mesh4d_v7.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
RUN_SCRIPT="$OUT_ROOT/run_mesh4d_v7.sh"
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
chmod +x "$OUT_ROOT/wait_and_run_mesh4d_v7.sh"

printf '%s\n%s\n%s\n' \
  "$OUT_ROOT/setup_mesh4d_v7.sh" \
  "$OUT_ROOT/run_mesh4d_v7.sh" \
  "$OUT_ROOT/wait_and_run_mesh4d_v7.sh"
