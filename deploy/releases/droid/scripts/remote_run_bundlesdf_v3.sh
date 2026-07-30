#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 6 ] || [ "$#" -gt 8 ]; then
  echo "usage: $0 REMOTE_ROOT DATASET_DIR OUTPUT_DIR GPU_ID ZFAR DEBUG_LEVEL [FINAL_NERF_FRAME_MODE] [DEPTH_WEIGHT]" >&2
  exit 2
fi

ROOT="$1"
DATASET_DIR="$2"
OUTPUT_DIR="$3"
GPU_ID="$4"
ZFAR="$5"
DEBUG_LEVEL="$6"
FINAL_NERF_FRAME_MODE="${7:-bundle_keyframes}"
DEPTH_WEIGHT="${8:-0}"

BUNDLE="$ROOT/BundleSDF"
PREFIX="$ROOT/micromamba_root/envs/bundlesdf_py311"
CONFIG="$BUNDLE/BundleTrack/config_ho3d.yml"
BACKUP="$CONFIG.ego_backup"

if [ ! -d "$BUNDLE" ]; then
  echo "missing BundleSDF checkout: $BUNDLE" >&2
  exit 1
fi
if [ ! -x "$PREFIX/bin/python" ]; then
  echo "missing BundleSDF Python env: $PREFIX" >&2
  exit 1
fi
for rel in rgb depth masks cam_K.txt; do
  if [ ! -e "$DATASET_DIR/$rel" ]; then
    echo "missing BundleSDF dataset entry: $DATASET_DIR/$rel" >&2
    exit 1
  fi
done

cleanup() {
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$CONFIG"
  fi
}
trap cleanup EXIT

cp "$CONFIG" "$BACKUP"
"$PREFIX/bin/python" - "$CONFIG" "$ZFAR" <<'PY'
from pathlib import Path
import sys
from ruamel.yaml import YAML

config = Path(sys.argv[1])
zfar = float(sys.argv[2])
if zfar <= 0.0:
    raise SystemExit("zfar must be positive")
yaml = YAML()
data = yaml.load(config.read_text(encoding="utf-8"))
data["depth_processing"]["zfar"] = zfar
yaml.dump(data, config.open("w", encoding="utf-8"))
print(f"set {config} depth_processing.zfar={zfar}")
PY

export CUDA_VISIBLE_DEVICES="$GPU_ID"
TORCH_LIB="$PREFIX/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$BUNDLE/BundleTrack/build:$PREFIX/lib:$TORCH_LIB:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
P3D_TRANSFORMS="${EGO_PYTORCH3D_TRANSFORMS_PATH:-/mnt/user-home/yiwen/.cache/uv/archive-v0/GQ7Vw61ILrlJefOs}"
export PYTHONPATH="$P3D_TRANSFORMS:$BUNDLE/mycuda:$BUNDLE/BundleTrack/build:$BUNDLE/BundleTrack:$BUNDLE${PYTHONPATH:+:$PYTHONPATH}"
export EGO_BUNDLESDF_FINAL_NERF_FRAME_MODE="$FINAL_NERF_FRAME_MODE"
export EGO_BUNDLESDF_DEPTH_WEIGHT="$DEPTH_WEIGHT"

cd "$BUNDLE"
"$PREFIX/bin/python" - <<'PY'
import my_cpp
import kaolin
import torch
import open3d
import pyrender
import xatlas
from skimage import measure
print("my_cpp", my_cpp.__file__)
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
print("open3d", open3d.__version__)
print("pyrender", pyrender.__version__)
print("xatlas", xatlas.__name__)
print("skimage_marching_cubes", measure.marching_cubes.__module__)
if not torch.cuda.is_available():
    raise SystemExit("torch CUDA unavailable")
print("kaolin", kaolin.__version__)
pts = torch.rand((16, 3), device="cuda") * 2.0 - 1.0
quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=4)
octree = kaolin.ops.spc.unbatched_points_to_octree(quantized, 4, sorted=False)
print("kaolin_spc_octree_bytes", len(octree))
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

path = Path("bundlesdf.py")
text = path.read_text(encoding="utf-8")
needle = "      pdb.set_trace()\n"
if needle in text:
    path.write_text(text.replace(needle, ""), encoding="utf-8")
if "pdb.set_trace()" in path.read_text(encoding="utf-8"):
    raise SystemExit("interactive pdb breakpoint remains in bundlesdf.py")
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

path = Path("bundlesdf.py")
text = path.read_text(encoding="utf-8")
marker = """    with open(cfg_nerf_dir,'r') as ff:\n      self.cfg_nerf = yaml.load(ff)\n    self.cfg_nerf['notes'] = ''\n"""
inject = """    with open(cfg_nerf_dir,'r') as ff:\n      self.cfg_nerf = yaml.load(ff)\n    ego_depth_weight = float(os.environ.get(\"EGO_BUNDLESDF_DEPTH_WEIGHT\", str(self.cfg_nerf.get('depth_weight', 0))))\n    if ego_depth_weight < 0:\n      raise RuntimeError(\"EGO_BUNDLESDF_DEPTH_WEIGHT must be nonnegative\")\n    self.cfg_nerf['depth_weight'] = ego_depth_weight\n    print(f\"EGO BundleSDF depth_weight {ego_depth_weight}\")\n    self.cfg_nerf['notes'] = ''\n"""
if marker in text:
    text = text.replace(marker, inject)
elif inject in text:
    pass
else:
    raise SystemExit("cannot find BundleSDF depth_weight assignment")
path.write_text(text, encoding="utf-8")
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

path = Path("bundlesdf.py")
text = path.read_text(encoding="utf-8")
marker = "    logging.info(f\"keyframes#: {len(keyframes)}\")\n"
inject = """    if os.environ.get(\"EGO_BUNDLESDF_FINAL_NERF_FRAME_MODE\") == \"all_ob_in_cam\":\n      pose_files = sorted(glob.glob(f\"{self.debug_dir}/ob_in_cam/*.txt\"))\n      if len(pose_files)==0:\n        raise RuntimeError(\"all_ob_in_cam requested but no ob_in_cam pose files exist\")\n      keyframes = {}\n      for pose_file in pose_files:\n        frame_id = os.path.basename(pose_file).replace('.txt','')\n        ob_in_cam = np.loadtxt(pose_file).reshape(4,4)\n        cam_in_ob = np.linalg.inv(ob_in_cam)\n        keyframes[f\"keyframe_{frame_id}\"] = {\"cam_in_ob\": cam_in_ob.reshape(-1).tolist()}\n      logging.info(f\"EGO final NeRF all_ob_in_cam keyframes#: {len(keyframes)}\")\n\n"""
if inject not in text:
    if marker not in text:
        raise SystemExit("cannot find BundleSDF keyframe logging insertion point")
    text = text.replace(marker, inject + marker)
    path.write_text(text, encoding="utf-8")
PY

"$PREFIX/bin/python" - <<'PY'
from pathlib import Path

utils = Path("Utils.py")
text = utils.read_text(encoding="utf-8")
helper = """
def remove_duplicate_faces_compat(mesh):
  if hasattr(mesh, "remove_duplicate_faces"):
    mesh.remove_duplicate_faces()
  else:
    unique, inverse = trimesh.grouping.unique_rows(mesh.faces)
    mesh.update_faces(unique)
  return mesh

"""
if "def remove_duplicate_faces_compat(mesh):" not in text:
    marker = "\n\ndef trimesh_clean(mesh):\n"
    if marker not in text:
        raise SystemExit("cannot find trimesh_clean insertion point in Utils.py")
    text = text.replace(marker, "\n" + helper + "\ndef trimesh_clean(mesh):\n")
text = text.replace("  mesh.remove_duplicate_faces()\n", "  remove_duplicate_faces_compat(mesh)\n")
utils.write_text(text, encoding="utf-8")

nerf = Path("nerf_runner.py")
text = nerf.read_text(encoding="utf-8")
text = text.replace("    mesh.remove_duplicate_faces()\n", "    remove_duplicate_faces_compat(mesh)\n")
nerf.write_text(text, encoding="utf-8")
PY

mesh_only_stop=0
set +e
"$PREFIX/bin/python" run_custom.py \
  --mode run_video \
  --video_dir "$DATASET_DIR" \
  --out_folder "$OUTPUT_DIR" \
  --use_segmenter 0 \
  --use_gui 0 \
  --stride 1 \
  --debug_level "$DEBUG_LEVEL" &
run_pid="$!"
while kill -0 "$run_pid" 2>/dev/null; do
  if [ -f "$OUTPUT_DIR/mesh_cleaned.obj" ]; then
    mesh_size_1="$(stat -c '%s' "$OUTPUT_DIR/mesh_cleaned.obj" 2>/dev/null || echo 0)"
    sleep 3
    mesh_size_2="$(stat -c '%s' "$OUTPUT_DIR/mesh_cleaned.obj" 2>/dev/null || echo 0)"
    if [ "$mesh_size_1" = "$mesh_size_2" ] && [ "$mesh_size_2" -gt 0 ]; then
      echo "BUNDLESDF_RUN_V3_MESH_CLEANED_READY_TERMINATE_TEXTURE_PATH pid=$run_pid"
      kill "$run_pid" 2>/dev/null || true
      sleep 5
      if kill -0 "$run_pid" 2>/dev/null; then
        kill -9 "$run_pid" 2>/dev/null || true
      fi
      mesh_only_stop=1
      break
    fi
  fi
  sleep 5
done
wait "$run_pid"
run_status="$?"
set -e

test -f "$OUTPUT_DIR/config_bundletrack.yml"
test -d "$OUTPUT_DIR/ob_in_cam"
test -f "$OUTPUT_DIR/mesh_cleaned.obj"
if [ "$mesh_only_stop" -eq 1 ]; then
  echo "BUNDLESDF_RUN_V3_MESH_ONLY_AFTER_TEXTURE_PATH_TERMINATION status=$run_status"
elif [ "$run_status" -ne 0 ]; then
  if [ -f "$OUTPUT_DIR/textured_mesh.obj" ]; then
    exit "$run_status"
  fi
  echo "BUNDLESDF_RUN_V3_MESH_ONLY_AFTER_NONFATAL_TEXTURE_FAILURE status=$run_status"
else
  test -f "$OUTPUT_DIR/textured_mesh.obj"
fi

echo BUNDLESDF_RUN_V3_OK
