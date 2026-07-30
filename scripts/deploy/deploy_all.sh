#!/usr/bin/env bash
# ==============================================================================
# A800 ego annotation service topology (run this script on A800)
#
# Python environments:
#   HaWoR + DROID : /home/zjh/miniconda3/envs/ray_serve_hawor/bin/python
#   Hands + WiLoR : /home/zjh/miniconda3/envs/ray_serve_hands/bin/python
#   UniDepth      : /home/zjh/miniconda3/envs/ray_serve_unidepth/bin/python
#   Dispatcher/API: /home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python
#   Cosmos3       : /home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python
#
# Weights/checkpoints:
#   DROID     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth
#   UniDepth  : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected
#   Hands YOLO: /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/detector.pt
#   Hands SAM2: /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/sam2.1/sam2.1_hiera_large.pt
#   WiLoR     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/wilor_final.ckpt
#   HaWoR     : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt
#   Infiller  : /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt
#   Cosmos3   : nvidia/Cosmos3-Nano, cached under /home/ylang/.cache/huggingface
#
# Vendored service code is resolved by each deploy script under deploy/releases/.
# External environments, upstream model repositories, and checkpoints are pinned
# in deploy/DEPENDENCIES.md.
#
# GPU / Ray head / Serve HTTP mapping:
#   GPU0 UniDepth       26000 / 29000 (dispatcher publishes it on 28000)
#   GPU1 Hands          27000 / 28001
#   GPU2 DROID owner    26400 / 29002
#        DROID importer 26420 / 28012
#        DROID importer 26440 / 28022
#   GPU3 HaWoR+Infiller 26600 / 28003
#   GPU4 WiLoR          27200 / 28004
#   GPU5 UniDepth       30800 / 28005 (moved from 26800 to avoid Cosmos3)
#   GPU6 Cosmos3        26801 / 28006 (Ray dashboard 26800)
#   GPU7 DROID owner    30000 / 28007
#        DROID importer 30020 / 28017
#        DROID importer 30040 / 28027
#   CPU  lane_dispatcher public ports 28002 (DROID), 28000 (UniDepth)
#   CPU  API manager    8092
#
# All Ray heads are independent. DROID BA concurrency is fixed at 1. DROID IPC
# owners are r0 (29002 and 28007); owners must start before and stop after importers.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

log() { printf '[deploy-all] %s\n' "$*"; }
for service in unidepth hands wilor droid hawor cosmos dispatcher manager; do
  log "deploying $service"
  "$SCRIPT_DIR/deploy_${service}.sh"
done
log "all ego annotation services are healthy"
