# A800 external deployment dependencies

The service implementations under `deploy/releases/` are vendored snapshots.  The
following items remain A800 environment dependencies and are deliberately not
included in this repository: upstream model repositories and compiled extensions,
Python environments, checkpoints, and the Hugging Face Cosmos model cache.  The
paths and identities below are the values used by the deployment scripts.

## Upstream model repositories and extensions

| Consumer | A800 path | Identity at capture | Deployment use |
| --- | --- | --- | --- |
| DROID | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam` | Located inside git worktree `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel`, HEAD `e0a637648f587bc487b5184799a8201a07e3c536`; that worktree has local modifications. | `EGO_DROID_REPO`; upstream DROID-SLAM/compiled extension. |
| UniDepth | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/unidepth_repo` | git HEAD `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`. | `EGO_UNIDEPTH_REPO` and second `PYTHONPATH` entry. |
| HaWoR | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR` | Located inside git worktree `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel`, HEAD `e0a637648f587bc487b5184799a8201a07e3c536`; that worktree has local modifications. | `EGO_HAWOR_REPO`. |

The DROID retry-fix source tree has no `droid_slam` directory, so its driver
continues to use the explicitly recorded external DROID-SLAM dependency.  This is
an upstream/environment dependency, not a vendored service implementation.

## Python environments

| Service(s) | Python executable | Python version | Ray version |
| --- | --- | --- | --- |
| DROID, HaWoR, Infiller | `/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python` | `3.10.20` | `2.55.1` |
| Hands, WiLoR | `/home/zjh/miniconda3/envs/ray_serve_hands/bin/python` | `3.10.20` | `2.55.1` |
| UniDepth | `/home/zjh/miniconda3/envs/ray_serve_unidepth/bin/python` | `3.11.15` | `2.55.1` |
| Dispatcher, API manager | `/home/zjh/miniconda3/envs/sharpa_isaaclab/bin/python` | `3.11.15` | Ray not installed in this environment |
| Cosmos3 | `/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python` | `3.13.14` | `2.55.1` |

## Checkpoints and model cache

| Consumer | A800 path | Captured identity |
| --- | --- | --- |
| DROID | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth` | 16,061,701 bytes; SHA-256 `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526` |
| UniDepth | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected` | `config.json` (3,848 bytes), `model.safetensors` (1,415,383,604 bytes), `pytorch_model.bin` (1,415,494,210 bytes) |
| Hands YOLO | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/detector.pt` | 53,582,271 bytes; SHA-256 `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| Hands SAM2 | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/sam2.1/sam2.1_hiera_large.pt` | 898,083,611 bytes; SHA-256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| WiLoR | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/wilor_final.ckpt` | 2,564,989,533 bytes; SHA-256 `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| HaWoR | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt` | 3,267,481,572 bytes; SHA-256 `4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2` |
| Infiller | `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt` | 418,603,497 bytes; SHA-256 `30715e7e72e91d4e164bb762c7ea613dcff5448dbda5fabf40b4054e408cc5c2` |
| Cosmos3 | Hugging Face cache under `/home/ylang/.cache/huggingface`; model id `nvidia/Cosmos3-Nano` | Service code is vendored; `ego_annotation/serving/cosmos3_deployment.py` is the HF-patched `c9ed9f40` version and is covered by `RUNNING_MANIFEST.sha256`. |

`RUNNING_MANIFEST.sha256` establishes byte equality for every vendored regular
file.  It does not assert availability of the external items above; each deploy
script retains its current explicit path and fails when a required dependency is
absent.
