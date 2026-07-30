# HaWoR exact-source recovery amendment

## Decision

The h3e1 HaWoR launch must import only the immutable bundle derived from:

- candidate: `/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-feat-parallel/third_party/algorithms/hawor`
- upstream repository: `ThunderVVV/HaWoR`
- required Git HEAD: `66c7d4108d58a716deccd192cb7645170cdc7bd7`
- amendment id: `recovered-hawor-core-exact-v1`

This candidate is exact-justified because the GPU3 healthy worker's
`runtime_env_agent.log` selected this exact path in `PYTHONPATH` while running
`/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python`, and its Serve replica
subsequently recorded successful HaWoR requests. The source provenance record is
`/vePFS-Mindverse/user/yiwen/user-home/zjh/evidence/hawor-source-provenance-20260721T171710Z/REPORT.json`
(SHA-256 `3dea41b437d76959afeb6aa0a0b8f5ea34af4b9a7a7dac73e14385725f479268`).

## Adapter import contract

`hawor_deployment.py` constructs `HaWoRAdapter` and `InfillerAdapter`. Their only
upstream loader imports are:

1. `scripts.scripts_test_video.hawor_video:load_hawor`;
2. `infiller.lib.model.network:TransformerModel`.

The exact adapter-module closure pinned in the source manifest is:

```text
scripts/scripts_test_video/hawor_video.py
hawor/configs/__init__.py
hawor/utils/{geometry,process,pylogger,render_openpose,rotation}.py
infiller/lib/model/network.py
lib/core/constants.py
lib/datasets/track_dataset.py
lib/eval_utils/{custom_utils,filling_utils}.py
lib/models/backbones/{__init__,vit}.py
lib/models/components/{__init__,pose_transformer,t_cond_mlp}.py
lib/models/{hawor,mano_wrapper,modules}.py
lib/pipeline/{__init__,tools}.py
lib/utils/{geometry,imutils}.py
lib/vis/{renderer,tools}.py
```

The ordered closure hash is
`26c2bf6e4c7b8c252941fbfaed414ba2c119b35507e3bf82c2893bab16cb6a64`.
Per-file SHA-256 values are in the release manifest and are checked against the
provenance-pinned exact values by `ego_annotation.serving.hawor_source`.

## Release and h3e1 launch contract

Build with `scripts.build_hawor_source_release` into the fresh zjh-owned root
`/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/hawor_source_releases`.
The builder strips cache/`.pyc`/`.pyo` material and rejects it if injected into a
published release. It rejects every candidate symlink except the two recorded MANO
asset links, whose target bytes are materialized as regular immutable files in the
release; any release symlink is rejected. It also rejects hard-linked manifest
files, module-hash drift, incomplete manifests, and copy-time mutation. It
publishes a read-only directory named by its complete-tree digest.

The built exact h3e1 replacement is:

```bash
EGO_HAWOR_REPO=/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_experiments/hawor_source_releases/939510597333bd2a42dfdce676e0b0a0807227a420aa4f87fb062697d1c2dbf3
PYTHONDONTWRITEBYTECODE=1
```

Its complete-tree digest is
`939510597333bd2a42dfdce676e0b0a0807227a420aa4f87fb062697d1c2dbf3`; its
manifest contains 462 regular files and no symlink/cache/bytecode entries.

The bundle is validated only by imports under the exact HaWoR interpreter. This
amendment does not authorize a model load, Ray deployment, GPU use, or soak.
