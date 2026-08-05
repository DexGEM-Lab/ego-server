# Ego 标注管线环境说明（A800 标杆，2026-08-05）

## 硬件与平台
- 8 × NVIDIA A800-SXM4-80GB；Docker（Compose）、tmux、Ray 为运行时三件套；**无 systemd 单元**。
- 公网入口：`https://115.190.235.210:31665` 是**外部平台 ingress/NAT** → 宿主机 `:8000`（Docker Caddy 容器）。重建时需在新服务器平台侧重建该映射。

## 代码仓库（三部分）
| 部分 | 仓库 | 部署路径 | 说明 |
|---|---|---|---|
| 前端管理栈（含控制平面后端） | github.com/DexGEM-Lab/ego-frontend | /root/dex-gem-ego-scale | 整个 ego-scale：backend（FastAPI 注册/API Key/annotation-jobs/worker）+ frontend（Vite/React/TS）+ deploy（Caddyfile/Dockerfile）+ compose.yaml。前端 pnpm 10.6.2 / node 22.22.3。密钥 .env 与 .local-data 不入库。 |
| 后端流程（算法） | github.com/DexGEM-Lab/ego_annotation（分支 ego_annotation_worktree） | /vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation_worktree | 线上不直接跑该目录，跑其不可变快照 |
| 服务端 | github.com/DexGEM-Lab/ego-server | /vePFS-Mindverse/user/yiwen/user-home/zjh/ego-server | 算法上游服务代码 |

> 说明：公网控制平面后端（注册/API Key/异步任务/队列）与 deploy/compose 已随整个 `dex-gem-ego-scale` 收入 **ego-frontend** 仓库（commit 44f7d5c）；`.env`、`.local-data/` 等按仓库 `.gitignore` 排除。三仓库即可完整重建。

## Python 环境
| 环境 | Python | 用途 | 关键包 |
|---|---|---|---|
| ~/miniconda3/envs/sharpa_isaaclab | 3.11.15 | 算法管线 + 适配器主环境 | torch 2.7.0, fastapi 0.115.7, uvicorn 0.29.0, opencv-headless 4.11, numpy 1.26, trimesh 4.5.1 |
| ~/miniconda3/envs/ray_serve_hands | 3.10 | hands 检测（GPU1）+ wilor（GPU4） | — |
| ~/miniconda3/envs/ray_serve_hawor | 3.10 | HaWoR（GPU3） | — |
| ~/miniconda3/envs/ray_serve_unidepth | — | UniDepth / DROID 车道 | — |
| uv python 3.13 | 3.13 | Cosmos（GPU6） | 模型在 /home/ylang/.cache/huggingface（**跨用户路径，迁移时需复制**） |
| ~/ego_annotation_adapter_envs/json-v1 | 3.11 | 适配器 venv 包装（**bin/pip shim 已坏，用 python -m pip**） | — |

## 模型资产
`/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/`：wilor 2.5G（wilor_final.ckpt + detector.pt）、hawor 3.5G（hawor.ckpt + infiller.pt）、unidepth 2.7G、sam2.1 857M、mano 7.3M；HF 缓存 unidepth-v2-vitl14 1.4G。

## 算法快照机制
线上算法 = worktree 的**不可变快照**，位于 `/home/zjh/ego_annotation_snapshots/<digest>-<utc>/`，digest 算法见适配器 `serve_worktree_annotation_json_api.py:_source_digest`（文件/符号链接的路径+可执行位+内容 sha256；排除 .git/.memory/ego_annotation_runs/__pycache__/.pytest_cache/SOURCE_MANIFEST.json/pyc）。**注意与 provider_builds 里的 compute_snapshot_digest.py 不同（后者含完整权限位，不要混用）；顺序：复制→适配器 digest→写 manifest→chmod 只读→按 digest 改名。**
当前钉在：`9d006c3b70a45aa19ab3f1a5ccdf37b25cb03197a8075b2c6abd67aa5f94e1d0-20260805T100927Z`（git c013e19a = 66fc4866 算法 + 交付版渲染器）。

## A800→GitHub 推送
A800 本机无 GitHub SSH 权限。既定路径：A800 上 `git bundle create` → scp 到有权限的机器 → 该机器 fetch bundle 后 `git push`。worktree 普通 `git commit` 有 reflog 权限问题，用 `git write-tree` + `git commit-tree` + 直接写 `refs/heads/<branch>`。
