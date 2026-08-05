# Ego 标注管线服务状态（2026-08-05 审计 + 快照切换后）

## 请求链
```
公网 :31665（平台 ingress）
 → ego-scale-caddy-1（Docker，host :8000/:8001 → 容器 :443；TLS + React 静态 + /api/* 反代）
 → ego-scale-api-1（uvicorn app.main:app；注册/api-keys/videos/annotation-jobs）
 → ego-scale-annotation-worker-1（无监听；上游 http://192.168.42.193:8093；并发 2）
 → provider PID 3319694（root tmux: ego_annotation:json-provider-v2；:8093）
 → 每 job 子进程跑算法快照 9d006c3b…（git c013e19a）
 → Ray 车道（见下）
```

## 部署身份
- Caddy 镜像 `ego-scale-caddy:async-json-api-v1-r2-20260802`（容器 /srv 内置前端资产 index-BbF-WWw8.js）
- API 镜像 `ego-scale-api:json-provider-v2-20260802`；DB `ego-scale-db-1`
- Compose 源：`/root/dex-gem-ego-scale/deploy/`（Caddyfile、Dockerfiles；**未版本化**）
- 算法适配器 release：`/home/zjh/ego_annotation_adapter_releases/json-direct-provider-v2-20260802-8ed619346f3fb0fe`
- 生产配置：`/home/zjh/ego_annotation_adapter_runtime/json-direct-provider-v2-production/{adapter.env,run.sh}`（备份 adapter.env.bak-20260805T101200Z）

## 2026-08-05 快照切换（0fc2ef10 → c013e19a）
- 旧 provider（PID 2986004，2026-08-02 启动）已终止；新 provider PID 3319694 于 10:12 UTC 启动，启动 digest 校验通过，:8093 正常监听。
- 新快照 14 项 gap/overlap 行为测试通过（20 帧排他边界、最终几何异侧分离、删除槽不回填）。
- 退化输入（30 帧视频）被适配器按设计拒绝（completed_with_warnings → annotation_failed），属既有校验行为，非回归。
- 10 秒真实视频端到端 smoke：进行中，结果见任务记录。

## Ray 车道（PID 会漂移，端口固定）
| 服务 | 端口 | GPU | 进程形态 |
|---|---|---|---|
| UniDepth | 28000 | 0/5 | dispatcher 3673782 → lanes 29000/28005 |
| hands | 28001 | 1 | proxy 2858025 → replica |
| DROID | 28002 | 多 | dispatcher → 6 lanes（29002/28007/28012/28017/28022/28027） |
| HaWoR | 28003 | 3 | proxy 2843322 → infer+infiller 两副本 |
| wilor | 28004 | 4 | proxy 2860473 → replica |
| Cosmos | 28006 | 6 | proxy 2003542 → replica |
车道 dispatcher 代码：/root/ego-recovery-evidence-20260728T153544Z/immutable-dispatcher-body-limit-1gib-20260730T091523Z（PYTHONPATH 注入）。

## 保活与恢复
- 无 systemd。provider/车道靠 tmux（root: ego_annotation 等；zjh: anno_manager_8092）；容器靠 Docker。
- **重启机器后恢复顺序**：① Docker compose up（db→api→caddy→worker）② 启动 Ray 车道与 dispatcher ③ 启动 provider：`tmux new-window -t ego_annotation -n json-provider-v2 "bash /home/zjh/ego_annotation_adapter_runtime/json-direct-provider-v2-production/run.sh"`。
- 遗留：zjh 的 8092 worktree API（PID 2924887）仍在跑但**不是**公网上游，勿误作部署目标。

## 已知缺口
1. :31665 平台映射配置不在机器上，重建需平台侧配合。
2. 公网控制平面后端（dex-gem-ego-scale/backend+deploy）未版本化——最高优先级补齐。
3. json-v1 venv 的 bin/pip shim 损坏（用 python -m pip）。
4. Cosmos 模型缓存在 /home/ylang 下，迁移需显式复制。
