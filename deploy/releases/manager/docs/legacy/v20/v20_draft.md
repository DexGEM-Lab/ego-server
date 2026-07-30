# V20 Draft Deployment Guide

本文是 V20 观测增强的部署草案，供后续 agent 直接执行、拆任务和落地脚本。V20 不是替换 V18/V19 的新外层 wrapper；它是在当前 Pi harness + runbook + 脚本工具模式下增加新的观测候选、候选筛选、优化变量和渲染层。

## 0. 核心约束：只能补充现有框架

V20 必须遵循当前 V18/V19 的 harness 交互模式：Pi 是 harness，脚本是可调用工具，`state/` 是 renderer boundary，最终进度由 renderable physical annotation 判断。

硬约束：

1. 不替换现有 V18/V19 pipeline 顺序、run contract、state boundary、renderer 或 accepted-state 语义。
2. 新算法默认只写 `candidate`、`prior`、`observation`、`diagnostic`、`uncertainty` 或 `render_only` 记录。
3. 新记录必须通过显式 adapter 进入现有优化、因子构造、post-graph validation 或 renderer。
4. 新记录不能直接覆盖 accepted hand pose、object geometry、object pose、contact、occlusion、nonpenetration 状态。
5. 任何 generated depth、mask、point cloud、mesh 或 hand shape 都必须携带 provenance、uncertainty、validation residuals 和 promotion status。
6. 如果新模块弱于现有 V18/V19 能力，保留旧输出为 base layer，把新模块作为 additional uncertain hypothesis。
7. 重模型推理、SAM2、depth/SLAM、TRELLIS/生成式 3D、MANO fitting 和批量渲染默认在 A800/server 运行，不能未经授权在本地重 GPU 跑。

## 1. V20 与 harness 的连接方式

### 1.1 继承 V19 run contract

V20 run 仍使用 V19 的基本运行结构：

```text
v20_runs/<run_id>/
  input/
    input_manifest.json
    frames/
  measurements/
    depth_candidates/
    hand_candidates/
    object_candidates/
    masks_tracks/
    geometry_completion/
    pose_fits/
    hand_shape/
    contact_visualization/
  state/
    v20_physical_state.json
    v20_uncertainty_state.json
    v20_agent_evidence.md
  renders/
    v20_overlay.mp4
    v20_world.mp4
    v20_side_by_side.mp4
    review_frames/
  evaluation/
    metrics.json
    ablations.json
  logs/
    harness_events.jsonl
```

如果当前 renderer 仍消费 V18-compatible annotation JSON，V20 可以继续使用 V18-compatible backbone，但必须在 `state/v20_physical_state.json` 中记录 renderer consumption path，不能让私有 measurement 文件直接驱动最终可见标注。

### 1.2 新增 harness 产物

V20 harness 需要新增以下中间产物：

```text
measurements/depth_candidates/depth_candidate_registry.json
measurements/depth_candidates/depth_selection_report.json
measurements/geometry_completion/geometry_candidate_registry.json
measurements/geometry_completion/agent_conditioning_packets/*.json
measurements/hand_shape/hand_shape_solve_report.json
measurements/contact_visualization/contact_point_render_rows.json
state/v20_observation_bundle.json
```

其中：

- `*_registry.json` 只列候选，不代表接受。
- `*_selection_report.json` 记录 harness/agent 如何比较、保留、降权或拒绝候选。
- `v20_observation_bundle.json` 是进入优化/validation 的受控入口。
- `contact_point_render_rows.json` 只能进入 renderer，不能进入 contact evidence 或 contact ownership solver。

### 1.3 新增层级

V20 新增功能分为五层：

| 层级 | 名称 | 是否产生证据 | 是否可进入优化 | 是否可直接 accepted |
|---|---|---:|---:|---:|
| `measurement` | depth、stereo、RGB-D、hand mask、MANO candidates | 是 | 是 | 否 |
| `candidate_generation` | TRELLIS/条件生成 mesh、点云补齐 | 是，作为候选/先验 | 是 | 否 |
| `selector_validation` | depth selector、geometry validator、scale evaluator | 是，作为评估证据 | 是 | 否 |
| `optimization` | MANO betas/scale solve、pose/shape graph | 是，作为 posterior | 是 | 仅通过 promotion |
| `render_only` | contact point visualization | 否 | 否 | 否 |

## 2. 部署阶段总览

后续 agent 应按以下顺序实现，不允许跳过 registry/uncertainty 直接把新算法写进 accepted state。

1. **Harness schema 扩展**：增加 depth/geometry/hand-shape/contact-render 的 registry、bundle 和 state 字段。
2. **Depth candidate registry**：支持原生 depth、RGB-D、双目、单目 RGB、video/multiview depth 候选。
3. **Depth selector/evaluator**：用同一套 residual 和 agent review 评估所有 depth 候选。
4. **Geometry completion candidates**：把 TRELLIS 确实接入候选池，并新增 agent/evidence-conditioned 生成分支。
5. **Geometry validation/promotion gate**：只在 visible depth/mask/free-space/contact/nonpenetration 支持后晋升候选权重。
6. **Per-hand-track MANO shape solve**：显式优化 track-level `betas` / hand scale，不再只消费上游 betas。
7. **Contact point visualization**：只作为 renderer layer，消费已有 contact/near-contact 状态。
8. **Full render review**：输出完整 overlay/world/side-by-side，agent 必须检查可见物理标注是否更好或更不确定。

## 3. Depth 观测增强

### 3.1 目标

V20 depth 不应只假设单目 RGB。输入视频可能自带 depth、RGB-D、双目或其他校准信息。不同输入模态使用不同 depth 算法分支，但最终都进入同一个 harness depth registry 和 selector。

### 3.2 输入模态识别

新增 `input/depth_modality_report.json`，由 harness 在运行早期写出：

```json
{
  "has_native_depth": false,
  "has_rgbd_stream": false,
  "has_stereo_pair": false,
  "has_calibration": false,
  "has_camera_trajectory": false,
  "rgb_only": true,
  "source_notes": []
}
```

判断规则：

- 如果原始数据带 depth map，优先注册为 `native_depth_candidate`。
- 如果数据是 RGB-D，注册 `rgbd_depth_candidate`，并检查 RGB/depth 时间同步、分辨率、intrinsics/extrinsics 和 depth scale。
- 如果数据是双目，注册 `stereo_depth_candidate`，并先做 rectification/calibration check。
- 如果只有单目 RGB，运行 monocular/video/multiview depth 候选。

### 3.3 原生 depth / RGB-D 分支

层级：`measurement`。

目的：把传感器 depth 作为 privileged candidate，但不默认认为它正确。

需要实现或复用的 adapter：

```text
scripts/register_v20_native_depth_candidate.py
scripts/register_v20_rgbd_depth_candidate.py
scripts/evaluate_v20_depth_candidate.py
```

输出：

```text
measurements/depth_candidates/native_or_rgbd/<source_id>/depth_candidate.npz
measurements/depth_candidates/native_or_rgbd/<source_id>/depth_candidate_report.json
```

必须记录：

- frame index 对齐；
- depth unit / scale；
- intrinsics/extrinsics；
- valid mask / missing depth；
- edge confidence / flying pixel risk；
- RGB-depth registration residual；
- 与 MANO、object visible surface、contact/depth-order 的 residual。

Harness 连接：该分支写入 `depth_candidate_registry.json`，由 selector 决定它是 primary depth、辅助 depth、局部深度证据、还是 rejected candidate。

### 3.4 双目分支

层级：`measurement`。

目的：当输入是 stereo 时，不绕回 monocular RGB depth，而是使用 stereo-specific disparity/depth 方法。

候选算法类别：

- classical baseline：rectification + SGBM，用作低成本 sanity baseline；
- deep stereo：RAFT-Stereo、IGEV-Stereo、CREStereo、FoundationStereo 等类别中的可部署模型；
- temporal stereo：如果 stereo video 足够稳定，可增加时序 smoothing 或 confidence fusion。

需要新增 adapter：

```text
scripts/run_v20_stereo_depth_candidate.py
scripts/register_v20_stereo_depth_candidate.py
```

输出字段与 native/RGB-D 一致，但额外记录：

- baseline、rectification、disparity confidence；
- left-right consistency；
- occlusion/disocclusion mask；
- stereo failure regions。

Harness 连接：stereo depth 进入同一个 `depth_candidate_registry.json`，不能因为是 stereo 就自动覆盖现有 depth；仍需通过 selector 和 residual 检查。

### 3.5 单目 RGB depth 算法簇

层级：`measurement`。

当前仓库已有 UniDepth、DepthPro、VGGT 等实验或脚本痕迹，但 V20 需要把它们统一成候选簇，而不是单一后端。

建议覆盖的算法类别：

1. **Metric monocular depth**：输出 metric depth/intrinsics 的模型，例如 UniDepth、Metric3D 类。
2. **Relative/foundation depth + scale correction**：输出相对深度或强泛化 depth，再用 V20 selector 做 scale alignment，例如 Depth Anything V2、DPT/MiDaS 类。
3. **Video/temporal depth**：减少单帧 flicker，给 temporal consistency factor 提供更稳定的 depth candidate。
4. **Multi-view / geometry foundation**：用视频帧间几何估计 camera/depth/point cloud，例如 VGGT、DUSt3R/MASt3R 类，用作 camera/depth/scale 交叉证据。
5. **Existing camera/depth sources**：DROID trajectory、V16/V18 depth artifacts、masked object depth 继续作为候选或 anchors。

需要新增 registry wrapper，而不是立即重写所有 model runner：

```text
scripts/register_v20_monocular_depth_candidate.py
scripts/build_v20_depth_candidate_registry.py
scripts/select_v20_depth_observation_bundle.py
```

现有可复用入口：

- `scripts/run_unidepth_full_frame_v3.py`
- `scripts/run_unidepth_metric_source_v3.py`
- `scripts/run_vggt_native_camera_v3.py`
- `scripts/run_vggt_scene_geometry_v3.py`
- `scripts/build_v18_camera_depth_correction.py`
- `scripts/audit_v18_metric_alignment.py`

### 3.6 Depth selector / harness evaluator

层级：`selector_validation`。

这是 V20 depth 的核心新增 harness 逻辑。它不只是“选最漂亮 depth”，而是比较每个 depth candidate 对物理变量的解释能力。

输入：

- all depth candidates；
- raw RGB frames；
- camera trajectory/intrinsics candidates；
- object/hand masks；
- MANO candidates；
- visible object surfaces；
- contact/nonpenetration/depth-order residuals；
- agent visual review notes。

输出：

```text
measurements/depth_candidates/depth_selection_report.json
state/v20_observation_bundle.json
```

评价 residual：

- object mask 内 depth continuity；
- visible surface projection residual；
- MANO hand depth consistency；
- hand/object depth-order consistency；
- contact/near-contact depth gap；
- nonpenetration/free-space contradictions；
- temporal smoothness；
- known native/RGB-D/stereo calibration residual；
- agent review of obvious scale failures。

候选状态：

```text
primary_for_visible_surface
primary_for_hand_depth
secondary_scale_anchor
local_patch_only
retained_uncertain
rejected_scale_conflict
rejected_temporal_inconsistent
unresolved
```

原则：如果不同 depth 对不同区域更可信，harness 可以保留多 depth 分支，并在 `v20_observation_bundle.json` 中标记区域/用途，不强制单一 depth map 解释所有物理状态。

## 4. Segmentation completion 暂缓

用户当前决定：V20 第一阶段先不推进“分割结果增强 / 被遮挡 mask shape completion”。

处理方式：

- 保留现有 SAM2、OWLv2→SAM2、promptable SAM proposal、part mask promotion gate。
- 不新增 amodal mask completion 为 V20 第一阶段必做项。
- 如果后续恢复该方向，必须把 completed mask 明确标成 `inferred_region`，不能和 observed SAM2 mask 混用。

## 5. 点云/几何补齐

### 5.1 目标

V20 必须把已有 TRELLIS 类补齐方法确实作为候选接入流程。同时新增一个能消费 agent 理解和算法证据作为 condition 的生成式几何分支。如果 TRELLIS 当前不能直接接收这些条件，就新增并行算法分支，而不是放弃条件化生成。

### 5.2 层级和数据流

层级：`candidate_generation` + `selector_validation`。

```text
object branch decision
  -> visible mask/depth surface
  -> agent conditioning packet
  -> TRELLIS/equivalent candidate
  -> conditional generation candidate
  -> visible alignment / scale fit
  -> mask-depth-free-space-contact-nonpenetration validation
  -> geometry_candidate_registry
  -> observation bundle / pose fitting / renderer uncertainty
```

### 5.3 Agent conditioning packet

新增：

```text
measurements/geometry_completion/agent_conditioning_packets/<object_id>.json
```

字段：

```json
{
  "object_id": "object:<id>",
  "semantic_name": "...",
  "physical_branch": "rigid|articulated|deformable|support_occluder|unresolved",
  "agent_shape_description": "...",
  "visible_evidence": {
    "keyframes": [],
    "masks": [],
    "visible_surfaces": [],
    "depth_candidate_ids": []
  },
  "size_scale_hints": {
    "metric_extent_range_m": null,
    "supporting_depth_sources": []
  },
  "occlusion_notes": [],
  "contact_notes": [],
  "negative_constraints": [
    "must_not_fill_observed_free_space",
    "must_not_replace_visible_depth_surface",
    "must_not_ignore_part_required_state"
  ]
}
```

这个 packet 是给生成式模型和 harness 共用的 condition，不是 accepted geometry。

### 5.4 TRELLIS/equivalent candidate branch

必须保留：

- `scripts/remote_run_trellis_shape_v3.py`
- `scripts/build_v18_compact_rigid_trellis_completion.py`
- `scripts/build_v18_scale_sane_compact_rigid_completion.py`

V20 需要新增 adapter，把 TRELLIS 输出标准化为：

```text
measurements/geometry_completion/<object_id>/trellis_candidate/geometry_candidate.json
measurements/geometry_completion/<object_id>/trellis_candidate/completed_mesh.ply
measurements/geometry_completion/<object_id>/trellis_candidate/face_labels.json
```

标准字段：

- `candidate_id`；
- `method_family = trellis_or_equivalent_image_to_3d_prior`；
- source image/crop/prompt；
- visible-surface alignment transform；
- scale estimate；
- visible/inferred face labels；
- residual summaries；
- uncertainty；
- `accepted_geometry = false` by default。

### 5.5 条件化生成式几何分支

如果 TRELLIS 不能充分消费 agent/evidence condition，新增并行分支：

```text
scripts/run_v20_conditioned_geometry_generation.py
scripts/register_v20_conditioned_geometry_candidate.py
```

候选算法类别：

- text/image-conditioned 3D generation；
- multi-view/crop-conditioned object mesh generation；
- part-aware generation for articulated/part-required objects；
- prompt-conditioned point-cloud/mesh completion；
- existing repo-traced candidates such as Hunyuan3D、TripoSG、SPAR3D、PartCrafter 类分支。

该分支输出与 TRELLIS 相同 schema，`method_family = agent_evidence_conditioned_geometry_generation`。

### 5.6 Geometry candidate validation

新增：

```text
scripts/build_v20_geometry_candidate_registry.py
scripts/validate_v20_geometry_candidates.py
scripts/select_v20_geometry_observation_bundle.py
```

Validation residual：

- visible surface alignment residual；
- mask silhouette projection；
- depth residual against selected depth candidates；
- free-space violation；
- contact patch compatibility；
- nonpenetration residual；
- temporal pose stability；
- part/deformable branch compatibility；
- agent visual review of obvious shape mismatch。

状态：

```text
retained_prior
retained_pose_candidate
downweighted_shape_prior
rejected_visible_depth_conflict
rejected_free_space_violation
rejected_part_branch_conflict
promoted_geometry_observation
unresolved
```

Promotion 规则：只有 `promoted_geometry_observation` 才能给 object pose/contact/nonpenetration 提供强约束；其他状态只能作为 weak prior 或 render uncertainty。

## 6. Per-hand-track MANO betas / hand-scale solve

### 6.1 目标

V20 要更精确地估计视频中人的手型：手的长短、胖瘦、掌宽、指长比例和整体尺度。当前系统主要把 MANO shape/betas 当作上游视觉模型给出的候选，再用投影、depth、骨长、时序和物理 residual 验证/调权；V20 应新增显式 track-level hand shape solve。

### 6.2 可行性判断

可行，但必须作为有不确定性的优化问题，而不是逐帧直接接受 betas。

原因：

- MANO `betas` 本来是低维 shape 参数，适合跨时间共享。
- 视频中同一只手的 shape 应该在整段 track 内稳定。
- 2D keypoints、hand mask/silhouette、visible hand depth、骨长、掌宽和非穿透/contact residual 可以共同约束 shape。
- 遮挡和不可见区域不能提供精确 shape，只能扩大 uncertainty。

当前代码边界：

- `solve_v18_joint_mano_interval_trajectory.py` 消费 `betas`，但不把 `betas` 作为优化变量。
- `refit_mano_pose_contact_v3.py` 优化 pose/orient/trans/log-scale，`betas` 仍来自上游候选。
- `build_v18_hand_baseline_branch.py` 有 bone-scale consistency 组件，但它是验证/打分，不是 hand shape solve。

### 6.3 新增变量

层级：`optimization`。

每个 hand track：

```text
beta_h          # MANO betas, shared over visible/inferred track
scale_h         # optional isotropic hand scale
shape_uncertainty_h
```

每帧仍保留：

```text
pose_t, root_t, trans_t, visibility_t, contact_t, occlusion_t
```

### 6.4 目标函数

概念目标：

```text
F_hand_shape =
  F_keypoint_2d(beta_h, pose_t, trans_t)
+ F_hand_mask_silhouette(beta_h, pose_t, trans_t)
+ F_visible_hand_depth(beta_h, pose_t, trans_t)
+ F_bone_length_prior(beta_h, scale_h)
+ F_temporal_pose(beta_h fixed, pose_t, trans_t)
+ F_nonpenetration_contact(beta_h, pose_t, trans_t, object_surface)
+ F_shape_prior(beta_h, scale_h)
```

只在可见/可信区域启用 silhouette 和 depth residual；遮挡帧只参与时序和 uncertainty，不强制 shape 精确。

### 6.5 输入观测

- WiLoR/HaWoR/HaMeR MANO candidates；
- RTMLib 2D keypoints；
- SAM2 hand masks 或 visible ownership hand masks；
- selected depth candidate / visible hand depth；
- bone-length measurements from visible frames；
- object visible surfaces / contact patches / nonpenetration evidence；
- occlusion state and visibility labels。

### 6.6 输出

新增：

```text
measurements/hand_shape/<hand_track_id>/hand_shape_solve_report.json
measurements/hand_shape/<hand_track_id>/mano_betas_posterior.npz
```

写入 state：

```json
{
  "hand_track_id": "left/right/<id>",
  "betas_estimate": [],
  "scale_estimate": 1.0,
  "uncertainty": {},
  "support_frames": [],
  "occluded_frames_not_constraining_shape": [],
  "residual_summary": {},
  "promotion_status": "retained_shape_posterior_candidate"
}
```

### 6.7 Promotion 规则

`betas` posterior 可以作为后续 MANO interval solve 的 prior 或 initial state，但不能直接声称“手型精确”。

可接受条件：

- 多帧可见 hand mask/silhouette 支持；
- 2D keypoint residual 不恶化；
- visible hand depth residual 不恶化；
- bone length/hand scale 合理；
- contact/nonpenetration residual 不恶化；
- 遮挡区 uncertainty 明确；
- render review 中手宽/手指比例没有明显退化。

## 7. Contact point visualization

### 7.1 层级

`render_only`。

该功能只利用其他模块结果，不生产新证据，不进入 contact solver，不提升 contact owner。

### 7.2 输入

- existing contact/near-contact/depth-occluded-contact-possible state；
- MANO hand vertices or contact patch vertices；
- object visible/completed surface candidate；
- selected geometry/pose state；
- uncertainty radius。

### 7.3 输出

```text
measurements/contact_visualization/contact_point_render_rows.json
```

字段：

```json
{
  "frame_idx": 0,
  "hand_side": "left",
  "object_id": "object:<id>",
  "render_point_world_m": [0, 0, 0],
  "render_point_source": "center_of_local_hand_object_interface",
  "input_contact_mode": "active_physical_contact|supported_near_noncontact|depth_occluded_contact_possible",
  "source_surfaces": [],
  "uncertainty_radius_m": 0.02,
  "evidence_created": false
}
```

### 7.4 禁止事项

- 不能用 contact point 反推 contact=true。
- 不能用 contact point 作为 contact owner evidence。
- 不能把 nearest-point center 当作真实物理接触点。
- 如果 input contact mode 不支持接触/近接触，只能不画或画 unresolved marker。

## 8. Harness 需要修改的地方

### 8.1 Run contract

当前 harness 已改为 GT 隔离路径：`scripts/prepare_v20_benchmark_dataset.py` 只把 RGB/depth/calibration/object model 写入 prediction manifests，把 GT 路径写入 `evaluation/reference_manifest.json`；`scripts/evaluate_v20_benchmark_gt.py` 只在 prediction-side state/render 完成后读 GT 并评估。旧 `scripts/run_v20_benchmark_oracle_bootstrap.py` 已禁用。

新增 V20 run contract 或在 V19 contract 上追加：

- V20 新增 `depth_candidate_registry`、`geometry_candidate_registry`、`hand_shape_solve_report`、`contact_point_render_rows`。
- 所有新增候选必须进入 `state/v20_observation_bundle.json` 后才能被 optimizer/renderer 消费。
- contact visualization 明确是 render-only。
- generated geometry 明确默认不是 accepted geometry。

### 8.2 Prompt / system prompt

后续 V20 prompt 应要求 agent：

1. 先识别输入 depth modality；
2. 为每个 heavy model 声明 compute target；
3. 记录 depth/geometry/hand-shape 候选的 live mechanism 和 discriminating residual；
4. 不用单一模型输出覆盖现有 state；
5. 渲染前检查候选是否已通过 observation bundle。

### 8.3 English orchestration

需要在 V19 runbook 中插入 V20 增强步骤：

```text
Establish timeline
  -> detect input depth modality
  -> build depth candidate registry
  -> select depth observation bundle
  -> run existing hand/object/mask measurements
  -> build geometry conditioning packets
  -> run TRELLIS and conditioned geometry candidates
  -> validate geometry candidates
  -> solve per-hand-track MANO betas/scale
  -> run existing pose/contact/occlusion/nonpenetration graph
  -> add contact point render rows
  -> assemble state
  -> render full videos
```

### 8.4 State schema

`state/v20_physical_state.json` 至少新增：

```json
{
  "depth": {
    "candidate_registry": "...",
    "selected_observation_bundle": "...",
    "primary_depth_by_scope": {}
  },
  "geometry_candidates": {
    "registry": "...",
    "candidate_count": 0,
    "promoted_count": 0
  },
  "hand_shape": {
    "track_solve_reports": [],
    "accepted_shape_prior_tracks": []
  },
  "contact_visualization": {
    "render_rows": "...",
    "evidence_created": false
  }
}
```

### 8.5 Validation / adversarial review

V20 必须新增 clean-room review 问题：

- 是否有任何 generated geometry 被当成 truth？
- 是否有任何 contact point 被用作 contact evidence？
- 是否有任何 depth candidate 因视觉效果好而绕过 metric residual？
- 是否有任何 native/RGB-D/stereo depth 未检查同步、intrinsics、scale？
- 是否有任何 hand betas 在遮挡帧被过度约束？
- 是否有任何新模块覆盖了 V18/V19 已接受能力？

## 9. 当前仓库已有相近实现

### 9.1 Depth

已有：

- UniDepth full-frame / metric source；
- DepthPro/VGGT 相关实验记录；
- DROID/VGGT camera/depth/camera source；
- V18 camera depth correction；
- metric alignment audit。

不足：

- 已有 `scripts/build_v20_depth_candidate_registry.py` 和 `scripts/select_v20_depth_observation_bundle.py`，可注册prediction-side depth NPZ/image-manifest 候选并按 residual 选择；
- native/RGB-D/stereo 分支仍要求真实深度候选输出，脚本不会伪造 stereo/deep model 结果；
- selector 已计算 mask continuity、visible-surface depth、hand-depth、temporal、contact-gap residual；
- 仍需在远端重模型执行层补齐更多 depth backend 输出。

### 9.2 Geometry completion

已有：

- TRELLIS rigid prior/scale-sane completion candidate；
- depth-fused visible reconstruction；
- Poisson/convex-hull candidates；
- PCA mirror fallback；
- V19 rigid branch correction path。

不足：

- 已有 `scripts/build_v20_geometry_candidate_registry.py` 标准化 TRELLIS/equivalent/conditioned mesh 候选；
- conditioned generation branch 需要真实 server/A800 mesh 输出，registry 不再写 missing-implementation 候选；
- 已有 `scripts/validate_v20_geometry_candidates.py` 计算 visible-surface、silhouette、free-space/nonpenetration、contact compatibility 并给出 promotion status。

### 9.3 Hand shape

已有：

- 上游 MANO `betas` 存储/消费；
- hand bone-scale consistency checks；
- MANO depth/scale refit；
- interval MANO pose/translation correction。

不足：

- 已有 `scripts/solve_v20_hand_shape_track.py` 从prediction-side MANO candidates、depth refit、可见几何摘要估计 track-level betas/scale posterior；
- 当前实现把 posterior 作为后续 MANO interval solve 的 prior，不直接声称精确手型；
- 完整 MANO layer 级 silhouette/depth/bone/nonpenetration 联合 betas 优化仍是后续增强。

### 9.4 Contact visualization

已有：

- contact/nonpenetration render 和 audit；
- physical contact mode fields；
- MANO/object surface distance evidence。

不足：

- 已有 `scripts/build_v20_contact_point_render_rows.py` 从现有 hand/object surfaces 生成 render-only rows；
- `scripts/build_v20_observation_bundle.py` 要求 `evidence_created=false`，禁止 contact rows 进入 evidence graph。

## 10. 第一阶段交付清单

V20 第一阶段完成标准：

1. `docs/v20_run_contract.md` 或 V19 contract amendment 写清新增 harness 字段。
2. `docs/v20_english_orchestration.md` 写清可执行顺序和缺失实现。
3. `scripts/prepare_v20_benchmark_dataset.py` 已能为 DexYCB/YCB 与 HO3D 写出 GT-isolated prediction manifests，并把 GT 限制在 `evaluation/reference_manifest.json`。
4. `scripts/build_v20_depth_candidate_registry.py` 与 `scripts/select_v20_depth_observation_bundle.py` 已能注册prediction-side depth candidates 并按 physical residual 选择/保留/拒绝。
5. `scripts/build_v20_geometry_candidate_registry.py` 与 `scripts/validate_v20_geometry_candidates.py` 已能标准化真实 mesh 候选并用prediction-side 可见证据给出 promotion status。
6. `scripts/solve_v20_hand_shape_track.py` 已能输出 track-level betas/scale posterior candidate 和 uncertainty，作为 MANO interval solve 的 prior。
7. `scripts/build_v20_contact_point_render_rows.py` 与 `scripts/build_v20_observation_bundle.py` 已确保 contact point visualization 只在 renderer 中使用且 `evidence_created=false`。
8. `scripts/assemble_v20_state_from_v19_annotations.py` 已能把prediction-side V18/V19-compatible annotations 与 V20 sidecars 汇成 prediction state；完整 native V20 renderer 和远端重模型编排仍需后续执行集成。

## 11. 明确非目标

V20 第一阶段不做：

- 不重写 V18/V19 pipeline 为一个新 wrapper；
- 不把 generated mesh/point cloud 当作 truth；
- 不把 visible point cloud、centroid、box、mask 当作 solved object pose；
- 不把 contact point visualization 当成接触证据；
- 不把单一 depth backend 当成无条件真值；
- 不在本地重 GPU 跑模型；
- 不默认启用慢速 per-instance NeRF/SDF/training loop；
- 不推进 occluded segmentation shape completion，除非后续明确恢复该方向。
