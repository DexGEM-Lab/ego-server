# Ego Annotation 优化问题视角说明

本文用“优化问题/因子图”的角度解释当前仓库的 V18 标注管线：哪些是要求解的物理量，哪些观测在约束这些物理量，哪些是物理约束，哪些是人为调节的参数，以及目标函数实际在最小化什么。

## 0. 总体结论

这个仓库的核心逻辑可以理解为：

1. 视频里存在一些真实但未知的物理量，例如手的 MANO 姿态、手腕位置、物体可见表面、物体或部件姿态、接触状态、遮挡关系、相机/深度尺度。
2. 每个物理量都有多个不完美观测源，例如 HaWoR、WiLoR、RTMLib、SAM2 mask、UniDepth depth、DROID camera、visible surface、mesh distance、depth-order evidence。
3. 这些物理量必须满足一些物理约束，例如时序平滑、手不能穿进物体、接触时距离不能太远、遮挡者必须在深度上位于前方、刚体/部件运动不能随意乱跳。
4. 代码通过调节权重、阈值、边界、因子开关，把这些观测和约束放进连续或离散优化/推理问题里。
5. 求解结果再被写回 annotation，并通过后验校验和视频渲染判断：哪些可以作为物理标注，哪些只能作为候选、不确定或被拒绝的证据。

但要注意：当前 V18 不是一个“所有变量统一端到端非线性最优化”的严格系统。它更像一个分层的近似因子图系统：先生成观测和候选，再用加权时序 least-squares、局部 LBFGS、Viterbi/min-energy labeling、acceptance gate 和后验审计串起来。本文的 `F_i` 是对这些阶段的优化视角抽象，不代表代码里存在一个一次性联合最小化所有变量的全局 solver。完整 object hidden geometry 和严格 object pose 仍然没有普遍解决。

## 1. 物理量有哪些

### 1.1 手指姿态

对应 MANO pose：root orientation 加 15 个手指关节旋转，以及 MANO shape/betas。它描述“手指怎么弯、手掌朝向如何”。

当前主线更多是 HaWoR/WiLoR bridge，而不是所有帧都逐参数优化 MANO pose delta。独立的 MANO 区间轨迹脚本会优化 root/pose/translation delta，但这不是主线里所有场景都严格闭合的统一求解器。

### 1.2 手腕/手部平移

对应手腕在相机或世界坐标里的 3D 位置。手腕一动，21 个 hand joints 和 778 个 MANO vertices 会整体跟着移动。

这个量很关键，因为很多接触、遮挡、深度一致性判断都依赖手在 3D 里到底在哪里。

### 1.3 物体几何

包括 visible surface、depth-fused mesh、part visible surface、候选完整 mesh 等。

当前实现里，mesh 顶点通常不是直接优化变量。系统更多是根据 mask+depth 得到可见表面，再通过物体/部件 pose、scale、registration 或候选验证间接改变几何在世界中的位置。

### 1.4 物体/部件姿态

包括 object translation、object rotation rotvec、part SE(3)、articulation coordinate/hinge 等。

它描述“物体或部件在 3D 里在哪里、朝向如何、部件之间如何相对运动”。当前很多对象只有 visible-surface pose evidence，不等于完整刚体 pose 已解决。

### 1.5 接触状态

包括每帧每只手和每个物体之间是否接触、接触概率、contact owner、contact patch、手和物体表面距离、depth gap 等。

图像重叠不是物理接触。接触必须尽量由 3D 距离、深度顺序、nonpenetration、局部 surface/mesh evidence 一起支持。

### 1.6 遮挡状态

包括手/物体是否可见、部分可见、被遮挡、出画、不确定；以及谁遮挡谁、深度顺序是什么。

遮挡不是“检测不到所以随便补”。只有 depth/order/temporal evidence 支持时，才可以提出 occlusion owner；即使 owner 被接受，也不自动等于被遮挡的 3D pose 已经确定。

### 1.7 相机/深度校正

包括 depth scale、depth shift、bounded camera/depth correction。

这个量描述“深度后端给出的米制深度是否要整体缩放或平移”。如果深度尺度错了，手和物体看起来就会在 3D 中前后错位，接触和遮挡判断都会被污染。

### 1.8 速度/时序辅助变量

速度不是直接观测出来的物理量，而是为了让相邻帧不要跳变太大而引入的辅助量。

它的作用是表达“前一帧到后一帧应该连续变化”，而不是突然瞬移。

## 2. 送入优化问题的观测来源

这里的“观测”只指送入后续优化、因子构造、状态选择或后验校验的输入证据，不指这些优化步骤已经求出来的结果。比如 `contact owner graph`、`occlusion owner graph`、`corrected_temporal_rigid_pose_graph` 是推理/优化输出；如果后续阶段再消费它们，它们只能算“上游求解结果作为下游输入”，不能和原始模型观测混为一谈。

每类观测分成两层：

- **原生算法结果**：模型或传感/几何算法直接产出的测量，例如检测框、mask、depth、camera pose、MANO candidate、2D keypoints。
- **基于原生结果生成的推测观测**：由原生结果进一步反投影、融合、补全、拟合或构造出的候选证据，例如 visible surface、PCA mirror completion、Poisson/convex-hull mesh candidate、TRELLIS hidden-surface prior、ICP pose observation。这些仍然是优化输入，但不应被当成优化最终结论。

### 2.1 手指姿态的观测

**原生算法结果：**

- HaWoR：提供全帧 MANO pose/shape、hand joints/vertices、时序 hand candidate。
- WiLoR：提供可见帧 MANO/3D joints/778 vertices/2D projection。
- HaMeR/HandDGP/OmniHands 等候选流：如果运行，提供额外 MANO 或 hand mesh candidate。

**基于原生结果生成的推测观测：**

- HaWoR/WiLoR bridge 后的 `metric_mano_state`：把 hand candidate 映射到当前项目的 metric camera/world 语义后，作为后续优化输入。
- MANO depth/refit candidate：用 metric depth、mask、2D keypoints 对原始 MANO 候选做尺度/深度/姿态再拟合后形成的候选观测。
- `hand_observation_visibility` / visibility-weighted hand factor：根据关节深度可见性、遮挡或 depth consistency 调整 MANO 观测的权重；它是因子权重/诊断输入，不是独立产出的手姿态观测。

### 2.2 手腕/手部平移的观测

**原生算法结果：**

- HaWoR：提供 `trans_world_m`、`joints3d_camera` 等 3D hand candidate。
- WiLoR：提供 3D hand joints/vertices 和 2D projection。
- RTMLib：提供独立 2D hand keypoints。
- UniDepth/metric depth：提供手部投影位置附近的 depth 值。

**基于原生结果生成的推测观测：**

- wrist/current hand world observation：把 HaWoR/WiLoR/metric depth/camera pose 对齐后得到的 world-frame hand observation。
- hand depth shift prior：由手/物体可见表面 depth-order 冲突推导出的 camera-z 平移先验。
- occluded-hand translation posterior：遮挡或零观测时，根据 bounded depth/translation grid 得到的手部平移候选区间；它是推测观测，不是确定手位姿。

### 2.3 物体几何的观测

**原生算法结果：**

- VLM/agent visual judgment：提供物体列表、物理状态类型、part/relative-motion proposal。
- OWLv2：根据文字 prompt 找 keyframe boxes。
- SAM2：从 box/point prompt 生成 object/part masks 和 mask tracks。
- UniDepth/DepthPro/metric depth：提供 mask 像素对应的深度。
- DROID/VGGT/camera source：提供把每帧深度点放到统一坐标系所需的 camera pose。

**基于原生结果生成的推测观测：**

- mask + depth 反投影 visible surface：把 SAM2 mask 内的 depth 像素变成 3D visible point cloud/surfels。
- 多帧 depth-fused visible geometry：把多帧 visible surfaces 用 object pose/camera pose 对齐后融合成 fused point cloud。
- Poisson/convex-hull mesh candidate：从 depth-fused visible points 重建出的 mesh 候选；它是 visible-depth completion candidate，不是已接受 hidden geometry。
- PCA mirror completion candidate：把可见点云沿 PCA 隐轴镜像，得到近似 hidden point cloud candidate。
- TRELLIS/Hunyuan/TripoSG/PartCrafter/SPAR3D 等 generated prior：从图像 crop 生成完整 mesh 先验；只有经过 mask/depth/free-space/pose/contact 校验后才可能作为下游物体几何观测。
- TRELLIS hidden-surface prior / scale-sane completion：把生成式 mesh 对齐到 observed visible surface 后，保留未被 observed surfel 覆盖的部分作为 hidden-surface candidate。

### 2.4 物体/部件姿态的观测

**原生算法结果：**

- SAM2 object/part masks：提供每帧物体或部件的图像区域。
- Metric depth + mask：提供每帧 object/part visible point cloud。
- CoTracker/material tracks：提供稀疏点或材质区域在帧间的对应关系。
- Camera pose source：提供跨帧比较 object/part motion 的坐标基准。

**基于原生结果生成的推测观测：**

- visible surface centroid/PCA pose observation：从可见点云中心和 PCA 主轴得到 translation/rotvec 候选。
- ICP/registration pose observation：把 completed/generated/depth-fused mesh 或 visible point cloud 对齐到当前帧 mask/depth 后得到 SE(3)/Sim(3) 观测。
- part visible surface center/PCA observation：从 part mask + depth 得到部件中心、方向和相对运动观测。
- articulation coordinate observation：由两个或多个 part visible centers 的相对距离/圆弧/平面拟合得到的铰接候选观测。
- corrected pose graph result：这是上游优化输出；如果被后续 MANO-object constraint 或 renderer 使用，它是下游输入，但不再是原生观测。

### 2.5 接触状态的观测

**原生算法结果：**

- bbox/mask overlap：手和物体在图像里是否重叠或靠近。
- Metric depth gap：手投影区域和物体 surface/mask depth 的前后关系或间距。
- MANO vertices/joints：来自 hand candidate 的手表面几何。
- Object/part visible surface 或 mesh candidate：来自可见点云、depth-fused mesh、generated prior 等几何候选。

**基于原生结果生成的推测观测：**

- MANO-to-object/part nearest distance：由 hand vertices 和 object/part surface/mesh 计算出的 3D 距离观测。
- contact patch candidate：从近距离 hand vertices 和 object surface patch 中选出的局部接触候选区域。
- signed/triangle nonpenetration evidence：基于 mesh/triangle/normal 查询得到的局部穿透证据。
- contact likelihood / contact prior：由 overlap、depth gap、surface distance、temporal continuity 汇总出的接触先验或候选概率。
- contact owner graph 输出：这是离散推理结果；若后续阶段使用它，只能算上游接触归属推理的下游输入。

### 2.6 遮挡状态的观测

**原生算法结果：**

- hand/object masks、boxes、tracks：提供图像空间的遮挡候选区域。
- Metric depth/depth buffer：提供前后深度顺序。
- Detector/mask temporal gaps：提供短时缺失、出画或遮挡的时间线索。
- Camera pose：把跨帧遮挡关系放到统一时序和 3D 语义下。

**基于原生结果生成的推测观测：**

- depth-order evidence：由 hand/object projection 和 metric depth 比较得到“谁在前、谁在后”的候选证据。
- occlusion owner candidates：由 bbox/mask overlap、depth order、temporal gap 生成的候选遮挡者列表。
- mesh owner evidence：用 object mesh/visible surface 与 hand/object projection 比较得到的遮挡支持或反驳。
- pose-fill gate evidence：在有遮挡 owner 和 depth support 时，对被遮挡 hand pose 是否可补的候选证据。
- occlusion owner graph 输出：这是离散推理结果；如果后续 pose-fill 或 renderer 使用它，它是下游输入，不是原生观测。

### 2.7 相机/深度校正的观测

**原生算法结果：**

- DROID-SLAM：提供每帧 camera pose 或 trajectory。
- VGGT/其他 camera source：提供替代 camera trajectory 或相机尺度候选。
- UniDepth/DepthPro：提供每帧 metric depth。
- Intrinsics source：提供 `fx, fy, cx, cy`。

**基于原生结果生成的推测观测：**

- depth scale/log-scale observation：由 V16 object depth target、mask depth、scene support 等和当前 depth source 比较得到的 scale 观测。
- camera/depth correction prior：由相邻帧插值、物体深度一致性、手/物体尺度一致性构造的校正先验。
- masked object depth source：由 object mask + metric depth 汇总出的物体中心/表面深度观测，用于后续 scale、pose 或 contact 判断。

## 3. 物理约束包括哪些

### 3.1 观测一致性

求解结果不能离观测太远。比如优化后的手腕位置应该接近 HaWoR/WiLoR/RTMLib/depth 支持的位置；物体 pose 应该接近 visible surface 或 mask/depth 支持的位置。

### 3.2 时序一致性

相邻帧不能乱跳。手、物体、部件、接触状态都应随时间连续变化，除非观测强烈支持快速运动或状态变化。

### 3.3 几何一致性

物体/部件如果被当作刚体或近似刚体，它的 shape 和 relative transform 不能每帧任意变形。articulated object 的部件运动也要符合相对运动或铰接模型。

### 3.4 接触一致性

如果系统声称手和物体接触，那么它们在 3D 中应该足够接近，depth gap 不能明显矛盾，局部 surface/mesh evidence 也要支持。

### 3.5 非穿透约束

手不能明显穿进物体。当前系统用 observed surface、signed normal、nearest triangle 等局部证据做 nonpenetration 检查或惩罚；它不是完整 watertight SDF solver。

### 3.6 遮挡深度顺序约束

如果 A 遮挡 B，那么 A 在相机深度上应当更靠前。没有 depth-order 支持时，遮挡 owner 不能被强行接受。

### 3.7 有界修正约束

优化不能无限制地把手、物体或深度尺度改到任意位置。代码里有 translation bound、rotation delta bound、depth shift/scale bound、visible projection shift bound 等限制。

## 4. 参数

这里的“参数”不是手工直接改物理量，而是优化器/图推理的超参数。手工调参的含义是：改变某类观测或约束在目标函数里的影响力、容忍范围、是否启用。

### 4.1 连续优化/平滑权重

- `temporal_weight`：相邻帧平滑项的权重。
- `observation weights`：不同观测源的可信度权重。
- `translation_prior_weight`：手部平移不要偏离初始估计太多的权重。
- `root_prior_weight`：手腕/root 旋转不要偏离初始估计太多的权重。
- `pose_prior_weight`：手指关节姿态不要偏离初始估计太多的权重。
- `smooth_weight`：MANO 修正中相邻帧变量平滑的权重。
- `accel_weight`：MANO 修正中二阶加速度平滑的权重。

### 4.2 边界/阈值

- `max_translation_m`：手部平移修正的最大幅度。
- `max_root_delta_rad`：手腕/root 旋转修正的最大幅度。
- `max_pose_delta_rad`：手指关节姿态修正的最大幅度。
- `visible_shift_limit_px`：2D 投影允许偏离原始投影的像素范围。
- `depth_shift_limit_m`：手部深度允许偏离原始深度的米制范围。
- `max_object_translation_m`：可选 object translation delta 的最大幅度。

### 4.3 接触参数

- `contact_patch_weight`：接触 patch 残差的权重。
- `contact_patch_band_m`：从当前手表面中选接触候选点的距离带宽。
- `contact_patch_target_margin_m`：接触 patch 允许的残差死区。
- `contact_state_prior_residual_scale_m`：把接触概率先验转成物理残差强度的尺度。
- `contact_state_temporal_strength`：接触状态相邻帧连续性的权重。

### 4.4 非穿透/深度顺序参数

- `observed_penetration_weight`：手穿入 observed object surface 的惩罚权重。
- `dense_observed_penetration_weight`：dense observed surface barrier 的惩罚权重。
- `visible_surface_depth_order_margin_m`：手和可见第一表面之间的深度顺序安全边距。
- `visible_surface_depth_order_weight`：深度顺序残差的权重。

### 4.5 离散图参数

- `contact_tolerance_m` / owner tolerance 类参数：接受 contact owner 时允许的几何误差范围；occlusion owner 在当前代码里更依赖 `accept_mesh_support_min`、`accept_energy_margin` 和 depth-order gate。
- `switch penalty`：从一个 owner 切换到另一个 owner 的代价。
- `on/off penalty`：接触状态从开到关或从关到开的代价。
- `acceptance margin`：最优候选必须比第二候选好多少才接受。
- `mesh support threshold`：mesh/visible surface 支持达到多少才允许接受 owner。

## 5. 优化目标函数

本节里的“参数”严格指第 4 节列出的可调参数。每个目标函数还会用到优化变量和观测/因子输入，例如 `x_t`、`y_t`、`C_t`、`V_hand`；这些不是手动调的参数。下面的 `F_i = F(...; parameters)` 中，分号前是优化变量或观测输入，分号后只放第 4 节的可调参数。当前代码里的这些目标是分阶段近似目标，不是同一个全局 solver 同时最小化的单一目标。

### 5.1 主线连续时序目标

```text
F1 = F_temporal_series(x_t, y_t; observation weights, temporal_weight)
   = Σ_t observation_weight(t) ||x_t - y_t||²
   + Σ_t temporal_weight(t) ||x_t - x_{t-1}||²
```

**物理意义：** 让手、物体、部件、articulation 等时间序列既贴近观测，又不要在相邻帧突然跳变。

**优化变量：**

- `x_t`：第 `t` 帧要求解的物理量，例如 hand wrist、object SE(3)、part SE(3)、articulation coordinate。

**观测输入：**

- `y_t`：第 `t` 帧的观测值，例如 HaWoR wrist、visible surface centroid、part center。

**来自第 4 节的参数：**

- `observation weights`：每个观测源的可信度。
- `temporal_weight`：相邻帧平滑项权重。

**代码位置：** `scripts/run_v18_full_pipeline.py:4303`、`scripts/run_v18_full_pipeline.py:4380`。

### 5.2 主线 contact switch 离散目标

```text
F2 = F_contact_switch(C_t, E_on(t), E_off(t); switch penalty, on/off penalty, acceptance margin)
   = Σ_t E_t(C_t) + Σ_t T(C_{t-1}, C_t; switch penalty, on/off penalty)
```

**物理意义：** 在整段时间里判断每帧是否接触，同时避免接触状态一帧开、一帧关地抖动。

**优化变量：**

- `C_t`：第 `t` 帧的接触开关，通常是 `on/off`。

**观测/代价输入：**

- `E_on(t)`：第 `t` 帧选择接触的证据代价，来自 overlap、depth、mesh distance、nonpenetration，以及被下游消费的上游 contact-owner graph rows。
- `E_off(t)`：第 `t` 帧选择不接触的证据代价。

**来自第 4 节的参数：**

- `switch penalty`：状态或 owner 切换代价。
- `on/off penalty`：接触开关开/关切换代价。
- `acceptance margin`：最优接触解释必须比备选解释好多少才接受。

**代码位置：** 主线使用 gap-aware binary Viterbi，见 `scripts/run_v18_full_pipeline.py:6696`。

### 5.3 主线 occlusion owner 离散目标

```text
F3 = F_occlusion_owner(O_t, E_owner(t, o); switch penalty, acceptance margin, mesh support threshold)
   = Σ_t E_owner(t, O_t) + Σ_t T(O_{t-1}, O_t; switch penalty)
```

**物理意义：** 在候选遮挡者和 `none` 之间选择最合理的遮挡归属，同时要求深度顺序、mesh support、时序连续性支持该选择。

**优化变量：**

- `O_t`：第 `t` 帧选择的遮挡者，可以是某个 object，也可以是 `none`。

**观测/代价输入：**

- `E_owner(t, o)`：第 `t` 帧选择 object `o` 作为遮挡者的代价，来自 bbox overlap、mesh temporal support、depth-order evidence 和 source acceptance gate。

**来自第 4 节的参数：**

- `switch penalty`：遮挡者在相邻帧之间切换的代价。
- `acceptance margin`：最优 owner 必须比其他候选明显更好，才可被接受。
- `mesh support threshold`：mesh/visible surface 支持达到多少才允许接受 owner。
- `mesh support threshold`、`acceptance margin` 和 depth-order gate 是当前 occlusion owner 接受的主要控制量；不要把 contact owner 的 `contact_tolerance_m` 误读成 occlusion owner 的核心参数。

**代码位置：** 主线使用 object-or-none min-energy choice，见 `scripts/run_v18_full_pipeline.py:6696`。

### 5.4 MANO 区间轨迹修正总目标

```text
F4 = F_mano_interval(root_delta, pose_delta, trans_delta, object_trans_delta, contact_logit,
                     MANO/depth/surface/contact factor inputs; 第4节参数)
```

展开为主要项：

```text
F4 = F5_prior
   + F6_temporal
   + F7_bounds
   + F8_penetration
   + F9_visible_projection
   + F10_depth_hinge
   + F11_depth_order
   + F12_contact_patch
   + F13_contact_state
```

**物理意义：** 在局部时间区间里微调 MANO root、手指姿态、手部平移，并可选地微调物体平移和接触概率，使手既不严重偏离原始 hand estimate，又满足非穿透、深度顺序、接触 patch 和时序平滑。

**优化变量：**

- `root_delta`：手腕/root 旋转修正量。
- `pose_delta`：15 个手指关节旋转修正量。
- `trans_delta`：手整体 3D 平移修正量。
- `object_trans_delta`：可选的物体 3D 平移修正量。
- `contact_logit`：可选的接触概率内部变量，经过 sigmoid 得到接触概率。

**观测/因子输入：**

- 初始 MANO joints/vertices、2D projection、metric depth、visible surface track、surface eligibility、contact patch、`hand_observation_visibility`、`hand_depth_shift_prior` 等 factor rows。

**来自第 4 节的参数：**

- 连续优化/平滑权重：`translation_prior_weight`、`root_prior_weight`、`pose_prior_weight`、`smooth_weight`、`accel_weight`。
- 边界/阈值：`max_translation_m`、`max_root_delta_rad`、`max_pose_delta_rad`、`visible_shift_limit_px`、`depth_shift_limit_m`、`max_object_translation_m`。
- 接触参数：`contact_patch_weight`、`contact_patch_band_m`、`contact_patch_target_margin_m`、`contact_state_prior_residual_scale_m`、`contact_state_temporal_strength`。
- 非穿透/深度顺序参数：`observed_penetration_weight`、`dense_observed_penetration_weight`、`visible_surface_depth_order_margin_m`、`visible_surface_depth_order_weight`。

**代码位置：** `scripts/solve_v18_joint_mano_interval_trajectory.py:1575`、`scripts/solve_v18_joint_mano_interval_trajectory.py:1744`。

### 5.5 MANO 先验目标

```text
F5 = F_prior(trans_delta, root_delta, pose_delta; translation_prior_weight, root_prior_weight, pose_prior_weight)
   = translation_prior_weight · ||trans_delta||²
   + root_prior_weight · ||root_delta||²
   + pose_prior_weight · ||pose_delta||²
```

**物理意义：** 防止优化器为了满足局部约束，把手的位置或姿态改得离原始 MANO/HaWoR 估计太远。

**优化变量：**

- `trans_delta`：手部平移修正量。
- `root_delta`：root 旋转修正量。
- `pose_delta`：手指姿态修正量。

**来自第 4 节的参数：**

- `translation_prior_weight`。
- `root_prior_weight`。
- `pose_prior_weight`。

### 5.6 MANO 时序平滑目标

```text
F6 = F_mano_temporal(z_t; smooth_weight, accel_weight)
   = smooth_weight · Σ_t ||z_t - z_{t-1}||²
   + accel_weight · Σ_t ||z_t - 2z_{t-1} + z_{t-2}||²
```

**物理意义：** 让手部修正量在时间上平滑，避免一帧突然跳到另一个位置或姿态。

**优化变量：**

- `z_t`：任一 MANO 修正序列，例如 `trans_delta`、`root_delta`、`pose_delta`。

**来自第 4 节的参数：**

- `smooth_weight`。
- `accel_weight`。

### 5.7 边界 hinge 目标

```text
F7 = F_bounds(trans_delta, root_delta, pose_delta, object_trans_delta;
              max_translation_m, max_root_delta_rad, max_pose_delta_rad, max_object_translation_m)
```

展开为：

```text
F7 = hinge(||trans_delta|| - max_translation_m)
   + hinge(||root_delta|| - max_root_delta_rad)
   + hinge(||pose_delta|| - max_pose_delta_rad)
   + hinge(||object_trans_delta|| - max_object_translation_m)
```

其中 `hinge(a) = max(0, a)²`。

**物理意义：** 修正可以发生，但不能无限大；超过允许范围才开始惩罚。

**优化变量：**

- `trans_delta`：手部平移修正。
- `root_delta`：root 旋转修正。
- `pose_delta`：手指姿态修正。
- `object_trans_delta`：物体平移修正。

**来自第 4 节的参数：**

- `max_translation_m`。
- `max_root_delta_rad`。
- `max_pose_delta_rad`。
- `max_object_translation_m`。

### 5.8 非穿透目标

```text
F8 = F_penetration(V_hand, S_obj, n_obj, d_obj;
                   observed_penetration_weight, dense_observed_penetration_weight)
```

可抽象为：

```text
F8 = penetration_weight · Σ_i max(0, d_i - n_i · Δv_i)²
```

**物理意义：** 惩罚手部 MANO 顶点穿入 observed object surface。这个目标让手不要钻进物体里面。

**优化变量派生量：**

- `V_hand`：由当前 MANO 变量生成的手部 vertices。

**观测/约束输入：**

- `S_obj`：物体 observed surface 或约束面片。
- `n_obj`：物体表面法向。
- `d_obj`：当前 penetration 或约束深度。

**来自第 4 节的参数：**

- `observed_penetration_weight`。
- `dense_observed_penetration_weight`。

### 5.9 2D 可见投影 hinge 目标

```text
F9 = F_visible_projection(u_hyp, u_base; visible_shift_limit_px)
   = Σ_j max(0, ||u_hyp_j - u_base_j|| - visible_shift_limit_px)²
```

**物理意义：** 手部 3D 修正后投影回图像时，不能离原始可见 hand evidence 太远。

**优化变量派生量：**

- `u_hyp_j`：优化后第 `j` 个手关节的 2D 投影。

**观测/基准输入：**

- `u_base_j`：优化前第 `j` 个手关节的 2D 投影。

**来自第 4 节的参数：**

- `visible_shift_limit_px`。

### 5.10 深度 hinge 目标

```text
F10 = F_depth_hinge(z_hyp, z_base; depth_shift_limit_m)
    = Σ_j max(0, |z_hyp_j - z_base_j| - depth_shift_limit_m)²
```

**物理意义：** 手部 3D 修正后，关节深度不能离原始深度估计太远。

**优化变量派生量：**

- `z_hyp_j`：优化后第 `j` 个手关节的 camera-depth。

**观测/基准输入：**

- `z_base_j`：优化前第 `j` 个手关节的 camera-depth。

**来自第 4 节的参数：**

- `depth_shift_limit_m`。

### 5.11 可见表面深度顺序目标

```text
F11 = F_depth_order(z_hand, z_surface;
                    visible_surface_depth_order_margin_m, visible_surface_depth_order_weight)
```

展开为：

```text
F11 = visible_surface_depth_order_weight · Σ_i max(0, z_surface_i - visible_surface_depth_order_margin_m - z_hand_i)²
```

**物理意义：** 如果某个可见第一表面应该在手前方，手就不能错误地跑到这个表面的前面。

**优化变量派生量：**

- `z_hand_i`：手部顶点在相机坐标里的深度。

**观测输入：**

- `z_surface_i`：同一投影位置附近可见物体表面的 metric depth。

**来自第 4 节的参数：**

- `visible_surface_depth_order_margin_m`。
- `visible_surface_depth_order_weight`。

### 5.12 接触 patch 目标

```text
F12 = F_contact_patch(V_contact, P_patch, n_patch, contact_prob;
                      contact_patch_weight, contact_patch_target_margin_m)
```

可抽象为：

```text
F12 = contact_patch_weight · contact_prob · Σ_i max(0, |(v_i - p_i) · n_i| - contact_patch_target_margin_m)²
```

**物理意义：** 当接触状态可信时，让手的局部接触顶点靠近物体表面的接触 patch；当接触状态不可信时，这个目标自动变弱。

**优化变量派生量：**

- `V_contact`：被选为接触候选的 MANO vertices。
- `contact_prob`：接触概率，由 `contact_logit` 经过 sigmoid 得到。

**观测/patch 输入：**

- `P_patch`：对应的物体/部件表面 patch 点。
- `n_patch`：patch 法向。

**来自第 4 节的参数：**

- `contact_patch_weight`。
- `contact_patch_target_margin_m`。
- `contact_patch_band_m`：用于选择哪些手部 vertices 进入 `V_contact`。

### 5.13 接触状态先验/时序目标

```text
F13 = F_contact_state(C_t, C_prior_t, C_geom_t;
                      contact_state_prior_residual_scale_m, contact_state_temporal_strength)
```

可抽象为：

```text
F13 = prior_strength(contact_state_prior_residual_scale_m) · Σ_t ||C_t - C_prior_t||²
    + prior_strength(contact_state_prior_residual_scale_m) · Σ_t ||C_t - C_geom_t||²
    + contact_state_temporal_strength · Σ_t ||C_t - C_{t-1}||²
```

**物理意义：** 接触概率既要接近观测先验，也要接近几何距离支持的概率，同时在时间上不要乱闪。

**优化变量：**

- `C_t`：优化后的接触概率。

**观测/派生输入：**

- `C_prior_t`：观测给出的接触先验概率。
- `C_geom_t`：由当前 hand-to-patch 距离换算出的几何接触概率。

**来自第 4 节的参数：**

- `contact_state_prior_residual_scale_m`。
- `contact_state_temporal_strength`。

### 5.14 概念化统一残差目标（非当前主线单一 solver）

```text
F14 = F_ideal_residual(residuals; observation weights, temporal_weight, 接触参数, 非穿透/深度顺序参数)
```

其中 `residuals` 包括：

```text
r_wilor_2d,
r_rtmlib_2d,
r_metric_depth,
r_translation_prior,
r_velocity_prior,
r_bone_scale,
r_contact_prior,
r_contact_depth,
r_penetration,
r_motion,
r_acceleration,
r_contact_smooth
```

**物理意义：** 这是文档里描述的理想化统一残差向量：把 2D 重投影、metric depth、平移/速度/骨长/接触先验、接触吸引、穿透、运动平滑、加速度平滑、接触平滑都拼到一个大 residual 里。当前主线没有把这些 residual 作为一个单独全局 solver 一次性最小化。

**概念输入：**

- `residuals`：由各观测源、优化变量和物理约束计算出的残差项。

**来自第 4 节的参数：**

- `observation weights`：控制 WiLoR/RTMLib/metric depth 等观测残差权重。
- `temporal_weight`：控制 motion/acceleration/contact smoothness 这类时序残差权重。
- `contact_patch_weight`、`contact_patch_target_margin_m`、`contact_state_prior_residual_scale_m`、`contact_state_temporal_strength`：控制接触相关残差。
- `observed_penetration_weight`、`dense_observed_penetration_weight`、`visible_surface_depth_order_margin_m`、`visible_surface_depth_order_weight`：控制非穿透和深度顺序残差。

**代码/文档位置：** 这是 `docs/ego_annotation_physical_variables_and_measurements.md:147` 中的理想化表达，不完全等同于当前主线 `run_v18_full_pipeline.py` 的实际求解方式。
