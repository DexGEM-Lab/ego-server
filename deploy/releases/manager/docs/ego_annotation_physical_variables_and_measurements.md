# Ego Annotation: Physical Variables and Measurements

本文档汇总 V18 管线中全部物理状态变量及其测量来源、优化修正方式。

## 变量总览

```
                测量源数量           优化变量          修正方式
─────────────────────────────────────────────────────────────────
手指姿态        2 (HaWoR, WiLoR)    MANO pose         当前是bridge，非逐参数delta
手腕平移        4 (HaWoR,WiLoR,     3D shift          加法 delta
                RTMLib,Depth)
物体mesh        2 (可见表面,         不做顶点优化      通过姿态间接移动
                深度融合)
物体姿态        3 (表面时序,ICP,     平移+旋转         加法 delta + rotvec
                轮廓)
接触状态        3 (图像overlap,      contact logit     logit空间
                深度gap,表面距离)
遮挡归属        1 (depth buffer)     discrete labels   min-energy labeling
相机/深度校正   2 (SLAM,UniDepth)    log_scale+shift   乘性+加性 delta
速度            0 (辅助变量)         3D velocity       无直接测量
```

## 1. 手部姿态 (Finger Pose)

**包含的子变量：** MANO pose（45维轴角：3 root orientation + 15×3 joint rotations）、MANO betas（10维 shape）。

**测量来源：**

| 测量 | 提供 | 特性 |
|---|---|---|
| HaWoR | 全帧 MANO pose/shape，时序插值过遮挡 | V18 主要 metric MANO，有平滑偏向 |
| WiLoR | 单帧 3D 关节、778 顶点、2D 投影 | 可见帧较准，遮挡帧缺失 |

**修正方式：** 当前不做逐参数 delta 优化，而是 HaWoR/WiLoR 间 bridge——以 HaWoR 面片为基准，叠加 V18 深度 scale 修正写入 `metric_mano_state`。

## 2. 手腕平移 (Wrist Translation)

**包含的子变量：** `cam_t`（相机坐标系下手腕 3D 位置），21 个 3D 关节，778 个 3D 顶点。

**测量来源：**

| 测量 | 提供 | 残差项 |
|---|---|---|
| HaWoR | `trans_world_m`，`joints3d_camera` | 通过 bridge 转为 metric state |
| WiLoR | 21 个 3D 关节（相机坐标系），778 顶点 | 2D 重投影残差 |
| RTMLib | 21 个 2D keypoint | 2D 重投影残差（独立检测器） |
| Metric Depth (UniDepth) | 21 关节对应像素深度值 | 深度残差 |

**修正方式：** 3D 加法 delta，初始化为零：
```
cam_t = base_cam_t + shift
joints = local_joints + cam_t
vertices = local_vertices + cam_t
```
其中 `base_cam_t` 来自 HaWoR/WiLoR 测量。

## 3. 物体几何 (Object Geometry)

**包含的子变量：** 可见表面（visible surface，深度反投影点云）、刚体 mesh 假设（多帧深度融合 / TRELLIS 检索 mesh）、部件表面（每个 part 独立的可见点云）。

**测量来源：**

| 测量 | 提供 |
|---|---|
| OWLv2 + SAM2 | 物体/部件 mask（2D 分割） |
| Metric Depth + mask | 可见表面 3D 点云（mask ∩ depth → 反投影） |
| 多帧深度融合 | 注册后的 fused point cloud（简化→Poisson mesh） |
| VLM | 物理状态类型（刚体/铰接/可变形） |

**修正方式：** 不直接优化顶点位置。通过物体姿态/深度 scale 间接移动 mesh。

## 4. 物体/部件姿态 (Object/Part Pose)

**包含的子变量：** 物体平移 `object_shift`（世界坐标系 3D）、物体旋转 `object_rotvec`（小角度旋转向量）、部件 SE(3)（同上 per-part）、铰接参数（铰链轴、圆心、角度范围）。

**测量来源：**

| 测量 | 提供 |
|---|---|
| 可见表面时序 | 点云中心位移 → 平移证据 |
| CoTracker 稀疏对应 | 帧间 2D 对应 → 运动约束 |
| ICP 注册 | 帧间点云对齐 → 相对 SE(3) |
| Mask 轮廓一致性 | 投影 mesh 与 SAM2 mask 的重合度 |

**修正方式：** 平移 = 加法 delta，旋转 = 小角度 rotvec：
```
mesh_corrected = vertices + object_shift + cross(rotvec, vertices - center)
```

物体姿态当前仍是"局部 visible-surface 运动证据"，不是完整刚体 pose。几乎所有 object 保持 `object_pose_requirement_met=false`。

## 5. 接触状态 (Contact State)

**包含的子变量：** 接触概率 `contact ∈ [0, 1]`（per 帧 per 手 per 物体）、接触深度差（hand vertex Z − object depth Z）。

**测量来源：**

| 测量 | 提供 | 限制 |
|---|---|---|
| 图像 overlap（bbox/mask 交叠） | image-overlap 信号 | 不是物理接触，V18 明确标记 |
| Metric depth gap | 手顶点 Z − 物体表面 Z | 深度噪声影响判断 |
| 接触 patch 表面距离 | 手部近 mask 顶点到物体 mesh 的 3D 距离 | 需要手/物都有可用 3D 几何 |

**修正方式：** 在 logit 空间优化，映射到 [0,1]：
```
contact = sigmoid(logit)
logit_prior = log(p0 / (1-p0))    # p0 来自 contact_seed
```
当 contact 高时激活深度吸引项 `√contact × gap/sigma`；低时不生效。

## 6. 遮挡归属 (Occlusion Owner)

**包含的子变量：**
- `hand_visibility`: visible / partially_visible / occluded / out-of-frame / unresolved
- `object_visibility`: 同上
- `occlusion_owner`: 哪个物体遮挡了哪个手/物体
- `depth_order`: 深度顺序

**测量来源：** Depth buffer 比较、Bbox 重叠分析、时序 gap 分析（短暂检测缺失可能是遮挡）。

**修正方式：** 离散变量，min-energy labeling。每个遮挡关系由 depth-order evidence 支持或反驳。

## 7. 相机/深度校正 (Camera/Depth Correction)

**包含的子变量：** 深度 scale `exp(log_scale)`（per 帧）、深度 shift（per 帧）、相机内参校正（bounded）。

**测量来源：**

| 测量 | 提供 |
|---|---|
| DROID-SLAM | `T_world_camera` (4×4 per frame) |
| UniDepth | metric depth 值 |
| V16 物体中心深度 vs 后端深度 | 比值 → 深度 scale 证据 |

**修正方式：** 乘性 scale + 加性 shift：
```
depth_corrected = depth_measured × exp(log_scale) + shift
```

## 8. 速度 (Velocity)

**辅助变量：** `velocity`（手部平移速度 3D 向量，per 帧）。

无直接测量源，纯粹为时序平滑引入：`shifts[b] ≈ shifts[a] + velocity[a] × dt`。

## 损失函数

所有归一化残差拼接为单一向量，用 `scipy.optimize.least_squares(loss="soft_l1")` 求解：

```python
residual = concat(
    # 帧级测量项
    (WiLoR_2D_residual) / sigma_wilor_reprojection_px,      # 2D 重投影
    (RTMLib_2D_residual) / sigma_rtmlib_reprojection_px,    # 独立 2D 锚点
    (MANO_depth - metric_depth) / sigma_metric_depth_m,      # 深度一致性
    shift / sigma_translation_prior_m,                       # 平移先验（归零）
    velocity / sigma_velocity_prior_m,                       # 速度先验
    (bone_scale - prior) / sigma_bone_scale,                 # 骨长先验
    (logit - logit_prior) / sigma_contact_logit,             # 接触先验
    √contact × gap / sigma_contact_depth,                    # 接触深度吸引
    min(gap, 0) / sigma_penetration,                         # 穿透惩罚

    # 时序项
    (shift[b] - predicted) / sigma_motion,                   # 运动预测
    (velocity[b] - velocity[a]) / sigma_acceleration,        # 加速度平滑
    (logit[b] - logit[a]) / sigma_contact_step,              # 接触状态平滑
)
```

每个 sigma 编码该约束的信任程度。`soft_l1` 对大残差项（outlier）自动降权，防止单帧坏测量绑架全局解。
