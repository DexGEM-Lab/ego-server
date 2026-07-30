# V21 完整性补全计划

## 缺失项清单（12项，按依赖排序）

### Phase 1: 补全观测证据源（可并行）
- [ ] **1.1 UniDepth深度对比器**：用hf-mirror下载，跑单目深度第二路，与DepthPro对比
- [ ] **1.2 OWLv2 bbox物体检测**：使用 `scripts/select_v21_agent_keyframes_from_plan.py` 选关键帧，`scripts/run_v21_owlv2_bbox_proposals.py` 产出 proposals，`scripts/approve_v21_owlv2_bbox_prompts.py` 产出 approved bbox prompts；默认源仅为 OWLv2 approved bbox prompts
- [ ] **1.3 TRELLIS网格生成**：用ego_foundation env（已装spconv），下载权重，生成网格候选
- [ ] **1.4 RTMLib 2D手部关键点**：pip安装，跑2D手部检测，作为WiLoR之外的第二路证据
- [ ] **1.5 HaMeR手部重建**：用hamer env，下载checkpoint，跑手部3D重建

### Phase 2: Harness校验/融合/加权（依赖Phase 1部分完成）
- [ ] **2.1 深度候选注册+比较+选择**：build_v20_depth_candidate_registry → compare → select
- [ ] **2.2 分割主链完整运行**：用 `scripts/run_v21_sam2_proper_segmentation.py` 从 approved OWLv2 bbox prompts 跑全视频 SAM2 propagation + contamination review
- [ ] **2.3 网格候选注册+验证**：build_v20_geometry_candidate_registry → validate
- [ ] **2.4 多候选手部融合**：merge_hand_candidate_streams_v7.py 合并WiLoR+RTMLib+HaMeR

### Phase 3: V19因子图优化（核心，依赖Phase 1+2）
- [ ] **3.1 时序pose graph**：solve_v21_temporal_pose_graph.py（带surface-fit data term）跑通
- [ ] **3.2 MANO active优化器**：optimize_contact_aware_mano_graph_v8.py
- [ ] **3.3 物体因子图**：optimize_object_factor_graph_v3.py
- [ ] **3.4 网格先验pose图**：optimize_mesh_prior_pose_graph_v3.py
- [ ] **3.5 联合camera-object图**：optimize_joint_camera_object_graph_v3.py
- [ ] **3.6 联合MANO-object图**：optimize_joint_mano_object_graph_v3.py
- [ ] **3.7 接触patch pose图**：optimize_contact_patch_object_pose_graph_v3.py

### Phase 4: 接触/遮挡/非穿透（依赖Phase 3）
- [ ] **4.1 接触所有权图**：build_v18_contact_ownership_graph.py
- [ ] **4.2 遮挡owner图**：build_v18_occlusion_owner_graph.py
- [ ] **4.3 签名非穿透证据**：build_v18_signed_nonpenetration_evidence.py
- [ ] **4.4 三角非穿透证据**：build_v18_triangle_nonpenetration_evidence.py
- [ ] **4.5 MANO-object约束状态**：build_v18_mano_object_constraint_state.py
- [ ] **4.6 应用MANO-object约束**：apply_v18_mano_object_constraint_state.py

### Phase 5: V18/V19全管线渲染与原子算法审计（依赖Phase 3+4）
- [ ] **5.1 组装完整V18/V21 annotation**：将所有阶段输出合并到 `state/` renderer boundary
- [ ] **5.2 运行V18/V21全管线渲染**：overlay + world + side-by-side
- [ ] **5.3 原子算法overlay/QC审计**：运行 `scripts/audit_v21_atomic_algorithm_overlays.py`，逐项补齐 data/overlay/qc/tuning 缺口

## 执行策略
- Heavy 模型运行必须在声明的 server/A800/授权目标上执行，不在本地工作站静默启动
- Phase 1 的深度、bbox、手部候选可并行，但必须保留 compute target、参数和失败日志
- Phase 2 在 Phase 1 完成后顺序执行并写 contamination/tuning 记录
- Phase 3 逐个跑通，先验证再批量
- Phase 4 在 Phase 3 的 pose/MANO 结果上构建
- Phase 5 最后一步，消费所有输出并用 audit JSON 暴露仍未完成的原子算法
