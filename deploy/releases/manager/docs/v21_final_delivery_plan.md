# V21 交付物重建与刷榜计划

## 问题诊断

### 1. 分割主链需要按最新清单重跑
- 当前 V21 active 设计必须从 agent-selected keyframes 进入 OWLv2 bbox proposals
- 必须产出 `owlv2_bbox_approved_prompts.json`，由审核确认目标框后再喂给 SAM2
- SAM2 输出必须是 `sam2_proper_summary.json` 和 `sam2_proper/<object_id>/sam2_masks/*.png`
- 分割进入几何前必须通过 `review/segmentation_sam2_proper/segmentation_contamination_review.json`
- **必须完成授权 heavy run 和视觉审核，达到像素级精度**

### 2. 深度没有利用双目
- 只在左眼跑了DepthPro，没有跑右眼
- 没有做真正的立体匹配（stereo matching）
- 立体深度可以提供度量尺度，校准单目深度
- **必须对两个视角都跑深度，并做合理选取**

### 3. 没有参数精调
- DepthPro、UniDepth、SAM2等算法都用默认参数
- 没有评估-调参-重跑的循环
- **必须实现参数精调循环，榨干每个算法性能**

### 4. 渲染不符合V18标准
- 当前只有简单overlay（骨架线+框）
- 需要两个窗格：(1)手骨架+半透明MANO点云overlay, (2)世界坐标系3D点云mesh
- **必须按V18标准重新渲染**

## 实施计划

### Phase 1: 深度修正（立体+单目）
1. 对左右眼分别跑DepthPro、UniDepth V2
2. 跑立体匹配（RAFT-Stereo或SGBM）获取度量深度
3. 对比单目vs立体深度，选取最优
4. 参数精调：调整DepthPro分辨率、focal prior等

### Phase 2: 分割重做（像素级精度）
1. 用 agent object plan 选择 OWLv2 detector keyframes，产出 `segmentation_stable_keyframes.json`
2. 用 OWLv2 在关键帧检测目标 bbox，产出 `owlv2_bbox_proposals.json`；GroundingDINO 不作为默认 bbox 源
3. 由 agent/审核确认目标框，产出 `owlv2_bbox_approved_prompts.json`
4. 用 SAM2.1 bbox prompt 视频传播做全视频分割，产出 `sam2_proper_summary.json` 和 `sam2_proper/<object_id>/sam2_masks/*.png`
5. 用 contamination review 和深度边缘辅助验证分割边界；检查是否覆盖物体且不包含背景
6. 参数精调：调整 keyframes、OWLv2 text/threshold、SAM2 box prompt/propagation 参数，并写入 `tuning/segmentation/`

### Phase 3: V18标准渲染
1. 窗格1：原始视频 + 细骨架 + 半透明MANO点云
2. 窗格2：世界坐标系3D点云mesh（手+物体）
3. 组合成side-by-side视频

### Phase 4: Benchmark刷榜
1. DexYCB: 评估→调参→重跑→再评估
2. HO3D: 同上
3. 优化到收敛（误差不再下降）

## 交付物保存位置
outputs/v21_final_deliverables/
  pico_infer/
    v21_side_by_side.mp4
    v21_overlay.mp4
    v21_world.mp4
  living_room_infer/
    v21_side_by_side.mp4
    v21_overlay.mp4
    v21_world.mp4
  benchmark_dexycb/
    evaluation_report.json
    v21_side_by_side.mp4
  benchmark_ho3d/
    evaluation_report.json
    v21_side_by_side.mp4.mp4
