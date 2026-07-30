# Ego Annotation V18 Algorithm Pipeline Flow

本文档描述当前仓库 `master` 分支的 V18 主线流水线。它不是逐函数 trace，而是把当前代码路径压缩成 **19 个粗粒度算法步骤**，用于读懂系统如何从缓存/视频证据生成最终 renderable annotation。编号按当前交付路径的代码执行顺序排列；每个步骤可能对应多个脚本、多个 helper，或 `run_v18_full_pipeline.py` 中的一段连续调用。

源码边界：

- measured/status evidence path：`scripts/run_v18_measured_status_pipeline.py`
- final artifact path：`scripts/run_v18_full_pipeline.py`
- 版本说明：`docs/pipeline_v18.md`

## 一句话概括

V18 把第一视角视频、V16/V17/V18 缓存感知结果、LLM/VLM 物理语义、hand MANO evidence、segmentation/part evidence、visible/depth-fused geometry、contact/occlusion/nonpenetration evidence 融合成逐帧手、物体、部件、接触、遮挡和不确定性标注，最后输出完整时长的 `annotations_v18_full.json`、overlay/world/side-by-side 视频、QC 和 reports。

## 19 步粗粒度流程

| 顺序 | 粗粒度步骤 | 类型 | 代码位置 | 做什么 | 关键输出/消费 |
|---:|---|---|---|---|---|
| 1 | Runtime + V16/camera-depth input contract | 支持/admin | `runtime_manifest`，V16 inputs，camera/depth roots | 建立 raw/V16/camera/depth/runtime 基础，不生成新标注语义 | raw frames、V16 annotations/renders、camera/depth support |
| 2 | LLM/VLM physical schema | 核心生成 | `build_v18_physical_state_schema.py` | 把 model physical notes 结构化为 object physical state、part/relative-motion requirement | `v18_physical_state_schema_report.json` |
| 3 | Hand evidence reducer | 支持 | `build_v18_hand_baseline_branch.py` | 归并 cached WiLoR、HaWoR、RTMLib、interior hand/depth evidence；这不是现场执行 HaWoR | `v18_hand_baseline_branch.json` |
| 4 | Visibility/occlusion + bounded reducers | 支持/校验 | `visibility_occlusion_state`、`fast_motion_state`、`consistency_graph`、`bounded_state_solution` | 生成 visibility、occlusion、motion、bounded contact/pose 状态，保留不确定性 | visibility/occlusion rows、bounded solution |
| 5 | Annotation state base | 核心生成 | `build_v18_annotation_state.py` | 把 timeline、reducers、基础 hand/object/contact rows 汇成逐帧 annotation state | `v18_annotation_state.json` |
| 6 | Visible object geometry + completion gate | 核心生成 + 校验 | `visible_geometry_archive`、`object_completion_gate` | 从 cached visible surfaces 得到 object visible geometry，并判断 hidden geometry/completion eligibility | object visible geometry archive、completion blockers |
| 7 | OWLv2/SAM2 part segmentation | 核心生成 | `build_v18_owlv2_sam2_part_tracks.py` | 用 VLM terms 生成 OWLv2 keyframe boxes，并用 SAM2 做 part video tracking | accepted/rejected part tracks、mask evidence |
| 8 | Part mask support gates | 支持/校验 | `part_track_source_manifest`、`sam_promptable_part_proposals`、`part_mask_acquisition_plan`、`part_mask_promotion_gate` | 记录 mask 来源、probe promptable SAM、阻止未验证 proposal 晋升 | source manifest、acquisition/promotion gates |
| 9 | Part surfaces + articulation/SE(3) evidence | 核心生成 + 校验 | `part_split_evidence`、`part_visible_surfaces`、`part_motion_state`、`part_motion_qc`、`part_model_candidates`、`articulation_fit_candidates`、`part_se3_surface_residuals`、`visible_part_subset_archive`、`part_object_blocker_manifest` | 把 part masks/depth 转成 part surface rows，计算 part motion、articulation、SE(3) residuals 和 blockers | part surface archive、articulation candidates、part pose/blocker evidence |
| 10 | Contact/occlusion/nonpenetration evidence | 支持/校验 | measured/status contact/occlusion stages + final roots | 准备 mesh contact、contact ownership、signed/triangle nonpenetration、occlusion mesh/owner/depth-order evidence | contact graph、NP reports、occlusion owner evidence |
| 11 | Final evidence load, including HaWoR bridge | 支持输入 | `build_case_annotations()` load block | `run_v18_full_pipeline.py` 按代码顺序加载 annotation state、V16、bounded、camera/depth、hand baseline、pose-fill gate、geometry、physical schema、depth-fused mesh、contact/NP/occlusion roots、part surfaces、articulation、HaWoR metric MANO bridge | in-memory evidence indexes |
| 12 | Hand state assembly | 核心生成 | `for raw_hand in src_frame.hands` | 生成 final hand state：bbox、metric MANO/HaWoR bridge state、support state、occlusion-owner hypothesis、uncertainty | per-frame hand annotations |
| 13 | Object/part/contact assembly | 核心生成 | `for raw_obj` / `for raw_part` / `for raw_contact` | 生成 object state、part rows、visible/depth-fused geometry candidates、contact hypotheses、NP/contact evidence fields | per-frame object/part/contact annotations |
| 14 | Factor graph solve | 核心生成 | `solve_v18_factor_graph()` | 融合 hand/object/part/contact/occlusion/camera-depth evidence；连续项 sparse LS，contact Viterbi，occlusion min-energy | per-frame `factor_graph_solution` |
| 15 | Post-graph geometry/pose validation | 支持/校验 | `attach_reconstructed_geometry_pose()`、`attach_frame_local_part_pose_validation(True)`、`attach_part_structured_object_pose_state()`、`attach_object_depth_silhouette_pose_validation()` | 把图解和深度/轮廓/part evidence 回写为 object/part pose support/rejection/completion fields | geometry pose fields、part/object pose validation |
| 16 | Contact physical modes + depth-order occlusion | 核心生成 + 校验 | `attach_contact_physical_modes()`、`summarize_physical_contact_states()`、`attach_contact_depth_order_occlusion()` | 把 contact 分成 active、near noncontact、episode hypothesis、depth-occluded possible 等，并附加局部 depth-order occlusion evidence | physical contact mode fields、contact summary |
| 17 | Write final annotations | 核心生成 | `write_json(...annotations_v18_full.json)` | 写出最终 renderable backing annotation，包括 sources、module counts、factor summary | `annotations_v18_full.json` |
| 18 | Overlay/world/side-by-side render | 非核心灰色 | `render_overlay()`、`render_world()`、`compose_side_by_side()` | 用 final annotations 和 V16 base render 生成完整视频 | `v18_overlay.mp4`、`v18_world.mp4`、`v18_side_by_side.mp4` |
| 19 | QC/report/self-inspection | 非核心灰色 | `ffprobe_frame_count()`、case QC、full report、self inspection | 检查 frame count、记录 paths/draw counts/module counts；不生成新标注语义 | `v18_full_pipeline_qc.json`、`v18_full_pipeline_report.json`、`v18_completion_self_inspection.json` |

## 关键语义边界

- `3` 是 hand evidence reducer，不是 HaWoR 执行；它消费 cached WiLoR/HaWoR/RTMLib 测量。
- HaWoR metric MANO bridge 在 `11` 的 final evidence load 中被 `load_hawor_bridge_index()` 加载，并在 `12` 的 hand state assembly 中成为 final hand state 的主要 metric MANO 来源。
- 渲染、拼接、manifest、audit、QC、report 是非核心步骤；它们验证或展示 annotation，不产生新的物理标注语义。
- `active contact` 只在 `16` 中由 final support gates 决定；episode-only evidence 不再独立支撑 active contact。
- clean-rigid completion 和 part pose readiness 是 post-graph validation/support 字段，不等于所有操作物体都完成完整 hidden geometry 或严格 pose。

## 当前仍有限制

- 当前 full pipeline 默认读取 `/data2/ego_annotation_outputs/...` 下的 V16/V17/V18 缓存证据；不是完整 fresh raw-video-to-final 端到端运行证明。
- 主操作物体的完整 hidden geometry 和严格 object pose 仍未普遍解决，尤其是 deformable、surface-changing、part-required 或遮挡严重对象。
- nonpenetration 仍是局部 signed/triangle evidence，不是完整 watertight SDF/nonpenetration solver。
- occlusion owner 和 pose-fill 是两件事；accepted owner 不自动推出被遮挡 3D hand pose。
- 最终质量仍要看完整 overlay/world/side-by-side 视频，程序化 QC 只约束帧数、路径、字段和部分统计。
