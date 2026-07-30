# Pipeline V20 Current Structure Audit

本文反映当前仓库中的 V20 链路状态，而不是 V20 的理想目标。结论来自直接检查当前文档、脚本入口和可见输出目录；`.memory` 里的历史总结只作为线索，不作为事实来源。

## 1. Critic 处理结论

critic 的审查意见是：不要相信 worker summary，要检查实际文件和输出。这个意见适用，并已经改变本次分析方法。

本次核对的直接证据包括：

- V20 文档与 prompt：`docs/v20_run_contract.md`、`docs/v20_english_orchestration.md`、`docs/v20_component_extraction.md`、`.pi/prompts/v20_infer.md`、`.pi/prompts/v20_benchmark.md`、`configs/v20_agent_system_prompt.md`。
- V19 对照文档：`docs/v19_run_contract.md`、`docs/v19_english_orchestration.md`、`docs/pipeline_v19_design_proposal.md`。
- V20 脚本入口：`scripts/prepare_v20_benchmark_dataset.py`、`scripts/build_v20_infer_base_annotations.py`、`scripts/build_v20_depth_candidate_registry.py`、`scripts/select_v20_depth_observation_bundle.py`、`scripts/build_v20_geometry_candidate_registry.py`、`scripts/validate_v20_geometry_candidates.py`、`scripts/solve_v20_infer_temporal_observation_graph.py`、`scripts/render_v20_benchmark_annotations.py`、`scripts/assemble_v20_state_from_v19_annotations.py`、`scripts/evaluate_v20_benchmark_gt.py` 等。
- 当前输出根：`outputs/v20_closed_loop_dexycb_20200813_151041_932122062010`、`outputs/v20_infer_requested_20260625/*`、`outputs/v20_infer_adjusted_20260625/*`，以及历史 `/tmp/v20_benchmark_*` oracle bootstrap 输出。
- 可视核对图像：`review/v20_side_by_side_*.jpg`、`review/v20_overlay_*.jpg`、`review/v20_world_*.jpg`。

处理结果：

- `/tmp/v20_benchmark_ycb_full` 和 `/tmp/v20_benchmark_ho3d_mc1_full` 是历史 oracle/bootstrap 产物。它们证明过 dataset loader、state boundary、renderer、GT evaluator 的 plumbing，但它们的 0 error 是 GT-copy oracle 结果；当前 `docs/v20_component_extraction.md` 已把 `scripts/run_v20_benchmark_oracle_bootstrap.py` 标为 deprecated/disabled，所以不能作为当前正常 V20 链路证据。
- 当前正常 V20 证据应以 `state/v20_physical_state.json`、`state/annotations_v20_renderable.json`、`renders/render_summary.json`、`evaluation/final_gt_evaluation/*` 为主。部分 `run_summary.json` 仍停在 timeline bootstrap 阶段并写着 `renders_created=false`，与后续 state/render 产物矛盾；因此它不是可靠的最终状态源。
- 当前 V20 的确能在若干样例上产生全时长 overlay/world/side-by-side render，但多数 infer 输出是 visible-surface-only 物体点云/掩码、HaWoR 手部候选和轻量 temporal smoothing；这不是完整的 object mesh reconstruction、rigid object pose、contact ownership 或 nonpenetration 链路。

## 2. Theorist 视角：V20 原始实现想法

V20 的合理设计思想是继承 V19 的 harness 架构，而不是另起一个外层 wrapper。Pi 仍是 harness；Python 脚本只是 measurement、optimization、render、dataset loading、evaluation 工具。最终进度仍由 renderable physical annotation 判断。

V20 相对 V19 的理论增量主要有四类。

### 2.1 输入模态支链

V19 的核心输入假设偏向 raw video + 可选外部几何。V20 试图把输入模态显式分支化：

- native depth / RGB-D：把传感器 depth 作为 privileged candidate，但仍需记录 scale、alignment、valid mask 和 residual。
- stereo：优先 calibrated/rectified stereo；没有 calibration 时只能保留弱 relative disparity，不能支持 metric object/contact claim。
- monocular/foundation depth：UniDepth、DepthPro、VGGT、DUSt3R/MASt3R 等应统一进入 depth candidate registry。
- benchmark dataset：DexYCB/HO3D loader 提供 RGB/depth/calibration/model library，并把 GT 严格隔离到 evaluation reference。

理论上这些分支不直接决定最终状态，而是进入统一的 depth registry 和 selector，由物理 residual 决定 primary depth 或不确定状态。

### 2.2 算法簇作为 optimization evidence

V20 不是把一个新算法结果直接写成 accepted state，而是把算法簇输出变成候选证据：

- depth candidate registry + selector；
- geometry candidate registry + validation/promotion；
- per-hand-track MANO betas/scale posterior；
- render-only contact point rows；
- V20 observation bundle；
- branch optimization / factor correction 后再组装 renderer state。

这个想法是对的：它保持 V19 的因果结构，把 noisy measurements 作为 robust optimization 的输入，而不是把模型输出当真值。

### 2.3 V19-style state boundary

理论链路仍应是：

```text
input contract
  -> timeline / camera / modality report
  -> target object plan from visual/task evidence
  -> SAM2 or prompt-conditioned object masks
  -> hand/depth/camera/object geometry measurements
  -> V20 observation bundle
  -> V19-style branch optimization / factor graph correction
  -> state/annotations_v20_renderable.json
  -> full-duration overlay/world/side-by-side render
  -> visual review and benchmark evaluation when GT exists
```

核心不变量是：`state/` 是 renderer boundary；measurement files、registries、validators、GT metrics、run summaries 都不能代替 render-consumed physical state。

### 2.4 Benchmark feedback loop

V20 benchmark 的理论目标是让 public GT 数据集在 prediction 完成后提供反馈：GT evaluator 只读 reference manifest，不修改 prediction state；controller 根据 metric + visual failure cluster 提出下一次 measurement/weight/branch intervention。

这个思路也正确，但只有在 prediction-side 链路本身已经真实运行时才有意义。GT loop 不能变成用 GT 生成 prediction，也不能把 0 error oracle bootstrap 误读成 inference accuracy。

## 3. 当前真实实现结构

当前仓库里的 V20 由文档/prompt、dataset adapter、measurement sidecars、minimal temporal graph、state assembler、renderer 和 evaluator 组成。它已经超过单纯 schema 草案，但还没有闭合成完整 object pose / hand-object physics 链路。

### 3.1 文档和 prompt 层

- `docs/v20_run_contract.md` 定义 `/v20_infer`、`/v20_benchmark`、dataset contracts、run directory、GT isolation、state boundary。
- `docs/v20_english_orchestration.md` 是当前最接近运行真相的 runbook。它明确说当前可以 closed minimal approximate V20 infer，但 full-strength claims 仍缺自动目标发现、完整 mesh reconstruction、metric stereo/multiview depth、高质量 MANO/contact。
- `docs/v20_component_extraction.md` 把可复用 V18/V19 组件和 V20 adapter 分开，并明确 deprecated `scripts/run_v20_benchmark_oracle_bootstrap.py`。
- `.pi/prompts/v20_infer.md` 和 `.pi/prompts/v20_benchmark.md` 约束 Pi 仍是 harness，脚本不能成为最终 authority。
- `configs/v20_agent_system_prompt.md` 强调 V20 additions 必须进入 V19-style branch optimization/factor correction；如果没有 branch optimization report，应停止而不是把 sidecar 渲染成 final state。

### 3.2 Input / benchmark adapter 层

已实现：

- `scripts/run_v19_v20_infer_bootstrap.py`：为 arbitrary video 写 raw frame manifest 和初始 run structure。
- `scripts/prepare_v20_benchmark_dataset.py`：支持 DexYCB/HO3D fail-fast loader，把 prediction manifest 写入 `input/`，把 GT 只写入 `evaluation/reference_manifest.json`。
- `scripts/evaluate_v20_benchmark_gt.py`：在读取 GT 前拒绝含 `gt` / `oracle` marker 的 prediction state，然后输出 `gt_metrics.json`、`gt_alignment.json`、`failure_clusters.json`、`evaluation_agent_report.md`。

真实状态：benchmark adapter 和 evaluator 是当前 V20 比 V19 更实的部分之一。`outputs/v20_closed_loop_dexycb_20200813_151041_932122062010/evaluation/final_gt_evaluation/gt_metrics.json` 显示了非 oracle prediction 的 hand/object errors；这比历史 oracle 0 error 更有意义。

局限：benchmark controller loop 还没有形成真正多轮自动改进。当前看到的是单次 prediction 后 evaluation，而不是多 iteration 的 physical intervention ledger。

### 3.3 Target object plan / mask 层

已实现或已有脚本：

- `scripts/build_v20_infer_point_prompts_from_object_plan.py`；
- `scripts/filter_v20_sam2_masks_by_prompt_components.py`；
- `scripts/build_v20_prompt_conditioned_local_masks.py`；
- benchmark 辅助的 `scripts/build_v20_benchmark_video_and_object_prompts.py`。

真实状态：当前 V20 已经修正了一个关键边界错误：dataset/public object roster 不能直接作为 target list。目标应来自 object plan。实际 infer 输出里有 `measurements/object_candidates/object_plan_agent.json` 或 `object_plan_final.json`。

局限：自动 target-object discovery 仍没有成为稳定机制。很多运行依赖 agent/人工选出的 object plan 和 prompt；一旦 object plan 错，后续 SAM2、local masks、visible surface 都会跟着错。

### 3.4 Base annotation 层

已实现：

- `scripts/build_v20_infer_base_annotations.py`：从 raw frame manifest、object plan、SAM2/local masks、可选 depth、可选 MANO 生成 `state/base_annotations_v20_infer.json`。
- `scripts/build_v20_dexycb_base_annotations.py`：DexYCB prediction-side base annotation builder。

真实状态：该层是当前 V20 infer 能 render 的直接基础。它生成一帧一行的 V18/V19-compatible annotation shape，并把 object mask、visible surface samples、centroid、hand candidate rows 放进去。

局限：其报告明确写着这是 approximate state，不是 final optimized physical truth。对于无 metric depth 的 stereo 样例，它使用 `weak_fallback_from_image_size_not_metric_calibration` 和 pseudo-depth/visible-surface-only 表示；这不能支撑 metric object pose、contact 或 nonpenetration。

### 3.5 Depth modality / selector 层

已实现：

- `scripts/build_v20_depth_candidate_registry.py`；
- `scripts/select_v20_depth_observation_bundle.py`；
- `scripts/build_v20_uncalibrated_stereo_disparity.py`。

真实状态：

- DexYCB benchmark prediction 使用 native DexYCB depth candidate，并在 state 中选为 `primary_for_visible_surface` 和 `primary_for_hand_depth`。
- RealSense/mocap infer 使用 native RealSense depth candidate。
- Pico/living-room stereo 只保留 OpenCV SGBM uncalibrated disparity report，state 中 `primary_depth_by_scope` 为 `null`，并标注为 weak non-metric evidence。

局限：

- calibrated stereo depth 尚未成为正常路径；无 calibration 的 stereo 不能给 metric depth。
- monocular/foundation depth 算法簇在文档中存在，但当前输出没有显示真实接入的多后端 selector 竞争。
- 部分 adjusted infer 输出的 `depth_selection_report.json` 仍写 `no_candidate_executed_in_bootstrap`，但后续 state 已包含 weak disparity report；这说明报告源之间有陈旧/不一致。

### 3.6 Geometry candidate / validation 层

已实现：

- `scripts/build_v20_geometry_candidate_registry.py`：要求候选 mesh tied to selected object plan，并拒绝 GT/oracle geometry candidate。
- `scripts/fit_v20_cad_mesh_to_visible_depth.py`：benchmark 中把 public CAD mesh 拟合到 prediction-side visible depth。
- `scripts/validate_v20_geometry_candidates.py`：用 visible surface alignment、silhouette projection、free-space/nonpenetration/contact residual 等验证 candidate。

真实状态：

- DexYCB closed-loop 输出里存在 4 个 geometry candidates，validation 只 promoted 1 个 `cad_visible_depth_fit`，其余因 visible-depth/silhouette conflict 被拒绝。
- Arbitrary infer 输出里 `geometry_candidates` 通常为空，最终 render 显示的是 visible surface point cloud / bounding box / mask，而不是 completed object mesh pose。

局限：

- 任意物体的 complete mesh reconstruction / conditioned generation 仍缺失。TRELLIS/conditioned geometry 在当前正常输出中没有作为真实 mesh backend 闭合。
- visible surface centroid smoothing 不能替代 object pose。按项目定义，object pose 在 manipulated object 场景中需要 reconstructed object geometry；点云、centroid、box 或局部 mask 不能声称是 object pose。

### 3.7 Hand / MANO 层

已实现：

- `scripts/adapt_hawor_camspace_to_v20_mano_npz.py`；
- `scripts/solve_v20_hand_shape_track.py`；
- `scripts/align_v20_hawor_overlay_to_detector_boxes.py`。

真实状态：

- 多个 infer 输出有 HaWoR MANO rows 和 track-level hand shape posterior。
- overlay alignment 使用 detector boxes 改善可视化位置；state/report 明确说这是 2D overlay alignment only。

局限：

- 2D detector-box alignment 不能修复 metric MANO。当前 hand state 多数仍是 uncertain measurement / render layer，不能支撑 contact ownership、nonpenetration 或 metric hand-object relation。
- hand-shape posterior 是 prior，不是 accepted hand truth。

### 3.8 Contact / occlusion / nonpenetration 层

已实现：

- `scripts/build_v20_contact_point_render_rows.py`。

真实状态：contact rows 在 state 中以 `evidence_created=false` 和 render-only policy 出现；occlusions 和 nonpenetration 通常为空数组。

局限：当前 V20 基本没有真实 contact、occlusion ownership、nonpenetration solver。可视化 contact points 或 hand/object proximity 不能作为 contact evidence。

### 3.9 Optimization 层

已实现：

- `scripts/solve_v20_infer_temporal_observation_graph.py`。

真实状态：这是当前 arbitrary infer 能通过 `--require-branch-optimization` 的主要报告。它优化 visible-surface centroid temporal smoothness 和 MANO joint/surface temporal smoothness，并输出 `state/v20_temporal_observation_graph_report.json`。

局限：这不是完整 V19-style robust physical factor graph。它没有把 completed mesh pose、contact/nonpenetration、occlusion、camera/depth scale 一起优化。对于 rigid object branch，它只能算 visible-surface-only approximation，不能替代 rigid mesh pose graph。

### 3.10 State assembly / renderer 层

已实现：

- `scripts/assemble_v20_state_from_v19_annotations.py`；
- `scripts/render_v20_benchmark_annotations.py`。

真实状态：

- 当前多个 outputs 有 full-duration `renders/v20_overlay.mp4`、`renders/v20_world.mp4`、`renders/v20_side_by_side.mp4`，且 `renders/render_summary.json` frame count 匹配。
- Renderer 的可见层主要画 object masks/boxes/visible-surface points 和 hand skeleton/overlay。review frame 显示：Pico/living-room 是点云+手骨架；DexYCB 里 marker mask/桌面污染在 render 中很明显。

局限：

- `render_v20_benchmark_annotations.py` 名字和 overlay 文案在 infer 输出中仍写 “benchmark prediction”，会误导读者。
- world panel 是 prediction-side geometry scatter / skeleton view，不是完整 metric mesh world render。
- 部分 run summary 没随最终 render 更新，导致最终状态判断必须绕开 `run_summary.json`。

## 4. 当前输出证据的分级

### 4.1 可作为当前正常 V20 证据的输出

- `outputs/v20_closed_loop_dexycb_20200813_151041_932122062010`：非 oracle DexYCB benchmark prediction。72/72/72 render frame count 匹配；native depth 被选中；geometry validation 有 4 个 CAD-visible-depth candidates、1 个 promoted；GT evaluator 在 prediction state 之后运行，报告 hand median error 约 3.4 cm、object translation median 约 5.7 cm、rotation median 约 1.98 rad。
- `outputs/v20_infer_requested_20260625/vtla_stereo_2026-06-09-17-28-55`：307 帧 full render；object plan + masks + weak uncalibrated stereo + HaWoR + temporal graph；primary metric depth 为空。
- `outputs/v20_infer_requested_20260625/mocap_multiview_prismatic_20260209_150537`：1115 帧 full render；native RealSense depth candidate selected；没有 hand rows；visible-surface temporal graph。
- `outputs/v20_infer_adjusted_20260625/pico_trackers_10_2_100s_120s_stereo`：500 帧 full render；weak uncalibrated stereo；HaWoR overlay alignment；object 是 visible-surface-only red tin。
- `outputs/v20_infer_adjusted_20260625/living_room_cleanup_camera2_camera1_full_stereo`：1266 帧 full render；weak uncalibrated stereo；HaWoR overlay alignment；object 是 visible-surface-only clear glass bowl/container。

这些输出支持的 scoped claim 是：当前 V20 能从 selected target masks、可用 depth/weak depth、可选 hand candidates 生成全时长 approximate renderable annotations。它不支持完整 mesh object pose、contact ownership、occlusion ownership 或 nonpenetration claim。

### 4.2 历史/diagnostic 输出

- `/tmp/v20_benchmark_ycb_full`；
- `/tmp/v20_benchmark_ho3d_mc1_full`。

这些输出是 oracle/bootstrap，`gt_metrics.json` 明确写 `state copied from GT for oracle/bootstrap harness validation` 和 `non_oracle_prediction_metrics_supported=false`。它们只能证明早期 benchmark plumbing，不是当前正常 V20 inference。

### 4.3 当前不可靠的状态源

- 一些最终 run root 的 `run_summary.json` 仍停留在 `timeline_bootstrap_complete_measurements_pending`，即使同目录后续已有 render/state。因此 `run_summary.json` 当前只能作为早期 bootstrap log，不能作为最终 V20 run closure 判断。

## 5. Critic 视角：设计与实现的偏差

### 5.1 已经真实实现的部分

- V20 prompt/contract/runbook 已经把 Pi harness、state boundary、GT isolation、target object plan、branch optimization requirement 写清楚。
- DexYCB/HO3D benchmark loader 和 post-prediction evaluator 已经存在。
- Depth candidate registry / selector、geometry registry / validator、hand shape posterior、render-only contact rows、observation bundle 已有脚本实现。
- Arbitrary infer 有最小闭环：object plan + masks + optional depth/hand -> base annotations -> temporal graph -> state assembly -> full-duration render。
- Prediction state 中有多处 GT/oracle marker guard。

### 5.2 仍停留在设计或弱实现的部分

- 自动 target discovery 没有稳定闭合；目前依赖 agent/人工 object plan。
- 输入模态支链没有全闭合：native depth 可用，uncalibrated stereo 只是弱 relative evidence，calibrated stereo 和 monocular/foundation depth cluster 没有成为默认正常路径。
- 任意 object complete mesh reconstruction 没有闭合；arbitrary infer 多数没有 geometry candidates。
- `solve_v20_infer_temporal_observation_graph.py` 是 temporal smoother，不是完整 physical factor graph。
- Hand overlay alignment 改善可视化，但不是 metric MANO 修复。
- Contact、occlusion、nonpenetration 几乎没有真实 solver 输出。
- Renderer 仍偏 diagnostic scatter/overlay，不是 V16/V18 意义上的 metric 3D MANO + object mesh render。
- Benchmark feedback loop 没有稳定多轮 controller intervention；GT metrics 还没有系统地反向定位机制错误。

### 5.3 最危险的误读

- 把 visible surface、mask、box、centroid、scatter world panel 误读成 object pose。
- 把 detector-box-aligned hand overlay 误读成 metric hand state。
- 把 render-only contact rows 误读成 contact evidence。
- 把 oracle bootstrap 0 error 误读成 V20 benchmark accuracy。
- 把 `run_summary.json` 的早期状态误读成最终状态，或反过来用最终 render 掩盖早期 summary 没更新的执行不一致。

## 6. 真实可行 V20 的优化路线

目标不是再加 registry 或 report，而是让 V20 的新增观测真正进入一个可复现、可渲染、可评估的 physical state loop。

### 6.1 第一优先级：修正 run closure 的 source of truth

先解决状态源不一致，否则后续优化无法判断是否有效。

需要做：

1. 让最终 stage 更新或重写 `run_summary.json`，至少记录 final state、render summary、branch reports、metric depth status、geometry status、known unresolved variables。
2. 修改 `render_v20_benchmark_annotations.py` 的模式命名和 overlay 文案，使 infer 输出不再显示 “benchmark prediction”。
3. 增加一个轻量 closure checker，只检查最终产物一致性：state 指向的 annotations 存在、render summary frame count 匹配、branch reports 存在、GT/oracle marker 不在 prediction state、run summary 与 final state 不矛盾。
4. 把 deprecated oracle bootstrap 从正常 prompt/runbook 路径中隔离，只作为 historical diagnostic。

这一步不产生新的物理能力，但它防止后续把错误状态源当成进度。

### 6.2 第二优先级：把输入模态支链做成真实 depth/camera adapters

V20 的“多输入模态”想法只有在每个分支能输出统一 metric/uncertain semantics 时才成立。

需要做：

1. 实现或收敛 `depth_modality_report`：native RGB-D、calibrated stereo、uncalibrated stereo、monocular RGB、multiview/fisheye 分开记录。
2. 对 stereo：先找 calibration/rectification。找不到时只能输出 weak nonmetric disparity，不允许被 selector 选为 primary metric depth。
3. 对 native RGB-D/RealSense/DexYCB：统一 depth scale、intrinsics、RGB-depth registration semantics，写入 candidate residual。
4. 对 monocular/foundation depth：只接入少数默认后端，避免算法簇 fan-out 造成 runtime 失控；先选 UniDepth/Metric3D 类 metric candidate + 一个 relative depth candidate 做对照。
5. 让 `select_v20_depth_observation_bundle.py` 的输出真正成为 final state 的 depth truth source；不能出现 selection report 还是 bootstrap placeholder，而 final state 又写 weak report 的不一致。

### 6.3 第三优先级：目标发现和 mask 质量要进入闭环

当前最大实际失败机制之一是 object plan / prompt / mask identity 错误，DexYCB marker-table 污染和 stereo prompt 偏移都说明了这个问题。

需要做：

1. 把 object plan 从手工 JSON 变成 harness step：sample keyframes -> open-vocabulary detector/segmenter proposals -> agent review -> selected target list。
2. 每个 selected target 必须有 visual evidence keyframes、positive/negative prompts、rejected alternatives。
3. SAM2/local mask 输出必须有 review sheet 和 contamination flags；低质量 mask 不应静默进入 geometry fit。
4. Dataset/public object roster 继续只能作为 model library，不得变成 target list。

这一步比继续加 geometry validator 更重要，因为 wrong mask 会污染 depth、geometry、pose、contact 的全部下游变量。

### 6.4 第四优先级：把 visible surface 升级成真实 object geometry/pose branch

这是 V20 能否成为真正 physical pipeline 的核心。

需要做：

1. 对 rigid branch，强制执行：visible surface evidence -> mesh completion/adaptation -> per-frame pose fit -> robust temporal/object factor graph -> mesh-pose render。
2. arbitrary infer 先选一个最简单对象闭合，比如 Pico red tin 或 DexYCB bleach/can，而不是同时追求所有对象。
3. 复用已有 V18/V19 几何脚本：`reconstruct_object_mesh_v2.py`、`reconstruct_scaled_observed_object_mesh_v3.py`、`reconstruct_object_visual_hull_depth_carve_v3.py`、`complete_object_heightfield_from_mask_depth_v3.py`、`optimize_object_factor_graph_v3.py`、`optimize_joint_camera_object_graph_v3.py`。
4. TRELLIS/conditioned generation 只有在真实 mesh 文件和 source report 存在时才注册 candidate；缺失时保持 `missing_v20_conditioned_geometry_generation_output`，不能写 fake candidate。
5. `solve_v20_infer_temporal_observation_graph.py` 应明确降级命名为 visible-surface smoother；它可以是 unresolved/visible-surface branch 的 branch report，但不能满足 rigid object pose branch。

验收标准不是 registry 非空，而是 render 中出现 fitted/adapted mesh，并且 benchmark/object residual 或视觉 review 显示它比 visible-surface scatter 更接近物体。

### 6.5 第五优先级：修复 metric MANO，而不是只修 overlay

当前 2D overlay alignment 对 demo 可读性有帮助，但不能支持物理接触。

需要做：

1. 定位 HaWoR/MANO projection error 的来源：camera intrinsics、resize/crop、coordinate convention、MANO global transform、left/right side mapping。
2. 对有 depth 的样例运行 metric depth refit；没有 metric depth 的样例只保留 overlay/uncertain hand state。
3. hand shape posterior 只作为 prior；accepted hand state 必须有 camera/world semantics、surface/joints、visibility、uncertainty。
4. renderer 同时显示 metric confidence：overlay-aligned-but-nonmetric 的手不能和 metric MANO 用同一视觉语义。

### 6.6 第六优先级：contact / occlusion / nonpenetration 只在 geometry 与 MANO 成立后开启

当前 contact rows 是 render-only。下一步不应先造 contact label，而应等 metric hand + object mesh + depth-order 有基本可信度后再建 factor。

需要做：

1. 先实现 visibility/occlusion state：visible、partial、occluded、out-of-frame、unresolved，并记录 occluder ownership uncertainty。
2. contact 只从 hand surface/object mesh distance、depth order、temporal co-motion 和 nonpenetration residual 推出。
3. contact renderer 必须区分 `near_contact_candidate`、`contact_supported`、`contact_unresolved`。
4. 如果没有 mesh/depth/MANO 支撑，继续显示 uncertainty，而不是 contact marker。

### 6.7 第七优先级：把 benchmark loop 变成机制反馈，而不是指标展示

当前 evaluator 已能算误差，但 controller 还没有真正利用 failure clusters 修链路。

需要做：

1. 对 DexYCB representative 建一个 2-iteration benchmark recipe：iteration 0 当前 prediction，iteration 1 只允许一个机制 intervention。
2. failure cluster 要把大 object rotation/translation error 归因到 mask identity、CAD fit、coordinate convention、depth scale、pose graph 等机制之一。
3. controller 只能改 measurement/weight/branch/camera/depth/renderer adapter 中的一个原子项，然后 rerender。
4. metric 改善必须同时通过 visual render review；否则记录为 metric-overfit failure。

## 7. 建议的最小下一版实现包

如果目标是把 V20 从“可渲染近似链路”推进到“真的可行链路”，我建议下一次只做三个原子包：

1. **Closure repair package**：修正 stale `run_summary.json`、infer renderer 文案、final closure checker、deprecated oracle 隔离。目标是让每个 run root 的最终状态可判定。
2. **Metric depth + object mask package**：为一个 stereo 或 RGB-D 样例闭合 modality report、depth candidate selection、object mask contamination review。目标是得到可信 primary metric depth 或明确 unresolved，不再靠 pseudo-depth。
3. **Single-object rigid branch package**：选一个目标物体执行 completed/adapted mesh candidate -> visible-frame pose fit -> temporal pose graph -> mesh-pose render。目标是让 `geometry_candidates` 不再为空，并让 render 从 point scatter 变成 object mesh pose。

这三个包完成后，V20 才能说真正继承了 V19 的物理 annotation 思路，并把 V20 新增模态/算法簇接入了可运行优化链路。当前版本更准确的描述是：V20 已有 harness、benchmark adapter、candidate sidecars 和 minimal visible-surface render closure；完整 physical hand-object pipeline 仍缺 metric depth/camera closure、object mesh reconstruction/pose branch、metric MANO repair、contact/occlusion/nonpenetration factors。
