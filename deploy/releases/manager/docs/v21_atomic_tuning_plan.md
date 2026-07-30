# V21 Atomic Algorithm Precision Tuning Plan

## Scope
深度、bbox/分割、手部/MANO、几何/pose graph、接触/遮挡/非穿透这些 master/V19-V18 原子算法，加上 V21 新增算法，都必须有可审计的 data path、overlay path 和 QC 记录。脚本存在不等于跑通；overlay 文件存在也不等于视觉质量通过。

V21 的新增算法只加入相关算法簇：depth/camera modality（包括 no-depth/V19 behavior、dataset camera parameters、stereo video/native depth/multiview where available）和额外 point-cloud/mesh-completion candidate。bbox/segmentation 保持 V19/master spine；除非 upstream/downstream schema 改变需要接口适配，不修改 bbox 或 SAM2 算法语义。只有当某个原子算法的实际 output 出现 large deviation 时，才在该原子算法内部做 sample-bound tuning 并写 tuning 记录。

GroundingDINO bbox 当前按用户要求禁用为 V21 默认路径。历史 `groundingdino` 产物只保留为 deprecated evidence，不进入默认 SAM2 prompt、QC 或 downstream geometry。默认替代 bbox 算法是 OWLv2；无自动 bbox 时只能使用 agent object-plan 的显式点/框 prompt，并在报告中标明来源。

## Output Structure
```text
outputs/v21_per_algorithm_results/
  atomic_algorithm_overlay_audit.json
  pico/
    depthpro/overlay.mp4 + qc.json
    unidepth/overlay.mp4 + qc.json
    depth_anything/overlay.mp4 + qc.json
    stereo_sgbm/overlay.mp4 + qc.json
    owlv2_bbox/overlay.mp4 + qc.json
    sam2/overlay.mp4 + qc.json
    # sam2_per_frame_bbox is deprecated non-V19 history, not an active V21 output
    # local_prompt_masks is deprecated V20 history, not an active V21 output
    rtmlib/overlay.mp4 + qc.json
    wilor/overlay.mp4 + qc.json
    wilor_metric/overlay.mp4 + qc.json
    hamer/overlay.mp4 + qc.json
    active_mano/overlay.mp4 + qc.json
    visible_surface/overlay.mp4 + qc.json
    v19_rigid_pose_graph/overlay.mp4 + qc.json
    object_factor_graph/overlay.mp4 + qc.json
    contact_ownership_graph/overlay.mp4 + qc.json
    occlusion_owner_graph/overlay.mp4 + qc.json
    signed_nonpenetration/overlay.mp4 + qc.json
  living_room/
    (same structure)
```

The machine-readable coverage check is:

```bash
python scripts/audit_v21_atomic_algorithm_overlays.py \
  --overlay-root outputs/v21_per_algorithm_results \
  --output outputs/v21_per_algorithm_results/atomic_algorithm_overlay_audit.json
```

## Algorithm List

### DEPTH / CAMERA
1. DepthPro — V18/V21 monocular metric depth baseline.
2. UniDepth V2 — master/V19 monocular depth + intrinsics comparator.
3. Depth Anything V2 — V21 additional monocular/depth-shape evidence.
4. Stereo SGBM / future RAFT-Stereo — weak relative stereo unless calibrated.
5. DROID/camera trajectory or imported camera-depth contract where available.

### BBOX / SEGMENTATION
6. Agent keyframe selection — `segmentation_stable_keyframes.json` records the OWLv2 detector frames chosen from the object plan.
7. OWLv2 bbox proposals — default replacement for GroundingDINO keyframe bbox evidence, not a continuous bbox tracker.
8. Approved OWLv2 bbox prompts — `owlv2_bbox_approved_prompts.json` records the target boxes that seed SAM2.
9. SAM2 proper bbox propagation — active full-video object masks under `sam2_proper/<object_id>/sam2_masks` plus `sam2_proper_summary.json`.
10. Segmentation contamination review — `review/segmentation_sam2_proper/segmentation_contamination_review.json` gates masks before geometry.

### HAND / MANO
11. RTMLib — 2D hand keypoints.
12. WiLoR — hand detection + MANO candidate.
13. WiLoR metric refit — V21 depth/scale refit.
14. HaMeR — 3D hand reconstruction candidate.
15. HaWoR / hand baseline branch — master temporal hand evidence where available.
16. Active MANO optimizer — V21 accepted MANO path or explicit unresolved blocker.

### GEOMETRY / POSE / PHYSICAL VARIABLES
17. Visible surface from accepted masks + selected depth.
18. Heightfield/observed mesh reconstruction.
19. V21 mesh candidate / completion adapter.
20. V19 rigid pose graph.
21. Object factor graph, mesh-prior graph, joint camera/object, joint MANO/object, contact-patch graph where inputs exist.
22. Contact ownership graph.
23. Occlusion owner graph.
24. Signed and triangle nonpenetration evidence.
25. Final overlay/world/side-by-side render.

## Workflow per algorithm
1. Run or import algorithm output on the declared compute target.
2. Generate overlay video with `scripts/generate_algorithm_overlay.py` or a native renderer.
3. Write `qc.json` with visual review, key residuals, and claim scope.
4. If the atom's output shows measured large deviation, write `tuning/<family>/<algorithm>/attempt_<k>.json` before downweighting; otherwise do not create a tuning obligation just because the atom exists.
5. Run `scripts/audit_v21_atomic_algorithm_overlays.py` and keep unresolved rows explicit.
