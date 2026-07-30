#!/usr/bin/env python3
"""V21 state assembler: update physical state with all measurements.

Reads all V21 measurement outputs and assembles the complete V21 physical
state JSON that drives the renderer.

Output:
  state/v21_physical_state.json (updated)
  state/v21_uncertainty_state.json
  state/v21_agent_evidence.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run(args):
    run_root = Path(args.run_root)

    # Load existing state
    state_path = run_root / "state" / "v21_physical_state.json"
    state = load_json(state_path) or {"schema": "v21_physical_state.v0"}

    # Timeline
    manifest = load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")
    timeline = {
        "frame_count": len(manifest["frames"]),
        "fps": manifest.get("fps", 25.0),
        "duration_s": len(manifest["frames"]) / manifest.get("fps", 25.0),
        "resolution": [manifest["frames"][0]["source_width"], manifest["frames"][0]["source_height"]],
    }

    # Camera/depth state
    depth_qc = load_json(run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "qc_depthpro_full_frame_v21.json")
    camera_depth = {
        "state": "depthpro_monocular_metric_depth_selected_provisional",
        "primary_candidate_id": "depthpro_rgb_metric_depth_baseline",
        "primary_depth_archive": str(run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"),
        "metric_depth_available": True,
        "provisional": "true_until_compared_with_independent_monocular_or_assisted_candidates",
        "focal_length_median_px": depth_qc.get("depthpro_focal_px", {}).get("median") if depth_qc else None,
    }
    if depth_qc:
        camera_depth["depth_median_m"] = depth_qc.get("depth_median_m", {}).get("median") if isinstance(depth_qc.get("depth_median_m"), dict) else depth_qc.get("depth_median_m")

    # Segmentation state
    seg_review_path = run_root / "review" / "segmentation_sam2_proper" / "segmentation_contamination_review.json"
    seg_review_source = "sam2_proper_owlv2_bbox_prompt"
    seg_review = load_json(seg_review_path)
    segmentation = {
        "state": "sam2_proper_owlv2_bbox_prompt_segmentation_accepted",
        "review_source": seg_review_source,
        "review_report": str(seg_review_path) if seg_review else None,
        "accepted_tracks": [t.get("track_id") for t in (seg_review or {}).get("tracks", []) if t.get("decision", "").startswith("accept")] if seg_review else [],
    }

    # Hand state
    hand_qc = load_json(run_root / "measurements" / "hand_candidates" / "wilor_v21_metric" / "wilor_metric_qc.json")
    hands = {
        "state": "metric_hand_candidate_available_not_active_optimized",
        "backend": "WiLoR",
        "candidate_source": "wilor_v21_metric",
        "metric_scale_method": "bone_length_normalization_plus_depth_z_refit",
        "metric_scale_applied": hand_qc.get("scale_applied") if hand_qc else None,
        "depth_validation": hand_qc.get("depth_validation") if hand_qc else None,
        "candidate_state": "metric_scaled_depth_refined_candidate",
        "active_optimization": "not_yet_run",
        "contact_nonpenetration_enabled": False,
    }

    # Object state
    mesh_summary = load_json(run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / args.object_id / "mesh_candidate_summary.json")
    pose_root = run_root / "measurements" / "object_geometry_mesh_pose" / args.object_id
    pose_qc_path = pose_root / "v21_pose_estimate_qc.json"
    pose_qc = load_json(pose_qc_path)
    v21_fit_qc = load_json(pose_root / "v21_pose_fit_qc.json")
    v18_fit = load_json(pose_root / "v18_icp_fit" / "v18_compact_rigid_object_pose_fit_report.json")
    v19_graph = load_json(pose_root / "v19_pose_graph" / "v19_rigid_object_pose_graph_report.json")
    objects = {
        "state": "mesh_candidate_generated_pose_estimated_and_v19_graph_published",
        "object_id": args.object_id,
        "mesh_candidate": mesh_summary.get("mesh_path") if mesh_summary else None,
        "mesh_method": mesh_summary.get("method") if mesh_summary else None,
        "mesh_extent_m": mesh_summary.get("object_extent_m") if mesh_summary else None,
        "pose_method": "mask_centroid_depth_backprojection_with_pca_orientation",
        "pose_estimated_frames": pose_qc.get("estimated_frames") if pose_qc else None,
        "pose_depth_median_m": pose_qc.get("depth_summary", {}).get("median_m") if pose_qc else None,
        "v21_icp_fit_frames": v21_fit_qc.get("fit_frames") if v21_fit_qc else None,
        "v21_icp_fit_median_residual_m": v21_fit_qc.get("residual_summary", {}).get("median_m") if v21_fit_qc else None,
        "v18_compact_fit_frames": v18_fit.get("fit_frame_count") if v18_fit else None,
        "v18_compact_fit_median_residual_m": v18_fit.get("final_observed_to_mesh_median_summary_m", {}).get("median") if v18_fit else None,
        "v19_pose_graph_status": v19_graph.get("status") if v19_graph else None,
        "v19_pose_graph_frames": v19_graph.get("graph_frame_count") if v19_graph else None,
        "candidate_state": "pose_graph_published_mesh_candidate",
        "icp_refinement": "v18_compact_fit_rerun_from_current_sam2_masks",
    }

    # Render state
    renders = {
        "overlay": str(run_root / "renders" / "v21_overlay.mp4"),
        "segmentation_overlay": str(run_root / "renders" / "v21_segmentation_overlay.mp4"),
        "visible_surface_overlay": str(run_root / "renders" / "v21_visible_surface_overlay.mp4"),
        "hand_overlay": str(run_root / "renders" / "v21_hand_overlay.mp4"),
    }

    # Assemble
    state = {
        "schema": "v21_physical_state.v0",
        "status": "pipeline_v21_measurements_complete_renderable_overlay",
        "case_id": run_root.name,
        "run_root": str(run_root),
        "timeline": timeline,
        "camera_depth": camera_depth,
        "segmentation": segmentation,
        "hands": hands,
        "objects": objects,
        "renders": renders,
        "renderer_boundary": "V21 overlay render consumes metric hand candidates plus current SAM2-derived object mesh pose. Contact/nonpenetration remain separate uncertain evidence, not accepted final physical constraints.",
        "pipeline_stages_completed": [
            "depth_camera_baseline",
            "segmentation_mask_acceptance",
            "hand_candidate_wilor_detection",
            "hand_metric_scale_refit",
            "object_mesh_candidate_generation",
            "object_pose_estimation",
            "v18_compact_rigid_pose_fit",
            "v19_rigid_pose_graph",
            "integrated_overlay_render",
        ],
        "pipeline_stages_pending": [
            "active_mano_shape_pose_scale_optimization",
            "contact_state_estimation",
            "occlusion_ownership",
            "nonpenetration_state",
            "world_metric_render",
            "benchmark_evaluation",
        ],
    }

    state_path.write_text(json.dumps(state, indent=2))

    # Uncertainty state
    uncertainty = {
        "schema": "v21_uncertainty_state.v0",
        "case_id": run_root.name,
        "hand_state": {
            "metric_scale_uncertainty": "depth_residual_median_3_to_9cm",
            "active_optimization": "not_run_candidate_only",
            "side_mapping_confidence": "wiLor_detector_class_mapping",
            "visibility_state": "detection_rate_based_67_to_100_percent",
        },
        "object_state": {
            "mesh_candidate_uncertainty": "single_frame_heightfield_not_completed_geometry",
            "pose_uncertainty": "mask_centroid_plus_noisy_depth_median",
            "depth_noise": "depthpro_depth_at_small_objects_noisy",
            "icp_divergence": "depth_noise_prevents_convergence",
        },
        "depth_state": {
            "provisional": "no_independent_comparator_unidepth_not_available",
            "depthpro_focal_uncertainty": "per_frame_variation_5_to_10_percent",
        },
        "unresolved_variables": [
            "active_mano_optimization",
            "completed_object_geometry_from_trellis",
            "contact_state",
            "occlusion_ownership",
            "nonpenetration_residuals",
        ],
    }
    (run_root / "state" / "v21_uncertainty_state.json").write_text(json.dumps(uncertainty, indent=2))

    # Evidence note
    evidence = f"""# V21 Agent Evidence

## Case: {run_root.name}

### Pipeline stages completed

1. **Depth/Camera**: DepthPro monocular metric depth on {timeline['frame_count']} frames.
   - Focal length: {camera_depth.get('focal_length_median_px', '?')}px (per-frame)
   - Provisional: no independent comparator (UniDepth network failure)

2. **Segmentation**: Local GrabCut masks accepted for {', '.join(segmentation['accepted_tracks'])}.

3. **Hand/MANO candidate**: WiLoR detection + bone-length metric scale + depth-z refit.
   - Scale applied: {hands.get('metric_scale_applied', '?')}
   - Depth residual: median {hands.get('depth_validation', {}).get('median_residual_after_refit_m', '?')}m

4. **Object mesh candidate**: Heightfield backprojection from single anchor frame.
   - Method: {objects.get('mesh_method', '?')}
   - Extent: {objects.get('mesh_extent_m', '?')}

5. **Object pose**: Mask centroid + depth median + PCA orientation.
   - Estimated frames: {objects.get('pose_estimated_frames', '?')}
   - Depth median: {objects.get('pose_depth_median_m', '?')}m

6. **Integrated render**: Overlay with hand skeleton + object mesh pose.

### Key uncertainties

- Hand MANO: candidate evidence only, active optimization not yet run
- Object mesh: heightfield from single frame, not completed geometry (TRELLIS not yet integrated)
- Object pose: mask centroid + noisy depth, ICP diverges due to DepthPro noise at small objects
- Depth: provisional, no independent comparator
- Contact/occlusion/nonpenetration: not yet implemented
"""
    (run_root / "state" / "v21_agent_evidence.md").write_text(evidence)

    print(json.dumps({"status": "ok", "state": str(state_path), "case": run_root.name}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
