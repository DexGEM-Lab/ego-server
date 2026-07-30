#!/usr/bin/env bash
# V21 batch helper over existing run roots.
# Pi remains the harness; this script is only a tool for rerunning selected stages.
# It does not by itself prove that all V18/V19/V21 algorithms are complete.

set -e

REPO="${V21_REMOTE_REPO_ROOT:-$HOME/ego-annotaion-jiahong-dev}"
PYTHON_HAMER=/home/zjh/miniconda3/envs/hamer/bin/python
PYTHON_EGO=/home/zjh/miniconda3/envs/ego_foundation/bin/python
WILOR_DIR=/mnt/user-home/zjh/ego-pipeline/v21_model_work/wilor_model
TRELLIS_DIR=/mnt/user-home/zjh/ego-pipeline/v21_model_work/trellis
COMPUTE_TARGET="${V21_COMPUTE_TARGET:-ssh -p 57938 zjh@115.190.235.210}"

mkdir -p $REPO/outputs/v21_infer_20260626/pipeline_status

run_stage() {
    local name=$1
    local case=$2
    shift 2
    local status_file="$REPO/outputs/v21_infer_20260626/pipeline_status/${case}_${name}.txt"
    
    echo "[$(date +%H:%M:%S)] Running: $name ($case)..."
    if "$@" > "$REPO/outputs/v21_infer_20260626/pipeline_status/${case}_${name}.log" 2>&1; then
        echo "OK" > "$status_file"
        echo "[$(date +%H:%M:%S)] ✓ $name ($case) succeeded"
    else
        echo "FAILED ($?)" > "$status_file"
        echo "[$(date +%H:%M:%S)] ✗ $name ($case) FAILED - see log"
    fi
}

echo "V21 Selected Batch Helper"
echo "==========================="
echo "Pi is the harness; this script only runs selected reusable tools."

for CASE_INFO in "pico_trackers_10_2_100s_120s_v21:red_scandic_tin" "living_room_cleanup_multiview_v21:clear_glass_bowl"; do
    CASE=${CASE_INFO%%:*}
    OBJ=${CASE_INFO##*:}
    RUN=$REPO/outputs/v21_infer_20260626/$CASE
    ANN=$RUN/state/annotations_v18_full_mano.json
    if [ ! -f "$ANN" ]; then
        ANN=$RUN/state/annotations_v18_compatible.json
    fi

    echo ""
    echo "=== CASE: $CASE (obj=$OBJ) ==="

    # Phase 0: V21 default bbox replacement. GroundingDINO is disabled.
    run_stage "owlv2_bbox_proposals" $CASE $PYTHON_HAMER $REPO/scripts/run_v21_owlv2_bbox_proposals.py \
        --run-root $RUN --object-plan $RUN/measurements/object_candidates/object_plan_seed_from_v20_visual_qc.json \
        --output $RUN/measurements/object_candidates/owlv2_bbox_proposals.json \
        --compute-target "$COMPUTE_TARGET" || true

    # Phase 1: Observation sources
    run_stage "rtmlib_2d" $CASE $PYTHON_HAMER $REPO/scripts/run_rtmlib_hand2d_v3.py \
        --input-video $RUN/input/clips/*.mp4 --output-dir $RUN/measurements/hand_candidates/rtmlib_2d || true

    run_stage "hamer_recon" $CASE $PYTHON_HAMER $REPO/scripts/run_hamer_rtmlib_hand_stream_v3.py \
        --clip $RUN/input/clips/*.mp4 --output-dir $RUN/measurements/hand_candidates/hamer || true

    run_stage "merge_hand_streams" $CASE $PYTHON_HAMER $REPO/scripts/merge_hand_candidate_streams_v7.py \
        --output-dir $RUN/measurements/hand_candidates/merged || true

    run_stage "refit_mano_depth" $CASE $PYTHON_HAMER $REPO/scripts/refit_mano_metric_depth_v3.py \
        --annotations $ANN --metric-depth-npz $RUN/measurements/depth_candidates/depthpro_full_frame/depthpro_full_frame_depth_v21.npz \
        --output-dir $RUN/measurements/hand_candidates/mano_refit || true
    
    # Phase 1: Object geometry
    run_stage "heightfield_completion" $CASE $PYTHON_HAMER $REPO/scripts/complete_object_heightfield_from_mask_depth_v3.py \
        --dataset $RUN/measurements/object_geometry/heightfield_observed/$OBJ/dataset \
        --manifest $RUN/input/raw_frame_manifest/manifest.json \
        --output-dir $RUN/measurements/object_geometry/heightfield_completed/$OBJ || true
    
    # Phase 2: Harness validation
    run_stage "geometry_registry" $CASE $PYTHON_HAMER $REPO/scripts/build_v20_geometry_candidate_registry.py \
        --run-root $RUN --output $RUN/measurements/object_geometry/candidate_registry.json || true
    
    # Phase 3: Factor graphs (already ran ICP + V19 graph)
    run_stage "mesh_prior_graph" $CASE $PYTHON_HAMER $REPO/scripts/optimize_mesh_prior_pose_graph_v3.py \
        --mesh-prior-camera $RUN/measurements/object_geometry/v21_mesh_candidate/$OBJ/mesh_candidate.obj \
        --observed-mesh-npz $RUN/measurements/object_geometry_mesh_pose/$OBJ/observed_mesh_archive/observed_mask_depth_meshes_world.npz \
        --dataset $RUN/measurements/object_geometry/heightfield_observed/$OBJ/dataset \
        --manifest $RUN/input/raw_frame_manifest/manifest.json \
        --annotations $ANN \
        --output-dir $RUN/measurements/object_geometry_mesh_pose/$OBJ/v3_mesh_prior_graph \
        --frame-start 0 --frame-end 99 --anchor-frame 193 --max-nfev 20 || true
    
    run_stage "joint_mano_object" $CASE $PYTHON_HAMER $REPO/scripts/optimize_joint_mano_object_graph_v3.py \
        --annotations $ANN \
        --observed-mesh-npz $RUN/measurements/object_geometry_mesh_pose/$OBJ/observed_mesh_archive/observed_mask_depth_meshes_world.npz \
        --mesh-prior $RUN/measurements/object_geometry/v21_mesh_candidate/$OBJ/mesh_candidate.obj \
        --output-dir $RUN/measurements/object_geometry_mesh_pose/$OBJ/v3_joint_mano_object \
        --frame-start 0 --frame-end 99 || true
    
    # Phase 4: Contact/occlusion/nonpenetration
    run_stage "mesh_contact_evidence" $CASE $PYTHON_HAMER $REPO/scripts/build_v18_mesh_contact_evidence.py \
        --full-pipeline-root $RUN --v16-root $RUN \
        --cases $CASE --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_mesh_contact || true
    
    run_stage "contact_ownership_graph" $CASE $PYTHON_HAMER $REPO/scripts/build_v18_contact_ownership_graph.py \
        --mesh-contact-root $RUN/measurements/contact_occlusion_nonpenetration/v18_mesh_contact \
        --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_contact_ownership \
        --cases $CASE || true
    
    run_stage "occlusion_owner_graph" $CASE $PYTHON_HAMER $REPO/scripts/build_v18_occlusion_owner_graph.py \
        --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_occlusion_owner \
        --cases $CASE || true
    
    run_stage "signed_nonpenetration" $CASE $PYTHON_HAMER $REPO/scripts/build_v18_signed_nonpenetration_evidence.py \
        --contact-ownership-root $RUN/measurements/contact_occlusion_nonpenetration/v18_contact_ownership \
        --full-pipeline-root $RUN \
        --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_signed_nonpenetration \
        --cases $CASE || true
    
    run_stage "mano_object_constraint" $CASE $PYTHON_HAMER $REPO/scripts/build_v18_mano_object_constraint_state.py \
        --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_mano_object_constraint \
        --cases $CASE || true
    
    run_stage "apply_mano_object_constraint" $CASE $PYTHON_HAMER $REPO/scripts/apply_v18_mano_object_constraint_state.py \
        --output-root $RUN/measurements/contact_occlusion_nonpenetration/v18_apply_constraint \
        --cases $CASE || true
    
    # Phase 5: Rendering
    run_stage "render_overlay" $CASE $PYTHON_HAMER $REPO/scripts/render_v21_v18_compositor.py \
        --run-root $RUN --object-id $OBJ || true
done

run_stage "atomic_overlay_audit" all $PYTHON_HAMER $REPO/scripts/audit_v21_atomic_algorithm_overlays.py \
    --overlay-root $REPO/outputs/v21_per_algorithm_results \
    --output $REPO/outputs/v21_per_algorithm_results/atomic_algorithm_overlay_audit.json || true

echo ""
echo "=== PIPELINE STATUS SUMMARY ==="
cat $REPO/outputs/v21_infer_20260626/pipeline_status/*.txt 2>/dev/null
