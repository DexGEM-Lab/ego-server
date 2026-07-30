from __future__ import annotations

from ego_annotation.evaluators import evaluate_hands, evaluate_head_camera


def test_head_camera_evaluator_computes_ate_rpe_rotation_and_scale() -> None:
    pred = [
        {"frame_idx": 0, "t_world_camera_m": [0.0, 0.0, 0.0], "q_world_camera_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"frame_idx": 1, "t_world_camera_m": [1.1, 0.0, 0.0], "q_world_camera_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ]
    gt = [
        {"frame_idx": 0, "t_world_camera_m": [0.0, 0.0, 0.0], "q_world_camera_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"frame_idx": 1, "t_world_camera_m": [1.0, 0.0, 0.0], "q_world_camera_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ]
    obs = evaluate_head_camera(pred, gt)
    assert obs["head_camera_ate_translation_m"] == [0.0, 0.10000000000000009]
    assert obs["head_camera_rpe_translation_m"] == [0.10000000000000009]
    assert obs["head_camera_scale_error_ratio"] == [1.1]
    assert obs["head_camera_rotation_deg"] == [0.0, 0.0]


def test_hand_evaluator_computes_wrist_mpjpe_visibility_reprojection_and_jitter() -> None:
    pred = [
        {
            "frame_idx": 0,
            "side": "right",
            "wrist_t_camera_m": [0.0, 0.0, 0.0],
            "joints_camera_m": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "vertices_camera_m": [[0.0, 0.0, 0.0]],
            "visibility": "visible",
            "reprojection_error_px": 5.0,
        },
        {
            "frame_idx": 1,
            "side": "right",
            "wrist_t_camera_m": [0.1, 0.0, 0.0],
            "visibility": "visible",
        },
    ]
    gt = [
        {
            "frame_idx": 0,
            "side": "right",
            "wrist_t_camera_m": [0.0, 0.1, 0.0],
            "joints_camera_m": [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
            "vertices_camera_m": [[0.0, 0.1, 0.0]],
            "visibility": "occluded",
        }
    ]
    obs = evaluate_hands(pred, gt)
    assert obs["hand_wrist_root_error_m"] == [0.1]
    assert obs["hand_all_joint_mpjpe_m"] == [0.1]
    assert obs["hand_root_relative_mpjpe_m"] == [0.0]
    assert obs["hand_mpvpe_surface_m"] == [0.1]
    assert obs["visibility_state_accuracy"] == [0.0]
    assert obs["hand_reprojection_error_px"] == [5.0]
    assert obs["temporal_wrist_jitter_m_per_frame"] == [0.1]
