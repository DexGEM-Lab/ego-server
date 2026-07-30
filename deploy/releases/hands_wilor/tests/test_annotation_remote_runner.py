from __future__ import annotations

from pathlib import Path

from scripts.annotation_remote_runner import RemoteConfig, build_pipeline_command, config_from_env


def test_remote_config_from_env() -> None:
    cfg = config_from_env({"ANNOTATION_REMOTE_HOST": "host", "ANNOTATION_REMOTE_REPO": "/repo"})
    assert cfg is not None
    assert cfg.host == "host"
    assert cfg.repo_root == Path("/repo")
    assert cfg.output_root == Path("/home/zjh/data/v22_api_jobs")


def test_build_pipeline_command_quotes_and_flags() -> None:
    cfg = RemoteConfig(
        host="host",
        repo_root=Path("/remote/repo"),
        output_root=Path("/remote/out"),
        upload_root=Path("/remote/uploads"),
        package_root=Path("/remote/packages"),
        python=Path("/remote/python"),
    )
    cmd = build_pipeline_command(
        cfg,
        job_id="job_001",
        remote_video=Path("/remote/uploads/job_001/input.mp4"),
        start_s=0.0,
        end_s=1.0,
        render_width=960,
        gpu_ids=None,
        run_preflight=False,
        run_camera_trajectory=True,
        run_hawor_metric_hands=True,
        run_hybrid_hands=True,
        run_gt_free_drift_self_calibration=True,
        run_captioning=True,
        run_self_consistency_qc=True,
        run_evaluator=True,
        actions_json=Path("/remote/uploads/job_001/actions.json"),
        captions_json=None,
        head_gt=None,
        hand_gt=None,
        write_product_bundle=True,
    )
    assert cmd.startswith("cd /remote/repo && /remote/python scripts/run_v22_minimal_annotation_pipeline.py")
    assert "--case-id job_001" in cmd
    assert "--input-video /remote/uploads/job_001/input.mp4" in cmd
    assert "--run-root /remote/out/job_001" in cmd
    assert "--run-camera-trajectory" in cmd
    assert "--run-hawor-metric-hands" in cmd
    assert "--run-hybrid-hands" in cmd
    assert "--run-gt-free-drift-self-calibration" in cmd
    assert "--run-captioning" in cmd
    assert "--run-self-consistency-qc" in cmd
    assert "--run-evaluator" in cmd
    assert "--actions-json /remote/uploads/job_001/actions.json" in cmd
    assert "--write-product-bundle" in cmd
    assert "--repo-root /remote/repo" in cmd
    assert "--render-width 960" in cmd


def test_build_pipeline_command_omits_render_width_when_none() -> None:
    cfg = RemoteConfig(
        host="host",
        repo_root=Path("/remote/repo"),
        output_root=Path("/remote/out"),
        upload_root=Path("/remote/uploads"),
        package_root=Path("/remote/packages"),
        python=Path("/remote/python"),
    )
    cmd = build_pipeline_command(
        cfg,
        job_id="job_002",
        remote_video=Path("/remote/uploads/job_002/input.mp4"),
        start_s=None,
        end_s=None,
        render_width=None,
        gpu_ids=None,
        run_preflight=False,
        run_camera_trajectory=True,
        run_hawor_metric_hands=True,
        run_hybrid_hands=True,
        run_gt_free_drift_self_calibration=True,
        run_captioning=True,
        run_self_consistency_qc=True,
        run_evaluator=True,
        actions_json=None,
        captions_json=None,
        head_gt=None,
        hand_gt=None,
        write_product_bundle=True,
    )
    assert "--render-width" not in cmd
