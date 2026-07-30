from __future__ import annotations

from pathlib import Path

from scripts import annotation_remote_runner as runner
from scripts.annotation_remote_runner import RemoteConfig, build_pipeline_command, build_stage_batch_command, config_from_env


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
        semantic_review_json=Path("/remote/uploads/job_001/v22_semantic_agent_review.json"),
        head_gt=None,
        hand_gt=None,
        write_product_bundle=True,
    )
    assert cmd.startswith("cd /remote/repo && /remote/python scripts/run_v22_api_job_with_admission.py")
    assert " -- " in cmd
    assert "/remote/python scripts/run_v22_minimal_annotation_pipeline.py" in cmd
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
    assert "--semantic-review-json /remote/uploads/job_001/v22_semantic_agent_review.json" in cmd
    assert "--write-product-bundle" in cmd
    assert "--repo-root /remote/repo" in cmd
    assert "--render-width 960" in cmd
    assert "--algorithm-inflight-multiplier 2" in cmd
    assert "--lock-root /remote/out/_algorithm_admission" in cmd


def test_build_pipeline_command_preserves_backend_and_endpoint_selection() -> None:
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
        job_id="job-feishu",
        remote_video=Path("/remote/uploads/job-feishu/input.mp4"),
        start_s=None,
        end_s=None,
        render_width=None,
        gpu_ids=None,
        run_preflight=False,
        run_camera_trajectory=True,
        run_hawor_metric_hands=True,
        run_hybrid_hands=True,
        run_gt_free_drift_self_calibration=True,
        run_captioning=False,
        run_self_consistency_qc=True,
        run_evaluator=True,
        actions_json=None,
        captions_json=None,
        semantic_review_json=None,
        head_gt=None,
        hand_gt=None,
        write_product_bundle=True,
        model_backend="feishu_ray",
        service_profile="feishu_ray_a800_server_local",
        service_endpoints={"droid": "http://127.0.0.1:28002"},
    )
    assert "--model-execution feishu_ray" in cmd
    assert "--feishu-service-profile /remote/repo/configs/feishu_ray_services.json" in cmd
    assert "--feishu-droid-base-url http://127.0.0.1:28002" in cmd
    assert "--upstream-endpoints-json '{\"droid\": \"http://127.0.0.1:28002\"}'" in cmd


def test_build_stage_batch_command_uses_video_list_and_resident_flags() -> None:
    cfg = RemoteConfig(
        host="host",
        repo_root=Path("/remote/repo"),
        output_root=Path("/remote/out"),
        upload_root=Path("/remote/uploads"),
        package_root=Path("/remote/packages"),
        python=Path("/remote/python"),
    )
    cmd = build_stage_batch_command(
        cfg,
        job_id="set_001",
        data_root=Path("/remote/uploads/set_001"),
        batch_root=Path("/remote/out/set_001"),
        max_items=2,
        video_list=Path("/remote/uploads/set_001/selected_video_paths.txt"),
        item_agents=16,
        gpu_count=8,
        gpu_ids="0,1,2,3",
        prepare_workers=8,
        calibration_workers=8,
        run_resident_droid=True,
        run_resident_hawor=True,
    )
    assert cmd.startswith("cd /remote/repo && /remote/python scripts/run_v22_stage_batch_job.py")
    assert "--data-root /remote/uploads/set_001" in cmd
    assert "--batch-root /remote/out/set_001" in cmd
    assert "--video-list /remote/uploads/set_001/selected_video_paths.txt" in cmd
    assert "--max-items 2" in cmd
    assert "--item-agents 16" in cmd
    assert "--gpu-ids 0,1,2,3" in cmd
    assert "--run-resident-droid" in cmd
    assert "--run-resident-hawor" in cmd


def test_remote_job_set_writes_structured_video_items(monkeypatch, tmp_path: Path) -> None:
    cfg = RemoteConfig(
        host="host",
        repo_root=Path("/remote/repo"),
        output_root=Path("/remote/out"),
        upload_root=Path("/remote/uploads"),
        package_root=Path("/remote/packages"),
        python=Path("/remote/python"),
    )
    commands: list[str] = []

    def fake_ssh(config: RemoteConfig, command: str, *, timeout: int | None = None) -> str:
        commands.append(command)
        if command.startswith("cat /remote/out/set_001/reports/batch_summary.json"):
            return '{"entry_count":1,"manifest_counts":{"completed":1,"failed":0},"artifact_counts":{"packages_for_manifest_completed":1,"completed_with_required_artifacts":1}}'
        if command.startswith("cat /remote/out/set_001/reports/resident_model_summary.json"):
            return '{"stage_satisfaction":{"unidepth_v2_depth_resident":"satisfied_true_resident_tensor_batch","wilor_v21_hand_candidates_resident":"satisfied_true_resident_frame_and_crop_batch","droid_camera_trajectory":"satisfied_resident_sequence_batch","hawor_metric_hands":"partial_resident_submodels_complete_item_coverage"}}'
        return ""

    def fake_scp_from(config: RemoteConfig, remote: str, local: Path, *, timeout: int | None = None) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("zip", encoding="utf-8")

    monkeypatch.setattr(runner, "ssh", fake_ssh)
    monkeypatch.setattr(runner, "scp_from", fake_scp_from)

    result = runner.run_remote_annotation_job_set(
        config=cfg,
        job_id="set_001",
        data_root=Path("/remote/uploads/set_001"),
        local_package_root=tmp_path,
        max_items=None,
        video_uris=["/remote/data/a.mp4"],
        video_items=[{"video_uri": "/remote/data/a.mp4", "item_id": "clip_a", "metadata": {"take": 1}}],
        item_agents=16,
        gpu_count=8,
        gpu_ids="0,1",
        prepare_workers=8,
        calibration_workers=8,
        run_resident_droid=True,
        run_resident_hawor=True,
    )

    assert result["status"] == "completed"
    assert result["summary"]["status"] == "completed"
    assert result["summary"]["completion_criteria"]["stages_satisfied"] is True
    assert result["summary"]["entry_count"] == 1
    assert any("selected_video_items.json" in command and '"item_id": "clip_a"' in command for command in commands)
    run_commands = [command for command in commands if "scripts/run_v22_stage_batch_job.py" in command]
    assert len(run_commands) == 2
    assert all("--video-list /remote/uploads/set_001/selected_video_items.json" in command for command in run_commands)


def test_derive_job_set_completion_requires_artifacts_and_stage_evidence() -> None:
    result = runner.derive_job_set_completion(
        {"entry_count": 1, "manifest_counts": {"completed": 1, "failed": 0}, "artifact_counts": {"packages_for_manifest_completed": 0, "completed_with_required_artifacts": 0}},
        {"stage_satisfaction": {"unidepth_v2_depth_resident": "satisfied_true_resident_tensor_batch", "wilor_v21_hand_candidates_resident": "missing"}},
        run_resident_droid=False,
        run_resident_hawor=False,
    )
    assert result["status"] == "incomplete"
    assert result["items_complete"] is False
    assert result["stages_satisfied"] is False
    assert result["observed_stage_statuses"]["wilor_v21_hand_candidates_resident"] == "missing"


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
        semantic_review_json=None,
        head_gt=None,
        hand_gt=None,
        write_product_bundle=True,
    )
    assert "--render-width" not in cmd
    assert "scripts/run_v22_api_job_with_admission.py" in cmd
