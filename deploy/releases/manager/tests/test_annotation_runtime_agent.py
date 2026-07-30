from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.annotation_runtime_agent import build_runtime_bundle, build_runtime_request
from scripts.execute_annotation_runtime_request import RuntimeRequestError, build_pipeline_cmd, execute


class DummyRemoteConfig:
    host = "host"
    repo_root = Path("/remote/repo")
    output_root = Path("/remote/out")
    upload_root = Path("/remote/uploads")
    package_root = Path("/remote/packages")
    python = Path("/remote/python")


def test_runtime_request_is_single_prediction_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "jobs" / "job_001"
    payload = build_runtime_request(
        repo_root=tmp_path / "repo",
        job_id="job_001",
        video_uri=str(tmp_path / "upload.mp4"),
        run_root=run_root,
        package_root=tmp_path / "downloads",
        local_video=True,
        remote_config=DummyRemoteConfig(),
        timeout_s=7200,
        metadata={"status": "ok", "frame_count": 60, "fps": 30.0, "width": 1920, "height": 1080},
        pipeline_flags={"run_camera_trajectory": True, "run_hawor_metric_hands": True, "run_evaluator": True},
    )
    assert payload["schema"] == "ego.annotation.runtime_request.v1"
    assert payload["execution_backend"] == "remote_ssh_script"
    assert payload["model_requests"]["expected_files"] == ["unidepth.json", "wilor.json", "droid.json", "hawor.json"]
    assert "head_gt" not in payload
    assert "hand_gt" not in payload
    assert payload["claim_scope"].startswith("Single uploaded video")

    paths = build_runtime_bundle(repo_root=tmp_path / "repo", run_root=tmp_path / "agent" / "job_001", request_payload=payload)
    assert paths["request"].exists()
    assert paths["prompt"].read_text(encoding="utf-8").count("execute_annotation_runtime_request.py") == 1
    assert "Do not inspect task memory" in paths["system_prompt"].read_text(encoding="utf-8")


def test_runtime_executor_passes_explicit_droid_backend(tmp_path: Path) -> None:
    request = {
        "job_id": "job_001",
        "run_root": str(tmp_path / "run"),
        "repo_root": str(tmp_path / "repo"),
        "video_uri": str(tmp_path / "video.mp4"),
        "pipeline_flags": {"camera_backend": "droid", "run_camera_trajectory": True, "run_hawor_metric_hands": True},
    }
    cmd = build_pipeline_cmd(request)
    assert cmd[cmd.index("--camera-backend") + 1] == "droid"
    assert "--run-camera-trajectory" in cmd
    assert "--run-hawor-metric-hands" in cmd


def test_runtime_executor_rejects_feishu_ray_before_script_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = tmp_path / "runtime_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.runtime_request.v1",
                "execution_backend": "local_script",
                "run_root": str(tmp_path / "run"),
                "repo_root": str(tmp_path / "repo"),
                "video_uri": str(tmp_path / "video.mp4"),
                "job_id": "job_001",
                "pipeline_flags": {
                    "model_backend": "feishu_ray",
                    "service_profile": "profile-a",
                    "service_endpoints": {"unidepth": "http://127.0.0.1:28000"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.execute_annotation_runtime_request.execute_local", lambda _request: pytest.fail("script dispatch attempted"))

    with pytest.raises(RuntimeRequestError, match="feishu_ray_pipeline_adapter_not_implemented"):
        execute(request, tmp_path / "result.json")


def test_runtime_executor_rejects_service_configuration_for_script_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = tmp_path / "runtime_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.runtime_request.v1",
                "execution_backend": "local_script",
                "run_root": str(tmp_path / "run"),
                "repo_root": str(tmp_path / "repo"),
                "video_uri": str(tmp_path / "video.mp4"),
                "job_id": "job_001",
                "pipeline_flags": {
                    "model_backend": "script",
                    "service_profile": "feishu_ray_a800_server_local",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.execute_annotation_runtime_request.execute_local", lambda _request: pytest.fail("script dispatch attempted"))

    with pytest.raises(RuntimeRequestError, match="service_configuration_requires_feishu_ray"):
        execute(request, tmp_path / "result.json")


def test_runtime_executor_rejects_unknown_backend(tmp_path: Path) -> None:
    request = tmp_path / "runtime_request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "ego.annotation.runtime_request.v1",
                "execution_backend": "typo_backend",
                "run_root": str(tmp_path / "run"),
                "repo_root": str(tmp_path / "repo"),
                "video_uri": str(tmp_path / "video.mp4"),
                "job_id": "job_001",
                "pipeline_flags": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeRequestError, match="unsupported execution_backend: typo_backend"):
        execute(request, tmp_path / "result.json")
