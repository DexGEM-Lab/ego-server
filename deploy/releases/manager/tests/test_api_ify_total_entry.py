from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.annotation_admission_proxy import admission_limits
from scripts.annotation_remote_runner import RemoteConfig, build_pipeline_command
from scripts.package_v22_annotation_result import PackageError, create_result_package
from scripts import run_single_video_api
from scripts.run_single_video_api import DROID_FINALIZE_CAPTURE_ENV, STAGES, annotation_client_driver_config, attach_cosmos_capture_hashes, parse_args, parse_service_origins, stage_capture_limits_from_environment
from ego_annotation.full_video_timeline import _droid_prefix_coverage
from scripts.run_v22_api_job_with_admission import API_IFY_STAGE_IDS


def remote_config(tmp_path: Path) -> RemoteConfig:
    return RemoteConfig(
        host="a800",
        repo_root=Path("/remote/repo"),
        output_root=Path("/remote/output"),
        upload_root=Path("/remote/uploads"),
        package_root=tmp_path / "packages",
        python=Path("/remote/python"),
    )


def test_annotation_client_defaults_to_truthful_diagnostic_monocular_droid(tmp_path: Path) -> None:
    args = parse_args([
        "--input", str(tmp_path / "input.mp4"),
        "--case-id", "task-4",
        "--run-root", str(tmp_path / "run"),
        "--fps-condition", "unidepth_full__droid_15fps",
    ])
    config = annotation_client_driver_config(
        fps_condition=args.fps_condition,
        frame_store_spill_dir=args.run_root / "frame_store",
    )

    assert args.diagnostic_monocular is True
    assert config.require_rgbd_capability is False
    assert config.allow_monocular_droid_smoke is True
    assert config.fps_condition == "unidepth_full__droid_15fps"
    assert config.droid_fps == 15.0


def test_official_api_ify_lane_accepts_wrapper_injected_cosmos_origin() -> None:
    origin = "http://127.0.0.1:34567"
    parsed = parse_service_origins(json.dumps({stage: origin for stage in API_IFY_STAGE_IDS}))
    assert "cosmos3.reason" in STAGES
    assert parsed["cosmos3.reason"] == origin


def test_cosmos_attempt_trace_links_exact_capture_hashes(tmp_path: Path) -> None:
    captures = tmp_path / "stage_captures"
    manifest = captures / "cosmos3_reason" / "request" / "response" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "ownership": {"scope": "cosmos3.reason:coarse:0000:repair:1"},
        "request": {"sha256": "request-sha"},
        "response": {"sha256": "response-sha"},
    }), encoding="utf-8")

    attempts = attach_cosmos_capture_hashes([
        {"scope": "cosmos3.reason:coarse:0000", "repair_count": 0},
        {"scope": "cosmos3.reason:coarse:0000:repair:1", "repair_count": 1},
    ], captures)

    assert "capture" not in attempts[0]
    capture = {
        "request_sha256": "request-sha",
        "response_sha256": "response-sha",
        "manifest": "cosmos3_reason/request/response/manifest.json",
    }
    assert attempts[1]["capture"] == capture
    anomaly = attach_cosmos_capture_hashes(
        [{"request_scope": "cosmos3.reason:coarse:0000:repair:1", "raw_field": "LA", "raw_value": "black"}],
        captures,
        scope_key="request_scope",
    )
    assert anomaly[0]["raw_response_capture"] == capture


def test_total_entry_parses_explicit_api_ify_diagnostic_request() -> None:
    pytest.importorskip("fastapi")
    from scripts.serve_v22_annotation_api import annotation_request_from_payload, ensure_total_backend_supported

    request = annotation_request_from_payload(
        {
            "video_uri": "/video.mp4",
            "job_id": "single-api",
            "model_backend": "api_ify",
            "diagnostic_monocular": True,
        },
        multipart=False,
    )

    assert request.model_backend == "api_ify"
    assert request.diagnostic_monocular is True
    ensure_total_backend_supported(request)


def test_total_entry_rejects_unproven_strict_api_ify_request() -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from scripts.serve_v22_annotation_api import AnnotationJobRequest, ensure_total_backend_supported

    request = AnnotationJobRequest(
        video_uri="/video.mp4",
        model_backend="api_ify",
        diagnostic_monocular=False,
    )

    with pytest.raises(HTTPException) as error:
        ensure_total_backend_supported(request)

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "api_ify_strict_rgbd_capability_unproven"


def test_remote_total_entry_routes_frozen_runner_through_manager(tmp_path: Path) -> None:
    command = build_pipeline_command(
        remote_config(tmp_path),
        job_id="single-api",
        remote_video=Path("/remote/uploads/single-api/video.mp4"),
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
        model_backend="api_ify",
        diagnostic_monocular=True,
    )

    assert "PYTHONPATH=/remote/repo${PYTHONPATH:+:$PYTHONPATH}" in command
    assert "scripts/run_single_video_api.py" in command
    assert "scripts/run_v22_minimal_annotation_pipeline.py" not in command
    assert "run_v22_api_job_with_admission.py" in command
    assert "--api-ify" in command
    assert "--diagnostic-monocular" in command
    assert "--run-hawor-metric-hands" not in command
    assert "cosmos3.reason" in API_IFY_STAGE_IDS
    assert "run_v22_cosmos_captioning_source.py" not in command


def test_manager_fixed_algorithm_admission_limits_are_retired() -> None:
    assert admission_limits(2) == {}


def test_1440_frame_partial_droid_manifest_uses_explicit_camera_coverage_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated product manifest must expose the unsubmitted DROID tail."""
    coverage = _droid_prefix_coverage(1440)
    state = type("State", (), {
        "semantic_rows": (),
        "semantic_review": {"attempts": [], "anomaly_ledger": []},
        "batch_request_traces": (),
        "module_timings_s": {"frame_store": 1.0, "unidepth": 2.0, "hands": 3.0, "wilor_build": 4.0, "wilor_service": 5.0, "droid": 6.0, "hawor": 7.0, "infiller": 8.0, "cosmos": 9.0},
        "unidepth_records": (),
        "hands_records": (),
        "wilor_records": (),
        "droid_records": type("Droid", (), {
            "coverage": coverage,
            "create_results": (),
            "push_results_by_attempt": ((),),
            "finalize_results": (),
        })(),
        "hawor_records": (),
        "infiller_windows": (),
        "hawor_geometry_diagnostics": {"status": "ok", "hawor_degenerate_slot_count": 0, "hawor_degenerate_slots": [], "hawor_chunks_with_degenerate_geometry": []},
        "semantic_request_count": 1,
        "semantic_status": "completed",
        "acceptance": type("Acceptance", (), {
            "accepted": True,
            "diagnostic_only": False,
            "scale_mode": "metric",
            "reasons": (),
        })(),
    })()
    result = type("Physical", (), {
        "frame_count": 1440,
        "duration_s": 48.0,
        "combined_video": "renders/v22_combined.mp4",
        "state_npz": "state/v22_physical_state.npz",
        "report_json": "renders/physical_report.json",
    })()

    monkeypatch.setattr(run_single_video_api, "preflight_single_video", lambda *_args, **_kwargs: type("Preflight", (), {"checks": ()})())
    fake_source = type("FrameSource", (), {"frame_store_report": lambda self: {"status": "built"}})()
    monkeypatch.setattr(run_single_video_api, "OpenCvFrameSource", type("Source", (), {"from_video": staticmethod(lambda _path: fake_source)}))
    monkeypatch.setattr(run_single_video_api, "ApiBackend", lambda _config: object())
    monkeypatch.setattr(run_single_video_api, "FullVideoTimelineDriver", lambda *_args: type("Driver", (), {"run": lambda *_args, **_kwargs: state})())
    monkeypatch.setattr(run_single_video_api, "PhysicalArtifactAdapter", lambda: type("Adapter", (), {"render": lambda *_args: result})())
    import ego_annotation.full_video_timeline as timeline
    monkeypatch.setattr(timeline, "LiveFrozenApiStageClient", lambda _backend: object())

    run_root = tmp_path / "run"
    payload = run_single_video_api.run(type("Args", (), {
        "input": tmp_path / "input.mp4",
        "case_id": "case-1440",
        "run_root": run_root,
        "diagnostic_monocular": True,
        "timeout_s": 1.0,
        "service_origins_json": "{}",
        "fps_condition": "unidepth_10fps__droid_10fps",
    })())
    manifest = json.loads((run_root / "annotation_pipeline_manifest.json").read_text(encoding="utf-8"))

    assert payload["status"] == "ok"
    assert manifest["status"] == "ok"
    assert manifest["droid"]["status"] == "completed_source_keyed_session_dag"
    assert manifest["droid"]["effective_unique_coverage_count"] == 1440
    assert manifest["droid"]["actual_pushed_count"] == 1888
    assert payload["renders"] == {"combined": "renders/v22_combined.mp4"}
    assert manifest["renders"] == {"v22_combined": "renders/v22_combined.mp4", "render_source": "PhysicalArtifactAdapter"}
    assert payload["cosmos"]["captioned_combined_video"] == "renders/v22_combined.mp4"
    assert payload["hand_geometry_diagnostics"]["hawor_degenerate_slot_count"] == 0
    assert manifest["hand_geometry_diagnostics"] == payload["hand_geometry_diagnostics"]
    assert set(payload["module_timings_s"]) == {"frame_store", "unidepth", "hands", "wilor_build", "wilor_service", "droid", "hawor", "infiller", "cosmos", "render"}
    assert manifest["module_timings_s"] == payload["module_timings_s"]
    assert payload["performance"]["service_batch_trace_status"] == "unavailable_without_complete_service_batch_trace"
    report_path = Path(payload["integrated_run_report"]["json"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["service_batches"]["status"] == "unavailable"
    assert manifest["integrated_run_report_path"] == str(report_path)


def test_api_ify_manifest_has_a_narrow_download_package(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    renders = run_root / "renders"
    state = run_root / "state"
    captures = run_root / "stage_captures"
    renders.mkdir(parents=True)
    state.mkdir()
    captures.mkdir()
    (renders / "v22_combined.mp4").write_bytes(b"combined-render")
    (renders / "physical_adapter_report.json").write_text("{}", encoding="utf-8")
    (state / "v22_physical_state.npz").write_bytes(b"npz")
    semantic = state / "semantic_clips"
    semantic.mkdir()
    (semantic / "v22_cosmos_semantic_review.json").write_text('{"semantic_rows":[{}]}', encoding="utf-8")
    (run_root / "run_result.json").write_text("{}", encoding="utf-8")
    (run_root / "integrated_run_report.json").write_text('{"schema":"ego.annotation.integrated_run_report.v1"}', encoding="utf-8")
    (run_root / "integrated_run_report.md").write_text("# Integrated run timing report\n", encoding="utf-8")
    (captures / "fixture_index.json").write_text('{"entries":[]}', encoding="utf-8")
    (run_root / "annotation_pipeline_manifest.json").write_text(
        json.dumps(
            {
                "schema": "ego.annotation.pipeline_manifest.v1",
                "pipeline": "single_video_api_ify",
                "case_id": "single-api",
                "renders": {"v22_combined": "renders/v22_combined.mp4"},
                "integrated_run_report_path": "integrated_run_report.json",
            }
        ),
        encoding="utf-8",
    )

    result = create_result_package(run_root, tmp_path / "packages", package_name="single-api")

    with zipfile.ZipFile(result["package_path"]) as archive:
        names = set(archive.namelist())
    assert "v22_combined.mp4" in names
    assert not any(name.endswith(".mp4") and name != "v22_combined.mp4" for name in names)
    assert "state/v22_physical_state.npz" in names
    assert "integrated_run_report.json" in names
    assert "integrated_run_report.md" in names
    assert result["final_report_path"] == str(run_root / "integrated_run_report.json")
    assert result["final_video_path"] == str(renders / "v22_combined.mp4")
    assert "stage_captures/fixture_index.json" in names
    assert "state/semantic_clips/v22_cosmos_semantic_review.json" in names
    assert "state/timing_ledger.json" not in names


def test_droid_finalize_capture_limit_is_opt_in_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DROID_FINALIZE_CAPTURE_ENV, raising=False)
    assert stage_capture_limits_from_environment() == {"cosmos3.reason": 128}
    monkeypatch.setenv(DROID_FINALIZE_CAPTURE_ENV, "0")
    assert stage_capture_limits_from_environment() == {"cosmos3.reason": 128}
    monkeypatch.setenv(DROID_FINALIZE_CAPTURE_ENV, "2")
    assert stage_capture_limits_from_environment() == {"cosmos3.reason": 128, "droid.finalize": 2}
    monkeypatch.setenv(DROID_FINALIZE_CAPTURE_ENV, "invalid")
    with pytest.raises(ValueError, match=DROID_FINALIZE_CAPTURE_ENV):
        stage_capture_limits_from_environment()
