from __future__ import annotations

import json
from pathlib import Path

from ego_annotation import AnnotationJobRequest, AnnotationJobRunner


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_alpha_job_writes_manifest_metrics_and_explicit_degraded_calibration(tmp_path: Path) -> None:
    payload = {
        "job_id": "alpha_test_job",
        "video_uri": "memory://clip.mp4",
        "output_root": str(tmp_path),
        "media": {"frame_count": 3, "fps": 30.0, "width": 640, "height": 480, "duration_s": 0.1, "sha256": "abc"},
        "semantic_sources": [
            {
                "start_s": 0.0,
                "end_s": 0.1,
                "caption": "hand reaches toward tomato",
                "confidence": 0.8,
                "evidence_frames": [0, 1, 2],
            }
        ],
        "state_inputs": {
            "head_camera": [{"frame_idx": 0, "valid": True, "gauge": "metadata_supplied_example"}],
            "hand_states": [{"frame_idx": 0, "side": "right", "visibility": "visible", "source": "test"}],
        },
        "metric_observations": {
            "hand_wrist_root_error_m": [0.01, 0.02, 0.03],
            "hand_all_joint_mpjpe_m": [0.04, 0.05],
        },
        "throughput_observations": [
            {"module": "ingest", "input_duration_s": 10.0, "elapsed_s": 2.0, "gpu_hours_per_video_hour": 0.0, "status": "ok"}
        ],
    }
    result = AnnotationJobRunner().run(AnnotationJobRequest.from_mapping(payload))
    assert result.status in {"ok", "completed_with_degraded_outputs"}
    manifest = read_json(result.manifest_path)
    assert manifest["schema"] == "ego.annotation.output"
    assert manifest["request"]["public_endpoint"] == "/v1/annotation-jobs"
    calibration = read_json(Path(manifest["calibration_contract"]))
    assert calibration["status"] == "estimated_low_confidence"
    metrics_path = Path(manifest["tables"]["validation_metrics"]["ndjson"])
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    wrist = next(row for row in metrics if row["metric_id"] == "hand_wrist_root_error_m")
    assert wrist["status"] == "measured"
    assert wrist["summary"]["count"] == 3
    errors_path = Path(manifest["events"]["errors"]["ndjson"])
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["code"] == "calibration_estimated_from_image_size" for row in errors)


def test_alpha_job_refuses_silent_missing_calibration_and_state(tmp_path: Path) -> None:
    payload = {
        "job_id": "missing_contract_job",
        "video_uri": "memory://clip.mp4",
        "output_root": str(tmp_path),
        "allow_estimated_calibration": False,
        "media": {"frame_count": 1, "fps": 30.0, "width": 640, "height": 480, "sha256": "abc"},
    }
    result = AnnotationJobRunner().run(payload)
    assert result.status == "completed_with_errors"
    codes = {row["code"] for row in result.errors}
    assert "calibration_unresolved" in codes
    assert "head_camera_unavailable" in codes
    assert "hand_states_unavailable" in codes
    assert "semantic_clips_unavailable" in codes
    assert "throughput_measurements_unavailable" in codes
