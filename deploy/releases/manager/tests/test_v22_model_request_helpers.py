from __future__ import annotations

import json
from pathlib import Path

from scripts.v22_model_request_helpers import write_droid_request, write_hawor_request, write_unidepth_request, write_vggt_camera_request, write_wilor_request
from scripts.v22_gpu_usage import record_gpu_snapshot


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    clip = run_root / "input" / "clips" / "case.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"video")
    write_json(run_root / "input" / "input_manifest.json", {"case_id": "case", "primary_video": str(clip)})
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {
            "frame_count": 2,
            "fps": 30.0,
            "video": {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 2, "duration_s": 2 / 30.0},
            "frames": [{"frame_idx": 0, "source_width": 1920, "source_height": 1080}],
        },
    )
    write_json(
        run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
        {"intrinsics_fx_fy_cx_cy": [1528.342616012172, 1528.342616012172, 960.0, 540.0], "intrinsics_source": "unidepth_median"},
    )
    return run_root


def test_model_requests_match_feishu_contract_shape(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    unidepth = write_unidepth_request(run_root)
    wilor = write_wilor_request(run_root)
    droid = write_droid_request(run_root)
    hawor = write_hawor_request(run_root)

    for payload, model in [(unidepth, "unidepth"), (wilor, "wilor"), (droid, "droid"), (hawor, "hawor")]:
        assert payload["schema"] == "ego.annotation.model_request.v1"
        assert payload["model"] == model
        assert payload["job_id"] == "case"
        assert payload["input_video"].endswith("case.mp4")
        assert payload["output_dir"]
        assert payload["video_meta"]["frame_count"] == 2

    assert "camera" not in unidepth
    assert "camera" not in wilor
    assert droid["camera"] == {
        "model": "pinhole",
        "intrinsics_px": [1528.342616012172, 1528.342616012172, 960.0, 540.0],
        "image_size": [1920, 1080],
        "distortion": None,
        "source": str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"),
        "source_method": "unidepth_median",
    }
    assert hawor["camera"]["intrinsics_px"] == droid["camera"]["intrinsics_px"]
    assert "droid_shared_manifest" not in hawor
    assert "droid_output_dir" not in hawor
    assert hawor["parameters"]["hawor_contract"]["droid_slam_included"] is False
    assert "raw_frame_manifest" in hawor["parameters"]["input_contract"]["required_fields"]
    assert droid["parameters"]["output_contract"]["shared_geometry_manifest"] == "droid_shared_geometry.json"
    assert droid["parameters"]["execution_contract"]["droid_instance_count_per_video"] == 1
    assert droid["parameters"]["execution_contract"]["hawor_service_must_not_instantiate_droid"] is True
    for name in ("unidepth", "wilor", "droid", "hawor"):
        assert (run_root / "requests" / f"{name}.json").exists()


def test_vggt_camera_request_can_target_droid_compatibility_path(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    request = write_vggt_camera_request(
        run_root,
        output_dir=run_root / "measurements" / "camera_trajectory" / "droid_full_frame",
        request_path=run_root / "requests" / "droid.json",
        backend="vggt_omega",
    )

    assert request["schema"] == "ego.annotation.model_request.v1"
    assert request["model"] == "vggt_omega_camera_geometry"
    assert request["stage"] == "D4_camera_trajectory"
    assert request["execution"]["mode"] == "resident_tensor_batch"
    assert request["execution"]["droid_replacement_candidate"] is True
    assert request["output_dir"].endswith("droid_full_frame")
    assert request["camera"]["intrinsics_px"] == [1528.342616012172, 1528.342616012172, 960.0, 540.0]
    assert request["parameters"]["batch_contract"]["tensor_shape"] == "[B,S,3,H,W]"
    assert request["parameters"]["batch_contract"]["requires_equal_sequence_length_or_bucketed_windows"] is True
    assert request["parameters"]["output_contract"]["writes_droid_compatible_camera_artifacts"] is True
    assert (run_root / "requests" / "droid.json").exists()


def test_gpu_snapshot_records_unavailable_nvidia_smi(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    snapshot = record_gpu_snapshot(run_root=run_root, stage="unidepth", phase="before", nvidia_smi="/missing/nvidia-smi")
    assert snapshot["status"] == "unavailable"
    rows = (run_root / "logs" / "gpu_usage_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["stage"] == "unidepth"
