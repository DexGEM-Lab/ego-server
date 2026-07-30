from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_v22_captioning_stage import build_captions, parse_args as parse_caption_args
from scripts.run_v22_evaluator_stage import parse_args as parse_eval_args, run as run_eval
from scripts.run_v22_gt_free_drift_self_calibration_stage import estimate, parse_args as parse_d8_args
from scripts.run_v22_self_consistency_qc_stage import parse_args as parse_qc_args, run as run_qc


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    frames = [{"frame_idx": i, "time_s": i / 30, "manifest_width": 640, "manifest_height": 480} for i in range(2)]
    write_json(run_root / "input/raw_frame_manifest/manifest.json", {"frame_count": 2, "fps": 30.0, "video": {"fps": 30.0, "duration_s": 2 / 30, "frame_count": 2}, "frames": frames})
    write_json(run_root / "state/calibration/v19_camera_calibration_contract.json", {"intrinsics_fx_fy_cx_cy": [100.0, 100.0, 50.0, 50.0], "intrinsics_source": "test"})
    write_json(
        run_root / "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json",
        {
            "frames": [
                {"frame_idx": 0, "raw_hands": [{"side": "left", "detector_score": 0.9, "joints2d": [[50.0, 50.0] for _ in range(21)]}]},
                {"frame_idx": 1, "raw_hands": [{"side": "left", "detector_score": 0.9, "joints2d": [[55.0, 50.0] for _ in range(21)]}]},
            ]
        },
    )
    (run_root / "state/hands_metric").mkdir(parents=True, exist_ok=True)
    frame_idx = np.asarray([0, 1], dtype=np.int32)
    payload = {"frame_idx": frame_idx, "R_c2w": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0), "t_c2w": np.zeros((2, 3), dtype=np.float32)}
    for side in ("left", "right"):
        payload[f"{side}_joints_world_m"] = np.zeros((2, 21, 3), dtype=np.float32)
        payload[f"{side}_joints_world_m"][:, :, 2] = 1.0
        payload[f"{side}_vertices_world_m"] = np.zeros((2, 778, 3), dtype=np.float32)
        payload[f"{side}_valid"] = np.ones(2, dtype=np.uint8)
        payload[f"{side}_wilor_fit_reprojection_median_px"] = np.asarray([5.0, 10.0], dtype=np.float32)
    np.savez_compressed(run_root / "state/hands_metric/v22_hybrid_hands_metric.npz", **payload)
    write_json(run_root / "measurements/camera_trajectory/droid_full_frame/droid_dense_trajectory.json", {"frames": [{"frame_idx": 0, "pose_world_camera_xyzw": [0, 0, 0, 0, 0, 0, 1]}, {"frame_idx": 1, "pose_world_camera_xyzw": [0.1, 0, 0, 0, 0, 0, 1]}]})
    overlay = run_root / "renders/v22_overlay.mp4"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"not-a-real-video")
    write_json(run_root / "annotation_pipeline_manifest.json", {"case_id": "case", "run_root": str(run_root), "steps": [], "renders": {"v22_overlay": str(overlay)}})
    return run_root


def test_d8_estimates_bias_rows(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    result = estimate(parse_d8_args(["--run-root", str(run_root), "--smooth-radius", "1", "--accept-residual-px", "1"]))
    assert result["status"] == "ok"
    assert result["summary"]["rows_with_same_frame_support"] == 2
    assert result["summary"]["accepted_correction_rows"] >= 1
    assert Path(run_root / "state/gt_free_self_calibration/v22_gt_free_drift_self_calibration.json").exists()


def test_captioning_stage_uses_action_json(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    actions = tmp_path / "actions.json"
    write_json(actions, {"actions": [{"start_frame": 0, "end_frame": 2, "description": "open drawer"}]})
    result = build_captions(parse_caption_args(["--run-root", str(run_root), "--actions-json", str(actions)]))
    assert result["status"] == "ok"
    assert result["summary"]["semantic_clip_count"] == 1
    assert result["semantic_rows"][0]["caption"] == "open drawer"


def test_evaluator_stage_records_no_gt_and_prediction_diagnostics(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    result = run_eval(parse_eval_args(["--run-root", str(run_root)]))
    assert result["status"] == "no_gt_unmeasured"
    assert result["prediction_rows"]["head_camera"] == 2
    assert result["prediction_rows"]["hand_states"] == 4
    assert result["summary"]["metrics_measured"] == 0
    assert result["summary"]["prediction_diagnostics"] >= 1
    assert "hand_reprojection_error_px" not in result["metric_observations"]
    assert "hand_reprojection_error_px" in result["diagnostic_observations"]
    assert any(row["metric_id"] == "hand_reprojection_error_px" and row["status"] == "prediction_diagnostic" for row in result["validation_metrics"])


def test_self_consistency_qc_sees_stage_artifacts(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    estimate(parse_d8_args(["--run-root", str(run_root)]))
    build_captions(parse_caption_args(["--run-root", str(run_root)]))
    run_eval(parse_eval_args(["--run-root", str(run_root)]))
    result = run_qc(parse_qc_args(["--run-root", str(run_root)]))
    assert result["d8_gt_free_drift_self_calibration"]["status"] == "ok"
    assert result["d9b_captioning"]["status"] == "source_absent_no_caption_rows"
    assert result["d11_evaluator"]["status"] == "no_gt_unmeasured"
    assert any(row["check"] == "d8_stage_present" and row["ok"] is True for row in result["checks"])
