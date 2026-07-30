from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.adapt_v22_minimal_run_to_annotation_bundle import adapt, parse_args


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ndjson(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "v22_run"
    frames = [
        {
            "index": 0,
            "frame_idx": 0,
            "source_frame_idx": 100,
            "time_s": 0.0,
            "source_time_s": 10.0,
            "rgb": str(run_root / "input/raw_frame_manifest/rgb/000000.jpg"),
            "source_width": 1920,
            "source_height": 1080,
            "manifest_width": 960,
            "manifest_height": 540,
        },
        {
            "index": 1,
            "frame_idx": 1,
            "source_frame_idx": 101,
            "time_s": 1 / 30,
            "source_time_s": 10 + 1 / 30,
            "rgb": str(run_root / "input/raw_frame_manifest/rgb/000001.jpg"),
            "source_width": 1920,
            "source_height": 1080,
            "manifest_width": 960,
            "manifest_height": 540,
        },
    ]
    write_json(
        run_root / "input/raw_frame_manifest/manifest.json",
        {
            "schema": "v22_raw_frame_manifest.v0",
            "status": "ok",
            "fps": 30.0,
            "frame_count": 2,
            "video": {"fps": 30.0, "frame_count": 2, "duration_s": 2 / 30, "width": 1920, "height": 1080},
            "frames": frames,
        },
    )
    write_json(
        run_root / "input/input_manifest.json",
        {
            "schema": "v22_input_manifest.v0",
            "case_id": "case_a",
            "original_video": "/data/video.mp4",
            "primary_video": str(run_root / "input/clips/case_a.mp4"),
            "source_fingerprint": {"sha256": "abc123", "size_bytes": 1234},
        },
    )
    write_json(
        run_root / "state/calibration/v19_camera_calibration_contract.json",
        {
            "status": "ok",
            "method": "build_v19_calibration_contract",
            "intrinsics_fx_fy_cx_cy": [839.0, 839.0, 960.0, 540.0],
            "intrinsics_source": "v19_calibration_contract_unidepth_robust_video_constant",
            "diagnostics": {"selected_frame_count": 2, "selected_stats": {"focal_geom": {"relative_mad_fraction": 0.01}}},
        },
    )
    write_json(
        run_root / "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json",
        {
            "schema": "wilor_raw_hands.v0",
            "frames": [
                {
                    "frame_idx": 0,
                    "raw_hands": [
                        {
                            "side": "right",
                            "detector_score": 0.91,
                            "bbox_xyxy": [10, 20, 100, 120],
                            "joints2d": [[1.0, 2.0], [3.0, 4.0]],
                            "joints3d_camera": [[0.1, 0.2, 1.0], [0.2, 0.3, 1.1]],
                            "cam_t": [0.1, 0.2, 1.0],
                        }
                    ],
                },
                {"frame_idx": 1, "raw_hands": []},
            ],
        },
    )
    write_json(
        run_root / "state/annotations_v22_renderable.json",
        {
            "schema": "v22_minimal_annotation_state.v0",
            "status": "ok",
            "case_id": "case_a",
            "measurements": {
                "raw_frame_manifest": str(run_root / "input/raw_frame_manifest/manifest.json"),
                "calibration_contract": str(run_root / "state/calibration/v19_camera_calibration_contract.json"),
                "wilor_raw_hands": str(run_root / "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json"),
            },
        },
    )
    write_json(
        run_root / "annotation_pipeline_manifest.json",
        {
            "status": "ok",
            "case_id": "case_a",
            "run_root": str(run_root),
            "steps": [
                {"step": "wilor_hands", "status": "ok", "elapsed_s": 2.0, "log": str(run_root / "logs/wilor.log")},
                {"step": "render_hand_overlay", "status": "ok", "elapsed_s": 1.0, "log": str(run_root / "logs/render.log")},
            ],
            "renders": {
                "v22_overlay": str(run_root / "renders/v22_overlay.mp4"),
                "overlay_source": "wilor_raw_candidates",
            },
            "ffprobe_overlay": {"status": "ok", "ffprobe": {"streams": [{"width": 960, "height": 540, "duration": "0.066667", "nb_read_frames": "2"}]}},
        },
    )
    return run_root


def test_adapts_v22_minimal_run_without_overclaiming_metric_mano(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    args = parse_args(["--run-root", str(run_root), "--output-root", str(tmp_path / "bundles")])
    manifest_path = adapt(args)
    manifest = read_json(manifest_path)
    assert manifest["schema"] == "ego.annotation.output"
    assert manifest["status"] == "completed_with_errors"
    assert manifest["tables"]["frames"]["rows"] == 2
    assert manifest["tables"]["hand_states"]["rows"] == 2
    assert manifest["renders"]["v22_minimal_renders"]["v22_overlay"].endswith("v22_overlay.mp4")
    assert manifest["renders"]["render_metadata"]["overlay_source"] == "wilor_raw_candidates"
    assert "wilor_raw_candidates" not in manifest["renders"]["optional_qc_demo"]

    calibration = read_json(Path(manifest["calibration_contract"]))
    assert calibration["intrinsics_fx_fy_cx_cy"] == [839.0, 839.0, 960.0, 540.0]
    assert calibration["distortion"]["model"] == "unresolved"

    hands = ndjson(Path(manifest["tables"]["hand_states"]["ndjson"]))
    visible = next(row for row in hands if row["visibility"] == "visible")
    assert visible["source"] == "wilor_v21_raw_candidates"
    assert visible["accepted_metric_mano"] is False
    assert visible["state_role"] == "visible_geometry_candidate_not_metric_mano"
    absent = next(row for row in hands if row["visibility"] == "not_detected")
    assert absent["state_role"] == "presence_absence_evidence_not_metric_mano"

    errors = ndjson(Path(manifest["events"]["errors"]["ndjson"]))
    codes = {row["code"] for row in errors}
    assert "head_camera_unavailable" in codes
    assert "hawor_metric_mano_unavailable" in codes
    assert "hybrid_temporal_hand_fusion_unavailable" in codes
    assert "caption_source_unavailable" in codes
    assert "offline_evaluator_unavailable" in codes

    qc_path = Path(manifest["renders"]["state_artifacts"]["self_consistency_qc"])
    qc = read_json(qc_path)
    assert qc["frame_count_raw_manifest"] == 2
    assert qc["render_overlay"]["frame_count_matches_raw_manifest"] is True
    assert qc["wilor_candidate_frame_coverage"] == 0.5


def test_action_json_becomes_semantic_clips(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    actions = tmp_path / "actions.json"
    write_json(actions, {"tasks": [{"actions": [{"start_frame": 0, "end_frame": 90, "description": "pick up a cup"}]}]})
    args = parse_args([
        "--run-root",
        str(run_root),
        "--output-root",
        str(tmp_path / "bundles"),
        "--job-id",
        "with_captions",
        "--actions-json",
        str(actions),
        "--max-semantic-clip-s",
        "3.0",
    ])
    manifest_path = adapt(args)
    manifest = read_json(manifest_path)
    assert manifest["tables"]["semantic_clips"]["rows"] == 1
    semantic = ndjson(Path(manifest["tables"]["semantic_clips"]["ndjson"]))
    assert semantic[0]["caption"] == "pick up a cup"
    assert semantic[0]["duration_s"] == 3.0
    errors = ndjson(Path(manifest["events"]["errors"]["ndjson"]))
    assert "semantic_clips_unavailable" not in {row["code"] for row in errors}


def add_camera_and_hybrid_outputs(run_root: Path) -> None:
    cam_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    write_json(
        cam_dir / "v22_camera_trajectory_stage.json",
        {
            "status": "ok",
            "outputs": {"dense_json": str(cam_dir / "droid_dense_trajectory.json")},
            "calibration_contract": str(run_root / "state/calibration/v19_camera_calibration_contract.json"),
            "gauge_declaration": {"trajectory_frame": "DROID arbitrary world gauge", "scale_status": "video_derived_uncertain_without_external_metric_anchor"},
        },
    )
    write_json(
        cam_dir / "droid_dense_trajectory.json",
        {
            "frames": [
                {"frame_idx": 0, "pose_world_camera_xyzw": [0, 0, 0, 0, 0, 0, 1], "T_world_camera": np.eye(4).tolist()},
                {"frame_idx": 1, "pose_world_camera_xyzw": [0.1, 0, 0, 0, 0, 0, 1], "T_world_camera": np.eye(4).tolist()},
            ]
        },
    )
    hawor_dir = run_root / "measurements" / "hand_candidates" / "hawor_world"
    hawor_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(hawor_dir / "hawor_world_hands.npz", frame_idx=np.asarray([0, 1], dtype=np.int32))
    hands_dir = run_root / "state" / "hands_metric"
    hands_dir.mkdir(parents=True, exist_ok=True)
    hybrid_npz = hands_dir / "v22_hybrid_hands_metric.npz"
    frame_idx = np.asarray([0, 1], dtype=np.int32)
    payload = {"frame_idx": frame_idx}
    for side in ("left", "right"):
        payload[f"{side}_valid"] = np.ones(2, dtype=np.uint8)
        payload[f"{side}_detected_same_frame"] = np.ones(2, dtype=np.uint8)
        payload[f"{side}_hybrid_source"] = np.asarray(["wilor_root_relative_on_hawor_metric_wrist_trajectory", "hawor_infill"], dtype="<U64")
        payload[f"{side}_wilor_fit_reprojection_median_px"] = np.asarray([10.0, 100.0], dtype=np.float32)
        payload[f"{side}_wilor_fit_reprojection_p90_px"] = np.asarray([12.0, 120.0], dtype=np.float32)
        payload[f"{side}_joints_world_m"] = np.zeros((2, 21, 3), dtype=np.float32)
        payload[f"{side}_vertices_world_m"] = np.zeros((2, 778, 3), dtype=np.float32)
        payload[f"{side}_root_orient_axis_angle"] = np.zeros((2, 3), dtype=np.float32)
        payload[f"{side}_hand_pose_axis_angle"] = np.zeros((2, 45), dtype=np.float32)
        payload[f"{side}_betas"] = np.zeros((2, 10), dtype=np.float32)
    np.savez_compressed(hybrid_npz, **payload)
    write_json(
        hands_dir / "v22_hybrid_hand_fusion_stage.json",
        {"status": "ok", "outputs": {"hybrid_npz": str(hybrid_npz), "report_json": str(hands_dir / "report.json")}},
    )


def test_adapter_promotes_camera_and_hybrid_rows_when_stages_exist(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    add_camera_and_hybrid_outputs(run_root)
    args = parse_args([
        "--run-root",
        str(run_root),
        "--output-root",
        str(tmp_path / "bundles"),
        "--job-id",
        "with_camera_hybrid",
    ])
    manifest_path = adapt(args)
    manifest = read_json(manifest_path)
    assert manifest["tables"]["head_camera"]["rows"] == 2
    assert manifest["tables"]["hand_states"]["rows"] == 4
    errors = ndjson(Path(manifest["events"]["errors"]["ndjson"]))
    codes = {row["code"] for row in errors}
    assert "head_camera_unavailable" not in codes
    assert "hawor_metric_mano_unavailable" not in codes
    assert "hybrid_temporal_hand_fusion_unavailable" not in codes
    assert "head_camera_video_derived_uncertain_gauge" in codes
    assert "hybrid_temporal_hand_fusion_quality_degraded" in codes
    hands = ndjson(Path(manifest["tables"]["hand_states"]["ndjson"]))
    assert {row["state_role"] for row in hands} == {"hybrid_metric_hand_candidate"}
    assert any(row["quality_status"] == "degraded_large_wilor_reprojection_residual" for row in hands)
    qc = read_json(Path(manifest["renders"]["state_artifacts"]["self_consistency_qc"]))
    assert qc["head_camera_rows"] == 2
    assert qc["hand_state_source"] == "hybrid_metric_hand_candidate"


def test_adapter_consumes_d8_d9b_d10_d11_stage_artifacts(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    add_camera_and_hybrid_outputs(run_root)
    d8 = run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json"
    d9b = run_root / "state" / "semantic_clips" / "v22_captioning_stage.json"
    d10 = run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json"
    d11 = run_root / "evaluation" / "v22_evaluator_stage.json"
    write_json(d8, {"schema": "v22_gt_free_drift_self_calibration.v0", "status": "ok", "summary": {"accepted_correction_rows": 1}})
    write_json(
        d9b,
        {
            "schema": "v22_captioning_stage.v0",
            "status": "ok",
            "semantic_rows": [{"clip_id": "c0", "start_s": 0.0, "end_s": 0.5, "duration_s": 0.5, "caption": "pick object", "source": "test"}],
            "caption_events": [{"event": "semantic_clip_from_caption_source", "clip_id": "c0"}],
            "summary": {"semantic_clip_count": 1},
        },
    )
    write_json(d10, {"schema": "v22_full_self_consistency_qc.v0", "status": "ok", "checks": [], "summary": {}})
    write_json(d11, {"schema": "v22_evaluator_stage.v0", "status": "no_gt_unmeasured", "metric_observations": {"hand_reprojection_error_px": [2.0]}, "summary": {"metrics_measured": 0, "prediction_diagnostics": 1}})
    manifest = read_json(run_root / "annotation_pipeline_manifest.json")
    manifest["stage_artifacts"] = {
        "gt_free_drift_self_calibration": str(d8),
        "captioning": str(d9b),
        "self_consistency_qc": str(d10),
        "evaluator": str(d11),
    }
    write_json(run_root / "annotation_pipeline_manifest.json", manifest)
    args = parse_args(["--run-root", str(run_root), "--output-root", str(tmp_path / "bundles"), "--job-id", "with_all_stage_artifacts"])
    manifest_path = adapt(args)
    product = read_json(manifest_path)
    assert product["tables"]["semantic_clips"]["rows"] == 1
    errors = ndjson(Path(product["events"]["errors"]["ndjson"]))
    codes = {row["code"] for row in errors}
    assert "gt_free_hand_self_calibration_unavailable" not in codes
    assert "caption_source_unavailable" not in codes
    assert "self_consistency_qc_unavailable" not in codes
    assert "offline_evaluator_unavailable" not in codes
    assert "offline_evaluator_gt_unavailable" in codes
    state_artifacts = product["renders"]["state_artifacts"]
    assert Path(state_artifacts["gt_free_drift_self_calibration"]).exists()
    assert Path(state_artifacts["captioning_stage"]).exists()
    assert Path(state_artifacts["full_self_consistency_qc"]).exists()
    assert Path(state_artifacts["evaluator_stage"]).exists()
    metrics = ndjson(Path(product["tables"]["validation_metrics"]["ndjson"]))
    assert any(row["metric_id"] == "hand_reprojection_error_px" and row["status"] == "prediction_diagnostic" for row in metrics)
    assert not any(row["metric_id"] == "hand_reprojection_error_px" and row["status"] == "measured" for row in metrics)
